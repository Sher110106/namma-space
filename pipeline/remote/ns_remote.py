#!/usr/bin/env python3
"""Stage runner for tyrone (system python3, standard library only).

    python3 pipeline/remote/ns_remote.py <run-name> [options]

Runs the stages of one run in order, each as
    nsdocker [--gpu] ns-<run>-<stage> <image> python3 -m pipeline.<stage> --run /w/runs/<run> [opts]
and skips a stage when its done-marker JSON already exists (CONTRACTS 2.11). Nothing
else tracks state, so a killed or dropped run resumes by running the same command again.

Options
    --from STAGE      force this stage and every later one to re-run (earlier ones still skip when done)
    --to STAGE        stop after this stage
    --only A,B        run just these stages (still skips done ones unless forced by --from)
    --quick           7k-step preview training: passes --quick to train and sets NS_QUICK=1 for every stage
    --opt STAGE:ARG   extra argument for one stage (repeatable), e.g. --opt train:--steps=3000
    --script STAGE=P  run the script P (repo-relative, called with the stage's arguments) instead of the module,
                      e.g. --script align=test/e2e/adapters/align_thin.py while W2's align is not ready
    --image STAGE=IMG use another docker image for one stage (e.g. export=namma-space-gpu:0.1)
    --rescale         run pipeline.align.rescale (args via --opt rescale:...), then force the grid stage;
                      nothing before the grid re-runs and nothing retrains
    --dry-run         print the plan and exit without running stages
    --status          print status.json and exit
    --allow-missing   a missing stage module is a warning, not exit 20
    --with-optional   also run optional P1 stages (geometry); by default they are skipped
    --no-docker       run stage modules directly with this python (tests, Mac)
    --stub-dir DIR    run DIR/<stage>.py instead of pipeline.<stage> (test stubs; relative to the repo)
    --fake-free-gb N  pretend only N GB of disk is free (test 0.3); stages then exit 20
    --min-free-gb N   disk floor in GB (default 15; tests on a full Mac lower it)
    --git SHA         recorded in run.json (the repo on tyrone has no .git)
    --runs-root DIR, --repo DIR, --ns-root DIR, --nsdocker PATH
    --stage-timeout STAGE=MIN  wall limit once the stage's container runs (STAGE_LIMIT_MIN); past it the
                      stage is stopped and recorded as exit 20. --queue-limit-min caps the wait for a slot.

Exit codes (CONTRACTS 2.11): 0 good, 10 capture problem, 11 partial reconstruction shipped,
20 infrastructure problem, 30 a quality gate failed (bundle exists, needs review).
Several stage results combine as 20 > 10 > 30 > 11 > 0. A required stage whose module does not
exist yet is skipped with a message and makes the final exit 20 ("incomplete") unless --allow-missing.

Files written in runs/<name>/: logs/<stage>.log, logs/runner.log (by nohup), status.json (live; final state is
done, partial (11), review (30), failed (10, 20) or stopped (signal)),
timings.json (seconds per stage plus total), run.json (git, image digests, flags, warnings,
manual_overrides, exit). Standard library only; no import from other packages' code.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_DEFAULT = HERE.parents[2]
sys.path.insert(0, str(REPO_DEFAULT))

from pipeline.common import stage as st  # noqa: E402

NS_ROOT_DEFAULT = os.environ.get("NS_ROOT", "/home/sher/namma-space")
NSDOCKER_DEFAULT = os.environ.get("NSDOCKER", f"{NS_ROOT_DEFAULT}/bin/nsdocker")
RUN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,40}$")
RESCALE = st.StageInfo("rescale", "pipeline.align.rescale", (), "", st.GPU_IMAGE, False)
EXIT_PRIORITY = {0: 0, 11: 1, 30: 2, 10: 3, 20: 4}
# Wall limit per stage in minutes, counted from the moment its container runs (queueing for a GPU slot is
# not counted; that has its own limit). A stage past its limit is stopped and recorded as exit 20, so a
# hung tool can never hold the run (or the GPU) for hours. Override: --stage-timeout STAGE=MIN.
STAGE_LIMIT_MIN = {"ingest": 45, "sfm": 150, "align": 30, "train": 330, "geometry": 60, "export": 60,
                   "grid": 20, "rescale": 15}
QUEUE_LIMIT_MIN = 240
BEAT_S = float(os.environ.get("NS_RUNNER_BEAT_S", "10"))   # status heartbeat and limit check interval

_stop = threading.Event()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def say(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} ns_remote: {msg}", flush=True)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.a = args
        self.repo = Path(args.repo).resolve()
        self.runs_root = Path(args.runs_root)
        self.run = args.run
        self.dir = self.runs_root / self.run
        self.status: dict = {"run": self.run, "state": "starting", "pid": os.getpid(),
                             "host": socket.gethostname(), "started": now(), "stages": {}}
        self.warnings: list = []
        self.final = 0
        self.incomplete = False
        self.proc: subprocess.Popen | None = None
        self.current_container = ""
        self.image_over = dict(x.split("=", 1) for x in args.image)
        self.scripts = dict(x.split("=", 1) for x in args.script)
        self.container_repo = self.repo_in_container()
        self.nsdocker = args.nsdocker
        if self.container_repo != "/w/repo" and not getattr(args, "nsdocker_given", False):
            # a frozen copy (ns run --repo-dir): its own wrapper honours NS_WORKDIR; an older bin/nsdocker may not
            own = self.repo / "docker" / "nsdocker"
            if own.exists():
                self.nsdocker = str(own)
        self.stage_limits = dict(STAGE_LIMIT_MIN)
        for x in args.stage_timeout:
            k, v = x.split("=", 1)
            self.stage_limits[k] = float(v)
        self.timed_out = ""

    def image_for(self, info: st.StageInfo) -> str:
        return self.image_over.get(info.name, info.image)

    # ---------------------------------------------------------------- plan

    def selected(self) -> list:
        names = list(st.STAGE_NAMES)
        if self.a.only:
            want = [x.strip() for x in self.a.only.split(",") if x.strip()]
            for w in want:
                if w not in st.STAGE_BY_NAME:
                    raise SystemExit(f"unknown stage {w!r}; stages are {', '.join(names)}")
            names = [n for n in names if n in want]
        lo = names.index(self.a.from_stage) if self.a.from_stage in names else 0
        hi = names.index(self.a.to) if self.a.to in names else len(names) - 1
        if self.a.from_stage and self.a.from_stage not in st.STAGE_BY_NAME:
            raise SystemExit(f"unknown stage {self.a.from_stage!r}")
        if self.a.to and self.a.to not in st.STAGE_BY_NAME:
            raise SystemExit(f"unknown stage {self.a.to!r}")
        return [st.STAGE_BY_NAME[n] for n in names[lo:hi + 1]]

    def forced(self, info: st.StageInfo) -> bool:
        if not self.a.from_stage:
            return False
        order = list(st.STAGE_NAMES)
        return order.index(info.name) >= order.index(self.a.from_stage)

    def script_for(self, info: st.StageInfo) -> str | None:
        return self.scripts.get(info.name)

    def module_path(self, info: st.StageInfo) -> Path:
        sc = self.script_for(info)
        if sc:
            return Path(sc) if Path(sc).is_absolute() else self.repo / sc
        if self.a.stub_dir:
            d = Path(self.a.stub_dir)
            d = d if d.is_absolute() else self.repo / d
            return d / f"{info.name}.py"
        parts = info.module.split(".")
        pkg = self.repo.joinpath(*parts)
        return pkg / "__main__.py" if pkg.is_dir() else pkg.with_suffix(".py")

    def module_exists(self, info: st.StageInfo) -> bool:
        return self.module_path(info).exists()

    # ---------------------------------------------------------------- commands

    def repo_in_container(self) -> str:
        """Where the code is inside containers: /w/<repo relative to the namma-space root>. A frozen copy made by
        `ns run --repo-dir repo-room` is /w/repo-room, so later edits to repo/ cannot change a running build."""
        rel = os.path.relpath(self.repo, Path(self.a.ns_root).resolve())
        if rel.startswith("..") or rel == ".":
            return "/w/repo"
        return "/w/" + rel

    def container_run_dir(self) -> str:
        rel = os.path.relpath(self.dir, self.a.ns_root)
        if rel.startswith(".."):
            raise SystemExit(f"{self.dir} is not under {self.a.ns_root}; use --no-docker for other locations")
        return "/w/" + rel

    def stage_argv(self, info: st.StageInfo, run_path: str) -> list:
        extra = [x.split(":", 1)[1] for x in self.a.opt if x.split(":", 1)[0] == info.name]
        quick = ["--quick"] if (self.a.quick and info.name == "train") else []
        return ["--run", run_path] + quick + extra

    def build_cmd(self, info: st.StageInfo) -> tuple:
        env = {"NS_RUNNER": "1"}
        if self.a.quick:
            env["NS_QUICK"] = "1"
        if self.a.fake_free_gb is not None:
            env[st.FAKE_FREE_ENV] = str(self.a.fake_free_gb)
        container = f"ns-{self.run}-{info.name}"
        sc = self.script_for(info)
        if sc:
            rel = os.path.relpath(self.module_path(info), self.repo)
            if self.a.no_docker:
                prog = [sys.executable, str(self.module_path(info))]
            else:
                prog = ["python3", f"{self.container_repo}/{rel}"]
            if self.a.no_docker:
                return prog + self.stage_argv(info, str(self.dir)), env, container
            pre = [self.nsdocker] + (["--gpu"] if info.gpu else []) + [container, self.image_for(info)]
            envargs = ["env"] + [f"{k}={v}" for k, v in env.items()]
            return pre + envargs + prog + self.stage_argv(info, self.container_run_dir()), env, container
        if self.a.no_docker:
            if self.a.stub_dir:
                prog = [sys.executable, str(self.module_path(info))]
            else:
                prog = [sys.executable, "-m", info.module]
            argv = prog + self.stage_argv(info, str(self.dir))
            return argv, env, container
        if self.a.stub_dir:
            d = Path(self.a.stub_dir)
            rel = d if not d.is_absolute() else d.relative_to(self.repo)
            prog = ["python3", f"{self.container_repo}/{rel}/{info.name}.py"]
        else:
            prog = ["python3", "-m", info.module]
        pre = [self.nsdocker] + (["--gpu"] if info.gpu else []) + [container, self.image_for(info)]
        envargs = ["env"] + [f"{k}={v}" for k, v in env.items()]
        return pre + envargs + prog + self.stage_argv(info, self.container_run_dir()), env, container

    # ---------------------------------------------------------------- status files

    def write_status(self) -> None:
        self.status["updated"] = now()
        st.write_json_atomic(self.dir / "status.json", self.status)

    def set_stage(self, name: str, **kw) -> None:
        s = self.status["stages"].setdefault(name, {})
        s.update(kw)
        self.write_status()

    def docker(self, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)

    def image_digests(self, images: list) -> dict:
        out: dict = {}
        if self.a.no_docker:
            return out
        for img in sorted(set(images)):
            try:
                r = self.docker("image", "inspect", "--format", "{{.Id}}", img)
                out[img] = r.stdout.strip() if r.returncode == 0 else f"missing ({r.stderr.strip()[:80]})"
            except (OSError, subprocess.SubprocessError) as exc:
                out[img] = f"unknown ({exc})"
        return out

    def write_run_json(self, exit_code: int | None, flags: dict, images: dict) -> None:
        p = self.dir / "run.json"
        old = st.read_json(p, {})
        history = old.get("history", [])
        meta = st.read_json(self.dir / "input" / "meta.json", {})
        overrides = list(old.get("manual_overrides", []))
        for o in self.a.manual_override + list(meta.get("manual_overrides", []) or []):
            if o not in overrides:
                overrides.append(o)
        warns = list(old.get("warnings", []))
        for w in self.warnings:
            if w not in warns:
                warns.append(w)
        rec = {"git": self.a.git or old.get("git", ""), "images": {**old.get("images", {}), **images},
               "flags": flags, "warnings": warns, "manual_overrides": overrides, "exit": exit_code,
               "incomplete": self.incomplete, "host": socket.gethostname(),
               "started": self.status["started"], "ended": now() if exit_code is not None else None,
               "stages": self.status["stages"], "argv": sys.argv[1:],
               "repo": str(self.repo), "repo_in_container": self.container_repo}
        if exit_code is not None:
            history = (history + [{"started": rec["started"], "ended": rec["ended"], "exit": exit_code,
                                   "flags": flags}])[-30:]
        rec["history"] = history
        st.write_json_atomic(p, rec)

    # ---------------------------------------------------------------- one stage

    def stash(self, info: st.StageInfo) -> list:
        """Move the stage's existing outputs aside so a forced re-run starts clean."""
        stashed = []
        for out in info.outputs:
            d = self.dir / out
            if d.exists() or d.is_symlink():
                prev = self.dir / f"{out}.prev"
                if prev.exists():
                    shutil.rmtree(prev, ignore_errors=True)
                os.replace(d, prev)
                stashed.append((d, prev))
            for junk in (self.dir / f"{out}.tmp", self.dir / f"{out}.failed"):
                if junk.exists():
                    shutil.rmtree(junk, ignore_errors=True)
        return stashed

    def unstash(self, stashed: list) -> None:
        """Drop the old outputs of a forced stage. They are never restored: the stage is done only when its
        JSON exists, so stale results from an older input must not come back after a failed re-run."""
        for _d, prev in stashed:
            shutil.rmtree(prev, ignore_errors=True)

    def stop_container(self, name: str) -> None:
        if self.a.no_docker or not name.startswith("ns-"):
            return
        try:
            self.docker("stop", "-t", "10", name, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass

    def run_cmd(self, info: st.StageInfo, label: str) -> int:
        """Run the stage command, streaming to logs/<label>.log. Returns the process exit code."""
        cmd, env, container = self.build_cmd(info)
        log_path = self.dir / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.a.no_docker:
            # a container left behind by a killed runner would block the name
            r = self.docker("ps", "--filter", f"name=^{container}$", "--format", "{{.Names}}")
            if container in r.stdout.split():
                say(f"stopping leftover container {container}")
                self.stop_container(container)
        proc_env = dict(os.environ, **env)
        if not self.a.no_docker:
            # read by nsdocker (docker/nsdocker): the mount root and the code directory inside the container
            proc_env["NS_ROOT"] = str(Path(self.a.ns_root).resolve())
            proc_env["NS_WORKDIR"] = self.container_repo
        proc_env["PYTHONPATH"] = f"{self.repo}{os.pathsep}{proc_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        with log_path.open("a") as fh:
            fh.write(f"=== {now()} start {label}: {' '.join(cmd)}\n")
            fh.flush()
            self.current_container = container
            self.proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         cwd=str(self.repo), env=proc_env, start_new_session=True)
            t0 = time.monotonic()
            last_beat = 0.0
            self.set_stage(label, child_pid=self.proc.pid, container=container if not self.a.no_docker else "")
            limit_s = 60 * float(self.stage_limits.get(label, 120))
            queue_s = 60 * float(self.a.queue_limit_min)
            t_run = t0 if self.a.no_docker else None   # when the container was first seen running
            self.timed_out = ""
            while True:
                try:
                    rc = self.proc.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if _stop.is_set():
                    say(f"stop requested; stopping {label}")
                    self.kill_current()
                    rc = self.proc.wait()
                    fh.write(f"=== {now()} interrupted\n")
                    self.proc = None
                    return -signal.SIGTERM
                if time.monotonic() - last_beat > BEAT_S:
                    last_beat = time.monotonic()
                    if t_run is None and self.container_running(container):
                        t_run = time.monotonic()
                        self.set_stage(label, container_started=now(), queued_s=round(t_run - t0, 1))
                    self.set_stage(label, elapsed_s=round(time.monotonic() - t0, 1), last_line=self.tail(log_path))
                    over = ""
                    if t_run is not None and time.monotonic() - t_run > limit_s:
                        over = f"ran {limit_s / 60:.0f} min (stage limit); stopped"
                    elif t_run is None and time.monotonic() - t0 > queue_s:
                        over = f"waited {queue_s / 60:.0f} min for a container or GPU slot; gave up"
                    if over:
                        say(f"{label}: {over}")
                        fh.write(f"=== {now()} {label}: {over}\n")
                        fh.flush()
                        self.timed_out = over
                        self.kill_current()
                        try:
                            rc = self.proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            rc = -signal.SIGKILL
                        break
            fh.write(f"=== {now()} end {label}: exit {rc} after {time.monotonic() - t0:.1f}s\n")
        self.proc = None
        self.current_container = ""
        return rc

    def container_running(self, name: str) -> bool:
        try:
            r = self.docker("ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}", timeout=20)
        except (OSError, subprocess.SubprocessError):
            return False
        return name in r.stdout.split()

    def kill_current(self) -> None:
        if self.current_container:
            self.stop_container(self.current_container)
            try:   # docker stop gives up after its timeout on a process stuck in the driver; kill it
                if self.container_running(self.current_container):
                    self.docker("kill", self.current_container, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @staticmethod
    def tail(path: Path, n: int = 1) -> str:
        try:
            with path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 4096))
                lines = fh.read().decode("utf-8", "replace").strip().splitlines()
            return " | ".join(l for l in lines[-n:] if not l.startswith("==="))[:300]
        except OSError:
            return ""

    @staticmethod
    def ns_result(path: Path) -> dict:
        """The last `NS_RESULT {json}` line a stage printed (ingest, align, ... use it), or {}."""
        try:
            with path.open("rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 65536))
                text = fh.read().decode("utf-8", "replace")
        except OSError:
            return {}
        for line in reversed(text.splitlines()):
            if line.startswith("NS_RESULT "):
                try:
                    obj = json.loads(line[len("NS_RESULT "):])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    return {}
        return {}

    # ---------------------------------------------------------------- main loop

    def need_missing(self, info: st.StageInfo) -> str:
        for rel in info.requires:
            if not (self.dir / rel).exists():
                return rel
        return ""

    def run_stage(self, info: st.StageInfo) -> bool:
        """Returns False when the whole run must stop."""
        name = info.name
        forced = self.forced(info)
        if not forced and self.dir.joinpath(info.json).exists():
            say(f"{name}: done already ({info.json}), skipping")
            self.set_stage(name, state="skipped-done")
            return True
        wanted = self.a.with_optional or (self.a.only and name in [x.strip() for x in self.a.only.split(",")])
        if info.optional and not wanted:
            # P1 stages (geometry) feed nothing later in the chain; they cost a GPU slot and minutes, so they run only
            # when asked: --with-optional (ns run --geometry) or --only <stage>
            say(f"{name}: optional, not requested; skipped")
            self.set_stage(name, state="skipped-optional")
            return True
        if not self.module_exists(info):
            msg = f"{name}: module {info.module} not found ({self.module_path(info)}); skipped"
            say("SKIP " + msg)
            self.warnings.append(msg)
            self.set_stage(name, state="skipped-missing", note=msg)
            if not info.optional and not self.a.allow_missing:
                self.incomplete = True
            return True
        miss = self.need_missing(info)
        if miss:
            msg = f"{name}: needs {miss}, which does not exist; skipped"
            say("SKIP " + msg)
            self.warnings.append(msg)
            self.set_stage(name, state="skipped-needs", note=msg)
            if not info.optional and not self.a.allow_missing:
                self.incomplete = True
            return True
        free = st.free_gb(self.runs_root)
        if free < self.a.min_free_gb:
            say(f"{name}: only {free:.1f} GB free (floor {self.a.min_free_gb:g} GB); exit 20")
            self.set_stage(name, state="failed", exit=20, note=f"disk guard: {free:.1f} GB free")
            self.final = max(self.final, 20, key=EXIT_PRIORITY.get)
            return False
        stashed = self.stash(info) if forced else []
        say(f"{name}: starting ({self.image_for(info)}{', gpu' if info.gpu else ''})")
        self.set_stage(name, state="running", started=now(), image=self.image_for(info))
        t0 = time.monotonic()
        rc = self.run_cmd(info, name)
        secs = time.monotonic() - t0
        st.add_timing(self.dir, name, secs)
        code = rc if rc in EXIT_PRIORITY else 20
        marker = (self.dir / info.json).exists()
        note = ""
        res = self.ns_result(self.dir / "logs" / f"{name}.log")
        if code == 0 and marker:   # a stage that shipped a partial result (sfm: under 95% in one model) -> 11
            sj = st.read_json(self.dir / info.json, {}) or {}
            if isinstance(sj, dict) and sj.get("partial") is True:
                code = 11
                note = "partial: " + str(sj.get("why") or "the stage flagged its result as partial")[:200]
                self.warnings.append(f"{name}: {note}")
        if code == 0 and res.get("gate_failed"):  # the stage finished but says its quality gate failed
            code = 30
            flags = res.get("flags") or []
            note = "quality gate failed" + (f": {'; '.join(str(f) for f in flags)[:200]}" if flags else "")
            self.warnings.append(f"{name}: {note}")
        if self.timed_out:
            code, note = 20, self.timed_out
        elif rc not in EXIT_PRIORITY:
            note = f"stage exited {rc} (not a contract code); treated as 20"
        elif rc in (0, 11, 30) and not marker:
            code, note = 20, f"exit {rc} but {info.json} was not written"
        elif rc == 20:
            note = "infrastructure problem"
        if code in (10, 20) and marker:  # a failed stage must not look done
            failed = self.dir / (info.json + ".failed")
            os.replace(self.dir / info.json, failed)
            self.warnings.append(f"{name}: exit {code}; its {Path(info.json).name} was renamed to {failed.name}")
        self.unstash(stashed)
        if info.optional and code != 0:
            # an optional stage (geometry, P1) never decides the run: record it, warn, go on
            msg = f"{name} (optional) exit {code}" + (f": {note}" if note else "") + "; ignored, the run continues"
            say(msg)
            self.warnings.append(msg)
            self.set_stage(name, state=f"exit-{code}-ignored", exit=code, seconds=round(secs, 1), ended=now(),
                           note=(note + "; " if note else "") + "optional stage, ignored",
                           last_line=self.tail(self.dir / "logs" / f"{name}.log", 2))
            return True
        self.set_stage(name, state="done" if code == 0 else f"exit-{code}", exit=code, seconds=round(secs, 1),
                       ended=now(), note=note, last_line=self.tail(self.dir / "logs" / f"{name}.log", 2))
        say(f"{name}: exit {code} in {secs:.1f}s" + (f" ({note})" if note else ""))
        if code != 0:
            self.final = max(self.final, code, key=EXIT_PRIORITY.get)
            if note:
                self.warnings.append(f"{name}: {note}")
            if code == 10:
                say(f"{name}: capture problem. Last lines of logs/{name}.log:")
                self.show_tail(name)
            if code in (10, 20):
                if code == 20:
                    self.show_tail(name)
                return False
        return True

    def show_tail(self, name: str, n: int = 12) -> None:
        p = self.dir / "logs" / f"{name}.log"
        try:
            for line in p.read_text(errors="replace").splitlines()[-n:]:
                print("    " + line, flush=True)
        except OSError:
            pass

    def do_rescale(self) -> bool:
        miss = [r for r in ("40_align/align.json", "60_export/export.json") if not (self.dir / r).exists()]
        if miss:
            say(f"rescale: needs {', '.join(miss)}; nothing to rescale")
            self.final = max(self.final, 20, key=EXIT_PRIORITY.get)
            return False
        if not self.module_exists(RESCALE):
            say(f"rescale: module {RESCALE.module} not found ({self.module_path(RESCALE)})")
            self.warnings.append("rescale: module missing")
            self.final = max(self.final, 20, key=EXIT_PRIORITY.get)
            return False
        self.set_stage("rescale", state="running", started=now())
        t0 = time.monotonic()
        rc = self.run_cmd(RESCALE, "rescale")
        secs = time.monotonic() - t0
        st.add_timing(self.dir, "rescale", secs)
        code = rc if rc in EXIT_PRIORITY else 20
        self.set_stage("rescale", state="done" if code == 0 else f"exit-{code}", exit=code, seconds=round(secs, 1),
                       ended=now())
        say(f"rescale: exit {code} in {secs:.1f}s")
        if code != 0:
            self.final = max(self.final, code, key=EXIT_PRIORITY.get)
            self.show_tail("rescale")
            return False
        return True

    def execute(self) -> int:
        a = self.a
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        stages = self.selected()
        if a.rescale:
            stages = [st.STAGE_BY_NAME["grid"]]
            a.from_stage = "grid"
        flags = {"from": a.from_stage, "to": a.to, "only": a.only, "quick": a.quick, "opt": a.opt,
                 "with_optional": a.with_optional,
                 "rescale": a.rescale, "no_docker": a.no_docker, "stub_dir": a.stub_dir,
                 "fake_free_gb": a.fake_free_gb, "allow_missing": a.allow_missing}
        plan = []
        for s in stages:
            if self.forced(s):
                plan.append(f"{s.name}: re-run (forced)")
            elif self.dir.joinpath(s.json).exists():
                plan.append(f"{s.name}: done, skip")
            else:
                plan.append(f"{s.name}: run")
        say("plan: " + "; ".join(plan))
        if a.dry_run:
            return 0
        images = self.image_digests([self.image_for(s) for s in stages] + ([RESCALE.image] if a.rescale else []))
        self.status["state"] = "running"
        self.status["plan"] = plan
        self.write_status()
        self.write_run_json(None, flags, images)
        ok = True
        if a.rescale:
            ok = self.do_rescale()
        if ok:
            for s in stages:
                if _stop.is_set():
                    self.final = max(self.final, 20, key=EXIT_PRIORITY.get)
                    self.warnings.append("stopped by signal")
                    break
                if not self.run_stage(s):
                    break
        if _stop.is_set():
            self.final = max(self.final, 20, key=EXIT_PRIORITY.get)
        if self.incomplete and self.final == 0:
            self.final = 20
            self.warnings.append("incomplete: a required stage module is missing (exit 20)")
            say("INCOMPLETE: at least one required stage module is missing, see warnings; exit 20")
        by_code = {0: "done", 11: "partial", 30: "review"}  # 30: a bundle exists but needs a human look
        self.status["state"] = "stopped" if _stop.is_set() else by_code.get(self.final, "failed")
        self.status["exit"] = self.final
        self.status["incomplete"] = self.incomplete
        self.status["ended"] = now()
        self.write_status()
        self.write_run_json(self.final, flags, images)
        say(f"finished with exit {self.final} ({st.EXIT_NAMES.get(self.final, '?')})")
        return self.final


def parse_args(argv: list) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Namma Space stage runner (tyrone)", add_help=True)
    ap.add_argument("run", help="run name; the run directory is <runs-root>/<run>")
    ap.add_argument("--from", dest="from_stage")
    ap.add_argument("--to")
    ap.add_argument("--only")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--image", action="append", default=[], help="STAGE=IMAGE overrides a stage's docker image")
    ap.add_argument("--script", action="append", default=[], help="STAGE=PATH runs a script (repo-relative) instead of the stage module")
    ap.add_argument("--rescale", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--with-optional", action="store_true", help="also run optional P1 stages (geometry)")
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--stub-dir")
    ap.add_argument("--fake-free-gb", type=float)
    ap.add_argument("--min-free-gb", type=float, default=st.MIN_FREE_GB, help="disk floor in GB (default 15)")
    ap.add_argument("--git", default="")
    ap.add_argument("--manual-override", action="append", default=[])
    ap.add_argument("--ns-root", default=NS_ROOT_DEFAULT)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--nsdocker", default=None, help="host wrapper (default <ns-root>/bin/nsdocker)")
    ap.add_argument("--stage-timeout", action="append", default=[],
                    help="STAGE=MINUTES wall limit once the stage's container runs (defaults: "
                         + ", ".join(f"{k} {v}" for k, v in STAGE_LIMIT_MIN.items()) + ")")
    ap.add_argument("--queue-limit-min", type=float, default=QUEUE_LIMIT_MIN,
                    help="give up on a stage that waits this long for a container or GPU slot")
    a = ap.parse_args(argv)
    if not RUN_NAME_RE.match(a.run):
        ap.error(f"run name {a.run!r} must match {RUN_NAME_RE.pattern}")
    a.runs_root = a.runs_root or os.path.join(a.ns_root, "runs")
    a.nsdocker_given = a.nsdocker is not None or "NSDOCKER" in os.environ
    a.nsdocker = a.nsdocker or os.environ.get("NSDOCKER") or os.path.join(a.ns_root, "bin", "nsdocker")
    for o in a.opt:
        if ":" not in o:
            ap.error(f"--opt needs STAGE:ARG, got {o!r}")
    for o in a.image + a.script + a.stage_timeout:
        if "=" not in o:
            ap.error(f"--image and --script need STAGE=VALUE, got {o!r}")
    return a


def main(argv: list | None = None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    run_dir = Path(a.runs_root) / a.run
    if a.fake_free_gb is not None:
        os.environ[st.FAKE_FREE_ENV] = str(a.fake_free_gb)  # the runner's own guard and its children see the fake
    if a.status:
        p = run_dir / "status.json"
        print(p.read_text() if p.exists() else json.dumps({"run": a.run, "state": "none"}))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _stop.set())
    lock = open(run_dir / ".runner.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        say(f"another runner already holds {run_dir}/.runner.lock; exit 20")
        return st.EXIT_INFRA
    lock.write(str(os.getpid()))
    lock.flush()
    r = Runner(a)
    if st.free_gb(a.runs_root) < a.min_free_gb:
        say(f"disk guard: {st.free_gb(a.runs_root):.1f} GB free is below {a.min_free_gb:g} GB; exit 20")
        r.status.update(state="failed", exit=20, note="disk guard at start")
        r.write_status()
        r.final = 20
        r.write_run_json(20, {"fake_free_gb": a.fake_free_gb}, {})
        return 20
    try:
        return r.execute()
    except SystemExit:
        raise
    except Exception as exc:  # the runner itself broke: say so, stop the child, exit 20
        import traceback
        traceback.print_exc()
        r.kill_current()
        r.status.update(state="failed", exit=20, note=f"runner crashed: {exc}")
        r.write_status()
        return 20


if __name__ == "__main__":
    sys.exit(main())
