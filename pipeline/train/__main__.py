"""Stage `train`: gsplat MCMC training, variants R0 and R1 (FINAL_PLAN 2.5, WP3).

    python -m pipeline.train --run /w/runs/<name> [--quick] [--variants r0,r1] [--parallel auto|yes|no]

Reads 40_align/ (images/, sparse/0/ in the final frame), writes 50_train/ through a temp
directory (50_train.tmp, renamed when complete). While running, 50_train.tmp/progress.json
lists each variant's step and, once it exists, the path of its 7k preview PLY.
Exit codes: 0 good, 20 infrastructure problem, 30 quality gate failed (train.json still written).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

from . import cmd as C
from . import gpu as G
from . import stats as S

STAGE = "50_train"
MIN_FREE_GB = 15
PSNR_GATE_DB = 20.0
TIME_GATE_S = 45 * 60
POLL_S = 2.0
PROGRESS_S = 8.0
KILL_GRACE_S = 30.0      # SIGTERM -> SIGKILL
ABANDON_S = 60.0         # SIGKILL -> give up waiting (process stuck in the driver)
STALL_S = 900.0          # no trainer output for this long = hung (default; --stall-minutes)


def log(msg: str) -> None:
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def check_data_dir(d: str) -> List[str]:
    """Return a list of problems (empty = fine)."""
    probs = []
    sp = os.path.join(d, "sparse", "0")
    if not os.path.isdir(sp):
        sp = os.path.join(d, "sparse")
    if not os.path.isfile(os.path.join(sp, "cameras.bin")):
        probs.append(f"{sp}/cameras.bin missing (binary COLMAP model required; gsplat's text reader crashes)")
    if not os.path.isfile(os.path.join(sp, "points3D.bin")):
        probs.append(f"{sp}/points3D.bin missing")
    if not os.path.isdir(os.path.join(d, "images")):
        probs.append(f"{d}/images missing")
    return probs


class Job:
    def __init__(self, name: str, opts: C.TrainOpts, vdir: str):
        self.name, self.opts, self.vdir = name, opts, vdir
        self.cmd = C.build_command(opts)
        self.proc: Optional[subprocess.Popen] = None
        self.t0 = 0.0
        self.t_end: Optional[float] = None
        self.t_ply7k: Optional[float] = None
        self._ply7k_size = -1
        self.rc: Optional[int] = None
        self.logpath = os.path.join(vdir, "log.txt")
        self.timed_out = False
        self.end_reason = ""            # "", "timeout" or "stalled"
        self.t_term: Optional[float] = None
        self.t_kill: Optional[float] = None
        self.abandoned = False
        self._log_size = -1
        self._log_t = 0.0

    @property
    def preview_path(self) -> str:
        return os.path.join(self.vdir, "ply", self.opts.preview_ply_name)

    def start(self) -> None:
        os.makedirs(self.vdir, exist_ok=True)
        self._lf = open(self.logpath, "wb")
        self.t0 = time.time()
        self._log_t = self.t0
        self.proc = subprocess.Popen(self.cmd, cwd=C.TRAINER_CWD, stdout=self._lf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"{self.name}: started pid {self.proc.pid}")

    def poll(self) -> bool:
        """True while running."""
        if self.proc is None or self.abandoned:
            return False
        if self.t_ply7k is None and os.path.isfile(self.preview_path):
            sz = os.path.getsize(self.preview_path)
            if sz > 0 and sz == self._ply7k_size:      # size stable over two polls: file complete
                self.t_ply7k = time.time() - self.t0
                log(f"{self.name}: 7k preview PLY ready after {self.t_ply7k:.0f} s")
            self._ply7k_size = sz
        rc = self.proc.poll()
        if rc is None:
            return True
        self.rc = rc
        self.t_end = time.time()
        self._lf.close()
        if self.t_ply7k is None and os.path.isfile(self.preview_path):
            self.t_ply7k = self.t_end - self.t0
        return False

    def terminate(self, reason: str = "timeout") -> None:
        """Ask the trainer to stop (SIGTERM to its process group). enforce() escalates."""
        if self.proc is None or self.rc is not None or self.t_term is not None:
            return
        self.end_reason = reason
        self.timed_out = reason == "timeout"
        self.t_term = time.time()
        log(f"{self.name}: {reason}; SIGTERM to process group {self.proc.pid}")
        self._signal(signal.SIGTERM)

    def _signal(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except OSError:
            try:
                self.proc.send_signal(sig)
            except OSError:
                pass

    def enforce(self, now: float) -> bool:
        """After terminate(): SIGKILL after KILL_GRACE_S, give up waiting after ABANDON_S more.
        Returns True when the job must be dropped from the running list (it is gone or abandoned)."""
        if self.t_term is None or self.proc is None:
            return False
        if self.proc.poll() is not None:
            return False                     # poll() in the loop records it
        if self.t_kill is None and now - self.t_term > KILL_GRACE_S:
            self.t_kill = now
            log(f"{self.name}: still alive {KILL_GRACE_S:.0f} s after SIGTERM; SIGKILL")
            self._signal(signal.SIGKILL)
            return False
        if self.t_kill is not None and now - self.t_kill > ABANDON_S:
            # unkillable (uninterruptible sleep in the driver); stop waiting so the stage can finish
            self.rc = -9
            self.t_end = now
            self.abandoned = True
            try:
                self._lf.close()
            except OSError:
                pass
            log(f"{self.name}: did not exit {ABANDON_S:.0f} s after SIGKILL; abandoned (pid {self.proc.pid})")
            return True
        return False

    def check_stall(self, now: float, stall_s: float) -> bool:
        """True when the log has not grown for stall_s seconds (trainer hung, GPU idle)."""
        if self.proc is None or self.rc is not None or self.t_term is not None or stall_s <= 0:
            return False
        try:
            sz = os.path.getsize(self.logpath)
        except OSError:
            sz = -1
        if sz != self._log_size:
            self._log_size, self._log_t = sz, now
            return False
        return now - self._log_t > stall_s


def run_jobs(jobs: List[Job], parallel: bool, tmp: str, max_wall_s: float,
             stall_s: float = 0.0) -> Dict[str, float]:
    """Run jobs (all at once if parallel, else one after another). Returns GPU-memory samples.

    Each job has its own wall limit (max_wall_s from its start) and a stall watchdog (no log output for
    stall_s). A job past either limit gets SIGTERM, then SIGKILL, and is abandoned if it still does not
    exit, so this loop always ends."""
    samples = {"peak_used_mb": 0, "base_used_mb": G.gpu_used_mb() or 0}
    t_start = time.time()
    pending = list(jobs)
    running: List[Job] = []
    last_prog = 0.0
    while pending or running:
        while pending and (parallel or not running):
            j = pending.pop(0)
            j.start()
            running.append(j)
        time.sleep(POLL_S)
        now = time.time()
        for j in list(running):
            if not j.poll():
                running.remove(j)
                log(f"{j.name}: exited rc={j.rc} after {j.t_end - j.t0:.0f} s"
                    + (f" ({j.end_reason})" if j.end_reason else ""))
                continue
            if j.t_term is None:
                if now - j.t0 > max_wall_s:
                    j.terminate("timeout")
                    log(f"{j.name}: wall limit {max_wall_s:.0f} s reached")
                elif j.check_stall(now, stall_s):
                    j.terminate("stalled")
                    log(f"{j.name}: no trainer output for {stall_s:.0f} s; treated as hung")
            if j.enforce(now):
                running.remove(j)
        if now - last_prog > PROGRESS_S:
            last_prog = now
            used = G.gpu_used_mb()
            if used:
                samples["peak_used_mb"] = max(samples["peak_used_mb"], used)
            prog = {"elapsed_s": round(now - t_start), "variants": {}}
            for j in jobs:
                p = S.progress_from_log(j.logpath) if os.path.exists(j.logpath) else None
                state = "pending" if j.proc is None else ("running" if j.rc is None else ("ok" if j.rc == 0 else "failed"))
                if j.end_reason and j.rc is None:
                    state = "stopping"
                elif j.end_reason:
                    state = j.end_reason
                prog["variants"][j.name] = {
                    "state": state,
                    "step": p[0] if p else None, "of": p[1] if p else j.opts.max_steps,
                    "preview_ply": j.preview_path if j.t_ply7k is not None else None,
                    "t_preview_s": j.t_ply7k}
            _atomic_json(os.path.join(tmp, "progress.json"), prog)
    return samples


def _sanitise(o):
    """Strict JSON: non-finite floats become null."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    return o


def _atomic_json(path: str, obj) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(_sanitise(obj), f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def tail(path: str, n: int = 25) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 5000))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(data.replace("\r", "\n").splitlines()[-n:])


def finish_variant(j: Job, eval_steps: List[int], gpu_samples: dict, parallel: bool, stage: str) -> dict:
    from pipeline.export import ply_io
    status = "ok" if (j.rc == 0 and not j.end_reason) else (j.end_reason or "failed")
    extra: dict = {
        "status": status, "rc": j.rc, "wall_s": round((j.t_end or time.time()) - j.t0, 1),
        "t_preview_s": round(j.t_ply7k, 1) if j.t_ply7k is not None else None,
        "parallel": parallel, "cmd": " ".join(j.cmd),
        "end_reason": j.end_reason or None, "abandoned": j.abandoned or None,
        "gpu_used_total_peak_gb": round(gpu_samples["peak_used_mb"] / 1024, 2),
        "gpu_used_total_base_gb": round(gpu_samples["base_used_mb"] / 1024, 2),
    }
    # last step that produced val stats (a timed-out or crashed run can still have the 7k one)
    final_step = None
    for s in reversed(eval_steps):
        if os.path.isfile(os.path.join(j.vdir, "stats", f"val_step{s - 1}.json")):
            final_step = s
            break
    steps_for_rec = eval_steps
    ply_name = f"point_cloud_{(final_step or eval_steps[-1]) - 1}.ply"
    ply_path = os.path.join(j.vdir, "ply", ply_name)
    if final_step is not None and final_step != eval_steps[-1]:
        extra["partial"] = True
        extra["partial_step"] = final_step
        # the record's headline metrics are those of the last step that was evaluated (the 7k preview)
        steps_for_rec = eval_steps[:eval_steps.index(final_step) + 1]
    if os.path.isfile(ply_path):
        n_ply = ply_io.count_vertices(ply_path)
        extra["final_ply"] = f"{stage}/{j.name}/ply/{ply_name}"   # relative to the run directory
        extra["n_gs_ply"] = n_ply
    if status != "ok":
        extra["log_tail"] = tail(j.logpath)
    rec = S.variant_record(j.name, j.vdir, steps_for_rec, extra)
    if rec.get("n_gs") is not None and rec.get("n_gs_ply") is not None:
        rec["nan_dropped"] = int(rec["n_gs"] - rec["n_gs_ply"])
    return rec


def cleanup(tmp: str, recs: List[dict], chosen: Optional[str], keep_losers: bool, quick: bool) -> None:
    for r in recs:
        vdir = os.path.join(tmp, r["name"])
        if r["name"] == chosen:
            # keep both PLYs and the final checkpoint; drop the 7k checkpoint and training renders
            for f in os.listdir(os.path.join(vdir, "ckpts")) if os.path.isdir(os.path.join(vdir, "ckpts")) else []:
                if "_6999_" in f and not quick:
                    os.remove(os.path.join(vdir, "ckpts", f))
            continue
        if keep_losers:
            continue
        for sub in ("ckpts", "ply"):
            shutil.rmtree(os.path.join(vdir, sub), ignore_errors=True)
        r["losing_checkpoints_deleted"] = True
        if "final_ply" in r:
            r["final_ply_deleted"] = r.pop("final_ply")
    # tensorboard logs are not needed
    for r in recs:
        shutil.rmtree(os.path.join(tmp, r["name"], "tb"), ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.train", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run directory, e.g. /w/runs/room")
    ap.add_argument("--data-dir", help="override <run>/40_align (smoke tests)")
    ap.add_argument("--quick", action="store_true", help="7k steps instead of 30k")
    ap.add_argument("--variants", default="r0,r1")
    ap.add_argument("--parallel", choices=["auto", "yes", "force", "no"], default="auto")
    ap.add_argument("--normalize", action="store_true",
                    help="gsplat world normalisation ON (frame-check tests only; the pipeline runs with it off)")
    ap.add_argument("--no-scene-unit", action="store_true",
                    help="do not rescale MCMC noise and scale_reg by the scene unit (reproduces the failure)")
    ap.add_argument("--cap-max", type=int, default=1_500_000)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="wall limit per variant, from its own start (default 90 full, 30 quick)")
    ap.add_argument("--stall-minutes", type=float, default=STALL_S / 60,
                    help="stop a variant whose log shows no output for this long (0 = off)")
    ap.add_argument("--keep-losers", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo even if train.json exists")
    ap.add_argument("--stage-dir", default=STAGE, help="output directory name inside the run (default 50_train)")
    ap.add_argument("--lock-dir", default=G.LOCK_DIR)
    a = ap.parse_args(argv)

    run = os.path.abspath(a.run)
    final = os.path.join(run, a.stage_dir)
    tmp = final + ".tmp"
    data_dir = os.path.abspath(a.data_dir) if a.data_dir else os.path.join(run, "40_align")

    if os.path.isfile(os.path.join(final, "train.json")) and not a.force:
        log(f"{final}/train.json exists; nothing to do (use --force to redo)")
        return 0
    probs = check_data_dir(data_dir)
    if probs:
        for p in probs:
            log("ERROR " + p)
        return 20
    os.makedirs(run, exist_ok=True)
    if free_gb(run) < MIN_FREE_GB:
        log(f"ERROR only {free_gb(run):.1f} GB free (< {MIN_FREE_GB})")
        return 20

    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    for v in names:
        if v not in ("r0", "r1", "r0nb"):
            log(f"ERROR unknown variant {v}")
            return 20
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    parallel, lock, why_par = G.decide_parallel(a.parallel, len(names), a.lock_dir)
    log(why_par)
    eval_steps = C.TrainOpts("", "", quick=a.quick).eval_steps
    max_wall_s = 60 * (a.max_minutes if a.max_minutes is not None else (30.0 if a.quick else 90.0))
    stall_s = 60 * a.stall_minutes
    t_stage = time.time()

    def mk_jobs(vnames: List[str]) -> List[Job]:
        return [Job(v, C.TrainOpts(data_dir=data_dir, result_dir=os.path.join(tmp, v), variant=v,
                                   quick=a.quick, normalize=a.normalize, cap_max=a.cap_max, scene_k=k), os.path.join(tmp, v))
                for v in vnames]

    k = None
    warnings: List[str] = []
    from pipeline.export import colmap_io
    try:
        for cam in colmap_io.read_cameras(data_dir).values():
            if cam.distorted:
                warnings.append(f"camera {cam.id} is {cam.model}, not PINHOLE: gsplat undistorts at load time "
                                "(the pipeline expects the undistorted model from 30_undist/40_align)")
    except (OSError, ValueError, KeyError) as e:
        warnings.append(f"could not read cameras.bin: {e}")
    if not a.normalize and not a.no_scene_unit:
        k = C.scene_unit([im.centre() for im in colmap_io.read_images(data_dir).values()])
        log(f"scene unit k = {k:.4f} model units (median camera distance); noise_lr/k^2, scale_reg/k")
    jobs = mk_jobs(names)
    try:
        samples = run_jobs(jobs, parallel, tmp, max_wall_s, stall_s)
    finally:
        if lock:
            lock.release()
    recs = [finish_variant(j, eval_steps, samples, parallel, a.stage_dir) for j in jobs]

    complete = [r for r in recs if S.usable(r) and not r.get("partial")]
    stopped = [r["name"] for r in recs if r.get("status") in ("timeout", "stalled")]
    for r in recs:
        if r.get("status") in ("timeout", "stalled"):
            warnings.append(f"{r['name']}: {r['status']} after {r.get('wall_s')} s"
                            + (" (process abandoned)" if r.get("abandoned") else ""))
    any_usable = any(S.usable(r) or (r.get("partial") and r.get("psnr") is not None) for r in recs)
    # r0nb is the NaN/crash fallback. After a timeout or stall, a 7k model (if any) is the next rung (6.5);
    # another full run would only repeat the wait.
    if not complete and "r0nb" not in names and not (stopped and any_usable):
        # NaN guard / crash: step down to R0 without the bilateral grid (fallback ladder)
        log("no complete usable variant; retrying as r0nb (no bilateral grid)")
        warnings.append("fallback: r0nb (no bilateral grid) after r0/r1 failed or produced NaN")
        j2 = mk_jobs(["r0nb"])
        s2 = run_jobs(j2, False, tmp, max_wall_s, stall_s)
        recs.append(finish_variant(j2[0], eval_steps, s2, False, a.stage_dir))

    pool = [r for r in recs if S.usable(r) and not r.get("partial")]
    degraded = False
    if pool:
        chosen, why = S.choose(pool)
    else:
        # fallback ladder 6.5, last rung: a partial model (the 7k PLY) of a variant that timed out,
        # stalled or crashed after its 7k evaluation
        part = [r for r in recs if r.get("partial") and r.get("final_ply") and r.get("psnr") is not None
                and not r.get("nan_dropped")]
        degraded = bool(part)
        if part:
            best_p = max(part, key=lambda r: r["psnr"])
            chosen = best_p["name"]
            why = (f"{chosen} stopped ({best_p.get('status')}) after its step-{best_p.get('partial_step')} "
                   f"evaluation; best held-out PSNR among partial models")
        else:
            chosen, why = S.choose([])
    if degraded and chosen:
        why = "DEGRADED (7k model only): " + why
        warnings.append("only a partial (7k) model exists")
    best = next((r for r in recs if r["name"] == chosen), None)

    if best:
        if best["psnr"] is not None and best["psnr"] < PSNR_GATE_DB:
            warnings.append(f"held-out PSNR {best['psnr']:.2f} dB < {PSNR_GATE_DB}")
        if best.get("time_s") and best["time_s"] > TIME_GATE_S:
            warnings.append(f"training took {best['time_s'] / 60:.0f} min > 45")
    for r in recs:
        if r.get("nan_dropped"):
            warnings.append(f"{r['name']}: {r['nan_dropped']} NaN/Inf splats dropped by the PLY exporter")
    cleanup(tmp, recs, chosen, a.keep_losers, a.quick)

    result = {
        "schema": "namma-space/train@1",
        "variants": recs, "chosen": chosen, "why": why,
        "chosen_ply": best.get("final_ply") if best else None,
        "preview_ply": f"{a.stage_dir}/{chosen}/ply/{C.TrainOpts('', '').preview_ply_name}" if chosen else None,
        "quick": a.quick, "normalize_world_space": a.normalize, "data_dir": data_dir,
        "parallel": parallel, "parallel_reason": why_par, "cap_max": a.cap_max, "scene_unit_k": k,
        "test_every": 8, "eval_steps": eval_steps, "wall_s": round(time.time() - t_stage, 1),
        "warnings": warnings,
    }
    _atomic_json(os.path.join(tmp, "train.json"), result)
    if os.path.isdir(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    log(f"done: chosen={chosen} ({why}); wrote {final}/train.json")
    if not chosen:
        return 20
    return 30 if (degraded or any("PSNR" in w and "<" in w for w in warnings)) else 0


if __name__ == "__main__":
    sys.exit(main())
