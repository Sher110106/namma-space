"""SfM stage: race COLMAP variants, filter the camera trajectory, undistort. Stdlib only.

python -m pipeline.sfm --run <run_dir> [options]
reads  <run>/10_frames/images (+ frames.json for times and blur scores)
writes <run>/20_sfm/{A,B,D[,C]}/ + sfm.json and <run>/30_undist/{images,sparse}  (temp dirs, renamed when complete)

Exit codes (CONTRACTS 2.11): 0 usable result (a "partial" one is flagged in sfm.json and printed),
10 nothing usable: no variant registered 60% of the frames in one model (time ranges to reshoot are printed),
20 infrastructure problem (no colmap, missing cache files, low disk, no images, every variant crashed).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

from . import choose as choose_mod
from . import cmds, colmapio, sanity

EXIT_OK, EXIT_CAPTURE, EXIT_INFRA = 0, 10, 20
TIMEOUT_RC = -1000
IMG_EXT = (".jpg", ".jpeg", ".png")
MIN_FREE_GB = 15.0
RANGE_MIN_FRAMES = 3
DESCRIPTIONS = {
    "A": "ALIKED_N32 + ALIKED_LIGHTGLUE, sequential + loop detection, incremental mapper",
    "B": "CPU SIFT + SIFT_LIGHTGLUE, sequential + loop detection, incremental mapper",
    "C": "ALIKED_N32 + ALIKED_LIGHTGLUE, exhaustive matching on A's features, incremental mapper",
    "D": "global_mapper on a copy of A's database after view_graph_calibrator",
}


def cpu_budget() -> int:
    if os.environ.get("NS_CPUS"):
        return max(1, int(os.environ["NS_CPUS"]))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 4


@dataclass
class Variant:
    name: str
    root: str
    status: str = "pending"      # pending | ok | crashed | timeout | failed | skipped
    error: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    steps: dict = field(default_factory=dict)       # label -> seconds
    ok_steps: set = field(default_factory=set)      # labels of steps that exited 0
    summary: dict = field(default_factory=dict)
    clean: str = ""              # post-filter model directory
    sane: sanity.SanityResult | None = None
    models: list = field(default_factory=list)

    @property
    def db(self) -> str:
        return os.path.join(self.root, "db.db")

    @property
    def sparse(self) -> str:
        return os.path.join(self.root, "sparse")

    @property
    def log(self) -> str:
        return os.path.join(self.root, "log.txt")


class Ctx:
    def __init__(self, a: argparse.Namespace):
        self.run_dir = os.path.abspath(a.run)
        self.images = os.path.abspath(a.images or os.path.join(self.run_dir, "10_frames", "images"))
        self.tmp20 = os.path.join(self.run_dir, "20_sfm.tmp")
        self.tmp30 = os.path.join(self.run_dir, "30_undist.tmp")
        self.cfg = cmds.Cfg(cache=a.cache, camera_model=a.camera_model, max_image_size=a.max_image_size,
                            overlap=a.overlap, loop_detection=0 if a.no_loop else 1, gpu=not a.cpu)
        self.timeout = a.variant_timeout
        self.ncpu = cpu_budget()
        self.threads = max(2, self.ncpu // 3)
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}
        self.names: list[str] = []
        self.meta: dict[str, dict] = {}
        self.variants: dict[str, Variant] = {}
        self.t0 = time.time()
        self.args = a
        self.driver_log = None

    def say(self, msg: str) -> None:
        line = f"[sfm {time.strftime('%H:%M:%S')}] {msg}"
        with self.lock:
            print(line, flush=True)
            if self.driver_log:
                self.driver_log.write(line + "\n")
                self.driver_log.flush()


# ----------------------------------------------------------------------------- running commands

def spawn(ctx: Ctx, v: Variant, argv: list[str], timeout_s: float) -> int:
    """Run one colmap command with output appended to the variant log. Returns the exit code, TIMEOUT_RC on timeout."""
    with open(v.log, "a") as f:
        f.write("\n$ " + " ".join(argv) + "\n")
        f.flush()
        try:
            p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        except OSError as e:
            f.write(f"cannot start: {e}\n")
            return 127
        ctx.procs[v.name] = p
        try:
            return p.wait(timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            f.write(f"\n-- killed after {timeout_s:.0f} s timeout\n")
            return TIMEOUT_RC


def step(ctx: Ctx, v: Variant, label: str, argv: list[str], retry_argv: list[str] | None = None, retries: int = 0) -> bool:
    """Run one pipeline step with a share of the variant's time budget; retry once on a non-timeout failure."""
    t = time.time()
    rc = 0
    for attempt in range(retries + 1):
        remaining = ctx.timeout - (time.time() - v.t_start)
        if remaining <= 5:
            rc = TIMEOUT_RC
            break
        use = retry_argv if (attempt > 0 and retry_argv) else argv
        rc = spawn(ctx, v, use, remaining)
        if rc == 0 or rc == TIMEOUT_RC:
            break
        if attempt < retries:
            ctx.say(f"{v.name} {label}: exit {rc}, retrying once in {ctx.args.retry_delay:.0f} s")
            time.sleep(ctx.args.retry_delay)
    v.steps[label] = round(time.time() - t, 1)
    if rc != 0:
        v.status = "timeout" if rc == TIMEOUT_RC else "crashed"
        v.error = f"{label} exited with {rc}" if rc != TIMEOUT_RC else f"{label} timed out after {ctx.timeout:.0f} s"
        ctx.say(f"{v.name} {label}: FAILED ({v.error})")
        return False
    v.ok_steps.add(label)
    ctx.say(f"{v.name} {label}: ok in {v.steps[label]:.0f} s")
    return True


def analyze(ctx: Ctx, v: Variant, model: str) -> dict:
    """Numbers from `colmap model_analyzer`, via a temp file so a failure never loses the log."""
    out = os.path.join(v.root, "analyzer.tmp")
    with open(out, "w") as f:
        p = subprocess.run(cmds.model_analyzer(model), stdout=f, stderr=subprocess.STDOUT, timeout=300)
    with open(out) as f:
        text = f.read()
    os.remove(out)
    with open(v.log, "a") as f:
        f.write(f"\n$ model_analyzer {model} (rc {p.returncode})\n{text}\n")
    return colmapio.parse_model_analyzer(text)


# ----------------------------------------------------------------------------- per-variant finishing

def finish_variant(ctx: Ctx, v: Variant) -> None:
    """After the mapper: pick the largest model, run the trajectory filter and point filtering, record numbers."""
    models = colmapio.list_models(v.sparse)
    v.models = []
    if not models:
        v.status, v.error = "failed", "the mapper wrote no model"
        ctx.say(f"{v.name}: no model written")
        return
    stats = []
    for m in models:
        s = analyze(ctx, v, m)
        stats.append((int(s.get("registered", 0)), m, s))
    stats.sort(key=lambda x: -x[0])
    v.models = [{"dir": os.path.basename(m), "registered": r} for r, m, _ in stats]
    best_n, best_dir, raw = stats[0]
    if best_n < 3:
        v.status, v.error = "failed", "largest model has fewer than 3 images"
        return
    # --- trajectory sanity filter
    index = {n: i for i, n in enumerate(ctx.names)}
    _, recs = colmapio.read_model(best_dir)
    cams = [sanity.Cam(index[r.name], r.name, r.centre) for r in recs if r.name in index]
    res = sanity.filter_trajectory(cams)
    v.sane = res
    src = best_dir
    if res.dropped:
        names_file = os.path.join(v.root, "drop_names.txt")
        with open(names_file, "w") as f:
            f.write("\n".join(d["name"] for d in res.dropped) + "\n")
        dropped_dir = os.path.join(v.root, "clean_nodrop")
        os.makedirs(dropped_dir, exist_ok=True)
        if spawn(ctx, v, cmds.image_deleter(best_dir, dropped_dir, names_file), 600) != 0:
            v.status, v.error = "crashed", "image_deleter failed"
            return
        src = dropped_dir
        ctx.say(f"{v.name}: sanity filter dropped {len(res.dropped)} cameras: "
                + ", ".join(d["name"] for d in res.dropped[:8]) + (" ..." if len(res.dropped) > 8 else ""))
    v.clean = os.path.join(v.root, "clean")
    os.makedirs(v.clean, exist_ok=True)
    if spawn(ctx, v, cmds.point_filtering(src, v.clean), 600) != 0:
        v.status, v.error = "crashed", "point_filtering failed"
        return
    post = analyze(ctx, v, v.clean)
    n_img = max(1, len(ctx.names))
    reg = int(post.get("registered", 0))
    v.status = "ok" if reg >= 3 else "failed"
    v.summary = {
        "name": v.name,
        "desc": DESCRIPTIONS.get(v.name, ""),
        "reg_frac": round(reg / n_img, 4),
        "reg_frac_raw": round(best_n / n_img, 4),
        "n_registered": reg,
        "n_registered_raw": best_n,
        "n_models": len(models),
        "reproj_px": round(post.get("reproj_px", 0.0), 4),
        "reproj_px_raw": round(raw.get("reproj_px", 0.0), 4),
        "n_pts": int(post.get("points", 0)),
        "n_pts_raw": int(raw.get("points", 0)),
        "track_len": round(post.get("track_len", 0.0), 3),
        "n_dropped": len(res.dropped),
        "max_step_ratio": res.max_step_ratio,
        "jumps_left": res.jumps_left,
        "model_sizes": [x["registered"] for x in v.models],
    }
    ctx.say(f"{v.name}: {reg}/{n_img} registered ({reg / n_img:.1%}) in the largest of {len(models)} model(s), "
            f"{v.summary['reproj_px']:.2f} px, {v.summary['n_pts']} points")


def timed(fn):
    def wrapper(ctx: Ctx, v: Variant, *a, **k):
        v.t_start = time.time()
        v.status = "running"
        try:
            fn(ctx, v, *a, **k)
        except Exception as e:  # noqa: BLE001 - one variant must never take the race down
            v.status, v.error = "crashed", f"{type(e).__name__}: {e}"
            ctx.say(f"{v.name}: exception {v.error}")
        finally:
            v.t_end = time.time()
            if v.status == "running":
                v.status = "ok"
    return wrapper


def _ensure_dirs(v: Variant) -> None:
    os.makedirs(v.sparse, exist_ok=True)


@timed
def run_a(ctx: Ctx, v: Variant, matched: threading.Event) -> None:
    _ensure_dirs(v)
    t = ctx.threads
    try:
        if not step(ctx, v, "extract", cmds.extract_aliked(v.db, ctx.images, ctx.cfg, t), retries=1):
            return
        if not step(ctx, v, "match", cmds.match_sequential_aliked(v.db, ctx.cfg, t), retries=1):
            return
    finally:
        matched.set()   # database is final (or A is dead): D may copy it while A maps, or give up
    if not step(ctx, v, "map", cmds.mapper(v.db, ctx.images, v.sparse, t)):
        return
    finish_variant(ctx, v)


@timed
def run_b(ctx: Ctx, v: Variant) -> None:
    _ensure_dirs(v)
    t = ctx.threads
    if not step(ctx, v, "extract", cmds.extract_sift_cpu(v.db, ctx.images, ctx.cfg, max(t, ctx.ncpu // 2)), retries=1):
        return
    if not step(ctx, v, "match", cmds.match_sequential_sift(v.db, ctx.cfg, t), retries=1):
        return
    if not step(ctx, v, "map", cmds.mapper(v.db, ctx.images, v.sparse, t)):
        return
    finish_variant(ctx, v)


@timed
def run_d(ctx: Ctx, v: Variant, a: Variant, a_matched: threading.Event) -> None:
    _ensure_dirs(v)
    a_matched.wait()
    v.t_start = time.time()   # D's own time: it reuses A's features and matches, which are counted under A
    if "match" not in a.ok_steps or not os.path.isfile(a.db):
        v.status, v.error = "skipped", "variant A has no finished database"
        ctx.say("D: skipped (A has no matched database)")
        return
    colmapio.copy_db(a.db, v.db)
    t = ctx.threads
    if not step(ctx, v, "calibrate", cmds.view_graph_calibrator(v.db)):
        return
    if not step(ctx, v, "map", cmds.global_mapper(v.db, ctx.images, v.sparse, t, gpu=ctx.cfg.gpu),
                retry_argv=cmds.global_mapper(v.db, ctx.images, v.sparse, t, gpu=False), retries=1):
        return
    finish_variant(ctx, v)


@timed
def run_c(ctx: Ctx, v: Variant, a: Variant) -> None:
    _ensure_dirs(v)
    if not os.path.isfile(a.db) or "extract" not in a.ok_steps:
        v.status, v.error = "skipped", "variant A has no feature database"
        return
    colmapio.copy_db(a.db, v.db)
    t = ctx.threads * 2
    if not step(ctx, v, "match", cmds.match_exhaustive_aliked(v.db, ctx.cfg, t), retries=1):
        return
    if not step(ctx, v, "map", cmds.mapper(v.db, ctx.images, v.sparse, t)):
        return
    finish_variant(ctx, v)


# ----------------------------------------------------------------------------- missing ranges, undistort

def runs_of(indices: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in sorted(set(indices)):
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def missing_info(ctx: Ctx, registered: set[str], dropped: list[dict], kp: dict[str, int]) -> tuple[list[dict], list[dict]]:
    drop_reason = {d["name"]: d["reason"] for d in dropped}
    missing = [i for i, n in enumerate(ctx.names) if n not in registered]
    unreg = []
    for i in missing:
        n = ctx.names[i]
        m = ctx.meta.get(n, {})
        unreg.append({"name": n, "idx": i, "t": m.get("t"), "sharp": m.get("sharp"),
                      "n_features": kp.get(n), "reason": "dropped: " + drop_reason[n] if n in drop_reason else "not registered"})
    ranges = []
    for a, b in runs_of(missing):
        if b - a + 1 < RANGE_MIN_FRAMES:
            continue
        ta, tb = ctx.meta.get(ctx.names[a], {}).get("t"), ctx.meta.get(ctx.names[b], {}).get("t")
        ranges.append({"first": ctx.names[a], "last": ctx.names[b], "start_idx": a, "end_idx": b, "n_frames": b - a + 1,
                       "t_start": ta, "t_end": tb})
    return unreg, ranges


def undistort(ctx: Ctx, v: Variant) -> dict:
    shutil.rmtree(ctx.tmp30, ignore_errors=True)
    os.makedirs(ctx.tmp30)
    rc = spawn(ctx, v, cmds.image_undistorter(ctx.images, v.clean, ctx.tmp30, ctx.cfg.max_image_size), 1800)
    if rc != 0:
        raise RuntimeError(f"image_undistorter exited with {rc}")
    # COLMAP leaves PMVS-style helper scripts that point into the temp directory; we do not use them.
    shutil.rmtree(os.path.join(ctx.tmp30, "stereo"), ignore_errors=True)
    for f in os.listdir(ctx.tmp30):
        if f.startswith("run-colmap"):
            os.remove(os.path.join(ctx.tmp30, f))
    sparse = os.path.join(ctx.tmp30, "sparse")
    cams, imgs = colmapio.read_model(sparse)
    models = {c.model for c in cams}
    if models != {"PINHOLE"}:
        raise RuntimeError(f"undistorted cameras are {models}, expected PINHOLE")
    # A real copy at sparse/0 so readers that want either layout (sparse/ or sparse/0/) find the binary model.
    shutil.copytree(sparse, os.path.join(sparse, "_0"), ignore=shutil.ignore_patterns("_0"))
    os.rename(os.path.join(sparse, "_0"), os.path.join(sparse, "0"))
    n_files = len([f for f in os.listdir(os.path.join(ctx.tmp30, "images")) if f.lower().endswith(IMG_EXT)])
    return {"n_images": len(imgs), "n_image_files": n_files, "camera_model": "PINHOLE",
            "width": cams[0].width, "height": cams[0].height, "max_size": ctx.cfg.max_image_size,
            "fx": cams[0].params[0], "fy": cams[0].params[1], "cx": cams[0].params[2], "cy": cams[0].params[3]}


# ----------------------------------------------------------------------------- main flow

def preflight(ctx: Ctx) -> int:
    a = ctx.args
    if shutil.which(cmds.COLMAP) is None:
        print(f"[sfm] colmap not found ({cmds.COLMAP}); run inside namma-colmap:0.2", file=sys.stderr)
        return EXIT_INFRA
    if not os.path.isdir(ctx.images):
        print(f"[sfm] no image directory {ctx.images}", file=sys.stderr)
        return EXIT_INFRA
    ctx.names = sorted(f for f in os.listdir(ctx.images) if f.lower().endswith(IMG_EXT))
    if len(ctx.names) < 5:
        print(f"[sfm] only {len(ctx.names)} images in {ctx.images}", file=sys.stderr)
        return EXIT_INFRA
    free = shutil.disk_usage(ctx.run_dir).free / 2**30
    if free < a.min_free_gb:
        print(f"[sfm] only {free:.1f} GB free (< {a.min_free_gb:.0f} GB); refusing to start", file=sys.stderr)
        return EXIT_INFRA
    need = sorted({f for v in a.variants for f in cmds.REQUIRED_CACHE_FILES.get(v, [])})
    missing = [f for f in need if not os.path.isfile(ctx.cfg.p(f))]
    if missing:
        print(f"[sfm] missing files in {ctx.cfg.cache}: {', '.join(missing)}", file=sys.stderr)
        return EXIT_INFRA
    fj = os.path.join(ctx.run_dir, "10_frames", "frames.json")
    if os.path.isfile(fj):
        try:
            with open(fj) as f:
                kept = json.load(f).get("kept", [])
            for k in kept:
                ctx.meta[k["name"]] = {"t": k.get("t"), "sharp": k.get("sharp")}
        except (OSError, ValueError, KeyError, AttributeError):
            pass
    return 0


def race(ctx: Ctx, wanted: list[str], sequential: bool) -> None:
    for n in wanted:
        d = os.path.join(ctx.tmp20, n)
        os.makedirs(d, exist_ok=True)
        ctx.variants[n] = Variant(n, d)
    a_matched = threading.Event()
    jobs: list[threading.Thread] = []
    if "A" in wanted:
        jobs.append(threading.Thread(target=run_a, args=(ctx, ctx.variants["A"], a_matched), name="A"))
    elif "D" in wanted:
        wanted.remove("D")
        ctx.say("D needs A's database; D skipped")
    if "B" in wanted:
        jobs.append(threading.Thread(target=run_b, args=(ctx, ctx.variants["B"]), name="B"))
    if "D" in wanted:
        jobs.append(threading.Thread(target=run_d, args=(ctx, ctx.variants["D"], ctx.variants["A"], a_matched), name="D"))
    if sequential:
        for j in jobs:  # A first (D waits for A's matching), then the rest in order
            j.start()
            if j.name == "A":
                a_matched.wait()
                j.join()
            else:
                j.join()
    else:
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()


def variant_row(ctx: Ctx, v: Variant) -> dict:
    row = {"name": v.name, "desc": DESCRIPTIONS.get(v.name, ""), "status": v.status, "error": v.error,
           "reuses": "A (features and matches)" if v.name in ("C", "D") else "",
           "time_s": round((v.t_end or time.time()) - v.t_start, 1) if v.t_start else 0.0, "steps_s": v.steps,
           "reg_frac": 0.0, "n_models": len(v.models), "reproj_px": 0.0, "n_pts": 0, "track_len": 0.0}
    if v.status == "ok":
        row.update(v.summary)
        row["status"] = "ok"
    return row


def write_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def _install_signal_handlers(ctx: Ctx) -> None:
    """On SIGTERM/SIGINT kill the colmap process groups we started, then exit, so `docker stop` leaves nothing behind."""
    def handler(signum, _frame):
        for p in list(ctx.procs.values()):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        os._exit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def run(a: argparse.Namespace) -> int:
    cmds.COLMAP = a.colmap
    ctx = Ctx(a)
    if threading.current_thread() is threading.main_thread():
        _install_signal_handlers(ctx)
    rc = preflight(ctx)
    if rc:
        return rc
    shutil.rmtree(ctx.tmp20, ignore_errors=True)
    os.makedirs(ctx.tmp20)
    ctx.driver_log = open(os.path.join(ctx.tmp20, "driver.log"), "w")
    ctx.say(f"{len(ctx.names)} images, {ctx.ncpu} cpus, {ctx.threads} threads per variant, variants {','.join(a.variants)}")
    wanted = list(a.variants)
    race(ctx, wanted, a.sequential)
    rows = [variant_row(ctx, ctx.variants[n]) for n in wanted if n in ctx.variants]
    # C: only if nothing reached 90% and the sequence is short enough for exhaustive matching.
    best_reg = max((r["reg_frac"] for r in rows if r["status"] == "ok"), default=0.0)
    a_var = ctx.variants.get("A")
    if (a.allow_c and best_reg < 0.90 and len(ctx.names) <= 350 and a_var and os.path.isfile(a_var.db)
            and "C" not in ctx.variants):
        ctx.say(f"best share {best_reg:.1%} < 90% and {len(ctx.names)} <= 350 frames: running variant C")
        c = Variant("C", os.path.join(ctx.tmp20, "C"))
        os.makedirs(c.root, exist_ok=True)
        ctx.variants["C"] = c
        run_c(ctx, c, a_var)
        rows.append(variant_row(ctx, c))
    verdict = choose_mod.choose(rows)
    chosen = ctx.variants.get(verdict["chosen"]) if verdict["chosen"] else None
    total_time = round(time.time() - ctx.t0, 1)
    out: dict = {
        "schema": "namma-space/sfm@1", "colmap": "4.2.1", "n_images": len(ctx.names),
        "variants": rows, "chosen": verdict["chosen"], "why": verdict["why"], "partial": verdict["partial"],
        "warnings": list(verdict["warnings"]), "missing_ranges": [], "unregistered": [], "dropped_cameras": [],
        "time_s": total_time, "cpus": ctx.ncpu, "threads_per_variant": ctx.threads,
        "flags": {"overlap": a.overlap, "loop_detection": 0 if a.no_loop else 1, "camera_model": a.camera_model,
                  "max_image_size": a.max_image_size, "cache": a.cache},
    }
    if chosen is None:
        # Nothing usable: report what we have, print the time ranges that failed, keep the temp dir as 20_sfm.
        reg_all = {v.name: v.summary.get("reg_frac", 0.0) for v in ctx.variants.values()}
        ctx.say(f"FAILED: {verdict['why']} ({reg_all})")
        all_crashed = all(r["status"] != "ok" for r in rows)
        best = max((v for v in ctx.variants.values() if v.status == "ok" and v.clean), default=None,
                   key=lambda v: v.summary.get("reg_frac", 0.0))
        if best is not None:   # report where the best (unusable) model is missing frames: these are the ranges to reshoot
            _, recs = colmapio.read_model(best.clean)
            drops = best.sane.dropped if best.sane else []
            out["unregistered"], out["missing_ranges"] = missing_info(ctx, {r.name for r in recs}, drops, {})
            out["dropped_cameras"] = drops
            out["best_unusable"] = best.name
            for r in out["missing_ranges"]:
                ctx.say(f"  reshoot {r['first']} .. {r['last']} ({r['n_frames']} frames" +
                        (f", t {r['t_start']} to {r['t_end']} s)" if r["t_start"] is not None else ")"))
        write_json(os.path.join(ctx.tmp20, "sfm.json"), out)
        _publish(ctx, None)
        return EXIT_INFRA if all_crashed else EXIT_CAPTURE
    # --- chosen variant: missing ranges, undistort
    _, recs = colmapio.read_model(chosen.clean)
    registered = {r.name for r in recs}
    try:
        kp = colmapio.keypoint_counts(chosen.db)
    except Exception:  # noqa: BLE001
        kp = {}
    dropped = chosen.sane.dropped if chosen.sane else []
    out["dropped_cameras"] = dropped
    out["unregistered"], out["missing_ranges"] = missing_info(ctx, registered, dropped, kp)
    if chosen.sane and chosen.sane.jumps_left:
        out["warnings"].append(f"{len(chosen.sane.jumps_left)} camera jump(s) between long runs were kept")
    try:
        out["undist"] = undistort(ctx, chosen)
    except Exception as e:  # noqa: BLE001
        out["warnings"].append(f"undistort failed: {e}")
        ctx.say(f"undistort FAILED: {e}")
        out["time_s"] = round(time.time() - ctx.t0, 1)
        write_json(os.path.join(ctx.tmp20, "sfm.json"), out)
        _publish(ctx, None)
        return EXIT_INFRA
    out["time_s"] = round(time.time() - ctx.t0, 1)
    _cleanup(ctx, chosen.name)
    write_json(os.path.join(ctx.tmp20, "sfm.json"), out)
    _publish(ctx, ctx.tmp30)
    tag = "PARTIAL" if verdict["partial"] else "OK"
    ctx.say(f"{tag}: chosen {chosen.name}: {verdict['why']}")
    for r in out["missing_ranges"]:
        ctx.say(f"  missing {r['first']} .. {r['last']} ({r['n_frames']} frames" +
                (f", t {r['t_start']} to {r['t_end']} s)" if r["t_start"] is not None else ")"))
    return EXIT_OK


def _cleanup(ctx: Ctx, keep: str) -> None:
    mode = ctx.args.keep_db
    for n, v in ctx.variants.items():
        for f in ("clean_nodrop", "drop_names.txt"):
            p = os.path.join(v.root, f)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p) and n != keep:
                os.remove(p)
        if mode == "none" or (mode == "chosen" and n != keep):
            for ext in ("", "-wal", "-shm"):
                if os.path.isfile(v.db + ext):
                    os.remove(v.db + ext)


def _publish(ctx: Ctx, undist_tmp: str | None) -> None:
    """Rename the temp directories into place. 30_undist first: 20_sfm/sfm.json marks the stage done."""
    final20 = os.path.join(ctx.run_dir, "20_sfm")
    final30 = os.path.join(ctx.run_dir, "30_undist")
    if ctx.driver_log:
        ctx.driver_log.close()
        ctx.driver_log = None
    if undist_tmp:
        shutil.rmtree(final30, ignore_errors=True)
        os.rename(undist_tmp, final30)
    else:
        shutil.rmtree(ctx.tmp30, ignore_errors=True)
    shutil.rmtree(final20, ignore_errors=True)
    os.rename(ctx.tmp20, final20)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m pipeline.sfm", description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="run directory (reads 10_frames/, writes 20_sfm/ and 30_undist/)")
    p.add_argument("--images", help="image directory (default <run>/10_frames/images)")
    p.add_argument("--variants", type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()], default=["A", "B", "D"],
                   help="comma list of A,B,D (C runs by itself when needed; see --no-c)")
    p.add_argument("--cache", default="/w/cache/colmap", help="directory with the ONNX models and vocabulary trees")
    p.add_argument("--colmap", default="colmap", help="colmap executable")
    p.add_argument("--camera-model", default="SIMPLE_RADIAL")
    p.add_argument("--max-image-size", type=int, default=1920)
    p.add_argument("--overlap", type=int, default=20, help="sequential matching window (quadratic)")
    p.add_argument("--no-loop", action="store_true", help="turn loop detection off")
    p.add_argument("--cpu", action="store_true", help="no GPU: run ONNX features, matching and global positioning on the CPU (slow)")
    p.add_argument("--no-c", dest="allow_c", action="store_false", help="never run the exhaustive variant C")
    p.add_argument("--sequential", action="store_true", help="run variants one after another, not concurrently")
    p.add_argument("--variant-timeout", type=float, default=2400.0, help="seconds per variant (default 40 min)")
    p.add_argument("--keep-db", choices=["all", "chosen", "none"], default="chosen", help="which databases to keep")
    p.add_argument("--retry-delay", type=float, default=20.0, help="seconds to wait before retrying a failed GPU step once")
    p.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))
