"""Time gsplat training at given frame sizes (review fix #4/#10, 30 Oct).

    python -m pipeline.train.bench --data /w/runs/<run>/30_undist --out /w/runs/fix-bench \
        --long-side 1920,1600 --steps 9000

For each long side it runs the pipeline's r0 recipe (C.build_command: MCMC, 1.5 M cap, antialiased,
bilateral grid, scene-unit noise scaling) for --steps steps, one size after the other, and samples the
trainer's step counter every few seconds. MCMC grows the splat count by 5% every 100 steps from step 500,
so the rate at the end of the run (once the cap is reached) is the rate for the rest of a 30k run.
Writes <out>/bench.json. Resized frames go to <out>/data_<L>/images and are deleted afterwards.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

from . import cmd as C
from . import gpu as G
from . import stats as S

LATE_FACTOR = 0.78   # rate after step 9k / rate at 7-9k, from w3-honka r1's log (run alone on tyrone, 30 Oct)


def long_side_of(images_dir: str) -> Tuple[int, int]:
    from PIL import Image
    first = sorted(f for f in os.listdir(images_dir) if not f.startswith("."))[0]
    with Image.open(os.path.join(images_dir, first)) as im:
        return im.size


def splat_count(n0: int, step: int, cap: int) -> int:
    """MCMC splat count after `step` steps from n0 initial points (5% growth every 100 steps after 500)."""
    k = max(0, (step - 1) // 100 - 5) if step > 500 else 0
    return int(min(cap, n0 * 1.05 ** k))


def steps_to_cap(n0: int, cap: int) -> int:
    if n0 >= cap:
        return 0
    return 500 + 100 * math.ceil(math.log(cap / n0) / math.log(1.05))


def project_30k(samples: List[Tuple[float, int]], steps: int, n0: int, cap: int) -> dict:
    """samples: (seconds since start, step). The rate over the last 1500 steps stands for every step after
    the cap is reached; the measured time to the end stands for the steps before it."""
    eval_s = round(samples[-1][0] - max(t for t, s in samples if s < steps - 5), 1) if samples else None
    samples = [(t, s) for t, s in samples if s < steps - 5]   # the last step also runs the eval and the save
    if len(samples) < 3:
        return {}
    t_end, s_end = samples[-1]
    tail = [(t, s) for t, s in samples if s >= s_end - 1500]
    t0, s0 = tail[0]
    rate = (s_end - s0) / max(1e-6, t_end - t0)
    first2k = [(t, s) for t, s in samples if s <= 2000]
    rate_2k = (first2k[-1][1] - first2k[0][1]) / max(1e-6, first2k[-1][0] - first2k[0][0]) if len(first2k) > 2 else None
    cap_step = steps_to_cap(n0, cap)
    out = {"rate_end_it_s": round(rate, 2), "rate_first_2k_it_s": round(rate_2k, 2) if rate_2k else None,
           "ms_per_step_end": round(1000 / rate, 1), "t_to_end_s": round(t_end, 1), "end_step": s_end,
           "cap_reached_at_step": cap_step, "cap_reached_in_run": cap_step <= s_end, "final_eval_save_s": eval_s}
    if cap_step <= s_end:
        # w3-honka r1 (alone, 0.73 MP) slowed from 25.3 it/s at 7-9k to 19.7 it/s after 9k: factor 0.78
        out["projected_30k_min"] = round((t_end + (30000 - s_end) / (rate * LATE_FACTOR) + 2 * (eval_s or 0)) / 60, 1)
    return out


def make_resized(src_dir: str, dst_dir: str, long_side: int) -> None:
    from PIL import Image
    os.makedirs(os.path.join(dst_dir, "images"), exist_ok=True)
    for f in sorted(os.listdir(os.path.join(src_dir, "images"))):
        with Image.open(os.path.join(src_dir, "images", f)) as im:
            w, h = im.size
            s = long_side / max(w, h)
            im.convert("RGB").resize((round(w * s), round(h * s)), Image.LANCZOS).save(
                os.path.join(dst_dir, "images", f), quality=95)
    sp = os.path.join(src_dir, "sparse")
    os.symlink(os.path.abspath(sp), os.path.join(dst_dir, "sparse"))


def run_one(data_dir: str, out_dir: str, steps: int, cap: int) -> dict:
    from pipeline.export import colmap_io
    k = C.scene_unit([im.centre() for im in colmap_io.read_images(data_dir).values()])
    o = C.TrainOpts(data_dir=data_dir, result_dir=out_dir, variant="r0", cap_max=cap, scene_k=k)
    cmd = C.build_command(o)
    i = cmd.index("--max-steps")
    cmd[i + 1] = str(steps)
    for flag in ("--eval-steps", "--save-steps", "--ply-steps"):
        j = cmd.index(flag)
        cmd[j + 1:j + 1 + len(o.eval_steps)] = [str(steps)]
    cmd.remove("--save-ply")
    os.makedirs(out_dir, exist_ok=True)
    logp = os.path.join(out_dir, "log.txt")
    t0 = time.time()
    samples: List[Tuple[float, int]] = []
    peak = 0
    with open(logp, "wb") as lf:
        p = subprocess.Popen(cmd, cwd=C.TRAINER_CWD, stdout=lf, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(3)
            pr = S.progress_from_log(logp)
            if pr and (not samples or pr[0] != samples[-1][1]):
                samples.append((round(time.time() - t0, 2), pr[0]))
            used = G.gpu_used_mb() or 0
            peak = max(peak, used)
    return {"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "cmd": " ".join(cmd), "scene_k": k,
            "samples": samples, "gpu_used_total_peak_mb": peak}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.train.bench")
    ap.add_argument("--data", required=True, help="undistorted COLMAP model dir (images/, sparse/0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-side", default="1920,1600")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--cap-max", type=int, default=1_500_000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    w, h = long_side_of(os.path.join(a.data, "images"))
    from pipeline.export import colmap_io
    n0 = len(colmap_io.read_points3d(a.data)[0])
    res = {"data": a.data, "frame_size": [w, h], "steps": a.steps, "cap_max": a.cap_max, "n_points_init": n0,
           "gpu_used_mb_at_start": G.gpu_used_mb(), "runs": []}
    for L in [int(x) for x in a.long_side.split(",")]:
        if L >= max(w, h):
            data = a.data
        else:
            data = os.path.join(a.out, f"data_{L}")
            shutil.rmtree(data, ignore_errors=True)
            make_resized(a.data, data, L)
        r = run_one(data, os.path.join(a.out, f"r0_{L}"), a.steps, a.cap_max)
        r["long_side"] = min(L, max(w, h))
        r["frame_size"] = list(long_side_of(os.path.join(data, "images")))
        if n0:
            r.update(project_30k(r["samples"], a.steps, n0, a.cap_max))
        res["runs"].append(r)
        if data != a.data:
            shutil.rmtree(data, ignore_errors=True)
        for sub in ("ckpts", "tb", "renders"):
            shutil.rmtree(os.path.join(a.out, f"r0_{L}", sub), ignore_errors=True)
        with open(os.path.join(a.out, "bench.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "samples"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
