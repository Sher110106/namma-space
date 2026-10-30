"""Score the camera-path filter against MuSHRoom ground-truth poses on saved SfM models (review #9, 30 Oct).

    python3 -m pipeline.sfm.sanity_gt --run /w/runs/rh-honka --ref <long_capture/transformations.json> \
        [--old OLD_SANITY.py] [--phone-hold 0.5 --phone-photos DIR] [--bad-m 0.30]

For each variant's raw models (20_sfm/<v>/sparse/<k>, before the filter), it fits a trimmed similarity (best 70% of
cameras, 8 rounds) from our camera centres onto the dataset's ARKit centres, calls a camera wrong when it lands
more than --bad-m from its true place, and counts, for the new filter (and the old one, given --old):
correct cameras dropped and wrong cameras kept. Frames are matched by 10_frames/frames.json kept[].src; for the
phone fixture (one photo held --phone-hold s each), photo = floor(t / hold).
numpy; runs in namma-space-gpu or any python3 with numpy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

from . import colmapio, sanity


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((a ** 2).sum() / len(src)))
    return s, R, md - s * R @ ms


def robust_errors(src, dst, keep=0.7, rounds=8):
    idx = np.arange(len(src))
    for _ in range(rounds):
        s, R, t = umeyama(src[idx], dst[idx])
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        idx = np.argsort(res)[:max(3, int(round(keep * len(src))))]
    s, R, t = umeyama(src[idx], dst[idx])
    return np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)


def load_old(path):
    spec = importlib.util.spec_from_file_location("sanity_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sanity_old"] = mod      # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


def score(run, ref, old_mod=None, hold=0.5, bad_m=0.30, photos=None):
    with open(os.path.join(run, "10_frames", "frames.json")) as f:
        fj = json.load(f)
    kept = fj["kept"]
    names = [k["name"] for k in kept]
    meta = {k["name"]: k for k in kept}
    with open(ref) as f:
        tj = json.load(f)
    ref_c = {os.path.basename(fr["file_path"]): np.array(fr["transform_matrix"], float)[:3, 3] for fr in tj["frames"]}
    ref_names = sorted(os.listdir(photos)) if photos else sorted(ref_c)

    def orig(n):
        k = meta[n]
        src = k.get("src") or ""
        if src in ref_c:
            return src
        if k.get("t") is not None:      # phone fixture: one photo per `hold` seconds
            i = int(k["t"] // hold)
            return ref_names[i] if 0 <= i < len(ref_names) else None
        return None

    index = {n: i for i, n in enumerate(names)}
    out = []
    sfm_dir = os.path.join(run, "20_sfm")
    for v in sorted(os.listdir(sfm_dir)):
        sp = os.path.join(sfm_dir, v, "sparse")
        if not os.path.isdir(sp):
            continue
        for m in colmapio.list_models(sp):
            _, recs = colmapio.read_model(m)
            recs = [r for r in recs if r.name in index and orig(r.name) in ref_c]
            if len(recs) < 20:
                continue
            src = np.array([r.centre for r in recs])
            dst = np.array([ref_c[orig(r.name)] for r in recs])
            err = dict(zip([r.name for r in recs], robust_errors(src, dst)))
            wrong = {n for n, e in err.items() if e > bad_m}
            row = {"run": os.path.basename(run.rstrip("/")), "variant": v, "model": os.path.basename(m),
                   "n": len(recs), "wrong_total": len(wrong)}
            for tag, mod, use_t in (("old", old_mod, False), ("new", sanity, True)):
                if mod is None:
                    continue
                cams = [mod.Cam(index[r.name], r.name, r.centre) for r in recs]
                if use_t:
                    for c in cams:
                        c.t = meta[c.name].get("t")
                res = mod.filter_trajectory(cams)
                dropped = {d["name"] for d in res.dropped}
                row[tag] = {"dropped": len(dropped), "correct_dropped": len(dropped - wrong),
                            "wrong_dropped": len(dropped & wrong), "wrong_kept": len(wrong - dropped),
                            "kept": len(recs) - len(dropped),
                            "wrong_kept_max_m": round(max([err[n] for n in wrong - dropped], default=0.0), 2)}
            out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, action="append")
    ap.add_argument("--ref", required=True, action="append", help="one per --run, in the same order")
    ap.add_argument("--old", help="path to the old sanity.py to compare against")
    ap.add_argument("--phone-hold", type=float, default=0.5)
    ap.add_argument("--phone-photos", help="folder of the photos the phone fixture was built from (their sorted order)")
    ap.add_argument("--bad-m", type=float, default=0.30)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    old = load_old(a.old) if a.old else None
    rows = []
    for run, ref in zip(a.run, a.ref):
        rows += score(run, ref, old, a.phone_hold, a.bad_m, a.phone_photos)
    for r in rows:
        line = f"{r['run']:10s} {r['variant']} sparse/{r['model']} n={r['n']:3d} wrong(>{a.bad_m:.2f} m)={r['wrong_total']:3d}"
        for tag in ("old", "new"):
            if tag in r:
                x = r[tag]
                line += (f" | {tag}: dropped {x['dropped']:3d} (correct {x['correct_dropped']:3d}), "
                         f"wrong kept {x['wrong_kept']:3d}")
        print(line)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
