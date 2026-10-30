"""Check the aligned model against a dataset's metric, gravity-aligned camera poses (MuSHRoom, nerfstudio style).

    python -m pipeline.align.metric_check --run <run_dir> --ref <transformations.json> [--up z] [--out F]

Matches our cameras (40_align/sparse/0, names from 10_frames/frames.json `kept[].src`) to the dataset's frames by
file name and fits a similarity ours -> dataset with a trimmed Umeyama (keep the best 70% of residuals, 8 rounds),
because COLMAP and ARKit/Polycam trajectories disagree on short bad segments.

Reports, all as measured on this run:
- scale_error_pct = 100 (1 / s_fit - 1): how far our metres are from the dataset's metres (+ means our model is too big);
- up_error_deg: angle between our +Y (mapped into the dataset frame) and the dataset's up axis (+Z for MuSHRoom);
- h_cap check: our align.json h_cap_m against the median dataset camera height above the dataset's own floor is NOT
  computed (the dataset frame has no floor); use the Faro cloud for that.
The transform_matrix of each frame is camera-to-world, OpenGL camera axes, metres.
Standard library plus numpy; runs in namma-space-gpu or any venv with numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

from . import colmap_io as cio


def umeyama(src: np.ndarray, dst: np.ndarray):
    """s, R, t with dst ~ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var)
    t = md - s * R @ ms
    return s, R, t


def robust_umeyama(src: np.ndarray, dst: np.ndarray, keep: float = 0.7, rounds: int = 8):
    idx = np.arange(len(src))
    for _ in range(rounds):
        s, R, t = umeyama(src[idx], dst[idx])
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        k = max(3, int(round(keep * len(src))))
        idx = np.argsort(res)[:k]
    s, R, t = umeyama(src[idx], dst[idx])
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    return s, R, t, res, idx


def name_map(run: str) -> Dict[str, str]:
    """Our frame name -> original file name (frames mode); identity when unknown."""
    try:
        with open(os.path.join(run, "10_frames", "frames.json")) as f:
            fj = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for k in fj.get("kept", []):
        src = k.get("src")
        if src and k.get("name"):
            out[k["name"]] = os.path.basename(src)
    return out


def check(run: str, ref: str, up_axis: str = "z") -> dict:
    d = os.path.join(run, "40_align", "sparse", "0")
    if not os.path.isfile(os.path.join(d, "images.bin")):
        d = os.path.join(run, "40_align", "sparse")
    m = cio.load_model(d)
    C = m.centers()
    with open(ref) as f:
        tj = json.load(f)
    ref_c = {os.path.basename(fr["file_path"]): np.array(fr["transform_matrix"], float)[:3, 3] for fr in tj["frames"]}
    nm = name_map(run)
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []
    names: List[str] = []
    for i, n in enumerate(m.image_name):
        orig = nm.get(n, n)
        if orig in ref_c:
            src.append(C[i])
            dst.append(ref_c[orig])
            names.append(orig)
    if len(src) < 10:
        raise SystemExit(f"only {len(src)} cameras matched by name; is 10_frames/frames.json present?")
    src_a, dst_a = np.array(src), np.array(dst)
    s, R, t, res, idx = robust_umeyama(src_a, dst_a)
    s_all, R_all, t_all = umeyama(src_a, dst_a)
    up_ref = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}[up_axis]
    our_up_in_ref = R @ np.array([0, 1.0, 0])
    up_err = float(np.degrees(np.arccos(np.clip(our_up_in_ref @ up_ref, -1, 1))))
    our_path = float(np.linalg.norm(np.ptp(src_a, axis=0)))
    return {
        "schema": "namma-space/metric_check@1",
        "ref": ref, "n_ours": int(m.n_images), "n_ref": len(ref_c), "n_matched": len(src), "n_fit": int(len(idx)),
        "s_fit": s, "scale_error_pct": 100.0 * (1.0 / s - 1.0),
        "s_fit_plain": s_all, "scale_error_pct_plain": 100.0 * (1.0 / s_all - 1.0),
        "up_error_deg": up_err,
        "residual_m": {"median": float(np.median(res)), "p90": float(np.percentile(res, 90)),
                       "median_fit_set": float(np.median(res[idx]))},
        "our_path_extent_m": our_path,
        "note": "trimmed Umeyama (70%, 8 rounds) of our aligned camera centres onto the dataset centres; "
                "scale_error_pct = 100 (1/s_fit - 1), + means our model is larger than the dataset's metric frame",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True, help="transformations.json (camera-to-world, OpenGL axes, metres)")
    ap.add_argument("--up", default="z", choices=["x", "y", "z"], help="dataset up axis (MuSHRoom: z)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = check(a.run, a.ref, a.up)
    txt = json.dumps(res, indent=2)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
