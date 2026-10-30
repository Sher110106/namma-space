"""Optional scale cue: MoGe-2 metric depth against SfM depth (FINAL_PLAN 2.4, P1).

Needs the `moge` package and the weights Ruicheng/moge-2-vitl-normal (MIT), so namma-space-gpu:0.2.  Behind `--moge`;
a missing package or weights is not an error: the stage records why and carries on with the camera-height cue.

For 40 frames spread over the capture: take the triangulated keypoints of the frame whose reprojection error is under
1 px and whose triangulation angle (widest pair of cameras in the track) is over 3 degrees; read MoGe's metric depth at
the keypoint; keep points where MoGe says 1.5 to 5 m; the frame's ratio is the median of MoGe depth / SfM depth
(metres per model unit).  The cue is the median over frames; sigma_log is the spread of the log ratios across frames
(1.4826 x MAD), at least 0.08.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import colmap_io as cio
from . import core

N_FRAMES = 40
MIN_TRI_DEG = 3.0
MAX_REPROJ = 1.0
DEPTH_RANGE_M = (1.5, 5.0)
MODEL_ID = "Ruicheng/moge-2-vitl-normal"
MAX_SIDE = 1024


def _load_model(device: str):
    from moge.model.v2 import MoGeModel  # type: ignore
    return MoGeModel.from_pretrained(MODEL_ID).to(device).eval()


def frame_ratio(model: cio.Model, i: int, tracks: Dict[int, np.ndarray], centers: np.ndarray, Rs: np.ndarray,
                depth_m: np.ndarray, scale_xy: float, cam: Dict[str, Any], idx_of: Dict[int, int]) -> Optional[Tuple[float, int]]:
    ob = cio.image_observations(model, i)
    ok = ob["p"] != np.uint64(2 ** 64 - 1)
    ob = ob[ok]
    if len(ob) < 30:
        return None
    ids = ob["p"].astype(np.int64)
    pt_row = {int(v): r for r, v in enumerate(model.pt_id)}
    rows = np.array([pt_row.get(int(p), -1) for p in ids])
    keep = rows >= 0
    ob, rows = ob[keep], rows[keep]
    X = model.xyz[rows]
    xc = X @ Rs[i].T + model.t[i]
    z = xc[:, 2]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    err = np.hypot(fx * xc[:, 0] / z + cx - ob["x"], fy * xc[:, 1] / z + cy - ob["y"])
    good = (z > 0) & (err < MAX_REPROJ)
    # triangulation angle: widest angle between rays to the track's cameras
    ang = np.zeros(len(rows))
    for k in np.nonzero(good)[0]:
        tr = tracks.get(int(model.pt_id[rows[k]]))
        if tr is None or len(tr) < 2:
            good[k] = False
            continue
        cs = np.array([centers[idx_of[int(t)]] for t in tr if int(t) in idx_of])
        if len(cs) < 2:
            good[k] = False
            continue
        d = cs - X[k]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        ang[k] = math.degrees(math.acos(np.clip((d @ d.T).min(), -1, 1)))
    good &= ang > MIN_TRI_DEG
    if good.sum() < 15:
        return None
    u = np.clip((ob["x"][good] * scale_xy).astype(int), 0, depth_m.shape[1] - 1)
    v = np.clip((ob["y"][good] * scale_xy).astype(int), 0, depth_m.shape[0] - 1)
    dm = depth_m[v, u]
    sf = z[good]
    sel = np.isfinite(dm) & (dm >= DEPTH_RANGE_M[0]) & (dm <= DEPTH_RANGE_M[1])
    if sel.sum() < 10:
        return None
    return float(np.median(dm[sel] / sf[sel])), int(sel.sum())


def estimate(run_dir: str, model: cio.Model, log=print, device: Optional[str] = None) -> Tuple[Optional[core.Cue], Dict[str, Any]]:
    info: Dict[str, Any] = {"requested": True, "available": False, "model": MODEL_ID}
    try:
        import torch
        import cv2
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = _load_model(dev)
    except Exception as ex:
        info["error"] = f"{type(ex).__name__}: {ex}"
        log(f"MoGe cue unavailable: {info['error']}")
        return None, info
    info["available"] = True
    cams = {c["id"]: c for c in cio.camera_summary(model.cameras)}
    c0 = cams[int(model.camera_id[0])]
    if c0["model"] == "PINHOLE":
        fx, fy, cx, cy = c0["params"]
    else:
        fx = fy = c0["params"][0]
        cx, cy = c0["params"][1:3]
    cam = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    Rs, C = model.rotations(), model.centers()
    tracks = cio.point_tracks(model)
    idx_of = {int(v): i for i, v in enumerate(model.image_id)}
    pick = np.unique(np.round(np.linspace(0, model.n_images - 1, N_FRAMES)).astype(int))
    ratios: List[float] = []
    per: List[Dict[str, Any]] = []
    imgdir = os.path.join(run_dir, "30_undist", "images")
    for i in pick:
        name = model.image_name[int(i)]
        img = cv2.imread(os.path.join(imgdir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        small = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
        t = torch.tensor(cv2.cvtColor(small, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            out = net.infer(t)
        depth = out["depth"].detach().cpu().numpy()
        mask = out.get("mask")
        if mask is not None:
            depth = np.where(mask.detach().cpu().numpy(), depth, np.nan)
        r = frame_ratio(model, int(i), tracks, C, Rs, depth, s, cam, idx_of)
        if r is not None:
            ratios.append(r[0])
            per.append({"image": name, "ratio": r[0], "n_points": r[1]})
    info["n_frames_used"] = len(ratios)
    if len(ratios) < 8:
        info["error"] = f"only {len(ratios)} usable frames (need 8)"
        log(f"MoGe cue skipped: {info['error']}")
        return None, info
    lr = np.log(np.array(ratios))
    med = float(np.median(lr))
    sig = max(core.SIGMA_MIN_MOGE, float(1.4826 * np.median(np.abs(lr - med))))
    info.update({"frames": per, "ratio_median": math.exp(med), "sigma_log_spread": float(1.4826 * np.median(np.abs(lr - med)))})
    return core.Cue("moge2_depth", math.exp(med), sig, f"median of {len(ratios)} frames, MoGe-2 depth / SfM depth"), info
