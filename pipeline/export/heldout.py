"""Held-out photo/pose pairs in the final frame (F5) and the three.js camera convention (CONTRACTS 2.0).

three.js camera for image i (COLMAP world-to-camera R_i, t_i in the aligned frame):
    centre C = T_final(-R_i^T t_i);  rotation R_i^T . diag(1, -1, -1);  vertical FOV 2 atan(H / 2 f_y);
    principal point through a custom projection matrix.
T_final: X_final = scale * Rf . X + translation  (Rf from `quaternion` [x, y, z, w]).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])


def quat_from_matrix(m: np.ndarray) -> List[float]:
    """Rotation matrix -> quaternion [x, y, z, w] (w >= 0)."""
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


def matrix_from_quat(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def identity_t_final() -> Dict[str, object]:
    return {"scale": 1.0, "quaternion": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, 0.0]}


def t_final_matrix(tf: Dict[str, object]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = float(tf["scale"]) * matrix_from_quat(tf["quaternion"])
    m[:3, 3] = np.asarray(tf["translation"], dtype=np.float64)
    return m


def threejs_pose(c2w: np.ndarray, tf: Dict[str, object]) -> Dict[str, object]:
    """Position and quaternion of the three.js camera in the final frame, from an OpenCV camera-to-world matrix
    in the aligned frame (c2w = [R_i^T | C])."""
    s = float(tf["scale"])
    rf = matrix_from_quat(tf["quaternion"])
    pos = s * rf @ c2w[:3, 3] + np.asarray(tf["translation"], dtype=np.float64)
    rot = rf @ c2w[:3, :3] @ FLIP
    return {"position": [float(v) for v in pos], "quaternion": quat_from_matrix(rot)}


def fov_y_deg(height: float, fy: float) -> float:
    return float(np.degrees(2.0 * np.arctan(height / (2.0 * fy))))


def pick_evenly(items: Sequence, n: int) -> list:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def write_pairs(out_dir: str, cams: Sequence, psnr_sd: Dict[str, float], psnr_trained: Dict[str, float],
                tf: Dict[str, object], max_side: int = 960) -> List[dict]:
    """Write <id>.jpg (longest side max_side) and heldout.json for each camera; returns the scene.json entries."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for c in cams:
        stem = os.path.splitext(os.path.basename(c.name))[0]
        with Image.open(c.image_path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            k = min(1.0, max_side / max(w0, h0))
            if k < 1.0:
                im = im.resize((round(w0 * k), round(h0 * k)), Image.LANCZOS)
            im.save(os.path.join(out_dir, stem + ".jpg"), quality=90)
        pose = threejs_pose(c.c2w, tf)
        entries.append({
            "id": stem, "photo": f"heldout/{stem}.jpg",
            "position": pose["position"], "quaternion": pose["quaternion"],
            "fov_y_deg": fov_y_deg(c.height, c.K[1, 1]),
            "principal_point": [float(c.K[0, 2] / c.width), float(c.K[1, 2] / c.height)],
            "aspect": float(c.width / c.height),
            "psnr": psnr_sd.get(c.name), "psnr_trained": psnr_trained.get(c.name),
        })
    with open(os.path.join(out_dir, "heldout.json"), "w") as f:
        json.dump(entries, f, indent=1)
    return entries


def write_parity(out_dir: str, cams: Sequence, entries: List[dict], render_fn, max_side: int = 960) -> List[dict]:
    """Reference renders for the viewer's H4b test: render_fn(list of Cam) -> list of HxWx3 float arrays made from the
    web file as decoded (what the browser loads), at the same poses as `entries`. Writes <id>.png and parity.json
    ({id, render, position, quaternion, fov_y_deg, principal_point, aspect, ...})."""
    import copy
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    small = []
    for c in cams:
        k = min(1.0, max_side / max(c.width, c.height))
        w, h = round(c.width * k), round(c.height * k)
        c2 = copy.copy(c)
        c2.K = c.K.copy()
        c2.K[0, :] *= w / c.width
        c2.K[1, :] *= h / c.height
        c2.width, c2.height = w, h
        small.append(c2)
    imgs = render_fn(small)
    out = []
    for e, im in zip(entries, imgs):
        Image.fromarray(np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, e["id"] + ".png"))
        row = {k: v for k, v in e.items() if k not in ("photo",)}
        row["render"] = e["id"] + ".png"
        out.append(row)
    with open(os.path.join(out_dir, "parity.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
