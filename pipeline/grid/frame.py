"""Frame helpers: T_final parsing, yaw/pitch conventions, align.json reading.

Conventions (CONTRACTS 2.0 and 2.12): metres, +Y up, three.js frame. ``yaw_deg`` turns about +Y,
0 looks along -Z, positive turns left, so the forward vector is (-sin yaw, 0, -cos yaw).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def quat_xyzw_to_rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def parse_tfinal(obj: Any) -> np.ndarray:
    """Parse T_final from export.json into a 4x4 matrix (aligned frame -> final frame).

    Accepts a 4x4 nested list, a flat list of 16 (row-major), a dict with ``matrix``, or a dict
    ``{position, quaternion [x, y, z, w], scale}`` like scene.json's splat transform.
    """
    if obj is None:
        return np.eye(4)
    if isinstance(obj, dict):
        for key in ("matrix", "T", "m", "S"):
            if key in obj:
                return parse_tfinal(obj[key])
        pos = obj.get("position", obj.get("translation", [0, 0, 0]))
        quat = obj.get("quaternion", obj.get("rotation", [0, 0, 0, 1]))
        s = obj.get("scale", 1.0)
        s = float(np.mean(s)) if isinstance(s, (list, tuple)) else float(s)
        m = np.eye(4)
        m[:3, :3] = s * quat_xyzw_to_rot(quat)
        m[:3, 3] = [float(v) for v in pos]
        return m
    a = np.array(obj, dtype=np.float64)
    if a.size == 16:
        return a.reshape(4, 4)
    raise ValueError(f"cannot parse T_final: {obj!r}")


def tfinal_scale(T: np.ndarray) -> float:
    return float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))


def apply_points(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def apply_dirs(T: np.ndarray, d: np.ndarray) -> np.ndarray:
    s = tfinal_scale(T)
    return d @ (T[:3, :3] / s).T


def yaw_from_forward(f: np.ndarray) -> float:
    """Yaw in degrees for a world forward vector; (0, 0, -1) gives 0, (-1, 0, 0) gives 90."""
    return math.degrees(math.atan2(-f[0], -f[2]))


def forward_from_yaw(yaw_deg: float) -> np.ndarray:
    a = math.radians(yaw_deg)
    return np.array([-math.sin(a), 0.0, -math.cos(a)])


def pitch_deg_from_forward(f: np.ndarray) -> np.ndarray:
    """Pitch in degrees (positive looks up) for forward vectors with shape (..., 3)."""
    f = np.asarray(f, dtype=np.float64)
    n = np.linalg.norm(f, axis=-1)
    return np.degrees(np.arcsin(np.clip(f[..., 1] / np.maximum(n, 1e-12), -1, 1)))


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())
