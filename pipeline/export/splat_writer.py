"""Own numpy writer/reader for the antimatter15 `.splat` layout (SH0 only, 32 bytes per splat).

Record: position 3 x float32, scale 3 x float32 (linear, metres), colour RGBA 4 x uint8
(RGB = 0.5 + C0 * f_dc, A = sigmoid(opacity)), rotation 4 x uint8 (w x y z, q * 128 + 128).
Splats are sorted by decreasing size * opacity, as the reference converter does.
This is the fallback web format when splat-transform is not available; Spark reads it.
"""
from __future__ import annotations

import numpy as np

from .ply_io import Splats, sigmoid

SH_C0 = 0.28209479177387814
RECORD = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
assert RECORD.itemsize == 32


def pack_splat(s: Splats, sort: bool = True) -> np.ndarray:
    n = len(s)
    rec = np.zeros(n, dtype=RECORD)
    rec["pos"] = s.means
    lin = np.exp(np.clip(s.scale.astype(np.float64), -30, 10))
    rec["scale"] = lin
    rgb = np.clip((0.5 + SH_C0 * s.f_dc.astype(np.float64)) * 255.0, 0, 255)
    alpha = sigmoid(s.opacity) * 255.0
    rec["rgba"][:, :3] = np.rint(rgb).astype(np.uint8)
    rec["rgba"][:, 3] = np.rint(np.clip(alpha, 0, 255)).astype(np.uint8)
    q = s.quat.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rec["rot"] = np.clip(np.rint(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
    if sort:
        key = -(lin.prod(axis=1)) * sigmoid(s.opacity)
        rec = rec[np.argsort(key, kind="stable")]
    return rec


def write_splat(path: str, s: Splats, sort: bool = True) -> int:
    rec = pack_splat(s, sort)
    with open(path, "wb") as f:
        f.write(rec.tobytes())
    return int(rec.shape[0])


def read_splat(path: str) -> Splats:
    rec = np.fromfile(path, dtype=RECORD)
    n = rec.shape[0]
    rgb = rec["rgba"][:, :3].astype(np.float64) / 255.0
    f_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    a = np.clip(rec["rgba"][:, 3].astype(np.float64) / 255.0, 1e-4, 1 - 1e-4)
    q = (rec["rot"].astype(np.float64) - 128.0) / 128.0
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return Splats(
        means=np.ascontiguousarray(rec["pos"]).astype(np.float32),
        f_dc=f_dc, f_rest=np.zeros((n, 0), np.float32),
        opacity=np.log(a / (1 - a)).astype(np.float32),
        scale=np.log(np.maximum(rec["scale"].astype(np.float64), 1e-9)).astype(np.float32),
        quat=q.astype(np.float32))
