"""Synthetic splat room for the grid tests (G3, G4, G5, G6) and a writer for a full fake run directory.

Room: 4 m (x) by 5 m (z), floor at y = 0, ceiling at 2.6 m, placed at (X0, Z0) so nothing lines up with the
grid by accident. A 0.9 m door in the far wall (z = Z0 + 5) leads to a 1.4 m x 1.2 m closet. A table
(1.2 x 0.8 m, top at 0.75 m, four legs), a pillar (0.6 x 0.3 m, 1.0 m tall, magenta), three floaters on the capture
path and a 0.3 m square hole in the left wall at 1 m height. Outside the hole there is floor, so a wall that
vanished in 2D would open a passage.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.grid import colmapio, plyio  # noqa: E402

X0, Z0 = -1.30, 0.90          # south-west corner of the room (x, z)
LX, LZ = 4.0, 5.0
CEIL = 2.6
DOOR_X = X0 + 2.8             # door centre along x, in the wall z = Z0 + LZ
DOOR_W = 0.9
CLOSET_D = 1.2                # closet depth beyond the door
CLOSET_HALF = 0.7
TABLE = (X0 + 0.9, X0 + 2.1, Z0 + 1.5, Z0 + 2.3)      # x0, x1, z0, z1
PILLAR = (X0 + 2.4, X0 + 3.0, Z0 + 2.0, Z0 + 2.3)
HOLE = dict(z=(Z0 + 3.5, Z0 + 3.8), y=(0.85, 1.15))   # in the wall x = X0
H_CAP = 1.40

WAYPOINTS = [(2.8, 5.5), (2.8, 4.2), (3.4, 3.5), (3.4, 1.0), (2.4, 0.6), (0.4, 0.6), (0.45, 1.9), (0.45, 3.4),
             (1.6, 3.3), (2.8, 4.0)]
FLOATERS = [((3.4, 2.0), 0.60), ((2.8, 4.6), 1.00), ((0.45, 2.5), 1.65)]     # (x, z) relative to (X0, Z0), height


@dataclass
class Rect:
    axis: int                 # axis that is constant: 0 = x, 1 = y, 2 = z
    value: float
    r1: tuple                 # range along the first free axis (x = 0: y; y = 1: x; z = 2: y)
    r2: tuple                 # range along the second free axis (x = 0: z; y = 1: z; z = 2: x)
    rgb: tuple
    holes: list = field(default_factory=list)   # (a1_lo, a1_hi, a2_lo, a2_hi) in free-axis order

    @property
    def area(self) -> float:
        return (self.r1[1] - self.r1[0]) * (self.r2[1] - self.r2[0])


def surfaces() -> list[Rect]:
    xa, xb, za, zb = X0, X0 + LX, Z0, Z0 + LZ
    dx0, dx1 = DOOR_X - DOOR_W / 2, DOOR_X + DOOR_W / 2
    wall, floor_c, ceil_c, table_c, pillar_c = (0.86, 0.82, 0.70), (0.55, 0.45, 0.35), (0.95, 0.95, 0.95), (0.45, 0.30, 0.15), (1.0, 0.0, 1.0)
    cx0, cx1 = DOOR_X - CLOSET_HALF, DOOR_X + CLOSET_HALF
    s = [
        Rect(1, 0.0, (xa, xb), (za, zb), floor_c),
        Rect(1, 0.0, (cx0, cx1), (zb, zb + CLOSET_D), floor_c),
        Rect(1, CEIL, (xa, xb), (za, zb), ceil_c),
        Rect(0, xa, (0, CEIL), (za, zb), wall, holes=[(HOLE["y"][0], HOLE["y"][1], HOLE["z"][0], HOLE["z"][1])]),
        Rect(0, xb, (0, CEIL), (za, zb), wall),
        Rect(2, za, (0, CEIL), (xa, xb), wall),
        Rect(2, zb, (0, CEIL), (xa, dx0), wall),            # far wall, left of the door
        Rect(2, zb, (0, CEIL), (dx1, xb), wall),            # far wall, right of the door
        Rect(2, zb, (2.0, CEIL), (dx0, dx1), wall),         # door header
        # closet: jambs, side walls, back wall
        Rect(0, dx0, (0, 2.0), (zb, zb + 0.12), wall), Rect(0, dx1, (0, 2.0), (zb, zb + 0.12), wall),
        Rect(0, cx0, (0, CEIL), (zb + 0.12, zb + CLOSET_D), wall), Rect(0, cx1, (0, CEIL), (zb + 0.12, zb + CLOSET_D), wall),
        Rect(2, zb + CLOSET_D, (0, CEIL), (cx0, cx1), wall),
        # outside the hole in the left wall: a strip of floor (a "terrace")
        Rect(1, 0.0, (xa - 1.5, xa), (Z0 + 2.8, Z0 + 4.4), floor_c),
        # table
        Rect(1, 0.75, (TABLE[0], TABLE[1]), (TABLE[2], TABLE[3]), table_c),
        # pillar (top and four sides), magenta
        Rect(1, 1.0, (PILLAR[0], PILLAR[1]), (PILLAR[2], PILLAR[3]), pillar_c),
        Rect(0, PILLAR[0], (0, 1.0), (PILLAR[2], PILLAR[3]), pillar_c), Rect(0, PILLAR[1], (0, 1.0), (PILLAR[2], PILLAR[3]), pillar_c),
        Rect(2, PILLAR[2], (0, 1.0), (PILLAR[0], PILLAR[1]), pillar_c), Rect(2, PILLAR[3], (0, 1.0), (PILLAR[0], PILLAR[1]), pillar_c),
    ]
    for lx in (TABLE[0] + 0.03, TABLE[1] - 0.03):
        for lz in (TABLE[2] + 0.03, TABLE[3] - 0.03):
            for dx, dz in ((-0.025, 0), (0.025, 0)):
                s.append(Rect(0, lx + dx, (0, 0.75), (lz - 0.025, lz + 0.025), table_c))
            for dz2 in (-0.025, 0.025):
                s.append(Rect(2, lz + dz2, (0, 0.75), (lx - 0.025, lx + 0.025), table_c))
    return s


def sample(n_total: int, seed: int = 0, noise: float = 0.005):
    """Return (xyz, rgb, opacity, scale) for about n_total splats on the surfaces, plus the floaters."""
    rng = np.random.default_rng(seed)
    surf = surfaces()
    area = np.array([r.area for r in surf])
    cnt = np.maximum(1, np.rint(n_total * area / area.sum()).astype(int))
    xyz, rgb = [], []
    for r, n in zip(surf, cnt):
        a1 = rng.uniform(r.r1[0], r.r1[1], n)
        a2 = rng.uniform(r.r2[0], r.r2[1], n)
        if r.holes:
            keep = np.ones(n, dtype=bool)
            for h in r.holes:
                keep &= ~((a1 >= h[0]) & (a1 <= h[1]) & (a2 >= h[2]) & (a2 <= h[3]))
            a1, a2 = a1[keep], a2[keep]
            n = len(a1)
        c = r.value + rng.normal(0, noise, n)
        cols = [None, None, None]
        free = {0: [1, 2], 1: [0, 2], 2: [1, 0]}[r.axis]      # free axes in the order r1, r2: (y, z), (x, z), (y, x)
        cols[r.axis], cols[free[0]], cols[free[1]] = c, a1, a2
        xyz.append(np.stack(cols, axis=1))
        rgb.append(np.tile(np.array(r.rgb) + rng.normal(0, 0.02, (n, 3)), (1, 1)))
    xyz, rgb = np.concatenate(xyz), np.clip(np.concatenate(rgb), 0, 1)
    for (fx, fz), fy in FLOATERS:
        c = np.array([X0 + fx, fy, Z0 + fz]) + rng.normal(0, 0.02, (12, 3))
        xyz = np.concatenate([xyz, c])
        rgb = np.concatenate([rgb, np.tile([0.3, 0.9, 0.3], (12, 1))])
    n = len(xyz)
    opacity = np.full(n, 0.9)
    scale = np.tile([0.03, 0.03, 0.004], (n, 1))
    return xyz.astype(np.float32), rgb.astype(np.float32), opacity.astype(np.float32), scale.astype(np.float32)


def camera_path(step: float = 0.12, pitch_up_every: int = 10):
    """Camera centres (x, y, z), forward vectors, names for the loop path at eye height H_CAP."""
    pts = np.array([(X0 + a, Z0 + b) for a, b in WAYPOINTS])
    cen, fwd = [], []
    for p, q in zip(pts[:-1], pts[1:]):
        d = q - p
        ln = np.linalg.norm(d)
        n = max(1, int(ln / step))
        for k in range(n):
            cen.append(p + d * (k / n))
            fwd.append(d / ln)
    cen.append(pts[-1])
    fwd.append(fwd[-1])
    cen, fwd = np.array(cen), np.array(fwd)
    c3 = np.stack([cen[:, 0], np.full(len(cen), H_CAP), cen[:, 1]], axis=1)
    f3 = np.stack([fwd[:, 0], np.zeros(len(cen)), fwd[:, 1]], axis=1)
    for k in range(0, len(f3), pitch_up_every):       # some frames tilt up 35 degrees
        f3[k] = [f3[k, 0] * np.cos(np.radians(35)), np.sin(np.radians(35)), f3[k, 2] * np.cos(np.radians(35))]
    names = [f"{k + 1:06d}.jpg" for k in range(len(c3))]
    return c3, f3, names


def w2c_from_forward(c: np.ndarray, f: np.ndarray):
    """OpenCV-style world-to-camera (R, t) for centre c and forward f in a y-up world."""
    f = f / np.linalg.norm(f)
    right = np.cross(f, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    c2w = np.stack([right, down, f], axis=1)
    R = c2w.T
    return R, -R @ c


def make_model(n_points: int = 4000, seed: int = 1, tilt_deg: float = 0.0):
    """COLMAP-style model for the camera path: poses, SfM points on the surfaces, and which camera sees which point."""
    rng = np.random.default_rng(seed)
    xyz, _, _, _ = sample(n_points, seed=seed + 7, noise=0.0)
    keep = (xyz[:, 1] > 0.01) | (rng.random(len(xyz)) < 0.3)
    pts = xyz[keep].astype(np.float64)
    c3, f3, names = camera_path()
    R_l, t_l, ids_l = [], [], []
    for c, f in zip(c3, f3):
        R, t = w2c_from_forward(c, f)
        v = pts - c
        dist = np.linalg.norm(v, axis=1)
        ang = np.degrees(np.arccos(np.clip(v @ f / np.maximum(dist, 1e-9), -1, 1)))
        seen = (ang < 35) & (dist < 9.0) & (dist > 0.3)
        R_l.append(R)
        t_l.append(t)
        ids_l.append(np.flatnonzero(seen).astype(np.int64) + 1)
    raw = [(k + 1, 1, np.zeros((len(ids), 2)), ids) for k, ids in enumerate(ids_l)]
    pt_ids = np.arange(1, len(pts) + 1, dtype=np.int64)
    return colmapio.Model(names, np.array(R_l), np.array(t_l), ids_l, pt_ids, pts, raw, {})


def write_run(run: Path, n_splats: int = 150_000, seed: int = 0, sh_rest: int = 0, with_tfinal=None, sharp=True) -> dict:
    """Write a fake run directory that the stage can consume. Returns the truth dict."""
    xyz, rgb, op, sc = sample(n_splats, seed=seed)
    model = make_model()
    (run / "40_align" / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (run / "60_export").mkdir(parents=True, exist_ok=True)
    (run / "10_frames").mkdir(parents=True, exist_ok=True)
    colmapio.write_model(run / "40_align" / "sparse" / "0", model)
    (run / "40_align" / "align.json").write_text(json.dumps({"h_cap": H_CAP, "up_resid_deg": 0.4,
                                                           "floor": {"support": 0.4, "rms": 0.01}}))
    plyio.write_ply(run / "60_export" / "clean.ply", xyz, op, sc, rgb, sh_rest=sh_rest)
    export = {"removed": {}}
    if with_tfinal is not None:
        export["T_final"] = with_tfinal
    (run / "60_export" / "export.json").write_text(json.dumps(export))
    if sharp:
        rng = np.random.default_rng(3)
        (run / "10_frames" / "frames.json").write_text(json.dumps(
            {"kept": [{"name": n, "t": k / 8, "sharp": float(rng.uniform(50, 150))} for k, n in enumerate(model.names)]}))
    return truth()


def truth() -> dict:
    return {"room": (X0, X0 + LX, Z0, Z0 + LZ), "door_x": DOOR_X, "door_w": DOOR_W, "wall_z": Z0 + LZ,
            "closet": (DOOR_X - CLOSET_HALF, DOOR_X + CLOSET_HALF, Z0 + LZ, Z0 + LZ + CLOSET_D),
            "table": TABLE, "pillar": PILLAR, "floaters": [((X0 + a, Z0 + b), y) for (a, b), y in FLOATERS],
            "h_cap": H_CAP, "hole": HOLE}
