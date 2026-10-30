"""Room extents: wall-to-wall length and width along the main wall axes (FINAL_PLAN 2.7).

Walls are found from the obstacle cells (cells holding at least three opaque splat centres between 0.1 and 1.25 x H_cap),
which counts wall length, not splat density. Along each main axis a wall is a column of obstacle cells that is long
compared with the camera path. The room runs between the outermost walls on each axis, so a table inside the path
loop, a closet behind the door or a sofa against a wall does not change it.
"""
from __future__ import annotations

import numpy as np


def _hist(u: np.ndarray, bin_m: float):
    lo = np.floor(u.min() / bin_m) * bin_m
    n = int(np.ceil((u.max() - lo) / bin_m)) + 2
    h = np.bincount(np.floor((u - lo) / bin_m).astype(np.int64), minlength=n).astype(np.float64)
    return h, lo


def _score(x: np.ndarray, z: np.ndarray, theta: float, bin_m: float) -> float:
    c, s = np.cos(theta), np.sin(theta)
    hu, _ = _hist(x * c + z * s, bin_m)
    hv, _ = _hist(-x * s + z * c, bin_m)
    return float((hu ** 2).sum() + (hv ** 2).sum())


def main_axis_deg(xz: np.ndarray, bin_m: float = 0.03, seed: int = 0) -> float:
    """Wall direction (degrees, in [-45, 45)) that makes wall splats pile into the fewest bins."""
    if len(xz) > 80_000:
        xz = xz[np.random.default_rng(seed).choice(len(xz), 80_000, replace=False)]
    x, z = xz[:, 0], xz[:, 1]
    coarse = np.arange(0.0, 90.0, 1.0)
    sc = [_score(x, z, np.radians(a), bin_m) for a in coarse]
    a0 = coarse[int(np.argmax(sc))]
    fine = np.arange(a0 - 1.0, a0 + 1.0001, 0.1)
    sf = [_score(x, z, np.radians(a), bin_m) for a in fine]
    a = float(fine[int(np.argmax(sf))])
    return ((a + 45.0) % 90.0) - 45.0      # the angle is only defined modulo 90


def _outer_wall(cell_u: np.ndarray, cell_v: np.ndarray, splat_u: np.ndarray, p_lo: float, p_hi: float,
                perp_range: float, cell: float, zone_m: float, side: str):
    """Position (along u) of the outermost wall on one side, or None. ``side`` is "lo" or "hi".

    A column band (+-1 cell across u) is a wall when the distinct cells it holds along v add up to a length of at least
    a fraction of the camera path's span along v (60%, then 40%, then 25%) and 1.5 m.
    """
    sel = (cell_u >= p_lo - zone_m) & (cell_u <= p_hi + zone_m)
    if sel.sum() < 10:
        return None
    cu, cv = cell_u[sel], cell_v[sel]
    u0, v0 = np.floor(cu.min() / cell) * cell, np.floor(cv.min() / cell) * cell
    ku = np.floor((cu - u0) / cell).astype(np.int64)
    kv = np.floor((cv - v0) / cell).astype(np.int64)
    occ = np.zeros((ku.max() + 3, kv.max() + 1), dtype=bool)
    occ[ku + 1, kv] = True                                   # one blank column on each side
    band = occ[:-2] | occ[1:-1] | occ[2:]
    length = band.sum(axis=1) * cell                          # metres of wall in the band centred on column k+1
    for frac in (0.6, 0.4, 0.25):
        need = max(1.5, frac * perp_range)
        ok = np.flatnonzero(length >= need)
        if len(ok) == 0:
            continue
        k = int(ok[0] if side == "lo" else ok[-1])
        w = np.arange(max(0, k - 2), min(len(length), k + 3))
        k = int(w[np.argmax(length[w])])
        uc = u0 + (k + 0.5) * cell                            # band centre: column k+1 of occ is cell column k
        near = np.abs(splat_u - uc) <= 0.08
        return float(splat_u[near].mean()) if near.sum() >= 5 else float(uc)
    return None


def room_extent(obst_cell_xz: np.ndarray, obst_splat_xz: np.ndarray, cam_xz: np.ndarray, cell: float = 0.05,
                zone_m: float = 3.0, seed: int = 0) -> dict | None:
    """Return {"axes_yaw_deg", "extent_axes_m": [u, v], "room_extent_m": [length, width]} or None.

    ``obst_cell_xz`` are centres of obstacle cells, ``obst_splat_xz`` centres of the splats in them, ``cam_xz`` camera
    centres. Positions are in the grid frame; the two axes are the main wall directions (angle in [-45, 45) degrees).
    """
    if len(obst_cell_xz) < 40 or len(obst_splat_xz) < 200 or len(cam_xz) < 2:
        return None
    a = main_axis_deg(obst_splat_xz, seed=seed)
    th = np.radians(a)
    c, s = np.cos(th), np.sin(th)

    def proj(p):
        return p[:, 0] * c + p[:, 1] * s, -p[:, 0] * s + p[:, 1] * c

    cu, cv = proj(obst_cell_xz)
    su, sv = proj(obst_splat_xz)
    pu, pv = proj(cam_xz)
    ext = []
    walls = []
    for cell_a, cell_b, splat_a, pa, perp in ((cu, cv, su, pu, pv), (cv, cu, sv, pv, pu)):
        rng = float(perp.max() - perp.min())
        lo = _outer_wall(cell_a, cell_b, splat_a, float(pa.min()), float(pa.max()), rng, cell, zone_m, "lo")
        hi = _outer_wall(cell_a, cell_b, splat_a, float(pa.min()), float(pa.max()), rng, cell, zone_m, "hi")
        if lo is None or hi is None or hi <= lo:
            return None
        ext.append(hi - lo)
        walls.append([round(lo, 4), round(hi, 4)])
    return {"axes_yaw_deg": round(float(a), 2), "extent_axes_m": [round(ext[0], 3), round(ext[1], 3)],
            "room_extent_m": [round(max(ext), 3), round(min(ext), 3)],
            "walls_uv": walls}   # outer wall positions along the two axes (u = x c + z s, v = -x s + z c)


def outside_walls(x: np.ndarray, z: np.ndarray, ext: dict, margin: float) -> np.ndarray:
    """True where (x, z) lies more than ``margin`` beyond the outer walls that room_extent found."""
    th = np.radians(ext["axes_yaw_deg"])
    c, s = np.cos(th), np.sin(th)
    u, v = x * c + z * s, -x * s + z * c
    (ulo, uhi), (vlo, vhi) = ext["walls_uv"]
    return (u < ulo - margin) | (u > uhi + margin) | (v < vlo - margin) | (v > vhi + margin)
