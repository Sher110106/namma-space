"""Walkable grid construction (FINAL_PLAN 2.7). numpy and scipy only.

All lengths that depend on the capture scale are multiples of H_cap (median camera height above the
floor), so a scale error can neither open nor close a door. Metre values assume H_cap = 1.4 m.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from . import raster
from .plyio import Splats


@dataclass
class Params:
    cell: float = 0.05
    floor_band: float = 0.07        # x H_cap: |y - floor| below this counts as floor
    floor_opacity: float = 0.2
    floor_r_min: float = 0.05       # m, splat footprint radius range
    floor_r_max: float = 0.30
    path_floor: float = 0.35        # x H_cap: cells this close to the path are floor
    close_holes: float = 0.30       # m, floor holes up to this size are closed
    obst_lo: float = 0.10           # x H_cap
    obst_hi: float = 1.25           # x H_cap
    obst_opacity: float = 0.5
    obst_min_count: int = 3
    floater_cells: int = 10         # blobs smaller than this that lie on the path are ignored
    floater_path_dist: float = 0.15  # m: "lies on the path"
    inflate: float = 0.18           # x H_cap
    inflate_max: float = 0.30       # m
    leash: float = 1.1              # x H_cap
    floor_reach: float = 2.2        # x H_cap: floor evidence farther than this from the path is dropped
    path_walk: float = 0.10         # m: cells this close to the capture path stay walkable (the phone went there)
    # Eye stand-off (review #39, 30 Oct): the eye stops farther from things at eye level (walls, cupboards, doors)
    # than the body does from low furniture, so a walk into a wall ends on a readable view, not a smear.
    eye_standoff: float = 0.28      # x H_cap (0.39 m at 1.4 m), from opaque splats at or above eye_tall x H_cap
    eye_standoff_max: float = 0.45  # m
    eye_tall: float = 0.75          # x H_cap
    eye_path_keep: float = 0.25     # m: cells this close to the capture path ignore the stand-off (doors stay open)
    eye_radius: float = 0.15        # m
    eye_max_count: int = 2
    path_gap_max: float = 1.0       # m: consecutive cameras farther apart are not joined
    min_path_coverage: float = 0.90
    corridor_radius: float = 0.40   # m: fallback corridor discs
    max_cells: int = 16_000_000
    seed: int = 0


@dataclass
class Layout:
    ox: float
    oz: float
    cell: float
    w: int
    h: int

    def ij(self, x, z):
        return (np.floor((np.asarray(z) - self.oz) / self.cell).astype(np.int64),
                np.floor((np.asarray(x) - self.ox) / self.cell).astype(np.int64))

    def xz(self, i, j):
        """Cell centre: x = ox + (j + 0.5) * cell, z = oz + (i + 0.5) * cell."""
        return (self.ox + (np.asarray(j) + 0.5) * self.cell, self.oz + (np.asarray(i) + 0.5) * self.cell)

    def inside(self, i, j):
        i, j = np.asarray(i), np.asarray(j)
        return (i >= 0) & (i < self.h) & (j >= 0) & (j < self.w)


@dataclass
class GridState:
    layout: Layout
    params: Params
    h_cap: float
    floor_y: float
    path_xz: np.ndarray               # dense path points (K, 2)
    d_path: np.ndarray                # metres to nearest path cell (h, w)
    floor: np.ndarray                 # floor evidence (bool)
    obstacle: np.ndarray              # obstacle cells after floater removal (bool)
    blocked: np.ndarray               # obstacle cells plus inflation within the floor/leash area (bool)
    walkable: np.ndarray              # final walkable mask (bool)
    obst_xz: np.ndarray               # obstacle splat centres (K, 2)
    obst_y: np.ndarray
    info: dict = field(default_factory=dict)

    def to_u8(self) -> np.ndarray:
        g = np.zeros(self.walkable.shape, dtype=np.uint8)
        g[self.floor & ~self.obstacle] = 128
        g[self.walkable] = 255
        return g


def path_points(centres_xz: np.ndarray, names: Optional[list[str]], gap_max: float, step: float) -> np.ndarray:
    """Camera centres in capture order plus points interpolated along short segments between them."""
    if names is not None and len(names) == len(centres_xz):
        order = np.argsort(np.array(names, dtype=object), kind="stable")
        c = centres_xz[order]
    else:
        c = centres_xz
    out = [c]
    if len(c) > 1:
        seg = c[1:] - c[:-1]
        ln = np.linalg.norm(seg, axis=1)
        for k in np.flatnonzero((ln > step) & (ln <= gap_max)):
            n = int(np.ceil(ln[k] / step))
            f = (np.arange(1, n) / n)[:, None]
            out.append(c[k] + f * seg[k])
    return np.concatenate(out, axis=0)


def make_layout(splats: Splats, path_xz: np.ndarray, h_cap: float, p: Params) -> Layout:
    c = p.cell
    sel = (splats.xyz[:, 1] > -0.3) & (splats.xyz[:, 1] < 2.0) & (splats.opacity > 0.3)
    pts = splats.xyz[sel][:, [0, 2]] if sel.sum() > 100 else splats.xyz[:, [0, 2]]
    lo = np.percentile(pts, 0.5, axis=0) if len(pts) else path_xz.min(axis=0)
    hi = np.percentile(pts, 99.5, axis=0) if len(pts) else path_xz.max(axis=0)
    pad = p.leash * h_cap + 0.5
    lo = np.minimum(lo, path_xz.min(axis=0) - pad)
    hi = np.maximum(hi, path_xz.max(axis=0) + pad)
    ox, oz = np.floor(lo[0] / c) * c, np.floor(lo[1] / c) * c
    w, h = int(np.ceil((hi[0] - ox) / c)) + 1, int(np.ceil((hi[1] - oz) / c)) + 1
    if w * h > p.max_cells:     # splat outliers made the box huge: keep the path box plus a margin
        lo, hi = path_xz.min(axis=0) - pad - 2.0, path_xz.max(axis=0) + pad + 2.0
        ox, oz = np.floor(lo[0] / c) * c, np.floor(lo[1] / c) * c
        w, h = int(np.ceil((hi[0] - ox) / c)) + 1, int(np.ceil((hi[1] - oz) / c)) + 1
    return Layout(float(ox), float(oz), float(c), w, h)


def floor_evidence(splats: Splats, lay: Layout, d_path: np.ndarray, h_cap: float, floor_y: float, p: Params) -> np.ndarray:
    c = p.cell
    y = splats.xyz[:, 1]
    m = (np.abs(y - floor_y) <= p.floor_band * h_cap) & (splats.opacity >= p.floor_opacity)
    xyz, sc = splats.xyz[m], splats.scale[m]
    F = np.zeros((lay.h, lay.w), dtype=bool)
    if len(xyz):
        top2 = np.sort(sc, axis=1)[:, 1:]                       # the two largest scales
        r = np.clip(2.0 * np.sqrt(top2[:, 0] * top2[:, 1]), p.floor_r_min, p.floor_r_max)
        rc = np.maximum(1, np.rint(r / c)).astype(np.int64)
        ii, jj = lay.ij(xyz[:, 0], xyz[:, 2])
        ok = lay.inside(ii, jj)
        ii, jj, rc = ii[ok], jj[ok], rc[ok]
        for k in np.unique(rc):
            s = rc == k
            F |= raster.dilate(raster.stamp(ii[s], jj[s], F.shape), float(k))
    F |= d_path <= p.path_floor * h_cap
    F = raster.close(F, p.close_holes / 2.0 / c)
    return F & (d_path <= p.floor_reach * h_cap)      # far-away floor splats say nothing about this room


def obstacle_cells(splats: Splats, lay: Layout, d_path: np.ndarray, h_cap: float, floor_y: float, p: Params):
    """Return (obstacle mask after floater removal, xz of obstacle splats, y of obstacle splats, info)."""
    y = splats.xyz[:, 1]
    sel = (splats.opacity > p.obst_opacity) & (y >= floor_y + p.obst_lo * h_cap) & (y <= floor_y + p.obst_hi * h_cap)
    idx = np.flatnonzero(sel)
    xz = splats.xyz[idx][:, [0, 2]]
    ii, jj = lay.ij(xz[:, 0], xz[:, 1])
    ok = lay.inside(ii, jj)
    idx, xz, ii, jj = idx[ok], xz[ok], ii[ok], jj[ok]
    cnt = np.bincount(ii * lay.w + jj, minlength=lay.h * lay.w).reshape(lay.h, lay.w)
    O = cnt >= p.obst_min_count
    lab, n = raster.label8(O)
    removed_blobs = removed_cells = 0
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        from scipy import ndimage as ndi
        mind = np.asarray(ndi.minimum(d_path, lab, index=np.arange(1, n + 1)))
        drop = (sizes < p.floater_cells) & (mind <= p.floater_path_dist)
        removed_blobs, removed_cells = int(drop.sum()), int(sizes[drop].sum())
        if drop.any():
            O = O & ~np.isin(lab, np.flatnonzero(drop) + 1)
    keep = O[ii, jj]
    return O, xz[keep], splats.xyz[idx[keep], 1], {"obstacle_blobs": int(n - removed_blobs),
                                                      "floater_blobs_ignored": removed_blobs,
                                                      "floater_cells_ignored": removed_cells,
                                                      "obstacle_cells": int(O.sum())}


def eye_violations(splats: Splats, lay: Layout, cand: np.ndarray, h_cap: float, floor_y: float, p: Params) -> np.ndarray:
    """Boolean mask over grid cells in ``cand`` whose eye point has more than eye_max_count opaque centres within eye_radius."""
    out = np.zeros_like(cand, dtype=bool)
    ci, cj = np.nonzero(cand)
    if len(ci) == 0:
        return out
    ey = floor_y + h_cap
    y = splats.xyz[:, 1]
    sel = (splats.opacity > p.obst_opacity) & (np.abs(y - ey) <= p.eye_radius)
    if not sel.any():
        return out
    tree = cKDTree(splats.xyz[sel])
    x, z = lay.xz(ci, cj)
    q = np.stack([x, np.full_like(x, ey), z], axis=1)
    cnt = tree.query_ball_point(q, r=p.eye_radius, return_length=True, workers=-1)
    out[ci, cj] = np.asarray(cnt) > p.eye_max_count
    return out


def measure_floor_offset(splats: Splats, d_path: np.ndarray, lay: Layout, h_cap: float, floor_y: float) -> Optional[float]:
    """Median height of opaque splats near the floor and near the path, relative to ``floor_y`` (None if unsupported)."""
    y = splats.xyz[:, 1]
    m = (np.abs(y - floor_y) <= 0.15 * h_cap) & (splats.opacity > 0.5)
    if m.sum() < 200:
        return None
    ii, jj = lay.ij(splats.xyz[m][:, 0], splats.xyz[m][:, 2])
    ok = lay.inside(ii, jj)
    near = np.zeros(len(ii), dtype=bool)
    near[ok] = d_path[ii[ok], jj[ok]] <= 1.1 * h_cap
    if near.sum() < 200:
        return None
    return float(np.median(y[m][near]) - floor_y)


def build(splats: Splats, cam_centres: np.ndarray, cam_names: Optional[list[str]], h_cap: float,
          p: Params, floor_y: float = 0.0, mode: str = "grid") -> GridState:
    t0 = time.time()
    info: dict = {}
    path_xz = path_points(cam_centres[:, [0, 2]], cam_names, p.path_gap_max, p.cell)
    lay = make_layout(splats, path_xz, h_cap, p)
    shape = (lay.h, lay.w)
    pi, pj = lay.ij(path_xz[:, 0], path_xz[:, 1])
    ok = lay.inside(pi, pj)
    path_mask = raster.stamp(pi[ok], pj[ok], shape)
    d_path = raster.dist_to(path_mask, p.cell)
    info["t_layout_s"] = round(time.time() - t0, 3)

    t1 = time.time()
    F = floor_evidence(splats, lay, d_path, h_cap, floor_y, p)
    info["t_floor_s"] = round(time.time() - t1, 3)

    t1 = time.time()
    O, oxz, oy, oinfo = obstacle_cells(splats, lay, d_path, h_cap, floor_y, p)
    info.update(oinfo)
    info["t_obstacles_s"] = round(time.time() - t1, 3)

    t1 = time.time()
    leash = d_path <= p.leash * h_cap
    area = F & leash
    r_eff = max(0.0, min(p.inflate * h_cap, p.inflate_max) - 0.5 * p.cell)
    info["inflate_radius_m"] = round(r_eff, 4)
    near_obst = np.zeros(shape, dtype=bool)
    if len(oxz):
        tree = cKDTree(oxz)
        ci, cj = np.nonzero(area)
        x, z = lay.xz(ci, cj)
        d, _ = tree.query(np.stack([x, z], axis=1), k=1, distance_upper_bound=r_eff, workers=-1)
        near_obst[ci, cj] = np.isfinite(d)
    near_obst |= O
    blocked = near_obst
    # the capture path itself stays walkable: the phone and its holder were there. The eye test below still applies.
    walk = (area & ~blocked) | (F & (d_path <= p.path_walk))
    if mode == "grid" and len(oxz) and p.eye_standoff > 0:
        tall = oy >= floor_y + p.eye_tall * h_cap
        r_eye = max(r_eff, min(p.eye_standoff * h_cap, p.eye_standoff_max) - 0.5 * p.cell)
        info["eye_standoff_m"] = round(r_eye, 4)
        if tall.any() and r_eye > r_eff:
            ci, cj = np.nonzero(walk & (d_path > p.eye_path_keep))
            x, z = lay.xz(ci, cj)
            d, _ = cKDTree(oxz[tall]).query(np.stack([x, z], axis=1), k=1, distance_upper_bound=r_eye, workers=-1)
            near = np.isfinite(d)
            walk[ci[near], cj[near]] = False
            info["eye_standoff_cells_removed"] = int(near.sum())
    info["t_inflate_s"] = round(time.time() - t1, 3)

    if mode == "corridor":
        walk = raster.dilate(path_mask, p.corridor_radius / p.cell)
        F = F | walk
    t1 = time.time()
    viol = eye_violations(splats, lay, walk, h_cap, floor_y, p)
    info["eye_cells_removed"] = int(viol.sum())
    walk = walk & ~viol
    info["t_eye_s"] = round(time.time() - t1, 3)

    # keep the component that holds most capture-path cells (the spawn is chosen inside it later)
    lab, n = raster.label4(walk)
    if n:
        ci, cj = lay.ij(cam_centres[:, 0], cam_centres[:, 2])
        okc = lay.inside(ci, cj)
        lbl = lab[ci[okc], cj[okc]]
        lbl = lbl[lbl > 0]
        if len(lbl):
            best = int(np.bincount(lbl).argmax())
        else:
            best = int(np.bincount(lab.ravel(), minlength=n + 1)[1:].argmax()) + 1
        info["components_before"] = int(n)
        walk = lab == best
    else:
        info["components_before"] = 0
    return GridState(lay, p, h_cap, floor_y, path_xz, d_path, F, O, blocked, walk, oxz, oy, info)
