"""Spawn, bookmarks and tour (FINAL_PLAN 2.7). numpy and scipy only."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .frame import yaw_from_forward
from .walkable import Layout


@dataclass
class Cams:
    names: list[str]
    centres: np.ndarray               # (N, 3) final frame
    forward: np.ndarray               # (N, 3) unit, final frame
    pitch: np.ndarray                 # (N,) degrees, positive looks up
    sharp: Optional[np.ndarray]       # (N,) or None
    n_pts: np.ndarray                 # (N,) SfM points seen
    med_depth: np.ndarray             # (N,) metres, NaN when no points
    order: np.ndarray                 # (N,) capture rank (position in the name-sorted list)


def camera_stats(pids: list[np.ndarray], pt_ids: np.ndarray, pt_xyz: np.ndarray, R: np.ndarray, t: np.ndarray,
                 depth_scale: float = 1.0):
    """Per image: number of observed SfM points and the median depth of those points (aligned frame, times depth_scale)."""
    n = len(pids)
    cnt = np.zeros(n, dtype=np.int64)
    med = np.full(n, np.nan)
    if len(pt_ids) == 0:
        return cnt, med
    srt = np.argsort(pt_ids)
    sid = pt_ids[srt]
    for k in range(n):
        ids = pids[k]
        if len(ids) == 0:
            continue
        pos = np.searchsorted(sid, ids)
        pos = np.clip(pos, 0, len(sid) - 1)
        ok = sid[pos] == ids
        if not ok.any():
            continue
        X = pt_xyz[srt[pos[ok]]]
        z = X @ R[k][2] + t[k][2]
        cnt[k] = int(ok.sum())
        med[k] = float(np.median(z)) * depth_scale
    return cnt, med


def kmeans(x: np.ndarray, k: int, seed: int = 0, n_init: int = 8, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd k-means with k-means++ seeding. Returns (centroids, labels)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    k = min(k, n)
    best = (np.inf, None, None)
    for _ in range(n_init):
        c = [x[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            tot = d2.sum()
            c.append(x[rng.choice(n, p=d2 / tot)] if tot > 0 else x[rng.integers(n)])
        c = np.array(c, dtype=np.float64)
        lab = np.zeros(n, dtype=np.int64)
        for _ in range(iters):
            d = ((x[:, None, :] - c[None]) ** 2).sum(-1)
            new = d.argmin(axis=1)
            for j in range(k):
                if (new == j).any():
                    c[j] = x[new == j].mean(axis=0)
            if (new == lab).all():
                break
            lab = new
        inertia = ((x - c[lab]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, c.copy(), lab.copy())
    return best[1], best[2]


def _view(cams: Cams, k: int, h_cap: float, floor_y: float, name: Optional[str] = None, vid: Optional[str] = None) -> dict:
    c = cams.centres[k]
    d = {"position": [round(float(c[0]), 4), round(floor_y + h_cap, 4), round(float(c[2]), 4)],
         "yaw_deg": round(yaw_from_forward(cams.forward[k]), 2), "pitch_deg": 0.0}
    if vid is not None:
        d = {"id": vid, "name": name, **d}
    return d


def cell_walkable(lay: Layout, walk: np.ndarray, xz: np.ndarray) -> np.ndarray:
    i, j = lay.ij(xz[:, 0], xz[:, 1])
    ok = lay.inside(i, j)
    out = np.zeros(len(xz), dtype=bool)
    out[ok] = walk[i[ok], j[ok]]
    return out


def choose_spawn(cams: Cams, lay: Layout, walk: np.ndarray, pitch_tol: float = 15.0, min_depth: float = 2.0):
    """Index of the spawn camera and a list of the relaxations used (empty when the plain rule held)."""
    on = cell_walkable(lay, walk, cams.centres[:, [0, 2]])
    level = np.abs(cams.pitch) <= pitch_tol
    sharp_ok = np.ones(len(on), dtype=bool)
    if cams.sharp is not None:
        sharp_ok = cams.sharp > np.median(cams.sharp)
    relaxed: list[str] = []
    cand = on & level & sharp_ok
    if not cand.any():
        cand = on & level
        relaxed.append("sharpness filter dropped")
    if not cand.any():
        cand = on & (np.abs(cams.pitch) <= 25.0)
        relaxed.append("pitch tolerance 25 deg")
    if not cand.any():
        cand = on.copy()
        relaxed.append("any walkable camera")
    if not cand.any():
        return None, relaxed + ["no camera on a walkable cell"]
    idx = np.flatnonzero(cand)
    depth = np.nan_to_num(cams.med_depth[idx], nan=0.0)
    deep = depth >= min_depth
    if deep.any():
        idx = idx[deep]
        best = idx[np.argmax(cams.n_pts[idx])]
    else:
        relaxed.append(f"no candidate with median depth >= {min_depth} m")
        best = idx[np.argmax(depth)]
    return int(best), relaxed


def choose_bookmarks(cams: Cams, lay: Layout, walk: np.ndarray, h_cap: float, floor_y: float, k: int = 5,
                     level_tol: float = 20.0, seed: int = 0, exclude: tuple[int, ...] = ()) -> tuple[list[dict], list[int]]:
    on = cell_walkable(lay, walk, cams.centres[:, [0, 2]])
    ok = on & (np.abs(cams.pitch) <= level_tol)
    if len(exclude):
        ok[list(exclude)] = False
    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        return [], []
    xz = cams.centres[:, [0, 2]]
    k = min(k, len(cand))
    cent, lab = kmeans(xz, k, seed=seed)
    used: set[int] = set()
    picks: list[tuple[float, int]] = []
    for j in range(len(cent)):
        d = np.linalg.norm(xz[cand] - cent[j], axis=1)
        for q in np.argsort(d):
            cam = int(cand[q])
            if cam not in used:
                used.add(cam)
                members = cams.order[lab == j]
                picks.append((float(members.mean()) if len(members) else float(cams.order[cam]), cam))
                break
    picks.sort()
    idxs = [c for _, c in picks]
    return [_view(cams, c, h_cap, floor_y, f"View {n + 1}", f"v{n + 1}") for n, c in enumerate(idxs)], idxs


# ---------------------------------------------------------------- tour

_NB = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))]


def _graph(walk: np.ndarray):
    h, w = walk.shape
    idx = -np.ones(walk.shape, dtype=np.int64)
    ci, cj = np.nonzero(walk)
    idx[ci, cj] = np.arange(len(ci))
    a_l, b_l, w_l = [], [], []
    for di, dj, wt in _NB:
        i2, j2 = ci + di, cj + dj
        ok = (i2 >= 0) & (i2 < h) & (j2 >= 0) & (j2 < w)
        ok[ok] = walk[i2[ok], j2[ok]]
        if di != 0 and dj != 0:           # no corner cutting
            ok &= walk[ci, np.clip(cj + dj, 0, w - 1)] & walk[np.clip(ci + di, 0, h - 1), cj]
        a_l.append(idx[ci[ok], cj[ok]])
        b_l.append(idx[i2[ok], j2[ok]])
        w_l.append(np.full(int(ok.sum()), wt))
    a, b, wt = np.concatenate(a_l), np.concatenate(b_l), np.concatenate(w_l)
    g = csr_matrix((wt, (a, b)), shape=(len(ci), len(ci)))
    return g, idx, ci, cj


def _los(walk: np.ndarray, lay: Layout, p: np.ndarray, q: np.ndarray) -> bool:
    n = max(2, int(np.ceil(np.linalg.norm(q - p) / (0.4 * lay.cell))) + 1)
    s = np.linspace(0, 1, n)[:, None]
    pts = p[None] * (1 - s) + q[None] * s
    return bool(cell_walkable(lay, walk, pts).all())


def _simplify(walk: np.ndarray, lay: Layout, pts: np.ndarray, window: int = 80) -> np.ndarray:
    out = [pts[0]]
    i, n = 0, len(pts)
    while i < n - 1:
        j = min(n - 1, i + window)
        while j > i + 1 and not _los(walk, lay, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def _catmull(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Centripetal Catmull-Rom through pts, sampled about every ``spacing`` metres."""
    if len(pts) < 3:
        return pts
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = [pts[0]]
    for k in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / spacing)))
        for t in np.linspace(t1, t2, n + 1)[1:]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.array(out)


def _resample(poly: np.ndarray, s: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    return np.stack([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])], axis=1), cum[-1]


def plan_tour(cams: Cams, lay: Layout, walk: np.ndarray, h_cap: float, floor_y: float, spawn_cam: int,
              n_stops: int = 6, duration: float = 60.0, seed: int = 0, step_s: float = 1.0) -> tuple[list[dict], list[dict], dict]:
    """Tour keyframes (one per ``step_s`` seconds) through ``n_stops`` real camera positions.

    Stops are real walkable level cameras spaced evenly along the capture path in capture order, starting at the spawn.
    Legs follow the shortest path over walkable cells, simplified by line of sight and smoothed with a
    Catmull-Rom spline; the spline is used only if every sample stays on a walkable cell.
    """
    info: dict = {}
    on = cell_walkable(lay, walk, cams.centres[:, [0, 2]])
    ok = on & (np.abs(cams.pitch) <= 20.0)
    if ok.sum() < 2:
        return [], [], {"tour": "fewer than two walkable level cameras"}
    xz = cams.centres[:, [0, 2]]
    # stops: spaced evenly along the capture path in capture order, starting at the spawn (wrapping round at the end)
    by_rank = np.argsort(cams.order)
    seq = np.r_[by_rank[int(np.flatnonzero(by_rank == spawn_cam)[0]):], by_rank[:int(np.flatnonzero(by_rank == spawn_cam)[0])]]
    seg = np.linalg.norm(np.diff(xz[seq], axis=0), axis=1)
    seg[seg > 1.0] = 0.0                                   # jumps (lost frames) add no length
    cum = np.r_[0.0, np.cumsum(seg)]
    stops: list[int] = [spawn_cam]
    cand_pos = np.flatnonzero(ok[seq])
    for k in range(1, n_stops):
        target = cum[-1] * k / n_stops
        for q in cand_pos[np.argsort(np.abs(cum[cand_pos] - target))]:
            cam = int(seq[q])
            if all(np.linalg.norm(xz[cam] - xz[s_]) > 1.0 for s_ in stops):
                stops.append(cam)
                break
    if len(stops) < 2:
        return [], [], {"tour": "stops closer than 1 m"}
    g, idx, ci, cj = _graph(walk)
    legs: list[np.ndarray] = []
    for a, b in zip(stops[:-1], stops[1:]):
        ia, ja = lay.ij(xz[a, 0], xz[a, 1])
        ib, jb = lay.ij(xz[b, 0], xz[b, 1])
        sa, sb = int(idx[ia, ja]), int(idx[ib, jb])
        if sa < 0 or sb < 0:
            continue
        _, pred = dijkstra(g, directed=False, indices=sa, return_predecessors=True)
        if sb != sa and pred[sb] < 0:
            continue
        chain = [sb]
        while chain[-1] != sa:
            chain.append(int(pred[chain[-1]]))
        chain = chain[::-1]
        px, pz = lay.xz(ci[chain], cj[chain])
        leg = np.stack([px, pz], axis=1)
        leg[0], leg[-1] = xz[a], xz[b]
        legs.append(_simplify(walk, lay, leg))
    if not legs:
        return [], [], {"tour": "no connected legs"}
    poly = legs[0]
    for leg in legs[1:]:
        poly = np.vstack([poly, leg[1:]])
    smooth = _catmull(poly, 0.1)
    use = "catmull-rom"
    if not cell_walkable(lay, walk, smooth).all():
        smooth, use = poly, "polyline (spline left the walkable area)"
    info["tour_smoothing"] = use
    n_key = int(round(duration / step_s)) + 1
    total = float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum())
    s = np.linspace(0, total, n_key)
    pos, total = _resample(smooth, s)
    if not cell_walkable(lay, walk, pos).all():
        pos, total = _resample(poly, np.linspace(0, float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()), n_key))
        info["tour_smoothing"] = "polyline (resampled spline left the walkable area)"
    # heading: path tangent over a +-1.5 m window, pulled toward the stop camera's heading near each stop,
    # then smoothed in time and rate-limited so the view never whips round
    step = total / max(n_key - 1, 1)
    w = max(1, int(round(1.5 / max(step, 1e-6))))
    fwd = np.zeros((n_key, 3))
    for k in range(n_key):
        a_, b_ = pos[max(0, k - w)], pos[min(n_key - 1, k + w)]
        d = b_ - a_
        nrm = np.linalg.norm(d)
        fwd[k] = [d[0] / nrm, 0.0, d[1] / nrm] if nrm > 1e-6 else (fwd[k - 1] if k else [0, 0, -1])
    for sc in stops:
        d = np.linalg.norm(pos - xz[sc], axis=1)
        wgt = np.clip(1.0 - d / 1.0, 0.0, 1.0)
        wgt = wgt * wgt * (3 - 2 * wgt)
        f_stop = np.array([cams.forward[sc][0], 0.0, cams.forward[sc][2]])
        f_stop /= max(np.linalg.norm(f_stop), 1e-9)
        fwd = (1 - wgt[:, None]) * fwd + wgt[:, None] * f_stop[None]
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    sig = max(1.0, 2.0 / step_s)                                   # about 2 s
    r = np.arange(-int(3 * sig), int(3 * sig) + 1)
    ker = np.exp(-0.5 * (r / sig) ** 2)
    ker /= ker.sum()
    pad = np.pad(fwd, ((len(r) // 2, len(r) // 2), (0, 0)), mode="edge")
    fwd = np.stack([np.convolve(pad[:, c], ker, mode="valid") for c in range(3)], axis=1)
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    yaws = np.degrees(np.unwrap(np.radians([yaw_from_forward(f) for f in fwd])))
    max_turn = 60.0 * step_s                                       # degrees per keyframe
    for k in range(1, len(yaws)):
        yaws[k] = yaws[k - 1] + float(np.clip(yaws[k] - yaws[k - 1], -max_turn, max_turn))
    yaws = (yaws + 180.0) % 360.0 - 180.0
    tour = [{"position": [round(float(p[0]), 4), round(floor_y + h_cap, 4), round(float(p[1]), 4)],
             "yaw_deg": round(float(y), 2), "pitch_deg": 0.0, "t": round(k * step_s, 3)}
            for k, (p, y) in enumerate(zip(pos, yaws))]
    stop_views = [_view(cams, s, h_cap, floor_y, f"Stop {n + 1}", f"s{n + 1}") for n, s in enumerate(stops)]
    info["tour_length_m"] = round(total, 2)
    info["tour_speed_mps"] = round(total / duration, 3)
    return tour, stop_views, info
