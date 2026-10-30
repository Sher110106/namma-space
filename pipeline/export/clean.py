"""P0 clean (FINAL_PLAN 2.6): five steps in a fixed order, each with its removal count.

1 NaN/Inf; 2 opacity < 0.05; 3 largest axis above both 1.0 m and the 99.9th percentile;
4 centre outside the SfM points' 0.5-99.5% box grown by 0.5 m; 5 camera bubble (centre within 0.15 m of a camera).
`largest axis` = exp(max log-scale), the one-sigma radius of the splat along its longest axis, in the PLY's units.
`freespace` (P1) is reported as 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np

from .ply_io import Splats, sigmoid


@dataclass
class CleanParams:
    """box_grow_m = inf skips step 4."""
    min_opacity: float = 0.05
    huge_axis_m: float = 1.0
    huge_percentile: float = 99.9
    box_percentiles: Tuple[float, float] = (0.5, 99.5)
    box_grow_m: float = 0.5
    bubble_m: float = 0.15


def _finite_mask(s: Splats) -> np.ndarray:
    ok = np.ones(len(s), dtype=bool)
    for a in (s.means, s.f_dc, s.f_rest, s.scale, s.quat):
        if a.size:
            ok &= np.isfinite(a).all(axis=1)
    ok &= np.isfinite(s.opacity)
    ok &= np.linalg.norm(np.nan_to_num(s.quat), axis=1) > 0
    return ok


def clean(s: Splats, sfm_points: np.ndarray, cam_centres: np.ndarray,
          p: CleanParams = CleanParams(), on_step=None) -> Tuple[Splats, Dict[str, int], dict]:
    """Return (cleaned splats, removal counts, details). on_step(name, splats_after_step) is called after each step."""
    def _step(name: str, cur_: Splats) -> None:
        if on_step is not None:
            on_step(name, cur_)

    removed = {"nan": 0, "opacity": 0, "huge": 0, "outside": 0, "bubble": 0, "freespace": 0}
    details: dict = {"n_in": len(s), "params": asdict(p)}

    keep = _finite_mask(s)
    removed["nan"] = int((~keep).sum())
    cur = s.take(keep) if removed["nan"] else s
    _step("nan", cur)

    keep = sigmoid(cur.opacity) >= p.min_opacity
    removed["opacity"] = int((~keep).sum())
    cur = cur.take(keep)
    _step("opacity", cur)

    axis = np.exp(cur.scale.max(axis=1).astype(np.float64))
    pct = float(np.percentile(axis, p.huge_percentile)) if len(cur) else 0.0
    thr = max(p.huge_axis_m, pct)
    keep = axis <= thr
    removed["huge"] = int((~keep).sum())
    details["huge_axis_threshold"] = thr
    details["huge_axis_p99_9"] = pct
    cur = cur.take(keep)
    _step("huge", cur)

    if len(sfm_points) and np.isfinite(p.box_grow_m):
        lo = np.percentile(sfm_points, p.box_percentiles[0], axis=0) - p.box_grow_m
        hi = np.percentile(sfm_points, p.box_percentiles[1], axis=0) + p.box_grow_m
        keep = ((cur.means >= lo) & (cur.means <= hi)).all(axis=1)
        details["box_min"] = lo.tolist()
        details["box_max"] = hi.tolist()
    else:
        keep = np.ones(len(cur), dtype=bool)
        details["box_min"] = details["box_max"] = None
    removed["outside"] = int((~keep).sum())
    cur = cur.take(keep)
    _step("outside", cur)

    if len(cam_centres):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.asarray(cam_centres, dtype=np.float64)).query(cur.means.astype(np.float64), workers=-1)
        keep = d >= p.bubble_m
    else:
        keep = np.ones(len(cur), dtype=bool)
    removed["bubble"] = int((~keep).sum())
    cur = cur.take(keep)
    _step("bubble", cur)

    details["n_out"] = len(cur)
    return cur, removed, details


def importance(s: Splats, mode: str = "area") -> np.ndarray:
    """Pruning score. 'area' = opacity * projected-footprint proxy (two largest axes squared)."""
    a = sigmoid(s.opacity)
    sc = np.sort(s.scale.astype(np.float64), axis=1)          # log scales ascending
    if mode == "opacity":
        return a
    if mode == "opacity_size":                                # opacity * geometric-mean size
        return a * np.exp(sc.mean(axis=1))
    if mode == "volume":
        return a * np.exp(sc.sum(axis=1))
    return a * np.exp(sc[:, 1] + sc[:, 2])                    # opacity * (s_mid * s_max): footprint proxy


def prune_to(s: Splats, n: int, mode: str = "area") -> Splats:
    """Keep the n highest-importance splats (original order preserved)."""
    if len(s) <= n:
        return s
    score = importance(s, mode)
    idx = np.argpartition(-score, n - 1)[:n]
    idx.sort()
    return s.take(idx)


def measure_floor_y(s: Splats, cam_centres: np.ndarray, bin_m: float = 0.02,
                    min_support_frac: float = 0.25, up_axis: int = 1):
    """Floor height from opaque splat footprints (FINAL_PLAN 2.4 rule): the lowest horizontal-plane peak of the
    footprint-weighted height histogram that holds >= 25% of the tallest peak and lies below the cameras.
    Returns (floor_y or None, info)."""
    a = sigmoid(s.opacity)
    m = a > 0.5
    if m.sum() < 1000 or len(cam_centres) == 0:
        return None, {"reason": "too few opaque splats"}
    y = s.means[m, up_axis].astype(np.float64)
    sc = np.sort(s.scale[m].astype(np.float64), axis=1)
    w = np.exp(sc[:, 1] + sc[:, 2])                               # footprint proxy of a flat splat
    cam_y = np.asarray(cam_centres)[:, up_axis]
    lo, hi = np.percentile(y, 0.5), min(np.percentile(y, 99.5), float(np.median(cam_y)))
    if hi - lo < 0.2:
        return None, {"reason": "height range too small"}
    edges = np.arange(lo, hi + bin_m, bin_m)
    h, _ = np.histogram(y, bins=edges, weights=w)
    k = np.ones(3) / 3
    hs = np.convolve(h, k, mode="same")
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]]
    if not peaks:
        return None, {"reason": "no peak"}
    tall = max(hs[i] for i in peaks)
    cand = [i for i in peaks if hs[i] >= min_support_frac * tall]
    i = min(cand)                                                 # lowest sufficiently large peak
    c = 0.5 * (edges[i] + edges[i + 1])
    sel = np.abs(y - c) <= 0.03
    fy = float(np.average(y[sel], weights=w[sel])) if sel.any() else float(c)
    return fy, {"peak_y": float(c), "support_frac_of_tallest": float(hs[i] / tall), "n_peaks": len(peaks),
                "tallest_peak_y": float(0.5 * (edges[max(peaks, key=lambda j: hs[j])] + edges[max(peaks, key=lambda j: hs[j]) + 1]))}
