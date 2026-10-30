"""Up direction, floor, camera height, metric scale and the alignment similarity (FINAL_PLAN 2.4).

numpy only (scipy is not needed).  No Manhattan aligner, no LSD: up comes from three cues,

  1. cameras   trimmed mean of the cameras' y-axes (image-down) in world coordinates;
  2. planes    RANSAC horizontal planes (floor, ceiling) on the SfM points;
  3. walls     the direction that is perpendicular to every RANSAC wall normal
               (smallest eigenvector of sum w m m^T), when the walls are not all parallel.

Cues that disagree by more than 5 degrees flag the scene.

World frame produced (CONTRACTS 2.0): metres, +Y up, floor at y = 0, origin at the camera-path
centroid dropped onto the floor, main wall along +X when the walls give a clear direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import colmap_io as cio

DEG = 180.0 / math.pi
HORIZ_TOL_DEG = 22.0      # a plane is "horizontal" if its normal is this close to the up estimate
WALL_TOL_DEG = 22.0       # ... "vertical" if its normal is this close to perpendicular
WALL_TOL_DEG_TIGHT = 14.0 # second pass, about the refined up
MIN_SUPPORT_FRAC = 0.03   # a plane needs this share of the points
FLOOR_REL_SUPPORT = 0.25  # floor needs this share of the largest horizontal plane's support
WALL_REL_SUPPORT = 0.30   # a wall needs this share of the largest wall's support
PITCH_LEVEL_DEG = 20.0
H_PRIOR_M = 1.40          # camera height without a user height
H_FACTOR = 0.80           # phone height = this x user height
SIGMA_H_WITH = 0.08
SIGMA_H_WITHOUT = 0.12
SIGMA_MIN_MOGE = 0.08
PLANE_AGREE_DEG = 5.0     # a horizontal plane joins the plane cue only within this angle of the current up
CUE_DISAGREE_DEG = 5.0    # planes vs walls (both geometric) may disagree by this much before the scene is flagged
CAMERA_CUE_MAX_DEG = 10.0 # the camera cue is biased by the capture's mean pitch (4.2 deg on honka, 4.9 on Mip room),
                          # so it is gated against the fused up at this looser bound, not at 5 deg (integrator, 30 Oct)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0)) * DEG)


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-12:                          # opposite: half turn about any perpendicular
        p = unit(np.cross(a, np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])))
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def rot_y(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------------------------------------
# cue 1: cameras

def up_from_cameras(Rs: np.ndarray, n_bins: int = 12, iters: int = 4, max_dev_deg: float = 40.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rs: (n,3,3) world-to-camera.  Image-down in world coordinates is row 1 of each R; the up cue is its negated mean.

    Averaging over headings is what cancels a mean downward pitch, so the mean is taken with the cameras weighted by
    1 / (number of cameras in their heading bin) (12 bins about the current up).  A plain or angle-trimmed mean is
    biased when the path favours some headings (honka: 4.8 deg off gravity plain, 1.7 deg heading-balanced).  Cameras
    more than 40 deg from the mean (a landscape roll, a broken pose) are left out."""
    down = Rs[:, 1, :]
    fwd = Rs[:, 2, :]
    m = unit(down.mean(0))
    keep = np.ones(len(down), bool)
    for _ in range(iters):
        u = -m
        e1 = unit(np.cross(u, np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])))
        e2 = np.cross(u, e1)
        az = np.arctan2(fwd @ e2, fwd @ e1)
        b = ((az + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
        ang = np.arccos(np.clip(down @ m, -1, 1)) * DEG
        keep = ang <= max_dev_deg
        cnt = np.bincount(b[keep], minlength=n_bins)
        w = np.where(keep, 1.0 / np.maximum(cnt[b], 1), 0.0)
        m = unit((w[:, None] * down).sum(0))
    ang = np.arccos(np.clip(down @ m, -1, 1)) * DEG
    return -m, {"n": int(len(down)), "n_kept": int(keep.sum()), "heading_bins_used": int((np.bincount(b[keep], minlength=n_bins) > 0).sum()),
                "spread_deg_median": float(np.median(ang)), "spread_deg_p90": float(np.quantile(ang, 0.9))}


# ---------------------------------------------------------------------------------------------
# cues 2 and 3: RANSAC planes

@dataclass
class Plane:
    n: np.ndarray            # unit normal
    d: float                 # the plane is n.x = d (for a horizontal plane, d is its height along n)
    support: int
    rms: float
    kind: str = ""           # "horizontal" | "vertical" | ""


def _fit_plane(P: np.ndarray) -> Tuple[np.ndarray, float]:
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    return n, float(n @ c)           # n.x = offset


def ransac_plane(P: np.ndarray, thr: float, rng: np.random.Generator, iters: int = 1500,
                 axis: Optional[np.ndarray] = None, mode: str = "any", tol_deg: float = 22.0) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Best plane by inlier count.  mode 'horizontal': normal within tol of `axis`;
    'vertical': normal within tol of perpendicular to `axis`.  Returns (n, offset, inlier mask)."""
    N = len(P)
    if N < 20:
        return None
    cos_h = math.cos(tol_deg / DEG)
    sin_v = math.sin(tol_deg / DEG)
    idx = rng.integers(0, N, size=(iters, 3))
    A, B, C = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    nrm = np.cross(B - A, C - A)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    nrm = nrm / np.where(ln > 0, ln, 1)[:, None]
    if mode == "horizontal":
        ok &= np.abs(nrm @ axis) >= cos_h
    elif mode == "vertical":
        ok &= np.abs(nrm @ axis) <= sin_v
    cand = np.nonzero(ok)[0]
    if len(cand) == 0:
        return None
    # score candidates on a subsample for speed, then re-score the best few on everything
    sub = P if N <= 8000 else P[rng.integers(0, N, size=8000)]
    offs = np.einsum("ij,ij->i", nrm[cand], A[cand])
    scores = np.empty(len(cand), np.int64)
    for k0 in range(0, len(cand), 256):
        sl = slice(k0, k0 + 256)
        dist = np.abs(sub @ nrm[cand[sl]].T - offs[sl][None, :])
        scores[sl] = (dist < thr).sum(0)
    best = None
    for k in np.argsort(-scores)[:5]:
        n, off = nrm[cand[k]], float(offs[k])
        for _ in range(4):                          # refine on inliers
            inl = np.abs(P @ n - off) < thr
            if inl.sum() < 10:
                break
            n2, off2 = _fit_plane(P[inl])
            if mode == "horizontal" and abs(n2 @ axis) < cos_h:
                break
            if mode == "vertical" and abs(n2 @ axis) > sin_v:
                break
            n, off = n2, off2
        inl = np.abs(P @ n - off) < thr
        if best is None or inl.sum() > best[2].sum():
            best = (n, off, inl)
    return best


def find_planes(P: np.ndarray, axis: np.ndarray, thr: float, rng: np.random.Generator, mode: str, n_planes: int,
                tol_deg: float, min_frac: float) -> List[Plane]:
    """Sequential RANSAC of one kind ('horizontal' or 'vertical' relative to `axis`), removing inliers each time."""
    N = len(P)
    min_sup = max(30, int(min_frac * N))
    out: List[Plane] = []
    rest = np.ones(N, bool)
    for _ in range(n_planes):
        res = ransac_plane(P[rest], thr, rng, axis=axis, mode=mode, tol_deg=tol_deg)
        if res is None:
            break
        n, off, inl = res
        if inl.sum() < min_sup:
            break
        full = (np.abs(P @ n - off) < thr) & rest
        n, off = _fit_plane(P[full])
        full = (np.abs(P @ n - off) < thr) & rest
        r = P[full] @ n - off
        if mode == "horizontal" and n @ axis < 0:
            n, off = -n, -off
        out.append(Plane(n=n, d=float(off), support=int(full.sum()), rms=float(np.sqrt(np.mean(r ** 2))), kind=mode))
        rest &= ~full
    return out


def up_from_walls(walls: List[Plane], up0: np.ndarray, min_keep: int = 2) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Direction perpendicular to the wall normals: smallest eigenvector of sum w n n^T, with w = support / rms^2.
    Walls more than max(3 deg, 1.5 x the median) off perpendicular are dropped and the fit repeated (twice).
    Returns None when fewer than 2 walls remain or they are all parallel (a corridor)."""
    use = list(walls)
    if len(use) < min_keep:
        return None, {"reason": "fewer than 2 walls"}
    u = None
    evals = None
    for _ in range(3):
        M = np.zeros((3, 3))
        for w in use:
            M += (w.support / max(w.rms, 1e-9) ** 2) * np.outer(w.n, w.n)
        evals, evecs = np.linalg.eigh(M)
        if evals[1] < 0.05 * evals[2]:
            return None, {"reason": "walls are nearly parallel", "eigenvalues": evals.tolist()}
        u = evecs[:, 0]
        if u @ up0 < 0:
            u = -u
        resid = np.array([abs(math.asin(min(1.0, abs(w.n @ u)))) * DEG for w in use])
        keep = [w for w, r in zip(use, resid) if r <= max(3.0, 1.5 * float(np.median(resid)))]
        if len(keep) == len(use) or len(keep) < min_keep:
            break
        use = keep
    resid = [float(abs(math.asin(min(1.0, abs(w.n @ u)))) * DEG) for w in use]
    return u, {"n_walls": len(walls), "n_walls_used": len(use), "eigenvalues": evals.tolist(),
               "wall_resid_deg_max": max(resid), "wall_resid_deg_mean": float(np.mean(resid))}


# ---------------------------------------------------------------------------------------------
# putting the cues together

@dataclass
class UpResult:
    up: np.ndarray
    cues: Dict[str, Optional[np.ndarray]]
    cue_weights: Dict[str, float]
    cue_angles_deg: Dict[str, float]
    cue_spread_deg: float
    resid_deg: float
    floor: Optional[Plane]
    floor_h: float                      # floor height along `up`, model units
    floor_info: Dict[str, Any]
    ceiling_h: Optional[float]
    horizontals: List[Plane]
    walls: List[Plane]
    thr: float
    info: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


def robust_extent(P: np.ndarray) -> float:
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    return float(np.linalg.norm(hi - lo))


def _fuse(cues: Dict[str, Optional[np.ndarray]], w: Dict[str, float], ref: np.ndarray) -> np.ndarray:
    acc = np.zeros(3)
    for k, v in cues.items():
        if v is not None and w.get(k, 0) > 0:
            acc += w[k] * (v if v @ ref >= 0 else -v)
    return unit(acc)


def _plane_heights(P: np.ndarray, pl: Plane, thr: float, up: np.ndarray) -> np.ndarray:
    return P[np.abs(P @ pl.n - pl.d) < thr] @ up


def estimate_up(model: cio.Model, seed: int = 0, thr_frac: float = 0.012, max_pts: int = 60000) -> UpResult:
    rng = np.random.default_rng(seed)
    Rs = model.rotations()
    C = model.centers()
    u_cam, cam_info = up_from_cameras(Rs)
    P = model.xyz
    # far outliers (a bad segment can sit 1000 units away) stay out of the RANSAC and the extent
    med = np.median(P, axis=0)
    dist = np.linalg.norm(P - med, axis=1)
    P = P[dist < 4.0 * np.quantile(dist, 0.9) + 1e-9]
    thr = thr_frac * robust_extent(P)
    Pr = P if len(P) <= max_pts else P[rng.choice(len(P), max_pts, replace=False)]
    f = len(P) / float(len(Pr))
    flags: List[str] = []

    # pass 1: horizontal planes about the camera up; their normals refine up
    horiz = find_planes(Pr, u_cam, thr, rng, "horizontal", 8, HORIZ_TOL_DEG, 0.02)
    for p in horiz:
        p.support = int(round(p.support * f))
    cues: Dict[str, Optional[np.ndarray]] = {"cameras": u_cam, "planes": None, "walls": None}
    w = {"cameras": 1.0, "planes": 2.0, "walls": 1.0}
    # The plane cue uses only planes that agree with the current up within PLANE_AGREE_DEG, re-selected as the up
    # improves (integrator, 30 Oct). Before, every plane with 25% of the largest support counted, weighted by
    # support / rms^2; on the real rooms most "horizontal" RANSAC planes were slanted slices through furniture
    # (13-20 deg off gravity), which put the plane cue 3.9 deg (honka) and 7.2 deg (vr_room) off gravity and the fused
    # up 2.0 and 4.2 deg off. With the selection: 1.4 and 1.9 deg (measured against the MuSHRoom poses).
    up1 = u_cam
    sel: List[Plane] = []
    for _ in range(5):
        sel = [p for p in horiz if angle_deg(p.n if p.n @ up1 >= 0 else -p.n, up1) <= PLANE_AGREE_DEG]
        cues["planes"] = unit(sum(p.support * (p.n if p.n @ up1 >= 0 else -p.n) for p in sel)) if sel else None
        up1 = _fuse(cues, w, u_cam)
    plane_info = {"n_found": len(horiz), "n_used": len(sel), "agree_deg": PLANE_AGREE_DEG,
                  "used_supports": [int(p.support) for p in sel]}

    # pass 2: walls about the refined up
    walls = find_planes(Pr, up1, thr, rng, "vertical", 8, WALL_TOL_DEG_TIGHT, MIN_SUPPORT_FRAC)
    for p in walls:
        p.support = int(round(p.support * f))
    # fragments and diagonal slices through a room's walls show up as small planes after the real walls are removed
    # (synthetic room: four slices at 45 deg with 25-28% of the largest wall); a wall is at least 30% of the largest
    if walls:
        top = max(p.support for p in walls)
        walls = [p for p in walls if p.support >= WALL_REL_SUPPORT * top]
    u_wall, wall_info = up_from_walls(walls, up1)
    cues["walls"] = u_wall
    if u_wall is not None:
        r = wall_info["wall_resid_deg_mean"]
        ev = wall_info["eigenvalues"]
        cond = ev[1] / ev[2]                 # 1 = walls in two perpendicular directions with equal weight; 0 = all parallel
        wall_info["conditioning"] = cond
        w["walls"] = 1.0 if (r <= 3.0 and cond >= 0.25) else 0.0
        wall_info["weight"] = w["walls"]
        if w["walls"] == 0.0:
            wall_info["note"] = (f"cue not used: wall normals scatter {r:.1f} deg from vertical, conditioning {cond:.2f} "
                                 f"(needs <= 3 deg and >= 0.25: walls in two directions)")
    up = _fuse(cues, w, u_cam)
    # the up with the walls may move the plane selection once more
    for _ in range(3):
        sel2 = [p for p in horiz if angle_deg(p.n if p.n @ up >= 0 else -p.n, up) <= PLANE_AGREE_DEG]
        if not sel2 or [id(p) for p in sel2] == [id(p) for p in sel]:
            break
        sel = sel2
        cues["planes"] = unit(sum(p.support * (p.n if p.n @ up >= 0 else -p.n) for p in sel))
        up = _fuse(cues, w, u_cam)
        plane_info.update(n_used=len(sel), used_supports=[int(p.support) for p in sel], reselected_with_walls=True)
    names = [k for k, v in cues.items() if v is not None]
    cue_angles = {k: angle_deg(cues[k], up) for k in names}
    used = [k for k in names if w[k] > 0]
    pair = [angle_deg(cues[a], cues[b]) for i, a in enumerate(used) for b in used[i + 1:]]
    spread = max(pair) if pair else 0.0
    resid = max(cue_angles[k] for k in used) if used else 0.0
    # Gate (FINAL_PLAN 2.4 "flag if the cues disagree by more than 5 deg"), applied to the geometric cues only.
    # On honka the literal rule fired on the camera cue's pitch bias (spread 6.2 deg) while the fused up was 2.0 deg
    # from gravity (metric_check against the MuSHRoom poses), so a good scene would always read "review".
    geo = [k for k in ("planes", "walls") if k in used]
    geo_pair = [angle_deg(cues[a], cues[b]) for i, a in enumerate(geo) for b in geo[i + 1:]]
    geo_spread = max(geo_pair) if geo_pair else 0.0
    cam_dev = cue_angles.get("cameras", 0.0) if "cameras" in used else 0.0
    if geo_spread > CUE_DISAGREE_DEG:
        flags.append(f"up cues disagree by {geo_spread:.1f} deg (planes vs walls; > {CUE_DISAGREE_DEG:g})")
    elif geo and cam_dev > CAMERA_CUE_MAX_DEG:
        flags.append(f"up cues disagree by {cam_dev:.1f} deg (camera cue vs fused up; > {CAMERA_CUE_MAX_DEG:g})")
    elif spread > CUE_DISAGREE_DEG:
        flags.append(f"note: camera up cue is {cam_dev:.1f} deg from the fused up (cue spread {spread:.1f} deg); "
                     f"expected from the capture's pitch, not gated")

    # floor and ceiling heights along the final up
    fl_pl, floor_h, finfo, ceil_h = choose_floor(P, C, up, horiz, walls, thr, flags)
    info = {"cameras": cam_info, "walls": wall_info, "planes": plane_info,
            "n_points_used": int(len(P)), "n_points_ransac": int(len(Pr)),
            "ransac_thr_model_units": thr}
    return UpResult(up=up, cues=cues, cue_weights=w, cue_angles_deg=cue_angles, cue_spread_deg=spread, resid_deg=resid,
                    floor=fl_pl, floor_h=floor_h, floor_info=finfo, ceiling_h=ceil_h, horizontals=horiz, walls=walls,
                    thr=thr, info=info, flags=flags)


def wall_extents(P: np.ndarray, walls: List[Plane], thr: float, up: np.ndarray) -> Optional[Dict[str, float]]:
    """Floor and ceiling heights from where the big walls start and stop (2nd and 98th percentile of inlier heights,
    support-weighted median over the walls that span the most height)."""
    rows = []
    for wl in walls:
        h = _plane_heights(P, wl, thr, up)
        if len(h) < 50:
            continue
        lo, hi = np.percentile(h, [2, 98])
        rows.append((lo, hi, len(h)))
    if len(rows) < 2:
        return None
    rng_max = max(hi - lo for lo, hi, _ in rows)
    rows = [r for r in rows if (r[1] - r[0]) >= 0.6 * rng_max]
    if len(rows) < 2:
        return None

    def wmedian(vals, wts):
        o = np.argsort(vals)
        c = np.cumsum(np.array(wts, float)[o])
        return float(np.array(vals)[o][np.searchsorted(c, 0.5 * c[-1])])
    lo = wmedian([r[0] for r in rows], [r[2] for r in rows])
    hi = wmedian([r[1] for r in rows], [r[2] for r in rows])
    return {"floor": lo, "ceiling": hi, "n_walls": len(rows)}


def choose_floor(P: np.ndarray, C: np.ndarray, up: np.ndarray, horiz: List[Plane], walls: List[Plane], thr: float,
                 flags: List[str]) -> Tuple[Optional[Plane], float, Dict[str, Any], Optional[float]]:
    """The lowest large horizontal plane with the cameras above it (FINAL_PLAN 2.4), cross-checked against
    where the walls end.  A plain floor has few SfM points, so a table top or bench can be the lowest plane
    that RANSAC finds; the walls continue below such a plane, which rejects it."""
    cam_h = C @ up
    cands = []
    for p in horiz:
        h = _plane_heights(P, p, thr, up)
        if len(h) < 10:
            continue
        hp = float(np.median(h))
        if float(np.mean(cam_h > hp)) >= 0.9:
            cands.append((hp, p))
    ext = wall_extents(P, walls, thr, up)
    info: Dict[str, Any] = {"wall_extent": ext, "n_horizontal_candidates": len(cands)}
    maxsup = max((p.support for _, p in cands), default=0)
    if ext is not None:
        rng_h = max(ext["ceiling"] - ext["floor"], 1e-9)
        acc = [(hp, p) for hp, p in cands if -0.25 * rng_h <= hp - ext["floor"] <= 0.12 * rng_h
               and p.support >= 0.1 * maxsup]
        if acc:
            hp, p = min(acc, key=lambda x: x[0])
            info.update({"method": "ransac_plane", "agrees_with_walls": True, "support": int(p.support),
                         "rms_model_units": float(p.rms)})
            return p, hp, info, ext["ceiling"]
        # no plane matches where the walls end: take the walls' floor
        near = int(np.sum(np.abs(P @ up - ext["floor"]) < thr))
        info.update({"method": "wall_extent", "agrees_with_walls": None, "support": near, "rms_model_units": float("nan")})
        flags.append("floor taken from where the walls end (no horizontal plane there)")
        return None, ext["floor"], info, ext["ceiling"]
    big = [(hp, p) for hp, p in cands if p.support >= FLOOR_REL_SUPPORT * maxsup]
    if big:
        hp, p = min(big, key=lambda x: x[0])
        info.update({"method": "ransac_plane", "agrees_with_walls": None, "support": int(p.support),
                     "rms_model_units": float(p.rms)})
        ceil = [h for h, q in cands if h > hp + 1e-9 and float(np.mean(cam_h < h)) >= 0.9]
        return p, hp, info, (max(ceil) if ceil else None)
    h = P @ up
    off = float(np.percentile(h, 2.0))
    info.update({"method": "point_height_percentile_2", "support": 0, "rms_model_units": float("nan")})
    flags.append("no floor plane found; floor from the 2nd height percentile of the points")
    return None, off, info, None


def camera_height(model: cio.Model, up: np.ndarray, floor_h: float) -> Dict[str, Any]:
    """H_cap: median camera height above the floor over frames pitched within +-20 deg of level."""
    Rs = model.rotations()
    C = model.centers()
    fwd = Rs[:, 2, :]                                    # optical axis in world coordinates
    pitch = np.arcsin(np.clip(fwd @ up, -1, 1)) * DEG    # + looking up
    h = C @ up - floor_h
    level = np.abs(pitch) <= PITCH_LEVEL_DEG
    note = None
    if level.sum() >= max(10, int(0.1 * len(h))):
        h_cap = float(np.median(h[level]))
        n_used = int(level.sum())
    else:
        h_cap = float(np.median(h))
        n_used = int(len(h))
        note = f"only {int(level.sum())} level frames; used all {len(h)}"
    return {"h_cap": h_cap, "n_level": int(level.sum()), "n_used": n_used, "pitch_median_deg": float(np.median(pitch)),
            "pitch_p10_p90_deg": [float(np.percentile(pitch, 10)), float(np.percentile(pitch, 90))],
            "h_all_median": float(np.median(h)), "frac_below_floor": float(np.mean(h < 0)), "note": note}


# ---------------------------------------------------------------------------------------------
# scale

@dataclass
class Cue:
    name: str
    value: float          # metres per model unit
    sigma_log: float
    note: str = ""
    used: bool = True


def fuse_scale(cues: List[Cue]) -> Tuple[float, float, List[Cue]]:
    """Inverse-variance fusion in log space with a 3-sigma outlier drop.  Returns (scale, sigma_log, cues)."""
    for c in cues:
        c.used = True
    if len(cues) == 1:
        return cues[0].value, cues[0].sigma_log, cues
    while True:
        act = [c for c in cues if c.used]
        if len(act) < 2:
            break
        worst, worst_z = None, 0.0
        for c in act:
            others = [o for o in act if o is not c]
            wo = np.array([1.0 / o.sigma_log ** 2 for o in others])
            mo = float(np.sum(wo * np.array([math.log(o.value) for o in others])) / wo.sum())
            so = math.sqrt(1.0 / wo.sum())
            z = abs(math.log(c.value) - mo) / math.sqrt(c.sigma_log ** 2 + so ** 2)
            if z > worst_z:
                worst, worst_z = c, z
        if worst is not None and worst_z > 3.0:
            worst.used = False
            worst.note = (worst.note + f" dropped: {worst_z:.1f} sigma from the other cues").strip()
        else:
            break
    act = [c for c in cues if c.used]
    w = np.array([1.0 / c.sigma_log ** 2 for c in act])
    mu = float(np.sum(w * np.array([math.log(c.value) for c in act])) / w.sum())
    return math.exp(mu), math.sqrt(1.0 / w.sum()), cues


def camera_height_cue(h_cap: float, height_m: Optional[float]) -> Cue:
    if height_m:
        return Cue("camera_height", H_FACTOR * height_m / h_cap, SIGMA_H_WITH,
                   f"{H_FACTOR} x user height {height_m:.2f} m / H_cap")
    return Cue("camera_height", H_PRIOR_M / h_cap, SIGMA_H_WITHOUT, f"generic {H_PRIOR_M:.2f} m / H_cap (no user height)")


# ---------------------------------------------------------------------------------------------
# yaw from walls

def yaw_from_walls(walls: List[Plane], up: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Unit horizontal direction (in the input frame) of the main wall, or None when unclear.

    Wall normals are projected on the horizontal plane; their angles mod 90 deg are averaged on the
    4-theta circle weighted by support.  The direction returned is the in-plane direction of the
    best-supported wall, snapped to the circular mean.  'Clear' needs two walls and concentration >= 0.8."""
    if len(walls) < 2:
        return None, {"clear": False, "reason": "fewer than 2 walls"}
    e1 = unit(np.cross(up, np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])))
    e2 = np.cross(up, e1)
    th, wt = [], []
    for w in walls:
        nh = w.n - (w.n @ up) * up
        if np.linalg.norm(nh) < 1e-6:
            continue
        th.append(math.atan2(nh @ e2, nh @ e1))
        wt.append(w.support)
    th, wt = np.array(th), np.array(wt, float)
    if len(th) < 2:
        return None, {"clear": False, "reason": "fewer than 2 usable walls"}
    z = np.sum(wt * np.exp(4j * th)) / wt.sum()
    rho = float(abs(z))
    mean4 = math.atan2(z.imag, z.real) / 4.0       # wall-normal direction mod 90 deg
    info = {"clear": rho >= 0.8, "concentration": rho, "n_walls": int(len(th)), "normal_angle_deg": float(mean4 * DEG)}
    if rho < 0.8:
        return None, info
    # the strongest wall's normal, snapped to the circular mean (mod 90)
    k = int(np.argmax(wt))
    t0 = th[k]
    snapped = mean4 + round((t0 - mean4) / (math.pi / 2)) * (math.pi / 2)
    normal = math.cos(snapped) * e1 + math.sin(snapped) * e2
    along = np.cross(up, normal)                    # in-plane direction of that wall
    info["main_wall_along_deg"] = float(math.atan2(along @ e2, along @ e1) * DEG)
    return unit(along), info


# ---------------------------------------------------------------------------------------------
# the similarity

@dataclass
class AlignResult:
    S: np.ndarray                 # 4x4, X' = S X
    s: float
    R: np.ndarray
    t: np.ndarray
    up: UpResult
    floor_h: float
    floor_info: Dict[str, Any]
    height: Dict[str, Any]
    cues: List[Cue]
    sigma_log: float
    yaw: Dict[str, Any]
    flags: List[str]
    h_cap_m: float


def build_similarity(model: cio.Model, ur: UpResult, floor_h: float, s: float, yaw_dir: Optional[np.ndarray]
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = ur.up
    R = rot_between(up, np.array([0.0, 1.0, 0.0]))
    if yaw_dir is not None:
        d = R @ yaw_dir
        phi = math.atan2(d[2], d[0])       # rot_y(phi) maps (dx, 0, dz) to (|d|, 0, 0)
        R = rot_y(phi) @ R
    C = model.centers()
    Cbar = C.mean(0)
    p0 = Cbar - (Cbar @ up - floor_h) * up
    t = -s * (R @ p0)
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    return S, R, t


def align_model(model: cio.Model, height_m: Optional[float] = None, seed: int = 0, extra_cues: Optional[List[Cue]] = None,
                use_yaw: bool = True) -> AlignResult:
    ur = estimate_up(model, seed=seed)
    floor_h, floor_info = ur.floor_h, ur.floor_info
    ht = camera_height(model, ur.up, floor_h)
    flags = list(ur.flags)
    if ht["h_cap"] <= 0:
        flags.append("H_cap is not positive: cameras are below the floor estimate")
        ht["h_cap"] = max(abs(ht["h_cap"]), 1e-6)
    if ht["note"]:
        flags.append(ht["note"])
    cues = [camera_height_cue(ht["h_cap"], height_m)] + list(extra_cues or [])
    s, sigma, cues = fuse_scale(cues)
    yaw_dir, yaw_info = (yaw_from_walls(ur.walls, ur.up) if use_yaw else (None, {"clear": False, "reason": "disabled"}))
    S, R, t = build_similarity(model, ur, floor_h, s, yaw_dir)
    h_cap_m = ht["h_cap"] * s
    if not (1.0 <= h_cap_m <= 1.8):
        flags.append(f"H_cap x scale = {h_cap_m:.2f} m is outside 1.0-1.8 m: scale or floor error")
    if ur.ceiling_h is not None:
        ceil_m = (ur.ceiling_h - floor_h) * s
        if not (2.0 <= ceil_m <= 4.5):
            flags.append(f"ceiling at {ceil_m:.2f} m is outside 2.0-4.5 m: the camera-height scale is suspect (phone held higher or lower than assumed?)")
    return AlignResult(S=S, s=s, R=R, t=t, up=ur, floor_h=floor_h, floor_info=floor_info, height=ht, cues=cues,
                       sigma_log=sigma, yaw=yaw_info, flags=flags, h_cap_m=h_cap_m)


def scale_interval(s: float, sigma_log: float) -> Dict[str, float]:
    lo, hi = s * math.exp(-sigma_log), s * math.exp(sigma_log)
    return {"lo": lo, "hi": hi, "pct_68": 100.0 * (math.exp(sigma_log) - 1.0)}
