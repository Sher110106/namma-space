"""Camera-trajectory sanity filter (proved necessary on the honka smoke test).

The incremental mapper once put frames 134-140 about 1,000 units from the rest, which made gsplat's scene scale
332 and training noise. COLMAP's point_filtering does not catch that. We drop cameras that

1. sit farther from the median camera centre than FAR_FACTOR x the median such distance;
2. leave the local path and come back: a stretch of cameras with a jump on each side, where the cameras just
   before and just after it (its neighbours in time) are close to each other, and the stretch sits well off the
   straight path between them. That is the mark of a stretch the mapper put in the wrong place. The test is local,
   so it does not depend on how long the stretch is (review fix #9, 30 Oct);
3. form a stub of at most SHORT_RUN_MIN cameras at either end of the path, cut off by a jump.

Steps are measured against capture time (`t`, seconds, from 10_frames/frames.json kept[].t), so frames that ingest
dropped as blurred or duplicate do not read as jumps. When any camera has no time (frames mode), the frame index
stands in for it, so frames that SfM did not register do not count as jumps either. A jump is a step whose speed
(distance / time) is more than JUMP_FACTOR x the median speed. Jumps between long stretches that cannot be
explained this way are kept and reported in jumps_left.

Pure functions on (frame index, name, centre, time); no I/O. Stdlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

FAR_FACTOR = 6.0
JUMP_FACTOR = 5.0
SHORT_RUN_MIN = 3        # a stub of at most this many cameras at either end, cut off by a jump, is dropped
MAX_SHARE = 0.5          # never drop a stretch holding this share of the cameras or more
MAX_ROUNDS = 64
SPIKE_DEV = 3.5          # a single camera this many typical steps off its neighbours' line is a spike ...
SPIKE_BRIDGE = 2.0       # ... when those neighbours join at no more than this x the median speed
DETOUR_SPEED = 0.5       # ... and farther than the camera could go and come back at this x its median speed
DETOUR_DEV = 5.0         # a skipped stretch must sit at least this many typical steps off the path that replaces it
TRIM_MAX = 0             # cameras a join may give up at each end it joins (3 was tried: it lets drifting wrong
                         # stretches join the main path on the MuSHRoom checks, so it is off)


@dataclass
class Cam:
    idx: int            # position of the frame in the sorted image list (time order)
    name: str
    centre: tuple[float, float, float]
    t: Optional[float] = None   # capture time in seconds (video path); None in frames mode


@dataclass
class SanityResult:
    kept: list[Cam]
    dropped: list[dict] = field(default_factory=list)       # {name, idx, reason, ratio}
    jumps_left: list[dict] = field(default_factory=list)    # jumps between long runs we kept: {after, before, ratio}
    median_step: float = 0.0                                # median step per unit time (speed)
    median_dist: float = 0.0
    max_step_ratio: float = 0.0
    time_axis: str = "index"                                # "capture_time" or "index"


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _median_centre(cams: list[Cam]) -> tuple[float, float, float]:
    return tuple(median(c.centre[i] for c in cams) for i in range(3))  # type: ignore[return-value]


def _use_time(cams: list[Cam]) -> bool:
    return bool(cams) and all(c.t is not None for c in cams)


def _tm(c: Cam, use_t: bool) -> float:
    return float(c.t) if use_t else float(c.idx)


def _dt_floor(cams: list[Cam], use_t: bool) -> float:
    dts = [_tm(b, use_t) - _tm(a, use_t) for a, b in zip(cams, cams[1:])]
    pos = [d for d in dts if d > 0]
    return 0.5 * median(pos) if pos else 1.0


def _speeds(cams: list[Cam], use_t: bool, floor: float) -> list[float]:
    """Speed between consecutive registered cameras (cams in time order): distance / elapsed time."""
    return [_dist(a.centre, b.centre) / max(floor, _tm(b, use_t) - _tm(a, use_t)) for a, b in zip(cams, cams[1:])]


def _segments(speeds: list[float], limit: float, n: int) -> list[tuple[int, int]]:
    cut = [i for i, s in enumerate(speeds) if s > limit]       # jump between cams[i] and cams[i+1]
    bounds = [0] + [i + 1 for i in cut] + [n]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _detour_score(cams: list[Cam], seg: tuple[int, int], use_t: bool, typ_step: float) -> float:
    """How far the stretch cams[seg] sits off the straight path between its neighbours in time (the cameras just
    before and after it), as a median, in units of a jump-sized step. Reported as the drop ratio."""
    a, b = cams[seg[0] - 1], cams[seg[1]]
    ta, tb = _tm(a, use_t), _tm(b, use_t)
    devs = []
    for c in cams[seg[0]:seg[1]]:
        u = min(1.0, max(0.0, (_tm(c, use_t) - ta) / (tb - ta))) if tb > ta else 0.5
        p = tuple(a.centre[i] + u * (b.centre[i] - a.centre[i]) for i in range(3))
        devs.append(_dist(c.centre, p))
    return median(devs) / max(JUMP_FACTOR * typ_step, 1e-12)


def _spikes(cams: list[Cam], use_t: bool, floor: float, ms: float, typ_step: float) -> list[tuple[int, float]]:
    """Single cameras that sit more than SPIKE_DEV typical steps off the line between their two neighbours in time,
    while those neighbours join at no more than SPIKE_BRIDGE x the median speed. Local maxima only, so a pair of
    neighbouring cameras is judged one at a time across rounds."""
    devs = [0.0] * len(cams)
    for i in range(1, len(cams) - 1):
        a, c, b = cams[i - 1], cams[i], cams[i + 1]
        ta, tc, tb = _tm(a, use_t), _tm(c, use_t), _tm(b, use_t)
        if _dist(a.centre, b.centre) / max(floor, tb - ta) > SPIKE_BRIDGE * ms:
            continue
        u = (tc - ta) / (tb - ta) if tb > ta else 0.5
        p = tuple(a.centre[k] + u * (b.centre[k] - a.centre[k]) for k in range(3))
        devs[i] = _dist(c.centre, p) / max(typ_step, 1e-12)
    return [(i, d) for i, d in enumerate(devs)
            if d > SPIKE_DEV and d >= devs[i - 1] and (i + 1 >= len(devs) or d >= devs[i + 1])]


def _off_line(cams: list[Cam], lo: int, hi: int, a: Cam, b: Cam, use_t: bool) -> float:
    """Median distance of cams[lo:hi] from the straight path a -> b, each camera placed on it by its time."""
    ta, tb = _tm(a, use_t), _tm(b, use_t)
    devs = []
    for c in cams[lo:hi]:
        u = min(1.0, max(0.0, (_tm(c, use_t) - ta) / (tb - ta))) if tb > ta else 0.5
        p = tuple(a.centre[k] + u * (b.centre[k] - a.centre[k]) for k in range(3))
        devs.append(_dist(c.centre, p))
    return median(devs) if devs else 0.0


def _join(cams: list[Cam], si: tuple[int, int], sj: tuple[int, int], use_t: bool, floor: float,
          v_lim: float) -> Optional[tuple[int, int]]:
    """Fewest cameras (ti off the end of segment si, tj off the start of sj, each at most TRIM_MAX and leaving at
    least one camera) whose removal lets si join sj without a jump; None if no such trim exists."""
    best = None
    for ti in range(min(TRIM_MAX, si[1] - si[0] - 1) + 1):
        a = cams[si[1] - 1 - ti]
        for tj in range(min(TRIM_MAX, sj[1] - sj[0] - 1) + 1):
            if best is not None and ti + tj >= sum(best):
                break
            b = cams[sj[0] + tj]
            if _dist(a.centre, b.centre) / max(floor, _tm(b, use_t) - _tm(a, use_t)) <= v_lim:
                best = (ti, tj)
                break
    return best


def _main_chain(cams: list[Cam], segs: list[tuple[int, int]], use_t: bool, floor: float,
                v_lim: float, typ_step: float, ms: float) -> tuple[list[int], list[int]]:
    """The main path: the chain of segments, in time order, that keeps the most cameras while each segment joins
    the next one kept without a jump. The time between them counts, so a skipped stretch of any length can be
    bridged; a join may also give up a few cameras at the ends it joins (a wrong camera just before a jump).
    Dynamic programme over segments. Returns (segment indices on the chain, camera indices trimmed off it)."""
    m = len(segs)
    size = [s[1] - s[0] for s in segs]
    best = list(size)
    prev: list[Optional[tuple[int, int, int]]] = [None] * m      # (i, ti, tj)
    for j in range(m):
        for i in range(j):
            if best[i] + size[j] <= best[j]:
                continue
            tr = _join(cams, segs[i], segs[j], use_t, floor, v_lim)
            if tr is not None and j > i + 1:
                # every stretch skipped by this join must sit clearly off the path it would replace
                a, b = cams[segs[i][1] - 1 - tr[0]], cams[segs[j][0] + tr[1]]
                # ... farther than the camera could go and come back at DETOUR_SPEED x its median speed
                reach = max(DETOUR_DEV * typ_step, DETOUR_SPEED * ms * (_tm(b, use_t) - _tm(a, use_t)) / 2)
                if any(_off_line(cams, segs[k][0], segs[k][1], a, b, use_t) < reach for k in range(i + 1, j)):
                    tr = None
            if tr is not None and best[i] + size[j] - tr[0] - tr[1] > best[j]:
                best[j] = best[i] + size[j] - tr[0] - tr[1]
                prev[j] = (i, tr[0], tr[1])
    j: Optional[int] = max(range(m), key=lambda k: (best[k], -k))
    chain: list[int] = []
    trimmed: list[int] = []
    while j is not None:
        chain.append(j)
        p = prev[j]
        if p is None:
            break
        i, ti, tj = p
        trimmed += list(range(segs[j][0], segs[j][0] + tj)) + list(range(segs[i][1] - ti, segs[i][1]))
        j = i
    return sorted(chain), trimmed


def filter_trajectory(cams: list[Cam], far_factor: float = FAR_FACTOR, jump_factor: float = JUMP_FACTOR) -> SanityResult:
    use_t = _use_time(cams)
    cams = sorted(cams, key=lambda c: (_tm(c, use_t), c.idx))
    dropped: list[dict] = []
    axis = "capture_time" if use_t else "index"
    if len(cams) < 4:
        return SanityResult(kept=cams, time_axis=axis)
    n_all = len(cams)
    ms0: Optional[float] = None
    floor0, typ0 = 1.0, 0.0
    for _ in range(MAX_ROUNDS):
        # Rule 1: far from the median centre.
        mc = _median_centre(cams)
        dists = [_dist(c.centre, mc) for c in cams]
        md = median(dists) or 1e-12
        far = [(c, d / md) for c, d in zip(cams, dists) if d > far_factor * md]
        if far:
            far_ids = {c.idx for c, _ in far}
            for c, r in far:
                dropped.append({"name": c.name, "idx": c.idx, "reason": "far_from_trajectory", "ratio": round(r, 2)})
            cams = [c for c in cams if c.idx not in far_ids]
            if len(cams) < 4:
                break
            continue
        speeds = _speeds(cams, use_t, floor0)
        if ms0 is None:
            # the scale of a normal step is fixed from the first pass, so later rounds do not move the jump bar
            floor0 = _dt_floor(cams, use_t)
            speeds = _speeds(cams, use_t, floor0)
            ms0 = median(speeds) or 1e-12
            dts = [_tm(b, use_t) - _tm(a, use_t) for a, b in zip(cams, cams[1:])]
            typ0 = ms0 * (median([d for d in dts if d > 0]) if any(d > 0 for d in dts) else 1.0)
        floor, ms, typ_step = floor0, ms0, typ0
        v_lim = jump_factor * ms
        segs = _segments(speeds, v_lim, len(cams))
        # Rule 2a: single cameras off the local path, whose two neighbours in time join with a normal step.
        spikes = _spikes(cams, use_t, floor, ms, typ_step)
        if spikes:
            for i, dev in spikes:
                dropped.append({"name": cams[i].name, "idx": cams[i].idx, "reason": "off_local_path",
                                "ratio": round(dev, 2)})
            ids = {cams[i].idx for i, _ in spikes}
            cams = [c for c in cams if c.idx not in ids]
            continue
        if len(segs) < 2:
            break
        chain, trimmed = _main_chain(cams, segs, use_t, floor, v_lim, typ_step, ms)
        first, last = min(chain), max(chain)
        drop: list[tuple[Cam, str, float]] = []
        for ci in trimmed:
            # a camera given up so that two stretches of the main path join without a jump
            drop.append((cams[ci], "off_local_path", round(max(speeds[max(0, ci - 1)], speeds[min(len(speeds) - 1, ci)]) / ms, 2)))
        for k, seg in enumerate(segs):
            if k in chain:
                continue
            size = seg[1] - seg[0]
            if first < k < last and size < MAX_SHARE * n_all:
                # Rule 2: a stretch the main path skips; its neighbours in time join without a jump.
                score = _detour_score(cams, seg, use_t, typ_step)
                for c in cams[seg[0]:seg[1]]:
                    drop.append((c, "off_local_path", round(score, 2)))
            elif (k < first or k > last) and size <= SHORT_RUN_MIN:
                # Rule 3: a short stub at either end, cut off by a jump.
                jump_i = seg[1] - 1 if k < first else seg[0] - 1
                for c in cams[seg[0]:seg[1]]:
                    drop.append((c, "short_run_at_end", round(speeds[jump_i] / ms, 2)))
        if not drop:
            break
        for c, why, ratio in drop:
            dropped.append({"name": c.name, "idx": c.idx, "reason": why, "ratio": ratio})
        ids = {c.idx for c, _, _ in drop}
        cams = [c for c in cams if c.idx not in ids]
        if len(cams) < 4:
            break
    # Final statistics on what is kept.
    floor = _dt_floor(cams, use_t) if len(cams) > 1 else 1.0
    speeds = _speeds(cams, use_t, floor) if len(cams) > 1 else []
    ms = median(speeds) if speeds else 0.0
    mc = _median_centre(cams) if cams else (0.0, 0.0, 0.0)
    md = median(_dist(c.centre, mc) for c in cams) if cams else 0.0
    jumps_left = [
        {"after": a.name, "before": b.name, "ratio": round(s / ms, 2)}
        for a, b, s in zip(cams, cams[1:], speeds) if ms > 0 and s > jump_factor * ms
    ]
    return SanityResult(kept=cams, dropped=sorted(dropped, key=lambda d: d["idx"]), jumps_left=jumps_left,
                        median_step=ms, median_dist=md,
                        max_step_ratio=round(max(speeds) / ms, 2) if ms > 0 and speeds else 0.0, time_axis=axis)
