"""Camera-trajectory sanity filter (proved necessary on the honka smoke test).

The incremental mapper once put frames 134-140 about 1,000 units from the rest, which made gsplat's scene scale
332 and training noise. COLMAP's point_filtering does not catch that. We drop cameras that

1. sit farther from the median camera centre than FAR_FACTOR x the median such distance, and
2. form a short run between two jumps: a jump is a step between consecutive registered frames larger than
   JUMP_FACTOR x the median step (per frame of index gap, so unregistered gaps do not count as jumps).
   Long runs between jumps are kept and reported; a long run is not obviously wrong.

Pure functions on (frame index, name, centre); no I/O. Stdlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

FAR_FACTOR = 6.0
JUMP_FACTOR = 5.0
SHORT_RUN_MIN = 3        # a run of at most max(SHORT_RUN_MIN, SHORT_RUN_FRAC * n) cameras between jumps is dropped
SHORT_RUN_FRAC = 0.05
MAX_ROUNDS = 4


@dataclass
class Cam:
    idx: int            # position of the frame in the sorted image list (time order)
    name: str
    centre: tuple[float, float, float]


@dataclass
class SanityResult:
    kept: list[Cam]
    dropped: list[dict] = field(default_factory=list)       # {name, idx, reason, ratio}
    jumps_left: list[dict] = field(default_factory=list)    # jumps between long runs we kept: {after, before, ratio}
    median_step: float = 0.0
    median_dist: float = 0.0
    max_step_ratio: float = 0.0


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _median_centre(cams: list[Cam]) -> tuple[float, float, float]:
    return tuple(median(c.centre[i] for c in cams) for i in range(3))  # type: ignore[return-value]


def _steps(cams: list[Cam]) -> list[float]:
    """Per-frame step length between consecutive registered cameras (cams sorted by idx)."""
    return [_dist(a.centre, b.centre) / max(1, b.idx - a.idx) for a, b in zip(cams, cams[1:])]


def filter_trajectory(cams: list[Cam], far_factor: float = FAR_FACTOR, jump_factor: float = JUMP_FACTOR) -> SanityResult:
    cams = sorted(cams, key=lambda c: c.idx)
    dropped: list[dict] = []
    if len(cams) < 4:
        return SanityResult(kept=cams)
    for _ in range(MAX_ROUNDS):
        changed = False
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
            changed = True
            if len(cams) < 4:
                break
        # Rule 2: short runs between jumps.
        steps = _steps(cams)
        ms = median(steps) or 1e-12
        cut = [i for i, s in enumerate(steps) if s > jump_factor * ms]  # jump between cams[i] and cams[i+1]
        if cut:
            bounds = [0] + [i + 1 for i in cut] + [len(cams)]
            runs = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
            limit = max(SHORT_RUN_MIN, math.ceil(SHORT_RUN_FRAC * len(cams)))
            drop_idx: set[int] = set()
            biggest = max(runs, key=lambda r: r[1] - r[0])
            for run in runs:
                if run != biggest and (run[1] - run[0]) <= limit:
                    for c in cams[run[0]:run[1]]:
                        drop_idx.add(c.idx)
                        dropped.append({"name": c.name, "idx": c.idx, "reason": "short_run_between_jumps",
                                        "ratio": round(max(steps[i] for i in cut) / ms, 2)})
            if drop_idx:
                cams = [c for c in cams if c.idx not in drop_idx]
                changed = True
        if not changed or len(cams) < 4:
            break
    # Final statistics on what is kept.
    steps = _steps(cams) if len(cams) > 1 else []
    ms = median(steps) if steps else 0.0
    mc = _median_centre(cams) if cams else (0.0, 0.0, 0.0)
    md = median(_dist(c.centre, mc) for c in cams) if cams else 0.0
    jumps_left = [
        {"after": a.name, "before": b.name, "ratio": round(s / ms, 2)}
        for a, b, s in zip(cams, cams[1:], steps) if ms > 0 and s > jump_factor * ms
    ]
    return SanityResult(kept=cams, dropped=sorted(dropped, key=lambda d: d["idx"]), jumps_left=jumps_left,
                        median_step=ms, median_dist=md, max_step_ratio=round(max(steps) / ms, 2) if ms > 0 and steps else 0.0)
