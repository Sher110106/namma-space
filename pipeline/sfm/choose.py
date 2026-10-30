"""Choice rule for the SfM race (FINAL_PLAN 2.3). Pure function on variant summaries; no I/O.

1. Drop crashed variants and any whose largest model registers under 60%.
2. Among variants at >= 95%, pick the lowest reprojection error, then the most 3D points.
3. If none reaches 95%, pick the highest registered share, flag it "partial", and list the missing time ranges.

Sanity checks (no camera jump over 5x the median step; mean track length >= 3) rank variants that pass above
variants that fail them, but never leave the race empty.
"""
from __future__ import annotations

FULL = 0.95
MIN_USABLE = 0.60
TRACK_MIN = 3.0
REPROJ_WARN = 1.2


def _sane(v: dict) -> bool:
    return v.get("track_len", 0.0) >= TRACK_MIN and not v.get("jumps_left")


def choose(variants: list[dict]) -> dict:
    """Return {chosen, why, partial, usable, warnings}. `chosen` is a variant name or None (nothing usable)."""
    warnings: list[str] = []
    ok = [v for v in variants if v.get("status") == "ok"]
    dropped_crash = [v["name"] for v in variants if v.get("status") != "ok"]
    usable = [v for v in ok if v.get("reg_frac", 0.0) >= MIN_USABLE]
    low = [f"{v['name']} {v['reg_frac']:.0%}" for v in ok if v.get("reg_frac", 0.0) < MIN_USABLE]
    notes = []
    if dropped_crash:
        notes.append("no result from " + ", ".join(dropped_crash))
    if low:
        notes.append("below 60%: " + ", ".join(low))
    if not usable:
        return {"chosen": None, "usable": False, "partial": True, "warnings": warnings,
                "why": "no variant registered 60% of the frames in one model" + (" (" + "; ".join(notes) + ")" if notes else "")}
    sane = [v for v in usable if _sane(v)]
    pool = sane or usable
    if not sane:
        warnings.append("every usable variant fails a sanity check (track length < 3 or a camera jump over 5x the median step)")
    full = [v for v in pool if v["reg_frac"] >= FULL]
    if full:
        best = min(full, key=lambda v: (v.get("reproj_px", 1e9), -v.get("n_pts", 0)))
        partial = False
        why = (f"{best['name']}: {best['reg_frac']:.1%} registered in one model, {best['reproj_px']:.2f} px, "
               f"{int(best['n_pts'])} points; lowest error among variants at 95% or more")
    else:
        best = max(pool, key=lambda v: (round(v["reg_frac"], 4), -v.get("reproj_px", 1e9)))
        partial = True
        why = (f"partial: no variant reached 95%; {best['name']} has the highest share, {best['reg_frac']:.1%} "
               f"in one model at {best['reproj_px']:.2f} px")
    if best.get("reproj_px", 0.0) > REPROJ_WARN:
        warnings.append(f"reprojection error {best['reproj_px']:.2f} px is above {REPROJ_WARN} px")
    if notes:
        why += "; " + "; ".join(notes)
    return {"chosen": best["name"], "usable": True, "partial": partial, "warnings": warnings, "why": why}
