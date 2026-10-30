"""Freeze a reference run's report.json into test/expected/report.json with tolerance bands (standard library only).

    python3 test/e2e/freeze_expected.py runs/rh-honka/report.json --case honka [--out test/expected/report.json]
        [--command "..."] [--note "..."]

Each value gets a band from TOL below: ('abs', d) gives [v - d, v + d], ('rel', f) gives [v (1 - f), v (1 + f)],
('lo', d) only a lower bound v - d, ('hi', f) only an upper bound v (1 + f), ('cap', c) an upper bound max(v + 2, c). Bands reflect run-to-run variation
(gsplat training is not deterministic; SfM is seeded but GPU matching is not) and a shared GPU (timings).
They are judgement calls, written next to each value, not measured spreads. Other cases in the file are kept.
check_expected.py --expected reads the result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_expected import resolve  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "expected" / "report.json"

# id, path in report.json, tolerance kind, amount, why
TOL = [
    ("frames.n_kept", "frames.n_kept", "abs", 0, "same input, deterministic ingest"),
    ("sfm.reg_frac", "sfm.variants[chosen].reg_frac", "abs", 0.04, "mapper order and GPU matching vary"),
    ("sfm.reproj_px", "sfm.variants[chosen].reproj_px", "abs", 0.15, ""),
    ("sfm.n_pts", "sfm.variants[chosen].n_pts", "rel", 0.25, ""),
    ("align.up_resid_deg", "align.up_resid_deg", "cap", 5.0, "varies run to run (0.0 and 3.3 deg on the same 60 frames); "
     "upper bound max(value + 2, 5)"),
    ("align.h_cap_m", "align.h_cap_m", "abs", 0.15, "1.40 m by construction with the camera-height cue alone"),
    ("align.scale.pct_68", "align.scale.pct_68", "abs", 1.0, ""),
    ("train.psnr", "train.variants[chosen].psnr", "abs", 0.6, "gsplat runs vary (plan: quote +-0.2 dB); SfM varies too"),
    ("train.ssim", "train.variants[chosen].ssim", "abs", 0.03, ""),
    ("train.lpips", "train.variants[chosen].lpips", "abs", 0.03, ""),
    ("train.cc_psnr", "train.variants[chosen].cc_psnr", "abs", 0.6, ""),
    ("train.n_gs", "train.variants[chosen].n_gs", "rel", 0.10, "MCMC fills its cap (1.5 M) on a full run"),
    ("export.sd.count", "export.sd.count", "rel", 0.10, ""),
    ("export.sd.mb", "export.sd.mb", "rel", 0.20, ""),
    ("export.sd.psnr", "export.sd.psnr", "abs", 0.7, ""),
    ("grid.walkable_m2", "grid.walkable_m2", "rel", 0.25, ""),
    ("grid.path_coverage", "grid.path_coverage", "lo", 0.05, ""),
    ("grid.room_extent_m.0", "grid.room_extent_m[0]", "abs", 0.3, "longer wall-to-wall extent"),
    ("grid.room_extent_m.1", "grid.room_extent_m[1]", "abs", 0.3, "shorter wall-to-wall extent"),
    ("walls.penetrations", "walls.penetrations", "abs", 0, "must stay 0"),
    ("viewer.fps_mean", "viewer.devices[*].fps_mean", "lo", 5.0, "M1 headless Chrome, vsync-capped at 60"),
    ("viewer.first_frame_s", "viewer.devices[*].first_frame_s", "hi", 2.0, "localhost"),
    ("timings.total", "timings.total", "hi", 0.75, "shared GPU: queueing for a slot dominates the spread"),
]


def band(v: float, kind: str, amt: float):
    if kind == "abs":
        return [round(v - amt, 4), round(v + amt, 4)]
    if kind == "rel":
        return [round(v * (1 - amt), 4), round(v * (1 + amt), 4)]
    if kind == "lo":
        return [round(v - amt, 4), None]
    if kind == "hi":
        return [None, round(v * (1 + amt), 4)]
    if kind == "cap":
        return [None, round(max(v + 2.0, amt), 4)]
    raise ValueError(kind)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--command", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--metric-check", help="JSON from pipeline.align.metric_check to keep with the case (not compared)")
    a = ap.parse_args(argv)
    rep = json.loads(Path(a.report).read_text())
    vals = []
    for vid, path, kind, amt, why in TOL:
        got = [x for x in resolve(rep, path) if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if not got:
            continue
        v = min(got) if kind == "lo" else max(got)
        vals.append({"id": vid, "path": path, "measured": v, "range": band(float(v), kind, amt),
                     "tolerance": f"{kind} {amt}", **({"why": why} if why else {})})
    timings = rep.get("timings") or {}
    out_p = Path(a.out)
    doc = json.loads(out_p.read_text()) if out_p.exists() else {}
    doc.setdefault("schema", "namma-space/expected@1")
    doc["doc"] = ("Measured values of reference runs with tolerance bands. Compare a new run with: python3 "
                  "test/e2e/check_expected.py runs/<name>/report.json --expected test/expected/report.json --case <case>. "
                  "'measured' is what the reference run produced; 'range' is the accepted band ([lo, hi], null = open). "
                  "Bands are judgement calls (see test/e2e/freeze_expected.py), not measured spreads. "
                  "Plan thresholds are in gates.json.")
    doc.setdefault("cases", {})
    doc["cases"][a.case] = {
        "source_run": rep.get("name"), "git": rep.get("git"), "frozen": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": a.command, "note": a.note, "quick": bool((rep.get("train") or {}).get("quick")),
        "timings_s": timings, "values": vals,
    }
    if a.metric_check:
        mc = json.loads(Path(a.metric_check).read_text())
        doc["cases"][a.case]["metric_check"] = {k: mc.get(k) for k in ("n_matched", "n_fit", "scale_error_pct",
                                                                         "scale_error_pct_plain", "up_error_deg",
                                                                         "residual_m", "note")}
    out_p.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote case {a.case} ({len(vals)} values) to {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
