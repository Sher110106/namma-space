"""python -m pipeline.ingest --run RUN_DIR [--frames DIR | --video FILE ...] [options]

Reads RUN_DIR/input/ (videos, or a frames folder) unless --video or --frames says otherwise,
writes RUN_DIR/10_frames/{images/000001.jpg ..., frames.json}.

Exit codes: 0 good | 10 capture problem (too short, too few sharp frames, cannot decode; the
time ranges to reshoot are printed) | 20 infrastructure problem (no disk, no ffmpeg).
The last stdout line is  NS_RESULT {json}  for the runner.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import List, Optional

from . import stage


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.ingest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (contains input/)")
    ap.add_argument("--video", action="append", help="video file (repeat for several takes); default: input/*.mp4 ...")
    ap.add_argument("--frames", help="frames folder: apply EXIF orientation, strip EXIF, re-encode, keep all (<= cap)")
    ap.add_argument("--select", action="store_true", help="in --frames mode, also apply the blur and duplicate filters")
    ap.add_argument("--max-frames", type=int, default=None, help=f"cap on kept frames (default {stage.CAP}; 600 with --take-b)")
    ap.add_argument("--take-b", action="store_true", help=f"take B allowance: cap {stage.CAP_TAKE_B}")
    ap.add_argument("--long-side", type=int, default=stage.LONG_SIDE, help="longest image side in px (default 1920)")
    ap.add_argument("--min-duration", type=float, default=stage.MIN_DURATION_S, help="shortest video in seconds (default 45)")
    ap.add_argument("--force", action="store_true", help="carry on with a short video or few frames (warnings only)")
    ap.add_argument("--tonemap", choices=["auto", "zscale", "numpy"], default="auto", help="HDR tone map path (default: zscale if ffmpeg has it)")
    ap.add_argument("--min-free-gb", type=float, default=3.0, help="refuse to start below this much free disk (default 3)")
    ap.add_argument("--keep-candidates", action="store_true", help="keep the candidate JPEGs (scratch) in 10_frames/")
    a = ap.parse_args(argv)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ingest: ffmpeg/ffprobe not found", file=sys.stderr)
        return stage.EXIT_INFRA
    cap = a.max_frames or (stage.CAP_TAKE_B if a.take_b else stage.CAP)

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        res = stage.run_stage(a.run, videos=a.video, frames_dir=a.frames, cap=cap, long_side=a.long_side,
                              min_duration=a.min_duration, force=a.force, select_in_frames=a.select,
                              keep_candidates=a.keep_candidates, tonemap=a.tonemap, min_free_gb=a.min_free_gb, log=log)
    except stage.CaptureProblem as ex:
        print(f"INGEST FAIL (capture problem): {ex.reason}", flush=True)
        for r in ex.ranges:
            print(f"  reshoot {r[0]:.1f} s to {r[1]:.1f} s", flush=True)
        print("NS_RESULT " + json.dumps({"exit": stage.EXIT_CAPTURE, "reason": ex.reason, "reshoot_s": ex.ranges}))
        return stage.EXIT_CAPTURE
    except stage.InfraProblem as ex:
        print(f"INGEST FAIL (infrastructure): {ex}", flush=True)
        print("NS_RESULT " + json.dumps({"exit": stage.EXIT_INFRA, "reason": str(ex)}))
        return stage.EXIT_INFRA
    print("NS_RESULT " + json.dumps({"exit": 0, "n_kept": res["n_kept"], "n_cand": res["n_cand"], "dropped": res["dropped"],
                                      "hdr": res["hdr"], "warnings": res["warnings"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
