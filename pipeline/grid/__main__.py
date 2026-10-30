"""python -m pipeline.grid --run <run_dir> [options]

Exit codes: 0 good, 20 input problem or crash, 30 fallback mode used or no spawn (the bundle exists but needs review).
The last stdout line is NS_RESULT {json} for the runner.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from .stage import run_stage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.grid", description="Walkable grid, minimap, spawn, bookmarks, tour.")
    ap.add_argument("--run", required=True, help="run directory (contains 40_align, 60_export, ...)")
    ap.add_argument("--out", help="output directory (default <run>/70_grid)")
    ap.add_argument("--ply", help="cleaned PLY (default: found under 60_export)")
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tfinal", choices=["auto", "skip"], default="auto",
                    help="auto: apply T_final from 60_export/export.json to the PLY and cameras; skip: they are already final")
    ap.add_argument("--force-mode", choices=["grid", "corridor", "bounds"], help="skip the fallback ladder")
    ap.add_argument("--floor-y", type=float, default=0.0)
    ap.add_argument("--h-cap", type=float, help="override H_cap (metres, final frame)")
    ap.add_argument("--activations", choices=["auto", "raw3dgs", "linear"], default="auto",
                    help="PLY opacity/scale encoding: raw3dgs = logit/log (default for gsplat PLYs, auto-detected)")
    ap.add_argument("--no-tour", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    summary: dict = {}
    try:
        rc = run_stage(a.run, out_dir=a.out, ply=a.ply, cell=a.cell, seed=a.seed, tfinal=a.tfinal,
                       force_mode=a.force_mode, floor_y=a.floor_y, make_tour=not a.no_tour,
                       h_cap_override=a.h_cap, activations=a.activations, quiet=a.quiet, summary=summary)
    except Exception as ex:                      # anything unexpected is an infrastructure problem (exit 20)
        traceback.print_exc()
        summary = {"exit": 20, "reason": f"{type(ex).__name__}: {ex}"}
        rc = 20
    print("NS_RESULT " + json.dumps(summary))     # the runner reads the last stdout line
    return rc


if __name__ == "__main__":
    sys.exit(main())
