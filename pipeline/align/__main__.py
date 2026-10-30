"""python -m pipeline.align --run RUN_DIR [--height-cm N] [--moge] [--seed N]

Reads RUN_DIR/30_undist/{images, sparse}, RUN_DIR/input/meta.json (height_cm, optional).
Writes RUN_DIR/40_align/{sparse/0, images -> ../30_undist/images, align.json, figures/level_before_after.png}.
Exit: 0 when 40_align is written (align.json "gate" says whether the quality gate passed: up cues within 5 deg,
a floor found, H_cap x scale inside 1.0-1.8 m; NS_RESULT carries gate_failed so the runner can end the run with 30)
| 30 only with --strict and a failed gate | 20 infrastructure (no model).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import stage


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.align", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--height-cm", type=float, default=None, help="user height; default from input/meta.json, else a generic 1.40 m phone height")
    ap.add_argument("--moge", action="store_true", help="add the MoGe-2 metric-depth cue (needs namma-space-gpu:0.2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 30 when the quality gate fails")
    a = ap.parse_args(argv)
    try:
        out = stage.run_stage(a.run, height_cm=a.height_cm, seed=a.seed, make_figure=not a.no_figure, moge=a.moge)
    except (FileNotFoundError, ValueError) as ex:
        print(f"ALIGN FAIL (infrastructure): {ex}")
        print("NS_RESULT " + json.dumps({"exit": stage.EXIT_INFRA, "reason": str(ex)}))
        return stage.EXIT_INFRA
    flags = out["flags"]
    gate = out["gate"]["reasons"]
    sc = out["scale"]
    print(f"up residual {out['up_resid_deg']:.2f} deg, cue spread {out['up']['cue_spread_deg']:.2f} deg; "
          f"floor support {out['floor']['support']} ({out['floor']['method']}); H_cap {out['h_cap_m']:.2f} m; "
          f"scale {sc['value']:.4g} m/unit (68%: +/-{sc['pct_68']:.0f}%, {sc['method']})")
    for f in flags:
        print("FLAG:", f)
    print("NS_RESULT " + json.dumps({"exit": stage.EXIT_GATE if (gate and a.strict) else 0, "gate_failed": bool(gate), "flags": flags, "scale": sc["value"],
                                      "pct_68": sc["pct_68"], "h_cap_m": out["h_cap_m"], "up_resid_deg": out["up_resid_deg"]}))
    return stage.EXIT_GATE if (gate and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
