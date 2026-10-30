"""ns rescale: apply a user scale as part of T_final.  Never retrains, never edits the PLY.

    python -m pipeline.align.rescale --run RUN --scale 1.08          # make everything 8% larger (multiplies the current user scale)
    python -m pipeline.align.rescale --run RUN --use-measures        # adopt the scale the tape measures imply (sets the total)
    python -m pipeline.align.rescale --run RUN --check               # compare tape measures with the automatic scale; change nothing
    python -m pipeline.align.rescale --run RUN --reset               # back to the automatic scale

Writes RUN/40_align/rescale.json (atomic replace).  The stage only records the correction; whoever builds the
scene applies it:

    T_user = diag(f, f, f, 1) about the world origin (the floor point under the camera-path centroid), f = factor_total.
    T_final_total = T_user @ T_final_from_export          (the floor offset first, then the scale; the floor stays at y = 0)

Positions in scene.json, the grid, bounds, spawn, bookmarks and heldout cameras are metres in the aligned frame, so
they scale by f; the splats get `splats.transform.scale = f` (the viewer multiplies the whole scene), and the grid
stage re-runs on CPU in seconds.  Eye height = H_cap x f.

Tape measures (input/meta.json  measures[{what, metres, method}]) validate; they never feed the automatic estimate.
`--check` reports the automatic scale's error against them first; `--use-measures` then adopts the tape scale, and
rescale.json keeps both numbers (auto and final).

Room size from the tape comparison: 70_grid/grid.json `room_extent_m` [x, z] if the grid stage has run (it may
carry `user_scale`, the factor it already applied; default 1), else the 2nd-98th percentile extent of the aligned
sparse points along X and Z (a rough stand-in, flagged in the output).  'length', 'long' or 'room_length' pair
with the larger extent; 'width', 'short' or 'room_width' with the smaller; 'ceiling' with the floor-to-ceiling height.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "namma-space/rescale@1"


def _jload(p: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def _jdump(p: str, obj: Dict[str, Any]) -> None:
    tmp = p + f".tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, p)


def T_user(f: float) -> List[List[float]]:
    return [[f, 0, 0, 0], [0, f, 0, 0], [0, 0, f, 0], [0, 0, 0, 1]]


def load(run_dir: str) -> Dict[str, Any]:
    r = _jload(os.path.join(run_dir, "40_align", "rescale.json"))
    if r:
        return r
    return {"schema": SCHEMA, "factor_total": 1.0, "T_user": T_user(1.0), "applied": [], "validation": None,
            "checked_against": None}


def factor(run_dir: str) -> float:
    """Current user scale factor (1.0 when never rescaled).  For W3, W4, W5 and root."""
    return float(load(run_dir)["factor_total"])


def compose_T_final(run_dir: str, T_prev: Optional[List[List[float]]] = None) -> List[List[float]]:
    """T_user @ T_prev as nested lists (T_prev from export.json, identity if None).  Pure Python: runs on the Mac."""
    f = factor(run_dir)
    S = T_user(f)
    P = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)] if T_prev is None else [[float(v) for v in row] for row in T_prev]
    return [[sum(S[i][k] * P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


# ---------------------------------------------------------------------------------------------
# tape measures against the model

def model_dimensions(run_dir: str) -> Tuple[Dict[str, float], str]:
    """Room dimensions in metres at the *automatic* scale, and where they came from."""
    f_now = factor(run_dir)
    g = _jload(os.path.join(run_dir, "70_grid", "grid.json"))
    a = _jload(os.path.join(run_dir, "40_align", "align.json")) or {}
    dims: Dict[str, float] = {}
    src = ""
    if g and g.get("room_extent_m"):
        ex = [float(v) for v in g["room_extent_m"]]
        applied = float(g.get("user_scale", 1.0))
        dims["long"], dims["short"] = max(ex) / applied, min(ex) / applied
        src = "grid.json room_extent_m" + (f" / user_scale {applied:g}" if applied != 1.0 else "")
    else:
        try:
            import numpy as np
            from . import colmap_io as cio
            m = cio.load_model(os.path.join(run_dir, "40_align", "sparse", "0"))
            P = m.xyz
            lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
            ex = [float(hi[0] - lo[0]), float(hi[2] - lo[2])]
            dims["long"], dims["short"] = max(ex), min(ex)
            src = "aligned sparse points, 2nd-98th percentile extent (rough; run the grid stage for wall-to-wall)"
        except Exception as ex_:   # includes ImportError on a Mac without numpy
            src = f"unavailable: {ex_}"
    ch = (a.get("floor") or {}).get("ceiling_height_m")
    if ch:
        dims["ceiling"] = float(ch)
    return dims, src


def classify(what: str) -> Optional[str]:
    w = what.lower()
    if "ceiling" in w:
        return "ceiling"
    if any(k in w for k in ("length", "long")):
        return "long"
    if any(k in w for k in ("width", "short", "breadth")):
        return "short"
    return None


def validate(run_dir: str) -> Dict[str, Any]:
    meta = _jload(os.path.join(run_dir, "input", "meta.json")) or {}
    a = _jload(os.path.join(run_dir, "40_align", "align.json"))
    if a is None:
        raise FileNotFoundError("40_align/align.json missing; run the align stage first")
    dims, src = model_dimensions(run_dir)
    rows: List[Dict[str, Any]] = []
    for m in meta.get("measures", []) or []:
        k = classify(str(m.get("what", "")))
        try:
            tape = float(m["metres"])
        except (KeyError, TypeError, ValueError):
            continue
        if k is None or k not in dims or tape <= 0:
            rows.append({"what": m.get("what"), "tape_m": tape, "model_m": None, "ratio": None,
                         "note": "not matched to a model dimension (use length, width or ceiling)"})
            continue
        rows.append({"what": m.get("what"), "tape_m": tape, "model_m": dims[k], "ratio": tape / dims[k],
                     "pair": k, "method": m.get("method")})
    ok = [r for r in rows if r["ratio"]]
    out: Dict[str, Any] = {"measures": rows, "model_dimensions_source": src, "n_used": len(ok),
                           "auto_scale": a["scale"]["value"], "auto_pct_68": a["scale"]["pct_68"]}
    if ok:
        gm = math.exp(sum(math.log(r["ratio"]) for r in ok) / len(ok))
        spread = max(abs(math.log(r["ratio"] / gm)) for r in ok) * 100.0
        out.update({"ratio_geomean": gm, "tape_scale": a["scale"]["value"] * gm,
                    "auto_error_pct": (1.0 / gm - 1.0) * 100.0,       # error of the automatic scale if the tape is right
                    "measure_spread_pct": spread,
                    "within_68": abs(math.log(gm)) <= a["scale"]["sigma_log"]})
    return out


# ---------------------------------------------------------------------------------------------
# applying

def record(run_dir: str, new_total: float, source: str, note: str = "", validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not (0.2 <= new_total <= 5.0):
        raise ValueError(f"scale factor {new_total:.3g} is outside 0.2-5: refusing")
    a = _jload(os.path.join(run_dir, "40_align", "align.json"))
    if a is None:
        raise FileNotFoundError("40_align/align.json missing; run the align stage first")
    r = load(run_dir)
    prev = float(r["factor_total"])
    r["schema"] = SCHEMA
    r["factor_total"] = new_total
    r["T_user"] = T_user(new_total)
    r.setdefault("applied", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "from": prev, "to": new_total,
                                        "source": source, "note": note})
    r["auto"] = {"scale": a["scale"]["value"], "pct_68": a["scale"]["pct_68"], "h_cap_m": a["h_cap_m"]}
    r["scale_final"] = a["scale"]["value"] * new_total
    r["h_cap_m_final"] = a["h_cap_m"] * new_total
    if validation is not None:
        r["validation"] = validation
        if validation.get("n_used"):
            r["checked_against"] = "tape"
    os.makedirs(os.path.join(run_dir, "40_align"), exist_ok=True)
    _jdump(os.path.join(run_dir, "40_align", "rescale.json"), r)
    return r


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.align.rescale", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scale", type=float, help="multiply the current user scale by this (1.08 = 8%% larger)")
    g.add_argument("--set-total", type=float, help="set the total user scale factor (relative to the automatic scale)")
    g.add_argument("--use-measures", action="store_true", help="adopt the scale implied by input/meta.json tape measures")
    g.add_argument("--check", action="store_true", help="compare tape measures with the automatic scale; change nothing")
    g.add_argument("--reset", action="store_true", help="remove the user scale (factor 1)")
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run)
    try:
        if a.check or a.use_measures:
            v = validate(run)
            if not v.get("n_used"):
                print("no usable tape measures in input/meta.json (measures[{what, metres, method}] with length, width or ceiling)")
                print("NS_RESULT " + json.dumps({"exit": 0, "changed": False, "validation": v}))
                return 0
            print(f"tape vs automatic scale: {v['n_used']} measure(s), automatic scale off by {v['auto_error_pct']:+.1f}% "
                  f"(68% interval +/-{v['auto_pct_68']:.0f}%, {'inside' if v['within_68'] else 'outside'}); "
                  f"model dimensions from {v['model_dimensions_source']}")
            if a.check:
                r = load(run)
                r["validation"] = v
                r["checked_against"] = "tape"
                r.setdefault("auto", {})
                os.makedirs(os.path.join(run, "40_align"), exist_ok=True)
                _jdump(os.path.join(run, "40_align", "rescale.json"), r)
                print("NS_RESULT " + json.dumps({"exit": 0, "changed": False, "validation": v}))
                return 0
            r = record(run, v["ratio_geomean"], "measures", f"{v['n_used']} tape measure(s)", v)
        elif a.reset:
            r = record(run, 1.0, "reset")
        elif a.set_total is not None:
            r = record(run, a.set_total, "user", "set total")
        else:
            r = record(run, factor(run) * a.scale, "user", f"x{a.scale:g}")
    except (FileNotFoundError, ValueError) as ex:
        print(f"RESCALE FAIL: {ex}")
        print("NS_RESULT " + json.dumps({"exit": 20, "reason": str(ex)}))
        return 20
    print(f"user scale {r['applied'][-1]['from']:.4g} -> {r['factor_total']:.4g}; final scale {r['scale_final']:.4g} m per SfM unit "
          f"(automatic {r['auto']['scale']:.4g}); H_cap {r['h_cap_m_final']:.2f} m")
    print("rebuild the grid and the bundle to apply it (no retraining); T_final = T_user @ T_final(export)")
    print("NS_RESULT " + json.dumps({"exit": 0, "changed": True, "factor_total": r["factor_total"], "scale_final": r["scale_final"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
