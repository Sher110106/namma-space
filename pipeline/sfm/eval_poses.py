"""Compare our camera centres with a reference model (dataset poses) after a similarity fit (acceptance test C2).

usage: python3 -m pipeline.sfm.eval_poses --ours <model dir> --ref <reference model dir> [--out result.json]
RMSE is reported in the reference units and as a percentage of the reference path extent (diagonal of the
axis-aligned box around the reference camera centres). Only images present in both models are used.
Stdlib only: runs in namma-colmap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from . import colmapio, similarity


def evaluate(ours_dir: str, ref_dir: str) -> dict:
    _, ours = colmapio.read_model(ours_dir)
    _, ref = colmapio.read_model(ref_dir)
    a = {r.name: r.centre for r in ours}
    b = {r.name: r.centre for r in ref}
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} images in common")
    src = [a[n] for n in common]
    dst = [b[n] for n in common]
    s, r, t = similarity.fit(src, dst)
    err = similarity.rmse(s, r, t, src, dst)
    lo = [min(p[i] for p in dst) for i in range(3)]
    hi = [max(p[i] for p in dst) for i in range(3)]
    diag = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    per = sorted(math.dist(similarity.apply(s, r, t, p), q) for p, q in zip(src, dst))
    return {"n_ours": len(a), "n_ref": len(b), "n_common": len(common), "scale": s,
            "rmse_ref_units": err, "path_extent_diag": diag, "rmse_pct_of_extent": 100.0 * err / diag if diag else None,
            "median_err": per[len(per) // 2], "max_err": per[-1],
            "registered_share_of_ref": len(common) / len(b)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = evaluate(a.ours, a.ref)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
