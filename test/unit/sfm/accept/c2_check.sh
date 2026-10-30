#!/bin/bash
# C2: Mip-NeRF 360 room: share registered and camera-centre RMSE after a similarity fit, vs dataset sparse/0.
# usage (inside namma-colmap:0.2): bash test/unit/sfm/accept/c2_check.sh <run_dir> <reference sparse/0>
set -euo pipefail
RUN=$1; REF=${2:-/w/data/mipnerf360/room/sparse/0}
cd /w/repo
python3 -m pipeline.sfm.eval_poses --ours "$RUN/30_undist/sparse" --ref "$REF" --out "$RUN/c2_poses.json"
python3 - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
sfm = json.load(open(f"{run}/20_sfm/sfm.json")); pose = json.load(open(f"{run}/c2_poses.json"))
share = sfm["variants"][[v["name"] for v in sfm["variants"]].index(sfm["chosen"])]["reg_frac"]
ok = share >= 0.98 and pose["rmse_pct_of_extent"] <= 1.0
print(f"RESULT C2 {'PASS' if ok else 'FAIL'} chosen={sfm['chosen']} registered={share:.3f} (need >= 0.98) "
      f"rmse={pose['rmse_pct_of_extent']:.3f}% of path extent (need <= 1%) n_common={pose['n_common']} scale={pose['scale']:.4f}")
PY
