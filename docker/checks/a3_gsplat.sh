#!/bin/bash
# A3: 1,000-step gsplat run with --antialiased --use_bilateral_grid on 30 honka frames; no NaN, writes a PLY.
# Needs a finished SfM run with 30 undistorted frames: /w/runs/w1-a1d/30_undist (made by the driver on honka frames 160-189).
set -uo pipefail
D=${D:-/w/runs/w1-a1d/30_undist}; OUT=${OUT:-/w/runs/w1-a3}
rm -rf "$OUT"; mkdir -p "$OUT"
cd /opt/gsplat/examples
python simple_trainer.py default --data_dir "$D" --data_factor 1 --result_dir "$OUT" --max_steps 1000 \
  --antialiased --use_bilateral_grid --save_ply --ply_steps 1000 --eval_steps 1000 --save_steps 1000 \
  --disable_viewer --test_every 8 > "$OUT/train.log" 2>&1
rc=$?
tail -5 "$OUT/train.log"
ply=$(ls "$OUT"/ply/*.ply 2>/dev/null | head -1)
nan=$(grep -c -i -E "\bnan\b" "$OUT/train.log")
echo "rc=$rc ply=$ply nan_mentions=$nan"
python - <<PY
import glob, numpy as np
from plyfile import PlyData
f = glob.glob("$OUT/ply/*.ply")
if f:
    v = PlyData.read(f[0])["vertex"].data
    a = np.stack([v[n] for n in v.dtype.names if n in ("x","y","z","opacity","scale_0","rot_0")], 1)
    print("ply splats", len(v), "all finite:", bool(np.isfinite(a).all()))
PY
cat "$OUT"/stats/*.json 2>/dev/null | head -c 600
echo
