#!/bin/bash
# Run a1_offline.sh on several 30-frame windows of the honka long capture; print registered counts per window.
# usage (host): nsdocker --gpu ns-w1-a1s namma-colmap:0.1 bash docker/checks/a1_sweep.sh
set -uo pipefail
for cfg in "1 1" "40 1" "80 1" "120 1" "160 1" "200 1" "240 1" "1 3" "100 3" "200 3"; do
  set -- $cfg
  W=/w/runs/w1-a1/sw_${1}_${2}
  START=$1 STEP=$2 N=30 W=$W bash /w/repo/docker/checks/a1_offline.sh > /tmp/sw.out 2>&1
  models=$(ls $W/sparse 2>/dev/null | wc -l)
  reg=$(for m in $W/sparse/*; do colmap model_analyzer --path $m 2>&1 | grep "Registered images" | sed 's/.*: //'; done | tr '\n' ',')
  echo "window start=$1 step=$2: models=$models registered_per_model=$reg  $(grep TIME /tmp/sw.out)"
done
