#!/bin/bash
# A4: PLY (100k splats) -> SPZ (SH1) -> PLY with splat-transform, same splat count. Runs inside namma-space-gpu:0.2.
# usage (from the host): nsdocker ns-w1-a4 namma-space-gpu:0.2 bash docker/checks/a4_roundtrip.sh
set -euo pipefail
SRC=${SRC:-/w/runs/smoke_honka/gs_out5/ply/point_cloud_29999.ply}
OUT=${OUT:-/w/runs/w1-a4}
mkdir -p "$OUT"; cd "$OUT"
python /w/repo/docker/checks/cut_ply.py "$SRC" cut100k.ply 100000 0
python /w/repo/docker/checks/count_ply.py cut100k.ply
echo "== PLY -> SPZ v4, SH1"
time splat-transform -w cut100k.ply -H 1 --spz-version 4 out_sh1.spz
ls -l out_sh1.spz
echo "== SPZ -> PLY"
time splat-transform -w out_sh1.spz back.ply
python /w/repo/docker/checks/count_ply.py back.ply
echo "== SPZ v3 variant"
splat-transform -w cut100k.ply -H 1 --spz-version 3 out_sh1_v3.spz && ls -l out_sh1_v3.spz
splat-transform -w out_sh1_v3.spz back_v3.ply && python /w/repo/docker/checks/count_ply.py back_v3.ply
echo "== --info on the spz"
splat-transform out_sh1.spz --info null 2>&1 | head -30 || true
