#!/bin/bash
# Run the align acceptance tests on tyrone and produce the D3 / D4 artefacts.
#   ssh sher@10.1.18.190 'nohup bash /home/sher/namma-space/repo/test/unit/align/run_tyrone.sh > /home/sher/namma-space/runs/w2-logs/align_tyrone.log 2>&1 < /dev/null &'
# Outputs under /home/sher/namma-space/runs/w2-honka-align and w2-mip (about 20 MB).
set -u
NS=/home/sher/namma-space
SM=$NS/runs/smoke_honka
H=$NS/runs/w2-honka-align
M=$NS/runs/w2-mip
mkdir -p $H/30_undist/sparse $M/30_undist
[ -d $H/30_undist/sparse/0 ] || cp -r $SM/clean/sparse/0 $H/30_undist/sparse/0
ln -sfn /w/runs/smoke_honka/undist/images $H/30_undist/images
[ -d $M/30_undist/sparse ] || cp -r $NS/data/mipnerf360/room/sparse $M/30_undist/sparse
ln -sfn /w/data/mipnerf360/room/images $M/30_undist/images
echo "== align on honka (smoke model as 30_undist), then Mip room (D3), then the unit tests"
$NS/bin/nsdocker w2-align namma-space-gpu:0.1 bash -c '
  python -m pipeline.align --run /w/runs/w2-honka-align
  echo "--- mip room (no user height, generic 1.40 m)"
  python -m pipeline.align --run /w/runs/w2-mip
  echo "--- unit tests"
  python -m unittest discover -s test/unit/align -p test_align.py -v 2>&1
'
echo "== D4: colmap model_analyzer before and after (COLMAP 4.2.1)"
$NS/bin/nsdocker w2-d4 namma-colmap:0.1 bash -c '
  for d in /w/runs/w2-honka-align/30_undist/sparse/0 /w/runs/w2-honka-align/40_align/sparse/0; do
    echo "### $d"; colmap model_analyzer --path $d 2>&1 | grep -E "Rigs|Cameras|Frames|Registered|Points|Observations|Mean track|Mean obs|Mean reproj"
  done
'
echo "== done"
