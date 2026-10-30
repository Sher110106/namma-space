#!/bin/bash
# Optional: align on honka with the MoGe-2 cue, in namma-space-gpu:0.2 (needs moge importable and the weights in /w/cache/hf).
#   ssh sher@10.1.18.190 'nohup bash /home/sher/namma-space/repo/test/unit/align/run_moge_tyrone.sh > /home/sher/namma-space/runs/w2-logs/moge.log 2>&1 < /dev/null &'
set -u
NS=/home/sher/namma-space
R=$NS/runs/w2-honka-moge
mkdir -p $R/30_undist/sparse
[ -d $R/30_undist/sparse/0 ] || cp -r $NS/runs/smoke_honka/clean/sparse/0 $R/30_undist/sparse/0
ln -sfn /w/runs/smoke_honka/undist/images $R/30_undist/images
$NS/bin/nsdocker --gpu w2-moge namma-space-gpu:0.2 bash -c '
  python -c "import moge, torch; print(\"moge\", moge.__file__, \"cuda\", torch.cuda.is_available())"
  ls /w/cache/hf /w/cache/hf/hub 2>/dev/null | head
  python -m pipeline.align --run /w/runs/w2-honka-moge --moge
'
echo "== done"
