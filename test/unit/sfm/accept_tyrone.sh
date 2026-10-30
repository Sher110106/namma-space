#!/bin/bash
# Acceptance tests for WP1 (A1-A6, C1-C5). Run ON TYRONE from the repo directory:
#   cd /home/sher/namma-space/repo && bash test/unit/sfm/accept_tyrone.sh <test> [<test> ...]
# Every container goes through nsdocker. Each test prints one "RESULT <id> PASS|FAIL <details>" line.
# Tests: a1 a2 a3 a4 a5 a6 c1 c2 c3a c3a2 c3b c3c c4 c5 prep3
set -uo pipefail
NS=/home/sher/namma-space/bin/nsdocker
W=/home/sher/namma-space
COL=namma-colmap:0.2
GPUI=namma-space-gpu:0.2
IMG_HONKA=/w/data/mushroom/room_datasets/honka/iphone/long_capture/images
FILTER='^(==========|== CUDA|CUDA Version|Container image|This container|By pulling|A copy of|WARNING|   Use the NVIDIA|$|https)'
quiet() { grep -v -E "$FILTER"; }

a1() {  # offline ALIKED + LightGlue + loop detection + mapper on 30 honka frames; nvidia-smi sees the process
  ( START=${A1_START:-160}; N=30
    # sample the GPU from the host while the container runs: a PID known to `docker top` AND listed by nvidia-smi
    ( while true; do
        pids=$(docker top ns-w1-a1 2>/dev/null | awk 'NR>1{print $2}' | tr '\n' ' ')
        for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
          case " $pids " in *" $p "*) echo "GPUPID $p $(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader | grep "^$p,")"; esac
        done; sleep 0.25; done ) > $W/runs/w1-a1.gpu 2>&1 &
    SAMP=$!
    $NS --gpu ns-w1-a1 $COL bash -c "START=$START N=$N bash docker/checks/a1_offline.sh" 2>&1 | quiet | tee $W/runs/w1-a1.out
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
    reg=$(grep "Registered images" $W/runs/w1-a1.out | head -1 | sed 's/.*: //')
    gpu=$(sort -u $W/runs/w1-a1.gpu | head -3 | tr '\n' ';')
    ok=FAIL; [ "${reg:-0}" -ge 28 ] && [ -n "$gpu" ] && ok=PASS
    echo "RESULT A1 $ok registered=${reg:-0}/30 (window start $START) gpu_evidence=[$gpu] downloads=$(grep -c 'Downloading file' $W/runs/w1-a1/*.log | tr '\n' ' ') network=on, explicit *_model_path (see NOTES)" )
}
a2() { $NS ns-w1-a2 $COL colmap version 2>&1 | quiet | head -2; echo "RESULT A2 $(docker image inspect $COL --format 'id={{.Id}}')"; }
a3() {  # ffmpeg filters + a 1,000-step gsplat run with --antialiased --use_bilateral_grid on 30 frames
  $NS ns-w1-a3f $GPUI bash -c "ffmpeg -hide_banner -filters 2>/dev/null | grep -E ' (zscale|tonemap) '" 2>&1 | quiet
  $NS --gpu ns-w1-a3 $GPUI bash docker/checks/a3_gsplat.sh 2>&1 | quiet | tail -25
}
a4() { $NS ns-w1-a4 $GPUI bash docker/checks/a4_roundtrip.sh 2>&1 | quiet; }
a5() { echo "RESULT A5 new_layers_MB=$(docker history --format '{{.Size}} {{.CreatedBy}}' $GPUI | head -3 | tr '\n' ' ' | cut -c1-200) free=$(df -h / | tail -1 | awk '{print $4}')"; }
a6() { $NS --gpu ns-w1-moge $GPUI bash -c "HF_HUB_OFFLINE=1 python docker/checks/a6_moge.py" 2>&1 | quiet | tail -12; }
run_sfm() {  # run_sfm <run name> <container name> [extra driver args]
  local run=$1 name=$2; shift 2
  $NS --gpu $name $COL python3 -m pipeline.sfm --run /w/runs/$run "$@" 2>&1 | quiet | tee $W/runs/$run.out
  echo "EXIT ${PIPESTATUS[0]}"
}
c1() {
  $NS ns-w1-prep $GPUI bash docker/checks/prep_run.sh w1-honka $IMG_HONKA 2>&1 | quiet
  s=$(date +%s); run_sfm w1-honka ns-w1-c1; e=$(date +%s)
  python3 - <<PY
import json
s=json.load(open("$W/runs/w1-honka/20_sfm/sfm.json"))
ch=[v for v in s["variants"] if v["name"]==s["chosen"]][0]
ok = ch["reg_frac"]>=0.90 and ch["reproj_px"]<=1.2 and ($e-$s)<=1500
print("RESULT C1", "PASS" if ok else "FAIL", f"chosen={s['chosen']} reg={ch['reg_frac']:.3f} reproj={ch['reproj_px']} race_s={$e-$s} unregistered={len(s['unregistered'])}")
PY
}
c2() {
  $NS --gpu ns-w1-c2 $COL bash -c "bash docker/checks/prep_subset.sh w1-mip /w/data/mipnerf360/room/images_4 1 400 >/dev/null && python3 -m pipeline.sfm --run /w/runs/w1-mip && bash test/unit/sfm/accept/c2_check.sh /w/runs/w1-mip" 2>&1 | quiet | tee $W/runs/w1-mip.out | tail -30
}
prep3() { $NS ns-w1-prep3 $GPUI python test/unit/sfm/accept/make_c3_runs.py /w/runs 2>&1 | quiet; }
c3a()  { run_sfm w1-c3a ns-w1-c3a --variants A --no-c; $NS ns-w1-c3a-chk $COL python3 test/unit/sfm/accept/c3_check.py a /w/runs/w1-c3a 2>&1 | quiet; }
c3a2() { run_sfm w1-c3a2 ns-w1-c3a2 --variants A --no-c; $NS ns-w1-c3a2-chk $COL python3 test/unit/sfm/accept/c3_check.py a2 /w/runs/w1-c3a2 2>&1 | quiet; }
c3b()  { run_sfm w1-c3b ns-w1-c3b --variants A,D --no-c; $NS ns-w1-c3b-chk $COL python3 test/unit/sfm/accept/c3_check.py b /w/runs/w1-c3b 2>&1 | quiet; }
c3c() {  # 120-frame race with B's colmap processes killed repeatedly from 15 s on
  $NS --gpu ns-w1-c3c $COL bash -c "
    python3 test/unit/sfm/accept/kill_variant.py B 15 400 > /w/runs/w1-c3c.kill 2>&1 &
    K=\$!
    rm -rf /w/runs/w1-c3c/20_sfm /w/runs/w1-c3c/30_undist
    mkdir -p /w/runs/w1-c3c-sub/10_frames/images; rm -f /w/runs/w1-c3c-sub/10_frames/images/*
    for f in \$(ls /w/runs/w1-c3c/10_frames/images | sed -n '100,219p'); do ln -s \$(readlink -f /w/runs/w1-c3c/10_frames/images/\$f) /w/runs/w1-c3c-sub/10_frames/images/\$f; done
    python3 -m pipeline.sfm --run /w/runs/w1-c3c-sub --no-c; echo EXIT \$?
    kill \$K 2>/dev/null; cat /w/runs/w1-c3c.kill | tail -3
    KILLED=B python3 test/unit/sfm/accept/c3_check.py c /w/runs/w1-c3c-sub" 2>&1 | quiet | tee $W/runs/w1-c3c.out | tail -30
}
c4() { $NS ns-w1-c4 $COL python3 -m pipeline.sfm.check_flags 2>&1 | quiet | sed 's/^C4/RESULT C4/'; }
c5() {  # B's log: CPU SIFT, ONNX LightGlue matcher, no SiftGPU. Needs a finished run with variant B (c1).
  L=$W/runs/w1-honka/20_sfm/B/log.txt
  cpu=$(grep -c "Creating SIFT CPU feature extractor" $L); lg=$(grep -c "SIFT_LIGHTGLUE" $L); onnx=$(grep -c "sift-lightglue.onnx" $L)
  gpu=$(grep -c -i "siftgpu" $L); usegpu0=$(grep -c "FeatureExtraction.use_gpu 0" $L)
  ok=FAIL; [ "$cpu" -gt 0 ] && [ "$lg" -gt 0 ] && [ "$onnx" -gt 0 ] && [ "$gpu" -eq 0 ] && ok=PASS
  echo "RESULT C5 $ok cpu_sift_lines=$cpu use_gpu0=$usegpu0 SIFT_LIGHTGLUE_mentions=$lg onnx_path_mentions=$onnx siftgpu_mentions=$gpu"
}
for t in "$@"; do echo "=== $t"; $t; done
