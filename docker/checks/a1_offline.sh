#!/bin/bash
# A1: ALIKED_N32 extraction + sequential ALIKED_LIGHTGLUE matching with loop detection + mapper on 30 honka frames,
# using only files from the host cache (explicit *_model_path options; XDG_CACHE_HOME points at an empty dir so any
# implicit download would fail). Runs inside namma-colmap:0.1 with a GPU.
# usage (host): nsdocker --gpu ns-w1-a1 namma-colmap:0.1 bash docker/checks/a1_offline.sh
set -euo pipefail
C=/w/cache/colmap
SRCDIR=${SRCDIR:-/w/data/mushroom/room_datasets/honka/iphone/long_capture/images}
W=${W:-/w/runs/w1-a1}
START=${START:-100}; N=${N:-30}; STEP=${STEP:-1}
rm -rf "$W"; mkdir -p "$W/images" "$W/sparse" "$W/emptycache"
i=0
for f in $(ls "$SRCDIR" | sed -n "${START},\$p"); do
  [ $((i % STEP)) -eq 0 ] && [ $(( i / STEP )) -lt "$N" ] && cp "$SRCDIR/$f" "$W/images/"
  i=$((i+1)); [ $((i / STEP)) -ge "$N" ] && break
done
echo "images: $(ls $W/images | wc -l)"
export XDG_CACHE_HOME=$W/emptycache
t0=$(date +%s)
colmap feature_extractor --database_path $W/db.db --image_path $W/images \
  --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 \
  --FeatureExtraction.type ALIKED_N32 --AlikedExtraction.max_num_features 4096 --FeatureExtraction.use_gpu 1 \
  --AlikedExtraction.n32_model_path $C/aliked-n32.onnx > $W/extract.log 2>&1
t1=$(date +%s)
colmap sequential_matcher --database_path $W/db.db --FeatureMatching.type ALIKED_LIGHTGLUE --FeatureMatching.use_gpu 1 \
  --AlikedMatching.lightglue_model_path $C/aliked-lightglue.onnx \
  --SequentialMatching.overlap 20 --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 1 --SequentialMatching.vocab_tree_path $C/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin > $W/match.log 2>&1
t2=$(date +%s)
colmap mapper --database_path $W/db.db --image_path $W/images --output_path $W/sparse > $W/map.log 2>&1
t3=$(date +%s)
echo "TIME extract $((t1-t0))s match $((t2-t1))s map $((t3-t2))s"
grep -c "Downloading file" $W/extract.log $W/match.log $W/map.log || true
colmap model_analyzer --path $W/sparse/0 2>&1 | grep -E "Registered|Points|Mean reproj|Mean track" || true
