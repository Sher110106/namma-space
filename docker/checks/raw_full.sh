#!/bin/bash
# Raw (driver-less) ALIKED sequential + loop detection + mapper on a whole image directory; used to compare with the driver.
# env: IMG, W
set -euo pipefail
C=/w/cache/colmap
IMG=${IMG:-/w/data/mushroom/room_datasets/honka/iphone/long_capture/images}
W=${W:-/w/runs/w1-c1raw}
rm -rf "$W"; mkdir -p "$W/sparse"
t0=$(date +%s)
colmap feature_extractor --database_path $W/db.db --image_path $IMG \
  --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 \
  --FeatureExtraction.type ALIKED_N32 --AlikedExtraction.max_num_features 4096 --FeatureExtraction.max_image_size 1920 \
  --AlikedExtraction.n32_model_path $C/aliked-n32.onnx > $W/extract.log 2>&1
t1=$(date +%s)
colmap sequential_matcher --database_path $W/db.db --FeatureMatching.type ALIKED_LIGHTGLUE \
  --AlikedMatching.lightglue_model_path $C/aliked-lightglue.onnx \
  --SequentialMatching.overlap 20 --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 1 --SequentialMatching.vocab_tree_path $C/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin > $W/match.log 2>&1
t2=$(date +%s)
colmap mapper --database_path $W/db.db --image_path $IMG --output_path $W/sparse > $W/map.log 2>&1
t3=$(date +%s)
echo "TIME extract $((t1-t0))s match $((t2-t1))s map $((t3-t2))s"
for m in $W/sparse/*; do echo $m; colmap model_analyzer --path $m 2>&1 | grep -E "Registered images|Points:|Mean track|Mean reproj"; done
