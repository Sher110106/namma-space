#!/bin/bash
# Download the model files the SfM stage reads from the host cache and check their SHA-256 (docker/IMAGES.md).
# Usage (on the GPU host): docker/fetch_cache.sh [ROOT]   (ROOT defaults to $NS_ROOT, else $HOME/namma-space)
# MoGe weights (optional scale cue) are not fetched here: run docker/checks/a6_moge.py once with HF_HOME=/w/cache/hf.
set -euo pipefail
ROOT=${1:-${NS_ROOT:-$HOME/namma-space}}
D="$ROOT/cache/colmap"
mkdir -p "$D" "$ROOT/cache/hf"
R13=https://github.com/colmap/colmap/releases/download/3.13.0
R11=https://github.com/colmap/colmap/releases/download/3.11.1
fetch() {   # url sha256
  local f; f=$(basename "$1")
  if [ ! -f "$D/$f" ]; then curl -fL --retry 3 -o "$D/$f.part" "$1" && mv "$D/$f.part" "$D/$f"; fi
  echo "$2  $D/$f" | sha256sum -c -
}
fetch $R13/aliked-n32.onnx a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c
fetch $R13/aliked-lightglue.onnx b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d
fetch $R13/sift-lightglue.onnx e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e
fetch $R13/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin 65619481045b8f933268f10c31ad180eb1ee7881182873efe0f5753972ef6a20
fetch $R11/vocab_tree_faiss_flickr100K_words256K.bin 96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc
echo "cache ready in $D"
