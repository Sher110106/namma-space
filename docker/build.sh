#!/bin/bash
# Rebuild the project images on tyrone (needs network for apt, npm, pip, GitHub). Run from this directory.
# Order matters: each image builds on the previous one. Build only on tyrone; nothing here pushes anywhere.
set -euo pipefail
cd "$(dirname "$0")"
docker build -f colmap.Dockerfile      -t namma-colmap:0.1 .       # colmap/colmap:latest + cuDNN 9
docker build -f colmap-0.2.Dockerfile  -t namma-colmap:0.2 .       # + python3 (stdlib) for pipeline.sfm
docker build -f gpu.Dockerfile         -t namma-space-gpu:0.1 .    # PyTorch 2.4.1 + gsplat 1.5.3
docker build -f gpu-0.2.Dockerfile     -t namma-space-gpu:0.2 .    # + Node 22 + splat-transform + MoGe-2 code
# Host cache (models are not baked into images). Files, hashes and licences: IMAGES.md.
#   mkdir -p /home/sher/namma-space/cache/colmap /home/sher/namma-space/cache/hf
#   curl -fLO https://github.com/colmap/colmap/releases/download/3.13.0/{aliked-n32,aliked-lightglue,sift-lightglue}.onnx
#   curl -fLO https://github.com/colmap/colmap/releases/download/3.13.0/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin
#   curl -fLO https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin
#   (MoGe weights: run docker/checks/a6_moge.py once with network; HF_HOME=/w/cache/hf)
