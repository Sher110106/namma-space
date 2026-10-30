#!/bin/bash
# Make a run directory with N consecutive frames (symlinks) of an image directory, starting at START (1-based).
# usage: prep_subset.sh <run_name> <image_dir> <start> <n>     (works in any image; no OpenCV; no frames.json)
set -euo pipefail
R=/w/runs/$1; SRC=$2; START=$3; N=$4
rm -rf "$R"; mkdir -p "$R/10_frames/images"
ls "$SRC" | sed -n "${START},$((START+N-1))p" | while read -r f; do ln -s "$SRC/$f" "$R/10_frames/images/$f"; done
echo "$(ls "$R/10_frames/images" | wc -l) frames in $R/10_frames/images"
