#!/bin/bash
# Make a run directory for a public image set: 10_frames/images (symlink) + frames.json (blur scores).
# usage (host): nsdocker ns-w1-prep namma-space-gpu:0.2 bash docker/checks/prep_run.sh <run_name> <image_dir>
set -euo pipefail
R=/w/runs/$1; SRC=$2
rm -rf "$R/10_frames"; mkdir -p "$R/10_frames"
ln -s "$SRC" "$R/10_frames/images"
python /w/repo/docker/checks/make_frames_json.py "$R/10_frames/images" "$R/10_frames/frames.json"
