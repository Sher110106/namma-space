#!/usr/bin/env bash
# One command: video/build_video.sh <run> [--out-name NAME] [--keep] [--reuse-footage] [--gif]
#   <run>        a run folder under runs/ with 80_web/ and 90_verify/ (the final room is "room")
#   output       video/out/Namma_Space_demo.mp4 + .srt (and copies 02_Demo_Video.*) for run "room"; Namma_Space_demo_dryrun.* for rh-honka; Namma_Space_demo_dryrun_<run>.* for others
#   capture clip optional: put the phone's own clip at video/inputs/<run>/capture_clip.mp4 (played at 4x in the capture segment)
# Needs: python3 + Pillow, node 22+, ffmpeg/ffprobe, macOS `say`, Playwright's headless Chrome shell (see docs/tools/chrome.py).
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 64; }
cd "$(dirname "$0")/.."
python3 video/nsvideo.py "$@"
