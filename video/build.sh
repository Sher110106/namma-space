#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/video/out/namma-space-demo.mp4"}
FRAMES=${FRAMES:-"$ROOT/video/frames"}
mkdir -p "$(dirname "$OUT")"
python3 "$ROOT/video/render_cards.py" >/dev/null
LIST=$(mktemp)
cleanup(){ rm -f "$LIST"; }
trap cleanup EXIT
# Cards and the short viewer capture are concatenated as image sequences.
for card in "$ROOT/video/frames/title.png" "$ROOT/video/frames/limits.png"; do
  printf "file '%s'\nduration 3\n" "$card" >> "$LIST"
done
if compgen -G "$FRAMES/tour-*.png" >/dev/null; then
  for f in "$FRAMES"/tour-*.png; do printf "file '%s'\n" "$f" >> "$LIST"; done
else
  printf "file '%s'\nduration 4\n" "$ROOT/video/frames/title.png" >> "$LIST"
fi
# Burn captions when the local ffmpeg has libass; the fallback still burns a short
# caption line for builds made with a minimal ffmpeg and preserves the full SRT.
if ffmpeg -hide_banner -filters 2>/dev/null | grep -q ' subtitles ' ; then
  VF="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p,subtitles=filename='$ROOT/video/captions.srt'"
else
  printf 'Namma Space · captions in the accompanying SRT\n' > "$ROOT/video/caption_burn.txt"
  echo "ffmpeg has no subtitles/drawtext filter; using clean video and shipping the SRT" >&2
  VF="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
fi
ffmpeg -y -f concat -safe 0 -i "$LIST" -vf "$VF" -r 30 -c:v libx264 -profile:v high -pix_fmt yuv420p -c:a aac -b:a 128k -shortest -movflags +faststart "$OUT"
cp "$ROOT/video/captions.srt" "${OUT%.mp4}.srt"
echo "wrote $OUT and ${OUT%.mp4}.srt"
