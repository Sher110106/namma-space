#!/usr/bin/env python3
"""Build a minimal 10_frames/frames.json for a plain image directory (public datasets): name + blur score.

`sharp` is the variance of the Laplacian of the grey image (higher = sharper). No timestamps (dataset images have none).
Runs in namma-space-gpu (needs OpenCV). usage: make_frames_json.py images_dir out.json
"""
import json
import os
import sys

import cv2


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    names = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    kept = []
    for n in names:
        g = cv2.imread(os.path.join(src, n), cv2.IMREAD_GRAYSCALE)
        kept.append({"name": n, "sharp": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2)})
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump({"n_kept": len(kept), "src": src, "kept": kept}, f)
    print(f"{len(kept)} frames -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
