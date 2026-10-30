#!/usr/bin/env python3
"""Build the run directories for the C3 failure tests from MuSHRoom images. Runs in namma-space-gpu (OpenCV).

w1-c3a  two scenes concatenated: honka long_capture 1-200 then vr_room long_capture 1-60 (the larger one is 77%)
w1-c3a2 two equal scenes: honka 1-150 then vr_room 1-150 (largest model <= 50%: expect "nothing usable", exit 10)
w1-c3b  honka long_capture (335) with 40 uniform grey frames inserted after every 8th real frame (375 frames)
w1-c3c  honka long_capture (335), plain: used with a variant killed mid-run

Frames are symlinks named 000001.jpg ... in time order; grey frames and resized frames are written as files.
usage: make_c3_runs.py /w/runs
"""
import json
import os
import sys

import cv2
import numpy as np

DATA = "/w/data/mushroom/room_datasets"


def src(room: str) -> str:
    return f"{DATA}/{room}/iphone/long_capture/images"


def names(d: str, n: int) -> list[str]:
    return [os.path.join(d, f) for f in sorted(os.listdir(d))[:n]]


def sharp(path: str) -> float:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2)


def build(root: str, name: str, items: list[tuple[str, str]], size: tuple[int, int]) -> None:
    """items: [(kind, path)] kind in real|grey. Writes images/ and frames.json."""
    base = os.path.join(root, name, "10_frames")
    os.system(f"rm -rf '{os.path.join(root, name)}'")
    os.makedirs(os.path.join(base, "images"))
    kept = []
    for i, (kind, path) in enumerate(items, start=1):
        out = os.path.join(base, "images", f"{i:06d}.jpg")
        if kind == "grey":
            cv2.imwrite(out, np.full((size[1], size[0], 3), 128, np.uint8))
        else:
            img = cv2.imread(path)
            if (img.shape[1], img.shape[0]) != size:
                cv2.imwrite(out, cv2.resize(img, size, interpolation=cv2.INTER_AREA))
            else:
                os.symlink(path, out)
        kept.append({"name": f"{i:06d}.jpg", "kind": kind, "sharp": sharp(out)})
    with open(os.path.join(base, "frames.json"), "w") as f:
        json.dump({"n_kept": len(kept), "kept": kept}, f)
    print(name, len(kept), "frames; grey:", sum(k["kind"] == "grey" for k in kept))


def main() -> int:
    root = sys.argv[1]
    h, v = names(src("honka"), 400), names(src("vr_room"), 400)
    probe = cv2.imread(h[0])
    size = (probe.shape[1], probe.shape[0])
    print("honka size", size, "vr_room size", cv2.imread(v[0]).shape[1::-1])
    build(root, "w1-c3a", [("real", p) for p in h[:200]] + [("real", p) for p in v[:60]], size)
    build(root, "w1-c3a2", [("real", p) for p in h[:150]] + [("real", p) for p in v[:150]], size)
    allh = names(src("honka"), 400)
    items: list[tuple[str, str]] = []
    for i, p in enumerate(allh, start=1):
        items.append(("real", p))
        if i % 8 == 0 and i <= 320:
            items.append(("grey", ""))
    build(root, "w1-c3b", items, size)
    build(root, "w1-c3c", [("real", p) for p in allh], size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
