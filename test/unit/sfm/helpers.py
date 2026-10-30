"""Test helpers: write COLMAP binary models and build a fake `colmap` executable (stdlib only)."""
from __future__ import annotations

import json
import os
import stat
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_colmap.py")


def write_cameras_bin(path: str, model_id: int = 2, w: int = 1920, h: int = 1080, params: tuple = (1500.0, 960.0, 540.0, 0.01)) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, model_id, w, h))
        f.write(struct.pack("<" + "d" * len(params), *params))


def write_images_bin(path: str, cams: list[tuple[str, tuple[float, float, float]]], n_points2d: int = 3) -> None:
    """cams: [(name, camera centre)] with identity rotation, so tvec = -centre."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for i, (name, c) in enumerate(cams, start=1):
            f.write(struct.pack("<i", i))
            f.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
            f.write(struct.pack("<ddd", -c[0], -c[1], -c[2]))
            f.write(struct.pack("<i", 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", n_points2d))
            for k in range(n_points2d):
                f.write(struct.pack("<ddq", 1.0 * k, 2.0 * k, -1))


def write_model(d: str, cams: list[tuple[str, tuple[float, float, float]]], stats: dict | None = None, model_id: int = 2) -> None:
    os.makedirs(d, exist_ok=True)
    write_cameras_bin(os.path.join(d, "cameras.bin"), model_id=model_id)
    write_images_bin(os.path.join(d, "images.bin"), cams)
    with open(os.path.join(d, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", 0))
    with open(os.path.join(d, "stats.json"), "w") as f:   # fake_colmap's model_analyzer reads this
        json.dump(stats or {}, f)


def install_fake_colmap(bin_dir: str) -> str:
    """Create <bin_dir>/colmap that runs fake_colmap.py with the current interpreter; returns its path."""
    os.makedirs(bin_dir, exist_ok=True)
    p = os.path.join(bin_dir, "colmap")
    with open(p, "w") as f:
        f.write(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return p


def line_path(n: int, step: float = 0.1, start: int = 0) -> list[tuple[str, tuple[float, float, float]]]:
    return [(f"{i + 1:06d}.jpg", ((i + start) * step, 0.0, 0.0)) for i in range(n)]
