"""Test similarity for the frame-independence check (E2).

Builds a plausible "aligned" frame from a COLMAP model without W2's align stage:
  +Y up from the mean camera down axis, a yaw, a scale, camera centroid dropped to 1.4 m above y = 0.
Writes <out>/sparse/0 (binary model after the similarity, our own writer: `colmap model_transformer` 4.2.1 left
points3D.bin and images.bin in different frames), <out>/images (symlink) and T_test.json.

    python -m pipeline.train.frame_test --model <data_dir or sparse/0> --out <dir> [--images <dir>] [--scale 2.5] [--yaw-deg 37]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pipeline.export import colmap_io as cio


def rodrigues_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def make_test_similarity(model: str, scale: float, yaw_deg: float, eye_height: float = 1.4):
    ims = cio.sorted_images(model)
    down = np.mean([im.R().T @ np.array([0.0, 1.0, 0.0]) for im in ims], axis=0)   # camera y axis in world
    r_up = rodrigues_align(down, np.array([0.0, -1.0, 0.0]))        # world "down" -> -Y
    th = np.deg2rad(yaw_deg)
    r_yaw = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    r = r_yaw @ r_up
    cen = np.array([im.centre() for im in ims])
    c_new = scale * (r @ cen.T).T
    t = np.array([0.0, eye_height, 0.0]) - c_new.mean(axis=0)
    t[1] = eye_height - c_new[:, 1].mean()
    return scale, r, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--yaw-deg", type=float, default=37.0)
    ap.add_argument("--images", help="photo directory to link as <out>/images")
    a = ap.parse_args()
    s, r, t = make_test_similarity(a.model, a.scale, a.yaw_deg)
    os.makedirs(a.out, exist_ok=True)
    cio.write_transformed_model(a.model, os.path.join(a.out, "sparse", "0"), s, r, t)
    if a.images:
        link = os.path.join(a.out, "images")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(a.images, link)
    with open(os.path.join(a.out, "T_test.json"), "w") as f:
        json.dump({"s": s, "R": r.tolist(), "t": t.tolist(), "yaw_deg": a.yaw_deg}, f, indent=1)
    print("wrote", os.path.join(a.out, "sparse", "0"))


if __name__ == "__main__":
    main()
