"""J5 deterministic synthetic-room TSDF acceptance check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.geometry.tsdf import depth_view, fuse_depth_views


def box_depth(c2w, K, width=64, height=64, bounds=(-1.5, 1.5, -1.0, 1.0, 0.0, 2.4)):
    yy, xx = np.mgrid[:height, :width]
    dcam = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1)
    dcam /= np.linalg.norm(dcam, axis=-1, keepdims=True)
    R, o = c2w[:3, :3], c2w[:3, 3]
    ray = dcam @ R.T
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lo - o) / ray
        t1 = (hi - o) / ray
    near, far = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
    t = np.where((far > np.maximum(near, 0)) & (far > 0), np.maximum(near, 0), 0)
    # OpenCV depth is z, not Euclidean ray length.
    return np.where(t > 0, t * dcam[..., 2], 0).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    w = h = 64
    K = np.array([[60., 0, 32.], [0, 60., 32.], [0, 0, 1.]])
    views = []
    for i, (x, z, yaw) in enumerate([(0, .8, 0), (0, 1.6, np.pi), (-.8, 1.2, np.pi / 2),
                                      (.8, 1.2, -np.pi / 2)]):
        c2w = np.eye(4)
        # camera looks +Z; yaw rotates the view around +Y
        c, s = np.cos(yaw), np.sin(yaw)
        c2w[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        c2w[:3, 3] = [x, 1.2, z]
        views.append(depth_view(box_depth(c2w, K, w, h), c2w, K, name=str(i)))
    mesh_path = a.out.with_suffix(".ply")
    fusion = {}
    mesh = fuse_depth_views(views, str(mesh_path), fusion_info=fusion)
    p = np.asarray(mesh.vertices)
    b = np.array([-1.5, 1.5, -1.0, 1.0, 0.0, 2.4])
    dist = np.minimum.reduce([np.abs(p[:, 0] - b[0]), np.abs(p[:, 0] - b[1]), np.abs(p[:, 1] - b[2]),
                              np.abs(p[:, 1] - b[3]), np.abs(p[:, 2] - b[4]), np.abs(p[:, 2] - b[5])])
    result = {"schema": "namma-space/eval@1", "name": "synthetic_room", "kind": "j5_tsdf",
              "status": "pass" if len(p) and float(np.median(dist)) <= .02 else "fail",
              "views": len(views), "vertices": len(p), "median_wall_error_m": float(np.median(dist)) if len(p) else None,
              "gate_m": .02, "voxel_size_m": .02, "sdf_trunc_m": .08, "depth_trunc_m": 5.0,
              "fusion": fusion}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
