"""Reproducible MuSHRoom LiDAR TSDF baseline scorer (J2).

Run inside ``namma-space-gpu:0.1`` on tyrone.  The source poses are Nerfstudio
OpenGL camera-to-world matrices; the conversion below is the one documented in
``pipeline/align/NOTES.md``.  Depth PNG values are millimetres.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.geometry.tsdf import depth_view, fuse_depth_views


def digest(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(path)).encode())
                h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def mushroom_views(root: Path, stride: int = 1):
    meta = json.loads((root / "iphone/long_capture/transformations.json").read_text())
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    out = []
    for i, fr in enumerate(meta["frames"]):
        if i % stride:
            continue
        d = np.asarray(Image.open(root / "iphone/long_capture" / fr["depth_file_path"].lstrip("./")), dtype=np.float32) / 1000.0
        c2w = np.asarray(fr["transform_matrix"], dtype=float) @ flip
        K = np.array([[fr["fl_x"], 0, fr["cx"]], [0, fr["fl_y"], fr["cy"]], [0, 0, 1.0]])
        out.append(depth_view(d, c2w, K, name=Path(fr["file_path"]).name))
    return out


def _load_points(path: Path):
    import open3d as o3d
    if path.suffix.lower() == ".ply":
        m = o3d.io.read_triangle_mesh(str(path))
        if len(m.vertices):
            return np.asarray(m.vertices)
        p = o3d.io.read_point_cloud(str(path))
        return np.asarray(p.points)
    raise ValueError(path)


def score_mesh(mesh_path: Path, gt_path: Path, icp_path: Path, threshold: float = 0.05):
    from scipy.spatial import cKDTree
    ours = _load_points(mesh_path)
    gt = _load_points(gt_path)
    T = np.asarray(json.loads(icp_path.read_text())["gt_transformation"], dtype=float)
    ours = (T[:3, :3] @ ours.T).T + T[:3, 3]
    d_ours = cKDTree(gt).query(ours, workers=-1)[0]
    d_gt = cKDTree(ours).query(gt, workers=-1)[0]
    precision = float(np.mean(d_ours <= threshold))
    recall = float(np.mean(d_gt <= threshold))
    return {"accuracy_m": float(np.mean(d_ours)), "completeness_m": float(np.mean(d_gt)),
            "accuracy_median_m": float(np.median(d_ours)), "completeness_median_m": float(np.median(d_gt)),
            "f_score_5cm": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "precision_5cm": precision, "recall_5cm": recall, "n_ours": int(len(ours)), "n_gt": int(len(gt))}


def figure(path: Path, score: dict):
    im = Image.new("RGB", (900, 500), "white")
    dr = ImageDraw.Draw(im)
    dr.text((35, 25), "MuSHRoom LiDAR TSDF baseline", fill="black")
    vals = [("accuracy (cm)", score["accuracy_m"] * 100), ("completeness (cm)", score["completeness_m"] * 100),
            ("F-score @ 5 cm", score["f_score_5cm"] * 100)]
    for i, (label, val) in enumerate(vals):
        y = 100 + i * 115
        dr.text((35, y), f"{label}: {val:.3f}", fill="black")
        width = int(min(760, max(2, val / (100 if i == 2 else 10) * 760)))
        dr.rectangle((35, y + 30, 35 + width, y + 75), fill=(44, 103, 172))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--name", default="mushroom_honka_lidar")
    a = ap.parse_args(argv)
    root = a.dataset
    views = mushroom_views(root, max(1, a.stride))
    mesh_path = a.out.with_suffix(".mesh.ply")
    fuse_depth_views(views, str(mesh_path))
    score = score_mesh(mesh_path, root / "gt_mesh.ply", root / "icp_iphone.json")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    result = {"schema": "namma-space/eval@1", "name": a.name, "kind": "mushroom_lidar_baseline",
              "dataset": str(root), "pose_source": "iphone/long_capture/transformations.json",
              "pose_convention": "OpenGL c2w converted with diag(1,-1,-1)", "depth_units": "millimetres / 1000",
              "views": len(views), "stride": a.stride, "voxel_size_m": 0.02, "sdf_trunc_m": 0.08,
              "depth_trunc_m": 5.0, "icp": str(root / "icp_iphone.json"), "score": score,
              "fusion": "Open3D ScalableTSDFVolume primary; deterministic point-surface fallback if volume extraction is empty",
              "git": sha, "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "image_digests": {"dataset": digest(root / "iphone/long_capture"), "gt_mesh": digest(root / "gt_mesh.ply")},
              "command": "python eval/score_mushroom.py --dataset ... --out ..."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    figure(a.out.with_suffix(".png"), score)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
