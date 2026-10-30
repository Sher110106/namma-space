"""Render any Gaussian PLY (or .splat) at given poses with gsplat; compare with photos.

Library use (W7):
    from pipeline.export.render_ply import load_gaussians, render_views, cameras_from_colmap, psnr_images
    g = load_gaussians("point_cloud_29999.ply")
    cams = cameras_from_colmap("/w/runs/room/40_align", held_out=True)        # list[Cam]
    imgs = render_views(g, cams)                                              # list of HxWx3 float32 in [0, 1]

Poses are camera-to-world in COLMAP/OpenCV axes (x right, y down, z forward) in the PLY's own frame.

CLI:
    python -m pipeline.export.render_ply --ply P --model <data_dir or sparse/0> [--images DIR]
        [--held-out | --all | --names a.jpg,b.jpg] [--out DIR] [--json FILE] [--sh-degree N] [--classic]
Writes PNG renders (and side-by-side PNGs with the photo when images exist) and prints per-image and mean PSNR.
Needs torch, gsplat and a GPU, so run it inside namma-space-gpu through nsdocker --gpu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import colmap_io as cio
from . import ply_io


@dataclass
class Cam:
    name: str
    c2w: np.ndarray            # (4, 4) camera to world, OpenCV axes
    K: np.ndarray              # (3, 3)
    width: int
    height: int
    image_path: Optional[str] = None


def cameras_from_colmap(model: str, images_dir: Optional[str] = None, held_out: bool = False,
                        test_every: int = 8, names: Optional[Sequence[str]] = None,
                        image_size_from_files: bool = True) -> List[Cam]:
    """Cameras of a COLMAP binary model, sorted by image name (gsplat's order).
    held_out=True keeps index % test_every == 0, the gsplat test split."""
    mdir = cio.model_dir(model)
    cams = cio.read_cameras(mdir)
    ims = cio.sorted_images(mdir)
    if held_out:
        ims = [im for i, im in enumerate(ims) if i % test_every == 0]
    if names:
        want = set(names)
        ims = [im for im in ims if im.name in want]
    root = images_dir or os.path.join(model if os.path.isdir(os.path.join(model, "images")) else os.path.dirname(os.path.dirname(mdir)), "images")
    out: List[Cam] = []
    for im in ims:
        cam = cams[im.camera_id]
        if cam.distorted:
            raise ValueError(f"camera {cam.id} is {cam.model}; undistort first (the pipeline uses PINHOLE)")
        fx, fy, cx, cy = cam.intrinsics()
        w, h = cam.width, cam.height
        path = os.path.join(root, im.name)
        path = path if os.path.isfile(path) else None
        if path and image_size_from_files:
            w2, h2 = _image_size(path)
            fx, cx, w = fx * w2 / w, cx * w2 / w, w2
            fy, cy, h = fy * h2 / h, cy * h2 / h, h2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        out.append(Cam(im.name, im.c2w(), K, int(w), int(h), path))
    return out


def cam_from_threejs(name: str, position, quaternion, fov_y_deg: float, width: int, height: int,
                     principal_point=(0.5, 0.5), t_final=None) -> Cam:
    """Camera for a viewer pose (CONTRACTS 2.0): three.js position and quaternion [x, y, z, w] in the final frame,
    vertical FOV, normalised principal point. t_final (export.json T_final: scale, quaternion, translation) is undone,
    so the render uses the PLY's own (aligned) frame. Square pixels."""
    from .heldout import FLIP, matrix_from_quat
    rot = matrix_from_quat(quaternion) @ FLIP            # camera-to-world rotation, OpenCV axes, final frame
    pos = np.asarray(position, dtype=np.float64)
    if t_final:
        sc, rf, tr = float(t_final["scale"]), matrix_from_quat(t_final["quaternion"]), np.asarray(t_final["translation"], float)
        rinv = rf.T
        pos = rinv @ (pos - tr) / sc
        rot = rinv @ rot
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = rot, pos
    fy = height / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    K = np.array([[fy, 0, principal_point[0] * width], [0, fy, principal_point[1] * height], [0, 0, 1.0]])
    return Cam(name, c2w, K, int(width), int(height))


def cameras_from_json(path: str) -> List[Cam]:
    """[{name, c2w (4x4 nested or 16 flat, OpenCV axes), fx, fy, cx, cy, width, height}] or, for viewer poses,
    [{name, position, quaternion, fov_y_deg, width, height, principal_point?}] (T_final under key "t_final" if needed)."""
    with open(path) as f:
        rows = json.load(f)
    out = []
    for i, r in enumerate(rows):
        name = r.get("name", r.get("id", f"{i:04d}"))
        if "c2w" in r:
            c2w = np.asarray(r["c2w"], dtype=np.float64).reshape(4, 4)
            K = np.array([[r["fx"], 0, r["cx"]], [0, r.get("fy", r["fx"]), r["cy"]], [0, 0, 1.0]])
            out.append(Cam(name, c2w, K, int(r["width"]), int(r["height"]), r.get("photo")))
        else:
            out.append(cam_from_threejs(name, r["position"], r["quaternion"], r["fov_y_deg"], int(r["width"]),
                                        int(r["height"]), r.get("principal_point", (0.5, 0.5)), r.get("t_final")))
    return out


def cameras_from_parser(data_dir: str, normalize: bool, held_out: bool = False, test_every: int = 8):
    """Cameras, camera centres and SfM points exactly as gsplat's example Parser sees them
    (used for models trained with world normalisation ON, whose PLY lives in the normalised frame)."""
    import sys as _sys
    ex = "/opt/gsplat/examples"
    if ex not in _sys.path:
        _sys.path.insert(0, ex)
    from datasets.colmap import Parser  # type: ignore
    p = Parser(data_dir=data_dir, factor=1, normalize=normalize, test_every=test_every)
    cams: List[Cam] = []
    for i, (name, path, c2w, cid) in enumerate(zip(p.image_names, p.image_paths, p.camtoworlds, p.camera_ids)):
        if held_out and i % test_every != 0:
            continue
        w, h = p.imsize_dict[cid]
        cams.append(Cam(name, np.asarray(c2w, dtype=np.float64), np.asarray(p.Ks_dict[cid], dtype=np.float64), int(w), int(h), path))
    return cams, np.asarray(p.camtoworlds)[:, :3, 3], np.asarray(p.points, dtype=np.float64)


def _image_size(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


def read_image(path: str, size: Optional[tuple] = None) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


@dataclass
class Gaussians:
    means: "object"        # torch (N, 3)
    quats: "object"        # torch (N, 4) w x y z
    scales: "object"       # torch (N, 3) linear
    opacities: "object"    # torch (N,) in 0..1
    colors: "object"       # torch (N, K, 3) SH coefficients
    sh_degree: int

    def __len__(self) -> int:
        return int(self.means.shape[0])


def gaussians_from_splats(s: ply_io.Splats, device: str = "cuda", sh_degree: Optional[int] = None) -> Gaussians:
    import torch
    deg = s.sh_degree if sh_degree is None else min(sh_degree, s.sh_degree)
    s = s.limited_to_degree(deg)
    n = len(s)
    dc = torch.from_numpy(s.f_dc).reshape(n, 1, 3)
    if s.f_rest.shape[1]:
        rest = torch.from_numpy(np.ascontiguousarray(s.sh_rest_cube()))
        colors = torch.cat([dc, rest], dim=1)
    else:
        colors = dc
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
    return Gaussians(
        means=t(s.means), quats=t(s.quat), scales=torch.exp(t(s.scale)),
        opacities=torch.sigmoid(t(s.opacity)), colors=colors.to(device).float(), sh_degree=deg)


def load_gaussians(path: str, device: str = "cuda", sh_degree: Optional[int] = None) -> Gaussians:
    if path.endswith(".splat"):
        from .splat_writer import read_splat
        s = read_splat(path)
    else:
        s = ply_io.read_ply(path)
    return gaussians_from_splats(s, device, sh_degree)


def render_views(g: Gaussians, cams: Sequence[Cam], antialiased: bool = True, near: float = 0.01,
                 far: float = 1e10, device: str = "cuda", return_alpha: bool = False,
                 background: float = 0.0):
    """Render each camera; returns a list of (H, W, 3) float32 arrays in [0, 1] (and alphas if asked)."""
    import torch
    from gsplat import rasterization
    outs, alphas = [], []
    with torch.no_grad():
        for c in cams:
            viewmat = torch.from_numpy(np.linalg.inv(c.c2w)).float().to(device)[None]
            K = torch.from_numpy(c.K).float().to(device)[None]
            rgb, alpha, _ = rasterization(
                means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacities, colors=g.colors,
                viewmats=viewmat, Ks=K, width=c.width, height=c.height, sh_degree=g.sh_degree,
                near_plane=near, far_plane=far, packed=False,
                rasterize_mode="antialiased" if antialiased else "classic", camera_model="pinhole")
            img = rgb[0]
            if background:
                img = img + background * (1 - alpha[0])
            outs.append(img.clamp(0, 1).cpu().numpy().astype(np.float32))
            alphas.append(alpha[0, ..., 0].cpu().numpy())
    return (outs, alphas) if return_alpha else outs


def psnr_images(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))


def evaluate(g: Gaussians, cams: Sequence[Cam], antialiased: bool = True) -> Dict[str, object]:
    """Per-camera PSNR against the photos and its mean (the mean of per-image PSNRs, as gsplat reports)."""
    imgs = render_views(g, cams, antialiased=antialiased)
    per = {}
    for c, im in zip(cams, imgs):
        if c.image_path:
            per[c.name] = psnr_images(im, read_image(c.image_path, (c.width, c.height)))
    vals = list(per.values())
    return {"psnr": float(np.mean(vals)) if vals else None, "n": len(vals), "per_image": per}


def _save_png(path: str, arr: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.export.render_ply", description=__doc__.split("\n")[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--model", help="COLMAP binary model, data dir or sparse dir")
    ap.add_argument("--images", help="photo directory (default: <data dir>/images)")
    ap.add_argument("--poses", help="JSON poses (see cameras_from_json) instead of --model cameras")
    g_ = ap.add_mutually_exclusive_group()
    g_.add_argument("--held-out", action="store_true", help="gsplat test split (every 8th by name)")
    g_.add_argument("--all", action="store_true")
    g_.add_argument("--names", help="comma-separated image names")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--out", help="directory for PNG renders")
    ap.add_argument("--json", help="write metrics JSON here")
    ap.add_argument("--sh-degree", type=int)
    ap.add_argument("--classic", action="store_true", help="classic instead of antialiased rasterisation")
    a = ap.parse_args(argv)
    if a.poses:
        cams = cameras_from_json(a.poses)
    elif a.model:
        cams = cameras_from_colmap(a.model, a.images, held_out=not (a.all or a.names), test_every=a.test_every,
                                   names=a.names.split(",") if a.names else None)
    else:
        ap.error("give --model or --poses")
    g = load_gaussians(a.ply, sh_degree=a.sh_degree)
    print(f"{len(g)} splats, SH{g.sh_degree}, {len(cams)} views", flush=True)
    imgs = render_views(g, cams, antialiased=not a.classic)
    res = {"ply": a.ply, "n_splats": len(g), "sh_degree": g.sh_degree, "antialiased": not a.classic, "per_image": {}}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for c, im in zip(cams, imgs):
        stem = os.path.splitext(c.name)[0].replace("/", "_")
        if c.image_path:
            ph = read_image(c.image_path, (c.width, c.height))
            res["per_image"][c.name] = psnr_images(im, ph)
            if a.out:
                _save_png(os.path.join(a.out, stem + "_cmp.png"), np.concatenate([ph, im], axis=1))
        elif a.out:
            _save_png(os.path.join(a.out, stem + ".png"), im)
    vals = list(res["per_image"].values())
    res["psnr"] = float(np.mean(vals)) if vals else None
    print(json.dumps({k: v for k, v in res.items() if k != "per_image"}))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
