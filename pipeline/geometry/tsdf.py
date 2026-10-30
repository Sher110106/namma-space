"""Gaussian-splat depth rendering and metric TSDF fusion.

The geometry stage deliberately separates the GPU part (rendering expected depth
from a PLY) from the CPU part (integrating depth maps with Open3D).  This makes
the latter useful in unit tests and for the MuSHRoom LiDAR baseline, where depth
maps already exist on disk.

All poses use COLMAP/OpenCV camera axes: x right, y down, z forward.  ``c2w``
is camera-to-world and Open3D receives its inverse as the RGB-D extrinsic.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


VOXEL_SIZE = 0.02
SDF_TRUNC = 0.08
DEPTH_TRUNC = 5.0
MAX_UNIFORM_RESOLUTION = 512
SURFACE_SAMPLE_POINTS = 100_000


@dataclass
class DepthView:
    """One rendered or measured depth view."""

    depth: np.ndarray
    alpha: np.ndarray
    c2w: np.ndarray
    K: np.ndarray
    rgb: Optional[np.ndarray] = None
    name: str = ""

    @property
    def width(self) -> int:
        return int(self.depth.shape[1])

    @property
    def height(self) -> int:
        return int(self.depth.shape[0])


def _as_pose(view: Any) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    """Extract c2w, K, optional RGB and name from Cam/dict-like objects."""
    if isinstance(view, DepthView):
        return view.c2w, view.K, view.rgb, view.name
    if hasattr(view, "c2w") and hasattr(view, "K"):
        return np.asarray(view.c2w, float), np.asarray(view.K, float), None, str(getattr(view, "name", ""))
    if isinstance(view, Mapping):
        if "c2w" in view:
            c2w = np.asarray(view["c2w"], float).reshape(4, 4)
            K = np.asarray(view.get("K", [[view["fx"], 0, view["cx"]],
                                           [0, view.get("fy", view["fx"]), view["cy"]], [0, 0, 1]]), float)
        else:
            c2w = np.asarray(view.get("camera_to_world", view["pose"]), float).reshape(4, 4)
            K = np.asarray(view["K"], float)
        rgb = view.get("rgb")
        return c2w, K, None if rgb is None else np.asarray(rgb), str(view.get("name", ""))
    raise TypeError("view must provide c2w and K")


def _render_one(g: Any, c2w: np.ndarray, K: np.ndarray, width: int, height: int,
                *, antialiased: bool = True, device: str = "cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render RGB, expected depth and alpha with gsplat's RGB+ED mode."""
    import torch
    from gsplat import rasterization

    viewmat = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)[None]
    kt = torch.from_numpy(np.asarray(K, dtype=np.float32)).to(device)[None]
    # RGB+ED is gsplat's depth mode: the fourth channel is expected depth.
    colors, alphas, _ = rasterization(
        means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacities,
        colors=g.colors, viewmats=viewmat, Ks=kt, width=int(width), height=int(height),
        sh_degree=int(g.sh_degree), near_plane=0.01, far_plane=DEPTH_TRUNC,
        packed=False, rasterize_mode="antialiased" if antialiased else "classic",
        camera_model="pinhole", render_mode="RGB+ED")
    # gsplat versions have returned either (H,W,4) or (H,W,1) for ED.
    arr = colors[0]
    if arr.shape[-1] < 4:
        raise RuntimeError(f"gsplat RGB+ED returned {tuple(arr.shape)}; expected RGB plus depth")
    rgb = arr[..., :3].clamp(0, 1).detach().cpu().numpy().astype(np.float32)
    depth = arr[..., 3].detach().cpu().numpy().astype(np.float32)
    alpha = alphas[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    depth[(alpha < 0.5) | ~np.isfinite(depth) | (depth <= 0) | (depth > DEPTH_TRUNC)] = 0.0
    return rgb, depth, alpha


def render_expected_depth(gaussians: Any, cameras: Sequence[Any], *, antialiased: bool = True,
                          device: str = "cuda") -> list[DepthView]:
    """Render expected depth at cameras using ``render_mode='RGB+ED'``.

    ``gaussians`` is the :class:`pipeline.export.render_ply.Gaussians` object;
    cameras are its ``Cam`` objects or dictionaries with ``c2w``, ``K``, width,
    and height.  The returned depth is metres in the PLY frame.
    """
    out: list[DepthView] = []
    for cam in cameras:
        c2w, K, _, name = _as_pose(cam)
        width, height = int(getattr(cam, "width", 0)), int(getattr(cam, "height", 0))
        if isinstance(cam, Mapping):
            width, height = int(cam["width"]), int(cam["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"camera {name!r} has invalid dimensions {width}x{height}")
        rgb, depth, alpha = _render_one(gaussians, c2w, K, width, height,
                                        antialiased=antialiased, device=device)
        out.append(DepthView(depth, alpha, c2w, K, rgb, name))
    return out


def render_ply_depth(ply: str, cameras: Sequence[Any], *, device: str = "cuda",
                     antialiased: bool = True) -> list[DepthView]:
    """Convenience wrapper loading a Gaussian PLY with the export reader."""
    from pipeline.export.render_ply import load_gaussians
    return render_expected_depth(load_gaussians(ply, device=device), cameras,
                                 device=device, antialiased=antialiased)


def depth_view(depth: np.ndarray, c2w: np.ndarray, K: np.ndarray, *, alpha: Optional[np.ndarray] = None,
               rgb: Optional[np.ndarray] = None, name: str = "") -> DepthView:
    """Construct and validate a :class:`DepthView` from arrays."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError("depth must be HxW")
    a = np.ones_like(d, dtype=np.float32) if alpha is None else np.asarray(alpha, dtype=np.float32)
    if a.shape != d.shape:
        raise ValueError("alpha and depth must have the same shape")
    return DepthView(d, a, np.asarray(c2w, float).reshape(4, 4), np.asarray(K, float).reshape(3, 3), rgb, name)


def _rgbd(v: DepthView, *, depth_trunc: float, color: bool):
    import open3d as o3d
    d = np.asarray(v.depth, dtype=np.float32).copy()
    valid = (v.alpha >= 0.5) & np.isfinite(d) & (d > 0) & (d <= depth_trunc)
    d[~valid] = 0.0
    di = o3d.geometry.Image(np.ascontiguousarray(d))
    if color and v.rgb is not None:
        rgb = np.clip(np.asarray(v.rgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        rgb = np.zeros((*d.shape, 3), np.uint8)
    ri = o3d.geometry.Image(np.ascontiguousarray(rgb))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        ri, di, depth_scale=1.0, depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False)


def _world_bounds(views: Sequence[DepthView], *, depth_trunc: float, sdf_trunc: float,
                  voxel_size: float, bbox: Optional[tuple[np.ndarray, np.ndarray]] = None):
    """Return a padded cubic world-space volume origin and side length."""
    if bbox is not None:
        lo, hi = (np.asarray(bbox[0], float), np.asarray(bbox[1], float))
    else:
        chunks = []
        for v in views:
            d = np.asarray(v.depth, dtype=np.float32)
            valid = (v.alpha >= 0.5) & np.isfinite(d) & (d > 0) & (d <= depth_trunc)
            ys, xs = np.nonzero(valid)
            if len(xs) == 0:
                continue
            # The bbox only needs a sparse geometric envelope, not every pixel.
            step = max(1, int(np.ceil(np.sqrt(len(xs) / 200_000))))
            xs, ys = xs[::step], ys[::step]
            z = d[ys, xs]
            cam = np.column_stack(((xs - v.K[0, 2]) * z / v.K[0, 0],
                                   (ys - v.K[1, 2]) * z / v.K[1, 1], z))
            chunks.append((v.c2w[:3, :3] @ cam.T).T + v.c2w[:3, 3])
            chunks.append(v.c2w[None, :3, 3])
        if not chunks:
            raise ValueError("no valid depth samples for UniformTSDFVolume bounds")
        pts = np.concatenate(chunks, axis=0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
    if np.any(hi <= lo):
        raise ValueError(f"invalid TSDF bounds {lo}..{hi}")
    lo = lo - float(sdf_trunc)
    hi = hi + float(sdf_trunc)
    span = float(np.max(hi - lo))
    # Keep a small voxel-aligned margin around the requested box.
    length = float(np.ceil(span / float(voxel_size)) * float(voxel_size))
    center = (lo + hi) * 0.5
    origin = center - length * 0.5
    resolution = int(round(length / float(voxel_size)))
    if resolution > MAX_UNIFORM_RESOLUTION:
        resolution = MAX_UNIFORM_RESOLUTION
        length = float(resolution * max(length / resolution, voxel_size))
        origin = center - length * 0.5
    return origin, length, resolution


def _integrate_open3d(views: Sequence[DepthView], *, voxel_size: float, sdf_trunc: float,
                      depth_trunc: float, color: bool = True, kind: str = "scalable",
                      bbox: Optional[tuple[np.ndarray, np.ndarray]] = None):
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - exercised on Mac without Open3D
        raise RuntimeError("Open3D is required for TSDF fusion; use namma-space-gpu on tyrone") from exc
    ctype = (o3d.pipelines.integration.TSDFVolumeColorType.RGB8
             if color else o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    if kind == "uniform":
        origin, length, resolution = _world_bounds(
            views, depth_trunc=depth_trunc, sdf_trunc=sdf_trunc,
            voxel_size=voxel_size, bbox=bbox)
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length, resolution, float(sdf_trunc), ctype, origin)
    else:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size), sdf_trunc=float(sdf_trunc), color_type=ctype)
    for v in views:
        rgbd = _rgbd(v, depth_trunc=depth_trunc, color=color)
        pin = o3d.camera.PinholeCameraIntrinsic(v.width, v.height, float(v.K[0, 0]), float(v.K[1, 1]),
                                                  float(v.K[0, 2]), float(v.K[1, 2]))
        volume.integrate(rgbd, pin, np.linalg.inv(v.c2w))
    return volume


def fuse_depth_views(views: Sequence[DepthView], out_mesh: Optional[str] = None, *,
                     voxel_size: float = VOXEL_SIZE, sdf_trunc: float = SDF_TRUNC,
                     depth_trunc: float = DEPTH_TRUNC, color: bool = True,
                     bbox: Optional[tuple[np.ndarray, np.ndarray]] = None,
                     fusion_info: Optional[dict] = None):
    """Fuse views with bounded UniformTSDFVolume, recording alternatives."""
    if not views:
        raise ValueError("at least one depth view is required")
    info = fusion_info if fusion_info is not None else {}
    info.update({"requested": "UniformTSDFVolume", "voxel_size_m": float(voxel_size),
                 "sdf_trunc_m": float(sdf_trunc), "depth_trunc_m": float(depth_trunc)})
    mesh = None
    try:
        origin, length, resolution = _world_bounds(
            views, depth_trunc=depth_trunc, sdf_trunc=sdf_trunc,
            voxel_size=voxel_size, bbox=bbox)
        info.update({"origin_m": origin.tolist(), "length_m": float(length),
                     "resolution": int(resolution),
                     "effective_voxel_size_m": float(length / resolution)})
        mesh = _integrate_open3d(views, voxel_size=voxel_size, sdf_trunc=sdf_trunc,
                                 depth_trunc=depth_trunc, color=color, kind="uniform", bbox=bbox).extract_triangle_mesh()
        if len(mesh.vertices):
            info["method"] = "UniformTSDFVolume"
        else:
            info["uniform_vertices"] = 0
    except Exception as exc:
        info["uniform_error"] = f"{type(exc).__name__}: {exc}"
    if mesh is None or len(mesh.vertices) == 0:
        try:
            mesh = _integrate_open3d(views, voxel_size=voxel_size, sdf_trunc=sdf_trunc,
                                     depth_trunc=depth_trunc, color=color, kind="scalable").extract_triangle_mesh()
            if len(mesh.vertices):
                info["method"] = "ScalableTSDFVolume"
            else:
                info["scalable_vertices"] = 0
        except Exception as exc:
            info["scalable_error"] = f"{type(exc).__name__}: {exc}"
    if mesh is None or len(mesh.vertices) == 0:
        mesh = _point_surface_fallback(views, depth_trunc=depth_trunc,
                                       voxel_size=voxel_size, sdf_trunc=sdf_trunc)
        info["method"] = "point_surface_fallback"
    mesh.compute_vertex_normals()
    if out_mesh:
        Path(out_mesh).parent.mkdir(parents=True, exist_ok=True)
        import open3d as o3d
        if not o3d.io.write_triangle_mesh(str(out_mesh), mesh, write_ascii=False):
            raise IOError(f"Open3D could not write mesh {out_mesh}")
    return mesh


def _point_surface_fallback(views: Sequence[DepthView], *, depth_trunc: float,
                            voxel_size: float, sdf_trunc: float):
    """Build a deterministic surface when ScalableTSDFVolume has no voxels."""
    import open3d as o3d
    clouds = []
    pixel_step = 4  # fallback only; ScalableTSDF path keeps every pixel
    for v in views:
        d = np.asarray(v.depth, np.float32).copy()
        d[(v.alpha < 0.5) | ~np.isfinite(d) | (d <= 0) | (d > depth_trunc)] = 0
        d = d[::pixel_step, ::pixel_step]
        rgb = (np.zeros((*d.shape, 3), np.uint8) if v.rgb is None else
               np.clip(np.asarray(v.rgb)[::pixel_step, ::pixel_step] * 255 + .5, 0, 255).astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(d)), depth_scale=1.0,
            depth_trunc=float(depth_trunc), convert_rgb_to_intensity=False)
        pin = o3d.camera.PinholeCameraIntrinsic(d.shape[1], d.shape[0], float(v.K[0, 0]) / pixel_step,
                                                  float(v.K[1, 1]) / pixel_step, float(v.K[0, 2]) / pixel_step,
                                                  float(v.K[1, 2]) / pixel_step)
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pin, np.linalg.inv(v.c2w))
        if len(cloud.points):
            clouds.append(cloud.voxel_down_sample(max(float(voxel_size), 0.02)))
    if not clouds:
        return o3d.geometry.TriangleMesh()
    pcd = clouds[0]
    for cloud in clouds[1:]:
        pcd += cloud
    pcd = pcd.voxel_down_sample(max(float(voxel_size), 0.02))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(sdf_trunc) * 2.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(min(30, max(3, len(pcd.points) - 1)))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=1.1)
    return mesh


def fuse_depth_maps(depths: Sequence[np.ndarray], cameras: Sequence[Any], out_mesh: Optional[str] = None,
                    *, alphas: Optional[Sequence[np.ndarray]] = None,
                    rgbs: Optional[Sequence[np.ndarray]] = None, voxel_size: float = VOXEL_SIZE,
                    sdf_trunc: float = SDF_TRUNC, depth_trunc: float = DEPTH_TRUNC):
    """Fuse plain arrays, useful for LiDAR and deterministic tests."""
    if len(depths) != len(cameras):
        raise ValueError("depths and cameras must have the same length")
    views = []
    for i, (d, c) in enumerate(zip(depths, cameras)):
        c2w, K, rgb, name = _as_pose(c)
        a = None if alphas is None else alphas[i]
        r = (None if rgbs is None else rgbs[i]) if rgbs is not None else rgb
        views.append(depth_view(d, c2w, K, alpha=a, rgb=r, name=name or str(i)))
    return fuse_depth_views(views, out_mesh, voxel_size=voxel_size, sdf_trunc=sdf_trunc,
                            depth_trunc=depth_trunc, color=any(v.rgb is not None for v in views))


def query_free_space(points: np.ndarray, views: Sequence[DepthView], *, margin: float = 0.06,
                     min_views: int = 1) -> np.ndarray:
    """Return points observed in free space at least ``margin`` in front of surfaces.

    A point is considered free when it projects inside a view, lies in front of
    its valid depth by ``margin`` (three 2 cm voxels by default), and is not
    occluded by an invalid/transparent pixel.  ``min_views`` avoids deleting a
    splat based on one marginal projection.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    hits = np.zeros(len(p), dtype=np.int32)
    for v in views:
        cam = (np.linalg.inv(v.c2w) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
        z = cam[:, 2]
        ok = z > 0
        u = np.rint(v.K[0, 0] * cam[:, 0] / np.maximum(z, 1e-12) + v.K[0, 2]).astype(int)
        vv = np.rint(v.K[1, 1] * cam[:, 1] / np.maximum(z, 1e-12) + v.K[1, 2]).astype(int)
        inside = ok & (u >= 0) & (u < v.width) & (vv >= 0) & (vv < v.height)
        idx = np.flatnonzero(inside)
        if len(idx):
            obs = v.depth[vv[idx], u[idx]]
            valid = (v.alpha[vv[idx], u[idx]] >= 0.5) & (obs > 0) & np.isfinite(obs)
            hits[idx[valid & (z[idx] + margin < obs)]] += 1
    return hits >= int(min_views)


def free_space_query(points: np.ndarray, depth_views: Sequence[DepthView], *, margin: float = 0.06,
                     min_views: int = 1) -> np.ndarray:
    """Compatibility alias used by the geometry stage and clean filter."""
    return query_free_space(points, depth_views, margin=margin, min_views=min_views)


# Short aliases used by early WP7 notebooks.
render_depth = render_expected_depth
fuse_tsdf = fuse_depth_views


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--poses", required=True, help="render_ply pose JSON")
    ap.add_argument("--out", required=True, help="mesh PLY")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--classic", action="store_true")
    a = ap.parse_args(argv)
    from pipeline.export.render_ply import cameras_from_json
    views = render_ply_depth(a.ply, cameras_from_json(a.poses), device=a.device, antialiased=not a.classic)
    fuse_depth_views(views, a.out)
    print(json.dumps({"mesh": a.out, "views": len(views), "voxel_size": VOXEL_SIZE,
                      "sdf_trunc": SDF_TRUNC, "depth_trunc": DEPTH_TRUNC}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
