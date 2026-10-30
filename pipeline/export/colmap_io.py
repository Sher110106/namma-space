"""Minimal COLMAP binary-model reader (numpy + stdlib). Binary only: text models are not supported.

Reads cameras.bin, images.bin, points3D.bin as written by COLMAP 3.x/4.x.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# model id -> (name, number of parameters)
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                 3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
                 6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
                 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]

    def intrinsics(self) -> Tuple[float, float, float, float]:
        """(fx, fy, cx, cy). Distortion terms are ignored; check `distorted` first."""
        p = self.params
        if self.model == "SIMPLE_PINHOLE" or self.model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]

    @property
    def distorted(self) -> bool:
        return self.model not in ("PINHOLE", "SIMPLE_PINHOLE")


@dataclass
class Image:
    id: int
    name: str
    camera_id: int
    qvec: Tuple[float, float, float, float]   # w x y z, world-to-camera
    tvec: Tuple[float, float, float]

    def R(self) -> np.ndarray:
        return qvec2rot(self.qvec)

    def w2c(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R()
        m[:3, 3] = self.tvec
        return m

    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c())

    def centre(self) -> np.ndarray:
        return -self.R().T @ np.array(self.tvec)


def qvec2rot(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def model_dir(path: str) -> str:
    """Accept a data dir (with sparse/0), a sparse dir, or the model dir itself."""
    for cand in (path, os.path.join(path, "0"), os.path.join(path, "sparse", "0"), os.path.join(path, "sparse")):
        if os.path.exists(os.path.join(cand, "cameras.bin")):
            return cand
    if os.path.exists(os.path.join(path, "cameras.txt")) or os.path.exists(os.path.join(path, "sparse", "0", "cameras.txt")):
        raise ValueError(f"{path}: text COLMAP model; convert to binary (gsplat's text reader crashes)")
    raise FileNotFoundError(f"no COLMAP binary model under {path}")


def read_cameras(path: str) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    with open(os.path.join(model_dir(path), "cameras.bin"), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = CAMERA_MODELS[mid]
            params = struct.unpack("<" + "d" * npar, f.read(8 * npar))
            cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def read_images(path: str) -> Dict[int, Image]:
    imgs: Dict[int, Image] = {}
    with open(os.path.join(model_dir(path), "images.bin"), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", f.read(4 + 56 + 4))
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00" or not ch:
                    break
                name += ch
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, os.SEEK_CUR)
            imgs[iid] = Image(iid, name.decode(), cid, (qw, qx, qy, qz), (tx, ty, tz))
    return imgs


def read_points3d(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz (P,3) float64, rgb (P,3) uint8, track_len (P,) int)."""
    xyz: List[Tuple[float, float, float]] = []
    rgb: List[Tuple[int, int, int]] = []
    tl: List[int] = []
    with open(os.path.join(model_dir(path), "points3D.bin"), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _pid, x, y, z, r, g, b, _err, t = struct.unpack("<QdddBBBdQ", f.read(8 + 24 + 3 + 8 + 8))
            f.seek(8 * t, os.SEEK_CUR)
            xyz.append((x, y, z)); rgb.append((r, g, b)); tl.append(t)
    return np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3), np.array(tl)


def sorted_images(path: str) -> List[Image]:
    """Images sorted by name, the order gsplat uses for its test split."""
    return sorted(read_images(path).values(), key=lambda im: im.name)


def heldout_images(path: str, test_every: int = 8) -> List[Image]:
    return [im for i, im in enumerate(sorted_images(path)) if i % test_every == 0]


def rot2qvec(r: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> COLMAP quaternion (w, x, y, z), w >= 0."""
    from .heldout import quat_from_matrix
    x, y, z, w = quat_from_matrix(r)
    return (w, x, y, z)


def write_transformed_model(src: str, dst: str, s: float, rot: np.ndarray, t: np.ndarray) -> None:
    """Apply X' = s * rot @ X + t to a binary COLMAP model (cameras.bin, images.bin, points3D.bin only).

    Cameras: R_i' = R_i rot^T, t_i' = s t_i - R_i' t (CONTRACTS 2.0). Points keep their colour, error and track.
    rigs.bin / frames.bin are not written (gsplat's parser does not read them).
    Written because `colmap model_transformer` 4.2.1 transformed points3D.bin and images.bin with different
    similarities for a non-unit-scale matrix (reprojection error 7215 px afterwards).
    """
    import shutil
    sd = model_dir(src)
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(sd, "cameras.bin"), os.path.join(dst, "cameras.bin"))
    rot = np.asarray(rot, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    with open(os.path.join(sd, "images.bin"), "rb") as f, open(os.path.join(dst, "images.bin"), "wb") as g:
        head = f.read(8)
        g.write(head)
        (n,) = struct.unpack("<Q", head)
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", f.read(64))
            ri = qvec2rot((qw, qx, qy, qz))
            rn = ri @ rot.T
            tn = s * np.array([tx, ty, tz]) - rn @ t
            g.write(struct.pack("<idddddddi", iid, *rot2qvec(rn), *[float(v) for v in tn], cid))
            name = b""
            while True:
                ch = f.read(1)
                g.write(ch)
                if ch == b"\x00" or not ch:
                    break
                name += ch
            n2d = f.read(8)
            g.write(n2d)
            (k,) = struct.unpack("<Q", n2d)
            g.write(f.read(24 * k))
    with open(os.path.join(sd, "points3D.bin"), "rb") as f, open(os.path.join(dst, "points3D.bin"), "wb") as g:
        head = f.read(8)
        g.write(head)
        (n,) = struct.unpack("<Q", head)
        for _ in range(n):
            pid, x, y, z, r, gg, b, err, tl = struct.unpack("<QdddBBBdQ", f.read(51))
            xn = s * rot @ np.array([x, y, z]) + t
            g.write(struct.pack("<QdddBBBdQ", pid, *[float(v) for v in xn], r, gg, b, err, tl))
            g.write(f.read(8 * tl))
