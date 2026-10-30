"""Stdlib readers for COLMAP binary models and databases (no numpy; runs inside namma-colmap).

Only what the SfM stage needs: registered image names, poses and camera centres, camera model ids,
keypoint counts per image, and the numbers printed by `colmap model_analyzer`.
"""
from __future__ import annotations

import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from typing import BinaryIO

# COLMAP camera model id -> (name, number of parameters). Ids 0-10 are stable across COLMAP 3.x/4.x.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class ImageRec:
    image_id: int
    name: str
    camera_id: int
    qvec: tuple[float, float, float, float]  # w, x, y, z (world-to-camera)
    tvec: tuple[float, float, float]
    n_points2d: int

    @property
    def centre(self) -> tuple[float, float, float]:
        return camera_centre(self.qvec, self.tvec)


@dataclass
class CameraRec:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


def qvec_to_rotmat(q: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def camera_centre(q: tuple[float, float, float, float], t: tuple[float, float, float]) -> tuple[float, float, float]:
    """C = -R^T t for a world-to-camera pose (R from the quaternion q, translation t)."""
    r = qvec_to_rotmat(q)
    return tuple(-(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]) for i in range(3))  # type: ignore[return-value]


def _read(f: BinaryIO, fmt: str) -> tuple:
    size = struct.calcsize(fmt)
    buf = f.read(size)
    if len(buf) != size:
        raise EOFError("truncated COLMAP binary file")
    return struct.unpack("<" + fmt, buf)


def find_model_files(model_dir: str) -> tuple[str, str]:
    """Return (cameras.bin, images.bin) paths; raises FileNotFoundError for text models (we keep binary)."""
    cams, imgs = os.path.join(model_dir, "cameras.bin"), os.path.join(model_dir, "images.bin")
    if not (os.path.isfile(cams) and os.path.isfile(imgs)):
        raise FileNotFoundError(f"no binary COLMAP model in {model_dir}")
    return cams, imgs


def read_images_bin(path: str) -> list[ImageRec]:
    out: list[ImageRec] = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            (image_id,) = _read(f, "i")
            q = _read(f, "dddd")
            t = _read(f, "ddd")
            (camera_id,) = _read(f, "i")
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise EOFError("truncated image name")
                if ch == b"\x00":
                    break
                name_bytes += ch
            (n2d,) = _read(f, "Q")
            f.seek(24 * n2d, os.SEEK_CUR)  # x, y (double) + point3D_id (int64) per point; not needed here
            out.append(ImageRec(image_id, name_bytes.decode("utf-8"), camera_id, q, t, n2d))  # type: ignore[arg-type]
    return out


def read_cameras_bin(path: str) -> list[CameraRec]:
    out: list[CameraRec] = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            camera_id, model_id = _read(f, "ii")
            width, height = _read(f, "QQ")
            name, n_params = CAMERA_MODELS.get(model_id, (f"MODEL_{model_id}", -1))
            if n_params < 0:
                raise ValueError(f"unknown camera model id {model_id}")
            params = _read(f, "d" * n_params)
            out.append(CameraRec(camera_id, name, width, height, params))
    return out


def read_model(model_dir: str) -> tuple[list[CameraRec], list[ImageRec]]:
    cams, imgs = find_model_files(model_dir)
    return read_cameras_bin(cams), read_images_bin(imgs)


def list_models(sparse_dir: str) -> list[str]:
    """Model sub-directories (0, 1, 2 ...) of a mapper output that hold a binary model, sorted numerically."""
    if not os.path.isdir(sparse_dir):
        return []
    subs = [d for d in os.listdir(sparse_dir) if d.isdigit() and os.path.isfile(os.path.join(sparse_dir, d, "images.bin"))]
    return [os.path.join(sparse_dir, d) for d in sorted(subs, key=int)]


def copy_db(src: str, dst: str) -> None:
    """Copy a COLMAP database with SQLite's backup API, so content still in the -wal file is not lost."""
    a = sqlite3.connect(src)
    try:
        b = sqlite3.connect(dst)
        try:
            a.backup(b)
        finally:
            b.close()
    finally:
        a.close()


def keypoint_counts(db_path: str) -> dict[str, int]:
    """Number of keypoints per image name, read straight from the COLMAP database."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT images.name, keypoints.rows FROM images LEFT JOIN keypoints USING (image_id)").fetchall()
    finally:
        con.close()
    return {name: int(n or 0) for name, n in rows}


_NUM = r"([-+0-9.eE]+)"


def parse_model_analyzer(text: str) -> dict[str, float]:
    """Pull the headline numbers out of `colmap model_analyzer` output (glog lines on stderr)."""
    pats = {
        "registered": r"Registered images:\s*(\d+)",
        "points": r"Points:\s*(\d+)",
        "observations": r"Observations:\s*(\d+)",
        "track_len": r"Mean track length:\s*" + _NUM,
        "obs_per_image": r"Mean observations per image:\s*" + _NUM,
        "reproj_px": r"Mean reprojection error:\s*" + _NUM + r"\s*px",
    }
    out: dict[str, float] = {}
    for key, pat in pats.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out
