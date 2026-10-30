"""COLMAP model reader and writer (binary and text), numpy only.

Reads what the grid stage needs: image names, world-to-camera poses, the 3D points each image
observes, and the 3D points. The writer exists for tests and for the honka stand-in.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def qvec_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_qvec(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q


@dataclass
class Model:
    names: list[str]
    R: np.ndarray                     # (N, 3, 3) world-to-camera
    t: np.ndarray                     # (N, 3)
    point_ids: list[np.ndarray]       # per image: ids of observed 3D points (>= 0)
    pt_ids: np.ndarray                # (M,) int64 ids of 3D points
    pt_xyz: np.ndarray                # (M, 3)
    raw_images: list = field(default_factory=list, repr=False)   # per image (image_id, camera_id, xy (n,2), pid (n,)); kept for writing
    raw_points: dict = field(default_factory=dict, repr=False)   # extra per-point fields kept for writing

    @property
    def centres(self) -> np.ndarray:
        return -np.einsum("nji,nj->ni", self.R, self.t)       # -R^T t

    @property
    def forward(self) -> np.ndarray:
        return self.R[:, 2, :]                                 # camera z axis in world coordinates


def _read_images_bin(buf: bytes):
    (n,) = struct.unpack_from("<Q", buf, 0)
    pos = 8
    names, Rs, ts, pids, raw = [], [], [], [], []
    pt_dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")])
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from("<idddddddi", buf, pos)
        pos += 64
        end = buf.index(b"\x00", pos)
        name = buf[pos:end].decode("utf-8")
        pos = end + 1
        (npts,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        pts = np.frombuffer(buf, dtype=pt_dt, count=npts, offset=pos)
        pos += 24 * npts
        names.append(name)
        Rs.append(qvec_to_rot(np.array([qw, qx, qy, qz])))
        ts.append([tx, ty, tz])
        ids = pts["pid"]
        pids.append(ids[ids >= 0].astype(np.int64))
        raw.append((iid, cid, np.stack([pts["x"], pts["y"]], axis=1), ids.copy()))
    return names, np.array(Rs), np.array(ts), pids, raw


def _read_points_bin(buf: bytes):
    (n,) = struct.unpack_from("<Q", buf, 0)
    pos = 8
    ids = np.empty(n, dtype=np.int64)
    xyz = np.empty((n, 3))
    extra = []
    for k in range(n):
        pid, x, y, z, r, g, b, err, tl = struct.unpack_from("<QdddBBBdQ", buf, pos)
        pos += 51
        track = buf[pos:pos + 8 * tl]
        pos += 8 * tl
        ids[k], xyz[k] = pid, (x, y, z)
        extra.append((r, g, b, err, track))
    return ids, xyz, extra


def _read_images_txt(text: str):
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]   # an image without points has an empty line
    names, Rs, ts, pids, raw = [], [], [], [], []
    for a, b in zip(lines[0::2], lines[1::2]):
        if not a.strip():
            continue
        p = a.split()
        q = np.array([float(v) for v in p[1:5]])
        names.append(p[9])
        Rs.append(qvec_to_rot(q))
        ts.append([float(v) for v in p[5:8]])
        v = b.split()
        arr = np.array(v, dtype=np.float64).reshape(-1, 3) if v else np.zeros((0, 3))
        ids = arr[:, 2].astype(np.int64)
        pids.append(ids[ids >= 0])
        raw.append((int(p[0]), int(p[8]), arr[:, :2], ids))
    return names, np.array(Rs), np.array(ts), pids, raw


def _read_points_txt(text: str):
    ids, xyz, extra = [], [], []
    for ln in text.splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        p = ln.split()
        ids.append(int(p[0]))
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        extra.append((int(p[4]), int(p[5]), int(p[6]), float(p[7]), b""))
    return np.array(ids, dtype=np.int64), np.array(xyz).reshape(-1, 3), extra


def read_model(model_dir: str | Path) -> Model:
    d = Path(model_dir)
    if (d / "images.bin").exists():
        names, R, t, pids, raw = _read_images_bin((d / "images.bin").read_bytes())
        if (d / "points3D.bin").exists():
            ids, xyz, extra = _read_points_bin((d / "points3D.bin").read_bytes())
        else:
            ids, xyz, extra = np.zeros(0, np.int64), np.zeros((0, 3)), []
    elif (d / "images.txt").exists():
        names, R, t, pids, raw = _read_images_txt((d / "images.txt").read_text())
        if (d / "points3D.txt").exists():
            ids, xyz, extra = _read_points_txt((d / "points3D.txt").read_text())
        else:
            ids, xyz, extra = np.zeros(0, np.int64), np.zeros((0, 3)), []
    else:
        raise FileNotFoundError(f"no images.bin or images.txt in {d}")
    return Model(names, R.reshape(-1, 3, 3), t.reshape(-1, 3), pids, ids, xyz, raw, {"extra": extra})


def write_model(out_dir: str | Path, model: Model, qvecs: np.ndarray | None = None) -> None:
    """Write images.bin and points3D.bin (cameras.bin is not written; copy it from the source)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = bytearray(struct.pack("<Q", len(model.names)))
    for k, name in enumerate(model.names):
        iid, cid, xy, ids = model.raw_images[k] if model.raw_images else (k + 1, 1, np.zeros((0, 2)), np.zeros(0, np.int64))
        q = rot_to_qvec(model.R[k]) if qvecs is None else qvecs[k]
        out += struct.pack("<idddddddi", iid, *q, *model.t[k], cid)
        out += name.encode("utf-8") + b"\x00"
        out += struct.pack("<Q", len(ids))
        if len(ids):
            rec = np.zeros(len(ids), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")]))
            rec["x"], rec["y"], rec["pid"] = xy[:, 0], xy[:, 1], ids
            out += rec.tobytes()
    (d / "images.bin").write_bytes(bytes(out))
    extra = model.raw_points.get("extra") or [(128, 128, 128, 0.5, b"")] * len(model.pt_ids)
    out = bytearray(struct.pack("<Q", len(model.pt_ids)))
    for k in range(len(model.pt_ids)):
        r, g, b, err, track = extra[k]
        out += struct.pack("<QdddBBBdQ", int(model.pt_ids[k]), *model.pt_xyz[k], r, g, b, err, len(track) // 8)
        out += track
    (d / "points3D.bin").write_bytes(bytes(out))
