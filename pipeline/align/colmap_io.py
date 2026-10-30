"""Read and patch COLMAP binary models (3.x and 4.x: with or without rigs.bin / frames.bin).

Only poses and points change under a similarity, so the model is held as the original bytes
and the numbers that change are patched in place.  Everything else (cameras, 2D points,
tracks, colours, errors, the rig layout) stays byte-identical, which is the best guarantee
that COLMAP reads the result as the same reconstruction.

Layouts (little endian):
  cameras.bin  u64 n; per camera: u32 id, i32 model, u64 w, u64 h, f64 params[model-specific]   (copied verbatim)
  images.bin   u64 n; per image: u32 id, f64 qw qx qy qz, f64 tx ty tz, u32 camera_id, name\\0,
               u64 n2d, n2d x (f64 x, f64 y, u64 point3d_id)
  points3D.bin u64 n; per point: u64 id, f64 xyz[3], u8 rgb[3], f64 error, u64 track_len,
               track_len x (u32 image_id, u32 point2d_idx)
  rigs.bin     u64 n; per rig: u32 rig_id, u32 n_sensors, [i32 type, u32 id for the reference sensor],
               per non-reference sensor: i32 type, u32 id, u8 has_pose, [f64 qw qx qy qz tx ty tz]
  frames.bin   u64 n; per frame: u32 frame_id, u32 rig_id, f64 qw qx qy qz, f64 tx ty tz, u32 n_data,
               n_data x (i32 sensor type, u32 sensor id, u64 data id)       (rig_from_world pose)
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to (w, x, y, z), w >= 0 (Shepperd's method)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


@dataclass
class Model:
    cameras: bytes
    images: bytearray
    points: bytearray
    frames: Optional[bytearray] = None
    rigs: Optional[bytearray] = None
    # parsed views
    image_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))
    image_name: List[str] = field(default_factory=list)
    camera_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))   # cam_from_world, (w, x, y, z)
    t: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    _img_off: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # byte offset of qvec
    pt_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))
    xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    err: np.ndarray = field(default_factory=lambda: np.zeros(0))
    track_len: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    _pt_off: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # byte offset of xyz

    @property
    def n_images(self) -> int:
        return len(self.image_name)

    @property
    def n_points(self) -> int:
        return len(self.xyz)

    def rotations(self) -> np.ndarray:
        return np.stack([qvec_to_rotmat(q) for q in self.q])

    def centers(self) -> np.ndarray:
        R = self.rotations()
        return -np.einsum("nji,nj->ni", R, self.t)   # -R^T t

    def mean_error(self) -> float:
        return float(self.err.mean()) if len(self.err) else float("nan")

    def mean_track_length(self) -> float:
        return float(self.track_len.mean()) if len(self.track_len) else float("nan")


def _scan_images(buf: bytes):
    n = struct.unpack_from("<Q", buf, 0)[0]
    o = 8
    ids = np.zeros(n, np.uint32)
    cams = np.zeros(n, np.uint32)
    q = np.zeros((n, 4))
    t = np.zeros((n, 3))
    offs = np.zeros(n, np.int64)
    names: List[str] = []
    for i in range(n):
        ids[i] = struct.unpack_from("<I", buf, o)[0]
        offs[i] = o + 4
        v = struct.unpack_from("<7d", buf, o + 4)
        q[i] = v[:4]
        t[i] = v[4:]
        cams[i] = struct.unpack_from("<I", buf, o + 60)[0]
        e = buf.index(b"\0", o + 64)
        names.append(buf[o + 64:e].decode("utf8"))
        n2d = struct.unpack_from("<Q", buf, e + 1)[0]
        o = e + 1 + 8 + n2d * 24
    if o != len(buf):
        raise ValueError(f"images.bin: parsed {o} of {len(buf)} bytes; layout not as expected")
    return ids, names, cams, q, t, offs


def _scan_points(buf: bytes):
    n = struct.unpack_from("<Q", buf, 0)[0]
    o = 8
    ids = np.zeros(n, np.uint64)
    xyz = np.zeros((n, 3))
    rgb = np.zeros((n, 3), np.uint8)
    err = np.zeros(n)
    tl = np.zeros(n, np.int64)
    offs = np.zeros(n, np.int64)
    unpack = struct.unpack_from
    for i in range(n):
        ids[i] = unpack("<Q", buf, o)[0]
        offs[i] = o + 8
        xyz[i] = unpack("<3d", buf, o + 8)
        rgb[i] = unpack("<3B", buf, o + 32)
        err[i] = unpack("<d", buf, o + 35)[0]
        k = unpack("<Q", buf, o + 43)[0]
        tl[i] = k
        o += 51 + k * 8
    if o != len(buf):
        raise ValueError(f"points3D.bin: parsed {o} of {len(buf)} bytes; layout not as expected")
    return ids, xyz, rgb, err, tl, offs


def load_model(d: str) -> Model:
    def rd(name: str, required: bool = True) -> Optional[bytes]:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            if required:
                raise FileNotFoundError(f"{p} missing (binary COLMAP model expected; convert text models with "
                                        f"`colmap model_converter --output_type BIN`)")
            return None
        with open(p, "rb") as fh:
            return fh.read()

    m = Model(cameras=rd("cameras.bin"), images=bytearray(rd("images.bin")), points=bytearray(rd("points3D.bin")))
    fr, rg = rd("frames.bin", False), rd("rigs.bin", False)
    m.frames = bytearray(fr) if fr is not None else None
    m.rigs = bytearray(rg) if rg is not None else None
    m.image_id, m.image_name, m.camera_id, m.q, m.t, m._img_off = _scan_images(bytes(m.images))
    m.pt_id, m.xyz, m.rgb, m.err, m.track_len, m._pt_off = _scan_points(bytes(m.points))
    return m


def camera_summary(buf: bytes) -> List[dict]:
    """Camera ids, model ids, sizes and params (for reporting only)."""
    n = struct.unpack_from("<Q", buf, 0)[0]
    nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
    models = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
              6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}
    o = 8
    out = []
    for _ in range(n):
        cid, mid, w, h = struct.unpack_from("<IiQQ", buf, o)
        k = nparams.get(mid, 0)
        params = struct.unpack_from(f"<{k}d", buf, o + 24)
        out.append({"id": cid, "model": models.get(mid, str(mid)), "width": w, "height": h, "params": list(params)})
        o += 24 + 8 * k
    return out


def apply_similarity(m: Model, s: float, R: np.ndarray, t: np.ndarray) -> Model:
    """X' = s R X + t on points; cameras get R_i' = R_i R^T, t_i' = s t_i - R_i' t.  Returns a new Model."""
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    out = Model(cameras=m.cameras, images=bytearray(m.images), points=bytearray(m.points),
                frames=bytearray(m.frames) if m.frames is not None else None,
                rigs=bytearray(m.rigs) if m.rigs is not None else None)
    out.image_id, out.image_name, out.camera_id = m.image_id.copy(), list(m.image_name), m.camera_id.copy()
    out._img_off, out.pt_id, out.rgb, out.err, out.track_len, out._pt_off = (
        m._img_off.copy(), m.pt_id.copy(), m.rgb.copy(), m.err.copy(), m.track_len.copy(), m._pt_off.copy())
    # images
    out.q, out.t = np.zeros_like(m.q), np.zeros_like(m.t)
    for i in range(m.n_images):
        Ri = qvec_to_rotmat(m.q[i]) @ R.T
        ti = s * m.t[i] - Ri @ t
        qi = rotmat_to_qvec(Ri)
        out.q[i], out.t[i] = qi, ti
        struct.pack_into("<7d", out.images, int(m._img_off[i]), *qi, *ti)
    # points
    out.xyz = (s * (R @ m.xyz.T)).T + t
    for i in range(m.n_points):
        struct.pack_into("<3d", out.points, int(m._pt_off[i]), *out.xyz[i])
    # frames and rigs
    if m.frames is not None:
        n = struct.unpack_from("<Q", m.frames, 0)[0]
        o = 8
        for _ in range(n):
            q = np.array(struct.unpack_from("<4d", m.frames, o + 8))
            tv = np.array(struct.unpack_from("<3d", m.frames, o + 40))
            Ri = qvec_to_rotmat(q) @ R.T
            ti = s * tv - Ri @ t
            struct.pack_into("<7d", out.frames, o + 8, *rotmat_to_qvec(Ri), *ti)
            nd = struct.unpack_from("<I", m.frames, o + 64)[0]
            o += 68 + nd * 16
        if o != len(m.frames):
            raise ValueError("frames.bin: layout not as expected")
    if m.rigs is not None:
        n = struct.unpack_from("<Q", m.rigs, 0)[0]
        o = 8
        for _ in range(n):
            _rid, ns = struct.unpack_from("<II", m.rigs, o)
            o += 8
            if ns > 0:
                o += 8                                  # reference sensor
            for _k in range(max(ns - 1, 0)):
                o += 8
                has_pose = struct.unpack_from("<B", m.rigs, o)[0]
                o += 1
                if has_pose:
                    tv = np.array(struct.unpack_from("<3d", m.rigs, o + 32))
                    struct.pack_into("<3d", out.rigs, o + 32, *(s * tv))   # sensor_from_rig translation scales
                    o += 56
        if o != len(m.rigs):
            raise ValueError("rigs.bin: layout not as expected")
    return out


def save_model(m: Model, d: str) -> None:
    os.makedirs(d, exist_ok=True)
    for name, buf in (("cameras.bin", m.cameras), ("images.bin", m.images), ("points3D.bin", m.points),
                      ("frames.bin", m.frames), ("rigs.bin", m.rigs)):
        if buf is not None:
            with open(os.path.join(d, name), "wb") as fh:
                fh.write(bytes(buf))


def reproj_error(m: Model) -> Optional[dict]:
    """Mean reprojection error over every track observation, from the keypoints stored in images.bin.

    Independent of COLMAP's own bookkeeping: it re-projects each 3D point through each image's pose and
    PINHOLE / SIMPLE_PINHOLE intrinsics.  Returns None for other camera models."""
    cams = {c["id"]: c for c in camera_summary(m.cameras)}
    if any(c["model"] not in ("PINHOLE", "SIMPLE_PINHOLE") for c in cams.values()):
        return None
    ibuf, pbuf = bytes(m.images), bytes(m.points)
    # keypoints per image
    kp = {}
    o = 8
    n = struct.unpack_from("<Q", ibuf, 0)[0]
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<u8")])
    for i in range(n):
        e = ibuf.index(b"\0", o + 64)
        n2d = struct.unpack_from("<Q", ibuf, e + 1)[0]
        kp[int(m.image_id[i])] = np.frombuffer(ibuf, dt, count=n2d, offset=e + 9)
        o = e + 9 + n2d * 24
    idx_of = {int(v): i for i, v in enumerate(m.image_id)}
    Rs = m.rotations()
    errs = []
    for j in range(m.n_points):
        off = int(m._pt_off[j]) + 43              # start of the track
        k = int(m.track_len[j])
        if k == 0:
            continue
        tr = np.frombuffer(pbuf, "<u4", count=2 * k, offset=off).reshape(k, 2)
        X = m.xyz[j]
        for img, p2 in tr:
            ii = idx_of.get(int(img))
            if ii is None:
                continue
            c = cams[int(m.camera_id[ii])]
            xc = Rs[ii] @ X + m.t[ii]
            if xc[2] <= 0:
                continue
            if c["model"] == "PINHOLE":
                fx, fy, cx, cy = c["params"]
            else:
                fx = fy = c["params"][0]
                cx, cy = c["params"][1], c["params"][2]
            u, v = fx * xc[0] / xc[2] + cx, fy * xc[1] / xc[2] + cy
            q = kp[int(img)][int(p2)]
            errs.append(math.hypot(u - q["x"], v - q["y"]))
    e = np.array(errs)
    return {"mean_px": float(e.mean()), "median_px": float(np.median(e)), "n_obs": int(len(e))}


def image_observations(m: Model, i: int) -> np.ndarray:
    """Structured array (x, y, p) of the keypoints of image index i; p is the point3D id (2**64-1 for untriangulated)."""
    ibuf = bytes(m.images)
    o = int(m._img_off[i]) + 60
    e = ibuf.index(b"\0", o + 4)
    n2d = struct.unpack_from("<Q", ibuf, e + 1)[0]
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<u8")])
    return np.frombuffer(ibuf, dt, count=n2d, offset=e + 9)


def point_tracks(m: Model) -> dict:
    """point3D id -> array of image ids in its track."""
    pbuf = bytes(m.points)
    out = {}
    for j in range(m.n_points):
        off = int(m._pt_off[j]) + 43
        k = int(m.track_len[j])
        if k:
            out[int(m.pt_id[j])] = np.frombuffer(pbuf, "<u4", count=2 * k, offset=off).reshape(k, 2)[:, 0]
    return out
