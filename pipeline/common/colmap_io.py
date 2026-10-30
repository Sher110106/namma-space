"""COLMAP binary model reader and writer, standard library only.

Copy-import friendly: this file has no imports from the rest of the repo.

Model layout handled (COLMAP 3.x and 4.x binary models)
    cameras.bin, images.bin, points3D.bin   parsed
    rigs.bin, frames.bin                    carried through; when a similarity
                                            transform is applied they are updated
                                            as well so the model stays consistent
    anything else in the folder             copied through unchanged
Text models (cameras.txt ...) are refused on purpose: gsplat's pycolmap fork
crashes on text models, so we keep everything binary
(convert with `colmap model_converter --output_type BIN`).

API
    read_model(dir)  -> Model
    write_model(model, dir)                 writes cameras/images/points3D + extras
    similarity_transform(model, s, R, t)    -> new Model with X' = s*R*X + t
    similarity_from_matrix(S4)              -> (s, R, t) from a 4x4 similarity
    apply_matrix(model, S4)                 -> similarity_transform with a 4x4
    model_summary(model)                    -> dict (registered, track length, error)
    reprojection_rmse(model)                -> RMS pixel error over all observations
    camera_centres(model)                   -> {image_id: (x, y, z)}
    qvec_to_rotmat(q), rotmat_to_qvec(R)    quaternions are COLMAP order (w, x, y, z)
    Image.rotmat(), Image.centre()          world-to-camera pose helpers
    Image.points2D, Point3D.track           lazy decoded views of the packed bytes

Conventions: qvec/tvec are COLMAP world-to-camera, x_cam = R(qvec) @ X + tvec.
Matrices are row-major lists of lists (3x3 or 4x4). Nothing here needs numpy.

Command line
    python -m pipeline.common.colmap_io info <model_dir>
    python -m pipeline.common.colmap_io transform <in_dir> <out_dir> --matrix <S4.json|"16 numbers"> [--key S]
    python -m pipeline.common.colmap_io selftest

Camera model ids follow colmap/src/colmap/sensor/models.h (0..11).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

Mat3 = list  # list[list[float]], 3x3
Vec3 = Sequence

CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
MODEL_IDS: dict[str, int] = {name: i for i, (name, _) in CAMERA_MODELS.items()}

_POINT2D = struct.Struct("<ddq")  # x, y, point3D_id (-1 = none)
_TRACK = struct.Struct("<ii")     # image_id, point2D_idx
_IMAGE_HEAD = struct.Struct("<i7di")  # id, qw qx qy qz, tx ty tz, camera_id
_POINT3D_HEAD = struct.Struct("<Q3d3BdQ")  # id, xyz, rgb, error, track_len
_CAMERA_HEAD = struct.Struct("<iiQQ")  # id, model_id, width, height


# --------------------------------------------------------------------------- math

def mat_mul(a: Sequence, b: Sequence) -> list:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def mat_T(a: Sequence) -> list:
    return [list(row) for row in zip(*a)]


def mat_vec(a: Sequence, v: Sequence) -> list:
    return [sum(a[i][k] * v[k] for k in range(len(v))) for i in range(len(a))]


def det3(m: Sequence) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def qvec_to_rotmat(q: Sequence[float]) -> list:
    """COLMAP quaternion (w, x, y, z) to a 3x3 rotation matrix (normalises q)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def rotmat_to_qvec(R: Sequence) -> tuple:
    """3x3 rotation matrix to a COLMAP quaternion (w, x, y, z) with w >= 0.

    Shepperd's method: picks the largest of the four candidate pivots, so it
    stays accurate for rotations near 180 degrees.
    """
    m = R
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return (w, x, y, z)


def check_rotation(R: Sequence, tol: float = 1e-6) -> None:
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i] * R[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > tol:
                raise ValueError(f"matrix is not orthonormal (R^T R [{i}][{j}] = {dot})")
    if det3(R) < 0:
        raise ValueError("matrix is a reflection (det < 0)")


def similarity_from_matrix(S: Sequence, tol: float = 1e-6) -> tuple:
    """Split a 4x4 similarity [[sR, t], [0, 1]] into (s, R, t)."""
    if len(S) != 4 or any(len(r) != 4 for r in S):
        raise ValueError("expected a 4x4 matrix")
    A = [[S[i][j] for j in range(3)] for i in range(3)]
    d = det3(A)
    if d <= 0:
        raise ValueError("matrix has det <= 0, not a similarity")
    s = d ** (1.0 / 3.0)
    R = [[A[i][j] / s for j in range(3)] for i in range(3)]
    check_rotation(R, tol)
    if any(abs(S[3][j] - (1.0 if j == 3 else 0.0)) > tol for j in range(4)):
        raise ValueError("last row of the 4x4 matrix must be [0, 0, 0, 1]")
    return s, R, [S[0][3], S[1][3], S[2][3]]


def matrix_from_similarity(s: float, R: Sequence, t: Sequence) -> list:
    return [[s * R[i][0], s * R[i][1], s * R[i][2], t[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


# --------------------------------------------------------------------------- data

@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list

    @property
    def model_id(self) -> int:
        return MODEL_IDS[self.model]


@dataclass
class Image:
    """One registered image. qvec (w,x,y,z) and tvec are world-to-camera."""
    image_id: int
    qvec: tuple
    tvec: tuple
    camera_id: int
    name: str
    points2D_raw: bytes = b""  # n * 24 bytes, packed "<ddq"

    @property
    def num_points2D(self) -> int:
        return len(self.points2D_raw) // _POINT2D.size

    @property
    def points2D(self) -> list:
        """[(x, y, point3D_id or -1), ...] decoded on demand."""
        return list(_POINT2D.iter_unpack(self.points2D_raw))

    @points2D.setter
    def points2D(self, pts: Iterable) -> None:
        self.points2D_raw = b"".join(_POINT2D.pack(x, y, pid) for x, y, pid in pts)

    def rotmat(self) -> list:
        return qvec_to_rotmat(self.qvec)

    def centre(self) -> list:
        """Camera centre in world coordinates: -R^T t."""
        R = self.rotmat()
        return [-sum(R[k][i] * self.tvec[k] for k in range(3)) for i in range(3)]


@dataclass
class Point3D:
    point3D_id: int
    xyz: tuple
    rgb: tuple
    error: float
    track_raw: bytes = b""  # n * 8 bytes, packed "<ii" (image_id, point2D_idx)

    @property
    def track(self) -> list:
        return list(_TRACK.iter_unpack(self.track_raw))

    @track.setter
    def track(self, pairs: Iterable) -> None:
        self.track_raw = b"".join(_TRACK.pack(i, j) for i, j in pairs)

    @property
    def track_length(self) -> int:
        return len(self.track_raw) // _TRACK.size


@dataclass
class Model:
    cameras: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    points3D: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)     # file name -> bytes (rigs.bin, frames.bin, ...)
    warnings: list = field(default_factory=list)


# --------------------------------------------------------------------------- read

def read_cameras(buf: bytes) -> dict:
    cams: dict = {}
    (n,) = struct.unpack_from("<Q", buf, 0)
    off = 8
    for _ in range(n):
        cid, mid, w, h = _CAMERA_HEAD.unpack_from(buf, off)
        off += _CAMERA_HEAD.size
        if mid not in CAMERA_MODELS:
            raise ValueError(f"unknown COLMAP camera model id {mid}")
        name, npar = CAMERA_MODELS[mid]
        params = list(struct.unpack_from(f"<{npar}d", buf, off))
        off += 8 * npar
        cams[cid] = Camera(cid, name, w, h, params)
    return cams


def read_images(buf: bytes) -> dict:
    imgs: dict = {}
    (n,) = struct.unpack_from("<Q", buf, 0)
    off = 8
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = _IMAGE_HEAD.unpack_from(buf, off)
        off += _IMAGE_HEAD.size
        end = buf.index(b"\0", off)
        name = buf[off:end].decode("utf-8")
        off = end + 1
        (n2d,) = struct.unpack_from("<Q", buf, off)
        off += 8
        raw = bytes(buf[off:off + n2d * _POINT2D.size])
        off += n2d * _POINT2D.size
        imgs[iid] = Image(iid, (qw, qx, qy, qz), (tx, ty, tz), cid, name, raw)
    if off != len(buf):
        raise ValueError(f"images.bin: {len(buf) - off} trailing bytes")
    return imgs


def read_points3D(buf: bytes) -> dict:
    pts: dict = {}
    (n,) = struct.unpack_from("<Q", buf, 0)
    off = 8
    for _ in range(n):
        pid, x, y, z, r, g, b, err, tl = _POINT3D_HEAD.unpack_from(buf, off)
        off += _POINT3D_HEAD.size
        raw = bytes(buf[off:off + tl * _TRACK.size])
        off += tl * _TRACK.size
        pts[pid] = Point3D(pid, (x, y, z), (r, g, b), err, raw)
    if off != len(buf):
        raise ValueError(f"points3D.bin: {len(buf) - off} trailing bytes")
    return pts


def read_model(path) -> Model:
    """Read a binary COLMAP model folder. Extra files are kept as raw bytes."""
    p = Path(path)
    if not (p / "cameras.bin").exists():
        if (p / "cameras.txt").exists():
            raise ValueError(f"{p} is a text model; convert it with "
                             "`colmap model_converter --output_type BIN`")
        raise FileNotFoundError(f"no cameras.bin in {p}")
    m = Model()
    m.cameras = read_cameras((p / "cameras.bin").read_bytes())
    m.images = read_images((p / "images.bin").read_bytes())
    m.points3D = read_points3D((p / "points3D.bin").read_bytes())
    for f in sorted(p.iterdir()):
        if f.is_file() and f.name not in ("cameras.bin", "images.bin", "points3D.bin"):
            m.extra[f.name] = f.read_bytes()
    return m


# --------------------------------------------------------------------------- write

def write_model(model: Model, path) -> None:
    """Write cameras.bin, images.bin, points3D.bin and every extra file."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    out = [struct.pack("<Q", len(model.cameras))]
    for cid in sorted(model.cameras):
        c = model.cameras[cid]
        mid = MODEL_IDS[c.model]
        if len(c.params) != CAMERA_MODELS[mid][1]:
            raise ValueError(f"camera {cid}: {c.model} needs {CAMERA_MODELS[mid][1]} params")
        out.append(_CAMERA_HEAD.pack(cid, mid, c.width, c.height))
        out.append(struct.pack(f"<{len(c.params)}d", *c.params))
    (p / "cameras.bin").write_bytes(b"".join(out))

    out = [struct.pack("<Q", len(model.images))]
    for iid in sorted(model.images):
        im = model.images[iid]
        out.append(_IMAGE_HEAD.pack(iid, *im.qvec, *im.tvec, im.camera_id))
        out.append(im.name.encode("utf-8") + b"\0")
        out.append(struct.pack("<Q", im.num_points2D))
        out.append(im.points2D_raw)
    (p / "images.bin").write_bytes(b"".join(out))

    out = [struct.pack("<Q", len(model.points3D))]
    for pid in sorted(model.points3D):
        pt = model.points3D[pid]
        out.append(_POINT3D_HEAD.pack(pid, *pt.xyz, *pt.rgb, pt.error, pt.track_length))
        out.append(pt.track_raw)
    (p / "points3D.bin").write_bytes(b"".join(out))

    for name, data in model.extra.items():
        (p / name).write_bytes(data)


# --------------------------------------------------------------------------- rigs and frames

def _transform_frames_bin(raw: bytes, s: float, R: Sequence, t: Sequence) -> bytes:
    """Update rig_from_world in frames.bin (COLMAP >= 3.12 layout, qvec as w,x,y,z).

    frame: frame_id u32, rig_id u32, qvec 4d, tvec 3d, n u32, n x (sensor type i32, sensor id u32, data id u64).
    """
    (n,) = struct.unpack_from("<Q", raw, 0)
    out = bytearray(raw[:8])
    off = 8
    for _ in range(n):
        head = struct.unpack_from("<II4d3dI", raw, off)
        fid, rid, qw, qx, qy, qz, tx, ty, tz, nd = head
        Rf = mat_mul(qvec_to_rotmat((qw, qx, qy, qz)), mat_T(R))
        tf = [s * (tx, ty, tz)[i] - sum(Rf[i][k] * t[k] for k in range(3)) for i in range(3)]
        q2 = rotmat_to_qvec(Rf)
        out += struct.pack("<II4d3dI", fid, rid, *q2, *tf, nd)
        off += struct.calcsize("<II4d3dI")
        out += raw[off:off + 16 * nd]
        off += 16 * nd
    if off != len(raw):
        raise ValueError("frames.bin: unexpected trailing bytes")
    return bytes(out)


def _transform_rigs_bin(raw: bytes, s: float) -> bytes:
    """rigs.bin needs no change for single-camera rigs (one sensor, no sensor_from_rig).

    Multi-sensor rigs would need their sensor_from_rig translations scaled by s. We never
    produce them (one phone, one camera), so they raise and the caller copies the file with a warning.
    """
    (n,) = struct.unpack_from("<Q", raw, 0)
    off = 8
    for _ in range(n):
        _rid, ns = struct.unpack_from("<II", raw, off)
        if ns != 1:
            raise ValueError(f"rig with {ns} sensors is not supported")
        off += 16
    if off != len(raw):
        raise ValueError("rigs.bin: unexpected trailing bytes")
    return raw


# --------------------------------------------------------------------------- similarity

def similarity_transform(model: Model, s: float, R: Sequence, t: Sequence) -> Model:
    """Apply X' = s*R*X + t to a model and return the new model.

    Points: X' = s*R*X + t. Cameras: R_i' = R_i R^T, t_i' = s*t_i - R_i' t.
    Registered count, 2D observations, tracks and reprojection errors do not change.
    The input model is not modified (immutable byte blobs are shared).
    """
    if s <= 0:
        raise ValueError("scale must be positive")
    check_rotation(R)
    Rt = mat_T(R)
    out = Model(cameras={k: Camera(c.camera_id, c.model, c.width, c.height, list(c.params))
                         for k, c in model.cameras.items()},
                warnings=list(model.warnings))
    for iid, im in model.images.items():
        Ri = qvec_to_rotmat(im.qvec)
        Rn = mat_mul(Ri, Rt)
        tn = [s * im.tvec[i] - sum(Rn[i][k] * t[k] for k in range(3)) for i in range(3)]
        out.images[iid] = Image(iid, rotmat_to_qvec(Rn), tuple(tn), im.camera_id, im.name, im.points2D_raw)
    for pid, pt in model.points3D.items():
        sx = [s * sum(R[i][k] * pt.xyz[k] for k in range(3)) + t[i] for i in range(3)]
        out.points3D[pid] = Point3D(pid, tuple(sx), pt.rgb, pt.error, pt.track_raw)
    for name, data in model.extra.items():
        try:
            if name == "frames.bin":
                data = _transform_frames_bin(data, s, R, t)
            elif name == "rigs.bin":
                data = _transform_rigs_bin(data, s)
        except (struct.error, ValueError) as exc:
            out.warnings.append(f"{name} could not be transformed ({exc}); copied unchanged")
        out.extra[name] = data
    return out


def apply_matrix(model: Model, S: Sequence) -> Model:
    s, R, t = similarity_from_matrix(S)
    return similarity_transform(model, s, R, t)


# --------------------------------------------------------------------------- analysis

def camera_centres(model: Model) -> dict:
    return {iid: im.centre() for iid, im in model.images.items()}


def project(camera: Camera, X: Sequence) -> tuple:
    """Project a camera-frame point to pixels. Supports the common perspective models."""
    x, y, z = X
    if z <= 0:
        raise ValueError("point behind the camera")
    u, v = x / z, y / z
    m, p = camera.model, camera.params
    if m == "SIMPLE_PINHOLE":
        return p[0] * u + p[1], p[0] * v + p[2]
    if m == "PINHOLE":
        return p[0] * u + p[2], p[1] * v + p[3]
    r2 = u * u + v * v
    if m == "SIMPLE_RADIAL":
        d = 1 + p[3] * r2
        return p[0] * u * d + p[1], p[0] * v * d + p[2]
    if m == "RADIAL":
        d = 1 + p[3] * r2 + p[4] * r2 * r2
        return p[0] * u * d + p[1], p[0] * v * d + p[2]
    if m == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        rad = k1 * r2 + k2 * r2 * r2
        du = u * rad + 2 * p1 * u * v + p2 * (r2 + 2 * u * u)
        dv = v * rad + p1 * (r2 + 2 * v * v) + 2 * p2 * u * v
        return fx * (u + du) + cx, fy * (v + dv) + cy
    raise NotImplementedError(f"projection for {m}")


def reprojection_rmse(model: Model, max_points: int | None = None) -> tuple:
    """(rms_px, n_obs) over track observations, computed from poses and 2D keypoints."""
    se, n = 0.0, 0
    rot = {iid: qvec_to_rotmat(im.qvec) for iid, im in model.images.items()}
    pts2d: dict = {}
    for k, (pid, pt) in enumerate(model.points3D.items()):
        if max_points is not None and k >= max_points:
            break
        for iid, j in _TRACK.iter_unpack(pt.track_raw):
            im = model.images.get(iid)
            if im is None:
                continue
            if iid not in pts2d:
                pts2d[iid] = im.points2D
            x, y, _ = pts2d[iid][j]
            R = rot[iid]
            Xc = [sum(R[a][b] * pt.xyz[b] for b in range(3)) + im.tvec[a] for a in range(3)]
            if Xc[2] <= 0:
                continue
            u, v = project(model.cameras[im.camera_id], Xc)
            se += (u - x) ** 2 + (v - y) ** 2
            n += 1
    return (math.sqrt(se / n) if n else float("nan")), n


def model_summary(model: Model) -> dict:
    """Counts and mean errors, the same quantities `colmap model_analyzer` prints."""
    n_obs = sum(p.track_length for p in model.points3D.values())
    errs = [p.error for p in model.points3D.values() if p.error >= 0]
    centres = list(camera_centres(model).values())
    box = None
    if centres:
        box = [[min(c[i] for c in centres) for i in range(3)], [max(c[i] for c in centres) for i in range(3)]]
    return {
        "cameras": len(model.cameras),
        "registered_images": len(model.images),
        "points3D": len(model.points3D),
        "observations": n_obs,
        "mean_track_length": (n_obs / len(model.points3D)) if model.points3D else 0.0,
        "mean_reprojection_error": (sum(errs) / len(errs)) if errs else float("nan"),
        "camera_centre_box": box,
        "extra_files": sorted(model.extra),
        "warnings": list(model.warnings),
    }


# --------------------------------------------------------------------------- CLI

def _load_matrix(spec: str, key: str | None) -> list:
    p = Path(spec)
    if p.exists():
        obj = json.loads(p.read_text())
        if key:
            for part in key.split("."):
                obj = obj[part]
        elif isinstance(obj, dict):
            obj = obj.get("S", obj)
    else:
        obj = [float(x) for x in spec.replace(",", " ").split()]
        if len(obj) != 16:
            raise SystemExit("--matrix needs a JSON file or 16 numbers")
        obj = [obj[i * 4:(i + 1) * 4] for i in range(4)]
    return obj


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="COLMAP binary model tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info")
    a.add_argument("model")
    a.add_argument("--reproj", action="store_true", help="also recompute reprojection RMSE (slow on big models)")
    b = sub.add_parser("transform")
    b.add_argument("src")
    b.add_argument("dst")
    b.add_argument("--matrix", required=True)
    b.add_argument("--key", default=None, help="dotted key when --matrix is a JSON file, e.g. S")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    if args.cmd == "info":
        m = read_model(args.model)
        info = model_summary(m)
        if args.reproj:
            info["reprojection_rmse_px"], info["reprojection_obs"] = reprojection_rmse(m)
        print(json.dumps(info, indent=2))
    elif args.cmd == "transform":
        m = read_model(args.src)
        out = apply_matrix(m, _load_matrix(args.matrix, args.key))
        write_model(out, args.dst)
        print(json.dumps(model_summary(out), indent=2))
    else:
        return _selftest()
    return 0


def _selftest() -> int:
    m = Model(cameras={1: Camera(1, "PINHOLE", 100, 80, [50.0, 50.0, 50.0, 40.0])})
    m.images[1] = Image(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "a.jpg")
    m.images[1].points2D = [(50.0, 40.0, 1)]
    m.points3D[1] = Point3D(1, (0.0, 0.0, 2.0), (1, 2, 3), 0.1)
    m.points3D[1].track = [(1, 0)]
    out = similarity_transform(m, 2.0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1.0, 0.0, 0.0])
    ok = abs(out.points3D[1].xyz[0] - 1.0) < 1e-12 and abs(out.points3D[1].xyz[2] - 4.0) < 1e-12
    print("selftest", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
