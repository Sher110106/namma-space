"""COLMAP to three.js camera conventions (CONTRACTS 2.0), standard library only.

Frames
    COLMAP/OpenCV camera: x right, y down, z forward. A pose is world-to-camera:
        x_cam = R * X + t
    three.js camera: x right, y up, looks down -z. A pose is camera-to-world:
        position = C, orientation = quaternion [x, y, z, w]
    World: metres, +Y up, right-handed. 0 yaw looks along -Z, positive yaw turns left.

Mapping (R = R_i' and t = t_i' from the aligned model)
    C                 = -R^T t                        camera centre
    R_threejs         = R^T * diag(1, -1, -1)         camera-to-world rotation
    position          = T_final(C)
    orientation       = R_final * R_threejs           (R_final is the rotation part of T_final)
    fov_y (degrees)   = 2 * atan(H / (2 * fy))
    principal point   = custom projection matrix, see projection_matrix()

T_final is a 4x4 row-major similarity [[s*Rf, tf], [0, 0, 0, 1]] (or None). It carries the
floor offset and the user rescale. Positions go through it fully; orientations only get
the rotation Rf (the scale does not change a direction).

test/data/conventions.json holds poses with their expected outputs. Python tests
(test/e2e/test_conventions.py) and the viewer's JS tests both read it. A convention
change edits that file first, through root.

API
    parse_t_final(obj)                           -> 4x4 row-major list or None
    threejs_from_colmap(R, t, T_final=None)      -> (position[3], quaternion_xyzw[4])
    colmap_from_threejs(position, quat, T_final=None) -> (R, t)
    fov_y_deg(height_px, fy)
    projection_matrix(fx, fy, cx, cy, width, height, near, far)  -> 4x4 row-major
    yaw_pitch_deg(quat_xyzw)                     -> (yaw_deg, pitch_deg), yaw positive = left
    quat_from_yaw_pitch_deg(yaw, pitch)          -> quaternion_xyzw
    quat_from_matrix(R), matrix_from_quat(q)     three.js order [x, y, z, w]
"""
from __future__ import annotations

import math
from typing import Sequence

DIAG_FLIP = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]


def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _T(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def matrix_from_quat(q: Sequence[float]) -> list:
    """Quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def quat_from_matrix(m: Sequence) -> list:
    """3x3 rotation matrix to quaternion [x, y, z, w], sign fixed so the first non-zero
    component among (w, x, y, z) is positive."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    for comp in (w, x, y, z):
        if abs(comp) > 1e-12:
            if comp < 0:
                w, x, y, z = -w, -x, -y, -z
            break
    return [x, y, z, w]


def parse_t_final(obj) -> list | None:
    """T_final in any form the packages write, as a 4x4 row-major list (None stays None).

    Accepted: a 4x4 nested list; a flat list of 16 (row-major); a dict holding one of the keys
    "matrix", "T" or "m" (any of the above); a dict {position|translation, quaternion|rotation [x,y,z,w], scale}
    as in scene.json's splat transform and W3's export.json. Meaning: X_final = scale * Rf * X + translation.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ("matrix", "T", "m"):
            if key in obj:
                return parse_t_final(obj[key])
        pos = obj.get("position", obj.get("translation", [0.0, 0.0, 0.0]))
        quat = obj.get("quaternion", obj.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        sc = obj.get("scale", 1.0)
        sc = sum(sc) / len(sc) if isinstance(sc, (list, tuple)) else float(sc)
        Rf = matrix_from_quat(quat)
        return [[sc * Rf[i][0], sc * Rf[i][1], sc * Rf[i][2], float(pos[i])] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    if isinstance(obj, (list, tuple)):
        if len(obj) == 16 and not isinstance(obj[0], (list, tuple)):
            return [list(obj[i * 4:(i + 1) * 4]) for i in range(4)]
        if len(obj) == 4 and all(len(r) == 4 for r in obj):
            return [list(r) for r in obj]
    raise ValueError(f"cannot read T_final: {obj!r}")


def _split_t_final(T_final):
    """(s, R, t) from T_final (any accepted form), or identity for None."""
    T_final = parse_t_final(T_final)
    if T_final is None:
        return 1.0, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [0.0, 0.0, 0.0]
    A = [[T_final[i][j] for j in range(3)] for i in range(3)]
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
           - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
           + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    if det <= 0:
        raise ValueError("T_final must have a positive determinant")
    s = det ** (1.0 / 3.0)
    R = [[A[i][j] / s for j in range(3)] for i in range(3)]
    return s, R, [T_final[0][3], T_final[1][3], T_final[2][3]]


def threejs_from_colmap(R: Sequence, t: Sequence, T_final=None) -> tuple:
    """COLMAP world-to-camera (R, t) to a three.js camera (position, quaternion [x,y,z,w])."""
    Rt = _T(R)
    C = [-sum(Rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    R_cam = _mm(Rt, DIAG_FLIP)
    s, Rf, tf = _split_t_final(T_final)
    pos = [s * sum(Rf[i][k] * C[k] for k in range(3)) + tf[i] for i in range(3)]
    R_cam = _mm(Rf, R_cam)
    return pos, quat_from_matrix(R_cam)


def colmap_from_threejs(position: Sequence, quat: Sequence, T_final=None) -> tuple:
    """Inverse of threejs_from_colmap: returns COLMAP (R, t) in the aligned model's frame."""
    s, Rf, tf = _split_t_final(T_final)
    Rfi = _T(Rf)
    C = [sum(Rfi[i][k] * (position[k] - tf[k]) for k in range(3)) / s for i in range(3)]
    R_cam = _mm(Rfi, matrix_from_quat(quat))
    R = _T(_mm(R_cam, DIAG_FLIP))  # R_cam = R^T diag  ->  R = (R_cam diag)^T
    t = [-sum(R[i][k] * C[k] for k in range(3)) for i in range(3)]
    return R, t


def fov_y_deg(height_px: float, fy: float) -> float:
    return math.degrees(2.0 * math.atan(height_px / (2.0 * fy)))


def projection_matrix(fx: float, fy: float, cx: float, cy: float, width: float, height: float,
                      near: float, far: float) -> list:
    """OpenGL/three.js projection for COLMAP pinhole intrinsics (pixels, origin top-left).

    Row-major 4x4. In three.js: camera.projectionMatrix.set(...rows...) and copy it to
    projectionMatrixInverse. NDC x = 2u/W - 1, NDC y = 1 - 2v/H, which is what makes the
    principal point land where COLMAP says.
    """
    return [
        [2.0 * fx / width, 0.0, 1.0 - 2.0 * cx / width, 0.0],
        [0.0, 2.0 * fy / height, 2.0 * cy / height - 1.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def forward_world(quat: Sequence[float]) -> list:
    """World direction a three.js camera with this orientation looks along (-Z rotated)."""
    m = matrix_from_quat(quat)
    return [-m[0][2], -m[1][2], -m[2][2]]


def yaw_pitch_deg(quat: Sequence[float]) -> tuple:
    """Yaw about +Y (0 = looking along -Z, positive = turning left) and pitch (positive = up)."""
    f = forward_world(quat)
    yaw = math.degrees(math.atan2(-f[0], -f[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, f[1]))))
    return yaw, pitch


def quat_from_yaw_pitch_deg(yaw: float, pitch: float) -> list:
    """Orientation R_y(yaw) * R_x(pitch) as [x, y, z, w] (three.js 'YXZ' order, no roll)."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    Ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    Rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]
    return quat_from_matrix(_mm(Ry, Rx))
