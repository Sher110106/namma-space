"""Writes test/data/conventions.json from first principles (does not import conventions.py).

Each case is built from the three.js side (yaw, pitch, position), where the expected
numbers are easy to check by hand, then turned into COLMAP (R, t) by the textbook route:
    R_cam2world(COLMAP axes) = R_threejs * diag(1, -1, -1)
    R_w2c = R_cam2world^T,   t = -R_w2c * C
Run: python3 test/e2e/gen_conventions.py   (root only; changes need a convention change)
"""
import json
import math
from pathlib import Path


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def T(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def Ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def Rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def qmul(a, b):  # Hamilton product, [x, y, z, w]
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def qaxis(axis, deg):
    h = math.radians(deg) / 2
    return [axis[0] * math.sin(h), axis[1] * math.sin(h), axis[2] * math.sin(h), math.cos(h)]


def canon(q):  # first non-zero of (w, x, y, z) positive
    for c in (q[3], q[0], q[1], q[2]):
        if abs(c) > 1e-12:
            if c < 0:
                q = [-v for v in q]
            break
    return q


DIAG = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]


def case(cid, note, yaw, pitch, C, flip_start=False):
    """Orientation = Ry(yaw) * Rx(pitch); yaw >0 turns left (looks toward -X from -Z)."""
    if flip_start:  # COLMAP identity: three.js camera turned 180 deg about X (looks +Z)
        R_three = DIAG
        q = [1.0, 0.0, 0.0, 0.0]
    else:
        R_three = mm(Ry(yaw), Rx(pitch))
        q = canon(qmul(qaxis([0, 1, 0], yaw), qaxis([1, 0, 0], pitch)))
    R_c2w = mm(R_three, DIAG)
    R_w2c = T(R_c2w)
    t = [-v for v in mv(R_w2c, C)]
    fwd = mv(R_three, [0, 0, -1])
    return {
        "id": cid, "note": note,
        "colmap": {"R_w2c": R_w2c, "t_w2c": t},
        "threejs": {"position": C, "quaternion_xyzw": q, "rotation_matrix": R_three,
                    "forward_world": fwd},
    }


def clean(o):
    """Round to 12 decimals and drop -0.0 so the JSON reads cleanly."""
    if isinstance(o, float):
        r = round(o, 12)
        return 0.0 if r == 0 else r
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    return o


def _flat(o):
    return isinstance(o, (int, float)) or (isinstance(o, list) and all(_flat(v) for v in o))


def dumps(o, level=0):
    """JSON with numeric arrays on one line, objects and lists of objects indented."""
    pad = " " * (level + 1)
    if _flat(o) or isinstance(o, (str, bool)) or o is None:
        return json.dumps(o)
    if isinstance(o, dict):
        return "{\n" + ",\n".join(f"{pad}{json.dumps(k)}: {dumps(v, level + 1)}" for k, v in o.items()) + "\n" + " " * level + "}"
    return "[\n" + ",\n".join(pad + dumps(v, level + 1) for v in o) + "\n" + " " * level + "]"


def main():
    poses = [
        case("identity", "COLMAP identity pose: camera at the origin looking along +Z. In three.js that is a "
             "180 degree turn about X, quaternion (1,0,0,0).", 0, 0, [0.0, 0.0, 0.0], flip_start=True),
        case("level_minus_z", "Level camera at (1,1.4,2) looking along -Z (three.js default). R_w2c = diag(1,-1,-1), "
             "t = -R*C = (-1,1.4,2). Quaternion is identity.", 0, 0, [1.0, 1.4, 2.0]),
        case("yaw_left_90", "Camera at (-0.5,1.5,3) turned 90 degrees left about +Y: looks along -X. "
             "Quaternion (0, sin45, 0, cos45).", 90, 0, [-0.5, 1.5, 3.0]),
        case("pitch_up_30", "Camera at (2,1.2,-1) looking 30 degrees above level along -Z. "
             "Quaternion (sin15, 0, 0, cos15); forward = (0, 0.5, -0.866).", 0, 30, [2.0, 1.2, -1.0]),
    ]
    by_id = {p["id"]: p for p in poses}
    for p, (yaw, pitch) in zip(poses[1:], [(0, 0), (90, 0), (0, 30)]):
        p["threejs"]["yaw_deg"] = yaw
        p["threejs"]["pitch_deg"] = pitch

    # T_final cases: P_final = s * Rf * P + tf, orientation = Rf * R_threejs
    t_cases = []
    # (a) scale 2 and a translation, no rotation
    base = by_id["yaw_left_90"]
    s, tf = 2.0, [1.0, -0.5, 0.25]
    C = base["threejs"]["position"]
    T_a = [[s, 0, 0, tf[0]], [0, s, 0, tf[1]], [0, 0, s, tf[2]], [0, 0, 0, 1]]
    t_cases.append({"id": "scale2_shift", "pose": base["id"], "T_final": T_a,
                    "expected": {"position": [s * C[i] + tf[i] for i in range(3)],
                                 "quaternion_xyzw": base["threejs"]["quaternion_xyzw"]}})
    # (b) scale 0.5, a 90 degree yaw (turn left) and a translation
    base = by_id["level_minus_z"]
    s, tf = 0.5, [0.0, 0.1, 0.0]
    Rf = Ry(90)
    C = base["threejs"]["position"]
    A = [[s * Rf[i][j] for j in range(3)] for i in range(3)]
    T_b = [A[0] + [tf[0]], A[1] + [tf[1]], A[2] + [tf[2]], [0, 0, 0, 1]]
    pos = [sum(A[i][k] * C[k] for k in range(3)) + tf[i] for i in range(3)]
    q = canon(qmul(qaxis([0, 1, 0], 90), base["threejs"]["quaternion_xyzw"]))
    t_cases.append({"id": "scale_half_yaw90", "pose": base["id"], "T_final": T_b,
                    "expected": {"position": pos, "quaternion_xyzw": q}})

    intr = []
    for (w, h, fx, fy, cx, cy) in [(1920, 1080, 1500.0, 1500.0, 960.0, 540.0),
                                   (1080, 1920, 1200.0, 1210.0, 545.0, 950.0)]:
        near, far = 0.05, 50.0
        intr.append({
            "width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "near": near, "far": far,
            "fov_y_deg": math.degrees(2 * math.atan(h / (2 * fy))),
            "projection_matrix_rowmajor": [
                [2 * fx / w, 0, 1 - 2 * cx / w, 0],
                [0, 2 * fy / h, 2 * cy / h - 1, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0]],
        })

    doc = {
        "schema": "namma-space/conventions@1",
        "doc": "COLMAP world-to-camera (R, t) to three.js camera. position = T_final(-R^T t); "
               "rotation = R_final * R^T * diag(1,-1,-1); quaternion is [x,y,z,w]; compare quaternions "
               "up to sign (q and -q are the same rotation). Matrices are row-major. See CONTRACTS 2.0.",
        "tolerance": {"position_m": 1e-9, "quaternion": 1e-9, "matrix": 1e-9},
        "poses": poses,
        "t_final_cases": t_cases,
        "intrinsics": intr,
    }
    doc = clean(doc)
    out = Path(__file__).resolve().parents[1] / "data" / "conventions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(doc) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
