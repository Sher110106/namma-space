"""Tests 0.x conventions: COLMAP <-> three.js poses against test/data/conventions.json (stdlib only).

Run: python3 test/e2e/test_conventions.py
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.common import conventions as cv  # noqa: E402

DATA = json.loads((ROOT / "test" / "data" / "conventions.json").read_text())
TOL = DATA["tolerance"]


def close(a, b, tol):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def quat_close(a, b, tol):
    """Same rotation up to sign."""
    return close(a, b, tol) or close(a, [-v for v in b], tol)


class ConventionsFile(unittest.TestCase):
    def test_file_has_three_poses_and_every_field(self):
        self.assertGreaterEqual(len(DATA["poses"]), 3)
        for p in DATA["poses"]:
            self.assertEqual(len(p["colmap"]["R_w2c"]), 3)
            self.assertEqual(len(p["threejs"]["quaternion_xyzw"]), 4)
            q = p["threejs"]["quaternion_xyzw"]
            self.assertAlmostEqual(sum(v * v for v in q), 1.0, places=9)

    def test_closed_forms(self):
        """Hand-checkable cases stated in the file's notes."""
        by = {p["id"]: p for p in DATA["poses"]}
        self.assertEqual(by["identity"]["threejs"]["quaternion_xyzw"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(by["level_minus_z"]["threejs"]["quaternion_xyzw"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(by["level_minus_z"]["colmap"]["t_w2c"], [-1.0, 1.4, 2.0])
        r = math.sqrt(0.5)
        self.assertTrue(close(by["yaw_left_90"]["threejs"]["quaternion_xyzw"], [0, r, 0, r], 1e-11))
        self.assertTrue(close(by["pitch_up_30"]["threejs"]["quaternion_xyzw"],
                              [math.sin(math.radians(15)), 0, 0, math.cos(math.radians(15))], 1e-11))


class ForwardMapping(unittest.TestCase):
    def test_colmap_to_threejs(self):
        for p in DATA["poses"]:
            pos, q = cv.threejs_from_colmap(p["colmap"]["R_w2c"], p["colmap"]["t_w2c"])
            exp = p["threejs"]
            self.assertTrue(close(pos, exp["position"], TOL["position_m"]), (p["id"], pos))
            self.assertTrue(quat_close(q, exp["quaternion_xyzw"], TOL["quaternion"]), (p["id"], q))
            m = cv.matrix_from_quat(q)
            for i in range(3):
                self.assertTrue(close(m[i], exp["rotation_matrix"][i], TOL["matrix"]), (p["id"], m))
            self.assertTrue(close(cv.forward_world(q), exp["forward_world"], 1e-9))

    def test_threejs_to_colmap(self):
        for p in DATA["poses"]:
            exp = p["threejs"]
            R, t = cv.colmap_from_threejs(exp["position"], exp["quaternion_xyzw"])
            for i in range(3):
                self.assertTrue(close(R[i], p["colmap"]["R_w2c"][i], TOL["matrix"]), (p["id"], R))
            self.assertTrue(close(t, p["colmap"]["t_w2c"], TOL["position_m"]), (p["id"], t))

    def test_t_final_cases_both_ways(self):
        by = {p["id"]: p for p in DATA["poses"]}
        for c in DATA["t_final_cases"]:
            p = by[c["pose"]]
            pos, q = cv.threejs_from_colmap(p["colmap"]["R_w2c"], p["colmap"]["t_w2c"], c["T_final"])
            self.assertTrue(close(pos, c["expected"]["position"], TOL["position_m"]), (c["id"], pos))
            self.assertTrue(quat_close(q, c["expected"]["quaternion_xyzw"], TOL["quaternion"]), (c["id"], q))
            R, t = cv.colmap_from_threejs(c["expected"]["position"], c["expected"]["quaternion_xyzw"], c["T_final"])
            for i in range(3):
                self.assertTrue(close(R[i], p["colmap"]["R_w2c"][i], 1e-8))
            self.assertTrue(close(t, p["colmap"]["t_w2c"], 1e-8))

    def test_t_final_forms_agree(self):
        """Every form other packages write gives the same result as the 4x4."""
        by = {p["id"]: p for p in DATA["poses"]}
        c = next(x for x in DATA["t_final_cases"] if x["id"] == "scale_half_yaw90")
        T = c["T_final"]
        p = by[c["pose"]]
        flat = [v for row in T for v in row]
        s = 0.5
        q_yaw90 = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
        forms = [T, flat, {"matrix": T}, {"matrix": flat},
                 {"scale": s, "translation": [T[0][3], T[1][3], T[2][3]], "quaternion": q_yaw90, "info": "extra keys are fine"},
                 {"scale": s, "position": [T[0][3], T[1][3], T[2][3]], "quaternion": q_yaw90}]
        for f in forms:
            pos, q = cv.threejs_from_colmap(p["colmap"]["R_w2c"], p["colmap"]["t_w2c"], f)
            self.assertTrue(close(pos, c["expected"]["position"], 1e-9), f)
            self.assertTrue(quat_close(q, c["expected"]["quaternion_xyzw"], 1e-9), f)
        self.assertIsNone(cv.parse_t_final(None))
        with self.assertRaises(ValueError):
            cv.parse_t_final("nope")

    def test_yaw_pitch(self):
        for p in DATA["poses"]:
            e = p["threejs"]
            if "yaw_deg" in e:
                yaw, pitch = cv.yaw_pitch_deg(e["quaternion_xyzw"])
                self.assertAlmostEqual(yaw, e["yaw_deg"], places=7)
                self.assertAlmostEqual(pitch, e["pitch_deg"], places=7)
                q = cv.quat_from_yaw_pitch_deg(e["yaw_deg"], e["pitch_deg"])
                self.assertTrue(quat_close(q, e["quaternion_xyzw"], 1e-9))

    def test_random_roundtrip(self):
        import random
        rnd = random.Random(7)
        for _ in range(200):
            q = cv.quat_from_yaw_pitch_deg(rnd.uniform(-180, 180), rnd.uniform(-80, 80))
            pos = [rnd.uniform(-5, 5) for _ in range(3)]
            R, t = cv.colmap_from_threejs(pos, q)
            pos2, q2 = cv.threejs_from_colmap(R, t)
            self.assertTrue(close(pos, pos2, 1e-9))
            self.assertTrue(quat_close(q, q2, 1e-9))


class ProjectionAgrees(unittest.TestCase):
    """A world point must land on the same pixel through the COLMAP route and the three.js route."""

    def test_pixels_match(self):
        import random
        rnd = random.Random(11)
        for intr in DATA["intrinsics"]:
            w, h, fx, fy, cx, cy = (intr[k] for k in ("width", "height", "fx", "fy", "cx", "cy"))
            self.assertAlmostEqual(cv.fov_y_deg(h, fy), intr["fov_y_deg"], places=9)
            P = cv.projection_matrix(fx, fy, cx, cy, w, h, intr["near"], intr["far"])
            for i in range(4):
                self.assertTrue(close(P[i], intr["projection_matrix_rowmajor"][i], 1e-9))
            for p in DATA["poses"]:
                R, t = p["colmap"]["R_w2c"], p["colmap"]["t_w2c"]
                pos, q = p["threejs"]["position"], p["threejs"]["quaternion_xyzw"]
                Rc = cv.matrix_from_quat(q)  # camera-to-world, three.js axes
                fwd = p["threejs"]["forward_world"]
                n = 0
                while n < 8:
                    # sample a point 1-6 m in front of the camera, inside the frustum
                    a, b, d = rnd.uniform(-0.4, 0.4), rnd.uniform(-0.3, 0.3), rnd.uniform(1, 6)
                    xc3 = [a * d, b * d, -d]  # three.js camera coordinates
                    X = [pos[i] + sum(Rc[i][k] * xc3[k] for k in range(3)) for i in range(3)]
                    # COLMAP route
                    xc = [sum(R[i][k] * X[k] for k in range(3)) + t[i] for i in range(3)]
                    self.assertGreater(xc[2], 0)
                    u, v = fx * xc[0] / xc[2] + cx, fy * xc[1] / xc[2] + cy
                    # three.js route: world -> camera (R^T), projection matrix, NDC -> pixels
                    d3 = [X[i] - pos[i] for i in range(3)]
                    c3 = [sum(Rc[k][i] * d3[k] for k in range(3)) for i in range(3)]
                    clip = [sum(P[i][k] * (c3 + [1.0])[k] for k in range(4)) for i in range(4)]
                    nx, ny = clip[0] / clip[3], clip[1] / clip[3]
                    u2, v2 = (nx + 1) * w / 2, (1 - ny) * h / 2
                    self.assertAlmostEqual(u, u2, places=6)
                    self.assertAlmostEqual(v, v2, places=6)
                    n += 1
                self.assertGreater(sum(f * f for f in fwd), 0.99)


if __name__ == "__main__":
    unittest.main()
