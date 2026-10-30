import os
import struct
import tempfile
import unittest

from _common import HAVE_NUMPY

if HAVE_NUMPY:
    import numpy as np
    from pipeline.export import colmap_io as cio
    from pipeline.export import heldout as HO


def rot_y(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


@unittest.skipUnless(HAVE_NUMPY, "numpy missing")
class ThreeJsConvention(unittest.TestCase):
    def test_identity_camera(self):
        """OpenCV camera at the origin looking along +z: three.js camera must look along +z too,
        i.e. be rotated 180 degrees about y (three.js looks down -z)."""
        c2w = np.eye(4)
        p = HO.threejs_pose(c2w, HO.identity_t_final())
        self.assertEqual(p["position"], [0.0, 0.0, 0.0])
        r = HO.matrix_from_quat(p["quaternion"])
        np.testing.assert_allclose(r @ np.array([0, 0, -1.0]), [0, 0, 1.0], atol=1e-12)   # forward
        np.testing.assert_allclose(r @ np.array([0, 1.0, 0]), [0, -1.0, 0], atol=1e-12)   # opencv y is down

    def test_matches_contract_formula(self):
        """centre = -R^T t ; rotation = R^T diag(1,-1,-1)."""
        R = rot_y(30) @ np.array([[1, 0, 0], [0, np.cos(0.2), -np.sin(0.2)], [0, np.sin(0.2), np.cos(0.2)]])
        t = np.array([0.3, -0.2, 1.5])
        c2w = np.linalg.inv(np.block([[R, t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        p = HO.threejs_pose(c2w, HO.identity_t_final())
        np.testing.assert_allclose(p["position"], -R.T @ t, atol=1e-12)
        np.testing.assert_allclose(HO.matrix_from_quat(p["quaternion"]), R.T @ np.diag([1, -1, -1]), atol=1e-12)

    def test_t_final_applies_to_position_and_rotation(self):
        c2w = np.eye(4)
        c2w[:3, 3] = [1, 2, 3]
        tf = {"scale": 2.0, "quaternion": HO.quat_from_matrix(rot_y(90)), "translation": [0, -1, 0]}
        p = HO.threejs_pose(c2w, tf)
        np.testing.assert_allclose(p["position"], 2.0 * rot_y(90) @ [1, 2, 3] + [0, -1, 0], atol=1e-9)
        np.testing.assert_allclose(HO.matrix_from_quat(p["quaternion"]), rot_y(90) @ np.diag([1, -1, -1]), atol=1e-9)

    def test_quaternion_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            np.testing.assert_allclose(HO.matrix_from_quat(HO.quat_from_matrix(q)), q, atol=1e-9)

    def test_fov(self):
        self.assertAlmostEqual(HO.fov_y_deg(1080, 1080 / (2 * np.tan(np.deg2rad(35)))), 70.0, places=6)

    def test_pick_evenly(self):
        self.assertEqual(HO.pick_evenly(list(range(36)), 8), [0, 5, 10, 15, 20, 25, 30, 35])
        self.assertEqual(HO.pick_evenly([1, 2], 8), [1, 2])


def write_tiny_model(d, imgs, pts):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, 640, 480) + struct.pack("<4d", 500.0, 510.0, 320.0, 240.0))
    with open(os.path.join(d, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for i, (name, q, t) in enumerate(imgs):
            f.write(struct.pack("<idddddddi", i + 1, *q, *t, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<ddq", 1.0, 2.0, -1) * 2)
    with open(os.path.join(d, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, p in enumerate(pts):
            f.write(struct.pack("<QdddBBBdQ", i + 1, *p, 10, 20, 30, 0.5, 2))
            f.write(struct.pack("<ii", 1, 0) * 2)


@unittest.skipUnless(HAVE_NUMPY, "numpy missing")
class ColmapBinary(unittest.TestCase):
    def test_read_tiny_model(self):
        with tempfile.TemporaryDirectory() as d:
            m = os.path.join(d, "sparse", "0")
            write_tiny_model(m, [("b.jpg", (1, 0, 0, 0), (0, 0, 1)), ("a.jpg", (1, 0, 0, 0), (1, 2, 3))],
                             [(0.0, 1.0, 2.0), (3.0, 4.0, 5.0)])
            cams = cio.read_cameras(d)
            self.assertEqual(cams[1].model, "PINHOLE")
            self.assertEqual(cams[1].intrinsics(), (500.0, 510.0, 320.0, 240.0))
            ims = cio.sorted_images(d)
            self.assertEqual([i.name for i in ims], ["a.jpg", "b.jpg"])
            np.testing.assert_allclose(ims[0].centre(), [-1, -2, -3])
            xyz, rgb, tl = cio.read_points3d(d)
            np.testing.assert_allclose(xyz, [[0, 1, 2], [3, 4, 5]])
            self.assertEqual(rgb.tolist(), [[10, 20, 30]] * 2)
            self.assertEqual(tl.tolist(), [2, 2])
            self.assertEqual([i.name for i in cio.heldout_images(d, 2)], ["a.jpg"])

    def test_text_model_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "sparse", "0"))
            open(os.path.join(d, "sparse", "0", "cameras.txt"), "w").write("# x")
            with self.assertRaises(ValueError):
                cio.read_cameras(d)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_NUMPY, "numpy missing")
class TransformModel(unittest.TestCase):
    def test_similarity_keeps_projections(self):
        """Project a point through the original and the transformed model: same pixel."""
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "a", "sparse", "0"), os.path.join(d, "b", "sparse", "0")
            q = (0.9238795325112867, 0.0, 0.3826834323650898, 0.0)     # 45 deg about y
            write_tiny_model(src, [("a.jpg", q, (0.5, -0.2, 2.0)), ("b.jpg", (1, 0, 0, 0), (0.0, 0.1, 1.0))],
                             [(0.1, 0.2, 3.0), (-1.0, 0.5, 4.0)])
            s, R, t = 2.5, rot_y(37.0), np.array([0.3, 1.4, -0.7])
            cio.write_transformed_model(src, dst, s, R, t)
            ia, ib = cio.sorted_images(os.path.join(d, "a")), cio.sorted_images(os.path.join(d, "b"))
            pa = cio.read_points3d(os.path.join(d, "a"))[0]
            pb = cio.read_points3d(os.path.join(d, "b"))[0]
            np.testing.assert_allclose(pb, (s * (R @ pa.T)).T + t, atol=1e-9)
            for x, y in zip(ia, ib):
                for X, Xn in zip(pa, pb):
                    cam_a = x.R() @ X + np.array(x.tvec)
                    cam_b = y.R() @ Xn + np.array(y.tvec)
                    np.testing.assert_allclose(cam_b, s * cam_a, atol=1e-9)       # same ray, depth scaled by s
                np.testing.assert_allclose(y.centre(), s * R @ x.centre() + t, atol=1e-9)
            self.assertEqual([i.name for i in ib], ["a.jpg", "b.jpg"])


@unittest.skipUnless(HAVE_NUMPY, "numpy missing")
class RenderCameraConversion(unittest.TestCase):
    def test_threejs_pose_round_trip(self):
        """heldout.threejs_pose and render_ply.cam_from_threejs are inverse (with and without T_final)."""
        from pipeline.export import render_ply as RP
        R = rot_y(33) @ np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, [0.4, 1.2, -2.0]
        for tf in (HO.identity_t_final(),
                   {"scale": 1.7, "quaternion": HO.quat_from_matrix(rot_y(25)), "translation": [0.2, -1.1, 0.3]}):
            p = HO.threejs_pose(c2w, tf)
            cam = RP.cam_from_threejs("x", p["position"], p["quaternion"], 62.0, 640, 480, (0.52, 0.49), tf)
            np.testing.assert_allclose(cam.c2w, c2w, atol=1e-9)
            self.assertAlmostEqual(HO.fov_y_deg(480, cam.K[1, 1]), 62.0, places=6)
            self.assertAlmostEqual(cam.K[0, 2], 0.52 * 640)
            self.assertAlmostEqual(cam.K[1, 2], 0.49 * 480)

    def test_cameras_from_json(self):
        import json
        from pipeline.export import render_ply as RP
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "poses.json")
            json.dump([{"name": "a", "c2w": np.eye(4).reshape(-1).tolist(), "fx": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
                       {"id": "b", "position": [0, 1, 0], "quaternion": [0, 0, 0, 1], "fov_y_deg": 60, "width": 320, "height": 240}], open(p, "w"))
            cams = RP.cameras_from_json(p)
        self.assertEqual([c.name for c in cams], ["a", "b"])
        self.assertEqual(cams[0].K[1, 1], 500)
        np.testing.assert_allclose(cams[1].c2w[:3, 3], [0, 1, 0])
