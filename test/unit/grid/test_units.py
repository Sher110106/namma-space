"""Unit tests for the small modules: PNG, PLY, COLMAP, frame conventions, k-means, extent, raster helpers."""
from __future__ import annotations

import json
import shutil
import unittest

import numpy as np

import common  # noqa: F401  (sets sys.path)
from common import tmpdir
from pipeline.grid import colmapio, extent, frame, pngio, plyio, raster, viewpoints


class Png(unittest.TestCase):
    def test_round_trip_grey_and_rgb(self):
        d = tmpdir()
        try:
            rng = np.random.default_rng(0)
            g = rng.integers(0, 256, (37, 53), dtype=np.uint8)
            c = rng.integers(0, 256, (21, 30, 3), dtype=np.uint8)
            pngio.write_png(d / "g.png", g)
            pngio.write_png(d / "c.png", c)
            self.assertTrue(np.array_equal(pngio.read_png(d / "g.png"), g))
            self.assertTrue(np.array_equal(pngio.read_png(d / "c.png"), c))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Ply(unittest.TestCase):
    def _data(self):
        rng = np.random.default_rng(1)
        n = 500
        return (rng.normal(size=(n, 3)).astype(np.float32), rng.uniform(0.05, 0.99, n).astype(np.float32),
                rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32), rng.uniform(0, 1, (n, 3)).astype(np.float32))

    def test_raw3dgs_round_trip_and_autodetect(self):
        d = tmpdir()
        try:
            xyz, op, sc, rgb = self._data()
            plyio.write_ply(d / "a.ply", xyz, op, sc, rgb, sh_rest=9, raw3dgs=True)
            s = plyio.read_splats(d / "a.ply")
            self.assertEqual(s.activations, "raw3dgs")
            np.testing.assert_allclose(s.xyz, xyz)
            np.testing.assert_allclose(s.opacity, op, atol=1e-5)
            np.testing.assert_allclose(s.scale, sc, rtol=1e-4)
            np.testing.assert_allclose(s.rgb, rgb, atol=1e-5)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_linear_values_are_kept(self):
        d = tmpdir()
        try:
            xyz, op, sc, rgb = self._data()
            plyio.write_ply(d / "b.ply", xyz, op, sc, rgb, raw3dgs=False)
            s = plyio.read_splats(d / "b.ply")
            self.assertEqual(s.activations, "linear")
            np.testing.assert_allclose(s.opacity, op, atol=1e-6)
            np.testing.assert_allclose(s.scale, sc, rtol=1e-5)
            s2 = plyio.read_splats(d / "b.ply", activations="linear")
            np.testing.assert_allclose(s2.opacity, op, atol=1e-6)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Colmap(unittest.TestCase):
    def test_qvec_round_trip(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            q = rng.normal(size=4)
            R = colmapio.qvec_to_rot(q)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            R2 = colmapio.qvec_to_rot(colmapio.rot_to_qvec(R))
            np.testing.assert_allclose(R, R2, atol=1e-10)

    def test_binary_round_trip(self):
        d = tmpdir()
        try:
            rng = np.random.default_rng(3)
            n, m = 6, 40
            Rs = np.array([colmapio.qvec_to_rot(rng.normal(size=4)) for _ in range(n)])
            ts = rng.normal(size=(n, 3))
            ids = [np.sort(rng.choice(np.arange(1, m + 1), 10, replace=False)).astype(np.int64) for _ in range(n)]
            raw = [(k + 1, 1, rng.uniform(0, 100, (10, 2)), ids[k]) for k in range(n)]
            model = colmapio.Model([f"{k:06d}.jpg" for k in range(n)], Rs, ts, ids, np.arange(1, m + 1, dtype=np.int64),
                                   rng.normal(size=(m, 3)), raw, {})
            colmapio.write_model(d, model)
            back = colmapio.read_model(d)
            self.assertEqual(back.names, model.names)
            np.testing.assert_allclose(back.R, Rs, atol=1e-10)
            np.testing.assert_allclose(back.t, ts)
            np.testing.assert_allclose(back.pt_xyz, model.pt_xyz)
            for a, b in zip(back.point_ids, ids):
                self.assertTrue(np.array_equal(a, b))
            # centre and forward conventions: C = -R^T t, forward = third row of R
            np.testing.assert_allclose(back.centres, -np.einsum("nji,nj->ni", Rs, ts))
            np.testing.assert_allclose(back.forward, Rs[:, 2, :])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_text_reader(self):
        d = tmpdir()
        try:
            (d / "images.txt").write_text(
                "# comment\n1 1 0 0 0 1 2 3 1 000001.jpg\n10.0 20.0 5 30.0 40.0 -1\n"
                "2 0.7071067811865476 0.7071067811865476 0 0 0 0 0 1 000002.jpg\n\n")
            (d / "points3D.txt").write_text("# c\n5 1.0 2.0 3.0 10 20 30 0.5 1 0 2 1\n")
            m = colmapio.read_model(d)
            self.assertEqual(m.names, ["000001.jpg", "000002.jpg"])
            np.testing.assert_allclose(m.t[0], [1, 2, 3])
            self.assertEqual(list(m.point_ids[0]), [5])
            self.assertEqual(len(m.point_ids[1]), 0)
            np.testing.assert_allclose(m.pt_xyz, [[1, 2, 3]])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Frame(unittest.TestCase):
    def test_yaw_convention(self):
        # 0 looks along -Z, positive turns left (towards -X)
        np.testing.assert_allclose(frame.forward_from_yaw(0), [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(frame.forward_from_yaw(90), [-1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(frame.forward_from_yaw(180), [0, 0, 1], atol=1e-12)
        for y in (-170, -90, -10, 0, 33, 120, 179):
            self.assertAlmostEqual(frame.yaw_from_forward(frame.forward_from_yaw(y)), y, places=6)

    def test_pitch(self):
        self.assertAlmostEqual(float(frame.pitch_deg_from_forward(np.array([0, 1, 0.0]))), 90.0)
        self.assertAlmostEqual(float(frame.pitch_deg_from_forward(np.array([1, -1, 0.0]))), -45.0)

    def test_tfinal_forms(self):
        m = np.eye(4)
        m[:3, 3] = [1, 2, 3]
        for form in (m.tolist(), m.flatten().tolist(), {"matrix": m.tolist()},
                     {"position": [1, 2, 3], "quaternion": [0, 0, 0, 1], "scale": 1.0}):
            np.testing.assert_allclose(frame.parse_tfinal(form), m)
        T = frame.parse_tfinal({"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 2.0})
        self.assertAlmostEqual(frame.tfinal_scale(T), 2.0)
        np.testing.assert_allclose(frame.apply_points(T, np.array([[1.0, 1, 1]])), [[2, 2, 2]])
        np.testing.assert_allclose(frame.apply_dirs(T, np.array([[0, 0, -1.0]])), [[0, 0, -1]])
        self.assertTrue(np.allclose(frame.parse_tfinal(None), np.eye(4)))
        # the form export.json carries (W3): scale, quaternion, translation and a flat row-major matrix
        w3 = {"scale": 1.0, "quaternion": [0, 0, 0, 1], "translation": [1, 2, 3], "matrix": m.flatten().tolist(), "info": {}}
        np.testing.assert_allclose(frame.parse_tfinal(w3), m)
        w3.pop("matrix")
        np.testing.assert_allclose(frame.parse_tfinal(w3), m)

    def test_camera_conversion_matches_synth(self):
        """w2c -> centre and forward in a y-up world: look-at along -Z has forward (0, 0, -1)."""
        import synth
        R, t = synth.w2c_from_forward(np.array([1.0, 1.4, 2.0]), np.array([0, 0, -1.0]))
        np.testing.assert_allclose(-R.T @ t, [1.0, 1.4, 2.0])
        np.testing.assert_allclose(R[2], [0, 0, -1])
        np.testing.assert_allclose(R[1], [0, -1, 0])        # camera y points down in a y-up world


class ConventionsJson(unittest.TestCase):
    """Camera centres, forward vectors and T_final against root's test/data/conventions.json (CONTRACTS 2.0)."""

    @classmethod
    def setUpClass(cls):
        path = common.ROOT / "test" / "data" / "conventions.json"
        cls.data = json.loads(path.read_text()) if path.exists() else None

    def _model(self, pose):
        R = np.array(pose["colmap"]["R_w2c"], dtype=float)[None]
        t = np.array(pose["colmap"]["t_w2c"], dtype=float)[None]
        return colmapio.Model(["a.jpg"], R, t, [np.zeros(0, np.int64)], np.zeros(0, np.int64), np.zeros((0, 3)), [], {})

    def test_poses(self):
        if not self.data:
            self.skipTest("test/data/conventions.json not present")
        for pose in self.data["poses"]:
            m = self._model(pose)
            exp = pose["threejs"]
            np.testing.assert_allclose(m.centres[0], exp["position"], atol=1e-9, err_msg=pose["id"])
            np.testing.assert_allclose(m.forward[0], exp["forward_world"], atol=1e-9, err_msg=pose["id"])

    def test_t_final_cases(self):
        if not self.data:
            self.skipTest("test/data/conventions.json not present")
        poses = {p["id"]: p for p in self.data["poses"]}
        for case in self.data["t_final_cases"]:
            m = self._model(poses[case["pose"]])
            T = frame.parse_tfinal(case["T_final"])
            pos = frame.apply_points(T, m.centres)[0]
            fwd = frame.apply_dirs(T, m.forward)[0]
            exp = case["expected"]
            np.testing.assert_allclose(pos, exp["position"], atol=1e-9, err_msg=case["id"])
            exp_fwd = frame.quat_xyzw_to_rot(exp["quaternion_xyzw"]) @ np.array([0.0, 0.0, -1.0])
            np.testing.assert_allclose(fwd, exp_fwd, atol=1e-9, err_msg=case["id"])
            # yaw and pitch of the forward vector agree with the quaternion's
            self.assertAlmostEqual(frame.yaw_from_forward(fwd), frame.yaw_from_forward(exp_fwd), places=6)


class Kmeans(unittest.TestCase):
    def test_finds_blobs_and_is_deterministic(self):
        rng = np.random.default_rng(0)
        centres = np.array([[0, 0], [5, 0], [0, 5], [5, 5], [10, 10.0]])
        x = np.concatenate([c + rng.normal(0, 0.2, (40, 2)) for c in centres])
        c1, l1 = viewpoints.kmeans(x, 5, seed=0)
        c2, l2 = viewpoints.kmeans(x, 5, seed=0)
        np.testing.assert_allclose(c1, c2)
        d = np.linalg.norm(c1[:, None] - centres[None], axis=2).min(axis=0)
        self.assertLess(d.max(), 0.2)


class Extent(unittest.TestCase):
    @staticmethod
    def _rect(L, W, ang, rng, density=1.0, shift=(3.0, -2.0)):
        n = int(6000 * density)
        e = [np.c_[rng.uniform(0, L, n), np.zeros(n)], np.c_[rng.uniform(0, L, n), np.full(n, W)],
             np.c_[np.zeros(n // 2), rng.uniform(0, W, n // 2)], np.c_[np.full(n // 2, L), rng.uniform(0, W, n // 2)]]
        p = np.concatenate(e) + rng.normal(0, 0.004, (n * 3, 2))
        a = np.radians(ang)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return p @ R.T + shift, R

    @staticmethod
    def _cells(p, cell=0.05):
        ij = np.unique(np.floor(p / cell).astype(int), axis=0)
        return (ij + 0.5) * cell

    def test_rotated_rectangle(self):
        rng = np.random.default_rng(0)
        for ang in (0.0, 17.0, -30.0):
            L, W = 6.2, 3.1
            pts, R = self._rect(L, W, ang, rng)
            cams = (np.array([[1.0, 1.0], [L - 1.0, W - 1.0], [3.0, 1.5]]) @ R.T) + (3.0, -2.0)
            r = extent.room_extent(self._cells(pts), pts, cams)
            self.assertAlmostEqual(r["room_extent_m"][0], L, delta=0.05, msg=f"angle {ang}")
            self.assertAlmostEqual(r["room_extent_m"][1], W, delta=0.05, msg=f"angle {ang}")

    def test_inner_object_and_short_closet_do_not_count(self):
        rng = np.random.default_rng(1)
        pts, R = self._rect(5.0, 4.0, 0.0, rng, shift=(0.0, 0.0))
        table = np.c_[np.full(3000, 2.0), rng.uniform(1.0, 3.0, 3000)]                  # a 2 m edge inside the room
        closet = np.c_[rng.uniform(1.8, 3.2, 1500), np.full(1500, 4.0 + 1.2)]           # 1.4 m wall beyond the far wall
        allp = np.concatenate([pts, table, closet])
        cams = np.array([[0.5, 0.5], [4.5, 3.5], [2.5, 4.4]])
        r = extent.room_extent(self._cells(allp), allp, cams)
        self.assertAlmostEqual(r["room_extent_m"][0], 5.0, delta=0.05)
        self.assertAlmostEqual(r["room_extent_m"][1], 4.0, delta=0.05)

    def test_sparse_wall_still_counts(self):
        """One wall with a third of the splat density of the others is still a wall (length counts, not density)."""
        rng = np.random.default_rng(2)
        pts, R = self._rect(5.0, 4.0, 0.0, rng, shift=(0.0, 0.0))
        right = (np.abs(pts[:, 0] - 5.0) < 0.05)
        keep = ~right | (rng.random(len(pts)) < 0.33)
        pts = pts[keep]
        cams = np.array([[0.5, 0.5], [4.5, 3.5], [2.5, 2.0]])
        r = extent.room_extent(self._cells(pts), pts, cams)
        self.assertAlmostEqual(r["room_extent_m"][0], 5.0, delta=0.05)

    def test_too_few_points(self):
        self.assertIsNone(extent.room_extent(np.zeros((10, 2)), np.zeros((10, 2)), np.zeros((3, 2))))


class Raster(unittest.TestCase):
    def test_close_fills_small_holes_only(self):
        m = np.ones((60, 60), dtype=bool)
        m[20:24, 20:24] = False        # 20 cm hole at 5 cm cells
        m[35:50, 35:50] = False        # 75 cm hole
        r = raster.close(m, 3)         # radius 3 cells = 15 cm
        self.assertTrue(r[20:24, 20:24].all())
        self.assertFalse(r[38:47, 38:47].any())

    def test_dilate_radius(self):
        m = np.zeros((21, 21), dtype=bool)
        m[10, 10] = True
        d = raster.dilate(m, 3)
        self.assertTrue(d[10, 13] and d[13, 10] and not d[10, 14] and d[12, 12] and not d[13, 13])


if __name__ == "__main__":
    unittest.main()
