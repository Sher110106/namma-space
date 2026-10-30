"""Unit tests for pipeline/common/colmap_io.py (stdlib only).

Run: python3 test/e2e/test_colmap_io.py
The honka test reads the smoke model on tyrone and is skipped elsewhere
(override the path with NS_SMOKE_MODEL).
"""
from __future__ import annotations

import math
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.common import colmap_io as cio  # noqa: E402

SMOKE_CANDIDATES = [
    os.environ.get("NS_SMOKE_MODEL", ""),
    "/w/runs/smoke_honka/clean/sparse/0",
    "/home/sher/namma-space/runs/smoke_honka/clean/sparse/0",
]


def rot_axis(axis, angle):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


def synthetic_model(n_img=6, n_pts=40, seed=3) -> cio.Model:
    rnd = random.Random(seed)
    m = cio.Model()
    m.cameras[1] = cio.Camera(1, "PINHOLE", 640, 480, [500.0, 505.0, 320.0, 240.0])
    m.cameras[2] = cio.Camera(2, "SIMPLE_RADIAL", 640, 480, [500.0, 320.0, 240.0, -0.02])
    for i in range(1, n_img + 1):
        R = cio.mat_mul(rot_axis((0, 1, 0), rnd.uniform(-0.4, 0.4)), rot_axis((1, 0, 0), rnd.uniform(-0.2, 0.2)))
        t = (rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        m.images[i] = cio.Image(i, cio.rotmat_to_qvec(R), t, 1 if i % 2 else 2, f"img_{i:04d}.jpg")
    pts = {}
    obs: dict = {i: [] for i in m.images}
    for pid in range(1, n_pts + 1):
        X = (rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(4, 8))
        track = []
        for iid, im in m.images.items():
            R = im.rotmat()
            Xc = [sum(R[a][b] * X[b] for b in range(3)) + im.tvec[a] for a in range(3)]
            if Xc[2] <= 0.5:
                continue
            u, v = cio.project(m.cameras[im.camera_id], Xc)
            obs[iid].append((u, v, pid))
            track.append((iid, len(obs[iid]) - 1))
        p = cio.Point3D(pid, X, (10, 20, 30), 0.0)
        p.track = track
        pts[pid] = p
    for iid, im in m.images.items():
        im.points2D = obs[iid]
    m.points3D = pts
    # a trivial one-camera-per-frame rig, as COLMAP >= 3.12 writes it
    m.extra["rigs.bin"] = struct.pack("<QIIiI", 1, 1, 1, 0, 1)
    fr = [struct.pack("<Q", len(m.images))]
    for iid, im in m.images.items():
        fr.append(struct.pack("<II4d3dI", iid, 1, *im.qvec, *im.tvec, 1))
        fr.append(struct.pack("<iIQ", 0, im.camera_id, iid))
    m.extra["frames.bin"] = b"".join(fr)
    return m


def frame_poses(raw: bytes) -> dict:
    (n,) = struct.unpack_from("<Q", raw, 0)
    off, out = 8, {}
    for _ in range(n):
        fid, rid, qw, qx, qy, qz, tx, ty, tz, nd = struct.unpack_from("<II4d3dI", raw, off)
        off += 68
        data_ids = [struct.unpack_from("<iIQ", raw, off + 16 * k)[2] for k in range(nd)]
        off += 16 * nd
        for d in data_ids:
            out[d] = ((qw, qx, qy, qz), (tx, ty, tz))
    return out


def quat_close(a, b, tol=1e-9):
    return abs(sum(x * y for x, y in zip(a, b))) > 1 - tol


class QuaternionTests(unittest.TestCase):
    def test_roundtrip_many(self):
        rnd = random.Random(1)
        for _ in range(500):
            axis = (rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1))
            R = rot_axis(axis, rnd.uniform(0, math.pi))
            R2 = cio.qvec_to_rotmat(cio.rotmat_to_qvec(R))
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(R[i][j], R2[i][j], places=12)

    def test_half_turns(self):
        for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            R = rot_axis(axis, math.pi)
            q = cio.rotmat_to_qvec(R)
            self.assertAlmostEqual(sum(v * v for v in q), 1.0, places=12)
            R2 = cio.qvec_to_rotmat(q)
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(R[i][j], R2[i][j], places=12)


class SyntheticModelTests(unittest.TestCase):
    def test_write_read_roundtrip(self):
        m = synthetic_model()
        with tempfile.TemporaryDirectory() as d:
            cio.write_model(m, d)
            m2 = cio.read_model(d)
            cio.write_model(m2, Path(d) / "again")
            for name in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
                self.assertEqual((Path(d) / name).read_bytes(), (Path(d) / "again" / name).read_bytes(), name)
        self.assertEqual(len(m2.images), len(m.images))
        self.assertEqual(m2.images[3].name, "img_0003.jpg")
        self.assertEqual(m2.images[3].points2D, m.images[3].points2D)
        self.assertEqual(m2.points3D[5].track, m.points3D[5].track)
        self.assertEqual(m2.cameras[2].model, "SIMPLE_RADIAL")

    def test_text_model_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cameras.txt").write_text("# x\n")
            with self.assertRaises(ValueError):
                cio.read_model(d)

    def test_exact_reprojection_is_zero(self):
        rms, n = cio.reprojection_rmse(synthetic_model())
        self.assertLess(rms, 1e-9)
        self.assertGreater(n, 100)

    def test_similarity_keeps_reprojection_and_counts(self):
        m = synthetic_model()
        s = 1.7
        R = rot_axis((0.3, 1.0, -0.4), 0.9)
        t = [0.5, -2.0, 3.25]
        m2 = cio.similarity_transform(m, s, R, t)
        self.assertEqual(len(m2.images), len(m.images))
        self.assertEqual(len(m2.points3D), len(m.points3D))
        rms, n = cio.reprojection_rmse(m2)
        self.assertLess(rms, 1e-8)
        # points and camera centres follow X' = s R X + t
        for pid in (1, 7, 40):
            X = m.points3D[pid].xyz
            want = [s * sum(R[i][k] * X[k] for k in range(3)) + t[i] for i in range(3)]
            for a, b in zip(want, m2.points3D[pid].xyz):
                self.assertAlmostEqual(a, b, places=10)
        for iid in m.images:
            C = m.images[iid].centre()
            want = [s * sum(R[i][k] * C[k] for k in range(3)) + t[i] for i in range(3)]
            for a, b in zip(want, m2.images[iid].centre()):
                self.assertAlmostEqual(a, b, places=10)
        # the 2D data is untouched
        self.assertEqual(m2.images[2].points2D_raw, m.images[2].points2D_raw)
        # frames.bin followed the images
        fp = frame_poses(m2.extra["frames.bin"])
        for iid, im in m2.images.items():
            q, tv = fp[iid]
            self.assertTrue(quat_close(q, im.qvec))
            for a, b in zip(tv, im.tvec):
                self.assertAlmostEqual(a, b, places=10)
        # and the input model was not touched
        self.assertAlmostEqual(m.images[1].tvec[0], synthetic_model().images[1].tvec[0])

    def test_inverse_transform_restores_model(self):
        m = synthetic_model()
        s, R, t = 2.5, rot_axis((1, 2, 3), 0.7), [1.0, 2.0, 3.0]
        m2 = cio.similarity_transform(m, s, R, t)
        # inverse: X = (1/s) R^T (X' - t)  ->  s' = 1/s, R' = R^T, t' = -(1/s) R^T t
        Rt = cio.mat_T(R)
        t_inv = [-(1 / s) * sum(Rt[i][k] * t[k] for k in range(3)) for i in range(3)]
        m3 = cio.similarity_transform(m2, 1 / s, Rt, t_inv)
        for iid in m.images:
            for a, b in zip(m.images[iid].tvec, m3.images[iid].tvec):
                self.assertAlmostEqual(a, b, places=9)
            self.assertTrue(quat_close(m.images[iid].qvec, m3.images[iid].qvec))

    def test_matrix_split(self):
        s, R, t = 0.37, rot_axis((0, 1, 0), 1.1), [1, 2, 3]
        S = cio.matrix_from_similarity(s, R, t)
        s2, R2, t2 = cio.similarity_from_matrix(S)
        self.assertAlmostEqual(s, s2, places=12)
        self.assertEqual(t2, [1, 2, 3])
        with self.assertRaises(ValueError):
            cio.similarity_from_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    def test_rejects_bad_inputs(self):
        m = synthetic_model()
        with self.assertRaises(ValueError):
            cio.similarity_transform(m, -1.0, rot_axis((0, 0, 1), 0.1), [0, 0, 0])
        with self.assertRaises(ValueError):
            cio.similarity_transform(m, 1.0, [[1, 0, 0], [0, 1, 0], [0, 0, 2]], [0, 0, 0])

    def test_multi_sensor_rig_warns_not_crashes(self):
        m = synthetic_model()
        m.extra["rigs.bin"] = struct.pack("<QII", 1, 1, 2) + b"\0" * 40
        m2 = cio.similarity_transform(m, 2.0, rot_axis((0, 0, 1), 0.1), [0, 0, 0])
        self.assertTrue(any("rigs.bin" in w for w in m2.warnings))

    def test_summary(self):
        info = cio.model_summary(synthetic_model())
        self.assertEqual(info["registered_images"], 6)
        self.assertEqual(info["points3D"], 40)
        self.assertGreater(info["mean_track_length"], 3)


def smoke_dir():
    for c in SMOKE_CANDIDATES:
        if c and (Path(c) / "images.bin").exists():
            return Path(c)
    return None


@unittest.skipUnless(smoke_dir(), "smoke_honka model not present (run on tyrone)")
class HonkaSmokeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = smoke_dir()
        cls.model = cio.read_model(cls.dir)

    def test_counts(self):
        self.assertEqual(len(self.model.images), 283)
        self.assertEqual(len(self.model.cameras), 1)
        cam = next(iter(self.model.cameras.values()))
        self.assertEqual(cam.model, "PINHOLE")
        self.assertGreater(len(self.model.points3D), 10000)
        self.assertEqual(sorted(self.model.extra), ["frames.bin", "rigs.bin"])

    def test_writer_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            cio.write_model(self.model, d)
            for name in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
                self.assertEqual((self.dir / name).read_bytes(), (Path(d) / name).read_bytes(), name)

    def test_frames_match_images(self):
        fp = frame_poses(self.model.extra["frames.bin"])
        for iid, im in self.model.images.items():
            q, tv = fp[iid]
            self.assertTrue(quat_close(q, im.qvec, 1e-12))

    def test_similarity_preserves_reprojection(self):
        before, n0 = cio.reprojection_rmse(self.model, max_points=4000)
        s, R, t = 0.63, rot_axis((0.2, 0.9, 0.1), 2.0), [5.0, -1.0, 2.0]
        m2 = cio.similarity_transform(self.model, s, R, t)
        after, n1 = cio.reprojection_rmse(m2, max_points=4000)
        self.assertEqual(n0, n1)
        self.assertLess(abs(after - before), 1e-5 * max(1.0, before))
        self.assertEqual(len(m2.images), 283)
        with tempfile.TemporaryDirectory() as d:
            cio.write_model(m2, d)
            m3 = cio.read_model(d)
        self.assertEqual(len(m3.images), 283)
        self.assertEqual(len(m3.points3D), len(self.model.points3D))


if __name__ == "__main__":
    unittest.main()
