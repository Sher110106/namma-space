import math
import os
import tempfile
import unittest

import _paths  # noqa: F401
import helpers
from pipeline.sfm import eval_poses, similarity


def rot_z(a):
    return [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]


def rot_x(a):
    return [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


PTS = [(0.0, 0.0, 0.0), (1.0, 0.2, -0.3), (2.0, 1.0, 0.5), (0.5, 3.0, 1.5), (-1.0, 2.0, 2.5), (3.0, -1.0, 0.1), (1.2, 1.1, -2.0)]


class TestSimilarity(unittest.TestCase):
    def test_recovers_known_similarity(self):
        r = matmul(rot_z(0.7), rot_x(-1.1))
        s, t = 2.5, (4.0, -2.0, 7.0)
        dst = [similarity.apply(s, r, t, p) for p in PTS]
        s2, r2, t2 = similarity.fit(PTS, dst)
        self.assertAlmostEqual(s2, s, places=9)
        self.assertLess(similarity.rmse(s2, r2, t2, PTS, dst), 1e-9)
        for i in range(3):
            self.assertAlmostEqual(t2[i], t[i], places=8)

    def test_noise_gives_small_rmse(self):
        r = rot_z(1.9)
        dst = [similarity.apply(0.5, r, (1, 2, 3), p) for p in PTS]
        dst[3] = (dst[3][0] + 0.01, dst[3][1], dst[3][2])
        s, r2, t = similarity.fit(PTS, dst)
        self.assertLess(similarity.rmse(s, r2, t, PTS, dst), 0.01)

    def test_eval_poses_on_written_models(self):
        r = rot_z(0.4)
        names = [f"{i:03d}.jpg" for i in range(len(PTS))]
        ref = list(zip(names, PTS))
        ours = [(n, similarity.apply(3.0, r, (1, 1, 1), p)) for n, p in ref]
        with tempfile.TemporaryDirectory() as d:
            helpers.write_model(os.path.join(d, "ref"), ref)
            helpers.write_model(os.path.join(d, "ours"), ours)
            res = eval_poses.evaluate(os.path.join(d, "ours"), os.path.join(d, "ref"))
        self.assertLess(res["rmse_pct_of_extent"], 1e-6)
        self.assertAlmostEqual(res["scale"], 1 / 3.0, places=6)
        self.assertEqual(res["n_common"], len(PTS))


if __name__ == "__main__":
    unittest.main()
