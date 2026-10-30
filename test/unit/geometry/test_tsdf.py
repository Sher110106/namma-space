import importlib.util
import unittest

import numpy as np

from eval.synthetic_room import box_depth
from pipeline.geometry.tsdf import depth_view, fuse_depth_views, query_free_space


class TsdfFreeSpaceTests(unittest.TestCase):
    def setUp(self):
        c2w = np.eye(4)
        K = np.array([[100.0, 0, 1.5], [0, 100.0, 1.5], [0, 0, 1.0]])
        depth = np.full((4, 4), 2.0, np.float32)
        self.view = depth_view(depth, c2w, K)

    def test_point_three_voxels_in_front_is_free(self):
        # With identity camera, points are (x, y, z) in camera coordinates.
        p = np.array([[0.0, 0.0, 1.9], [0.0, 0.0, 1.95], [0.0, 0.0, 2.1]])
        got = query_free_space(p, [self.view], margin=0.06)
        np.testing.assert_array_equal(got, [True, False, False])

    def test_invalid_alpha_does_not_mark_free(self):
        view = depth_view(self.view.depth, self.view.c2w, self.view.K,
                          alpha=np.zeros_like(self.view.depth))
        self.assertFalse(bool(query_free_space([[0, 0, 1]], [view])[0]))

    def test_min_views(self):
        other = depth_view(np.full((4, 4), 3.0, np.float32), self.view.c2w, self.view.K)
        other.alpha[:] = 0
        p = np.array([[0.0, 0.0, 1.9]])
        self.assertFalse(bool(query_free_space(p, [self.view, other], min_views=2)[0]))


class UniformTsdfSyntheticRoomTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("open3d"), "Open3D is available in nsdocker")
    def test_median_wall_error_is_within_two_cm(self):
        w = h = 64
        K = np.array([[60.0, 0, 32.0], [0, 60.0, 32.0], [0, 0, 1.0]])
        bounds = (-1.5, 1.5, -1.0, 1.0, 0.0, 2.4)
        views = []
        for i, (x, z, yaw) in enumerate([(0, .8, 0), (0, 1.6, np.pi),
                                          (-.8, 1.2, np.pi / 2),
                                          (.8, 1.2, -np.pi / 2)]):
            c2w = np.eye(4)
            c, s = np.cos(yaw), np.sin(yaw)
            c2w[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            c2w[:3, 3] = [x, 1.2, z]
            views.append(depth_view(box_depth(c2w, K, w, h, bounds), c2w, K, name=str(i)))
        mesh = fuse_depth_views(views)
        p = np.asarray(mesh.vertices)
        distances = np.minimum.reduce([
            np.abs(p[:, 0] - bounds[0]), np.abs(p[:, 0] - bounds[1]),
            np.abs(p[:, 1] - bounds[2]), np.abs(p[:, 1] - bounds[3]),
            np.abs(p[:, 2] - bounds[4]), np.abs(p[:, 2] - bounds[5])])
        self.assertGreater(len(p), 0)
        self.assertLessEqual(float(np.median(distances)), 0.02)


if __name__ == "__main__":
    unittest.main()
