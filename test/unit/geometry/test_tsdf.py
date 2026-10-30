import unittest

import numpy as np

from pipeline.geometry.tsdf import depth_view, query_free_space


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


if __name__ == "__main__":
    unittest.main()
