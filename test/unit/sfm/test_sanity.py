import unittest

import _paths  # noqa: F401
from pipeline.sfm.sanity import Cam, filter_trajectory


def path(n, step=0.1):
    return [Cam(i, f"{i + 1:06d}.jpg", (i * step, 0.0, 0.0)) for i in range(n)]


class TestSanity(unittest.TestCase):
    def test_clean_path_untouched(self):
        r = filter_trajectory(path(100))
        self.assertEqual(r.dropped, [])
        self.assertEqual(len(r.kept), 100)
        self.assertEqual(r.jumps_left, [])

    def test_honka_like_far_segment_dropped(self):
        # frames 134-140 registered about 1000 units away (the smoke-test failure)
        cams = path(290)
        for c in cams[133:140]:
            c.centre = (1000.0 + c.idx, 3.0, 0.0)
        r = filter_trajectory(cams)
        self.assertEqual({d["idx"] for d in r.dropped}, set(range(133, 140)))
        self.assertTrue(all(d["reason"] == "far_from_trajectory" for d in r.dropped))
        self.assertEqual(len(r.kept), 283)

    def test_short_run_between_jumps_dropped(self):
        # 4 cameras displaced 3 m sideways: not far (path is 20 m) but two jumps of 30x the median step
        cams = path(200)
        for c in cams[50:54]:
            c.centre = (c.centre[0], 3.0, 0.0)
        r = filter_trajectory(cams)
        self.assertEqual({d["idx"] for d in r.dropped}, {50, 51, 52, 53})
        self.assertTrue(all(d["reason"] == "short_run_between_jumps" for d in r.dropped))

    def test_gap_of_unregistered_frames_is_not_a_jump(self):
        # frames 60-79 unregistered; the camera kept moving at the same speed, so the step per frame is normal
        cams = [c for c in path(200) if not 60 <= c.idx < 80]
        r = filter_trajectory(cams)
        self.assertEqual(r.dropped, [])
        self.assertEqual(r.jumps_left, [])

    def test_two_long_runs_with_a_jump_are_kept_and_reported(self):
        a = path(100)
        b = [Cam(100 + i, f"{101 + i:06d}.jpg", (50.0 + i * 0.1, 0.0, 0.0)) for i in range(100)]
        r = filter_trajectory(a + b)
        self.assertEqual(r.dropped, [])
        self.assertEqual(len(r.jumps_left), 1)
        self.assertGreater(r.jumps_left[0]["ratio"], 5)

    def test_too_few_cameras_pass_through(self):
        r = filter_trajectory(path(3))
        self.assertEqual(len(r.kept), 3)
        self.assertEqual(r.dropped, [])

    def test_loop_trajectory_is_not_far(self):
        import math
        cams = [Cam(i, f"{i}.jpg", (3 * math.cos(i / 40), 0.0, 3 * math.sin(i / 40))) for i in range(251)]
        r = filter_trajectory(cams)
        self.assertEqual(r.dropped, [])


if __name__ == "__main__":
    unittest.main()
