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
        self.assertTrue(all(d["reason"] == "off_local_path" for d in r.dropped))

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


def loop_room(n_windows, gaps=(), wrong=(), offset=(0.0, 0.0, 3.0)):
    """A 4 x 5 m room walked round at 0.25 m per 0.5 s window; `gaps` are windows ingest dropped (no frame),
    `wrong` are windows whose camera the mapper put `offset` away. Frames are numbered in kept order."""
    per = 18.0
    cams = []
    for k in range(n_windows):
        if k in gaps:
            continue
        s = (k * 0.25) % per
        if s < 4:
            x, z = s, 0.0
        elif s < 9:
            x, z = 4.0, s - 4
        elif s < 13:
            x, z = 4 - (s - 9), 5.0
        else:
            x, z = 0.0, 5 - (s - 13)
        c = (x, 1.4, z)
        if k in wrong:
            c = (c[0] + offset[0], c[1] + offset[1], c[2] + offset[2])
        i = len(cams)
        cams.append(Cam(i, f"{i + 1:06d}.jpg", c, k * 0.5))
    return cams


class TestReviewFix9(unittest.TestCase):
    """Review finding #9 (30 Oct): ingest gaps read as jumps; long wrong stretches were kept."""

    def test_ingest_gap_is_not_a_jump_with_capture_time(self):
        # rh-phone: 6 blur windows dropped at 5.5-8 s; the old filter dropped frames 000001-000011
        cams = loop_room(396, gaps=set(range(11, 17)) | set(range(60, 64)) | set(range(120, 125)))
        r = filter_trajectory(cams)
        self.assertEqual(r.dropped, [])
        self.assertEqual(r.jumps_left, [])
        self.assertEqual(r.time_axis, "capture_time")

    def test_long_wrong_stretch_is_dropped(self):
        # rh-phone variant C: 16 frames (over 5% of 300) placed about 3 m off; the old filter kept them
        cams = loop_room(300, wrong=set(range(200, 216)), offset=(2.0, 0.0, 2.5))
        r = filter_trajectory(cams)
        self.assertEqual({d["idx"] for d in r.dropped}, set(range(200, 216)))
        self.assertTrue(all(d["reason"] == "off_local_path" for d in r.dropped))

    def test_single_wrong_camera_next_to_a_good_stretch(self):
        # rh-phone B: frame 66 was 84 cm off; the main-path rule alone then dropped the good frames after it
        cams = loop_room(300, wrong={65, 140}, offset=(0.0, 0.0, 1.2))
        r = filter_trajectory(cams)
        self.assertEqual({d["idx"] for d in r.dropped}, {65, 140})

    def test_correct_stretch_between_two_fast_steps_is_kept(self):
        # a quick sidestep out and back: two big steps, but the stretch between them is where the camera was
        cams = loop_room(200)
        for c in cams[80:95]:
            c.centre = (c.centre[0], 1.4, c.centre[2] + 0.4)
        r = filter_trajectory(cams)
        self.assertEqual(r.dropped, [])

    def test_frames_mode_falls_back_to_index(self):
        cams = loop_room(300, wrong=set(range(200, 216)), offset=(2.0, 0.0, 2.5))
        for c in cams:
            c.t = None
        r = filter_trajectory(cams)
        self.assertEqual(r.time_axis, "index")
        self.assertEqual({d["idx"] for d in r.dropped}, set(range(200, 216)))

    def test_wrong_start_stub_is_dropped_but_long_start_is_kept(self):
        cams = loop_room(200, wrong={0, 1}, offset=(3.0, 0.0, 0.0))
        r = filter_trajectory(cams)
        self.assertEqual({d["idx"] for d in r.dropped}, {0, 1})
        cams = loop_room(200, wrong=set(range(0, 20)), offset=(3.0, 0.0, 0.0))
        r = filter_trajectory(cams)
        self.assertEqual(r.dropped, [])
        self.assertEqual(len(r.jumps_left), 1)


if __name__ == "__main__":
    unittest.main()
