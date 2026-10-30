import unittest

import _paths  # noqa: F401
from pipeline.sfm.choose import choose


def v(name, reg, reproj=0.8, pts=1000, track=4.0, status="ok", jumps=None):
    return {"name": name, "status": status, "reg_frac": reg, "reproj_px": reproj, "n_pts": pts,
            "track_len": track, "jumps_left": jumps or []}


class TestChoose(unittest.TestCase):
    def test_lowest_error_among_full(self):
        r = choose([v("A", 0.97, 0.9), v("B", 0.96, 0.7), v("D", 0.99, 1.1)])
        self.assertEqual(r["chosen"], "B")
        self.assertFalse(r["partial"])

    def test_tie_on_error_most_points(self):
        r = choose([v("A", 0.97, 0.8, 500), v("B", 0.97, 0.8, 900)])
        self.assertEqual(r["chosen"], "B")

    def test_below_95_is_not_preferred_for_low_error(self):
        r = choose([v("A", 0.96, 1.0), v("B", 0.90, 0.5)])
        self.assertEqual(r["chosen"], "A")

    def test_none_full_picks_highest_share_and_flags_partial(self):
        r = choose([v("A", 0.87, 0.75), v("B", 0.91, 0.9), v("D", 0.70, 0.5)])
        self.assertEqual(r["chosen"], "B")
        self.assertTrue(r["partial"])
        self.assertTrue(r["usable"])

    def test_crashed_and_low_variants_dropped(self):
        r = choose([v("A", 0.0, status="crashed"), v("B", 0.55), v("D", 0.80)])
        self.assertEqual(r["chosen"], "D")
        self.assertIn("below 60%", r["why"])

    def test_nothing_usable(self):
        r = choose([v("A", 0.5), v("B", 0.0, status="crashed")])
        self.assertIsNone(r["chosen"])
        self.assertFalse(r["usable"])

    def test_sane_variant_preferred_over_insane(self):
        r = choose([v("A", 0.99, 0.5, track=2.0), v("B", 0.96, 0.9)])
        self.assertEqual(r["chosen"], "B")
        r = choose([v("A", 0.99, 0.5, jumps=[{"ratio": 9}]), v("B", 0.96, 0.9)])
        self.assertEqual(r["chosen"], "B")

    def test_all_insane_still_chooses_and_warns(self):
        r = choose([v("A", 0.99, 0.5, track=2.0)])
        self.assertEqual(r["chosen"], "A")
        self.assertTrue(r["warnings"])

    def test_high_reprojection_warns(self):
        r = choose([v("A", 0.99, 1.5)])
        self.assertTrue(any("1.2" in w for w in r["warnings"]))


if __name__ == "__main__":
    unittest.main()
