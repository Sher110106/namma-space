"""Review #25 (30 Oct): the scale check must not pretend to check a scale that rests on one prior."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    from pipeline.align import core as C
    HAVE = True
except ImportError:   # numpy missing
    HAVE = False

GATE = ("up cues disagree", "no floor", "H_cap x scale", "H_cap is not", "scale cues disagree")


@unittest.skipUnless(HAVE, "numpy missing (use pipeline/export/.venv/bin/python)")
class ScaleHonesty(unittest.TestCase):
    def test_one_prior_says_so_and_does_not_gate(self):
        cues = [C.camera_height_cue(2.8, None)]
        s, _, cues = C.fuse_scale(cues)
        flags = C.scale_flags(cues, 2.8 * s)
        self.assertTrue(any(f.startswith("scale rests on one prior") for f in flags))
        self.assertFalse([f for f in flags if f.startswith(GATE)])

    def test_typo_in_height_still_gates(self):
        cues = [C.camera_height_cue(2.8, 3.0)]       # --height 300 cm by mistake: 0.8 x 3.0 = 2.4 m
        s, _, cues = C.fuse_scale(cues)
        flags = C.scale_flags(cues, 2.8 * s)
        self.assertTrue([f for f in flags if f.startswith(GATE)])

    def test_second_cue_that_disagrees_gates(self):
        h = C.camera_height_cue(2.8, 1.75)
        tape = C.Cue("measures", h.value * 1.6, 0.03, "tape length")
        s, _, cues = C.fuse_scale([h, tape])
        flags = C.scale_flags(cues, 2.8 * s)
        self.assertTrue([f for f in flags if f.startswith("scale cues disagree")], flags)

    def test_second_cue_that_agrees_passes(self):
        h = C.camera_height_cue(2.8, 1.75)
        tape = C.Cue("measures", h.value * 1.03, 0.03, "tape length")
        s, _, cues = C.fuse_scale([h, tape])
        self.assertEqual(C.scale_flags(cues, 2.8 * s), [])


if __name__ == "__main__":
    unittest.main()
