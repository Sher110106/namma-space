"""B8: `ns check` (pipeline/ingest/check_mac.py) gives the right verdict on b1-b7, each in <= 90 s on the Mac.

Standard library only.  Run on the Mac:

    python3 -m unittest discover -s test/unit/ingest -p 'test_check_mac.py' -v

Expected verdicts follow from how the fixtures were built (see make_fixtures.py) and the thresholds in FINAL_PLAN 2.2:
  b1 WARN  60 s is in the 45-90 s band; everything else passes
  b2 WARN  60 s, and HLG is flagged (we tone-map)
  b3 WARN  60 s; portrait 1080x1920 passes resolution, the rotation tag is honoured
  b4 WARN  60 s, and a third of the windows are blurred on purpose: 67% sharp is in the 60-80% band
  b5 FAIL  15 s is under 45 s
  b6 WARN  60 s and variable frame rate
  b7 PASS  3 min of 4K HEVC at 30 fps
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import _common as C  # noqa: E402
from pipeline.ingest import check_mac  # noqa: E402

EXPECT = {"b1": "WARN", "b2": "WARN", "b3": "WARN", "b4": "WARN", "b5": "FAIL", "b6": "WARN", "b7": "PASS"}
ONLY_FAILING = {  # check that must be non-PASS, by fixture
    "b2": "hdr", "b4": "blur", "b5": "duration", "b6": "frame_rate"}


class TestCheckMac(unittest.TestCase):
    pass


def _mk(name: str, want: str):
    def t(self):
        v = C.video(name)
        if v is None:
            self.skipTest(f"fixture {name}.mp4 not found")
        res = check_mac.check_all([str(v)])
        f = res["files"][0]
        print(f"{name}: {res['verdict']} in {f['seconds']:.0f} s  " + ", ".join(f"{c['name']}={c['status']}" for c in f["checks"]))
        self.assertEqual(res["verdict"], want, [(c["name"], c["status"], c["detail"]) for c in f["checks"]])
        self.assertLessEqual(f["seconds"], 90.0)
        if name in ONLY_FAILING:
            st = {c["name"]: c["status"] for c in f["checks"]}
            self.assertNotEqual(st[ONLY_FAILING[name]], "PASS")
        for c in f["checks"]:
            if c["status"] != "PASS":
                self.assertTrue(c["fix"], f"{c['name']} has no fix text")
    return t


for _n, _w in EXPECT.items():
    setattr(TestCheckMac, f"test_B8_{_n}_{_w.lower()}", _mk(_n, _w))


class TestCheckProbe(unittest.TestCase):
    def test_b3_rotation_and_b2_hdr(self):
        v = C.video("b3")
        if v is None:
            self.skipTest("no b3")
        p = check_mac.probe_video(str(v))
        self.assertEqual((p["display_width"], p["display_height"]), (1080, 1920))
        self.assertEqual(p["rotation_cw_deg"], 90)
        v = C.video("b2")
        if v is None:
            self.skipTest("no b2")
        p = check_mac.probe_video(str(v))
        self.assertEqual(p["hdr"], "hlg")
        self.assertEqual(p["bit_depth"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
