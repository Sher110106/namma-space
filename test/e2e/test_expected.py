"""Tests for check_expected.py and gates.json. Run: python3 test/e2e/test_expected.py"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_expected as ce  # noqa: E402

GATES = json.loads((HERE.parent / "expected" / "gates.json").read_text())


def report(**over):
    rep = {"sfm": {"chosen": "A", "variants": [{"name": "B", "reg_frac": 0.5, "reproj_px": 3.0},
                                               {"name": "A", "reg_frac": 0.96, "reproj_px": 0.8}]},
           "align": {"up_resid_deg": 1.0, "h_cap_m": 1.4},
           "train": {"chosen": "r0", "variants": [{"name": "r0", "psnr": 23.0, "n_gs": 900000, "steps": 30000}]},
           "export": {"sd": {"count": 800000, "mb": 30}, "hd": None, "preview": None},
           "grid": {"spawn_ok": True, "path_coverage": 0.97}, "walls": {"penetrations": 0},
           "viewer": {"devices": [{"fps_mean": 60, "p95_ms": 16, "first_frame_s": 2.0}]}, "timings": {"total": 3000}}
    rep.update(over)
    return rep


class Gates(unittest.TestCase):
    def status(self, rep, mode="full"):
        return {gid: s for gid, s, _ in ce.evaluate(rep, GATES, mode)}

    def test_good_report_passes_everything_it_has(self):
        s = self.status(report())
        self.assertNotIn("FAIL", s.values())
        self.assertEqual(s["sfm.reg_frac"], "PASS")
        self.assertEqual(s["export.hd.mb"], "NA")

    def test_chosen_variant_is_the_one_checked(self):
        rep = report()
        rep["sfm"]["chosen"] = "B"
        self.assertEqual(self.status(rep)["sfm.reg_frac"], "FAIL")

    def test_each_failure_is_caught(self):
        self.assertEqual(self.status(report(align={"up_resid_deg": 9, "h_cap_m": 1.4}))["align.up_resid_deg"], "FAIL")
        self.assertEqual(self.status(report(align={"up_resid_deg": 1, "h_cap_m": 2.5}))["align.h_cap_m"], "FAIL")
        self.assertEqual(self.status(report(walls={"penetrations": 2}))["walls.penetrations"], "FAIL")
        self.assertEqual(self.status(report(grid={"spawn_ok": False, "path_coverage": 0.97}))["grid.spawn_ok"], "FAIL")

    def test_any_device_below_the_fps_gate_fails(self):
        rep = report(viewer={"devices": [{"fps_mean": 60, "p95_ms": 16, "first_frame_s": 2}, {"fps_mean": 30, "p95_ms": 40, "first_frame_s": 9}]})
        s = self.status(rep)
        self.assertEqual((s["viewer.fps_mean"], s["viewer.p95_ms"], s["viewer.first_frame_s"]), ("FAIL", "FAIL", "FAIL"))

    def test_psnr_gate_is_for_full_runs_only(self):
        rep = report()
        rep["train"]["variants"][0]["psnr"] = 18.0
        self.assertEqual(self.status(rep, "full")["train.psnr"], "FAIL")
        self.assertEqual(self.status(rep, "quick")["train.psnr"], "SKIP")

    def test_missing_values_are_na_not_fail(self):
        s = self.status({})
        self.assertNotIn("FAIL", s.values())
        self.assertEqual(s["sfm.reg_frac"], "NA")

    def test_gates_file_has_a_reference_for_every_gate(self):
        for g in GATES["gates"]:
            self.assertTrue(g["ref"], g["id"])
            self.assertIn(g["op"], (">=", "<=", "==", "between"))


if __name__ == "__main__":
    unittest.main()
