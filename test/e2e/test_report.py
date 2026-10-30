"""Tests for pipeline/report: report.json keys (CONTRACTS 2.13), explicit nulls with reasons, html.

Run: python3 test/e2e/test_report.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "e2e"))

from pipeline.report.build import REQUIRED, build_report, lookup  # noqa: E402
from test_runner import STUBS, make_run, runner_cmd  # noqa: E402

TOP_KEYS = ["schema", "name", "git", "images", "input", "frames", "sfm", "align", "train", "export", "grid", "viewer",
            "walls", "timings", "manual_overrides"]


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rhreport"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def full_stub_run(self, name="rep"):
        run = make_run(self.tmp, name)
        r = subprocess.run(runner_cmd(self.tmp, name, "--git", "deadbee"), capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return run

    def test_empty_run_gives_all_keys_as_nulls_with_reasons(self):
        run = self.tmp / "empty"
        run.mkdir()
        rep = build_report(run)
        for k in TOP_KEYS:
            self.assertIn(k, rep)
        self.assertIsNone(rep["sfm"]["chosen"])
        self.assertIn("20_sfm/sfm.json not found", rep["missing"]["sfm.chosen"])
        self.assertIsNone(rep["train"]["variants"])
        self.assertIsNone(rep["align"]["h_cap_m"])
        self.assertEqual(set(REQUIRED) - set(rep["empty_required"]), {"name"})
        self.assertIn("grid.walkable_m2", rep["empty_required"])
        self.assertTrue((run / "report.json").exists() and (run / "report.html").exists())

    def test_full_stub_run_fills_every_section(self):
        run = self.full_stub_run()
        rep = build_report(run)
        self.assertEqual(rep["git"], "deadbee")
        self.assertEqual(rep["frames"]["n_kept"], 3)
        self.assertEqual(rep["sfm"]["chosen"], "A")
        self.assertEqual(rep["train"]["chosen"], "r0")
        self.assertEqual(rep["align"]["scale"]["pct_68"], 10.0)
        self.assertEqual(rep["export"]["sd"]["count"], 800000)
        self.assertTrue(rep["grid"]["spawn_ok"])
        self.assertAlmostEqual(rep["timings"]["total"], sum(v for k, v in rep["timings"].items() if k != "total"), places=1)
        self.assertGreaterEqual(len(rep["stub_stages"]), 6)
        # only the Mac-side verify keys (and images, since there is no docker here) are still empty
        self.assertEqual(sorted(rep["empty_required"]),
                         sorted(["images", "viewer.devices", "walls.walks", "walls.pushes", "walls.penetrations"]))
        html_text = (run / "report.html").read_text()
        self.assertIn("Test stubs wrote data", html_text)
        self.assertIn("<h2>Training</h2>", html_text)

    def test_verify_json_completes_required_keys(self):
        run = self.full_stub_run("rep2")
        (run / "90_verify").mkdir()
        # the format W5's viewer/test/run.mjs writes
        (run / "90_verify" / "verify.json").write_text(json.dumps({
            "viewer": {"devices": [{"name": "M1", "gpu": "Apple M1", "browser": "chrome", "first_frame_s": 1.2, "fps_mean": 60,
                                    "p95_ms": 16.9, "mb": 30, "splats": 800000}]},
            "walls": {"walks": 200, "pushes": 16, "penetrations": 0, "wall_stops": 9}}))
        rj = json.loads((run / "run.json").read_text())
        rj["images"] = {"namma-space-gpu:0.1": "sha256:abc"}
        (run / "run.json").write_text(json.dumps(rj))
        rep = build_report(run)
        self.assertEqual(rep["empty_required"], [])
        self.assertEqual(rep["walls"]["penetrations"], 0)
        self.assertEqual(rep["viewer"]["devices"][0]["gpu"], "Apple M1")
        r = subprocess.run([sys.executable, "-m", "pipeline.report", "--run", str(run), "--strict"], cwd=str(ROOT),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_strict_fails_on_empty_required(self):
        run = self.tmp / "strict"
        run.mkdir()
        r = subprocess.run([sys.executable, "-m", "pipeline.report", "--run", str(run), "--strict"], cwd=str(ROOT),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("empty required: sfm.chosen", r.stdout)

    def test_unreadable_json_is_reported_not_fatal(self):
        run = self.tmp / "bad"
        (run / "20_sfm").mkdir(parents=True)
        (run / "20_sfm" / "sfm.json").write_text("{not json")
        rep = build_report(run)
        self.assertIsNone(rep["sfm"]["chosen"])
        self.assertIn("sfm.json", rep["missing"]["sfm.chosen"])

    def test_html_escapes_values(self):
        run = self.tmp / "esc"
        (run / "input").mkdir(parents=True)
        (run / "input" / "meta.json").write_text(json.dumps({"takes": [{"kind": "frames"}], "measures": []}))
        (run / "run.json").write_text(json.dumps({"git": "<script>x</script>", "manual_overrides": ["<b>hi</b>"]}))
        build_report(run)
        text = (run / "report.html").read_text()
        self.assertNotIn("<script>x", text)
        self.assertIn("&lt;script&gt;", text)

    def test_frames_mode_check_is_not_applicable(self):
        run = self.tmp / "fm"
        (run / "input").mkdir(parents=True)
        (run / "input" / "meta.json").write_text(json.dumps({"takes": [{"kind": "frames"}]}))
        rep = build_report(run)
        self.assertTrue(rep["missing"]["input.check"].startswith("n/a:"))


if __name__ == "__main__":
    unittest.main()
