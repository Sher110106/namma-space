"""Tests for pipeline/common/stage.py: atomic stage dirs, timing, disk guard (test 0.3, stdlib only).

Run: python3 test/e2e/test_stage.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.common import stage as st  # noqa: E402


class AtomicStage(unittest.TestCase):
    def test_success_renames_and_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            with st.atomic_stage(run, "10_frames") as out:
                (out / "a.txt").write_text("one")
                self.assertTrue((run / "10_frames.tmp").exists())
                self.assertFalse((run / "10_frames").exists())
            self.assertEqual((run / "10_frames" / "a.txt").read_text(), "one")
            self.assertFalse((run / "10_frames.tmp").exists())
            with st.atomic_stage(run, "10_frames") as out:
                (out / "b.txt").write_text("two")
            self.assertEqual(sorted(p.name for p in (run / "10_frames").iterdir()), ["b.txt"])
            self.assertFalse((run / "10_frames.old").exists())

    def test_failure_keeps_old_result_and_moves_tmp_aside(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            with st.atomic_stage(run, "s") as out:
                (out / "good").write_text("1")
            with self.assertRaises(RuntimeError):
                with st.atomic_stage(run, "s") as out:
                    (out / "bad").write_text("2")
                    raise RuntimeError("boom")
            self.assertEqual((run / "s" / "good").read_text(), "1")
            self.assertTrue((run / "s.failed" / "bad").exists())
            # a later good run clears the .failed leftover
            with st.atomic_stage(run, "s") as out:
                (out / "new").write_text("3")
            self.assertFalse((run / "s.failed").exists())

    def test_stale_tmp_from_killed_run_is_removed(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "s.tmp").mkdir()
            (run / "s.tmp" / "junk").write_text("x")
            with st.atomic_stage(run, "s") as out:
                self.assertEqual(list(out.iterdir()), [])


class Json(unittest.TestCase):
    def test_roundtrip_and_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x" / "y.json"
            st.write_json_atomic(p, {"a": 1})
            self.assertEqual(st.read_json(p), {"a": 1})
            self.assertEqual(st.read_json(Path(d) / "none.json", {"z": 0}), {"z": 0})
            with self.assertRaises(FileNotFoundError):
                st.read_json(Path(d) / "none.json")
            (Path(d) / "bad.json").write_text("{")
            with self.assertRaises(ValueError):
                st.read_json(Path(d) / "bad.json")


class Timing(unittest.TestCase):
    def test_timed_accumulates_total(self):
        with tempfile.TemporaryDirectory() as d:
            with st.timed(d, "a"):
                time.sleep(0.02)
            st.add_timing(d, "b", 1.5)
            t = st.read_json(Path(d) / "timings.json")
            self.assertEqual(set(t["stages"]), {"a", "b"})
            self.assertAlmostEqual(t["total"], t["stages"]["a"] + 1.5, places=2)


class DiskGuard(unittest.TestCase):
    def test_guard_passes_with_normal_disk(self):
        os.environ.pop(st.FAKE_FREE_ENV, None)
        self.assertGreater(st.disk_guard(tempfile.gettempdir(), 0.001), 0)

    def test_fake_low_disk_exits_20_fast(self):
        env = dict(os.environ, **{st.FAKE_FREE_ENV: "3"})
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, str(ROOT / "pipeline" / "common" / "stage.py"), "guard", tempfile.gettempdir()],
                           env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(r.returncode, 20, r.stderr)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIn("3.0 GB free", r.stderr)

    def test_stage_context_refuses_on_low_disk(self):
        os.environ[st.FAKE_FREE_ENV] = "1"
        try:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(SystemExit) as cm:
                    with st.stage_context(d, "ingest"):
                        self.fail("stage body must not run")
                self.assertEqual(cm.exception.code, 20)
                self.assertFalse((Path(d) / "10_frames.tmp").exists())
        finally:
            os.environ.pop(st.FAKE_FREE_ENV, None)


class StageContextFlow(unittest.TestCase):
    def test_full_flow_writes_dir_log_timing(self):
        os.environ.pop(st.FAKE_FREE_ENV, None)
        with tempfile.TemporaryDirectory() as d:
            with st.stage_context(d, "align", min_gb=0.01) as ctx:
                st.write_json_atomic(ctx.out / "align.json", {"ok": True})
                ctx.log("hello")
            run = st.Run(d)
            self.assertTrue(run.done("align"))
            self.assertIn("hello", (Path(d) / "logs" / "align.log").read_text())
            self.assertIn("align", st.read_json(Path(d) / "timings.json")["stages"])

    def test_stage_table_is_consistent(self):
        names = [s.name for s in st.STAGES]
        self.assertEqual(names, ["ingest", "sfm", "align", "train", "geometry", "export", "grid"])
        for s in st.STAGES:
            self.assertTrue(s.json.startswith(s.outputs[0] + "/"), s)


if __name__ == "__main__":
    unittest.main()
