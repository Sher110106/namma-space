"""Mac-only tests of the ns CLI that need no ssh: name checks, serve, package, publish gating, check.

Run: python3 test/e2e/test_ns_cli.py
"""
from __future__ import annotations

import json
import os
import pty
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NS = str(ROOT / "ns")


def ns(*args, env=None, timeout=60, stdin=None):
    return subprocess.run([sys.executable, NS, *args], capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ, **(env or {})), stdin=stdin)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class NsCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rhns"))
        self.env = {"NS_LOCAL_RUNS": str(self.tmp)}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_run(self, name="demo", big=False):
        web = self.tmp / name / "80_web"
        (web / "scenes" / name).mkdir(parents=True)
        (web / "index.html").write_text("<html>hi</html>")
        (web / "scenes" / "index.json").write_text(json.dumps({"default": name, "scenes": [{"name": name, "title": name}]}))
        (web / "scenes" / name / "splat_sd.spz").write_bytes(b"\0" * 2048)
        if big:
            with (web / "scenes" / name / "splat_hd.spz").open("wb") as fh:
                fh.truncate(100 * 1024 * 1024)  # sparse, costs no disk
        (self.tmp / name / "report.json").write_text("{}")
        (self.tmp / name / "input").mkdir()
        (self.tmp / name / "input" / "meta.json").write_text(json.dumps(
            {"height_cm": 175, "measures": [{"what": "length", "metres": 4.12, "method": "tape"}]}))
        return self.tmp / name

    def test_help_lists_every_subcommand(self):
        r = ns("--help")
        self.assertEqual(r.returncode, 0)
        for sub in ("doctor", "sync", "check", "run", "status", "pull", "serve", "rescale", "verify", "package", "publish"):
            self.assertIn(sub, r.stdout)

    def test_bad_run_name_is_refused(self):
        r = ns("status", "Bad Name")
        self.assertEqual(r.returncode, 2)
        self.assertIn("must match", r.stderr)

    def test_run_needs_an_existing_video(self):
        r = ns("run", "/no/such/video.mov", "--name", "x1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no such video", r.stderr)

    def test_measure_syntax_is_checked(self):
        r = ns("run", "--frames", str(self.tmp), "--name", "x2", "--measure", "oops", "--no-sync", "--dry-run", env=self.env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--measure", r.stderr)

    def test_serve_serves_bundle_with_cors_and_mime(self):
        self.fake_run()
        port = free_port()
        p = subprocess.Popen([sys.executable, NS, "serve", "demo", "--port", str(port)], env=dict(os.environ, **self.env),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for _ in range(50):
                try:
                    r = urllib.request.urlopen(f"http://127.0.0.1:{port}/scenes/index.json", timeout=2)
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                self.fail("server did not start")
            self.assertEqual(json.loads(r.read())["default"], "demo")
            self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")
            r2 = urllib.request.urlopen(f"http://127.0.0.1:{port}/scenes/demo/splat_sd.spz", timeout=2)
            self.assertEqual(len(r2.read()), 2048)
            self.assertEqual(r2.headers["Content-Type"], "application/octet-stream")
        finally:
            p.terminate()
            p.wait(timeout=10)

    def test_serve_without_bundle_says_what_to_do(self):
        r = ns("serve", "nothing", env=self.env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("80_web", r.stderr)

    def test_package_builds_offline_zip_and_lists_missing_items(self):
        run = self.fake_run()
        r = ns("package", "demo", env=self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        pkg = run / "package"
        z = zipfile.ZipFile(pkg / "05_Viewer_Offline.zip")
        names = z.namelist()
        self.assertIn("web/index.html", names)
        self.assertIn("web/scenes/demo/splat_sd.spz", names)
        self.assertTrue(any(n.endswith("START_HERE.txt") for n in names))
        manifest = (pkg / "MANIFEST.txt").read_text()
        self.assertIn("missing:", manifest)
        self.assertIn("01_Technical_Abstract.pdf", manifest)
        self.assertIn("length: 4.12 m", (pkg / "04_Inputs_Used" / "measurements.txt").read_text())

    def test_publish_is_a_dry_run_by_default_and_flags_big_files(self):
        self.fake_run(big=True)
        r = ns("publish", "demo", env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dry run only", r.stdout)
        self.assertIn("BLOCKER", r.stdout)
        self.assertIn("splat_sd.spz", r.stdout)

    def test_publish_execute_needs_a_terminal(self):
        self.fake_run()
        r = ns("publish", "demo", "--execute", env=self.env, stdin=subprocess.DEVNULL)
        self.assertEqual(r.returncode, 2)
        self.assertIn("interactive terminal", r.stderr)

    def test_publish_execute_wrong_phrase_publishes_nothing(self):
        self.fake_run()
        master, slave = pty.openpty()
        p = subprocess.Popen([sys.executable, NS, "publish", "demo", "--execute"], stdin=slave, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=dict(os.environ, **self.env), text=True)
        os.close(slave)
        os.write(master, b"yes please\n")
        out, _ = p.communicate(timeout=20)
        os.close(master)
        self.assertEqual(p.returncode, 2, out)
        self.assertIn("nothing published", out)
        self.assertIn("PUBLISH demo TO THE PUBLIC INTERNET", out)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg missing")
    def test_check_fails_a_three_second_clip_with_exit_10(self):
        if not (ROOT / "pipeline" / "ingest" / "check_mac.py").exists():
            self.skipTest("check_mac.py not present")
        v = self.tmp / "tiny.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "3",
                        "-pix_fmt", "yuv420p", str(v)], check=True)
        r = ns("check", str(v), timeout=120)
        self.assertEqual(r.returncode, 10, r.stdout + r.stderr)
        self.assertIn("FAIL", r.stdout)


if __name__ == "__main__":
    unittest.main()
