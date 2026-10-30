"""Runs conventions.test.mjs (real three.js against test/data/conventions.json). Needs node on the PATH."""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(shutil.which("node"), "node not found")
@unittest.skipUnless((HERE.parents[1] / "viewer" / "vendor" / "three.module.js").exists(), "viewer/vendor/three.module.js missing")
class ConventionsJs(unittest.TestCase):
    def test_three_js_agrees_with_conventions_json(self):
        r = subprocess.run(["node", str(HERE / "conventions.test.mjs")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("checks passed", r.stdout)


if __name__ == "__main__":
    unittest.main()
