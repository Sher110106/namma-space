"""Review fix 3 (30 Oct): `ns run --repo-dir repo-x` runs from a frozen copy of the code on tyrone.

    NS_E2E_REMOTE=1 python3 test/e2e/test_frozen_repo.py
Stub stages run in real containers through docker/nsdocker; the run is runs/fix-e2e-frozen and the copy
<root>/repo-fixtest. Both are deleted at the end.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "e2e"))
from test_remote import FRAMES, RROOT, enabled, remote_json, ssh  # noqa: E402

NS = str(ROOT / "ns")
NAME = "fix-e2e-frozen"
COPY = "repo-fixtest"


@unittest.skipUnless(enabled(), "set NS_E2E_REMOTE=1 and make sure ssh to tyrone works")
class FrozenRepo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nsfrozen"))
        self.frames = self.tmp / "frames"
        self.frames.mkdir()
        for f in sorted(FRAMES.glob("*.jpg"))[:6]:
            shutil.copy2(f, self.frames / f.name)
        self.env = dict(os.environ, NS_LOCAL_RUNS=str(self.tmp / "runs"))

    def tearDown(self):
        ssh(f"rm -rf {RROOT}/runs/{NAME} {RROOT}/{COPY} {RROOT}/{COPY}.old-*", 60)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_uses_the_frozen_copy_inside_containers(self):
        r = subprocess.run([sys.executable, NS, "run", "--frames", str(self.frames), "--name", NAME, "--no-check",
                            "--no-pull", "--quick", "--repo-dir", COPY, "--to", "align",
                            "--opt", "ingest:--force", "--runner-flag", "--stub-dir test/e2e/stubs"],
                           capture_output=True, text=True, timeout=1500, env=self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"code frozen in {RROOT}/{COPY}", r.stdout)
        frozen = json.loads(ssh(f"cat {RROOT}/{COPY}/FROZEN.json").stdout)
        self.assertTrue(frozen["git"])
        rj = remote_json(NAME, "run.json")
        self.assertEqual(rj["repo_in_container"], f"/w/{COPY}")
        self.assertTrue(rj["repo"].endswith(f"/{COPY}"))
        where = ssh(f"cat {RROOT}/runs/{NAME}/stub_where.log").stdout.split("\n")
        stages = [ln.split() for ln in where if ln.strip()]
        self.assertEqual([s[0] for s in stages], ["ingest", "sfm", "align"])
        for _, path in stages:
            self.assertEqual(path, f"/w/{COPY}/test/e2e/stubs")
        # a second freeze moves the first copy aside instead of deleting it
        r2 = subprocess.run([sys.executable, NS, "run", "--name", NAME, "--no-pull", "--repo-dir", COPY, "--to", "align",
                             "--runner-flag", "--stub-dir test/e2e/stubs"], capture_output=True, text=True,
                            timeout=600, env=self.env)
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        self.assertIn(".old-", r2.stdout)


if __name__ == "__main__":
    unittest.main()
