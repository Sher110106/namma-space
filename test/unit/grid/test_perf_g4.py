"""G4: the grid builds in 60 s or less for 1.5 M splats (whole stage, including reading a 59-property PLY).

Set W4_SKIP_PERF=1 to skip (the test writes a 350 MB PLY to the temp directory).
"""
from __future__ import annotations

import json
import os
import shutil
import time
import unittest

import common  # noqa: F401
import synth
from common import Grid, tmpdir
from pipeline.grid.stage import run_stage


@unittest.skipIf(os.environ.get("W4_SKIP_PERF"), "W4_SKIP_PERF set")
class PerfG4(unittest.TestCase):
    def test_G4_1_5M_splats_within_60s(self):
        run = tmpdir("w4perf_")
        try:
            # a real gsplat PLY has 45 SH rest properties (about 350 MB for 1.5 M splats); fall back to none if the disk is tight
            sh_rest = int(os.environ.get("W4_SH_REST", "45"))
            if shutil.disk_usage(run).free < 1.5e9 and sh_rest:
                print(f"\nG4: only {shutil.disk_usage(run).free / 1e6:.0f} MB free in the temp directory; using sh_rest=0")
                sh_rest = 0
            t0 = time.time()
            synth.write_run(run, n_splats=1_520_000, sh_rest=sh_rest)
            t_gen = time.time() - t0
            size_mb = (run / "60_export" / "clean.ply").stat().st_size / 1e6
            t0 = time.time()
            rc = run_stage(str(run), quiet=True)
            wall = time.time() - t0
            g = Grid(run / "70_grid")
            n = g.j["counts"]["splats"]
            print(f"\nG4: {n:,} splats, {size_mb:.0f} MB PLY, stage wall {wall:.1f} s (generation {t_gen:.0f} s), "
                  f"timings {json.dumps(g.j['timings_s'])}")
            self.assertGreaterEqual(n, 1_500_000)
            self.assertEqual(rc, 0)
            self.assertLessEqual(wall, 60.0)
            self.assertLessEqual(g.j["timings_s"]["total_s"], 60.0)
        finally:
            shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
