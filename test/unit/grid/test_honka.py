"""G1 and G2 on the honka stand-in run (see pipeline/grid/standin_honka.py and NOTES.md: it is a stand-in, not the
align/export output). The run directory comes from W4_HONKA_RUN, else /w/runs/w4-honka_standin (tyrone container),
else /tmp/w4/honka_run (a copy on the Mac). The test is skipped when none exists.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import common  # noqa: F401
from common import Grid, tmpdir
from pipeline.grid import colmapio, plyio
from pipeline.grid.stage import run_stage


def find_run():
    for p in (os.environ.get("W4_HONKA_RUN"), "/w/runs/w4-honka_standin", "/tmp/w4/honka_run"):
        if p and (Path(p) / "40_align" / "sparse" / "0").is_dir() and (Path(p) / "60_export").is_dir():
            return Path(p)
    return None


RUN = find_run()


@unittest.skipIf(RUN is None, "honka stand-in run not found (set W4_HONKA_RUN)")
class HonkaStandIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir("w4honka_")
        t0 = time.time()
        cls.rc = run_stage(str(RUN), out_dir=str(cls.out / "70_grid"), quiet=True)
        cls.wall = time.time() - t0
        cls.g = Grid(cls.out / "70_grid")
        cls.walk = cls.g.walkable
        cls.model = colmapio.read_model(RUN / "40_align" / "sparse" / "0")
        cls.hcap = json.loads((RUN / "40_align" / "align.json").read_text())["h_cap"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_G1_spawn_walkable_one_component_coverage_bookmarks(self):
        g = self.g
        self.assertEqual(self.rc, 0)
        self.assertEqual(g.j["mode"], "grid")
        # the spawn is on a walkable cell
        sp = g.j["spawn"]["position"]
        i, j = g.pix(sp[0], sp[2])
        self.assertTrue(self.walk[i, j], "spawn not walkable")
        # one component
        lab, n = ndi.label(self.walk, structure=ndi.generate_binary_structure(2, 1))
        self.assertEqual(n, 1, "walkable area is not one component")
        # at least 95% of camera floor points are inside it
        c = self.model.centres
        ii = np.floor((c[:, 2] - g.oz) / g.c).astype(int)
        jj = np.floor((c[:, 0] - g.ox) / g.c).astype(int)
        cov = float(self.walk[ii, jj].mean())
        self.assertGreaterEqual(cov, 0.95, f"path coverage {cov:.3f}")
        self.assertAlmostEqual(cov, g.j["path_coverage"], places=3)
        # every bookmark is reachable: it sits on a walkable cell of the spawn's component
        self.assertEqual(len(g.j["bookmarks"]), 5)
        for b in g.j["bookmarks"]:
            bi, bj = g.pix(b["position"][0], b["position"][2])
            self.assertEqual(lab[bi, bj], lab[i, j], f"{b['id']} unreachable")
        # tour stays on the walkable area
        for k in g.j["tour"]:
            ti, tj = g.pix(k["position"][0], k["position"][2])
            self.assertTrue(self.walk[ti, tj])
        print(f"\nhonka G1: walkable {g.j['walkable_m2']} m2, path coverage {cov:.3f}, bookmarks {len(g.j['bookmarks'])}, "
              f"room extent {g.j['room_extent_m']}, spawn {g.j['spawn']}")

    def test_G2_no_walkable_cell_has_eye_in_wall(self):
        ply = plyio.read_splats(RUN / "60_export" / "clean.ply")
        tree = cKDTree(ply.xyz[ply.opacity > 0.5])
        wx, wz = self.g.centres(self.walk)
        q = np.stack([wx, np.full_like(wx, self.hcap), wz], axis=1)
        n = tree.query_ball_point(q, r=0.15, return_length=True)
        print(f"\nhonka G2: {len(q)} walkable cells, max opaque centres within 0.15 m of the eye point = {int(np.max(n))}")
        self.assertLessEqual(int(np.max(n)), 2)

    def test_fast(self):
        self.assertLess(self.wall, 60.0)


if __name__ == "__main__":
    unittest.main()
