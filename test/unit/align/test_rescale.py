"""pipeline.align.rescale: records T_user in 40_align/rescale.json, validates against tape, never touches the model."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the bare Mac has no numpy: skip, do not error
    raise unittest.SkipTest("numpy missing (run in the container or a venv)")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.align import rescale  # noqa: E402


def make_run(measures=None, room_extent=None, ceiling=2.5):
    d = tempfile.mkdtemp(prefix="ns_rescale_")
    os.makedirs(os.path.join(d, "40_align", "sparse", "0"))
    Path(d, "40_align", "sparse", "0", "images.bin").write_bytes(b"model bytes")
    json.dump({"scale": {"value": 0.5, "sigma_log": 0.12, "pct_68": 12.7}, "h_cap_m": 1.4,
               "floor": {"ceiling_height_m": ceiling}}, open(os.path.join(d, "40_align", "align.json"), "w"))
    os.makedirs(os.path.join(d, "input"))
    json.dump({"height_cm": 175, "measures": measures or []}, open(os.path.join(d, "input", "meta.json"), "w"))
    if room_extent:
        os.makedirs(os.path.join(d, "70_grid"))
        json.dump({"room_extent_m": room_extent}, open(os.path.join(d, "70_grid", "grid.json"), "w"))
    return d


def digest(d):
    return hashlib.sha256(Path(d, "40_align", "sparse", "0", "images.bin").read_bytes()).hexdigest()


class TestRescale(unittest.TestCase):
    def test_scale_multiplies_and_model_untouched(self):
        d = make_run()
        h = digest(d)
        self.assertEqual(rescale.main(["--run", d, "--scale", "1.1"]), 0)
        self.assertEqual(rescale.main(["--run", d, "--scale", "1.1"]), 0)
        r = rescale.load(d)
        self.assertAlmostEqual(r["factor_total"], 1.21)
        self.assertAlmostEqual(r["scale_final"], 0.5 * 1.21)
        self.assertAlmostEqual(r["h_cap_m_final"], 1.4 * 1.21)
        self.assertEqual(len(r["applied"]), 2)
        self.assertTrue(np.allclose(np.array(r["T_user"]), np.diag([1.21, 1.21, 1.21, 1.0])))
        self.assertEqual(digest(d), h)
        self.assertEqual(rescale.main(["--run", d, "--reset"]), 0)
        self.assertEqual(rescale.factor(d), 1.0)
        shutil.rmtree(d)

    def test_compose_floor_offset_then_scale(self):
        d = make_run()
        rescale.main(["--run", d, "--set-total", "2.0"])
        Tprev = np.eye(4)
        Tprev[1, 3] = -0.05
        T = np.array(rescale.compose_T_final(d, Tprev.tolist()))
        self.assertTrue(np.allclose(T @ np.array([1, 0.05, 1, 1.0]), [2, 0, 2, 1]))   # floor point stays at y = 0
        shutil.rmtree(d)

    def test_use_measures_adopts_tape_and_keeps_both_numbers(self):
        # model says 5.0 x 4.0 m and 2.5 m; tape says 5.5, 4.4, 2.75: the model is 10% small
        d = make_run(measures=[{"what": "room length", "metres": 5.5, "method": "tape"},
                               {"what": "room width", "metres": 4.4, "method": "tape"},
                               {"what": "ceiling height", "metres": 2.75, "method": "tape"}], room_extent=[4.0, 5.0])
        self.assertEqual(rescale.main(["--run", d, "--check"]), 0)
        r = rescale.load(d)
        self.assertEqual(r["factor_total"], 1.0)                       # check changes nothing
        v = r["validation"]
        self.assertEqual(v["n_used"], 3)
        self.assertAlmostEqual(v["ratio_geomean"], 1.1, places=6)
        self.assertAlmostEqual(v["auto_error_pct"], -9.0909, places=3)
        self.assertEqual(rescale.main(["--run", d, "--use-measures"]), 0)
        r = rescale.load(d)
        self.assertAlmostEqual(r["factor_total"], 1.1, places=6)
        self.assertAlmostEqual(r["auto"]["scale"], 0.5)                # both numbers kept
        self.assertAlmostEqual(r["scale_final"], 0.55, places=6)
        self.assertEqual(r["checked_against"], "tape")
        self.assertEqual(rescale.main(["--run", d, "--use-measures"]), 0)   # idempotent
        self.assertAlmostEqual(rescale.factor(d), 1.1, places=6)
        shutil.rmtree(d)

    def test_grid_user_scale_is_divided_out(self):
        d = make_run(measures=[{"what": "length", "metres": 5.0, "method": "tape"}], room_extent=[4.4, 5.5])
        g = json.load(open(os.path.join(d, "70_grid", "grid.json")))
        g["user_scale"] = 1.1
        json.dump(g, open(os.path.join(d, "70_grid", "grid.json"), "w"))
        v = rescale.validate(d)
        self.assertAlmostEqual(v["ratio_geomean"], 1.0, places=6)
        shutil.rmtree(d)

    def test_refuses_absurd_factor_and_missing_align(self):
        d = make_run()
        self.assertEqual(rescale.main(["--run", d, "--scale", "9"]), 20)
        shutil.rmtree(d)
        e = tempfile.mkdtemp()
        self.assertEqual(rescale.main(["--run", e, "--scale", "1.1"]), 20)
        shutil.rmtree(e)

    def test_no_measures_is_not_an_error(self):
        d = make_run()
        self.assertEqual(rescale.main(["--run", d, "--use-measures"]), 0)
        self.assertEqual(rescale.factor(d), 1.0)
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
