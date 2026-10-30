"""Unit tests for tools/bundle.py (stdlib unittest). Run: python3 viewer/test/test_bundle.py"""
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import bundle  # noqa: E402

FIX = HERE / "fixtures" / "honka_run"


def quat_mat(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


class TFinal(unittest.TestCase):
    def test_identity_when_missing(self):
        self.assertEqual(bundle.parse_t_final(None)["scale"], 1.0)

    def test_w3_shape_scale_quaternion_translation(self):
        t = bundle.parse_t_final({"scale": 1.25, "quaternion": [0, 0, 0, 1], "translation": [0, 0.3, 0]})
        self.assertEqual(t["position"], [0, 0.3, 0]); self.assertEqual(t["scale"], 1.25)

    def test_matrix_roundtrip(self):
        q = [0.5, 0.5, 0.5, 0.5]
        r = quat_mat(q); s = 2.5
        m = [[s * r[i][j] for j in range(3)] + [[1.0, 2.0, 3.0][i]] for i in range(3)] + [[0, 0, 0, 1]]
        for form in (m, [v for row in m for v in row], {"matrix": [v for row in m for v in row]}):
            t = bundle.parse_t_final(form)
            self.assertAlmostEqual(t["scale"], s, places=9)
            self.assertEqual([round(v, 9) for v in t["position"]], [1.0, 2.0, 3.0])
            dot = abs(sum(a * b for a, b in zip(t["quaternion"], q)))
            self.assertAlmostEqual(dot, 1.0, places=9)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            bundle.parse_t_final("nope")


class Files(unittest.TestCase):
    def test_png_grey_reads_fixture_grid(self):
        w, h, data = bundle.png_grey(FIX / "70_grid" / "grid.png")
        info = bundle.png_info(FIX / "70_grid" / "grid.png")
        self.assertEqual((w, h), (info["width"], info["height"]))
        self.assertTrue(set(data) <= {0, 128, 255}); self.assertIn(255, set(data))

    def test_spz_header_v3_fixture(self):
        h = bundle.spz_header(FIX / "60_export" / "splat_sd.spz")
        self.assertEqual((h["version"], h["count"], h["sh_degree"]), (3, 800000, 1))

    def test_spz_header_v4_sample_if_present(self):
        p = HERE.parents[1] / "pipeline" / "export" / "samples" / "honka_smoke_sd.spz"
        if not p.exists():
            self.skipTest("W3 sample not present")
        h = bundle.spz_header(p)
        self.assertEqual((h["version"], h["sh_degree"]), (4, 1)); self.assertGreater(h["count"], 0)


class Views(unittest.TestCase):
    def test_view_entry_shapes(self):
        a = bundle.view_entry({"position": [1, 2], "yaw_deg": 30}, 1.4, 0)
        self.assertEqual(a["position"], [1.0, 1.4, 2.0])
        b = bundle.view_entry([1, 2, 45], 1.4, 0)
        self.assertEqual((b["position"], b["yaw_deg"]), ([1.0, 1.4, 2.0], 45.0))
        c = bundle.view_entry({"position": [1, 1.3, 2], "yaw": 10, "pitch": -5}, 1.4, 0)
        self.assertEqual((c["position"][1], c["yaw_deg"], c["pitch_deg"]), (1.3, 10.0, -5.0))


class Validate(unittest.TestCase):
    def test_validate_rejects_empty_folder(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            r = bundle.validate_web(Path(d))
            self.assertFalse(r["ok"]); self.assertTrue(r["errors"])

    def test_compose_fixture_scene_keys(self):
        import argparse
        a = argparse.Namespace(title="T", eye_height=None, kind="fixture", source="s", license="l", attribution="", captured="", manual_step=None)
        warns = []
        scene, files = bundle.compose_scene(FIX, "t", a, warns)
        for k in bundle.REQUIRED_TOP:
            self.assertIn(k, scene)
        self.assertIn("grid.png", files); self.assertEqual(scene["splats"]["sd"]["count"], 800000)
        q = scene["splats"]["transform"]["quaternion"]
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in q)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
