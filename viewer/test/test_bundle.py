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


class Places(unittest.TestCase):
    """--places: merge, defaults, and clear errors (Round 2 preview)."""

    def load(self, obj, grid=None):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "places.json"
            f.write_text(json.dumps(obj))
            warns = []
            return bundle.load_places(f, grid, warns), warns

    def test_list_and_wrapped_forms(self):
        p = {"name": "Kitchen", "kind": "room", "position": [1, 0, 2], "aliases": ["rasoi"]}
        a, _ = self.load([p]); b, _ = self.load({"places": [p]})
        self.assertEqual(a, b); self.assertEqual(a[0]["id"], "kitchen"); self.assertEqual(a[0]["radius_m"], 1.5); self.assertEqual(a[0]["aliases"], ["rasoi"])

    def test_defaults_for_xz_positions_and_alias_string(self):
        out, _ = self.load([{"name": "Study desk", "position": [2.0, 3.0], "aliases": "work table, laptop desk"}, {"name": "Hall", "kind": "zone", "position": [0, 1]}])
        self.assertEqual(out[0]["position"], [2.0, 1.0, 3.0]); self.assertEqual(out[0]["kind"], "item"); self.assertEqual(out[0]["radius_m"], 0.4)
        self.assertEqual(out[0]["aliases"], ["work table", "laptop desk"]); self.assertEqual(out[1]["position"], [0.0, 0.0, 1.0])

    def test_duplicate_names_get_unique_ids(self):
        out, _ = self.load([{"name": "Chair", "position": [0, 0]}, {"name": "Chair", "position": [1, 0]}, {"name": "Chair!", "position": [2, 0]}])
        self.assertEqual([p["id"] for p in out], ["chair", "chair-2", "chair-3"])

    def test_errors_are_clear(self):
        for bad, frag in (([{"name": "", "position": [0, 0]}], "name is required"), ([{"name": "A", "position": [0]}], "position"),
                          ([{"name": "A", "position": [0, 0], "kind": "pet"}], "kind"), ([{"name": "A", "position": [0, 0], "radius_m": -1}], "radius_m"),
                          ([{"name": "A", "position": [float("nan"), 0]}], "position"), ({"places": 3}, "expected a JSON list"), (["x"], "not an object")):
            with self.assertRaises(ValueError) as cm:
                self.load(bad)
            self.assertIn(frag, str(cm.exception))

    def test_warns_when_far_from_walkable_floor(self):
        w, h, data = bundle.png_grey(FIX / "70_grid" / "grid.png")
        grid = (w, h, data, [-6.75, -6.25], 0.05)
        _, warns = self.load([{"name": "Far away", "position": [40, 0, 40]}, {"name": "Near", "position": [0.0, 0.0, 0.0]}], grid)
        self.assertEqual(len(warns), 1); self.assertIn("Far away", warns[0])

    def test_compose_merges_places_and_run_dir_file(self):
        import argparse
        import json
        import tempfile
        a = argparse.Namespace(title="T", eye_height=None, kind="fixture", source="s", license="l", attribution="", captured="", manual_step=None, places=str(HERE / "fixtures" / "honka_places.json"))
        scene, _ = bundle.compose_scene(FIX, "t", a, [])
        self.assertEqual(len(scene["places"]), 8); self.assertEqual(scene["places"][0]["id"], "wall-display")
        a.places = None
        scene2, _ = bundle.compose_scene(FIX, "t", a, [])
        self.assertEqual(scene2["places"], [])                        # no file, no places.json in the fixture run
        with tempfile.TemporaryDirectory() as d:                      # RUN_DIR/places.json is picked up when --places is absent
            run = Path(d) / "run"; run.mkdir()
            for entry in FIX.iterdir():
                (run / entry.name).symlink_to(entry)
            (run / "places.json").write_text(json.dumps([{"name": "Sofa", "position": [0, 0, 0]}]))
            scene3, _ = bundle.compose_scene(run, "t", a, [])
            self.assertEqual([p["name"] for p in scene3["places"]], ["Sofa"])

    def test_validator_rejects_bad_places(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            web = Path(d) / "web"
            import shutil
            shutil.copytree(HERE.parent / "dist", web)
            sp = web / "scenes" / "honka_demo" / "scene.json"
            j = json.loads(sp.read_text())
            self.assertTrue(bundle.validate_web(web)["ok"])
            j["places"] = j["places"] + [dict(j["places"][0])]       # repeated id
            sp.write_text(json.dumps(j))
            self.assertFalse(bundle.validate_web(web)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
