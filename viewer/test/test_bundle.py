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


def _args(**kw):
    import argparse
    base = dict(title="T", eye_height=None, kind="fixture", source="s", license="l", attribution="", captured="", manual_step=None, places=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _run_copy(tmp: Path) -> Path:
    """A run directory made of symlinks to the fixture, with its own 60_export and 40_align folders that tests may change."""
    run = tmp / "run"; run.mkdir()
    for entry in FIX.iterdir():
        if entry.name in ("60_export", "40_align"):
            (run / entry.name).mkdir()
            for f in entry.iterdir():
                (run / entry.name / f.name).symlink_to(f)
        else:
            (run / entry.name).symlink_to(entry)
    return run


class HdAndClear(unittest.TestCase):
    """Review findings 19, 47 (no HD button for a copy of SD) and 46 (clear colour)."""

    def test_identical_hd_is_dropped_and_not_copied(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            run = _run_copy(Path(d))
            (run / "60_export" / "splat_hd.spz").symlink_to(FIX / "60_export" / "splat_sd.spz")
            warns = []
            scene, files = bundle.compose_scene(run, "t", _args(), warns)
            self.assertIsNone(scene["splats"]["hd"]); self.assertNotIn("splat_hd.spz", files)
            self.assertTrue(any("byte-identical" in w for w in warns), warns)
            self.assertIsNotNone(scene["splats"]["sd"])

    def test_different_hd_is_kept(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            run = _run_copy(Path(d))
            hd = run / "60_export" / "splat_hd.spz"
            hd.write_bytes((FIX / "60_export" / "splat_sd.spz").read_bytes())   # same size, then one byte flipped inside the stream
            b = bytearray(hd.read_bytes()); b[-5] ^= 0xFF; hd.write_bytes(bytes(b))
            warns = []
            scene, files = bundle.compose_scene(run, "t", _args(), warns)
            self.assertIsNotNone(scene["splats"]["hd"]); self.assertIn("splat_hd.spz", files)
            self.assertTrue(any("different bytes" in w for w in warns), warns)

    def test_no_hd_file_means_no_hd_entry(self):
        scene, _ = bundle.compose_scene(FIX, "t", _args(), [])
        self.assertIsNone(scene["splats"]["hd"])

    def test_neutral_clear_stays_between_the_limits_and_loses_most_colour(self):
        for rgb in ([200, 120, 60], [5, 5, 5], [250, 250, 250], [30, 60, 200], [0, 0, 0]):
            c = bundle.neutral_clear(rgb)
            luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            self.assertTrue(bundle.CLEAR_LUMA[0] - 1.5 <= luma <= bundle.CLEAR_LUMA[1] + 1.5, (rgb, c, luma))
            self.assertTrue(all(0 <= v <= 255 for v in c))
        warm = bundle.neutral_clear([200, 120, 60])
        self.assertTrue(warm[0] > warm[1] > warm[2])                   # keeps the warm cast
        self.assertTrue(warm[0] - warm[2] < 200 - 60)                  # but much less of it

    def test_clear_colour_written_with_its_source(self):
        scene, _ = bundle.compose_scene(FIX, "t", _args(), [])
        r = scene["splats"]["render"]
        self.assertEqual(len(r["clear_color"]), 3); self.assertTrue(all(isinstance(v, int) and 0 <= v <= 255 for v in r["clear_color"]))
        self.assertTrue(r["clear_color_source"])
        # not the old near-black (11, 14, 20)
        self.assertGreater(sum(r["clear_color"]), 3 * 30)

    def test_clear_colour_from_photos_when_pillow_is_there(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ps = []
            for i, col in enumerate(((200, 120, 60), (180, 110, 70))):
                p = Path(d) / f"{i}.jpg"; Image.new("RGB", (32, 32), col).save(p, quality=95); ps.append(p)
            c, src = bundle.scene_clear_colour(ps, FIX / "70_grid" / "minimap.png")
            self.assertIn("2 held-out photos", src)
            self.assertTrue(c[0] > c[1] > c[2], c)

    def test_clear_colour_falls_back_to_the_default_grey(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "x.png"; bad.write_bytes(b"not a png")
            c, src = bundle.scene_clear_colour([], bad)
            self.assertEqual((c, src), (bundle.DEFAULT_CLEAR, "default grey"))

    def test_png_rgb_reads_the_minimap(self):
        w, h, px = bundle.png_rgb(FIX / "70_grid" / "minimap.png")
        self.assertEqual(len(px), w * h); self.assertEqual(len(px[0]), 3)


class ScaleAndPlaces(unittest.TestCase):
    def test_tape_check_reaches_scene_json_only_through_rescale_json(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            run = _run_copy(Path(d))
            scene, _ = bundle.compose_scene(run, "t", _args(), [])
            self.assertIsNone(scene["scale"]["checked_against"]); self.assertNotIn("check_error_pct", scene["scale"])
            (run / "40_align" / "rescale.json").write_text(json.dumps({"checked_against": "tape", "validation": {"n_used": 2, "auto_error_pct": 2.1349}}))
            scene, _ = bundle.compose_scene(run, "t", _args(), [])
            self.assertEqual(scene["scale"]["checked_against"], "tape"); self.assertEqual(scene["scale"]["check_error_pct"], 2.13); self.assertEqual(scene["scale"]["check_n"], 2)
            (run / "40_align" / "rescale.json").write_text(json.dumps({"checked_against": "tape", "validation": {"n_used": 0}}))
            scene, _ = bundle.compose_scene(run, "t", _args(), [])
            self.assertIsNone(scene["scale"]["checked_against"])          # no usable measure: not checked

    def test_place_notes_pass_through(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "p.json"
            f.write_text(json.dumps([{"name": "Door", "position": [0, 0, 0], "notes": "placed by hand"}, {"name": "Sofa", "position": [1, 0, 0]}]))
            out = bundle.load_places(f, None, [])
            self.assertEqual(out[0]["notes"], "placed by hand"); self.assertNotIn("notes", out[1])

    def test_video_link_survives_a_rebuild(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "index.html").write_text('<meta name="ns-video" content="https://example.org/v.mp4">')
            self.assertEqual(bundle.existing_video_url(Path(d)), "https://example.org/v.mp4")
            self.assertEqual(bundle.existing_video_url(Path(d) / "none"), "")


class PublicRoomPlaces(unittest.TestCase):
    """viewer/places/<run>.json: the hand-placed places for the public demo rooms (review findings 16 and 45)."""

    PROV = "placed by hand with the place editor (build team), 30 Oct"

    def test_files_are_valid_and_carry_their_provenance(self):
        files = sorted((Path(bundle.__file__).resolve().parent.parent / "places").glob("*.json"))
        self.assertGreaterEqual(len(files), 2, files)
        for f in files:
            warns = []
            places = bundle.load_places(f, None, warns)
            self.assertTrue(5 <= len(places) <= 7, f"{f.name}: {len(places)} places")
            ids = [p["id"] for p in places]
            self.assertEqual(len(ids), len(set(ids)), f.name)
            for p in places:
                self.assertTrue(p.get("notes", "").startswith(self.PROV), (f.name, p["id"]))
                self.assertTrue(len(p["aliases"]) >= 2, (f.name, p["id"]))
                self.assertEqual(len(p["position"]), 3)
                self.assertGreater(p["position"][1], 0.0, (f.name, p["id"]))           # picked on a surface, not below the floor


if __name__ == "__main__":
    unittest.main(verbosity=1)
