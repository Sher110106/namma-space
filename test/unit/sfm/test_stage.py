"""Orchestration tests for pipeline.sfm with a fake colmap (see fake_colmap.py). No GPU, no real SfM."""
import json
import os
import shutil
import tempfile
import unittest

import _paths  # noqa: F401
import helpers
from pipeline.sfm import stage

N = 100


class StageCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sfmtest")
        self.run_dir = os.path.join(self.tmp, "run")
        self.images = os.path.join(self.run_dir, "10_frames", "images")
        os.makedirs(self.images)
        for i in range(1, N + 1):
            with open(os.path.join(self.images, f"{i:06d}.jpg"), "wb") as f:
                f.write(b"x")
        with open(os.path.join(self.run_dir, "10_frames", "frames.json"), "w") as f:
            json.dump({"kept": [{"name": f"{i:06d}.jpg", "t": round(i / 8, 3), "sharp": 100.0 - i} for i in range(1, N + 1)]}, f)
        self.cache = os.path.join(self.tmp, "cache")
        os.makedirs(self.cache)
        for name in ("aliked-n32.onnx", "aliked-lightglue.onnx", "sift-lightglue.onnx",
                     "vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin", "vocab_tree_faiss_flickr100K_words256K.bin"):
            with open(os.path.join(self.cache, name), "wb") as f:
                f.write(b"x")
        self.colmap = helpers.install_fake_colmap(os.path.join(self.tmp, "bin"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def go(self, scenario, *extra):
        os.environ["FAKE_SCENARIO"] = json.dumps(scenario)
        self.addCleanup(os.environ.pop, "FAKE_SCENARIO", None)
        argv = ["--run", self.run_dir, "--cache", self.cache, "--colmap", self.colmap, "--min-free-gb", "0.001",
                "--variant-timeout", "60", "--retry-delay", "0.1", *extra]
        rc = stage.main(argv)
        path = os.path.join(self.run_dir, "20_sfm", "sfm.json")
        sfm = None
        if os.path.isfile(path):
            with open(path) as f:
                sfm = json.load(f)
        return rc, sfm

    @staticmethod
    def good(models=None, **k):
        d = {"models": models or [{"names": f"1-{N}"}], "reproj": 0.8, "pts": 1000, "track": 4.0}
        d.update(k)
        return d


class TestRace(StageCase):
    def test_all_variants_ok_lowest_error_wins(self):
        sc = {"A": self.good(reproj=0.9), "B": self.good(reproj=0.7), "D": self.good(reproj=1.0)}
        rc, sfm = self.go(sc)
        self.assertEqual(rc, 0)
        self.assertEqual(sfm["chosen"], "B")
        self.assertFalse(sfm["partial"])
        self.assertEqual({v["name"] for v in sfm["variants"]}, {"A", "B", "D"})
        for v in sfm["variants"]:
            for key in ("name", "reg_frac", "n_models", "reproj_px", "n_pts", "track_len", "time_s", "status"):
                self.assertIn(key, v)
        self.assertTrue(os.path.isfile(os.path.join(self.run_dir, "30_undist", "sparse", "cameras.bin")))
        self.assertTrue(os.path.isfile(os.path.join(self.run_dir, "30_undist", "sparse", "0", "images.bin")))
        self.assertEqual(len(os.listdir(os.path.join(self.run_dir, "30_undist", "images"))), N)
        self.assertEqual(sfm["undist"]["camera_model"], "PINHOLE")
        self.assertFalse(os.path.exists(os.path.join(self.run_dir, "20_sfm.tmp")))
        self.assertFalse(os.path.exists(os.path.join(self.run_dir, "30_undist.tmp")))
        self.assertFalse(os.path.exists(os.path.join(self.run_dir, "30_undist", "stereo")))

    def test_crashed_variant_does_not_stop_the_race(self):
        sc = {"A": self.good(fail="match"), "B": self.good(), "D": self.good()}
        rc, sfm = self.go(sc)
        self.assertEqual(rc, 0)
        status = {v["name"]: v["status"] for v in sfm["variants"]}
        self.assertEqual(status["A"], "crashed")
        self.assertEqual(status["D"], "skipped")
        self.assertEqual(status["B"], "ok")
        self.assertEqual(sfm["chosen"], "B")

    def test_far_cameras_are_dropped_and_counted_unregistered(self):
        sc = {"A": self.good(models=[{"names": f"1-{N}", "far": "40-46"}])}
        rc, sfm = self.go(sc, "--variants", "A")
        self.assertEqual(rc, 0)
        self.assertEqual(len(sfm["dropped_cameras"]), 7)
        a = sfm["variants"][0]
        self.assertEqual(a["n_registered"], N - 7)
        self.assertAlmostEqual(a["reg_frac"], (N - 7) / N, places=3)
        self.assertAlmostEqual(a["reg_frac_raw"], 1.0, places=3)
        self.assertEqual(len(sfm["missing_ranges"]), 1)
        r = sfm["missing_ranges"][0]
        self.assertEqual((r["first"], r["last"], r["n_frames"]), ("000040.jpg", "000046.jpg", 7))
        self.assertAlmostEqual(r["t_start"], 5.0)
        un = {u["name"]: u for u in sfm["unregistered"]}
        self.assertEqual(len(un), 7)
        self.assertTrue(un["000040.jpg"]["reason"].startswith("dropped"))
        self.assertEqual(un["000040.jpg"]["n_features"], 140)
        self.assertEqual(un["000040.jpg"]["sharp"], 60.0)

    def test_partial_when_no_variant_reaches_95(self):
        sc = {"A": self.good(models=[{"names": "1-85"}]), "B": self.good(models=[{"names": "1-80"}]), "D": {"fail": "map"}}
        rc, sfm = self.go(sc, "--no-c")
        self.assertEqual(rc, 0)
        self.assertTrue(sfm["partial"])
        self.assertEqual(sfm["chosen"], "A")
        self.assertEqual([(r["first"], r["last"]) for r in sfm["missing_ranges"]], [("000086.jpg", "000100.jpg")])
        self.assertEqual(len(sfm["unregistered"]), 15)

    def test_two_scenes_give_two_models_largest_chosen_partial(self):
        sc = {"A": self.good(models=[{"names": "1-60"}, {"names": "61-100"}]), "B": {"fail": "extract"}, "D": {"fail": "map"}}
        rc, sfm = self.go(sc, "--no-c")
        self.assertEqual(rc, 0)
        a = [v for v in sfm["variants"] if v["name"] == "A"][0]
        self.assertEqual(a["n_models"], 2)
        self.assertEqual(a["n_registered"], 60)
        self.assertTrue(sfm["partial"])

    def test_variant_c_runs_when_best_is_below_90(self):
        sc = {"A": self.good(models=[{"names": "1-70"}]), "B": self.good(models=[{"names": "1-60"}]),
              "D": {"fail": "map"}, "C": self.good(models=[{"names": f"1-{N}"}], reproj=0.9)}
        rc, sfm = self.go(sc)
        self.assertEqual(rc, 0)
        self.assertEqual(sfm["chosen"], "C")
        self.assertIn("C", {v["name"] for v in sfm["variants"]})

    def test_nothing_usable_exits_10_and_still_writes_sfm_json(self):
        sc = {"A": self.good(models=[{"names": "1-40"}]), "B": self.good(models=[{"names": "1-30"}]), "D": {"fail": "map"}}
        rc, sfm = self.go(sc, "--no-c")
        self.assertEqual(rc, 10)
        self.assertIsNone(sfm["chosen"])
        self.assertEqual(sfm["best_unusable"], "A")
        self.assertEqual(sfm["missing_ranges"][0]["first"], "000041.jpg")
        self.assertEqual(sfm["missing_ranges"][0]["last"], "000100.jpg")
        self.assertFalse(os.path.exists(os.path.join(self.run_dir, "30_undist")))

    def test_all_crashed_is_an_infrastructure_problem(self):
        sc = {"A": {"fail": "extract"}, "B": {"fail": "extract"}, "D": {}}
        rc, sfm = self.go(sc, "--no-c")
        self.assertEqual(rc, 20)

    def test_timeout_marks_variant_and_finishes(self):
        sc = {"A": self.good(), "B": self.good(sleep={"feature_extractor": 30})}
        rc, sfm = self.go(sc, "--variants", "A,B", "--variant-timeout", "6")
        self.assertEqual(rc, 0)
        status = {v["name"]: v["status"] for v in sfm["variants"]}
        self.assertEqual(status["B"], "timeout")
        self.assertEqual(sfm["chosen"], "A")

    def test_sequential_mode(self):
        sc = {"A": self.good(), "B": self.good(reproj=0.5), "D": self.good()}
        rc, sfm = self.go(sc, "--sequential")
        self.assertEqual(rc, 0)
        self.assertEqual(sfm["chosen"], "B")

    def test_loser_databases_removed_chosen_kept(self):
        sc = {"A": self.good(reproj=0.9), "B": self.good(reproj=0.7), "D": self.good(reproj=1.0)}
        rc, sfm = self.go(sc)
        self.assertTrue(os.path.isfile(os.path.join(self.run_dir, "20_sfm", "B", "db.db")))
        self.assertFalse(os.path.isfile(os.path.join(self.run_dir, "20_sfm", "A", "db.db")))
        self.assertTrue(os.path.isfile(os.path.join(self.run_dir, "20_sfm", "A", "log.txt")))


class TestPreflight(StageCase):
    def test_missing_colmap(self):
        rc = stage.main(["--run", self.run_dir, "--cache", self.cache, "--colmap", "/nonexistent/colmap"])
        self.assertEqual(rc, 20)

    def test_too_few_images(self):
        for f in os.listdir(self.images)[:97]:
            os.remove(os.path.join(self.images, f))
        rc, _ = self.go({})
        self.assertEqual(rc, 20)

    def test_missing_cache_file(self):
        os.remove(os.path.join(self.cache, "sift-lightglue.onnx"))
        rc, _ = self.go({})
        self.assertEqual(rc, 20)

    def test_low_disk(self):
        rc = stage.main(["--run", self.run_dir, "--cache", self.cache, "--colmap", self.colmap, "--min-free-gb", "99999999"])
        self.assertEqual(rc, 20)


class TestRanges(unittest.TestCase):
    def test_runs_of(self):
        self.assertEqual(stage.runs_of([1, 2, 3, 7, 9, 10]), [(1, 3), (7, 7), (9, 10)])
        self.assertEqual(stage.runs_of([]), [])


if __name__ == "__main__":
    unittest.main()
