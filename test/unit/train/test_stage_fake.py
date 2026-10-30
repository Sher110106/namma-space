"""Run the train stage end to end against a fake gsplat trainer (no GPU, no gsplat)."""
import json
import os
import shutil
import struct
import sys
import tempfile
import textwrap
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

FAKE = textwrap.dedent('''
    import json, os, struct, sys, time
    a = sys.argv
    def val(flag, n=1):
        i = a.index(flag); return a[i + 1:i + 1 + n]
    out = val("--result-dir")[0]
    quick = val("--max-steps")[0] == "7000"
    psnr = {"r0": 21.0, "r1": 21.5}["r1" if "--depth-loss" in a else "r0"]
    if os.environ.get("FAKE_FAIL_R1") and "--depth-loss" in a:
        sys.exit(3)
    for sub in ("stats", "ply", "ckpts", "renders", "tb"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    def ply(path, n):
        hdr = "ply\\nformat binary_little_endian 1.0\\nelement vertex %d\\nproperty float x\\nproperty float y\\nproperty float z\\nend_header\\n" % n
        open(path, "wb").write(hdr.encode() + struct.pack("<3f", 0, 0, 0) * n)
    hang = os.environ.get("FAKE_HANG")      # "ignore": ignore SIGTERM; "silent": stop writing output
    for step in ([6999] if quick else [6999, 29999]):
        if hang and step == 29999:
            import signal
            if hang == "ignore":
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                while True:
                    print("loss=0.1| : 70%%| 21000/30000 [00:01<00:01, 9.0it/s]", flush=True)
                    time.sleep(0.05)
            while True:
                time.sleep(1)
        print("loss=0.1| : 50%%| %d/30000 [00:01<00:01, 9.0it/s]" % step, flush=True)
        time.sleep(0.15)
        n = 1000 + step % 7
        ply(os.path.join(out, "ply", "point_cloud_%d.ply" % step), n)
        json.dump({"mem": 3.25, "ellipse_time": 12.5, "num_GS": n}, open(os.path.join(out, "stats", "train_step%d_rank0.json" % step), "w"))
        json.dump({"psnr": psnr, "ssim": 0.8, "lpips": 0.2, "cc_psnr": psnr + 3, "num_GS": n}, open(os.path.join(out, "stats", "val_step%d.json" % step), "w"))
        open(os.path.join(out, "ckpts", "ckpt_%d_rank0.pt" % step), "wb").write(b"x")
''')


def make_data(d):
    os.makedirs(os.path.join(d, "sparse", "0"))
    os.makedirs(os.path.join(d, "images"))
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        open(os.path.join(d, "sparse", "0", f), "wb").write(b"\0" * 8)


@unittest.skipUnless(HAVE_NUMPY, "numpy missing (use pipeline/export/.venv/bin/python)")
class FakeStage(unittest.TestCase):
    def setUp(self):
        from pipeline.train import __main__ as M, cmd as C, gpu as G
        self.M, self.C, self.G = M, C, G
        self.tmp = tempfile.mkdtemp()
        self.fake = os.path.join(self.tmp, "fake_trainer.py")
        open(self.fake, "w").write(FAKE)
        self.old = (C.TRAINER, C.TRAINER_CWD, M.POLL_S, G.gpu_free_mb, G.try_extra_slot, M.G.gpu_used_mb, M.free_gb, C.PYTHON)
        M.free_gb = lambda path: 100.0
        C.PYTHON = sys.executable
        C.TRAINER, C.TRAINER_CWD, M.POLL_S, M.PROGRESS_S = self.fake, self.tmp, 0.05, 0.1
        G.gpu_free_mb = lambda: 40000
        G.gpu_used_mb = lambda: 8000
        G.try_extra_slot = lambda lock_dir=None: None
        self.old_grace = (M.KILL_GRACE_S, M.ABANDON_S)
        M.KILL_GRACE_S, M.ABANDON_S = 0.5, 0.5
        self.run_dir = os.path.join(self.tmp, "run")
        make_data(os.path.join(self.run_dir, "40_align"))

    def tearDown(self):
        C, M, G = self.C, self.M, self.G
        C.TRAINER, C.TRAINER_CWD, M.POLL_S, G.gpu_free_mb, G.try_extra_slot, G.gpu_used_mb, M.free_gb, C.PYTHON = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("FAKE_FAIL_R1", None)
        os.environ.pop("FAKE_HANG", None)
        self.M.KILL_GRACE_S, self.M.ABANDON_S = self.old_grace

    def stage(self, *extra):
        # the fake needs the real colmap model for the scene unit: skip it
        return self.M.main(["--run", self.run_dir, "--no-scene-unit", "--parallel", "no", *extra])

    def test_full_run_chooses_r1_and_cleans_up(self):
        rc = self.stage("--variants", "r0,r1")
        self.assertEqual(rc, 0)
        out = os.path.join(self.run_dir, "50_train")
        self.assertFalse(os.path.exists(out + ".tmp"))
        tj = json.load(open(os.path.join(out, "train.json")))
        self.assertEqual(tj["chosen"], "r1")
        self.assertEqual([v["name"] for v in tj["variants"]], ["r0", "r1"])
        r1 = tj["variants"][1]
        self.assertEqual((r1["psnr"], r1["n_gs"], r1["time_s"], r1["peak_gpu_gb"]), (21.5, 1000 + 29999 % 7, 12.5, 3.25))
        self.assertEqual(tj["chosen_ply"], "50_train/r1/ply/point_cloud_29999.ply")
        self.assertTrue(os.path.isfile(os.path.join(self.run_dir, tj["chosen_ply"])))
        self.assertIn("why", tj)
        # losing variant: checkpoints and PLYs deleted, stats kept
        self.assertFalse(os.path.exists(os.path.join(out, "r0", "ckpts")))
        self.assertFalse(os.path.exists(os.path.join(out, "r0", "ply")))
        self.assertNotIn("final_ply", tj["variants"][0])
        self.assertEqual(tj["variants"][0]["final_ply_deleted"], "50_train/r0/ply/point_cloud_29999.ply")
        self.assertTrue(os.path.isfile(os.path.join(out, "r0", "stats", "val_step29999.json")))
        # winner keeps the final checkpoint, drops the 7k one
        self.assertTrue(os.path.isfile(os.path.join(out, "r1", "ckpts", "ckpt_29999_rank0.pt")))
        self.assertFalse(os.path.exists(os.path.join(out, "r1", "ckpts", "ckpt_6999_rank0.pt")))
        self.assertIsNotNone(r1["t_preview_s"])
        # strict JSON
        json.loads(open(os.path.join(out, "train.json")).read(), parse_constant=lambda c: self.fail(c))

    def test_default_is_r0_only_with_150_min_limit(self):
        # review fix #4/#10: one variant by default, 150 min per variant, frames kept up to 1920 px
        rc = self.stage()
        self.assertEqual(rc, 0)
        tj = json.load(open(os.path.join(self.run_dir, "50_train", "train.json")))
        self.assertEqual([v["name"] for v in tj["variants"]], ["r0"])
        self.assertEqual(tj["max_minutes"], 150.0)
        self.assertEqual(tj["long_side"], self.M.TRAIN_LONG_SIDE)
        self.assertIn("sequential: one variant", tj["parallel_reason"])

    def test_done_stage_is_not_redone(self):
        self.assertEqual(self.stage(), 0)
        mtime = os.path.getmtime(os.path.join(self.run_dir, "50_train", "train.json"))
        self.assertEqual(self.stage(), 0)
        self.assertEqual(mtime, os.path.getmtime(os.path.join(self.run_dir, "50_train", "train.json")))

    def test_failed_variant_falls_back_to_other(self):
        os.environ["FAKE_FAIL_R1"] = "1"
        rc = self.stage("--variants", "r0,r1")
        tj = json.load(open(os.path.join(self.run_dir, "50_train", "train.json")))
        self.assertEqual(tj["chosen"], "r0")
        r1 = tj["variants"][1]
        self.assertEqual(r1["status"], "failed")
        self.assertEqual(r1["rc"], 3)
        self.assertEqual(rc, 0)

    def test_quick_uses_7k_ply(self):
        rc = self.stage("--quick", "--variants", "r0")
        tj = json.load(open(os.path.join(self.run_dir, "50_train", "train.json")))
        self.assertEqual(tj["chosen_ply"], "50_train/r0/ply/point_cloud_6999.ply")
        self.assertTrue(tj["quick"])
        self.assertEqual(rc, 0)

    def test_wall_limit_kills_a_trainer_that_ignores_sigterm(self):
        # the 5-hour loop of 30 Oct: SIGTERM ignored, the stage printed "terminating" forever
        os.environ["FAKE_HANG"] = "ignore"
        t0 = time.time()
        rc = self.stage("--variants", "r0", "--max-minutes", "0.03")
        self.assertLess(time.time() - t0, 20)
        tj = json.load(open(os.path.join(self.run_dir, "50_train", "train.json")))
        r0 = tj["variants"][0]
        self.assertEqual(r0["status"], "timeout")
        self.assertEqual(r0["end_reason"], "timeout")
        self.assertTrue(r0["partial"])
        # the 7k model is the last rung of the ladder: shipped, flagged, exit 30
        self.assertEqual(tj["chosen"], "r0")
        self.assertEqual(tj["chosen_ply"], "50_train/r0/ply/point_cloud_6999.ply")
        self.assertTrue(os.path.isfile(os.path.join(self.run_dir, tj["chosen_ply"])))
        self.assertTrue(any("timeout" in w for w in tj["warnings"]))
        self.assertEqual(rc, 30)
        # no r0nb retry after a timeout
        self.assertEqual([v["name"] for v in tj["variants"]], ["r0"])

    def test_stall_watchdog_stops_a_silent_trainer(self):
        os.environ["FAKE_HANG"] = "silent"
        t0 = time.time()
        rc = self.stage("--variants", "r0", "--stall-minutes", "0.03")
        self.assertLess(time.time() - t0, 20)
        tj = json.load(open(os.path.join(self.run_dir, "50_train", "train.json")))
        self.assertEqual(tj["variants"][0]["status"], "stalled")
        self.assertEqual(tj["chosen"], "r0")
        self.assertEqual(rc, 30)

    def test_missing_model_is_infrastructure_error(self):
        shutil.rmtree(os.path.join(self.run_dir, "40_align", "sparse"))
        self.assertEqual(self.stage(), 20)

    def test_parallel_decision(self):
        G = self.G
        G.gpu_free_mb = lambda: 40000
        par, lk, why = G.decide_parallel("force", 2, self.tmp)
        self.assertTrue(par)
        G.gpu_free_mb = lambda: 10000
        par, lk, why = G.decide_parallel("force", 2, self.tmp)
        self.assertFalse(par)
        self.assertIn("10000", why)
        G.gpu_free_mb = lambda: 40000
        par, lk, why = G.decide_parallel("auto", 2, self.tmp)       # no second slot (try_extra_slot -> None)
        self.assertFalse(par)
        self.assertIn("no second GPU slot", why)
        par, lk, why = G.decide_parallel("auto", 1, self.tmp)
        self.assertFalse(par)


try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


@unittest.skipUnless(HAVE_PIL, "PIL missing (runs in namma-space-gpu: ./ns test train --remote)")
class ResizeTest(unittest.TestCase):
    def test_frames_above_the_long_side_are_resized(self):
        from PIL import Image
        from pipeline.train import __main__ as M
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "40_align")
            make_data(src)
            for i in range(3):
                Image.new("RGB", (1920, 1080), (i * 40, 90, 120)).save(os.path.join(src, "images", f"{i:06d}.jpg"))
            out, size = M.resized_data_dir(src, os.path.join(d, "tmp"), 1600)
            self.assertEqual(size, (1600, 900))
            self.assertEqual(M.image_size(os.path.join(out, "images")), (1600, 900))
            self.assertEqual(len(os.listdir(os.path.join(out, "images"))), 3)
            self.assertTrue(os.path.isfile(os.path.join(out, "sparse", "0", "cameras.bin")))
            # at or below the long side: used as is
            same, size2 = M.resized_data_dir(src, os.path.join(d, "tmp2"), 1920)
            self.assertEqual((same, size2), (src, (1920, 1080)))


class SlotLockTest(unittest.TestCase):
    def test_second_slot_lock(self):
        from pipeline.train import gpu as G
        with tempfile.TemporaryDirectory() as d:
            held = G.SlotLock(os.path.join(d, "gpu.0"))
            self.assertTrue(held.try_acquire())
            lk = G.try_extra_slot(d)           # gpu.0 is held, gpu.1 is free
            self.assertIsNotNone(lk)
            self.assertTrue(lk.path.endswith("gpu.1"))
            self.assertIsNone(G.try_extra_slot(d))
            lk.release()
            held.release()
            self.assertIsNotNone(G.try_extra_slot(d))


if __name__ == "__main__":
    unittest.main()
