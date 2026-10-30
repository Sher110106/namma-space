import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pipeline.train import stats as S  # noqa: E402


def rec(name, psnr, lpips, **kw):
    r = {"name": name, "psnr": psnr, "lpips": lpips, "status": "ok"}
    r.update(kw)
    return r


class Choose(unittest.TestCase):
    def test_clear_psnr_winner(self):
        c, why = S.choose([rec("r0", 21.0, 0.30), rec("r1", 21.5, 0.31)])
        self.assertEqual(c, "r1")
        c, _ = S.choose([rec("r0", 22.0, 0.30), rec("r1", 21.5, 0.25)])
        self.assertEqual(c, "r0")

    def test_psnr_tie_lpips_decides(self):
        c, why = S.choose([rec("r0", 21.00, 0.250), rec("r1", 21.05, 0.300)])
        self.assertEqual(c, "r0")
        self.assertIn("LPIPS", why)

    def test_full_tie_takes_r1(self):
        c, why = S.choose([rec("r0", 21.00, 0.250), rec("r1", 20.98, 0.252)])
        self.assertEqual(c, "r1")
        self.assertIn("tie", why)

    def test_failed_variant_is_skipped(self):
        c, why = S.choose([rec("r0", 21.0, 0.3), rec("r1", None, None, status="failed")])
        self.assertEqual(c, "r0")
        self.assertIn("r1", why)

    def test_nan_variant_is_skipped(self):
        c, _ = S.choose([rec("r0", 25.0, 0.2, nan_dropped=12), rec("r1", 21.0, 0.3)])
        self.assertEqual(c, "r1")

    def test_none_usable(self):
        c, why = S.choose([rec("r0", None, None, status="failed")])
        self.assertIsNone(c)


class Stats(unittest.TestCase):
    def test_variant_record(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "stats"))
            json.dump({"psnr": 20.6, "ssim": 0.8, "lpips": 0.24, "cc_psnr": 24.0, "num_GS": 1000},
                      open(os.path.join(d, "stats", "val_step29999.json"), "w"))
            json.dump({"mem": 3.5, "ellipse_time": 1100.0, "num_GS": 1000},
                      open(os.path.join(d, "stats", "train_step29999_rank0.json"), "w"))
            r = S.variant_record("r0", d, [7000, 30000], {"status": "ok"})
            self.assertEqual((r["psnr"], r["n_gs"], r["time_s"], r["peak_gpu_gb"]), (20.6, 1000, 1100.0, 3.5))
            self.assertEqual(r["cc_psnr"], 24.0)

    def test_progress_from_log(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("loss=0.1| sh degree=0| :   1%|  | 300/30000 [00:10<20:00, 40.0it/s]\r"
                    "loss=0.1| sh degree=0| :   2%|  | 450/30000 [00:12<20:00, 40.0it/s]")
        try:
            self.assertEqual(S.progress_from_log(f.name), (450, 30000))
        finally:
            os.remove(f.name)
        self.assertIsNone(S.progress_from_log("/nonexistent"))


if __name__ == "__main__":
    unittest.main()
