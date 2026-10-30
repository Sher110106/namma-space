import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pipeline.train import cmd as C  # noqa: E402


def flag_value(cmd, flag, n=1):
    i = cmd.index(flag)
    return cmd[i + 1:i + 1 + n]


class BuildCommand(unittest.TestCase):
    def test_r0_matches_plan(self):
        c = C.build_command(C.TrainOpts("/d", "/r"))
        self.assertEqual(c[:4], ["python", C.TRAINER, "mcmc", "--data-dir"])
        self.assertIn("--no-normalize-world-space", c)
        self.assertNotIn("--normalize-world-space", c)
        self.assertIn("--antialiased", c)
        self.assertIn("--use-bilateral-grid", c)
        self.assertNotIn("--depth-loss", c)
        self.assertIn("--disable-viewer", c)
        self.assertEqual(flag_value(c, "--strategy.cap-max"), ["1500000"])
        self.assertEqual(flag_value(c, "--max-steps"), ["30000"])
        self.assertEqual(flag_value(c, "--eval-steps", 2), ["7000", "30000"])
        self.assertEqual(flag_value(c, "--ply-steps", 2), ["7000", "30000"])
        self.assertEqual(flag_value(c, "--test-every"), ["8"])
        self.assertIn("--save-ply", c)
        self.assertNotIn("--pose-opt", c)
        self.assertNotIn("--app-opt", c)

    def test_r1_adds_depth_loss_only(self):
        r0 = C.build_command(C.TrainOpts("/d", "/r0", variant="r0"))
        r1 = C.build_command(C.TrainOpts("/d", "/r1", variant="r1"))
        self.assertIn("--depth-loss", r1)
        self.assertEqual(sorted(set(r1) - set(r0)), ["--depth-loss", "/r1"])

    def test_no_bilateral_fallback(self):
        c = C.build_command(C.TrainOpts("/d", "/r", variant="r0nb"))
        self.assertNotIn("--use-bilateral-grid", c)

    def test_quick_is_7k(self):
        o = C.TrainOpts("/d", "/r", quick=True)
        c = C.build_command(o)
        self.assertEqual(flag_value(c, "--max-steps"), ["7000"])
        self.assertEqual(flag_value(c, "--eval-steps"), ["7000"])
        self.assertEqual(o.final_ply_name, "point_cloud_6999.ply")
        self.assertEqual(C.TrainOpts("/d", "/r").final_ply_name, "point_cloud_29999.ply")

    def test_normalize_flag_for_smoke_tests(self):
        c = C.build_command(C.TrainOpts("/d", "/r", normalize=True))
        self.assertIn("--normalize-world-space", c)

    def test_scene_unit_rescales_noise_and_scale_reg(self):
        c = C.build_command(C.TrainOpts("/d", "/r", scene_k=2.0))
        self.assertAlmostEqual(float(flag_value(c, "--strategy.noise-lr")[0]), 5e5 / 4)
        self.assertAlmostEqual(float(flag_value(c, "--scale-reg")[0]), 0.005)
        c = C.build_command(C.TrainOpts("/d", "/r"))
        self.assertNotIn("--strategy.noise-lr", c)

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            C.build_command(C.TrainOpts("/d", "/r", variant="r9"))


class SceneUnit(unittest.TestCase):
    def test_median_distance(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy missing")
        pts = [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0]]
        self.assertAlmostEqual(C.scene_unit(pts), 2.0)


if __name__ == "__main__":
    unittest.main()
