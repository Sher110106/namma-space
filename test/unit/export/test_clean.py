import unittest

from _common import HAVE_NUMPY, make_splats

if HAVE_NUMPY:
    import numpy as np
    from pipeline.export import clean as CL


def base(n=200):
    s = make_splats(n, seed=1, extent=1.0)
    s.opacity[:] = 3.0                # sigmoid 0.95
    s.scale[:] = -4.0                 # 1.8 cm
    return s


@unittest.skipUnless(HAVE_NUMPY, "numpy missing")
class CleanSteps(unittest.TestCase):
    def setUp(self):
        self.pts = np.random.default_rng(3).uniform(-1, 1, size=(500, 3))
        self.cams = np.array([[5.0, 5.0, 5.0]])           # far from everything

    def run_clean(self, s):
        return CL.clean(s, self.pts, self.cams)

    def test_each_step_counts_and_order(self):
        s = base(200)
        s.means[0, 0] = np.nan                                   # nan
        s.opacity[1] = -5.0                                      # opacity < 0.05  (sigmoid 0.0067)
        s.scale[2, 1] = np.log(3.0)                              # huge axis 3 m
        s.means[3] = [50.0, 0, 0]                                # outside box
        s.means[4] = [1.0, 1.0, 1.0]                             # 5 cm from the camera below: bubble
        cams = np.array([[1.0, 1.0, 1.05]])
        out, removed, det = CL.clean(s, self.pts, cams)
        self.assertEqual(removed, {"nan": 1, "opacity": 1, "huge": 1, "outside": 1, "bubble": 1, "freespace": 0})
        self.assertEqual(len(out), 195)
        self.assertEqual(det["n_in"], 200)
        self.assertEqual(det["n_out"], 195)

    def test_huge_needs_both_conditions(self):
        s = base(300)
        s.scale[:, :] = np.log(0.5)                              # all 0.5 m: above p99.9? no: all equal
        out, removed, _ = self.run_clean(s)
        self.assertEqual(removed["huge"], 0)                     # none exceeds 1.0 m
        s.scale[7, 2] = np.log(1.5)                              # 1.5 m and the largest
        out, removed, _ = self.run_clean(s)
        self.assertEqual(removed["huge"], 1)

    def test_clean_nothing_to_remove(self):
        s = base(100)
        out, removed, _ = self.run_clean(s)
        self.assertEqual(sum(removed.values()), 0)
        self.assertEqual(len(out), 100)

    def test_prune_to_keeps_most_important(self):
        s = base(100)
        s.opacity[:] = np.linspace(-3, 3, 100)
        out = CL.prune_to(s, 10, "opacity")
        self.assertEqual(len(out), 10)
        self.assertGreater(out.opacity.min(), s.opacity[89] - 1e-6)
        self.assertIs(CL.prune_to(s, 500), s)

    def test_floor_measurement(self):
        rng = np.random.default_rng(0)
        n = 20000
        s = base(n)
        s.means[:, 0] = rng.uniform(-2, 2, n)
        s.means[:, 2] = rng.uniform(-2, 2, n)
        s.means[:, 1] = np.where(rng.random(n) < 0.7, rng.normal(0.12, 0.01, n), rng.uniform(0.0, 2.5, n))
        s.scale[:] = np.log(0.03)
        s.opacity[:] = 3.0
        cams = np.array([[0, 1.4, 0], [1, 1.5, 1]])
        fy, info = CL.measure_floor_y(s, cams)
        self.assertIsNotNone(fy)
        self.assertAlmostEqual(fy, 0.12, delta=0.03)

    def test_floor_local_ignores_a_peak_under_the_floor(self):
        """vr_room and honka: the lowest-peak rule picked splats under the floor; the local rule must not."""
        rng = np.random.default_rng(1)
        n = 30000
        s = base(n)
        s.means[:, 0] = rng.uniform(-2, 2, n)
        s.means[:, 2] = rng.uniform(-2, 2, n)
        r = rng.random(n)
        s.means[:, 1] = np.where(r < 0.5, rng.normal(0.03, 0.01, n),                 # the floor, 3 cm above y = 0
                                 np.where(r < 0.7, rng.normal(-0.30, 0.01, n),       # junk under the floor
                                          rng.uniform(0.2, 2.5, n)))
        s.scale[:] = np.log(0.03)
        s.opacity[:] = 3.0
        cams = np.array([[0, 1.4, 0], [1, 1.5, 1]])
        fy_peak, _ = CL.measure_floor_y(s, cams)
        fy, info = CL.measure_floor_local(s, cams)
        self.assertAlmostEqual(fy_peak, -0.30, delta=0.03)      # the old rule is fooled
        self.assertAlmostEqual(fy, 0.03, delta=0.01)


if __name__ == "__main__":
    unittest.main()
