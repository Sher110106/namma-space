import os
import tempfile
import unittest

from _common import HAVE_NUMPY, make_splats

if HAVE_NUMPY:
    import numpy as np
    from pipeline.export import ply_io, splat_writer


@unittest.skipUnless(HAVE_NUMPY, "numpy missing")
class PlyRoundTrip(unittest.TestCase):
    def test_roundtrip_exact(self):
        s = make_splats(500)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.ply")
            ply_io.write_ply(p, s)
            r = ply_io.read_ply(p)
            self.assertEqual(ply_io.count_vertices(p), 500)
        for f in ("means", "f_dc", "f_rest", "opacity", "scale", "quat"):
            np.testing.assert_array_equal(getattr(s, f), getattr(r, f))
        self.assertEqual(r.sh_degree, 3)

    def test_sh_limit(self):
        s = make_splats(10)
        self.assertEqual(s.limited_to_degree(1).f_rest.shape, (10, 9))
        self.assertEqual(s.limited_to_degree(0).f_rest.shape, (10, 0))
        self.assertEqual(s.limited_to_degree(1).sh_degree, 1)
        # channel-major: first 3 coefficients of each channel
        np.testing.assert_array_equal(s.limited_to_degree(1).f_rest[:, 3:6], s.f_rest[:, 15:18])

    def test_bad_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.ply")
            open(p, "wb").write(b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n")
            with self.assertRaises(ValueError):
                ply_io.read_header(p)


@unittest.skipUnless(HAVE_NUMPY, "numpy missing")
class SplatFormat(unittest.TestCase):
    def test_32_bytes_per_splat_and_roundtrip(self):
        s = make_splats(2000, sh_rest=0)
        s.opacity = np.clip(s.opacity, -4, 4)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.splat")
            n = splat_writer.write_splat(p, s)
            self.assertEqual(n, 2000)
            self.assertEqual(os.path.getsize(p), 2000 * 32)
            r = splat_writer.read_splat(p)
        self.assertEqual(len(r), 2000)
        # sorted by size*opacity, so compare as sets of positions
        a = s.means[np.lexsort(s.means.T)]
        b = r.means[np.lexsort(r.means.T)]
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_colour_opacity_quantisation_error_small(self):
        s = make_splats(500, sh_rest=0)
        s.f_dc = np.clip(s.f_dc, -1.5, 1.5)
        s.opacity = np.clip(s.opacity, -3, 3)
        rec = splat_writer.pack_splat(s, sort=False)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.splat")
            rec.tofile(p)
            r = splat_writer.read_splat(p)
        rgb0 = np.clip(0.5 + splat_writer.SH_C0 * s.f_dc, 0, 1)
        rgb1 = 0.5 + splat_writer.SH_C0 * r.f_dc
        self.assertLess(np.abs(rgb0 - rgb1).max(), 1.0 / 255)
        a0, a1 = ply_io.sigmoid(s.opacity), ply_io.sigmoid(r.opacity)
        self.assertLess(np.abs(a0 - a1).max(), 1.0 / 255 + 1e-3)
        # rotation: same rotation up to sign and 8-bit quantisation
        dots = np.abs((s.quat / np.linalg.norm(s.quat, axis=1, keepdims=True) * r.quat).sum(1))
        self.assertGreater(dots.min(), 0.999)


if __name__ == "__main__":
    unittest.main()
