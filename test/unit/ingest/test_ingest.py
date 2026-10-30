"""Acceptance tests B1-B7 (FINAL_PLAN WP2) for pipeline.ingest.

Run where numpy, OpenCV, Pillow and ffmpeg exist (the container, or a venv):

    python -m unittest discover -s test/unit/ingest -p 'test_ingest.py' -v

Fixtures: NS_FIXTURES, else /w/data/fixtures/videos (tyrone container), else /tmp/namma-w2/videos.
Build them with  python -m pipeline.ingest.make_fixtures --out DIR  (container).
A test whose fixture is missing is skipped, not passed.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import _common as C  # noqa: E402

try:
    import cv2
    import numpy as np
    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False

CACHE: dict = {}
HONKA = [os.environ.get("NS_HONKA", ""), "/w/data/mushroom/room_datasets/honka/iphone/long_capture/images",
         "/tmp/namma-w2/honka"]


def need(name: str):
    if not HAVE:
        raise unittest.SkipTest("numpy/OpenCV missing")
    if C.video(name) is None:
        raise unittest.SkipTest(f"fixture {name}.mp4 not found")


def video_thumbs(path):
    """The video's own 8 fps stream as 64x36 grey thumbnails, decoded by ffmpeg in one pass."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "fps=8,scale=64:36:flags=area:out_range=full,format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 36, 64).astype(float)


def luma_stats(run: str):
    means, rms = [], []
    for f in sorted(glob.glob(os.path.join(run, "10_frames", "images", "*.jpg"))):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        means.append(float(g.mean()))
        rms.append(float(g.std()))
    return float(np.mean(means)), float(np.mean(rms))


def tearDownModule():
    import shutil
    for key, val in CACHE.items():
        shutil.rmtree(val[1], ignore_errors=True)


class TestIngest(unittest.TestCase):
    def test_B1_sdr_1080p(self):
        need("b1")
        rc, run, sec, out = C.run_ingest("b1", cache=CACHE)
        self.assertEqual(rc, 0, out[-600:])
        fj = C.frames_json(run)
        self.assertTrue(100 <= fj["n_kept"] <= 130, fj["n_kept"])
        imgs = sorted(glob.glob(os.path.join(run, "10_frames", "images", "*.jpg")))
        self.assertEqual(len(imgs), fj["n_kept"])
        self.assertEqual([os.path.basename(p) for p in imgs[:3]], ["000001.jpg", "000002.jpg", "000003.jpg"])
        for p in imgs:
            h, w = cv2.imread(p).shape[:2]
            self.assertEqual((w, h), (1920, 1080), p)
        t = [k["t"] for k in fj["kept"]]
        self.assertEqual(t, sorted(t))
        self.assertEqual(len(set(t)), len(t))
        self.assertEqual(fj["hdr"], "none")
        # capture order and timestamps: each kept frame matches the video's own 8 fps stream at index round(t * 8)
        T = video_thumbs(C.video("b1"))
        at_t, offs = [], []
        for k, rec in enumerate(fj["kept"]):
            i = int(round(rec["t"] * 8))
            b = cv2.resize(cv2.imread(imgs[k], 0), (64, 36), interpolation=cv2.INTER_AREA).astype(float)
            lo = max(0, i - 3)
            d = [float(np.abs(b - T[j]).mean()) for j in range(lo, min(len(T), i + 4))]
            at_t.append(d[i - lo])
            offs.append(int(np.argmin(d)) + lo - i)
        # the picture at index round(t*8) is the kept frame (frames held for a few samples can tie with a neighbour)
        self.assertLess(max(at_t), 25.0, f"worst match to its own time {max(at_t):.1f}")
        self.assertLess(float(np.median(at_t)), 6.0)
        self.assertLessEqual(max(abs(o) for o in offs), 3)

    def test_B1b_colours_are_not_shifted(self):
        """JPEG has no matrix tag; the ingest converts to BT.601 so viewers see the video's colours."""
        need("b1")
        rc, run, sec, out = C.run_ingest("b1", cache=CACHE)
        fj = C.frames_json(run)
        k = len(fj["kept"]) // 3
        cmd = ["ffmpeg", "-v", "error", "-i", str(C.video("b1")), "-vf", "fps=8,scale=64:36:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        V = np.frombuffer(raw, np.uint8).reshape(-1, 36, 64, 3).astype(float)
        i = int(round(fj["kept"][k]["t"] * 8))
        got = cv2.imread(os.path.join(run, "10_frames", "images", fj["kept"][k]["name"]))[:, :, ::-1].astype(float)
        got = cv2.resize(got, (64, 36), interpolation=cv2.INTER_AREA)
        d = np.abs(V[i].mean((0, 1)) - got.mean((0, 1)))
        self.assertLess(float(d.max()), 4.0, d)

    def test_B2_hlg_tonemapped(self):
        need("b2")
        need("b1")
        rc, run, sec, out = C.run_ingest("b2", cache=CACHE)
        self.assertEqual(rc, 0, out[-800:])
        fj = C.frames_json(run)
        self.assertEqual(fj["hdr"], "hlg")
        self.assertEqual(fj["tonemap"], "zscale")
        _, run1, _, _ = C.run_ingest("b1", cache=CACHE)
        m2, c2 = luma_stats(run)
        m1, c1 = luma_stats(run1)
        self.assertTrue(60 <= m2 <= 190, m2)
        self.assertGreaterEqual(c2, 0.8 * c1, (c2, c1))

    def test_B2b_numpy_fallback(self):
        need("b2")
        rc, run, sec, out = C.run_ingest("b2", ["--tonemap", "numpy"], cache=CACHE)
        self.assertEqual(rc, 0, out[-800:])
        fj = C.frames_json(run)
        self.assertEqual(fj["tonemap"], "numpy")
        _, runz, _, _ = C.run_ingest("b2", cache=CACHE)
        mn, cn = luma_stats(run)
        mz, cz = luma_stats(runz)
        self.assertTrue(60 <= mn <= 190, mn)
        self.assertLess(abs(mn - mz), 25.0, (mn, mz))

    def test_B3_portrait_upright(self):
        need("b3")
        rc, run, sec, out = C.run_ingest("b3", cache=CACHE)
        self.assertEqual(rc, 0, out[-600:])
        fj = C.frames_json(run)
        self.assertEqual(fj["size"], [1080, 1920])
        src = next((h for h in HONKA if h and glob.glob(os.path.join(h, "frame_*.jpg"))), None)
        imgs = sorted(glob.glob(os.path.join(run, "10_frames", "images", "*.jpg")))
        for p in imgs:
            h, w = cv2.imread(p).shape[:2]
            self.assertEqual((w, h), (1080, 1920))
        if src is None:
            self.skipTest("honka source frames not found; size checked, orientation not")

        def feat(g, size):
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            g -= g.mean()
            return (g / (np.linalg.norm(g) + 1e-9)).ravel()
        S = np.array([feat(cv2.imread(f, 0)[:, (738 - 559) // 2:(738 - 559) // 2 + 559], (54, 96))
                      for f in sorted(glob.glob(os.path.join(src, "frame_*.jpg")))])
        score = {0: [], 180: []}
        for p in imgs[::6]:
            g = cv2.imread(p, 0)
            score[0].append(float((S @ feat(g, (54, 96))).max()))
            score[180].append(float((S @ feat(cv2.rotate(g, cv2.ROTATE_180), (54, 96))).max()))
        self.assertGreater(np.mean(score[0]), 0.8)
        self.assertGreater(np.mean(score[0]), np.mean(score[180]) + 0.2)

    def test_B4_blur_segments(self):
        need("b4")
        rc, run, sec, out = C.run_ingest("b4", cache=CACHE)
        self.assertEqual(rc, 0, out[-600:])
        fj = C.frames_json(run)
        segs = C.sidecar("b4")["blurred_segments_s"]
        kept_win = {int(k["t"] // 0.5) for k in fj["kept"]}
        n_win = int(C.sidecar("b4")["duration_s"] // 0.5)
        blurred = set()
        for a, b in segs:
            blurred |= set(range(int(a / 0.5), int(b / 0.5)))
        sharp = set(range(n_win)) - blurred
        no_frame = sum(1 for w in blurred if w not in kept_win) / len(blurred)
        keep_sharp = sum(1 for w in sharp if w in kept_win) / len(sharp)
        print(f"B4: blurred windows with no frame {no_frame:.0%} (need >= 75%), sharp windows kept {keep_sharp:.0%} (need >= 90%)")
        self.assertGreaterEqual(no_frame, 0.75)
        self.assertGreaterEqual(keep_sharp, 0.90)

    def test_B5_too_short_exits_10(self):
        need("b5")
        rc, run, sec, out = C.run_ingest("b5", cache=CACHE)
        self.assertEqual(rc, 10, out[-400:])
        self.assertIn("too short", out)
        self.assertFalse(os.path.exists(os.path.join(run, "10_frames")))

    def test_B6_vfr_in_order(self):
        need("b6")
        rc, run, sec, out = C.run_ingest("b6", cache=CACHE)
        self.assertEqual(rc, 0, out[-600:])
        fj = C.frames_json(run)
        t = [k["t"] for k in fj["kept"]]
        self.assertEqual(t, sorted(t))
        self.assertGreaterEqual(fj["n_kept"], 90)
        self.assertTrue(fj["inputs"][0]["vfr"])

    def test_B7_4k_hevc_in_six_minutes(self):
        need("b7")
        rc, run, sec, out = C.run_ingest("b7", cache=CACHE)
        self.assertEqual(rc, 0, out[-600:])
        fj = C.frames_json(run)
        print(f"B7: {sec:.0f} s wall, {fj['n_kept']} frames of {fj['n_cand']} candidates")
        self.assertLessEqual(sec, 360.0)
        self.assertLessEqual(fj["n_kept"], 450)
        self.assertGreaterEqual(fj["n_kept"], 120)
        self.assertEqual(fj["size"], [1920, 1080])


class TestFramesMode(unittest.TestCase):
    def test_exif_applied_and_stripped(self):
        if not HAVE:
            self.skipTest("no numpy/OpenCV")
        from PIL import Image
        d = tempfile.mkdtemp(prefix="ns_frames_")
        src = os.path.join(d, "in")
        os.makedirs(src)
        # a 200x100 landscape picture stored with EXIF orientation 6 (rotate 90 cw to display): shows as 100x200
        for i in range(5):
            im = Image.new("RGB", (200, 100), (30 * i, 80, 120))
            im.paste((255, 255, 255), (0, 0, 40, 100))      # white band on the stored left edge
            ex = im.getexif()
            ex[0x0112] = 6
            im.save(os.path.join(src, f"IMG_{i + 1}.jpg"), exif=ex.tobytes())
        run = os.path.join(d, "run")
        r = subprocess.run([sys.executable, "-m", "pipeline.ingest", "--run", run, "--frames", src, "--min-free-gb", "0.1", "--force"],
                           cwd=str(C.REPO), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-500:] + r.stderr[-500:])
        out = Image.open(os.path.join(run, "10_frames", "images", "000001.jpg"))
        self.assertEqual(out.size, (100, 200))                       # rotated upright
        self.assertNotIn(0x0112, out.getexif())                      # EXIF gone
        a = np.asarray(out.convert("L"))
        # orientation 6 rotates the stored picture 90 deg clockwise: the left band moves to the top
        self.assertGreater(a[:30].mean(), a[-30:].mean() + 50)
        fj = C.frames_json(run)
        self.assertEqual(fj["src"], "frames")
        self.assertEqual(fj["rotation"]["exif_rotated"], 5)
        self.assertEqual([k["name"] for k in fj["kept"]][:2], ["000001.jpg", "000002.jpg"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
