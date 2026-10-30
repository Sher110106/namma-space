"""Test of root's thin align adapter on a synthetic room (also exercises colmap_io + conventions together).

Run: python3 test/e2e/test_align_thin.py
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "e2e"))

from pipeline.common import colmap_io as cio  # noqa: E402
from pipeline.common import conventions as cv  # noqa: E402
from test_colmap_io import rot_axis  # noqa: E402

ADAPTER = ROOT / "test" / "e2e" / "adapters" / "align_thin.py"


def room_model(cam_height=1.4, n_cams=40, seed=5) -> cio.Model:
    """Cameras on a 3 m circle looking at the centre, level, in a Y-up metre world. Floor points at y=0."""
    rnd = random.Random(seed)
    m = cio.Model(cameras={1: cio.Camera(1, "PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0])})
    for k in range(n_cams):
        th = 2 * math.pi * k / n_cams
        pos = [3 * math.cos(th), cam_height + rnd.uniform(-0.05, 0.05), 3 * math.sin(th)]
        yaw = math.degrees(math.atan2(-(-pos[0]), -(-pos[2])))  # face the origin (forward = -pos)
        yaw = math.degrees(math.atan2(pos[0], pos[2]))
        q = cv.quat_from_yaw_pitch_deg(yaw, 0.0)
        R, t = cv.colmap_from_threejs(pos, q)
        m.images[k + 1] = cio.Image(k + 1, cio.rotmat_to_qvec(R), tuple(t), 1, f"{k + 1:06d}.jpg")
    pid = 0
    for _ in range(600):
        pid += 1
        h = 0.0 if rnd.random() < 0.7 else rnd.uniform(0.2, 2.5)
        m.points3D[pid] = cio.Point3D(pid, (rnd.uniform(-2.5, 2.5), h, rnd.uniform(-2.5, 2.5)), (9, 9, 9), 0.5)
    return m


class AlignThin(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rhalign"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_adapter(self, model, height_cm=None):
        run = self.tmp / "run"
        (run / "input").mkdir(parents=True)
        (run / "input" / "meta.json").write_text(json.dumps({"height_cm": height_cm} if height_cm else {}))
        (run / "30_undist" / "images").mkdir(parents=True)
        cio.write_model(model, run / "30_undist" / "sparse")
        r = subprocess.run([sys.executable, str(ADAPTER), "--run", str(run)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return run

    def test_recovers_up_floor_scale_and_origin(self):
        world = room_model()
        # hide the metric world behind an arbitrary similarity, as SfM would
        R0 = rot_axis((0.3, -0.7, 0.5), 2.1)
        scrambled = cio.similarity_transform(world, 0.37, R0, [4.0, -2.0, 9.0])
        run = self.run_adapter(scrambled)
        out = cio.read_model(run / "40_align" / "sparse" / "0")
        aj = json.loads((run / "40_align" / "align.json").read_text())
        self.assertTrue(aj["thin"])
        self.assertEqual(len(out.images), 40)
        # level cameras: every camera's y axis points to world -Y (down)
        for im in out.images.values():
            R = im.rotmat()
            ang = math.degrees(math.acos(max(-1, min(1, -R[1][1]))))  # angle between cam y and -Y
            self.assertLess(ang, 1.0)
        heights = sorted(c[1] for c in cio.camera_centres(out).values())
        self.assertAlmostEqual(heights[len(heights) // 2], 1.4, delta=0.05)
        floor = sorted(p.xyz[1] for p in out.points3D.values())[int(0.02 * len(out.points3D))]
        self.assertAlmostEqual(floor, 0.0, delta=0.03)
        cx = sum(c[0] for c in cio.camera_centres(out).values()) / 40
        cz = sum(c[2] for c in cio.camera_centres(out).values()) / 40
        self.assertAlmostEqual(cx, 0.0, places=6)
        self.assertAlmostEqual(cz, 0.0, places=6)
        # the scale recovers the metre scale of the original world (3 m circle)
        radius = math.sqrt(sum(v * v for v in (cio.camera_centres(out)[1][0], cio.camera_centres(out)[1][2])))
        self.assertAlmostEqual(radius, 3.0, delta=0.15)
        self.assertEqual(aj["checks"]["registered_before"], aj["checks"]["registered_after"])
        self.assertTrue((run / "40_align" / "images").is_symlink())
        self.assertEqual(json.dumps(aj["S"]).count("NaN"), 0)

    def test_height_changes_the_scale_target(self):
        run = self.run_adapter(room_model(), height_cm=150)
        aj = json.loads((run / "40_align" / "align.json").read_text())
        self.assertAlmostEqual(aj["h_cap"], 1.2, places=3)  # 0.80 * 1.50 m
        self.assertEqual(aj["scale"]["sigma_log"], 0.08)

    def test_missing_sfm_output_is_an_error(self):
        run = self.tmp / "empty"
        run.mkdir()
        r = subprocess.run([sys.executable, str(ADAPTER), "--run", str(run)], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 20)


if __name__ == "__main__":
    unittest.main()
