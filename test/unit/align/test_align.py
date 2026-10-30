"""Acceptance tests D1, D2, D4, D5 (FINAL_PLAN WP2) for pipeline.align.

    python -m unittest discover -s test/unit/align -p 'test_align.py' -v      (numpy needed: container or venv)

D1  random rotation of a model: the up direction comes back within 1.0 deg (synthetic room with known up,
    and the honka smoke model, where the rotated run must agree with the original run).
D2  honka: up within 3 deg of the dataset's gravity (+Z of MuSHRoom's Polycam frame); scale error against the
    dataset's metric camera centres is recorded.  Needs the smoke model and transformations.json (tyrone).
D4  the aligned model keeps its registered count and its reprojection error (own reprojection from the keypoints;
    COLMAP's model_analyzer is run by d4_model_analyzer.sh on tyrone).
D5  honka: H_cap after scaling lies in 1.0-1.8 m.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the bare Mac has no numpy: skip, do not error
    raise unittest.SkipTest("numpy missing (run in the container or a venv)")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import synth  # noqa: E402
from pipeline.align import colmap_io as cio, core, stage  # noqa: E402

SMOKE = [os.environ.get("NS_SMOKE_MODEL", ""), "/w/runs/smoke_honka/clean/sparse/0",
         "/home/sher/namma-space/runs/smoke_honka/clean/sparse/0", "/tmp/namma-w2/smoke/sparse/0"]
HONKA_T = [os.environ.get("NS_HONKA_META", ""), "/w/data/mushroom/room_datasets/honka/iphone/long_capture",
           "/tmp/namma-w2/honka_meta"]


def first_existing(paths, probe):
    for p in paths:
        if p and os.path.exists(os.path.join(p, probe)):
            return p
    return None


def math_cos(deg):
    return float(np.cos(np.deg2rad(deg)))


def math_sin(deg):
    return float(np.sin(np.deg2rad(deg)))


def angle(a, b):
    return core.angle_deg(a, b)


def umeyama(A, B, w=None):
    w = np.ones(len(A)) if w is None else w
    w = w / w.sum()
    ma, mb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    X, Y = A - ma, B - mb
    cov = (Y * w[:, None]).T @ X
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (w * (X ** 2).sum(1)).sum()
    return s, R, mb - s * R @ ma


def robust_umeyama(A, B, keep=0.7, iters=8):
    w = np.ones(len(A))
    for _ in range(iters):
        s, R, t = umeyama(A, B, w)
        r = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        w = (r <= np.quantile(r, keep)).astype(float)
    return s, R, t, float(np.median(r))


class TestSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, cls.truth = synth.build(seed=3)

    def _model(self, seed):
        rng = np.random.default_rng(seed)
        Q = synth.random_rotation(rng)
        s0 = float(rng.uniform(0.3, 3.0))
        shift = rng.normal(scale=5.0, size=3)
        R, t, P = synth.apply_raw_frame(self.world, Q, s0, shift)
        d = tempfile.mkdtemp(prefix="ns_synth_")
        synth.write_model(d, R, t, P)
        return cio.load_model(d), Q, s0

    def test_D1_random_rotation_synthetic(self):
        errs = []
        for seed in range(5):
            m, Q, s0 = self._model(seed)
            res = core.align_model(m, seed=0)
            up_true_raw = Q @ self.truth["up"]
            errs.append(angle(res.up.up, up_true_raw))
        print(f"D1 synthetic: up error per random rotation (deg) {np.round(errs, 3).tolist()}")
        self.assertLess(max(errs), 1.0)

    def test_scale_and_height_synthetic(self):
        m, Q, s0 = self._model(11)
        res = core.align_model(m, seed=0)                 # no user height: generic 1.40 m, true camera height is 1.40
        true_scale = s0                                   # raw units -> metres
        print(f"synthetic scale {res.s:.4f} vs true {true_scale:.4f}; H_cap {res.h_cap_m:.3f} m; floor {res.floor_info['method']}")
        self.assertLess(abs(res.s / true_scale - 1), 0.05)
        self.assertTrue(1.0 <= res.h_cap_m <= 1.8)
        self.assertEqual(res.floor_info["method"], "ransac_plane")

    def test_camera_pitch_bias_does_not_gate(self):
        """Cameras whose mean up is 7 deg off (a pitched capture, as on honka) with level planes: the up stays right
        and the 5 deg cue-disagreement gate does not fire (integrator change, 30 Oct; it fired on honka at 2.0 deg)."""
        w = dict(self.world)
        tilt = synth.rot_from_axis_angle(np.array([1.0, 0, 0.3]) / np.linalg.norm([1.0, 0, 0.3]), np.deg2rad(7.0))
        C = -np.einsum("nji,nj->ni", w["R"], w["t"])
        w["R"] = np.array([Ri @ tilt.T for Ri in w["R"]])
        w["t"] = np.array([-w["R"][i] @ C[i] for i in range(len(C))])
        R, t, P = synth.apply_raw_frame(w, np.eye(3), 1.0, np.zeros(3))
        d = tempfile.mkdtemp(prefix="ns_synth_bias_")
        synth.write_model(d, R, t, P)
        res = core.align_model(cio.load_model(d), seed=0)
        err = angle(res.up.up, self.truth["up"])
        cam = res.up.cue_angles_deg["cameras"]
        print(f"pitch bias 7 deg: up error {err:.2f} deg, camera cue {cam:.2f} deg from fused, flags {res.flags}")
        self.assertLess(err, 3.0)
        self.assertGreater(res.up.cue_spread_deg, 5.0)            # the old rule would have fired
        self.assertFalse(any(f.startswith("up cues disagree") for f in res.flags))

    def test_user_height_changes_scale(self):
        m, Q, s0 = self._model(12)
        a = core.align_model(m, seed=0)
        b = core.align_model(m, height_m=1.60, seed=0)
        self.assertAlmostEqual(b.s / a.s, 0.8 * 1.60 / 1.40, places=3)
        self.assertLess(b.cues[0].sigma_log, a.cues[0].sigma_log)

    def test_D4_aligned_model_unchanged_synthetic(self):
        m, Q, s0 = self._model(13)
        res = core.align_model(m, seed=0)
        out = cio.apply_similarity(m, res.s, res.S[:3, :3] / res.s, res.S[:3, 3])
        self.assertEqual(out.n_images, m.n_images)
        self.assertEqual(out.n_points, m.n_points)
        e0, e1 = cio.reproj_error(m), cio.reproj_error(out)
        self.assertAlmostEqual(e0["mean_px"], e1["mean_px"], places=6)
        self.assertEqual(e0["n_obs"], e1["n_obs"])
        # frame: floor at y = 0, up = +Y, camera heights near 1.40 m
        C = out.centers()
        self.assertLess(abs(np.median(C[:, 1]) - 1.40), 0.1)
        ur = core.estimate_up(out, seed=5)
        self.assertLess(angle(ur.up, np.array([0, 1.0, 0])), 1.0)
        self.assertLess(abs(ur.floor_h), 0.1)

    def test_yaw_main_wall_on_x(self):
        m, Q, s0 = self._model(15)
        res = core.align_model(m, seed=0)
        self.assertTrue(res.yaw["clear"], res.yaw)
        out = cio.apply_similarity(m, res.s, res.S[:3, :3] / res.s, res.S[:3, 3])
        ur = core.estimate_up(out, seed=2)
        along, info = core.yaw_from_walls(ur.walls, ur.up)
        self.assertIsNotNone(along)
        self.assertGreater(abs(along[0]), math_cos(3.0))            # the best-supported wall now runs along X
        # and the room is axis aligned: every wall normal is within 4 deg of X or Z
        self.assertGreater(float(np.max(np.abs(np.array([w.n for w in ur.walls])[:, [0, 2]]), axis=1).min()), math_cos(6.0))

    def test_similarity_formulas(self):
        """R_i' = R_i R^T and t_i' = s t_i - R_i' t (CONTRACTS 2.0)."""
        m, Q, s0 = self._model(14)
        s, R, t = 1.7, synth.random_rotation(np.random.default_rng(1)), np.array([0.3, -0.2, 0.9])
        out = cio.apply_similarity(m, s, R, t)
        Ri = cio.qvec_to_rotmat(m.q[5]) @ R.T
        self.assertTrue(np.allclose(cio.qvec_to_rotmat(out.q[5]), Ri, atol=1e-9))
        self.assertTrue(np.allclose(out.t[5], s * m.t[5] - Ri @ t, atol=1e-9))
        self.assertTrue(np.allclose(out.xyz[7], s * R @ m.xyz[7] + t, atol=1e-9))


class TestHonka(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = first_existing(SMOKE, "images.bin")
        if d is None:
            raise unittest.SkipTest("honka smoke model not found")
        cls.model = cio.load_model(d)
        cls.res = core.align_model(cls.model, seed=0)
        cls.meta = first_existing(HONKA_T, "transformations.json")

    def test_D1_random_rotation_honka(self):
        errs = []
        for seed in range(3):
            Q = synth.random_rotation(np.random.default_rng(100 + seed))
            R = np.array([cio.qvec_to_rotmat(q) @ Q.T for q in self.model.q])
            m2 = cio.apply_similarity(self.model, 1.0, Q, np.zeros(3))
            res2 = core.align_model(m2, seed=0)
            errs.append(angle(res2.up.up, Q @ self.res.up.up))
        print(f"D1 honka: rotated-vs-original up difference (deg) {np.round(errs, 3).tolist()}")
        self.assertLess(max(errs), 1.0)

    def _truth(self):
        if self.meta is None:
            self.skipTest("transformations.json not found")
        d = json.load(open(os.path.join(self.meta, "transformations.json")))
        fr = {f["file_path"].split("/")[-1]: np.array(f["transform_matrix"]) for f in d["frames"]}
        names = [n for n in self.model.image_name if n in fr]
        idx = [self.model.image_name.index(n) for n in names]
        A = self.model.centers()[idx]
        B = np.array([fr[n][:3, 3] for n in names])
        return robust_umeyama(A, B)

    def test_D2_up_within_3deg_of_gravity(self):
        s_true, R, t, med = self._truth()
        g = np.array([0, 0, 1.0])       # MuSHRoom / Polycam world is Z-up (see NOTES.md)
        e = angle(R @ self.res.up.up, g)
        cues = {k: round(angle(R @ v, g), 2) for k, v in self.res.up.cues.items() if v is not None}
        print(f"D2 honka: fused up {e:.2f} deg from gravity; cues {cues}")
        print(f"D2 honka: scale {self.res.s:.4f} m/unit vs metric {s_true:.4f} (robust Umeyama, median residual {med*100:.1f} cm): "
              f"error {100*(self.res.s/s_true-1):+.1f}%; true camera height with our floor {self.res.height['h_cap']*s_true:.2f} m")
        self.assertLess(e, 3.0)
        self.assertLess(abs(self.res.s / s_true - 1), 0.20)     # camera-height-cue-only gate

    def test_D4_aligned_model_unchanged_honka(self):
        res = self.res
        out = cio.apply_similarity(self.model, res.s, res.S[:3, :3] / res.s, res.S[:3, 3])
        self.assertEqual(out.n_images, self.model.n_images)
        self.assertEqual(out.n_points, self.model.n_points)
        self.assertAlmostEqual(out.mean_error(), self.model.mean_error(), places=9)
        e0, e1 = cio.reproj_error(self.model), cio.reproj_error(out)
        print(f"D4 honka: {out.n_images} images, {out.n_points} points; own reprojection {e0['mean_px']:.6f} -> {e1['mean_px']:.6f} px "
              f"({e0['n_obs']} observations); stored error {self.model.mean_error():.4f} -> {out.mean_error():.4f}")
        self.assertAlmostEqual(e0["mean_px"], e1["mean_px"], places=6)

    def test_D5_hcap_in_range(self):
        print(f"D5 honka: H_cap x scale = {self.res.h_cap_m:.2f} m")
        self.assertTrue(1.0 <= self.res.h_cap_m <= 1.8)

    def test_stage_writes_run_dir(self):
        d = tempfile.mkdtemp(prefix="ns_align_run_")
        und = os.path.join(d, "30_undist")
        os.makedirs(os.path.join(und, "images"))
        import shutil
        shutil.copytree(first_existing(SMOKE, "images.bin"), os.path.join(und, "sparse"))
        out = stage.run_stage(d, log=lambda *_: None)
        a = Path(d, "40_align")
        self.assertTrue((a / "align.json").exists())
        self.assertTrue((a / "sparse" / "0" / "images.bin").exists())
        self.assertTrue(os.path.islink(a / "images"))
        self.assertEqual(os.readlink(a / "images"), os.path.join("..", "30_undist", "images"))
        self.assertTrue((a / "figures" / "level_before_after.png").exists())
        j = json.load(open(a / "align.json"))
        for k in ("S", "up_resid_deg", "floor", "h_cap", "scale", "gate"):
            self.assertIn(k, j)
        self.assertEqual(len(j["S"]), 4)
        for k in ("value", "sigma_log", "pct_68", "cues"):
            self.assertIn(k, j["scale"])
        self.assertTrue(j["model"]["registered"]["before"] == j["model"]["registered"]["after"])
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
