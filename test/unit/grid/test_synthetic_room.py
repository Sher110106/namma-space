"""G3, G5, G6 and the general output checks on the synthetic room (see synth.py)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import common
import synth
from common import run_and_load, tmpdir
from pipeline.grid import colmapio, frame, plyio, viewpoints

C = 0.05
TOL = C + 1e-9          # "within one cell"


def dist_to_rect(px, pz, x0, x1, z0, z1):
    dx = np.maximum(np.maximum(x0 - px, 0), px - x1)
    dz = np.maximum(np.maximum(z0 - pz, 0), pz - z1)
    return np.hypot(dx, dz)


def solids(t):
    """Rectangles (x0, x1, z0, z1) of everything solid in the room except the floaters."""
    X0, X1, Z0, Z1 = t["room"]
    d0, d1 = t["door_x"] - t["door_w"] / 2, t["door_x"] + t["door_w"] / 2
    cx0, cx1, _, cz1 = t["closet"]
    return [(X0, X0, Z0, Z1), (X1, X1, Z0, Z1), (X0, X1, Z0, Z0), (X0, d0, Z1, Z1), (d1, X1, Z1, Z1),
            (d0, d0, Z1, Z1 + 0.12), (d1, d1, Z1, Z1 + 0.12), (cx0, cx0, Z1 + 0.12, cz1), (cx1, cx1, Z1 + 0.12, cz1),
            (cx0, cx1, cz1, cz1), t["table"], t["pillar"]]


class SyntheticRoom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rundir = tmpdir("w4room_")
        cls.t = synth.write_run(cls.rundir, n_splats=250_000)
        cls.rc, cls.g = run_and_load(cls.rundir)
        cls.walk = cls.g.walkable
        cls.wx, cls.wz = cls.g.centres(cls.walk)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.rundir, ignore_errors=True)

    def test_exit_code_and_mode(self):
        self.assertEqual(self.rc, 0)
        self.assertEqual(self.g.j["mode"], "grid")
        self.assertEqual(self.g.j["warnings"], [])

    # ---------------------------------------------------------------- G3
    def test_G3_no_walkable_cell_outside_room_except_door(self):
        X0, X1, Z0, Z1 = self.t["room"]
        cx0, cx1, cz0, cz1 = self.t["closet"]
        in_room = (self.wx >= X0) & (self.wx <= X1) & (self.wz >= Z0) & (self.wz <= Z1)
        in_closet = (self.wx >= cx0) & (self.wx <= cx1) & (self.wz >= cz0) & (self.wz <= cz1)
        outside = ~(in_room | in_closet)
        self.assertEqual(int(outside.sum()), 0, "walkable cells outside the room and the door closet")
        # the hole in the left wall (0.3 m, at 1 m height) must not let anyone out to the terrace
        self.assertTrue(((self.wx < X0)).sum() == 0)

    def test_G3_table_plus_025_blocked(self):
        x0, x1, z0, z1 = self.t["table"]
        d = dist_to_rect(self.wx, self.wz, x0, x1, z0, z1)
        # one cell (5 cm) of tolerance: the grid has 5 cm cells, so "table + 0.25 m" is blocked out to 0.20 m for sure
        self.assertEqual(int((d <= 0.25 - C).sum()), 0, "walkable cells within 0.20 m of the table")
        # and most of the 0.20-0.25 m ring is blocked too (the inflation radius is 0.227 m: 0.25 m less half a cell)
        gx, gz = np.meshgrid(np.arange(x0 - 0.3, x1 + 0.3, C) + C / 2, np.arange(z0 - 0.3, z1 + 0.3, C) + C / 2)
        ring = (dist_to_rect(gx, gz, x0, x1, z0, z1) <= 0.25)
        i = np.floor((gz[ring] - self.g.oz) / C).astype(int)
        j = np.floor((gx[ring] - self.g.ox) / C).astype(int)
        blocked = ~self.walk[i, j]
        self.assertGreater(blocked.mean(), 0.90)
        # and it is not blocked far away: cells 0.5 m from the table (still in the room) are walkable
        gx, gz = np.meshgrid(np.arange(x0 - 0.8, x1 + 0.8, C) + C / 2, np.arange(z0 - 0.8, z1 + 0.8, C) + C / 2)
        far = (dist_to_rect(gx, gz, x0, x1, z0, z1) >= 0.5) & (dist_to_rect(gx, gz, x0, x1, z0, z1) <= 0.6)
        i = np.floor((gz[far] - self.g.oz) / C).astype(int)
        j = np.floor((gx[far] - self.g.ox) / C).astype(int)
        self.assertGreater(self.walk[i, j].mean(), 0.5)

    def test_G3_floaters_ignored(self):
        rects = solids(self.t)
        n_checked = 0
        for (fx, fz), fy in self.t["floaters"]:
            fi, fj = self.g.pix(fx, fz)
            self.assertTrue(self.walk[fi, fj], f"cell under the floater at {(fx, fz)} is not walkable")
            gx, gz = np.meshgrid(np.arange(fx - 0.35, fx + 0.35, C / 2) + C / 4, np.arange(fz - 0.35, fz + 0.35, C / 2) + C / 4)
            r = np.hypot(gx - fx, gz - fz)
            far_from_solids = np.min([dist_to_rect(gx, gz, *q) for q in rects], axis=0) > 0.30
            ring = (r >= 0.20) & (r <= 0.26) & far_from_solids
            i = np.floor((gz[ring] - self.g.oz) / C).astype(int)
            j = np.floor((gx[ring] - self.g.ox) / C).astype(int)
            if ring.sum() < 20:
                continue
            n_checked += 1
            self.assertGreater(self.walk[i, j].mean(), 0.95, f"floater at {(fx, fz)} y={fy} blocks its surroundings")
        self.assertGreaterEqual(n_checked, 2)
        # the stage says it dropped three blobs
        self.assertEqual(self.g.j["diagnostics"]["floater_blobs_ignored"], 3)

    def test_G3_door_keeps_040_m(self):
        t = self.t
        z = t["wall_z"] + 0.06           # inside the 12 cm thick wall, between the jambs
        i, j0 = self.g.pix(t["door_x"], z)
        row = self.walk[i]
        self.assertTrue(row[j0])
        a = j0
        while a > 0 and row[a - 1]:
            a -= 1
        b = j0
        while b < len(row) - 1 and row[b + 1]:
            b += 1
        clear = (b - a + 1) * C
        self.assertGreaterEqual(clear, 0.40 - 1e-9, f"door clear width {clear:.2f} m")
        self.assertLessEqual(clear, 0.60)

    def test_walkable_is_one_component_and_spawn_inside(self):
        lab, n = ndi.label(self.walk, structure=ndi.generate_binary_structure(2, 1))
        self.assertEqual(n, 1)
        sp = self.g.j["spawn"]["position"]
        i, j = self.g.pix(sp[0], sp[2])
        self.assertTrue(self.walk[i, j])
        self.assertTrue(self.g.j["spawn_ok"])
        self.assertAlmostEqual(sp[1], synth.H_CAP, places=3)

    # ---------------------------------------------------------------- G5
    def test_G5_room_extent_within_5cm(self):
        L, W = self.g.j["room_extent_m"]
        self.assertAlmostEqual(L, 5.0, delta=0.05)
        self.assertAlmostEqual(W, 4.0, delta=0.05)

    # ---------------------------------------------------------------- G6
    def test_G6_pillar_lands_within_one_cell_on_grid_and_minimap(self):
        x0, x1, z0, z1 = self.t["pillar"]
        # grid.png: value-0 cells in a window around the pillar that stays clear of the table
        i0, j0 = self.g.pix(x0 - 0.12, z0 - 0.12)
        i1, j1 = self.g.pix(x1 + 0.12, z1 + 0.12)
        win = self.g.png[i0:i1 + 1, j0:j1 + 1] == 0
        ii, jj = np.nonzero(win)
        gx = self.g.ox + (jj + j0 + 0.5) * C
        gz = self.g.oz + (ii + i0 + 0.5) * C
        self.assertGreater(len(ii), 20)
        self.assertLessEqual(abs(gx.mean() - (x0 + x1) / 2), TOL)
        self.assertLessEqual(abs(gz.mean() - (z0 + z1) / 2), TOL)
        self.assertLessEqual(abs((gx.min() - C / 2) - x0), TOL)
        self.assertLessEqual(abs((gx.max() + C / 2) - x1), TOL)
        self.assertLessEqual(abs((gz.min() - C / 2) - z0), TOL)
        self.assertLessEqual(abs((gz.max() + C / 2) - z1), TOL)
        # minimap.png: magenta pixels
        m = self.g.mini.astype(int)
        mag = (m[..., 0] > 200) & (m[..., 1] < 70) & (m[..., 2] > 200)
        mi, mj = np.nonzero(mag)
        self.assertGreater(len(mi), 100)
        ppm = self.g.j["minimap"]["px_per_m"]
        mx = self.g.ox + (mj + 0.5) / ppm
        mz = self.g.oz + (mi + 0.5) / ppm
        self.assertLessEqual(abs(mx.mean() - (x0 + x1) / 2), TOL)
        self.assertLessEqual(abs(mz.mean() - (z0 + z1) / 2), TOL)
        self.assertLessEqual(abs((mx.min() - 0.5 / ppm) - x0), TOL)
        self.assertLessEqual(abs((mx.max() + 0.5 / ppm) - x1), TOL)
        self.assertLessEqual(abs((mz.min() - 0.5 / ppm) - z0), TOL)
        self.assertLessEqual(abs((mz.max() + 0.5 / ppm) - z1), TOL)

    # ---------------------------------------------------------------- format
    def test_png_and_json_format(self):
        g = self.g
        self.assertEqual(g.png.dtype, np.uint8)
        self.assertEqual(g.png.shape, (g.j["h"], g.j["w"]))
        self.assertEqual(set(np.unique(g.png)) - {0, 128, 255}, set())
        self.assertEqual(g.j["row0"], "min_z")
        self.assertEqual(g.j["cell"], 0.05)
        ppm = g.j["minimap"]["px_per_m"]
        self.assertEqual(g.mini.shape, (g.j["h"] * round(ppm * C), g.j["w"] * round(ppm * C), 3))
        for k in ("origin_xz", "cell", "w", "h", "spawn", "bookmarks", "tour", "room_extent_m", "walkable_m2", "path_coverage"):
            self.assertIn(k, g.j)
        self.assertAlmostEqual(g.j["walkable_m2"], float(g.walkable.sum()) * C * C, places=2)

    def test_png_is_standard(self):
        raw = (self.g.dir / "grid.png").read_bytes()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        im = Image.open(self.g.dir / "grid.png")
        self.assertEqual(im.mode, "L")
        self.assertTrue(np.array_equal(np.array(im), self.g.png))

    # ---------------------------------------------------------------- spawn, bookmarks, tour, coverage, eye
    def test_G1_style_checks(self):
        j = self.g.j
        self.assertGreaterEqual(j["path_coverage"], 0.95)
        model = colmapio.read_model(self.rundir / "40_align" / "sparse" / "0")
        cen = model.centres
        i = np.array([self.g.pix(x, z)[0] for x, z in zip(cen[:, 0], cen[:, 2])])
        jj = np.array([self.g.pix(x, z)[1] for x, z in zip(cen[:, 0], cen[:, 2])])
        self.assertGreaterEqual(self.walk[i, jj].mean(), 0.95)
        self.assertEqual(len(j["bookmarks"]), 5)
        seen = set()
        for b in j["bookmarks"]:
            x, y, z = b["position"]
            ii, jx = self.g.pix(x, z)
            self.assertTrue(self.walk[ii, jx], f"bookmark {b['id']} not walkable")
            self.assertAlmostEqual(y, synth.H_CAP, places=3)
            d = np.hypot(cen[:, 0] - x, cen[:, 2] - z)
            k = int(d.argmin())
            self.assertLess(d[k], 1e-3, "bookmark is not a real camera position")
            pitch = float(frame.pitch_deg_from_forward(model.forward[k]))
            self.assertLessEqual(abs(pitch), 20.0)
            seen.add(k)
        self.assertEqual(len(seen), 5)

    def test_spawn_rule(self):
        model = colmapio.read_model(self.rundir / "40_align" / "sparse" / "0")
        cnt, med = viewpoints.camera_stats(model.point_ids, model.pt_ids, model.pt_xyz, model.R, model.t)
        pitch = frame.pitch_deg_from_forward(model.forward)
        fj = json.loads((self.rundir / "10_frames" / "frames.json").read_text())
        sharp = np.array([k["sharp"] for k in fj["kept"]])
        sp = self.g.j["spawn"]
        k = int(np.hypot(model.centres[:, 0] - sp["position"][0], model.centres[:, 2] - sp["position"][2]).argmin())
        self.assertLessEqual(abs(pitch[k]), 15.0)
        self.assertGreater(sharp[k], np.median(sharp))
        self.assertGreaterEqual(med[k], 2.0)
        ok = (np.abs(pitch) <= 15) & (sharp > np.median(sharp)) & (np.nan_to_num(med) >= 2.0)
        i = np.array([self.g.pix(x, z)[0] for x, z in zip(model.centres[:, 0], model.centres[:, 2])])
        jj = np.array([self.g.pix(x, z)[1] for x, z in zip(model.centres[:, 0], model.centres[:, 2])])
        ok &= self.walk[i, jj]
        self.assertEqual(cnt[k], cnt[ok].max())
        # yaw convention: 0 looks along -Z, positive turns left
        f = frame.forward_from_yaw(sp["yaw_deg"])
        self.assertAlmostEqual(float(np.dot(f, model.forward[k] * [1, 0, 1] / np.linalg.norm(model.forward[k] * [1, 0, 1]))), 1.0, places=2)

    def test_G2_eye_not_in_wall(self):
        ply = plyio.read_splats(self.rundir / "60_export" / "clean.ply")
        tree = cKDTree(ply.xyz[ply.opacity > 0.5])
        q = np.stack([self.wx, np.full_like(self.wx, synth.H_CAP), self.wz], axis=1)
        n = tree.query_ball_point(q, r=0.15, return_length=True)
        self.assertLessEqual(int(np.max(n)), 2)
        self.assertEqual(self.g.j["diagnostics"]["eye_violations_final"], 0)

    def test_tour(self):
        tour = self.g.j["tour"]
        self.assertEqual(len(tour), 61)
        self.assertEqual(tour[0]["t"], 0)
        self.assertEqual(tour[-1]["t"], 60)
        pos = np.array([k["position"] for k in tour])
        i = np.array([self.g.pix(x, z)[0] for x, z in zip(pos[:, 0], pos[:, 2])])
        jj = np.array([self.g.pix(x, z)[1] for x, z in zip(pos[:, 0], pos[:, 2])])
        self.assertTrue(self.walk[i, jj].all(), "a tour keyframe is off the walkable area")
        np.testing.assert_allclose(pos[:, 1], synth.H_CAP, atol=1e-3)
        sp = np.array(self.g.j["spawn"]["position"])
        self.assertLess(np.linalg.norm(pos[0] - sp), 0.1)
        step = np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1)
        self.assertLess(step.max(), 1.5)                       # metres per second
        self.assertEqual(len(self.g.j["tour_stops"]), 6)
        yaw = np.array([k["yaw_deg"] for k in tour])
        dyaw = np.abs((np.diff(yaw) + 180) % 360 - 180)
        self.assertLess(dyaw.max(), 90.0, "tour turns faster than 90 degrees per second")

    def test_cli_runs(self):
        run = tmpdir("w4cli_")
        try:
            synth.write_run(run, n_splats=60_000)
            r = subprocess.run([sys.executable, "-m", "pipeline.grid", "--run", str(run), "--quiet"], cwd=str(common.ROOT),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            last = r.stdout.strip().splitlines()[-1]
            self.assertTrue(last.startswith("NS_RESULT "))
            res = json.loads(last[len("NS_RESULT "):])
            self.assertEqual(res["exit"], 0)
            self.assertEqual(res["mode"], "grid")
            self.assertTrue(res["spawn_ok"])
            self.assertTrue((run / "70_grid" / "grid.png").exists())
            self.assertFalse(any(p.name.startswith("70_grid.") for p in run.iterdir()), "temp dir left behind")
            r = subprocess.run([sys.executable, "-m", "pipeline.grid", "--run", str(run / "nope"), "--quiet"], cwd=str(common.ROOT),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 20)
            self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1][len("NS_RESULT "):])["exit"], 20)
        finally:
            shutil.rmtree(run, ignore_errors=True)


class TFinal(unittest.TestCase):
    def test_t_final_moves_everything(self):
        """A rotation and shift in T_final must carry the grid, spawn and extents with it (CONTRACTS 2.0)."""
        run = tmpdir("w4tf_")
        try:
            q = [0.0, np.sin(np.radians(45)), 0.0, np.cos(np.radians(45))]      # 90 degrees about +Y
            tf = {"position": [2.0, 0.0, -3.0], "quaternion": q, "scale": 1.0}
            synth.write_run(run, n_splats=120_000, with_tfinal=tf)
            rc, g = run_and_load(run)
            self.assertEqual(rc, 0)
            T = frame.parse_tfinal(tf)
            X0, X1, Z0, Z1 = synth.truth()["room"]
            corners = frame.apply_points(T, np.array([[X0, 0, Z0], [X1, 0, Z1]]))
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            wx, wz = g.centres(g.walkable)
            eps = 0.02
            inside_room = (wx >= lo[0] - eps) & (wx <= hi[0] + eps) & (wz >= lo[2] - eps) & (wz <= hi[2] + eps)
            # walkable cells are inside the moved room, or in the moved closet (door side), never elsewhere
            cl = synth.truth()["closet"]
            cc = frame.apply_points(T, np.array([[cl[0], 0, cl[2]], [cl[1], 0, cl[3]]]))
            clo, chi = cc.min(axis=0), cc.max(axis=0)
            in_closet = (wx >= clo[0] - eps) & (wx <= chi[0] + eps) & (wz >= clo[2] - eps) & (wz <= chi[2] + eps)
            self.assertTrue((inside_room | in_closet).all())
            self.assertGreater(inside_room.mean(), 0.7)
            L, W = g.j["room_extent_m"]
            self.assertAlmostEqual(L, 5.0, delta=0.05)
            self.assertAlmostEqual(W, 4.0, delta=0.05)
            self.assertTrue(g.j["inputs"]["t_final_applied"])
            self.assertGreaterEqual(g.j["path_coverage"], 0.95)
        finally:
            shutil.rmtree(run, ignore_errors=True)


class SubCellPlacement(unittest.TestCase):
    """Shifting the whole room by fractions of a cell (through T_final) must not close the door or move the numbers."""

    def test_offsets(self):
        offsets = [(0.0, 0.0), (0.0125, 0.02), (0.025, 0.037), (0.037, 0.0125), (0.049, 0.049), (0.02, 0.005)]
        t = synth.truth()
        for dx, dz in offsets:
            run = tmpdir("w4off_")
            try:
                synth.write_run(run, n_splats=250_000, with_tfinal={"position": [dx, 0.0, dz], "quaternion": [0, 0, 0, 1], "scale": 1.0})
                rc, g = run_and_load(run)
                self.assertEqual(rc, 0, f"offset {(dx, dz)}")
                walk = g.walkable
                wx, wz = g.centres(walk)
                # door
                i, j0 = g.pix(t["door_x"] + dx, t["wall_z"] + 0.06 + dz)
                row = walk[i]
                a = b = j0
                self.assertTrue(row[j0], f"door cell not walkable at offset {(dx, dz)}")
                while a > 0 and row[a - 1]:
                    a -= 1
                while b < len(row) - 1 and row[b + 1]:
                    b += 1
                self.assertGreaterEqual((b - a + 1) * C, 0.40 - 1e-9, f"door at offset {(dx, dz)}")
                # table
                x0, x1, z0, z1 = t["table"]
                d = dist_to_rect(wx, wz, x0 + dx, x1 + dx, z0 + dz, z1 + dz)
                self.assertEqual(int((d <= 0.25 - C).sum()), 0, f"table at offset {(dx, dz)}")
                # extents
                L, W = g.j["room_extent_m"]
                self.assertAlmostEqual(L, 5.0, delta=0.05)
                self.assertAlmostEqual(W, 4.0, delta=0.05)
                # pillar centroid on the grid
                px0, px1, pz0, pz1 = t["pillar"]
                i0, j0_ = g.pix(px0 + dx - 0.12, pz0 + dz - 0.12)
                i1, j1_ = g.pix(px1 + dx + 0.12, pz1 + dz + 0.12)
                ii, jj = np.nonzero(g.png[i0:i1 + 1, j0_:j1_ + 1] == 0)
                gx = g.ox + (jj + j0_ + 0.5) * C
                gz = g.oz + (ii + i0 + 0.5) * C
                self.assertLessEqual(abs(gx.mean() - ((px0 + px1) / 2 + dx)), TOL)
                self.assertLessEqual(abs(gz.mean() - ((pz0 + pz1) / 2 + dz)), TOL)
            finally:
                shutil.rmtree(run, ignore_errors=True)


class FallbackLadder(unittest.TestCase):
    """FINAL_PLAN 6.5: grid, then corridor discs along the path, then free fly inside the SfM bounds (labelled)."""

    def test_forced_corridor(self):
        run = tmpdir("w4fb_")
        try:
            synth.write_run(run, n_splats=120_000)
            rc, g = run_and_load(run, force_mode="corridor")
            self.assertEqual(rc, 30)
            self.assertEqual(g.j["mode"], "corridor")
            self.assertTrue(g.j["spawn_ok"])
            self.assertGreaterEqual(g.j["path_coverage"], 0.95)
            self.assertLess(g.j["walkable_m2"], 40)
        finally:
            shutil.rmtree(run, ignore_errors=True)

    def test_forced_bounds(self):
        run = tmpdir("w4fb_")
        try:
            synth.write_run(run, n_splats=120_000)
            rc, g = run_and_load(run, force_mode="bounds")
            self.assertEqual(rc, 30)
            self.assertEqual(g.j["mode"], "bounds")
            self.assertTrue(g.j["spawn_ok"])
            self.assertTrue(any("free fly" in w for w in g.j["warnings"]))
        finally:
            shutil.rmtree(run, ignore_errors=True)

    def test_bounds_without_sfm_points(self):
        run = tmpdir("w4fb_")
        try:
            synth.write_run(run, n_splats=120_000)
            md = run / "40_align" / "sparse" / "0"
            m = colmapio.read_model(md)
            m.pt_ids, m.pt_xyz, m.raw_points = m.pt_ids[:0], m.pt_xyz[:0], {"extra": []}
            m.point_ids = [ids[:0] for ids in m.point_ids]
            m.raw_images = [(a, b, xy[:0], ids[:0]) for a, b, xy, ids in m.raw_images]
            colmapio.write_model(md, m)
            rc, g = run_and_load(run, force_mode="bounds")
            self.assertEqual(rc, 30)
            self.assertEqual(g.j["mode"], "bounds")
            self.assertTrue(g.j["spawn_ok"])
        finally:
            shutil.rmtree(run, ignore_errors=True)

    def test_fog_at_eye_height_walks_down_the_ladder(self):
        """Opaque junk all through the eye band blocks every cell, so grid and corridor both fail and the stage says so."""
        run = tmpdir("w4fb_")
        try:
            synth.write_run(run, n_splats=120_000)
            ply = plyio.read_splats(run / "60_export" / "clean.ply")
            rng = np.random.default_rng(5)
            n = 60_000
            fog = np.stack([rng.uniform(synth.X0, synth.X0 + 4, n), rng.uniform(1.25, 1.55, n), rng.uniform(synth.Z0, synth.Z0 + 6.2, n)], axis=1)
            plyio.write_ply(run / "60_export" / "clean.ply", np.concatenate([ply.xyz, fog.astype(np.float32)]),
                            np.concatenate([ply.opacity, np.full(n, 0.9, np.float32)]),
                            np.concatenate([ply.scale, np.full((n, 3), 0.03, np.float32)]),
                            np.concatenate([ply.rgb, np.full((n, 3), 0.5, np.float32)]))
            rc, g = run_and_load(run)
            self.assertEqual(rc, 30)
            self.assertEqual(g.j["mode"], "bounds")
            warn = " ".join(g.j["warnings"])
            self.assertIn("mode grid failed", warn)
            self.assertIn("mode corridor failed", warn)
        finally:
            shutil.rmtree(run, ignore_errors=True)


class Variants(unittest.TestCase):
    """Inputs the real pipeline may hand over in other shapes."""

    def test_no_frames_json_no_h_cap_uses_cameras(self):
        run = tmpdir("w4var_")
        try:
            synth.write_run(run, n_splats=120_000, sharp=False)
            (run / "40_align" / "align.json").write_text(json.dumps({"up_resid_deg": 0.4}))
            rc, g = run_and_load(run)
            self.assertEqual(rc, 0)
            self.assertEqual(g.j["h_cap_source"], "cameras (no h_cap in align.json)")
            self.assertAlmostEqual(g.j["h_cap_m"], synth.H_CAP, places=2)
            self.assertFalse(g.j["inputs"]["frames_json"])
            self.assertTrue(any("no h_cap" in w for w in g.j["warnings"]))
        finally:
            shutil.rmtree(run, ignore_errors=True)

    def test_t_final_scale_scales_everything(self):
        run = tmpdir("w4var_")
        try:
            tf = {"scale": 1.5, "quaternion": [0, 0, 0, 1], "translation": [0.0, 0.0, 0.0]}
            synth.write_run(run, n_splats=250_000, with_tfinal=tf)
            rc, g = run_and_load(run)
            self.assertEqual(g.j["mode"], "grid")
            self.assertAlmostEqual(g.j["h_cap_m"], synth.H_CAP * 1.5, places=3)
            L, W = g.j["room_extent_m"]
            self.assertAlmostEqual(L, 7.5, delta=0.075)
            self.assertAlmostEqual(W, 6.0, delta=0.075)
            self.assertGreaterEqual(g.j["path_coverage"], 0.95)
            self.assertAlmostEqual(g.j["spawn"]["position"][1], synth.H_CAP * 1.5, places=3)
            self.assertTrue(any("outside 1.0-1.8 m" in w for w in g.j["warnings"]))
        finally:
            shutil.rmtree(run, ignore_errors=True)

    def test_tfinal_skip_ignores_export_json(self):
        run = tmpdir("w4var_")
        try:
            tf = {"scale": 1.0, "quaternion": [0, 0, 0, 1], "translation": [5.0, 0.0, 5.0]}
            synth.write_run(run, n_splats=120_000, with_tfinal=tf)
            rc, g = run_and_load(run, tfinal="skip")
            self.assertFalse(g.j["inputs"]["t_final_applied"])
            sp = g.j["spawn"]["position"]
            self.assertLess(sp[0], 3.0)           # the room sits around x in [-1.3, 2.7]: not shifted by 5 m
        finally:
            shutil.rmtree(run, ignore_errors=True)

    def test_text_model(self):
        run = tmpdir("w4var_")
        try:
            synth.write_run(run, n_splats=120_000)
            md = run / "40_align" / "sparse" / "0"
            m = colmapio.read_model(md)
            lines = ["# images"]
            for k, name in enumerate(m.names):
                q = colmapio.rot_to_qvec(m.R[k])
                lines.append(f"{k + 1} {q[0]} {q[1]} {q[2]} {q[3]} {m.t[k][0]} {m.t[k][1]} {m.t[k][2]} 1 {name}")
                lines.append(" ".join(f"0 0 {int(i)}" for i in m.point_ids[k]))
            (md / "images.txt").write_text("\n".join(lines) + "\n")
            pl = ["# points"] + [f"{int(i)} {x} {y} {z} 128 128 128 0.5 1 0" for i, (x, y, z) in zip(m.pt_ids, m.pt_xyz)]
            (md / "points3D.txt").write_text("\n".join(pl) + "\n")
            (md / "images.bin").unlink()
            (md / "points3D.bin").unlink()
            rc, g = run_and_load(run)
            self.assertEqual(rc, 0)
            self.assertEqual(g.j["counts"]["cameras"], len(m.names))
            self.assertEqual(g.j["counts"]["sfm_points"], len(m.pt_ids))
        finally:
            shutil.rmtree(run, ignore_errors=True)


class EyeAtHeight(unittest.TestCase):
    def test_clump_at_eye_height_blocks_nearby_cells(self):
        """Three or more opaque centres within 0.15 m of the eye point make that cell not walkable (G2), even when no cell has 3."""
        run = tmpdir("w4eye_")
        try:
            synth.write_run(run, n_splats=150_000)
            ply = plyio.read_splats(run / "60_export" / "clean.ply")
            cx, cz = synth.X0 + 1.2, synth.Z0 + 4.2        # free floor in the north of the room
            ring = np.array([[cx + dx, synth.H_CAP, cz + dz] for dx, dz in ((0, 0), (0.07, 0), (-0.07, 0), (0, 0.07), (0, -0.07))])
            xyz = np.concatenate([ply.xyz, ring.astype(np.float32)])
            op = np.concatenate([ply.opacity, np.full(5, 0.9, np.float32)])
            sc = np.concatenate([ply.scale, np.full((5, 3), 0.03, np.float32)])
            rgb = np.concatenate([ply.rgb, np.full((5, 3), 0.5, np.float32)])
            plyio.write_ply(run / "60_export" / "clean.ply", xyz, op, sc, rgb)
            rc, g = run_and_load(run)
            self.assertEqual(g.j["diagnostics"]["eye_violations_final"], 0)
            self.assertGreater(g.j["diagnostics"]["eye_cells_removed"], 0)
            wx, wz = g.centres(g.walkable)
            near = np.hypot(wx - cx, wz - cz) <= 0.07
            self.assertEqual(int(near.sum()), 0)
        finally:
            shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
