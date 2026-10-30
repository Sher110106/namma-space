import os
import sqlite3
import tempfile
import unittest

import _paths  # noqa: F401
from pipeline.sfm import colmapio
import helpers


class TestColmapIO(unittest.TestCase):
    def test_roundtrip_images_and_cameras(self):
        with tempfile.TemporaryDirectory() as d:
            helpers.write_model(d, helpers.line_path(5))
            cams, imgs = colmapio.read_model(d)
            self.assertEqual(len(imgs), 5)
            self.assertEqual(imgs[2].name, "000003.jpg")
            self.assertAlmostEqual(imgs[2].centre[0], 0.2, places=9)
            self.assertEqual(cams[0].model, "SIMPLE_RADIAL")
            self.assertEqual((cams[0].width, cams[0].height), (1920, 1080))

    def test_camera_centre_with_rotation(self):
        # 90 degrees about z: R = [[0,-1,0],[1,0,0],[0,0,1]]; t = (1,2,3) -> C = -R^T t = (-2, 1, -3)
        q = (2 ** -0.5, 0.0, 0.0, 2 ** -0.5)
        c = colmapio.camera_centre(q, (1.0, 2.0, 3.0))
        for got, want in zip(c, (-2.0, 1.0, -3.0)):
            self.assertAlmostEqual(got, want, places=9)

    def test_text_model_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "images.txt"), "w").close()
            with self.assertRaises(FileNotFoundError):
                colmapio.read_model(d)

    def test_list_models_sorted_numerically(self):
        with tempfile.TemporaryDirectory() as d:
            for k in (10, 2, 0):
                helpers.write_model(os.path.join(d, str(k)), helpers.line_path(3))
            os.makedirs(os.path.join(d, "junk"))
            self.assertEqual([os.path.basename(m) for m in colmapio.list_models(d)], ["0", "2", "10"])

    def test_parse_model_analyzer(self):
        text = """I20261029 22:10:47.689342  2387 model.cc:437] Registered frames: 17
I20261029 22:10:47.689349  2387 model.cc:440] Registered images: 17
I20261029 22:10:47.689353  2387 model.cc:442] Points: 1520
I20261029 22:10:47.689363  2387 model.cc:445] Mean track length: 4.825000
I20261029 22:10:47.689375  2387 model.cc:450] Mean reprojection error: 0.683403px"""
        s = colmapio.parse_model_analyzer(text)
        self.assertEqual(s["registered"], 17)
        self.assertEqual(s["points"], 1520)
        self.assertAlmostEqual(s["track_len"], 4.825)
        self.assertAlmostEqual(s["reproj_px"], 0.683403)

    def test_copy_db_keeps_uncheckpointed_wal_content(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "a.db"), os.path.join(d, "b.db")
            con = sqlite3.connect(src)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA wal_autocheckpoint=0")
            con.execute("CREATE TABLE t (x INTEGER)")
            con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(1000)])
            con.commit()   # rows sit in a.db-wal, not in a.db; the connection stays open like a crashed writer
            self.assertTrue(os.path.getsize(src + "-wal") > 0)
            colmapio.copy_db(src, dst)
            con.close()
            out = sqlite3.connect(dst)
            self.assertEqual(out.execute("SELECT count(*) FROM t").fetchone()[0], 1000)
            out.close()

    def test_keypoint_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "db.db")
            con = sqlite3.connect(p)
            con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
            con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
            con.execute("INSERT INTO images VALUES (1,'a.jpg',1),(2,'b.jpg',1)")
            con.execute("INSERT INTO keypoints VALUES (1, 321, 2, x'')")
            con.commit()
            con.close()
            self.assertEqual(colmapio.keypoint_counts(p), {"a.jpg": 321, "b.jpg": 0})


if __name__ == "__main__":
    unittest.main()
