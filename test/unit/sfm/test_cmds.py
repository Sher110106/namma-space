import os
import tempfile
import unittest

import _paths  # noqa: F401
from pipeline.sfm import check_flags, cmds
import helpers


class TestCmds(unittest.TestCase):
    def setUp(self):
        self.cfg = cmds.Cfg(cache="/c")

    def flat(self, argv):
        return " ".join(argv)

    def test_variant_a_uses_aliked_lightglue_and_vocab(self):
        e = self.flat(cmds.extract_aliked("db", "img", self.cfg, 4))
        m = self.flat(cmds.match_sequential_aliked("db", self.cfg, 4))
        self.assertIn("--FeatureExtraction.type ALIKED_N32", e)
        self.assertIn("--AlikedExtraction.n32_model_path /c/aliked-n32.onnx", e)
        self.assertIn("--FeatureMatching.type ALIKED_LIGHTGLUE", m)
        self.assertIn("--AlikedMatching.lightglue_model_path /c/aliked-lightglue.onnx", m)
        self.assertIn("--SequentialMatching.loop_detection 1", m)
        self.assertIn("--SequentialMatching.vocab_tree_path /c/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin", m)

    def test_variant_b_is_cpu_sift_plus_onnx_lightglue(self):
        e = self.flat(cmds.extract_sift_cpu("db", "img", self.cfg, 4))
        m = self.flat(cmds.match_sequential_sift("db", self.cfg, 4))
        self.assertIn("--FeatureExtraction.type SIFT", e)
        self.assertIn("--FeatureExtraction.use_gpu 0", e)
        self.assertIn("--SiftExtraction.domain_size_pooling 0", e)
        self.assertIn("--FeatureMatching.type SIFT_LIGHTGLUE", m)
        self.assertIn("--SiftMatching.lightglue_model_path /c/sift-lightglue.onnx", m)

    def test_no_banned_tokens_anywhere(self):
        text = " ".join(" ".join(c) for c in cmds.all_example_commands(self.cfg)).lower()
        for banned in ("siftgpu", "superpoint", "superglue"):
            self.assertNotIn(banned, text)

    def test_no_glog_files_dropped_in_cwd(self):
        for c in cmds.all_example_commands(self.cfg):
            self.assertIn("--log_target stderr", self.flat(c))

    def test_filter_flags_match_the_plan(self):
        f = self.flat(cmds.point_filtering("a", "b"))
        for want in ("--min_track_len 3", "--max_reproj_error 2", "--min_tri_angle 2"):
            self.assertIn(want, f)
        u = self.flat(cmds.image_undistorter("img", "m", "out"))
        self.assertIn("--output_type COLMAP", u)
        self.assertIn("--max_image_size 1920", u)

    def test_global_mapper_cpu_fallback_flags(self):
        self.assertIn("--GlobalMapper.gp_use_gpu 0", self.flat(cmds.global_mapper("db", "i", "o", 4, gpu=False)))
        self.assertIn("--GlobalMapper.gp_use_gpu 1", self.flat(cmds.global_mapper("db", "i", "o", 4, gpu=True)))


class TestCheckFlags(unittest.TestCase):
    HELP = """  -h [ --help ]
  --database_path arg
  --FeatureExtraction.type arg (=SIFT)
"""

    def fake(self, d, text):
        p = os.path.join(d, "colmap")
        with open(p, "w") as f:
            f.write("#!/bin/sh\ncat <<'EOF'\n" + text + "EOF\n")
        os.chmod(p, 0o755)
        return p

    def test_known_flags_parses_help(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(check_flags.known_flags("x", self.fake(d, self.HELP)),
                             {"help", "database_path", "FeatureExtraction.type"})

    def test_unknown_flag_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            colmap = self.fake(d, self.HELP)
            problems = check_flags.check([[colmap, "feature_extractor", "--database_path", "x", "--Bogus.flag", "1"]])
            self.assertTrue(any("Bogus.flag" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
