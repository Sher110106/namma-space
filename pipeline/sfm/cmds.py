"""Every colmap command line the SfM stage runs, built in one place (so check_flags.py can verify them all).

Offline by design: model and vocabulary-tree files are passed as explicit local paths (`--*_model_path`,
`--*.vocab_tree_path`). COLMAP treats an existing local path as the file itself and never downloads it.
Variant B uses CPU SIFT (`--FeatureExtraction.use_gpu 0`) and the SIFT LightGlue ONNX matcher; SiftGPU is never called.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

COLMAP = "colmap"

# Files in the host cache (/home/sher/namma-space/cache/colmap, mounted at /w/cache/colmap).
ALIKED_N32 = "aliked-n32.onnx"
ALIKED_LIGHTGLUE = "aliked-lightglue.onnx"
SIFT_LIGHTGLUE = "sift-lightglue.onnx"
VOCAB_ALIKED = "vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin"
VOCAB_SIFT = "vocab_tree_faiss_flickr100K_words256K.bin"
REQUIRED_CACHE_FILES = {
    "A": [ALIKED_N32, ALIKED_LIGHTGLUE, VOCAB_ALIKED],
    "B": [SIFT_LIGHTGLUE, VOCAB_SIFT],
    "C": [ALIKED_LIGHTGLUE],
    "D": [],
}

COMMON = ["--log_target", "stderr", "--log_color", "0"]   # no glog files dropped into the working directory


@dataclass
class Cfg:
    cache: str = "/w/cache/colmap"
    camera_model: str = "SIMPLE_RADIAL"
    max_image_size: int = 1920
    aliked_features: int = 4096
    sift_features: int = 8192
    # Variant B extracts SIFT at most at this size (integrator, 30 Oct). COLMAP 4.2.1 stored far more SIFT keypoints
    # than max_num_features (honka: mean 2,846, max 14,925; b7 at 1920 px: mean 7,959, max 24,523), and the SIFT
    # LightGlue ONNX matcher on 1920 px frames then asked for a 6.9 GB buffer and aborted (exit -6, twice). Keypoint
    # count scales with image area. At 1280 px b7 still had a max of 19,358 keypoints and the matcher failed on a 6.0 GB
    # buffer while holding 33.5 GB of GPU memory (11:58). 1024 px is the regime that worked on honka (994 px: max 14,925
    # keypoints, B was the chosen variant); honka itself is not affected.
    sift_max_image_size: int = 1024
    overlap: int = 20
    loop_detection: int = 1
    gpu: bool = True          # False: ONNX models and global positioning on the CPU (slow; no GPU slot needed)

    def p(self, name: str) -> str:
        return os.path.join(self.cache, name)


def _c(sub: str, *args: object) -> list[str]:
    return [COLMAP, sub, *COMMON, *[str(a) for a in args]]


def extract_aliked(db: str, images: str, cfg: Cfg, threads: int) -> list[str]:
    return _c("feature_extractor",
              "--database_path", db, "--image_path", images,
              "--ImageReader.camera_model", cfg.camera_model, "--ImageReader.single_camera", 1,
              "--FeatureExtraction.type", "ALIKED_N32", "--FeatureExtraction.use_gpu", int(cfg.gpu),
              "--FeatureExtraction.max_image_size", cfg.max_image_size,
              "--FeatureExtraction.num_threads", threads,
              "--AlikedExtraction.max_num_features", cfg.aliked_features,
              "--AlikedExtraction.n32_model_path", cfg.p(ALIKED_N32))


def extract_sift_cpu(db: str, images: str, cfg: Cfg, threads: int) -> list[str]:
    return _c("feature_extractor",
              "--database_path", db, "--image_path", images,
              "--ImageReader.camera_model", cfg.camera_model, "--ImageReader.single_camera", 1,
              "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.use_gpu", 0,
              "--FeatureExtraction.max_image_size", min(cfg.max_image_size, cfg.sift_max_image_size),
              "--FeatureExtraction.num_threads", threads,
              "--SiftExtraction.max_num_features", cfg.sift_features,
              "--SiftExtraction.domain_size_pooling", 0)


def match_sequential_aliked(db: str, cfg: Cfg, threads: int) -> list[str]:
    return _c("sequential_matcher",
              "--database_path", db,
              "--FeatureMatching.type", "ALIKED_LIGHTGLUE", "--FeatureMatching.use_gpu", int(cfg.gpu),
              "--FeatureMatching.num_threads", threads,
              "--AlikedMatching.lightglue_model_path", cfg.p(ALIKED_LIGHTGLUE),
              "--SequentialMatching.overlap", cfg.overlap, "--SequentialMatching.quadratic_overlap", 1,
              "--SequentialMatching.loop_detection", cfg.loop_detection,
              "--SequentialMatching.vocab_tree_path", cfg.p(VOCAB_ALIKED),
              "--SequentialMatching.num_threads", threads)


def match_sequential_sift(db: str, cfg: Cfg, threads: int) -> list[str]:
    return _c("sequential_matcher",
              "--database_path", db,
              "--FeatureMatching.type", "SIFT_LIGHTGLUE", "--FeatureMatching.use_gpu", int(cfg.gpu),
              "--FeatureMatching.num_threads", threads,
              "--SiftMatching.lightglue_model_path", cfg.p(SIFT_LIGHTGLUE),
              "--SequentialMatching.overlap", cfg.overlap, "--SequentialMatching.quadratic_overlap", 1,
              "--SequentialMatching.loop_detection", cfg.loop_detection,
              "--SequentialMatching.vocab_tree_path", cfg.p(VOCAB_SIFT),
              "--SequentialMatching.num_threads", threads)


def match_exhaustive_aliked(db: str, cfg: Cfg, threads: int) -> list[str]:
    return _c("exhaustive_matcher",
              "--database_path", db,
              "--FeatureMatching.type", "ALIKED_LIGHTGLUE", "--FeatureMatching.use_gpu", int(cfg.gpu),
              "--FeatureMatching.num_threads", threads,
              "--AlikedMatching.lightglue_model_path", cfg.p(ALIKED_LIGHTGLUE))


def mapper(db: str, images: str, out: str, threads: int) -> list[str]:
    return _c("mapper", "--database_path", db, "--image_path", images, "--output_path", out,
              "--Mapper.num_threads", threads)


def view_graph_calibrator(db: str) -> list[str]:
    return _c("view_graph_calibrator", "--database_path", db)


def global_mapper(db: str, images: str, out: str, threads: int, gpu: bool = True) -> list[str]:
    return _c("global_mapper", "--database_path", db, "--image_path", images, "--output_path", out,
              "--GlobalMapper.num_threads", threads,
              "--GlobalMapper.gp_use_gpu", int(gpu), "--GlobalMapper.ba_ceres_use_gpu", int(gpu))


def model_analyzer(model: str) -> list[str]:
    return _c("model_analyzer", "--path", model)


def image_deleter(src: str, dst: str, names_file: str) -> list[str]:
    return _c("image_deleter", "--input_path", src, "--output_path", dst, "--image_names_path", names_file)


def point_filtering(src: str, dst: str) -> list[str]:
    return _c("point_filtering", "--input_path", src, "--output_path", dst,
              "--min_track_len", 3, "--max_reproj_error", 2, "--min_tri_angle", 2)


def image_undistorter(images: str, model: str, out: str, max_size: int = 1920) -> list[str]:
    return _c("image_undistorter", "--image_path", images, "--input_path", model, "--output_path", out,
              "--output_type", "COLMAP", "--max_image_size", max_size)


def all_example_commands(cfg: Cfg | None = None) -> list[list[str]]:
    """One instance of each command, for the C4 flag check."""
    cfg = cfg or Cfg()
    return [
        extract_aliked("db", "img", cfg, 4), extract_sift_cpu("db", "img", cfg, 4),
        match_sequential_aliked("db", cfg, 4), match_sequential_sift("db", cfg, 4),
        match_exhaustive_aliked("db", cfg, 4),
        mapper("db", "img", "out", 4), view_graph_calibrator("db"), global_mapper("db", "img", "out", 4),
        model_analyzer("m"), image_deleter("a", "b", "names.txt"), point_filtering("a", "b"),
        image_undistorter("img", "m", "out"),
    ]
