# Docker images (tyrone)

Written by W1, 30 Oct 2026. Local images have no registry digest; the pinned identity is the image ID (the
sha256 of the image config). Base images are pinned by registry digest. Recipes sit next to this file.

| Image | Image ID (sha256) | Size on disk | Added over parent | Recipe |
|---|---|---|---|---|
| `colmap/colmap:latest` (pulled) | `b4f32886029a72b20df1410bd85743554f8c1feb7d05b23bd427677c5e103e9f`; registry digest `colmap/colmap@sha256:c1de51615dff33066cd305071226b6343c1168324ef3bd622f0825e55feb5efe` | 5.78 GB | n/a | upstream |
| `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` (pulled) | `a50d89ca31b2aaddabf9c5c367c20f34f04033de8f382a1d2728d66b67ced9ea`; registry digest `pytorch/pytorch@sha256:9859f8978cdfad549d72baa41d0b0bb7a5b46210a1446e09bf32600a968badb8` | 13.1 GB | n/a | upstream |
| `namma-colmap:0.1` | `6b5593a4a3b7d51d1d09d1cbb30282818d9a9993de1f5712225c27b17637f0f6` | 7.00 GB | 1.2 GB (cuDNN 9) | `colmap.Dockerfile` |
| `namma-colmap:0.2` | `7857dc32cbb6875510c38064a325f7a46b8573852d13d3948bf5a7e0adcec50a` | 7.03 GB | 37 MB (python3, distro stdlib only) | `colmap-0.2.Dockerfile` |
| `namma-space-gpu:0.1` | `8b54e7911d38654dbf7c9ff4f893489d48c0a49980553e44cdf152c6ddc375a3` | 15.69 GB | PyTorch 2.4.1 + gsplat 1.5.3 + geometry libs | `gpu.Dockerfile` |
| `namma-space-gpu:0.2` | `fc9768a1d61aa59dc2f515b1aecdcfb8169530ce000f5f84a687c6eb5b7d033f` | 16.18 GB | 0.49 GB (Node 195 MB, splat-transform 279 MB, MoGe-2 code 19 MB) | `gpu-0.2.Dockerfile` |

New layers in total: about 0.53 GB (plan A5 allows 6 GB). `namma-node:0.1` was not built: `namma-space-gpu:0.2` already
carries Node 22 and splat-transform (A4 runs in it).

## Which image runs which stage

| Stage | Image | Why |
|---|---|---|
| `pipeline.sfm` | **`namma-colmap:0.2`** | `0.1` has no Python at all; the driver is stdlib Python 3 around the `colmap` CLI |
| ingest, align, train, geometry, grid | `namma-space-gpu:0.1` (or `0.2`) | unchanged |
| export | `namma-space-gpu:0.2` | Node 22 + `splat-transform` |
| align, MoGe cue | `namma-space-gpu:0.2` | `moge` + cached weights |

## Pinned versions

| Component | Version / pin | Where |
|---|---|---|
| COLMAP | 4.2.1 (CUDA 12.9.1, cuDNN 9) | `colmap/colmap:latest`, image ID above |
| Python in `namma-colmap:0.2` | 3.12.3 (Ubuntu 24.04), SQLite 3.45.1 | apt |
| Node.js | 22.23.4, tarball SHA-256 `b8492284c76a54baf58d61668dd0e41ee0fc98adabf9503d30f05d08f55b4406` | `gpu-0.2.Dockerfile` |
| `@playcanvas/splat-transform` | 3.10.3 (npm `latest` on 30 Oct 2026; build `4f90012`) | `gpu-0.2.Dockerfile` |
| MoGe code (v2.0.0) | `microsoft/MoGe@07444410f1e33f402353b99d6ccd26bd31e469e8` | `gpu-0.2.Dockerfile` |
| `utils3d` | `EasternJournalist/utils3d@3fab839f0be9931dac7c8488eb0e1600c236e183` (the pin MoGe v2.0.0 asks for) | `gpu-0.2.Dockerfile` |
| `huggingface_hub` | 0.36.2 (`<1`) | `gpu-0.2.Dockerfile` |
| MoGe-2 weights | `Ruicheng/moge-2-vitl-normal`, HF revision `cb0e8bbd6b1e243589717c78e750b1ba4c093acf`, `model.pt` 1.3 GB | host cache `cache/hf` (not in the image) |

MoGe: the repository HEAD is now MoGe-3 (V3, needs numpy >= 2, FlexGEMM). We pin the last MoGe-2 commit (v2.0.0, 2 Nov 2025),
which keeps `numpy<2` for gsplat. Installed with `--no-deps`; `moge 2.0.0` still lists `gradio` and `pipeline` as requirements.
Neither is needed for `MoGeModel.from_pretrained(...).infer(...)`. pip prints a conflict warning at build time; it is harmless.

## Host cache (`/home/sher/namma-space/cache`, mounted at `/w/cache`)

| Path | Files | SHA-256 | Licence |
|---|---|---|---|
| `colmap/aliked-n32.onnx` | 4.2 MB | `a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c` | BSD-3 (ALIKED), COLMAP release 3.13.0 |
| `colmap/aliked-n16rot.onnx` | 3.0 MB | `39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547` | same; unused, cached for completeness |
| `colmap/aliked-lightglue.onnx` | 45.8 MB | `b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d` | Apache-2.0 (LightGlue), release 3.13.0 |
| `colmap/sift-lightglue.onnx` | 45.8 MB | `e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e` | Apache-2.0, release 3.13.0 |
| `colmap/bruteforce-matcher.onnx` | 5 KB | `3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47` | COLMAP release 3.13.0; unused |
| `colmap/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin` | 18.8 MB | `65619481045b8f933268f10c31ad180eb1ee7881182873efe0f5753972ef6a20` | no licence file found [U]; COLMAP project asset, release 3.13.0 |
| `colmap/vocab_tree_faiss_flickr100K_words256K.bin` (SIFT) | 72.4 MB | `96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc` | no licence file found [U]; COLMAP asset, release 3.11.1 |
| `hf/hub/models--Ruicheng--moge-2-vitl-normal` | 1.3 GB | blob `280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01` | MIT (code and weights; DINOv2 parts Apache-2.0); training-data terms [U] |

The three ONNX hashes equal the ones COLMAP 4.2.1 prints as defaults in `colmap feature_extractor -h` and `colmap sequential_matcher -h`.

## Licence notes for the images

- COLMAP is BSD-3. The `colmap/colmap` image also compiles SiftGPU (educational, research and non-profit use only) and LSD (AGPL-3.0).
  `pipeline.sfm` never calls either: variant B extracts SIFT on the CPU (`--FeatureExtraction.use_gpu 0`) and matches with the
  SIFT LightGlue ONNX model. `pipeline/sfm/check_flags.py` and `test_cmds.py` assert that no command mentions `siftgpu`,
  `superpoint` or `superglue`. Disclose SiftGPU and LSD as present but unused.
- NVIDIA CUDA 12.x runtime and cuDNN 9 are proprietary, free to use and redistribute; they come from the base images
  (system libraries, like the driver). NGC container terms apply to the `nvidia/cuda` and `pytorch/pytorch` bases.
- Ubuntu 22.04 (gpu image) and 24.04 (colmap image) packages: distro licences. `ffmpeg` (GPL-2+) is a separate process.
- Node.js (MIT), `@playcanvas/splat-transform` (MIT; depends on `webgpu` 0.6.2 and `@adobe/spz` 0.2.3), MoGe (MIT; DINOv2 code under `moge/model/dinov2` is Apache-2.0, read from the repo LICENSE at the pinned commit),
  `utils3d` (MIT, GitHub licence metadata), `huggingface_hub` (Apache-2.0).
- Nothing in these images is non-commercial apart from the unused SiftGPU.
