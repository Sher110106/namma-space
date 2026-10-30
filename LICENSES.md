# Namma Space licences and credits

Generated from `docs/licenses.yaml`. Verdicts describe this submission's use.

## Product path

| Component | Version | Licence | Role | Verdict | Source |
|---|---|---|---|---|---|
| COLMAP | 4.2.1 | BSD-3-Clause (core) | SfM and camera poses | allowed; SiftGPU and LSD are never called | [https://github.com/colmap/colmap](https://github.com/colmap/colmap) |
| ONNX Runtime | bundled with COLMAP | MIT | runs ALIKED and LightGlue | allowed | [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| ALIKED N32 ONNX | COLMAP release asset | BSD-3-Clause | image features | allowed | [https://github.com/colmap/colmap](https://github.com/colmap/colmap) |
| LightGlue ONNX (ALIKED and SIFT) | COLMAP release assets | Apache-2.0 | feature matching | allowed; SuperPoint is not used | [https://github.com/colmap/colmap](https://github.com/colmap/colmap) |
| COLMAP vocabulary trees | ALIKED-N32 and SIFT | not separately stated | loop detection | allowed as project assets; unverified separate notice | [https://github.com/colmap/colmap/releases](https://github.com/colmap/colmap/releases) |
| PyTorch | 2.4.1 | BSD-3-Clause | training runtime | allowed | [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) |
| gsplat | 1.5.3 | Apache-2.0 | Gaussian splat training and rendering | allowed | [https://github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) |
| fused-ssim | git 328dc98 | MIT | training loss helper | allowed | [https://github.com/rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) |
| fused-bilagrid | git 90f9788 | MIT | bilateral grid for exposure changes (gsplat example dependency) | allowed | [https://github.com/harry7557558/fused-bilagrid](https://github.com/harry7557558/fused-bilagrid) |
| torchmetrics | pinned image version | Apache-2.0 | LPIPS evaluation | allowed; weights are eval-only | [https://github.com/Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics) |
| LPIPS AlexNet ImageNet weights | torchmetrics checkpoint | weight terms unverified | evaluation metric only | unverified; outside the product path | [https://github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) |
| tyro | image-pinned | MIT | gsplat example dependency (trainer CLI, unused viewer, COLMAP model reader) | allowed; licence read from package metadata in namma-space-gpu:0.2 | [https://github.com/brentyi/tyro](https://github.com/brentyi/tyro) |
| nerfview | image-pinned | MIT | gsplat example dependency (trainer CLI, unused viewer, COLMAP model reader) | allowed; licence read from package metadata in namma-space-gpu:0.2 | [https://github.com/nerfstudio-project/nerfview](https://github.com/nerfstudio-project/nerfview) |
| viser | image-pinned | MIT | gsplat example dependency (trainer CLI, unused viewer, COLMAP model reader) | allowed; licence read from package metadata in namma-space-gpu:0.2 | [https://github.com/nerfstudio-project/viser](https://github.com/nerfstudio-project/viser) |
| pycolmap | image-pinned | MIT (fork rmbrualla/pycolmap, package metadata) | gsplat example dependency (trainer CLI, unused viewer, COLMAP model reader) | allowed; licence read from package metadata in namma-space-gpu:0.2 | [https://github.com/rmbrualla/pycolmap](https://github.com/rmbrualla/pycolmap) |
| moge | 2.0.0 code; moge-2-vitl-normal weights (2026-09) | MIT; DINOv2 parts Apache-2.0 | optional metric depth cue | allowed; training-data terms unverified | [https://github.com/microsoft/MoGe](https://github.com/microsoft/MoGe) |
| Open3D | 0.20.0 | MIT | geometry and free-space checks | allowed | [https://github.com/isl-org/Open3D](https://github.com/isl-org/Open3D) |
| numpy | image-pinned | BSD-3-Clause | geometry, grid and file IO | allowed | [https://github.com/numpy/numpy](https://github.com/numpy/numpy) |
| scipy | image-pinned | BSD-3-Clause | geometry, grid and file IO | allowed | [https://github.com/scipy/scipy](https://github.com/scipy/scipy) |
| scikit-image | image-pinned | BSD-3-Clause | geometry, grid and file IO | allowed | [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) |
| scikit-learn | image-pinned | BSD-3-Clause | geometry, grid and file IO | allowed | [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| shapely | image-pinned | BSD-3-Clause | geometry, grid and file IO | allowed | [https://github.com/shapely/shapely](https://github.com/shapely/shapely) |
| trimesh | image-pinned | MIT | geometry, grid and file IO | allowed | [https://github.com/mikedh/trimesh](https://github.com/mikedh/trimesh) |
| opencv-python-headless | 4.10.0.84 | Apache-2.0 | frame checks and image IO | allowed; headless wheel, no non-free modules | [https://github.com/opencv/opencv](https://github.com/opencv/opencv) |
| ffmpeg | Ubuntu build | GPL-2+ binary, separate process | decode, frame selection and video encode | copyleft allowed; not linked into product | [https://ffmpeg.org/legal.html](https://ffmpeg.org/legal.html) |
| Node.js | 22 | MIT | SPZ export tool runtime | allowed | [https://nodejs.org/](https://nodejs.org/) |
| splat-transform | 3.10.3 | MIT | PLY to SPZ export | allowed | [https://github.com/playcanvas/splat-transform](https://github.com/playcanvas/splat-transform) |
| SPZ format | 3/4 | MIT | compact web splat asset | allowed | [https://github.com/nianticlabs/spz](https://github.com/nianticlabs/spz) |
| Spark | 2.4.0 | MIT | browser splat renderer | allowed | [https://github.com/sparkjsdev/spark](https://github.com/sparkjsdev/spark) |
| three.js | r180 (vendored build) | MIT | browser camera and controls | allowed | [https://github.com/mrdoob/three.js](https://github.com/mrdoob/three.js) |
| Playwright headless Chrome | cached | Apache-2.0 / BSD components | PDF, screenshot and viewer checks | dev tool only | [https://playwright.dev/](https://playwright.dev/) |

## Evaluation data and credits

| Component | Version | Licence | Role | Verdict | Source |
|---|---|---|---|---|---|
| MuSHRoom | dataset release | CC BY 4.0 (Zenodo records 10230733, 10222321) | public reconstruction test data | credited; not user data | [https://zenodo.org/records/10230733](https://zenodo.org/records/10230733) |
| Mip-NeRF 360 room | dataset release | CC BY 4.0 (stated on the SMERF data page; medium confidence) | public evaluation data | credited | [https://smerf-3d.github.io/#data](https://smerf-3d.github.io/#data) |
| Zip-NeRF Berlin | dataset release | CC BY 4.0 (stated on the SMERF data page; medium confidence) | public evaluation data | credited; used for internal tests only; never hosted (private home) | [https://smerf-3d.github.io/#data](https://smerf-3d.github.io/#data) |

## System libraries

| Component | Version | Licence | Role | Verdict | Source |
|---|---|---|---|---|---|
| NVIDIA driver, CUDA 12.x runtime, cuDNN 9 | host/image supplied | NVIDIA proprietary system libraries | GPU execution | system dependency; not bundled by our code | [https://developer.nvidia.com/cuda-eula](https://developer.nvidia.com/cuda-eula) |

## Installed in the GPU image but not used by our code

| Component | Version | Licence | Role | Verdict | Source |
|---|---|---|---|---|---|
| plyfile |  | GPL-3.0-or-later | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | copyleft; installed by our image recipe but unused (our code reads PLY with numpy); remove from the next image | [https://github.com/dranjan/python-plyfile](https://github.com/dranjan/python-plyfile) |
| hypothesis |  | MPL-2.0 | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | weak copyleft test tool; unused | [https://github.com/HypothesisWorks/hypothesis](https://github.com/HypothesisWorks/hypothesis) |
| conda-build |  | BSD-3-Clause | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; unused | [https://github.com/conda/conda-build](https://github.com/conda/conda-build) |
| mamba |  | BSD-3-Clause | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; unused | [https://github.com/mamba-org/mamba](https://github.com/mamba-org/mamba) |
| lief |  | Apache-2.0 | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; unused | [https://github.com/lief-project/LIEF](https://github.com/lief-project/LIEF) |
| lintrunner |  | BSD-3-Clause | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; unused | [https://github.com/suo/lintrunner](https://github.com/suo/lintrunner) |
| torchaudio |  | BSD-2-Clause | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; unused | [https://github.com/pytorch/audio](https://github.com/pytorch/audio) |
| torchelastic |  | BSD-3-Clause | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; unused | [https://github.com/pytorch/elastic](https://github.com/pytorch/elastic) |
| tensorboard |  | Apache-2.0 | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; gsplat example logging | [https://github.com/tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) |
| tensorly |  | BSD-3-Clause | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; gsplat example dependency | [https://github.com/tensorly/tensorly](https://github.com/tensorly/tensorly) |
| splines |  | MIT | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; gsplat example dependency | [https://github.com/AudioSceneDescriptionFormat/splines](https://github.com/AudioSceneDescriptionFormat/splines) |
| torch-fidelity |  | Apache-2.0 | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; unused | [https://github.com/toshas/torch-fidelity](https://github.com/toshas/torch-fidelity) |
| imageio-ffmpeg |  | BSD-2-Clause | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; bundles an ffmpeg binary run as a separate process | [https://github.com/imageio/imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |
| astunparse, brotli, cffi, colorama, exceptiongroup, expecttest, h2, importlib-resources, optree, pickleshare, pip, pkgutil-resolve-name, pysocks, types-dataclasses |  | BSD, MIT or Apache-2.0 (package metadata) | ships with the PyTorch base image or as a transitive dependency; not imported by Namma Space code | allowed; base-image helpers, unused | [https://pypi.org/](https://pypi.org/) |

## Rejected or dropped

| Item | Reason |
|---|---|
| SuperPoint, SuperGlue, superpoint_lightglue | organizer ruling excludes these non-commercial components |
| Inria 3DGS and forks (2DGS, Mip-Splatting, PGSR, Taming-3DGS) | Inria Gaussian-Splatting licence: research and evaluation only, non-commercial |
| DUSt3R, MASt3R, MASt3R-SLAM | CC BY-NC-SA 4.0 code and weights (non-commercial) |
| VGGT-1B and VGGT-Omega | custom or unclear licence |
| Pi3/Pi3X, Depth Pro, UniDepth, DSINE, Omnidata, Metric3D weights | weight licence is non-commercial or not stated |
| Depth Anything 3 Large/Giant/Nested and DA-V2 Base/Large | weight licence or model terms do not fit this submission |
| SAM 3 and VGGT-1B-Commercial | custom licence, not OSI-approved |
| Ultralytics YOLO | AGPL dependency is unnecessary for the Round 1 product path |
| LoMa weights and hloc r2d2/superpoint/superglue modules | unverified or non-commercial weights |

## Disclosures

- The stock COLMAP CUDA image compiles SiftGPU and LSD. Namma Space never calls either path: feature extraction uses ALIKED or CPU SIFT, and alignment does not use the LSD-based Manhattan aligner.
- NVIDIA driver, CUDA and cuDNN are proprietary system libraries supplied by the host or image. They are execution dependencies, not code shipped in this repository.

## Repository licence

The repository code is Apache-2.0; see `LICENSE`.
