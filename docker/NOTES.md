# docker/ notes (W1, 30 Oct 2026)

Commands run, numbers measured, failures and decisions. Times are IST. Nothing here is estimated unless marked [E].

## Timeline and builds

- 03:36 `namma-space-gpu:0.2` build started from `gpu-0.2.Dockerfile`; finished in about 2 minutes (layers: Node 22 195 MB,
  splat-transform 279 MB, MoGe-2 code + utils3d + huggingface_hub 19 MB; total 0.49 GB over `0.1`).
- 03:50 `namma-colmap:0.2` built (37 MB). **Decision:** `namma-colmap:0.1` has no Python (`python3: command not found`, checked
  inside the container), so the stdlib driver `pipeline.sfm` cannot run there. The stage table must say `namma-colmap:0.2`.
  First attempt used `python3-minimal`, which lacks `sqlite3`; the recipe now installs the `python3` package.
- Disk: 45 GB free at 03:35, 40 GB at 04:02 (other builders). Our images add about 0.53 GB. Our cache: `colmap/` 182 MB, `hf/` 1.3 GB.

## MoGe: HEAD is MoGe-3, we pin MoGe-2

`git ls-remote microsoft/MoGe HEAD` is the MoGe-3 commit `74fbce05` (19 Aug 2026), which needs `numpy>=2`, FlexGEMM and gradio 6.
gsplat's image is pinned to `numpy<2`. We pin the last MoGe-2 commit, `07444410f1e33f402353b99d6ccd26bd31e469e8`
(v2.0.0, 2 Nov 2025), `--no-deps`, plus `utils3d@3fab839f` (the pin that commit asks for). `import moge, utils3d` and
`from moge.model.v2 import MoGeModel` work in the image. Weights are not baked in.

**A6 (pass).** `HF_HOME=/w/cache/hf`. First load with network: 24.7 s (downloads 1.3 GB to `cache/hf`). Reload with
`HF_HUB_OFFLINE=1`: 6.6 s on the A6000. Inference on honka long_capture frame 100 (738 x 994): outputs `points`,
`intrinsics`, `depth`, `mask`, `normal`; median depth 2.385 m (metric-scale output, not checked against truth here).
Usage in code: `MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval().infer(tensor_chw_0_1)`.
Script: `docker/checks/a6_moge.py`.

## splat-transform (A4, pass)

Pinned `@playcanvas/splat-transform` 3.10.3 (`splat-transform --version` prints `v3.10.3 (4f90012)`), Node 22.23.4.
Test file: 100,000 splats cut at random (seed 0) from `/w/runs/smoke_honka/gs_out5/ply/point_cloud_29999.ply`
(1,000,000 splats, 59 PLY fields, SH3). Exact flags:

```
splat-transform -w cut100k.ply -H 1 --spz-version 4 out_sh1.spz     # 100K gaussians, 1 SH band, 1,900,154 bytes (1.8 MB)
splat-transform -w out_sh1.spz back.ply                              # 100K gaussians, 1 SH band (f_rest = 9)
splat-transform -w cut100k.ply -H 1 --spz-version 3 out_sh1_v3.spz   # 1,965,320 bytes; round trip also 100,000
splat-transform out_sh1.spz --info null                              # format: spz, gaussians: 100000, sh bands: 1
```

- `-H`/`--filter-harmonics <0|1|2|3>` is what drops SH bands; SPZ output defaults to version 4 (`--spz-version <3|4>`).
- Count before = count after = 100,000 in both versions. 0.6 s and 0.4 s on one core.
- Other flags worth knowing for export (from `--help`): `-N` filter NaN, `-V name,cmp,value` (e.g. `opacity,gt,...`), `-B` box filter,
  `-d n%` decimate, `-m` Morton order, `--stats`, `--info`. Not needed for the round trip.
- "Offline": splat-transform does not touch the network for PLY/SPZ conversion (no URL inputs used). We could not run the
  container with `--network none` (see next section); the container had network during the test.
- Script: `docker/checks/a4_roundtrip.sh` (uses `cut_ply.py`, `count_ply.py`).

## `--network none` and the model cache (A1)

`nsdocker` takes only `[--gpu] NAME IMAGE CMD...`; it has no hook for extra `docker run` arguments, and `unshare -n` inside the
container fails (`Operation not permitted`, no CAP_SYS_ADMIN). We did not bypass `nsdocker`. So the literal `--network none` run
was **not done**. What we did instead, which proves the runner never needs the network:

- COLMAP 4.2.1 accepts a plain local path in every `*_model_path` option (the defaults are `url;name;sha256` strings that it
  downloads and caches under `$XDG_CACHE_HOME/colmap/<sha256>-<name>`). `pipeline.sfm` always passes explicit local paths:
  `--AlikedExtraction.n32_model_path`, `--AlikedMatching.lightglue_model_path`, `--SiftMatching.lightglue_model_path`,
  `--SequentialMatching.vocab_tree_path` (all under `/w/cache/colmap`).
- `docker/checks/a1_offline.sh` also points `XDG_CACHE_HOME` at an empty directory, so any implicit download would have to hit
  the network; the logs contain **0** "Downloading file" lines.
- The three ONNX files have the same SHA-256 as the defaults printed by `colmap ... -h` (listed in `IMAGES.md`).
- Suggested change for root: add `${NS_DOCKER_ARGS:-}` to the `docker run` line in `nsdocker`, then A1 can be repeated with
  `NS_DOCKER_ARGS="--network none"`. Until then A1 is "offline by construction", not "offline by network namespace".

## ALIKED loop detection (task item 2)

An ALIKED vocabulary tree exists: COLMAP release 3.13.0 ships `vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin` (18.8 MB,
also `..._aliked_n16rot.bin`, same size). Cached under `cache/colmap/`. SIFT: `vocab_tree_faiss_flickr100K_words256K.bin`
(72 MB, release 3.11.1). No licence file accompanies them [U].

It works: on the honka long capture (335 frames) with `--SequentialMatching.loop_detection 1` and the ALIKED tree, the log shows
"Generating image pairs with vocabulary tree...", indexes all 335 images and queries 34 of them. In the resulting database
**642 of the 1,885 geometrically verified pairs link frames more than 25 apart**, which sequential overlap 20 + quadratic
cannot reach, so they come from loop detection. That run registered 316/335 (94.3%) in one model at 0.82 px, against 290/335
(87%, 0.75 px) in the smoke test without loop detection. We did not run the control (same settings, loop detection off), and the
mapper is not seeded, so we do not claim the 7-point gain is caused by loop detection.
