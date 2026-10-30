# Evaluation evidence

This folder holds the checks behind our numbers: how close phone-only geometry, scale and depth come to the public MuSHRoom reference data, whether our trainer matches published image scores on Mip-NeRF 360, and a code check of the fusion step. Every figure and table comes from the JSON files in `results/`.

Redraw all figures: `python eval/make_figures.py` (needs matplotlib). Print the tables below: `python eval/make_figures.py --table`.

## Results

<!-- table:start -->
Geometry against the Faro scan (cropped). Ours is scored after the 7-DoF fit.

| Room | Method | F@5cm | Precision | Recall | Accuracy (m) | Completeness (m) | F@5cm ours / LiDAR |
|---|---|---:|---:|---:|---:|---:|---:|
| honka | Phone-only (ours) | 0.519 | 0.460 | 0.594 | 0.113 | 0.057 | 64.6% |
| honka | iPhone LiDAR fusion | 0.803 | 0.723 | 0.902 | 0.119 | 0.031 |  |
| vr_room | Phone-only (ours) | 0.401 | 0.336 | 0.499 | 0.215 | 0.091 | 54.5% |
| vr_room | iPhone LiDAR fusion | 0.737 | 0.657 | 0.838 | 0.171 | 0.039 |  |

Scale (automatic, before the fit) and depth against iPhone LiDAR depth (phone-only).

| Room | Scale error | True scale in 68% interval | Fit scale | Depth median error | Within 5 cm | Within 10 cm | Held-out views |
|---|---:|:---:|---:|---:|---:|---:|---:|
| honka | 4.29% | yes | 0.959 | 6.08 cm | 43.9% | 65.6% | 40 |
| vr_room | 3.10% | yes | 1.032 | 6.77 cm | 41.1% | 61.7% | 53 |

Other checks.

| Check | Result |
|---|---|
| Mip-NeRF 360 room, held-out PSNR | ours 32.48 dB, published 31.36 dB, difference +1.12 dB |
| Mip-NeRF 360 room, SSIM / LPIPS (AlexNet) | 0.938 / 0.132 |
| Synthetic room, median wall error | 0.0036 m (pass limit 0.02 m, status pass) |

These rows score the current default pipeline (variant r0) on runs fix-honka and fix-vrroom (30 Oct, after the review fixes). An earlier honka run (rh-honka, older code) shipped the depth-loss variant r1 and scored F@5cm 0.630 (78.5% of LiDAR), median depth error 4.91 cm and scale error 3.26%: depth loss improves geometry at equal image quality. r1 is available with `--variants r0,r1`; it crashed once on vr_room (CUDA assert), in which case the runner ships r0.
<!-- table:end -->

Figures: `mushroom_geometry.png`, `mushroom_scale.png`, `mip360_room.png`, plus one card per JSON.

## Read this beside the geometry and depth numbers

Before scoring, we fitted our COLMAP camera centres to the dataset's camera path with a robust 7-DoF similarity (Umeyama: rotation, offset and scale; `score_phone_only.py`). The fit scale is in the table (`scale.fit_scale_aligned_to_dataset`). The fit removes our own scale error from the geometry and depth scores. So those scores are a best case for shape. They are not what the pipeline delivers end to end.

Our scale error without the fit is the "Scale error" column: it comes from camera height alone and is measured before any fit. The "True scale in 68% interval" column says whether the dataset's true scale fell inside the interval the pipeline reported.

The LiDAR baseline uses the dataset's own poses and no scale fit.

## Not claimed

- No claim that phone-only geometry is close to LiDAR. `docs/claims.md` (row 2) sets a rule: ours / LiDAR F@5cm must reach 80%. Ours reaches the ratios in the table. The rule fails in both rooms.
- `vr_room` is not an unseen room. Three code fixes came from working on it.
- No claim of photo-like quality without held-out image scores beside it.
- Browser speed, splat counts and file sizes are not in this folder. They sit in each run's `90_verify/verify.json`.
- The Mip-NeRF 360 result shows our trainer matches the published figure. It says nothing about phone video.

## How each result was made

Both methods fuse depth with Open3D `UniformTSDFVolume` (voxel 2 cm, truncation 8 cm, depth cap 5 m). Both are scored on the same crop: predicted points inside the Faro mesh bounding box grown by 5 cm. Faro points stay whole, so completeness covers the full scan. Each surface is sampled with 100,000 points. The dataset's ICP transform places both in the Faro frame.

- Phone-only (`mushroom_*_ours.json`): the pipeline's cleaned splat file is rendered to depth at our cameras and fused. Depth error uses every eighth view as held out. Commands, as recorded in the JSON `command` field (paths are inside the GPU container):
  `python -m eval.score_phone_only --run /w/runs/rh-honka --dataset /w/data/mushroom/room_datasets/honka --out /w/runs/w7-honka/mushroom_honka_ours.json`
  and the same with `rh-vrroom`, `vr_room`, `w7-vrroom`.
- iPhone LiDAR fusion (`mushroom_*_lidar.json`): the dataset's iPhone depth and poses, every view (stride 1).
  `python -m eval.score_mushroom --dataset /w/data/mushroom/room_datasets/honka --out /w/runs/w7-honka/mushroom_honka_lidar_uniform2.json`
  The `command` field names `*_lidar_uniform2.json`. The shipped files are `*_lidar.json`. The names differ; the field is a slip in the file name.
- Mip-NeRF 360 room (`mip360_room.json`): the gsplat trainer with MCMC defaults, 30,000 steps, images at half size, every eighth image held out. The JSON holds the exact command. The JSON does not name the source of the published figure.
- Synthetic room (`synthetic_room.json`): four exact depth views of a box room, fused with the same settings. It checks the fusion code. The median distance from mesh vertices to the true walls must stay under the pass limit. It is run as `python -m eval.synthetic_room --out eval/results/synthetic_room.json` (the JSON stores no command).

## Superseded and known flaws

- Zip-NeRF Berlin (J6, a multi-room SfM stress test) was never run. Early `not_run` stub files were removed on 30 Oct; the `*_ours` and `*_lidar` files hold the MuSHRoom results.
- `fallback_point_surface` inside the MuSHRoom JSONs holds the older scores from a point-surface fallback, used when Open3D fusion returned an empty mesh. They are superseded. Use the headline `geometry.cropped` and `score.cropped` values only.
- The four MuSHRoom JSONs show `"git": "unknown"`. Cause: `score_phone_only.py` reads the commit only when `run/../.git` exists, and `score_mushroom.py` hides a failed `git rev-parse` inside the container. The scores were made just before commit `3346c14`, which added these files. `mip360_room.json` records its commit.
- Container images: the Mip-NeRF 360 run used `namma-space-gpu:0.1`; the MuSHRoom scoring used `namma-space-gpu:0.2`. The JSONs do not record the image.
- The accuracy mean sits well above its median; the medians are in the JSON and on the cards.
- Earlier PNGs said "tyrone had no usable NVIDIA driver" and showed fallback scores. They are replaced.

## Files in `results/`

- `mushroom_honka_ours.json`, `.png`: phone-only scores, honka.
- `mushroom_vr_room_ours.json`, `.png`: phone-only scores, vr_room.
- `mushroom_honka_lidar.json`, `.png`: iPhone LiDAR fusion baseline, honka.
- `mushroom_vr_room_lidar.json`, `.png`: iPhone LiDAR fusion baseline, vr_room.
- `mushroom_geometry.png`: ours against LiDAR, both rooms (drawn from the four JSONs above).
- `mushroom_scale.png`: scale error and depth error, both rooms (from the two `_ours` JSONs).
- `mip360_room.json`, `.png`: trainer check on Mip-NeRF 360 room.
- `synthetic_room.json`, `.png`: fusion code check.
