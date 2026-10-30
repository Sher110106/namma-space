# pipeline/sfm notes (W1, 30 Oct 2026)

Stage `python -m pipeline.sfm --run <run_dir>`: reads `10_frames/images` (and `frames.json` for times and blur scores),
writes `20_sfm/` and `30_undist/`. Stdlib Python 3 around the `colmap` CLI. It runs in **`namma-colmap:0.2`**
(`0.1` has no Python). Root's stage table must change the image name for this stage.

```
nsdocker --gpu ns-<run>-sfm namma-colmap:0.2 python3 -m pipeline.sfm --run /w/runs/<run>
```

## What it does

1. Preflight (exit 20 on failure): `colmap` on PATH, at least 5 images, at least 15 GB free, every cache file the chosen variants need.
2. Race. A and B start together; D starts when A's matching ends (it copies A's database with SQLite's backup API, then
   `view_graph_calibrator` and `global_mapper`). Threads per variant: CPU quota / 3 (8 on the 24-CPU container); B's CPU SIFT
   extraction gets half the quota. Each variant has a time limit (`--variant-timeout`, default 2400 s). A failed GPU step is retried once
   (D's retry runs `global_mapper` on the CPU). A variant that crashes, times out or is killed gets a status and the race goes on.
3. Per variant: pick the largest model, run the **camera-trajectory filter**, `image_deleter` on the dropped cameras, then
   `point_filtering --min_track_len 3 --max_reproj_error 2 --min_tri_angle 2`, then `model_analyzer` on the clean model.
   The numbers in `sfm.json` (`reg_frac`, `reproj_px`, `n_pts`, `track_len`) are the post-filter ones; `*_raw` keeps the pre-filter ones.
4. C (exhaustive matching on A's features) runs only if no variant reached 90% and there are at most 350 frames.
5. Choice rule (FINAL_PLAN 2.3, `choose.py`): drop crashed variants and any under 60%; among those at 95% or more take the lowest
   reprojection error, then most points; otherwise take the highest share and mark `partial`. Variants that pass the sanity checks
   (mean track length at least 3, no jump over 5x the median step left) rank above those that fail them.
6. `image_undistorter` (PINHOLE, `--max_image_size 1920`) into `30_undist/`; checks every camera is PINHOLE. Binary models only.
7. Writes `20_sfm/sfm.json`. Temp dirs `20_sfm.tmp` and `30_undist.tmp` are renamed at the end; `30_undist` first, `sfm.json` last.

### Exit codes
- `0`: usable result. A partial one (best variant under 95%) is flagged `"partial": true` in `sfm.json` and printed as `PARTIAL`.
  Root should turn `partial` into pipeline exit 11 when it reads the JSON.
- `10`: no variant registered 60% of the frames in one model. The reshoot ranges of the best model are printed and written to
  `missing_ranges`; `chosen` is null; no `30_undist`.
- `20`: infrastructure (no colmap, too few images, low disk, missing cache file, every variant crashed, undistort failed).

### `30_undist` layout
`images/` and `sparse/*.bin` (COLMAP 4.2.1 writes `cameras.bin frames.bin images.bin points3D.bin rigs.bin`). A real copy of the
same files sits in `sparse/0/`, so readers that want either `sparse/` or `sparse/0/` work (gsplat's parser reads `sparse/` first).

### `sfm.json`
Keys: `schema`, `colmap`, `n_images`, `variants[]` (`name, desc, status, error, time_s, steps_s, reg_frac, reg_frac_raw, n_registered,
n_registered_raw, n_models, model_sizes, reproj_px, reproj_px_raw, n_pts, n_pts_raw, track_len, n_dropped, max_step_ratio,
jumps_left, reuses`), `chosen`, `why`, `partial`, `warnings[]`, `missing_ranges[]` (runs of 3 or more missing frames:
`first, last, start_idx, end_idx, n_frames, t_start, t_end`), `unregistered[]` (every missing frame: `name, idx, t, sharp,
n_features, reason`), `dropped_cameras[]` (`name, idx, reason, ratio`), `undist{}`, `time_s`, `flags{}`. `t` and `sharp` come from
`10_frames/frames.json` `kept[]` when present. D's `time_s` counts only D's own steps; it reuses A's features and matches.

## The camera-trajectory filter (`sanity.py`)

Rule 1: drop a camera farther from the median camera centre than 6x the median such distance. Rule 2: a jump is a step between
consecutive registered frames larger than 5x the median step (steps are divided by the frame-index gap, so a stretch of
unregistered frames is not a jump); a run of at most max(3, 5% of cameras) cameras between jumps is dropped. Long runs between jumps
are kept and reported in `jumps_left`. Up to 4 rounds.

**Check on the real smoke model** (`runs/smoke_honka/sparse/0`, 290 registered; no run, read-only, filter on the host with
`pipeline.sfm.colmapio`): rule 1 drops exactly frames 134-140 (298x the median distance), the set the smoke test found by hand.
Rule 2 drops 15 more: frames 248-261 and 335. Scored against the iPhone (ARKit/Polycam) poses in `transformations.json`
after a similarity fit on the kept cameras: all 22 dropped cameras are off by 1.07 m or more (median 1.33 m); the 268 kept cameras
have median error 0.086 m, 95th percentile 0.26 m. No dropped camera has an error under 0.3 m. In the ground truth, consecutive frames
there are 0.06-0.10 m apart (median step 0.14 m, maximum 0.38 m), so the 3-unit jumps in the COLMAP model are registration errors.
Miss: frames 1-12 stay in the model with 1.1-1.3 m error against the ARKit poses. They form no jump, so the filter cannot see
them. ARKit poses drift too, so we do not know which side is wrong. Scale of the COLMAP model to ARKit: 0.499 (arbitrary).

## Commands for tests

- Unit tests (Mac, stdlib only, about 40 s): `bash test/unit/sfm/run.sh`. They use a fake `colmap` (`fake_colmap.py`) to test
  orchestration: crash, timeout, partial, exit 10, variant C, dropped cameras, missing ranges, preflight, disk gate.
- Acceptance tests (tyrone): `cd /home/sher/namma-space/repo && bash test/unit/sfm/accept_tyrone.sh <a1|a2|a3|a4|a5|a6|c1|c2|c3a|c3a2|c3b|c3c|c4|c5>`.
- C4 alone: `python3 -m pipeline.sfm.check_flags` inside `namma-colmap:0.2`.
