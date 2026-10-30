# WP2 align: notes (W2, 30 Oct 2026)

Stage `python -m pipeline.align --run <run>` (image `namma-space-gpu:0.1`, CPU only, numpy + matplotlib).
Reads `30_undist/{images, sparse}` (binary COLMAP model, PINHOLE; `sparse/` or `sparse/0`) and `input/meta.json` (`height_cm`, optional).
Writes `40_align/{sparse/0, images -> ../30_undist/images, align.json, figures/level_before_after.png}`.
`python -m pipeline.align.rescale --run <run> ...` is the `ns rescale` back end (see the end of this file).

## What it does

1. **Up.** Three cues, fused as a weighted mean of unit vectors.
   - cameras (weight 1): trimmed mean of the cameras' image-down axes (trim 20%, 6 passes), negated.
   - planes (weight 2): sequential RANSAC for horizontal planes (normal within 22 deg of the camera up, up to 8 planes, 2% of points each). The cue is the mean of the normals of the planes with at least 25% of the largest support, each weighted by `support / rms^2`, so tight, well-supported planes count most.
   - walls (weight 1, or 0): second RANSAC pass for vertical planes about the refined up (tolerance 14 deg, up to 8). The cue is the smallest eigenvector of `sum w n n^T`, `w = support / rms^2`, with walls trimmed at max(3 deg, 1.5 x median residual). It is used only when the mean residual is at most 3 deg **and** the conditioning (second / largest eigenvalue) is at least 0.25, that is when the walls run in two directions. Otherwise weight 0 and `up.cues.walls` keeps the vector for the record.
   - No LSD, no COLMAP Manhattan aligner.
   - `up_resid_deg` = largest angle between a used cue and the fused up. `cue_spread_deg` = largest angle between two used cues; above 5 deg the run is flagged (gate).
2. **Floor.** Lowest large horizontal plane with the cameras above it (FINAL_PLAN 2.4), cross-checked against where the walls end (see "Floor: what went wrong first"). `floor.method` says `ransac_plane` (agrees with the walls), `wall_extent`, or `point_height_percentile_2` (last resort).
3. **H_cap.** Median camera height above the floor over frames pitched within +-20 deg of level (all frames if fewer than max(10, 10%)).
4. **Scale.** Log-space inverse-variance fusion with a 3-sigma outlier drop. Cue `camera_height`: `0.80 x user height / H_cap` (sigma_log 0.08) or `1.40 m / H_cap` (0.12). The MoGe-2 cue (`moge_cue.py`, `--moge`) is optional. `scale.pct_68 = 100 (exp(sigma_log) - 1)`, and `scale.interval_68 = [s e^-sigma, s e^sigma]`.
5. **Yaw.** Wall normals projected on the horizontal plane, angles mod 90 deg averaged on the 4-theta circle (support weighted). If at least 2 walls and concentration >= 0.8 the best-supported wall's in-plane direction goes to +X, else yaw is skipped (`yaw.clear`).
6. **Similarity** `X' = s R X + t`, `R = R_yaw R_up` (`R_up` = smallest rotation taking up to +Y), `t = -s R p0`, `p0` = camera centroid dropped onto the floor along up. Applied by patching the bytes of `images.bin`, `points3D.bin`, `frames.bin`, `rigs.bin` in place (`colmap_io.apply_similarity`); `cameras.bin` is copied. Cameras: `R_i' = R_i R^T`, `t_i' = s t_i - R_i' t` (test `test_similarity_formulas`).
7. `align.json` keys: `S` (4x4), `s`, `R`, `t`, `up_resid_deg`, `up{sfm_frame, cues, weights, cue_spread_deg}`, `floor{support, rms (m), method, ceiling_height_m, ...}`, `h_cap` and `h_cap_m` (both metres, aligned frame; `h_cap_sfm_units` too), `scale{value, sigma_log, pct_68, interval_68, method, cues[], user_height_m, checked_against}`, `yaw`, `walls`, `flags[]`, `gate{passed, reasons[]}`, `model{registered, points, mean_reproj_px, mean_track_len}` before/after, `after{...}` (re-measured on the aligned model), `moge`.
8. **Exit codes.** 0 whenever `40_align` is written. `gate.passed = false` (up cues disagree > 5 deg, no floor, H_cap x scale outside 1.0-1.8 m) shows in `align.json`, in the `NS_RESULT` line (`gate_failed`) and as a `FLAG:` line; `--strict` turns it into exit 30. Reason: a flagged run is still usable (honka is flagged), so the stage should not stop the chain; root maps `gate_failed` to the final exit 30. 20 if there is no model.

## MuSHRoom pose format (the "[U pose format]" unknown)

`iphone/long_capture/transformations.json` is nerfstudio style:
- `camera_model: OPENCV`, per frame `fl_x fl_y cx cy w h` (738 x 994 portrait), `file_path ./images/frame_00001.jpg`, `depth_file_path`, and `transform_matrix` (4x4).
- `transform_matrix` is camera-to-world in the **OpenGL** camera convention (x right, y up, z backward), translation in metres. 334 frames for 335 images. `transformations_colmap.json` has one intrinsics block and the same layout; its poses differ from the first file by 1 to 3 cm.
- The world frame is **Z-up** (Polycam/ARKit, gravity-aligned): the mean camera-up axis (column 1 of the rotation) is (0.062, 0.001, 0.998); `icp_iphone.json` (iPhone frame to Faro frame) is a rotation of about 47 deg about Z plus a shift of 1.34 m in Z, so Z stays Z. So **gravity = +Z** in this frame. The Faro cloud `gt_pd.ply` has its floor peak at z = 0.02 m, a table-top peak at 0.74 m and a ceiling peak at 2.36 m (z histogram, 2 cm bins, 1.5 M points).
- Camera height above the Faro floor: median camera z 0.07 in the iPhone frame becomes 1.41 m in the Faro frame, so about **1.39 m** (the 1.40 m generic prior is almost exact for this capture, by luck).

## Numbers (smoke model `/w/runs/smoke_honka/clean`, 283 images, 21,429 points, as `30_undist`)

Metric truth: robust Umeyama (70% trimmed, 8 iterations) of our camera centres onto the dataset's `transformations.json` centres, matched by file name: **s_true = 0.5110 m per SfM unit, median residual 1.5 cm, 198 of 283 cameras** (a plain Umeyama gives 0.4947 with a 0.38 m RMS, dominated by a bad trajectory segment, so it is not used).

| Item | Result | Gate |
|---|---|---|
| D2 up vs gravity | fused **1.63 deg**; cues: cameras 4.83, planes 2.14, walls 8.37 (weight 0, conditioning 0.13) | <= 3 deg: pass |
| D2 scale, camera-height cue only, generic 1.40 m | 0.5602 vs 0.5110: **+9.6%** (68% interval +-13%) | +-20%: pass (target, not a promise) |
| D2 scale fused with MoGe-2 | not run (see MoGe) | +-12% |
| D1 random rotation, honka | rotated run vs original: 0.16, 0.11, 0.12 deg | <= 1.0 deg: pass |
| D1 random rotation, synthetic room (known up) | 0.26, 0.26, 0.44, 0.43, 0.45 deg | <= 1.0 deg: pass |
| D4 registered count / reprojection | 283 -> 283 images, 21,429 -> 21,429 points; own re-projection 0.618170 -> 0.618170 px over 118,536 observations; stored mean error 0.5926 -> 0.5926 | same: pass. COLMAP `model_analyzer` before/after: see below |
| D5 H_cap after scaling | 1.40 m | 1.0-1.8 m: pass, **but trivial**: with the camera-height cue alone H_cap x scale is 1.40 m by construction (0.8 x height with a height). It only constrains the fusion once MoGe or another cue moves the scale. The informative numbers: with the metric scale, the camera height above our floor is 1.28 m (truth 1.39 m, floor 0.3 SfM units too high); our ceiling-floor distance is 2.42 m (Faro: 2.34 m). |

The scale error is the floor error: with H_cap measured from the right floor, `1.40 / (1.39 / 0.511)` is within 1% of s_true.

## Floor: what went wrong first (kept because W6 mines failures)

1. First version: floor = lowest horizontal RANSAC plane with the cameras above it and at least 25% of the largest plane's support (the plan's rule). On honka the largest horizontal plane is the table top (0.74 m; 6,410 points). The plain grey carpet has few SfM points, the lowest plane passing the 25% rule sat about 0.35 m above the real floor, and H_cap came out 1.895 units, so scale 0.7386 against 0.511: **+44% error** (+49% against the plain Umeyama).
2. Fix: where the large walls end. For each wall plane take the 2nd and 98th percentile of its inliers' height along up; the support-weighted median over the walls that span at least 60% of the tallest wall gives floor and ceiling heights (honka: -2.52 and +1.89 SfM units; truth -2.72 and +1.86). A horizontal plane is accepted as the floor only if it lies between 25% of the wall height below and 12% above the walls' floor; the walls continue below a table or bench, which rejects those. If no plane qualifies the walls' floor is used (`floor.method = wall_extent`, flagged). Result: the plane at -2.44 (875 points) qualifies; error +9.6%.
3. The same rule on the Mip-NeRF 360 `room`: 43,809 floor points, agrees with the walls.

## Other things measured or decided

- **Up cue errors on honka** (vs gravity): cameras 4.8 deg (median pitch -12.8 deg and a path that favours some headings, so the "average over headings" does not fully cancel the pitch), planes 2.1 deg, walls 8.4 deg. Seven of eight wall planes have normals along one axis, so the wall cue is nearly unconstrained in the other direction: that is why it needs the conditioning test. Individual horizontal planes are 0.5 deg (ceiling, rms 0.045) to 12 deg off, so the plane cue weights by `support / rms^2`.
- Mip-NeRF 360 `room` (311 images, 112,627 points): median camera pitch -24.7 deg, so only 116 frames are "level"; cameras cue 4.9 deg off the fused up, planes 2.5, walls rejected (15.8 deg). Flagged (cue spread 7.4 deg). Floor 43,809 points. With the generic 1.40 m the ceiling lands at 1.58 m, which the new plausibility check flags ("ceiling outside 2.0-4.5 m: the camera-height scale is suspect"): the phone was probably held higher than 1.40 m. No metric truth for this scene. Figure: `pipeline/align/figures/d3_mip_room_level_before_after.png`.
- Honka flag: "up cues disagree by 6.1 deg" (cameras vs planes), so honka reports `gate.passed = false` even though the fused up is 1.6 deg from gravity. The flag is the plan's rule working on a biased camera cue; a user would see "review" on a good result. Open question for root: keep the 5 deg rule on the camera cue, or compare only the plane and wall cues.
- Reprojection check: `colmap_io.reproj_error` recomputes the mean reprojection error from the keypoints, poses and PINHOLE intrinsics (does not trust COLMAP's stored error). Before and after the similarity it agrees to 1e-14 px.
- `colmap_io` handles models with and without `rigs.bin` / `frames.bin` (COLMAP 4.2.1 writes both, and `images.bin` still carries the pose). Layouts are in its docstring; frames and rigs are patched too (rig sensor translations scale by s).
- Text models are not read; convert with `colmap model_converter --output_type BIN`.
- Far outliers (a bad segment 1000 units away) are dropped before RANSAC (points beyond 4 x the 90th percentile of the distance to the median).
- RANSAC threshold = 1.2% of the 2nd-98th percentile bounding-box diagonal of the points (about 10 cm in a 6 m room). Seeded (`--seed`, default 0).
- The MoGe cue needs `namma-space-gpu:0.2` with `moge` importable. See the status line at the bottom.

## rescale (`ns rescale`)

`rescale.py` writes `40_align/rescale.json`: `{factor_total, T_user (4x4 uniform scale about the world origin), applied[], auto{scale, pct_68, h_cap_m}, scale_final, h_cap_m_final, validation, checked_against}`.
- `--scale 1.08` multiplies the current user scale; `--set-total F`; `--use-measures` adopts the scale the tape measures in `input/meta.json` imply (geometric mean of tape / model over length, width, ceiling; idempotent); `--check` only compares (the automatic scale's error against the tape comes first, as the plan asks); `--reset`.
- Room size for the comparison: `70_grid/grid.json room_extent_m` if present (it may carry `user_scale`, the factor already applied, default 1), else the 2nd-98th percentile X and Z extent of the aligned sparse points (rough, flagged).
- It never touches the model, the PLY or any training output. Whoever builds the scene applies it: `T_final_total = T_user @ T_final(export)`; positions, grid, bounds, spawn and bookmarks scale by f; `splats.transform.scale = f`; eye height = H_cap x f. `rescale.compose_T_final(run, T_prev)` and `rescale.factor(run)` are the helpers. **Needs from others:** W3 export to read `rescale.json` into `export.json T_final`, W4 grid to re-run after a rescale and to write `user_scale` into `grid.json`, W5 bundle to apply the factor. Not wired by me.
- Refuses factors outside 0.2-5.

## Commands run

```
# Mac, venv in /tmp/namma-w2/venv (numpy 2.5, OpenCV 5, matplotlib) since the Mac has no numpy
python -m unittest discover -s test/unit/align -p 'test_*.py'
# tyrone, one script: align on honka and Mip, the unit tests, then COLMAP model_analyzer before/after
bash test/unit/align/run_tyrone.sh        # runs through bin/nsdocker (namma-space-gpu:0.1, namma-colmap:0.1)
```
