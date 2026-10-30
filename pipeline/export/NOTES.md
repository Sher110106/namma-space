# WP3 export: notes (W3, 30 Oct 2026)

Stage: `python -m pipeline.export --run /w/runs/<name> [--frame aligned|raw|gsplat-normalised] [--no-hd] [--no-preview]`
Image `namma-space-gpu:0.1` or `:0.2` through `nsdocker --gpu` (gsplat parity renders need the GPU).
Writes `60_export/` via `60_export.tmp/`: `clean.ply` (full SH, source of every web file and what W4 reads),
`splat_sd.spz`, `splat_hd.spz`, `splat_preview.spz`, `heldout/{<id>.jpg, heldout.json}`, `export.json`.

## Frame and T_final
All splat files stay in the aligned frame. `export.json.T_final` = `{scale, quaternion [x,y,z,w], translation, matrix(16, row-major)}`
with `X_final = scale * R * X + translation`. Today it is a translation only: the floor height measured from the cleaned
splats (lowest footprint-weighted height peak with >= 25% of the tallest, below the cameras), applied only if it is
within 0.30 m of y = 0, else identity and a warning. Consumers (W4 grid, W5 bundle) apply it. Held-out poses in
`heldout.json` already include it (three.js position and quaternion, CONTRACTS 2.0 formula).

## P0 clean (counts in `export.json.removed`; per-step PSNR in `clean.psnr_after_step`)
1 NaN/Inf; 2 opacity < 0.05; 3 largest axis (exp of max log-scale, one-sigma radius) above both 1.0 m and the 99.9th
percentile; 4 centre outside the SfM points' 0.5-99.5% box grown by G; 5 camera bubble 0.15 m. `freespace` is 0 (P1).
**Step 4 deviates from the plan.** With G = 0.5 m it cost 1.03 dB of held-out PSNR on the honka smoke model
(4,650 splats): it deletes background seen through windows and doors, where the sparse SfM has no points
(untextured walls, outdoors). `--outside-grow auto` (default) renders the held-out views for G in 0.5, 1, 2, 4, 8, 16 m
and takes the first G whose step costs <= 0.10 dB, else skips the step. Honka smoke: G = 2 m (0.017 dB, 443 splats).
`--outside-grow 0.5` gives the plan's literal rule. The search table is in `clean.details.outside_search`.

## Web files
- `splat-transform` 3.10.3 (MIT). Flags used: `-H <0-3>` (drop SH bands above n), `--spz-version <3|4>` (default 4),
  `-w`, `--no-tty`. Decode = `splat-transform in.spz out.ply`. Found through `$SPLAT_TRANSFORM`, `PATH`
  (image 0.2), or the wrapper `/w/runs/w3-tools/splat-transform` (my private Node 22 + the npm package, 471 MB under
  `/home/sher/namma-space/runs/w3-tools/`, used because image 0.2 did not exist yet). Exact commands are in `export.json.cmd`.
- Without splat-transform the stage ships the `.splat` fallback (own numpy writer, SH0, 32 B/splat; `ply_io.py`,
  `splat_writer.py`), pruned to fit the budget, HD skipped.
- SD <= 1.0 M, preview 200 k (SH0), HD <= 1.5 M (SH3, shrunk by 15% steps until <= 90 MB). Pruning keeps the
  highest-opacity splats (`--prune-mode opacity`). Measured on the honka smoke model, 483 k -> 250 k splats, SH1, held-out
  PSNR of the decoded file: opacity 19.38 dB, opacity x footprint 19.15, opacity x geometric-mean size 18.92,
  opacity x volume 16.81 (no pruning: 19.71).
- splat-transform stores the PLY's numbers unchanged: for the smoke PLY the first three splat positions are
  (1.4859, -0.8020, -0.2289), (-1.5341, -0.0720, -0.4440), (-1.2627, 0.4311, -0.2150) in the PLY and
  (1.4858, -0.802, -0.229), (-1.5342, -0.072, -0.4441), (-1.2627, 0.4312, -0.2151) in the SPZ v3 (24-bit fixed point, 12 fractional
  bits, read back with a stdlib gzip parser). No axis flip in the file. An aligned-frame (+Y up) PLY therefore gives a +Y-up SPZ.
- SPZ -> PLY keeps the count, position error 0.21 mm max (limit 2 mm), splat order preserved. splat-transform's
  writer and reader agree on coordinates (PLY -> SPZ -> PLY is the identity), so any axis flip is Spark's, to be
  found by H4. Scales are clamped to log -10 by SPZ (invisible splats).
- **SH level of SD.** The contract says SH1. On honka, SH1 costs 0.79 dB against SH3 before any quantisation
  (SH0 1.14 dB, SH2 0.33 dB). So `--sd-sh auto` (default) starts at SH1 and raises it to 2, 3 while the decoded file
  loses more than 0.5 dB against the trained model and stays <= 40 MB; `sd.sh` and `sd.sh_search` record the result.
  `--sd-sh 1` gives the contract file. `scene.json.splats.sd.sh_degree` must be read from `export.json`.

## Held-out pairs, parity renders and W5's tests
`60_export/heldout/heldout.json` = the `scene.json.heldout` entries (id, photo, position, quaternion, fov_y_deg,
principal_point, aspect, psnr = decoded SD file vs the photo) plus `psnr_trained`; photos are 960 px on the long side.
`60_export/parity/{<id>.png, parity.json}` are gsplat renders of the decoded SD file (what the browser loads) at the same
8 poses, for W5's H4b: `node viewer/test/run.mjs --only H4b --parity <run>/60_export/parity`.
The smoke copy is `samples/honka_smoke_parity/` (poses in the smoke frame, so the viewer must load the smoke SPZ untransformed).
`export.json.render` carries the CONTRACTS render block that `bundle.py` looks for.

## Parity (F2) and renderer
`render_ply.py` renders any PLY/.splat with gsplat at COLMAP poses and gives PSNR against photos. It reproduces the
trainer's own held-out PSNR (smoke PLY: 20.625 vs 20.62 reported with `--classic`; the honka 7k PLY: 16.233 vs 16.210).
Held-out set = every 8th image by name (gsplat's `test_every 8`). The smoke PLY was trained without `--antialiased`;
`--classic-render` matches it.

Library use (W7):
```python
from pipeline.export.render_ply import load_gaussians, render_views, cameras_from_colmap, evaluate
g = load_gaussians("/w/runs/x/50_train/r0/ply/point_cloud_29999.ply")
cams = cameras_from_colmap("/w/runs/x/40_align", held_out=True)       # or held_out=False for all, names=[...]
imgs = render_views(g, cams)                                          # list of HxWx3 float32, 0..1
print(evaluate(g, cams)["psnr"])
```
CLI: `python -m pipeline.export.render_ply --ply P --model <data_dir> --held-out --out DIR --json F`
or `--poses poses.json` (rows `{name, c2w, fx, fy, cx, cy, width, height}` or viewer poses
`{name, position, quaternion, fov_y_deg, width, height, principal_point?, t_final?}`).
For viewer poses (off-path screenshots) use `cam_from_threejs(name, position, quaternion, fov_y_deg, w, h, pp, t_final)`;
it is the exact inverse of `heldout.threejs_pose` (unit-tested).
For a PLY trained with world normalisation ON, `cameras_from_parser(data_dir, normalize=True)` gives the matching cameras.

## Sample for W5 (smoke model, NOT metric, NOT y-up)
`pipeline/export/samples/` (git-ignored, over 40 MB in total; on tyrone `runs/w3-smoke/60_export*/`):
`honka_smoke_sd.spz` (SH1, 483,100 splats, 9.1 MB), `honka_smoke_sd.splat` (SH0 fallback, 15.5 MB),
`honka_smoke_hd.spz` (SH3, 12.7 MB), `honka_smoke_preview.spz` (200 k, SH0, 3.3 MB), `honka_smoke_export.json`,
`honka_smoke_heldout/` (8 photos, poses in the smoke frame).
Frame: gsplat world-normalised, so units are scaled (median camera distance 1, not metres) and **+Z is up**
(mean camera-down axis = (-0.06, -0.04, -1.00)), right-handed, camera forward mean (-0.15, 0.52, -0.84). To get y-up for
three.js rotate by -90 degrees about X. `T_final` is identity in the sample's export.json. The cleaned PLY is on tyrone
(`runs/w3-smoke/60_export_sh1/clean.ply`, 114 MB).

## Tests
`pipeline/export/.venv/bin/python -m unittest discover -s test/unit/export` (venv: numpy 2.5, scipy 1.18, Mac).
