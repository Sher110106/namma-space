# Contracts

Copied from the final plan (namma-space-work/design/FINAL_PLAN.md) at 03:35 IST, 30 Oct. Root owns this file; change it only through root.

### 2.0 One frame for everything

- **World frame:** metres, +Y up, floor plane at y = 0, right-handed (the three.js convention).
  - Origin: the camera-path centroid dropped onto the floor.
  - Yaw: the main wall direction lies on +X when it is clear; otherwise skip it.
- **Cameras** keep COLMAP/OpenCV axes (x right, y down, z forward), with world-to-camera R_i and t_i.
- **Align stage.** It computes one similarity X' = s·R·X + t and applies it to the undistorted COLMAP model before training.
  - Cameras: R_i' = R_i·Rᵀ and t_i' = s·t_i − R_i'·t.
  - `model_analyzer` must report the same registered count and reprojection error before and after.
- **Training** runs with gsplat's world normalisation off. The PLY, cameras, grid, plan and viewer then share this frame.
  - Normalisation's PCA rotation can tip a narrow room on its side.
  - The rival repo ships "flip" buttons for exactly this.
- **Later corrections** (floor offset measured from the splats, a user rescale) travel as `T_final` in `scene.json`.
  - The PLY is never edited, and a rescale never retrains (about 3-5 min [E]).
- **three.js camera for image i:**
  - centre C = T_final(−R_i'ᵀ·t_i');
  - rotation R_i'ᵀ·diag(1, −1, −1);
  - vertical FOV 2·atan(H / 2f_y);
  - principal point through a custom projection matrix.
- **`test/data/conventions.json`** holds 3 poses with their expected three.js position and quaternion.
  - Python and JS tests both check it.
  - Any convention change edits this file first, through root.
- **If SPZ or Spark flips an axis,** the export stage writes the correcting quaternion into `scene.json`, found by test H4. Nobody hand-edits it.


### 2.11 Run directory contract

```
runs/<name>/
  input/            videos or frames; meta.json {height_cm, measures[{what, metres, method}], notes, takes[]}
  00_preflight/     check.json  {verdict, warnings[], probe{...}}
  10_frames/        images/000001.jpg …; frames.json {n_cand, n_kept, dropped{blur, dup, thin}, hdr, rotation, src, kept[{name, t, sharp}]}
  20_sfm/<variant>/ db.db, sparse/<k>/, log.txt
  20_sfm/sfm.json   {variants[{name, reg_frac, n_models, reproj_px, n_pts, track_len, time_s, status}], chosen, why, missing_ranges[]}
  30_undist/        images/, sparse/   (PINHOLE, 1920 px max)
  40_align/         sparse/0/, images -> ../30_undist/images, align.json, figures/level_before_after.png
  50_train/<v>/     ply/point_cloud_6999.ply, ply/point_cloud_29999.ply, stats/*.json, renders/
  50_train/train.json  {variants[{name, psnr, ssim, lpips, cc_psnr, n_gs, time_s, peak_gpu_gb}], chosen, why}
  55_geometry/      mesh.ply, geometry.json   (P1)
  60_export/        splat_sd.spz, splat_hd.spz, splat_preview.spz, heldout/, export.json
                    {T_final, sd{count, mb, sh, psnr}, hd{…}, preview{…}, removed{nan, opacity, huge, outside, bubble, freespace}, cmd}
  70_grid/          grid.png, minimap.png, grid.json {origin_xz, cell, w, h, spawn, bookmarks[], tour[], room_extent_m[2], walkable_m2, path_coverage}
  80_web/           index.html, js/, vendor/, scenes/<name>/…   (written on the Mac)
  90_verify/        verify.json, shots/*.png   (written on the Mac)
  report.json, report.html, timings.json, run.json {git, images{digests}, flags, warnings[], manual_overrides[], exit}
```

Rules:
- A stage is done when its JSON exists; nothing else tracks state.
- Stages write to a temp directory and rename it when complete.
- `--from <stage>` re-runs from that stage onward.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Good |
| 10 | Capture problem; the time ranges to reshoot are printed |
| 11 | Partial reconstruction shipped |
| 20 | Infrastructure problem |
| 30 | A quality gate failed; a bundle exists but needs review |

### 2.12 Scene bundle: what the viewer loads

```
web/
  index.html  js/*.js  css/*.css  vendor/{spark.module.js, three.module.js, addons/…, LICENSES/}
  scenes/index.json                 {"default": "<name>", "scenes": [{"name", "title"}]}
  scenes/<name>/scene.json
  scenes/<name>/splat_sd.spz        required; ≤ 1.0 M splats, SH1, ≤ 40 MB
  scenes/<name>/splat_hd.spz        optional; ≤ 1.5 M, SH3, ≤ 90 MB
  scenes/<name>/splat_preview.spz   optional; ≤ 200 k, SH0, ≤ 4 MB
  scenes/<name>/grid.png            8-bit grey: 255 walkable, 128 floor not walkable, 0 blocked/unknown
  scenes/<name>/minimap.png         RGB, same origin as the grid, px_per_m given
  scenes/<name>/heldout/<id>.jpg    optional (reality check)
```

`scene.json` (schema `namma-space/scene@1`; the numbers are placeholders, not results):

```json
{
  "schema": "namma-space/scene@1",
  "name": "room",
  "title": "Our room",
  "provenance": {"kind": "captured", "source": "iPhone, 4K30, 3 min 40 s, take A", "license": "CC BY 4.0",
                 "attribution": "", "captured": "2026-10-30", "manual_steps": []},
  "frame": {"units": "m", "up": "+y", "floor_y": 0.0, "handedness": "right"},
  "splats": {
    "sd": {"url": "splat_sd.spz", "count": 1000000, "sh_degree": 1, "bytes": 0},
    "hd": {"url": "splat_hd.spz", "count": 1500000, "sh_degree": 3, "bytes": 0},
    "preview": null,
    "transform": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 1.0},
    "render": {"antialiased": true, "preBlurAmount": 0.0, "blurAmount": 0.3, "maxStdDev": 2.828}
  },
  "camera": {"eye_height": 1.42, "fov_y_deg": 70, "near": 0.05, "far": 50, "walk_mps": 1.4, "run_mult": 2.0},
  "spawn": {"position": [0.4, 1.42, -1.1], "yaw_deg": 120, "pitch_deg": 0},
  "bounds": {"min": [-3.2, -0.1, -2.8], "max": [3.1, 2.9, 2.9]},
  "grid": {"url": "grid.png", "origin_xz": [-3.2, -2.8], "cell": 0.05, "width": 128, "height": 112,
           "row0": "min_z", "values": {"walkable": 255, "floor": 128, "blocked": 0}},
  "minimap": {"url": "minimap.png", "origin_xz": [-3.2, -2.8], "px_per_m": 40},
  "bookmarks": [{"id": "v1", "name": "View 1", "position": [0, 1.42, 0], "yaw_deg": 0, "pitch_deg": 0}],
  "tour": [{"position": [0, 1.42, 0], "yaw_deg": 0, "pitch_deg": 0, "t": 0}],
  "heldout": [{"id": "000123", "photo": "heldout/000123.jpg", "position": [0, 0, 0], "quaternion": [0, 0, 0, 1],
               "fov_y_deg": 0, "principal_point": [0.5, 0.5], "aspect": 1.778, "psnr": 0}],
  "places": [],
  "scale": {"value": 1.0, "pct_68": 0, "method": "", "cues": [], "checked_against": null},
  "metrics": {"heldout_n": 0, "psnr": 0, "ssim": 0, "lpips": 0, "lpips_net": "alex", "cc_psnr": 0, "sd_psnr": 0},
  "build": {"git": "", "pipeline": "0.9.0", "date": "", "report": "report.json"}
}
```

Conventions:
- Positions are in metres in the world frame (2.0).
- `yaw_deg` turns about +Y, with 0 looking along −Z and positive turning left.
- Quaternions are `[x, y, z, w]` for three.js cameras, which look down −Z.
- A grid pixel (row i, column j) is the cell centred at x = origin_x + (j + 0.5)·cell and z = origin_z + (i + 0.5)·cell.
- W5 owns the writer (`viewer/tools/bundle.py`, standard library only) and the reader. Schema changes go through root.

### 2.13 `report.json` (documents bind only to these keys)

Top-level keys:
- `schema`, `name`, `git`, `images`;
- `input` (files[], check);
- `frames`;
- `sfm` (chosen, variants[]);
- `align` (up_resid_deg, h_cap_m, scale{value, pct_68, cues[], checked_against});
- `train` (chosen, variants[]);
- `export` (sd, hd, preview, removed);
- `grid` (walkable_m2, path_coverage, room_extent_m, spawn_ok);
- `viewer` (devices[{name, gpu, browser, first_frame_s, fps_mean, p95_ms, mb, splats}]);
- `walls` (walks, pushes, penetrations);
- `timings` (per stage in seconds, plus total);
- `manual_overrides`.

Evaluation results live in `eval/results/*.json` with their own keys (WP7).


### 2.15 Host rules for tyrone (ssh sher@10.1.18.190)

- **Layout:** `/home/sher/namma-space/{repo (rsync target, read-only by rule), data, runs, cache, locks, docker (existing recipes; copied into repo/docker by W1)}`.
- **Code lives only on the Mac.** `./ns sync [subdir]` rsyncs it to `repo/`. Nobody edits code on tyrone.
- **Containers:**
  - named `ns-<run>-<stage>`;
  - each started with `--memory 20g --memory-swap 20g --cpus 24 --shm-size 8g`;
  - at most 3 of ours at once (60 GB in total). The host ran out of memory on 28 Oct [V memory notes].
  - Never stop, exec into or remove any container we did not start (`asp-noetic` belongs to another experiment).
- **GPU:**
  - two slots, `flock` on `locks/gpu.0` and `locks/gpu.1`;
  - a GPU job starts only if ≥ 12 GB of GPU memory is free; otherwise it waits.
- **Disk:**
  - every stage refuses to start below 15 GB free (exit 20);
  - delete candidate frames after ingest and losing variants' checkpoints after training;
  - budget 6 GB per run.
- **Long jobs** run under `nohup` or tmux. The Mac runs `caffeinate -i` during `ns run`.
- **After each stage,** `ns` copies its JSON, logs and key files to the Mac. The Mac copy is the record.
- **Run names carry the owner's prefix** (`w1-`, `w3-`, `rh-` for rehearsals, `room-` for the user's room).

---

## 3. Work packages

### 3.0 Ownership

Repo: `/Users/sher/project/namma-space` (git, Apache-2.0). Workers are Sonnet (or Luna on Codex). The reviewer is Opus (or Astra) and gets the code and the brief, never our conclusions.

| Agent | Owns (nobody else edits) | Runs on |
|---|---|---|
| Root | `ns`, `CONTRACTS.md`, `LICENSE`, `pipeline/common/`, `pipeline/remote/`, `pipeline/report/`, `test/e2e/`, `test/data/conventions.json`, `test/README.md`, `test/data/<sample>/`, `test/expected/`; tyrone `runs/rh-*`, `runs/room-*`, `locks/` | Mac and tyrone |
| W1 infra + SfM | `docker/`, `pipeline/sfm/`, `test/unit/sfm/`; tyrone `cache/`, images `namma-*` | tyrone |
| W2 ingest + check + align | `pipeline/ingest/`, `pipeline/align/`, `test/unit/ingest/`, `test/unit/align/`; tyrone `data/fixtures/videos/` | tyrone CPU, Mac |
| W3 train + export | `pipeline/train/`, `pipeline/export/`, `test/unit/train/`, `test/unit/export/` | tyrone GPU |
| W4 grid + plan | `pipeline/grid/`, `test/unit/grid/` | tyrone CPU |
| W5 viewer + bundle | `viewer/` (src, vendor, tools/bundle.py, test, dist) | Mac |
| W6 docs + video | `docs/`, `video/`, `README.md`, `LICENSES.md` | Mac |
| W7 eval + geometry | `eval/` (scripts, results), `pipeline/geometry/`, `test/unit/geometry/` | tyrone |
| Reviewer | read-only | gates G2, G5 and the final claims check |

Rules:
- Each stage is a module run as `python -m pipeline.<stage> --run <run_dir> [opts]`. It reads earlier stages' JSON and writes only its own stage directory.
- Contracts (sections 2.0, 2.11-2.13) are copied into `CONTRACTS.md` by root at 04:00. Changes go through root.
- Every package keeps a `NOTES.md` inside its own directory, with commands run, failures and decisions. W6 mines these for the failure table.
- Every package's acceptance tests run with one command: `./ns test <pkg>`.


## Stage table (root, 03:45 IST)

Every stage is `python -m pipeline.<stage> --run <run_dir> [opts]` with a `pipeline/<stage>/__main__.py`. On tyrone the runner starts it through `/home/sher/namma-space/bin/nsdocker` (mounts `/home/sher/namma-space` at `/w`, code at `/w/repo`, cwd `/w/repo`), so run dirs are `/w/runs/<name>` inside containers. A stage reads earlier stages' JSON and writes only its own directory, through a temp dir renamed on success.

| Stage module | Writes | Image | GPU | Owner |
|---|---|---|---|---|
| `pipeline.ingest` | `10_frames/` | `namma-space-gpu:0.1` | no | W2 |
| `pipeline.sfm` | `20_sfm/`, `30_undist/` | `namma-colmap:0.2` (0.1 plus `python3`, needed for the stdlib driver; 0.1 has no Python); stdlib-only Python 3 driving the `colmap` CLI (no numpy in that image) | yes | W1 |
| `pipeline.align` | `40_align/` | `namma-space-gpu:0.1` (MoGe cue needs `0.2`; optional) | optional | W2 |
| `pipeline.train` | `50_train/` | `namma-space-gpu:0.1` | yes | W3 |
| `pipeline.geometry` (P1) | `55_geometry/` | `namma-space-gpu:0.1` | yes | W7 |
| `pipeline.export` | `60_export/` | `namma-space-gpu:0.2` (adds Node 22 + splat-transform); falls back to its own numpy `.splat` writer on `0.1` | yes (parity renders) | W3 |
| `pipeline.grid` | `70_grid/` | `namma-space-gpu:0.1` | no | W4 |
| bundle | `80_web/` | Mac, `viewer/tools/bundle.py` (stdlib only) | - | W5 |
| verify | `90_verify/` | Mac, `viewer/test/` (headless Chrome) | - | W5 |
| report | `report.json`, `report.html` | Mac, `pipeline/report/` (stdlib only) | - | Root |

Mac preflight: `pipeline/ingest/check_mac.py` (stdlib only, ffprobe/ffmpeg), called by `./ns check`.

Shared data on tyrone (read-only): `/w/data/mushroom/room_datasets/{honka,vr_room}/` (iPhone `long_capture/{images,depth,transformations.json,transformations_colmap.json}`, `gt_mesh.ply`, `gt_pd.ply`, `icp_iphone.json`), `/w/data/mipnerf360/room/` (`images`, `images_2/4/8`, `sparse/0`), `/w/data/zipnerf/berlin/`.

Smoke artefacts (read-only, root's): `/w/runs/smoke_honka/clean/` (undistorted PINHOLE model after the camera-outlier filter, 283 images) and `/w/runs/smoke_honka/gs_out5/ply/point_cloud_29999.ply` (30k MCMC + bilateral grid, gsplat world normalisation ON, so not in the metric frame; held-out PSNR 20.62, cc_psnr 23.96, SSIM 0.807, LPIPS 0.242).
