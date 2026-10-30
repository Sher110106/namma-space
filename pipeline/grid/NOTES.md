# WP4 grid: notes (W4, 30 Oct 2026)

Stage: `python -m pipeline.grid --run <run_dir> [--out DIR] [--ply P] [--tfinal auto|skip] [--force-mode grid|corridor|bounds]
[--h-cap M] [--floor-y 0] [--cell 0.05] [--no-tour]`
Image `namma-space-gpu:0.1`, no GPU (numpy 1.26, scipy 1.17 in the image; the Mac venv has numpy 2.5, scipy 1.18).
Reads `40_align/sparse/0` (binary or text COLMAP), `40_align/align.json` (`h_cap`), `60_export/clean.ply` and
`60_export/export.json` (`T_final`), optional `10_frames/frames.json` (`kept[].sharp`). Writes `70_grid/` through
`70_grid.tmp` (same naming as `pipeline.common.stage.atomic_stage`): `grid.png`, `minimap.png`, `grid.json`.
Exit codes: 0 good; 20 input missing or too small (under 3 cameras or 1000 splats) or an unexpected exception; 30 the stage
used a fallback (corridor or bounds) or found no spawn, so `grid.json` exists but needs review. The last stdout line is
`NS_RESULT {exit, gate_failed, mode, spawn_ok, walkable_m2, path_coverage, room_extent_m, warnings}`.

## Real W2/W3 output: one look, and it was thin
G1 and G2 ran on a **stand-in** for honka (below), not on `align` and `export` output, because those were still being
written when I tested. Root ran this stage on `rh-honka30` at 04:18 (30 frames, 20 registered cameras, root's thin align
adapter with an 11 deg up residual and camera heights 0.6-2.1 m, 17,885 splats of which 488 have opacity > 0.5). The
formats all read (W3's `clean.ply` and `export.json`, W2's binary model with `rigs.bin`/`frames.bin`, `frames.json`), and
the ladder did its job: grid and corridor reached 55% path coverage, so the stage fell to `bounds` and exited 30. That
is the right answer for that input, not a grid bug. It did expose one bug: the `bounds` box (SfM points 1-99%, shrunk
0.3 m) held only 30% of the cameras; it is now grown to hold the whole camera path (coverage 1.0 on a local rerun). The
first run on a full-size honka model from `align` and `export` is still to do.

## Contract points the integrator should know
- **PLY.** `60_export/clean.ply` (W3's name; fallbacks: `clean/cleaned/splat_clean.ply`, any `*.ply` in 60_export, then
  the trained 29999 PLY). Standard 3DGS layout. Opacity is a logit and scale a log when any opacity is outside [0, 1] or
  any scale is negative (auto-detected, `--activations` overrides). Colour is SH0 (`f_dc_*`).
- **T_final** is applied here to the PLY, the SfM points and the camera centres/forward vectors, as W3's `export.json`
  and root's `parse_t_final` define it (`X_final = scale * R * X + translation`; every form parses, checked against
  `test/data/conventions.json` poses and `t_final_cases`). `--tfinal skip` if a PLY ever arrives already in the final
  frame. H_cap becomes `align.json.h_cap * scale(T_final)`.
- **grid.json** keys (contract ones first): `origin_xz, cell, w, h, spawn{position,yaw_deg,pitch_deg}, bookmarks[{id,name,
  position,yaw_deg,pitch_deg}], tour[{position,yaw_deg,pitch_deg,t}], room_extent_m[2] (length >= width),
  walkable_m2, path_coverage`; extras: `row0, values, minimap{url,origin_xz,px_per_m,w,h}, spawn_ok, tour_stops[],
  room_extent_axes_m, room_axes_yaw_deg, mode, eye_height_m, h_cap_m, h_cap_source, h_cap_cameras_m, floor_y,
  floor_offset_measured_m, bounds{min,max}, counts, diagnostics, inputs, params, warnings[], timings_s`.
  `bounds` is the 0.5-99.5% box of splat centres (y from the floor - 0.3 m); W5's bundle.py already reads it, `floor_y`
  and `eye_height_m`. `walkable_m2`, `path_coverage`, `room_extent_m`, `spawn_ok` are the `report.json.grid` keys.
- Pixel (i, j) of `grid.png` is x = ox + (j + 0.5) * cell, z = oz + (i + 0.5) * cell, row 0 = min z. `minimap.png` has the
  same origin and 40 px/m (2 px per cell; 20 px/m when the grid is over 2048 cells on a side).
  Values in `grid.png`: 255 walkable, 128 floor evidence but not walkable (inflated zone, beyond the leash, other
  components), 0 obstacle cells or no floor evidence.
- Yaw: 0 looks along -Z, positive turns left, forward = (-sin yaw, 0, -cos yaw). Same formula as `conventions.yaw_pitch_deg`.
  Eye y = floor_y + H_cap (W5 overrides it with its clamped `eye_height`).

## What it does, and where it departs from FINAL_PLAN 2.7
Everything scales with H_cap. Cells are 5 cm.
1. **Floor evidence.** Opaque (>= 0.2) splats with |y| <= 0.07 H_cap, each stamped as a disc of radius clip(2*sqrt(s1*s2),
   5 cm, 30 cm) from its two largest scales; plus cells within 0.35 H_cap of the capture path; closing with a 15 cm disc
   (holes up to 30 cm); then dropped beyond 2.2 H_cap from the path (cosmetic: far splats made grey blobs in `grid.png`).
2. **Capture path.** Camera centres in name order plus points interpolated along segments shorter than 1 m.
3. **Obstacles.** Cells with >= 3 splat centres (opacity > 0.5) between 0.10 and 1.25 H_cap. 8-connected blobs under 10
   cells with a cell within 15 cm of the path are floaters and are dropped.
4. **Inflation** is measured from the obstacle splat centres (KD-tree), not from 5 cm cells, so cell quantisation adds no
   error. **Radius = min(0.18 H_cap, 0.30 m) minus half a cell** (0.227 m at H_cap 1.4 m). The half cell is a choice:
   with exactly 0.252 m a 0.9 m door leaves 0.396 m of centres, which a 5 cm grid shows as 7 or 8 cells depending on
   alignment, so "the door keeps 0.4 m" would fail by luck. With 0.227 m a door is 8 or 9 cells for any placement
   (tested at six sub-cell offsets). The price: "table + 0.25 m is blocked" holds to 0.225 m measured (the nearest
   walkable cell to the table is 0.225 m away); the test accepts one cell of tolerance there.
5. **Walkable** = floor evidence, within 1.1 H_cap of the path, outside the inflated obstacles, **plus every floor cell
   within 10 cm of the capture path** (added after honka: 19 of 283 cameras, 6.7%, sat inside the inflation zone, giving
   93.3% coverage; the phone and its holder were there, so the path stays walkable). Then the eye test (below), then
   the 4-connected component holding the most camera cells; the spawn is chosen inside it.
6. **Eye test (G2).** Walkable cells whose eye point (x, floor + H_cap, z) has more than 2 opaque centres within 0.15 m
   are removed, then the component is taken. It counts every opaque splat, floaters included, so a floater at exactly
   eye height blocks the few cells around it (G3's floaters sit at 0.6, 1.0 and 1.65 m; the eye test has its own test).
7. **Fallback ladder** (6.5): `grid`; if the spawn is missing or path coverage < 0.90, `corridor` (0.4 m discs along
   the path, eye test still applies); then `bounds` (SfM box shrunk 0.3 m, or the path box +-1 m without points; always grown to hold the camera path).
   Every failed step is in `warnings`. Exit code 30 for the last two.
8. **Spawn.** Frames sharper than the median (frames.json) and pitched within +-15 deg whose cell is walkable; among
   them the camera seeing the most SfM points, restricted to median depth >= 2 m. Relaxations (no sharpness, 25 deg,
   any walkable camera, depth rule dropped) are listed in `warnings`.
9. **Bookmarks.** k-means (k = 5, k-means++, 8 starts, seed 0) on camera xz; each centre snapped to the nearest walkable
   camera within +-20 deg pitch, no camera twice; ordered by mean capture index; named View 1..5, ids v1..v5.
10. **Tour (P1, done).** Six stops: the spawn plus real walkable level cameras spaced evenly along the capture path
    (capture order, wrapping), at least 1 m apart. Legs follow the shortest path over walkable cells (8-neighbour
    Dijkstra, no corner cutting), simplified by line of sight, joined, smoothed with a centripetal Catmull-Rom spline
    (kept only if every sample stays on a walkable cell, else the polyline), resampled to 61 keyframes, 1 s apart,
    uniform speed over 60 s. Heading: path tangent over +-1.5 m, pulled toward the stop camera's heading within 1 m of a
    stop, Gaussian-smoothed (2 s) and limited to 60 deg/s. The first earlier version used k-means stops ordered by mean
    index and doubled back on a loop path; the capture-order stops fixed it.
11. **Room extent.** Main wall angle from the obstacle splats (the angle that piles them into the fewest 3 cm bins, modulo
    90 deg). Along each axis a wall is a band of obstacle cells (+-1 cell across) whose distinct cells along the wall add
    up to at least 60% (then 40%, then 25%) of the camera path's span along the wall, and at least 1.5 m. The room runs
    between the outermost such walls within 3 m of the path's range; the position is the mean of the obstacle splat
    centres within 8 cm of the wall column. Length counts, not splat density: the first version used splat counts and on
    honka dropped the right wall (a third of the left wall's density), giving 2.78 m instead of 4.03 m. A 1.4 m closet
    behind a door is shorter than 60% of the path span and does not count. A table inside the path loop does not count
    (the walls are outermost). Limit: furniture that is also a long, tall, outermost line (a bookcase along the far
    side of the walls) is a wall here; a very sparse wall is dropped only after the 25% step fails.
12. **Minimap.** Splats below 2.0 m (opacity > 0.3), per 2.5 cm pixel the second-highest splat sets the surface (one stray
    splat cannot), colours of splats within 8 cm below it averaged; holes filled from the nearest painted pixel within
    6 px, background grey; the walkable outline drawn 2 px wide in green on top. "Max projection" read as highest-surface.
    P1 (gsplat orthographic render) not done.
13. **Floor offset.** The median height of opaque splats near the floor band is reported as `floor_offset_measured_m`
    (honka stand-in: +2.7 cm); the stage never moves the floor (W3 owns T_final). Warns above 5% of H_cap.

## Acceptance tests (one command: `test/unit/grid/run.sh`; 49 tests)
Mac: `test/unit/grid/run.sh` (uses `pipeline/grid/.venv`; do not commit it: `pipeline/grid/.gitignore` lists it).
Tyrone: `ssh sher@10.1.18.190 /home/sher/namma-space/bin/nsdocker ns-w4-test namma-space-gpu:0.1 test/unit/grid/run.sh`
(after `rsync`; stdlib `unittest`, no pytest needed). `W4_SKIP_PERF=1` skips the 350 MB G4 file.

| Test | Result (synthetic room unless noted) |
|---|---|
| G3 walkable only inside the room or the door closet; the 0.3 m hole at 1 m height leaks nothing | 0 cells outside |
| G3 table + 0.25 m blocked | nearest walkable cell 0.225 m from the table; 100% blocked within 0.20 m, over 90% in the 0.20-0.25 m ring |
| G3 3 floaters on the path ignored | 3 blobs dropped (8 cells); cells at 0.20-0.26 m from each floater walkable (>95%) |
| G3 door keeps >= 0.4 m | 0.40 m (8 cells) at the base placement, >= 0.40 m at six sub-cell offsets |
| G5 room extent within 5 cm | 4.996 x 3.995 m for 5.0 x 4.0 (also at the six offsets and after a 90 deg rotation in T_final) |
| G6 pillar within 1 cell on both images | centroid error 0.3 / 0.2 cm in `grid.png`, 0.8 / 0.04 cm in `minimap.png`; bounding boxes within 5 cm; 0.6 x 0.3 m orientation correct |
| G4 1.5 M splats in <= 60 s | 1,518,722 splats, 358 MB PLY (59 properties): stage 1.5-1.9 s on tyrone (ns-w4-test), PLY read 0.28-0.40 s; 1.0 s on the Mac with a PLY without SH rest (84 MB, disk was short) |
| G1 honka stand-in | spawn walkable; 1 component; path coverage 1.000 (283 cameras); 5 bookmarks, all in the component; 11.1 m2 walkable |
| G2 honka stand-in (independent KD-tree over the PLY) | max opaque centres within 0.15 m of an eye point = 1 over 4,444 walkable cells; 1 cell removed by the stage |
| Also | PNG/PLY/COLMAP round trips; text COLMAP model through the stage; no `frames.json` and no `h_cap`; T_final scale 1.5 (extent 7.5 x 6.0 m, spawn at 2.1 m); `--tfinal skip`; Pillow reads `grid.png` as mode L; yaw convention; `conventions.json` poses and T_final cases; k-means; extent; T_final with rotation and shift; eye-height clump; fallback ladder (forced corridor, forced bounds, bounds without SfM points, fog at eye height); CLI exit codes 0 and 20; spawn rule; bookmarks are real cameras; tour on walkable cells, 61 keyframes, under 90 deg/s |

Stress (not a test): 1.8 M random splats over 60 x 60 m, 3000 cameras, 1190 x 1189 cells: 4.2 s (minimap 1.6 s, PLY 1.3 s).

## The honka stand-in (NOT align/export output)
`pipeline/grid/standin_honka.py`, run once in `namma-space-gpu:0.1` into `/w/runs/w4-honka_standin` (34 MB; a copy is
under `/tmp/w4/honka_run` on the Mac and `test_honka.py` finds either). It:
1. runs gsplat's own `Parser(normalize=True)` on `/w/runs/smoke_honka/clean` to get the 4x4 normalisation transform and
   undoes it on the PLY (so the gsplat normalisation is removed exactly, not estimated);
2. fits a similarity from the COLMAP camera centres to MuSHRoom's iPhone metric poses (`transformations.json`, 283
   shared frames): scale 0.4947, RMS residual 0.38 m, median 0.17 m (the two trajectories disagree, so "rough metric");
   path length ratio 1.07; up from the dataset cameras' up axes (+z of that frame);
3. levels the floor with a RANSAC plane through splats at the lowest strong height peak: first round found a 3.8 deg tilt,
   second 0.6 deg (a first version without it put the floor 0.4 m too high); floor residual std 2.4 cm;
4. floor to y = 0, camera-path centroid to x = z = 0; H_cap = 1.383 m (camera heights p10-p90 1.15-1.49 m, splat tops
   near 2.3 m: plausible metric);
5. applies P0 clean steps 2, 4 (grow 0.5 m) and 5 of 2.6: 1,000,000 -> 480,617 splats (512,508 under opacity 0.05).
The room is a 4 x 6 m rectangle turned 13.6 deg from the axes with a long object (about 1.3 x 3.5 m) in the middle; the
capture walked a U around it (left, bottom, right), never across the top, so the walkable area is that U. The stage
gives a room extent of 5.94 x 4.03 m for it (MuSHRoom's Faro mesh was not measured; compare when real output exists).

## Failures and fixes during development (for the failure table)
- Synthetic walls on the z axis had their two free axes swapped in my generator; the room looked right except for the
  wall spans (found by looking at `grid.png`/`minimap.png` before trusting the numbers).
- COLMAP points3D.bin record is 51 bytes (I first used 43): reader and writer disagreed. Round-trip test added.
- text images.txt: images with no 2D points have an empty line; stripping empty lines broke the pairing.
- Room extent from splat counts: the closet back wall (1.4 m) counted at a 20% threshold and added 1.2 m; a wall with a
  third of the density vanished on honka. Replaced by wall length (item 11).
- Path coverage on honka 93.3% (below the 95% of G1) because 19 cameras lay inside the inflation zone: fixed by item 5.
- Tour with k-means stops doubled back and turned 142 deg per second; capture-order stops and yaw smoothing fixed both.
- `bounds` fallback crashed without SfM points (stress run): fixed, tested.
- Mac disk was at 100% (about 400 MB free); the G4 test shrinks the PLY (no SH rest) when under 1.5 GB is free and says so.
- nsdocker queues (max 3 containers): a stuck `ns-w4-dbg` of mine waited 10 min; killed by exact PID.

## Needs from others / open
- W3: `clean.ply` stays in the aligned frame with `T_final` in export.json (as written now). If that changes, use `--tfinal skip`.
- W2: `align.json.h_cap` in metres in the aligned frame (as written now). The stage warns when it differs from the median
  camera height above y = 0 by more than 10%, and when H_cap is outside 1.0-1.8 m.
- W5: `grid.json` needs no translation; `bundle.py` already reads `eye_height_m`, `bounds`, `floor_y`, `spawn`, `bookmarks`,
  `tour`, `minimap.px_per_m`.
- Not done: P1 orthographic gsplat minimap; the `places` / POI overlay; any view of the plan with furniture labels.
- Honest limits: thin or glassy obstacles with fewer than 3 centres per cell are missed (inflation closes gaps up to
  about 0.45 m between dots); the capture path is always walkable even through a low obstacle the phone was held over;
  room extent follows the outermost long wall within 3 m of the path.
