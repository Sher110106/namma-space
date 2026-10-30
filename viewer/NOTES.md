# Viewer and bundle tool: notes

Working notes from the build. AI coding agents wrote them during the build; "W1" to "W7" and "root" are the work packages and the coordinating session named in CONTRACTS.md (W5 is this package). Nothing here is a claim about a result unless a number and a test name sit beside it.

Everything here ran on the Mac (Apple M1, headless Chrome 156 with Metal ANGLE) unless a line says tyrone.
Numbers below are measured in this session on the test fixture; nothing here is a result on the user's own room.

## What is in the folder

```
viewer/
  src/            index.html, js/*.js, css/viewer.css   (the viewer; plain ES modules, no build step)
                  nav.js (grid A*, string pulling, follow steering), places.js (fuzzy search, place JSON), guide.js (search UI, pins, route ribbon, editor, measure)
  vendor/         spark.module.js, three.module.js, three.core.js, addons/postprocessing/Pass.js, LICENSES/
  tools/bundle.py        run dir -> 80_web/ and validate (stdlib only)
  tools/serve.py         local static server with Range (stdlib only), default http://127.0.0.1:5178/
  tools/fixture/         make_fixture.py (numpy, runs on tyrone), spz_transcode.mjs (PLY -> SPZ with Spark's writer)
  test/                  run.mjs (runner), tests.mjs (H1-H9 + feature checks), lib/ (CDP driver, static server), run_all.sh
  test/fixtures/honka_run/   fake run dir (CONTRACTS 2.11 layout) built from the honka smoke run: the TEST FIXTURE
  test/fixtures/honka_places.json   8 placeholder places for the fixture (passed to bundle.py with --places)
  test/last_run/         verify.json and shots from the last full run
  places/                rh-honka.json, rh-vrroom.json: hand-placed places for the two public demo rooms (see "Review fixes 30 Oct", finding 16/45); shots/ has one jump-view screenshot per place
  dist/                  bundle.py output for the fixture: a working site (26 MB; scene honka_demo)
```

Commands:

```
sh viewer/test/run_all.sh                       # rebuild dist from the fixture, run every test (about 3 min), write test/last_run/
sh viewer/test/run_all.sh runs/<name>           # bundle a real run into 80_web/ and test it; writes 90_verify/verify.json
python3 viewer/tools/bundle.py --run runs/room --title "Our room" --source "iPhone 4K30, 3 min 40 s" --license "CC BY 4.0" \
        --kind captured --video-url <url> --default           # writes runs/room/80_web/
        --places runs/room/places.json                        # merge named places (or leave the file at runs/room/places.json: picked up when --places is absent)
python3 viewer/tools/bundle.py --validate <web dir> [--strict]  # H8 validator alone
python3 viewer/tools/serve.py runs/room/80_web                  # http://127.0.0.1:5178/?scene=room
node viewer/test/run.mjs --web <dir> --out <dir> [--only H1,H2] [--pairs file.json] [--parity dir]
```

## The test fixture (read this before quoting any picture)

`scenes/honka_demo` is MuSHRoom honka (CC BY 4.0), not our room. It is built from root's smoke run:
`/w/runs/smoke_honka/gs_out5/ply/point_cloud_29999.ply` (1.0 M Gaussians, 30k MCMC steps, gsplat normalisation on, trained classic, not antialiased).

- Real: the splats, the 283 camera poses (taken from the same `Parser` the trainer used, so they sit in the PLY's frame), the photos.
- Assumed: metric scale (median camera height set to 1.42 m, so `scale.pct_68` = 15 is a placeholder, not a measurement) and the floor level (densest splat layer below the cameras).
- Placeholder: `grid.png`, `minimap.png`, bookmarks, tour. They are a simplified version of FINAL_PLAN 2.7, not W4's output.
- Frame: the PLY is untouched. `splats.transform` in scene.json carries the similarity (position, quaternion, scale 2.5058) that puts it in the metres / +Y up / floor y=0 frame. This exercises the same path W3's `T_final` will use.
- `scene.json` says `provenance.kind = "fixture"` and the start card prints "Test fixture."

Rebuild: `python viewer/tools/fixture/make_fixture.py --data /w/runs/smoke_honka/clean --ply <ply> --out <dir> [--render]` inside `namma-space-gpu:0.1` (via nsdocker; `--render` needs a GPU slot), then
`node viewer/tools/fixture/spz_transcode.mjs <ply> <out.spz> <maxSh> 3 12` for each splat file.
Files: SD 800k splats SH1 = 15.5 MB (`splat_sd.spz`, SPZ v3), preview 150k SH0 = 2.5 MB. Largest file is under 16 MB.

## Measured (fixture, M1, 1280x720, pixel ratio 1; full table in test/last_run/verify.json)

| Test | Result | Pass rule |
|---|---|---|
| H1 | first splat frame 0.5-0.9 s from localhost with cache off (preview file); 0 console errors; 0 requests to other origins | <= 5 s |
| H2 | 20 s scripted walk (real W key plus mouse-drag turns), SD 800k: 59-60 fps mean, p95 16.7-16.8 ms, p99 16.8 ms | >= 55 fps, p95 <= 20 ms |
| H2 with W3's `honka_smoke_sd.spz` (SPZ v4, 1.0 M splats, 18.6 MB) | 58.8 fps mean, p95 16.8 ms (first runs at sqrt(8) gave 49-54 fps, p95 33-50 ms; see "Quality" below) | same |
| H3 | 200 random 5 s walks + 16 headings x 6 s + 4 real-key walks: 28,245 samples, 0 outside walkable cells; wall stops 16/16; moves >= 0.5 m 16/16 | 0; >= 8; >= 12 |
| H4 | browser render vs undistorted photo at 3 training poses: 21.3 / 22.8 / 22.0 dB; the same poses turned 10 deg: 10.9 / 11.2 / 13.8 dB (gain 8.3-11.6 dB). Held-out poses 20.0-22.3 dB, matching gsplat's own 20.6 dB for this run | >= 18 dB and >= 3 dB gain |
| H4b | browser vs gsplat render of the same file: see the H4b line in "Open items" | >= 25 dB |
| H5 | touch emulation 844x390: left thumb moved the player 1.3-1.5 m, right thumb turned 22.9 deg, start button tapped by touch | moves |
| H6 | `?stress=1` (pixel ratio 2): governor stepped to 1.4 at 1.8-1.9 s after the first splat frame. Since 30 Oct a step down is a trial and goes back if the frame rate does not rise by 10% (see "Review fixes 30 Oct", finding 15); on a busy Mac it sometimes goes back to 2 | <= 3 s |
| H7 | `?asset=` from a second origin with CORS: loaded, 800k splats; origin without CORS: falls back to the scene's own file | loads |
| H8 | bundle.py output: 73 checks, 0 failed; 6 of 6 corrupted copies rejected (missing key, size mismatch, missing grid, wrong count, wrong grid dims, missing held-out photo) | valid |
| H9 | server throttled to 2.5 MB/s: first frame 2.4-2.5 s (preview), SD fully loaded 8.7-8.8 s (preview 2.5 MB then SD 15.5 MB, in that order) | <= 18 s, <= 4 s |
| X1-X12 | WebGL2-missing card; keys R/H/M/?/Esc; tour benchmark T (59.7 fps, p95 16.7 ms); reality check C; bookmarks 1-5; `?pose` `?keys` `?q=low`; minimap click; HD toggle (mechanism test with the SD file standing in); `?scene=` and the error card; fly mode F; weak-GPU fallback to the preview; spot link L | features |

| R0 | Node: A* equals an independent Dijkstra on 60 pairs; 400 routes on the honka grid and 600 on 5 random synthetic grids (220x160 cells, 38 obstacles): 0 of 101,704 samples at 5 cm in a blocked cell; worst length 1.023x (honka) and 1.049x (synthetic) the grid optimum, mean 0.96-0.97x (string pulling beats the 8-connected path); typo search 9/9; follow simulation | blocked 0; <= 1.10x |
| R1 | the viewer's own code, 8 places x 12 start cells = 96 routes on the fixture: 0 failed, 0 of 10,634 samples blocked, worst 1.013x, mean 0.972x | same |
| R2 | jump: 8 of 8 land on a walkable cell, the nearest one at the right stand-back distance, yaw error 0 deg | walkable; <= 2 deg |
| R3 | search box: 11 of 11 queries (5 of them with one typo, 5 by alias); "zzzqqq" shows "No match"; Esc closes | all found |
| R4 | follow: 10.3 m route walked in 7.6 s (1.35 m/s mean, start turn included), 0 of 73 samples off walkable cells, ended 0.20 m from the stand spot and 3 deg off facing the place; the W key stops it | arrives; walkable |
| R5 | route re-plans after a jump 0.71 m off the path, keeps its route after a 0.3 m step; wall off the floor and the card says "No walking route: Coffee counter is in an area you cannot reach on foot." | as stated |
| R6 | measure: floor pair expected 1.015 m, tool 1.015 m (error 0 cm), "+/- 0.15 m" and the 68% text shown; splat pair 4.274 m with both clicks on splat surfaces, reprojection error 0 px | <= 1 cm |
| R7 | place editor end to end: P, click (splat hit), type name and alias, Enter; found by alias; Copy places JSON (headless refuses the clipboard, so the text box opened); list survives a reload; `bundle.py --places` merged it and validated | all steps |
| R8 | minimap: 56 marker pixels; click on a marker selects the place and does not move the player; route drawn (route-colour pixels 0 -> 185) | as stated |
| R9 | start card and ? help list the new keys; `?place=door&route=1` opens with the route drawn; picking another place clears the old ribbon; 0 console errors | as stated |
| R10 | phone 390x844 touch: search button 40x36 px, result row 44 px, card 366 px wide, Route by touch works | fits; >= 40 px rows |

Caveats on those numbers:
- The M1 is faster than the Intel/AMD iGPUs judges may have. No iGPU was measured. Headless Chrome on the M1 is vsync-capped at 60 Hz, so "59.9 fps" means no dropped frames, not headroom (research/viewer.md, trap 1).
- The Mac was busy with other builders during the runs (1-minute load average 8-19). One H2 run with 1.0 M splats at sqrt(8) failed on that account plus fill rate; see Quality.
- H1 first-frame time includes the preview file only. The full SD file is active about 0.3-0.7 s later on localhost (`__ns.loadedMs`).

## Quality: why SD starts at max std dev sqrt(5)

Tuning matrix on W3's 1.0 M-splat file, tour benchmark, 15 s each (`?bench=1&benchsecs=15&std=...`):

| setting | avg fps | p95 ms |
|---|---|---|
| maxStdDev sqrt(8) (Spark default, scene.json placeholder) | 53.4 / 48.7 | 33-50 |
| maxStdDev sqrt(5) | 59.9 | 16.8 |
| sqrt(5) + `minSortIntervalMs` 40 | 58.1 | 16.8 |
| sqrt(8) + `sortRadial` false | 48.2 | 50 |
| sqrt(8) + `minSortIntervalMs` 40 | 50.1 | 50 |

So fill rate, not sorting, is what the M1 runs out of at 1.0 M splats. Decision: the SD preset ("standard") starts at sqrt(5) (Spark's own docs call it perceptually very similar); the HD preset ("full") uses `scene.splats.render.maxStdDev` (sqrt(8)). PSNR against photos did not change (H4 21.3/22.8/22.0 both ways). Governor ladder (`governor.js`): standard -> low (pixel ratio 0.85) -> min (0.7); stress adds 2.0 and 1.4 on top. Rule as in the plan: mean fps < 40 over 2 s steps down; median fps > 55 over 2 s steps up, with an 8 s hold-off after a step down. Since 30 Oct every step down is a trial: it stays only if the frame rate rises by at least 10%; a capped frame rate (battery saver, half-rate vsync, a background tab) does not rise, so the step goes back (finding 15). Pixel ratio starts at 1 (0.85 when the GPU string looks integrated or mobile, 0.7 for software rendering).

## Decisions and traps

- **Spark API.** `SplatMesh({url, fileType, onProgress})`, `await mesh.initialized`, transform set on the mesh. `scene.json` `render.preBlurAmount` and `blurAmount` go straight to `SparkRenderer`; `render.antialiased` is informational. Spark's defaults are preBlur 0 and blur 0.3, which is right for antialiased training. The fixture was trained classic, so its export.json carries `{"antialiased": false, "preBlurAmount": 0.3, "blurAmount": 0.0}` (gsplat classic adds eps2d 0.3 to the covariance without opacity compensation).
- **SPZ.** Spark 2.4.0 reads SPZ v3 (my fixture) and v4 (W3's `pipeline/export/samples/honka_smoke_sd.spz`, header outside the zstd stream, 1.0 M splats SH1). Both load in the same frame, so no axis flip is needed for the splat-transform path. Spark writes only v2/v3 (`transcodeSpz`), which is how the fixture SPZ was made without splat-transform.
- **Camera for photos** (reality check, H4): FOV from `fov_y_deg`, aspect from the photo, principal point written into `projectionMatrix.elements[8]` and `[9]` (`1 - 2 cx/W`, `2 cy/H - 1`). The `quaternion` in scene.json is used as given (it already includes `diag(1,-1,-1)`). Verified numerically by H4.
- **Pointer lock cannot be tested headless.** The viewer therefore also supports drag-to-look when the pointer is not locked (also what in-app browsers get); click without dragging requests the lock. Esc releases the lock, a pill says "Click to resume". Not verified in a real Chrome, Safari or Firefox window.
- **First-frame detection.** After each render the viewer reads 48 pixels and compares them with the clear colour, until it sees splats. `__ns.firstFrameMs` is then `performance.now()` (since navigation start). Cheap and only runs until the first hit.
- **`?pose=x,y,z,yaw,pitch`** sets the eye height from `y` until the next respawn (R); other flags: `?scene ?asset ?bench ?benchsecs ?stress ?keys ?q`, plus tuning flags `?std ?sortms ?radial`.
- **`?asset=`**: a file URL replaces the SD file; a URL ending in `/` replaces the folder of all three files. If the override fails (no CORS, dead host) the viewer loads the scene's own file once and shows a toast.
- **Phones.** Left half of the screen walks (virtual stick), right half looks. The minimap shrinks to 36% of the short side; tapping it enlarges it, a tap in the large map jumps. The start card keeps its button sticky at the bottom and drops the lede on short screens. Touch is tested by CDP emulation only, not on a phone or in WhatsApp's browser.
- **Reality check** resizes the canvas to the photo's aspect, so the photo and render share the same pixels, and shows the pose's PSNR from scene.json when it is > 0. It does not compute a live PSNR in the UI (the test hook `__ns.psnrAgainstPhoto` does).
- **Test hooks** live on `window.__ns` (pose, walker, grid, sim.*, governor, bench result, `psnrAgainstPhoto`). Harmless in production; leave them.
- **Screenshots disturb the governor** (a screenshot stalls frames for 100+ ms and the mean drops). Tests never take screenshots while measuring.
- **bundle.py** writes through a temp folder and renames it only if validation passes. It keeps other scenes already in `scenes/` (add a second scene with the same `--out` and a different `--name`; `scenes/index.json` is merged; the start card then links to the other scenes).

## What bundle.py reads from the other packages (check against your writers)

- `60_export/export.json`: `T_final` as `{scale, quaternion[x,y,z,w], translation}` (W3's shape; also accepts `position`, a 4x4 matrix, or the `matrix` key W3 writes, row-major 16 numbers), `sd/hd/preview.{count, mb, sh, psnr}` (counts are taken from the SPZ header when readable, v1-v4), and an optional `render` block `{antialiased, preBlurAmount, blurAmount, maxStdDev}`. **W3: please write `render`.** Without it bundle.py assumes antialiased training (preBlur 0, blur 0.3) and warns.
- `60_export/heldout/heldout.json`: list of entries exactly as in scene.json (`id, photo, position, quaternion, fov_y_deg, principal_point, aspect, psnr`) next to the photos. Matches `pipeline/export/heldout.py`; `psnr: null` is accepted.
- `70_grid/grid.json`: `origin_xz, cell, spawn, bookmarks, tour` (dense tours are fine), optional `bounds`, `eye_height_m`, `floor_y`, `minimap.px_per_m`. Matches `pipeline/grid/stage.py`. `grid.png` and `minimap.png` are read for their size; grid values must be 0/128/255.
- `40_align/align.json`: `scale.{value, pct_68, method, cues, checked_against}`, `h_cap_m` (used for the eye height if grid.json has none). `50_train/train.json`: chosen variant's `psnr, ssim, lpips, cc_psnr` go into `metrics`. `40_align/rescale.json` (optional): when it has `checked_against` and `validation.n_used`, `scale.checked_against`, `scale.check_error_pct` (from `validation.auto_error_pct`) and `scale.check_n` are filled; without it the scene says the scale is an estimate. `input/meta.json` and `run.json` (`git`) are optional.
- Eye height: `--eye-height`, else grid.json `eye_height_m`, else `align.h_cap_m`, else 1.42; clamped to 1.2-1.8 m.

## Features beyond P0 that are built and tested

Preview-then-full loading (preview first, SD swapped in after 8 frames); HD button (swaps `splat_hd`, switches to the `full` preset; bundle.py leaves HD out when it is byte-identical to SD, and the viewer shows no button then); tour benchmark T (also `?bench=1`, `?benchsecs=` compresses it; result card with Copy button); reality check C; bookmarks 1-5; touch; fly mode F; `L` copies a spot link (`?scene=&pose=`); other scenes listed on the start card; weak-GPU fallback (the last governor step swaps to the 200 k-splat preview; two consecutive frames over 400 ms step down at once); software rendering (SwiftShader, detected from the GPU string) starts on the preview at pixel ratio 0.5 with a toast, `?q=sd` overrides; a download that shows no byte for 20 s is retried (the scene's own file if `?asset=` failed).

SwiftShader check (headless Chrome with `--use-angle=swiftshader`, Mac under load): SD at 800k splats gave 0.1-8 fps and the governor never moved before the emergency rule existed; with it the viewer drops to the preview within two slow frames. The preview itself still ran at about 0.6 fps in software on the loaded Mac, so a machine with no GPU acceleration will not be pleasant; this is a floor, not a promise.

Offline copy: `bundle.py --zip out.zip` writes the web folder plus `START_HERE.txt` (one command: `python3 -m http.server 8000`). The plain `http.server` path was tested (no Range support needed).

## Round 2 preview: places, search, route, follow, place editor, measure (30 Oct, P2/P3)

Built and tested on the fixture. The two public demo bundles (rh-honka with 6 places, rh-vrroom with 7) now carry hand-placed places from `viewer/places/`, so search, jump, route and follow show there (30 Oct, findings 16 and 45). Everything the user sees that belongs to Round 2 (search, jump to a place, route, follow) carries the label "Round 2 preview" (search box footer, place card, start card). Round 1 (roaming) does not depend on any of it: with `places: []` the search button and the search lines on the start card and help stay hidden; K (measure) and P (place editor) still work.

### What the user gets

| Key / control | Does |
|---|---|
| `/` or the Search button (top right) | opens the search box; empty box lists all places nearest first; `Enter` picks the top match, arrows move, `Esc` closes |
| place card (appears on pick) | name, kind, distance; buttons Jump `J`, Route `G`, Follow `Space` |
| Jump | teleport to the nearest walkable cell, facing the place (yaw and pitch; items: stands radius + 0.6 m back, 0.8-2.0 m) |
| Route | glowing chevron ribbon on the floor (y = floor_y + 0.03) and a line with arrows on the minimap; "11 m left, about 8 s on foot" at the scene's walk speed (1.4 m/s), live; re-plans when the player is more than 0.5 m off the path |
| Follow | the player auto-walks the route at walking speed, turns to face the place, "Arrived: <name>"; any move key stops it; a route blocked for 2 s re-plans once, then stops with a message |
| pin | marker with name and distance, scaled by distance (0.8x-1.35x) and always readable; off screen it becomes an edge arrow; labels step out from under panels; also a diamond on the minimap (click it to select; the large map labels every place) |
| `K` | measure: click two points (phones: the Measure button in the bottom bar, then tap); shows metres, the estimated 68% range from `scale.pct_68` (labelled an estimate; a tape check from `40_align/rescale.json` shows as "tape check +x.x%" instead) and the scale method; clicks use a splat raycast, floor plane when no splat is hit |
| `P` | add place (owner tool): click a spot, type a name, pick room / item / zone, optional other names; stored in this browser (localStorage, per scene) and added to the search at once; "Copy places JSON" copies scene places plus the new ones (a text box opens if the browser refuses the clipboard) |
| URL | `?place=<id>` selects, `&route=1` draws the route, `&follow=1` starts following (once the start card is closed) |

### Files and decisions

- `src/js/nav.js` (pure, Node-testable): 8-connected A* with octile distance on walkable cells (255 only), diagonal only when both side cells are walkable; a light wall-avoidance cost (0.15 / 0.08 / 0.03 for cells 1 / 2 / 3 cells from a blocked cell); a second and third A* that keep 12 cm and 7 cm from walls are used when they add under 3% to the smoothed length; string pulling by exact grid traversal (corners count) with 10 cm side clearance, then 5 cm, then none; a smoothed path that fails a 2 cm walkability sample falls back to the raw A* cells. The route ends at the stand spot, not inside the place. Connected components are computed once per grid; a place cut off from the player (different component) gives "No walking route: <name> is in an area you cannot reach on foot", or a route to the closest reachable spot with a note when that spot is within 1.5 m of the ideal one.
- Why these numbers: with the first wall cost (0.6 / 0.4 / 0.2 / 0.1) worst length was 1.165x the grid optimum (fails the 10% rule); the light cost plus the clearance levels gives 1.02-1.05x. The clearance levels are a design aim (waypoints 12 cm or 7 cm from walls where the floor allows); the tests check that every sample is on a walkable cell, not the clearance. Follow originally cut a corner into a wall and stalled (simulation test caught it); steering now aims at the farthest route point it can see in a straight line (grid line of sight, 4 cm clearance).
- `src/js/places.js` (pure): place schema reader (fills defaults, accepts `[x, z]`), fuzzy search over name and aliases: every query word must match a word of the name or alias; exact, prefix, substring, or one typo for words of 4+ letters (two for 8+; swaps count as one); words of 3 letters or less must match exactly or by prefix; alias matches rank 5% lower. `yawToward` follows the Walker convention (checked against it in the unit tests).
- `src/js/guide.js`: UI, route ribbon (ShaderMaterial, drawn after the splats with depth test off: Spark splats write no depth, so an occluding floor does not exist; the ribbon therefore also shows through walls when the route turns behind one), overlay canvas for pins and the measure line (skipped entirely when nothing is selected), picking, editor, follow controller. `main.js` changes are small: key dispatch, follow input, `guide.update` after each render, tool clicks and touch taps, `touchBlocked` selector.
- Picking: Spark's `SplatMesh.raycast` returns every splat the ray passes through. Thin surfaces (a TV, glass) give 1-3 hits, floaters give lone hits. The viewer casts 9 rays (centre plus 8 at 7 px) with `minRaycastOpacity` 0.12 and takes the first centre hit that at least 2 other rays confirm within 8 cm + 5% of the distance (else the best hit with one confirmation, else the floor plane y = floor_y, only when the ray hits it within 15 m). Cost: about 200 ms per click on 800k splats (M1). The first single-ray version found no surface on the fixture's wall display.
- Places live in scene.json `places[]` exactly as CONTRACTS 2.12 shows (`id, name, aliases[], kind, position[x,y,z], yaw_deg, radius_m`). The viewer also accepts `[x, z]` positions (pin height 1.0 for items, 0.0 otherwise). `yaw_deg` is used only when the stand spot is within 0.3 m of the place; the editor records the viewing yaw there. `radius_m` sets the stand-back distance for items.
- `tools/bundle.py --places FILE`: list or `{"places": [...]}`; validates (name, kind, position, radius, yaw; unique ids with `-2`, `-3` suffixes), warns when a place is more than 3 m from any walkable cell (usually a wrong x or z), writes the normalised list into scene.json. Without `--places` it uses `<run>/places.json` when that file exists. The H8 validator also checks every place (shape, unique id); a bad place is one of the 7 corrupted copies that must be rejected.
- Loader bar: it no longer blocks taps (`pointer-events: none`) and sits at the top on phones. Found by test R10: on a 390x844 phone it covered the start button while the full file was still loading, and the tap did nothing.
- Phone hint bar moves to the left under 720 px so the small map does not cover it.

### Known limits (read before quoting any of this)

- The places in the two public bundles were placed with the place editor on 30 Oct and checked in screenshots (one screenshot per place, jump view); no route was walked on those real grids except by tests R1-R5 and R8 (see Review fixes). Every number above is on the honka fixture. Its grid is a placeholder and walks through the table; its place positions were picked with the viewer's own pick tool and are only meant to exercise the code. Its floor is not at y = 0: the median of 19 splat hits on floor-ward rays was -0.67 m (most between -0.05 and -1.1, a few outliers further down), because the fixture's scale and floor were assumed. The route ribbon therefore floats above the visible floor in the fixture. On a real run `frame.floor_y` and the grid come from W4/W3; check `frame.floor_y` against the splat floor on the user's room before filming the route.
- The route follows the grid. If the grid marks something walkable that is not (or blocks a real doorway), the route is wrong in the same way; the Round 2 claim is only as good as `grid.png`.
- The measure tool reports the distance between two splat surfaces. Its accuracy is the scene's scale accuracy (`pct_68`) plus the depth error of the splat picked, which the tests do not measure (no ground truth on the fixture; the floor-plane pair is exact to the mouse pixel).
- Not tested: real pointer lock (Follow and the tools release it, so they do not depend on it), Safari, Firefox, a phone browser, `navigator.clipboard` over plain http (the text box fallback exists and was exercised headless).
- The ribbon draws over furniture it passes behind (no depth test). Chevron animation speed is fixed, not tied to the walking speed.
- Search is local and in memory (hundreds of places would be fine). No voice, no OCR, no semantic search: names and aliases only.

## verify.json (written by test/run.mjs)

```
{schema: "namma-space/verify@1", date, web, scene, browser, gpu, viewport,
 summary: {passed, failed, not_run},
 tests: [{id, name, pass (true|false|null = not run), value, metrics{...}, detail, seconds}],
 viewer: {devices: [{name, gpu, browser, first_frame_s, fps_mean, p95_ms, mb, splats, source}]},   // CONTRACTS 2.13 viewer.devices
 walls: {walks, pushes, penetrations, wall_stops}}                                               // CONTRACTS 2.13 walls
```
`pipeline/report/build.py` already reads `viewer.devices` and `walls.{walks,pushes,penetrations}` from this file.

## Needs and handoffs

- Root: wire `./ns bundle` to `bundle.py --run runs/<name> ...` and `./ns verify` to `node viewer/test/run.mjs --web runs/<name>/80_web --out runs/<name>/90_verify --pairs <pairs.json>`. Without `--pairs`, H4 uses three of the scene's held-out pairs instead of training poses (the value line says which). For the report's `viewer.devices[]` and `walls`, read them straight from `90_verify/verify.json` (same key names as CONTRACTS 2.13).
- **Places for the user's room (root).** Open the bundled scene (`python3 viewer/tools/serve.py runs/room/80_web`, then `?scene=room`), press `P`, click each room, desk, TV, door and so on (use the hand sketch for the list), type its name and optional other names, press Done, then "Copy places JSON" and save it as `runs/room/places.json`. Rebuild with `bundle.py --run runs/room ...` (it picks that file up; or pass `--places`). A `[x, z]` list in metres from the plan also works if clicking is not wanted. Aim for 3-6 places: rooms and big items; give each two or three everyday aliases (TV: tv, television, screen). Check each one with `/`, `J`, `G`. Places added in the viewer stay in that browser's localStorage until copied; a place whose id is already in scene.json is hidden from the local list.
- `./ns bundle` needs no change; add `--places runs/<name>/places.json` only if the file lives elsewhere.
- Check `frame.floor_y` and `grid.png` on the user's room before recording the route (see Known limits). If the ribbon floats or sinks, tell W4/W3; the viewer only reads `floor_y`.
- For the video: `?place=<id>&route=1&follow=1` opens a walk to a place; press the start button and it walks. `?keys=1` still shows the key overlay.
- The `video_url` for the WebGL2-missing card is filled by `--video-url`; empty means no link is shown. A rebuild without `--video-url` now keeps the link already in the bundle's index.html. `./ns bundle` passes none; root: pass it once the video is uploaded.
- No change to CONTRACTS was needed. Open question for root: `scene.json.splats.render.maxStdDev` is now read as the HD value; SD always caps at sqrt(5).

## Failures and false starts (for the failure table)

- First Spark/three import map: Spark imports `three/addons/postprocessing/Pass.js`, so the addons mapping is required (vendored that one file only).
- The first start card was taller than a phone in landscape; the start button fell off screen and the touch test could not tap it. Fixed with a sticky button and a compact layout under 560 px height.
- The CDP touch test dropped its look gesture on the minimap (the 220 px map covers part of the right half of a phone screen). Fixed by shrinking the map on small screens and moving the test point; also clear stale touch state when no finger is left.
- First H2 run with the 1.0 M-splat SPZ v4 file failed (54 fps, p95 33 ms) at sqrt(8); fixed by the sqrt(5) default (see Quality).
- `Network.emulateNetworkConditions` is not used for H9: the test server paces bytes instead, which also throttles requests made from Spark's workers.

- Route first version: wall-avoidance cost 0.6 / 0.4 / 0.2 / 0.1 made 3 of 600 synthetic routes more than 10% longer than the grid optimum (worst 1.165x). Lowered the cost and added the clearance levels (above).
- Follow first version aimed 0.6 m ahead on the route, cut the corner of a wall end and stalled against it (found by the Node follow simulation). Now it aims at the farthest route point in line of sight.
- Pick first version: one ray, 4 hits within 15 cm required; the wall display gave 1-3 hits and nothing was found. Now 9 rays and a confirmation rule (see Picking).
- The loader bar covered the start button on a 390x844 phone while the full file loaded (R10 failed on it). Fixed in CSS.
- A test expectation of mine was wrong twice (route length through a wall gap, an island test where the "closest reachable spot" rule correctly returned a route); the code was right, the tests were fixed.
- Picking floor-ward rays on the fixture returns y from -0.05 to -1.1: the fixture's splat floor is noisy and not at y = 0 (see Known limits).

## Open items

- H4b (browser vs gsplat render, >= 25 dB) needs reference renders. `make_fixture.py --render` writes them (`parity/parity.json` plus PNGs) but the GPU job was queued behind W3's training jobs when this note was written; see the last section of this file for the result if it ran.
- Not measured: any GPU other than the M1; Safari, Firefox, Android Chrome, WhatsApp in-app browser; real pointer lock.
- P2/P3 are built (places, search, measure, spot links, route, follow, place editor); see "Round 2 preview". Not built: multi-floor, doors that open, semantic search, voice.
- A hosted copy does not exist (no publishing was done). `dist/` is the offline copy: `python3 -m http.server` in that folder is enough.

## Review fixes 30 Oct

An independent review (each finding confirmed by a second reviewer) listed the problems below. A first fixer began and stopped when the server restarted. The second fixer checked that work (ran its tests, looked at the diff), finished what was half done and added what was left. One entry per finding number. Test ids are the ones in `test/tests.mjs`, `test/tests_guide.mjs`, `test/unit.mjs` and `test/test_bundle.py`.

**15 Governor drops quality when the frame rate is capped or paused: fixed.**
- `governor.js`: a step down is now a trial. It stays only if the mean frame rate rises by at least 10% in the next 2 s window. A capped rate (battery saver, half-rate vsync) cannot rise, so the step goes back, and no new step down happens until the rate falls to 80% of the capped one.
- Added by the second fixer: a cap can hide behind a loading hiccup. If the first window reads low, the first step looks like a gain and only the second step shows the cap. X13 caught this on the busy Mac (history `low, min, low`, ended at pixel ratio 0.85). Now, when a trial fails after earlier steps of the same run were kept, the governor measures the top of the run again and keeps the top if the rate there is within 10% (`verify`). X13 then ended on `standard`.
- A frame longer than 2 s counts as a pause and is dropped. The "keep at least 3 samples" rule is gone. `visibilitychange` and `pageshow` clear the frame timer and the samples (`main.js`); leaving the benchmark or the reality check clears them too.
- The emergency rule stays (two frames in a row over 400 ms step down at once).
- The "low" step is left out when it equals "standard" (weak GPUs start at 0.85). The "Quality: x" and "Reduced detail" toasts show only after a trial is kept.
- A step up that drops the rate under 40 again blocks that level for 60 s, so the viewer does not flip between the preview and SD.
- Tests: unit 29 of 29 (capped at 30, 35 and 38 fps; capped with a slow start; a real drop to 20 fps after a cap; a GPU-bound stressed run; hidden tab; emergency; ladder). Browser: X13 (page capped at 30 fps), X15 (6 s pause), H6 (`?stress=1`), X11 (preview step). Results are in the table below.
- Stressed run, measured with the governor off (M1 under load average about 25): 2.0 gave 17-25 fps, 1.4 gave 30-33, 1.0 gave 44-46, 0.85 gave 47-55 fps. On the first full run H6 stepped to 1.4 and then went back to 2.0, because 1.4 gave under 10% more than the 26.6 fps before. On the second run it kept 1.4. Both runs pass H6 (a step within 3 s); which one you see depends on the host load.
- Not done: no real battery-saver or Low Power Mode test (no such device here). The cap is simulated by limiting `requestAnimationFrame` in the page. A tab hidden for real was not tested either; headless Chrome keeps running frames.

**40 A failed SD download replaces a working preview with an error card: fixed.**
- `main.js loadSplats`: any failure (reset, drop, 5xx, 20 s silence) now gets one retry from the same URL (or the scene's own file when `?asset=` failed). If the retry also fails and the preview is on screen, the viewer keeps the preview, hides the loader, and shows a 9 s toast ("The full-quality file did not load. Showing the preview ... Reload the page to try again."). The error card shows only when nothing renders. The error card has a "Reload the page" button. A failed HD download keeps SD and shows a toast.
- Test X14 (4 cases with injected failures: HTTP 500 every time; connection cut after 1 MB every time; first request fails and the retry works; no preview in the scene and SD fails). In the first two the player stays on the preview and walks 0.9 m with W; in the third SD loads after the retry; in the fourth the error card has its Reload button.
- Not done: `./ns bundle` still passes no `--video-url` (`ns` belongs to root). bundle.py now keeps the link already in `index.html` on a rebuild without the flag (`test_video_link_survives_a_rebuild`). Root: pass `--video-url` once the video is uploaded.

**46 Close-to-wall poses show the dark clear colour: fixed as a mask, not as a cure.**
- `bundle.py` writes `splats.render.clear_color` into scene.json: the mean colour of up to 16 held-out photos, mostly grey (40% of the colour kept), brightness limited to 40-120. Fallbacks: the minimap, then a fixed grey (60, 58, 55). Public bundles: honka [125, 119, 113], vr_room [121, 120, 117] (mean of 8 held-out photos each).
- `main.js` keeps the dark page colour (11, 14, 20) until the first splat frame, because the first-frame probe compares pixels with it. Then it switches to the scene colour (`useSceneClear`). Test X18 checks both.
- This hides holes; it does not fill them. The verifier measured that H4b at rh-phone 000012 still scores about 20.5 dB. Not done: the verifier's `clipXY: 2` for Spark (removes the off-screen cull hole, +0.1 to +0.6 dB on the parity poses). I did not measure its frame-rate cost on a weak GPU, so I left it out.

**48 About 5 MB of JS loads before any progress feedback: fixed (text and preload; the libraries stay as they are).**
- `index.html`: the loader panel is in the HTML with the line "Loading the 3D viewer code. This takes longer on a slow link.", a sliding bar (CSS, off for reduced motion) and a `<noscript>` line. `main.js` changes the line to "Viewer code loaded. Reading the scene description." and "Loading the room files." while it boots; the splat byte count takes over once bytes arrive. `modulepreload` links for three, three.core, Spark and main.js start the downloads in parallel.
- Test X17 delays `spark.module.js` by 2.5 s and checks the text, the sliding bar, 4 preload links, and that the loader goes away after the load.
- Not done: minified builds (the verifier measured about 230 KB less gzipped, about 0.5 s at 4 Mbps, for re-hashing the vendor files the day before the deadline). The host has to gzip JS (GitHub Pages does; the offline zip does not).

**49 Touchscreen laptops get the phone path: fixed.**
- `main.js`: the touch path is on only when the main pointer is not fine and hover-capable (`(pointer: fine)` and `(hover: hover)` both match means desktop). A touchscreen laptop keeps "Click to start", the "Click to resume" pill, the desktop hint bar and pointer lock. A mouse that moves on a device that first looked like a tablet switches it to the desktop path. A start click by touch (`pointerType touch`) still skips the lock. The hint bar and the Measure button follow the same flag (`applyTouchUi`).
- Test X16: a page with `maxTouchPoints 5` and `ontouchstart` but a fine pointer gets the desktop path and one pointer-lock request after a mouse click; a 390x844 touch phone still gets "Tap to start". H5 and R10 (phone touch) still pass.
- Not tested: a real Windows touchscreen laptop.

**29 The scale badge says "measured" but shows a fixed prior: fixed.**
- New `scaleInfo()` in `scene.js` is the single source for the badge, the HUD chip, the measure tool and the start card. Without a tape check it prints "est. ±13%" and "Estimated scale, 68% range ±13%. It comes from a prior on camera height; it is not a measured error." The chip reads "scale est. ±13%".
- `bundle.py` now reads `40_align/rescale.json`. When it has `checked_against` and `validation.n_used`, scene.json gets `scale.checked_against`, `scale.check_error_pct` and `scale.check_n`, and the viewer shows "tape check +2.1%" (no range from the prior). Test `test_tape_check_reaches_scene_json_only_through_rescale_json`; unit test for the text.
- The start card line 6 says the badge is an estimate and not a measured error; it changes when the scene was checked. Tests: R6 and the unit test.
- Not done: the LiDAR-measured error (3.3% honka, 3.1% vr_room) is not in the badge. Those come from `eval/results`, not from the run.

**44 / 22 Wording, internal files and git markers: partly fixed.**
- Fixed in my files: start card step 1 no longer sends the judge into a wall ("Walk round the room for a minute. Walls and furniture stop you, and you slide along them."); step 6 and the K help line (finding 29); the lede reads "a Gaussian-splat reconstruction of a real room" (no "photoreal twin"; fixtures say only "reconstruction"); the HUD chip says "scale est. ±13%". The top of this file says AI agents wrote these notes and what "W1" to "W7" and "root" mean.
- Not mine (docs, README, pipeline, ns): the PoC "H: help", the "not built" lines for search and routes in the abstract, narration, captions, storyboard and limits card (places now exist in the two public bundles, so those texts need the "Round 2 preview" wording; see 16/45), the README note on honka's exit 30 and the up error, moving the dev logs.
- Git markers: scene.json `build.git` comes from `run.json` of the run (`2a5add6-dirty`, `a57c8c1-dirty`). Only a commit before the final room run fixes that. Root decides.

**36 Wrong dataset credit in the fixture: fixed.**
- `run_all.sh` now passes "MuSHRoom dataset (Ren et al. 2024), room honka". The fixture was rebuilt and `test/last_run/shots/r9_start_card.png` was replaced by the full run (it shows the new credit).

**19 / 47 The HD button reloads an identical file: fixed.**
- `bundle.py`: when `splat_hd.spz` has the same size and SHA-256 as `splat_sd.spz`, the HD entry is dropped, the file is not copied (about 16 MB less per scene) and a warning says so. The viewer shows no HD button. If the sizes, count and SH degree match but the bytes differ, the file is kept with a warning. The validator warns about a bundle whose hd is the same file as sd.
- The viewer also hides the button for an older bundle that lists an HD entry equal to SD (`hdSameAsSd`). Tests: `HdAndClear` in test_bundle.py (identical dropped, different kept, none), the unit test, X8 (button shown for a real HD file, hidden for an equal one).
- Public bundles after the rebuild: honka and vr_room have no HD file (their HD was a byte copy of SD). rh-phone was not rebuilt (it has a real SH3 HD).
- Not mine: the abstract's "HD splats / size" row (prints the same numbers as SD) and the storyboard's HD-toggle shot. Root should change both.

**16 / 45 Search, jump, route and follow hidden in every bundle: fixed for the two public demo rooms.**
- Places for rh-honka (6) and rh-vrroom (7) sit in `viewer/places/rh-honka.json` and `rh-vrroom.json`. Each entry carries `notes`: "placed by hand with the place editor (build team), 30 Oct". I placed them by driving the viewer's own editor (P, click on the canvas, name, other names, kind, Enter) from a stand point chosen in a screenshot; the click is the editor's nine-ray splat pick. Radius (stand-back distance) was set by hand in the JSON for the large items (table, TVs, mural, painting).
- Honka: Meeting table, Wall TV, Forest picture, Drinks shelf, Door, Armchair. vr_room: Green sofa, Colourful painting, Wall TV, Fire extinguisher, Door, Sink, VR equipment cabinet. Each has 2-5 other names (tv, television; couch; wash basin; ...).
- Checked by jumping to each place (`J`) and taking a screenshot (`places/shots/*.jpg`, 800 px, from the final bundles): the pin sits on the named object in all 13. Four changes came out of the check. (1) The first honka "Wall TV" pick sat at x = 5.19, 1.8 m behind the wall: the screen reflects the room and gives no depth. I took the pin from the soundbar under the screen instead (a raycast on a solid object: 14 splat hits on the centre ray, all 8 other rays agree). (2) The first "Windows" place sent the jump to 0.6 m from the blinds, a smear. I dropped it (honka has 6 places). (3) The first vr_room "Door" pin was not clearly on the door; I re-picked it on the door leaf. (4) The vr_room "Wall TV" pin sat on the screen's left edge; I re-picked it at the centre of the screen. The TV stand-back radii were raised (honka 1.0, vr_room 1.2) so the jump view shows the whole screen from the front.
- `bundle.py --places viewer/places/<run>.json` rebuilds the two public bundles (done here; the pipeline fixer can pass the same files). `ns bundle` does not pick them up by itself, because `ns` reads only `runs/<name>/places.json`. The places sit in the world frame of the current splat files. If the pipeline retrains or re-exports a run, jump to each place again (`J`) and look before shipping: a new frame can move every pin.
- R1-R5 and R8 now run on the real bundles: see the table below for the results. A route was not walked by hand on these grids, and the honka TV with the first radius (0.8) sent the jump to (3.72, 1.53), where the view was a smear of wall and blinds, probably inside or behind the wall. The grid is not mine; a radius of 1.0 avoids the spot.
- The docs still say search and routes are "not built" (finding 16 / 44): root changes them to "Round 2 preview: search, jump, route and follow over hand-placed places" for the two public rooms. The user's own room has no places until root adds them.
- Test: `PublicRoomPlaces` in test_bundle.py (5-7 places per file, unique ids, the provenance note, aliases, surface heights).

### Results after the fixes (30 Oct, about 17:00 IST; M1, headless Chrome 156, other builders kept the Mac busy: load average 15-35)

| Suite | Passed | Failed | Not run | Command |
|---|---|---|---|---|
| unit.mjs | 29 | 0 | 0 | `node viewer/test/unit.mjs` |
| nav_unit.mjs | 19 | 0 | 0 | `node viewer/test/nav_unit.mjs` |
| test_bundle.py | 29 | 0 | 0 | `python3 viewer/test/test_bundle.py` |
| fixture (`last_run/verify.json`) | 38 | 0 | 1 (H4b: the fixture has no gsplat reference renders, as before) | `sh viewer/test/run_all.sh` |
| rh-honka (`runs/rh-honka/90_verify/verify.json`) | 39 | 0 | 0 | `node viewer/test/run.mjs --web runs/rh-honka/80_web --out runs/rh-honka/90_verify --scene rh-honka --parity runs/rh-honka/60_export/parity` |
| rh-vrroom (`runs/rh-vrroom/90_verify/verify.json`) | 39 | 0 | 0 | same with `rh-vrroom` |

- All R tests (R0-R10) run and pass on both public bundles. Before this work R1-R5 and R8 were "not run: scene has no places" there, and R9 and R10 failed on the vr_room.
- honka / vr_room: H2 60 fps mean, p95 16.7 / 16.8 ms; H4b 34.22, 35.36, 33.10 dB / 33.53, 34.13, 34.24 dB (the grey clear colour did not change H4 or H4b: honka H4 23.1, 23.46, 21.46 dB as before); H9 first frame 2.80 s / 2.82 s, SD loaded 9.88 s / 9.79 s at 20 Mbps (16.1 / 15.7 MB); X3 tour benchmark 60 fps, p95 16.7 / 16.8 ms. R1: 72 routes (honka, worst 1.032x the grid optimum) and 84 routes (vr_room, worst 1.002x), 0 blocked samples on each; R3 16/16 and 18/18 queries; R4 walked 2.6 m and 3.4 m and ended facing the place.
- Governor on the three suites: X13 (30 fps cap) ended on `standard` at pixel ratio 1 in every run: history `low:down, standard:revert` (on the busy Mac once `low:down, min:down, standard:revert`); H6 stepped to 1.4 at 1.8-2.2 s and settled at `stress-mid`; X15 had no step judged on the pause.
- Earlier runs the same day failed, and I fixed the cause each time: X13 and X15 (governor, test too strict about a busy Mac), X8 (the stand-in HD entry equalled SD on a scene whose SD is already SH3), R3 and R5 (hard-coded fixture place names: now built from the scene's own places), R4 on vr_room (yaw measured from the end spot, which is 0.2 m off the stand spot: 6.4 deg at a 1.8 m stand-back; now measured from the stand spot).
- Numbers from the busy Mac are noisy: an earlier fixture run gave X3 38.5 fps and H6 settled back at 2.0. Quote the final runs above, not those.

### For root

- Commit before the final room run (finding 22): `scene.json build.git` shows `2a5add6-dirty` / `a57c8c1-dirty` for the public runs.
- `./ns bundle`: pass `--video-url` once the video is uploaded (finding 40), and pass `--places viewer/places/<run>.json` for the two public runs (or copy the file to `runs/<run>/places.json`). A rebuild without them loses the places and the video link; the places do not survive a retrain without a new look.
- Texts outside my folder that now disagree with the viewer: abstract template, narration, captions, storyboard, render_cards, the PoC "Known limits" and "H: help" (search and routes exist in the two public bundles; HD equals SD there; the badge is an estimate). Findings 16, 19, 22, 44.
- Findings 20 (first view tilted up) and 39 (wall stand-off) are not in this list. Only the wording of step 1 on the start card changed. The camera fov (`bundle.py`, 70) and the spawn pitch (`pipeline/grid/viewpoints.py`) are unchanged.
- The user's room: no places yet. Use the editor (P) as for the public rooms, save `runs/<room>/places.json`, rebundle, run the suite (R1-R10 then run on the real grid).
