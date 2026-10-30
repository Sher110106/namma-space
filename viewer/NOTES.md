# WP5 viewer and bundle: notes

Owner: W5. Everything here ran on the Mac (Apple M1, headless Chrome 156 with Metal ANGLE) unless a line says tyrone.
Numbers below are measured in this session on the test fixture; nothing here is a result on the user's own room.

## What is in the folder

```
viewer/
  src/            index.html, js/*.js, css/viewer.css   (the viewer; plain ES modules, no build step)
  vendor/         spark.module.js, three.module.js, three.core.js, addons/postprocessing/Pass.js, LICENSES/
  tools/bundle.py        run dir -> 80_web/ and validate (stdlib only)
  tools/serve.py         local static server with Range (stdlib only), default http://127.0.0.1:5178/
  tools/fixture/         make_fixture.py (numpy, runs on tyrone), spz_transcode.mjs (PLY -> SPZ with Spark's writer)
  test/                  run.mjs (runner), tests.mjs (H1-H9 + feature checks), lib/ (CDP driver, static server), run_all.sh
  test/fixtures/honka_run/   fake run dir (CONTRACTS 2.11 layout) built from the honka smoke run: the TEST FIXTURE
  test/last_run/         verify.json and shots from the last full run
  dist/                  bundle.py output for the fixture: a working site (26 MB; scene honka_demo)
```

Commands:

```
sh viewer/test/run_all.sh                       # rebuild dist from the fixture, run every test (about 3 min), write test/last_run/
sh viewer/test/run_all.sh runs/<name>           # bundle a real run into 80_web/ and test it; writes 90_verify/verify.json
python3 viewer/tools/bundle.py --run runs/room --title "Our room" --source "iPhone 4K30, 3 min 40 s" --license "CC BY 4.0" \
        --kind captured --video-url <url> --default           # writes runs/room/80_web/
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
| H6 | `?stress=1` (pixel ratio 2): governor stepped to 1.4 at 1.8-1.9 s after the first splat frame | <= 3 s |
| H7 | `?asset=` from a second origin with CORS: loaded, 800k splats; origin without CORS: falls back to the scene's own file | loads |
| H8 | bundle.py output: 73 checks, 0 failed; 6 of 6 corrupted copies rejected (missing key, size mismatch, missing grid, wrong count, wrong grid dims, missing held-out photo) | valid |
| H9 | server throttled to 2.5 MB/s: first frame 2.4-2.5 s (preview), SD fully loaded 8.7-8.8 s (preview 2.5 MB then SD 15.5 MB, in that order) | <= 18 s, <= 4 s |
| X1-X10 | WebGL2-missing card; keys R/H/M/?/Esc; tour benchmark T (59.7 fps, p95 16.7 ms); reality check C; bookmarks 1-5; `?pose` `?keys` `?q=low`; minimap click; HD toggle (mechanism test with the SD file standing in); `?scene=` and the error card; fly mode F | features |

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

So fill rate, not sorting, is what the M1 runs out of at 1.0 M splats. Decision: the SD preset ("standard") starts at sqrt(5) (Spark's own docs call it perceptually very similar); the HD preset ("full") uses `scene.splats.render.maxStdDev` (sqrt(8)). PSNR against photos did not change (H4 21.3/22.8/22.0 both ways). Governor ladder (`governor.js`): standard -> low (pixel ratio 0.85) -> min (0.7); stress adds 2.0 and 1.4 on top. Rule as in the plan: mean fps < 40 over 2 s steps down; median fps > 55 over 2 s steps up, with an 8 s hold-off after a step down. Pixel ratio starts at 1 (0.85 when the GPU string looks integrated or mobile, 0.7 for software rendering).

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
- `40_align/align.json`: `scale.{value, pct_68, method, cues, checked_against}`, `h_cap_m` (used for the eye height if grid.json has none). `50_train/train.json`: chosen variant's `psnr, ssim, lpips, cc_psnr` go into `metrics`. `input/meta.json` and `run.json` (`git`) are optional.
- Eye height: `--eye-height`, else grid.json `eye_height_m`, else `align.h_cap_m`, else 1.42; clamped to 1.2-1.8 m.

## Features beyond P0 that are built and tested

Preview-then-full loading (preview first, SD swapped in after 8 frames); HD button (swaps `splat_hd`, switches to the `full` preset); tour benchmark T (also `?bench=1`, `?benchsecs=` compresses it; result card with Copy button); reality check C; bookmarks 1-5; touch; fly mode F; `L` copies a spot link (`?scene=&pose=`); other scenes listed on the start card; weak-GPU fallback (the last governor step swaps to the 200 k-splat preview; two consecutive frames over 400 ms step down at once); software rendering (SwiftShader, detected from the GPU string) starts on the preview at pixel ratio 0.5 with a toast, `?q=sd` overrides; a download that shows no byte for 20 s is retried (the scene's own file if `?asset=` failed).

SwiftShader check (headless Chrome with `--use-angle=swiftshader`, Mac under load): SD at 800k splats gave 0.1-8 fps and the governor never moved before the emergency rule existed; with it the viewer drops to the preview within two slow frames. The preview itself still ran at about 0.6 fps in software on the loaded Mac, so a machine with no GPU acceleration will not be pleasant; this is a floor, not a promise.

Offline copy: `bundle.py --zip out.zip` writes the web folder plus `START_HERE.txt` (one command: `python3 -m http.server 8000`). The plain `http.server` path was tested (no Range support needed).

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
- The `video_url` for the WebGL2-missing card is filled by `--video-url`; empty means no link is shown.
- No change to CONTRACTS was needed. Open question for root: `scene.json.splats.render.maxStdDev` is now read as the HD value; SD always caps at sqrt(5).

## Failures and false starts (for the failure table)

- First Spark/three import map: Spark imports `three/addons/postprocessing/Pass.js`, so the addons mapping is required (vendored that one file only).
- The first start card was taller than a phone in landscape; the start button fell off screen and the touch test could not tap it. Fixed with a sticky button and a compact layout under 560 px height.
- The CDP touch test dropped its look gesture on the minimap (the 220 px map covers part of the right half of a phone screen). Fixed by shrinking the map on small screens and moving the test point; also clear stale touch state when no finger is left.
- First H2 run with the 1.0 M-splat SPZ v4 file failed (54 fps, p95 33 ms) at sqrt(8); fixed by the sqrt(5) default (see Quality).
- `Network.emulateNetworkConditions` is not used for H9: the test server paces bytes instead, which also throttles requests made from Spark's workers.

## Open items

- H4b (browser vs gsplat render, >= 25 dB) needs reference renders. `make_fixture.py --render` writes them (`parity/parity.json` plus PNGs) but the GPU job was queued behind W3's training jobs when this note was written; see the last section of this file for the result if it ran.
- Not measured: any GPU other than the M1; Safari, Firefox, Android Chrome, WhatsApp in-app browser; real pointer lock.
- Not built (P2/P3 in the plan): place search, measure tool, spot links, route preview. `Preview-then-full`, HD button, F, T, C, 1-5 and touch are built.
- A hosted copy does not exist (no publishing was done). `dist/` is the offline copy: `python3 -m http.server` in that folder is enough.
