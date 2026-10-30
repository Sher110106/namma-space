# pipeline/remote and the ns command: notes (root, WP0)

## What exists

- `pipeline/remote/ns_remote.py`: the stage runner on tyrone (system python3, standard library only). Docstring lists every option.
- `ns` (repo root): the Mac command line, Python 3 standard library only, one file. `./ns --help` lists the subcommands.
- Stub stages for tests: `test/e2e/stubs/`. Thin align adapter: `test/e2e/adapters/align_thin.py` (see "Adapters").

## How a run works

`./ns run` checks the video (Mac), syncs code, uploads `input/` and `input/meta.json`, starts `ns_remote.py` under nohup, and
polls `runs/<name>/status.json` over ssh. After each stage it copies the JSON and logs to the Mac (light pull). At the end it
does a full pull, then bundle, verify and report on the Mac. The Mac runs `caffeinate -i` for the length of `ns run`.

The runner skips a stage when its done-marker JSON exists, so a killed or dropped run resumes by running the same command.
`--from <stage>` forces that stage and every later one: their old outputs are moved to `<dir>.prev` first, deleted when the
stage succeeds, and restored when it fails.

Exit codes follow CONTRACTS 2.11. Stage results combine as 20 > 10 > 30 > 11 > 0. A code 10 or 20 stops the run;
11 and 30 continue (a bundle should still exist). A stage that exits 0, 11 or 30 without writing its JSON counts as 20.
Any exit code outside 0/10/11/20/30 counts as 20.

Stage modules that do not exist yet are skipped with a message. The final exit is then 20 ("incomplete") unless the
stage is optional (geometry) or `--allow-missing` is given. A stage whose inputs are missing is skipped the same way.

## Interfaces the runner assumes (tell root if a package differs)

- `python3 -m pipeline.<stage> --run /w/runs/<name>` (container path). Only `train` gets `--quick`. Every stage sees
  `NS_QUICK=1` and `NS_RUNNER=1` in its environment (set with `env` in front of the command, since `nsdocker` passes none).
- Done markers: `10_frames/frames.json`, `20_sfm/sfm.json`, `40_align/align.json`, `50_train/train.json`,
  `55_geometry/geometry.json`, `60_export/export.json`, `70_grid/grid.json`.
- Exit codes 0/10/11/20/30. Stage stdout goes to `logs/<stage>.log`.
- A stage may print `NS_RESULT {json}` as a last line (ingest and align do). If it exits 0 and the JSON has
  `"gate_failed": true`, the runner records exit 30 for the stage, keeps going, and ends the run with 30. W2's align uses
  this (it exits 0 and sets `gate_failed`; with `--strict` it exits 30 itself).
- `ns run --opt STAGE:ARG` passes one argument to one stage. The 30-frame rehearsal needs `--opt ingest:--force`
  because ingest wants at least 60 frames.
- Rescale: `ns rescale` runs `python3 -m pipeline.align.rescale --run <dir> [--use-measures | --scale=K | --factor=K]`, then
  re-runs the grid stage. W2's `rescale.py` does not exist yet, so only the stub version is tested. It must update
  `T_final` in `60_export/export.json` (or tell root where else) because the grid stage reads it from there.
- The Mac-side bundle step calls `viewer/tools/bundle.py --run <dir> --name <name> --out <dir>/80_web --default` plus
  provenance flags from `input/meta.json`; verify calls `node viewer/test/run.mjs --web ... --out ... --scene <name>`.

## Measured (stubs and real)

All figures are from this build session (30 Oct 2026).
- `./ns doctor`: 11 s end to end. ssh 0.4 s. Copy to tyrone 25.1 MB/s up, 20.9 MB/s down (100 MB each way).
- Test 0.3: with `--fake-free-gb 3` the runner exits 20 in under a second (limit 5 s).
- Test 0.1 with stubs through ssh and `ns run --from train`: earlier stage JSON files keep their timestamps; train ran twice, the
  earlier three once each.
- Test 0.2 with stubs: after killing `ns` and its ssh children mid-training, the remote runner finished (exit 0) and
  `./ns pull` recovered every file.
- Test 0.4: two runs at once finished without touching each other; a second runner on the same name is refused (exit 20).
- Test 0.5: `./ns doctor` takes 11 s and prints ssh, disk, GPU memory, image digests, Mac tools and the copy speed.
- Test 0.8 with stubs: `./ns rescale --factor 2` takes about 10 s including the pull; train ran once, grid twice.
- `./ns test e2e`: 88 tests, about 24 s on the Mac and 16 s on tyrone (8-10 skipped without NS_E2E_REMOTE or the honka model).

## Rehearsal on 30 honka frames (`rh-honka30`, test 0.6 attempt)

Command: `./ns run --frames test/data/honka_subset/frames --name rh-honka30 --quick --no-check --opt ingest:--force
--runner-flag="--script align=test/e2e/adapters/align_thin.py" ...` (30 frames, every third from frame 101).
It did not stop anywhere: all stages ran and the run ended with exit 30 (quality gates), with a bundle and a report.

| Stage | Result |
|---|---|
| ingest | first try exit 10 ("30 frames, need 60"), which is correct; with `--opt ingest:--force` 1.3 s |
| sfm | 322 s including the wait for a GPU slot; variant C chosen, 20 of 30 registered (66.7%), 0.585 px |
| align | root's thin adapter, 0.8 s. W2's real align was run separately on the same SfM (below) |
| train | 710 s for r0 and r1 one after the other (7k steps each; no second GPU slot was free). Held-out PSNR 14.36 dB, so exit 30 |
| geometry | skipped, module missing (optional) |
| export | 10.2 s, exit 30 (floor not measured, clean drop over the gate) |
| grid | 1.4 s, exit 30 (path coverage 0.55 in grid mode, fell back to free fly) |

Remote total 1046 s (about 17 min, of which several minutes were queueing for GPU slots). Then: full pull (0.01 GB),
`bundle.py` (68 checks, 0 failed, 6.6 MB), report. `./ns verify` (W5's harness, 1 min 51 s): 18 passed, 2 failed
(H4 held-out PSNR 14.6 and 19.4 dB on this weak model; H6 governor), 1 not run. The report had one empty required key
(`grid.room_extent_m`, null in grid.json because no walls were found). The weak result comes from the subset, not the
pipeline: every third frame leaves too little overlap, so SfM registered only 20 images. A subset of consecutive frames should
register better; a stride-1 and a stride-2 subset were queued to find out (see below).

Real align (W2) on the same SfM (`rh-honka30r`, `--from align --to align`): 273 s mostly queueing, exit 0, `gate.passed` false
(up cues disagree by 15.2 deg), H_cap 1.40 m, scale 0.659.

## Failures and fixes along the way

- `--fake-free-gb` first only reached the child stages, not the runner's own guard. The Mac test passed anyway because the
  Mac disk really is below 15 GB. Running the same tests on tyrone (40 GB free) exposed it. Fixed; both platforms pass.
- `namma-colmap:0.1` has no python3, so `pipeline.sfm` could not start in it. W1 built 0.2; the stage table now uses it.
- A runner killed with `kill -9` left `ns run` polling forever. `ns` now checks the runner pid every 20 s and exits 20 with a
  resume hint. Test: `test_remote.py::test_dead_runner_is_detected`.
- `ns pull` first demanded 3 GB free; the Mac has under 1 GB. It now sizes the transfer with `rsync --dry-run` and refuses
  only if the data would not fit.
- Mac free disk fell from 1.8 GB to 0.4 GB during the session. Not ours (our `runs/` and temp files are under 1 MB),
  but pulls, zips and the video build need room.
- The GPU cap: `nsdocker` waits for one of 2 GPU slots and 3 containers in total. With W1, W2, W3 busy, the first real
  stage of the rehearsal queued for 5 minutes. Stub stages should use `--no-docker` to avoid waiting.

## Not done or not tested

- `ns publish --execute` builds a gh-pages branch and pushes it. It has never run (publishing is forbidden in this build);
  only the dry run and the typed-phrase gate are tested.
- `ns package` lists what is missing (the PDFs, the video, the repo zip before the first commit) instead of inventing it.
- `ns verify` and the bundle step were only run on stub data or not at all (see the rehearsal section in test/README.md).

## Integration changes on 30 Oct morning (see INTEGRATION_LOG.md for the evidence)

- **Input already on tyrone.** `ns run tyrone:/abs/IMG.MOV --name N`, `ns run --frames tyrone:/abs/dir --name N`,
  `ns check tyrone:/abs/IMG.MOV`. Under `/home/sher/namma-space` the input is a relative symlink in `runs/N/input/`
  (containers see it under `/w`); elsewhere it is hard-linked, or copied if that fails. The check of a remote video runs
  in `namma-space-gpu:0.1` on tyrone (no ffmpeg on the host) and `00_preflight/check.json` comes back to the Mac.
  Drop big videos in `/home/sher/namma-space/inbox/`.
- **Wall limits.** The runner stops a stage that runs longer than `STAGE_LIMIT_MIN` (ingest 45, sfm 150, align 30,
  train 240, geometry 60, export 60, grid 20, rescale 15 min; `--stage-timeout STAGE=MIN`), counted from when its
  container is seen running, and gives up after `--queue-limit-min` (240) of waiting for a slot. Exit 20, reason in
  the stage note. The train stage has its own per-variant limit and a stall watchdog (pipeline/train/NOTES.md).
- **Partial = 11.** A stage JSON with `"partial": true` and exit 0 becomes 11 (sfm under 95% in one model).
- **Optional stages** (geometry) never stop the run: a failure is `exit-N-ignored` plus a warning.
- **Mac steps count.** `ns run` ends 20 if bundle or report fail, 30 if the viewer checks fail.
  With `--to STAGE` short of the end, the Mac steps are skipped.
- **Pull** leaves out `60_export/*.ply` and `55_geometry/*.ply` (use `--with-ply` or `--all`).
- **Verify** passes `60_export/parity` to the viewer harness when present (test H4b).
- **Rescale** now reaches the grid and the bundle: `rescale.py` writes the user scale into `export.json T_final`
  (`T_final_export` keeps the export stage's own) and scales the held-out and parity poses. `ns rescale --factor K`
  multiplies; `--scale K` sets the total factor (1.0 = automatic).

## Review fixes 30 Oct (pipeline fixer; findings in namma-space-work/review/FINDINGS_pipeline.md)

A first fixer started these and was cut off at about 15:50; a second one checked its work, ran its tests and
finished the rest. Per finding:

- **#9 camera-path filter** (fixed). `pipeline/sfm/sanity.py` measures steps against capture time and judges each
  stretch by the local path, not by its length; `sfm/stage.py` passes `t` from frames.json. Scored on the saved
  models against MuSHRoom poses with the new `pipeline/sfm/sanity_gt.py`; tables in pipeline/sfm/NOTES.md. On the
  models that were chosen: rh-phone C went from 31 dropped (23 of them correct) and 16 wrong kept to 24 dropped
  (0 correct) and 0 wrong kept; rh-honka B from 18 (4 correct) and 1 wrong kept to 15 (0) and 0; rh-vrroom had no
  wrong cameras and nothing is dropped either way. Unit tests: test_sanity.py (14 pass).
- **#4/#10 training time** (fixed). The bench at 1919x1278 projects 64.4 min per 30k steps (1600 px: 49.2 min), under
  root's 75-min bar, so frames stay at up to 1920 px. Defaults: r0 only (`--variants r0,r1` for both), 150 min per
  variant, runner train limit 330 min. pipeline/train/NOTES.md "1920 px timing". Tests: test/unit/train (27 pass in
  the container).
- **Root decision 3, frozen code** (done). `ns run --repo-dir repo-room` copies the tree to `<root>/repo-room` (an
  older copy is moved to `.old-<time>`, never deleted) and the runner, the wrapper and every container use it
  (`/w/repo-room`); `--no-sync` reuses it; `ns rescale` reuses the copy a run was built from (run.json `repo`).
  Test: test/e2e/test_frozen_repo.py (stub stages in real containers, passes). The fix-honka and fix-vrroom rebuilds
  ran from `repo-fixrun`; `docker inspect` showed the container's working dir as `/w/repo-fixrun`.
- **#6/#23 Mac disk** (fixed). bundle, verify and report stop with exit 20 below `NS_MAC_MIN_FREE_GB` (1.5 GB) and say
  what to do; nothing on tyrone is lost. `ns run` with a Mac video uploads it to tyrone's `inbox/` and links it into
  the run; `--move` deletes the Mac copy after the SHA-256 on both sides match; a half-done upload is finished, not
  refused. Tests: test_ns_cli.py `test_mac_steps_refuse_when_the_disk_is_nearly_full`; test_remote.py
  `test_mac_video_goes_to_the_inbox_move_deletes_it_and_package_on_host_places_it` (passes against tyrone).
- **#7/#14/#26 inputs used** (fixed in code; uploading the room's own files stays a manual step). `ns package` reads
  meta.json `files[]` (now with `on_host` and `mac_path`) for the run and its take B. With `--with-inputs` it copies
  takes still on the Mac (if the disk allows), the `--sketch` photo(s) and meta.json into `04_Inputs_Used/`, writes
  `INPUTS.txt` (size, status, place, SHA-256) and lists every input it could not copy with its size and place.
  `--on-host` writes the package on tyrone (`runs/<name>/package/`), with the takes hard-linked from the inbox, and
  writes MANIFEST.txt there. Tests: two in test_ns_cli.py, one in test_remote.py (all pass).
- **#13 setup** (fixed for our part). `docker/nsdocker` (root and code folder from `NS_ROOT`/`NS_WORKDIR`),
  `docker/install_host.sh`, `docker/fetch_cache.sh` (hash check ran OK on tyrone's cache); `ns` passes `--ns-root`
  so `NS_REMOTE_ROOT` reaches the runner; `ns doctor` honours `NS_CHROME`. install_host.sh plus the wrapper were run
  in a scratch root on tyrone and started a container there. Setup steps: test/README.md "What you need".
  The root README is not ours.
- **#27, #30, #35 test/README claims** (fixed). The 4K sentence now says what was tested (ingest on synthetic HLG
  clips, SfM at 1920 px, a 9k-step training bench) and that no 1920 px run has gone through export and verify;
  vr_room is called a rehearsal room with three logged restarts, and test 0.7 is marked not passed as specified;
  the table cites MuSHRoom supplementary Table 1 and says which code each column ran.
- **#24 HDR message** (fixed). check_mac.py tells the user to turn HDR Video off and reshoot, and says the tone map
  was tested only on SDR clips mapped into HLG.
- **#25 scale gate** (fixed). With only the camera-height cue, align now says the scale rests on one prior and does
  not pretend to check it; a typo in `--height` still gates; a second cue that disagrees gates
  ("scale cues disagree"). Tests: test/unit/align/test_scale_honesty.py (in the 23 align tests, pass).
- **#19/#38/#47 HD equal to SD** (fixed). Export writes no HD file when SD already holds every cleaned splat at SH3,
  and drops an HD file that comes out byte-identical to SD; `export.json` says why in `hd_skipped`.
- **#20/#43 spawn and spill** (fixed). Spawn, bookmarks and stop views take the source camera's pitch clamped to
  -12..-5 deg instead of 0; the spawn is the well-seen deep view that stands far from the middle of the walkable area
  and looks at it; walkable cells more than 0.15 m past the outer walls are cut unless within 0.35 x H_cap of the
  capture path (the doorway), and only the part joined to the path is kept. fov 70 -> 60 is in viewer/tools/bundle.py
  (viewer fixer). Bookmark names stay "View n": nothing can name them automatically.
- **#39 eye stand-off** (fixed). The eye now stops 0.28 x H_cap (0.39 m at 1.4 m, at most 0.45 m) from splats at or
  above 0.75 x H_cap; low furniture keeps 0.23 m, and cells within 0.25 m of the capture path ignore the stand-off so
  doors stay open.
- **#17/#42 minimap** (fixed). Splats down to opacity 0.05 are drawn (was 0.3), painting is limited to floor and
  obstacle cells grown by 0.3 m, and bare walkable pixels get the median walkable colour. Grid tests: 49 pass.

Test runs at the end: `./ns test e2e` 102 tests OK (14 skipped, the remote ones); with `NS_E2E_REMOTE=1
NS_E2E_PREFIX=fix-e2e` all remote tests pass (two failed on the first try: an rsync exit 23 while other tools were
writing files, now retried, and a newline bug in the on-host package script, fixed; both re-run and pass). Unit
tests: sfm 56, train 27, align 23, export 22, grid 49 pass. ingest: 18 run in the container and 1 fails,
`test_B1_sdr_1080p` fails (median match 8.56 against a bound of 6.0). pipeline/ingest/stage.py and that test are
unchanged from HEAD, so this was not caused by these fixes; not investigated.

### Rebuilds (30 Oct, 16:19-17:28)
`./ns run --frames tyrone:<honka images> --name fix-honka ... --repo-dir repo-fixrun` (from scratch) and
`./ns run --name fix-vrroom --from align --repo-dir repo-fixrun --no-sync` on a copy of rh-vrroom whose
10_frames, 20_sfm and 30_undist are hard links (the new filter drops nothing on vr_room's models, so SfM did not need
to run again; align re-ran for the #25 flag). The rh-* runs were not touched. Pulled with `./ns pull`, then
`./ns bundle`, `verify`, `report` on the Mac. Places: `runs/fix-vrroom/places.json` is viewer/places/rh-vrroom.json
unchanged (the camera fit between the two runs is exact); `runs/fix-honka/places.json` is viewer/places/rh-honka.json
moved into the new frame, because the new align turned honka's frame by 180 deg about +Y (yaw from walls has two
signs), scale x1.0098, tilt 1.3 deg, plus fix-honka's T_final floor shift of +0.028 m (fit on 313 shared cameras,
median residual 1 mm; the transform is in runs/fix-honka/places_transform.json, the one-off scripts in /tmp/nsfix/scripts).

| | rh-honka (old) | fix-honka | rh-vrroom (old) | fix-vrroom |
|---|---|---|---|---|
| Exit | 30 | 11 (SfM partial, 94.6%) | 0 | 0 |
| SfM chosen, kept, filter drops | B, 317 of 335, 18 | B, 317 of 335, 18 (0 correct, 2 wrong kept) | D, 420 of 422, 0 | same SfM |
| Up / scale error vs MuSHRoom (metric_check) | 2.0 deg / +3.26% | 1.31 deg / +4.29% | 1.85 deg / -3.1% | 1.85 deg / -3.1% |
| Align gate | failed (up cues 6.2 deg apart) | passed; flag: scale rests on one prior | passed | passed; same flag |
| Train (chosen) | r1 of r0,r1: 21.35 dB, SSIM 0.812, LPIPS 0.225 | r0: 21.51 dB, SSIM 0.822, LPIPS 0.220 | r0: 23.97 dB, 0.870, 0.245 | r0: 24.04 dB, 0.871, 0.241 |
| Train time | 5,156 s (two variants) | 2,245 s | 2,749 s | 2,655 s |
| SD / HD | SH3 16.1 MB / identical copy | SH2 13.3 MB / SH3 15.7 MB | SH3 15.7 MB / identical copy | SH3 15.5 MB / none (`hd_skipped`) |
| Bundle (du) | 24 MB (rebundled by the viewer fixer, HD copy dropped) | 37 MB (SD, HD, preview) | 24 MB (same) | 24 MB |
| Walkable, coverage | 12.9 m2, 100% | 12.4 m2, 100% | 12.3 m2, 100% | 10.7 m2, 100% |
| Walkable cells reach (z max) | 2.17 m | 1.73 m | 4.08 m (1.5 m past the last camera, out the door) | 3.02 m (604 cells past the walls cut) |
| Spawn pitch | 0 | -5 | 0 | -11.3 (the doorway start, facing in) |
| Minimap: bare walkable px / painted px over 1 m outside walkable | 30.6% / 19.9% | 0% / 4.4% | 12.4% / 15.5% | 0% / 4.3% |
| verify | 39 of 39 (viewer fixer's re-run) | 39 of 39 | 39 of 39 (same) | 39 of 39 (first try H2 failed at 54.6 fps while the e2e suite ran; re-run 57.5, then 59.7 after a rescale) |
| Total stage time | 107 min | 60 min | 73 min | 46 min (from align) |

Both new reports pass all 23 bands of their case in test/expected/report.json. fix-honka still fails the plan's
22 dB training gate. `./ns rescale fix-vrroom --scale 1.0` (17:24) ran rescale and grid from `repo-fixrun` (the
wrapper path in the logs), then bundle, verify (39 of 39) and report.

### For root
- Swap or not: fix-honka and fix-vrroom pass; rh-honka and rh-vrroom are untouched. If you swap them in, use
  runs/fix-honka/places.json, not viewer/places/rh-honka.json (the frame turned 180 deg). The bundles are named
  fix-*; a rebundle under another name needs `bundle.py --name`.
- tyrone's bin/nsdocker is the old copy; docker/nsdocker adds NS_ROOT/NS_WORKDIR and env limits with the same
  defaults. Only frozen-copy runs use the new one. Installing it (docker/install_host.sh) is your call.
- Scratch on tyrone: runs/fix-honka (1.7 GB), runs/fix-vrroom (1.8 GB, early stages are hard links into
  rh-vrroom), runs/fix-bench, runs/fix-sanity, repo-fixrun (120 MB; the fix-* runs record it). 26 GB free at 17:30.
- Room frames in test/data (#14) need Sher's yes; nothing was added.
- For the room run: `--repo-dir repo-room` (README section 3). r0 only and 1920 px are now the defaults; with a
  shared GPU, `--opt train:--long-side=1600` is the fallback.
