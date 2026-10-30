# test/

How to check that Namma Space works: a 10-minute smoke test on a small public sample, the full reproduction on a
public room, and your own video. Every command runs from the repo root on the Mac.

```
test/
  data/honka60/frames/      60 public photos (MuSHRoom honka, CC BY 4.0, credited in its README) for the smoke test
  data/honka_subset/        30 photos of the same room, used by test/e2e
  data/conventions.json     COLMAP to three.js poses with expected outputs (Python and JS tests both read it)
  expected/gates.json       pass/fail gates from the plan (thresholds, not results)
  expected/report.json      measured values of reference runs, with tolerance bands
  e2e/                      runner, ns, report, conventions, colmap_io tests; stub stages; adapters
  unit/<pkg>/               each package's own tests (sfm, ingest, align, train, export, grid, ...)
```

## What you need

- A Mac (or any Unix machine) with Python 3.10+, `ffmpeg`/`ffprobe`, `rsync`, `ssh`, Node 20+ and Chrome (for the
  viewer checks). No pip packages on the Mac.
- A GPU host reachable with ssh keys, set in `NS_HOST` (default `sher@10.1.18.190`) and `NS_REMOTE_ROOT`
  (default `/home/sher/namma-space`). It needs Docker with the NVIDIA runtime, the images `namma-colmap:0.2`,
  `namma-space-gpu:0.1` and `namma-space-gpu:0.2` (recipes in `docker/`), the wrapper `bin/nsdocker` and at least
  15 GB free disk. One GPU with 24 GB or more.

Check all of that in about 10 s:

    ./ns doctor

A FAIL line says what is missing. The Mac disk line warns under 10 GB free and fails under 3 GB.

## 1. Smoke test (10-25 minutes, depending on what else uses the GPU)

    ./ns run --frames test/data/honka60/frames --name smoke --quick \
        --kind public --license "CC BY 4.0" --attribution "MuSHRoom dataset (Ren et al. 2024), room honka"

What happens: the 60 photos go to the GPU host, then ingest, SfM, align, training (`--quick` = 7,000 steps instead of
30,000), export, grid; the results come back to `runs/smoke/`, and the Mac builds the web bundle (`80_web/`), runs the
headless-browser checks (`90_verify/`) and writes `report.json` and `report.html`. The last lines print the exit code
and the time of each stage.

Then:

    python3 test/e2e/check_expected.py runs/smoke/report.json --expected test/expected/report.json --case smoke
    ./ns serve smoke            # open http://127.0.0.1:5178/?scene=smoke ; click, then WASD and the mouse

What we measured on 30 Oct (shared RTX A6000, M1 Mac): SfM registers all 60 photos (0.46 px); held-out PSNR about
20 dB after 7,000 steps; a walkable grid of 6-7 m2 with full path coverage; the viewer at 60 fps. The run ended with
exit 30, which is expected for this sample: a 7,000-step model of 60 photos is a preview, so the training gate
(held-out PSNR 20 dB) or the export's cleaning gate can fail. Stage time was about 13 minutes; one run took 24 minutes
because its export waited 8 minutes for GPU memory held by another job. Exact values and tolerances:
`test/expected/report.json`, case `smoke`.

Exit codes: 0 good; 30 a quality gate failed but a bundle exists (the report says which gate); 11 a partial
reconstruction; 10 a capture problem (the report lists the time ranges to reshoot); 20 an infrastructure problem
(see `runs/<name>/logs/runner.log`).

If a run stops (ssh drop, Mac asleep), the GPU host keeps going. Run the same command again, or:

    ./ns status smoke
    ./ns run --name smoke                 # resume: finished stages are skipped
    ./ns run --name smoke --from train    # redo training and everything after it

## 2. Full reproduction on a public room (frames already on the GPU host)

The two MuSHRoom rooms we tested sit on the GPU host under `data/mushroom/room_datasets/`. `tyrone:<path>` tells
`ns` the input is already there, so nothing is copied over the network:

    ./ns doctor
    ./ns run --frames tyrone:/home/sher/namma-space/data/mushroom/room_datasets/honka/iphone/long_capture/images \
        --name honka --kind public --license "CC BY 4.0" \
        --attribution "MuSHRoom dataset (Ren et al. 2024), room honka, iPhone long capture"
    python3 test/e2e/check_expected.py runs/honka/report.json --expected test/expected/report.json --case honka
    python3 test/e2e/check_expected.py runs/honka/report.json          # the plan's gates
    ./ns serve honka

Full quality: 30,000 training steps, two variants (R0, and R1 with a depth loss), the better one by held-out PSNR is
shipped. The same command with the `vr_room` path is the unseen-room rehearsal (`--name vrroom`).

Measured on 30 Oct (details and tolerances in `test/expected/report.json`, cases `honka` and `vrroom`):

| | honka (335 photos) | vr_room (422 photos) |
|---|---|---|
| SfM registered, reprojection | 94.6%, 0.74 px (flagged partial: under 95%) | 99.5%, 0.45 px |
| Up error, scale error vs the dataset's metric poses | 2.0 deg, +3.3% | 1.8 deg, -3.1% |
| Held-out PSNR / SSIM / LPIPS (cc_psnr) | 21.35 / 0.812 / 0.225 (25.05) | 23.97 / 0.870 / 0.245 (26.53) |
| Web file (SD) | 629,993 splats, SH3, 16.1 MB | 646,048 splats, SH3, 15.7 MB |
| Walkable area, path coverage | 12.9 m2, 100% | 12.3 m2, 100% |
| Room extent (dataset's own figure) | 6.33 x 4.45 m (6.1 x 3.9 m) | 4.90 x 4.04 m (5.1 x 4.4 m) |
| Viewer, M1 headless Chrome (20 s walk) | first frame 0.42 s, 59.8 fps, p95 16.8 ms | first frame 0.30 s, 59.6 fps, p95 16.8 ms |
| Stage time | 107 min (SfM 17, training 86: two variants one after the other) | 73 min (SfM 24, training 46) |

Times depend on what else is using the GPU; each run's `timings.json` has its own. Use `--geometry` to also build
the optional TSDF mesh (`55_geometry/`).

## 3. Your own video

    ./ns check inbox/IMG_0421.MOV                    # about 20-60 s on a Mac: PASS / WARN / FAIL, each with the fix
    ./ns run inbox/IMG_0421.MOV --name room --height 175 [--measure length=4.12 --measure width=3.35]

If the video is too big for the Mac, copy it to the GPU host (for example `scp IMG_0421.MOV
sher@10.1.18.190:/home/sher/namma-space/inbox/`) and give the host path; the check then runs on the host:

    ./ns check tyrone:/home/sher/namma-space/inbox/IMG_0421.MOV
    ./ns run tyrone:/home/sher/namma-space/inbox/IMG_0421.MOV --name room --height 175

`--quick` gives a 7,000-step preview in about a third of the time. iPhone HEVC, 4K and HDR (HLG) videos are
handled: HDR is tone-mapped to SDR at ingest, portrait videos are rotated upright (tested on 30 Oct with a 4K HEVC clip,
an HLG clip and a portrait HEVC 10-bit HLG clip; see pipeline/remote/INTEGRATION_LOG.md).
To try the video path without a phone, build a phone-like clip (HEVC 10-bit, HLG, portrait rotation tag) from the
honka photos on the GPU host and run it the same way (measured: case `phone` in test/expected/report.json):

    ssh sher@10.1.18.190 'cd /home/sher/namma-space && bin/nsdocker rh-mkphone namma-space-gpu:0.1 python3 -m \
        pipeline.ingest.make_phone_fixture --frames /w/data/mushroom/room_datasets/honka/iphone/long_capture/images \
        --out /w/inbox/honka_phone_hlg.mov'
    ./ns run tyrone:/home/sher/namma-space/inbox/honka_phone_hlg.mov --name phone --quick

After the run: `./ns serve room`, and `./ns rescale room --use-measures` if you gave tape measures
(`--factor 1.05` makes everything 5% larger; `--scale 1.0` goes back to the automatic scale). A rescale re-runs only
the grid and the bundle, never the training.

## Run the tests

One command per package:

    ./ns test e2e          # root: runner, ns, report, conventions (about 25 s, no GPU, no ssh)
    ./ns test sfm          # likewise ingest, align, train, export, grid
    NS_E2E_REMOTE=1 ./ns test e2e     # adds the tests that talk to the GPU host
    ./ns test train --remote          # runs test/unit/train inside a container on the GPU host

Tests use `unittest`; packages that need numpy (train, export, grid, align) skip those tests on a bare Mac, so run
them with `--remote` or a venv that has numpy and scipy. Run test files by path or with
`unittest discover -s test/e2e`; the name `test` clashes with the standard library.

## Test ids

Root acceptance tests (FINAL_PLAN WP0):

| Id | Test | Where |
|---|---|---|
| 0.1 | kill during training, `ns run --from train`, earlier stages not re-run | test_runner.py, test_remote.py |
| 0.2 | drop ssh mid-run; the remote job continues; `ns pull` recovers | test_remote.py |
| 0.3 | fake low disk: a stage exits 20 within 5 s | test_runner.py, test_stage.py |
| 0.4 | two runs with different names at once | test_runner.py, test_remote.py |
| 0.5 | `ns doctor` within 60 s | test_remote.py |
| 0.6 | `ns run --frames <honka>` writes `80_web/` and a full `report.json` (0 empty required keys on 30 Oct) | section 2; expected/report.json |
| 0.7 | unseen room (`vr_room`) with the exact commands of section 2, unattended | section 2; pipeline/remote/INTEGRATION_LOG.md |
| 0.8 | `ns rescale` changes T_final, rebuilds grid and bundle, no retraining | test_runner.py, test_remote.py (stubs) |
| hang | a stage past its wall limit is stopped and recorded as exit 20 | test_runner.py |
| hang | a trainer that ignores SIGTERM or goes silent is killed; the 7k model ships | test/unit/train/test_stage_fake.py |
