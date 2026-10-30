# test/

What is here and how to run it.

```
test/
  data/conventions.json     COLMAP to three.js poses with expected outputs (root; Python and JS tests both read it)
  data/honka_subset/        30 public frames (MuSHRoom honka, CC BY 4.0 per the plan) for the smoke run
  expected/                 pass/fail gates for report.json (thresholds from the plan, no results)
  e2e/                      root's tests: runner, ns, report, conventions, colmap_io, stubs, adapters
  unit/<pkg>/               each package's own tests (sfm, ingest, align, train, export, grid, ...)
```

## Run the tests

One command per package, from the repo root:

    ./ns test e2e          # root: 88 tests, about 25 s on a Mac (no GPU, no ssh needed)
    ./ns test sfm          # likewise ingest, align, train, export, grid, geometry
    NS_E2E_REMOTE=1 ./ns test e2e     # adds the tests that talk to the GPU host (ssh, doctor, kill and resume, rescale)
    ./ns test train --remote          # runs test/unit/train inside a container on the GPU host

Tests use `unittest`, so no pip packages are needed on the Mac. Tests that need the GPU host or a dataset say so and skip
when it is missing. Run test files by path or with `unittest discover -s test/e2e`; the name `test` clashes with the
standard library, so `python -m unittest test.e2e...` does not work.

## Smoke test: one command, a few minutes of compute

    ./ns run --frames test/data/honka_subset/frames --name smoke --quick \
        --kind public --license "CC BY 4.0" --opt ingest:--force

`--quick` trains for 7k steps instead of 30k. `--opt ingest:--force` is needed because ingest wants 60 frames or more and
this sample has 30. The run needs a GPU host reachable with ssh (set `NS_HOST` and `NS_REMOTE_ROOT`) that has the images
`namma-colmap:0.2` and `namma-space-gpu:0.1` and the `bin/nsdocker` wrapper. `./ns doctor` checks all of that.

Then:

    python3 test/e2e/check_expected.py runs/smoke/report.json     # gates from test/expected/gates.json
    ./ns serve smoke                                              # http://127.0.0.1:5178/?scene=smoke

## Full reproduction

    ./ns doctor
    ./ns check inbox/IMG_0421.MOV
    ./ns run inbox/IMG_0421.MOV --name room --height 175 [--measure length=4.12 --measure width=3.35]

The plan expects 75-90 minutes for a 4-minute video on one shared RTX A6000 [E]. Measured times are in each run's
`timings.json` and `report.json`. Do not quote the plan's estimate as a result.

## Test ids

Root acceptance tests (FINAL_PLAN WP0) and where they run:

| Id | Test | File |
|---|---|---|
| 0.1 | kill during training, `ns run --from train`, earlier stages not re-run | test_runner.py, test_remote.py |
| 0.2 | drop ssh mid-run; the remote job continues; `ns pull` recovers | test_remote.py |
| 0.3 | fake low disk: a stage exits 20 within 5 s | test_runner.py, test_stage.py |
| 0.4 | two runs with different names at once | test_runner.py, test_remote.py |
| 0.5 | `ns doctor` within 60 s | test_remote.py |
| 0.6 | `ns run --frames <honka>` writes `80_web/` and a full `report.json` | rehearsal, see `pipeline/remote/NOTES.md` |
| 0.8 | `ns rescale` changes T_final, rebuilds grid and bundle, no retraining | test_runner.py, test_remote.py (stubs) |
