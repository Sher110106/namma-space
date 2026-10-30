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
