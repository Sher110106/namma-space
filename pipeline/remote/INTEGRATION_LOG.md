# Integration log (integrator, 30 Oct 2026)

Times are IST. Each entry: what ran, the result, the fix. Numbers come from the files named; nothing is estimated
unless marked [E].

## 09:14 State at start
- tyrone: 38 GB free, GPU 2.2 GB used of 48 GB, no `ns-*` containers. No `runs/room-*` yet.
- `runs/w3-honka-norm/train_stage.log`: 7,703 lines, almost all "wall limit 3600 s reached; terminating ['r0']",
  printed every 2 s from 04:55 to 09:12. The trainer's own log (`50_train.tmp/r0/log.txt`) stopped growing at 04:24 at
  step 19,746 of 30,000. The kernel log shows two memory-cgroup OOM kills of `colmap` (18 and 21 GB anon RSS, 20 GB cap)
  at 04:07 and 04:11 in another container, nothing at 04:24. Why the trainer stopped is not known.

## 09:15 Fix 1: train stage could loop forever (task 1)
Cause: `run_jobs` sent SIGTERM once per poll and waited for the process to exit; a trainer that does not exit on
SIGTERM (hung in the driver, or ignoring it) kept the loop alive, and the wall limit counted from the stage start, so
the message repeated every 2 s.
Fix (`pipeline/train/__main__.py`):
- per-variant wall limit from the variant's own start (`--max-minutes`, default 90 full, 30 quick);
- stall watchdog: no trainer output for `--stall-minutes` (default 15) = hung;
- SIGTERM to the process group, SIGKILL after 30 s, and after 60 s more the job is abandoned and the loop moves on;
- the variant record says `status: timeout|stalled`, `end_reason`, `abandoned`; train.json `warnings` says it too;
- a stopped variant that has its 7k evaluation becomes a partial model and the stage ships it as the last rung of the
  fallback ladder (exit 30). Before, that "DEGRADED (7k model only)" branch could not be reached (the record read the
  30k stats only, so `psnr` was null). The r0nb retry now runs only after crashes or NaN, not after a timeout.
Tests: `test/unit/train/test_stage_fake.py` gains a fake trainer that ignores SIGTERM and one that goes silent;
25 tests pass (`pipeline/export/.venv/bin/python -m unittest discover -s test/unit/train`).

## 09:18 Fix 2: a stage-level wall limit in the runner (similar hangs anywhere)
`pipeline/remote/ns_remote.py`: each stage has a limit counted from when its container is seen running
(`STAGE_LIMIT_MIN`: ingest 45, sfm 150, align 30, train 240, geometry 60, export 60, grid 20, rescale 15 min;
`--stage-timeout STAGE=MIN` overrides) and a 240 min cap on waiting for a container or GPU slot. Past it the runner
stops the container by name (`docker stop`, then `docker kill`) and records exit 20 with the reason.
`./ns test e2e`: 93 tests pass.

## 09:19 `ns run` accepts input already on tyrone (task 2, 4)
`./ns run tyrone:/abs/path/IMG.MOV --name N` and `./ns run --frames tyrone:/abs/dir --name N`. Under
`/home/sher/namma-space` the input becomes a relative symlink in `runs/N/input/` (containers see it under `/w`); elsewhere
it is hard-linked or copied on tyrone. Nothing crosses the network. The preflight check of a remote video runs on tyrone
inside `namma-space-gpu:0.1` (the host has no ffmpeg) and `00_preflight/check.json` is copied back.
`./ns check tyrone:/abs/path/IMG.MOV` does the same without a run.

## 09:20 rh-honka launched (task 2)
    ./ns run --frames tyrone:/home/sher/namma-space/data/mushroom/room_datasets/honka/iphone/long_capture/images \
      --name rh-honka --kind public --license "CC BY 4.0" \
      --attribution "MuSHRoom dataset (Ren et al. 2024), room honka, iPhone long capture" \
      --title "MuSHRoom honka (public test room)" --source "MuSHRoom honka, iPhone long capture, 335 frames" \
      --opt train:--parallel=no
`--opt train:--parallel=no` keeps one GPU slot free for root's room run (host rule); a user run would leave it out.
Ingest 6.1 s.
Runner test added: `test/e2e/test_runner.py::ExitCodes::test_hung_stage_is_stopped_at_its_wall_limit` (a stub that
sleeps 600 s is stopped at a 0.03 min limit, exit 20, note "stage limit"); passes.

## 09:22-09:25 Video path: remote and Mac checks (task 4)
| Command | Result | Time |
|---|---|---|
| `./ns check tyrone:/home/sher/namma-space/data/fixtures/videos/b7.mp4 --name rh-b7check` | PASS (3840x2160 HEVC 30 fps 180 s, blur 89%, exposure 3.5%) | 36 s wall, 33 s to check |
| `./ns check tyrone:/home/sher/namma-space/data/fixtures/videos/b2.mp4 --name rh-b2check` | WARN (60 s; HLG, tone-mapped) | 10 s |
| `./ns check /tmp/nsint/videos/b7.mp4` (Mac copy) | PASS, same verdicts; luma 65-184 vs 67-186 on tyrone (ffmpeg 9 vs 4.4) | 19 s |
`ns run tyrone:/home/sher/namma-space/data/fixtures/videos/b7.mp4 --name rh-b7 --quick ...` launched 09:24.
Note: `ns-w7-mip360` (eval builder) holds a GPU slot, so rh-b7's GPU stages queue behind rh-honka's.

## 09:26 Sample for test/data (task 5)
`test/data/honka60/frames`: honka `frame_00013.jpg` to `frame_00131.jpg`, every second frame, 60 photos, 12 MB,
credited in `test/data/honka60/README.md`. The GitHub README of MuSHRoom asks for a citation but states no licence;
CC BY 4.0 is what the plan recorded from Zenodo. Not re-checked on Zenodo today.

## 09:28 ns doctor
1 FAIL: Mac disk 1.9 GB free. Everything else PASS (tyrone 37 GB free, 27.9 GB GPU free, images present);
WARN `namma-node:0.1` absent (export uses splat-transform in `namma-space-gpu:0.2`, so not needed).
Full pulls now leave out `60_export/*.ply` and `55_geometry/*.ply` (hundreds of MB, not used on the Mac;
`ns pull --with-ply` or `--all` still gets them).

## 09:29 rh-vrroom launched (task 3), exact user command
    ./ns run --frames tyrone:/home/sher/namma-space/data/mushroom/room_datasets/vr_room/iphone/long_capture/images \
      --name rh-vrroom --kind public --license "CC BY 4.0" \
      --attribution "MuSHRoom dataset (Ren et al. 2024), room vr_room, iPhone long capture" \
      --title "MuSHRoom vr_room (public test room)"
Ingest 17.7 s.

## 09:30 `ns run` exit now includes the Mac steps
Before, a failed bundle, verify or report did not change the exit of `ns run`. Now: bundle or report failure gives
20 (nothing to ship), failed viewer checks give 30 (a bundle exists, needs review).

## 09:31 rh-vrroom and rh-b7 stopped while queued; vr_room to restart from scratch
Both were waiting in `nsdocker` for a GPU slot (rh-honka's SfM and `ns-w7-mip360` held the two). Each would have taken
the slot that rh-honka's training needs next. Stopped their runners by exact PID (SIGTERM; each recorded "interrupted",
exit 20). `runs/rh-vrroom` deleted on tyrone and the Mac (only ingest had run) so the rehearsal can start clean and
unattended later with the same command. rh-b7 kept (ingest done: 305 of 1,440 candidates kept, 1920x1080, 54.3 s);
it resumes with `./ns run --name rh-b7 --quick`.

## 09:37 rh-honka SfM: 1,015.6 s, B chosen, flagged partial (94.6%)
`20_sfm/sfm.json`: A 86.0% after the sanity filter (95.2% raw), 0.65 px, 18,216 points, one jump left (ratio 35 at
frames 203/204); B 94.6% (100% raw; 18 cameras dropped by the filter: 75-85 and 246-251), 0.74 px, 47,741 points,
no jumps, 1,012 s; D 96.4%, 0.52 px, 17,528 points, but jumps left (ratio 36 at 204/205, 16 at 251/252), 184 s.
D fails the sanity check, so B wins with the plan's rule 3 (none of the sane variants reached 95%): `partial: true`.
The runner did not turn `partial` into exit 11 (CONTRACTS 2.11 says partial = 11; W1 asked root to do it).
Fix: `ns_remote.py` maps a stage JSON with `"partial": true` and exit 0 to 11 with the stage's `why` in the note.
Test `test_partial_sfm_json_gives_11_and_the_run_continues`. rh-honka's runner started before this fix, so its exit
will not show the 11.

## 09:38 rh-honka align: 5.7 s, gate failed on the camera cue; real up error 2.0 deg
`40_align/align.json`: cue spread 6.2 deg (cameras 4.16 deg from the fused up, planes 2.08), walls not used;
floor from where the walls end (`wall_extent`, no horizontal plane qualified), ceiling 2.00 m above it;
H_cap 1.40 m (generic prior, no height given), scale 0.5257 m per SfM unit, 68% interval +-12.7%.
Model unchanged by the transform: 317 images, 47,741 points, 0.7432 px before and after.
New tool `python -m pipeline.align.metric_check --run <run> --ref <transformations.json>` (trimmed Umeyama of our
aligned camera centres onto the MuSHRoom ARKit/Polycam centres, matched by original file name). On rh-honka:
316 cameras matched, 221 in the fit; **scale error +3.3%** (plain Umeyama +4.1%); **up error 2.0 deg** against
gravity (+Z); median residual 1.2 cm, p90 4.4 cm. Output `/tmp/nsint/honka_metric.json` (copied into
test/expected/report.json at the end).
So the gate fired on a correctly levelled scene (D2 limit 3 deg: 2.0 passes). Fix in `pipeline/align/core.py`: the
5 deg rule now applies between the geometric cues (planes vs walls); the camera cue, which carries the capture's
mean pitch, is gated only when it is more than 10 deg from the fused up; otherwise a non-gating "note:" flag records
the spread. Align unit tests: 17 pass. rh-honka keeps its exit 30 from this gate (its align ran before the fix; the
transform itself does not change, only the flag).

## 09:44 `ns rescale` was not wired end to end (test 0.8 only passed with stubs)
Found: `rescale.py` wrote only `40_align/rescale.json`; the grid stage and `bundle.py` read `T_final` from
`60_export/export.json` and never looked at it, so a rescale changed nothing visible. Also `ns rescale --factor K`
passed `--factor=K`, which the real `rescale.py` rejects (its option is `--scale`), and `ns rescale --scale K` meant
"absolute" in ns but "multiply" in rescale.py.
Fix:
- `rescale.apply_to_export` (called after every change): keeps the export stage's own `T_final` as
  `T_final_export`, writes `T_final = T_user @ T_final_export` (matrix, scale, translation) and `user_scale`, and
  scales the held-out and parity camera positions by f (originals kept as `heldout_export.json`, `parity_export.json`).
  Idempotent and reversible (`--reset`).
- the grid stage records `user_scale` in grid.json, which `rescale.py --use-measures` already divides out.
- `ns rescale --factor K` -> `rescale.py --scale K` (multiply); `ns rescale --scale K` -> `--set-total K` (total
  factor relative to the automatic scale). Only one of them at a time.
Tests: `test_user_scale_reaches_export_json_and_poses` (align unit tests 7/7 rescale, 18 align); grid 49/49; e2e pass.
Not yet run on a real run; planned on rh-honka after it finishes.

## 09:48 Optional stage could stop the whole run
`pipeline/geometry/__main__.py` (W7) appeared on disk this morning, so runners now run the optional geometry stage.
A failure there (any exception = exit 1 = 20) stopped the run before export and grid. Fix: an optional stage's
non-zero exit is recorded as `exit-N-ignored` with a warning and the run continues
(`test_optional_stage_failure_does_not_stop_the_run`). rh-honka's runner started with the old code; if geometry fails
there, resume with the same `ns run --name rh-honka`.
Also: the export stage now re-applies a recorded user scale (40_align/rescale.json) after a re-export.

## 09:47 rh-vrroom relaunched (clean start, same command as 09:29)
Reason for not waiting longer: rh-honka's r0 runs at about 9-10 steps/s next to `ns-w7-mip360` (GPU at 100%), so its
two sequential 30k variants end about 11:20; starting vr_room after that would end after 13:30. vr_room's SfM queues
for the slot `ns-w7-mip360` holds. No `runs/room-*` exists yet. Ingest 7.3 s (422 frames).
rh-honka r0 at 7k: held-out PSNR 19.40, SSIM 0.743, LPIPS 0.366, cc_psnr 21.91, 1,083,723 Gaussians; the 7k PLY was
ready 415.5 s after r0 started (`50_train.tmp/r0/stats/val_step6999.json`, progress.json).

## 09:48 Remote video outside the namma-space root, and the FAIL path
`./ns check tyrone:/home/sher/ns_int_inbox/b5.mp4 --name rh-b5check`: hard-linked into the run (same inode, no extra
disk), check FAIL ("15 s is under 45 s"), exit 10, 2 s. Test files removed. Created `/home/sher/namma-space/inbox/`
on tyrone as the place to drop big phone videos (`scp IMG.MOV sher@10.1.18.190:/home/sher/namma-space/inbox/`).

## 09:55 Remote e2e tests (`NS_E2E_REMOTE=1`, stubs on tyrone)
First run: 2 failures (0.1 resume, 0.8 rescale) because the new "bundle failure = 20" rule fired on stub runs, which
have no grid images. Stub runs (export.json `"stub": true`) no longer count Mac-step failures. Rerun: 8 tests OK
(1 skipped), 104 s. Local `./ns test e2e` OK.

## 10:08 Geometry stage (W7) fails under the runner; made opt-in
`pipeline/geometry/__main__.py` reads `train.json chosen_ply`, which is relative to the run dir, as a path relative
to the cwd (`/w/repo`), so it raises FileNotFoundError at once (after waiting for a GPU slot). Not my file:
written up in `pipeline/remote/HANDOFF_GEOMETRY.md`. Runner change: optional P1 stages run only when asked
(`ns run --geometry`, runner `--with-optional`, or `--only geometry`); nothing later in the chain reads 55_geometry.
Tests updated (`test_optional_stage_runs_only_when_asked`; full-run tests pass `--with-optional`); e2e 97 OK.
rh-honka's runner (old code, started 09:20) will still try geometry after training, fail, and stop with 20; it then
needs `./ns run --name rh-honka` to resume (logged as a manual step when it happens).

## 10:16 rh-honka r0 done
`50_train.tmp/r0/stats/val_step29999.json`: held-out PSNR 21.27, SSIM 0.812, LPIPS 0.226, cc_psnr 24.85,
1,500,000 Gaussians (the MCMC cap); wall 2,320 s (38.7 min) next to vr_room's SfM and, until 09:52, W7's job.
(Smoke model of 03:35 with normalisation on: 20.62 / 0.807 / 0.242 / cc 23.96.) r1 started.

## 10:11-10:19 rh-vrroom SfM and align; the up estimate was 4.2 deg off; align fixed
SfM 1,441.5 s (about 6 min of it waiting for a slot): A 100%, 0.55 px, 34,250 points; B 100%, 0.80 px, 50,768
points; D 99.5%, 0.45 px, 30,409 points, chosen (lowest error at >= 95%), not partial.
Align 5.2 s: gate failed "up cues disagree by 7.3 deg (planes vs walls)". `metric_check` against the MuSHRoom poses:
**up error 4.2 deg, scale error -3.1%** (420 matched, 294 in the fit, median residual 1.1 cm). This time the gate was
right. Per-cue errors against gravity (`/tmp/nsint/cue_err.py`): walls 0.29 deg, cameras 2.82, planes 7.18 (6 of 8
"horizontal" RANSAC planes were 7-17 deg off: slices through furniture). Honka: cameras 2.67, planes 3.92.
Fix in `pipeline/align/core.py`: the plane cue takes only planes within 5 deg of the current up, re-selected
iteratively (details in pipeline/align/NOTES.md). Re-running the new estimate on both real aligned models: up error
**0.30 deg** (vr_room) and **1.36 deg** (honka), no gate flags. Align tests 19/19.
rh-vrroom keeps running unattended with its 4.2 deg tilt (training had started); it will end 30 for a real reason.
Synced at 10:19, so any align that starts from now on uses the fix.

## 10:20 rh-vrroom restarted from align with the fix (manual step, recorded)
Stopped the runner (PID 2045247, SIGTERM; train container stopped; r0 had run about 8 min) and ran
`./ns run --name rh-vrroom --from align --manual-override "..."` (new `ns run --manual-override TEXT` flag; the text is
in `runs/rh-vrroom/run.json manual_overrides`). A first try with `--runner-flag=--manual-override=...` failed in
argument parsing (the text was split on spaces) and changed nothing.
New align (5.1 s): gate passed, no flags; cues cameras 1.29, planes 0.61, walls 1.74 deg from the fused up; floor
`ransac_plane`, ceiling 2.78 m above it (MuSHRoom lists vr room as 5.1 x 4.4 x 2.8 m); H_cap 1.40 m (prior);
scale 0.4265, +-12.7%; model unchanged (420 images, 0.4482 px before and after).
`metric_check`: **up error 1.85 deg** (was 4.20), **scale error -3.1%**. The 0.30 deg of the offline check above came
from running the estimate on the already-aligned model; RANSAC samples differently in another frame, so the real run is
the number to quote.

## 10:25 Mac disk 1.4 GB free; full pulls trimmed further
The stopped rh-vrroom pull brought 104 MB, almost all COLMAP models (`20_sfm/*/clean`, `30_undist/sparse`,
`40_align/sparse`) that nothing on the Mac reads. Full pulls now leave them out (`ns pull --all` still gets them).
Deleted those copies and `runs/rh-honka30r` (21 MB, mine) on the Mac.

## 10:35 Mac-side steps get wall limits too
`ns run` waited on `bundle.py` and the Chrome harness without a limit. Now bundle 10 min, verify 20 min; past it the
process group is killed and the step counts as failed (exit 124 -> run exit 20 or 30). Checked with a 1 s limit on
`sleep 30` (returned 124 after 1.0 s).

## 11:03 rh-honka training done (5,155.6 s = 85.9 min, two variants one after the other)
`50_train/train.json`: r0 PSNR 21.27, SSIM 0.812, LPIPS 0.226, cc_psnr 24.85, 1,500,000 Gaussians, 2,277 s of training
(2,320 s wall); r1 PSNR 21.35, SSIM 0.812, LPIPS 0.225, cc_psnr 25.05, 1,500,000, 2,792 s (2,834 s wall).
Chosen r1: "tie on PSNR (+0.09 dB) and LPIPS (-0.001); r1 taken for its depth-supervised geometry".
Warning "training took 47 min > 45" (shared GPU: rh-vrroom's training ran beside it).
Geometry (old runner, so it runs): the geometry owner had already fixed the relative-path bug, so it is running
(TSDF), not failing.

## 11:07 Both runs reach the end; the Mac disk fills
- rh-honka: geometry 154.9 s (mesh 223,546 vertices; the owner's path fix was in), export 52.6 s (all F gates pass),
  grid 1.8 s, runner exit 30 (the old align gate). Its Mac process (started 09:20, old `ns` code) then hit
  **ENOSPC on the Mac**: verify could not write verify.json and report crashed; it still printed exit 30.
- rh-vrroom: train 2,748.8 s: r0 PSNR 23.97, SSIM 0.870, LPIPS 0.245, cc_psnr 26.53, 1.5 M Gaussians, 2,657 s;
  **r1 (depth loss) crashed after 46 s** with a CUDA device-side assert in MCMC `relocate` (torch.bincount on the
  sampled indices; the usual sign of NaN opacities). The ladder did its job: r0 chosen ("only r0 finished usable").
  geometry skipped (opt-in), export 53.8 s exit 30, grid 1.7 s, runner exit 30.
- Freed my own Mac copies (cleaned PLY 149 MB, COLMAP models, a test video): 117 MB -> about 600 MB free.

## 11:10 vr_room export lost 2 dB in the "huge splat" step; fixed
`60_export/export.json` `clean.psnr_after_step`: opacity 24.00 -> huge 21.99 dB (312 splats removed: the big splats on
its plain walls and ceiling). F2 gates failed (clean drop 2.00 dB, SD drop 2.25 dB). On honka the same step cost
0.07 dB. Fix in `pipeline/export/__main__.py`: step 3 takes the first axis limit of 1, 2, 4, 8 m whose cost is
<= 0.10 dB, else skips the step (as step 4 already did), recorded in `clean.details.huge_search`. Also the outside-box
search no longer re-applies step 3 to the held subset. Export unit tests pass. Re-run with
`./ns run --name rh-vrroom --from export --manual-override "..."` at 11:15.

## 11:12-11:15 rh-honka Mac steps redone by hand (`./ns bundle|verify|report rh-honka`)
bundle: 78 checks, 0 failed, 41.6 MB. verify (2 min 31 s): 25 passed, 2 failed, 6 not run. H1 first frame 0.42 s;
H2 59.8 fps mean, p95 16.8 ms, 629,993 splats; H3 0 blocked samples of 28,245; H4 browser vs held-out photo 21.1-23.1 dB;
H4b browser vs gsplat 33.2-35.3 dB; H9 first frame 2.78 s, SD loaded 9.90 s at 20 Mbps. Failed: R9 and R10, the
viewer's new Round-2 checks (search/route) on a scene without places (HANDOFF_VIEWER.md). report: 0 empty required
keys (test 0.6 passes). Plan gates: 15 pass, 2 fail (train.psnr 21.35 < 22 [E]; total 6,392 s > 6,000).

## 11:16-11:20 rh-vrroom after the huge-step fix
export 72.3 s, exit 0, all F gates pass: step 3 chose 8 m (6 splats; 1 m would cost 2.01 dB, 2 m 1.16, 4 m 0.106);
clean PSNR 23.97 (no loss); SD 646,048 splats, SH3, 15.7 MB, held-out PSNR of the decoded SD file 23.55 (drop 0.43 dB).
Remote exit 0. Mac: bundle 78/78; verify 25 passed, 2 failed (R9, R10, as on honka) -> `ns run` exit 30.
H1 0.44 s; H2 60 fps, p95 16.7 ms; H4 22.5-24.9 dB; H4b 32.8-34.0 dB; H9 SD 9.90 s at 20 Mbps.
Grid: walkable 11.14 m2, path coverage 1.0, room 4.90 x 4.03 m, warning "measured floor is +13.4 cm from y = 0".

## 11:22 Export floor offset was wrong on both rooms; fixed
The export's floor rule (lowest footprint-weighted height peak) found y = -0.317 m on honka (rejected only by its
0.30 m limit) and -0.123 m on vr_room (applied: the scene moved up 12 cm, and the grid then measured the real floor at
+13.4 cm). The align floor and the grid's own measure agree within 1-2 cm on both. The lowest peak comes from splats
under the floor. New `clean.measure_floor_local`: median height of opaque splats within 0.20 m of the align floor and
1.5 m of the path (same idea as the grid's check); applied up to 0.15 m; the old rule is kept in export.json for the
record and warns when they differ by > 0.10 m. Test `test_floor_local_ignores_a_peak_under_the_floor`. Export tests
pass. rh-vrroom re-run `--from export` at 11:24 (second manual step, recorded in run.json).

## 11:23-11:26 rh-vrroom final (after the floor fix)
export (155 s while the smoke run trained), exit 0, all F gates pass; T_final floor shift -0.027 m (local rule; the old
rule's -0.123 m recorded and flagged "not used"); grid: walkable 12.35 m2, path coverage 1.0, room 4.90 x 4.04 m,
floor offset +0.004 m, no warnings. Remote exit 0; `ns run` exit 30 only because of viewer R9/R10.
Report: 0 empty required keys; plan gates 17 pass, 0 fail (full mode).

## 11:15-11:25 Smoke test from test/README.md, exactly as written
`./ns run --frames test/data/honka60/frames --name smoke --quick --kind public --license "CC BY 4.0" --attribution ...`
Wall time 553 s (9.2 min) from start to the report, while rh-vrroom's export shared the GPU. Stages: ingest 1.8 s,
sfm 123.6 s (exit 11: B 81.7% at 0.70 px; A 75%, C 76.7%, D 41.7%; missing 000001-000004, 000035-000037,
000057-000060), align 2.8 s (no flags), train 254 s (r0 16.11 dB, r1 16.73 dB, SSIM 0.711, LPIPS 0.436, 155,596
Gaussians; exit 30: PSNR < 20), export 14.6 s (SD 78,249 splats, 1.81 MB), grid 1.4 s (exit 30: path coverage 0.86
in grid mode, 0.84 corridor, so free fly inside the SfM bounds; no wall columns found). Mac: bundle 76/76; verify 23
passed, 4 failed (H3 wall stops 4/16 and H4 15.8-21.8 dB on this weak model; R9, R10). Exit 30.
So the 10-minute claim holds, but the sample gives a weak model. Trying 60 consecutive frames (160-219) as rh-smoke2.

## 11:28 `ns rescale` on a real run (smoke, --factor 1.1)
rescale 141 s (mostly waiting for a container slot), grid re-run. Results after pull and bundle: export.json T_final
scale 1.1 (`T_final_export` keeps 1.0), grid walkable 160.93 -> 196.25 m2, eye height 1.40 -> 1.54 m, spawn x1.1,
scene.json `splats.transform.scale` 1.1, held-out positions x1.1, rescale.json scale_final 0.917, H_cap 1.54 m. Bug
found: `ns rescale` treated the grid's exit 30 (fallback) as failure and skipped the pull and bundle; now 0/11/30
continue. Reset with `./ns rescale smoke --scale 1.0`.

## 11:27-11:41 rh-smoke2: 60 consecutive honka frames (160-219), from tyrone, `--quick`
sfm 215.9 s: D 100%, 0.46 px, 5,319 points (A 65%, B 70%); align 2.5 s; train 487.7 s (r1 chosen: PSNR 20.21,
SSIM 0.786, LPIPS 0.376, cc_psnr 22.07, 120,588 Gaussians; r0 19.76); export 15.0 s, exit 30 (F2: the opacity step
costs 0.38 dB on a 7k model; floor not measured, too few opaque floor splats); grid 1.2 s, mode grid, walkable
5.99 m2, path coverage 1.0, no room extent (no wall columns). Mac: verify 24 passed, 3 failed (H6 governor did not
step in 9 s at pixel ratio 2, with rh-b7's SfM and a second Chrome-free Mac load; R9, R10). Plan gates (quick):
15 pass, 0 fail. Stage total 724 s. The sample in test/data/honka60 is now these frames (11 MB).
`./ns rescale smoke --scale 1.0` reset the earlier test (147 s, exit 30 from the grid fallback and R9/R10).
11:42: deleted runs/smoke (tyrone and Mac) and ran the README smoke command again on the new sample.

## 11:40 rh-b7 (video path, 4K HEVC fixture) SfM: variant B aborts on 1920 px frames; fixed
`./ns run --name rh-b7 --quick` (resume; ingest had run at 09:24). SfM log: B match exit -6, retried, failed again:
`ONNX Runtime error ... Failed to allocate memory for requested buffer of size 6894977536` in the SIFT LightGlue
matcher. The databases show why: COLMAP 4.2.1 kept far more SIFT keypoints than `max_num_features 8192` (honka 738x994:
mean 2,846, max 14,925 per image; b7 1920x1080: mean 7,959, max 24,523), and LightGlue's attention grows with the
square. A user's 1920 px frames would hit the same. D registered 36.1% of b7 (7 models; the fixture is a crop pan of
still photos, not real camera motion). Fix in `pipeline/sfm/cmds.py`: B extracts at most at 1280 px
(`sift_max_image_size`; COLMAP scales keypoints back to full size); honka (994 px) unchanged. SfM unit tests 50/50.
Synced 11:44.

## 11:42-12:05 The README smoke command on the new sample (`--name smoke`)
Wall 1,418 s (23.6 min): ingest 1.8 s, sfm 128.9 s (D 100%, 0.46 px), align 2.6 s, train 615.2 s (r1 19.97 dB,
SSIM 0.776, LPIPS 0.375, cc 22.00, 120,384 Gaussians; exit 30 as r1 < 20 dB this time), export 510.2 s of which
**502 s waiting for GPU memory** (below), grid 1.2 s (mode grid, walkable 6.81 m2, coverage 1.0), Mac verify 23 passed,
4 failed (H6 under load; R6, R9, R10 viewer-side). Exit 30. Plan gates (quick) 15/15 pass.
Frozen as case `smoke` in test/expected/report.json; rh-smoke2 (same frames) is within every band.

## 11:58 SfM variant B also hogs the GPU
`nvidia-smi`: the rh-b7 B matcher (`colmap sequential_matcher ... SIFT_LIGHTGLUE`) held 33.5 GB while retrying, so
`nsdocker` (needs 12 GB free) held back the smoke export for 502 s. At 1280 px b7 still had up to 19,358 keypoints and
failed on a 6.0 GB buffer. Cap lowered to 1024 px (`sift_max_image_size`), the regime where honka's B worked (994 px).
Not yet proven on a 1920 px clip. rh-b7 stopped (12:00; its SfM was not informative: the fixture is a crop pan of
still photos, D registered 28-36%).

## 12:08 A phone-like clip from real frames; the user's video path end to end
New `pipeline/ingest/make_phone_fixture.py`: the 335 honka photos in order, each held 0.5 s, 30 fps, HEVC Main10
(hvc1), BT.2020 + HLG, stored landscape with a "rotate 90" display matrix, as an iPhone stores a portrait HDR clip.
Built on tyrone in 137 s: `/home/sher/namma-space/inbox/honka_phone_hlg.mov`, 34.7 MB, 167 s.
`./ns run tyrone:/home/sher/namma-space/inbox/honka_phone_hlg.mov --name rh-phone --quick --kind fixture ...` (12:10):
check on tyrone 24 s: WARN (short side 738 px; HLG), shown 738x994 (rotation read); ingest 29.6 s: 1,340 candidates,
303 kept (32 blurry windows dropped), 738x994 upright, `hdr: hlg`, `tonemap: zscale`, `rotation: 90`.

## 12:25-12:42 Variant B on 1920 px frames after the cap (Mip-NeRF 360 room, `rh-mip1920`, `--to sfm`)
`./ns run --frames tyrone:/home/sher/namma-space/data/mipnerf360/room/images --name rh-mip1920 --to sfm ...`:
ingest 66 s, 311 frames at 1920x1279. B extract 26 s; **B match ok in 915 s** (shared GPU); B keypoints per image
mean 2,948, median 3,200, max 5,096 (A, ALIKED: max 4,096). Each matcher process held about 3.5 GB of GPU memory
(nvidia-smi), not 33 GB. So the 1024 px cap works on 1920 px input.
rh-phone SfM meanwhile: A 56.1%, B 87.1% (0.77 px, 32,326 points, 2 models), best 87.8% < 90% with 303 frames, so
variant C (exhaustive ALIKED matching) started at 12:24.
- 12:44 deleted tyrone runs rh-b7 (883 MB, stopped, uninformative), rh-b2, rh-b7check, rh-b2check (their JSON evidence is on the Mac under runs/).
- rh-mip1920 SfM finished (1,861 s on the shared GPU): A 95.8% 0.75 px 55,393 points; **B 95.8% 0.78 px 47,232
  points (ran to the end at 1920 px input)**; D 95.8% 0.60 px 41,240 points, chosen. Not partial.

## 12:51-13:01 rh-phone (phone-like HEVC 10-bit HLG portrait clip from tyrone) end to end
sfm 2,370 s: A 56.1%, B 87.1%, D 87.8%, so variant C (exhaustive) ran (about 27 min) and won with 88.8%, 0.78 px,
14,413 points; partial -> exit 11, with the reshoot ranges in seconds (0.1-5.1 s, 35.0-42.5 s, 123.0-126.3 s).
align 3.9 s, gate passed, no flags; train 420.6 s (quick; r1 20.21 dB, SSIM 0.794, LPIPS 0.382, cc 22.85,
327,038 Gaussians); export 31.3 s, all F gates pass (the old floor rule would have said -0.237 m; the local rule
0.024 m); grid 1.5 s, mode grid, walkable 17.47 m2, coverage 1.0, no room extent. Remote exit 11.
Mac: bundle 78/78 (13.4 MB); verify 22 passed, 5 failed (H4 17.1-22.8 dB and H4b 19.9 dB on one pose for this
7k-step model; H6 under load; R9, R10). Exit 30. Report: 1 empty required key (grid.room_extent_m: no wall columns).
So a phone-style HEVC Main10 HLG portrait clip on tyrone goes through check, ingest (tone map, rotation), SfM, align,
train, export, grid, pull, bundle, verify and report with one command.
Note: on captures under 90% with <= 350 frames, variant C adds about 25 min to SfM.

## 13:05 Summary for root
Code changed (all in my ownership): `ns`; `pipeline/remote/ns_remote.py`; `pipeline/train/__main__.py`;
`pipeline/align/{core.py, rescale.py, metric_check.py (new)}`; `pipeline/export/{__main__.py, clean.py}`;
`pipeline/grid/stage.py` (user_scale); `pipeline/sfm/cmds.py`; `pipeline/ingest/make_phone_fixture.py` (new);
tests in `test/e2e/` and `test/unit/{train,align,export}/`; `test/README.md`, `test/expected/{README.md,report.json}`,
`test/data/honka60/`. Notes appended in the train, align, export, sfm, ingest and remote NOTES.md.
Handoffs: `pipeline/remote/HANDOFF_VIEWER.md` (R9/R10 make every run end 30), `pipeline/remote/HANDOFF_GEOMETRY.md`.
Tests at 13:00: e2e 97 OK (12 skipped), remote e2e 8 OK, train 25, export OK, align 19, sfm 50, grid 49, check_mac OK.
Runs kept on tyrone (all under 6 GB): rh-honka 1.8 GB, rh-vrroom 1.8 GB, smoke, rh-smoke2, rh-phone, rh-mip1920.
Manual steps: rh-vrroom 3 (in its run.json manual_overrides); rh-b7 1 (deleted); rh-honka Mac steps redone after
ENOSPC (not in run.json: the runner had finished; recorded here).
