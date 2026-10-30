# WP3 train: notes (W3, 30 Oct 2026)

Stage: `python -m pipeline.train --run /w/runs/<name> [--quick] [--variants r0,r1] [--parallel auto|yes|force|no]`
Image `namma-space-gpu:0.1`, started through `nsdocker --gpu`. Writes `50_train/` via `50_train.tmp/` (renamed when done).

## What it does
- Wraps `/opt/gsplat/examples/simple_trainer.py mcmc`. Flag spellings confirmed with `-h` (tyro uses hyphens):
  `--no-normalize-world-space`, `--antialiased`, `--use-bilateral-grid`, `--depth-loss`, `--strategy.cap-max`,
  `--disable-viewer`, `--disable-video`, `--save-ply`, `--ply-steps`. Defaults left alone otherwise
  (no `--pose-opt`, no `--app-opt`). `--max-steps 7000` for `--quick`.
- Variants: `r0`; `r1` = r0 + `--depth-loss`; `r0nb` = r0 without the bilateral grid (NaN fallback, run only if r0 and r1
  both fail or produce NaN splats).
- Parallel R0 + R1: `auto` needs a second GPU slot (we try `flock -n` on `/w/locks/gpu.0` and `gpu.1`; the one our own
  container holds refuses us) and >= 24 GB GPU free. `force` skips the slot lock and needs only the 24 GB (use it
  when the container was started with enough of the GPU). Otherwise R0 then R1.
- Choice: held-out PSNR, then LPIPS; PSNR differences <= 0.10 dB and LPIPS differences <= 0.005 count as ties;
  a full tie takes R1. `train.json.why` says which rule fired.
- `train.json`: `variants[{name, psnr, ssim, lpips, cc_psnr, n_gs, time_s, peak_gpu_gb, at_7k{...}, status, wall_s,
  t_preview_s, n_gs_ply, nan_dropped, final_ply, cmd, gpu_used_total_peak_gb}]`, `chosen`, `why`, `chosen_ply`,
  `preview_ply`, `warnings`. `time_s` is gsplat's `ellipse_time` (training incl. the 7k eval),
  `peak_gpu_gb` is `torch.cuda.max_memory_allocated` (not reserved); `gpu_used_total_peak_gb` is nvidia-smi for the
  whole GPU (includes the other tenant, so it overstates ours).
- While running, `50_train.tmp/progress.json` has each variant's step and, once it exists, `preview_ply`
  (`50_train.tmp/<v>/ply/point_cloud_6999.ply`). Time to that file is `t_preview_s` (E4).
- Losing variant: `ckpts/` and `ply/` deleted, stats and cfg kept. Winner keeps both PLYs and the final checkpoint.
- Exit codes: 0 good, 20 infrastructure (missing/text model, disk < 15 GB, nothing usable), 30 quality gate
  (best PSNR < 20 dB, or only a 7k model exists).
- gsplat's PLY exporter silently drops NaN/Inf splats, so NaN shows up as `nan_dropped = num_GS - ply count`.

## Finding 1: MCMC is not scale-invariant, so world normalisation off needs a rescale (verify below)
gsplat's `inject_noise_to_position` moves means by `covariance x lr x noise_lr x randn`. `lr` is proportional to
`scene_scale` (a length) and the covariance is a length squared, so the noise grows as length^3. `scale_reg` has units of
length too. World normalisation hides this: it scales every scene so the median camera distance is 1.
Measured on the GPU with gsplat's own function (`python -m pipeline.train.noise_probe`): relative to scene size, the
displacement of a nearly transparent splat grows 7.84x at k = 2.8 and 68.4x at k = 8.27, and is exactly 1.00x again with
`noise_lr / k^2`. (The noise only moves splats with opacity near 0.005, so its damage shows in how well the
sparse-opacity splats explore, not as a direct blur.)
With normalisation off, `pipeline.train` computes k = median camera distance from the camera centroid (model units) and
passes `--strategy.noise-lr 5e5/k^2` and `--scale-reg 0.01/k`. `--no-scene-unit` switches this off.
See the E2 section for the measured effect.

## Finding 2: `colmap model_transformer` 4.2.1 broke a model (not used any more)
With a 4x4 `[sR | t]` matrix (s = 2.5) it wrote `images.bin` poses with one similarity (s 0.835) and `points3D.bin` with
another (not even a similarity of the originals). `model_analyzer` still reported the original 0.5926 px, because it
reads the rig/frame poses. gsplat reads `images.bin` + `points3D.bin`, so training started from scattered points
(7k held-out PSNR 12.0 and 16.2 on honka, two runs; both stopped). We now apply the similarity ourselves
(`pipeline/export/colmap_io.write_transformed_model`, tested: projections unchanged, reprojection 0.6182 px before
and after). W2's align stage should check reprojection from `images.bin` + `points3D.bin` after its transform.

## Tests
`pipeline/export/.venv/bin/python -m unittest discover -s test/unit/train` (Mac; the venv holds numpy/scipy) or
`python3 -m unittest discover -s test/unit/train` (stdlib only, the numpy tests skip). 23 tests: command flags,
choice rules, stats parsing, a full stage run against a fake trainer, second-slot lock.

## Commands run (tyrone, 30 Oct 2026; `nsdocker` = `/home/sher/namma-space/bin/nsdocker`)
- Help and sources: `nsdocker w3-help namma-space-gpu:0.1 bash -c "cd /opt/gsplat/examples && python simple_trainer.py mcmc -h"`;
  `docker run --rm --entrypoint cat namma-space-gpu:0.1 /opt/gsplat/examples/simple_trainer.py` (also `datasets/colmap.py`,
  `gsplat/exporter.py`, `gsplat/strategy/ops.py`) to read them.
- Frame-test model: `python -m pipeline.train.frame_test --model /w/runs/smoke_honka/clean --out /w/runs/w3-honka/40_align
  --images /w/runs/smoke_honka/undist/images` (s 2.5, yaw 37 deg, camera centroid at (0, 1.4, 0)).
- Normalisation-on baseline: `nsdocker --gpu w3-trainB namma-space-gpu:0.1 python -B -m pipeline.train --run /w/runs/w3-honka-norm
  --data-dir /w/runs/smoke_honka/clean --normalize --variants r0`.
- Quick R0 + R1 together on the aligned copy (inside my own running container, `docker exec`, 7k steps):
  `python -B -m pipeline.train --run /w/runs/w3-honka-quick --data-dir /w/runs/w3-honka/40_align --quick --parallel force`.
- Ablation: same with `--variants r0 --parallel no --no-scene-unit` into `/w/runs/w3-honka-abl`.

## Integrator fix (30 Oct, 09:15): the wall-limit loop
`ns-w3-trainB` (`runs/w3-honka-norm`) printed "wall limit 3600 s reached; terminating ['r0']" every 2 s from 04:55
to 09:12 with the GPU idle: the trainer's log had stopped at step 19,746 at 04:24 and the process never exited after
SIGTERM, so `run_jobs` never left its loop. Now:
- `--max-minutes` is per variant, from that variant's start (default 90 full, 30 quick);
- `--stall-minutes` (default 15): no new trainer output for that long counts as hung;
- SIGTERM to the process group, SIGKILL 30 s later, and 60 s after that the job is abandoned so the stage finishes;
- the record says `status: timeout | stalled`, `end_reason`, `abandoned`, and `partial_step` when it has a 7k
  evaluation; such a partial model is shipped as the last rung of the ladder (exit 30). r0nb runs only after crashes
  or NaN, not after a timeout.
Tests: `test_wall_limit_kills_a_trainer_that_ignores_sigterm`, `test_stall_watchdog_stops_a_silent_trainer`.
