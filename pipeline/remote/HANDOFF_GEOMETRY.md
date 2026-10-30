# Handoff to the geometry/eval builder (from the integrator, 30 Oct 10:10 IST)

1. **(Fixed by the geometry owner before 11:03: `run / path` when relative.)** **`pipeline/geometry/__main__.py` cannot find the PLY under the runner.** `_chosen_ply` returns
   `Path(train.json["chosen_ply"])`, but `chosen_ply` is relative to the run directory
   (`"50_train/r0/ply/point_cloud_29999.ply"`). The stage runs with cwd `/w/repo`, so `ply.exists()` is false and the
   stage raises `FileNotFoundError: Gaussian PLY not found` before any work. Suggested fix: `run / obj[key]` when the
   path is not absolute.
2. **The stage writes straight into `55_geometry/`** (`out.mkdir` at the start). CONTRACTS 2.11 asks for a temp
   directory renamed when complete; the `geometry.json` marker is written last, so a crash leaves a directory without a
   marker, which the runner treats as not done. Low risk.
3. **Runner change:** optional (P1) stages now run only when asked, because nothing later in the chain reads
   `55_geometry` and it costs a GPU slot on the user's run. To run it: `./ns run ... --geometry`, or on a finished run
   `./ns run --name <run> --runner-flag=--only=geometry`. A failure of an optional stage is recorded as
   `exit-N-ignored` and never stops the run.
