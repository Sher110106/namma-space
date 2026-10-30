# test/expected

`gates.json` holds pass/fail thresholds for a run's `report.json`. Each one comes from a gate or test in FINAL_PLAN
(the `ref` field names it). It holds no measured results.

Check a run:

    python3 test/e2e/check_expected.py runs/<name>/report.json            # mode picked from the train steps
    python3 test/e2e/check_expected.py runs/<name>/report.json --mode full

`PASS` and `FAIL` are as they read. `NA` means the report has no value for that gate (it never fails). `SKIP` means the
gate does not apply to this mode (the PSNR and timing gates are for full 30k-step runs).

Measured values for the sample runs go into `eval/results/*.json` (W7) and into each run's own `report.json`.
A frozen `report.json` with tolerances will be added here once a rehearsal on `vr_room` has produced real numbers.
