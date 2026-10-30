# test/expected

Two files, two purposes.

- `gates.json`: pass/fail thresholds from FINAL_PLAN (the `ref` field names the gate). It holds no results.
- `report.json`: values measured on reference runs (one entry per case: `honka`, `vrroom`, `smoke`, `phone`), each with a
  tolerance band. Bands are judgement calls written in `test/e2e/freeze_expected.py` (for example +-0.6 dB of held-out
  PSNR, because gsplat runs are not deterministic and SfM varies), not measured spreads.

Check a run:

    python3 test/e2e/check_expected.py runs/<name>/report.json                                   # plan gates
    python3 test/e2e/check_expected.py runs/<name>/report.json --expected test/expected/report.json --case honka

`PASS`/`FAIL` (gates) and `OK`/`DIFF` (reference values) are as they read. `NA` means the report has no value for
that entry; it never fails. `SKIP` means a gate does not apply to this mode (PSNR and timing gates are for full
30k-step runs).

Refreeze a case after a deliberate change (and say why in the commit):

    python3 test/e2e/freeze_expected.py runs/<name>/report.json --case <case> --command "<the ns run command>"

Each case also keeps the command that produced it, its stage timings and notes such as the scale and up error
against the dataset's metric poses (`pipeline/align/metric_check.py`).
