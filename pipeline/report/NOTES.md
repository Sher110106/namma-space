# pipeline/report: notes (root, WP0)

`python -m pipeline.report --run runs/<name> [--strict]` writes `report.json` and a plain `report.html` (no scripts, no
external files) from whatever stage JSON exists. `./ns report <name>` calls it; `./ns run` calls it last.

- Keys follow CONTRACTS 2.13. A key with no source is an explicit `null`, and the reason sits in the top-level `missing`
  map (`"sfm.chosen": "20_sfm/sfm.json not found"`). A reason that starts with `n/a:` means the key does not apply (frames
  input has no video check; no tape measure means nothing to check against; HD and preview are P1).
- `empty_required` lists required keys that are null for a real reason. `--strict` exits 1 when it is not empty. Test 0.6
  is this list being empty after a full run.
- Stage JSON with `"stub": true` (test stubs) puts the stage in `stub_stages` and the HTML shows a banner, so placeholder
  numbers cannot pass for results.
- Source files and the keys read from them: frames.json (all keys; `kept` is summarised), sfm.json (`chosen`, `variants`,
  `why`, `missing_ranges`), align.json (`up_resid_deg`, `h_cap_m` or `h_cap`, `scale{value, pct_68, cues, checked_against}`,
  `floor`), train.json (`chosen`, `variants`, `why`), export.json (`sd`, `hd`, `preview`, `removed`, `T_final`),
  grid.json (`walkable_m2`, `path_coverage`, `room_extent_m`, `spawn_ok` or `spawn.walkable`), 90_verify/verify.json
  (`viewer.devices`, `walls`), timings.json, run.json (`git`, `images`, `exit`, `warnings`, `manual_overrides`).
- Gates against plan thresholds: `test/expected/gates.json` and `python3 test/e2e/check_expected.py runs/<name>/report.json`.
  The gates file holds thresholds only, never results.
- Not built: figures beyond `40_align/figures/level_before_after.png`; no held-out image pairs in the HTML.
