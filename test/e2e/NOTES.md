# test/e2e: notes (root, WP0)

One command: `./ns test e2e` (88 tests, about 24 s on the Mac). Remote parts: `NS_E2E_REMOTE=1 ./ns test e2e`.

| File | Covers |
|---|---|
| test_colmap_io.py | colmap_io on a synthetic model and on `smoke_honka/clean` (honka tests skip off tyrone) |
| test_conventions.py, conventions.test.mjs, test_conventions_js.py | conventions.py and conventions.json; the Node test runs the real three.js r180 |
| test_stage.py | atomic stage dirs, timing, disk guard (0.3 at stage level) |
| test_runner.py | runner: order, skip, `--from`, kill and resume (0.1), low disk (0.3), two runs (0.4), rescale (0.8), exit codes, missing modules |
| test_report.py | report.json keys, nulls with reasons, html escaping |
| test_expected.py, check_expected.py | gates in `test/expected/gates.json` |
| test_align_thin.py | the thin align adapter on a synthetic room (also checks colmap_io and conventions together) |
| test_ns_cli.py | ns without ssh: name checks, serve (CORS, MIME), package, publish gate (typed phrase through a pty), check |
| test_remote.py | needs ssh: 0.1 through `ns run --from train`, 0.2 ssh drop, 0.4, 0.5 doctor, 0.8 rescale, dead runner |

Stubs (`stubs/`) write JSON marked `"stub": true`; behaviour per run comes from `runs/<name>/stub_cfg.json`
(`{"train": {"sleep": 30}, "sfm": {"exit": 10}}`). Remote tests run them with `--no-docker`, so they never wait for
container or GPU slots. `NS_E2E_DOCKER=1` adds a run through real containers.

Notes:
- The unittest module name `test` clashes with the standard library, so tests run by path (`python3 test/e2e/test_x.py`)
  or `unittest discover -s test/e2e`, never as `test.e2e.x`.
- The Mac has under 1 GB free, so the tests lower the runner's disk floor (`--min-free-gb`) and never write more than a few KB.
- `gen_conventions.py` regenerates `test/data/conventions.json`. Only root runs it, after a convention change.
