#!/bin/bash
# One command for all grid tests. Uses pipeline/grid/.venv when it exists (Mac), else python3 (tyrone container).
#   test/unit/grid/run.sh            all tests
#   W4_SKIP_PERF=1 test/unit/grid/run.sh   skip G4 (writes a 350 MB PLY)
# On tyrone:
#   /home/sher/namma-space/bin/nsdocker ns-w4-test namma-space-gpu:0.1 test/unit/grid/run.sh
# G1/G2 need the honka stand-in run (W4_HONKA_RUN, default /w/runs/w4-honka_standin); they are skipped without it.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/pipeline/grid/.venv/bin/python" ]; then PY="$ROOT/pipeline/grid/.venv/bin/python"; else PY=python3; fi
fi
cd "$ROOT"
exec "$PY" -m unittest discover -s test/unit/grid -v "$@"
