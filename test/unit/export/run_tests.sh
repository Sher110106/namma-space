#!/bin/bash
# WP3 unit tests (train + export). One command: test/unit/export/run_tests.sh
# On the Mac the numpy tests need pipeline/export/.venv (python3 -m venv ...; pip install numpy scipy);
# in namma-space-gpu the system python has both. Without numpy the numpy tests skip.
set -e
cd "$(dirname "$0")/../../.."
PY=python3
[ -x pipeline/export/.venv/bin/python ] && PY=pipeline/export/.venv/bin/python
command -v python >/dev/null 2>&1 && ! [ -x pipeline/export/.venv/bin/python ] && python -c "import numpy" 2>/dev/null && PY=python
$PY -B -m unittest discover -s test/unit/train
$PY -B -m unittest discover -s test/unit/export
