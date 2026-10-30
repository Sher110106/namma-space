#!/bin/sh
# One command for the viewer acceptance tests (H1-H9 and feature checks) on the test fixture.
#   sh viewer/test/run_all.sh            -> rebuilds viewer/dist from the fixture run, runs every test, writes viewer/test/last_run/
#   sh viewer/test/run_all.sh <run_dir>  -> bundles and tests a real run: <run_dir>/80_web, <run_dir>/90_verify/verify.json
set -e
V="$(cd "$(dirname "$0")/.." && pwd)"
node "$V/test/unit.mjs"
python3 "$V/test/test_bundle.py"
if [ -n "$1" ]; then
  RUN="$(cd "$1" && pwd)"
  python3 "$V/tools/bundle.py" --run "$RUN" ${NS_BUNDLE_ARGS:-}
  node "$V/test/run.mjs" --web "$RUN/80_web" --out "$RUN/90_verify" ${NS_TEST_ARGS:-}
else
  F="$V/test/fixtures/honka_run"
  python3 "$V/tools/bundle.py" --run "$F" --out "$V/dist" --name honka_demo --title "Honka room (test fixture)" --kind fixture \
    --source "MuSHRoom honka, iPhone long capture; TEST FIXTURE: splats from the gsplat smoke run, assumed scale, placeholder grid" \
    --license "CC BY 4.0" --attribution "MuSHRoom dataset (Zhao et al.)" --default --verify-dir "$V/test/last_run"
  PAR=""
  [ -f "$F/parity/parity.json" ] && PAR="--parity $F/parity"
  node "$V/test/run.mjs" --web "$V/dist" --out "$V/test/last_run" --pairs "$F/test_photos/train_poses.json" $PAR
fi
