#!/bin/bash
# Unit tests for pipeline/sfm: stdlib only, no GPU, no network (a fake colmap stands in for the real one). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -W error::ResourceWarning -m unittest discover -s test/unit/sfm -t test/unit/sfm "$@"
