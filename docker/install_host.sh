#!/bin/bash
# Set up a GPU host for Namma Space: folders, GPU slot locks and the nsdocker wrapper.
# Usage (on the host, from a copy of the repo): docker/install_host.sh [ROOT]
#   ROOT defaults to $NS_ROOT, else $HOME/namma-space. Needs Linux, Docker, the NVIDIA container toolkit, flock.
# Then build the images (docker/build.sh, see docker/IMAGES.md for the cache files and their hashes) and,
# on the Mac, point ns at the host: NS_HOST=user@host NS_REMOTE_ROOT=ROOT ./ns doctor
set -eu
ROOT=${1:-${NS_ROOT:-$HOME/namma-space}}
HERE=$(cd "$(dirname "$0")" && pwd)
for t in docker flock nvidia-smi rsync python3; do
  command -v "$t" >/dev/null || { echo "install_host: $t not found" >&2; exit 20; }
done
mkdir -p "$ROOT"/bin "$ROOT"/runs "$ROOT"/locks "$ROOT"/cache "$ROOT"/data "$ROOT"/inbox "$ROOT"/logs "$ROOT"/repo
touch "$ROOT"/locks/gpu.0 "$ROOT"/locks/gpu.1
install -m 755 "$HERE/nsdocker" "$ROOT/bin/nsdocker"
echo "installed $ROOT/bin/nsdocker; folders under $ROOT"
echo "next: docker/build.sh (images), then on the Mac: NS_HOST=$(id -un)@$(hostname) NS_REMOTE_ROOT=$ROOT ./ns doctor"
