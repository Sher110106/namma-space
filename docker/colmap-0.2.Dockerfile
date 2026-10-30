# namma-colmap:0.2 = namma-colmap:0.1 + python3 (distro stdlib only; no pip packages).
# 0.1 ships no Python, but pipeline.sfm is a stdlib Python 3 driver around the colmap CLI.
# COLMAP 4.2.1, CUDA 12.9.1 and cuDNN 9 are unchanged.
FROM namma-colmap:0.1
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 --version \
 && python3 -c "import sqlite3, json, struct, statistics, subprocess, threading; print('stdlib ok', sqlite3.sqlite_version)"
