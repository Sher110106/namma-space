"""Web formats: SPZ through splat-transform (preferred) and our own .splat writer (fallback).

splat-transform 3.10.3 (MIT) flags used, confirmed with --help on 30 Oct 2026:
  -H/--filter-harmonics <0|1|2|3>   drop SH bands above n
  --spz-version <3|4>               SPZ format version (default 4)
  -w/--overwrite, --no-tty, -q
SPZ -> PLY decode: `splat-transform in.spz out.ply`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import List, Optional, Tuple

DEFAULT_WRAPPER = "/w/runs/w3-tools/splat-transform"      # private Node 22 + splat-transform on tyrone


def find_splat_transform() -> Optional[List[str]]:
    env = os.environ.get("SPLAT_TRANSFORM")
    if env:
        return env.split()
    p = shutil.which("splat-transform")
    if p:
        return [p]
    if os.path.isfile(DEFAULT_WRAPPER) and os.access(DEFAULT_WRAPPER, os.X_OK):
        return [DEFAULT_WRAPPER]
    return None


def spz_command(tool: List[str], src: str, dst: str, sh_degree: int, version: int = 4) -> List[str]:
    return tool + ["--no-tty", "-w", src, "-H", str(sh_degree), "--spz-version", str(version), dst]


def decode_command(tool: List[str], src: str, dst: str) -> List[str]:
    return tool + ["--no-tty", "-w", src, dst]


def run(cmd: List[str], timeout: int = 900) -> Tuple[int, float, str]:
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, time.time() - t, (p.stdout + p.stderr)[-2000:]


def tool_version(tool: List[str]) -> Optional[str]:
    try:
        p = subprocess.run(tool + ["--version"], capture_output=True, text=True, timeout=60)
        return (p.stdout + p.stderr).strip().splitlines()[0] if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
