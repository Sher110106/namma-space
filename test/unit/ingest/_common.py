"""Shared helpers for the ingest tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[3]
CANDIDATES = [os.environ.get("NS_FIXTURES", ""), "/w/data/fixtures/videos",
              "/home/sher/namma-space/data/fixtures/videos", "/tmp/namma-w2/videos"]


def fixtures_dir() -> Optional[Path]:
    for c in CANDIDATES:
        if c and Path(c, "b1.mp4").exists():
            return Path(c)
    return None


def video(name: str) -> Optional[Path]:
    d = fixtures_dir()
    p = d / f"{name}.mp4" if d else None
    return p if p and p.exists() else None


def sidecar(name: str) -> Dict:
    p = video(name)
    return json.loads(p.with_suffix(".json").read_text()) if p else {}


def run_ingest(name: str, extra: Optional[List[str]] = None, cache: Optional[Dict] = None):
    """Run `python -m pipeline.ingest` on fixture `name`; returns (returncode, run_dir, seconds, stdout).
    Results are cached per (name, extra) for the life of the test process."""
    key = (name, tuple(extra or []))
    if cache is not None and key in cache:
        return cache[key]
    v = video(name)
    run = tempfile.mkdtemp(prefix=f"ns_test_{name}_")
    cmd = [sys.executable, "-m", "pipeline.ingest", "--run", run, "--video", str(v), "--min-free-gb", "0.2"] + (extra or [])
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    out = (r.returncode, run, time.time() - t0, r.stdout + r.stderr)
    if cache is not None:
        cache[key] = out
    return out


def frames_json(run: str) -> Dict:
    return json.loads(Path(run, "10_frames", "frames.json").read_text())
