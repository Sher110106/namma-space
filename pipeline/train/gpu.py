"""GPU probes and the second-slot lock (FINAL_PLAN 2.5, 2.15).

The stage container already holds one GPU slot (nsdocker flock on locks/gpu.N).
R1 runs in parallel only if we can take the other slot and the GPU has >= 24 GB free.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
from typing import Optional

LOCK_DIR = "/w/locks"
PARALLEL_MIN_FREE_MB = 24000


def _smi(query: str) -> Optional[list]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_free_mb() -> Optional[int]:
    rows = _smi("memory.free")
    return int(float(rows[0][0])) if rows else None


def gpu_used_mb() -> Optional[int]:
    rows = _smi("memory.used")
    return int(float(rows[0][0])) if rows else None


class SlotLock:
    """Non-blocking flock on one GPU slot file; held until release()."""

    def __init__(self, path: str):
        self.path = path
        self.fd: Optional[int] = None

    def try_acquire(self) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def try_extra_slot(lock_dir: str = LOCK_DIR) -> Optional[SlotLock]:
    """Return a held lock on a free GPU slot, or None. The slot of our own container
    (held by nsdocker) refuses us, so success means a second slot."""
    if not os.path.isdir(lock_dir):
        return None
    for s in (0, 1):
        lk = SlotLock(os.path.join(lock_dir, f"gpu.{s}"))
        if lk.try_acquire():
            return lk
    return None


def decide_parallel(mode: str, n_variants: int, lock_dir: str = LOCK_DIR):
    """mode: auto | yes | force | no. Returns (parallel: bool, lock or None, reason).
    auto/yes need a second GPU slot lock (auto also needs >= 24 GB free); force needs only >= 24 GB free
    (the caller already holds enough of the GPU, e.g. a container started with its own slot)."""
    if n_variants < 2 or mode == "no":
        return False, None, "sequential: " + ("one variant" if n_variants < 2 else "--parallel no")
    free = gpu_free_mb()
    if free is None:
        return False, None, "sequential: nvidia-smi unavailable"
    if mode in ("auto", "force") and free < PARALLEL_MIN_FREE_MB:
        return False, None, f"sequential: {free} MB GPU free < {PARALLEL_MIN_FREE_MB}"
    if mode == "force":
        return True, None, f"parallel (forced, no second slot taken): {free} MB GPU free"
    lk = try_extra_slot(lock_dir)
    if lk is None:
        return False, None, "sequential: no second GPU slot free"
    return True, lk, f"parallel: second slot {os.path.basename(lk.path)} taken, {free} MB GPU free"
