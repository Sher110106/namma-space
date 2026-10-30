#!/usr/bin/env python3
"""Kill every colmap process that works on one variant's database, repeatedly, for a while (C3: a variant dies).

usage: kill_variant.py <variant> <start_after_s> <for_s>
Scans /proc for processes whose command line names `/<variant>/db.db` and kills them with SIGKILL by exact PID.
Only meaningful inside the container that runs the driver (same PID namespace).
"""
import os
import signal
import sys
import time


def pids_for(variant: str) -> list[int]:
    needle = f"/{variant}/db.db"
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            try:
                cmd = open(f"/proc/{d}/cmdline", "rb").read().decode(errors="replace").replace("\0", " ")
            except OSError:
                continue
            if needle in cmd and "colmap" in cmd:
                out.append(int(d))
    return out


def main() -> int:
    variant, start, span = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    time.sleep(start)
    killed = 0
    t_end = time.time() + span
    while time.time() < t_end:
        for pid in pids_for(variant):
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
                print(f"killed pid {pid} (variant {variant})", flush=True)
            except ProcessLookupError:
                pass
        time.sleep(0.5)
    print(f"killer done: {killed} kills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
