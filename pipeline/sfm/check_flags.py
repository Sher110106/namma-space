"""C4: check every flag the SfM stage passes against `colmap <cmd> -h`.

usage: python3 -m pipeline.sfm.check_flags      (needs the colmap binary; run inside namma-colmap:0.2)
Exit 0 when every flag is known to the installed colmap, 1 otherwise. Also asserts the licence guard rails:
no SiftGPU, no SuperPoint/SuperGlue in any command, variant B extracts SIFT on the CPU.
"""
from __future__ import annotations

import re
import subprocess
import sys

from . import cmds


def known_flags(sub: str, colmap: str = cmds.COLMAP) -> set[str]:
    out = subprocess.run([colmap, sub, "-h"], capture_output=True, text=True, timeout=60)
    return set(re.findall(r"--([A-Za-z0-9_.]+)", out.stdout + out.stderr))


def check(commands: list[list[str]]) -> list[str]:
    problems: list[str] = []
    cache: dict[str, set[str]] = {}
    for argv in commands:
        sub = argv[1]
        if sub not in cache:
            cache[sub] = known_flags(sub, argv[0])
            if not cache[sub]:
                problems.append(f"{sub}: `colmap {sub} -h` printed no flags")
        for tok in argv[2:]:
            if tok.startswith("--") and tok[2:] not in cache[sub]:
                problems.append(f"{sub}: unknown flag {tok}")
    text = " ".join(" ".join(c) for c in commands).lower()
    for banned in ("siftgpu", "superpoint", "superglue"):
        if banned in text:
            problems.append(f"banned token in a command: {banned}")
    b_extract = " ".join(cmds.extract_sift_cpu("db", "img", cmds.Cfg(), 4))
    if "--FeatureExtraction.use_gpu 0" not in b_extract:
        problems.append("variant B must extract SIFT with --FeatureExtraction.use_gpu 0")
    return problems


def main() -> int:
    commands = cmds.all_example_commands()
    problems = check(commands)
    n_flags = sum(1 for c in commands for t in c[2:] if t.startswith("--"))
    if problems:
        print("C4 FAIL")
        for p in problems:
            print("  ", p)
        return 1
    print(f"C4 PASS: {n_flags} flags in {len(commands)} commands all exist in this colmap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
