#!/usr/bin/env python3
"""Cut a random N-splat subset (fixed seed) out of a 3DGS PLY. Runs in namma-space-gpu (numpy, plyfile).

usage: cut_ply.py in.ply out.ply N [seed]
"""
import sys

import numpy as np
from plyfile import PlyData, PlyElement


def main() -> int:
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    ply = PlyData.read(src)
    v = ply["vertex"].data
    if n > len(v):
        raise SystemExit(f"only {len(v)} splats in {src}")
    idx = np.sort(np.random.default_rng(seed).choice(len(v), size=n, replace=False))
    PlyData([PlyElement.describe(v[idx], "vertex")], text=False).write(dst)
    print(f"cut {n} of {len(v)} splats -> {dst}; fields: {len(v.dtype.names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
