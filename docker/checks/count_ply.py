#!/usr/bin/env python3
"""Print vertex count and SH band count of a PLY (reads only the header). usage: count_ply.py file.ply"""
import re
import sys


def main() -> int:
    n, rest = None, 0
    with open(sys.argv[1], "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").strip()
            m = re.match(r"element vertex (\d+)", line)
            if m:
                n = int(m.group(1))
            if line.startswith("property") and "f_rest_" in line:
                rest += 1
            if line == "end_header":
                break
    # 3 channels per coefficient: 0 -> SH0, 9 -> SH1, 24 -> SH2, 45 -> SH3
    bands = {0: 0, 9: 1, 24: 2, 45: 3}.get(rest, f"?{rest}")
    print(f"count={n} f_rest={rest} sh_degree={bands}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
