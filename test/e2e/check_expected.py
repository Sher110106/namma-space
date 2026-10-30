"""Compare a run's report.json with test/expected/gates.json (standard library only).

    python3 test/e2e/check_expected.py runs/<name>/report.json [--mode quick|full] [--gates FILE]

Each gate prints PASS, FAIL or NA (value missing; NA never fails, but 'optional' gates are quiet).
Exit 0 when nothing FAILs, 1 otherwise. Mode defaults to 'quick' when the report's train variant
says quick, else 'full'. Gates are plan thresholds, not measurements (see gates.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "expected" / "gates.json"


def resolve(rep: dict, path: str) -> list:
    """Values at a dotted path. [chosen] picks the variant named by the parent's 'chosen'; [*] fans out."""
    cur = [rep]
    for part in path.split("."):
        name, _, sel = part.partition("[")
        sel = sel.rstrip("]")
        nxt = []
        for c in cur:
            if not isinstance(c, dict) or c.get(name) is None:
                continue
            v = c[name]
            if sel == "chosen":
                chosen = c.get("chosen")
                nxt += [x for x in v if isinstance(x, dict) and x.get("name") == chosen]
            elif sel == "*":
                nxt += list(v)
            else:
                nxt.append(v)
        cur = nxt
    return cur


def check(op: str, value, want) -> bool:
    if op == ">=":
        return value >= want
    if op == "<=":
        return value <= want
    if op == "==":
        return value == want
    if op == "between":
        return want[0] <= value <= want[1]
    raise ValueError(f"unknown op {op}")


def evaluate(rep: dict, gates: dict, mode: str) -> list:
    out = []
    for g in gates["gates"]:
        if "any" not in g["modes"] and mode not in g["modes"]:
            out.append((g["id"], "SKIP", f"not for {mode} runs"))
            continue
        vals = [v for v in resolve(rep, g["path"]) if v is not None]
        if not vals:
            out.append((g["id"], "NA", "value missing" if not g.get("optional") else "optional, not produced"))
            continue
        bad = [v for v in vals if not check(g["op"], v, g["value"])]
        status = "FAIL" if bad else "PASS"
        shown = vals if len(vals) > 1 else vals[0]
        out.append((g["id"], status, f"{shown} {g['op']} {g['value']}   [{g['ref']}]"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--mode", choices=["quick", "full"])
    ap.add_argument("--gates", default=str(DEFAULT))
    a = ap.parse_args(argv)
    rep = json.loads(Path(a.report).read_text())
    gates = json.loads(Path(a.gates).read_text())
    mode = a.mode
    if not mode:
        tr = rep.get("train") or {}
        tv = tr.get("variants") or []
        quick = tr.get("quick") or any(isinstance(v, dict) and v.get("steps") and v["steps"] < 30000 for v in tv)
        mode = "quick" if quick else "full"
    results = evaluate(rep, gates, mode)
    for gid, status, detail in results:
        print(f"  {status:<4}  {gid:<22} {detail}")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"mode {mode}: {sum(1 for _, s, _ in results if s == 'PASS')} pass, {n_fail} fail, "
          f"{sum(1 for _, s, _ in results if s == 'NA')} not available")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
