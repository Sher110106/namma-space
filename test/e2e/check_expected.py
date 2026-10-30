"""Compare a run's report.json with test/expected/gates.json (standard library only).

    python3 test/e2e/check_expected.py runs/<name>/report.json [--mode quick|full] [--gates FILE]
    python3 test/e2e/check_expected.py runs/<name>/report.json --expected test/expected/report.json [--case NAME]

With --expected the report is compared with measured values from a reference run, each with a tolerance
band (`range`), instead of the plan's gates. Values outside their band print DIFF; the exit code is 1 then.

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
            elif sel.isdigit():
                if isinstance(v, list) and int(sel) < len(v):
                    nxt.append(v[int(sel)])
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


def compare_expected(rep: dict, exp: dict, case: str | None) -> int:
    cases = exp.get("cases") or {}
    if not cases:
        print("no cases in the expected file")
        return 1
    if case is None:
        case = next(iter(cases))
    if case not in cases:
        print(f"no case {case!r}; cases: {', '.join(cases)}")
        return 1
    c = cases[case]
    n_ok = n_diff = n_na = 0
    for v in c["values"]:
        vals = [x for x in resolve(rep, v["path"]) if x is not None]
        if not vals:
            n_na += 1
            print(f"  NA    {v['id']:<26} value missing (reference {v.get('measured')})")
            continue
        lo, hi = v["range"]
        bad = [x for x in vals if not ((lo is None or x >= lo) and (hi is None or x <= hi))]
        shown = vals if len(vals) > 1 else vals[0]
        if bad:
            n_diff += 1
            print(f"  DIFF  {v['id']:<26} {shown} outside [{lo}, {hi}] (reference {v.get('measured')})")
        else:
            n_ok += 1
            print(f"  OK    {v['id']:<26} {shown} in [{lo}, {hi}] (reference {v.get('measured')})")
    print(f"case {case}: {n_ok} within tolerance, {n_diff} outside, {n_na} not available")
    return 1 if n_diff else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--mode", choices=["quick", "full"])
    ap.add_argument("--gates", default=str(DEFAULT))
    ap.add_argument("--expected", help="test/expected/report.json: compare with a reference run's values")
    ap.add_argument("--case", help="which case in the expected file (default: the first)")
    a = ap.parse_args(argv)
    rep = json.loads(Path(a.report).read_text())
    if a.expected:
        return compare_expected(rep, json.loads(Path(a.expected).read_text()), a.case)
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
