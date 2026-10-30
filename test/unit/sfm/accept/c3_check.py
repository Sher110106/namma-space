#!/usr/bin/env python3
"""Evaluate the C3 failure tests from a finished run. Stdlib only (runs in namma-colmap).

usage: c3_check.py a|a2|b|c <run_dir>      prints one RESULT line per test and exits 0 when it passes
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from pipeline.sfm import colmapio  # noqa: E402


def load(run: str) -> dict:
    with open(os.path.join(run, "20_sfm", "sfm.json")) as f:
        return json.load(f)


def main() -> int:
    kind, run = sys.argv[1], sys.argv[2]
    sfm = load(run)
    rows = {v["name"]: v for v in sfm["variants"]}
    if kind == "a":
        a = max(rows.values(), key=lambda v: v.get("n_registered_raw", 0))
        ok = sfm["chosen"] is not None and sfm["partial"] and max(v["n_models"] for v in rows.values()) >= 2
        print(f"RESULT C3a {'PASS' if ok else 'FAIL'} chosen={sfm['chosen']} partial={sfm['partial']} "
              f"n_models={ {k: v['n_models'] for k, v in rows.items()} } model_sizes={ {k: v.get('model_sizes') for k, v in rows.items()} } "
              f"reg_frac={ {k: v['reg_frac'] for k, v in rows.items()} }")
        return 0 if ok else 1
    if kind == "a2":
        ok = sfm["chosen"] is None
        print(f"RESULT C3a2 {'PASS' if ok else 'FAIL'} (equal scenes: expected nothing usable, exit 10) chosen={sfm['chosen']} "
              f"model_sizes={ {k: v.get('model_sizes') for k, v in rows.items()} } why={sfm['why']}")
        return 0 if ok else 1
    if kind == "b":
        with open(os.path.join(run, "10_frames", "frames.json")) as f:
            frames = json.load(f)["kept"]
        real = {k["name"] for k in frames if k["kind"] == "real"}
        grey = {k["name"] for k in frames if k["kind"] == "grey"}
        if sfm["chosen"] is None:
            print("RESULT C3b FAIL no variant usable")
            return 1
        _, imgs = colmapio.read_model(os.path.join(run, "30_undist", "sparse"))
        names = {i.name for i in imgs}
        share = len(names & real) / len(real)
        per_variant = {k: v["reg_frac"] for k, v in rows.items()}
        ok = share >= 0.95
        print(f"RESULT C3b {'PASS' if ok else 'FAIL'} real frames registered {len(names & real)}/{len(real)} = {share:.3f}; "
              f"grey registered {len(names & grey)}/{len(grey)}; chosen={sfm['chosen']} reg_frac={per_variant}")
        return 0 if ok else 1
    if kind == "c":
        killed = os.environ.get("KILLED", "B")
        st = {k: v["status"] for k, v in rows.items()}
        ok = sfm["chosen"] is not None and st.get(killed) in ("crashed", "failed", "timeout") and sfm["chosen"] != killed
        print(f"RESULT C3c {'PASS' if ok else 'FAIL'} variant {killed} killed; statuses={st} chosen={sfm['chosen']}")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
