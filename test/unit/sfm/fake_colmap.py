#!/usr/bin/env python3
"""A fake `colmap` for orchestration tests. Behaviour comes from the FAKE_SCENARIO environment variable (JSON):

{"A": {"models": [{"names": "1-100", "far": "5-7"}], "reproj": 0.8, "pts": 1000, "track": 4.0,
       "fail": "match", "sleep": {"match": 0.0}}, ...}

The variant is taken from the path of the database or output directory (/A/, /B/, /C/, /D/).
`names` and `far` are 1-based inclusive index ranges into the sorted image list ("1-100,120-130").
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import write_model  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pipeline.sfm import colmapio  # noqa: E402


def arg(argv: list[str], flag: str, default: str | None = None) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else default


def ranges(spec: str) -> list[int]:
    out: list[int] = []
    for part in (spec or "").split(","):
        if part.strip():
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def variant_of(argv: list[str]) -> str:
    for flag in ("--database_path", "--output_path", "--input_path", "--path"):
        v = arg(argv, flag)
        if v:
            for part in os.path.normpath(v).split(os.sep):
                if part in ("A", "B", "C", "D"):
                    return part
    return "A"


def main() -> int:
    argv = sys.argv[1:]
    sub = argv[0]
    if sub in ("-h", "--help", "help"):
        print("fake colmap")
        return 0
    sc = json.loads(os.environ.get("FAKE_SCENARIO", "{}"))
    v = variant_of(argv)
    s = sc.get(v, {})
    time.sleep(s.get("sleep", {}).get(sub, 0.0))
    alias = {"extract": ("feature_extractor",), "match": ("sequential_matcher", "exhaustive_matcher"),
             "map": ("mapper", "global_mapper")}
    if s.get("fail") and (s["fail"] == sub or sub in alias.get(s["fail"], ())):
        print(f"fake failure in {sub}", file=sys.stderr)
        return 1
    if sub == "feature_extractor":
        db, img = arg(argv, "--database_path"), arg(argv, "--image_path")
        names = sorted(os.listdir(img))
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
        con.execute("CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        for i, n in enumerate(names, start=1):
            con.execute("INSERT INTO images VALUES (?,?,1)", (i, n))
            con.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (i, 100 + i, 2, b""))
        con.commit()
        con.close()
    elif sub in ("sequential_matcher", "exhaustive_matcher", "view_graph_calibrator"):
        pass
    elif sub in ("mapper", "global_mapper"):
        img = arg(argv, "--image_path")
        names = sorted(os.listdir(img))
        out = arg(argv, "--output_path")
        for k, m in enumerate(s.get("models", [])):
            idxs, far = ranges(m["names"]), set(ranges(m.get("far", "")))
            cams = [(names[i - 1], (1000.0 + i, 5.0, 0.0) if i in far else (i * 0.1, 0.0, 0.0)) for i in idxs]
            write_model(os.path.join(out, str(k)), cams, {"points": s.get("pts", 1000), "track": s.get("track", 4.0),
                                                        "reproj": s.get("reproj", 0.8)})
    elif sub == "model_analyzer":
        m = arg(argv, "--path")
        _, imgs = colmapio.read_model(m)
        st = json.load(open(os.path.join(m, "stats.json")))
        print(f"I0000 00:00:00.000000 1 model.cc:437] Registered images: {len(imgs)}")
        print(f"I0000 00:00:00.000000 1 model.cc:442] Points: {st.get('points', 0)}")
        print(f"I0000 00:00:00.000000 1 model.cc:445] Mean track length: {st.get('track', 4.0)}")
        print(f"I0000 00:00:00.000000 1 model.cc:450] Mean reprojection error: {st.get('reproj', 0.8)}px")
    elif sub == "image_deleter":
        src, dst = arg(argv, "--input_path"), arg(argv, "--output_path")
        drop = {ln.strip() for ln in open(arg(argv, "--image_names_path")) if ln.strip()}
        _, imgs = colmapio.read_model(src)
        st = json.load(open(os.path.join(src, "stats.json")))
        write_model(dst, [(r.name, r.centre) for r in imgs if r.name not in drop], st)
    elif sub == "point_filtering":
        src, dst = arg(argv, "--input_path"), arg(argv, "--output_path")
        _, imgs = colmapio.read_model(src)
        st = json.load(open(os.path.join(src, "stats.json")))
        st["reproj"] = round(st.get("reproj", 0.8) * 0.9, 4)
        write_model(dst, [(r.name, r.centre) for r in imgs], st)
    elif sub == "image_undistorter":
        img, src, dst = arg(argv, "--image_path"), arg(argv, "--input_path"), arg(argv, "--output_path")
        _, imgs = colmapio.read_model(src)
        os.makedirs(os.path.join(dst, "images"), exist_ok=True)
        for r in imgs:
            shutil.copy(os.path.join(img, r.name), os.path.join(dst, "images", r.name))
        write_model(os.path.join(dst, "sparse"), [(r.name, r.centre) for r in imgs], {}, model_id=1)
        os.makedirs(os.path.join(dst, "stereo"), exist_ok=True)
        open(os.path.join(dst, "run-colmap-geometric.sh"), "w").write("#fake\n")
    else:
        print(f"fake colmap: unknown command {sub}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
