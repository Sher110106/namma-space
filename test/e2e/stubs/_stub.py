"""Stub stage used by the runner tests. Writes plausible, clearly marked ("stub": true) stage JSON.

Behaviour is set per run in <run>/stub_cfg.json, e.g.
    {"train": {"sleep": 30}, "sfm": {"exit": 10}}
so it works the same on the Mac and inside a container. Every call appends a line to
<run>/stub_calls.log ("<stage> start|end <unix time>").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipeline.common import stage as st  # noqa: E402

IDENT = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def call_log(run: Path, stage: str, what: str) -> None:
    with (run / "stub_calls.log").open("a") as fh:
        fh.write(f"{stage} {what} {time.time():.3f}\n")


def tfinal_scale(run: Path) -> float:
    t = st.read_json(run / "60_export" / "export.json", {}).get("T_final")
    return float(t[0][0]) if t else 1.0


def payload(stage: str, run: Path, quick: bool) -> dict:
    meta = st.read_json(run / "input" / "meta.json", {})
    if stage == "ingest":
        frames = sorted((run / "input" / "frames").glob("*")) if (run / "input" / "frames").is_dir() else []
        n = len(frames)
        return {"n_cand": n, "n_kept": n, "dropped": {"blur": 0, "dup": 0, "thin": 0}, "hdr": None,
                "rotation": 0, "src": "frames" if n else "none", "kept": [{"name": f"{i + 1:06d}.jpg", "t": i / 8, "sharp": 1.0} for i in range(n)]}
    if stage == "sfm":
        return {"variants": [{"name": "A", "reg_frac": 0.97, "n_models": 1, "reproj_px": 0.8, "n_pts": 20000,
                              "track_len": 5.0, "time_s": 1.0, "status": "ok"}], "chosen": "A", "why": "stub", "missing_ranges": []}
    if stage == "align":
        return {"S": IDENT, "up_resid_deg": 0.5, "floor": {"support": 0.4, "rms": 0.01}, "h_cap": 1.4,
                "scale": {"value": 1.0, "sigma_log": 0.1, "pct_68": 10.0, "cues": [{"name": "camera_height", "value": 1.0}],
                          "checked_against": None}, "height_cm": meta.get("height_cm")}
    if stage == "train":
        steps = 7000 if quick else 30000
        return {"variants": [{"name": "r0", "psnr": 24.0, "ssim": 0.8, "lpips": 0.25, "cc_psnr": 25.0, "n_gs": 900000,
                              "time_s": 2.0, "peak_gpu_gb": 6.0, "steps": steps}], "chosen": "r0", "why": "stub", "quick": quick}
    if stage == "geometry":
        return {"mesh": "mesh.ply", "note": "stub"}
    if stage == "export":
        return {"T_final": IDENT, "sd": {"count": 800000, "mb": 30.0, "sh": 1, "psnr": 23.8},
                "hd": None, "preview": None, "removed": {"nan": 0, "opacity": 10, "huge": 1, "outside": 5, "bubble": 7, "freespace": 0},
                "cmd": "stub"}
    if stage == "grid":
        k = tfinal_scale(run)
        return {"origin_xz": [-3.2 * k, -2.8 * k], "cell": 0.05, "w": 128, "h": 112, "spawn": {"position": [0.4 * k, 1.42, -1.1 * k], "yaw_deg": 0},
                "spawn_ok": True, "bookmarks": [], "tour": [], "room_extent_m": [6.4 * k, 5.6 * k], "walkable_m2": 20.0 * k * k, "path_coverage": 0.99,
                "t_final_scale_seen": k}
    return {}


def main(stage: str) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--factor", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)   # what ns rescale --factor now passes (rescale.py's name)
    args, _ = ap.parse_known_args()
    run = Path(args.run)
    cfg = st.read_json(run / "stub_cfg.json", {}).get(stage, {})
    call_log(run, stage, "start")
    if stage == "rescale":
        exp = run / "60_export" / "export.json"
        j = st.read_json(exp)
        k = args.factor or args.scale or 2.0
        j["T_final"] = [[k * IDENT[i][j2] if i < 3 and j2 < 3 else IDENT[i][j2] for j2 in range(4)] for i in range(4)]
        st.write_json_atomic(exp, j)
        call_log(run, stage, "end")
        return 0
    with st.stage_context(run, stage, min_gb=0.001) as ctx:
        ctx.log(f"stub {stage} quick={args.quick}")
        sleep = float(cfg.get("sleep", 0))
        if sleep:
            time.sleep(sleep)
        if cfg.get("exit"):
            call_log(run, stage, "end")
            raise st.StageExit(int(cfg["exit"]), f"stub {stage} told to exit {cfg['exit']}")
        if cfg.get("gate_failed"):
            print('NS_RESULT {"exit": 0, "gate_failed": true, "flags": ["stub gate flag"]}', flush=True)
        data = payload(stage, run, args.quick)
        data["stub"] = True
        data.update(cfg.get("json_extra", {}))
        name = Path(st.STAGE_BY_NAME[stage].json).name
        st.write_json_atomic(ctx.out / name, data)
        if stage == "sfm":
            (run / "30_undist.tmp_stub").mkdir(exist_ok=True)
            (run / "30_undist.tmp_stub").rmdir()
    if stage == "sfm":
        (run / "30_undist").mkdir(exist_ok=True)
        (run / "30_undist" / "stub.txt").write_text("stub\n")
    call_log(run, stage, "end")
    return 0


def entry(stage: str) -> None:
    st.run_main(lambda: main(stage))
