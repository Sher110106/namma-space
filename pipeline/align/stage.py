"""Align stage: read 30_undist (binary PINHOLE model), write 40_align (similarity applied)."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import colmap_io as cio
from . import core

EXIT_OK, EXIT_CAPTURE, EXIT_PARTIAL, EXIT_INFRA, EXIT_GATE = 0, 10, 11, 20, 30


def find_model_dir(base: str) -> str:
    """30_undist/sparse (COLMAP image_undistorter) or 30_undist/sparse/0 (mapper layout)."""
    for cand in (os.path.join(base, "sparse"), os.path.join(base, "sparse", "0")):
        if os.path.exists(os.path.join(cand, "images.bin")):
            return cand
    raise FileNotFoundError(f"no binary COLMAP model under {base}/sparse[/0]")


def read_meta(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, "input", "meta.json")
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def after_check(model: cio.Model, res: core.AlignResult) -> Dict[str, Any]:
    """Re-measure the aligned model: up should be +Y, the floor should sit at y = 0."""
    ur = core.estimate_up(model, seed=1)
    ydir = np.array([0.0, 1.0, 0.0])
    out: Dict[str, Any] = {"up_vs_y_deg": core.angle_deg(ur.up, ydir),
                           "camera_cue_vs_y_deg": core.angle_deg(ur.cues["cameras"], ydir)}
    if ur.cues["planes"] is not None:
        out["plane_cue_vs_y_deg"] = core.angle_deg(ur.cues["planes"], ydir)
    out["floor_y_m_remeasured"] = float(ur.floor_h)          # floor height along the re-estimated up, aligned frame
    out["floor_method_remeasured"] = ur.floor_info.get("method")
    h = core.camera_height(model, ydir, 0.0)
    out["h_cap_m_remeasured"] = h["h_cap"]
    return out


def run_stage(run_dir: str, height_cm: Optional[float] = None, seed: int = 0, make_figure: bool = True,
              moge: bool = False, log=print) -> Dict[str, Any]:
    t0 = time.time()
    run_dir = os.path.abspath(run_dir)
    undist = os.path.join(run_dir, "30_undist")
    mdir = find_model_dir(undist)
    meta = read_meta(run_dir)
    if height_cm is None and meta.get("height_cm"):
        height_cm = float(meta["height_cm"])
    height_m = height_cm / 100.0 if height_cm else None
    log(f"loading {mdir}")
    model = cio.load_model(mdir)
    if model.n_images < 10 or model.n_points < 200:
        raise ValueError(f"model too small to align: {model.n_images} images, {model.n_points} points (need 10 and 200)")
    log(f"{model.n_images} images, {model.n_points} points, mean reprojection {model.mean_error():.3f} px, "
        f"mean track {model.mean_track_length():.2f}")

    extra: List[core.Cue] = []
    moge_info: Dict[str, Any] = {"requested": moge, "available": False}
    if moge:
        try:
            from . import moge_cue
            cue, moge_info = moge_cue.estimate(run_dir, model, log=log)
            if cue is not None:
                extra.append(cue)
        except Exception as ex:  # optional cue: never stop the stage
            moge_info = {"requested": True, "available": False, "error": f"{type(ex).__name__}: {ex}"}
            log(f"MoGe cue skipped: {moge_info['error']}")

    res = core.align_model(model, height_m=height_m, seed=seed, extra_cues=extra)
    aligned = cio.apply_similarity(model, res.s, res.S[:3, :3] / res.s, res.S[:3, 3])
    chk = after_check(aligned, res)

    tmp = os.path.join(run_dir, f"40_align.tmp-{os.getpid()}")
    final = os.path.join(run_dir, "40_align")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(os.path.join(tmp, "sparse"))
    cio.save_model(aligned, os.path.join(tmp, "sparse", "0"))
    os.symlink(os.path.join("..", "30_undist", "images"), os.path.join(tmp, "images"))

    ur = res.up
    iv = core.scale_interval(res.s, res.sigma_log)
    floor = ur.floor
    fi = ur.floor_info
    fl: Dict[str, Any] = {
        "support": int(fi.get("support", 0)),
        "rms": (float(fi["rms_model_units"] * res.s) if fi.get("rms_model_units") == fi.get("rms_model_units") else None),   # metres
        "rms_sfm_units": fi.get("rms_model_units") if fi.get("rms_model_units") == fi.get("rms_model_units") else None,
        "method": fi.get("method"), "agrees_with_walls": fi.get("agrees_with_walls"),
        "n_points_total": int(model.n_points),
        "support_frac": float(fi.get("support", 0) / max(1, model.n_points)),
        "height_sfm_units": float(ur.floor_h),
    }
    if ur.ceiling_h is not None:
        fl["ceiling_height_m"] = float((ur.ceiling_h - ur.floor_h) * res.s)
    if fi.get("wall_extent"):
        fl["wall_extent_sfm_units"] = fi["wall_extent"]
    out: Dict[str, Any] = {
        "schema": "namma-space/align@1",
        "S": res.S.tolist(), "s": res.s, "R": (res.S[:3, :3] / res.s).tolist(), "t": res.S[:3, 3].tolist(),
        "up_resid_deg": ur.resid_deg,
        "up": {"sfm_frame": ur.up.tolist(), "cue_spread_deg": ur.cue_spread_deg,
               "cues": {k: ({"vec": v.tolist(), "angle_to_final_deg": ur.cue_angles_deg[k]} if v is not None else None)
                        for k, v in ur.cues.items()},
               "weights": ur.cue_weights},
        "floor": fl,
        "h_cap": float(res.h_cap_m), "h_cap_m": float(res.h_cap_m), "h_cap_sfm_units": float(res.height["h_cap"]),
        "camera_height": res.height,
        "scale": {"value": res.s, "sigma_log": res.sigma_log, "pct_68": iv["pct_68"], "interval_68": [iv["lo"], iv["hi"]],
                  "method": "+".join(c.name for c in res.cues if c.used),
                  "cues": [{"name": c.name, "value": c.value, "sigma_log": c.sigma_log, "used": c.used, "note": c.note}
                           for c in res.cues],
                  "user_height_m": height_m, "checked_against": None},
        "yaw": res.yaw,
        "walls": [{"normal_sfm": w.n.tolist(), "support": w.support, "rms_sfm_units": w.rms} for w in ur.walls],
        "horizontal_planes": [{"normal_sfm": p.n.tolist(), "offset": p.d, "support": p.support} for p in ur.horizontals],
        "flags": res.flags,
        "moge": moge_info,
        "model": {"registered": {"before": model.n_images, "after": aligned.n_images},
                  "points": {"before": model.n_points, "after": aligned.n_points},
                  "mean_reproj_px": {"before": model.mean_error(), "after": aligned.mean_error()},
                  "mean_track_len": {"before": model.mean_track_length(), "after": aligned.mean_track_length()},
                  "path_extent_m": float(np.linalg.norm(np.ptp(aligned.centers(), axis=0)))},
        "after": chk,
        "seed": seed, "seconds": round(time.time() - t0, 1),
    }
    gate_reasons = [f for f in out["flags"] if f.startswith(("up cues disagree", "no floor", "H_cap x scale", "H_cap is not"))]
    out["gate"] = {"passed": not gate_reasons, "reasons": gate_reasons}
    if make_figure:
        try:
            from . import figure
            os.makedirs(os.path.join(tmp, "figures"), exist_ok=True)
            fn = floor.n if floor else ur.up
            fo = floor.d if floor else ur.floor_h
            figure.render(os.path.join(tmp, "figures", "level_before_after.png"), model.xyz, model.centers(),
                          aligned.xyz, aligned.centers(), ur.up, fn, fo,
                          {"h_cap_m": res.h_cap_m, "up_resid_deg": ur.resid_deg, "cue_spread_deg": ur.cue_spread_deg,
                           "scale": res.s, "pct_68": iv["pct_68"]},
                          title=os.path.basename(run_dir))
        except Exception as ex:
            out["flags"].append(f"figure failed: {type(ex).__name__}: {ex}")
    with open(os.path.join(tmp, "align.json"), "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    log(f"wrote {final}")
    return out
