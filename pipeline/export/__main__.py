"""Stage `export`: P0 clean, web files (SD, HD, preview), parity renders, held-out pairs (FINAL_PLAN 2.6, 2.8, WP3).

    python -m pipeline.export --run /w/runs/<name> [--frame aligned|raw] [--no-hd] [--no-preview]

Reads 50_train/train.json (chosen PLY), 40_align/ (cameras, SfM points, photos). Writes 60_export/ through a temp
directory (60_export.tmp, renamed when complete): clean.ply, splat_sd.spz, splat_hd.spz, splat_preview.spz,
heldout/, export.json. All splat files stay in the aligned frame; T_final in export.json (a measured floor offset,
later a user rescale) is applied by consumers: X_final = scale * R * X + translation. The PLY is never edited.
Exit codes: 0 good, 20 infrastructure problem, 30 a size or loss gate failed (files and export.json still written).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from . import clean as CL
from . import colmap_io as cio
from . import heldout as HO
from . import ply_io, webfmt
from .splat_writer import write_splat, read_splat

STAGE = "60_export"
MIN_FREE_GB = 15
BUDGET_MB = {"sd": 40.0, "hd": 90.0, "preview": 4.0}
CLEAN_DROP_MAX_DB = 0.3
SD_DROP_MAX_DB = 0.5


def log(msg: str) -> None:
    print(f"[export {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sanitise(o):
    """Strict JSON: non-finite floats become null."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    return o


def _json(path: str, obj) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(_sanitise(obj), f, indent=1, allow_nan=False)
    os.replace(tmp, path)


ANTIALIASED = True      # the pipeline trains with --antialiased; --classic-render for smoke PLYs trained without it
OUTSIDE_CANDIDATES_M = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
OUTSIDE_STEP_MAX_DB = 0.10
# Step 3 ("huge": largest axis above both 1.0 m and the 99.9th percentile) cost 2.0 dB of held-out PSNR on MuSHRoom
# vr_room (312 splats: the large splats that cover its plain walls and ceiling; 24.00 -> 21.99 dB), and 0.07 dB on
# honka. Same rule as step 4: take the first axis limit whose step costs <= 0.10 dB, else skip the step (integrator,
# 30 Oct).
HUGE_CANDIDATES_M = (1.0, 2.0, 4.0, 8.0)
HUGE_STEP_MAX_DB = 0.10


def eval_psnr(s: ply_io.Splats, cams, label: str) -> Optional[dict]:
    """Held-out PSNR of a splat set by gsplat renders; None if torch/gsplat/GPU are missing."""
    try:
        import torch
        from . import render_ply as RP
    except ImportError as e:
        log(f"{label}: parity not run ({e})")
        return None
    if not torch.cuda.is_available():
        log(f"{label}: parity not run (no CUDA)")
        return None
    g = RP.gaussians_from_splats(s)
    t = time.time()
    r = RP.evaluate(g, cams, antialiased=ANTIALIASED)
    del g
    torch.cuda.empty_cache()
    log(f"{label}: held-out PSNR {r['psnr']:.3f} dB over {r['n']} views ({time.time() - t:.0f} s)")
    return r


def box_fraction(s: ply_io.Splats, pts: np.ndarray, lo_pct=1.0, hi_pct=99.0, grow=0.2) -> Optional[float]:
    """E3: share of opaque (> 0.5) splat centres inside the SfM points' 1-99% box grown by 20% of its extent."""
    if not len(pts):
        return None
    lo, hi = np.percentile(pts, lo_pct, axis=0), np.percentile(pts, hi_pct, axis=0)
    ext = hi - lo
    lo, hi = lo - grow * ext, hi + grow * ext
    m = ply_io.sigmoid(s.opacity) > 0.5
    c = s.means[m]
    return float(((c >= lo) & (c <= hi)).all(axis=1).mean()) if len(c) else None


def make_spz(tool, s: ply_io.Splats, sh: int, dst: str, scratch: str, version: int):
    """Write s (limited to SH `sh`) to a scratch PLY and convert to SPZ. Returns (ok, cmd, seconds, log)."""
    src = os.path.join(scratch, f"src_sh{sh}.ply")
    ply_io.write_ply(src, s.limited_to_degree(sh))
    cmd = webfmt.spz_command(tool, src, dst, sh, version)
    rc, dt, out = webfmt.run(cmd)
    os.remove(src)
    return rc == 0 and os.path.isfile(dst), cmd, dt, out


def decode_spz(tool, spz: str, scratch: str) -> Optional[ply_io.Splats]:
    back = os.path.join(scratch, "decoded.ply")
    rc, _, out = webfmt.run(webfmt.decode_command(tool, spz, back))
    if rc != 0 or not os.path.isfile(back):
        log("decode failed: " + out[-300:])
        return None
    s = ply_io.read_ply(back)
    os.remove(back)
    return s


def roundtrip_error(a: ply_io.Splats, b: ply_io.Splats) -> dict:
    """F3: count and position error between a source set and its decoded SPZ."""
    res = {"count_in": len(a), "count_out": len(b), "count_equal": len(a) == len(b)}
    if len(a) != len(b):
        return res
    d = np.linalg.norm(a.means.astype(np.float64) - b.means.astype(np.float64), axis=1)
    if np.median(d) < 0.05:                       # order preserved
        res.update(order_preserved=True, pos_err_max_mm=float(d.max() * 1000), pos_err_mean_mm=float(d.mean() * 1000))
    else:                                         # reordered: nearest neighbour
        from scipy.spatial import cKDTree
        dn, _ = cKDTree(a.means.astype(np.float64)).query(b.means.astype(np.float64), workers=-1)
        res.update(order_preserved=False, pos_err_max_mm=float(dn.max() * 1000), pos_err_mean_mm=float(dn.mean() * 1000))
    return res


def build_target(name: str, s: ply_io.Splats, n_max: int, sh: int, out_dir: str, scratch: str, tool, version: int,
                 mode: str, cams, base_psnr: Optional[float], budget_mb: float, warnings: List[str],
                 want_splat_fallback: bool, retry_shrink: bool = False) -> dict:
    """One web file: prune to n_max, write SPZ (or fall back to .splat), decode back, measure PSNR drop."""
    sel = CL.prune_to(s, n_max, mode)
    info: dict = {"count": len(sel), "sh": sh, "pruned_from": len(s), "prune_mode": mode if len(s) > n_max else None}
    ext = "spz"
    path = os.path.join(out_dir, f"splat_{name}.spz")
    decoded: Optional[ply_io.Splats] = None
    if tool is not None:
        shrink = 1.0
        for attempt in range(5):
            cur = sel if attempt == 0 else CL.prune_to(s, int(n_max * shrink), mode)
            ok, cmd, dt, out = make_spz(tool, cur, sh, path, scratch, version)
            if not ok:
                warnings.append(f"{name}: splat-transform failed: {out[-200:]}")
                break
            mb = os.path.getsize(path) / 1e6
            info.update(count=len(cur), cmd=" ".join(cmd), write_s=round(dt, 1))
            if mb <= budget_mb or not retry_shrink:
                sel = cur
                break
            shrink *= 0.85
            warnings.append(f"{name}: {mb:.1f} MB > {budget_mb} MB, retrying with {int(n_max * shrink)} splats")
        else:
            warnings.append(f"{name}: could not meet {budget_mb} MB")
        if os.path.isfile(path):
            decoded = decode_spz(tool, path, scratch)
            if decoded is not None:
                info["roundtrip"] = roundtrip_error(sel.limited_to_degree(sh), decoded)
    if decoded is None:
        # fallback: our own .splat writer (SH0)
        ext = "splat"
        path = os.path.join(out_dir, f"splat_{name}.splat")
        fit = int(budget_mb * 1e6 // 32)               # 32 bytes per splat
        if len(sel) > fit:
            sel = CL.prune_to(sel, fit, mode)
            warnings.append(f"{name}: .splat fallback pruned to {fit} splats to fit {budget_mb} MB")
        n = write_splat(path, sel)
        info.update(count=n, sh=0, format_note="own numpy writer, SH0, 32 B/splat (splat-transform unavailable or failed)")
        decoded = read_splat(path)
    info["file"] = os.path.basename(path)
    info["format"] = {"spz": f"spz{info.get('cmd', '').split('--spz-version ')[-1].split()[0] if 'cmd' in info else ''}", "splat": "splat"}[ext]
    info["bytes"] = os.path.getsize(path)
    info["mb"] = round(info["bytes"] / 1e6, 3)
    info["budget_mb"] = budget_mb
    info["budget_ok"] = info["mb"] <= budget_mb
    r = eval_psnr(decoded, cams, f"{name} (decoded)") if cams else None
    info["psnr"] = r["psnr"] if r else None
    info["psnr_drop_db"] = (base_psnr - r["psnr"]) if (r and base_psnr is not None) else None
    info["_per_image"] = r["per_image"] if r else {}
    info["_decoded"] = decoded
    if want_splat_fallback and ext == "spz":
        p2 = os.path.join(out_dir, f"splat_{name}.splat")
        write_splat(p2, sel)
        info["also_splat"] = {"file": os.path.basename(p2), "mb": round(os.path.getsize(p2) / 1e6, 3)}
    return info


def _drop_candidate_files(out_dir: str, info: dict) -> None:
    tag = info["_tag"]
    for f in os.listdir(out_dir):
        if f.startswith(f"splat_{tag}."):
            os.remove(os.path.join(out_dir, f))


def _finalise_candidate(out_dir: str, info: dict, name: str) -> None:
    """Rename splat_<tag>.* to splat_<name>.* and fix the recorded file names and command."""
    tag = info.pop("_tag")
    for f in sorted(os.listdir(out_dir)):
        if f.startswith(f"splat_{tag}."):
            os.rename(os.path.join(out_dir, f), os.path.join(out_dir, f.replace(f"splat_{tag}.", f"splat_{name}.", 1)))
    info["file"] = info["file"].replace(f"splat_{tag}.", f"splat_{name}.", 1)
    if "cmd" in info:
        info["cmd"] = info["cmd"].replace(f"splat_{tag}.", f"splat_{name}.")
    if "also_splat" in info:
        info["also_splat"]["file"] = info["also_splat"]["file"].replace(f"splat_{tag}.", f"splat_{name}.", 1)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.export", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", help="COLMAP data dir (default <run>/40_align)")
    ap.add_argument("--ply", help="trained PLY (default: chosen_ply of <run>/50_train/train.json)")
    ap.add_argument("--train-dir", default="50_train")
    ap.add_argument("--stage-dir", default=STAGE)
    ap.add_argument("--frame", choices=["aligned", "raw", "gsplat-normalised"], default="aligned",
                    help="raw = not the metric aligned frame (smoke tests): no floor measurement, T_final identity; "
                         "gsplat-normalised = PLY trained with world normalisation ON (cameras and points are "
                         "normalised the way gsplat's Parser does)")
    ap.add_argument("--no-hd", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--sd-max", type=int, default=1_000_000)
    ap.add_argument("--hd-max", type=int, default=1_500_000)
    ap.add_argument("--preview-n", type=int, default=200_000)
    ap.add_argument("--prune-mode", choices=["opacity", "area", "opacity_size", "volume"], default="opacity")
    ap.add_argument("--spz-version", type=int, default=4, choices=[3, 4])
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--also-splat", action="store_true", help="also write the .splat fallback of every target")
    ap.add_argument("--no-spz", action="store_true", help="skip splat-transform; ship the .splat fallback")
    ap.add_argument("--huge-axis", default="auto",
                    help="step 3 axis limit in metres, or auto: the first of 1, 2, 4, 8 m costing <= 0.10 dB, else skip")
    ap.add_argument("--outside-grow", default="auto",
                    help="auto (smallest of 0.5..16 m whose step costs <= 0.10 dB) or a number of metres (plan: 0.5)")
    ap.add_argument("--classic-render", action="store_true", help="parity renders with classic rasterisation")
    ap.add_argument("--sd-sh", default="auto",
                    help="SH degree of the SD file: 0-3, or auto = start at 1 and raise it (2, 3) while the decoded "
                         f"file loses more than {SD_DROP_MAX_DB} dB against the trained model and the file stays "
                         f"within {BUDGET_MB['sd']:.0f} MB")
    ap.add_argument("--no-breakdown", action="store_true", help="skip the per-step PSNR of the clean (saves ~10 s)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    t_stage = time.time()
    global ANTIALIASED
    ANTIALIASED = not a.classic_render

    run = os.path.abspath(a.run)
    final = os.path.join(run, a.stage_dir)
    tmp = final + ".tmp"
    data_dir = os.path.abspath(a.data_dir) if a.data_dir else os.path.join(run, "40_align")
    if os.path.isfile(os.path.join(final, "export.json")) and not a.force:
        log(f"{final}/export.json exists; nothing to do (use --force)")
        return 0
    ply = a.ply
    train_info = {}
    if not ply:
        tj = os.path.join(run, a.train_dir, "train.json")
        if not os.path.isfile(tj):
            log(f"ERROR no {tj} and no --ply")
            return 20
        train_info = json.load(open(tj))
        if not train_info.get("chosen_ply"):
            log("ERROR train.json has no chosen_ply")
            return 20
        ply = os.path.join(run, train_info["chosen_ply"])
    if not os.path.isfile(ply):
        log(f"ERROR PLY not found: {ply}")
        return 20
    try:
        mdir = cio.model_dir(data_dir)
    except (FileNotFoundError, ValueError) as e:
        log(f"ERROR {e}")
        return 20
    if shutil.disk_usage(run).free / 1e9 < MIN_FREE_GB:
        log("ERROR disk below 15 GB free")
        return 20

    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "scratch"))
    scratch = os.path.join(tmp, "scratch")
    warnings: List[str] = []

    log(f"reading {ply}")
    s_raw = ply_io.read_ply(ply)
    cams_ho = []
    try:
        from . import render_ply as RP
        if a.frame == "gsplat-normalised":
            cams_ho, centres, pts = RP.cameras_from_parser(data_dir, normalize=True, held_out=True)
            centres = RP.cameras_from_parser(data_dir, normalize=True)[1]
        else:
            pts, _, _ = cio.read_points3d(mdir)
            centres = np.array([im.centre() for im in cio.sorted_images(mdir)])
            cams_ho = RP.cameras_from_colmap(data_dir, held_out=True)
    except Exception as e:                                   # PIL/torch missing, photos missing ...
        warnings.append(f"held-out cameras unavailable: {e}")
        pts, _, _ = cio.read_points3d(mdir)
        centres = np.array([im.centre() for im in cio.sorted_images(mdir)])

    res: dict = {"schema": "namma-space/export@1", "source": {
        "ply": os.path.relpath(ply, run) if ply.startswith(run) else ply, "n_gs": len(s_raw),
        "variant": train_info.get("chosen"), "train_psnr_reported": None}, "frame": a.frame}
    if train_info:
        ch = next((v for v in train_info["variants"] if v["name"] == train_info["chosen"]), None)
        if ch:
            res["source"]["train_psnr_reported"] = ch.get("psnr")
    res["checks"] = {"E3_opaque_in_sfm_box_frac": box_fraction(s_raw, pts)}
    res["render"] = {"antialiased": ANTIALIASED, "preBlurAmount": 0.0, "blurAmount": 0.3, "maxStdDev": 2.8284271247461903,
                     "note": "CONTRACTS 2.12 defaults for a model trained with --antialiased; parity renders use gsplat "
                             + ("antialiased" if ANTIALIASED else "classic") + " rasterisation"}

    # ---- parity of the trained model, then clean
    base = eval_psnr(s_raw, cams_ho, "trained") if cams_ho else None
    res["trained"] = {"psnr": base["psnr"] if base else None, "n_views": base["n"] if base else 0}
    base_psnr = base["psnr"] if base else None

    steps_psnr: Dict[str, Optional[float]] = {}
    params = CL.CleanParams()
    outside_search: List[dict] = []
    huge_search: List[dict] = []
    huge_m = params.huge_axis_m
    NO_HUGE = dict(huge_axis_m=float("inf"), huge_percentile=100.0)   # re-cleaning a held subset must not re-run step 3
    if cams_ho and a.huge_axis == "auto" and base_psnr is not None:
        held_h: Dict[str, ply_io.Splats] = {}
        CL.clean(s_raw, np.zeros((0, 3)), np.zeros((0, 3)), CL.CleanParams(box_grow_m=float("inf")),
                 on_step=lambda n, c: held_h.setdefault(n, c) if n == "opacity" else None)
        ref_h = eval_psnr(held_h["opacity"], cams_ho, "huge-search base")["psnr"]
        huge_m = float("inf")
        for hm in HUGE_CANDIDATES_M:
            sub, rm, _ = CL.clean(held_h["opacity"], np.zeros((0, 3)), np.zeros((0, 3)),
                                  CL.CleanParams(min_opacity=0.0, huge_axis_m=hm, box_grow_m=float("inf")))
            if rm["huge"] == 0:
                huge_m = hm
                huge_search.append({"axis_m": hm, "removed": 0, "psnr": ref_h, "cost_db": 0.0})
                break
            r = eval_psnr(sub, cams_ho, f"huge axis {hm} m")
            cost = ref_h - r["psnr"]
            huge_search.append({"axis_m": hm, "removed": rm["huge"], "psnr": r["psnr"], "cost_db": cost})
            if cost <= HUGE_STEP_MAX_DB:
                huge_m = hm
                break
        del held_h
        if not np.isfinite(huge_m):
            warnings.append(f"huge-splat step skipped: every axis limit cost more than {HUGE_STEP_MAX_DB} dB ({huge_search})")
        params = CL.CleanParams(huge_axis_m=huge_m, huge_percentile=(100.0 if not np.isfinite(huge_m) else 99.9))
        log(f"huge step: axis limit {huge_m} m chosen from {huge_search}")
    elif a.huge_axis != "auto":
        huge_m = float(a.huge_axis)
        params = CL.CleanParams(huge_axis_m=huge_m)
    if cams_ho and a.outside_grow == "auto" and base_psnr is not None:
        # Step 4 can delete background splats seen through windows and doors, which the sparse SfM box does not
        # cover. Take the smallest growth whose step costs <= 0.10 dB of held-out PSNR; otherwise skip the step.
        held: Dict[str, ply_io.Splats] = {}
        CL.clean(s_raw, pts, centres, params, on_step=lambda n, c: held.setdefault(n, c) if n == "huge" else None)
        ref = eval_psnr(held["huge"], cams_ho, "outside-search base")["psnr"]
        chosen_grow = float("inf")
        for g in OUTSIDE_CANDIDATES_M:
            p2 = CL.CleanParams(box_grow_m=g, **NO_HUGE)
            sub, rm, _ = CL.clean(held["huge"], pts, np.zeros((0, 3)), p2)
            if rm["outside"] == 0:
                chosen_grow = g
                outside_search.append({"grow_m": g, "removed": 0, "psnr": ref, "cost_db": 0.0})
                break
            r = eval_psnr(sub, cams_ho, f"outside grow {g} m")
            cost = ref - r["psnr"]
            outside_search.append({"grow_m": g, "removed": rm["outside"], "psnr": r["psnr"], "cost_db": cost})
            if cost <= OUTSIDE_STEP_MAX_DB:
                chosen_grow = g
                break
        del held
        params = CL.CleanParams(box_grow_m=chosen_grow, huge_axis_m=params.huge_axis_m, huge_percentile=params.huge_percentile)
        log(f"outside-box step: grow {chosen_grow} m chosen from {outside_search}")
    elif a.outside_grow != "auto":
        params = CL.CleanParams(box_grow_m=float(a.outside_grow), huge_axis_m=params.huge_axis_m,
                                huge_percentile=params.huge_percentile)

    def on_step(name: str, cur: ply_io.Splats) -> None:
        if cams_ho and not a.no_breakdown:
            r = eval_psnr(cur, cams_ho, f"after {name}")
            steps_psnr[name] = r["psnr"] if r else None
    s_clean, removed, det = CL.clean(s_raw, pts, centres, params, on_step=on_step)
    det["outside_search"] = outside_search
    det["huge_search"] = huge_search
    del s_raw
    log(f"clean: {det['n_in']} -> {det['n_out']}  removed {removed}")
    ply_io.write_ply(os.path.join(tmp, "clean.ply"), s_clean)
    cr = eval_psnr(s_clean, cams_ho, "clean") if cams_ho else None
    res["removed"] = removed
    res["clean"] = {"file": "clean.ply", "n_in": det["n_in"], "n_out": det["n_out"], "details": det,
                    "psnr": cr["psnr"] if cr else None, "psnr_after_step": steps_psnr,
                    "psnr_drop_db": (base_psnr - cr["psnr"]) if (cr and base_psnr is not None) else None}

    # ---- T_final: floor offset measured from the cleaned splats
    tf = HO.identity_t_final()
    tf_info: dict = {"method": f"none (frame {a.frame})" if a.frame != "aligned" else None}
    if a.frame == "aligned":
        fy, finfo = CL.measure_floor_local(s_clean, centres)
        fy_peak, pinfo = CL.measure_floor_y(s_clean, centres)
        tf_info = {"method": "median height of opaque splats within 0.20 m of the align floor and 1.5 m of the path",
                   "floor_y_measured": fy, **finfo,
                   "lowest_peak_rule": {"floor_y": fy_peak, **pinfo, "used": False}}
        if fy_peak is not None and fy is not None and abs(fy_peak - fy) > 0.10:
            warnings.append(f"lowest splat-height peak at y = {fy_peak:.3f} m disagrees with the floor near the "
                            f"path ({fy:.3f} m); splats under the floor? (not used)")
        if fy is None:
            warnings.append(f"floor not measured: {finfo.get('reason')}")
        elif abs(fy) > 0.15:
            warnings.append(f"measured floor y = {fy:.3f} m is more than 0.15 m from 0; T_final left at identity")
            tf_info["applied"] = False
        else:
            tf["translation"] = [0.0, -fy, 0.0]
            tf_info["applied"] = abs(fy) >= 0.005
    res["T_final"] = {**tf, "matrix": HO.t_final_matrix(tf).reshape(-1).tolist(), "info": tf_info,
                      "note": "X_final = scale * R(quaternion) * X + translation; splat files are NOT transformed"}

    # ---- web files
    tool = None if a.no_spz else webfmt.find_splat_transform()
    res["tools"] = {"splat_transform": webfmt.tool_version(tool) if tool else None,
                    "splat_transform_cmd": " ".join(tool) if tool else None, "spz_version": a.spz_version}
    if tool is None:
        warnings.append("splat-transform not found: shipping .splat (SH0) fallback")
    sd_levels = [1, 2, 3] if a.sd_sh == "auto" else [int(a.sd_sh)]
    sd_tried = []
    res["sd"] = None
    for sh in sd_levels:
        cand = build_target(f"sd_sh{sh}", s_clean, a.sd_max, sh, tmp, scratch, tool, a.spz_version, a.prune_mode,
                            cams_ho, base_psnr, BUDGET_MB["sd"], warnings, a.also_splat)
        cand["_tag"] = f"sd_sh{sh}"
        sd_tried.append({"sh": cand["sh"], "mb": cand["mb"], "psnr": cand["psnr"], "psnr_drop_db": cand["psnr_drop_db"],
                         "budget_ok": cand["budget_ok"]})
        good = cand["budget_ok"] and (cand["psnr_drop_db"] is None or cand["psnr_drop_db"] <= SD_DROP_MAX_DB)
        prev = res["sd"]
        if cand["budget_ok"] or prev is None:      # keep the last size-compliant candidate
            res["sd"] = cand
            if prev is not None:
                _drop_candidate_files(tmp, prev)
        else:
            _drop_candidate_files(tmp, cand)
        if good or tool is None or cand["format"] == "splat" or not cand["budget_ok"]:
            break
    _finalise_candidate(tmp, res["sd"], "sd")
    res["sd"]["sh_search"] = sd_tried
    sd_per_image = res["sd"].pop("_per_image")
    sd_decoded = res["sd"].pop("_decoded")
    res["hd"] = None
    if tool is None and not a.no_hd:
        warnings.append("hd skipped: the .splat fallback is SH0, so a bigger file adds nothing over SD")
        a.no_hd = True
    if not a.no_hd:
        res["hd"] = build_target("hd", s_clean, a.hd_max, 3, tmp, scratch, tool, a.spz_version, a.prune_mode, cams_ho,
                                 base_psnr, BUDGET_MB["hd"], warnings, False, retry_shrink=True)
        res["hd"].pop("_per_image", None)
        res["hd"].pop("_decoded", None)
    res["preview"] = None
    if not a.no_preview:
        res["preview"] = build_target("preview", s_clean, a.preview_n, 0, tmp, scratch, tool, a.spz_version,
                                      a.prune_mode, cams_ho, base_psnr, BUDGET_MB["preview"], warnings, False,
                                      retry_shrink=True)
        res["preview"].pop("_per_image", None)
        res["preview"].pop("_decoded", None)

    # ---- held-out pairs in the final frame
    if cams_ho:
        pick = HO.pick_evenly(cams_ho, a.n_heldout)
        tr_per = (base or {}).get("per_image", {})
        entries = HO.write_pairs(os.path.join(tmp, "heldout"), pick, sd_per_image, tr_per, tf)
        parity_n = 0
        try:
            import torch
            from . import render_ply as RP
            if torch.cuda.is_available() and sd_decoded is not None:
                g = RP.gaussians_from_splats(sd_decoded)
                rows = HO.write_parity(os.path.join(tmp, "parity"), pick, entries,
                                       lambda cs: RP.render_views(g, cs, antialiased=ANTIALIASED))
                parity_n = len(rows)
                del g
                torch.cuda.empty_cache()
        except ImportError as e:
            warnings.append(f"parity renders not written: {e}")
        res["heldout"] = {"n": len(entries), "dir": "heldout", "file": "heldout/heldout.json",
                          "parity_dir": "parity" if parity_n else None, "parity_n": parity_n,
                          "psnr_basis": "SD web file decoded and rendered with gsplat vs the photo",
                          "ids": [e["id"] for e in entries]}
    else:
        res["heldout"] = {"n": 0, "note": "not written: no held-out cameras/photos"}

    # ---- gates
    gates = {
        "F1_sizes_ok": all(res[k]["budget_ok"] for k in ("sd", "hd", "preview") if res.get(k)),
        "F2_clean_drop_ok": (res["clean"]["psnr_drop_db"] is not None and res["clean"]["psnr_drop_db"] <= CLEAN_DROP_MAX_DB),
        "F2_sd_drop_ok": (res["sd"]["psnr_drop_db"] is not None and res["sd"]["psnr_drop_db"] <= SD_DROP_MAX_DB),
        "F3_roundtrip_ok": None,
        "F5_heldout_written": res["heldout"]["n"] >= min(a.n_heldout, len(cams_ho)) and res["heldout"]["n"] > 0,
    }
    rt = res["sd"].get("roundtrip")
    if rt:
        gates["F3_roundtrip_ok"] = bool(rt["count_equal"] and rt.get("pos_err_max_mm", 1e9) <= 2.0)
    res["gates"] = gates
    res["warnings"] = warnings
    res["cmd"] = {"spz_sd": res["sd"].get("cmd"), "spz_hd": (res["hd"] or {}).get("cmd"),
                  "spz_preview": (res["preview"] or {}).get("cmd"),
                  "stage": "python -m pipeline.export " + " ".join(argv if argv is not None else sys.argv[1:])}
    res["time_s"] = round(time.time() - t_stage, 1)
    shutil.rmtree(scratch, ignore_errors=True)
    _json(os.path.join(tmp, "export.json"), res)
    if os.path.isdir(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    # a user scale recorded earlier (ns rescale) survives a re-export: compose it into T_final again
    try:
        from pipeline.align import rescale as RS
        if a.frame == "aligned" and abs(RS.factor(run) - 1.0) > 1e-9:
            ex = RS.apply_to_export(run)
            log(f"user scale {RS.factor(run):.4g} from 40_align/rescale.json applied to T_final: {ex}")
    except Exception as e:  # never fail the export over this; the grid and bundle then use the automatic scale
        log(f"WARNING could not apply the user scale from 40_align/rescale.json: {e}")
    log(f"done in {res['time_s']} s; gates {gates}")
    bad = [k for k, v in gates.items() if v is False]
    return 30 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
