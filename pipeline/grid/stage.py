"""The grid stage: reads the earlier stages' outputs, writes <run>/70_grid/ (FINAL_PLAN 2.7, WP4)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import colmapio, extent as extent_mod, frame, minimap as minimap_mod, pngio, viewpoints
from .plyio import read_splats
from .walkable import Params, build, eye_violations, measure_floor_offset

SCHEMA = "namma-space/grid@1"
PLY_CANDIDATES = ("60_export/clean.ply", "60_export/splat_clean.ply", "60_export/cleaned.ply",
                  "60_export/splat_hd.ply", "60_export/splat_sd.ply")


class InputError(Exception):
    """A required input is missing or unreadable (exit code 20)."""


def find_ply(run: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise InputError(f"--ply {p} does not exist")
        return p
    for rel in PLY_CANDIDATES:
        if (run / rel).exists():
            return run / rel
    cands = sorted((run / "60_export").glob("*.ply")) if (run / "60_export").is_dir() else []
    if cands:
        return max(cands, key=lambda q: q.stat().st_size)
    tj = frame.read_json(run / "50_train" / "train.json", {}) or {}
    chosen = tj.get("chosen")
    for pat in ([f"50_train/{chosen}/ply/point_cloud_29999.ply"] if chosen else []) + ["50_train/*/ply/point_cloud_29999.ply"]:
        hit = sorted(run.glob(pat))
        if hit:
            return hit[0]
    raise InputError(f"no cleaned PLY found under {run}/60_export (pass --ply)")


def load_frame_sharpness(run: Path, names: list[str]) -> Optional[np.ndarray]:
    fj = frame.read_json(run / "10_frames" / "frames.json", None)
    if not fj or not fj.get("kept"):
        return None
    by = {Path(k.get("name", "")).name: k.get("sharp") for k in fj["kept"]}
    vals = [by.get(Path(n).name) for n in names]
    if sum(v is not None for v in vals) < max(3, len(names) // 2):
        return None
    arr = np.array([np.nan if v is None else float(v) for v in vals])
    arr[np.isnan(arr)] = np.nanmedian(arr)
    return arr


def _walk_stats(state, cam_xz, lay):
    i, j = lay.ij(cam_xz[:, 0], cam_xz[:, 1])
    ok = lay.inside(i, j)
    inside = np.zeros(len(cam_xz), dtype=bool)
    inside[ok] = state.walkable[i[ok], j[ok]]
    return float(inside.mean()) if len(inside) else 0.0


def _bounds_state(state, pt_xyz: np.ndarray, p: Params):
    """Last fallback: every cell inside the SfM point box shrunk by 0.3 m, grown to hold the camera path (free fly, labelled)."""
    lay = state.layout
    if len(pt_xyz) >= 10:
        lo = np.percentile(pt_xyz[:, [0, 2]], 1, axis=0) + 0.3
        hi = np.percentile(pt_xyz[:, [0, 2]], 99, axis=0) - 0.3
    else:                                       # no SfM points: the camera path box grown by 1 m
        lo = state.path_xz.min(axis=0) - 1.0 + 0.3
        hi = state.path_xz.max(axis=0) + 1.0 - 0.3
    lo = np.minimum(lo, state.path_xz.min(axis=0) - 0.2)      # the cameras are always inside (they saw the points from outside the box)
    hi = np.maximum(hi, state.path_xz.max(axis=0) + 0.2)
    x, z = lay.xz(np.arange(lay.h)[:, None], np.arange(lay.w)[None, :])
    walk = (x >= lo[0]) & (x <= hi[0]) & (z >= lo[1]) & (z <= hi[1])
    state.walkable = walk
    state.floor = state.floor | walk
    state.obstacle = np.zeros_like(walk)
    return state


def run_stage(run_dir: str, out_dir: Optional[str] = None, ply: Optional[str] = None, cell: float = 0.05,
              seed: int = 0, tfinal: str = "auto", force_mode: Optional[str] = None, floor_y: float = 0.0,
              make_tour: bool = True, h_cap_override: Optional[float] = None, activations: str = "auto",
              quiet: bool = False, summary: Optional[dict] = None) -> int:
    t_start = time.time()
    run = Path(run_dir)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def log(msg: str) -> None:
        if not quiet:
            print(f"[grid] {msg}", file=sys.stderr, flush=True)

    # ---- inputs
    try:
        align_dir = run / "40_align"
        model_dir = align_dir / "sparse" / "0"
        if not model_dir.is_dir():
            raise InputError(f"missing {model_dir}")
        plyp = find_ply(run, ply)
        align = frame.read_json(align_dir / "align.json", {}) or {}
        export = frame.read_json(run / "60_export" / "export.json", {}) or {}
        t0 = time.time()
        model = colmapio.read_model(model_dir)
        timings["read_model_s"] = round(time.time() - t0, 3)
        t0 = time.time()
        splats = read_splats(plyp, activations=activations)
        timings["read_ply_s"] = round(time.time() - t0, 3)
    except (InputError, FileNotFoundError, ValueError) as e:
        print(f"[grid] input problem: {e}", file=sys.stderr)
        if summary is not None:
            summary.update({"exit": 20, "reason": str(e)})
        return 20
    if len(model.names) < 3 or len(splats) < 1000:
        print(f"[grid] input problem: {len(model.names)} cameras and {len(splats)} splats are too few", file=sys.stderr)
        if summary is not None:
            summary.update({"exit": 20, "reason": "too few cameras or splats"})
        return 20
    log(f"{len(splats):,} splats from {plyp.name}, {len(model.names)} cameras ({timings['read_ply_s']} s for the PLY)")

    # ---- final frame: apply T_final to the PLY and the model (CONTRACTS 2.0)
    T = np.eye(4)
    tf_used = False
    if tfinal != "skip" and export.get("T_final") is not None:
        T = frame.parse_tfinal(export["T_final"])
        tf_used = not np.allclose(T, np.eye(4))
    s_T = frame.tfinal_scale(T)
    if tf_used:
        splats.xyz = frame.apply_points(T, splats.xyz.astype(np.float64)).astype(np.float32)
        splats.scale = (splats.scale * s_T).astype(np.float32)
    centres = frame.apply_points(T, model.centres)
    fwd = frame.apply_dirs(T, model.forward)
    pt_xyz = frame.apply_points(T, model.pt_xyz) if len(model.pt_xyz) else model.pt_xyz
    pitch = frame.pitch_deg_from_forward(fwd)

    # ---- H_cap
    level = np.abs(pitch) <= 20.0
    h_cams = float(np.median(centres[level, 1] - floor_y)) if level.any() else float(np.median(centres[:, 1] - floor_y))
    h_align = align.get("h_cap", align.get("h_cap_m"))
    if h_cap_override is not None:
        h_cap, h_src = float(h_cap_override), "override"
    elif isinstance(h_align, (int, float)) and h_align > 0:
        h_cap, h_src = float(h_align) * s_T, "align.json"
        if abs(h_cap / max(h_cams, 1e-9) - 1.0) > 0.10:
            warnings.append(f"align.json h_cap {h_cap:.2f} m differs from camera height above y={floor_y} ({h_cams:.2f} m) by more than 10%")
    else:
        h_cap, h_src = h_cams, "cameras (no h_cap in align.json)"
        warnings.append("no h_cap in align.json; used the median camera height above y = floor_y")
    if not (1.0 <= h_cap <= 1.8):
        warnings.append(f"H_cap {h_cap:.2f} m is outside 1.0-1.8 m: scale or floor error")
    log(f"H_cap {h_cap:.3f} m ({h_src}); camera-derived {h_cams:.3f} m; T_final scale {s_T:.4f}")

    names = [Path(n).name for n in model.names]
    order = np.empty(len(names), dtype=np.int64)
    order[np.argsort(np.array(names, dtype=object), kind="stable")] = np.arange(len(names))
    cnt, med = viewpoints.camera_stats(model.point_ids, model.pt_ids, model.pt_xyz, model.R, model.t, depth_scale=s_T)
    cams = viewpoints.Cams(names, centres, fwd, pitch, load_frame_sharpness(run, names), cnt, med, order)

    p = Params(cell=cell, seed=seed)

    def attempt(mode: str, prev):
        t0 = time.time()
        if mode == "bounds":
            st = _bounds_state(prev if prev is not None else build(splats, centres, names, h_cap, p, floor_y=floor_y), pt_xyz, p)
        else:
            st = build(splats, centres, names, h_cap, p, floor_y=floor_y, mode=mode)
        timings[f"build_{mode}_s"] = round(time.time() - t0, 3)
        c_ = _walk_stats(st, centres[:, [0, 2]], st.layout)
        sp, rl = viewpoints.choose_spawn(cams, st.layout, st.walkable)
        log(f"mode {mode}: coverage {c_:.3f}, spawn {'ok' if sp is not None else 'none'}, "
            f"walkable {st.walkable.sum() * cell * cell:.1f} m2")
        return st, c_, sp, rl

    ladder = [force_mode] if force_mode else ["grid", "corridor", "bounds"]
    mode, state, cov, spawn_idx, relaxed = None, None, 0.0, None, []
    for m in ladder:
        state, cov, spawn_idx, relaxed = attempt(m, state)
        mode = m
        if spawn_idx is not None and cov >= p.min_path_coverage:
            break
        warnings.append(f"mode {m} failed: path coverage {cov:.3f}, spawn {'found' if spawn_idx is not None else 'not found'}")
    if mode == "bounds":
        warnings.append("free fly inside the SfM bounds shrunk by 0.3 m (last fallback)")
    lay = state.layout
    walk = state.walkable
    info = dict(state.info)
    warnings += [f"spawn rule relaxed: {r}" for r in relaxed]

    # ---- independent re-check of the eye test on the final mask
    ev = eye_violations(splats, lay, walk, h_cap, floor_y, p) if mode != "bounds" else np.zeros_like(walk)
    info["eye_violations_final"] = int(ev.sum())

    # ---- spawn, bookmarks, tour
    t0 = time.time()
    spawn = None
    spawn_ok = spawn_idx is not None
    bookmarks: list[dict] = []
    tour: list[dict] = []
    tour_stops: list[dict] = []
    if spawn_ok:
        spawn = viewpoints._view(cams, spawn_idx, h_cap, floor_y)
        bookmarks, bm_idx = viewpoints.choose_bookmarks(cams, lay, walk, h_cap, floor_y, k=5, seed=seed)
        if make_tour:
            tour, tour_stops, tinfo = viewpoints.plan_tour(cams, lay, walk, h_cap, floor_y, spawn_idx, seed=seed)
            info.update(tinfo)
            if not tour:
                warnings.append(f"no tour: {tinfo.get('tour')}")
    else:
        warnings.append("no spawn found")
    timings["viewpoints_s"] = round(time.time() - t0, 3)

    # ---- room extent, floor offset
    t0 = time.time()
    ext = None
    if len(state.obst_xz):
        oi, oj = np.nonzero(state.obstacle)
        ox_, oz_ = lay.xz(oi, oj)
        ext = extent_mod.room_extent(np.stack([ox_, oz_], axis=1), state.obst_xz, centres[:, [0, 2]], cell=cell, seed=seed)
    if ext is None:
        warnings.append("room extent not measured: no wall-like obstacle columns found")
    floor_off = measure_floor_offset(splats, state.d_path, lay, h_cap, floor_y)
    if floor_off is not None and abs(floor_off) > 0.05 * h_cap:
        warnings.append(f"measured floor is {floor_off * 100:+.1f} cm from y = {floor_y}: check T_final")
    timings["extent_s"] = round(time.time() - t0, 3)

    # ---- bounds for the scene
    y = splats.xyz[:, 1]
    sel = (y > floor_y - 0.3) & (splats.opacity > 0.3)
    pts = splats.xyz[sel] if sel.sum() > 100 else splats.xyz
    bmin = np.percentile(pts, 0.5, axis=0)
    bmax = np.percentile(pts, 99.5, axis=0)

    # ---- images
    t0 = time.time()
    grid_u8 = state.to_u8()
    ppm = minimap_mod.choose_px_per_m(lay)
    mini = minimap_mod.render(splats, lay, walk, ppm, floor_y=floor_y)
    timings["minimap_s"] = round(time.time() - t0, 3)

    walkable_m2 = float(walk.sum()) * cell * cell
    result = {
        "schema": SCHEMA,
        "origin_xz": [round(lay.ox, 6), round(lay.oz, 6)],
        "cell": cell, "w": lay.w, "h": lay.h, "row0": "min_z",
        "values": {"walkable": 255, "floor": 128, "blocked": 0},
        "minimap": {"url": "minimap.png", "origin_xz": [round(lay.ox, 6), round(lay.oz, 6)], "px_per_m": ppm,
                    "w": int(mini.shape[1]), "h": int(mini.shape[0])},
        "spawn": spawn, "spawn_ok": bool(spawn_ok),
        "bookmarks": bookmarks, "tour": tour, "tour_stops": tour_stops,
        "room_extent_m": ext["room_extent_m"] if ext else None,
        "room_extent_axes_m": ext["extent_axes_m"] if ext else None,
        "room_axes_yaw_deg": ext["axes_yaw_deg"] if ext else None,
        "walkable_m2": round(walkable_m2, 3),
        "path_coverage": round(cov, 4),
        "mode": mode,
        "eye_height_m": round(h_cap, 4), "h_cap_m": round(h_cap, 4), "h_cap_source": h_src,
        "h_cap_cameras_m": round(h_cams, 4),
        "floor_y": floor_y, "floor_offset_measured_m": None if floor_off is None else round(floor_off, 4),
        "bounds": {"min": [round(float(v), 3) for v in bmin], "max": [round(float(v), 3) for v in bmax]},
        "counts": {"splats": len(splats), "cameras": len(model.names), "sfm_points": int(len(model.pt_ids)),
                   "walkable_cells": int(walk.sum()), "floor_cells": int(state.floor.sum()),
                   "obstacle_cells": int(state.obstacle.sum())},
        "diagnostics": info,
        "inputs": {"ply": str(plyp), "model": str(model_dir), "ply_activations": splats.activations,
                   "t_final_applied": bool(tf_used), "t_final_scale": round(s_T, 6),
                   "frames_json": bool(cams.sharp is not None)},
        "params": {k: v for k, v in vars(p).items()},
        "warnings": warnings,
        "timings_s": timings,
    }
    timings["total_s"] = round(time.time() - t_start, 3)

    # ---- write through <dir>.tmp, then rename (same naming as pipeline.common.stage.atomic_stage)
    out = Path(out_dir) if out_dir else run / "70_grid"
    tmp, failed, old = (out.with_name(out.name + sfx) for sfx in (".tmp", ".failed", ".old"))
    for stale in (tmp, failed, old):
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        pngio.write_png(tmp / "grid.png", grid_u8)
        pngio.write_png(tmp / "minimap.png", mini)
        (tmp / "grid.json").write_text(json.dumps(result, indent=1))
    except BaseException:
        os.replace(tmp, failed)
        raise
    if out.exists():
        os.replace(out, old)
    os.replace(tmp, out)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    log(f"wrote {out} in {timings['total_s']} s: walkable {walkable_m2:.1f} m2, coverage {cov:.3f}, mode {mode}")
    rc = 0 if (mode == "grid" and spawn_ok) else 30
    if summary is not None:
        summary.update({"exit": rc, "gate_failed": rc == 30, "mode": mode, "spawn_ok": bool(spawn_ok),
                        "walkable_m2": result["walkable_m2"], "path_coverage": result["path_coverage"],
                        "room_extent_m": result["room_extent_m"], "warnings": len(warnings)})
    return rc
