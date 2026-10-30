"""Score a metric-frame phone-only Gaussian PLY on a MuSHRoom room (J3/J4)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.align.metric_check import robust_umeyama, name_map
from pipeline.export.render_ply import cameras_from_colmap
from pipeline.geometry.tsdf import render_ply_depth, fuse_depth_views
from eval.score_mushroom import digest, mushroom_views, score_mesh


def _dataset_meta(root: Path):
    meta = json.loads((root / "iphone/long_capture/transformations.json").read_text())
    by_name = {}
    for fr in meta["frames"]:
        by_name[Path(fr["file_path"]).name] = fr
    return by_name


def _figure(path: Path, result: dict):
    d = result["depth_vs_lidar"]
    g = result["geometry"]["cropped"]
    im = Image.new("RGB", (1000, 600), "white")
    dr = ImageDraw.Draw(im)
    dr.text((30, 25), result["name"] + " phone-only evaluation", fill="black")
    rows = [
        f"depth median abs error: {d['median_abs_error_cm']:.3f} cm",
        f"depth within 5 / 10 cm: {d['within_5cm_frac']*100:.2f}% / {d['within_10cm_frac']*100:.2f}%",
        f"geometry accuracy / completeness: {g['accuracy_m']:.4f} / {g['completeness_m']:.4f} m",
        f"F-score @ 5 cm: {g['f_score_5cm']:.4f}",
        f"scale error: {result['scale']['automatic_error_pct']:.3f}%",
    ]
    for i, row in enumerate(rows):
        dr.text((30, 90 + i * 70), row, fill=(30, 70, 130))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    run, root = a.run, a.dataset
    align = json.loads((run / "40_align/align.json").read_text())
    export = json.loads((run / "60_export/export.json").read_text())
    ply = run / "60_export" / export["clean"]["file"]
    ref = _dataset_meta(root)
    nm = name_map(str(run))
    cams = cameras_from_colmap(str(run / "40_align"), held_out=False)
    # The frame stage names images 000001.jpg, while MuSHRoom's reference
    # uses frame_00001.jpg.  frames.json is the authoritative correspondence.
    for c in cams:
        c.ref_name = nm.get(c.name, c.name)  # type: ignore[attr-defined]
    cams = [c for c in cams if c.ref_name in ref]  # type: ignore[attr-defined]
    if len(cams) < 10:
        raise RuntimeError(f"only {len(cams)} cameras match MuSHRoom names")

    src = np.array([c.c2w[:3, 3] for c in cams])
    dst = np.array([np.asarray(ref[c.ref_name]["transform_matrix"], float)[:3, 3] for c in cams])  # type: ignore[attr-defined]
    s_fit, R_fit, t_fit, residuals, fit_idx = robust_umeyama(src, dst)
    T_fit = np.eye(4)
    T_fit[:3, :3] = s_fit * R_fit
    T_fit[:3, 3] = t_fit
    s_ours = float(align["s"])
    s_true = float(s_fit * s_ours)
    interval = [float(x) for x in align["scale"]["interval_68"]]

    views = render_ply_depth(str(ply), cams, device=a.device,
                             antialiased=bool(export.get("render", {}).get("antialiased", True)))
    held_names = {c.name for c in cameras_from_colmap(str(run / "40_align"), held_out=True)}
    abs_errors = []
    for v in views:
        if v.name not in held_names:
            continue
        fr = ref[nm.get(v.name, v.name)]
        dp = root / "iphone/long_capture" / fr["depth_file_path"].lstrip("./")
        lidar = np.asarray(Image.open(dp), dtype=np.float32) / 1000.0
        if lidar.shape != v.depth.shape:
            # Undistorted COLMAP images are 735x990 while the MuSHRoom depth
            # PNGs are 738x994.  Nearest-neighbour keeps metric depth values
            # unchanged while putting both maps on the render raster.
            lidar = np.asarray(Image.fromarray(lidar).resize((v.width, v.height), Image.Resampling.NEAREST), dtype=np.float32)
        # A similarity changes metric depth by its positive scale; rotation and
        # translation are already represented by the matched camera pose.
        pred = v.depth * float(s_fit)
        valid = (v.alpha >= 0.5) & (pred > 0) & np.isfinite(pred) & (lidar > 0) & np.isfinite(lidar)
        if np.any(valid):
            abs_errors.append(np.abs(pred[valid] - lidar[valid]))
    err = np.concatenate(abs_errors) if abs_errors else np.empty(0)
    depth_score = {
        "heldout_every": 8, "heldout_views": len(abs_errors),
        "pixels": int(err.size),
        "median_abs_error_cm": float(np.median(err) * 100) if err.size else None,
        "within_5cm_frac": float(np.mean(err <= .05)) if err.size else None,
        "within_10cm_frac": float(np.mean(err <= .10)) if err.size else None,
        "fit_scale_applied": float(s_fit),
    }

    mesh_path = a.out.with_suffix(".mesh.ply")
    if mesh_path.exists():
        # A rerun after a metadata-only failure can reuse the completed CPU
        # fusion; rendering is still repeated because depth maps are transient.
        mesh = None
    else:
        mesh = fuse_depth_views(views, str(mesh_path), color=False)
    geometry = score_mesh(mesh_path, root / "gt_mesh.ply", root / "icp_iphone.json", pre_transform=T_fit)
    result = {
        "schema": "namma-space/eval@1", "name": a.out.stem,
        "kind": "mushroom_phone_only", "status": "complete",
        "run": str(run), "dataset": str(root), "ply": str(ply),
        "views": len(views), "matched_views": len(cams),
        "pose_source": "40_align/sparse/0 COLMAP cameras matched by basename to MuSHRoom transformations.json",
        "pose_convention": "COLMAP/OpenCV c2w; MuSHRoom OpenGL c2w centres matched, depth scalar compared in camera rays",
        "scale": {
            "s_ours_m_per_sfm_unit": s_ours, "s_true_m_per_raw_sfm_unit": s_true,
            "fit_scale_aligned_to_dataset": float(s_fit),
            "fit_rmse_m_all": float(np.sqrt(np.mean(residuals ** 2))),
            "fit_rmse_m_inliers": float(np.sqrt(np.mean(residuals[fit_idx] ** 2))),
            "fit_n": int(len(fit_idx)), "matched_n": int(len(cams)),
            "automatic_error_pct": float(abs(s_ours / s_true - 1.0) * 100),
            "automatic_error_definition": "abs(s_ours/s_true - 1) * 100",
            "interval_68_m_per_raw_sfm_unit": interval,
            "true_inside_interval_68": bool(interval[0] <= s_true <= interval[1]),
            "similarity_matrix": T_fit.tolist(),
        },
        "depth_vs_lidar": depth_score,
        "geometry": geometry,
        "headline_protocol": "cropped predicted points inside Faro mesh bbox grown by 5 cm",
        "fusion": "Open3D ScalableTSDFVolume attempted; pinned Open3D 0.20 returned zero vertices in the synthetic probe, so deterministic point-surface fallback was used",
        "voxel_size_m": 0.02, "sdf_trunc_m": 0.08, "depth_trunc_m": 5.0,
        "git": (subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                if (run / "../.git").exists() else "unknown"),
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "image_digests": {"dataset": digest(root / "iphone/long_capture"), "gt_mesh": digest(root / "gt_mesh.ply")},
        "command": f"python -m eval.score_phone_only --run {run} --dataset {root} --out {a.out}",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    _figure(a.out.with_suffix(".png"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
