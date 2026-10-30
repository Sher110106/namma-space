"""Run the optional 55_geometry stage for a pipeline run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .tsdf import DEPTH_TRUNC, SDF_TRUNC, VOXEL_SIZE, fuse_depth_views, render_ply_depth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.geometry")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ply", help="Gaussian PLY (defaults to chosen train PLY)")
    ap.add_argument("--out", help="output directory (default <run>/55_geometry)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--classic", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "55_geometry"
    out.mkdir(parents=True, exist_ok=True)
    ply = Path(a.ply) if a.ply else _chosen_ply(run)
    if not ply.exists():
        raise FileNotFoundError(f"Gaussian PLY not found: {ply}")
    model = run / "40_align"
    if not model.exists():
        raise FileNotFoundError(f"aligned COLMAP model not found: {model}")
    from pipeline.export.render_ply import cameras_from_colmap
    cams = cameras_from_colmap(str(model), held_out=False)
    views = render_ply_depth(str(ply), cams, device=a.device, antialiased=not a.classic)
    mesh_path = out / "mesh.ply"
    mesh = fuse_depth_views(views, str(mesh_path), voxel_size=VOXEL_SIZE,
                            sdf_trunc=SDF_TRUNC, depth_trunc=DEPTH_TRUNC)
    # Store a compact query sample for the P1 free-space filter.  The full
    # query is exposed by tsdf.free_space_query and is intentionally not run on
    # every Gaussian here because the clean stage controls its keep rule.
    (out / "free_space.json").write_text(json.dumps({"status": "available", "margin_m": 3 * VOXEL_SIZE,
                                                       "views": len(views)}) + "\n")
    result = {"schema": "namma-space/geometry@1", "ply": str(ply), "mesh": str(mesh_path),
              "views": len(views), "vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
              "voxel_size_m": VOXEL_SIZE, "sdf_trunc_m": SDF_TRUNC, "depth_trunc_m": DEPTH_TRUNC,
              "render_mode": "RGB+ED", "alpha_threshold": 0.5,
              "free_space": {"margin_m": 3 * VOXEL_SIZE, "query": "pipeline.geometry.tsdf.free_space_query"}}
    (out / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print("NS_RESULT " + json.dumps(result))
    return 0


def _chosen_ply(run: Path) -> Path:
    train = run / "50_train"
    try:
        obj = json.loads((train / "train.json").read_text())
        for key in ("chosen_ply", "final_ply"):
            if obj.get(key):
                return Path(obj[key])
        chosen = obj.get("chosen")
        if chosen:
            v = train / str(chosen) / "ply/point_cloud_29999.ply"
            if v.exists():
                return v
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    found = sorted(train.glob("**/ply/point_cloud_*.ply"))
    if not found:
        raise FileNotFoundError(f"no PLY under {train}")
    return found[-1]


if __name__ == "__main__":
    raise SystemExit(main())
