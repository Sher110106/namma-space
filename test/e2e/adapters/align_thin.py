"""Thin stand-in for pipeline.align until W2's module lands (root's adapter, standard library only).

    python3 test/e2e/adapters/align_thin.py --run <run_dir>

Reads 30_undist/{images, sparse[/0]} and writes 40_align/{sparse/0, images -> ../30_undist/images, align.json}
with the contract keys of CONTRACTS 2.4 (S, up_resid_deg, floor, h_cap, scale). Crude on purpose:
  up      = minus the mean of the cameras' y axes (trimmed), no plane fitting
  floor   = 2nd percentile of the SfM points' height; h_cap = median camera height above it
  scale   = 1.40 m / h_cap (the generic camera-height prior; sigma_log 0.12), or 0.80 * height / h_cap
  origin  = camera-path centroid dropped onto the floor
align.json carries "thin": true so nobody mistakes it for W2's result. Run with --script align=... in ns_remote.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipeline.common import colmap_io as cio  # noqa: E402
from pipeline.common import stage as st  # noqa: E402


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def rotation_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b (Rodrigues)."""
    v = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    c = sum(x * y for x, y in zip(a, b))
    if c < -0.999999:  # opposite: half turn about any perpendicular axis
        axis = norm([1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
        axis = norm([axis[1] * a[2] - axis[2] * a[1], axis[2] * a[0] - axis[0] * a[2], axis[0] * a[1] - axis[1] * a[0]])
        x, y, z = axis
        return [[2 * x * x - 1, 2 * x * y, 2 * x * z], [2 * x * y, 2 * y * y - 1, 2 * y * z], [2 * x * z, 2 * y * z, 2 * z * z - 1]]
    vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    vx2 = cio.mat_mul(vx, vx)
    return [[(1.0 if i == j else 0.0) + vx[i][j] + vx2[i][j] / (1 + c) for j in range(3)] for i in range(3)]


def clean(x):
    """JSON has no NaN: report it as null."""
    return None if isinstance(x, float) and math.isnan(x) else x


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p / 100 * len(xs))))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quick", action="store_true")
    a, _ = ap.parse_known_args(argv)
    run = Path(a.run)
    meta = st.read_json(run / "input" / "meta.json", {})
    und = run / "30_undist"
    sp = und / "sparse"
    if (sp / "0").is_dir():
        sp = sp / "0"
    if not (sp / "cameras.bin").exists():
        raise st.StageExit(20, f"{sp}/cameras.bin not found: run the sfm stage first")
    with st.stage_context(run, "align", min_gb=0.05) as ctx:
        m = cio.read_model(sp)
        before = cio.model_summary(m)
        rep0, _ = cio.reprojection_rmse(m, max_points=3000)
        downs = [cio.qvec_to_rotmat(im.qvec)[1] for im in m.images.values()]  # row 1 of R = R^T e_y
        mean = norm([sum(d[i] for d in downs) / len(downs) for i in range(3)])
        # trimmed: drop the 20% of cameras that disagree most with the mean, then average again
        scored = sorted(downs, key=lambda d: -sum(x * y for x, y in zip(d, mean)))[: max(3, int(0.8 * len(downs)))]
        mean = norm([sum(d[i] for d in scored) / len(scored) for i in range(3)])
        up = [-x for x in mean]
        angs = sorted(math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(d, mean)))))) for d in downs)
        up_resid = angs[len(angs) // 2]  # median spread of the cameras' down axes around the mean (not a plane residual)
        R = rotation_between(up, [0.0, 1.0, 0.0])
        rot = cio.similarity_transform(m, 1.0, R, [0.0, 0.0, 0.0])
        cams = list(cio.camera_centres(rot).values())
        floor_y = percentile([p.xyz[1] for p in rot.points3D.values()], 2)
        cam_y = sorted(c[1] for c in cams)
        h_cap = cam_y[len(cam_y) // 2] - floor_y
        if h_cap <= 0:
            raise st.StageExit(20, f"median camera height above the floor is {h_cap:.3f} (up direction wrong?)")
        height = meta.get("height_cm")
        target = 0.80 * height / 100.0 if height else 1.40
        s = target / h_cap
        cx = sum(c[0] for c in cams) / len(cams)
        cz = sum(c[2] for c in cams) / len(cams)
        t = [-s * cx, -s * floor_y, -s * cz]
        out = cio.similarity_transform(m, s, R, t)
        sp_out = ctx.out / "sparse" / "0"
        cio.write_model(out, sp_out)
        (ctx.out / "images").symlink_to("../30_undist/images")
        after = cio.model_summary(out)
        rep1, _ = cio.reprojection_rmse(out, max_points=3000)
        S = cio.matrix_from_similarity(s, R, t)
        heights = sorted(c[1] * 1.0 for c in cio.camera_centres(out).values())
        sigma = 0.08 if height else 0.12
        data = {
            "thin": True, "note": "root's thin adapter (test/e2e/adapters/align_thin.py), not W2's align",
            "S": S, "up_resid_deg": round(up_resid, 3), "up_resid_note": "median angle of camera y axes to their mean",
            "floor": {"support": None, "rms": None, "method": "2nd percentile of SfM point heights"},
            "h_cap": round(h_cap * s, 4), "h_cap_m": round(h_cap * s, 4),
            "scale": {"value": s, "sigma_log": sigma, "pct_68": round(100 * (math.exp(sigma) - 1), 1),
                      "cues": [{"name": "camera_height_prior", "value": s, "sigma_log": sigma,
                                "basis": f"{target:.2f} m / h_cap"}], "checked_against": None},
            "checks": {"registered_before": before["registered_images"], "registered_after": after["registered_images"],
                       "reproj_before_px": clean(rep0), "reproj_after_px": clean(rep1),
                       "camera_height_range_m": [heights[0], heights[-1]]},
        }
        st.write_json_atomic(ctx.out / "align.json", data)
        ctx.log(f"aligned {after['registered_images']} images; s={s:.4f}, h_cap={h_cap * s:.2f} m, "
                f"reproj {rep0:.3f} -> {rep1:.3f} px")
    return 0


if __name__ == "__main__":
    st.run_main(main)
