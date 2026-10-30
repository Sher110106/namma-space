"""STAND-IN: build a rough metric run directory from the honka smoke-test artefacts.

This is NOT the pipeline's align/export output. It exists so the grid stage can be tested on a real capture before
W2 (align) and W3 (train, export) deliver. What it does:

1. Reads the COLMAP model of the undistorted smoke run (original COLMAP frame) and gsplat's Parser with
   normalize=True to get the 4x4 transform gsplat applied to the cameras and to the trained PLY.
2. Fits a similarity (Umeyama) from the COLMAP camera centres to the MuSHRoom iPhone metric camera centres
   (transformations.json, same frame names), which gives metric scale and the dataset's own up direction.
   The honest label is "rough metric": it is the dataset's tracking, not our align stage.
3. Rotates so up is +Y, finds the floor as the lowest strong peak of splat heights near the camera path, and moves
   it to y = 0. Puts the camera-path centroid at x = z = 0.
4. Applies the P0 clean steps of FINAL_PLAN 2.6 that are cheap (opacity < 0.05, camera bubble 0.15 m, outside the
   SfM box grown by 0.5 m), writes the PLY and a COLMAP model in the new frame, plus align.json / export.json.

Run inside namma-space-gpu:0.1 (needs gsplat's example Parser for the normalisation transform):
  nsdocker ns-w4-standin namma-space-gpu:0.1 python -m pipeline.grid.standin_honka --out /w/runs/w4-honka_standin
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import colmapio, plyio


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / xs.var(0).sum()
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0]) if abs(a[0]) < 0.9 else np.diag([-1.0, 1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def floor_height(y: np.ndarray, h_cam: float) -> tuple[float, dict]:
    """Lowest strong peak of the splat-height histogram below the cameras (>= 25% of the largest peak)."""
    lo, hi = np.percentile(y, 0.2), min(np.percentile(y, 99.8), h_cam - 0.6)
    bins = np.arange(lo, hi + 0.02, 0.02)
    h, _ = np.histogram(y, bins=bins)
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    pk = [k for k in range(1, len(hs) - 1) if hs[k] >= hs[k - 1] and hs[k] > hs[k + 1] and hs[k] >= 0.25 * hs.max()]
    k = pk[0] if pk else int(np.argmax(hs))
    sel = np.abs(y - (bins[k] + 0.01)) <= 0.04
    return float(np.median(y[sel])), {"peak_bins": [round(float(bins[p] + 0.01), 3) for p in pk[:6]], "largest": int(hs.max())}


def fit_floor_plane(pts: np.ndarray, seed: int = 0, thr: float = 0.03, iters: int = 800):
    """RANSAC plane through points that are already near-horizontal; returns the unit normal (pointing up) and the inlier mask."""
    rng = np.random.default_rng(seed)
    best, best_n = -1, None
    for _ in range(iters):
        q = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if n[1] < 0:
            n = -n
        if n[1] < 0.9:
            continue
        cnt = int((np.abs((pts - q[0]) @ n) < thr).sum())
        if cnt > best:
            best, best_n, best_q = cnt, n, q[0]
    inl = np.abs((pts - best_q) @ best_n) < thr
    P = pts[inl]
    _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    n = Vt[-1]
    return (n if n[1] > 0 else -n), inl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/w/runs/smoke_honka/clean")
    ap.add_argument("--ply", default="/w/runs/smoke_honka/gs_out5/ply/point_cloud_29999.ply")
    ap.add_argument("--dataset", default="/w/data/mushroom/room_datasets/honka/iphone/long_capture/transformations.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    notes: dict = {"stand_in": True, "note": "rough metric stand-in from MuSHRoom tracking, not the align/export stage"}

    sys.path.insert(0, "/opt/gsplat/examples")
    from datasets.colmap import Parser           # noqa: E402  (gsplat example code, Apache-2.0)
    parser = Parser(data_dir=a.model, factor=1, normalize=True, test_every=8)
    T_norm = np.array(parser.transform, dtype=np.float64)           # original COLMAP frame -> gsplat normalised frame
    model = colmapio.read_model(Path(a.model) / "sparse" / "0")
    names = [Path(n).name for n in model.names]

    # dataset metric camera centres and up vectors
    ds = json.load(open(a.dataset))
    by = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in ds["frames"]}
    idx = [k for k, n in enumerate(names) if n in by]
    src = model.centres[idx]
    dst = np.array([by[names[k]][:3, 3] for k in idx])
    s, Ru, tu = umeyama(src, dst)
    res = np.linalg.norm(dst - (s * (Ru @ src.T).T + tu), axis=1)
    notes["umeyama"] = {"n": len(idx), "scale": s, "rms_m": float(np.sqrt((res ** 2).mean())), "median_m": float(np.median(res))}
    ups = np.array([by[names[k]][:3, 1] for k in idx])               # OpenGL camera up axis in the dataset frame
    up_ds = ups.mean(0)
    up_ds /= np.linalg.norm(up_ds)
    Rup = rot_between(up_ds, np.array([0.0, 1.0, 0.0]))
    notes["dataset_up"] = [round(float(v), 4) for v in up_ds]

    # similarity COLMAP frame -> y-up metric frame (floor and origin still to fix)
    Rm = Rup @ Ru
    tm = Rup @ tu
    sm = s
    cen = sm * (Rm @ model.centres.T).T + tm
    fwd = (Rm @ model.forward.T).T
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 1], -1, 1)))
    notes["pitch_deg_median"] = float(np.median(pitch))

    # splats: normalised frame -> COLMAP frame -> y-up metric frame
    sp = plyio.read_splats(a.ply)
    Tn_inv = np.linalg.inv(T_norm)
    xyz_colmap = sp.xyz.astype(np.float64) @ Tn_inv[:3, :3].T + Tn_inv[:3, 3]
    sc_norm = np.cbrt(abs(np.linalg.det(T_norm[:3, :3])))
    xyz = sm * (Rm @ xyz_colmap.T).T + tm
    scale = sp.scale * (sm / sc_norm)
    notes["gsplat_normalisation_scale"] = float(sc_norm)

    # floor: lowest strong height peak near the camera path, then a plane fit to level it (two rounds)
    path = cen[:, [0, 2]]
    near = cKDTree(path).query(xyz[:, [0, 2]], k=1)[0] < 2.0
    ok = near & (sp.opacity > 0.5)
    tilt_log = []
    for rnd in range(2):
        fy, finfo = floor_height(xyz[ok, 1], float(np.median(cen[:, 1])))
        cand = ok & (np.abs(xyz[:, 1] - fy) < 0.15)
        n, inl = fit_floor_plane(xyz[cand])
        tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
        tilt_log.append({"round": rnd, "tilt_deg": round(tilt, 3), "candidates": int(cand.sum()), "inliers": int(inl.sum())})
        Rx = rot_between(n, np.array([0.0, 1.0, 0.0]))
        xyz = xyz @ Rx.T
        cen = cen @ Rx.T
        Rm, tm = Rx @ Rm, Rx @ tm
        fwd = (Rm @ model.forward.T).T
        pitch = np.degrees(np.arcsin(np.clip(fwd[:, 1], -1, 1)))
    fy, finfo = floor_height(xyz[ok, 1], float(np.median(cen[:, 1])))
    inl_y = xyz[ok & (np.abs(xyz[:, 1] - fy) < 0.05), 1]
    notes["floor"] = {"y_before": fy, "plane_fit": tilt_log, "residual_std_m": float(inl_y.std()), **finfo}
    shift = np.array([-cen[:, 0].mean(), -fy, -cen[:, 2].mean()])
    xyz += shift
    cen += shift
    tm = tm + shift
    lvl = np.abs(pitch) <= 20.0
    h_cap = float(np.median(cen[lvl, 1]))
    notes["h_cap_m"] = h_cap
    notes["camera_height_p10_p90_m"] = [float(np.percentile(cen[:, 1], 10)), float(np.percentile(cen[:, 1], 90))]

    # cheap P0 clean (FINAL_PLAN 2.6, steps 2, 4, 5)
    keep = sp.opacity >= 0.05
    n0 = len(keep)
    pts = sm * (Rm @ model.pt_xyz.T).T + tm
    lo, hi = np.percentile(pts, 0.5, axis=0) - 0.5, np.percentile(pts, 99.5, axis=0) + 0.5
    box = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    bubble = cKDTree(cen).query(xyz, k=1)[0] < 0.15
    notes["clean"] = {"n_in": n0, "low_opacity": int((~keep).sum()), "outside_box": int((~box).sum()), "bubble": int(bubble.sum())}
    keep &= box & ~bubble
    notes["clean"]["n_out"] = int(keep.sum())

    # write the run directory
    if out.exists():
        shutil.rmtree(out)
    (out / "40_align" / "sparse" / "0").mkdir(parents=True)
    (out / "60_export").mkdir(parents=True)
    R_new = np.einsum("nij,kj->nik", model.R, Rm)                    # R_i' = R_i R^T
    t_new = sm * model.t - np.einsum("nij,j->ni", R_new, tm)         # t_i' = s t_i - R_i' t
    new = colmapio.Model(model.names, R_new, t_new, model.point_ids, model.pt_ids, pts, model.raw_images, model.raw_points)
    colmapio.write_model(out / "40_align" / "sparse" / "0", new)
    cam_bin = Path(a.model) / "sparse" / "0" / "cameras.bin"
    if cam_bin.exists():
        shutil.copy(cam_bin, out / "40_align" / "sparse" / "0" / "cameras.bin")
    check = colmapio.read_model(out / "40_align" / "sparse" / "0")
    notes["camera_centre_roundtrip_max_err_m"] = float(np.abs(check.centres - cen).max())
    (out / "40_align" / "align.json").write_text(json.dumps({"h_cap": h_cap, "stand_in": True, "up_resid_deg": None,
                                                           "floor": {"support": None, "rms": None}}, indent=1))
    plyio.write_ply(out / "60_export" / "clean.ply", xyz[keep].astype(np.float32), sp.opacity[keep], scale[keep], sp.rgb[keep])
    (out / "60_export" / "export.json").write_text(json.dumps({"T_final": np.eye(4).tolist(), "stand_in": True}, indent=1))
    (out / "STANDIN.json").write_text(json.dumps(notes, indent=1))
    print(json.dumps(notes, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
