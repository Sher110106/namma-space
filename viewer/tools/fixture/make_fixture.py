#!/usr/bin/env python3
"""Build the viewer TEST FIXTURE from the honka smoke run (numpy; runs in namma-space-gpu:0.1).

This is a stand-in for the real pipeline output until W2/W3/W4 deliver. It writes a fake run
directory with the layout of CONTRACTS 2.11 (input/, 40_align/, 50_train/, 60_export/, 70_grid/)
so that viewer/tools/bundle.py is exercised on the same shape of data.

What is real: the splats (gsplat MCMC, 30k steps, honka iPhone long capture), the camera poses
(from the parser the trainer used, so they sit in the same frame as the PLY), the photos.
What is assumed: metric scale (median camera height is set to 1.42 m) and the floor level
(densest splat layer below the cameras). The grid is a simplified placeholder in the spirit of
FINAL_PLAN 2.7, not W4's output.

Usage (inside the container, repo at /w/repo):
  python viewer/tools/fixture/make_fixture.py --data /w/runs/smoke_honka/clean \
      --ply /w/runs/smoke_honka/gs_out5/ply/point_cloud_29999.ply --out /w/runs/w5-fixture/run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/opt/gsplat/examples")

C0 = 0.28209479177387814
EYE_HEIGHT = 1.42
CELL = 0.05


def read_ply(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        names: list[str] = []
        n = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype=np.float32, count=n * len(names)).reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}
    out = {
        "xyz": data[:, [col["x"], col["y"], col["z"]]],
        "dc": data[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]],
        "rest": data[:, [col[f"f_rest_{i}"] for i in range(45)]],
        "opacity": data[:, col["opacity"]],
        "scale": data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]],
        "quat": data[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]],
    }
    return out


def write_ply(path: str, d: dict[str, np.ndarray], sh_degree: int) -> None:
    nrest = {0: 0, 1: 9, 2: 24, 3: 45}[sh_degree]
    per = nrest // 3
    cols = [d["xyz"], d["dc"]]
    if per:
        rest = d["rest"].reshape(-1, 3, 15)[:, :, :per].reshape(-1, nrest)
        cols.append(rest)
    cols += [d["opacity"][:, None], d["scale"], d["quat"]]
    arr = np.ascontiguousarray(np.concatenate(cols, axis=1).astype(np.float32))
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(nrest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {arr.shape[0]}\n".encode())
        for nm in names:
            f.write(f"property float {nm}\n".encode())
        f.write(b"end_header\n")
        arr.tofile(f)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def mat_to_quat_xyzw(m: np.ndarray) -> list[float]:
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return [round(float(v), 7) for v in q]


def yaw_pitch_deg(fwd: np.ndarray) -> tuple[float, float]:
    f = fwd / np.linalg.norm(fwd)
    return math.degrees(math.atan2(-f[0], -f[2])), math.degrees(math.asin(max(-1, min(1, f[1]))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sd", type=int, default=800_000)
    ap.add_argument("--n-preview", type=int, default=150_000)
    ap.add_argument("--render", action="store_true", help="render the held-out and training poses with gsplat (needs a GPU) for per-photo PSNR and parity checks")
    args = ap.parse_args()

    from datasets.colmap import Parser  # type: ignore
    import cv2
    from scipy import ndimage as ndi

    out = args.out
    for sub in ["input", "40_align", "50_train", "60_export/heldout", "70_grid", "test_photos", "parity"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # ---- cameras in the trainer's frame -------------------------------------------------
    P = Parser(args.data, factor=1, normalize=True, test_every=8)
    c2w = np.asarray(P.camtoworlds, dtype=np.float64)
    n_img = len(c2w)
    cen = c2w[:, :3, 3]
    up = -np.mean(c2w[:, :3, 1], axis=0)
    up /= np.linalg.norm(up)
    hc = cen @ up
    print("images", n_img, "up (trainer frame)", up, "cam height range", hc.min(), hc.max())

    # ---- splats -------------------------------------------------------------------------
    d = read_ply(args.ply)
    n_all = len(d["xyz"])
    op = sigmoid(d["opacity"])
    vol = np.exp(d["scale"].sum(axis=1))
    score = op * np.cbrt(vol)
    order = np.argsort(-score)
    keep_sd = np.sort(order[: args.n_sd])
    keep_pv = np.sort(order[: args.n_preview])
    sd = {k: v[keep_sd] for k, v in d.items()}
    pv = {k: v[keep_pv] for k, v in d.items()}
    write_ply(os.path.join(out, "60_export", "sd_sh1.ply"), sd, 1)
    write_ply(os.path.join(out, "60_export", "preview_sh0.ply"), pv, 0)
    print("splats total", n_all, "sd", len(keep_sd), "preview", len(keep_pv))

    # ---- floor, scale, frame ------------------------------------------------------------
    opaque = op > 0.5
    hs = d["xyz"][opaque] @ up
    med_cam = float(np.median(hc))
    lo = float(np.percentile(hs, 0.5))
    bins = np.linspace(lo, med_cam, 300)
    hist, edges = np.histogram(hs[(hs > lo) & (hs < med_cam)], bins=bins)
    cand = np.where(edges[:-1] < med_cam - 0.35 * (med_cam - lo))[0]
    peak = cand[int(np.argmax(hist[cand]))]
    floor_h = float(0.5 * (edges[peak] + edges[peak + 1]))
    s = EYE_HEIGHT / (med_cam - floor_h)
    print(f"median camera height {med_cam:.3f}, floor {floor_h:.3f}, scale s = {s:.4f} (m per unit)")

    Ra = rot_between(up, np.array([0.0, 1.0, 0.0]))
    cmean = cen.mean(axis=0)
    X0 = cmean - ((cmean @ up) - floor_h) * up
    position = (-s * Ra @ X0)
    transform = {"position": [round(float(v), 6) for v in position], "quaternion": mat_to_quat_xyzw(Ra), "scale": round(float(s), 6)}

    def to_world(p: np.ndarray) -> np.ndarray:
        return s * (p - X0) @ Ra.T

    cw = to_world(cen)                       # camera centres, world frame
    Rw = np.einsum("ij,njk->nik", Ra, c2w[:, :3, :3])  # camera-to-world rotation, world frame
    fwd = Rw[:, :, 2]                        # OpenCV +z forward
    yaw_pitch = np.array([yaw_pitch_deg(f) for f in fwd])

    # ---- grid (placeholder, simplified FINAL_PLAN 2.7) ----------------------------------
    pw = to_world(sd["xyz"])
    opq = sigmoid(sd["opacity"]) > 0.5
    pxz = pw[:, [0, 2]]
    ext_lo = np.minimum(cw[:, [0, 2]].min(axis=0) - 3.0, np.percentile(pxz, 1, axis=0))
    ext_hi = np.maximum(cw[:, [0, 2]].max(axis=0) + 3.0, np.percentile(pxz, 99, axis=0))
    ext_lo = np.maximum(ext_lo, cw[:, [0, 2]].min(axis=0) - 8.0)
    ext_hi = np.minimum(ext_hi, cw[:, [0, 2]].max(axis=0) + 8.0)
    origin = np.floor(ext_lo / CELL) * CELL
    gw, gh = (np.ceil((ext_hi - origin) / CELL)).astype(int)

    def cell_of(xz: np.ndarray) -> np.ndarray:
        j = np.floor((xz[:, 0] - origin[0]) / CELL).astype(int)
        i = np.floor((xz[:, 1] - origin[1]) / CELL).astype(int)
        return np.stack([i, j], axis=1)

    def disk(r: int) -> np.ndarray:
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        return (xx * xx + yy * yy) <= r * r

    # camera path rasterised densely
    path = np.zeros((gh, gw), bool)
    for a, b in zip(cw[:-1], cw[1:]):
        if np.linalg.norm((b - a)[[0, 2]]) > 1.5:  # a jump in the trajectory: do not bridge it
            continue
        for t in np.linspace(0, 1, 40):
            ij = cell_of(((1 - t) * a + t * b)[None, [0, 2]])[0]
            if 0 <= ij[0] < gh and 0 <= ij[1] < gw:
                path[ij[0], ij[1]] = True
    ij = cell_of(cw[:, [0, 2]])
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < gh) & (ij[:, 1] >= 0) & (ij[:, 1] < gw)
    path[ij[ok, 0], ij[ok, 1]] = True

    inb = lambda ij_: (ij_[:, 0] >= 0) & (ij_[:, 0] < gh) & (ij_[:, 1] >= 0) & (ij_[:, 1] < gw)
    ijs = cell_of(pxz)
    ib = inb(ijs)
    floor_band = np.abs(pw[:, 1]) <= 0.07 * EYE_HEIGHT
    fl = np.zeros((gh, gw), bool)
    sel = ib & floor_band
    fl[ijs[sel, 0], ijs[sel, 1]] = True
    fl = ndi.binary_dilation(fl, structure=disk(3))
    fl |= ndi.binary_dilation(path, structure=disk(int(0.35 * EYE_HEIGHT / CELL)))
    fl = ndi.binary_closing(np.pad(fl, 8), structure=disk(6))[8:-8, 8:-8]

    band = (pw[:, 1] > 0.1 * EYE_HEIGHT) & (pw[:, 1] < 1.25 * EYE_HEIGHT) & opq
    cnt = np.zeros((gh, gw), np.int32)
    sel = ib & band
    np.add.at(cnt, (ijs[sel, 0], ijs[sel, 1]), 1)
    obst = cnt >= 3
    lab, nl = ndi.label(obst)
    if nl:
        sizes = ndi.sum(obst, lab, index=np.arange(1, nl + 1))
        near_path = ndi.binary_dilation(path, structure=disk(int(0.5 / CELL)))
        touches = ndi.maximum(near_path, lab, index=np.arange(1, nl + 1)) > 0
        small = np.where((sizes < 10) & touches)[0] + 1
        obst[np.isin(lab, small)] = False
    infl = ndi.binary_dilation(obst, structure=disk(int(round(0.18 * EYE_HEIGHT / CELL))))
    corridor = ndi.binary_dilation(path, structure=disk(int(0.2 / CELL)))  # the camera walked here: free by definition
    near = ndi.binary_dilation(path, structure=disk(int(1.1 * EYE_HEIGHT / CELL)))
    walk = ((fl & ~infl) | corridor) & near
    # spawn: camera nearest the path centroid that lies in a walkable cell
    order_c = np.argsort(np.linalg.norm(cw[:, [0, 2]] - cw[:, [0, 2]].mean(axis=0), axis=1))
    spawn_i = int(order_c[0])
    for k in order_c:
        ii, jj = cell_of(cw[k : k + 1, [0, 2]])[0]
        if walk[ii, jj]:
            spawn_i = int(k)
            break
    lab, nl = ndi.label(walk)
    si, sj = cell_of(cw[spawn_i : spawn_i + 1, [0, 2]])[0]
    walk = lab == lab[si, sj]
    floor_out = fl | walk
    grid = np.zeros((gh, gw), np.uint8)
    grid[floor_out] = 128
    grid[walk] = 255
    cv2.imwrite(os.path.join(out, "70_grid", "grid.png"), grid)
    print("grid", gw, "x", gh, "walkable m2", round(float(walk.sum()) * CELL * CELL, 1))

    # ---- minimap: top-down max projection of SH0 colours below 2.0 m --------------------
    ppm = 40
    mw, mh = int(math.ceil(gw * CELL * ppm)), int(math.ceil(gh * CELL * ppm))
    sel = (pw[:, 1] < 2.0) & (pw[:, 1] > -0.2) & opq
    px = np.floor((pxz[sel, 0] - origin[0]) * ppm).astype(int)
    pz = np.floor((pxz[sel, 1] - origin[1]) * ppm).astype(int)
    colr = np.clip(0.5 + C0 * sd["dc"][sel], 0, 1)
    ht = pw[sel, 1]
    ok = (px >= 0) & (px < mw) & (pz >= 0) & (pz < mh)
    px, pz, colr, ht = px[ok], pz[ok], colr[ok], ht[ok]
    o = np.argsort(ht)  # ascending: later (higher) writes win
    img = np.zeros((mh, mw, 3), np.float32)
    have = np.zeros((mh, mw), bool)
    img[pz[o], px[o]] = colr[o]
    have[pz[o], px[o]] = True
    idx = ndi.distance_transform_edt(~have, return_distances=False, return_indices=True)
    filled = img[idx[0], idx[1]]
    dist = ndi.distance_transform_edt(~have)
    filled[dist > 10] = 0.06
    cv2.imwrite(os.path.join(out, "70_grid", "minimap.png"), cv2.cvtColor((filled * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # ---- spawn, bookmarks, tour ----------------------------------------------------------
    def view(k: int) -> dict:
        return {"position": [round(float(cw[k, 0]), 3), EYE_HEIGHT, round(float(cw[k, 2]), 3)],
                "yaw_deg": round(float(yaw_pitch[k, 0]), 1), "pitch_deg": 0}

    spawn = view(spawn_i)
    # bookmarks: k-means (k=5) on camera centres in the walkable region, snapped to the nearest camera
    in_walk = np.array([walk[tuple(cell_of(cw[k : k + 1, [0, 2]])[0])] if 0 <= cell_of(cw[k : k + 1, [0, 2]])[0][0] < gh and 0 <= cell_of(cw[k : k + 1, [0, 2]])[0][1] < gw else False for k in range(n_img)])
    pts = cw[in_walk][:, [0, 2]]
    rng = np.random.default_rng(1)
    cent = pts[rng.choice(len(pts), 5, replace=False)]
    for _ in range(30):
        lab_k = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), axis=1)
        for k in range(5):
            if (lab_k == k).any():
                cent[k] = pts[lab_k == k].mean(axis=0)
    cam_ids = np.where(in_walk)[0]
    books = []
    for k in range(5):
        j = cam_ids[int(np.argmin(((pts - cent[k]) ** 2).sum(-1)))]
        b = view(int(j))
        b.update({"id": f"v{k+1}", "name": f"View {k+1}"})
        books.append(b)
    # tour: follow the real walk, every ~0.6 m, timed at 1.2 m/s
    tour, acc, last, tour_last = [], 0.0, None, None
    for k in range(n_img):
        if not in_walk[k]:
            continue
        p = cw[k][[0, 2]]
        if last is not None:
            seg = float(np.linalg.norm(p - last))
            if seg > 1.5:
                last = p
                continue
            acc += seg
        if tour_last is None or np.linalg.norm(p - tour_last) >= 0.6:
            v = view(k)
            v["t"] = round(acc / 1.2, 2)
            tour.append(v)
            tour_last = p
        last = p
    print("bookmarks", len(books), "tour stops", len(tour))

    # ---- photos: held-out (every 8th) and training poses ---------------------------------
    def entry(k: int, dst_dir: str, tag: str) -> dict:
        cid = P.camera_ids[k]
        K = np.asarray(P.Ks_dict[cid])
        W, H = P.imsize_dict[cid]
        name = f"{k:06d}"
        shutil.copyfile(P.image_paths[k], os.path.join(dst_dir, name + ".jpg"))
        rc = Rw[k] @ np.diag([1.0, -1.0, -1.0])
        return {"id": name, "photo": f"heldout/{name}.jpg" if tag == "heldout" else f"{name}.jpg",
                "position": [round(float(v), 4) for v in cw[k]], "quaternion": mat_to_quat_xyzw(rc),
                "fov_y_deg": round(math.degrees(2 * math.atan(H / (2 * K[1, 1]))), 4),
                "principal_point": [round(float(K[0, 2] / W), 5), round(float(K[1, 2] / H), 5)],
                "aspect": round(W / H, 5), "width": int(W), "height": int(H), "psnr": 0, "image_name": P.image_names[k]}

    def gsplat_render(ks: list[int]) -> dict[int, np.ndarray]:
        """Render the SD splat set (SH1, classic rasterisation, trainer frame) at cameras ks with gsplat; returns uint8 HxWx3."""
        import torch
        from gsplat import rasterization
        dev = "cuda"
        q = torch.tensor(sd["quat"], device=dev, dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True)
        rest = sd["rest"].reshape(-1, 3, 15)[:, :, :3].transpose(0, 2, 1)           # N x 3 coeffs x 3 channels
        sh = np.concatenate([sd["dc"][:, None, :], rest], axis=1)                    # N x 4 x 3
        out_imgs = {}
        with torch.no_grad():
            for k in ks:
                cid = P.camera_ids[k]
                K = torch.tensor(np.asarray(P.Ks_dict[cid]), device=dev, dtype=torch.float32)[None]
                W, H = P.imsize_dict[cid]
                w2c = torch.tensor(np.linalg.inv(c2w[k]), device=dev, dtype=torch.float32)[None]
                img, _, _ = rasterization(torch.tensor(sd["xyz"], device=dev), q, torch.exp(torch.tensor(sd["scale"], device=dev)),
                                          torch.sigmoid(torch.tensor(sd["opacity"], device=dev)), torch.tensor(sh, device=dev, dtype=torch.float32),
                                          w2c, K, int(W), int(H), sh_degree=1, near_plane=0.01, far_plane=1e10, rasterize_mode="classic", packed=False)
                out_imgs[k] = (img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        return out_imgs

    def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
        mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        return 10 * math.log10(255.0 * 255.0 / max(mse, 1e-9))

    held_idx = [k for k in range(n_img) if k % 8 == 0][2:-2]   # skip the first and last held-out frames (poorest registered)
    pick = np.linspace(0, len(held_idx) - 1, 8).round().astype(int)
    held_k = [held_idx[i] for i in pick]
    heldout = [entry(k, os.path.join(out, "60_export", "heldout"), "heldout") for k in held_k]
    train_idx = [k for k in range(n_img) if k % 8 != 0]
    pick = np.linspace(len(train_idx) * 0.15, len(train_idx) * 0.85, 3).round().astype(int)
    train_k = [train_idx[i] for i in pick]
    train_poses = [entry(k, os.path.join(out, "test_photos"), "train") for k in train_k]
    if args.render:
        import imageio.v2 as imageio
        os.makedirs(os.path.join(out, "parity"), exist_ok=True)
        renders = gsplat_render(held_k + train_k)
        parity = []
        for k, e in zip(held_k + train_k, heldout + train_poses):
            photo = imageio.imread(P.image_paths[k])[..., :3]
            r = renders[k]
            if photo.shape != r.shape:
                photo = cv2.resize(photo, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
            e["psnr"] = round(psnr_u8(r, photo), 2)               # gsplat render of the shipped splat set against the photo
            name = f"{e['id']}.png"
            cv2.imwrite(os.path.join(out, "parity", name), cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
            parity.append({**{kk: e[kk] for kk in ("id", "position", "quaternion", "fov_y_deg", "principal_point", "aspect", "width", "height")}, "render": name, "psnr_vs_photo": e["psnr"]})
        with open(os.path.join(out, "parity", "parity.json"), "w") as f:
            json.dump(parity, f, indent=1)
        print("gsplat renders", len(parity), "psnr vs photo", [p_["psnr_vs_photo"] for p_ in parity])
    with open(os.path.join(out, "60_export", "heldout", "heldout.json"), "w") as f:
        json.dump(heldout, f, indent=1)
    with open(os.path.join(out, "test_photos", "train_poses.json"), "w") as f:
        json.dump(train_poses, f, indent=1)

    # ---- run-dir JSON files in the shapes of CONTRACTS 2.11 ------------------------------
    walk_m2 = float(walk.sum()) * CELL * CELL
    ys, xs = np.where(walk)
    extent = sorted([float((xs.max() - xs.min()) * CELL), float((ys.max() - ys.min()) * CELL)], reverse=True)
    grid_json = {"origin_xz": [round(float(origin[0]), 3), round(float(origin[1]), 3)], "cell": CELL, "w": int(gw), "h": int(gh),
                 "spawn": spawn, "bookmarks": books, "tour": tour, "room_extent_m": [round(e, 2) for e in extent],
                 "walkable_m2": round(walk_m2, 1), "path_coverage": None,
                 "note": "TEST FIXTURE placeholder grid written by viewer/tools/fixture/make_fixture.py, not W4 output"}
    with open(os.path.join(out, "70_grid", "grid.json"), "w") as f:
        json.dump(grid_json, f, indent=1)
    T_final = {"position": transform["position"], "quaternion": transform["quaternion"], "scale": transform["scale"],
               "note": "trainer frame (gsplat normalised) to metric y-up floor-at-0; scale assumed from camera height"}
    with open(os.path.join(out, "60_export", "export.json"), "w") as f:
        json.dump({"T_final": T_final,
                   "sd": {"count": int(len(keep_sd)), "mb": 0, "sh": 1, "psnr": None},
                   "preview": {"count": int(len(keep_pv)), "mb": 0, "sh": 0, "psnr": None},
                   "removed": {}, "cmd": "viewer/tools/fixture (top-N by opacity x volume, then Spark transcodeSpz)"}, f, indent=1)
    with open(os.path.join(out, "40_align", "align.json"), "w") as f:
        json.dump({"up_resid_deg": None, "h_cap_m": EYE_HEIGHT,
                   "scale": {"value": round(float(s), 6), "pct_68": 15.0, "method": "assumed: median camera height = 1.42 m",
                             "cues": [], "checked_against": None}}, f, indent=1)
    with open(os.path.join(out, "50_train", "train.json"), "w") as f:
        json.dump({"variants": [{"name": "smoke_honka_mcmc30k_bilagrid", "psnr": 20.62, "ssim": 0.807, "lpips": 0.242,
                                 "cc_psnr": 23.96, "n_gs": int(n_all), "time_s": 1080, "peak_gpu_gb": None}],
                   "chosen": "smoke_honka_mcmc30k_bilagrid", "why": "only variant (fixture)"}, f, indent=1)
    with open(os.path.join(out, "input", "meta.json"), "w") as f:
        json.dump({"height_cm": None, "measures": [], "notes": "TEST FIXTURE: MuSHRoom honka iPhone long capture, CC BY 4.0", "takes": []}, f, indent=1)
    with open(os.path.join(out, "test_photos", "frame.json"), "w") as f:
        json.dump({"up_trainer": up.tolist(), "floor_h": floor_h, "median_cam_h": med_cam, "X0": X0.tolist(), "transform": transform,
                   "n_images": n_img, "n_all": n_all}, f, indent=1)
    print("done", out)


if __name__ == "__main__":
    main()
