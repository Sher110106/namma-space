"""A synthetic room as a COLMAP binary model, with known up, floor, camera height and scale.

Room 5.0 x 4.0 x 2.6 m (x, z horizontal; y up), floor y = 0.  A table top (1.2 x 0.8 m at 0.75 m).
Cameras walk a loop at 1.40 m, pitched -25 to +5 deg, yaw following the loop, small roll.
Points sit on the floor, ceiling, four walls and the table, with noise.  A random similarity
(rotation Q, scale s0, translation) then turns it into a 'raw SfM frame' like COLMAP would give.
"""
from __future__ import annotations

import os
import struct
from typing import Tuple

import numpy as np

from pipeline.align import colmap_io as cio


def rot_from_axis_angle(axis, ang):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return cio.qvec_to_rotmat(q)


def build(seed: int = 0, n_cams: int = 120, n_pts: int = 9000, floor_pts: int = 2500) -> Tuple[dict, dict]:
    """Return (world, truth): world has cam_from_world R (n,3,3), t (n,3), points (m,3) in the true metric frame."""
    rng = np.random.default_rng(seed)
    X, Y, Z = 5.0, 2.6, 4.0
    pts = []
    def plane(n, f):
        pts.append(f(rng.uniform(size=(n, 2))))
    plane(floor_pts, lambda u: np.c_[u[:, 0] * X - X / 2, np.zeros(len(u)), u[:, 1] * Z - Z / 2])
    plane(n_pts // 4, lambda u: np.c_[u[:, 0] * X - X / 2, np.full(len(u), Y), u[:, 1] * Z - Z / 2])
    plane(n_pts // 5, lambda u: np.c_[u[:, 0] * X - X / 2, u[:, 1] * Y, np.full(len(u), -Z / 2)])
    plane(n_pts // 5, lambda u: np.c_[u[:, 0] * X - X / 2, u[:, 1] * Y, np.full(len(u), Z / 2)])
    plane(n_pts // 6, lambda u: np.c_[np.full(len(u), -X / 2), u[:, 0] * Y, u[:, 1] * Z - Z / 2])
    plane(n_pts // 6, lambda u: np.c_[np.full(len(u), X / 2), u[:, 0] * Y, u[:, 1] * Z - Z / 2])
    plane(n_pts // 10, lambda u: np.c_[u[:, 0] * 1.2 - 0.6, np.full(len(u), 0.75), u[:, 1] * 0.8 - 0.4])   # table
    P = np.vstack(pts) + rng.normal(scale=0.01, size=(sum(len(p) for p in pts), 3))
    # camera loop (ellipse), looking along the direction of travel turned a little inward, pitch -25..5, roll +-3 deg
    th = np.linspace(0, 2 * np.pi, n_cams, endpoint=False)
    C = np.c_[1.6 * np.cos(th), 1.40 + rng.normal(scale=0.05, size=n_cams), 1.2 * np.sin(th)]
    Rs, ts = [], []
    for i, a in enumerate(th):
        look = np.array([np.cos(a + 0.6 + rng.normal(scale=0.2)), 0, np.sin(a + 0.6 + rng.normal(scale=0.2))])
        yaw = np.arctan2(look[2], look[0])
        pitch = np.deg2rad(rng.uniform(-25, 5))
        roll = np.deg2rad(rng.normal(scale=2.0))
        # camera axes in world: z forward, x right, y down
        f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(pitch), np.sin(yaw) * np.cos(pitch)])
        right = np.cross(f, [0, 1.0, 0]); right /= np.linalg.norm(right)
        down = np.cross(f, right)
        right, down = rot_from_axis_angle(f, roll) @ right, rot_from_axis_angle(f, roll) @ down
        Rcw = np.stack([right, down, f])          # rows are camera axes in world coordinates: cam_from_world
        Rs.append(Rcw)
        ts.append(-Rcw @ C[i])
    world = {"R": np.array(Rs), "t": np.array(ts), "P": P}
    truth = {"up": np.array([0.0, 1.0, 0.0]), "floor_y": 0.0, "h_cap": float(np.median(C[:, 1])), "scale": 1.0}
    return world, truth


def write_model(path: str, R: np.ndarray, t: np.ndarray, P: np.ndarray, width=1280, height=720, f=900.0) -> None:
    """Binary COLMAP 3.x model: one PINHOLE camera, every point seen by 3 cameras with exact keypoints."""
    os.makedirs(path, exist_ok=True)
    n = len(R)
    with open(os.path.join(path, "cameras.bin"), "wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<IiQQ4d", 1, 1, width, height, f, f, width / 2, height / 2))
    # choose, per point, the 3 cameras that see it best (in front, nearest to the optical axis)
    obs = [[] for _ in range(n)]
    tracks = []
    for j, X in enumerate(P):
        xc = np.einsum("nij,j->ni", R, X) + t
        z = xc[:, 2]
        u = f * xc[:, 0] / np.where(z > 0.1, z, 1) + width / 2
        v = f * xc[:, 1] / np.where(z > 0.1, z, 1) + height / 2
        ok = (z > 0.1) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        cand = np.nonzero(ok)[0]
        tr = []
        for ci in cand[:3]:
            obs[ci].append((u[ci], v[ci], j))
            tr.append((ci, len(obs[ci]) - 1))
        tracks.append(tr)
    with open(os.path.join(path, "images.bin"), "wb") as fh:
        fh.write(struct.pack("<Q", n))
        for i in range(n):
            q = cio.rotmat_to_qvec(R[i])
            fh.write(struct.pack("<I4d3dI", i + 1, *q, *t[i], 1))
            fh.write(f"img_{i + 1:04d}.jpg".encode() + b"\0")
            fh.write(struct.pack("<Q", len(obs[i])))
            for (u, v, j) in obs[i]:
                fh.write(struct.pack("<ddQ", u, v, j + 1))
    with open(os.path.join(path, "points3D.bin"), "wb") as fh:
        fh.write(struct.pack("<Q", len(P)))
        for j, X in enumerate(P):
            fh.write(struct.pack("<Q3d3Bd", j + 1, *X, 128, 128, 128, 0.5))
            fh.write(struct.pack("<Q", len(tracks[j])))
            for ci, k in tracks[j]:
                fh.write(struct.pack("<II", ci + 1, k))


def apply_raw_frame(world: dict, Q: np.ndarray, s0: float, shift: np.ndarray):
    """Move the metric world into an arbitrary SfM frame: X_raw = (1/s0) Q X + shift."""
    P = (Q @ world["P"].T).T / s0 + shift
    R = np.array([Ri @ Q.T for Ri in world["R"]])           # cam_from_world after X_raw = Q X / s0 + shift
    C = -np.einsum("nji,nj->ni", world["R"], world["t"])
    Craw = (Q @ C.T).T / s0 + shift
    t = np.array([-R[i] @ Craw[i] for i in range(len(R))])
    return R, t, P
