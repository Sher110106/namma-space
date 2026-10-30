"""Gaussian-splat PLY reader and writer (binary little endian, standard 3DGS layout).

Only the fields the grid stage needs are materialised: centres, opacity, scales, SH0 colour.
Raw 3DGS files store opacity as a logit and scales as logs; ``activations="auto"`` detects that.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
          "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
          "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


@dataclass
class Splats:
    xyz: np.ndarray        # (N, 3) float32, metres
    opacity: np.ndarray    # (N,) float32 in [0, 1]
    scale: np.ndarray      # (N, 3) float32, metres (linear)
    rgb: np.ndarray        # (N, 3) float32 in [0, 1] (SH0)
    activations: str = "raw3dgs"   # what auto-detection decided: "raw3dgs" or "linear"

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def subset(self, mask: np.ndarray) -> "Splats":
        return Splats(self.xyz[mask], self.opacity[mask], self.scale[mask], self.rgb[mask], self.activations)


def read_header(path: str | Path) -> tuple[int, np.dtype, int]:
    """Return (vertex count, numpy dtype of one vertex, byte offset of the data)."""
    with open(path, "rb") as f:
        head = f.read(65536)
    end = head.find(b"end_header")
    if end < 0:
        raise ValueError("no end_header in first 64 KiB")
    nl = head.find(b"\n", end)
    lines = head[:end].decode("ascii", "replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not a PLY file")
    fmt_line = next((ln for ln in lines if ln.startswith("format")), "")
    if "binary_little_endian" not in fmt_line:
        raise ValueError(f"unsupported PLY format: {fmt_line!r}")
    n, fields, in_vertex = 0, [], False
    for ln in lines:
        p = ln.split()
        if not p:
            continue
        if p[0] == "element":
            in_vertex = p[1] == "vertex"
            if in_vertex:
                n = int(p[2])
            elif fields:
                break
        elif p[0] == "property" and in_vertex:
            if p[1] == "list":
                raise ValueError("list property in vertex element")
            fields.append((p[2], _TYPES[p[1]]))
    return n, np.dtype(fields), nl + 1


def read_splats(path: str | Path, activations: str = "auto") -> Splats:
    n, dt, off = read_header(path)
    names = dt.names or ()
    mm = np.memmap(path, dtype=dt, mode="r", offset=off, shape=(n,))
    xyz = np.stack([np.asarray(mm["x"]), np.asarray(mm["y"]), np.asarray(mm["z"])], axis=1).astype(np.float32)
    if "opacity" in names:
        op = np.asarray(mm["opacity"], dtype=np.float32)
    else:
        op = np.ones(n, dtype=np.float32)
    if all(f"scale_{k}" in names for k in range(3)):
        sc = np.stack([np.asarray(mm[f"scale_{k}"]) for k in range(3)], axis=1).astype(np.float32)
    else:
        sc = np.full((n, 3), 0.02, dtype=np.float32)
    if activations == "auto":
        raw = bool(n) and (float(op.min()) < 0.0 or float(op.max()) > 1.0 or float(sc.min()) < 0.0)
        activations = "raw3dgs" if raw else "linear"
    if activations == "raw3dgs":
        op = (1.0 / (1.0 + np.exp(-op.astype(np.float64)))).astype(np.float32)
        sc = np.exp(sc.astype(np.float64)).astype(np.float32)
    if "f_dc_0" in names:
        rgb = 0.5 + SH_C0 * np.stack([np.asarray(mm[f"f_dc_{k}"]) for k in range(3)], axis=1)
    elif "red" in names:
        rgb = np.stack([np.asarray(mm[k]) for k in ("red", "green", "blue")], axis=1) / 255.0
    else:
        rgb = np.full((n, 3), 0.5)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    del mm
    return Splats(xyz, op, sc, rgb, activations)


def write_ply(path: str | Path, xyz: np.ndarray, opacity: np.ndarray, scale: np.ndarray, rgb: np.ndarray,
              sh_rest: int = 0, raw3dgs: bool = True) -> None:
    """Write a 3DGS-layout PLY. ``opacity`` and ``scale`` are linear; they are stored as logit / log when raw3dgs."""
    n = len(xyz)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{k}" for k in range(sh_rest)] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dt = np.dtype([(p, "<f4") for p in props])
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    dc = (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0
    for k in range(3):
        arr[f"f_dc_{k}"] = dc[:, k]
    op = np.clip(np.asarray(opacity, dtype=np.float64), 1e-6, 1 - 1e-6)
    sc = np.asarray(scale, dtype=np.float64)
    if raw3dgs:
        op = np.log(op / (1 - op))
        sc = np.log(np.maximum(sc, 1e-8))
    arr["opacity"] = op
    for k in range(3):
        arr[f"scale_{k}"] = sc[:, k]
    arr["rot_0"] = 1.0
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
             "".join(f"property float {p}\n" for p in props) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
