"""Binary little-endian Gaussian-splat PLY reader/writer (numpy only).

Layout written by gsplat's export_splats (also the Inria layout):
  x y z, [nx ny nz], f_dc_0..2, f_rest_0..K-1 (channel-major: R's coefficients, then G, then B),
  opacity (logit), scale_0..2 (log), rot_0..3 (quaternion w x y z, not normalised)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
          "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
          "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
          "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_header(path: str) -> Tuple[int, List[Tuple[str, str]], int]:
    """Return (vertex count, [(name, numpy dtype str)], byte offset of the data)."""
    props: List[Tuple[str, str]] = []
    n = 0
    in_vertex = False
    with open(path, "rb") as f:
        first = f.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header ended early")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format" and tok[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY is supported, got {tok[1]}")
            if tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError("list properties are not supported in vertex element")
                props.append((tok[2], _TYPES[tok[1]]))
            elif tok[0] == "end_header":
                return n, props, f.tell()


@dataclass
class Splats:
    """Gaussian splats as float32 arrays (PLY parameterisation: logit opacity, log scale)."""
    means: np.ndarray        # (N, 3)
    f_dc: np.ndarray         # (N, 3)
    f_rest: np.ndarray       # (N, 3*K) channel-major; K = 0, 3, 8 or 15
    opacity: np.ndarray      # (N,) logit
    scale: np.ndarray        # (N, 3) log
    quat: np.ndarray         # (N, 4) w x y z

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_rest_per_channel(self) -> int:
        return int(self.f_rest.shape[1] // 3)

    @property
    def sh_degree(self) -> int:
        k = self.sh_rest_per_channel
        return {0: 0, 3: 1, 8: 2, 15: 3}[k]

    def take(self, sel) -> "Splats":
        return Splats(self.means[sel], self.f_dc[sel], self.f_rest[sel], self.opacity[sel],
                      self.scale[sel], self.quat[sel])

    def sh_rest_cube(self) -> np.ndarray:
        """(N, K, 3) rest coefficients, as gsplat's shN."""
        n, k = len(self), self.sh_rest_per_channel
        return self.f_rest.reshape(n, 3, k).transpose(0, 2, 1)

    def limited_to_degree(self, degree: int) -> "Splats":
        """Keep SH bands up to `degree` (0..3)."""
        keep_k = {0: 0, 1: 3, 2: 8, 3: 15}[degree]
        k = self.sh_rest_per_channel
        if keep_k >= k:
            return self
        rest = self.f_rest.reshape(len(self), 3, k)[:, :, :keep_k].reshape(len(self), 3 * keep_k)
        return Splats(self.means, self.f_dc, np.ascontiguousarray(rest), self.opacity, self.scale, self.quat)


def read_ply(path: str) -> Splats:
    n, props, off = read_header(path)
    names = [p[0] for p in props]
    if not all(p[1] == "<f4" for p in props):
        dt = np.dtype([(nm, t) for nm, t in props])
        raw = np.fromfile(path, dtype=dt, count=n, offset=off)
        a = np.stack([raw[nm].astype(np.float32) for nm in names], axis=1)
    else:
        a = np.fromfile(path, dtype="<f4", count=n * len(props), offset=off).reshape(n, len(props))

    def cols(prefix: str) -> List[int]:
        idx = []
        i = 0
        while f"{prefix}{i}" in names:
            idx.append(names.index(f"{prefix}{i}"))
            i += 1
        return idx

    def col(nm: str) -> int:
        return names.index(nm)

    rest = cols("f_rest_")
    return Splats(
        means=np.ascontiguousarray(a[:, [col("x"), col("y"), col("z")]]),
        f_dc=np.ascontiguousarray(a[:, cols("f_dc_")]),
        f_rest=np.ascontiguousarray(a[:, rest]) if rest else np.zeros((n, 0), np.float32),
        opacity=np.ascontiguousarray(a[:, col("opacity")]),
        scale=np.ascontiguousarray(a[:, cols("scale_")]),
        quat=np.ascontiguousarray(a[:, cols("rot_")]),
    )


def write_ply(path: str, s: Splats) -> None:
    n = len(s)
    k = s.f_rest.shape[1]
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
           "property float x", "property float y", "property float z"]
    hdr += [f"property float f_dc_{i}" for i in range(3)]
    hdr += [f"property float f_rest_{i}" for i in range(k)]
    hdr += ["property float opacity"]
    hdr += [f"property float scale_{i}" for i in range(3)]
    hdr += [f"property float rot_{i}" for i in range(4)]
    hdr += ["end_header"]
    data = np.concatenate([s.means, s.f_dc, s.f_rest, s.opacity[:, None], s.scale, s.quat], axis=1)
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data.astype("<f4")).tobytes())


def count_vertices(path: str) -> int:
    return read_header(path)[0]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
