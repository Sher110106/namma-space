"""Similarity fit (Horn's absolute orientation with scale) between two sets of 3D points. Stdlib only.

fit(src, dst) -> (s, R, t) minimising sum |s R src_i + t - dst_i|^2. The rotation comes from the dominant
eigenvector of Horn's 4x4 matrix, found with Jacobi sweeps (no numpy in namma-colmap).
"""
from __future__ import annotations

import math

Vec = tuple[float, float, float]
Mat3 = list[list[float]]


def _jacobi_eigh(a: list[list[float]], sweeps: int = 60) -> tuple[list[float], list[list[float]]]:
    n = len(a)
    a = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-24:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-30:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


def _quat_to_mat(q: tuple[float, float, float, float]) -> Mat3:
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def apply(s: float, r: Mat3, t: Vec, p: Vec) -> Vec:
    return tuple(s * sum(r[i][j] * p[j] for j in range(3)) + t[i] for i in range(3))  # type: ignore[return-value]


def fit(src: list[Vec], dst: list[Vec]) -> tuple[float, Mat3, Vec]:
    n = len(src)
    if n < 3 or n != len(dst):
        raise ValueError("need at least 3 matching points")
    cs = [sum(p[i] for p in src) / n for i in range(3)]
    cd = [sum(p[i] for p in dst) / n for i in range(3)]
    xs = [tuple(p[i] - cs[i] for i in range(3)) for p in src]
    ys = [tuple(p[i] - cd[i] for i in range(3)) for p in dst]
    sxx = [[sum(x[i] * y[j] for x, y in zip(xs, ys)) for j in range(3)] for i in range(3)]
    n4 = [[sxx[0][0] + sxx[1][1] + sxx[2][2], sxx[1][2] - sxx[2][1], sxx[2][0] - sxx[0][2], sxx[0][1] - sxx[1][0]],
          [sxx[1][2] - sxx[2][1], sxx[0][0] - sxx[1][1] - sxx[2][2], sxx[0][1] + sxx[1][0], sxx[2][0] + sxx[0][2]],
          [sxx[2][0] - sxx[0][2], sxx[0][1] + sxx[1][0], -sxx[0][0] + sxx[1][1] - sxx[2][2], sxx[1][2] + sxx[2][1]],
          [sxx[0][1] - sxx[1][0], sxx[2][0] + sxx[0][2], sxx[1][2] + sxx[2][1], -sxx[0][0] - sxx[1][1] + sxx[2][2]]]
    vals, vecs = _jacobi_eigh(n4)
    k = max(range(4), key=lambda i: vals[i])
    q = tuple(vecs[r][k] for r in range(4))
    r = _quat_to_mat(q)  # type: ignore[arg-type]
    # scale: s = sum(y . R x) / sum(|x|^2)
    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        rx = [sum(r[i][j] * x[j] for j in range(3)) for i in range(3)]
        num += sum(y[i] * rx[i] for i in range(3))
        den += sum(x[i] * x[i] for i in range(3))
    s = num / den if den > 0 else 1.0
    t = tuple(cd[i] - s * sum(r[i][j] * cs[j] for j in range(3)) for i in range(3))
    return s, r, t  # type: ignore[return-value]


def rmse(s: float, r: Mat3, t: Vec, src: list[Vec], dst: list[Vec]) -> float:
    err = 0.0
    for p, q in zip(src, dst):
        a = apply(s, r, t, p)
        err += sum((a[i] - q[i]) ** 2 for i in range(3))
    return math.sqrt(err / len(src))
