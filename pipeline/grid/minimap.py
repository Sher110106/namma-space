"""Top-down minimap: SH0 colours of splats below 2 m, top surface per pixel, plus the walkable outline."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import raster
from .plyio import Splats
from .walkable import Layout

OUTLINE_RGB = (0, 220, 130)
BACKGROUND_RGB = (28, 30, 34)


def choose_px_per_m(layout: Layout, max_side: int = 4096) -> int:
    """40 px/m (2 px per 5 cm cell) when it fits, else 20 px/m (1 px per cell). Must divide into whole cells."""
    for ppm in (40, 20):
        f = ppm * layout.cell
        if abs(f - round(f)) < 1e-9 and max(layout.w, layout.h) * round(f) <= max_side:
            return ppm
    return 20


def render(splats: Splats, layout: Layout, walkable: np.ndarray, px_per_m: int, floor_y: float = 0.0,
           ceiling: float = 2.0, slab: float = 0.08, outline: bool = True) -> np.ndarray:
    """RGB uint8 (H*k, W*k, 3). Pixel (i, j) is the square centred at x = ox + (j + .5)/ppm, z = oz + (i + .5)/ppm."""
    k = int(round(px_per_m * layout.cell))
    H, W = layout.h * k, layout.w * k
    y = splats.xyz[:, 1]
    sel = (y <= floor_y + ceiling) & (y >= floor_y - 0.3) & (splats.opacity > 0.3)
    xyz, rgb = splats.xyz[sel], splats.rgb[sel]
    i = np.floor((xyz[:, 2] - layout.oz) * px_per_m).astype(np.int64)
    j = np.floor((xyz[:, 0] - layout.ox) * px_per_m).astype(np.int64)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    i, j, yy, rgb = i[ok], j[ok], xyz[ok, 1], rgb[ok]
    img = np.zeros((H, W, 3), dtype=np.float64)
    have = np.zeros((H, W), dtype=bool)
    if len(i):
        pix = i * W + j
        order = np.lexsort((yy, pix))
        pix, yy, rgb = pix[order], yy[order], rgb[order]
        starts = np.flatnonzero(np.r_[True, pix[1:] != pix[:-1]])
        ends = np.r_[starts[1:], len(pix)]
        cnt = ends - starts
        yref = yy[np.maximum(ends - 2, starts)]       # second highest splat: one stray splat cannot set the surface
        gid = np.repeat(np.arange(len(starts)), cnt)
        inslab = yy >= np.repeat(yref, cnt) - slab
        wsum = np.bincount(gid, weights=inslab.astype(np.float64), minlength=len(starts))
        col = np.stack([np.bincount(gid, weights=rgb[:, c] * inslab, minlength=len(starts)) for c in range(3)], axis=1)
        col /= np.maximum(wsum, 1.0)[:, None]
        p = pix[starts]
        img.reshape(-1, 3)[p] = col * 255.0
        have.reshape(-1)[p] = True
    # fill small gaps from the nearest painted pixel (up to 6 px), the rest is background
    if have.any() and not have.all():
        d, (ri, rj) = ndi.distance_transform_edt(~have, return_indices=True)
        fill = (~have) & (d <= 6.0)
        img[fill] = img[ri[fill], rj[fill]]
        have = have | fill
    out = np.empty((H, W, 3), dtype=np.uint8)
    out[:] = BACKGROUND_RGB
    out[have] = np.clip(np.rint(img[have]), 0, 255).astype(np.uint8)
    if outline:
        up = np.kron(walkable.astype(np.uint8), np.ones((k, k), dtype=np.uint8)).astype(bool)
        edge = raster.outline(up)
        edge = ndi.binary_dilation(edge, structure=np.ones((2, 2), dtype=bool)) & up if k >= 2 else edge
        out[edge] = OUTLINE_RGB
    return out
