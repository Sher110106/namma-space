"""Raster helpers on boolean grids. Row i is z, column j is x (see CONTRACTS 2.12)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def dilate(mask: np.ndarray, r_cells: float) -> np.ndarray:
    """Cells within r_cells (Euclidean, centre to centre) of any True cell."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return ndi.distance_transform_edt(~mask) <= r_cells


def close(mask: np.ndarray, r_cells: float) -> np.ndarray:
    """Morphological closing with a disc; array borders do not erode."""
    if not mask.any():
        return mask.copy()
    d = dilate(mask, r_cells)
    if d.all():
        return d
    return ndi.distance_transform_edt(d) > r_cells


def dist_to(mask: np.ndarray, cell: float) -> np.ndarray:
    """Distance in metres from each cell centre to the nearest True cell centre."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndi.distance_transform_edt(~mask) * cell


def stamp(ii: np.ndarray, jj: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[ii, jj] = True
    return m


def outline(mask: np.ndarray) -> np.ndarray:
    """Boundary cells of a mask (cells of the mask with a 4-neighbour outside it)."""
    er = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(2, 1), border_value=0)
    return mask & ~er


def label4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndi.label(mask, structure=ndi.generate_binary_structure(2, 1))


def label8(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndi.label(mask, structure=np.ones((3, 3), dtype=int))
