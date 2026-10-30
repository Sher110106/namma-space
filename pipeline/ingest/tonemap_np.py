"""HDR to SDR in numpy, for ffmpeg builds without zscale/tonemap.

Input: float32 RGB in [0, 1], BT.2020 primaries, HLG or PQ non-linear signal (what ffmpeg's
swscale gives after YUV to RGB with the bt2020 matrix).  Output: float32 RGB in [0, 1], BT.709
primaries, gamma-encoded.  The scene is anchored so that HLG diffuse white (signal 0.75) and PQ 203 nits
land at SDR reference white; below a knee the curve is linear, above it a smooth shoulder rolls the
highlights off towards 1.  This tracks the zscale + tonemap=reinhard chain the ingest uses by default
(both bring the HLG version of b1 back near b1; see NOTES.md).
"""
from __future__ import annotations

import numpy as np

_A, _B, _C = 0.17883277, 0.28466892, 0.55991073
_M2020_TO_709 = np.array([[1.660491, -0.587641, -0.072850],
                          [-0.124550, 1.132900, -0.008349],
                          [-0.018151, -0.100579, 1.118730]], dtype=np.float32)


def _hlg_inv_oetf(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return np.where(s <= 0.5, (s * s) / 3.0, (np.exp((s - _C) / _A) + _B) / 12.0)


def _pq_eotf(s: np.ndarray) -> np.ndarray:
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    p = np.clip(s, 0.0, 1.0) ** (1.0 / m2)
    return 10000.0 * (np.maximum(p - c1, 0.0) / (c2 - c3 * p)) ** (1.0 / m1)


def _shoulder(x: np.ndarray, knee: float = 0.6) -> np.ndarray:
    """Identity up to `knee`, then an exponential shoulder with unit slope at the knee, approaching 1."""
    k = 1.0 - knee
    return np.where(x <= knee, x, knee + k * (1.0 - np.exp(-(x - knee) / k)))


def tonemap(rgb: np.ndarray, kind: str = "hlg") -> np.ndarray:
    """kind 'hlg' or 'pq'.  Returns gamma-encoded BT.709 RGB in [0, 1]."""
    if kind == "pq":
        lin = _pq_eotf(rgb) / 203.0          # 1.0 = diffuse white
    else:
        lin = _hlg_inv_oetf(rgb) / 0.265     # scene light 0.265 = diffuse white = 1.0
    lin = np.einsum("hwc,dc->hwd", lin, _M2020_TO_709).astype(np.float32)
    out = np.clip(_shoulder(np.maximum(lin, 0.0)), 0.0, 1.0)
    return np.where(out <= 0.0031308, 12.92 * out, 1.055 * np.power(out, 1 / 2.4) - 0.055).astype(np.float32)
