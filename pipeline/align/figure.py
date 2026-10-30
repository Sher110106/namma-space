"""Level before/after figure: side and top views of points and camera path, raw frame above, aligned frame below."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _subsample(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]


def render(path: str, pts_before: np.ndarray, cam_before: np.ndarray, pts_after: np.ndarray, cam_after: np.ndarray,
           up_before: np.ndarray, floor_normal_before: Optional[np.ndarray], floor_off_before: Optional[float],
           info: Dict[str, Any], title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pb, pa = _subsample(pts_before, 30000, rng), _subsample(pts_after, 30000, rng)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), dpi=130)

    def fit(ax_, P, C, a, b, floor_zero=False):
        """Frame the points between their 2nd and 98th percentiles (SfM leaves far stray points) and keep the aspect equal."""
        allp = np.vstack([P[:, [a, b]], C[:, [a, b]]])
        lo, hi = np.percentile(allp, 2, axis=0), np.percentile(allp, 98, axis=0)
        pad = 0.15 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        if floor_zero:
            lo[1] = min(lo[1], -0.3)
        ax_.set_xlim(lo[0], hi[0])
        ax_.set_ylim(lo[1], hi[1])
        ax_.set_aspect("equal", adjustable="box")

    def lims(P, C, a, b):
        allp = np.vstack([P[:, [a, b]], C[:, [a, b]]])
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
        pad = 0.08 * (hi - lo + 1e-9)
        return lo - pad, hi + pad

    # before: view along the raw axis that is most horizontal, so tilt shows as a slope
    k = int(np.argmin(np.abs(up_before)))
    a, b = [i for i in range(3) if i != k]
    names = "XYZ"
    ax0 = ax[0, 0]
    ax0.scatter(pb[:, a], pb[:, b], s=0.25, c="#6b7280", alpha=0.5, linewidths=0)
    ax0.plot(cam_before[:, a], cam_before[:, b], "-", c="#2563eb", lw=0.8)
    ax0.scatter(cam_before[:, a], cam_before[:, b], s=4, c="#2563eb", zorder=3)
    if floor_normal_before is not None:
        n = floor_normal_before
        allp = np.vstack([pb[:, [a, b]], cam_before[:, [a, b]]])
        xs = np.array([np.percentile(allp[:, 0], 0.5) - 1.0, np.percentile(allp[:, 0], 99.5) + 1.0])
        # plane n.x = off, other axis k held at the data median; solve for coordinate b
        kk = float(np.median(pts_before[:, k]))
        if abs(n[b]) > 1e-6:
            ys = (floor_off_before - n[a] * xs - n[k] * kk) / n[b]
            ax0.plot(xs, ys, "-", c="#dc2626", lw=1.4, label="floor plane")
    u2 = np.array([up_before[a], up_before[b]])
    c0 = np.median(cam_before[:, [a, b]], axis=0)
    ext = float(np.ptp(pb[:, a])) * 0.15
    ax0.annotate("", xy=c0 + ext * u2 / (np.linalg.norm(u2) + 1e-9), xytext=c0,
                 arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2))
    ax0.set_xlabel(f"raw {names[a]}")
    ax0.set_ylabel(f"raw {names[b]}")
    ax0.set_title(f"before: raw SfM frame, side view along raw {names[k]}\n(red = floor plane; the raw frame is arbitrary, here its up is -Y)", fontsize=9)
    fit(ax0, pb, cam_before, a, b)
    ax0.legend(loc="lower right", fontsize=7)

    # before, top: along estimated up is not available in raw coordinates; show the two raw axes with the smallest up component
    ax1 = ax[0, 1]
    o = np.argsort(np.abs(up_before))[:2]
    ax1.scatter(pb[:, o[0]], pb[:, o[1]], s=0.25, c="#6b7280", alpha=0.5, linewidths=0)
    ax1.plot(cam_before[:, o[0]], cam_before[:, o[1]], "-", c="#2563eb", lw=0.8)
    ax1.set_xlabel(f"raw {names[o[0]]}")
    ax1.set_ylabel(f"raw {names[o[1]]}")
    ax1.set_title(f"before: raw axes {names[o[0]]}{names[o[1]]} (the up direction is mostly along {names[int(np.argmax(np.abs(up_before)))]})", fontsize=9)
    fit(ax1, pb, cam_before, o[0], o[1])

    # after: metres, +Y up, floor at y = 0
    ax2 = ax[1, 0]
    ax2.scatter(pa[:, 0], pa[:, 1], s=0.25, c="#6b7280", alpha=0.5, linewidths=0)
    ax2.plot(cam_after[:, 0], cam_after[:, 1], "-", c="#2563eb", lw=0.8)
    ax2.scatter(cam_after[:, 0], cam_after[:, 1], s=4, c="#2563eb", zorder=3)
    ax2.axhline(0.0, c="#dc2626", lw=1.4, label="floor y = 0")
    ax2.axhline(info.get("h_cap_m", np.nan), c="#16a34a", lw=1.0, ls="--", label=f"H_cap = {info.get('h_cap_m', float('nan')):.2f} m")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m, up)")
    ax2.set_title("after: side view (X, Y), floor level at y = 0", fontsize=9)
    fit(ax2, pa, cam_after, 0, 1, floor_zero=True)
    ax2.legend(loc="upper right", fontsize=7)

    ax3 = ax[1, 1]
    ax3.scatter(pa[:, 0], pa[:, 2], s=0.25, c="#6b7280", alpha=0.5, linewidths=0)
    ax3.plot(cam_after[:, 0], cam_after[:, 2], "-", c="#2563eb", lw=0.8)
    ax3.scatter([0], [0], s=40, c="#dc2626", marker="x", zorder=4, label="origin")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("after: top view (X, Z)", fontsize=9)
    fit(ax3, pa, cam_after, 0, 2)
    ax3.legend(loc="upper right", fontsize=7)

    txt = (f"up residual {info.get('up_resid_deg', float('nan')):.2f} deg | cue spread {info.get('cue_spread_deg', float('nan')):.2f} deg | "
           f"scale {info.get('scale', float('nan')):.4g} m per unit (68%: +/-{info.get('pct_68', float('nan')):.0f}%)")
    fig.suptitle((title + "\n" if title else "") + txt, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
