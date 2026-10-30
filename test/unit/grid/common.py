"""Shared helpers for the grid tests."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline.grid import pngio  # noqa: E402
from pipeline.grid.stage import run_stage  # noqa: E402


class Grid:
    """A finished 70_grid directory: arrays and the pixel <-> world mapping of CONTRACTS 2.12."""

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.j = json.loads((self.dir / "grid.json").read_text())
        self.png = pngio.read_png(self.dir / "grid.png")
        self.mini = pngio.read_png(self.dir / "minimap.png")
        self.ox, self.oz = self.j["origin_xz"]
        self.c = self.j["cell"]

    @property
    def walkable(self) -> np.ndarray:
        return self.png == 255

    def centres(self, mask: np.ndarray):
        i, j = np.nonzero(mask)
        return self.ox + (j + 0.5) * self.c, self.oz + (i + 0.5) * self.c

    def pix(self, x, z):
        return int(np.floor((z - self.oz) / self.c)), int(np.floor((x - self.ox) / self.c))


def run_and_load(run: Path, **kw) -> tuple[int, Grid]:
    rc = run_stage(str(run), quiet=True, **kw)
    return rc, Grid(run / "70_grid")


def tmpdir(prefix: str = "w4t_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
