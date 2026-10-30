import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:      # the Mac system Python has no numpy; use pipeline/export/.venv/bin/python
    HAVE_NUMPY = False


def make_splats(n=1000, seed=0, sh_rest=45, extent=2.0):
    import numpy as np
    from pipeline.export.ply_io import Splats
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Splats(
        means=(rng.uniform(-extent, extent, size=(n, 3))).astype(np.float32),
        f_dc=rng.normal(0, 0.8, size=(n, 3)).astype(np.float32),
        f_rest=rng.normal(0, 0.05, size=(n, sh_rest)).astype(np.float32),
        opacity=rng.normal(1.0, 2.0, size=n).astype(np.float32),
        scale=rng.normal(-4.0, 0.7, size=(n, 3)).astype(np.float32),
        quat=q)
