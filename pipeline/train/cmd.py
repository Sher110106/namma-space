"""Build the gsplat simple_trainer command line (FINAL_PLAN 2.5).

Flag spellings confirmed with `simple_trainer.py mcmc -h` (tyro, 30 Oct 2026):
--no-normalize-world-space, --antialiased, --use-bilateral-grid, --depth-loss,
--strategy.cap-max, --disable-viewer, --disable-video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

TRAINER = "/opt/gsplat/examples/simple_trainer.py"
TRAINER_CWD = "/opt/gsplat/examples"
PYTHON = "python"

VARIANTS = ("r0", "r1")
MCMC_NOISE_LR = 5e5     # gsplat default of --strategy.noise-lr
SCALE_REG = 0.01        # gsplat default of --scale-reg in the mcmc preset


def scene_unit(centres) -> float:
    """k: median distance of the camera centres from their centroid (numpy array (N, 3))."""
    import numpy as np
    c = np.asarray(centres, dtype=np.float64)
    return float(np.median(np.linalg.norm(c - c.mean(axis=0), axis=1)))


@dataclass
class TrainOpts:
    data_dir: str
    result_dir: str
    variant: str = "r0"          # r0 | r1 (= r0 + depth loss) | r0nb (= r0 without bilateral grid)
    quick: bool = False          # 7k steps instead of 30k
    normalize: bool = False      # gsplat world normalisation; OFF in the pipeline
    cap_max: int = 1_500_000
    data_factor: int = 1
    test_every: int = 8
    # Scene unit k (median camera distance from the camera centroid, in model units). gsplat's MCMC
    # position noise grows as length^3 (covariance x lr x noise_lr, with lr proportional to scene
    # scale), and scale_reg has units of length. World normalisation hides this by scaling every scene
    # to k = 1. With normalisation off, noise_lr / k^2 and scale_reg / k restore the same behaviour.
    # None = leave the defaults (normalised scenes).
    scene_k: Optional[float] = None
    extra: List[str] = field(default_factory=list)

    @property
    def max_steps(self) -> int:
        return 7000 if self.quick else 30000

    @property
    def eval_steps(self) -> List[int]:
        return [7000] if self.quick else [7000, 30000]

    @property
    def final_ply_name(self) -> str:
        return f"point_cloud_{self.max_steps - 1}.ply"

    @property
    def preview_ply_name(self) -> str:
        return "point_cloud_6999.ply"


def build_command(o: TrainOpts) -> List[str]:
    if o.variant not in ("r0", "r1", "r0nb"):
        raise ValueError(f"unknown variant {o.variant!r}")
    steps = [str(s) for s in o.eval_steps]
    cmd = [
        PYTHON, TRAINER, "mcmc",
        "--data-dir", o.data_dir,
        "--data-factor", str(o.data_factor),
        "--result-dir", o.result_dir,
        "--normalize-world-space" if o.normalize else "--no-normalize-world-space",
        "--strategy.cap-max", str(o.cap_max),
        "--max-steps", str(o.max_steps),
        "--antialiased",
        "--test-every", str(o.test_every),
        "--eval-steps", *steps,
        "--save-steps", *steps,
        "--save-ply",
        "--ply-steps", *steps,
        "--disable-viewer",
        "--disable-video",
    ]
    if o.scene_k is not None:
        cmd += ["--strategy.noise-lr", f"{MCMC_NOISE_LR / o.scene_k ** 2:.6g}",
                "--scale-reg", f"{SCALE_REG / o.scene_k:.6g}"]
    if o.variant != "r0nb":
        cmd.append("--use-bilateral-grid")
    if o.variant == "r1":
        cmd.append("--depth-loss")
    cmd.extend(o.extra)
    return cmd
