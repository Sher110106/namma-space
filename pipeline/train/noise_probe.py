"""Evidence for NOTES finding 1: gsplat MCMC position noise is not scale-invariant (needs a GPU and gsplat).

Moves nearly transparent splats (the only ones the noise touches) with gsplat's inject_noise_to_position in a scene
scaled by k, with the means learning rate scaled by k as the trainer does, and prints the displacement relative to
scene size against k = 1. Measured 30 Oct 2026 on the A6000: k=2.8 -> 7.84x, k=8.27 -> 68.39x; with noise_lr/k^2 -> 1.00x.
Run: python -m pipeline.train.noise_probe
"""
import math, torch
from gsplat.strategy.ops import inject_noise_to_position
torch.manual_seed(0)
N = 20000
means0 = torch.randn(N, 3, device="cuda")
quats = torch.nn.functional.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
log_scales0 = torch.log(torch.rand(N, 3, device="cuda") * 0.05 + 0.01)
opac = torch.full((N,), 3.0, device="cuda")      # logit; sigmoid 0.95 -> 1 - 0.95 = 0.05 -> op_sigmoid ~ 0
opac[: N // 2] = -6.0                              # opacity 0.0025: the only splats the noise moves
def displacement(k, noise_lr_scale):
    params = {"means": torch.nn.Parameter(means0 * k), "scales": torch.nn.Parameter(log_scales0 + math.log(k)),
              "quats": torch.nn.Parameter(quats.clone()), "opacities": torch.nn.Parameter(opac.clone())}
    lr = 1.6e-4 * 2.0 * k            # means_lr * scene_scale; scene_scale grows with k
    before = params["means"].detach().clone()
    torch.manual_seed(1)
    inject_noise_to_position(params, {}, {}, scaler=lr * 5e5 * noise_lr_scale)
    d = (params["means"].detach() - before).norm(dim=1)
    return float(d.mean() / k)         # displacement relative to scene size
base = displacement(1.0, 1.0)
for k in (1.0, 2.8, 8.27):
    print(f"k={k}: relative displacement {displacement(k, 1.0)/base:.2f}x the k=1 value (unscaled); "
          f"{displacement(k, 1.0 / k**2)/base:.2f}x with noise_lr/k^2")
