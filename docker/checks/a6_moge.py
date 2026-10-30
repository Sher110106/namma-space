#!/usr/bin/env python3
"""A6: import moge and load Ruicheng/moge-2-vitl-normal from the host cache, offline, then run one image.

Run inside namma-space-gpu:0.2 with HF_HOME=/w/cache/hf (image default) and HF_HUB_OFFLINE=1.
usage: a6_moge.py [image.jpg]   (needs a GPU for the inference part; falls back to CPU)
"""
import os
import sys
import time

import numpy as np
import torch


def main() -> int:
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    print("HF_HOME", os.environ.get("HF_HOME"), "HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE"))
    from moge.model.v2 import MoGeModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    print(f"loaded in {time.time() - t0:.1f}s on {dev}; offline={offline}")
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path:
        import cv2

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        t = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
    else:
        t = torch.rand(3, 480, 640, device=dev)
    with torch.no_grad():
        out = model.infer(t)
    for k, v in out.items():
        print(k, tuple(v.shape), str(v.dtype))
    depth = out["depth"][torch.isfinite(out["depth"])]
    print("depth median (m, metric scale)", float(depth.median()))
    print("intrinsics", out["intrinsics"].cpu().numpy().round(3).tolist())
    print("A6 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
