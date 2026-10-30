"""Dev tool: compare HLG tone-map chains on b2 against the SDR original b1 at the same instants.

    nsdocker w2-sweep namma-space-gpu:0.1 python -m pipeline.ingest.tonemap_sweep /w/data/fixtures/videos

b2 is b1 encoded as HLG, so a good tone map brings b2 back near b1: same mean luma, same RMS contrast.
Prints mean luma, RMS contrast (std of luma) and the mean absolute error to b1's frame for each variant.
"""
from __future__ import annotations

import subprocess
import sys
from typing import List

import numpy as np

HEAD = "zscale=tin=arib-std-b67:pin=bt2020:min=bt2020nc:rin=limited:t=linear:npl={npl},format=gbrpf32le,zscale=p=bt709,"
TAIL = ",zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
VARIANTS = {
    "plan: npl100 hable desat0": HEAD.format(npl=100) + "tonemap=hable:desat=0" + TAIL,
    "npl100 hable desat0 peak1": HEAD.format(npl=100) + "tonemap=hable:desat=0:peak=1" + TAIL,
    "npl100 mobius desat0": HEAD.format(npl=100) + "tonemap=mobius:desat=0" + TAIL,
    "npl100 reinhard desat0": HEAD.format(npl=100) + "tonemap=reinhard:desat=0" + TAIL,
    "npl100 clip": HEAD.format(npl=100) + "tonemap=clip:desat=0" + TAIL,
    "npl203 hable desat0": HEAD.format(npl=203) + "tonemap=hable:desat=0" + TAIL,
    "npl1000 hable desat0": HEAD.format(npl=1000) + "tonemap=hable:desat=0" + TAIL,
    "npl1000 clip": HEAD.format(npl=1000) + "tonemap=clip:desat=0" + TAIL,
    "npl100 linear": HEAD.format(npl=100) + "tonemap=linear:desat=0" + TAIL,
}


def frames(path: str, vf: str, times: List[float]) -> np.ndarray:
    out = []
    for t in times:
        cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", "-vf", vf + ",scale=320:180,format=gray",
               "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        out.append(np.frombuffer(raw, np.uint8).reshape(180, 320).astype(float) if len(raw) == 320 * 180 else None)
    return np.array([o for o in out if o is not None])


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else "/w/data/fixtures/videos"
    times = [3.0, 9.5, 17.0, 24.5, 31.0, 38.5, 45.0, 52.5]
    ref = frames(f"{d}/b1.mp4", "null", times)
    print(f"b1 reference: mean luma {ref.mean():.1f}  rms contrast {ref.std(axis=(1, 2)).mean():.1f}")
    for name, vf in VARIANTS.items():
        f = frames(f"{d}/b2.mp4", vf, times)
        if len(f) != len(ref):
            print(f"{name:32s} failed")
            continue
        print(f"{name:32s} mean luma {f.mean():6.1f}  rms {f.std(axis=(1, 2)).mean():5.1f}  rms/b1 {f.std(axis=(1, 2)).mean() / ref.std(axis=(1, 2)).mean():.2f}"
              f"  MAE to b1 {np.abs(f - ref).mean():5.1f}")


if __name__ == "__main__":
    main()
