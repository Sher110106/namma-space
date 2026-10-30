"""A phone-like test video from a folder of real photos (integrator, 30 Oct).

    python -m pipeline.ingest.make_phone_fixture --frames DIR --out FILE.mov [--hold 0.5] [--fps 30]

The fixtures b1-b7 are crop pans over still photos: good for ingest tests, but SfM sees no real camera motion
(b7 registered 28-36%). This one keeps each real photo whole, in capture order, so SfM sees the real motion, and
packs it the way an iPhone stores a portrait HDR clip:
- HEVC Main10 (libx265), tag hvc1, BT.2020 primaries, HLG transfer (arib-std-b67), limited range;
- stored pixels turned 90 degrees counter-clockwise plus a display matrix "rotate 90 clockwise" in tkhd, so players and
  our ingest must rotate it back to portrait;
- 30 fps, each photo held for --hold seconds (duplicated frames), so the preflight's frame-rate and duration checks
  see a normal clip.
SDR photos are mapped to HLG the same way as fixture b2 (BT.1886 to linear, BT.709 to BT.2020, SDR white at 0.265
scene light). Runs in namma-space-gpu (ffmpeg with libx265, numpy, OpenCV).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .make_fixtures import _hlg_oetf, patch_rotation


def main(argv=None) -> int:
    import cv2
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", type=float, default=0.5, help="seconds each photo is shown")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)
    names = sorted(n for n in os.listdir(a.frames) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise SystemExit(f"no images in {a.frames}")
    first = cv2.imread(os.path.join(a.frames, names[0]))
    h, w = first.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    sw, sh = h2, w2                      # stored size after the 90 deg turn
    m709_to_2020 = np.array([[0.627404, 0.329283, 0.043313],
                             [0.069097, 0.919540, 0.011362],
                             [0.016391, 0.088013, 0.895595]], dtype=np.float32)
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** 2.4
    out = Path(a.out)
    rate = 1.0 / a.hold
    enc_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{sw}x{sh}",
               "-framerate", f"{rate:g}", "-i", "-", "-r", str(a.fps),
               "-vf", "scale=out_color_matrix=bt2020nc:out_range=tv,format=yuv420p10le",
               "-c:v", "libx265", "-preset", "fast", "-crf", "18",
               "-x265-params", "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited",
               "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_range", "tv",
               "-tag:v", "hvc1", "-movflags", "+faststart", str(out)]
    print("+", " ".join(enc_cmd), flush=True)
    t0 = time.time()
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    for n in names:
        bgr = cv2.imread(os.path.join(a.frames, n))[:h2, :w2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)        # stored landscape, shown portrait
        lin = cv2.transform(lut[rgb], m709_to_2020)
        sig = _hlg_oetf(lin * 0.265)
        enc.stdin.write((np.clip(sig, 0, 1) * 65535.0 + 0.5).astype("<u2").tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("encode failed")
    patch_rotation(out, 90)
    print(f"wrote {out}: {len(names)} photos x {a.hold} s at {a.fps} fps, stored {sw}x{sh}, shown {w2}x{h2}, "
          f"HEVC 10-bit HLG, rotate 90; {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
