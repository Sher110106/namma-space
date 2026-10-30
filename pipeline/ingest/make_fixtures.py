"""Build the ingest/check test videos b1-b7 from MuSHRoom honka frames.

Runs inside namma-space-gpu:0.1 (ffmpeg 4.4.2 with libx264/libx265, numpy, OpenCV):

    /home/sher/namma-space/bin/nsdocker w2-fixtures namma-space-gpu:0.1 \
        python -m pipeline.ingest.make_fixtures --out /w/data/fixtures/videos

Every video comes from the same 335 honka frames (portrait 738x994, CC BY 4.0),
held at the source rate and cropped with a slow vertical pan, so the content has
real texture and the frames differ from window to window.  Each video gets a
sidecar <name>.json with what was done to it (blur segments, tags, rotation),
which the tests read.

  b1  60 s, 1920x1080, H.264, SDR bt709, 30 fps
  b2  b1 as 10-bit HEVC, BT.2100 HLG, BT.2020 (encoded in numpy from b1)
  b3  portrait: stored 1920x1080 with rotate=90, shows upright 1080x1920
  b4  b1 with 10 blurred segments of 2 s (20 s in all)
  b5  15 s
  b6  variable frame rate (30 / 20 / 30 fps stretches)
  b7  3 min of 3840x2160 HEVC
  b2pq (extra) b2 with the PQ transfer, for the PQ branch of the tone map
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SRC = "/w/data/mushroom/room_datasets/honka/iphone/long_capture/images"
N_SRC = 335  # set by prepare_source(): the number of source frames after the sharpness filter
SHARP_KEEP = 0.5  # keep source frames whose Laplacian variance is >= this x the median

BT709_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def prepare_source(src: str, work: Path) -> str:
    """Link the sharp honka frames as frame_00001.jpg ... in `work`.

    Some honka frames are motion-blurred or face a blank wall (12-15% fall under 0.35 x the
    median Laplacian variance).  The blur test (b4) must see only the blur we inject, so the
    base clips use frames at or above SHARP_KEEP x median.  The natural-content result (all
    335 frames, b1: 102 frames kept, 18 of 120 windows dropped as blurry) is in NOTES.md.
    """
    import cv2
    import numpy as np
    names = sorted(n for n in os.listdir(src) if n.lower().endswith(".jpg"))
    var = []
    for n in names:
        g = cv2.imread(os.path.join(src, n), cv2.IMREAD_GRAYSCALE)
        s = 640.0 / max(g.shape)
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        var.append(cv2.Laplacian(g, cv2.CV_64F).var())
    var_a = np.array(var)
    keep = [n for n, v in zip(names, var_a) if v >= SHARP_KEEP * np.median(var_a)]
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("frame_*.jpg"):
        old.unlink()
    for i, n in enumerate(keep, 1):
        os.symlink(os.path.join(src, n), work / f"frame_{i:05d}.jpg")
    global N_SRC
    N_SRC = len(keep)
    print(f"source: {len(keep)} of {len(names)} honka frames at >= {SHARP_KEEP} x median sharpness", flush=True)
    return str(work)


def run(cmd: List[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def landscape_filter(w: int, h: int, period_s: float, fps: int = 30) -> str:
    """Crop a 16:9 band out of the portrait frame, pan it slowly, scale to w x h."""
    period_n = period_s * fps
    return (
        f"fps={fps},"
        f"crop=738:415:0:'(ih-415)*(0.5+0.5*sin(2*PI*n/{period_n:g}))',"
        f"scale={w}:{h}:flags=lanczos:out_color_matrix=bt709:out_range=tv,setsar=1,format=yuv420p"
    )


def src_args(src: str, seconds: float) -> List[str]:
    return ["-framerate", f"{N_SRC}/{seconds:g}", "-start_number", "1", "-i", f"{src}/frame_%05d.jpg"]


def make_b1(src: str, out: Path, seconds: float = 60.0, name: str = "b1", extra_vf: str = "") -> None:
    vf = landscape_filter(1920, 1080, 10.0) + (("," + extra_vf) if extra_vf else "")
    run(["ffmpeg", "-y", "-loglevel", "error", *src_args(src, 60.0), "-vf", vf, "-t", f"{seconds:g}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", *BT709_TAGS,
         "-movflags", "+faststart", out])


def patch_rotation(path: Path, clockwise_deg: int = 90) -> None:
    """Write a display matrix into the first tkhd box (ffmpeg 4.4's mp4 muxer ignores -metadata rotate).

    Clockwise 90 for display is the QuickTime matrix a=0 b=1 c=-1 d=0; ffprobe then reports rotation -90."""
    import struct
    data = bytearray(path.read_bytes())
    pos = data.find(b"tkhd")
    if pos < 0:
        raise RuntimeError("no tkhd box")
    version = data[pos + 4]
    moff = pos + (44 if version == 0 else 56)
    one, w = 0x00010000, 0x40000000
    mats = {0: (one, 0, 0, 0, one, 0, 0, 0, w), 90: (0, one, 0, -one, 0, 0, 0, 0, w),
            180: (-one, 0, 0, 0, -one, 0, 0, 0, w), 270: (0, -one, 0, one, 0, 0, 0, 0, w)}
    struct.pack_into(">9i", data, moff, *mats[clockwise_deg])
    path.write_bytes(bytes(data))


def make_b3(src: str, out: Path) -> None:
    # Upright 1080x1920 content, then turn the stored pixels 90 deg counter-clockwise and tag
    # rotate=90 (clockwise for display), as a phone does for portrait video.
    vf = ("fps=30,crop=559:994:'(iw-559)/2':0,"
          "scale=1080:1920:flags=lanczos:out_color_matrix=bt709:out_range=tv,transpose=2,setsar=1,format=yuv420p")
    run(["ffmpeg", "-y", "-loglevel", "error", *src_args(src, 60.0), "-vf", vf, "-t", "60",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", *BT709_TAGS,
         "-movflags", "+faststart", out])
    patch_rotation(out, 90)


BLUR_SEGMENTS = [[3 + 6 * i, 5 + 6 * i] for i in range(10)]  # [3,5) [9,11) ... [57,59)


def make_b4(src: str, out: Path) -> None:
    en = "+".join(f"between(t,{a},{b - 0.001})" for a, b in BLUR_SEGMENTS)
    make_b1(src, out, 60.0, "b4", extra_vf=f"gblur=sigma=14:enable='{en}'")


def make_b6(src: str, out: Path) -> None:
    sel = "select='lt(n,750)+between(n,750,1349)*lt(mod(n,3),2)+gte(n,1350)'"
    vf = landscape_filter(1920, 1080, 10.0) + "," + sel
    run(["ffmpeg", "-y", "-loglevel", "error", *src_args(src, 60.0), "-vf", vf, "-vsync", "vfr", "-t", "60",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", *BT709_TAGS,
         "-movflags", "+faststart", out])


def make_b7(src: str, out: Path) -> None:
    vf = landscape_filter(3840, 2160, 20.0)
    run(["ffmpeg", "-y", "-loglevel", "error", *src_args(src, 180.0), "-vf", vf, "-t", "180",
         "-c:v", "libx265", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", *BT709_TAGS,
         "-x265-params", "log-level=error", "-tag:v", "hvc1", "-movflags", "+faststart", out])


# ---- HDR: encode b1's pictures as BT.2100 HLG (or PQ) in numpy ------------------------------

def _hlg_oetf(e):
    import numpy as np
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    e = np.clip(e, 0.0, 1.0)
    lo = np.sqrt(3.0 * e)
    hi = a * np.log(np.maximum(12.0 * e - b, 1e-9)) + c
    return np.where(e <= 1.0 / 12.0, lo, hi)


def _pq_oetf(nits):
    import numpy as np
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    y = np.clip(nits / 10000.0, 0.0, 1.0) ** m1
    return ((c1 + c2 * y) / (1 + c3 * y)) ** m2


def make_hdr(b1: Path, out: Path, transfer: str) -> None:
    """Decode b1, treat it as SDR display light, re-encode as 10-bit HEVC with an HDR transfer.

    HLG: SDR white goes to scene light 0.265, which is HLG signal 0.75 (BT.2408 diffuse white).
    PQ:  SDR white (100 nits nominal) goes to 203 nits.
    """
    import cv2
    import numpy as np
    w, h = 1920, 1080
    m709_to_2020 = np.array([[0.627404, 0.329283, 0.043313],
                             [0.069097, 0.919540, 0.011362],
                             [0.016391, 0.088013, 0.895595]], dtype=np.float32)
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** 2.4  # BT.1886 display light, 0..1
    trc = "arib-std-b67" if transfer == "hlg" else "smpte2084"
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(b1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", "30",
               "-i", "-", "-vf", "scale=out_color_matrix=bt2020nc:out_range=tv,format=yuv420p10le",
               "-c:v", "libx265", "-preset", "fast", "-crf", "20",
               "-x265-params", f"log-level=error:colorprim=bt2020:transfer={trc}:colormatrix=bt2020nc:range=limited",
               "-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc", "-color_range", "tv",
               "-tag:v", "hvc1", "-movflags", "+faststart", str(out)]
    print("+", " ".join(enc_cmd), flush=True)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    nbytes = w * h * 3
    n = 0
    t0 = time.time()
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        lin = lut[rgb]
        lin = cv2.transform(lin, m709_to_2020)
        if transfer == "hlg":
            sig = _hlg_oetf(lin * 0.265)
        else:
            sig = _pq_oetf(lin * 203.0)
        enc.stdin.write((np.clip(sig, 0, 1) * 65535.0 + 0.5).astype("<u2").tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"  {transfer}: {n} frames in {time.time() - t0:.0f} s", flush=True)
    if enc.returncode != 0:
        raise RuntimeError("hdr encode failed")


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def write_sidecar(path: Path, extra: dict) -> None:
    p = probe(path)
    s = p["streams"][0]
    d = {"file": path.name, "codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
         "pix_fmt": s.get("pix_fmt"), "color_transfer": s.get("color_transfer"), "avg_frame_rate": s.get("avg_frame_rate"),
         "r_frame_rate": s.get("r_frame_rate"), "duration_s": float(p["format"]["duration"]),
         "bytes": int(p["format"]["size"]), "source": "MuSHRoom honka iphone long_capture (CC BY 4.0)",
         "source_frames": N_SRC, "source_filter": f"Laplacian variance >= {SHARP_KEEP} x median"}
    d.update(extra)
    path.with_suffix(".json").write_text(json.dumps(d, indent=2) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default="/w/data/fixtures/videos")
    ap.add_argument("--only", default="b1,b2,b3,b4,b5,b6,b7", help="comma list; add b2pq for the PQ variant")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want = [x.strip() for x in a.only.split(",") if x.strip()]
    a.src = prepare_source(a.src, out / ".src_sharp")
    for name in want:
        dst = out / f"{name}.mp4"
        if dst.exists() and not a.force:
            print(f"{name}: exists, skipping", flush=True)
            continue
        tmp = out / f".{name}.tmp.mp4"
        t0 = time.time()
        extra: Dict[str, object] = {}
        if name == "b1":
            make_b1(a.src, tmp)
        elif name == "b2" or name == "b2pq":
            b1 = out / "b1.mp4"
            if not b1.exists():
                make_b1(a.src, b1)
                write_sidecar(b1, {"desc": "60 s, 1080p, H.264, SDR"})
            make_hdr(b1, tmp, "hlg" if name == "b2" else "pq")
            extra = {"hdr": "hlg" if name == "b2" else "pq", "derived_from": "b1.mp4"}
        elif name == "b3":
            make_b3(a.src, tmp)
            extra = {"rotate_tag": 90, "stored": "1920x1080", "displays": "1080x1920 upright"}
        elif name == "b4":
            make_b4(a.src, tmp)
            extra = {"blurred_segments_s": BLUR_SEGMENTS, "blur": "gblur sigma=14"}
        elif name == "b5":
            make_b1(a.src, tmp, 15.0, "b5")
        elif name == "b6":
            make_b6(a.src, tmp)
            extra = {"fps_stretches": [[0, 25, 30], [25, 45, 20], [45, 60, 30]]}
        elif name == "b7":
            make_b7(a.src, tmp)
        else:
            print(f"unknown fixture {name}", file=sys.stderr)
            return 2
        os.replace(tmp, dst)
        write_sidecar(dst, extra)
        print(f"{name}: done in {time.time() - t0:.0f} s -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
