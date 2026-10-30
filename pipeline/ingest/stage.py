"""Frame selection (FINAL_PLAN 2.2).  Runs in namma-space-gpu:0.1 (ffmpeg, OpenCV, numpy, Pillow).

Video path:
  1. ffprobe: rotation, HDR transfer, bit depth, variable frame rate.
  2. ffmpeg decodes with autorotate, resamples to 8 fps, scales the long side to at most 1920
     (lanczos), tone-maps HLG/PQ to SDR (zscale + tonemap=reinhard, or a numpy fallback when the
     build lacks them) and writes candidate JPEGs (q 2).
     Order differs from the plan on purpose: resample and scale first, tone-map after, so the
     float tone-map runs on 8 fps at the final size, not on every 4K frame.
  3. Laplacian variance on a 640 px grey copy; keep the sharpest frame of each window of 4.
  4. Drop windows under 0.35 x the rolling median (window 31) of those winners.
  5. Drop near-duplicates: mean absolute difference of 64x36 grey thumbnails under 1.5 against
     the previous kept frame.
  6. Thin uniformly to the cap (450, or 600 for take B).
  7. Rename the winners 000001.jpg ... in capture order.

Frames path (--frames DIR): sorted by name, EXIF orientation applied, EXIF stripped, re-encoded
(JPEG q 95), long side capped at 1920, no blur or duplicate filter unless --select.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import check_mac

SAMPLE_FPS = 8
WINDOW = 4
BLUR_REL = 0.35
ROLL = 31
DUP_MAD = 1.5
GREY_LONG = 640
THUMB = (64, 36)
LONG_SIDE = 1920
CAP = 450
CAP_TAKE_B = 600
MIN_DURATION_S = 45.0
MIN_FRAMES = 60
WARN_FRAMES = 120
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".3gp"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

EXIT_OK, EXIT_CAPTURE, EXIT_INFRA = 0, 10, 20


class CaptureProblem(Exception):
    """Exit 10: the capture cannot be used; ranges (seconds) say what to reshoot."""

    def __init__(self, reason: str, ranges: Optional[List[List[float]]] = None):
        super().__init__(reason)
        self.reason = reason
        self.ranges = ranges or []


class InfraProblem(Exception):
    """Exit 20."""


# ---------------------------------------------------------------------------------------------
# ffmpeg helpers

def ffmpeg_filters() -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return r.stdout


def has_filters(*names: str) -> bool:
    text = ffmpeg_filters()
    return all(re.search(rf"\s{re.escape(n)}\s", text) for n in names)


def ffmpeg_version() -> str:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        return "unknown"


def scale_expr(long_side: int) -> str:
    # Landscape: width = min(long_side, iw); portrait: height = min(long_side, ih); never upscale.
    return (f"scale=w='if(gte(iw,ih),min({long_side},iw),-2)':h='if(gte(iw,ih),-2,min({long_side},ih))':flags=lanczos")


def tonemap_chain(probe: Dict[str, Any]) -> str:
    """HLG or PQ to SDR bt709.  The input description is explicit so untagged primaries do not stop zscale.

    The plan names tonemap=hable.  On b2 (b1 re-encoded as HLG) hable returns mean luma 105 against b1's 139 and
    0.80 of b1's contrast; reinhard returns 137 and 0.93 (tonemap_sweep.py, NOTES.md), so reinhard is used."""
    trc = probe.get("color_transfer") or "arib-std-b67"
    prim = probe.get("color_primaries") or "bt2020"
    mat = probe.get("color_space") or "bt2020nc"
    rng = probe.get("color_range") or "tv"
    rng = "limited" if rng in ("tv", "limited", "mpeg") else "full"
    return (f"zscale=tin={trc}:pin={prim}:min={mat}:rin={rng}:t=linear:npl=100,format=gbrpf32le,"
            f"zscale=p=bt709,tonemap=reinhard:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def video_filter(probe: Dict[str, Any], long_side: int, hdr_mode: str) -> str:
    parts = [f"fps={SAMPLE_FPS}", "showinfo", scale_expr(long_side)]
    if hdr_mode == "zscale":
        parts.append(tonemap_chain(probe))
    # JPEG carries no matrix tag; readers assume BT.601, so convert to it (full range) here.
    parts.append("scale=in_color_matrix=auto:out_color_matrix=smpte170m:out_range=pc")
    parts.append("format=yuvj420p")
    return ",".join(parts)


# ---------------------------------------------------------------------------------------------
# image measurements

def grey_small(img_gray: np.ndarray, long_side: int = GREY_LONG) -> np.ndarray:
    import cv2
    h, w = img_gray.shape[:2]
    s = long_side / float(max(h, w))
    if s >= 1.0:
        return img_gray
    return cv2.resize(img_gray, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def measure_file(path: str) -> Tuple[float, np.ndarray]:
    """Laplacian variance on a 640 px grey copy, and a 64x36 (or 36x64) thumbnail."""
    import cv2
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"cannot read {path}")
    return measure_gray(g)


def measure_gray(g: np.ndarray) -> Tuple[float, np.ndarray]:
    import cv2
    small = grey_small(g)
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    th = THUMB if g.shape[1] >= g.shape[0] else (THUMB[1], THUMB[0])
    thumb = cv2.resize(small, th, interpolation=cv2.INTER_AREA).astype(np.float32)
    return sharp, thumb


def measure_many(paths: List[str], workers: int = 16) -> List[Tuple[float, np.ndarray]]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(measure_file, paths))


# ---------------------------------------------------------------------------------------------
# selection (pure functions on arrays, so tests can drive them)

def best_of_windows(sharp: List[float], size: int = WINDOW) -> List[int]:
    """Index of the sharpest candidate in each window of `size` consecutive candidates."""
    out: List[int] = []
    for i in range(0, len(sharp), size):
        w = sharp[i:i + size]
        if not w:
            continue
        out.append(i + int(np.argmax(w)))
    return out


def rolling_median(x: np.ndarray, win: int = ROLL) -> np.ndarray:
    n = len(x)
    half = win // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(x[max(0, i - half):min(n, i + half + 1)])
    return out


def blur_drop(sharp_win: np.ndarray, rel: float = BLUR_REL, win: int = ROLL) -> np.ndarray:
    """Boolean mask of winners to drop for blur."""
    if len(sharp_win) == 0:
        return np.zeros(0, bool)
    return sharp_win < rel * rolling_median(sharp_win, win)


def dup_drop(thumbs: List[np.ndarray], alive: np.ndarray, thr: float = DUP_MAD) -> np.ndarray:
    """Mask of windows dropped as near-duplicates of the previous kept window."""
    drop = np.zeros(len(thumbs), bool)
    prev: Optional[np.ndarray] = None
    for i, th in enumerate(thumbs):
        if not alive[i]:
            continue
        if prev is not None and float(np.mean(np.abs(th - prev))) < thr:
            drop[i] = True
            continue
        prev = th
    return drop


def thin_indices(n: int, cap: int) -> List[int]:
    if n <= cap:
        return list(range(n))
    idx = np.unique(np.round(np.linspace(0, n - 1, cap)).astype(int))
    return [int(i) for i in idx]


def merge_ranges(spans: List[Tuple[float, float]], gap: float = 1e-6) -> List[List[float]]:
    out: List[List[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


@dataclass
class Selection:
    n_cand: int = 0
    n_windows: int = 0
    kept: List[Dict[str, Any]] = field(default_factory=list)  # {cand_index, t, sharp, w}
    dropped: Dict[str, int] = field(default_factory=lambda: {"blur": 0, "dup": 0, "thin": 0})
    dropped_windows: List[Dict[str, Any]] = field(default_factory=list)  # {t0, t1, reason}


def select_frames(t: List[float], sharp: List[float], thumbs: List[np.ndarray], cap: int,
                  window: int = WINDOW, fps: float = SAMPLE_FPS) -> Selection:
    sel = Selection(n_cand=len(t))
    win_idx = best_of_windows(sharp, window)
    sel.n_windows = len(win_idx)
    sw = np.array([sharp[i] for i in win_idx])
    th = [thumbs[i] for i in win_idx]
    blur = blur_drop(sw)
    dup = dup_drop(th, ~blur)
    alive = ~(blur | dup)

    def span(w: int) -> Tuple[float, float]:
        a = (w * window) / fps
        return a, a + window / fps

    for w in range(len(win_idx)):
        if blur[w]:
            sel.dropped["blur"] += 1
            sel.dropped_windows.append({"t0": span(w)[0], "t1": span(w)[1], "reason": "blur"})
        elif dup[w]:
            sel.dropped["dup"] += 1
            sel.dropped_windows.append({"t0": span(w)[0], "t1": span(w)[1], "reason": "dup"})
    live = [w for w in range(len(win_idx)) if alive[w]]
    keep_pos = thin_indices(len(live), cap)
    keep_set = {live[p] for p in keep_pos}
    for w in live:
        if w not in keep_set:
            sel.dropped["thin"] += 1
            sel.dropped_windows.append({"t0": span(w)[0], "t1": span(w)[1], "reason": "thin"})
    for w in sorted(keep_set):
        i = win_idx[w]
        sel.kept.append({"cand_index": i, "t": t[i], "sharp": float(sharp[i]), "w": w})
    return sel


# ---------------------------------------------------------------------------------------------
# video path

_SHOWINFO = re.compile(r"pts_time:\s*([0-9.eE+-]+)")


def decode_candidates(path: str, probe: Dict[str, Any], out_dir: str, long_side: int, hdr_mode: str,
                      log) -> List[float]:
    os.makedirs(out_dir, exist_ok=True)
    vf = video_filter(probe, long_side, hdr_mode)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i", path, "-an", "-sn", "-dn",
           "-vf", vf, "-q:v", "2", "-start_number", "1", os.path.join(out_dir, "%06d.jpg")]
    log("+ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-6:])
        raise CaptureProblem(f"cannot decode {os.path.basename(path)}: {tail}. Re-export the video as H.264 "
                             f"(iPhone: Settings > Camera > Formats > Most Compatible) or send a frames folder.")
    ts = [float(m.group(1)) for m in _SHOWINFO.finditer(r.stderr)]
    return ts


def decode_candidates_numpy_tonemap(path: str, probe: Dict[str, Any], out_dir: str, long_side: int, log) -> List[float]:
    """Fallback for ffmpeg builds without zscale/tonemap: raw rgb48 frames, numpy tone map, cv2 JPEG."""
    import cv2
    from . import tonemap_np
    os.makedirs(out_dir, exist_ok=True)
    w, h = probe["display_width"], probe["display_height"]
    s = min(1.0, long_side / float(max(w, h)))
    ow, oh = (int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2)
    vf = f"fps={SAMPLE_FPS},showinfo,scale={ow}:{oh}:flags=lanczos:in_color_matrix=bt2020:in_range=tv,format=rgb48le"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i", path, "-an", "-sn", "-dn", "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    log("+ " + " ".join(cmd) + "   (numpy tone map)")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import threading
    err_chunks: List[bytes] = []
    th = threading.Thread(target=lambda: err_chunks.append(p.stderr.read()), daemon=True)
    th.start()
    nbytes = ow * oh * 6
    k = 0
    while True:
        buf = p.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        arr = np.frombuffer(buf, "<u2").reshape(oh, ow, 3).astype(np.float32) / 65535.0
        rgb = tonemap_np.tonemap(arr, probe.get("hdr", "hlg"))
        k += 1
        cv2.imwrite(os.path.join(out_dir, f"{k:06d}.jpg"), cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    p.wait()
    th.join()
    if p.returncode != 0:
        raise CaptureProblem(f"cannot decode {os.path.basename(path)}")
    return [float(m.group(1)) for m in _SHOWINFO.finditer(b"".join(err_chunks).decode("utf8", "replace"))]


def ingest_video(path: str, work: str, long_side: int, cap: int, min_duration: float, force: bool, log,
                 hdr_mode_pref: str = "auto") -> Dict[str, Any]:
    t0 = time.time()
    probe = check_mac.probe_video(path)
    log(f"{probe['file']}: {probe['display_width']}x{probe['display_height']} {probe['codec']} {probe['pix_fmt']} "
        f"{probe['avg_fps']} fps {probe['duration_s']} s rot={probe['rotation_cw_deg']} hdr={probe['hdr']} vfr={probe['vfr']}")
    warnings: List[str] = []
    if probe["duration_s"] < min_duration and not force:
        raise CaptureProblem(f"too short: {probe['file']} is {probe['duration_s']:.0f} s, need at least {min_duration:.0f} s "
                             f"(film 90 to 420 s)", [[0.0, probe["duration_s"]]])
    hdr_mode = "none"
    if probe["hdr"] != "none":
        if hdr_mode_pref == "numpy" or not has_filters("zscale", "tonemap"):
            hdr_mode = "numpy"
            warnings.append("ffmpeg lacks zscale/tonemap (or --tonemap numpy); used the numpy tone map")
        else:
            hdr_mode = "zscale"
    cand = os.path.join(work, "cand_" + re.sub(r"[^A-Za-z0-9]", "_", probe["file"]))
    if hdr_mode == "numpy":
        ts = decode_candidates_numpy_tonemap(path, probe, cand, long_side, log)
    else:
        ts = decode_candidates(path, probe, cand, long_side, hdr_mode, log)
    files = sorted(str(p) for p in Path(cand).glob("*.jpg"))
    if len(ts) != len(files):
        warnings.append(f"showinfo timestamps ({len(ts)}) differ from candidate files ({len(files)}); used k/{SAMPLE_FPS}")
        ts = [i / SAMPLE_FPS for i in range(len(files))]
    if not files:
        raise CaptureProblem(f"no frames decoded from {probe['file']}")
    t_dec = time.time()
    m = measure_many(files)
    sharp = [x[0] for x in m]
    thumbs = [x[1] for x in m]
    sel = select_frames(ts, sharp, thumbs, cap=10 ** 9)  # thinning happens after all takes are joined
    log(f"{probe['file']}: {len(files)} candidates, {sel.n_windows} windows, kept {len(sel.kept)} "
        f"(blur {sel.dropped['blur']}, dup {sel.dropped['dup']}); decode {t_dec - t0:.0f} s, measure {time.time() - t_dec:.0f} s")
    return {"probe": probe, "cand_dir": cand, "files": files, "sel": sel, "hdr_mode": hdr_mode,
            "warnings": warnings, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------
# frames path

def natural_key(s: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ingest_frames(src_dir: str, work: str, long_side: int, log) -> Dict[str, Any]:
    from PIL import Image, ImageOps
    import cv2
    names = sorted((n for n in os.listdir(src_dir) if os.path.splitext(n)[1].lower() in IMAGE_EXT), key=natural_key)
    skipped = [n for n in os.listdir(src_dir) if os.path.splitext(n)[1].lower() not in IMAGE_EXT and not n.startswith(".")]
    out = os.path.join(work, "cand_frames")
    os.makedirs(out, exist_ok=True)
    warnings: List[str] = []
    if skipped:
        warnings.append(f"skipped {len(skipped)} non-image files, e.g. {skipped[:3]}")
    exif_rot = 0
    sizes: Dict[Tuple[int, int], int] = {}
    files: List[str] = []
    sharp: List[float] = []
    thumbs: List[np.ndarray] = []
    for i, n in enumerate(names):
        im = Image.open(os.path.join(src_dir, n))
        orient = im.getexif().get(0x0112, 1) if hasattr(im, "getexif") else 1
        if orient not in (None, 1):
            exif_rot += 1
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = long_side / float(max(w, h))
        if s < 1.0:
            im = im.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)
        sizes[im.size] = sizes.get(im.size, 0) + 1
        p = os.path.join(out, f"{i + 1:06d}.jpg")
        im.save(p, "JPEG", quality=95)  # no exif= argument: EXIF is dropped
        files.append(p)
        g = np.asarray(im.convert("L"))
        sh, th = measure_gray(g)
        sharp.append(sh)
        thumbs.append(th)
    if len(sizes) > 1:
        warnings.append(f"images have {len(sizes)} different sizes {dict((f'{a}x{b}', c) for (a, b), c in sizes.items())}; "
                        "COLMAP will need one camera per size")
    log(f"frames: {len(names)} images from {src_dir}, {exif_rot} had EXIF rotation")
    return {"names": names, "files": files, "sharp": sharp, "thumbs": thumbs, "exif_rot": exif_rot,
            "sizes": sizes, "warnings": warnings}


# ---------------------------------------------------------------------------------------------
# the stage

def find_inputs(run_dir: str) -> Tuple[str, List[str]]:
    inp = os.path.join(run_dir, "input")
    if not os.path.isdir(inp):
        raise CaptureProblem(f"{inp} does not exist; put videos or a frames folder there")
    vids = sorted((os.path.join(inp, n) for n in os.listdir(inp) if os.path.splitext(n)[1].lower() in VIDEO_EXT),
                  key=natural_key)
    if vids:
        return "video", vids
    for sub in ("frames", "images"):
        d = os.path.join(inp, sub)
        if os.path.isdir(d):
            return "frames", [d]
    if any(os.path.splitext(n)[1].lower() in IMAGE_EXT for n in os.listdir(inp)):
        return "frames", [inp]
    raise CaptureProblem(f"no video or images found in {inp}")


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def run_stage(run_dir: str, videos: Optional[List[str]] = None, frames_dir: Optional[str] = None,
              cap: int = CAP, long_side: int = LONG_SIDE, min_duration: float = MIN_DURATION_S,
              force: bool = False, select_in_frames: bool = False, keep_candidates: bool = False,
              tonemap: str = "auto", min_free_gb: float = 3.0, log=print) -> Dict[str, Any]:
    t_start = time.time()
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    if free_gb(run_dir) < min_free_gb:
        raise InfraProblem(f"only {free_gb(run_dir):.1f} GB free in {run_dir} (need {min_free_gb:g})")
    if frames_dir:
        mode, items = "frames", [frames_dir]
    elif videos:
        mode, items = "video", videos
    else:
        mode, items = find_inputs(run_dir)
    final = os.path.join(run_dir, "10_frames")
    work = os.path.join(run_dir, f"10_frames.tmp-{os.getpid()}")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "images"))
    try:
        res = _run(mode, items, work, cap, long_side, min_duration, force, select_in_frames, tonemap, log)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    # the candidates are scratch: gone before the stage directory appears
    for d in Path(work).glob("cand_*"):
        if not keep_candidates:
            shutil.rmtree(d, ignore_errors=True)
    res["seconds"] = round(time.time() - t_start, 1)
    res["tool"] = {"ffmpeg": ffmpeg_version(), "opencv": _cv2_version(), "numpy": np.__version__}
    with open(os.path.join(work, "frames.json"), "w") as fh:
        json.dump(res, fh, indent=2)
        fh.write("\n")
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(work, final)
    log(f"wrote {final}: {res['n_kept']} frames in {res['seconds']} s")
    return res


def _cv2_version() -> str:
    try:
        import cv2
        return cv2.__version__
    except Exception:
        return "missing"


def _run(mode: str, items: List[str], work: str, cap: int, long_side: int, min_duration: float, force: bool,
         select_in_frames: bool, tonemap: str, log) -> Dict[str, Any]:
    images = os.path.join(work, "images")
    warnings: List[str] = []
    kept: List[Dict[str, Any]] = []
    entries: List[Tuple[str, Dict[str, Any]]] = []  # (candidate file, kept record)
    dropped_ranges: List[Dict[str, Any]] = []
    n_cand = 0
    dropped = {"blur": 0, "dup": 0, "thin": 0}
    inputs: List[Dict[str, Any]] = []
    hdr = "none"
    rotation: Any = 0
    tonemap_used = "none"
    size = None

    if mode == "video":
        for v in items:
            r = ingest_video(v, work, long_side, cap, min_duration, force, log, tonemap)
            p, sel = r["probe"], r["sel"]
            warnings += [f"{p['file']}: {w}" for w in r["warnings"]]
            n_cand += sel.n_cand
            for k in ("blur", "dup"):
                dropped[k] += sel.dropped[k]
            for dw in sel.dropped_windows:
                dropped_ranges.append({"src": p["file"], **dw})
            for k in sel.kept:
                entries.append((r["files"][k["cand_index"]], {"t": round(k["t"], 3), "sharp": round(k["sharp"], 2), "src": p["file"]}))
            inputs.append({"file": p["file"], "kind": "video", "duration_s": p["duration_s"], "width": p["display_width"],
                           "height": p["display_height"], "codec": p["codec"], "avg_fps": p["avg_fps"], "vfr": p["vfr"],
                           "rotation_cw_deg": p["rotation_cw_deg"], "hdr": p["hdr"], "n_cand": sel.n_cand,
                           "n_windows": sel.n_windows, "n_kept": len(sel.kept), "seconds": r["seconds"]})
            if p["hdr"] != "none":
                hdr = p["hdr"]
                tonemap_used = r["hdr_mode"]
            if p["rotation_cw_deg"]:
                rotation = p["rotation_cw_deg"]
        src = "video"
    else:
        r = ingest_frames(items[0], work, long_side, log)
        warnings += r["warnings"]
        n_cand = len(r["files"])
        rotation = {"exif_rotated": r["exif_rot"]}
        if select_in_frames:
            t = [float(i) for i in range(n_cand)]
            sel = select_frames(t, r["sharp"], r["thumbs"], cap=10 ** 9, window=1)
            keep_idx = [k["cand_index"] for k in sel.kept]
            dropped["blur"], dropped["dup"] = sel.dropped["blur"], sel.dropped["dup"]
        else:
            keep_idx = list(range(n_cand))
        for i in keep_idx:
            entries.append((r["files"][i], {"t": None, "sharp": round(r["sharp"][i], 2), "src": r["names"][i]}))
        inputs.append({"file": items[0], "kind": "frames", "n_input": n_cand, "sizes": {f"{a}x{b}": c for (a, b), c in r["sizes"].items()}})
        src = "frames"

    # join all takes, thin uniformly to the cap
    pos = thin_indices(len(entries), cap)
    dropped["thin"] = len(entries) - len(pos)
    final_entries = [entries[i] for i in pos]
    if len(final_entries) < MIN_FRAMES and not force:
        rng = merge_ranges([(d["t0"], d["t1"]) for d in dropped_ranges if d["reason"] in ("blur", "dup")])
        raise CaptureProblem(f"too few usable frames: {len(final_entries)} (need at least {MIN_FRAMES}); "
                             f"{dropped['blur']} windows were blurry. Reshoot slower, with both hands on the phone.", rng)
    if len(final_entries) < WARN_FRAMES:
        warnings.append(f"only {len(final_entries)} frames (under {WARN_FRAMES}); the reconstruction may be thin")
    import cv2
    for k, (cand_file, rec) in enumerate(final_entries):
        name = f"{k + 1:06d}.jpg"
        os.replace(cand_file, os.path.join(images, name))
        rec = dict(rec)
        rec["name"] = name
        kept.append({"name": name, "t": rec["t"], "sharp": rec["sharp"], "src": rec["src"]})
    first = cv2.imread(os.path.join(images, "000001.jpg"))
    size = [int(first.shape[1]), int(first.shape[0])] if first is not None else None
    thin_names = len(pos) < len(entries)
    return {
        "schema": "namma-space/frames@1",
        "src": src, "inputs": inputs,
        "n_cand": n_cand, "n_kept": len(kept), "dropped": dropped,
        "hdr": hdr, "tonemap": tonemap_used, "rotation": rotation,
        "sample_fps": SAMPLE_FPS, "window": WINDOW, "cap": cap, "long_side_max": long_side, "size": size,
        "rules": {"blur_rel": BLUR_REL, "blur_window": ROLL, "dup_mad": DUP_MAD, "grey_long_side": GREY_LONG,
                  "thumb": list(THUMB), "jpeg": "q2 (video) / q95 (frames)"},
        "kept": kept,
        "dropped_windows": dropped_ranges,
        "warnings": warnings,
    }
