#!/usr/bin/env python3
"""Namma Space preflight: is this phone video good enough to reconstruct?

Standard library only (ffprobe and ffmpeg do the work), so it runs on a bare Mac.

    python3 pipeline/ingest/check_mac.py VIDEO [VIDEO ...] [--out check.json] [--json]

Prints one PASS / WARN / FAIL line per check, with the fix for anything that is not PASS,
then the verdict, and writes check.json  {verdict, warnings[], probe{...}, checks[], files[]}.
Exit code: 0 for PASS and WARN, 10 for FAIL (a capture problem, as in CONTRACTS.md), 2 for bad usage.

Checks (FINAL_PLAN 2.2):
  duration     PASS 90-420 s | WARN 45-90 s or > 420 s | FAIL < 45 s
  resolution   short side after rotation: PASS >= 1080 | WARN 720-1079 | FAIL < 720
  frame rate   PASS 24-60 | WARN variable frame rate (or > 60) | FAIL < 24
  HDR          PASS absent | WARN HLG or PQ (we tone-map)
  blur         share of 0.5 s windows whose sharpest frame is >= 0.5 x the clip median:
               PASS >= 80% | WARN 60-80% | FAIL < 60%
  exposure     median luma change between samples (per cent of the median luma):
               PASS < 8% | WARN 8-15% | FAIL > 15%

Sampling: ffmpeg decodes at 8 fps, scales to 640 px wide and prints, per frame, a sharpness
value (mean absolute Laplacian, computed by ffmpeg's convolution filter) and the mean luma
(signalstats).  Eight frames a second give four frames per 0.5 s window, the same window the
ingest stage uses to pick its sharpest frame.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

SAMPLE_FPS = 8
WINDOW = 4  # frames per 0.5 s window at 8 fps

# thresholds (FINAL_PLAN 2.2)
DUR_PASS = (90.0, 420.0)
DUR_FAIL = 45.0
RES_PASS, RES_FAIL = 1080, 720
FPS_PASS = (24.0, 60.0)
BLUR_PASS, BLUR_FAIL = 80.0, 60.0
EXPO_PASS, EXPO_FAIL = 8.0, 15.0
SHARP_REL = 0.5


def worst(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


# ---------------------------------------------------------------------------------------------
# ffprobe

def _run(cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _frac(s: Optional[str]) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def packet_times(path: str) -> List[float]:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
              "-of", "csv=p=0", path], timeout=120)
    out: List[float] = []
    for line in r.stdout.splitlines():
        line = line.strip().rstrip(",")
        if line and line != "N/A":
            try:
                out.append(float(line))
            except ValueError:
                pass
    out.sort()
    return out


def vfr_stats(times: List[float]) -> Dict[str, Any]:
    """Variable frame rate if more than 5% of the gaps differ from the median gap by over 25%."""
    if len(times) < 10:
        return {"vfr": False, "n_packets": len(times)}
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not gaps:
        return {"vfr": False, "n_packets": len(times)}
    med = statistics.median(gaps)
    odd = sum(1 for g in gaps if abs(g - med) > 0.25 * med)
    share = odd / len(gaps)
    return {"vfr": share > 0.05, "n_packets": len(times), "median_gap_s": round(med, 5),
            "odd_gap_share": round(share, 4), "min_gap_s": round(min(gaps), 5), "max_gap_s": round(max(gaps), 5),
            "mean_fps": round((len(times) - 1) / (times[-1] - times[0]), 3) if times[-1] > times[0] else 0.0}


def probe_video(path: str) -> Dict[str, Any]:
    r = _run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path], timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()[:300]}")
    j = json.loads(r.stdout)
    vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        raise RuntimeError("no video stream")
    s = vs[0]
    fmt = j.get("format", {})
    rot = 0.0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = float(sd["rotation"])
    if not rot and "rotate" in (s.get("tags") or {}):
        try:
            rot = -float(s["tags"]["rotate"])  # tag is clockwise degrees; ffprobe's rotation is counter-clockwise
        except ValueError:
            pass
    rot_deg = int(round(-rot)) % 360  # clockwise degrees to turn the stored picture for display
    w, h = int(s.get("width", 0)), int(s.get("height", 0))
    dw, dh = (h, w) if rot_deg in (90, 270) else (w, h)
    try:
        duration = float(fmt.get("duration") or s.get("duration") or 0)
    except ValueError:
        duration = 0.0
    pix = s.get("pix_fmt", "")
    bits = 8
    m = re.search(r"(\d+)(?:le|be)$", pix)
    if m:
        bits = int(m.group(1))
    if s.get("bits_per_raw_sample", "").isdigit():
        bits = max(bits, int(s["bits_per_raw_sample"]))
    trc = s.get("color_transfer", "") or ""
    hdr = "hlg" if trc == "arib-std-b67" else "pq" if trc == "smpte2084" else "none"
    times = packet_times(path)
    v = vfr_stats(times)
    avg = _frac(s.get("avg_frame_rate")) or v.get("mean_fps", 0.0) or 0.0
    return {
        "file": os.path.basename(path), "path": os.path.abspath(path),
        "bytes": int(fmt.get("size", 0) or 0), "container": fmt.get("format_name"),
        "codec": s.get("codec_name"), "profile": s.get("profile"), "pix_fmt": pix, "bit_depth": bits,
        "width": w, "height": h, "rotation_cw_deg": rot_deg, "display_width": dw, "display_height": dh,
        "short_side": min(dw, dh), "duration_s": round(duration, 3),
        "avg_fps": round(avg, 3), "r_frame_rate": s.get("r_frame_rate"), "nb_frames": s.get("nb_frames"),
        "vfr": v["vfr"], "vfr_detail": v,
        "color_transfer": trc or None, "color_primaries": s.get("color_primaries"), "color_space": s.get("color_space"),
        "color_range": s.get("color_range"), "hdr": hdr,
        "has_audio": any(x.get("codec_type") == "audio" for x in j.get("streams", [])),
        "creation_time": (fmt.get("tags") or {}).get("creation_time"),
    }


# ---------------------------------------------------------------------------------------------
# per-frame sharpness and luma, from ffmpeg

# Sharpness: convolve with the Laplacian; the filter clips at 0, so the mean of the result is
# half the mean absolute Laplacian.  Only the ratio to the clip median is used.
_LAP = "0 1 0 1 -4 1 0 1 0"


def _parse_series(text: str, key: str = "lavfi.signalstats.YAVG") -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    t = 0.0
    for line in text.splitlines():
        m = re.search(r"pts_time:([0-9.eE+-]+)", line)
        if m:
            t = float(m.group(1))
        elif line.startswith(key + "="):
            out.append((t, float(line.split("=", 1)[1])))
    return out


def sample_clip(path: str, hw: bool = False, timeout: float = 900.0) -> Dict[str, List[float]]:
    """Decode once at 8 fps; return {'t': [...], 'sharp': [...], 'luma': [...]}."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ns_check_") as td:
        sfile = os.path.join(td, "sharp.txt")
        graph = (
            f"[0:v:0]fps={SAMPLE_FPS},scale=640:-2:flags=bilinear,format=gray,split=2[a][b];"
            f"[a]convolution='{_LAP}:{_LAP}:{_LAP}:{_LAP}:5:5:5:5:0:0:0:0:0:0:0:0',signalstats,"
            f"metadata=mode=print:key=lavfi.signalstats.YAVG:file={sfile}[la];"
            f"[b]signalstats,metadata=mode=print:key=lavfi.signalstats.YAVG:file=-[lb]"
        )
        cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error"]
        if hw:
            cmd += ["-hwaccel", "videotoolbox"]
        cmd += ["-i", path, "-an", "-sn", "-dn", "-filter_complex", graph,
                "-map", "[la]", "-f", "null", "-", "-map", "[lb]", "-f", "null", "-"]
        r = _run(cmd, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg sampling failed: {r.stderr.strip()[-300:]}")
        luma = _parse_series(r.stdout)
        sharp = []
        if os.path.exists(sfile):
            with open(sfile) as fh:
                sharp = _parse_series(fh.read())
    n = min(len(luma), len(sharp))
    return {"t": [x[0] for x in luma[:n]], "sharp": [x[1] for x in sharp[:n]], "luma": [x[1] for x in luma[:n]]}


def windows(values: List[float], size: int = WINDOW) -> List[List[float]]:
    return [values[i:i + size] for i in range(0, len(values), size) if len(values[i:i + size]) >= max(2, size // 2)]


def blur_metrics(sharp: List[float]) -> Dict[str, Any]:
    ws = windows(sharp)
    if len(ws) < 4:
        return {"share_ok_pct": None, "n_windows": len(ws)}
    best = [max(w) for w in ws]
    med = statistics.median(best)
    ok = [b >= SHARP_REL * med for b in best]
    bad_ranges: List[List[float]] = []
    start: Optional[int] = None
    for i, good in enumerate(ok + [True]):
        if not good and start is None:
            start = i
        if good and start is not None:
            bad_ranges.append([start * WINDOW / SAMPLE_FPS, i * WINDOW / SAMPLE_FPS])
            start = None
    return {"share_ok_pct": round(100.0 * sum(ok) / len(ok), 1), "n_windows": len(ws),
            "median_best_sharp": round(med, 3), "blurry_ranges_s": bad_ranges}


def exposure_metrics(luma: List[float]) -> Dict[str, Any]:
    ws = [sum(w) / len(w) for w in windows(luma)]
    if len(ws) < 3:
        return {"median_change_pct": None, "n_windows": len(ws)}
    med = statistics.median(ws)
    deltas = [abs(b - a) for a, b in zip(ws, ws[1:])]
    pct = 100.0 * statistics.median(deltas) / max(med, 1.0)
    return {"median_change_pct": round(pct, 2), "median_luma": round(med, 1),
            "min_luma": round(min(ws), 1), "max_luma": round(max(ws), 1), "n_windows": len(ws)}


# ---------------------------------------------------------------------------------------------
# the checks

def _check(name: str, status: str, value: Any, detail: str, fix: str = "") -> Dict[str, Any]:
    return {"name": name, "status": status, "value": value, "detail": detail, "fix": fix if status != PASS else ""}


def evaluate(probe: Dict[str, Any], blur: Optional[Dict[str, Any]], expo: Optional[Dict[str, Any]],
             sample_error: Optional[str] = None) -> List[Dict[str, Any]]:
    cs: List[Dict[str, Any]] = []

    d = probe["duration_s"]
    if d < DUR_FAIL:
        cs.append(_check("duration", FAIL, d, f"{d:.0f} s is under 45 s; a room needs a slow walk of at least 90 s",
                         "Film again, slower and longer: 90 to 420 s, covering every wall, corner and doorway."))
    elif d < DUR_PASS[0]:
        cs.append(_check("duration", WARN, d, f"{d:.0f} s is short (ideal 90 to 420 s)",
                         "Film a longer, slower pass; add a second loop at a different height if the room is large."))
    elif d > DUR_PASS[1]:
        cs.append(_check("duration", WARN, d, f"{d:.0f} s is long; frames are capped at 450 and the run takes longer",
                         "Split it into two takes (one per half of the space) or accept the longer run."))
    else:
        cs.append(_check("duration", PASS, d, f"{d:.0f} s"))

    s = probe["short_side"]
    dim = f"{probe['display_width']}x{probe['display_height']}"
    if s < RES_FAIL:
        cs.append(_check("resolution", FAIL, dim, f"{dim}: short side {s} px is under 720",
                         "Film at 1080p or 4K; turn off any 'low data' or 'space saver' camera mode."))
    elif s < RES_PASS:
        cs.append(_check("resolution", WARN, dim, f"{dim}: short side {s} px is under 1080",
                         "Film at 1080p or 4K."))
    else:
        cs.append(_check("resolution", PASS, dim, dim))

    fps = probe["avg_fps"]
    if fps and fps < FPS_PASS[0]:
        cs.append(_check("frame_rate", FAIL, fps, f"{fps:.1f} fps is under 24; frames will be too far apart",
                         "Film at 30 fps or 60 fps (iPhone: Settings > Camera > Record Video)."))
    elif probe["vfr"]:
        det = probe["vfr_detail"]
        cs.append(_check("frame_rate", WARN, fps, f"variable frame rate, mean {fps:.1f} fps "
                         f"(gaps {det.get('min_gap_s')} to {det.get('max_gap_s')} s); we resample to 8 fps",
                         "Usually harmless. If frames look out of order, re-export at a fixed 30 fps."))
    elif fps > FPS_PASS[1] + 0.5:
        cs.append(_check("frame_rate", WARN, fps, f"{fps:.1f} fps is above 60; we resample to 8 fps",
                         "Harmless; 30 or 60 fps is enough."))
    else:
        cs.append(_check("frame_rate", PASS, fps, f"{fps:.1f} fps"))

    if probe["hdr"] != "none":
        cs.append(_check("hdr", WARN, probe["hdr"],
                         f"HDR transfer {probe['color_transfer']} ({probe['hdr'].upper()}); we tone-map to SDR",
                         "Nothing needed. To avoid tone mapping, turn off HDR video "
                         "(iPhone: Settings > Camera > Record Video > HDR Video)."))
    else:
        cs.append(_check("hdr", PASS, "none", "SDR"))

    if blur is None or blur.get("share_ok_pct") is None:
        msg = sample_error or "too few frames to measure blur"
        cs.append(_check("blur", WARN, None, f"not measured: {msg}", "Re-run the check; if it repeats, the video may not decode."))
    else:
        sh = blur["share_ok_pct"]
        rng = ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in blur["blurry_ranges_s"][:8])
        more = "" if len(blur["blurry_ranges_s"]) <= 8 else f" (+{len(blur['blurry_ranges_s']) - 8} more)"
        det = f"{sh:.0f}% of 0.5 s windows have a sharp frame" + (f"; soft at {rng}{more}" if rng else "")
        fix = "Move more slowly, hold the phone with both hands, avoid turning fast at corners; reshoot the soft stretches."
        if sh < BLUR_FAIL:
            cs.append(_check("blur", FAIL, sh, det, fix))
        elif sh < BLUR_PASS:
            cs.append(_check("blur", WARN, sh, det, fix))
        else:
            cs.append(_check("blur", PASS, sh, det))

    if expo is None or expo.get("median_change_pct") is None:
        cs.append(_check("exposure", WARN, None, f"not measured: {sample_error or 'too few frames'}",
                         "Re-run the check."))
    else:
        e = expo["median_change_pct"]
        det = f"median luma change {e:.1f}% between samples (luma {expo['min_luma']:.0f} to {expo['max_luma']:.0f})"
        fix = "Lock exposure and white balance (iPhone: tap and hold on the wall until AE/AF LOCK shows), and keep windows out of frame."
        if e > EXPO_FAIL:
            cs.append(_check("exposure", FAIL, e, det, fix))
        elif e >= EXPO_PASS:
            cs.append(_check("exposure", WARN, e, det, fix))
        else:
            cs.append(_check("exposure", PASS, e, det))
    return cs


def check_video(path: str, hw: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    probe = probe_video(path)
    blur = expo = None
    err = None
    try:
        sm = sample_clip(path, hw=hw)
        blur = blur_metrics(sm["sharp"])
        expo = exposure_metrics(sm["luma"])
    except Exception as ex:  # keep the probe-based checks even if sampling fails
        err = str(ex)
    checks = evaluate(probe, blur, expo, err)
    verdict = PASS
    for c in checks:
        verdict = worst(verdict, c["status"])
    return {"file": probe["file"], "verdict": verdict, "checks": checks, "probe": probe,
            "blur": blur, "exposure": expo, "seconds": round(time.time() - t0, 1)}


def check_all(paths: List[str], hw: bool = False) -> Dict[str, Any]:
    files = [check_video(p, hw=hw) for p in paths]
    verdict = PASS
    warnings: List[str] = []
    for f in files:
        verdict = worst(verdict, f["verdict"])
        for c in f["checks"]:
            if c["status"] != PASS:
                warnings.append(f"{f['file']}: {c['name']} {c['status']}: {c['detail']}")
    return {"schema": "namma-space/check@1", "verdict": verdict, "warnings": warnings,
            "probe": files[0]["probe"] if files else {}, "checks": files[0]["checks"] if files else [],
            "files": files, "tool": {"ffmpeg": _ffmpeg_version(), "sample_fps": SAMPLE_FPS},
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def _ffmpeg_version() -> str:
    try:
        return _run(["ffmpeg", "-version"]).stdout.splitlines()[0]
    except Exception:
        return "unknown"


def render(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    for f in result["files"]:
        p = f["probe"]
        lines.append(f"{f['file']}: {p['display_width']}x{p['display_height']} {p['codec']} {p['avg_fps']:.1f} fps "
                     f"{p['duration_s']:.0f} s  [{f['seconds']:.0f} s to check]")
        for c in f["checks"]:
            lines.append(f"  {c['status']:<4}  {c['name']:<11} {c['detail']}")
            if c["fix"]:
                lines.append(f"        fix: {c['fix']}")
        lines.append(f"  => {f['verdict']}")
    if len(result["files"]) > 1:
        lines.append(f"overall: {result['verdict']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Preflight check for a phone video (stdlib only).")
    ap.add_argument("videos", nargs="+", help="one or more video files (take A, take B)")
    ap.add_argument("--out", default="check.json", help="where to write check.json (default ./check.json; '' to skip)")
    ap.add_argument("--run-dir", help="write <run-dir>/00_preflight/check.json instead of --out")
    ap.add_argument("--json", action="store_true", help="print the JSON instead of the text report")
    ap.add_argument("--hwaccel", action="store_true", help="try VideoToolbox decoding (faster on big HEVC)")
    a = ap.parse_args(argv)
    for tool in ("ffprobe", "ffmpeg"):
        if not shutil.which(tool):
            print(f"{tool} not found on PATH (brew install ffmpeg)", file=sys.stderr)
            return 2
    missing = [v for v in a.videos if not os.path.isfile(v)]
    if missing:
        print("no such file: " + ", ".join(missing), file=sys.stderr)
        return 2
    try:
        result = check_all(a.videos, hw=a.hwaccel)
    except RuntimeError as ex:
        print(f"FAIL  cannot read the video: {ex}", file=sys.stderr)
        return 10
    out = os.path.join(a.run_dir, "00_preflight", "check.json") if a.run_dir else a.out
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    print(json.dumps(result, indent=2) if a.json else render(result) + (f"\nwrote {out}" if out else ""))
    return 10 if result["verdict"] == FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
