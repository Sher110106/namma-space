#!/usr/bin/env python3
"""Bundle a run directory into a static web folder (80_web) and validate it (test H8).

Standard library only. Reads (CONTRACTS 2.11):
  60_export/{splat_sd.spz, splat_hd.spz?, splat_preview.spz?, export.json, heldout/}
  70_grid/{grid.png, minimap.png, grid.json}
  40_align/align.json   50_train/train.json   input/meta.json   run.json (optional)
Writes (CONTRACTS 2.12):
  <out>/index.html js/ css/ vendor/ scenes/index.json scenes/<name>/scene.json + data files

Usage:
  bundle.py --run RUN_DIR [--out DIR] [--name NAME] [--title T] [--kind captured|fixture|public]
            [--source S] [--license L] [--attribution A] [--captured YYYY-MM-DD] [--eye-height M]
            [--video-url URL] [--default] [--strict] [--no-viewer] [--places places.json]
  bundle.py --validate WEB_DIR [--strict]
  bundle.py --zip OUT.zip --run RUN_DIR ...      also write the offline copy (web folder + a one-line start note)

Places (Round 2 preview): --places FILE merges named places into scene.json "places". FILE is a JSON list (what the viewer's
"Copy places JSON" button gives) or {"places": [...]}. Each place: name (required), kind room|item|zone (default item),
position [x, y, z] in metres or [x, z] (y is then 1.0 for items, 0.0 otherwise), aliases (list or "a, b"), yaw_deg, radius_m, id.
Each place may also carry "notes" (free text, kept as given: who placed it and how).
Without --places, RUN_DIR/places.json is used when it exists.

Other things worth knowing (30 Oct review fixes): splat_hd.spz is left out when it is byte-identical to splat_sd.spz (the viewer then shows no
HD button); scene.json splats.render.clear_color is a neutral colour from the held-out photos' mean (or the minimap, or a fixed grey) that the
viewer draws behind the splats; scale.checked_against / check_error_pct come from 40_align/rescale.json when a tape check was made; a rebuild
without --video-url keeps the video link already in out/index.html.

Exit codes: 0 valid, 1 invalid (errors), 2 bad usage / missing inputs.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VIEWER = HERE.parent
SCHEMA = "namma-space/scene@1"
REQUIRED_TOP = ["schema", "name", "title", "provenance", "frame", "splats", "camera", "spawn", "bounds", "grid", "minimap",
                "bookmarks", "tour", "heldout", "places", "scale", "metrics", "build"]
LIMITS = {  # CONTRACTS 2.12: (max splats, max bytes, sh_degree)
    "sd": (1_000_000, 40_000_000, 1), "hd": (1_500_000, 90_000_000, 3), "preview": (200_000, 4_000_000, 0)}
PAGES_FILE_LIMIT = 100 * 1024 * 1024


# ----------------------------------------------------------------------------------------------- small readers
def read_json(p: Path, default: Any = None) -> Any:
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def png_info(path: Path) -> dict[str, Any]:
    """Header of a PNG: width, height, bit depth, colour type."""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path}: not a PNG")
        n, typ = struct.unpack(">I4s", f.read(8))
        if typ != b"IHDR":
            raise ValueError(f"{path}: no IHDR")
        w, h, depth, ctype, _c, _f, interlace = struct.unpack(">IIBBBBB", f.read(13))
    return {"width": w, "height": h, "depth": depth, "ctype": ctype, "interlace": interlace}


def png_grey(path: Path) -> tuple[int, int, bytes]:
    """Decode an 8-bit non-interlaced PNG to one byte per pixel (first channel). Enough for grid.png."""
    w, h, chans, rows = _png_rows(path)
    out = bytearray(w * h)
    for y in range(h):
        line = rows[y]
        for x in range(w):
            out[y * w + x] = line[x * chans]
    return w, h, bytes(out)


def png_rgb(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Decode an 8-bit non-interlaced PNG to (r, g, b) tuples; palette images are not supported (ValueError)."""
    info = png_info(path)
    if info["ctype"] == 3:
        raise ValueError(f"{path}: palette PNG not supported here")
    w, h, chans, rows = _png_rows(path)
    px = []
    for y in range(h):
        line = rows[y]
        for x in range(w):
            o = x * chans
            px.append((line[o], line[o], line[o]) if chans < 3 else (line[o], line[o + 1], line[o + 2]))
    return w, h, px


def _png_rows(path: Path) -> tuple[int, int, int, list[bytearray]]:
    info = png_info(path)
    if info["depth"] != 8 or info["interlace"] != 0 or info["ctype"] not in (0, 2, 3, 4, 6):
        raise ValueError(f"{path}: need 8-bit non-interlaced PNG, got {info}")
    chans = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[info["ctype"]]
    raw = b""
    with open(path, "rb") as f:
        f.read(8)
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            n, typ = struct.unpack(">I4s", head)
            data = f.read(n)
            f.read(4)
            if typ == b"IDAT":
                raw += data
            elif typ == b"IEND":
                break
    raw = zlib.decompress(raw)
    w, h = info["width"], info["height"]
    stride = w * chans
    rows: list[bytearray] = []
    prev = bytearray(stride)
    pos = 0
    for y in range(h):
        ft = raw[pos]
        line = bytearray(raw[pos + 1: pos + 1 + stride])
        pos += 1 + stride
        if ft == 1:
            for i in range(chans, stride):
                line[i] = (line[i] + line[i - chans]) & 255
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif ft == 3:
            for i in range(stride):
                left = line[i - chans] if i >= chans else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif ft == 4:
            for i in range(stride):
                a = line[i - chans] if i >= chans else 0
                b = prev[i]
                c = prev[i - chans] if i >= chans else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        elif ft != 0:
            raise ValueError(f"{path}: bad PNG filter {ft}")
        rows.append(line)
        prev = line
    return w, h, chans, rows


def spz_header(path: Path) -> dict[str, Any] | None:
    """SPZ v1-3 are gzip around a 16-byte header (NGSP, version, numPoints, shDegree, fractionalBits, flags)."""
    try:
        with gzip.open(path, "rb") as f:
            h = f.read(16)
        if len(h) == 16 and h[:4] == b"NGSP":
            ver, n, sh, frac, flags, _r = struct.unpack("<IIBBBB", h[4:16])
            return {"version": ver, "count": n, "sh_degree": sh, "fractional_bits": frac}
    except (OSError, EOFError, zlib.error):
        pass
    try:
        with open(path, "rb") as f:
            h = f.read(32)
        if h[:4] == b"NGSP":  # SPZ v4: header is outside the zstd stream
            ver, n = struct.unpack("<II", h[4:12])
            return {"version": ver, "count": n, "sh_degree": h[12], "fractional_bits": h[13]}
    except OSError:
        pass
    return None


def mat_to_pqs(m: list[list[float]]) -> dict[str, Any]:
    """Decompose a 4x4 row-major similarity matrix into position, quaternion [x,y,z,w], uniform scale."""
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    cols = [math.sqrt(sum(r[i][j] ** 2 for i in range(3))) for j in range(3)]
    s = sum(cols) / 3.0
    r = [[r[i][j] / s for j in range(3)] for i in range(3)]
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0:
        k = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * k, (r[2][1] - r[1][2]) / k, (r[0][2] - r[2][0]) / k, (r[1][0] - r[0][1]) / k
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        k = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w, x, y, z = (r[2][1] - r[1][2]) / k, 0.25 * k, (r[0][1] + r[1][0]) / k, (r[0][2] + r[2][0]) / k
    elif r[1][1] > r[2][2]:
        k = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w, x, y, z = (r[0][2] - r[2][0]) / k, (r[0][1] + r[1][0]) / k, 0.25 * k, (r[1][2] + r[2][1]) / k
    else:
        k = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w, x, y, z = (r[1][0] - r[0][1]) / k, (r[0][2] + r[2][0]) / k, (r[1][2] + r[2][1]) / k, 0.25 * k
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return {"position": [m[0][3], m[1][3], m[2][3]], "quaternion": [x / n, y / n, z / n, w / n], "scale": s}


def parse_t_final(t: Any) -> dict[str, Any]:
    """Accept {position|translation, quaternion|rotation, scale}, a 4x4 (nested or flat 16), or nothing."""
    ident = {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0], "scale": 1.0}
    if t is None:
        return ident
    if isinstance(t, dict):
        for k in ("matrix", "T", "m"):
            if k in t:
                return parse_t_final(t[k])
        pos = t.get("position", t.get("translation", ident["position"]))
        q = t.get("quaternion", t.get("rotation", ident["quaternion"]))
        if isinstance(q, list) and len(q) == 9:  # a rotation matrix
            q = mat_to_pqs([q[0:3] + [0], q[3:6] + [0], q[6:9] + [0], [0, 0, 0, 1]])["quaternion"]
        return {"position": [float(v) for v in pos], "quaternion": [float(v) for v in q], "scale": float(t.get("scale", 1.0))}
    if isinstance(t, list):
        if len(t) == 16:
            t = [t[0:4], t[4:8], t[8:12], t[12:16]]
        if len(t) == 4 and all(len(r) == 4 for r in t):
            return mat_to_pqs(t)
    raise ValueError(f"cannot read T_final: {t!r}")


def round_list(v: list[float], nd: int = 4) -> list[float]:
    return [round(float(x), nd) for x in v]


def pos3(p: Any, y: float) -> list[float]:
    p = [float(v) for v in p]
    return [p[0], y, p[1]] if len(p) == 2 else [p[0], p[1] if p[1] else y, p[2]]


def view_entry(e: Any, eye: float, i: int) -> dict[str, Any]:
    """Normalise a spawn / bookmark / tour stop to {position, yaw_deg, pitch_deg, ...}."""
    if isinstance(e, (list, tuple)):  # [x, z, yaw] or [x, y, z, yaw]
        e = {"position": list(e[:3]) if len(e) >= 4 else list(e[:2]), "yaw_deg": e[3] if len(e) >= 4 else (e[2] if len(e) == 3 else 0)}
    out = dict(e)
    out["position"] = round_list(pos3(e.get("position", e.get("pos", e.get("xz"))), eye), 3)
    out["yaw_deg"] = round(float(e.get("yaw_deg", e.get("yaw", 0.0))), 2)
    out["pitch_deg"] = round(float(e.get("pitch_deg", e.get("pitch", 0.0))), 2)
    return out


# ----------------------------------------------------------------------------------------------- build
def git_rev(cwd: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_bytes(a: Path, b: Path) -> bool:
    """Same size first (cheap), then the same SHA-256."""
    return a.stat().st_size == b.stat().st_size and file_sha256(a) == file_sha256(b)


DEFAULT_CLEAR = [60, 58, 55]
CLEAR_LUMA = (40, 120)       # keep the clear colour between these grey levels: not a black void, not a bright one
CLEAR_KEEP_COLOUR = 0.4      # share of the scene's own colour that stays; the rest is grey of the same brightness


def neutral_clear(mean_rgb: list[float]) -> list[int]:
    """A neutral colour from the scene's mean colour: mostly grey of the same brightness, brightness limited to CLEAR_LUMA."""
    r, g, b = mean_rgb
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    c = [luma + CLEAR_KEEP_COLOUR * (v - luma) for v in (r, g, b)]
    target = min(CLEAR_LUMA[1], max(CLEAR_LUMA[0], luma))
    k = target / luma if luma > 1e-6 else 0.0
    out = [v * k if luma > 1e-6 else target for v in c]
    return [int(round(min(255, max(0, v)))) for v in out]


def scene_clear_colour(held_photos: list[Path], minimap: Path) -> tuple[list[int], str]:
    """The colour drawn where no splat covers the pixel (a hole, a wall closer than the model has data for).
    Source 1: mean of the held-out photos (needs Pillow; used when it is installed). Source 2: mean of the minimap pixels that differ
    from its background. Source 3: a fixed neutral grey. Returns ([r, g, b], source)."""
    try:
        from PIL import Image  # optional; the bundle tool needs only the standard library without it
        acc, n = [0.0, 0.0, 0.0], 0
        for ph in held_photos[:16]:
            with Image.open(ph) as im:
                px = im.convert("RGB").resize((1, 1), Image.BOX).getpixel((0, 0))
            acc = [a + v for a, v in zip(acc, px)]
            n += 1
        if n:
            return neutral_clear([a / n for a in acc]), f"mean of {n} held-out photos"
    except Exception:
        pass
    try:
        w, h, px = png_rgb(minimap)
        bg = px[0]
        keep = [p for p in px if abs(p[0] - bg[0]) + abs(p[1] - bg[1]) + abs(p[2] - bg[2]) > 40]
        if len(keep) > 0.02 * len(px):
            m = [sum(p[i] for p in keep) / len(keep) for i in range(3)]
            return neutral_clear(m), "mean of the minimap"
    except Exception:
        pass
    return list(DEFAULT_CLEAR), "default grey"


def compose_scene(run: Path, name: str, args: argparse.Namespace, warnings: list[str]) -> tuple[dict[str, Any], dict[str, Path]]:
    """Build scene.json from the run directory. Returns (scene, {relative target name: source path})."""
    exp_dir, grid_dir = run / "60_export", run / "70_grid"
    export = read_json(exp_dir / "export.json", {}) or {}
    grid_json = read_json(grid_dir / "grid.json", {}) or {}
    align = read_json(run / "40_align" / "align.json", {}) or {}
    train = read_json(run / "50_train" / "train.json", {}) or {}
    meta = read_json(run / "input" / "meta.json", {}) or {}
    runj = read_json(run / "run.json", {}) or {}
    files: dict[str, Path] = {}

    def need(p: Path) -> Path:
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
        return p

    gpng = need(grid_dir / "grid.png")
    mpng = need(grid_dir / "minimap.png")
    gi = png_info(gpng)
    mi = png_info(mpng)
    files["grid.png"], files["minimap.png"] = gpng, mpng

    eye = args.eye_height or grid_json.get("eye_height_m") or align.get("h_cap_m") or 1.42
    eye = round(min(1.8, max(1.2, float(eye))), 2)

    # splats
    splats: dict[str, Any] = {}
    for kind, fname in (("sd", "splat_sd.spz"), ("hd", "splat_hd.spz"), ("preview", "splat_preview.spz")):
        src = exp_dir / fname
        if not src.exists():
            if kind == "sd":
                need(src)
            splats[kind] = None
            continue
        hdr = spz_header(src)
        count = hdr["count"] if hdr else (export.get(kind, {}) or {}).get("count", 0)
        sh = hdr["sh_degree"] if hdr else (export.get(kind, {}) or {}).get("sh", 0)
        if not hdr:
            warnings.append(f"{fname}: SPZ header not readable here; count/sh_degree taken from export.json")
        exp_count = (export.get(kind, {}) or {}).get("count")
        if exp_count and count and exp_count != count:
            warnings.append(f"{fname}: header says {count} splats, export.json says {exp_count}")
        splats[kind] = {"url": fname, "count": int(count), "sh_degree": int(sh), "bytes": src.stat().st_size}
        files[fname] = src
    sd_e, hd_e = splats.get("sd"), splats.get("hd")
    if sd_e and hd_e:
        if same_bytes(files["splat_sd.spz"], files["splat_hd.spz"]):
            # the export wrote the same file twice (the cleaned set is under the SD cap and SD already needs SH3): the HD button would
            # download it again for nothing (review findings 19 and 47)
            warnings.append("splat_hd.spz is byte-identical to splat_sd.spz: HD left out, so the viewer shows no HD button")
            splats["hd"] = None
            files.pop("splat_hd.spz", None)
        elif (sd_e["bytes"], sd_e["count"], sd_e["sh_degree"]) == (hd_e["bytes"], hd_e["count"], hd_e["sh_degree"]):
            warnings.append("splat_hd.spz has the same size, count and SH degree as splat_sd.spz but different bytes; kept")
    tf = parse_t_final(export.get("T_final"))
    splats["transform"] = {"position": round_list(tf["position"], 6), "quaternion": round_list(tf["quaternion"], 7), "scale": round(tf["scale"], 6)}
    render = export.get("render") or train.get("render")
    if not render:
        render = {"antialiased": True, "preBlurAmount": 0.0, "blurAmount": 0.3, "maxStdDev": 2.828}
        warnings.append("export.json/train.json carry no 'render' block; assumed antialiased training (preBlur 0, blur 0.3)")
    splats["render"] = {"antialiased": bool(render.get("antialiased", True)), "preBlurAmount": float(render.get("preBlurAmount", 0.0)),
                        "blurAmount": float(render.get("blurAmount", 0.3)), "maxStdDev": float(render.get("maxStdDev", 2.828))}
    held_files = sorted((exp_dir / "heldout").glob("*.jpg")) if (exp_dir / "heldout").is_dir() else []
    clear, clear_src = scene_clear_colour(held_files, mpng)
    splats["render"]["clear_color"] = clear
    splats["render"]["clear_color_source"] = clear_src
    if clear_src == "default grey":
        warnings.append("clear colour: no held-out photo or minimap colour could be read (install Pillow for the photos); using the default grey")

    # grid
    cell = float(grid_json.get("cell", 0.05))
    origin = [float(v) for v in grid_json.get("origin_xz", [0.0, 0.0])]
    gw, gh, gdata = png_grey(gpng)
    if (grid_json.get("w") and grid_json["w"] != gw) or (grid_json.get("h") and grid_json["h"] != gh):
        warnings.append(f"grid.json says {grid_json.get('w')}x{grid_json.get('h')}, grid.png is {gw}x{gh}; using the PNG")
    cells = [(i, j) for i in range(gh) for j in range(gw) if gdata[i * gw + j] != 0]
    gb = grid_json.get("bounds")
    if isinstance(gb, dict) and len(gb.get("min", [])) == 3 and len(gb.get("max", [])) == 3:
        bounds = {"min": round_list(gb["min"], 3), "max": round_list(gb["max"], 3)}   # W4 measured these from the splats
    elif cells:
        zs = [origin[1] + i * cell for i, _ in cells] + [origin[1] + (max(i for i, _ in cells) + 1) * cell]
        xs = [origin[0] + j * cell for _, j in cells] + [origin[0] + (max(j for _, j in cells) + 1) * cell]
        bounds = {"min": [round(min(xs), 3), -0.1, round(min(zs), 3)], "max": [round(max(xs), 3), round(max(2.9, eye + 1.4), 2), round(max(zs), 3)]}
    else:
        bounds = {"min": [origin[0], -0.1, origin[1]], "max": [origin[0] + gw * cell, 2.9, origin[1] + gh * cell]}
        warnings.append("grid.png has no non-zero cells")
    walk_cells = sum(1 for v in gdata if v == 255)
    spawn_in = grid_json.get("spawn")
    if spawn_in is None:
        warnings.append("grid.json has no spawn; using the first walkable cell")
        k = next((k for k, v in enumerate(gdata) if v == 255), 0)
        spawn_in = {"position": [origin[0] + (k % gw + 0.5) * cell, eye, origin[1] + (k // gw + 0.5) * cell], "yaw_deg": 0}
    spawn = view_entry(spawn_in, eye, 0)
    spawn["position"][1] = eye
    bookmarks = []
    for i, b in enumerate(grid_json.get("bookmarks", [])):
        v = view_entry(b, eye, i)
        v["position"][1] = eye
        v["id"] = str(b.get("id", f"v{i + 1}")) if isinstance(b, dict) else f"v{i + 1}"
        v["name"] = str(b.get("name", f"View {i + 1}")) if isinstance(b, dict) else f"View {i + 1}"
        bookmarks.append(v)
    tour = []
    for i, t in enumerate(grid_json.get("tour", [])):
        v = view_entry(t, eye, i)
        v["position"][1] = eye
        v["t"] = round(float(t.get("t", i)) if isinstance(t, dict) else float(i), 2)
        tour.append(v)

    # held-out photos
    held: list[dict[str, Any]] = []
    hdir = exp_dir / "heldout"
    hj = read_json(hdir / "heldout.json") or read_json(hdir / "index.json")
    if isinstance(hj, dict):
        hj = hj.get("items") or hj.get("heldout") or []
    for e in hj or []:
        photo = hdir / Path(e["photo"]).name
        if not photo.exists():
            warnings.append(f"held-out photo missing: {photo}")
            continue
        rel = f"heldout/{photo.name}"
        files[rel] = photo
        held.append({"id": str(e.get("id", photo.stem)), "photo": rel, "position": round_list(e["position"], 4), "quaternion": round_list(e["quaternion"], 6),
                     "fov_y_deg": round(float(e["fov_y_deg"]), 4), "principal_point": round_list(e.get("principal_point", [0.5, 0.5]), 5),
                     "aspect": round(float(e["aspect"]), 5), "psnr": round(float(e.get("psnr") or 0), 2)})
    if not held:
        warnings.append("no held-out pairs: the reality check (C) will be unavailable")

    # scale, metrics
    sc = align.get("scale", {}) or {}
    scale = {"value": float(sc.get("value", 1.0)), "pct_68": float(sc.get("pct_68") or 0), "method": sc.get("method", ""), "cues": sc.get("cues", []), "checked_against": sc.get("checked_against")}
    # a tape check (ns rescale --check / --use-measures) lands in 40_align/rescale.json, not in align.json; only then may the viewer say "checked"
    resc = read_json(run / "40_align" / "rescale.json", {}) or {}
    rval = resc.get("validation") or {}
    if resc.get("checked_against") and rval.get("n_used"):
        scale["checked_against"] = str(resc["checked_against"])
        if isinstance(rval.get("auto_error_pct"), (int, float)):
            scale["check_error_pct"] = round(float(rval["auto_error_pct"]), 2)
        scale["check_n"] = int(rval["n_used"])
    variants = {v.get("name"): v for v in train.get("variants", [])}
    chosen = variants.get(train.get("chosen"), next(iter(variants.values()), {}))
    metrics = {"heldout_n": len(held), "psnr": chosen.get("psnr") or 0, "ssim": chosen.get("ssim") or 0, "lpips": chosen.get("lpips") or 0,
               "lpips_net": chosen.get("lpips_net", "alex"), "cc_psnr": chosen.get("cc_psnr") or 0, "sd_psnr": (export.get("sd", {}) or {}).get("psnr") or 0}

    ppm = (grid_json.get("minimap") or {}).get("px_per_m") or (mi["width"] / (gw * cell) if gw * cell else 40)
    prov = {"kind": args.kind, "source": args.source or meta.get("notes", ""), "license": args.license or "", "attribution": args.attribution or "",
            "captured": args.captured or "", "manual_steps": list(meta.get("manual_steps", [])) + list(args.manual_step or [])}
    places: list[dict[str, Any]] = []
    pfile = Path(args.places) if getattr(args, "places", None) else (run / "places.json" if (run / "places.json").exists() else None)
    if pfile is not None:
        places = load_places(pfile, (gw, gh, gdata, origin, cell), warnings)
        print(f"bundle: {len(places)} places from {pfile}", file=sys.stderr)
    scene = {
        "schema": SCHEMA, "name": name, "title": args.title or name, "provenance": prov,
        "frame": {"units": "m", "up": "+y", "floor_y": float(grid_json.get("floor_y") or 0.0), "handedness": "right"},
        "splats": splats,
        "camera": {"eye_height": eye, "fov_y_deg": 70, "near": 0.05, "far": 50, "walk_mps": 1.4, "run_mult": 2.0},
        "spawn": spawn, "bounds": bounds,
        "grid": {"url": "grid.png", "origin_xz": origin, "cell": cell, "width": gw, "height": gh, "row0": "min_z", "values": {"walkable": 255, "floor": 128, "blocked": 0}},
        "minimap": {"url": "minimap.png", "origin_xz": origin, "px_per_m": round(ppm, 3)},
        "bookmarks": bookmarks, "tour": tour, "heldout": held, "places": places, "scale": scale, "metrics": metrics,
        "build": {"git": runj.get("git", "") or git_rev(VIEWER), "pipeline": "0.9.0", "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "report": "report.json"},
    }
    if walk_cells == 0:
        warnings.append("grid.png has no walkable (255) cells")
    return scene, files


# ----------------------------------------------------------------------------------------------- places
PLACE_KINDS = ("room", "item", "zone")
PLACE_RADIUS = {"room": 1.5, "zone": 1.0, "item": 0.4}


def slugify(text: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:40] or "place"


def nearest_walkable_m(data: bytes, w: int, h: int, origin: list[float], cell: float, x: float, z: float) -> float:
    """Distance in metres from (x, z) to the nearest walkable (255) cell centre; inf when there is none."""
    best = math.inf
    for k, v in enumerate(data):
        if v == 255:
            cx = origin[0] + (k % w + 0.5) * cell
            cz = origin[1] + (k // w + 0.5) * cell
            d = math.hypot(cx - x, cz - z)
            if d < best:
                best = d
    return best


def load_places(path: Path, grid: tuple[int, int, bytes, list[float], float] | None, warnings: list[str]) -> list[dict[str, Any]]:
    """Read and normalise a places file. Raises ValueError with a clear message on bad input."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("places", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of places or {{\"places\": [...]}}")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, p in enumerate(raw):
        where = f"{path.name} place {i + 1}"
        if not isinstance(p, dict):
            raise ValueError(f"{where}: not an object")
        name = str(p.get("name", "")).strip()
        if not name:
            raise ValueError(f"{where}: name is required")
        kind = p.get("kind", "item")
        if kind not in PLACE_KINDS:
            raise ValueError(f"{where} ({name}): kind must be one of {', '.join(PLACE_KINDS)}, got {kind!r}")
        pos = p.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) not in (2, 3) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos):
            raise ValueError(f"{where} ({name}): position must be [x, y, z] or [x, z] in metres")
        pos = [float(v) for v in pos]
        if len(pos) == 2:
            pos = [pos[0], 1.0 if kind == "item" else 0.0, pos[1]]
        al = p.get("aliases", [])
        aliases = [a.strip() for a in al.split(",")] if isinstance(al, str) else [str(a).strip() for a in al]
        aliases = [a for a in aliases if a]
        radius = p.get("radius_m", PLACE_RADIUS[kind])
        if not isinstance(radius, (int, float)) or not radius > 0:
            raise ValueError(f"{where} ({name}): radius_m must be a positive number")
        pid = slugify(str(p.get("id") or name))
        base, n = pid, 2
        while pid in seen:
            pid, n = f"{base}-{n}", n + 1
        seen.add(pid)
        yaw = p.get("yaw_deg", 0.0)
        if not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
            raise ValueError(f"{where} ({name}): yaw_deg must be a number")
        entry = {"id": pid, "name": name, "aliases": aliases, "kind": kind, "position": [round(v, 3) for v in pos], "yaw_deg": round(float(yaw), 2), "radius_m": round(float(radius), 3)}
        notes = str(p.get("notes", "") or "").strip()
        if notes:
            entry["notes"] = notes          # optional: who placed it and how
        out.append(entry)
        if grid:
            gw, gh, gdata, origin, cell = grid
            d = nearest_walkable_m(gdata, gw, gh, origin, cell, pos[0], pos[2])
            if d > 3.0:
                warnings.append(f"place {name!r}: the nearest walkable cell is {d:.1f} m away; check its position (x, z in metres, same frame as the grid)")
    return out


def existing_video_url(out: Path) -> str:
    """The video link already in out/index.html, so a rebuild without --video-url does not erase it."""
    try:
        m = re.search(r'<meta name="ns-video" content="([^"]*)"', (out / "index.html").read_text())
        return m.group(1) if m else ""
    except OSError:
        return ""


def copy_viewer(out: Path, video_url: str) -> None:
    src = VIEWER / "src"
    for rel in ("index.html", "js", "css"):
        s, d = src / rel, out / rel
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
        else:
            shutil.copy2(s, d)
    vend = out / "vendor"
    if vend.exists():
        shutil.rmtree(vend)
    shutil.copytree(VIEWER / "vendor", vend, ignore=shutil.ignore_patterns(".DS_Store"))
    html = (out / "index.html").read_text()
    html = html.replace('<meta name="ns-video" content="">', f'<meta name="ns-video" content="{video_url}">')
    (out / "index.html").write_text(html)


def update_index(out: Path, name: str, title: str, make_default: bool) -> None:
    p = out / "scenes" / "index.json"
    idx = read_json(p, None) or {"default": name, "scenes": []}
    idx["scenes"] = [s for s in idx["scenes"] if s.get("name") != name] + [{"name": name, "title": title}]
    if make_default or not idx.get("default") or idx["default"] not in [s["name"] for s in idx["scenes"]]:
        idx["default"] = name
    p.write_text(json.dumps(idx, indent=2) + "\n")


# ----------------------------------------------------------------------------------------------- validate (H8)
def validate_web(web: Path, strict: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warns: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    for rel in ("index.html", "js/main.js", "css/viewer.css", "vendor/spark.module.js", "vendor/three.module.js", "vendor/three.core.js",
                "vendor/addons/postprocessing/Pass.js", "vendor/LICENSES/spark-LICENSE.txt", "vendor/LICENSES/three-LICENSE.txt"):
        check(f"file {rel}", (web / rel).is_file(), "missing")
    idx = read_json(web / "scenes" / "index.json")
    check("scenes/index.json", isinstance(idx, dict) and idx.get("scenes"), "missing or empty")
    names = [s.get("name") for s in (idx or {}).get("scenes", [])]
    check("index default listed", (idx or {}).get("default") in names, f"default={((idx or {}).get('default'))} scenes={names}")
    for name in names:
        sdir = web / "scenes" / str(name)
        sj = read_json(sdir / "scene.json")
        check(f"{name}: scene.json readable", isinstance(sj, dict), "missing or invalid JSON")
        if not isinstance(sj, dict):
            continue
        for k in REQUIRED_TOP:
            check(f"{name}: key {k}", k in sj, "missing")
        check(f"{name}: schema", sj.get("schema") == SCHEMA, str(sj.get("schema")))
        check(f"{name}: name matches folder", sj.get("name") == name, f"{sj.get('name')} vs {name}")
        sp = sj.get("splats", {}) or {}
        check(f"{name}: splats.sd present", bool(sp.get("sd")), "required")
        tr = sp.get("transform", {}) or {}
        check(f"{name}: transform shape", len(tr.get("position", [])) == 3 and len(tr.get("quaternion", [])) == 4 and isinstance(tr.get("scale"), (int, float)), str(tr))
        q = tr.get("quaternion", [0, 0, 0, 1])
        qn = math.sqrt(sum(float(v) ** 2 for v in q)) if len(q) == 4 else 0
        check(f"{name}: quaternion unit length", abs(qn - 1) < 1e-3, f"|q|={qn:.5f}")
        for kind in ("sd", "hd", "preview"):
            e = sp.get(kind)
            if not e:
                continue
            f = sdir / e["url"]
            ok = f.is_file()
            check(f"{name}: {kind} file", ok, f"missing {f}")
            if not ok:
                continue
            size = f.stat().st_size
            check(f"{name}: {kind} bytes match", size == e.get("bytes"), f"file {size} vs scene.json {e.get('bytes')}")
            check(f"{name}: {kind} under Pages file limit", size < PAGES_FILE_LIMIT, f"{size} bytes")
            hdr = spz_header(f) if f.suffix == ".spz" else None
            if hdr:
                check(f"{name}: {kind} count matches SPZ header", hdr["count"] == e.get("count"), f"header {hdr['count']} vs scene.json {e.get('count')}")
                check(f"{name}: {kind} sh_degree matches header", hdr["sh_degree"] == e.get("sh_degree"), f"header {hdr['sh_degree']} vs {e.get('sh_degree')}")
            mx_n, mx_b, _sh = LIMITS[kind]
            if e.get("count", 0) > mx_n:
                warns.append(f"{name}: {kind} has {e['count']} splats, budget {mx_n}")
            if size > mx_b:
                warns.append(f"{name}: {kind} is {size / 1e6:.1f} MB, budget {mx_b / 1e6:.0f} MB")
        if sp.get("sd") and sp.get("hd") and (sdir / sp["sd"]["url"]).is_file() and (sdir / sp["hd"]["url"]).is_file():
            if sp["sd"]["url"] == sp["hd"]["url"] or same_bytes(sdir / sp["sd"]["url"], sdir / sp["hd"]["url"]):
                warns.append(f"{name}: hd is the same file as sd, so the HD button would download it again for nothing")
        cc = (sp.get("render", {}) or {}).get("clear_color")
        if cc is not None:
            check(f"{name}: render.clear_color shape", isinstance(cc, list) and len(cc) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in cc), str(cc))
        g = sj.get("grid", {}) or {}
        gp = sdir / g.get("url", "grid.png")
        check(f"{name}: grid.png", gp.is_file(), "missing")
        if gp.is_file():
            info = png_info(gp)
            check(f"{name}: grid dims match", (info["width"], info["height"]) == (g.get("width"), g.get("height")), f"png {info['width']}x{info['height']} vs {g.get('width')}x{g.get('height')}")
            check(f"{name}: grid is 8-bit", info["depth"] == 8 and info["interlace"] == 0, str(info))
            gw, gh, data = png_grey(gp)
            vals = set(data)
            check(f"{name}: grid values in {{0,128,255}}", vals <= {0, 128, 255}, f"found {sorted(vals)[:8]}")
            sp_pos = (sj.get("spawn", {}) or {}).get("position", [0, 0, 0])
            ci = math.floor((sp_pos[2] - g["origin_xz"][1]) / g["cell"])
            cj = math.floor((sp_pos[0] - g["origin_xz"][0]) / g["cell"])
            inside = 0 <= ci < gh and 0 <= cj < gw
            check(f"{name}: spawn inside grid", inside, f"cell ({ci},{cj})")
            if inside and data[ci * gw + cj] != 255:
                warns.append(f"{name}: spawn cell is not walkable (the viewer will push it to the nearest walkable cell)")
        m = sj.get("minimap", {}) or {}
        mp = sdir / m.get("url", "minimap.png")
        check(f"{name}: minimap.png", mp.is_file(), "missing")
        if mp.is_file():
            mi = png_info(mp)
            exp_w = (g.get("width", 0) * g.get("cell", 0)) * m.get("px_per_m", 0)
            check(f"{name}: minimap px_per_m consistent", abs(mi["width"] - exp_w) <= 2, f"png width {mi['width']} vs {exp_w:.1f}")
        seen_ids: set[str] = set()
        for pl in sj.get("places", []) or []:
            pid = pl.get("id") if isinstance(pl, dict) else None
            pos = pl.get("position", []) if isinstance(pl, dict) else []
            ok = (isinstance(pid, str) and pid and pid not in seen_ids and str(pl.get("name", "")).strip() != "" and pl.get("kind") in PLACE_KINDS
                  and len(pos) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)
                  and isinstance(pl.get("radius_m"), (int, float)) and pl["radius_m"] > 0 and isinstance(pl.get("aliases", []), list))
            check(f"{name}: place {pid} shape", ok, str(pl))
            if isinstance(pid, str):
                seen_ids.add(pid)
        for b in sj.get("bookmarks", []):
            check(f"{name}: bookmark {b.get('id')} shape", len(b.get("position", [])) == 3 and "yaw_deg" in b, str(b))
        for h in sj.get("heldout", []):
            hp = sdir / h.get("photo", "")
            check(f"{name}: heldout {h.get('id')} photo", hp.is_file(), f"missing {hp}")
            check(f"{name}: heldout {h.get('id')} pose", len(h.get("position", [])) == 3 and len(h.get("quaternion", [])) == 4 and h.get("fov_y_deg", 0) > 0 and h.get("aspect", 0) > 0, "bad pose")
        if len(sj.get("tour", [])) < 2:
            warns.append(f"{name}: tour has fewer than 2 stops (benchmark T unavailable)")
        scale = sj.get("scale", {}) or {}
        if not scale.get("pct_68"):
            warns.append(f"{name}: scale.pct_68 is 0, so the HUD badge says 'unchecked'")
    total = sum(f.stat().st_size for f in web.rglob("*") if f.is_file())
    if strict:
        errors += [f"strict: {w}" for w in warns]
    return {"web": str(web), "ok": not errors, "n_checks": len(checks), "n_failed": len([c for c in checks if not c["ok"]]), "errors": errors,
            "warnings": warns, "total_bytes": total, "scenes": names}


# ----------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory (runs/<name>)")
    ap.add_argument("--out", help="output folder (default: <run>/80_web)")
    ap.add_argument("--name", help="scene name (default: run directory name)")
    ap.add_argument("--title")
    ap.add_argument("--kind", default="captured", choices=["captured", "fixture", "public"])
    ap.add_argument("--source", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--attribution", default="")
    ap.add_argument("--captured", default="")
    ap.add_argument("--manual-step", action="append")
    ap.add_argument("--eye-height", type=float)
    ap.add_argument("--video-url", default="")
    ap.add_argument("--places", help="JSON file of named places to merge into scene.json (see the module docstring)")
    ap.add_argument("--default", action="store_true", help="make this the default scene in scenes/index.json")
    ap.add_argument("--no-viewer", action="store_true", help="write only scenes/<name>/ (viewer files already there)")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--zip", help="after a valid bundle, write an offline zip (web folder plus START_HERE.txt) to this path")
    ap.add_argument("--validate", help="validate an existing web folder and exit")
    ap.add_argument("--verify-dir", help="where to write bundle.json (default: <run>/90_verify)")
    ap.add_argument("--json", action="store_true", help="print the validation result as JSON")
    a = ap.parse_args(argv)

    if a.validate:
        res = validate_web(Path(a.validate), a.strict)
        print(json.dumps(res, indent=2) if a.json else summarise(res))
        return 0 if res["ok"] else 1
    if not a.run:
        ap.error("--run or --validate is required")
    run = Path(a.run).resolve()
    name = a.name or run.name
    out = Path(a.out).resolve() if a.out else run / "80_web"
    warnings: list[str] = []
    try:
        scene, files = compose_scene(run, name, a, warnings)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"bundle: {e}", file=sys.stderr)
        return 2
    tmp = out.parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    if out.exists() and a.no_viewer:
        shutil.copytree(out, tmp)
    else:
        tmp.mkdir(parents=True)
        if out.exists():  # keep other scenes already bundled here
            if (out / "scenes").exists():
                shutil.copytree(out / "scenes", tmp / "scenes")
    if not a.no_viewer:
        copy_viewer(tmp, a.video_url or existing_video_url(out))
    sdir = tmp / "scenes" / name
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    for rel, src in files.items():
        dst = sdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (sdir / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    update_index(tmp, name, scene["title"], a.default)
    res = validate_web(tmp, a.strict)
    res["warnings"] = warnings + res["warnings"]
    if res["ok"]:  # stages write to a temp folder and rename it when complete
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        res["web"] = str(out)
    else:
        print(f"bundle: validation failed; temp folder kept at {tmp}", file=sys.stderr)
    if a.zip and res["ok"]:
        res["zip"] = write_zip(out, Path(a.zip), scene["title"])
    verify_dir = Path(a.verify_dir) if a.verify_dir else run / "90_verify"
    try:
        verify_dir.mkdir(parents=True, exist_ok=True)
        (verify_dir / "bundle.json").write_text(json.dumps(res, indent=2) + "\n")
    except OSError:
        pass
    print(json.dumps(res, indent=2) if a.json else summarise(res))
    return 0 if res["ok"] else 1


def write_zip(web: Path, dest: Path, title: str) -> str:
    """Offline copy for the Drive folder: the web folder plus a note with the one command that serves it."""
    import zipfile
    note = (f"Namma Space viewer, offline copy ({title})\n\n"
            "1. Unzip this folder.\n"
            "2. In the folder, run:   python3 -m http.server 8000\n"
            "3. Open http://127.0.0.1:8000/ in Chrome, Edge, Firefox or Safari.\n\n"
            "The page needs a web server because browsers block file:// requests for the scene files. Nothing is fetched from the internet.\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as z:
        z.writestr("START_HERE.txt", note)
        for f in sorted(web.rglob("*")):
            if f.is_file() and f.name != ".DS_Store":
                z.write(f, f.relative_to(web).as_posix(), compress_type=zipfile.ZIP_STORED if f.suffix in (".spz", ".jpg", ".png") else zipfile.ZIP_DEFLATED)
    return f"{dest} ({dest.stat().st_size / 1e6:.1f} MB)"


def summarise(r: dict[str, Any]) -> str:
    lines = [f"{'OK' if r['ok'] else 'INVALID'}: {r['web']}  ({r['n_checks']} checks, {r['n_failed']} failed, {r['total_bytes'] / 1e6:.1f} MB, scenes {r['scenes']})"]
    lines += [f"  error: {e}" for e in r["errors"]]
    lines += [f"  warning: {w}" for w in r["warnings"]]
    if r.get("zip"):
        lines.append(f"  offline zip: {r['zip']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
