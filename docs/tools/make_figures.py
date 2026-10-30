#!/usr/bin/env python3
"""Make the figures for the documents from a run directory: usage make_figures.py <run>

Writes docs/build/<run>/figures/:
  architecture.svg / architecture.png  the pipeline with measured values (needs values.json from doc_values.py)
  pairs.jpg    three held-out photos beside our render at the same pose (lowest, middle, highest PSNR)
  plan.png     the walkable plan from the same grid the viewer uses, with the tour path
  twin.jpg     one render from inside the twin (the best held-out pair's render)
  frame.jpg    one input frame (only if the run keeps its frames)
Needs Pillow. The PNG of the architecture needs Playwright's headless Chrome (see chrome.py).
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] + FONTS if bold else FONTS
    for n in names:
        if Path(n).exists():
            return ImageFont.truetype(n, size)
    return ImageFont.load_default()


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def esc(t: str) -> str:
    return html.escape(t, quote=True)


# ---------------------------------------------------------------- architecture
BOXES = [
    # key, title, tool, extra value key(s), where
    ("phone", "Phone video", "one lens, locked exposure", "{capture}", "phone"),
    ("check", "Capture check", "ffprobe + sharpness", "verdict: {check_verdict}", "mac"),
    ("frames", "Frames", "ffmpeg, OpenCV blur filter", "{frames_kept} kept of {frames_cand}", "gpu"),
    ("sfm", "Pose race", "COLMAP 4.2.1 + LightGlue", "chose {sfm_name}: {sfm_reg} registered", "gpu"),
    ("align", "Level + scale", "gravity, floor, height cue", "{scale_interval}", "gpu"),
    ("train", "Train splats", "gsplat (MCMC), PyTorch", "{n_gs} splats (variant {train_name})", "gpu"),
    ("export", "Clean + export", "splat-transform, SPZ", "{sd_count} splats, {sd_mb} MB", "gpu"),
    ("grid", "Walk grid", "numpy, 5 cm cells", "{walkable} m² walkable", "gpu"),
    ("viewer", "Browser viewer", "three.js + Spark", "{dev_fps} fps (M1 only)", "web"),
]
TIME = {"check": None, "frames": "t_ingest", "sfm": "t_sfm", "align": "t_align", "train": "t_train",
        "export": "t_export", "grid": "t_grid"}
FILL = {"phone": "#f4f4f4", "mac": "#dbe9f8", "gpu": "#fbe3c4", "web": "#d9efd9"}


def architecture_svg(v: dict) -> str:
    w, h, bw, bh = 1000, 520, 172, 150
    xs = [14 + i * 196 for i in range(5)]
    pos = [(xs[i], 40) for i in range(5)] + [(xs[i] + 98, 250) for i in range(4)]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           'font-family="Helvetica, Arial, sans-serif">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
           '<path d="M0,0 L0,6 L8,3 z" fill="#222"/></marker></defs>']
    for i, ((key, title, tool, val, where), (x, y)) in enumerate(zip(BOXES, pos)):
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{FILL[where]}" stroke="#222" stroke-width="1.5"/>')
        out.append(f'<text x="{x + 10}" y="{y + 28}" font-size="19" font-weight="bold" fill="#111">{esc(title)}</text>')
        out.append(f'<text x="{x + 10}" y="{y + 52}" font-size="12" fill="#333">{esc(tool)}</text>')
        text = val.format(**v)
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 24:
                lines.append(cur); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines.append(cur)
        for j, ln in enumerate(lines[:3]):
            out.append(f'<text x="{x + 10}" y="{y + 82 + 18 * j}" font-size="15" fill="#111">{esc(ln)}</text>')
        tk = TIME.get(key)
        if tk:
            out.append(f'<text x="{x + 10}" y="{y + bh - 12}" font-size="17" font-weight="bold" fill="#8a1c1c">{esc(v[tk])}</text>')
        out.append(f'<text x="{x + bw - 10}" y="{y + 20}" font-size="14" text-anchor="end" fill="#555">{i + 1}</text>')
    for i in range(4):  # row 1 arrows
        x0 = pos[i][0] + bw
        out.append(f'<path d="M{x0 + 2} {pos[i][1] + 75} H{pos[i + 1][0] - 4}" stroke="#222" stroke-width="2" marker-end="url(#a)"/>')
    # elbow from box 5 down to box 6
    x5, y5 = pos[4][0] + bw / 2, pos[4][1] + bh
    out.append(f'<path d="M{x5} {y5 + 2} V{y5 + 25} H{pos[5][0] + bw / 2} V{pos[5][1] - 4}" fill="none" stroke="#222" stroke-width="2" marker-end="url(#a)"/>')
    for i in range(5, 8):
        x0 = pos[i][0] + bw
        out.append(f'<path d="M{x0 + 2} {pos[i][1] + 75} H{pos[i + 1][0] - 4}" stroke="#222" stroke-width="2" marker-end="url(#a)"/>')
    # legend, in one row under the second row of boxes
    lx, ly = 14, 428
    for j, (k, label) in enumerate([("phone", "phone"), ("mac", "Mac (check, bundle, test)"), ("gpu", "shared RTX A6000 host (Docker)"), ("web", "any static web host")]):
        x = lx + j * 235
        out.append(f'<rect x="{x}" y="{ly}" width="16" height="16" fill="{FILL[k]}" stroke="#222"/>')
        out.append(f'<text x="{x + 24}" y="{ly + 14}" font-size="14" fill="#111">{esc(label)}</text>')
    out.append(f'<text x="14" y="{h - 30}" font-size="14" fill="#333">Red: measured stage time of this run. '
               f'Total {esc(v["t_total_min"])} (GPU stages {esc(v["t_gpu_min"])}).</text>')
    out.append(f'<text x="14" y="{h - 10}" font-size="14" fill="#333">Splats are for looks; the grid is for walking and the minimap.</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------- pairs and plan
def heldout_pairs(run: str):
    d = ROOT / f"runs/{run}/60_export"
    ho = json.loads((d / "heldout/heldout.json").read_text())
    items = []
    for h in ho:
        render = d / "parity" / f'{h["id"]}.png'
        photo = d / "heldout" / Path(h["photo"]).name
        if render.exists() and photo.exists() and h.get("psnr"):
            items.append((h["psnr"], h["id"], photo, render))
    items.sort()
    return items


def make_pairs(run: str, out: Path) -> list:
    items = heldout_pairs(run)
    if not items:
        return []
    chosen = [items[0], items[len(items) // 2], items[-1]] if len(items) >= 3 else items
    labels = [f"lowest of {len(items)}", "middle", "highest"] if len(items) >= 3 else [""] * len(items)
    th, gap, pad = 560, 14, 12
    fs = 30
    panels = []
    for (psnr, hid, photo, render), lab in zip(chosen, labels):
        a, b = Image.open(photo).convert("RGB"), Image.open(render).convert("RGB")
        w = round(a.width * th / a.height)
        panels.append((a.resize((w, th), Image.LANCZOS), b.resize((w, th), Image.LANCZOS), psnr, hid, lab))
    width = sum(2 * p[0].width + 6 for p in panels) + gap * (len(panels) - 1) + 2 * pad
    sheet = Image.new("RGB", (width, th + 2 * pad + 112), "white")
    dr = ImageDraw.Draw(sheet)
    x = pad
    for a, b, psnr, hid, lab in panels:
        sheet.paste(a, (x, pad)); sheet.paste(b, (x + a.width + 6, pad))
        dr.text((x, pad + th + 8), "held-out photo", font=font(fs), fill="#222")
        dr.text((x + a.width + 6, pad + th + 8), "our render, same pose", font=font(fs), fill="#222")
        dr.text((x, pad + th + 8 + fs + 10), f"frame {hid}: PSNR {psnr:.1f} dB" + (f" ({lab})" if lab else ""), font=font(fs, True), fill="#111")
        x += 2 * a.width + 6 + gap
    sheet.save(out / "pairs.jpg", quality=88)
    best = chosen[-1]
    Image.open(best[3]).convert("RGB").save(out / "twin.jpg", quality=90)
    return [c[1] for c in chosen]


def make_plan(run: str, out: Path) -> None:
    g = load(f"runs/{run}/70_grid/grid.json")
    scene = load(f"runs/{run}/80_web/scenes/{run}/scene.json")
    im = Image.open(ROOT / f"runs/{run}/70_grid/grid.png").convert("L")
    k = max(3, min(8, 900 // max(im.width, 1)))
    cell, (ox, oz), vals = g["cell"], g["origin_xz"], g["values"]
    rgb = Image.new("RGB", im.size)
    px, dst = im.load(), rgb.load()
    for yy in range(im.height):
        for xx in range(im.width):
            p = px[xx, yy]
            dst[xx, yy] = (47, 158, 140) if p == vals["walkable"] else ((176, 176, 176) if p == vals["floor"] else (32, 34, 40))
    rgb = rgb.resize((im.width * k, im.height * k), Image.NEAREST)
    pad = 20
    sheet = Image.new("RGB", (rgb.width + 2 * pad, rgb.height + 2 * pad + 56), "white")
    sheet.paste(rgb, (pad, pad))
    d = ImageDraw.Draw(sheet)

    def to_px(x, z):
        return pad + (x - ox) / cell * k, pad + (z - oz) / cell * k  # row0 = min z, as in grid.json

    tour = g.get("tour") or []
    pts = [to_px(t["position"][0], t["position"][2]) for t in tour]
    if len(pts) > 1:
        d.line(pts, fill="white", width=3)
    for h in scene.get("heldout", []):
        cx, cy = to_px(h["position"][0], h["position"][2])
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline="#f2b134", width=3)
    sx, sy = to_px(g["spawn"]["position"][0], g["spawn"]["position"][2])
    d.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], fill="#e04040", outline="white", width=2)
    bar = 1.0 / cell * k
    by = rgb.height + pad + 22
    d.line([(pad, by), (pad + bar, by)], fill="black", width=4)
    d.text((pad + bar + 10, by - 11), "1 m", font=font(20), fill="black")
    d.text((pad + bar + 70, by - 18), "teal: walkable   grey: floor, not walkable   dark: blocked", font=font(15), fill="#333")
    d.text((pad + bar + 70, by + 2), "red: spawn   white line: tour   gold rings: held-out photo poses", font=font(15), fill="#333")
    sheet.save(out / "plan.png")


def make_frame(run: str, out: Path) -> None:
    imgs = sorted((ROOT / f"runs/{run}/10_frames/images").glob("*.jpg"))
    if imgs:
        Image.open(imgs[len(imgs) // 3]).convert("RGB").save(out / "frame.jpg", quality=88)


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "rh-honka"
    out = ROOT / "docs/build" / run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    values = json.loads((ROOT / "docs/build" / run / "values.json").read_text())
    (out / "architecture.svg").write_text(architecture_svg(values))
    ids = make_pairs(run, out)
    make_plan(run, out)
    make_frame(run, out)
    try:
        from chrome import screenshot
        wrap = out / ".arch.html"
        wrap.write_text(f'<body style="margin:0">{(out / "architecture.svg").read_text()}</body>')
        screenshot(wrap, out / "architecture.png", 1000, 520, scale=2.0)
        wrap.unlink()
    except Exception as e:  # PNG is only for the video; the PDF uses the SVG
        print("architecture.png skipped:", e)
    print(f"figures for {run} in {out}; pairs: {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
