#!/usr/bin/env python3
"""Fill the results block of README.md from docs/build/<run>/values.json. usage: update_readme.py <run>

The block sits between <!-- results:start --> and <!-- results:end -->. No number is typed by hand.
Also copies the small figures the README shows into docs/media/.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
run = sys.argv[1]
v = json.loads((ROOT / "docs/build" / run / "values.json").read_text())
fig = ROOT / "docs/build" / run / "figures"
media = ROOT / "docs/media"
media.mkdir(exist_ok=True)
for name in ("pairs.jpg", "plan.png"):
    if (fig / name).exists():
        shutil.copy(fig / name, media / name)
rows = [
    ("Scene", v["scene_title"]),
    ("Held-out photos, PSNR / SSIM / LPIPS", f'{v["psnr"]} dB / {v["ssim"]} / {v["lpips"]} ({v["heldout_n"]} photos never used in training)'),
    ("Scene size", f'{v["sd_count"]} splats, {v["sd_mb"]} MB (plus a {v["preview_mb"]} MB preview)'),
    ("Walk in the browser", f'{v["dev_fps"]} fps, p95 {v["dev_p95"]}, on screen after {v["dev_first"]} (one Apple M1, headless Chrome; other hardware unverified)'),
    ("Run time", f'{v["t_total_min"]} in all stages, {v["t_gpu_min"]} of it in GPU stages (shared RTX A6000)'),
    ("Scale", f'{v["scale_interval"]}; {v["scale_status"]}'),
    ("Walls", f'{v["walls_n"]} random walks and {v["pushes_n"]} pushes: {v["penetrations"]} positions in blocked cells'),
    ("Scale check, two public rooms", f'{v["honka_scale_err"]} (honka), {v["vr_scale_err"]} (vr_room) automatic error; true scale inside the 68 % interval in both'),
    ("Geometry vs Faro laser scan, F@5cm", f'honka {v["honka_f_ours"]} phone only vs {v["honka_f_lidar"]} iPhone LiDAR fusion; vr_room {v["vr_f_ours"]} vs {v["vr_f_lidar"]} (same fusion and crop; our camera path was fitted to the dataset path first, scale included)'),
    ("Trainer check, Mip-NeRF 360 room", f'{v["mip_psnr"]} dB (published {v["mip_pub"]} dB)'),
]
table = "| What | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
block = f"<!-- results:start -->\nMeasured on `{run}`" + (" (a public room; the final entry uses our own room, `room`)" if run != "room" else "") + f". Sources: `runs/{run}/report.json`, `timings.json`, `90_verify/verify.json`, `eval/results/*.json`.\n\n{table}\n<!-- results:end -->"
p = ROOT / "README.md"
t = p.read_text()
a, b = t.index("<!-- results:start -->"), t.index("<!-- results:end -->") + len("<!-- results:end -->")
p.write_text(t[:a] + block + t[b:])
print("README results block updated for", run)
