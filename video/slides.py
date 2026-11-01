"""Slide HTML for the demo video. Every number comes in through the dict `V` built by nsvideo.py. 1920x1080, dark, large type."""
from __future__ import annotations

import html

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1920px;height:1080px;background:#0e1116;color:#f3f5f8;font-family:-apple-system,'Helvetica Neue',Helvetica,Arial,sans-serif;overflow:hidden}
.s{position:relative;width:1920px;height:1080px;padding:90px 120px 250px 120px}
.eyebrow{font-size:40px;letter-spacing:.06em;text-transform:uppercase;color:#59d6b0;font-weight:700;margin-bottom:26px}
h1{font-size:84px;line-height:1.08;font-weight:800;margin-bottom:34px}
h2{font-size:64px;line-height:1.12;font-weight:800;margin-bottom:34px}
p,li,td,th,.t{font-size:44px;line-height:1.3}
ul{list-style:none}
li{padding:12px 0 12px 54px;position:relative}
li:before{content:'';position:absolute;left:6px;top:32px;width:18px;height:18px;border-radius:50%;background:#59d6b0}
.dim{color:#aab3c0}
.acc{color:#59d6b0}
.warn{color:#ffb454}
.cols{display:flex;gap:80px}.cols>div{flex:1}
.card{background:#171c24;border:2px solid #2b3442;border-radius:22px;padding:34px 40px}
table{border-collapse:collapse;width:100%}
th,td{padding:16px 22px;text-align:left;border-bottom:2px solid #2b3442;vertical-align:top}
th{color:#aab3c0;font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.big{font-size:150px;font-weight:800;line-height:1}
.mono{font-family:Menlo,Monaco,'Courier New',monospace}
.term{background:#05070a;border:2px solid #2b3442;border-radius:18px;padding:40px 48px;font-size:72px;color:#e8f0ff}
.term .p{color:#59d6b0}
.foot{position:absolute;left:120px;bottom:300px;font-size:40px;color:#8d97a6}
.arch{display:grid;grid-template-columns:repeat(5,1fr);gap:46px 38px}
.box{background:#171c24;border:2px solid #2b3442;border-radius:20px;padding:20px 22px;position:relative;min-height:262px}
.box b{display:block;font-size:42px;margin-bottom:6px}
.box .tool{font-size:30px;line-height:1.2;color:#aab3c0;min-height:76px;display:block}
.box .tm{font-size:42px;font-weight:800;color:#59d6b0;margin-top:8px;display:block}
.box.skip .tm{color:#ffb454;font-size:34px}
.box:after{content:'\\2192';position:absolute;right:-36px;top:42%;font-size:44px;color:#59d6b0}
.box.last:after,.box.rowend:after{content:''}
.tag{display:inline-block;font-size:34px;font-weight:700;padding:2px 16px;border-radius:10px;background:#59d6b0;color:#06100c;vertical-align:4px;margin-left:14px}
"""


def e(x) -> str:
    return html.escape(str(x))


def page(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body><div class='s'>{body}</div></body></html>"


def slides(V: dict) -> dict:
    S: dict[str, str] = {}
    S["title"] = page(f"""
<div style='padding-top:20px'><div class='eyebrow'>Techfest IIT Bombay &middot; Round 1</div>
<div style='font-size:150px;font-weight:800;line-height:1.05;margin-bottom:40px'>Namma Space</div>
<p style='font-size:60px;line-height:1.25'>One phone video.<br>One command.<br><span class='acc'>A room you can walk through.</span></p></div>
<div class='foot'>{e(V['foot'])}</div>""")
    S["scope"] = page(f"""
<div class='eyebrow'>Scope</div><h2>One indoor room, from one phone video</h2>
<div class='cols'><div class='card'><p class='acc' style='font-weight:700'>In this build</p><ul>
<li>Phone camera only</li><li>One indoor room</li><li>Walk it in a browser</li><li>Open tools only</li></ul></div>
<div class='card'><p class='warn' style='font-weight:700'>Not built</p><ul>
<li>Multiple floors</li><li>WebXR AR</li><li>Plan registration</li></ul></div></div>
<div class='foot'>This video: dry run on {e(V['room_label'])}</div>""")
    S["sop1"] = page(f"""
<div class='eyebrow'>Capture routine &middot; setup</div><h2>One lens. Locked. Slow.</h2>
<ul><li>One lens (1x), landscape, 4K at 30 fps</li><li>Lock focus and exposure. HDR and stabilisation off</li>
<li>Flickering lights: film at 25 frames per second</li><li>Hold at chest height. Turn your body, not your wrist</li></ul>""")
    S["sop2"] = page(f"""
<div class='eyebrow'>Capture routine &middot; three passes</div><h2>Walls, corners, floor</h2>
<ul><li><b>Pass 1</b> Outer loop along the walls, camera across the room</li>
<li><b>Pass 2</b> Inner loop facing the walls. Tilt up at each corner</li>
<li><b>Pass 3</b> Low pass around the furniture</li>
<li>Second space: cross the door slowly, both ways</li></ul>""")
    S["cmd"] = page(f"""
<div class='eyebrow'>The pipeline</div><h2>One command</h2>
<div class='term mono'><span class='p'>$</span> ./ns run video.mp4</div>
<ul style='margin-top:36px'><li>Runs on a shared RTX A6000 GPU</li><li>{e(V['room_word'])}: {e(V['t_total_min'])} in total</li><li>Every stage time on the next slides comes from the run's own records</li></ul>""")
    boxes = V["arch"]
    bx = []
    for i, b in enumerate(boxes):
        cls = "box" + (" skip" if b.get("skip") else "") + (" last" if i == len(boxes) - 1 else "") + (" rowend" if i == 4 else "")
        bx.append(f"<div class='{cls}'><b>{e(b['name'])}</b><span class='tool'>{e(b['tool'])}</span><span class='tm'>{e(b['time'])}</span></div>")
    S["arch"] = page(f"<div class='eyebrow'>Pipeline &middot; measured time per stage</div><h2>Phone to browser</h2><div class='arch'>{''.join(bx)}</div>")
    rows = "".join(f"<tr><td>{e(a)}</td><td class='n acc'>{e(b)}</td><td class='dim'>{e(c)}</td></tr>" for a, b, c in V["stage_rows"])
    S["stages"] = page(f"<div class='eyebrow'>Pipeline &middot; this run</div><h2>What each stage produced</h2><table>{rows}</table>")
    # results
    r = V["res"]
    S["res_scale"] = page(f"""
<div class='eyebrow'>Results &middot; scale and depth</div><h2>Scale check on two public rooms</h2>
<table><tr><th></th><th class='n'>honka</th><th class='n'>vr_room</th></tr>
<tr><td>Automatic scale error</td><td class='n acc'>{e(r['honka']['scale_err'])}</td><td class='n acc'>{e(r['vr']['scale_err'])}</td></tr>
<tr><td>True scale inside our 68% interval</td><td class='n acc'>{e(r['honka']['inside'])}</td><td class='n acc'>{e(r['vr']['inside'])}</td></tr>
<tr><td>Depth error vs LiDAR, median</td><td class='n acc'>{e(r['honka']['depth'])}</td><td class='n acc'>{e(r['vr']['depth'])}</td></tr></table>
<p class='dim' style='margin-top:26px;font-size:40px'>The scale here is estimated from camera height. The room's true scale came from the dataset.</p>""")
    S["res_geom"] = page(f"""
<div class='eyebrow'>Results &middot; geometry</div><h2>Against a Faro laser scan: F-score at 5 cm</h2>
<table><tr><th></th><th class='n'>honka</th><th class='n'>vr_room</th></tr>
<tr><td>Phone only (ours)</td><td class='n acc'>{e(r['honka']['f_ours'])}</td><td class='n acc'>{e(r['vr']['f_ours'])}</td></tr>
<tr><td>iPhone LiDAR fusion</td><td class='n acc'>{e(r['honka']['f_lidar'])}</td><td class='n acc'>{e(r['vr']['f_lidar'])}</td></tr></table>
<p style='margin-top:26px;font-size:40px' class='warn'>We did not match LiDAR.</p>
<p class='dim' style='font-size:38px;margin-top:10px'>Both use the same TSDF fusion and the same crop. Before scoring, our camera path was similarity-fitted (rotation, offset, scale) to the dataset's camera path.</p>""")
    S["res_view"] = page(f"""
<div class='eyebrow'>Results &middot; quality, trainer and viewer</div><h2>This run, held-out photos, and the browser</h2>
<table><tr><td>Held-out PSNR / SSIM / LPIPS ({e(V['run'])})</td><td class='n acc'>{e(r['psnr'])} / {e(r['ssim'])} / {e(r['lpips'])}</td></tr>
<tr><td>Trainer on Mip-NeRF 360 room, PSNR (published)</td><td class='n acc'>{e(r['mip'])} ({e(r['mip_pub'])})</td></tr>
<tr><td>Viewer, {e(r['dev'])}: fps, p95 frame</td><td class='n acc'>{e(r['fps'])} fps, {e(r['p95'])}</td></tr>
<tr><td>Wall test: walks, positions inside a wall</td><td class='n acc'>{e(r['walks'])}, {e(r['pen'])}</td></tr></table>
<p class='dim' style='margin-top:20px;font-size:38px'>Browser numbers: Apple M1, headless Chrome only. Other hardware is unverified.</p>""")
    S["limits"] = page(f"""
<div class='eyebrow'>Limits</div><h2>What this does not do</h2>
<div class='cols'><div class='card'><p class='warn' style='font-weight:700'>Not built</p><ul><li>Multiple floors</li><li>WebXR AR</li><li>Plan registration</li></ul></div>
<div class='card'><p class='warn' style='font-weight:700'>Estimated or untested</p><ul><li>{e(V['scale_limit'])}</li><li>Other hardware: untested</li></ul></div></div>""")
    S["links"] = page(f"""
<div class='eyebrow'>Links</div><h2>Submission slots</h2>
<table class='mono'><tr><td class='dim'>Viewer link</td><td>[HOSTED LINK: added after approval to publish]</td></tr>
<tr><td class='dim'>Team ID</td><td>[TEAM ID]</td></tr>
<tr><td class='dim'>Repository</td><td>[REPO LINK]</td></tr>
<tr><td class='dim'>PoC PDF</td><td>[POC PDF LINK]</td></tr>
<tr><td class='dim'>Abstract PDF</td><td>[ABSTRACT PDF LINK]</td></tr></table>""")
    S["route_none"] = page("""
<div style='padding-top:240px'><div class='eyebrow'>Round 2 preview</div><h1>Search and routes</h1>
<p class='dim' style='font-size:56px'>In the final cut.</p></div>""")
    return S
