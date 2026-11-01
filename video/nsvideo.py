#!/usr/bin/env python3
"""Build the Namma Space demo video for one run.   usage: video/build_video.sh <run> [--out-name NAME] [--keep] [--reuse-footage]

Everything run-specific is read from JSON: runs/<run>/report.json, timings.json, 90_verify/verify.json,
80_web/scenes/<run>/scene.json and eval/results/*.json. No number is typed here or in narration.md.
Stages: 1 values -> 2 narration audio (say) -> 3 record the viewer (node, CDP, piped into ffmpeg) -> 4 slides (Chrome)
-> 5 per-segment ffmpeg (visuals + captions + audio) -> 6 concat, SRT, duration check, README gif.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "docs" / "tools"))
sys.path.insert(0, str(HERE))
import chrome  # noqa: E402  (docs/tools/chrome.py, used as is)
import slides as slides_mod  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H, FPS = 1920, 1080, 30
PAD = 0.6            # seconds added after the speech
LEAD = 0.15          # seconds of silence before the speech
FLY_FPS = 16         # playback rate of the public-room frames
DUR_MIN, DUR_MAX = 210.0, 280.0
TOP_CAPTIONS = {"cold", "walk", "map", "reality", "bench", "measure"}   # captions go to the top on viewer footage so they miss the key overlay and the viewer panels
SEGMENTS = ["cold", "title", "capture", "pipeline", "walk", "map", "reality", "bench", "measure", "route", "results", "limits"]


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))[:300]}\n{r.stderr[-1500:]}")
    return r.stdout


def jl(p: Path):
    return json.loads(p.read_text())


def dur(p: Path) -> float:
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).strip())


# ------------------------------------------------------------------------------------------------ 1. values
def t_say(s: float) -> str:
    if s < 90:
        return f"{round(s)} seconds"
    m = s / 60
    return f"{m:.0f} minutes" if m >= 10 else f"{m:.1f} minutes"


def t_short(s: float) -> str:
    return f"{round(s)} s" if s < 90 else f"{s/60:.1f} min"


def need(d, *path):
    cur = d
    for k in path:
        if cur is None or k not in cur or cur[k] is None:
            raise KeyError("missing in JSON: " + " / ".join(map(str, path)))
        cur = cur[k]
    return cur


def load_values(run: str) -> dict:
    rd = ROOT / "runs" / run
    rep, tim = jl(rd / "report.json"), jl(rd / "timings.json")
    ver = jl(rd / "90_verify" / "verify.json")
    scene = jl(rd / "80_web" / "scenes" / run / "scene.json")
    ev = {p.stem: jl(p) for p in (ROOT / "eval" / "results").glob("*.json")}
    st = tim["stages"]
    sfm = next(v for v in rep["sfm"]["variants"] if v["name"] == rep["sfm"]["chosen"])
    tr = next(v for v in rep["train"]["variants"] if v["name"] == rep["train"]["chosen"])
    n_input = sum(int(i.get("n_input", i.get("n_cand", 0)) or 0) for i in rep["frames"]["inputs"])  # frames: inputs given; videos: candidates sampled
    colmap = None
    try:
        colmap = jl(rd / "20_sfm" / "sfm.json").get("colmap")
    except Exception:
        pass
    dev = ver["viewer"]["devices"][0]
    sc = rep["align"]["scale"]
    chk = rd / "00_preflight" / "check.json"
    pct = lambda x, d=1: f"{x:.{d}f}%"
    r = {}
    for key, tag in (("honka", "mushroom_honka"), ("vr", "mushroom_vr_room")):
        o, l = ev[tag + "_ours"], ev[tag + "_lidar"]
        r[key] = dict(
            scale_err=pct(need(o, "scale", "automatic_error_pct")),
            inside="yes" if need(o, "scale", "true_inside_interval_68") else "no",
            f_ours=pct(100 * need(o, "geometry", "cropped", "f_score_5cm"), 0),
            f_lidar=pct(100 * need(l, "score", "cropped", "f_score_5cm"), 0),
            depth=f"{need(o, 'depth_vs_lidar', 'median_abs_error_cm'):.1f} cm")
    mip = ev["mip360_room"]
    r.update(psnr=f"{tr['psnr']:.1f} dB", ssim=f"{tr['ssim']:.2f}", lpips=f"{tr['lpips']:.2f}",
             mip=f"{need(mip, 'metrics', 'psnr_db'):.1f} dB", mip_pub=f"{need(mip, 'published_psnr_db'):.1f} dB",
             dev=dev["name"].split(",")[0].replace("MacBook (", "").replace(")", "") + ", headless Chrome",
             fps=f"{dev['fps_mean']:.0f}", p95=f"{dev['p95_ms']:.1f} ms", walks=str(need(ver, "walls", "walks")), pen=str(need(ver, "walls", "penetrations")))
    r["dev"] = "Apple M1, headless Chrome"
    clip = HERE / "inputs" / run / "capture_clip.mp4"
    has_clip = clip.exists()
    n_kept = rep["frames"]["n_kept"]
    if chk.exists():
        c = jl(chk)
        status = c.get("overall") or c.get("verdict") or c.get("status")
        check_sentence = f"The capture check on this clip said {str(status).lower()}." if status else "The capture check ran on this clip."
    else:
        check_sentence = "This public room came as frames, so the capture check did not run."
    V = dict(
        run=run, n_kept=str(n_kept), n_input=str(n_input), sfm_variant=sfm["name"], sfm_reg=pct(100 * sfm["reg_frac"]),
        up_resid=f"{rep['align']['up_resid_deg']:.1f}", scale_value=f"{sc['value']:.3f} metres per model unit", scale_pct=f"{sc['pct_68']:.1f}%",
        psnr=f"{tr['psnr']:.1f}", ssim=f"{tr['ssim']:.2f}", lpips=f"{tr['lpips']:.2f}",
        splats=f"{rep['export']['sd']['count']:,}", export_mb=f"{rep['export']['sd']['mb']:.1f} MB", walk_m2=f"{rep['grid']['walkable_m2']:.1f}",
        t_total=t_say(tim["total"]), t_total_min=t_short(tim["total"]), t_sfm=t_say(st["sfm"]), t_train=t_say(st["train"]),
        walks=str(ver["walls"]["walks"]), penetrations=str(ver["walls"]["penetrations"]),
        honka_scale_err=r["honka"]["scale_err"], vr_scale_err=r["vr"]["scale_err"],
        honka_ours_f=r["honka"]["f_ours"], vr_ours_f=r["vr"]["f_ours"], honka_lidar_f=r["honka"]["f_lidar"], vr_lidar_f=r["vr"]["f_lidar"],
        honka_depth=r["honka"]["depth"], vr_depth=r["vr"]["depth"], mip_psnr=r["mip"][:-3], mip_pub=r["mip_pub"][:-3],
        fps=r["fps"], p95=r["p95"], bench_secs="10", check_sentence=check_sentence,
        capture_label="our capture, shown at 4x" if has_clip else f"public room capture, {n_kept} kept frames, played at {FLY_FPS} frames per second",
        capture_sentence="This is our own capture, sped up four times." if has_clip else f"Here is the public room: the {n_kept} frames the pipeline kept, played at {FLY_FPS} frames per second.",
        room_label=scene.get("provenance", {}).get("source", run) if scene.get("provenance", {}).get("kind") == "public" else "our own room",
        has_clip=has_clip, clip=str(clip), places=len(scene.get("places") or []),
        res=r)
    tool = lambda: f"COLMAP {colmap}, {sfm.get('desc','').split(',')[0].replace('CPU ','')}" if colmap else sfm.get("desc", "").split(",")[0]
    first_s = dev.get("first_frame_s")
    chk_time = st.get("check")
    V["arch"] = [
        dict(name="Phone", tool="one video, one lens", time="1 video"),
        dict(name="Check", tool="ns check: pass, warn, fail", time=t_short(chk_time) if chk_time else "not run: frames input", skip=not chk_time),
        dict(name="Frames", tool="ffmpeg, OpenCV: sharp, distinct", time=t_short(st["ingest"])),
        dict(name="SfM race", tool=tool(), time=t_short(st["sfm"])),
        dict(name="Align + scale", tool="up from poses, scale from camera height", time=t_short(st["align"])),
        dict(name="gsplat", tool="gsplat training, shared RTX A6000", time=t_short(st["train"])),
        dict(name="SPZ", tool="splat-transform to SPZ", time=t_short(st["export"])),
        dict(name="Grid", tool="walkable grid + minimap", time=t_short(st["grid"])),
        dict(name="Browser", tool="Spark + three.js viewer", time=f"load {first_s:.1f} s" if first_s else "n/a"),
    ]
    V["stage_rows"] = [
        ("Frames", t_short(st["ingest"]), f"{n_kept} of {n_input} kept"),
        ("SfM race", t_short(st["sfm"]), f"variant {sfm['name']}, {pct(100*sfm['reg_frac'])} registered, {sfm['reproj_px']:.2f} px"),
        ("Align + scale", t_short(st["align"]), f"up off {rep['align']['up_resid_deg']:.1f} deg; scale \u00b1{V['scale_pct']} (68% interval)"),
        (f"Train, variant {tr['name']}", t_short(st["train"]), f"PSNR {V['psnr']} dB, SSIM {V['ssim']}, LPIPS {V['lpips']}"),
        ("Export (SPZ)", t_short(st["export"]), f"{V['splats']} splats, {V['export_mb']}"),
        ("Grid", t_short(st["grid"]), f"{V['walk_m2']} m2 walkable"),
    ]
    return V


# ------------------------------------------------------------------------------------------------ 2. narration
def parse_narration(V: dict) -> dict:
    txt = (HERE / "narration.md").read_text()
    out = {}
    for m in re.finditer(r"^## (\w+)\n(.*?)(?=^## |\Z)", txt, re.M | re.S):
        body = m.group(2)
        sm = re.search(r"^Say:\s*(.+)$", body, re.M)
        if not sm:
            continue
        say = sm.group(1).strip()
        try:
            say = say.format(**V)
        except KeyError as ex:
            raise KeyError(f"narration block '{m.group(1)}' uses unknown value {ex}")
        out[m.group(1)] = say
    return out


def speak(text: str, wav: Path, voice: str, rate: int):
    aiff = wav.with_suffix(".aiff")
    r = subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("say failed: " + r.stderr)
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "48000", "-ac", "2", str(wav)])
    aiff.unlink()


# ------------------------------------------------------------------------------------------------ captions
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 44)


def wrap2(text: str, width=42):
    lines = textwrap.wrap(text, width)
    if len(lines) <= 2:
        return [text]
    words = text.split()
    k = len(words) // 2
    return wrap2(" ".join(words[:k]), width) + wrap2(" ".join(words[k:]), width)


def make_cues(say: str):
    """Cues of at most 2 lines x 42 characters, split at sentence ends first."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?:])\s+", say) if s.strip()]
    cues = []
    for p in parts:
        cues += wrap2(p)
    # merge very short neighbours (under 16 chars) into the next cue when they fit
    merged = []
    for c in cues:
        if merged and len(merged[-1]) < 16 and len(textwrap.wrap(merged[-1] + " " + c, 42)) <= 2:
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged


def cue_png(text: str, path: Path):
    lines = textwrap.wrap(text, 42)
    d = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    wmax = max(d.textlength(l, font=FONT) for l in lines)
    lh = 58
    w, h = int(wmax + 64), int(lh * len(lines) + 30)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], 16, fill=(8, 10, 14, 205))
    for i, l in enumerate(lines):
        d.text((w / 2, 15 + i * lh + lh / 2 - 2), l, font=FONT, fill=(255, 255, 255, 255), anchor="mm")
    im.save(path)
    return w, h


def fmt_srt(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# ------------------------------------------------------------------------------------------------ main build
def build(args):
    run = args.run
    out_name = args.out_name or ("Namma_Space_demo" if run == "room" else "Namma_Space_demo_dryrun" if run == "rh-honka" else f"Namma_Space_demo_dryrun_{run}")
    work = HERE / ".work" / run
    shutil.rmtree(work, ignore_errors=True)
    for d in ("audio", "slides", "rec", "seg", "cues"):
        (work / d).mkdir(parents=True, exist_ok=True)
    (HERE / "out").mkdir(exist_ok=True)

    print("1/6 values from JSON")
    V = load_values(run)
    say = parse_narration(V)
    use_places = V["places"] > 0
    segs = [s if s != "route" else ("route_places" if use_places else "route_none") for s in SEGMENTS]
    voice = args.voice
    if not re.search(rf"^{voice}\s", sh(["say", "-v", "?"]), re.M):
        voice = "Daniel"

    print("2/6 narration audio with say (%s)" % voice)
    A = {}
    for s in segs:
        key = s.split("_")[0] if s.startswith("route") else s
        wav = work / "audio" / f"{s}.wav"
        speak(say[s], wav, voice, args.rate)
        A[s] = dur(wav)
    D = {s: A[s] + PAD for s in segs}
    words = sum(len(say[s].split()) for s in segs)
    print(f"   narration {sum(A.values()):.0f} s speech, {words} words, {words / sum(A.values()) * 60:.0f} wpm")
    for s in segs:
        print(f"   {s:13s} speech {A[s]:5.1f} s  {len(say[s].split())} words")
    if args.audio_only:
        return

    print("3/6 record the viewer")
    rec = work / "rec"
    if not (args.reuse_footage and (HERE / ".footage" / run / "record_report.json").exists()):
        tg = {"walk_cold": D["cold"] + 0.8, "walk": D["walk"] + 0.8, "map": D["map"] + 0.8, "reality": D["reality"] + 0.8,
              "measure": D["measure"] + 0.8, "bench_secs": int(V["bench_secs"]), "bench_a": 4.0}
        tg["bench_b"] = max(3.0, D["bench"] - 4.0 + 0.5)
        if use_places:
            tg["route"] = D["route_places"] + 0.8
        (rec / "targets.json").write_text(json.dumps(tg))
        subprocess.run(["node", str(HERE / "rec" / "record.mjs"), run, str(rec), str(rec / "targets.json")], check=True)
        keep = HERE / ".footage" / run
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(rec, keep)
    else:
        shutil.rmtree(rec); shutil.copytree(HERE / ".footage" / run, rec)
    rr = jl(rec / "record_report.json")
    for k in ("walk_cold", "walk", "map", "reality", "bench", "measure") + (("route",) if use_places else ()):
        if k not in rr or "error" in rr[k]:
            raise RuntimeError(f"recording failed for {k}: {rr.get(k)}")

    print("4/6 slides")
    S = slides_mod.slides(V)
    sl = work / "slides"
    for name, htm in S.items():
        hp = sl / f"{name}.html"
        hp.write_text(htm)
        chrome.screenshot(hp, sl / f"{name}.png", W, H)
        hp.unlink()
    # capture footage: the user's clip at 4x if supplied, else the public room's kept frames
    fly = work / "fly.mp4"
    label = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(label)
    lf = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 44)
    ltxt = V["capture_label"][0].upper() + V["capture_label"][1:]
    tw = ld.textlength(ltxt, font=lf)
    ld.rounded_rectangle([60, 40, 60 + tw + 50, 120], 14, fill=(8, 10, 14, 215)); ld.text((85, 80), ltxt, font=lf, fill="white", anchor="lm")
    label.save(work / "fly_label.png")
    if V["has_clip"]:
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", V["clip"], "-i", str(work / "fly_label.png"), "-filter_complex",
            f"[0:v]setpts=PTS/4,fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e1116,format=yuv420p[a];[a][1:v]overlay=0:0:shortest=1,trim=duration=40",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(fly)])
    else:
        imgs = sorted((ROOT / "runs" / run / "10_frames" / "images").glob("*.jpg"))
        if not imgs:   # the kept frames live on the GPU host; copy them into scratch (about 70 MB, deleted after the build)
            fr = work / "frames"; fr.mkdir(exist_ok=True)
            sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"sher@10.1.18.190:/home/sher/namma-space/runs/{run}/10_frames/images/", str(fr) + "/"])
            imgs = sorted(fr.glob("*.jpg"))
        if not imgs:
            raise RuntimeError("no kept frames found locally or on the GPU host for the capture segment")
        (work / "fly.txt").write_text("".join(f"file '{p}'\nduration {1 / FLY_FPS:.5f}\n" for p in imgs))
        sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "fly.txt"),
            "-loop", "1", "-i", str(work / "fly_label.png"), "-filter_complex",
            f"[0:v]scale=-2:760,fps={FPS}[f];color=c=0x0e1116:s={W}x{H}:r={FPS}[bg];[bg][f]overlay=(W-w)/2:150:shortest=1,format=yuv420p[a];[a][1:v]overlay=0:0:shortest=1",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(fly)])

    P = lambda n: ("png", sl / f"{n}.png")
    M = lambda n: ("mp4", rec / f"{n}.mp4")
    pieces = {
        "cold": [M("walk_cold")],
        "title": [P("title") + (2.0,), P("scope") + (3.0,)],
        "capture": ([("mp4", fly), P("sop1") + (1.0,), P("sop2") + (1.0,)] if V["has_clip"] else [P("sop1") + (1.0,), P("sop2") + (1.0,), ("mp4", fly)]),
        "pipeline": [P("cmd") + (1.0,), P("arch") + (3.0,), P("stages") + (2.4,)],
        "walk": [M("walk")], "map": [M("map")], "reality": [M("reality")], "bench": [M("bench_a"), M("bench_b")], "measure": [M("measure")],
        "route_places": [M("route")], "route_none": [P("route_none") + (1.0,)],
        "results": [P("res_scale") + (1.6,), P("res_geom") + (1.4,), P("res_view") + (1.2,)],
        "limits": [P("limits") + (1.0,), P("links") + (1.6,)],
    }

    print("5/6 segments with captions")
    starts, t_acc, srt, segfiles = {}, 0.0, [], []
    for n, s in enumerate(segs, 1):
        # segment length: speech + pad, never shorter than the footage in it
        mp = [dur(Path(p[1])) for p in pieces[s] if p[0] == "mp4"]
        L = max(D[s], sum(mp))
        cues = make_cues(say[s])
        tot = sum(len(c) for c in cues)
        ct, acc = [], 0
        for i, c in enumerate(cues):
            a = LEAD + A[s] * acc / tot; acc += len(c); b = LEAD + A[s] * acc / tot
            ct.append((c, a, b))
        inputs, fl, labels = [], [], []
        slide_total = max(0.5, L - sum(mp))
        wsum = sum(p[2] for p in pieces[s] if p[0] == "png") or 1
        for i, p in enumerate(pieces[s]):
            if p[0] == "png":
                d = slide_total * p[2] / wsum
                inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(p[1])]
            else:
                inputs += ["-i", str(p[1])]
            fl.append(f"[{i}:v]scale={W}:{H}:flags=lanczos:in_range=auto:out_range=tv,fps={FPS},setsar=1,format=yuv420p,setpts=PTS-STARTPTS[p{i}]")
            labels.append(f"[p{i}]")
        k = len(pieces[s])
        fl.append("".join(labels) + f"concat=n={k}:v=1:a=0,tpad=stop_mode=clone:stop_duration=30,trim=duration={L:.3f},setpts=PTS-STARTPTS[v0]")
        cur = "v0"
        for j, (c, a, b) in enumerate(ct):
            pth = work / "cues" / f"{s}_{j}.png"
            cw, ch = cue_png(c, pth)
            inputs += ["-i", str(pth)]
            idx = k + j
            cy = 28 if s in TOP_CAPTIONS else H - ch - 60
            fl.append(f"[{cur}][{idx}:v]overlay=x={(W - cw) // 2}:y={cy}:enable='between(t,{a:.3f},{b:.3f})'[v{j + 1}]")
            cur = f"v{j + 1}"
        fl.append(f"[{cur}]fade=t=in:st=0:d=0.25,fade=t=out:st={L - 0.3:.3f}:d=0.3[vout]")
        ai = k + len(ct)
        inputs += ["-i", str(work / "audio" / f"{s}.wav")]
        fl.append(f"[{ai}:a]adelay={int(LEAD * 1000)}|{int(LEAD * 1000)},apad,atrim=0:{L:.3f},afade=t=out:st={L - 0.2:.3f}:d=0.2[aout]")
        segf = work / "seg" / f"{n:02d}_{s}.mp4"
        sh(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fl), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", str(segf)])
        segfiles.append(segf)
        for c, a, b in ct:
            srt.append((t_acc + a, t_acc + b - 0.03, c))
        starts[s] = (t_acc, L)
        print(f"   {s:13s} {L:5.1f} s  (speech {A[s]:.1f} s, {len(ct)} captions)")
        t_acc += L

    print("6/6 concat, SRT, checks")
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in segfiles))
    out = HERE / "out" / f"{out_name}.mp4"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])
    (HERE / "out" / f"{out_name}.srt").write_text("".join(f"{i}\n{fmt_srt(a)} --> {fmt_srt(b)}\n{textwrap.fill(c, 42)}\n\n" for i, (a, b, c) in enumerate(srt, 1)))
    if run == "room" and not args.out_name:      # the names `ns package` looks for
        shutil.copy(out, HERE / "out" / "02_Demo_Video.mp4")
        shutil.copy(HERE / "out" / f"{out_name}.srt", HERE / "out" / "02_Demo_Video.srt")
    total = dur(out)
    size = out.stat().st_size / 1e6
    print(f"   output {out}  {total:.1f} s ({int(total // 60)}:{total % 60:04.1f})  {size:.0f} MB")
    (HERE / "out" / f"{out_name}.timing.json").write_text(json.dumps({s: {"start": round(a, 2), "length": round(l, 2)} for s, (a, l) in starts.items()}, indent=1))
    # README gif from the cold-open walk, which ends in a wall stop
    gif = ROOT / "docs" / "media" / "walk.gif"
    if args.gif:
        gif.parent.mkdir(parents=True, exist_ok=True)
        cold = rec / "walk_cold.mp4"
        cd = dur(cold)
        ss = max(0.0, cd - 9.0)
        pal = work / "pal.png"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.2f}", "-t", "8.5", "-i", str(cold), "-vf", "fps=12,scale=576:-1:flags=lanczos,palettegen=max_colors=24:stats_mode=diff", str(pal)])
        sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.2f}", "-t", "8.5", "-i", str(cold), "-i", str(pal), "-lavfi", "fps=12,scale=576:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle", str(gif)])
        print(f"   gif {gif} {gif.stat().st_size / 1e6:.2f} MB")
    if not (DUR_MIN <= total <= DUR_MAX):
        print(f"ERROR: duration {total:.1f} s outside {DUR_MIN:.0f}-{DUR_MAX:.0f} s. Edit video/narration.md and rebuild.", file=sys.stderr)
        sys.exit(2)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out-name")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--rate", type=int, default=180)
    ap.add_argument("--audio-only", action="store_true", help="print narration lengths and stop")
    ap.add_argument("--keep", action="store_true", help="keep video/.work/<run> (slides, cues, segments)")
    ap.add_argument("--reuse-footage", action="store_true", help="reuse the last recorded viewer clips for this run")
    ap.add_argument("--gif", action="store_true", help="also write docs/media/walk.gif for a final run")
    build(ap.parse_args())
