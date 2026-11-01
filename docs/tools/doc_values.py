#!/usr/bin/env python3
"""Turn a run record and the evaluation results into flat display strings for the documents.

usage: doc_values.py <run> [--team-id ID] [--viewer-url URL] [--repo-url URL] [--video-url URL]
                           [--out docs/build/<run>/values.json]

Every number comes from a JSON file. Rounding and units are applied here, once.
Slots that cannot exist yet (team ID, hosted link) print as marked placeholders, never as fake URLs.
Standard library only.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL_RUN = "room"
SLOT_TEAM = "[TEAM ID: add before submitting]"
SLOT_VIEWER = "[HOSTED LINK: added after approval to publish]"
SLOT_REPO = "[REPOSITORY LINK: added after approval to publish]"
SLOT_VIDEO = "[VIDEO LINK: added when the video is uploaded]"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def pick(variants: list, name: str) -> dict:
    return next(v for v in variants if v["name"] == name)


def mins(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"


def hms(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m} min {s:02d} s"


def slot(text: str, is_slot: bool) -> str:
    esc = html.escape(text)
    if is_slot:
        return f'<span class="slot">{esc}</span>'
    # real links are clickable in the PDF
    return f'<a href="{esc}">{esc}</a>' if text.startswith(("http://", "https://")) else esc


def room_eval(tag: str) -> dict:
    ours = load(f"eval/results/mushroom_{tag}_ours.json")
    lidar = load(f"eval/results/mushroom_{tag}_lidar.json")
    f_o = ours["geometry"]["cropped"]["f_score_5cm"]
    f_l = lidar["score"]["cropped"]["f_score_5cm"]
    return {
        "scale_err": f'{ours["scale"]["automatic_error_pct"]:.1f} %',
        "inside": "yes" if ours["scale"]["true_inside_interval_68"] else "no",
        "fit_scale": f'{ours["scale"]["fit_scale_aligned_to_dataset"]:.3f}',
        "f_ours": f"{f_o:.2f}",
        "f_lidar": f"{f_l:.2f}",
        "f_ratio": f"{100 * f_o / f_l:.1f} %",
        "acc_ours": f'{ours["geometry"]["cropped"]["accuracy_m"] * 100:.1f}',
        "acc_lidar": f'{lidar["score"]["cropped"]["accuracy_m"] * 100:.1f}',
        "comp_ours": f'{ours["geometry"]["cropped"]["completeness_m"] * 100:.1f}',
        "comp_lidar": f'{lidar["score"]["cropped"]["completeness_m"] * 100:.1f}',
        "depth_med": f'{ours["depth_vs_lidar"]["median_abs_error_cm"]:.1f}',
        "depth_5": f'{ours["depth_vs_lidar"]["within_5cm_frac"] * 100:.0f} %',
        "f_ours_raw": f_o, "f_lidar_raw": f_l,
        "fusion": ours["fusion"]["method"],
    }


def public_scores(run: str) -> dict:
    rep = load(f"runs/{run}/report.json")
    tv = pick(rep["train"]["variants"], rep["train"]["chosen"])
    return {"psnr": f'{tv["psnr"]:.2f}', "ssim": f'{tv["ssim"]:.3f}', "lpips": f'{tv["lpips"]:.3f}'}



LICENCE_SHOW = ["COLMAP", "gsplat", "PyTorch", "Open3D", "splat-transform", "Spark", "three.js",
                "ffmpeg", "opencv-python-headless"]


def licence_html() -> dict:
    """Licence rows, rejected list and disclosures, read from docs/licenses.yaml (JSON inside)."""
    y = json.loads((ROOT / "docs/licenses.yaml").read_text())
    by = {x["name"]: x for x in y["product"]}
    rows = "".join(f'<tr><td>{html.escape(n)}</td><td>{html.escape(by[n]["license"])}</td><td>{html.escape(by[n]["role"])}</td></tr>'
                   for n in LICENCE_SHOW if n in by)
    rej = "; ".join(html.escape(x["name"]) for x in y["rejected"])
    dis = "".join(f"<li>{html.escape(d)}</li>" for d in y["disclosures"])
    ev = "".join(f'<li>{html.escape(x["name"])}: {html.escape(x["license"])}</li>' for x in y["evaluation"] if "Berlin" not in x["name"])
    return {"licence_rows_html": rows, "rejected_text": rej, "disclosures_html": dis, "eval_credits_html": ev}


def build(run: str, args: argparse.Namespace) -> dict:
    rep = load(f"runs/{run}/report.json")
    tim = load(f"runs/{run}/timings.json")
    ver = load(f"runs/{run}/90_verify/verify.json")
    scene = load(f"runs/{run}/80_web/scenes/{run}/scene.json")
    sfm_v = pick(rep["sfm"]["variants"], rep["sfm"]["chosen"])
    trn_v = pick(rep["train"]["variants"], rep["train"]["chosen"])
    st = tim["stages"]
    final = run == FINAL_RUN
    dev = ver["viewer"]["devices"]
    walk, bench = dev[0], dev[-1]
    bench_test = next((t for t in ver["tests"] if t["id"] == "X3"), {})
    bm = bench_test.get("metrics") or {}

    probe = ((rep["input"].get("check") or {}).get("probe") or {})
    vids = [i for i in (rep["frames"].get("inputs") or []) if i.get("kind") == "video"]
    if len(vids) > 1:
        # several takes joined into one reconstruction
        tot = sum(float(i.get("duration_s") or 0) for i in vids)
        i0 = vids[0]
        capture = f'{len(vids)} videos, {hms(tot)} in total, {i0["width"]}x{i0["height"]} at {float(i0["avg_fps"]):.0f} fps'
    elif probe.get("duration_s"):
        capture = f'{hms(probe["duration_s"])} at {probe["display_width"]}x{probe["display_height"]}, {probe["avg_fps"]:.0f} fps'
    elif vids:
        i0 = vids[0]
        capture = f'{hms(float(i0.get("duration_s") or 0))} at {i0["width"]}x{i0["height"]}, {float(i0["avg_fps"]):.0f} fps'
    else:
        sizes = ", ".join(f"{k}" for k in rep["frames"]["inputs"][0]["sizes"])
        capture = f'frame folder ({rep["frames"]["n_cand"]} frames, {sizes} px)'

    scale = rep["align"]["scale"]
    cues = "; ".join(c["note"] for c in scale["cues"] if c.get("used"))
    checked = scale["checked_against"]
    chk = scale.get("check") or {}
    if checked and scale.get("user_factor") and chk.get("auto_error_pct") is not None:
        # a rescale checked against measured or rough sizes (ns rescale): say what was wrong and what changed
        scale_status = (f"estimated from camera height, then corrected x{float(scale['user_factor']):.2f}: the automatic estimate "
                        f"was {float(chk['auto_error_pct']):+.0f}% against the {checked} ({chk.get('note', 'not a tape')})")
    else:
        scale_status = f"checked against {checked}" if checked else "estimated, not checked by tape"

    gpu_s = sum(st.get(k, 0) for k in ("sfm", "train", "geometry", "export"))
    honka, vr = room_eval("honka"), room_eval("vr_room")
    mip = load("eval/results/mip360_room.json")
    places = scene.get("places") or []

    v = {
        "run": run,
        "mode": "final" if final else "dryrun",
        "banner_style": "display:none" if final else "",
        "dryrun_banner": "" if final else f"Dry run on a public room: MuSHRoom {run.replace('rh-', '')} (CC BY 4.0). The final submission is built from our own room with the same command.",
        "team_id_html": slot(args.team_id or SLOT_TEAM, not args.team_id),
        "viewer_link_html": slot(args.viewer_url or SLOT_VIEWER, not args.viewer_url),
        "repo_link_html": slot(args.repo_url or SLOT_REPO, not args.repo_url),
        "video_link_html": slot(args.video_url or SLOT_VIDEO, not args.video_url),
        "viewer_pending": "" if args.viewer_url else "No hosted link exists yet. It goes here after we approve publishing.",
        "scene_title": scene.get("title", run),
        "provenance": scene.get("provenance", {}).get("source", ""),
        # capture and frames
        "capture": capture,
        "frames_kept": f'{rep["frames"]["n_kept"]}',
        "frames_cand": f'{rep["frames"]["n_cand"]}',
        "blur_drops": f'{rep["frames"]["dropped"]["blur"]}',
        "check_verdict": (rep["input"].get("check") or {}).get("verdict", "not run" if vids else "not run (frame folder)"),
        # SfM
        "sfm_name": rep["sfm"]["chosen"],
        "sfm_desc": sfm_v["desc"],
        "sfm_reg": f'{sfm_v["reg_frac"] * 100:.1f} %',
        "sfm_reproj": f'{sfm_v["reproj_px"]:.2f} px',
        "sfm_n_variants": f'{len(rep["sfm"]["variants"])}',
        "sfm_why": rep["sfm"]["why"],
        # align and scale
        "up_resid": f'{rep["align"]["up_resid_deg"]:.1f}°',
        "scale_interval": f'±{scale["pct_68"]:.0f} % (68 % interval)',
        "scale_cue": cues,
        "scale_status": scale_status,
        # training
        "train_name": rep["train"]["chosen"],
        "train_why": rep["train"]["why"],
        "psnr": f'{trn_v["psnr"]:.2f}', "ssim": f'{trn_v["ssim"]:.3f}', "lpips": f'{trn_v["lpips"]:.3f}',
        "cc_psnr": f'{trn_v["cc_psnr"]:.2f}',
        "n_gs": f'{trn_v["n_gs"] / 1e6:.1f} M',
        "peak_gpu": f'{trn_v["peak_gpu_gb"]:.1f}',
        "heldout_n": f'{scene["metrics"]["heldout_n"]}',
        # export and grid
        "sd_count": f'{rep["export"]["sd"]["count"]:,}',
        "sd_mb": f'{rep["export"]["sd"]["mb"]:.1f}',
        "preview_mb": f'{rep["export"]["preview"]["mb"]:.1f}',
        "sd_psnr_drop": f'{rep["export"]["sd"]["psnr_drop_db"]:.2f}',
        "walkable": f'{rep["grid"]["walkable_m2"]:.1f}',
        "extent": " x ".join(f"{x:.1f}" for x in (rep["grid"].get("room_extent_m") or [])) or
                  ("about 3.0 x 2.2 (team's rough size; not measured by the grid stage)" if (scale.get("check") or {}) else "n/a"),
        "walls_n": f'{rep["walls"]["walks"]}', "pushes_n": f'{rep["walls"]["pushes"]}',
        "penetrations": f'{rep["walls"]["penetrations"]}',
        # timings
        "t_total_min": mins(tim["total"]),
        "t_gpu_min": mins(gpu_s),
        "t_ingest": mins(st["ingest"]), "t_sfm": mins(st["sfm"]), "t_align": mins(st["align"]),
        "t_train": mins(st["train"]), "t_geometry": mins(st.get("geometry", 0)) if "geometry" in st else "not run",
        "t_export": mins(st["export"]), "t_grid": mins(st["grid"]),
        # viewer
        "dev_name": walk["name"],
        "dev_first": f'{walk["first_frame_s"]:.2f} s',
        "dev_fps": f'{walk["fps_mean"]:.1f}', "dev_p95": f'{walk["p95_ms"]:.1f} ms',
        "dev_mb": f'{walk["mb"]:.1f} MB', "dev_splats": f'{walk["splats"]:,}',
        "bench_fps": f'{bench["fps_mean"]:.1f}', "bench_p95": f'{bench["p95_ms"]:.1f} ms',
        "bench_secs": f'{bm["secs"]:.0f} s' if bm.get("secs") else "",
        "places_n": f"{len(places)}",
        "places_note": ("Round 2 preview: search, jump, route and follow over %d hand-placed places." % len(places))
        if places else "Round 2 preview: search, jump, route and follow are built; this scene carries no places yet.",
        # evaluation (public rooms)
        "mip_psnr": f'{mip["metrics"]["psnr_db"]:.2f}', "mip_pub": f'{mip["published_psnr_db"]:.2f}',
        "mip_delta": f'{mip["psnr_delta_db"]:+.2f}',
        "mip_ssim": f'{mip["metrics"]["ssim"]:.3f}', "mip_lpips": f'{mip["metrics"]["lpips_alex"]:.3f}',
        "fusion": honka["fusion"],
    }
    for tag, e in (("honka", honka), ("vr", vr)):
        for k, val in e.items():
            if not k.endswith("_raw") and k != "fusion":
                v[f"{tag}_{k}"] = val
    for tag, r in (("honka", "rh-honka"), ("vr", "rh-vrroom")):
        for k, val in public_scores(r).items():
            v[f"{tag}_img_{k}"] = val
    v["honka_vr_f_note"] = "%.2f / %.2f" % (honka["f_ours_raw"], vr["f_ours_raw"])
    v["run_warnings_html"] = "".join(f"<li>{html.escape(w)}</li>" for w in rep.get("warnings", [])) or "<li>none</li>"
    v.update(licence_html())
    v["overrides"] = f'{len(rep.get("manual_overrides", []))}'
    v["git"] = rep.get("git", "")
    v["fig_dir"] = f"file://{ROOT}/docs/build/{run}/figures"
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--team-id", default="")
    ap.add_argument("--viewer-url", default="")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--video-url", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "docs/build" / a.run / "values.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(a.run, a), indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
