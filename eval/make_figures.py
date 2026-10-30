"""Redraw every figure in eval/results from the JSON files in the same folder.

    python eval/make_figures.py            # write all PNGs next to the JSON files
    python eval/make_figures.py --out DIR  # write them to DIR instead
    python eval/make_figures.py --table    # print the Markdown tables used in eval/README.md

No number is typed here. Each value is read from a JSON file in eval/results.
Needs matplotlib only (Agg backend).
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
ROOMS = ["honka", "vr_room"]
OURS, LIDAR, GREY, INK = "#0072B2", "#E69F00", "#8A8A8A", "#222222"
OURS_LIGHT = "#8FC1E3"
DPI = 200


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def data() -> dict:
    d = {"mip": load("mip360_room"), "syn": load("synthetic_room")}
    for r in ROOMS:
        d[r] = {"ours": load(f"mushroom_{r}_ours"), "lidar": load(f"mushroom_{r}_lidar")}
    return d


def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": "#555555", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 9.5, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })


def geo(d: dict, room: str) -> tuple[dict, dict]:
    return d[room]["ours"]["geometry"]["cropped"], d[room]["lidar"]["score"]["cropped"]


def ratio(d: dict, room: str) -> float:
    o, l = geo(d, room)
    return o["f_score_5cm"] / l["f_score_5cm"]


def label_bars(ax, bars, fmt="{:.3f}", pad=0.012):
    top = ax.get_ylim()[1]
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * pad, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8.5)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- a. geometry
def fig_geometry(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(8, 4.1))
    specs = [("f_score_5cm", "F-score at 5 cm\n(higher is better)", 1.0),
             ("accuracy_m", "Accuracy, mean (m)\n(lower is better)", None),
             ("completeness_m", "Completeness, mean (m)\n(lower is better)", None)]
    w = 0.38
    for ax, (key, title, top) in zip(axes, specs):
        vals_o = [geo(d, r)[0][key] for r in ROOMS]
        vals_l = [geo(d, r)[1][key] for r in ROOMS]
        ax.set_ylim(0, top or max(vals_o + vals_l) * 1.22)
        b1 = ax.bar([i - w / 2 for i in range(2)], vals_o, w, color=OURS, label="Phone-only (ours)")
        b2 = ax.bar([i + w / 2 for i in range(2)], vals_l, w, color=LIDAR, label="iPhone LiDAR fusion")
        label_bars(ax, b1)
        label_bars(ax, b2)
        ax.set_xticks(range(2), ROOMS)
        ax.set_title(title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.905))
    fig.suptitle("MuSHRoom rooms: geometry against the Faro scan", fontsize=11.5, y=0.985)
    fig.text(0.5, 0.925,
             "Same fusion (UniformTSDFVolume), same crop. Ours: cameras fitted to the dataset camera path (7-DoF) first.",
             ha="center", va="top", fontsize=8, color="#444444")
    fig.text(0.5, 0.012,
             "F@5cm ours / LiDAR: " + ", ".join(f"{r} {ratio(d, r) * 100:.1f}%" for r in ROOMS) +
             ".  Cropped scores; accuracy and completeness are means.",
             ha="center", fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.70, bottom=0.14, wspace=0.38)
    save(fig, out, "mushroom_geometry.png")


# ---------------------------------------------------------------- b. scale and depth
def fig_scale(d, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    sc = [d[r]["ours"]["scale"] for r in ROOMS]
    dp = [d[r]["ours"]["depth_vs_lidar"] for r in ROOMS]
    err = [s["automatic_error_pct"] for s in sc]
    a1.set_ylim(0, max(err) * 1.3)
    b = a1.bar(range(2), err, 0.5, color=OURS)
    label_bars(a1, b, "{:.2f}%")
    a1.set_xticks(range(2), [f"{r}\ntrue scale inside\n68% interval: "
                             f"{'yes' if s['true_inside_interval_68'] else 'no'}\n"
                             f"fit scale {s['fit_scale_aligned_to_dataset']:.3f}"
                             for r, s in zip(ROOMS, sc)], fontsize=7.5)
    a1.set_ylabel("Automatic scale error (%)")
    a1.set_title("Scale from camera height,\nbefore any fit")
    w = 0.38
    f5 = [x["within_5cm_frac"] * 100 for x in dp]
    f10 = [x["within_10cm_frac"] * 100 for x in dp]
    a2.set_ylim(0, 100)
    b5 = a2.bar([i - w / 2 for i in range(2)], f5, w, color=OURS_LIGHT, label="within 5 cm")
    b10 = a2.bar([i + w / 2 for i in range(2)], f10, w, color=OURS, label="within 10 cm")
    label_bars(a2, b5, "{:.1f}%")
    label_bars(a2, b10, "{:.1f}%")
    a2.set_xticks(range(2), [f"{r}\nmedian error {x['median_abs_error_cm']:.2f} cm\n{x['heldout_views']} held-out views"
                             for r, x in zip(ROOMS, dp)], fontsize=7.5)
    a2.set_ylabel("Pixels (%)")
    a2.set_title("Depth of our model against iPhone LiDAR depth\n(after the fit scale)")
    a2.legend(frameon=False, loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("MuSHRoom rooms: scale and depth (phone-only)", fontsize=11.5, y=0.985)
    fig.text(0.5, 0.925, "Scale error is automatic. Depth uses cameras fitted to the dataset camera path (7-DoF) first.",
             ha="center", va="top", fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.22, wspace=0.28)
    save(fig, out, "mushroom_scale.png")


# ---------------------------------------------------------------- c. Mip-NeRF 360
def fig_mip(d, out):
    m = d["mip"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    vals = [m["metrics"]["psnr_db"], m["published_psnr_db"]]
    ax.set_ylim(0, max(vals) * 1.18)
    b = ax.bar([0, 1], vals, 0.5, color=[OURS, GREY])
    label_bars(ax, b, "{:.2f} dB")
    ax.set_xticks([0, 1], [f"Ours (gsplat, {m['steps']:,} steps)", "Published figure"])
    ax.set_ylabel("Held-out PSNR (dB)")
    ax.set_title(f"Mip-NeRF 360 'room': ours minus published {m['psnr_delta_db']:+.2f} dB", fontsize=11, loc="left")
    n = len(m["image_ids"])
    ax.text(0.5, -0.2,
            f"SSIM {m['metrics']['ssim']:.3f}   LPIPS (AlexNet) {m['metrics']['lpips_alex']:.3f}   "
            f"{m['metrics']['num_gs']:,} Gaussians\n{n} held-out images (every {m['test_every']}th of {m['dataset_images']}), "
            f"images scaled by 1/{m['data_factor']}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.25)
    save(fig, out, "mip360_room.png")


# ---------------------------------------------------------------- d. cards
def card(out, name, title, sub, status, color, sections, footer):
    foot = textwrap.wrap(footer, 98)
    H = 1.0 + sum(0.34 + 0.25 * len(rows) + 0.12 for _, rows in sections) + 0.14 * len(foot) + 0.3
    fig = plt.figure(figsize=(6.5, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, H)
    ax.add_patch(plt.Rectangle((0, 0), 0.018, H, color=color))
    ink = {OURS: OURS, LIDAR: "#8A5A00", GREY: "#555555"}[color]
    ax.text(0.05, H - 0.3, title, fontsize=12.5, fontweight="bold", va="center")
    ax.text(0.05, H - 0.58, sub, fontsize=8, color="#444444", va="center")
    ax.text(0.95, H - 0.3, f"status: {status}", fontsize=9, ha="right", va="center", color=ink, fontweight="bold")
    y = H - 0.95
    for head, rows in sections:
        ax.text(0.05, y, head, fontsize=8.5, fontweight="bold", va="center")
        ax.plot([0.05, 0.95], [y - 0.1, y - 0.1], color="#CCCCCC", lw=0.8)
        y -= 0.3
        for lab, val in rows:
            ax.text(0.07, y, lab, fontsize=8.5, va="center")
            ax.text(0.95, y, val, fontsize=8.5, va="center", ha="right", fontweight="bold")
            y -= 0.25
        y -= 0.17
    ax.text(0.05, 0.14, "\n".join(foot), fontsize=7.3, color="#444444", va="bottom")
    save(fig, out, name)


def geo_rows(g):
    return [("F-score at 5 cm", f"{g['f_score_5cm']:.3f}"),
            ("Precision / recall at 5 cm", f"{g['precision_5cm']:.3f} / {g['recall_5cm']:.3f}"),
            ("Accuracy, mean / median (m)", f"{g['accuracy_m']:.3f} / {g['accuracy_median_m']:.3f}"),
            ("Completeness, mean / median (m)", f"{g['completeness_m']:.3f} / {g['completeness_median_m']:.3f}")]


def fusion_line(j):
    f = j["fusion"]
    return (f"{f['method']}, voxel {f['voxel_size_m'] * 100:g} cm, trunc {f['sdf_trunc_m'] * 100:g} cm, "
            f"depth cap {f['depth_trunc_m']:g} m")


def cards(d, out):
    for r in ROOMS:
        o, l = d[r]["ours"], d[r]["lidar"]
        go, gl = geo(d, r)
        sc, dp = o["scale"], o["depth_vs_lidar"]
        card(out, f"mushroom_{r}_ours.png", f"Phone-only (ours), {r}",
             f"{o['date'][:10]}   {o['matched_views']} of {o['views']} views matched to dataset poses   run {Path(o['run']).name}",
             o["status"], OURS,
             [("Geometry against the Faro scan (cropped)",
               [("Fusion", fusion_line(o))] + geo_rows(go) + [("F@5cm ratio to iPhone LiDAR fusion", f"{ratio(d, r) * 100:.1f}%")]),
              ("Scale (automatic, before the fit)",
               [("Error", f"{sc['automatic_error_pct']:.2f}%"),
                ("True scale inside 68% interval", "yes" if sc["true_inside_interval_68"] else "no")]),
              (f"Depth against iPhone LiDAR depth ({dp['heldout_views']} held-out views)",
               [("Median absolute error", f"{dp['median_abs_error_cm']:.2f} cm"),
                ("Pixels within 5 cm / 10 cm",
                 f"{dp['within_5cm_frac'] * 100:.1f}% / {dp['within_10cm_frac'] * 100:.1f}%")])],
             f"Before scoring, our camera centres were fitted to the dataset camera path with a 7-DoF similarity "
             f"(fit scale {sc['fit_scale_aligned_to_dataset']:.3f}, applied to geometry and depth). This removes our own "
             f"scale error, so the geometry and depth scores are a best case for shape, not the end-to-end result.")
        card(out, f"mushroom_{r}_lidar.png", f"iPhone LiDAR fusion (baseline), {l['dataset'].rsplit('/', 1)[-1]}",
             f"{l['date'][:10]}   {l['views']} views, stride {l['stride']}   dataset's own poses and depth",
             l["status"], LIDAR,
             [("Geometry against the Faro scan (cropped)", [("Fusion", fusion_line(l))] + geo_rows(gl))],
             "The dataset's own poses are used, with no scale fit. Same fusion and same crop as the phone-only "
             "run (predicted points inside the Faro mesh bounding box grown by "
             f"{l['score']['crop']['grow_m'] * 100:g} cm). The dataset's ICP transform places the mesh in the Faro frame.")
    s = d["syn"]
    card(out, "synthetic_room.png", "Synthetic room, fusion code check",
         "4 synthetic depth views of a box room (eval/synthetic_room.py)", s["status"], GREY,
         [("TSDF fusion of synthetic depth",
           [("Fusion", fusion_line(s)), ("Views / mesh vertices", f"{s['views']} / {s['vertices']:,}"),
            ("Median distance to the true walls", f"{s['median_wall_error_m']:.4f} m"),
            ("Pass limit", f"{s['gate_m']:.2f} m")])],
         "This tests the fusion code on exact depth. It says nothing about phone video.")


# ---------------------------------------------------------------- table
def table(d) -> str:
    L = ["Geometry against the Faro scan (cropped). Ours is scored after the 7-DoF fit.", "",
         "| Room | Method | F@5cm | Precision | Recall | Accuracy (m) | Completeness (m) | F@5cm ours / LiDAR |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in ROOMS:
        go, gl = geo(d, r)
        for who, g, extra in (("Phone-only (ours)", go, f"{ratio(d, r) * 100:.1f}%"), ("iPhone LiDAR fusion", gl, "")):
            L.append(f"| {r} | {who} | {g['f_score_5cm']:.3f} | {g['precision_5cm']:.3f} | {g['recall_5cm']:.3f} | "
                     f"{g['accuracy_m']:.3f} | {g['completeness_m']:.3f} | {extra} |")
    L += ["", "Scale (automatic, before the fit) and depth against iPhone LiDAR depth (phone-only).", "",
          "| Room | Scale error | True scale in 68% interval | Fit scale | Depth median error | Within 5 cm | Within 10 cm | Held-out views |",
          "|---|---:|:---:|---:|---:|---:|---:|---:|"]
    for r in ROOMS:
        o = d[r]["ours"]
        s, p = o["scale"], o["depth_vs_lidar"]
        L.append(f"| {r} | {s['automatic_error_pct']:.2f}% | {'yes' if s['true_inside_interval_68'] else 'no'} | "
                 f"{s['fit_scale_aligned_to_dataset']:.3f} | {p['median_abs_error_cm']:.2f} cm | "
                 f"{p['within_5cm_frac'] * 100:.1f}% | {p['within_10cm_frac'] * 100:.1f}% | {p['heldout_views']} |")
    m, s = d["mip"], d["syn"]
    L += ["", "Other checks.", "", "| Check | Result |", "|---|---|",
          f"| Mip-NeRF 360 room, held-out PSNR | ours {m['metrics']['psnr_db']:.2f} dB, published {m['published_psnr_db']:.2f} dB, "
          f"difference {m['psnr_delta_db']:+.2f} dB |",
          f"| Mip-NeRF 360 room, SSIM / LPIPS (AlexNet) | {m['metrics']['ssim']:.3f} / {m['metrics']['lpips_alex']:.3f} |",
          f"| Synthetic room, median wall error | {s['median_wall_error_m']:.4f} m (pass limit {s['gate_m']:.2f} m, status {s['status']}) |"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--table", action="store_true", help="print the Markdown tables and exit")
    a = ap.parse_args(argv)
    d = data()
    if a.table:
        print(table(d))
        return
    style()
    fig_geometry(d, a.out)
    fig_scale(d, a.out)
    fig_mip(d, a.out)
    cards(d, a.out)


if __name__ == "__main__":
    main()
