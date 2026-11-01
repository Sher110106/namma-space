"""Build report.json and report.html for a run from whatever stage JSON exists (standard library only).

    python -m pipeline.report --run runs/<name> [--strict] [--out-dir DIR]

report.json has the keys of CONTRACTS 2.13. A key whose source file is missing is an explicit
null, and the reason goes in the top-level "missing" map ("sfm.chosen": "20_sfm/sfm.json not found").
A reason that starts with "n/a:" means the key does not apply to this run (not a gap).
"empty_required" lists required keys that are null for a real reason; --strict exits 1 when it is not empty.
Documents bind only to these keys. Nothing here invents a number: every value is copied from a stage JSON.
Stage JSON marked "stub": true (test stubs) is reported in "stub_stages" and flagged in the HTML.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "namma-space/report@1"
_MISSING = object()

# leaf keys that must be non-null in a finished run (test 0.6); "n/a:" reasons do not count
REQUIRED = [
    "name", "git", "images", "input.files", "frames.n_kept", "frames.n_cand", "sfm.chosen", "sfm.variants",
    "align.up_resid_deg", "align.h_cap_m", "align.scale.value", "align.scale.pct_68", "align.scale.cues",
    "train.chosen", "train.variants", "export.sd", "export.removed",
    "grid.walkable_m2", "grid.path_coverage", "grid.room_extent_m", "grid.spawn_ok",
    "viewer.devices", "walls.walks", "walls.pushes", "walls.penetrations", "timings.total",
]

# (section, stage json relative to the run, producing stage)
SOURCES = {
    "frames": "10_frames/frames.json", "sfm": "20_sfm/sfm.json", "align": "40_align/align.json",
    "train": "50_train/train.json", "export": "60_export/export.json", "grid": "70_grid/grid.json",
    "verify": "90_verify/verify.json", "check": "00_preflight/check.json", "geometry": "55_geometry/geometry.json",
}


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return _MISSING
    except (json.JSONDecodeError, OSError) as exc:
        return {"_unreadable": f"{path.name}: {exc}"}


def first(d: Any, *keys, default=None):
    """First present, non-None key of a dict."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def _read_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Builder:
    def __init__(self, run: Path) -> None:
        self.run = run
        self.missing: dict = {}
        self.stub_stages: list = []
        self.src: dict = {}
        for key, rel in SOURCES.items():
            v = load(run / rel)
            self.src[key] = v
            if isinstance(v, dict) and v.get("stub"):
                self.stub_stages.append(key)

    def have(self, key: str) -> bool:
        v = self.src[key]
        return v is not _MISSING and not (isinstance(v, dict) and "_unreadable" in v)

    def get(self, path: str, source: str, *keys, na: str | None = None, default=None) -> Any:
        """Value for report key `path` from stage JSON `source` under any of `keys`; null plus a reason if absent."""
        if not self.have(source):
            self.missing[path] = na or (f"{SOURCES[source]} not found" if self.src[source] is _MISSING
                                        else self.src[source]["_unreadable"])
            return default
        v = first(self.src[source], *keys)
        if v is None:
            present = isinstance(self.src[source], dict) and any(k in self.src[source] for k in keys)
            self.missing[path] = na or (f"{SOURCES[source]} has {keys[0]} but it is null (the stage could not measure it)"
                                        if present else f"{SOURCES[source]} has none of: {', '.join(keys)}")
            return default
        return v

    # ------------------------------------------------------------ sections

    def build(self) -> dict:
        run = self.run
        rj = load(run / "run.json")
        rj = {} if rj is _MISSING else rj
        meta = load(run / "input" / "meta.json")
        meta = {} if meta is _MISSING else meta
        tim = load(run / "timings.json")

        out: dict = {"schema": SCHEMA, "name": run.name}
        out["git"] = rj.get("git") or None
        if not out["git"]:
            self.missing["git"] = "run.json has no git (run.json missing or runner started without --git)"
        out["images"] = rj.get("images") or None
        if not out["images"]:
            self.missing["images"] = "run.json has no image digests (no docker on this run, or run.json missing)"
        out["exit"] = rj.get("exit")

        files = []
        inp = run / "input"
        if inp.is_dir():
            for f in sorted(inp.iterdir()):
                if f.is_dir():
                    n = sum(1 for _ in f.glob("*") if _.is_file())
                    files.append({"name": f.name + "/", "files": n})
                elif f.name != "meta.json":
                    files.append({"name": f.name, "bytes": f.stat().st_size})
        if meta.get("files"):  # written by ns run from the uploaded originals; the Mac copy of input/ is empty
            files = meta["files"]
        takes = meta.get("takes") or []
        frames_mode = bool(takes) and all(t.get("kind") == "frames" for t in takes)
        check = self.src["check"]
        if check is _MISSING:
            check_val = None
            self.missing["input.check"] = "n/a: frames input, no video preflight" if frames_mode else \
                "00_preflight/check.json not found (run ./ns check or ./ns run with a video)"
        else:
            check_val = check
        out["input"] = {"files": files or None, "check": check_val, "height_cm": meta.get("height_cm"),
                        "measures": meta.get("measures", []), "takes": takes}
        if not files:
            self.missing["input.files"] = "no input/ folder in the Mac copy of the run"

        fr = self.src["frames"]
        if self.have("frames"):
            frd = {k: v for k, v in fr.items() if k != "kept"}
            kept = fr.get("kept") or []
            sharp = sorted(k.get("sharp") for k in kept if isinstance(k, dict) and k.get("sharp") is not None)
            frd["kept_summary"] = {"n": len(kept), "sharp_min": sharp[0] if sharp else None,
                                   "sharp_median": sharp[len(sharp) // 2] if sharp else None,
                                   "sharp_max": sharp[-1] if sharp else None}
            out["frames"] = frd
            for k in ("n_kept", "n_cand"):
                if frd.get(k) is None:
                    self.missing[f"frames.{k}"] = f"{SOURCES['frames']} has no {k}"
        else:
            out["frames"] = {"n_cand": None, "n_kept": None, "dropped": None, "hdr": None, "rotation": None, "src": None}
            for k in out["frames"]:
                self.missing[f"frames.{k}"] = f"{SOURCES['frames']} not found"

        out["sfm"] = {"chosen": self.get("sfm.chosen", "sfm", "chosen"),
                      "variants": self.get("sfm.variants", "sfm", "variants"),
                      "why": first(self.src["sfm"], "why"), "missing_ranges": first(self.src["sfm"], "missing_ranges")}

        al = self.src["align"]
        sc = first(al, "scale") or {}
        mea = meta.get("measures") or []
        out["align"] = {
            "up_resid_deg": self.get("align.up_resid_deg", "align", "up_resid_deg"),
            "h_cap_m": self.get("align.h_cap_m", "align", "h_cap_m", "h_cap"),
            "scale": {
                "value": sc.get("value") if isinstance(sc, dict) else None,
                "pct_68": first(sc, "pct_68", "pct68"),
                "cues": first(sc, "cues"),
                "checked_against": sc.get("checked_against") if isinstance(sc, dict) else None,
            },
            "floor": first(al, "floor"),
        }
        for k in ("value", "pct_68", "cues"):
            if out["align"]["scale"][k] is None:
                self.missing[f"align.scale.{k}"] = f"{SOURCES['align']} not found" if not self.have("align") \
                    else f"{SOURCES['align']} has no scale.{k}"
        resc = _read_json_safe(os.path.join(self.run, "40_align", "rescale.json")) if hasattr(self, "run") else None
        if resc and resc.get("checked_against"):
            # a user rescale checked against measured or rough sizes (ns rescale) is recorded in rescale.json
            out["align"]["scale"]["checked_against"] = resc["checked_against"]
            out["align"]["scale"]["user_factor"] = resc.get("factor_total")
            out["align"]["scale"]["check"] = resc.get("validation")
        if out["align"]["scale"]["checked_against"] is None:
            self.missing["align.scale.checked_against"] = "n/a: no tape measure given" if not mea else \
                ("n/a: align stage has not run" if not self.have("align") else "no comparison against the tape measure recorded")

        out["train"] = {"chosen": self.get("train.chosen", "train", "chosen"),
                        "variants": self.get("train.variants", "train", "variants"),
                        "why": first(self.src["train"], "why"), "quick": first(self.src["train"], "quick"),
                        "warnings": first(self.src["train"], "warnings")}

        ex = self.src["export"]
        out["export"] = {"sd": self.get("export.sd", "export", "sd"),
                         "hd": first(ex, "hd"), "preview": first(ex, "preview"),
                         "removed": self.get("export.removed", "export", "removed"),
                         "T_final": first(ex, "T_final")}
        for k in ("hd", "preview"):
            if out["export"][k] is None:
                self.missing[f"export.{k}"] = "n/a: optional (P1) file not produced" if self.have("export") else \
                    f"{SOURCES['export']} not found"

        gr = self.src["grid"]
        spawn_ok = first(gr, "spawn_ok")
        if spawn_ok is None and self.have("grid") and isinstance(gr.get("spawn"), dict):
            spawn_ok = gr["spawn"].get("walkable")
        out["grid"] = {"walkable_m2": self.get("grid.walkable_m2", "grid", "walkable_m2"),
                       "path_coverage": self.get("grid.path_coverage", "grid", "path_coverage"),
                       "room_extent_m": self.get("grid.room_extent_m", "grid", "room_extent_m"),
                       "spawn_ok": spawn_ok}
        if spawn_ok is None:
            self.missing["grid.spawn_ok"] = f"{SOURCES['grid']} not found" if not self.have("grid") else \
                f"{SOURCES['grid']} has no spawn_ok"

        vf = self.src["verify"]
        walls = first(vf, "walls") or {}
        devices = first(first(vf, "viewer") or {}, "devices") or first(vf, "devices")
        out["viewer"] = {"devices": devices}
        if devices is None:
            self.missing["viewer.devices"] = f"{SOURCES['verify']} not found (run ./ns verify)" if not self.have("verify") \
                else f"{SOURCES['verify']} has no viewer.devices (H2 did not run)"
        out["walls"] = {k: (walls.get(k) if isinstance(walls, dict) else None) for k in ("walks", "pushes", "penetrations")}
        for k, v in out["walls"].items():
            if v is None:
                self.missing[f"walls.{k}"] = f"{SOURCES['verify']} not found (run ./ns verify)" if not self.have("verify") \
                    else f"{SOURCES['verify']} has no walls.{k}"

        if tim is _MISSING:
            out["timings"] = {"total": None}
            self.missing["timings.total"] = "timings.json not found"
        else:
            t = dict(tim.get("stages", {}))
            t["total"] = tim.get("total")
            out["timings"] = t
        out["manual_overrides"] = rj.get("manual_overrides", [])
        out["warnings"] = rj.get("warnings", [])
        out["geometry"] = first(self.src["geometry"], "summary") if self.have("geometry") else None

        out["stub_stages"] = self.stub_stages
        out["missing"] = dict(sorted(self.missing.items()))
        out["empty_required"] = [k for k in REQUIRED if lookup(out, k) is None and not
                                 self.missing.get(k, "").startswith("n/a:")]
        return out


def lookup(d: dict, path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- html

CSS = ("body{font:15px/1.45 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#111}"
       "h1{font-size:1.5rem}h2{font-size:1.1rem;margin-top:1.6rem;border-bottom:1px solid #ccc}"
       "table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:.2rem .5rem;text-align:left;font-size:.9rem}"
       "th{background:#f4f4f4}.null{color:#a00}.na{color:#777}.banner{background:#fff3cd;border:1px solid #e0c060;padding:.5rem .8rem}"
       "code{background:#f4f4f4;padding:0 .2rem}img{max-width:100%}.w{overflow-x:auto}")


def esc(v: Any) -> str:
    return html.escape(str(v))


def cell(v: Any) -> str:
    if v is None:
        return '<span class="null">null</span>'
    if isinstance(v, float):
        return esc(f"{v:.4g}")
    if isinstance(v, (dict, list)):
        return f"<code>{esc(json.dumps(v)[:300])}</code>"
    return esc(v)


def kv_table(d: dict, prefix: str, missing: dict) -> str:
    rows = []
    for k, v in d.items():
        reason = missing.get(f"{prefix}.{k}")
        extra = f' <span class="{"na" if reason and reason.startswith("n/a") else "null"}">({esc(reason)})</span>' if v is None and reason else ""
        rows.append(f"<tr><th>{esc(k)}</th><td>{cell(v)}{extra}</td></tr>")
    return "<div class='w'><table>" + "".join(rows) + "</table></div>"


def list_table(rows: list | None) -> str:
    if not rows or not isinstance(rows[0], dict):
        return "<p class='null'>none</p>" if not rows else f"<p>{cell(rows)}</p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{cell(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<div class='w'><table><tr>{head}</tr>{body}</table></div>"


def render_html(rep: dict, run: Path) -> str:
    m = rep["missing"]
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Namma Space report: {esc(rep['name'])}</title>"
             f"<style>{CSS}</style></head><body>",
             f"<h1>Namma Space report: {esc(rep['name'])}</h1>",
             f"<p>git <code>{esc(rep['git'] or 'unknown')}</code>, exit code {cell(rep['exit'])}. "
             "Every number below is copied from <code>report.json</code>.</p>"]
    if rep["stub_stages"]:
        parts.append(f"<p class='banner'>Test stubs wrote data for: {esc(', '.join(rep['stub_stages']))}. "
                     "These numbers are placeholders, not results.</p>")
    if rep["empty_required"]:
        parts.append(f"<p class='banner'>{len(rep['empty_required'])} required keys are empty: "
                     f"{esc(', '.join(rep['empty_required']))}</p>")
    parts.append("<h2>Input</h2>" + kv_table({k: rep['input'][k] for k in ("height_cm", "measures", "check")}, "input", m)
                 + list_table(rep["input"]["files"]))
    parts.append("<h2>Frames</h2>" + kv_table(rep["frames"], "frames", m))
    parts.append("<h2>SfM</h2>" + kv_table({"chosen": rep["sfm"]["chosen"], "why": rep["sfm"]["why"],
                                           "missing_ranges": rep["sfm"]["missing_ranges"]}, "sfm", m)
                 + list_table(rep["sfm"]["variants"]))
    a = rep["align"]
    parts.append("<h2>Align and scale</h2>" + kv_table({"up_resid_deg": a["up_resid_deg"], "h_cap_m": a["h_cap_m"], "floor": a["floor"]}, "align", m)
                 + kv_table(a["scale"], "align.scale", m))
    fig = run / "40_align" / "figures" / "level_before_after.png"
    if fig.exists():
        parts.append("<img src='40_align/figures/level_before_after.png' alt='level before and after'>")
    parts.append("<h2>Training</h2>" + kv_table({"chosen": rep["train"]["chosen"], "why": rep["train"]["why"]}, "train", m)
                 + list_table(rep["train"]["variants"]))
    e = rep["export"]
    parts.append("<h2>Export</h2>" + kv_table(e, "export", m))
    parts.append("<h2>Grid</h2>" + kv_table(rep["grid"], "grid", m))
    parts.append("<h2>Viewer and walls</h2>" + list_table(rep["viewer"]["devices"]) + kv_table(rep["walls"], "walls", m))
    parts.append("<h2>Timings (seconds)</h2>" + kv_table(rep["timings"], "timings", m))
    if rep["manual_overrides"]:
        parts.append("<h2>Manual overrides</h2><ul>" + "".join(f"<li>{esc(o)}</li>" for o in rep["manual_overrides"]) + "</ul>")
    if rep["warnings"]:
        parts.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{esc(w)}</li>" for w in rep["warnings"]) + "</ul>")
    parts.append("<h2>Missing keys</h2>")
    if m:
        parts.append("<table>" + "".join(f"<tr><th>{esc(k)}</th><td class='{'na' if v.startswith('n/a') else 'null'}'>{esc(v)}</td></tr>"
                                         for k, v in m.items()) + "</table>")
    else:
        parts.append("<p>none</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_report(run_dir, out_dir=None) -> dict:
    run = Path(run_dir)
    rep = Builder(run).build()
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "report.json", json.dumps(rep, indent=2, default=str) + "\n")
    write_atomic(out / "report.html", render_html(rep, run))
    return rep


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a required key is empty")
    a = ap.parse_args(argv)
    if not Path(a.run).is_dir():
        print(f"report: {a.run} is not a directory", file=sys.stderr)
        return 2
    rep = build_report(a.run, a.out_dir)
    n_missing = len([k for k, v in rep["missing"].items() if not v.startswith("n/a")])
    print(f"report: wrote {Path(a.out_dir or a.run) / 'report.json'} and report.html; "
          f"{n_missing} keys null, {len(rep['empty_required'])} required empty"
          + (f", stubs in: {', '.join(rep['stub_stages'])}" if rep["stub_stages"] else ""))
    for k in rep["empty_required"]:
        print(f"  empty required: {k}: {rep['missing'].get(k, '?')}")
    return 1 if (a.strict and rep["empty_required"]) else 0
