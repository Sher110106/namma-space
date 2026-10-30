"""Read gsplat's stats JSON, build train.json records, choose the winner."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

PSNR_TIE_DB = 0.10     # held-out PSNR differences below this are a tie (runs vary +-0.2 dB)
LPIPS_TIE = 0.005      # LPIPS differences below this are a tie


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_variant_stats(vdir: str, steps: List[int]) -> dict:
    """steps are the eval steps (7000, 30000); gsplat names files with step-1."""
    out: dict = {}
    for s in steps:
        val = _load(os.path.join(vdir, "stats", f"val_step{s - 1}.json"))
        trn = _load(os.path.join(vdir, "stats", f"train_step{s - 1}_rank0.json"))
        out[s] = {"val": val, "train": trn}
    return out


def variant_record(name: str, vdir: str, steps: List[int], extra: dict) -> dict:
    st = read_variant_stats(vdir, steps)
    last = steps[-1]
    val = st[last]["val"]
    trn = st[last]["train"]
    rec = {
        "name": name,
        "psnr": None, "ssim": None, "lpips": None, "cc_psnr": None,
        "n_gs": None, "time_s": None, "peak_gpu_gb": None,
    }
    if val:
        rec.update(psnr=val.get("psnr"), ssim=val.get("ssim"), lpips=val.get("lpips"),
                   cc_psnr=val.get("cc_psnr"), cc_ssim=val.get("cc_ssim"),
                   cc_lpips=val.get("cc_lpips"), n_gs=val.get("num_GS"),
                   render_s_per_image=val.get("ellipse_time"))
    if trn:
        rec.update(time_s=trn.get("ellipse_time"), peak_gpu_gb=trn.get("mem"))
    first = steps[0]
    if first != last and st[first]["val"]:
        v7 = st[first]["val"]
        rec["at_7k"] = {"psnr": v7.get("psnr"), "ssim": v7.get("ssim"), "lpips": v7.get("lpips"),
                        "cc_psnr": v7.get("cc_psnr"), "n_gs": v7.get("num_GS"),
                        "time_s": (st[first]["train"] or {}).get("ellipse_time")}
    rec.update(extra)
    return rec


def usable(rec: dict) -> bool:
    return (rec.get("status", "ok") == "ok" and rec.get("psnr") is not None
            and rec.get("lpips") is not None and not rec.get("nan_dropped"))


def choose(recs: List[dict]) -> Tuple[Optional[str], str]:
    """Held-out PSNR, then LPIPS; on a tie take R1 (better geometry for the grid)."""
    ok = [r for r in recs if usable(r)]
    if not ok:
        return None, "no variant finished with usable held-out metrics"
    if len(ok) == 1:
        only = ok[0]
        why = f"only {only['name']} finished usable"
        failed = [r["name"] for r in recs if not usable(r)]
        if failed:
            why += f" ({', '.join(failed)} failed or had NaN splats)"
        return only["name"], why
    r0 = next((r for r in ok if r["name"] == "r0"), None)
    r1 = next((r for r in ok if r["name"] == "r1"), None)
    if r0 is None or r1 is None:
        best = max(ok, key=lambda r: r["psnr"])
        return best["name"], f"best held-out PSNR among {[r['name'] for r in ok]}"
    dp = r1["psnr"] - r0["psnr"]
    dl = r1["lpips"] - r0["lpips"]
    if abs(dp) > PSNR_TIE_DB:
        win = r1 if dp > 0 else r0
        return win["name"], (f"held-out PSNR {r0['psnr']:.2f} (r0) vs {r1['psnr']:.2f} (r1): "
                             f"difference {dp:+.2f} dB exceeds the {PSNR_TIE_DB} dB tie band")
    if abs(dl) > LPIPS_TIE:
        win = r1 if dl < 0 else r0
        return win["name"], (f"PSNR tie ({dp:+.2f} dB); LPIPS {r0['lpips']:.3f} (r0) vs "
                             f"{r1['lpips']:.3f} (r1) decides")
    return "r1", (f"tie on PSNR ({dp:+.2f} dB) and LPIPS ({dl:+.3f}); r1 taken for its "
                  f"depth-supervised geometry")


_STEP_RE = re.compile(r"(\d+)/(\d+) \[")


def progress_from_log(path: str) -> Optional[Tuple[int, int]]:
    """Last 'k/N [' tqdm counter in the log tail, or None."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 6000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    m = None
    for m in _STEP_RE.finditer(tail):
        pass
    return (int(m.group(1)), int(m.group(2))) if m else None
