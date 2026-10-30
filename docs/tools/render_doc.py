#!/usr/bin/env python3
"""Fill an HTML template from JSON files and print it to PDF.

usage: render_doc.py template.html output.pdf --data values.json [--data more.json] [--max-pages 5]

A placeholder is ${key} or ${a.b.c} (nested keys). Files are merged key by key, so a later file
adds to an earlier one and never wipes a nested record. A missing key stops the build: it is safer
than printing a blank where a link or number belongs. Every value is printed as given; format
numbers before they reach this script (see doc_values.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chrome import pdf_pages, print_pdf  # noqa: E402

TOKEN = re.compile(r"\$\{([A-Za-z_][\w.-]*)\}")


def merge(into: dict, new: dict) -> None:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            merge(into[k], v)
        else:
            into[k] = v


def lookup(data: dict, key: str):
    if key in data:
        return data[key]
    cur = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def fill(text: str, data: dict) -> str:
    missing = []

    def sub(m):
        try:
            val = lookup(data, m.group(1))
        except KeyError:
            missing.append(m.group(1))
            return m.group(0)
        if val is None:
            missing.append(m.group(1) + " (null)")
            return m.group(0)
        if isinstance(val, (dict, list)):
            raise SystemExit(f"placeholder {m.group(1)} is a record, not a value; format it first")
        return str(val)

    out = TOKEN.sub(sub, text)
    if missing:
        raise SystemExit("unresolved placeholders: " + ", ".join(sorted(set(missing))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("output")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--max-pages", type=int, default=5)
    a = ap.parse_args()
    data: dict = {}
    for p in a.data:
        obj = json.loads(Path(p).read_text())
        if isinstance(obj, dict):
            merge(data, obj)
    html = fill(Path(a.template).read_text(), data)
    out = Path(a.output)
    tmp = out.with_suffix(".filled.html")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(html)
    try:
        print_pdf(tmp, out, page_css="@page { size: A4; margin: 12mm; }")
    finally:
        tmp.unlink(missing_ok=True)
    n = pdf_pages(out)
    if n and n > a.max_pages:
        raise SystemExit(f"page count {n} exceeds {a.max_pages}")
    print(f"wrote {out} ({n} pages)")


if __name__ == "__main__":
    main()
