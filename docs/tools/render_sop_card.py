#!/usr/bin/env python3
"""Render docs/SOP_card.html to PDF with headless Chrome.

  SOP_card.pdf      one page, 390 CSS px wide and as tall as the content: reads on a phone without zooming
  SOP_card_A4.pdf   one A4 page in two columns, for printing

    python3 docs/tools/render_sop_card.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chrome  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent
SRC = DOCS / "SOP_card.html"
PHONE_W = 390
A4_COLUMNS = "main{max-width:none;column-count:2;column-gap:18px;padding:0} body{font-size:13.5px} h1{font-size:20px}"


def main() -> int:
    h = chrome.content_height(SRC, PHONE_W)
    chrome.print_pdf(SRC, DOCS / "SOP_card.pdf", page_css=f"@page{{size:{PHONE_W}px {h + 2}px;margin:0}}")
    chrome.print_pdf(SRC, DOCS / "SOP_card_A4.pdf",
                     page_css="@page{size:A4;margin:10mm}", extra_head=f"<style>{A4_COLUMNS}</style>")
    for name in ("SOP_card.pdf", "SOP_card_A4.pdf"):
        n = chrome.pdf_pages(DOCS / name)
        print(f"{name}: {n} page(s)")
        if n != 1:
            print(f"error: {name} should be one page", file=sys.stderr)
            return 1
    print(f"phone page: {PHONE_W} x {h} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
