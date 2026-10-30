"""Headless Chrome helpers for the docs and video tooling (standard library only).

Uses Playwright's cached chrome-headless-shell, or the binary named in NS_CHROME.
Every function takes plain file paths and runs one short Chrome process.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

CACHE = Path.home() / "Library" / "Caches" / "ms-playwright"


def chrome_path() -> str:
    env = os.environ.get("NS_CHROME")
    if env:
        return env
    found: List[Path] = []
    if CACHE.exists():
        for d in sorted(CACHE.glob("chromium_headless_shell-*"), reverse=True):
            found += list(d.glob("chrome-headless-shell-*/chrome-headless-shell"))
    if not found:
        raise RuntimeError("chrome-headless-shell not found; install Playwright's Chromium or set NS_CHROME")
    return str(found[0])


def _run(args: List[str], timeout: int = 90) -> subprocess.CompletedProcess:
    base = [chrome_path(), "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
            "--no-first-run", "--allow-file-access-from-files"]
    return subprocess.run(base + args, capture_output=True, text=True, timeout=timeout)


def _inject(html: str, head: str = "", body_end: str = "") -> str:
    if head:
        html = html.replace("</head>", head + "</head>", 1) if "</head>" in html else head + html
    if body_end:
        html = html.replace("</body>", body_end + "</body>", 1) if "</body>" in html else html + body_end
    return html


def _temp_copy(src: Path, text: str) -> Path:
    """Write the modified HTML next to the source so relative links still work."""
    fd, name = tempfile.mkstemp(prefix=".tmp_render_", suffix=".html", dir=str(src.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return Path(name)


def content_height(html_path: Path, width: int, extra_head: str = "") -> int:
    """Height in CSS px of the page laid out at the given width."""
    probe = "<script>document.documentElement.setAttribute('data-h',String(Math.ceil(document.documentElement.scrollHeight)))</script>"
    tmp = _temp_copy(html_path, _inject(html_path.read_text(encoding="utf-8"), extra_head, probe))
    try:
        r = _run([f"--window-size={width},900", "--virtual-time-budget=4000", "--dump-dom", tmp.resolve().as_uri()])
    finally:
        tmp.unlink(missing_ok=True)
    m = re.search(r'data-h="(\d+)"', r.stdout)
    if not m:
        raise RuntimeError("could not measure page height: " + r.stderr[-400:])
    return int(m.group(1))


def print_pdf(html_path: Path, pdf_path: Path, page_css: str = "", extra_head: str = "") -> None:
    """Print to PDF. page_css is an @page rule (size and margin)."""
    head = f"<style>{page_css}</style>{extra_head}" if (page_css or extra_head) else ""
    tmp = _temp_copy(html_path, _inject(html_path.read_text(encoding="utf-8"), head))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = _run(["--virtual-time-budget=6000", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path.resolve()}",
                  tmp.resolve().as_uri()], timeout=180)
    finally:
        tmp.unlink(missing_ok=True)
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise RuntimeError("PDF not written: " + r.stderr[-500:])


def screenshot(html_path: Path, png_path: Path, width: int, height: int, scale: float = 1.0,
               extra_head: str = "", transparent: bool = False) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_copy(html_path, _inject(html_path.read_text(encoding="utf-8"), extra_head))
    try:
        args = [f"--window-size={width},{height}", f"--force-device-scale-factor={scale}",
                "--virtual-time-budget=4000", f"--screenshot={png_path.resolve()}"]
        if transparent:
            args.append("--default-background-color=00000000")
        r = _run(args + [tmp.resolve().as_uri()])
    finally:
        tmp.unlink(missing_ok=True)
    if not png_path.exists():
        raise RuntimeError("screenshot not written: " + r.stderr[-500:])


def pdf_pages(pdf: Path) -> Optional[int]:
    """Page count through pdfinfo; None if pdfinfo is missing."""
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else None
