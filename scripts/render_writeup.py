"""Check the HTML write-up's print layout with headless Edge and produce a PDF.

Usage: python scripts/render_writeup.py [out_dir]

For every `.page` section it reports whether the content fits an 11-inch page (the print CSS clips
overflow, so this is the only reliable check), then prints the document to PDF and, if PyMuPDF is
installed, rasterises each page to PNG for a visual pass.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "TLH_Engine_with_YANG___Technical___Positioning_Write-Up" / "index.html"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "var" / "writeup_render"
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]
PAGE_PX = 11 * 96


def browser() -> Path:
    for c in EDGE_CANDIDATES:
        if c.exists():
            return c
    sys.exit("no headless-capable browser found (Edge or Chrome)")


def run(exe: Path, *args: str) -> str:
    base = [str(exe), "--headless=new", "--disable-gpu", "--no-first-run",
            f"--user-data-dir={OUT / 'profile'}", "--virtual-time-budget=10000"]
    res = subprocess.run(base + list(args), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.stdout


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    exe = browser()
    url = HTML.resolve().as_uri()

    dom = run(exe, "--window-size=1000,1200", "--dump-dom", url + "?measure=1")
    m = re.search(r'<pre id="measure">(.*?)</pre>', dom, re.S)
    if not m:
        print("measurement block not found; is the self-check script present in index.html?")
        return 1
    bad = 0
    for p in json.loads(m.group(1)):
        over = p["height"] - PAGE_PX
        flag = "  <-- OVERFLOWS 11in" if over > 0 else ("  (tight)" if p["slack"] < 12 else "")
        if over > 0:
            bad += 1
        print(f"page {p['page']:02d}: height {p['height']:4d}px  slack {p['slack']:4d}px  last={p['last']}{flag}")
    print(f"{bad} page(s) overflow")

    pdf = OUT / "writeup.pdf"
    run(exe, "--no-pdf-header-footer", f"--print-to-pdf={pdf}", url)
    print("pdf:", pdf, pdf.stat().st_size if pdf.exists() else "missing")
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return bad
    doc = fitz.open(str(pdf))
    for i, page in enumerate(doc):
        page.get_pixmap(dpi=120).save(str(OUT / f"pdf_{i + 1:02d}.png"))
    print(f"{len(doc)} pages rasterised to {OUT}")
    return bad


if __name__ == "__main__":
    sys.exit(main())
