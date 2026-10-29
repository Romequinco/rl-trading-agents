"""Genera results/informe.pdf a partir de results/informe.md (D-3.1).

Markdown -> HTML con estilos de impresión A4 -> PDF con Chrome sin interfaz. Falla si el PDF
supera 2 páginas. Con --png guarda además cada página como imagen para revisarla.

Uso: python build_informe.py [--png]
"""
from pathlib import Path
import shutil
import subprocess
import sys

import fitz  # PyMuPDF
import markdown

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SRC, HTML, PDF = RESULTS / "informe.md", RESULTS / "informe.html", RESULTS / "informe.pdf"
MAX_PAGES = 2
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chrome",
]

CSS = """
@page { size: A4; margin: 10mm 11mm 9mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 8.6pt; line-height: 1.32; color: #1f2328; margin: 0; }
h1 { font-size: 15pt; margin: 0 0 1mm; color: #102a43; }
h2 { break-after: avoid; font-size: 10.5pt; margin: 2.6mm 0 1.2mm; padding: 0.8mm 2mm; background: #e8eef5; color: #102a43; border-left: 2.5pt solid #2B6CB0; }
h3 { font-size: 9pt; margin: 1.8mm 0 0.6mm; color: #102a43; }
p { margin: 0 0 1.2mm; }
ul { margin: 0 0 1.2mm; padding-left: 4.2mm; }
li { margin: 0 0 0.4mm; }
img { width: 100%; display: block; break-inside: avoid; margin: 0.6mm 0 0.4mm; }
table { border-collapse: collapse; width: 100%; font-size: 7.4pt; margin: 0.8mm 0 0.6mm; }
th, td { border-bottom: 0.4pt solid #c9d1d9; padding: 0.5mm 1mm; text-align: right; }
th:first-child, td:first-child, th:last-child, td:last-child { text-align: left; }
th { background: #f3f5f8; }
.sub { color: #57606a; font-size: 8pt; margin-bottom: 1.5mm; }
.cap { font-size: 7.4pt; color: #57606a; margin: 0 0 1.2mm; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.6mm; margin: 1mm 0 1.4mm; }
.card { border: 0.5pt solid #c9d1d9; border-radius: 1.5mm; padding: 1.2mm 1.6mm; font-size: 7.6pt; }
.card b { display: block; color: #102a43; }
.verdict { background: #e6f4ea; color: #14532d; border-radius: 1.5mm; padding: 1.2mm 2mm; margin: 1mm 0 1.4mm; font-weight: 600; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 3.5mm; }
.note { background: #fff8e1; border-radius: 1.5mm; padding: 1.2mm 2mm; }
.pagebreak { break-after: page; }
"""


def find_chrome():
    for c in CHROME_CANDIDATES:
        path = c if Path(c).exists() else shutil.which(c)
        if path:
            return path
    raise FileNotFoundError("No se encuentra Chrome/Edge para imprimir el PDF")


def main():
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "md_in_html", "attr_list"])
    HTML.write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Agentes de RL tabular para trading de un activo</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    doc = fitz.open(PDF)
    pages = doc.page_count
    if "--png" in sys.argv:
        for i, page in enumerate(doc):
            page.get_pixmap(dpi=110).save(RESULTS / f"informe_p{i + 1}.png")
    doc.close()
    print(f"{PDF.name}: {pages} páginas")
    if pages > MAX_PAGES:
        raise SystemExit(f"El informe ocupa {pages} páginas; el máximo es {MAX_PAGES}")


if __name__ == "__main__":
    main()
