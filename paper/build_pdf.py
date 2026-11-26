#!/usr/bin/env python3
"""Render nullyn-algorithm.md to HTML and a print PDF (11.5 pt, dark on light)."""
from pathlib import Path
import subprocess
import sys

import markdown

HERE = Path(__file__).resolve().parent
MD = HERE / "nullyn-algorithm.md"
HTML = HERE / "nullyn-algorithm.html"
PDF = HERE / "nullyn-algorithm.pdf"

CSS = """
@page { size: letter; margin: 0.85in 0.9in 0.9in 0.9in; }
html, body { background: #ffffff; color: #161616; }
body {
  font-family: "Liberation Serif", "Times New Roman", Times, serif;
  font-size: 11.5pt;
  line-height: 1.45;
  max-width: 7.2in;
  margin: 0 auto;
}
h1 { font-size: 20pt; font-weight: 700; line-height: 1.2; margin: 0 0 0.4em; letter-spacing: -0.01em; }
h2 { font-size: 13.5pt; font-weight: 700; margin: 1.35em 0 0.45em; page-break-after: avoid; }
h3 { font-size: 12pt; font-weight: 700; margin: 1.1em 0 0.35em; page-break-after: avoid; }
p, li { margin: 0 0 0.55em; }
em { font-style: italic; }
strong { font-weight: 700; }
a { color: #161616; text-decoration: none; }
hr { display: none; }
/* byline block: first paragraphs after h1 */
h1 + p { margin-bottom: 0.15em; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.6em 0 1em;
  font-size: 11pt;
  line-height: 1.3;
  page-break-inside: avoid;
}
th, td {
  border: 0.4pt solid #444;
  padding: 3.5pt 5pt;
  text-align: left;
  vertical-align: top;
}
th { background: #f0f0f0; font-weight: 700; }
code, pre {
  font-family: "Liberation Mono", "Courier New", monospace;
  font-size: 10pt;
}
code { background: #f4f4f4; padding: 0 2pt; }
"""

def main():
    md = MD.read_text()
    body = markdown.markdown(md, extensions=["tables", "sane_lists", "smarty"])
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nullyn AI: The Feedback-Suppression Algorithm</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML.write_text(html)
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        HTML.resolve().as_uri(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not PDF.exists():
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"chrome failed: {r.returncode}")
    print(f"wrote {HTML} and {PDF} ({PDF.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
