#!/usr/bin/env python3
"""
build_pdf.py
------------
Concatenates all monograph Markdown files in canonical pedagogical order
and compiles them into a single publication-ready PDF (manuscript.pdf).

Usage:
    python3 manuscript/scripts/build_pdf.py
    # or specify custom output:
    python3 manuscript/scripts/build_pdf.py [output.pdf]
"""

import os
import sys
import glob
import time
import subprocess
import mistune

def find_manuscript_dir():
    if os.path.isdir("manuscript"):
        return os.path.abspath("manuscript")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, ".."))
    if os.path.isdir(candidate) and any(f.endswith(".md") for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath(".")

def get_ordered_files(manuscript_dir):
    files = []
    # Preface / Title
    readme = os.path.join(manuscript_dir, "README.md")
    if os.path.exists(readme):
        files.append(readme)

    # Chapters and Companion Workbooks in sequence
    for i in range(1, 14):
        ch_str = f"{i:02d}"
        ch_matches = [
            f for f in glob.glob(os.path.join(manuscript_dir, f"ch{ch_str}_*.md"))
            if not f.endswith("_exercises.md")
        ]
        if ch_matches:
            files.append(ch_matches[0])
        wb = os.path.join(manuscript_dir, f"ch{ch_str}_exercises.md")
        if os.path.exists(wb):
            files.append(wb)

    # Mathematical Appendices & Notation Index
    for app in ["app_a.md", "app_b.md", "app_c.md", "app_d.md", "notation_index.md"]:
        p = os.path.join(manuscript_dir, app)
        if os.path.exists(p):
            files.append(p)

    return files

def build_pdf(output_pdf="manuscript.pdf"):
    t0 = time.time()
    manuscript_dir = find_manuscript_dir()
    files = get_ordered_files(manuscript_dir)

    print(f"Concatenating {len(files)} Markdown files from: {manuscript_dir}")
    parts = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            parts.append(fh.read())

    # Join with clean page break
    combined_md = "\n\n<div class=\"page-break\"></div>\n\n".join(parts)
    print(f"Total lines: {len(combined_md.splitlines()):,} | Total characters: {len(combined_md):,}")

    print("Converting Markdown to HTML...")
    body_html = mistune.html(combined_md)

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Theoretical Foundations of Spatial Statistics and Geostatistics</title>
<style>
@page {{
    size: letter;
    margin: 1in;
    @bottom-right {{
        content: counter(page);
        font-size: 9pt;
        color: #555;
    }}
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #111;
}}
h1, h2, h3, h4, h5 {{
    color: #111;
    page-break-after: avoid;
}}
h1 {{ font-size: 20pt; margin-top: 1.5em; border-bottom: 2px solid #222; padding-bottom: 0.3em; }}
h2 {{ font-size: 14pt; margin-top: 1.3em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
h3 {{ font-size: 12pt; margin-top: 1.1em; }}
pre {{
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 8.5pt;
    page-break-inside: avoid;
}}
code {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 9pt;
}}
p code {{
    background: #f3f3f3;
    padding: 2px 4px;
    border-radius: 3px;
}}
blockquote {{
    border-left: 4px solid #0366d6;
    margin: 1em 0;
    padding: 0.5em 1em;
    background: #f1f8ff;
    color: #24292e;
    page-break-inside: avoid;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 9pt;
    page-break-inside: avoid;
}}
th, td {{
    border: 1px solid #dfe2e5;
    padding: 6px 10px;
    text-align: left;
}}
th {{
    background: #f6f8fa;
    font-weight: 600;
}}
.page-break {{
    page-break-before: always;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    temp_html = "manuscript_temp.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"Compiling PDF via WeasyPrint -> {output_pdf}...")
    try:
        subprocess.run(["weasyprint", temp_html, output_pdf], check=True, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(temp_html):
            os.remove(temp_html)

    elapsed = time.time() - t0
    pdf_size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
    print("=" * 65)
    print(f"SUCCESS: Generated {output_pdf}")
    print(f"File Size : {pdf_size_mb:.2f} MB")
    print(f"Build Time: {elapsed:.2f} seconds")
    print("=" * 65)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "manuscript.pdf"
    build_pdf(out)
