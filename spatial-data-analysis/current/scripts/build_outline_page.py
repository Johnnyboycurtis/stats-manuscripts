#!/usr/bin/env python3
"""Assemble a single HTML page that renders OUTLINE.md with math.

The page embeds the Markdown source between scripts/outline_page/head.html
and tail.html. In the browser, marked turns it into HTML and MathJax
typesets the equations, with a jump bar for the parts and a button that
copies the Markdown. It was used to publish the outline as a shareable
Claude artifact.

Usage:
    python3 scripts/build_outline_page.py            # writes build/outline.html
    python3 scripts/build_outline_page.py OUT.html

Publishing the page as an artifact is done with the Artifact tool in a
Claude session; this script only assembles the file. Re-run it after every
change to OUTLINE.md. The page loads marked and MathJax from cdnjs, so it
needs an internet connection to render.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "scripts" / "outline_page"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "build" / "outline.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    markdown = (REPO / "OUTLINE.md").read_text()
    if "</script" in markdown:
        sys.exit("OUTLINE.md contains '</script', which would end the page early")
    html = ((PAGE / "head.html").read_text() + markdown
            + (PAGE / "tail.html").read_text())
    out.write_text(html)
    print(f"wrote {out} ({len(html):,} characters)")


if __name__ == "__main__":
    main()
