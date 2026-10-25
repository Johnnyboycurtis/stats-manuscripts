#!/usr/bin/env python3
"""Check chapters against the mechanical rules in MARKDOWNSTYLE.md.

This catches the things that break the pandoc build or drift from the house
style. It does not judge voice, correctness, or whether a symbol appears
before its first introduction in NOTATION.md; those need a reader.

Usage:
    python3 scripts/lint_chapter.py chapters/ch03_empirical_variography.md
    python3 scripts/lint_chapter.py --all

What it checks (outside code blocks unless noted):
    - heading levels: one "#" title, no skipped levels
    - banned phrases and em dashes
    - GitHub alert syntax ("> [!NOTE]"), which pandoc prints literally
    - non-ASCII characters other than "§", en dashes, and Latin letters
      (Unicode math symbols and Greek letters cannot be typeset by pdflatex)
    - stray "$ ... $" with spaces just inside the dollars
    - blank lines inside "$$" display blocks
    - a bare "|" inside math in a table row
    - $\\epsilon$, X^T, \\bm{...}, \\text{Var} and similar style slips
    - a "$$" block that is not on its own line
    - TAB characters, and TeX commands not in the KNOWN_TEX list (catches
      typos such as \\logb)
    - code lines over 79 characters (python blocks)
    - counts of **[STORY NEEDED]** placeholders (reported, not an error)

Exit status is 1 if any error is found.
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_CODE_LINE = 79

BANNED = [r"\btrivial(ly)?\b", r"\bit is easy to see\b", r"\beasy to show\b",
          r"\bclearly\b", r"\bobviously\b", r"\bsimply\b", r"\bnote that\b",
          r"\bit is worth noting\b", r"\bworth noting\b", r"\bcrucial(ly)?\b",
          r"\bpowerful\b", r"\belegant\b"]
STYLE_SLIPS = [(r"\\epsilon\b", r"use \varepsilon"),
               (r"\^T\b|\^\\top", "transpose is a prime: X'"),
               (r"\\bm\{", r"use \boldsymbol or \mathbf"),
               (r"\\text\{(Var|Cov|E|tr)\}", r"use \operatorname"),
               (r"\\begin\{align\}", r"use aligned inside $$"),
               (r"\\mathbbm\{|\\mathds\{", r"use \mathbb{1}")]
ALLOWED_NON_ASCII = set("§–")

# TeX commands that appear in the chapters and are known to work in both
# pdflatex (with amsmath, amssymb, dsfont) and MathJax. A command that is
# not listed is reported once, so typos such as \logb are caught. If a new
# command is valid in both, add it here.
KNOWN_TEX = frozenset("""
    Big Delta Gamma Lambda Longleftrightarrow Longrightarrow Omega Phi Pi Pr
    Psi Sigma Xi alpha approx arccos arcsin bar begin beta big bigl bigr
    binom boldsymbol cap cdot cdots chi circ cos cup ddot delta dfrac dot
    dots ell end equiv eta exp frac gamma ge gg hat iff in infty int kappa
    lVert lambda langle ldots le left ll ln log longrightarrow lvert mathbb
    mathbf mathcal mathrm max mid middle min mu nabla ne notin nu omega
    ominus operatorname otimes overline partial perp phi pi pm prime prod
    propto psi qquad quad rVert rangle rho right rvert setminus sigma sim
    sin sqrt square star subseteq sum tau text tfrac theta tilde times to
    underbrace varepsilon varsigma vartheta vdots widehat widetilde xi zeta
""".split())


def split_regions(lines):
    """Yield (line_number, text, in_code, in_display) for each line."""
    in_code = False
    in_display = False
    for number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            yield number, raw, True, False
            continue
        if in_code:
            yield number, raw, True, False
            continue
        if stripped == "$$":
            yield number, raw, False, True
            in_display = not in_display
            continue
        yield number, raw, False, in_display


def inline_math_spans(text):
    """Pair single dollars left to right; return spans and unpaired count."""
    positions = [m.start() for m in re.finditer(r"(?<!\\)\$", text)]
    spans = [(positions[i], positions[i + 1])
             for i in range(0, len(positions) - 1, 2)]
    return spans, len(positions) % 2


def lint(path):
    lines = path.read_text().split("\n")
    errors, notes = [], []

    def err(number, message):
        errors.append(f"{path.name}:{number}: {message}")

    heading_levels = []
    display_open_line = None
    stories = 0
    for number, raw, in_code, in_display in split_regions(lines):
        stripped = raw.strip()
        if in_code:
            if stripped.startswith("```"):
                continue
            if len(raw) > MAX_CODE_LINE and not raw.lstrip().startswith("#"):
                err(number, f"code line has {len(raw)} characters "
                            f"(limit {MAX_CODE_LINE})")
            continue

        if stripped == "$$":
            display_open_line = None if display_open_line else number
            continue
        if in_display:
            if "\t" in raw:
                err(number, "TAB character inside display math")
            for cmd in re.findall(r"\\([A-Za-z]+)", raw):
                if cmd not in KNOWN_TEX:
                    err(number, f"unrecognized TeX command \\{cmd}")
            if stripped == "":
                err(number, "blank line inside a $$ display block")
            for pattern, hint in STYLE_SLIPS:
                if re.search(pattern, raw):
                    err(number, f"style slip in display math: {hint}")
            continue

        if "$$" in raw:
            err(number, "'$$' must be on its own line")
        if re.match(r"\s*>\s*\[!\w+\]", raw):
            err(number, "GitHub alert syntax; use a plain blockquote")
        if "STORY NEEDED" in raw:
            stories += 1
        m = re.match(r"(#{1,6}) ", raw)
        if m:
            heading_levels.append((number, len(m.group(1))))
        if "\t" in raw:
            err(number, "TAB character (it swallows a following 't' in "
                        "'\\text' and similar); use spaces")
        if "—" in raw:
            err(number, "em dash")
        for pattern in BANNED:
            if re.search(pattern, raw, re.I):
                err(number, f"banned phrase matching /{pattern}/")
        for ch in raw:
            if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
                name = unicodedata.name(ch, "?")
                if not (unicodedata.category(ch).startswith("L")
                        and "LATIN" in name):
                    err(number, f"non-ASCII character {ch!r} ({name}); "
                                "write it as math or plain text")
                    break
        for cmd in re.findall(r"\\([A-Za-z]+)", raw):
            if cmd not in KNOWN_TEX:
                err(number, f"unrecognized TeX command \\{cmd}")
        spans, odd = inline_math_spans(raw)
        if odd:
            err(number, "odd number of single '$' on this line "
                        "(unpaired math, or an unescaped dollar sign)")
        in_table = raw.lstrip().startswith("|")
        for a, b in spans:
            inner = raw[a + 1:b]
            if inner != inner.strip() and inner.strip():
                err(number, f"spaces just inside dollars: ${inner[:30]}$")
            if in_table and re.search(r"(?<!\\)\|", inner):
                err(number, "bare '|' inside math in a table row; "
                            r"use \lvert and \rvert")
            for pattern, hint in STYLE_SLIPS:
                if re.search(pattern, inner):
                    err(number, f"style slip: {hint}")

    if display_open_line:
        err(display_open_line, "unclosed $$ display block")

    h1 = [n for n, level in heading_levels if level == 1]
    if len(h1) != 1:
        err(h1[1] if len(h1) > 1 else 1,
            f"expected exactly one '#' title, found {len(h1)}")
    previous = 0
    for number, level in heading_levels:
        if previous and level > previous + 1:
            err(number, f"heading jumps from level {previous} to {level}")
        previous = level
    if stories:
        notes.append(f"{stories} [STORY NEEDED] placeholder(s)")
    return errors, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("chapters", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        paths = sorted((REPO / "chapters").glob("ch*.md"))
        paths += sorted((REPO / "chapters").glob("app*.md"))
    else:
        paths = [Path(p).resolve() for p in args.chapters]
    if not paths:
        parser.error("give chapter files or --all")
    total = 0
    for path in paths:
        errors, notes = lint(path)
        total += len(errors)
        status = "clean" if not errors else f"{len(errors)} problem(s)"
        extra = f" ({'; '.join(notes)})" if notes else ""
        print(f"{path.name}: {status}{extra}")
        for e in errors[:25]:
            print("   ", e)
        if len(errors) > 25:
            print(f"    ... and {len(errors) - 25} more")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
