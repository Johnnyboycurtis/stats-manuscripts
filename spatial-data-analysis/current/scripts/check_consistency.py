#!/usr/bin/env python3
"""Mechanical cross-chapter consistency checks.

Checks that need no judgment:
    - every "Section N.N(.N)" or "§N.N(.N)" reference points to a heading
      that exists in chapter N
    - every figure link points to a file that exists
    - every file in figures/chNN/ is linked from chapter NN
    - chapter titles in the files match the titles in OUTLINE.md
    - the section numbers listed in OUTLINE.md match the headings (and the
      numbered lab items) actually present in each chapter
    - every "First introduced" section in NOTATION.md exists
    - references to appendices (written later) are listed, not flagged

Usage:
    python3 scripts/check_consistency.py
Exit status is 1 if any problem is found.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CH = REPO / "chapters"
HEAD = re.compile(r"^(#{2,4}) ((?:\d+)(?:\.\d+){1,2})\b(.*)$")


def strip_code(text):
    out, in_code = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append("")
        else:
            out.append("" if in_code else line)
    return "\n".join(out)


def load_chapters():
    chapters = {}
    for path in sorted(CH.glob("ch*.md")):
        num = int(path.name[2:4])
        text = path.read_text()
        clean = strip_code(text)
        heads = {}
        for i, line in enumerate(clean.split("\n"), start=1):
            m = HEAD.match(line)
            if m:
                heads[m.group(2)] = (i, m.group(3).strip())
        title = re.match(r"# Chapter \d+: (.*)", text)
        chapters[num] = {"path": path, "text": text, "clean": clean,
                         "heads": heads,
                         "title": title.group(1).strip() if title else None}
    return chapters


REF = re.compile(r"(?:Sections?|§)\s*(\d+(?:\.\d+){1,2})"
                 r"((?:\s*(?:,|and|to)\s*\d+(?:\.\d+){1,2})*)")
EXTRA = re.compile(r"\d+(?:\.\d+){1,2}")


def check_refs(chapters, problems, appendix_refs):
    for num, c in chapters.items():
        for i, line in enumerate(c["clean"].split("\n"), start=1):
            for m in REF.finditer(line):
                refs = [m.group(1)] + EXTRA.findall(m.group(2) or "")
                for ref in refs:
                    target = int(ref.split(".")[0])
                    if target not in chapters:
                        problems.append(f"{c['path'].name}:{i}: reference "
                                        f"{ref}: no chapter {target}")
                        continue
                    if ref not in chapters[target]["heads"]:
                        problems.append(f"{c['path'].name}:{i}: reference "
                                        f"{ref} does not match a heading in "
                                        f"{chapters[target]['path'].name}")
            for m in re.finditer(r"Appendix ([A-D](?:\.\d+)?)", line):
                appendix_refs.setdefault(m.group(1), []).append(
                    f"{c['path'].name}:{i}")


def check_figures(chapters, problems):
    link = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for num, c in chapters.items():
        linked = set()
        for i, line in enumerate(c["text"].split("\n"), start=1):
            for m in link.finditer(line):
                target = (c["path"].parent / m.group(1)).resolve()
                linked.add(target.name)
                if not target.exists():
                    problems.append(f"{c['path'].name}:{i}: missing figure "
                                    f"{m.group(1)}")
        folder = REPO / "figures" / f"ch{num:02d}"
        if folder.exists():
            for f in sorted(folder.glob("*.png")):
                if f.name not in linked:
                    problems.append(f"{f.relative_to(REPO)} is not linked "
                                    f"from {c['path'].name}")


def parse_outline():
    text = (REPO / "OUTLINE.md").read_text()
    chapters, current = {}, None
    for line in text.split("\n"):
        m = re.match(r"### Chapter (\d+): (.*)", line)
        if m:
            current = int(m.group(1))
            chapters[current] = {"title": m.group(2).strip(), "nums": set()}
            continue
        if line.startswith("## ") or line.startswith("### Appendix"):
            current = None
        if current is not None:
            m = re.match(r"\s*\* ((?:\d+)(?:\.\d+){1,2})\b", line)
            if m:
                chapters[current]["nums"].add(m.group(1))
    return chapters


def check_outline(chapters, problems):
    outline = parse_outline()
    for num, c in chapters.items():
        o = outline.get(num)
        if not o:
            problems.append(f"OUTLINE.md has no Chapter {num}")
            continue
        if c["title"] != o["title"]:
            problems.append(f"{c['path'].name}: title '{c['title']}' differs "
                            f"from OUTLINE.md '{o['title']}'")
        have = set(c["heads"])
        missing = sorted(o["nums"] - have, key=natural)
        extra = sorted(have - o["nums"], key=natural)
        if missing:
            problems.append(f"{c['path'].name}: in OUTLINE.md but no heading: "
                            + ", ".join(missing[:12])
                            + (" ..." if len(missing) > 12 else ""))
        if extra:
            problems.append(f"{c['path'].name}: heading not in OUTLINE.md: "
                            + ", ".join(extra[:12])
                            + (" ..." if len(extra) > 12 else ""))


def natural(s):
    return [int(p) for p in s.split(".")]


def check_notation(chapters, problems):
    text = (REPO / "NOTATION.md").read_text()
    for i, line in enumerate(text.split("\n"), start=1):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        last = cells[-2] if len(cells) > 2 else ""
        for m in re.finditer(r"§(\d+(?:\.\d+){1,2})", last):
            ref = m.group(1)
            target = int(ref.split(".")[0])
            if target in chapters and ref not in chapters[target]["heads"]:
                problems.append(f"NOTATION.md:{i}: first-introduced section "
                                f"{ref} has no heading")


def main():
    chapters = load_chapters()
    problems, appendix_refs = [], {}
    check_refs(chapters, problems, appendix_refs)
    check_figures(chapters, problems)
    check_outline(chapters, problems)
    check_notation(chapters, problems)
    print(f"{len(chapters)} chapters, "
          f"{sum(len(c['heads']) for c in chapters.values())} numbered "
          "headings checked")
    for p in problems:
        print("-", p)
    print(f"\n{len(problems)} problem(s)")
    print("\nAppendix references (appendices not yet written):")
    for key in sorted(appendix_refs):
        where = appendix_refs[key]
        print(f"  Appendix {key}: {len(where)} reference(s), "
              f"e.g. {where[0]}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
