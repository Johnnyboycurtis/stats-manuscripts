"""Combine multi-chapter markdown documents into a unified manuscript."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from markdown_it import MarkdownIt

from tools.manuscript.book import Book, load_book


def github_slug(title: str) -> str:
    """Generate a GitHub-compatible markdown heading anchor slug."""
    # Strip markdown bold, italics, code markup
    text = re.sub(r"[*_`]", "", title)
    # Remove math markers ($...$) but preserve the math expression content
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    # Strip LaTeX command prefixes like \mathbf, \operatorname, etc.
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Lowercase
    text = text.lower()
    # Retain alphanumeric characters, whitespace, and hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    # Collapse spaces into hyphens
    return re.sub(r"\s+", "-", text.strip())


def rewrite_image_paths(
    content: str, chapter_file: Path, target_dir: Path
) -> str:
    """Rewrite relative image paths so they resolve relative to target_dir."""
    md = MarkdownIt()
    tokens = md.parse(content)

    replacements = {}
    for token in tokens:
        if token.type == "inline" and token.children:
            for child in token.children:
                if child.type == "image":
                    src = child.attrs.get("src")
                    if src and not src.startswith(
                        ("http://", "https://", "data:", "/")
                    ):
                        abs_img_path = (chapter_file.parent / src).resolve()
                        new_rel_path = Path(
                            os.path.relpath(abs_img_path, target_dir)
                        ).as_posix()
                        replacements[src] = new_rel_path

    result = content
    for old_src, new_src in replacements.items():
        pattern = rf"\(\s*{re.escape(old_src)}(\s+[^)]*)?\)"

        def repl(match: re.Match) -> str:
            extra = match.group(1) or ""
            return f"({new_src}{extra})"

        result = re.sub(pattern, repl, result)

    return result


def extract_headings(content: str) -> List[Tuple[int, str]]:
    """Extract h1 and h2 markdown headings, ignoring fenced code blocks."""
    headings = []
    in_code_block = False
    for line in content.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            headings.append((1, title))
        elif line.startswith("## ") and not line.startswith("### "):
            title = line[3:].strip()
            headings.append((2, title))
    return headings


def find_chapter_files(manuscript_dir: Path, chapters_dir: Optional[Path] = None) -> List[Path]:
    """Find and return chapter markdown files in natural sorted order."""
    if chapters_dir is not None:
        search_dir = chapters_dir
    else:
        # Check standard locations
        if (manuscript_dir / "current" / "chapters").is_dir():
            search_dir = manuscript_dir / "current" / "chapters"
        elif (manuscript_dir / "chapters").is_dir():
            search_dir = manuscript_dir / "chapters"
        else:
            search_dir = manuscript_dir

    # Find chapters matching ch*.md
    files = sorted(search_dir.glob("ch*.md"))
    if not files:
        files = sorted(search_dir.glob("*.md"))
    return files


def extract_book_title(manuscript_dir: Path, default: str = "Manuscript") -> str:
    """Extract book title from README.md or OUTLINE.md if available."""
    for candidate in [
        manuscript_dir / "current" / "README.md",
        manuscript_dir / "README.md",
        manuscript_dir / "current" / "OUTLINE.md",
        manuscript_dir / "OUTLINE.md",
    ]:
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
    return default


SECTION_NUMBER = re.compile(r"^(\d+)\.(\d+)\s")


def section_anchor(chapter: int, title: str) -> str:
    """Stable anchor for an h2: s3-1 for '3.1 ...', else ch3-<slug>."""
    m = SECTION_NUMBER.match(title)
    if m:
        return f"s{m.group(1)}-{m.group(2)}"
    return f"ch{chapter}-{github_slug(title)}"


def book_comment(book: Book) -> str:
    """The book's structure as an HTML comment, so a later PDF step can
    rebuild the front matter from MANUSCRIPT.md alone."""
    data = {
        "title": book.title,
        "subtitle": book.subtitle,
        "author": book.author,
        "status": book.status,
        "tagline": book.tagline,
        "stamp": book.stamp,
        "parts": [
            {
                "label": p.label,
                "title": p.title,
                "description": p.description,
                "chapters": [
                    {"number": c.number, "title": c.title,
                     "description": c.description}
                    for c in p.chapters
                ],
            }
            for p in book.parts
        ],
    }
    text = json.dumps(data, ensure_ascii=False)
    text = text.replace("--", "-\\u002d").replace(">", "\\u003e")
    return f"<!-- manuscript:book {text} -->"


def front_matter(book: Book, sections: Dict[int, List[str]]) -> str:
    """Title block and two-level contents, for readers of the Markdown."""
    out = [
        "<!-- manuscript:front:begin -->",
        '<div align="center">',
        "",
        f"# {book.title}",
        "",
    ]
    if book.subtitle:
        out += [f"### {book.subtitle}", ""]
    if book.author:
        out += [f"**{book.author}**", ""]
    line = ", ".join(x for x in (book.status, book.stamp) if x)
    if line:
        out += [f"{line}", ""]
    if book.tagline:
        out += [f"*{book.tagline}*", ""]
    out += ["</div>", "", "---", "", "## Contents", ""]

    for part in book.parts:
        if part.title:
            out += [f"### {part.label}: {part.title}", ""]
            if part.description:
                out += [f"*{part.description}*", ""]
        else:
            out += [f"### {part.label}", ""]
        for c in part.chapters:
            out.append(f"- **[Chapter {c.number}: {c.title}](#ch{c.number})**")
            if c.description:
                out.append(f"  <br>{c.description}")
        out.append("")

    out += [
        "<details>",
        "<summary><strong>Detailed contents</strong> (every section)"
        "</summary>",
        "",
    ]
    for c in book.chapters:
        out.append(f"- **[Chapter {c.number}: {c.title}](#ch{c.number})**")
        for heading in sections[c.number]:
            out.append(
                f"  - [{heading}](#{section_anchor(c.number, heading)})"
            )
    out += ["", "</details>", "<!-- manuscript:front:end -->"]
    return "\n".join(out)


def part_divider(index: int, part) -> str:
    """Visible Part heading plus a marker the PDF step turns into a page."""
    payload = json.dumps(
        {"label": part.label, "title": part.title,
         "description": part.description,
         "chapters": [c.number for c in part.chapters]},
        ensure_ascii=False,
    ).replace("--", "-\\u002d").replace(">", "\\u003e")
    lines = [
        f"<!-- manuscript:part:begin {payload} -->",
        f'<a id="part-{index}"></a>',
        "",
        f"# {part.label}: {part.title}",
        "",
    ]
    if part.description:
        lines += [f"*{part.description}*", ""]
    lines.append("<!-- manuscript:part:end -->")
    return "\n".join(lines)


def combine_manuscript(
    manuscript_dir: Path,
    output_md: Optional[Path] = None,
    chapters_dir: Optional[Path] = None,
    title: Optional[str] = None,
) -> Path:
    """Combine individual chapter files into a single unified markdown document.

    The result opens with a title block and a contents page grouped by part
    (with a collapsible section-level contents), then the chapters, with a
    divider before each part. Structure that the PDF step needs is carried
    in HTML comments (`manuscript:book`, `:front:`, `:part:`), which
    Markdown viewers ignore.

    Args:
        manuscript_dir: Root directory of the manuscript project.
        output_md: Path where the merged markdown file will be written.
                   Defaults to `<manuscript_dir>/MANUSCRIPT.md`.
        chapters_dir: Directory containing individual chapter files.
        title: Title of the manuscript. Read from book.toml if omitted.

    Returns:
        Path to the generated combined markdown file.
    """
    manuscript_dir = manuscript_dir.resolve()
    if output_md is None:
        output_md = manuscript_dir / "MANUSCRIPT.md"
    else:
        output_md = output_md.resolve()

    chapter_files = find_chapter_files(manuscript_dir, chapters_dir)
    if not chapter_files:
        raise FileNotFoundError(
            f"No chapter files found in {manuscript_dir} or its chapters subdirectories."
        )

    book = load_book(manuscript_dir, chapter_files, title=title)
    target_dir = output_md.parent

    sections: Dict[int, List[str]] = {}
    for c in book.chapters:
        sections[c.number] = [
            h for lvl, h in extract_headings(c.path.read_text("utf-8"))
            if lvl == 2
        ]

    blocks = [book_comment(book), front_matter(book, sections)]
    for index, part in enumerate(book.parts):
        if part.title:
            blocks.append(part_divider(index, part))
        for c in part.chapters:
            raw = c.path.read_text(encoding="utf-8")
            text = rewrite_image_paths(raw, c.path, target_dir)
            lines = text.split("\n")
            out_lines, in_code, seen_h1 = [], False, False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                if not in_code and line.startswith("# ") and not seen_h1:
                    out_lines += [f'<a id="ch{c.number}"></a>', "", line]
                    seen_h1 = True
                elif (not in_code and line.startswith("## ")
                      and not line.startswith("### ")):
                    anchor = section_anchor(c.number, line[3:].strip())
                    out_lines += [f'<a id="{anchor}"></a>', "", line]
                else:
                    out_lines.append(line)
            blocks.append("\n".join(out_lines))

    full_manuscript = "\n\n---\n\n".join(blocks) + "\n"
    output_md.write_text(full_manuscript, encoding="utf-8")
    return output_md
