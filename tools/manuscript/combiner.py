"""Combine multi-chapter markdown documents into a unified manuscript."""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from markdown_it import MarkdownIt


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


def combine_manuscript(
    manuscript_dir: Path,
    output_md: Optional[Path] = None,
    chapters_dir: Optional[Path] = None,
    title: Optional[str] = None,
) -> Path:
    """Combine individual chapter files into a single unified markdown document.

    Args:
        manuscript_dir: Root directory of the manuscript project.
        output_md: Path where the merged markdown file will be written.
                   Defaults to `<manuscript_dir>/MANUSCRIPT.md`.
        chapters_dir: Directory containing individual chapter files.
        title: Title of the manuscript. Extracted automatically if omitted.

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

    if title is None:
        title = extract_book_title(manuscript_dir)

    target_dir = output_md.parent

    toc_lines = [
        f"# {title}\n",
        "## Table of Contents\n",
    ]

    all_chapters_content = []

    for ch in chapter_files:
        raw_text = ch.read_text(encoding="utf-8")
        headings = extract_headings(raw_text)
        for level, heading_title in headings:
            slug = github_slug(heading_title)
            indent = "  " * (level - 1)
            toc_lines.append(f"{indent}- [{heading_title}](#{slug})")

        transformed_text = rewrite_image_paths(raw_text, ch, target_dir)
        all_chapters_content.append(transformed_text)

    full_manuscript = (
        "\n".join(toc_lines)
        + "\n\n---\n\n"
        + "\n\n---\n\n".join(all_chapters_content)
        + "\n"
    )

    output_md.write_text(full_manuscript, encoding="utf-8")
    return output_md
