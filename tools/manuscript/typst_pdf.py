"""Compile Markdown manuscripts to a typeset PDF using Pandoc and Typst."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tools.manuscript.frontmatter import front_matter, typ_str


def _find_binary(name: str) -> Optional[str]:
    """Locate executable binary across PATH, active Python bin dir, and base conda."""
    # 1. Standard PATH
    found = shutil.which(name)
    if found:
        return found

    # 2. Alongside the active Python interpreter
    py_dir = Path(sys.executable).parent
    candidate = py_dir / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)

    # 3. If running inside a conda env, check base conda bin
    if py_dir.parent.name == "envs":
        base_bin = py_dir.parent.parent / "bin" / name
        if base_bin.is_file() and os.access(base_bin, os.X_OK):
            return str(base_bin)

    return None


def _find_pandoc() -> Optional[str]:
    """pandoc on PATH, next to Python, or inside the pypandoc_binary wheel."""
    found = _find_binary("pandoc")
    if found:
        return found
    try:
        import pypandoc

        path = pypandoc.get_pandoc_path()
        return path if path and Path(path).is_file() else None
    except Exception:
        return None


def fix_typst_compatibility(
    typst_content: str, font: str = "Liberation Serif"
) -> str:
    """Apply compatibility normalization between Pandoc output and modern Typst."""
    content = typst_content

    # 1. Provide fallback font if Pandoc defaults to empty font list
    content = re.sub(r"font:\s*\(\)", f'font: ("{font}",)', content)

    # 2. Modern Typst symbol updates
    content = content.replace("times.circle", "times.o")
    content = content.replace("minus.circle", "minus.o")
    content = content.replace("angle.l", "chevron.l").replace("angle.r", "chevron.r")
    content = re.sub(r"(\bsect\b)", "inter", content)

    # 3. Math expression normalization:
    # Convert escaped parens \( \) to normal parens ( ) to prevent Typst treating them as arrays
    # Escape unescaped semicolons so they are not parsed as function/array row delimiters
    def fix_math_block(match: re.Match) -> str:
        math_text = match.group(0)
        math_text = math_text.replace(r"\(", "(").replace(r"\)", ")")
        math_text = re.sub(r"(?<!\\);", r"\;", math_text)
        return math_text

    # Match genuine unescaped math expressions ($ ... $)
    content = re.sub(
        r"(?<!\\)\$(.*?)(?<!\\)\$", fix_math_block, content, flags=re.DOTALL
    )

    # 4. Handle citation keys like @key when no .bib bibliography is mounted
    content = re.sub(r"@([a-zA-Z][a-zA-Z0-9_-]*)", r"[\1]", content)

    return content


TEMPLATE = Path(__file__).parent / "templates" / "book.typ"

BOOK_RE = re.compile(r"<!-- manuscript:book (.*?) -->", re.DOTALL)
FRONT_RE = re.compile(
    r"<!-- manuscript:front:begin -->.*?<!-- manuscript:front:end -->\s*(?:---\s*)?",
    re.DOTALL,
)
PART_RE = re.compile(
    r"<!-- manuscript:part:begin (.*?) -->.*?<!-- manuscript:part:end -->",
    re.DOTALL,
)
ANCHOR_RE = re.compile(r'^<a id="[^"]*"></a>\s*$', re.MULTILINE)
CHAPTER_RE = re.compile(r"^# Chapter (\d+): (.*)$", re.MULTILINE)


def read_book_meta(markdown: str) -> Optional[Dict]:
    """The book structure that `combine` stored in MANUSCRIPT.md."""
    m = BOOK_RE.search(markdown)
    return json.loads(m.group(1)) if m else None


def _raw_typst(code: str) -> str:
    return f"```{{=typst}}\n{code}\n```"


def prepare_body(markdown: str, book: Optional[Dict]) -> str:
    """Turn the combined Markdown into what pandoc should see.

    The Markdown front matter and part headings are for GitHub readers; the
    PDF gets typeset equivalents (title page, contents, part pages, chapter
    openers) from the template instead.
    """
    text = BOOK_RE.sub("", markdown)
    text = FRONT_RE.sub("", text)

    def part_block(m: re.Match) -> str:
        data = json.loads(m.group(1))
        titles = {
            c["number"]: c["title"]
            for part in (book or {}).get("parts", [])
            for c in part["chapters"]
        }
        rows = ", ".join(
            f"({n}, {typ_str(titles.get(n, ''))})" for n in data["chapters"]
        )
        code = (
            f"#part({typ_str(data['label'])}, {typ_str(data['title'])}, "
            f"{typ_str(data['description'])}, ({rows},))"
        )
        return _raw_typst(code)

    text = PART_RE.sub(part_block, text)
    text = ANCHOR_RE.sub("", text)

    first = {"done": False}

    def chapter_block(m: re.Match) -> str:
        n, title = int(m.group(1)), m.group(2).strip()
        code = f"#chapter({n}, {typ_str(title)}) <ch{n}>"
        if not first["done"]:
            first["done"] = True
            code = (
                '#set page(numbering: "1")\n#counter(page).update(1)\n'
                + code
            )
        return _raw_typst(code)

    text = CHAPTER_RE.sub(chapter_block, text)
    # Drop the rules that only separated blocks in the Markdown file.
    text = re.sub(r"^---\s*\n+(?=```\{=typst\})", "", text, flags=re.MULTILINE)
    return text


def build_typst(markdown: str, font: str, paper: str, body_typ: str) -> str:
    """Template + generated front matter + the pandoc body."""
    book = read_book_meta(markdown)
    fonts = [font, "New Computer Modern", "Libertinus Serif", "Liberation Serif"]
    seen = []
    for f in fonts:
        if f not in seen:
            seen.append(f)
    template = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__PAPER__", paper)
        .replace("__FONTS__", ", ".join(typ_str(f) for f in seen) + ",")
    )
    front = front_matter(book) if book else ""
    return "\n".join([template, front, body_typ])


def _compile_typst(typ_file: Path, output_pdf: Path, root: Path) -> None:
    typst_bin = _find_binary("typst")
    if typst_bin:
        res = subprocess.run(
            [typst_bin, "compile", "--root", str(root),
             str(typ_file), str(output_pdf)],
            capture_output=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                "Typst compilation failed:\n"
                + res.stderr.decode("utf-8").strip()
            )
        return
    try:
        import typst
    except ImportError:
        raise RuntimeError(
            "Neither the typst executable nor the `typst` Python package "
            "was found. Install one (for example `pip install typst`)."
        )
    try:
        typst.compile(str(typ_file), output=str(output_pdf), root=str(root))
    except Exception as e:
        raise RuntimeError(f"Typst compilation failed:\n{e}")


def compile_markdown_to_pdf(
    input_md: Path,
    output_pdf: Optional[Path] = None,
    project_root: Optional[Path] = None,
    font: str = "New Computer Modern",
    paper: str = "us-letter",
    keep_typ: bool = False,
) -> Path:
    """Compile a combined manuscript (see `combine`) to PDF.

    Args:
        input_md: Path to MANUSCRIPT.md.
        output_pdf: Output path. Defaults to input_md with a .pdf extension.
        project_root: Directory that must contain the Markdown file and
                      every image it references. Defaults to its parent.
        font: Preferred body font; the others in the template are fallbacks.
        paper: Paper size name understood by Typst.
        keep_typ: Keep the generated .typ file next to the Markdown.

    Returns:
        Path to the generated PDF file.
    """
    input_md = input_md.resolve()
    output_pdf = (
        input_md.with_suffix(".pdf") if output_pdf is None
        else output_pdf.resolve()
    )
    project_root = (
        input_md.parent if project_root is None else project_root.resolve()
    )

    pandoc_bin = _find_pandoc()
    if not pandoc_bin:
        raise RuntimeError(
            "pandoc executable not found in PATH or conda environment. "
            "Please install pandoc (or `pip install pypandoc_binary`)."
        )

    markdown = input_md.read_text(encoding="utf-8")
    book = read_book_meta(markdown)
    prepared = prepare_body(markdown, book)

    # Body only (no -s): the template supplies everything around it.
    pandoc_res = subprocess.run(
        [pandoc_bin, "-f", "markdown", "-t", "typst",
         "--resource-path", f"{input_md.parent}:{project_root}"],
        input=prepared, capture_output=True, text=True,
    )
    if pandoc_res.returncode != 0:
        raise RuntimeError(
            f"Pandoc conversion failed:\n{pandoc_res.stderr.strip()}"
        )

    body = fix_typst_compatibility(pandoc_res.stdout, font=font)
    typ_file = input_md.with_name(f".{input_md.stem}.typ")
    typ_file.write_text(
        build_typst(markdown, font, paper, body), encoding="utf-8"
    )
    try:
        _compile_typst(typ_file, output_pdf, project_root)
    finally:
        if not keep_typ and typ_file.exists():
            typ_file.unlink()

    return output_pdf
