"""Compile Markdown manuscripts to publication-grade PDF using Pandoc and Typst."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


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


def compile_markdown_to_pdf(
    input_md: Path,
    output_pdf: Optional[Path] = None,
    project_root: Optional[Path] = None,
    font: str = "Liberation Serif",
    paper: str = "us-letter",
) -> Path:
    """Compile a Markdown document to PDF via Pandoc and Typst.

    Args:
        input_md: Path to the input Markdown file.
        output_pdf: Path where the output PDF will be created.
                    Defaults to input_md with a .pdf extension.
        project_root: Root directory for resolving images and assets.
                      Defaults to input_md's parent directory.
        font: Main body font family. Defaults to 'Liberation Serif'.
        paper: Paper size format. Defaults to 'us-letter'.

    Returns:
        Path to the generated PDF file.
    """
    input_md = input_md.resolve()
    if output_pdf is None:
        output_pdf = input_md.with_suffix(".pdf")
    else:
        output_pdf = output_pdf.resolve()

    if project_root is None:
        project_root = input_md.parent
    else:
        project_root = project_root.resolve()

    pandoc_bin = _find_binary("pandoc")
    if not pandoc_bin:
        raise RuntimeError(
            "pandoc executable not found in PATH or conda environment. Please install pandoc."
        )

    typst_bin = _find_binary("typst")
    if not typst_bin:
        raise RuntimeError(
            "typst executable not found in PATH or conda environment. Please install typst."
        )

    # Convert Markdown to Typst markup using Pandoc
    pandoc_cmd = [
        pandoc_bin,
        "-s",
        str(input_md),
        "-t",
        "typst",
        "-V",
        f"mainfont={font}",
        "-V",
        f"papersize={paper}",
        "--resource-path",
        f"{input_md.parent}:{project_root}",
    ]

    pandoc_res = subprocess.run(pandoc_cmd, capture_output=True, text=True)
    if pandoc_res.returncode != 0:
        raise RuntimeError(
            f"Pandoc conversion failed:\n{pandoc_res.stderr.strip()}"
        )

    # Apply compatibility adjustments
    typst_markup = fix_typst_compatibility(pandoc_res.stdout, font=font)

    # Compile with Typst
    typst_cmd = [
        typst_bin,
        "compile",
        "--root",
        str(project_root),
        "-",
        str(output_pdf),
    ]

    typst_res = subprocess.run(
        typst_cmd,
        input=typst_markup.encode("utf-8"),
        cwd=str(project_root),
        capture_output=True,
    )

    if typst_res.returncode != 0:
        raise RuntimeError(
            f"Typst compilation failed:\n{typst_res.stderr.decode('utf-8').strip()}"
        )

    return output_pdf
