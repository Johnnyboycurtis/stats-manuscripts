"""Book metadata and structure: the material that surrounds the chapters."""

import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Chapter:
    number: int
    title: str
    description: str
    path: Path


@dataclass
class Part:
    label: str
    title: str
    description: str
    chapters: List[Chapter] = field(default_factory=list)


@dataclass
class Book:
    title: str
    subtitle: str
    author: str
    status: str
    tagline: str
    parts: List[Part]
    stamp: str

    @property
    def chapters(self) -> List[Chapter]:
        return [c for p in self.parts for c in p.chapters]


def chapter_number(path: Path) -> Optional[int]:
    """Chapter number from a file name such as ch04_kriging_blup.md."""
    m = re.match(r"ch(\d+)", path.name)
    return int(m.group(1)) if m else None


def first_heading_title(text: str) -> Optional[str]:
    """Title of the first h1, without the 'Chapter N:' prefix."""
    for line in text.splitlines():
        m = re.match(r"# (?:Chapter (\d+):\s*)?(.*)$", line)
        if m and not line.startswith("## "):
            return m.group(2).strip()
    return None


def find_book_toml(manuscript_dir: Path) -> Optional[Path]:
    for candidate in (
        manuscript_dir / "current" / "book.toml",
        manuscript_dir / "book.toml",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_book(
    manuscript_dir: Path,
    chapter_files: List[Path],
    title: Optional[str] = None,
    today: Optional[date] = None,
) -> Book:
    """Read book.toml (when present) and attach the chapter files to it.

    Without a book.toml the chapters form a single untitled part, so the
    tools still work on a bare folder of chapter files.
    """
    meta: Dict = {}
    toml_path = find_book_toml(manuscript_dir)
    if toml_path is not None:
        meta = tomllib.loads(toml_path.read_text(encoding="utf-8"))

    by_number: Dict[int, Chapter] = {}
    for f in chapter_files:
        n = chapter_number(f)
        if n is None:
            continue
        t = first_heading_title(f.read_text(encoding="utf-8")) or f.stem
        desc = meta.get("chapters", {}).get(str(n), "")
        by_number[n] = Chapter(n, t, desc, f)

    parts: List[Part] = []
    placed = set()
    for p in meta.get("parts", []):
        chs = [by_number[n] for n in p.get("chapters", []) if n in by_number]
        placed.update(c.number for c in chs)
        if chs:
            parts.append(
                Part(p.get("label", ""), p.get("title", ""),
                     p.get("description", ""), chs)
            )
    rest = [by_number[n] for n in sorted(by_number) if n not in placed]
    if rest:
        parts.append(Part("", "", "", rest))

    today = today or date.today()
    return Book(
        title=title or meta.get("title", "Manuscript"),
        subtitle=meta.get("subtitle", ""),
        author=meta.get("author", ""),
        status=meta.get("status", ""),
        tagline=meta.get("tagline", ""),
        parts=parts,
        stamp=today.strftime("%B %Y"),
    )
