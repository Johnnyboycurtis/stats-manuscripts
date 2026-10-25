#!/usr/bin/env python3
"""Run every ```python block of one or more chapters, top to bottom.

This is the check behind the rule in MARKDOWNSTYLE.md, Section 6.3: every
number in the text must come from running the chapter's own code.

Usage (from anywhere; code always runs from the repository root, which is
where chapters expect to run so that figures land in figures/chNN/):

    python3 scripts/run_chapter_code.py chapters/ch03_empirical_variography.md
    python3 scripts/run_chapter_code.py --all
    python3 scripts/run_chapter_code.py --all --save-output build/code_output

Options:
    --all             check every chapters/ch*.md and chapters/app*.md
    --show-output     print each chapter's output as well as the summary
    --save-output D   write each chapter's output to D/<chapter>.txt, so the
                      numbers can be compared with the text by eye or diff
    --timeout SEC     per-chapter time limit (default 900)

Exit status is 0 only if every chapter ran without an error.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_LINE = 79
FENCE = re.compile(r"```python\n(.*?)```", re.S)

# The chapters use small matrices (a few hundred rows). On a busy machine a
# multithreaded BLAS can make them thousands of times slower, so run the
# chapter code with one BLAS thread.
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1")


def extract_blocks(text):
    """Return the contents of all python code blocks, in order."""
    return FENCE.findall(text)


def long_lines(blocks):
    """Code lines over MAX_LINE characters, as (length, text) pairs."""
    return [(len(line), line) for block in blocks
            for line in block.splitlines() if len(line) > MAX_LINE]


def run_chapter(path, timeout):
    text = path.read_text()
    blocks = extract_blocks(text)
    result = {"name": path.name, "blocks": len(blocks), "status": "no code",
              "seconds": 0.0, "output": "", "long": long_lines(blocks)}
    if not blocks:
        return result
    source = "\n\n".join(blocks)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "chapter_code.py"
        script.write_text(source)
        start = time.time()
        try:
            proc = subprocess.run([sys.executable, str(script)], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=timeout, env=ENV)
            result["output"] = proc.stdout + proc.stderr
            result["status"] = "ok" if proc.returncode == 0 else "ERROR"
        except subprocess.TimeoutExpired as exc:
            result["output"] = (exc.stdout or "") + (exc.stderr or "")
            result["status"] = "TIMEOUT"
        result["seconds"] = time.time() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("chapters", nargs="*", help="chapter .md files")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--show-output", action="store_true")
    parser.add_argument("--save-output", metavar="DIR")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if args.all:
        paths = sorted((REPO / "chapters").glob("ch*.md"))
        paths += sorted((REPO / "chapters").glob("app*.md"))
    else:
        paths = [Path(p).resolve() for p in args.chapters]
    if not paths:
        parser.error("give chapter files or --all")

    out_dir = Path(args.save_output) if args.save_output else None
    if out_dir:
        out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    print(f"{'chapter':40s} {'blocks':>6s} {'status':>8s} {'sec':>6s} "
          f"{'long lines':>10s}")
    for path in paths:
        r = run_chapter(path, args.timeout)
        print(f"{r['name']:40s} {r['blocks']:6d} {r['status']:>8s} "
              f"{r['seconds']:6.1f} {len(r['long']):10d}")
        if r["status"] in ("ERROR", "TIMEOUT"):
            failed = True
            print("    last lines of output:")
            for line in r["output"].strip().splitlines()[-8:]:
                print("    |", line)
        for length, line in r["long"][:3]:
            print(f"    code line of {length} characters: {line.strip()[:60]}")
        if args.show_output:
            print(r["output"])
        if out_dir:
            (out_dir / (path.stem + ".txt")).write_text(r["output"])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
