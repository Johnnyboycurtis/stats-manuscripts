"""Command-line interface for manuscript compilation tools."""

import argparse
import sys
import time
from pathlib import Path

from tools.manuscript.combiner import combine_manuscript
from tools.manuscript.typst_pdf import compile_markdown_to_pdf


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.manuscript",
        description="Manuscript assembly and PDF generation tools for stats-manuscripts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Combine subcommand
    combine_parser = subparsers.add_parser(
        "combine",
        help="Combine chapter Markdown files into a single unified Markdown file.",
    )
    combine_parser.add_argument(
        "manuscript_dir",
        type=Path,
        help="Root directory of the manuscript project.",
    )
    combine_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output combined Markdown path (default: <manuscript_dir>/MANUSCRIPT.md).",
    )
    combine_parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=None,
        help="Custom directory containing chapter files.",
    )
    combine_parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom manuscript title.",
    )

    # 2. PDF subcommand
    pdf_parser = subparsers.add_parser(
        "pdf",
        help="Compile a Markdown file into a PDF using Pandoc and Typst.",
    )
    pdf_parser.add_argument(
        "input_md",
        type=Path,
        help="Path to the input Markdown file to compile.",
    )
    pdf_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: input_md with .pdf extension).",
    )
    pdf_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for asset resolution (default: input_md parent).",
    )
    pdf_parser.add_argument(
        "--font",
        type=str,
        default="Liberation Serif",
        help="Main body font (default: 'Liberation Serif').",
    )
    pdf_parser.add_argument(
        "--paper",
        type=str,
        default="us-letter",
        help="Paper size (default: 'us-letter').",
    )

    # 3. Build subcommand (both combine and pdf)
    build_parser = subparsers.add_parser(
        "build",
        help="Combine chapters into Markdown and compile directly into PDF.",
    )
    build_parser.add_argument(
        "manuscript_dir",
        type=Path,
        help="Root directory of the manuscript project.",
    )
    build_parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output Markdown path (default: <manuscript_dir>/MANUSCRIPT.md).",
    )
    build_parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Output PDF path (default: <manuscript_dir>/MANUSCRIPT.pdf).",
    )
    build_parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=None,
        help="Custom directory containing chapter files.",
    )
    build_parser.add_argument(
        "--font",
        type=str,
        default="Liberation Serif",
        help="Main body font (default: 'Liberation Serif').",
    )
    build_parser.add_argument(
        "--paper",
        type=str,
        default="us-letter",
        help="Paper size (default: 'us-letter').",
    )

    args = parser.parse_args()

    try:
        t0 = time.time()
        if args.command == "combine":
            out_file = combine_manuscript(
                manuscript_dir=args.manuscript_dir,
                output_md=args.output,
                chapters_dir=args.chapters_dir,
                title=args.title,
            )
            print(f"[Done] Generated combined Markdown: {out_file} ({time.time() - t0:.2f}s)")

        elif args.command == "pdf":
            out_file = compile_markdown_to_pdf(
                input_md=args.input_md,
                output_pdf=args.output,
                project_root=args.root,
                font=args.font,
                paper=args.paper,
            )
            print(f"[Done] Compiled PDF: {out_file} ({time.time() - t0:.2f}s)")

        elif args.command == "build":
            out_md = combine_manuscript(
                manuscript_dir=args.manuscript_dir,
                output_md=args.output_md,
                chapters_dir=args.chapters_dir,
            )
            print(f"[1/2] Combined Markdown generated: {out_md} ({time.time() - t0:.2f}s)")

            t1 = time.time()
            out_pdf = compile_markdown_to_pdf(
                input_md=out_md,
                output_pdf=args.output_pdf,
                project_root=args.manuscript_dir,
                font=args.font,
                paper=args.paper,
            )
            print(f"[2/2] PDF compiled: {out_pdf} ({time.time() - t1:.2f}s)")
            print(f"[Finished] Full build complete in {time.time() - t0:.2f}s")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
