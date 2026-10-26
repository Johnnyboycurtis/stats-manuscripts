"""Generate the Typst source for the title page and contents pages."""

import json
from typing import Dict, List


def typ_str(s: str) -> str:
    """A Typst string literal."""
    return json.dumps(s, ensure_ascii=False)


# Deterministic scatter around the curve, as (h, offset) in unit coordinates.
_SCATTER = [
    (0.05, 0.020), (0.10, -0.030), (0.16, 0.035), (0.22, -0.020),
    (0.29, 0.030), (0.36, -0.040), (0.44, 0.025), (0.52, -0.020),
    (0.61, 0.035), (0.70, -0.030), (0.79, 0.020), (0.88, -0.025),
    (0.96, 0.030),
]


def variogram_motif(width_in: float = 4.6, height_in: float = 1.9) -> str:
    """A small drawn semivariogram: nugget, sill, range, noisy estimates.

    A quiet nod to the book's job of decoding spatial jargon.
    """
    import math

    nugget, sill, rng = 0.18, 1.0, 0.62
    pts = []
    for i in range(0, 101):
        h = i / 100
        g = nugget + (sill - nugget) * (1 - math.exp(-3 * h / rng))
        pts.append((h, g))
    top = 1.25  # y-axis extent in gamma units

    def X(h: float) -> str:
        return f"{h * width_in:.4f}in"

    def Y(g: float) -> str:
        return f"{(1 - g / top) * height_in:.4f}in"

    curve_pts = ", ".join(f"({X(h)}, {Y(g)})" for h, g in pts)
    dots = []
    for h, e in _SCATTER:
        g = nugget + (sill - nugget) * (1 - math.exp(-3 * h / rng)) + e * 3
        dots.append(
            f"place(top + left, dx: {X(h)} - 1.8pt, dy: {Y(g)} - 1.8pt, "
            f"circle(radius: 1.8pt, fill: accent.lighten(55%), stroke: none))"
        )
    sill_y = Y(sill)
    nug_y = Y(nugget)
    range_x = X(rng)
    base = Y(0)
    lines = [
        f"#box(width: {width_in}in, height: {height_in}in)[",
        "  #set text(size: 8.5pt, fill: muted, style: \"italic\")",
        f"  #place(top + left, line(start: (0in, 0in), end: (0in, {base}), stroke: 0.7pt + muted))",
        f"  #place(top + left, line(start: (0in, {base}), end: ({width_in}in, {base}), stroke: 0.7pt + muted))",
        f"  #place(top + left, line(start: (0in, {sill_y}), end: ({width_in}in, {sill_y}), stroke: (paint: muted, thickness: 0.6pt, dash: \"dashed\")))",
        f"  #place(top + left, line(start: ({range_x}, {sill_y}), end: ({range_x}, {base}), stroke: (paint: muted, thickness: 0.6pt, dash: \"dotted\")))",
        f"  #place(top + left, curve(stroke: 1.8pt + accent, curve.move((0in, {Y(nugget)})), "
        + "".join(
            f"curve.line(({X(h)}, {Y(g)})), " for h, g in pts[1:]
        )
        + "))",
    ]
    lines += [f"  #{d}" for d in dots]
    lines += [
        f"  #place(top + left, dx: 0.1in, dy: {sill_y} - 1.35em)[sill]",
        f"  #place(top + left, dx: 0.12in, dy: {nug_y} + 0.25em)[nugget]",
        f"  #place(top + left, dx: {range_x} + 0.08in, dy: {base} - 1.5em)[range]",
        f"  #place(top + left, dx: {width_in}in - 0.9in, dy: {base} + 0.35em)[distance]",
        "]",
    ]
    return "\n".join(lines)


def title_page(book: Dict) -> str:
    stamp = ", ".join(x for x in (book.get("status"), book.get("stamp")) if x)
    out = [
        "#v(1.15in)",
        "#block(width: 100%)[",
        "  #set par(justify: false, leading: 0.32em)",
        f"  #text(size: 42pt, weight: \"bold\", hyphenate: false)[#{typ_str(book['title'])}]",
        "]",
        "#v(0.8em)",
        "#line(length: 2.6cm, stroke: 2.6pt + accent)",
        "#v(1.0em)",
    ]
    if book.get("subtitle"):
        out.append(
            f"#text(size: 17pt, fill: muted)[#{typ_str(book['subtitle'])}]"
        )
    out += ["#v(0.9in)", variogram_motif(), "#v(1fr)"]
    if book.get("tagline"):
        out.append(
            f"#text(size: 11.5pt, style: \"italic\")[#{typ_str(book['tagline'])}]"
        )
        out.append("#v(1.2em)")
    if book.get("author"):
        out.append(
            f"#text(size: 14pt, weight: \"bold\")[#{typ_str(book['author'])}]"
        )
        out.append("#v(0.25em)")
    if stamp:
        out.append(f"#text(size: 9.5pt, fill: muted)[#{typ_str(stamp)}]")
    out.append("#v(0.3in)")
    return "\n".join(out)


def page_title(text: str) -> str:
    return (
        "#block(below: 1.1em)[\n"
        f"  #text(size: 24pt, weight: \"bold\")[{text}]\n"
        "  #v(0.35em)\n"
        "  #line(length: 2.2cm, stroke: 2.2pt + accent)\n"
        "]\n"
    )


def contents_page(book: Dict) -> str:
    """One page: parts, chapters, one-line descriptions, page numbers."""
    out = ["#pagebreak()", page_title("Contents")]
    for part in book["parts"]:
        label = typ_str(part["label"].upper())
        title = part["title"]
        head = f"#text(size: 8pt, weight: \"bold\", tracking: 0.14em, fill: accent)[#{label}]"
        if title:
            head += f" #h(0.7em) #text(weight: \"bold\", size: 11pt)[#eval({typ_str(title)}, mode: \"markup\")]"
        out.append("#block(above: 0.95em, below: 0.3em, sticky: true)[" + head + "]")
        for ch in part["chapters"]:
            n = ch["number"]
            desc = ch.get("description", "")
            row = (
                f"#link(label(\"ch{n}\"))[#set text(fill: ink)\n#grid(columns: (2.2em, 1fr, 2.2em), "
                f"column-gutter: 4pt, align: (right, left, right), "
                f"text(fill: accent, weight: \"bold\")[{n}], "
                f"[#eval({typ_str(ch['title'])}, mode: \"markup\") #box(width: 1fr, repeat(text(fill: hairline)[. #h(1.5pt)]))], "
                f"page-of({n}))]"
            )
            out.append("#block(above: 0.42em, below: 0em, breakable: false)[")
            out.append("  " + row)
            if desc:
                out.append(
                    f"  #pad(left: 2.2em + 4pt, right: 2.6em, top: 0.1em, "
                    f"text(size: 8.3pt, fill: muted)[#{typ_str(desc)}])"
                )
            out.append("]")
    return "\n".join(out)


def detailed_contents_page() -> str:
    return "\n".join(
        [
            "#pagebreak()",
            page_title("Detailed Contents"),
            "#show outline.entry.where(level: 1): it => {",
            "  v(0.9em, weak: true)",
            "  set text(weight: \"bold\", size: 10pt)",
            "  if it.element.supplement.text == \"Part\" {",
            "    set text(fill: accent)",
            "    it",
            "  } else {",
            "    it",
            "  }",
            "}",
            "#show outline.entry.where(level: 2): it => {",
            "  set text(size: 8.8pt)",
            "  set block(spacing: 0.42em)",
            "  it",
            "}",
            "#outline(title: none, depth: 2, indent: 1.2em)",
        ]
    )


def front_matter(book: Dict) -> str:
    return "\n".join(
        [
            title_page(book),
            contents_page(book),
            detailed_contents_page(),
        ]
    )
