// Book template for manuscripts compiled from Markdown.
// Python (tools/manuscript/typst_pdf.py) fills the __PLACEHOLDERS__, appends
// the generated front matter, then the body that pandoc produced.

#let paper = "__PAPER__"
#let body-font = (__FONTS__)
#let mono-font = ("DejaVu Sans Mono", "Liberation Mono")

#let ink = rgb("#1b1f24")
#let accent = rgb("#1f4e79")
#let muted = rgb("#6a737d")
#let hairline = rgb("#d5dae0")
#let wash = rgb("#f4f7fa")

// --- definitions pandoc's Typst output relies on ---------------------------
#let horizontalrule = align(center, line(length: 30%, stroke: 0.5pt + hairline))

#show terms.item: it => block(breakable: false)[
  #text(weight: "bold")[#it.term]
  #block(inset: (left: 1.5em, top: -0.4em))[#it.description]
]

// --- page, text, paragraphs -------------------------------------------------
#set page(
  paper: paper,
  margin: (x: 1.3in, top: 1.15in, bottom: 1.1in),
  numbering: "i",
)
#set text(font: body-font, size: 10.5pt, fill: ink, lang: "en", hyphenate: true)
#set par(justify: true, leading: 0.68em, spacing: 0.95em)
#set smartquote(enabled: false)
#set quote(quotes: false)
#set list(indent: 0.6em, body-indent: 0.6em)
#set enum(indent: 0.6em, body-indent: 0.6em)
#show link: set text(fill: accent)
#show math.equation.where(block: true): set block(spacing: 1.1em)

// --- the page's furniture ---------------------------------------------------
#let page-has-opener() = {
  let here-page = here().page()
  query(heading.where(level: 1)).any(h => h.location().page() == here-page)
}

#let running-header = context {
  let pg = counter(page).get().first()
  let chapters = query(heading.where(level: 1).before(here()))
  if page-has-opener() or chapters.len() == 0 {
    none
  } else {
    let ch = chapters.last()
    let sections = query(heading.where(level: 2).before(here()))
    set text(size: 8.5pt, fill: muted)
    if calc.even(pg) {
      smallcaps[#numbering(ch.numbering, 1) #h(0.7em) #ch.body]
    } else if sections.len() > 0 {
      align(right, sections.last().body)
    }
    v(-0.4em)
    line(length: 100%, stroke: 0.4pt + hairline)
  }
}

#let running-footer = context {
  let here-page = here().page()
  let on-part-page = query(heading.where(level: 1)).any(
    h => h.location().page() == here-page and h.supplement.text == "Part"
  )
  if not on-part-page and here-page > 1 {
    align(center, text(size: 9pt, fill: muted)[
      #counter(page).display(page.numbering)
    ])
  }
}

#set page(header: running-header, footer: running-footer)

// --- headings ---------------------------------------------------------------
#let chapter(n, title) = heading(
  level: 1,
  supplement: [Chapter],
  numbering: (..) => [Chapter #n],
  eval(title, mode: "markup"),
)

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  let label-text = numbering(it.numbering, 1)
  v(1.1in)
  block(width: 100%, below: 2.2em)[
    #set par(justify: false, leading: 0.5em)
    #text(size: 10.5pt, fill: accent, weight: "bold", tracking: 0.16em)[
      #upper(label-text)
    ]
    #v(0.5em)
    #text(size: 25pt, weight: "bold", hyphenate: false)[#it.body]
    #v(0.7em)
    #line(length: 2.2cm, stroke: 2.2pt + accent)
  ]
}

#show heading.where(level: 2): it => {
  v(1.7em, weak: true)
  block(sticky: true, below: 0.8em)[
    #set par(justify: false, leading: 0.5em)
    #text(size: 14pt, weight: "bold")[#it.body]
  ]
}

#show heading.where(level: 3): it => {
  v(1.2em, weak: true)
  block(sticky: true, below: 0.6em)[
    #text(size: 11.5pt, weight: "bold", style: "italic")[#it.body]
  ]
}

#show heading.where(level: 4): it => {
  v(0.9em, weak: true)
  block(sticky: true, below: 0.5em)[#text(size: 10.5pt, style: "italic", weight: "bold")[#it.body]]
}

// --- boxes, code, tables, figures -------------------------------------------
#show quote.where(block: true): it => block(
  width: 100%,
  fill: wash,
  stroke: (left: 2.4pt + accent),
  inset: (x: 13pt, y: 10pt),
  radius: (right: 3pt),
  above: 1.1em,
  below: 1.1em,
  breakable: true,
)[
  #set par(justify: false)
  #it.body
]

#show raw: set text(font: mono-font)
#show raw.where(block: false): it => box(
  fill: luma(244),
  inset: (x: 2.5pt),
  outset: (y: 2.5pt),
  radius: 2pt,
  text(size: 0.9em, it),
)
#show raw.where(block: true): it => block(
  width: 100%,
  fill: luma(247),
  stroke: 0.5pt + luma(218),
  radius: 3pt,
  inset: 9pt,
  above: 1.1em,
  below: 1.1em,
  text(size: 7.8pt, it),
)

#set table(
  inset: (x: 6pt, y: 5pt),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt + ink } else { 0pt },
    bottom: if y == 0 { 0.5pt + ink } else { 0.3pt + hairline },
  ),
)
#show table: set text(size: 9.2pt)
#show table: set par(justify: false)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: image): set figure.caption(position: bottom)
#show figure: set block(above: 1.3em, below: 1.3em)
#show figure.caption: set text(size: 9.2pt)
#show figure.caption: set par(justify: false)

// --- parts ------------------------------------------------------------------
#let page-of(n) = context {
  let hits = query(label("ch" + str(n)))
  if hits.len() == 0 {
    []
  } else {
    numbering("1", counter(page).at(hits.first().location()).first())
  }
}

#let dotted() = box(width: 1fr, repeat(text(fill: hairline)[. #h(1.5pt)]))

#let part(kicker, title, description, chapters) = {
  heading(
    level: 1,
    supplement: [Part],
    numbering: (..) => [#kicker],
    eval(title, mode: "markup"),
  )
  v(0.2in)
  text(size: 12pt, style: "italic", fill: muted)[#description]
  v(0.5in)
  for (n, ch-title) in chapters {
    link(label("ch" + str(n)))[
      #set text(fill: ink)
      #grid(
        columns: (2.3em, 1fr, auto),
        column-gutter: 4pt,
        text(fill: accent, weight: "bold")[#n],
        eval(ch-title, mode: "markup"),
        page-of(n),
      )
    ]
    v(0.25em)
  }
}
