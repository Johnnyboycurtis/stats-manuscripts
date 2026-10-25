# Markdown Style Guide

This guide is for anyone, human or agent, who writes or edits chapters of *Spatial Data Analysis: A Linear Models and Matrix Approach*. Follow it exactly. When it conflicts with habits from other projects, this guide wins. When it conflicts with an explicit instruction from the author, the author wins.

The book is written in Markdown and compiled to PDF with **pandoc** and **pdflatex**. Most rules below exist so that one source file reads well on GitHub (which renders math with MathJax) and also compiles to clean LaTeX without errors. A rule that looks fussy usually prevents a broken build.

Related files:

- `OUTLINE.md`: the table of contents, with the running stories for each chapter. Section numbers, chapter titles, and scope come from there.
- `NOTATION.md`: the book's notation section, written for readers. It is the single source for symbols and says where each one is first introduced.
- `USECASES.md`: domain ideas for examples.
- `chapters/ch00_introduction.md`: the reference chapter. When in doubt, imitate it.

---

## 1. Build Toolchain

### 1.1 What to install

Nothing below is installed by default. On macOS:

```bash
brew install pandoc
brew install --cask mactex-no-gui
```

MacTeX is large (several GB) but includes every LaTeX package the book needs. A smaller alternative is BasicTeX plus the extra packages:

```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install dsfont collection-fontsrecommended
```

Use **pdflatex** as the engine. Do not switch to xelatex, lualatex, or tectonic without testing: pandoc's template loads `unicode-math` for those engines, and with the default math font `\boldsymbol` may stop producing bold Greek letters, which this book uses everywhere.

### 1.2 Preamble

The build uses a small LaTeX preamble, kept at `pandoc/preamble.tex`:

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{dsfont}
% Blackboard bold with digits, so \mathbb{1} (indicator) renders.
% \mathbb{R} also switches to the dsfont style, which is intended.
\renewcommand{\mathbb}[1]{\mathds{#1}}
```

Do not add `\newcommand` macros for use in chapters. A macro defined here does not exist on GitHub, so any chapter that uses it would render as raw LaTeX there.

### 1.3 Build commands

One chapter, for checking:

```bash
pandoc chapters/ch00_introduction.md -o build/ch00.pdf \
  --pdf-engine=pdflatex -H pandoc/preamble.tex \
  -V documentclass=book -V fontsize=11pt -V geometry:margin=1in \
  --top-level-division=chapter --highlight-style=tango \
  --resource-path=chapters
```

The whole book (chapter files sort in order because of the two-digit prefix):

```bash
pandoc chapters/ch*.md chapters/app*.md -o build/manuscript.pdf \
  --pdf-engine=pdflatex -H pandoc/preamble.tex \
  -V documentclass=book -V fontsize=11pt -V geometry:margin=1in \
  --top-level-division=chapter --toc --toc-depth=2 \
  --highlight-style=tango --resource-path=chapters
```

Add `--citeproc --bibliography=references.bib` once the book has citations (Section 7). Do not pass `-N`: section numbers are written by hand in the headings (Section 3.2), and pandoc's automatic numbering would duplicate them.

To find a LaTeX error quickly, convert to `.tex` first and inspect the output:

```bash
pandoc chapters/ch00_introduction.md -s -o build/ch00.tex -H pandoc/preamble.tex
```

---

## 2. Files and Layout

- Chapters live in `chapters/` and are named `chNN_short_title.md`, with a two-digit number: `ch00_introduction.md`, `ch07_sar_models.md`. Appendices are `appA_matrix_algebra.md`, and so on.
- Figures live in `figures/chNN/` and are produced by the code in the chapter (Section 6.4). Reusable tools live in `scripts/` (Section 10).
- Build output goes to `build/`. Never commit it.
- Write one paragraph per line. Do not hard-wrap prose at a fixed width. This keeps diffs readable and avoids accidental breaks inside inline math.
- Use UTF-8. Prose may contain ASCII, `§`, en dashes (`–`), and accented letters such as the é in Matérn. Do not type Unicode math symbols (≥, ×, −, →, σ, ∑) in prose. pdflatex cannot typeset them. Write them as math: `$\ge$`, `$\times$`, `$-$`, `$\sigma$`.

---

## 3. Chapter Structure

Every chapter follows the same skeleton. The reference implementations are `chapters/ch00_introduction.md` and `chapters/ch01_spatial_dependence.md`. Chapter 0 deviates in one way: its exercises are in a numbered section `## 0.7 Warm-Up Exercises`, a single list organized by tool, instead of a final `## Exercises` section.

```markdown
# Chapter N: Title From OUTLINE.md

Two to four opening paragraphs: the problem, why it matters, and what
the reader will be able to do by the end.

> **Chapter at a Glance**
>
> - **Topic (§N.1):** One sentence.
> - **Topic (§N.2):** One sentence.

---

## N.1 First Section Title

Opening paragraph.

### N.1.1 First Subsection Title

...

---

## N.k Walkthrough and Lab N

---

## Exercises

---

## What Comes Next
```

### 3.1 Headings

| Level | Use | Example |
| :--- | :--- | :--- |
| `#` | Chapter title, once per file | `# Chapter 4: Spatial Prediction: Kriging as Linear Model BLUP` |
| `##` | Numbered section | `## 4.2 Simple Kriging` |
| `###` | Numbered subsection | `### 4.2.1 Deriving the Kriging Equations` |
| `####` | Unnumbered label inside a subsection, used sparingly | `#### Why the Cholesky step works` |

- Never skip a level. A `###` must sit inside a `##`. With `--top-level-division=chapter`, `#` becomes a LaTeX chapter, `##` a section, and `###` a subsection, so a skipped level produces wrong sizes and a broken table of contents.
- Section numbers and titles must match `OUTLINE.md`. If a draft needs a section the outline lacks, add it and tell the author so the outline can be updated.
- Use title case in headings. Do not end headings with punctuation, except a question mark when the heading is a question.
- Put a line containing only `---` before each `##` section, as in the reference chapter.

### 3.2 The "Chapter at a Glance" box

Use a plain blockquote with a bold first line. Do not use GitHub alert syntax (`> [!NOTE]`). Pandoc does not understand it and prints `[!NOTE]` literally in the PDF.

This is the only boxed element in a chapter. Do not add other callout boxes, sidebars, icons, or emoji. Notes, warnings, and asides belong in the prose.

The box is a preview of the chapter, so it may name symbols and objects (for example $\mathbf{W}$ or $\boldsymbol{\lambda}$) that the chapter defines later. The first-use rule in Section 5.1 applies from the first section onward and does not apply to the box.

### 3.3 The rhythm inside a chapter

Each major topic follows the sequence set in Section 0.5.2 of the book:

1. **A linear-model bridge.** Connect the spatial idea to a regression or GLM idea the reader knows, and translate any spatial jargon into those terms.
2. **A full derivation.** Show every algebraic step. Never write "it is easy to show" or "it is trivial."
3. **A hand-sized example.** Work 3 to 5 locations with every matrix printed in full, so the reader can check each entry with a calculator.
4. **Code and a known-truth simulation.** Translate the equations into a few lines of NumPy, then simulate data with known parameters and show whether the method recovers them.
5. **Exercises.** Analytical problems first, then computational and simulation problems.

The rhythm is a sequence, not a template of labeled boxes. Do not add headings such as "Linear Model Bridge" or "Hand Example." Let the prose carry the structure.

### 3.4 Stories and placeholders

A story is a concrete setting that makes an abstract idea memorable: rents on one block, monitors along a highway, cells on a slide. Use one wherever it helps the reader understand a concept, and skip it where the mathematics is already concrete. Each chapter's planned stories are listed under its heading in `OUTLINE.md`; follow them unless a better one fits.

When a section needs a story but no good one is available yet, leave a placeholder instead of forcing one. Write it as its own paragraph, starting with the marker and saying what the story should do:

```markdown
**[STORY NEEDED]** A setting where two nearby measurements share an
unobserved cause, to motivate the spatial random effect before the algebra.
```

Keep the marker spelled exactly as shown so it can be found with `grep -rn "STORY NEEDED" chapters/`. Placeholders must be resolved or removed before a chapter is final.

### 3.5 Optional sections

Material that a reader can skip on a first pass without losing the thread, such as a full proof whose result has already been explained another way, is marked optional. Add `(Optional)` at the end of the heading and open the section with one italic sentence saying what it contains and that it can be skipped:

```markdown
### 1.3.6 Proof of Aitken's Theorem (Optional)

*This subsection proves the result of Section 1.3.5 directly, without
transforming the data. It can be skipped on a first reading.*
```

Later sections must not depend on optional ones. If a result proved in an optional section is needed later, state the result in the main text.

---

## 4. Voice and Prose

The book is written for a first-year master's student who passed mathematical statistics but did not love it. The goal is that the reader understands what is going on under the hood and can use the methods correctly.

- **Translate, do not hide.** Introduce every spatial term with its linear-model meaning: kriging is best linear unbiased prediction; the nugget is the variance of independent noise. Keep the spatial term, since readers need it for the literature.
- **Explain why before how.** Each new tool appears because a problem needs it. State the problem first.
- **Write plainly.** Short sentences, active voice, concrete nouns. Prefer "the estimate is too small" to "the estimate exhibits downward bias characteristics."
- **Avoid these:** "trivially," "it is easy to see," "clearly," "obviously," "simply," "note that," "it is worth noting," "crucially," "powerful," "elegant," "robust" (except as a technical term), and rhetorical questions used as filler.
- **Avoid em dashes** and "not X, but Y" constructions. Use a comma, a colon, parentheses, or a new sentence.
- **Address the reader as "you"** sparingly, mainly in instructions and reading advice. Use "we" for derivations done together.
- **Be honest about limits.** When a result is taken from the literature instead of derived (for example, validity of a covariance family), say so in one sentence.
- **Cross-references:** write "Section 4.2" and "Chapter 5" in running prose. Use "§4.2" only inside parentheses, tables, and the Chapter at a Glance box. Appendices are "Appendix A."

---

## 5. Mathematics

### 5.1 Notation

All symbols come from `NOTATION.md`. Two rules apply:

- **First use.** A symbol may not appear in a chapter before the section listed as its first introduction in `NOTATION.md`, except in the Chapter at a Glance box. When a symbol is introduced, define it in words in the same sentence or the next one. A forward reference to a later chapter is written in words ("the autoregressive parameter of Chapter 7"), not with the symbol.
- **New symbols.** If a draft needs a symbol that is not in `NOTATION.md`, check for clashes, add it to the right table (or to "Symbols Used in One Chapter Only"), and tell the author.

The essentials:

- The model is $Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})$ with $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$. Write $\varepsilon$ (`\varepsilon`), not $\epsilon$.
- $\sigma^2$ is the spatial variance scale (partial sill), $\tau^2$ the nugget, $\phi$ the range parameter, and $\boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n$.
- Uppercase letters are random quantities, lowercase their observed values: $\mathbf{Y}$ and $\mathbf{y}$.
- The domain is $\mathcal{D}$ (`\mathcal{D}`). The bold $\mathbf{D}$ is the degree matrix.
- Use the book's names, not other texts': $Y(\mathbf{s})$, not $Z(\mathbf{s})$; $u(\mathbf{s})$, not $\delta(\mathbf{s})$ or $W(\mathbf{s})$. `NOTATION.md` maps these to Schabenberger and Gotway.
- Transpose is a prime: $\mathbf{X}'$, not $\mathbf{X}^T$ or $\mathbf{X}^\top$.

### 5.2 Typesetting rules

| Write | Not | Why |
| :--- | :--- | :--- |
| `\mathbf{X}`, `\mathbf{y}` | `\bm{X}`, `\vec{y}` | Latin vectors and matrices; `\bm` is not in MathJax by default |
| `\boldsymbol{\beta}`, `\boldsymbol{\Sigma}` | `\mathbf{\beta}` | `\mathbf` does not bold Greek letters |
| `\operatorname{E}`, `\operatorname{Var}`, `\operatorname{Cov}`, `\operatorname{tr}` | `E`, `Var`, `\text{Var}` | Upright operator names with correct spacing |
| `\lvert x \rvert`, `\lVert \mathbf{h} \rVert` | `|x|`, `\|h\|` | A bare `|` breaks Markdown tables |
| `\text{OLS}` in subscripts | `_{OLS}` | Upright text in math |
| `\mathbb{1}` for indicators, `\mathbb{R}` for the reals | `\mathbbm{1}`, `\mathds{1}`, `I(\cdot)` | Works in MathJax; the preamble maps it for pdflatex |
| `\dots` | `...` | Proper ellipsis |
| `\begin{aligned}` inside `$$` | `\begin{align}` | `align` cannot sit inside display math in LaTeX |
| `\tfrac{1}{2}` inline | `\frac{1}{2}` inline | Keeps line spacing even |

### 5.3 Inline and display math

- Inline math uses single dollars: `$\hat{\boldsymbol{\beta}}$`. Pandoc requires no space just inside the dollars. Write `$x$`, never `$ x $`.
- A literal dollar sign in prose must be escaped: `\$3,000 per month`.
- Display math uses `$$` on its own line before and after, with **no blank lines inside**. A blank line ends the math block in pandoc and breaks the build.

```markdown
The variance of the sample mean is

$$
\operatorname{Var}(\bar{Y}) = \frac{1}{n^2}\,\mathbf{1}'\boldsymbol{\Sigma}\mathbf{1}
$$

where ...
```

- Put a blank line before and after every display block. Punctuation after a display equation is optional, but be consistent within a chapter. The reference chapter omits it.
- Do not number equations. Refer to them by what they say ("the kriging equations in Section 4.2") or by section.
- Matrices use `\begin{bmatrix} ... \end{bmatrix}`.

### 5.4 Derivations

- Show each step on its own line inside `\begin{aligned}`, or in prose between displays. Do not compress two algebraic moves into one line.
- After a result, say in one sentence what it means: "With independent data this reduces to $\sigma^2/n$."
- When an identity is used, name it ("by Tool 1," "by the block inversion formula in Appendix A.1").

---

## 6. Code

### 6.1 Allowed libraries

Only `numpy`, `scipy`, `sympy`, `matplotlib`, and the Python standard library. No GIS or spatial packages (`geopandas`, `shapely`, `sf`, `pysal`, `spatstat`, `scikit-gstat`, `pykrige`) and no `pandas` or `scikit-learn` in chapter code. SciPy submodules such as `scipy.spatial` (`cKDTree`, `cdist`) are allowed because they are part of SciPy; use them when scale requires it, as in Chapter 13, and say why.

### 6.2 Style

- Fence code with ` ```python `. Use ` ```bash ` for shell commands and ` ```latex ` for LaTeX.
- Keep every code line at **79 characters or fewer**. Highlighted code does not wrap in the PDF, so a long line runs off the page.
- Import as `import numpy as np`, `from scipy import stats` (or the specific submodule), `import sympy as sp`, `import matplotlib.pyplot as plt`.
- Seed every random computation with `rng = np.random.default_rng(<seed>)` and pass `rng` explicitly. Never use `np.random.seed` or the legacy global functions.
- Prefer vectorized NumPy over Python loops over locations. Loops over simulation replicates are fine.
- Solve linear systems with `np.linalg.solve` or `scipy.linalg.cho_solve`. Do not form an explicit inverse unless the inverse itself is the point of the example. `np.linalg.inv` is acceptable for a small matrix whose inverse is wanted, such as $(\mathbf{X}'\mathbf{X})^{-1}$ or the covariance of an estimate.
- Each chapter's code blocks must run top to bottom as one script. Later blocks may use names defined in earlier blocks; never rely on a name from another chapter.

### 6.3 Numbers in the text

Every number reported in prose or tables that comes from computation must come from running the code shown in the chapter, with the seeds shown. Do not invent, round loosely, or carry over numbers from a different run. After any edit to code, re-run it and update the numbers.

Check a chapter with the saved scripts (Section 10):

```bash
python3 scripts/run_chapter_code.py chapters/ch03_empirical_variography.md
python3 scripts/run_chapter_code.py --all --save-output build/code_output
```

The runner extracts every python block, runs them top to bottom from the repository root (so figures are written to `figures/chNN/`), and reports errors, run time, and code lines over 79 characters. With `--save-output` it keeps each chapter's printed output, which is what the numbers in the text should be compared against.

### 6.4 Figures

- Figures are produced by code that the reader can run: either a plotting block in the chapter's lab, or a script `scripts/chNN_figures.py` when the plotting code would distract from the chapter. Either way, the code is run from the top folder of the repository and saves to `figures/chNN/fig_N_k.png` (PNG displays on GitHub; switch to PDF for the final book if sharper output is needed).
- Use `matplotlib.use("Agg")` before importing `pyplot`, so the code runs without a display, and call `fig.tight_layout()` or pass `bbox_inches="tight"` so labels are not cut off.
- Link figures with a path relative to the chapter file, so they display on GitHub. The build commands pass `--resource-path=chapters` so pandoc resolves the same paths:

```markdown
![Figure 2.1. A simulated rent district in three layers. The neighborhood effect creates patches of high and low rent; the nugget adds speckle.](../figures/ch02/fig_2_1_rent_layers.png)
```

- Number figures within the chapter (Figure 2.1, Figure 2.2), start the caption with the number, and say what the reader should notice.
- Look at every figure before describing it in the text. Describe only what the figure shows.
- Use colorblind-safe colormaps such as `viridis` and a single font size. Every axis has a label, with units where there are units.

---

## 7. Examples, Data, and Citations

- **Domains.** Draw examples from urban economics (rents, housing), public health (disease rates, overdose deaths, exposure), cell biology (immune cells on tissue slides, spatial transcriptomics), economics (tax competition, spillovers), public safety (crashes), and sports (shot charts). Avoid soil, mining, forestry, and meteorology as the main story. See `USECASES.md`.
- **Recurring cast.** Reuse the same story for a data type across chapters: rents for continuous data, county health rates for areal data (tax competition for SAR models), and immune cells and disease cases for point patterns.
- **Map examples to the right data type.** A quantity that exists everywhere and is measured at points is geostatistical. A value attached to each region of a fixed map is areal. A list of event locations, possibly with labels, is a point pattern. For example, the probability of scoring by court location comes from shots with made or missed labels, so it is a labeled point pattern, not a geostatistical field.
- **Simulated data is labeled as simulated.** The domain supplies the story and the parameters supply the truth. Never present simulated numbers as real findings about a real place.
- **Citations.** Use pandoc citations, `[@diggle1990]` or `Diggle [-@diggle1990]` for "Diggle (1990)", with entries in `references.bib`. Cite only sources whose DOI you have checked. Never cite from memory or from an unverified list, including `USECASES.md`. Rule: wherever the text credits a named author or paper, give a `[@key]` and add the entry to `references.bib` with a checked DOI; a sentence that names no author needs no key. Until `references.bib` exists, build without `--citeproc`, because every key would print as a missing reference.

---

## 8. Markdown Details

- Bullets use `-`. Numbered lists use `1.`, `2.`, and so on.
- Bold (`**...**`) is for lead-ins in lists and for a term at its definition. Italic (`*...*`) is for book titles and light emphasis. Do not use bold for emphasis in running prose.
- Tables use pipe syntax with a header row and a `| :--- |` separator. Keep tables to five columns or fewer and cells short. In the PDF, pandoc sets column widths from the relative number of dashes in the separator row when a row is long, so lengthen a column's dashes to give it more room.
- Never put a bare `|` inside a table cell, even in math. Use `\lvert` and `\rvert`.
- Do not use raw HTML, footnotes, emoji, or images of equations.
- Do not use ASCII-art diagrams. Draw a figure instead.
- Escape Markdown-sensitive characters in prose when they are literal: `\$`, `\*`, `\_`.

---

## 9. Checklist Before Handing Off a Chapter

- [ ] Headings follow `#`, `##`, `###` without skipping, and match `OUTLINE.md`.
- [ ] The Chapter at a Glance box is a plain blockquote, and there are no other boxes.
- [ ] All symbols match `NOTATION.md`, none appears before its first-introduced section, and any new symbol has been added there; no $Z(\mathbf{s})$, $\epsilon$, $\mathbf{X}^T$, or bare `|`.
- [ ] Every abstract concept has a story or a `**[STORY NEEDED]**` placeholder, and the chapter's planned stories from `OUTLINE.md` are used.
- [ ] Skippable material is marked `(Optional)` and nothing later depends on it.
- [ ] No Unicode math symbols in prose; no `$ x $` with inner spaces; no blank lines inside `$$` blocks.
- [ ] `scripts/lint_chapter.py` reports the chapter clean.
- [ ] Every code line is 79 characters or fewer, and all code runs top to bottom (`scripts/run_chapter_code.py`).
- [ ] Every computed number in the text matches the output of the code shown.
- [ ] Examples come from the recommended domains and are labeled as simulated.
- [ ] No banned phrases ("trivially," "clearly," "it is easy to see").
- [ ] The chapter compiles alone with the single-chapter command in Section 1.3, once pandoc and LaTeX are installed.

---

## 10. Scripts

Reusable tools live in `scripts/`. If you write a script that you run more than once while drafting, save it there with a docstring that says what it does, and add it to this list. Do not leave working tools in a temporary folder.

| Script | What it does |
| :--- | :--- |
| `scripts/run_chapter_code.py` | Runs all python blocks of a chapter (or `--all`) from the repository root; reports errors, timing, and over-long code lines; `--save-output DIR` keeps the printed output |
| `scripts/lint_chapter.py` | Checks the mechanical rules in this guide: heading levels, banned phrases, em dashes, non-ASCII math, math spacing and blank lines in `$$` blocks, bare `\|` in tables, GitHub alert syntax, style slips (`\epsilon`, `X^T`), long code lines; counts `**[STORY NEEDED]**` placeholders |
| `scripts/check_consistency.py` | Cross-chapter checks: "Section N.N.N" references resolve, figure links work, outline section numbers and titles match the chapters, NOTATION first-use sections exist |
| `scripts/build_outline_page.py` | Assembles `build/outline.html`, a page that renders `OUTLINE.md` with math (templates in `scripts/outline_page/`), for publishing as an artifact |

Run both checks before handing off a chapter:

```bash
python3 scripts/lint_chapter.py chapters/chNN_title.md
python3 scripts/run_chapter_code.py chapters/chNN_title.md
```

The lint script cannot judge voice, correctness, or whether a symbol is used before its first introduction in `NOTATION.md`; those still need a reader.
