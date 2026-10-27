# Spatial Data Analysis: A Linear Models and Matrix Approach

This repository holds the manuscript of a textbook on spatial statistics for first-year master's students and strong senior undergraduates in statistics. The book teaches geostatistics, areal (lattice) models, and point patterns as extensions of linear models and generalized linear models, using matrix algebra, full derivations, small worked examples, and known-truth simulation in Python.

This README is written so that a new contributor, including an AI agent, can pick up the project without the conversation history behind it. Read it in full before editing anything.

---

## Start Here

Read these files in this order:

1. **This README**: purpose, current status, decisions already made, and open work.
2. **`OUTLINE.md`**: the table of contents and the running stories for each chapter. It is the source of truth for chapter and section numbers, titles, and scope.
3. **`NOTATION.md`**: the book's notation section, written for readers. It is the single source for symbols, and its "First introduced" column says where each symbol may first appear.
4. **`MARKDOWNSTYLE.md`**: the writing, math, code, and build rules. Every chapter must follow it.
5. **`chapters/ch00_introduction.md` and `chapters/ch01_spatial_dependence.md`**: the reference chapters. They show the voice, structure, and level of detail every later chapter should match.
6. **`USECASES.md`**: brainstormed example domains and references. Useful for ideas, but not verified (see "Known Issues").

---

## What the Book Is

**Audience.** Readers who have taken multivariate calculus, linear algebra, applied regression in matrix form, and undergraduate mathematical statistics. Picture a student who passed mathematical statistics and found it intimidating. The book should make that reader say, "Now I understand what is going on under the hood, and I know how to use these tools correctly."

**Models in style.** Dobson and Barnett (*An Introduction to Generalized Linear Models*) and Faraway (*Linear Models with R*) for the theory-then-application rhythm and complete derivations; Rizzo (*Statistical Computing with R*) for simulation as a teaching tool. Notation is kept compatible with Schabenberger and Gotway (*Statistical Methods for Spatial Data Analysis*, 2005) where that does not cost clarity.

**Core ideas.**

- **Anti-gatekeeping.** Spatial jargon is kept, because readers need it for the literature, but every term is translated into linear-model language: kriging is best linear unbiased prediction, the nugget is independent noise variance, a CAR model is a precision matrix.
- **One model equation.** Everything is a version of $Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})$: a regression trend, a spatial random effect, and independent noise. Each part of the book changes one term.
- **Full derivations and hand-sized examples.** No step is skipped. Each major result is worked on 3 to 5 locations with every matrix printed.
- **Known-truth simulation instead of real data sets.** Examples are simulated with stated parameters, so the reader can check whether a method recovers the truth. Domain stories (rents, disease rates, immune cells) give the simulations meaning, and simulated numbers are always labeled as simulated.
- **No measure theory or functional analysis.** Random fields are handled through finite-dimensional distributions. Where a result needs deeper theory (for example, validity of a covariance family), the book says so and cites it.
- **No spatial software.** Code uses only NumPy, SciPy, SymPy, and matplotlib. Readers build each method themselves.

**Structure.** Fourteen chapters plus appendices:

| Part | Chapters | Content |
| :--- | :--- | :--- |
| Getting Started | 0 | Orientation, three motivating simulations, the five math tools the book relies on |
| I. Matrix Foundations | 1–2 | Why OLS fails under spatial dependence; the spatial linear model, covariance functions, simulation |
| II. Geostatistics | 3–5 | Variograms, kriging as BLUP, likelihood and REML, model selection |
| III. Areal Data | 6–9 | Spatial weights and Moran's $I$; SAR models; CAR models and sparse precision; spatial GLMMs for counts |
| IV. Point Patterns | 10–12 | Exploration; point processes as Poisson and logistic GLMs; cluster and Gibbs models |
| V. Large Data | 13 | Low-rank, tapering, Vecchia, and GMRF approximations |
| Appendices | A–D | Matrix algebra, Python patterns, jargon and notation concordance, further directions |

---

## Current Status

As of 2026-10-26. Chapters 0 to 13 are drafted; the appendices are not. The author has read Chapters 0, 1, 2, 3, and 4 and approved them; Chapters 5 to 13 are unreviewed. The drafts lean long on derivations so they can be trimmed in review.

| Chapter | File | Words (including code and equations) | Figures |
| :--- | :--- | :--- | :--- |
| 0 | `chapters/ch00_introduction.md` | 5,743 | 0 |
| 1 | `chapters/ch01_spatial_dependence.md` | 9,484 | 0 |
| 2 | `chapters/ch02_spatial_linear_model.md` | 8,020 | 2 |
| 3 | `chapters/ch03_empirical_variography.md` | 8,180 | 3 |
| 4 | `chapters/ch04_kriging_blup.md` | 10,823 | 2 |
| 5 | `chapters/ch05_likelihood_reml.md` | 10,719 | 2 |
| 6 | `chapters/ch06_areal_units_moran.md` | 15,613 | 4 |
| 7 | `chapters/ch07_sar_models.md` | 12,771 | 3 |
| 8 | `chapters/ch08_car_models.md` | 11,992 | 4 |
| 9 | `chapters/ch09_spatial_glmms.md` | 14,242 | 3 |
| 10 | `chapters/ch10_exploring_point_patterns.md` | 16,010 | 5 |
| 11 | `chapters/ch11_point_processes_glms.md` | 15,903 | 5 |
| 12 | `chapters/ch12_clustering_inhibition.md` | 18,799 | 7 |
| 13 | `chapters/ch13_large_n_methods.md` | 15,081 | 4 |

| Item | State |
| :--- | :--- |
| `OUTLINE.md` | Complete for 14 chapters and four appendices, with running stories per chapter. Updated after each draft. |
| `NOTATION.md` | Written and kept in sync with the chapters, including first-use sections and the list of symbols with more than one use. |
| Shared data sets | Chapters 3 to 5 use `rent_district(n=250, seed=2030)` (truth $\sigma^2 = 4$, $\tau^2 = 1$, $\phi = 0.15$, trend $18 + 6\,s_{\text{east}}$). Chapters 6 to 9 use `county_map(m=12)` (144 counties, rook adjacency, populations with median about 5,000, a standardized deprivation covariate `x`). Chapters 10 to 12 use the unit-square setting (`road_distance`, `pop_density`, `excess_risk`) and a case-control recipe whose plant signal is weak at the planned size (see `OPEN_QUESTIONS.md`). Chapter 13 uses uniform random locations with seed 2110. |
| Checks | `scripts/lint_chapter.py --all` is clean, `scripts/run_chapter_code.py --all` runs every chapter's code without errors, and `scripts/check_consistency.py` reports no problems. |
| Consistency review | Done on 2026-10-25. See `CONSISTENCY_REVIEW.md`; the build-breaking items and the cross-chapter promises were fixed; the rest are listed there. |
| `OPEN_QUESTIONS.md` | Decisions and checks waiting for the author, by chapter. |
| `APPENDIX_REQUIREMENTS.md` | What the chapters expect from Appendices A to D, for when they are drafted. |
| `MARKDOWNSTYLE.md` | Written. Its build rules have not been tested yet because pandoc and LaTeX are not installed. |
| PDF build | Working, from the repository root: `python -m tools.manuscript build spatial-data-analysis` (pandoc to Typst, then Typst compiles the PDF). The template, title page, and contents pages live in `tools/manuscript/`; the title, parts, and chapter descriptions in `book.toml`. |
| Version control | This folder is not a git repository yet. |

---

## Repository Layout

```
Spatial-Data-Analysis-Manuscript/
├── README.md            This file
├── OUTLINE.md           Table of contents and running stories (source of truth)
├── book.toml            Title page text, grouping of chapters into parts, one-line chapter descriptions for the contents pages
├── NOTATION.md          Reader-facing notation section (single source for symbols)
├── OPEN_QUESTIONS.md    Decisions and checks waiting for the author
├── TODO.md              Remaining work, author review items first
├── CONSISTENCY_REVIEW.md  Cross-chapter consistency review (2026-10-25) with a fix list
├── APPENDIX_REQUIREMENTS.md  What the chapters expect from Appendices A to D
├── reviews/             Raw reviewer reports behind CONSISTENCY_REVIEW.md
├── MARKDOWNSTYLE.md     Writing, math, code, and build rules
├── USECASES.md          Example-domain ideas and unverified references
├── chapters/            One Markdown file per chapter: chNN_short_title.md
├── figures/             Generated figures, one folder per chapter (figures/chNN/)
├── scripts/             Reusable tools: chapter code runner, lint, outline page builder
├── build/               Generated output (code output, outline page, PDFs); not committed
├── .gitignore           Ignores build/, __pycache__/, .DS_Store
└── files/               Empty. Purpose not yet assigned; ask the author before using it
```

Files that `MARKDOWNSTYLE.md` expects but that do not exist yet: `references.bib` (citations). Its pdflatex rules (for example the `dsfont` preamble for `\mathbb{1}`) predate the Typst build and have not been rechecked against it.

---

## Decisions Already Made

These were settled with the author. Do not reopen them without being asked.

**Scope and structure**

- The book was re-planned from an earlier draft that lives in a different repository (`semantic-search-models/manuscript/`). That draft follows an older outline, uses different notation ($Z$, $\delta$, $D$), and is heavier on theory. **Do not copy text, structure, or notation from it.** This repository is a new manuscript.
- Chapter 0 exists as an onboarding chapter. Chapters 1–13 keep their numbers.
- SAR and CAR models are separate chapters (7 and 8), because together they formed one overloaded chapter with two different modeling ideas.
- Part IV has three chapters: exploration (10), point processes as GLMs including the case-control logistic bridge and scan statistics (11), and cluster and Gibbs models (12).
- Every part completes an analysis cycle: explore, fit, infer, predict or interpret, check.

**Notation** (full tables in `NOTATION.md`)

- $Y(\mathbf{s})$ for the process (not $Z$), $u(\mathbf{s})$ for the spatial random effect (not $\delta$ or $W$), $\varepsilon(\mathbf{s})$ for the nugget, $\mathcal{D}$ for the domain.
- $\sigma^2$ is always the spatial variance scale: the partial sill in geostatistics and the innovation variance in SAR and CAR models, where precisions are written as $\sigma^{-2}(\cdots)$.
- Kriging weights are $\boldsymbol{\lambda}$; the kriging target is the noise-free value $\tilde{Y}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + u(\mathbf{s}_0)$.
- Spatial weights are $\mathbf{W}$ with entries $w_{ij}$; IRLS weights are $\mathbf{V}$.
- Symbols that would collide were renamed deliberately (for example $\delta = \tau^2/\sigma^2$, $\boldsymbol{\Pi}$ for the residual precision, $\mathbf{T}$ for the REML contrast matrix, $\zeta$ for the Strauss interaction, $r_0$ for interaction radii, $\mathbf{v}_k$ for eigenvectors). Check `NOTATION.md` before introducing any new symbol, and add new symbols there.
- The classical regression error variance is written $\tau^2$, not $\sigma^2$, because $\sigma^2$ is always the spatial variance.
- No symbol may appear before the section listed as its first introduction in `NOTATION.md`.

**Content and style**

- Examples come from urban economics, public health, cell biology, economics, public safety, and sports, not from soil, mining, forestry, or meteorology. A small recurring cast is reused across chapters: apartment rents for continuous data, county health rates for areal data (tax competition for SAR models), and immune cells and disease cases for point patterns.
- Prose is plain and direct, with no callout boxes other than "Chapter at a Glance," no "trivially" or "clearly," and no em dashes.
- Stories are used wherever they help understanding, following the running stories listed under each chapter in `OUTLINE.md`. Where a story is needed but not yet found, a paragraph starting `**[STORY NEEDED]**` marks the spot.
- Skippable material, such as a second proof of a result already explained, is marked `(Optional)` in its heading, as in other textbooks. Nothing later depends on it.
- New tools are introduced with a small worked example before they are used, so the difficulty rises gradually from Chapter 0.
- Every computed number in a chapter must come from running that chapter's code with the seeds shown.

**Build**

- pandoc converts the combined Markdown to Typst and Typst typesets it (`tools/manuscript/`, in the repository root). Typst's own math engine is used, so LaTeX preamble rules in `MARKDOWNSTYLE.md` Section 1 do not apply. `MANUSCRIPT.md` is generated; do not edit it. Part groupings and contents descriptions come from `book.toml`.

---

## How to Work in This Repository

**Scripts.** Before handing off a chapter, run both checks from anywhere:

```bash
python3 scripts/lint_chapter.py --all
python3 scripts/run_chapter_code.py --all --save-output build/code_output
```

The first checks the mechanical style rules; the second runs every chapter's code and saves the output so the numbers in the text can be compared with it. The code runner limits math libraries to one thread (many small matrix operations are thousands of times slower with a multithreaded BLAS on a busy machine). `scripts/build_outline_page.py` assembles an HTML page of `OUTLINE.md`. Any script you use more than once belongs in `scripts/` with a docstring, and in the table in `MARKDOWNSTYLE.md`, Section 10.

- **Before drafting a chapter**, read its section of `OUTLINE.md`, the chapters it depends on, and `MARKDOWNSTYLE.md`. Match `chapters/ch00_introduction.md` in voice and level.
- **Keep the outline in sync.** If a draft needs to add, remove, or rename a section, update `OUTLINE.md` in the same change and tell the author what changed.
- **Verify before reporting.** Run every code block in a chapter (the command is in `MARKDOWNSTYLE.md`, Section 6.3) and make sure the numbers in the text match. If you could not run something, say so.
- **Check derivations.** The outline was reviewed for mathematical errors, and several were caught and fixed (for example, the leave-one-out kriging shortcut uses $\Pi_{ii}$, not $(\boldsymbol{\Sigma}^{-1})_{ii}$, when covariates are present). Re-derive results rather than trusting memory.
- **Cite carefully.** Only cite sources whose DOI you have checked, through `references.bib`.
- **Ask the author** before changing scope, chapter order, notation, or the toolchain.

---

## Known Issues and Open Work

- **Appendices A to D are not written.** `OUTLINE.md` lists their sections and `APPENDIX_REQUIREMENTS.md` lists what the chapters expect from them. Appendix A is planned to open with a review of the five tools of Chapter 0 (A.0) and to end with matrix facts (A.6).
- **Build.** Works through `tools/manuscript` (see Current Status). Update `MARKDOWNSTYLE.md` Section 1 to describe the Typst route and drop the pdflatex-only rules once they are checked.
- **Citations.** `references.bib` does not exist. About 13 keys are used (`gneiting2013`, `zhang2004`, six in Chapter 9, five in Chapter 13) and many more authors are named without keys. Each entry needs a checked DOI. See `OPEN_QUESTIONS.md` and `MARKDOWNSTYLE.md`, Section 7. Until then, build without `--citeproc`.
- **Chapter reviews pending.** The author will review the drafts; trim candidates are listed per chapter in `OPEN_QUESTIONS.md`.
- **`USECASES.md` references are unverified** (a duplicate Blonigen entry with different coauthors, `google.com/search` links, one "suspicious link removed," possible name-order errors). Verify before citing anything from it.
- **The shared case-control signal is weak** (Chapters 10 and 11); the choice to strengthen it or keep it as a power lesson is open.
