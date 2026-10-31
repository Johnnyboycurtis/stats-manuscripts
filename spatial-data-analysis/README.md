# Spatial Data Analysis: A Linear Models and Matrix Approach

**Author:** Jonathan Navarrete  
**Status:** Working Draft (Active Development)  
**Tagline:** *Spatial statistics, translated into the linear models you already know.*

---

## Overview

This directory contains the manuscript, computation, and build pipeline for *Spatial Data Analysis: A Linear Models and Matrix Approach*, an advanced graduate/upper-undergraduate textbook on spatial statistics. 

The text demystifies geostatistics, areal (lattice) models, and point patterns by grounding them directly in the linear models and generalized linear models students already know—emphasizing matrix formulations, explicit derivations, transparent hand-sized numerical examples, and known-truth simulation using core Python (`numpy`, `scipy`, `sympy`, and `matplotlib`) without reliance on black-box GIS packages.

---

## Directory Organization

```
spatial-data-analysis/
├── README.md               # This project-level overview
├── MANUSCRIPT.md           # Assembled single-file manuscript (generated, do not edit directly)
├── MANUSCRIPT.pdf          # Typeset book PDF compiled via Typst (generated)
├── current/                # Active manuscript codebase under continuous development and publication
│   ├── README.md           # Detailed contributor onboarding, roadmap, and status
│   ├── OUTLINE.md          # Canonical table of contents and chapter running stories
│   ├── NOTATION.md         # Master notation index and symbol introduction rules
│   ├── MARKDOWNSTYLE.md    # Writing, math typesetting, and code conventions
│   ├── book.toml           # Manuscript metadata and structure for the build pipeline
│   ├── chapters/           # Source markdown chapters (ch00 - ch13, appendices)
│   ├── figures/            # Generated manuscript figures organized by chapter
│   ├── scripts/            # Linters, chapter code runner, and outline utilities
│   ├── reviews/            # Consistency and editorial reviewer reports
│   └── build/              # Local build artifacts and execution logs (git-ignored)
└── past/                   # Archival directory containing past iterations (for historical reference only)
    ├── manuscript_valpha/  # Version Alpha draft
    └── manuscript_vbeta/   # Version Beta draft
```

### Iteration Architecture: `current/` vs. `past/`

- **[`current/`](current/): Active & Canonical**  
  `current/` contains the latest iteration of the textbook and is the **only version under active development and slated for publication**. All ongoing writing, exercises, code updates, reviews, and builds take place here.
- **[`past/`](past/): Archival Reference Only**  
  `past/` preserves earlier iterations of this project (`manuscript_valpha` and `manuscript_vbeta`). These drafts explored alternative outlines, heavier measure-theoretic formalisms, and older notation schemes ($Z$, $\delta$, $D$). They are kept strictly for historical provenance and should not be edited or used for ongoing work.

---

## Book Structure (`current/`)

The textbook is organized into five core parts plus appendices, unified by a single core decomposition: $Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})$ (trend + spatial random effect + independent noise):

| Part | Chapters | Title & Description |
| :--- | :--- | :--- |
| **Getting Started** | 0 | **The Big Picture and the Math You Already Know**: Orientation, three motivating simulations, and a review of the five essential matrix tools. |
| **Part I: Matrix Foundations** | 1–2 | **Matrix Foundations and Spatial Covariance**: Why OLS fails under spatial dependence, effective sample size, spatial linear models, and covariance kernels. |
| **Part II: Geostatistics** | 3–5 | **Geostatistical Models and Generalized Least Squares**: Empirical variography, kriging as best linear unbiased prediction (BLUP), maximum likelihood, and REML. |
| **Part III: Areal Data** | 6–9 | **Discrete Spatial Models (Lattice and Areal Data)**: Spatial weights matrices, Moran's $I$, SAR models, CAR models with sparse precision, and spatial GLMMs (BYM2). |
| **Part IV: Point Patterns** | 10–12 | **Spatial Point Patterns**: Exploring spatial intensity and complete spatial randomness, point processes as GLMs (Berman–Turner device), and cluster/Gibbs models. |
| **Part V: Scalability** | 13 | **Matrix Scalability for Large Spatial Systems**: Low-rank approximations, covariance tapering, Vecchia approximations, and Gaussian Markov Random Fields. |
| **Appendices** | A–D | Matrix algebra reference, Python scientific idioms, Jargon & Notation Rosetta Stone (`appC`), and further directions. |

---

## Working with the Manuscript

### Essential Guides

Before contributing to or editing any chapter in `current/`, consult:
1. **[`current/README.md`](current/README.md)**: Project status, chapter word counts, and editorial decisions.
2. **[`current/OUTLINE.md`](current/OUTLINE.md)**: Source of truth for chapter titles, section breakdowns, and running narratives.
3. **[`current/NOTATION.md`](current/NOTATION.md)**: Definitive symbol registry and first-introduction rules.
4. **[`current/MARKDOWNSTYLE.md`](current/MARKDOWNSTYLE.md)**: Style manual for mathematical typography, tone, and code conventions.

### Running Verification Scripts

From the `spatial-data-analysis/current/` directory:

```bash
# Check mechanical and style consistency rules across all chapters
python3 scripts/lint_chapter.py --all

# Execute all code blocks to verify execution and numerical reproducibility
python3 scripts/run_chapter_code.py --all --save-output build/code_output

# Run cross-chapter mathematical and notation consistency checks
python3 scripts/check_consistency.py
```

### Compiling the Manuscript

The build pipeline is managed at the root of the `stats-manuscripts` repository:

```bash
# From the repository root (/home/jonathan/Projects/stats-manuscripts)
python3 -m tools.manuscript build spatial-data-analysis
```

This command parses `current/book.toml`, stitches the chapter Markdown files into `MANUSCRIPT.md`, and compiles the complete typeset book into `MANUSCRIPT.pdf` via Typst.
