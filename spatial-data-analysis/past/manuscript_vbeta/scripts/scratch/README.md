# Monograph Tool Harness & Quality Assurance Suite

This directory contains the automated auditing, verification, and benchmarking toolchain for the graduate monograph:
**"Theoretical Foundations of Spatial Statistics and Geostatistics: A Rigorous Synthesis and Computational Companion"** (Schabenberger & Gotway Modernization).

When returning to this repository in a few days or months, you do not need to start with a blank slate. This tool harness provides immediate, push-button verification of mathematical notation, Python syntax, runtime laboratory execution, pedagogical workbook structures, and cross-reference integrity across all 34 monograph files.

---

## 1. Quickstart (Returning to the Project)

To immediately verify that the entire monograph is in a pristine, working state, run the **Master Verification Suite**:

```bash
# Run from repository root:
python3 manuscript/scripts/scratch/verify_all.py

# Or run from inside this directory:
cd manuscript/scripts/scratch && python3 verify_all.py
```

### Execution Output
In **~0.35 seconds**, the master runner checks all 8 production verification suites and displays an executive dashboard:

```text
==============================================================================
EXECUTIVE QUALITY & INTEGRITY DASHBOARD
==============================================================================
Verification Suite                            |   Duration |          Status
------------------------------------------------------------------------------
Zero External Dependencies                    |      0.06s |          PASSED
Python AST Syntax (All Blocks)                |      0.04s |          PASSED
Computational Lab Runtime                     |      0.14s |          PASSED
Workbook 5-Tier & 208 Ex/Sol                  |      0.03s |          PASSED
Bidirectional Chapter/Workbook Links          |      0.00s |          PASSED
Internal Prose Hyperlinks                     |      0.01s |          PASSED
LaTeX Delimiter Whitespace                    |      0.03s |          PASSED
Raw Unicode Math Glyphs                       |      0.03s |          PASSED
------------------------------------------------------------------------------
Total Execution Time: 0.34 seconds
Overall Result: 100% ALL CHECKS PASSED (GREEN)
==============================================================================
```

For fast linting and syntax-only checks (skipping computational lab runtime execution), append the `--fast` flag:
```bash
python3 manuscript/scripts/scratch/verify_all.py --fast
```

To view current corpus line counts, file sizes, and cumulative monograph metrics:
```bash
python3 manuscript/scripts/scratch/audit_corpus_sizes.py
```

---

## 2. Core Architectural Principles

All tools in this repository adhere strictly to three foundational engineering principles:

1. **Zero External Dependencies (Standard Library Only):**
   - Every script is written in 100% pure Python standard library (`sys`, `os`, `re`, `glob`, `ast`, `io`, `time`, `contextlib`, `math`, `random`).
   - No `numpy`, `scipy`, `pandas`, `sklearn`, `torch`, `matplotlib`, or C-compilation dependencies are required.
   - Any machine with standard Python 3.8+ can clone and immediately run the entire suite out-of-the-box.
2. **Dynamic Working Directory Resolution:**
   - Every tool features automated directory discovery via `find_manuscript_dir()`.
   - Scripts execute identically whether invoked from the repository root (`python3 manuscript/scripts/scratch/...`), from inside `manuscript/scripts/scratch/`, or from inside `manuscript/`.
3. **High-Velocity Execution:**
   - By leveraging Python's built-in `ast` parser and lightweight standard-library runtime harnesses, all 67 embedded code blocks and 30 interactive labs across 34 Markdown files (~33,800 lines) are verified in fractions of a second.

---

## 3. Production Tool Catalog

| Script | Command | Purpose & Enforced Standard |
| :--- | :--- | :--- |
| **`verify_all.py`** | `python3 manuscript/scripts/scratch/verify_all.py` | **Master Test Suite Runner.** Executes all 8 verification engines in sequence and prints a consolidated executive quality dashboard. |
| **`audit_no_external_deps.py`** | `python3 manuscript/scripts/scratch/audit_no_external_deps.py` | **Zero-Dependency Policy Enforcer.** Scans all 34 files to ensure no unauthorized third-party libraries (`numpy`, `scipy`, `sklearn`, `pandas`, `torch`, etc.) are imported. |
| **`verify_all_python.py`** | `python3 manuscript/scripts/scratch/verify_all_python.py` | **Python AST Syntax Validator.** Compiles all 67 embedded Python code blocks across chapters, appendices, and workbooks into abstract syntax trees to guarantee 0 syntax errors. |
| **`test_lab_execution.py`** | `python3 manuscript/scripts/scratch/test_lab_execution.py` | **Computational Lab Runtime Harness.** Dynamically executes all 30 computational lab blocks across all 13 companion workbooks in isolated namespaces, verifying 0 runtime crashes. |
| **`verify_exercise_workbooks.py`** | `python3 manuscript/scripts/scratch/verify_exercise_workbooks.py` | **5-Tier Pedagogical Architecture Validator.** Verifies that all 13 companion workbooks contain all 5 pedagogical tiers and exactly 16 exercises with 16 matching step-by-step solutions (208 total). |
| **`verify_bidirectional_links.py`** | `python3 manuscript/scripts/scratch/verify_bidirectional_links.py` | **Cross-Reference Sentry.** Verifies that every chapter explicitly links to its companion workbook and every companion workbook links back to its parent chapter (0 orphaned documents). |
| **`check_prose_links.py`** | `python3 manuscript/scripts/scratch/check_prose_links.py` | **Hyperlink Integrity Auditor.** Scans all 34 Markdown files for broken relative links, missing anchor targets, or broken intra-manuscript paths. |
| **`audit_latex_accurate.py`** | `python3 manuscript/scripts/scratch/audit_latex_accurate.py` | **LaTeX Whitespace Precision Scanner.** Enforces the monograph delimiter standard (`$math$` with zero inner whitespace padding; flags `$ math $`). |
| **`check_unicode_math.py`** | `python3 manuscript/scripts/scratch/check_unicode_math.py` | **Unicode Math Glyph Detector.** Scans monograph text for prohibited raw Unicode math symbols (`ℝ`, `σ`, `μ`, `λ`, `β`, `γ`, `∑`, `∫`, `≤`, `≥`), enforcing pure LaTeX typesetting. |
| **`audit_corpus_sizes.py`** | `python3 manuscript/scripts/scratch/audit_corpus_sizes.py` | **Corpus Scale & Metrics Generator.** Produces a detailed breakdown of line counts, raw byte counts, and formatted file sizes for all 34 monograph documents. |
| **`build_pdf.py`** | `python3 manuscript/scripts/scratch/build_pdf.py` | **Monograph PDF Compiler.** Concatenates all 32 monograph Markdown files in canonical order and compiles a clean, publication-ready PDF (`manuscript.pdf`). |


---

## 4. Tool Deep-Dives & Failure Recovery

### `verify_all.py` (Master Test Suite)
- **What it does:** Imports and executes each checker, measures runtimes, catches unhandled exceptions, and returns exit code `0` (all pass) or `1` (one or more fail).
- **Options:**
  - `--fast` or `--skip-labs`: Runs static analysis and AST checks, skipping dynamic runtime execution of labs.
- **When to use:** Before committing changes, after major text edits, or when resuming work after an extended hiatus.

### `audit_no_external_deps.py`
- **What it does:** Uses regex AST pattern matching to search all Python blocks for `import numpy`, `from scipy import ...`, etc.
- **Why it matters:** The monograph's core pedagogical identity is *first-principles algorithmic clarity*. Readers must be able to inspect every Cholesky decomposition, covariance inversion, and kriging weight solution without opaque library wrappers.
- **Failure recovery:** If a violation is reported, rewrite the algorithm using Python standard library primitives or import routines from `app_d.md` (which provides full pure-Python matrix operations: LU decomposition, Cholesky factorization, determinant calculation, and Gauss-Jordan inversion).

### `verify_all_python.py`
- **What it does:** Extracts all fenced Python code blocks (` ```python ... ``` `) and passes them to `ast.parse()`.
- **Why it matters:** Guarantees that code snippets presented to students and researchers are free of syntax errors, indentation mistakes, or unclosed brackets.
- **Failure recovery:** The tool reports the exact file, block index, and Python syntax error description for rapid resolution.

### `test_lab_execution.py`
- **What it does:** Loops through the 13 exercise workbooks, isolates each Tier III computational laboratory code block, and invokes `exec()` inside a sandboxed namespace while capturing `stdout` and `stderr`.
- **Why it matters:** Syntax validity does not guarantee numerical convergence or zero division errors. This harness proves that all 30 simulation engines (variogram estimators, Ordinary Kriging, Universal Kriging, Block Kriging, Indicator Kriging, SGS, Spatial Point Processes, CAR/SAR, and Nonstationary deformation) execute and terminate cleanly.
- **Failure recovery:** If a block fails, inspect the stack trace printed in the output. Common issues include zero division in small lag bins or negative matrix pivots under invalid variogram parameters.

### `verify_exercise_workbooks.py`
- **What it does:** Scans `ch01_exercises.md` through `ch13_exercises.md` to confirm:
  1. Each file contains headers for **Tier 0**, **Tier I**, **Tier II**, **Tier III**, and **Tier IV**.
  2. Exactly 16 exercises exist per workbook, numbered sequentially `Exercise X.1` to `Exercise X.16`.
  3. Exactly 16 solutions exist per workbook, numbered sequentially `Solution to Exercise X.1` to `Solution to Exercise X.16`.
  4. All Python blocks in the workbooks compile without error.
- **Total Expected:** 208 exercises and 208 complete, step-by-step mathematical solutions.

### `verify_bidirectional_links.py` & `check_prose_links.py`
- **What they do:** Ensure navigation between theory chapters and computational workbooks is seamless and that no internal markdown links are broken.
- **Failure recovery:** If a broken link is flagged, ensure target file basenames and relative paths match exactly.

### `audit_latex_accurate.py`
- **What it does:** Scans text outside of code blocks for inline math `$ ... $` where the opening or closing delimiter is padded with whitespace (e.g. `$ Z(\mathbf{s}) $`).
- **Why it matters:** KaTeX and MathJax markdown parsers fail to render inline math when inner whitespace padding is present. Single dollar signs must be adjacent to characters (e.g. `$Z(\mathbf{s})$`).

### `check_unicode_math.py`
- **What it does:** Checks monograph content files (30 core files) for raw Unicode mathematical characters.
- **Why it matters:** Raw Unicode glyphs render inconsistently across EPUB, PDF, and web viewports. All mathematical notation must use LaTeX (`\mathbb{R}^d`, `\sigma^2`, `\le`, `\boldsymbol{\beta}`).
- **Note:** Explanatory negative examples in style-guide documentation (`README.md`, `TODO.md`, `notation_index.md`, `antigravity_notes.md`) are whitelisted by default. Use `--include-meta` to inspect meta-documents as well.

---

## 5. Historical Peer-Review Audit Reports

This directory also preserves the permanent academic peer-review records produced during Phase 5 of the monograph's development. These reports document the mathematical rigor, equation reconciliations, and proof verifications conducted across all 13 chapters and companion workbooks:

- **[`audit_report_batch5_1.md`](audit_report_batch5_1.md):**
  *Chapters 1 & 2, Appendices A & B, Workbooks 1 & 2.*
  Audits spatial variance inflation, effective sample size $n_{\text{eff}}$, infill vs. increasing-domain asymptotics, positive definiteness proofs, Bochner's Theorem, and Matérn covariance Bessel asymptotics.
- **[`audit_report_batch5_2.md`](audit_report_batch5_2.md):**
  *Chapters 3 & 4, Appendix D, Workbooks 3 & 4.*
  Audits empirical semivariogram estimators, Cressie-Hawkins robust modulus, REML contrast invariance $\mathbf{A}'\mathbf{X} = \mathbf{0}$, spatial projection operator $\mathbf{P}$, and pure-Python standard-library matrix algebra routines (Cholesky, LU, Gauss-Jordan).
- **[`audit_report_batch5_3.md`](audit_report_batch5_3.md):**
  *Chapters 5, 6, & 7, Appendix C, Workbooks 5, 6, & 7.*
  Audits Simple/Ordinary Kriging bordered linear systems, Lagrange multiplier signs, Universal Kriging dual saddle-point blocks, IRF-$k$ generalized covariances, and Linear Model of Coregionalization (LMC) cross-covariance tensors.
- **[`audit_report_batch5_4.md`](audit_report_batch5_4.md):**
  *Chapters 8, 9, & 10, Workbooks 8, 9, & 10.*
  Audits Krige's Additivity Theorem, volume regularization integrals, Lognormal Kriging Jensen corrections, Indicator Kriging PAVA isotonic monotonicity, and Journel's Fundamental Conditioning Theorem.
- **[`audit_report_batch5_5.md`](audit_report_batch5_5.md):**
  *Chapters 11, 12, & 13, Workbooks 11, 12, & 13.*
  Audits Spatial Point Processes (Ripley's $K$, Campbell's Theorem, pair correlation $g(r)$), Lattice and Areal Models (CAR, SAR, Brook's Lemma, eigenvalues of $\mathbf{W}$), and Nonstationary Covariance Models (Higdon kernel convolution, Sampson-Guttorp space deformation, Paciorek-Schervish nonstationary Matérn).

---

## 6. Guidelines for Extending or Adding Scripts

When authoring new tools for this repository:
1. **Never introduce third-party imports:** Rely strictly on Python's built-in standard library.
2. **Use dynamic path discovery:** Always use `find_manuscript_dir()` or equivalent logic so the script can be run from any working directory.
3. **Return exit codes:** Return exit code `0` on success and `1` on failure so CI/CD and the master runner can capture results.
4. **Register with `verify_all.py`:** If the script is a persistent quality check, add it as a numbered step in `verify_all.py` and document it in this README.
