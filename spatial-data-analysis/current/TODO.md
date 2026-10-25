# TODO

Last updated 2026-10-25. Status: Chapters 0 to 13 are drafted, the consistency review (Sections 1 to 6) is applied, lint and `check_consistency.py` are clean, and every chapter's code runs. Items are ordered by what I would do first. `[ ]` open, `[x]` done.

## A. Things for you to review (start here)

Each item says where to look and what decision is needed. Most take minutes, not hours.

- [ ] **Decisions made on your behalf.** `OPEN_QUESTIONS.md`, section "Decisions Made on Your Behalf in the Consistency Fixes (2026-10-25)", ten items. I picked the recommended option each time (Appendix A scope, where the notation guide lives, first-use rule exemption for the Chapter at a Glance boxes, citation rule, symbol renames, and so on). Overrule any you dislike.
- [ ] **Schabenberger and Gotway concordance** in `NOTATION.md`. It was filled in from memory, not from the book. Check it against your copy (especially their decomposition symbols and the kriging weights notation) before the book cites it.
- [ ] **Review the drafts, chapter by chapter.** I added extra context on purpose ("add more, then trim in review"). Trim candidates and per-chapter questions are in `OPEN_QUESTIONS.md` under each chapter heading. Suggested order: 0 and 1 (they set the voice), then 2 to 5 (kriging), 6 to 9 (areal), 10 to 12 (point patterns), 13.
- [ ] **Stories.** Search for `**[STORY NEEDED]**` (one remains, in Chapter 6) and check each chapter's running story against the line in `OUTLINE.md`. These are placeholders for your own anecdotes or motivating examples.
- [ ] **Look at the republished outline page** (https://claude.ai/artifact/6oLhcyNiv1Ys5BPdVpX1Tg, version 7): math rendering and layout. I could not inspect it visually.
- [ ] **Chapter 10 and 11 case-control signal is weak.** Decide whether to strengthen it or keep it as a lesson about power (`OPEN_QUESTIONS.md`, "Across Chapters 10 to 12").
- [ ] **Chapter 10 symbol names.** The "shared setting" paragraph (Section 10.7.3) writes pop(s) and ex(s) while the code uses `pop_density` and `excess_risk`. Rename the symbols, the code, or leave it.
- [ ] **Unverified factual claims.** Chapter 4's history sentence (Krige and Matheron), Chapter 7's economics literature, and the named authors in Chapters 6, 8, 10, 11, 12 need checking against real sources.

## B. Work I can do next (say the word)

- [ ] **Write Appendices A to D.** Requirements are in `APPENDIX_REQUIREMENTS.md`; sections are in `OUTLINE.md`. Appendix A is the largest (A.0 review of the five tools, A.1 block inversion with Woodbury and determinant lemma, A.6 matrix facts). Appendix B lists canonical helper functions (note: Chapters 10, 11, 12 each define slightly different `kernel_intensity` and `fit_bt`). Appendix C is the jargon-to-linear-model Rosetta stone. Appendix D is the "where to go next" guide (finite-element SPDE, Spatial Durbin, plug-in alternatives, covariance validity).
- [ ] **Create `references.bib`.** About 13 keys are already used in the chapters; many more authors are named without keys. Each entry needs a checked DOI, so this needs web verification, not memory.
- [ ] **Chapter exercises and solutions pass.** Check that exercise numbering, hints, and "see Section" pointers survived all the edits (the checker covers section references, not exercise text).
- [ ] **Second consistency pass** after your review edits. `python3 scripts/check_consistency.py`, `python3 scripts/lint_chapter.py --all`, and `python3 scripts/run_chapter_code.py --all` rerun everything in about 4 minutes.
- [ ] **Verify `USECASES.md` references** (a duplicate Blonigen entry with different coauthors, `google.com/search` links, one "suspicious link removed," possible name-order errors) before anything from it is cited.

## C. Setup you need to do or decide

- [ ] **Install pandoc and a LaTeX distribution** (for example `brew install pandoc` and `brew install --cask mactex-no-gui`), then tell me. I will create `pandoc/preamble.tex` and a build script, compile Chapter 0, and fix any `MARKDOWNSTYLE.md` rule that fails. (You said to skip this for now; a GitHub Actions build is the no-install alternative.)
- [ ] **`git init`** in this repository before more edits, so changes can be reviewed and reverted. Right now nothing here is under version control.
- [ ] **`[!NOTE]` boxes.** The Chapter at a Glance box is a plain blockquote because pandoc prints `[!NOTE]` literally. A small Lua filter would allow GitHub-style boxes in both GitHub and the PDF. Decide if it is worth it.
- [ ] **Notation first-use checker** (declined earlier; listed only so it is not forgotten).

## D. Longer-term

- [ ] Figures: all are matplotlib PNGs in `figures/chNN/`; Chapters 0 to 1 have none. Decide whether any hand-drawn diagrams are wanted (for example the block-matrix pictures in Chapters 4 and 13).
- [ ] Decide the index and glossary approach (not planned in `OUTLINE.md` yet).
- [ ] Real data sets and case studies are deliberately out for now; revisit when you are ready (`USECASES.md` has candidate domains).
- [ ] Final copy-edit and typography pass once the PDF builds (widows, table widths, long code lines in the PDF).

## E. Known small items (low priority)

- Chapter 5 hand log-determinant is -0.5530 and the code prints -0.5531; the text already explains this as rounding, left on purpose.
- Chapter 6's MAUP simulation uses seed 2050 (also Chapter 7's response seed); left because the data differ.
- Chapter 8's grid illustration says "a few thousand spots" (78 x 64 grid) rather than a precise figure.
- Chapter 5 runtime in the lab is about 15 to 40 seconds; Chapter 13's lab is about 100 seconds (mostly the timing comparisons).
