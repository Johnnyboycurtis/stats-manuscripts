# Open Questions

Decisions and checks waiting for the author, collected while drafting Chapters 3 to 13 (October 2026). Each item has a recommendation; nothing here blocks reading the drafts. Delete an item once it is decided.

---

## Across Chapters 3 to 5

1. **The shared data set is "unlucky" in instructive ways.** All three chapters use `rent_district(n=250, seed=2030)`. In this realization the spatial effect $u$ drifts east (an estimated slope of 1.83), so:
   - Chapter 3's raw-rent variogram is inflated beyond the trend term.
   - Chapter 5's OLS east slope is 7.63 against a true 6, and REML lands far from the truth ($\hat{\sigma}^2 = 7.19$, $\hat{\phi} = 0.327$; truth 4 and 0.15) while $\hat{\sigma}^2/\hat{\phi}$ stays close.

   The drafts use both as teaching points: one realization, and the $(\sigma^2, \phi)$ ridge. **Recommendation:** keep the seed. A "cleaner" seed would hide exactly the lessons the chapters are about. Changing it means re-running and re-writing numbers in all three chapters.

2. **Citations.** `references.bib` does not exist (the full list across the book is in the README and in each chapter's block below). These need entries with checked DOIs before the first build with `--citeproc`:
   - `[@gneiting2013]`, Chapter 1 §1.6.4 (Matérn validity with great-circle distance).
   - `[@zhang2004]`, Chapter 5 §5.4.4 (only $\sigma^2/\phi$ is consistently estimable under infill).
   - Cressie and Hawkins (1980) and Cressie (1985), named in Chapter 3 without keys.
   - The "at least 30 pairs per bin" rule of thumb in Chapter 3, usually attributed to Journel and Huijbregts (1978). Cite or keep as "a common rule of thumb"?
   - Krige and Matheron, named in Chapter 4's opening. Cite or drop the names?

   **Recommendation:** add keys for all of them and verify each DOI in one pass.

3. **Length.** The drafts run 7,600 (Ch3), 9,400 (Ch4), and 10,250 (Ch5) words, deliberately on the long side. Candidates for trimming or for marking (Optional) are listed under each chapter.

---

## Chapter 3: Empirical Variography

1. **Trim candidates:** The Optional derivation of the Cressie-Hawkins constants (§3.2.3) could move to Appendix A if trimming is desired later. (Items 1-5 settled on 2026-10-25: keep presentation order, keep 9.5x bin variance lesson, keep runaway fit diagnostic, keep G = diag(1, 9), and keep Matern residual fit).

## Chapter 4: Kriging as BLUP

1. **Block-kriging story.** The outline's running-stories line is now updated, but the original story was "the average noise level over a park," but the chapter's data are rents, so §4.6 predicts the average rent over a neighborhood (a housing agency's benchmark) and mentions the park only as a parallel. **Recommendation:** keep rents for continuity, and update the outline's running-stories line (not yet changed).
2. **A second simulation for truth checks.** The chapter uses `rent_district()` for most examples, plus a second generator, `joint_field()`, for checks against the true surface at unlisted locations (log kriging and the coverage experiment). Is the explanation in §4.5.4 of why a second generator is needed clear enough?
3. **What else to mark Optional.** Only the leave-one-out proof (§4.7.4) is Optional. The block-inversion tool (§4.3.4) and the kernel ridge derivation (§4.5.3) are in the main text. **Recommendation:** make §4.5.3's kernel ridge derivation Optional and keep §4.3.4, which Chapters 5 and 8 use.
4. **Lognormal correction with an estimated mean.** §4.5.4 derives the exact back-transformation correction only for simple kriging and calls the ordinary and universal versions "a common approximation." Add the exact ordinary-kriging correction, or leave it for Appendix D?
5. **The symbol $\mathbf{K}$** for the bordered kriging matrix. It is bold, so it does not clash with $K(r)$ or $K_\nu$, and it is listed in `NOTATION.md`. OK?

## Chapter 5: Likelihood, REML, and Diagnostics

1. **The boundary bootstrap disagrees with the textbook mixture.** For $\tau^2 = 0$, the 200-replicate bootstrap puts 67.5% of the null distribution at zero, with a 95th percentile of 1.74, below the mixture's 2.71, so the mixture is conservative here. The text reports this as an empirical observation. Add a sentence on the likely reason (finite sample and dependence limit the information about $\tau^2$), or leave it?
2. **AIC prefers the wrong family, narrowly.** AIC picks Matérn 3/2 over the true exponential by less than 1 unit (991.17 against 991.90). The text presents this as "the data cannot tell them apart." Keep, or switch to a case where AIC picks the truth? **Recommendation:** keep; it is the honest lesson.
3. **Spatial confounding** (§5.6.2) is brief, with a pointer to Appendix D. Expand here (for example, restricted spatial regression), or keep it brief?
4. **Runtime.** The lab takes about 15 to 40 seconds of wall time, mostly the bootstrap. Acceptable?
5. **Trim candidates:** the two Optional proofs (§5.3.4 and §5.4.6) are long; either could move to Appendix A.

---

## Across Chapters 6 to 9

1. **Shared map.** All four chapters use `county_map(m=12)`, a 12 x 12 grid of square counties. Real county maps are irregular; the grid keeps the adjacency matrix checkable by hand and makes bipartite effects visible (the eigenvalue of $-1$ in Chapter 7). Chapter 6 shows rook, queen, band, and nearest-neighbor weights on it, and Chapter 8 uses a hexagonal grid for the sparse-matrix timings. **Recommendation:** keep it. An irregular map would need Voronoi code that the no-spatial-packages rule makes awkward.
2. **Citations.** New keys needed, each with a checked DOI: Clayton and Kaldor (1987), Besag, York, and Mollie (1991), Breslow and Clayton (1993), Leroux, Lei, and Breslow (2000), Riebler et al. (2016), and Sorbye and Rue (2014) (all used as keys in Chapter 9). Named without keys: Moran (1950), Geary (1954), Cliff and Ord (1981), Anselin (1995), Openshaw (MAUP), Robinson (1950, ecological fallacy), Freedman and Lane (1983), Pitman and Koopmans (Chapter 6); Besley and Case, Brueckner, Anselin-Varga-Acs, LeSage and Pace, Anselin-Bera-Florax-Yoon (1996), and Ord (1975) (Chapter 7, unattributed); Brook (1964), Besag (1974), and Kirchhoff (Chapter 8). **Recommendation:** add all keys in one pass and decide whether Chapter 7 should cite the economics literature at all.
3. **Length.** The drafts are 10,000 to 14,000 words each, with Chapter 9 the longest. Trim candidates are listed per chapter below.
4. **Part III depends on earlier chapters.** Chapter 8 draws on the Schur complement of Chapter 4 (§4.3.4) and on Chapter 7's SAR precision and Ord identity; Chapter 9 restates Chapter 8's CAR definitions. If sections are renumbered, check these references.

## Chapter 6: Areal Units, Weights, and Moran's I

1. **Multiple testing for local indicators.** §6.7.2 shows that 26 of 144 counties are flagged at 5% (and 100 pure-noise maps average 5.6 flagged counties), and recommends Bonferroni or false-discovery-rate corrections without implementing one. Add a short implementation, or leave as advice?
2. **Unequal variances break both tests.** With variances that differ across counties and no spatial structure, the z-test rejects 26.2% and the permutation test 28.8% (nominal 5%). The text points to Chapter 9 for the fix. Should Chapter 6 show the fix itself?
3. **Symmetrized weights have eigenvalues outside $\pm 1$** (for the hub map from $-0.812$ to $1.052$; for the county map $\pm 1.0023$), so $I$ is not confined to $[-1, 1]$. §6.4.4 says so. Keep?
4. **MAUP simulation is invented.** The signs (a local effect of $+1$ and a regional effect of $-0.5$) are labeled as simulated, the regional part is anisotropic so that east-west and north-south strips differ, and cells are averaged rather than population-weighted. Prefer an isotropic version?
5. **One placeholder.** §6.3.3 has a `**[STORY NEEDED]**`: a setting where a reader would care about the eigenvalues of $\mathbf{W}$ before Chapter 7.
6. **Trim candidates:** §6.5.6 (Optional), the residual-permutation experiment in §6.8.5, the null simulation in §6.7.2, exercises 15 to 20.

## Chapter 7: SAR Models

1. **The covariate is the deprivation index, not a policy variable.** The stories (tax rates, minimum wage, R&D) are told in prose only; the lab reuses the shared index. Dress the lab in one of the stories?
2. **Spatial Durbin Model** gets one sentence ("which we do not develop here") and no pointer, because Appendix D does not list it. Add a subsection, or add it to Appendix D? (This is the earlier open item on the Durbin model.)
3. **Standard errors by numerical Hessian.** The analytic information for SAR models is messier than for Chapter 5; the lab uses central differences. Add the analytic form as an Optional subsection?
4. **Error-model tests are harder.** Robust LM-error rejects 78% at $\rho = 0.6$ against 100% for the lag test; the text states it without an explanation. ML is also slightly low for $\rho$ (mean 0.586 against 0.6). Both are reported as small-sample observations.
5. **Impact intervals** use a normal approximation truncated to $|\rho| < 0.999$; coverage over many data sets is Exercise 12 and was not run.
6. **Length:** about 12,700 words. Candidates: the Optional LM derivation (§7.6.4), the eight lab subsections.

## Chapter 8: CAR Models

1. **Only one proper-CAR variant** ($c_{ij} = \rho A_{ij}/d_i$, $\sigma_i^2 = \sigma^2/d_i$) is covered. General weights and per-region variances are not. Enough?
2. **Unverified factual claim.** §8.4.3 says a standard spatial-transcriptomics capture area has about 5,000 spots (softened in Chapter 8 to "a few thousand spots"; the common 10x Visium array has 4,992). Cite, soften to "a few thousand," or drop?
3. **The incidence-matrix sampler** ($\mathbf{N}'\mathbf{N} = \mathbf{D} - \rho\mathbf{A}$) is a teaching device because SciPy has no sparse Cholesky. Production software uses CHOLMOD-style factorization. Keep, or shorten?
4. **ML, not REML, in the lab;** REML is Exercise 10. The lab truth $\rho = 0.95$ gives a downward-biased $\hat{\rho}$ (median 0.918); a truth of 0.9 or a larger map would look better. Keep both?
5. **Timings are machine-dependent;** the text reports ratios (about 5, 30, 80 times faster) with a caveat.
6. **Optional marking.** §8.2.5 and §8.3.5 are Optional. Should the Brook's-lemma subsection (§8.1.3) also be Optional?
7. **Length:** about 11,900 words. Candidates: the two Optional proofs, the hexagonal timing table, the "every SAR is a CAR" paragraph in §8.5.1.

## Chapter 9: Spatial GLMMs

1. **The truth is the BYM2 model itself** ($\sigma^2 = 0.15$, $\rho = 0.6$, $\beta_1 = 0.3$), a friendly choice. In this draw the spatial share came out at 0.40 and $\hat{\rho}$ at 0.21 (95% likelihood region 0.02 to 0.8), which the text turns into the weak-identification lesson. A smooth-plus-noise truth was tried first and gave a spatial effect so weak that $\hat{\rho}$ went to the boundary. Keep?
2. **The slope estimate is low** in this realization (GLM 0.177, BYM2 0.226, truth 0.3), partly by chance; the 100-replicate experiment (coverage 35% for the Poisson GLM, 46% for quasi-Poisson, 96% for BYM2) carries the repeated-sampling point.
3. **Four hand examples** in the main text (three-county IRLS, empirical Bayes, the BYM2 scaling constant, one penalized IRLS step) where the outline had one in the lab. Too many?
4. **Vaccination rates** (the second half of the running story) appear only in a table row and Exercise 12. Add a binomial worked example?
5. **PQL versus Laplace** give different $\hat{\rho}$ for Leroux (0.22 against 0.064). The text says only that $\rho$ is small and poorly determined.
6. **Optional marking.** Only §9.3.7 is Optional. Trim candidates: the table in §9.5.2, the marginal-variance table in §9.3.6, the derivation in §9.1.1.
7. **Appendix A** does not list the Woodbury identity by name; Chapter 9 cites it as following from block inversion, and Chapter 13 uses it.

---

## Across Chapters 10 to 12

1. **The shared case-control signal is weak, by design of the setting.** With controls at $800\,\text{pop\_density}$ and cases at $120\,\text{pop\_density}\cdot\text{excess\_risk}$, the plant adds only about 4 expected cases to a baseline of 54. Chapter 10's kernel relative-risk map and random-labelling tests cannot see it at that size (46 cases, 349 controls; they recover it only with ten times the data). Chapter 11 reports that one year of cases (58) is too few and adds a five-year pooled data set (296 cases) as its main analysis. Both chapters present this honestly as a power lesson. **Recommendation:** strengthen the signal (for example `exp(3 * exp(...))` for the plant, or a higher case rate) so the shared size shows a recoverable effect, then re-run Chapters 10 and 11 and update the numbers in §10.7.3, §10.10.2, and §11.4 to §11.5; or keep it as is and keep the lesson. Your call.
2. **Chapter 11's finding cuts against Chapter 10's kernel map.** On the pooled data the kernel and cubic-logistic relative-risk maps do worse than a constant map (RMSE of the log relative risk 0.462 and 0.390 against 0.166). Chapter 10 should perhaps cross-reference §11.5.4.
3. **One symbol, two edge corrections.** Chapter 10 defines $\omega_{ij}$ as Ripley's isotropic weight (estimator normalized by $n(n-1)$). Chapters 11 and 12 use the translation weight $1/[(1 - \lvert\Delta x\rvert)(1 - \lvert\Delta y\rvert)]$ for the unit square (normalized by $1/\operatorname{area}$ times the weighted pair sum in Chapter 11). Both are valid and each chapter says which it uses, but the notation table now lists both under $\omega_{ij}$. **Recommendation:** keep, but add a sentence to Chapter 10 pointing ahead, or give the translation weight its own symbol.
4. **A stronger trend in Chapter 12.** The shared `pop_density` covariate was too weak for the trend-only pattern to separate from CSR, so Chapter 12 uses intensity $750\,\text{pop\_density}^2$ (log-slope 2) and describes `pop_density` as local tissue density. Fine?
5. **Citations (none used as keys in Chapters 10 to 12).** Chapter 10: Ripley (1976, 1977), Besag's $L$, Diggle (1983, 2013), van Lieshout and Baddeley (1996), Hanisch (1984), Baddeley and Gill (1997), Scott (1992), Loader (1999), Lewis and Shedler (1979), Clark and Evans (1954), Kelsall and Diggle (1995). Chapter 11: Berman and Turner (1992), Baddeley and Turner (2000, 2005), Diggle (1990), Kulldorff (1997), Baddeley, Moller, and Waagepetersen (2000), Ohser. Chapter 12: Neyman and Scott (1958), Thomas (1949), Matern (1960, 1986), Moller, Syversveen, and Waagepetersen (1998), Strauss (1975), Besag (1975, 1977), Diggle (1978, 2013), Loosmore and Ford (2006), Feder and Renyi (the sequential-adsorption limit 0.5472, quoted from the literature). **Recommendation:** add keys for the main ones with checked DOIs, and keep the rest unattributed.
6. **Seeds and unlucky draws are used as lessons.** Chapter 10's CSR example is borderline for the quadrat test (p = 0.058); Chapter 12's immune-cell slide has 137 cells against 200 expected, so its minimum-contrast fit is poor; both are explained in the text. Prefer cleaner seeds?
7. **Length.** Chapter 12 is the longest chapter so far (about 14,600 words of prose, 35 code blocks). Trim candidates: §12.2.7 (Optional), the pixel-grid GMRF subsection §12.3.3, the sequential-inhibition subsection, exercises 15 to 19.
8. **Forward references.** Chapters 10 and 11 point to Chapter 12 for the Thomas process, minimum contrast, the LGCP, and Strauss; Chapter 12 restates Chapter 9's Leroux precision and Laplace approximation and points to Chapter 13's SPDE bridge, which has not been drafted yet.

## Chapter 10: Exploring Point Patterns

1. **No section is marked Optional.** Candidates: the exact naive-$K$ mean in §10.8.1 and the inclusion-exclusion argument in §10.8.3.
2. **Basketball story** appears only in the opening of §10.7 and in Exercise 21. Enough, or should the lab use a shot chart?
3. **The variance formula for $\hat{K}$** in §10.6.3 ignores edge effects and pair dependence, and understates the standard deviation by up to 17% at $r = 0.2$. The text says so. Fine?
4. **Hanisch estimator** uses the eroded area $(1 - 2d)^2$, valid for the unit square only.
5. **Trim candidates:** the bias table and ten-times-data comparison in §10.7.3, §10.9.3, exercises 15 to 21.

## Chapter 11: Point Processes as GLMs

1. **The logistic fit loses 68% in standard deviation** relative to the offset fit with five years of cases (0.475 against 0.282), because 385 controls estimate the baseline. Exercise 16 asks for 4,000 controls; the experiment was not run in the text.
2. **The block sandwich undercovers** (80 to 88%) while the pair-correlation sandwich is nearly exact (94 to 97%). The explanation for the block estimator (clusters straddle block boundaries) is plausible but untested; Exercise 22 asks the reader to vary the number of blocks.
3. **Bootstrap size.** The repeated experiment uses only 50 patterns and 99 resamples to keep the runtime near 17 seconds (coverage 0.90, 0.98, 0.92, with a standard error of 3 to 4%). Raise it?
4. **The scan test** rejects 6.5% under the null in 200 data sets (nominal 5%, standard error about 1.5%); a single fixed window's test is conservative (0.013 against 0.025) with only a hand-wave explanation.
5. **Minimum contrast is previewed** in §11.8.4 and developed in Chapter 12; the single clustered pattern's $\hat{\kappa}$ (181.7 against 100) is poor, which the text uses to explain an undersized standard error for that pattern.
6. **Trim candidates:** §11.8.2 (Optional), the two derivations of the translation weights in §11.7.2, exercises 17 to 22.

## Chapter 12: Clustering and Inhibition

1. **Thomas "fits" the trend-only pattern** (Monte Carlo p = 0.80), using a fit with $\hat{\sigma}_c$ at its upper bound of 0.2. The bounds on the minimum contrast ($\kappa \in [0.5, 5000]$, $\sigma_c \in [0.002, 0.2]$) are an arbitrary choice that prevents degenerate fits. Fine?
2. **Goodness-of-fit tests are conservative** because parameters are estimated from the same data and the reference distribution is not refit (except the $K_{\text{inhom}}$ column). Exercise 19 asks the reader to measure it. Add a worked calibration instead?
3. **Strauss pseudolikelihood is slightly biased** toward weak interaction (median $\hat{\zeta} = -1.76$ against $-1.90$), reported without a derivation; bootstrap SD 0.266, fresh-slide SD 0.327, naive SE 0.215.
4. **Hand examples are in the main text** (§12.2.5, §12.4.4); the lab is a gallery of seven figures organized by figure rather than numbered subsections.

---

## Chapter 13: Large-n Methods

1. **Citations.** Keys used without `references.bib` entries (each needs a checked DOI): `banerjee2008` (predictive process), `cressie2008` (fixed-rank kriging), `wendland1995`, `vecchia1988`, `lindgren2011`. Named without keys: Sherman-Morrison-Woodbury; Furrer, Genton, and Nychka (2006) for tapering; Katzfuss and Guinness (2021) for Vecchia orderings; the Kullback-Leibler optimality of the Vecchia model (stated as known).
2. **Exponential models are hard for low rank.** At the shared range, 625 knots on 2,000 listings leave a log-likelihood shortfall of 10.9; the smooth Matern 5/2 model needs about 256 knots for a shortfall of 6. The chapter reports roughness, not range, as what matters, and does not fit parameters by low-rank maximum likelihood (Exercise 13). Add that experiment?
3. **The 20,000-listing data are not an exact draw.** They come from the Vecchia model with 30 neighbors, because exact simulation is out of reach; the text says so. The fit recovers the nugget (1.018) and $\sigma^2/\phi$ (25.84 against 26.67), while $\hat{\sigma}^2$ and $\hat{\phi}$ sit on Chapter 5's ridge. Acceptable?
4. **Tapering reads as a weak section, honestly.** SciPy has no sparse Cholesky, and sparse LU is no faster than dense Cholesky at $n \le 5{,}000$. Keep, add a conjugate-gradient implementation (Exercise 7 only describes it), or shorten?
5. **Unexplained observations.** The low-rank shortfall grows faster than $n$ with 144 knots held fixed (0.03 per listing at $n = 1{,}000$, 0.08 at $n = 5{,}000$); the text says it did not investigate why. The neighbor search and maximin ordering are $O(n^2)$ in this teaching implementation, which is why Vecchia is run only up to 20,000.
6. **Timings are machine-dependent.** All seconds come from one run with BLAS limited to one thread; ratios are the finding.
7. **Unproved statements.** The SPDE result, the unbounded variance of the CAR field as the grid refines ($\nu = 0$), and the Kullback-Leibler optimality of the Vecchia model are stated without proof. Appendix A.1 should list the Woodbury identity and the determinant lemma by name, and Appendix D should say "finite-element mesh" for the full SPDE approach.
8. **Notation.** $\kappa$ now has three meanings (knot location, SPDE inverse range, parent intensity). Use another letter for the SPDE parameter (for example $a$)?
9. **Length and trimming.** About 14,700 words including code, 15 code blocks, 20 exercises. Candidates: §13.3 (tapering), §13.5.3, the Kronecker proofs in §13.6.2, exercises 13 to 20. No section is marked Optional.

---

## Decisions Made on Your Behalf in the Consistency Fixes (2026-10-25)

You asked to proceed through Sections 2 to 6 of `CONSISTENCY_REVIEW.md` without answering the questions in them, so I chose the recommended option each time. Overrule any of these:

1. **Appendix A scope.** Chapter 0 promises a review of the five tools in Appendix A, so `OUTLINE.md` now has A.0 (the five-tool review, one subsection per tool) and A.6 (eigendecomposition and Cholesky facts, projections, Lagrange multipliers, the Neumann series, the Gaussian integral), and A.1 now names the Woodbury identity and determinant lemma. The alternative was to soften Chapter 0's promises.
2. **Where the notation guide lives.** The Schabenberger and Gotway concordance stays in `NOTATION.md`; the sentences in Chapters 0 and 13 and in `MARKDOWNSTYLE.md` that said Appendix C holds it now point to the notation section.
3. **First-use rule and the Chapter at a Glance boxes.** The boxes are exempt (they preview symbols defined later); the rule applies from the first section on. Written into `MARKDOWNSTYLE.md` and `NOTATION.md`.
4. **Citation rule.** Use a `[@key]` wherever the text credits a named author; add the entry to `references.bib` with a checked DOI; build without `--citeproc` until the file exists. About 13 keys are in use and many authors are named without keys.
5. **Symbol renames** (to remove clashes): in Chapter 9 the Laplacian eigenvalues are $\xi_k$ (as in Chapter 8), the eigenvalues of $\operatorname{Var}(\mathbf{u})$ are $\varsigma_k$, and the simulation normals are $z_k$; in Chapter 6 the normalized adjacency is $\tilde{\mathbf{A}}$ (not $\mathbf{S}_{\mathbf{A}}$) and the quadratic-form normals are $z_k$; the mixed-model matrix of §2.1.5 is $\mathbf{Z}_b$; Chapter 12's Leroux precision is written $\sigma^{-2}[\ldots]$ instead of with a scale $\xi$; the Chapter 6 Moran null uses $\tau^2$.
6. **Chapter 8's four-region map was relabeled** to Chapter 6's numbering (the hub map: region 2 is the hub, region 1 the pendant); results are the same up to relabeling.
7. **Chapter 4 gained §4.8.5** (kriging standard error against the number of listings, averaged over 20 location sets; one draw per size was not monotone). Chapter 5 gained standardized leave-one-out residuals in §5.6.1 and a WLS-versus-ML-versus-REML row in §5.2.2. Chapter 4's remark about the cost of plugging in estimates was softened rather than backed by a new coverage experiment.
8. **Nearest-neighbor weights in Chapter 6** break ties by region number, which leans the neighbors toward the south and west; the quoted values changed slightly and the text says so.
9. **Seeds:** Chapter 6's MAUP simulation uses seed 2050 (Chapter 7's response seed); left as is because the data differ.
10. **SciPy submodules:** `scipy.spatial` is allowed (it is part of SciPy); noted in `MARKDOWNSTYLE.md` and Chapter 13.

## Still Open From the Consistency Review

1. **Check the Schabenberger and Gotway concordance against the book.** It was filled in from memory ($W$ and $\eta$ in their decomposition; $\boldsymbol{\lambda}$ for kriging weights). Please verify against your copy before the book cites it.
2. **Chapter 10's "shared setting" paragraph** still writes $\text{pop}(\mathbf{s})$ and $\text{ex}(\mathbf{s})$ while the code uses `pop_density` and `excess_risk`; Chapter 11 maps the names, Chapter 10 does not.
3. **Two reviewer findings turned out to be wrong or already fine:** the "0.404 versus 0.408" relative risk at the plant (only 0.404 appears), and the claim that Chapter 4 does not mention the exchanged-blocks form of the block inverse (it does; Chapter 13 now matches it).
4. **Skipped on purpose:** the fourth-decimal difference between the hand log-determinant (−0.5530) and the code (−0.5531) in Chapter 5, which the text already explains as rounding of the eigenvalues.
5. **Unattributed material.** Chapter 4's history sentence (Krige and Matheron), Chapter 7's economics literature, and most named authors in Chapters 6, 8, 10, 11, and 12 have no citation keys yet.

---

## Carried Over From Earlier Sessions

1. **PDF build.** Pandoc and LaTeX are not installed yet. Next steps when they are: create `pandoc/preamble.tex` and a build script, compile Chapters 0 to 5, and adjust `MARKDOWNSTYLE.md` if a rule fails. A GitHub Actions build is an alternative that needs no local install.
2. **GitHub-style `[!NOTE]` boxes.** The "Chapter at a Glance" box is a plain blockquote because pandoc prints `[!NOTE]` literally. A small Lua filter would allow `[!NOTE]` in both GitHub and the PDF. Not yet decided.
3. **Version control.** The repository is not under git. Recommended before more chapters are added.
4. **`USECASES.md` references** have known problems (duplicate Blonigen entry with different coauthors, `google.com/search` links, one "suspicious link removed," possible name-order errors). Verify before citing anything from it.
5. **Spatial Durbin Model** in Chapter 7: see Chapter 7, item 2 above.
