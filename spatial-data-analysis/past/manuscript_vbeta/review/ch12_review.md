# Review: Chapter 12 — Areal Lattice Models (SAR, CAR, SMA), Spatial Autocorrelation & Spatial GLMMs

**File reviewed:** `ch12_lattice_areal_models.md` (the 500-line, 26,697-byte version last modified 12:56:36; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch12_exercises.md` was not reviewed. The chapter points there for the Moran variance derivation, Brook's lemma, and the disease-mapping labs, so several results the chapter *uses* are proved only in a file outside this review.
**Version note:** this file is dated earlier than the other chapters I reviewed (12:56 against 20:09–20:14) and has a different format: no numbered theorems, no "Beat" headings, one short anatomy box. It looks like a draft that has not yet been brought up to the Chapter 9–11 template. The review treats it as it stands.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, Monte Carlo, and the chapter's own code, exec'd from the file). Statements marked "from memory" or "by reasoning" were not tested.

---

## 1. Executive Summary

**Grade: 5 / 10**

The choice of ideas is right and the chapter is short and readable. Time series AR(1) as the doorway to spatial autoregression (L12–40) is the correct bridge for this audience. Moran's $I$ as a spatial Pearson correlation, the LISA identity, the SAR covariance, the CAR precision matrix $\mathbf Q=\tau^{-2}(\mathbf D-\phi\mathbf W)$, and the ICAR singularity are all stated correctly, and I verified most of them numerically (details below). The Moran variance formula in the code is exactly right: over 20,000 white-noise fields on an $8\times8$ grid the simulated standard deviation was 0.0932 against the formula's 0.0932, and the nominal 5% test rejected 5.2%.

The trouble is that the chapter *states* where it should *teach*, and it promises more than it delivers.

1. **The code promises two things it does not contain.** L327–328 advertise an "exact Moran's $I$ analytical permutation test" and "a profile likelihood Maximum Likelihood estimator for the SAR spatial autoregression parameter $\rho$." The code has neither: it computes Moran's $I$ with a normal-theory variance and simulates a SAR field. The SAR estimator, the central algorithm of the SAR section, is described in one sentence ("a fast 1D line search," L213) and never given.
2. **The bridge is half right.** L35–38 argue that OLS of $Y_i$ on its neighbors' average is inconsistent (it is: that is the *spatial lag* model), but the SAR model written at L176 is the *spatial error* model, in which OLS for $\boldsymbol\beta$ is consistent. The two models are never distinguished, so the bridge and the model do not match.
3. **Several statements are false as written.** Geary's $c$ is not confined to $[0,2]$; the SAR "invertible if and only if $\rho$ lies in the interval" is not an invertibility condition; the ICAR rank is $n-1$ only for a connected map; and the SMA "no common neighbors" range statement is incomplete. All four are easy fixes.
4. **There is no depth where a student needs it.** Brook's lemma and Hammersley–Clifford are named, not explained (L227). Nothing says *when to use SAR versus CAR*, how to fit a CAR or BYM model, why disease mapping exists (the problem with raw rate maps), or how to test spatial autocorrelation in the residuals of a regression, which is how Moran's $I$ is actually used.

**Highest-value changes**
1. Deliver the algorithm: add the profile-likelihood recipe for SAR (given $\rho$, GLS for $\boldsymbol\beta$ and $\sigma^2$; add the log-determinant; scan $\rho$) and a permutation test for Moran's $I$. I checked that this recipe recovers $\rho$ from simulated data.
2. Use the 1D AR(1) as the Rosetta stone: it is simultaneously a causal SAR-type model and a two-sided *conditional* (CAR) model, and the two coincide in 1D. Showing why they split apart in 2D is the cleanest way to teach SAR versus CAR.
3. Add a worked example with a map: raw rates, Moran scatterplot, SAR fit, CAR or BYM smoothing. The chapter currently never shows an areal data set.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: yes, with two jargon leaks.** The chapter lives comfortably inside matrix algebra, eigenvalues, conditional normals, and Poisson regression.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L227–229 | "Brook's Lemma and the Hammersley–Clifford Theorem": named, neither explained | **Medium** | The heading promises two results and delivers a one-line assertion. Say in words what is needed: *a set of full conditionals defines a valid joint distribution only if the implied precision matrix is symmetric and positive definite*, show the 2-region case (two lines), and cite Brook/Besag in a footnote. |
| L83 | "By Gershgorin's Circle Theorem" | Low | Correct but unnecessary. "Every row sums to 1, so $\mathbf W^*\mathbf 1=\mathbf 1$ (an eigenvalue of 1), and no eigenvalue can exceed the largest row sum." |
| L304–315 | "improper prior" | Low | Fine; add a plain-language sentence: *a prior that does not integrate to 1 but still yields a proper posterior once data are added.* |
| L315, L229 | "Intrinsic," "improper," "Markov random field" | Low | Define at first use. |

No measure-theory vocabulary. This is the best chapter on that axis.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 12.1 | Yes (strong bridge) | n/a | Bridge mismatch (below) |
| 12.2 | Brief | **Missing** (code is in §12.6) | **Missing** |
| 12.3 | Yes | Partly (Moran only, in code) | **Missing** |
| 12.4 | Yes | **Missing** (no estimation algorithm) | **Missing** |
| 12.5 | Yes | **Missing** | **Missing** |
| 12.6 | n/a | Code present | Demo only |

No proofs, no worked numeric examples, no theorems, no in-chapter exercises (Chapters 9–10 each have one). The chapter reads as a well-chosen outline, about a third the length of its neighbors.

### Header and §12.1 From Time Series AR(1) to Spatial Lattices

**What works.** The AR(1)-to-lattice story is a good opening: *time has a direction, space does not.* The two ASCII pictures (L20–33) make the contrast concrete.

**Issues**
- **The bridge and the model do not match (L35–37 versus L176).** L37 says that regressing $Y_i$ on the neighbor average $\sum_jw_{ij}Y_j$ by OLS is inconsistent because the lag is correlated with the error. That is true of the *spatial lag* model $\mathbf Y=\rho\mathbf W\mathbf Y+\mathbf X\boldsymbol\beta+\boldsymbol\epsilon$. The model the chapter then defines, $\mathbf Y=\mathbf X\boldsymbol\beta+\rho\mathbf W(\mathbf Y-\mathbf X\boldsymbol\beta)+\boldsymbol\epsilon$ (L176), is the *spatial error* model, in which the correlation sits in the errors and OLS estimates of $\boldsymbol\beta$ stay consistent (but inefficient, with wrong standard errors). Students will hit both in practice. Name both models, say which one this chapter develops, and say which of the two the OLS inconsistency applies to.
- **"Causal asymmetry" (L18) is a good idea overstated.** A sentence that OLS "yields consistent, asymptotically efficient estimates" for AR(1) is true for the conditional likelihood; say "conditional on the first observation."
- **The AR(1) is a missed Rosetta stone.** In one dimension, AR(1) can be read as a *structural* (SAR-like) equation $Y_t=\rho Y_{t-1}+\epsilon_t$ or as a *two-sided conditional* (CAR-like) model, $Y_t\mid\text{rest}\sim\mathcal N\!\left(\frac{\rho}{1+\rho^2}(Y_{t-1}+Y_{t+1}),\,\frac{\sigma^2}{1+\rho^2}\right)$ (a standard result, from memory), and the two descriptions give the same distribution in 1D. In 2D they do not, which is exactly why there are two families. This is the one bridge that explains the whole of §12.4, and it costs half a page.
- **The comparison table is missing** (see §12.4).
- **No dedication paragraph here,** which is a relief, but the opening still lacks a promise. A sentence such as *"By the end you will be able to test a map for spatial autocorrelation, fit a regression that accounts for it, and smooth a noisy map of disease rates"* would orient the reader.

### §12.2 Spatial Proximity and Weight Matrices

**What works.** The rook/queen definitions are clear. The row-standardization callout (L76–83) is correct: $\mathbf W^*$ is asymmetric but similar to a symmetric matrix, so its eigenvalues are real, and its largest eigenvalue is 1. I checked this on an $8\times8$ rook grid: eigenvalues of $\mathbf W^*$ range over $[-1,1]$ and the largest imaginary part is $10^{-16}$.

**Issues**
- **Islands break row-standardization.** A region with no neighbors has $d_i=0$ and $w_{ij}/d_i$ is undefined; the chapter's code quietly writes a zero row (L379–382), but the text never mentions the case. Add a sentence on islands (common in real maps: islands, exclaves), and that the "similar to a symmetric matrix" argument needs $d_i>0$.
- **$k$-nearest neighbors is named but not given a formula (L59),** and the distance formula $w_{ij}=d_{ij}^{-\alpha}\mathbf 1\{d_{ij}\le d_{\max}\}$ is a distance band, not $k$-NN. Also, $k$-NN weights are asymmetric by construction ($i$ may be among $j$'s nearest neighbors without the reverse), which matters for §12.4's symmetry requirement. Say so.
- **No guidance on choosing $\mathbf W$.** The single most consequential modeling choice is not discussed: results are sensitive to $\mathbf W$, row-standardization changes the interpretation, and the same area can be tiled differently (the modifiable areal unit problem; from memory a standard caveat). Two sentences would do.
- **Beat 2 and 3 are missing.** A 5-region toy map, written out as a $5\times5$ matrix, would make every later formula checkable by hand.

### §12.3 Global and Local Diagnostics

**What works.** The Moran–Pearson analogy (L91) is the right framing. The formulas for $I$, $c$, and the local $I_i$ are correct, and the LISA identity $\sum_iI_i=S_0\,I$ (L132–139) is correct (I checked the algebra and the row-standardized special case). The Moran-scatterplot quadrants are labelled correctly.

**Issues**
- **One sentence would make Moran's $I$ a regression slope.** For a row-standardized $\mathbf W$, $I=\mathbf z'\mathbf W\mathbf z/\mathbf z'\mathbf z$ is exactly the OLS slope of the spatial lag $\mathbf W\mathbf z$ on $\mathbf z$. I verified numerically (both equal $-0.13102$ on a random field). That is the Moran scatterplot's fitted line (L140), a direct regression bridge, and it is not mentioned.
- **"Under CSR" (L104) and "CSR Null" (L477) collide with Chapter 11,** where CSR means a point process with Poisson counts. Use "no spatial autocorrelation" or "exchangeability."
- **Moran's null needs a sentence of explanation.** $\operatorname{E}[I]=-1/(n-1)$ holds when values are exchangeable (permutation null); say what is permuted. The chapter's explanation, "because the mean is estimated" (L108), is true in spirit; the cleaner reason is that the centered values sum to zero, so they cannot all be positively related.
- **The variance, the heart of the test, is not in the chapter.** L112 says the closed form "is derived in Exercise 12.5." The test statistic $Z_I$ is useless without it. Put the formula in the text (it is in the code at L420–423: I checked that it reproduces simulation) and keep the derivation in the exercises. State which null it is for: independent normal values (not regression residuals) with the *given* $\mathbf W$.
- **Missing, and important: test the residuals, not the raw data.** In practice you fit a mean model (Chapter 6) and test the *residuals* for leftover spatial autocorrelation. The null mean and variance of $I$ for regression residuals differ from the raw-data formulas (they involve $\mathbf X$). The chapter never says this, and its $Z_I$ applies only to raw data.
- **Geary's range is wrong (L122).** "$c\in[0,2]$" is a common rule of thumb, not a bound. With binary weights on a star graph (one hub, $k$ leaves, hub value 1 and leaves 0), I get $c=2.5,\ 4.5,\ 10.5$ for $k=4,8,20$ (the value is $(k+1)/2$ in these cases). Say "values near 1 mean no autocorrelation; values below 1 mean positive; above 1 mean negative; the upper limit depends on $\mathbf W$." It is also worth one sentence on why Geary and Moran can disagree (cross-products versus squared differences; Geary is more sensitive to local differences).
- **Moran's range** is also not stated. For row-standardized weights it lies between the smallest and largest eigenvalues of $\mathbf W^*$ (here $[-1,1]$), not "$[-1,1]$ like a correlation" in general. Note that it is not bounded by 1 for every $\mathbf W$.
- **LISA has no inference.** Local $I_i$ is shown without a test. The usual approach is a conditional permutation test (permute the other values holding $z_i$ fixed) with a multiple-testing correction (from memory). Without that, "hotspot" is a label for a number, not a finding. Say so.
- **Beat 3 is absent.** Pitfalls: the result depends on $\mathbf W$; Moran's $I$ detects *global* autocorrelation, not its cause (a trend in the mean produces it too: the same trend-versus-interaction issue as Chapter 11); and outliers dominate $\sum z^2$ in the denominator.

### §12.4 SAR vs. CAR (and SMA)

**What works.** The side-by-side comparison box (L152–168) is the right idea. The SAR covariance (L190), the CAR precision (L236), and the symmetric-weights construction (L242–249) are correct. The Jacobian remark (L208) is **correct and important**: I fit a SAR model with and without the log-determinant to 100 simulated $8\times8$ data sets with true $\rho=0.65$. With the Jacobian the estimate averages 0.60 (sd 0.10); without it, 0.94 (sd 0.09), pinned near the upper boundary. "Fails catastrophically" is not hyperbole; quote those numbers instead of the adverb.

**Issues**
- **"Invertible if and only if $\rho$ lies in $(1/\lambda_{\min},1/\lambda_{\max})$" (L195–201) is not an invertibility condition.** $\mathbf I-\rho\mathbf W$ is invertible for *every* $\rho$ except the finitely many values $1/\lambda_i$. I checked $\rho=0.9,1.5,2.5$ on the $8\times8$ grid: $\mathbf I-\rho\mathbf W^*$ is invertible each time and $\boldsymbol\Sigma_{\text{SAR}}$ is positive definite each time. The interval is the stretch *containing $\rho=0$* on which the model varies continuously from independence and the power series $\sum_k\rho^k\mathbf W^k$ converges. Say that, and use the word "convention," not "iff." Also: the bounds assume real eigenvalues, which is guaranteed for row-standardized symmetric $\mathbf W$ (L83) but not for a general weight matrix, and the lower bound for row-standardized $\mathbf W^*$ is $1/\lambda_{\min}\le-1$, usually written as $(-1,1)$.
- **The SAR estimation algorithm is missing.** Give it. Given $\rho$, let $\mathbf B=\mathbf I-\rho\mathbf W$; then $\hat{\boldsymbol\beta}(\rho)=(\mathbf X'\mathbf B'\mathbf B\mathbf X)^{-1}\mathbf X'\mathbf B'\mathbf B\mathbf Y$ (this is GLS with precision $\mathbf B'\mathbf B/\sigma^2$; link to Chapter 6), $\hat\sigma^2(\rho)=\mathbf e'\mathbf B'\mathbf B\mathbf e/n$, and the concentrated log-likelihood is $\ell(\rho)=-\tfrac n2\ln\hat\sigma^2(\rho)+\sum_i\ln(1-\rho\lambda_i)$ up to constants. Scan $\rho$ over a grid and take the maximum. I implemented it (grid of 397 points) and it recovers $\rho$ with a small downward bias (mean 0.60, sd 0.11 over 100 data sets of $n=64$, true 0.65). That is 12 lines, and it is the chapter's promised "profile likelihood" (L328).
- **Why the Jacobian exists is not explained.** The density of $\mathbf Y$ is the density of $\boldsymbol\epsilon$ transformed through $\mathbf B$, so the change-of-variables factor $|\det\mathbf B|$ appears. Say that in one sentence ("the same reason a Jacobian appears in any transformation of random variables"), and give the intuition: without it, the likelihood can be raised just by making $\mathbf B\mathbf Y$ small, which is what pushes $\hat\rho$ toward 1.
- **Brook's lemma is named and not used (L227–229).** Show the 2-region example: $Y_1\mid Y_2\sim\mathcal N(c_{12}Y_2,\tau_1^2)$, $Y_2\mid Y_1\sim\mathcal N(c_{21}Y_1,\tau_2^2)$ define a joint normal only if $c_{12}/\tau_1^2=c_{21}/\tau_2^2$. This is the entire content of L229–232 and a student can verify it by writing down two precision-matrix entries.
- **Notation slip (L242).** "When $\mathbf C=\phi\mathbf W$ for a row-standardized matrix $\mathbf W^*=\mathbf D^{-1}\mathbf W$" should read $\mathbf C=\phi\mathbf W^*$ (and then $\mathbf W$ in L248 is the *binary* matrix). Three different $\mathbf W$s appear in 10 lines.
- **The valid range of $\phi$ is not given.** $\mathbf Q=\tau^{-2}(\mathbf D-\phi\mathbf W)$ is positive definite for $\phi\in(1/\lambda_{\min},1)$ where the eigenvalues are those of $\mathbf D^{-1/2}\mathbf W\mathbf D^{-1/2}$; at $\phi=1$ it becomes the ICAR of §12.5. Say that, and say that is *why* ICAR is the limit case.
- **There is no "SAR versus CAR: when do I use which?"** A student cannot choose a model from this chapter. Useful content: both define a covariance through one parameter and a weights matrix; the SAR precision is $\mathbf B'\mathbf B/\sigma^2$ (which has a CAR form with $\mathbf C=\rho(\mathbf W+\mathbf W')-\rho^2\mathbf W'\mathbf W$; from the algebra, not tested), so every SAR is a CAR with a wider neighborhood, but not conversely; SAR is the habit of econometrics and regression error modelling, CAR the habit of hierarchical Bayesian disease mapping; CAR's sparse precision is what makes MCMC fast. A comparison table with these rows would be the chapter's most useful single addition.
- **The "Lecture Hall" connection to Chapter 6 is missing.** $\boldsymbol\Sigma_{\text{SAR}}^{-1}=\mathbf B'\mathbf B/\sigma^2$ is a precision matrix to be used in generalized least squares exactly as in Chapter 6, with covariance replaced by precision. One sentence.
- **SMA (L269–279).** The covariance formula is right. The range statement is incomplete: "covariance drops strictly to zero between regions that share no common neighbors" is wrong for regions that are *themselves neighbors* with no common neighbor. On a 5-region path with $\theta=0.5$ the covariance matrix has entry 1.0 between adjacent end regions even though they share no neighbor; covariance is zero only at graph distance $\ge3$. State: "nonzero for neighbors and for regions sharing a neighbor; zero beyond that." SMA also gets no estimation or identifiability discussion and no note that it is rarely used in practice (from memory).
- **Beat 3 is absent for the whole section:** what do the fitted $\rho$ and $\phi$ *mean*? (They are not comparable across SAR/CAR or across different $\mathbf W$.) How do you check the fit (Moran's $I$ on the residuals, which closes the loop with §12.3)?

### §12.5 Spatial GLMMs and Disease Mapping (BYM)

**What works.** The BYM decomposition $\eta_i=\mathbf x_i'\boldsymbol\beta+u_i+v_i$ with a structured $u_i$ and unstructured $v_i$ is explained clearly (L296–302), and ICAR as a two-dimensional random walk (L315) is a good intuition. The ICAR conditional (L307) and precision (L311) are correct.

**Issues**
- **The motivation is missing, and it is a good one.** Why does disease mapping exist? Because raw ratios $Y_i/E_i$ (standardized mortality ratios) have variance proportional to $1/E_i$, so the most extreme values on a rate map are the *smallest* populations. BYM shrinks noisy small-area estimates toward neighbors (and toward the overall rate). Without this paragraph, §12.5 reads as notation. A tiny example (a county with $E=0.5$ expected cases and 2 observed, SMR 4.0) makes the point.
- **Rank statement is conditional on connectivity (L314).** $\mathbf D-\mathbf W$ has rank $n-1$ *only if the map is one connected piece*. With $c$ connected components the rank is $n-c$ (I verified: $8\times8$ grid, rank 63; a map with two components, rank 4 of 6). Islands (D=0) are components of size 1. The zero-sum constraint (L316–319) must then be imposed *per component*. Real maps often have islands, so this is not a technicality.
- **$u_i$ and $v_i$ are not separately identified in a simple way.** Both absorb overall spatial smoothing; the usual practical fix is to rescale the ICAR component so that $u$ and $v$ have comparable marginal variances (from memory, a 2010s refinement). A sentence acknowledging that the two variance parameters trade off, and that the *total* smoothing is better determined than the split, is enough.
- **No algorithm.** "MCMC or INLA" (L316) is the only reference to fitting. Beat 2 should at least say what an MCMC step looks like for this model (each $u_i$ has an ICAR conditional given its neighbors, and the Poisson likelihood), or say plainly that fitting is done with software and give the conceptual loop. The reader cannot currently go from this section to a smoothed map.
- **The classical bridge is absent in this section.** BYM is a Poisson regression (with an offset $\log E_i$) plus random effects. Say that, with a Lecture Hall box on Poisson regression with an offset (the same offset device as in Chapter 11's inhomogeneous Poisson likelihood), and on random effects as shrinkage (the student's mixed-model course).
- **No link back to Chapters 9–11.** The log-Gaussian Cox process of Chapter 11 §11.5 is the continuous-space cousin of BYM (a Poisson count whose log-intensity is a Gaussian field), and Chapter 9's probit/logit remark in §9.1 is the non-spatial version of the link function. Both connections are free and are not made.
- **"GLMM" is in the title and the section is a BYM-only treatment.** Either broaden (spatial logistic with a CAR random effect, one paragraph) or retitle to "Disease Mapping with BYM."

### §12.6 Algorithmic Engines in Pure Python

**What works.** The class design is tidy; the grid builder and row-standardization are clear; the Moran code reproduces the formula. On the demonstration: the white-noise field gives $I=0.104$ ($Z=1.29$, $p=0.20$) and the SAR field with $\rho=0.65$ gives $I=0.319$ ($Z=3.59$, $p=0.0003$). For power, across 5,000 simulated SAR fields the test rejected 98% of the time, so the demo is representative.

**Issues**
- **The description promises more than the code has** (see Executive Summary). L327 says "exact Moran's $I$ analytical permutation test," but the only test is a normal approximation with the Cliff–Ord variance under normality (L420, which is itself fine). There is no permutation test and no SAR estimator.
- **Add the missing pieces, each about a dozen lines:** a permutation test (shuffle $\mathbf y$ 999 times, recompute $I$, report the rank) and the SAR profile-likelihood scan described above. They are the algorithms that students most need to see.
- **A single white-noise realization is a weak "null experiment" (L472–481).** It shows one draw landing inside the null band, which proves nothing and can mislead if the seed had given $p=0.03$. Say it is one draw, or run 1,000 and report the rejection rate (5.2% in my run).
- **Small points:** (a) the comment "25 terms give $<10^{-6}$ error for $\rho=0.6$" (L450) is slightly optimistic and the demo uses $\rho=0.65$; the truncation factor is $0.65^{26}\approx1.4\times10^{-5}$, which is harmless, but the comment should match; (b) the code uses dense $n\times n$ lists and $O(n^2)$ loops everywhere. For a teaching example it is fine, but one sentence should say real maps use sparse matrices (hundreds to thousands of regions) and that CAR's sparse precision is exactly why it scales; (c) `random.seed(seed)` changes the global generator, which would surprise a reader who uses `random` elsewhere; (d) the label "CSR Null" (see above).
- **No code for CAR or BYM,** consistent with the lack of an algorithm in §12.5.

### Exercises

There is no in-chapter exercise section, and several results are deferred to `ch12_exercises.md` (Moran variance, Brook's lemma, SAR vs. CAR matrix inversions, disease-mapping labs). For a textbook, the principal results should be provable from the text. Candidates for in-chapter worked exercises with the chapter's own material: (1) build $\mathbf W$ and $\mathbf W^*$ for a 5-region map and compute Moran's $I$ by hand; (2) show the 2-region Brook symmetry condition; (3) compute $\boldsymbol\Sigma_{\text{SAR}}$ for 3 regions in a line and compare with the CAR covariance; (4) show $\mathbf Q_{\text{ICAR}}\mathbf 1=\mathbf 0$ and find the rank for a map with two components; (5) the SMR for a small county versus its BYM-shrunk estimate.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **4.5 / 10**

- **No Three-Beat structure at all.** The chapter has a strong opening bridge, then a sequence of formulas. Beats 2 and 3 are mostly absent, and the code at the end is not tied to a particular section.
- **Good opening, unfinished middle.** §12.1 is the voice the book wants. §12.4 and §12.5 have the substance but no scaffolding: no worked numbers, no pictures of maps, no "what goes wrong."
- **No map, no data.** Areal data analysis without a single areal picture is hard to learn from. A 5- or 8-region toy map, reused in every section, would hold the chapter together.
- **No "From the Lecture Hall" boxes.** Candidates: simple linear regression slope $=$ cov/var (Moran as a slope); Jacobian of a transformation (SAR likelihood); AR(1) in two views (SAR vs. CAR); Poisson regression with an offset; random effects as shrinkage (BYM).
- **The chapter is brief,** which is a strength in places, but material that a student must supply on their own (what $\rho$ means, how to fit, how to check) is precisely the half that makes it useful.

### Dimension 2 — Algebraic transparency and scaffolding: **6 / 10**

- **Strong:** the row-standardization similarity argument, the LISA identity, the SAR covariance, the CAR symmetry construction, and the ICAR row-sum argument. All are short, correct, and checkable.
- **Needs work:** (a) Brook's lemma needs its two-region example; (b) the invertibility statement (L195–201); (c) the Jacobian's origin; (d) the three $\mathbf W$s at L242–248; (e) the Moran variance formula, which lives only in the code and the exercise file.
- **Anatomy blocks:** one (the CAR precision, L251–265), which is good. SAR's log-likelihood (L206) and BYM's linear predictor (L294) each deserve a term-by-term anatomy in the same style.

### Dimension 3 — Structural consistency and forward referencing: **4 / 10**

- **Format drift:** this chapter lacks the numbered theorems, anatomy blocks, worked exercises, and Beat structure of the others. Even the headings use `##` where the neighboring chapters use `###`.
- **Missing back-references:** Chapter 6 (GLS: SAR's $\mathbf B'\mathbf B/\sigma^2$ is a precision matrix to be used in GLS; testing residuals), Chapter 2 and Chapter 5 (the geostatistical covariance models and BLUP that these lattice models replace), Chapter 9 §9.1 (link functions), Chapter 11 §11.5 (LGCP as the continuous-space cousin of BYM).
- **Missing forward reference:** Chapter 13 (nonstationarity) once weights or $\rho$ vary across space.
- **Notation:** $\mathbf W$ means three different matrices in L242–248 ($\mathbf W$, $\mathbf W^*$, and a row-standardized $\mathbf W$ in the code); $\lambda$ is eigenvalue here and intensity in Chapter 11; $\phi$ is a CAR parameter here and a normal density in Chapter 9; $\tau$, $\sigma$ vary across SAR, CAR, and BYM without a table. A one-page notation table for this chapter would fix it.
- **Terminology collisions:** "CSR" (Chapter 11), "spatial autocorrelation" (used loosely for both global and local).

### Dimension 4 — Empirical and computational utility: **5.5 / 10**

- **Verified correct:** the row-standardization claim; the LISA identity; the Moran–regression-slope identity; $\operatorname{E}[I]$ and the normal-theory variance (simulated sd 0.0932 against formula 0.0932; 5.15% rejection at nominal 5%; power 98% at $\rho=0.65$); the SAR covariance and CAR precision; the Jacobian claim (without it $\hat\rho\approx0.94$ against true 0.65); the ICAR singularity for a connected map.
- **Verified wrong or incomplete:** Geary's range (values 2.5, 4.5, 10.5 on star graphs); the SAR "iff" statement; the ICAR rank for disconnected maps; the SMA range wording.
- **Promised but absent:** the permutation test and the SAR profile-likelihood estimator (L327–328).
- **Not shown:** nothing is fitted to data; no CAR or BYM computation; no residual diagnostics; no map.
- **Code optional, algorithms not.** The SAR estimator and the Moran permutation test are core and must be explained; the code can come second. CAR/BYM fitting needs at least a conceptual description of the sampling loop.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)

1. **Deliver or delete the promises (L327–328).** Add the SAR profile-likelihood recipe and a Moran permutation test, or change the description to what the code does.
2. **Geary's range (L122).** Replace "$c\in[0,2]$" with "near 1 means no autocorrelation; the upper limit depends on $\mathbf W$" (star-graph counterexample: $c=2.5,4.5,10.5$).
3. **SAR invertibility (L195–201).** Replace "invertible iff" with the true statement (invertible except at $\rho=1/\lambda_i$; the interval is the connected region around 0 and the series-convergence region). Say that the covariance is positive definite whenever the matrix is invertible.
4. **ICAR rank (L314) and zero-sum constraint (L316–319).** State "rank $n-c$ for $c$ connected components; constrain each component." Add the island case to §12.2.
5. **Bridge/model mismatch (L35–37 versus L176).** Name the spatial lag and spatial error models; say which one is developed and to which the OLS inconsistency applies.
6. **Put the Moran variance formula in the text (L112)** with its null stated (independent normal values, raw data, given $\mathbf W$), and say that for regression residuals the null changes.
7. **SMA range statement (L279).** Correct to "nonzero for neighbors and for regions sharing a neighbor; zero beyond."
8. **Notation slip (L242–248).** $\mathbf C=\phi\mathbf W^*$, and keep $\mathbf W$ (binary) distinct from $\mathbf W^*$.

### Should fix (pace, flow, voice)

9. **Use AR(1) as the Rosetta stone** for SAR vs. CAR (structural vs. two-sided conditional), then add the "when do I use which" comparison table.
10. **Add one toy map** (5–8 regions) reused throughout: weights matrix, Moran's $I$ by hand, SAR covariance, CAR precision, ICAR singularity.
11. **Add the Moran-as-regression-slope sentence** and connect the Moran scatterplot's fitted line to $I$ (verified identity).
12. **Explain the Jacobian** (change of variables) and quote the numbers (0.94 without, 0.60 with, true 0.65).
13. **Explain Brook's lemma through the two-region example;** drop Hammersley–Clifford from the heading or explain it.
14. **Motivate BYM with the small-area noise problem** (SMR variance $\propto1/E_i$), and show the shrinkage in a tiny example.
15. **Say how BYM is fitted** (conceptual MCMC loop or "use software"), and add the Poisson-regression-with-offset bridge.
16. **Add residual diagnostics:** fit a mean model, then test the residuals with Moran's $I$ (permutation test), which is what Moran's $I$ is for.
17. **LISA inference:** conditional permutation and a multiple-testing caveat, or soften "hotspot."
18. **Add Beat 3 to each section:** sensitivity to $\mathbf W$; Moran's $I$ detects but does not explain autocorrelation (trend vs. interaction); $\rho$ and $\phi$ are not comparable across models or $\mathbf W$s; $u$ vs. $v$ trade-off.
19. **Add an in-chapter worked-exercise section** (see §3).
20. **Add a one-sentence promise** at the top and a notation table for the chapter.
21. **Cross-reference** Chapters 2, 5, 6 (GLS and residual testing), 9 (link functions), 11 (LGCP as cousin), 13.

### Nice to have

22. A three-panel figure: a smooth map, a random map, a checkerboard map, with $I$ and $c$ for each, and their Moran scatterplots.
23. A sparse-matrix remark (real maps are sparse; CAR's sparse precision is why it scales).
24. Retitle §12.5 to "Disease Mapping with BYM" or broaden to one more GLMM.
25. Update the comment at L450 to match $\rho=0.65$ (truncation $1.4\times10^{-5}$); avoid `random.seed` on the global generator; label the white-noise experiment as a single draw.
26. Mention the modifiable areal unit problem and the sensitivity of results to $\mathbf W$.

---

## 6. What to Keep Exactly As It Is

- The AR(1)-to-lattice opening and the two ASCII pictures (L12–38).
- The row-standardization callout (L76–83).
- The Moran–Pearson framing and the LISA identity (L91–139).
- The SAR and CAR comparison box (L152–168) as the skeleton for the missing comparison table.
- The Jacobian paragraph (L208), quoted with the numbers above.
- The CAR precision anatomy box (L251–265) as the template for two more.
- The ICAR-as-random-walk intuition (L315).
- The Moran code, whose variance formula I verified (L385–437).
