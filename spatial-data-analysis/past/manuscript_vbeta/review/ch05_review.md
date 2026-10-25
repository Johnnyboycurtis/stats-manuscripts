# Review: Chapter 5 — Best Linear Unbiased Prediction (BLUP) Framework

**File reviewed:** `ch05_blup_kriging.md` (1,483 lines)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch05_exercises.md` was not reviewed. The four exercises inside §5.6 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy; I also ran copies of the chapter's kriging code). Literature attributions marked "from memory" are not verified.

---

## 1. Executive Summary

**Grade: 7 / 10**

This is the heart of the book and the chapter is mostly worthy of it. The opening bridge (bivariate-normal conditioning → simple kriging → ordinary kriging as "estimate the mean, then krige the residuals") is exactly the classical-to-spatial move the mission calls for, and the proofs are short, explicit, and correct: the projection theorem, the simple-kriging normal equations, the Gaussian-conditioning proof by "$W$ is independent of $\mathbf{Z}$", the Lagrange derivation of ordinary kriging, and the exact-interpolation theorem. It stays within the ceiling more cleanly than any earlier chapter. Its weaknesses are a set of confident statements that are wrong or only half true, a tangle of sign and symbol conventions around the Lagrange multiplier, three passages that retell the same story, and a notable gap between what the chapter proves and what a student needs to *use* kriging (nugget, diagnostics, honest uncertainty).

**Highest-value changes**
1. Fix the Lagrange-multiplier sign claim. Theorem 5.3.3(2) and its FAQ say the multipliers of the covariance and semivariogram systems have opposite signs. The chapter's own systems and variance formulas make them *equal*, and I confirmed that numerically (both give $0.045299$ on a random example). The "dissimilarity flips the sign" explanation (L743) is an invented story for a convention artifact.
2. Separate simple-kriging screening (exact) from ordinary-kriging screening (approximate). For an interior target with exponential covariance, simple kriging puts weight only on the two flanking points; ordinary kriging puts about 8% of its weight on points further out, because it also estimates the mean from all the data. Chapter sections 5.4–5.5 mix the two.
3. Add the three things that make kriging usable: what a nugget does (kriging stops reproducing the data), the leave-one-out shortcut (the spatial cousin of the regression PRESS residual), and what the kriging variance does and does not tell you.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: yes, the cleanest so far.** No measure theory, no Hilbert-space axioms (the $L^2$ name appears only as a parenthetical at L104, exactly as it should), no special functions. Small items:

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L3 | "functional-analytic and statistical foundations" | Low | Opening sentence promises more than the chapter (rightly) delivers. Say "geometric and statistical." |
| L892–900 | Screening proof leans on "the Ornstein–Uhlenbeck process possesses the Markov property," cited, not proved | Low–Medium | A two-line direct argument avoids stochastic-process language entirely (§5.4 below). |
| L1191 | Covariance tapering via "Wendland polynomial kernel" | Low | A pointer, not a derivation, which is right. One sentence of intuition for why the *product* of two valid covariances is valid (it is the Hadamard/Schur product of two positive semidefinite matrices) would make it linear-algebra-checkable. |
| L166, L383 | "in mean square," "almost surely" | Negligible | Fine at this level. |

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| Intro | Yes (excellent) | n/a | n/a |
| 5.1 | Yes | Procedure explained in five steps | Thin |
| 5.2 | Yes (but repeats the intro) | Yes (code text mismatch) | Good |
| 5.3 | Yes | Yes | Contains a false claim |
| 5.4 | Yes | **Mismatched** (describes neighborhood selection, which belongs in 5.5) | Overclaims |
| 5.5 | Yes | Yes (well explained) | Good |
| 5.6 | n/a | n/a | n/a |

### Intro (before §5.1)

**What works.** The three-part reading of the classical conditional-mean formula (L23–26: baseline, regression correction, variance reduction), the side-by-side "classical vs. spatial" diagram (L28–36), and "ordinary kriging is not an ad-hoc algorithm but the natural two-stage BLUP" (L66) are the right ideas in the right order. I verified that ordinary kriging equals "GLS estimate of the mean, then simple kriging on the residuals" numerically.

**Issues**
1. **The L5 dedication to Rizzo and S&G** should become what the student will be able to do after the chapter: build the kriging system by hand for a few points, explain why the weights can be negative, and say what the kriging variance means.
2. **The bridge is told three times.** The bivariate-normal story appears in the intro (L11–26), again almost verbatim in §5.2.1 (L231–243), and the "unknown mean" story appears in the intro (L55–66), in §5.3.1 (L510), and again two paragraphs later in §5.3.1 (L514). Keep it once, in the intro; in §5.2 and §5.3 begin from "now we make it precise."

### §5.1 Vector Space Geometry of Random Variables

**What works.** The OLS-column-space bridge (L76) and the five-item geometry list (L85–102) are the right way to introduce the geometry. Theorem 5.1.1's proofs of sufficiency and necessity are elementary and clear (one line each). The Pythagorean identity gives a good interpretation: a predictor can never be more variable than the thing it predicts (simple case).

**Issues**
1. **Duplication with §5.2.** Theorem 5.1.1(1) and Theorem 5.2.1's proof derive the same expansion $\operatorname{Var}(X) - 2\boldsymbol\lambda'\mathbf{v}_0 + \boldsymbol\lambda'\mathbf{V}\boldsymbol\lambda$ and the same normal equations. Let §5.2 cite §5.1 and add only what is new (the known mean, $k_0$).
2. **Corollary 5.1.2 is introduced and never used.** (I searched the chapter.) Ordinary kriging is derived by Lagrange multipliers instead. Either use it (the set of unbiased predictors is an affine space, so the projection picture applies to OK too) or cut it.
3. **The diagram (L195–209)** tries to draw the projection and is hard to read: the right angle is not visible. A plain right triangle, with the hypotenuse labeled "$Z(\mathbf{s}_0) - \mu$," one leg "$\widehat{Z}_{\text{SK}} - \mu$" lying along the subspace, and the other leg "error," says it in five lines.
4. **Beat 3, singular $\mathbf{V}$ (L221).** Correct that the projection stays unique while the weights are not. But "adding a nugget restores invertibility" quietly *changes the problem*: with a nugget, observations at the same site differ by independent noise, and the predictor is no longer exactly the old one. Say so, and point to the nugget discussion that is missing from the chapter (below).
5. **Beat 3 is thin.** It says nothing a practitioner needs: how to compute the weights (never invert $\mathbf{V}$, solve), or what "orthogonal" buys (the error is uncorrelated with every datum, so there is no leftover linear signal).

### §5.2 Simple Kriging

**What works.** The anatomy (L326–334), the "shrinkage toward the prior mean" Beat 3 (L488–495), and Theorem 5.2.2 are good. The proof of Theorem 5.2.2 (define $W = Z(\mathbf{s}_0) - \mathbf{v}_0'\mathbf{V}^{-1}\mathbf{Z}$, show it is uncorrelated with, hence independent of, $\mathbf{Z}$) is a model proof for this audience: pure linear model, no conditional-density calculus. I verified the chapter's simple-kriging code against a direct linear solve.

**Issues**
1. **Notation drift.** $\mathbf{v}_0$ and $\mathbf{c}_0$ are used for the same vector ("also denoted $\mathbf{v}_0$," L331). The intro uses $\mathbf{c}_0$, §5.1–5.3 use $\mathbf{v}_0$. Pick one.
2. **Definition 5.2.1** says "second-order stationary ... with known mean function $\mu(\mathbf{s})$." Stationarity of $Z$ requires a constant mean. The intended meaning is "the residual field is stationary." (Same issue as Chapter 1.)
3. **Beat 2 text and code disagree.** The text says the system is solved "via Cholesky decomposition" (L389); the code does Gauss–Jordan elimination with partial pivoting (L443–470). Either is fine; say which.
4. **The nugget in the SK code (L394, L410–411, L433, L479).** The docstring calls `nugget` "microscale variance added to the diagonal for regularization." That conflates two different things:
   - a tiny *numerical jitter* added only to keep the solve stable, and
   - the model's nugget $\tau^2$, which changes what is being predicted.
   I checked what happens: with three points, exponential covariance, and a nugget of $0.3$ added to the diagonal, predicting at an observed site gives weights $(0.069, 0.730, 0.069)$ instead of $(0, 1, 0)$, a prediction of $-0.247$ against an observed datum of $-0.5$, and an error variance of $0.219$ for the latent surface but $0.519$ for a new noisy measurement (the code returns the latter, since it adds the nugget to $C(\mathbf{0})$). A student who sets `nugget` to "stabilize" the solve will silently change the answer and the meaning of the variance. Separate the two ideas, and add the missing section (below).
5. **`max(0.0, ...)` on the variance (L481, L842, L1181)** silently hides a negative kriging variance, which signals an invalid covariance model (Chapter 2's theme). A student should be told: if this clamp ever triggers, stop and check the model.
6. **Beat 3 is good but could add one practical item:** the leave-one-out shortcut (see Dimension 4).

### §5.3 Ordinary Kriging

**What works.** The OLS-intercept bridge (L506–512) and the bias formula for a wrong guessed mean, $(1 - \sum\lambda_i)(\mu^* - \mu)$ (L510), are both right; I checked the latter. The Lagrange derivation (L554–590) is explicit. The exact-interpolation proof (L642–660) is a model of "verify the solution fits the equations."

**Issues**
1. **The sign and symbol of the Lagrange multiplier is a four-way tangle.** The chapter uses: $-\mu_{\text{OK}}$ in the system vector (Theorem 5.3.1, L538); a multiplier called plain $\mu$ with "$\mu \equiv -\mu_{\text{OK}}$" (the anatomy blocks, L592–621); $\mu_{\text{OK}}^{(\gamma)}$ (Theorem 5.3.3); and $\mu_{\text{var}}$, $\mu_{\text{cov}}$ (FAQ, L734–743). The multiplier called $\mu$ in the anatomy is also the *same letter as the field's mean*. A student cannot follow this. Pick one name (say $m$) and one sign convention, and say it once, in a box.
2. **Theorem 5.3.3(2) and the FAQ are wrong about the sign.** The proof's own algebra (L714–720) arrives at $\mathbf{V}\boldsymbol\lambda - \mu_{\text{OK}}^{(\gamma)}\mathbf{1} = \mathbf{v}_0$ and then compares it with $\mathbf{V}\boldsymbol\lambda - \mu_{\text{OK}}^{(C)}\mathbf{1} = \mathbf{v}_0$, which gives *equality*, not $\mu^{(\gamma)} = -\mu^{(C)}$. The two variance formulas ($C(\mathbf{0}) - \boldsymbol\lambda'\mathbf{v}_0 + \mu_{\text{OK}}$ and $\boldsymbol\lambda'\boldsymbol\gamma_0 + \mu_{\text{OK}}^{(\gamma)}$) force the same conclusion, since $\boldsymbol\lambda'\boldsymbol\gamma_0 = C(\mathbf{0}) - \boldsymbol\lambda'\mathbf{v}_0$ when the weights sum to 1. Checked numerically on a random 7-point example: $\mu_{\text{OK}} = 0.045299$ from the covariance system and $\mu^{(\gamma)} = 0.045299$ from the chapter's variogram code. The chapter's convention happens to be the common one in which the same number appears in both systems. The FAQ's story (L743: variograms measure "dissimilarity," so "the sign of the cross-terms is flipped") is not a reason: the sign only reflects whether you write the constraint term as $\pm 2\mu(\boldsymbol\lambda'\mathbf{1} - 1)$ in the Lagrangian. Replace the FAQ with: "The multiplier's sign is a bookkeeping choice. We use the convention under which the same number appears in both systems."
   (I had reached a different conclusion about this in my Chapter 2 review; I have corrected that note. The sign depends on convention.)
3. **Uniqueness is asserted, not shown (Theorem 5.3.1; L651).** The bordered matrix $\begin{bmatrix}\mathbf{V} & \mathbf{1}\\ \mathbf{1}' & 0\end{bmatrix}$ is invertible because its Schur complement is $-\mathbf{1}'\mathbf{V}^{-1}\mathbf{1} \neq 0$. One line, and it is the first place a student learns what "bordered" means. Also, for the semivariogram system (Theorem 5.3.3), the stationary point is a *minimum* because $-\boldsymbol\Gamma$ is positive semidefinite on the zero-sum subspace (Chapter 2); say so.
4. **A very good fact is stated but not proved (L621).** "Ordinary kriging variance exceeds simple kriging variance by $(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2/(\mathbf{1}'\mathbf{V}^{-1}\mathbf{1})$." I verified it (both sides $= 0.66694$ on my example). The proof is three lines and makes the two-stage idea concrete: $\widehat{Z}_{\text{OK}} - \widehat{Z}_{\text{SK}} = (\widehat\mu - \mu)(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)$; the SK error is uncorrelated with every datum, hence with $\widehat\mu$; and $\operatorname{Var}(\widehat\mu_{\text{GLS}}) = 1/(\mathbf{1}'\mathbf{V}^{-1}\mathbf{1})$ from the GLS chapter. So the extra variance is "the cost of not knowing the mean."
5. **Beat 3, "variance deficit" (L853–857) is stated more strongly than it holds.** The identity $\operatorname{Var}(\widehat{Z}_{\text{OK}}) = C(\mathbf{0}) - \sigma_{\text{OK}}^2 + 2\mu_{\text{OK}}$ is right (verified). But the next sentence ("because $\sigma_{\text{OK}}^2 > 0$, the variance of the predicted surface is systematically smaller than $C(\mathbf{0})$") does not follow: the extra term $2\mu_{\text{OK}}$ can be positive. For *simple* kriging the claim is always true; for ordinary kriging in extrapolation with a smooth kernel it can fail. Example I checked: Gaussian covariance $e^{-h^2}$, data at $0$ and $0.2$, target at $0.5$. The weights are $(-1.223, 2.223)$ and $\operatorname{Var}(\widehat{Z}_{\text{OK}}) = 1.213 > 1 = C(\mathbf{0})$ (at target $1.0$: $1.304$). Say "typically smaller in interpolation; can exceed $C(\mathbf{0})$ when extrapolating."
6. **Exact interpolation holds only without a nugget (Theorem 5.3.2).** Say so; link to the nugget section (below).
7. **Beat 2 code.** Correct and clear (I verified it against a direct solve). Add to the docstring that `variogram_kernel(0)` must return $0$ (the diagonal of $\boldsymbol\Gamma$) and that a nugget enters as $\gamma(0^+) = \tau^2$ for $h > 0$ only. As written, the code does not say how a nugget should enter; a kernel that returns $\tau^2$ rather than $0$ at $h = 0$ changes the meaning of the diagonal of $\boldsymbol\Gamma$, and the result is not the usual nugget model (I did not test what it does).

### §5.4 Geometric Properties of Kriging Weights and Spatial Screening

**What works.** The AR(1) bridge to screening (L866) is a good classical anchor. Exercise 5.1 (see below) is the cleanest proof of screening in the chapter. The tensor-product theorem is nice linear algebra: the Kronecker structure makes the 2D weights a product of 1D weights.

**Issues**
1. **Simple kriging versus ordinary kriging.** Theorem 5.4.1 is correctly stated for *simple* kriging. Elsewhere the chapter speaks of "the screening effect" as if it applied to everything, and §5.5 says that "thanks to the screening effect, the loss from discarding distant observations is negligible." For ordinary kriging, screening is approximate. I checked six points $0, \dots, 5$, exponential covariance, target $2.5$: simple kriging weights are $(0, 0, 0.4434, 0.4434, 0, 0)$; ordinary kriging weights are $(0.025, 0.0158, 0.4592, 0.4592, 0.0158, 0.025)$, so about 8% of the weight goes to non-flanking points. The reason is the one the chapter itself teaches: OK estimates the mean from *all* the data. Exercise 5.3(b) (extrapolation) shows the same thing, with weights $(0.240, 0.152, 0.608)$ on three points where simple kriging would put everything on the nearest. Add one paragraph: "screening is exact for SK, approximate for OK."
2. **The proof of Theorem 5.4.1 uses a heavy tool when a light one is available.** It cites that the Ornstein–Uhlenbeck process is Markov. Exercise 5.1 already shows the direct argument for three points: multiply the second normal equation by $b$ and subtract it from the third. This works for *any* number of outer points, because the exponential covariance factorizes, $C(a + b) = C(a)C(b)/\sigma^2$, so each outer row of the system is a multiple of the flanking row. Gaussianity is not needed (kriging only uses covariances), but the theorem says "Gaussian." Replace the proof with this algebra and drop "Gaussian."
3. **Beat 2 is about the wrong topic.** The section's theorems are about screening and tensor grids; Beat 2 presents a quadrant-search neighborhood selector, which is the subject of §5.5. Move it there. Note also that §5.5's local kriging function uses plain nearest neighbors and never calls the quadrant selector, so the quadrant code is defined but not used.
4. **Theorem 5.4.2 requires a complete rectangular grid and a *separable* kernel (a product of one-dimensional exponentials).** Isotropic exponential $e^{-\|\mathbf{h}\|}$ is not separable, and in 2D it does not give exact four-corner screening. Say so; otherwise a student will expect four-corner screening from the usual isotropic model. Consider whether this result earns its space in an MS-level chapter; the Kronecker proof is good, but the situation is rare in practice.
5. **The negative-weights discussion (L1037–1048) needs work.**
   - *Duplication:* the "derivative estimation" story is told twice (L1041 and the FAQ at L1048).
   - *Label:* "Curvature Matching ... (Poisson Analogy)" (L1042): I could not find anything Poisson about it, and the second-difference stencil $(1, -2, 1)$ has weights summing to $0$, while prediction weights sum to $1$. Negative weights in extrapolation arise mainly from extrapolating a *slope* (first difference), not curvature.
   - *Evidence:* nothing in the section shows a negative weight arising. Here is a two-line example I checked. Target at $0$, data at $1$ and $2$. Exponential covariance: simple kriging weights are $(0.3679, 0)$ (the farther point is screened). Gaussian covariance $e^{-h^2}$: weights $(0.4177, -0.1353)$. Same geometry, only the smoothness changes. That belongs in the main text.
   - *Overclaim:* "Negative weights ... preserve physical reality by declustering" (L1048) is not right. Negative weights can produce predictions outside the range of the data and even negative predictions for quantities that cannot be negative (concentrations, grades). Say that, and say that it motivates positivity-preserving transformations or constraints (not covered here).
   - *Overclaim:* "Exponential kernels exhibit minimal or no negative weights" holds exactly in 1D (Theorem 5.4.1) but not in general 2D configurations. The "$\nu \ge 2$" threshold has no support.
6. **The 3-weight example in Exercise 5.3(c)** is the right kind of illustration but is mis-described (see Dimension 4).

### §5.5 Local vs. Global Neighborhood Kriging

**What works.** The "rolling window" bridge (L1057) is a good classical anchor. Beat 2's explanation (find the $n_0$ nearest neighbors, build the local $(n_0 + 1)$ system, solve) is clear and the code is faithful. Beat 3 on directional shadowing and covariance tapering is practical and accurate (Furrer, Genton, and Nychka 2006 is the right reference, from memory). In a small test (30 random points, target at the center, exponential-type variogram) the kriging variance was $0.2298$, $0.2293$, and $0.2292$ for $n_0 = 8, 16, 30$: local is within 0.3% of global, which supports the "loss is small" message.

**Issues**
1. **Proposition 5.5.2's condition is wrong, and it has no proof.** It says a neighbor's weight jumps from $0$ to a non-zero value "unless $C(\|\mathbf{s}_i - \mathbf{s}_0\|) = 0$ at the boundary." A counterexample: simple kriging, exponential covariance, 1D, two flanking points at $1$ and $2$, target $1.5$. Adding an outer point at $3$ (where $C(1.5) = 0.223 \ne 0$) leaves the weights exactly $(0.4434, 0.4434, 0)$: no jump at all, because of screening. Conversely, a point with zero covariance with the target can still get a non-zero weight if it is correlated with other neighbors. The honest statement: *generically* the weights change when the neighborhood changes, by an amount that is small when screening is strong. Prove it or state it as an observation.
2. **Proposition 5.5.1** (local is never better than global) is correct and elegantly stated (nested subspaces). Add that this holds for simple kriging directly and for ordinary kriging because the constraint set is also nested.
3. **Beat 3 numbers.** "$n = 50{,}000$ ... an impossibility in standard RAM" (L1188): $2.5\times10^9$ doubles is about 20 GB, which fits on a large workstation. Say "impractical on a laptop and costly even on a large machine."
4. **The covariance-tapering item** says "$O(n^{3/2})$ scaling." This is the typical figure for sparse Cholesky on 2D problems with a good ordering (from memory); qualify it as typical.

### §5.6 Analytical Exercises

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **7.5 / 10**
- All the sections open with a real classical bridge, and 5.1–5.3 are paced well: geometry, then the known-mean case, then the unknown-mean case. This is the best-organized sequence in the book so far.
- Pace slows when the bridge is retold (intro, §5.2.1, §5.3.1) and in §5.4, where the two-dimensional tensor theorem and the neighborhood code interrupt the screening-and-negative-weights story.
- **Missing signposts and examples:** "Chapter at a glance," a "Key ideas" recap, and a single *worked numeric example* (four or five points on a line, solve the SK and OK systems by hand, compare the weights, and compute both variances). The exercises have pieces of this, but a student reading the chapter body sees only symbols.
- **Visible scaffolding:** "Beat 1/2/3" headings, very long bridge titles (e.g., L9).

### Dimension 2 — Algebraic transparency and scaffolding: **7.5 / 10**
- Strengths: Theorem 5.1.1, Theorem 5.2.1, Theorem 5.2.2's independence proof, Theorem 5.3.1's Lagrangian, Theorem 5.3.2, and the algebra in all four exercises are written out.
- Gaps: invertibility of the bordered matrix; the OK-versus-SK variance formula (stated, not proved); the unproved Markov property in Theorem 5.4.1; Proposition 5.5.2 has no proof at all.
- **"From the Lecture Hall" callouts: none present.** Natural ones: (1) "Kriging is regression with the data as predictors and the covariances as the design"; (2) "What is a Lagrange multiplier? The price of the sum-to-one constraint"; (3) "Why does a smoother kernel invite negative weights?" (the two-point example above); (4) "Reading a kriging variance: what it covers and what it leaves out."

### Dimension 3 — Structural consistency and forward referencing: **6 / 10**
- **Duplication:** the bivariate bridge (intro and §5.2.1); the SK mean-squared-error derivation (§5.1 and §5.2); the "unknown mean" paragraph (§5.3.1 twice); the negative-weights explanation (twice in §5.4).
- **Notation:** $\mathbf{v}_0$ vs. $\mathbf{c}_0$; four conventions for the Lagrange multiplier, one of which reuses the symbol for the mean; $\sigma^2$ vs. $C(\mathbf{0})$.
- **Unused material:** Corollary 5.1.2; the quadrant-search function.
- **Overlap with Chapter 10:** the variance-deficit idea is introduced here (good) and its consequence for thresholds is deferred (good). Keep it short.

### Dimension 4 — Empirical and computational utility: **6.5 / 10**
- **Exercise 5.1** (screening in 1D): correct and the best exercise in the chapter. Its subtraction trick *is* the proof of Theorem 5.4.1 for any number of points. Move its idea into the theorem.
- **Exercise 5.2** (two-point OK in variogram form): correct. I checked weights, multiplier, and the variance formula, $\sigma_{\text{OK}}^2 = \gamma_{01} + \gamma_{02} - \gamma_{12}/2 - (\gamma_{01} - \gamma_{02})^2/(2\gamma_{12})$, and the midpoint case $\theta h^\alpha(2^{1-\alpha} - \tfrac12)$. In (d) the "$\alpha \to 2$" limit gives zero variance, which is right for the degenerate linear-trend case; remind the student that the power variogram is a *valid* model only up to $\alpha = 2$ (see the Chapter 2 review) and that $\alpha = 2$ is the boundary.
- **Exercise 5.3** (negative weights): parts (a) and (b) are correct (I verified $\lambda = (0.2402, 0.1518, 0.6080)$). In (c) the numbers are right ($\lambda = (0.3285, -0.0316, 0.7031)$, $\mu_{\text{OK}} = 0.3296$) but the *explanation* contradicts them. The text says the negative weight on $s_2$ "combined with $\lambda_3 > 1$" gives a forward-difference extrapolation $(1 + \alpha)Z(s_3) - \alpha Z(s_2)$. But $\lambda_3 = 0.7031 < 1$ and the farthest point $s_1$ keeps $0.3285$, so this is not the described pattern. Either choose a configuration that does show $\lambda_3 > 1$ (the two-point Gaussian example above does) or describe what the numbers do show: with a smooth kernel, ordinary kriging gives a small negative weight to the middle point, whereas the exponential gives all-positive weights. Also (b)'s "all weights are positive" is true but the pattern is *not* screening, which deserves a sentence (see §5.4 issue 1).
- **Exercise 5.4** (variance deficit on a lattice): correct (I checked the formula numerically). It is the only worked demonstration of the variance deficit; consider promoting it into §5.3 Beat 3. Note it uses the isotropic exponential with four neighbors, not the separable kernel of Theorem 5.4.2.
- **Missing from the exercises and the chapter:**
  - *Leave-one-out.* For simple kriging, the leave-one-out residual at site $i$ is $[\mathbf{V}^{-1}(\mathbf{Z} - \boldsymbol\mu)]_i / [\mathbf{V}^{-1}]_{ii}$ and its error variance is $1/[\mathbf{V}^{-1}]_{ii}$. I verified this exactly against recomputing from scratch. It is the spatial analogue of the PRESS residual $e_i/(1 - h_{ii})$ that students know from regression, it needs one matrix inverse and no refitting, and it gives the most useful diagnostic in kriging (standardized residuals should look like $\mathcal{N}(0,1)$). It belongs in Beat 3 of §5.2.
  - *Kriging with a nugget (smoothing, not interpolation).* One short subsection: with $\tau^2 > 0$ the predictor no longer reproduces the data; the weights at a data site spread over neighbors (my example: $0.069$, $0.730$, $0.069$); there are two targets (the latent surface, with error variance $\sigma^2 - \mathbf{c}'\mathbf{V}^{-1}\mathbf{c}$, and a new noisy observation, which adds $\tau^2$); the exact-interpolation theorem holds only without a nugget.
  - *What the kriging variance leaves out.* It assumes the covariance parameters are known. Plugging in estimates (Chapter 4) ignores their uncertainty, so intervals $\widehat{Z} \pm 1.96\,\sigma_{\text{OK}}$ are somewhat too narrow, and are valid as stated only for Gaussian fields. One paragraph.
  - *A numerical worked example,* including a small picture of weights versus distance.
- **Chapter ending:** no "Key ideas" or "what comes next" (Chapter 6 adds trends).

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Fix Theorem 5.3.3(2) and its FAQ:** with the chapter's own systems the two multipliers are equal; replace the invented explanation with "the sign is a convention."
2. **Fix the OK variance-deficit claim** (L853–857): $\operatorname{Var}(\widehat{Z}_{\text{OK}})$ can exceed $C(\mathbf{0})$ when extrapolating (example above).
3. **Distinguish SK screening (exact) from OK screening (approximate)** in §5.4–5.5 and Exercise 5.3(b).
4. **Fix Proposition 5.5.2** (wrong condition; no proof). Counterexample: SK with exponential covariance.
5. **Fix Exercise 5.3(c)'s explanation** ($\lambda_3 = 0.703 < 1$, not $> 1$).
6. **Resolve the multiplier notation** (one name, one sign, a box that says it; stop using $\mu$ for the multiplier).
7. **Fix "negative weights preserve physical reality"** (L1048) and add the consequence (predictions outside the data range or negative).
8. **Separate numerical jitter from the model nugget** in the SK code and docstring, and add the missing nugget subsection.

### Should fix (pace, flow, voice)
9. **Add a worked numeric example** (a handful of points on a line: solve SK and OK by hand, compare weights and variances).
10. **Add the leave-one-out identity** to §5.2 Beat 3 (verified).
11. **Prove the OK-vs-SK variance formula** (three lines; the two-stage idea made concrete).
12. **Replace the Markov-property proof of Theorem 5.4.1** with the Exercise 5.1 factorization argument; drop "Gaussian."
13. **Show the bordered matrix is invertible** (Schur complement); note the minimum for the variogram system.
14. **Move the quadrant-search code to §5.5** (or call it from the local kriging function).
15. **Add the plug-in caveat** about kriging variance and prediction intervals.
16. **De-duplicate** the bridge, the SK derivation, the "unknown mean" paragraphs, and the negative-weights explanation.
17. **Show a negative weight arising** with the two-point example (exponential vs. Gaussian).
18. **Add four "From the Lecture Hall" callouts** and "Chapter at a glance" / "Key ideas."
19. **Replace "Beat" headings;** replace the L5 dedication.

### Nice to have
20. Fix the Beat 2 text/code mismatch (Cholesky vs. Gauss–Jordan).
21. Clarify the SK definition ("residual field is stationary").
22. Use or remove Corollary 5.1.2.
23. Redraw the projection diagram as a right triangle.
24. Qualify the 2D separable-kernel scope of Theorem 5.4.2, and the $O(n^{3/2})$ and "RAM" statements.
25. Note that a clamped negative variance signals an invalid model.
26. Add a sentence on why a product of covariances is a covariance (Schur product) in the tapering item.
27. Soften "$\nu \ge 2$" and "no negative weights for exponential."
28. Verify the Furrer–Genton–Nychka (2006) reference and the sparse-Cholesky scaling claim.

---

## 6. What to Keep Exactly As It Is

- The three-part reading of the classical conditional-mean formula and the classical-vs-spatial diagram (L23–36).
- "Ordinary kriging is the natural two-stage BLUP" (L66).
- The OLS column-space bridge (L76) and the proofs of Theorem 5.1.1.
- The independence proof of Theorem 5.2.2 ($W \perp \mathbf{Z}$).
- The "shrinkage toward the prior mean" Beat 3 for simple kriging (L488–495).
- The unknown-intercept bridge and the bias formula for a wrong mean (L506–512).
- The Lagrangian derivation and the exact-interpolation proof (L554–590, L642–660).
- The "rolling window" bridge and the directional-shadowing and tapering notes in §5.5.
- Exercises 5.1, 5.2, and 5.4, and the numbers in 5.3.
