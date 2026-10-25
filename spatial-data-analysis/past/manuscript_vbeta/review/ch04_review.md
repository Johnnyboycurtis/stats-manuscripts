# Review: Chapter 4 — Covariance Parameter Estimation and Theoretical Validations

**File reviewed:** `ch04_parameter_estimation.md` (1,341 lines)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch04_exercises.md` was not reviewed. The four exercises inside §4.7 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy). Literature attributions marked "from memory" are not verified.

---

## 1. Executive Summary

**Grade: 6.5 / 10**

Chapter 4's best idea is also the book's best linear-models bridge so far: the classical $(n-1)$ correction, to $(n-p)/n$, to error contrasts, to the projection matrix $\mathbf{P}$, which plays the role the hat-matrix complement $\mathbf{I} - \mathbf{H}$ plays in ordinary regression. The structure is clean and the algebra is mostly explicit. The weaknesses are of three kinds. First, several statements are wrong (an exercise's MSE formula and its conclusion, an "if and only if" for the coregionalization model, a reversed claim about pair counts, and reversed intuitions about both log-determinant terms). Second, the chapter is about *estimating* parameters but never shows the estimation workflow: Beat 2 gives objective-function evaluators and a hand-written optimizer, but no explanation of how the pieces fit together, no standard errors, and no warning about the likelihood surface's well-known trouble spots. Third, Theorem 4.4.1 and the microergodicity FAQ lean on advanced asymptotic and measure-theoretic ideas that the mission says to avoid.

**Highest-value changes**
1. Say the one sentence that makes the whole chapter click: *whiten with the Cholesky factor, then run ordinary least squares.* The Beat 2 code does exactly this, but the text never says it. I checked that $\mathbf{Z}'\mathbf{P}\mathbf{Z}$ equals the residual sum of squares of that whitened regression, and that the two forms of $\mathbf{P}$ agree. A student who sees "GLS is OLS on whitened data" can derive most of §4.2–4.3 without help.
2. Fix Exercise 4.2(c). The stated MSE for ML is wrong, and so is the conclusion. With the correct formula, REML has the *smaller* MSE once $p$ is moderately large; for $n = 50$, $p = 6$ (the chapter's own motivating numbers) ML's MSE is $0.0496$ against REML's $0.0455$.
3. Add the missing workflow: starting values from the variogram fit, maximize REML over the range and nugget-to-sill ratio (profiling out $\sigma^2$ in closed form), look at standard errors and the profile, and know when the answer is not to be trusted.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes, with three areas of drift.** Sections 4.1–4.3 are exactly at the ceiling (traces, projections, Cholesky). The drift is in 4.4–4.5:

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L756–759 | FAQ on microergodicity: "two Gaussian measures ... are mutually absolutely continuous (equivalent) if and only if" the ratio is the same, "likelihood ratio tends to a non-degenerate random variable" | Medium | This is the Feldman–Hájek theorem in everyday clothing, and it is the most measure-theoretic passage in the book so far. Replace with the Taylor-expansion explanation from the Chapter 1 review: $\gamma(h) \approx (\sigma^2/\phi)\,h$ near the origin, so dense data pin down the slope $\sigma^2/\phi$ and nothing else. Then say what students will *see* (an elongated likelihood ridge). Keep the $d \le 3$ caveat the FAQ correctly includes. |
| L743–754 | Theorem 4.4.1: almost-sure consistency, asymptotic normality with a sandwich covariance, and a Loewner ordering of REML/GLS/WLS covariances, all without proof or intuition | Medium | A black-box theorem. Either cite it (from memory, Lahiri, Lee and Cressie, 2002, on least-squares variogram estimators; please verify) with a plain statement, or replace with the classical analogy students know: *Gauss–Markov says a better weight matrix gives a smaller covariance* ($\operatorname{Var}(\widehat\beta_{\text{OLS}}) - \operatorname{Var}(\widehat\beta_{\text{GLS}}) \succeq 0$), and WLS is GLS with a rough, diagonal weight matrix. |
| L877–909 | Cramér's theorem with complex Hermitian matrices, and a necessity proof that assumes what it proves | Medium | Treat it as Chapter 2 treated Bochner: prove the easy (sufficiency) direction, cite the other, and note that for real, symmetric cross-covariances the spectral matrix is real symmetric. See §4.5 below. |
| L165–169 | Jacobi's formula "taking differentials" | Low | A short Taylor argument suffices: $\det(\mathbf{V} + \epsilon\mathbf{E}) = \det(\mathbf{V})\det(\mathbf{I} + \epsilon\mathbf{V}^{-1}\mathbf{E}) = \det(\mathbf{V})(1 + \epsilon\operatorname{Tr}(\mathbf{V}^{-1}\mathbf{E}) + O(\epsilon^2))$, using $\prod(1 + \epsilon\lambda_i) = 1 + \epsilon\sum\lambda_i + O(\epsilon^2)$. |
| L236 | "Gaussian quadratic form covariance theorem" $\operatorname{Cov}(\boldsymbol\varepsilon'\mathbf{A}\boldsymbol\varepsilon, \boldsymbol\varepsilon'\mathbf{B}\boldsymbol\varepsilon) = 2\operatorname{Tr}(\mathbf{A}\mathbf{V}\mathbf{B}\mathbf{V})$ used without proof | Low–Medium | Avoid it. Exercise 4.1's Hessian route proves the same Fisher information using only $\operatorname{E}[\mathbf{Z}'\mathbf{M}\mathbf{Z}] = \operatorname{Tr}(\mathbf{M}\mathbf{V})$. Put that argument in the theorem. |
| L89–96 | Composite likelihood with consistency and normality claimed | Low | A pointer is fine; it is a tangent (see §Intro below). |

No other measure theory, no Bessel integrals, no functional analysis. The determinant manipulations, trace identities, and Cholesky computations are all at the right level.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

Beat 2 is read as "how you would compute it." For a section with a core algorithm, that means a prose explanation of the steps; code is a bonus.

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| Intro | Yes (excellent) | n/a | n/a |
| 4.1 | Yes | Cholesky route explained; code relies on undefined helpers | **Thin** (repeats earlier nugget material) |
| 4.2 | Yes | **Code only**, near-duplicate of 4.3's | **Thin** |
| 4.3 | Yes | **Code only** | Mixed (good FAQ; garbled explanation) |
| 4.4 | Yes | **Code only** (an optimizer, no explanation) | Contains factual errors |
| 4.5 | Yes | Explained in one line; finite-grid limitation unsaid | One example that does not match its cross-reference |
| 4.6 | Yes | Code only | Mixed |
| 4.7 | n/a | n/a | n/a |

### Intro (before §4.1)

**What works**
- The progression $(n-1) \to (n-p)/n \to$ error contrasts $\to \mathbf{P}$ (L15–70) is the best classical-to-spatial bridge in the book. The two-line diagram (L23–29) is the right kind of picture.
- The "10% to 20% for $n = 50$ to $100$ and $p = 6$ to $10$" motivation (L39) is concrete.

**Issues**
1. **The $(n-p)/n$ factor is exact only when the covariance is a known correlation matrix times a single scale.** The text (L35–39) states it as a general fact about "the covariance sill." Proposition 4.2.1 (L393–402) correctly restricts to $\mathbf{V} = \sigma^2\mathbf{R}(\boldsymbol\phi)$ with $\boldsymbol\phi$ fixed. When the range and nugget are estimated at the same time, the bias of the sill is not $(n-p)/n$. I ran a quick simulation to see how far off it can be (one scenario, 120 simulated datasets: $n = 50$ random sites in the unit square, a cubic trend surface so $p = 10$, exponential covariance with range $0.3$ and sill $1$, nugget fixed at $0$, range estimated). The *median* sill estimate was $0.26$ for ML and $0.88$ for REML, against the "10–20%" in the text. REML's *mean* was dominated by a few runs where the range ran off to a huge value and the sill with it (the $\sigma^2$–$\phi$ ridge, §4.4 FAQ). This is one scenario, but it shows the (n−p)/n story is the right *mechanism* and the wrong *magnitude* in the joint-estimation case. Say so, and add the scale-only caveat to the intro.
2. **"One can show ... (see Theorem 4.2.1 and Theorem 4.3.3)" (L67) is a gap and a forbidden phrase.** The claim is that $\mathbf{A}(\mathbf{A}'\mathbf{V}\mathbf{A})^{-1}\mathbf{A}' = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}$. I verified numerically that it is true, but neither cited theorem proves it (Theorem 4.2.1 only studies the closed form). A three-line whitening proof fits the ceiling: with $\mathbf{V} = \mathbf{L}\mathbf{L}'$, set $\tilde{\mathbf{X}} = \mathbf{L}^{-1}\mathbf{X}$ and $\tilde{\mathbf{A}} = \mathbf{L}'\mathbf{A}$. Then $\tilde{\mathbf{X}}'\tilde{\mathbf{A}} = \mathbf{X}'\mathbf{A} = \mathbf{0}$, so the columns of $\tilde{\mathbf{A}}$ span the orthogonal complement of $\operatorname{col}(\tilde{\mathbf{X}})$. Both sides of the equation are then $\mathbf{L}^{-\prime}$ times "the projection onto that complement" times $\mathbf{L}^{-1}$, hence equal.
3. **Composite likelihood (L89–96) is a tangent in an intro that is already carrying a lot.** It is never used again. The claim "strongly consistent and asymptotically normal under increasing-domain asymptotics" is stated without a source. Either move it to the end as a short "scaling up" box (where it can point to Chapter 13), or drop it.
4. **Notation changes mid-chapter.** The intro uses $\mathbf{A}$ for the contrast matrix and $\mathbf{U}$ for the contrasts; §4.3 uses $\mathbf{K}$ and $\mathbf{W}$; Theorem 4.1.2's proof uses $\mathbf{A}_k$ for something else; Chapter 3 used $\mathbf{A}(\mathbf{h})$ for Genton's matrix. The data vector is $\mathbf{Y}$ in some places and $\mathbf{Z}$ in others (L33, L135, L573). Pick one set.
5. **Duplication.** The "Fundamental Invariance Properties of $\mathbf{P}$" proposition (L72–87) and Theorem 4.2.1 (L371–391) have essentially the same proof. The REML idea (contrasts, $\beta$-free distribution) is introduced in the intro (L41–60), again in §4.3.1 (L499–503), and again in Definition 4.3.1 and Theorem 4.3.1. State it once in the intro, then in §4.3 start from "now let us make this precise."

### §4.1 The Spatial Linear Model and Gaussian Likelihood

**What works.** The Mahalanobis-plus-determinant picture (L111–112) is a good bridge from the OLS log-likelihood. Theorem 4.1.1's derivation is explicit and correct. Exercise 4.1 is a nice second route to the Fisher information.

**Issues**
1. **The determinant intuition is backwards (L112, L145).** The text says the log-determinant "penalizes models that assume artificially broad correlation ranges" and "grows large if variance or range are inflated." I checked: for a fixed sill, $\ln|\mathbf{V}|$ gets *smaller* as the range grows. On 30 random points with an exponential correlation, $\ln|\mathbf{R}|$ is $-2.4, -7.0, -16.3, -35.6, -53.3$ for ranges $0.5, 1, 2, 5, 10$. A longer range makes the matrix more nearly singular, so the determinant shrinks and the likelihood *rises*; the Mahalanobis term is what pushes back. Rewrite: "The determinant rewards strong correlation (the data are more compressible); the quadratic form punishes a model whose correlation the data do not show. Maximum likelihood balances the two." Only the *sill* enters $\ln|\mathbf{V}|$ as a penalty that grows.
2. **Lemma 4.1.1(2) (Jacobi's formula)** is correct but "taking differentials" skips the step; use the Taylor argument above.
3. **Theorem 4.1.2 mixes two definitions of Fisher information** (expected negative Hessian for $\mathcal{I}_{\boldsymbol\beta\boldsymbol\beta}$, covariance of the score for $\mathcal{I}_{\boldsymbol\theta\boldsymbol\theta}$) without saying that they are equal. Students know this from mathematical statistics; one sentence reminds them. Use the Hessian route for the $\boldsymbol\theta$ block (Exercise 4.1) so you do not need the unproved quadratic-form-covariance theorem.
4. **Beat 2 (L244–311).** The Cholesky explanation (L246–258) is clear and correct: $\ln|\mathbf{V}| = 2\sum\ln L_{ii}$ and the quadratic form is $\|\mathbf{w}\|^2$ after forward substitution. Two practical problems:
   - The code calls `cholesky_decompose` and `solve_triangular_lower` (and later `solve_cholesky_system`), which are not defined anywhere in this chapter; I found them only in the companion workbook. The first three snippets also lack `import math` (they use `math.inf`, `math.log`). Either define them once at the start of the chapter or point to where they live.
   - Name the efficiency idea out loud: *never form $\mathbf{V}^{-1}$*. The text hints but does not say it.
5. **Beat 3 (L317–325) is a third retelling of the nugget-as-regularizer idea** (Chapters 1, 2, and here), repeating the overclaim "strictly invertible regardless of how tightly clustered" (see the Chapter 2 review for the conditioning bound). More useful here: what goes wrong when *fitting*. For example: the likelihood is flat along a ridge in $(\sigma^2, \phi)$; the optimizer can drift to a huge range; estimates sit on the boundary (nugget exactly zero); starting values matter. None of that appears anywhere in this chapter except the microergodicity FAQ in a different section.

### §4.2 Profile Maximum Likelihood

**What works.** The hat-matrix bridge (L335–345) is the right comparison, and the explicit steps $\widehat{\boldsymbol\beta}(\boldsymbol\theta) \to \ell_p(\boldsymbol\theta)$ are easy to follow. Proposition 4.2.1 is a clean proof of the $(n-p)/n$ bias, with the correct scope.

**Issues**
1. **One equality is stated, not shown.** Definition 4.2.1 (L361–364) asserts $(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol\beta})'\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol\beta}) = \mathbf{Z}'\mathbf{P}\mathbf{Z}$. It is a two-line check: $\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol\beta} = \mathbf{M}\mathbf{Z}$ with $\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}$, and $\mathbf{M}'\mathbf{V}^{-1}\mathbf{M} = \mathbf{V}^{-1}\mathbf{M} = \mathbf{P}$.
2. **The profile estimator of $\sigma^2$ (L394–397) is asserted.** Add the two lines: with $\mathbf{V} = \sigma^2\mathbf{R}$, $\ell_p = \text{const} - \frac{n}{2}\ln\sigma^2 - \frac{1}{2\sigma^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}$; setting the derivative to zero gives $\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}/n$.
3. **The scope caveat belongs in the statement.** Proposition 4.2.1 holds for fixed $\boldsymbol\phi$ and a fixed nugget-to-sill ratio (the nugget fits in this form if its ratio to $\sigma^2$ is held fixed).
4. **Beat 2 is code only and nearly identical to §4.3's.** The text never says that the code is *ordinary regression on whitened data*: solve $\mathbf{L}\tilde{\mathbf{X}} = \mathbf{X}$ and $\mathbf{L}\tilde{\mathbf{z}} = \mathbf{Z}$, regress $\tilde{\mathbf{z}}$ on $\tilde{\mathbf{X}}$, then $\widehat{\boldsymbol\beta}_{\text{GLS}}$ is the OLS slope and $\mathbf{Z}'\mathbf{P}\mathbf{Z}$ is its residual sum of squares (I verified both). Say that; it replaces six numbered code comments. Show the difference between ML and REML in two lines instead of two near-copies of the function.
5. **Beat 3 (L486–489) is thin.** A concrete reader's question to answer here: "so how do I actually get $\widehat{\boldsymbol\theta}$?" (see the workflow below).

### §4.3 Restricted Maximum Likelihood

**What works.** The contrast-matrix story is cleanly told. Theorem 4.3.2 (invariance to the choice of contrast matrix) is a good, short proof. The FAQ on why REML cannot be used to compare different trend designs (L691–694) is correct and exactly the kind of thing students get wrong; keep it.

**Issues**
1. **Theorem 4.3.3's proof is a sketch (L546–564).** The Jacobian identity $|\det\mathbf{T}| = |\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}|^{-1/2}|\mathbf{K}'\mathbf{V}\mathbf{K}|^{1/2}|\mathbf{V}|^{-1/2}$ is correct; I checked it numerically and also that it does not depend on $\boldsymbol\theta$. But the last step ("factoring out the profile drift ... simplifies to") is not shown. A short, honest derivation fits the ceiling: since $\mathbf{Z}$ is a one-to-one transform of $(\widehat{\boldsymbol\beta}_{\text{GLS}}, \mathbf{W})$ with independent parts, $\ln f_{\mathbf{W}} = \ln f_{\mathbf{Z}} - \ln f_{\widehat{\boldsymbol\beta}} - \ln|\det\mathbf{T}|$; evaluate at $\boldsymbol\beta = \widehat{\boldsymbol\beta}$, where $\ln f_{\widehat{\boldsymbol\beta}} = -\frac p2\ln 2\pi + \frac12\ln|\mathbf{G}|$ and the first term is the profile likelihood $\ell_p$. That is three lines and explains where $-\frac12\ln|\mathbf{G}|$ comes from.
2. **Harville's formula is the contrast likelihood only up to a constant.** The text says the log-likelihood of the contrasts "simplifies to" Harville's expression. I checked: with an orthonormal contrast matrix, Harville's expression equals the contrast log-likelihood minus $\frac12\ln|\mathbf{X}'\mathbf{X}|$, a constant that does not depend on $\boldsymbol\theta$. Say "up to a constant free of $\boldsymbol\theta$." (Theorem 4.3.2 is about exactly this kind of constant, so the pieces fit.)
3. **The explanation of the $\ln|\mathbf{G}|$ penalty is garbled (L572, L686–687).** "A candidate $\mathbf{V}$ that attributes too much variance to the deterministic trend causes $|\mathbf{G}|$ to inflate." The trend is deterministic, and $\mathbf{G} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ gets *larger* when $\mathbf{V}$ gets *smaller*. A clearer story: $\ln|\mathbf{G}|$ measures how precisely $\boldsymbol\beta$ is pinned down *given* $\mathbf{V}$; REML charges for that precision, and in the scale case the charge is exactly $(p/2)\ln\sigma^2$, which converts $n$ into $n - p$. Exercise 4.2 already shows this (L1227–1231); move that two-line computation here.
4. **"ln det(G) restores the p lost degrees of freedom, eliminating variance downward bias!" (L682)** is true in the scale case only (see Intro issue 1). Soften.
5. **Theorem 4.3.5 (REML score and information) is stated without proof.** Its score follows from the ML score exactly as 4.1.1 did, in three lines: $\partial\ln|\mathbf{G}|/\partial\theta_k = -\operatorname{Tr}(\mathbf{G}^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{V}_k\mathbf{V}^{-1}\mathbf{X})$, and $\operatorname{Tr}(\mathbf{V}^{-1}\mathbf{V}_k) - \operatorname{Tr}(\mathbf{G}^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{V}_k\mathbf{V}^{-1}\mathbf{X}) = \operatorname{Tr}(\mathbf{P}\mathbf{V}_k)$. Show it.
6. **Beat 2 is code only** and the same objection applies as in §4.2.

### §4.4 Weighted and Generalized Least Squares Variogram Fitting

**What works.** The heteroscedastic-regression bridge (L704–710) is right, and the term-by-term anatomy of $Q_{\text{WLS}}$ is good. I checked the chapter's Nelder–Mead WLS code on synthetic data (true $(\sigma^2, \phi, \tau^2) = (1, 10, 0.2)$ with noisy empirical semivariances): it returned $(1.032, 10.284, 0.194)$, identical to scipy's Nelder–Mead on the same objective.

**Issues**
1. **The weights rest on $\operatorname{Var}(\widehat\gamma) \approx 2\gamma^2/N$, which the previous chapter shows can be badly off (L706–710).** Chapter 3's own Theorem 3.3.1 gives the true variance; in my Chapter 3 simulation the independent-pairs approximation was close at lag 1 ($0.0012$ against $0.0013$) but underestimated the variance by about 1.5 times at lag 2, 2 times at lag 3, and 4.6 times at lag 10. That is why GLS exists. Say that Cressie's WLS is a deliberate cheap approximation and show what it misses.
2. **A factual reversal in Beat 3 (L849).** "At small separation distances $h$, $\gamma(h)$ is small and pair count $|N(h)|$ is typically large." The pair count is *smallest* at the shortest lags: the number of pairs in a lag bin grows roughly in proportion to the bin's area (for $d = 2$, proportional to $h$ times the bin width). Checking with 400 random points in the unit square: pair counts in successive lag bins of width $0.1$ are $2{,}294$, $5{,}909$, $8{,}573$, $10{,}558$, $11{,}337$. The reason short lags get large weight is that $\gamma$ is *small there* (the $1/\gamma^2$ factor), not that pairs are plentiful. Also, with a nugget, $\gamma(h) \to \tau^2$, so "enormous weights" occur only when $\tau^2$ is near zero.
3. **"Spatial interpolation weights are governed almost exclusively by the slope of the variogram near the origin" (L851) overstates.** The near-origin shape matters most when sampling is dense relative to the range; when spacing is comparable to the range, the range and sill matter too, and the kriging variance depends on the whole curve. Say "mostly" and add the condition.
4. **Theorem 4.4.1 (see calibration table).** It also has no definition of $\mathbf{J}$ beyond "the limiting variance of the gradient," and its $\sqrt{n}$ scaling is unexplained when the sums are over lag bins.
5. **Beat 2 is a hand-written Nelder–Mead optimizer with no explanation.** The core idea (evaluate $Q_{\text{WLS}}$, hand it to an optimizer, keep the parameters positive) takes five lines of prose. The optimizer itself is plumbing and adds 35 lines that obscure the statistics. Suggest: show the objective function, say "hand this to any optimizer (Nelder–Mead is a good first choice because it needs no derivatives)," and add the practical advice the chapter omits: use positive starting values from the empirical variogram (nugget about the first-lag semivariance, sill about the plateau, range about a third of the distance where it levels off), and *try two or three different starting points* because the surface can have local minima.
6. **The microergodicity FAQ is in the wrong place.** It sits in the least-squares section but is really about the likelihood ridge in §4.2–4.3, and about infill asymptotics whereas Theorem 4.4.1 above it is about increasing-domain asymptotics. See the calibration table for the replacement explanation.

### §4.5 Multivariate Admissibility and Cramér's Theorem

**What works.** The Cauchy–Schwarz bridge (L863–867) and the Hermitian condition are well motivated. Sufficiency (L889–906) is elementary and nice. Exercise 4.3 is excellent (see Dimension 4).

**Issues**
1. **The necessity proof is circular (L908–909).** It takes any $\mathbf{v}$, notes $Y = \mathbf{v}'\mathbf{Z}$ has covariance $\mathbf{v}'\mathbf{C}\mathbf{v}$, applies Bochner to get a density $f_Y$, and asserts $f_Y = \mathbf{v}'\mathbf{f}\mathbf{v}$. But that equality presupposes the matrix $\mathbf{f}(\boldsymbol\omega)$ exists, which is the statement being proved. The standard fix defines $f_{ij}$ by Fourier inversion of $C_{ij}$ and uses polarization; at this level, simply cite the necessity direction (as Chapter 2 did for Bochner) and note, as there, that the full theorem uses a spectral *measure*; densities cover every model in this book.
2. **A sign slip in the sufficiency proof (L896).** The coefficient on $e^{i\boldsymbol\omega'\mathbf{s}_l}$ is conjugated in the wrong place relative to the exponent $e^{i\boldsymbol\omega'(\mathbf{s}_r - \mathbf{s}_l)}$ in the line above, so as written it computes $\sum\sum\mathbf{a}_r'\mathbf{C}(\mathbf{s}_l - \mathbf{s}_r)\mathbf{a}_l$. For real symmetric cross-covariances the answer is the same; for asymmetric ones it is not. Use $\mathbf{u}(\boldsymbol\omega) = \sum_l e^{-i\boldsymbol\omega'\mathbf{s}_l}\mathbf{a}_l$.
3. **The finite-grid check cannot prove admissibility (Beat 2).** `verify_bivariate_spectral_admissibility` evaluates the determinant on a finite grid of frequencies. A grid can only *refute* a candidate, not confirm it. Say so, and note the absolute tolerance $10^{-12}$ is scale dependent.
4. **Beat 3's example does not match its cross-reference (L964–966).** It uses exponential models with parameters $(10, 2, 5)$ and $\rho_{12} = 0.9$ and says "see Exercise 4.3." Exercise 4.3 uses rates $(1, 2, 1.5)$ and finds the bound $|\rho_{12}| \le \sqrt{70}/9 \approx 0.930$. I checked the Beat 3 example under both readings of the numbers: the largest admissible $|\rho_{12}|$ is about $0.74$ whether $(10, 2, 5)$ are ranges or decay rates, so $\rho_{12} = 0.9$ is inadmissible either way. But the worst frequency is $\omega \approx 0.17$ (if ranges) or $\omega \approx 5.8$ (if rates), not "intermediate," and the numbers are not Exercise 4.3's. Pick one reading, say which, and put the worked example there or cross-reference correctly.
5. **Beyond what Exercise 4.3 does, a student needs the idea in the finite world.** The simplest route is the one the chapter already uses in §4.6: build the joint covariance matrix for a small set of points and look at its smallest eigenvalue. A one-paragraph "check it by brute force" remark would connect the spectral condition to what students can compute.
6. **Hermitian vocabulary.** A "From the Lecture Hall" note: for the usual case of even cross-covariances the spectral matrix is real and symmetric, so "Hermitian PSD" means "real symmetric PSD at every frequency."

### §4.6 The Linear Model of Coregionalization

**What works.** The factor-analysis bridge ($\boldsymbol\Sigma = \mathbf{A}\mathbf{A}' + \boldsymbol\Psi$) is an ideal classical anchor. Theorem 4.6.2 (nearest positive semidefinite matrix by clipping eigenvalues) is a nice, self-contained linear-algebra result with a short proof. The cone picture, once redrawn, is a good Beat 3 anchor. I checked the chapter's Jacobi eigenvalue code on 200 random symmetric matrices (sizes 2 to 6): it matches numpy's eigenvalues and reconstruction to about $10^{-12}$.

**Issues**
1. **Theorem 4.6.1's "if and only if" is false (L1001–1004).** Positive semidefinite $\mathbf{B}_u$ is *sufficient* for validity (the proof shows exactly this). It is not necessary. Here is a counterexample I checked: take two exponential correlation functions with rates $2$ and $1$, $\mathbf{B}_1 = \operatorname{diag}(1, -0.1)$ and $\mathbf{B}_2 = \operatorname{diag}(0.5, 1)$. $\mathbf{B}_1$ has a negative eigenvalue, yet the joint covariance matrix on 60 points has smallest eigenvalue $+0.081$, and the spectral density of the second variable stays positive ($\ge 2\times10^{-5}$ over a wide frequency range). The negative part is out-weighed by the other structure's spectral mass. The correct statement: PSD $\mathbf{B}_u$ guarantees validity; validity requires only that $\sum_u\mathbf{B}_u f_u(\boldsymbol\omega) \succeq 0$ at every frequency. (It is necessary when the structures have different enough ranges and no other can compensate, but that is a different and subtler claim.)
2. **Projection repairs a candidate; it does not fit the data.** Theorem 4.6.2 minimizes the Frobenius distance to the *candidate matrix*, not the weighted least-squares criterion of §4.4. The clipped matrix is valid but may fit the empirical cross-variograms worse than the best valid matrix. Say so in a sentence and mention that iterative fitting schemes handle this (from memory, Goulard and Voltz, 1992; please verify).
3. **Beat 3 (L1154–1155).** "The nugget matrix $\mathbf{B}_0$ **must** be strictly positive definite." Validity needs only $\mathbf{B}_0 \succeq 0$ (e.g., a variable measured without error has a zero diagonal entry). Strict definiteness is a numerical-safety *recommendation*. Also $\mathbf{B}_0$ is not defined: Definition 4.6.1 sums over $u = 1, \dots, U$ and has no index $0$.
4. **The proof of Theorem 4.6.2 (L1033)** says "we must set off-diagonal entries to zero." The cleaner argument is a lower bound: for any PSD $\mathbf{M}$, the diagonal entries are non-negative and off-diagonal squares are non-negative, so the objective is at least $\sum_j(\lambda_j - \max(0, \lambda_j))^2$, which the clipped diagonal matrix achieves. (Needs "diagonal entries of a PSD matrix are non-negative"; one line.)
5. **Code vs. theorem.** The code clips at a floor `min_eigenvalue=1e-8`, not at $0$ as Theorem 4.6.2 says. That makes the result strictly positive definite, a fine choice, but the docstring says "matching Theorem 4.6.2." Say "(slightly modified)."
6. **Duplication within the section.** L978 and L980 say the same thing twice (the LMC as the way to guarantee PSD without Cramér integrals, attributed to Matheron 1982). The attribution is from memory; the LMC is usually credited to Journel and Huijbregts (1978) with Matheron's later work; please verify.
7. **The cone diagram (L1136–1150)** draws a slanted boundary where, in eigenvalue coordinates, the valid region is simply the non-negative quadrant. Redraw: the candidate sits below the $\lambda_1$ axis, and clipping drops it vertically onto the axis. A 2 × 2 hand example makes it concrete: $\widetilde{\mathbf{B}} = \begin{pmatrix}1 & 1.2\\1.2 & 1\end{pmatrix}$ has eigenvalues $2.2$ and $-0.2$ (checked), so $\mathbf{B}^* = 2.2\cdot\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix} = \begin{pmatrix}1.1&1.1\\1.1&1.1\end{pmatrix}$.

### §4.7 Analytical Exercises

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **6.5 / 10**
- Beat 1 bridges are the chapter's strength (all six sections).
- Beat 2 for four of the six sections is code only, so the reader gets an implementation without the story. This chapter is where the book's central object, the *estimator*, appears, and it never walks the student through "find $\widehat{\boldsymbol\theta}$" end to end.
- Beat 3 is the weakest beat (thin, repeated, or mismatched).
- **Missing workflow.** After §4.3, a student needs one page that says: (1) get starting values from a WLS fit to the empirical variogram (§4.4); (2) maximize the REML criterion over the range and the nugget-to-sill ratio, profiling out $\sigma^2$ in closed form ($\widehat\sigma^2 = \mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}/(n-p)$, Theorem 4.3.4), which turns a three-parameter search into a one- or two-parameter one (this is exactly how I coded the simulation above, and it works); (3) look at standard errors from the inverse REML information (Theorem 4.3.5) and a profile plot, and be suspicious of an estimate that sits on a boundary or flat ridge; (4) when comparing trend designs, switch to ML (the FAQ). The ingredients are all in the chapter; the recipe is not.
- **Decision guidance** (when to use WLS, ML, or REML) is missing.
- **Visible scaffolding:** "Beat 1/2/3" headings; also the very long section-bridge headings. **L11 dedication** to Rizzo and S&G should become what the student can do after the chapter.
- **Missing signposts:** "Chapter at a glance" and a "Key ideas" recap.

### Dimension 2 — Algebraic transparency and scaffolding: **7 / 10**
- Strengths: Theorem 4.1.1, Proposition 4.2.1, Theorem 4.3.2, Theorem 4.3.4, Theorem 4.6.1's expansion, and all four exercises' algebra.
- Gaps: equivalence of the two forms of $\mathbf{P}$; $\mathbf{Z}'\mathbf{P}\mathbf{Z}$ as the residual sum of squares; the profile $\sigma^2$ estimator; the last step of Theorem 4.3.3; Theorem 4.3.5; Theorem 4.4.1 entirely.
- **"From the Lecture Hall" callouts: none present.** Natural ones: (1) "GLS is OLS on whitened data" (boxed, with the Cholesky picture); (2) "Why REML is the $n-1$ trick in disguise"; (3) "Reading an information matrix: curvature of the likelihood"; (4) "What a spectral density matrix is, in the real symmetric case."

### Dimension 3 — Structural consistency and forward referencing: **6 / 10**
- **Duplication:** the invariance proposition and Theorem 4.2.1; REML introduced three times; nugget-as-regularizer a third time.
- **Notation:** $\mathbf{A}/\mathbf{U}$ vs. $\mathbf{K}/\mathbf{W}$; $\mathbf{Y}$ vs. $\mathbf{Z}$; $\mathbf{A}_k$ reused; $\mathbf{B}_0$ undefined.
- **Forward references and cross-links** are mostly well handled (Chapter 3 Theorem 3.3.1 is correctly cited in Definition 4.4.1).
- **Placement:** composite likelihood in the intro; microergodicity FAQ in the least-squares section; §4.5–4.6 (multivariate) arrive before cokriging (Chapter 7). Consider moving §4.5–4.6 to Chapter 7 (to be checked when I review it) or labeling them optional.
- **Code self-containment:** helpers not defined in the chapter; three snippets missing `import math`.

### Dimension 4 — Empirical and computational utility: **5.5 / 10**
- **No fitted example anywhere.** The chapter never fits a model to data, so the reader never sees a profile likelihood, a REML estimate beside an ML estimate, or a standard error. One small worked example (say 20 points in 1D, exponential covariance, an intercept plus a slope) with a table of ML and REML estimates and a profile over the range would carry the whole chapter.
- **Exercise 4.1** (Hessian and Fisher information): correct (I checked each step). It is the elementary route; use it in the theorem (see §4.1).
- **Exercise 4.2** (REML derivation and MSE): parts (a) and (b) are correct and instructive; the algebra for $\ln|\sigma^2\mathbf{R}| + \ln|\mathbf{X}'(\sigma^2\mathbf{R})^{-1}\mathbf{X}| = (n-p)\ln\sigma^2 + \text{const}$ is exactly what the main text needs. **Part (c) is wrong.** The ML mean squared error is variance plus squared bias, $\frac{2(n-p)\sigma^4}{n^2} + \frac{p^2\sigma^4}{n^2} = \frac{2n - 2p + p^2}{n^2}\sigma^4$, not $\frac{2n-p}{n^2}\sigma^4$ (the two agree only when $p = 1$). With the correct formula, ML has the smaller MSE only when $n(p - 4) < p(p - 2)$, which always holds for $p \le 4$ but fails for larger $p$ unless $n$ is small. Checked: ($n = 50$, $p = 3$): ML $0.0412$ vs. REML $0.0426$ (ML slightly lower); ($50$, $6$): $0.0496$ vs. $0.0455$ (REML lower); ($50$, $10$): $0.0720$ vs. $0.0500$; ($100$, $10$): $0.0280$ vs. $0.0222$. The conclusion "ML achieves a slightly lower MSE" is therefore wrong for the chapter's own motivating cases. Rewrite part (c) with the correct formula and the table, and mention the classical fact that the MSE-optimal divisor for this problem is $n - p + 2$. Also "$\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}/\sigma^2 \sim \chi^2_{n-p}$" is asserted; one line (whitening) justifies it.
- **Exercise 4.3** (Cramér admissibility): excellent, and I verified it end to end. The cross-covariance bound is $|\rho_{12}| \le \sqrt{70}/9 \approx 0.9296$, with the minimum at $\omega = \sqrt{6.5} \approx 2.55$ (numerical search gives $\omega = 2.5496$ and $\rho^2 = 0.86420 = 70/81$). It is a model of the finite, calculus-only counterexample the mission wants. Bring a version of it into the main text and fix the Beat 3 example to match.
- **Exercise 4.4** (Sylvester and Cholesky): correct (I checked the determinant $= 5$ and the factor). But the matrix is already positive definite, so it does not exercise the chapter's actual point, which is repairing an *invalid* matrix. Add a $2 \times 2$ invalid example and its projection (see §4.6 issue 7).
- No exercise asks the student to compute a likelihood or an estimate by hand for a tiny dataset (e.g., $n = 3$) or to interpret a profile.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Fix Exercise 4.2(c):** correct ML MSE is $\frac{2n - 2p + p^2}{n^2}\sigma^4$; replace the conclusion with the table; ML wins only for small $p$ or $n$.
2. **Fix Theorem 4.6.1:** PSD $\mathbf{B}_u$ is sufficient, not necessary; add the counterexample or at least the correct statement.
3. **Fix the pair-count claim in §4.4 Beat 3** (short lags have the *fewest* pairs); fix "almost exclusively."
4. **Fix the log-determinant intuition** (L112, L145): a longer range makes $\ln|\mathbf{V}|$ smaller, not larger.
5. **Fix the $\ln|\mathbf{G}|$ explanation** (L572, L686–687).
6. **Prove or correct the key equalities:** the two forms of $\mathbf{P}$; $\mathbf{Z}'\mathbf{P}\mathbf{Z}$ equals the whitened residual sum of squares; the last step of Theorem 4.3.3 (and say "up to a constant").
7. **State the scope of the $(n-p)/n$ factor** (scale-only, other parameters fixed); note the magnitude is not the story when other parameters are estimated.
8. **Replace the measure-theoretic microergodicity FAQ** with the slope-at-the-origin explanation; move it to the likelihood sections.
9. **Fix the sign slip** in the Cramér sufficiency proof and the circular necessity proof (cite instead).
10. **Fix the Beat 3 example in §4.5** so it matches Exercise 4.3, or cross-reference correctly.

### Should fix (pace, flow, voice)
11. **Add the whitening sentence** ("GLS is OLS on whitened data") and make it the Beat 2 explanation for §4.2–4.3.
12. **Add the estimation workflow page** (start values from WLS, profile $\sigma^2$, optimize over range and nugget ratio, standard errors, ridges and boundaries, ML vs. REML for model comparison).
13. **Add one worked fitted example** (tiny dataset, ML and REML side by side, a profile over the range).
14. **Replace Theorem 4.4.1** with a cited statement plus the Gauss–Markov analogy; state that WLS uses an approximate variance (and how badly it can miss).
15. **Make the code self-contained** (define helpers once, add imports) or point to where they live; present the WLS objective and say any optimizer will do, instead of the hand-written Nelder–Mead.
16. **Add Beat 3 content for 4.1–4.3** (flat ridges, boundary estimates, starting values, nugget at zero).
17. **Move composite likelihood** to the end or drop; **consider moving §4.5–4.6** to Chapter 7.
18. **Add four "From the Lecture Hall" callouts** and "Chapter at a glance" / "Key ideas."
19. **Replace "Beat" headings** and shorten long titles; replace the L11 dedication.
20. **Unify notation** ($\mathbf{Z}$ throughout; one set of letters for contrasts; define $\mathbf{B}_0$); remove the duplicated proposition and the repeated REML introductions.
21. **Use the Hessian proof of the Fisher information** in Theorem 4.1.2; add the 3-line REML score derivation for Theorem 4.3.5.

### Nice to have
22. Replace the Jacobi's-formula proof with the Taylor argument.
23. State that the finite-grid spectral check can only refute; note the tolerance.
24. Redraw the cone diagram as the non-negative quadrant; add the $2 \times 2$ projection example.
25. Say Frobenius projection repairs a candidate, not a fit.
26. Clarify "$\mathbf{B}_0$ must be strictly PD" as a numerical recommendation; Definition 4.6.1 sums from $u = 1$.
27. Fix "code matches Theorem 4.6.2" (floor $10^{-8}$ vs. $0$).
28. Add a $2 \times 2$ invalid-matrix exercise; add a hand-sized likelihood exercise.
29. Verify the attributions (Patterson & Thompson 1971; Harville 1974; Cressie 1985; Lahiri–Lee–Cressie 2002; Goulard & Voltz 1992; LMC origin).

---

## 6. What to Keep Exactly As It Is

- The $(n-1) \to (n-p)/n \to$ contrasts $\to \mathbf{P}$ bridge and its diagram (intro, L15–60).
- The hat-matrix-to-$\mathbf{P}$ comparison (L335–345).
- Proposition 4.2.1's proof of the $(n-p)/n$ bias, with the scope fix.
- The REML-versus-ML-for-model-comparison FAQ (L691–694).
- The heteroscedastic-regression bridge for WLS (L704–710).
- The sufficiency proof of Cramér's theorem (with the sign fix) and Exercise 4.3 in full.
- The factor-analysis bridge for the LMC and the clipping theorem (L976–1033).
- Exercises 4.1 and 4.4 (with the added invalid-matrix example).
