# Review: Chapter 6 — Spatial Regression and Non-Stationary Kriging

**File reviewed:** `ch06_spatial_regression.md` (the 1,790-line, 116,576-byte version last modified 19:45; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch06_exercises.md` was not reviewed. The four exercises inside §6.7 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy, and copies of the chapter's code). Literature attributions marked "from memory" are not verified.

---

## 1. Executive Summary

**Grade: 6.5 / 10**

The opening of this chapter is as good as anything in the book. The pipeline "OLS fails with correlated errors → whiten with the Cholesky factor → GLS → add simple kriging of the residuals → universal kriging" is exactly the classical-to-spatial bridge the mission calls for, and the Gauss–Markov proof in §6.2 (write any competing estimator as the GLS weights plus a correction $\mathbf{u}$ with $\mathbf{X}'\mathbf{u} = \mathbf{0}$) is a model of what a short, complete proof should look like for this audience. The chapter loses points for four reasons. Several confident claims are wrong, and I checked each by computation. Some topics lean on advanced machinery the mission asks us to avoid (thin-plate splines via calculus of variations). The same OLS-versus-GLS story is retold four times, while the chapter's best bridge (the structural parallel between the OLS prediction variance and the universal kriging variance) is never drawn. And the chapter tries to hold six topics: the core (OLS/GLS/UK) plus KED, dual kriging with splines, and intrinsic random functions, so the second half feels like several short chapters stapled on.

**Highest-value changes**
1. Correct the OLS-versus-GLS message. The chapter says GLS's efficiency gain over OLS "increases as spatial correlation strengthens" (L516). I computed the ratio on a 50-point transect with intercept and slope: the OLS variance exceeds the GLS variance by only 1% to 28%, rises and then falls with range, and is not monotone. The large effect is on the *standard errors*: the naive OLS standard error for the slope is 0.69 to 0.12 times the true one as the range grows. That is the lesson students need.
2. Fix Exercise 6.3 and the KED "blow-up" story. The weights are not large because the matrix is ill-conditioned; they are large because the target covariate lies far outside a tiny sampled range. With the exercise's own numbers, a target covariate of $100.005$ gives perfectly reasonable weights and variance ($0.469$); a target of $101$ gives a variance of $8{,}241$. And most of the quoted condition number ($4.5\times10^{12}$) disappears if the covariate is centered ($4.5\times10^{4}$).
3. Remove the claimed advantage of IRF-$k$ over universal kriging ("origin invariance"). Universal kriging predictions are invariant to shifting the coordinate origin. I verified that (agreement to $10^{-12}$ even with coordinates of order $10^5$). The real difference is that IRF-$k$ uses a *generalized* covariance that need not be bounded; the system is otherwise the same.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes for §6.1–6.4; drifts in §6.5–6.6.**

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L1063–1080 | Thin-plate splines via "the calculus of variations," "the biharmonic equation $\Delta^2 f = \sum b_i\delta_{\mathbf{s}_i}$," and "the fundamental Green's function" $K(h) = h^2\log h$ | **High** | This is the PDE-and-distributions material the mission excludes, and the key equivalence is asserted ("leads to"), not derived. Replace with the one-dimensional version students already know: the natural cubic spline. I checked numerically that "radial basis $|h|^3$ plus a linear drift" on six points reproduces the natural cubic spline to $9\times10^{-15}$. State the 2D thin-plate result as a cited remark. |
| L1258 | Nugget written as "$c_0\,\delta(h)$" (a Dirac delta) | Low | Use the indicator $\mathbb{1}\{h = 0\}$. A Dirac delta is a distribution, not a function. |
| L618 | "Karush–Kuhn–Tucker block structure" | Low | An optimization-theory label for what is just Lagrange multipliers with equality constraints. Drop. |
| L809 | "Runge's phenomenon" | Low | A misattribution: Runge's phenomenon is about high-degree polynomial *interpolation* at equally spaced nodes. What the section describes is polynomial trend *extrapolation*. See §6.3. |
| L1232–1263 | Authorized linear combinations, generalized covariances "defined modulo an even polynomial" | Medium | The linear algebra is fine ($\mathbf{X}_k'\boldsymbol\lambda = \mathbf{0}$), but the presentation is formal. Lead with the observation that Theorem 6.6.1 *is* the universal kriging system with a kernel $K$ in place of $C$ (§6.6 below). |
| L37, L516 | Loewner ordering | None | Define once ($\mathbf{A} \succeq \mathbf{B}$ means $\mathbf{A} - \mathbf{B}$ is positive semidefinite); fine. |

No measure theory, no Hilbert-space jargon, no Bessel functions.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| Intro | Yes (excellent) | n/a | n/a |
| 6.1 | Yes (repeats the intro) | Code of questionable value; no advice on the fix | Good diagnosis; workflow missing |
| 6.2 | Yes | Code ok; text/code mismatch | **Contains a false claim** |
| 6.3 | Yes (best bridge, half used) | Yes (clear) | Wrong growth exponents |
| 6.4 | Yes | Yes (explanation thin) | Misdiagnoses the failure |
| 6.5 | Yes (ridge-regression bridge is good) | Yes (short, good) | Overstates the speedup |
| 6.6 | Yes (ARIMA differencing bridge is excellent) | Yes (code ok) | **Contains a false claim** |
| 6.7 | n/a | n/a | n/a |

### Intro (before §6.1)

**What works.** L13–67 walk the student from OLS to whitened OLS to GLS with every line shown: $\operatorname{Var}(\boldsymbol\varepsilon^*) = \mathbf{L}^{-1}\mathbf{V}(\mathbf{L}')^{-1} = \mathbf{I}$ and $(\mathbf{L}')^{-1}\mathbf{L}^{-1} = \mathbf{V}^{-1}$. The "two-component additive structure" for the universal kriging predictor (L80–87) and the three special cases (no correlation, known mean, constant mean) are the right payoff.

**Issues**
1. **The L5 dedication to Rizzo and S&G** should become what students will be able to do after the chapter: fit a trend when the errors are correlated, predict with it, and know when to distrust it.
2. **"Non-stationary kriging" in the title (and L3) is misleading.** Everything in §6.1–6.4 has a non-constant *mean* and a *stationary* covariance for the residuals. "Non-stationary covariance" is Chapter 13. Retitle ("Spatial Regression and Universal Kriging"), or say "non-stationary in the mean" at the top.
3. **L37 says the OLS covariance is "strictly larger" than the GLS covariance.** It is larger or equal: there are designs (an intercept with equal correlations, for instance) where OLS and GLS coincide. L281 and L516 correctly say "$\ge$"/"positive semidefinite"; use that.
4. **L38 says the naive variance underestimates uncertainty "by factors of 2 to 10."** Say whether that is variance or standard error, and give numbers. From my checks: for 100 random sites in the unit square with a plane trend and exponential covariance, the true OLS standard errors are 2.2 times the naive ones (range 0.1) to about 4.4 times (range 0.3); on a 1D transect with an intercept and slope, the naive slope standard error is 0.69 down to 0.12 of the truth as the range grows from 0.02 to 10 (i.e., 1.5 to 8 times too small).
5. **The OLS-fails story is told four times** (intro L27–39; §6.1.1 L96–100; §6.1 Beat 3 L274–296; §6.2.1 L317, with the next paragraph L319 repeating the same sentence). Tell it once, with numbers.

### §6.1 Spatial Linear Models with Correlated Residuals

**What works.** The three-line decomposition picture (trend plus correlated residual) and the "OLS residuals are orthogonal to $\mathbf{X}$, so the fitted trend absorbs low-frequency variation" explanation (L298–308) are clear and correct. I checked the claim about the residual variogram quantitatively: for 100 random sites in the unit square with exponential covariance (range 0.3), the *expected* semivariogram of OLS residuals divided by the true one is $0.99, 0.93, 0.82, 0.63$ in lag bins $(0, 0.1], (0.2, 0.3], (0.4, 0.5], (0.6, 0.8]$ for a plane trend ($p = 3$) and $0.97, 0.80, 0.59, 0.43$ for a quadratic trend ($p = 6$). So: nearly unbiased at short lags, badly attenuated at long lags, and worse for more flexible trends. A small table like this would make the section land.

**Issues**
1. **The effect of ignoring a trend has a one-line proof the chapter omits (L100).** It says the variogram "displays an apparent parabolic upward curvature." For a linear drift $\mu(\mathbf{s}) = \mathbf{a}'\mathbf{s}$, $\operatorname{E}[(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))^2] = 2\gamma(\mathbf{h}) + (\mathbf{a}'\mathbf{h})^2$, so the empirical semivariogram is $\gamma(\mathbf{h}) + \frac12(\mathbf{a}'\mathbf{h})^2$. (This is the same mean-offset result as Chapter 3's cross-variogram theorem.)
2. **Notation.** Definition 6.1.1 gives $\varepsilon = \delta + \epsilon$ and says the kernel $C$ is "continuous," but $\varepsilon$ includes a nugget, so its covariance jumps at $\mathbf{0}$. L129 defines $\mathbf{V} = [C(\mathbf{s}_i - \mathbf{s}_j)]$ with no $\tau^2$, while the diagram at L166 writes $\mathbf{V} = \mathbf{C}(\boldsymbol\theta) + \tau^2\mathbf{I}_n$. (Same $C$-versus-$\mathbf{V}$ drift as Chapters 1–4.) Also $n > p$ (L120) and $p \le n$ (L126, L145) disagree.
3. **Beat 2 code (design matrix and conditioning check, about 100 lines) does not earn its place, and omits the fix.** Building a design matrix is a three-line loop. The conditioning function reports $\kappa(\mathbf{X}'\mathbf{X})$ but never says what to do about a bad value. The fix is to center and scale the coordinates (the chapter says so only at L1420, in the IRF-$k$ section). I checked: with a quadratic trend and UTM-sized coordinates (about $5\times10^5$ and $4\times10^6$), $\kappa(\mathbf{X}'\mathbf{X}) \approx 3.7\times10^{35}$; after centering and scaling it is about $20$. Even coordinates in $[0, 10]$ give $1.2\times10^5$ versus $16$. Replace the code with those three numbers and the advice.
4. **L296: the circularity dilemma "necessitates either Universal Kriging, REML, or Intrinsic Random Functions."** Universal kriging is a *prediction* method that *uses* an estimated $\mathbf{V}$; it does not resolve the circularity. The resolutions are: iterate (OLS residuals → variogram fit → GLS → recompute residuals → refit), maximize the REML likelihood from Chapter 4, or avoid the issue with IRF-$k$. Give the iteration as the recipe it is.

### §6.2 Generalized Least Squares Estimation

**What works.** Theorem 6.2.1's proof (L341–397) is exemplary: pre-whitening gives the estimator, then "write $\mathbf{w} = \mathbf{w}_0 + \mathbf{u}$ with $\mathbf{X}'\mathbf{u} = \mathbf{0}$; the cross term vanishes; so the variance is the GLS variance plus $\mathbf{u}'\mathbf{V}\mathbf{u} \ge 0$." Pencil-checkable, no assumptions beyond the Gauss–Markov setup. Keep exactly as is. I checked the chapter's GLS code against a direct solve: coefficients and covariance agree.

**Issues**
1. **Beat 3, L516, is false as stated.** "The efficiency gain of GLS over OLS increases as spatial correlation strengthens (i.e., as range $\phi$ grows)." I computed $\operatorname{Var}_{\text{OLS}}/\operatorname{Var}_{\text{GLS}}$ for 50 equally spaced sites on $[0,1]$, exponential covariance, intercept and slope. For ranges $0.02, 0.05, 0.1, 0.3, 1, 3, 10$:
   - intercept: $1.01, 1.06, 1.13, 1.21, 1.15, 1.07, 1.02$
   - slope: $1.02, 1.08, 1.17, 1.28, 1.24, 1.20, 1.19$

   So GLS buys between 1% and 28% in variance, the gain peaks when the range is comparable to the study region, and then it *falls* (for the intercept) or levels off. That is a known feature: when the trend columns are smooth, OLS is nearly efficient. Meanwhile the ratio of the naive OLS standard error to the true OLS standard error for the slope is $0.69, 0.45, 0.32, 0.20, 0.14, 0.13, 0.12$. **The message to teach:** the point estimates barely change; the standard errors you would have reported are drastically too small. Replace L516 with that.
2. **L515 describes whitening as rotating "into the eigenspace of $\mathbf{V}$."** Cholesky whitening is $\mathbf{L}^{-1}$, not an eigenvector rotation. Say "decorrelate."
3. **The "Spatial Confounding" FAQ (L518–527) has the right topic and an inaccurate remedy.** Item 3 says REML "directly mitigates this confounding." REML removes $\boldsymbol\beta$ from the likelihood for the covariance parameters; it does not change the fact that smooth covariates are nearly collinear with large-scale spatial variation, which is what makes $\widehat{\boldsymbol\beta}$ imprecise. (A separate line of work, "restricted spatial regression," addresses confounding directly; from memory.) Items 1–2 ("dramatically inflates," "sign reversals") are plausible but unsupported. The 1D table above is the evidence the FAQ lacks.
4. **Beat 2's text says "using Cholesky factorizations" (L412); the code does Gauss–Jordan elimination.** Pick one description. Also the code gives $\mathbf{V}$ no nugget parameter, while Chapter 5's simple-kriging code had one (see the Chapter 5 review on what a nugget changes).
5. **Duplicate sentence (L317, L319).**

### §6.3 Universal Kriging

**What works.** The OLS-prediction bridge (L536) is the right anchor: $\operatorname{Var}(Y_0 - \widehat{Y}_0) = \sigma^2 + \sigma^2\mathbf{x}_0'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_0$. The derivation of the augmented system, the two-stage decomposition, and the variance decomposition (Theorem 6.3.2) are correct, and the decomposition diagram (L691–700) has the best interpretive line in the chapter: the drift penalty is the variance of an estimated quantity, $\operatorname{Var}\!\big((\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)'\widehat{\boldsymbol\beta}_{\text{GLS}}\big)$. I verified numerically: universal kriging equals GLS plus simple kriging of the residuals; the variance decomposition holds (Exercise 6.1: $0.4702 = 0.4621 + 0.0080$); and the dual surface equals the primal.

**Issues**
1. **The chapter never draws the parallel it sets up.** The classical prediction variance is "noise plus parameter uncertainty": $\sigma^2[1 + \mathbf{x}_0'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_0]$. The universal kriging variance is "what remains after kriging plus parameter uncertainty": $\sigma_{\text{SK}}^2 + \mathbf{r}'(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{r}$ with $\mathbf{r} = \mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0$. Put the two formulas side by side in Beat 1 and again in Beat 3. It costs five lines and ties the chapter together. (Also, "the price of estimating drift" at L808 is stated without saying the price vanishes when $\mathbf{x}_0 = \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0$, which Theorem 6.3.2(4) does say.)
2. **Multiplier sign convention differs from Chapter 5.** Here the system is $\mathbf{V}\boldsymbol\lambda + \mathbf{X}\boldsymbol\mu_{\text{UK}} = \mathbf{v}_0$ with variance $C(\mathbf{0}) - \boldsymbol\lambda'\mathbf{v}_0 - \boldsymbol\mu_{\text{UK}}'\mathbf{x}_0$; for $\mathbf{X} = \mathbf{1}$ that gives $\mu_{\text{UK}} = -\mu_{\text{OK}}$ in Chapter 5's convention. The chapter says UK "reduces to Ordinary Kriging" when $p = 1$ (L87), so a student will compare the two systems and find a sign flip. Use one convention across chapters (see the Chapter 5 review).
3. **Beat 3 on polynomial drift has the wrong exponents and the wrong name (L809).** "Explodes cubically or quintically" is not right: for a drift of degree $k$ the penalty grows like (distance)$^{2k}$ far outside the data (the entries of $\mathbf{x}_0$ grow like distance$^k$ and the penalty is a quadratic form in $\mathbf{x}_0$). I checked: for 12 sites on $[0, 10]$, exponential covariance, the log-log slope of the penalty against distance beyond the edge approaches $3.99$ for a quadratic drift and $5.99$ for a cubic one (quartic and sextic). At moderate distances the apparent slopes are lower ($2.4$ and $3.6$ between distances 2 and 16), so "it grows rapidly" is safer than a specific power. And this is not Runge's phenomenon (see the calibration table); call it "polynomial extrapolation."
4. **A very common practical alternative is not mentioned: regression kriging** (fit the trend by OLS, krige the residuals). It is simpler, nearly as good when $n$ is large, and what many practitioners use. One paragraph would help: when is the shortcut safe, and where does the OLS-residual bias from §6.1 bite?
5. **Beat 2 code (L706–803)** is correct and is the chapter's first fully explained algorithm; I checked it against a direct solve. It has no nugget argument; see the Chapter 5 review.
6. **Algebra in the proof of Theorem 6.3.2(3) (L677–684)** collapses three terms into one in a single line. Add: $-\mathbf{r}'\mathbf{G}^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0 + \mathbf{x}_0'\mathbf{G}^{-1}\mathbf{r} = \mathbf{r}'\mathbf{G}^{-1}\mathbf{r}$ (with $\mathbf{G} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$). Also, say once that the bordered matrix is invertible (full-rank $\mathbf{X}$ and positive definite $\mathbf{V}$).

### §6.4 Kriging with External Drift (KED)

**What works.** The multiple-regression bridge (L818) and the four "regularity conditions" (L829–837) are mostly well chosen: especially "the covariate must be known at every target site" and the contrast with cokriging.

**Issues**
1. **Condition 3 (L836) gives the wrong remedy.** "The conditional expectation must be strictly linear in the coefficients. Non-linear relationships require non-linear anamorphosis prior to kriging." The requirement is linear *in the coefficients*, not in the covariate: a curved relationship is handled by adding columns ($q^2$, $\log q$) to the design matrix. Anamorphosis (Chapter 9) transforms the *response*, which is a different fix for a different problem.
2. **KED is universal kriging with different columns, and the section should say so up front.** L841 says it after a page of setup. Beat 2's code repeats the augmented-system solver a fifth time; consider presenting KED as three lines on top of the §6.3 function.
3. **Beat 2's fallback logic is under-explained and inconsistent with Exercise 6.3.** The code reverts to ordinary kriging when a covariate's sample variance is below an *absolute* $10^{-6}$; the exercise's remedy says a *relative* threshold ($10^{-4}$ of the global variance). The check is per covariate, so it cannot detect collinearity among several covariates. And a silent switch to a different model should at least return a warning (the code does return a mode flag, which is good).
4. **Beat 3 mis-diagnoses the failure (L973–974).** "$\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ is singular, the matrix cannot be inverted, the system blows up." It is *near*-singular unless the covariate is exactly constant, and the blow-up depends on where you predict. This is the section's most important correction; see Exercise 6.3 under Dimension 4. The second bullet (extrapolating in the covariate, L974) is the correct and important one; make it the headline.

### §6.5 Dual Kriging and Thin-Plate Spline Equivalence

**What works.** The ridge-regression bridge via the "push-through identity" $(\mathbf{X}'\mathbf{X} + \lambda\mathbf{I}_p)^{-1}\mathbf{X}' = \mathbf{X}'(\mathbf{X}\mathbf{X}' + \lambda\mathbf{I}_n)^{-1}$ (L992) is a good classical anchor: solve once in data space, then predict anywhere with inner products. Theorem 6.5.1's proof is short and correct, and the Beat 2 code (two small functions) is a good use of code: it shows the "solve once, evaluate many times" idea. I verified the dual surface equals the primal prediction.

**Issues**
1. **Prove the push-through identity in one line:** $\mathbf{X}'(\mathbf{X}\mathbf{X}' + \lambda\mathbf{I}_n) = (\mathbf{X}'\mathbf{X} + \lambda\mathbf{I}_p)\mathbf{X}'$, then multiply by the two inverses.
2. **Name the idea students may already know.** The dual form is the kernel-ridge / Gaussian-process posterior mean: $\sum_i b_i\,k(\mathbf{s}, \mathbf{s}_i)$ with weights found once. For a data-science audience that is the hook.
3. **The speedup claim is inflated (L1000, L1205).** It compares $10^6$ separate $O(n^3)$ solves with one solve. A practitioner factors $\mathbf{V}$ once and reuses the factor, so each target costs $O(n^2)$ (two triangular solves), not $O(n^3)$. The dual form costs $O(n)$ per target. So the saving is a factor of $n$, which is real but not "several orders of magnitude" unless $n$ is in the thousands.
4. **A major limitation is not mentioned: dual kriging gives predictions but no prediction variances.** The variance needs the per-target system.
5. **Notation.** The drift coefficients are called $\mathbf{d}$, which is also the dimension of the domain ($D \subset \mathbb{R}^d$). Also the theorem is stated in covariance form while Exercise 6.2 uses the variogram form with a sign flip on $\mathbf{b}$ (explained in the solution at L1650–1662, but only there). State the sign convention in the theorem, or state it for a general kernel.
6. **Forward reference.** §6.5.2 says the thin-plate spline equals dual kriging with a "generalized covariance kernel," a concept that is not introduced until §6.6. Either reorder (put IRF-$k$ first) or define it here.
7. **The thin-plate spline discussion** (calibration table) should be replaced by the one-dimensional story: "a cubic smoothing spline *is* kriging with kernel $|h|^3$ and a linear drift." I verified it: six data points, solve the dual system with $K(h) = |h|^3$ and drift $(1, s)$, evaluate; the result equals SciPy's natural cubic spline to $9\times10^{-15}$. Then state the 2D thin-plate result as a remark with a pointer.
8. **Also worth one sentence:** a smoothing spline (with a penalty parameter) corresponds to kriging *with a nugget*; interpolation corresponds to none.

### §6.6 Intrinsic Random Functions of Order $k$

**What works.** The ARIMA-differencing bridge (L1216–1224) is excellent: "the difference operator filters out polynomials without ever estimating their parameters," with the weights $(1, -1)$ and $(1, -2, 1)$ and the two annihilation checks. It is exactly the intuition students need, and it connects to Chapter 2's contrast weights. Exercise 6.4's second-difference example is a good concrete illustration.

**Issues**
1. **Beat 3's "origin invariance" claim is false (L1418).** It says universal kriging's drift coefficients change when the origin is shifted and, by contrast, IRF-$k$ predictions are "strictly invariant." The coefficients do change, but the *predictions* of universal kriging do not: shifting the coordinates by $(10^5, 2\times10^5)$ changed the prediction in the 12th digit ($1.899217820177889$ versus $1.899217820181189$), even with a condition number of $6\times10^{14}$; a modest shift gave identical output. The span of the drift basis is unchanged, so the predictor is unchanged. The only real issue is numerical conditioning (center and scale). What IRF-$k$ actually offers is a *generalized covariance* $K$ that need not be bounded or have a sill.
2. **Theorem 6.6.1 is the universal kriging system with $K$ in place of $C$.** Say so in one sentence and cut the proof to a pointer. The prediction error is an authorized combination, so $\boldsymbol\lambda'\mathbf{K}\boldsymbol\lambda$ is a legitimate variance; the rest is Theorem 6.3.1.
3. **The practical heart of IRF-$k$ is missing: how do you choose and fit $K$?** The chapter gives the family ($K(h) = -b_0 h + b_1 h^3 - \dots$) and a default in the code ($-h + 0.05h^3$, with no justification), but never says how the $b_m$ are estimated, how to check validity (the quadratic form on ALC-$k$ weights must be non-negative), or when $k = 0$ versus $k = 1$ is appropriate. Beat 3's advice to compare cross-validation errors is good; give the procedure (or say it is beyond scope and point to a reference).
4. **Definition 6.6.3's family formula states no validity conditions** (sign and size of the $b_m$; dimension). "Valid" isotropic polynomial forms need $b_m \ge 0$ (I verified signs against the standard conditional-positive-definiteness results, from memory), and it is worth stating.
5. **Beat 2 code:** I checked it against a direct solve; weights, prediction, and variance agree. Issues: the error message for $k > 1$ says "requires explicit basis specification," but the function has no parameter for a basis, so there is no way to supply one; and a slightly negative variance is silently returned for a kernel that is not valid.
6. **Collinearity advice belongs earlier.** "Always center and scale coordinates" (L1420) is the most useful practical sentence in the chapter and is buried in §6.6.

### §6.7 Analytical Exercises

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **7 / 10**
- All sections open with a real bridge, and the best ones (OLS→GLS, ridge→dual, ARIMA→IRF-$k$) are very good.
- Pacing suffers from scope: six topics in one chapter. The core (§6.1–6.3, plus KED as a short subsection) is a satisfying arc. Dual kriging with splines and IRF-$k$ are advanced extras; they would sit better in an "advanced topics" section, in Chapter 13, or at least be labeled optional.
- **Missing signposts:** "Chapter at a glance," a "Key ideas" recap, and a single *running example* that carries from OLS through GLS to UK. The 1D transect of Exercise 6.1 (three sites, linear drift) is already such an example; the chapter body never uses it.
- **Visible scaffolding:** "Beat 1/2/3" headings; very long bridge titles.
- **Missing workflow** (as in Chapter 4): the recipe from raw data to a kriged map. Here it is: (1) center and scale; (2) OLS trend; (3) variogram of residuals; (4) fit; (5) GLS or REML; (6) recompute residuals and iterate; (7) krige; (8) leave-one-out check.

### Dimension 2 — Algebraic transparency and scaffolding: **7.5 / 10**
- Strengths: the whitening derivation, Theorem 6.2.1, the augmented-system derivation, Theorem 6.5.1, and the algebra in Exercises 6.1–6.2.
- Gaps: push-through identity; the combination step in Theorem 6.3.2(3); invertibility of the bordered matrix; the mean-offset formula for a trend; and the entire thin-plate equivalence.
- **"From the Lecture Hall" callouts: none present.** Natural ones: (1) "Whitening in one picture: decorrelate, then it's just regression"; (2) "Noise plus parameter uncertainty: why prediction variance has two terms"; (3) "Why a spline is a kriging predictor (the cubic-spline case)"; (4) "Differencing kills polynomials" (the ARIMA bridge, boxed).

### Dimension 3 — Structural consistency and forward referencing: **5.5 / 10**
- **Duplication:** the OLS-fails story four times; the whitening derivation in the intro and in Theorem 6.2.1's proof; the "GLS + simple kriging of residuals" decomposition in the intro (L80–87), §6.3.1 (L536), and Theorem 6.3.2; the augmented-system solver coded five times (UK, KED, dual, IRF-$k$, plus GLS).
- **Notation:** $\mathbf{v}_0$/$\mathbf{c}_0$; the multiplier sign (flips relative to Chapter 5); $\mathbf{d}$ for coefficients and dimension; $C$ with and without the nugget; $n > p$ vs. $p \le n$.
- **Forward reference:** generalized covariance used in §6.5 before §6.6 defines it.
- **Title:** "non-stationary" means non-stationary mean only.

### Dimension 4 — Empirical and computational utility: **6 / 10**
- **Exercise 6.1** (universal kriging on a 3-point transect): correct. I recomputed everything: $\boldsymbol\lambda = (0.014705, 0.470591, 0.514705)$ matches; $\sigma_{\text{SK}}^2 = 0.462117$ matches. The multipliers and variance have small arithmetic slips from rounding: the chapter gets $\mu_0 = -0.058778$, $\mu_1 = -0.024463$, $\sigma_{\text{UK}}^2 = 0.470115$; exact values are $-0.058819$, $-0.024466$, $0.470159$ (difference $0.00804$, not $0.00800$). The conclusion is unaffected. Use more digits. This exercise would make an excellent running example for the whole chapter.
- **Exercise 6.2** (primal = dual): correct (checked: $\mathbf{b} = (1.5, -1.5)$, $d_1 = 3.5$, surface $2 + 3s$). Good, and elementary. Add a sentence early in the solution on why the variogram form flips the sign of $\mathbf{b}$.
- **Exercise 6.3** (KED collinearity): **its conclusions are wrong as stated.**
  - Part (b)'s numbers are right ($\kappa(\mathbf{X}'\mathbf{X}) = 4.5\times10^{12}$, determinant $0.0002$) but it never states the rank (it is 2), and nearly all of the ill-conditioning comes from not centering: with $q$ centered, $\kappa = 4.5\times10^{4}$.
  - Part (c) says $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ is "proportional to $\mathbf{X}'\mathbf{X}$" (it is not; $\mathbf{V}^{-1}$ is not a scalar) and that the weights satisfy $|\lambda_i| > 10^6$ and the variance "explodes." The constraints say $\lambda_1 + \lambda_2 + \lambda_3 = 1$ and $\sum\lambda_i q_i = q(s_0)$, which with these data gives $\lambda_2 = (q(s_0) - 100)/0.01$. The weights are large only when the *target covariate* is far from $100$. I checked: $q(s_0) = 100.005$ gives weights $(0.028, 0.500, 0.472)$ and variance $0.469$; $100.02$ gives $(-0.722, 2.0, -0.278)$ and variance $2.4$; $101$ gives $(-49.7, 100, -49.3)$ and variance $8{,}241$; $105$ gives variance $2.1\times10^5$. A weight above $10^6$ requires a target about $10^4$ units away.
  - The correct lesson has two parts: a *numerical* one (center and scale; condition number falls by eight orders of magnitude) and a *statistical* one (when a covariate barely varies in the sample, its coefficient is nearly unidentified, so predicting at a very different covariate value is a long extrapolation and the large variance is the model correctly saying so). Falling back to ordinary kriging is then a deliberate modeling choice that discards the covariate, not a numerical repair.
- **Exercise 6.4** (IRF-1): correct: the second difference with weights $(1, -2, 1)$ is an ALC-1 and has variance $12$ under $K(h) = -|h| + |h|^3$. Two cosmetic issues: part (b) introduces four points and then uses only three (weight $0$ on the fourth), and the exercise does not say why this $K$ is valid. A nice extra line: the $-|h|$ piece alone contributes $4$ and the $|h|^3$ piece $8$, so each valid piece gives a non-negative variance.
- **Missing from the chapter and the exercises:** a hand-sized "compare naive and true OLS standard errors for three points" exercise (the classical-to-spatial heart of §6.1–6.2); a data example; any exercise on regression kriging versus UK.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Replace the false OLS/GLS efficiency claim** (L516) with the real message: small variance gain, large standard-error correction (use the numbers above).
2. **Fix Exercise 6.3 and §6.4 Beat 3:** the instability is extrapolation in the covariate relative to its sampled range; center the covariate (condition number $4.5\times10^{12} \to 4.5\times10^{4}$); remove "proportional to $\mathbf{X}'\mathbf{X}$" and "$|\lambda_i| > 10^6$."
3. **Delete the "origin invariance" advantage of IRF-$k$** (L1418); universal kriging predictions are invariant. State the true difference (unbounded generalized covariance; same system).
4. **Fix the polynomial-extrapolation exponents** (quartic/sextic for quadratic/cubic drift; not Runge's phenomenon).
5. **Fix §6.4 condition 3:** linearity in the coefficients; add columns, not anamorphosis.
6. **Correct the REML-resolves-confounding claim** in the FAQ.
7. **Remove "strictly larger"** (L37).
8. **Replace the thin-plate spline derivation** (calculus of variations, biharmonic Green's function) with the 1D cubic-spline equivalence (verified) and a pointer.
9. **Unify the Lagrange-multiplier convention with Chapter 5.**

### Should fix (pace, flow, voice)
10. **Draw the parallel** between the OLS prediction variance and the UK variance (side by side, twice).
11. **Tell the OLS-fails story once,** with numbers; cut the other three tellings.
12. **Move centering and scaling to §6.1** with the condition numbers; drop the 100-line conditioning code.
13. **Give the workflow** (center, OLS, variogram of residuals, fit, GLS/REML, iterate, krige, leave-one-out), and say how it resolves the circularity.
14. **Add the one-line proof** of the trend effect on the variogram ($+\tfrac12(\mathbf{a}'\mathbf{h})^2$).
15. **Mention regression kriging** and when the shortcut is safe.
16. **Dual kriging:** prove the push-through identity; name it kernel/GP regression; correct the speedup baseline ($O(n^2)$ vs. $O(n)$ per target); say it gives no variances; fix the $d$ clash and the sign convention.
17. **IRF-$k$:** present Theorem 6.6.1 as "UK with $K$ in place of $C$"; say how $K$ is chosen and checked; state validity conditions; fix the code's error message.
18. **Restructure:** move the forward reference (generalized covariance) or reorder §6.5/6.6; label §6.5–6.6 optional or move to an advanced section; retitle the chapter.
19. **Use Exercise 6.1 as a running example** from §6.1 on.
20. **Add four "From the Lecture Hall" callouts** and "Chapter at a glance" / "Key ideas."
21. **Replace "Beat" headings;** replace the L5 dedication.

### Nice to have
22. Fix the small numeric slips in Exercise 6.1 (more digits).
23. Replace the nugget notation "$c_0\delta(h)$" with an indicator; drop "Karush–Kuhn–Tucker."
24. Fix the Beat 2 text/code mismatch (Cholesky vs. Gauss–Jordan) and the duplicate sentence at L317/L319.
25. Reconcile KED's fallback threshold (absolute in code, relative in the exercise); mention collinearity among several covariates.
26. Clarify that "UK reduces to SK" means $\boldsymbol\beta$ known.
27. Add a hand-sized exercise comparing naive and true OLS standard errors.
28. Fix "(b)" in Exercise 6.4 (four points, three used); state the rank in Exercise 6.3(b).
29. Explain how a smoothing spline corresponds to kriging with a nugget.

---

## 6. What to Keep Exactly As It Is

- The OLS → whitening → GLS → UK bridge in the intro (L9–87), minus the repetition elsewhere.
- Theorem 6.2.1's proof (the $\mathbf{w}_0 + \mathbf{u}$ argument).
- The OLS-prediction-variance bridge into UK (L536) and the drift-penalty-as-a-variance diagram (L691–700).
- The "OLS residuals are orthogonal to $\mathbf{X}$, so the trend absorbs low-frequency variation" explanation, with the trace identity (L298–308).
- The ridge-regression push-through bridge to dual kriging (L984–998) and the "solve once, evaluate many times" code.
- The ARIMA-differencing bridge to IRF-$k$ (L1216–1224).
- Exercises 6.1 (with more digits), 6.2, and 6.4.
