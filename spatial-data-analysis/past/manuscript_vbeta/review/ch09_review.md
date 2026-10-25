# Review: Chapter 9 — Non-Gaussian and Nonlinear Spatial Estimation

**File reviewed:** `ch09_nonlinear_geostatistics.md` (the 940-line, 62,039-byte version last modified 20:09:16; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch09_exercises.md` was not reviewed. The three exercises inside §9.8 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy, Monte Carlo, and copies of the chapter's own code). Literature attributions marked "from memory" are not verified.

---

## 1. Executive Summary

**Grade: 5.5 / 10**

The chapter picks the right story and has some excellent moments. The lesson that a naive back-transform $\exp(\widehat Y)$ is the conditional *median*, not the mean (L216, L227), is exactly the kind of insight a student keeps. The lognormal moment proofs (L142–157) are clean, and Exercises 9.1 and 9.3 are correct and well scaffolded. The Hermite orthogonality proof by integrating Rodrigues' formula by parts (L564–568) and the Mehler proof through the generating function (L576–586) are short and honest. I checked Mehler's formula numerically and it holds to about 1e-14.

But this is the first chapter where the *correctness* problems outweigh the pedagogy problems, and two of them sit in the chapter's two algorithms. Four things stand out.

1. **The PAVA algorithm as written is not the projection it claims to be.** Clamping to $[0,1]$ *before* pooling (L422, code L461) gives the wrong answer in most cases. On 300 random vectors, the chapter's order disagreed with a direct quadratic-programming solve in 222 of them. Pooling first and clamping afterwards matched in all 300.
2. **Theorem 9.2.3 has a sign error** given its own convention, and its "exact UMVUE" and "conditionally unbiased" language is not supported. With the chapter's stated system, the multiplier correction should be $+\mu_L$, not $-\mu_L$.
3. **The Hermite anamorphosis code does not produce a usable transform.** The fitted $\Phi$ ripples (non-monotone) inside the data range, and the bisection inverse sends the largest samples to the wrong Gaussian scores by about 2 units.
4. **The middle of the chapter (§9.4–§9.7) is definitions and theorems with no worked example, no running data set, and Beats 2 and 3 mostly absent.** A student learns what the objects are but never watches IK or DK produce a number or a decision.

**Highest-value changes**
1. Fix PAVA (pool, then clamp) and Theorem 9.2.3 (sign, and say what "unbiased" means). These are one-line corrections with large consequences.
2. Add one **running synthetic example** that threads §9.2–§9.7. Simulate a lognormal field, so the exact conditional distribution at the target is known, then show lognormal kriging, IK at three thresholds, and DK all aimed at the same target and graded against the truth. This also supplies the Beat 2 and Beat 3 that §9.2–§9.4 and §9.7 lack.
3. Make the classical-to-spatial bridge actually pay off. §9.1 criticizes the linear probability model (L15) and then never notices that indicator kriging *is* a spatial linear probability model. Order-relation violations (§9.5) are the LPM's out-of-range predictions in spatial dress. Saying so ties §9.1, §9.4, §9.5, and the Gaussian/probit alternative in §9.6 into one story.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes, with a handful of leaks.** The core computations (Gaussian MGF, conditional normal, Jensen, orthogonal polynomials, projection, Lagrange multipliers) are within the ceiling. The leaks are mostly vocabulary.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L5, L68 | "$L^2(\Omega,\mathcal F,P)$"; "Let $(\Omega,\mathcal F,P)$ be a complete probability space" | **High** (mission violation) | Delete. Say "among all linear combinations of the data" and "among all functions of the data." Nothing downstream uses $\Omega$ or $\mathcal F$. |
| L694 | "orthonormal basis $\{\chi_n\}$ in $L^2(\mathbb R, dF_0)$" | Medium | Say "polynomials that are orthonormal under the expectation inner product $\operatorname{E}[f(Y)g(Y)]$." You already define that inner product at L531–534. |
| L702 | "For any measurable transformation" | Low | Drop "measurable." |
| L433 | "strictly convex, continuous, and coercive … standard convex optimization theory" | Medium | The projection claim is true, but "coercive" and "closed convex cone" are optimization jargon. A picture of $K=2$ (project a point onto the half-plane $x_1\le x_2$) does more than the proof. |
| L434, L510 | "Karush–Kuhn–Tucker" | Medium | Inequality-constraint multipliers go one step past Lagrange multipliers. Use the direct argument: within a pooled block the best common value is the weighted mean (it is a least-squares fit to a constant). |
| L307–311 | "Mills' ratio asymptotic expansions for bivariate Gaussian lower tails" | **High** | A black-box citation, and the constant is wrong (see §4). Replace with the two-line bound in the remediation list. |
| L206 | "exact uniformly minimum variance unbiased estimator (UMVUE)" | Medium | Not shown and not true as stated (see §3, §9.2). |
| L715–725 | "orthogonality principle in the vector space of random variables" | Low | Fine. It is the same normal-equations argument as Chapter 5. Name it that way. |

The measure-theory tone is much lighter than in Chapters 7–8. The Hermite and PAVA material stays at the level of calculus and least squares.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 9.1 | Yes (long bridge) | **Missing** | Present, one example, mislabelled |
| 9.2 | Yes | **Missing** (no recipe) | Present, duplicated |
| 9.3 | Yes | **Missing** | **Missing** |
| 9.4 | Yes | **Missing** | **Missing** |
| 9.5 | Yes | Yes, but wrong order | Yes, with unsupported claims |
| 9.6 | Yes | Yes, but the code misbehaves | **Missing** |
| 9.7 | Yes | **Missing** | **Missing** |
| 9.8 | n/a | n/a | n/a |

Seven of eight numbered sections are missing at least one beat. The two "Wait—Why Did We Do That?" callouts (L224, L506) are the book's best voice. They read like a person talking. Use that voice for the missing beats.

### Intro and §9.1 Non-Gaussian Fields and the Limits of Linear Kriging

**What works.** The motivation (grades, plumes, permeability; L3, L48) is concrete. Pointing out that decisions are about nonlinear functionals $\psi(Z)$, not about $Z$ itself (L50–54), is the right framing for the whole chapter.

**Issues**
- **The dedication paragraph (L5).** "Following the pedagogical architecture of Maria L. Rizzo and … Schabenberger & Gotway" tells the reader who the author admires, not what they are about to learn. It has appeared in every chapter. Replace with one sentence that names the chapter's promise: *"By the end you will be able to predict a nonlinear quantity such as the probability of exceeding a regulatory limit, and say why a smooth kriging map cannot answer that question."*
- **The bridge (L13–66) is the longest in the book and its two halves are of unequal value.** The LPM → probit/logit half is excellent classroom material. The Hermite half ("classical statistics expands $f(Y)$ in Hermite polynomials," L21–25) is not something an MS student has met. The closer classical analogue is **orthogonal polynomial regression** (`poly(x, 3)` in R): orthogonal columns make the normal equations diagonal, so each coefficient is fit independently. That is the whole reason Disjunctive Kriging decouples (§9.7), and it is a prerequisite they own. Move the Hermite bridge to §9.6, where it is needed, and keep §9.1 to the LPM story.
- **The LPM bridge is set up and then abandoned.** L15 lists exactly what goes wrong with a linear probability model: predictions outside $[0,1]$ and heteroskedastic errors. IK (§9.4) is that model, kriged. §9.5's order-relation violations are the same failure. Neither section says so. One sentence in each, plus a forward reference to Chapter 12's spatial GLMs as the likelihood-based alternative, would make the chapter feel designed rather than assembled.
- **Notation collisions in the bridge, already visible.** $\Phi$ is the probit CDF at L19, the anamorphosis function at L64 and L523, and $G$ is then used for the normal CDF at L527. $\phi_k$ are Hermite coefficients at L64 but $\phi$ is the normal density at L530. $\psi_n$ are Hermite coefficients in §9.6–9.7 but $\psi$ is the target function at L52 and L689. $n$ is the sample size and the Hermite degree (see §9.7). Pick one letter per job, put it in `notation_index.md`, and keep it.
- **The three-layer decomposition (L68–72) is introduced and never used.** The nugget $\epsilon$ and the drift $\mu(\mathbf s)$ do not appear again in IK, anamorphosis, or DK, and a noisy $\epsilon$ would break the transform (you cannot take $\log$ of "signal + noise" and keep a Gaussian). Either delete it or add a sentence saying the chapter assumes noise-free data and a constant mean.
- **Proposition 9.1.1 (L90–99) overreaches.** Line 97 equates $\psi(\operatorname{E}[Z_0\mid\mathbf Z])$ with $\psi(\widehat Z_{\text{SK}})$. That holds only when the conditional mean is linear in the data, which is the Gaussian case. For the lognormal field of §9.2, $\widehat Z_{\text{SK}}$ built from $Z$ itself is *not* the conditional mean. Jensen's inequality also supports "underestimation" only for convex $\psi$. The chapter's own first example, the indicator $\mathbb I\{Z\le u\}$ (L52), is not convex, so Jensen says nothing about it. "Fails catastrophically" (L54) is also stronger than the evidence.
- **The "Jensen's Inequality Trap" (L106) is about a step function, not Jensen.** The failure is that plugging a smooth prediction into a step function throws away the spread. The 45% figure is unexplained. It is reproducible: if the conditional standard deviation is about $0.40\,z_c$ and the conditional distribution is symmetric around $0.95\,z_c$, then $P(Z\ge z_c)\approx 0.45$. State that, and the number becomes a lesson instead of an assertion.
- **Beat 2 is absent.** A short "Which question are you asking?" table would serve: *mean of a nonlinear function* → lognormal/DK; *probability of exceeding a limit* → IK; *a map the client can act on* → E-type or quantile from the CCDF.

### §9.2 Lognormal Spatial Fields and Lognormal Kriging

**What works.** Theorems 9.2.1 and 9.2.2 are correct and the proofs are the kind of MGF calculation the book should want more of. I checked Exercise 9.1 end to end: $\lambda=0.44341$, $\widehat Y=1.5224$, $\sigma^2=0.2311$, naive $4.583$ vs. back-transformed $5.144$, ratio $0.891$ (10.9% low), and the 95% interval $[1.79, 11.76]$ all reproduce. The explanation of the $\tfrac12\sigma^2$ term as "the conditional median, lifted to the mean" (L216–217, L227) is the best paragraph in the chapter.

**Issues**
- **Theorem 9.2.3 has a sign error under its own convention, and the surrounding claims are not supported.** The chapter defines the OK system as $\mathbf V_Y\boldsymbol\lambda+\mu_L\mathbf 1=\mathbf c_Y$ (L205). Under that convention the OK error variance is $\sigma^2_{\text{OK}}=C(0)-\boldsymbol\lambda'\mathbf c-\mu_L$ (checked: 0.3740 both ways), and the correction that makes the back-transform unbiased is $+\mu_L$, not $-\mu_L$ (L208, L214). I tested a three-point toy ($\sigma_Y^2=0.8$, exponential covariance, range 2, $\mu_Y=1$). The ratio $\operatorname{E}[\widehat Z]/m_Z$ is **1.121** with the chapter's $-\mu_L$, **1.000** with $+\mu_L$, and **1.059** with no multiplier term. Monte Carlo (400,000 draws) agrees: 1.120, 1.000, 1.058. The "−μ" form is correct in conventions where the system is written $\mathbf V\boldsymbol\lambda-\mu\mathbf 1=\mathbf c$ (this is the Chapter 5 convention per my Ch5 review). State which convention you are in, once, and use the matching sign.
- **"Exact UMVUE" (L206) and "conditionally unbiased" (L218) are not what this estimator is.** What the correction buys is *unconditional* unbiasedness, $\operatorname{E}[\widehat Z]=m_Z$. In the same simulation, the regression slope of the true $Z_0$ on $\widehat Z$ was 0.87 (not 1), so it is conditionally biased. The anatomy also says $\tfrac12\sigma^2_{\text{OK}}$ lifts the median to "the strictly unbiased MMSE conditional arithmetic mean" (L217), which is the Simple Kriging story; with an unknown mean it no longer holds in the same sense. Say "unbiased on average" and drop "UMVUE," or prove it.
- **Beat 2 is missing.** The recipe is five steps: *check that the logs look Gaussian (normal Q–Q plot) → compute the variogram of the logs → krige the logs → add $\tfrac12\sigma^2$ (and the multiplier term for OK) → back-transform.* A student cannot currently find these in one place. Add the list.
- **Beat 3 repeats itself.** The "loses 52% of the mass" example appears at L228 and again at L233, with the same numbers ($\exp(-0.75)=0.472$, which I confirmed). Keep the FAQ version. The accurate caveat is that the 52% applies to nodes far from the data ($\sigma_{\text{SK}}^2\approx\sigma_Y^2$), not to the whole map. L228 reads as if it does.
- **Missing diagnostic.** The chapter warns (L235) that the back-transform "can explode." Give the check: compare the back-transformed estimate with $\exp(\widehat Y)$; if their ratio $\exp(\sigma^2/2)$ is large, report that the result is dominated by the variance correction, not the data. Also note that variogram estimates of logs are themselves sensitive to the smallest values (a $\log$ of a near-zero outlier is a huge negative).
- **Missing link to §9.3.** Indicators are unchanged by monotone transforms: $\mathbb I\{\exp(Y)\le u\}=\mathbb I\{Y\le\ln u\}$. So a lognormal field has exactly the same indicator structure as a Gaussian one, including the destructuration of extremes. One sentence here sets up §9.3 and §9.6.

### §9.3 Indicator Random Fields and Spatial Threshold Geometry

**What works.** The moment calculations (Prop. 9.3.1 parts 1–3) are right. The mosaic factorization (L300–304) is correct and it is a lovely counterexample: indicator correlation independent of threshold. The ASCII figure (L245–260) shows the idea of thresholding well, though the lower half is hard to read.

**Issues**
- **Proposition 9.3.1(4) says "equals" where it should say "is at most" (L286).** $\gamma_I(\mathbf h;u)=\tfrac12P(\text{the indicator differs at the two ends})$. That is the probability of an *odd* number of crossings of level $u$ between $\mathbf s$ and $\mathbf s+\mathbf h$, which is not the probability of at least one crossing. The intuitive version, "half the probability that the level set separates the two points," is correct and lovely. "Intersects the line segment" is not.
- **Theorem 9.3.1's proof uses a wrong constant (L309).** The standard bivariate lower-tail asymptotic is $\dfrac{(1+\rho)^2}{2\pi\sqrt{1-\rho^2}\,y^2}\exp\!\left(-\dfrac{y^2}{1+\rho}\right)$. The chapter has an extra factor $\dfrac1{1-\rho}$. I computed the joint probability by one-dimensional numerical integration. For $\rho=0.5$, exact/chapter is 0.40, 0.47, 0.49 at $y=-4,-8,-12$ (heading to 0.5), while exact/standard is 0.81, 0.94, 0.97 (heading to 1). The conclusion does not change, because the exponent decides it.
- **A simpler proof avoids the black box.** $P(Y_1\le y,Y_2\le y)\le P(Y_1+Y_2\le 2y)=\Phi\!\left(y\sqrt{2/(1+\rho)}\right)$, since the sum is normal with variance $2(1+\rho)$. Because $\sqrt{2/(1+\rho)}>1$, this is exponentially smaller than $\Phi(y)$. Two lines, only the normal sum and a tail bound. I checked the inequality numerically ($\rho=0.5$, $y=-5$: joint $8.2\times10^{-10}$, bound $3.9\times10^{-9}$).
- **Give the reader numbers.** For $\rho=0.5$ I computed the indicator correlation $\rho_I$ at $y=-1,-2,-3,-5$: 0.28, 0.16, 0.059, 0.0029. A four-row table or a three-line plot makes "destructuration" visible, and it is much more persuasive than the limit statement.
- **Statement of (1) over-specifies the hypothesis.** "Continuous Gaussian random field" (L294) is not used; what the argument needs is $|\rho(\mathbf h)|<1$ at the lag in question. "Isolated singleton peaks" (L298) is true for point samples but for a continuous field the excursions shrink rather than become singletons. Say "small, scattered patches."
- **"Boolean models" are not partitions (L300).** Voronoi cells are, Boolean models overlap. Drop "Boolean."
- **Beats 2 and 3 are absent.** The practical question is "does my data look more like Gaussian or mosaic?" Give the check: compute indicator correlograms at the 10th, 50th, and 90th percentiles; if the extremes' ranges collapse, the multi-Gaussian family (lognormal, anamorphosis, DK) is plausible. If they do not, it is not. That diagnostic is the reason this theorem exists. It is currently stated as a theorem and never used.

### §9.4 Indicator Kriging (IK)

**What works.** Definition 9.4.1 sets SIK and OIK side by side, and the term-by-term anatomy (L350–375) has the right shape.

**Issues**
- **The anatomy has a wrong word.** L370 says $\widehat F$ evaluates "the posterior *exceedance* probability $P(Z\le u\mid\mathbf Z)$." That is the non-exceedance probability (the CDF). $1-\widehat F$ is the exceedance. A single wrong word in a section about probabilities is costly.
- **IK does not estimate $\operatorname{E}[I(\mathbf s_0;u)\mid\mathbf Z]$ (L59, L326, L370).** It is the best *linear* estimate of that probability from the indicators *at one threshold*. It discards the actual data values and ignores the other thresholds' indicators (cokriging would use them). The conditional expectation of the indicator, given all the data, is what you cannot compute without a model of the full joint distribution. Say so; this is the central trade-off of IK ("distribution-free" means "model only the pairwise indicator covariance, and accept a linear predictor").
- **"Bypassing all parametric distribution assumptions" (L61, L243) is too strong.** IK avoids assuming a *marginal* distribution but still assumes stationarity of each indicator and a valid variogram model at each of $K$ thresholds. State the assumption exchanged.
- **Terminology.** "Universal unbiasedness condition" (L375) is the ordinary condition $\sum\lambda_i=1$. "Universal" has another meaning in this book (UK, Chapter 6).
- **Multiplier sign.** The OIK system (L344–348) uses $+\mu$ with a semivariogram matrix, and the variance formula that goes with it is never given. Give $\sigma^2_{\text{OIK}}=\boldsymbol\lambda'\boldsymbol\gamma_{I,0}+\mu_I$ in this convention.
- **Median IK (Def. 9.4.2) contradicts §9.3 and should say so.** MIK assumes every threshold's indicator has the *median's* correlogram. Theorem 9.3.1 just showed that, for Gaussian-type fields, extremes have much shorter range. MIK is therefore the mosaic assumption. Say so; it tells the student when to use it.
- **Nothing shows IK producing a number.** This is the biggest gap in §9.4. A five-sample transect, three thresholds, hand-solved OIK at one target, would take half a page, give you the raw vector $(\widehat F(u_1),\widehat F(u_2),\widehat F(u_3))$, and set up §9.5 with a violation the student has seen arise.
- **The payoff is missing.** Why estimate a CCDF at all? Because you can read off $P(Z>z_c)$, the **E-type** (mean of the CCDF), and quantiles. None of these is shown, and the "between thresholds" and "tail" interpolation needed to get a full distribution is not mentioned. Add the three read-offs, with the formula $\widehat{\operatorname{E}}[Z]=\sum_k z_k^*\,[\widehat F(u_k)-\widehat F(u_{k-1})]$ and its verbal meaning.
- **Indicator variogram modeling is not mentioned.** $K$ thresholds means $K$ variograms. Say how many thresholds are typical (often 5–9 deciles) and what happens at the extreme ones (very few indicator ones, so noisy variograms).

### §9.5 Order Relation Violations and Monotonicity Corrections

**What works.** The motivation (L394–409) is clear, and the picture of "negative probability mass in $(u_k,u_{k+1}]$" is exactly right. Exercise 9.2 is correct and well-laid-out. The PAVA code (L441–499) is a faithful implementation of what the theorem says.

**Issues**
- **PAVA as written (clamp first) is not the projection.** Theorem 9.5.1 step 2 (L422) and the code (L461) clamp to $[0,1]$ and *then* pool. The correct, and simple, order is pool *then* clamp. Counterexample: $\mathbf y=(1.3,0.9)$, equal weights. The chapter's order gives $(0.95,0.95)$ with squared distance 0.125. Pooling first gives $(1.1,1.1)$, which clamps to $(1,1)$ with squared distance 0.10. A second case, $\mathbf y=(0.2,-0.4,0.1)$: chapter gives $(0.1,0.1,0.1)$, error 0.26; correct is $(0,0,0.1)$, error 0.20. On 300 random vectors, the chapter's order disagreed with a direct constrained least-squares solve in 222 cases (max difference 0.58), while pool-then-clamp disagreed in 0. Exercise 9.2 happens to be unaffected (I checked it against the solver), which is why the error is easy to miss.
- **"Preserves the mean probability across pooled thresholds" (L515)** is true for pooling, but the clamp step breaks it (raw mean 0.625 vs. 0.550 after the chapter's procedure for $\mathbf y=(0.2,0.1,0.9,1.3)$). With pool-then-clamp it holds *within* each pooled block.
- **The ad-hoc comparison is a straw man (L515).** A one-sided upward sweep does bias upward, but the practice in the literature (as I recall, e.g. GSLIB) averages an upward and a downward sweep. On $(0.1,0.5,0.3,0.4,0.9)$: up gives $(0.1,0.5,0.5,0.5,0.9)$, down gives $(0.1,0.3,0.3,0.4,0.9)$, their average is $(0.1,0.4,0.4,0.45,0.9)$, and PAVA gives $(0.1,0.4,0.4,0.4,0.9)$. Present PAVA as the principled version of "average the two sweeps," not as a fix for a broken heuristic.
- **The weights $w_k$ are never given meaning (L412).** Equal weights are used in the exercise. What would unequal weights be? Inverse kriging variance, so that thresholds estimated with more certainty are moved less, is the natural answer. Say it.
- **The worked example shows one merge only.** Add a case that *cascades*, e.g. $\mathbf y=(0.5,0.4,0.3,0.9)$: pool $(0.5,0.4)\to0.45$, now $0.45>0.3$, pool again $\to0.40$ (all three). That backtrack step is the reason the algorithm is called "pool adjacent violators" and is the line students stumble on (L429, L487–488).
- **The proof (L433–434) is not a proof.** "PAVA maintains these KKT … conditions at every pooling step" is an assertion. The two-element picture does the job: if $y_1>y_2$, the closest point with $x_1\le x_2$ lies on the line $x_1=x_2$, namely the weighted mean. Do that, then say the general case is the same argument repeated.
- **Illustrative example in Beat 3 (L513) cannot show what it claims.** With identical indicator data at both thresholds, the kriging weights differ *only* because of the two fitted variograms, but no variogram is shown, so the student cannot see a negative weight happen. Use the transect example suggested for §9.4.
- **Missing practical points:** (1) violations are typically small and concentrated at extreme thresholds; report how often they occur (a diagnostic: count and mean size of violations per map); (2) after correction, the CCDF is a step function that still needs interpolation to be a distribution.

### §9.6 Gaussian Anamorphosis via Hermite Polynomial Expansions

**What works.** The orthogonality proof (L564–568) is a model: define Hermite polynomials through derivatives of the Gaussian density, integrate by parts, done. I checked Rodrigues' $H_4$, the recurrence, and Mehler's formula numerically. The generating-function proof of Mehler (L576–586) is short and correct.

**Issues**
- **Hermite-as-classical-statistics (L21) has no payoff here either.** Say in two sentences what a Hermite coefficient *is*: $\psi_n=\operatorname{E}[\Phi(Y)H_n(Y)]/n!$, the regression coefficient of $Z$ on the orthogonal predictor $H_n(Y)$. That is L641 of the code, not stated in the text. Exercise 9.3 (L891) cites "By Definition 9.6.1" for this formula, but Definition 9.6.1 does not state it.
- **Some proof steps are left as "follow directly."** The recurrence and derivative identity (L568) are not derived from the generating function, and the statement that the Hermite family is *complete* (needed for the expansion of an arbitrary $\Phi$ and for Mehler) is never mentioned. For this audience, one sentence ("any function with finite variance can be expanded this way; we will not prove it") is enough.
- **The code does not work as a transform.** I ran the chapter's function on samples from a lognormal(0,1), where the true coefficients are $\psi_n=e^{1/2}/n!$ exactly.
  - **Coefficients are noisy.** At $n=200$, $\psi_3,\psi_4,\psi_5$ are $0.233, 0.023, -0.013$ versus true $0.275, 0.069, 0.014$. At $n=10{,}000$ they are $0.269, 0.064, 0.011$.
  - **The fitted $\Phi$ ripples inside the data range.** On $[-2.5,2.5]$ it decreases on 72 of 1,400 grid steps at $n=200$ (124 at $n=1{,}000$). A transform must be monotone.
  - **The bisection inverse (L668–676) assumes monotonicity, so it fails at the extremes.** For the largest sample ($z=14.8$ at $n=200$), `forward_transform` returns $5.05$ where the normal score is $2.81$; at $n=1{,}000$, $4.95$ versus $3.29$. The largest error over the sample is about 2.2 Gaussian units, while the median error is 0.03. The extremes are exactly where lognormal-type data need the transform to work.
  - **A degree-15 polynomial blows up outside the data.** At $y=\pm6$ the fitted function takes values like $-235$ and $-48$ (checked), so the $[-6,6]$ bisection bracket is not safe.
  - **Recommended fix:** do not invert the polynomial. Use the empirical normal-score transform (rank → $\Phi^{-1}$) for $Z\to Y$, and use the Hermite series only where it is needed, in DK and for expectations. Plot the fitted $\Phi$ against the empirical quantile function as a standard diagnostic, and report the **variance captured**, $\sum_{n\ge1}n!\,\psi_n^2$ against the sample variance, as the criterion for choosing the degree. (For that sample at $n=200$ the series captured 4.7 of a sample variance of 7.0.)
- **The docstring and comments do not match the code.** The docstring says "via Gauss–Hermite numerical quadrature" and the parameter is `n_gauss_nodes`, but L643–650 is a midpoint Riemann sum on $[-5,5]$. L627 says "Hart's algorithm" but the function uses `math.erf`. Both are harmless but confusing to a student reading closely.
- **Tails are silently truncated.** `quantile_z` clamps $p$ to $[0.5/n,\,1-0.5/n]$ (L619), so nothing beyond the sample's extremes is ever represented. Any honest treatment says the tail model is the weak point of anamorphosis, and that real software lets you pick a tail extrapolation.
- **Declustering is ignored.** The sample quantile function treats clustered samples as equally informative. A one-sentence warning suffices.
- **No Beat 3.** Add the two diagnostics above (monotonicity of $\Phi$, variance captured) and a note that the Gaussian assumption is checked in §9.3's indicator correlograms.

### §9.7 Disjunctive Kriging (DK) and Isofactorial Models

**What works.** The theorem's structure is correct: orthogonal polynomials of different degree are uncorrelated at every lag (Mehler), so the normal equations separate degree by degree. I checked the weights and the variance formula against a simulation. That is the chapter's most elegant idea, and the anatomy block (L727–735) conveys it well.

**Issues**
- **The symbol $n$ means two things in one theorem.** $\boldsymbol\lambda^{(n)}=(\lambda_1^{(n)},\dots,\lambda_n^{(n)})'$ (L706) has $n$ as the Hermite degree (superscript) and the number of data (subscript range and sum limit) at once. Written $\sum_{j=1}^n\lambda_j^{(n)}[\rho(\cdot)]^n$ (L708), a reader cannot tell which $n$ is which. Use $k$ for the degree and keep $n$ for the sample size, as in the rest of the book. The same collision recurs in the statement of the variance (L713) and in the proof (L719–723).
- **$\psi$ means two things (L702).** It is the target function and its Hermite coefficients $\psi_n$. Call the coefficients $c_k$.
- **The motivation is missing: when would you use DK rather than the exact answer?** Under a fully multivariate Gaussian $\mathbf Y$ (which is what "Gaussian anamorphosis" assumes), the conditional expectation $\operatorname{E}[\psi(Z_0)\mid\mathbf Z]$ is available in closed form, exactly as in Theorem 9.2.2. The reason DK exists is that it needs only **bivariate** Gaussianity (the isofactorial structure of Def. 9.7.1) and gives the best *additive* predictor. The chapter never says this and never mentions multi-Gaussian kriging. I simulated a lognormal field ($\sigma^2=1$, exponential correlation, three sites, one target; 600,000 draws, no error bars computed). The mean squared errors were: exact conditional mean 2.48, DK 2.57, linear SK on $Z$ 2.65, do-nothing 4.54. The theoretical DK variance from the chapter's formula, summed to degree 13, was 2.64. In this toy, DK improves a little on linear kriging and the exact Gaussian answer is a little better still. A student should see why that happens and when it does not. When data are not multi-Gaussian but bivariate Gaussianity is plausible, DK earns its keep.
- **The definition of DK and the theorem do not meet.** L687–691 defines DK as the best additive $\sum f_i(Z(\mathbf s_i))$, and the theorem decouples it only for the Hermite case. State in one sentence why the Hermite expansion makes "additive in the $Y_i$" the same as "additive in the $Z_i$" (anamorphosis is a monotone function applied to each datum, so an additive function of $Y_i$ is an additive function of $Z_i$).
- **Computational reality is not discussed.** There is one $n\times n$ system per degree, with entries $\rho^k$. For large $k$ those matrices approach the identity (as $\rho<1$), so the weights approach "use only the nearby points." This is a nice intuition, and the chapter's anatomy (L735) hints at it. State the practical conclusion: a truncated degree, and a check that the retained variance $\sum_k k!\,c_k^2$ is nearly all of $\operatorname{Var}(Z)$.
- **DK inherits the destructuration of extremes.** An isofactorial Hermite model is a Gaussian structure, so its high-order terms die out quickly in space. That is precisely §9.3's Theorem 9.3.1. One sentence connects the chapter's start to its end, and tells the student when DK is the wrong tool (mosaic-like data).
- **Beats 2 and 3 are absent.** A worked 3-point example is half a page: compute $\mathbf R^{(1)},\mathbf R^{(2)},\mathbf R^{(3)}$, solve, assemble. The code can wait.

### §9.8 Analytical Exercises

**Exercise 9.1** (lognormal back-transform) is correct throughout, including the interval, and it is a good template for the whole chapter. **Exercise 9.2** (PAVA) is arithmetically correct and matches a direct solve; its single-merge solution never exercises the backtrack, and the step "(c) KKT verification" is a stationarity check on one block, which is fine but uses the heavy name. **Exercise 9.3** (indicator Hermite coefficients) is correct: I confirmed $\psi_n=-H_{n-1}(y_u)\phi(y_u)/n!$ numerically for $n=1,\dots,5$ at $y_u=0.5$, and the median-case series $\tfrac12-\phi(0)\,[H_1-\tfrac16H_3+\tfrac{3}{120}H_5-\cdots]$ is right. It is also the nicest bridge between §9.4 and §9.7 in the chapter: DK of an indicator gives a CCDF estimate. Say so, and call out that it avoids order-relation violations by construction because it is built on a valid joint distribution (in the Gaussian case, a worthwhile claim; I did not test it).

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **5 / 10**

- **Beat pacing is front-heavy.** Beat 1 is present everywhere and often generous. Beats 2 and 3 are missing in most sections (table in §3). The student never gets to compute IK, never sees an order violation arise, and never sees DK produce a number.
- **The chapter reads as a catalogue.** Lognormal, indicator, anamorphosis, and DK are four methods with one sentence of connective tissue. The unifying story is available and strong: *each method is a different way to escape "linear kriging of the raw data is the wrong tool for nonlinear questions."* Lognormal = assume a model and use Gaussian conditioning. IK = distribution-free but linear and prone to LPM-style violations. Anamorphosis/DK = a Gaussian structure for any marginal. Say that on the first page and recap it in a final comparison table (assumption, output, cost, failure mode).
- **Visible "Beat 1/2/3" headings** appear in every section. Keep the rhythm, drop the labels, and let the transitions carry it ("So much for why; here is how to compute it").
- **The voice is strongest in the two callouts and weakest in §9.4, §9.6, §9.7.** Those three sections are nearly pure definition-theorem-proof.
- **No "From the Lecture Hall" boxes.** Candidates: Jensen's inequality (convex function lies above its chord) and the normal MGF before §9.2; the linear probability model before §9.4; isotonic regression as "least squares with an ordering constraint" before §9.5; orthogonal polynomial regression before §9.6.

### Dimension 2 — Algebraic transparency and scaffolding: **6 / 10**

- **Strong:** the lognormal proofs; the Rodrigues integration-by-parts; the Mehler generating-function argument; all three exercises' step-by-step arithmetic.
- **Needs work:** (a) the Mills-ratio proof (wrong constant, black box; replace with the two-line bound); (b) the PAVA proof ("KKT is maintained" is an assertion); (c) the Theorem 9.2.3 derivation is *absent*, and a four-line derivation from the variance identity would also have exposed the sign error; (d) §9.7's double use of $n$ makes the one decoupling theorem hard to read; (e) several results are introduced by "follows directly" (L568).
- **Anatomy blocks:** present for IK and DK, but the Lognormal anatomy (L212–218) makes the unsupported unbiasedness claims. The IK anatomy has the "exceedance" error.

### Dimension 3 — Structural consistency and forward referencing: **5 / 10**

- **Notation** collides in at least five places (§3: $\Phi$/$\phi$/$\psi$, $n$, $K$ vs. $k$). It matters here more than elsewhere, because the chapter uses Gaussian CDF, density, anamorphosis, and coefficients all at once.
- **The back-references that should exist are missing.** (i) Chapter 5 (kriging as conditional expectation in the Gaussian case, Lagrange multiplier convention). (ii) Chapter 8 §8.5–§8.7: everything in this chapter is at *point* support, but the decisions the chapter motivates (recoverable reserves, exceedance over blocks) are at block support. IK and DK as written predict point-support quantities. Say so and point to Chapter 8 for the change of support. (iii) Chapter 1 for the sample-mean/effective-size ideas, if used for the lognormal bias discussion.
- **Forward references that should exist:** Chapter 10 (conditional simulation is the standard remedy when you need the distribution of a *nonlinear functional of many nodes*, not just a pointwise CCDF; IK and DK give marginal answers at each node only), and Chapter 12 (spatial GLMs as the likelihood-based answer to the LPM problem in §9.1).
- **De-duplication:** the "52% of the mass" example appears twice (L228, L233); the "$\tfrac12\sigma^2$ is Jensen" explanation appears three times (L174–175 statement, L217, L227). Say it once, well.
- **Convention consistency:** two OK-type systems appear (L205 and L344) with the $+\mu$ convention; Chapters 5 and 7 use $-\mu$. One sentence at first use fixes it.

### Dimension 4 — Empirical and computational utility: **5 / 10**

- **Verified correct:** Exercises 9.1, 9.2 (the arithmetic), 9.3; Theorems 9.2.1 and 9.2.2; Proposition 9.3.1 parts 1–3; the mosaic factorization; Mehler's formula; the isofactorial decoupling and its variance formula (simulation agrees to within its noise: 2.64 predicted vs. 2.57 simulated; I computed no error bar).
- **Verified wrong or fragile:** PAVA order (222/300 mismatches); Theorem 9.2.3 sign (12% bias in the toy); the Mills constant; Hermite code monotonicity and inverse (2+ units of error at extremes); Prop. 9.3.1(4) "equals."
- **No running data and no truth to compare to.** Nothing in the chapter is checked against anything. The best remedy is also the cheapest: simulate a lognormal field (so the exact CCDF at the target is known in closed form from Theorem 9.2.2), sample it, and compare (a) lognormal kriging, (b) IK with three thresholds before and after PAVA, (c) DK, and (d) the exact answer. A single table (E-type mean, $P(Z>z_c)$, 90th percentile, versus truth) shows every method's strengths and failures and makes the chapter's comparison concrete.
- **Code optional, but algorithms are not.** The PAVA algorithm needs the cascade example and a corrected order. The anamorphosis fitting procedure needs a clear statement of what it does (quantile function → normal CDF → project onto Hermite polynomials by quadrature) and its failure modes. The IK procedure needs a half-page by-hand example. DK needs a three-point numerical walk-through.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)

1. **PAVA order (L422, L461; Beat 3 L515).** Pool first, then clamp to $[0,1]$. Update the theorem statement, the code (move the clamp to the end), and the "preserves the mean" sentence. Exercise 9.2 stays as is. Add a cascading example.
2. **Theorem 9.2.3 (L200–218).** Fix the sign under the chapter's own convention (or switch convention and say so); replace "exact UMVUE" and "conditionally unbiased" with "unbiased on average, $\operatorname{E}[\widehat Z]=m_Z$"; add the four-line derivation (the variance of $\widehat Y_{\text{OK}}$ is $\sigma_Y^2-\sigma_{\text{OK}}^2-2\mu_L$ in this convention).
3. **Theorem 9.3.1 proof (L307–316).** Replace the Mills-ratio expansion with the two-line bound $P(Y_1\le y,Y_2\le y)\le\Phi\!\left(y\sqrt{2/(1+\rho)}\right)$. If the asymptotic is kept, the correct constant has no $1/(1-\rho)$.
4. **Proposition 9.3.1(4) (L286).** Replace "equals half the probability that the level set intersects the segment" with "half the probability that the two points fall on opposite sides of the level $u$."
5. **IK anatomy (L370).** "exceedance" → "non-exceedance (cumulative)." Add the sentence that IK is the best *linear* estimate of that probability from threshold-$u$ indicators only, not $\operatorname{E}[I\mid\mathbf Z]$ (L59, L326, L370).
6. **Hermite code (L592–679).** Do not invert the polynomial by bisection. Use the empirical normal-score transform for the forward map, add the monotonicity check, and report variance captured. Fix the docstring (Gauss–Hermite vs. midpoint) and the "Hart's algorithm" comment, or switch to genuine Gauss–Hermite nodes.
7. **Proposition 9.1.1 (L90–99).** Restrict the Jensen statement to convex $\psi$; do not equate $\psi(\operatorname{E}[Z_0\mid\mathbf Z])$ with $\psi(\widehat Z_{\text{SK}})$ unless the field is Gaussian; fix the "Jensen trap" example to say what actually fails (a step function applied to a smooth prediction) and explain the 45%.
8. **§9.7 notation (L700–725).** Separate the Hermite degree $k$ from the sample size $n$, and the coefficients $c_k$ from the target $\psi$. This is the single largest readability fix in the chapter.
9. **Remove measure-theory vocabulary (L5, L68, L694, L702).** Delete $(\Omega,\mathcal F,P)$, "complete probability space," "$L^2(\mathbb R,dF_0)$," and "measurable."

### Should fix (pace, flow, voice)

10. **Add the running synthetic example** described under Dimension 4. It supplies the missing Beat 2 and Beat 3 for §9.2, §9.4, §9.5, and §9.7 and doubles as a chapter-ending comparison table.
11. **Make the LPM bridge pay off.** In §9.1 name IK as "kriging the linear probability model." In §9.5 say order relation violations are the LPM's out-of-range predictions. In §9.7 or the closing note point to probit/logit-style alternatives (Chapter 12).
12. **Give §9.4 its payoff:** the hand-worked five-point example; read-offs of $P(Z>z_c)$, the E-type mean, and quantiles; a note on interpolation between thresholds and in the tails; how many thresholds are typical.
13. **Add the missing diagnostics:** Q–Q plot of the logs before lognormal kriging (§9.2); indicator correlograms at three quantiles, as a Gaussian-versus-mosaic check (§9.3); the violation count and size per map (§9.5); the monotonicity and variance-captured checks (§9.6); the retained-variance check (§9.7).
14. **State when to choose what.** Add a one-page comparison table: lognormal kriging, IK, MIK, DK. Columns: assumption, output, what it costs, where it fails. Include the sentence that DK is for bivariate-Gaussian data where the exact multivariate answer is unavailable, and mention multi-Gaussian kriging.
15. **Connect §9.2 to §9.3 and §9.6:** indicators are unchanged by monotone transforms, so lognormal and anamorphosed-Gaussian fields all destructure. Say it once, in §9.2.
16. **Replace the dedication paragraph (L5)** with a one-sentence promise.
17. **Weights in PAVA:** say what $w_k$ should be (inverse kriging variance is the natural choice).
18. **Fix the §9.5 proof** with the two-point picture instead of KKT language.
19. **Unify notation** ($\Phi$, $\phi$, $G$, $\psi$, $c_k$, $n$, $k$, $K$) and add entries to `notation_index.md`.
20. **Cross-reference** Chapter 8 (support), Chapter 10 (simulation as the remedy for functionals of many nodes), Chapter 12 (spatial GLMs), Chapter 5 (multiplier convention).

### Nice to have

21. Lecture Hall boxes (Jensen; normal MGF; LPM; isotonic regression; orthogonal polynomial regression).
22. A small figure for destructuration: indicator correlation versus lag at the 10th, 50th, and 90th percentiles for a Gaussian field, with the mosaic case as a flat reference. (For $\rho=0.5$, the values at $y=-1,-2,-3,-5$ are 0.28, 0.16, 0.059, 0.0029.)
23. A figure of the PAVA cascade (four bars merging into one) and one of a fitted $\Phi$ with its empirical quantile curve.
24. Declustering and tail-extrapolation remarks for the anamorphosis (one sentence each).
25. Remove the visible "Beat 1/2/3" headings; let the transitions carry the rhythm.

---

## 6. What to Keep Exactly As It Is

- The "conditional median versus conditional mean" explanation of the $\tfrac12\sigma^2$ term (L216–217, L227). The best paragraph in the chapter.
- Theorem 9.2.1 and its MGF proofs (L123–157), and Theorem 9.2.2 (L165–198).
- The mosaic factorization (L300–304) as a counterexample to destructuration.
- The Rodrigues integration-by-parts proof of orthogonality (L564–568) and the generating-function proof of Mehler (L576–586).
- The anatomy of the covariance decoupling (L727–735), especially the explanation of why a Kronecker delta is the engine of DK.
- Exercise 9.1 as the template for what a worked example looks like, and Exercise 9.3 as the bridge between indicators and Hermite expansions.
- The two "Wait—Why Did We Do That?" callouts as the voice for the missing beats.
