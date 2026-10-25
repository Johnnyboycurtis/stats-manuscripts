# Review: Chapter 1 — Stochastic Foundations of Spatial Linear Models

**File reviewed:** `ch01_foundations.md` (583 lines)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch01_exercises.md` was not reviewed. The two short exercises inside §1.7 are covered under Dimension 4.
**Review convention on code:** Python snippets are *optional*. I suggest them where they would help, but a section is not marked down for lacking one. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be: its inputs and outputs, why each step is correct, and where it can go wrong.

---

## 1. Executive Summary

**Grade: 6.5 / 10**

Chapter 1 has the right instincts and several passages a student would remember: the opening "How much is where?", the variance-inflation and effective-sample-size algebra, "we only ever get one realization of the Earth," and the nugget-discontinuity example. It is already almost free of measure theory. The chapter does not yet read like a teacher wrote it, for four reasons:

- Only §1.3 follows the Three-Beat rhythm, and the one algorithm in the chapter (Algorithm 1.1) is listed but not fully explained.
- The central idea of the chapter (the single-realization problem) is explained four separate times.
- Notation shifts under the reader's feet: `C(0)` and `σ²` each carry three different meanings.
- Several statements are stronger than the math supports, and the chapter gives no worked data example.

None of this needs a rewrite from scratch. Most of it is moving, trimming, or adding a few short blocks.

**Highest-value change:** explain why `Lw` has the right covariance (Algorithm 1.1, one line). **Optional but worth considering:** a small Python snippet in §1.1.3 that computes `n_eff` on a 1-D transect. I ran the numbers: 100 evenly spaced sites on [0, 10] with exponential correlation (scale 1) give `n_eff ≈ 5.6`. Quadrupling to 400 sites on the same transect still gives `n_eff ≈ 5.6`. That one result shows infill asymptotics better than the four paragraphs currently devoted to it.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes.** The README advertises Carathéodory extensions, Feldman–Hájek, and measure-theoretic proofs for this chapter, but the chapter does not contain them (a search for Feldman, Carathéodory, σ-algebra, and 𝓕 returns nothing). That is good news for the mission. The few remaining drift points are mild:

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L215 | "$L^2$ Hilbert space" appears only as a parenthetical note after the geometry bullets. | None. | This is the right treatment: geometry first, jargon as a footnote. Keep it. |
| L273–274 | "Kolmogorov Consistency Principle" stated without proof and without the word σ-algebra. | Low. | Keep the idea, but add a "From the Lecture Hall" box (see §4). Fix the overstatement noted below. |
| L512–513 | $\omega_i \in \Omega$ in FAQ 2, and "Day 1 ($\omega_1$) … Day ∞ ($\omega_\infty$)" in the L172–181 table. | Low. | Drop $\Omega$ from the FAQ. The table can say "Realization 1, 2, 3 …". Here it is also not just style but wrong (see §3, §1.6). |
| L486 | "set covariogram $K_{D_n}(\mathbf{h})$" in the proof of Theorem 1.6.1. | Medium. | A geometric-probability object the reader has not met. Replace with a change of variables (a substitution) written out in two lines. |
| L492–497, L501 | Microergodicity. | Medium. | Concept is fine at this level, but the definition given is not the standard one and the "microscope" explanation is imprecise. A Taylor expansion (allowed) replaces it. Details in §1.6. |

No Hankel/Bessel integrals, Lebesgue–Stieltjes measures, or hypergeometric functions appear. The one place a special function would sneak in is the Matérn/exponential references, which are handled in the forward-reference notes below.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

Beat 2 is read as "how you would compute it". For a section with a core algorithm, that means a clear step-by-step explanation. Code is a bonus.

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 1.1 | Yes (strong bridge in 1.1.3) | Formulas given; no core algorithm needed (a snippet is optional) | Partial (FAQ 1 "practical catch") |
| 1.2 | Yes (geometry list, rows-vs-columns table) | Not needed | **Missing** |
| 1.3 | Yes | Algorithm 1.1 present but under-explained (see §1.3) | Yes |
| 1.4 | **Missing** (opens straight on definitions) | Not needed | Yes (nugget example) |
| 1.5 | Yes ("one realization of the Earth") | Not needed (a short demo is optional) | Yes |
| 1.6 | Yes (short) | Not needed | Yes |
| 1.7 | n/a | n/a | n/a |

Only §1.3 satisfies the full contract, and even there the algorithm explanation has gaps. The chapter has no Python blocks; Algorithm 1.1 is plain-text pseudocode. That is acceptable. What needs fixing is the missing explanation (below), not the format.

### §1.1 The Need for Spatial Foundations and the Three-Layer Decomposition

**What works**
- The opening paragraph (L3) is exactly the voice the book should have throughout.
- The term-by-term anatomy blocks (L34–38, L72–76) are the right scaffolding.
- FAQ 1 (L78–91) answers a question the reader would really ask, in plain language.
- The 100-samples-equals-2-independent-samples example (L139–143) is memorable.

**Issues**
1. **Stationarity is assumed before it is introduced.** From L37 on, the chapter writes $C(\mathbf{h})$ and $C(\mathbf{s}_i - \mathbf{s}_j)$, which is shorthand that only makes sense under stationarity (§1.5). A reader who notices this will wonder whether they missed a definition. One sentence ("for now we assume the covariance depends only on the separation; §1.5 makes this precise") fixes it.
2. **`C`, `V`, and `σ²` are used inconsistently.**
   - L37, L74: $C(\mathbf{h}) = \operatorname{Cov}(\delta(\mathbf{s}),\delta(\mathbf{s}+\mathbf{h}))$, so $C(\mathbf{0}) = \sigma^2$ (the partial sill, excluding the nugget).
   - L75: $V_{ii} = \sigma^2 + \tau^2$.
   - L106 then writes $\frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1} = \frac{1}{n^2}\sum\sum C(\mathbf{s}_i-\mathbf{s}_j)$. That equality is false when $\tau^2 > 0$, because $\mathbf{V} = \mathbf{C} + \tau^2\mathbf{I}$.
   - L404: $C(\mathbf{0}) = \sigma^2 + \tau^2$. L449: $C(\mathbf{0}) = \sigma^2$.
   - Exercise 1.1 (L545): $\sigma^2 = C(\mathbf{0})$, now meaning the *total* point variance.

   A careful student will lose time reconciling these. **Pick one convention.** The simplest: call the total point variance $\sigma_Z^2 = \sigma^2 + \tau^2$ and let $C_Z(\mathbf{h})$ be the total covariance (as already done at L366). Use $\mathbf{V}$ and $\sigma_Z^2$ throughout 1.1.3, §1.5 and Exercise 1.1. Alternatively, say once in 1.1.3 "for clarity we set $\tau^2 = 0$ here."
3. **"Must be symmetric positive definite" (L73) is stated without a reason.** Two fixes:
   - It should read *positive semidefinite* in general (strictly definite only if no exact linear dependence among sites).
   - The reason is a one-liner the reader already has the tools for: $\operatorname{Var}(\mathbf{a}'\mathbf{Z}) = \mathbf{a}'\mathbf{V}\mathbf{a} \ge 0$ for every $\mathbf{a}$. This is the best place in the chapter to pay off the geometry in §1.2 ("variances are squared lengths, so they cannot be negative") and to motivate why Chapter 2 needs a theory of valid covariance functions.
4. **`δ ∼ (0, C)` notation (L63–64) is not defined** for a reader who has only seen $\mathcal{N}(\cdot,\cdot)$. Add: "$\mathbf{Y} \sim (\boldsymbol{\mu}, \boldsymbol{\Sigma})$ means mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$, with no distributional claim."
5. **Claims in 1.1.1 (L16–19, L21)** overreach slightly:
   - "$\operatorname{Cov} \ne 0$ for $\mathbf{s}_i \ne \mathbf{s}_j$" — many valid covariances are exactly zero beyond a finite range. Prefer "typically positive and decaying with distance."
   - "Severely biased downward" is asserted, then only the *sign* is proved in Exercise 1.1. Either soften to "biased downward" or connect it to the quantitative result (see Exercise 1.1 below).
6. **The "range" label (L70, L76) needs care.** For the exponential model used at L501, $\phi$ is the *e-folding scale*, not the distance at which correlation vanishes (the "practical range" is about $3\phi$). Say "range parameter" and note the convention once.
7. **n_eff example is unrealistic as written (L139–143).** An average pairwise correlation of 0.5 over 100 sites is extreme, and the phrase "convey no more information than two independent measurements" is true only *for estimating the mean by the plain average*. Two upgrades:
   - Use a number a student could believe (e.g., $\bar\rho = 0.1$ gives $n_{\text{eff}} \approx 9.2$).
   - Add the one-line observation that as $n \to \infty$, $n_{\text{eff}} \to 1/\bar\rho$, which previews the infill result.
8. **A missed unification.** Dividing the exercise's $\operatorname{E}[S^2]$ by $\sigma^2$ gives $\operatorname{E}[S^2] = \sigma^2(1-\bar\rho)$, which sits right next to $\operatorname{Var}(\bar Z) = \frac{\sigma^2}{n}[1 + (n-1)\bar\rho]$. Showing both side by side with the same $\bar\rho$ would be a very satisfying "aha," and it ties 1.1.1, 1.1.3 and Exercise 1.1 together.
9. **L145–156 ("Asymptotic Divergence") duplicates §1.6** (see Cross-cutting issue 1 below).

**Optional snippet:** a short dependency-free example right after the $n_{\text{eff}}$ derivation would make the idea concrete, though the section stands without it:

```python
import math

def n_eff(coords, sigma2, phi):
    """Effective sample size for the sample mean, exponential covariance (tau^2 = 0)."""
    n = len(coords)
    total = sum(sigma2 * math.exp(-abs(a - b) / phi) for a in coords for b in coords)  # 1'V1
    return n * n * sigma2 / total

sites_100 = [10 * i / 99 for i in range(100)]
sites_400 = [10 * i / 399 for i in range(400)]
print(n_eff(sites_100, 1.0, 1.0))   # ~5.6
print(n_eff(sites_400, 1.0, 1.0))   # ~5.6  -- four times the data, same information
```

I ran this logic; the results are about 5.6 for both. The "four times the data, same information" printout is the infill lesson in two lines. **Caution:** this uses the exponential covariance before it is formally introduced (Ch 2). Either note that, or use a one-line "correlation drops off exponentially with distance" sentence.

### §1.2 Spatial Random Fields and the Geometry of Covariance

**What works**
- The rows-vs-columns table (L171–182) is exactly what a student needs. "Looking down a column vs. across a row" is a clear, concrete image.
- The geometry list (L188–213) treats $L^2$ as intuition rather than axioms, which is the mission.
- The Hilbert-space name is demoted to a footnote (L215).

**Issues**
1. **Overstatement at L213.** "Finding the BLUP ... is geometrically identical to dropping a 90° perpendicular ... The prediction error is orthogonal (uncorrelated) to every observed variable."
   - This is true for *simple kriging* (known mean).
   - For ordinary kriging (unknown mean, the usual "BLUP"), the unbiasedness constraint means the prediction error has equal covariance $m$ (the Lagrange multiplier) with every observation, not zero.
   - Fix: say "for the simple case where the mean is known" and flag that unknown-mean prediction adds one constraint (Chapter 5).
2. **"Kriging" is used (L88, L213, L503) before any definition.** One sentence on first use ("kriging is spatial best linear prediction; Chapter 5") suffices.
3. **The geometry is not used again in Chapter 1.** The single best payoff for it, $\operatorname{Var}(\sum a_i Z_i) = \mathbf{a}'\mathbf{V}\mathbf{a} = \|\sum a_i Z_i\|^2 \ge 0$, explains why $\mathbf{V}$ must be positive semidefinite (see §1.1 issue 3). A tiny worked two-site example would also help: predicting $Z(\mathbf{s}_0)$ from one neighbor gives slope $\lambda = \operatorname{Cov}/\operatorname{Var}$, the *same* number as simple-linear-regression slope. That is the bridge from OLS to kriging in one line.
4. **Beat 2 and Beat 3 are missing.** A natural Beat 3: "what if two sites have correlation exactly 1 (or two sites coincide)?" — $\mathbf{V}$ becomes singular, which seeds §1.3's Cholesky discussion.
5. **Minor.** The table's last row "Day ∞ ($\omega_\infty$)" introduces probability-space notation for no benefit; "Realization 1, 2, 3, …" says the same thing. The "stock market" framing is announced as an analogy but is not referred to again except "portfolio" (L189); fine to keep, but ensure it earns its space.

### §1.3 Finite-Dimensional Distributions and Gaussian Random Fields

**What works.** The only fully three-beat section. The Gaussian marginal-consistency argument (L267–271) is a good example of explaining a deep-sounding idea (Kolmogorov consistency) with a fact the reader already knows (marginals of a multivariate normal are normal).

**Issues**
1. **Principle 1.3.1 (L274) overclaims.** "Choosing a valid spatial mean function and a valid positive definite covariance function completely and uniquely specifies the entire continuous random field." That is true for *Gaussian* fields (as Definition 1.3.2 says correctly), but **false for general fields**: many non-Gaussian fields share the same mean and covariance. Also, "unique" is only true of the finite-dimensional distributions (the law of the process), not of path properties such as continuity (§1.4 depends on this distinction). Suggested rewrite: "For Gaussian fields, mean and covariance function specify every finite-dimensional distribution, so they specify the field's probabilistic behavior. For other fields, they do not."
2. **Beat 2 intro (L284) credits Kolmogorov wrongly.** Generating one finite vector from $\mathcal{N}(\boldsymbol{\mu}, \mathbf{V})$ needs only that the multivariate normal exists, not Kolmogorov's theorem. Kolmogorov is what lets us talk about the whole field and know that simulated grids at different resolutions are mutually consistent. Say that instead; it is a more accurate and more interesting statement.
3. **Algorithm 1.1 is the chapter's one core algorithm, and its explanation has gaps.**
   - Step 5 never shows *why* $\mathbf{L}\mathbf{w}$ has the right covariance. It is one line: $\operatorname{Var}(\mathbf{L}\mathbf{w}) = \mathbf{L}\,\operatorname{Var}(\mathbf{w})\,\mathbf{L}' = \mathbf{L}\mathbf{L}' = \mathbf{V}$. This is the Cholesky payoff and belongs right under the algorithm. (Also say what $\mathbf{L}$ *is*: the matrix square root that Cholesky provides, the same way $\sigma$ scales a standard normal to get $\sigma Z$ in one dimension. That is the classical-to-spatial bridge for this algorithm.)
   - Step 2 is ambiguous about the nugget: whether `cov_fn(0)` returns $\sigma^2$ or $\sigma^2 + \tau^2$ changes the simulated field. State it explicitly.
   - Say what the *outputs* mean: one draw is one possible "map" of the field at those sites. Repeating steps 4–7 gives a new map each time.
4. **Optional:** the algorithm is plain-text pseudocode. A runnable Python version is not required, but if you want one, a "check that the empirical covariance of many draws matches $\mathbf{V}$" ending would be a nice confirmation that the algorithm works.
5. **Beat 3 overstates (L317).** "Adding $\tau^2 \mathbf{I}_n$ ... guarantees numerical stability." It improves conditioning: $\kappa(\mathbf{V} + \tau^2\mathbf{I}) \le (\lambda_{\max} + \tau^2)/\tau^2$. If $\tau^2$ is tiny, that bound is huge. Teach students the bound; it is a perfect "eigenvalue shift" fact at the right level.
6. **Another forward reference:** "positive definite" (L274) is used before it is defined (see §1.1 issue 3).

### §1.4 Continuity of Spatial Random Fields

**What works.** The nugget-discontinuity example (L364–376) is excellent: it turns an abstract limit into "the covariance has a jump of size $\tau^2$ at zero, so $\operatorname{E}[(Z(\mathbf{s}+\mathbf{h})-Z(\mathbf{s}))^2] \to 2\tau^2$." The Cauchy–Schwarz proof is at the right level.

**Issues**
1. **No Beat 1.** The section opens with "continuity can be defined in multiple modes" and goes straight to definitions. Ask the student's question first: "If I zoom in on a map, does it look smooth or jagged?" Why should a student who has never thought about this care? Because smoothness at the origin controls how much a nearby observation tells us — which drives the shape of every covariance model in Chapter 2. **Move the nugget example up as the motivation**, then give the theorem as the tidy general statement.
2. **A magic step at L349.** "Because both the variance and squared mean difference are non-negative, the sum vanishes ... if and only if $\mu(\mathbf{s}) \to \mu(\mathbf{s}_0)$, $C(\mathbf{s},\mathbf{s}) \to C(\mathbf{s}_0,\mathbf{s}_0)$, and $C(\mathbf{s},\mathbf{s}_0) \to C(\mathbf{s}_0,\mathbf{s}_0)$." Non-negativity gives only "sum of two non-negatives → 0 means each → 0." It does *not* explain why $\operatorname{Var}(Z(\mathbf{s}) - Z(\mathbf{s}_0)) \to 0$ forces the *diagonal* term $C(\mathbf{s},\mathbf{s})$ to converge. That step needs the reverse triangle inequality for standard deviations: $\left|\operatorname{sd}(Z(\mathbf{s})) - \operatorname{sd}(Z(\mathbf{s}_0))\right| \le \operatorname{sd}(Z(\mathbf{s}) - Z(\mathbf{s}_0))$. State it (it is a one-line consequence of the geometry in §1.2!) and the proof becomes airtight.
3. **Stationary simplification skipped (L349).** Show $\operatorname{E}[(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))^2] = 2(C(\mathbf{0}) - C(\mathbf{h}))$ explicitly. This is the formula that Chapter 2 will keep returning to.
4. **The nugget example's last sentence** (L376) is good but should say plainly: *mean-square discontinuity is not an error of the model; it is what measurement error looks like in the covariance.*
5. **Missing caveat:** mean-square continuity does not imply sample-path continuity (the third mode in Definition 1.4.1). A sentence is enough; students will otherwise assume the three modes are interchangeable.

### §1.5 Strict, Second-Order, and Intrinsic Stationarity

**What works.** "We only ever get one realization of the Earth" (L387) and the "spatial pseudo-replicates" idea (L391) are the heart of the chapter's pedagogy. The Brownian-motion counterexample is a good choice.

**Issues**
1. **The chapter's own model is not stationary, and the text never says so.** Definition 1.5.2 requires a *constant mean* (L402), but §1.1.2 introduced $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$, which varies with $\mathbf{s}$. A careful reader will ask how these fit. Answer: stationarity is a statement about the *residual* field $\delta$ (+ noise) after removing the trend; the full $Z$ is stationary only when $\mathbf{X}\boldsymbol{\beta}$ is constant. **This is the single most important clarity fix in §1.5.** Without it, the Ch 6 trend-estimation story feels contradictory.
2. **Strict stationarity does not sit "inside" second-order (L431–447 diagram).** A strictly stationary field need not have finite variance (e.g., a field of i.i.d. Cauchy variables), so strict stationarity does not imply second-order stationarity. The correct statement: *strict stationarity plus finite second moments* implies second-order stationarity; for **Gaussian** fields the two notions coincide. Fix the diagram caption and add a sentence of reasoning (a Gaussian field is determined by its mean and covariance, §1.3).
3. **Notation slips in Definition 1.5.2.** Item 2 introduces $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$ although the item is stated with $\mathbf{s}$ and $\mathbf{s}+\mathbf{h}$. Item 3 says $C(\mathbf{0}) = \sigma^2 + \tau^2$, while L449 says $C(\mathbf{0}) = \sigma^2$ (see §1.1 issue 2).
4. **Beat 1 promises a payoff it does not deliver.** The "pseudo-replicates" idea is announced, but the chapter never shows the estimator it makes possible (average of $(Z(\mathbf{s}_i)-\bar Z)(Z(\mathbf{s}_j)-\bar Z)$ over pairs at lag $\approx \mathbf{h}$). Even one sentence of what the averaging looks like, with a pointer to Chapter 3, completes the thought.
5. **Brownian motion is defined on $s \ge 0$ (L428)** while Definition 1.5.3 is stated for $\mathbb{R}^d$. Use two-sided Brownian motion, or say the definition is restricted to the domain. Also write $\gamma(h) = \tfrac12|h|$.
6. **Word choice at L450:** "topological elevations" should be "topographic."
7. **Optional demonstration.** Simulate (with Algorithm 1.1) a stationary exponential field and a Brownian path on a grid, then compare the sample variance of the first half with the second half. Stationary: similar. Brownian: wildly different. That would show the difference between 1.5.2 and 1.5.3 without a single symbol, but a picture or a short verbal example would also do.
8. **Isotropy is never named.** Algorithm 1.1 uses a covariance that depends only on distance; that is isotropy, an extra assumption beyond stationarity. A short note avoids a surprise in Chapter 2.

### §1.6 Spatial Asymptotics: Increasing-Domain vs. Infill

**What works.** The two-regime contrast is important and correctly stated. FAQ 2's conclusion ("sampling more densely in a fixed region never reduces variance to zero") is right.

**Issues**
1. **FAQ 2, point 1 (L512) is misleading.** "Every additional observation draws an independent outcome $\omega_i \in \Omega$." In the standard i.i.d. model there is a *single* outcome; each $Y_i$ is a separate random variable on the same probability space, and the independence is *between* variables. Spatial data are also one realization. The true contrast is: independent observations each carry *fresh* information, while neighboring spatial observations mostly re-measure the same underlying surface. A student trained on Casella & Berger would object to the current wording. Suggested rewrite: "Ten thousand independent sensors have ten thousand independent errors, and averaging cancels them. Ten thousand sensors in one square kilometer all measure the same hill; averaging them pins down that hill very well but tells you nothing new about whether this region is higher than average."
2. **Theorem 1.6.1 proof has two magic steps (L485–490).**
   - The covariogram bound $\iint C(\mathbf{s}_1-\mathbf{s}_2)\,d\mathbf{s}_1 d\mathbf{s}_2 = \int C(\mathbf{h})K(\mathbf{h})\,d\mathbf{h}$ is never derived. Show it with the substitution $\mathbf{h} = \mathbf{s}_1 - \mathbf{s}_2$ and an indicator; then you do not need the word "covariogram."
   - "The double integral ... remains strictly positive": non-negativity is automatic ($\operatorname{Var} \ge 0$); strict positivity holds unless the average is degenerate, and the text should say so in a sentence.
3. **Theorem 1.6.1 mixes two objects.** §1.1.3 and the infill formula talk about the *discrete* mean of $n$ points; the theorem is about the *integral* average. Say that the discrete mean converges to the integral average as sampling gets dense (needs mean-square continuity — which the nugget breaks for the noise part, but the noise averages out like $\tau^2/n$).
4. **Definition 1.6.2 (microergodicity, L492–497) is not the standard definition.** As written it is a consistency property ("variance of an estimator → 0 under infill"). The usual definition is about whether different parameter values produce *mutually singular* distributions for the field on the bounded domain; consistent estimability is a consequence. For this audience a good compromise: define it informally as "a parameter (or combination) that dense data *can* pin down," and say the formal definition lives in the research literature. Avoid presenting the consistency property as the definition.
5. **The microscope analogy (L501–502) can be replaced with an equation a student can check.** For the exponential covariance:
   $$\gamma(h) = \sigma^2\left(1 - e^{-h/\phi}\right) = \frac{\sigma^2}{\phi}\,h + \mathcal{O}(h^2)$$
   So dense short-lag data identify the *slope at the origin*, $\sigma^2/\phi$, and nothing else. This is a Taylor expansion (explicitly permitted) and it is the actual reason. It also previews Chapter 2's "behavior at the origin."
6. **Fact-check list (microergodicity).** Please verify before publishing; I am citing from memory:
   - Zhang (2004), *JASA*, shows that for the Matérn model in dimension $d \le 3$ the combination $\sigma^2\phi^{-2\nu}$ is microergodic.
   - Ying (1991) is usually cited for the exponential model in $d = 1$.
   - Anderes (2010), *Annals of Statistics*, shows that for $d > 4$, $\sigma^2$ and $\phi$ are separately consistently estimable.
   - The chapter says only "Zhang, 2004" with no dimension condition. Add the $d \le 3$ caveat and a full reference. The chapter also does not define $\phi$ as the scale parameter used by those papers (they often use $\alpha = 1/\phi$).
7. **The "prediction is harmless" claim (L503)** is a result about asymptotically equivalent predictors under equivalent Gaussian measures. True, but it is a lot of weight for a Beat 3 bullet at this stage and cannot be demonstrated until Chapter 5. Mark it "(we return to this in Chapter 5)."
8. **Exponential model used before it is defined** (L501): see the forward-reference list below.

### §1.7 Analytical Exercises with Complete Solutions

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **6 / 10**
- Beat 1 bridges are excellent where present (1.1.3, 1.5). §1.4 has none.
- Beat 2: the chapter has one core algorithm (Algorithm 1.1) and it needs a fuller explanation (§1.3 issue 3). Other sections do not need an algorithm; optional snippets are suggested in §1.1 and §1.5.
- Beat 3 is present in most sections and is the strongest beat, but absent from §1.2.
- **The scaffolding is visible.** Headings such as "Beat 1: The Mathematical Anchor (The 'Why')" are project jargon. A student does not know what a "beat" is. Use student-facing headings: **"Why we need this," "Computing it," "What can go wrong."**
- **Pace:** §1.1 is rich and satisfying but front-loads the chapter's main surprise (infill) before the reader has the vocabulary. Then §1.6 repeats it. §1.4 is dry (definition → theorem → proof → one example). §1.6 ends with microergodicity, a research-level idea.
- **Missing signposts:** the chapter has no "what you will learn" box at the top and no summary at the bottom. A short "Chapter at a glance" (5 bullets) and a "Key ideas" recap are both inexpensive and very reader-friendly.
- **Opening paragraphs (L5):** "Following the pedagogical philosophy of Maria L. Rizzo and the semantic discipline of..." is a dedication for the author's own spec, not for the student. Replace with a sentence on what the student will be able to *do* after the chapter. It also conflicts with the mission's stated references (Faraway, Dobson & Barnett, Schabenberger & Gotway).

### Dimension 2 — Algebraic transparency and scaffolding: **6.5 / 10**
- Strengths: trace/quadratic-form algebra in Exercise 1.1 and the variance expansion in 1.1.3 are written out step by step.
- Magic steps to fill in:
  - $\operatorname{Var}(\mathbf{L}\mathbf{w}) = \mathbf{V}$ (Algorithm 1.1).
  - Reverse triangle inequality for standard deviations (Theorem 1.4.1).
  - $\operatorname{E}[(Z(\mathbf{s}+\mathbf{h})-Z(\mathbf{s}))^2] = 2(C(\mathbf{0}) - C(\mathbf{h}))$ under stationarity.
  - Substitution $\mathbf{h} = \mathbf{s}_1 - \mathbf{s}_2$ in Theorem 1.6.1.
  - Idempotence $(\mathbf{I} - \tfrac1n\mathbf{J})^2 = \mathbf{I} - \tfrac1n\mathbf{J}$ (Exercise 1.1 (a)), and the trace identity $\operatorname{E}[\mathbf{Z}'\mathbf{A}\mathbf{Z}] = \operatorname{Tr}(\mathbf{A}\mathbf{V}) + \boldsymbol{\mu}'\mathbf{A}\boldsymbol{\mu}$, which is used and not derived.
- **No "From the Lecture Hall: Bridging the Prerequisite Gap" callouts appear in the chapter at all.** The chapter has the FAQ-style "Wait—Why Did We Do That?" boxes (L78, L505), which are good but do a different job. Recommended Lecture Hall boxes:
  1. §1.2: "$L^2$ as a spreadsheet" (columns are random variables, covariance is the dot product). Move the L215 footnote here.
  2. §1.3: "What Kolmogorov's theorem is really saying" (marginals of a MVN are MVN).
  3. §1.5: "Why a single map can stand in for many experiments."
  4. §1.6: "Reading a semivariogram near zero" (the Taylor expansion above).

### Dimension 3 — Structural consistency and forward referencing: **6 / 10**
**Duplication (the biggest flow problem).** The "variance of the sample mean does not vanish under infill" result, with the same double-integral formula, appears **four times**: L145–156 (§1.1.3), L478–482 (Theorem 1.6.1), L500 (§1.6 Beat 3), and L507–516 (FAQ 2). Recommendation:
- Keep §1.1.3 to what is pure algebra: variance inflation, $\bar\rho$, $n_{\text{eff}}$ (and the new code), then say "we return to what happens when $n \to \infty$ in §1.6."
- Keep the full infill discussion in §1.6 only, using FAQ 2 (rewritten) as the explanation.

**Forward references to repair**
- Stationarity notation $C(\mathbf{h})$ used from §1.1 onward, defined in §1.5.
- "Positive definite" (L73, L274, L292) defined nowhere in this chapter.
- "Kriging" and "BLUP" (L88, L213, L503) not defined.
- Exponential covariance (L501) used before Chapter 2 defines it.
- "Semivariogram" first defined at L413 but stated in text earlier only by implication.

**Notation drift:** `C(0)` and `σ²` (see §1.1 issue 2); the "ε(s)" nugget is called "microscale white noise" in one place and "measurement error and micro-heterogeneity" in another (fine, but say once that both are lumped into $\tau^2$). Also the three-layer decomposition is attributed to Schabenberger & Gotway (L28); their decomposition (as I recall it) separates a smooth small-scale term from micro-scale variation and measurement error, i.e., four components. **Worth a check against the book**, since if the chapter collapses their four into three, the attribution should say "adapted from."

### Dimension 4 — Empirical and computational utility: **5 / 10**
- **No data, no running example.** The chapter never shows a number computed from a realistic spatial layout. A small running example (a 1-D transect of 10 sites with an exponential covariance, reused in later chapters) would let the reader *see* variance inflation, the Cholesky step, and the stationary-vs-Brownian contrast with the same object. This can be a worked table of numbers; code is not needed.
- **Exercise 1.1** is the right kind of exercise (matrix algebra and a trace identity, verifiable on paper). The solution is correct: I checked $\mathbf{A} = \frac{1}{n-1}(\mathbf{I}-\tfrac1n\mathbf{J})$ and the final $\operatorname{E}[S^2] = \sigma^2 - \frac{1}{n(n-1)}\sum_{i\neq j}C(\mathbf{s}_i-\mathbf{s}_j)$ line by line. Small fixes: L541 has a garbled "$\mathbf{1}(n)$" and should read $\tfrac1n\mathbf{1}(\mathbf{1}'\mathbf{1}) = \mathbf{1}$; the idempotence step is skipped; add the $\sigma^2(1-\bar\rho)$ simplification. It also uses $\sigma^2 = C(\mathbf{0})$ (total variance), which conflicts with §1.1's meaning.
- **Exercise 1.2** (fractional Brownian motion) checks out: the increment variance $h^{2H}$ and $\gamma(h) = \tfrac12 h^{2H}$ are correct, and $H = \tfrac12$ recovers the Brownian example. It is a clean generalization but quite abstract for a chapter opener; consider moving it to the workbook and putting a *numerical* exercise in its place (e.g., "simulate 200 realizations; estimate $\operatorname{Var}(\bar Z)$; compare with the formula").
- Only two exercises, both purely analytical. One numerical exercise (compute $n_{\text{eff}}$ for a small layout by hand, or via the optional snippet) would balance them.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Resolve the `C(0)` / `σ²` / `V` inconsistency** (L74–75, L106, L404, L449, L545). Adopt $\sigma_Z^2 = \sigma^2 + \tau^2$ for total variance, or set $\tau^2 = 0$ explicitly where the formulas need it.
2. **Fix Principle 1.3.1 (L274):** mean and covariance specify a *Gaussian* field only.
3. **Fix the BLUP-as-projection claim (L213):** orthogonality of the error holds for simple kriging; ordinary kriging adds a constraint.
4. **Fix the stationarity hierarchy diagram (L431–447):** strict stationarity implies second-order stationarity only with finite variance; for Gaussian fields they coincide.
5. **State how the trend model relates to stationarity** (add to §1.5, referencing L402 and §1.1.2).
6. **Correct FAQ 2 (L512):** remove the $\omega_i$ story and use the "same hill, many sensors" explanation.
7. **Replace Definition 1.6.2** with an informal definition, add the Taylor expansion, add the $d \le 3$ caveat and full citations (Ying 1991; Zhang 2004; Anderes 2010 — verify).

### Should fix (pace, flow, voice)
8. **De-duplicate the infill result:** keep algebra in 1.1.3, full discussion in §1.6.
9. **Explain Algorithm 1.1 fully:** why $\mathbf{L}\mathbf{w}$ has covariance $\mathbf{V}$, the nugget convention in step 2, and what one output means. *(Optional: Python snippets for $n_{\text{eff}}$ in §1.1 and a stationary-vs-Brownian demo in §1.5.)*
10. **Add Beat 1 to §1.4** (move the nugget example up as motivation) and a Beat 3 to §1.2.
11. **Replace "Beat 1/2/3" headings** with student-facing headings.
12. **Add four "From the Lecture Hall" callouts** (list in Dimension 2).
13. **Replace the L5 dedication** with a "what you will be able to do" paragraph; add "Chapter at a glance" at the top and "Key ideas" at the bottom.
14. **Fill the magic steps** listed in Dimension 2.
15. **Add one-line definitions on first use:** kriging, positive (semi)definite, notation $\mathbf{Y} \sim (\boldsymbol{\mu}, \boldsymbol{\Sigma})$. Add the one-line reason that $\mathbf{V}$ must be positive semidefinite: $\operatorname{Var}(\mathbf{a}'\mathbf{Z}) = \mathbf{a}'\mathbf{V}\mathbf{a} \ge 0$.

### Nice to have
16. Use a realistic $\bar\rho$ (e.g., 0.1) in the $n_{\text{eff}}$ example; add $n_{\text{eff}} \to 1/\bar\rho$.
17. Show $\operatorname{E}[S^2] = \sigma^2(1-\bar\rho)$ next to $\operatorname{Var}(\bar Z)$.
18. Soften "severely" (L21), "guarantees" (L317), and "$\operatorname{Cov}\neq 0$" (L18).
19. Clarify "range" vs. e-folding scale $\phi$; mention isotropy.
20. Fix small items: "topological" → "topographic" (L450), the $\mathbf{1}(n)$ typo (L541), Brownian motion domain (L428), $\mathbf{h}$ definition in Definition 1.5.2.
21. Consider replacing the second §1.7 exercise with a numerical one; move fractional Brownian motion to the workbook.
22. Check the S&G decomposition attribution (three vs. four terms).

---

## 6. What to Keep Exactly As It Is

Preserve these; they are the book's voice at its best:
- Opening paragraph (L3): "How much is where?"
- The term-by-term anatomy blocks (L34–38, L72–76).
- FAQ 1 and its "practical catch" (L78–91).
- The variance-inflation derivation and $n_{\text{eff}}$ (L104–143).
- The rows-vs-columns table (L171–182) and the geometry list (L188–213), with the L213 fix.
- The nugget-discontinuity example (L364–376).
- "We only ever get one realization of the Earth" and "spatial pseudo-replicates" (L387–391).
- The Brownian-motion counterexample (L428).
