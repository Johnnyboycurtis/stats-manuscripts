# Review: Chapter 2 — Spatial Covariance and Variogram Theory

**File reviewed:** `ch02_covariance_variogram.md` (1,173 lines)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch02_exercises.md` was not reviewed. The three exercises inside §2.6 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy, sympy). Where I cite the literature from memory, I say so.

---

## 1. Executive Summary

**Grade: 6 / 10**

Chapter 2 has the best teaching ideas in the manuscript so far: the "variance of a linear combination cannot be negative" bridge that motivates positive definiteness, the ANOVA-contrast bridge into the variogram, the "closed forms first, general Matérn second" ordering, and a metric-tensor proof that a student can follow with pencil and paper. It follows the Three-Beat rhythm in every numbered section and stays almost entirely within the linear-model ceiling. The weaknesses are correctness, not tone: three headline results are stated or "proved" incorrectly (Bochner's theorem as an "if and only if" with a density, the sub-quadratic growth theorem, and the spherical-model dimension barrier), several formulas are slightly off, the boxed theorems will not render, and the chapter is missing the one picture every student needs first (a labeled nugget/sill/range sketch).

**Highest-value changes**
1. Repair the spherical-model dimension barrier (§2.3). The "finite matrix counterexample" does not demonstrate what it claims. I checked it: for the star configuration the covariance matrix is perfectly valid at $d = 4$ (smallest eigenvalue $+0.67$) and first fails at $d = 38$. The claim ($d \ge 4$ is invalid) is true, but the argument given does not prove it.
2. Fix Theorem 2.3 (sub-quadratic growth). As stated it is false; $\gamma(h) = c\,h^2$ is a valid semivariogram. The clean half of the proof, $\gamma(m\mathbf{h}) \le m^2\gamma(\mathbf{h})$, is worth keeping and is the real theorem.
3. Put the definitions and theorems back into the page. All of them sit inside code fences (L61–73, L79–90, L135–145, L336–348, L353–361, L396–406, L518–531, L746–755, L772–786, L891–899, L904–915), so the LaTeX inside them will display as raw `$...$` text.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes, and notably better than the README advertised.** There is no measure theory and no Hankel/Bessel integral calculus. The chapter openly chooses "finite matrix eigenvalue counterexample" and Taylor expansions where it can. The drift points are small:

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L59 | "violates Kolmogorov's second axiom of probability" | Low (but wrong) | Variance being non-negative is not one of Kolmogorov's axioms; it follows from the definition. Say "a variance cannot be negative because it is an average of squares." Dropping the name also removes unneeded gatekeeping. |
| L77–90 | Bochner's theorem as an "if and only if" with a *density* $f(\boldsymbol\omega)$ | Medium | The full theorem needs a *spectral measure*; densities cover every model in this book but not all valid covariances (e.g., $C(h) = \cos(\omega_0 h)$, a random sinusoid, is a valid covariance with no density). Label the theorem as the *sufficiency* direction you actually prove, and say in one sentence that the full theorem allows point masses. No measure language needed. |
| L238 | "By the Rayleigh quotient and Weyl's eigenvalue perturbation theorem" | Low | Overkill for a one-line fact ($\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$ implies $(\mathbf{C} + \tau^2\mathbf{I})\mathbf{v} = (\lambda + \tau^2)\mathbf{v}$). Name-dropping makes an easy fact sound hard. |
| L397–407 | Schoenberg's theorem stated without proof | Low–Medium | Acceptable as a cited theorem, but one direction has a three-line Taylor-series proof at exactly the right level (see §3, §2.2). |
| L575–579 | Closed-form Matérn spectral density | Low | Accept as a cited formula (I checked it against the standard form in angular-frequency convention; it is right). Point to Appendix B; do not derive it. |
| L77 | "characteristic functions of non-negative spectral densities" | Low | Same density-vs-measure caveat as above. |
| L503–508, L563–573 | $K_\nu$ named, not integrated | None | Fine as a definition. See the §2.3 note on computing it. |

No hypergeometric functions or Lebesgue–Stieltjes integrals appear. Where the chapter needs a result that normally uses Bessel integrals (the spherical-model barrier), it chooses a finite-matrix argument, which is exactly the right instinct. It just needs to work (see §2.3).

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 2.1 | Yes (excellent bridge) | Yes (matrix assembly explained well) | Yes (conditioning, nugget as ridge) |
| 2.2 | Yes (ANOVA contrast bridge) | Yes (projection idea excellent; code diverges from text) | Thin and partly mismatched |
| 2.3 | Yes (table, then Matérn from simple to general) | Partly (closed forms good; general branch wrong) | Gaussian hazard good; no model-choice guidance |
| 2.4 | Yes | Yes (diagnostic; explanation incomplete) | Nugget discussion good; nothing on misreading curvature |
| 2.5 | Yes (strong proof) | Yes (correct; explanation incomplete) | Good warning; diagram confusing |
| 2.6 | n/a | n/a | n/a |

The structure is the strongest in the manuscript so far. The recurring issue is the visible "Beat 1/2/3" headings, discussed under Dimension 1.

### §2.1 Positive Definite Functions and Bochner's Theorem

**What works**
- The two-variable warm-up $\operatorname{Var}(aX + bY) \ge 0$ (L26–38), generalized to $\mathbf{c}'\mathbf{V}\mathbf{c} \ge 0$ (L41–49), is exactly the classical-to-spatial bridge the mission describes. It also pays off the missing "why PSD" from Chapter 1.
- The Bochner "$\Leftarrow$" proof (L102–127) is a model of how to prove something without measure theory: write the quadratic form, move the sum inside, and observe a squared modulus.
- The closing nugget-as-ridge idea (L237–246) is a good link to what students already know about regularization.

**Issues**
1. **Boxed definitions and theorems do not render** (see Executive Summary). Use a callout style (e.g., a blockquote with a bold label) that allows math.
2. **L30 and L59.** Variance non-negativity is "an absolute mathematical axiom" and a "violation of Kolmogorov's second axiom." Both are inaccurate. It follows from variance being the average of a squared quantity. (Also drops a measure-flavored name.)
3. **Definition 2.1 wording (L65).** "positive definite (strictly, positive semi-definite)" is confusing: "strictly" and "semi-definite" pull in opposite directions. Say: "positive semi-definite (often just called 'positive definite' in geostatistics); *strictly* positive definite if the quadratic form is $0$ only when $\mathbf{a} = \mathbf{0}$."
4. **Theorem 2.1 is not an "if and only if" with a density** (see calibration table). Also "bounded" (L87) is not required; "integrable" is the condition.
5. **L98 cause and effect.** The sine term integrates to zero because $f$ is symmetric (equivalently, because $C(-\mathbf{h}) = C(\mathbf{h})$), not because "spatial covariance is symmetric" as a separate fact. Say it once, plainly.
6. **Beat 2 uses models before they are introduced.** The assembly function (L158–217) takes `family="exponential" | "gaussian" | "spherical"`, and Beat 3 uses the exponential model ($C(\Delta) \approx \sigma^2(1 - \Delta/\phi)$, L229). None of these is defined until §2.3. One sentence of forward-pointing ("we use the exponential model $\sigma^2 e^{-h/\phi}$ for now; §2.3 introduces more") fixes it.
7. **The assembly algorithm is explained well** (three numbered steps, L153–156). Small improvements: (i) the docstring says the matrix is "symmetric positive definite," but nothing guarantees that (e.g., the spherical model in dimension $\ge 4$, or very close points); (ii) `2.718281828459045 ** x` is an odd way to write $e^x$ (`math.exp` is in the standard library); this is style only.
8. **Beat 3, conditioning bound (L238–252).**
   - The bound $\kappa(\mathbf{V}) \le (n\sigma^2 + \tau^2)/\tau^2$ is *correct*, but the step that gives $\lambda_{\max}(\mathbf{C}) \le n\sigma^2$ is never shown. It is one line: eigenvalues of a PSD matrix are non-negative, so $\lambda_{\max} \le \sum \lambda_i = \operatorname{tr}(\mathbf{C}) = n\sigma^2$.
   - "Guaranteeing numerical invertibility" (L246) overstates. The bound can be huge: with $n = 1000$, $\sigma^2 = 1$, $\tau^2 = 10^{-6}$ it is about $10^9$. Say "improves conditioning in proportion to $\tau^2/(n\sigma^2)$."
   - The diagram (L248–252) claims "$\kappa \le 10^3$ [STABLE]" for any nugget. That number only holds for one choice of $n\sigma^2/\tau^2$. Replace with a small worked example.
9. **Duplication with Chapter 1.** The "nugget stabilizes $\mathbf{V}$" idea appears in Ch 1 (§1.3 Beat 3) and again here. Fine to revisit, but cross-reference instead of re-teaching, and make this version the quantitative one (the bound above).
10. **Section header mismatch.** Beat 3 opens "Why does positive definiteness fail in practice?" but then discusses *numerical* singularity from clustered points, not invalid functions. Separate the two causes in one sentence each.

### §2.2 Semivariogram Theory and Conditionally Negative Definite Functions

**What works**
- The ANOVA contrast bridge (L262–274) is superb and exactly what the mission asks for.
- The CND derivation (L312–334) is a model of explicit algebra: the zero-sum weights visibly kill the $Z_i^2$ terms. Keep it as written.
- The centering-matrix idea in Beat 2 (L415–424), "project onto the zero-sum subspace and check the result is positive semidefinite," is the right algorithm to teach, and it connects to the projection matrix students know from regression.

**Issues**
1. **Theorem 2.3 is false as stated, and the proof's last step does not follow** (L350–393).
   - $\gamma(\mathbf{h}) = c\|\mathbf{h}\|^2$ is a valid semivariogram: it belongs to the field $Z(\mathbf{s}) = \mathbf{A}'\mathbf{s} + B$ with a zero-mean random slope $\mathbf{A}$ (the text itself says this at L393). Here $\gamma/\|\mathbf{h}\|^2 \to c$, not $0$.
   - Steps 1–6 prove the correct and useful fact $\gamma(m\mathbf{h}) \le m^2\gamma(\mathbf{h})$ cleanly, using only Cauchy–Schwarz. Keep this.
   - Step 7 assumes "if the limit is $c > 0$ then $\gamma(h) = ch^2$." That does not follow (the limit only says it behaves like $ch^2$ for large $h$). Remove step 7. State the theorem as: *a semivariogram grows at most quadratically, and exactly quadratically only when the field has a random linear trend.* Fix the "entirely deterministic" language: a random slope is random, not deterministic.
2. **Power-law model (L508, Table 2.1, Exercise 2.3).** $\gamma = bh^\alpha$ is valid (CND) for $0 < \alpha \le 2$, not $0 < \alpha < 2$; $\alpha = 2$ is the degenerate random-slope boundary. Exercise 2.3 itself computes $Q(2) = 0$ and calls it valid, so the chapter currently contradicts itself. Say "strictly valid for $0 < \alpha < 2$; $\alpha = 2$ is a degenerate boundary; invalid for $\alpha > 2$."
3. **Schoenberg's theorem (L397–407)** says "strictly positive definite." The theorem gives positive (semi)definite; $\gamma \equiv 0$ gives the constant $1$, which is not strictly PD. **Opportunity:** the direction "$\exp(-t\gamma)$ PD for all $t$ implies $\gamma$ CND" has a short Taylor proof at the ceiling: for $\sum\lambda_i = 0$,
   $$\sum_{i,j}\lambda_i\lambda_j e^{-t\gamma_{ij}} = \underbrace{\Big(\sum\lambda_i\Big)^2}_{=0} - t\sum_{i,j}\lambda_i\lambda_j\gamma_{ij} + O(t^2) \ge 0 \quad\text{for all small } t>0,$$
   which forces $\sum\lambda_i\lambda_j\gamma_{ij} \le 0$. Include it and cite the converse.
4. **FAQ 1, point 2 (L310)** says kriging with $\gamma$ gives "identical numerical weights and Lagrange multipliers." The weights are identical. Whether the multipliers are identical depends on the sign convention: with the convention Chapter 5 uses (covariance system $\mathbf{V}\boldsymbol\lambda - \mu_{\text{OK}}\mathbf{1} = \mathbf{v}_0$, variogram system $\boldsymbol\Gamma\boldsymbol\lambda + \mu^{(\gamma)}\mathbf{1} = \boldsymbol\gamma_0$) they are the same number (I checked this numerically while reviewing Chapter 5; my first draft of this note said "opposite sign," which holds only under the other common convention). So the claim is fine if the convention is stated. Either say "(with the sign convention of Chapter 5)" or drop the Lagrange-multiplier part. Also this is a forward reference to ordinary kriging (Chapter 5), so mark it "(Chapter 5)."
5. **Beat 2 code vs. explanation mismatch (L426–496).** The text says "check all eigenvalues of $\mathbf{Q}$ are non-negative." The code instead runs an LDL$'$ pivot test, and:
   - $\mathbf{Q} = -\mathbf{P}\boldsymbol\Gamma\mathbf{P}$ always has a zero eigenvalue (since $\mathbf{P}\mathbf{1} = \mathbf{0}$), which is exactly the case where an unpivoted LDL$'$ test is fragile; the code skips zero pivots instead of checking consistency.
   - The docstring says it returns eigenvalues but it returns `Q`.
   - I ran it on six cases (valid and invalid) and it gave the right answer each time, so this is a robustness concern, not a demonstrated bug. A small eigenvalue routine, or a Cholesky on $\mathbf{Q} + \varepsilon\mathbf{I}$, would match the text.
   - **A worked example would make this algorithm land.** On the three points $\{0, 1, 2\}$ with $\gamma(h) = h^3$, the smallest eigenvalue of $\mathbf{Q}$ is about $-1.33$ (invalid); with $\gamma(h) = h$ it is $0$ (valid, to machine precision); with $\gamma(h) = h^{2.5}$ it is about $-0.55$. (All computed.) Three numbers tell the whole story.
6. **Beat 3 is mismatched (L502–508).** It asks "why can we not fit $\gamma(h) = a + bh^2 + ch^4$" but then discusses $bh^{2.5}$; the "Parabolic Trap" bullet names a different problem from its heading. "Will inevitably possess large negative eigenvalues" overstates (negative eigenvalues appear for *some* configurations). The kriging-variance sentence also uses kriging before it is introduced. **Exercise 2.3's finite counterexample (second-difference weights $(1, -2, 1)$ on points $0, 1, 2$) is the cleanest in the chapter and belongs here.**
7. **Definition 2.2 (L339)** requires $\gamma$ continuous with $\gamma(\mathbf{0}) = 0$, which excludes nugget models (the jump at $0^+$). Say "with $\gamma(\mathbf{0}) = 0$, continuous away from the origin."
8. **Vocabulary picture missing.** *Nugget, sill, and range* appear in separate places (sill at L292, nugget at L872, range at the spherical model). A single labeled sketch (an ASCII variogram rising from the nugget to the sill at the range) belongs right after L292. It is the first picture every spatial-statistics student expects.

### §2.3 Admissible Parametric Families

**What works.** Introducing Matérn as "accessible polynomials first, unified family second" (L533–560) is excellent pedagogy. The three closed forms are correct (I checked all three, including the $\sqrt{3}$ and $\sqrt{5}$ scalings). The smoothness-spectrum diagram (L724–730) is a good one-glance summary.

**Issues**
1. **Table 2.1 does not render** (code fence). It is also the right place to list each model's *practical range* (the distance at which correlation falls to about 5%): about $3\phi$ for the exponential, $\sqrt{3}\phi$ for the Gaussian, $\phi$ for the spherical. Without that, a student comparing "$\phi = 10$" across models will mis-compare them.
2. **The Matérn $\nu \to \infty$ limit is not the Gaussian model in the table (L555–559).** With the $\sqrt{2\nu}$ scaling, the limit is $\sigma^2\exp(-h^2/(2\phi^2))$, not $\sigma^2\exp(-h^2/\phi^2)$. I checked numerically: at $h = \phi$, $\nu = 100$ gives $0.604$, matching $e^{-1/2} = 0.607$ and not $e^{-1} = 0.368$. Either define the Gaussian model as $\exp(-h^2/(2\phi^2))$ or say "the Gaussian model with $\phi$ rescaled."
3. **Spherical-model dimension barrier (L581–653, plus Exercise 2.1).** This is the most important correctness issue in the chapter.
   - **What is true.** The spherical covariance is valid in $\mathbb{R}^1, \mathbb{R}^2, \mathbb{R}^3$ and invalid for $d \ge 4$. I confirmed the second half independently: computing the spherical model's $d$-dimensional spectral density numerically, it stays non-negative for $d = 1, 2, 3$ and goes negative for $d = 4$ and $d = 5$.
   - **What the chapter's argument shows.** The star configuration (a center and $2d$ axis points at distance $\phi/\sqrt{2}$) requires $2d > 1/\rho^2$ with $\rho = 0.11612$, i.e., $d \ge 38$ (the exercise states the 74.2 threshold; the main text hides it). I checked: the smallest eigenvalue at $d = 4$ is $+0.67$ (valid), at $d = 40$ it is $-0.039$. So the argument proves "compactly supported models fail in high dimension," a valid but different claim, and the sentence at L640 ("In $\mathbb{R}^4$ ... confirms ... indefinite. Therefore invalid for $d \ge 4$. $\blacksquare$") has no support. Exercise 2.1 (L1062) repeats it.
   - **Notation slip.** In the variance computation (L630–633), $C(r)$ stands for the *correlation* $\rho$ in some places and the *covariance* $0.116\sigma^2$ in others.
   - **I could not find a small counterexample.** A quick numerical search over 8–12 points in $\mathbb{R}^4$ turned up no negative eigenvalue (not exhaustive). A short finite counterexample for $d = 4$ may not exist, or may be hard to find. Do not promise one.
   - **Suggested fix.** (a) State the result and attribute it (the spherical model's $d$-dimensional spectral density becomes negative for $d \ge 4$; cite Gneiting 1999 and Wendland — from memory, please verify). (b) Replace the "packing" explanation (L650) with the nesting principle: *a function valid in $\mathbb{R}^d$ is automatically valid in $\mathbb{R}^k$ for $k < d$ (just use fewer coordinates), but not conversely.* The cubic is the overlap volume of two 3-D balls; it is not the overlap volume in 4-D, so the "it's a volume, so it's PSD" argument does not extend. (c) Keep the star argument, honestly labeled: "compact support cannot work in *every* dimension." (d) Fix "spheres ... in $\mathbb{R}^d$" (L582) to "in $\mathbb{R}^3$."
4. **General-$\nu$ branch of the Matérn code is wrong (L702–717).** I compared it with the true Matérn (using scipy's $K_\nu$): at $\nu = 4$, $h = \phi$ the code returns $0.059$ vs. the true $0.552$; at $\nu = 1$, $h = \phi$: $0.362$ vs. $0.444$; at $\nu = 3.5$, $h = 2\phi$: $0.050$ vs. $0.138$. It uses a first-order large-argument approximation that is badly off at moderate arguments and silently returns it. **Suggested fix (stays dependency-free and at the ceiling):** compute $K_\nu$ from its integral representation
   $$K_\nu(x) = \int_0^\infty e^{-x\cosh t}\cosh(\nu t)\,dt$$
   with the trapezoid rule on $[0, 12]$. I checked it against scipy at $(\nu, x) = (0.75, 0.5), (2, 2), (4, 2)$ and it agrees to about $10^{-15}$ with 4,000 points. It is an ordinary Riemann integral. Alternatively, restrict the function to $\nu \in \{1/2, 3/2, 5/2\}$ and say so; the docstring ("arbitrary distances") should match whichever you choose.
5. **Gaussian hazard (L732–736).**
   - $f(\rho) \propto e^{-\pi^2\phi^2\rho^2}$ is the *cycles-per-unit* convention; this chapter uses angular frequency (L131), where the Gaussian model gives $f \propto e^{-\phi^2\rho^2/4}$. Use one convention.
   - "Routinely hits $\kappa \sim 10^{19}$" depends on $n$, spacing, and $\phi$. Say "can exceed $10^{16}$ for modest $n$ and spacing" and show one example.
   - "Brownian-like topography" for $\nu = 1/2$ (L542): the exponential model corresponds to the Ornstein–Uhlenbeck process, not Brownian motion, which is not stationary. Say "rough."
6. **Beat 3 lacks decision guidance.** The student's real question is "which model do I use?" Four bullets would do: start with Matérn $3/2$ or $5/2$; use exponential for rough phenomena; avoid Gaussian unless smooth physics justifies it; use spherical only in $d \le 3$.
7. **Notation.** $\rho$ means radial frequency here (L133) and correlation in Chapter 1. Pick one.

### §2.4 Regularity at the Origin and Mean-Square Differentiability

**What works.** The curvature-at-the-origin story (linear, parabolic, discontinuous) is the right conceptual summary of the chapter, and Exercise 2.2's Taylor expansion approach is exactly at the ceiling.

**Issues**
1. **Theorem 2.5 duplicates Theorem 1.4.1 (Chapter 1).** Restate and refer back; do not re-prove.
2. **Theorem 2.6's proof has an invalid step (L801).** "A sequence of random variables converges in mean square if and only if the variance of its differences converges to a finite limit" is not correct. The criterion is a *Cauchy* condition: $\operatorname{E}[(D_h - D_{h'})^2] \to 0$ as $h, h' \to 0$. **A three-line replacement at the ceiling:** write $D_h = (Z(\mathbf{s} + h\mathbf{e}) - Z(\mathbf{s}))/h$. Then
   $$\operatorname{Cov}(D_h, D_{h'}) = \frac{C(h - h') - C(h) - C(h') + C(0)}{hh'} \to -C''(0),$$
   using $C(u) = C(0) + \tfrac12 C''(0)u^2 + o(u^2)$, and $\operatorname{Var}(D_h) \to -C''(0)$, so $\operatorname{E}[(D_h - D_{h'})^2] \to -C'' - C'' + 2C'' = 0$. (I verified the algebra.) Also the equivalent condition "$\lim \gamma(h)/h^2 < \infty$" should say the limit *exists*.
3. **Classification is incomplete (L803–813).** Only $\alpha = 1$ and $\alpha = 2$ are named. Between them ($1 < \alpha < 2$, e.g., the powered-exponential family) the field is mean-square continuous but not differentiable ($\gamma/h^2 \to \infty$). The expansion is also called a "power series," but $h^\alpha$ is generally not one.
4. **The diagnostic algorithm (L821–865) is under-explained and its classification can mislead.**
   - It fits a pure power law $\gamma = ch^\alpha$, but the model above has an added $\tau^2$; with a nugget the log-log slope is near zero. The text does not say how to handle that.
   - The threshold `alpha < 1.3` calls anything above it "Parabolic (differentiable)," which misclassifies $1.3 \le \alpha < 2$ as differentiable.
   - `semivariances[0] > 0.1 * semivariances[-1]` compares with the last lag of the whole list, not the five used for the fit, and the 10% cutoff is arbitrary.
   - Beat 3 never warns that empirical semivariances at the shortest lags are noisy and few; origin behavior is hard to read from data.
   - This is not a core algorithm. Either drop it, or rewrite it as "how to look at an empirical variogram near zero" with those caveats.
5. **Duplication.** The nugget discussion in Beat 3 (L872–881) repeats Chapter 1 (FAQ 1 and §1.4 Beat 3). The new and valuable content here is the micro-scale vs. measurement-error split and the duplicate-assay design note (L881); keep that and cut the rest.
6. **Detail.** "Parabolic origin: Matérn with $\nu \ge 1.5$" (L813): the threshold is $\nu > 1$ ($\nu = 1$ gives $h^2\log h$, borderline).

### §2.5 Spatial Anisotropy

**What works.** Theorem 2.7 (L906–935) is the best proof in the chapter for this audience: spectral theorem, $\mathbf{M} = \boldsymbol\Lambda^{1/2}\mathbf{R}'$, then "it's the same quadratic form at new coordinates." Everything is pencil-checkable. The zonal-anisotropy section correctly explains why stretching cannot change a sill.

**Issues**
1. **Zonal model: one missing line (L940–947).** Say why a sum of covariance functions is a covariance function: $\sum a_ia_j(C_1 + C_2)(\mathbf{s}_i - \mathbf{s}_j) = Q_1 + Q_2 \ge 0$ (or independence of the two fields). One line, no theory.
2. **The coordinate-transform algorithm is correct but incompletely explained (L955–993).** I checked: the rotation by $90° -$ azimuth and the minor-axis scaling are right. Missing: (i) the assumed axes ($x$ = East, $y$ = North); (ii) *which range to use afterwards*: since the minor axis is stretched by $a_{\max}/a_{\min}$, the isotropic kernel's range is `range_major`; (iii) the link to the theorem's $\mathbf{M}$ (here $\mathbf{M}$ is scaled by $a_{\max}$).
3. **Diagram (L1000–1010) is confusing.** The "Major Axis" and "Minor Axis" labels sit on a single line. The two axes should be perpendicular.
4. **L1013–1014** claims the ad-hoc directional formula "will have negative eigenvalues for diagonal point configurations." Say "can." A concrete counterexample would be better than the assertion.

### §2.6 Analytical Exercises

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **7 / 10**
- This chapter follows the three beats in every numbered section, which is better than Chapter 1.
- Pacing is good through §2.2: each new idea gets a bridge before it gets a theorem. It slows in §2.3 (a long, flawed spherical-model argument sits between the Matérn story and the Gaussian hazard) and thins out in §2.4–2.5 Beat 3.
- **Visible scaffolding:** "Beat 1: The Mathematical Anchor (The 'Why')" is project vocabulary. Students will not know what a "beat" is. Use "Why we need this," "Computing it," "What can go wrong." The same applies to labels like "CRITICAL THEOREM 2.2" (L137): let the content carry the emphasis.
- **The L15 paragraph** names Rizzo, S&G and "the coordinated Three-Beat framework." Replace with what the student will be able to do after this chapter: check whether a candidate covariance is valid, choose a model, and read a variogram near the origin.
- **Missing signposts:** a "Chapter at a glance" at the top, a labeled nugget/sill/range picture, a "which model do I use?" box, and a "Key ideas" recap.

### Dimension 2 — Algebraic transparency and scaffolding: **6.5 / 10**
- Strengths: the CND derivation (L312–334), Bochner $\Leftarrow$ (L102–127), Theorem 2.7 (L917–935), Exercises 2.2–2.3 expansions are written step by step.
- Magic or incorrect steps to fill: $\lambda_{\max} \le \operatorname{tr}(\mathbf{C})$ (L244); Theorem 2.3 step 7 (remove); Theorem 2.6's convergence criterion (replace with the Taylor/Cauchy computation); Schoenberg (add the Taylor proof of one direction); "the sum of covariance functions is a covariance function" (§2.5).
- **"From the Lecture Hall: Bridging the Prerequisite Gap" callouts: none are present.** The FAQ boxes are good but do a different job. Recommended Lecture Hall boxes:
  1. Bochner: "a covariance is a mixture of cosines" (characteristic functions, as the chapter already hints at L77).
  2. CND: "why variograms need a zero-sum constraint" (reuse the ANOVA contrast idea, boxed).
  3. Matérn: "what $K_\nu$ is and why you rarely need it."
  4. Mean-square differentiability: "a derivative is a limit of differences, so smoothness lives in the second difference of $C$."

### Dimension 3 — Structural consistency and forward referencing: **6 / 10**
- **Forward references:** the exponential/Gaussian/spherical families used in §2.1 Beat 2–3 before §2.3 defines them; kriging variance used at L505 and L310 before Chapter 5; "nugget," "sill," and "range" used before they are all defined.
- **Notation drift:** $C(\mathbf{0}) = \sigma^2$ (L100) vs. $C(\mathbf{0}) = \sigma^2 + \tau^2$ (L156, L309); $\rho$ as radial frequency (L133) vs. correlation (Ch 1); Gaussian model $e^{-h^2/\phi^2}$ vs. the Matérn limit $e^{-h^2/(2\phi^2)}$; $\phi$ means different things across models (hence the practical-range table).
- **Duplication:** Theorem 2.5 repeats Theorem 1.4.1; nugget-as-regularizer and nugget-as-discontinuity repeat Chapter 1.
- **Rendering:** all boxed definitions, theorems, and Table 2.1 (see Executive Summary).

### Dimension 4 — Empirical and computational utility: **5.5 / 10**
- **Exercise 2.1** (spherical barrier) inherits the problem in §2.3 and asserts $d \ge 4$ without support. Retitle it "compact support fails in high dimension," keep the correct $d \ge 38$ computation, and note that the exact $d = 4$ threshold is a research-level fact.
- **Exercise 2.2** (Matérn Taylor expansions) is a good exercise and fully within the ceiling, but contains an error: for $\nu = 5/2$ the fourth-order coefficient is $w^4/24$, not $w^4/72$ (sympy gives $1 - w^2/6 + w^4/24 - w^5/45 + \dots$). The corrected form is $C(h) = \sigma^2[1 - \tfrac{5h^2}{6\phi^2} + \tfrac{25h^4}{24\phi^4} + O(h^5)]$. The conclusion (the first odd power is $h^5$, so $m = 2$) is unchanged. Also step 1 cites "Theorem 2.6" for the $m$-th derivative, which that theorem does not state.
- **Exercise 2.3** (power variogram, $\alpha > 2$) is excellent: elementary, correct (I verified $Q(\alpha) = 2(2^\alpha - 4)$), and verifiable by hand. Move it into the main text (see §2.2) and fix the $\alpha = 2$ language.
- **No running data example.** A single small example reused across §2.1–§2.4 (say five points on a transect) would let students *see* a valid $\mathbf{V}$, an invalid $\boldsymbol\Gamma$, and the effect of the nugget on conditioning. It can be a table of numbers; code is not required.
- Only three exercises, all analytical, none using the algorithms taught in the chapter. One short "compute by hand" exercise (e.g., check CND for three points) would complement them.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Render the boxed definitions, theorems, and Table 2.1** (move out of code fences).
2. **Fix the spherical-model barrier** (§2.3, Exercise 2.1): state the true result with a citation, explain via nesting of dimensions, keep the star argument but label it "fails by $d \ge 38$," and fix the $C(r)$ vs. $\rho$ slip.
3. **Fix Theorem 2.3:** keep $\gamma(m\mathbf{h}) \le m^2\gamma(\mathbf{h})$, drop step 7, and state the correct conclusion (quadratic growth only with a random linear trend).
4. **Fix the power-law range:** valid for $0 < \alpha \le 2$ (degenerate at 2), in L508, Table 2.1, and Exercise 2.3.
5. **Fix Theorem 2.1's statement:** label as sufficiency with a density, note the spectral measure caveat in one sentence, replace "bounded" with "integrable."
6. **Fix the Matérn/Gaussian limit** ($e^{-h^2/(2\phi^2)}$) or redefine the Gaussian model.
7. **Fix the general-$\nu$ Matérn code** (use the $K_\nu$ integral, or restrict to half-integers and say so).
8. **Fix Exercise 2.2's $\nu = 5/2$ coefficient** ($w^4/24$).
9. **Replace the invalid step in Theorem 2.6** with the Taylor/Cauchy computation.
10. **Remove "Kolmogorov's second axiom" (L59) and "absolute axiom" (L30).**

### Should fix (pace, flow, voice)
11. Add a labeled **nugget/sill/range picture** after L292.
12. Add **practical ranges** to Table 2.1 and a **"which model?" box** in §2.3 Beat 3.
13. Add the **Schoenberg Taylor proof** (one direction) and the line "sum of covariances is a covariance."
14. **Move Exercise 2.3's $(1, -2, 1)$ counterexample into §2.2 Beat 3**; fix the header/bullet mismatch.
15. **Make the CND algorithm match its code:** eigenvalues or a ridge-Cholesky; fix the docstring; add the three worked numbers ($h^3$: $\approx -1.33$; $h$: $0$; $h^{2.5}$: $\approx -0.55$ on points $0, 1, 2$).
16. **Complete the origin-diagnostic explanation** (nugget, thresholds, $1 < \alpha < 2$) or drop it.
17. **Explain the anisotropy transform's conventions** (axes, which range to use, link to $\mathbf{M}$) and fix the diagram.
18. **Add four "From the Lecture Hall" callouts** (list in Dimension 2) and a "Chapter at a glance" / "Key ideas."
19. **Replace "Beat" headings** with student-facing ones; replace the L15 dedication.
20. **Define the models** (or point to §2.3) the first time they are used in §2.1.
21. **Cross-reference Chapter 1** instead of re-teaching Theorem 1.4.1 and the nugget explanation.
22. **Show $\lambda_{\max} \le \operatorname{tr}(\mathbf{C})$** and temper "guaranteeing invertibility" and the "$\le 10^3$" diagram.

### Nice to have
23. Use `math.exp` instead of `2.718... **`; fix the "returns eigenvalues" docstring.
24. Fix "Brownian-like topography" (Ornstein–Uhlenbeck / "rough").
25. Reconcile $C(\mathbf{0})$ and $\rho$ notation across chapters; one frequency convention.
26. Soften "routinely $10^{19}$" and "will have negative eigenvalues" claims.
27. Add one short numerical exercise using the chapter's algorithms.
28. Fix Definition 2.2's continuity wording and the Definition 2.1 "strictly, semi-definite" phrasing.

---

## 6. What to Keep Exactly As It Is

- The $\operatorname{Var}(aX + bY) \ge 0$ bridge into positive definiteness (L26–49).
- The Bochner $\Leftarrow$ proof (L102–127).
- The ANOVA-contrast bridge and the explicit CND derivation (L262–334).
- "Accessible polynomials first, unified Matérn second" (L533–564) and the smoothness-spectrum diagram (L724–730).
- The metric-tensor proof of Theorem 2.7 (L917–935).
- The duplicate-assay design note at the end of §2.4 (L881).
- Exercise 2.3's second-difference contrast counterexample (as a model of the whole book's "finite-matrix counterexample" style).
