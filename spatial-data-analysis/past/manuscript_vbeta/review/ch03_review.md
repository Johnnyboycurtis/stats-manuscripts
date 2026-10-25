# Review: Chapter 3 — Nonparametric Covariance and Semivariogram Estimation

**File reviewed:** `ch03_estimation.md` (1,492 lines)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch03_exercises.md` was not reviewed. The four exercises inside §3.7 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy; Monte Carlo with 200,000 simulated fields where noted). Literature attributions marked "from memory" are not verified.

---

## 1. Executive Summary

**Grade: 7 / 10**

This is the strongest chapter so far. Every numbered section follows the three-beat structure, the algebra is mostly written out with no hand-waving, the "variance theft" explanation of why the sample covariance sags is memorable, and the "where do the mysterious 0.457 and 0.494 come from?" box is exactly the voice the book is after. It stays well within the linear-model ceiling. What holds it back: a few claims are stronger than the math supports (most importantly, the Cressie–Hawkins correction is presented as making the estimator unbiased, but it is derived assuming independent increments); one formula has a wrong symbolic constant; the chapter introduces Genton's matrix $\mathbf{A}(\mathbf{h})$ with great care and then never uses it; the chapter never says that all its estimators assume a constant mean; and §3.5–3.6 (cross-variograms, delay effects) interrupt the univariate story long before the student has fitted or used a single variogram.

**Highest-value changes**
1. State the constant-mean assumption up front and show the one-line consequence of a trend. The chapter opens with the three-layer model including a trend, then quietly assumes no trend.
2. Use the matrix $\mathbf{A}(\mathbf{h})$ the chapter already builds. For Gaussian data, $\operatorname{Cov}(\widehat\gamma(\mathbf{h}_1), \widehat\gamma(\mathbf{h}_2)) = \dfrac{\operatorname{Tr}(\mathbf{A}_1\mathbf{V}\mathbf{A}_2\mathbf{V})}{2|N_1||N_2|}$. I checked that this equals the chapter's long double-sum formula exactly, and both match a 200,000-draw simulation. This replaces a fiddly four-index sum with something a linear-models student recognizes.
3. Correct the claim that Cressie–Hawkins is unbiased for spatial data. In my simulation (Gaussian field, no outliers, exponential covariance) it overestimates by about 10% at moderate lags because pairs are dependent. Say so, and say when it is worth using.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: yes, with one improvable proof.** There is no measure theory, no functional analysis, and no Bessel or Hankel integrals. The Gamma and Beta integrals in the Cressie–Hawkins derivation and Exercise 3.3 are ordinary undergraduate calculus.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L538–543 | Isserlis' identity "proved" by differentiating the joint characteristic function | Low–Medium | This is a "follows from a bigger tool" proof, exactly what the mission asks to avoid, and it is the only step the chapter needs in the special case $\operatorname{Cov}(U^2, V^2) = 2\operatorname{Cov}(U, V)^2$. A linear-model proof is available: write $V = \beta U + W$ with $W$ independent of $U$ ($\beta = \operatorname{Cov}(U,V)/\operatorname{Var}(U)$), so $\operatorname{E}[U^2V^2] = \beta^2\operatorname{E}[U^4] + \operatorname{E}[U^2]\operatorname{E}[W^2] = 3\beta^2\sigma_U^4 + \sigma_U^2(\sigma_V^2 - \beta^2\sigma_U^2) = \sigma_U^2\sigma_V^2 + 2\operatorname{Cov}(U,V)^2$. (Uses only $\operatorname{E}[U^4] = 3\sigma_U^4$.) State the special case; mention the four-variable version as an aside. |
| L75–98 | Graph Laplacian vocabulary for $\mathbf{A}(\mathbf{h})$ | Low | Fine, but the term adds jargon for something the text then does not use. Either use $\mathbf{A}$ (see below) or drop the vocabulary. |
| L826–832 | $\alpha$-trimmed semivariogram with "$k_\alpha(N)$ calibrated via $\chi_1^2$ order statistics" | Low | An unspecified constant with no derivation is a black box. Give it or drop the estimator. |

Nothing else drifts. The chapter's treatment of fractional moments (L732–740) is a good model of proving a "heavy" result with a substitution and a Gamma integral.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 3.1 | Yes (but late; see below) | Yes (explained twice) | Yes |
| 3.2 | Yes (excellent bridge) | Yes (a bias calculator) | Yes (excellent) |
| 3.3 | Yes | Yes | **Thin** |
| 3.4 | Yes (Box–Cox bridge) | Yes | **Thin** (no guidance on when to use it) |
| 3.5 | Yes | Yes (but see the direction issue) | Modest |
| 3.6 | Yes | Yes (coarse) | Diagram contradicts text |
| 3.7 | n/a | n/a | n/a |

### §3.1 Nonparametric Estimators and Spatial Lag Binning

**What works**
- The identity $S^2 = \frac{1}{2n(n-1)}\sum_i\sum_j (Y_i - Y_j)^2$ (L52–60), "sample variance without ever computing the mean," is exactly the right bridge. I checked it.
- The term-by-term anatomy (L107–115) and the practical advice to plot against the *actual mean separation* rather than the nominal bin center (L275) are both good.
- Beat 2 explains the algorithm in four steps (L132–136), then in a procedure box, then in Python. The empirical semivariogram is a core algorithm and is well explained.

**Issues**
1. **The bridge arrives late.** The section opens with a set $N(\mathbf{h})$, tolerance regions, and a cone-frustum defined with an arccos (L17–41) before the reader meets the familiar pairwise-difference identity (L43). Reverse the order: start with "variance from differences," then ask "which differences?" and introduce lag bins as the answer.
2. **Genton's matrix is introduced and then abandoned (L69–98).** The properties (PSD, $\mathbf{A}\mathbf{1} = \mathbf{0}$, trace formula) are shown, but §3.3 re-derives variances from scratch with explicit double sums. Using $\mathbf{A}$ is the payoff (see Executive Summary item 2): for Gaussian $\mathbf{Z}$ and mean-annihilating $\mathbf{A}_1, \mathbf{A}_2$, $\operatorname{Cov}(\mathbf{Z}'\mathbf{A}_1\mathbf{Z}, \mathbf{Z}'\mathbf{A}_2\mathbf{Z}) = 2\operatorname{Tr}(\mathbf{A}_1\mathbf{V}\mathbf{A}_2\mathbf{V})$. It also gives a one-line proof of Theorem 3.2.1 (unbiasedness) from L97.
3. **Constant mean is assumed but never stated up front.** L3–7 restate the model with a trend $\mathbf{x}'\boldsymbol\beta$, then jump to "the stationary residual." Definition 3.1.1 says "intrinsic stationarity with constant mean." If a trend is present, $\operatorname{E}[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2] = 2\gamma(\mathbf{h}) + (\mu(\mathbf{s}_i) - \mu(\mathbf{s}_j))^2$, so Matheron's estimator is biased *upward*, and the bias grows with lag. This is the direct-variogram twin of the mean-offset result the chapter proves for the pseudo-cross variogram (Theorem 3.5.2). One sentence plus that formula, with a pointer to Chapter 6 (residuals), fixes it.
4. **Ordered vs. unordered pairs (L26, L65, L105).** $N(\mathbf{h})$ is defined with ordered pairs, Definition 3.1.1 sums over "distinct ordered pairs," and the algorithm uses unordered pairs ($i < j$). The estimate is the same either way, but say so once.
5. **Anatomy at L115 says "strictly unbiased" without the caveat** that applies only when every pair is at *exactly* lag $\mathbf{h}_k$ (as in Theorem 3.2.1). FAQ 1 later explains that binning introduces bias. Keep the two consistent; note that on a regular grid exact lags exist.
6. **Beat 2 code and docstring.**
   - `if dist < 1e-12: continue` (L215) discards co-located duplicates, which are exactly the pairs that identify the nugget (Chapter 2's own design advice). Keep them in a first bin at lag zero, or say why they are dropped.
   - The docstring calls `angle_tolerance_rad` a "half-tolerance," but the code halves it again (`half_angle = angle_tolerance_rad / 2.0`). The text uses $\Delta\theta/2$, so the *code* is consistent with the text; fix the docstring.
   - Nothing says how to *choose* `lag_centers` and `lag_bandwidth`. A default recipe would help: bandwidth about the typical nearest-neighbor spacing, centers at multiples of it, stop at half the domain extent, merge bins with fewer than about 30 pairs.
   - The procedure box and the Python function say the same thing twice. Keep one; the Python is the more informative.
7. **Beat 3 claim about variance (L254).** "The variance of $\widehat\gamma(\mathbf{h}_k)$ is inversely proportional to the pair count." Pairs are dependent, so this is not even approximately true. §3.3 shows it. In my simulation (40 regular sites, exponential covariance) the true variance at lag 10 is $0.20$, about 4.6 times larger than the independent-pairs figure $2\gamma^2/N = 0.044$. Revise: "fewer pairs means noisier, and dependence makes it noisier still (§3.3)."
8. **The first figure a student expects is missing.** The chapter has no empirical-variogram plot (not even ASCII) and no worked example. The *variogram cloud* (all $(h_{ij}, \frac12 D_{ij}^2)$ points) before binning is the natural first picture.

### §3.2 Expectations, Unbiasedness, and Systematic Downward Bias

**What works.** This is the chapter's best section. The $(n-1)/n$ classical bridge (L283–289) sets up the spatial story perfectly, Theorem 3.2.1's proof is two lines and airtight, "variance theft" (L494–495) is a good name, and the "10,000 points in a 10 km × 10 km basin" example (L497) is concrete.

**Issues**
1. **Theorem 3.2.2's strong form is not proved.** The statement (L348) says $\operatorname{E}[\widehat C(\mathbf{h}_k)] < C(\mathbf{h}_k)$ "for all lags" when $C \ge 0$, and L385–389 give $\approx -\operatorname{Var}(\bar Z)$ as the bias. The proof only gives the exact expression $C - 2\overline C(N,S) + \operatorname{Var}(\bar Z)$ and then says "$\approx$." I computed the exact bias for 60 regular sites with an exponential covariance (scale 10): it is negative at every lag, but its size is about $-0.28$ for lags up to 30 (roughly $-\operatorname{Var}(\bar Z) = -0.278$) and shrinks to about $-0.07$ at the very largest lags (a handful of pairs near the two edges). So "$\approx -\operatorname{Var}(\bar Z)$ at large separations" (L389) holds only inside the recommended lag range (up to half the extent). State the exact formula as the theorem, the leading-order approximation as a remark, and the edge-effect caveat in Beat 3.
2. **Exercise 3.2(c)** states the bias as $-\frac{\sigma^2}{n}\frac{1+\rho}{1-\rho} + O(n^{-2})$. I checked: for $\rho = 0.8$ the exact bias is $-0.087$ at $n = 100$ (the formula gives $-0.090$) but $-0.36$ at $n = 20$ (formula: $-0.45$). The $O(n^{-2})$ term is not small unless $n$ is much larger than the variance inflation factor (here 9). Say that.
3. **Beat 2 (a bias calculator, L404–469).** It is correct (it implements the exact expression). It is not a core algorithm for the reader's learning goal, and it measures bias against $C(h_k)$ at the nominal lag, mixing mean-estimation bias with binning bias. Consider replacing it with a three-number table from the example above: lags 1, 10, 30, 59 with biases $-0.28, -0.30, -0.28, -0.07$. A table teaches the same lesson with less code.
4. **A missing sentence about the semivariogram's own price.** The text correctly says the semivariogram avoids mean estimation. It should also say it is *not free*: it is biased under a trend (§3.1 issue 3), and its values at different lags are strongly correlated (§3.3).
5. **Scope of "bias persists."** Corollary 3.2.1 (infill) is fine, but the chapter earlier (Ch 1) and here repeat the same infill formula yet again (L297–301, L391–396). A cross-reference would do.

### §3.3 Sampling Distributions and Fourth-Order Moments

**What works.** The bridge from $\operatorname{Var}(S^2) = 2\sigma^4/(n-1)$ (L513–517) is right. Proposition 3.3.1 and Theorem 3.3.1 are correct: I checked them against a Monte Carlo simulation of 200,000 Gaussian fields (40 regular sites, exponential covariance). Variance of $\widehat\gamma$ at lags 1, 2, 3, 10: simulation $0.00125, 0.00621, 0.01646, 0.2007$; formula $0.00125, 0.00623, 0.01654, 0.1999$. Correlation between $\widehat\gamma(1)$ and $\widehat\gamma(2)$: simulation $0.793$, formula $0.792$.

**Issues**
1. **Theorem 3.3.1 has no proof** (L588–596). It follows in one line from Proposition 3.3.1: $\widehat\gamma = \frac{1}{2|N|}\sum D^2$, so $\operatorname{Cov} = \frac{1}{4|N_1||N_2|}\sum\sum\operatorname{Cov}(D^2, D^2)$. Add it.
2. **Notation clash.** The index $k$ names the lag bin in $\mathbf{h}_k$ and a *site* in $(\mathbf{s}_k, \mathbf{s}_l)$ in the same formula (L595). Use $(\mathbf{s}_a, \mathbf{s}_b)$ for the second pair.
3. **Isserlis' proof** (see calibration table): use the regression-decomposition proof of the special case.
4. **Beat 2 (L604–657).** Correct and faithful to the theorem. Offered as an *optional* simplification: with $\mathbf{A}(\mathbf{h})$ the whole function is `trace(A1 @ V @ A2 @ V) / (2 * N1 * N2)`. The explanation (iterate over pair sets) is clear; just note the expense ($O(N_1 N_2)$ terms per entry).
5. **Beat 3 is too thin for how important this is (L663–668).** It lists two reasons and a pointer to Chapter 4. Two things a student should see here:
   - *Numbers.* From my example: $\operatorname{Corr}(\widehat\gamma(1), \widehat\gamma(2)) \approx 0.79$, with lag 3 about $0.65$, with lag 10 about $0.29$. Empirical variograms therefore look smoother than they should, and a "nice-looking curve" is partly an artifact of neighbors sharing information.
   - *Correct the explanation.* "The shared observation $Z(\mathbf{s}_2)$ introduces a structural covariance" is half right. Even with *no* shared sites, Gaussian dependence gives $\operatorname{Cov}(D^2, D^2) = 2\operatorname{Cov}(D, D)^2 > 0$. "Always correlated" (L664) overstates; with zero cross-covariances and no shared sites they would be uncorrelated.

### §3.4 Robust Estimation of Spatial Continuity

**What works.** The Box–Cox / square-root bridge (L676–678) is exactly the classical-to-spatial move the mission wants. FAQ 2 (L798–824), "where do the mysterious numbers come from?", is a model of the book's voice. The derivation of $E|U|^{1/2}$ by a substitution and a Gamma integral (L732–744) is pencil-checkable.

**Issues**
1. **The finite-sample correction assumes independent increments, and spatial pairs are not independent.** Theorem 3.4.1 says "under independent increments" (L723), but the estimator definition (L694–699), the FAQ (L824), and Beat 3 then present it as "asymptotically unbiased." In a simulated *Gaussian* field (no outliers), with 40 regular sites and exponential covariance (scale 6), I found the estimator's average relative to the true $\gamma$: $+0.1\%$ at lag 1, $+0.9\%$ at lag 2, $+2.0\%$ at lag 3, and **$+10.1\%$ at lag 10**, while Matheron's estimator was unbiased in all four cases. The $0.494/N$ term corrects for the Jensen effect of averaging *independent* values; with dependent pairs the effective $N$ is smaller and the correction is too small. State this limit plainly. It does not make CH useless, but it affects how it should be recommended.
2. **Wrong symbolic constant (L721, L763).** The text writes $(\operatorname{E}[Y])^4 = 2\gamma\cdot\frac{\sqrt{2}[\Gamma(3/4)]^4}{\pi^2}$. Since $(2^{1/4})^4 = 2$, not $\sqrt{2}$, it should be $\frac{2[\Gamma(3/4)]^4}{\pi^2}$. I checked: the correct expression gives $0.45695$; the written one gives $0.3231$. The *numeric* value is what the rest of the section uses, so the damage is limited to those two lines.
3. **Slightly off decimals.** $B = (0.822177)^4$ is $0.45695$, not $0.457066$ (L710, L767, and the code at L894). Correspondingly $B_1 \approx 0.4944$, not $0.49457$. The published constants $0.457$ and $0.494$ are right to three digits; use those in the code, since over-precise wrong digits look like a mistake.
4. **"Resistant to outliers" (L707).** The anatomy says that because the increments are near-Gaussian "their sample mean is highly resistant to isolated outliers." An average is *not* resistant in the technical sense; it is *less sensitive* because the transform compresses extremes (an outlier enters as $|D|^{1/2}$ instead of $D^2$). Also, "near-Gaussian, so the mean is resistant" is a non-sequitur. Rephrase.
5. **Skewness and kurtosis statements are repeated three times and inconsistent.** L684, L706, L905 repeat skewness $2.83 \to 0.08$. Kurtosis is called $15$ (L684) and $12$ (L905); these are raw and excess kurtosis of $\chi^2_1$. I checked all three numbers by simulation ($2.84$, $0.085$, $15.0$/$12.1$). Say "kurtosis 15 (excess 12)" once, and say it once.
6. **Third-central-moment step (L774–778).** The expansion drops $4\mu_Y\operatorname{E}[\delta^3]$ without saying why. It is $O(N^{-2})$ (third central moment of a mean of $N$ independent terms is $O(N^{-2})$). One clause.
7. **Beat 3 (L904–908) repeats the introduction and gives no decision guidance.** What students need: when to use which estimator. A short rule: plot Matheron and Cressie–Hawkins together; if they agree, use Matheron; if they disagree at a few lags, look for outliers at the sites in those bins; do not "fix" a variogram by switching estimators without finding out why.
8. **"CH underestimates" (L908, Exercise 3.3).** Under contamination, the estimator is *designed* to track the bulk of the data and ignore the contaminants, so it "underestimates" the *total* variance by design. Exercise 3.3's ratio ($0.3617$ for $t_5$; I checked it numerically) correctly shows a 21% shortfall relative to the variance; say the shortfall is the price of robustness.
9. **Trimmed semivariogram** (see calibration table).
10. **Pairwise relative semivariogram** (L833–838) matches the standard definition as I remember it (not independently checked) but arrives with no worked motivation or caveat. If the chapter is trimmed (see §3.5), this is a candidate to drop.

### §3.5 Multivariate Spatial Continuity: Cross-Covariances and Cross-Variograms

**What works.** The isotopic/heterotopic vocabulary is clear. Theorem 3.5.1 and Theorem 3.5.2 are correct and proved with explicit expansions (L954–972, L1001–1017). The unit-sensitivity warning (L1090–1095) is practical and good.

**Issues**
1. **Placement and pace.** This is a *second* topic. Students have not yet fitted or used a single univariate variogram (Chapter 4) or kriged anything (Chapter 5). Cross-variograms are needed for cokriging (Chapter 7). Consider moving §3.5–3.6 to Chapter 7, or keeping them here as an explicitly labeled optional section. I have not yet read Chapter 7; its section titles suggest overlap with §3.5–3.6, which I will check when I get there.
2. **The pseudo-cross code ignores direction (L1023–1078).** It bins by scalar distance, but Proposition 3.6.1 says the pseudo-cross variogram "captures the shift" $\mathbf{v}$, which needs *signed* lag vectors. As written, the function returns an omnidirectional curve that cannot show a delay. Either label it omnidirectional and say that delay detection needs the directional version, or add a direction argument.
3. **Notation.** Theorem 3.5.2 defines $C_{ji}(\mathbf{h}) = \operatorname{Cov}(Z_j(\mathbf{s}), Z_i(\mathbf{s}+\mathbf{h}))$ while Definition 3.5.1 defines $C_{ij}$ with the first index at $\mathbf{s}$. Both are consistent, but a reader will trip on the index swap. One line of explanation.
4. **"Intercept" claim (L1084–1089).** Correct ($\gamma^P_{ij}(\mathbf{0}) = \frac12(\sigma_i^2 + \sigma_j^2) - C_{ij}(\mathbf{0}) > 0$). It deserves a plain-language sentence: even two perfectly aligned variables have a positive pseudo-cross value at lag zero unless they are identical.

### §3.6 The Delay / Lag Effect and Asymmetric Spatial Cross-Continuity

**What works.** The time-series bridge (rainfall to river discharge three days later, L1105–1109) is good. Theorem 3.6.2 (the classical cross-variogram sees only the even part of the cross-covariance) is a real insight, proved in four lines.

**Issues**
1. **The diagram contradicts the text (L1285–1297).** The caption says "Peak at $h = +v$," but the curve's apex sits directly above the "$0$" tick, not above "$+v$." Redraw the peak to the right.
2. **Proposition 3.6.1 needs $\alpha > 0$.** "Achieves its maximum at $\mathbf{h} = \mathbf{v}$" and "reaches its minimum at $\mathbf{h} = -\mathbf{v}$" hold only for positive $\alpha$ (for $\alpha < 0$ they flip). Say so.
3. **The directional cross-covariance code is coarse (L1208–1278).** It classifies each pair as "$+$" or "$-$" by the *sign* of its projection on $\mathbf{u}$, with no angular tolerance. A pair that is nearly perpendicular to $\mathbf{u}$ is counted as "downstream." That is a half-plane split, not a directional estimate along $\mathbf{u}$; the angular filter from §3.1 should be reused. The code also centers by sample means, which reintroduces exactly the downward bias the chapter just taught (Theorem 3.2.2). Acknowledge it or, for this display, subtract the known mean.
4. **Definition 3.6.1** is a *fact* (transpose symmetry), not a definition; call it a proposition. It is one line.
5. **Physical example (L1300).** A smokestack and its downwind deposition is a good example, but the emission field is not stationary. Fine for intuition; add "idealized."

### §3.7 Analytical Exercises

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **7.5 / 10**
- All six numbered sections follow Beat 1, 2, 3. §3.2 is the model for the book.
- Pacing is good through §3.4. It stalls at §3.5–3.6, which add a new topic (multivariate) before the univariate story has paid off.
- **Visible scaffolding:** "Beat 1/2/3" headings again; use student-facing ones.
- **Heavy headings:** e.g., "3.4.1 Classical-to-Spatial Bridge: From Classical Box-Cox / Power Transformations to Cressie-Hawkins Modulus Estimation" (L676). Short heading, then the bridge.
- **The L9 dedication** to Rizzo and S&G should become a sentence about what the student can do after the chapter: build, plot, and interpret an empirical variogram, know when to distrust it, and know that neighboring lags are correlated.
- **Missing signposts:** "Chapter at a glance," a worked example (see below), and a "Key ideas" recap.

### Dimension 2 — Algebraic transparency and scaffolding: **7.5 / 10**
- Strengths: Theorem 3.2.1, Theorem 3.2.2's expansion, Proposition 3.3.1, the Gamma-integral substitution in Theorem 3.4.1, and Theorems 3.5.1–3.6.2 are step by step.
- Gaps: Theorem 3.3.1's proof (one line); the $E[\delta^3]$ term; Isserlis proof by regression decomposition; the unused $\mathbf{A}(\mathbf{h})$.
- **"From the Lecture Hall: Bridging the Prerequisite Gap" callouts: none present.** Natural ones: (1) "Why pairs, not points" ($\operatorname{Var}$ from differences, already in §3.1.1, but boxed); (2) "Isserlis in one idea: write one Gaussian as a regression on the other"; (3) "Jensen's inequality and why $\bar Y^4$ is biased upward"; (4) "Even and odd parts of a function."

### Dimension 3 — Structural consistency and forward referencing: **6.5 / 10**
- **Duplication:** the infill variance formula and "bias persists" appear in §3.2.1 (twice), Corollary 3.2.1, and Chapter 1; skewness $2.83 \to 0.08$ appears three times.
- **Notation:** ordered vs. unordered pairs; $k$ as bin index and site index; $c_0$ for the sill in Exercise 3.1 (conventionally "nugget" in S&G; Chapter 2 uses $\sigma^2$); $\rho$ as a correlation parameter in Exercise 3.2 (Chapter 2 used $\rho$ for radial frequency).
- **Forward references** are well signposted ("see Chapter 4"). Good.
- **Possible overlap with Chapter 7** (to check).

### Dimension 4 — Empirical and computational utility: **6 / 10**
- **No running example, no plot.** A small simulated dataset carried through §3.1–§3.3 would show the cloud, the binned variogram, the bin counts, the error bars, and the correlated neighbors. My numbers from a 40-point 1D example (variances $0.00125 \to 0.20$ across lags 1 to 10, adjacent-lag correlation $0.79$) are exactly the kind of table that would make the theory visible.
- **Exercise 3.1** (covariance of adjacent increments) is correct and elegant: $\operatorname{Cov}(D_1, D_2) = \gamma(2\mathbf{h}) - 2\gamma(\mathbf{h})$, which is $-c_0(1 - e^{-h/\phi})^2 < 0$ for the exponential. Keep. (Rename $c_0$.)
- **Exercise 3.2** (variance inflation on a lattice) is correct (I verified the closed form for $n = 2$ and numerically for larger $n$). Add the note about when $O(n^{-2})$ is small (§3.2 issue 2).
- **Exercise 3.3** (heavy-tailed increments) is correct: I verified $E|D|^{1/2} = 0.8811\,\tau^{1/2}$ and the ratio $0.3617$ numerically. Add the sentence about the price of robustness.
- **Exercise 3.4** (pseudo-cross symmetry) is correct. It repeats Theorem 3.6.2 in a different guise; consider replacing with a numerical exercise (compute $\widehat\gamma$ for six points by hand).
- None of the exercises asks the student to *compute an empirical variogram* from a small dataset, which is the chapter's main skill. One hand-sized exercise (six points, two bins) would help.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Fix $\sqrt{2}$ to $2$** in $(\operatorname{E}[Y])^4 = 2\gamma\cdot\frac{2[\Gamma(3/4)]^4}{\pi^2}$ (L721, L763).
2. **State that Cressie–Hawkins' correction assumes independent pairs** and that for dependent pairs it can overestimate (about +10% at lag 10 in my Gaussian example). Revise the "asymptotically unbiased" claims (L695, L824, L727).
3. **State the constant-mean assumption** at the start, and add the formula for the trend-induced bias in Matheron's estimator, with a pointer to Chapter 6.
4. **Soften or prove Theorem 3.2.2's "for all lags" and "$\approx -\operatorname{Var}(\bar Z)$ at large lags."** State the exact expression; note the edge effect.
5. **Add the proof of Theorem 3.3.1** (one line) and fix the $k$ index clash.
6. **Fix the delay diagram** (L1285–1297) so the peak is at $+v$.
7. **Correct the "variance inversely proportional to pair count" claim** (L254).

### Should fix (pace, flow, voice)
8. **Use Genton's matrix:** give the trace form of the variance and covariance, and the one-line unbiasedness proof; or drop the graph-Laplacian vocabulary.
9. **Replace the characteristic-function proof of Isserlis** with the regression-decomposition proof of the special case.
10. **Move the pairwise-difference identity ahead of the lag-bin definitions** in §3.1.
11. **Add a worked example with a plot or table** (variogram cloud, bins, pair counts, adjacent-lag correlations).
12. **Strengthen Beat 3 of §3.3 and §3.4** (correlation numbers; a rule for when to use which estimator).
13. **Consider moving §3.5–3.6 to Chapter 7,** or label it optional; check against Chapter 7's overlap.
14. **Make the pseudo-cross code directional,** or label it omnidirectional; add an angular tolerance to the directional cross-covariance code.
15. **Give a recipe** for choosing lag centers and bandwidth.
16. **Add four "From the Lecture Hall" callouts** (list above) and a "Chapter at a glance" / "Key ideas."
17. **Replace "Beat" headings and shorten the long section-bridge titles;** replace the L9 dedication.
18. **De-duplicate** skewness/kurtosis and infill statements.

### Nice to have
19. Fix the decimals: $B \approx 0.45695$, $B_1 \approx 0.4944$ (use $0.457$ and $0.494$ in code).
20. "Resistant" to "less sensitive" (L707); "price of robustness" in Exercise 3.3.
21. Keep co-located pairs in a lag-zero bin; fix the docstring on angular tolerance.
22. State $\alpha > 0$ in Proposition 3.6.1; call Definition 3.6.1 a proposition.
23. Rename $c_0$ and the Exercise 3.2 $\rho$; unify ordered/unordered pair language.
24. Replace the Beat 2 bias calculator with a short table (optional).
25. Give $k_\alpha(N)$ or drop the trimmed estimator.
26. Consider one hand-sized "compute an empirical variogram" exercise.

---

## 6. What to Keep Exactly As It Is

- The pairwise-difference identity for $S^2$ (L52–60), as the opening bridge.
- Theorem 3.2.1 and its two-line proof.
- The $(n-1)/n$ bridge into spatial variance inflation (L283–301) and the "variance theft" explanation (L494–497).
- The substitution-and-Gamma-integral derivation of the Cressie–Hawkins constants (L732–796) and FAQ 2 (L798–824).
- Proposition 3.3.1's four-line covariance expansion (L564–586).
- The time-series delay bridge (L1105–1109) and Theorem 3.6.2.
- Exercises 3.1–3.3.
