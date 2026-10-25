# Review: Chapter 8 — Change of Support and Spatial Regularization

**File reviewed:** `ch08_support_regularization.md` (the 1,128-line, 74,046-byte version last modified 20:20; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch08_exercises.md` was not reviewed. The four exercises inside §8.9 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy, and copies of the chapter's code). Literature attributions marked "from memory" are not verified.

---

## 1. Executive Summary

**Grade: 6 / 10**

The chapter's core story is the right one and is told well in places: a measurement over a volume is an average of point values, averaging removes variance but not the mean, and that single fact changes everything from block estimates to mine economics. The ANOVA bridge (L19–47), the FAQ explaining why the mean survives and the variance does not (L108–121), the regularized-covariance theorem and its polarization-identity proof (L246–292), the block-kriging derivation (L747–762), and Exercises 8.1, 8.3, and 8.4 are all good. The weaknesses are concentrated and fixable. This is the most drift-prone chapter so far from the mission, with an existence theorem phrased in mean-square limits, a Dirac-delta "nugget" proposition, and a central limit theorem proved by invoking mixing conditions. And several statements that look like theorems are wrong or unproved, and I could show it numerically: Cartier's relation for a fixed sub-block, the average-of-point-variances identity in the block-kriging corollary, the "4×4×4 grid is accurate to 1%" claim, the Exercise 8.2 auxiliary function, and the affine "negative grade" warning.

**Highest-value changes**
1. Connect the chapter to Chapter 1. A block average over a volume is the sample mean $\bar Z$ of Chapter 1, over a continuum of points: $\operatorname{Var}(Z_V) = \overline{C}(V, V)$ is Chapter 1's $\mathbf{1}'\mathbf{V}\mathbf{1}/n^2$ with integrals for sums, and the long-block limit is Chapter 1's effective sample size, $|V|/A$ with $A$ the "integral range" $\int\rho(\mathbf{h})\,d\mathbf{h}$. I checked it on Exercise 8.1: for an exponential model, $\overline{C}(v,v) \cdot (L/(2a\sigma^2))$ is $0.37, 0.80, 0.95, 0.99$ at $L/a = 1, 5, 20, 100$. This one idea replaces the mean-square-integral machinery and the CLT theorem with something students already own.
2. Give Krige's additivity theorem the proof its own ANOVA bridge calls for: the finite ANOVA identity for one realization, then take expectations. The current "probabilistic proof" mixes up which randomness is meant, and the "algebraic proof" is a telescoping sum over definitions that were never derived.
3. Fix the numerical and logical errors listed in §3 and §4. Two of them silently corrupt a worked example (Exercise 8.2's $\overline{\gamma}(W, W)$ is wrong, yet the additivity "check" passes, because it passes for any numbers).

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: partially.** The chapter leaves the ceiling in four places, more than any earlier chapter.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L73–106 | Theorem 8.1.1: existence of $Z_v$ as a "mean-square limit of Riemann sums," Cauchy convergence in mean square, "possesses continuous sample paths" | **High** | Measure-and-functional-analysis flavor, and the sample-path claim (L81) is asserted with no proof (mean-square continuity does not give continuous paths). Define the block variable by its covariance: $\operatorname{Cov}(Z_v, Z_w) = \overline{C}(v, w)$, and say the interchange of expectation and integration is allowed because $C$ is continuous. Or work with $M$ points first (a finite average) and let $M \to \infty$. |
| L111 | "By Fubini's theorem (Theorem 8.1.1)" | Medium | Mislabels the reference ($8.1.1$ is the existence theorem, not Fubini) and names a measure-theory tool for what is "swap a sum and an expectation." |
| L294–319 | Proposition 8.2.1: nugget as white noise with a **Dirac delta covariance**, "microscale integral capacity," convolution with the geometric covariogram | **High** | Distribution-valued noise is the opposite of the mission. The proposition is also internally inconsistent (below). Replace with the integral-range idea: a micro-scale component with correlation length $a_0$ contributes about $\tau_{\text{micro}}^2\,A/|v|$ to the block variance, where $A \approx a_0^d$ is its integral range. |
| L521–545 | Theorem 8.5.1: a spatial central limit theorem under $\alpha$-mixing, "dominated convergence," "Bernstein's spatial blocking technique" | **High** | A black-box theorem whose proof is a sentence. For this audience, state the result informally (block averages become approximately normal with variance $\sigma_\infty^2/|v|$ when blocks are large relative to the range) and cite a reference. |
| L547–552 | Proposition 8.5.1: skewness is $O(|v|^{-1/2})$, no proof | Low | Say "heuristically, a large block averages about $|v|/A$ nearly independent pieces, so skewness shrinks like $\sqrt{A/|v|}$." |
| L618 | "mean-preserving contraction (second-order stochastic dominance)" | Medium | Use the clean statement: a block average is an average of point-support values, so by Jensen's inequality $\operatorname{E}[\varphi(Z_V)] \le \operatorname{E}[\varphi(Z_v)]$ for every convex $\varphi$. Apply it to $\varphi(z) = (z - z_c)^+$. |
| L53 | "squared sinc-type transfer function" | Low | Fine as an aside; label it optional. |

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 8.1 | Yes (strong bridge) | Yes (explained; mislabelled "Monte Carlo") | Contains a false accuracy claim |
| 8.2 | Yes | In 8.1 | Wrong claims (nugget, "always parabolic") |
| 8.3 | Yes | none needed | Thin |
| 8.4 | Yes | none | Misstates Cartier |
| 8.5 | Yes (short) | **Missing** | **Missing** |
| 8.6 | Yes (short) | **Missing** | **Missing** |
| 8.7 | Yes (short) | **Missing** | Wrong warning |
| 8.8 | Yes | Yes (correct) | Overclaims |
| 8.9 | n/a | n/a | n/a |

### Intro and §8.1 The Volume-Support Problem

**What works.** The motivating examples (drill cores, satellite pixels, mining blocks; L3) are well chosen, and the three nested supports $v \subset V \subset W$ with the "within versus between" reading (L35–47) set up everything that follows. The FAQ (L108–121) is a model of the book's voice: expectation is linear so the mean survives; variance is quadratic so highs and lows cancel. The drill-core versus shovel example (L202–204) is memorable.

**Issues**
1. **The L5 dedication** to Rizzo and S&G ("stochastic foundation") should become what students will be able to do after this chapter: compute a block variance, convert a point histogram into a block histogram, and predict a block value with its uncertainty.
2. **L29: "by the classical Central Limit Theorem, the variance of an arithmetic group mean shrinks in inverse proportion to $m$."** $\operatorname{Var}(\bar Y) = \sigma^2/m$ is variance algebra for independent observations, not the CLT (which is a statement about the *distribution* of $\bar Y$). Say "by the variance of a mean."
3. **L53–55 says low frequencies, including "regional trends," are "preserved exactly."** The *mean* is preserved exactly; a non-constant trend is averaged (preserved at the block center for a linear trend, not for a quadratic one). Say "the mean."
4. **Theorem 8.1.1** (calibration table). Its part 3 (L81) claims continuous sample paths without proof.
5. **L111 reference mislabel** (calibration table).
6. **Beat 2's code is a regular-grid quadrature, not Monte Carlo (name, docstring, and L206).** The function is called `..._mc` and L206 speaks of "Monte Carlo approximation error ... $O(M^{-1/d})$." A midpoint grid is not random. (Monte Carlo error is $O(M^{-1/2})$ in any dimension.) The explanation of the algorithm is otherwise clear, with the discretization diagram, the two double loops, and the average over $M^2$ pairs.
7. **Beat 3's accuracy claim is wrong (L206).** "A $4\times4\times4$ discretization ($M = 64$ points) achieves $< 1\%$ error for isotropic exponential models, whereas $2\times2\times2$ systematically overestimates." I measured the error of the chapter's grid rule for the within-block average covariance of an isotropic exponential model in a cube, against a 4-million-pair Monte Carlo reference:

   | cube side, range | $2^3$ | $4^3$ | $8^3$ |
   |---|---|---|---|
   | side 1, range 1 | +10.9% | +2.0% | +0.44% |
   | side 1, range 0.3 | +55% | +7.0% | +1.3% |
   | side 3, range 1 | +46% | +6.1% | +1.2% |

   So $4^3$ is not within 1%; the right grid depends on block size relative to range, and the rule always *overestimates* (the diagonal pairs contribute $C(\mathbf{0})$). Replace the sentence with this table and the rule of thumb "increase until the answer stops moving." The "overestimates" part is correct.
8. **Discretization quietly reinterprets a nugget.** If the point semivariogram has a nugget, the discretized $\overline{\gamma}(v, v)$ counts the diagonal pairs as zero distance, so a nugget $\tau^2$ contributes $\tau^2(1 - 1/M)$, and the within-block covariance retains $\tau^2/M$: the nugget variance of a block becomes "$\tau^2$ divided by the number of grid points," which depends on the grid. Say so (and see §8.2).

### §8.2 Regularized Covariances and Semivariograms

**What works.** The average-covariance and average-semivariogram definitions (L216–226), the reading of $\overline{C}(A, B)$ as the expected covariance between two uniformly random points (L239), and Theorem 8.2.1 with its proof. I checked the polarization identity (L279) and the derived identity (L281) term by term; the proof is correct and pencil-checkable.

**Issues**
1. **Proposition 8.2.1 is the chapter's weakest piece** (calibration table for the notation). Beyond the Dirac delta:
   - *Item 4 is false in the proposition's own setting.* It says the regularized field is mean-square differentiable "even if the point field is non-differentiable." Block averaging a *continuous* covariance does give a mean-square differentiable field (I checked: for an exponential model with block length 1 and range 1, $\gamma_v(h)/h^2 = 0.599, 0.616, 0.626, 0.629$ as $h \to 0$, parabolic). But the proposition is stated for a field *with a nugget*, and block-averaging micro-scale white noise gives a triangular covariance, linear at the origin, not differentiable: its semivariogram contribution is $\tau_0^2 h/|v|^2$, linear in $h$.
   - *The proof's smoothness step is wrong.* "$K_v$ is Lipschitz ... convolution with any locally integrable kernel produces a *continuously differentiable* function." A Lipschitz function convolved with a locally integrable function is continuous, not necessarily $C^1$. That suffices for item 3 (no jump at the origin) but not item 4.
   - *Measurement error is not filtered by support.* The proposition (and Beat 3, L344) treats the whole nugget as spatial micro-structure that averages down by volume. The nugget has two parts (Chapter 2): micro-scale *variation* (which does average out over a block) and *measurement error* (which does not: an assay of a bulk sample has its own error, whatever the sample's volume). Only the first is filtered.
   - *Units.* L295 defines $\tau_0^2 = \tau^2 a_0^d$ (variance times volume), but L344 writes the block nugget variance as $\tau^2/|v|$ (variance divided by volume), which is dimensionally inconsistent.
2. **Beat 3, "regularized semivariograms *always* leave the origin parabolically" (L345),** is true only when there is no micro-scale component (above).
3. **The most useful practical picture is missing.** The ASCII figure (L325–342) draws the point and block semivariograms but the text never states the three facts a practitioner needs: the block sill is lower ($C(\mathbf{0}) - \overline{\gamma}(v, v)$, as drawn); the *range* of the block variogram is longer (by about the block dimension along each direction); and the nugget is replaced by a smooth rise. For an exponential model on a unit segment I checked the parabolic behavior numerically; give the one-line range statement and a table of sill versus block length (Exercise 8.1's formula is ideal).
4. **Beat 2 is absent here** (the code lives in §8.1), which is fine, but say so.

### §8.3 Dispersion Variance and Krige's Additivity Theorem

**What works.** Krige's additivity as the spatial ANOVA identity is exactly the right framing, and Beat 3's ANOVA comparison (L427–431) is clear.

**Issues**
1. **Both proofs are weaker than they look.**
   - The *algebraic* proof (L383–400) cancels $\overline{\gamma}(V, V)$ across three formulas. But those formulas (Definition 8.3.1(2) and (3)) are *stated*, not derived: the definition says $D^2(v \mid V)$ is "the spatial variance of block averages as $v$ ranges over positions inside $V$" and then gives $\overline{\gamma}(V, V) - \overline{\gamma}(v, v)$ without showing why. So the theorem reduces to a telescoping sum over things that were assumed.
   - The *probabilistic* proof (L402–421) applies the law of total variance and then equates $\operatorname{Var}(Z_v)$ with $D^2(v \mid W)$ (L412). With the field random as well as the block position, $\operatorname{Var}(Z_v)$ is the *point variance of a block*, $\overline{C}(v, v)$ (variance about the true mean), not the dispersion variance (variance about the realized domain average). The two randomnesses (which block, and which realization) are not kept apart.
2. **What to do, using the chapter's own ANOVA bridge.** For one realization, with $z_{kj}$ the values of the small blocks and $\bar z_k$, $\bar z$ the panel and overall averages, the ANOVA identity $\frac{1}{NM}\sum_{k,j}(z_{kj} - \bar z)^2 = \frac1N\sum_k\frac1M\sum_j(z_{kj} - \bar z_k)^2 + \frac1N\sum_k(\bar z_k - \bar z)^2$ is exact algebra. Taking expectations over the field, the three terms are $D^2(v \mid W)$, $D^2(v \mid V)$, and $D^2(V \mid W)$ *by definition*. Then derive the formula for $D^2$ in two lines: the variance of $Z_v - Z_V$ is $\overline{C}(v, v) - 2\overline{C}(v, V) + \overline{C}(V, V)$, and averaging the middle term over all positions of $v$ inside $V$ gives $\overline{C}(V, V)$, because the positions tile $V$. Hence $D^2(v \mid V) = \overline{C}(v, v) - \overline{C}(V, V)$.
3. **No numeric example.** The only one is Exercise 8.2, which is wrong (below). A one-line worked example (spherical or exponential model, 1 m / 4 m / 12 m supports, with a table) would make the additivity concrete.

### §8.4 Cartier's Relation and Variance Reduction

**What works.** The name of the idea (averaging "conserves mass") and its mining reading (Beat 3, L491: a random core from a 2.5 g/t block has expected grade 2.5 g/t) are good.

**Issues**
1. **Theorem 8.4.1 part 2 is false for a fixed sub-block, and its proof invalid (L452–459, L467–470).** Part 1 is correct and trivial: for a *randomly placed* point $S$ (independent of the field), $\operatorname{E}[Z(S) \mid \text{field}] = Z_V$, so $\operatorname{E}[Z(S) \mid Z_V] = Z_V$. Part 2 then asserts $\operatorname{E}[Z_v \mid Z_V] = Z_V$ for a *given* sub-block $v \subset V$ and "proves" it by applying part 1 pointwise to each $\mathbf{u} \in v$. But part 1 holds only on average over positions. For a fixed $v$ the conditional expectation of a Gaussian field is $\mu + \frac{\operatorname{Cov}(Z_v, Z_V)}{\operatorname{Var}(Z_V)}(Z_V - \mu)$, and the slope is not 1. I computed it for an exponential covariance (range 1) on $V = [0, 1]$: slope $0.948$ for the edge sub-block $[0, 0.25]$ and $1.065$ for the central sub-block $[0.375, 0.625]$ (the slopes average to 1 over positions, which is the true Cartier statement). The theorem should say "a *randomly positioned* sub-block."
2. **Corollary 8.4.1's proof inherits the error (L480–485),** though its conclusion is correct. A valid short proof: if $V$ is a union of translates $v_1, \dots, v_m$ of $v$, then $Z_V$ is the average of $Z_{v_1}, \dots, Z_{v_m}$, so $\operatorname{sd}(Z_V) \le \frac1m\sum\operatorname{sd}(Z_{v_j}) = \operatorname{sd}(Z_v)$ (the same triangle-inequality step as Theorem 2.3 in Chapter 2). Or use the ANOVA identity above.
3. **The FAQ (L493–508) blurs two things.** Cartier's relation is a property that *true* block values satisfy automatically; it is not something an analyst "enforces." The over-estimation of recovered metal from point data arises because point values have a wider distribution (the support effect), which is §8.5–8.6. The FAQ's closing sentence says Cartier's relation "guarantees that change-of-support models (such as the Discrete Gaussian Model or Indirect Lognormal Correction) satisfy ..." it; rather, these models are *designed* to respect it. Reword.

### §8.5 Distributional Transformations under Change of Support

**What works.** The list of three effects (variance falls, shape approaches normal, skewness fades; L516–519) is the right summary.

**Issues**
1. **Theorem 8.5.1 is the chapter's biggest calibration problem** (table). It also has a link the chapter never draws: with $\sigma_\infty^2 = \int C(\mathbf{h})\,d\mathbf{h}$, the block variance is $\operatorname{Var}(Z_V) \approx \sigma_\infty^2/|V|$, i.e., $C(\mathbf{0})\,A/|V|$ with integral range $A = \sigma_\infty^2/C(\mathbf{0})$. For the exponential model on a segment, $A = 2a$, and Exercise 8.1's limit $2\sigma^2 a/L$ is exactly this (I verified the ratio converges: $0.37, 0.80, 0.95, 0.99$ at $L/a = 1, 5, 20, 100$). Chapter 1's $n_{\text{eff}} = n/(1 + (n-1)\bar\rho)$ is the same idea in discrete form.
2. **Beat 2 and Beat 3 are absent.** A numeric illustration (the shape of the block histogram of a lognormal point distribution, with skewness for 1, 10, 100 effective cells) would do.
3. **Proposition 8.5.1 is unproved** (table).

### §8.6 Selectivity Curves and Cut-off Economics

**What works.** Definition 8.6.1 (tonnage, quantity of metal, mean recovered grade) is clear and the identity $Q(z_c) = z_cT(z_c) + \int_{z_c}^\infty T(z)\,dz$ is correct. The pair of density sketches (L562–581) is a good picture.

**Issues**
1. **Theorem 8.6.1's hypothesis "crossing point $z^* \approx m$" is misleading for skewed distributions, and its own exercise contradicts the picture.** In Exercise 8.4's lognormal example the point and block distributions cross at $z^* = 3.56$, well above the mean $m = 2.72$. I computed the tonnage at several cut-offs: at $z_c = 1$: block $0.943$ vs. point $0.692$; at $2$: $0.617$ vs. $0.423$; at $3$ (above the mean): $0.325$ vs. $0.275$ (the *block* tonnage is larger even though $z_c > m$); at $3.56$: equal; at $5$: $0.081$ vs. $0.134$; at $8$: $0.012$ vs. $0.057$. So "for cut-offs above the mean, blocks recover less tonnage" is true only above the crossing point, and for skewed grades that point lies above the mean.
2. **The proof is a sketch.** Part 1 asserts "$F_V$ is a mean-preserving contraction of $F_v$" without reason; use the Jensen argument in the calibration table, which gives, for every cut-off, $\operatorname{E}[(Z_V - z_c)^+] \le \operatorname{E}[(Z_v - z_c)^+]$, i.e., the *excess metal above the cut-off after subtracting $z_c$ times the tonnage* is smaller for blocks. Part 3 (mean recovered grade decreases with support) is "proved" by the sentence "because probability mass on $V$ is concentrated tightly near $m$, the tail conditional expectation is pulled down." That is not a proof; convex ordering does not by itself give it. It is true for the normal and affine models and held for the lognormal example I checked (for all $z_c$ from the mean up to 40), but state which model it is proved for. The derivative formula quoted ($dm/dz_c = \frac{f(z_c)}{T(z_c)}[m(z_c) - z_c]$) describes dependence on the cut-off, not on support, and does not bear on the comparison.
3. **An unstated assumption.** The "probability" $T_v(z_c) = P(Z_v \ge z_c)$ is used as the *fraction of the deposit* with block grade above the cut-off. That identification (spatial histogram equals probability distribution) is the ergodic assumption of Chapter 1. Say it.
4. **Beats 2–3 are absent.** The worked example is in §8.9; reference it here.

### §8.7 Analytical Change-of-Support Models

**What works.** The affine model (L630–639) and the indirect lognormal model (L650–678, with the "complete the square" proof for $Q_v$) are clear and the derivations are short and correct. The remark that the affine model preserves skewness (and therefore cannot be right for skewed grades) is exactly the lesson.

**Issues**
1. **The affine warning (L641–642) is false.** It says the model "can predict impossible negative grades when $r < 1 - \frac{m}{m - z_{\min}}$." For $0 < r \le 1$ the block value is $(1 - r)m + r z$, a weighted average of $m$ and a point value $z \ge 0$; it can never be negative. (The threshold quoted is itself negative for any $z_{\min} \ge 0$, so the condition can never hold.) The hazard is the *reverse* direction: expanding block values to point values ($r > 1$).
2. **The lognormal model is a marginal approximation, and should say so.** Averages of lognormal variables are not exactly lognormal. The "indirect" correction matches the mean and variance with a lognormal. Say "approximation," and say it preserves the *ranks* of grades (same quantile ordering) which is its practical appeal. (The chapter calls the numbering "Theorem 8.7.2" with no 8.7.1; the affine statement is a definition.)
3. **Missing:** the connection back to §8.2: the only spatial information these models use is the single number $\overline{C}(v, v)$. That is the honest limitation, and worth a sentence.

### §8.8 Block Kriging

**What works.** The derivation (L747–762) is clean and correct, the anatomy block explains every term (especially why $-\overline{\gamma}(v, v)$ is subtracted, L738–745), the observation that $\boldsymbol\Gamma$ does not depend on the block (L732) is exactly right, and Corollary 8.8.1's first claim is correct and elegant. I checked the chapter's block-kriging code against an independent solve (weights match), and verified numerically that the average of point-kriging weights over the block equals the block-kriging weights.

**Issues**
1. **Corollary 8.8.1's variance identity is false (L769–772, L780).** The corollary says the average of the point-kriging variances over a block equals the block-kriging variance plus $\overline{\gamma}(v, v)$. The proof averages $\boldsymbol\lambda(\mathbf{u})'\boldsymbol\gamma(\mathbf{u}) + \mu(\mathbf{u})$ and claims it equals $\boldsymbol\lambda_v'\overline{\boldsymbol\gamma}_v + \mu_{\text{OK}}$, but the average of a *product* ($\boldsymbol\lambda(\mathbf{u})$ times $\boldsymbol\gamma(\mathbf{u})$, both varying with $\mathbf{u}$) is not the product of averages. Numerical check (1D, exponential variogram with range 1.5, five data points, block $[2, 4]$): block-kriging variance $0.0949$; average of point-kriging variances $0.3151$; $\overline{\gamma}(v, v) = 0.3284$; the chapter's identity predicts $0.4233$. The correct statement is
   $$\text{average point variance} = \text{block variance} + \overline{\gamma}(v, v) - \text{(within-block dispersion of the kriged surface)},$$
   which I verified: $0.3284 - 0.1083 = 0.2202$, and the numerical difference is $0.2202$. The chapter's equality holds only when the kriged surface is constant over the block (a single datum, or a block far from the data). That is exactly Exercise 8.3's situation, which is why the exercise works. The qualitative message (point variances averaged over a block overstate the block's uncertainty) is right; the exact formula needs this correction.
2. **Beat 3's "data point inside the target block" bullet (L894–895)** contradicts Exercise 8.3. It says a single borehole inside a block "does *not* receive weight 1.0." With a single datum the ordinary-kriging weight must be exactly 1 (Exercise 8.3(a)); with other data present it does not. Say "when other data are present."
3. **Discretized nugget** (see §8.1 issue 8): in this code the diagonal pairs have $\gamma(\mathbf{0}) = 0$ and the other pairs get the full nugget; the result depends on $M$.
4. **No numeric example,** and no mention of the practical lesson that block-kriged estimates are smoother than true block grades, so applying a cut-off to the *kriged* values is not the same as applying one to the true block distribution (the "information effect" and conditional bias). A short remark would connect §8.6–8.8.

### §8.9 Analytical Exercises

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **6.5 / 10**
- The first two sections and the last (block kriging) are well paced, and the ANOVA bridge sets up the key idea.
- Sections 8.3–8.7 are Beat 1 (plus an occasional Beat 3) with no Beat 2 and often no example. The chapter's most important applied claims (grade–tonnage, support effect on recovery) get one worked example, at the very end.
- **A missing bridge:** the connection to Chapter 1's variance of a mean (above) is the natural classical-to-spatial move, and is not made.
- **Missing signposts:** "Chapter at a glance," a single running example (one mine-style problem from point variogram to block variance to recovered tonnage), and a "Key ideas" recap.
- **Visible scaffolding:** "Beat" headings, long bridge titles; the L5 dedication.

### Dimension 2 — Algebraic transparency and scaffolding: **6 / 10**
- Strengths: Theorem 8.2.1's proof, the block-kriging derivation, the lognormal formulas, and Exercises 8.1, 8.3, 8.4.
- Gaps: the formula for $D^2(v \mid V)$ is asserted; Cartier part 2; Theorem 8.6.1; Corollary 8.8.1; Theorem 8.5.1 and Proposition 8.5.1 are black boxes.
- **"From the Lecture Hall" callouts: none present.** Natural ones: (1) "A block average is Chapter 1's sample mean over a continuum" (with the integral range); (2) "Spatial ANOVA: the same identity you know from one-way ANOVA, with expectations"; (3) "Why a bigger sample volume means less variance but the same mean"; (4) "Reading a cutoff: tonnage, metal, and recovered grade."

### Dimension 3 — Structural consistency and forward referencing: **5.5 / 10**
- **Notation:** $v$ small and $V$ large throughout, but §8.4's FAQ writes $Z(v)$ and $Z(V)$ for a point and a block; $D^2(\mathbf{0} \mid V)$ uses $\mathbf{0}$ for "point support" (fine, say so); the multiplier convention is consistent with Chapter 5.
- **Dirac delta** is used for the nugget while Chapters 1–2 treat the nugget as an independent error with variance $\tau^2$ (no delta).
- **Theorem numbering:** "Theorem 8.7.2" with no 8.7.1; the Fubini reference is mislabelled.
- **Rendering:** all displayed math is in ordinary Markdown here (good), unlike Chapters 2 and 7.

### Dimension 4 — Empirical and computational utility: **5.5 / 10**
- **Exercise 8.1** (regularized covariance of a segment under the exponential model): correct and good. The Taylor-expansion limit and the $L \to \infty$ behavior are exactly right. Add one sentence: $2\sigma^2 a$ is the integral of $C$, and the result is Chapter 1's effective-sample-size idea for a continuum (verified numerically above).
- **Exercise 8.2** (dispersion variances for a spherical model): **the auxiliary function for $L > a$ is wrong.** The chapter gives $\chi(L) = c\left(1 - \frac{3a}{4L} + \frac{a^3}{20L^3}\right)$. Direct integration (I checked by numerical quadrature) gives $\chi(L) = c\left(1 - \frac{3a}{4L} + \frac{a^2}{5L^2}\right)$. The chapter's version is not continuous at $L = a$ (it gives $0.30c$ where the first formula gives $0.45c$). The correct values: $\overline{\gamma}(v, v) = 0.8310$ and $\overline{\gamma}(V, V) = 3.1852$ (both as in the chapter), but $\overline{\gamma}(W, W) = 6.75$ (the chapter has $6.3125$), so $D^2(V \mid W) = 3.5648$ (chapter: $3.1273$) and $D^2(v \mid W) = 5.919$ (chapter: $5.4815$). The additivity check passes with either set of numbers (the identity is a telescoping sum), so the check cannot detect the error. This makes it a poor verification exercise: better to verify *independently*, e.g., compute $D^2(v \mid W)$ by averaging the semivariogram over all pairs of 1 m segments, or by the ANOVA identity on a simulated realization.
- **Exercise 8.3** (single datum): correct, and a nice minimal example. Note it is also the special case where the corollary's equality holds (see §8.8).
- **Exercise 8.4** (indirect lognormal selectivity): the arithmetic is correct (I recomputed: $m = 2.718$, $\sigma_0^2 = 12.70$, $\sigma_{Y_v}^2 = 0.2914$, $\mu_{Y_v} = 0.8543$; tonnage $0.3254$ vs. $0.2747$; metal $1.4536$ vs. $1.7830$; mean recovered grade $4.47$ vs. $6.49$). But the closing sentence ("a 31.2% over-estimation of grade") is mislabelled: the point-support figure is *45.3% higher* than the block figure; equivalently the block figure is $31.2\%$ *lower* than the point figure. And the more telling comparison is metal: point-support evaluation overstates recoverable metal by $22.7\%$. Also point out that block tonnage is *higher* at this cut-off (crossing point $3.56$), which surprises students: more blocks qualify, at a lower mean grade.
- **Missing:** a numerical exercise that uses the chapter's own code (compute $\overline{C}(v, v)$ for a 2D block with an exponential model at two grid sizes and compare with the table above); an exercise that goes from a point variogram to a block variogram (sill, range); and a data-style example.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Fix Exercise 8.2's $\chi(L)$ for $L > a$** ($a^2/(5L^2)$) and its results ($\overline{\gamma}(W, W) = 6.75$, etc.); give the additivity exercise an *independent* check.
2. **Fix Corollary 8.8.1's variance identity:** average point variance $=$ block variance $+ \overline{\gamma}(v, v) -$ within-block dispersion of the kriged surface; note the equality is special to a constant kriged surface.
3. **Fix Cartier part 2** (randomly positioned sub-block; slope is not 1 for a fixed sub-block) and re-prove Corollary 8.4.1 (average of translates, or ANOVA).
4. **Fix Proposition 8.2.1:** drop the Dirac delta; correct item 4 (blocks of a *smooth* field are differentiable; micro-scale noise averaged over a block is linear at the origin); separate micro-scale variation (filtered) from measurement error (not filtered); fix the units mismatch.
5. **Fix the grid-accuracy claim** (the table above); rename the "Monte Carlo" function and text.
6. **Fix the affine "negative grade" warning** (impossible for $r \le 1$).
7. **Fix Theorem 8.6.1's "crossing point $\approx m$" assumption** and replace the proof by the Jensen argument; state what is and is not proved for the mean recovered grade.
8. **Fix Exercise 8.4's closing percentages.**

### Should fix (pace, flow, voice)
9. **Add the Chapter 1 bridge** ($\operatorname{Var}(Z_V)$ as $\mathbf{1}'\mathbf{V}\mathbf{1}/n^2$ with integrals; integral range; $|V|/A$ effective cells).
10. **Rewrite Krige's additivity proof** via the finite ANOVA identity, with the two-line derivation of $D^2(v \mid V)$.
11. **Replace Theorems 8.1.1 and 8.5.1** with a definition by covariance and an informal statement with a pointer; drop "mean-square limit," "Fubini," "$\alpha$-mixing."
12. **Add the block-variogram facts** (lower sill, longer range, no nugget jump) with Exercise 8.1's formula as a table.
13. **Add Beats 2–3 or at least a worked example** to §8.5–8.7; point to §8.9.
14. **State the ergodic identification** (spatial histogram = probability distribution) in §8.6.
15. **Add the remark about kriged blocks vs. true blocks** and the information effect.
16. **Fix the "single borehole" bullet** to match Exercise 8.3.
17. **Add four "From the Lecture Hall" callouts** and "Chapter at a glance" / "Key ideas."
18. **Replace "Beat" headings;** replace the L5 dedication.

### Nice to have
19. Correct "by the classical Central Limit Theorem" (L29) to "by the variance of a mean."
20. "Preserved exactly" applies to the mean, not trends (L53–55).
21. Say the lognormal correction is an approximation that preserves ranks.
22. Fix the theorem numbering (8.7.1) and the Fubini label.
23. Explain the grid-dependent nugget.
24. Add a numerical exercise using the chapter's code, and a point-to-block variogram exercise.

---

## 6. What to Keep Exactly As It Is

- The ANOVA bridge (L19–47) and the FAQ on why the mean survives and the variance does not (L108–121).
- Theorem 8.2.1 and its polarization-identity proof (L246–292).
- The drill-core versus shovel example (L202–204).
- The block-kriging derivation, the term-by-term anatomy (especially "decoding $-\overline{\gamma}(v, v)$"), and the first half of Corollary 8.8.1.
- The affine and indirect lognormal formulas, and the "affine preserves skewness" lesson.
- Exercises 8.1, 8.3, and (with the percentages fixed) 8.4.
