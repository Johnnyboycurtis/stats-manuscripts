# Review: Chapter 13 — Non-Stationary Covariance Architectures & Scalable Low-Rank Kriging

**File reviewed:** `ch13_nonstationary_covariance.md` (the 519-line, 31,142-byte version last modified Oct 22 22:37:49; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch13_exercises.md` was not reviewed. The chapter defers the proof of its main theorem, the Fixed Rank Kriging matrix mechanics, and the multidimensional scaling derivation to that file (L518), so several results are *stated* here and *proved* elsewhere.
**Version note:** this is the oldest file in the manuscript (a day older than any other chapter I reviewed) and, like Chapter 12, is in a different format from Chapters 9–11: no numbered theorems, no Beat headings, no in-chapter worked exercises. It looks like a draft that has not yet been brought up to the current template. I review it as it stands.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy, and the chapter's own code, exec'd from the file). Statements marked "from memory" or "by reasoning" were not tested.

---

## 1. Executive Summary

**Grade: 5 / 10**

The chapter has two genuinely strong ideas, each stated well. The first is the positive-definiteness argument for process convolutions (L78–84): *a covariance built as $\int k_{\mathbf s_i}k_{\mathbf s_j}$ is the inner product of two functions, so $\mathbf a'\mathbf V\mathbf a$ is the integral of a square and can never be negative.* That is exactly the kind of one-line reason that makes a hard idea feel obvious. The second is the Sherman–Morrison–Woodbury step for Fixed Rank Kriging (L231–239): invert an $r\times r$ matrix instead of an $n\times n$ one. I verified the identity (error $1.6\times10^{-13}$), and I verified that the chapter's FRK code returns exactly the same predictions as an exact $n\times n$ kriging with the same $\mathbf K$ (largest difference $4\times10^{-13}$).

The weaknesses are of three kinds.

1. **The central theorem does not match the kernel it is derived from.** The Gaussian kernel written at L94 does not produce the Paciorek–Schervish covariance at L102. By direct integration, it gives a correlation of $\exp(-Q_{ij}/4)$ (not $\exp(-Q_{ij}/2)$, the Gaussian limit of the chapter's Matérn) and a variance of $(4\pi)^{-d/2}$ (not 1). And a Gaussian kernel can only ever give a *Gaussian* covariance; the Matérn family does not come from a Gaussian kernel. The theorem is cited, not proved, and the proof the chapter points to would not go through as written.
2. **Several statements in the practical parts are wrong or misleading.** Exact kriging at $n=10^5$ is said to take "several weeks of CPU time" (L249), when $\tfrac13n^3\approx3.3\times10^{14}$ operations is about 9 hours at 10 GFLOP/s and about one hour at 100. The memory (80 GB) is the real obstacle, not time. And the final FRK demonstration (L505) predicts at three points whose true value is exactly zero, so the printout cannot show whether the model works.
3. **The chapter stops where the statistics begins.** It shows how to *build* a nonstationary covariance and how to *invert* a low-rank one, but not how to *estimate* either: how to choose $\boldsymbol\Sigma(\mathbf s)$ from data, how to estimate $\mathbf K$ (the code hard-codes `K_diag = 2.0`), how to tell that a field is nonstationary in the first place. There is no Beat 3 anywhere, and no worked example.

**Highest-value changes**
1. Fix the kernel/theorem mismatch (a three-line change to the kernel exponent and normalization) and add the two-Gaussians-multiply proof for the Gaussian case. Say plainly that Matérn follows from a different argument.
2. Unify §13.2 and §13.5: a process convolution on a grid of knots **is** a low-rank basis expansion $\mathbf Z=\mathbf S\boldsymbol\eta$ with $\boldsymbol\eta$ i.i.d., and §13.5's model with $\mathbf K=\mathbf I$ is exactly that. Right now the chapter reads as two unrelated topics ("non-stationarity" and "big $n$") under one title, and this one observation joins them.
3. Replace the FRK demo with one that can fail (score predictions against truth on a grid; I got an RMSE of 0.167 against 0.499 for predicting zero) and add a diagnostics beat: how to detect nonstationarity, and how to tell it apart from a trend (Chapter 6).

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes, with three leaks.** The linear algebra (determinants, Woodbury, MDS) and the Gaussian integrals are within the ceiling. The leaks are the white-noise integral, one matrix inequality, and an unexplained spline.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L57–66, L72–75 | "Gaussian white noise on $\mathbb R^d$ with Dirac-delta covariance"; $W(\mathrm d\mathbf u)$; "$\operatorname{E}[W(\mathrm d\mathbf u)W(\mathrm d\mathbf v)]$" | **High** | Distribution-valued noise and stochastic-integral notation: the exact vocabulary the mission bans (compare my Chapter 8 flag on the Dirac-delta nugget). Replace with the finite version: pick knots $\mathbf u_1,\dots,\mathbf u_m$ and let $Z(\mathbf s)=\sum_{l}k_{\mathbf s}(\mathbf u_l)\,w_l$ with $w_l$ i.i.d. $\mathcal N(0,1)$. Then $\operatorname{Cov}=\sum_lk_{\mathbf s_i}(\mathbf u_l)k_{\mathbf s_j}(\mathbf u_l)$, which is $\mathbf S\mathbf S'$ for the matrix $\mathbf S$ of kernel values, and the positive-definiteness argument is one line of matrix algebra. Mention the integral as the limit as knots get dense. |
| L138 | "arithmetic–geometric mean inequality for matrices" | Medium | True, but a named inequality for a fact a student can test: $\bigl|\tfrac12(\boldsymbol\Sigma_i+\boldsymbol\Sigma_j)\bigr|\ge|\boldsymbol\Sigma_i|^{1/2}|\boldsymbol\Sigma_j|^{1/2}$ (in one dimension this is $\tfrac{a+b}2\ge\sqrt{ab}$, which every student knows). Give the 1D case and say the matrix version holds. |
| L180–184 | Thin-plate spline $f(x,y)=\mathbf A(1,x,y)'+\sum_m\mathbf w_m\phi(\|\mathbf s-\mathbf s_m\|)$, $\phi(r)=r^2\log r$ | Medium | Not explained. A student has met smoothing splines or at best polynomial regression. One sentence ("a flexible surface that bends as little as possible while passing near the points") and the side conditions are enough, or move to a Lecture Hall box. |
| L109–112 | Matérn with the modified Bessel function $K_\nu$ | Accepted | Per the agreed handling, an acceptable named definition. Add the closed forms for $\nu=\tfrac12$ and $\tfrac32$ (see §3), so no Bessel function is needed in practice. |
| L6 | "Bochner's theorem" prerequisite | Accepted | Fine as a pointer. |

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 13.1 | Yes (good) | n/a | A caution box; no diagnostics |
| 13.2 | Yes | **Missing** | **Missing** |
| 13.3 | Yes | In §13.6 code | **Missing** |
| 13.4 | Yes | Described in four steps; no code | **Missing** (folding, smoothing) |
| 13.5 | Yes | Inversion only; no estimation | Overclaims |
| 13.6 | n/a | Two engines | Demo uninformative |

### Header and §13.1 From Homoskedastic Stationarity to Spatially Varying Geometries

**What works.** The framing is excellent: Gauss–Markov assumes constant variance; geostatistics assumed constant everything; real fields (ridges versus valleys, plains versus mountains, a curved coastline) violate each piece differently (L34–37). The ASCII figure (L21–32) with a rotated ellipse, a circle, an elongated ellipse and a sheared ellipse is exactly the right picture. The caution box (L39–44) names the practical costs.

**Issues**
- **The classical bridge stops one step short.** The classical response to non-constant variance is *weighted least squares* or modeling $\operatorname{Var}(\epsilon_i)$ as a function of covariates. The spatial analogue is the simplest nonstationary covariance there is: $C(\mathbf s_i,\mathbf s_j)=\sigma(\mathbf s_i)\sigma(\mathbf s_j)\rho(\|\mathbf s_i-\mathbf s_j\|)$, valid for *any* positive function $\sigma(\cdot)$ because it is $\mathbf D\mathbf R\mathbf D$ with $\mathbf D$ diagonal. Starting there gives the student a first, easy, valid nonstationary model and then shows why varying the *range and direction* (not only the variance) is the hard part. The chapter jumps straight to the hard part.
- **A second easy bridge is missing: geometric anisotropy (Chapter 2) is a single global stretching and rotation of the coordinates.** Sampson–Guttorp (§13.4) lets the stretching vary from place to place; Paciorek–Schervish (§13.3) lets the local ellipse vary. Say so; it organizes the whole chapter as "the same idea at three levels of flexibility."
- **How do you know a field is nonstationary? The chapter never says.** Candidate diagnostics: fit variograms in moving windows or subregions and compare ranges and sills; map the residual variance; compare directional variograms across regions. It must also warn that *much apparent nonstationarity is an unmodeled trend* (Chapter 6, universal kriging). Fix the mean first; model the covariance second.
- **The caution box (L39–44) mixes three claims with no numbers.** Item 3 ("the kriging prediction variance only reflects sample geometry") is correct and important: for known covariance parameters, OK variance does not depend on the observed values. It deserves its own sentence and perhaps a small figure.
- **No dedication paragraph** (a relief) but no opening promise either. A sentence like *"By the end you will be able to write down a valid covariance whose range and direction change across the map, and to krige 100,000 points on a laptop"* orients the reader.
- **Beat 2/3:** none. A natural Beat 3 content: *when is nonstationary modeling worth the cost?* (more parameters, harder estimation, and the stationary model may be adequate for prediction even when it is wrong for uncertainty).

### §13.2 Process Convolutions

**What works.** The central argument (L78–84) is the best passage in the chapter. The derivation $C(\mathbf s_i,\mathbf s_j)=\int k_{\mathbf s_i}k_{\mathbf s_j}$ (L69–76) is correct, and the statement that positivity holds "regardless of how radically the kernel shape changes across space" is exactly the right take-away. It also explains why an arbitrary function of two locations is not a covariance (L52).

**Issues**
- **The white-noise-integral formulation hides the idea.** See §2. The finite-knots version is easier, is exactly what is computed in practice, and leads directly to §13.5.
- **"Positive semi-definite" (L84) is what is shown.** Strict positive definiteness requires the kernel functions $k_{\mathbf s_1},\dots,k_{\mathbf s_n}$ to be linearly independent, true for Gaussian kernels at distinct locations. A clause suffices.
- **The chapter does not connect §13.2 to §13.5.** In the finite version, $\mathbf Z=\mathbf S\mathbf w$ with $\mathbf S$ the $n\times m$ matrix of kernel values and $\mathbf w\sim\mathcal N(\mathbf 0,\mathbf I)$. That *is* §13.5's model with $\boldsymbol\eta=\mathbf w$ and $\mathbf K=\mathbf I$, and its covariance $\mathbf S\mathbf S'$ is nonstationary whenever the kernels differ in shape or size across space. One paragraph makes the chapter a single story: *move the kernels around to get nonstationarity; use few knots to get speed.*
- **Beat 2 and 3 absent.** A recipe: choose knots on a grid; for each data location compute kernel values at the knots; the covariance is $\mathbf S\mathbf S'$. Pitfalls: knots too sparse give artifacts (a lattice pattern in the correlation); the kernel must be wide enough to overlap neighboring knots.
- **Credit and detail.** The chapter attributes the method to Higdon and to Barry & Ver Hoef (L54). Fine; but "solved this" is generous: the convolution gives a *valid* covariance and leaves open how to *choose* $k_{\mathbf s}$ from data. Say so.

### §13.3 The Paciorek–Schervish Theorem

**What works.** The final formula (L102–108) is the right object, and the anatomy box (L114–132), the four-point reading (L135–139), and the reduction to the stationary case (L139) are well done: the stationary limit shows the prefactor equal to 1 and $Q_{ij}=\|\mathbf s_i-\mathbf s_j\|^2/\phi^2$. The code reproduces the formula.

**Issues**
- **The kernel at L94 does not yield the theorem at L102.** I integrated the product of two kernels numerically (one dimension, $\Sigma_i=0.3$, $\Sigma_j=0.7$, locations 0 and 0.6) and compared to the closed form:

  | Kernel | $\int k_ik_j\,du$ divided by its own variance | Closed form with $\exp(-Q/4)$ | Closed form with $\exp(-Q/2)$ |
  |---|---|---|---|
  | Chapter's L94 kernel (exponent $-\tfrac12(\mathbf u-\mathbf s)'\boldsymbol\Sigma^{-1}(\mathbf u-\mathbf s)$, constant $(2\pi)^{-d/2}|\boldsymbol\Sigma|^{-1/4}$) | 0.7996 | **0.7996** | 0.6679 |
  | Same shape without the $\tfrac12$ (exponent $-(\mathbf u-\mathbf s)'\boldsymbol\Sigma^{-1}(\mathbf u-\mathbf s)$), constant $(2/\pi)^{d/4}|\boldsymbol\Sigma|^{-1/4}$ | 0.6679 | 0.7996 | **0.6679** |

  So the chapter's kernel produces a correlation $\exp(-Q_{ij}/4)$ with variance $(4\pi)^{-d/2}$ rather than 1. The chapter's Matérn $\mathcal M_\nu$ (L111) tends, as $\nu\to\infty$, to $\exp(-d^2/2)=\exp(-Q/2)$, and the code uses $\exp(-0.5\,Q)$ (L353). To make kernel, theorem, and code agree, drop the $\tfrac12$ in the kernel exponent and use the constant $(2/\pi)^{d/4}|\boldsymbol\Sigma|^{-1/4}$ (the second row; its variance is exactly 1).
- **A Gaussian kernel gives a Gaussian covariance, not a Matérn.** L92 says the theorem follows by taking the kernel to be Gaussian, and L100 then states the Matérn result. With Gaussian kernels the convolution can only give the Gaussian ($\nu=\infty$) case. The Matérn version follows from a different fact (from memory: any stationary isotropic correlation function that is valid in every dimension can be written as a mixture of Gaussians, and the same mixture applied to the nonstationary Gaussian form stays valid). The chapter should state the Gaussian case with its short proof (below), and then say the Matérn extension holds by a mixture argument, citing Paciorek and Schervish (2006).
- **A short proof is available and is within the ceiling.** For the Gaussian case, multiply two Gaussian densities and complete the square: the product is a Gaussian in $\mathbf u$ times a function of $\mathbf s_i-\mathbf s_j$ alone, and the leftover Gaussian in $\mathbf s_i-\mathbf s_j$ has "variance" $\boldsymbol\Sigma_i+\boldsymbol\Sigma_j$. The determinants in the prefactor come from $|\boldsymbol\Sigma_i^{-1}+\boldsymbol\Sigma_j^{-1}|=|\boldsymbol\Sigma_i+\boldsymbol\Sigma_j|/(|\boldsymbol\Sigma_i||\boldsymbol\Sigma_j|)$. That is about 8 lines, it is the *whole* theorem for $\nu=\infty$, and it removes the black box. At present the proof is in the companion file, and the chapter's claim that Gaussian kernels give the Matérn theorem would not survive that proof.
- **The prefactor is what makes the covariance valid; show it.** I tested positive definiteness on 80 random points with a varying and rotating $\boldsymbol\Sigma(\mathbf s)$ in 2D. With the determinant prefactor the matrices are positive semi-definite for the Gaussian, Matérn-$\tfrac32$, and exponential cases (smallest eigenvalues $7.7\times10^{-10}$, $2.7\times10^{-4}$, $2.7\times10^{-2}$). Dropping the prefactor (the natural "obvious" attempt of averaging the metrics and stopping) gives an *invalid* covariance for the Gaussian and Matérn-$\tfrac32$ cases (smallest eigenvalues $-0.39$ and $-0.13$); the exponential case happened to stay valid in this sample, so I would not claim it always fails. One sentence plus these numbers turns "mathematically preventing correlation distortion" (L138, which is vague) into a concrete reason.
- **Matérn without Bessel functions.** Give $\mathcal M_{1/2}(d)=e^{-d}$ and $\mathcal M_{3/2}(d)=(1+\sqrt3\,d)e^{-\sqrt3\,d}$ in the chapter's parametrization. Then the evaluator can use the Matérn the theorem is about, and the "dependency-free" code is no longer limited to the Gaussian limit (L320).
- **Variance prefactor naming.** The theorem uses $\sigma(\mathbf s)$ in the formula and "$\sigma^2(\mathbf s)$" in the text (L99). Pick one, and say that at $\mathbf s_i=\mathbf s_j$ the formula returns $\sigma^2(\mathbf s_i)$ (the prefactor is 1 and $Q=0$), so the marginal variance is exactly what you specify.
- **How would you *choose* $\boldsymbol\Sigma(\mathbf s)$?** The theorem is a recipe for building a valid covariance from a *given* field of ellipses. The statistical problem is estimating that field: parametrize the ellipse's two axis lengths and angle as smooth functions of coordinates or covariates (splines, or a low-dimensional parametric form), then maximize the likelihood or fit local variograms and smooth the results. The chapter says none of this, which leaves a student able to evaluate the covariance but not to fit it.
- **Beat 3:** two honest caveats (from memory): the model is highly flexible and its parameters are weakly identified from a single realization, and the damping prefactor can make correlation decay faster than a student expects when neighboring ellipses differ strongly. A sentence each, or a small figure.

### §13.4 Space Deformation (Sampson–Guttorp)

**What works.** The geographic-space/deformed-space picture (L147–164) is a good way to introduce the idea, and the four-step recipe (L173–185) is clear in outline. It correctly notes that the method needs replicates (L174).

**Issues**
- **The base covariance must be valid in the dimension of the deformed space.** L171 says $C_0$ is a "valid stationary, isotropic 1D covariance kernel." A function that is a valid covariance on the line need not be valid in the plane; what is needed is validity in $\mathbb R^k$ for the dimension $k$ of $\mathcal D$-space (typically $k=2$). The triangle (tent) function is the standard example (from memory). My random-point test did not expose it (a tent with range 1.5 gave positive eigenvalues in both 1D and 2D on 60 random points), so I cannot show a numerical counterexample; the claim rests on the standard result. For this audience, the safe wording is "an isotropic covariance valid in $\mathbb R^k$ (exponential, Matérn, and spherical in $k\le3$ all are)."
- **"Bijective" (L171) with $k\ge2$ (L149) is inconsistent.** A map from $\mathbb R^2$ to $\mathbb R^k$ with $k>2$ cannot be onto. Say $k=2$ in the main text and note that higher $k$ is an option.
- **The central failure mode is not mentioned: folding.** An interpolating thin-plate spline through noisy points can fold the plane over itself, so that two distant places map near each other, and the model is then invalid as a deformation. The remedy is a smoothing (penalty) parameter. This is the most important practical point of the method and it is absent.
- **The method requires choosing the variogram model before MDS** (L175–177: invert $\gamma_0$ to get target distances), and the resulting coordinates feed back into estimating $\gamma_0$; the real algorithm alternates. State the loop in one sentence.
- **Variance is held constant.** The deformation covariance (L169) has a constant sill, so §13.1's *variance* heterogeneity is not handled by this method; combine with $\sigma(\mathbf s)$ as in the simple bridge above.
- **"Strictly stationary" (L152)** should be "second-order stationary and isotropic."
- **No example and no figure that a student can follow.** The ASCII drawing at L154–164 is hard to read. A before/after sketch of a regular grid and its deformed image, with the points and the ellipses of equal correlation, would show the idea in one look.
- **Beat 3:** folding; the need for many replicates; prediction at new locations requires evaluating the spline *outside* the stations (extrapolation); and uncertainty in the deformation itself is ignored by the final kriging step (from memory, an acknowledged limitation).

### §13.5 Scalable Low-Rank Kriging and FRK

**What works.** The opening arithmetic ($n=10^5$: $10^{10}$ matrix entries, 80 GB; L196–200) is clear, the model (L206–213) is stated compactly, and the Woodbury derivation (L231–239) is correct. The complexity box (L241–259) is a good summary. The multiresolution bisquare basis (L261–270) is the right tool.

**Issues**
- **"Several weeks of CPU time" (L249) is wrong by about two orders of magnitude.** $\tfrac13n^3=3.3\times10^{14}$ operations takes about 9.3 hours at 10 GFLOP/s and 0.9 hour at 100 GFLOP/s. L198 states $n^3=10^{15}$ (without the $\tfrac13$) and L249 states $3.3\times10^{14}$: the same quantity quoted two ways, with no explanation. The true obstacle at $n=10^5$ is *memory* (the 80 GB matrix), which the chapter does say; lead with that and drop "weeks."
- **The kriging predictor itself is never written.** The chapter inverts $\boldsymbol\Sigma$ but does not give $\widehat Z(\mathbf s_0)=\mathbf x_0'\hat{\boldsymbol\beta}+\mathbf S(\mathbf s_0)'\mathbf K\mathbf S'\boldsymbol\Sigma^{-1}(\mathbf Z-\mathbf X\hat{\boldsymbol\beta})$ or its variance, nor show that, via Woodbury, this equals $\mathbf S(\mathbf s_0)'\hat{\boldsymbol\eta}$ with $\hat{\boldsymbol\eta}=(\mathbf K^{-1}+\sigma^{-2}\mathbf S'\mathbf S)^{-1}\sigma^{-2}\mathbf S'(\mathbf Z-\mathbf X\hat{\boldsymbol\beta})$. This is the formula the code uses (L421–423), and it is a ridge-regression-style estimate that students recognize from the linear-model course. Say that: *FRK is regression of the data on the basis functions with a ridge penalty $\mathbf K^{-1}$, and the penalty is what makes it a spatial model.* That is the cleanest bridge in the section and it is not drawn. I verified that this expression matches exact kriging to $4\times10^{-13}$.
- **The estimation of $\mathbf K$, $\sigma^2_\epsilon$, and the basis is missing.** The code hard-codes `K_diag = 2.0` with a comment that "in full FRK, $\mathbf K$ is estimated via method of moments or EM" (L407–409), and the text never says how. This is the *statistical* core of FRK: choosing the number, position, and scale of the basis functions; estimating $\mathbf K$ from an empirical covariance (Cressie and Johannesson's method of moments; from memory); estimating the nugget. Without it, the chapter teaches the linear algebra of FRK but not the method.
- **Overclaims (L220, L270).** "Automatically non-stationary … capture complex local spatial geometries" and "seamlessly captures both broad regional trends and sharp local spatial features" are stronger than the model supports. The covariance $\mathbf S(\mathbf s_i)'\mathbf K\mathbf S(\mathbf s_j)$ is nonstationary *as a by-product of where the basis functions are placed*, and with compactly supported bisquares the prior variance $\mathbf S(\mathbf s)'\mathbf K\mathbf S(\mathbf s)$ shrinks to zero away from every center, so predictions in gaps revert to the mean and the variance collapses. Say what a student should check: basis coverage and overlap, and whether the prior variance is roughly flat where you have data. Also note the standard FRK model includes a fine-scale term in addition to the nugget (from memory), which the chapter's model omits.
- **Wording (L262).** "A canonical authorized basis function" reads oddly; "a common choice" is enough.
- **No Beat 3.** What goes wrong: too few basis functions oversmooth; too many bring back the cost; edge effects; prediction variance too small if the fine-scale term is omitted; and the model is a *smooth* approximation, so sharp features at scales below the finest basis are missed.
- **No comparison with the other scalable routes.** One paragraph situating FRK (low rank) next to the sparse-precision route of Chapter 12 (CAR/GMRF; $\mathbf Q$ sparse) and, as pointers, covariance tapering and nearest-neighbor processes (from memory) would help a student choose.

### §13.6 Algorithmic Engines

**What works.** The two classes are clean and short. The Paciorek–Schervish evaluator reproduces the formula in the chapter; the FRK engine is a faithful implementation of the ridge-type estimator and agrees with exact kriging (see above). Runtime is trivial (0.02 s for $n=1{,}000$, $r=16$).

**Issues**
- **Experiment 1 does not show the claimed contrast (L478–484).** It compares covariance at lag 0.10 in the west (ranges from 0.28 to 0.36) and the east (0.84 to 0.92). Because 0.10 is short compared with both ranges, both correlations are near 1: I get covariances $0.938$ and $0.992$, a ratio of $1.06$, and the printed message says "Covariance is 1.06x stronger in East for identical lag!" That is not a dramatic difference. Use a lag comparable to the west range (about 0.3–0.4), where the west correlation has fallen substantially and the east has not. (The product of the prefactor and correlation also depends on the *difference* in metrics between the two points; with a gently varying tensor the prefactor is near 1, $0.984$ west and $0.998$ east here.)
- **Experiment 2's targets all have true value zero (L505).** The surface is $\sin(2\pi x)\cos(2\pi y)$, and at $(0.25,0.25)$, $(0.5,0.5)$, $(0.75,0.75)$ the true values are $0$, $0$, and $0$ to machine precision. The printed predictions ($+0.243$, $-0.046$, $-0.218$) therefore look like arbitrary small numbers and teach nothing, and a trivial "predict 0" would beat them. The model does work: on a $19\times19$ grid the root-mean-square error is $0.167$, against $0.499$ for predicting zero. Report that.
- **The printed "Kriging SE" (L434, L511) is the standard error for a *new noisy observation*,** because the code adds the nugget `sig2`. It prints about 0.201 for all three points, which is essentially the nugget's standard deviation ($\sqrt{0.04}=0.2$). The uncertainty in the latent surface itself is about 0.02 here (from $\sqrt{0.201^2-0.2^2}$, by subtraction). Label which is which, since students will read it as the kriging standard error of the signal.
- **The label "non-linear hill" (L492)** describes a $\sin\times\cos$ wave pattern with several peaks and valleys, not a hill. And the data are a deterministic surface plus white noise, so this is smooth surface fitting rather than a test of spatial correlation; fine for a demo, but say so.
- **Silent fallbacks.** `_inv_2x2` returns the identity matrix if the determinant is below $10^{-12}$ (L313–315), which would hide a degenerate metric (for example scale parameters around $10^{-3}$) behind plausible-looking numbers. Raise an error or warn. `det1**0.25` would fail for a negative determinant if a metric function returned a non-positive-definite matrix; add a check.
- **The docstring says "Matérn / Gaussian" (L292) but only the Gaussian limit is implemented (L320).** Add $\nu=\tfrac12,\tfrac32$ (closed forms above) and run the positive-definiteness check as the code's Beat 3.
- **No timing or comparison against exact kriging,** though the chapter's headline claim is speed. A two-line comparison on $n=2{,}000$ would substantiate it.

### Exercises

There is no in-chapter worked-exercise section; the proofs and matrix mechanics are in `ch13_exercises.md`. Candidates for in-chapter exercises: (1) show $\sigma(\mathbf s_i)\sigma(\mathbf s_j)\rho(\cdot)$ is valid; (2) the two-Gaussians proof for the stationary-prefactor case in 1D; (3) check the prefactor $\le1$ in 1D ($\tfrac{a+b}{2}\ge\sqrt{ab}$) and in a $2\times2$ diagonal case; (4) apply Woodbury to a $3\times3$ example with $r=1$; (5) the flop count for $n=10^5$, $r=100$ and the memory for $\mathbf S$ ($80$ MB).

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **4.5 / 10**

- **A strong first page, then an unfinished middle.** §13.1 has the book's voice. §13.2's one-line positivity argument and §13.5's cost comparison are also good. In between, §13.3–§13.4 are formulas and recipes with no worked example, no picture of a fitted nonstationary field, and no failure modes.
- **Two topics, one title.** Nonstationarity (§13.2–13.4) and scalability (§13.5) are different problems. The chapter says so only implicitly ("Whether stationary or non-stationary," L191). The unifying story is *a low-rank basis expansion with spatially varying basis functions is both*, which is available and not told.
- **No map or data.** A single figure of a nonstationary field (a stretch of coast with varying range and direction) with the fitted ellipses, plus a before/after kriging map, would carry the whole chapter.
- **No "From the Lecture Hall" boxes.** Candidates: weighted least squares and heteroskedasticity (§13.1); the integral of a square is non-negative / $\mathbf S\mathbf S'$ is always positive semi-definite (§13.2); completing the square for Gaussians (§13.3); ridge regression (§13.5); smoothing splines (§13.4).
- **Visible template remnants:** none; the chapter has no Beat headings. That makes it consistent with Chapter 12 and inconsistent with Chapters 9–11.

### Dimension 2 — Algebraic transparency and scaffolding: **6.5 / 10**

- **Strong:** the positive-definiteness argument, the Woodbury step, the stationary-limit check (L139), and the complexity box.
- **Needs work:** (a) the main theorem is cited, with a kernel that does not match (§3); (b) the matrix AM–GM statement; (c) the missing predictor formula for FRK; (d) the spline formula without conditions; (e) "bijective" and "1D kernel" in the deformation model.
- **Anatomy blocks:** one (the Paciorek–Schervish covariance), which is good. FRK's model and the Woodbury identity each deserve one, term by term.

### Dimension 3 — Structural consistency and forward referencing: **4 / 10**

- **Format drift:** no numbered theorems or proofs, no exercises, no Beat structure; headings use `##`.
- **Missing back-references:** Chapter 2 (geometric anisotropy, Matérn, the validity of covariance models in different dimensions), Chapter 5 (kriging with a given covariance, the $O(n^3)$ cost discussed again at L195), Chapter 6 (trend versus covariance non-stationarity; ridge/GLS), Chapter 10 §10.3 (the same Cholesky cost argument), Chapter 12 (sparse-precision alternative).
- **Missing forward/outward references:** a pointer to what comes after (spatio-temporal models, SPDE and Gaussian Markov random field approximations of Matérn fields; from memory) would close the book's arc.
- **Internal inconsistency:** kernel (L94) vs. theorem (L102) vs. code (L353); $n^3$ vs. $\tfrac13n^3$ (L198 vs. L249); "1D kernel" vs. "valid in $\mathbb R^k$" (L171); $\sigma(\mathbf s)$ vs. $\sigma^2(\mathbf s)$ (L99).
- **Notation:** $\boldsymbol\Sigma(\mathbf s)$ is a kernel covariance here while $\boldsymbol\Sigma$ without argument is the data covariance in §13.5 (L228). Rename one (for example $\mathbf A(\mathbf s)$ for the local ellipse, as is common).

### Dimension 4 — Empirical and computational utility: **5.5 / 10**

- **Verified correct:** the PSD argument; the Woodbury identity (error $1.6\times10^{-13}$); the equality of the FRK code with exact kriging for the same $\mathbf K$ (max difference $4\times10^{-13}$); positive definiteness of the Paciorek–Schervish matrices with the prefactor; the FRK complexity count $nr^2=10^9$; the fit quality (grid RMSE 0.167 against 0.499).
- **Verified wrong or inconsistent:** the kernel at L94 (factor of 2 in the exponent and variance $(4\pi)^{-d/2}$); "several weeks" (L249); the unit-of-measure of the printed SE (new observation, not latent signal); the uninformative demo targets (true value 0) and the 1.06× contrast.
- **Not demonstrated:** fitting $\boldsymbol\Sigma(\mathbf s)$ or $\mathbf K$; any real or simulated nonstationary data; a comparison with a stationary fit; the deformation method at all (no code).
- **Code optional, algorithms not.** The theorem needs its proof and the estimation story; FRK needs the estimation of $\mathbf K$ and basis choice; Sampson–Guttorp needs a worked small example and the folding caveat.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)

1. **Kernel/theorem mismatch (L94, L102).** Change the kernel to exponent $-(\mathbf u-\mathbf s)'\boldsymbol\Sigma^{-1}(\mathbf u-\mathbf s)$ with constant $(2/\pi)^{d/4}|\boldsymbol\Sigma|^{-1/4}$ (unit variance, and gives $\exp(-Q/2)$), or restate the theorem to match the current kernel. Add the short completing-the-square proof; say the Matérn extension is a separate (mixture) argument and cite it.
2. **"Several weeks" (L249) and $n^3$ vs. $\tfrac13n^3$ (L198).** Use $\tfrac13n^3\approx3.3\times10^{14}$ (hours, not weeks, at $10$–$100$ GFLOP/s) and lead with the 80 GB memory problem.
3. **White-noise integral (L57–76).** Replace with the finite-knots version ($\mathbf Z=\mathbf S\mathbf w$); keep the integral as the dense-knots limit. This removes the Dirac-delta vocabulary and prepares §13.5.
4. **FRK predictor (§13.5).** Write $\hat{\boldsymbol\eta}$ and $\widehat Z(\mathbf s_0)$, say what the variance includes, and say it is ridge regression with penalty $\mathbf K^{-1}$.
5. **Estimation of $\mathbf K$, the nugget, and the basis** (L407–409). Add a paragraph and an honest statement of what the demo's hard-coded `K_diag = 2.0` is.
6. **Experiment 2 (L505).** Use targets where the true value is not zero, or report the grid RMSE (0.167 vs. 0.499 for predicting zero); label the printed SE as a new-observation SE or print the latent SE.
7. **Experiment 1 (L478–484).** Use a lag comparable to the west range so the contrast is visible (the current 1.06× is not).
8. **Deformation model (L149, L171).** $C_0$ must be valid in $\mathbb R^k$, not "1D"; drop "bijective" for $k>2$.

### Should fix (pace, flow, voice)

9. **Unify §13.2 and §13.5:** a process convolution on a grid of knots is a low-rank basis expansion with $\mathbf K=\mathbf I$; add that paragraph and retitle the chapter as one story.
10. **Add the easy bridges to §13.1:** weighted least squares → $\sigma(\mathbf s_i)\sigma(\mathbf s_j)\rho$; geometric anisotropy → three levels of flexibility.
11. **Add diagnostics:** how to detect nonstationarity (moving-window variograms, residual variance maps) and how not to mistake a trend for it (Chapter 6).
12. **Show why the prefactor matters** (without it the Gaussian and Matérn-$\tfrac32$ versions are not valid; min eigenvalues $-0.39$ and $-0.13$ on 80 points) and give the 1D AM–GM check.
13. **Add Matérn closed forms** ($\nu=\tfrac12,\tfrac32$) so the evaluator is not limited to the Gaussian limit; make the positive-definiteness test the code's Beat 3.
14. **Add the folding caveat and the smoothing parameter** to Sampson–Guttorp; state the MDS/variogram alternation; add a before/after picture.
15. **Temper the FRK overclaims (L220, L270)** and add basis-coverage checks.
16. **Say how to choose $\boldsymbol\Sigma(\mathbf s)$** (parametrize axis lengths and angle by smooth functions; fit by likelihood or smoothed local variograms), with the identifiability caveat.
17. **Add a short comparison of scalable routes** (low rank here; sparse precision in Chapter 12; tapering and nearest-neighbor methods as pointers).
18. **Fix silent fallbacks in the code** (`_inv_2x2`, determinant checks); add a timing comparison with exact kriging.
19. **Bring the chapter to the common template:** numbered statements, worked exercises, Beat structure, an opening promise, one anatomy block per main equation.
20. **Unify notation:** $\boldsymbol\Sigma(\mathbf s)$ (kernel ellipse) vs. $\boldsymbol\Sigma$ (data covariance); $\sigma$ vs. $\sigma^2$.

### Nice to have

21. A Lecture Hall box each on heteroskedasticity/WLS, ridge regression, smoothing splines, and completing the square.
22. A figure: a nonstationary field with its local ellipses, and kriging maps with and without the nonstationary model.
23. One sentence on what comes next (SPDE and Gaussian Markov random field approximations; spatio-temporal models; from memory).
24. Replace "authorized" (L262) with "common."

---

## 6. What to Keep Exactly As It Is

- The framing of §13.1 and the five-way picture of ellipses (L21–37), plus the caution box (L39–44).
- The positive-definiteness argument (L78–84): *the integral of a square.*
- The anatomy box for the Paciorek–Schervish covariance (L114–132) and the stationary-limit check (L139).
- The geographic-space/deformed-space picture of Sampson–Guttorp (L147–164), once it is redrawn more legibly.
- The Woodbury derivation (L231–239) and the complexity box (L241–259), with the corrected flop and time figures.
- The bisquare basis definition (L261–270).
- The FRK engine's structure (L358–462), which I verified against exact kriging.
