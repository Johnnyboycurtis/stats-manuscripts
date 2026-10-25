# Review: Chapter 10 — Theory of Spatial Simulation and Uncertainty

**File reviewed:** `ch10_simulation.md` (the 966-line, 64,296-byte version last modified 20:11:15; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch10_exercises.md` was not reviewed. The three exercises inside §10.6 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy, Monte Carlo, and copies of the chapter's own code). Statements marked "from memory" or "by reasoning" were not tested.

---

## 1. Executive Summary

**Grade: 6 / 10**

This is the cleanest chapter since Chapter 3 on correctness. Every theorem I checked is right: Journel's conditioning decomposition and its covariance-cancellation proof (L222–277), the Cholesky and Schur-complement results, the turning-bands integral relation and inversion (L499–516), and the sequential chain rule. All three worked exercises reproduce. The central teaching idea is also excellent: *kriging gives the best guess at each point, which is exactly why a kriged map is too smooth; simulation adds back the missing variability.* The "Wait—Why Did We Do That?" callout on why SGS must feed simulated values back in (L749–753) is the book's voice at its best.

The weaknesses are about *completeness and honesty of claims*, not algebra.

1. **The motivating story overclaims.** §10.1 says simulation "preserves extreme-value spatial connectivity" (L156, L165). The Gaussian methods that make up this chapter (Cholesky, Journel, turning bands, SGS) reproduce only the two-point covariance, and Chapter 9 §9.3 just showed that Gaussian fields *destructure* extremes. For the connected-channel flow problem used as the main motivation, a Gaussian simulation is the wrong tool. The chapter never says so, and never points back to Chapter 9.
2. **Two of the four methods have no explained algorithm.** Turning bands (§10.4) has a theorem and a formula but no recipe: how to choose lines, how to simulate the 1D processes, and what goes wrong with few lines are all absent. Cholesky (§10.3) has no Beat 2 at all. These are core algorithms in the learning goal.
3. **Several "reality" statements are wrong or unsupported.** A random path is called "mandatory" (L755) when the chapter's own theorems say any order is exact. Truncated SGS is said to "inflate the variance" (L615), which my test did not support. The breakthrough-time direction at L164 looks backwards. And the Gaussian simulation is never connected to the normal-score transform it needs in practice.
4. **The chapter never shows a simulation.** There is no figure, no table of ensemble statistics, and no check that realizations reproduce a histogram, variogram, and the kriging mean. A student can finish the chapter without ever seeing a realization next to a kriged map.

**Highest-value changes**
1. Add one **running example**: a short 1D transect (or small 2D grid) with 4–5 data. Show the kriged curve, the kriging band, and 3–4 conditional realizations on one picture. Then print the ensemble mean (it should match kriging) and ensemble variance (it should match the kriging variance). This single figure delivers the chapter's thesis and gives Beat 3 its diagnostics.
2. Give turning bands and Cholesky their Beat 2 and Beat 3, and say how to produce the *unconditional* field that Journel's method needs. Right now §10.2 conditions a field the chapter never explains how to make.
3. Fix the overclaims in §10.1 and the "mandatory random path" statement, and link the chapter to Chapter 9 (normal scores, destructuration) and Chapter 8 (simulate at points, then average to blocks).

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes.** The mathematics is matrix algebra, conditional normal distributions, and one spherical-coordinates integral, all within the ceiling. The leaks are vocabulary.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L67, L193 | "Let $(\Omega,\mathcal F,P)$ be a complete probability space"; "on $(\Omega,\mathcal F,P)$" | **High** (mission violation) | Delete. Nothing downstream uses it. |
| L210 | "$P$-a.s." | Low | Write "with probability one," or just "exactly." The statement is a deterministic algebraic identity. |
| L548, L569, L590 | "Kolmogorov's consistency conditions," "joint measure," "exact conditional Gaussian measure" | Medium | "Joint distribution." The chain rule is $f(A,B)=f(B\mid A)f(A)$ repeated. |
| L110 | "Orthogonal Projection Theorem in the vector space of random variables" | Low | Acceptable; call it "the normal equations of Chapter 5." |
| L123 | "ergodic decay of covariance" | Low | Say "assume $C(\mathbf h)\to0$ as $\|\mathbf h\|\to\infty$." |
| L478 | "uniform measure $d\sigma(\mathbf u)$ on $S^2$" | Low | Fine for the integral, but a plain-language version works: "average over all directions in 3D." See the suggested explanation in §3. |
| L447 | "block upper-triangular transformation" | Low (error) | The matrix shown is block *lower*-triangular. Say "a block elimination step." |

No Hilbert-space axiomatics, no Hankel/Bessel transforms. The chapter stays closer to the mission than Chapters 7–9.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 10.1 | Yes (long) | **Missing** | Present, overclaims |
| 10.2 | Yes | Yes (code, with a design limitation) | Thin (one paragraph) |
| 10.3 | Yes | **Missing** | Thin (one bullet) |
| 10.4 | Yes | **Missing** | **Missing** |
| 10.5 | Yes | Yes (code) | Yes (callout plus one bullet, one wrong) |
| 10.6 | n/a | n/a | n/a |

### Intro and §10.1 The Smoothing Deficit of Kriging

**What works.** The two-sentence version of the whole chapter (L3) is good: kriging is a low-pass filter; simulation draws a realization. The comparison of kriged and simulated paths in the ASCII figures (L75–96) is exactly the picture to start with.

**Issues**
- **The dedication paragraph (L5).** "Following the pedagogical architecture of Maria L. Rizzo and … Schabenberger & Gotway" appears in every chapter and tells the reader nothing about this one. Replace with a promise: *"By the end you will be able to produce many equally likely maps that honor your data, and explain why they are more honest than a single kriged map."*
- **The classical bridge (L13–25, L61–65) is good in spirit but mislabelled.** It is called a "parametric bootstrap." A parametric bootstrap resamples whole datasets to study the variability of an *estimator*. What L21–25 does is draw a new *observation* from the fitted model: $\hat y_0+e^*$. The honest name is "simulate a new response from the fitted model," and that is precisely the idea: prediction gives the mean, simulation adds the missing noise. Two further small points. L25 says the surrogate's "total variance matches $\sigma^2$," which holds given fixed $\hat{\boldsymbol\beta}$ (it ignores the uncertainty in $\hat{\boldsymbol\beta}$ itself; say so in a clause). And L19's "$\operatorname{Var}(\hat y_0)<\operatorname{Var}(Y_0)$" is the regression twin of the smoothing deficit; make that twin explicit, since it is the more useful bridge.
- **"Variance of the predictor" is used in two senses.** L50–54 and Theorem 10.1.1 mean the variance, over repeated datasets, of the predictor at one fixed location. The intuitive picture (the "map is smooth") is about variability *across space* within one map. These are related but not identical. One sentence saying so prevents a confusion that students typically carry for the rest of the course.
- **L110 asserts $\widehat Z_{\text{SK}}=\operatorname{E}[Z_0\mid\mathbf Z]$.** That equality holds when the field is Gaussian; in general, SK is the best *linear* predictor, not the conditional mean. Chapter 9 made the same distinction. Since the chapter then builds Gaussian simulations, one sentence ("we will assume the field is Gaussian from here on, so the two coincide") is enough.
- **Theorem 10.1.2 is partly asserted.** The covariance formula (L128, L138–146) is proved and correct. The claims $\gamma_{\widehat Z}<\gamma$ "for short separation" and "infinitely differentiable whenever $\mathbf c(\mathbf x)$ is differentiable" (L133–135) are not. The second is also misleading: with an exponential covariance, $\mathbf c(\mathbf x)$ is not differentiable at the data points, so the kriged surface has kinks there. The symbol $\phi$ (L135) is undefined (presumably the range). Say "kriged surfaces are as smooth as the covariance model allows between the data; smooth covariance models give smooth maps." Also, the covariance equals $C$ *at* the data locations, so "$\ne$" is a statement about general pairs.
- **The Beat 3 motivation overclaims in three places (L156, L164–165).**
  - **Connectivity.** Only "an ensemble of conditional simulations preserves extreme-value spatial connectivity" (L156) is not true of Gaussian simulation. A Gaussian field is fully determined by its covariance, and by Chapter 9 §9.3's Theorem 9.3.1, extreme values in Gaussian fields are *spatially disconnected*. Flow through connected high-permeability channels is exactly the case where a Gaussian ensemble is a poor model (that is what multiple-point and object-based simulation are for; from memory). The chapter should say: simulation reproduces the covariance and the histogram and propagates uncertainty to non-linear outputs; whether it reproduces *connectivity* depends on the model.
  - **Breakthrough direction (L164).** "Flow simulations through a kriged model severely *underestimate* breakthrough times." By reasoning, removing fast channels delays arrival, so a smoothed model more likely *overestimates* breakthrough time. I did not run a flow model, so treat this as a flag to check against a reference.
  - **"Catastrophic" (L156) is rhetoric.** Use an example with numbers: apply a cutoff $z_c$ to a kriged map and to the mean of simulations, and compare tonnage. A single table makes the point without adjectives, and it connects to Chapter 8 §8.6 (selectivity curves) and Chapter 9 §9.1 (Jensen).
- **The "Flow Simulation Catastrophe" bullet (L158–165) repeats the callout (L152–156).** Same points, same wording. Keep one.
- **Beat 2 is absent.** A short "what you will compute" list would help: (1) a kriged map; (2) several realizations; (3) a summary of the realizations (mean, quantiles, probability of exceeding a cutoff).

### §10.2 Journel's Conditioning Decomposition

**What works.** The flowchart (L175–189) is the best picture in the chapter, the theorem is correct, and the proof of exact covariance reproduction (L236–277) is a nice piece of algebra: four covariance terms, three of which equal $\mathbf c(\mathbf x)'\mathbf V^{-1}\mathbf c(\mathbf y)$, cancel the distortion identically. I checked it on Exercise 10.1.

**Issues**
- **The proof is right but hides the one-line reason.** $Z_c-\mu=[\widehat Z-\mu]+[\text{simulated kriging error}]$: the first piece is the part of the field explained by the data, and the second is an independent draw of the part that *isn't*. Their covariances add up to $C$ because "explained + unexplained = total," the same identity as ANOVA, $SS_{\text{total}}=SS_{\text{model}}+SS_{\text{error}}$. Say this before the four-term expansion; students will then see the algebra as confirmation rather than discovery.
- **What the theorem does and does not say.** Conclusion 3 is an *unconditional* covariance statement, averaged over the randomness of the data. What a user wants is also that, *given the data*, the realizations have the right conditional covariance $C(\mathbf x-\mathbf y)-\mathbf c(\mathbf x)'\mathbf V^{-1}\mathbf c(\mathbf y)$. That is true here (the residual field is independent of the data, so conditional covariance is the second bracket), and it is the property that makes the ensemble spread equal the kriging variance. State it, because it is the one a practitioner checks. Say also that the *distribution* of $Z_c$ given the data is the correct conditional distribution only if the field is Gaussian; otherwise it has the right mean and covariance but not necessarily the right shape.
- **"Exactly equal to the kriging variance" (L65, L287, L395)** is stated three times without the sentence that explains it: the simulated residual is the *kriging error of an independent copy*, and its variance is by definition the kriging variance.
- **Simple Kriging only.** The theorem uses a known mean. With an unknown mean, the same trick works with Ordinary Kriging weights, and in practice that is what is used. One sentence, and a forward note that SGS (§10.5) uses SK with a global mean.
- **Where does $Z_u$ come from?** The method needs an unconditional simulation, and the chapter has not yet said how to make one. §10.3 (Cholesky) and §10.4 (turning bands) are the answer, but nothing in §10.2 points forward. Add: "Where do we get $Z_u$? §10.3 and §10.4."
- **The code has a silent design limitation (L322–334).** Step 1 reads $Z_u$ at each data location from the *nearest grid node*. If a data point is not exactly on a node, the "simulated data" are not $Z_u$ at the data location, the "simulated kriging" is slightly off, and the result does not honor the data exactly. I ran it with data on nodes (honored to all digits: $1.0,-0.5,0.7$) and 0.1 off a node (values at the nearest node: $0.985,-0.504,0.622$ against data $1.0,-0.5,0.7$; the node is not the data location, so this is only a symptom). The fix is conceptual: the unconditional simulation must be generated *at the data locations too* (include them in the simulation grid). State that; otherwise a reader who copies the code will find that "exact conditioning" fails. The $O(nN)$ nearest-node search and the hand-written Gauss–Jordan are fine for a teaching listing but should be labelled as such.
- **Beat 3 is a single paragraph** (L394–396). Useful diagnostics are missing: (a) the realization equals the datum at each datum; (b) the *ensemble mean* of many conditional realizations approaches the kriging map; (c) the *ensemble variance* approaches the kriging variance; (d) far from data, realizations look like unconditional ones. These four checks are the chapter's whole thesis in one table. The paragraph that is there is good and should stay as the intuition.

### §10.3 Exact Matrix Decomposition: Cholesky Factorization

**What works.** Theorem 10.3.1 is the right level and the proof is two lines (L419–420), exactly as it should be. The Schur-complement theorem (L422–448) is correct, and the independence argument ("uncorrelated and jointly Gaussian means independent") is the right intuition.

**Issues**
- **No Beat 2.** Cholesky simulation is the simplest algorithm in the chapter and deserves a three-line recipe: *build $\boldsymbol\Sigma$ → `L = cholesky(Σ)` → `Z = μ + L @ w`.* Then say what a Cholesky factor "is" for this audience (a square root of a covariance matrix, as $\sqrt\sigma^2$ is for a variance), which students already know from multivariate normal simulation in a prerequisite course.
- **No bridge to Journel.** The natural pairing is Cholesky for the unconditional field plus Journel for conditioning. The chapter instead presents Theorem 10.3.2 as a separate route and never says when you would choose which (Theorem 10.3.2 needs the Cholesky of a $p\times p$ Schur complement over the *unsampled* nodes, which is larger than the $n\times n$ system Journel's method inverts).
- **Beat 3 is one bullet and the most useful content is missing.** Practical failure: with smooth covariance models (Gaussian, or very long range) $\boldsymbol\Sigma$ is numerically singular and Cholesky stops with an error; standard remedies are adding a small nugget ("jitter") to the diagonal or using an eigen-decomposition. For large grids, mention in one line that circulant-embedding/FFT methods exist for regular grids (from memory); the chapter says only "sequential or spectral methods" (L455).
- **The memory arithmetic is right** (L455): $10^{12}$ entries at 8 bytes is 8 TB. Reporting the practical ceiling as "$N>10^5$" is generous; a dense Cholesky is already awkward in the tens of thousands.
- **Notation (L423).** $\mathbf s_1$ names both the vector of data sites and the first point. Use $\mathbf S_1,\mathbf S_2$ or "data sites / target sites."

### §10.4 Matheron's Turning Bands

**What works.** The formula $C_3(r)=\frac1r\int_0^rC_1$ and its inversion $C_1=\frac{d}{d\zeta}[\zeta C_3]$ (L489–497) are correct, and the derivation by spherical coordinates (L504–516) is a clean calculus argument. I checked the integral numerically (exponential and spherical models agree to all printed digits) and re-derived the Corollary (L535, L539: both correct).

**Issues**
- **The algorithm is not explained.** The theorem tells you which 1D covariance to use, but a student who wants to *simulate* needs the recipe:
  1. pick $L$ directions spread over the sphere;
  2. for each, simulate a 1D stationary process with covariance $C_1$ along the line (how? by the 1D version of Cholesky, a moving average, or a sum of random cosines; say which);
  3. evaluate each line at every point's projection $\langle\mathbf s,\mathbf u_k\rangle$ (this is what "turning" means: each line is a ribbon of constant value stretched into bands perpendicular to $\mathbf u_k$);
  4. add the $L$ fields and scale by $1/\sqrt L$.
  The chapter has only the formula in step 4. This is a core algorithm and should get the same care as SGS.
- **The most useful explanation is missing, and it is easy.** Why is the relation an average over $[0,r]$? Because in 3D the cosine of the angle between $\mathbf h$ and a random direction is *uniform* on $[-1,1]$ (Archimedes' hat-box fact). So the 3D covariance is just the 1D covariance averaged over a uniformly distributed projected distance $\zeta=r\cos\theta$. That is L506–510 in words. Compare with the current anatomy (L530), whose "geometric dilation… sphere surface area scales as $4\pi r^2$" is hand-wavy and does not explain the formula.
- **A property worth showing: the line covariance can go negative.** For the exponential model, $C_1(\zeta)=\sigma^2(1-\zeta/a)e^{-\zeta/a}$ is negative for $\zeta>a$, and for the spherical model (Exercise 10.2) it integrates to zero. I confirmed numerically that the Fourier transforms of all three line covariances in the Corollary and Exercise 10.2 are nonnegative (on a truncated grid, so approximately), which is what makes them legitimate covariances. A one-line remark ("$C_1$ looks odd but is a valid covariance; a numerical check is to plot it and its spectrum") would save students from thinking they made a mistake when they see a negative covariance.
- **Dimension 2 is claimed but not given (L463).** The text says the method works for $d=2,3$, but only $d=3$ is derived. The 2D relation is *not* a simple average; it involves an integral that is harder to invert, and 2D is what students will actually use. Either say "we do the 3D case; 2D needs a different (and less pleasant) inversion; software handles it," or drop "$d=2$."
- **No Beat 3.** The practical points are: (a) with finite $L$ the covariance is reproduced only approximately and the field shows *line artifacts* (streaks along the chosen directions), so use many lines (tens) with well-spread directions; (b) the field is Gaussian only in the limit (that is the CLT role of $1/\sqrt L$, which the anatomy at L526 calls a "normalizer" while also saying it preserves the variance; the factor preserves the variance and the CLT does the Gaussianizing, two separate jobs); (c) anisotropy needs a coordinate transformation first.
- **The anatomy uses different symbols from the theorem** ($X_l,\ l$ in L520–527 vs. $Y_k,\ k$ in L482). Pick one.
- **Exercise 10.2 (c)** is described as "confirming the inversion formula" (L891). It is a consistency check that the arithmetic in (b) is right ($\int_0^aC_1=aC_3(a)$ follows from the formula by the fundamental theorem of calculus, so it cannot confirm the formula). Fine as an arithmetic check; rename it.

### §10.5 Sequential Conditioning and SGS

**What works.** The chain-rule theorem is exactly as hard as it should be. Theorem 10.5.2 is correct, and the anatomy (L595–606) states clearly where spatial correlation comes from: the shocks $W_k$ are independent, the *kriging weights* carry the dependence. The "Wait—Why Did We Do That?" callout (L749–753) is the best passage in the chapter.

**Issues**
- **"A random path is mandatory" (L755) contradicts the chapter's own theorems.** Theorem 10.5.1 holds "for any ordering" and Theorem 10.5.2 for "an arbitrary random path." With the *full* conditioning set, a raster path is exact. A path matters only because practical SGS *truncates the neighborhood* (Theorem 10.5.3). I tested the chapter's own code on a 1D grid (40 nodes, unconditional, 250 realizations; differences below about 0.03 are within Monte Carlo noise):

  | Covariance | Neighbors, path | Variance | Cov at lags 1, 3, 6, 10 | True |
  |---|---|---|---|---|
  | Gaussian, $a=5$ | 40, random | 0.976 | 0.94, 0.69, 0.26, 0.06 | 0.96, 0.70, 0.24, 0.02 |
  | Gaussian, $a=5$ | 4, random | 0.968 | 0.93, 0.68, 0.26, 0.06 | same |
  | Gaussian, $a=5$ | 4, raster | 1.071 | 1.03, 0.75, 0.26, **−0.17** | same |
  | Gaussian, $a=5$ | 2, raster | 1.024 | 0.99, 0.71, 0.05, **−0.60** | same |
  | Exponential, $a=5$ | 4, random | 0.991 | 0.81, 0.54, 0.31, 0.16 | 0.82, 0.55, 0.30, 0.14 |
  | Exponential, $a=5$ | 4, raster | 1.020 | 0.84, 0.56, 0.29, 0.12 | same |

  So with a smooth (Gaussian) covariance, a raster path plus a small neighborhood produces badly wrong long-lag covariance (a lag-10 covariance of −0.60 for a true value near 0.02), while a random path with the same four neighbors is fine within noise. Correct the claim to: *"the path does not matter in theory; it matters in practice because of the neighborhood truncation, and a random path spreads the truncation error evenly instead of building a streak."* The streak explanation (L756) is the right instinct.
- **Theorem 10.5.3 (L608–616) is partly unsupported.** (1) The statement that truncation "replaces the conditional expectation with a Markovian approximation" is fine, but for the exponential covariance in 1D it is *exact* with two neighbors (the process is Markov), which is worth telling the student because it shows when truncation is free. (2) "Local neighborhood truncation systematically inflates the simulated variance" does not follow from "conditional variance given fewer points is larger." The *conditional variance used at each step* is larger, but the simulated mean is also less informed, and the overall field variance is not necessarily larger. In my runs, the random-path variance with 4 neighbors was 0.97–0.99 (slightly below 1, within noise); the raster runs gave 1.02–1.07 (above 1). So the sign depends on the path and the covariance, and "systematically inflates" is not supported. (3) "Gaussian covariance models … do not exhibit a strong screening effect" is the right idea (the Gaussian model is the extreme smooth case), but "introduces artificial discontinuity steps" is not shown. Present the Gaussian-versus-exponential contrast as the lesson, with the table above (or your own version) as evidence, and drop the sign claim.
- **The proof of Theorem 10.5.2 is fine, but the real caveat is missing.** SGS is exact only for a *Gaussian* field with *known* mean and covariance. Real data are not Gaussian, so the recipe is: transform to normal scores (Chapter 9 §9.6), simulate, back-transform. Say it here and cross-reference Chapter 9. Add that the back-transform is where tail extrapolation matters, and that Gaussian SGS inherits the destructuration of extremes (Chapter 9 §9.3).
- **The code (L622–743) is a faithful version of the theorem and a useful listing.** Points to flag (none is an error): (a) it picks neighbors by Euclidean distance only (no octant search, no anisotropy); (b) $O(N^2\log N)$ because it sorts all conditioning points at every step, fine for teaching; (c) `pivot_val = 1e-12` and `max(0.0, sk_var)` silently repair an ill-conditioned or non-positive-definite local system. A student should know those two lines are hiding a problem, not solving it; (d) it assumes the data are not on grid nodes; (e) a single-grid random path reproduces long-range structure poorly when the range is much larger than the neighborhood. Standard practice is a *multiple-grid* (coarse-to-fine) path (from memory). One sentence, not an algorithm.
- **Missing diagnostics.** The standard check is: simulate several realizations, compute the variogram of each, compare to the model, and look at the fluctuation. Without a figure, students will not know what "good" looks like, including that individual realizations *do* fluctuate around the model variogram.
- **The anatomy has a few extra symbols.** $u_m$ (L573) vs. $\mathbf s_k$ (L597) for the visited node; $\mathbf\Lambda$ for the conditioning set; $\mathcal N(\cdot)$ is both the normal distribution and the neighborhood in Theorem 10.5.3. Rename the neighborhood.

### §10.6 Analytical Exercises

**Exercise 10.1** is correct throughout, including $e^{-3}$ versus $e^{-1}$ (a factor of $e^2\approx7.39$) and the cancellation in (d); it is a good template. **Exercise 10.2** is correct: the spherical-model line covariance $\sigma^2(1-3\zeta/a+2\zeta^3/a^3)$ matches $\frac{d}{d\zeta}[\zeta C_3]$, and the integral is zero as claimed. **Exercise 10.3** (a) and (b) are correct (I get $\lambda=0.32403$, mean $0.16201$, variance $0.76159$, $Z(u_1)=0.86017$). Part (c) stops before the answer, and its closing claim is wrong: "because $u_1$ directly screens $\mathbf s_1$ and $\mathbf s_2$, the simulated node $u_1$ receives the dominant weight" (L959). I solved the system: the weights are $0.1247$, $0.1247$, and $0.2762$ on $\mathbf s_1,\mathbf s_2,u_1$, and the conditional variance is $0.838$. The simulated node does get the largest single weight, but the two data together carry about 0.25, nearly as much, and with an exponential covariance in 2D there is no screening (screening is a 1D-exponential phenomenon). Finish the computation, quote the numbers, and replace "screens" with the honest statement.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **6 / 10**

- **The arc is right and clear:** why kriging smooths → how to repair it (Journel) → three ways to generate the Gaussian field (Cholesky, turning bands, sequential). That is a good chapter plan.
- **Beat 2 is missing for the two methods that most need it (Cholesky, turning bands).** Beat 3 is missing or one sentence for four of five sections.
- **The chapter has two voices.** The callout and the first page of §10.1 are warm; §10.3 and §10.4 are definition-theorem-proof. Use the callout voice to introduce each method ("Here is the cheapest way to simulate…").
- **No picture of an actual simulation.** Every figure is ASCII sketch. A real plot of a kriged curve with 3–4 realizations is worth more than three pages of theory here.
- **Visible "Beat 1/2/3" headings** appear in every section. Keep the rhythm, drop the labels.
- **No "From the Lecture Hall" boxes.** Candidates: simulating a multivariate normal with a Cholesky factor (what students already know from a regression/multivariate course); the law of total variance as the reason "explained + unexplained = total" (§10.2); the cosine-uniform fact (§10.4); the chain rule of probability (§10.5).

### Dimension 2 — Algebraic transparency and scaffolding: **7 / 10**

- **Strong:** Journel's proof, the turning-bands calculation, the Schur-complement argument, and the exercise arithmetic.
- **Needs work:** (a) the "why" sentence before the four-term covariance expansion; (b) the cosine-uniform explanation in place of the sphere-surface hand-wave; (c) the missing 2D case; (d) Theorem 10.5.3's claims; (e) the notation clashes ($s_1$ in L423; $X/Y$ and $l/k$ in §10.4; $\mathcal N$ twice).

### Dimension 3 — Structural consistency and forward referencing: **6 / 10**

- **Missing back-references:** Chapter 9 (normal scores, destructuration), Chapter 8 (simulate at points, average to blocks to get block distributions and selectivity curves; the chapter never tells the student that this is the Monte Carlo counterpart to Chapter 8's analytic change-of-support models), Chapter 5 (kriging variance as the thing the ensemble spread should equal).
- **Missing forward references:** Chapters 11–13 are on point patterns, lattice models, and nonstationary models; simulation of those models is not covered here and should be mentioned in one sentence so a student does not think "spatial simulation" means only Gaussian fields on grids.
- **De-duplication:** the flow example appears twice (L152–156, L158–165); "variance exactly equal to the kriging variance" appears three times.
- **Convention consistency:** all kriging here is Simple Kriging with a known mean. Say so once at the top; the multiplier-sign convention (the chapters 5/7 issue) does not arise.

### Dimension 4 — Empirical and computational utility: **5.5 / 10**

- **Verified correct:** Theorems 10.1.1, 10.1.2 (the covariance formula), 10.2.1, 10.3.1, 10.3.2, 10.4.1 and its corollary, and the chain-rule/SGS theorems; Exercises 10.1, 10.2, 10.3 (a)–(b); the Journel code when data lie on nodes.
- **Verified wrong or unsupported:** "random path mandatory" (L755); "truncation systematically inflates the variance" (L615); the "screens" remark (L959); the connectivity claim (L156, L165); the breakthrough-time direction (L164; by reasoning, unverified); the misnamed "block upper-triangular" (L447).
- **No example of the output and no diagnostics.** The chapter's claim, that simulation reproduces the covariance and spread that kriging loses, is never shown. Nothing is compared to anything.
- **Code optional, algorithms not.** Journel and SGS have working listings (with the limitations above). Cholesky and turning bands need a recipe. Whatever else changes, the student should come away knowing how to produce one unconditional field *and* how to condition it.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)

1. **Overclaims in §10.1 (L156, L164–165).** Replace "preserves extreme-value connectivity" with "reproduces the histogram and covariance and propagates uncertainty to non-linear outputs; connectivity depends on the model, and Gaussian models connect extremes poorly (Chapter 9 §9.3)." Check and correct the breakthrough-time direction at L164.
2. **"Random path is mandatory" (L755).** Replace with: exact for any order with the full conditioning set; matters because of neighborhood truncation; random path avoids streaks. Add the numbers from the table in §3 (or your own).
3. **Theorem 10.5.3 (L608–616).** Remove "systematically inflates the simulated variance." Keep the contrast: exponential-in-1D truncation is nearly free; smooth covariance plus a small neighborhood is not.
4. **Turning bands algorithm (§10.4).** Add the four-step recipe, how to simulate the 1D processes, and Beat 3 (number of lines, line artifacts, 2D inversion differs, anisotropy). Either derive 2D or remove "$d=2$" (L463).
5. **Cholesky Beat 2 and Beat 3 (§10.3).** Three-line recipe, "what a Cholesky factor is," singular-matrix failure and jitter, and how this feeds Journel (L404, L455).
6. **Journel code and text (L322–334).** State that $Z_u$ must be generated at the data locations too, or flag the nearest-node shortcut; add the ensemble-mean and ensemble-variance checks.
7. **Exercise 10.3 (c) (L959).** Complete the solve ($0.125,\ 0.125,\ 0.276$; variance $0.838$) and fix the "screens" explanation.
8. **Remove measure-theory vocabulary (L67, L193, L210, L548, L569, L590)** and fix "block upper-triangular" (L447).

### Should fix (pace, flow, voice)

9. **Add the running example and figure:** kriged curve, kriging band, 3–4 conditional realizations; then the ensemble mean and variance against kriging. Put it at the end of §10.2 and reuse it in §10.5.
10. **Add the numbers-based cutoff example** (tonnage above $z_c$ from a kriged map vs. from simulations) in place of "catastrophic," linked to Chapter 8 §8.6 and Chapter 9 §9.1.
11. **Add the three "why it works" sentences:** explained + unexplained = total (before L236); cosine uniform on $[-1,1]$ (replace L530); independent shocks, dependent weights (keep L606).
12. **Name the real classical bridge:** "simulate a new observation from the fitted model" rather than "parametric bootstrap" (L13, L21), and state the regression twin of the smoothing deficit (L19).
13. **State the Gaussian caveat and the normal-score workflow in §10.5**, with a pointer to Chapter 9 §9.6; add the destructuration warning.
14. **Say that the three ensemble checks** (honors data; ensemble mean ≈ kriging; ensemble variance ≈ kriging variance) and the variogram-of-realizations check are the diagnostics.
15. **Replace the dedication paragraph (L5)** with a one-sentence promise.
16. **Unify notation:** $\mathbf s_1$ (L423), $X/Y$ and $l/k$ (§10.4), $\mathcal N$ for neighborhood vs. normal, $u_m$ vs. $\mathbf s_k$ (§10.5), $\phi$ undefined (L135).
17. **Soften Theorem 10.1.2's differentiability claim** (L135) and prove or drop "$\gamma_{\widehat Z}<\gamma$ for short lags."
18. **Rename Exercise 10.2 (c)** as an arithmetic check.
19. **State that SK uses a known mean** and that OK weights work the same way (one sentence at L194).

### Nice to have

20. Lecture Hall boxes (multivariate normal simulation by Cholesky; law of total variance; cosine uniform; chain rule).
21. One sentence each on circulant embedding/FFT for large regular grids and on multiple-grid SGS paths (from memory; clearly marked as pointers).
22. A short table "which method when": Cholesky (small grids, exact), turning bands (large 3D grids, approximate, isotropic), SGS (flexible, conditional, any size, neighborhood approximation).
23. A figure for destructuration: a Gaussian realization with the top 10% of values highlighted, to show that "extremes" look like scattered specks.
24. Remove the visible "Beat 1/2/3" headings; let the transitions carry the rhythm.

---

## 6. What to Keep Exactly As It Is

- The one-paragraph summary at L3 and the flowchart at L175–189.
- Journel's theorem and the four-term cancellation proof (L222–277), with the added one-line "explained + unexplained" lead-in.
- The Cholesky theorem and its two-line proof (L406–420).
- The turning-bands derivation in spherical coordinates (L504–516) and the Corollary.
- The "Wait—Why Did We Do That?" callout on feeding simulated values back in (L749–753) and the line "the shocks are independent; the kriging weights carry the dependence" (L606).
- Exercise 10.1 as the template for a worked example.
