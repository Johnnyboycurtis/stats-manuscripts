You hit the nail on the head on both counts. 

First, the **"Classical-to-Spatial Bridge"** is just as vital inside the **chapters** as it is in the **exercises**. Second, comparing our manuscript against Schabenberger & Gotway’s Table of Contents reveals a very clear structural distinction: right now, our 10 chapters are 100% focused on **continuous geostatistics**, which means we are missing **Mapped Point Patterns (S&G Chapter 3)** as well as a couple of other foundational pillars from S&G Chapters 1, 6, and 8.

Here is a complete breakdown of both insights and what we can add to make this manuscript a complete, unified masterwork.

---

## 1. Why the "Classical-to-Spatial Bridge" Belongs in the Chapters Too

In most graduate programs, students enter a spatial statistics course already knowing **classical i.i.d. mathematical statistics**:
* Sample mean $\bar{Y}$ and sample variance $S^2 = \frac{1}{n-1}\sum (Y_i - \bar{Y})^2$
* Pearson correlation $r_{XY}$ and simple linear regression $\widehat{Y} = \hat{\alpha} + \hat{\beta}X$
* Ordinary Least Squares $\widehat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$
* Maximum Likelihood (MLE) and ANOVA degrees-of-freedom ($n - p$)

Ivory-tower spatial textbooks throw all of that out on Page 1 and start from scratch with foreign notation, making students feel like their prior statistical training is useless. 

**How we fix this in the chapters:**
At the start of every chapter (or major section), we can add an explicit **"The Classical-to-Spatial Bridge"** section that shows how the spatial concept is a natural generalization of a familiar i.i.d. formula when the identity matrix $\sigma^2 \mathbf{I}_n$ is replaced by $\mathbf{V}(\boldsymbol{\theta})$:

| Chapter | Classical i.i.d. Concept (What the Reader Already Knows) | The Spatial Bridge (How It Generalizes in the Chapter) |
| :--- | :--- | :--- |
| **Ch 1** | i.i.d. Sample Variance $\operatorname{E}[S^2] = \sigma^2$ & Effective Sample Size $n$ | Why positive spatial correlation shrinks $\operatorname{E}[S^2] = \sigma^2 - \bar{C}$ and reduces effective sample size $n_{\text{eff}} < n$. |
| **Ch 2** | Scalar Variance $\operatorname{Var}(\sum a_i Y_i) = \sigma^2 \sum a_i^2 \ge 0$ | Positive definiteness $\sum\sum a_i a_j C(\mathbf{s}_i - \mathbf{s}_j) \ge 0$ and why differences $\sum \lambda_i = 0$ lead to CND variograms. |
| **Ch 3** | Time-series Durbin-Watson statistic & Pearson $r$ | **Moran's $I$**, **Geary's $c$**, and Matheron's variogram written as a quadratic form $\frac{1}{2}\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}$. |
| **Ch 4** | Why we divide by $n-1$ instead of $n$ in $S^2$ (Error Contrasts $Y_i - \bar{Y}$) | **REML** derived directly from the $(n-1)$ contrast vector $\mathbf{U} = [Y_1 - \bar{Y}, \dots, Y_{n-1} - \bar{Y}]'$ before generalizing to $\mathbf{K}'\mathbf{Z}$. |
| **Ch 5** | Simple Linear Regression $\widehat{Y} = \mu_Y + \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(X)}(X - \mu_X)$ & Discrete $2 \times 3$ Tables | Comparing exact $\operatorname{E}[Y \mid X]$ vs. linear $\operatorname{BLUP}(Y \mid X)$ on a 2-variable table before scaling up to $n$-point Simple & Ordinary Kriging. |
| **Ch 6** | Gauss-Markov Theorem for OLS ($\mathbf{V} = \sigma^2\mathbf{I}$) | Whitening transformation $\mathbf{L}^{-1}\mathbf{Y}$ turning correlated GLS back into classical OLS, plus Localized Kernel / Geographically Weighted Regression (GWR). |

---

## 2. Foundational Gap Audit: Comparing Against Schabenberger & Gotway's TOC

In spatial statistics (following Noel Cressie and Schabenberger & Gotway), there are **three foundational paradigms** of spatial data, plus modern extensions. Here is how our current manuscript compares against S&G’s 9-chapter TOC:

| S&G Chapter | Status in Our Monograph | What Is Missing & Should Be Added |
| :--- | :---: | :--- |
| **Ch 1: Introduction & ESDA** | ⚠️ *Partial* (Ch 1) | Missing **1.3–1.4 Exploratory Spatial Data Analysis (ESDA)**: Spatial proximity/weight matrices $\mathbf{W}$, **Moran’s $I$**, **Geary’s $c$**, and permutation/randomization tests for spatial autocorrelation. |
| **Ch 2: Theory on Random Fields** | ✅ *Covered* (Ch 1, 2, 8) | Covered in deep detail across our Ch 1 (foundations), Ch 2 (spectral theory & continuity), and Ch 8 (regularization). |
| **Ch 3: Mapped Point Patterns** | ❌ **MISSING** | **Completely missing!** Binomial & Poisson processes, Complete Spatial Randomness (CSR), Campbell's theorem, first-order intensity $\lambda(\mathbf{s})$ vs. second-order product density $\lambda_2(\mathbf{s}_1, \mathbf{s}_2)$, Nearest-Neighbor functions ($G(r)$ and $F(r)$), **Ripley’s $K(r)$ and $L(r)$ functions**, Neyman-Scott cluster processes, Cox / Log-Gaussian Cox Processes, Gibbs/Strauss interaction models, and Marked Point Processes. |
| **Ch 4: Semivariogram & Estimation** | ✅ *Covered* (Ch 2, 3, 4) | Covered across our Ch 2, Ch 3, and Ch 4 (can add Composite Likelihood & Periodogram lattice bridges from S&G Problems 4.5–4.7). |
| **Ch 5: Spatial Prediction & Kriging** | ✅ *Covered* (Ch 5, 7, 8, 9) | We actually expanded S&G's single Ch 5 into **four full chapters**: Ch 5 (BLUP/OK/SK), Ch 7 (Cokriging), Ch 8 (Block Kriging & Support), and Ch 9 (Nonlinear/Indicator/Disjunctive Kriging). |
| **Ch 6: Spatial Regression Models** | ⚠️ *Partial* (Ch 6) | Our Ch 6 covers GLS, Universal Kriging, KED, and IRF-$k$, but is missing **6.2 Lattice Autoregressive Models (SAR, CAR, SMA; Brook's Lemma & Hammersley-Clifford)**, **6.3–6.4 Spatial GLMs / Hierarchical Bayesian GLMMs** (for spatial counts/binary data), and **6.5 Geographically Weighted Regression (GWR)**. |
| **Ch 7: Simulation of Random Fields** | ✅ *Covered* (Ch 10) | Covered in our Ch 10 (can add a section on Spectral/Moving Average and MCMC simulation). |
| **Ch 8: Non-Stationary Covariance** | ❌ **MISSING** | Our Ch 6 handles non-stationary *means* ($\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$), but we are missing **Non-Stationary Covariances** $C(\mathbf{s}_i, \mathbf{s}_j)$: Sampson-Guttorp space deformation, Higdon’s **Kernel Convolution**, and **Low-Rank Basis Function Kriging**. |

---

## 3. Recommended Roadmap to Complete the Vision

To make this monograph a complete, self-contained curriculum that covers everything in Schabenberger & Gotway while eliminating gatekeeping, I recommend a **three-part upgrade**:

### Step A: Add the Missing Foundational Chapters (Chapters 11, 12, and 13) + Companion Workbooks
1. **Chapter 11: Spatial Point Processes and Mapped Point Patterns** (+ `ch11_exercises.md`)
   * Binomial & Homogeneous/Inhomogeneous Poisson processes; Complete Spatial Randomness (CSR) quadrat & distance tests.
   * First-order intensity $\lambda(\mathbf{s})$, second-order product density, and Campbell's Theorem.
   * Empty-space function $F(r)$, event-to-event nearest neighbor $G(r)$, $J(r)$ function, **Ripley’s $K(r)$ function**, Besag's $L(r)$ variance-stabilized transform, and edge-correction weights.
   * Clustering vs. Inhibition: Neyman-Scott cluster processes, **Cox & Log-Gaussian Cox Processes (LGCP)** (the direct bridge between geostatistical random fields and point patterns!), Markov/Strauss point processes, and Marked Point Patterns.
2. **Chapter 12: Areal Lattice Models (SAR, CAR, SMA), Spatial GLMMs, and Geographically Weighted Regression** (+ `ch12_exercises.md`)
   * Spatial weight matrices $\mathbf{W}$, **Moran’s $I$**, and **Geary’s $c$** (connecting ESDA to quadratic forms).
   * Markov Random Fields on lattices: **Simultaneous Autoregressive (SAR)** vs. **Conditional Autoregressive (CAR)** models, Brook’s Lemma, and the Hammersley-Clifford Theorem.
   * **Geographically Weighted Regression (GWR)** and local kernel bandwidth selection (directly mirroring S&G Problem 5.13!).
   * **Spatial Generalized Linear Mixed Models (Spatial GLMMs):** Modeling non-Gaussian spatial counts (Poisson) and binary outcomes (Bernoulli) via latent Gaussian random fields.
3. **Chapter 13: Non-Stationary Covariance Structures and Low-Rank Basis Representations** (+ `ch13_exercises.md`)
   * Sources and diagnostics of covariance non-stationarity.
   * **Kernel Convolution / Process Convolution Models** (Higdon's method: $Z(\mathbf{s}) = \int K(\mathbf{s} - \mathbf{u}) X(\mathbf{u}) d\mathbf{u}$).
   * **Spatial Deformation Models** (Sampson-Guttorp multidimensional scaling to an isotropic latent space).
   * **Basis Function Approximations & Low-Rank Kriging** (Fixed Rank Kriging and Karhunen-Loève expansions for massive spatial datasets).

### Step B: Infuse "Classical-to-Spatial Bridges" + Equation Anatomies into Chapters 1–10
* Add a **"Classical-to-Spatial Bridge"** subsection to every chapter showing how classical i.i.d. estimators generalize into spatial estimators.
* Add **"Term-by-Term Equation Anatomy"** callouts and **"Wait—Why Did We Do That?"** micro-FAQs across the text.

### Step C: Expand All Exercise Workbooks to 16 Exercises per Chapter
* Add a **"Tier 0: Classical-to-Spatial Warm-Ups"** (4 new exercises per chapter, including all the classic Schabenberger & Gotway problems from your screenshots: Goldberger's $2\times 3$ table, continuous polynomial BLUP vs. $\operatorname{E}[Y\mid X]$, i.i.d. REML contrast vector $\mathbf{U}$, Genton's quadratic form matrix $\mathbf{A}(\mathbf{h})$, orthogonal Gaussian correlation, and convex hull extrapolation).

---

Created TODO.md

I have saved the master roadmap to [`manuscript/TODO.md`](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/TODO.md) so we can track and check off every item as we go. Here is the complete checklist:

---

# Master Monograph Roadmap & TODO Checklist

**Monograph Title:** *Theoretical Foundations of Spatial Statistics and Geostatistics*  
**Pedagogical Framework:** Maria L. Rizzo (*Three-Beat Rhythm*) $\times$ Oliver Schabenberger & Carol A. Gotway (*Semantic Linear-Model Discipline & Classical-to-Spatial Bridges*)

---

## Phase 0: Completed Foundation (100% Done)
- [x] **Core Geostatistical Chapters (Chapters 1–10):**
  - [x] [`ch01_foundations.md`](ch01_foundations.md) — Measure-Theoretic and Stochastic Foundations
  - [x] [`ch02_covariance_variogram.md`](ch02_covariance_variogram.md) — Spatial Covariance and Variogram Theory
  - [x] [`ch03_estimation.md`](ch03_estimation.md) — Semivariogram and Covariance Estimation
  - [x] [`ch04_parameter_estimation.md`](ch04_parameter_estimation.md) — Spatial Likelihood and Parametric Estimation (MLE, REML, WLS)
  - [x] [`ch05_blup_kriging.md`](ch05_blup_kriging.md) — Spatial Prediction and Kriging Theory (BLUP, SK, OK)
  - [x] [`ch06_spatial_regression.md`](ch06_spatial_regression.md) — Spatial Regression and Non-Stationary Means (GLS, UK, KED, IRF-$k$)
  - [x] [`ch07_cokriging.md`](ch07_cokriging.md) — Multivariate Geostatistics and Cokriging Theory
  - [x] [`ch08_support_regularization.md`](ch08_support_regularization.md) — Change of Support and Volume Regularization
  - [x] [`ch09_nonlinear_geostatistics.md`](ch09_nonlinear_geostatistics.md) — Non-Gaussian and Nonlinear Spatial Estimation (Lognormal, Indicator, Disjunctive)
  - [x] [`ch10_simulation.md`](ch10_simulation.md) — Spatial Simulation and Uncertainty Quantification (Journel, Cholesky, Turning Bands, SGS)
- [x] **Mathematical Appendices & Notation Index:**
  - [x] [`app_a.md`](app_a.md) — Functional Analysis, Hilbert Spaces, and RKHS
  - [x] [`app_b.md`](app_b.md) — Spectral Theory, Fourier Transforms, and Bochner's Theorem
  - [x] [`app_c.md`](app_c.md) — Matrix Calculus, Schur Complements, and Block System Inverses
  - [x] [`app_d.md`](app_d.md) — Numerical Optimization and Restricted Likelihood Geometry
  - [x] [`notation_index.md`](notation_index.md) — Master Notation Index
- [x] **Companion Exercise Workbooks — Baseline (12 Exercises per Chapter = 120 Solved Exercises):**
  - [x] [`ch01_exercises.md`](ch01_exercises.md) through [`ch10_exercises.md`](ch10_exercises.md) (Tiers I–IV, 20 pure Python simulation labs)

---

## Phase 1: Missing Foundational Paradigms from Schabenberger & Gotway (Chapters 11–13) (100% Done)

### 1.1 Chapter 11: Mapped Point Patterns and Spatial Point Processes
- [x] **Author [`ch11_point_processes.md`](ch11_point_processes.md):**
  - [x] **Classical-to-Spatial Bridge:** From classical 1D Poisson arrival counts and binomial trials to 2D/3D spatial point patterns $N(A)$ on bounded windows $W \subset \mathbb{R}^d$.
  - [x] **Complete Spatial Randomness (CSR):** Homogeneous Poisson Process axioms, void probabilities, conditional binomial property, and quadrat count $\chi^2$ dispersion index tests.
  - [x] **Intensity & Second-Order Moment Measures:** First-order intensity $\lambda(\mathbf{s})$, second-order product density $\lambda_2(\mathbf{s}_1, \mathbf{s}_2)$, pair correlation function $g(\mathbf{h})$, and **Campbell's Theorem** (with full proof).
  - [x] **Distance & Second-Order Summary Functions:**
    - [x] Event-to-event nearest-neighbor distribution $G(r)$ and point-to-event empty-space hazard function $F(r)$, plus van Lieshout & Baddeley's $J(r) = \frac{1 - G(r)}{1 - F(r)}$.
    - [x] **Ripley's $K(r)$ function** and Besag's variance-stabilized **$L(r) = \sqrt{K(r)/\pi}$** transform, with isotropic/translation edge-correction weights $w(\mathbf{s}_i, \mathbf{s}_j)$.
  - [x] **Contagion, Clustering, and Inhibition Models:**
    - [x] Inhomogeneous Poisson Processes $\lambda(\mathbf{s})$ and Berman-Turner likelihood approximation.
    - [x] **Cox Processes & Log-Gaussian Cox Processes (LGCP):** Doubly stochastic processes $\Lambda(\mathbf{s}) = \exp(Y(\mathbf{s}))$ bridging geostatistical random fields and point patterns.
    - [x] **Neyman-Scott & Matérn Cluster Processes:** Analytical derivation of $K(r)$ via parent-offspring convolution.
    - [x] **Gibbs / Markov Point Processes & Strauss Inhibition Process:** Pairwise interaction potentials and Papangelou conditional intensity.
    - [x] **Multivariate & Marked Point Patterns:** Cross-$K_{ij}(r)$ functions and random labeling vs. independence.
  - [x] **Pure Python Algorithmic Engines (Beat 2):**
    - [x] Exact Lewis-Shedler thinning algorithm for Inhomogeneous Poisson & Log-Gaussian Cox Processes.
    - [x] Empirical Ripley's $\widehat{K}(r)$ and $\widehat{L}(r)$ estimator with rectangular border/isotropic edge correction and CSR Monte Carlo simulation envelopes.
- [x] **Author Companion Workbook [`ch11_exercises.md`](ch11_exercises.md) (16 Fully Solved Exercises across Tiers 0–IV).**

### 1.2 Chapter 12: Areal/Lattice Models (SAR, CAR, SMA), Spatial Autocorrelation & Spatial GLMMs
- [x] **Author [`ch12_lattice_areal_models.md`](ch12_lattice_areal_models.md):**
  - [x] **Classical-to-Spatial Bridge:** From time-series lag-1 autocorrelation and the Durbin-Watson statistic $d = \frac{\sum (e_t - e_{t-1})^2}{\sum e_t^2}$ to 2D graph neighborhood topologies.
  - [x] **Exploratory Spatial Data Analysis (ESDA) & Proximity Matrices:**
    - [x] Adjacency / spatial weight matrices $\mathbf{W}$ (rook, queen, $k$-nearest, row-standardized) and graph Laplacians.
    - [x] **Moran's $I$** and **Geary's $c$** written as explicit matrix quadratic forms $\frac{\mathbf{Z}'\mathbf{P}\mathbf{W}\mathbf{P}\mathbf{Z}}{\mathbf{Z}'\mathbf{P}\mathbf{Z}}$; exact mean and variance under normality vs. randomization.
    - [x] Local Indicators of Spatial Association (LISA).
  - [x] **Markov Random Fields on Lattices:**
    - [x] **Simultaneous Autoregressive (SAR)** models: $(\mathbf{I} - \rho \mathbf{W})\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon} \implies \boldsymbol{\Sigma}_{\text{SAR}} = \sigma^2 (\mathbf{I} - \rho \mathbf{W})^{-1}(\mathbf{I} - \rho \mathbf{W}')^{-1}$.
    - [x] **Conditional Autoregressive (CAR)** models: full conditionals $Y_i \mid \mathbf{Y}_{-i} \sim \mathcal{N}(\mu_i + \sum_j c_{ij}(y_j - \mu_j), \tau_i^2)$, **Brook's Lemma** (full proof), **Hammersley-Clifford Theorem**, and $\boldsymbol{\Sigma}_{\text{CAR}} = (\mathbf{I} - \mathbf{C})^{-1}\mathbf{M}$.
    - [x] **Spatial Moving Average (SMA)** models and Jacobian log-determinant $\ln|\mathbf{I} - \rho \mathbf{W}|$ eigenvalue bounds $(\lambda_{\min}^{-1} < \rho < \lambda_{\max}^{-1})$.
  - [x] **Geographically Weighted Regression (GWR) & Local Kernel Smoothing:**
    - [x] Local weighted least squares $\widehat{\boldsymbol{\beta}}(\mathbf{s}_0) = (\mathbf{X}'\mathbf{W}(\mathbf{s}_0)\mathbf{X})^{-1}\mathbf{X}'\mathbf{W}(\mathbf{s}_0)\mathbf{Y}$, effective degrees of freedom $\operatorname{tr}(\mathbf{S}_\lambda)$, and CV bandwidth selection $\lambda$.
  - [x] **Spatial Generalized Linear Mixed Models (Spatial GLMMs / Model-Based Geostatistics):**
    - [x] Non-Gaussian spatial responses (Poisson counts, Binomial proportions) conditioned on latent Gaussian random fields $Y_i \mid S(\mathbf{s}_i) \sim \text{EF}(g^{-1}(\mathbf{x}_i'\boldsymbol{\beta} + S(\mathbf{s}_i)))$.
    - [x] Laplace approximation, Penalized Quasi-Likelihood (PQL), and Langevin-Hastings MCMC.
  - [x] **Pure Python Algorithmic Engines (Beat 2):**
    - [x] Moran's $I$ & Geary's $c$ exact & permutation test engine.
    - [x] SAR / CAR Maximum Likelihood solver and Geographically Weighted Regression (GWR) bandwidth optimizer.
- [x] **Author Companion Workbook [`ch12_exercises.md`](ch12_exercises.md) (16 Fully Solved Exercises across Tiers 0–IV, including S&G Problem 5.13 localized kernel regression table).**

### 1.3 Chapter 13: Non-Stationary Covariance Architectures & Scalable Low-Rank Kriging
- [x] **Author [`ch13_nonstationary_covariance.md`](ch13_nonstationary_covariance.md):**
  - [x] **Classical-to-Spatial Bridge:** From heteroskedastic regression (weighted least squares with diagonal $\operatorname{diag}(\sigma_i^2)$) and PCA / SVD low-rank matrix approximation to non-stationary spatial kernels $C(\mathbf{s}_i, \mathbf{s}_j)$.
  - [x] **Process / Kernel Convolution Models (Higdon, Paciorek & Schervish):**
    - [x] Moving average representation $Z(\mathbf{s}) = \int_{\mathbb{R}^d} K_{\mathbf{s}}(\mathbf{u}) W(d\mathbf{u})$ with spatially varying kernel matrices $\boldsymbol{\Sigma}(\mathbf{s})$.
    - [x] Complete proof of the **Paciorek-Schervish Non-Stationary Matérn Closed-Form Covariance** and its automatic positive definiteness.
  - [x] **Spatial Deformation Models (Sampson-Guttorp Method):**
    - [x] Bijective warping $f: D \to \tilde{D}$ mapping non-stationary geographic coordinates into an isotropic dispersion space: $C(\mathbf{s}_i, \mathbf{s}_j) = C_0(\|f(\mathbf{s}_i) - f(\mathbf{s}_j)\|)$ via thin-plate splines and multidimensional scaling.
  - [x] **Basis Function Representations & Fixed Rank Kriging (Cressie & Johannesson):**
    - [x] Karhunen-Loève orthogonal expansion $Z(\mathbf{s}) = \sum_{k=1}^r \eta_k \phi_k(\mathbf{s}) + \epsilon(\mathbf{s})$ with $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$.
    - [x] **Sherman-Morrison-Woodbury Matrix Identity** reducing $\mathcal{O}(n^3)$ inversion of $\mathbf{V} = \mathbf{S}\mathbf{K}\mathbf{S}' + \tau^2 \mathbf{I}_n$ to $\mathcal{O}(n r^2 + r^3)$ operations!
  - [x] **Pure Python Algorithmic Engines (Beat 2):**
    - [x] Paciorek-Schervish non-stationary covariance matrix assembler.
    - [x] Fixed Rank Kriging (FRK) Woodbury $\mathcal{O}(n r^2)$ fast solver vs. exact $\mathcal{O}(n^3)$ benchmark.
- [x] **Author Companion Workbook [`ch13_exercises.md`](ch13_exercises.md) (16 Fully Solved Exercises across Tiers 0–IV).**

---

## Phase 2: Anti-Gatekeeping & Pedagogical Upgrades Across Chapters 1–10

> [!TIP] Subagent Strategy: HIGH UTILITY
> **Recommended Approach:** Spawn 3–4 parallel subagents partitioned across chapter clusters:
> - **Subagent A (Ch 1–3):** Measure foundations, covariance theory, and semivariogram estimation.
> - **Subagent B (Ch 4–6):** Spatial likelihood/REML, Kriging BLUP, and spatial drift regression.
> - **Subagent C (Ch 7–8):** Multivariate cokriging and support regularization.
> - **Subagent D (Ch 9–10):** Nonlinear/indicator geostatistics and stochastic simulation.
> *Reasoning:* Each chapter is an independent file with locked core text. The task is modular surgical insertion (Bridges, Equation Anatomies, and Micro-FAQs), making it embarrassingly parallel. The primary agent reviews diffs to ensure existing proofs remain intact.

Add three signature pedagogical elements to every existing chapter (`ch01` through `ch10`):
1. **The Classical-to-Spatial Bridge:** Explicit derivation connecting i.i.d. statistical estimators ($\bar{Y}$, $S^2$, OLS, Pearson $r$, ANOVA contrasts) to their spatial counterparts.
2. **Term-by-Term Equation Anatomy Callouts:** Plain-English structural decoders immediately beneath major matrix/functional equations.
3. **"Wait—Why Did We Do That?" Micro-FAQ Boxes:** Clarifying subtle sign flips, normalization constants, and geometric assumptions right where students typically get stuck.

- [ ] **Upgrade [`ch01_foundations.md`](ch01_foundations.md):** Add i.i.d. vs. correlated sample mean variance $\operatorname{Var}(\bar{Z})$ and effective sample size $n_{\text{eff}}$ bridge + Equation Anatomies.
- [ ] **Upgrade [`ch02_covariance_variogram.md`](ch02_covariance_variogram.md):** Add scalar variance $\to$ PD/CND quadratic form bridge + Bessel/Hankel Equation Anatomies.
- [ ] **Upgrade [`ch03_estimation.md`](ch03_estimation.md):** Add Genton's quadratic form spatial design matrix $\widehat{\gamma}(\mathbf{h}) = \frac{1}{2}\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}$ + periodogram-covariance bridge + FAQ on the $\frac{1}{2}$ factor.
- [ ] **Upgrade [`ch04_parameter_estimation.md`](ch04_parameter_estimation.md):** Add i.i.d. $(n-1)$ REML contrast vector $\mathbf{U} = [Y_i - \bar{Y}]'$ bridge + Composite Likelihood section + REML projection matrix $\mathbf{P}$ Anatomy.
- [ ] **Upgrade [`ch05_blup_kriging.md`](ch05_blup_kriging.md):** Add Classical Prediction Theorem ($\operatorname{E}[Y \mid \mathbf{U}]$ vs. linear BLUP on bivariate distributions) + FAQ on Lagrange multiplier sign flip between covariance and variogram forms.
- [ ] **Upgrade [`ch06_spatial_regression.md`](ch06_spatial_regression.md):** Add Cholesky whitening bridge ($\mathbf{L}^{-1}\mathbf{Y}$ reducing GLS to OLS) + UK saddle-point block matrix Anatomy.
- [ ] **Upgrade [`ch07_cokriging.md`](ch07_cokriging.md):** Add classical multivariate regression / canonical correlation bridge + LMC & Cokriging block matrix Anatomy.
- [ ] **Upgrade [`ch08_support_regularization.md`](ch08_support_regularization.md):** Add classical central limit theorem / group-means ANOVA variance decomposition bridge to Krige's Additivity Theorem.
- [x] **Upgrade [`ch09_nonlinear_geostatistics.md`](ch09_nonlinear_geostatistics.md):** Add classical logistic/probit indicator bridge and Fourier series $\to$ Hermite polynomial expansion bridge.
- [x] **Upgrade [`ch10_simulation.md`](ch10_simulation.md):** Add classical regression conditional prediction interval vs. Monte Carlo residual bootstrap bridge to Journel's conditioning theorem.

---

## Phase 3: "Tier 0: Classical-to-Spatial Warm-Ups" Exercise Expansion (Ch 1–10 Workbooks: 12 $\to$ 16 Exercises Each)

> [!TIP] Subagent Strategy: VERY HIGH UTILITY (Highest ROI)
> **Recommended Approach:** Dispatch 5 parallel `exercise_author` subagents assigned to pairs of workbooks:
> - **Subagent 1:** `ch01_exercises.md` & `ch02_exercises.md` (+8 solved problems)
> - **Subagent 2:** `ch03_exercises.md` & `ch04_exercises.md` (+8 solved problems)
> - **Subagent 3:** `ch05_exercises.md` & `ch06_exercises.md` (+8 solved problems)
> - **Subagent 4:** `ch07_exercises.md` & `ch08_exercises.md` (+8 solved problems)
> - **Subagent 5:** `ch09_exercises.md` & `ch10_exercises.md` (+8 solved problems)
> *Reasoning:* Authoring 40 unabridged multi-part mathematical solutions is highly compute- and token-intensive (~10,000+ lines of derivations). Because each workbook is entirely decoupled, this is perfectly parallelizable. The primary agent's role is to supply the exact problem specs and run post-hoc automated verification on problem numbering (16 per workbook) and LaTeX delimiters.

Expand each existing companion workbook (`ch01_exercises.md` through `ch10_exercises.md`) from 12 to **16 fully solved exercises** (+40 new exercises), incorporating the classic Schabenberger & Gotway problem archetypes:

- [ ] **Expand [`ch01_exercises.md`](ch01_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] Effective sample size $n_{\text{eff}}$ of equicorrelated and AR(1) spatial transects.
  - [ ] Explicit $t$-test Type I error inflation under ignored spatial correlation.
  - [ ] Bivariate Gaussian orthogonal decomposition $Y = \rho X + \sqrt{1-\rho^2}U$ and proving $\operatorname{Corr}[X^2, Y^2] = \rho^2$ (**S&G Problem 4.3**).
  - [ ] Mean-square derivative existence via second-order finite differences.
- [ ] **Expand [`ch02_exercises.md`](ch02_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] Proving positive definiteness of elementary cosine/exponential kernels via trigonometric identities.
  - [ ] Constructing valid covariance kernels via linear combinations, products, and scale mixtures.
  - [ ] Proving that $\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0))$ can become negative when using an invalid (non-CND) "variogram".
  - [ ] Explicit calculation of integral scale $I = \int_0^\infty \rho(h) dh$ across canonical families.
- [ ] **Expand [`ch03_exercises.md`](ch03_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] **Genton's Spatial Design Matrix $\mathbf{A}(\mathbf{h})$ (S&G Problem 4.4):** Explicit construction of $\widehat{\gamma}(\mathbf{h}) = \frac{1}{2}\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}$ on a 1D regular transect and a $3 \times 3$ spatial lattice.
  - [ ] Exact expectation and variance of $\widehat{\gamma}(\mathbf{h}) = \frac{1}{2}\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}$ via Gaussian quadratic form trace formulas $\operatorname{E}[\mathbf{Z}'\mathbf{A}\mathbf{Z}] = \operatorname{tr}(\mathbf{A}\mathbf{V})$ and $\operatorname{Var}(\mathbf{Z}'\mathbf{A}\mathbf{Z}) = 2\operatorname{tr}((\mathbf{A}\mathbf{V})^2)$.
  - [ ] **Periodogram & Sample Covariance Equivalence (S&G Problem 4.7):** Proving the discrete Fourier transform duality on an $r \times c$ lattice.
  - [ ] Effect of power transformation $Z^\alpha$ on empirical semivariograms (**S&G Problem 4.1**).
- [ ] **Expand [`ch04_exercises.md`](ch04_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] **Classical i.i.d. REML via Difference Vector $\mathbf{U}$ (S&G Problem 4.2):** Full derivation of MLE vs. REML on $Y_i = \beta + e_i$ using $\mathbf{U} = [Y_1 - \bar{Y}, \dots, Y_{n-1} - \bar{Y}]'$.
  - [ ] **Composite Likelihood Score Equation (S&G Problem 4.5):** Deriving pairwise difference log-likelihood and score equations for $T_{ij}^{(2)} = Z(\mathbf{s}_i) - Z(\mathbf{s}_j) \sim \mathcal{N}(0, 2\gamma(\mathbf{h}_{ij}, \boldsymbol{\theta}))$.
  - [ ] **Squared-Difference Composite Likelihood (S&G Problem 4.6):** Deriving the composite score from the $\chi_1^2$ distribution of $T_{ij}^{(3)} = (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2$.
  - [ ] Explicit $3 \times 3$ spatial projection matrix $\mathbf{P}$ trace and rank verification.
- [ ] **Expand [`ch05_exercises.md`](ch05_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] **The Fundamental Prediction Theorem (S&G Problem 5.1 & 5.2):** Proving $\operatorname{E}[(Y - \operatorname{E}[Y \mid \mathbf{U}])^2] \le \operatorname{E}[(Y - g(\mathbf{U}))^2]$ and $\text{MSPE} = \operatorname{Var}[Z(\mathbf{s}_0)] - \operatorname{Var}[p^0(\mathbf{Z}; \mathbf{s}_0)]$.
  - [ ] **Goldberger's $2 \times 3$ Discrete Contingency Table (S&G Problem 5.5):** Explicit point-by-point comparison of non-linear $\operatorname{E}[Y \mid X]$ vs. linear $\operatorname{BLUP}(Y \mid X)$ and their MSEs.
  - [ ] **Continuous Bivariate Density $f(x,y) = \frac{6}{7}(x+y)^2$ (S&G Problem 5.6):** Analytical derivation of $\operatorname{E}[Y \mid X]$ vs. $\operatorname{BLUP}(Y \mid X)$ on $[0,1]^2$.
  - [ ] **Extrapolation Outside the Convex Hull & Screening Diagnostics (S&G Problem 5.11):** Comparing interpolation vs. extrapolation behavior across nugget, sill, and range variations.
- [ ] **Expand [`ch06_exercises.md`](ch06_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] **OLS vs. MLE vs. BLUP Equivalence Bridge (S&G Problem 5.3):** Revisiting moment assumptions vs. Gaussianity.
  - [ ] Explicit Cholesky whitening transformation $\mathbf{Y}^* = \mathbf{L}^{-1}\mathbf{Y}$ on a 3-point correlated trend model.
  - [ ] Verifying that Ordinary Kriging is identical to Simple Kriging with $\mu$ replaced by $\widehat{\mu}_{\text{GLS}}$ (**S&G Problem 5.9**).
  - [ ] Generalized covariance order-$k$ polynomial annihilation by hand.
- [ ] **Expand [`ch07_exercises.md`](ch07_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] Classical bivariate regression when one variable is partially missing (the i.i.d. precursor to heterotopic cokriging).
  - [ ] Positive definiteness of $2 \times 2$ coregionalization matrices via Cauchy-Schwarz cross-variogram bounds $|\gamma_{12}(\mathbf{h})| \le \sqrt{\gamma_{11}(\mathbf{h})\gamma_{22}(\mathbf{h})}$.
  - [ ] Pseudo-cross-variogram vs. traditional cross-variogram under complete heterotopy.
  - [ ] Principal Component Kriging (orthogonalizing coregionalized variables) vs. full Cokriging.
- [ ] **Expand [`ch08_exercises.md`](ch08_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] **Bilinear Block Covariance Proof (S&G Problem 5.12):** Proving $\operatorname{Cov}[Z(B_i), Z(B_j)] = \frac{1}{|B_i||B_j|}\int_{B_i}\int_{B_j} C(\mathbf{u}, \mathbf{v}) d\mathbf{u}d\mathbf{v}$ from Fubini's theorem.
  - [ ] Classical one-way ANOVA variance decomposition as the i.i.d. limit of Krige's Additivity Theorem.
  - [ ] Exact 1D analytical regularization of the Spherical variogram over segment support $L$.
  - [ ] Proportional effect and heteroskedastic block variance scaling.
- [ ] **Expand [`ch09_exercises.md`](ch09_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] Classical delta-method bias correction vs. exact lognormal MGF back-transformation.
  - [ ] Bivariate Gaussian indicator covariance $C_I(\mathbf{h}; u)$ calculation at the median $u = 0$ (Sheppard's arcsine formula: $C_I(\mathbf{h}; 0) = \frac{1}{2\pi}\arcsin\rho(\mathbf{h})$).
  - [ ] Explicit 3-term Hermite polynomial expansion of $f(Y) = \exp(\sigma Y)$ and comparison against exact lognormal variance.
  - [ ] Uniform score transform (rank/empirical CDF) and grade-tonnage selectivity curves (Lorenz curves & Gini coefficient).
- [ ] **Expand [`ch10_exercises.md`](ch10_exercises.md) (+4 Warm-Up & Bridge Exercises):**
  - [ ] Classical parametric bootstrap vs. conditional simulation at a single unsampled point.
  - [ ] Spectral simulation via cosine waves with random phase $Z(\mathbf{s}) = \sqrt{2}\cos(\boldsymbol{\omega}'\mathbf{s} + \Phi)$ ($\Phi \sim \text{Unif}(0, 2\pi)$): proving exact covariance reproduction via trigonometric expectation.
  - [ ] Moving average / discrete convolution simulation on a 1D grid: deriving filter weights $w_k$ for exponential/spherical covariances.
  - [ ] Quantifying joint exceedance probability $P(Z(\mathbf{s}_1) > u \text{ and } Z(\mathbf{s}_2) > u \mid \mathbf{Z})$ via ensemble realizations vs. naive product of marginal kriging probabilities.

---

## Phase 4: Master Repository Synchronization & Verification

> [!CAUTION] Subagent Strategy: LOW UTILITY / COUNTER-PRODUCTIVE (Handle in Primary Thread)
> **Recommended Approach:** Keep this phase exclusively within the primary agent session.
> *Reasoning:* Phase 4 operates on central, shared global indices (`README.md`, `TODO.md`, `notation_index.md`). Concurrently writing to these files from subagents causes race conditions and git write conflicts. Furthermore, verifying execution across 26 computational Python scripts and checking all 34 Markdown files requires interactive CLI access with immediate error feedback.

- [ ] Update [`README.md`](README.md) to index all 13 Chapters, 13 Companion Workbooks (208 Major Exercises, ~750 Solved Sub-Problems, 26 Pure Python Engines), and Appendices A–D.
- [ ] Update [`notation_index.md`](notation_index.md) with Point Process ($N(A), \lambda(\mathbf{s}), K(r), L(r), G(r), F(r)$), Lattice/Areal ($\mathbf{W}$, Moran's $I$, Geary's $c$, SAR/CAR), and Non-Stationary Covariance symbols.
- [ ] Execute automated verification across all 29 markdown files (testing all 26 Python simulation scripts and validating all LaTeX delimiters).

---

## Phase 5: Quality Checks, Notation Audits & Accuracy Review

+────────────────────────────────────────────────────────────────────────────────────────────────────+
|                               PHASE 5: MULTI-REVIEWER AUDIT ARCHITECTURE                           |
+───────────────────────────────────────────────────┬────────────────────────────────────────────────+
|  REVIEWER 1: Mathematical Rigor & Proofs          |  REVIEWER 2: Semantic Notation & S&G Discipline|
|  - Verifies every derivation step and bound       |  - Strict bold uppercase matrix: X, V, W, C    |
|  - Validates measure-theoretic regularity         |  - Strict bold lowercase vector: s, h, beta, u |
|  - Invertibility, PD/CND checks, rank conditions  |  - Continuous vector indexing: s in D subset R^d|
+───────────────────────────────────────────────────┼────────────────────────────────────────────────+
|  REVIEWER 3: Pedagogical Flow & Anti-Gatekeeping  |  REVIEWER 4: Pure Python Code & Algorithms     |
|  - Enforces Rizzo Three-Beat Cadence              |  - Zero third-party dependencies (std lib only)|
|  - Classical-to-Spatial Bridges (i.i.d. on-ramps) |  - 1-to-1 formula-to-variable naming parity    |
|  - Term-by-Term Equation Anatomies & Micro-FAQs   |  - Numerical stability, pivots, and tolerance  |
+───────────────────────────────────────────────────┴────────────────────────────────────────────────+
                                        │
                                        ▼
                   [Structured Audit Reports: audit_report_chXX.md]
                   - Exact file and line numbers
                   - Severity: [CRITICAL] | [WARNING] | [PEDAGOGICAL GAP] | [STYLE]
                   - Existing text snippet vs. proposed drop-in patch
                                        │
                                        ▼
                  [Primary Agent Synthesis, Triage & Surgical Commit]

> [!TIP] Subagent Strategy: EXTREMELY HIGH UTILITY (Embarrassingly Parallel)
> **Recommended Approach:** Dispatch specialized reviewer subagents across chapter batches running read-only audits.
> - **Why Subagents Excel Here:** Auditing 13 theoretical chapters, 13 exercise workbooks, and 4 appendices (~30,000 lines of rigorous mathematics and Python code) requires deep inspection across distinct orthogonal dimensions (proof steps, notation conventions, pedagogical flow, code correctness). Because auditing is read-only, multiple subagents can review chapters simultaneously without risk of git conflicts or race conditions.
> - **Audit Artifact Protocol:** Reviewer subagents do not edit chapter files directly. Instead, each reviewer compiles a structured audit report (`audit_report_chXX.md`) cataloging findings by severity:
>   - `[CRITICAL]`: Analytical error, invalid proof step, missing regularity condition, wrong matrix dimension, or runtime code bug.
>   - `[WARNING]`: Notation inconsistency (e.g., lowercase matrix, unbolded vector), ambiguous term, skipped intermediate derivation step.
>   - `[PEDAGOGICAL GAP]`: Contextual gap, unmotivated definition, missing physical intuition, or broken "Classical-to-Spatial Bridge".
>   - `[STYLE]`: LaTeX whitespace padding inside `$...$`, raw Unicode characters instead of LaTeX commands, or formatting anomaly.
>   Every finding must record: (1) exact file and line number, (2) current text snippet, (3) specific failure mode, and (4) proposed drop-in correction.
> - **Primary Agent Role:** Triage audit reports, verify findings, apply surgical corrections to the manuscript, and ensure repository-wide consistency.

### 5.0 Four Specialized Reviewer Audit Personas
- [ ] **Reviewer Persona 1: Mathematical Rigor & Proof Integrity Specialist**
  - [ ] Audit all theorem, lemma, and proposition proofs for completeness: zero skipped steps, explicit justification for every algebraic transformation, and verified convergence/regularity conditions.
  - [ ] Check matrix-vector dimension compatibility across all quadratic forms (e.g., $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$, $\mathbf{Z}'\mathbf{P}\mathbf{W}\mathbf{P}\mathbf{Z}$) and verify nonsingularity/invertibility assumptions.
  - [ ] Audit asymptotic arguments (infill vs. increasing-domain, spectral density existence, domain boundary integrals).
- [ ] **Reviewer Persona 2: Semantic Notation & Schabenberger & Gotway Discipline Auditor**
  - [ ] Enforce strict S&G linear model discipline: matrices bold uppercase ($\mathbf{X}, \mathbf{V}, \mathbf{W}, \mathbf{Z}, \mathbf{C}$), vectors bold lowercase ($\mathbf{s}, \mathbf{h}, \boldsymbol{\beta}, \boldsymbol{\lambda}, \mathbf{u}$), scalars math italic ($s_i, h, \sigma^2, \tau^2, \gamma(\mathbf{h})$).
  - [ ] Audit coordinate vector indexing: spatial coordinates must be vector-valued $\mathbf{s} = (s_1, \dots, s_d)' \in D \subset \mathbb{R}^d$, discrete sample locations $\mathbf{s}_i$, lag separation vectors $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$, Euclidean distance $h = \|\mathbf{h}\|$.
  - [ ] Audit LaTeX delimiter standards: single dollar signs with zero inner whitespace (e.g. `$Z(\mathbf{s})$` without leading or trailing spaces inside delimiters), display math on dedicated standalone lines (`$$...$$`), and zero raw Unicode math symbols (`\mathbb{R}^d` not `ℝ^d`, `\sigma^2` not `σ²`, `\le` not `<=`).
- [ ] **Reviewer Persona 3: Pedagogical Flow & Anti-Gatekeeping Auditor**
  - [ ] Enforce Maria L. Rizzo's Three-Beat Cadence: Beat 1 (Mathematical Anchor & Statistical Dilemma) $\to$ Beat 2 (Algorithmic Translation) $\to$ Beat 3 (Reality Diagnostics).
  - [ ] Audit "Classical-to-Spatial Bridges": Verify that every chapter grounds spatial concepts in non-spatial 1D / i.i.d. precursors (sample mean variance, OLS, Durbin-Watson, Poisson arrival counts, ANOVA).
  - [ ] Audit Term-by-Term Equation Anatomies: Ensure dense matrix equations have structured plain-English decoders.
  - [ ] Audit Micro-FAQ callouts: Ensure common student stumbling blocks (Lagrange multiplier sign flips, $\frac{1}{2}$ variogram factors, nugget singularities) are proactively resolved.
- [ ] **Reviewer Persona 4: Pure Python Code & Algorithmic Accuracy Auditor**
  - [ ] Enforce strict zero-dependency policy: Python standard library only (`math`, `random`, procedural loops; no `numpy`, `scipy`, `sklearn`, `gstat`).
  - [ ] Check 1-to-1 variable-to-formula naming parity with mathematical derivations in the text.
  - [ ] Numerical stability verification: check condition number handling, positive definiteness guards, and division-by-zero protection at $\|\mathbf{h}\| = 0$.
  - [ ] Execute all simulation engines and labs to ensure 100% test passing and deterministic reproducibility.

---

### 5.1 Chapter-by-Chapter Review Batches
- [ ] **Batch 5.1: Part I — Probabilistic Foundations & Covariance Theory (Chapters 1–2, Appendices A–B)**
  - [ ] Review [`ch01_foundations.md`](ch01_foundations.md): Kolmogorov consistency proof steps, Feldman-Hájek dichotomy, infill asymptotics, coordinate notation.
  - [ ] Review [`ch02_covariance_variogram.md`](ch02_covariance_variogram.md): Bochner's theorem, Schoenberg CND equivalence, Matérn closed forms, Hankel transforms, spherical barrier in $\mathbb{R}^4$.
  - [ ] Review [`app_a.md`](app_a.md) & [`app_b.md`](app_b.md): Functional analysis, Hilbert spaces, RKHS, Fourier transforms, spectral representation.
- [ ] **Batch 5.2: Part II — Spatial Estimation, Likelihood & Parametric Inference (Chapters 3–4, Appendix D)**
  - [ ] Review [`ch03_estimation.md`](ch03_estimation.md): Matheron method-of-moments unbiasedness proof, Cressie-Hawkins $0.457/0.494$ derivation via $\Gamma(3/4)$, Genton design matrix $\mathbf{A}(\mathbf{h})$.
  - [ ] Review [`ch04_parameter_estimation.md`](ch04_parameter_estimation.md): Profile likelihood gradients, REML projection matrix $\mathbf{P}$ properties ($\mathbf{P}\mathbf{X} = \mathbf{0}$, $\mathbf{P}\mathbf{V}\mathbf{P} = \mathbf{P}$), Cressie WLS weights, Fisher Information matrix.
  - [ ] Review [`app_d.md`](app_d.md): Numerical optimization geometry, Fisher scoring, profile likelihood profiling.
- [ ] **Batch 5.3: Part III — Optimal Prediction, Spatial Regression & Cokriging (Chapters 5–7, Appendix C)**
  - [ ] Review [`ch05_blup_kriging.md`](ch05_blup_kriging.md): BLUP Hilbert space projection theorem, Simple vs. Ordinary Kriging duality, Lagrange multiplier derivations, smoothing deficit bounds.
  - [ ] Review [`ch06_spatial_regression.md`](ch06_spatial_regression.md): Gauss-Markov BLUE for GLS, Universal Kriging saddle-point block system, Cholesky whitening, IRF-$k$ generalized covariance.
  - [ ] Review [`ch07_cokriging.md`](ch07_cokriging.md): Cross-covariance transpose symmetry ($C_{ij}(\mathbf{h}) = C_{ji}(-\mathbf{h})$), Linear Model of Coregionalization (LMC) positive definiteness, autokrigeability Kronecker reduction.
  - [ ] Review [`app_c.md`](app_c.md): Matrix calculus, block matrix inverses, Schur complements, Woodbury formula derivations.
- [ ] **Batch 5.4: Part IV — Support Regularization, Nonlinearities & Simulation (Chapters 8–10)**
  - [ ] Review [`ch08_support_regularization.md`](ch08_support_regularization.md): Dispersion variance $D^2(v \mid V)$, Krige's Additivity Theorem proof, Cartier's relation, block variogram numerical integration.
  - [ ] Review [`ch09_nonlinear_geostatistics.md`](ch09_nonlinear_geostatistics.md): Lognormal MGF back-transformation bias correction, Indicator Kriging order relations, PAVA isotonic regression, Hermite polynomial expansions.
  - [ ] Review [`ch10_simulation.md`](ch10_simulation.md): Kriging covariance distortion proof, Journel's Fundamental Conditioning Theorem, Sequential Gaussian Simulation (SGS) conditional distribution chain rule, Turning Bands 3D-to-1D Radon inversion.
- [ ] **Batch 5.5: Part V — Advanced Paradigms: Point Processes, Lattices & Non-Stationarity (Chapters 11–13)**
  - [ ] Review [`ch11_point_processes.md`](ch11_point_processes.md): Campbell's theorem proof, Ripley's $K(r)$ and Besag's $L(r)$ variance-stabilizing transform, edge-correction derivations, Cox/LGCP intensity models.
  - [ ] Review [`ch12_lattice_areal_models.md`](ch12_lattice_areal_models.md): Brook's Lemma proof, Hammersley-Clifford theorem, SAR vs. CAR precision matrix structures, Moran's $I$ & Geary's $c$ quadratic forms and expectation/variance derivations.
  - [ ] Review [`ch13_nonstationary_covariance.md`](ch13_nonstationary_covariance.md): Paciorek-Schervish non-stationary Matérn positive definiteness proof, Sampson-Guttorp coordinate deformation, Fixed Rank Kriging (FRK) Woodbury $\mathcal{O}(nr^2)$ reduction.
- [ ] **Batch 5.6: Comprehensive Companion Workbooks Audit (Chapters 1–13 Workbooks)**
  - [ ] Audit all 13 companion workbooks ([`ch01_exercises.md`](ch01_exercises.md) through [`ch13_exercises.md`](ch13_exercises.md)):
    - [ ] Verify that every problem has a complete, unabridged step-by-step solution (`##### Solution to Exercise X.Y`).
    - [ ] Check exact numerical calculations in Tier II pencil-and-paper mechanics exercises against matrix equations.
    - [ ] Test and run all Tier III Python computational lab engines (26 pure Python scripts) for zero-dependency conformance, clean execution, and correct empirical assertions.
    - [ ] Check Tier IV edge cases and diagnostic reflections for clarity and anti-gatekeeping pedagogical impact.
- [ ] **Batch 5.7: Master Triage, Synthesis & Patch Consolidation**
  - [ ] Collate all reviewer audit reports into a unified quality log.
  - [ ] Primary agent applies verified mathematical, notational, and contextual corrections across all files.
  - [ ] Run repository-wide automated verification suite (Python scripts + LaTeX delimiter audit) to confirm 100% clean compilation.
