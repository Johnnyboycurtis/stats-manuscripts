# Quality Audit & Notation Review Report: Batch 5.5
**Monograph:** *Theoretical Foundations of Spatial Statistics and Geostatistics*  
**Reviewer:** Reviewer Subagent 5 (Phase 5 Quality Assurance & Notation Audit)  
**Target Batch:** Chapters 11, 12, 13 and Corresponding Exercise Suites (11, 12, 13)  
**Date of Audit:** October 23, 2026  
**Files Audited:**
1. `manuscript/ch11_point_processes.md` (669 lines)
2. `manuscript/ch11_exercises.md` (1,343 lines)
3. `manuscript/ch12_lattice_areal_models.md` (494 lines)
4. `manuscript/ch12_exercises.md` (1,319 lines)
5. `manuscript/ch13_nonstationary_covariance.md` (519 lines)
6. `manuscript/ch13_exercises.md` (1,074 lines)

---

## Executive Summary

A comprehensive, end-to-end mathematical, algorithmic, and pedagogical review was performed across Batch 5.5. The review covered theoretical derivations, statistical notation conventions, analytical proofs, algebraic consistency, and pure Python implementations (verified using Python 3 standard library only: `math`, `random`, `time`).

### Summary of Audit Classifications
- **`[CRITICAL]` Findings:** 2 items
  1. *Chapter 11 (Line 110):* Algebraic proof error in the Poisson conditional uniform distribution derivation where an extraneous $n!$ in the numerator yields an algebraic contradiction $\frac{(n!)^2}{|W|^n} \ne \frac{1}{|W|^n}$.
  2. *Chapter 12 (Lines 416–417):* Algorithmic bug in the reference implementation of Moran's $I$ analytical variance (`compute_morans_i`), conflating Cliff-Ord normality and randomization formulas, hardcoding an incorrect kurtosis constant, corrupting the second numerator term, and omitting the subtraction of $\operatorname{E}[I]^2$.
- **`[WARNING]` Findings:** 2 items
  1. *Chapter 12 (Line 112):* Erroneous cross-reference citing Exercise 12.2 (which covers spectral bounds) instead of Exercise 12.5 for the analytical derivation of Moran's $I$ variance.
  2. *Chapter 12 (Line 134):* Mathematical condition omission in Local Moran's $I_i$ summation identity; $\sum_{i=1}^n I_i = S_0 \cdot I$, which only equals $n \cdot I$ under row-standardized weights.
- **`[PEDAGOGICAL GAP]` Findings:** 1 item
  1. *Chapter 11 Exercises (Lines 195–208, Exercise 11.3):* Subtle unstated scale distinction between between-quadrat spatial autocorrelation (which depresses sample variance $S^2$) and within-quadrat point clustering (which inflates underlying population variance $\sigma^2$).
- **`[STYLE]` Findings:** 1 item
  1. *Chapter 11 (Line 366):* Typographical duplication ("Total Pair Pairs" instead of "Total Ordered Pairs") in equation anatomy.
- **`[CLEAN]` Confirmations:**
  - Chapter 11 (Core Foundations, Campbell's & Slivnyak's Theorems, Ripley's $K$, LGCP, Python Thinning & Estimator)
  - Chapter 11 Exercises (All 16 Exercises, PGFLs, Neyman-Scott / Thomas clustering, Monte Carlo Envelopes)
  - Chapter 12 (Brook's Lemma, Besag Factorization, CAR/SAR Formulations, Ord Determinants, BYM Model)
  - Chapter 12 Exercises (All 16 Exercises, Spectral Graph Theory, Cliff-Ord Normality Proof, Golden Section Profile MLE)
  - Chapter 13 (Kernel Convolutions, Paciorek-Schervish Theorem, Metric MDS Deformation, Fixed Rank Kriging, Pure Python Scaling Benchmark)
  - Chapter 13 Exercises (All 16 Exercises, Woodbury Inversion Proof, Bisquare Smoothness, 2-Point Analytic Evaluation, MDS Double-Centering, Scalability Lab)

---

## Detailed Audit Findings by Severity

### 1. `[CRITICAL]` Issues

#### Finding 1.1: Factorial Simplification Error in Conditional Poisson Uniformity Derivation
- **File & Line Number:** `manuscript/ch11_point_processes.md:105-115`
- **Current Text / Code Snippet:**
  ```latex
  \begin{aligned}
  f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) &= \frac{\mathbb{P}(N(d\mathbf{s}_1)=1, \dots, N(d\mathbf{s}_n)=1, N(W \setminus \bigcup d\mathbf{s}_i)=0)}{\mathbb{P}(N(W) = n)} \\
  &= \frac{n! \, \lambda^n e^{-\lambda |W|} d\mathbf{s}_1 \cdots d\mathbf{s}_n}{\frac{\lambda^n |W|^n}{n!} e^{-\lambda |W|}} \\
  &= \frac{1}{|W|^n}
  \end{aligned}
  ```
- **Exact Failure Mode:**
  Evaluating the fraction as written in the second line:
  $$
  \frac{n! \, \lambda^n e^{-\lambda |W|}}{\frac{\lambda^n |W|^n}{n!} e^{-\lambda |W|}} = \frac{n!}{\frac{1}{n!}} \frac{1}{|W|^n} = \frac{(n!)^2}{|W|^n} \ne \frac{1}{|W|^n}
  $$
  The numerator has an extraneous $n!$. For disjoint infinitesimal subsets $d\mathbf{s}_1, \dots, d\mathbf{s}_n \subset W$, the joint probability that the point process places exactly one point in each $d\mathbf{s}_i$ and zero in the remainder is:
  $$
  \mathbb{P}\left(\bigcap_{i=1}^n \{N(d\mathbf{s}_i)=1\} \cap \{N(W \setminus \bigcup d\mathbf{s}_i)=0\}\right) = \lambda^n e^{-\lambda |W|} d\mathbf{s}_1 \cdots d\mathbf{s}_n
  $$
  Dividing this by $\mathbb{P}(N(W)=n) = \frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}$ yields the conditional density of the *unordered* point configuration:
  $$
  \frac{\lambda^n e^{-\lambda |W|} d\mathbf{s}_1 \cdots d\mathbf{s}_n}{\frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}} = \frac{n!}{|W|^n} d\mathbf{s}_1 \cdots d\mathbf{s}_n
  $$
  Because there are $n!$ equally likely orderings of $n$ distinct points in the coordinate tuple $(\mathbf{s}_1, \dots, \mathbf{s}_n)$, the conditional joint density of the *ordered* coordinates is:
  $$
  f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W)=n) = \frac{1}{n!} \cdot \frac{n!}{|W|^n} = \frac{1}{|W|^n}
  $$
  The manuscript inadvertently placed $n!$ into the numerator while retaining $\frac{1}{n!}$ in the denominator, creating an algebraic contradiction.
- **Proposed Drop-In Replacement:**
  ```latex
  \begin{aligned}
  f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) &= \frac{1}{n!} \frac{\mathbb{P}(N(d\mathbf{s}_1)=1, \dots, N(d\mathbf{s}_n)=1, N(W \setminus \bigcup_{i=1}^n d\mathbf{s}_i)=0) / (d\mathbf{s}_1 \cdots d\mathbf{s}_n)}{\mathbb{P}(N(W) = n)} \\
  &= \frac{1}{n!} \frac{\lambda^n e^{-\lambda |W|}}{\frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}} \\
  &= \frac{1}{n!} \frac{n!}{|W|^n} = \frac{1}{|W|^n}
  \end{aligned}
  ```

---

#### Finding 1.2: Code Bug in Moran's $I$ Variance Implementation
- **File & Line Number:** `manuscript/ch12_lattice_areal_models.md:416-417`
- **Current Text / Code Snippet:**
  ```python
  var_I = (n * ((n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0) - 1.0 * (S1 - 2 * n * S1 + 6 * S0 * S0)) / (
      (n - 1) * (n - 2) * (n - 3) * S0 * S0
  )
```
- **Exact Failure Mode:**
  1. The code attempts to evaluate the Cliff-Ord (1973, 1981) randomization variance formula, but corrupts the algebraic coefficients:
     - It replaces the sample kurtosis $b_2 = m_4 / m_2^2$ with a hardcoded constant `1.0` (which is neither the normal kurtosis $3.0$ nor the empirical kurtosis).
     - The second numerator term is written as `(S1 - 2 * n * S1 + 6 * S0 * S0)` instead of the correct Cliff-Ord polynomial `((n * n - n) * S1 - 2 * n * S2 + 6 * S0 * S0)`.
     - It fails to subtract $\operatorname{E}[I]^2 = \frac{1}{(n - 1)^2}$.
  2. The function docstring and surrounding text state that the code calculates the analytical variance *under the normality assumption*. Under normality, the exact Cliff-Ord variance is:
     $$
     \operatorname{Var}_{\text{norm}}(I) = \frac{n^2 S_1 - n S_2 + 3 S_0^2}{(n - 1)(n + 1) S_0^2} - \frac{1}{(n - 1)^2}
     $$
     The code formula is completely broken, causing incorrect test statistics, $z$-scores, and $p$-values.
- **Proposed Drop-In Replacement:**
  ```python
  # Analytical variance under normality assumption (Cliff and Ord, 1981; see Exercise 12.5)
  numerator = n * n * S1 - n * S2 + 3.0 * S0 * S0
  denominator = (n - 1) * (n + 1) * S0 * S0
  var_I = (numerator / denominator) - (expected_I * expected_I)
```

---

### 2. `[WARNING]` Issues

#### Finding 2.1: Erroneous Cross-Reference for Moran's $I$ Variance Derivation
- **File & Line Number:** `manuscript/ch12_lattice_areal_models.md:112`
- **Current Text / Code Snippet:**
  ```markdown
  The analytical expression for this variance is derived rigorously via the moments of quadratic forms in Exercise 12.2:
  ```
- **Exact Failure Mode:**
  Exercise 12.2 is entitled *"Spectral Properties of Spatial Adjacency Matrices and Eigenvalue Bounds"*, which addresses $\lambda_{\min}, \lambda_{\max}$ and valid autoregressive parameters $\rho$. The derivation of $\operatorname{Var}_{\text{norm}}(I)$ via quadratic form tracial moments $\operatorname{E}[\mathbf{z}'\mathbf{A}\mathbf{z} \mathbf{z}'\mathbf{B}\mathbf{z}]$ is given in **Exercise 12.5** (*"Analytical Derivation of Moran's I Variance Under the Normality Assumption"*).
- **Proposed Drop-In Replacement:**
  ```markdown
  The analytical expression for this variance is derived rigorously via the moments of quadratic forms in Exercise 12.5:
  ```

---

#### Finding 2.2: Missing Condition in Local Moran's Summation Identity
- **File & Line Number:** `manuscript/ch12_lattice_areal_models.md:133-134`
- **Current Text / Code Snippet:**
  ```latex
  A fundamental property connecting local and global indicators is that the sum of local Moran's $I_i$ is proportional to global Moran's $I$:
  $$
  \sum_{i=1}^n I_i = \frac{n}{S_0} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij} z_i z_j}{\sum_{k=1}^n z_k^2} = n \cdot I
  $$
  ```
- **Exact Failure Mode:**
  By definition, Local Moran's $I_i$ is:
  $$
  I_i = \frac{z_i}{m_2} \sum_{j=1}^n w_{ij} z_j = \frac{n z_i}{\sum_{k=1}^n z_k^2} \sum_{j=1}^n w_{ij} z_j
  $$
  Summing across all $i = 1, \dots, n$:
  $$
  \sum_{i=1}^n I_i = \frac{n}{\sum_{k=1}^n z_k^2} \sum_{i=1}^n \sum_{j=1}^n w_{ij} z_i z_j
  $$
  Recall that global Moran's $I$ is:
  $$
  I = \frac{n}{S_0 \sum_{k=1}^n z_k^2} \sum_{i=1}^n \sum_{j=1}^n w_{ij} z_i z_j \implies \sum_{i=1}^n I_i = S_0 \cdot I
  $$
  Therefore, $\sum_{i=1}^n I_i = S_0 \cdot I$. It equals $n \cdot I$ **if and only if** the spatial weight matrix $\mathbf{W}$ is row-standardized, in which case $\sum_{j} w_{ij} = 1$ for all $i$, so $S_0 = \sum_{i=1}^n \sum_{j=1}^n w_{ij} = n$. For general binary or unstandardized weights, $S_0 \ne n$, and the formula as written in the manuscript is incorrect. Furthermore, the intermediate equation step in the manuscript writes $\frac{n}{S_0}$ in front of $\sum I_i$, which contradicts the definition of $I_i$.
- **Proposed Drop-In Replacement:**
  ```latex
  A fundamental property connecting local and global indicators is that the sum of local Moran's $I_i$ is directly proportional to global Moran's $I$:
  $$
  \sum_{i=1}^n I_i = \frac{n}{\sum_{k=1}^n z_k^2} \sum_{i=1}^n \sum_{j=1}^n w_{ij} z_i z_j = S_0 \cdot I
  $$
  When the spatial weight matrix $\mathbf{W}$ is row-standardized such that each row sums to 1, $S_0 = n$, and this relation simplifies to the familiar textbook identity:
  $$
  \sum_{i=1}^n I_i = n \cdot I \quad (\text{for row-standardized } \mathbf{W})
  $$
  ```

---

### 3. `[PEDAGOGICAL GAP]` Issues

#### Finding 3.1: Scale Distinction in Between-Quadrat Autocorrelation vs. Within-Quadrat Clustering
- **File & Line Number:** `manuscript/ch11_exercises.md:195-208`
- **Current Text / Code Snippet:**
  Exercise 11.3, Solution to Part (c):
  ```markdown
  **(c)** Aggregate patterns have VMR > 1 because points cluster within quadrats: some quadrats have high counts while others are empty. The compound Poisson / negative binomial model has variance:
  $$
  \operatorname{Var}(X) = \mu(1 + \rho_{\text{cluster}}) > \mu
  $$
  ```
- **Exact Failure Mode:**
  In Part (b), the exercise proves that positive spatial autocorrelation between quadrats ($\bar{\rho} > 0$) causes the expected sample variance across quadrats to be *depressed*:
  $$
  \operatorname{E}[S^2] = \sigma^2(1 - \bar{\rho}) < \sigma^2
  $$
  In Part (c), the text introduces within-quadrat clustering inflating variance. Without a clear distinction between the two distinct variance mechanisms, students are frequently confused: why does spatial clustering *inflate* the Variance-to-Mean Ratio ($\text{VMR} > 1$) if positive spatial correlation *depresses* sample variance $\operatorname{E}[S^2] < \sigma^2$?
  The resolution is a matter of spatial scale:
  1. *Within-quadrat point clustering* inflates the true underlying population variance $\sigma^2$ far above Poisson mean $\mu$ ($\sigma^2 = \mu + \mu^2 \int \dots \gg \mu$).
  2. *Between-quadrat spatial correlation* causes the sample variance estimator $S^2$ to underestimate $\sigma^2$ by a factor $(1 - \bar{\rho})$, but because $\sigma^2 \gg \mu$, the observed sample $\text{VMR} = S^2 / \bar{X}$ remains well above 1.
- **Proposed Drop-In Replacement:**
  ```markdown
  **(c)** Reconciling Part (b) and Part (c) requires distinguishing between two distinct spatial scales:
  1. *Within-Quadrat Scale (Population Variance Inflation):* Clustered point processes (such as Cox or Neyman-Scott processes) induce strong positive point-to-point correlation within individual quadrats. As shown in Campbell's Theorem, this inflates the true quadrat population variance:
     $$
     \sigma^2 = \operatorname{Var}(N(A)) = \mu + \mu^2 \int_{A}\int_{A} [g(\mathbf{u} - \mathbf{v}) - 1] \, \mathrm{d}\mathbf{u}\mathrm{d}\mathbf{v} > \mu
     $$
  2. *Between-Quadrat Scale (Sample Covariance Bias):* Positive spatial autocorrelation between neighboring quadrats introduces a downward finite-sample estimation bias on the sample variance:
     $$
     \operatorname{E}[S^2] = \sigma^2(1 - \bar{\rho})
     $$
  Because within-quadrat clustering inflates the true variance $\sigma^2$ by an order of magnitude, the observed sample ratio $\text{VMR} = S^2 / \bar{X} \approx (\sigma^2 / \mu)(1 - \bar{\rho})$ remains substantially greater than 1 despite the minor downward deflation $(1 - \bar{\rho})$ from inter-quadrat correlation.
  ```

---

### 4. `[STYLE]` Issues

#### Finding 4.1: Word Duplication in Equation Underbrace Annotation
- **File & Line Number:** `manuscript/ch11_point_processes.md:366`
- **Current Text / Code Snippet:**
  ```latex
  \widehat{K}(r) = \underbrace{\frac{|W|}{n(n-1)}}_{\text{Sample Area / Total Pair Pairs}} \sum_{i=1}^n \sum_{j \ne i}^n \dots
  ```
- **Exact Failure Mode:**
  Typographical duplication: `"Total Pair Pairs"`.
- **Proposed Drop-In Replacement:**
  ```latex
  \widehat{K}(r) = \underbrace{\frac{|W|}{n(n-1)}}_{\text{Sample Area / Total Ordered Pairs}} \sum_{i=1}^n \sum_{j \ne i}^n \dots
  ```

---

## Detailed Clean Confirmations & Mathematical Soundness

The following sections and chapters have undergone rigorous derivation audits and algorithmic verification, with no mathematical or implementation defects detected.

### 1. `manuscript/ch11_point_processes.md`
- **Poisson Postulates & Completely Random Point Processes (CSR):**
  - Section 11.1–11.3: Definition of homogeneous and inhomogeneous Poisson point processes (IPPP), completely independent counts on disjoint Borel sets, and intensity measure $\Lambda(B) = \int_B \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}$ are mathematically standard and complete.
  - Section 11.4: Campbell's Theorems for linear and quadratic functionals:
    $$
    \operatorname{E}\left[\sum_{\mathbf{x} \in \Phi} f(\mathbf{x})\right] = \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
    $$
    and
    $$
    \operatorname{Var}\left[\sum_{\mathbf{x} \in \Phi} f(\mathbf{x})\right] = \int_{\mathbb{R}^d} f(\mathbf{s})^2 \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
    $$
    are derived with correct measure-theoretic foundations.
  - Section 11.5: Reduced second moment measure $\mathcal{K}(r)$, Ripley's $K$-function, pair correlation function $g(r) = \frac{K'(r)}{2\pi r}$ in $\mathbb{R}^2$, and Besag's variance-stabilizing transformation $L(r) = \sqrt{K(r)/\pi} - r$ are completely sound.
  - Section 11.6: Ripley's isotropic edge-correction weight $w(\mathbf{s}_i, \mathbf{s}_j) = \frac{2\pi \|\mathbf{s}_i - \mathbf{s}_j\|}{\text{perimeter of } b(\mathbf{s}_i, \|\mathbf{s}_i - \mathbf{s}_j\|) \cap W}$ is mathematically exact.
  - Section 11.7: Log-Gaussian Cox Processes (LGCP) and cluster processes (Thomas, Matérn) are rigorously formulated.
  - Section 11.8: Dependency-free Python implementation of Lewis-Shedler thinning and empirical Ripley's $K$ function executed cleanly and confirmed spatial clustering vs. CSR.

### 2. `manuscript/ch11_exercises.md`
- **Exercises 11.1–11.16:**
  - Ex 11.1: Rigorous proof that conditional distribution of $n$ points given $N(W)=n$ under an inhomogeneous Poisson process with intensity $\lambda(\mathbf{s})$ is i.i.d. with density $f(\mathbf{s}) = \lambda(\mathbf{s}) / \int_W \lambda(\mathbf{u}) \, \mathrm{d}\mathbf{u}$.
  - Ex 11.4: Campbell's theorem proof via indicator approximation of simple functions.
  - Ex 11.5: Probability Generating Functionals (PGFL) derivation: $G[h] = \exp\left(-\int (1 - h(\mathbf{s}))\lambda(\mathbf{s})\,\mathrm{d}\mathbf{s}\right)$.
  - Ex 11.6: Slivnyak's Theorem proof demonstrating that conditioning on an arbitrary point $\mathbf{s}_0 \in \Phi$ under CSR leaves the remainder of the process $\Phi \setminus \{\mathbf{s}_0\}$ distributed identically as the original unconditioned Poisson process.
  - Ex 11.7: Moments of LGCP: $\operatorname{E}[\Lambda(\mathbf{s})] = \exp(\mu + \sigma^2/2)$ and exact pair correlation $g(r) = \exp(C(r))$.
  - Ex 11.8: Thomas cluster process convolution yielding $g(r) = 1 + \frac{1}{4\pi \sigma_s^2 \lambda_p} \exp\left(-\frac{r^2}{4\sigma_s^2}\right)$ and $K(r) = \pi r^2 + \frac{1}{\lambda_p}(1 - e^{-r^2/(4\sigma_s^2)})$.
  - Ex 11.9–11.11: Pencil-and-paper quad-tree calculations, edge correction geometries, and Clark-Evans nearest neighbor statistics $R = 2\bar{r}\sqrt{\lambda}$.
  - Ex 11.12 & 11.13: Dependency-free Python code for 2D Cox LGCP simulation and Monte Carlo CSR envelope testing executed and confirmed.

### 3. `manuscript/ch12_lattice_areal_models.md`
- **Areal & Lattice Foundations:**
  - Section 12.1: Spatial weight topologies (rook, queen, $k$-nearest neighbors, inverse distance) and row-standardization properties.
  - Section 12.2: Global Moran's $I$ and Geary's $C$ formulations; asymptotic normal moments under the null hypothesis.
  - Section 12.3: Local Indicators of Spatial Association (LISA) decomposition and Moran scatterplot quadrant interpretations.
  - Section 12.4: Brook's Lemma and Besag's Factorization Theorem:
    $$
    \frac{p(\mathbf{y})}{p(\mathbf{0})} = \prod_{i=1}^n \frac{p(y_i \mid y_1, \dots, y_{i-1}, 0, \dots, 0)}{p(0 \mid y_1, \dots, y_{i-1}, 0, \dots, 0)}
    $$
    The expansion guarantees validity and symmetric conditionals without circular reasoning.
  - Section 12.5: Conditionally Autoregressive (CAR) precision matrix $\mathbf{Q}_{\text{CAR}} = \mathbf{D}^{-1}(\mathbf{I} - \alpha \mathbf{W})$ and positive-definiteness condition $\alpha \in (1/\lambda_{\min}, 1/\lambda_{\max})$.
  - Section 12.6: Simultaneously Autoregressive (SAR) error formulation $\mathbf{Y} = \rho \mathbf{W}\mathbf{Y} + \boldsymbol{\epsilon} \implies \mathbf{Q}_{\text{SAR}} = \sigma^{-2}(\mathbf{I} - \rho \mathbf{W})'(\mathbf{I} - \rho \mathbf{W})$.
  - Section 12.7: Intrinsic CAR (ICAR) formulation, graph Laplacian rank deficiency ($n - 1$), and sum-to-zero constraint $\sum y_i = 0$.
  - Section 12.8: Besag-York-Mollié (BYM) convolution model combining ICAR spatial spillover with exchangeable Gaussian white noise.

### 4. `manuscript/ch12_exercises.md`
- **Exercises 12.1–12.16:**
  - Ex 12.2: Gershgorin circle theorem applied to $\mathbf{W}$ proving real eigenvalues bounded in $[-1, 1]$ for symmetric/row-standardized weights.
  - Ex 12.3: Brook's lemma step-by-step expansion.
  - Ex 12.4: Mathematical proof demonstrating that CAR and SAR precision structures are non-identical: $\mathbf{Q}_{\text{SAR}}$ introduces second-order neighbors ($w_{ik} w_{jk}$) absent from $\mathbf{Q}_{\text{CAR}}$.
  - Ex 12.5: Rigorous derivation of Moran's $I$ variance under normality using trace identities for quadratic forms $\operatorname{tr}(\mathbf{W})$ and $\operatorname{tr}(\mathbf{W}^2)$.
  - Ex 12.6: Ord log-determinant identity $\log|\mathbf{I} - \rho \mathbf{W}| = \sum_{i=1}^n \log(1 - \rho \lambda_i)$ enabling $\mathcal{O}(n)$ evaluation of the SAR profile likelihood.
  - Ex 12.7: ICAR graph Laplacian rank $n-1$ null space proof via graph connectedness.
  - Ex 12.8: BYM variance decomposition and Leroux prior formulation.
  - Ex 12.9–12.11: Pencil-and-paper $3 \times 3$ CAR joint distribution, $4 \times 4$ SAR log-likelihood, and Moran quadrant classifications.
  - Ex 12.12: Pure Python Golden Section Search for SAR profile MLE executed cleanly (optimal $\widehat{\rho} \approx 0.5487$).
  - Ex 12.13: Pure Python CAR spatial autocorrelation decay simulation executed and verified.

### 5. `manuscript/ch13_nonstationary_covariance.md`
- **Non-Stationary & Scalable Covariances:**
  - Section 13.1: Kernel convolution framework $Z(\mathbf{s}) = \int k_{\mathbf{s}}(\mathbf{u}) \, \mathrm{d}W(\mathbf{u})$; proof of positive-definiteness for arbitrary spatially varying kernels.
  - Section 13.2: Paciorek-Schervish Theorem:
    $$
    C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j) \frac{|\boldsymbol{\Sigma}_i|^{1/4} |\boldsymbol{\Sigma}_j|^{1/4}}{\left|\frac{\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j}{2}\right|^{1/2}} \mathcal{M}_\nu\left(\sqrt{Q_{ij}}\right)
    $$
    Derivation of determinant prefactor cancellation and local Mahalanobis distance metric $Q_{ij} = (\mathbf{s}_i - \mathbf{s}_j)' \left(\frac{\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j}{2}\right)^{-1} (\mathbf{s}_i - \mathbf{s}_j)$ is completely sound.
  - Section 13.3: Sampson-Guttorp non-metric and metric MDS space deformation $\mathbf{s} \in \mathcal{G} \mapsto \mathbf{d} \in \mathcal{D}$ via thin-plate splines.
  - Section 13.4: Fixed Rank Kriging (FRK) spatial model $Z(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \mathbf{S}(\mathbf{s})\boldsymbol{\eta} + \epsilon(\mathbf{s})$; Sherman-Morrison-Woodbury inversion identity:
    $$
    \boldsymbol{\Sigma}^{-1} = \frac{1}{\sigma_\epsilon^2}\mathbf{I}_n - \frac{1}{\sigma_\epsilon^4}\mathbf{S}\left(\mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}\right)^{-1}\mathbf{S}'
    $$
    reducing computational complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n r^2 + r^3)$ with $r \ll n$.
  - Section 13.5: Pure Python implementation benchmarking Exact Kriging vs. FRK executed and verified.

### 6. `manuscript/ch13_exercises.md`
- **Exercises 13.1–13.16:**
  - Ex 13.1: Complete proof of positive-definiteness for process convolutions using Cauchy-Schwarz and inner product integrals.
  - Ex 13.2: Analytical integration of the convolution of two Gaussian kernels $\mathcal{N}(\mathbf{s}_1, \boldsymbol{\Sigma}_1)$ and $\mathcal{N}(\mathbf{s}_2, \boldsymbol{\Sigma}_2)$, proving the exact Paciorek-Schervish prefactor.
  - Ex 13.3: Diffeomorphic condition and Jacobian determinant non-degeneracy $\det(J_{\mathbf{d}}(\mathbf{s})) > 0$ to prevent physical space folding.
  - Ex 13.4: Spectral representation of non-stationary fields via orthogonal increments and evolutionary spectra.
  - Ex 13.7: Direct verification of the Woodbury identity $(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})\mathbf{B} = \mathbf{I}_n$ and memory-efficient matrix-free evaluation of kriging weights.
  - Ex 13.8: Mathematical proof that the bisquare basis function $(1 - r^2/R^2)^2$ is $C^1$ continuously differentiable everywhere with jump discontinuity in the second radial derivative $\Delta = 8/R^2$.
  - Ex 13.9: Pencil-and-paper evaluation of Paciorek-Schervish covariance between two points: $|\boldsymbol{\Sigma}_1|=1, |\boldsymbol{\Sigma}_2|=4, |\boldsymbol{\Sigma}_{\text{avg}}|=2.5$, prefactor $P = \sqrt{0.8} \approx 0.8944$, $Q_{12} = 0.4$, covariance $C = 1.4646$ (verified exact).
  - Ex 13.10: Pencil-and-paper classical MDS double-centering on a $3 \times 3$ right-triangle dissimilarity matrix ($3, 4, 5$), proving double-centered matrix $\mathbf{B}$ has rank 2, zero row/column sums, and exact Euclidean coordinates recovery.
  - Ex 13.11: Pencil-and-paper $2 \times 2$ FRK system ($M = 5.515625$, $\widehat{\eta} \approx 4.034$, $\widehat{Z}(0.5, 0) \approx 3.5455$, $\sigma_{\text{FRK}}^2 \approx 0.3901$).
  - Ex 13.12: Pure Python Paciorek-Schervish 1D non-stationary simulator executed and verified (West roughness = 0.1046, East roughness = 0.0422, ratio = 2.48x).
  - Ex 13.13: Diagnostic scalability laboratory executed and verified across $n \in \{100, 250, 500\}$, demonstrating $\mathcal{O}(n r^2)$ computational speedup.
  - Ex 13.14–13.16: In-depth conceptual analyses of stationary over-smoothing in alpine topography, basis rank selection trade-offs, and decisive selection criteria across real-world domains.

---

## Actionable Patch Instructions

To facilitate immediate editorial integration into the manuscript, the recommended patches are summarized below:

```diff
--- manuscript/ch11_point_processes.md
+++ manuscript/ch11_point_processes.md
@@ -107,3 +107,3 @@
 \begin{aligned}
-f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) &= \frac{\mathbb{P}(N(d\mathbf{s}_1)=1, \dots, N(d\mathbf{s}_n)=1, N(W \setminus \bigcup d\mathbf{s}_i)=0)}{\mathbb{P}(N(W) = n)} \\
-&= \frac{n! \, \lambda^n e^{-\lambda |W|} d\mathbf{s}_1 \cdots d\mathbf{s}_n}{\frac{\lambda^n |W|^n}{n!} e^{-\lambda |W|}} \\
-&= \frac{1}{|W|^n}
+f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) &= \frac{1}{n!} \frac{\mathbb{P}(N(d\mathbf{s}_1)=1, \dots, N(d\mathbf{s}_n)=1, N(W \setminus \bigcup_{i=1}^n d\mathbf{s}_i)=0) / (d\mathbf{s}_1 \cdots d\mathbf{s}_n)}{\mathbb{P}(N(W) = n)} \\
+&= \frac{1}{n!} \frac{\lambda^n e^{-\lambda |W|}}{\frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}} \\
+&= \frac{1}{n!} \frac{n!}{|W|^n} = \frac{1}{|W|^n}
 \end{aligned}
@@ -366,3 +366,3 @@
-\underbrace{\frac{|W|}{n(n-1)}}_{\text{Sample Area / Total Pair Pairs}}
+\underbrace{\frac{|W|}{n(n-1)}}_{\text{Sample Area / Total Ordered Pairs}}

--- manuscript/ch12_lattice_areal_models.md
+++ manuscript/ch12_lattice_areal_models.md
@@ -112,1 +112,1 @@
-The analytical expression for this variance is derived rigorously via the moments of quadratic forms in Exercise 12.2:
+The analytical expression for this variance is derived rigorously via the moments of quadratic forms in Exercise 12.5:
@@ -134,3 +134,3 @@
-$$
-\sum_{i=1}^n I_i = \frac{n}{S_0} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij} z_i z_j}{\sum_{k=1}^n z_k^2} = n \cdot I
-$$
+$$
+\sum_{i=1}^n I_i = \frac{n}{\sum_{k=1}^n z_k^2} \sum_{i=1}^n \sum_{j=1}^n w_{ij} z_i z_j = S_0 \cdot I
+$$
+When the spatial weight matrix $\mathbf{W}$ is row-standardized ($S_0 = n$), this simplifies to $\sum_{i=1}^n I_i = n \cdot I$.
@@ -416,3 +416,3 @@
-    var_I = (n * ((n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0) - 1.0 * (S1 - 2 * n * S1 + 6 * S0 * S0)) / (
-        (n - 1) * (n - 2) * (n - 3) * S0 * S0
-    )
+    # Analytical variance under normality assumption (Cliff and Ord, 1981; see Exercise 12.5)
+    num_var = n * n * S1 - n * S2 + 3.0 * S0 * S0
+    den_var = (n - 1) * (n + 1) * S0 * S0
+    var_I = (num_var / den_var) - (expected_I * expected_I)
```

---

## Conclusion
The mathematical architecture and pedagogical depth of Batch 5.5 are outstanding. The core proofs—such as Campbell's theorems, Slivnyak's invariance, Brook's lemma factorization, Ord's log-determinant identity, Paciorek-Schervish non-stationary kernels, and the Sherman-Morrison-Woodbury Fixed Rank Kriging reduction—are constructed with utmost graduate-level rigor. Applying the four targeted mathematical and code corrections detailed in this report will bring the batch to 100% publication readiness.
