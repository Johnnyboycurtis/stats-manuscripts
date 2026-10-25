# Phase 5 Reviewer Audit Report: Batch 5.1
**Monograph:** *Theoretical Foundations of Spatial Statistics and Geostatistics*  
**Auditor:** Reviewer Subagent 1 (Mathematical Rigor, Spectral Theory & Notation Audit)  
**Date:** October 2026  
**Scope:** Batch 5.1 (`manuscript/ch01_foundations.md`, `manuscript/ch02_covariance_variogram.md`, `manuscript/app_a.md`, `manuscript/app_b.md`, `manuscript/ch01_exercises.md`, `manuscript/ch02_exercises.md`)

---

## Executive Summary

This comprehensive audit evaluates Batch 5.1 against the rigorous standards of a premier graduate-level mathematical monograph. The audit examined mathematical derivations, measure-theoretic foundations, spectral representations (Bochner, Hankel, Schoenberg), notation uniformity (Schabenberger & Gotway standards), pure Python algorithm implementations, and pedagogical continuity across 4,770 manuscript lines.

### Summary Metrics
| Audit Category | Count | Status / Summary |
| :--- | :---: | :--- |
| **`[CRITICAL]` Defects** | **2** | Microergodic parameter ratio ($\mathbb{R}^3$) in Ch. 1; Erroneous $4\pi^2$ frequency scaling in Matérn spectral density in Ch. 2. |
| **`[WARNING]` Discrepancies** | **5** | Notation drift in Appendix A & B ($\mathbf{s}$ vs $s$, $\mathbf{h}'$ vs $h^T$, $\boldsymbol{\omega}'\mathbf{h}$ vs $\langle \omega, h \rangle$); Trace-only pseudo-check in CND algorithm; Matérn pure Python small-$z$ singularity; Ambiguous continuity condition. |
| **`[PEDAGOGICAL GAP]`** | **2** | Incomplete draft ellipses in Spherical model proof (App. B); Sub-quadratic variogram growth link in Ch. 2 Exercise 2.3. |
| **`[STYLE]` Minor Fixes** | **1** | Mismatched subscript parameter in Schwartz space mollifier (App. A). |
| **`[CLEAN]` Validations** | **32** | All 16 exercises in Ch. 1, all 16 exercises in Ch. 2, core Bochner/Schoenberg theorems, and all pure Python standard library scripts. |

### Code Compilation & Verification Status
Across all 6 files, 11 distinct Python code blocks were extracted and tested in a clean Python 3.12 environment with strictly zero third-party packages (`numpy`, `scipy` omitted):
- `manuscript/ch02_covariance_variogram.md`: 5 algorithms / labs $\to$ **All compiled and executed successfully.**
- `manuscript/ch01_exercises.md`: 3 standalone scripts $\to$ **All compiled and executed successfully.**
- `manuscript/ch02_exercises.md`: 3 standalone scripts $\to$ **All compiled and executed successfully.**

---

## Detailed Audit Findings by Severity

```
================================================================================
                               CRITICAL DEFECTS
================================================================================
```

### Finding 5.1-C01: Erroneous Microergodic Parameter Ratio in $\mathbb{R}^3$
- **File:** `manuscript/ch01_foundations.md`
- **Line:** 418
- **Category:** `[CRITICAL]`
- **Severity Score:** 10/10 (Direct theoretical error contradicting asymptotic literature and Chapter 4)
- **Current Text:**
  ```markdown
  Under infill asymptotics in $\mathbb{R}^3$, neither the spatial variance $\sigma^2$ nor the spatial range $\phi$ can be estimated consistently from a single realization in a bounded domain. Only the ratio $\sigma^2/\phi^3$ is microergodic and consistently estimable (Zhang, 2004).
  ```
- **Failure Mode & Mathematical Rationale:**
  Under infill asymptotics in spatial dimension $d$, two Gaussian measures with Matérn covariances $(\sigma_1^2, \phi_1)$ and $(\sigma_2^2, \phi_2)$ on a bounded domain $D \subset \mathbb{R}^d$ are mutually absolutely continuous (equivalent) if and only if their spectral densities have identical tail behavior as $\|\boldsymbol{\omega}\| \to \infty$.
  For an Exponential covariance model ($\nu = 0.5$) in $\mathbb{R}^d$, the spectral density behaves asymptotically as:
  $$
  f(\boldsymbol{\omega}) \sim \frac{\sigma^2 \phi^d}{\pi^{(d+1)/2}} \frac{\Gamma((d+1)/2)}{(1 + \phi^2 \|\boldsymbol{\omega}\|^2)^{(d+1)/2}} \sim c_d \frac{\sigma^2 \phi^d}{\phi^{d+1} \|\boldsymbol{\omega}\|^{d+1}} = c_d \frac{\sigma^2}{\phi} \|\boldsymbol{\omega}\|^{-(d+1)}
  $$
  Specifically in $\mathbb{R}^3$ ($d = 3$), the asymptotic tail is:
  $$
  f(\boldsymbol{\omega}) \sim \frac{\sigma^2 \phi^3}{\pi^2} \frac{1}{\phi^4 \|\boldsymbol{\omega}\|^4} = \frac{1}{\pi^2} \frac{\sigma^2}{\phi} \|\boldsymbol{\omega}\|^{-4}
  $$
  Notice that $\phi^3 / \phi^4 = \phi^{-1}$. Therefore, the ratio governing measure equivalence and consistent infill estimability is **$\sigma^2 / \phi$**, NOT $\sigma^2 / \phi^3$!
  For general Matérn smoothness $\nu$, the microergodic parameter is $\sigma^2 / \phi^{2\nu}$ (which for $\nu = 0.5$ equals $\sigma^2 / \phi$).
  Writing $\sigma^2 / \phi^3$ contradicts Zhang (2004), contradicts `manuscript/ch01_exercises.md:979` ($\sigma^2/\phi$), contradicts `manuscript/ch04_parameter_estimation.md:705`, and contradicts `README.md:75`.
- **Proposed Drop-In Replacement:**
  ```markdown
  Under infill asymptotics in $\mathbb{R}^3$, neither the spatial variance $\sigma^2$ nor the spatial range $\phi$ can be estimated consistently from a single realization in a bounded domain. Only the ratio $\sigma^2/\phi$ (or more generally $\sigma^2/\phi^{2\nu}$ for the Matérn family with fixed smoothness $\nu$) is microergodic and consistently estimable (Zhang, 2004).
  ```

---

### Finding 5.1-C02: Erroneous $4\pi^2$ Frequency Scaling in Matérn Spectral Density
- **File:** `manuscript/ch02_covariance_variogram.md`
- **Line:** 548
- **Category:** `[CRITICAL]`
- **Severity Score:** 9/10 (Internal contradiction between Chapter 2 and Appendix B)
- **Current Text:**
  ```latex
  $$f(\rho) = \frac{\sigma^2 \phi^d \Gamma(\nu + d/2)}{\pi^{d/2} \Gamma(\nu) (2\nu)^\nu} \left(\frac{2\nu}{\phi^2} + 4\pi^2 \rho^2\right)^{-\left(\nu + \frac{d}{2}\right)}$$
  ```
- **Failure Mode & Mathematical Rationale:**
  In Chapter 2 (e.g. Lines 240, 260, 388), the forward and inverse Fourier transform pair is explicitly defined in angular/radian spatial frequencies $\boldsymbol{\omega}$ (kernel $e^{-i \boldsymbol{\omega}'\mathbf{h}}$ without $2\pi$ in the exponent).
  Under this angular frequency convention, the Fourier transform of the Matérn covariance yields:
  $$
  f(\boldsymbol{\omega}) = \frac{\sigma^2 \phi^d \Gamma(\nu + d/2)}{\pi^{d/2} \Gamma(\nu) (2\nu)^\nu} \left(\frac{2\nu}{\phi^2} + \|\boldsymbol{\omega}\|^2\right)^{-\left(\nu + \frac{d}{2}\right)}
  $$
  This exact formula is correctly and rigorously derived in `manuscript/app_b.md:220`.
  The term $4\pi^2 \rho^2$ only arises if the Fourier transform is defined in cyclical frequencies $\mathbf{k} \in \mathbb{R}^d$ where $\boldsymbol{\omega} = 2\pi \mathbf{k}$ and the Fourier kernel is $e^{-i 2\pi \mathbf{k}'\mathbf{h}}$. Because $\rho = \|\boldsymbol{\omega}\|$ is defined in Chapter 2 as radian frequency, multiplying $\rho^2$ by $4\pi^2$ artificially scales the frequency by $4\pi^2 \approx 39.48$, causing the integral $\int_{\mathbb{R}^d} f(\boldsymbol{\omega})\,d\boldsymbol{\omega}$ to fail to integrate to $\sigma^2$ and directly contradicting Appendix B.
- **Proposed Drop-In Replacement:**
  ```latex
  $$f(\rho) = \frac{\sigma^2 \phi^d \Gamma(\nu + d/2)}{\pi^{d/2} \Gamma(\nu) (2\nu)^\nu} \left(\frac{2\nu}{\phi^2} + \rho^2\right)^{-\left(\nu + \frac{d}{2}\right)}$$
  ```

---

```
================================================================================
                            WARNINGS & DISCREPANCIES
================================================================================
```

### Finding 5.1-W01: Systematic Semantic Notation Drift in Appendix A & Appendix B
- **File:** `manuscript/app_a.md` & `manuscript/app_b.md`
- **Lines:** Pervasive throughout `app_a.md` (lines 10-470) and `app_b.md` (lines 15-560)
- **Category:** `[WARNING]`
- **Severity Score:** 7/10 (Notation inconsistency against Master Notation Index)
- **Current Text Examples:**
  - `app_a.md:14`: `$s, t \in \mathbb{R}^d$` instead of `$\mathbf{s}, \mathbf{t} \in \mathbb{R}^d$`
  - `app_a.md:17`: `$\langle \omega, s \rangle$` instead of `$\boldsymbol{\omega}' \mathbf{s}$`
  - `app_a.md:25`: `$h^T \mathbf{A} h$` and `$c^T \mathbf{V} c$` instead of `$\mathbf{h}' \mathbf{A} \mathbf{h}$` and `$\mathbf{c}' \mathbf{V} \mathbf{c}$`
  - `app_b.md:24`: `$\sum_{i=1}^n \sum_{j=1}^n c_i c_j C(x_i - x_j) \ge 0$` (using $x_i$ instead of $\mathbf{s}_i$)
  - `app_b.md:73`: `$\langle \omega, x \rangle$` instead of `$\boldsymbol{\omega}' \mathbf{s}$`
  - `app_b.md:118`: `$\mathbf{c}^T \boldsymbol{\Gamma} \mathbf{c} \le 0$` instead of `$\mathbf{c}' \boldsymbol{\Gamma} \mathbf{c} \le 0$`
- **Failure Mode & Mathematical Rationale:**
  The project Master Notation Index (`manuscript/notation_index.md`) mandates Schabenberger & Gotway (2005) conventions:
  1. Spatial coordinate vectors in $\mathbb{R}^d$ must be bold lowercase: $\mathbf{s}, \mathbf{s}_0, \mathbf{h}$.
  2. Spatial frequency vectors in $\mathbb{R}^d$ must be bold Greek: $\boldsymbol{\omega}$.
  3. Matrix and vector transposition must use prime notation: $\mathbf{c}', \mathbf{A}', \mathbf{h}'$, avoiding uppercase transpose superscripts ($\mathbf{c}^T$).
  4. Inner products in $\mathbb{R}^d$ must be written as matrix vector products: $\boldsymbol{\omega}'\mathbf{s}$ or $\boldsymbol{\omega}'\mathbf{h}$, rather than functional Hilbert inner product brackets $\langle \omega, s \rangle$.
  Chapters 1, 2, and both Exercise files follow this standard strictly. Appendices A and B exhibit stylistic drift from classical pure functional analysis texts.
- **Proposed Drop-In Replacement:**
  Systematically replace across `app_a.md` and `app_b.md`:
  - $s \to \mathbf{s}$, $t \to \mathbf{t}$, $h \to \mathbf{h}$, $x_i \to \mathbf{s}_i$.
  - $\omega \to \boldsymbol{\omega}$.
  - $\langle \omega, s \rangle \to \boldsymbol{\omega}'\mathbf{s}$.
  - $h^T \mathbf{A} h \to \mathbf{h}' \mathbf{A} \mathbf{h}$ and $\mathbf{c}^T \to \mathbf{c}'$.

---

### Finding 5.1-W02: Flawed Trace-Only Eigenvalue Heuristic in Algorithm 2.2
- **File:** `manuscript/ch02_covariance_variogram.md`
- **Lines:** 488–492
- **Category:** `[WARNING]`
- **Severity Score:** 8/10 (Algorithmic logic flaw: necessary condition presented as sufficient)
- **Current Text:**
  ```python
  # 3. Spectral decomposition check
  # In production, compute eigenvalues of Q via QR algorithm.
  # For valid CND, Q must be positive semi-definite (lambda_min >= -tol).
  trace_val = sum(Q[i][i] for i in range(n))
  is_cnd = (trace_val >= -1e-9)
  ```
- **Failure Mode & Mathematical Rationale:**
  The trace of a matrix is the sum of its eigenvalues: $\operatorname{Tr}(\mathbf{Q}) = \sum_{i=1}^n \lambda_i$.
  Testing whether $\operatorname{Tr}(\mathbf{Q}) \ge 0$ is a **necessary** condition for positive semi-definiteness, but it is **not sufficient**.
  For example, a matrix with eigenvalues $\lambda = (+10, -4, -3)$ has trace $\operatorname{Tr} = +3 > 0$, yet it has two negative eigenvalues and is strictly indefinite!
  If an invalid variogram produces $\mathbf{Q}$ with one large positive eigenvalue and several negative eigenvalues, `trace_val >= -1e-9` will return `is_cnd = True`, generating false negatives for invalid variograms.
  In contrast, `ch02_exercises.md` (Lines 806–864) provides a complete, 35-line Jacobi eigenvalue solver in pure standard-library Python that accurately finds all eigenvalues.
- **Proposed Drop-In Replacement:**
  Replace the trace calculation with either:
  1. A Gershgorin circle lower bound or pure Python 2x2/3x3 characteristic polynomial check; or
  2. Embed the Jacobi rotation solver or Cholesky LDL factorization check:
  ```python
  def is_positive_semidefinite_ldl(A, tol=1e-10):
      """Verifies positive semi-definiteness via LDL^T factorization."""
      n = len(A)
      L = [[0.0] * n for _ in range(n)]
      D = [0.0] * n
      for i in range(n):
          L[i][i] = 1.0
          for j in range(i):
              s = sum(L[i][k] * L[j][k] * D[k] for k in range(j))
              if abs(D[j]) > 1e-14:
                  L[i][j] = (A[i][j] - s) / D[j]
          s = sum(L[i][k] ** 2 * D[k] for k in range(i))
          D[i] = A[i][i] - s
          if D[i] < -tol:
              return False
      return True

  is_cnd = is_positive_semidefinite_ldl(Q)
  ```

---

### Finding 5.1-W03: Asymptotic Bessel Approximation Singularity in Algorithm 2.3
- **File:** `manuscript/ch02_covariance_variogram.md`
- **Lines:** 673–680
- **Category:** `[WARNING]`
- **Severity Score:** 6/10 (Numerical instability at origin)
- **Current Text:**
  ```python
  else:
      # General nu approximation via asymptotic expansion for Bessel K_nu(z)
      # For production, integrate or use specialized polynomial fits.
      k_nu = math.sqrt(math.pi / (2.0 * z)) * math.exp(-z)
      gamma_nu = math.gamma(nu)
      cov = (sigma2 / (2.0 ** (nu - 1.0) * gamma_nu)) * (z ** nu) * k_nu
  ```
- **Failure Mode & Mathematical Rationale:**
  The asymptotic expansion $K_\nu(z) \sim \sqrt{\frac{\pi}{2z}} e^{-z}$ is valid only for large arguments ($z \gg 1$).
  As $z \to 0$ (i.e. lag distance $h \to 0$), the factor $\sqrt{\pi / (2z)}$ exhibits a $1/\sqrt{z}$ singularity.
  When multiplied by $z^\nu$, the expression scales as $z^{\nu - 0.5}$:
  - If $\nu < 0.5$ (e.g. $\nu = 0.25$): $z^{-0.25} \to \infty$ as $h \to 0$, causing covariance to diverge to infinity!
  - If $\nu > 0.5$ (e.g. $\nu = 1.0$ or $2.0$): $z^{\nu - 0.5} \to 0$ as $h \to 0$, falsely driving $C(0) \to 0$ instead of $C(0) = \sigma^2$!
  While the algorithm handles the discrete cases $\nu = 0.5, 1.5, 2.5$ explicitly, if a user queries any non-half-integer $\nu$ at small $h$, the algorithm yields catastrophic numerical errors.
- **Proposed Drop-In Replacement:**
  Provide a small-$z$ series expansion branch for $z < 0.5$:
  ```python
  else:
      # Series expansion for small z; asymptotic for large z
      gamma_nu = math.gamma(nu)
      if z < 1e-4:
          cov = sigma2 * (1.0 - (z ** (2.0 * min(nu, 1.0))) / (2.0 * nu))
      elif z < 2.0:
          # Power series: z^nu * K_nu(z) ~ 2^(nu-1)*Gamma(nu) - 2^(nu-3)*Gamma(nu-1)*z^2 + ...
          term0 = 2.0 ** (nu - 1.0) * gamma_nu
          term1 = 2.0 ** (nu - 3.0) * math.gamma(nu - 1.0) * (z ** 2) if nu > 1.0 else 0.0
          cov = sigma2 * (1.0 - term1 / term0)
      else:
          k_nu = math.sqrt(math.pi / (2.0 * z)) * math.exp(-z)
          cov = (sigma2 / (2.0 ** (nu - 1.0) * gamma_nu)) * (z ** nu) * k_nu
  ```

---

### Finding 5.1-W04: Ambiguous Mean-Square Continuity Formulation
- **File:** `manuscript/ch01_foundations.md`
- **Line:** 298
- **Category:** `[WARNING]`
- **Severity Score:** 5/10 (Mathematical rigor & precision)
- **Current Text:**
  ```markdown
  $\lim_{\mathbf{h} \to \mathbf{0}} \operatorname{E}[(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s}))^2] = 0 \iff \lim_{\mathbf{s} \to \mathbf{s}_0} C(\mathbf{s}, \mathbf{s}_0) = C(\mathbf{s}_0, \mathbf{s}_0) = C(\mathbf{s}, \mathbf{s})$.
  ```
- **Failure Mode & Mathematical Rationale:**
  For general (possibly non-stationary) random fields, the variance $C(\mathbf{s}, \mathbf{s}) = \operatorname{Var}(Z(\mathbf{s}))$ is not a constant; writing $C(\mathbf{s}_0, \mathbf{s}_0) = C(\mathbf{s}, \mathbf{s})$ incorrectly equates variances at two different spatial points $\mathbf{s} \ne \mathbf{s}_0$.
  The exact necessary and sufficient condition for mean-square continuity at $\mathbf{s}_0$ is:
  $$
  \lim_{\mathbf{s}_1, \mathbf{s}_2 \to \mathbf{s}_0} C(\mathbf{s}_1, \mathbf{s}_2) = C(\mathbf{s}_0, \mathbf{s}_0)
  $$
  Under second-order stationarity, where $C(\mathbf{s}_1, \mathbf{s}_2) = C(\mathbf{s}_1 - \mathbf{s}_2)$, this simplifies to the continuity of the stationary covariance function at the origin: $\lim_{\mathbf{h} \to \mathbf{0}} C(\mathbf{h}) = C(\mathbf{0})$.
- **Proposed Drop-In Replacement:**
  ```markdown
  $\lim_{\mathbf{h} \to \mathbf{0}} \operatorname{E}[(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s}))^2] = 0 \iff \lim_{\mathbf{s}_1 \to \mathbf{s}, \, \mathbf{s}_2 \to \mathbf{s}} C(\mathbf{s}_1, \mathbf{s}_2) = C(\mathbf{s}, \mathbf{s})$, which under second-order stationarity simplifies to continuity of the covariance function at the origin: $\lim_{\mathbf{h} \to \mathbf{0}} C(\mathbf{h}) = C(\mathbf{0})$.
  ```

---

### Finding 5.1-W05: Incomplete Deduction for Power Variograms in Chapter 2 Exercise
- **File:** `manuscript/ch02_covariance_variogram.md`
- **Lines:** 1082–1085
- **Category:** `[WARNING]`
- **Severity Score:** 4/10 (Deductive completeness)
- **Current Text:**
  ```markdown
  Since $Q(3) = 12 > 0$, the quadratic form is strictly positive, violating the condition for conditional negative definiteness. Thus, $\gamma(h) = h^\alpha$ is not conditionally negative definite for any $\alpha > 2$.
  ```
- **Failure Mode & Mathematical Rationale:**
  Demonstrating that $Q(3) > 0$ for the specific value $\alpha = 3$ proves that $\alpha = 3$ is invalid. Concluding that it fails for *all* $\alpha > 2$ is an unproven leap unless either:
  1. It is shown that the derivative $\frac{d}{d\alpha} Q(\alpha) > 0$ for $\alpha \ge 2$, with $Q(2) = 0$; or
  2. The proof cites the fundamental sub-quadratic asymptotic bound (Matheron 1971 / Theorem 2.3) that $\lim_{h \to \infty} \gamma(h)/h^2 = 0$.
  (Note: `ch02_exercises.md` handles this rigorously in Exercise 2.7 and 2.16; the main text in `ch02_covariance_variogram.md` should match this rigor).
- **Proposed Drop-In Replacement:**
  ```markdown
  Since $Q(3) = 12 > 0$, the quadratic form is strictly positive, violating conditional negative definiteness for $\alpha = 3$. More generally, because $Q(2) = 0$ and $Q'(\alpha) = 12 \cdot 2^\alpha \ln 2 - 2 \cdot 3^\alpha \ln 3 > 0$ for all $\alpha > 2$, $Q(\alpha) > 0$ for all $\alpha > 2$. Combined with the sub-quadratic growth bound (Theorem 2.3, requiring $\lim_{h \to \infty} \gamma(h)/h^2 = 0$), power variograms $\gamma(h) = b h^\alpha$ are mathematically valid if and only if $\alpha \in [0, 2)$.
  ```

---

```
================================================================================
                               PEDAGOGICAL GAPS
================================================================================
```

### Finding 5.1-P01: Unfinished Ellipses in Higher-Dimensional Spherical Model Proof
- **File:** `manuscript/app_b.md`
- **Lines:** 199, 203–204
- **Category:** `[PEDAGOGICAL GAP]`
- **Severity Score:** 7/10 (Drafting placeholder left in proof)
- **Current Text:**
  ```latex
  $$36(3 - 0.960) - 6(6)(-0.279) - 6(0.960) + 6 \dots$$
  ```
  and
  ```latex
  $$\psi \in \Phi_d \iff \int_0^\infty r^2 C_{\text{sph}}''(r) \dots$$
  ```
- **Failure Mode & Mathematical Rationale:**
  In Section B.2.2 of Appendix B, the derivation proving that the 3D Spherical model is invalid in dimensions $d \ge 4$ cuts off mid-expression with literal dots `\dots`.
  This is a significant pedagogical gap in an otherwise rigorous appendix. Graduate students attempting to verify the Hankel transform or Matheron's Montée turning bands criteria will find the proof unfinished.
- **Proposed Drop-In Replacement:**
  Complete the evaluation of the 4D Hankel transform or Montée integral explicitly:
  ```latex
  Using the 4D Hankel transform:
  $$f_4(\rho) = \frac{\sigma^2 \phi^4}{4\pi^2} \frac{1}{(\rho \phi)^3} \left[ 8 J_2(\rho \phi) - \rho \phi J_1(\rho \phi) \right]$$
  Evaluating $g(x) = 8 J_2(x) - x J_1(x)$ at $x = \rho \phi = 6.0$:
  $$g(6.0) = 8(-0.2429) - 6.0(-0.2767) = -1.9432 + 1.6602 = -0.2830 < 0$$
  Since $f_4(\rho) < 0$ for frequencies near $\rho = 6.0/\phi$, Bochner's theorem is violated. Hence, the 3D Spherical covariance function is strictly invalid in $\mathbb{R}^4$ and all higher dimensions ($d \ge 4$).
  ```

---

```
================================================================================
                                 STYLE ISSUES
================================================================================
```

### Finding 5.1-S01: Subscript Variable Discrepancy in Schwartz Mollifier
- **File:** `manuscript/app_a.md`
- **Line:** 267
- **Category:** `[STYLE]`
- **Current Text:**
  ```latex
  $$\psi_T(x) = \frac{1}{\delta} \psi\left(\frac{x}{\delta}\right)$$
  ```
- **Failure Mode & Rationale:**
  The mollifying kernel is defined with subscript $T$, but the argument scaling uses $\delta$.
- **Proposed Drop-In Replacement:**
  ```latex
  $$\psi_\delta(x) = \frac{1}{\delta^d} \psi\left(\frac{x}{\delta}\right)$$
  ```

---

```
================================================================================
                               CLEAN VALIDATIONS
================================================================================
```

### Finding 5.1-V01: Verification of Chapter 1 Exercises (`ch01_exercises.md`)
- **Category:** `[CLEAN]`
- **Status:** **Exemplary Graduate Monograph Quality.**
- **Highlights:**
  - Complete 16-exercise coverage spanning Tiers 0 to IV.
  - Exercise 1.5: Flawless measure-theoretic proof of Kolmogorov consistency theorem with finite-dimensional cylinder sets.
  - Exercise 1.6: Exact derivation of effective sample size $N_{\text{eff}} = n / (1 + (n-1)\rho)$.
  - Exercise 1.8: Rigorous Feldman-Hájek dichotomy derivation using infinite product likelihood ratios and Kakutani's theorem.
  - Exercise 1.12–1.13: Dependency-free Python algorithms for Cholesky spatial factorization and Infill vs. Increasing-domain empirical estimators.
  - Notation adheres strictly to Schabenberger & Gotway standards throughout.

### Finding 5.1-V02: Verification of Chapter 2 Exercises (`ch02_exercises.md`)
- **Category:** `[CLEAN]`
- **Status:** **Exemplary Graduate Monograph Quality.**
- **Highlights:**
  - Exercise 2.1: Direct harmonic decomposition proof of cosine kernel semi-definiteness and rank-2 deficiency for $n > 2$.
  - Exercise 2.3: Closed-form demonstration of negative kriging variance ($\sigma_{\text{OK}}^2 = -0.2148$) under non-CND cubic variogram $\gamma(h) = h^3$.
  - Exercise 2.4: Bessel integral evaluation of the Matérn integral scale $I_{\text{mat}}(\nu, \phi) = \frac{\phi \sqrt{\pi}\Gamma(\nu + 1/2)}{\sqrt{2\nu}\Gamma(\nu)}$.
  - Exercise 2.7: First-principles $L^2(\Omega)$ telescoping sum proof of the sub-quadratic growth bound $\lim_{h\to\infty} \gamma(h)/h^2 = 0$.
  - Exercise 2.12: Pure Python Jacobi rotation eigen-solver rigorously verifying conditional negative definiteness.
  - Exercise 2.14: Analytic Bessel verification of the Spherical model 3D dimension barrier ($f_4(6/\phi) < 0$).

---

## Actionable Recommendations & Implementation Roadmap

1. **Immediate Mathematical Rectifications:**
   - [ ] Patch `manuscript/ch01_foundations.md:418`: Change $\sigma^2/\phi^3$ to $\sigma^2/\phi$.
   - [ ] Patch `manuscript/ch02_covariance_variogram.md:548`: Remove $4\pi^2$ from the Matérn spectral density formula.
   - [ ] Patch `manuscript/app_b.md:199, 203-204`: Replace draft ellipses with completed 4D Hankel evaluations.

2. **Algorithm & Numerical Stability Enhancements:**
   - [ ] Update `Algorithm 2.2` in `manuscript/ch02_covariance_variogram.md:490`: Port the pure Python Jacobi eigen-solver from `ch02_exercises.md:806-857` or insert LDL^T factorization to replace the faulty trace-only check.
   - [ ] Update `Algorithm 2.3` in `manuscript/ch02_covariance_variogram.md:673`: Add small-$z$ series expansion branch for non-half-integer Matérn parameter $\nu$.

3. **Notation Standardization Campaign:**
   - [ ] Perform a global notation pass on `manuscript/app_a.md` and `manuscript/app_b.md` to ensure 100% compliance with `notation_index.md` ($\mathbf{s}, \mathbf{h}, \boldsymbol{\omega}, \mathbf{c}'$).
