# Comprehensive Academic Audit & Peer Review: Batch 5.3
**Monograph:** *Theoretical Foundations of Spatial Statistics and Geostatistics*  
**Audited Batch:** Batch 5.3 — Spatial Prediction, BLUP, Kriging & Non-Stationary Models  
**Files Audited (7 files, 8,830 lines total):**
1. `manuscript/ch05_blup_kriging.md` (1,308 lines)
2. `manuscript/ch06_spatial_regression.md` (1,401 lines)
3. `manuscript/ch07_cokriging.md` (1,133 lines)
4. `manuscript/app_c.md` (421 lines)
5. `manuscript/ch05_exercises.md` (1,546 lines)
6. `manuscript/ch06_exercises.md` (1,400 lines)
7. `manuscript/ch07_exercises.md` (1,621 lines)

**Audit Date:** October 2026  
**Reviewer Role:** Senior Academic Peer Reviewer & Statistical Editor  

---

## Executive Summary

This peer review constitutes an exhaustive, line-by-line quality audit of Batch 5.3 of *Theoretical Foundations of Spatial Statistics and Geostatistics*. The audit scrutinized mathematical proofs, linear-algebraic systems, notation consistency (following Schabenberger & Gotway 2005 and Cressie 1993), LaTeX formatting, pedagogical narrative flow (Bridges, Anatomies, Micro-FAQs), and executed all pure standard-library Python code blocks.

### Overall Assessment
- **Mathematical Rigor & Depth:** Exceptional. The functional analysis foundation in $L^2(\Omega, \mathcal{F}, P)$ (Appendix C), the Gauss-Markov GLS proofs and Kruskal's Invariance Theorem (Chapter 6 & Workbook), the 1D Markov screening property of the Exponential covariance (Chapter 5), and the Kronecker-product proof of the Autokrigeability Theorem (Chapter 7) are mathematically sound, beautifully motivated, and rigorous.
- **Python Code Integrity:** **100% Pass Rate.** All 14 embedded Python blocks across all 7 files were extracted and executed in an isolated standard-library environment. Every block ran with zero syntax errors, zero runtime exceptions, and zero third-party dependencies (strictly avoiding `numpy`, `scipy`, `sklearn`, or `matplotlib`).
- **Delimiter & Typography Discipline:** **100% Delimiter Compliance.** Zero unescaped or whitespace-padded `$math$` delimiters exist. All 573 display math blocks (`$$`) across the 7 files are on dedicated standalone lines. Zero raw Unicode math symbols were detected in narrative text prose.
- **Identified Issues:**
  - **1 `[CRITICAL]` / `[WARNING]` Algebraic Sign Error**: In `app_c.md:207, 231`, an inverted sign error in the Schur complement derivation of the Ordinary Kriging Lagrange multiplier $\mu_{\text{OK}}$.
  - **1 `[WARNING]` Conversational Draft Artifact**: In `ch06_spatial_regression.md:1260-1277`, unedited conversational stream-of-consciousness draft text (`... wait, let us check the sign convention`) was inadvertently retained in the Dual Kriging exercise solution.
  - **1 `[WARNING]` Numerical Discrepancy**: In `ch05_blup_kriging.md:1223`, the printed numerical weights for Exercise 5.3(c) Gaussian covariance do not match the exact numerical solution of the $4 \times 4$ system stated.
  - **1 `[PEDAGOGICAL GAP]`**: In `ch07_exercises.md:258`, an informal shorthand for the Ordinary Cokriging variance formula was printed in prose discussion.
  - **2 `[STYLE]` Items**: Machine-specific absolute `file:///Users/jonathannavarrete/...` links in the three companion workbooks, and raw Unicode Greek letters inside ASCII art code blocks.

---

## Detailed Findings by File and Severity

```
================================================================================
FINDING CLASSIFICATION TAXONOMY
[CRITICAL]        Mathematical error, invalid proof, or broken theorem
[WARNING]         Numerical discrepancy, draft artifact, or potential confusion
[PEDAGOGICAL GAP] Incomplete derivation step or ambiguous notation
[STYLE]           Formatting inconsistency, link portability, or cosmetic glyph issue
[CLEAN]           Explicit verification and validation of exemplary sections
================================================================================
```

---

### 1. `manuscript/app_c.md` (Hilbert Space Projections and Block-Matrix Derivations)

#### Finding C.1: Inverted Sign in Schur Complement Derivation of $\mu_{\text{OK}}$
- **Severity:** `[CRITICAL]` / `[WARNING]`
- **Location:** `manuscript/app_c.md:205-208` (Theorem C.2.1, Item 4) & `app_c.md:230-233` (Proof of Theorem C.2.1, Step 4)
- **Current Text Snippet:**
  *Theorem C.2.1 (Item 4, lines 205–208):*
  ```latex
  4. The Lagrange multiplier \mu_{\text{OK}} is:
     $$
     \mu_{\text{OK}} = \frac{\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{c}_0 - 1}{\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1}}
     $$
  ```
  *Proof Step 4 (lines 230–233):*
  ```latex
  -\mu_{\text{OK}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} - \frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \implies \mu_{\text{OK}} = \frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \cdot (-1) = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}
  Notice that with the convention \mathbf{\Sigma} \boldsymbol{\lambda} - \mu_{\text{OK}} \mathbf{1} = \mathbf{c}_0, \mu_{\text{OK}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}.
  ```
- **Exact Failure Mode:**
  The Ordinary Kriging system is formulated in line 187 as:
  $$
  \begin{bmatrix} \mathbf{\Sigma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ -\mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
  $$
  The bottom row of the block inverse derived in Theorem C.2.1 (line 199) is $\begin{bmatrix} \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} & -\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \end{bmatrix}$. Multiplying by the RHS vector $\begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}$ yields the $(n+1)$-th component of the solution vector, which by definition is $-\mu_{\text{OK}}$:
  $$
  -\mu_{\text{OK}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}
  $$
  Multiplying both sides by $-1$ algebraically requires:
  $$
  \mu_{\text{OK}} = -\left(\frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}\right) = \frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}
  $$
  However, in line 231, the proof states $\mu_{\text{OK}} = \frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \cdot (-1) = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$, which negates the numerator twice. Consequently, Theorem C.2.1 Item 4 states the negated quantity.
- **Proposed Drop-In Replacement:**
  *Replace lines 205–208 with:*
  ```latex
  4. The Lagrange multiplier \mu_{\text{OK}} is:
     $$
     \mu_{\text{OK}} = \frac{1 - \mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{c}_0}{\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1}}
     $$
  ```
  *Replace lines 230–233 with:*
  ```latex
  4. Computing the bottom scalar component:
     $$
     -\mu_{\text{OK}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} - \frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \implies \mu_{\text{OK}} = -\left(\frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}\right) = \frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}
     $$
     Notice that with the convention $\mathbf{\Sigma} \boldsymbol{\lambda} - \mu_{\text{OK}} \mathbf{1} = \mathbf{c}_0$, pre-multiplying by $\mathbf{1}^T\mathbf{\Sigma}^{-1}$ gives $\mathbf{1}^T\boldsymbol{\lambda} - \mu_{\text{OK}}\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1} = \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 \implies 1 - \mu_{\text{OK}}(\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}) = \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0$, confirming $\mu_{\text{OK}} = \frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$. This completes the derivation. $\blacksquare$
  ```

---

### 2. `manuscript/ch06_spatial_regression.md` (Spatial Regression and Non-Stationary Kriging)

#### Finding C.2: Conversational Draft Artifacts in Dual Kriging Exercise 6.2(c)
- **Severity:** `[WARNING]` / `[STYLE]`
- **Location:** `manuscript/ch06_spatial_regression.md:1260-1280` (Exercise 6.2, Part c Solution)
- **Current Text Snippet:**
  ```latex
  \widehat{Y}_{\text{dual}}(s) = 3.5 - 1.5s + 1.5 - 1.5s = 5.0 - 3.0s \quad \text{... wait, let us check the sign convention.}
  $$
  In semivariogram form, recall Theorem 5.3.3: $\mathbf{\Gamma}\boldsymbol{\lambda} + \mu \mathbf{1} = \boldsymbol{\gamma}_0$.
  The Dual Kriging system in semivariogram form is $\mathbf{\Gamma}\mathbf{b} + \mathbf{1} d_1 = \mathbf{Y}$ and $\mathbf{1}'\mathbf{b} = 0$.
  Then $\widehat{Y}(s) = d_1 - \sum b_i \gamma(s - s_i)$.
  Let us check at $s = 0$:
  $$
  \widehat{Y}(0) = d_1 - b_1(0) - b_2(1) = 3.5 - 0 - (-1.5)(1) = 3.5 + 1.5 = 5.0 \ne 2.
  $$
  Why did $Y(s_1)$ and $Y(s_2)$ flip? Because $[\mathbf{\Gamma}]_{12} = \gamma(s_1 - s_2) = 1$, so:
  Row 1 is $0 \cdot b_1 + 1 \cdot b_2 + d_1 = Y_1 = 2 \implies b_2 + d_1 = 2$.
  Row 2 is $1 \cdot b_1 + 0 \cdot b_2 + d_1 = Y_2 = 5 \implies b_1 + d_1 = 5$.
  Thus at $s = s_1 = 0$, $\sum b_i \gamma(s_1 - s_i) = b_2 \gamma(s_1 - s_2) = b_2$.
  So $\widehat{Y}(s_1) = d_1 + b_2 = 2$.
  Therefore the dual equation with semivariograms is:
  $$
  \widehat{Y}_{\text{dual}}(s) = d_1 + \sum_{i=1}^n b_i \gamma(s - s_i)
  $$
  Let us check with a plus sign:
  At $s = 0$: $d_1 + b_1(0) + b_2(1) = 3.5 + (-1.5) = 2.0 = Y(s_1)$!
  At $s = 1$: $d_1 + b_1(1) + b_2(0) = 3.5 + 1.5 = 5.0 = Y(s_2)$!
  ```
- **Exact Failure Mode:**
  The text contains an informal stream-of-consciousness author note (`... wait, let us check the sign convention.`, `Why did $Y(s_1)$ and $Y(s_2)$ flip?`, and exclamation points `!`). In an advanced academic graduate monograph, working out sign conventions through trial-and-error prose detracts from didactic authority. The correct sign convention for the dual semivariogram interpolator should be directly defined, proven, and evaluated.
- **Proposed Drop-In Replacement:**
  *Replace lines 1254–1280 with:*
  ```latex
  - **(c)** The dual interpolating surface expressed via semivariograms requires careful attention to the sign of the basis kernel.
    In covariance form, $\widehat{Y}_{\text{dual}}(\mathbf{s}) = \mathbf{p}(\mathbf{s})'\mathbf{d} + \sum_{i=1}^n b_i C(\mathbf{s} - \mathbf{s}_i)$ with $\mathbf{V}\mathbf{b} + \mathbf{X}\mathbf{d} = \mathbf{Y}$.
    Under second-order stationarity, substituting $C(\mathbf{h}) = C(\mathbf{0}) - \gamma(\mathbf{h})$ and using the dual unbiasedness constraint $\mathbf{X}'\mathbf{b} = \mathbf{0}$ (here $\mathbf{1}'\mathbf{b} = 0$) gives:
    $$
    \sum_{i=1}^n b_i C(\mathbf{s} - \mathbf{s}_i) = C(\mathbf{0}) \sum_{i=1}^n b_i - \sum_{i=1}^n b_i \gamma(\mathbf{s} - \mathbf{s}_i) = -\sum_{i=1}^n b_i \gamma(\mathbf{s} - \mathbf{s}_i)
    $$
    When the dual system is formulated directly with the semivariogram matrix $\mathbf{\Gamma}\mathbf{b} + \mathbf{1} d_1 = \mathbf{Y}$, evaluating row $k$ gives:
    $$
    d_1 + \sum_{i=1}^n b_i \gamma(\mathbf{s}_k - \mathbf{s}_i) = Y(\mathbf{s}_k)
    $$
    Consequently, the exact semivariogram dual interpolating surface is:
    $$
    \widehat{Y}_{\text{dual}}(s) = d_1 + \sum_{i=1}^n b_i \gamma(s - s_i)
    $$
    Substituting $d_1 = 3.5$, $b_1 = 1.5$, $b_2 = -1.5$, and $\gamma(h) = |h|$:
    $$
    \widehat{Y}_{\text{dual}}(s) = 3.5 + 1.5|s - 0| - 1.5|s - 1|
    $$
    For $s \in [0, 1]$, this simplifies to the exact linear interpolant:
    $$
    \widehat{Y}_{\text{dual}}(s) = 3.5 + 1.5s - 1.5(1 - s) = 2.0 + 3.0s
    $$
    Evaluating at the observation sites confirms exact interpolation:
    - At $s = s_1 = 0$: $\widehat{Y}_{\text{dual}}(0) = 2.0 + 3.0(0) = 2.0 = Y(s_1)$.
    - At $s = s_2 = 1$: $\widehat{Y}_{\text{dual}}(1) = 2.0 + 3.0(1) = 5.0 = Y(s_2)$. $\blacksquare$
  ```

---

### 3. `manuscript/ch05_blup_kriging.md` (Best Linear Unbiased Prediction and Kriging)

#### Finding C.3: Numerical Discrepancy in Exercise 5.3(c) Gaussian Weights
- **Severity:** `[WARNING]`
- **Location:** `manuscript/ch05_blup_kriging.md:1221-1225`
- **Current Text Snippet:**
  ```latex
  - **(c)** Now let $C(h) = \exp(-h^2)$. The covariances are:
    $$
    C(0) = 1, \quad C(1) = e^{-1} \approx 0.367879, \quad C(2) = e^{-4} \approx 0.018316, \quad C(3) = e^{-9} \approx 0.000123
    $$
    Solving the augmented system $\mathbf{A} \mathbf{x} = \mathbf{b}$ numerically yields:
    $$
    \lambda_1 \approx +0.0461, \quad \lambda_2 \approx -0.1974, \quad \lambda_3 \approx +1.1513, \quad \mu_{\text{OK}} \approx -0.1341
    $$
    Notice that $\lambda_2 = -0.1974 < 0$ is strictly negative, and $\lambda_3 = 1.1513 > 1$.
  ```
- **Exact Failure Mode:**
  The stated problem configuration is $s_1 = -1, s_2 = 0, s_3 = 1$, target $s_0 = 2$, under $C(h) = \exp(-h^2)$.
  The distances to target $s_0$ are $h_{01} = 3, h_{02} = 2, h_{03} = 1$.
  The augmented Ordinary Kriging matrix system $\mathbf{A}\mathbf{x} = \mathbf{b}$ is:
  $$
  \begin{bmatrix} 1 & e^{-1} & e^{-4} & 1 \\ e^{-1} & 1 & e^{-1} & 1 \\ e^{-4} & e^{-1} & 1 & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ -\mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} e^{-9} \\ e^{-4} \\ e^{-1} \\ 1 \end{bmatrix}
  $$
  Direct numerical solution of this system yields:
  $$
  \lambda_1 \approx 0.328471, \quad \lambda_2 \approx -0.031559, \quad \lambda_3 \approx 0.703088, \quad -\mu_{\text{OK}} \approx -0.329615 \implies \mu_{\text{OK}} \approx +0.329615
  $$
  While $\lambda_2$ is indeed negative ($\lambda_2 \approx -0.0316 < 0$, satisfying the pedagogical objective of demonstrating negative weights), the printed figures ($\lambda_1 \approx 0.0461, \lambda_2 \approx -0.1974, \lambda_3 \approx 1.1513, \mu_{\text{OK}} \approx -0.1341$) do not correspond to the solution of this system. (Those figures correspond to an interpolation configuration with target $s_0 \approx 1.4$ or a different range parameter).
- **Proposed Drop-In Replacement:**
  *Replace lines 1221–1225 with:*
  ```latex
  - **(c)** Now let $C(h) = \exp(-h^2)$. The covariances are:
    $$
    C(0) = 1, \quad C(1) = e^{-1} \approx 0.367879, \quad C(2) = e^{-4} \approx 0.018316, \quad C(3) = e^{-9} \approx 0.000123
    $$
    Solving the augmented system $\mathbf{A} \mathbf{x} = \mathbf{b}$ numerically yields:
    $$
    \lambda_1 \approx +0.3285, \quad \lambda_2 \approx -0.0316, \quad \lambda_3 \approx +0.7031, \quad \mu_{\text{OK}} \approx +0.3296
    $$
    Notice that $\lambda_2 = -0.0316 < 0$ is strictly negative, despite all covariances in $\mathbf{V}$ and $\mathbf{v}_0$ being strictly positive.
  ```

---

### 4. `manuscript/ch07_exercises.md` (Companion Exercises: Multivariate Prediction)

#### Finding C.4: Informal Shorthand in Ordinary Cokriging Variance Formula
- **Severity:** `[PEDAGOGICAL GAP]`
- **Location:** `manuscript/ch07_exercises.md:258`
- **Current Text Snippet:**
  ```latex
  - The resulting ordinary cokriging prediction error variance $\sigma_{\text{OCK}}^2(\mathbf{s}_0) = \mathbf{c}_0'\boldsymbol{\lambda} + \mathbf{m}'\mathbf{1}$ can evaluate to negative values ($\sigma_{\text{OCK}}^2(\mathbf{s}_0) < 0$), producing undefined (imaginary) prediction standard errors.
  ```
- **Exact Failure Mode:**
  In covariance form, the Ordinary Cokriging prediction variance (as derived in Chapter 7, Theorem 7.3.2) is:
  $$
  \sigma_{\text{OCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1
  $$
  The shorthand expression $\mathbf{c}_0'\boldsymbol{\lambda} + \mathbf{m}'\mathbf{1}$ omits the prior process variance $C_{11}(\mathbf{0})$ and reverses the sign of $\boldsymbol{\lambda}'\mathbf{c}_0$. In an advanced workbook solution, using informal shorthand in a formal mathematical explanation can mislead students.
- **Proposed Drop-In Replacement:**
  *Replace line 258 with:*
  ```latex
  - The resulting ordinary cokriging prediction error variance $\sigma_{\text{OCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1$ can evaluate to negative values ($\sigma_{\text{OCK}}^2(\mathbf{s}_0) < 0$), producing undefined (imaginary) prediction standard errors.
  ```

---

### 5. Cross-File Style & Portability Audit

#### Finding C.5: Machine-Specific Absolute `file:///` URLs in Companion Workbooks
- **Severity:** `[STYLE]`
- **Locations:**
  - `manuscript/ch05_exercises.md:1544`
  - `manuscript/ch06_exercises.md:1398`
  - `manuscript/ch07_exercises.md:1619`
- **Current Text Snippets:**
  - `[Chapter 5: Spatial Prediction and Kriging Theory (BLUP)](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch05_blup_kriging.md)`
  - `[Chapter 6: Spatial Regression and Non-Stationary Processes](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch06_spatial_regression.md)`
  - `[Chapter 7: Multivariate Geostatistics and Cokriging Theory](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch07_cokriging.md)`
- **Exact Failure Mode:**
  Hardcoding absolute file URLs tied to a specific username and filesystem path (`file:///Users/jonathannavarrete/...`) breaks portability across other developer workstations, Linux servers, GitHub/GitLab web rendering, and Docker CI pipelines.
- **Proposed Drop-In Replacement:**
  *Use clean relative markdown links:*
  - In `ch05_exercises.md`: `[Chapter 5: Spatial Prediction and Kriging Theory (BLUP)](ch05_blup_kriging.md)`
  - In `ch06_exercises.md`: `[Chapter 6: Spatial Regression and Non-Stationary Processes](ch06_spatial_regression.md)`
  - In `ch07_exercises.md`: `[Chapter 7: Multivariate Geostatistics and Cokriging Theory](ch07_cokriging.md)`

#### Finding C.6: Unicode Greek Glyphs Inside ASCII Diagrams
- **Severity:** `[STYLE]`
- **Locations:**
  - `ch05_blup_kriging.md:174, 178, 181, 185` (`μ, σ, ∈`), `L871` (`λ`)
  - `ch06_spatial_regression.md:140, 143, 151` (`β, μ, θ`), `L576, 580` (`τ, σ`)
- **Exact Failure Mode:**
  Inside fenced `text` or plain ASCII art diagram boxes, multi-byte Unicode Greek characters (`μ`, `σ`, etc.) can render with irregular character widths in certain monospaced fonts, leading to box misalignment.
- **Recommendation:**
  In pure ASCII diagrams, prefer standard ASCII names (`mu`, `sigma`, `beta`, `theta`, `tau`, `lambda`, `in`) to guarantee pixel-perfect box alignment across all terminal emulators and markdown viewers.

---

## Exemplary Sections Validated as `[CLEAN]`

1. **Functional-Analytic Foundations of Prediction (`app_c.md:7-131` — Theorem C.1.1)**
   - *Proof of the Orthogonal Projection Theorem:* Rigorous application of the Parallelogram Law (Lemma C.1.1) to establish Cauchy convergence in $L^2(\Omega, \mathcal{F}, P)$, uniqueness of the closest point, necessary and sufficient orthogonality $\langle X - \widehat{X}, Y \rangle = 0$, and projection operator properties ($\Pi_{\mathcal{M}}^2 = \Pi_{\mathcal{M}}$, $\Pi_{\mathcal{M}}^* = \Pi_{\mathcal{M}}$, $\|\Pi_{\mathcal{M}}\| = 1$).

2. **Kruskal's Invariance Theorem & Whitening Proofs (`ch06_exercises.md:17-113` — Exercise 6.1 & 6.2)**
   - *Theorem 6.2.1 & Kruskal's Theorem:* Rigorous proof that $\widehat{\boldsymbol{\beta}}_{\text{OLS}} \equiv \widehat{\boldsymbol{\beta}}_{\text{GLS}}$ if and only if $\mathcal{C}(\mathbf{X})$ is an invariant subspace of $\mathbf{V}$ ($\mathbf{V}\mathcal{C}(\mathbf{X}) = \mathcal{C}(\mathbf{X})$). Exemplary demonstration for compound symmetry covariance where $\bar{Y}$ is strictly BLUE for any $\rho$.

3. **1D Markov Screening Effect (`ch05_blup_kriging.md:812-869` — Theorem 5.4.1)**
   - Unabridged proof demonstrating that for collinear configurations $s_1 < s_2 < \dots < s_n$ and target $s_0 \in (s_k, s_{k+1})$, all weights outside the adjacent bounding neighbors $s_k$ and $s_{k+1}$ solve identically to zero under the Exponential covariance:
     $$
     \lambda_i = 0 \quad \forall i \notin \{k, k+1\}
     $$
     Hyperbolic trigonometric factorization cleanly demonstrates exact screening.

4. **Autokrigeability Theorem via Kronecker Products (`ch07_cokriging.md:816-945` — Theorem 7.5)**
   - Matrix factorization $\boldsymbol{\Sigma} = \mathbf{B} \otimes \mathbf{R}$, $\mathbf{X} = \mathbf{I}_p \otimes \mathbf{1}_n$, $\mathbf{c}_{0,1} = (\mathbf{B}\mathbf{e}_1) \otimes \mathbf{r}_0$.
   - Candidate solution $\boldsymbol{\lambda} = \mathbf{e}_1 \otimes \boldsymbol{\lambda}^*$ and $\boldsymbol{\mu}_{\text{OCK}} = \nu^* \mathbf{B}\mathbf{e}_1$.
   - Flawless algebraic verification via the mixed-product rule $(\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D}) = (\mathbf{AC}) \otimes (\mathbf{BD})$. Proves secondary weights are identically zero and cokriging collapses to univariate Ordinary Kriging.

5. **Pure Python Standard-Library Implementations (All 14 code blocks)**
   - Every single algorithm runs in native Python without a single external import.
   - Includes Gaussian elimination with partial pivoting, Cholesky decomposition $\mathbf{L}\mathbf{L}' = \mathbf{V}$, triangular back-substitution, Ordinary Kriging solver, Universal Kriging / GLS solver, Multivariate Block Cokriging engine, and cross-validation residual diagnostics.

---

## Actionable Recommendations & Prioritized Checklist

| Item # | Target File | Line(s) | Severity | Action Required |
|:---:|:---|:---:|:---:|:---|
| 1 | `manuscript/app_c.md` | 207, 231 | `[CRITICAL]` | Fix sign flip in Theorem C.2.1 (Item 4) and Proof Step 4 to $\mu_{\text{OK}} = \frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$. |
| 2 | `manuscript/ch06_spatial_regression.md` | 1260–1277 | `[WARNING]` | Remove conversational draft notes in Exercise 6.2(c) and present the direct dual semivariogram interpolator. |
| 3 | `manuscript/ch05_blup_kriging.md` | 1223 | `[WARNING]` | Update numerical values in Exercise 5.3(c) Gaussian weights to match the exact solution: $\lambda_1 \approx 0.3285, \lambda_2 \approx -0.0316, \lambda_3 \approx 0.7031, \mu_{\text{OK}} \approx 0.3296$. |
| 4 | `manuscript/ch07_exercises.md` | 258 | `[PEDAGOGICAL GAP]` | Replace informal variance shorthand with exact expression $\sigma_{\text{OCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1$. |
| 5 | `manuscript/ch05_exercises.md` | 1544 | `[STYLE]` | Convert `file:///Users/jonathannavarrete/...` link to relative `ch05_blup_kriging.md`. |
| 6 | `manuscript/ch06_exercises.md` | 1398 | `[STYLE]` | Convert `file:///Users/jonathannavarrete/...` link to relative `ch06_spatial_regression.md`. |
| 7 | `manuscript/ch07_exercises.md` | 1619 | `[STYLE]` | Convert `file:///Users/jonathannavarrete/...` link to relative `ch07_cokriging.md`. |
| 8 | `ch05_blup_kriging.md` / `ch06_spatial_regression.md` | Multiple | `[STYLE]` | Replace raw Unicode Greek letters in ASCII art boxes with ASCII names (`mu`, `sigma`, etc.). |

---
*Report completed and submitted by Antigravity Peer Review Engine.*
