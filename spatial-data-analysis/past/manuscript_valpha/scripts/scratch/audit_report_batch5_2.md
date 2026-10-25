# Comprehensive Phase 5 Academic Peer Review & Mathematical Audit Report
**Graduate Monograph:** *Theoretical Foundations of Spatial Statistics and Geostatistics*  
**Batch Scope:** Batch 5.2  
**Target Files:**
1. `manuscript/ch03_estimation.md` (1,412 lines)
2. `manuscript/ch04_parameter_estimation.md` (1,170 lines)
3. `manuscript/app_d.md` (477 lines)
4. `manuscript/ch03_exercises.md` (1,156 lines)
5. `manuscript/ch04_exercises.md` (1,256 lines)  
**Total Volume Audited:** 5,471 lines  
**Date of Audit:** October 2026  
**Auditor:** Reviewer Subagent 2 (Verification Engine)

---

## Executive Summary & Quality Dashboard

This audit delivers an exhaustive mathematical, semantic, pedagogical, and algorithmic verification of Batch 5.2 of the monograph *Theoretical Foundations of Spatial Statistics and Geostatistics*. The audit strictly adheres to the editorial mandates: read-only inspection of manuscript files, Schabenberger & Gotway (2005) / Cressie (1993) semantic notation discipline, Rizzo pedagogical three-beat structure, and pure standard-library Python implementations (strictly zero `numpy`/`scipy` in the core text).

### Summary Table of Findings

| File | Severity | Issue Summary | Impact / Root Cause |
| :--- | :---: | :--- | :--- |
| `ch03_estimation.md` | **CRITICAL** | All 6 Python code blocks import and utilize `numpy` | Violates strict pure Python standard library mandate. |
| `ch03_estimation.md` | **CRITICAL** | FAQ 1 (line 764): Double-division by $N$ in Taylor expansion for $\operatorname{E}[\bar{Y}^4]$ | Evaluates to $\mathcal{O}(N^{-2})$ variance term instead of $\mathcal{O}(N^{-1})$ due to multiplying by $1/N$ when $\operatorname{Var}(\bar{Y})$ is already used. |
| `ch03_estimation.md` | **STYLE** | Duplicate prose token (line 255: `"erratic erratic"`) | Minor typographical stutter in prose. |
| `ch04_parameter_estimation.md` | **CRITICAL** | All 6 Python code blocks import `numpy`; Block 4 imports `scipy.optimize.minimize` | Violates strict pure Python standard library mandate. |
| `ch04_parameter_estimation.md` | **CRITICAL** | Harville REML conditioning argument (lines 501–515): asserts $\widehat{\boldsymbol{\beta}}_{\text{OLS}} \mid \mathbf{W}$ is Gaussian with mean $\boldsymbol{\beta}$ and variance $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$ | Mathematically invalid: $\operatorname{Cov}(\widehat{\boldsymbol{\beta}}_{\text{OLS}}, \mathbf{W}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}\mathbf{K} \ne \mathbf{0}$. Independence holds for GLS $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$, not OLS. |
| `ch04_parameter_estimation.md` | **WARNING** | Notation drift: random field observation vector alternates arbitrarily between $\mathbf{Z}$ and $\mathbf{Y}$ | Inconsistency across Chapter 4 definitions, anatomy boxes, and theorems. |
| `ch03_exercises.md` | **WARNING** | Exercise 3.8(d) solution (line 596): arithmetic intermediate typo writing $0.456947$ instead of $0.675978$ in $\operatorname{Var}(\|W\|^{1/2})$ | Final value $0.121907$ is correct, but intermediate printed difference is inconsistent. |
| `ch04_exercises.md` | **STYLE** | Line 396: raw unescaped `\blacksquare` missing math delimiters | Renders as raw text string `\blacksquare` rather than KaTeX Q.E.D. symbol. |
| `app_d.md` | **CLEAN** | All geometric covariograms, dimensional reduction proofs, and chord distributions verified | Complete mathematical rigor; zero defects detected. |

---

## Detailed Audit: Chapter 3 (`manuscript/ch03_estimation.md`)

### 1. Proof Integrity & Theoretical Rigor

#### A. Theorem 3.2.1 (Matheron MoM Unbiasedness)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 280–305. The derivation takes the expectation $\operatorname{E}[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2] = \operatorname{Var}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j)) + (\operatorname{E}[Z(\mathbf{s}_i) - Z(\mathbf{s}_j)])^2$. Under intrinsic stationarity of order 0, the first moment difference vanishes identically ($\operatorname{E}[Z(\mathbf{s}_i) - Z(\mathbf{s}_j)] = \mu - \mu = 0$). By definition, the variance of the increment equals $2\gamma(\mathbf{s}_i - \mathbf{s}_j) = 2\gamma(\mathbf{h})$. The sum over $|N(\mathbf{h})|$ terms yields $\frac{1}{2|N(\mathbf{h})|} \sum 2\gamma(\mathbf{h}) = \gamma(\mathbf{h})$. No centering, estimation, or knowledge of $\mu$ is required, maintaining complete rigor.

#### B. Theorem 3.2.2 (Downward Bias of Sample Covariance)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 330–365. The expansion of $(Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z})$ produces four terms: $Z(\mathbf{s}_i)Z(\mathbf{s}_j) - \bar{Z}Z(\mathbf{s}_i) - \bar{Z}Z(\mathbf{s}_j) + \bar{Z}^2$. Taking expectations under second-order stationarity yields:
  $$\operatorname{E}[\widehat{C}(\mathbf{h}_k)] = C(\mathbf{h}_k) - 2\bar{C}(N(\mathbf{h}_k), S) + \operatorname{Var}(\bar{Z})$$
  The proof correctly emphasizes that because $\operatorname{Var}(\bar{Z}) > 0$ and $\bar{C}(N(\mathbf{h}_k), S) \to 0$ as $h_k$ grows large, $\operatorname{E}[\widehat{C}(\mathbf{h}_k)] \approx -\operatorname{Var}(\bar{Z}) < 0$, providing an elegant pedagogical explanation for why sample autocovariance estimators artificially dip into negative values at large lags.

#### C. Genton Graph Laplacian $\mathbf{A}(\mathbf{h})$ (Definition 3.1.2)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 75–105. Genton's quadratic form is expressed as $\widehat{\gamma}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|} \mathbf{Z}' \mathbf{A}(\mathbf{h}) \mathbf{Z}$ where $\mathbf{A}(\mathbf{h}) = \mathbf{D}(\mathbf{h}) - \mathbf{W}(\mathbf{h})$ is the combinatorial graph Laplacian of the undirected lag-network $G_{\mathbf{h}} = (V, E_{\mathbf{h}})$.
  - Row-sum property: $\mathbf{A}(\mathbf{h})\mathbf{1} = \mathbf{0}$ follows immediately from $D_{ii} = \sum_{j} W_{ij}$, establishing drift invariance: $(\mathbf{Z} + c\mathbf{1})' \mathbf{A}(\mathbf{h}) (\mathbf{Z} + c\mathbf{1}) = \mathbf{Z}' \mathbf{A}(\mathbf{h}) \mathbf{Z}$.
  - Rank-Nullity Theorem: $\operatorname{rank}(\mathbf{A}(\mathbf{h})) = n - c(\mathbf{h})$ where $c(\mathbf{h})$ is the number of connected components in $G_{\mathbf{h}}$. The text accurately proves that the nullspace is spanned by the component indicator vectors.

#### D. Theorem 3.4.1 (Cressie-Hawkins $0.457 / 0.494$ Derivation)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 675–745.
  1. For $W \sim \mathcal{N}(0, 1)$, $|W|^{1/2}$ has folded normal expectation:
     $$\operatorname{E}[|W|^{1/2}] = \frac{2^{1/4}}{\sqrt{\pi}} \Gamma(3/4) \approx \frac{1.189207}{1.772454} \times 1.225417 \approx 0.822179$$
  2. The fourth power is $(0.822179)^4 = 0.456947 \approx 0.457$.
  3. The delta-method Taylor series expansion of $\bar{Y}^4$ around $\mu_Y$ correctly derives the second-order Jensen's inequality correction:
     $$\frac{6 \operatorname{Var}(Y)}{|N(\mathbf{h})| (\operatorname{E}[Y])^2} \times 0.457 \approx \frac{6 \times 0.121907}{(0.822179)^2} \times 0.457 \approx 0.494$$
  The mathematical steps in the main proof are airtight.

---

### 2. Discovered Defects in Chapter 3

#### Defect 3.1 [CRITICAL]: Python Code Standard Library Violations
- **File & Lines:** `manuscript/ch03_estimation.md`, lines 171–241, 380–444, 564–615, 790–845, 958–1013, 1129–1199.
- **Failure Mode:** All six Python code blocks import `numpy as np` and use NumPy arrays, vectorized operations (`np.dot`, `np.sqrt`, `np.sum`, `np.arccos`), and array slicing. This directly violates the project guideline requiring pure standard library Python (`math`, procedural loops, nested lists).
- **Remediation:** Full drop-in replacements in pure Python standard library are provided in Section 4 of this report.

#### Defect 3.2 [CRITICAL]: Formula Variance Typo in FAQ 1
- **File & Line:** `manuscript/ch03_estimation.md`, line 764.
- **Current Text:**
  ```latex
  $$\operatorname{E}[\bar{Y}^4] = (\operatorname{E}[\bar{Y}])^4 + \frac{6}{N}(\operatorname{E}[\bar{Y}])^2 \operatorname{Var}(\bar{Y}) + \mathcal{O}(N^{-2})$$
  ```
- **Failure Mode:** In this formula, $\bar{Y} = \frac{1}{N}\sum_{i=1}^N Y_i$ has variance $\operatorname{Var}(\bar{Y}) = \frac{\operatorname{Var}(Y)}{N}$. The text includes both the explicit divisor $\frac{6}{N}$ and $\operatorname{Var}(\bar{Y})$, which accidentally divides by $N$ twice:
  $$\frac{6}{N}(\operatorname{E}[\bar{Y}])^2 \operatorname{Var}(\bar{Y}) = \frac{6}{N^2}(\operatorname{E}[Y])^2 \operatorname{Var}(Y) = \mathcal{O}(N^{-2})$$
  This contradicts the main proof on line 724, which correctly writes:
  $$\operatorname{E}[\bar{Y}^4] \approx \mu_Y^4 + 6\mu_Y^2 \frac{\sigma_Y^2}{N}$$
- **Proposed Drop-In Replacement:**
  ```latex
  $$\operatorname{E}[\bar{Y}^4] = (\operatorname{E}[Y])^4 + \frac{6}{N}(\operatorname{E}[Y])^2 \operatorname{Var}(Y) + \mathcal{O}(N^{-2})$$
  ```
  *(Alternatively, expressed in terms of the sample mean: $\operatorname{E}[\bar{Y}^4] = (\operatorname{E}[\bar{Y}])^4 + 6(\operatorname{E}[\bar{Y}])^2 \operatorname{Var}(\bar{Y}) + \mathcal{O}(N^{-2})$).*

#### Defect 3.3 [STYLE]: Prose Duplicate Word Stutter
- **File & Line:** `manuscript/ch03_estimation.md`, line 255.
- **Current Text:**
  ```markdown
  inducing erratic erratic point-to-point fluctuations in empirical semivariograms.
  ```
- **Failure Mode:** The word `"erratic"` is accidentally repeated twice consecutively.
- **Proposed Drop-In Replacement:**
  ```markdown
  inducing erratic point-to-point fluctuations in empirical semivariograms.
  ```

---

## Detailed Audit: Chapter 4 (`manuscript/ch04_parameter_estimation.md`)

### 1. Proof Integrity & Theoretical Rigor

#### A. Theorem 4.1.1 & Theorem 4.1.2 (Profile ML and Information Matrix)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 140–235. The score equation $\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \mathbf{X}'\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$ yields the exact GLS estimator $\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z}$.
  - The cross-derivative $\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \theta_k} = \mathbf{X}'\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})$ has expectation $\mathbf{0}$ because $\operatorname{E}[\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}] = \mathbf{0}$.
  - This rigorously establishes block-diagonality of the Fisher Information matrix: $\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\theta}} = \mathbf{0}$, proving asymptotic independence between drift parameters and covariance parameters under Gaussianity.

#### B. Spatial Projection Operator $\mathbf{P}(\boldsymbol{\theta})$ Invariance Properties
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 320–375.
  1. $\mathbf{P} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}$.
  2. $\mathbf{P}\mathbf{X} = \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) = \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X} = \mathbf{0}$.
  3. $\mathbf{P}\mathbf{V}\mathbf{P} = \mathbf{P}(\mathbf{V}\mathbf{V}^{-1} - \mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}) = \mathbf{P}(\mathbf{I} - \mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}) = \mathbf{P} - \mathbf{0} = \mathbf{P}$.
  4. Trace identity: $\operatorname{Tr}(\mathbf{P}\mathbf{V}) = \operatorname{Tr}(\mathbf{I}_n - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}') = \operatorname{Tr}(\mathbf{I}_n) - \operatorname{Tr}((\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) = n - p$.
  The algebraic manipulations and degrees-of-freedom corrections are fully verified.

#### C. Theorem 4.6.2 (Frobenius Projection onto PSD Cone)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 905–920. Minimizing $\|\mathbf{B} - \mathbf{M}\|_F^2$ subject to $\mathbf{M} \succeq 0$. Using the unitary invariance of the Frobenius norm, $\|\mathbf{B} - \mathbf{M}\|_F^2 = \|\mathbf{\Lambda} - \mathbf{Q}'\mathbf{M}\mathbf{Q}\|_F^2 = \sum_{i \ne j} (\mathbf{Q}'\mathbf{M}\mathbf{Q})_{ij}^2 + \sum_j (\lambda_j - (\mathbf{Q}'\mathbf{M}\mathbf{Q})_{jj})^2$. The minimum is achieved when off-diagonals are zero and diagonal entries are projected onto the non-negative real axis: $(\mathbf{Q}'\mathbf{M}\mathbf{Q})_{jj} = \max(0, \lambda_j)$. Thus $\mathbf{B}^* = \mathbf{Q}\mathbf{\Lambda}^+\mathbf{Q}'$. Fully verified.

---

### 2. Discovered Defects in Chapter 4

#### Defect 4.1 [CRITICAL]: Python Code Standard Library Violations
- **File & Lines:** `manuscript/ch04_parameter_estimation.md`, lines 251–299, 397–438, 561–622, 712–758, 825–848, 925–959.
- **Failure Mode:** All six code blocks import `numpy as np`, use NumPy linear algebra routines (`np.linalg.cholesky`, `np.linalg.solve`, `np.linalg.eigh`), and Block 4 imports `from scipy.optimize import minimize`. This violates the pure Python mandate.
- **Remediation:** Full drop-in pure Python replacements with embedded numerical routines (pure procedural Cholesky factorization, triangular forward/back substitution, Nelder-Mead simplex optimization, and cyclic Jacobi eigenvalue decomposition) are provided in Section 4.

#### Defect 4.2 [CRITICAL / MATH]: Flawed Conditioning Argument in Harville REML Derivation
- **File & Lines:** `manuscript/ch04_parameter_estimation.md`, lines 501–515.
- **Current Text:**
  ```markdown
  Consider the non-singular linear transformation of the data vector $\mathbf{Z}$:
  $$
  \mathbf{T}\mathbf{Z} = \begin{bmatrix} \widehat{\boldsymbol{\beta}}_{\text{OLS}} \\ \mathbf{W} \end{bmatrix} = \begin{bmatrix} (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}' \\ \mathbf{K}' \end{bmatrix}\mathbf{Z}
  $$
  The conditional distribution of $\widehat{\boldsymbol{\beta}}$ given $\mathbf{W}$ is Gaussian with mean $\boldsymbol{\beta}$ and variance $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$. The joint density factors as:
  $$
  f_{\mathbf{T}\mathbf{Z}}(\widehat{\boldsymbol{\beta}}, \mathbf{w}; \boldsymbol{\theta}, \boldsymbol{\beta}) = f_{\widehat{\boldsymbol{\beta}}|\mathbf{W}}(\widehat{\boldsymbol{\beta}}|\mathbf{w}; \boldsymbol{\theta}, \boldsymbol{\beta}) \cdot f_{\mathbf{W}}(\mathbf{w}; \boldsymbol{\theta})
  $$
  ```
- **Failure Mode:** This argument is mathematically flawed:
  1. The transformation uses $\widehat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Z}$. In the presence of spatial correlation $\mathbf{V}$, $\operatorname{Cov}(\widehat{\boldsymbol{\beta}}_{\text{OLS}}, \mathbf{W}) = \operatorname{Cov}((\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Z}, \mathbf{K}'\mathbf{Z}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}\mathbf{K}$. Because $\mathbf{K}$ annihilates $\mathbf{X}$ ($\mathbf{K}'\mathbf{X} = \mathbf{0}$), it does **NOT** annihilate $\mathbf{X}'\mathbf{V}$ unless $\mathbf{V} \propto \mathbf{I}$.
  2. Because $\operatorname{Cov}(\widehat{\boldsymbol{\beta}}_{\text{OLS}}, \mathbf{W}) \ne \mathbf{0}$, $\widehat{\boldsymbol{\beta}}_{\text{OLS}}$ and $\mathbf{W}$ are correlated under Gaussianity. Therefore, the conditional distribution of $\widehat{\boldsymbol{\beta}}_{\text{OLS}} \mid \mathbf{W}$ depends heavily on $\mathbf{W}$, its conditional mean is not $\boldsymbol{\beta}$, and its conditional variance is not $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$.
  3. The estimator that is **strictly independent** of $\mathbf{W} = \mathbf{K}'\mathbf{Z}$ is the **Generalized Least Squares (GLS)** estimator:
     $$\widehat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z}$$
     Proof of independence:
     $$\operatorname{Cov}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{W}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\operatorname{Var}(\mathbf{Z})\mathbf{K} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}(\sigma^2\mathbf{V})\mathbf{K} = \sigma^2(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{K} = \mathbf{0}$$
     Because $\mathbf{K}'\mathbf{X} = \mathbf{0}$, the covariance vanishes identically! Thus $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and $\mathbf{W}$ are jointly Gaussian and uncorrelated, hence independent.
- **Proposed Drop-In Replacement:**
  ```latex
  Consider the non-singular linear transformation of the data vector $\mathbf{Z}$:
  $$
  \mathbf{T}\mathbf{Z} = \begin{bmatrix} \widehat{\boldsymbol{\beta}}_{\text{GLS}} \\ \mathbf{W} \end{bmatrix} = \begin{bmatrix} (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1} \\ \mathbf{K}' \end{bmatrix}\mathbf{Z}
  $$
  where $\mathbf{K}'\mathbf{X} = \mathbf{0}$. We verify the cross-covariance between $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and error contrasts $\mathbf{W}$:
  $$
  \operatorname{Cov}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{W}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\operatorname{Var}(\mathbf{Z})\mathbf{K} = \sigma^2(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{K} = \mathbf{0}
  $$
  Because $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and $\mathbf{W}$ are jointly Gaussian with zero cross-covariance, they are statistically independent. The marginal distribution of $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ is $\mathcal{N}_p(\boldsymbol{\beta}, (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1})$, which contains all information regarding the drift coefficients $\boldsymbol{\beta}$. Consequently, the joint density factors as the product of independent marginals:
  $$
  f_{\mathbf{T}\mathbf{Z}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{w}; \boldsymbol{\theta}, \boldsymbol{\beta}) = f_{\widehat{\boldsymbol{\beta}}_{\text{GLS}}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}; \boldsymbol{\theta}, \boldsymbol{\beta}) \cdot f_{\mathbf{W}}(\mathbf{w}; \boldsymbol{\theta})
  $$
  Applying the change-of-variables formula with the block-triangular Jacobian determinant $|\det(\mathbf{T}\mathbf{V}\mathbf{T}')|^{1/2} = |\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}|^{-1/2} |\mathbf{K}'\mathbf{V}\mathbf{K}|^{1/2}$ immediately isolates the marginal likelihood of the error contrasts:
  $$
  \ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{1}{2}\left[(n - p)\ln(2\pi) + \ln|\mathbf{V}(\boldsymbol{\theta})| + \ln|\mathbf{X}'\mathbf{V}^{-1}(\boldsymbol{\theta})\mathbf{X}| + \mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}\right]
  $$
  ```

#### Defect 4.3 [WARNING / NOTATION]: Notation Drift Between $\mathbf{Z}$ and $\mathbf{Y}$
- **File & Lines:** `manuscript/ch04_parameter_estimation.md`, lines 9, 33, 125, 132, 524.
- **Failure Mode:** The chapter vacillates between $\mathbf{Z}$ and $\mathbf{Y}$:
  - Line 9 uses $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}(\boldsymbol{\theta}))$.
  - Line 33 uses $\mathbf{Y} \sim \mathcal{N}(\mathbf{X}\boldsymbol{\beta}, \mathbf{V})$.
  - Line 125 defines the log-likelihood in terms of $\mathbf{Z}$.
  - Line 132 (Term-by-Term Anatomy) defines the quadratic form using $\mathbf{Y}'\mathbf{V}^{-1}\mathbf{Y}$.
  - Line 524 (REML Anatomy) defines the projected form using $\mathbf{Y}'\mathbf{P}\mathbf{Y}$.
- **Proposed Drop-In Replacement:** Standardize exclusively to $\mathbf{Z}$ throughout Chapter 4 to match the rest of the monograph:
  - Replace $\mathbf{Y}'\mathbf{V}^{-1}\mathbf{Y} \to (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})$ on line 132.
  - Replace $\mathbf{Y}'\mathbf{P}\mathbf{Y} \to \mathbf{Z}'\mathbf{P}\mathbf{Z}$ on line 524.

---

## Detailed Audit: Appendix D (`manuscript/app_d.md`)

### 1. Mathematical Rigor & Proof Integrity

#### A. Theorem D.1.1 & D.1.2 (Geometric Covariogram Properties)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 20–60. $K_v(\mathbf{r}) = |v \cap (v - \mathbf{r})| = \int_{\mathbb{R}^d} \mathbf{1}_v(\mathbf{x})\mathbf{1}_v(\mathbf{x} + \mathbf{r}) \, d\mathbf{x} = (\mathbf{1}_v * \check{\mathbf{1}}_v)(\mathbf{r})$.
  - At $\mathbf{r} = \mathbf{0}$, $K_v(\mathbf{0}) = |v|$.
  - Symmetry: $K_v(-\mathbf{r}) = K_v(\mathbf{r})$ follows by changing variables $\mathbf{y} = \mathbf{x} - \mathbf{r}$.
  - Compact support: $K_v(\mathbf{r}) = 0$ for $\|\mathbf{r}\| > \operatorname{diam}(v)$.
  - Energy integral: $\int_{\mathbb{R}^d} K_v(\mathbf{r}) \, d\mathbf{r} = |v|^2$, matching Fubini's theorem.

#### B. Theorem D.2.1 (Dimensional Reduction via Isotropization)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 95–140. Double integral over $v \times v$ under isotropic stationary covariance $C(\|\mathbf{x} - \mathbf{y}\|)$ reduces to 1D radial integral:
  $$\iint_{v \times v} C(\|\mathbf{x} - \mathbf{y}\|) \, d\mathbf{x} d\mathbf{y} = \int_{\mathbb{R}^d} K_v(\mathbf{r}) C(\|\mathbf{r}\|) \, d\mathbf{r} = S_{d-1} \int_0^\infty r^{d-1} \bar{K}_v(r) C(r) \, dr$$
  where $\bar{K}_v(r) = \frac{1}{S_{d-1}} \int_{S^{d-1}} K_v(r\mathbf{u}) \, d\sigma(\mathbf{u})$ is the directional average. The proof substitution using spherical coordinate Jacobians is exact.

#### C. Theorems D.2.2 & D.4.1 (Cauchy-Crofton Chord Length Distributions)
- **Status:** **VERIFIED (Rigorously Complete)**
- **Audit Details:** Evaluated lines 150–190 and 370–415. The relationship between the directional derivative of the geometric covariogram along unit vector $\mathbf{u}$ and the chord length distribution $f_L(l; \mathbf{u})$:
  $$\frac{d^2 K_v(r\mathbf{u})}{dr^2} = A_v(\mathbf{u}) f_L(r; \mathbf{u})$$
  and the total boundary measure $\left. \frac{d\bar{K}_v}{dr} \right|_{r=0^+} = -\frac{\omega_d}{\pi \omega_{d-1}} \partial |v|$ are classic Matheron (1975) results. Every derivation is mathematically flawless.

#### D. Note on Prompt Scope (Fisher Scoring)
- **Clarification:** The initial audit prompt mentioned checking "Fisher scoring in App D". Examination confirms that `app_d.md` is strictly and exclusively dedicated to *Integral Geometry Relations for Spatial Regularization*. Fisher scoring is treated in Chapter 4 / Exercise 4.7. The header in `TODO.md` was an outdated artifact from an earlier monograph outline.

---

## Detailed Audit: Chapter Exercises (`ch03_exercises.md` & `ch04_exercises.md`)

### 1. Chapter 3 Exercises (`manuscript/ch03_exercises.md`)

- **Code Block Audit:** Lines 818–905 (Exercise 3.12: `empirical_semivariogram_engine`) and lines 920–990 (Exercise 3.13: `evaluate_covariance_bias`). Both scripts strictly use pure standard-library Python (`math`, procedural loops, nested lists). Zero NumPy/SciPy usage.
- **Proof & Solution Integrity:**
  - Exercise 3.1 (Genton matrix rank on 1D transect: $\operatorname{rank}(\mathbf{A}) = n - \gcd(h, n)$ components).
  - Exercise 3.6 (Analytical downward bias derivation).
  - Exercise 3.11 (Cross-variogram vs pseudo-cross variogram symmetries: $\gamma_{ij}(\mathbf{h}) = \gamma_{ji}(\mathbf{h})$ vs $2\gamma_{ij}^*(\mathbf{h}) = 2\gamma_{ji}^*(-\mathbf{h})$).
- **Discovered Minor Defect:**
  - **Defect 3E.1 [WARNING / MATH]:** Exercise 3.8(d) solution, line 596:
    ```markdown
    Evaluating $\operatorname{Var}(|W|^{1/2}) = \operatorname{E}[|W|] - (\operatorname{E}[|W|^{1/2}])^2 = \sqrt{2/\pi} - 0.456947 \approx 0.797885 - 0.456947 = 0.121907$.
    ```
    *Failure Mode:* The calculation writes $0.456947$ (which is $(\operatorname{E}[|W|^{1/2}])^4$) instead of $(\operatorname{E}[|W|^{1/2}])^2 = (0.822179)^2 \approx 0.675978$. Note that $0.797885 - 0.675978 = 0.121907$. The final numerical result $0.121907$ is exact, but the intermediate subtraction displayed contains an arithmetic typographical error ($0.797885 - 0.456947 = 0.340938 \ne 0.121907$).
    *Proposed Drop-In Replacement:*
    ```markdown
    Evaluating $\operatorname{Var}(|W|^{1/2}) = \operatorname{E}[|W|] - (\operatorname{E}[|W|^{1/2}])^2 = \sqrt{2/\pi} - (0.822179)^2 \approx 0.797885 - 0.675978 = 0.121907$.
    ```

### 2. Chapter 4 Exercises (`manuscript/ch04_exercises.md`)

- **Code Block Audit:** Lines 872–981 (Exercise 4.12: `cholesky_factor`, `solve_triangular_lower`, `solve_triangular_upper`, `invert_spd_matrix`, `evaluate_reml_objective`) and lines 998–1055 (Exercise 4.13: `project_lmc_matrices`, `check_sylvester_admissibility_2x2`). All code is 100% pure Python standard library.
- **Proof & Solution Integrity:**
  - Exercise 4.2 & 4.3: Proves that pairwise difference composite log-likelihood score equations are algebraically identical to Cressie's WLS estimating equations $\sum \frac{|N(h_k)|}{2\gamma^2(h_k)} (\widehat{\gamma} - \gamma) \nabla \gamma = \mathbf{0}$.
  - Exercise 4.4: Hand derivation of the explicit $3 \times 3$ REML projection matrix $\mathbf{P}_{\mathbf{V}} = \frac{2}{5}\mathbf{v}\mathbf{v}'$ where $\mathbf{v} = (1, -2, 1)'$. Verified that $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \mathbf{0}_{3 \times 2}$, $\mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} = \mathbf{P}_{\mathbf{V}}$, and $\operatorname{Tr}(\mathbf{P}_{\mathbf{V}}\mathbf{V}) = 3 - 2 = 1$.
  - Exercise 4.11: Eigen-clipping projection of $\mathbf{B} = \begin{pmatrix} 10 & 8 \\ 8 & 5 \end{pmatrix}$.
- **Discovered Minor Defect:**
  - **Defect 4E.1 [STYLE / FORMATTING]:** Line 396:
    ```markdown
    ...corresponding to the estimated drift coefficients $\widehat{\boldsymbol{\beta}}$. \blacksquare
    ```
    *Failure Mode:* Raw `\blacksquare` is missing math delimiters `$`, causing it to render as plain text `\blacksquare`.
    *Proposed Drop-In Replacement:*
    ```markdown
    ...corresponding to the estimated drift coefficients $\widehat{\boldsymbol{\beta}}$. $\blacksquare$
    ```

---

## Complete Pure Python Drop-In Code Replacements

Below are the complete, tested, pure standard-library Python implementations to replace the NumPy/SciPy code blocks in `ch03_estimation.md` and `ch04_parameter_estimation.md`.

### Chapter 3 Replacements

#### Block 3.1: Empirical Semivariogram (`ch03_estimation.md`, lines 171–241)
```python
import math

def compute_empirical_semivariogram(coords, values, lag_centers, lag_bandwidth, 
                                    direction_vector=None, angle_tolerance_rad=None):
    """
    Computes Matheron's empirical semivariogram matching Definition 3.1.1
    in pure standard Python.
    
    Parameters:
        coords: list of lists, shape (n, d), spatial coordinates s_i
        values: list of floats, shape (n,), observations Z(s_i)
        lag_centers: list of floats, shape (K,), nominal lag bin centers h_k
        lag_bandwidth: float, radial width of lag bins Delta_h
        direction_vector: optional list of floats, shape (d,), unit vector u
        angle_tolerance_rad: optional float, angular half-tolerance in radians
        
    Returns:
        actual_lags: list of floats of length K, actual mean distance per bin
        gamma_hat: list of floats of length K, estimated semivariances
        pair_counts: list of ints of length K, number of pairs per bin |N(h_k)|
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    sum_sq_diff = [0.0] * K
    sum_distance = [0.0] * K
    pair_counts = [0] * K
    
    # Normalize direction vector if provided
    is_directional = direction_vector is not None and angle_tolerance_rad is not None
    if is_directional:
        norm_u = math.sqrt(sum(x * x for x in direction_vector))
        u = [x / norm_u for x in direction_vector]
        half_angle = angle_tolerance_rad / 2.0
        
    # Symmetric all-pairs traversal (unordered pairs i < j)
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Displacement vector h = s_i - s_j
            h_vec = [coords[i][m] - coords[j][m] for m in range(d)]
            dist = math.sqrt(sum(x * x for x in h_vec))
            if dist < 1e-12:
                continue
                
            # Directional verification
            if is_directional:
                dot_val = sum(h_vec[m] * u[m] for m in range(d))
                cos_theta = min(1.0, max(-1.0, abs(dot_val) / dist))
                theta = math.acos(cos_theta)
                if theta > half_angle:
                    continue
                    
            # Check membership across lag bins
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    diff = values[i] - values[j]
                    sum_sq_diff[k] += diff * diff
                    sum_distance[k] += dist
                    pair_counts[k] += 1
                    break
                    
    # Form Matheron estimator
    actual_lags = [None] * K
    gamma_hat = [None] * K
    for k in range(K):
        if pair_counts[k] > 0:
            actual_lags[k] = sum_distance[k] / pair_counts[k]
            gamma_hat[k] = sum_sq_diff[k] / (2.0 * pair_counts[k])
            
    return actual_lags, gamma_hat, pair_counts
```

#### Block 3.2: Downward Bias Computation (`ch03_estimation.md`, lines 380–444)
```python
import math

def compute_covariance_downward_bias(coords, lag_centers, lag_bandwidth, cov_func):
    """
    Evaluates the exact analytical downward bias of the empirical covariance
    estimator across nominal lags according to Theorem 3.2.2 in pure Python.
    
    Parameters:
        coords: list of lists, shape (n, d), sample sites s_i
        lag_centers: list of floats, shape (K,), nominal lags h_k
        lag_bandwidth: float, radial width Delta_h
        cov_func: callable, C(h) returning covariance for scalar distance h
        
    Returns:
        var_zbar: float, Var(Z_bar) = (1/n^2) 1' V 1
        expected_Chat: list of floats of length K, E[C_hat(h_k)]
        true_C: list of floats of length K, C(h_k)
        bias: list of floats of length K, expected_Chat - true_C
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    # 1. Compute full n x n covariance matrix V and distance matrix
    V = [[0.0] * n for _ in range(n)]
    dist_mat = [[0.0] * n for _ in range(n)]
    total_V_sum = 0.0
    
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            dist_mat[i][j] = dist
            c_val = cov_func(dist)
            V[i][j] = c_val
            total_V_sum += c_val
            
    # 2. Var(Z_bar) = (1/n^2) sum_{i,j} V_{ij}
    var_zbar = total_V_sum / (n * n)
    
    # Precompute row sums: (1/n) sum_{r=1}^n C(s_i - s_r)
    row_means_V = [sum(V[i]) / n for i in range(n)]
    
    expected_Chat = [None] * K
    true_C = [cov_func(h_k) for h_k in lag_centers]
    bias = [None] * K
    
    for k in range(K):
        h_k = lag_centers[k]
        pairs_in_bin = []
        for i in range(n):
            for j in range(n):
                if i != j and (h_k - half_bw) <= dist_mat[i][j] < (h_k + half_bw):
                    pairs_in_bin.append((i, j))
                    
        N_k = len(pairs_in_bin)
        if N_k == 0:
            continue
            
        c_bar_pair_sample = sum(row_means_V[i] for i, j in pairs_in_bin) / N_k
        exp_c = true_C[k] - 2.0 * c_bar_pair_sample + var_zbar
        expected_Chat[k] = exp_c
        bias[k] = exp_c - true_C[k]
        
    return var_zbar, expected_Chat, true_C, bias
```

#### Block 3.3: Semivariogram Covariance Matrix (`ch03_estimation.md`, lines 564–615)
```python
import math

def compute_variogram_covariance_matrix(coords, lag_bins, gamma_func):
    """
    Computes the full K x K Gaussian covariance matrix of the empirical
    semivariogram estimator matching Theorem 3.3.1 in pure Python.
    
    Parameters:
        coords: list of lists, shape (n, d), sample sites s_i
        lag_bins: list of length K, where lag_bins[k] is a list of index pairs (i, j)
        gamma_func: callable, gamma(h) returning semivariance for scalar distance h
        
    Returns:
        V_gamma: list of lists, shape (K, K), Cov(gamma_hat(h_p), gamma_hat(h_q))
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_bins)
    V_gamma = [[0.0] * K for _ in range(K)]
    
    # Precompute pairwise semivariances between all spatial locations
    gamma_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            gamma_mat[i][j] = gamma_func(dist)
            
    for p in range(K):
        pairs_p = lag_bins[p]
        N_p = len(pairs_p)
        if N_p == 0:
            continue
            
        for q in range(p, K):
            pairs_q = lag_bins[q]
            N_q = len(pairs_q)
            if N_q == 0:
                continue
                
            total_sum = 0.0
            for (i, j) in pairs_p:
                for (k, l) in pairs_q:
                    # Covariance of increments: gamma(i,l) + gamma(j,k) - gamma(i,k) - gamma(j,l)
                    cov_d = (gamma_mat[i][l] + gamma_mat[j][k] 
                             - gamma_mat[i][k] - gamma_mat[j][l])
                    total_sum += cov_d * cov_d
                    
            entry = total_sum / (2.0 * N_p * N_q)
            V_gamma[p][q] = entry
            V_gamma[q][p] = entry
            
    return V_gamma
```

#### Block 3.4: Cressie-Hawkins Semivariogram (`ch03_estimation.md`, lines 790–845)
```python
import math

def compute_cressie_hawkins_semivariogram(coords, values, lag_centers, lag_bandwidth):
    """
    Computes the robust Cressie-Hawkins modulus semivariogram matching Definition 3.4.1
    in pure standard Python.
    
    Parameters:
        coords: list of lists, shape (n, d), sample sites s_i
        values: list of floats, shape (n,), observations Z(s_i)
        lag_centers: list of floats, shape (K,), nominal lag bin centers h_k
        lag_bandwidth: float, radial width of lag bins Delta_h
        
    Returns:
        actual_lags: list of floats of length K, mean distance per bin
        gamma_ch: list of floats of length K, robust semivariance estimates
        pair_counts: list of ints of length K, pair count per bin |N(h_k)|
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    sum_root_diff = [0.0] * K
    sum_distance = [0.0] * K
    pair_counts = [0] * K
    
    # Traverse all unique pairs i < j
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    abs_diff_root = math.sqrt(abs(values[i] - values[j]))
                    sum_root_diff[k] += abs_diff_root
                    sum_distance[k] += dist
                    pair_counts[k] += 1
                    break
                    
    gamma_ch = [None] * K
    actual_lags = [None] * K
    
    for k in range(K):
        N_k = pair_counts[k]
        if N_k > 0:
            actual_lags[k] = sum_distance[k] / N_k
            mean_root = sum_root_diff[k] / N_k
            denominator = 2.0 * (0.457066 + (0.49457 / N_k))
            gamma_ch[k] = (mean_root ** 4) / denominator
            
    return actual_lags, gamma_ch, pair_counts
```

#### Block 3.5: Heterotopic Pseudo-Cross Variogram (`ch03_estimation.md`, lines 958–1013)
```python
import math

def compute_heterotopic_pseudo_cross_variogram(coords_i, values_i, coords_j, values_j, 
                                                lag_centers, lag_bandwidth):
    """
    Computes Clark's empirical pseudo-cross semivariogram matching Definition 3.5.3
    for totally heterotopic sampling geometries in pure Python.
    
    Parameters:
        coords_i: list of lists, shape (n_i, d), locations of variable i
        values_i: list of floats, shape (n_i,), measurements Z_i
        coords_j: list of lists, shape (n_j, d), locations of variable j
        values_j: list of floats, shape (n_j,), measurements Z_j
        lag_centers: list of floats, shape (K,), nominal lag bin centers h_k
        lag_bandwidth: float, radial width Delta_h
        
    Returns:
        actual_lags: list of floats of length K, mean distance per bin
        pseudo_gamma: list of floats of length K, estimated pseudo-cross semivariance
        pair_counts: list of ints of length K, count of cross-pairs
    """
    n_i = len(coords_i)
    n_j = len(coords_j)
    d = len(coords_i[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    sum_sq_diff = [0.0] * K
    sum_distance = [0.0] * K
    pair_counts = [0] * K
    
    # Bipartite traversal across disparate networks S_i and S_j
    for r in range(n_i):
        for l in range(n_j):
            dist = math.sqrt(sum((coords_i[r][m] - coords_j[l][m]) ** 2 for m in range(d)))
            
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    diff = values_i[r] - values_j[l]
                    sum_sq_diff[k] += diff * diff
                    sum_distance[k] += dist
                    pair_counts[k] += 1
                    break
                    
    pseudo_gamma = [None] * K
    actual_lags = [None] * K
    
    for k in range(K):
        if pair_counts[k] > 0:
            actual_lags[k] = sum_distance[k] / pair_counts[k]
            pseudo_gamma[k] = sum_sq_diff[k] / (2.0 * pair_counts[k])
            
    return actual_lags, pseudo_gamma, pair_counts
```

#### Block 3.6: Directional Cross-Covariance (`ch03_estimation.md`, lines 1129–1199)
```python
import math

def compute_directional_cross_covariance(coords, values_i, values_j, 
                                         lag_centers, lag_bandwidth, direction_vector):
    """
    Computes directional cross-covariances C_ij(+h) and C_ij(-h) along vector u
    to detect spatial delay / advective shifts in pure standard Python.
    
    Parameters:
        coords: list of lists, shape (n, d), shared sampling locations
        values_i: list of floats, shape (n,), measurements Z_i
        values_j: list of floats, shape (n,), measurements Z_j
        lag_centers: list of floats, shape (K,), scalar lag distances h_k
        lag_bandwidth: float, radial width Delta_h
        direction_vector: list of floats, shape (d,), orientation vector u
        
    Returns:
        signed_lags: list of floats of length 2K, signed lags [-h_K, ..., +h_K]
        C_ij: list of floats of length 2K, directional cross-covariances
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    norm_u = math.sqrt(sum(x * x for x in direction_vector))
    u = [x / norm_u for x in direction_vector]
    half_bw = lag_bandwidth / 2.0
    
    # Center variables by sample means
    mean_i = sum(values_i) / n
    mean_j = sum(values_j) / n
    z_i_c = [val - mean_i for val in values_i]
    z_j_c = [val - mean_j for val in values_j]
    
    pos_sum = [0.0] * K
    pos_count = [0] * K
    neg_sum = [0.0] * K
    neg_count = [0] * K
    
    # Directed pair traversal (ordered pairs r, l)
    for r in range(n):
        for l in range(n):
            if r == l:
                continue
            h_vec = [coords[l][m] - coords[r][m] for m in range(d)]
            dist = math.sqrt(sum(x * x for x in h_vec))
            proj = sum(h_vec[m] * u[m] for m in range(d))
            
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    prod = z_i_c[r] * z_j_c[l]
                    if proj > 0:  # s_l is downstream of s_r
                        pos_sum[k] += prod
                        pos_count[k] += 1
                    elif proj < 0:  # s_l is upstream of s_r
                        neg_sum[k] += prod
                        neg_count[k] += 1
                    break
                    
    # Assemble signed curve
    signed_lags = [-h for h in reversed(lag_centers)] + list(lag_centers)
    C_ij = [None] * (2 * K)
    
    for k in range(K):
        if neg_count[k] > 0:
            C_ij[K - 1 - k] = neg_sum[k] / neg_count[k]
        if pos_count[k] > 0:
            C_ij[K + k] = pos_sum[k] / pos_count[k]
            
    return signed_lags, C_ij
```

---

### Chapter 4 Replacements

#### Embedded Helper Routines (Pure Python Matrix Solvers)
The following compact linear algebra routines are utilized across Chapter 4 pure Python functions:
```python
import math

def cholesky_decompose(A):
    """Computes lower-triangular Cholesky factor L of positive definite matrix A."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 1e-14:
                    return None  # Non-positive definite
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def solve_triangular_lower(L, b):
    """Solves L x = b for lower triangular matrix L."""
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]
    return x

def solve_triangular_upper(U, b):
    """Solves U x = b for upper triangular matrix U."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def solve_cholesky_system(L, b):
    """Solves A x = (L L') x = b using forward and back substitution."""
    y = solve_triangular_lower(L, b)
    # L' is upper triangular: U_{ij} = L_{ji}
    n = len(L)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
    return x
```

#### Block 4.1: Gaussian Spatial Log-Likelihood (`ch04_parameter_estimation.md`, lines 251–299)
```python
def compute_gaussian_log_likelihood(Z, X, beta, theta, coords, cov_func):
    """
    Evaluates the Gaussian spatial log-likelihood matching Definition 4.1.1
    using stable pure Python Cholesky decomposition.
    
    Parameters:
        Z: list of floats, shape (n,), observations
        X: list of lists, shape (n, p), design matrix
        beta: list of floats, shape (p,), drift parameters
        theta: tuple (sigma2, phi, tau2), covariance parameters
        coords: list of lists, shape (n, d), spatial sites s_i
        cov_func: callable, C(h, phi) returning correlation for scalar lag h
        
    Returns:
        log_lik: float, Gaussian log-likelihood value
    """
    n = len(Z)
    p = len(beta)
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Assemble covariance matrix V(theta)
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            r_val = cov_func(dist, phi)
            V[i][j] = sigma2 * r_val + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization: V = L L'
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    # 3. Log-determinant via Cholesky diagonal
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 4. Residual vector: e = Z - X beta
    residuals = [Z[i] - sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    
    # 5. Solve L w = e by forward substitution
    w = solve_triangular_lower(L, residuals)
    quad_form = sum(val * val for val in w)
    
    # 6. Gaussian log-likelihood
    log_lik = -0.5 * (n * math.log(2.0 * math.pi) + log_det_V + quad_form)
    return log_lik
```

#### Block 4.2: Profile ML Log-Likelihood (`ch04_parameter_estimation.md`, lines 397–438)
```python
def compute_profile_ml_log_likelihood(theta, X, Z, coords, cov_func):
    """
    Evaluates the profile Maximum Likelihood log-likelihood ell_p(theta)
    matching Definition 4.2.1 in pure standard Python.
    """
    n = len(Z)
    p = len(X[0])
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Covariance matrix assembly
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            V[i][j] = sigma2 * cov_func(dist, phi) + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization of V
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 3. Solve L W_x = X and L u_z = Z
    # W_x has dimensions (n, p), columns solved separately
    W_x = [[0.0] * p for _ in range(n)]
    for j in range(p):
        col_x = [X[i][j] for i in range(n)]
        sol_col = solve_triangular_lower(L, col_x)
        for i in range(n):
            W_x[i][j] = sol_col[i]
            
    u_z = solve_triangular_lower(L, Z)
    
    # 4. Form information matrix G = X' V^-1 X = W_x' W_x
    G = [[sum(W_x[k][i] * W_x[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    rhs = [sum(W_x[k][i] * u_z[k] for k in range(n)) for i in range(p)]
    
    # 5. Factorize G and solve for beta_GLS
    L_g = cholesky_decompose(G)
    if L_g is None:
        return -math.inf
    beta_gls = solve_cholesky_system(L_g, rhs)
    
    # 6. Residual quadratic form: ||u_z - W_x beta_gls||^2
    quad_form_P = 0.0
    for i in range(n):
        res_i = u_z[i] - sum(W_x[i][j] * beta_gls[j] for j in range(p))
        quad_form_P += res_i * res_i
        
    # 7. Profile log-likelihood
    log_lik_p = -0.5 * (n * math.log(2.0 * math.pi) + log_det_V + quad_form_P)
    return log_lik_p
```

#### Block 4.3: REML Log-Likelihood (`ch04_parameter_estimation.md`, lines 561–622)
```python
def compute_reml_log_likelihood(theta, X, Z, coords, cov_func):
    """
    Evaluates Harville's REML log-likelihood ell_REML(theta) matching Theorem 4.3.3
    in pure standard Python.
    """
    n = len(Z)
    p = len(X[0])
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Covariance matrix assembly
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            V[i][j] = sigma2 * cov_func(dist, phi) + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization of V
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 3. Solve L W_x = X and L u_z = Z
    W_x = [[0.0] * p for _ in range(n)]
    for j in range(p):
        col_x = [X[i][j] for i in range(n)]
        sol_col = solve_triangular_lower(L, col_x)
        for i in range(n):
            W_x[i][j] = sol_col[i]
            
    u_z = solve_triangular_lower(L, Z)
    
    # 4. Form G = X' V^-1 X = W_x' W_x
    G = [[sum(W_x[k][i] * W_x[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    rhs = [sum(W_x[k][i] * u_z[k] for k in range(n)) for i in range(p)]
    
    # 5. Factorize G to compute log|G| and beta_GLS
    L_g = cholesky_decompose(G)
    if L_g is None:
        return -math.inf
        
    log_det_G = 2.0 * sum(math.log(L_g[i][i]) for i in range(p))
    beta_gls = solve_cholesky_system(L_g, rhs)
    
    # 6. Projected residual quadratic form: Z' P Z = ||u_z - W_x beta_gls||^2
    quad_form_P = 0.0
    for i in range(n):
        res_i = u_z[i] - sum(W_x[i][j] * beta_gls[j] for j in range(p))
        quad_form_P += res_i * res_i
        
    # 7. Harville REML log-likelihood
    reml_log_lik = -0.5 * ((n - p) * math.log(2.0 * math.pi) 
                           + log_det_V 
                           + log_det_G 
                           + quad_form_P)
    return reml_log_lik
```

#### Block 4.4: Cressie WLS Variogram Fitter (`ch04_parameter_estimation.md`, lines 712–758)
```python
import math

def fit_variogram_cressie_wls(lags, gamma_hat, pair_counts, initial_theta, model_type='exponential'):
    """
    Fits parametric semivariogram parameters via Cressie's WLS criterion
    matching Definition 4.4.1 using a pure Python Nelder-Mead simplex optimizer.
    
    Parameters:
        lags: list of floats of length K, lag distances h_k
        gamma_hat: list of floats of length K, empirical semivariances
        pair_counts: list of ints of length K, pair counts |N(h_k)|
        initial_theta: tuple (sigma2_0, phi_0, tau2_0), starting guess
        model_type: str, 'exponential' or 'spherical'
        
    Returns:
        opt_theta: tuple (sigma2, phi, tau2), fitted parameters
    """
    def theoretical_gamma(h, theta):
        sigma2, phi, tau2 = theta
        if model_type == 'exponential':
            return tau2 + sigma2 * (1.0 - math.exp(-h / phi))
        elif model_type == 'spherical':
            hr = h / phi
            sph = 1.5 * hr - 0.5 * (hr ** 3) if hr <= 1.0 else 1.0
            return tau2 + sigma2 * sph
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def wls_objective(theta):
        sigma2, phi, tau2 = theta
        if sigma2 <= 1e-6 or phi <= 1e-6 or tau2 < 0.0:
            return 1e12  # Strict positivity penalty
            
        wls_val = 0.0
        for h_k, g_hat, n_k in zip(lags, gamma_hat, pair_counts):
            g_theo = theoretical_gamma(h_k, theta)
            g_theo = max(g_theo, 1e-10)
            rel_diff = (g_hat / g_theo) - 1.0
            wls_val += n_k * (rel_diff * rel_diff)
        return wls_val

    # Pure Python Nelder-Mead simplex optimization
    dim = len(initial_theta)
    sim = [list(initial_theta)]
    for i in range(dim):
        p = list(initial_theta)
        p[i] = p[i] * 1.1 if p[i] != 0.0 else 0.05
        sim.append(p)
        
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(500):
        sim.sort(key=wls_objective)
        if abs(wls_objective(sim[-1]) - wls_objective(sim[0])) < 1e-7:
            break
            
        x_o = [sum(sim[i][d] for i in range(dim)) / dim for d in range(dim)]
        x_r = [x_o[d] + alpha * (x_o[d] - sim[-1][d]) for d in range(dim)]
        f_r = wls_objective(x_r)
        
        if wls_objective(sim[0]) <= f_r < wls_objective(sim[-2]):
            sim[-1] = x_r
            continue
        if f_r < wls_objective(sim[0]):
            x_e = [x_o[d] + gamma * (x_r[d] - x_o[d]) for d in range(dim)]
            sim[-1] = x_e if wls_objective(x_e) < f_r else x_r
            continue
        x_c = [x_o[d] + rho * (sim[-1][d] - x_o[d]) for d in range(dim)]
        if wls_objective(x_c) < wls_objective(sim[-1]):
            sim[-1] = x_c
            continue
        x_best = sim[0]
        for i in range(1, len(sim)):
            sim[i] = [x_best[d] + sigma * (sim[i][d] - x_best[d]) for d in range(dim)]
            
    return tuple(sim[0])
```

#### Block 4.5: Bivariate Spectral Admissibility (`ch04_parameter_estimation.md`, lines 825–848)
```python
import math

def verify_bivariate_spectral_admissibility(f11_func, f22_func, f12_func, omega_grid):
    """
    Verifies Cramér's positive semi-definiteness condition across frequencies
    matching Corollary 4.5.1 in pure standard Python.
    """
    admissible = True
    min_det = math.inf
    
    for omega in omega_grid:
        f11 = f11_func(omega)
        f22 = f22_func(omega)
        f12 = f12_func(omega)
        
        # Absolute square: |f12|^2 = Re(f12)^2 + Im(f12)^2
        f12_abs_sq = (f12.real ** 2 + f12.imag ** 2) if isinstance(f12, complex) else (f12 * f12)
        det_f = f11 * f22 - f12_abs_sq
        
        if det_f < min_det:
            min_det = det_f
        if det_f < -1e-12 or f11 < 0.0 or f22 < 0.0:
            admissible = False
            
    return admissible, min_det
```

#### Block 4.6: LMC Positive Semi-Definite Projector (`ch04_parameter_estimation.md`, lines 925–959)
```python
import math

def project_lmc_positive_semidefinite(candidate_B_matrices, min_eigenvalue=1e-8):
    """
    Projects candidate LMC coregionalization matrices onto the cone of 
    positive semi-definite matrices matching Theorem 4.6.2 using a pure Python
    cyclic Jacobi eigenvalue decomposition.
    
    Parameters:
        candidate_B_matrices: list of symmetric matrices B_u in R^(m x m)
        min_eigenvalue: float, floor for eigenvalues (Tikhonov regularizer)
        
    Returns:
        projected_B: list of positive semi-definite matrices B_u*
    """
    def jacobi_eigh(A, tol=1e-12, max_iter=100):
        m = len(A)
        V = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
        D = [[A[i][j] for j in range(m)] for i in range(m)]
        
        for _ in range(max_iter):
            max_val = 0.0
            p, q = 0, 1
            for i in range(m):
                for j in range(i + 1, m):
                    if abs(D[i][j]) > max_val:
                        max_val = abs(D[i][j])
                        p, q = i, j
            if max_val < tol:
                break
                
            diff = D[q][q] - D[p][p]
            if abs(D[p][q]) < 1e-15:
                t = 0.0
            else:
                phi = diff / (2.0 * D[p][q])
                t = math.copysign(1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0)), phi)
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            
            D_pp = D[p][p]
            D_qq = D[q][q]
            D_pq = D[p][q]
            D[p][p] = c * c * D_pp - 2.0 * s * c * D_pq + s * s * D_qq
            D[q][q] = s * s * D_pp + 2.0 * s * c * D_pq + c * c * D_qq
            D[p][q] = 0.0
            D[q][p] = 0.0
            
            for k in range(m):
                if k != p and k != q:
                    d_kp = D[k][p]
                    d_kq = D[k][q]
                    D[k][p] = c * d_kp - s * d_kq
                    D[p][k] = D[k][p]
                    D[k][q] = s * d_kp + c * d_kq
                    D[q][k] = D[k][q]
                    
            for k in range(m):
                v_kp = V[k][p]
                v_kq = V[k][q]
                V[k][p] = c * v_kp - s * v_kq
                V[k][q] = s * v_kp + c * v_kq
                
        eigenvalues = [D[i][i] for i in range(m)]
        # V columns are the orthonormal eigenvectors
        return eigenvalues, V

    projected_B = []
    for B in candidate_B_matrices:
        m = len(B)
        # Symmetrize
        B_sym = [[0.5 * (B[i][j] + B[j][i]) for j in range(m)] for i in range(m)]
        
        # Spectral decomposition: B = Q Lambda Q'
        eigs, Q = jacobi_eigh(B_sym)
        
        # Eigenvalue clipping: lambda_j^+ = max(min_eigenvalue, lambda_j)
        eigs_clipped = [max(min_eigenvalue, val) for val in eigs]
        
        # Reconstruct B* = Q Lambda^+ Q' = sum lambda_k q_k q_k'
        B_proj = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                B_proj[i][j] = sum(eigs_clipped[k] * Q[i][k] * Q[j][k] for k in range(m))
                
        # Symmetrize reconstruction
        B_proj = [[0.5 * (B_proj[i][j] + B_proj[j][i]) for j in range(m)] for i in range(m)]
        projected_B.append(B_proj)
        
    return projected_B
```

---

## Action Plan for Remediation Phase

When the editing subagent executes the remediations in Batch 5.2, follow this sequenced execution checklist:

1. **Chapter 3 (`manuscript/ch03_estimation.md`):**
   - [ ] Replace all 6 Python code blocks with pure standard library implementations (Blocks 3.1 to 3.6).
   - [ ] Correct FAQ 1 formula variance typo on line 764: change $\frac{6}{N}(\operatorname{E}[\bar{Y}])^2 \operatorname{Var}(\bar{Y})$ to $\frac{6}{N}(\operatorname{E}[Y])^2 \operatorname{Var}(Y)$.
   - [ ] Remove duplicate prose word on line 255: `"erratic erratic"` $\to$ `"erratic"`.

2. **Chapter 4 (`manuscript/ch04_parameter_estimation.md`):**
   - [ ] Replace all 6 Python code blocks with pure standard library implementations (Blocks 4.1 to 4.6).
   - [ ] Rewrite Harville REML conditioning argument (lines 501–515) to use $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and error contrast block-independence.
   - [ ] Standardize notation drift: convert isolated $\mathbf{Y}$ references to $\mathbf{Z}$ across definitions and anatomy boxes (lines 33, 132, 524).

3. **Chapter 3 Exercises (`manuscript/ch03_exercises.md`):**
   - [ ] Fix intermediate subtraction typo in Exercise 3.8(d) solution on line 596: change $0.456947$ to $(0.822179)^2 \approx 0.675978$.

4. **Chapter 4 Exercises (`manuscript/ch04_exercises.md`):**
   - [ ] Wrap unescaped `\blacksquare` in math delimiters on line 396: change `\blacksquare` to `$\blacksquare$`.

5. **Appendix D (`manuscript/app_d.md`):**
   - [ ] Maintain as pristine (zero changes needed).

---
*Report compiled and certified by Reviewer Subagent 2.*
