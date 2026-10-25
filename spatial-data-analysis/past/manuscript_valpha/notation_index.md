# Master Notation Index and Mathematical Conventions

This document establishes the invariant mathematical notation, operator definitions, typographic standards, and algorithmic conventions for the graduate monograph:
**"Theoretical Foundations of Spatial Statistics and Geostatistics"**.

The monograph combines the pedagogical clarity and procedural transparency of **Maria L. Rizzo** with the semantic linear-model discipline and spatial indexing of **Oliver Schabenberger & Carol A. Gotway** (*Statistical Methods for Spatial Data Analysis*).

All chapters (`ch01_foundations.md` through `ch13_nonstationary_covariance.md`) and appendices (`app_a.md` through `app_d.md`) strictly adhere to the standards codified below.

---

### 1. Typographic & Delimiter Standards

1. **LaTeX Separation:**
   - Markdown is reserved strictly for document structuring (`#`, `##`, `###`, lists, bold, italics).
   - LaTeX is used exclusively for all mathematical symbols, scalar constants, variables, operators, sets, and indices.
   - Never use Unicode or plain-text approximations in mathematical text (e.g., write `\mathbb{R}^d`, `\in`, `\sigma^2`, `\gamma(\mathbf{h})`, `\le`, `\ge`, `\ne`; never write `ℝ^d`, `∈`, `σ²`, `γ(h)`, `<=`, `>=`).

2. **Delimiters & Whitespace:**
   - **Inline Math:** Wrapped in single dollar signs with zero inner whitespace padding (e.g., `$\mathbf{s} \in D$`, `$Z(\mathbf{s})$`).
   - **Display / Block Math:** Wrapped in double dollar signs placed on their own dedicated lines:
     $$
     \gamma(\mathbf{h}) = \frac{1}{2} \operatorname{Var}(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s}))
     $$

3. **Matrix–Vector Hierarchy (Schabenberger & Gotway):**
   - **Lower-case bold** for vectors: $\mathbf{s}, \mathbf{h}, \mathbf{y}, \mathbf{z}, \boldsymbol{\beta}, \boldsymbol{\lambda}, \boldsymbol{\theta}, \boldsymbol{\epsilon}$.
   - **Upper-case bold** for matrices: $\mathbf{X}, \mathbf{V}, \mathbf{G}, \mathbf{\Sigma}, \mathbf{I}_n$.
   - **Math italic** for scalars and random variables: $Z(\mathbf{s}), \sigma^2, \phi, \tau^2, c_0, c$.

4. **Algorithmic Pseudocode (Rizzo Style):**
   - Presented in clean, readable Pythonic mathematical pseudocode.
   - Variable names match their mathematical symbols directly (`beta_gls`, `V_inv`, `gamma_h`, `lag_h`, `dist_matrix`).
   - Zero black-box third-party libraries; explicit procedural steps matching the analytical derivations.

---

### 2. Spatial Indexing and the Three-Layer Decomposition

| Concept | LaTeX Syntax | Formal Definition / Context |
| :--- | :--- | :--- |
| Spatial Domain | `$D \subset \mathbb{R}^d$` | Non-empty continuous subset of Euclidean space $\mathbb{R}^d$ ($d \ge 1$) |
| Continuous Spatial Coordinate | `$\mathbf{s} = (s_1, \dots, s_d)' \in D$` | Vector of spatial coordinates in $\mathbb{R}^d$ |
| Discrete Sample Locations | `$\mathbf{s}_1, \dots, \mathbf{s}_n \in D$` | $n$ observed spatial sites |
| Spatial Separation / Lag Vector | `$\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j \in \mathbb{R}^d$` | Coordinate separation vector |
| Scalar Euclidean Distance | `$h = \|\mathbf{h}\| = \sqrt{\mathbf{h}'\mathbf{h}}$` | Distance lag $\|\mathbf{s}_i - \mathbf{s}_j\|$ |
| Random Field | `$\{Z(\mathbf{s}) : \mathbf{s} \in D\}$` | Continuous-index spatial stochastic process |
| **Three-Layer Decomposition** | `$Z(\mathbf{s}) = \mu(\mathbf{s}) + \delta(\mathbf{s}) + \epsilon(\mathbf{s})$` | Large-scale trend + spatially correlated smooth random effect + microscale white noise |
| Linear Trend / Drift | `$\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$` | Deterministic mean function with spatial basis vector $\mathbf{x}(\mathbf{s}) \in \mathbb{R}^p$ |
| Correlated Spatial Effect | `$\delta(\mathbf{s})$` | Zero-mean second-order process with covariance $C(\mathbf{h}) = \operatorname{Cov}(\delta(\mathbf{s}), \delta(\mathbf{s}+\mathbf{h}))$ |
| Microscale White Noise (Nugget) | `$\epsilon(\mathbf{s})$` | Uncorrelated measurement error / micro-variability with $\operatorname{Var}(\epsilon(\mathbf{s})) = \tau^2$ |

---

### 3. Second-Order Moments & Spatial Structure Functions

| Operator / Function | LaTeX Syntax | Definition / Linear Model Context |
| :--- | :--- | :--- |
| Expectation Operator | `\operatorname{E}[X]` | Population mean $\int_\Omega X(\omega) dP(\omega)$ |
| Variance Operator | `\operatorname{Var}(X)` | $\operatorname{E}[(X - \operatorname{E}[X])^2]$ |
| Covariance Operator | `\operatorname{Cov}(X, Y)` | $\operatorname{E}[(X - \operatorname{E}[X])(Y - \operatorname{E}[Y])]$ |
| Spatial Covariance Kernel | `$C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h}))$` | Covariance between locations separated by lag $\mathbf{h}$ |
| Spatial Semivariogram | `$\gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))$` | Half the increment variance under intrinsic stationarity |
| Covariance Parameters | `$\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$` | Structural parameters: partial sill $\sigma^2$, spatial range $\phi$, nugget effect $\tau^2$ |
| Total Sill | `$\sigma^2 + \tau^2 = C(\mathbf{0})$` | Global variance under second-order stationarity |
| Sample Semivariogram | `$\widehat{\gamma}(\mathbf{h})$` | Matheron's empirical method-of-moments estimator |
| Robust Semivariogram | `$\bar{\gamma}_{CH}(\mathbf{h})$` | Cressie-Hawkins fourth-root modulus estimator |

---

### 4. Linear Model & Kriging Grammar (GLM / BLUP)

| Entity | LaTeX Syntax | Algebraic Dimension / Definition |
| :--- | :--- | :--- |
| Observed Data Vector | `$\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$` | Column vector in $\mathbb{R}^n$ |
| Spatial Design Matrix | `$\mathbf{X} = (\mathbf{x}(\mathbf{s}_1), \dots, \mathbf{x}(\mathbf{s}_n))'$` | Known matrix in $\mathbb{R}^{n \times p}$ ($p < n$) |
| Regression Coefficients | `$\boldsymbol{\beta} \in \mathbb{R}^p$` | Large-scale trend parameters |
| Covariance Matrix | `$\mathbf{V}(\boldsymbol{\theta}) = \operatorname{Var}(\mathbf{Z}) = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$` | Symmetric, positive definite matrix in $\mathbb{R}^{n \times n}$ |
| Point-to-Data Covariance Vector | `$\mathbf{v}_0 = (C(\mathbf{s}_0 - \mathbf{s}_1), \dots, C(\mathbf{s}_0 - \mathbf{s}_n))'$` | Vector in $\mathbb{R}^n$ |
| Kriging Weight Vector | `$\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)'$` | BLUP weights in $\mathbb{R}^n$ |
| Best Linear Unbiased Predictor | `$\widehat{Z}(\mathbf{s}_0) = \boldsymbol{\lambda}'\mathbf{Z}$` | Minimum mean-squared error linear predictor |
| GLS Trend Estimator | `$\widehat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z}$` | Best Linear Unbiased Estimator (BLUE) of drift |
| Lagrange Multiplier | `$\mu$` or `$\boldsymbol{\mu} \in \mathbb{R}^p$` | Equality constraint multipliers for unbiasedness |
| Kriging Prediction Variance | `$\sigma_{\text{OK}}^2(\mathbf{s}_0)$` or `$\sigma_{\text{SK}}^2(\mathbf{s}_0)$` | Minimized mean-squared prediction error |

---

### 5. The Three-Beat Pedagogical Rhythm (Rizzo Template)

Every major theoretical concept across the monograph is presented through three coordinated beats:

```
+-------------------------------------------------------------------------------+
| BEAT 1: THE MATHEMATICAL ANCHOR (THE "WHY")                                   |
| - Stated in clean, rigorous linear-model notation.                            |
| - Identifies the concrete statistical dilemma before introducing the theorem. |
| - Unabridged step-by-step mathematical proof.                                 |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
| BEAT 2: THE ALGORITHMIC TRANSLATION (THE "HOW")                               |
| - Procedural breakdown: Step 1, Step 2, Step 3...                             |
| - Transparent Pythonic pseudocode with 1-to-1 variable-to-formula naming.     |
| - Zero black-box libraries: clear loops, matrix inverses, and projections.    |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-------------------------------------------------------------------------------+
| BEAT 3: INTUITION, EDGE CASES & DIAGNOSTIC VERIFICATION (THE "REALITY")       |
| - Geometric and physical interpretation (why weights go negative, why         |
|   covariance curves sag, why smoothing flattens variance).                    |
| - Numerical stability checks: condition numbers, singularity, edge effects.   |
+-------------------------------------------------------------------------------+
```

---

### 6. Mapped Point Patterns and Spatial Point Processes (Chapter 11)

| Entity / Operator | LaTeX Syntax | Formal Mathematical Context |
| :--- | :--- | :--- |
| Observation Window | `$W \subset \mathbb{R}^d$` | Compact, bounded spatial window ($|W| = \int_W d\mathbf{s} > 0$) |
| Point Process Realization | `$\mathbf{x} = \{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subset W$` | Random configuration of discrete events in $W$ |
| Random Counting Measure | `$N(A)$` | Number of events located within Borel set $A \in \mathcal{B}(W)$ |
| Intensity Measure | `$\Lambda(A) = \operatorname{E}[N(A)] = \int_A \lambda(\mathbf{s}) d\mathbf{s}$` | Expected point count across spatial region $A$ |
| First-Order Intensity | `$\lambda(\mathbf{s}) = \lim_{|d\mathbf{s}| \to 0} \frac{\operatorname{E}[N(d\mathbf{s})]}{|d\mathbf{s}|}$` | Point generation rate per unit volume at location $\mathbf{s}$ |
| Second-Order Product Density | `$\lambda_2(\mathbf{u}, \mathbf{v})$` | Joint probability density of finding events at both $\mathbf{u}$ and $\mathbf{v}$ |
| Pair Correlation Function | `$g(r) = \frac{\lambda_2(r)}{\lambda^2}$` | Normalized event interaction ratio at distance $r$ ($g \equiv 1$ under CSR) |
| Campbell's Theorem | `$\operatorname{E}\left[\sum_{\mathbf{x} \in X} f(\mathbf{x})\right] = \int_W f(\mathbf{s}) \lambda(\mathbf{s}) d\mathbf{s}$` | Expectation of spatial point process functional sums |
| Ripley's $K$-Function | `$K(r) = \frac{1}{\lambda}\operatorname{E}[\text{extra events within distance } r \text{ of an arbitrary event}]$` | Second-order cumulative inter-point distance measure ($K(r) = \pi r^2$ in $\mathbb{R}^2$) |
| Besag's $L$-Function | `$L(r) = \sqrt{\frac{K(r)}{\pi}}$` | Variance-stabilized transformation ($L(r) - r \equiv 0$ under CSR) |
| Ripley Edge-Correction Weight | `$w(\mathbf{s}_i, \mathbf{s}_j) \in [0, 1]$` | Proportion of circle perimeter centered at $\mathbf{s}_i$ with radius $\|\mathbf{s}_i - \mathbf{s}_j\|$ inside $W$ |
| Nearest-Neighbor Distribution | `$G(r) = P(d(\mathbf{s}_i, X \setminus \{\mathbf{s}_i\}) \le r)$` | Event-to-event nearest neighbor CDF ($G_{\text{CSR}}(r) = 1 - e^{-\lambda \pi r^2}$) |
| Empty-Space Function | `$F(r) = P(d(\mathbf{u}, X) \le r)$` | Arbitrary grid point-to-event hazard CDF ($F_{\text{CSR}}(r) = 1 - e^{-\lambda \pi r^2}$) |
| Van Lieshout–Baddeley Index | `$J(r) = \frac{1 - G(r)}{1 - F(r)}$` | Clustering ($J < 1$) vs. CSR ($J \equiv 1$) vs. regularity ($J > 1$) summary metric |
| Log-Gaussian Cox Process | `$\Lambda(\mathbf{s}) = \exp(Y(\mathbf{s})), \ Y(\mathbf{s}) \sim \mathcal{GP}(\mu, C)$` | Doubly stochastic point process driven by continuous latent random field |

---

### 7. Areal Lattices, Spatial Autocorrelation & Spatial GLMMs (Chapter 12)

| Entity / Operator | LaTeX Syntax | Formal Mathematical Context |
| :--- | :--- | :--- |
| Spatial Adjacency Matrix | `$\mathbf{W} = [w_{ij}]_{i,j=1}^n, \quad w_{ii} = 0$` | Binary or distance-weighted areal proximity matrix ($w_{ij} > 0$ if $i \sim j$) |
| Diagonal Node Degree Matrix | `$\mathbf{D} = \operatorname{diag}(w_{i+}) = \operatorname{diag}\left(\sum_{j=1}^n w_{ij}\right)$` | Diagonal sum matrix of neighborhood degrees |
| Row-Standardized Matrix | `$\mathbf{W}^* = \mathbf{D}^{-1}\mathbf{W}$` | Normalized spatial weight matrix ($\sum_j w_{ij}^* = 1$) |
| Graph Laplacian Matrix | `$\mathbf{L} = \mathbf{D} - \mathbf{W}$` | Symmetric positive semidefinite spatial difference operator ($\mathbf{x}'\mathbf{L}\mathbf{x} = \frac{1}{2}\sum w_{ij}(x_i-x_j)^2$) |
| Centering Projection Matrix | `$\mathbf{P} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'$` | Orthogonal projection operator annihilating constant vector $\mathbf{1}$ |
| Moran's $I$ Statistic | `$I = \frac{n}{S_0} \frac{\mathbf{Y}'\mathbf{P}\mathbf{W}\mathbf{P}\mathbf{Y}}{\mathbf{Y}'\mathbf{P}\mathbf{Y}}$` | Global product-moment spatial autocorrelation index (expected value $-\frac{1}{n-1}$) |
| Geary's $c$ Statistic | `$c = \frac{n-1}{2 S_0} \frac{\sum_i \sum_j w_{ij}(Y_i - Y_j)^2}{\sum_i (Y_i - \bar{Y})^2} = \frac{n-1}{S_0}\frac{\mathbf{Y}'\mathbf{P}\mathbf{L}\mathbf{P}\mathbf{Y}}{\mathbf{Y}'\mathbf{P}\mathbf{Y}}$` | Global paired-difference spatial autocorrelation index (expected value $1.0$) |
| Local Moran's $I_i$ (LISA) | `$I_i = \frac{Y_i - \bar{Y}}{m_2} \sum_{j=1}^n w_{ij}(Y_j - \bar{Y})$` | Local Indicator of Spatial Association identifying spatial clusters/outliers |
| SAR Model (Simultaneous) | `$(\mathbf{I} - \rho \mathbf{W})\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$` | Spatial lag linear model ($\boldsymbol{\Sigma}_{\text{SAR}} = \sigma^2 (\mathbf{I} - \rho \mathbf{W})^{-1}(\mathbf{I} - \rho \mathbf{W}')^{-1}$) |
| CAR Model (Conditional) | `$Y_i \mid \mathbf{Y}_{-i} \sim \mathcal{N}\left(\mathbf{x}_i'\boldsymbol{\beta} + \sum_j c_{ij}(Y_j - \mathbf{x}_j'\boldsymbol{\beta}), \tau_i^2\right)$` | Spatial Markov random field ($\boldsymbol{\Sigma}_{\text{CAR}} = (\mathbf{I} - \mathbf{C})^{-1}\mathbf{M}$) |
| CAR Precision Matrix | `$\mathbf{Q} = \boldsymbol{\Sigma}_{\text{CAR}}^{-1} = \mathbf{M}^{-1}(\mathbf{I} - \mathbf{C})$` | Sparse precision matrix satisfying Brook's Lemma symmetry $c_{ij}/\tau_i^2 = c_{ji}/\tau_j^2$ |
| Intrinsic CAR (ICAR) | `$\mathbf{Q}_{\text{ICAR}} = \tau (\mathbf{D} - \mathbf{W}) = \tau \mathbf{L}$` | Improper spatial prior invariant to arbitrary additive shifts ($\mathbf{Q}\mathbf{1} = \mathbf{0}$) |
| Besag–York–Mollié (BYM) | `$\eta_i = \mathbf{x}_i'\boldsymbol{\beta} + u_i + v_i$` | Disease mapping GLMM: $u_i \sim \text{ICAR}$ (structured) and $v_i \sim \mathcal{N}(0, \sigma_v^2)$ (unstructured) |
| GWR Local Weight Matrix | `$\mathbf{W}(\mathbf{s}_0) = \operatorname{diag}\left(K\left(\frac{\|\mathbf{s}_i - \mathbf{s}_0\|}{b}\right)\right)$` | Kernel diagonal weight matrix for Geographically Weighted Regression |

---

### 8. Non-Stationary Covariance and Scalable Low-Rank Architectures (Chapter 13)

| Entity / Operator | LaTeX Syntax | Formal Mathematical Context |
| :--- | :--- | :--- |
| Non-Stationary Kernel | `$C(\mathbf{s}_i, \mathbf{s}_j) = \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j))$` | General spatial covariance kernel depending explicitly on locations $(\mathbf{s}_i, \mathbf{s}_j)$ |
| Local Metric Tensor | `$\boldsymbol{\Sigma}(\mathbf{s}) \in \mathbb{R}^{d \times d}, \quad \boldsymbol{\Sigma}(\mathbf{s}) \succ \mathbf{0}$` | Spatially varying covariance matrix governing local range and anisotropy orientation |
| Mahalanobis Separation Metric | `$Q_{ij} = (\mathbf{s}_i - \mathbf{s}_j)' \left(\frac{\boldsymbol{\Sigma}(\mathbf{s}_i) + \boldsymbol{\Sigma}(\mathbf{s}_j)}{2}\right)^{-1} (\mathbf{s}_i - \mathbf{s}_j)$` | Spatially averaged quadratic form between sampling locations $\mathbf{s}_i$ and $\mathbf{s}_j$ |
| Paciorek–Schervish Correlation | `$R_{\text{NS}}(\mathbf{s}_i, \mathbf{s}_j) = \|\boldsymbol{\Sigma}_i\|^{1/4}\|\boldsymbol{\Sigma}_j\|^{1/4} \left\|\frac{\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j}{2}\right\|^{-1/2} M_\nu(\sqrt{Q_{ij}})$` | Closed-form valid positive-definite non-stationary correlation function |
| Space Deformation Mapping | `$f: D \subset \mathbb{R}^d \to \tilde{D} \subset \mathbb{R}^{d^*}$` | Sampson–Guttorp bijective spatial coordinate warping: $C(\mathbf{s}_i, \mathbf{s}_j) = C_0(\|f(\mathbf{s}_i) - f(\mathbf{s}_j)\|)$ |
| Spatial Basis Vector | `$\mathbf{S}(\mathbf{s}) = (S_1(\mathbf{s}), \dots, S_r(\mathbf{s}))' \in \mathbb{R}^r$` | Vector of $r$ multi-scale spatial basis functions ($r \ll n$, e.g., local bisquare functions) |
| Random Basis Coefficients | `$\boldsymbol{\eta} \sim \mathcal{N}_r(\mathbf{0}, \mathbf{K}), \quad \mathbf{K} \in \mathbb{R}^{r \times r}$` | Latent low-rank spatial random vector with prior covariance $\mathbf{K} \succeq \mathbf{0}$ |
| Fixed Rank Kriging Model | `$Z(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \mathbf{S}(\mathbf{s})'\boldsymbol{\eta} + \epsilon(\mathbf{s})$` | Cressie & Johannesson scalable spatial linear model with diagonal error $\mathbf{D}_\epsilon = \sigma_\epsilon^2 \mathbf{I}_n$ |
| Low-Rank Spatial Covariance | `$\mathbf{V} = \mathbf{S}\mathbf{K}\mathbf{S}' + \mathbf{D}_\epsilon$` | $n \times n$ structured covariance matrix decomposable into low-rank plus diagonal terms |
| Sherman–Morrison–Woodbury Inversion | `$\mathbf{V}^{-1} = \mathbf{D}_\epsilon^{-1} - \mathbf{D}_\epsilon^{-1}\mathbf{S}(\mathbf{K}^{-1} + \mathbf{S}'\mathbf{D}_\epsilon^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{D}_\epsilon^{-1}$` | Fast linear system solver reducing computational complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n r^2 + r^3)$ |
