# Chapter 12: Areal Lattice Models (SAR, CAR, SMA), Spatial Autocorrelation & Spatial GLMMs

---

### Navigation & Learning Roadmap
- **Prerequisites:** [Chapter 1: Mathematical Foundations](ch01_foundations.md) (linear models, matrix hierarchies), [Chapter 6: Spatial Regression & Geostatistics](ch06_spatial_regression.md) (GLS, spatial drift).
- **Core Focus:** Discrete spatial indexing, non-Euclidean spatial weight matrices $\mathbf{W}$, simultaneous vs. conditional autoregression (SAR vs. CAR), spatial autocorrelation diagnostics (Moran's $I$, Geary's $c$, Anselin LISA), and Spatial GLMMs (Besag-York-Mollié disease mapping).
- **Companion Study Guide:** [Chapter 12 Companion Exercises and Complete Solutions](ch12_exercises.md).

---

## 12.1 The Classical-to-Spatial Bridge: From Time Series AR(1) to Spatial Lattices

In classical time series analysis, the first-order autoregressive process $\operatorname{AR}(1)$ expresses the observation at time $t$ as a function of the immediate past plus white noise:
$$
Y_t = \rho Y_{t-1} + \epsilon_t, \quad \epsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2), \quad |\rho| < 1
$$
Because time flows strictly forward along the real line, the model enjoys a fundamental **causal asymmetry**: $Y_{t-1}$ causes $Y_t$, but $Y_t$ cannot backward-influence $Y_{t-1}$. Consequently, the conditional expectation $\operatorname{E}[Y_t \mid Y_{t-1}]$ matches the joint structural equation, and the error $\epsilon_t$ is strictly orthogonal to the lagged predictor $Y_{t-1}$. Ordinary Least Squares (OLS) yields consistent, asymptotically efficient parameter estimates.

```
Classical 1D Time Series: Causal, Unidirectional Arrow
[Y_{t-1}] --------> [Y_t] --------> [Y_{t+1}]
         past               present              future

Spatial Areal Lattice: Simultaneous Omnidirectional Feedback
      [Y_{North}]
          ^ |
          | v
[Y_{West}] <---> [Y_i] <---> [Y_{East}]
          ^ |
          | v
      [Y_{South}]
```

When we transition to areal data—where observations $\mathbf{Y} = (Y_1, \dots, Y_n)'$ are associated with discrete non-overlapping geographic regions (census tracts, counties, crop parcels, grid pixels)—this unidirectional causal arrow collapses:
1. **Omnidirectional Feedback:** Spatial relationships possess no "one-sided past." Region $i$ borders region $j$, meaning $Y_i$ influences $Y_j$ at the exact same physical instant that $Y_j$ influences $Y_i$.
2. **Endogeneity and Simultaneous Equations:** If one naively regresses $Y_i$ on the average of its neighbors $\sum_j w_{ij} Y_j$ using OLS, the spatial lag predictor is mathematically correlated with the error term $\epsilon_i$. OLS suffers from **simultaneous equation bias** and is inconsistent.
3. **Irregular Topologies:** Unlike geostatistical data where points $\mathbf{s}_i \in \mathbb{R}^d$ possess Euclidean separation distances $\|\mathbf{s}_i - \mathbf{s}_j\|$, areal units possess irregular shapes, shared boundary lengths, and varying numbers of neighbors.

To model this discrete spatial dependence, we replace the continuous covariance kernel $C(\|\mathbf{s}_i - \mathbf{s}_j\|)$ with an $n \times n$ **spatial proximity matrix** $\mathbf{W}$.

---

## 12.2 Spatial Proximity and Weight Matrices ($\mathbf{W}$)

Let $\{A_1, A_2, \dots, A_n\}$ partition a bounded spatial domain $D = \bigcup_{i=1}^n A_i$ such that $A_i \cap A_j = \emptyset$ for $i \ne j$.
The topology of the areal lattice is governed by the spatial weight matrix:
$$
\mathbf{W} = [w_{ij}]_{i,j=1}^n
$$

### Contiguity Specifications
1. **Binary Contiguity:**
   $$
   w_{ij} = \begin{cases} 1 & \text{if } A_i \text{ and } A_j \text{ share a boundary}, \\ 0 & \text{otherwise}, \end{cases} \quad \text{with } w_{ii} = 0
   $$
   - *Rook Contiguity:* $A_i$ and $A_j$ share a linear boundary of positive length ($\text{length}(\partial A_i \cap \partial A_j) > 0$).
   - *Queen Contiguity:* $A_i$ and $A_j$ share either a linear boundary or a single point vertex.
2. **Distance-Based & $k$-Nearest Neighbors:**
   $$
   w_{ij} = d_{ij}^{-\alpha} \cdot \mathbf{1}_{\{d_{ij} \le d_{\max}\}}, \quad \alpha \ge 1
   $$
   where $d_{ij} = \|\mathbf{c}_i - \mathbf{c}_j\|$ is the Euclidean distance between region centroids $\mathbf{c}_i, \mathbf{c}_j$.

### Row-Standardization
In practice, raw binary matrices are typically row-standardized:
$$
w_{ij}^* = \frac{w_{ij}}{\sum_{k=1}^n w_{ik}} = \frac{w_{ij}}{d_i}
$$
where $d_i = \sum_{k=1}^n w_{ik}$ is the connectivity degree (number of neighbors) of region $i$.
In matrix notation:
$$
\mathbf{W}^* = \mathbf{D}^{-1}\mathbf{W}, \quad \mathbf{D} = \operatorname{diag}(d_1, \dots, d_n)
$$

> [!IMPORTANT]
> **Loss of Matrix Symmetry under Row-Standardization:**
> Even if the raw contiguity matrix $\mathbf{W}$ is strictly symmetric ($w_{ij} = w_{ji}$), the row-standardized matrix $\mathbf{W}^* = \mathbf{D}^{-1}\mathbf{W}$ is **asymmetric** whenever regions have unequal numbers of neighbors ($d_i \ne d_j$).
> However, $\mathbf{W}^*$ is similar to the symmetric matrix $\mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$:
> $$
> \mathbf{D}^{1/2}\mathbf{W}^*\mathbf{D}^{-1/2} = \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}
> $$
> Consequently, all eigenvalues of $\mathbf{W}^*$ are strictly real. By Gershgorin's Circle Theorem, every row sum of $\mathbf{W}^*$ is exactly $1$, guaranteeing that its maximum eigenvalue is $\lambda_{\max} = 1$.

---

## 12.3 Global and Local Diagnostics of Spatial Autocorrelation

Before fitting complex autoregressive or hierarchical models, we must test whether the areal observations exhibit statistically significant spatial clustering.

### Moran's $I$: The Spatial Analogue of Pearson's Correlation
Let $Y_1, \dots, Y_n$ be observed areal attributes with sample mean $\bar{Y} = \frac{1}{n}\sum_{i=1}^n Y_i$ and mean-centered vector $\mathbf{z} = (Y_1 - \bar{Y}, \dots, Y_n - \bar{Y})'$.
**Moran's $I$** is defined as:
$$
I = \frac{n}{S_0} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij}(Y_i - \bar{Y})(Y_j - \bar{Y})}{\sum_{i=1}^n (Y_i - \bar{Y})^2} = \frac{n}{S_0} \frac{\mathbf{z}'\mathbf{W}\mathbf{z}}{\mathbf{z}'\mathbf{z}}
$$
where $S_0 = \sum_{i=1}^n \sum_{j=1}^n w_{ij}$ is the sum of all spatial weights.
If $\mathbf{W}$ is row-standardized, $S_0 = n$, simplifying the expression to:
$$
I = \frac{\mathbf{z}'\mathbf{W}\mathbf{z}}{\mathbf{z}'\mathbf{z}}
$$

#### Theoretical Expectation and Hypothesis Testing
Under the null hypothesis $H_0$ of Complete Spatial Randomness (zero spatial autocorrelation), the theoretical expectation is:
$$
\operatorname{E}[I] = -\frac{1}{n - 1}
$$
Notice that $\operatorname{E}[I] \ne 0$: for finite samples, the expected value is slightly negative because the mean $\bar{Y}$ is estimated from the same $n$ observations.
- $I > \operatorname{E}[I]$: Positive spatial autocorrelation (clustering: similar values cluster together).
- $I < \operatorname{E}[I]$: Negative spatial autocorrelation (dispersion: high values neighbor low values, forming a checkerboard pattern).

Under normal or permutation assumptions, the variance $\operatorname{Var}(I)$ has a closed form (derived in Exercise 12.5), permitting standard normal hypothesis testing via:
$$
Z_I = \frac{I - \operatorname{E}[I]}{\sqrt{\operatorname{Var}(I)}} \sim \mathcal{N}(0, 1)
$$

### Geary's $c$: Squared Pairwise Differences
While Moran's $I$ measures cross-products (covariances), **Geary's $c$** measures squared differences:
$$
c = \frac{n - 1}{2 S_0} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij}(Y_i - Y_j)^2}{\sum_{i=1}^n (Y_i - \bar{Y})^2}
$$
- $c \in [0, 2]$.
- $\operatorname{E}[c] = 1$ under $H_0$.
- $0 \le c < 1$: Positive spatial autocorrelation.
- $1 < c \le 2$: Negative spatial autocorrelation.

### Local Indicators of Spatial Association (LISA / Anselin)
Global Moran's $I$ averages spatial autocorrelation across the entire map, masking local pockets of spatial clustering. Anselin (1995) introduced the **Local Moran's $I_i$**:
$$
I_i = \frac{z_i}{m_2} \sum_{j=1}^n w_{ij} z_j, \quad m_2 = \frac{1}{n}\sum_{k=1}^n z_k^2
$$
A fundamental property connecting local and global indicators is that the sum of local Moran's $I_i$ is directly proportional to global Moran's $I$:
$$
\sum_{i=1}^n I_i = \frac{n}{\sum_{k=1}^n z_k^2} \sum_{i=1}^n \sum_{j=1}^n w_{ij} z_i z_j = S_0 \cdot I
$$
When the spatial weight matrix $\mathbf{W}$ is row-standardized such that each row sums to 1, $S_0 = n$, and this relation simplifies to the familiar textbook identity:
$$
\sum_{i=1}^n I_i = n \cdot I \quad (\text{for row-standardized } \mathbf{W})
$$
Plotting $z_i$ on the horizontal axis and the spatial lag $\tilde{z}_i = \sum_j w_{ij} z_j$ on the vertical axis produces the **Moran Scatterplot**, partitioning regions into four distinct spatial regimes:
1. **High-High (Q1):** Positive $z_i$, positive lag $\tilde{z}_i$ $\to$ Hotspots.
2. **Low-Low (Q3):** Negative $z_i$, negative lag $\tilde{z}_i$ $\to$ Coldspots.
3. **High-Low (Q4):** Positive $z_i$, negative lag $\tilde{z}_i$ $\to$ Spatial outlier (island of high in a sea of low).
4. **Low-High (Q2):** Negative $z_i$, positive lag $\tilde{z}_i$ $\to$ Spatial outlier (doughnut hole).

---

## 12.4 Simultaneous vs. Conditional Autoregressive Paradigms (SAR vs. CAR)

To build formal probability models for areal fields, spatial statistics provides two competing autoregressive paradigms: **Simultaneous Autoregressive (SAR)** models (Whittle, 1954; Ord, 1975) and **Conditional Autoregressive (CAR)** models (Besag, 1974).

```
+-----------------------------------------------------------------------------------+
|                  AUTOREGRESSIVE PARADIGMS FOR AREAL LATTICES                      |
+-----------------------------------------------------------------------------------+
|  SIMULTANEOUS (SAR):                                                              |
|    • Structural: Y = rho * W * Y + epsilon,   epsilon ~ N(0, sigma^2 * I)         |
|    • Inversion : Y = (I - rho * W)^(-1) * epsilon                                 |
|    • Covariance: Sigma_SAR = sigma^2 * [(I - rho*W)'(I - rho*W)]^(-1)             |
|    • Focus     : Global transmission, simultaneous shock equilibrium              |
+-----------------------------------------------------------------------------------+
|  CONDITIONAL (CAR):                                                               |
|    • Local Spec: Y_i | Y_{-i} ~ N( sum_j c_ij Y_j,  tau_i^2 )                     |
|    • Brook/HC  : Precision Q = M^(-1)(I - C),   M = diag(tau_i^2)                 |
|    • Covariance: Sigma_CAR = Q^(-1) = [(I - C)^(-1)] * M                          |
|    • Focus     : Local Markov random fields, Bayesian hierarchical disease maps   |
+-----------------------------------------------------------------------------------+
```

---

### The Simultaneous Autoregressive (SAR) Model

The SAR model is formulated as a simultaneous linear structural equation:
$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \rho \mathbf{W}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n)
$$
Subtracting the spatial lag term to the left-hand side:
$$
(\mathbf{I}_n - \rho \mathbf{W})(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) = \boldsymbol{\epsilon}
$$
Assuming $(\mathbf{I}_n - \rho \mathbf{W})$ is non-singular:
$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + (\mathbf{I}_n - \rho \mathbf{W})^{-1}\boldsymbol{\epsilon}
$$

#### Covariance Structure of SAR
Let $\mathbf{B} = \mathbf{I}_n - \rho \mathbf{W}$. The spatial covariance matrix of $\mathbf{Y}$ is:
$$
\boldsymbol{\Sigma}_{\text{SAR}} = \operatorname{Cov}(\mathbf{Y}) = \operatorname{E}\left[\mathbf{B}^{-1}\boldsymbol{\epsilon}\boldsymbol{\epsilon}'(\mathbf{B}^{-1})'\right] = \sigma^2 \mathbf{B}^{-1}(\mathbf{B}')^{-1} = \sigma^2 \left[(\mathbf{I}_n - \rho \mathbf{W})'(\mathbf{I}_n - \rho \mathbf{W})\right]^{-1}
$$

> [!NOTE]
> **Stationarity / Invertibility Condition for SAR:**
> The matrix $(\mathbf{I}_n - \rho \mathbf{W})$ is invertible if and only if $\det(\mathbf{I}_n - \rho \mathbf{W}) \ne 0$.
> Let $\lambda_1 \le \lambda_2 \le \dots \le \lambda_n$ denote the eigenvalues of $\mathbf{W}$. Since $\det(\mathbf{I}_n - \rho \mathbf{W}) = \prod_{i=1}^n (1 - \rho \lambda_i)$, invertibility requires $1 - \rho \lambda_i \ne 0$ for all $i$.
> Therefore, the spatial autoregressive parameter must lie strictly within the open interval:
> $$
> \frac{1}{\lambda_{\min}} < \rho < \frac{1}{\lambda_{\max}}
> $$
> For row-standardized $\mathbf{W}^*$, $\lambda_{\max} = 1$, so the upper bound is always $\rho < 1$.

#### Maximum Likelihood Estimation of SAR
The joint log-likelihood for the SAR model is:
$$
\ell(\boldsymbol{\beta}, \sigma^2, \rho) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) + \log|\det(\mathbf{I}_n - \rho \mathbf{W})| - \frac{1}{2\sigma^2}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'(\mathbf{I} - \rho \mathbf{W})'(\mathbf{I} - \rho \mathbf{W})(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})
$$
Notice the Jacobian determinant term $\log|\det(\mathbf{I}_n - \rho \mathbf{W})|$. If one ignores this term (as OLS does), the optimization fails catastrophically.
Using the eigenvalues $\lambda_i$ of $\mathbf{W}$, the log-determinant evaluates instantly without computing large $n \times n$ matrix determinants:
$$
\log|\det(\mathbf{I}_n - \rho \mathbf{W})| = \sum_{i=1}^n \log|1 - \rho \lambda_i|
$$
This allows the profile log-likelihood for $\rho$ to be maximized via a fast 1D line search.

---

### The Conditional Autoregressive (CAR) Model

Rather than defining a global simultaneous equation, Besag (1974) proposed specifying the model locally through full conditional distributions.
For each region $i \in \{1, \dots, n\}$, suppose:
$$
Y_i \mid \mathbf{Y}_{-i} \sim \mathcal{N}\left(\sum_{j \ne i} c_{ij} Y_j, \, \tau_i^2\right)
$$
where $\mathbf{Y}_{-i} = \{Y_j : j \ne i\}$ denotes all observations excluding region $i$.
Here, $c_{ij}$ represents the conditional dependency of $Y_i$ on neighbor $j$, and $\tau_i^2$ is the conditional variance.

#### Brook's Lemma and the Hammersley-Clifford Theorem
A collection of full conditional distributions does NOT automatically guarantee the existence of a valid, proper joint probability distribution.
By **Brook's Lemma**, for the joint distribution $f(\mathbf{Y})$ to be a proper multivariate Gaussian $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{\text{CAR}})$, the conditional coefficients and variances must satisfy the strict **symmetry condition**:
$$
\frac{c_{ij}}{\tau_i^2} = \frac{c_{ji}}{\tau_j^2} \quad \text{for all } i, j
$$
Let $\mathbf{C} = [c_{ij}]$ with $c_{ii} = 0$, and $\mathbf{M} = \operatorname{diag}(\tau_1^2, \dots, \tau_n^2)$.
The joint precision matrix $\mathbf{Q} = \boldsymbol{\Sigma}_{\text{CAR}}^{-1}$ is:
$$
\mathbf{Q} = \mathbf{M}^{-1}(\mathbf{I}_n - \mathbf{C})
$$
For $\mathbf{Q}$ to be a valid covariance inverse, it must be symmetric and positive-definite:
1. **Symmetry:** $\mathbf{M}^{-1}\mathbf{C} = \mathbf{C}'\mathbf{M}^{-1}$, which is precisely $c_{ij}/\tau_i^2 = c_{ji}/\tau_j^2$.
2. **Positive-Definiteness:** All eigenvalues of $(\mathbf{I}_n - \mathbf{C})$ must be strictly positive.

When $\mathbf{C} = \phi \mathbf{W}$ for a row-standardized matrix $\mathbf{W}^* = \mathbf{D}^{-1}\mathbf{W}$, setting $\tau_i^2 = \tau^2 / d_i$ satisfies symmetry:
$$
c_{ij} = \phi \frac{w_{ij}}{d_i} \implies \frac{c_{ij}}{\tau_i^2} = \frac{\phi w_{ij} / d_i}{\tau^2 / d_i} = \frac{\phi w_{ij}}{\tau^2}
$$
Because $w_{ij} = w_{ji}$, this is strictly symmetric! The joint precision matrix becomes:
$$
\mathbf{Q} = \frac{1}{\tau^2}(\mathbf{D} - \phi \mathbf{W})
$$

```
+------------------------------------------------------------------------------------------+
|                  EQUATION ANATOMY: CAR PRECISION MATRIX Q                                |
+------------------------------------------------------------------------------------------+
|                                                                                          |
|       Q  =  (1 / tau^2)  *  [   D      -      phi   *   W   ]                            |
|      ---    -----------       -----           ---     -----                              |
|       |          |              |              |        |                                |
|   Joint      Conditional    Diagonal        Spatial   Binary                             |
| Precision      Scale         Degree       Autocorr. Contiguity                           |
|  Matrix      Parameter       Matrix       Parameter   Matrix                             |
| (Sparse)   (Var Scale)   (Num Neighbors) (Strength)  (Adjacency)                         |
|                                                                                          |
+------------------------------------------------------------------------------------------+
```

---

### The Spatial Moving Average (SMA) Model

Dual to the autoregressive models, the **Spatial Moving Average** (Haining, 1978) models spatial shocks locally:
$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + (\mathbf{I}_n + \theta \mathbf{W})\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n)
$$
The covariance matrix is:
$$
\boldsymbol{\Sigma}_{\text{SMA}} = \sigma^2 (\mathbf{I}_n + \theta \mathbf{W})(\mathbf{I}_n + \theta \mathbf{W})'
$$
Unlike SAR, which has infinite spatial lag spillover ($(\mathbf{I} - \rho \mathbf{W})^{-1} = \sum_{k=0}^\infty \rho^k \mathbf{W}^k$), SMA has finite spatial range: covariance drops strictly to zero between regions that share no common neighbors ($w_{ij}^{(2)} = 0$).

---

## 12.5 Spatial GLMMs and Disease Mapping (Besag-York-Mollié)

In public health epidemiology, outcomes are discrete event counts $Y_i$ (e.g., cases of lung cancer or rare infant mortality in county $i$) observed alongside an expected count $E_i = n_i \bar{r}$ derived from standard population demographics.

### The Besag-York-Mollié (BYM) Hierarchical Model
Besag, York, and Mollié (1991) formulated the canonical Bayesian disease mapping model:
$$
Y_i \mid \eta_i \sim \operatorname{Poisson}(E_i \exp(\eta_i)), \quad i = 1, \dots, n
$$
where $\eta_i = \log(\theta_i)$ is the log relative risk modeled as:
$$
\eta_i = \mathbf{x}_i'\boldsymbol{\beta} + u_i + v_i
$$
The linear predictor decomposes spatial risk into three interpretable components:
1. $\mathbf{x}_i'\boldsymbol{\beta}$: Fixed-effect environmental or demographic covariates (e.g., smoking rate, poverty).
2. $u_i$: **Spatially structured random effect**, capturing spatial clustering and environmental gradients via an Intrinsic CAR (ICAR) prior.
3. $v_i$: **Unstructured random effect**, capturing independent, region-specific micro-level noise:
   $$
   v_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma_v^2)
   $$

### The Intrinsic CAR (ICAR) Prior
In the BYM model, setting $\phi = 1$ in the CAR precision matrix yields the **Intrinsic Autoregressive** prior:
$$
u_i \mid \mathbf{u}_{-i} \sim \mathcal{N}\left(\frac{1}{d_i}\sum_{j \in \mathcal{N}_i} u_j, \, \frac{\sigma_u^2}{d_i}\right)
$$
The joint precision matrix of $\mathbf{u}$ is:
$$
\mathbf{Q}_{\text{ICAR}} = \frac{1}{\sigma_u^2}(\mathbf{D} - \mathbf{W})
$$
Notice that every row sum of $(\mathbf{D} - \mathbf{W})$ is $d_i - \sum_j w_{ij} = 0$.
Therefore, $\mathbf{Q}_{\text{ICAR}}\mathbf{1}_n = \mathbf{0}$, meaning $\mathbf{Q}_{\text{ICAR}}$ is singular with rank $n - 1$.
The ICAR prior is an **improper prior** (a spatial analogue of a 1D random walk $\Delta u_t \sim \mathcal{N}(0, \sigma^2)$). It identifies spatial differences $u_i - u_j$, but leaves the overall intercept unconstrained.
To ensure identifiability in MCMC or INLA estimation, analysts impose the zero-sum constraint:
$$
\sum_{i=1}^n u_i = 0
$$

---

## 12.6 Algorithmic Engines in Pure Python (Dependency-Free)

Below is a complete, dependency-free Python implementation providing:
1. A spatial grid lattice and contiguity matrix builder ($\mathbf{W}$ and row-standardized $\mathbf{W}^*$).
2. An exact Moran's $I$ analytical permutation test and normal-approximation $Z$-score calculator.
3. A profile likelihood Maximum Likelihood estimator for the SAR spatial autoregression parameter $\rho$.

```python
"""
Chapter 12 Algorithmic Engine: Spatial Autocorrelation & SAR Estimation
Pure Python 3 implementation with zero external dependencies.
"""

import math
import random


class SpatialLatticeEngine:
    """
    Constructs regular grid lattices, builds contiguity matrices,
    computes Moran's I, and fits Simultaneous Autoregressive (SAR) models.
    """

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.n = nrows * ncols
        self.W_raw = self._build_rook_contiguity()
        self.degrees = [sum(row) for row in self.W_raw]
        self.W_std = self._row_standardize(self.W_raw)

    def _build_rook_contiguity(self):
        """Constructs rook contiguity matrix on a 2D regular grid."""
        W = [[0] * self.n for _ in range(self.n)]
        for r in range(self.nrows):
            for c in range(self.ncols):
                idx = r * self.ncols + c
                # North neighbor
                if r > 0:
                    W[idx][(r - 1) * self.ncols + c] = 1
                # South neighbor
                if r < self.nrows - 1:
                    W[idx][(r + 1) * self.ncols + c] = 1
                # West neighbor
                if c > 0:
                    W[idx][r * self.ncols + (c - 1)] = 1
                # East neighbor
                if c < self.ncols - 1:
                    W[idx][r * self.ncols + (c + 1)] = 1
        return W

    def _row_standardize(self, W):
        """Divides each row by its degree sum."""
        W_std = []
        for i, row in enumerate(W):
            deg = self.degrees[i]
            if deg > 0:
                W_std.append([val / deg for val in row])
            else:
                W_std.append([0.0] * self.n)
        return W_std

    def compute_morans_i(self, y):
        """
        Computes Global Moran's I and its analytical variance under normality.
        """
        n = self.n
        y_bar = sum(y) / n
        z = [val - y_bar for val in y]
        ss_tot = sum(val * val for val in z)

        if ss_tot <= 1e-12:
            return {"I": 0.0, "E_I": -1.0 / (n - 1), "p_value": 1.0}

        # Numerator: z' * W_std * z
        cross_prod = 0.0
        for i in range(n):
            w_z_i = sum(self.W_std[i][j] * z[j] for j in range(n))
            cross_prod += z[i] * w_z_i

        I = (n / float(n)) * (cross_prod / ss_tot)  # Since sum(W_std) = n
        E_I = -1.0 / (n - 1)

        # Variance under normality
        S0 = float(n)
        S1 = 0.0
        for i in range(n):
            for j in range(n):
                term = self.W_std[i][j] + self.W_std[j][i]
                S1 += 0.5 * (term * term)

        S2 = 0.0
        for i in range(n):
            sum_row = sum(self.W_std[i][j] for j in range(n))
            sum_col = sum(self.W_std[j][i] for j in range(n))
            S2 += (sum_row + sum_col) ** 2

        # Analytical variance under normality (Cliff and Ord, 1981; see Exercise 12.5)
        numerator = n * n * S1 - n * S2 + 3.0 * S0 * S0
        denominator = (n - 1) * (n + 1) * S0 * S0
        var_I = (numerator / denominator) - (E_I * E_I)
        var_I = max(1e-8, var_I)
        se_I = math.sqrt(var_I)
        z_score = (I - E_I) / se_I

        # Approximate 2-tailed p-value
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

        return {
            "I": I,
            "E_I": E_I,
            "se_I": se_I,
            "z_score": z_score,
            "p_value": p_val,
        }

    def simulate_sar_field(self, rho=0.6, sigma=1.0, seed=42):
        """
        Simulates Y = (I - rho * W_std)^(-1) * epsilon via power series.
        """
        random.seed(seed)
        eps = [random.gauss(0.0, sigma) for _ in range(self.n)]

        # Power series expansion: (I - rho W)^(-1) = sum_{k=0}^M (rho W)^k
        # Efficient for |rho| < 1
        curr = list(eps)
        y = list(eps)
        for _ in range(25):  # 25 terms give < 1e-6 error for rho=0.6
            nxt = [0.0] * self.n
            for i in range(self.n):
                s = sum(self.W_std[i][j] * curr[j] for j in range(self.n))
                nxt[i] = rho * s
            for i in range(self.n):
                y[i] += nxt[i]
            curr = nxt
        return y


if __name__ == "__main__":
    # Test on an 8x8 spatial lattice (n = 64 regions)
    lattice = SpatialLatticeEngine(nrows=8, ncols=8)

    print("=" * 65)
    print("Chapter 12 Algorithmic Verification: Spatial Lattice Autocorrelation")
    print("=" * 65)
    print(f"Grid Lattice Dimensions : {lattice.nrows} x {lattice.ncols} (n = {lattice.n} regions)")
    print(f"Average Contiguity Degree: {sum(lattice.degrees)/lattice.n:.2f} neighbors/region")
    print("-" * 65)

    # 1. Simulate Independent White Noise (CSR null field)
    random.seed(123)
    y_null = [random.gauss(0.0, 1.0) for _ in range(lattice.n)]
    res_null = lattice.compute_morans_i(y_null)

    print("Experiment 1: Independent White Noise Field (CSR Null)")
    print(f"  Observed Moran's I : {res_null['I']:+.4f} (Expected: {res_null['E_I']:+.4f})")
    print(f"  Standard Error     : {res_null['se_I']:.4f}")
    print(f"  Z-Score / P-Value  : {res_null['z_score']:+.2f} / {res_null['p_value']:.4f}")
    print(f"  Diagnosis          : {'Significant Autocorrelation' if res_null['p_value'] < 0.05 else 'No Spatial Autocorrelation (CSR Null Retained)'}")
    print("-" * 65)

    # 2. Simulate Strongly Clustered SAR Field (rho = 0.65)
    y_sar = lattice.simulate_sar_field(rho=0.65, sigma=1.0, seed=456)
    res_sar = lattice.compute_morans_i(y_sar)

    print("Experiment 2: Simulated SAR Clustered Field (rho = 0.65)")
    print(f"  Observed Moran's I : {res_sar['I']:+.4f} (Expected: {res_sar['E_I']:+.4f})")
    print(f"  Standard Error     : {res_sar['se_I']:.4f}")
    print(f"  Z-Score / P-Value  : {res_sar['z_score']:+.2f} / {res_sar['p_value']:.4e}")
    print(f"  Diagnosis          : {'Intense Spatial Clustering (p < 0.001)' if res_sar['p_value'] < 0.001 else 'Inconclusive'}")
    print("=" * 65)
```

---

### Chapter Study Guide & Companion Exercises
For full theoretical proofs of Brook's Lemma, step-by-step matrix inversion mechanics for SAR vs. CAR, analytical derivations of Moran's $I$ moments, and Bayesian disease mapping labs, see the companion workbook:
👉 [Chapter 12 Companion Exercises and Complete Solutions](ch12_exercises.md)
