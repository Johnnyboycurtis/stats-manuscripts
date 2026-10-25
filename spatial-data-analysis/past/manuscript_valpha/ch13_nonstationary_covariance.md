# Chapter 13: Non-Stationary Covariance Architectures & Scalable Low-Rank Kriging

---

### Navigation & Learning Roadmap
- **Prerequisites:** [Chapter 2: Covariance Functions & Semivariograms](ch02_covariance_variogram.md) (positive-definiteness, Bochner's theorem), [Chapter 5: Best Linear Unbiased Prediction (BLUP) & Kriging](ch05_blup_kriging.md) (kriging system, $\mathcal{O}(n^3)$ Cholesky inversion).
- **Core Focus:** Overcoming the stationarity straightjacket; process and kernel convolutions (Higdon); the Paciorek-Schervish non-stationary Matérn theorem; the Sampson-Guttorp space deformation method; and scalable Fixed Rank Kriging (FRK) with the Sherman-Morrison-Woodbury inversion identity.
- **Companion Study Guide:** [Chapter 13 Companion Exercises and Complete Solutions](ch13_exercises.md).

---

## 13.1 The Classical-to-Spatial Bridge: From Homoskedastic Stationarity to Spatially Varying Geometries

In classical statistics, the Gauss-Markov theorem assumes homoskedastic errors: $\operatorname{Var}(\epsilon_i) = \sigma^2$ for all observations $i$. In classical geostatistics (Chapters 2–5), this assumption was extended to spatial stationarity:
1. **Variance Stationarity:** $\operatorname{Var}(Z(\mathbf{s})) = \sigma^2$ is constant across the entire geographic domain $D \subset \mathbb{R}^d$.
2. **Correlation Stationarity & Isotropy:** The spatial covariance depends solely on the Euclidean separation vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$:
   $$
   \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = C(\|\mathbf{s}_i - \mathbf{s}_j\|) = \sigma^2 \rho(\|\mathbf{h}\|)
   $$

```
Classical Stationary Kriging Assumption:
Uniform correlation circles everywhere across the domain D
   ( O )      ( O )      ( O )      ( O )
   same r     same r     same r     same r

Real-World Non-Stationary Physical Geometry:
Mountains (tight, rotated) vs. Plains (broad, isotropic) vs. Coastal Shelf (elongated along shore)
   ( / )      (   O   )      (-----)      ( \ )
  rotated      broad,       elongated    sheared
  ellipse     circular       ellipse     ellipse
```

In environmental science, meteorology, geophysics, and oceanography, this stationarity assumption fails fundamentally:
- **Spatial Heteroskedasticity:** Natural processes are more volatile in certain subregions (e.g., wind speed variance is much higher on mountain ridges than in sheltered valleys).
- **Spatially Varying Correlation Scales:** In open plains, atmospheric moisture correlates over hundreds of kilometers; in complex mountainous terrain, correlation drops within ten kilometers.
- **Spatially Rotating Anisotropy:** Across a coastline, air quality and temperature correlate strongly parallel to the shoreline, but decay abruptly perpendicular to it. As the coastline curves, the local principal axis of anisotropy rotates continuously across space.

> [!CAUTION]
> **The Cost of Forcing a Stationary Variogram on Non-Stationary Data:**
> If an analyst fits a single global stationary semivariogram $\gamma(h)$ to a non-stationary field:
> 1. In regions with short physical correlation lengths, the global model over-predicts spatial continuity, producing severely over-smoothed kriging maps that miss local physical peaks and valleys.
> 2. In regions with long physical correlation lengths, the global model under-predicts continuity, wasting valuable neighbor information.
> 3. The kriging prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ falsely reflects only sample geometry, failing to capture true regional physical uncertainty.

To model real-world spatial systems, we must construct **valid, non-stationary covariance functions** $C(\mathbf{s}_i, \mathbf{s}_j)$ that vary smoothly across coordinates while remaining mathematically positive-definite.

---

## 13.2 Process Convolutions and Moving-Kernel Architectures

The central mathematical challenge of non-stationary geostatistics is that an arbitrary function of two coordinates $f(\mathbf{s}_i, \mathbf{s}_j)$ will almost never produce a positive-definite covariance matrix for an arbitrary configuration of $n$ points.

Higdon (1998, 2002), building on pioneering work by Barry and Ver Hoef (1996), solved this via **spatial process convolutions**.

### Constructive Process Convolution
Let $W(\mathbf{u})$ denote a continuous Gaussian white noise process on $\mathbb{R}^d$ satisfying:
$$
\operatorname{E}[W(\mathbf{u})] = 0, \quad \operatorname{Cov}(W(\mathbf{u}), W(\mathbf{v})) = \delta(\mathbf{u} - \mathbf{v})
$$
where $\delta(\cdot)$ is the Dirac delta function.
Define the spatial random field $Z(\mathbf{s})$ as the moving-average kernel convolution of this white noise process:
$$
Z(\mathbf{s}) = \int_{\mathbb{R}^d} k_{\mathbf{s}}(\mathbf{u}) W(\mathrm{d}\mathbf{u})
$$
where $k_{\mathbf{s}}(\mathbf{u})$ is a spatially localized, square-integrable smoothing kernel centered at location $\mathbf{s}$.

### Guaranteed Positive-Definiteness
Using the covariance properties of Gaussian white noise integrals, the covariance between field values at any two locations $\mathbf{s}_i$ and $\mathbf{s}_j$ is:
$$
\begin{aligned}
C(\mathbf{s}_i, \mathbf{s}_j) &= \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = \operatorname{E}\left[\left(\int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) W(\mathrm{d}\mathbf{u})\right) \left(\int_{\mathbb{R}^d} k_{\mathbf{s}_j}(\mathbf{v}) W(\mathrm{d}\mathbf{v})\right)\right] \\
&= \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{v}) \operatorname{E}[W(\mathrm{d}\mathbf{u})W(\mathrm{d}\mathbf{v})] \\
&= \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, \mathrm{d}\mathbf{u}
\end{aligned}
$$

> [!TIP]
> **Why Kernel Convolutions Never Fail Positive-Definiteness:**
> For any finite collection of $n$ locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ and any arbitrary coefficient vector $\mathbf{a} \in \mathbb{R}^n$:
> $$
> \mathbf{a}'\mathbf{V}\mathbf{a} = \sum_{i=1}^n \sum_{j=1}^n a_i a_j \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, \mathrm{d}\mathbf{u} = \int_{\mathbb{R}^d} \left(\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u})\right)^2 \mathrm{d}\mathbf{u} \ge 0
> $$
> Because the integrand is an exact square, the resulting covariance matrix $\mathbf{V}$ is **guaranteed to be positive semi-definite**, regardless of how radically the kernel shape $k_{\mathbf{s}}(\cdot)$ changes across space!

---

## 13.3 The Paciorek-Schervish Non-Stationary Matérn Theorem

While Higdon's convolution integral is mathematically robust, evaluating the integral $\int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, \mathrm{d}\mathbf{u}$ numerically for millions of point pairs is computationally prohibitive.

Paciorek and Schervish (2006) derived a breakthrough closed-form analytical theorem by parameterizing the local kernel $k_{\mathbf{s}}(\mathbf{u})$ as a spatially varying multivariate Gaussian density:
$$
k_{\mathbf{s}}(\mathbf{u}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}(\mathbf{s})|^{1/4}} \exp\left(-\frac{1}{2}(\mathbf{u} - \mathbf{s})' \boldsymbol{\Sigma}(\mathbf{s})^{-1} (\mathbf{u} - \mathbf{s})\right)
$$
where $\boldsymbol{\Sigma}(\mathbf{s})$ is a $d \times d$ spatially varying, symmetric positive-definite **local metric tensor** that governs the local variance, correlation range, and anisotropy orientation at coordinate $\mathbf{s}$.

### The Closed-Form Non-Stationary Covariance Theorem
Let $\boldsymbol{\Sigma}(\mathbf{s})$ be a continuous field of local covariance matrices on $\mathbb{R}^d$, and let $\sigma^2(\mathbf{s})$ be a continuous field of local variances.
The general non-stationary Matérn covariance function between any two coordinates $\mathbf{s}_i, \mathbf{s}_j \in \mathbb{R}^d$ is:
$$
C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j) \left|\boldsymbol{\Sigma}(\mathbf{s}_i)\right|^{1/4} \left|\boldsymbol{\Sigma}(\mathbf{s}_j)\right|^{1/4} \left|\frac{\boldsymbol{\Sigma}(\mathbf{s}_i) + \boldsymbol{\Sigma}(\mathbf{s}_j)}{2}\right|^{-1/2} \mathcal{M}_\nu\left(\sqrt{Q_{ij}}\right)
$$
where:
1. $Q_{ij}$ is the squared **averaged Mahalanobis distance**:
   $$
   Q_{ij} = (\mathbf{s}_i - \mathbf{s}_j)' \left(\frac{\boldsymbol{\Sigma}(\mathbf{s}_i) + \boldsymbol{\Sigma}(\mathbf{s}_j)}{2}\right)^{-1} (\mathbf{s}_i - \mathbf{s}_j)
   $$
2. $\mathcal{M}_\nu(d)$ is the standard isotropic Matérn correlation function with smoothness parameter $\nu > 0$:
   $$
   \mathcal{M}_\nu(d) = \frac{2^{1 - \nu}}{\Gamma(\nu)} (\sqrt{2\nu} d)^\nu K_\nu(\sqrt{2\nu} d)
   $$

```
+-------------------------------------------------------------------------------------------------------------------+
|                     EQUATION ANATOMY: PACIOREK-SCHERVISH NON-STATIONARY COVARIANCE                        |
+-------------------------------------------------------------------------------------------------------------------+
|                                                                                                                   |
|  C(s_i, s_j)  =  [ sigma(s_i) sigma(s_j) ] * [ |Sigma_i|^(1/4) |Sigma_j|^(1/4) |(Sigma_i+Sigma_j)/2|^(-1/2) ]    |
|                  -------------------------   ---------------------------------------------------------------    |
|                             |                                               |                                     |
|                      Heteroskedastic                                 Geometric Mean Ratio                         |
|                       Scale Product                                 (Determinant Prefactor)                       |
|                                                                                                                   |
|                                              *  M_nu( sqrt{ (s_i - s_j)' [ (Sigma_i+Sigma_j)/2 ]^(-1) (s_i - s_j) } )
|                                                 -------------------------------------------------------------------
|                                                                                 |                                 |
|                                                                     Isotropic Matérn Kernel on                    |
|                                                                    Average Metric Mahalanobis Space               |
|                                                                                                                   |
+-------------------------------------------------------------------------------------------------------------------+
```

### Deep Geometric Intuition
Notice the four moving parts of this equation:
1. **Variance Heteroskedasticity:** $\sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j)$ allows the marginal variance to fluctuate smoothly across the map.
2. **Harmonizing Metric Tensors:** When comparing location $\mathbf{s}_i$ (with local ellipse $\boldsymbol{\Sigma}_i$) and location $\mathbf{s}_j$ (with local ellipse $\boldsymbol{\Sigma}_j$), the separation vector $\mathbf{s}_i - \mathbf{s}_j$ is evaluated against the arithmetic average metric tensor $\frac{1}{2}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j)$.
3. **The Determinant Prefactor:** The term $\frac{|\boldsymbol{\Sigma}_i|^{1/4} |\boldsymbol{\Sigma}_j|^{1/4}}{|(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j)/2|^{1/2}} \le 1$ represents the ratio of the geometric mean volume to the arithmetic mean volume of the two local ellipses. By the arithmetic-geometric mean inequality for matrices, this prefactor is strictly $\le 1$, reaching exactly $1$ when $\boldsymbol{\Sigma}_i = \boldsymbol{\Sigma}_j$. When two neighboring regions have radically different correlation orientations or ranges, this prefactor smoothly damps the correlation between them, mathematically preventing correlation distortion.
4. **Reduction to Stationary Matérn:** If $\boldsymbol{\Sigma}(\mathbf{s}) \equiv \phi^2 \mathbf{I}_d$ everywhere, the determinant prefactor becomes $\frac{(\phi^2)^{1/4}(\phi^2)^{1/4}}{\phi^2} = 1$, and $Q_{ij} = \frac{\|\mathbf{s}_i - \mathbf{s}_j\|^2}{\phi^2}$, recovering the classical stationary Matérn covariance.

---

## 13.4 Space Deformation: The Sampson-Guttorp Method

A conceptually distinct approach to non-stationarity was pioneered by Sampson and Guttorp (1992).

Rather than modifying the covariance equation on the geographic coordinates, they proposed **deforming the spatial coordinate system**:
$$
\mathbf{s} \in \mathcal{G} \subset \mathbb{R}^2 \xrightarrow{\quad f(\mathbf{s}) \quad} \mathbf{d} \in \mathcal{D} \subset \mathbb{R}^k, \quad k \ge 2
$$
1. **Geographic Space ($\mathcal{G}$-space):** The true physical map where coordinates $\mathbf{s} = (x, y)'$ reside, and where the physical field exhibits complex, non-stationary, non-isotropic variation.
2. **Deformed Latent Space ($\mathcal{D}$-space):** An unobserved, warped mathematical space where the process becomes **strictly stationary and isotropic**.

```
Geographic Space (G-Space)                    Deformed Space (D-Space)
Complex, warped non-stationarity              Homogeneous, isotropic Euclidean plane
        [Mountain]                                     d_i           d_j
        s_i     s_j                                     *             *
         *       *        --- Warping f(s) --->          \           /
          \     /                                         \         /
             *                                             *-------*
            s_k                                           d_k     d_m
         [Valley]
```

### The Deformation Covariance Model
The covariance between two points $\mathbf{s}_i, \mathbf{s}_j \in \mathcal{G}$ is simply:
$$
C(\mathbf{s}_i, \mathbf{s}_j) = C_0(\|f(\mathbf{s}_i) - f(\mathbf{s}_j)\|)
$$
where $C_0(h)$ is a valid stationary, isotropic 1D covariance kernel (e.g., exponential or spherical), and $f: \mathbb{R}^2 \to \mathbb{R}^k$ is a smooth, bijective coordinate transformation.

### Algorithmic Implementation via Splines and MDS
1. **Dispersion Estimation:** When replicate time-series observations are available at $n$ fixed monitoring stations (e.g., daily ozone across 10 years), compute empirical pairwise covariances $\widehat{C}_{ij}$ and semivariances $\widehat{\gamma}_{ij}$.
2. **Metric Multidimensional Scaling (MDS):** Invert the stationary variogram $\gamma_0(d)$ to find target latent distances:
   $$
   d_{ij}^* = \gamma_0^{-1}(\widehat{\gamma}_{ij})
   $$
   Use non-metric or metric MDS to find point coordinates $\{\mathbf{d}_1, \dots, \mathbf{d}_n\} \subset \mathbb{R}^k$ whose pairwise Euclidean distances best match $d_{ij}^*$.
3. **Thin-Plate Splines (TPS):** Fit a pair of smooth 2D Thin-Plate Splines mapping $\mathbf{s}_i \mapsto \mathbf{d}_i$:
   $$
   f(x, y) = \mathbf{A}\begin{pmatrix} 1 \\ x \\ y \end{pmatrix} + \sum_{m=1}^n \mathbf{w}_m \phi(\|\mathbf{s} - \mathbf{s}_m\|)
   $$
   where $\phi(r) = r^2 \log(r)$ is the biharmonic spline radial basis function.
4. **Prediction in $\mathcal{D}$-space:** To krige at an unsampled location $\mathbf{s}_0$, evaluate its warped coordinate $\mathbf{d}_0 = f(\mathbf{s}_0)$, and compute standard Ordinary Kriging in $\mathcal{D}$-space using Euclidean distances $\|\mathbf{d}_0 - \mathbf{d}_i\|$.

---

## 13.5 Scalable Low-Rank Kriging & Fixed Rank Kriging (FRK)

Whether stationary or non-stationary, classical geostatistical kriging requires solving the $n \times n$ linear system:
$$
\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}(\mathbf{s}_0)
$$
As detailed in Chapter 5, the Cholesky factorization of $\mathbf{V}$ requires $\mathcal{O}(n^3)$ floating-point operations and $\mathcal{O}(n^2)$ memory.
When $n = 100,000$ (e.g., satellite remote sensing retrievals or LiDAR canopy points):
$$
n^3 = (10^5)^3 = 10^{15}\text{ operations}, \quad n^2 \times 8\text{ bytes} \approx 80\text{ Gigabytes}
$$
Exact kriging becomes computationally impossible on standard hardware.

To solve this **Big $n$ problem**, Cressie and Johannesson (2008) developed **Fixed Rank Kriging (FRK)**.

### The Low-Rank Spatial Process Architecture
Decompose the spatial random field $Z(\mathbf{s})$ into a fixed-rank basis expansion plus independent measurement noise:
$$
Z(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \mathbf{S}(\mathbf{s})'\boldsymbol{\eta} + \epsilon(\mathbf{s})
$$
where:
1. $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ is the large-scale deterministic drift ($p$ covariates).
2. $\mathbf{S}(\mathbf{s}) = (S_1(\mathbf{s}), S_2(\mathbf{s}), \dots, S_r(\mathbf{s}))'$ is an $r$-dimensional vector of pre-specified, multi-scale spatial **basis functions** (e.g., local bisquare wavelets or radial basis functions), with $r \ll n$ (typically $r \approx 50$ to $500$).
3. $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$ is an $r \times 1$ latent random coefficient vector with $r \times r$ covariance matrix $\mathbf{K}$.
4. $\epsilon(\mathbf{s}) \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)$ is fine-scale white noise / measurement error (the nugget effect).

### The Fixed Rank Covariance Structure
For any two coordinates $\mathbf{s}_i, \mathbf{s}_j$, the model covariance is:
$$
C(\mathbf{s}_i, \mathbf{s}_j) = \mathbf{S}(\mathbf{s}_i)' \mathbf{K} \mathbf{S}(\mathbf{s}_j) + \sigma_\epsilon^2 \mathbf{1}_{\{\mathbf{s}_i = \mathbf{s}_j\}}
$$
Notice that $C(\mathbf{s}_i, \mathbf{s}_j)$ is **automatically non-stationary**: the basis functions $\mathbf{S}(\mathbf{s})$ capture complex local spatial geometries without requiring translation invariance!

In matrix form for $n$ observations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$:
$$
\mathbf{Z} = \mathbf{X}\boldsymbol{\beta} + \mathbf{S}\boldsymbol{\eta} + \boldsymbol{\epsilon}, \quad \mathbf{S} \in \mathbb{R}^{n \times r}, \quad \boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_\epsilon^2 \mathbf{I}_n)
$$
The $n \times n$ covariance matrix of the data is:
$$
\boldsymbol{\Sigma} = \operatorname{Cov}(\mathbf{Z}) = \mathbf{S}\mathbf{K}\mathbf{S}' + \sigma_\epsilon^2 \mathbf{I}_n
$$

### Fast $\mathcal{O}(n r^2 + r^3)$ Inversion via Sherman-Morrison-Woodbury
By the **Woodbury Matrix Identity**:
$$
(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}
$$
Setting $\mathbf{A} = \sigma_\epsilon^2 \mathbf{I}_n$, $\mathbf{U} = \mathbf{S}$, $\mathbf{C} = \mathbf{K}$, and $\mathbf{V} = \mathbf{S}'$:
$$
\boldsymbol{\Sigma}^{-1} = \left(\sigma_\epsilon^2 \mathbf{I}_n + \mathbf{S}\mathbf{K}\mathbf{S}'\right)^{-1} = \frac{1}{\sigma_\epsilon^2}\mathbf{I}_n - \frac{1}{\sigma_\epsilon^4}\mathbf{S}\left(\mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}\right)^{-1}\mathbf{S}'
$$

```
+-------------------------------------------------------------------------------------------------------------------+
|                        COMPUTATIONAL COMPLEXITY: EXACT KRIGING VS. FIXED RANK KRIGING                             |
+-------------------------------------------------------------------------------------------------------------------+
|                                                                                                                   |
|  EXACT KRIGING:                                                                                                   |
|    • Inversion: V^(-1) via Cholesky decomposition of (n x n) matrix                                               |
|    • Flops    : (1/3) * n^3                                                                                       |
|    • At n=100,000: ~ 3.3 x 10^14 FLOPs  (Several weeks of CPU time, 80 GB RAM)                                   |
|                                                                                                                   |
|  FIXED RANK KRIGING (FRK):                                                                                        |
|    • Gram Matrix: S' * S               -->  O(n * r^2)                                                            |
|    • Small Inverse: [ K^(-1) + (1/sigma^2)*S'S ]^(-1)  (r x r matrix) -->  O(r^3)                                 |
|    • Prediction Vector: S * [ ... ]    -->  O(n * r)                                                              |
|    • Total Flops: O(n * r^2 + r^3)                                                                                |
|    • At n=100,000, r=100: ~ 10^9 FLOPs  (< 0.5 seconds of CPU time, < 100 MB RAM)                                |
|                                                                                                                   |
+-------------------------------------------------------------------------------------------------------------------+
```

### Multiresolution Bisquare Basis Functions
A canonical authorized basis function is the **compactly supported bisquare function**:
$$
S_l(\mathbf{s}) = \begin{cases}
\left(1 - \left(\frac{\|\mathbf{s} - \mathbf{c}_l\|}{r_l}\right)^2\right)^2 & \text{if } \|\mathbf{s} - \mathbf{c}_l\| \le r_l, \\
0 & \text{otherwise},
\end{cases}
$$
where $\mathbf{c}_l$ is the center of the $l$-th basis function and $r_l$ is its aperture radius.
By placing basis functions across multiple spatial resolutions (e.g., Level 1: coarse regional coverage; Level 2: intermediate; Level 3: fine local coverage), FRK seamlessly captures both broad regional trends and sharp local spatial features while preserving lightning-fast $\mathcal{O}(n r^2)$ computational efficiency.

---

## 13.6 Algorithmic Engines in Pure Python (Dependency-Free)

Below is a complete, dependency-free Python implementation providing:
1. The **Paciorek-Schervish Non-Stationary Covariance Evaluator** supporting coordinate-dependent scale parameters and anisotropic rotations.
2. The **Fixed Rank Kriging (FRK) Solver** with local bisquare basis functions and fast Sherman-Morrison-Woodbury inversion.

```python
"""
Chapter 13 Algorithmic Engine: Non-Stationary Geostatistics & Fixed Rank Kriging
Pure Python 3 implementation with zero external dependencies.
"""

import math
import random


class PaciorekSchervishKernel:
    """
    Evaluates closed-form non-stationary Matérn / Gaussian covariance kernels
    with spatially varying metric tensors Sigma(s) and local variances sigma^2(s).
    """

    def __init__(self, metric_tensor_fn, variance_fn=None):
        """
        Parameters
        ----------
        metric_tensor_fn : callable
            Function taking (x, y) and returning 2x2 metric tensor [[a, b], [b, c]].
        variance_fn : callable or None
            Function taking (x, y) and returning marginal variance sigma^2(s).
        """
        self.metric_tensor_fn = metric_tensor_fn
        self.variance_fn = variance_fn if variance_fn is not None else (lambda x, y: 1.0)

    def _det_2x2(self, M):
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    def _inv_2x2(self, M):
        d = self._det_2x2(M)
        if abs(d) < 1e-12:
            return [[1.0, 0.0], [0.0, 1.0]]
        return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]

    def evaluate_covariance(self, s1, s2):
        """
        Computes non-stationary covariance between s1=(x1, y1) and s2=(x2, y2).
        Uses Gaussian smoothness limit (nu -> infinity).
        """
        x1, y1 = s1
        x2, y2 = s2

        sig1 = math.sqrt(self.variance_fn(x1, y1))
        sig2 = math.sqrt(self.variance_fn(x2, y2))

        M1 = self.metric_tensor_fn(x1, y1)
        M2 = self.metric_tensor_fn(x2, y2)

        det1 = self._det_2x2(M1)
        det2 = self._det_2x2(M2)

        # Average metric tensor M_avg = (M1 + M2) / 2
        M_avg = [
            [(M1[0][0] + M2[0][0]) * 0.5, (M1[0][1] + M2[0][1]) * 0.5],
            [(M1[1][0] + M2[1][0]) * 0.5, (M1[1][1] + M2[1][1]) * 0.5],
        ]
        det_avg = self._det_2x2(M_avg)

        # Determinant prefactor
        prefactor = (det1**0.25 * det2**0.25) / math.sqrt(det_avg)

        # Squared Mahalanobis distance Q = (s1 - s2)' * M_avg^(-1) * (s1 - s2)
        inv_avg = self._inv_2x2(M_avg)
        dx = x1 - x2
        dy = y1 - y2
        qx = inv_avg[0][0] * dx + inv_avg[0][1] * dy
        qy = inv_avg[1][0] * dx + inv_avg[1][1] * dy
        Q = dx * qx + dy * qy

        # Gaussian kernel correlation = exp(-0.5 * Q)
        corr = math.exp(-0.5 * Q)

        return sig1 * sig2 * prefactor * corr


class FixedRankKrigingEngine:
    """
    Implements scalable Fixed Rank Kriging (FRK) using multi-center bisquare
    basis functions and Sherman-Morrison-Woodbury inversion.
    """

    def __init__(self, basis_centers, basis_radii, nugget_variance=0.05):
        self.centers = basis_centers
        self.radii = basis_radii
        self.r = len(basis_centers)
        self.sigma2_eps = nugget_variance

    def _bisquare_basis(self, x, y):
        """Evaluates r bisquare basis functions at location (x, y)."""
        vec = []
        for (cx, cy), rad in zip(self.centers, self.radii):
            d = math.hypot(x - cx, y - cy)
            if d < rad:
                val = (1.0 - (d / rad) ** 2) ** 2
            else:
                val = 0.0
            vec.append(val)
        return vec

    def fit_and_predict(self, sample_coords, sample_values, pred_coords):
        """
        Fits FRK model and predicts at unsampled target coordinates.
        Complexity: O(n * r^2 + r^3).
        """
        n = len(sample_coords)
        r = self.r
        sig2 = self.sigma2_eps

        # 1. Construct Basis Matrix S (n x r)
        S = []
        for x, y in sample_coords:
            S.append(self._bisquare_basis(x, y))

        # 2. Compute Gram Matrix S' * S (r x r) and S' * Z (r x 1)
        StS = [[0.0] * r for _ in range(r)]
        StZ = [0.0] * r
        for i in range(n):
            zi = sample_values[i]
            row_i = S[i]
            for j in range(r):
                StZ[j] += row_i[j] * zi
                for k in range(r):
                    StS[j][k] += row_i[j] * row_i[k]

        # Prior covariance K on basis coefficients (scaled identity for simplicity)
        # In full FRK, K is estimated via Method of Moments or EM
        K_diag = 2.0  # prior variance of basis weights

        # Small r x r matrix to invert: A = (K^(-1) + (1/sig2) * S'S)
        A = [[0.0] * r for _ in range(r)]
        for j in range(r):
            for k in range(r):
                A[j][k] = (1.0 / sig2) * StS[j][k]
            A[j][j] += 1.0 / K_diag

        # Invert A (r x r) via Gauss-Jordan elimination
        invA = self._invert_matrix_gauss_jordan(A)

        # Compute basis weights eta_hat = invA * ((1/sig2) * StZ)
        rhs = [(1.0 / sig2) * StZ[j] for j in range(r)]
        eta_hat = [sum(invA[j][k] * rhs[k] for k in range(r)) for j in range(r)]

        # 3. Predict at new target coordinates: Z_hat(s0) = S(s0)' * eta_hat
        predictions = []
        variances = []
        for x0, y0 in pred_coords:
            s0_vec = self._bisquare_basis(x0, y0)
            pred_val = sum(s0_vec[j] * eta_hat[j] for j in range(r))

            # Kriging variance = S0' * invA * S0 + nugget
            s0_invA = [sum(s0_vec[k] * invA[k][j] for k in range(r)) for j in range(r)]
            pred_var = sum(s0_invA[j] * s0_vec[j] for j in range(r)) + sig2

            predictions.append(pred_val)
            variances.append(pred_var)

        return predictions, variances

    def _invert_matrix_gauss_jordan(self, mat):
        """Pure Python matrix inversion for r x r matrices."""
        dim = len(mat)
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(dim)] for i, row in enumerate(mat)]
        for i in range(dim):
            # Pivot
            pivot = aug[i][i]
            if abs(pivot) < 1e-12:
                for k in range(i + 1, dim):
                    if abs(aug[k][i]) > abs(pivot):
                        aug[i], aug[k] = aug[k], aug[i]
                        pivot = aug[i][i]
                        break
            inv_p = 1.0 / pivot
            for j in range(2 * dim):
                aug[i][j] *= inv_p
            for k in range(dim):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * dim):
                        aug[k][j] -= factor * aug[i][j]
        return [row[dim:] for row in aug]


if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 13 Verification: Non-Stationary Covariance & FRK Engine")
    print("=" * 70)

    # 1. Test Paciorek-Schervish Non-Stationary Kernel
    # Metric tensor rotates with x: Sigma(x, y) has range increasing along x
    def sample_metric_tensor(x, y):
        scale_x = 0.2 + 0.8 * x  # range grows from 0.2 (west) to 1.0 (east)
        scale_y = 0.2
        return [[scale_x**2, 0.0], [0.0, scale_y**2]]

    ps_kernel = PaciorekSchervishKernel(sample_metric_tensor)
    cov_west = ps_kernel.evaluate_covariance((0.1, 0.5), (0.2, 0.5))
    cov_east = ps_kernel.evaluate_covariance((0.8, 0.5), (0.9, 0.5))

    print("Experiment 1: Paciorek-Schervish Spatially Varying Metric Tensor")
    print(f"  West (Short Range) Covariance [dist=0.10]: {cov_west:.4f}")
    print(f"  East (Long Range)  Covariance [dist=0.10]: {cov_east:.4f}")
    print(f"  Diagnosis: Covariance is {cov_east/cov_west:.2f}x stronger in East for identical lag!")
    print("-" * 70)

    # 2. Test Scalable Fixed Rank Kriging Engine on n = 1,000 synthetic points
    print("Experiment 2: Fixed Rank Kriging (FRK) on n = 1,000 points")
    random.seed(42)
    n_sample = 1000
    samples = [(random.random(), random.random()) for _ in range(n_sample)]
    # True underlying spatial trend: non-linear hill
    vals = [math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y) + random.gauss(0, 0.2) for x, y in samples]

    # Grid of r = 16 basis centers
    centers = []
    radii = []
    for gx in [0.2, 0.4, 0.6, 0.8]:
        for gy in [0.2, 0.4, 0.6, 0.8]:
            centers.append((gx, gy))
            radii.append(0.35)

    frk = FixedRankKrigingEngine(centers, radii, nugget_variance=0.04)

    test_targets = [(0.25, 0.25), (0.50, 0.50), (0.75, 0.75)]
    preds, vars_est = frk.fit_and_predict(samples, vals, test_targets)

    print(f"  Basis Rank r                 : {frk.r} multi-scale bisquare functions")
    print(f"  Sample Size n                : {n_sample} observations")
    for pt, p_val, v_val in zip(test_targets, preds, vars_est):
        print(f"  Target {pt} -> Predicted: {p_val:+7.4f} (Kriging SE: {math.sqrt(v_val):.4f})")
    print("=" * 70)
```

---

### Chapter Study Guide & Companion Exercises
For full theoretical proofs of the Paciorek-Schervish theorem, step-by-step matrix inversion mechanics for Fixed Rank Kriging, multidimensional scaling derivations for Sampson-Guttorp deformation, and high-performance pure Python labs, see the companion workbook:
👉 [Chapter 13 Companion Exercises and Complete Solutions](ch13_exercises.md)
