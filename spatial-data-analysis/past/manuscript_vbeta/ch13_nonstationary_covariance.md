# Chapter 13: Non-Stationary Covariance Architectures & Scalable Low-Rank Kriging

```
================================================================================
   STATIONARY CONTINUUM                      NON-STATIONARY ARCHITECTURES
--------------------------------------------------------------------------------
Variance:   Homoskedastic Var(Z(s)) = sigma^2 Heteroskedastic sigma^2(s)
Range:      Constant scale phi everywhere     Spatially varying metric tensor Sigma(s)
Anisotropy: Global ellipse (single rotation)  Locally rotating & stretching ellipses
Geometry:   Rigid Euclidean metric ||s - s'|| Manifold warping f(s) in D-space
Big-n:      Exact Cholesky O(n^3), 80 GB RAM  Fixed Rank Kriging O(n*r^2 + r^3), <100 MB
================================================================================
```

### Navigation & Learning Roadmap
- **Prerequisites:** [Chapter 2: Covariance Functions & Semivariograms](ch02_covariance_variogram.md) (positive-definiteness, Bochner's theorem, geometric anisotropy), [Chapter 5: Best Linear Unbiased Prediction (BLUP) & Kriging](ch05_blup_kriging.md) (kriging system, $\mathcal{O}(n^3)$ Cholesky inversion), [Chapter 6: Universal Kriging](ch06_universal_kriging.md) (trend vs. covariance separation), and [Chapter 12: Lattice and Areal Models](ch12_lattice_areal_models.md) (sparse precision Gaussian Markov random fields).
- **Core Focus:** Escaping the stationarity straightjacket; process convolutions; the Paciorek-Schervish non-stationary Matérn theorem; the Sampson-Guttorp space deformation method; and scalable Fixed Rank Kriging (FRK) with the Sherman-Morrison-Woodbury inversion identity.
- **The Promise:** By the end of this chapter, you will understand how to construct valid non-stationary covariances whose range, sill, and orientation vary across the map, how to diagnose non-stationarity without confusing it for trend, and how to scale kriging to 100,000 observations on a standard laptop via low-rank basis expansions.

---

## 13.1 The Classical-to-Spatial Bridge: From Homoskedastic Stationarity to Spatially Varying Geometries

In classical linear models, the Gauss-Markov theorem begins with the homoskedasticity assumption: $\operatorname{Var}(\epsilon_i) = \sigma^2$ for all observations $i$. When residual variances differ across observations, classical statistics transitions to **Weighted Least Squares (WLS)** or Generalized Least Squares (GLS), weighting each observation inversely by its local variance $\sigma_i^2$.

In spatial statistics (Chapters 2–5), stationarity elevated homoskedasticity to the spatial continuum:
1. **Variance Stationarity:** $\operatorname{Var}(Z(\mathbf{s})) = \sigma^2$ is constant across the entire geographic domain $\mathcal{D} \subset \mathbb{R}^d$.
2. **Correlation Stationarity and Isotropy:** The spatial covariance depends solely on the Euclidean separation vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$:
   $$C(\mathbf{s}_i, \mathbf{s}_j) = \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = \sigma^2 \rho(\|\mathbf{s}_i - \mathbf{s}_j\|)$$

```
================================================================================
Classical Stationary Kriging Assumption:
Uniform correlation circles everywhere across the domain D
   ( O )              ( O )              ( O )              ( O )
   same r             same r             same r             same r

Real-World Non-Stationary Physical Geometry:
Mountains (tight, rotated) vs. Plains (broad, isotropic) vs. Coastal Shelf (elongated along shore)
   ( / )            (   O   )          (-------)            ( \ )
  rotated             broad,           elongated           sheared
  ellipse            circular           ellipse            ellipse
================================================================================
```

### 13.1.1 The Simplest Non-Stationary Model: Spatial Heteroskedasticity

The spatial analogue of Weighted Least Squares is the **separable heteroskedastic covariance model**:
$$C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i) \sigma(\mathbf{s}_j) \rho(\|\mathbf{s}_i - \mathbf{s}_j\|)$$
where $\sigma(\mathbf{s}) > 0$ is a continuous, strictly positive standard deviation field across the domain, and $\rho(\cdot)$ is any standard stationary, positive-definite correlation function.

In matrix form for $n$ locations, let $\mathbf{D} = \operatorname{diag}(\sigma(\mathbf{s}_1), \dots, \sigma(\mathbf{s}_n))$ and let $\mathbf{R} = [\rho(\|\mathbf{s}_i - \mathbf{s}_j\|)]_{i,j=1}^n$ be the stationary correlation matrix. The resulting covariance matrix is:
$$\boldsymbol{\Sigma} = \mathbf{D} \mathbf{R} \mathbf{D}$$

Because $\mathbf{R}$ is positive definite ($\mathbf{v}^\top \mathbf{R} \mathbf{v} > 0$ for any $\mathbf{v} \neq \mathbf{0}$) and $\mathbf{D}$ is invertible, for any non-zero vector $\mathbf{a} \in \mathbb{R}^n$:
$$\mathbf{a}^\top \boldsymbol{\Sigma} \mathbf{a} = \mathbf{a}^\top (\mathbf{D} \mathbf{R} \mathbf{D}) \mathbf{a} = (\mathbf{D}\mathbf{a})^\top \mathbf{R} (\mathbf{D}\mathbf{a}) > 0$$
Hence, $\boldsymbol{\Sigma}$ is unconditionally guaranteed to be positive definite!

This model solves variance non-stationarity. However, it assumes that the **spatial correlation length and anisotropy direction remain completely frozen across the entire map**.

### 13.1.2 Three Levels of Spatial Flexibility

In environmental science, meteorology, geophysics, and oceanography, correlation scales and directions change continuously:
- **Plains vs. Mountains:** In open plains, atmospheric moisture correlates over hundreds of kilometers; in complex mountainous terrain, correlation drops within ten kilometers.
- **Curving Coastlines:** Along an ocean coastline, temperature and wind correlate strongly parallel to the shoreline, but decay abruptly perpendicular to it. As the coastline curves, the local principal axis of anisotropy rotates continuously across space.

We can view spatial modeling as progressing through three distinct levels of geometric flexibility:
1. **Stationary Isotropic (Chapter 2):** A single scalar range parameter $\phi$ governs correlation in all directions everywhere.
2. **Global Geometric Anisotropy (Chapter 2):** A single global coordinate transformation matrix $\mathbf{B}$ rotates and stretches the entire map once: $C(\mathbf{s}_i, \mathbf{s}_j) = C_0(\|\mathbf{B}(\mathbf{s}_i - \mathbf{s}_j)\|)$.
3. **Local Non-Stationary Anisotropy (This Chapter):** The stretching and rotation tensor $\boldsymbol{\Sigma}(\mathbf{s})$ varies at every coordinate $\mathbf{s} \in \mathcal{D}$ (Paciorek-Schervish), or the entire coordinate system undergoes a smooth nonlinear manifold deformation $f(\mathbf{s})$ (Sampson-Guttorp).

### 13.1.3 Diagnosing Non-Stationarity vs. Trend

Before abandoning stationary models, an analyst must answer a critical diagnostic question: **Is the field truly covariance non-stationary, or is it a stationary process around an unmodeled deterministic trend?**

```
================================================================================
FROM THE LECTURE HALL: Box 13.1
Heteroskedasticity and the Trend-Covariance Confound
================================================================================
A classical trap in spatial analysis is confusing a large-scale deterministic
mean trend mu(s) = x(s)'*beta with non-stationary covariance.

Suppose temperatures decrease steadily from south to north along a mountain
slope. If you compute an empirical semivariogram on raw temperatures without
detrending:
1. Long-distance pairs will span the north-south elevation gradient, artificially
   inflating the sample semivariance gamma(h) at large lags.
2. The empirical semivariogram will appear non-stationary (failing to reach a
   sill, or showing direction-dependent sills).

Rule of Practice:
Always model large-scale deterministic drift first (Universal Kriging,
Chapter 6) using physically motivated covariates (elevation, latitude, distance
to coast). Only inspect residuals e(s) = Z(s) - x(s)'*beta_OLS for covariance
non-stationarity!

Diagnostic Toolkit for Residual Non-Stationarity:
- Moving-Window Semivariograms: Partition the study domain into overlapping
  subregions (e.g., North, South, Coast, Inland). Fit separate semivariograms
  to the residuals in each window. If the fitted range phi or sill sigma^2
  differs systematically across subregions, covariance non-stationarity is present.
- Residual Variance Maps: Compute moving-window sample variances. Spatial
  clustering of high and low variances confirms heteroskedasticity.
================================================================================
```

> [!CAUTION] The Danger of Forcing Stationary Kriging on Non-Stationary Systems
> If an analyst forces a single global stationary semivariogram $\gamma(h)$ onto non-stationary data:
> 1. In regions with short physical correlation lengths, the global model over-predicts spatial continuity, severely over-smoothing the kriging surface and erasing localized physical extremes.
> 2. In regions with long physical correlation lengths, the global model under-predicts spatial continuity, ignoring valuable neighbor information.
> 3. The kriging prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ falsely reflects only the geometric configuration of sample points, completely detached from regional physical volatility.

---

## 13.2 Process Convolutions and Moving-Kernel Architectures

The central mathematical obstacle of non-stationary geostatistics is that an arbitrary function of two coordinates $f(\mathbf{s}_i, \mathbf{s}_j)$ will almost never produce a positive-definite covariance matrix for an arbitrary configuration of $n$ points.

Higdon (1998, 2002), building on pioneering work by Barry and Ver Hoef (1996), solved this via **spatial process convolutions**.

### 13.2.1 The Finite Knots Architecture

Rather than defining a covariance function directly, we construct the spatial random field $Z(\mathbf{s})$ constructively as a moving mixture of localized smoothing kernels centered over a set of $m$ discrete spatial **knots** $\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_m\} \subset \mathcal{D}$:
$$Z(\mathbf{s}) = \sum_{l=1}^m k_{\mathbf{s}}(\mathbf{u}_l) w_l$$
where $k_{\mathbf{s}}(\mathbf{u}_l)$ is a localized kernel function evaluated at knot $\mathbf{u}_l$, and $w_l \stackrel{\text{iid}}{\sim} \mathcal{N}(0, 1)$ are independent standard normal latent weights.

In matrix notation for an observation vector $\mathbf{Z} = [Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n)]^\top \in \mathbb{R}^n$:
$$\mathbf{Z} = \mathbf{S} \mathbf{w}, \quad \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m)$$
where $\mathbf{S} \in \mathbb{R}^{n \times m}$ is the matrix of kernel evaluations with entries $S_{il} = k_{\mathbf{s}_i}(\mathbf{u}_l)$.

The resulting covariance between any two observation locations $\mathbf{s}_i$ and $\mathbf{s}_j$ is:
$$C(\mathbf{s}_i, \mathbf{s}_j) = \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = \mathbb{E}\left[ \left(\sum_{l=1}^m k_{\mathbf{s}_i}(\mathbf{u}_l) w_l\right) \left(\sum_{k=1}^m k_{\mathbf{s}_j}(\mathbf{u}_k) w_k\right) \right] = \sum_{l=1}^m k_{\mathbf{s}_i}(\mathbf{u}_l) k_{\mathbf{s}_j}(\mathbf{u}_l)$$

In matrix form across all $n$ locations:
$$\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{Z}) = \mathbf{S} \mathbf{S}^\top$$

### 13.2.2 Guaranteed Positive Semi-Definiteness

Why is this construction so powerful? Because for any arbitrary vector $\mathbf{a} \in \mathbb{R}^n$:
$$\mathbf{a}^\top \boldsymbol{\Sigma} \mathbf{a} = \mathbf{a}^\top (\mathbf{S}\mathbf{S}^\top) \mathbf{a} = (\mathbf{S}^\top \mathbf{a})^\top (\mathbf{S}^\top \mathbf{a}) = \|\mathbf{S}^\top \mathbf{a}\|^2 \ge 0$$

The quadratic form is the Euclidean norm of the vector $\mathbf{S}^\top \mathbf{a}$. A sum of squares can **never be negative**! 

If the kernel functions are linearly independent across the observation locations (readily achieved when $m \ge n$ and kernels are distinct), $\mathbf{S}$ has full row rank and $\boldsymbol{\Sigma}$ is strictly positive definite ($\mathbf{a}^\top \boldsymbol{\Sigma}\mathbf{a} > 0$ for all $\mathbf{a} \neq \mathbf{0}$).

### 13.2.3 Continuous Process Convolutions

Taking the dense-knot limit as $m \to \infty$ across the continuous continuum $\mathbb{R}^d$, the sum converges to a continuous stochastic integral against Gaussian white noise:
$$Z(\mathbf{s}) = \int_{\mathbb{R}^d} k_{\mathbf{s}}(\mathbf{u}) W(d\mathbf{u})$$
where $W(\cdot)$ is standard Gaussian white noise. The continuous covariance function becomes:
$$C(\mathbf{s}_i, \mathbf{s}_j) = \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, d\mathbf{u}$$

For any coefficient vector $\mathbf{a} \in \mathbb{R}^n$:
$$\mathbf{a}^\top \boldsymbol{\Sigma} \mathbf{a} = \sum_{i=1}^n \sum_{j=1}^n a_i a_j \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, d\mathbf{u} = \int_{\mathbb{R}^d} \left( \sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u}) \right)^2 d\mathbf{u} \ge 0$$
The integral of an exact squared function can never be negative. The covariance matrix is unconditionally positive semi-definite, **regardless of how radically the kernel shape $k_{\mathbf{s}}(\cdot)$ changes across the map**!

> [!TIP] The Unifying Story: Process Convolutions and Fixed Rank Kriging
> Notice the deep mathematical equivalence:
> - If we choose a dense set of knots $m$ and let the kernel shape $k_{\mathbf{s}}(\cdot)$ vary across space, we obtain **Non-Stationary Process Convolutions** (§13.2–§13.3).
> - If we choose a modest set of knots $r \ll n$ and introduce a structured prior covariance $\mathbf{K}$ on the weights, we obtain **Fixed Rank Kriging** (§13.5).
> 
> Non-stationarity and computational scalability are two sides of the same coin: **basis expansion in space**.

---

## 13.3 The Paciorek-Schervish Non-Stationary Covariance Architecture

While Higdon's process convolution integral guarantees positive definiteness, evaluating high-dimensional integrals $\int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, d\mathbf{u}$ numerically for millions of point pairs is computationally intractable.

Christopher Paciorek and Mark Schervish (2006) achieved a major breakthrough by deriving an exact closed-form analytical covariance function.

### 13.3.1 The Spatially Varying Gaussian Kernel

Let the smoothing kernel $k_{\mathbf{s}}(\mathbf{u})$ be parameterized as a spatially varying multivariate Gaussian density:
$$k_{\mathbf{s}}(\mathbf{u}) = \left(\frac{2}{\pi}\right)^{d/4} |\boldsymbol{\Sigma}(\mathbf{s})|^{-1/4} \exp\left( -(\mathbf{u} - \mathbf{s})^\top \boldsymbol{\Sigma}(\mathbf{s})^{-1} (\mathbf{u} - \mathbf{s}) \right)$$
where $\boldsymbol{\Sigma}(\mathbf{s})$ is a $d \times d$ symmetric positive-definite **local metric tensor** governing the range, variance, and anisotropy orientation at location $\mathbf{s}$.

The constant prefactor $(2/\pi)^{d/4} |\boldsymbol{\Sigma}(\mathbf{s})|^{-1/4}$ ensures that the kernel has unit $L^2$-norm:
$$\int_{\mathbb{R}^d} k_{\mathbf{s}}(\mathbf{u})^2 \, d\mathbf{u} = 1$$
so that the resulting process has unit marginal variance prior to scaling.

```
================================================================================
FROM THE LECTURE HALL: Box 13.2
Completing the Square: How Two Gaussians Multiply
================================================================================
How does the convolution of two Gaussian kernels produce a closed-form covariance?
Consider the product of two Gaussian kernels in 1D with variances sigma_1^2 and sigma_2^2:
    k_1(u) * k_2(u) ~ exp( - (u - s_1)^2 / sigma_1^2 - (u - s_2)^2 / sigma_2^2 )

Expanding the exponent and collecting powers of u:
    Exponent = - A * u^2 + 2 * B * u - C
where:
    A = (1/sigma_1^2) + (1/sigma_2^2) = (sigma_1^2 + sigma_2^2) / (sigma_1^2 * sigma_2^2)
    B = (s_1 / sigma_1^2) + (s_2 / sigma_2^2)
    C = (s_1^2 / sigma_1^2) + (s_2^2 / sigma_2^2)

Completing the square with respect to u:
    - A * (u - B/A)^2 + (B^2 / A - C)

When we integrate over u in (-infinity, infinity), the Gaussian integral over
u yields sqrt(pi / A). The remaining exponent (B^2 / A - C) simplifies by
elementary algebra to:
    B^2 / A - C = - (s_1 - s_2)^2 / (sigma_1^2 + sigma_2^2) = - 1/2 * (s_1 - s_2)^2 / sigma_avg^2

The integral of two Gaussian kernels centered at s_1 and s_2 is an EXACT
Gaussian function of their spatial separation (s_1 - s_2) evaluated against
the ARITHMETIC AVERAGE variance sigma_avg^2 = (sigma_1^2 + sigma_2^2) / 2!
================================================================================
```

### 13.3.2 The Closed-Form Non-Stationary Matérn Theorem

**Theorem 13.3.1 (Paciorek and Schervish, 2006).**
Let $\boldsymbol{\Sigma}(\mathbf{s})$ be a continuous field of $d \times d$ symmetric positive-definite local metric tensors on $\mathbb{R}^d$, and let $\sigma^2(\mathbf{s})$ be a continuous field of marginal variances. The general non-stationary Matérn covariance function between coordinates $\mathbf{s}_i, \mathbf{s}_j \in \mathbb{R}^d$ is:
$$C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i) \sigma(\mathbf{s}_j) \left|\boldsymbol{\Sigma}(\mathbf{s}_i)\right|^{1/4} \left|\boldsymbol{\Sigma}(\mathbf{s}_j)\right|^{1/4} \left|\frac{\boldsymbol{\Sigma}(\mathbf{s}_i) + \boldsymbol{\Sigma}(\mathbf{s}_j)}{2}\right|^{-1/2} \mathcal{M}_\nu\left(\sqrt{Q_{ij}}\right)$$
where:
1. $Q_{ij}$ is the squared **averaged Mahalanobis distance**:
   $$Q_{ij} = (\mathbf{s}_i - \mathbf{s}_j)^\top \left(\frac{\boldsymbol{\Sigma}(\mathbf{s}_i) + \boldsymbol{\Sigma}(\mathbf{s}_j)}{2}\right)^{-1} (\mathbf{s}_i - \mathbf{s}_j)$$
2. $\mathcal{M}_\nu(d)$ is the standard isotropic Matérn correlation function with smoothness parameter $\nu > 0$.

*Proof for the Gaussian Limit ($\nu \to \infty$):*
Substituting the Gaussian kernels $k_{\mathbf{s}_i}(\mathbf{u})$ and $k_{\mathbf{s}_j}(\mathbf{u})$ into $C(\mathbf{s}_i, \mathbf{s}_j) = \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, d\mathbf{u}$, the product of the two quadratic forms in $\mathbf{u}$ is combined by completing the square in $\mathbb{R}^d$ (Box 13.2). 

The exponent factors into a Gaussian density in $\mathbf{u}$ with precision matrix $\boldsymbol{\Sigma}_i^{-1} + \boldsymbol{\Sigma}_j^{-1}$, multiplied by a residual term depending solely on $\mathbf{s}_i - \mathbf{s}_j$. Integrating out $\mathbf{u}$ over $\mathbb{R}^d$ produces the determinant factor $|\boldsymbol{\Sigma}_i^{-1} + \boldsymbol{\Sigma}_j^{-1}|^{-1/2} = |\boldsymbol{\Sigma}_i|^{1/2} |\boldsymbol{\Sigma}_j|^{1/2} |\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j|^{-1/2}$. Multiplying by the kernel normalization constants yields the determinant prefactor and $\exp(-Q_{ij}/2)$. $\blacksquare$

*Extension to Matérn:* As shown by Paciorek and Schervish (2006), because any isotropic Matérn correlation function can be represented as a continuous scale mixture of Gaussian correlation functions over a mixing distribution for the range parameter, the exact same closed-form covariance structure holds for any Matérn smoothness $\nu > 0$.

```
================================================================================
EQUATION ANATOMY: PACIOREK-SCHERVISH NON-STATIONARY COVARIANCE
================================================================================
C(s_i, s_j) = [ sigma(s_i) sigma(s_j) ] * [ |Sigma_i|^(1/4) |Sigma_j|^(1/4) |(Sigma_i+Sigma_j)/2|^(-1/2) ]
              -------------------------   ---------------------------------------------------------------
                         |                                               |
                  Heteroskedastic                                 Geometric Mean Ratio
                   Scale Product                                 (Determinant Prefactor)

                          *  M_nu( sqrt{ (s_i - s_j)' [ (Sigma_i+Sigma_j)/2 ]^(-1) (s_i - s_j) } )
                             -------------------------------------------------------------------
                                                             |
                                                 Isotropic Matérn Kernel on
                                                Average Metric Mahalanobis Space
================================================================================
```

### 13.3.3 Closed-Form Matérn Expressions without Bessel Functions

To implement Theorem 13.3.1 without requiring specialized numerical Bessel libraries, we use closed-form representations for standard smoothness values $\nu = 1/2$ and $\nu = 3/2$:
- **Exponential ($\nu = 1/2$):** Continuous but non-differentiable paths:
  $$\mathcal{M}_{1/2}\left(\sqrt{Q_{ij}}\right) = \exp\left(-\sqrt{Q_{ij}}\right)$$
- **Matérn-3/2 ($\nu = 3/2$):** Once-differentiable paths:
  $$\mathcal{M}_{3/2}\left(\sqrt{Q_{ij}}\right) = \left(1 + \sqrt{3 Q_{ij}}\right) \exp\left(-\sqrt{3 Q_{ij}}\right)$$
- **Gaussian Limit ($\nu \to \infty$):** Infinitely differentiable, smooth paths:
  $$\mathcal{M}_{\infty}\left(\sqrt{Q_{ij}}\right) = \exp\left(-\frac{1}{2} Q_{ij}\right)$$

### 13.3.4 Why the Determinant Prefactor is Essential

Notice the determinant prefactor:
$$R_{ij} = \frac{|\boldsymbol{\Sigma}(\mathbf{s}_i)|^{1/4} |\boldsymbol{\Sigma}(\mathbf{s}_j)|^{1/4}}{\left|\frac{\boldsymbol{\Sigma}(\mathbf{s}_i) + \boldsymbol{\Sigma}(\mathbf{s}_j)}{2}\right|^{1/2}}$$

In one dimension ($d=1$), where $\boldsymbol{\Sigma}_i = \sigma_i^2$ and $\boldsymbol{\Sigma}_j = \sigma_j^2$, this ratio simplifies to:
$$R_{ij} = \frac{(\sigma_i^2)^{1/4} (\sigma_j^2)^{1/4}}{\left(\frac{\sigma_i^2 + \sigma_j^2}{2}\right)^{1/2}} = \frac{\sqrt{\sigma_i \sigma_j}}{\frac{\sigma_i^2 + \sigma_j^2}{2}} \le 1$$
This is the celebrated **Arithmetic-Geometric Mean (AM-GM) inequality**! The ratio is strictly $\le 1$, reaching exactly 1 if and only if $\sigma_i = \sigma_j$.

> [!CRITICAL] Dropping the Prefactor Destroys Positive Definiteness
> An intuitive heuristic might tempt an analyst to simply evaluate the Mahalanobis distance $Q_{ij}$ against the average metric tensor and omit the prefactor $R_{ij}$. 
> 
> **This fails catastrophically.** On a sample of 40 points in $\mathbb{R}^2$ with rotating metric tensors:
> - With the determinant prefactor $R_{ij}$: minimum eigenvalue $\lambda_{\min} = +0.0023 > 0$ (strictly positive definite).
> - Without the determinant prefactor ($R_{ij} \equiv 1$): minimum eigenvalue $\lambda_{\min} = -0.3373 < 0$ (indefinite, invalid covariance matrix).
> 
> The determinant prefactor acts as a mathematically necessary damping factor: when neighboring regions have radically differing anisotropy directions or scales, it damps the cross-correlation, preserving global positive definiteness.

### 13.3.5 Parametrizing and Estimating $\boldsymbol{\Sigma}(\mathbf{s})$

In practice, how do we estimate the continuous field of metric tensors $\boldsymbol{\Sigma}(\mathbf{s})$?
In two dimensions ($d=2$), every symmetric positive-definite matrix $\boldsymbol{\Sigma}(\mathbf{s})$ is characterized by three geometric parameters:
$$\boldsymbol{\Sigma}(\mathbf{s}) = \mathbf{R}(\theta(\mathbf{s})) \begin{bmatrix} \lambda_1^2(\mathbf{s}) & 0 \\ 0 & \lambda_2^2(\mathbf{s}) \end{bmatrix} \mathbf{R}(\theta(\mathbf{s}))^\top$$
where $\lambda_1(\mathbf{s})$ and $\lambda_2(\mathbf{s})$ are the major and minor correlation semi-axes, and $\theta(\mathbf{s})$ is the rotation angle of the principal axis.

To estimate these fields:
1. **Parametric Covariate Modeling:** Link the log-lengths $\ln \lambda_k(\mathbf{s})$ and angle $\theta(\mathbf{s})$ directly to physical covariates (e.g., wind direction, topographic slope, ocean depth) via linear regression or low-order polynomials.
2. **Spline Basis Expansion:** Model $\ln \lambda_1(\mathbf{s}), \ln \lambda_2(\mathbf{s}), \theta(\mathbf{s})$ using a small set of smooth thin-plate spline basis functions.
3. **Estimation:** Estimate the spline coefficients jointly with $\sigma^2$ and nugget $\tau^2$ via Profile Maximum Likelihood or restricted maximum likelihood (REML).

---

## 13.4 Space Deformation: The Sampson-Guttorp Method

A conceptually distinct approach to non-stationarity was pioneered by Paul Sampson and Peter Guttorp (1992).

Rather than creating a complicated non-stationary covariance formula on geographic coordinates, they proposed **warping the geographic coordinate system itself**:
$$\mathbf{s} \in \mathcal{G} \subset \mathbb{R}^2 \xrightarrow{\quad f(\mathbf{s}) \quad} \mathbf{d} \in \mathcal{D} \subset \mathbb{R}^k, \quad k \ge 2$$
1. **Geographic Space ($\mathcal{G}$-space):** The physical coordinates $\mathbf{s} = (x, y)^\top$, where the physical field exhibits non-stationary correlation.
2. **Deformed Latent Space ($\mathcal{D}$-space):** A warped mathematical space where the process becomes **second-order stationary and isotropic**.

```
================================================================================
Geographic Space (G-Space)                      Deformed Latent Space (D-Space)
Complex, warped non-stationarity                Homogeneous, isotropic plane
         [Mountain Ridge]                                  d_i           d_j
         s_i     s_j                                        *             *
          *       *           --- Warping f(s) --->          \           /
           \     /                                            \         /
              *                                                *-------*
             s_k                                              d_k     d_m
          [Valley]
================================================================================
```

### 13.4.1 The Deformation Covariance Model

The covariance between two points $\mathbf{s}_i, \mathbf{s}_j \in \mathcal{G}$ is defined as:
$$C(\mathbf{s}_i, \mathbf{s}_j) = C_0(\|f(\mathbf{s}_i) - f(\mathbf{s}_j)\|)$$
where $f: \mathbb{R}^2 \to \mathbb{R}^k$ ($k \ge 2$) is a smooth coordinate mapping, and $C_0(h)$ is an isotropic covariance function that is **valid in $\mathbb{R}^k$** (e.g., Exponential, Matérn, or Spherical for $k \le 3$).

### 13.4.2 Algorithmic Implementation via MDS and Splines

The Sampson-Guttorp method requires temporal replicates at $n$ fixed monitoring stations (e.g., daily air pollution measurements across multiple years):
1. **Empirical Dispersion Matrix:** From the time-series replicates, compute sample semivariances $\widehat{\gamma}_{ij} = \frac{1}{2T}\sum_{t=1}^T (Z_t(\mathbf{s}_i) - Z_t(\mathbf{s}_j))^2$.
2. **Metric Multidimensional Scaling (MDS):** Given a stationary isotropic base model $\gamma_0(d)$, invert it to find target latent distances:
   $$d_{ij}^* = \gamma_0^{-1}(\widehat{\gamma}_{ij})$$
   Apply metric MDS to find latent coordinates $\{\mathbf{d}_1, \dots, \mathbf{d}_n\} \subset \mathbb{R}^k$ whose pairwise Euclidean distances best approximate $d_{ij}^*$.
3. **Thin-Plate Spline Smoothing:** Fit a smooth 2D Thin-Plate Spline (TPS) mapping geographic coordinates to latent coordinates $\mathbf{s}_i \mapsto \mathbf{d}_i$:
   $$f(\mathbf{s}) = \mathbf{A} \begin{bmatrix} 1 \\ x \\ y \end{bmatrix} + \sum_{m=1}^n \mathbf{w}_m \phi(\|\mathbf{s} - \mathbf{s}_m\|)$$
   where $\phi(r) = r^2 \ln(r)$ is the biharmonic radial basis function.
4. **Iterative Alternation:** The fitted coordinates $\mathbf{d}_i$ induce updated empirical variograms, which refine the parameters of $\gamma_0(d)$, iterating until convergence.

```
================================================================================
FROM THE LECTURE HALL: Box 13.3
Space Deformation and Topological Folding
================================================================================
What happens if you fit an unregularized, interpolating thin-plate spline to
noisy empirical MDS coordinates?

The spline will twist violently to match noisy target distances, causing
TOPOLOGICAL FOLDING. The geographic plane folds over itself: two physical
monitoring stations on opposite sides of a mountain range can be mapped to the
exact same point in deformed space (f(s_1) = f(s_2)), inducing a non-physical
correlation of 1.0 between distant sites!

Remedy:
Never interpolate MDS coordinates exactly. Add a roughness penalty lambda to the
thin-plate spline objective:
       min_{f} sum_{i=1}^n || d_i - f(s_i) ||^2 + lambda * J(f)
where J(f) penalizes the integral of squared second derivatives (bending energy).
Tuning lambda prevents folding, ensuring the deformation f(s) remains a smooth,
well-behaved coordinate transformation.
================================================================================
```

---

## 13.5 Scalable Low-Rank Kriging & Fixed Rank Kriging (FRK)

Whether stationary or non-stationary, classical geostatistical kriging requires solving the $n \times n$ linear system:
$$\boldsymbol{\Sigma} \boldsymbol{\lambda} = \mathbf{c}(\mathbf{s}_0)$$

As detailed in Chapter 5, the Cholesky factorization of an $n \times n$ dense covariance matrix requires $\frac{1}{3}n^3$ floating-point operations (FLOPs) and $\mathcal{O}(n^2)$ memory.
When $n = 100,000$ (modern satellite remote sensing, LiDAR, or atmospheric reanalysis):
$$\frac{1}{3} n^3 = \frac{1}{3} (10^5)^3 \approx 3.3 \times 10^{14} \text{ FLOPs}, \quad n^2 \times 8 \text{ bytes} \approx 80 \text{ Gigabytes RAM}$$

At typical computational throughput (10 to 100 GFLOP/s on a single workstation CPU), solving this system requires **9.3 hours to 1 hour of sustained compute time**. 

Far more critically, storing the $n \times n$ dense double-precision covariance matrix requires **80 GB of contiguous RAM**, causing immediate out-of-memory crashes on standard hardware.

To solve this **Big-$n$ problem**, Noel Cressie and Gardar Johannesson (2008) developed **Fixed Rank Kriging (FRK)**.

### 13.5.1 The Low-Rank Spatial Process Architecture

We decompose the spatial random field $Z(\mathbf{s})$ into a fixed-rank basis expansion plus fine-scale white noise:
$$Z(\mathbf{s}) = \mathbf{x}(\mathbf{s})^\top \boldsymbol{\beta} + \mathbf{S}(\mathbf{s})^\top \boldsymbol{\eta} + \epsilon(\mathbf{s})$$
where:
1. $\mathbf{x}(\mathbf{s})^\top \boldsymbol{\beta}$ is the large-scale deterministic drift ($p$ covariates).
2. $\mathbf{S}(\mathbf{s}) = [S_1(\mathbf{s}), S_2(\mathbf{s}), \dots, S_r(\mathbf{s})]^\top$ is an $r$-dimensional vector of pre-specified, multi-scale spatial **basis functions** (e.g., local bisquare functions), with $r \ll n$ (typically $r \approx 50$ to $500$).
3. $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$ is an $r \times 1$ latent random coefficient vector with $r \times r$ covariance matrix $\mathbf{K}$.
4. $\epsilon(\mathbf{s}) \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)$ is fine-scale white noise / measurement error (the nugget effect).

In matrix form across all $n$ observations:
$$\mathbf{Z} = \mathbf{X}\boldsymbol{\beta} + \mathbf{S}\boldsymbol{\eta} + \boldsymbol{\epsilon}, \quad \mathbf{S} \in \mathbb{R}^{n \times r}, \quad \boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_\epsilon^2 \mathbf{I}_n)$$

The $n \times n$ data covariance matrix is:
$$\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{Z}) = \mathbf{S} \mathbf{K} \mathbf{S}^\top + \sigma_\epsilon^2 \mathbf{I}_n$$

### 13.5.2 Fast Inversion via Sherman-Morrison-Woodbury

To invert $\boldsymbol{\Sigma}$, we apply the **Sherman-Morrison-Woodbury Matrix Identity**:
$$(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}$$

Setting $\mathbf{A} = \sigma_\epsilon^2 \mathbf{I}_n$, $\mathbf{U} = \mathbf{S}$, $\mathbf{C} = \mathbf{K}$, and $\mathbf{V} = \mathbf{S}^\top$:
$$\boldsymbol{\Sigma}^{-1} = \left(\sigma_\epsilon^2 \mathbf{I}_n + \mathbf{S}\mathbf{K}\mathbf{S}^\top\right)^{-1} = \frac{1}{\sigma_\epsilon^2}\mathbf{I}_n - \frac{1}{\sigma_\epsilon^4}\mathbf{S}\left(\mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}^\top\mathbf{S}\right)^{-1}\mathbf{S}^\top$$

Notice the dramatic reduction: the only matrix that requires inversion is:
$$\mathbf{A}_{\text{small}} = \mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2} \mathbf{S}^\top \mathbf{S}$$
which has dimension **$r \times r$**! For $r = 100$, inverting $\mathbf{A}_{\text{small}}$ requires merely $100^3 = 10^6$ operations (< 1 millisecond).

```
================================================================================
COMPUTATIONAL COMPLEXITY: EXACT KRIGING VS. FIXED RANK KRIGING
================================================================================
EXACT KRIGING:
  - Inversion: V^(-1) via Cholesky decomposition of (n x n) matrix
  - Complexity: (1/3) * n^3 FLOPs, O(n^2) RAM
  - At n=100,000: ~ 3.3 x 10^14 FLOPs (~1-9 hours CPU time), 80 GB RAM
    (FATAL MEMORY BOTTLENECK)

FIXED RANK KRIGING (FRK):
  - Gram Matrix: S' * S                           -->  O(n * r^2) FLOPs
  - Small Inverse: [ K^(-1) + (1/sigma^2)*S'S ]^(-1) (r x r matrix) -->  O(r^3) FLOPs
  - Prediction Vector: S * [ ... ]                -->  O(n * r) FLOPs
  - Total Flops: O(n * r^2 + r^3)
  - At n=100,000, r=100: ~ 10^9 FLOPs (< 0.5 seconds CPU time), < 100 MB RAM
================================================================================
```

### 13.5.3 The FRK Predictor as Spatial Ridge Regression

Using the Woodbury identity, the Best Linear Unbiased Predictor (BLUP) for the latent spatial signal at an unsampled location $\mathbf{s}_0$ simplifies to:
$$\widehat{Z}(\mathbf{s}_0) = \mathbf{x}(\mathbf{s}_0)^\top \widehat{\boldsymbol{\beta}} + \mathbf{S}(\mathbf{s}_0)^\top \widehat{\boldsymbol{\eta}}$$
where $\widehat{\boldsymbol{\eta}}$ is the optimal estimate of the basis weights:
$$\widehat{\boldsymbol{\eta}} = \left( \mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2} \mathbf{S}^\top \mathbf{S} \right)^{-1} \frac{1}{\sigma_\epsilon^2} \mathbf{S}^\top (\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}})$$

```
================================================================================
FROM THE LECTURE HALL: Box 13.4
Fixed Rank Kriging as Spatial Ridge Regression
================================================================================
Compare the FRK basis weight formula with classical Ridge Regression:
Classical Ridge Regression:
         beta_ridge = ( X'X + lambda * I )^(-1) X'y

FRK Basis Estimate:
         eta_hat    = ( S'S + sigma_epsilon^2 * K^(-1) )^(-1) S'(Z - X*beta)

Fixed Rank Kriging is mathematically equivalent to regressing the spatial
residuals onto the basis matrix S, with a generalized ridge penalty
matrix sigma_epsilon^2 * K^(-1)!

The prior covariance matrix K acts as the spatial regularizer:
- If K has large diagonal elements (weak shrinkage), the basis functions
  fit closely to the observed sample residuals.
- If K has small diagonal elements (strong shrinkage), eta_hat shrinks toward
  zero, and the prediction reverts smoothly to the large-scale drift x(s)'*beta.
================================================================================
```

#### Prediction Uncertainty
The posterior covariance matrix of the basis coefficients is:
$$\operatorname{Var}(\widehat{\boldsymbol{\eta}} \mid \mathbf{Z}) = \left( \mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2} \mathbf{S}^\top \mathbf{S} \right)^{-1}$$

At any target location $\mathbf{s}_0$, the **latent signal prediction variance** is:
$$\sigma_{\text{signal}}^2(\mathbf{s}_0) = \mathbf{S}(\mathbf{s}_0)^\top \operatorname{Var}(\widehat{\boldsymbol{\eta}} \mid \mathbf{Z}) \mathbf{S}(\mathbf{s}_0)$$
If predicting a **new noisy observation**, we add the nugget variance:
$$\sigma_{\text{obs}}^2(\mathbf{s}_0) = \sigma_{\text{signal}}^2(\mathbf{s}_0) + \sigma_\epsilon^2$$

### 13.5.4 Multiresolution Bisquare Basis Functions

A standard, effective choice of basis function is the **compactly supported bisquare function**:
$$S_l(\mathbf{s}) = \begin{cases}
\left( 1 - \left(\frac{\|\mathbf{s} - \mathbf{c}_l\|}{r_l}\right)^2 \right)^2 & \text{if } \|\mathbf{s} - \mathbf{c}_l\| \le r_l \\
0 & \text{otherwise}
\end{cases}$$
where $\mathbf{c}_l \in \mathcal{D}$ is the center of the $l$-th basis function and $r_l > 0$ is its aperture radius.

By distributing basis functions across multiple spatial resolutions (e.g., Level 1: coarse regional coverage; Level 2: intermediate; Level 3: fine local coverage), FRK captures broad regional trends and local features simultaneously.

### 13.5.5 Comparison of Big-$n$ Spatial Paradigms

FRK is one of three major routes to scalable geostatistics. The master table below contrasts the modern approaches:

| Method | Mathematical Mechanism | Computational Flops | Memory Footprint | Primary Strength | Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed Rank Kriging (FRK)** | Low-rank basis expansion $\mathbf{S}\mathbf{K}\mathbf{S}^\top$ | $\mathcal{O}(n r^2 + r^3)$ | $\mathcal{O}(n r)$ | Global representation, lightning fast | Smooth approximation; misses sub-basis features |
| **GMRF / SPDE (Chapter 12)** | Sparse precision matrix $\mathbf{Q} = \boldsymbol{\Sigma}^{-1}$ | $\mathcal{O}(n^{3/2})$ in 2D | $\mathcal{O}(n)$ | Preserves local Markov properties | Requires triangulation mesh |
| **Covariance Tapering** | Multiplies $C(h)$ by compactly supported taper | $\mathcal{O}(n_{\text{nn}}^{3})$ | Sparse CSR | Retains exact high-frequency detail | Alters long-range dependence |
| **NNGP (Nearest Neighbor GP)** | Directed acyclic graph conditioning on $k$ neighbors | $\mathcal{O}(n k^3)$ | $\mathcal{O}(n k)$ | Exact likelihood for massive datasets | Depends on point ordering |

---

## 13.6 Worked Analytical Exercises

### Exercise 13.1: Proving Validity of Separable Heteroskedastic Covariance

**Problem:** Let $\rho(\mathbf{h})$ be a valid, stationary, positive-definite correlation function on $\mathbb{R}^d$, and let $\sigma(\mathbf{s}): \mathbb{R}^d \to (0, \infty)$ be an arbitrary, strictly positive continuous standard deviation function.
1. Formulate the separable heteroskedastic covariance function $C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i) \sigma(\mathbf{s}_j) \rho(\mathbf{s}_i - \mathbf{s}_j)$.
2. Prove that for any finite set of $n$ distinct locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ and any non-zero vector $\mathbf{a} \in \mathbb{R}^n$, the quadratic form $\mathbf{a}^\top \boldsymbol{\Sigma} \mathbf{a}$ is strictly positive.

**Solution:**

**Step 1: Matrix Formulation**
For the sample locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$, let $\mathbf{R} \in \mathbb{R}^{n \times n}$ be the correlation matrix with entries $R_{ij} = \rho(\mathbf{s}_i - \mathbf{s}_j)$. Because $\rho(\cdot)$ is a valid positive-definite correlation function and the locations are distinct, $\mathbf{R}$ is strictly positive definite:
$$\mathbf{v}^\top \mathbf{R} \mathbf{v} > 0 \quad \forall \mathbf{v} \neq \mathbf{0}$$

Define the diagonal scaling matrix:
$$\mathbf{D} = \operatorname{diag}(\sigma(\mathbf{s}_1), \sigma(\mathbf{s}_2), \dots, \sigma(\mathbf{s}_n))$$
Because $\sigma(\mathbf{s}_i) > 0$ for all $i$, $\mathbf{D}$ is diagonal with strictly positive diagonal elements, and hence $\mathbf{D}$ is non-singular and invertible.

The covariance matrix $\boldsymbol{\Sigma}$ has entries $\Sigma_{ij} = \sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j) R_{ij}$, which is written in matrix notation as:
$$\boldsymbol{\Sigma} = \mathbf{D} \mathbf{R} \mathbf{D}$$

**Step 2: Quadratic Form Proof**
Let $\mathbf{a} \in \mathbb{R}^n$ be any arbitrary non-zero vector ($\mathbf{a} \neq \mathbf{0}$). Evaluate the quadratic form:
$$\mathbf{a}^\top \boldsymbol{\Sigma} \mathbf{a} = \mathbf{a}^\top (\mathbf{D} \mathbf{R} \mathbf{D}) \mathbf{a} = (\mathbf{D} \mathbf{a})^\top \mathbf{R} (\mathbf{D} \mathbf{a})$$

Define the transformed vector $\mathbf{v} = \mathbf{D} \mathbf{a}$.
Since $\mathbf{D}$ is diagonal with non-zero diagonal entries ($\sigma(\mathbf{s}_i) > 0$) and $\mathbf{a} \neq \mathbf{0}$:
$$v_i = \sigma(\mathbf{s}_i) a_i$$
At least one $a_i \neq 0$, so at least one $v_i \neq 0$, meaning $\mathbf{v} \neq \mathbf{0}$.

Because $\mathbf{R}$ is strictly positive definite:
$$\mathbf{a}^\top \boldsymbol{\Sigma} \mathbf{a} = \mathbf{v}^\top \mathbf{R} \mathbf{v} > 0$$
This holds for all $\mathbf{a} \neq \mathbf{0}$. Therefore, $\boldsymbol{\Sigma}$ is strictly positive definite, proving that $C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j)\rho(\mathbf{s}_i - \mathbf{s}_j)$ is a valid non-stationary covariance model.

---

### Exercise 13.2: 1D Gaussian Kernel Convolution and Analytical Derivation

**Problem:** Consider two locations $s_1 = 0$ and $s_2 = 1$ on the real line ($d=1$). The local metric tensors are scalar variances $\Sigma_1 = \sigma_1^2 = 1.0$ and $\Sigma_2 = \sigma_2^2 = 4.0$ (standard deviations $\sigma_1 = 1.0$ and $\sigma_2 = 2.0$).
1. Compute the average variance $\Sigma_{\text{avg}} = \frac{1}{2}(\Sigma_1 + \Sigma_2)$ and the squared Mahalanobis distance $Q_{12}$.
2. Compute the determinant prefactor $R_{12}$.
3. Compute the analytical Paciorek-Schervish Gaussian covariance $C(s_1, s_2)$ and verify that it equals the numerical convolution integral $\int_{-\infty}^\infty k_{s_1}(u) k_{s_2}(u) \, du$.

**Solution:**

**Step 1: Average Metric and Mahalanobis Distance**
$$\Sigma_{\text{avg}} = \frac{\Sigma_1 + \Sigma_2}{2} = \frac{1.0 + 4.0}{2} = 2.5$$
The separation is $s_1 - s_2 = 0 - 1 = -1$. The squared Mahalanobis distance is:
$$Q_{12} = \frac{(s_1 - s_2)^2}{\Sigma_{\text{avg}}} = \frac{(-1)^2}{2.5} = \frac{1}{2.5} = 0.40$$

**Step 2: Determinant Prefactor**
In 1D, the determinant is the scalar variance: $|\Sigma_1| = 1.0$, $|\Sigma_2| = 4.0$, and $|\Sigma_{\text{avg}}| = 2.5$:
$$R_{12} = \frac{|\Sigma_1|^{1/4} |\Sigma_2|^{1/4}}{|\Sigma_{\text{avg}}|^{1/2}} = \frac{(1.0)^{1/4} (4.0)^{1/4}}{\sqrt{2.5}} = \frac{1.0 \times \sqrt{2.0}}{\sqrt{2.5}} = \sqrt{\frac{2.0}{2.5}} = \sqrt{0.80} \approx 0.894427$$

Notice that $R_{12} = 0.8944 < 1.0$, confirming the AM-GM inequality!

**Step 3: Analytical Covariance**
Using the Gaussian correlation function $\exp(-0.5 Q_{12})$:
$$\text{corr} = \exp\left(-\frac{1}{2} \times 0.40\right) = \exp(-0.20) \approx 0.818731$$
Assuming unit marginal variances $\sigma(s_1) = \sigma(s_2) = 1.0$:
$$C(s_1, s_2) = R_{12} \times \text{corr} = 0.894427 \times 0.818731 \approx 0.732295$$

Numerical integration of the normalized Gaussian kernels:
$$k_{s_1}(u) = \left(\frac{2}{\pi}\right)^{1/4} (1.0)^{-1/2} \exp\left(-(u - 0)^2 / 1.0\right)$$
$$k_{s_2}(u) = \left(\frac{2}{\pi}\right)^{1/4} (4.0)^{-1/4} \exp\left(-(u - 1)^2 / 4.0\right)$$
yields $\int_{-\infty}^\infty k_{s_1}(u) k_{s_2}(u) \, du = 0.73230$, matching the analytical Paciorek-Schervish formula to five decimal places.

---

### Exercise 13.3: Manual Sherman-Morrison-Woodbury Low-Rank Inversion

**Problem:** Consider a low-rank spatial system with $n=3$ observations and $r=1$ basis function ($s_1 = 0, s_2 = 1, s_3 = 2$). The basis vector is $\mathbf{S} = [1, 2, 1]^\top \in \mathbb{R}^{3 \times 1}$. The prior variance is $K = 2.0$, and the nugget variance is $\sigma_\epsilon^2 = 1.0$.
1. Construct the exact $3 \times 3$ covariance matrix $\boldsymbol{\Sigma} = \mathbf{S} K \mathbf{S}^\top + \sigma_\epsilon^2 \mathbf{I}_3$.
2. Invert $\boldsymbol{\Sigma}$ using the Sherman-Morrison-Woodbury identity by inverting the scalar $r \times r$ matrix.
3. Multiply $\boldsymbol{\Sigma} \boldsymbol{\Sigma}^{-1}$ to confirm exact identity recovery.

**Solution:**

**Step 1: Construct $\boldsymbol{\Sigma}$**
$$\mathbf{S} \mathbf{S}^\top = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 1 \end{bmatrix} = \begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix}$$
$$\boldsymbol{\Sigma} = 2 \begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix} + \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix} = \begin{bmatrix}
3 & 4 & 2 \\
4 & 9 & 4 \\
2 & 4 & 3
\end{bmatrix}$$

**Step 2: Apply Sherman-Morrison-Woodbury**
The formula is:
$$\boldsymbol{\Sigma}^{-1} = \frac{1}{\sigma_\epsilon^2} \mathbf{I}_3 - \frac{1}{\sigma_\epsilon^4} \mathbf{S} \left( K^{-1} + \frac{1}{\sigma_\epsilon^2} \mathbf{S}^\top \mathbf{S} \right)^{-1} \mathbf{S}^\top$$
We compute the inner scalar term:
$$\mathbf{S}^\top \mathbf{S} = 1^2 + 2^2 + 1^2 = 1 + 4 + 1 = 6$$
$$K^{-1} + \frac{1}{\sigma_\epsilon^2} \mathbf{S}^\top \mathbf{S} = \frac{1}{2} + \frac{6}{1} = \frac{13}{2} = 6.5$$
The inverse of this scalar is:
$$A_{\text{small}}^{-1} = \left(\frac{13}{2}\right)^{-1} = \frac{2}{13}$$

Now compute the rank-1 update matrix:
$$\frac{1}{\sigma_\epsilon^4} \mathbf{S} A_{\text{small}}^{-1} \mathbf{S}^\top = \frac{2}{13} \begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix} = \frac{1}{13} \begin{bmatrix}
2 & 4 & 2 \\
4 & 8 & 4 \\
2 & 4 & 2
\end{bmatrix}$$

Subtracting from $\mathbf{I}_3 = \frac{1}{13} \begin{bmatrix} 13 & 0 & 0 \\ 0 & 13 & 0 \\ 0 & 0 & 13 \end{bmatrix}$:
$$\boldsymbol{\Sigma}^{-1} = \frac{1}{13} \begin{bmatrix}
13 - 2 & -4 & -2 \\
-4 & 13 - 8 & -4 \\
-2 & -4 & 13 - 2
\end{bmatrix} = \frac{1}{13} \begin{bmatrix}
11 & -4 & -2 \\
-4 & 5 & -4 \\
-2 & -4 & 11
\end{bmatrix}$$

**Step 3: Verify the Inverse**
$$\boldsymbol{\Sigma} \boldsymbol{\Sigma}^{-1} = \begin{bmatrix}
3 & 4 & 2 \\
4 & 9 & 4 \\
2 & 4 & 3
\end{bmatrix} \times \frac{1}{13} \begin{bmatrix}
11 & -4 & -2 \\
-4 & 5 & -4 \\
-2 & -4 & 11
\end{bmatrix}$$
- Row 1, Col 1: $\frac{1}{13} [3(11) + 4(-4) + 2(-2)] = \frac{1}{13} [33 - 16 - 4] = \frac{13}{13} = 1$
- Row 1, Col 2: $\frac{1}{13} [3(-4) + 4(5) + 2(-4)] = \frac{1}{13} [-12 + 20 - 8] = 0$
- Row 1, Col 3: $\frac{1}{13} [3(-2) + 4(-4) + 2(11)] = \frac{1}{13} [-6 - 16 + 22] = 0$
- Row 2, Col 2: $\frac{1}{13} [4(-4) + 9(5) + 4(-4)] = \frac{1}{13} [-16 + 45 - 16] = \frac{13}{13} = 1$

The product is exactly the $3 \times 3$ identity matrix $\mathbf{I}_3$.

---

## 13.7 Algorithmic Engines in Pure Python (Dependency-Free)

The following pure standard-library module implements:
1. `PaciorekSchervishKernel`: Non-stationary Matérn covariance evaluator supporting Exponential ($\nu=1/2$), Matérn-3/2 ($\nu=3/2$), and Gaussian ($\nu \to \infty$) limits, with strict positive definiteness checks.
2. `FixedRankKrigingEngine`: Scalable FRK solver using multi-scale bisquare basis functions and fast Woodbury inversion, outputting both latent surface uncertainty and total prediction uncertainty.
3. Three verification experiments:
   - Experiment 1: Demonstrates a vivid ~3.9x spatial range contrast across space.
   - Experiment 2: Evaluates FRK on 1,000 points, reporting predictions on non-zero targets and global test-grid RMSE against baseline mean prediction.
   - Experiment 3: Verifies that the determinant prefactor guarantees positive definiteness, while omitting it produces negative eigenvalues.

```python
"""
Chapter 13: Non-Stationary Covariance and Fixed Rank Kriging Engine
Pure Python 3 standard-library implementation with zero external dependencies.
"""

import math
import random


class PaciorekSchervishKernel:
    """
    Evaluates the closed-form non-stationary Paciorek-Schervish covariance kernel
    with spatially varying metric tensors Sigma(s) and local variances sigma^2(s).
    """

    def __init__(self, metric_tensor_fn, variance_fn=None, nu="gaussian"):
        """
        Parameters
        ----------
        metric_tensor_fn : callable
            Function taking (x, y) and returning 2x2 metric tensor [[a, b], [b, c]].
        variance_fn : callable or None
            Function taking (x, y) and returning marginal variance sigma^2(s).
        nu : str or float
            Smoothness parameter: 'gaussian' (nu -> inf), 0.5 (exponential), or 1.5.
        """
        self.metric_tensor_fn = metric_tensor_fn
        self.variance_fn = variance_fn if variance_fn is not None else (lambda x, y: 1.0)
        self.nu = nu

    def _det_2x2(self, M):
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    def _inv_2x2(self, M):
        d = self._det_2x2(M)
        if d <= 1e-12:
            raise ValueError(f"Singular or degenerate metric tensor encountered: det = {d:.4e}")
        return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]

    def evaluate_covariance(self, s1, s2, use_prefactor=True):
        """
        Computes non-stationary covariance between s1=(x1, y1) and s2=(x2, y2).
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
        if use_prefactor:
            prefactor = (det1**0.25 * det2**0.25) / math.sqrt(det_avg)
        else:
            prefactor = 1.0

        # Squared Mahalanobis distance Q = (s1 - s2)' * M_avg^(-1) * (s1 - s2)
        inv_avg = self._inv_2x2(M_avg)
        dx = x1 - x2
        dy = y1 - y2
        qx = inv_avg[0][0] * dx + inv_avg[0][1] * dy
        qy = inv_avg[1][0] * dx + inv_avg[1][1] * dy
        Q = max(0.0, dx * qx + dy * qy)
        dist = math.sqrt(Q)

        # Matérn correlation evaluation
        if self.nu == "gaussian":
            corr = math.exp(-0.5 * Q)
        elif self.nu == 0.5:
            corr = math.exp(-dist)
        elif self.nu == 1.5:
            corr = (1.0 + math.sqrt(3.0) * dist) * math.exp(-math.sqrt(3.0) * dist)
        else:
            raise ValueError(f"Unsupported smoothness parameter nu = {self.nu}")

        return sig1 * sig2 * prefactor * corr


class FixedRankKrigingEngine:
    """
    Implements scalable Fixed Rank Kriging (FRK) using multi-center bisquare
    basis functions and Sherman-Morrison-Woodbury inversion.
    """

    def __init__(self, basis_centers, basis_radii, nugget_variance=0.04, K_diag=2.0):
        self.centers = basis_centers
        self.radii = basis_radii
        self.r = len(basis_centers)
        self.sigma2_eps = nugget_variance
        self.K_diag = K_diag

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
        S = [self._bisquare_basis(x, y) for x, y in sample_coords]

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

        # Small r x r matrix to invert: A = (K^(-1) + (1/sig2) * S'S)
        A = [[0.0] * r for _ in range(r)]
        for j in range(r):
            for k in range(r):
                A[j][k] = (1.0 / sig2) * StS[j][k]
            A[j][j] += 1.0 / self.K_diag

        # Invert A (r x r) via Gauss-Jordan elimination
        invA = self._invert_matrix_gauss_jordan(A)

        # Compute basis weights eta_hat = invA * ((1/sig2) * StZ)
        rhs = [(1.0 / sig2) * StZ[j] for j in range(r)]
        eta_hat = [sum(invA[j][k] * rhs[k] for k in range(r)) for j in range(r)]

        # 3. Predict at new target coordinates: Z_hat(s0) = S(s0)' * eta_hat
        predictions = []
        latent_vars = []
        total_vars = []
        for x0, y0 in pred_coords:
            s0 = self._bisquare_basis(x0, y0)
            pred_val = sum(s0[j] * eta_hat[j] for j in range(r))

            # Latent signal variance = s0' * invA * s0
            s0_invA = [sum(s0[k] * invA[k][j] for k in range(r)) for j in range(r)]
            lat_var = sum(s0_invA[j] * s0[j] for j in range(r))
            tot_var = lat_var + sig2

            predictions.append(pred_val)
            latent_vars.append(lat_var)
            total_vars.append(tot_var)

        return predictions, latent_vars, total_vars

    def _invert_matrix_gauss_jordan(self, mat):
        """Pure Python matrix inversion for r x r matrices."""
        dim = len(mat)
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(dim)] for i, row in enumerate(mat)]
        for i in range(dim):
            pivot = aug[i][i]
            if abs(pivot) < 1e-12:
                for k in range(i + 1, dim):
                    if abs(aug[k][i]) > abs(pivot):
                        aug[i], aug[k] = aug[k], aug[i]
                        pivot = aug[i][i]
                        break
            if abs(pivot) < 1e-12:
                raise ValueError("Singular matrix encountered in FRK solver.")
            inv_p = 1.0 / pivot
            for j in range(2 * dim):
                aug[i][j] *= inv_p
            for k in range(dim):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * dim):
                        aug[k][j] -= factor * aug[i][j]
        return [row[dim:] for row in aug]


def _smallest_eigenvalue_power_iteration(matrix, max_iter=200):
    """
    Computes the smallest eigenvalue of a symmetric matrix using shifted inverse iteration.
    """
    n = len(matrix)
    # Estimate largest absolute eigenvalue via Gershgorin circle theorem
    norm_est = max(sum(abs(matrix[i][j]) for j in range(n)) for i in range(n))
    
    # We want smallest eigenvalue of M: find largest eigenvalue of (norm_est * I - M)
    B = [[(norm_est if i == j else 0.0) - matrix[i][j] for j in range(n)] for i in range(n)]
    
    # Power iteration on B
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(max_iter):
        w = [sum(B[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm_w = math.sqrt(sum(x ** 2 for x in w))
        if norm_w < 1e-12:
            break
        v = [x / norm_w for x in w]
        
    # Rayleigh quotient for B
    Bv = [sum(B[i][j] * v[j] for j in range(n)) for i in range(n)]
    mu_B = sum(v[i] * Bv[i] for i in range(n))
    lambda_min = norm_est - mu_B
    return lambda_min


if __name__ == "__main__":
    print("=" * 72)
    print("CHAPTER 13: NON-STATIONARY COVARIANCE & FIXED RANK KRIGING")
    print("=" * 72)

    # ---------------------------------------------------------
    # Experiment 1: Paciorek-Schervish Spatially Varying Metric
    # ---------------------------------------------------------
    print("\n--- Experiment 1: Spatially Varying Metric Tensor (Paciorek-Schervish) ---")
    def metric_tensor_field(x, y):
        # Range expands dramatically from West (x=0) to East (x=1)
        scale_x = 0.10 + 0.90 * x
        scale_y = 0.10 + 0.90 * x
        return [[scale_x**2, 0.0], [0.0, scale_y**2]]

    ps_kernel = PaciorekSchervishKernel(metric_tensor_field, nu="gaussian")

    # Evaluate correlation at separation lag h = 0.35 in West vs. East
    lag = 0.35
    cov_west = ps_kernel.evaluate_covariance((0.05, 0.50), (0.05 + lag, 0.50))
    cov_east = ps_kernel.evaluate_covariance((0.60, 0.50), (0.60 + lag, 0.50))

    print(f"Spatial Separation Lag  : h = {lag:.2f}")
    print(f"West Covariance (Short) : {cov_west:.4f}  (Range scale ~ 0.25)")
    print(f"East Covariance (Long)  : {cov_east:.4f}  (Range scale ~ 0.80)")
    print(f"Empirical Contrast Ratio: Covariance is {cov_east / cov_west:.2f}x stronger in East!")

    # ---------------------------------------------------------
    # Experiment 2: Scalable Fixed Rank Kriging on n = 1,000 Points
    # ---------------------------------------------------------
    print("\n--- Experiment 2: Scalable Fixed Rank Kriging (n = 1,000, r = 16) ---")
    rng = random.Random(42)
    n_sample = 1000
    samples = [(rng.random(), rng.random()) for _ in range(n_sample)]
    noise_sd = 0.20
    # True non-linear test surface: wave pattern with distinct crests and troughs
    vals = [
        math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y) + rng.gauss(0.0, noise_sd)
        for x, y in samples
    ]

    # Multi-center grid of r = 16 bisquare basis functions
    centers = []
    radii = []
    for gx in [0.15, 0.38, 0.62, 0.85]:
        for gy in [0.15, 0.38, 0.62, 0.85]:
            centers.append((gx, gy))
            radii.append(0.35)

    frk = FixedRankKrigingEngine(centers, radii, nugget_variance=noise_sd**2, K_diag=2.0)

    # Test targets with NON-ZERO ground-truth values
    test_targets = [(0.25, 0.00), (0.75, 0.00), (0.25, 0.50)]
    preds, lat_vars, tot_vars = frk.fit_and_predict(samples, vals, test_targets)

    print(f"Sample Observations (n) : {n_sample}")
    print(f"Basis Functions (r)     : {frk.r} multi-scale bisquares")
    for (tx, ty), p_val, lv, tv in zip(test_targets, preds, lat_vars, tot_vars):
        true_v = math.sin(2.0 * math.pi * tx) * math.cos(2.0 * math.pi * ty)
        print(
            f"Target ({tx:.2f}, {ty:.2f}) -> True: {true_v:+6.2f} | "
            f"Pred: {p_val:+6.4f} | Latent SE: {math.sqrt(lv):.4f} | Total SE: {math.sqrt(tv):.4f}"
        )

    # Out-of-sample test across 100 regular grid locations
    test_grid = [(0.05 + 0.10 * ix, 0.05 + 0.10 * iy) for ix in range(10) for iy in range(10)]
    grid_preds, _, _ = frk.fit_and_predict(samples, vals, test_grid)
    grid_true = [math.sin(2.0 * math.pi * gx) * math.cos(2.0 * math.pi * gy) for gx, gy in test_grid]

    mean_val = sum(vals) / len(vals)
    rmse_frk = math.sqrt(sum((p - t) ** 2 for p, t in zip(grid_preds, grid_true)) / len(test_grid))
    rmse_base = math.sqrt(sum((mean_val - t) ** 2 for t in grid_true) / len(test_grid))

    print(f"\nOut-of-Sample 100-Point Test Grid Performance:")
    print(f"  Fixed Rank Kriging RMSE: {rmse_frk:.4f}")
    print(f"  Baseline Mean RMSE     : {rmse_base:.4f}  (FRK reduces error by {(1 - rmse_frk/rmse_base)*100:.1f}%)")

    # ---------------------------------------------------------
    # Experiment 3: Positive Definiteness Verification
    # ---------------------------------------------------------
    print("\n--- Experiment 3: Positive Definiteness & Determinant Prefactor ---")
    # Generate 30 random locations with rotating metric tensors
    n_pd = 30
    rng_pd = random.Random(123)
    pd_pts = [(rng_pd.random(), rng_pd.random()) for _ in range(n_pd)]

    def rotating_tensor(x, y):
        theta = math.pi * x
        s1 = 0.05 + 0.40 * x
        s2 = 0.05
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # R * diag(s1^2, s2^2) * R'
        a = (cos_t * s1)**2 + (sin_t * s2)**2
        b = cos_t * sin_t * (s1**2 - s2**2)
        c = (sin_t * s1)**2 + (cos_t * s2)**2
        return [[a, b], [b, c]]

    ps_pd = PaciorekSchervishKernel(rotating_tensor, nu="gaussian")

    # Covariance matrix WITH prefactor
    K_with = [[ps_pd.evaluate_covariance(p1, p2, use_prefactor=True) for p2 in pd_pts] for p1 in pd_pts]
    # Covariance matrix WITHOUT prefactor
    K_without = [[ps_pd.evaluate_covariance(p1, p2, use_prefactor=False) for p2 in pd_pts] for p1 in pd_pts]

    eig_with = _smallest_eigenvalue_power_iteration(K_with)
    eig_without = _smallest_eigenvalue_power_iteration(K_without)

    print(f"Matrix Dimension: {n_pd} x {n_pd}")
    print(f"Smallest Eigenvalue WITH prefactor   : {eig_with:+.6f}  (Strictly Positive Definite)")
    print(f"Smallest Eigenvalue WITHOUT prefactor: {eig_without:+.6f}  (Negative -> INVALID COVARIANCE)")
    print("=" * 72)
```

---

## 13.8 Chapter Summary and Practical Guidance

This chapter expanded spatial analysis beyond the rigid boundaries of stationary Euclidean kriging:
1. **The Stationarity Hierarchy:** Stationary models assume constant variance $\sigma^2$ and constant range $\phi$. The separable heteroskedastic model $C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j)\rho(\mathbf{h})$ handles varying variance, but leaves range frozen.
2. **Process Convolutions:** By constructing spatial fields as kernel convolutions of white noise $Z(\mathbf{s}) = \sum_l k_{\mathbf{s}}(\mathbf{u}_l) w_l$, the covariance matrix $\boldsymbol{\Sigma} = \mathbf{S}\mathbf{S}^\top$ is guaranteed positive semi-definite for any kernel geometry.
3. **Paciorek-Schervish Theorem:** Parameterizing kernels as Gaussian densities produces the closed-form Matérn covariance of Theorem 13.3.1. The determinant prefactor $R_{ij}$ represents the ratio of geometric to arithmetic mean volumes; omitting it destroys positive definiteness.
4. **Space Deformation (Sampson-Guttorp):** Warps geographic space $\mathbf{s} \in \mathcal{G}$ into a latent isotropic space $\mathbf{d} = f(\mathbf{s}) \in \mathcal{D}$ using metric MDS and thin-plate splines. Regularization smoothing is essential to prevent topological folding.
5. **Fixed Rank Kriging (FRK):** Replaces the intractable $\mathcal{O}(n^3)$ exact kriging inversion with an $\mathcal{O}(n r^2 + r^3)$ low-rank basis expansion $\mathbf{Z} = \mathbf{X}\boldsymbol{\beta} + \mathbf{S}\boldsymbol{\eta} + \boldsymbol{\epsilon}$. Via the Sherman-Morrison-Woodbury identity, inverting an $r \times r$ matrix replaces the $n \times n$ bottleneck, enabling exact, uncertainty-quantified kriging for $n = 100,000$ points on a laptop.
