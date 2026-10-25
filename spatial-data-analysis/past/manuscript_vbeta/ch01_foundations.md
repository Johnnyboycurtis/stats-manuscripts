# Chapter 1: Stochastic Foundations of Spatial Linear Models

Spatial statistics begins with a fundamental shift in how we think about data. In classical mathematical statistics, observations $Y_1, \dots, Y_n$ are typically treated as an independent and identically distributed sequence where the index $i \in \{1, \dots, n\}$ simply counts data points in a ledger. In spatial statistics, the observation's physical location is an active component of the data-generating mechanism. We are no longer asking only *"How much?"*—we are asking *"How much is where?"*

This chapter establishes the stochastic foundations of spatial linear models. Rather than relying on abstract measure-theoretic machinery, we ground spatial random fields directly in linear model geometry, multivariate distributions, covariance matrices, and mean-square calculus. By the end of this chapter, you will understand how spatial autocorrelation alters variance and effective sample size, how to formulate and simulate a Gaussian random field via Cholesky factorization, how to distinguish second-order from intrinsic stationarity, and how the geometry of sampling dictates what parameters can—and cannot—be consistently estimated.

> [!NOTE]
> **Chapter at a Glance**
> - **The Spatial Dilemma (§1.1):** Proximity induces autocorrelation. We decompose spatial variance into large-scale trend, continuous spatial variation, and microscale noise, discovering why positive dependence inflates standard errors and collapses the effective sample size $n_{\text{eff}}$.
> - **The Geometry of Covariance (§1.2):** We perform the "statistician's flip," viewing random variables as geometric vectors in an inner-product space where lengths are standard deviations, dot products are covariances, and optimal linear prediction (kriging) is an orthogonal projection.
> - **Finite-Dimensional Distributions & Gaussian Realizations (§1.3):** Kolmogorov consistency ensures that specifying mean and covariance functions governs the spatial process. We derive Algorithm 1.1 to simulate Gaussian random fields via Cholesky decomposition and explain why the transformation works.
> - **Continuity & The Nugget Effect (§1.4):** We differentiate mean-square continuity from path continuity and prove that the microscale nugget creates an unavoidable mean-square discontinuity at the origin.
> - **Stationarity & Spatial Pseudo-Replication (§1.5):** Facing the "single-realization problem" (we only have one Earth), we formulate second-order and intrinsic stationarity, showing how stationarity allows spatial lags to serve as pseudo-replicates.
> - **Spatial Asymptotics & Microergodicity (§1.6):** We distinguish expanding-domain from infill asymptotics, revealing why dense infill sampling never drives the sample mean variance to zero and why only specific parameter ratios (like $\sigma^2/\phi$) are consistently estimable.

---

### 1.1 The Need for Spatial Foundations and the Three-Layer Decomposition

#### 1.1.1 The Spatial Dilemma: Proximity and Dependence
Consider measuring an environmental attribute (e.g., soil pH, sub-surface porosity, or atmospheric temperature) at $n$ geographical sites. Let $\mathbf{s} \in D \subset \mathbb{R}^d$ ($d \ge 1$) denote the spatial coordinate vector of a location:
$$
\mathbf{s} = (s_1, \dots, s_d)'
$$
If two sample sites $\mathbf{s}_i$ and $\mathbf{s}_j$ are separated by a small distance $\|\mathbf{s}_i - \mathbf{s}_j\|$, their observed values $Z(\mathbf{s}_i)$ and $Z(\mathbf{s}_j)$ are almost invariably more alike than values observed far apart. This empirical reality—formalized as Tobler's First Law of Geography—violates the classical assumption of independent sampling:
$$
\operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) \quad \text{is typically positive and decays as } \|\mathbf{s}_i - \mathbf{s}_j\| \text{ increases}
$$
(While some valid spatial covariance models drop strictly to zero beyond a finite separation distance, independent sampling throughout the domain is almost never observed in physical systems.)

Because observations share spatial covariance, classical statistical estimators break down:
1. The sample variance $s^2 = \frac{1}{n-1}\sum_{i=1}^n (Z(\mathbf{s}_i) - \bar{Z})^2$ is systematically biased downward under positive spatial correlation.
2. Standard confidence intervals $\bar{Z} \pm t_{\alpha/2, n-1} s / \sqrt{n}$ dramatically undercover the true mean, falsely conveying high precision.
3. Ordinary least squares (OLS) regression coefficients remain unbiased but are statistically inefficient, and standard errors are heavily distorted.

To resolve these failures, we must explicitly model the spatial covariance mechanism.

#### 1.1.2 The Three-Layer Structural Decomposition
Adapted from Oliver Schabenberger & Carol A. Gotway, we conceptualize any spatial observation $Z(\mathbf{s})$ not as an indivisible number, but as the superposition of three distinct sources of variation:

$$
Z(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \delta(\mathbf{s}) + \epsilon(\mathbf{s})
$$

Here and throughout this introductory section, we assume for clarity that the spatial covariance between two sites depends only on their spatial separation vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$; Section 1.5 formalizes this assumption as second-order stationarity.

##### Term-by-Term Equation Anatomy: The Three-Layer Decomposition
- $Z(\mathbf{s})$: The total scalar continuous response observed at spatial coordinate $\mathbf{s} \in D \subset \mathbb{R}^d$.
- $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$: The deterministic mean structure (large-scale spatial trend or drift). Here $\mathbf{x}(\mathbf{s}) = (x_1(\mathbf{s}), \dots, x_p(\mathbf{s}))' \in \mathbb{R}^p$ denotes the vector of known spatially referenced covariates (e.g., polynomial geographic coordinates, elevation, distance to coastline) evaluated at site $\mathbf{s}$, and $\boldsymbol{\beta} \in \mathbb{R}^p$ is the fixed, unknown vector of regression parameters.
- $\delta(\mathbf{s})$: The small-scale correlated stochastic variation (continuous spatial random field). Satisfies $\operatorname{E}[\delta(\mathbf{s})] = 0$ and $\operatorname{Cov}(\delta(\mathbf{s}), \delta(\mathbf{s}+\mathbf{h})) = C(\mathbf{h})$, with partial sill variance $\sigma^2 = \operatorname{Var}(\delta(\mathbf{s})) = C(\mathbf{0})$. This term captures Tobler's First Law of Geography: nearby locations have similar values because they share unobserved latent environmental drivers.
- $\epsilon(\mathbf{s})$: The microscale white noise (nugget effect). A zero-mean, spatially uncorrelated disturbance satisfying $\operatorname{E}[\epsilon(\mathbf{s})] = 0$, $\operatorname{Var}(\epsilon(\mathbf{s})) = \tau^2$, and $\operatorname{Cov}(\epsilon(\mathbf{s}_i), \epsilon(\mathbf{s}_j)) = 0$ for all $\mathbf{s}_i \ne \mathbf{s}_j$. It captures non-spatial instrument measurement error and micro-heterogeneity occurring below the minimum inter-site sampling distance.

```
The Spatial Decomposition Hierarchy:
+---------------------------------------------------------------------------------+
| Z(s): Total Observed Value at Location s                                        |
+---------------------------------------------------------------------------------+
        │
        ├── 1. Large-Scale Deterministic Trend:     μ(s) = x(s)' β
        │      (Smooth spatial drift, regional gradient, or elevation effect)
        │
        ├── 2. Small-Scale Correlated Stochastic Variation: δ(s)
        │      (Continuous spatial random field with Cov(δ(s), δ(s+h)) = C(h))
        │
        └── 3. Microscale Noise / Measurement Error:  ϵ(s)
               (White noise / nugget effect with Var(ϵ(s)) = τ^2, independent)
```

In matrix form across $n$ observed locations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$:
$$
\mathbf{Z} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\delta} + \boldsymbol{\epsilon}
$$
where:
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the design matrix containing known spatial coordinates or auxiliary covariates at each location: $\mathbf{X} = (\mathbf{x}(\mathbf{s}_1), \dots, \mathbf{x}(\mathbf{s}_n))'$.
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the vector of regression coefficients describing the large-scale trend.
- $\boldsymbol{\delta} \sim (\mathbf{0}, \mathbf{C})$ is the vector of spatially autocorrelated random effects with covariance matrix $\mathbf{C} = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$. (Here the notation $\mathbf{Y} \sim (\boldsymbol{\mu}, \boldsymbol{\Sigma})$ indicates a random vector with mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$, without imposing a specific distributional assumption.)
- $\boldsymbol{\epsilon} \sim (\mathbf{0}, \tau^2 \mathbf{I}_n)$ is the vector of independent microscale errors (measurement error and micro-heterogeneity).

The total variance-covariance matrix of the observed spatial vector is:
$$
\operatorname{Var}(\mathbf{Z}) = \mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n
$$
where $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$ represents the vector of covariance parameters: partial sill $\sigma^2$, spatial range $\phi$, and nugget effect $\tau^2$. The total point variance at any site is denoted $\sigma_Z^2 = \operatorname{Var}(Z(\mathbf{s})) = \sigma^2 + \tau^2$.

##### Term-by-Term Equation Anatomy: Total Spatial Covariance Matrix
- $\mathbf{V}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$: The total variance-covariance matrix of the observed spatial vector $\mathbf{Z}$. Every off-diagonal entry $V_{ij}$ contains the joint covariance between site $\mathbf{s}_i$ and site $\mathbf{s}_j$. It must be symmetric positive semidefinite for all point configurations (and strictly positive definite when observation sites are distinct and not collinear). This requirement follows directly from the definition of variance: for any non-zero coefficient vector $\mathbf{a} \in \mathbb{R}^n$, $\operatorname{Var}(\mathbf{a}'\mathbf{Z}) = \mathbf{a}'\mathbf{V}\mathbf{a} \ge 0$. As we will see in Section 1.2, variances are squared geometric lengths, and lengths can never be negative.
- $\mathbf{C}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$: The continuous spatial covariance matrix. Its entries are evaluated via the parametric kernel $C_{ij} = C(\mathbf{s}_i - \mathbf{s}_j; \sigma^2, \phi)$, with diagonal entries equal to the partial sill $\sigma^2 = \operatorname{Var}(\delta(\mathbf{s})) = C(\mathbf{0})$ and off-diagonal entries decaying smoothly toward zero as inter-site Euclidean distance $\|\mathbf{s}_i - \mathbf{s}_j\|$ increases.
- $\tau^2 \mathbf{I}_n \in \mathbb{R}^{n \times n}$: The microscale nugget matrix. Multiplies the $n \times n$ identity matrix $\mathbf{I}_n$ by the nugget variance $\tau^2 = \operatorname{Var}(\epsilon(\mathbf{s}))$. Because $\epsilon(\mathbf{s})$ is white noise, its off-diagonal contributions are identically zero, perturbing only the diagonal entries: $V_{ii} = \sigma_Z^2 = \sigma^2 + \tau^2$.
- $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$: The structural parameter vector governing spatial dependence: the partial sill $\sigma^2$ (variance of the continuous field), the spatial range parameter $\phi$, and the nugget effect $\tau^2$ (variance of the discontinuous noise component). In models such as the exponential covariance, $\phi$ represents the *e-folding distance* (the distance over which correlation drops to $e^{-1} \approx 0.368$), whereas the *practical range*—the distance at which correlation drops to 0.05—is approximately $3\phi$. When the covariance kernel depends solely on the Euclidean separation distance $\|\mathbf{h}\|$, the random field is called **isotropic**.

> [!NOTE]
> **Wait—Why Did We Do That?**
> **FAQ 1: Why is the nugget effect $\tau^2$ separated from the continuous spatial field $\delta(\mathbf{s})$?**
>
> In classical ordinary least squares, residuals are assumed to be independent errors $\epsilon_i \sim \text{i.i.d.} (0, \sigma^2)$. In spatial statistics, one might wonder: why split the stochastic residual into two distinct terms $\delta(\mathbf{s}) + \epsilon(\mathbf{s})$ rather than using a single spatial covariance function?
>
> The reason is physical and geometric. In spatial data acquisition, variability operates across two non-overlapping physical scales:
> 1. **The Continuous Latent Field ($\delta(\mathbf{s})$):** Real physical phenomena (sub-surface porosity, ambient temperature, hydraulic conductivity) are spatially continuous. As two sampling points $\mathbf{s}_i$ and $\mathbf{s}_j$ are placed arbitrarily close together ($\|\mathbf{s}_i - \mathbf{s}_j\| \to 0$), the true underlying attribute values converge: $\lim_{\|\mathbf{h}\| \to 0} \operatorname{Cov}(\delta(\mathbf{s}), \delta(\mathbf{s}+\mathbf{h})) = \operatorname{Var}(\delta(\mathbf{s})) = \sigma^2$.
> 2. **Microscale Noise and Measurement Error ($\epsilon(\mathbf{s})$):** Real sampling campaigns are subject to instrument imprecision, sample handling noise, and microscale heterogeneity (e.g., mineral veinlets, micro-fractures) operating at distances smaller than the smallest distance between sampling locations $h_{\min} = \min_{i \ne j} \|\mathbf{s}_i - \mathbf{s}_j\|$. Because the observational network cannot resolve spatial variation below $h_{\min}$, this variation appears as an apparent step-discontinuity at zero separation distance.
>
> Separating $\tau^2$ from $\mathbf{C}(\boldsymbol{\theta})$ allows spatial kriging algorithms (spatial best linear unbiased prediction; developed in Chapter 5) to distinguish between predicting the smooth, underlying physical latent surface $\mathbf{x}(\mathbf{s}_0)'\boldsymbol{\beta} + \delta(\mathbf{s}_0)$ (filtering out sensor noise) versus predicting an individual noisy future instrument readout $Z(\mathbf{s}_0)$.
>
> **The Practical Catch: Separating $\sigma^2$ and $\tau^2$ Requires Short-Lag Data**
> Because the nugget effect $\epsilon(\mathbf{s})$ occurs at distances smaller than the minimum inter-site spacing $h_{\min} = \min_{i \ne j}\|\mathbf{s}_i - \mathbf{s}_j\|$, separating $\tau^2$ from $\sigma^2$ in practice depends entirely on sampling design. Without closely spaced sample pairs or collocated duplicate measurements, a rapidly decaying continuous field $\delta(\mathbf{s})$ and pure white noise $\epsilon(\mathbf{s})$ are numerically indistinguishable.

#### 1.1.3 Classical-to-Spatial Bridge: From i.i.d. Sample Mean Variance to Spatial Variance Inflation and Effective Sample Size
To understand why classical statistical inference breaks down in the presence of spatial correlation, consider estimating a constant regional mean $\mu$ using the ordinary sample mean $\bar{Z} = \frac{1}{n}\sum_{i=1}^n Z(\mathbf{s}_i)$.

##### Classical Independent Sampling
In classical i.i.d. statistics, observations $Y_1, \dots, Y_n$ are mutually independent with common variance $\sigma_Z^2$. The variance of the sample mean is:
$$
\operatorname{Var}(\bar{Y}) = \operatorname{Var}\left(\frac{1}{n}\sum_{i=1}^n Y_i\right) = \frac{1}{n^2}\sum_{i=1}^n \operatorname{Var}(Y_i) = \frac{n \sigma_Z^2}{n^2} = \frac{\sigma_Z^2}{n}
$$
Every additional sample point brings an independent piece of statistical information, driving the variance of the estimator to zero at rate $\mathcal{O}(n^{-1})$.

##### The Spatial Reality: Variance Inflation
In spatial statistics, writing the sample mean in vector notation gives $\bar{Z} = \frac{1}{n}\mathbf{1}'\mathbf{Z}$, where $\mathbf{1} = (1, \dots, 1)' \in \mathbb{R}^n$. Using the variance operator on linear combinations of random vectors:
$$
\operatorname{Var}(\bar{Z}) = \operatorname{Var}\left(\frac{1}{n}\mathbf{1}'\mathbf{Z}\right) = \frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1} = \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n V_{ij}
$$
Decomposing the double summation into diagonal ($i = j$) and off-diagonal ($i \ne j$) pairs:
$$
\operatorname{Var}(\bar{Z}) = \frac{1}{n^2}\left( \sum_{i=1}^n V_{ii} + \sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j) \right) = \frac{\sigma_Z^2}{n} + \frac{1}{n^2}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)
$$
where $\sigma_Z^2 = \sigma^2 + \tau^2$ is the total point variance. (To isolate the effect of spatial autocorrelation cleanly, one often sets $\tau^2 = 0$ for illustrative purposes so that $\sigma_Z^2 = \sigma^2 = C(\mathbf{0})$; even without sensor noise, spatial correlation inflates variance.)

Because natural spatial processes exhibit positive autocorrelation ($C(\mathbf{s}_i - \mathbf{s}_j) > 0$), the off-diagonal summation is strictly positive. This causes **spatial variance inflation**:
$$
\operatorname{Var}(\bar{Z}) > \frac{\sigma_Z^2}{n}
$$
An analyst who calculates the classical confidence interval $\bar{Z} \pm 1.96 s / \sqrt{n}$ severely underestimates the true standard error, producing falsely narrow confidence bounds that fail to achieve their nominal 95% coverage rate.

##### Spatial Effective Sample Size
To quantify the information deficit caused by spatial redundancy, we equate the true spatial variance to the variance of a hypothetical sample of $n_{\text{eff}}$ independent observations with variance $\sigma_Z^2$:
$$
\frac{\sigma_Z^2}{n_{\text{eff}}} = \operatorname{Var}(\bar{Z}) = \frac{\mathbf{1}'\mathbf{V}\mathbf{1}}{n^2}
$$
Solving for $n_{\text{eff}}$ gives:
$$
n_{\text{eff}} = \frac{n^2 \sigma_Z^2}{\mathbf{1}'\mathbf{V}\mathbf{1}} = \frac{n \sigma_Z^2}{\frac{1}{n}\mathbf{1}'\mathbf{V}\mathbf{1}}
$$
Let $\bar{\rho}$ denote the average off-diagonal spatial correlation across all pairs of distinct observation sites:
$$
\bar{\rho} = \frac{1}{n(n-1)}\sum_{i \ne j} \rho(\mathbf{s}_i - \mathbf{s}_j) = \frac{1}{n(n-1)\sigma_Z^2}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)
$$
Then the quadratic form simplifies to:
$$
\mathbf{1}'\mathbf{V}\mathbf{1} = n\sigma_Z^2 + n(n-1)\sigma_Z^2 \bar{\rho} = n\sigma_Z^2 \left[1 + (n - 1)\bar{\rho}\right]
$$
Substituting this expression into $n_{\text{eff}}$ yields the classical **spatial effective sample size formula**:
$$
n_{\text{eff}} = \frac{n}{1 + (n - 1)\bar{\rho}}
$$
Notice that as $n \to \infty$ with average correlation bounded away from zero ($\bar{\rho} > 0$), the effective sample size saturates at a finite horizontal asymptote:
$$
\lim_{n \to \infty} n_{\text{eff}} = \frac{1}{\bar{\rho}}
$$
Consider a realistic environmental survey with $n = 100$ spatial samples exhibiting a modest average pairwise correlation of $\bar{\rho} = 0.10$. The effective sample size is:
$$
n_{\text{eff}} = \frac{100}{1 + (99 \times 0.10)} = \frac{100}{1 + 9.9} = \frac{100}{10.9} \approx 9.17
$$
The 100 spatially correlated measurements convey no more statistical information about the regional mean $\mu$ than roughly 9 independent measurements! 

##### A Striking Unification: Sample Mean vs. Sample Variance
Notice the remarkable algebraic symmetry connecting the sample mean and the sample variance under spatial dependence:
$$
\operatorname{Var}(\bar{Z}) = \frac{\sigma_Z^2}{n}\left[1 + (n - 1)\bar{\rho}\right], \qquad \operatorname{E}[S^2] = \sigma_Z^2 \left[1 - \bar{\rho}\right]
$$
While positive autocorrelation **inflates** the variance of the sample mean by the factor $[1 + (n-1)\bar{\rho}]$, it simultaneously **deflates** the expected sample variance $S^2 = \frac{1}{n-1}\sum (Z(\mathbf{s}_i) - \bar{Z})^2$ by the factor $[1 - \bar{\rho}]$! (We prove this sample variance result step by step in Exercise 1.1.) Consequently, a practitioner using the sample variance $S^2$ to estimate $\operatorname{Var}(\bar{Z})$ suffers a compounding double penalty: $S^2$ underestimates $\sigma_Z^2$, while the formula $S^2 / n$ omits the inflation multiplier $[1 + (n-1)\bar{\rho}]$.

The following Python code verifies this effective sample size saturation on a 1-D spatial transect:

```python
import math

def n_eff_exponential(coords, sigma2, phi):
    """
    Computes effective sample size for the sample mean on a coordinate grid
    under an isotropic exponential covariance kernel C(h) = sigma2 * exp(-h / phi).
    Here tau^2 = 0, so sigma_Z^2 = sigma2.
    """
    n = len(coords)
    # Total quadratic form 1'V1 = sum_{i,j} C(s_i - s_j)
    total_cov = sum(
        sigma2 * math.exp(-abs(c1 - c2) / phi)
        for c1 in coords
        for c2 in coords
    )
    return (n ** 2) * sigma2 / total_cov

# Compare 100 vs. 400 evenly spaced sites on the interval [0, 10] (phi = 1.0)
coords_100 = [10.0 * i / 99 for i in range(100)]
coords_400 = [10.0 * i / 399 for i in range(400)]

print(f"n = 100: n_eff = {n_eff_exponential(coords_100, 1.0, 1.0):.2f}")  # ~ 5.60
print(f"n = 400: n_eff = {n_eff_exponential(coords_400, 1.0, 1.0):.2f}")  # ~ 5.57
```

Notice the profound takeaway: quadrupling the sample size from 100 to 400 points along the same 10-unit transect yields virtually zero new information ($n_{\text{eff}} \approx 5.6$ in both cases). The additional 300 points simply interpolate between existing points.

##### Preview of Spatial Asymptotics
How does $\operatorname{Var}(\bar{Z})$ behave as $n \to \infty$? In classical statistics, $\operatorname{Var}(\bar{Y}) = \sigma_Z^2 / n \to 0$ unconditionally. In spatial statistics, the limit depends entirely on the sampling geometry: expanding the geographic territory (increasing-domain asymptotics) drives $\operatorname{Var}(\bar{Z}) \to 0$, whereas packing points ever more densely into a fixed region (infill asymptotics) leaves $\operatorname{Var}(\bar{Z})$ strictly positive. We develop this critical distinction formally in Section 1.6.

---

### 1.2 Spatial Random Fields and the Geometry of Covariance

#### Why We Need This: Random Variables as Geometric Vectors

##### The Conceptual Bridge: "The Statistician's Flip" (Rows vs. Columns)
In introductory linear regression $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$, we observe a finite sample of $n$ cases and construct vectors in sample space $\mathbb{R}^n$. However, environmental and physical properties—such as soil pH, aquifer permeability, or air temperature—exist continuously at *every* coordinate $\mathbf{s} \in D \subset \mathbb{R}^d$, not merely at the $n$ sites where we drilled a borehole or placed a sensor.

Predicting the value of the surface at an unmeasured location $\mathbf{s}_0$ requires a conceptual shift: rather than thinking of our dataset as $n$ numbers, we treat random variables themselves as geometric vectors.

To build operational intuition, consider the **Spreadsheet Analogy**. In multivariate data, we track variables across multiple realizations:

```
                  Variables / Coordinate Sites (Columns: Random Variables)
                  Z(s_1)        Z(s_2)        Z(s_3)   ...   Z(s_0) [Target]
Realizations    +------------------------------------------------------------+
(Rows: Maps)    |                                                            |
Realization 1   | 102.4        104.1         98.2      ...   ???            |
Realization 2   |  99.1         98.5         95.0      ...   ???            |
Realization 3   | 105.7        107.0        101.4      ...   ???            |
   :            |   :            :            :        ...    :             |
Realization M   |  ...          ...          ...       ...   ...            |
                +------------------------------------------------------------+
```

Notice the duality:
- **Looking Down a Column (Fixed $\mathbf{s}$, Random Realizations):** A single column represents the random variable $Z(\mathbf{s})$. It fluctuates across possible states of nature (realizations).
- **Looking Across a Row (Fixed Realization, Variable Coordinates):** A single row represents a spatial surface or map across the landscape—a deterministic function $\mathbf{s} \mapsto Z(\mathbf{s})$ for one specific realization of the world.

In this geometric view:
1. **Adding Vectors:** Adding random variables corresponds to building a linear combination or spatial prediction:
   $$
   \widehat{Z} = \sum_{i=1}^n w_i Z(\mathbf{s}_i)
   $$
2. **Vector Length:** The length (norm) of a zero-mean random variable $X$ is its standard deviation:
   $$
   \|X\| = \sqrt{\operatorname{Var}(X)} = \sqrt{\operatorname{E}[X^2]}
   $$
   A random variable has finite length if and only if it has finite second moments ($\operatorname{E}[X^2] < \infty$).
3. **Dot Product and Angle:** The inner product between two centered random variables $X$ and $Y$ is their covariance:
   $$
   \langle X, Y \rangle = \operatorname{Cov}(X, Y) = \operatorname{E}[(X - \mu_X)(Y - \mu_Y)]
   $$
   The cosine of the angle $\theta$ between two random variables is their Pearson correlation:
   $$
   \cos(\theta) = \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}} = \operatorname{Corr}(X, Y)
   $$
   - If two sites are perfectly correlated ($\rho = 1$), their vectors point in the identical direction ($\theta = 0^\circ$).
   - If two sites are uncorrelated ($\rho = 0$), their vectors meet at a $90^\circ$ perpendicular right angle ($\theta = 90^\circ$).
   - If two sites are negatively correlated ($\rho = -1$), they point in diametrically opposite directions ($\theta = 180^\circ$).
4. **Distance Between Vectors:** The distance between two random variables is the Root-Mean-Square Prediction Error (RMSPE):
   $$
   \|Y - X\| = \sqrt{\operatorname{E}[(Y - X)^2]}
   $$
5. **Linear Prediction (Simple vs. Ordinary Kriging):** In spatial statistics, best linear unbiased prediction is termed **kriging** (developed systematically in Chapter 5). When the mean $\mu$ is known (*simple kriging*), predicting an unobserved site $Z(\mathbf{s}_0)$ via $\widehat{Z}(\mathbf{s}_0) = \sum \lambda_i Z(\mathbf{s}_i)$ is geometrically identical to **dropping a $90^\circ$ perpendicular plumb-line (orthogonal projection)** from target vector $Z(\mathbf{s}_0)$ onto the linear subspace spanned by observed data vectors $Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n)$. The prediction error $Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)$ is orthogonal (uncorrelated) to every observed variable. When the mean is unknown (*ordinary kriging*), an unbiasedness constraint $\sum \lambda_i = 1$ is added, shifting the orthogonality by a Lagrange multiplier.

To connect this directly to classical statistics, consider predicting $Z(\mathbf{s}_0)$ from a single neighbor $Z(\mathbf{s}_1)$ with variance $\sigma_Z^2$ and covariance $C(h)$. The optimal linear prediction weight is $\lambda = \frac{C(h)}{\sigma_Z^2} = \rho(h)$. This is the exact same formula as the slope coefficient $\widehat{\beta} = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}$ in simple bivariate regression: kriging is linear regression geometry applied to random fields.

> [!NOTE]
> **From the Lecture Hall: Bridging the Prerequisite Gap**
> **$L^2$ Space as a Spreadsheet**
> In advanced probability and functional analysis, the geometric space of square-integrable random variables equipped with covariance as an inner product is formally called the **$L^2$ Hilbert space**. While that name sounds intimidating, the concept is simply linear algebra on a spreadsheet:
> - Each **column** in your data table is a random variable (a vector).
> - The **dot product** between two columns $X$ and $Y$ is defined as their covariance $\operatorname{Cov}(X, Y) = \operatorname{E}[(X - \mu_X)(Y - \mu_Y)]$.
> - The **squared length** of a column vector is its variance $\|X\|^2 = \operatorname{Var}(X)$.
> - Because lengths can never be negative ($\|X\|^2 \ge 0$), any valid variance-covariance matrix $\mathbf{V}$ must be positive semidefinite: for any linear combination $\mathbf{a}'\mathbf{Z}$, $\operatorname{Var}(\mathbf{a}'\mathbf{Z}) = \mathbf{a}'\mathbf{V}\mathbf{a} = \|\sum a_i Z(\mathbf{s}_i)\|^2 \ge 0$.
> Everything you know about Euclidean geometry—the Pythagorean theorem, orthogonal projections, angles, and distances—applies directly to spatial random variables.

#### Definition 1.2.1 (Spatial Random Field)
Let $D \subset \mathbb{R}^d$ ($d \ge 1$) denote a spatial continuous domain. A real-valued **spatial random field** is an indexed collection of random variables:
$$
\{Z(\mathbf{s}) : \mathbf{s} \in D\}
$$
where each coordinate vector $\mathbf{s} \in D$ indexes a univariate random variable $Z(\mathbf{s})$. For any single realization of the random mechanism, the mapping:
$$
\mathbf{s} \mapsto z(\mathbf{s}), \quad \mathbf{s} \in D
$$
is a deterministic spatial surface or regionalized variable across the landscape.

#### Definition 1.2.2 (Second-Order Spatial Process)
A spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is called a **second-order random field** if every site possesses a finite second moment ($\operatorname{E}[Z(\mathbf{s})^2] < \infty$ for all $\mathbf{s} \in D$). For such fields:
1. The **spatial mean function** is:
   $$
   \mu(\mathbf{s}) = \operatorname{E}[Z(\mathbf{s})]
   $$
2. The **spatial covariance kernel** is:
   $$
   C(\mathbf{s}_1, \mathbf{s}_2) = \operatorname{Cov}(Z(\mathbf{s}_1), Z(\mathbf{s}_2)) = \operatorname{E}\left[\left(Z(\mathbf{s}_1) - \mu(\mathbf{s}_1)\right)\left(Z(\mathbf{s}_2) - \mu(\mathbf{s}_2)\right)\right]
   $$
3. The **spatial variance function** is:
   $$
   \sigma_Z^2(\mathbf{s}) = C(\mathbf{s}, \mathbf{s}) = \operatorname{Var}(Z(\mathbf{s})) < \infty
   $$

#### What Can Go Wrong: Collinear Sites and Singular Covariance Matrices
What happens geometrically if two sample locations are placed at the exact same location ($\mathbf{s}_1 = \mathbf{s}_2$), or if two distinct sites have correlation $\rho = 1$?
In that case, the angle between their vector representations in $L^2$ space is $\theta = 0^\circ$. Geometrically, the two vectors lie along the identical ray. Algebraically, the corresponding rows and columns in the covariance matrix $\mathbf{V}$ become identical, rendering $\mathbf{V}$ singular ($\det(\mathbf{V}) = 0$). Inverting $\mathbf{V}$ or projecting uniquely onto the data subspace becomes impossible. This geometric singularity motivates why numerical simulation and prediction algorithms require the regularization introduced in Section 1.3.

---

### 1.3 Finite-Dimensional Distributions and Gaussian Random Fields

#### Why We Need This: Specifying Continuous Fields Through Finite Windows
How can we specify the probability distribution of an entire continuous surface $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ when we can only ever record finite vectors of numbers $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$? 

To define a valid random field, we do not need to construct an infinite-dimensional probability distribution directly. Instead, we specify the joint distribution for every possible finite subset of spatial locations. These are the **finite-dimensional distributions (f.d.d.)** of the process.

#### Definition 1.3.1 (Finite-Dimensional Distributions and Consistency)
For any finite collection of $n$ locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in D$, the random vector $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ has joint cumulative distribution function:
$$
F_{\mathbf{s}_1, \dots, \mathbf{s}_n}(z_1, \dots, z_n) = P(Z(\mathbf{s}_1) \le z_1, \dots, Z(\mathbf{s}_n) \le z_n)
$$
A family of finite-dimensional distribution functions is mathematically valid if and only if it satisfies two universal consistency requirements:
1. **Permutation Symmetry (Ordering Invariance):** Permuting the order of the spatial coordinates permutes the arguments of the joint distribution function in the exact same manner. For any permutation $\pi$ of $\{1, \dots, n\}$:
   $$
   F_{\mathbf{s}_{\pi(1)}, \dots, \mathbf{s}_{\pi(n)}}(z_{\pi(1)}, \dots, z_{\pi(n)}) = F_{\mathbf{s}_1, \dots, \mathbf{s}_n}(z_1, \dots, z_n)
   $$
2. **Marginal Consistency (Dimensional Collapse):** Integrating or taking the limit as an unobserved coordinate goes to infinity yields the joint distribution of the remaining coordinates:
   $$
   \lim_{z_{n+1} \to \infty} F_{\mathbf{s}_1, \dots, \mathbf{s}_n, \mathbf{s}_{n+1}}(z_1, \dots, z_n, z_{n+1}) = F_{\mathbf{s}_1, \dots, \mathbf{s}_n}(z_1, \dots, z_n)
   $$

In linear modeling, this marginal consistency property is an automatic and natural property of the **Multivariate Normal (Gaussian) distribution**. If an $(n+1)$-dimensional vector is jointly Gaussian:
$$
\begin{bmatrix} \mathbf{Z}_n \\ Z(\mathbf{s}_{n+1}) \end{bmatrix} \sim \mathcal{N}_{n+1}\left(\begin{bmatrix} \boldsymbol{\mu}_n \\ \mu(\mathbf{s}_{n+1}) \end{bmatrix}, \begin{bmatrix} \mathbf{V}_n & \mathbf{c}_{n+1} \\ \mathbf{c}_{n+1}' & \sigma^2(\mathbf{s}_{n+1}) \end{bmatrix}\right)
$$
then integrating out the coordinate $Z(\mathbf{s}_{n+1})$ leaves the remaining $n$ coordinates distributed *exactly* as $\mathbf{Z}_n \sim \mathcal{N}_n(\boldsymbol{\mu}_n, \mathbf{V}_n)$. The marginal distributions remain Gaussian with precisely the same sub-means and sub-covariance matrices.

> [!NOTE]
> **From the Lecture Hall: Bridging the Prerequisite Gap**
> **What Kolmogorov's Consistency Principle Is Really Saying**
> Kolmogorov's theorem is often introduced in advanced probability with heavy measure-theoretic topology, but its operational meaning for linear modelers is remarkably intuitive: **consistency under marginalization**.
> Consider a multivariate normal vector $(Z_1, Z_2, Z_3)' \sim \mathcal{N}_3(\boldsymbol{\mu}, \mathbf{V})$. If you decide to ignore $Z_3$ and examine only $(Z_1, Z_2)'$, normal distribution theory tells you that $(Z_1, Z_2)' \sim \mathcal{N}_2(\boldsymbol{\mu}_{1:2}, \mathbf{V}_{1:2, 1:2})$—you simply strike out the third row and column. The joint distribution of $(Z_1, Z_2)$ is completely unaffected by whether $Z_3$ was ever measured.
> Kolmogorov consistency guarantees that this holds across arbitrary collections of points: observing a spatial surface at 100 locations is probabilistically consistent with observing it at 1,000 locations.

#### Principle 1.3.1 (Kolmogorov Consistency Principle)
Kolmogorov's extension theorem guarantees that as long as a family of finite-dimensional distributions satisfies permutation symmetry and marginal consistency, there exists a well-defined spatial stochastic process possessing those exact finite-dimensional distributions. 

For **Gaussian random fields**, specifying a valid spatial mean function $\mu(\mathbf{s})$ and a valid positive definite covariance function $C(\mathbf{s}_1, \mathbf{s}_2)$ completely and uniquely specifies *all* finite-dimensional distributions, because any multivariate normal distribution is fully determined by its mean vector and covariance matrix. For **non-Gaussian fields**, the mean and covariance specify only the second-moment structure, leaving higher-order joint distributions undetermined.

#### Definition 1.3.2 (Gaussian Random Field)
A spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is a **Gaussian Random Field (GRF)** if for every finite integer $n \ge 1$ and every point configuration $\mathbf{s}_1, \dots, \mathbf{s}_n \in D$, the random vector $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ follows a multivariate normal distribution:
$$
\mathbf{Z} \sim \mathcal{N}_n(\boldsymbol{\mu}, \mathbf{V})
$$
where $\boldsymbol{\mu} = (\mu(\mathbf{s}_1), \dots, \mu(\mathbf{s}_n))'$ and $V_{ij} = C(\mathbf{s}_i, \mathbf{s}_j)$. A Gaussian random field is fully determined by its mean function and covariance function; no higher-order moments or distributions are needed.

#### Computing It: Generating Field Realizations via Cholesky Factorization
In practical computation, generating a realization of a Gaussian vector $\mathbf{Z} \sim \mathcal{N}_n(\boldsymbol{\mu}, \mathbf{V})$ on a discrete grid requires only multivariate normal distribution theory. Kolmogorov's theorem ensures that realizations generated on a fine grid are consistent with those on any coarser sub-grid.

```
Algorithm 1.1: Finite-Dimensional Gaussian Realization (Cholesky Method)
-------------------------------------------------------------------------
Input:
  - locations: array of spatial coordinates [s_1, s_2, ..., s_n] in D
  - mean_fn: deterministic trend function mu(s) = x(s)' * beta
  - cov_fn: positive definite covariance kernel C(s_i, s_j; theta)
  - tau2: nugget variance (default 0 for latent field, > 0 for noisy observations)
Output:
  - z_realization: simulated vector [z(s_1), ..., z(s_n)] honoring f.d.d.

Procedural Steps:
  1. Initialize empty n x n covariance matrix V
  2. For i from 1 to n:
       For j from 1 to n:
         dist = EuclideanNorm(locations[i] - locations[j])
         if i == j:
           V[i, j] = cov_fn(0) + tau2   # partial sill sigma^2 (+ tau^2 if noisy)
         else:
           V[i, j] = cov_fn(dist)
  3. Compute lower-triangular Cholesky factor L such that V = L * L^T
  4. Generate vector w of n independent standard normal variates:
       w[k] ~ Normal(0, 1) for k = 1, ..., n
  5. Compute spatially correlated zero-mean residuals:
       delta = L * w
  6. Compute deterministic spatial trend:
       mu_vec = [mean_fn(s) for s in locations]
  7. Return z_realization = mu_vec + delta
```

##### Why the Cholesky Transformation Works
In Step 3, the Cholesky decomposition computes a lower-triangular matrix $\mathbf{L}$ such that $\mathbf{V} = \mathbf{L}\mathbf{L}'$. The matrix $\mathbf{L}$ acts as the **matrix square root** of $\mathbf{V}$—the exact spatial analogue of scaling a univariate standard normal $w \sim \mathcal{N}(0, 1)$ by standard deviation $\sigma$ to obtain $\sigma w \sim \mathcal{N}(0, \sigma^2)$.

In Step 5, because $\mathbf{w} \sim \mathcal{N}_n(\mathbf{0}, \mathbf{I}_n)$, the linear combination $\boldsymbol{\delta} = \mathbf{L}\mathbf{w}$ has expectation $\operatorname{E}[\boldsymbol{\delta}] = \mathbf{L}\operatorname{E}[\mathbf{w}] = \mathbf{0}$ and covariance:
$$
\operatorname{Var}(\boldsymbol{\delta}) = \operatorname{Var}(\mathbf{L}\mathbf{w}) = \mathbf{L}\,\operatorname{Var}(\mathbf{w})\,\mathbf{L}' = \mathbf{L}\,\mathbf{I}_n\,\mathbf{L}' = \mathbf{L}\mathbf{L}' = \mathbf{V}
$$
Because linear combinations of normal random variables remain normal, $\boldsymbol{\delta} \sim \mathcal{N}_n(\mathbf{0}, \mathbf{V})$. Adding the deterministic spatial drift $\boldsymbol{\mu}$ in Step 7 yields $\mathbf{Z} \sim \mathcal{N}_n(\boldsymbol{\mu}, \mathbf{V})$, exactly honoring the required finite-dimensional joint distribution.

##### What the Output Represents
A single run of Algorithm 1.1 produces one simulated realization—a single "map" or snapshot of the spatial surface across the $n$ sites. By keeping $\mathbf{L}$ and drawing fresh standard normal vectors $\mathbf{w}^{(1)}, \mathbf{w}^{(2)}, \dots$, we obtain an ensemble of alternative realizations reflecting spatial uncertainty.

#### What Can Go Wrong: Matrix Ill-Conditioning and Nugget Regularization
- **The Cholesky Gatekeeper:** Step 3 of Algorithm 1.1 will fail with a numerical error if the matrix $\mathbf{V}$ is not strictly positive definite. If two sample points $\mathbf{s}_i$ and $\mathbf{s}_j$ are placed infinitesimally close together ($\|\mathbf{s}_i - \mathbf{s}_j\| \to 0$), the rows become linearly dependent, $\det(\mathbf{V}) \to 0$, and the condition number explodes:
  $$
  \kappa(\mathbf{V}) = \frac{\lambda_{\max}(\mathbf{V})}{\lambda_{\min}(\mathbf{V})} \to \infty
  $$
- **Nugget Regularization:** Adding a microscale nugget effect $\tau^2 > 0$ replaces $\mathbf{V}$ with $\mathbf{V} + \tau^2 \mathbf{I}_n$. This shifts every eigenvalue upward by $\tau^2$: $\lambda_i(\mathbf{V} + \tau^2 \mathbf{I}_n) = \lambda_i(\mathbf{V}) + \tau^2$. Consequently, the condition number is bounded by:
  $$
  \kappa(\mathbf{V} + \tau^2 \mathbf{I}_n) \le \frac{\lambda_{\max}(\mathbf{V}) + \tau^2}{\tau^2}
  $$
  While a very small nugget still permits a large condition number, adding $\tau^2$ guarantees that the smallest eigenvalue is strictly bounded away from zero ($\lambda_{\min} \ge \tau^2 > 0$), preventing strict numerical singularity.

---

### 1.4 Continuity of Spatial Random Fields

#### Why We Need This: Smooth Landscapes vs. Jagged Realities
If you zoom in on a contour map of an environmental variable, does the surface appear silky smooth and differentiable, or jagged, noisy, and fractured?

In spatial linear modeling, this is not merely an aesthetic question. The smoothness of a random field at the origin ($\mathbf{h} \to \mathbf{0}$) dictates how much predictive information a neighboring borehole or sensor provides about an unobserved location. If a process contains independent measurement error or sub-scale heterogeneity, two sensors placed a millimeter apart will report noticeably different numbers, breaking continuity. To formulate valid spatial covariance models in Chapter 2, we must understand the modes of spatial continuity and how microscale noise alters them.

Because the spatial domain $D \subset \mathbb{R}^d$ contains an uncountable continuum of locations, continuity can be defined in multiple distinct statistical modes.

#### Definition 1.4.1 (Modes of Spatial Continuity)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a spatial process, and let $\mathbf{s}_0 \in D$.
1. **Continuity in Probability:** For every $\varepsilon > 0$:
   $$
   \lim_{\mathbf{s} \to \mathbf{s}_0} P(|Z(\mathbf{s}) - Z(\mathbf{s}_0)| > \varepsilon) = 0
   $$
2. **Mean-Square Continuity:** For a second-order field:
   $$
   \lim_{\mathbf{s} \to \mathbf{s}_0} \operatorname{E}\left[(Z(\mathbf{s}) - Z(\mathbf{s}_0))^2\right] = 0
   $$
3. **Sample Path Continuity:** The realization $\mathbf{s} \mapsto z(\mathbf{s})$ is continuous at $\mathbf{s}_0$ for almost all realized surfaces.

*(Caution: Mean-square continuity is a property of average squared errors; it does not automatically imply that individual realized sample paths $\mathbf{s} \mapsto z(\mathbf{s})$ are continuous functions. However, for Gaussian random fields, if the covariance satisfies a mild Hölder condition $|C(\mathbf{0}) - C(\mathbf{h})| \le c \|\mathbf{h}\|^\alpha$ for $\alpha > 0$, the Kolmogorov-Chentsov theorem ensures that continuous sample paths exist.)*

#### Theorem 1.4.1 (Characterization of Mean-Square Continuity)
A second-order spatial field $Z(\mathbf{s})$ with mean $\mu(\mathbf{s})$ and covariance kernel $C(\mathbf{s}_1, \mathbf{s}_2)$ is mean-square continuous at $\mathbf{s}_0$ if and only if:
1. The mean function $\mu(\mathbf{s})$ is continuous at $\mathbf{s}_0$.
2. The covariance kernel $C(\mathbf{s}_1, \mathbf{s}_2)$ is continuous at the diagonal point $(\mathbf{s}_0, \mathbf{s}_0)$.

Furthermore, if $C$ is continuous at all diagonal points $(\mathbf{s}, \mathbf{s})$, it is continuous everywhere on $D \times D$.

*Proof.*
Expanding the mean-square distance into variance and squared bias:
$$
\operatorname{E}\left[(Z(\mathbf{s}) - Z(\mathbf{s}_0))^2\right] = \operatorname{Var}(Z(\mathbf{s}) - Z(\mathbf{s}_0)) + (\mu(\mathbf{s}) - \mu(\mathbf{s}_0))^2
$$
where
$$
\operatorname{Var}(Z(\mathbf{s}) - Z(\mathbf{s}_0)) = C(\mathbf{s}, \mathbf{s}) - 2C(\mathbf{s}, \mathbf{s}_0) + C(\mathbf{s}_0, \mathbf{s}_0)
$$
Because both the variance and squared mean difference are non-negative, the sum vanishes as $\mathbf{s} \to \mathbf{s}_0$ if and only if $\mu(\mathbf{s}) \to \mu(\mathbf{s}_0)$ and $\operatorname{Var}(Z(\mathbf{s}) - Z(\mathbf{s}_0)) \to 0$.

To see why $\operatorname{Var}(Z(\mathbf{s}) - Z(\mathbf{s}_0)) \to 0$ forces the diagonal term $C(\mathbf{s}, \mathbf{s})$ to converge to $C(\mathbf{s}_0, \mathbf{s}_0)$, apply the reverse triangle inequality for the standard deviation (the $L^2$ norm from Section 1.2):
$$
\left| \sqrt{C(\mathbf{s}, \mathbf{s})} - \sqrt{C(\mathbf{s}_0, \mathbf{s}_0)} \right| = \left| \operatorname{sd}(Z(\mathbf{s})) - \operatorname{sd}(Z(\mathbf{s}_0)) \right| \le \operatorname{sd}(Z(\mathbf{s}) - Z(\mathbf{s}_0)) = \sqrt{\operatorname{Var}(Z(\mathbf{s}) - Z(\mathbf{s}_0))}
$$
As $\operatorname{Var}(Z(\mathbf{s}) - Z(\mathbf{s}_0)) \to 0$, the right-hand side converges to zero, forcing $C(\mathbf{s}, \mathbf{s}) \to C(\mathbf{s}_0, \mathbf{s}_0)$. Substituting this diagonal limit back into the variance expansion yields:
$$
\lim_{\mathbf{s} \to \mathbf{s}_0} \left( 2C(\mathbf{s}_0, \mathbf{s}_0) - 2C(\mathbf{s}, \mathbf{s}_0) \right) = 0 \implies \lim_{\mathbf{s} \to \mathbf{s}_0} C(\mathbf{s}, \mathbf{s}_0) = C(\mathbf{s}_0, \mathbf{s}_0)
$$
Under second-order stationarity where $C(\mathbf{s}, \mathbf{s}) = C(\mathbf{s}_0, \mathbf{s}_0) = C(\mathbf{0})$ and $\mathbf{h} = \mathbf{s} - \mathbf{s}_0$, this simplifies explicitly to:
$$
\operatorname{E}\left[(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))^2\right] = 2(C(\mathbf{0}) - C(\mathbf{h}))
$$
Thus, mean-square continuity of a stationary field is equivalent to continuity of the covariance function at the origin: $\lim_{\mathbf{h} \to \mathbf{0}} C(\mathbf{h}) = C(\mathbf{0})$.

For joint continuity on $D \times D$, let centered variables be $\widetilde{Z}(\mathbf{s}) = Z(\mathbf{s}) - \mu(\mathbf{s})$. Applying the standard covariance inequality $|\operatorname{Cov}(X, Y)| \le \sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}$:
$$
\begin{aligned}
|C(\mathbf{s}_1 + \mathbf{h}_1, \mathbf{s}_2 + \mathbf{h}_2) - C(\mathbf{s}_1, \mathbf{s}_2)| &= |\operatorname{Cov}(\widetilde{Z}(\mathbf{s}_1 + \mathbf{h}_1), \widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2)) - \operatorname{Cov}(\widetilde{Z}(\mathbf{s}_1), \widetilde{Z}(\mathbf{s}_2))| \\
&\le |\operatorname{Cov}(\widetilde{Z}(\mathbf{s}_1 + \mathbf{h}_1) - \widetilde{Z}(\mathbf{s}_1), \widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2))| \\
&\quad + |\operatorname{Cov}(\widetilde{Z}(\mathbf{s}_1), \widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2) - \widetilde{Z}(\mathbf{s}_2))| \\
&\le \sqrt{\operatorname{Var}(\widetilde{Z}(\mathbf{s}_1 + \mathbf{h}_1) - \widetilde{Z}(\mathbf{s}_1))} \sqrt{\operatorname{Var}(\widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2))} \\
&\quad + \sqrt{\operatorname{Var}(\widetilde{Z}(\mathbf{s}_1))} \sqrt{\operatorname{Var}(\widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2) - \widetilde{Z}(\mathbf{s}_2))}
\end{aligned}
$$
Diagonal continuity forces $\operatorname{Var}(\widetilde{Z}(\mathbf{s}_k + \mathbf{h}_k) - \widetilde{Z}(\mathbf{s}_k)) \to 0$ as $\mathbf{h}_k \to \mathbf{0}$, driving the entire upper bound to zero. $\blacksquare$

#### What Can Go Wrong: The Nugget Discontinuity
What happens to continuity when an empirical dataset includes a microscale nugget effect $\tau^2 > 0$? The total covariance function of the observed process is:
$$
C_Z(\mathbf{h}) = \begin{cases} \sigma^2 + \tau^2, & \mathbf{h} = \mathbf{0} \\ C_\delta(\mathbf{h}), & \mathbf{h} \ne \mathbf{0} \end{cases}
$$
Taking the limit as separation lag shrinks to zero:
$$
\lim_{\mathbf{h} \to \mathbf{0}} C_Z(\mathbf{h}) = \sigma^2 \ne C_Z(\mathbf{0}) = \sigma^2 + \tau^2
$$
Because $C_Z$ exhibits a step-jump discontinuity of size $\tau^2$ at the origin $\mathbf{h} = \mathbf{0}$, any spatial process containing independent measurement error or microscale noise is **not mean-square continuous**:
$$
\lim_{\mathbf{h} \to \mathbf{0}} \operatorname{E}\left[(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))^2\right] = 2\tau^2 > 0
$$
Mean-square discontinuity is not an error or failure of the mathematical model; it is precisely what independent measurement error and sub-scale spatial variation look like when projected onto a continuous coordinate domain. The underlying physical latent field $\delta(\mathbf{s})$ is mean-square continuous, but the observed instrument readouts $Z(\mathbf{s})$ jump discontinuously from point to point.

---

### 1.5 Strict, Second-Order, and Intrinsic Stationarity

#### Why We Need This: The Single-Realization Problem and Spatial Pseudo-Replication

##### The Conceptual Bridge: Inferring Variance from a Single Map
In classical mathematical statistics, estimating the variance $\sigma^2$ or covariance of a random variable requires repeated trials: drawing $N$ independent realization vectors across repeated runs of an experiment.

In spatial statistics, we face an existential sampling dilemma: **we only ever get one realization of the Earth**. We cannot roll the geological dice again to generate twenty alternative realizations of Texas or the North Sea to estimate the covariance between two specific boreholes $\mathbf{s}_1$ and $\mathbf{s}_2$.

How can we compute statistical expectations and covariances from a single observed map?

The solution is **stationarity**. By assuming that the probabilistic behavior of the process is translation-invariant across the spatial landscape, we can treat pairs of observation sites separated by the same displacement vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$ as **spatial pseudo-replicates**. Instead of averaging across hypothetical unobserved parallel universes, we average across space within our single observed realization.

##### Trend Models vs. Residual Stationarity
Does adopting stationarity require assuming that the natural world has a flat, uniform mean everywhere? Certainly not. In spatial regression (universal kriging, geoadditive models, and spatial GLMMs), deterministic environmental covariates—such as topographic elevation, solar radiation, or distance from a coastline—induce large-scale spatial trends:
$$
\operatorname{E}[Z(\mathbf{s})] = \mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}
$$
Because the mean $\mu(\mathbf{s})$ varies from point to point, the raw observed process $Z(\mathbf{s})$ is **not** stationary. 

Instead, stationarity is assumed for the *unobserved residual stochastic process*:
$$
\delta(\mathbf{s}) + \epsilon(\mathbf{s}) = Z(\mathbf{s}) - \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}
$$
Once the deterministic large-scale trend $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ is removed, the remaining spatial perturbations $\delta(\mathbf{s})$ are assumed to satisfy translation-invariant second-order or intrinsic stationarity.

#### Definition 1.5.1 (Strict Stationarity)
A random field $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ is **strictly stationary** if its joint distributions are invariant under arbitrary spatial translations:
$$
(Z(\mathbf{s}_1 + \mathbf{h}), \dots, Z(\mathbf{s}_n + \mathbf{h}))' \stackrel{d}{=} (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'
$$
for all $n \in \mathbb{N}$, all locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in \mathbb{R}^d$, and all lag vectors $\mathbf{h} \in \mathbb{R}^d$.

#### Definition 1.5.2 (Second-Order / Weak Stationarity)
A random field $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ is **second-order stationary** (or weakly stationary) if:
1. **Constant Mean:** $\operatorname{E}[Z(\mathbf{s})] = \mu$ for all $\mathbf{s} \in \mathbb{R}^d$.
2. **Shift-Invariant Covariance:** $\operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s} + \mathbf{h})) = C(\mathbf{h})$ depends only on the separation lag $\mathbf{h} \in \mathbb{R}^d$, independent of base location $\mathbf{s}$.
3. **Finite Total Variance:** $\operatorname{Var}(Z(\mathbf{s})) = C_Z(\mathbf{0}) = \sigma_Z^2 = \sigma^2 + \tau^2 < \infty$.

If $C(\mathbf{h}) = C(\|\mathbf{h}\|)$ depends solely on the Euclidean length $h = \|\mathbf{h}\|$, the covariance function is **isotropic**. If the covariance depends on the direction or angle of separation (e.g., due to prevailing winds or geological faulting), it is **anisotropic**.

##### The Practical Payoff: Empirical Pseudo-Replicate Estimators
Under second-order stationarity, all site pairs separated by approximately lag vector $\mathbf{h}$ share the exact same theoretical covariance $C(\mathbf{h})$. Defining the spatial lag tolerance bin:
$$
N(\mathbf{h}) = \{(\mathbf{s}_i, \mathbf{s}_j) : \mathbf{s}_i - \mathbf{s}_j \approx \mathbf{h}\}
$$
we can estimate the theoretical covariance by spatial sample averaging:
$$
\widehat{C}(\mathbf{h}) = \frac{1}{|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} (Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z})
$$
where $|N(\mathbf{h})|$ is the number of paired sites in the bin.

#### Definition 1.5.3 (Intrinsic Stationarity)
A random field $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ is **intrinsically stationary** (or satisfies the intrinsic hypothesis of order 0) if:
1. **Zero Mean Increment:** $\operatorname{E}[Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})] = 0$ for all $\mathbf{s}, \mathbf{h} \in \mathbb{R}^d$.
2. **Stationary Increment Variance:**
   $$
   \operatorname{Var}(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})) = 2\gamma(\mathbf{h})
   $$
   The function $\gamma(\mathbf{h})$ is the **semivariogram** (and $2\gamma(\mathbf{h})$ is the **variogram**).

Matheron's classical method-of-moments estimator for the semivariogram is:
$$
\widehat{\gamma}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2
$$
Notice a profound practical advantage: $\widehat{\gamma}(\mathbf{h})$ operates on pairwise differences $(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2$ without requiring knowledge or subtraction of the global mean $\bar{Z}$.

#### Proposition 1.5.1 (Stationarity Inclusion Hierarchy)
1. **Second-Order $\implies$ Intrinsic:** Every second-order stationary field is intrinsically stationary, with semivariogram:
   $$
   \gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})
   $$
2. **Strict Stationarity with Finite Second Moments $\implies$ Second-Order:** Strict stationarity implies second-order stationarity *if and only if* second moments exist ($\operatorname{E}[|Z(\mathbf{s})|^2] < \infty$). A strictly stationary process with heavy tails (e.g., a Cauchy random field) has undefined variance and is therefore not second-order stationary.
3. **Gaussian Equivalence:** For Gaussian random fields, second-order stationarity and strict stationarity are mathematically identical, because Gaussian joint distributions are uniquely specified by their mean vector and covariance matrix.
4. **Intrinsic $\centernot\implies$ Second-Order:** An intrinsically stationary field may have an infinite or undefined global variance $C(\mathbf{0})$.

*Proof.*
- If $Z(\mathbf{s})$ is second-order stationary:
  $$
  \operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})) = \operatorname{Var}(Z(\mathbf{s}+\mathbf{h})) + \operatorname{Var}(Z(\mathbf{s})) - 2\operatorname{Cov}(Z(\mathbf{s}+\mathbf{h}), Z(\mathbf{s})) = 2C(\mathbf{0}) - 2C(\mathbf{h})
  $$
  Dividing by 2 yields $\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$.
- Conversely, consider standard Brownian motion $B(s)$ on $\mathbb{R}^1$ restricted to $s \ge 0$ with $B(0) = 0$. The increments satisfy $\operatorname{Var}(B(s+h) - B(s)) = |h| = 2\gamma(h)$ with $\gamma(h) = \frac{1}{2}|h|$, which depends only on lag $h$. Thus $B(s)$ is intrinsically stationary. However, the point variance is $\operatorname{Var}(B(s)) = s$, which grows unboundedly as $s \to \infty$. The global covariance $C(h)$ does not exist because there is no finite global sill $C(0)$. $\blacksquare$

```
Stationarity Inclusion Hierarchy:
+-----------------------------------------------------------------------------------------+
| Intrinsic Stationarity                                                                  |
| Increment variance 2γ(h) exists. Global variance may be infinite (e.g. Brownian motion)|
|                                                                                         |
|   +---------------------------------------------------------------------------------+   |
|   | Second-Order (Weak) Stationarity                                                |   |
|   | Constant mean μ, finite covariance C(h), sill C(0) < ∞                          |   |
|   |                                                                                 |   |
|   |   +-------------------------------------------------------------------------+   |   |
|   |   | Strict Stationarity with Finite Second Moments: E[|Z(s)|²] < ∞          |   |   |
|   |   | All finite-dimensional distributions translation-invariant              |   |   |
|   |   | (Mathematically EQUIVALENT to Second-Order for Gaussian Random Fields)  |   |   |
|   |   +-------------------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------+
* Note: A strictly stationary process with infinite second moments (e.g. Cauchy) sits 
  outside Second-Order Stationarity because its covariance is undefined.
```

> [!NOTE]
> **From the Lecture Hall: Bridging the Prerequisite Gap**
> **Why a Single Map Can Stand in for Many Experiments**
> In a classical laboratory experiment, you estimate sample variability by repeating an assay 50 times under identical conditions. But a geoscientist studying soil contamination across a watershed cannot repeat the geological history of the river basin 50 times.
>
> Stationarity is the conceptual bridge that makes spatial statistics possible. It asserts that spatial translation acts like replication. If the physical mechanism that correlates soil readings 20 meters apart in the north of the watershed is identical to the mechanism operating 20 meters apart in the south, then every pair of boreholes separated by 20 meters acts as a replicate trial of the same random experiment. Spatial pseudo-replication replaces *repetition across time* with *translation across space*.

#### What Can Go Wrong: Non-Stationary Trends and Infinite Global Variance
- **Choosing Between Second-Order and Intrinsic Models:** Second-order stationarity requires that the process has a finite, constant global variance $C(\mathbf{0}) = \sigma_Z^2 < \infty$. While mathematically convenient, this assumption fails for many natural spatial phenomena that exhibit long-range drift, diffusion, or cumulative deposition (such as topographic elevations across mountain ranges, hydraulic heads in unconfined aquifers, or Brownian surfaces). For such processes, the variance between two points grows indefinitely as their separation distance increases, making a global covariance $C(\mathbf{h})$ mathematically undefined.
- **The Robustness of Increments:** Intrinsic stationarity provides a broader and more robust modeling foundation: it requires only that local first differences $Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})$ have stationary mean and variance $2\gamma(\mathbf{h})$. This mirrors time series modeling, where non-stationary random walks ($I(1)$ series) become strictly stationary upon taking first differences. By fitting a semivariogram directly rather than a covariance function, spatial analysts avoid assuming a finite global variance when only local contrast behavior is well-behaved.

---

### 1.6 Spatial Asymptotics: Increasing-Domain vs. Infill

#### Why We Need This: The Two Regimes of Spatial Asymptotics
In time series analysis ($d = 1$), asymptotic limits follow the single arrow of time: $t \to \infty$. In spatial statistics on $\mathbb{R}^d$, the geometry of sampling introduces two non-equivalent asymptotic regimes with radically different statistical consequences.

#### Definition 1.6.1 (Asymptotic Regimes)
Let $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ be $n$ observation locations.
1. **Increasing-Domain Asymptotics:** The spatial domain expands as $n \to \infty$:
   $$
   |D_n| \to \infty \quad \text{with} \quad \inf_{i \ne j} \|\mathbf{s}_i - \mathbf{s}_j\| \ge \delta > 0
   $$
   New observations are gathered by exploring new geographical territory.
2. **Infill (Fixed-Domain) Asymptotics:** The spatial domain $D$ remains bounded ($|D| < \infty$):
   $$
   \lim_{n \to \infty} \sup_{\mathbf{s} \in D} \min_{1 \le i \le n} \|\mathbf{s} - \mathbf{s}_i\| = 0
   $$
   New observations are sampled densely between existing locations.

#### Theorem 1.6.1 (Spatial Mean-Square Ergodicity)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ be second-order stationary with constant mean $\mu$ and integrable covariance $\int_{\mathbb{R}^d} |C(\mathbf{h})| d\mathbf{h} < \infty$. The continuous spatial average $\bar{Z}_{D_n} = \frac{1}{|D_n|} \int_{D_n} Z(\mathbf{s}) d\mathbf{s}$ is mean-square consistent:
$$
\lim_{n \to \infty} \operatorname{E}\left[(\bar{Z}_{D_n} - \mu)^2\right] = 0 \quad \text{under increasing-domain asymptotics}
$$
However, under infill asymptotics on a fixed bounded domain $D$:
$$
\lim_{n \to \infty} \operatorname{Var}(\bar{Z}_n) = \frac{1}{|D|^2} \int_D \int_D C(\mathbf{s}_1 - \mathbf{s}_2) d\mathbf{s}_1 d\mathbf{s}_2 > 0
$$
The variance of the spatial mean does not vanish under infill sampling.

*Proof.*
The variance of the continuous spatial average across domain $D_n$ is:
$$
\operatorname{Var}(\bar{Z}_{D_n}) = \frac{1}{|D_n|^2} \int_{D_n} \int_{D_n} C(\mathbf{s}_1 - \mathbf{s}_2) d\mathbf{s}_1 d\mathbf{s}_2
$$
Apply the multivariable change of variables $\mathbf{h} = \mathbf{s}_1 - \mathbf{s}_2$ and $\mathbf{u} = \mathbf{s}_2$. Under this transformation, for each fixed separation lag $\mathbf{h}$, the inner integral over $\mathbf{u} = \mathbf{s}_2$ evaluates to the geometric volume of the shifted intersection $D_n \cap (D_n - \mathbf{h})$, which cannot exceed the total domain volume $|D_n|$:
$$
|D_n \cap (D_n - \mathbf{h})| \le |D_n|
$$
Therefore:
$$
\begin{aligned}
\operatorname{Var}(\bar{Z}_{D_n}) &= \frac{1}{|D_n|^2} \int_{\mathbb{R}^d} C(\mathbf{h}) \cdot |D_n \cap (D_n - \mathbf{h})| \, d\mathbf{h} \\
&\le \frac{1}{|D_n|^2} \int_{\mathbb{R}^d} |C(\mathbf{h})| \cdot |D_n| \, d\mathbf{h} \\
&= \frac{1}{|D_n|} \int_{\mathbb{R}^d} |C(\mathbf{h})| \, d\mathbf{h}
\end{aligned}
$$
Under increasing-domain asymptotics, $|D_n| \to \infty$. Because $\int_{\mathbb{R}^d} |C(\mathbf{h})| d\mathbf{h} < \infty$, the bound decays to zero as $\mathcal{O}(1/|D_n|)$, proving mean-square consistency: $\operatorname{Var}(\bar{Z}_{D_n}) \to 0$.

Under infill asymptotics, the domain $D$ remains fixed ($|D| < \infty$). As the discrete sampling grid becomes infinitely dense ($n \to \infty$), the discrete sample mean $\bar{Z}_n = \frac{1}{n} \sum_{i=1}^n Z(\mathbf{s}_i)$ converges to the continuous spatial surface integral $\bar{Z}_D = \frac{1}{|D|} \int_D Z(\mathbf{s}) d\mathbf{s}$ via standard Riemann-sum convergence. However, $\bar{Z}_D$ is a single spatial average computed over a single realization of a random field. Across the ensemble of potential realizations, its variance is:
$$
\operatorname{Var}(\bar{Z}_D) = \frac{1}{|D|^2} \int_D \int_D C(\mathbf{s}_1 - \mathbf{s}_2) d\mathbf{s}_1 d\mathbf{s}_2 > 0
$$
Because $C(\mathbf{h}) > 0$ over non-trivial volume, this integral is strictly positive. Dense sampling inside a fixed domain merely eliminates quadrature discretization error; it cannot eliminate the fundamental stochastic variability of the spatial surface itself across realizations. $\blacksquare$

#### Definition 1.6.2 (Microergodicity and Consistent Estimability)
A covariance parameter $\theta$ is **microergodic** on a bounded domain $D$ if it can be estimated consistently from a single continuous realization of $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ under fixed-domain infill asymptotics.*

If an estimator's variance remains strictly positive or bounded away from zero regardless of how densely the domain is sampled ($n \to \infty$), the parameter $\theta$ is **non-microergodic**.

---
*\*In formal measure-theoretic terms (Ibragimov & Rozanov, 1978), a parameter $\theta$ is microergodic if the probability measures $\mathbb{P}_{\theta_1}$ and $\mathbb{P}_{\theta_2}$ on the path space of $D$ corresponding to distinct values $\theta_1 \ne \theta_2$ are mutually singular ($\mathbb{P}_{\theta_1} \perp \mathbb{P}_{\theta_2}$). Non-microergodic parameters yield equivalent (mutually absolutely continuous) measures, making them fundamentally indistinguishable from a single realization.*

#### What Can Go Wrong: Infill Saturation and Microergodic Blindspots
- **The Infill Reality:** Under infill sampling, adding 10,000 extra points inside a 1-kilometer square does not give you 10,000 independent pieces of information about the global mean $\mu$ or the individual sill $\sigma^2$. 
- **The Microscope Zoom Analogy and Taylor Expansion:** Consider an Exponential covariance model $C(h) = \sigma^2 \exp(-h/\phi)$. Expanding the covariance and semivariogram as a Taylor series near the origin ($h \to 0$):
  $$
  C(h) = \sigma^2 \left(1 - \frac{h}{\phi} + \frac{h^2}{2\phi^2} - \mathcal{O}(h^3)\right)
  $$
  $$
  \gamma(h) = C(0) - C(h) = \frac{\sigma^2}{\phi} h - \frac{\sigma^2}{2\phi^2} h^2 + \mathcal{O}(h^3)
  $$
  Under infill asymptotics, the distance $h$ between adjacent sample locations shrinks toward zero. The quadratic and higher-order terms vanish rapidly, meaning the dense sample data contain information almost exclusively about the leading-order coefficient—the slope of the semivariogram at the origin:
  $$
  \lim_{h \downarrow 0} \frac{\gamma(h)}{h} = \frac{\sigma^2}{\phi}
  $$
  Dense sampling in a fixed region provides an exquisitely clear picture of the slope $\sigma^2/\phi$, but cannot untangle the vertical amplitude $\sigma^2$ (which dictates behavior as $h \to \infty$) from the horizontal range $\phi$. The individual parameters $\sigma^2$ and $\phi$ are non-microergodic and cannot be estimated consistently under infill asymptotics. Only their ratio $\sigma^2/\phi$ is microergodic and consistently estimable:
  - **Ying (1991)** established the consistency and asymptotic normality of the maximum likelihood estimator for $\sigma^2/\phi$ in dimension $d=1$.
  - **Zhang (2004)** generalized this fundamental result to dimensions $d \le 3$, proving that $\sigma^2/\phi$ is microergodic, while $\sigma^2$ and $\phi$ separately are not.
  - **Anderes (2010)** demonstrated that in dimensions $d > 4$, all covariance parameters become non-microergodic under Gaussian infill.
- **Harmless for Prediction:** Fortunately, this identification blindspot has a reassuring silver lining for spatial prediction: any two parameter pairs $(\sigma_1^2, \phi_1)$ and $(\sigma_2^2, \phi_2)$ sharing the exact same microergodic ratio $\sigma_1^2/\phi_1 = \sigma_2^2/\phi_2$ yield asymptotically identical kriging weights and identical mean-square prediction errors (we examine this result in depth in Chapter 5).

> [!NOTE]
> **From the Lecture Hall: Bridging the Prerequisite Gap**
> **Reading a Semivariogram Near Zero**
> When you plot an empirical semivariogram $\widehat{\gamma}(h)$, your eyes should immediately dart to two locations: the behavior at large $h$ (which tells you about the spatial extent and sill), and the behavior near $h = 0$ (which dictates your entire interpolation mechanics).
>
> 1. **The Origin Intercept:** Does $\gamma(h)$ head toward zero, or does it strike the vertical axis at a positive value $\tau^2 > 0$? If it hits $\tau^2 > 0$, you have a nugget effect—either sensor noise or micro-heterogeneity below your sampling resolution.
> 2. **The Local Shape at Zero:** How does $\gamma(h)$ lift off from the nugget?
>    - If it lifts off with a **linear slope** ($\gamma(h) \sim c_1 h$, like the Exponential model), your random field is continuous but nowhere differentiable. The landscape is rough, jagged, and realistic for terrain and hydrology.
>    - If it lifts off with a **parabolic curvature** ($\gamma(h) \sim c_2 h^2$, like the Gaussian model), your field is infinitely mean-square differentiable. The resulting predictions are ultra-smooth, glassy surfaces that can produce severe numerical instability (ringing and collinear covariance matrices).
>
> Infill sampling gives you an ultra-powerful microscope focused precisely at $h = 0$. It can resolve the slope and local smoothness with extreme precision, but it cannot tell you what happens past the boundary of your field.

> [!NOTE]
> **Wait—Why Did We Do That?**
> **FAQ 2: Why does increasing sample size $n$ in a fixed region fail to drive the sample mean variance to zero?**
>
> In introductory mathematical statistics, the Law of Large Numbers guarantees that as sample size $n \to \infty$, the sample mean $\bar{Y}$ converges in probability to the population mean $\mu$, with variance $\sigma^2 / n \to 0$. Why does this foundational guarantee collapse under fixed-domain spatial sampling?
>
> Consider the difference between measuring temperature with sensors in independent experiments versus sensors placed on the same hill:
> 1. **Independent Experiments:** Imagine deploying 10,000 temperature sensors, each placed in a different independent laboratory room around the world. Every sensor provides a completely independent draw from the global climate process. As $n \to \infty$, positive and negative random anomalies cancel out completely, driving $\operatorname{Var}(\bar{Y}) \to 0$.
> 2. **Spatial Sensors on the Same Hill:** Now imagine clustering all 10,000 temperature sensors inside a 1-kilometer valley on Tuesday morning. If that entire valley happens to be blanketed by an inversion layer of dense cold fog on Tuesday, every single one of those 10,000 sensors will read 3°C colder than usual together.
> 3. **The Limits of Interpolation:** Adding another 90,000 sensors inside that same valley gives you exquisite detail about the fog inside that valley on Tuesday morning, but **zero additional independent information** about the 30-year regional climate mean $\mu$. 
>
> As $n \to \infty$ within a fixed domain $D$, the discrete sample mean $\bar{Z}_n$ converges to the spatial surface average $\frac{1}{|D|}\int_D Z(\mathbf{s}) d\mathbf{s}$. Because that spatial integral is itself a random variable with strictly positive variance $\frac{1}{|D|^2} \int_D \int_D C(\mathbf{s}_1 - \mathbf{s}_2) d\mathbf{s}_1 d\mathbf{s}_2 > 0$ across realizations of the weather, the variance of $\bar{Z}_n$ can never fall below this positive lower bound. Sampling more densely in a fixed region interpolates the current realization; it does not generate new independent realizations.

---

### Key Ideas and Takeaways

Before proceeding to the analytical exercises, let us synthesize the six foundational concepts that underpin the rest of this text:

1. **The Spatial Linear Model Framework:** A continuous spatial observation $Z(\mathbf{s})$ is decomposed into large-scale deterministic drift $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$, spatially correlated residual variation $\delta(\mathbf{s})$, and independent measurement or microscale error $\epsilon(\mathbf{s})$.
2. **Positive Semidefiniteness Is Non-Negotiable:** Valid spatial covariance functions $C(\mathbf{h})$ and semivariograms $\gamma(\mathbf{h})$ must yield positive semidefinite covariance matrices $\mathbf{V}$ for any configuration of sampling sites, guaranteeing non-negative linear combination variances $\operatorname{Var}(\mathbf{a}'\mathbf{Z}) = \mathbf{a}'\mathbf{V}\mathbf{a} \ge 0$.
3. **The Meaning of the Nugget Effect:** A positive intercept $\tau^2 > 0$ at the origin of the semivariogram creates a step-jump discontinuity in $C_Z(\mathbf{h})$ at $\mathbf{h} = \mathbf{0}$. This renders the observed process mean-square discontinuous, reflecting instrument noise or spatial variation operating below the minimum inter-site separation distance.
4. **Stationarity as Spatial Pseudo-Replication:** Because we only observe a single realization of the Earth, translation invariance (stationarity) enables us to treat site pairs separated by displacement $\mathbf{h}$ as replicate trials of the same stochastic relationship. In regression settings, stationarity is assumed for the residual process $\delta(\mathbf{s})$, not the trended response $Z(\mathbf{s})$.
5. **The Dual Regimes of Spatial Asymptotics:** Expanding the domain ($|D| \to \infty$) gathers genuine new territory, enabling mean-square consistent estimation of the global mean $\mu$. Densely infilling a fixed domain ($|D| < \infty$) interpolates the local surface with vanishing quadrature error, but leaves the variance of the sample mean strictly positive and only identifies microergodic parameter ratios (such as $\sigma^2/\phi$).
6. **Spatial Prediction as Orthogonal Projection:** Kriging is best linear unbiased prediction (BLUP). In simple kriging, the predictor is the geometric orthogonal projection of the unobserved target $Z(\mathbf{s}_0)$ onto the closed linear subspace $\mathcal{M}_{\mathbf{Z}}$ in the Hilbert space $L^2(\Omega)$, satisfying the orthogonality condition $\operatorname{E}[(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)) Z(\mathbf{s}_i)] = 0$.

---

### 1.7 Analytical Exercises with Complete Solutions

#### Exercise 1.1: Downward Bias of Sample Variance under Spatial Dependence
Let $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ be observed from a second-order stationary field with mean $\mu \mathbf{1}$ and covariance matrix $\mathbf{V}$. Let $S^2 = \frac{1}{n-1}\sum_{i=1}^n (Z(\mathbf{s}_i) - \bar{Z})^2$ be the classical sample variance.
- **(a)** Express $S^2$ as a quadratic form $\mathbf{Z}'\mathbf{A}\mathbf{Z}$ and verify that $\mathbf{A}\mathbf{1} = \mathbf{0}$.
- **(b)** Prove that $\operatorname{E}[S^2] = \sigma_Z^2(1 - \bar{\rho})$, where $\sigma_Z^2 = C_Z(\mathbf{0}) = \sigma^2 + \tau^2$ and $\bar{\rho} = \frac{1}{n(n-1)\sigma_Z^2}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)$, demonstrating the downward bias under positive spatial correlation. Compare this with $\operatorname{Var}(\bar{Z})$.

*Solution.*
- **(a)** The sample mean can be written as $\bar{Z} = \frac{1}{n}\mathbf{1}'\mathbf{Z}$. The vector of centered observations is:
  $$
  \mathbf{Z} - \bar{Z}\mathbf{1} = \mathbf{Z} - \frac{1}{n}\mathbf{1}\mathbf{1}'\mathbf{Z} = \left(\mathbf{I}_n - \frac{1}{n}\mathbf{J}_n\right)\mathbf{Z}
  $$
  where $\mathbf{J}_n = \mathbf{1}\mathbf{1}'$ is the $n \times n$ matrix of all ones. Let $\mathbf{P} = \mathbf{I}_n - \frac{1}{n}\mathbf{J}_n$ denote the classical centering projection matrix. Note that $\mathbf{P}$ is symmetric ($\mathbf{P}' = \mathbf{P}$) and idempotent:
  $$
  \mathbf{P}^2 = \left(\mathbf{I}_n - \frac{1}{n}\mathbf{J}_n\right)\left(\mathbf{I}_n - \frac{1}{n}\mathbf{J}_n\right) = \mathbf{I}_n - \frac{2}{n}\mathbf{J}_n + \frac{1}{n^2}\mathbf{J}_n^2 = \mathbf{I}_n - \frac{2}{n}\mathbf{J}_n + \frac{1}{n^2}(n\mathbf{J}_n) = \mathbf{I}_n - \frac{1}{n}\mathbf{J}_n = \mathbf{P}
  $$
  The sum of squared deviations is:
  $$
  (n-1)S^2 = (\mathbf{P}\mathbf{Z})'(\mathbf{P}\mathbf{Z}) = \mathbf{Z}'\mathbf{P}'\mathbf{P}\mathbf{Z} = \mathbf{Z}'\mathbf{P}\mathbf{Z} = \mathbf{Z}'\left(\mathbf{I}_n - \frac{1}{n}\mathbf{J}_n\right)\mathbf{Z}
  $$
  Therefore, $S^2 = \mathbf{Z}'\mathbf{A}\mathbf{Z}$ with:
  $$
  \mathbf{A} = \frac{1}{n-1}\left(\mathbf{I}_n - \frac{1}{n}\mathbf{J}_n\right) = \frac{1}{n-1}\mathbf{P}
  $$
  Multiplying $\mathbf{A}$ by the vector of ones $\mathbf{1}$:
  $$
  \mathbf{A}\mathbf{1} = \frac{1}{n-1}\left(\mathbf{I}_n\mathbf{1} - \frac{1}{n}\mathbf{J}_n\mathbf{1}\right) = \frac{1}{n-1}\left(\mathbf{1} - \frac{1}{n}\mathbf{1}(\mathbf{1}'\mathbf{1})\right) = \frac{1}{n-1}\left(\mathbf{1} - \frac{n}{n}\mathbf{1}\right) = \mathbf{0}
  $$

- **(b)** For any symmetric matrix $\mathbf{A}$ and random vector $\mathbf{Z}$ with mean $\boldsymbol{\mu} = \mu \mathbf{1}$ and covariance matrix $\mathbf{V}$, the expectation of the quadratic form is:
  $$
  \operatorname{E}[\mathbf{Z}'\mathbf{A}\mathbf{Z}] = \operatorname{Tr}(\mathbf{A}\mathbf{V}) + \boldsymbol{\mu}'\mathbf{A}\boldsymbol{\mu} = \operatorname{Tr}(\mathbf{A}\mathbf{V}) + \mu^2 \mathbf{1}'\mathbf{A}\mathbf{1}
  $$
  Since $\mathbf{A}\mathbf{1} = \mathbf{0}$, the second term vanishes identically: $\mu^2 \mathbf{1}'(\mathbf{A}\mathbf{1}) = 0$. Now evaluate the trace:
  $$
  \operatorname{Tr}(\mathbf{A}\mathbf{V}) = \frac{1}{n-1}\left[\operatorname{Tr}(\mathbf{V}) - \frac{1}{n}\operatorname{Tr}(\mathbf{J}_n \mathbf{V})\right]
  $$
  Because all diagonal elements of $\mathbf{V}$ equal the total point variance $C_Z(\mathbf{0}) = \sigma_Z^2 = \sigma^2 + \tau^2$:
  $$
  \operatorname{Tr}(\mathbf{V}) = \sum_{i=1}^n V_{ii} = n\sigma_Z^2
  $$
  Using the cyclic trace property $\operatorname{Tr}(\mathbf{1}\mathbf{1}'\mathbf{V}) = \mathbf{1}'\mathbf{V}\mathbf{1}$:
  $$
  \operatorname{Tr}(\mathbf{J}_n \mathbf{V}) = \mathbf{1}'\mathbf{V}\mathbf{1} = \sum_{i=1}^n \sum_{j=1}^n V_{ij} = \sum_{i=1}^n V_{ii} + \sum_{i \ne j} V_{ij} = n\sigma_Z^2 + \sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)
  $$
  Substituting these two results into the trace expression:
  $$
  \begin{aligned}
  \operatorname{E}[S^2] &= \frac{1}{n-1}\left[n\sigma_Z^2 - \frac{1}{n}\left(n\sigma_Z^2 + \sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)\right)\right] \\
  &= \frac{1}{n-1}\left[(n - 1)\sigma_Z^2 - \frac{1}{n}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)\right] \\
  &= \sigma_Z^2 - \frac{1}{n(n-1)}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)
  \end{aligned}
  $$
  Defining the average off-diagonal correlation $\bar{\rho} = \frac{1}{n(n-1)\sigma_Z^2}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)$, we obtain:
  $$
  \operatorname{E}[S^2] = \sigma_Z^2(1 - \bar{\rho})
  $$
  When spatial correlation is positive ($\bar{\rho} > 0$), $\operatorname{E}[S^2] < \sigma_Z^2$: the classical sample variance systematically underestimates true process variability.

  Recall from Equation (1.5) that the variance of the sample mean is:
  $$
  \operatorname{Var}(\bar{Z}) = \frac{\sigma_Z^2}{n}\left[1 + (n - 1)\bar{\rho}\right]
  $$
  Notice the profound contrast: positive spatial correlation simultaneously **inflates the variance of the sample mean** by a factor of $[1 + (n-1)\bar{\rho}]$ while **deflating the expectation of the sample variance** by $(1 - \bar{\rho})$. An analyst who ignores spatial correlation will doubly deceive themselves: using $S^2 / n$ as the estimated variance of $\bar{Z}$ leads to dramatically overconfident, falsely narrow confidence intervals. $\blacksquare$

#### Exercise 1.2: Intrinsic Stationarity of the 1D Fractional Brownian Motion
Let $B_H(s)$ ($s \ge 0$) be a centered Gaussian process with $B_H(0) = 0$ and covariance:
$$
C(s_1, s_2) = \frac{1}{2}(s_1^{2H} + s_2^{2H} - |s_1 - s_2|^{2H}), \quad 0 < H < 1
$$
- **(a)** Prove that $B_H(s)$ is intrinsically stationary and determine its semivariogram $\gamma(h)$.
- **(b)** Is $B_H(s)$ second-order stationary?

*Solution.*
- **(a)** The increment between $s$ and $s+h$ ($h > 0$) has expectation $\operatorname{E}[B_H(s+h) - B_H(s)] = 0 - 0 = 0$. Its variance is:
  $$
  \begin{aligned}
  \operatorname{Var}(B_H(s+h) - B_H(s)) &= C(s+h, s+h) + C(s, s) - 2C(s+h, s) \\
  &= (s+h)^{2H} + s^{2H} - 2\left[\frac{1}{2}\left((s+h)^{2H} + s^{2H} - |(s+h) - s|^{2H}\right)\right] \\
  &= (s+h)^{2H} + s^{2H} - (s+h)^{2H} - s^{2H} + h^{2H} \\
  &= h^{2H}
  \end{aligned}
  $$
  The increment variance depends solely on separation lag $h$, completely independent of base location $s$. Therefore, $B_H(s)$ is intrinsically stationary with semivariogram:
  $$
  \gamma(h) = \frac{1}{2}\operatorname{Var}(B_H(s+h) - B_H(s)) = \frac{1}{2}h^{2H}
  $$
- **(b)** The point variance is $\operatorname{Var}(B_H(s)) = C(s, s) = s^{2H}$. Because this depends explicitly on location $s$ and diverges as $s \to \infty$, $B_H(s)$ does not possess a constant global variance. Hence $B_H(s)$ is not second-order stationary. $\blacksquare$

#### Exercise 1.3: Effective Sample Size and Variance Inflation in an Equicorrelated Network
An environmental monitoring agency deploys $n = 100$ air quality sensors across an industrial park. Because of atmospheric dispersion, the sensor readings exhibit an equicorrelated spatial structure: every distinct pair of sensors has correlation $\rho_{ij} = \bar{\rho} = 0.10$.
- **(a)** Calculate the effective sample size $n_{\text{eff}}$ of the sensor network for estimating the mean pollution level.
- **(b)** By what factor is the true variance of the sample mean $\operatorname{Var}(\bar{Z})$ inflated compared to a naive calculation that assumes independent observations?
- **(c)** If the true point variance is $\sigma_Z^2 = 25.0$, what is the expected value of the classical sample variance $S^2$?

*Solution.*
- **(a)** Using the effective sample size formula:
  $$
  n_{\text{eff}} = \frac{n}{1 + (n - 1)\bar{\rho}} = \frac{100}{1 + (100 - 1)(0.10)} = \frac{100}{1 + 99(0.10)} = \frac{100}{1 + 9.9} = \frac{100}{10.9} \approx 9.17
  $$
  Despite collecting 100 physical measurements, the network contains only as much statistical information about the regional mean as approximately $9$ independent sensors.

- **(b)** The variance inflation factor (VIF) for the sample mean is:
  $$
  \text{VIF} = \frac{\operatorname{Var}(\bar{Z})_{\text{spatial}}}{\operatorname{Var}(\bar{Z})_{\text{iid}}} = 1 + (n - 1)\bar{\rho} = 1 + 9.9 = 10.9
  $$
  The true uncertainty in estimating the regional mean is nearly **11 times larger** than classical i.i.d. formulas suggest. The standard error is inflated by $\sqrt{10.9} \approx 3.30$.

- **(c)** Using the downward bias formula from Exercise 1.1:
  $$
  \operatorname{E}[S^2] = \sigma_Z^2(1 - \bar{\rho}) = 25.0 \times (1 - 0.10) = 25.0 \times 0.90 = 22.5
  $$
  The classical sample variance underestimates the true process variance by 10% on average. $\blacksquare$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 1: Stochastic Foundations of Spatial Linear Models — Companion Exercises and Complete Solutions](ch01_exercises.md)
