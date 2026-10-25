# Chapter 1: Measure-Theoretic and Stochastic Foundations

Spatial statistics begins with a fundamental shift in how we think about data. In classical mathematical statistics, observations $Y_1, \dots, Y_n$ are typically treated as an independent and identically distributed sequence where the index $i \in \{1, \dots, n\}$ simply counts data points in a ledger. In spatial statistics, the observation's physical location is an active component of the data-generating mechanism. We are no longer asking only *"How much?"*—we are asking *"How much is where?"*

This chapter establishes the probabilistic foundations of continuous-index spatial random fields. We formulate spatial processes on abstract probability spaces, define the geometry of Euclidean coordinates, establish the Hilbert space structure of square-integrable variables, and prove the foundational existence and regularity theorems governing spatial fields in $\mathbb{R}^d$. Following the pedagogical philosophy of Maria L. Rizzo and the semantic discipline of Oliver Schabenberger & Carol A. Gotway, every formal theorem is anchored in linear model intuition, translated into clear algorithmic logic, and accompanied by geometric diagnostics.

---

### 1.1 The Need for Spatial Foundations and the Three-Layer Decomposition

#### 1.1.1 The Spatial Dilemma: Proximity and Dependence
Consider measuring an environmental attribute (e.g., soil pH, sub-surface porosity, or atmospheric temperature) at $n$ geographical sites. Let $\mathbf{s} \in D \subset \mathbb{R}^d$ ($d \ge 1$) denote the spatial coordinate vector of a location:
$$
\mathbf{s} = (s_1, \dots, s_d)'
$$
If two sample sites $\mathbf{s}_i$ and $\mathbf{s}_j$ are separated by a small distance $\|\mathbf{s}_i - \mathbf{s}_j\|$, their observed values $Z(\mathbf{s}_i)$ and $Z(\mathbf{s}_j)$ are almost invariably more alike than values observed far apart. This empirical reality—formalized as Tobler's First Law of Geography—violates the classical assumption of independent sampling:
$$
\operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) \ne 0 \quad \text{for } \mathbf{s}_i \ne \mathbf{s}_j
$$
Because observations share spatial covariance, classical statistical estimators break down:
1. The sample variance $s^2 = \frac{1}{n-1}\sum_{i=1}^n (Z(\mathbf{s}_i) - \bar{Z})^2$ is severely biased downward under positive spatial correlation.
2. Standard confidence intervals $\bar{Z} \pm t_{\alpha/2, n-1} s / \sqrt{n}$ dramatically undercover the true mean, falsely conveying high precision.
3. Ordinary least squares (OLS) regression coefficients remain unbiased but are statistically inefficient, and standard errors are heavily distorted.

To resolve these failures, we must explicitly model the spatial covariance mechanism.

#### 1.1.2 The Three-Layer Structural Decomposition
Following Schabenberger & Gotway, we conceptualize any spatial observation $Z(\mathbf{s})$ not as an indivisible number, but as the superposition of three distinct sources of variation:

$$
Z(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \delta(\mathbf{s}) + \epsilon(\mathbf{s})
$$

##### Term-by-Term Equation Anatomy: The Three-Layer Decomposition
- $Z(\mathbf{s})$: The total scalar continuous response observed at spatial coordinate $\mathbf{s} \in D \subset \mathbb{R}^d$.
- $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$: The deterministic mean structure (large-scale spatial trend or drift). Here $\mathbf{x}(\mathbf{s}) = (x_1(\mathbf{s}), \dots, x_p(\mathbf{s}))' \in \mathbb{R}^p$ denotes the vector of known spatially referenced covariates (e.g., polynomial geographic coordinates, elevation, distance to coastline) evaluated at site $\mathbf{s}$, and $\boldsymbol{\beta} \in \mathbb{R}^p$ is the fixed, unknown vector of regression parameters.
- $\delta(\mathbf{s})$: The small-scale correlated stochastic variation (continuous spatial random field). Satisfies $\operatorname{E}[\delta(\mathbf{s})] = 0$ and $\operatorname{Cov}(\delta(\mathbf{s}), \delta(\mathbf{s}+\mathbf{h})) = C(\mathbf{h})$. This term captures Tobler's First Law of Geography: nearby locations have similar values because they share unobserved latent environmental drivers.
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
- $\boldsymbol{\delta} \sim (\mathbf{0}, \mathbf{C})$ is the vector of spatially autocorrelated random effects with covariance matrix $\mathbf{C} = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$.
- $\boldsymbol{\epsilon} \sim (\mathbf{0}, \tau^2 \mathbf{I}_n)$ is the vector of independent microscale errors (measurement error and micro-heterogeneity).

The total variance-covariance matrix of the observed spatial vector is:
$$
\operatorname{Var}(\mathbf{Z}) = \mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n
$$
where $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$ represents the vector of covariance parameters: partial sill $\sigma^2$, spatial range $\phi$, and nugget effect $\tau^2$.

##### Term-by-Term Equation Anatomy: Total Spatial Covariance Matrix
- $\mathbf{V}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$: The total variance-covariance matrix of the observed spatial vector $\mathbf{Z}$. Every off-diagonal entry $V_{ij}$ contains the joint covariance between site $\mathbf{s}_i$ and site $\mathbf{s}_j$. It must be symmetric positive definite for all possible spatial point configurations.
- $\mathbf{C}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$: The continuous spatial covariance matrix. Its entries are evaluated via the parametric kernel $C_{ij} = C(\mathbf{s}_i - \mathbf{s}_j; \sigma^2, \phi)$, with diagonal entries equal to the partial sill $\sigma^2 = \operatorname{Var}(\delta(\mathbf{s}))$ and off-diagonal entries decaying smoothly toward zero as inter-site Euclidean distance $\|\mathbf{s}_i - \mathbf{s}_j\|$ increases.
- $\tau^2 \mathbf{I}_n \in \mathbb{R}^{n \times n}$: The microscale nugget matrix. Multiplies the $n \times n$ identity matrix $\mathbf{I}_n$ by the nugget variance $\tau^2 = \operatorname{Var}(\epsilon(\mathbf{s}))$. Because $\epsilon(\mathbf{s})$ is white noise, its off-diagonal contributions are identically zero, perturbing only the diagonal entries: $V_{ii} = \sigma^2 + \tau^2$.
- $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$: The structural parameter vector governing spatial dependence: the partial sill $\sigma^2$ (variance of the continuous field), the spatial range $\phi$ (distance metric determining the spatial decay rate), and the nugget effect $\tau^2$ (variance of the discontinuous noise component).

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
> Separating $\tau^2$ from $\mathbf{C}(\boldsymbol{\theta})$ allows spatial kriging algorithms to distinguish between predicting the smooth, underlying physical latent surface $\mathbf{x}(\mathbf{s}_0)'\boldsymbol{\beta} + \delta(\mathbf{s}_0)$ (filtering out sensor noise) versus predicting an individual noisy future instrument readout $Z(\mathbf{s}_0)$.

#### 1.1.3 Classical-to-Spatial Bridge: From i.i.d. Sample Mean Variance to Spatial Variance Inflation and Effective Sample Size
To understand why classical statistical inference breaks down in the presence of spatial correlation, consider estimating a constant regional mean $\mu$ using the ordinary sample mean $\bar{Z} = \frac{1}{n}\sum_{i=1}^n Z(\mathbf{s}_i)$.

##### Classical Independent Sampling
In classical i.i.d. statistics, observations $Y_1, \dots, Y_n$ are mutually independent with common variance $\sigma^2$. The variance of the sample mean is:
$$
\operatorname{Var}(\bar{Y}) = \operatorname{Var}\left(\frac{1}{n}\sum_{i=1}^n Y_i\right) = \frac{1}{n^2}\sum_{i=1}^n \operatorname{Var}(Y_i) = \frac{n \sigma^2}{n^2} = \frac{\sigma^2}{n}
$$
Every additional sample point brings an independent piece of statistical information, driving the variance of the estimator to zero at rate $\mathcal{O}(n^{-1})$.

##### The Spatial Reality: Variance Inflation
In spatial statistics, writing the sample mean in vector notation gives $\bar{Z} = \frac{1}{n}\mathbf{1}'\mathbf{Z}$, where $\mathbf{1} = (1, \dots, 1)' \in \mathbb{R}^n$. Using the variance operator on linear combinations of random vectors:
$$
\operatorname{Var}(\bar{Z}) = \operatorname{Var}\left(\frac{1}{n}\mathbf{1}'\mathbf{Z}\right) = \frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1} = \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n C(\mathbf{s}_i - \mathbf{s}_j)
$$
Decomposing the double summation into diagonal ($i = j$) and off-diagonal ($i \ne j$) pairs:
$$
\operatorname{Var}(\bar{Z}) = \frac{1}{n^2}\left( \sum_{i=1}^n C(\mathbf{0}) + \sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j) \right) = \frac{\sigma^2}{n} + \frac{1}{n^2}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)
$$
Because natural spatial processes exhibit positive autocorrelation ($C(\mathbf{s}_i - \mathbf{s}_j) > 0$), the off-diagonal summation is strictly positive. This causes **spatial variance inflation**:
$$
\operatorname{Var}(\bar{Z}) > \frac{\sigma^2}{n}
$$
An analyst who calculates the classical confidence interval $\bar{Z} \pm 1.96 \sigma / \sqrt{n}$ severely underestimates the true standard error, producing falsely narrow confidence bounds that fail to achieve their nominal 95% coverage rate.

##### Spatial Effective Sample Size
To quantify the information deficit caused by spatial redundancy, we equate the true spatial variance to the variance of a hypothetical sample of $n_{\text{eff}}$ independent observations:
$$
\frac{\sigma^2}{n_{\text{eff}}} = \operatorname{Var}(\bar{Z}) = \frac{\mathbf{1}'\mathbf{V}\mathbf{1}}{n^2}
$$
Solving for $n_{\text{eff}}$ gives:
$$
n_{\text{eff}} = \frac{n^2 \sigma^2}{\mathbf{1}'\mathbf{V}\mathbf{1}} = \frac{n \sigma^2}{\frac{1}{n}\mathbf{1}'\mathbf{V}\mathbf{1}}
$$
Let $\bar{\rho}$ denote the average off-diagonal spatial correlation across all pairs of distinct observation sites:
$$
\bar{\rho} = \frac{1}{n(n-1)}\sum_{i \ne j} \rho(\mathbf{s}_i - \mathbf{s}_j) = \frac{1}{n(n-1)\sigma^2}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)
$$
Then the quadratic form simplifies to:
$$
\mathbf{1}'\mathbf{V}\mathbf{1} = n\sigma^2 + n(n-1)\sigma^2 \bar{\rho} = n\sigma^2 \left[1 + (n - 1)\bar{\rho}\right]
$$
Substituting this expression into $n_{\text{eff}}$ yields the classical **spatial effective sample size formula**:
$$
n_{\text{eff}} = \frac{n}{1 + (n - 1)\bar{\rho}}
$$
If a dataset contains $n = 100$ spatial samples with an average pairwise correlation of $\bar{\rho} = 0.5$, the effective sample size is:
$$
n_{\text{eff}} = \frac{100}{1 + (99 \times 0.5)} = \frac{100}{50.5} \approx 1.98 \approx 2
$$
The 100 spatially clustered observations convey no more statistical information about the regional mean $\mu$ than two independent measurements!

##### Asymptotic Divergence: Increasing-Domain vs. Infill
The behavior of $\operatorname{Var}(\bar{Z})$ as $n \to \infty$ depends fundamentally on the spatial sampling geometry:
1. **Increasing-Domain Asymptotics:** As new points are gathered across an expanding geographical landscape ($|D_n| \to \infty$ with $\inf_{i \ne j}\|\mathbf{s}_i - \mathbf{s}_j\| \ge \delta > 0$), inter-site distances grow and spatial correlation decays:
   $$
   \operatorname{Var}(\bar{Z}) = \mathcal{O}(n^{-1}) \longrightarrow 0 \quad \text{as } n \to \infty
   $$
   The sample mean remains consistent, albeit converging at a slower constant factor than the i.i.d. rate.
2. **Fixed-Domain Infill Asymptotics:** In contrast, if the domain $D$ is bounded ($|D| < \infty$) and sample points are packed more and more densely into the fixed space:
   $$
   \lim_{n \to \infty} \operatorname{Var}(\bar{Z}) = \frac{1}{|D|^2} \int_D \int_D C(\mathbf{u}, \mathbf{v}) d\mathbf{u} d\mathbf{v} > 0
   $$
   Under fixed-domain infill sampling, the variance of the sample mean **never converges to zero**, no matter how many millions of points are measured. Sampling more densely in a fixed region provides redundant spatial detail, but cannot overcome the fundamental uncertainty of the single realized spatial surface.

---

### 1.2 Probability Spaces and the Hilbert Space $L^2(\Omega, \mathcal{F}, P)$

#### Definition 1.2.1 (Continuous Spatial Random Field)
Let $(\Omega, \mathcal{F}, P)$ be a complete probability space:
- $\Omega$ is the set of all elementary outcomes $\omega$.
- $\mathcal{F}$ is a $\sigma$-algebra of events.
- $P: \mathcal{F} \to [0, 1]$ is a countably additive probability measure with $P(\Omega) = 1$.

Let $D \subset \mathbb{R}^d$ ($d \ge 1$) be a spatial domain equipped with the Borel $\sigma$-algebra $\mathcal{B}(D)$ induced by the Euclidean norm $\|\mathbf{s}_1 - \mathbf{s}_2\| = \sqrt{\sum_{k=1}^d (s_{1k} - s_{2k})^2}$. 

A real-valued **spatial random field** is a family of random variables:
$$
\{Z(\mathbf{s}) : \mathbf{s} \in D\}
$$
such that for each fixed spatial coordinate $\mathbf{s} \in D$, the mapping $\omega \mapsto Z(\mathbf{s}, \omega)$ is an $(\mathcal{F}, \mathcal{B}(\mathbb{R}))$-measurable function from $\Omega$ into $\mathbb{R}$.

For any fixed outcome $\omega \in \Omega$, the deterministic function:
$$
\mathbf{s} \mapsto Z(\mathbf{s}, \omega), \quad \mathbf{s} \in D
$$
is called a **realization**, **sample path**, or **spatial surface**.

#### Definition 1.2.2 (The Hilbert Space $L^2(\Omega, \mathcal{F}, P)$)
Let $L^2(\Omega, \mathcal{F}, P)$ denote the vector space of equivalence classes of real random variables $X$ on $(\Omega, \mathcal{F}, P)$ satisfying:
$$
\operatorname{E}[|X|^2] = \int_\Omega |X(\omega)|^2 dP(\omega) < \infty
$$
where variables equal $P$-almost surely are identified. The space is equipped with the inner product:
$$
\langle X, Y \rangle = \operatorname{E}[XY] = \int_\Omega X(\omega)Y(\omega) dP(\omega)
$$
which induces the norm $\|X\|_{L^2} = \sqrt{\langle X, X \rangle} = \sqrt{\operatorname{E}[X^2]}$.

#### Proposition 1.2.1 (Completeness of the State Space)
The inner product space $(L^2(\Omega, \mathcal{F}, P), \langle \cdot, \cdot \rangle)$ is complete, and therefore a real Hilbert space.

*Proof.*
Let $\{X_n\}_{n=1}^\infty$ be a Cauchy sequence in $L^2$, meaning $\lim_{n,m \to \infty} \|X_n - X_m\|_{L^2} = 0$. By the Riesz-Fischer theorem for $L^p$ spaces with $p = 2$, there exists $X \in L^2(\Omega, \mathcal{F}, P)$ such that $\lim_{n \to \infty} \|X_n - X\|_{L^2} = 0$, and there exists a subsequence $\{X_{n_k}\}$ converging $P$-almost surely to $X$. $\blacksquare$

#### Definition 1.2.3 (Second-Order Spatial Process)
A spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is called a **second-order random field** if $Z(\mathbf{s}) \in L^2(\Omega, \mathcal{F}, P)$ for all $\mathbf{s} \in D$. For such fields:
1. The **spatial mean function** is $\mu(\mathbf{s}) = \operatorname{E}[Z(\mathbf{s})]$.
2. The **spatial covariance kernel** is:
   $$
   C(\mathbf{s}_1, \mathbf{s}_2) = \operatorname{Cov}(Z(\mathbf{s}_1), Z(\mathbf{s}_2)) = \langle Z(\mathbf{s}_1) - \mu(\mathbf{s}_1), Z(\mathbf{s}_2) - \mu(\mathbf{s}_2) \rangle
   $$
3. The **spatial variance function** is $\sigma^2(\mathbf{s}) = C(\mathbf{s}, \mathbf{s}) = \|Z(\mathbf{s}) - \mu(\mathbf{s})\|_{L^2}^2$.

---

### 1.3 Finite-Dimensional Distributions and Kolmogorov's Existence Theorem

#### Beat 1: The Mathematical Anchor (The "Why")
How can we be certain that an uncountable, continuous physical surface $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ exists in a mathematically rigorous sense when we can only ever observe finite subsets of points? Kolmogorov's Existence Theorem provides the answer: if we specify consistent joint distributions for every finite configuration of points, there exists a well-defined probability space hosting the continuous field.

#### Definition 1.3.1 (Finite-Dimensional Distributions)
For any finite collection of $n$ distinct locations $\mathbf{s} = (\mathbf{s}_1, \dots, \mathbf{s}_n)' \in D^n$, the random vector is $\mathbf{Z}_{\mathbf{s}} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$. Its **finite-dimensional distribution (f.d.d.)** is the probability measure $P_{\mathbf{s}}$ on $(\mathbb{R}^n, \mathcal{B}(\mathbb{R}^n))$:
$$
P_{\mathbf{s}}(B) = P(\{\omega \in \Omega : \mathbf{Z}_{\mathbf{s}}(\omega) \in B\}), \quad B \in \mathcal{B}(\mathbb{R}^n)
$$

A family of measures $\{P_{\mathbf{s}}\}$ is **projective** if:
1. **Permutation Symmetry:** For any permutation $\pi$, $P_{(\mathbf{s}_{\pi(1)}, \dots, \mathbf{s}_{\pi(n)})}(B_{\pi(1)} \times \dots \times B_{\pi(n)}) = P_{(\mathbf{s}_1, \dots, \mathbf{s}_n)}(B_1 \times \dots \times B_n)$.
2. **Marginal Consistency:** $P_{(\mathbf{s}_1, \dots, \mathbf{s}_n, \mathbf{s}_{n+1})}(B \times \mathbb{R}) = P_{(\mathbf{s}_1, \dots, \mathbf{s}_n)}(B)$.

#### Theorem 1.3.1 (Kolmogorov's Existence Theorem)
Let $\{P_{\mathbf{s}}\}$ be a projective family of probability measures across all finite subsets of $D$. Then there exists a probability space $(\Omega, \mathcal{F}, P)$ and a spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ defined on it whose finite-dimensional distributions coincide exactly with $\{P_{\mathbf{s}}\}$.

*Proof.*
1. **Canonical Sample Space:** Set $\Omega = \mathbb{R}^D = \{\omega : \omega \text{ is a function from } D \text{ into } \mathbb{R}\}$. The evaluation coordinates are $Z(\mathbf{s}, \omega) = \omega(\mathbf{s})$.
2. **Cylinder Sets:** For any $\mathbf{s} = (\mathbf{s}_1, \dots, \mathbf{s}_n)'$ and $B \in \mathcal{B}(\mathbb{R}^n)$, define cylinder sets $C(\mathbf{s}, B) = \{\omega \in \mathbb{R}^D : (\omega(\mathbf{s}_1), \dots, \omega(\mathbf{s}_n))' \in B\}$. The collection of all cylinder sets forms a Boolean algebra $\mathcal{C}$.
3. **Set Function:** Define $P_0(C(\mathbf{s}, B)) = P_{\mathbf{s}}(B)$. By marginal consistency and permutation symmetry, $P_0$ is well-defined and finitely additive on $\mathcal{C}$.
4. **Countable Additivity via Inner Regularity:** Let $\{A_k\}_{k=1}^\infty \subset \mathcal{C}$ be a decreasing sequence with $\lim P_0(A_k) = \delta > 0$. Because Euclidean Borel measures are Radon measures (inner regular), there exist compact bases $K_k \subset B_k$ approximating each cylinder set. Cantor's diagonal argument extracts a pointwise convergent subsequence, demonstrating that $\bigcap_{k=1}^\infty A_k \ne \emptyset$. Thus $P_0$ is countably additive on $\mathcal{C}$.
5. **Carathéodory Extension:** By Carathéodory's Theorem, $P_0$ extends uniquely to a probability measure $P$ on the product $\sigma$-algebra $\mathcal{F} = \sigma(\mathcal{C})$. $\blacksquare$

#### Beat 2: Algorithmic Translation (The "How")
In practical computation, Kolmogorov's theorem justifies generating realizations of Gaussian random fields on any discrete grid by factoring the finite-dimensional covariance matrix.

```
Algorithm 1.1: Finite-Dimensional Gaussian Realization (Cholesky Method)
-------------------------------------------------------------------------
Input:
  - locations: array of spatial coordinates [s_1, s_2, ..., s_n] in D
  - mean_fn: deterministic trend function mu(s) = x(s)' * beta
  - cov_fn: positive definite covariance kernel C(s_i, s_j; theta)
Output:
  - z_realization: simulated vector [z(s_1), ..., z(s_n)] honoring f.d.d.

Procedural Steps:
  1. Initialize empty n x n covariance matrix V
  2. For i from 1 to n:
       For j from 1 to n:
         dist = EuclideanNorm(locations[i] - locations[j])
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

#### Beat 3: Intuition and Diagnostic Checks (The "Reality")
- **The Cholesky Gatekeeper:** Step 3 of Algorithm 1.1 will fail with a numerical error if the matrix $\mathbf{V}$ is not strictly positive definite. If two sample points $\mathbf{s}_i$ and $\mathbf{s}_j$ are placed infinitesimally close together ($\|\mathbf{s}_i - \mathbf{s}_j\| \to 0$), the rows become linearly dependent, $\det(\mathbf{V}) \to 0$, and the condition number explodes:
  $$
  \kappa(\mathbf{V}) = \frac{\lambda_{\max}(\mathbf{V})}{\lambda_{\min}(\mathbf{V})} \to \infty
  $$
- **Nugget Stabilization:** Adding a microscale nugget effect $\tau^2 > 0$ replaces $\mathbf{V}$ with $\mathbf{V} + \tau^2 \mathbf{I}_n$, which shifts all eigenvalues upward by $\tau^2$ and guarantees numerical stability.

---

### 1.4 Continuity of Spatial Random Fields

Because the spatial domain $D \subset \mathbb{R}^d$ contains an uncountable continuum of locations, continuity can be defined in multiple distinct topological modes.

#### Definition 1.4.1 (Modes of Spatial Continuity)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a spatial process, and let $\mathbf{s}_0 \in D$.
1. **Continuity in Probability:** For every $\varepsilon > 0$:
   $$
   \lim_{\mathbf{s} \to \mathbf{s}_0} P(|Z(\mathbf{s}) - Z(\mathbf{s}_0)| > \varepsilon) = 0
   $$
2. **Mean-Square ($L^2$) Continuity:** For a second-order field:
   $$
   \lim_{\mathbf{s} \to \mathbf{s}_0} \|Z(\mathbf{s}) - Z(\mathbf{s}_0)\|_{L^2}^2 = \lim_{\mathbf{s} \to \mathbf{s}_0} \operatorname{E}\left[(Z(\mathbf{s}) - Z(\mathbf{s}_0))^2\right] = 0
   $$
3. **Sample Path Continuity:** The realization $\mathbf{s} \mapsto Z(\mathbf{s}, \omega)$ is continuous at $\mathbf{s}_0$ for $P$-almost every $\omega \in \Omega$.

#### Theorem 1.4.1 (Characterization of Mean-Square Continuity)
A second-order spatial field $Z(\mathbf{s})$ with mean $\mu(\mathbf{s})$ and covariance kernel $C(\mathbf{s}_1, \mathbf{s}_2)$ is mean-square continuous at $\mathbf{s}_0$ if and only if:
1. The mean function $\mu(\mathbf{s})$ is continuous at $\mathbf{s}_0$.
2. The covariance kernel $C(\mathbf{s}_1, \mathbf{s}_2)$ is continuous at the diagonal point $(\mathbf{s}_0, \mathbf{s}_0)$.

Furthermore, if $C$ is continuous at all diagonal points $(\mathbf{s}, \mathbf{s})$, it is continuous everywhere on $D \times D$.

*Proof.*
Expanding the mean-square distance:
$$
\operatorname{E}\left[(Z(\mathbf{s}) - Z(\mathbf{s}_0))^2\right] = C(\mathbf{s}, \mathbf{s}) - 2C(\mathbf{s}, \mathbf{s}_0) + C(\mathbf{s}_0, \mathbf{s}_0) + (\mu(\mathbf{s}) - \mu(\mathbf{s}_0))^2
$$
Because both the variance and squared mean difference are non-negative, the sum vanishes as $\mathbf{s} \to \mathbf{s}_0$ if and only if $\mu(\mathbf{s}) \to \mu(\mathbf{s}_0)$, $C(\mathbf{s}, \mathbf{s}) \to C(\mathbf{s}_0, \mathbf{s}_0)$, and $C(\mathbf{s}, \mathbf{s}_0) \to C(\mathbf{s}_0, \mathbf{s}_0)$. Under second-order stationarity where $C(\mathbf{s}, \mathbf{s}) = C(\mathbf{s}_0, \mathbf{s}_0) = C(\mathbf{0})$, this simplifies to continuity of the covariance function at the origin: $\lim_{\mathbf{h} \to \mathbf{0}} C(\mathbf{h}) = C(\mathbf{0})$.

For joint continuity, let centered variables be $\widetilde{Z}(\mathbf{s}) = Z(\mathbf{s}) - \mu(\mathbf{s})$. Applying the Cauchy-Schwarz inequality in $L^2$:
$$
\begin{aligned}
|C(\mathbf{s}_1 + \mathbf{h}_1, \mathbf{s}_2 + \mathbf{h}_2) - C(\mathbf{s}_1, \mathbf{s}_2)| &= |\langle \widetilde{Z}(\mathbf{s}_1 + \mathbf{h}_1), \widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2) \rangle - \langle \widetilde{Z}(\mathbf{s}_1), \widetilde{Z}(\mathbf{s}_2) \rangle| \\
&\le \|\widetilde{Z}(\mathbf{s}_1 + \mathbf{h}_1) - \widetilde{Z}(\mathbf{s}_1)\|_{L^2} \|\widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2)\|_{L^2} \\
&\quad + \|\widetilde{Z}(\mathbf{s}_1)\|_{L^2} \|\widetilde{Z}(\mathbf{s}_2 + \mathbf{h}_2) - \widetilde{Z}(\mathbf{s}_2)\|_{L^2}
\end{aligned}
$$
Diagonal continuity forces $\|\widetilde{Z}(\mathbf{s}_k + \mathbf{h}_k) - \widetilde{Z}(\mathbf{s}_k)\|_{L^2} \to 0$ as $\mathbf{h}_k \to \mathbf{0}$, driving the upper bound to zero. $\blacksquare$

---

### 1.5 Strict, Second-Order, and Intrinsic Stationarity

#### Definition 1.5.1 (Strict Stationarity)
A random field $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ is **strictly stationary** if its joint distributions are invariant under arbitrary spatial translations:
$$
(Z(\mathbf{s}_1 + \mathbf{h}), \dots, Z(\mathbf{s}_n + \mathbf{h}))' \stackrel{d}{=} (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'
$$
for all $n \in \mathbb{N}$, all locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in \mathbb{R}^d$, and all lag vectors $\mathbf{h} \in \mathbb{R}^d$.

#### Definition 1.5.2 (Second-Order / Weak Stationarity)
A random field $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ is **second-order stationary** if:
1. Constant Mean: $\operatorname{E}[Z(\mathbf{s})] = \mu$ for all $\mathbf{s} \in \mathbb{R}^d$.
2. Shift-Invariant Covariance: $\operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s} + \mathbf{h})) = C(\mathbf{h})$ depends only on the separation vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$.
3. Finite Global Variance: $\operatorname{Var}(Z(\mathbf{s})) = C(\mathbf{0}) = \sigma^2 + \tau^2 < \infty$.

#### Definition 1.5.3 (Intrinsic Stationarity)
A random field $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ is **intrinsically stationary** (or satisfies the intrinsic hypothesis of order 0) if:
1. Zero Mean Increment: $\operatorname{E}[Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})] = 0$ for all $\mathbf{s}, \mathbf{h} \in \mathbb{R}^d$.
2. Stationary Increment Variance:
   $$
   \operatorname{Var}(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})) = 2\gamma(\mathbf{h})
   $$
   The function $\gamma(\mathbf{h})$ is the **semivariogram** (and $2\gamma(\mathbf{h})$ is the **variogram**).

#### Proposition 1.5.1 (Stationarity Inclusion Hierarchy)
1. Every second-order stationary field is intrinsically stationary, with semivariogram:
   $$
   \gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})
   $$
2. The converse is false: an intrinsically stationary field may have an infinite or undefined global variance $C(\mathbf{0})$.

*Proof.*
- If $Z(\mathbf{s})$ is second-order stationary:
  $$
  \operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})) = \operatorname{Var}(Z(\mathbf{s}+\mathbf{h})) + \operatorname{Var}(Z(\mathbf{s})) - 2\operatorname{Cov}(Z(\mathbf{s}+\mathbf{h}), Z(\mathbf{s})) = 2C(\mathbf{0}) - 2C(\mathbf{h})
  $$
  Dividing by 2 yields $\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$.
- Conversely, consider standard Brownian motion $B(s)$ on $\mathbb{R}^1$ ($s \ge 0$). The increments satisfy $\operatorname{Var}(B(s+h) - B(s)) = h = 2\gamma(h)$ with $\gamma(h) = \frac{1}{2}h$. It is intrinsically stationary. However, the point variance is $\operatorname{Var}(B(s)) = s$, which diverges as $s \to \infty$. Thus global covariance $C(h)$ does not exist. $\blacksquare$

```
Hierarchy of Stationarity Classes:
+---------------------------------------------------------------------------------+
| Intrinsic Stationarity                                                          |
| Increment variance 2γ(h) exists. Global variance may be infinite (e.g. Brownian)|
|                                                                                 |
|   +-------------------------------------------------------------------------+   |
|   | Second-Order (Weak) Stationarity                                        |   |
|   | Constant mean μ, finite covariance C(h), sill C(0) < ∞                  |   |
|   |                                                                         |   |
|   |   +-----------------------------------------------------------------+   |   |
|   |   | Strict Stationarity                                             |   |   |
|   |   | All finite-dimensional distributions invariant under shift      |   |   |
|   |   | (Identical to Second-Order for Gaussian Random Fields)          |   |   |
|   |   +-----------------------------------------------------------------+   |   |
|   +-------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------+
```

---

### 1.6 Spatial Asymptotics: Increasing-Domain vs. Infill

#### Beat 1: The Mathematical Anchor (The "Why")
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
Let $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ be second-order stationary with constant mean $\mu$ and integrable covariance $\int_{\mathbb{R}^d} |C(\mathbf{h})| d\mathbf{h} < \infty$. The spatial average $\bar{Z}_n = \frac{1}{|D_n|} \int_{D_n} Z(\mathbf{s}) d\mathbf{s}$ is mean-square consistent:
$$
\lim_{n \to \infty} \operatorname{E}\left[(\bar{Z}_n - \mu)^2\right] = 0 \quad \text{under increasing-domain asymptotics}
$$
However, under infill asymptotics on a fixed domain $D$:
$$
\lim_{n \to \infty} \operatorname{Var}(\bar{Z}_n) = \frac{1}{|D|^2} \int_D \int_D C(\mathbf{s}_1 - \mathbf{s}_2) d\mathbf{s}_1 d\mathbf{s}_2 > 0
$$
The variance of the spatial mean does not vanish under infill sampling.

*Proof.*
The variance is $\operatorname{Var}(\bar{Z}_n) = \frac{1}{|D_n|^2} \int_{D_n} \int_{D_n} C(\mathbf{s}_1 - \mathbf{s}_2) d\mathbf{s}_1 d\mathbf{s}_2$.
Using the set covariogram $K_{D_n}(\mathbf{h}) = |D_n \cap (D_n - \mathbf{h})| \le |D_n|$:
$$
\operatorname{Var}(\bar{Z}_n) \le \frac{1}{|D_n|} \int_{\mathbb{R}^d} |C(\mathbf{h})| d\mathbf{h}
$$
Under increasing domain, $|D_n| \to \infty$, so $\operatorname{Var}(\bar{Z}_n) \to 0$. Under infill asymptotics, $|D_n| = |D|$ is constant, and the double integral over $D \times D$ remains strictly positive. $\blacksquare$

#### Definition 1.6.2 (Microergodicity)
A covariance parameter $\theta$ is **microergodic** on a bounded domain $D$ if two Gaussian measures $P_{\theta_1}$ and $P_{\theta_2}$ with $\theta_1 \ne \theta_2$ are mutually singular (orthogonal):
$$
P_{\theta_1} \perp P_{\theta_2}
$$
If $P_{\theta_1} \equiv P_{\theta_2}$ (equivalent measures), the parameter $\theta$ is non-microergodic.

#### Beat 3: Intuition and Diagnostic Consequences
- **The Infill Reality:** Under infill sampling, adding 10,000 extra points inside a 1-kilometer square does not give you 10,000 independent pieces of information about the global mean $\mu$ or the individual sill $\sigma^2$. 
- For an Exponential covariance model $C(h) = \sigma^2 \exp(-h/\phi)$ in $\mathbb{R}^3$, the individual parameters $\sigma^2$ and $\phi$ are non-microergodic and cannot be estimated consistently under infill sampling. Only the ratio $\sigma^2/\phi$ (or more generally $\sigma^2/\phi^{2\nu}$ for the Matérn family with fixed smoothness $\nu$) is microergodic and consistently estimable (Zhang, 2004).
- However, for spatial prediction (kriging), the inability to separate $\sigma^2$ and $\phi$ is harmless: any pair of $(\sigma^2, \phi)$ sharing the same microergodic ratio produces asymptotically identical kriging predictions and prediction variances!

> [!NOTE]
> **Wait—Why Did We Do That?**
> **FAQ 2: Why does increasing sample size $n$ in a fixed region fail to drive the sample mean variance to zero?**
>
> In introductory mathematical statistics, the Law of Large Numbers guarantees that as sample size $n \to \infty$, the sample mean $\bar{Y}$ converges in probability to the population mean $\mu$, with variance $\sigma^2 / n \to 0$. Why does this foundational guarantee collapse under fixed-domain spatial sampling?
>
> The reason lies in the distinction between **information accumulation** and **data interpolation**:
> 1. **Classical i.i.d. Sampling:** Every additional observation draws an independent outcome $\omega_i \in \Omega$ from the probability space. Ten thousand independent data points yield ten thousand independent realizations of the random variable.
> 2. **Fixed-Domain Spatial Infill:** In spatial statistics, we observe only a single realization $\omega \in \Omega$ across the spatial study region $D$. The entire continuous spatial surface across $D$ may happen to sit slightly higher or lower than the global ensemble mean $\mu$ because of regional, long-wavelength stochastic fluctuations.
> 3. **Sampling Redundancy:** Placing 100,000 sensors inside a 1-kilometer square yields 100,000 measurements that are almost perfectly correlated with one another. These sensors provide nearly exhaustive knowledge about that *specific realization's* surface within that square kilometer, but zero additional independent information about the *ensemble* expectation $\mu$ across alternative realizations of the universe.
>
> As $n \to \infty$ within a fixed domain $D$, the discrete sample mean $\bar{Z}$ converges to the spatial surface average $\frac{1}{|D|}\int_D Z(\mathbf{s}) d\mathbf{s}$. Because that spatial integral is itself a random variable with positive variance $\frac{1}{|D|^2} \int_D \int_D C(\mathbf{u}, \mathbf{v}) d\mathbf{u} d\mathbf{v} > 0$, the variance of $\bar{Z}$ is strictly bounded away from zero. Sampling more densely in a fixed region NEVER reduces variance to zero.

---

### 1.7 Analytical Exercises with Complete Solutions

#### Exercise 1.1: Downward Bias of Sample Variance under Spatial Dependence
Let $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ be observed from a second-order stationary field with mean $\mu$ and covariance matrix $\mathbf{V}$. Let $S^2 = \frac{1}{n-1}\sum_{i=1}^n (Z(\mathbf{s}_i) - \bar{Z})^2$ be the classical sample variance.
- **(a)** Express $S^2$ as a quadratic form $\mathbf{Z}'\mathbf{A}\mathbf{Z}$ and find $\mathbf{A}$.
- **(b)** Prove that $\operatorname{E}[S^2] = \sigma^2 - \frac{1}{n(n-1)}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)$, demonstrating the downward bias under positive spatial correlation.

*Solution.*
- **(a)** We write $\bar{Z} = \frac{1}{n}\mathbf{1}'\mathbf{Z}$. Then the centered vector is:
  $$
  \mathbf{Z} - \bar{Z}\mathbf{1} = (\mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}')\mathbf{Z}
  $$
  The sum of squared deviations is:
  $$
  (n-1)S^2 = \mathbf{Z}'(\mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}')'(\mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}')\mathbf{Z} = \mathbf{Z}'(\mathbf{I}_n - \frac{1}{n}\mathbf{J}_n)\mathbf{Z}
  $$
  where $\mathbf{J}_n = \mathbf{1}\mathbf{1}'$. Thus $\mathbf{A} = \frac{1}{n-1}(\mathbf{I}_n - \frac{1}{n}\mathbf{J}_n)$.
- **(b)** For any quadratic form with $\operatorname{E}[\mathbf{Z}] = \mu \mathbf{1}$:
  $$
  \operatorname{E}[\mathbf{Z}'\mathbf{A}\mathbf{Z}] = \operatorname{Tr}(\mathbf{A}\mathbf{V}) + \mu^2 \mathbf{1}'\mathbf{A}\mathbf{1}
  $$
  Since $\mathbf{A}\mathbf{1} = \frac{1}{n-1}(\mathbf{1} - \frac{1}{n}\mathbf{1}(n)) = \mathbf{0}$, the second term vanishes. Now evaluate the trace:
  $$
  \operatorname{Tr}(\mathbf{A}\mathbf{V}) = \frac{1}{n-1}\left(\operatorname{Tr}(\mathbf{V}) - \frac{1}{n}\operatorname{Tr}(\mathbf{J}_n \mathbf{V})\right)
  $$
  Since $\operatorname{Tr}(\mathbf{V}) = n C(\mathbf{0}) = n\sigma^2$ and $\operatorname{Tr}(\mathbf{J}_n \mathbf{V}) = \mathbf{1}'\mathbf{V}\mathbf{1} = \sum_{i=1}^n \sum_{j=1}^n C(\mathbf{s}_i - \mathbf{s}_j) = n\sigma^2 + \sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)$:
  $$
  \begin{aligned}
  \operatorname{E}[S^2] &= \frac{1}{n-1}\left(n\sigma^2 - \sigma^2 - \frac{1}{n}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)\right) \\
  &= \sigma^2 - \frac{1}{n(n-1)}\sum_{i \ne j} C(\mathbf{s}_i - \mathbf{s}_j)
  \end{aligned}
  $$
  If spatial correlation is positive ($C(\mathbf{s}_i - \mathbf{s}_j) > 0$), the second term is strictly positive, proving $\operatorname{E}[S^2] < \sigma^2$. Classical sample variance systematically underestimates true variability. $\blacksquare$

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
  The variance depends solely on lag $h$, independent of location $s$. Therefore, $B_H(s)$ is intrinsically stationary with semivariogram:
  $$
  \gamma(h) = \frac{1}{2}h^{2H}
  $$
- **(b)** The point variance is $\operatorname{Var}(B_H(s)) = C(s, s) = s^{2H}$. Because this depends explicitly on location $s$ and diverges as $s \to \infty$, $B_H(s)$ is not second-order stationary. $\blacksquare$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 1: Measure-Theoretic and Stochastic Foundations — Companion Exercises and Complete Solutions](ch01_exercises.md)
