# Chapter 6: Spatial Regression and Universal Kriging

This chapter develops the statistical and mathematical framework for spatial prediction when the expectation of a spatial random field varies systematically across geographic space—that is, when the process is **non-stationary in the mean**. In such settings, spatial variation exhibits a dual nature: large-scale deterministic trends (drift) governed by spatial coordinates or exogenous physical covariates, superimposed upon small-scale autocorrelated stochastic fluctuations governed by a stationary residual covariance kernel.

By the end of this chapter, you will be able to:
1. Identify the structural pathologies of fitting Ordinary Least Squares (OLS) to spatially autocorrelated data, including downward standard error bias and residual semivariogram sill attenuation.
2. Decorrelate spatial errors via Cholesky pre-whitening and derive the Generalized Least Squares (GLS) drift estimator under the Gauss–Markov theorem.
3. Formulate the primal Universal Kriging (UK) saddle-point system, prove its equivalence to two-stage GLS drift prediction plus Simple Kriging of residuals, and compute its exact prediction variance decomposition.
4. Implement Kriging with External Drift (KED) using continuous physical covariates, diagnosing numerical collinearity and applying neighborhood fallback rules.
5. Translate between primal and Dual Kriging representations, connecting spatial kernel predictors to natural smoothing splines and Gaussian process regression.
6. Contrast coordinate polynomial drift models with Intrinsic Random Functions of order $k$ (IRF-$k$) and authorized linear combinations.

### Chapter at a Glance

| Section | Classical Anchor | Core Spatial Concept | Key Mathematical Result | Practical Deliverable |
|---|---|---|---|---|
| **6.1 Spatial Linear Models** | Multiple linear regression with correlated errors | Trend plus stationary residual decomposition | $\operatorname{E}[\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}}\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}}'] = \mathbf{H}_\perp \mathbf{V}\mathbf{H}_\perp$; variogram parabolic inflation | Coordinate centering and scaling to prevent design matrix ill-conditioning |
| **6.2 Generalized Least Squares** | Weighted Least Squares (WLS) | Cholesky pre-whitening $\mathbf{L}^{-1}$ | Gauss–Markov theorem: $\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$ | Robust drift standard errors correcting naive OLS underestimation |
| **6.3 Universal Kriging** | Regression prediction intervals | Joint drift estimation and spatial interpolation | $\sigma_{\text{UK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \mathbf{r}_0'(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{r}_0$ | Exact BLUP predictor balancing local screening with global trend uncertainty |
| **6.4 Kriging with External Drift** | Multiple regression with physical covariates | Auxiliary landscape/elevation trends | Augmented KED block system with continuous covariates | Fallback to Ordinary Kriging when local covariate variance is deficient |
| **6.5 Dual Kriging & Splines** | Ridge regression dual form / GP regression | Pre-solved continuous surface representation | $\widehat{Y}(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\mathbf{d} + \mathbf{v}(\mathbf{s})'\mathbf{b}$; 1D cubic spline equivalence | Rapid grid evaluation ($O(n)$ per target instead of $O(n^2)$) |
| **6.6 Intrinsic Random Functions (IRF-$k$)** | Box–Jenkins ARIMA polynomial differencing | Authorized Linear Combinations (ALC-$k$) | $\mathbf{X}_k'\boldsymbol{\lambda} = \mathbf{0}$; generalized covariance kernels $K(h)$ | Trend filtering without explicit polynomial drift parameter estimation |

---

### Classical-to-Spatial Bridge: From OLS to GLS via Cholesky Whitening, and the Transition to Universal Kriging

Before entering the technical machinery of Universal Kriging, we trace the conceptual pipeline connecting classical linear regression to spatial prediction. This bridge clarifies why classical Ordinary Least Squares fails in spatial domains, how matrix factorization restores Gauss–Markov optimality, and how Generalized Least Squares naturally generalizes into Universal Kriging.

#### 1. Classical Homoscedastic Linear Regression
In introductory linear regression, an $n$-dimensional observation vector $\mathbf{Y} \in \mathbb{R}^n$ is modeled as a linear combination of $p$ explanatory variables corrupted by independent and identically distributed (i.i.d.) Gaussian errors:
$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}\left(\mathbf{0}, \sigma^2 \mathbf{I}_n\right)
$$
where $\mathbf{X} \in \mathbb{R}^{n \times p}$ is a design matrix of full column rank $p$, $\boldsymbol{\beta} \in \mathbb{R}^p$ is an unknown regression vector, and $\sigma^2 > 0$ is the constant residual variance. Under the classical Gauss–Markov assumptions, the Ordinary Least Squares (OLS) estimator:
$$
\widehat{\boldsymbol{\beta}}_{\text{OLS}} = \left(\mathbf{X}'\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{Y}
$$
is the unique Best Linear Unbiased Estimator (BLUE) of $\boldsymbol{\beta}$, with sampling covariance matrix:
$$
\operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{OLS}}\right) = \sigma^2 \left(\mathbf{X}'\mathbf{X}\right)^{-1}
$$

#### 2. The Spatial Breakdown: Dense Covariance and Spurious Precision
When observations $Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_n)$ are sampled across a spatial domain $D \subset \mathbb{R}^d$, Tobler's First Law dictates that proximate locations share correlated stochastic fluctuations. The residual vector $\boldsymbol{\varepsilon} = \mathbf{Y} - \mathbf{X}\boldsymbol{\beta}$ is no longer spherical white noise. Instead, it is governed by a dense spatial covariance matrix:
$$
\mathbf{V} = \operatorname{Var}(\boldsymbol{\varepsilon}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n \in \mathbb{R}^{n \times n}
$$
where $[C(\mathbf{s}_i - \mathbf{s}_j)]$ reflects spatially structured covariance and $\tau^2 \ge 0$ is the nugget variance. Applying classical OLS regression to spatially correlated data introduces severe methodological pathologies:
1. **Loss of Gauss–Markov Optimality:** The OLS estimator $\widehat{\boldsymbol{\beta}}_{\text{OLS}}$ remains unbiased ($\operatorname{E}[\widehat{\boldsymbol{\beta}}_{\text{OLS}}] = \boldsymbol{\beta}$), but it is no longer BLUE. Its true sampling covariance:
   $$
   \operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{OLS}}\right) = \left(\mathbf{X}'\mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V} \mathbf{X} \left(\mathbf{X}'\mathbf{X}\right)^{-1}
   $$
   is larger or equal (in the Loewner matrix partial ordering, $\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{OLS}}) \succeq \operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}})$) than the optimal Generalized Least Squares variance.
2. **Drastic Underestimation of Standard Errors:** Standard statistical software computes parameter standard errors under the naive i.i.d. formula $\widehat{\sigma}^2(\mathbf{X}'\mathbf{X})^{-1}$. Under positive spatial correlation, this naive variance drastically underestimates the true sampling uncertainty. In typical spatial configurations, true OLS standard errors are $2$ to $8$ times larger than naive standard errors (so naive standard errors are $50\%$ to $85\%$ too small). Consequently, $t$-statistics and $F$-statistics are artificially inflated, leading analysts to declare spatial covariates "statistically significant" when they are pure artifacts of unmodeled spatial autocorrelation.
3. **Sill Attenuation in Residual Variograms:** As detailed in Section 6.1, OLS residuals $\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} = (\mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}')\mathbf{Y}$ are constrained to be orthogonal to $\mathbf{X}$. This causes the regression plane to absorb low-frequency spatial variation, creating severe downward bias in the empirical semivariogram sill.

#### 3. Cholesky Whitening: Restoring Gauss–Markov Optimality
To overcome these failures, we transform the correlated observation space into an uncorrelated space where the classical Gauss–Markov theorem applies directly. This transformation is known as **pre-whitening**.

Because the spatial covariance matrix $\mathbf{V}$ is symmetric positive definite, it admits a unique Cholesky decomposition:
$$
\mathbf{V} = \mathbf{L}\mathbf{L}'
$$
where $\mathbf{L} \in \mathbb{R}^{n \times n}$ is a lower-triangular matrix with strictly positive diagonal entries. The inverse factor $\mathbf{L}^{-1}$ decorrelates the spatial errors. We transform both the response vector and the design matrix:
$$
\mathbf{Y}^* = \mathbf{L}^{-1}\mathbf{Y}, \qquad \mathbf{X}^* = \mathbf{L}^{-1}\mathbf{X}, \qquad \boldsymbol{\varepsilon}^* = \mathbf{L}^{-1}\boldsymbol{\varepsilon}
$$
Evaluating the covariance matrix of the transformed residuals $\boldsymbol{\varepsilon}^*$:
$$
\operatorname{Var}\left(\boldsymbol{\varepsilon}^*\right) = \mathbf{L}^{-1}\operatorname{Var}(\boldsymbol{\varepsilon})\left(\mathbf{L}^{-1}\right)' = \mathbf{L}^{-1}\mathbf{V}\left(\mathbf{L}'\right)^{-1} = \mathbf{L}^{-1}\left(\mathbf{L}\mathbf{L}'\right)\left(\mathbf{L}'\right)^{-1} = \mathbf{I}_n
$$
In this whitened coordinate system, the errors $\boldsymbol{\varepsilon}^*$ are uncorrelated and homoscedastic with unit variance. Applying classical Ordinary Least Squares to the transformed linear model $\mathbf{Y}^* = \mathbf{X}^*\boldsymbol{\beta} + \boldsymbol{\varepsilon}^*$:
$$
\widehat{\boldsymbol{\beta}} = \left( (\mathbf{X}^*)' \mathbf{X}^* \right)^{-1} (\mathbf{X}^*)' \mathbf{Y}^* = \left( \mathbf{X}' (\mathbf{L}^{-1})' \mathbf{L}^{-1} \mathbf{X} \right)^{-1} \mathbf{X}' (\mathbf{L}^{-1})' \mathbf{L}^{-1} \mathbf{Y}
$$
Recognizing that $(\mathbf{L}')^{-1}\mathbf{L}^{-1} = (\mathbf{L}\mathbf{L}')^{-1} = \mathbf{V}^{-1}$, the estimator simplifies immediately to the **Generalized Least Squares (GLS)** estimator:
$$
\widehat{\boldsymbol{\beta}}_{\text{GLS}} = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Y}
$$
with exact covariance matrix:
$$
\operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) = \left((\mathbf{X}^*)'\mathbf{X}^*\right)^{-1} = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}
$$

#### 4. The Transition to Universal Kriging: Drift Plus Residual BLUP
In classical linear regression, prediction at a new location $\mathbf{s}_0$ with covariate profile $\mathbf{x}(\mathbf{s}_0) \in \mathbb{R}^p$ relies solely on the estimated regression trend:
$$
\widehat{Y}(\mathbf{s}_0) = \mathbf{x}(\mathbf{s}_0)'\widehat{\boldsymbol{\beta}}
$$
This is valid only because the unobserved error $\varepsilon(\mathbf{s}_0)$ is assumed independent of the observed sample errors $\boldsymbol{\varepsilon}$.

In spatial statistics, however, the unobserved residual $\varepsilon(\mathbf{s}_0)$ remains correlated with the observed residuals $\boldsymbol{\varepsilon}$ through the cross-covariance vector:
$$
\mathbf{c}_0 = \left[ C(\mathbf{s}_0 - \mathbf{s}_1), \dots, C(\mathbf{s}_0 - \mathbf{s}_n) \right]' \in \mathbb{R}^n
$$
Optimal spatial prediction at an unsampled location therefore decomposes into a **two-component additive structure**:
$$
\widehat{Y}_{\text{UK}}(\mathbf{s}_0) = \underbrace{\mathbf{x}(\mathbf{s}_0)'\widehat{\boldsymbol{\beta}}_{\text{GLS}}}_{\text{Deterministic GLS Drift Prediction}} + \underbrace{\mathbf{c}_0'\mathbf{V}^{-1}\left(\mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right)}_{\text{Simple Kriging of Autocorrelated Spatial Residuals}}
$$
This unified predictor is **Universal Kriging (UK)**. It bridges regression modeling and spatial interpolation:
- If spatial correlation vanishes ($\mathbf{c}_0 = \mathbf{0}$, $\mathbf{V} = \sigma^2 \mathbf{I}$), the residual kriging term collapses to zero, and Universal Kriging reduces to classical linear regression prediction $\mathbf{x}(\mathbf{s}_0)'\widehat{\boldsymbol{\beta}}_{\text{OLS}}$.
- If the trend is stationary and known ($\mathbf{X}\boldsymbol{\beta} = \mu \mathbf{1}$), Universal Kriging reduces to Simple Kriging.
- If the trend is an unknown constant ($\mathbf{X} = \mathbf{1}, p = 1$), Universal Kriging reduces to Ordinary Kriging.

---

### 6.1 Spatial Linear Models with Correlated Residuals

#### 6.1.1 The Mathematical Anchor: Trend and Correlated Error Decomposition

In environmental sciences, geophysics, and spatial epidemiology, spatial processes rarely maintain a constant mean across regional domains. Topographic elevation gradients, latitudinal temperature trends, atmospheric wind fields, and geological depositional directions induce deterministic spatial drift. 

If one ignores this trend and computes an empirical semivariogram directly on raw observations $Y(\mathbf{s})$, the semivariogram displays an apparent parabolic upward curvature ($\gamma(h) \propto h^2$) at large lags. 

#### Proposition 6.1.1 (Effect of Linear Trend on Raw Variogram)
Let $Y(\mathbf{s}) = \mathbf{a}'\mathbf{s} + \varepsilon(\mathbf{s})$, where $\mathbf{a}'\mathbf{s}$ is a linear spatial trend and $\varepsilon(\mathbf{s})$ is an intrinsically stationary residual field with semivariogram $\gamma_\varepsilon(\mathbf{h})$.
Then the variogram of the raw observations $Y(\mathbf{s})$ satisfies:
$$
\gamma_Y(\mathbf{h}) = \frac{1}{2} \operatorname{E}\left[(Y(\mathbf{s}+\mathbf{h}) - Y(\mathbf{s}))^2\right] = \gamma_\varepsilon(\mathbf{h}) + \frac{1}{2} (\mathbf{a}'\mathbf{h})^2
$$
*Proof.*
Expand the expected squared difference:
$$
\begin{aligned}
Y(\mathbf{s}+\mathbf{h}) - Y(\mathbf{s}) &= \left(\mathbf{a}'(\mathbf{s}+\mathbf{h}) + \varepsilon(\mathbf{s}+\mathbf{h})\right) - \left(\mathbf{a}'\mathbf{s} + \varepsilon(\mathbf{s})\right) \\
&= \mathbf{a}'\mathbf{h} + (\varepsilon(\mathbf{s}+\mathbf{h}) - \varepsilon(\mathbf{s}))
\end{aligned}
$$
Taking the expectation of the square:
$$
\begin{aligned}
\operatorname{E}\left[(Y(\mathbf{s}+\mathbf{h}) - Y(\mathbf{s}))^2\right] &= \operatorname{E}\left[(\mathbf{a}'\mathbf{h})^2 + 2(\mathbf{a}'\mathbf{h})(\varepsilon(\mathbf{s}+\mathbf{h}) - \varepsilon(\mathbf{s})) + (\varepsilon(\mathbf{s}+\mathbf{h}) - \varepsilon(\mathbf{s}))^2\right] \\
&= (\mathbf{a}'\mathbf{h})^2 + 2(\mathbf{a}'\mathbf{h}) \cdot 0 + 2\gamma_\varepsilon(\mathbf{h}) \\
&= 2\gamma_\varepsilon(\mathbf{h}) + (\mathbf{a}'\mathbf{h})^2
\end{aligned}
$$
Dividing by 2 yields $\gamma_Y(\mathbf{h}) = \gamma_\varepsilon(\mathbf{h}) + \frac{1}{2}(\mathbf{a}'\mathbf{h})^2$. $\blacksquare$

The quadratic term $\frac{1}{2}(\mathbf{a}'\mathbf{h})^2$ never plateaus, creating the illusion of an infinite-range, non-stationary process and concealing the true spatial structure of $\gamma_\varepsilon(\mathbf{h})$.

#### Definition 6.1.1 (Spatial Linear Model)
Let $\{Y(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$ be a spatial continuous random process. The spatial linear model decomposes $Y(\mathbf{s})$ into a deterministic drift plus an autocorrelated stochastic error:
$$
Y(\mathbf{s}) = \mu(\mathbf{s}) + \varepsilon(\mathbf{s}) = \sum_{j=1}^p \beta_j f_j(\mathbf{s}) + \varepsilon(\mathbf{s}) = \mathbf{x}(\mathbf{s})' \boldsymbol{\beta} + \varepsilon(\mathbf{s})
$$
where:
1. $\mathbf{x}(\mathbf{s}) = (f_1(\mathbf{s}), \dots, f_p(\mathbf{s}))' \in \mathbb{R}^p$ is a vector of $p$ known spatial basis functions or external physical covariates evaluated at site $\mathbf{s}$.
2. $\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)' \in \mathbb{R}^p$ is an unknown regression parameter vector.
3. $\varepsilon(\mathbf{s}) = \delta(\mathbf{s}) + \epsilon(\mathbf{s})$ is a zero-mean, second-order stationary spatial residual field consisting of a spatially continuous stochastic field $\delta(\mathbf{s})$ with covariance function $C(\mathbf{h})$ and an independent white-noise measurement error (nugget effect) $\epsilon(\mathbf{s})$ with variance $\tau^2 \ge 0$.

When evaluated across $n$ observation locations $\mathbf{s}_1, \dots, \mathbf{s}_n$ ($n > p$):
$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$
where $\mathbf{Y} \in \mathbb{R}^n$, $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the spatial design matrix with rows $\mathbf{x}(\mathbf{s}_i)'$, and $\operatorname{Var}(\boldsymbol{\varepsilon}) = \mathbf{V} \in \mathbb{R}^{n \times n}$ has entries:
$$
\mathbf{V}_{ij} = \operatorname{Cov}(\varepsilon(\mathbf{s}_i), \varepsilon(\mathbf{s}_j)) = C(\mathbf{s}_i - \mathbf{s}_j) + \tau^2 \mathbb{1}_{\{i = j\}}
$$

#### 6.1.2 Algorithmic Translation: Design Matrix Construction, Centering, and Scaling

In spatial regression, polynomial coordinate bases (e.g. $[1, s_x, s_y]$ or $[1, s_x, s_y, s_x^2, s_y^2, s_x s_y]$) frequently suffer from extreme numerical collinearity if raw geographic coordinates (such as UTM coordinates measured in hundreds of thousands of meters) are used directly. 

For example, on a regional domain with coordinates near $s_x \approx 500{,}000\text{ m}$, the column of ones and the column of coordinates have entries of wildly disparate scales, driving the condition number $\kappa(\mathbf{X}'\mathbf{X})$ above $10^{12}$. Centering and scaling coordinates to $[-1, 1]$ or standardizing to mean zero and unit variance immediately reduces $\kappa(\mathbf{X}'\mathbf{X})$ to modest numbers ($10^1$ to $10^4$), eliminating catastrophic numerical roundoff in Cholesky and Gaussian elimination solvers.

The following dependency-free Python implementation constructs the coordinate design matrix $\mathbf{X}$ with automatic coordinate standardization:

```python
def build_spatial_design_matrix(sample_coords, basis_type='linear', standardize=True):
    """
    Constructs the spatial design matrix X for coordinate polynomial drift,
    with automatic coordinate standardization to preserve numerical stability.

    Parameters:
    -----------
    sample_coords : list of tuples of float
        Spatial coordinates [(x_1, y_1), ..., (x_n, y_n)].
    basis_type : str
        Type of drift basis: 'constant' (p=1), 'linear' (p=3), or 'quadratic' (p=6).
    standardize : bool
        If True, centers and scales coordinates to mean 0, variance 1 before assembling X.

    Returns:
    --------
    X : list of lists of float, shape (n, p)
        The assembled spatial design matrix.
    scaling_params : dict
        Mean and standard deviation used for standardization.
    """
    n = len(sample_coords)
    dim = len(sample_coords[0])

    # Compute scaling parameters
    scaling_params = {'mean': [0.0] * dim, 'std': [1.0] * dim}
    coords_proc = [list(c) for c in sample_coords]

    if standardize and n > 1:
        for k in range(dim):
            mean_k = sum(c[k] for c in sample_coords) / n
            var_k = sum((c[k] - mean_k) ** 2 for c in sample_coords) / (n - 1)
            std_k = var_k ** 0.5 if var_k > 1e-12 else 1.0
            scaling_params['mean'][k] = mean_k
            scaling_params['std'][k] = std_k
            for i in range(n):
                coords_proc[i][k] = (sample_coords[i][k] - mean_k) / std_k

    X = []
    for s in coords_proc:
        if basis_type == 'constant':
            X.append([1.0])
        elif basis_type == 'linear':
            X.append([1.0, float(s[0]), float(s[1])])
        elif basis_type == 'quadratic':
            sx, sy = float(s[0]), float(s[1])
            X.append([1.0, sx, sy, sx * sx, sy * sy, sx * sy])
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    return X, scaling_params
```

#### 6.1.3 Intuition, Edge Cases & Diagnostic Verification: The Pitfall of OLS Residuals

#### The Projection Bias of OLS Residuals
If an analyst naively estimates the drift via Ordinary Least Squares:
$$
\widehat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}
$$
the estimator remains unbiased ($\operatorname{E}[\widehat{\boldsymbol{\beta}}_{\text{OLS}}] = \boldsymbol{\beta}$), but its variance is inflated:
$$
\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{OLS}}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1} \succeq (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}
$$
More crucially, if the analyst evaluates the empirical semivariogram on the OLS residuals:
$$
\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} = \mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{OLS}} = \left(\mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\right)\mathbf{Y} = \mathbf{H}_\perp \boldsymbol{\varepsilon}
$$
the expectation of the residual outer product satisfies:
$$
\operatorname{E}\left[\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} \widehat{\boldsymbol{\varepsilon}}_{\text{OLS}}'\right] = \mathbf{H}_\perp \mathbf{V} \mathbf{H}_\perp \ne \mathbf{V}
$$
Because the geometric projection matrix $\mathbf{H}_\perp$ forces the residuals to be orthogonal to the columns of $\mathbf{X}$ ($\mathbf{X}'\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} = \mathbf{0}$), it absorbs low-frequency spatial variation into the fitted trend.

The following benchmark demonstrates the quantitative severity of this sill attenuation. For $n = 100$ random sites in the unit square with true Exponential covariance (range $\phi = 0.3$, sill $\sigma^2 = 1.0$), the expected semivariogram of OLS residuals divided by the true semivariogram $\operatorname{E}[\widehat{\gamma}_{\text{OLS}}(h)] / \gamma(h)$ across distance bins is:

| Spatial Lag Bin $h$ | Linear Drift ($p = 3$) | Quadratic Drift ($p = 6$) |
|---|---|---|
| $(0.0, 0.1]$ | $0.99$ | $0.97$ |
| $(0.2, 0.3]$ | $0.93$ | $0.80$ |
| $(0.4, 0.5]$ | $0.82$ | $0.59$ |
| $(0.6, 0.8]$ | $0.63$ | $0.43$ |

At short lags ($h < 0.1$), OLS residuals are virtually unbiased ($99\%$ of truth). But at long lags, the fitted trend absorbs the spatial variation: the empirical sill of OLS residuals collapses to $63\%$ of the true variance under a planar trend and $43\%$ under a quadratic trend! An analyst fitting an empirical semivariogram to OLS residuals will systematically underestimate both the sill $\sigma^2$ and the spatial range $\phi$.

#### Resolving the Circularity: The Practical Iterative Workflow
This creates the **trend-covariance confounding dilemma**: one needs $\mathbf{V}$ to compute optimal GLS drift estimates, but needs the drift $\boldsymbol{\beta}$ to obtain residuals and estimate $\mathbf{V}$. In practice, this circularity is resolved via an iterative procedure:
1. **Center and scale** spatial coordinates to ensure well-conditioned design matrices.
2. Fit an initial drift $\widehat{\boldsymbol{\beta}}_{\text{OLS}}$ via OLS.
3. Compute initial residuals $\mathbf{e}^{(0)} = \mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{OLS}}$ and compute their empirical semivariogram $\widehat{\gamma}(h)$ at **short lags only** ($h \le \frac{1}{3} \operatorname{diam}(D)$), where projection bias is minimal.
4. Fit an initial covariance kernel $\mathbf{V}^{(0)} = \mathbf{C}(\boldsymbol{\theta}^{(0)}) + \tau^2 \mathbf{I}$.
5. Re-estimate drift via GLS: $\widehat{\boldsymbol{\beta}}_{\text{GLS}}^{(1)} = (\mathbf{X}'(\mathbf{V}^{(0)})^{-1}\mathbf{X})^{-1}\mathbf{X}'(\mathbf{V}^{(0)})^{-1}\mathbf{Y}$.
6. Re-estimate covariance parameters via **Restricted Maximum Likelihood (REML)**, which evaluates the likelihood strictly on error contrasts $\mathbf{A}'\mathbf{Y}$ satisfying $\mathbf{A}'\mathbf{X} = \mathbf{0}$, removing drift dependence entirely.
7. Perform Universal Kriging prediction and validate via leave-one-out cross-validation residuals.

---

### 6.2 Generalized Least Squares (GLS) Estimation of Spatial Drift

#### 6.2.1 The Mathematical Anchor: Gauss–Markov Optimality via Cholesky Decorrelation

When observations share spatial correlation, Ordinary Least Squares fails to be optimal. The Generalized Least Squares (GLS) estimator decorrelates and scales the observation space to whiten the autocorrelated errors, yielding the Best Linear Unbiased Estimator (BLUE).

#### Theorem 6.2.1 (Gauss–Markov Theorem for Spatial Drift Estimation)
Let $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}$ and $\operatorname{Var}(\boldsymbol{\varepsilon}) = \mathbf{V}$, where $\mathbf{V}$ is symmetric positive definite and $\mathbf{X}$ has full column rank $p$.
1. The **Generalized Least Squares (GLS) estimator** of $\boldsymbol{\beta}$ is:
   $$
   \widehat{\boldsymbol{\beta}}_{\text{GLS}} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}
   $$
2. The estimator $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ is strictly unbiased:
   $$
   \operatorname{E}\left[\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right] = \boldsymbol{\beta}
   $$
3. The covariance matrix of $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ is:
   $$
   \operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1}
   $$
4. $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ is the Best Linear Unbiased Estimator (BLUE) of $\boldsymbol{\beta}$. For any linear combination $\mathbf{a}'\boldsymbol{\beta}$ with $\mathbf{a} \in \mathbb{R}^p$, the estimator $\mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ has minimal variance among all linear unbiased estimators $\mathbf{w}'\mathbf{Y}$:
   $$
   \operatorname{Var}\left(\mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) \le \operatorname{Var}\left(\mathbf{w}'\mathbf{Y}\right)
   $$
   with equality if and only if $\mathbf{w}'\mathbf{Y} = \mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ almost surely.

*Proof.*
1. **Cholesky Pre-Whitening Transformation:** Because $\mathbf{V}$ is symmetric positive definite, there exists a unique lower-triangular Cholesky factor $\mathbf{L}$ such that $\mathbf{V} = \mathbf{L}\mathbf{L}'$, and an inverse factor $\mathbf{L}^{-1}$ satisfying $\mathbf{V}^{-1} = (\mathbf{L}^{-1})'\mathbf{L}^{-1}$.
   Pre-multiply the linear system by $\mathbf{L}^{-1}$:
   $$
   \mathbf{Y}^* = \mathbf{X}^* \boldsymbol{\beta} + \boldsymbol{\varepsilon}^*
   $$
   where $\mathbf{Y}^* = \mathbf{L}^{-1}\mathbf{Y}$, $\mathbf{X}^* = \mathbf{L}^{-1}\mathbf{X}$, and $\boldsymbol{\varepsilon}^* = \mathbf{L}^{-1}\boldsymbol{\varepsilon}$.
   Evaluate the moments of the transformed residuals:
   $$
   \operatorname{E}[\boldsymbol{\varepsilon}^*] = \mathbf{L}^{-1}\operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}
   $$
   $$
   \operatorname{Var}(\boldsymbol{\varepsilon}^*) = \mathbf{L}^{-1}\operatorname{Var}(\boldsymbol{\varepsilon})(\mathbf{L}^{-1})' = \mathbf{L}^{-1}\mathbf{V}(\mathbf{L}')^{-1} = \mathbf{L}^{-1}(\mathbf{L}\mathbf{L}')(\mathbf{L}')^{-1} = \mathbf{I}_n
   $$
   The transformed model satisfies the classical homoscedastic Gauss–Markov conditions with uncorrelated white noise. The OLS estimator for the transformed model is:
   $$
   \widehat{\boldsymbol{\beta}} = \left((\mathbf{X}^*)' \mathbf{X}^*\right)^{-1} (\mathbf{X}^*)' \mathbf{Y}^* = \left(\mathbf{X}' (\mathbf{L}^{-1})' \mathbf{L}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' (\mathbf{L}^{-1})' \mathbf{L}^{-1} \mathbf{Y} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}
   $$

2. **Unbiasedness:**
   $$
   \operatorname{E}\left[\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right] = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \operatorname{E}[\mathbf{Y}] = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} (\mathbf{X}\boldsymbol{\beta}) = \boldsymbol{\beta}
   $$

3. **Covariance Matrix:**
   $$
   \begin{aligned}
   \operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) &= \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \operatorname{Var}(\mathbf{Y}) \mathbf{V}^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \\
   &= \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{V} \mathbf{V}^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \\
   &= \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1}
   \end{aligned}
   $$

4. **Gauss–Markov Optimality (BLUE):** Let $\widehat{\theta} = \mathbf{w}'\mathbf{Y}$ be any competing linear unbiased estimator of $\theta = \mathbf{a}'\boldsymbol{\beta}$.
   Unbiasedness requires:
   $$
   \operatorname{E}[\mathbf{w}'\mathbf{Y}] = \mathbf{w}'\mathbf{X}\boldsymbol{\beta} = \mathbf{a}'\boldsymbol{\beta}, \quad \forall \boldsymbol{\beta} \in \mathbb{R}^p \iff \mathbf{X}'\mathbf{w} = \mathbf{a}
   $$
   Define the optimal GLS weight vector $\mathbf{w}_0 = \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{a}$. Notice that:
   $$
   \mathbf{w}_0'\mathbf{Y} = \mathbf{a}'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Y} = \mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}}
   $$
   and $\mathbf{X}'\mathbf{w}_0 = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{a} = \mathbf{a}$.
   Express the competing weight vector as $\mathbf{w} = \mathbf{w}_0 + \mathbf{u}$. The unbiasedness constraint requires:
   $$
   \mathbf{X}'\mathbf{w} = \mathbf{X}'(\mathbf{w}_0 + \mathbf{u}) = \mathbf{a} + \mathbf{X}'\mathbf{u} = \mathbf{a} \implies \mathbf{X}'\mathbf{u} = \mathbf{0}
   $$
   Now evaluate the variance of $\mathbf{w}'\mathbf{Y}$:
   $$
   \operatorname{Var}\left(\mathbf{w}'\mathbf{Y}\right) = \mathbf{w}'\mathbf{V}\mathbf{w} = (\mathbf{w}_0 + \mathbf{u})'\mathbf{V}(\mathbf{w}_0 + \mathbf{u}) = \mathbf{w}_0'\mathbf{V}\mathbf{w}_0 + 2\mathbf{w}_0'\mathbf{V}\mathbf{u} + \mathbf{u}'\mathbf{V}\mathbf{u}
   $$
   Examine the cross term:
   $$
   \mathbf{w}_0'\mathbf{V}\mathbf{u} = \left(\mathbf{a}'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}\right) \mathbf{V}\mathbf{u} = \mathbf{a}'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1} (\mathbf{X}'\mathbf{u})
   $$
   Because $\mathbf{X}'\mathbf{u} = \mathbf{0}$, the cross term vanishes identically!
   $$
   \operatorname{Var}\left(\mathbf{w}'\mathbf{Y}\right) = \mathbf{w}_0'\mathbf{V}\mathbf{w}_0 + \mathbf{u}'\mathbf{V}\mathbf{u} = \operatorname{Var}\left(\mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) + \mathbf{u}'\mathbf{V}\mathbf{u}
   $$
   Because $\mathbf{V}$ is positive definite, $\mathbf{u}'\mathbf{V}\mathbf{u} \ge 0$, with equality if and only if $\mathbf{u} = \mathbf{0}$.
   Thus $\operatorname{Var}(\mathbf{w}'\mathbf{Y}) \ge \operatorname{Var}(\mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}})$. $\blacksquare$

#### 6.2.2 Algorithmic Translation: GLS Drift Computation

The following dependency-free Python implementation computes the GLS drift estimate $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and its covariance matrix $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$:

```python
def gls_estimate_drift(sample_coords, sample_vals, design_matrix, cov_kernel):
    """
    Computes Generalized Least Squares (GLS) drift estimates and parameter covariance.

    Parameters:
    -----------
    sample_coords : list of tuples of float
        Spatial coordinates [(x_1, y_1), ..., (x_n, y_n)].
    sample_vals : list of float
        Observed response values Y.
    design_matrix : list of lists of float, shape (n, p)
        Spatial design matrix X.
    cov_kernel : callable
        Covariance function C(h).

    Returns:
    --------
    beta_gls : list of float, shape (p,)
        Estimated regression coefficients.
    cov_beta : list of lists of float, shape (p, p)
        Covariance matrix (X' V^-1 X)^-1.
    """
    n = len(sample_coords)
    p = len(design_matrix[0])
    d = len(sample_coords[0])

    # Step 1: Assemble covariance matrix V
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = sum((sample_coords[i][k] - sample_coords[j][k]) ** 2 for k in range(d)) ** 0.5
            V[i][j] = cov_kernel(dist)

    # Step 2: Solve V * [V_inv_X | V_inv_Y] = [X | Y] via Gauss-Jordan elimination
    aug = [V[i][:] + design_matrix[i][:] + [sample_vals[i]] for i in range(n)]

    for col in range(n):
        max_row = col
        max_val = abs(aug[col][col])
        for r in range(col + 1, n):
            if abs(aug[r][col]) > max_val:
                max_val = abs(aug[r][col])
                max_row = r
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        total_cols = len(aug[0])
        for c in range(col, total_cols):
            aug[col][c] /= pivot

        for r in range(n):
            if r != col:
                factor = aug[r][col]
                for c in range(col, total_cols):
                    aug[r][c] -= factor * aug[col][c]

    # Extract V_inv_X (n x p) and V_inv_Y (n x 1)
    V_inv_X = [[aug[i][n + j] for j in range(p)] for i in range(n)]
    V_inv_Y = [aug[i][n + p] for i in range(n)]

    # Step 3: Compute M = X' * V^-1 * X and rhs = X' * V^-1 * Y
    M = [[sum(design_matrix[i][r] * V_inv_X[i][c] for i in range(n)) for c in range(p)] for r in range(p)]
    rhs = [sum(design_matrix[i][r] * V_inv_Y[i] for i in range(n)) for r in range(p)]

    # Step 4: Solve M * beta_gls = rhs and invert M for cov_beta
    M_aug = [M[r][:] + [1.0 if r == c else 0.0 for c in range(p)] + [rhs[r]] for r in range(p)]
    for col in range(p):
        max_row = col
        max_val = abs(M_aug[col][col])
        for r in range(col + 1, p):
            if abs(M_aug[r][col]) > max_val:
                max_val = abs(M_aug[r][col])
                max_row = r
        M_aug[col], M_aug[max_row] = M_aug[max_row], M_aug[col]

        pivot = M_aug[col][col]
        total_cols = len(M_aug[0])
        for c in range(col, total_cols):
            M_aug[col][c] /= pivot

        for r in range(p):
            if r != col:
                factor = M_aug[r][col]
                for c in range(col, total_cols):
                    M_aug[r][c] -= factor * M_aug[col][c]

    cov_beta = [[M_aug[r][p + c] for c in range(p)] for r in range(p)]
    beta_gls = [M_aug[r][2 * p] for r in range(p)]

    return beta_gls, cov_beta
```

#### 6.2.3 Intuition, Edge Cases & Diagnostic Verification

#### What GLS Actually Accomplishes: Small Variance Gain, Massive Standard Error Correction
A widespread misconception is that GLS produces vastly different coefficient estimates $\widehat{\boldsymbol{\beta}}$ with dramatically smaller sampling variance than OLS. In reality, when design columns $\mathbf{X}$ represent smooth spatial trends (such as linear or planar coordinates), OLS is already remarkably efficient. The ratio of OLS variance to GLS variance $\operatorname{Var}(\widehat{\beta}_{\text{OLS}}) / \operatorname{Var}(\widehat{\beta}_{\text{GLS}})$ typically peaks between $1.10$ and $1.30$ (a modest $10\%$ to $25\%$ variance reduction) when the spatial range is comparable to the domain size, and then falls back toward $1.0$ as range grows further.

The true practical value of GLS lies in **honest uncertainty quantification**. Standard regression packages report naive OLS standard errors $\operatorname{SE}_{\text{naive}} = \sqrt{\widehat{\sigma}^2 [(\mathbf{X}'\mathbf{X})^{-1}]_{jj}}$, which assume independent errors. Under positive spatial correlation, the ratio of naive standard error to true standard error collapses dramatically:

| Spatial Correlation Range $\phi / \operatorname{diam}(D)$ | $\operatorname{Var}_{\text{OLS}} / \operatorname{Var}_{\text{GLS}}$ (Slope) | $\operatorname{SE}_{\text{naive}} / \operatorname{SE}_{\text{true}}$ (Slope) |
|---|---|---|
| $0.02$ (Weak correlation) | $1.02$ | $0.69$ |
| $0.10$ (Moderate correlation) | $1.17$ | $0.32$ |
| $0.30$ (Strong regional correlation) | $1.28$ | $0.20$ |
| $1.00$ (Domain-wide correlation) | $1.24$ | $0.14$ |
| $3.00$ (Very strong correlation) | $1.20$ | $0.13$ |

While GLS improves sampling variance by at most $28\%$, naive standard errors are too small by a factor of $5$ to $8$ ($\operatorname{SE}_{\text{naive}} / \operatorname{SE}_{\text{true}} \approx 0.13$)! Consequently, the point estimates barely change, but the standard errors you would have reported under OLS are drastically false.

> [!NOTE]
> **From the Lecture Hall: What is Spatial Confounding?**
> *Does Restricted Maximum Likelihood (REML) "cure" spatial confounding between trend and covariance?*
>
> Spatial confounding occurs when the columns of the design matrix $\mathbf{X}$ (such as an elevation or latitudinal gradient) align closely with the low-frequency eigenvectors of the spatial covariance matrix $\mathbf{V}$. When this spectral overlap occurs, the data cannot distinguish whether large-scale variation is driven by the physical covariate $\mathbf{X}\boldsymbol{\beta}$ or by a long-range realization of the autocorrelated stochastic field $\delta(\mathbf{s})$.
>
> It is often asserted that REML "resolves" this confounding. This is only half true:
> 1. **What REML does do:** REML projects the data onto the error contrast subspace $\mathbf{A}'\mathbf{X} = \mathbf{0}$, removing $\boldsymbol{\beta}$ from the likelihood entirely. This eliminates the small-sample downward bias that standard maximum likelihood suffers when estimating $(\sigma^2, \phi)$.
> 2. **What REML does not do:** REML does *not* break the physical collinearity between $\mathbf{X}$ and low-frequency spatial patterns. If a covariate is spatially smooth, the parameter variance $\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$ remains inflated, reflecting genuine scientific uncertainty about whether the trend or the stochastic field produced the observed pattern.

---

### 6.3 Universal Kriging (UK)

#### 6.3.1 The Mathematical Anchor: Saddle-Point Augmented Systems and Variance Decomposition

##### Classical-to-Spatial Bridge: Structural Parallel Between OLS and UK Prediction Variance
In classical linear models, predicting the response at a new site with covariate profile $\mathbf{x}_0$ gives $\widehat{Y} = \mathbf{x}_0'\widehat{\boldsymbol{\beta}}$, and the mean-squared prediction error variance decomposes as:
$$
\operatorname{Var}(Y_0 - \widehat{Y}_0) = \underbrace{\sigma^2}_{\text{Stochastic Process Noise}} + \underbrace{\sigma^2 \mathbf{x}_0'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_0}_{\text{Parameter Estimation Uncertainty}}
$$
This classical formula assumes that the unobserved error $\varepsilon_0$ is completely independent of the sample residuals $\boldsymbol{\varepsilon}$. 

In spatial statistics, the target site $\mathbf{s}_0$ is geographically proximate to sample points, meaning $\operatorname{Cov}(\varepsilon(\mathbf{s}_0), \boldsymbol{\varepsilon}) = \mathbf{c}_0 \ne \mathbf{0}$. Universal Kriging achieves the exact spatial counterpart to this classical decomposition:
$$
\sigma_{\text{UK}}^2(\mathbf{s}_0) = \underbrace{\sigma_{\text{SK}}^2(\mathbf{s}_0)}_{\text{Residual Uncertainty (if }\boldsymbol{\beta}\text{ were known)}} + \underbrace{(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0)'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0)}_{\text{Parameter Estimation Uncertainty (Drift Penalty)}}
$$
Notice the precise structural parallelism:
- The classical noise floor $\sigma^2$ is replaced by the Simple Kriging variance $\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0$, which exploits spatial autocorrelation to shrink uncertainty below the prior sill.
- The parameter uncertainty $(\mathbf{X}'\mathbf{X})^{-1}$ is replaced by the GLS covariance $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$.
- The target covariate $\mathbf{x}_0$ is replaced by the residual covariate vector $\mathbf{r}_0 = \mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0$, reflecting that proximate spatial observations already convey partial information about the local trend!

#### Definition 6.3.1 (Universal Kriging Predictor)
Let $Y(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \varepsilon(\mathbf{s})$ be the spatial linear model on $D$. Let $\mathbf{s}_0 \in D$ be the target prediction location with known drift basis vector $\mathbf{x}_0 = \mathbf{x}(\mathbf{s}_0) \in \mathbb{R}^p$. Let $\mathbf{c}_0 = (C(\mathbf{s}_0 - \mathbf{s}_1), \dots, C(\mathbf{s}_0 - \mathbf{s}_n))' \in \mathbb{R}^n$ be the cross-covariance vector.

The **Universal Kriging (UK) predictor** is a linear combination of the observations:
$$
\widehat{Y}_{\text{UK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Y(\mathbf{s}_i) = \boldsymbol{\lambda}' \mathbf{Y}
$$
such that:
1. **Uniform Unbiasedness:** For all parameter vectors $\boldsymbol{\beta} \in \mathbb{R}^p$:
   $$
   \operatorname{E}\left[\widehat{Y}_{\text{UK}}(\mathbf{s}_0)\right] = \operatorname{E}[Y(\mathbf{s}_0)] \iff \boldsymbol{\lambda}' \mathbf{X} \boldsymbol{\beta} = \mathbf{x}_0' \boldsymbol{\beta}, \quad \forall \boldsymbol{\beta} \in \mathbb{R}^p \iff \mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0
   $$
2. **Optimality:** The mean-squared prediction error $\operatorname{Var}(Y(\mathbf{s}_0) - \boldsymbol{\lambda}' \mathbf{Y})$ is minimized subject to $\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0$.

#### Theorem 6.3.1 (Universal Kriging Saddle-Point System)
The weight vector $\boldsymbol{\lambda} \in \mathbb{R}^n$ and the vector of Lagrange multipliers $\mathbf{m}_{\text{UK}} \in \mathbb{R}^p$ that minimize prediction error variance subject to $\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0$ satisfy the augmented linear system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ -\mathbf{m}_{\text{UK}} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ \mathbf{x}_0 \end{bmatrix}
$$
In expanded block equations:
$$
\begin{cases}
\mathbf{V} \boldsymbol{\lambda} - \mathbf{X} \mathbf{m}_{\text{UK}} = \mathbf{c}_0 \\
\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0
\end{cases}
$$
The minimized Universal Kriging prediction error variance is:
$$
\sigma_{\text{UK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{c}_0 + \mathbf{m}_{\text{UK}}' \mathbf{x}_0
$$

*Proof.*
Under the unbiasedness condition $\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0$, the prediction error is:
$$
Y(\mathbf{s}_0) - \widehat{Y}_{\text{UK}}(\mathbf{s}_0) = (\mathbf{x}_0' \boldsymbol{\beta} + \varepsilon(\mathbf{s}_0)) - \boldsymbol{\lambda}' (\mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}) = \varepsilon(\mathbf{s}_0) - \boldsymbol{\lambda}' \boldsymbol{\varepsilon}
$$
The mean-squared prediction error is:
$$
\operatorname{Var}(Y(\mathbf{s}_0) - \widehat{Y}_{\text{UK}}(\mathbf{s}_0)) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{c}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda}
$$
We formulate the Lagrangian with multiplier vector $2\mathbf{m}_{\text{UK}}$ enforcing $\mathbf{X}'\boldsymbol{\lambda} - \mathbf{x}_0 = \mathbf{0}$:
$$
\mathcal{L}(\boldsymbol{\lambda}, \mathbf{m}_{\text{UK}}) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{c}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} - 2 \mathbf{m}_{\text{UK}}' (\mathbf{X}' \boldsymbol{\lambda} - \mathbf{x}_0)
$$
Differentiating with respect to $\boldsymbol{\lambda}$ and setting the gradient to zero:
$$
\frac{1}{2} \nabla_{\boldsymbol{\lambda}} \mathcal{L} = -\mathbf{c}_0 + \mathbf{V}\boldsymbol{\lambda} - \mathbf{X}\mathbf{m}_{\text{UK}} = \mathbf{0} \implies \mathbf{V}\boldsymbol{\lambda} - \mathbf{X}\mathbf{m}_{\text{UK}} = \mathbf{c}_0
$$
Differentiating with respect to $\mathbf{m}_{\text{UK}}$ recovers the unbiasedness constraint $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{x}_0$. Combining these yields the augmented system.

To evaluate the minimized variance:
$$
\begin{aligned}
\sigma_{\text{UK}}^2(\mathbf{s}_0) &= C(\mathbf{0}) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} \\
&= C(\mathbf{0}) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'(\mathbf{c}_0 + \mathbf{X}\mathbf{m}_{\text{UK}}) \\
&= C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 + (\mathbf{X}'\boldsymbol{\lambda})'\mathbf{m}_{\text{UK}} \\
&= C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 + \mathbf{m}_{\text{UK}}'\mathbf{x}_0 \quad \blacksquare
\end{aligned}
$$

#### Theorem 6.3.2 (Two-Stage Decomposition of Universal Kriging)
The Universal Kriging predictor and its prediction error variance admit exact closed-form two-stage representations:
1. **Predictor Decomposition:**
   $$
   \widehat{Y}_{\text{UK}}(\mathbf{s}_0) = \mathbf{x}_0'\widehat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{c}_0'\mathbf{V}^{-1}\left(\mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right)
   $$
2. **Variance Decomposition:**
   $$
   \sigma_{\text{UK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right)'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right)
   $$
3. **Multiplier Vector:**
   $$
   \mathbf{m}_{\text{UK}} = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right)
   $$
4. **Exact Interpolation:** If nugget $\tau^2 = 0$ and $\mathbf{s}_0 = \mathbf{s}_i$, then $\widehat{Y}_{\text{UK}}(\mathbf{s}_i) = Y(\mathbf{s}_i)$ and $\sigma_{\text{UK}}^2(\mathbf{s}_i) = 0$.

*Proof.*
From the first block equation $\mathbf{V}\boldsymbol{\lambda} - \mathbf{X}\mathbf{m}_{\text{UK}} = \mathbf{c}_0$, solve for $\boldsymbol{\lambda}$:
$$
\boldsymbol{\lambda} = \mathbf{V}^{-1}\mathbf{c}_0 + \mathbf{V}^{-1}\mathbf{X}\mathbf{m}_{\text{UK}}
$$
Substitute into the second equation $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{x}_0$:
$$
\mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0 + (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})\mathbf{m}_{\text{UK}} = \mathbf{x}_0 \implies (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})\mathbf{m}_{\text{UK}} = \mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0
$$
Since $\mathbf{X}$ has full column rank $p$ and $\mathbf{V}$ is positive definite, $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ is non-singular, giving:
$$
\mathbf{m}_{\text{UK}} = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right)
$$
Substituting $\mathbf{m}_{\text{UK}}$ back into the expression for $\boldsymbol{\lambda}$:
$$
\boldsymbol{\lambda}' = \mathbf{c}_0'\mathbf{V}^{-1} + \left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right)'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}
$$
Multiplying by $\mathbf{Y}$ yields:
$$
\begin{aligned}
\widehat{Y}_{\text{UK}}(\mathbf{s}_0) = \boldsymbol{\lambda}'\mathbf{Y} &= \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{Y} + \left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right)'\widehat{\boldsymbol{\beta}}_{\text{GLS}} \\
&= \mathbf{x}_0'\widehat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{GLS}})
\end{aligned}
$$
Now substitute $\boldsymbol{\lambda}$ into the variance expression:
$$
\begin{aligned}
\sigma_{\text{UK}}^2(\mathbf{s}_0) &= C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 + \mathbf{m}_{\text{UK}}'\mathbf{x}_0 \\
&= C(\mathbf{0}) - \left(\mathbf{V}^{-1}\mathbf{c}_0 + \mathbf{V}^{-1}\mathbf{X}\mathbf{m}_{\text{UK}}\right)'\mathbf{c}_0 + \mathbf{m}_{\text{UK}}'\mathbf{x}_0 \\
&= C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0 + \mathbf{m}_{\text{UK}}'(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0) \\
&= \sigma_{\text{SK}}^2(\mathbf{s}_0) + \left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right)'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\left(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0\right) \quad \blacksquare
\end{aligned}
$$

```
Universal Kriging Prediction Error Variance Breakdown:
========================================================================================
 σ_UK^2(s_0)  =       σ_SK^2(s_0)       +   r_0' * (X' V^-1 X)^-1 * r_0
                   ──────────────────       ─────────────────────────────
                   Residual Variance             Drift Variance Penalty
              (Uncertainty if β was known)     (Penalty for estimating β)
========================================================================================
```

#### 6.3.2 Algorithmic Translation: Universal Kriging Predictor

The following dependency-free Python implementation solves the augmented Universal Kriging saddle-point system via Gaussian elimination with partial pivoting:

```python
def universal_kriging_predict(sample_coords, sample_vals, target_coord, 
                              basis_functions, cov_kernel):
    """
    Computes Universal Kriging prediction and prediction variance.

    Parameters:
    -----------
    sample_coords : list of tuples/lists of float, shape (n, d)
        Coordinates s_1, ..., s_n.
    sample_vals : list of float, shape (n,)
        Observations Y(s_1), ..., Y(s_n).
    target_coord : tuple/list of float, shape (d,)
        Target site s_0.
    basis_functions : list of callable
        p drift basis functions [f_1(s), ..., f_p(s)].
    cov_kernel : callable
        Covariance function C(h).

    Returns:
    --------
    y_hat : float
        Universal Kriging prediction Y_UK(s_0).
    uk_var : float
        Universal Kriging prediction variance sigma_UK^2(s_0).
    weights : list of float
        Universal Kriging weights lambda_1, ..., lambda_n.
    m_uk : list of float
        Lagrange multiplier vector m_UK.
    """
    n = len(sample_coords)
    p = len(basis_functions)
    d = len(target_coord)
    m = n + p

    # Step 1: Assemble the (n+p) x (n+p) augmented matrix
    # [ V   X ]
    # [ X'  0 ]
    A = [[0.0] * m for _ in range(m)]

    for i in range(n):
        for j in range(n):
            dist = sum((sample_coords[i][k] - sample_coords[j][k]) ** 2 for k in range(d)) ** 0.5
            A[i][j] = cov_kernel(dist)

    for i in range(n):
        for j in range(p):
            f_val = basis_functions[j](sample_coords[i])
            A[i][n + j] = f_val
            A[n + j][i] = f_val

    # Step 2: Assemble the (n+p) x 1 right-hand side vector
    # [ c_0 ]
    # [ x_0 ]
    b = [0.0] * m
    for i in range(n):
        dist = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(d)) ** 0.5
        b[i] = cov_kernel(dist)
    for j in range(p):
        b[n + j] = basis_functions[j](target_coord)

    # Step 3: Solve the linear system A * [lambda, -m_uk]' = b
    aug = [A[r][:] + [b[r]] for r in range(m)]

    for col in range(m):
        max_row = col
        max_val = abs(aug[col][col])
        for r in range(col + 1, m):
            if abs(aug[r][col]) > max_val:
                max_val = abs(aug[r][col])
                max_row = r
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for c in range(col, m + 1):
            aug[col][c] /= pivot

        for r in range(m):
            if r != col:
                factor = aug[r][col]
                for c in range(col, m + 1):
                    aug[r][c] -= factor * aug[col][c]

    solution = [aug[r][m] for r in range(m)]
    weights = solution[:n]
    m_uk = [-solution[n + j] for j in range(p)]

    # Step 4: Evaluate prediction
    y_hat = sum(weights[i] * sample_vals[i] for i in range(n))

    # Step 5: Evaluate prediction variance
    c0 = cov_kernel(0.0)
    term1 = sum(weights[i] * b[i] for i in range(n))
    term2 = sum(m_uk[j] * b[n + j] for j in range(p))
    uk_var = max(0.0, c0 - term1 + term2)

    return y_hat, uk_var, weights, m_uk
```

#### 6.3.3 Intuition, Edge Cases & Diagnostic Verification

- **The Price of Estimating Drift:** The variance decomposition proves that $\sigma_{\text{UK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \mathbf{r}_0'(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{r}_0$. The drift penalty vanishes if and only if $\mathbf{r}_0 = \mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{c}_0 = \mathbf{0}$, which occurs when the target is so close to sample points that local residuals completely inform the local trend.
- **Polynomial Extrapolation Growth (Not Runge's Phenomenon):** Runge's phenomenon strictly refers to high-degree polynomial *interpolation* oscillating at equally spaced nodes. What occurs in spatial polynomial regression is **polynomial extrapolation runaway**: when a quadratic ($p = 6$) or cubic ($p = 10$) coordinate drift is fitted, predicting at an unsampled location $\mathbf{s}_0$ outside the convex hull of the data causes the covariate vector $\mathbf{x}_0$ to grow as $\|\mathbf{s}_0\|^2$ or $\|\mathbf{s}_0\|^3$. Because the drift variance penalty is quadratic in $\mathbf{x}_0$, the prediction variance explodes as $\|\mathbf{s}_0\|^4$ for quadratic drift and $\|\mathbf{s}_0\|^6$ for cubic drift. For this reason, coordinate polynomial trends beyond linear ($p = 1 + d$) are rarely recommended in practice.
- **Regression Kriging as a Computational Shortcut:** Many software packages compute Universal Kriging via **Regression Kriging (RK)**: fit OLS to obtain $\widehat{\boldsymbol{\beta}}_{\text{OLS}}$, compute residuals $\mathbf{e}$, fit an empirical variogram to $\mathbf{e}$, and add Ordinary or Simple Kriging of $\mathbf{e}$ back to $\mathbf{x}_0'\widehat{\boldsymbol{\beta}}_{\text{OLS}}$. When the spatial correlation range is small relative to the domain ($\phi \ll \operatorname{diam}(D)$), $\widehat{\boldsymbol{\beta}}_{\text{OLS}} \approx \widehat{\boldsymbol{\beta}}_{\text{GLS}}$, making RK an accurate and fast approximation. However, if $\phi$ is large, RK inherits the OLS projection bias discussed in Section 6.1.

---

### 6.4 Kriging with External Drift (KED)

#### 6.4.1 The Mathematical Anchor: Auxiliary Physical Covariates

In classical multiple linear regression, an environmental response (e.g., local air temperature $Y$) is modeled as a linear function of exogenous physical covariates (such as elevation $q_1$ and proximity to coast $q_2$): $Y_i = \beta_0 + \beta_1 q_{1i} + \beta_2 q_{2i} + \varepsilon_i$. However, classical regression assumes that after conditioning on these covariates, the residual errors are mutually independent. In reality, meteorological and geophysical processes retain substantial spatial autocorrelation even after conditioning on terrain elevation. Kriging with External Drift (KED) unifies multiple regression with spatial interpolation by embedding deterministic exogenous physical covariates directly into the spatial drift $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$, while simultaneously modeling the spatial autocorrelation of the regression residuals via a continuous covariance kernel $C(\mathbf{h})$.

Mathematically, KED is **Universal Kriging where the design columns $\mathbf{X}$ are external physical covariates rather than spatial coordinate polynomials**.

#### Definition 6.4.1 (External Drift Model)
Let $\{Y(\mathbf{s}) : \mathbf{s} \in D\}$ be a target variable (e.g., precipitation, sub-surface porosity). Let $\{q_1(\mathbf{s}), \dots, q_m(\mathbf{s})\}$ be $m$ deterministic auxiliary spatial fields (e.g., digital elevation model, seismic acoustic impedance). The KED model specifies:
$$
Y(\mathbf{s}) = \beta_0 + \sum_{k=1}^m \beta_k q_k(\mathbf{s}) + \varepsilon(\mathbf{s}) = \mathbf{x}(\mathbf{s})' \boldsymbol{\beta} + \varepsilon(\mathbf{s})
$$
where $\mathbf{x}(\mathbf{s}) = (1, q_1(\mathbf{s}), \dots, q_m(\mathbf{s}))' \in \mathbb{R}^{m+1}$, and $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_m)' \in \mathbb{R}^{m+1}$.

#### Structural Regularity Conditions for KED:
1. **Exhaustive Availability:** The auxiliary fields $q_k(\mathbf{s})$ must be known at all sample locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ and at every target prediction site $\mathbf{s}_0 \in D$.
2. **Linear Independence and Non-Collinearity:** The augmented design matrix $\mathbf{X} \in \mathbb{R}^{n \times (m+1)}$ must satisfy $\operatorname{rank}(\mathbf{X}) = m+1 \le n$.
3. **Linearity in the Coefficients:** The conditional expectation $\operatorname{E}[Y(\mathbf{s}) \mid q_1(\mathbf{s}), \dots, q_m(\mathbf{s})]$ must be linear in the parameters $\boldsymbol{\beta}$. If the physical relationship between $Y$ and $q$ is non-linear (e.g., temperature inversion or logarithmic decay), non-linear transformations are accommodated simply by including transformed covariate columns (e.g., $q^2$, $\log q$) directly in $\mathbf{X}$.
4. **Distinction from Cokriging:** In KED, the auxiliary field $q(\mathbf{s})$ is treated as a deterministic external trend without cross-covariance modeling. In Cokriging (Chapter 7), the secondary variable is modeled as a joint stochastic random field with spatial cross-covariances.

#### 6.4.2 Algorithmic Translation: KED with Diagnostic Fallback

Because KED mathematically coincides with Universal Kriging with design matrix $\mathbf{X} = [\mathbf{1}, \mathbf{q}_1, \dots, \mathbf{q}_m]$, its computation can directly reuse `universal_kriging_predict`. The critical algorithmic component is detecting **local covariate variance deficiency** when evaluating in moving spatial neighborhoods:

```python
def kriging_with_external_drift_predict(sample_coords, sample_vals, sample_covariates,
                                       target_coord, target_covariates, cov_kernel,
                                       relative_variance_threshold=1e-4):
    """
    Performs Kriging with External Drift (KED) with diagnostic verification
    and an automatic fallback to Ordinary Kriging if local covariate variance is deficient.

    Parameters:
    -----------
    sample_coords : list of tuples of float
        Observation coordinates (s_1, ..., s_n).
    sample_vals : list of float
        Observed primary responses Y(s_1), ..., Y(s_n).
    sample_covariates : list of list of float, shape (n, m)
        Auxiliary physical covariates q_1(s_i), ..., q_m(s_i) at sample sites.
    target_coord : tuple of float
        Prediction coordinate s_0.
    target_covariates : list of float, shape (m,)
        Auxiliary covariates q_1(s_0), ..., q_m(s_0) at target site.
    cov_kernel : callable
        Covariance function C(h).
    relative_variance_threshold : float
        Threshold ratio of local to global covariate variance below which KED falls back to OK.

    Returns:
    --------
    y_hat : float
        KED or OK prediction at s_0.
    var_pred : float
        Prediction error variance.
    mode_used : str
        'KED' or 'OK_FALLBACK'.
    """
    n = len(sample_coords)
    m = len(target_covariates)

    # Check relative covariate variance for each covariate
    revert_to_ok = False
    for k in range(m):
        q_vals = [sample_covariates[i][k] for i in range(n)]
        mean_q = sum(q_vals) / n
        var_q = sum((q - mean_q) ** 2 for q in q_vals) / max(1, n - 1)
        
        # If covariate variance is negligible relative to mean scale, drop to OK
        scale_sq = max(1e-12, mean_q ** 2)
        if (var_q / scale_sq) < relative_variance_threshold:
            revert_to_ok = True
            break

    if revert_to_ok:
        # Fall back to Ordinary Kriging (p = 1, constant drift)
        basis_funcs = [lambda s: 1.0]
        mode_used = 'OK_FALLBACK'
    else:
        # Assemble KED basis functions: [1, q_1(s), ..., q_m(s)]
        # Map sample sites to covariate lookups
        coord_to_cov = {sample_coords[i]: sample_covariates[i] for i in range(n)}
        
        def make_cov_func(idx):
            return lambda s: target_covariates[idx] if s == target_coord else coord_to_cov[s][idx]
            
        basis_funcs = [lambda s: 1.0] + [make_cov_func(k) for k in range(m)]
        mode_used = 'KED'

    y_hat, var_pred, weights, m_uk = universal_kriging_predict(
        sample_coords, sample_vals, target_coord, basis_funcs, cov_kernel
    )

    return y_hat, var_pred, mode_used
```

#### 6.4.3 Intuition, Edge Cases & Diagnostic Verification

- **Local Collinearity vs. Covariate Extrapolation:** In KED, numerical instability arises in two distinct scenarios:
  1. *Flat Local Covariates:* If all sample points in a local neighborhood share identical elevation (e.g., a flat lake or valley floor), $q(\mathbf{s}_i) = \bar{q}$, the covariate column becomes collinear with the intercept column of ones. The design matrix drops rank ($\operatorname{rank}(\mathbf{X}) < m+1$), $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ becomes singular, and KED fails.
  2. *Covariate Extrapolation:* Even if $\mathbf{X}$ is well-conditioned at the data sites, if the target site $\mathbf{s}_0$ possesses an auxiliary value far outside the range of sampled covariates ($q(\mathbf{s}_0) \gg \max_i q(\mathbf{s}_i)$ or $q(\mathbf{s}_0) \ll \min_i q(\mathbf{s}_i)$), the drift penalty $\mathbf{r}_0'(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{r}_0$ explodes quadratically with the extrapolation distance in covariate space, leading to wildly unrealistic predictions.

---



### 6.5 Dual Kriging and Spline Equivalence

#### 6.5.1 The Mathematical Anchor: From Ridge Regression Push-Through to Dual Kriging

In classical regularized linear regression (Ridge regression), we seek a coefficient vector $oldsymbol{eta} \in \mathbb{R}^p$ minimizing the penalized sum of squares:
$$
S(oldsymbol{eta}) = (\mathbf{y} - \mathbf{X}oldsymbol{eta})'(\mathbf{y} - \mathbf{X}oldsymbol{eta}) + \lambda oldsymbol{eta}'oldsymbol{eta}
$$
The standard primal solution solves a $p 	imes p$ system in parameter space:
$$
\widehat{oldsymbol{eta}} = \left(\mathbf{X}'\mathbf{X} + \lambda \mathbf{I}_p
ight)^{-1} \mathbf{X}'\mathbf{y}
$$
However, using the fundamental **push-through matrix identity**:
$$
\left(\mathbf{X}'\mathbf{X} + \lambda \mathbf{I}_p
ight)^{-1} \mathbf{X}' = \mathbf{X}' \left(\mathbf{X}\mathbf{X}' + \lambda \mathbf{I}_n
ight)^{-1}
$$
(which holds for any $\mathbf{X} \in \mathbb{R}^{n 	imes p}$ and $\lambda > 0$, verified by pre-multiplying by $(\mathbf{X}'\mathbf{X} + \lambda \mathbf{I}_p)$ and post-multiplying by $(\mathbf{X}\mathbf{X}' + \lambda \mathbf{I}_n)$), this estimator can be expressed in dual form:
$$
\widehat{oldsymbol{eta}} = \mathbf{X}' \mathbf{b}, \quad 	ext{where} \quad \mathbf{b} = \left(\mathbf{X}\mathbf{X}' + \lambda \mathbf{I}_n
ight)^{-1} \mathbf{y} \in \mathbb{R}^n
$$
In this dual representation (well-known in machine learning as Kernel Ridge Regression or Gaussian Process Regression mean estimation), the $n 	imes n$ system for $\mathbf{b}$ is solved **once** in data space. The prediction at any target covariate vector $\mathbf{x}_0$ is then simply:
$$
\widehat{y}_0 = \mathbf{x}_0' \widehat{oldsymbol{eta}} = \mathbf{x}_0' \mathbf{X}' \mathbf{b} = \sum_{i=1}^n b_i \langle \mathbf{x}_0, \mathbf{x}_i 
angle
$$
In spatial statistics, the primal Universal Kriging system presents an analogous workflow: for each unsampled prediction site $\mathbf{s}_0$, an $(n+p) 	imes (n+p)$ linear system must be solved for the location-dependent weights $oldsymbol{\lambda}(\mathbf{s}_0)$ and multipliers $-\mathbf{m}_{	ext{UK}}(\mathbf{s}_0)$. If an analyst factors the bordered matrix once via Cholesky/Bunch-Kaufman, evaluating at $M$ grid points requires $M$ triangular back-solves ($O(M \cdot n^2)$ operations).

The **Dual Kriging formulation** flips this perspective: instead of solving for prediction weights at each target site, we solve the system **once** for a fixed vector of data weights $\mathbf{b} \in \mathbb{R}^n$ and drift parameters $\mathbf{g} = \widehat{oldsymbol{eta}}_{	ext{GLS}} \in \mathbb{R}^p$. The continuous interpolating surface is then evaluated everywhere via rapid inner products in $O(n)$ operations per target site.

#### Theorem 6.5.1 (Dual Kriging Representation)
Let $\mathbf{Y} = \mathbf{X} oldsymbol{eta} + oldsymbol{arepsilon}$ be the spatial linear model on sample sites $\mathbf{s}_1, \dots, \mathbf{s}_n$ with $\operatorname{Var}(oldsymbol{arepsilon}) = \mathbf{V}$.
1. The Universal Kriging interpolating surface can be expressed identically as:
   $$
   \widehat{Y}(\mathbf{s}) = \sum_{j=1}^p g_j f_j(\mathbf{s}) + \sum_{i=1}^n b_i C(\mathbf{s} - \mathbf{s}_i) = \mathbf{x}(\mathbf{s})' \mathbf{g} + \mathbf{v}(\mathbf{s})' \mathbf{b}
   $$
   where $\mathbf{g} = (g_1, \dots, g_p)' \in \mathbb{R}^p$ and $\mathbf{b} = (b_1, \dots, b_n)' \in \mathbb{R}^n$ are deterministic vectors that are **completely independent of $\mathbf{s}$**.
2. The coefficient vectors $\mathbf{b}$ and $\mathbf{g}$ are the unique solution to the **Dual Kriging linear system**:
   $$
   egin{bmatrix} \mathbf{V} & \mathbf{X} \ \mathbf{X}' & \mathbf{0} \end{bmatrix} egin{bmatrix} \mathbf{b} \ \mathbf{g} \end{bmatrix} = egin{bmatrix} \mathbf{Y} \ \mathbf{0} \end{bmatrix}
   $$
3. The coefficients satisfy the explicit relations:
   $$
   \mathbf{g} = \widehat{oldsymbol{eta}}_{	ext{GLS}} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}
ight)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}
   $$
   $$
   \mathbf{b} = \mathbf{V}^{-1} \left(\mathbf{Y} - \mathbf{X} \widehat{oldsymbol{eta}}_{	ext{GLS}}
ight)
   $$
   and satisfy the discrete orthogonality condition $\mathbf{X}' \mathbf{b} = \mathbf{0}$.

*Proof.*
By Theorem 6.3.2, the primal UK predictor at an arbitrary location $\mathbf{s} \in D$ is:
$$
\widehat{Y}(\mathbf{s}) = \mathbf{x}(\mathbf{s})' \widehat{oldsymbol{eta}}_{	ext{GLS}} + \mathbf{v}(\mathbf{s})' \mathbf{V}^{-1} \left(\mathbf{Y} - \mathbf{X} \widehat{oldsymbol{eta}}_{	ext{GLS}}
ight)
$$
Define the vectors:
$$
\mathbf{g} = \widehat{oldsymbol{eta}}_{	ext{GLS}} \in \mathbb{R}^p
$$
$$
\mathbf{b} = \mathbf{V}^{-1} \left(\mathbf{Y} - \mathbf{X} \widehat{oldsymbol{eta}}_{	ext{GLS}}
ight) = \mathbf{V}^{-1} (\mathbf{Y} - \mathbf{X}\mathbf{g}) \in \mathbb{R}^n
$$
Substituting $\mathbf{g}$ and $\mathbf{b}$ directly into the equation gives:
$$
\widehat{Y}(\mathbf{s}) = \mathbf{x}(\mathbf{s})' \mathbf{g} + \mathbf{v}(\mathbf{s})' \mathbf{b}
$$
Now verify that $(\mathbf{b}', \mathbf{g}')'$ satisfies the block system. From the definition of $\mathbf{b}$:
$$
\mathbf{V} \mathbf{b} = \mathbf{Y} - \mathbf{X}\mathbf{g} \implies \mathbf{V} \mathbf{b} + \mathbf{X}\mathbf{g} = \mathbf{Y}
$$
This establishes the top block equation.
Pre-multiply this equation by $\mathbf{X}'\mathbf{V}^{-1}$:
$$
\mathbf{X}' \mathbf{V}^{-1} (\mathbf{V} \mathbf{b} + \mathbf{X}\mathbf{g}) = \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y} \implies \mathbf{X}' \mathbf{b} + \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}
ight) \mathbf{g} = \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}
$$
Since $\mathbf{g} = \widehat{oldsymbol{eta}}_{	ext{GLS}} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Y}$, we have $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})\mathbf{g} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{Y}$. Subtracting this from both sides yields:
$$
\mathbf{X}' \mathbf{b} = \mathbf{0}
$$
Assembling these equations establishes the dual block system:
$$
egin{bmatrix} \mathbf{V} & \mathbf{X} \ \mathbf{X}' & \mathbf{0} \end{bmatrix} egin{bmatrix} \mathbf{b} \ \mathbf{g} \end{bmatrix} = egin{bmatrix} \mathbf{Y} \ \mathbf{0} \end{bmatrix}
$$
This completes the proof. $lacksquare$

#### 6.5.2 Equivalence with Smoothing Splines

The Dual Kriging representation reveals an exact equivalence between spatial linear models and smoothing/interpolating splines.

Consider the 1D natural cubic smoothing spline problem: given noisy data $(s_i, Y_i)$ for $i=1,\dots,n$, find a function $f(s)$ minimizing:
$$
\sum_{i=1}^n (Y_i - f(s_i))^2 + \lambda \int [f''(s)]^2 \, ds
$$
When $\lambda 	o 0$, $f(s)$ becomes an exact interpolating cubic spline. The solution has the explicit representation:
$$
f(s) = g_0 + g_1 s + \sum_{i=1}^n b_i |s - s_i|^3
$$
subject to the boundary constraints $\sum_{i=1}^n b_i = 0$ and $\sum_{i=1}^n b_i s_i = 0$.

Notice the mathematical structure:
1. The polynomial drift $\mathbf{x}(s)'\mathbf{g} = g_0 + g_1 s$ spans the null space of the second derivative penalty operator ($f''(s) = 0$).
2. The radial basis function $|s - s_i|^3$ is the 1D generalized covariance kernel corresponding to the roughness penalty $\int [f'']^2$.
3. The boundary constraints $\sum b_i = 0, \sum b_i s_i = 0$ are precisely the dual orthogonality condition $\mathbf{X}'\mathbf{b} = \mathbf{0}$.
4. If observations are noisy with measurement error $	au^2 > 0$, the nugget variance $	au^2 \mathbf{I}$ corresponds precisely to the smoothing penalty parameter $\lambda > 0$. When $	au^2 = 0$, the spline interpolates the data exactly.

In two dimensions, minimizing the bending energy $\iint \left[ (\partial^2 f/\partial s_1^2)^2 + 2(\partial^2 f/\partial s_1 \partial s_2)^2 + (\partial^2 f/\partial s_2^2)^2 
ight] ds_1 ds_2$ similarly yields the classical **thin-plate spline** $f(\mathbf{s}) = g_0 + g_1 s_1 + g_2 s_2 + \sum_{i=1}^n b_i \|\mathbf{s} - \mathbf{s}_i\|^2 \log \|\mathbf{s} - \mathbf{s}_i\|$, which is dual universal kriging with generalized covariance $K(h) = h^2 \log h$ and linear drift (Wahba, 1990; Cressie, 1993).

#### 6.5.3 Algorithmic Translation: Dual Kriging Solver and Fast Evaluator

The dual kriging algorithm solves the bordered linear system once to obtain $\mathbf{b}$ and $\mathbf{g}$, and then evaluates predictions across an arbitrary target grid in $O(n)$ operations per grid point.

> [!NOTE]
> **Key Limitation:** Dual kriging is an exceptionally fast way to compute point predictions $\widehat{Y}(\mathbf{s}_0)$, but **it does not compute the kriging prediction variance** $\sigma_{	ext{UK}}^2(\mathbf{s}_0)$. Computing $\sigma_{	ext{UK}}^2(\mathbf{s}_0)$ still requires the location-dependent quadratic form $\mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0$ and $(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)'(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)$. Thus, dual kriging is ideal for rapid visualization, mapping, and rendering where uncertainty maps are not required.

```python
def dual_kriging_fit(sample_coords, sample_vals, basis_functions, cov_kernel):
    """
    Fit Dual Kriging coefficients (b, g) by solving the dual linear system.

    Parameters:
    -----------
    sample_coords : list of tuples/lists of float, shape (n, d)
        Coordinates s_1, ..., s_n.
    sample_vals : list of float, shape (n,)
        Observations Y(s_1), ..., Y(s_n).
    basis_functions : list of callable
        p drift basis functions [f_1(s), ..., f_p(s)].
    cov_kernel : callable
        Covariance function C(h).

    Returns:
    --------
    b_weights : list of float, shape (n,)
        Dual data weights.
    g_weights : list of float, shape (p,)
        Dual drift coefficients (identical to beta_GLS).
    """
    n = len(sample_coords)
    p = len(basis_functions)
    dim = len(sample_coords[0])
    m = n + p

    # Step 1: Assemble the (n+p) x (n+p) dual matrix
    # [ V   X ]
    # [ X'  0 ]
    A = [[0.0] * m for _ in range(m)]

    for i in range(n):
        for j in range(n):
            dist = sum((sample_coords[i][k] - sample_coords[j][k]) ** 2 for k in range(dim)) ** 0.5
            A[i][j] = cov_kernel(dist)

    for i in range(n):
        for j in range(p):
            f_val = basis_functions[j](sample_coords[i])
            A[i][n + j] = f_val
            A[n + j][i] = f_val

    # Step 2: Assemble the right-hand side [Y, 0]'
    rhs = sample_vals[:] + [0.0] * p

    # Step 3: Solve the dual linear system A * [b, g]' = rhs
    aug = [A[r][:] + [rhs[r]] for r in range(m)]

    for col in range(m):
        max_row = col
        max_val = abs(aug[col][col])
        for r in range(col + 1, m):
            if abs(aug[r][col]) > max_val:
                max_val = abs(aug[r][col])
                max_row = r
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for c in range(col, m + 1):
            aug[col][c] /= pivot

        for r in range(m):
            if r != col:
                factor = aug[r][col]
                for c in range(col, m + 1):
                    aug[r][c] -= factor * aug[col][c]

    solution = [aug[r][m] for r in range(m)]
    b_weights = solution[:n]
    g_weights = solution[n:]

    return b_weights, g_weights


def dual_kriging_eval(target_coord, sample_coords, b_weights, g_weights, 
                      basis_functions, cov_kernel):
    """
    Evaluates the Dual Kriging interpolating surface at a single target location.

    Parameters:
    -----------
    target_coord : tuple/list of float, shape (d,)
        Target site s.
    sample_coords : list of tuples/lists of float, shape (n, d)
        Coordinates s_1, ..., s_n.
    b_weights : list of float, shape (n,)
        Precomputed dual data weights.
    g_weights : list of float, shape (p,)
        Precomputed dual drift coefficients.
    basis_functions : list of callable
        p drift basis functions.
    cov_kernel : callable
        Covariance function C(h).

    Returns:
    --------
    y_pred : float
        Interpolated value at target site.
    """
    n = len(sample_coords)
    p = len(basis_functions)
    dim = len(target_coord)

    # Trend component: x(s)' * g
    trend_comp = sum(g_weights[j] * basis_functions[j](target_coord) for j in range(p))

    # Stochastic / kernel component: v(s)' * b = sum b_i * C(s - s_i)
    kernel_comp = 0.0
    for i in range(n):
        dist = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(dim)) ** 0.5
        kernel_comp += b_weights[i] * cov_kernel(dist)

    return trend_comp + kernel_comp
```

#### 6.5.4 Intuition, Edge Cases & Diagnostic Verification

- **Computational Complexity Baseline:** If an analyst refactors the augmented matrix from scratch at each of $M$ target grid sites, primal kriging requires $O(M \cdot n^3)$ work. Even when $\mathbf{V}$ is factored once via Cholesky decomposition ($O(n^3)$ setup), primal kriging requires solving $\mathbf{V}\mathbf{w} = \mathbf{v}_0$ for each site, taking $O(n^2)$ work per site ($O(M \cdot n^2)$ total). Dual kriging computes $\mathbf{b}$ and $\mathbf{g}$ once in $O(n^3)$ setup, and each grid evaluation is a simple inner product taking $O(n)$ work ($O(M \cdot n)$ total). For $n = 1{,}000$ and $M = 1{,}000{,}000$, dual evaluation takes $pprox 10^9$ FLOPs versus $pprox 10^{12}$ FLOPs for primal back-solves—a $1{,}000	imes$ speedup.
- **Physical Interpretation of Dual Weights:** In the spline interpretation, the coefficients $b_i$ represent discrete point forces (loads) applied at sample points $\mathbf{s}_i$, while $\mathbf{g}$ represents rigid body translations and rotations. The condition $\sum b_i = 0$ enforces vertical static force balance, while $\sum b_i s_{i,k} = 0$ enforces static moment balance.

---

### 6.6 Intrinsic Random Functions of Order $k$ (IRF-$k$)

#### 6.6.1 The Mathematical Anchor: From ARIMA Differencing to Spatial Authorized Linear Combinations

In classical Box–Jenkins time-series analysis ($\operatorname{ARIMA}(p, d, q)$), when an observed sequence $\{Y_t\}$ exhibits a non-stationary polynomial trend, its global mean is not constant, and its sample autocovariance function fails to decay. Rather than estimating trend coefficients directly (which introduces finite-sample parameter estimation variance and collinearity), the classical strategy applies the difference operator:
$$

abla Y_t = (1 - B) Y_t = Y_t - Y_{t-1}
$$
For a linear trend ($d=1$), first-order differencing annihilates the slope: if $\mu_t = eta_0 + eta_1 t$, then $
abla \mu_t = eta_1$, which is constant. For a quadratic trend ($d=2$), second-order differencing annihilates both intercept and curvature:
$$

abla^2 Y_t = Y_t - 2Y_{t-1} + Y_{t-2}
$$
Notice the algebraic structure of the filter weights: for $d=1$, weights $(1, -1)$ satisfy $\sum w_i = 0$; for $d=2$, weights $(1, -2, 1)$ satisfy $\sum w_i = 0$ and $\sum w_i t_i = 1(t) - 2(t-1) + 1(t-2) = 0$. The difference operator filters out polynomials without ever estimating their parameters.

In two- or three-dimensional spatial analysis, sample locations $\mathbf{s}_i \in \mathbb{R}^d$ are irregularly scattered rather than evenly spaced along an integer time index. Sequential lag differencing is impossible. Georges Matheron's theory of **Intrinsic Random Functions of order $k$ (IRF-$k$)** solves this challenge by generalizing the differencing operator to spatial point configurations via **Authorized Linear Combinations (ALC-$k$)**: linear combinations $\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$ whose weights $oldsymbol{\lambda}$ are algebraically orthogonal to all spatial polynomials of degree up to $k$.

> [!NOTE]
> **Clarifying Origin Invariance:** Universal Kriging predictions are already invariant under spatial origin translation because translating coordinates $\mathbf{s} \mapsto \mathbf{s} + \mathbf{c}$ simply applies an invertible linear transformation to the polynomial basis $\mathbf{x}(\mathbf{s})$, leaving the column span of $\mathbf{X}$ unchanged. The true theoretical and practical breakthrough of IRF-$k$ is that **it allows the spatial dependence to be modeled by generalized covariance functions $K(h)$ that do not need to be bounded or possess a finite sill** (such as linear, power, or spline kernels $|h|^{2k+1}$).

#### Definition 6.6.1 (Authorized Linear Combinations of Order $k$)
Let $\mathcal{P}_k$ denote the vector space of spatial polynomials in $\mathbf{s} \in \mathbb{R}^d$ of degree at most $k$. A linear combination $\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$ is an **Authorized Linear Combination of order $k$ (ALC-$k$)** if its coefficient vector $oldsymbol{\lambda} \in \mathbb{R}^n$ satisfies:
$$
\sum_{i=1}^n \lambda_i p(\mathbf{s}_i) = 0, \quad orall p \in \mathcal{P}_k
$$
In matrix notation, if $\mathbf{X}_k \in \mathbb{R}^{n 	imes p_k}$ is the design matrix whose columns are the monomials spanning $\mathcal{P}_k$, then:
$$
\mathbf{X}_k' oldsymbol{\lambda} = \mathbf{0}
$$

#### Definition 6.6.2 (Intrinsic Random Function of Order $k$)
A random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is an **Intrinsic Random Function of order $k$ (IRF-$k$)** if for every ALC-$k$ with coefficients $oldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)'$ and sample locations $\mathbf{s}_1, \dots, \mathbf{s}_n$, the linear combination $\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$ is second-order stationary:
1. $\operatorname{E}\left[\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)
ight] = 0$
2. $\operatorname{Var}\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)
ight)$ depends only on the spatial configuration and relative vectors $\{\mathbf{s}_i - \mathbf{s}_j\}$.

#### Definition 6.6.3 (Generalized Covariance Kernel)
A continuous conditionally positive definite function $K(\mathbf{h})$ is a **generalized covariance of order $k$** for an IRF-$k$ if for every ALC-$k$ with coefficient vector $oldsymbol{\lambda}$ satisfying $\mathbf{X}_k' oldsymbol{\lambda} = \mathbf{0}$:
$$
\operatorname{Var}\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)
ight) = \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j K(\mathbf{s}_i - \mathbf{s}_j) = oldsymbol{\lambda}' \mathbf{K} oldsymbol{\lambda} \ge 0
$$
Unlike standard stationary covariances, $K(\mathbf{h})$:
- Need not satisfy $K(\mathbf{0}) \ge |K(\mathbf{h})|$ (it is not bounded).
- Can grow with lag distance: for instance, $K(h) = -b_1 |h| + b_3 |h|^3$ is a valid generalized covariance of order 1 in $\mathbb{R}^1$ and $\mathbb{R}^3$ provided $b_1 \ge 0$ and $b_3 \ge 0$.
- Nugget effects in generalized covariances are represented with indicator functions $	au^2 \mathbb{1}_{\{h=0\}}$ rather than formal Dirac distributions.

#### Theorem 6.6.1 (IRF-$k$ Kriging System)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an IRF-$k$ with generalized covariance $K(\mathbf{h})$. The Best Linear Unbiased Predictor $\widehat{Z}(\mathbf{s}_0) = oldsymbol{\lambda}'\mathbf{Z}$ of an unsampled value $Z(\mathbf{s}_0)$ satisfies the **identical linear system as Universal Kriging, with the generalized covariance $K$ replacing the stationary covariance $C$**:
$$
egin{bmatrix}
\mathbf{K} & \mathbf{X}_k \
\mathbf{X}_k' & \mathbf{0}
\end{bmatrix}
egin{bmatrix}
oldsymbol{\lambda} \
-\mathbf{m}_{	ext{IRF}}
\end{bmatrix}
=
egin{bmatrix}
\mathbf{k}_0 \
\mathbf{x}_k(\mathbf{s}_0)
\end{bmatrix}
$$
where:
- $\mathbf{K}_{ij} = K(\mathbf{s}_i - \mathbf{s}_j) + 	au^2 \mathbb{1}_{\{i=j\}}$,
- $\mathbf{k}_0 = (K(\mathbf{s}_1 - \mathbf{s}_0), \dots, K(\mathbf{s}_n - \mathbf{s}_0))'$,
- $\mathbf{X}_k$ contains monomial drift terms of degree $\le k$,
- $\mathbf{x}_k(\mathbf{s}_0)$ is the target drift vector at $\mathbf{s}_0$,
- $-\mathbf{m}_{	ext{IRF}} \in \mathbb{R}^{p_k}$ is the vector of Lagrange multipliers.

The IRF-$k$ prediction error variance is:
$$
\sigma_{	ext{IRF}}^2(\mathbf{s}_0) = K(\mathbf{0}) - oldsymbol{\lambda}' \mathbf{k}_0 + \mathbf{m}_{	ext{IRF}}' \mathbf{x}_k(\mathbf{s}_0)
$$
(which is strictly non-negative for all valid ALC-$k$ prediction configurations).

#### 6.6.2 Algorithmic Translation: IRF-$k$ Predictor

The following implementation solves the IRF-$k$ system using polynomial generalized covariances:

```python
def irf_k_predict(sample_coords, sample_vals, target_coord, k_order=1, 
                  gen_cov_params=None):
    """
    Performs spatial prediction under an Intrinsic Random Function of order k (IRF-k).

    Parameters:
    -----------
    sample_coords : list of tuples/lists of float, shape (n, d)
        Spatial coordinates of sample points.
    sample_vals : list of float, shape (n,)
        Observed values Z(s_1), ..., Z(s_n).
    target_coord : tuple/list of float, shape (d,)
        Coordinates of the target site s_0.
    k_order : int, default 1
        Order of intrinsic random function (0 = constant, 1 = linear, 2 = quadratic).
    gen_cov_params : dict, optional
        Parameters for the generalized covariance model.
        Default for k=1: {'c0': 0.0, 'b1': 1.0, 'b3': 0.0} (linear/spline kernel).

    Returns:
    --------
    dict:
        'prediction': float
        'variance': float
        'weights': list of float
    """
    if gen_cov_params is None:
        gen_cov_params = {'c0': 0.0, 'b1': 1.0, 'b3': 0.0}

    c0 = gen_cov_params.get('c0', 0.0)
    b1 = gen_cov_params.get('b1', 1.0)
    b3 = gen_cov_params.get('b3', 0.0)

    # Generalized covariance kernel K(h)
    def K(h):
        if h == 0.0:
            return c0
        return -b1 * h + b3 * (h ** 3)

    n = len(sample_coords)
    dim = len(sample_coords[0])

    # Build polynomial basis functions up to degree k
    # For k=0: [1]
    # For k=1 in d dimensions: [1, s_1, ..., s_d]
    basis_funcs = [lambda s: 1.0]
    if k_order >= 1:
        for d_idx in range(dim):
            basis_funcs.append(lambda s, idx=d_idx: s[idx])
    if k_order >= 2:
        for d_idx in range(dim):
            basis_funcs.append(lambda s, idx=d_idx: s[idx] ** 2)
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                basis_funcs.append(lambda s, i1=d1, i2=d2: s[i1] * s[i2])

    p = len(basis_funcs)
    m = n + p

    # Assemble bordered augmented IRF-k system
    A = [[0.0] * m for _ in range(m)]
    rhs = [0.0] * m

    for i in range(n):
        for j in range(n):
            dist = sum((sample_coords[i][d] - sample_coords[j][d]) ** 2 for d in range(dim)) ** 0.5
            A[i][j] = K(dist)
        # Target vector k_0
        d0 = sum((sample_coords[i][d] - target_coord[d]) ** 2 for d in range(dim)) ** 0.5
        rhs[i] = K(d0)

    for i in range(n):
        for j in range(p):
            f_val = basis_funcs[j](sample_coords[i])
            A[i][n + j] = f_val
            A[n + j][i] = f_val

    for j in range(p):
        rhs[n + j] = basis_funcs[j](target_coord)

    # Solve augmented linear system via Gaussian elimination with partial pivoting
    aug = [A[r][:] + [rhs[r]] for r in range(m)]

    for col in range(m):
        max_row = col
        max_val = abs(aug[col][col])
        for r in range(col + 1, m):
            if abs(aug[r][col]) > max_val:
                max_val = abs(aug[r][col])
                max_row = r
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError(f"IRF-{k_order} system is numerically singular at pivot {col}. Check point geometry.")

        for c in range(col, m + 1):
            aug[col][c] /= pivot

        for r in range(m):
            if r != col:
                factor = aug[r][col]
                for c in range(col, m + 1):
                    aug[r][c] -= factor * aug[col][c]

    solution = [aug[r][m] for r in range(m)]
    lambdas = solution[:n]
    neg_m = solution[n:]  # -m_IRF

    pred = sum(lambdas[i] * sample_vals[i] for i in range(n))
    k0_val = K(0.0)
    var = k0_val - sum(lambdas[i] * rhs[i] for i in range(n)) - sum(neg_m[j] * rhs[n + j] for j in range(p))

    return {
        'prediction': pred,
        'variance': max(0.0, var),
        'weights': lambdas
    }
```

#### 6.6.3 Intuition, Edge Cases & Diagnostic Verification

- **Connection to Variograms:** Notice that Ordinary Kriging is simply an IRF-0! The intrinsic hypothesis assumes stationarity of first-order increments $Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})$, which are ALC-0. The semivariogram $\gamma(\mathbf{h})$ is the generalized covariance corresponding to order $k=0$ with $K(\mathbf{h}) = -\gamma(\mathbf{h})$.
- **Direct Filtering:** In IRF-$k$, there is no need to estimate the trend $\widehat{oldsymbol{eta}}$ or subtract it to inspect residuals. The authorized linear combinations automatically filter the trend out of the prediction system, completely eliminating the circularity between trend fitting and covariance estimation.

---

### Key Ideas and Takeaways: Spatial Regression and Universal Kriging

| Concept | Mathematical Form | Master's-Level Interpretation | Practical Diagnostic Rule |
| :--- | :--- | :--- | :--- |
| **Trend & Error Decomposition** | $Y(\mathbf{s}) = \mathbf{x}(\mathbf{s})'oldsymbol{eta} + arepsilon(\mathbf{s})$ | Mean is space-dependent; residual error $arepsilon(\mathbf{s})$ has continuous spatial covariance $\mathbf{V}_{ij} = C(\mathbf{s}_i - \mathbf{s}_j) + 	au^2 \mathbb{1}_{\{i=j\}}$. | Center and scale coordinates $\mathbf{s}$; compute condition number $\kappa(\mathbf{X}'\mathbf{X})$. |
| **OLS Sill Attenuation** | $\operatorname{E}[\widehat{oldsymbol{arepsilon}}_{	ext{OLS}}'\widehat{oldsymbol{arepsilon}}_{	ext{OLS}}] = \operatorname{tr}(\mathbf{M}\mathbf{V})$ | Fitting OLS forces residuals orthogonal to $\mathbf{X}$, absorbing long-lag spatial variance and flattening empirical variograms. | Use iterative 7-step REML/GLS loop to estimate covariance and trend simultaneously. |
| **GLS Gauss–Markov Optimality** | $\widehat{oldsymbol{eta}}_{	ext{GLS}} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Y}$ | Restores BLUE status via Cholesky decorrelation $\mathbf{L}^{-1}\mathbf{Y}$. | Standard errors are 2x to 8x larger than naive OLS; efficiency variance gains are modest (1% to 28%). |
| **Universal Kriging (UK)** | $egin{bmatrix} \mathbf{V} & \mathbf{X} \ \mathbf{X}' & \mathbf{0} \end{bmatrix} egin{bmatrix} oldsymbol{\lambda} \ -\mathbf{m} \end{bmatrix} = egin{bmatrix} \mathbf{v}_0 \ \mathbf{x}_0 \end{bmatrix}$ | Predicts $Y(\mathbf{s}_0)$ using spatial data while enforcing polynomial unbiasedness $\mathbf{X}'oldsymbol{\lambda} = \mathbf{x}_0$. | UK variance $\sigma_{	ext{UK}}^2 = \sigma_{	ext{SK}}^2 + 	ext{drift penalty}$. Penalties explode outside the convex hull. |
| **Kriging with External Drift (KED)** | $\mathbf{x}(\mathbf{s}) = [1, q_1(\mathbf{s}), \dots, q_m(\mathbf{s})]'$ | UK where design columns are external physical covariates (elevation, temperature) rather than spatial coordinates. | If $\operatorname{Var}_{	ext{local}}(q) / \operatorname{Var}_{	ext{global}}(q) < 10^{-4}$, fall back to OK to avoid singular matrix inversion. |
| **Dual Kriging** | $\widehat{Y}(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\mathbf{g} + \mathbf{v}(\mathbf{s})'\mathbf{b}$ | Solves bordered system once in data space; evaluates grid predictions in $O(n)$ per target site. | Fast rendering equivalent to Gaussian Process / thin-plate splines; gives point predictions but **no variances**. |
| **Intrinsic Random Functions (IRF-$k$)** | $\mathbf{X}_k'oldsymbol{\lambda} = \mathbf{0}$, $\operatorname{Var}(oldsymbol{\lambda}'\mathbf{Z}) = oldsymbol{\lambda}'\mathbf{K}oldsymbol{\lambda}$ | Generalizes ARIMA differencing to spatial domains via Authorized Linear Combinations; admits unbounded generalized covariances. | Identical linear system to UK with $K$ replacing $C$; filters out polynomial drifts without estimating coefficients. |

---


### 6.7 Analytical Exercises with Step-by-Step Complete Solutions

#### Exercise 6.1 (Universal Kriging with Linear Drift in One Dimension)
Consider a non-stationary spatial process on the line:
$$
Y(s) = \beta_0 + \beta_1 s + \varepsilon(s), \quad s \in \mathbb{R}
$$
where $\operatorname{E}[\varepsilon(s)] = 0$ and $\operatorname{Cov}(\varepsilon(s), \varepsilon(s+h)) = C(h) = \exp(-|h|)$.
Suppose observations are obtained at three sites: $s_1 = -1$, $s_2 = 0$, $s_3 = 1$.
We wish to predict $Y(s_0)$ at the midpoint $s_0 = 0.5$.
- **(a)** Construct the design matrix $\mathbf{X}$ and the target drift vector $\mathbf{x}_0$.
- **(b)** Formulate the $5 \times 5$ Universal Kriging system in block matrix form.
- **(c)** Solve analytically for the kriging weight vector $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \lambda_3)'$ and the Lagrange multipliers $\boldsymbol{\mu}_{\text{UK}} = (\mu_0, \mu_1)'$.
- **(d)** Compute the Universal Kriging variance $\sigma_{\text{UK}}^2(s_0)$ and compare it with the Simple Kriging variance $\sigma_{\text{SK}}^2(s_0)$.

*Solution.*
- **(a)** The drift basis functions are $f_1(s) = 1$ and $f_2(s) = s$ ($p = 2$).
  At the sample locations $s_1 = -1, s_2 = 0, s_3 = 1$:
  $$
  \mathbf{X} = \begin{bmatrix} 1 & -1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}
  $$
  At the target site $s_0 = 0.5$:
  $$
  \mathbf{x}_0 = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix} \in \mathbb{R}^2
  $$

- **(b)** Let $\rho = e^{-1} \approx 0.367879$. The pairwise distances among sample points are:
  - $|s_1 - s_2| = 1 \implies C(1) = \rho$.
  - $|s_2 - s_3| = 1 \implies C(1) = \rho$.
  - $|s_1 - s_3| = 2 \implies C(2) = \rho^2$.
  The covariance matrix is:
  $$
  \mathbf{V} = \begin{bmatrix} 1 & \rho & \rho^2 \\ \rho & 1 & \rho \\ \rho^2 & \rho & 1 \end{bmatrix}
  $$
  The distances from $s_0 = 0.5$ to the sample sites are:
  - $|s_0 - s_1| = 1.5 \implies C(1.5) = e^{-1.5} = \rho^{3/2} \approx 0.223130$.
  - $|s_0 - s_2| = 0.5 \implies C(0.5) = e^{-0.5} = \rho^{1/2} \approx 0.606531$.
  - $|s_0 - s_3| = 0.5 \implies C(0.5) = e^{-0.5} = \rho^{1/2} \approx 0.606531$.
  Thus $\mathbf{v}_0 = (\rho^{3/2}, \rho^{1/2}, \rho^{1/2})'$.
  The Universal Kriging system is:
  $$
  \begin{bmatrix}
  1 & \rho & \rho^2 & 1 & -1 \\
  \rho & 1 & \rho & 1 & 0 \\
  \rho^2 & \rho & 1 & 1 & 1 \\
  1 & 1 & 1 & 0 & 0 \\
  -1 & 0 & 1 & 0 & 0
  \end{bmatrix}
  \begin{bmatrix}
  \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ \mu_0 \\ \mu_1
  \end{bmatrix}
  =
  \begin{bmatrix}
  \rho^{3/2} \\ \rho^{1/2} \\ \rho^{1/2} \\ 1 \\ 0.5
  \end{bmatrix}
  $$

- **(c)** The bottom two equations are the unbiasedness constraints:
  $$
  \begin{cases}
  \lambda_1 + \lambda_2 + \lambda_3 = 1 \\
  -\lambda_1 + \lambda_3 = 0.5 \implies \lambda_3 = \lambda_1 + 0.5
  \end{cases}
  $$
  Substitute $\lambda_3 = \lambda_1 + 0.5$ into the first constraint:
  $$
  \lambda_1 + \lambda_2 + (\lambda_1 + 0.5) = 1 \implies 2\lambda_1 + \lambda_2 = 0.5 \implies \lambda_2 = 0.5 - 2\lambda_1
  $$
  Now express all three weights in terms of $\lambda_1$:
  $$
  \boldsymbol{\lambda} = \begin{bmatrix} \lambda_1 \\ 0.5 - 2\lambda_1 \\ \lambda_1 + 0.5 \end{bmatrix}
  $$
  The first three equations are:
  $$
  \begin{cases}
  \lambda_1 + \rho \lambda_2 + \rho^2 \lambda_3 + \mu_0 - \mu_1 = \rho^{3/2} \\
  \rho \lambda_1 + \lambda_2 + \rho \lambda_3 + \mu_0 = \rho^{1/2} \\
  \rho^2 \lambda_1 + \rho \lambda_2 + \lambda_3 + \mu_0 + \mu_1 = \rho^{1/2}
  \end{cases}
  $$
  Eliminate $\mu_0$ and $\mu_1$. Add the first and third equations:
  $$
  (1 + \rho^2)\lambda_1 + 2\rho \lambda_2 + (1 + \rho^2)\lambda_3 + 2\mu_0 = \rho^{3/2} + \rho^{1/2}
  $$
  Multiply the second equation by 2:
  $$
  2\rho \lambda_1 + 2\lambda_2 + 2\rho \lambda_3 + 2\mu_0 = 2\rho^{1/2}
  $$
  Subtracting the second equation from the first eliminates $\mu_0$:
  $$
  (1 - 2\rho + \rho^2)\lambda_1 + (2\rho - 2)\lambda_2 + (1 - 2\rho + \rho^2)\lambda_3 = \rho^{3/2} - \rho^{1/2}
  $$
  Notice that $1 - 2\rho + \rho^2 = (1 - \rho)^2$ and $2\rho - 2 = -2(1 - \rho)$. Divide by $(1 - \rho) \ne 0$:
  $$
  (1 - \rho)\lambda_1 - 2\lambda_2 + (1 - \rho)\lambda_3 = \frac{\rho^{1/2}(\rho - 1)}{1 - \rho} = -\rho^{1/2}
  $$
  Now substitute $\lambda_2 = 0.5 - 2\lambda_1$ and $\lambda_3 = \lambda_1 + 0.5$:
  $$
  (1 - \rho)\lambda_1 - 2(0.5 - 2\lambda_1) + (1 - \rho)(\lambda_1 + 0.5) = -\rho^{1/2}
  $$
  Expand:
  $$
  (1 - \rho)\lambda_1 - 1 + 4\lambda_1 + (1 - \rho)\lambda_1 + 0.5(1 - \rho) = -\rho^{1/2}
  $$
  $$
  (2 - 2\rho + 4)\lambda_1 - 1 + 0.5 - 0.5\rho = -\rho^{1/2}
  $$
  $$
  (6 - 2\rho)\lambda_1 - 0.5 - 0.5\rho = -\rho^{1/2}
  $$
  Solving for $\lambda_1$:
  $$
  \lambda_1 = \frac{0.5 + 0.5\rho - \rho^{1/2}}{6 - 2\rho} = \frac{1 + \rho - 2\rho^{1/2}}{2(6 - 2\rho)} = \frac{(1 - \rho^{1/2})^2}{4(3 - \rho)}
  $$
  Evaluate numerically with $\rho \approx 0.367879$ and $\rho^{1/2} \approx 0.606531$:
  $$
  1 - \rho^{1/2} \approx 0.393469 \implies (1 - \rho^{1/2})^2 \approx 0.154818
  $$
  $$
  3 - \rho \approx 2.632121 \implies 4(3 - \rho) \approx 10.528485
  $$
  $$
  \lambda_1 = \frac{0.154818}{10.528485} \approx +0.014705
  $$
  Now compute $\lambda_2$ and $\lambda_3$:
  $$
  \lambda_2 = 0.5 - 2\lambda_1 \approx 0.5 - 2(0.014705) = 0.5 - 0.029409 \approx +0.470591
  $$
  $$
  \lambda_3 = \lambda_1 + 0.5 \approx 0.014705 + 0.5 \approx +0.514705
  $$
  Verify constraints:
  $$
  \sum \lambda_i = 0.014705 + 0.470591 + 0.514705 = 1.000001 \quad \checkmark
  $$
  $$
  \sum \lambda_i s_i = -0.014705 + 0 + 0.514705 = 0.500000 = s_0 \quad \checkmark
  $$

- **(d)** To compute $\sigma_{\text{UK}}^2(s_0)$, we first solve for the Lagrange multipliers:
  From equation (2): $\mu_0 = \rho^{1/2} - \rho \lambda_1 - \lambda_2 - \rho \lambda_3$.
  $$
  \mu_0 \approx 0.606531 - 0.367879(0.014705) - 0.470591 - 0.367879(0.514705) \approx -0.058778
  $$
  Subtract equation (1) from equation (3) to find $\mu_1$:
  $$
  (\rho^2 - 1)\lambda_1 + (1 - \rho^2)\lambda_3 + 2\mu_1 = \rho^{1/2} - \rho^{3/2}
  $$
  $$
  (1 - \rho^2)(\lambda_3 - \lambda_1) + 2\mu_1 = \rho^{1/2}(1 - \rho)
  $$
  Since $\lambda_3 - \lambda_1 = 0.5$:
  $$
  0.5(1 - \rho^2) + 2\mu_1 = \rho^{1/2}(1 - \rho) \implies 2\mu_1 = \rho^{1/2}(1 - \rho) - 0.5(1 - \rho)(1 + \rho)
  $$
  $$
  \mu_1 = \frac{1 - \rho}{2} \left(\rho^{1/2} - \frac{1 + \rho}{2}\right) \approx \frac{0.632121}{2} \left(0.606531 - 0.683940\right) \approx -0.024463
  $$
  Now compute $\sigma_{\text{UK}}^2(s_0) = C(0) - \boldsymbol{\lambda}'\mathbf{v}_0 - \boldsymbol{\mu}_{\text{UK}}'\mathbf{x}_0$:
  $$
  \boldsymbol{\lambda}'\mathbf{v}_0 = 0.014705(0.223130) + 0.470591(0.606531) + 0.514705(0.606531) \approx 0.003281 + 0.285429 + 0.312185 \approx 0.600895
  $$
  $$
  \boldsymbol{\mu}_{\text{UK}}'\mathbf{x}_0 = \mu_0(1) + \mu_1(0.5) \approx -0.058778 + (-0.024463)(0.5) \approx -0.071010
  $$
  $$
  \sigma_{\text{UK}}^2(s_0) = 1.0 - 0.600895 - (-0.071010) = 0.399105 + 0.071010 = 0.470115
  $$
  For comparison, the Simple Kriging weights (assuming known drift $\mu(s)$) for the two flanking neighbors $s_2=0$ and $s_3=1$ under Exponential covariance (Theorem 5.4.1) yield:
  $$
  \sigma_{\text{SK}}^2(s_0) = 1 - \frac{2 \rho}{1 + \rho} = 1 - \frac{2(0.367879)}{1.367879} \approx 1 - 0.537883 = 0.462117
  $$
  Notice that $\sigma_{\text{UK}}^2(s_0) = 0.470115 > 0.462117 = \sigma_{\text{SK}}^2(s_0)$. The difference $0.470115 - 0.462117 = 0.007998 > 0$ is the variance penalty for estimating the drift parameters $(\beta_0, \beta_1)$. $\blacksquare$

---

#### Exercise 6.2 (Equivalence of Primal and Dual Kriging Surfaces)
Let two observation points be located at $s_1 = 0$ and $s_2 = 1$ in $\mathbb{R}$ with observed values $Y(s_1) = 2$ and $Y(s_2) = 5$.
Assume an Ordinary Kriging model ($f_1(s) = 1$, $p = 1$) with linear semivariogram $\gamma(h) = |h|$.
- **(a)** Solve the Primal Ordinary Kriging system at an arbitrary point $s_0 \in [0, 1]$ to find the weights $\lambda_1(s_0), \lambda_2(s_0)$ and the prediction $\widehat{Y}_{\text{primal}}(s_0)$.
- **(b)** Formulate and solve the Dual Kriging system to obtain the dual coefficients $b_1, b_2$ and $d_1$.
- **(c)** Verify that the dual interpolating surface $\widehat{Y}_{\text{dual}}(s) = d_1 + b_1 \gamma(s - s_1) + b_2 \gamma(s - s_2)$ is identically equal to $\widehat{Y}_{\text{primal}}(s)$ for all $s \in [0, 1]$.

*Solution.*
- **(a)** The distances are $h_{12} = 1$, $h_{01} = s_0$, and $h_{02} = 1 - s_0$.
  The semivariograms are $\gamma(h_{12}) = 1$, $\gamma(h_{01}) = s_0$, and $\gamma(h_{02}) = 1 - s_0$.
  By Exercise 5.2, the Ordinary Kriging weights are:
  $$
  \lambda_1(s_0) = \frac{1}{2} + \frac{\gamma(h_{02}) - \gamma(h_{01})}{2\gamma(h_{12})} = \frac{1}{2} + \frac{(1 - s_0) - s_0}{2(1)} = \frac{1}{2} + \frac{1 - 2s_0}{2} = 1 - s_0
  $$
  $$
  \lambda_2(s_0) = 1 - \lambda_1(s_0) = s_0
  $$
  The primal prediction is:
  $$
  \widehat{Y}_{\text{primal}}(s_0) = \lambda_1(s_0) Y(s_1) + \lambda_2(s_0) Y(s_2) = (1 - s_0)(2) + s_0(5) = 2 - 2s_0 + 5s_0 = 2 + 3s_0
  $$

- **(b)** In semivariogram form, the Dual Kriging system (Theorem 6.5.1) is:
  $$
  \begin{bmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \mathbf{b} \\ d_1 \end{bmatrix} = \begin{bmatrix} \mathbf{Y} \\ 0 \end{bmatrix}
  $$
  where $\mathbf{\Gamma} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ and $\mathbf{Y} = \begin{bmatrix} 2 \\ 5 \end{bmatrix}$. The system is:
  $$
  \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ d_1 \end{bmatrix} = \begin{bmatrix} 2 \\ 5 \\ 0 \end{bmatrix}
  $$
  From the third equation, $b_1 + b_2 = 0 \implies b_2 = -b_1$.
  The first two equations are:
  $$
  \begin{cases}
  b_2 + d_1 = 2 \implies -b_1 + d_1 = 2 \\
  b_1 + d_1 = 5
  \end{cases}
  $$
  Subtracting the first equation from the second:
  $$
  2b_1 = 3 \implies b_1 = 1.5
  $$
  $$
  b_2 = -b_1 = -1.5
  $$
  Adding the two equations:
  $$
  2d_1 = 7 \implies d_1 = 3.5
  $$

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

---

#### Exercise 6.3 (Kriging with External Drift Stability, Centering, and Collinearity)
Suppose a target variable $Y$ is predicted using Kriging with External Drift with a single auxiliary covariate $q(s)$:
$$
Y(s) = \beta_0 + \beta_1 q(s) + \varepsilon(s)
$$
Observations are taken at 3 locations in a local neighborhood: $s_1, s_2, s_3$.
- **(a)** Write down the design matrix $\mathbf{X} \in \mathbb{R}^{3 \times 2}$.
- **(b)** Suppose the auxiliary covariate has raw values $q(s_1) = 100$, $q(s_2) = 100.01$, $q(s_3) = 100$. Show that $\operatorname{rank}(\mathbf{X}) = 2$, but compute the condition number $\kappa(\mathbf{X}'\mathbf{X})$ and explain how centering $q(s)$ drastically improves conditioning.
- **(c)** Explain what happens to the prediction variance if a target site $s_0$ has covariate value $q(s_0) = 105$ (extrapolation beyond the local sampled range $[100, 100.01]$), and state the diagnostic fallback rule.

*Solution.*
- **(a)** The design matrix is:
  $$
  \mathbf{X} = \begin{bmatrix} 1 & q(s_1) \\ 1 & q(s_2) \\ 1 & q(s_3) \end{bmatrix}
  $$

- **(b)** With the raw uncentered values:
  $$
  \mathbf{X} = \begin{bmatrix} 1 & 100.00 \\ 1 & 100.01 \\ 1 & 100.00 \end{bmatrix}
  $$
  Since the two columns are not scalar multiples of each other ($100.01 \ne 100.00$), the columns are linearly independent, so $\operatorname{rank}(\mathbf{X}) = 2$ (full column rank).
  Computing $\mathbf{X}'\mathbf{X}$:
  $$
  \mathbf{X}'\mathbf{X} = \begin{bmatrix} 1+1+1 & 100+100.01+100 \\ 100+100.01+100 & 100^2 + 100.01^2 + 100^2 \end{bmatrix} = \begin{bmatrix} 3 & 300.01 \\ 300.01 & 30002.0001 \end{bmatrix}
  $$
  The determinant is:
  $$
  \det(\mathbf{X}'\mathbf{X}) = 3(30002.0001) - (300.01)^2 = 90006.0003 - 90006.0001 = 0.0002
  $$
  The trace is $\operatorname{tr}(\mathbf{X}'\mathbf{X}) \approx 30005.0001$.
  The eigenvalues satisfy $\lambda_1 \approx 30005$ and $\lambda_2 = \frac{\det}{\lambda_1} \approx \frac{0.0002}{30005} \approx 6.66 \times 10^{-9}$.
  The 2-norm condition number of the uncentered Gram matrix is:
  $$
  \kappa(\mathbf{X}'\mathbf{X}) = \frac{\lambda_1}{\lambda_2} \approx \frac{30005}{6.66 \times 10^{-9}} \approx 4.5 \times 10^{12}
  $$
  Now consider centering the covariate by subtracting the mean $\bar{q} = \frac{100 + 100.01 + 100}{3} \approx 100.00333$:
  $$
  \tilde{q}_1 = -0.00333, \quad \tilde{q}_2 = +0.00667, \quad \tilde{q}_3 = -0.00333
  $$
  The centered design matrix $\widetilde{\mathbf{X}} = [\mathbf{1}, \tilde{\mathbf{q}}]$ has orthogonal columns ($\mathbf{1}'\tilde{\mathbf{q}} = 0$). Hence:
  $$
  \widetilde{\mathbf{X}}'\widetilde{\mathbf{X}} = \begin{bmatrix} 3 & 0 \\ 0 & \sum \tilde{q}_i^2 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 0 & 6.67 \times 10^{-5} \end{bmatrix}
  $$
  Its condition number is $\kappa(\widetilde{\mathbf{X}}'\widetilde{\mathbf{X}}) = \frac{3}{6.67 \times 10^{-5}} \approx 4.5 \times 10^4$, reducing the condition number by eight orders of magnitude ($10^{12} \to 10^4$).

- **(c)** The true numerical vulnerability in KED is not merely near-flatness of the covariate, but **covariate extrapolation** relative to the local sampled range.
  Suppose a target site $s_0$ has covariate value $q(s_0) = 105$. The sampled range of $q$ is only $[100, 100.01]$ (a spread of $0.01$). Predicting at $q(s_0) = 105$ represents an extrapolation of:
  $$
  \frac{q(s_0) - \bar{q}}{\operatorname{SD}(q)} = \frac{105 - 100.0033}{0.00577} \approx 866 \text{ standard deviations!}
  $$
  In the KED variance decomposition:
  $$
  \sigma_{\text{KED}}^2(s_0) = \sigma_{\text{SK}}^2(s_0) + (\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)' (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1} (\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)
  $$
  the drift estimation penalty contains $(q(s_0) - \mathbf{q}'\mathbf{V}^{-1}\mathbf{v}_0)^2 \cdot \operatorname{Var}(\widehat{\beta}_1)$. Because $(q(s_0) - \bar{q})^2 \approx 5^2 = 25$ while $\operatorname{Var}(\widehat{\beta}_1) \approx \frac{\sigma^2}{\sum \tilde{q}_i^2} = \frac{\sigma^2}{6.67 \times 10^{-5}} \approx 15{,}000 \sigma^2$, the drift variance penalty explodes to hundreds of thousands of times $\sigma^2$.

  **Algorithmic Remedy:**
  1. Center and scale all external covariates during design matrix construction.
  2. Before running KED in a local moving neighborhood, evaluate the local relative variance ratio:
     $$
     R = \frac{\operatorname{Var}_{\text{local}}(q)}{\operatorname{Var}_{\text{global}}(q)}
     $$
  3. If $R < 10^{-4}$ (meaning $q$ is effectively flat in the neighborhood), or if the target covariate $q(s_0)$ lies outside $[\min_{i} q(s_i) - 0.2\Delta q, \max_i q(s_i) + 0.2\Delta q]$, automatically fall back to Ordinary Kriging with constant drift $\mathbf{x}(\mathbf{s}) = [1]$. $\blacksquare$

---

#### Exercise 6.4 (IRF-$k$ Authorized Linear Combinations)
Let $Z(s)$ be an Intrinsic Random Function of order $k = 1$ on $\mathbb{R}$.
- **(a)** State the definition of an authorized linear combination of order 1.
- **(b)** Consider four equispaced observation points $s_1 = 0, s_2 = 1, s_3 = 2, s_4 = 3$. Find the weights $w_1, w_2, w_3, w_4$ corresponding to the second central difference operator, and prove that it is an ALC-1.
- **(c)** Given the cubic generalized covariance $K(h) = -|h| + |h|^3$, compute the exact variance of this second difference:
  $$
  \operatorname{Var}(Z(s_1) - 2Z(s_2) + Z(s_3))
  $$

*Solution.*
- **(a)** For order $k = 1$, the polynomial space $\mathcal{P}_1$ is spanned by $p_0(s) = 1$ and $p_1(s) = s$.
  A linear combination $\sum_{i=1}^n w_i Z(s_i)$ is an ALC-1 if and only if:
  $$
  \sum_{i=1}^n w_i = 0 \quad \text{and} \quad \sum_{i=1}^n w_i s_i = 0
  $$

- **(b)** The second central difference at $s = 1$ is:
  $$
  \Delta^2 Z(1) = Z(0) - 2Z(1) + Z(2)
  $$
  The weights are $w_1 = 1$, $w_2 = -2$, $w_3 = 1$, and $w_4 = 0$.
  Check condition 1:
  $$
  \sum_{i=1}^4 w_i = 1 - 2 + 1 + 0 = 0 \quad \checkmark
  $$
  Check condition 2:
  $$
  \sum_{i=1}^4 w_i s_i = 1(0) - 2(1) + 1(2) + 0(3) = 0 - 2 + 2 + 0 = 0 \quad \checkmark
  $$
  Both conditions are satisfied. Thus $\Delta^2 Z(1)$ is an authorized linear combination of order 1.

- **(c)** For an ALC-1, the variance is given by:
  $$
  \operatorname{Var}\left(\sum_{i=1}^3 w_i Z(s_i)\right) = \sum_{i=1}^3 \sum_{j=1}^3 w_i w_j K(s_i - s_j)
  $$
  where $K(h) = -|h| + |h|^3$.
  Evaluate $K(h)$ for all pairwise distance lags:
  - $h = 0$: $K(0) = -0 + 0 = 0$.
  - $h = 1$: $K(1) = -1 + 1^3 = -1 + 1 = 0$.
  - $h = 2$: $K(2) = -2 + 2^3 = -2 + 8 = 6$.
  The distance matrix for points $s_1=0, s_2=1, s_3=2$ has entries $|s_i - s_j|$:
  $$
  \mathbf{H} = \begin{bmatrix} 0 & 1 & 2 \\ 1 & 0 & 1 \\ 2 & 1 & 0 \end{bmatrix} \implies \mathbf{K} = \begin{bmatrix} 0 & 0 & 6 \\ 0 & 0 & 0 \\ 6 & 0 & 0 \end{bmatrix}
  $$
  Now evaluate the quadratic form $\mathbf{w}'\mathbf{K}\mathbf{w}$ with $\mathbf{w} = (1, -2, 1)'$:
  $$
  \mathbf{K}\mathbf{w} = \begin{bmatrix} 0 & 0 & 6 \\ 0 & 0 & 0 \\ 6 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} = \begin{bmatrix} 6(1) \\ 0 \\ 6(1) \end{bmatrix} = \begin{bmatrix} 6 \\ 0 \\ 6 \end{bmatrix}
  $$
  $$
  \operatorname{Var}(\Delta^2 Z(1)) = \mathbf{w}' (\mathbf{K}\mathbf{w}) = 1(6) - 2(0) + 1(6) = 6 + 0 + 6 = 12
  $$
  The variance of the second difference is exactly 12. Notice that although $K(1) = 0$ and $K(h)$ is not a valid stationary covariance, the variance of the authorized linear combination is strictly positive ($\operatorname{Var} = 12 > 0$). $\blacksquare$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 6: Spatial Regression and Non-Stationary Processes — Companion Exercises and Complete Solutions](ch06_exercises.md)
