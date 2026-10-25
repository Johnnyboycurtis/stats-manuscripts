# Chapter 6: Spatial Regression and Non-Stationary Kriging

This chapter develops the statistical and mathematical framework for non-stationary spatial prediction. When the expectation of a spatial random field varies systematically across geographic space, the assumption of a constant mean fails. In such settings, spatial variation exhibits a dual nature: large-scale deterministic trends (drift) governed by spatial coordinates or exogenous physical covariates, superimposed upon small-scale autocorrelated stochastic fluctuations.

Following the pedagogical rhythm of Maria L. Rizzo and the semantic linear-model discipline of Oliver Schabenberger & Carol A. Gotway, we formulate the spatial linear model with correlated residuals, prove the Gauss-Markov theorem for Generalized Least Squares (GLS) drift estimation, derive the Universal Kriging (UK) system, examine Kriging with External Drift (KED), establish the Dual Kriging formulation and its equivalence with spline interpolation in Reproducing Kernel Hilbert Spaces (RKHS), and formalize the theory of Intrinsic Random Functions of order $k$ (IRF-$k$).

---

### Classical-to-Spatial Bridge: From OLS to GLS via Cholesky Whitening, and the Transition to Universal Kriging

Before entering the technical machinery of Universal Kriging and non-stationary random fields, we trace the conceptual pipeline connecting classical linear regression to spatial prediction. This bridge clarifies why classical Ordinary Least Squares fails in spatial domains, how matrix factorization restores Gauss-Markov optimality, and how Generalized Least Squares naturally generalizes into Universal Kriging.

#### 1. Classical Homoscedastic Linear Regression
In introductory linear regression, an $n$-dimensional observation vector $\mathbf{Y} \in \mathbb{R}^n$ is modeled as a linear combination of $p$ explanatory variables corrupted by independent and identically distributed (i.i.d.) Gaussian errors:
$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}\left(\mathbf{0}, \sigma^2 \mathbf{I}_n\right)
$$
where $\mathbf{X} \in \mathbb{R}^{n \times p}$ is a design matrix of full column rank $p$, $\boldsymbol{\beta} \in \mathbb{R}^p$ is an unknown regression vector, and $\sigma^2 > 0$ is the constant residual variance. Under the classical Gauss-Markov assumptions, the Ordinary Least Squares (OLS) estimator:
$$
\widehat{\boldsymbol{\beta}}_{\text{OLS}} = \left(\mathbf{X}'\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{Y}
$$
is the unique Best Linear Unbiased Estimator (BLUE) of $\boldsymbol{\beta}$, with sampling covariance matrix:
$$
\operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{OLS}}\right) = \sigma^2 \left(\mathbf{X}'\mathbf{X}\right)^{-1}
$$

#### 2. The Spatial Breakdown: Dense Covariance and Spurious Precision
When observations $Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_n)$ are sampled across a spatial domain $D \subset \mathbb{R}^d$, Tobler's First Law dictates that proximate locations share correlated stochastic fluctuations. The residual vector $\boldsymbol{\varepsilon} = \mathbf{Y} - \mathbf{X}\boldsymbol{\beta}$ is no longer spherical white noise. Instead, it is governed by a dense, non-diagonal spatial covariance matrix:
$$
\mathbf{V} = \operatorname{Var}(\boldsymbol{\varepsilon}) = \left[ C(\mathbf{s}_i - \mathbf{s}_j) \right]_{i,j=1}^n \in \mathbb{R}^{n \times n}
$$
Applying classical OLS regression to spatially correlated data introduces severe methodological pathologies:
1. **Loss of Gauss-Markov Optimality:** The OLS estimator $\widehat{\boldsymbol{\beta}}_{\text{OLS}}$ remains unbiased ($\operatorname{E}[\widehat{\boldsymbol{\beta}}_{\text{OLS}}] = \boldsymbol{\beta}$), but it is no longer BLUE. Its true sampling covariance:
   $$
   \operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{OLS}}\right) = \left(\mathbf{X}'\mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V} \mathbf{X} \left(\mathbf{X}'\mathbf{X}\right)^{-1}
   $$
   is strictly larger (in the Loewner matrix partial ordering) than the optimal Generalized Least Squares variance.
2. **Spurious Precision and False Significance:** Standard statistical software computes parameter standard errors under the naive i.i.d. assumption $\widehat{\sigma}^2(\mathbf{X}'\mathbf{X})^{-1}$. Under positive spatial correlation, this naive variance drastically underestimates the true sampling uncertainty—often by factors of 2 to 10. Consequently, standard errors are spuriously narrow, $t$-statistics and $F$-statistics are artificially inflated, and analysts routinely identify "statistically significant" spatial covariates that are pure artifacts of unmodeled spatial autocorrelation.
3. **Sill Attenuation in Residual Variograms:** As detailed in Section 6.1, OLS residuals $\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} = (\mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}')\mathbf{Y}$ are constrained to be orthogonal to $\mathbf{X}$. This causes the regression plane to absorb low-frequency spatial variation, creating severe downward bias in the empirical semivariogram sill.

#### 3. Cholesky Whitening: Restoring Gauss-Markov Optimality
To overcome these failures, we transform the correlated observation space into an uncorrelated space where the classical Gauss-Markov theorem applies directly. This transformation is known as **pre-whitening**.

Because the spatial covariance matrix $\mathbf{V}$ is symmetric positive definite, it admits a unique Cholesky decomposition:
$$
\mathbf{V} = \mathbf{L}\mathbf{L}'
$$
where $\mathbf{L} \in \mathbb{R}^{n \times n}$ is a lower-triangular matrix with strictly positive diagonal entries. The inverse factor $\mathbf{L}^{-1}$ acts as a spatial whitening filter. We transform both the response vector and the design matrix:
$$
\mathbf{Y}^* = \mathbf{L}^{-1}\mathbf{Y}, \qquad \mathbf{X}^* = \mathbf{L}^{-1}\mathbf{X}, \qquad \boldsymbol{\varepsilon}^* = \mathbf{L}^{-1}\boldsymbol{\varepsilon}
$$
Evaluating the covariance matrix of the transformed residuals $\boldsymbol{\varepsilon}^*$:
$$
\operatorname{Var}\left(\boldsymbol{\varepsilon}^*\right) = \mathbf{L}^{-1}\operatorname{Var}(\boldsymbol{\varepsilon})\left(\mathbf{L}^{-1}\right)' = \mathbf{L}^{-1}\mathbf{V}\left(\mathbf{L}'\right)^{-1} = \mathbf{L}^{-1}\left(\mathbf{L}\mathbf{L}'\right)\left(\mathbf{L}'\right)^{-1} = \mathbf{I}_n
$$
In this whitened coordinate system, the errors $\boldsymbol{\varepsilon}^*$ are uncorrelated and homoscedastic with unit variance! Applying classical Ordinary Least Squares to the transformed linear model $\mathbf{Y}^* = \mathbf{X}^*\boldsymbol{\beta} + \boldsymbol{\varepsilon}^*$:
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
This is valid only because the unobserved error $\varepsilon(\mathbf{s}_0)$ is assumed completely independent of the observed sample errors $\boldsymbol{\varepsilon}$.

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

#### Beat 1: The Mathematical Anchor (The "Why")

In environmental sciences, geophysics, and spatial epidemiology, spatial processes rarely maintain a constant mean across regional domains. Topographic elevation gradients, latitudinal temperature trends, atmospheric wind fields, and geological depositional directions induce deterministic spatial drift. 

If one ignores this trend and fits an Ordinary Kriging model, the empirical semivariogram displays an apparent parabolic upward curvature ($\gamma(h) \propto h^2$) at large lags, violating second-order stationarity and contaminating the spatial covariance structure. Conversely, if one naively removes the trend via Ordinary Least Squares (OLS) regression, the resulting residuals violate the classical Gauss-Markov assumptions because they remain spatially autocorrelated.

#### Definition 6.1.1 (Spatial Linear Model)
Let $D \subset \mathbb{R}^d$ ($d \ge 1$) be a spatial domain. A continuous-index spatial random field $\{Y(\mathbf{s}) : \mathbf{s} \in D\}$ follows a **spatial linear model** if for each $\mathbf{s} \in D$:
$$
Y(\mathbf{s}) = \mu(\mathbf{s}) + \varepsilon(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \varepsilon(\mathbf{s})
$$
where:
1. $\mu(\mathbf{s}) = \operatorname{E}[Y(\mathbf{s})]$ is the deterministic **spatial drift (trend)** function, modeled as a linear combination of $p$ known deterministic spatial basis functions $\{f_1(\mathbf{s}), \dots, f_p(\mathbf{s})\}$:
   $$
   \mu(\mathbf{s}) = \sum_{j=1}^p f_j(\mathbf{s}) \beta_j = \mathbf{x}(\mathbf{s})' \boldsymbol{\beta}
   $$
   Here $\mathbf{x}(\mathbf{s}) = (f_1(\mathbf{s}), \dots, f_p(\mathbf{s}))' \in \mathbb{R}^p$ is the known basis evaluation vector at $\mathbf{s}$, and $\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)' \in \mathbb{R}^p$ is a vector of fixed, unknown regression parameters.
2. $\{\varepsilon(\mathbf{s}) : \mathbf{s} \in D\}$ is a zero-mean, second-order stationary spatial residual random field with $\operatorname{E}[\varepsilon(\mathbf{s})] = 0$ and continuous stationary covariance kernel:
   $$
   C(\mathbf{h}) = \operatorname{Cov}(\varepsilon(\mathbf{s}), \varepsilon(\mathbf{s}+\mathbf{h})) = \operatorname{E}[\varepsilon(\mathbf{s})\varepsilon(\mathbf{s}+\mathbf{h})], \quad \mathbf{h} \in \mathbb{R}^d
   $$
   Following the three-layer decomposition (Chapter 1), the residual decomposes into a spatially correlated smooth effect and an uncorrelated microscale error: $\varepsilon(\mathbf{s}) = \delta(\mathbf{s}) + \epsilon(\mathbf{s})$, where $\operatorname{Var}(\epsilon(\mathbf{s})) = \tau^2$.

#### Definition 6.1.2 (Matrix Representation of Sample Observations)
Suppose the field is observed at $n$ distinct locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in D$ with $n > p$. We define:
- The observation vector: $\mathbf{Y} = (Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_n))' \in \mathbb{R}^n$.
- The spatial design matrix:
  $$
  \mathbf{X} = \begin{bmatrix} \mathbf{x}(\mathbf{s}_1)' \\ \vdots \\ \mathbf{x}(\mathbf{s}_n)' \end{bmatrix} = \begin{bmatrix} f_1(\mathbf{s}_1) & \cdots & f_p(\mathbf{s}_1) \\ \vdots & \ddots & \vdots \\ f_1(\mathbf{s}_n) & \cdots & f_p(\mathbf{s}_n) \end{bmatrix} \in \mathbb{R}^{n \times p}
  $$
  We assume $\operatorname{rank}(\mathbf{X}) = p \le n$, ensuring full column rank.
- The residual vector: $\boldsymbol{\varepsilon} = (\varepsilon(\mathbf{s}_1), \dots, \varepsilon(\mathbf{s}_n))' \in \mathbb{R}^n$ with $\operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}$ and $n \times n$ symmetric positive definite covariance matrix:
  $$
  \mathbf{V} = \operatorname{Var}(\boldsymbol{\varepsilon}) = \operatorname{Var}(\mathbf{Y}) = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}
  $$
The observation vector satisfies the spatial linear model:
$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad \operatorname{E}[\mathbf{Y}] = \mathbf{X}\boldsymbol{\beta}, \quad \operatorname{Var}(\mathbf{Y}) = \mathbf{V}
$$

#### Beat 2: Algorithmic Translation (The "How")

The spatial linear model requires specifying both the trend basis functions and the residual covariance structure.

```
Spatial Linear Model Structural Decomposition:
+---------------------------------------------------------------------------------+
| Y(s) = x(s)' β + δ(s) + ϵ(s)                                                    |
+---------------------------------------------------------------------------------+
         │
         ├── 1. Deterministic Trend / Drift:       μ(s) = x(s)' β
         │      Basis choices:
         │      - Constant mean: x(s) = [1] (Ordinary Kriging, p = 1)
         │      - Linear coordinates: x(s) = [1, s_x, s_y]' (p = 3)
         │      - Quadratic coordinates: x(s) = [1, s_x, s_y, s_x^2, s_y^2, s_x s_y]' (p = 6)
         │      - External physical covariates: x(s) = [1, Elevation(s), Temperature(s)]'
         │
         └── 2. Correlated Residuals:              ε(s) = δ(s) + ϵ(s)
                Var(Y) = V = C(θ) + τ^2 I_n
```

Procedural steps for setting up a spatial linear regression:
1. Construct the $n \times p$ coordinate design matrix $\mathbf{X}$ by evaluating basis functions $\{f_1, \dots, f_p\}$ at each site $\mathbf{s}_i$.
2. Verify full column rank by evaluating the condition number $\kappa(\mathbf{X}'\mathbf{X}) < 10^8$.
3. Specify the residual spatial covariance kernel $C(\mathbf{h}; \boldsymbol{\theta})$ with parameter vector $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$.
4. Assemble the $n \times n$ covariance matrix $\mathbf{V}$.

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

#### The Pitfall of OLS Residuals: Projection Bias
If an analyst naively estimates the drift via Ordinary Least Squares:
$$
\widehat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}
$$
the estimator remains unbiased ($\operatorname{E}[\widehat{\boldsymbol{\beta}}_{\text{OLS}}] = \boldsymbol{\beta}$), but its variance is inflated:
$$
\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{OLS}}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1} \ge (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}
$$
More crucially, if the analyst evaluates the empirical semivariogram on the OLS residuals:
$$
\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} = \mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{OLS}} = \left(\mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\right)\mathbf{Y} = \mathbf{H}_\perp \boldsymbol{\varepsilon}
$$
the expectation of the residual outer product satisfies:
$$
\operatorname{E}\left[\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} \widehat{\boldsymbol{\varepsilon}}_{\text{OLS}}'\right] = \mathbf{H}_\perp \mathbf{V} \mathbf{H}_\perp \ne \mathbf{V}
$$
Because the geometric projection matrix $\mathbf{H}_\perp$ forces the residuals to be orthogonal to the columns of $\mathbf{X}$ ($\mathbf{X}'\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} = \mathbf{0}$), it absorbs low-frequency spatial variation into the fitted trend. As a result:
- The empirical semivariogram of OLS residuals is **severely biased downward**, particularly at large lag distances.
- The fitted spatial range $\phi$ is systematically underestimated.
- The sill $\sigma^2$ is systematically underestimated.

This fundamental circularity—one needs the covariance matrix $\mathbf{V}$ to correctly estimate $\boldsymbol{\beta}$ via GLS, but needs $\boldsymbol{\beta}$ to calculate residuals and estimate $\mathbf{V}$—is known as the **trend-covariance confounding dilemma**. It necessitates either Universal Kriging, Restricted Maximum Likelihood (REML), or Intrinsic Random Functions.

> [!NOTE]
> **Wait—Why Did We Do That?**
> *Why is the empirical semivariogram computed on OLS residuals biased downward at long separation distances (sill attenuation)?*
>
> When spatial drift is fitted via OLS, the optimization criterion minimizes $\sum_{i=1}^n (Y(\mathbf{s}_i) - \mathbf{x}(\mathbf{s}_i)'\boldsymbol{\beta})^2$ under the false assumption of independent, uncorrelated errors. To minimize this global sum of squared deviations, the OLS regression plane or polynomial surface bends to follow large-scale, low-frequency spatial patterns that actually belong to the autocorrelated stochastic process $\delta(\mathbf{s})$. The resulting residual projection operator $\mathbf{H}_\perp = \mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ strictly enforces $p$ orthogonality constraints ($\mathbf{X}'\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}} = \mathbf{0}$).
>
> This forces the sample variance of the residuals to be strictly smaller than the true process variance:
> $$
> \operatorname{E}\left[\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}}'\widehat{\boldsymbol{\varepsilon}}_{\text{OLS}}\right] = \operatorname{tr}(\mathbf{H}_\perp \mathbf{V}) = \operatorname{tr}(\mathbf{V}) - \operatorname{tr}\left((\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}\mathbf{X}\right) < \operatorname{tr}(\mathbf{V})
> $$
> At large spatial separation distances $\mathbf{h}$, where pairs of locations span across the spatial domain, the fitted trend absorbs the regional variance contrast. Consequently, the empirical semivariogram values $\widehat{\gamma}_{\text{OLS}}(\mathbf{h})$ plateau at an artificially diminished level (attenuated sill) and fail to capture the long-range correlation structure, misleading the analyst into fitting an artificially short spatial range parameter $\phi$.

---

### 6.2 Generalized Least Squares (GLS) Estimation of Spatial Drift

#### Beat 1: The Mathematical Anchor (The "Why")

When observations share spatial correlation, Ordinary Least Squares fails to be optimal. The Generalized Least Squares (GLS) estimator rotates and scales the observation space to whiten the autocorrelated errors, yielding the Best Linear Unbiased Estimator (BLUE).

#### Theorem 6.2.1 (Gauss-Markov Theorem for Spatial Drift Estimation)
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
1. **Cholesky Pre-Whitening Transformation:** Because $\mathbf{V}$ is symmetric positive definite, there exists a unique lower-triangular Cholesky factor $\mathbf{L}$ such that $\mathbf{V} = \mathbf{L}\mathbf{L}'$, and an inverse square root $\mathbf{V}^{-1/2} = \mathbf{L}^{-1}$ satisfying $\mathbf{V}^{-1} = (\mathbf{V}^{-1/2})'\mathbf{V}^{-1/2}$.
   Pre-multiply the linear system by $\mathbf{V}^{-1/2}$:
   $$
   \mathbf{Y}^* = \mathbf{X}^* \boldsymbol{\beta} + \boldsymbol{\varepsilon}^*
   $$
   where $\mathbf{Y}^* = \mathbf{V}^{-1/2}\mathbf{Y}$, $\mathbf{X}^* = \mathbf{V}^{-1/2}\mathbf{X}$, and $\boldsymbol{\varepsilon}^* = \mathbf{V}^{-1/2}\boldsymbol{\varepsilon}$.
   Evaluate the moments of the transformed residuals:
   $$
   \operatorname{E}[\boldsymbol{\varepsilon}^*] = \mathbf{V}^{-1/2}\operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}
   $$
   $$
   \operatorname{Var}(\boldsymbol{\varepsilon}^*) = \mathbf{V}^{-1/2}\operatorname{Var}(\boldsymbol{\varepsilon})(\mathbf{V}^{-1/2})' = \mathbf{V}^{-1/2}\mathbf{V}(\mathbf{V}^{-1/2})' = \mathbf{I}_n
   $$
   The transformed model satisfies the classical homoscedastic Gauss-Markov conditions with uncorrelated white noise. The OLS estimator for the transformed model is:
   $$
   \widehat{\boldsymbol{\beta}} = \left((\mathbf{X}^*)' \mathbf{X}^*\right)^{-1} (\mathbf{X}^*)' \mathbf{Y}^* = \left(\mathbf{X}' \mathbf{V}^{-1/2} (\mathbf{V}^{-1/2})' \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1/2} (\mathbf{V}^{-1/2})' \mathbf{Y} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}
   $$
   which is identically $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$.

2. **Unbiasedness:**
   $$
   \operatorname{E}\left[\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right] = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \operatorname{E}[\mathbf{Y}] = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right) \boldsymbol{\beta} = \boldsymbol{\beta}
   $$

3. **Covariance Matrix:**
   $$
   \begin{aligned}
   \operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) &= \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \operatorname{Var}(\mathbf{Y}) \mathbf{V}^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \\
   &= \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{V} \mathbf{V}^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \\
   &= \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right) \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1}
   \end{aligned}
   $$

4. **BLUE Optimality:** Let $\mathbf{w}'\mathbf{Y}$ be any linear unbiased estimator of $\mathbf{a}'\boldsymbol{\beta}$. Unbiasedness requires:
   $$
   \operatorname{E}[\mathbf{w}'\mathbf{Y}] = \mathbf{w}'\mathbf{X}\boldsymbol{\beta} = \mathbf{a}'\boldsymbol{\beta}, \quad \forall \boldsymbol{\beta} \in \mathbb{R}^p \implies \mathbf{X}'\mathbf{w} = \mathbf{a}
   $$
   Define the GLS weight vector $\mathbf{w}_0 = \mathbf{V}^{-1}\mathbf{X}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{a}$, so that $\mathbf{w}_0'\mathbf{Y} = \mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}}$.
   Notice that $\mathbf{X}'\mathbf{w}_0 = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{a} = \mathbf{a}$.
   Write $\mathbf{w} = \mathbf{w}_0 + \mathbf{u}$. Then $\mathbf{X}'\mathbf{w} = \mathbf{X}'\mathbf{w}_0 + \mathbf{X}'\mathbf{u} = \mathbf{a} + \mathbf{X}'\mathbf{u} = \mathbf{a}$, which implies:
   $$
   \mathbf{X}'\mathbf{u} = \mathbf{0}
   $$
   Now evaluate the variance of $\mathbf{w}'\mathbf{Y}$:
   $$
   \operatorname{Var}(\mathbf{w}'\mathbf{Y}) = \mathbf{w}'\mathbf{V}\mathbf{w} = (\mathbf{w}_0 + \mathbf{u})'\mathbf{V}(\mathbf{w}_0 + \mathbf{u}) = \mathbf{w}_0'\mathbf{V}\mathbf{w}_0 + 2\mathbf{u}'\mathbf{V}\mathbf{w}_0 + \mathbf{u}'\mathbf{V}\mathbf{u}
   $$
   Evaluate the cross-product term:
   $$
   \mathbf{u}'\mathbf{V}\mathbf{w}_0 = \mathbf{u}'\mathbf{V}\left(\mathbf{V}^{-1}\mathbf{X}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{a}\right) = \mathbf{u}'\mathbf{X}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{a} = (\mathbf{X}'\mathbf{u})'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{a} = \mathbf{0}'\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{a} = 0
   $$
   Therefore:
   $$
   \operatorname{Var}(\mathbf{w}'\mathbf{Y}) = \operatorname{Var}\left(\mathbf{a}'\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) + \mathbf{u}'\mathbf{V}\mathbf{u}
   $$
   Because $\mathbf{V}$ is strictly positive definite, $\mathbf{u}'\mathbf{V}\mathbf{u} \ge 0$, with equality if and only if $\mathbf{u} = \mathbf{0}$, meaning $\mathbf{w} = \mathbf{w}_0$. $\blacksquare$

##### Equation Anatomy: The Generalized Least Squares (GLS) Estimator and Parameter Covariance

$$
\widehat{\boldsymbol{\beta}}_{\text{GLS}} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}, \qquad \operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1}
$$

- $\left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1}$ (The Precision Normalizer / Information Metric): The inverse of the Fisher information matrix for $\boldsymbol{\beta}$ under Gaussian errors. Expanding via the Cholesky factor $\mathbf{V} = \mathbf{L}\mathbf{L}'$, the kernel matrix $\mathbf{X}' \mathbf{V}^{-1} \mathbf{X} = (\mathbf{L}^{-1}\mathbf{X})'(\mathbf{L}^{-1}\mathbf{X})$ computes the inner product of the pre-whitened spatial covariates. It measures the effective independent spatial sample size provided by the design matrix $\mathbf{X}$. When observations are clustered geographically, off-diagonal elements of $\mathbf{V}$ are large, which diminishes the eigenvalues of $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ and inflates the sampling variance $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$ relative to naive OLS.
- $\mathbf{X}' \mathbf{V}^{-1}$ (The Whitened Projection Operator): The linear weighting operator that maps the $n$ spatial observations into the $p$-dimensional parameter space. Factoring $\mathbf{V}^{-1} = (\mathbf{L}')^{-1}\mathbf{L}^{-1}$, this operator is equivalent to $(\mathbf{L}^{-1}\mathbf{X})'\mathbf{L}^{-1}$, which first filters out spatial autocorrelation by pre-multiplying $\mathbf{Y}$ by $\mathbf{L}^{-1}$ (down-weighting clustered points) and then projects the decorrelated signal onto the whitened covariate directions $\mathbf{L}^{-1}\mathbf{X}$.
- $\mathbf{Y} \in \mathbb{R}^n$ (The Spatial Observation Vector): The realization of the spatial process at the $n$ sampling locations $\mathbf{s}_1, \dots, \mathbf{s}_n$.
- $\operatorname{Var}\left(\widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1}$ (Minimal Sampling Variance): The theoretical lower bound (BLUE) on estimation uncertainty among all linear unbiased estimators. Crucially, unlike the naive formula $\sigma^2(\mathbf{X}'\mathbf{X})^{-1}$, this variance incorporates the true spatial geometry of the sampling network, properly expanding the confidence ellipsoids around $\boldsymbol{\beta}$ to account for spatial redundancy.

#### Beat 2: Algorithmic Translation (The "How")

The GLS estimation algorithm computes $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and its parameter covariance matrix without inverting $\mathbf{V}$ directly, using Cholesky factorizations.

```python
def gls_trend_estimation(sample_coords, sample_vals, basis_functions, cov_kernel):
    """
    Computes Generalized Least Squares (GLS) drift parameter estimates and covariance.

    Parameters:
    -----------
    sample_coords : list of tuples/lists of float, shape (n, d)
        Coordinates s_1, ..., s_n.
    sample_vals : list of float, shape (n,)
        Observations Y(s_1), ..., Y(s_n).
    basis_functions : list of callable
        p spatial basis functions [f_1(s), ..., f_p(s)].
    cov_kernel : callable
        Covariance function C(h).

    Returns:
    --------
    beta_gls : list of float, shape (p,)
        Estimated drift coefficients.
    cov_beta : list of list of float, shape (p, p)
        Covariance matrix Var(beta_gls) = (X' V^-1 X)^-1.
    """
    n = len(sample_coords)
    p = len(basis_functions)
    d = len(sample_coords[0])

    # Step 1: Construct the n x p design matrix X
    X = [[basis_functions[j](sample_coords[i]) for j in range(p)] for i in range(n)]

    # Step 2: Construct the n x n covariance matrix V
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = sum((sample_coords[i][k] - sample_coords[j][k]) ** 2 for k in range(d)) ** 0.5
            V[i][j] = cov_kernel(dist)

    # Step 3: Compute V^-1 * X and V^-1 * Y via linear solve
    # For transparency, solve V * W_x = X and V * w_y = Y column by column
    # using Gauss-Jordan elimination on [V | X | Y]
    aug = [V[i][:] + X[i][:] + [sample_vals[i]] for i in range(n)]

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

    # Step 4: Compute the p x p Fisher information matrix M = X' * V^-1 * X
    # and the p x 1 vector rhs = X' * V^-1 * Y
    M = [[sum(X[i][r] * V_inv_X[i][c] for i in range(n)) for c in range(p)] for r in range(p)]
    rhs = [sum(X[i][r] * V_inv_Y[i] for i in range(n)) for r in range(p)]

    # Step 5: Solve M * beta_gls = rhs for beta_gls, and compute cov_beta = M^-1
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

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Geometric Whitening:** GLS rotates the coordinate system into the eigenspace of $\mathbf{V}$, scaling axes by the inverse square roots of the eigenvalues $1/\sqrt{\lambda_i}$. Observations clustered closely together are down-weighted because they share high spatial redundancy (collinear rows in $\mathbf{V}$).
- **The Loewner Ordering:** The variance difference matrix $\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{OLS}}) - \operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}})$ is always positive semi-definite ($\mathbf{A} \ge_{\text{Loewner}} \mathbf{B}$). The efficiency gain of GLS over OLS increases as spatial correlation strengthens (i.e., as range $\phi$ grows).

> [!NOTE]
> **Wait—Why Did We Do That?**
> *What is "Spatial Confounding" between the deterministic trend $\mathbf{X}\boldsymbol{\beta}$ and the stochastic field $\delta(\mathbf{s})$?*
>
> Spatial confounding arises when physical covariates or trend basis functions (such as elevation, temperature, or coordinate polynomials) vary smoothly across the spatial domain, causing the columns of the design matrix $\mathbf{X}$ to be nearly collinear with the low-frequency eigenvectors of the spatial covariance matrix $\mathbf{V}$ (those corresponding to large spatial eigenvalues).
>
> When this spectral overlap occurs, the statistical model cannot cleanly separate whether regional-scale variations in $\mathbf{Y}$ are driven by the deterministic causal effect of the covariates ($\mathbf{X}\boldsymbol{\beta}$) or by a large-scale realization of the autocorrelated stochastic field ($\delta(\mathbf{s})$). In the likelihood or REML surface, this creates a shallow, ill-conditioned ridge where changes in $\boldsymbol{\beta}$ are almost perfectly counterbalanced by shifts in the spatial sill $\sigma^2$ and range $\phi$. As a consequence:
> 1. The parameter covariance $\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$ inflates dramatically relative to naive OLS, causing fixed-effect confidence intervals to widen substantially.
> 2. Estimated regression coefficients $\widehat{\boldsymbol{\beta}}$ may exhibit severe instability, variance inflation, or counterintuitive sign reversals compared to marginal regressions.
> 3. Restricted Maximum Likelihood (REML) directly mitigates this confounding by projecting the observations onto the error contrast subspace $\mathbf{A}'\mathbf{X} = \mathbf{0}$, filtering out $\mathbf{X}\boldsymbol{\beta}$ entirely before optimizing the covariance parameters $(\sigma^2, \phi, \tau^2)$.

---

### 6.3 Universal Kriging (UK)

#### Beat 1: The Mathematical Anchor (The "Why")

Universal Kriging provides the simultaneous solution to two coupled spatial problems:
1. Optimal linear prediction of the non-stationary field $Y(\mathbf{s}_0)$ at an unsampled location $\mathbf{s}_0 \in D$.
2. Optimal filtering of the unknown drift coefficient vector $\boldsymbol{\beta}$.

#### Definition 6.3.1 (Universal Kriging Predictor)
Let $Y(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \varepsilon(\mathbf{s})$ be the spatial linear model on $D$. Let $\mathbf{s}_0 \in D$ be the target prediction location with known drift basis vector $\mathbf{x}_0 = \mathbf{x}(\mathbf{s}_0) \in \mathbb{R}^p$. Let $\mathbf{v}_0 = (C(\mathbf{s}_0 - \mathbf{s}_1), \dots, C(\mathbf{s}_0 - \mathbf{s}_n))' \in \mathbb{R}^n$ be the point-to-data covariance vector.

The **Universal Kriging (UK) predictor** is a linear combination of the observations:
$$
\widehat{Y}_{\text{UK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Y(\mathbf{s}_i) = \boldsymbol{\lambda}' \mathbf{Y}
$$
such that:
1. **Uniform Unbiasedness:** For all parameter vectors $\boldsymbol{\beta} \in \mathbb{R}^p$:
   $$
   \operatorname{E}\left[\widehat{Y}_{\text{UK}}(\mathbf{s}_0)\right] = \operatorname{E}[Y(\mathbf{s}_0)] \iff \boldsymbol{\lambda}' \mathbf{X} \boldsymbol{\beta} = \mathbf{x}_0' \boldsymbol{\beta}, \quad \forall \boldsymbol{\beta} \in \mathbb{R}^p
   $$
   which is equivalent to the system of $p$ linear equality constraints:
   $$
   \mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0
   $$
2. **Optimality:** The mean-squared prediction error $\operatorname{Var}(Y(\mathbf{s}_0) - \boldsymbol{\lambda}' \mathbf{Y})$ is minimized subject to $\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0$.

#### Theorem 6.3.1 (Universal Kriging System)
The weight vector $\boldsymbol{\lambda} \in \mathbb{R}^n$ and the vector of Lagrange multipliers $\boldsymbol{\mu}_{\text{UK}} \in \mathbb{R}^p$ that minimize the prediction error variance subject to $\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0$ satisfy the augmented linear system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \boldsymbol{\mu}_{\text{UK}} \end{bmatrix} = \begin{bmatrix} \mathbf{v}_0 \\ \mathbf{x}_0 \end{bmatrix}
$$
In expanded block equations:
$$
\begin{cases}
\mathbf{V} \boldsymbol{\lambda} + \mathbf{X} \boldsymbol{\mu}_{\text{UK}} = \mathbf{v}_0 \\
\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0
\end{cases}
$$
The minimized Universal Kriging prediction error variance is:
$$
\sigma_{\text{UK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{v}_0 - \boldsymbol{\mu}_{\text{UK}}' \mathbf{x}_0
$$

*Proof.*
Under the unbiasedness condition $\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0$, the prediction error satisfies:
$$
Y(\mathbf{s}_0) - \widehat{Y}_{\text{UK}}(\mathbf{s}_0) = (\mathbf{x}_0' \boldsymbol{\beta} + \varepsilon(\mathbf{s}_0)) - \boldsymbol{\lambda}' (\mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}) = \varepsilon(\mathbf{s}_0) - \boldsymbol{\lambda}' \boldsymbol{\varepsilon}
$$
The mean-squared prediction error is:
$$
\operatorname{Var}(Y(\mathbf{s}_0) - \widehat{Y}_{\text{UK}}(\mathbf{s}_0)) = \operatorname{Var}(\varepsilon(\mathbf{s}_0) - \boldsymbol{\lambda}' \boldsymbol{\varepsilon}) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{v}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda}
$$
We formulate the Lagrangian with vector Lagrange multiplier $2\boldsymbol{\mu}_{\text{UK}} \in \mathbb{R}^p$:
$$
\mathcal{L}(\boldsymbol{\lambda}, \boldsymbol{\mu}_{\text{UK}}) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{v}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} + 2 \boldsymbol{\mu}_{\text{UK}}' \left(\mathbf{X}' \boldsymbol{\lambda} - \mathbf{x}_0\right)
$$
Differentiating with respect to $\boldsymbol{\lambda}$:
$$
\nabla_{\boldsymbol{\lambda}} \mathcal{L} = -2 \mathbf{v}_0 + 2 \mathbf{V} \boldsymbol{\lambda} + 2 \mathbf{X} \boldsymbol{\mu}_{\text{UK}} = \mathbf{0} \implies \mathbf{V} \boldsymbol{\lambda} + \mathbf{X} \boldsymbol{\mu}_{\text{UK}} = \mathbf{v}_0
$$
Differentiating with respect to $\boldsymbol{\mu}_{\text{UK}}$ recovers the constraints:
$$
\nabla_{\boldsymbol{\mu}_{\text{UK}}} \mathcal{L} = 2(\mathbf{X}' \boldsymbol{\lambda} - \mathbf{x}_0) = \mathbf{0} \implies \mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0
$$
Combining these matrix equations produces the augmented block system.

To evaluate the minimized prediction error variance, pre-multiply $\mathbf{V}\boldsymbol{\lambda} + \mathbf{X}\boldsymbol{\mu}_{\text{UK}} = \mathbf{v}_0$ by $\boldsymbol{\lambda}'$:
$$
\boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} + \boldsymbol{\lambda}' \mathbf{X} \boldsymbol{\mu}_{\text{UK}} = \boldsymbol{\lambda}' \mathbf{v}_0 \implies \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} = \boldsymbol{\lambda}' \mathbf{v}_0 - \mathbf{x}_0' \boldsymbol{\mu}_{\text{UK}}
$$
where we used $\boldsymbol{\lambda}'\mathbf{X} = \mathbf{x}_0'$. Substituting this into the variance expression:
$$
\sigma_{\text{UK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{v}_0 + (\boldsymbol{\lambda}' \mathbf{v}_0 - \mathbf{x}_0' \boldsymbol{\mu}_{\text{UK}}) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{v}_0 - \boldsymbol{\mu}_{\text{UK}}' \mathbf{x}_0
$$
This establishes the theorem. $\blacksquare$

##### Equation Anatomy: The Universal Kriging Augmented Saddle-Point Block System

$$
\begin{bmatrix} \mathbf{V} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \boldsymbol{\mu}_{\text{UK}} \end{bmatrix} = \begin{bmatrix} \mathbf{v}_0 \\ \mathbf{x}_0 \end{bmatrix}
$$

- $\mathbf{V} \in \mathbb{R}^{n \times n}$ (Data-to-Data Spatial Covariance Matrix): The upper-left block containing mutual spatial covariances $[C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$ among all observed sample locations. It accounts for geometric clustering and inter-point correlation among the sample sites, automatically down-weighting redundant clustered observations.
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ (Drift Design Matrix): The upper-right coupling block containing evaluations of the $p$ deterministic drift basis functions at the $n$ sample locations. It couples the optimal prediction weights $\boldsymbol{\lambda}$ with the Lagrange multiplier vector $\boldsymbol{\mu}_{\text{UK}}$.
- $\mathbf{X}' \in \mathbb{R}^{p \times n}$ and $\mathbf{0} \in \mathbb{R}^{p \times p}$ (Constraint Block): The lower row block enforcing the $p$ simultaneous unbiasedness conditions $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{x}_0$. Because the Lagrangian contains no quadratic penalty on the multipliers themselves, the lower-right block is identically a zero matrix $\mathbf{0}_{p \times p}$, conferring the classical saddle-point (Karush-Kuhn-Tucker) block structure.
- $\boldsymbol{\lambda} \in \mathbb{R}^n$ (Universal Kriging Prediction Weight Vector): The $n$ optimal linear weights applied to the observations to predict $Y(\mathbf{s}_0)$. Unlike Ordinary Kriging weights which only sum to 1, UK weights must simultaneously satisfy $p$ moment conditions $\sum_{i=1}^n \lambda_i f_j(\mathbf{s}_i) = f_j(\mathbf{s}_0)$ for $j = 1, \dots, p$.
- $\boldsymbol{\mu}_{\text{UK}} \in \mathbb{R}^p$ (Vector of Drift Lagrange Multipliers): The $p$ shadow costs associated with the $p$ polynomial or external drift unbiasedness constraints. Each entry $\mu_j$ reflects the variance penalty incurred by forcing the predictor to filter out the $j$-th drift basis function.
- $\mathbf{v}_0 \in \mathbb{R}^n$ (Target-to-Data Covariance Vector, also denoted $\mathbf{c}_0$): The cross-covariance vector $[C(\mathbf{s}_0 - \mathbf{s}_1), \dots, C(\mathbf{s}_0 - \mathbf{s}_n)]'$ between the unobserved target site $\mathbf{s}_0$ and all $n$ sample locations, driving the spatial interpolation toward nearby observations.
- $\mathbf{x}_0 = \mathbf{x}(\mathbf{s}_0) \in \mathbb{R}^p$ (Target Drift Basis Vector): The evaluation of the $p$ deterministic basis functions at the prediction point $\mathbf{s}_0$ (e.g., $(1, s_{01}, s_{02})'$ for linear coordinate drift). It anchors the required expectation $\operatorname{E}[\widehat{Y}(\mathbf{s}_0)] = \mathbf{x}_0'\boldsymbol{\beta}$.

#### Theorem 6.3.2 (Decomposition of the Universal Kriging Predictor and Variance)
Let $\widehat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}' \mathbf{V}^{-1} \mathbf{X})^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}$ be the BLUE of the drift parameters.
1. The Universal Kriging predictor decomposes additively into the estimated drift plus the Simple Kriging prediction of the residual field:
   $$
   \widehat{Y}_{\text{UK}}(\mathbf{s}_0) = \mathbf{x}_0' \widehat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{v}_0' \mathbf{V}^{-1} \left(\mathbf{Y} - \mathbf{X} \widehat{\boldsymbol{\beta}}_{\text{GLS}}\right)
   $$
2. The Lagrange multiplier vector is:
   $$
   \boldsymbol{\mu}_{\text{UK}} = -\left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \left(\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0\right)
   $$
3. The Universal Kriging variance decomposes additively into the Simple Kriging variance plus the variance penalty due to drift parameter estimation:
   $$
   \sigma_{\text{UK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \left(\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0\right)' \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \left(\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0\right)
   $$
   where $\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0$.
4. For all locations $\mathbf{s}_0 \in D$:
   $$
   \sigma_{\text{UK}}^2(\mathbf{s}_0) \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)
   $$
   with equality if and only if $\mathbf{x}_0 = \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0$.

*Proof.*
From the first block equation of the UK system:
$$
\mathbf{V} \boldsymbol{\lambda} + \mathbf{X} \boldsymbol{\mu}_{\text{UK}} = \mathbf{v}_0 \implies \boldsymbol{\lambda} = \mathbf{V}^{-1} \mathbf{v}_0 - \mathbf{V}^{-1} \mathbf{X} \boldsymbol{\mu}_{\text{UK}}
$$
Substitute this expression into the constraint equation $\mathbf{X}' \boldsymbol{\lambda} = \mathbf{x}_0$:
$$
\mathbf{X}' \left(\mathbf{V}^{-1} \mathbf{v}_0 - \mathbf{V}^{-1} \mathbf{X} \boldsymbol{\mu}_{\text{UK}}\right) = \mathbf{x}_0 \implies \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0 - \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right) \boldsymbol{\mu}_{\text{UK}} = \mathbf{x}_0
$$
Since $\mathbf{X}$ has full column rank and $\mathbf{V}$ is positive definite, $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ is invertible. Solving for $\boldsymbol{\mu}_{\text{UK}}$:
$$
\boldsymbol{\mu}_{\text{UK}} = -\left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} (\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0)
$$
This establishes assertion (2).

Now substitute $\boldsymbol{\mu}_{\text{UK}}$ back into the expression for $\boldsymbol{\lambda}$:
$$
\boldsymbol{\lambda} = \mathbf{V}^{-1} \mathbf{v}_0 + \mathbf{V}^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} (\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0)
$$
The kriging prediction is $\widehat{Y}_{\text{UK}}(\mathbf{s}_0) = \boldsymbol{\lambda}' \mathbf{Y}$:
$$
\begin{aligned}
\widehat{Y}_{\text{UK}}(\mathbf{s}_0) &= \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{Y} + (\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0)' \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y} \\
&= \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{Y} + (\mathbf{x}_0' - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{X}) \widehat{\boldsymbol{\beta}}_{\text{GLS}} \\
&= \mathbf{x}_0' \widehat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{v}_0' \mathbf{V}^{-1} (\mathbf{Y} - \mathbf{X} \widehat{\boldsymbol{\beta}}_{\text{GLS}})
\end{aligned}
$$
This proves assertion (1).

To evaluate the prediction variance, use Theorem 6.3.1: $\sigma_{\text{UK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{v}_0 - \mathbf{x}_0' \boldsymbol{\mu}_{\text{UK}}$.
Evaluate $\boldsymbol{\lambda}'\mathbf{v}_0$:
$$
\boldsymbol{\lambda}' \mathbf{v}_0 = \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0 + (\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0)' \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0
$$
Substitute $\boldsymbol{\lambda}'\mathbf{v}_0$ and $\boldsymbol{\mu}_{\text{UK}}$:
$$
\begin{aligned}
\sigma_{\text{UK}}^2(\mathbf{s}_0) &= C(\mathbf{0}) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0 - (\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0)' \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0 \\
&\quad + \mathbf{x}_0' \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} (\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0) \\
&= \sigma_{\text{SK}}^2(\mathbf{s}_0) + (\mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0)' \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \left[ \mathbf{x}_0 - \mathbf{X}' \mathbf{V}^{-1} \mathbf{v}_0 \right]
\end{aligned}
$$
This proves assertion (3).

Finally, because $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$ is strictly positive definite, the quadratic form in (3) is strictly non-negative, vanishing if and only if $\mathbf{x}_0 = \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0$. Thus $\sigma_{\text{UK}}^2(\mathbf{s}_0) \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)$. $\blacksquare$

```
Universal Kriging Variance Decomposition:
========================================================================================
Total UK Variance:        σ_UK^2(s_0) = σ_SK^2(s_0) + Var(x_0' β_GLS - v_0' V^-1 X β_GLS)
                                            |                         |
                                            v                         v
                                  Residual Uncertainty        Drift Estimation Error
                              (Uncertainty if β was known)  (Penalty for estimating β)
========================================================================================
```

#### Beat 2: Algorithmic Translation (The "How")

Universal Kriging can be implemented either by directly solving the $(n+p) \times (n+p)$ augmented block system, or by the equivalent two-stage procedure: GLS drift estimation followed by Simple Kriging on the residuals. The solver below implements the augmented block system.

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
    mu_uk : list of float
        Lagrange multiplier vector mu_UK.
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
    # [ v_0 ]
    # [ x_0 ]
    b = [0.0] * m
    for i in range(n):
        dist = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(d)) ** 0.5
        b[i] = cov_kernel(dist)
    for j in range(p):
        b[n + j] = basis_functions[j](target_coord)

    # Step 3: Solve the linear system A * [lambda, mu]' = b
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
    mu_uk = solution[n:]

    # Step 4: Evaluate the Universal Kriging prediction
    y_hat = sum(weights[i] * sample_vals[i] for i in range(n))

    # Step 5: Evaluate the Universal Kriging variance
    # sigma_UK^2(s_0) = C(0) - sum lambda_i * v_0[i] - sum mu_j * x_0[j]
    c0 = cov_kernel(0.0)
    term1 = sum(weights[i] * b[i] for i in range(n))
    term2 = sum(mu_uk[j] * b[n + j] for j in range(p))
    uk_var = max(0.0, c0 - term1 - term2)

    return y_hat, uk_var, weights, mu_uk
```

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **The Price of Estimating Drift:** The variance decomposition proves that $\sigma_{\text{UK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \operatorname{Var}(\text{drift penalty})$. If $\boldsymbol{\beta}$ were known with certainty, the prediction uncertainty would be $\sigma_{\text{SK}}^2(\mathbf{s}_0)$. Estimating $\boldsymbol{\beta}$ from noisy data inflates the error variance.
- **Polynomial Runaway (Runge's Phenomenon):** When coordinate polynomials of degree $k \ge 2$ (quadratic or cubic drift) are employed, the design matrix $\mathbf{X}$ extrapolates rapidly outside the spatial convex hull of the data. The drift variance penalty $(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)'(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)$ explodes cubically or quintically with distance from the boundary. Universal Kriging maps with high-order coordinate drift exhibit severe boundary edge oscillations (ringing). Consequently, linear drift ($p = 1 + d$) is strongly preferred over higher-order polynomials.

---

### 6.4 Kriging with External Drift (KED)

#### Beat 1: The Mathematical Anchor (The "Why")

While classical Universal Kriging uses geometric coordinates as basis functions, **Kriging with External Drift (KED)** replaces abstract polynomials with continuous auxiliary environmental or geophysical fields.

#### Definition 6.4.1 (External Drift Model)
Let $\{Y(\mathbf{s}) : \mathbf{s} \in D\}$ be a target variable (e.g., precipitation, sub-surface porosity). Let $\{q_1(\mathbf{s}), \dots, q_m(\mathbf{s})\}$ be $m$ deterministic auxiliary spatial fields (e.g., digital elevation model, seismic acoustic impedance). The KED model specifies:
$$
Y(\mathbf{s}) = \beta_0 + \sum_{k=1}^m \beta_k q_k(\mathbf{s}) + \varepsilon(\mathbf{s}) = \mathbf{x}(\mathbf{s})' \boldsymbol{\beta} + \varepsilon(\mathbf{s})
$$
where $\mathbf{x}(\mathbf{s}) = (1, q_1(\mathbf{s}), \dots, q_m(\mathbf{s}))' \in \mathbb{R}^{m+1}$, and $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_m)' \in \mathbb{R}^{m+1}$.

#### Structural Regularity Conditions for KED:
1. **Exhaustive Availability:** The auxiliary fields $q_k(\mathbf{s})$ must be known at all sample locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ and at every target prediction site $\mathbf{s}_0 \in D$.
2. **Linear Independence and Non-Collinearity:** The augmented design matrix:
   $$
   \mathbf{X} = \begin{bmatrix} 1 & q_1(\mathbf{s}_1) & \cdots & q_m(\mathbf{s}_1) \\ \vdots & \vdots & \ddots & \vdots \\ 1 & q_1(\mathbf{s}_n) & \cdots & q_m(\mathbf{s}_n) \end{bmatrix} \in \mathbb{R}^{n \times (m+1)}
   $$
   must satisfy $\operatorname{rank}(\mathbf{X}) = m+1 \le n$.
3. **Linearity of the Physical Relation:** The conditional expectation $\operatorname{E}[Y(\mathbf{s}) \mid q_1(\mathbf{s}), \dots, q_m(\mathbf{s})]$ must be strictly linear in the coefficients. Non-linear relationships require non-linear anamorphosis prior to kriging.
4. **Distinction from Cokriging:** In KED, the auxiliary field $q(\mathbf{s})$ is treated as a deterministic external trend without cross-covariance modeling. In Cokriging (Chapter 7), the secondary variable is modeled as a joint stochastic random field with spatial cross-covariances.

#### Beat 2: Algorithmic Translation (The "How")

The mathematical equations for KED are identical to the Universal Kriging equations of Theorem 6.3.1, with $\mathbf{x}(\mathbf{s}) = (1, q_1(\mathbf{s}), \dots, q_m(\mathbf{s}))'$. 

Procedural requirements for KED:
1. **Collinearity Check:** Compute the condition number of the auxiliary data $\kappa(\mathbf{X}'\mathbf{X})$. If $\kappa > 10^5$, apply ridge regularization or eliminate collinear covariates.
2. **Local Variance Verification:** When using local moving neighborhoods $\mathcal{S}(\mathbf{s}_0)$, compute the sample variance of the covariate within the local neighborhood:
   $$
   s_q^2 = \frac{1}{n_0 - 1} \sum_{i \in \mathcal{S}(\mathbf{s}_0)} (q(\mathbf{s}_i) - \bar{q})^2
   $$
   If $s_q^2 < \epsilon$ (the covariate is nearly constant across the search window), the local design matrix becomes rank-deficient, and KED must automatically revert to Ordinary Kriging.

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Why KED Fails Under Flat Covariates:** If an auxiliary variable $q(\mathbf{s})$ represents elevation, and a local moving neighborhood falls entirely within a flat valley floor or lake surface, $q(\mathbf{s}_i) \approx \bar{q}$ for all points in the neighborhood. The second column of $\mathbf{X}$ becomes a scalar multiple of the first column of ones: $\mathbf{q} \approx \bar{q} \mathbf{1}$. Consequently, $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ is singular, the augmented KED matrix cannot be inverted, and the kriging system blows up.
- **Extrapolation Instability:** If the target location $\mathbf{s}_0$ possesses an auxiliary value $q(\mathbf{s}_0)$ that lies far outside the range of values observed at sample sites ($\min_i q(\mathbf{s}_i) \le q(\mathbf{s}_0) \le \max_i q(\mathbf{s}_i)$ is violated), the drift prediction $\mathbf{x}_0'\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ extrapolates linearly into unphysical territory, and the prediction variance explodes.

---

### 6.5 Dual Kriging and Spline Equivalence in RKHS

#### Beat 1: The Mathematical Anchor (The "Why")

In the **primal formulation** of Universal Kriging, for each unsampled prediction site $\mathbf{s}_0$, an $(n+p) \times (n+p)$ linear system must be assembled and solved to find $\boldsymbol{\lambda}(\mathbf{s}_0)$ and $\boldsymbol{\mu}_{\text{UK}}(\mathbf{s}_0)$. For a regional grid with $10^6$ nodes, this requires $10^6$ linear system solves.

The **Dual Kriging formulation** reformulates the prediction problem: the linear system is solved **once** for a fixed set of data coefficients $\mathbf{b} \in \mathbb{R}^n$ and trend coefficients $\mathbf{d} \in \mathbb{R}^p$. Once computed, the kriging prediction at *any* site $\mathbf{s}$ is evaluated via a single inner product. Furthermore, Dual Kriging reveals the deep mathematical equivalence between geostatistical kriging and variational spline interpolation in Reproducing Kernel Hilbert Spaces (RKHS).

#### Theorem 6.5.1 (Dual Kriging Representation)
Let $\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$ be the spatial linear model on sample sites $\mathbf{s}_1, \dots, \mathbf{s}_n$.
1. The Universal Kriging interpolating surface can be expressed identically as:
   $$
   \widehat{Y}(\mathbf{s}) = \sum_{j=1}^p d_j f_j(\mathbf{s}) + \sum_{i=1}^n b_i C(\mathbf{s} - \mathbf{s}_i) = \mathbf{x}(\mathbf{s})' \mathbf{d} + \mathbf{v}(\mathbf{s})' \mathbf{b}
   $$
   where $\mathbf{d} = (d_1, \dots, d_p)' \in \mathbb{R}^p$ and $\mathbf{b} = (b_1, \dots, b_n)' \in \mathbb{R}^n$ are deterministic vectors that are **completely independent of $\mathbf{s}$**.
2. The coefficient vectors $\mathbf{b}$ and $\mathbf{d}$ are the unique solution to the **Dual Kriging linear system**:
   $$
   \begin{bmatrix} \mathbf{V} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{b} \\ \mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{Y} \\ \mathbf{0} \end{bmatrix}
   $$
3. The coefficients satisfy the explicit relations:
   $$
   \mathbf{d} = \widehat{\boldsymbol{\beta}}_{\text{GLS}} = \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}
   $$
   $$
   \mathbf{b} = \mathbf{V}^{-1} \left(\mathbf{Y} - \mathbf{X} \widehat{\boldsymbol{\beta}}_{\text{GLS}}\right)
   $$
   and satisfy the discrete orthogonality condition $\mathbf{X}' \mathbf{b} = \mathbf{0}$.

*Proof.*
By Theorem 6.3.2, the primal UK predictor at an arbitrary location $\mathbf{s} \in D$ is:
$$
\widehat{Y}(\mathbf{s}) = \mathbf{x}(\mathbf{s})' \widehat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{v}(\mathbf{s})' \mathbf{V}^{-1} \left(\mathbf{Y} - \mathbf{X} \widehat{\boldsymbol{\beta}}_{\text{GLS}}\right)
$$
Define the vectors:
$$
\mathbf{d} = \widehat{\boldsymbol{\beta}}_{\text{GLS}} \in \mathbb{R}^p
$$
$$
\mathbf{b} = \mathbf{V}^{-1} \left(\mathbf{Y} - \mathbf{X} \widehat{\boldsymbol{\beta}}_{\text{GLS}}\right) = \mathbf{V}^{-1} (\mathbf{Y} - \mathbf{X}\mathbf{d}) \in \mathbb{R}^n
$$
Substituting $\mathbf{d}$ and $\mathbf{b}$ into the equation gives:
$$
\widehat{Y}(\mathbf{s}) = \mathbf{x}(\mathbf{s})' \mathbf{d} + \mathbf{v}(\mathbf{s})' \mathbf{b}
$$
Now verify that $(\mathbf{b}', \mathbf{d}')'$ satisfies the block system. From the definition of $\mathbf{b}$:
$$
\mathbf{V} \mathbf{b} = \mathbf{Y} - \mathbf{X}\mathbf{d} \implies \mathbf{V} \mathbf{b} + \mathbf{X}\mathbf{d} = \mathbf{Y}
$$
This establishes the top block equation.
Pre-multiply this equation by $\mathbf{X}'\mathbf{V}^{-1}$:
$$
\mathbf{X}' \mathbf{V}^{-1} (\mathbf{V} \mathbf{b} + \mathbf{X}\mathbf{d}) = \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y} \implies \mathbf{X}' \mathbf{b} + \left(\mathbf{X}' \mathbf{V}^{-1} \mathbf{X}\right) \mathbf{d} = \mathbf{X}' \mathbf{V}^{-1} \mathbf{Y}
$$
Since $\mathbf{d} = \widehat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Y}$, we have $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})\mathbf{d} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{Y}$. Subtracting this from both sides yields:
$$
\mathbf{X}' \mathbf{b} = \mathbf{0}
$$
Assembling these equations establishes the dual block system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{b} \\ \mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{Y} \\ \mathbf{0} \end{bmatrix}
$$
This completes the proof. $\blacksquare$

#### 6.5.1 Equivalence with Thin-Plate Splines and Reproducing Kernel Hilbert Spaces (RKHS)

The Dual Kriging representation establishes a direct mathematical isomorphism between statistical geostatistics and variational approximation theory.

Consider the classical thin-plate spline variational problem in $\mathbb{R}^2$: find an interpolating surface $f: \mathbb{R}^2 \to \mathbb{R}$ in the Beppo-Levi Sobolev space $D^{-2} L^2(\mathbb{R}^2)$ that minimizes the bending energy:
$$
J(f) = \int_{\mathbb{R}^2} \left[ \left(\frac{\partial^2 f}{\partial x_1^2}\right)^2 + 2\left(\frac{\partial^2 f}{\partial x_1 \partial x_2}\right)^2 + \left(\frac{\partial^2 f}{\partial x_2^2}\right)^2 \right] d\mathbf{x}
$$
subject to exact interpolation at sample locations: $f(\mathbf{s}_i) = Y(\mathbf{s}_i)$ for $i = 1, \dots, n$.

By the Duchon-Meinguet theorem of surface splines, the unique variational minimizer is:
$$
f(\mathbf{s}) = d_0 + d_1 s_1 + d_2 s_2 + \sum_{i=1}^n b_i K(\|\mathbf{s} - \mathbf{s}_i\|)
$$
where $K(h) = h^2 \log h$ is the biharmonic Green's function in $\mathbb{R}^2$ ($\Delta^2 K = \delta_0$), and the coefficients satisfy $\sum_{i=1}^n b_i = 0$, $\sum_{i=1}^n b_i s_{i,1} = 0$, and $\sum_{i=1}^n b_i s_{i,2} = 0$.

This variational spline solution is **identically equal to Dual Universal Kriging** with:
- Linear drift basis: $f_0(\mathbf{s}) = 1, f_1(\mathbf{s}) = s_1, f_2(\mathbf{s}) = s_2$ ($p = 3$).
- Generalized covariance kernel: $K(h) = h^2 \log h$.
- Orthogonality constraints: $\mathbf{X}'\mathbf{b} = \mathbf{0}$.

In a Reproducing Kernel Hilbert Space $\mathcal{H}_K$ equipped with kernel $K$, the representer theorem guarantees that the optimal interpolator is a finite linear combination of kernel translates $\sum b_i K(\cdot, \mathbf{s}_i)$ plus an element of the null space of the regularization operator. Kriging and splines are two faces of the same mathematical coin.

#### Beat 2: Algorithmic Translation (The "How")

The Dual Kriging algorithm solves the dual system once to obtain $\mathbf{b}$ and $\mathbf{d}$, and then evaluates predictions across an arbitrary grid in $O(n)$ operations per grid point.

```python
def dual_kriging_fit(sample_coords, sample_vals, basis_functions, cov_kernel):
    """
    Fits Dual Kriging coefficients (b, d) by solving the dual linear system.

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
    d_weights : list of float, shape (p,)
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

    # Step 3: Solve the dual linear system A * [b, d]' = rhs
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
    d_weights = solution[n:]

    return b_weights, d_weights


def dual_kriging_eval(target_coord, sample_coords, b_weights, d_weights, 
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
    d_weights : list of float, shape (p,)
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

    # Trend component: x(s)' * d
    trend_comp = sum(d_weights[j] * basis_functions[j](target_coord) for j in range(p))

    # Stochastic / kernel component: v(s)' * b = sum b_i * C(s - s_i)
    kernel_comp = 0.0
    for i in range(n):
        dist = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(dim)) ** 0.5
        kernel_comp += b_weights[i] * cov_kernel(dist)

    return trend_comp + kernel_comp
```

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Computational Speedup:** Evaluating $10^6$ grid points with Primal Universal Kriging requires $10^6$ matrix solves ($O(10^6 \cdot n^3)$). Dual Kriging requires one single solve ($O(n^3)$), followed by $10^6$ vector dot-products ($O(10^6 \cdot n)$), achieving a speedup of several orders of magnitude.
- **Physical Bending Analogy:** In the thin-plate spline interpretation, the coefficients $b_i$ represent discrete point forces (loads) applied at sample points $\mathbf{s}_i$, while $\mathbf{d}$ represents rigid body translations and rotations. The condition $\sum b_i = 0$ enforces vertical static force balance, while $\sum b_i s_{i,k} = 0$ enforces static moment balance.

---

### 6.6 Intrinsic Random Functions of Order $k$ (IRF-$k$)

#### Beat 1: The Mathematical Anchor (The "Why")

When a spatial field exhibits a non-stationary polynomial trend of degree $k$, its global mean $\mu(\mathbf{s})$ is not stationary, and its covariance is undefined. Ordinary Kriging filters out a constant mean ($k = 0$) by working with differences $Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})$ whose weights sum to zero. 

Georges Matheron generalized this concept to arbitrary polynomial drifts of degree $k \ge 0$, creating the theory of **Intrinsic Random Functions of order $k$ (IRF-$k$)**. Instead of estimating the drift coefficients $\boldsymbol{\beta}$ (which depends on the choice of spatial origin), the IRF-$k$ framework filters out all polynomials of degree $\le k$ via **authorized linear combinations**.

#### Definition 6.6.1 (Authorized Linear Combinations of Order $k$)
Let $\mathcal{P}_k$ denote the vector space of real polynomials on $\mathbb{R}^d$ of degree at most $k$:
$$
\mathcal{P}_k = \left\{ p(\mathbf{s}) = \sum_{|\boldsymbol{\alpha}| \le k} c_{\boldsymbol{\alpha}} \mathbf{s}^{\boldsymbol{\alpha}} \right\}
$$
A discrete measure $\lambda = \sum_{i=1}^n \lambda_i \delta_{\mathbf{s}_i}$ (or linear combination of random variables $\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$) is an **Authorized Linear Combination of order $k$ (ALC-$k$)** if it annihilates all polynomials in $\mathcal{P}_k$:
$$
\sum_{i=1}^n \lambda_i p(\mathbf{s}_i) = 0, \quad \forall p \in \mathcal{P}_k
$$
In matrix terms, if $\mathbf{X}_k \in \mathbb{R}^{n \times \dim(\mathcal{P}_k)}$ is the polynomial design matrix, an ALC-$k$ satisfies:
$$
\mathbf{X}_k' \boldsymbol{\lambda} = \mathbf{0}
$$

#### Definition 6.6.2 (Intrinsic Random Function of Order $k$)
A spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is an **Intrinsic Random Function of order $k$ (IRF-$k$)** if for any ALC-$k$ measure $\lambda$, the random variable $Z(\lambda) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$ is second-order stationary:
1. $\operatorname{E}[Z(\lambda)] = 0$.
2. $\operatorname{Var}(Z(\lambda))$ is invariant under spatial translations of the sample configuration.

#### Definition 6.6.3 (Generalized Covariance Kernel)
A continuous function $K: \mathbb{R}^d \to \mathbb{R}$ is a **generalized covariance** for an IRF-$k$ if for every ALC-$k$ $\lambda = \sum_{i=1}^n \lambda_i \delta_{\mathbf{s}_i}$:
$$
\operatorname{Var}\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) = \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j K(\mathbf{s}_i - \mathbf{s}_j) \ge 0
$$
The function $K(\mathbf{h})$ is defined only up to an even polynomial of degree $\le 2k$. Valid isotropic polynomial generalized covariances in $\mathbb{R}^d$ have the form:
$$
K(h) = c_0 \delta(h) + \sum_{m=0}^k (-1)^{m+1} b_m h^{2m+1}
$$
For instance:
- For $k = 0$ (Intrinsic stationarity): $K(h) = -b_0 h = -\gamma(h)$ (linear variogram).
- For $k = 1$ (Linear drift filtering): $K(h) = -b_0 h + b_1 h^3$ (cubic generalized covariance).
- For $k = 2$ (Quadratic drift filtering): $K(h) = -b_0 h + b_1 h^3 - b_2 h^5$ (spline generalized covariance).

#### Theorem 6.6.1 (IRF-$k$ Kriging System)
Let $Z(\mathbf{s})$ be an IRF-$k$ with generalized covariance $K(\mathbf{h})$. The optimal predictor $\widehat{Z}(\mathbf{s}_0) = \boldsymbol{\lambda}'\mathbf{Z}$ satisfies the linear system:
$$
\begin{bmatrix} \mathbf{K} & \mathbf{X}_k \\ \mathbf{X}_k' & \mathbf{0} \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \boldsymbol{\mu} \end{bmatrix} = \begin{bmatrix} \mathbf{k}_0 \\ \mathbf{x}_0 \end{bmatrix}
$$
where $\mathbf{K} = [K(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$, $\mathbf{k}_0 = (K(\mathbf{s}_0 - \mathbf{s}_1), \dots, K(\mathbf{s}_0 - \mathbf{s}_n))'$, and $\mathbf{X}_k$ contains the polynomial basis evaluations up to degree $k$.
The prediction error variance is:
$$
\sigma_{\text{IRF}}^2(\mathbf{s}_0) = K(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{k}_0 - \boldsymbol{\mu}' \mathbf{x}_0
$$

*Proof.*
The prediction error $Z(\mathbf{s}_0) - \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$ can be written as $\sum_{i=0}^n w_i Z(\mathbf{s}_i)$ with $w_0 = 1$ and $w_i = -\lambda_i$. The condition $\mathbf{X}_k'\boldsymbol{\lambda} = \mathbf{x}_0$ ensures that $\sum_{i=0}^n w_i p(\mathbf{s}_i) = p(\mathbf{s}_0) - \sum_{i=1}^n \lambda_i p(\mathbf{s}_i) = 0$ for all $p \in \mathcal{P}_k$.
Thus, the prediction error is an authorized linear combination of order $k$. Its variance is well-defined by the generalized covariance $K(\mathbf{h})$, and minimizing this quadratic form subject to $\mathbf{X}_k'\boldsymbol{\lambda} = \mathbf{x}_0$ yields the augmented system. $\blacksquare$

##### Equation Anatomy: The IRF-$k$ Generalized Covariance and Prediction System

$$
\operatorname{Var}\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) = \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j K(\mathbf{s}_i - \mathbf{s}_j), \quad \text{subject to} \quad \sum_{i=1}^n \lambda_i p(\mathbf{s}_i) = 0 \quad \forall p \in \mathcal{P}_k
$$

- $\sum_{i=1}^n \lambda_i p(\mathbf{s}_i) = 0 \quad \forall p \in \mathcal{P}_k$ (The Polynomial Annihilation Condition): The defining structural constraint of an Authorized Linear Combination of order $k$ (ALC-$k$). It filters out all polynomials up to degree $k$ without ever estimating their unknown coefficients $\boldsymbol{\beta}$. For $k = 0$, this requires $\sum_{i=1}^n \lambda_i = 0$ (filtering a constant mean); for $k = 1$, it simultaneously filters out constant and linear spatial gradients ($\sum_{i=1}^n \lambda_i = 0$ and $\sum_{i=1}^n \lambda_i \mathbf{s}_i = \mathbf{0}$).
- $K(\mathbf{s}_i - \mathbf{s}_j) = K(\mathbf{h})$ (The Generalized Covariance Kernel): A conditionally positive definite function of order $k$. Unlike stationary covariance functions $C(\mathbf{h})$ which must be bounded by $C(\mathbf{0}) = \sigma^2$, $K(\mathbf{h})$ can grow unboundedly with lag distance $h = \|\mathbf{h}\|$ (e.g., $K(h) = -b_0 h + b_1 h^3$). It is defined only modulo an even polynomial of degree $\le 2k$, which vanishes identically when evaluated under the double sum of ALC-$k$ weights.
- $\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j K(\mathbf{s}_i - \mathbf{s}_j)$ (The Intrinsic Increment Variance): The strictly non-negative variance of the filtered spatial increment. Despite $K(\mathbf{h})$ potentially taking negative values or unbounded magnitudes, the conditional positive definiteness on the subspace of ALC-$k$ weights guarantees $\boldsymbol{\lambda}'\mathbf{K}\boldsymbol{\lambda} \ge 0$.
- $\begin{bmatrix} \mathbf{K} & \mathbf{X}_k \\ \mathbf{X}_k' & \mathbf{0} \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \boldsymbol{\mu} \end{bmatrix} = \begin{bmatrix} \mathbf{k}_0 \\ \mathbf{x}_0 \end{bmatrix}$ (The IRF-$k$ Linear Prediction System): The augmented block system governing optimal interpolation. Here $\mathbf{K} = [K(\mathbf{s}_i - \mathbf{s}_j)] \in \mathbb{R}^{n \times n}$ is the generalized covariance matrix, $\mathbf{X}_k \in \mathbb{R}^{n \times \dim(\mathcal{P}_k)}$ contains polynomial basis evaluations, and $\mathbf{k}_0 = [K(\mathbf{s}_0 - \mathbf{s}_1), \dots, K(\mathbf{s}_0 - \mathbf{s}_n)]'$ is the generalized point-to-data covariance vector.

#### Beat 2: Algorithmic Translation (The "How")

The IRF-$k$ algorithm constructs an augmented block system with generalized covariance values $K(h)$ and polynomial constraints. Because $K(h)$ can grow unboundedly with $h$ (e.g., $K(h) \propto h^3$), the matrix entries are not bounded, but the authorized linear combination guarantees that the resulting variance $\sigma_{\text{IRF}}^2(\mathbf{s}_0)$ is strictly non-negative.

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Origin Invariance:** In Universal Kriging, the estimated drift parameter vector $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ changes if the spatial origin is shifted ($\mathbf{s} \mapsto \mathbf{s} - \mathbf{s}_c$). In IRF-$k$, drift coefficients are never estimated; the authorized linear combinations automatically eliminate all polynomials of degree $\le k$, making predictions strictly invariant under affine transformations of the coordinate system.

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

#### Exercise 6.3 (Kriging with External Drift Stability and Collinearity)
Suppose a target variable $Y$ is predicted using Kriging with External Drift with a single auxiliary covariate $q(s)$:
$$
Y(s) = \beta_0 + \beta_1 q(s) + \varepsilon(s)
$$
Observations are taken at 3 locations in a local neighborhood: $s_1, s_2, s_3$.
- **(a)** Write down the design matrix $\mathbf{X} \in \mathbb{R}^{3 \times 2}$.
- **(b)** Suppose the auxiliary covariate has values $q(s_1) = 100$, $q(s_2) = 100.01$, $q(s_3) = 100$. Compute the rank and condition number of $\mathbf{X}'\mathbf{X}$.
- **(c)** Explain what happens to the augmented KED matrix and the prediction variance when $q(s)$ is nearly constant, and formulate an algorithmic remedy.

*Solution.*
- **(a)** The design matrix is:
  $$
  \mathbf{X} = \begin{bmatrix} 1 & q(s_1) \\ 1 & q(s_2) \\ 1 & q(s_3) \end{bmatrix}
  $$

- **(b)** With the given values:
  $$
  \mathbf{X} = \begin{bmatrix} 1 & 100 \\ 1 & 100.01 \\ 1 & 100 \end{bmatrix}
  $$
  Compute $\mathbf{X}'\mathbf{X}$:
  $$
  \mathbf{X}'\mathbf{X} = \begin{bmatrix} 1+1+1 & 100+100.01+100 \\ 100+100.01+100 & 100^2 + 100.01^2 + 100^2 \end{bmatrix} = \begin{bmatrix} 3 & 300.01 \\ 300.01 & 30002.0001 \end{bmatrix}
  $$
  The determinant is:
  $$
  \det(\mathbf{X}'\mathbf{X}) = 3(30002.0001) - (300.01)^2 = 90006.0003 - 90006.0001 = 0.0002
  $$
  The trace is $\operatorname{tr}(\mathbf{X}'\mathbf{X}) \approx 30005$.
  The eigenvalues satisfy $\lambda_1 \approx 30005$ and $\lambda_2 = \frac{\det}{\lambda_1} \approx \frac{0.0002}{30005} \approx 6.66 \times 10^{-9}$.
  The condition number is:
  $$
  \kappa(\mathbf{X}'\mathbf{X}) = \frac{\lambda_1}{\lambda_2} \approx \frac{30005}{6.66 \times 10^{-9}} \approx 4.5 \times 10^{12}
  $$
  The matrix is severely ill-conditioned.

- **(c)** In the KED system, the drift estimation variance contains $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$.
  Because $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ is proportional to $\mathbf{X}'\mathbf{X}$, its determinant is near zero:
  $$
  \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1} \approx \frac{1}{\det} \operatorname{adj}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})
  $$
  As $\det \to 0$, the inverse explodes toward infinity. Consequently:
  1. The weights $\boldsymbol{\lambda}$ oscillate wildly between large positive and large negative numbers ($|\lambda_i| > 10^6$).
  2. The prediction variance $\sigma_{\text{KED}}^2(s_0) = \sigma_{\text{SK}}^2(s_0) + (\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)'(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}(\mathbf{x}_0 - \mathbf{X}'\mathbf{V}^{-1}\mathbf{v}_0)$ explodes to catastrophic values.
  
  **Algorithmic Remedy:**
  Before assembling the local KED system, evaluate the variance of $q$ within the moving neighborhood:
  $$
  \operatorname{Var}_{\text{local}}(q) = \frac{1}{n_0 - 1} \sum_{i=1}^{n_0} (q(\mathbf{s}_i) - \bar{q})^2
  $$
  If $\operatorname{Var}_{\text{local}}(q) < \tau_{\text{thresh}}$ (e.g., $10^{-4}$ of global variance), dynamically drop the external drift term and fall back to Ordinary Kriging with design vector $\mathbf{x}(\mathbf{s}) = [1]$. $\blacksquare$

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
