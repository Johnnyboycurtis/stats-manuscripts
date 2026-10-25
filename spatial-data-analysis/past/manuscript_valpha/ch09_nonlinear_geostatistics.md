# Chapter 9: Non-Gaussian and Nonlinear Spatial Estimation

This chapter establishes the mathematical theory of non-Gaussian and nonlinear spatial estimation. In environmental sciences, mining engineering, and hydrology, physical attributes frequently violate Gaussian assumptions: mineral grades exhibit extreme positive skewness, pollutant distributions concentrate along complex plume geometries, and spatial permeability spans multiple orders of magnitude. 

While Best Linear Unbiased Prediction (Ordinary and Simple Kriging) provides the optimal linear predictor in $L^2(\Omega, \mathcal{F}, P)$, it is optimal among *all* predictors only when the spatial random field is Gaussian. For non-Gaussian fields, spatial threshold exceedances, and non-additive physical attributes, linear combinations of data are fundamentally suboptimal. Following the pedagogical architecture of Maria L. Rizzo and the semantic linear-model discipline of Oliver Schabenberger & Carol A. Gotway, this chapter establishes the foundations of nonlinear spatial prediction: lognormal random fields and exact MGF back-transformation, indicator geostatistics and threshold destructuration, isotonic regression via the Pool-Adjacent-Violators Algorithm (PAVA), and Gaussian anamorphosis via Hermite orthogonal polynomial expansions and Disjunctive Kriging.

---

### 9.1 Classical-to-Spatial Bridge: Non-Gaussian Fields, Threshold Exceedances, and the Limits of Linear Kriging

#### Beat 1: The Mathematical Anchor (The "Why")

##### Classical-to-Spatial Bridge: From Classical Binary Probit / Logistic Regression and Polynomial Series to Indicator & Disjunctive Geostatistics

In classical regression analysis, modeling the probability that a response variable $Y$ exceeds or falls below a fixed threshold $u$ cannot be handled adequately by standard ordinary least squares (OLS) linear regression. If one fits a linear probability model $\operatorname{E}[I(u) \mid X] = \beta_0 + \beta_1 X$, predictions routinely fall outside the admissible probability interval $[0, 1]$, and the error variance $P(Y \le u)(1 - P(Y \le u))$ is intrinsically heteroskedastic. 

Classical generalized linear models (GLMs) resolve this dilemma via binary probit or logistic regression:
$$
P(Y \le u \mid X) = \Phi(\beta_0 + \beta_1 X) \quad \text{or} \quad P(Y \le u \mid X) = \frac{1}{1 + \exp\left(-(\beta_0 + \beta_1 X)\right)}
$$
where the non-linear link function guarantees that predicted probabilities are monotonically constrained within $[0, 1]$. Similarly, when modeling complex non-linear response curves $f(Y)$, classical statistics expands the function into an orthogonal polynomial series—such as Hermite polynomials $H_k(y)$ under standard Gaussian measures:
$$
f(Y) = \sum_{k=0}^\infty c_k H_k(Y), \quad c_k = \frac{1}{k!} \int_{-\infty}^\infty f(y) H_k(y) \phi(y) dy
$$
yielding optimal non-linear approximations $\widehat{f(Y)} = \sum_{k=0}^K c_k H_k(Y)$.

```
Classical Binary Probit / Logistic Regression
   P(Y <= u | X)
      ^
    1 +------------------------/--_---- Upper asymptote = 1
      |                       /
      |                      /  Link Function: \Phi(\beta_0 + \beta_1 X)
      |                    _-
    0 +----_--------------+------------ Lower asymptote = 0
      +-------------------+------------> Predictor X

Spatial Indicator & Disjunctive Geostatistics
   CCDF F(u | Z) = P(Z(s_0) <= u | Z)
      ^
    1 +                *-----------+--- Kriged indicator cCDF
      |               /
      |              *   Disjunctive Hermite Expansion:
      |             /    Z(s) = \sum \phi_k H_k(Y(s))
    0 +------------*-------------------> Threshold u
```

In spatial statistics, physical attributes frequently violate Gaussian assumptions: mineral deposit grades exhibit heavy-tailed lognormal skewness, pollutant plumes exhibit zero-inflated spatial spikes, and aquifer transmissivity spans multiple orders of magnitude. When sample data $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ are heavily skewed, Best Linear Unbiased Prediction (Ordinary and Simple Kriging) suffers from two fundamental pathologies:
1. **Outlier Distortion:** Extreme sample values exert severe lever-arm effects on linear kriging weights, artificially inflating estimated grades across entire spatial neighborhoods.
2. **The Non-Linear Plug-in Failure:** In regulatory and resource engineering, the decision criterion is almost never the point value $Z(\mathbf{s}_0)$ itself, but a non-linear threshold exceedance or economic transfer function:
   $$
   \psi(Z(\mathbf{s}_0)) = \mathbb{I}\{Z(\mathbf{s}_0) \le u\} \quad \text{or} \quad \psi(Z(\mathbf{s}_0)) = \max(0, Z(\mathbf{s}_0) - u)
   $$
   Plugging linear kriging predictions into non-linear functions $\psi(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))$ fails catastrophically due to Jensen's inequality and the smoothing deficit of kriging.

To bridge classical binary and polynomial regression to continuous spatial fields:
- **Indicator Geostatistics:** We define the spatial indicator random variable $I(\mathbf{s}; u) = \mathbb{I}(Z(\mathbf{s}) \le u)$ at cut-off threshold $u$. Kriging these binary indicator variables directly estimates the spatial conditional cumulative distribution function (CCDF):
  $$
  F(u \mid \mathbf{Z}) = P\left(Z(\mathbf{s}_0) \le u \mid \mathbf{Z}\right) = \operatorname{E}\left[I(\mathbf{s}_0; u) \mid \mathbf{Z}\right]
  $$
  bypassing all parametric distribution assumptions and providing full non-parametric spatial uncertainty quantification.
- **Disjunctive Geostatistics:** We map an arbitrary non-Gaussian continuous field $Z(\mathbf{s})$ onto a standard Gaussian field $Y(\mathbf{s}) \sim \mathcal{N}(0, 1)$ via Gaussian anamorphosis, and expand the spatial process into an orthogonal series of Hermite polynomials:
  $$
  Z(\mathbf{s}) = \Phi(Y(\mathbf{s})) = \sum_{k=0}^\infty \phi_k H_k(Y(\mathbf{s}))
  $$
  Because Hermite polynomials are orthogonal under Gaussian measures, Mehler's formula completely decouples the bivariate spatial covariance. Disjunctive Kriging computes the optimal non-linear predictor $\widehat{\psi(Z(\mathbf{s}_0))}$ by solving an independent sequence of linear kriging systems for each polynomial degree $k$, generalizing classical orthogonal series regression to continuous spatial domains.

Let $(\Omega, \mathcal{F}, P)$ be a complete probability space, and let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a real-valued second-order spatial random field on $D \subset \mathbb{R}^d$ ($d \ge 1$). Consider the three-layer structural decomposition (Schabenberger & Gotway):
$$
Z(\mathbf{s}) = \mu(\mathbf{s}) + \delta(\mathbf{s}) + \epsilon(\mathbf{s})
$$
where $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ is the deterministic spatial drift, $\delta(\mathbf{s})$ is a correlated zero-mean stationary random effect with covariance kernel $C(\mathbf{h}) = \operatorname{Cov}(\delta(\mathbf{s}), \delta(\mathbf{s}+\mathbf{h}))$, and $\epsilon(\mathbf{s})$ is microscale white noise with variance $\tau^2$.

Let $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ denote the vector of observations at $n$ sample sites. In many engineering and regulatory contexts, our objective is not simply to predict the point value $Z(\mathbf{s}_0)$, but rather a nonlinear transformation $\psi(Z(\mathbf{s}_0))$, such as:
1. Indicator of environmental contamination: $\psi(Z(\mathbf{s}_0)) = \mathbb{I}\{Z(\mathbf{s}_0) \ge z_c\}$.
2. Logarithmic transfer function (e.g., hydraulic conductivity): $\psi(Z(\mathbf{s}_0)) = \ln Z(\mathbf{s}_0)$.
3. Power-law economic recovery: $\psi(Z(\mathbf{s}_0)) = \max(0, Z(\mathbf{s}_0) - z_c)$.

##### Definition 9.1.1 (Minimum Mean Squared Error Nonlinear Predictor)
Let $\psi: \mathbb{R} \to \mathbb{R}$ be a Borel measurable function such that $\operatorname{E}[|\psi(Z(\mathbf{s}_0))|^2] < \infty$. The optimal Minimum Mean Squared Error (MMSE) predictor of $\psi(Z(\mathbf{s}_0))$ given $\mathbf{Z}$ is the conditional expectation:
$$
\widehat{\psi}(\mathbf{s}_0) = \operatorname{E}\left[\psi(Z(\mathbf{s}_0)) \mid \mathbf{Z}\right] = \operatorname{E}\left[\psi(Z(\mathbf{s}_0)) \mid \sigma(Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))\right]
$$
which achieves the global infimum:
$$
\operatorname{E}\left[\left(\psi(Z(\mathbf{s}_0)) - \widehat{\psi}(\mathbf{s}_0)\right)^2\right] = \inf_{g \in \mathcal{M}} \operatorname{E}\left[\left(\psi(Z(\mathbf{s}_0)) - g(\mathbf{Z})\right)^2\right]
$$
where $\mathcal{M}$ is the Hilbert space of all Borel measurable functions $g: \mathbb{R}^n \to \mathbb{R}$ with $\operatorname{E}[|g(\mathbf{Z})|^2] < \infty$.

##### Proposition 9.1.1 (Suboptimality of Plug-in Linear Kriging)
Let $\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu + \boldsymbol{\lambda}'(\mathbf{Z} - \mu \mathbf{1})$ denote the Simple Kriging linear predictor. If $\psi$ is nonlinear, then in general:
$$
\psi\left(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)\right) \ne \operatorname{E}\left[\psi(Z(\mathbf{s}_0)) \mid \mathbf{Z}\right]
$$
Furthermore, if $\psi$ is strictly convex, Jensen's inequality guarantees:
$$
\operatorname{E}\left[\psi(Z(\mathbf{s}_0)) \mid \mathbf{Z}\right] > \psi\left(\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]\right) = \psi\left(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)\right) \quad P\text{-a.s.}
$$
whenever the conditional distribution of $Z(\mathbf{s}_0)$ given $\mathbf{Z}$ is non-degenerate. Plugging linear kriging predictions into nonlinear functions induces systematic structural underestimation.

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **The Jensen's Inequality Trap:**
  Suppose an environmental threshold requires remedial action if the pollutant exceedance probability $P(Z(\mathbf{s}_0) \ge z_c \mid \mathbf{Z}) > 0.10$. If an engineer first calculates the smooth kriging predictor $\widehat{Z}(\mathbf{s}_0)$ and then evaluates the indicator step $\mathbb{I}\{\widehat{Z}(\mathbf{s}_0) \ge z_c\}$, the result will be binary: either 0 or 1. If $\widehat{Z}(\mathbf{s}_0) = 0.95 z_c$, the plug-in method returns 0, falsely concluding that remediation is unnecessary. In reality, the conditional distribution around $0.95 z_c$ may have a 45% tail probability of exceeding $z_c$. Nonlinear estimation requires evaluating the expectation of the function, not the function of the expectation.

---

### 9.2 Lognormal Spatial Fields and Lognormal Kriging

#### Beat 1: The Mathematical Anchor (The "Why")

Lognormal fields are the standard parametric model for strictly positive, positively skewed physical attributes such as hydraulic permeability, trace metal concentrations, and air contaminants.

##### Definition 9.2.1 (Lognormal Random Field)
A spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is called a **lognormal random field** if:
$$
Z(\mathbf{s}) = \exp(Y(\mathbf{s})), \quad \mathbf{s} \in D
$$
where $\{Y(\mathbf{s}) : \mathbf{s} \in D\}$ is a Gaussian random field with mean $\mu_Y(\mathbf{s}) = \operatorname{E}[Y(\mathbf{s})]$ and covariance kernel $C_Y(\mathbf{s}_1, \mathbf{s}_2) = \operatorname{Cov}(Y(\mathbf{s}_1), Y(\mathbf{s}_2))$.

##### Theorem 9.2.1 (Moment Relations of Lognormal Spatial Fields)
Let $\{Y(\mathbf{s})\}$ be a stationary Gaussian field with mean $\mu_Y$, stationary variance $\sigma_Y^2 = C_Y(\mathbf{0})$, and covariance $C_Y(\mathbf{h})$. Then $\{Z(\mathbf{s}) = \exp(Y(\mathbf{s}))\}$ is strictly stationary with:
1. Spatial mean:
   $$
   m_Z = \operatorname{E}[Z(\mathbf{s})] = \exp\left(\mu_Y + \frac{1}{2}\sigma_Y^2\right)
   $$
2. Spatial variance:
   $$
   \sigma_Z^2 = \operatorname{Var}(Z(\mathbf{s})) = \exp\left(2\mu_Y + \sigma_Y^2\right)\left[\exp(\sigma_Y^2) - 1\right] = m_Z^2 \left[\exp(\sigma_Y^2) - 1\right]
   $$
3. Spatial covariance:
   $$
   C_Z(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h})) = m_Z^2 \left[\exp(C_Y(\mathbf{h})) - 1\right]
   $$
4. Spatial semivariogram:
   $$
   \gamma_Z(\mathbf{h}) = m_Z^2 \exp(\sigma_Y^2) \left[1 - \exp(-\gamma_Y(\mathbf{h}))\right]
   $$

*Proof.*
1. Since $Y(\mathbf{s}) \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$, its moment generating function is $M_Y(t) = \operatorname{E}[\exp(t Y(\mathbf{s}))] = \exp(t\mu_Y + \frac{1}{2}t^2\sigma_Y^2)$. Setting $t = 1$ yields $\operatorname{E}[Z(\mathbf{s})] = M_Y(1) = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2) = m_Z$.
2. The second raw moment is $M_Y(2) = \operatorname{E}[\exp(2Y(\mathbf{s}))] = \exp(2\mu_Y + 2\sigma_Y^2)$. The variance is:
   $$
   \sigma_Z^2 = M_Y(2) - [M_Y(1)]^2 = \exp(2\mu_Y + 2\sigma_Y^2) - \exp(2\mu_Y + \sigma_Y^2) = m_Z^2 [\exp(\sigma_Y^2) - 1]
   $$
3. For any lag $\mathbf{h}$, the pair $(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h}))'$ is bivariate normal with covariance $C_Y(\mathbf{h})$. The sum $Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h})$ is univariate normal with mean $2\mu_Y$ and variance $\operatorname{Var}(Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h})) = 2\sigma_Y^2 + 2C_Y(\mathbf{h})$. The joint expectation is:
   $$
   \operatorname{E}[Z(\mathbf{s}) Z(\mathbf{s}+\mathbf{h})] = \operatorname{E}[\exp(Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h}))] = \exp\left(2\mu_Y + \sigma_Y^2 + C_Y(\mathbf{h})\right)
   $$
   Subtracting $(\operatorname{E}[Z(\mathbf{s})])^2 = \exp(2\mu_Y + \sigma_Y^2)$ gives $C_Z(\mathbf{h}) = m_Z^2 [\exp(C_Y(\mathbf{h})) - 1]$.
4. Since $\gamma_Z(\mathbf{h}) = C_Z(\mathbf{0}) - C_Z(\mathbf{h})$:
   $$
   \gamma_Z(\mathbf{h}) = m_Z^2 [\exp(\sigma_Y^2) - \exp(C_Y(\mathbf{h}))] = m_Z^2 \exp(\sigma_Y^2) [1 - \exp(C_Y(\mathbf{h}) - \sigma_Y^2)]
   $$
   Substituting $C_Y(\mathbf{h}) - \sigma_Y^2 = -\gamma_Y(\mathbf{h})$ completes the proof. $\blacksquare$

##### Corollary 9.2.1 (Covariance Inversion)
Given the stationary mean $m_Z$ and covariance $C_Z(\mathbf{h})$ of the original lognormal process, the covariance kernel of the Gaussian log-transformed field $Y(\mathbf{s})$ is uniquely given by:
$$
C_Y(\mathbf{h}) = \ln\left(1 + \frac{C_Z(\mathbf{h})}{m_Z^2}\right)
$$

##### Theorem 9.2.2 (Simple Lognormal Kriging and Exact Back-Transformation)
Let $Z(\mathbf{s}) = \exp(Y(\mathbf{s}))$, where $Y(\mathbf{s})$ is a Gaussian field with known mean $\mu_Y$ and known covariance $C_Y(\mathbf{h})$. Let $\mathbf{Y} = (\ln Z(\mathbf{s}_1), \dots, \ln Z(\mathbf{s}_n))'$ be the Gaussian log-transformed observations.
1. The conditional distribution of $Y(\mathbf{s}_0)$ given $\mathbf{Y}$ is Gaussian:
   $$
   Y(\mathbf{s}_0) \mid \mathbf{Y} \sim \mathcal{N}\left(\widehat{Y}_{\text{SK}}(\mathbf{s}_0), \, \sigma_{\text{SK}}^2(\mathbf{s}_0)\right)
   $$
   where $\widehat{Y}_{\text{SK}}(\mathbf{s}_0) = \mu_Y + \mathbf{c}_Y'\mathbf{V}_Y^{-1}(\mathbf{Y} - \mu_Y \mathbf{1})$ and $\sigma_{\text{SK}}^2(\mathbf{s}_0) = C_Y(\mathbf{0}) - \mathbf{c}_Y'\mathbf{V}_Y^{-1}\mathbf{c}_Y$, with $\mathbf{V}_Y = [C_Y(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$ and $\mathbf{c}_Y = (C_Y(\mathbf{s}_0 - \mathbf{s}_1), \dots, C_Y(\mathbf{s}_0 - \mathbf{s}_n))'$.
2. The optimal MMSE predictor of the original attribute $Z(\mathbf{s}_0)$ is the exact back-transformation:
   $$
   \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \operatorname{E}\left[Z(\mathbf{s}_0) \mid \mathbf{Z}\right] = \exp\left(\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)\right)
   $$
3. The predictor $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ is strictly unbiased: $\operatorname{E}[\widehat{Z}_{\text{SK}}(\mathbf{s}_0)] = m_Z$.
4. The conditional estimation variance is:
   $$
   \operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z}) = \left(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)\right)^2 \left[\exp\left(\sigma_{\text{SK}}^2(\mathbf{s}_0)\right) - 1\right]
   $$

*Proof.*
1. Follows directly from Gaussian conditioning theory.
2. Because $\sigma(\mathbf{Z}) = \sigma(\mathbf{Y})$, the conditional expectation of $Z(\mathbf{s}_0) = \exp(Y(\mathbf{s}_0))$ is the moment generating function of the conditional normal variable $Y(\mathbf{s}_0) \mid \mathbf{Y}$ evaluated at $t = 1$:
   $$
   \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] = \operatorname{E}[\exp(Y(\mathbf{s}_0)) \mid \mathbf{Y}] = \exp\left(\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)\right)
   $$
3. To prove unconditional unbiasedness, note that $\widehat{Y}_{\text{SK}}(\mathbf{s}_0)$ is a linear combination of the Gaussian vector $\mathbf{Y}$. Hence $\widehat{Y}_{\text{SK}}(\mathbf{s}_0) \sim \mathcal{N}\left(\mu_Y, \, \sigma_Y^2 - \sigma_{\text{SK}}^2(\mathbf{s}_0)\right)$.
   Taking the unconditional expectation of the back-transformed predictor:
   $$
   \begin{aligned}
   \operatorname{E}\left[\widehat{Z}_{\text{SK}}(\mathbf{s}_0)\right] &= \operatorname{E}\left[\exp\left(\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)\right)\right] \\
   &= \exp\left(\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)\right) \operatorname{E}\left[\exp\left(\widehat{Y}_{\text{SK}}(\mathbf{s}_0)\right)\right] \\
   &= \exp\left(\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)\right) \exp\left(\mu_Y + \frac{1}{2}\left(\sigma_Y^2 - \sigma_{\text{SK}}^2(\mathbf{s}_0)\right)\right) \\
   &= \exp\left(\mu_Y + \frac{1}{2}\sigma_Y^2\right) = m_Z
   \end{aligned}
   $$
4. The conditional variance is $\operatorname{E}[Z(\mathbf{s}_0)^2 \mid \mathbf{Y}] - (\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}])^2 = \exp(2\widehat{Y}_{\text{SK}} + 2\sigma_{\text{SK}}^2) - \exp(2\widehat{Y}_{\text{SK}} + \sigma_{\text{SK}}^2) = (\widehat{Z}_{\text{SK}})^2 [\exp(\sigma_{\text{SK}}^2) - 1]$. $\blacksquare$

##### Theorem 9.2.3 (Ordinary Lognormal Kriging and Lagrange Multiplier Correction)
When the Gaussian mean $\mu_Y$ is unknown, it is estimated using Ordinary Kriging on $\mathbf{Y}$:
$$
\widehat{Y}_{\text{OK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Y(\mathbf{s}_i), \quad \sum_{i=1}^n \lambda_i = 1
$$
where $\mathbf{V}_Y \boldsymbol{\lambda} + \mu_L \mathbf{1} = \mathbf{c}_Y$ and $\mu_L$ is the Lagrange multiplier.
The exact uniformly minimum variance unbiased estimator (UMVUE) of $Z(\mathbf{s}_0)$ takes the form:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \exp\left(\widehat{Y}_{\text{OK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{OK}}^2(\mathbf{s}_0) - \mu_L\right)
$$
when covariance parameters are known.

##### Term-by-Term Equation Anatomy: The Lognormal Kriging Back-Transformation
$$
\widehat{Z}_{\text{LN}}(\mathbf{s}_0) = \exp\left(\underbrace{\widehat{Y}_{\text{OK}}(\mathbf{s}_0)}_{\text{Conditional Median Predictor}} + \underbrace{\frac{1}{2}\sigma_{\text{OK}}^2(\mathbf{s}_0)}_{\text{Jensen Inequality Variance Correction}} - \underbrace{\mu_L}_{\text{Lagrange Multiplier / Mean Adjustment}}\right)
$$
- **$\exp(\widehat{Y}_{\text{OK}}(\mathbf{s}_0))$:** The conditional median predictor on the original physical scale. Since the logarithmic transformation is strictly monotonic, exponentiating the Gaussian conditional expectation $\widehat{Y}_{\text{OK}}(\mathbf{s}_0) = \operatorname{E}[Y(\mathbf{s}_0) \mid \mathbf{Y}]$ preserves the conditional median (the 50th percentile), but systematically fails to predict the conditional arithmetic mean.
- **$+\frac{1}{2}\sigma_{\text{OK}}^2(\mathbf{s}_0)$:** The Jensen's inequality variance correction. For any normally distributed random variable $W \sim \mathcal{N}(\mu_W, \sigma_W^2)$, the moment generating function gives $\operatorname{E}[\exp(W)] = \exp(\mu_W + \frac{1}{2}\sigma_W^2)$. Because the exponential function is strictly convex, $\operatorname{E}[\exp(W)] > \exp(\operatorname{E}[W])$. Adding half the Ordinary Kriging estimation error variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ compensates for the skewness-induced compression of the logarithm, elevating the conditional median into the strictly unbiased Minimum Mean Squared Error (MMSE) conditional arithmetic mean $\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]$.
- **$-\mu_L$:** The Lagrange multiplier from the OK constraint $\sum \lambda_i = 1$. When the Gaussian mean $\mu_Y$ is unknown and estimated via Ordinary Kriging, the uncertainty in estimating the regional baseline induces a positive bias in the back-transformed expectation. Subtracting the Lagrange multiplier $\mu_L$ cancels the estimation bias of the unknown mean, producing the conditionally unbiased predictor (Journel 1980; Cressie 1993, Section 3.2.2).

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why does naive back-transformation $\exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))$ severely underestimate the arithmetic spatial mean?*  
> By Jensen's inequality, for any strictly convex function $g$, $\operatorname{E}[g(Y)] > g(\operatorname{E}[Y])$ whenever $Y$ is non-degenerate. Because the exponential map is strictly convex, if $Y \sim \mathcal{N}(\mu, \sigma^2)$, then $\operatorname{E}[\exp(Y)] = \exp(\mu + \frac{1}{2}\sigma^2) > \exp(\mu)$. Simple Kriging predicts the conditional expectation on the log-scale: $\widehat{Y}_{\text{SK}}(\mathbf{s}_0) = \operatorname{E}[Y(\mathbf{s}_0) \mid \mathbf{Y}]$. Simply exponentiating this value yields $\exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))$, which represents the conditional *median*, not the conditional mean!  
> Omitting the Jensen correction factor $+\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)$ underestimates the arithmetic mean grade by a multiplicative factor of $\exp(-\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0))$. In unsampled regions where the estimation variance is substantial (e.g., $\sigma_{\text{SK}}^2(\mathbf{s}_0) \approx 1.5$), the naive back-transformation underestimates the true spatial mean by $\exp(-0.75) \approx 0.472$—systematically destroying over 52% of the physical quantity (ore reserve tonnage, soil nutrient content, or pollutant mass)!

- **Why Naive Back-Transformation Fails:**
  The naive back-transformation $\widetilde{Z}(\mathbf{s}_0) = \exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))$ omits the term $\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)$. 
  Its expectation is $\operatorname{E}[\widetilde{Z}(\mathbf{s}_0)] = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2 - \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)) = m_Z \exp(-\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)) < m_Z$.
  In unsampled regions where $\sigma_{\text{SK}}^2(\mathbf{s}_0) \approx \sigma_Y^2$, the naive predictor underestimates the true mean grade by a factor of $\exp(-\frac{1}{2}\sigma_Y^2)$. For a typical log-variance of $\sigma_Y^2 = 1.5$, $\exp(-0.75) \approx 0.472$: **the naive predictor loses over 52% of the true mass!**
- **Sensitivity to Outliers:**
  Notice that $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ exponentiates $\sigma_{\text{SK}}^2(\mathbf{s}_0)$. In sparse data areas with high estimation variance, this exponential back-transformation can explode, producing unrealistically gigantic predicted grades. This extreme sensitivity is the primary vulnerability of lognormal kriging in practice.

---

### 9.3 Indicator Random Fields and Spatial Threshold Geometry

#### Beat 1: The Mathematical Anchor (The "Why")

Indicator geostatistics transforms continuous data into binary variables across cut-off thresholds, bypassing all parametric distributional assumptions.

```
Continuous Field Z(s)
Grade
  ^
  |          * High
  |         / \
u +--------/---\-----------+-------- Cut-off threshold u
  |  *    /     \         /
  |   \  /       \       * Low
  +----\----------*-----+------------> Space s
       |          |
Indicator Field I(s; u) = \mathbb{I}{Z(s) <= u}
  1 +--+----------+-----+------------+
    |  |          |     |            |
  0 +--+----------+-----+------------+--> Space s
```

##### Definition 9.3.1 (Indicator Random Field)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a spatial random field. For any fixed cut-off threshold $u \in \mathbb{R}$, the **indicator random field** $I(\mathbf{s}; u)$ is:
$$
I(\mathbf{s}; u) = \mathbb{I}\{Z(\mathbf{s}) \le u\} = \begin{cases} 1, & \text{if } Z(\mathbf{s}) \le u \\ 0, & \text{if } Z(\mathbf{s}) > u \end{cases}
$$

##### Proposition 9.3.1 (First and Second Order Moments of Indicators)
Let $F(u) = P(Z(\mathbf{s}) \le u)$ and $F(u_1, u_2; \mathbf{h}) = P(Z(\mathbf{s}) \le u_1, Z(\mathbf{s}+\mathbf{h}) \le u_2)$.
1. The indicator expectation is the marginal cumulative probability:
   $$
   \operatorname{E}[I(\mathbf{s}; u)] = 1 \cdot P(Z(\mathbf{s}) \le u) + 0 \cdot P(Z(\mathbf{s}) > u) = F(u)
   $$
2. The indicator stationary variance is:
   $$
   \operatorname{Var}(I(\mathbf{s}; u)) = F(u)[1 - F(u)]
   $$
3. The spatial indicator covariance at threshold $u$ is:
   $$
   C_I(\mathbf{h}; u) = \operatorname{Cov}(I(\mathbf{s}; u), I(\mathbf{s}+\mathbf{h}; u)) = F(u, u; \mathbf{h}) - [F(u)]^2
   $$
4. The spatial indicator semivariogram is:
   $$
   \gamma_I(\mathbf{h}; u) = \frac{1}{2}\operatorname{Var}(I(\mathbf{s}+\mathbf{h}; u) - I(\mathbf{s}; u)) = \frac{1}{2} P\left((Z(\mathbf{s}) \le u < Z(\mathbf{s}+\mathbf{h})) \cup (Z(\mathbf{s}+\mathbf{h}) \le u < Z(\mathbf{s}))\right)
   $$
   which equals half the probability that the boundary isosurface $\{\mathbf{x} : Z(\mathbf{x}) = u\}$ intersects the line segment between $\mathbf{s}$ and $\mathbf{s}+\mathbf{h}$. Under stationarity:
   $$
   \gamma_I(\mathbf{h}; u) = C_I(\mathbf{0}; u) - C_I(\mathbf{h}; u) = F(u)[1 - F(u)] - C_I(\mathbf{h}; u)
   $$

##### Theorem 9.3.1 (Destructuration of Extreme Thresholds)
Let $\rho_I(\mathbf{h}; u) = \frac{C_I(\mathbf{h}; u)}{F(u)[1 - F(u)]}$ denote the indicator correlation function at threshold $u$. The spatial behavior of $\rho_I(\mathbf{h}; u)$ as $u \to \infty$ or $u \to -\infty$ differentiates the fundamental geometry of the spatial random field:
1. **Gaussian Random Fields (Geometric Destructuration):**
   If $\{Z(\mathbf{s})\}$ is a continuous Gaussian random field, then for every fixed non-zero lag $\mathbf{h} \ne \mathbf{0}$:
   $$
   \lim_{u \to -\infty} \rho_I(\mathbf{h}; u) = \lim_{u \to \infty} \rho_I(\mathbf{h}; u) = 0
   $$
   The indicator correlation range contracts toward zero for extreme thresholds: extreme values in Gaussian fields are spatially disconnected and appear as isolated singleton peaks.
2. **Mosaic Random Fields (Structural Permanence):**
   If $\{Z(\mathbf{s})\}$ is a mosaic random field generated by random geometric cell partitions (e.g., Voronoi cells or Boolean models), the bivariate CDF factors as $F(u, u; \mathbf{h}) = \rho_0(\mathbf{h}) F(u) + (1 - \rho_0(\mathbf{h})) [F(u)]^2$, where $\rho_0(\mathbf{h})$ is the probability that $\mathbf{s}$ and $\mathbf{s}+\mathbf{h}$ fall inside the same cell. Then:
   $$
   C_I(\mathbf{h}; u) = F(u)[1 - F(u)] \rho_0(\mathbf{h}) \implies \rho_I(\mathbf{h}; u) = \rho_0(\mathbf{h})
   $$
   The indicator correlation is strictly independent of threshold $u$: extreme high and low values maintain identical spatial connectivity as median values.

*Proof of (1).*
For a standard bivariate normal vector $(Y_1, Y_2)'$ with correlation $\rho \in (-1, 1)$, Mills' ratio asymptotic expansions for bivariate Gaussian lower tails establish that as $y \to -\infty$:
$$
P(Y_1 \le y, Y_2 \le y) \sim \frac{(1+\rho)^2}{2\pi \sqrt{1 - \rho^2} y^2 (1-\rho)} \exp\left(-\frac{y^2}{1+\rho}\right)
$$
whereas $[P(Y_1 \le y)]^2 \sim \frac{1}{2\pi y^2} \exp(-y^2)$.
Since $\rho < 1$, $\frac{1}{1+\rho} > \frac{1}{2}$, so $P(Y_1 \le y, Y_2 \le y)$ decays to zero exponentially faster than $P(Y_1 \le y)$ as $y \to -\infty$. Therefore:
$$
\lim_{y \to -\infty} \frac{P(Y_1 \le y, Y_2 \le y) - [P(Y_1 \le y)]^2}{P(Y_1 \le y)[1 - P(Y_1 \le y)]} = 0
$$
This establishes the destructuration of extremes for Gaussian fields. $\blacksquare$

---

### 9.4 Indicator Kriging (IK)

#### Beat 1: The Mathematical Anchor (The "Why")

Indicator Kriging provides a distribution-free framework to predict the entire conditional cumulative distribution function (cCDF) at an unobserved location $\mathbf{s}_0$:
$$
F(u \mid \text{data}) = P\left(Z(\mathbf{s}_0) \le u \mid \mathbf{Z}\right) = \operatorname{E}\left[I(\mathbf{s}_0; u) \mid \mathbf{Z}\right]
$$

##### Definition 9.4.1 (Simple and Ordinary Indicator Kriging)
Let $\{u_k\}_{k=1}^K$ be an ordered set of $K$ thresholds: $u_1 < u_2 < \dots < u_K$. For each threshold $u_k$, define the indicator data $I(\mathbf{s}_i; u_k) = \mathbb{I}\{Z(\mathbf{s}_i) \le u_k\}$ for $i = 1, \dots, n$.
1. **Simple Indicator Kriging (SIK):**
   When the stationary marginal probability $F(u_k) = P(Z(\mathbf{s}) \le u_k)$ is known:
   $$
   \widehat{F}_{\text{SIK}}(u_k; \mathbf{s}_0) = F(u_k) + \sum_{i=1}^n \lambda_i^{\text{SIK}}(u_k) \left[I(\mathbf{s}_i; u_k) - F(u_k)\right]
   $$
   where $\mathbf{C}_I(u_k) \boldsymbol{\lambda}^{\text{SIK}}(u_k) = \mathbf{c}_{I, 0}(u_k)$.
2. **Ordinary Indicator Kriging (OIK):**
   When local proportion drifts occur:
   $$
   \widehat{F}_{\text{OIK}}(u_k; \mathbf{s}_0) = \sum_{i=1}^n \lambda_i^{\text{OIK}}(u_k) I(\mathbf{s}_i; u_k)
   $$
   where weights satisfy the constrained system:
   $$
   \begin{pmatrix} \mathbf{\Gamma}_I(u_k) & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix}
   \begin{pmatrix} \boldsymbol{\lambda}^{\text{OIK}}(u_k) \\ \mu_I(u_k) \end{pmatrix}
   =
   \begin{pmatrix} \boldsymbol{\gamma}_{I, 0}(u_k) \\ 1 \end{pmatrix}
   $$

##### Term-by-Term Equation Anatomy: Indicator Kriging CCDF Estimator and Linear System
$$
\widehat{F}(u \mid \mathbf{Z}) = \sum_{i=1}^n \underbrace{\lambda_i(u)}_{\text{Threshold-Specific Kriging Weight}} \cdot \underbrace{I(\mathbf{s}_i; u)}_{\text{Binary Spatial Indicator}}
$$
governed by the constrained Ordinary Indicator Kriging system:
$$
\begin{pmatrix} 
\underbrace{\mathbf{\Gamma}_I(u)}_{\text{Indicator Semivariogram Matrix}} & \underbrace{\mathbf{1}}_{\text{Unbiasedness Vector}} \\ 
\mathbf{1}' & 0 
\end{pmatrix}
\begin{pmatrix} 
\boldsymbol{\lambda}^{\text{OIK}}(u) \\ 
\mu_I(u) 
\end{pmatrix}
=
\begin{pmatrix} 
\underbrace{\boldsymbol{\gamma}_{I, 0}(u)}_{\text{Data-to-Target Indicator Continuity}} \\ 
1 
\end{pmatrix}
$$
- **$\widehat{F}(u \mid \mathbf{Z})$:** The estimated conditional cumulative distribution function (CCDF) at target location $\mathbf{s}_0$, evaluating the posterior exceedance probability $P(Z(\mathbf{s}_0) \le u \mid \mathbf{Z}) = \operatorname{E}[I(\mathbf{s}_0; u) \mid \mathbf{Z}]$.
- **$\lambda_i(u)$:** The threshold-specific spatial kriging weight assigned to sample $i$. Because spatial continuity varies with cut-off level (Theorem 9.3.1), kriging weights adapt across thresholds, reflecting changing spatial ranges and screening geometries.
- **$I(\mathbf{s}_i; u) = \mathbb{I}(Z(\mathbf{s}_i) \le u)$:** The binary spatial indicator taking value $1$ if the observed attribute at sample site $\mathbf{s}_i$ is at or below threshold $u$, and $0$ otherwise.
- **$\mathbf{\Gamma}_I(u) = [\gamma_I(\mathbf{s}_i - \mathbf{s}_j; u)]_{i,j=1}^n$:** The $n \times n$ indicator semivariogram matrix measuring spatial transition probabilities between sample pairs at threshold $u$.
- **$\boldsymbol{\gamma}_{I, 0}(u) = (\gamma_I(\mathbf{s}_0 - \mathbf{s}_1; u), \dots, \gamma_I(\mathbf{s}_0 - \mathbf{s}_n; u))'$:** The spatial continuity vector between data locations and the target prediction node $\mathbf{s}_0$.
- **$\mu_I(u)$:** The Lagrange multiplier enforcing the universal unbiasedness condition $\sum_{i=1}^n \lambda_i^{\text{OIK}}(u) = 1$.

##### Definition 9.4.2 (Median Indicator Kriging)
If the spatial continuity across all thresholds satisfies the intrinsic correlation hypothesis:
$$
C_I(\mathbf{h}; u_k) = \kappa(u_k) \rho_M(\mathbf{h}), \quad \forall k \in \{1, \dots, K\}
$$
where $\rho_M(\mathbf{h})$ is the spatial correlogram at the median threshold $u_{\text{med}}$ (where $F(u_{\text{med}}) = 0.5$) and $\kappa(u_k) > 0$ is a scalar factor, then the Ordinary Kriging weights are identical across all thresholds:
$$
\boldsymbol{\lambda}^{\text{OIK}}(u_k) \equiv \boldsymbol{\lambda}_M, \quad \forall k = 1, \dots, K
$$
This method, termed **Median Indicator Kriging (MIK)**, requires inverting only a single kriging matrix for all $K$ thresholds.

---

### 9.5 Order Relation Violations and Monotonicity Corrections

#### Beat 1: The Mathematical Anchor (The "Why")

Because linear kriging weights $\lambda_i$ can be negative due to the screening effect, and because the kriging systems at different cut-offs $u_k$ and $u_{k+1}$ are solved independently, raw indicator kriging estimates frequently violate the axiomatic properties of probability measures.

##### Definition 9.5.1 (Order Relation Violations)
A vector of estimated conditional probabilities $\mathbf{y} = (\widehat{F}(u_1), \dots, \widehat{F}(u_K))'$ at thresholds $u_1 < \dots < u_K$ exhibits an **order relation violation** if:
1. **Out-of-Bounds Probability:**
   $$
   \widehat{F}(u_k) < 0 \quad \text{or} \quad \widehat{F}(u_k) > 1 \quad \text{for some } k \in \{1, \dots, K\}
   $$
2. **Monotonicity Reversal:**
   $$
   \widehat{F}(u_{k+1}) < \widehat{F}(u_k) \quad \text{for some } k \in \{1, \dots, K-1\}
   $$
   which implies a physically impossible negative probability:
   $$
   P(u_k < Z(\mathbf{s}_0) \le u_{k+1} \mid \text{data}) = \widehat{F}(u_{k+1}) - \widehat{F}(u_k) < 0
   $$

##### Theorem 9.5.1 (Isotonic Projection via the Pool-Adjacent-Violators Algorithm)
Let $\mathbf{y} = (\widehat{F}(u_1), \dots, \widehat{F}(u_K))' \in \mathbb{R}^K$ be the raw indicator kriging vector, and let $\mathbf{w} = (w_1, \dots, w_K)'$ be strictly positive weights ($w_k > 0$).
Let $\mathcal{C} \subset \mathbb{R}^K$ be the closed convex cone of monotonically non-decreasing vectors bounded in $[0, 1]$:
$$
\mathcal{C} = \left\{\mathbf{x} \in \mathbb{R}^K : 0 \le x_1 \le x_2 \le \dots \le x_K \le 1\right\}
$$
1. There exists a unique projection $\mathbf{x}^* = \Pi_{\mathcal{C}}^{\mathbf{w}}(\mathbf{y}) \in \mathcal{C}$ that minimizes the weighted squared Euclidean distance:
   $$
   \min_{\mathbf{x} \in \mathcal{C}} \sum_{k=1}^K w_k (x_k - y_k)^2
   $$
2. The optimal vector $\mathbf{x}^*$ is computed exactly by the **Pool-Adjacent-Violators Algorithm (PAVA)**:
   - Initial pass: Clamp each entry to $[0, 1]$: $y_k \leftarrow \max(0, \min(1, y_k))$.
   - Partition into $K$ blocks: $B_k = \{k\}$ with values $\bar{y}_k = y_k$ and weights $W_k = w_k$.
   - Whenever an adjacent inversion occurs ($\bar{y}_j > \bar{y}_{j+1}$), pool the adjacent blocks $B_j$ and $B_{j+1}$ into a combined block $B = B_j \cup B_{j+1}$, assign the pooled weighted average:
     $$
     \bar{y}_B = \frac{W_j \bar{y}_j + W_{j+1} \bar{y}_{j+1}}{W_j + W_{j+1}}
     $$
     and update the combined weight $W_B = W_j + W_{j+1}$.
   - Backtrack and pool to the left if the new value violates monotonicity with preceding blocks.
   - Continue until all blocks satisfy $\bar{y}_1 \le \bar{y}_2 \le \dots \le \bar{y}_m$.

*Proof.*
1. The objective $J(\mathbf{x}) = \sum w_k (x_k - y_k)^2$ is strictly convex, continuous, and coercive. The constraint set $\mathcal{C}$ is closed and convex. By the Hilbert Projection Theorem, there exists a unique global minimizer $\mathbf{x}^* \in \mathcal{C}$.
2. The Karush-Kuhn-Tucker (KKT) conditions require that at an active monotonicity constraint $x_k^* = x_{k+1}^*$, the derivative of the Lagrangian with respect to the common value must vanish, forcing the solution to be the weighted average $\frac{w_k y_k + w_{k+1} y_{k+1}}{w_k + w_{k+1}}$. PAVA maintains these KKT stationarity and dual feasibility conditions at every pooling step, terminating in at most $K-1$ merges with the exact global minimizer. $\blacksquare$

---

#### Beat 2: Algorithmic Translation (The "How")

```python
def pava_isotonic_regression(
    raw_probs: list[float],
    weights: list[float] | None = None
) -> list[float]:
    """
    Applies the Pool-Adjacent-Violators Algorithm (PAVA) to correct
    order relation violations in an estimated cumulative distribution.
    
    Parameters:
        raw_probs: list of raw estimated probabilities [F_hat(u_1), ..., F_hat(u_K)]
        weights: list of positive weights w_k (defaults to equal weights 1.0)
        
    Returns:
        corrected_probs: monotonically non-decreasing probabilities in [0, 1]
    """
    K = len(raw_probs)
    if weights is None:
        weights = [1.0] * K
        
    # Step 1: Enforce box constraints [0, 1]
    clamped = [max(0.0, min(1.0, float(p))) for p in raw_probs]
    
    # Step 2: Initialize blocks for PAVA
    # Each block is stored as: [weighted_val, total_weight, [indices]]
    blocks = [[clamped[k], float(weights[k]), [k]] for k in range(K)]
    
    # Step 3: Iterative adjacent pooling
    i = 0
    while i < len(blocks) - 1:
        # Check adjacent violation
        if blocks[i][0] > blocks[i + 1][0]:
            # Merge block i and block i+1
            w1 = blocks[i][1]
            w2 = blocks[i + 1][1]
            v1 = blocks[i][0]
            v2 = blocks[i + 1][0]
            
            merged_val = (w1 * v1 + w2 * v2) / (w1 + w2)
            merged_weight = w1 + w2
            merged_indices = blocks[i][2] + blocks[i + 1][2]
            
            # Replace blocks i and i+1 with the merged block
            blocks[i] = [merged_val, merged_weight, merged_indices]
            del blocks[i + 1]
            
            # Backtrack to verify monotonicity with preceding block
            if i > 0:
                i -= 1
        else:
            i += 1
            
    # Step 4: Unpack block values into the final corrected vector
    corrected_probs = [0.0] * K
    for blk in blocks:
        val = blk[0]
        for idx in blk[2]:
            corrected_probs[idx] = val
            
    return corrected_probs
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why does Indicator Kriging produce order-relation violations ($\widehat{F}(u_1) > \widehat{F}(u_2)$ for $u_1 < u_2$, or values outside $[0, 1]$), and how does PAVA fix it?*  
> Indicator Kriging estimates the spatial conditional CDF by solving independent linear kriging systems at each cut-off threshold $u_k$ ($k=1, \dots, K$). Because each threshold has its own empirical semivariogram $\gamma_I(\mathbf{h}; u_k)$, the kriging weights $\lambda_i(u_k)$ change from threshold to threshold. Furthermore, due to the spatial screening effect, kriging weights can be negative for samples shielded by closer points. Without explicit cross-threshold constraints, an estimated probability $\widehat{F}(u_k)$ can drop below zero, exceed one, or decrease at a higher threshold ($\widehat{F}(u_{k+1}) < \widehat{F}(u_k)$), implying an axiomatic impossibility: negative probability mass $P(u_k < Z(\mathbf{s}_0) \le u_{k+1} \mid \mathbf{Z}) < 0$.  
> The Pool-Adjacent-Violators Algorithm (PAVA) fixes this by computing the unique orthogonal projection of the violating vector $\mathbf{y}$ onto the closed convex cone $\mathcal{C} = \{\mathbf{x} \in \mathbb{R}^K : 0 \le x_1 \le \dots \le x_K \le 1\}$ in $L^2$ space. Whenever an adjacent inversion occurs ($\bar{y}_j > \bar{y}_{j+1}$), PAVA pools the blocks into a combined weighted average, satisfying all Karush-Kuhn-Tucker (KKT) stationary conditions in at most $K-1$ steps without introducing the upward or downward heuristic biases of ad-hoc clamping.

- **Why Monotonicity Reversals Happen:**
  Suppose two close drill cores have values $Z(\mathbf{s}_1) = 2.5$ and $Z(\mathbf{s}_2) = 5.0$. At cut-off $u_1 = 3.0$, the indicators are $I_1 = 1$ and $I_2 = 0$. At cut-off $u_2 = 4.0$, the indicators are still $I_1 = 1$ and $I_2 = 0$. However, if we fit separate semivariogram models $\gamma_I(\mathbf{h}; u_1)$ and $\gamma_I(\mathbf{h}; u_2)$, the kriging weights change. A negative screening weight $\lambda_2 < 0$ at threshold $u_2$ can cause the estimated probability $\widehat{F}(u_2)$ to drop *below* $\widehat{F}(u_1)$, implying negative probability mass in $(3.0, 4.0]$.
- **PAVA vs. Ad-Hoc Clamping:**
  Simple upward-sweeping heuristics ($F(u_k) \leftarrow \max(F(u_{k-1}), F(u_k))$) bias the entire distribution upward. PAVA is the *unique* orthogonal projection in $L^2$ space: it preserves the mean probability across pooled thresholds and minimizes the total distortion $\|\mathbf{x}^* - \mathbf{y}\|_2^2$.

---

### 9.6 Gaussian Anamorphosis via Hermite Polynomial Expansions

#### Beat 1: The Mathematical Anchor (The "Why")

Gaussian anamorphosis establishes an analytical bridge between arbitrary non-Gaussian physical fields $Z(\mathbf{s})$ and standard Gaussian fields $Y(\mathbf{s}) \sim \mathcal{N}(0, 1)$ through a monotonic transfer function $\Phi$:
$$
Z(\mathbf{s}) = \Phi(Y(\mathbf{s})) = F_Z^{-1}\left(G(Y(\mathbf{s}))\right)
$$
where $G(y) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^y e^{-t^2/2} dt$ is the standard normal CDF and $F_Z$ is the marginal CDF of $Z$.

##### Definition 9.6.1 (The Gaussian Hilbert Space and Hermite Polynomials)
Let $\phi(y) = \frac{1}{\sqrt{2\pi}} e^{-y^2/2}$ be the standard Gaussian density on $\mathbb{R}$.
1. The Hilbert space $L^2(\mathbb{R}, \phi(y)dy)$ is equipped with the inner product:
   $$
   \langle f, g \rangle_\phi = \int_{-\infty}^\infty f(y) g(y) \phi(y) dy
   $$
2. The **Hermite polynomials** $\{H_n(y)\}_{n=0}^\infty$ are defined by Rodrigues' formula:
   $$
   H_n(y) = \frac{(-1)^n}{\phi(y)} \frac{d^n}{dy^n}\phi(y) = (-1)^n e^{y^2/2} \frac{d^n}{dy^n}\left(e^{-y^2/2}\right)
   $$
   The first five Hermite polynomials are:
   $$
   H_0(y) = 1, \quad H_1(y) = y, \quad H_2(y) = y^2 - 1, \quad H_3(y) = y^3 - 3y, \quad H_4(y) = y^4 - 6y^2 + 3
   $$

##### Theorem 9.6.1 (Orthogonality and Recurrence)
The family $\{H_n(y)\}_{n=0}^\infty$ forms a complete orthogonal basis of $L^2(\mathbb{R}, \phi(y)dy)$ satisfying:
1. **Orthogonality:**
   $$
   \langle H_n, H_m \rangle_\phi = \int_{-\infty}^\infty H_n(y) H_m(y) \phi(y) dy = n! \, \delta_{nm}
   $$
2. **Three-Term Recurrence:**
   $$
   H_{n+1}(y) = y H_n(y) - n H_{n-1}(y), \quad \forall n \ge 1
   $$
3. **Derivative Identity:**
   $$
   \frac{d}{dy} H_n(y) = n H_{n-1}(y), \quad \forall n \ge 1
   $$
4. **Generating Function:**
   $$
   \exp\left(t y - \frac{1}{2}t^2\right) = \sum_{n=0}^\infty \frac{t^n}{n!} H_n(y)
   $$

*Proof.*
Using Rodrigues' formula and integration by parts $n$ times:
$$
\int_{-\infty}^\infty H_n(y) H_m(y) \phi(y) dy = (-1)^n \int_{-\infty}^\infty H_m(y) \frac{d^n\phi}{dy^n} dy = \int_{-\infty}^\infty \left(\frac{d^n H_m}{dy^n}\right) \phi(y) dy
$$
If $m < n$, $\frac{d^n H_m}{dy^n} = 0$. If $m = n$, since the leading term of $H_n(y)$ is $y^n$, $\frac{d^n H_n}{dy^n} = n!$. Hence $\langle H_n, H_m \rangle = n! \delta_{nm}$. The recurrence and derivative relations follow directly from differentiating the generating function $\frac{\phi(y-t)}{\phi(y)} = \exp(ty - \frac{1}{2}t^2)$. $\blacksquare$

##### Theorem 9.6.2 (Mehler's Formula for Bivariate Gaussian Densities)
Let $(Y_1, Y_2)'$ be a standard bivariate normal vector with correlation $\rho \in (-1, 1)$. The joint probability density function $g_\rho(y_1, y_2)$ admits the bilinear expansion:
$$
g_\rho(y_1, y_2) = \phi(y_1) \phi(y_2) \sum_{n=0}^\infty \frac{\rho^n}{n!} H_n(y_1) H_n(y_2)
$$

*Proof.*
Expand the density ratio $K_\rho(y_1, y_2) = \frac{g_\rho(y_1, y_2)}{\phi(y_1)\phi(y_2)}$ in the product basis:
$$
K_\rho(y_1, y_2) = \sum_{n=0}^\infty \sum_{m=0}^\infty a_{nm} H_n(y_1) H_m(y_2)
$$
The expansion coefficients are $a_{nm} = \frac{1}{n! m!} \operatorname{E}[H_n(Y_1) H_m(Y_2)]$.
Using the bivariate moment generating function of the generating functions:
$$
\operatorname{E}\left[\exp\left(s Y_1 - \frac{1}{2}s^2\right) \exp\left(t Y_2 - \frac{1}{2}t^2\right)\right] = \exp(\rho s t) = \sum_{n=0}^\infty \frac{\rho^n}{n!} s^n t^n
$$
Matching powers of $s^n t^m$ gives $\operatorname{E}[H_n(Y_1) H_m(Y_2)] = n! \rho^n \delta_{nm}$. Hence $a_{nm} = \frac{\rho^n}{n!} \delta_{nm}$, establishing Mehler's formula. $\blacksquare$

---

#### Beat 2: Algorithmic Translation (The "How")

```python
def fit_hermite_gaussian_anamorphosis(
    sample_values: list[float],
    n_poly: int = 15,
    n_gauss_nodes: int = 128
) -> tuple[list[float], callable, callable]:
    """
    Fits Gaussian anamorphosis expansion Phi(y) = sum_{n=0}^{n_poly} psi_n * H_n(y)
    via Gauss-Hermite numerical quadrature.
    
    Parameters:
        sample_values: empirical sample data [z_1, ..., z_n]
        n_poly: maximum Hermite polynomial order
        n_gauss_nodes: number of quadrature evaluation nodes
        
    Returns:
        (coeffs, forward_transform, inverse_transform):
            coeffs: Hermite coefficients [psi_0, ..., psi_{n_poly}]
            forward_transform: callable z -> y
            inverse_transform: callable y -> z
    """
    import math
    n = len(sample_values)
    sorted_z = sorted(sample_values)
    
    # Step 1: Define empirical quantile function F_Z^{-1}(p)
    def quantile_z(p: float) -> float:
        p = max(0.5 / n, min(1.0 - 0.5 / n, p))
        idx = p * (n - 1)
        i = int(idx)
        frac = idx - i
        if i >= n - 1:
            return sorted_z[-1]
        return (1.0 - frac) * sorted_z[i] + frac * sorted_z[i + 1]
        
    # Standard normal CDF approximation (Hart's algorithm)
    def standard_normal_cdf(y: float) -> float:
        return 0.5 * (1.0 + math.erf(y / math.sqrt(2.0)))
        
    # Step 2: Hermite polynomial evaluation via 3-term recurrence
    def evaluate_hermite(order: int, y: float) -> list[float]:
        H = [0.0] * (order + 1)
        H[0] = 1.0
        if order >= 1:
            H[1] = y
        for k in range(1, order):
            H[k + 1] = y * H[k] - k * H[k - 1]
        return H
        
    # Step 3: Compute expansion coefficients psi_n = (1/n!) * E[Phi(Y) * H_n(Y)]
    # Use midpoint Riemann quadrature over standard normal density [-5, 5]
    y_min, y_max = -5.0, 5.0
    dy = (y_max - y_min) / n_gauss_nodes
    quad_y = [y_min + (k + 0.5) * dy for k in range(n_gauss_nodes)]
    
    coeffs = [0.0] * (n_poly + 1)
    for y_val in quad_y:
        phi_wt = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * y_val * y_val) * dy
        p_val = standard_normal_cdf(y_val)
        z_val = quantile_z(p_val)
        H_vals = evaluate_hermite(n_poly, y_val)
        for k in range(n_poly + 1):
            coeffs[k] += z_val * H_vals[k] * phi_wt
            
    # Divide by factorial n!
    fact = 1.0
    for k in range(1, n_poly + 1):
        fact *= k
        coeffs[k] /= fact
        
    # Step 4: Inverse anamorphosis function z = Phi(y)
    def inverse_transform(y: float) -> float:
        H_vals = evaluate_hermite(n_poly, y)
        return sum(coeffs[k] * H_vals[k] for k in range(n_poly + 1))
        
    # Forward transform y = Phi^{-1}(z) via bisection inversion
    def forward_transform(z: float) -> float:
        low, high = -6.0, 6.0
        for _ in range(40):
            mid = 0.5 * (low + high)
            if inverse_transform(mid) < z:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)
        
    return coeffs, forward_transform, inverse_transform
```

---

### 9.7 Disjunctive Kriging (DK) and Isofactorial Models

#### Beat 1: The Mathematical Anchor (The "Why")

Disjunctive Kriging (Matheron, 1976) seeks the optimal nonlinear spatial predictor within the class of additive nonlinear transformations of each observation:
$$
Z^*(\mathbf{s}_0) = \sum_{i=1}^n f_i(Z(\mathbf{s}_i))
$$
minimizing $\operatorname{E}[(Z(\mathbf{s}_0) - Z^*(\mathbf{s}_0))^2]$.

##### Definition 9.7.1 (Bivariate Isofactorial Models)
A bivariate distribution $F(z_1, z_2)$ is **isofactorial** if there exists an orthonormal basis $\{\chi_n(z)\}_{n=0}^\infty$ in $L^2(\mathbb{R}, dF_0)$ such that:
$$
f(z_1, z_2) = f_0(z_1) f_0(z_2) \sum_{n=0}^\infty T_n(\mathbf{h}) \chi_n(z_1) \chi_n(z_2)
$$
where $T_n(\mathbf{h})$ are spatial transition factors with $|T_n(\mathbf{h})| \le 1$. Under Gaussian anamorphosis, $\chi_n(z) = \frac{1}{\sqrt{n!}} H_n(y)$ and $T_n(\mathbf{h}) = [\rho(\mathbf{h})]^n$ by Mehler's formula.

##### Theorem 9.7.1 (Decoupling of Disjunctive Kriging under Gaussian Anamorphosis)
Let $Z(\mathbf{s}) = \Phi(Y(\mathbf{s})) = \sum_{n=0}^\infty \psi_n H_n(Y(\mathbf{s}))$ be the Hermite anamorphosis expansion. Let $\rho(\mathbf{h})$ denote the spatial correlogram of Gaussian field $Y(\mathbf{s})$.
1. For any measurable transformation $\psi(Z(\mathbf{s}_0)) = \sum_{n=0}^\infty \psi_n H_n(Y(\mathbf{s}_0))$, the Disjunctive Kriging predictor decouples into an infinite sequence of independent linear kriging systems for each Hermite order $n$:
   $$
   [\psi(Z(\mathbf{s}_0))]_{\text{DK}}^* = \psi_0 + \sum_{n=1}^\infty \psi_n \left(\sum_{i=1}^n \lambda_i^{(n)} H_n(Y(\mathbf{s}_i))\right)
   $$
2. For each polynomial degree $n \ge 1$, the weight vector $\boldsymbol{\lambda}^{(n)} = (\lambda_1^{(n)}, \dots, \lambda_n^{(n)})'$ satisfies:
   $$
   \sum_{j=1}^n \lambda_j^{(n)} [\rho(\mathbf{s}_i - \mathbf{s}_j)]^n = [\rho(\mathbf{s}_i - \mathbf{s}_0)]^n, \quad i = 1, \dots, n
   $$
   or in matrix form: $\mathbf{R}^{(n)} \boldsymbol{\lambda}^{(n)} = \mathbf{r}_0^{(n)}$, where $\mathbf{R}_{ij}^{(n)} = [\rho(\mathbf{s}_i - \mathbf{s}_j)]^n$ and $[\mathbf{r}_0^{(n)}]_i = [\rho(\mathbf{s}_i - \mathbf{s}_0)]^n$.
3. The Disjunctive Kriging estimation variance is:
   $$
   \sigma_{\text{DK}}^2(\mathbf{s}_0) = \sum_{n=1}^\infty n! \, \psi_n^2 \left(1 - \sum_{i=1}^n \lambda_i^{(n)} [\rho(\mathbf{s}_i - \mathbf{s}_0)]^n\right)
   $$

*Proof.*
By definition, $[\psi(Z(\mathbf{s}_0))]^* = \sum_{i=1}^n f_i(Y(\mathbf{s}_i))$. Expanding each function in Hermite polynomials $f_i(y) = \sum_{n=0}^\infty f_{in} H_n(y)$:
$$
[\psi(Z(\mathbf{s}_0))]^* = \sum_{n=0}^\infty \left(\sum_{i=1}^n f_{in} H_n(Y(\mathbf{s}_i))\right)
$$
By Hilbert projection, the error must be orthogonal to $H_m(Y(\mathbf{s}_j))$ for all $j=1,\dots,n$ and $m \ge 0$. Using Mehler's formula, $\operatorname{E}[H_k(Y(\mathbf{s}_a)) H_m(Y(\mathbf{s}_b))] = k! [\rho(\mathbf{s}_a - \mathbf{s}_b)]^k \delta_{km}$. All terms with $k \ne m$ vanish identically. The orthogonality equation collapses to:
$$
m! \, \psi_m [\rho(\mathbf{s}_j - \mathbf{s}_0)]^m - \sum_{i=1}^n f_{im} m! \, [\rho(\mathbf{s}_j - \mathbf{s}_i)]^m = 0
$$
Dividing by $m!$ and setting $f_{im} = \psi_m \lambda_i^{(m)}$ yields the decoupled system. Because polynomials of different degrees are mutually orthogonal, the error variance is the sum of the variances of each degree, establishing the variance formula. $\blacksquare$

##### Term-by-Term Equation Anatomy: Bivariate Hermite Expansion and Covariance Decoupling
$$
\operatorname{Cov}\left(H_j(Y(\mathbf{s})), \, H_k(Y(\mathbf{s}+\mathbf{h}))\right) = \underbrace{k!}_{\text{Hermite Factorial Norm}} \cdot \underbrace{\delta_{jk}}_{\text{Cross-Degree Orthogonality}} \cdot \underbrace{[\rho(\mathbf{h})]^k}_{\text{Damped Spatial Correlogram}}
$$
(or in normalized form $\operatorname{Cov}(\chi_j(Y(\mathbf{s})), \chi_k(Y(\mathbf{s}+\mathbf{h}))) = \delta_{jk} [\rho(\mathbf{h})]^k$ where $\chi_k(y) = \frac{1}{\sqrt{k!}} H_k(y)$):
- **$H_j(Y(\mathbf{s}))$ and $H_k(Y(\mathbf{s}+\mathbf{h}))$:** Orthogonal Hermite polynomials of degree $j$ and $k$, evaluated on the standard Gaussian transformed field $Y(\mathbf{s}) \sim \mathcal{N}(0, 1)$ at spatial separation lag $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$.
- **$k!$:** Factorial scaling factor corresponding to the $L^2(\mathbb{R}, \phi(y)dy)$ orthogonal basis norm $\langle H_k, H_k \rangle_\phi = k!$.
- **$\delta_{jk}$:** The Kronecker delta symbol ($\delta_{jk} = 1$ if $j = k$, and $0$ if $j \ne k$). This orthogonality property is the mathematical engine of Disjunctive Kriging: polynomials of distinct degrees $j \ne k$ possess zero covariance at *all* spatial separation distances $\mathbf{h} \in \mathbb{R}^d$. This completely eliminates cross-polynomial terms, decoupling an intractable infinite-dimensional joint optimization problem into an infinite sequence of independent, standard single-degree linear kriging systems.
- **$[\rho(\mathbf{h})]^k$:** The $k$-th power of the stationary Gaussian spatial correlogram $\rho(\mathbf{h}) = \operatorname{Corr}(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h}))$. Because $|\rho(\mathbf{h})| < 1$ for all non-zero separation lags $\mathbf{h} \ne \mathbf{0}$, raising $\rho(\mathbf{h})$ to the $k$-th power exponentially dampens spatial correlation as polynomial degree $k$ increases ($[\rho(\mathbf{h})]^k \to 0$ rapidly as $k \to \infty$). Consequently, higher-order non-linear terms possess rapidly contracting spatial correlation ranges, confining non-linear interactions to immediate local neighborhoods.

---

### 9.8 Analytical Exercises and Complete Step-by-Step Solutions

#### Exercise 9.1: Exact Derivation of Lognormal Back-Transformation
Let $Z(\mathbf{s}) = \exp(Y(\mathbf{s}))$ be a lognormal random field, where $Y(\mathbf{s})$ is a Gaussian field with known mean $\mu_Y = 1.0$ and exponential covariance $C_Y(h) = 0.5 \exp(-|h|/2.0)$.
Two observations are available: $Z(\mathbf{s}_1) = 4.0$ at $\mathbf{s}_1 = 0$ and $Z(\mathbf{s}_2) = 6.0$ at $\mathbf{s}_2 = 2.0$.
**(a)** Compute the Gaussian transformed data $Y(\mathbf{s}_1)$ and $Y(\mathbf{s}_2)$.
**(b)** Formulate and solve the Simple Kriging system for $Y(\mathbf{s}_0)$ at $\mathbf{s}_0 = 1.0$, computing $\widehat{Y}_{\text{SK}}(\mathbf{s}_0)$ and $\sigma_{\text{SK}}^2(\mathbf{s}_0)$.
**(c)** Calculate the exact back-transformed MMSE predictor $\widehat{Z}(\mathbf{s}_0) = \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]$ and compare it with the naive predictor $\exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))$.
**(d)** Derive the exact conditional 95% prediction interval for $Z(\mathbf{s}_0)$.

##### Solution:
**(a)** Logarithmic transformation:
$$
Y(\mathbf{s}_1) = \ln(4.0) \approx 1.3863, \quad Y(\mathbf{s}_2) = \ln(6.0) \approx 1.7918
$$

**(b)** Covariance values:
$$
C_Y(0) = 0.5, \quad C_Y(\mathbf{s}_1 - \mathbf{s}_2) = C_Y(2.0) = 0.5 \exp(-2.0/2.0) = 0.5 e^{-1} \approx 0.18394
$$
$$
C_Y(\mathbf{s}_0 - \mathbf{s}_1) = C_Y(1.0) = 0.5 \exp(-1.0/2.0) = 0.5 e^{-0.5} \approx 0.30327
$$
$$
C_Y(\mathbf{s}_0 - \mathbf{s}_2) = C_Y(1.0) = 0.5 e^{-0.5} \approx 0.30327
$$
The covariance matrix $\mathbf{V}_Y$ and vector $\mathbf{c}_Y$:
$$
\mathbf{V}_Y = \begin{pmatrix} 0.50000 & 0.18394 \\ 0.18394 & 0.50000 \end{pmatrix}, \quad \mathbf{c}_Y = \begin{pmatrix} 0.30327 \\ 0.30327 \end{pmatrix}
$$
By symmetry, Simple Kriging weights are equal: $\lambda_1 = \lambda_2 = \lambda$:
$$
(0.50000 + 0.18394) \lambda = 0.30327 \implies 0.68394 \lambda = 0.30327 \implies \lambda = \frac{0.30327}{0.68394} \approx 0.44342
$$
The Simple Kriging estimate on $Y$:
$$
\begin{aligned}
\widehat{Y}_{\text{SK}}(\mathbf{s}_0) &= \mu_Y + \lambda (Y(\mathbf{s}_1) - \mu_Y) + \lambda (Y(\mathbf{s}_2) - \mu_Y) \\
&= 1.0 + 0.44342 (1.3863 - 1.0) + 0.44342 (1.7918 - 1.0) \\
&= 1.0 + 0.44342 (0.3863 + 0.7918) = 1.0 + 0.44342 (1.1781) = 1.0 + 0.52240 \approx 1.5224
\end{aligned}
$$
The Simple Kriging variance on $Y$:
$$
\begin{aligned}
\sigma_{\text{SK}}^2(\mathbf{s}_0) &= C_Y(0) - [\lambda C_Y(1.0) + \lambda C_Y(1.0)] = 0.5 - 2 (0.44342)(0.30327) \\
&= 0.5 - 2 (0.13448) = 0.5 - 0.26896 = 0.23104
\end{aligned}
$$

**(c)** Back-transformation:
Naive predictor:
$$
\widetilde{Z}(\mathbf{s}_0) = \exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0)) = \exp(1.5224) \approx 4.5832
$$
Exact MMSE back-transformed predictor:
$$
\widehat{Z}(\mathbf{s}_0) = \exp\left(\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0)\right) = \exp\left(1.5224 + \frac{1}{2}(0.23104)\right) = \exp(1.5224 + 0.11552) = \exp(1.6379) \approx 5.1444
$$
Ratio of naive to exact predictor:
$$
\frac{\widetilde{Z}(\mathbf{s}_0)}{\widehat{Z}(\mathbf{s}_0)} = \exp(-0.11552) \approx 0.8909
$$
The naive predictor underestimates the conditional mean grade by **10.91%**.

**(d)** 95% conditional interval:
The standard error on $Y$ is $\sigma_{\text{SK}} = \sqrt{0.23104} \approx 0.48067$.
The 95% interval on $Y(\mathbf{s}_0)$ is:
$$
Y_{0.025} = 1.5224 - 1.96 (0.48067) = 1.5224 - 0.9421 = 0.5803
$$
$$
Y_{0.975} = 1.5224 + 1.96 (0.48067) = 1.5224 + 0.9421 = 2.4645
$$
Exponentiating these bounds yields the exact 95% conditional interval for $Z(\mathbf{s}_0)$:
$$
Z_{0.025} = \exp(0.5803) \approx 1.7866, \quad Z_{0.975} = \exp(2.4645) \approx 11.7576
$$
The 95% probability interval is $[1.79, 11.76]$.

---

#### Exercise 9.2: Isotonic Projection of Violating Indicator Kriging Estimates via PAVA
At a target location $\mathbf{s}_0$, Ordinary Indicator Kriging at $K=5$ thresholds produces the raw estimated vector:
$$
\mathbf{y} = \begin{pmatrix} \widehat{F}(u_1) \\ \widehat{F}(u_2) \\ \widehat{F}(u_3) \\ \widehat{F}(u_4) \\ \widehat{F}(u_5) \end{pmatrix} = \begin{pmatrix} -0.05 \\ 0.35 \\ 0.25 \\ 0.80 \\ 1.10 \end{pmatrix}
$$
with equal weights $\mathbf{w} = (1, 1, 1, 1, 1)'$.
**(a)** Identify all order-relation violations in $\mathbf{y}$.
**(b)** Carry out the step-by-step PAVA algorithm to compute the exact isotonic projection $\mathbf{x}^* \in \mathcal{C}$.
**(c)** Verify that the resulting vector satisfies all KKT conditions.

##### Solution:
**(a)** Identification of violations:
1. $y_1 = -0.05 < 0$ (out-of-bounds probability).
2. $y_3 = 0.25 < 0.35 = y_2$ (monotonicity reversal).
3. $y_5 = 1.10 > 1$ (out-of-bounds probability).

**(b)** Step-by-step PAVA execution:
- Initial clamping to $[0, 1]$:
  $$
  y_1 \leftarrow 0.00, \quad y_2 \leftarrow 0.35, \quad y_3 \leftarrow 0.25, \quad y_4 \leftarrow 0.80, \quad y_5 \leftarrow 1.00
  $$
  Initial blocks:
  $$
  B_1 = \{1\}, \bar{y}_1 = 0.00, W_1 = 1; \quad B_2 = \{2\}, \bar{y}_2 = 0.35, W_2 = 1;
  $$
  $$
  B_3 = \{3\}, \bar{y}_3 = 0.25, W_3 = 1; \quad B_4 = \{4\}, \bar{y}_4 = 0.80, W_4 = 1; \quad B_5 = \{5\}, \bar{y}_5 = 1.00, W_5 = 1
  $$
- Check $B_1$ vs $B_2$: $\bar{y}_1 = 0.00 \le \bar{y}_2 = 0.35$ (satisfied).
- Check $B_2$ vs $B_3$: $\bar{y}_2 = 0.35 > \bar{y}_3 = 0.25$ (violation!).
  Merge $B_2$ and $B_3$:
  $$
  B_{2,3} = \{2, 3\}, \quad W_{2,3} = 1 + 1 = 2, \quad \bar{y}_{2,3} = \frac{1(0.35) + 1(0.25)}{2} = \frac{0.60}{2} = 0.30
  $$
- Backtrack to check $B_1$ vs $B_{2,3}$: $\bar{y}_1 = 0.00 \le \bar{y}_{2,3} = 0.30$ (satisfied).
- Check $B_{2,3}$ vs $B_4$: $\bar{y}_{2,3} = 0.30 \le \bar{y}_4 = 0.80$ (satisfied).
- Check $B_4$ vs $B_5$: $\bar{y}_4 = 0.80 \le \bar{y}_5 = 1.00$ (satisfied).

The final blocks are:
$$
B_1 = \{1\} \implies x_1^* = 0.00
$$
$$
B_{2,3} = \{2, 3\} \implies x_2^* = x_3^* = 0.30
$$
$$
B_4 = \{4\} \implies x_4^* = 0.80
$$
$$
B_5 = \{5\} \implies x_5^* = 1.00
$$
Final corrected cumulative probability vector:
$$
\mathbf{x}^* = \begin{pmatrix} 0.00 \\ 0.30 \\ 0.30 \\ 0.80 \\ 1.00 \end{pmatrix}
$$

**(c)** KKT verification:
- Monotonicity: $0.00 \le 0.30 \le 0.30 \le 0.80 \le 1.00$ is satisfied.
- Box bounds: $0 \le x_k^* \le 1$ for all $k$ is satisfied.
- For the active constraint $x_2 = x_3$: $\frac{\partial}{\partial x_2} + \frac{\partial}{\partial x_3} = 2(0.30 - 0.35) + 2(0.30 - 0.25) = -0.10 + 0.10 = 0$. Stationary optimality holds.

---

#### Exercise 9.3: Hermite Expansion Coefficients of an Indicator Function
Let $Z = \Phi(Y)$ be a Gaussian anamorphosis. Consider the indicator transfer function $\psi(Y) = \mathbb{I}\{Y \le y_u\}$ where $y_u \in \mathbb{R}$ is the Gaussian quantile corresponding to cut-off $u$.
**(a)** Compute the Hermite expansion coefficient $\psi_0$.
**(b)** Using Rodrigues' formula, derive the analytical expression for $\psi_n$ for any $n \ge 1$.
**(c)** For $y_u = 0$ (the median), show which Hermite coefficients vanish.

##### Solution:
**(a)** By Definition 9.6.1:
$$
\psi_0 = \frac{1}{0!} \int_{-\infty}^\infty \mathbb{I}\{y \le y_u\} H_0(y) \phi(y) dy = \int_{-\infty}^{y_u} 1 \cdot \phi(y) dy = G(y_u)
$$
The zero-order coefficient is the marginal probability $G(y_u) = F(u)$.

**(b)** For $n \ge 1$, using Rodrigues' formula $H_n(y)\phi(y) = (-1)^n \frac{d^n\phi}{dy^n}(y)$:
$$
\psi_n = \frac{1}{n!} \int_{-\infty}^\infty \mathbb{I}\{y \le y_u\} H_n(y) \phi(y) dy = \frac{1}{n!} \int_{-\infty}^{y_u} (-1)^n \frac{d^n\phi}{dy^n}(y) dy
$$
Applying the Fundamental Theorem of Calculus:
$$
\int_{-\infty}^{y_u} \frac{d^n\phi}{dy^n}(y) dy = \left[ \frac{d^{n-1}\phi}{dy^{n-1}}(y) \right]_{-\infty}^{y_u} = \frac{d^{n-1}\phi}{dy^{n-1}}(y_u)
$$
since $\lim_{y \to -\infty} \frac{d^{n-1}\phi}{dy^{n-1}}(y) = 0$.
Now re-express $\frac{d^{n-1}\phi}{dy^{n-1}}(y_u)$ using Rodrigues' formula for order $n-1$:
$$
\frac{d^{n-1}\phi}{dy^{n-1}}(y_u) = (-1)^{n-1} H_{n-1}(y_u) \phi(y_u)
$$
Substituting this back into the expression for $\psi_n$:
$$
\psi_n = \frac{(-1)^n}{n!} \left((-1)^{n-1} H_{n-1}(y_u) \phi(y_u)\right) = \frac{(-1)^{2n-1}}{n!} H_{n-1}(y_u) \phi(y_u) = -\frac{1}{n!} H_{n-1}(y_u) \phi(y_u)
$$
This establishes the closed-form Hermite expansion of an indicator:
$$
\mathbb{I}\{Y \le y_u\} = G(y_u) - \phi(y_u) \sum_{n=1}^\infty \frac{H_{n-1}(y_u)}{n!} H_n(Y)
$$

**(c)** For $y_u = 0$:
$\phi(0) = \frac{1}{\sqrt{2\pi}}$ and $G(0) = 0.5$.
The Hermite polynomials evaluated at $0$:
$$
H_0(0) = 1, \quad H_1(0) = 0, \quad H_2(0) = -1, \quad H_3(0) = 0, \quad H_4(0) = 3, \quad H_5(0) = 0
$$
In general, $H_k(0) = 0$ whenever $k$ is odd.
Since $\psi_n = -\frac{1}{n!} H_{n-1}(0) \phi(0)$, $\psi_n = 0$ whenever $n-1$ is odd, which means **$\psi_n = 0$ for all even $n \ge 2$**:
$$
\psi_2 = 0, \quad \psi_4 = 0, \quad \psi_6 = 0, \dots
$$
The expansion at the median contains strictly odd Hermite polynomials:
$$
\mathbb{I}\{Y \le 0\} = 0.5 - \frac{1}{\sqrt{2\pi}}\left[ H_1(Y) - \frac{1}{6} H_3(Y) + \frac{3}{120} H_5(Y) - \dots \right]
$$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 9: Non-Gaussian and Nonlinear Spatial Estimation — Companion Exercises and Complete Solutions](ch09_exercises.md)
