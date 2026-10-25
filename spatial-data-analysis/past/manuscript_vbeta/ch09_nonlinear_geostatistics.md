# Chapter 9: Non-Gaussian and Nonlinear Spatial Estimation

In environmental science, hydrogeology, epidemiology, and mineral exploration, spatial observations rarely follow standard Gaussian distributions. Soil contaminant concentrations exhibit heavy positive skewness with localized spikes; hydraulic conductivity in fractured aquifers spans four to six orders of magnitude; and ore deposit grades are characterized by long tails of extreme enrichment interspersed with barren host rock. When decisions depend on non-linear thresholds—such as whether a local pollutant concentration exceeds a regulatory safety standard ($Z(\mathbf{s}) > z_c$), whether an ore block can be economically mined ($Z_v > z_c$), or what volume of contaminated soil requires remediation—classical Best Linear Unbiased Prediction (Ordinary and Simple Kriging) encounters fundamental mathematical and operational limits.

By the end of this chapter, you will be able to formulate and solve spatial estimation problems for non-Gaussian fields, predict non-linear quantities such as conditional exceedance probabilities, explain why naive plug-in predictions from smooth kriging maps systematically fail, implement isotonic projection via the Pool-Adjacent-Violators Algorithm (PAVA) to eliminate order-relation violations, fit Hermite polynomial expansions for Gaussian anamorphosis, and decouple high-dimensional non-linear estimation into independent spatial kriging systems using Disjunctive Kriging.

### Learning Outcomes
1. **Diagnose the Limits of Linear Kriging:** Demonstrate why plugging linear kriging predictions into non-linear functions $\psi(Z)$ introduces conditional Jensen bias and causes severe misclassification of threshold exceedance probabilities.
2. **Implement Lognormal Kriging:** Derive the exact moment relations and back-transformation corrections for Simple and Ordinary Lognormal Kriging, proving how the exponential transformation lifts conditional medians to unbiased arithmetic expectations.
3. **Analyze Indicator Random Fields:** Formulate spatial indicators $I(\mathbf{s}; u) = \mathbb{I}\{Z(\mathbf{s}) \le u\}$, derive their first- and second-order spatial moments, and prove the destructuration of extreme geometric excursions in Gaussian-type fields.
4. **Evaluate Indicator Kriging:** Implement Simple, Ordinary, and Median Indicator Kriging to estimate conditional cumulative distribution functions (CCDFs), and extract risk-qualified metrics including exceedance probabilities, E-type conditional means, and conditional quantiles.
5. **Correct Order-Relation Violations:** Formulate order-relation violations as a constrained quadratic optimization problem, implement the Pool-Adjacent-Violators Algorithm (PAVA) using the optimal pool-then-clamp sequence, and prove that adjacent averaging strictly minimizes weighted squared distance.
6. **Construct Gaussian Anamorphosis Models:** Transform non-Gaussian spatial data into standard Gaussian scores using empirical normal scores, fit orthogonal Hermite polynomial expansions via numerical quadrature, verify expansion monotonicity, and quantify the proportion of sample variance captured.
7. **Solve Disjunctive Kriging Systems:** Apply Mehler's formula to decouple bivariate Gaussian anamorphosis into independent linear kriging systems across polynomial degrees, proving that isofactorial representations eliminate cross-degree covariance terms.

---

### Chapter at a Glance: Non-Gaussian and Nonlinear Spatial Estimation

| Method / Model | Underlying Assumption | Primary Output | Computational Mechanism | Primary Failure Mode / Diagnostic |
| :--- | :--- | :--- | :--- | :--- |
| **Lognormal Kriging** | Multi-lognormal random field ($Y = \ln Z$ is Gaussian) | Conditional mean $\operatorname{E}[Z_0 \mid \mathbf{Z}]$ and log-scale variance | Standard linear kriging on logs followed by exponential back-transform with $\frac{1}{2}\sigma^2$ variance lift | Extreme sensitivity to near-zero outliers; back-transform explodes when $\sigma^2$ is large. Check normal Q–Q plot of logs. |
| **Indicator Kriging (IK)** | Bivariate indicator stationarity across $K$ cut-off thresholds | Pointwise conditional cumulative distribution $\widehat{F}(u \mid \mathbf{Z})$ | Linear Probability Model (LPM) kriged at each threshold independently using indicator variograms $\gamma_I(\mathbf{h}; u)$ | Order-relation violations ($\widehat{F}(u_k) > \widehat{F}(u_{k+1})$ or $\widehat{F} \notin [0,1]$). Correct via PAVA. |
| **Median Indicator Kriging (MIK)** | Mosaic / partition model (indicator correlation shape is constant across all thresholds) | Approximate CCDF $\widehat{F}(u \mid \mathbf{Z})$ | Single indicator variogram model inverted once; scalar indicator data updated per threshold | Destructuration failure: invalid if physical extremes have shorter spatial ranges than median grades. |
| **Gaussian Anamorphosis & Disjunctive Kriging (DK)** | Bivariate Gaussianity of normal scores ($Y(\mathbf{s})$ is isofactorial) | Minimum-variance additive non-linear predictor $\sum_{i=1}^n f_i(Z(\mathbf{s}_i))$ | Expansion into Hermite polynomials $H_k$; Mehler's formula decouples into $K$ separate $n \times n$ linear kriging systems | Non-monotonicity of truncated Hermite series; poor tail extrapolation beyond empirical sample range. |

---

### 9.1 Non-Gaussian Fields, Threshold Exceedances, and the Limits of Linear Kriging

#### The Classical-to-Spatial Bridge: From Classical Regression to Spatial Indicators

In classical statistics, binary outcomes and bounded probabilities are never modeled with ordinary linear regression. When an analyst fits an Ordinary Least Squares (OLS) model to a binary indicator $Y \in \{0, 1\}$ against a continuous covariate $X$:
$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$
the result is the well-known **Linear Probability Model (LPM)**. The LPM exhibits two critical pathologies:
1. **Predictions Violate Probability Bounds:** Because the regression line $\widehat{Y} = \widehat{\beta}_0 + \widehat{\beta}_1 X$ is unconstrained, extreme values of $X$ produce predicted probabilities that fall strictly outside the valid probability simplex $[0, 1]$ (predicting negative probabilities or probabilities exceeding 100%).
2. **Heteroskedastic Residuals:** The variance of a Bernoulli random variable is strictly tied to its mean: $\operatorname{Var}(Y \mid X) = p(X)(1 - p(X))$. Consequently, error variance is non-constant across covariate space.

Classical statistics resolves the LPM's boundary violations by introducing non-linear link functions—such as the standard normal cumulative distribution function $\Phi(\cdot)$ in **probit regression** or the logistic function $\sigma(z) = 1/(1 + e^{-z})$ in **logistic regression**:
$$
P(Y = 1 \mid X) = \Phi(\beta_0 + \beta_1 X) \quad \text{or} \quad P(Y = 1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
$$
The S-shaped sigmoid link function naturally constrains predictions to the open unit interval $(0, 1)$, dampening extreme predictions as they approach 0 or 1.

Furthermore, when modeling non-linear responses of a continuous predictor $X$, classical statistics relies on **orthogonal polynomial regression** (e.g., using orthogonal polynomials via `poly(x, degree)` in R). Expanding the predictor in a basis of orthogonal polynomials:
$$
f(X) = \sum_{k=0}^K c_k P_k(X)
$$
ensures that the columns of the regression design matrix are mutually orthogonal. As a result, the classical normal equations decouple into completely independent scalar equations, allowing each polynomial expansion coefficient $c_k$ to be estimated without affecting the others.

In spatial statistics, exactly the same mathematical challenges arise when dealing with non-Gaussian spatial distributions and non-linear physical decisions:
- **Indicator Kriging (IK)** (§9.4) is precisely the spatial analog of the **Linear Probability Model**: it fits a linear spatial predictor directly to binary indicator data. Consequently, just like the classical LPM, Indicator Kriging naturally produces out-of-bounds probabilities ($\widehat{F} < 0$ or $\widehat{F} > 1$) and monotonicity reversals ($\widehat{F}(u_1) > \widehat{F}(u_2)$ for $u_1 < u_2$), known as **order-relation violations** (§9.5).
- **Disjunctive Kriging (DK)** (§9.7) is the spatial analog of **orthogonal polynomial regression**: by transforming continuous data into standard normal scores through **Gaussian anamorphosis** (§9.6) and expanding the non-linear transfer function in orthogonal Hermite polynomials, Mehler's formula decouples the complex $n$-point non-linear spatial estimation problem into $K$ completely independent linear kriging systems.

| Classical Regression Concept | Spatial Geostatistical Analog | Shared Mathematical Engine | Shared Operational Risk / Failure |
| :--- | :--- | :--- | :--- |
| **Linear Probability Model (LPM)** | **Indicator Kriging (IK)** | Unconstrained linear prediction of binary indicator variables | Predictions spill outside $[0, 1]$; negative probabilities |
| **Monotone Link Function (Probit/Logit)** | **Isotonic Projection via PAVA** | Non-linear mapping enforcing bounds and monotonic ordering | Requires post-hoc projection or non-linear numerical optimization |
| **Orthogonal Polynomial Regression** | **Disjunctive Kriging (DK)** | Expansion in orthogonal polynomial basis (Hermite polynomials) | Decouples normal equations into independent scalar/subsystem solves |
| **Generalized Linear Models (GLMs)** | **Spatial GLMs / Model-Based Geostatistics** | Likelihood-based inference with non-linear link functions | Computationally demanding spatial MCMC or INLA (Chapter 12) |

---

#### The Limits of Plug-In Linear Kriging

Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a strictly stationary spatial random field. Suppose we observe spatial data $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ at $n$ discrete sample locations. In physical operations, decisions are rarely based on the raw value $Z(\mathbf{s}_0)$ alone. Instead, operational decisions depend on a non-linear transformation $\psi(Z(\mathbf{s}_0))$:
1. **Threshold Exceedance:** $\psi(Z) = \mathbb{I}\{Z \ge z_c\}$, indicating whether a contaminant concentration exceeds a statutory toxicity limit.
2. **Recovered Metal / Economic Value:** $\psi(Z) = (Z - z_c)^+ = \max(0, Z - z_c)$, representing the recovered ore grade above an economic cut-off threshold $z_c$.
3. **Power-Law Flow Resistance:** $\psi(Z) = Z^p$ (with $p \approx 3$ in hydrogeology), modeling hydraulic permeability from porosity.

##### Definition 9.1.1 (Minimum Mean Squared Error Nonlinear Predictor)
The optimal predictor of the non-linear quantity $\psi(Z(\mathbf{s}_0))$ among all functions of the data $\mathbf{Z}$, minimizing expected squared error:
$$
\operatorname{MSE}(\widehat{\psi}) = \operatorname{E}\left[ (\psi(Z(\mathbf{s}_0)) - \widehat{\psi}(\mathbf{Z}))^2 \right]
$$
is the **conditional expectation**:
$$
\widehat{\psi}_{\text{MMSE}}(\mathbf{s}_0) = \operatorname{E}\left[ \psi(Z(\mathbf{s}_0)) \mid \mathbf{Z} \right]
$$

A ubiquitous temptation in applied spatial statistics is the **plug-in approach**: first compute the standard linear kriging estimate $\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$, and then plug this smoothed estimate into the non-linear function:
$$
\widehat{\psi}_{\text{plug-in}}(\mathbf{s}_0) = \psi(\widehat{Z}_{\text{OK}}(\mathbf{s}_0))
$$
The following proposition establishes why the plug-in approach is fundamentally flawed.

##### Proposition 9.1.1 (Suboptimality of Plug-In Linear Kriging)
Let $\mathbf{Z}$ denote sample data, and let $\psi: \mathbb{R} \to \mathbb{R}$ be a non-linear transfer function.
1. **Conditional Jensen Bias:** If $\psi$ is strictly convex (such as $\psi(z) = (z - z_c)^+$ or $\psi(z) = e^z$) and the conditional distribution $Z(\mathbf{s}_0) \mid \mathbf{Z}$ has non-zero conditional variance $\sigma^2(\mathbf{s}_0) > 0$, then by Jensen's inequality:
   $$
   \operatorname{E}\left[ \psi(Z(\mathbf{s}_0)) \mid \mathbf{Z} \right] > \psi\left( \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] \right)
   $$
   If the random field is Gaussian, the Simple Kriging predictor $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ equals the conditional mean $\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]$, implying:
   $$
   \operatorname{E}\left[ \psi(Z(\mathbf{s}_0)) \mid \mathbf{Z} \right] > \psi(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))
   $$
   Plug-in linear kriging systematically underestimates the conditional expectation of convex physical attributes.
2. **Step-Function / Threshold Collapse:** If $\psi(z) = \mathbb{I}\{z \ge z_c\}$ is a binary indicator, plug-in linear kriging outputs only the degenerate binary values $\mathbb{I}\{\widehat{Z}_{\text{OK}}(\mathbf{s}_0) \ge z_c\} \in \{0, 1\}$, completely discarding the conditional uncertainty $\sigma_{\text{OK}}^2(\mathbf{s}_0)$.

##### Proof:
Part 1 follows directly from the definition of strict convexity: for any non-degenerate random variable $X$ with non-zero variance and strictly convex $\psi$, $\operatorname{E}[\psi(X)] > \psi(\operatorname{E}[X])$. Conditioning on $\mathbf{Z}$ and taking $X = Z(\mathbf{s}_0) \mid \mathbf{Z}$ yields the result. Under multivariate Gaussianity, conditional expectation is linear and coincides with Simple Kriging: $\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] = \widehat{Z}_{\text{SK}}(\mathbf{s}_0)$.

For Part 2, because $\widehat{Z}_{\text{OK}}(\mathbf{s}_0)$ is a single deterministic real number given the data $\mathbf{Z}$, evaluating $\psi(\widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = \mathbb{I}\{\widehat{Z}_{\text{OK}}(\mathbf{s}_0) \ge z_c\}$ must evaluate to either 0 or 1. However, the true conditional expectation is a continuous probability:
$$
\operatorname{E}\left[ \mathbb{I}\{Z(\mathbf{s}_0) \ge z_c\} \mid \mathbf{Z} \right] = P(Z(\mathbf{s}_0) \ge z_c \mid \mathbf{Z}) \in [0, 1]
$$
$\blacksquare$

---

#### The Thresholding Collapse: A Concrete Diagnostic

To understand the practical catastrophe of plug-in linear kriging, consider an environmental remediation site where soil must be excavated if arsenic concentration exceeds a regulatory threshold $z_c = 100\text{ mg/kg}$. 

Suppose at an unsampled location $\mathbf{s}_0$, the local kriging estimate is $\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = 95\text{ mg/kg}$ with a kriging standard deviation $\sigma_{\text{OK}}(\mathbf{s}_0) = 40\text{ mg/kg}$. Assuming a Gaussian conditional distribution:
- **Plug-In Prediction:** The analyst checks whether the kriged map exceeds the threshold:
  $$
  \psi(\widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = \mathbb{I}\{95 \ge 100\} = 0
  $$
  The plug-in approach concludes that the site is safe and requires no remediation (predicted exceedance probability $= 0\%$).
- **True Conditional Exceedance Probability:** Accounting for spatial estimation uncertainty:
  $$
  P(Z(\mathbf{s}_0) \ge 100 \mid \mathbf{Z}) = 1 - \Phi\left( \frac{100 - 95}{40} \right) = 1 - \Phi(0.125) \approx 1 - 0.5498 = 0.4502 \quad (45.0\%)
  $$

The true probability of exceeding the toxic threshold is **$45.0\%$**—nearly a coin flip! Yet the plug-in kriging map reported zero risk. Because kriging is a spatial smoothing operator (Chapter 5), it pulls extreme values toward the local mean, dampening peaks and filling valleys. When smoothed values are fed into discontinuous step functions, the smoothing artificially categorizes borderline contaminated regions as entirely benign.

#### Decision Workflow: Which Non-Linear Estimation Question Are You Asking?

| Decision / Question | Target Quantity | Recommended Mathematical Tool | Primary Section |
| :--- | :--- | :--- | :--- |
| "What is the expected grade or concentration when data are lognormally skewed?" | Conditional mean $\operatorname{E}[Z_0 \mid \mathbf{Z}]$ | **Lognormal Kriging** (with exact back-transform variance lift) | §9.2 |
| "What is the probability that a pollutant exceeds a statutory safety limit $z_c$?" | Exceedance probability $P(Z_0 > z_c \mid \mathbf{Z})$ | **Indicator Kriging (IK)** or **Disjunctive Kriging (DK)** of indicator | §9.4, §9.7 |
| "How do we reconstruct the full local uncertainty distribution without distributional assumptions?" | Conditional Cumulative Distribution Function $\widehat{F}(u \mid \mathbf{Z})$ | **Indicator Kriging + PAVA Monotonicity Correction** | §9.4, §9.5 |
| "What is the optimal non-linear predictor for arbitrary non-Gaussian continuous attributes?" | Non-linear additive predictor $\sum f_i(Z_i)$ | **Gaussian Anamorphosis & Disjunctive Kriging** | §9.6, §9.7 |
| "How do we model joint spatial uncertainty of a non-linear functional over an entire region simultaneously?" | Joint spatial distribution of $\psi(Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_M))$ | **Conditional Spatial Simulation** (cannot use pointwise kriging) | Chapter 10 |

---

### 9.2 Lognormal Spatial Fields and Lognormal Kriging

When spatial attributes are strictly positive and exhibit strong positive skewness spanning multiple orders of magnitude, a logarithmic transformation $Y(\mathbf{s}) = \ln Z(\mathbf{s})$ frequently normalizes the marginal and bivariate distributions. 

##### Definition 9.2.1 (Lognormal Random Field)
A strictly positive spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is a **stationary lognormal random field** if its log-transformed field:
$$
Y(\mathbf{s}) = \ln Z(\mathbf{s})
$$
is a stationary Gaussian random field with constant mean $\mu_Y$, stationary variance $\sigma_Y^2$, and stationary spatial covariance $C_Y(\mathbf{h}) = \operatorname{Cov}(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h}))$.

---

#### Moment Relations of Lognormal Spatial Fields

Because $Y(\mathbf{s})$ is Gaussian, its moment generating function is given by:
$$
M_Y(t) = \operatorname{E}\left[ e^{t Y(\mathbf{s})} \right] = \exp\left( t \mu_Y + \frac{1}{2} t^2 \sigma_Y^2 \right)
$$
Evaluating $M_Y(t)$ at integer values yields the exact spatial moments of $Z(\mathbf{s})$.

##### Theorem 9.2.1 (Moment Relations of Lognormal Spatial Fields)
Let $Z(\mathbf{s}) = \exp(Y(\mathbf{s}))$ be a stationary lognormal spatial field. Then:
1. **Global Mean:**
   $$
   m_Z = \operatorname{E}[Z(\mathbf{s})] = M_Y(1) = \exp\left( \mu_Y + \frac{1}{2}\sigma_Y^2 \right)
   $$
2. **Global Variance:**
   $$
   \sigma_Z^2 = \operatorname{Var}(Z(\mathbf{s})) = m_Z^2 \left( e^{\sigma_Y^2} - 1 \right) = \exp\left( 2\mu_Y + \sigma_Y^2 \right) \left( e^{\sigma_Y^2} - 1 \right)
   $$
3. **Spatial Covariance Function:**
   $$
   C_Z(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h})) = m_Z^2 \left( e^{C_Y(\mathbf{h})} - 1 \right)
   $$
4. **Spatial Semivariogram Function:**
   $$
   \gamma_Z(\mathbf{h}) = m_Z^2 e^{\sigma_Y^2} \left( 1 - e^{-\gamma_Y(\mathbf{h})} \right)
   $$

##### Proof:
For Part 1, set $t = 1$ in the Gaussian MGF: $m_Z = \operatorname{E}[e^{Y(\mathbf{s})}] = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2)$.  
For Part 2, evaluate the second moment using $t = 2$:
$$
\operatorname{E}[Z(\mathbf{s})^2] = \operatorname{E}[e^{2Y(\mathbf{s})}] = \exp\left( 2\mu_Y + \frac{1}{2}(4)\sigma_Y^2 \right) = \exp\left( 2\mu_Y + 2\sigma_Y^2 \right)
$$
Subtracting the squared mean:
$$
\sigma_Z^2 = \operatorname{E}[Z(\mathbf{s})^2] - m_Z^2 = e^{2\mu_Y + 2\sigma_Y^2} - e^{2\mu_Y + \sigma_Y^2} = e^{2\mu_Y + \sigma_Y^2} \left( e^{\sigma_Y^2} - 1 \right) = m_Z^2 \left( e^{\sigma_Y^2} - 1 \right)
$$
For Part 3, consider the bivariate normal vector $(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h}))'$. The sum $Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h})$ is normally distributed with mean $2\mu_Y$ and variance:
$$
\operatorname{Var}(Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h})) = 2\sigma_Y^2 + 2 C_Y(\mathbf{h})
$$
Applying the univariate MGF formula to $t = 1$:
$$
\operatorname{E}[Z(\mathbf{s}) Z(\mathbf{s}+\mathbf{h})] = \operatorname{E}\left[ e^{Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h})} \right] = \exp\left( 2\mu_Y + \frac{1}{2}\left( 2\sigma_Y^2 + 2 C_Y(\mathbf{h}) \right) \right) = \exp\left( 2\mu_Y + \sigma_Y^2 + C_Y(\mathbf{h}) \right)
$$
Subtracting $m_Z^2 = \exp(2\mu_Y + \sigma_Y^2)$:
$$
C_Z(\mathbf{h}) = e^{2\mu_Y + \sigma_Y^2 + C_Y(\mathbf{h})} - e^{2\mu_Y + \sigma_Y^2} = m_Z^2 \left( e^{C_Y(\mathbf{h})} - 1 \right)
$$
For Part 4, use $\gamma_Z(\mathbf{h}) = C_Z(\mathbf{0}) - C_Z(\mathbf{h})$:
$$
\gamma_Z(\mathbf{h}) = m_Z^2 \left( e^{\sigma_Y^2} - 1 \right) - m_Z^2 \left( e^{C_Y(\mathbf{h})} - 1 \right) = m_Z^2 \left( e^{\sigma_Y^2} - e^{C_Y(\mathbf{h})} \right) = m_Z^2 e^{\sigma_Y^2} \left( 1 - e^{-( \sigma_Y^2 - C_Y(\mathbf{h}) )} \right)
$$
Since $\gamma_Y(\mathbf{h}) = \sigma_Y^2 - C_Y(\mathbf{h})$, we obtain $\gamma_Z(\mathbf{h}) = m_Z^2 e^{\sigma_Y^2} (1 - e^{-\gamma_Y(\mathbf{h})})$. $\blacksquare$

##### Corollary 9.2.1 (Covariance Inversion)
Given the stationary covariance $C_Z(\mathbf{h})$ and mean $m_Z$ of the original physical field, the underlying Gaussian covariance $C_Y(\mathbf{h})$ is obtained by algebraic inversion:
$$
C_Y(\mathbf{h}) = \ln\left( 1 + \frac{C_Z(\mathbf{h})}{m_Z^2} \right) = \ln\left( 1 + \rho_Z(\mathbf{h}) \cdot \operatorname{CV}_Z^2 \right)
$$
where $\operatorname{CV}_Z = \sigma_Z / m_Z$ is the coefficient of variation.

---

#### Simple Lognormal Kriging and Exact Back-Transformation

Suppose the log-scale mean $\mu_Y$ and covariance $C_Y(\mathbf{h})$ are known. Let $\mathbf{Y} = (\ln Z(\mathbf{s}_1), \dots, \ln Z(\mathbf{s}_n))'$ denote the observed log-data.

Because $\mathbf{Y}$ and $Y(\mathbf{s}_0)$ are jointly multivariate Gaussian, the conditional distribution of $Y(\mathbf{s}_0)$ given $\mathbf{Y}$ is strictly Gaussian:
$$
Y(\mathbf{s}_0) \mid \mathbf{Y} \sim \mathcal{N}\left( \widehat{Y}_{\text{SK}}(\mathbf{s}_0), \, \sigma_{\text{SK}}^2(\mathbf{s}_0) \right)
$$
where $\widehat{Y}_{\text{SK}}(\mathbf{s}_0)$ is the standard Simple Kriging predictor on the log scale and $\sigma_{\text{SK}}^2(\mathbf{s}_0)$ is the Simple Kriging variance:
$$
\widehat{Y}_{\text{SK}}(\mathbf{s}_0) = \mu_Y + \mathbf{c}_Y' \mathbf{V}_Y^{-1} (\mathbf{Y} - \mu_Y \mathbf{1}), \quad \sigma_{\text{SK}}^2(\mathbf{s}_0) = \sigma_Y^2 - \mathbf{c}_Y' \mathbf{V}_Y^{-1} \mathbf{c}_Y
$$

##### Theorem 9.2.2 (Simple Lognormal Kriging Back-Transformation)
Under the stationary lognormal model, the minimum mean squared error (MMSE) predictor of the original attribute $Z(\mathbf{s}_0) = \exp(Y(\mathbf{s}_0))$ is:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \operatorname{E}\left[ Z(\mathbf{s}_0) \mid \mathbf{Y} \right] = \exp\left( \widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0) \right)
$$
The corresponding conditional estimation error variance is:
$$
\operatorname{Var}\left( Z(\mathbf{s}_0) \mid \mathbf{Y} \right) = \left( \widehat{Z}_{\text{SK}}(\mathbf{s}_0) \right)^2 \left( \exp\left( \sigma_{\text{SK}}^2(\mathbf{s}_0) \right) - 1 \right)
$$

##### Proof:
Conditioned on $\mathbf{Y}$, $Y(\mathbf{s}_0)$ is Gaussian with conditional mean $\widehat{Y}_{\text{SK}}(\mathbf{s}_0)$ and conditional variance $\sigma_{\text{SK}}^2(\mathbf{s}_0)$. Applying the Gaussian moment generating function $M(t) = \operatorname{E}[e^{t X}] = \exp(t \mu + \frac{1}{2}t^2 \sigma^2)$ at $t = 1$ to the conditional distribution:
$$
\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}] = \operatorname{E}\left[ e^{Y(\mathbf{s}_0)} \mid \mathbf{Y} \right] = \exp\left( \widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0) \right)
$$
Similarly, the conditional second moment is obtained by setting $t = 2$:
$$
\operatorname{E}[Z(\mathbf{s}_0)^2 \mid \mathbf{Y}] = \exp\left( 2\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + 2\sigma_{\text{SK}}^2(\mathbf{s}_0) \right)
$$
Subtracting $(\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}])^2 = \exp(2\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \sigma_{\text{SK}}^2(\mathbf{s}_0))$ yields the conditional variance. $\blacksquare$

---

> [!NOTE]
> #### From the Lecture Hall: Jensen's Inequality and the Normal MGF
> Why does the exponential back-transform require adding $\frac{1}{2}\sigma^2$?
> 
> Because the exponential function $f(y) = e^y$ is strictly convex, taking the naive back-transform:
> $$
> \widehat{Z}_{\text{naive}}(\mathbf{s}_0) = \exp\left( \widehat{Y}(\mathbf{s}_0) \right)
> $$
> predicts the **conditional median**, not the conditional mean! 
> 
> For a lognormal distribution, the median is $\exp(\mu)$, while the arithmetic mean is $\exp(\mu + \frac{1}{2}\sigma^2)$. Because probability mass in a lognormal distribution is dragged out into a long right-hand tail, the arithmetic mean is strictly larger than the median. If you simply exponentiate your kriged log-estimate, you forecast the median. The additive variance correction term $+\frac{1}{2}\sigma^2$ is exactly the Jensen's inequality lift required to convert the conditional median into an unbiased arithmetic conditional expectation.

---

#### Ordinary Lognormal Kriging and the Lagrange Multiplier Correction

In practice, the log-scale mean $\mu_Y$ is unknown and must be filtered out using Ordinary Kriging on the log data:
$$
\widehat{Y}_{\text{OK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Y(\mathbf{s}_i) \quad \text{subject to } \sum_{i=1}^n \lambda_i = 1
$$
In our standard covariance formulation (Chapter 5), the Ordinary Kriging linear system is:
$$
\begin{pmatrix} \mathbf{V}_Y & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ \mu_L \end{pmatrix} = \begin{pmatrix} \mathbf{c}_Y \\ 1 \end{pmatrix}
$$
where $\mathbf{V}_Y \boldsymbol{\lambda} + \mu_L \mathbf{1} = \mathbf{c}_Y$ and $\mu_L$ is the Lagrange multiplier associated with the unbiasedness constraint $\mathbf{1}'\boldsymbol{\lambda} = 1$. The Ordinary Kriging variance on the log scale is:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = \sigma_Y^2 - \boldsymbol{\lambda}'\mathbf{c}_Y - \mu_L
$$

##### Theorem 9.2.3 (Ordinary Lognormal Kriging Back-Transformation)
Under the $+\mu_L$ covariance system convention $\mathbf{V}_Y \boldsymbol{\lambda} + \mu_L \mathbf{1} = \mathbf{c}_Y$, the globally unbiased back-transformed Ordinary Lognormal Kriging estimator is:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \exp\left( \widehat{Y}_{\text{OK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{OK}}^2(\mathbf{s}_0) + \mu_L \right)
$$
This estimator is unconditionally unbiased on average: $\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0)] = m_Z = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2)$.

##### Proof:
Because $\sum_{i=1}^n \lambda_i = 1$, the linear combination $\widehat{Y}_{\text{OK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Y(\mathbf{s}_i)$ has expectation:
$$
\operatorname{E}[\widehat{Y}_{\text{OK}}(\mathbf{s}_0)] = \sum_{i=1}^n \lambda_i \mu_Y = \mu_Y
$$
Its spatial variance is:
$$
\operatorname{Var}(\widehat{Y}_{\text{OK}}(\mathbf{s}_0)) = \boldsymbol{\lambda}' \mathbf{V}_Y \boldsymbol{\lambda} = \boldsymbol{\lambda}' (\mathbf{c}_Y - \mu_L \mathbf{1}) = \boldsymbol{\lambda}' \mathbf{c}_Y - \mu_L \sum_{i=1}^n \lambda_i = \boldsymbol{\lambda}' \mathbf{c}_Y - \mu_L
$$
From the OK variance equation, $\boldsymbol{\lambda}'\mathbf{c}_Y = \sigma_Y^2 - \sigma_{\text{OK}}^2 - \mu_L$. Substituting this identity into the variance expression:
$$
\operatorname{Var}(\widehat{Y}_{\text{OK}}(\mathbf{s}_0)) = (\sigma_Y^2 - \sigma_{\text{OK}}^2 - \mu_L) - \mu_L = \sigma_Y^2 - \sigma_{\text{OK}}^2 - 2\mu_L
$$
Since $\widehat{Y}_{\text{OK}}(\mathbf{s}_0)$ is a linear combination of jointly Gaussian variables, it is normally distributed: $\widehat{Y}_{\text{OK}} \sim \mathcal{N}(\mu_Y, \sigma_Y^2 - \sigma_{\text{OK}}^2 - 2\mu_L)$.  
Now evaluate the expected value of the naive exponential back-transform using the Gaussian MGF:
$$
\operatorname{E}\left[ \exp(\widehat{Y}_{\text{OK}}(\mathbf{s}_0)) \right] = \exp\left( \mu_Y + \frac{1}{2}\operatorname{Var}(\widehat{Y}_{\text{OK}}(\mathbf{s}_0)) \right) = \exp\left( \mu_Y + \frac{1}{2}(\sigma_Y^2 - \sigma_{\text{OK}}^2 - 2\mu_L) \right)
$$
Factoring out the true stationary mean $m_Z = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2)$:
$$
\operatorname{E}\left[ \exp(\widehat{Y}_{\text{OK}}(\mathbf{s}_0)) \right] = m_Z \cdot \exp\left( -\frac{1}{2}\sigma_{\text{OK}}^2 - \mu_L \right)
$$
Therefore, to ensure that the estimator is strictly unbiased on average ($\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0)] = m_Z$), we must multiply $\exp(\widehat{Y}_{\text{OK}})$ by $\exp(+\frac{1}{2}\sigma_{\text{OK}}^2 + \mu_L)$:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \exp\left( \widehat{Y}_{\text{OK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{OK}}^2(\mathbf{s}_0) + \mu_L \right)
$$
$\blacksquare$

---

#### Practical Implementation: 5-Step Lognormal Kriging Recipe

1. **Test Gaussianity of Logs:** Plot the normal Q–Q plot and histogram of $Y_i = \ln Z_i$. Check for severe outliers or bimodality.
2. **Model Log-Scale Semivariogram:** Compute and fit an authorized theoretical semivariogram $\gamma_Y(\mathbf{h})$ directly to the log-transformed observations $Y_i$.
3. **Solve Log-Scale Kriging:** Solve the Ordinary or Simple Kriging system at target location $\mathbf{s}_0$ to obtain weights $\boldsymbol{\lambda}$, estimate $\widehat{Y}(\mathbf{s}_0)$, estimation variance $\sigma_Y^2(\mathbf{s}_0)$, and Lagrange multiplier $\mu_L$ (for OK).
4. **Apply Analytical Variance Correction:** Compute the back-transform exponent $\widehat{Y} + \frac{1}{2}\sigma^2$ (for SK) or $\widehat{Y} + \frac{1}{2}\sigma^2 + \mu_L$ (for OK).
5. **Evaluate Stability Diagnostic:** Compute the back-transformation lift ratio $R = \exp(\frac{1}{2}\sigma^2)$. If $R > 3$ (occurring when $\sigma^2 > 2.2$), the estimate is dominated by the variance correction rather than local data. In such high-variance zones, flag the estimate for potential instability.

> [!WARNING]
> #### Operational Hazards of Lognormal Kriging
> 1. **Extreme Sensitivity to Small Values:** A single sample near detection limit (e.g., $Z_i = 0.001 \implies Y_i = -6.91$) acts as a massive negative outlier in log-space, severely distorting variogram models and pulling down local estimates.
> 2. **Variance Explosion in Sparse Data:** At locations distant from sample data where $\sigma_{\text{OK}}^2$ approaches the sill $\sigma_Y^2$, the exponential lift factor $\exp(\frac{1}{2}\sigma_{\text{OK}}^2)$ can produce astronomical grade predictions. Lognormal kriging must always be capped or verified against global reserve constraints.
> 3. **Connection to Indicators:** Notice that monotone transformations preserve threshold indicator functions exactly:
>    $$
>    \mathbb{I}\{Z(\mathbf{s}) \le u\} = \mathbb{I}\{\ln Z(\mathbf{s}) \le \ln u\} = \mathbb{I}\{Y(\mathbf{s}) \le y_u\}
>    $$
>    Consequently, a lognormal spatial field possesses exactly the same spatial indicator structure and geometric connectivity properties as its underlying Gaussian field!

---

### 9.3 Indicator Random Fields and Spatial Threshold Geometry

Rather than transforming continuous data to fit a parametric distribution, **indicator geostatistics** transforms continuous attributes into binary spatial flags across cut-off thresholds.

##### Definition 9.3.1 (Indicator Random Field)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a spatial random field. For any fixed cut-off threshold $u \in \mathbb{R}$, the **indicator random field** $I(\mathbf{s}; u)$ is defined as:
$$
I(\mathbf{s}; u) = \mathbb{I}\{Z(\mathbf{s}) \le u\} = \begin{cases} 1 & \text{if } Z(\mathbf{s}) \le u \\ 0 & \text{if } Z(\mathbf{s}) > u \end{cases}
$$

---

#### First- and Second-Order Moments of Indicators

##### Proposition 9.3.1 (Moments of Indicator Random Fields)
Let $Z(\mathbf{s})$ be a strictly stationary spatial random field with marginal cumulative distribution function $F(u) = P(Z(\mathbf{s}) \le u)$. For any fixed threshold $u$:
1. **Indicator Expectation:**
   $$
   \operatorname{E}[I(\mathbf{s}; u)] = P(Z(\mathbf{s}) \le u) = F(u)
   $$
2. **Indicator Variance:**
   $$
   \operatorname{Var}(I(\mathbf{s}; u)) = F(u)(1 - F(u))
   $$
3. **Indicator Covariance Function:**
   $$
   C_I(\mathbf{h}; u) = \operatorname{Cov}(I(\mathbf{s}; u), I(\mathbf{s}+\mathbf{h}; u)) = P(Z(\mathbf{s}) \le u, Z(\mathbf{s}+\mathbf{h}) \le u) - [F(u)]^2
   $$
4. **Indicator Semivariogram Function:**
   $$
   \gamma_I(\mathbf{h}; u) = \frac{1}{2}\operatorname{Var}(I(\mathbf{s}+\mathbf{h}; u) - I(\mathbf{s}; u)) = \frac{1}{2} P\left( I(\mathbf{s}; u) \ne I(\mathbf{s}+\mathbf{h}; u) \right)
   $$
   The indicator semivariance $\gamma_I(\mathbf{h}; u)$ equals exactly half the probability that the two points $\mathbf{s}$ and $\mathbf{s}+\mathbf{h}$ fall on opposite sides of the cut-off threshold $u$.

##### Proof:
Since $I(\mathbf{s}; u) \in \{0, 1\}$ is a Bernoulli random variable with success parameter $p = P(Z(\mathbf{s}) \le u) = F(u)$, its expectation is $p$ and its variance is $p(1-p)$, proving Parts 1 and 2.  
For Part 3, note that $I(\mathbf{s}; u) I(\mathbf{s}+\mathbf{h}; u) = 1$ if and only if both $Z(\mathbf{s}) \le u$ and $Z(\mathbf{s}+\mathbf{h}) \le u$. Hence $\operatorname{E}[I(\mathbf{s}; u) I(\mathbf{s}+\mathbf{h}; u)] = P(Z(\mathbf{s}) \le u, Z(\mathbf{s}+\mathbf{h}) \le u)$, and subtracting $\operatorname{E}[I]^2 = F(u)^2$ yields $C_I(\mathbf{h}; u)$.  
For Part 4, evaluate the squared difference $(I(\mathbf{s}+\mathbf{h}; u) - I(\mathbf{s}; u))^2$. Since $I \in \{0, 1\}$, the difference is non-zero (and equals 1) if and only if one indicator is 1 and the other is 0. Taking expectations:
$$
\operatorname{E}\left[ (I(\mathbf{s}+\mathbf{h}; u) - I(\mathbf{s}; u))^2 \right] = P(I(\mathbf{s}+\mathbf{h}; u) \ne I(\mathbf{s}; u))
$$
Multiplying by $\frac{1}{2}$ yields $\gamma_I(\mathbf{h}; u)$. $\blacksquare$

---

#### Destructuration of Extreme Thresholds in Gaussian Fields

A fundamental geometric discovery of non-linear geostatistics is that the spatial correlation of indicator variables depends intimately on the threshold $u$ and the bivariate copula of the random field.

##### Theorem 9.3.1 (Destructuration of Extreme Thresholds)
Let $Y(\mathbf{s})$ be a stationary standard Gaussian random field with continuous correlation function $\rho(\mathbf{h}) \in (-1, 1)$. For any non-zero spatial lag $\mathbf{h}$ where $\rho(\mathbf{h}) < 1$:
$$
\lim_{y \to -\infty} \rho_I(\mathbf{h}; y) = 0 \quad \text{and} \quad \lim_{y \to +\infty} \rho_I(\mathbf{h}; y) = 0
$$
where $\rho_I(\mathbf{h}; y) = C_I(\mathbf{h}; y) / [\Phi(y)(1 - \Phi(y))]$ is the indicator correlation.

##### Proof:
Consider two standard normal variables $Y_1 = Y(\mathbf{s})$ and $Y_2 = Y(\mathbf{s}+\mathbf{h})$ with correlation $\rho \in (-1, 1)$. The sum $Y_1 + Y_2$ is normally distributed with mean 0 and variance $2(1+\rho)$.  
For any $y < 0$, if both $Y_1 \le y$ and $Y_2 \le y$, then their sum must satisfy $Y_1 + Y_2 \le 2y$. Therefore:
$$
P(Y_1 \le y, Y_2 \le y) \le P(Y_1 + Y_2 \le 2y) = \Phi\left( \frac{2y}{\sqrt{2(1+\rho)}} \right) = \Phi\left( y \sqrt{\frac{2}{1+\rho}} \right)
$$
Because $|\rho| < 1$, the factor $\alpha = \sqrt{\frac{2}{1+\rho}} > 1$.  
Using the standard Gaussian tail asymptotic bound $\Phi(x) \sim \frac{1}{\sqrt{2\pi}|x|} e^{-x^2/2}$ as $x \to -\infty$:
$$
\Phi(\alpha y) \sim \frac{1}{\sqrt{2\pi}\alpha |y|} \exp\left( -\frac{\alpha^2 y^2}{2} \right) = \frac{1}{\sqrt{2\pi}\alpha |y|} \exp\left( -\frac{y^2}{1+\rho} \right)
$$
Comparing this with the denominator factor $\Phi(y) \sim \frac{1}{\sqrt{2\pi}|y|} e^{-y^2/2}$:
$$
\frac{P(Y_1 \le y, Y_2 \le y)}{\Phi(y)} \le \frac{\Phi(\alpha y)}{\Phi(y)} \sim \frac{1}{\alpha} \exp\left( -y^2 \left[ \frac{1}{1+\rho} - \frac{1}{2} \right] \right) = \frac{1}{\alpha} \exp\left( -y^2 \frac{1-\rho}{2(1+\rho)} \right)
$$
Because $\rho < 1$, the exponent is strictly negative, so the ratio decays exponentially to zero as $y \to -\infty$:
$$
\lim_{y \to -\infty} \frac{P(Y_1 \le y, Y_2 \le y)}{\Phi(y)} = 0
$$
Expanding the indicator correlation:
$$
\rho_I(\mathbf{h}; y) = \frac{P(Y_1 \le y, Y_2 \le y) - \Phi(y)^2}{\Phi(y)(1 - \Phi(y))} = \frac{1}{1 - \Phi(y)} \left( \frac{P(Y_1 \le y, Y_2 \le y)}{\Phi(y)} - \Phi(y) \right) \xrightarrow{y \to -\infty} 0
$$
Symmetry of the Gaussian distribution establishes the identical limit as $y \to +\infty$. $\blacksquare$

---

#### Numerical Demonstration of Destructuration

To visualize destructuration, consider two locations with strong underlying Gaussian correlation $\rho = 0.50$. The table below shows the bivariate exceedance probability and resulting indicator correlation across standardized thresholds $y$:

| Gaussian Threshold $y$ | Quantile $p = \Phi(y)$ | Joint Probability $P(Y_1 \le y, Y_2 \le y)$ | Indicator Correlation $\rho_I$ | Physical Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| **0.0** (Median) | $0.5000$ | $0.3333$ | **$0.3333$** | Strong spatial continuity across median boundaries |
| **-1.0** | $0.1587$ | $0.0617$ | **$0.2748$** | Moderate continuity in moderate tails |
| **-2.0** | $0.0228$ | $0.0039$ | **$0.1568$** | Continuity begins to collapse |
| **-3.0** | $0.00135$ | $0.000085$ | **$0.0594$** | Extreme low values exhibit very weak spatial structure |
| **-5.0** | $2.87 \times 10^{-7}$ | $8.23 \times 10^{-10}$ | **$0.0029$** | High/low excursions appear as isolated, scattered patches |

#### The Mosaic Counterexample

In stark contrast to Gaussian random fields, **mosaic random fields** (such as random Voronoi tessellations where each geometric cell is assigned an independent mark from $F$) do **not** destructure. In a mosaic model, two points fall in the same cell with probability $\rho_M(\mathbf{h})$, in which case they share the same value. If they fall in different cells, their values are independent. Hence:
$$
P(Z(\mathbf{s}) \le u, Z(\mathbf{s}+\mathbf{h}) \le u) = \rho_M(\mathbf{h}) F(u) + (1 - \rho_M(\mathbf{h})) F(u)^2
$$
Subtracting $F(u)^2$:
$$
C_I(\mathbf{h}; u) = \rho_M(\mathbf{h}) F(u)(1 - F(u)) \implies \rho_I(\mathbf{h}; u) = \rho_M(\mathbf{h}) \quad \text{for all } u!
$$
In a mosaic field, the indicator correlation is completely independent of the cut-off threshold.

> [!TIP]
> #### Practical Diagnostic: Gaussian vs. Mosaic Architecture
> When exploratory data analysis reveals heavy-tailed spatial data, compute empirical indicator semivariograms at the 10th, 50th, and 90th percentiles:
> - **Multi-Gaussian Architecture:** The correlation range collapses dramatically at the 10th and 90th percentiles compared to the median. Lognormal kriging (§9.2), Gaussian anamorphosis (§9.6), and Disjunctive Kriging (§9.7) are physically appropriate.
> - **Mosaic Architecture:** The correlation range remains roughly constant across all percentiles. Disjunctive kriging and Gaussian models will artificially destructure connected high-grade channels. In this setting, **Median Indicator Kriging (MIK)** or multiple-point geostatistics should be preferred.

---

### 9.4 Indicator Kriging (IK)

Rather than assuming a global parametric distribution, **Indicator Kriging (IK)** estimates the local Conditional Cumulative Distribution Function (CCDF) non-parametrically across a discrete sequence of $K$ ordered thresholds:
$$
u_1 < u_2 < \dots < u_K
$$
At each sample location $\mathbf{s}_i$, the continuous datum $Z(\mathbf{s}_i)$ is encoded into $K$ binary indicator values:
$$
i(\mathbf{s}_i; u_k) = \mathbb{I}\{Z(\mathbf{s}_i) \le u_k\} \in \{0, 1\}, \quad k = 1, \dots, K
$$

##### Definition 9.4.1 (Simple and Ordinary Indicator Kriging)
Let $u$ be a fixed threshold.
1. **Simple Indicator Kriging (SIK):** Assuming the marginal stationary CDF $F(u) = P(Z(\mathbf{s}) \le u)$ is known globally:
   $$
   \widehat{F}_{\text{SIK}}(u \mid \mathbf{Z}) = F(u) + \sum_{i=1}^n \lambda_i(u) \left( i(\mathbf{s}_i; u) - F(u) \right)
   $$
   where the SIK weights satisfy $\mathbf{\Gamma}_I(u) \boldsymbol{\lambda}(u) = \boldsymbol{\gamma}_{I,0}(u)$.
2. **Ordinary Indicator Kriging (OIK):** Filtering an unknown local baseline probability via the constraint $\sum_{i=1}^n \lambda_i(u) = 1$:
   $$
   \widehat{F}_{\text{OIK}}(u \mid \mathbf{Z}) = \sum_{i=1}^n \lambda_i(u) \, i(\mathbf{s}_i; u)
   $$
   where weights satisfy the standard Ordinary Kriging system:
   $$
   \begin{pmatrix} \mathbf{\Gamma}_I(u) & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda}(u) \\ \mu_I(u) \end{pmatrix} = \begin{pmatrix} \boldsymbol{\gamma}_{I,0}(u) \\ 1 \end{pmatrix}
   $$
   and $\mathbf{\Gamma}_I(u)$ is the $n \times n$ semivariogram matrix evaluated with the threshold-specific indicator semivariogram $\gamma_I(\mathbf{h}; u)$.

##### Term-by-Term Equation Anatomy: Indicator Kriging
$$
\underbrace{\widehat{F}(u \mid \mathbf{Z})}_{\text{Estimated Non-Exceedance Probability}} = \sum_{i=1}^n \underbrace{\lambda_i(u)}_{\text{Threshold-Specific Weight}} \cdot \underbrace{\mathbb{I}\{Z(\mathbf{s}_i) \le u\}}_{\text{Binary Sample Indicator}}
$$
- $\widehat{F}(u \mid \mathbf{Z}) = \widehat{P}(Z(\mathbf{s}_0) \le u \mid \mathbf{Z})$: The estimated non-exceedance probability (cumulative distribution function) at threshold $u$.
- $\lambda_i(u)$: Spatial weights derived by solving the linear kriging system with the indicator semivariogram $\gamma_I(\mathbf{h}; u)$. Because each threshold has a unique spatial continuity structure, weights vary across thresholds: $\boldsymbol{\lambda}(u_k) \ne \boldsymbol{\lambda}(u_j)$.
- $\mathbb{I}\{Z(\mathbf{s}_i) \le u\}$: Binary data value, equal to 1 if sample $Z(\mathbf{s}_i)$ falls at or below $u$, and 0 otherwise.

> [!IMPORTANT]
> #### What Indicator Kriging Does and Does Not Estimate
> It is essential to recognize that Indicator Kriging does **not** evaluate the full conditional expectation $\operatorname{E}[I(\mathbf{s}_0; u) \mid Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n)]$. Evaluating the exact conditional expectation given all continuous data requires knowing the full multivariate joint distribution of $(Z(\mathbf{s}_0), Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$.
> 
> Instead, Indicator Kriging provides the **best linear unbiased estimate** of $I(\mathbf{s}_0; u)$ based **only** on the binary indicators at that single threshold $u$. It discards the continuous numerical values of samples and ignores cross-correlations with indicators at other thresholds.

---

> [!NOTE]
> #### From the Lecture Hall: Kriging the Linear Probability Model
> Why does Indicator Kriging produce order-relation violations?
> 
> Because Indicator Kriging is literally the spatial equivalent of classical OLS on a Linear Probability Model! When we perform Ordinary Indicator Kriging:
> $$
> \widehat{F}(u) = \sum_{i=1}^n \lambda_i(u) I(\mathbf{s}_i; u)
> $$
> we are taking an unconstrained linear combination of binary numbers subject only to $\sum \lambda_i = 1$. Just as in ordinary kriging of continuous variables, kriging weights can be negative (due to data screening). If a sample with $I = 1$ receives a negative weight, or if samples with $I = 0$ receive weights summing to $> 1$, the predicted probability $\widehat{F}(u)$ can spill outside $[0, 1]$.
> 
> Furthermore, because each threshold $u_k$ is solved independently with its own semivariogram $\gamma_I(\mathbf{h}; u_k)$, the weights shift between thresholds. There is no mathematical constraint in independent kriging systems preventing $\widehat{F}(u_1) > \widehat{F}(u_2)$ when $u_1 < u_2$.

---

#### Median Indicator Kriging (MIK)

To circumvent the computational burden of modeling and solving $K$ separate indicator semivariograms, **Median Indicator Kriging (MIK)** assumes that the spatial correlation structure of indicators is invariant across thresholds:
$$
\frac{\gamma_I(\mathbf{h}; u_k)}{\sigma_I^2(u_k)} \approx \frac{\gamma_I(\mathbf{h}; u_{\text{med}})}{\sigma_I^2(u_{\text{med}})} = \rho_M(\mathbf{h}) \quad \text{for all } k = 1, \dots, K
$$
Under MIK:
1. The semivariogram matrix $\mathbf{\Gamma}_I$ is assembled and inverted **once** using the median correlogram.
2. The spatial weights $\boldsymbol{\lambda}$ are identical across all $K$ thresholds.
3. The CCDF at any target location $\mathbf{s}_0$ is evaluated by simple matrix-vector multiplication with the indicator data vector.

> [!CAUTION]
> #### Physical Validity of Median IK
> By assuming a constant correlogram across all thresholds, MIK implicitly adopts the **mosaic random field model** (§9.3). If the true physical phenomenon is diffusional or Gaussian-like (where extreme grades destructure into short ranges), MIK severely overstates the spatial continuity of high-grade ore shoots and contaminant spikes.

---

#### Decision Payoff: Extracting Risk-Qualified Metrics from Kriged CCDFs

Once an estimated CCDF $\widehat{F}(u_k \mid \mathbf{Z})$ is obtained across $K$ thresholds (and corrected for monotonicity via PAVA, §9.5), three operational metrics can be computed at every target location $\mathbf{s}_0$:

1. **Probability of Exceeding Statutory Limit $z_c$:**
   $$
   P(Z(\mathbf{s}_0) > z_c \mid \mathbf{Z}) = 1 - \widehat{F}(z_c \mid \mathbf{Z})
   $$
2. **E-Type Estimate (Conditional Mean):**
   The expectation of the estimated cumulative distribution:
   $$
   \widehat{Z}_{\text{E-type}}(\mathbf{s}_0) = \int_0^\infty (1 - \widehat{F}(u \mid \mathbf{Z}))\,du \approx \sum_{k=1}^{K+1} \bar{u}_k \left[ \widehat{F}(u_k \mid \mathbf{Z}) - \widehat{F}(u_{k-1} \mid \mathbf{Z}) \right]
   $$
   where $\bar{u}_k$ is the class midpoint in interval $(u_{k-1}, u_k]$ (or class mean from declustered global data), with $\widehat{F}(u_0) = 0$ and $\widehat{F}(u_{K+1}) = 1$.
3. **Risk-Averse Conditional Quantiles:**
   For any risk tolerance $\alpha \in (0, 1)$ (such as a 90th percentile upper confidence limit for remediation):
   $$
   z_\alpha(\mathbf{s}_0) = \widehat{F}^{-1}(\alpha \mid \mathbf{Z}) = \inf\{ u : \widehat{F}(u \mid \mathbf{Z}) \ge \alpha \}
   $$

---

### 9.5 Order Relation Violations and Monotonicity Corrections

#### The Mathematical Formulation of Order Relation Violations

Because each threshold $u_k$ is solved as an independent linear kriging system without joint monotonicity constraints, raw Indicator Kriging estimates frequently violate the fundamental axioms of probability:

##### Definition 9.5.1 (Order Relation Violations)
A vector of estimated cumulative probabilities $\widehat{\mathbf{F}} = (\widehat{F}_1, \dots, \widehat{F}_K)'$ across ordered cut-off thresholds $u_1 < u_2 < \dots < u_K$ suffers from **order-relation violations** if either:
1. **Bound Violations:** $\widehat{F}_k < 0$ or $\widehat{F}_k > 1$ for any $k \in \{1, \dots, K\}$.
2. **Monotonicity Violations:** $\widehat{F}_k > \widehat{F}_{k+1}$ for any adjacent pair $k \in \{1, \dots, K-1\}$.

A monotonicity reversal implies that the estimated probability mass in the interval $(u_k, u_{k+1}]$ is strictly negative:
$$
\widehat{P}\left( u_k < Z(\mathbf{s}_0) \le u_{k+1} \mid \mathbf{Z} \right) = \widehat{F}_{k+1} - \widehat{F}_k < 0
$$
which renders subsequent economic selection and quantile estimation mathematically impossible.

---

#### Isotonic Projection via PAVA

The mathematically principled solution to order-relation violations is **isotonic regression**: projecting the raw violating probability vector $\widehat{\mathbf{F}}$ onto the closed convex cone of monotonically non-decreasing vectors bounded in $[0, 1]$.

##### Theorem 9.5.1 (Isotonic Projection via PAVA)
Let $\widehat{\mathbf{F}} = (\widehat{F}_1, \dots, \widehat{F}_K)' \in \mathbb{R}^K$ be a vector of unconstrained indicator kriging estimates, and let $w_1, \dots, w_K > 0$ be strictly positive weights. The constrained weighted least-squares problem:
$$
\min_{\mathbf{p} \in \mathbb{R}^K} \sum_{k=1}^K w_k (p_k - \widehat{F}_k)^2 \quad \text{subject to } 0 \le p_1 \le p_2 \le \dots \le p_K \le 1
$$
is solved exactly by:
1. **Unconstrained Isotonic Pooling (PAVA):** Solving the unconstrained monotonicity problem $\min \sum w_k (p_k - \widehat{F}_k)^2$ subject only to $p_1 \le p_2 \le \dots \le p_K$ via the **Pool-Adjacent-Violators Algorithm**.
2. **Box Constraint Projection (Clamping):** Setting $p_k^* = \max(0, \min(1, p_k^{\text{PAVA}}))$.

##### Proof:
Consider first two points $y_1 > y_2$ with weights $w_1, w_2$. The unconstrained minimizer of $f(x_1, x_2) = w_1(x_1 - y_1)^2 + w_2(x_2 - y_2)^2$ subject to $x_1 \le x_2$ cannot occur in the interior $x_1 < x_2$ (where the unconstrained minimum $(y_1, y_2)$ violates the constraint). Hence the optimum lies on the boundary $x_1 = x_2 = \bar{x}$. Differentiating with respect to $\bar{x}$:
$$
\frac{d}{d\bar{x}}\left[ w_1(\bar{x} - y_1)^2 + w_2(\bar{x} - y_2)^2 \right] = 2 w_1(\bar{x} - y_1) + 2 w_2(\bar{x} - y_2) = 0 \implies \bar{x} = \frac{w_1 y_1 + w_2 y_2}{w_1 + w_2}
$$
The optimal value for any adjacent violating pair is exactly their weighted arithmetic mean. Repeating this argument across adjacent blocks establishes that whenever a violation occurs, the two blocks must be merged into a single pooled block with their weighted mean. Because the objective function is strictly convex and the constraint set is a convex cone, PAVA terminates at the unique global minimizer $\mathbf{p}^{\text{PAVA}}$.

Finally, the box constraint set $\mathcal{B} = [0, 1]^K$ is an orthogonal hypercube. Because $\mathbf{p}^{\text{PAVA}}$ is already monotone non-decreasing, the coordinate-wise clamping operation $T(p) = \max(0, \min(1, p))$ is a non-decreasing contraction: if $p_i \le p_j$, then $T(p_i) \le T(p_j)$. Therefore, clamping after pooling preserves monotonicity and strictly yields the unique orthogonal projection onto the intersection of the isotonic cone and $[0, 1]^K$. $\blacksquare$

> [!IMPORTANT]
> #### Algorithmic Order: Pool First, Then Clamp!
> A common coding mistake is clamping raw probabilities to $[0, 1]$ *before* running PAVA. Clamping first distorts the least-squares projection!
> 
> Consider raw vector $\mathbf{y} = (1.3, 0.9)$ with equal weights $w_1 = w_2 = 1.0$:
> - **Clamp First (Wrong):** Clamping to $[0, 1]$ yields $(1.0, 0.9)$. Since $1.0 > 0.9$, PAVA pools them to $(0.95, 0.95)$. The squared error distance to original $\mathbf{y}$ is $(1.3 - 0.95)^2 + (0.9 - 0.95)^2 = 0.35^2 + (-0.05)^2 = 0.1225 + 0.0025 = \mathbf{0.1250}$.
> - **Pool First, Then Clamp (Correct):** PAVA pools $1.3 > 0.9$ directly to $(1.3 + 0.9)/2 = 1.1$. Clamping to $[0, 1]$ gives $(1.0, 1.0)$. The squared error distance is $(1.3 - 1.0)^2 + (0.9 - 1.0)^2 = 0.3^2 + 0.1^2 = \mathbf{0.1000}$.
> 
> Pooling first achieves strictly lower error ($0.1000 < 0.1250$) and recovers the exact constrained projection.

---

> [!NOTE]
> #### From the Lecture Hall: Isotonic Regression and PAVA
> In classical non-parametric statistics, isotonic regression is used when a response is known to be monotonic with respect to an ordering (e.g., dose-response curves in clinical pharmacology). 
> 
> In spatial indicator kriging, we use inverse estimation variance as weights:
> $$
> w_k = \frac{1}{\sigma_{\text{IK}}^2(u_k)}
> $$
> This ensures that thresholds estimated with high certainty (small kriging variance) resist modification, while noisy thresholds (large kriging variance) are adjusted to conform to the monotonic profile.

---

#### Algorithmic Translation: PAVA Python Implementation

```python
def pava_isotonic_regression(
    raw_probs: list[float],
    weights: list[float] | None = None
) -> list[float]:
    """
    Applies the Pool-Adjacent-Violators Algorithm (PAVA) to correct
    order-relation violations in an estimated cumulative distribution.
    
    Correct mathematical sequence: Pool first (with backtracking),
    then clamp to [0, 1].
    
    Parameters:
        raw_probs: list of raw estimated probabilities [F_hat(u_1), ..., F_hat(u_K)]
        weights: list of positive weights w_k (defaults to equal weights 1.0)
        
    Returns:
        corrected_probs: monotonically non-decreasing probabilities in [0, 1]
    """
    K = len(raw_probs)
    if weights is None:
        weights = [1.0] * K
        
    # Step 1: Initialize blocks directly from raw unconstrained probabilities
    # Each block: [weighted_mean, total_weight, [indices]]
    blocks = [[float(raw_probs[k]), float(weights[k]), [k]] for k in range(K)]
    
    # Step 2: Iterative adjacent pooling with backtracking
    i = 0
    while i < len(blocks) - 1:
        # Check adjacent monotonicity violation
        if blocks[i][0] > blocks[i + 1][0]:
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
            
    # Step 3: Unpack block values into the unconstrained monotone vector
    pooled = [0.0] * K
    for blk in blocks:
        for idx in blk[2]:
            pooled[idx] = blk[0]
            
    # Step 4: Enforce box constraints [0, 1] after pooling
    return [max(0.0, min(1.0, p)) for p in pooled]
```

#### Step-by-Step Cascading Backtrack Walk-Through

To see how PAVA's backtracking step functions, consider a raw estimate vector with a **cascading violation**: $\widehat{\mathbf{F}} = (0.55, 0.45, 0.35, 0.85)$ with equal weights $w_k = 1.0$.

1. **Initial Blocks:**  
   $B_0 = [0.55, \text{wt}=1], B_1 = [0.45, \text{wt}=1], B_2 = [0.35, \text{wt}=1], B_3 = [0.85, \text{wt}=1]$.
2. **First Merge ($i = 0$):**  
   Compare $B_0 (0.55)$ and $B_1 (0.45)$. Violation ($0.55 > 0.45$).  
   Merge: $(0.55 + 0.45)/2 = 0.50$, total weight $= 2$.  
   Blocks: $B_0 = [0.50, \text{wt}=2], B_1 = [0.35, \text{wt}=1], B_2 = [0.85, \text{wt}=1]$.
3. **Second Merge ($i = 0$):**  
   Compare $B_0 (0.50)$ and $B_1 (0.35)$. Violation ($0.50 > 0.35$).  
   Merge: $(2 \times 0.50 + 1 \times 0.35) / (2 + 1) = 1.35 / 3 = 0.45$, total weight $= 3$.  
   Blocks: $B_0 = [0.45, \text{wt}=3], B_1 = [0.85, \text{wt}=1]$.
4. **Third Comparison ($i = 0$):**  
   Compare $B_0 (0.45)$ and $B_1 (0.85)$. Monotone ($0.45 \le 0.85$). Increment $i \to 1$. End of list.
5. **Final Projection:**  
   Unpack: $(0.45, 0.45, 0.45, 0.85)$. All values fall inside $[0, 1]$, so clamping leaves them unchanged.

Notice how the backtrack cascade merged the first three thresholds into a single consistent flat plateau at probability $0.45$!

---

### 9.6 Gaussian Anamorphosis via Hermite Polynomial Expansions

#### Classical Foundations: Orthogonal Polynomial Series

In classical functional approximation, an arbitrary square-integrable function $f(x)$ is represented by projection onto a complete orthogonal polynomial basis. In Gaussian spatial statistics, the natural inner product between two functions $f, g$ is the expectation under the standard Gaussian density $\phi(y) = \frac{1}{\sqrt{2\pi}} e^{-y^2/2}$:
$$
\langle f, g \rangle_\phi = \operatorname{E}[f(Y) g(Y)] = \int_{-\infty}^\infty f(y) g(y) \phi(y)\,dy
$$
The unique family of orthogonal polynomials associated with the Gaussian weight function $\phi(y)$ is the sequence of **probabilists' Hermite polynomials** $\{H_k(y)\}_{k=0}^\infty$.

##### Definition 9.6.1 (Hermite Polynomials via Rodrigues' Formula)
The probabilists' Hermite polynomial of degree $k$ is defined by Rodrigues' formula:
$$
H_k(y) = \frac{(-1)^k}{\phi(y)} \frac{d^k}{dy^k} \phi(y) = e^{y^2/2} \left( -\frac{d}{dy} \right)^k e^{-y^2/2}
$$
The first five Hermite polynomials are:
$$
H_0(y) = 1, \quad H_1(y) = y, \quad H_2(y) = y^2 - 1, \quad H_3(y) = y^3 - 3y, \quad H_4(y) = y^4 - 6y^2 + 3
$$

##### Theorem 9.6.1 (Orthogonality and Recurrence of Hermite Polynomials)
1. **Orthogonality:**
   $$
   \operatorname{E}[H_j(Y) H_k(Y)] = \int_{-\infty}^\infty H_j(y) H_k(y) \phi(y)\,dy = k! \, \delta_{jk}
   $$
   where $\delta_{jk}$ is the Kronecker delta.
2. **Three-Term Recurrence Relation:**
   $$
   H_{k+1}(y) = y H_k(y) - k H_{k-1}(y) \quad \text{for } k \ge 1
   $$
3. **Derivative Property:**
   $$
   \frac{d}{dy} H_k(y) = k H_{k-1}(y)
   $$

##### Proof:
For Part 1, assume without loss of generality that $j \le k$. Using Rodrigues' formula and integrating by parts $k$ times:
$$
\int_{-\infty}^\infty H_j(y) H_k(y) \phi(y)\,dy = (-1)^k \int_{-\infty}^\infty H_j(y) \frac{d^k}{dy^k}\phi(y)\,dy = \int_{-\infty}^\infty \left( \frac{d^k}{dy^k} H_j(y) \right) \phi(y)\,dy
$$
If $j < k$, the $k$-th derivative of a degree-$j$ polynomial is identically zero. If $j = k$, $\frac{d^k}{dy^k} H_k(y) = k!$, so the integral reduces to $k! \int \phi(y)\,dy = k!$.  
Parts 2 and 3 follow directly from differentiating the generating function $G(y, t) = \exp(t y - t^2/2) = \sum_{k=0}^\infty H_k(y) \frac{t^k}{k!}$. $\blacksquare$

---

#### Mehler's Formula: The Engine of Isofactorial Decoupling

The critical mathematical theorem that makes non-linear geostatistics computationally feasible is **Mehler's formula**, which expands the bivariate Gaussian density in terms of univariate Hermite polynomials.

##### Theorem 9.6.2 (Mehler's Formula)
Let $(Y_1, Y_2)'$ be a bivariate standard normal vector with correlation coefficient $\rho \in (-1, 1)$. The joint probability density function $\phi_2(y_1, y_2; \rho)$ satisfies:
$$
\phi_2(y_1, y_2; \rho) = \phi(y_1) \phi(y_2) \sum_{k=0}^\infty \frac{\rho^k}{k!} H_k(y_1) H_k(y_2)
$$
Equivalently, for any two Hermite polynomials of degrees $j$ and $k$:
$$
\operatorname{E}\left[ H_j(Y_1) H_k(Y_2) \right] = \begin{cases} k! \, \rho^k & \text{if } j = k \\ 0 & \text{if } j \ne k \end{cases}
$$

##### Proof:
Consider the bivariate normal generating function $\operatorname{E}[\exp(t_1 Y_1 + t_2 Y_2)] = \exp(\frac{1}{2}(t_1^2 + 2\rho t_1 t_2 + t_2^2))$.  
Factoring the joint MGF:
$$
\exp\left( \frac{1}{2}t_1^2 + \frac{1}{2}t_2^2 + \rho t_1 t_2 \right) = \exp\left( \frac{1}{2}t_1^2 \right) \exp\left( \frac{1}{2}t_2^2 \right) \sum_{k=0}^\infty \frac{\rho^k t_1^k t_2^k}{k!}
$$
Expanding the bivariate density in its Hermite series and equating coefficients of $t_1^j t_2^k / (j! k!)$ proves the identity. $\blacksquare$

---

#### Fitting Gaussian Anamorphosis

**Gaussian anamorphosis** defines a strictly increasing mathematical function $\Phi: \mathbb{R} \to \mathbb{R}$ mapping standard Gaussian scores $Y(\mathbf{s}) \sim \mathcal{N}(0, 1)$ to original physical values $Z(\mathbf{s})$:
$$
Z(\mathbf{s}) = \Phi(Y(\mathbf{s})) = F_Z^{-1}\left( \Phi_0(Y(\mathbf{s})) \right)
$$
where $\Phi_0$ is the standard normal CDF and $F_Z^{-1}$ is the empirical quantile function of sample data.

Expanding the transformation $\Phi(y)$ in the orthogonal Hermite basis:
$$
\Phi(y) = \sum_{k=0}^K c_k H_k(y)
$$
The expansion coefficients $c_k$ are obtained by projection:
$$
c_k = \frac{1}{k!} \langle \Phi, H_k \rangle_\phi = \frac{1}{k!} \operatorname{E}\left[ \Phi(Y) H_k(Y) \right] = \frac{1}{k!} \int_{-\infty}^\infty \Phi(y) H_k(y) \phi(y)\,dy
$$
Because the expansion is orthogonal, the total variance of the fitted transformation is:
$$
\operatorname{Var}(\Phi(Y)) = \sum_{k=1}^K k! \, c_k^2
$$

```python
def fit_hermite_gaussian_anamorphosis(
    sample_values: list[float],
    n_poly: int = 10,
    n_quad_nodes: int = 200
) -> tuple[list[float], callable, callable, float, float, bool]:
    """
    Fits Gaussian anamorphosis expansion Phi(y) = sum_{k=0}^{n_poly} c_k * H_k(y)
    using empirical normal score transform for the forward map and Hermite quadrature.
    
    Parameters:
        sample_values: empirical sample data [z_1, ..., z_n]
        n_poly: maximum Hermite polynomial order K
        n_quad_nodes: number of numerical quadrature nodes
        
    Returns:
        (coeffs, forward_transform, inverse_transform, var_captured, sample_var, is_monotone)
    """
    import math
    import bisect
    from scipy.special import ndtri
    
    n = len(sample_values)
    sorted_z = sorted(sample_values)
    
    # Forward transform: Empirical normal score transform (rank -> Phi_0^{-1})
    def forward_transform(z_val: float) -> float:
        idx = bisect.bisect_right(sorted_z, z_val)
        p = (max(1, min(n, idx)) - 0.5) / n
        return float(ndtri(p))
        
    # Empirical quantile function F_Z^{-1}(p)
    def quantile_z(p: float) -> float:
        p = max(0.5 / n, min(1.0 - 0.5 / n, p))
        idx = p * (n - 1)
        i = int(idx)
        frac = idx - i
        if i >= n - 1:
            return sorted_z[-1]
        return (1.0 - frac) * sorted_z[i] + frac * sorted_z[i + 1]
        
    def standard_normal_cdf(y: float) -> float:
        return 0.5 * (1.0 + math.erf(y / math.sqrt(2.0)))
        
    def evaluate_hermite(order: int, y: float) -> list[float]:
        H = [0.0] * (order + 1)
        H[0] = 1.0
        if order >= 1:
            H[1] = y
        for k in range(1, order):
            H[k + 1] = y * H[k] - k * H[k - 1]
        return H
        
    # Numerical quadrature over standard normal density [-5, 5]
    y_min, y_max = -5.0, 5.0
    dy = (y_max - y_min) / n_quad_nodes
    quad_y = [y_min + (k + 0.5) * dy for k in range(n_quad_nodes)]
    
    coeffs = [0.0] * (n_poly + 1)
    for y_val in quad_y:
        phi_wt = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * y_val * y_val) * dy
        p_val = standard_normal_cdf(y_val)
        z_val = quantile_z(p_val)
        H_vals = evaluate_hermite(n_poly, y_val)
        for k in range(n_poly + 1):
            coeffs[k] += z_val * H_vals[k] * phi_wt
            
    fact = 1.0
    for k in range(1, n_poly + 1):
        fact *= k
        coeffs[k] /= fact
        
    # Inverse transform: Back-transform series z = Phi(y)
    def inverse_transform(y: float) -> float:
        H_vals = evaluate_hermite(n_poly, y)
        return sum(coeffs[k] * H_vals[k] for k in range(n_poly + 1))
        
    # Diagnostics: variance captured and monotonicity
    var_captured = sum(math.factorial(k) * (coeffs[k] ** 2) for k in range(1, n_poly + 1))
    sample_mean = sum(sample_values) / n
    sample_var = sum((x - sample_mean) ** 2 for x in sample_values) / (n - 1)
    
    # Monotonicity check on [-3, 3]
    test_y = [ -3.0 + i * (6.0 / 100) for i in range(101) ]
    test_z = [ inverse_transform(y) for y in test_y ]
    is_monotone = all(test_z[i] <= test_z[i+1] for i in range(len(test_z) - 1))
    
    return coeffs, forward_transform, inverse_transform, var_captured, sample_var, is_monotone
```

---

### 9.7 Disjunctive Kriging (DK) and Isofactorial Models

#### The Isofactorial Representation

Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a stationary random field whose normal scores $Y(\mathbf{s})$ form a bivariate standard Gaussian system. Suppose we wish to predict an arbitrary non-linear functional $\psi(Z(\mathbf{s}_0))$ at an unsampled location $\mathbf{s}_0$.

##### Definition 9.7.1 (Bivariate Isofactorial Models)
A stationary bivariate random field $(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h}))'$ is **isofactorial** with respect to orthogonal polynomials $\{\chi_k\}$ if:
$$
\operatorname{E}\left[ \chi_j(Y(\mathbf{s})) \chi_k(Y(\mathbf{s}+\mathbf{h})) \right] = \delta_{jk} \rho_k(\mathbf{h})
$$
where $\rho_k(\mathbf{h})$ is the spatial autocorrelation of the $k$-th factor.

Under Gaussian anamorphosis, Mehler's formula (Theorem 9.6.2) establishes that the Hermite polynomials $H_k(y)$ are the exact isofactorial factors, with factor correlation:
$$
\rho_k(\mathbf{h}) = [\rho(\mathbf{h})]^k
$$

---

#### Decoupling of Disjunctive Kriging

##### Definition 9.7.2 (Disjunctive Kriging Estimator)
The **Disjunctive Kriging (DK)** estimator of $\psi(Z(\mathbf{s}_0))$ is the best additive non-linear predictor:
$$
\widehat{\psi}_{\text{DK}}(\mathbf{s}_0) = \sum_{i=1}^n f_i(Z(\mathbf{s}_i)) = \sum_{i=1}^n g_i(Y(\mathbf{s}_i))
$$
minimizing the expected mean squared error $\operatorname{E}[(\psi(Z(\mathbf{s}_0)) - \widehat{\psi}_{\text{DK}}(\mathbf{s}_0))^2]$.

##### Theorem 9.7.1 (Isofactorial Decoupling of Disjunctive Kriging)
Let the target functional have Hermite expansion $\psi(Z) = \sum_{k=0}^K c_k H_k(Y)$. Under bivariate Gaussian anamorphosis, the Disjunctive Kriging system decouples completely degree by degree:
$$
\widehat{\psi}_{\text{DK}}(\mathbf{s}_0) = \sum_{k=0}^K c_k \sum_{i=1}^n \lambda_i^{(k)} H_k(Y(\mathbf{s}_i))
$$
For each polynomial degree $k \in \{0, 1, \dots, K\}$, the spatial weights $\boldsymbol{\lambda}^{(k)} = (\lambda_1^{(k)}, \dots, \lambda_n^{(k)})'$ are obtained by solving the independent $n \times n$ linear system:
$$
\mathbf{R}^{(k)} \boldsymbol{\lambda}^{(k)} = \mathbf{r}_0^{(k)}
$$
where:
$$
R_{ij}^{(k)} = [\rho(\mathbf{s}_i - \mathbf{s}_j)]^k, \quad r_{0, i}^{(k)} = [\rho(\mathbf{s}_i - \mathbf{s}_0)]^k
$$
The corresponding estimation error variance is:
$$
\sigma_{\text{DK}}^2(\mathbf{s}_0) = \sum_{k=1}^K k! \, c_k^2 \left( 1 - \sum_{i=1}^n \lambda_i^{(k)} [\rho(\mathbf{s}_i - \mathbf{s}_0)]^k \right)
$$

##### Proof:
By the projection theorem on the Hilbert space of random variables, the optimal additive predictor must satisfy the orthogonality condition:
$$
\operatorname{E}\left[ \left( \psi(Z(\mathbf{s}_0)) - \sum_{j=1}^n g_j(Y(\mathbf{s}_j)) \right) h(Y(\mathbf{s}_i)) \right] = 0
$$
for every sample location $i \in \{1, \dots, n\}$ and every function $h$. Expanding both the target functional and the candidate functions $g_j$ in the Hermite basis:
$$
\psi(Z(\mathbf{s}_0)) = \sum_{k=0}^K c_k H_k(Y(\mathbf{s}_0)), \quad g_j(Y(\mathbf{s}_j)) = \sum_{k=0}^K \lambda_j^{(k)} c_k H_k(Y(\mathbf{s}_j))
$$
Testing with $h(Y(\mathbf{s}_i)) = H_m(Y(\mathbf{s}_i))$:
$$
\sum_{k=0}^K c_k \operatorname{E}\left[ H_k(Y(\mathbf{s}_0)) H_m(Y(\mathbf{s}_i)) \right] = \sum_{j=1}^n \sum_{k=0}^K \lambda_j^{(k)} c_k \operatorname{E}\left[ H_k(Y(\mathbf{s}_j)) H_m(Y(\mathbf{s}_i)) \right]
$$
By Mehler's formula (Theorem 9.6.2), cross-degree terms with $k \ne m$ vanish identically:
$$
\operatorname{E}[H_k(Y) H_m(Y)] = m! \, [\rho(\mathbf{h})]^m \delta_{km}
$$
Dividing both sides by $m! \, c_m$:
$$
[\rho(\mathbf{s}_i - \mathbf{s}_0)]^m = \sum_{j=1}^n \lambda_j^{(m)} [\rho(\mathbf{s}_i - \mathbf{s}_j)]^m
$$
which is precisely the independent $n \times n$ linear kriging system $\mathbf{R}^{(m)} \boldsymbol{\lambda}^{(m)} = \mathbf{r}_0^{(m)}$. $\blacksquare$

---

> [!NOTE]
> #### From the Lecture Hall: Orthogonal Regression and Isofactorial Decoupling
> Why is Disjunctive Kriging so computationally elegant?
> 
> In general, solving for an optimal non-linear predictor $\sum_{i=1}^n f_i(Z_i)$ involves solving a coupled system of $n \times K$ non-linear equations where every sample interacts with every polynomial power.
> 
> Mehler's formula diagonalizes the entire system! Because Hermite polynomials of different degrees are mutually uncorrelated across space ($\operatorname{Cov}(H_j(Y_1), H_k(Y_2)) = 0$ for $j \ne k$), the massive $n K \times n K$ system collapses into $K$ completely independent $n \times n$ linear systems. Degree 1 is standard Simple Kriging of normal scores; degree 2 kriges squared anomalies; and degree $k$ kriges higher-order non-linear features.

---

#### Why Use Disjunctive Kriging Instead of Exact Conditional Expectation?

If a spatial field is strictly multivariate Gaussian, the exact conditional expectation $\operatorname{E}[\psi(Z_0) \mid \mathbf{Z}]$ is known in closed form (as in lognormal kriging, §9.2). Why, then, was Disjunctive Kriging developed?

1. **Weakened Distributional Assumptions:** Disjunctive Kriging requires **only bivariate Gaussianity** of normal score pairs $(Y(\mathbf{s}_i), Y(\mathbf{s}_j))'$, rather than full $n$-dimensional multivariate Gaussianity. In practice, spatial data frequently pass bivariate symmetry checks while failing high-dimensional joint normality.
2. **Behavior of Higher-Order Weights:** As degree $k$ increases, the off-diagonal entries $R_{ij}^{(k)} = [\rho(\mathbf{s}_i - \mathbf{s}_j)]^k$ decay rapidly toward zero (since $|\rho| < 1$). Consequently, for large $k$, the matrix $\mathbf{R}^{(k)}$ approaches the identity matrix $\mathbf{I}$, and weights $\lambda_i^{(k)}$ concentrate almost entirely on the single closest sample datum. Higher-order non-linear terms automatically act as local nearest-neighbor corrections!

---

#### Comprehensive Synthesis: The Running Benchmark Example

To ground every method in the chapter, consider a 1D spatial transect with three sample locations conditioning an estimate at target location $s_0 = 1.0$:
- **Sample Coordinates:** $s_1 = 0.0, \, s_2 = 1.5, \, s_3 = 4.0$. Target $s_0 = 1.0$.
- **True Underlying Gaussian Field:** $Y(s) \sim \mathcal{GP}(\mu_Y = 1.0, \, \sigma_Y^2 = 0.8)$ with exponential covariance $C_Y(h) = 0.8 e^{-|h|/2.0}$.
- **Observed Log-Values:** $y_1 = 1.2, \, y_2 = 0.7, \, y_3 = 1.4 \implies$ Physical values $z_1 = e^{1.2} = 3.32, \, z_2 = e^{0.7} = 2.01, \, z_3 = e^{1.4} = 4.06\text{ g/t}$.

We evaluate the conditional distribution and exceedance probability at cut-off threshold $z_c = 3.0\text{ g/t}$:

| Method | Target Metric at $s_0 = 1.0$ | Numerical Result | Comparison to True Benchmark |
| :--- | :--- | :--- | :--- |
| **True Conditional Expectation** | Expected Grade $m(s_0)$ | **$2.717\text{ g/t}$** | Exact theoretical reference |
| | Exceedance Prob $P(Z > 3.0)$ | **$0.3267$** ($32.7\%$) | Exact theoretical reference |
| **Naive Log-Kriging Back-Transform** | Naive Median $\exp(\widehat{Y}_{\text{SK}})$ | **$2.390\text{ g/t}$** | Underestimates conditional mean by $-12.0\%$ |
| **Simple Lognormal Kriging** | Unbiased Mean $\exp(\widehat{Y} + \frac{1}{2}\sigma^2)$ | **$2.717\text{ g/t}$** | Exactly matches conditional expectation ($0.0\%$ error) |
| **Ordinary Lognormal Kriging** | Unbiased Mean $\exp(\widehat{Y} + \frac{1}{2}\sigma^2 + \mu_L)$ | **$2.678\text{ g/t}$** | Globally unbiased; within $-1.4\%$ of exact local mean |
| **Indicator Kriging (Raw)** | Exceedance Prob $1 - \widehat{F}(3.0)$ | **$0.4180$** ($41.8\%$) | Suffers order violation ($\widehat{F}(2.5) = 0.648 > \widehat{F}(3.0) = 0.582$) |
| **Indicator Kriging + PAVA** | Exceedance Prob $1 - \widehat{F}_{\text{PAVA}}(3.0)$ | **$0.3850$** ($38.5\%$) | Monotone projection resolves violation; close to truth |
| **Disjunctive Kriging (DK)** | Exceedance Prob via Hermite series | **$0.3340$** ($33.4\%$) | Excellent approximation; tracks exact truth within $+0.7\%$ |

---

#### Master Method Comparison Table

| Metric / Dimension | Lognormal Kriging | Indicator Kriging (IK) | Median Indicator Kriging (MIK) | Disjunctive Kriging (DK) |
| :--- | :--- | :--- | :--- | :--- |
| **Required Hypothesis** | Multivariate lognormal | Bivariate indicator stationarity per cut-off | Mosaic random field (identical correlation shapes) | Bivariate Gaussianity of normal scores |
| **Output Type** | Point conditional mean & variance | Discrete CCDF $\widehat{F}(u_k)$ | Discrete CCDF $\widehat{F}(u_k)$ | Additive non-linear functional estimate |
| **Handles Zero Values?** | No (requires arbitrary positive constant $\epsilon$) | Yes (natural handling of zeros/detection limits) | Yes | Yes (spikes mapped via flat steps in anamorphosis) |
| **Order Violations?** | None (continuous exponential map) | Frequent (requires PAVA post-processing) | Rare (identical weights across thresholds) | None (in continuous isofactorial representation) |
| **Computational Cost** | Single $n \times n$ kriging solve | $K$ independent $n \times n$ kriging solves | Single $n \times n$ solve + $K$ matrix-vector products | $K$ independent $n \times n$ kriging solves |
| **Primary Failure Mode** | High-variance explosion; outlier sensitivity | Discards continuous data; LPM boundary violations | Over-smoothes extreme channels/destructuration | Truncation rippling; tail extrapolation errors |

---

### 9.8 Analytical Exercises and Complete Step-by-Step Solutions

#### Exercise 9.1: Exact Derivation of Lognormal Back-Transformation
Consider a stationary lognormal random field $Z(\mathbf{s}) = \exp(Y(\mathbf{s}))$, where $Y(\mathbf{s})$ is a Gaussian random field with known stationary mean $\mu_Y = 1.0$ and known covariance function:
$$
C_Y(\mathbf{h}) = 0.6 \exp\left( -\frac{\|\mathbf{h}\|}{10} \right)
$$
Two sample observations are recorded:
$$
Z(\mathbf{s}_1) = 3.0 \implies Y(\mathbf{s}_1) = \ln(3.0) \approx 1.09861
$$
$$
Z(\mathbf{s}_2) = 6.0 \implies Y(\mathbf{s}_2) = \ln(6.0) \approx 1.79176
$$
The spatial separation between the two samples is $\|\mathbf{s}_1 - \mathbf{s}_2\| = 10\text{ m}$. We wish to estimate the attribute at an unsampled location $\mathbf{s}_0$, situated $\|\mathbf{s}_1 - \mathbf{s}_0\| = 4\text{ m}$ from $\mathbf{s}_1$ and $\|\mathbf{s}_2 - \mathbf{s}_0\| = 8\text{ m}$ from $\mathbf{s}_2$.

**(a)** Assemble the Simple Kriging covariance matrix $\mathbf{V}_Y$ and vector $\mathbf{c}_Y$ on the log scale, and solve for the Simple Kriging weights $\boldsymbol{\lambda} = (\lambda_1, \lambda_2)'$.  
**(b)** Compute the Simple Kriging estimate $\widehat{Y}_{\text{SK}}(\mathbf{s}_0)$ and the Simple Kriging estimation variance $\sigma_{\text{SK}}^2(\mathbf{s}_0)$.  
**(c)** Calculate the naive back-transformed estimate $\widehat{Z}_{\text{naive}}(\mathbf{s}_0) = \exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))$.  
**(d)** Calculate the strictly unbiased MMSE conditional expectation estimate $\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}]$, and evaluate the percentage by which the naive estimate underestimates the true conditional mean.  
**(e)** Construct the exact asymmetric 95% conditional confidence interval for $Z(\mathbf{s}_0)$.

##### Solution:
**(a)** Covariance values on the log scale:
$$
C_Y(\mathbf{0}) = 0.6
$$
$$
C_Y(\mathbf{s}_1 - \mathbf{s}_2) = 0.6 \exp(-10 / 10) = 0.6 e^{-1} \approx 0.6 \times 0.367879 = 0.220728
$$
$$
C_Y(\mathbf{s}_1 - \mathbf{s}_0) = 0.6 \exp(-4 / 10) = 0.6 e^{-0.4} \approx 0.6 \times 0.670320 = 0.402192
$$
$$
C_Y(\mathbf{s}_2 - \mathbf{s}_0) = 0.6 \exp(-8 / 10) = 0.6 e^{-0.8} \approx 0.6 \times 0.449329 = 0.269597
$$

Covariance system:
$$
\mathbf{V}_Y = \begin{pmatrix} 0.600000 & 0.220728 \\ 0.220728 & 0.600000 \end{pmatrix}, \quad \mathbf{c}_Y = \begin{pmatrix} 0.402192 \\ 0.269597 \end{pmatrix}
$$
The determinant of $\mathbf{V}_Y$ is:
$$
\det(\mathbf{V}_Y) = 0.6^2 - 0.220728^2 = 0.36 - 0.048721 = 0.311279
$$
Inverting $\mathbf{V}_Y$:
$$
\mathbf{V}_Y^{-1} = \frac{1}{0.311279} \begin{pmatrix} 0.600000 & -0.220728 \\ -0.220728 & 0.600000 \end{pmatrix} \approx \begin{pmatrix} 1.927531 & -0.709099 \\ -0.709099 & 1.927531 \end{pmatrix}
$$
Solving for Simple Kriging weights:
$$
\lambda_1 = 1.927531(0.402192) - 0.709099(0.269597) = 0.775237 - 0.191171 = 0.584066
$$
$$
\lambda_2 = -0.709099(0.402192) + 1.927531(0.269597) = -0.285194 + 0.519656 = 0.234462
$$

**(b)** The Simple Kriging estimate on the log scale is:
$$
\widehat{Y}_{\text{SK}}(\mathbf{s}_0) = \mu_Y + \lambda_1(Y(\mathbf{s}_1) - \mu_Y) + \lambda_2(Y(\mathbf{s}_2) - \mu_Y)
$$
$$
\widehat{Y}_{\text{SK}}(\mathbf{s}_0) = 1.0 + 0.584066(1.09861 - 1.0) + 0.234462(1.79176 - 1.0)
$$
$$
\widehat{Y}_{\text{SK}}(\mathbf{s}_0) = 1.0 + 0.584066(0.09861) + 0.234462(0.79176) = 1.0 + 0.057595 + 0.185637 \approx 1.24323
$$

Simple Kriging estimation variance:
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = C_Y(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{c}_Y
$$
$$
\boldsymbol{\lambda}' \mathbf{c}_Y = 0.584066(0.402192) + 0.234462(0.269597) = 0.234907 + 0.063210 = 0.298117
$$
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = 0.600000 - 0.298117 = 0.301883
$$
Standard deviation: $\sigma_{\text{SK}}(\mathbf{s}_0) = \sqrt{0.301883} \approx 0.54944$.

**(c)** Naive back-transformation (conditional median):
$$
\widehat{Z}_{\text{naive}}(\mathbf{s}_0) = \exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0)) = \exp(1.24323) \approx 3.46678
$$

**(d)** Unbiased MMSE conditional expectation:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \exp\left( \widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0) \right) = \exp(1.24323 + 0.5 \times 0.301883) = \exp(1.24323 + 0.150942) = \exp(1.39417) \approx 4.03164
$$

Percentage underestimation of the naive estimate relative to the unbiased mean:
$$
\text{Underestimation} = \frac{4.03164 - 3.46678}{4.03164} \times 100\% = \frac{0.56486}{4.03164} \times 100\% \approx 14.01\%
$$
Equivalently, the naive estimate is $\exp(-\frac{1}{2}\sigma_{\text{SK}}^2) = \exp(-0.15094) = 0.8599$ (14.01% below the mean).

**(e)** A 95% confidence interval on the Gaussian log scale is:
$$
Y_{0.025} = \widehat{Y}_{\text{SK}} - 1.96 \, \sigma_{\text{SK}} = 1.24323 - 1.96(0.54944) = 1.24323 - 1.07690 = 0.16633
$$
$$
Y_{0.975} = \widehat{Y}_{\text{SK}} + 1.96 \, \sigma_{\text{SK}} = 1.24323 + 1.96(0.54944) = 1.24323 + 1.07690 = 2.32013
$$
Exponentiating back to original physical units yields the exact 95% asymmetric confidence interval:
$$
\left[ \exp(0.16633), \, \exp(2.32013) \right] = \left[ 1.181, \, 10.177 \right]
$$
Notice the strong asymmetry around the median ($3.467$): the upper limit extends $6.71$ units above the median, while the lower limit extends only $2.29$ units below it!

---

#### Exercise 9.2: Isotonic Projection of Violating Indicator Kriging Estimates via PAVA
At an unsampled location $\mathbf{s}_0$, Ordinary Indicator Kriging is performed independently across five cut-off thresholds $u_1 < u_2 < u_3 < u_4 < u_5$. The resulting raw probability estimates are:
$$
\widehat{\mathbf{F}} = \left( \widehat{F}(u_1), \, \widehat{F}(u_2), \, \widehat{F}(u_3), \, \widehat{F}(u_4), \, \widehat{F}(u_5) \right)' = (0.22, \, 0.48, \, 0.38, \, 0.71, \, 0.95)'
$$
Assume equal weights $w_1 = \dots = w_5 = 1.0$.

**(a)** Identify the order-relation violation and state its physical impossibility.  
**(b)** Trace the step-by-step execution of the Pool-Adjacent-Violators Algorithm (PAVA).  
**(c)** Show that the resulting isotonic vector strictly minimizes the quadratic objective function.

##### Solution:
**(a)** Examining the sequence:
- $\widehat{F}(u_1) = 0.22 \le \widehat{F}(u_2) = 0.48$ (Valid)
- $\widehat{F}(u_2) = 0.48 > \widehat{F}(u_3) = 0.38$ (**Order-Relation Violation!**)
- $\widehat{F}(u_3) = 0.38 \le \widehat{F}(u_4) = 0.71 \le \widehat{F}(u_5) = 0.95$ (Valid)

The violation implies negative probability mass in interval $(u_2, u_3]$:
$$
\widehat{P}(u_2 < Z(\mathbf{s}_0) \le u_3) = \widehat{F}(u_3) - \widehat{F}(u_2) = 0.38 - 0.48 = -0.10 \quad (-10\%)
$$
Negative physical probabilities are mathematically invalid.

**(b)** PAVA Execution Trace:
- **Initialization:** Create five singleton blocks:  
  $B_1 = [0.22, w=1, \{1\}], \, B_2 = [0.48, w=1, \{2\}], \, B_3 = [0.38, w=1, \{3\}], \, B_4 = [0.71, w=1, \{4\}], \, B_5 = [0.95, w=1, \{5\}]$.
- **Step 1 ($i = 1$):** Compare $B_1 (0.22)$ and $B_2 (0.48)$. $0.22 \le 0.48$. Valid. Advance $i \to 2$.
- **Step 2 ($i = 2$):** Compare $B_2 (0.48)$ and $B_3 (0.38)$. Violation ($0.48 > 0.38$).  
  Pool $B_2$ and $B_3$:
  $$
  v_{\text{merged}} = \frac{1(0.48) + 1(0.38)}{1 + 1} = \frac{0.86}{2} = 0.43, \quad w_{\text{merged}} = 2
  $$
  New blocks: $B_1 = [0.22, w=1], \, B_2^* = [0.43, w=2, \{2, 3\}], \, B_4 = [0.71, w=1], \, B_5 = [0.95, w=1]$.
- **Backtrack ($i = 1$):** Compare $B_1 (0.22)$ with merged block $B_2^* (0.43)$.  
  Since $0.22 \le 0.43$, monotonicity is maintained. Advance $i \to 2$.
- **Step 3 ($i = 2$):** Compare $B_2^* (0.43)$ and $B_4 (0.71)$. $0.43 \le 0.71$. Valid. Advance $i \to 3$.
- **Step 4 ($i = 3$):** Compare $B_4 (0.71)$ and $B_5 (0.95)$. $0.71 \le 0.95$. Valid. Termination.

All block values lie within $[0, 1]$, so clamping leaves the values unchanged:
$$
\mathbf{p}^* = (0.22, \, 0.43, \, 0.43, \, 0.71, \, 0.95)'
$$

**(c)** Verification of optimality:  
The unconstrained sum of squares for any common value $p_{23}$ assigned to indices 2 and 3 is:
$$
Q(p_{23}) = (p_{23} - 0.48)^2 + (p_{23} - 0.38)^2 = 2 p_{23}^2 - 2(0.86) p_{23} + (0.48^2 + 0.38^2)
$$
Differentiating with respect to $p_{23}$:
$$
Q'(p_{23}) = 4 p_{23} - 1.72 = 0 \implies p_{23} = \frac{1.72}{4} = 0.43
$$
Because $0.22 \le 0.43 \le 0.71$, the boundary constraints with adjacent elements are strictly satisfied. Hence $\mathbf{p}^*$ is the unique global orthogonal projection.

---

#### Exercise 9.3: Hermite Expansion Coefficients of an Indicator Function
Let $Y \sim \mathcal{N}(0, 1)$ be a standard Gaussian variable. We wish to expand the non-linear threshold indicator function:
$$
I(Y; y_u) = \mathbb{I}\{Y \le y_u\}
$$
in the probabilists' Hermite polynomial basis $\{H_k(y)\}_{k=0}^\infty$:
$$
\mathbb{I}\{Y \le y_u\} = \sum_{k=0}^\infty c_k H_k(Y)
$$

**(a)** Derive the general analytical formula for the expansion coefficients $c_k = \frac{1}{k!} \operatorname{E}[\mathbb{I}\{Y \le y_u\} H_k(Y)]$ for all $k \ge 0$.  
**(b)** Write out the explicit series expansion through degree $k = 4$.  
**(c)** Evaluate the coefficients numerically for the median threshold $y_u = 0$, and comment on the connection between Disjunctive Kriging and Indicator Kriging.

##### Solution:
**(a)** For degree $k = 0$:
$$
c_0 = \frac{1}{0!} \operatorname{E}\left[ \mathbb{I}\{Y \le y_u\} H_0(Y) \right] = \operatorname{E}\left[ \mathbb{I}\{Y \le y_u\} \cdot 1 \right] = P(Y \le y_u) = \Phi_0(y_u)
$$
For degree $k \ge 1$, recall Rodrigues' formula:
$$
H_k(y) \phi(y) = (-1)^k \frac{d^k}{dy^k} \phi(y) = -\frac{d}{dy}\left( (-1)^{k-1} \frac{d^{k-1}}{dy^{k-1}}\phi(y) \right) = -\frac{d}{dy}\left( H_{k-1}(y) \phi(y) \right)
$$
Substituting into the definition of $c_k$:
$$
c_k = \frac{1}{k!} \int_{-\infty}^\infty \mathbb{I}\{y \le y_u\} H_k(y) \phi(y)\,dy = \frac{1}{k!} \int_{-\infty}^{y_u} H_k(y) \phi(y)\,dy
$$
Using the derivative relation:
$$
c_k = -\frac{1}{k!} \int_{-\infty}^{y_u} \frac{d}{dy}\left[ H_{k-1}(y) \phi(y) \right] dy = -\frac{1}{k!} \left[ H_{k-1}(y) \phi(y) \right]_{-\infty}^{y_u}
$$
Since $\lim_{y \to -\infty} H_{k-1}(y) \phi(y) = 0$, evaluating at the upper limit yields:
$$
c_k = -\frac{1}{k!} H_{k-1}(y_u) \phi(y_u) \quad \text{for all } k \ge 1
$$

**(b)** Using the explicit Hermite polynomials:
- $H_0(y) = 1 \implies c_1 = -\phi(y_u)$
- $H_1(y) = y \implies c_2 = -\frac{1}{2} y_u \phi(y_u)$
- $H_2(y) = y^2 - 1 \implies c_3 = -\frac{1}{6} (y_u^2 - 1) \phi(y_u)$
- $H_3(y) = y^3 - 3y \implies c_4 = -\frac{1}{24} (y_u^3 - 3y_u) \phi(y_u)$

The Hermite series expansion through order 4 is:
$$
\mathbb{I}\{Y \le y_u\} = \Phi_0(y_u) - \phi(y_u) \left[ H_1(Y) + \frac{y_u}{2} H_2(Y) + \frac{y_u^2 - 1}{6} H_3(Y) + \frac{y_u^3 - 3y_u}{24} H_4(Y) + \dots \right]
$$

**(c)** For the median cut-off $y_u = 0$:
$$
\Phi_0(0) = 0.5, \quad \phi(0) = \frac{1}{\sqrt{2\pi}} \approx 0.398942
$$
Evaluating the polynomial terms at $y_u = 0$:
- $c_0 = 0.5$
- $c_1 = -\phi(0) \approx -0.398942$
- $c_2 = 0$ (all even $k$ vanish because $H_{2m-1}(0) = 0$)
- $c_3 = -\frac{1}{6}(0 - 1)\phi(0) = \frac{1}{6}\phi(0) \approx +0.066490$
- $c_4 = 0$
- $c_5 = -\frac{1}{120} H_4(0) \phi(0) = -\frac{3}{120}\phi(0) = -\frac{1}{40}\phi(0) \approx -0.009974$

The median indicator series reduces to:
$$
\mathbb{I}\{Y \le 0\} = \frac{1}{2} - \phi(0) \left[ H_1(Y) - \frac{1}{6} H_3(Y) + \frac{1}{40} H_5(Y) - \frac{1}{336} H_7(Y) + \dots \right]
$$

**Connection between DK and IK:**  
When Disjunctive Kriging is applied to an indicator function $\psi(Z) = \mathbb{I}\{Z \le u\}$, it computes the kriged indicator by estimating each Hermite polynomial $H_k(Y)$ in the series above. Because Disjunctive Kriging is built on a coherent bivariate Gaussian distribution, the estimated CCDF across thresholds is derived from a valid joint probability model, naturally mitigating the erratic order-relation reversals that plague standard independent Indicator Kriging!

---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:  
👉 [Chapter 9: Non-Gaussian and Nonlinear Spatial Estimation — Companion Exercises and Complete Solutions](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch09_exercises.md)
