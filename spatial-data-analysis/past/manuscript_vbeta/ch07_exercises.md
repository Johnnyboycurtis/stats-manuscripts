# Chapter 7 Companion Exercises: Multivariate Spatial Prediction and Cokriging

This study guide and exercise workbook accompanies [Chapter 7: Multivariate Geostatistics and Cokriging Theory](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch07_cokriging.md). Synthesizing the pedagogical clarity of Maria L. Rizzo (*Statistical Computing with R*) and the semantic linear-model discipline of Oliver Schabenberger and Carol A. Gotway (*Statistical Methods for Spatial Data Analysis*), these exercises are organized into five comprehensive tiers:

1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 7.1–7.4)
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 7.5–7.8)
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 7.9–7.11)
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 7.12–7.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 7.14–7.16)

Every exercise includes an unabridged, step-by-step mathematical proof or computational solution.

---

## Tier 0: Classical-to-Spatial Warm-Ups

### Exercise 7.1: Classical Bivariate Regression with Partially Missing Data: Precursor to Heterotopic Cokriging
In classical multivariate analysis, when measuring two correlated continuous random variables $(Y_1, Y_2)$, experimental budgets or physical collection constraints frequently produce an unbalanced observation scheme. An auxiliary or secondary attribute $Y_1$ (e.g., automated sensor telemetry, satellite spectral reflectance, or optical density) is inexpensive to collect and is measured across $n$ independent experimental units. Conversely, the primary attribute of interest $Y_2$ (e.g., wet chemical assay, destructive laboratory extraction, or core physical porosity) is expensive, labor-intensive, or destructive, and is consequently measured only on an initial subset of $m < n$ units. 

Thus, the sample consists of $m$ complete, collocated pairs $(y_{1,1}, y_{2,1}), \dots, (y_{1,m}, y_{2,m})$ and $n - m$ incomplete units where only the secondary variable is observed: $y_{1,m+1}, \dots, y_{1,n}$.

Assume that the population pairs $(Y_{1,i}, Y_{2,i})'$ are independent and identically distributed according to a bivariate normal distribution:
$$
\begin{pmatrix} Y_{1,i} \\ Y_{2,i} \end{pmatrix} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}_2\left(\boldsymbol{\mu} = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{pmatrix}\right), \quad i = 1, \dots, n
$$
where $|\rho| < 1$, $\sigma_1^2 > 0$, and $\sigma_2^2 > 0$.

**(a)** State the joint density function $f(y_1, y_2; \boldsymbol{\theta})$ for a complete pair, where $\boldsymbol{\theta} = (\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)'$, and specify the joint likelihood function $L(\boldsymbol{\theta})$ for the entire sample of $n$ units under the missing-at-random (MAR) assumption.

**(b)** Using the conditional probability identity $f(y_{1,i}, y_{2,i}) = f(y_{1,i}) f(y_{2,i} \mid y_{1,i})$, factorize the likelihood function into a marginal likelihood for $Y_1$ and a conditional likelihood for $Y_2 \mid Y_1$. Show how the $n-m$ observations of $Y_1$ alone contribute to the precision of estimating the secondary mean $\mu_1$ and variance $\sigma_1^2$, and derive the Maximum Likelihood Estimator (MLE) for the primary mean $\mu_2$:
$$
\widehat{\mu}_2 = \bar{y}_{2,m} + \widehat{\beta}\left(\bar{y}_{1,n} - \bar{y}_{1,m}\right)
$$
where $\bar{y}_{1,n} = \frac{1}{n}\sum_{i=1}^n y_{1,i}$, $\bar{y}_{1,m} = \frac{1}{m}\sum_{i=1}^m y_{1,i}$, $\bar{y}_{2,m} = \frac{1}{m}\sum_{i=1}^m y_{2,i}$, and $\widehat{\beta}$ is the sample regression slope of $Y_2$ on $Y_1$ computed from the $m$ complete cases.

**(c)** Derive the conditional minimum mean squared error (MMSE) predictor of $Y_{2,i}$ for each of the $n-m$ units ($i = m+1, \dots, n$) where only $Y_{1,i}$ is observed:
$$
\widehat{Y}_{2,i} = \widehat{\mu}_2 + \widehat{\rho}\frac{\widehat{\sigma}_2}{\widehat{\sigma}_1}(Y_{1,i} - \widehat{\mu}_1)
$$
Derive the conditional prediction error variance $\operatorname{Var}(Y_{2,i} - \widehat{Y}_{2,i} \mid Y_{1,i})$ (ignoring parameter estimation error), and compare it to the unconditional variance of predicting $Y_{2,i}$ without knowledge of $Y_{1,i}$.

**(d)** Connect this classical missing-data formulation to spatial heterotopic cokriging: explain why densely sampled secondary attributes (e.g., digital elevation models, airborne gamma-ray spectrometry, or satellite NDVI) can significantly improve the spatial prediction of an undersampled primary attribute (e.g., topsoil organic carbon or heavy-metal concentrations), and identify the spatial analogues to $\rho$, $\sigma_2/\sigma_1$, and the variance reduction factor $1 - \rho^2$.

##### Solution to Exercise 7.1
**(a)** The joint probability density function of a bivariate normal random vector $(Y_1, Y_2)' \sim \mathcal{N}_2(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ is given by:
$$
f(y_1, y_2; \boldsymbol{\theta}) = \frac{1}{2\pi \sigma_1 \sigma_2 \sqrt{1 - \rho^2}} \exp\left(-\frac{1}{2(1 - \rho^2)} \left[ \left(\frac{y_1 - \mu_1}{\sigma_1}\right)^2 - 2\rho \left(\frac{y_1 - \mu_1}{\sigma_1}\right)\left(\frac{y_2 - \mu_2}{\sigma_2}\right) + \left(\frac{y_2 - \mu_2}{\sigma_2}\right)^2 \right]\right)
$$
For units $i = 1, \dots, m$, both $y_{1,i}$ and $y_{2,i}$ are observed. For units $i = m+1, \dots, n$, only $y_{1,i}$ is observed, and the contribution to the likelihood is obtained by integrating out the unobserved variable $y_{2,i}$:
$$
\int_{-\infty}^\infty f(y_{1,i}, y_{2,i}; \boldsymbol{\theta}) \, dy_{2,i} = f_1(y_{1,i}; \mu_1, \sigma_1^2) = \frac{1}{\sqrt{2\pi \sigma_1^2}} \exp\left(-\frac{(y_{1,i} - \mu_1)^2}{2\sigma_1^2}\right)
$$
Under the assumption that the missingness mechanism does not depend on the unobserved values (Missing at Random, MAR), the joint likelihood function across all $n$ independent units is the product:
$$
L(\boldsymbol{\theta}) = \left(\prod_{i=1}^m f(y_{1,i}, y_{2,i}; \boldsymbol{\theta})\right) \left(\prod_{i=m+1}^n f_1(y_{1,i}; \mu_1, \sigma_1^2)\right)
$$

**(b)** Applying the conditional probability factorization $f(y_{1,i}, y_{2,i}) = f_1(y_{1,i}) f(y_{2,i} \mid y_{1,i})$, the joint distribution of $Y_2$ given $Y_1 = y_1$ is Gaussian:
$$
Y_2 \mid Y_1 = y_1 \sim \mathcal{N}\left(\alpha + \beta y_1, \, \sigma_{2|1}^2\right)
$$
where:
$$
\beta = \rho \frac{\sigma_2}{\sigma_1}, \quad \alpha = \mu_2 - \beta \mu_1, \quad \sigma_{2|1}^2 = \sigma_2^2(1 - \rho^2)
$$
Substituting this factorization into the joint likelihood yields:
$$
\begin{aligned}
L(\boldsymbol{\theta}) &= \left(\prod_{i=1}^m f_1(y_{1,i}; \mu_1, \sigma_1^2) f(y_{2,i} \mid y_{1,i}; \alpha, \beta, \sigma_{2|1}^2)\right) \left(\prod_{i=m+1}^n f_1(y_{1,i}; \mu_1, \sigma_1^2)\right) \\
&= \left(\prod_{i=1}^n f_1(y_{1,i}; \mu_1, \sigma_1^2)\right) \left(\prod_{i=1}^m f(y_{2,i} \mid y_{1,i}; \alpha, \beta, \sigma_{2|1}^2)\right) \\
&= L_1(\mu_1, \sigma_1^2) \times L_{2|1}(\alpha, \beta, \sigma_{2|1}^2)
\end{aligned}
$$
Because the parameter spaces of $(\mu_1, \sigma_1^2)$ and $(\alpha, \beta, \sigma_{2|1}^2)$ are variation independent (the transformation $(\mu_1, \sigma_1^2, \mu_2, \sigma_2^2, \rho) \leftrightarrow (\mu_1, \sigma_1^2, \alpha, \beta, \sigma_{2|1}^2)$ is a bijective diffeomorphism on $\mathbb{R} \times \mathbb{R}_+ \times \mathbb{R} \times \mathbb{R} \times \mathbb{R}_+$), the log-likelihood partitions additively:
$$
\ell(\boldsymbol{\theta}) = \ell_1(\mu_1, \sigma_1^2) + \ell_{2|1}(\alpha, \beta, \sigma_{2|1}^2)
$$
Maximizing $\ell_1(\mu_1, \sigma_1^2)$ utilizes all $n$ observations:
$$
\widehat{\mu}_1 = \bar{y}_{1,n} = \frac{1}{n}\sum_{i=1}^n y_{1,i}, \quad \widehat{\sigma}_1^2 = \frac{1}{n}\sum_{i=1}^n (y_{1,i} - \bar{y}_{1,n})^2
$$
The sampling variance of this estimator is $\operatorname{Var}(\widehat{\mu}_1) = \frac{\sigma_1^2}{n}$. Had we discarded the $n-m$ incomplete observations, the variance would be $\frac{\sigma_1^2}{m} > \frac{\sigma_1^2}{n}$. The additional $n-m$ units reduce the estimation variance of the secondary mean by a factor of $m/n$.

Next, maximizing $\ell_{2|1}(\alpha, \beta, \sigma_{2|1}^2)$ is identical to standard ordinary least squares (OLS) regression of $Y_2$ on $Y_1$ based on the $m$ complete cases:
$$
\widehat{\beta} = \frac{\sum_{i=1}^m (y_{1,i} - \bar{y}_{1,m})(y_{2,i} - \bar{y}_{2,m})}{\sum_{i=1}^m (y_{1,i} - \bar{y}_{1,m})^2}, \quad \widehat{\alpha} = \bar{y}_{2,m} - \widehat{\beta}\bar{y}_{1,m}
$$
By the invariance property of Maximum Likelihood Estimators, since $\mu_2 = \alpha + \beta \mu_1$, the MLE of $\mu_2$ is:
$$
\widehat{\mu}_2 = \widehat{\alpha} + \widehat{\beta}\widehat{\mu}_1 = \left(\bar{y}_{2,m} - \widehat{\beta}\bar{y}_{1,m}\right) + \widehat{\beta}\bar{y}_{1,n} = \bar{y}_{2,m} + \widehat{\beta}\left(\bar{y}_{1,n} - \bar{y}_{1,m}\right)
$$
This classic formula (Anderson 1957) reveals that the estimated primary mean is adjusted away from the complete-case mean $\bar{y}_{2,m}$ by an amount proportional to the difference between the full-sample secondary mean $\bar{y}_{1,n}$ and the sub-sample secondary mean $\bar{y}_{1,m}$. The variance of this estimator satisfies:
$$
\operatorname{Var}(\widehat{\mu}_2) \approx \frac{\sigma_2^2}{m}\left(1 - \rho^2 \frac{n - m}{n}\right) \le \frac{\sigma_2^2}{m}
$$
When $|\rho| > 0$ and $n > m$, the $n-m$ observations of $Y_1$ directly shrink the estimation variance of the primary mean $\mu_2$.

**(c)** For an unobserved primary unit $i \in \{m+1, \dots, n\}$, the secondary measurement $Y_{1,i} = y_{1,i}$ is known. Under the bivariate normal model, the minimum mean squared error predictor is the conditional expectation:
$$
\widehat{Y}_{2,i} = \operatorname{E}[Y_{2,i} \mid Y_{1,i} = y_{1,i}] = \mu_2 + \rho \frac{\sigma_2}{\sigma_1}(y_{1,i} - \mu_1)
$$
Substituting the MLE estimates yields the empirical conditional predictor:
$$
\widehat{Y}_{2,i} = \widehat{\mu}_2 + \widehat{\rho}\frac{\widehat{\sigma}_2}{\widehat{\sigma}_1}\left(Y_{1,i} - \widehat{\mu}_1\right)
$$
The conditional prediction error variance (assuming parameters are known or sample sizes are large such that estimation error is asymptotically negligible) is:
$$
\operatorname{Var}(Y_{2,i} - \widehat{Y}_{2,i} \mid Y_{1,i}) = \operatorname{Var}(Y_{2,i} \mid Y_{1,i}) = \sigma_2^2(1 - \rho^2)
$$
In contrast, if the secondary variable $Y_1$ were ignored or unobserved, the best unbiased predictor of $Y_{2,i}$ would be its marginal mean $\mu_2$, with prediction error variance:
$$
\operatorname{Var}(Y_{2,i} - \mu_2) = \sigma_2^2
$$
The relative variance reduction achieved by conditioning on the secondary attribute is:
$$
\frac{\sigma_2^2 - \sigma_2^2(1 - \rho^2)}{\sigma_2^2} = \rho^2
$$
As $|\rho| \to 1$, the conditional prediction error variance approaches zero.

**(d)** Spatial connection to heterotopic cokriging:
In spatial statistics, sampling is rarely isotopic (collocated at all points). In environmental, agronomic, and mining applications:
1. The **primary variable** $Z_1(\mathbf{s})$ (e.g., heavy metal concentration, hydraulic conductivity, soil organic carbon) is destructive, laboratory-bound, and expensive, resulting in a sparse spatial sampling footprint $S_1 = \{\mathbf{s}_1, \dots, \mathbf{s}_m\}$.
2. The **secondary variable** $Z_2(\mathbf{s})$ (e.g., digital elevation models (DEM), airborne radiometrics, satellite NDVI, electrical conductivity) is sensed remotely or automatically at negligible marginal cost across an exhaustive grid $S_2 = \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ with $n \gg m$.

Spatial heterotopic cokriging generalizes the classical conditional regression result to continuous space:
- The correlation coefficient $\rho$ generalizes to the spatial cross-covariance function $C_{12}(\mathbf{h}) = \operatorname{Cov}(Z_1(\mathbf{s}), Z_2(\mathbf{s}+\mathbf{h}))$ and cross-variogram $\gamma_{12}(\mathbf{h})$.
- The variance ratio $\sigma_2/\sigma_1$ generalizes to the ratio of spatial sill variances $\sqrt{C_{22}(\mathbf{0})/C_{11}(\mathbf{0})}$.
- The variance shrinkage factor $1 - \rho^2$ generalizes to the cokriging error variance reduction:
$$
\sigma_{\text{OCK}}^2(\mathbf{s}_0) \le \sigma_{\text{OK}}^2(\mathbf{s}_0)
$$
where $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ is the univariate ordinary kriging variance using $Z_1$ alone, and $\sigma_{\text{OCK}}^2(\mathbf{s}_0)$ is the ordinary cokriging variance incorporating the dense secondary network $Z_2$. Just as the $n-m$ auxiliary observations in classical regression anchor the primary mean and reduce prediction uncertainty via $\rho$, dense secondary spatial data "pull" the primary predictions toward local secondary features and drastically shrink the kriging estimation variance, especially in regions far from any primary sampling point. $\blacksquare$

---

### Exercise 7.2: Positive Definiteness of $2 \times 2$ Coregionalization Matrices and Cauchy-Schwarz Cross-Variogram Bounds
In the Linear Model of Coregionalization (LMC; Journel & Huijbregts 1978; Schabenberger & Gotway 2005), an $m$-variate second-order stationary spatial covariance function $\mathbf{C}(\mathbf{h}) \in \mathbb{R}^{m \times m}$ is modeled as a linear combination of $K$ basic spatial correlation functions:
$$
\mathbf{C}(\mathbf{h}) = \sum_{k=1}^K \mathbf{B}_k \rho_k(\mathbf{h})
$$
where each $\rho_k(\mathbf{h})$ is an authorized, continuous, positive definite spatial correlation function ($\rho_k(\mathbf{0}) = 1$), and each $\mathbf{B}_k \in \mathbb{R}^{m \times m}$ is a real symmetric coregionalization matrix representing the variance-covariance contribution of the $k$-th spatial scale.

For a bivariate system ($m = 2$), each coregionalization matrix has the form:
$$
\mathbf{B}_k = \begin{pmatrix} b_{11,k} & b_{12,k} \\ b_{12,k} & b_{22,k} \end{pmatrix}
$$

**(a)** State Sylvester's criterion for positive semi-definiteness of a real symmetric matrix. Prove that $\mathbf{B}_k$ is positive semi-definite ($\mathbf{B}_k \ge 0$) if and only if the following three conditions simultaneously hold:
$$
b_{11,k} \ge 0, \quad b_{22,k} \ge 0, \quad \text{and} \quad |b_{12,k}| \le \sqrt{b_{11,k} b_{22,k}}
$$

**(b)** Consider an LMC with a single spatial correlation structure ($K = 1$), such that $\mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(\mathbf{h})$. Show that the corresponding matrix semivariogram is $\boldsymbol{\Gamma}(\mathbf{h}) = \mathbf{B}(1 - \rho(\mathbf{h}))$. Prove that the positive semi-definiteness of $\mathbf{B}$ guarantees the **Cauchy-Schwarz cross-variogram inequality**:
$$
|\gamma_{12}(\mathbf{h})| \le \sqrt{\gamma_{11}(\mathbf{h}) \gamma_{22}(\mathbf{h})} \quad \forall \mathbf{h} \in \mathbb{R}^d
$$

**(c)** Consider a numerical counterexample: an analyst fits independent spherical semivariograms to two variables:
$$
\gamma_{11}(h) = 1.0 \operatorname{Sph}\left(\frac{h}{10}\right), \quad \gamma_{22}(h) = 1.0 \operatorname{Sph}\left(\frac{h}{10}\right)
$$
and naively sets the cross-variogram sill to $b_{12} = 1.2$, attempting to capture a strong positive cross-correlation. 
1. Verify that this choice violates Sylvester's criterion.
2. Construct the simple linear combination $Z_{\text{comb}}(\mathbf{s}) = Z_1(\mathbf{s}) - Z_2(\mathbf{s})$ at an arbitrary spatial point $\mathbf{s}$, and calculate its variance $\operatorname{Var}(Z_{\text{comb}}(\mathbf{s}))$.
3. Discuss the catastrophic consequences of using an indefinite coregionalization matrix in an Ordinary Cokriging system.

##### Solution to Exercise 7.2
**(a)** **Sylvester's Criterion for Positive Semi-Definiteness:**
A real symmetric matrix $\mathbf{B} \in \mathbb{R}^{m \times m}$ is positive semi-definite ($\mathbf{a}'\mathbf{B}\mathbf{a} \ge 0$ for all $\mathbf{a} \in \mathbb{R}^m$) if and only if all of its principal minors (not merely the leading principal minors) are non-negative.

For the $2 \times 2$ symmetric matrix $\mathbf{B}_k = \begin{pmatrix} b_{11,k} & b_{12,k} \\ b_{12,k} & b_{22,k} \end{pmatrix}$:
1. The principal minors of order 1 are the diagonal entries:
$$
M_1^{(1)} = b_{11,k}, \quad M_2^{(1)} = b_{22,k}
$$
Non-negativity requires:
$$
b_{11,k} \ge 0 \quad \text{and} \quad b_{22,k} \ge 0
$$
2. The principal minor of order 2 is the full determinant:
$$
M^{(2)} = \det(\mathbf{B}_k) = b_{11,k} b_{22,k} - b_{12,k}^2
$$
Non-negativity requires:
$$
\det(\mathbf{B}_k) \ge 0 \iff b_{11,k} b_{22,k} - b_{12,k}^2 \ge 0 \iff b_{12,k}^2 \le b_{11,k} b_{22,k}
$$
Taking the square root of both sides gives:
$$
|b_{12,k}| \le \sqrt{b_{11,k} b_{22,k}}
$$
To verify sufficiency via the quadratic form, let $\mathbf{a} = (a_1, a_2)' \in \mathbb{R}^2$:
$$
Q(\mathbf{a}) = \mathbf{a}'\mathbf{B}_k\mathbf{a} = b_{11,k} a_1^2 + 2b_{12,k} a_1 a_2 + b_{22,k} a_2^2
$$
- If $b_{11,k} = 0$, then $|b_{12,k}| \le \sqrt{0 \cdot b_{22,k}} = 0$, which forces $b_{12,k} = 0$. Then $Q(\mathbf{a}) = b_{22,k} a_2^2 \ge 0$ since $b_{22,k} \ge 0$.
- If $b_{11,k} > 0$, we complete the square with respect to $a_1$:
$$
Q(\mathbf{a}) = b_{11,k}\left(a_1 + \frac{b_{12,k}}{b_{11,k}} a_2\right)^2 + \left(b_{22,k} - \frac{b_{12,k}^2}{b_{11,k}}\right) a_2^2 = b_{11,k}\left(a_1 + \frac{b_{12,k}}{b_{11,k}} a_2\right)^2 + \frac{\det(\mathbf{B}_k)}{b_{11,k}} a_2^2
$$
Because $b_{11,k} > 0$, the first term is a square multiplied by a positive scalar, hence non-negative. The second term is non-negative for all $a_2$ if and only if $\det(\mathbf{B}_k) \ge 0$. Thus $Q(\mathbf{a}) \ge 0$ for all $\mathbf{a} \in \mathbb{R}^2$ if and only if $b_{11,k} \ge 0$, $b_{22,k} \ge 0$, and $|b_{12,k}| \le \sqrt{b_{11,k} b_{22,k}}$.

**(b)** For an LMC with $K=1$, the cross-covariance matrix is $\mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(\mathbf{h})$. At lag $\mathbf{h} = \mathbf{0}$, $\rho(\mathbf{0}) = 1$, so $\mathbf{C}(\mathbf{0}) = \mathbf{B}$.
The matrix semivariogram is defined under second-order stationarity as:
$$
\boldsymbol{\Gamma}(\mathbf{h}) = \mathbf{C}(\mathbf{0}) - \frac{1}{2}\left(\mathbf{C}(\mathbf{h}) + \mathbf{C}(-\mathbf{h})\right)
$$
Since $\rho(\mathbf{h})$ is an isotropic or symmetric correlation function ($\rho(\mathbf{h}) = \rho(-\mathbf{h})$) and $\mathbf{B} = \mathbf{B}'$:
$$
\boldsymbol{\Gamma}(\mathbf{h}) = \mathbf{B} - \mathbf{B} \rho(\mathbf{h}) = \mathbf{B}\left(1 - \rho(\mathbf{h})\right)
$$
Component-wise, the auto- and cross-semivariograms are:
$$
\gamma_{11}(\mathbf{h}) = b_{11}\left(1 - \rho(\mathbf{h})\right), \quad \gamma_{22}(\mathbf{h}) = b_{22}\left(1 - \rho(\mathbf{h})\right), \quad \gamma_{12}(\mathbf{h}) = b_{12}\left(1 - \rho(\mathbf{h})\right)
$$
Because $\rho(\mathbf{h})$ is a valid correlation function, $|\rho(\mathbf{h})| \le \rho(\mathbf{0}) = 1$, which ensures that $1 - \rho(\mathbf{h}) \ge 0$ for all $\mathbf{h} \in \mathbb{R}^d$.
Now consider the absolute cross-semivariance:
$$
|\gamma_{12}(\mathbf{h})| = |b_{12}| \left(1 - \rho(\mathbf{h})\right)
$$
From part (a), positive semi-definiteness of $\mathbf{B}$ guarantees $|b_{12}| \le \sqrt{b_{11} b_{22}}$. Substituting this inequality:
$$
\begin{aligned}
|\gamma_{12}(\mathbf{h})| &\le \sqrt{b_{11} b_{22}} \left(1 - \rho(\mathbf{h})\right) \\
&= \sqrt{b_{11}\left(1 - \rho(\mathbf{h})\right)} \cdot \sqrt{b_{22}\left(1 - \rho(\mathbf{h})\right)} \\
&= \sqrt{\gamma_{11}(\mathbf{h}) \gamma_{22}(\mathbf{h})}
\end{aligned}
$$
This establishes the Cauchy-Schwarz cross-variogram inequality for all spatial lags $\mathbf{h} \in \mathbb{R}^d$.

**(c)**
1. **Verification of violation:**
Here, $b_{11} = 1.0$, $b_{22} = 1.0$, and $b_{12} = 1.2$.
The determinant of the proposed coregionalization matrix is:
$$
\det(\mathbf{B}) = b_{11} b_{22} - b_{12}^2 = (1.0)(1.0) - (1.2)^2 = 1.0 - 1.44 = -0.44 < 0
$$
Because $\det(\mathbf{B}) < 0$, Sylvester's criterion is violated; the matrix $\mathbf{B}$ has one positive eigenvalue ($\lambda_1 = 1.0 + 1.2 = 2.2$) and one negative eigenvalue ($\lambda_2 = 1.0 - 1.2 = -0.2$). Hence $\mathbf{B}$ is indefinite.

2. **Variance of the linear combination:**
Consider the linear combination $Z_{\text{comb}}(\mathbf{s}) = Z_1(\mathbf{s}) - Z_2(\mathbf{s}) = \mathbf{a}'\mathbf{Z}(\mathbf{s})$ with $\mathbf{a} = (1, -1)'$.
Its variance is given by the quadratic form in $\mathbf{C}(\mathbf{0}) = \mathbf{B}$:
$$
\begin{aligned}
\operatorname{Var}(Z_{\text{comb}}(\mathbf{s})) &= \operatorname{Var}(Z_1(\mathbf{s}) - Z_2(\mathbf{s})) \\
&= \operatorname{Var}(Z_1(\mathbf{s})) + \operatorname{Var}(Z_2(\mathbf{s})) - 2\operatorname{Cov}(Z_1(\mathbf{s}), Z_2(\mathbf{s})) \\
&= b_{11} + b_{22} - 2b_{12} \\
&= 1.0 + 1.0 - 2(1.2) = 2.0 - 2.4 = -0.4 < 0
\end{aligned}
$$
The linear combination has a strictly negative variance of $-0.4$.

3. **Catastrophic consequences in Cokriging:**
By Kolmogorov's second axiom of probability theory, the variance of any real-valued random variable must be non-negative: $\operatorname{Var}(W) \ge 0$. A negative variance is a mathematical impossibility.
In an Ordinary Cokriging system:
- The global cokriging covariance matrix $\mathbf{V} = \begin{pmatrix} \mathbf{V}_{11} & \mathbf{V}_{12} \\ \mathbf{V}_{21} & \mathbf{V}_{22} \end{pmatrix}$ inherits the indefiniteness of $\mathbf{B}$. Specifically, for two collocated observations at the same site, the submatrix is identical to $\mathbf{B}$, making $\mathbf{V}$ indefinite.
- Indefiniteness destroys the guarantees of the Karush-Kuhn-Tucker (KKT) conditions: the cokriging linear system no longer corresponds to a convex quadratic minimization problem, and the resulting stationary point can be a saddle point or a local maximum of the error variance.
- The resulting ordinary cokriging prediction error variance $\sigma_{\text{OCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1$ can evaluate to negative values ($\sigma_{\text{OCK}}^2(\mathbf{s}_0) < 0$), producing undefined (imaginary) prediction standard errors.
- The linear system matrix can become singular or severely ill-conditioned, yielding wild, oscillatory cokriging weights $\lambda_{1,i}, \lambda_{2,j} \gg 10^3$ that completely destabilize predictions.
Therefore, ensuring that every coregionalization matrix $\mathbf{B}_k$ is strictly positive semi-definite is an absolute requirement in multivariate geostatistics. $\blacksquare$

---

### Exercise 7.3: Pseudo-Cross-Variogram vs. Traditional Cross-Variogram Under Complete Heterotopy
In multivariate geostatistics, modeling the spatial cross-dependence between two variables $Z_1(\mathbf{s})$ and $Z_2(\mathbf{s})$ is traditionally achieved using the cross-variogram:
$$
\gamma_{12}(\mathbf{h}) = \frac{1}{2}\operatorname{E}\left[(Z_1(\mathbf{s}+\mathbf{h}) - Z_1(\mathbf{s}))(Z_2(\mathbf{s}+\mathbf{h}) - Z_2(\mathbf{s}))\right]
$$
However, in environmental remediation, mining, and agricultural ecology, sample collection is often **completely heterotopic**: variable $Z_1$ is measured at locations $S_1 = \{\mathbf{s}_1^{(1)}, \dots, \mathbf{s}_{n_1}^{(1)}\}$ and variable $Z_2$ is measured at locations $S_2 = \{\mathbf{s}_1^{(2)}, \dots, \mathbf{s}_{n_2}^{(2)}\}$, with strictly disjoint sample footprints ($S_1 \cap S_2 = \emptyset$).

**(a)** State the classical Matheron method-of-moments estimator $\widehat{\gamma}_{12}(\mathbf{h})$ for the cross-variogram. Rigorously explain why $\widehat{\gamma}_{12}(\mathbf{h})$ cannot be computed from empirical data under complete heterotopy.

**(b)** To overcome this limitation, Clark, Basinger, and Roebuck (1989) introduced the **pseudo-cross-variogram**:
$$
g_{12}(\mathbf{h}) = \frac{1}{2}\operatorname{Var}\left(Z_1(\mathbf{s}+\mathbf{h}) - Z_2(\mathbf{s})\right)
$$
Formulate the sample estimator $\widehat{g}_{12}(\mathbf{h})$ and prove that it can be calculated directly from completely heterotopic data where $S_1 \cap S_2 = \emptyset$.

**(c)** Assuming second-order stationarity with means $\mu_1, \mu_2$, auto-covariance functions $C_{11}(\mathbf{h}), C_{22}(\mathbf{h})$, and cross-covariance function $C_{12}(\mathbf{h}) = \operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s}))$, expand the variance definition of $g_{12}(\mathbf{h})$ and prove the fundamental algebraic identity:
$$
g_{12}(\mathbf{h}) = \frac{1}{2}\left(C_{11}(\mathbf{0}) + C_{22}(\mathbf{0})\right) - C_{12}(\mathbf{h})
$$

**(d)** Compare the behavior of the traditional cross-variogram $\gamma_{12}(\mathbf{h})$ and the pseudo-cross-variogram $g_{12}(\mathbf{h})$ at the spatial origin ($\mathbf{h} \to \mathbf{0}$). Prove that while $\gamma_{12}(\mathbf{0}) = 0$, $g_{12}(\mathbf{0}) > 0$ unless $Z_1(\mathbf{s})$ and $Z_2(\mathbf{s})$ are identical random fields almost surely. What physical interpretation does the intercept $g_{12}(\mathbf{0})$ possess?

##### Solution to Exercise 7.3
**(a)** The classical Matheron method-of-moments estimator for the cross-variogram is:
$$
\widehat{\gamma}_{12}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} \left(Z_1(\mathbf{s}_i) - Z_1(\mathbf{s}_j)\right)\left(Z_2(\mathbf{s}_i) - Z_2(\mathbf{s}_j)\right)
$$
where $N(\mathbf{h}) = \{(\mathbf{s}_i, \mathbf{s}_j) : \mathbf{s}_i - \mathbf{s}_j \approx \mathbf{h}\}$.
Notice the structure of the summand: it involves the product of two spatial increments:
$$
\Delta Z_1 = Z_1(\mathbf{s}_i) - Z_1(\mathbf{s}_j) \quad \text{and} \quad \Delta Z_2 = Z_2(\mathbf{s}_i) - Z_2(\mathbf{s}_j)
$$
Evaluating $\Delta Z_1$ requires observing variable $Z_1$ at both site $\mathbf{s}_i$ and site $\mathbf{s}_j$. Similarly, evaluating $\Delta Z_2$ requires observing variable $Z_2$ at both site $\mathbf{s}_i$ and site $\mathbf{s}_j$.
Therefore, a pair $(\mathbf{s}_i, \mathbf{s}_j)$ can contribute to the sum if and only if both variables $Z_1$ and $Z_2$ are simultaneously measured at site $\mathbf{s}_i$ AND both variables are simultaneously measured at site $\mathbf{s}_j$:
$$
\mathbf{s}_i \in (S_1 \cap S_2) \quad \text{and} \quad \mathbf{s}_j \in (S_1 \cap S_2)
$$
Under complete heterotopy, by definition, $S_1 \cap S_2 = \emptyset$. There is not a single location in the study domain where both variables are collocated. Consequently:
$$
|N(\mathbf{h})| = 0 \quad \forall \mathbf{h} \in \mathbb{R}^d
$$
The pair set is empty, the denominator is zero, and the classical cross-variogram estimator $\widehat{\gamma}_{12}(\mathbf{h})$ is completely undefined.

**(b)** The pseudo-cross-variogram of Clark, Basinger, and Roebuck (1989) is defined as:
$$
g_{12}(\mathbf{h}) = \frac{1}{2}\operatorname{Var}\left(Z_1(\mathbf{s}+\mathbf{h}) - Z_2(\mathbf{s})\right)
$$
Under the assumption of constant means $\operatorname{E}[Z_1(\mathbf{s})] = \mu_1$ and $\operatorname{E}[Z_2(\mathbf{s})] = \mu_2$, the sample estimator is:
$$
\widehat{g}_{12}(\mathbf{h}) = \frac{1}{2|N_{12}(\mathbf{h})|} \sum_{(\mathbf{s}_i^{(1)}, \mathbf{s}_j^{(2)}) \in N_{12}(\mathbf{h})} \left[\left(Z_1(\mathbf{s}_i^{(1)}) - Z_2(\mathbf{s}_j^{(2)})\right) - (\bar{Z}_1 - \bar{Z}_2)\right]^2
$$
where:
$$
N_{12}(\mathbf{h}) = \left\{(\mathbf{s}_i^{(1)}, \mathbf{s}_j^{(2)}) : \mathbf{s}_i^{(1)} \in S_1, \, \mathbf{s}_j^{(2)} \in S_2, \, \mathbf{s}_i^{(1)} - \mathbf{s}_j^{(2)} \approx \mathbf{h}\right\}
$$
and $\bar{Z}_1 = \frac{1}{n_1}\sum_{i=1}^{n_1} Z_1(\mathbf{s}_i^{(1)})$, $\bar{Z}_2 = \frac{1}{n_2}\sum_{j=1}^{n_2} Z_2(\mathbf{s}_j^{(2)})$.
Notice that each term in the summation involves exactly one measurement of $Z_1$ at location $\mathbf{s}_i^{(1)} \in S_1$ and exactly one measurement of $Z_2$ at location $\mathbf{s}_j^{(2)} \in S_2$. Neither site is required to have collocated observations. As long as there exist pairs of sites across the two disjoint survey networks whose displacement vector $\mathbf{s}_i^{(1)} - \mathbf{s}_j^{(2)}$ matches lag $\mathbf{h}$, $|N_{12}(\mathbf{h})| > 0$ and $\widehat{g}_{12}(\mathbf{h})$ can be directly computed.

**(c)** Expanding the variance of the difference:
For any two random variables $X$ and $Y$, the variance of their difference is:
$$
\operatorname{Var}(X - Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) - 2\operatorname{Cov}(X, Y)
$$
Setting $X = Z_1(\mathbf{s}+\mathbf{h})$ and $Y = Z_2(\mathbf{s})$:
$$
g_{12}(\mathbf{h}) = \frac{1}{2}\operatorname{Var}\left(Z_1(\mathbf{s}+\mathbf{h}) - Z_2(\mathbf{s})\right) = \frac{1}{2}\left[\operatorname{Var}(Z_1(\mathbf{s}+\mathbf{h})) + \operatorname{Var}(Z_2(\mathbf{s})) - 2\operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s}))\right]
$$
Under second-order stationarity:
1. $\operatorname{Var}(Z_1(\mathbf{s}+\mathbf{h})) = C_{11}(\mathbf{0})$ (the sill/variance of variable 1).
2. $\operatorname{Var}(Z_2(\mathbf{s})) = C_{22}(\mathbf{0})$ (the sill/variance of variable 2).
3. $\operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s})) = C_{12}(\mathbf{h})$ (the spatial cross-covariance at lag $\mathbf{h}$).
Substituting these terms into the variance expansion:
$$
\begin{aligned}
g_{12}(\mathbf{h}) &= \frac{1}{2}\left[C_{11}(\mathbf{0}) + C_{22}(\mathbf{0}) - 2C_{12}(\mathbf{h})\right] \\
&= \frac{1}{2}\left(C_{11}(\mathbf{0}) + C_{22}(\mathbf{0})\right) - C_{12}(\mathbf{h})
\end{aligned}
$$
This proves the fundamental identity connecting the pseudo-cross-variogram to the underlying covariance and cross-covariance functions.

**(d)** **Behavior at the spatial origin ($\mathbf{h} \to \mathbf{0}$):**
For the traditional cross-variogram:
$$
\gamma_{12}(\mathbf{0}) = \frac{1}{2}\operatorname{E}\left[(Z_1(\mathbf{s}) - Z_1(\mathbf{s}))(Z_2(\mathbf{s}) - Z_2(\mathbf{s}))\right] = \frac{1}{2}\operatorname{E}[0 \cdot 0] = 0
$$
Thus, the traditional cross-variogram is strictly anchored at 0 at zero separation.
For the pseudo-cross-variogram, evaluating at $\mathbf{h} = \mathbf{0}$:
$$
g_{12}(\mathbf{0}) = \frac{1}{2}\operatorname{Var}\left(Z_1(\mathbf{s}) - Z_2(\mathbf{s})\right) = \frac{1}{2}\left(C_{11}(\mathbf{0}) + C_{22}(\mathbf{0})\right) - C_{12}(\mathbf{0})
$$
By the properties of variance, $\operatorname{Var}(Z_1(\mathbf{s}) - Z_2(\mathbf{s})) \ge 0$, with equality holding if and only if $Z_1(\mathbf{s}) - Z_2(\mathbf{s}) = c$ (a constant) almost surely. If the variables are centered or have equal means, equality implies $Z_1(\mathbf{s}) = Z_2(\mathbf{s})$ almost surely.
When $Z_1$ and $Z_2$ represent two distinct physical attributes (e.g., copper ppm and zinc ppm, or soil moisture % and elevation m):
$$
g_{12}(\mathbf{0}) = \frac{1}{2}\operatorname{Var}\left(Z_1(\mathbf{s}) - Z_2(\mathbf{s})\right) > 0
$$
The intercept $g_{12}(\mathbf{0})$ does NOT vanish! 
**Physical Interpretation:**
The intercept $g_{12}(\mathbf{0})$ represents the intrinsic baseline variance between the two different physical variables when evaluated at the exact same spatial location. Because $Z_1$ and $Z_2$ are fundamentally different physical quantities (potentially measured in different physical units), their difference at a single site is not zero; it reflects both their difference in variance and their imperfect point-collocated correlation ($C_{12}(\mathbf{0}) < \sqrt{C_{11}(\mathbf{0})C_{22}(\mathbf{0})}$).
Furthermore, while the traditional cross-variogram satisfies reflection symmetry $\gamma_{12}(\mathbf{h}) = \gamma_{21}(-\mathbf{h})$, the pseudo-cross-variogram generally satisfies $g_{12}(\mathbf{h}) \ne g_{12}(-\mathbf{h})$ whenever the cross-covariance is asymmetric ($C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})$). $\blacksquare$

---

### Exercise 7.4: Principal Component Kriging vs. Full Ordinary Cokriging
Full Ordinary Cokriging (OCK) of an $m$-variate spatial random field $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_m(\mathbf{s}))'$ measured at $n$ isotopic locations $\mathbf{s}_1, \dots, \mathbf{s}_n$ requires setting up and solving a joint linear system of dimension $(mn + m) \times (mn + m)$. For large sample sizes $n$ or moderate numbers of variables $m$, the cubic computational complexity $\mathcal{O}(m^3 n^3)$ becomes prohibitively expensive.

Suppose the $m$ spatial processes follow the **intrinsic correlation model** (an LMC with a single spatial correlation structure):
$$
\mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(\mathbf{h})
$$
where $\rho(\mathbf{h})$ is a valid scalar spatial correlation function ($\rho(\mathbf{0}) = 1$), and $\mathbf{B} \in \mathbb{R}^{m \times m}$ is a symmetric positive definite coregionalization matrix.

**(a)** State the assumption of intrinsic correlation and express the covariance between $Z_j(\mathbf{s})$ and $Z_k(\mathbf{s}+\mathbf{h})$ in terms of the elements $b_{jk}$ of $\mathbf{B}$ and $\rho(\mathbf{h})$.

**(b)** Let $\mathbf{B} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}'$ be the spectral (eigendecomposition) of $\mathbf{B}$, where $\mathbf{Q} \in \mathbb{R}^{m \times m}$ is an orthonormal matrix of eigenvectors ($\mathbf{Q}'\mathbf{Q} = \mathbf{Q}\mathbf{Q}' = \mathbf{I}_m$) and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_m)$ is the diagonal matrix of positive eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_m > 0$. Define the transformed spatial principal component vector field:
$$
\mathbf{Y}(\mathbf{s}) = \mathbf{Q}'\mathbf{Z}(\mathbf{s})
$$

**(c)** Compute the cross-covariance matrix function of the transformed variables:
$$
\mathbf{C}_{\mathbf{Y}}(\mathbf{h}) = \operatorname{Cov}\left(\mathbf{Y}(\mathbf{s}), \mathbf{Y}(\mathbf{s}+\mathbf{h})\right)
$$
Prove that $\mathbf{C}_{\mathbf{Y}}(\mathbf{h}) = \boldsymbol{\Lambda} \rho(\mathbf{h})$, and deduce that the transformed components $Y_1(\mathbf{s}), \dots, Y_m(\mathbf{s})$ are mutually uncorrelated at all spatial separation vectors $\mathbf{h} \in \mathbb{R}^d$.

**(d)** Under isotopic sampling (all $m$ attributes observed at sites $\mathbf{s}_1, \dots, \mathbf{s}_n$), prove that:
1. The full Ordinary Cokriging predictor for each transformed variable $Y_j(\mathbf{s}_0)$ completely decouples into independent univariate Ordinary Kriging (OK) on $Y_j$.
2. The back-transformed predictor:
$$
\widehat{\mathbf{Z}}(\mathbf{s}_0) = \mathbf{Q} \widehat{\mathbf{Y}}(\mathbf{s}_0)
$$
is mathematically identical to the full multivariate Ordinary Cokriging predictor $\widehat{\mathbf{Z}}_{\text{OCK}}(\mathbf{s}_0)$.
3. Quantify the computational speedup: contrast the FLOP complexity of full OCK against Principal Component Kriging (PCK).

##### Solution to Exercise 7.4
**(a)** Under the intrinsic correlation model, all auto-covariances and cross-covariances share the exact same spatial correlation structure $\rho(\mathbf{h})$. The covariance between variable $j$ at location $\mathbf{s}$ and variable $k$ at location $\mathbf{s}+\mathbf{h}$ factors into a non-spatial coregionalization coefficient $b_{jk}$ and the spatial correlation $\rho(\mathbf{h})$:
$$
\operatorname{Cov}(Z_j(\mathbf{s}), Z_k(\mathbf{s}+\mathbf{h})) = C_{jk}(\mathbf{h}) = b_{jk} \rho(\mathbf{h})
$$
In matrix form:
$$
\mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(\mathbf{h})
$$
where $\mathbf{B} = [b_{jk}]_{j,k=1}^m$ is symmetric and positive definite ($\mathbf{B} \succ 0$).

**(b)** Because $\mathbf{B}$ is a real symmetric $m \times m$ matrix, the Spectral Theorem guarantees the existence of an orthonormal basis of eigenvectors. That is:
$$
\mathbf{B} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}' = \sum_{j=1}^m \lambda_j \mathbf{q}_j \mathbf{q}_j'
$$
where $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_m]$ satisfies $\mathbf{Q}'\mathbf{Q} = \mathbf{Q}\mathbf{Q}' = \mathbf{I}_m$, and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_m)$ with eigenvalues $\lambda_j > 0$.
The spatial principal component transformation defines a new $m$-variate random field $\mathbf{Y}(\mathbf{s}) = (Y_1(\mathbf{s}), \dots, Y_m(\mathbf{s}))'$ via:
$$
\mathbf{Y}(\mathbf{s}) = \mathbf{Q}'\mathbf{Z}(\mathbf{s}) \iff Y_j(\mathbf{s}) = \mathbf{q}_j'\mathbf{Z}(\mathbf{s}), \quad j = 1, \dots, m
$$
Because $\mathbf{Q}$ is orthogonal, the inverse transformation is simply:
$$
\mathbf{Z}(\mathbf{s}) = \mathbf{Q}\mathbf{Y}(\mathbf{s})
$$

**(c)** By bilinearity of the covariance operator:
$$
\mathbf{C}_{\mathbf{Y}}(\mathbf{h}) = \operatorname{Cov}\left(\mathbf{Y}(\mathbf{s}), \mathbf{Y}(\mathbf{s}+\mathbf{h})\right) = \operatorname{Cov}\left(\mathbf{Q}'\mathbf{Z}(\mathbf{s}), \mathbf{Q}'\mathbf{Z}(\mathbf{s}+\mathbf{h})\right) = \mathbf{Q}' \operatorname{Cov}\left(\mathbf{Z}(\mathbf{s}), \mathbf{Z}(\mathbf{s}+\mathbf{h})\right) \mathbf{Q} = \mathbf{Q}' \mathbf{C}(\mathbf{h}) \mathbf{Q}
$$
Substituting the intrinsic correlation structure $\mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(\mathbf{h})$:
$$
\mathbf{C}_{\mathbf{Y}}(\mathbf{h}) = \mathbf{Q}' \left(\mathbf{B} \rho(\mathbf{h})\right) \mathbf{Q} = \left(\mathbf{Q}' \mathbf{B} \mathbf{Q}\right) \rho(\mathbf{h})
$$
Since $\mathbf{B} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}'$ and $\mathbf{Q}'\mathbf{Q} = \mathbf{I}_m$:
$$
\mathbf{Q}' \mathbf{B} \mathbf{Q} = \mathbf{Q}' (\mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}') \mathbf{Q} = (\mathbf{Q}'\mathbf{Q}) \boldsymbol{\Lambda} (\mathbf{Q}'\mathbf{Q}) = \mathbf{I}_m \boldsymbol{\Lambda} \mathbf{I}_m = \boldsymbol{\Lambda}
$$
Therefore:
$$
\mathbf{C}_{\mathbf{Y}}(\mathbf{h}) = \boldsymbol{\Lambda} \rho(\mathbf{h}) = \begin{pmatrix} \lambda_1 \rho(\mathbf{h}) & 0 & \cdots & 0 \\ 0 & \lambda_2 \rho(\mathbf{h}) & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_m \rho(\mathbf{h}) \end{pmatrix}
$$
For any two distinct principal components $j \ne k$ and for any arbitrary spatial lag $\mathbf{h} \in \mathbb{R}^d$:
$$
\operatorname{Cov}\left(Y_j(\mathbf{s}), Y_k(\mathbf{s}+\mathbf{h})\right) = [\mathbf{C}_{\mathbf{Y}}(\mathbf{h})]_{jk} = 0
$$
The principal components are orthogonal and mutually uncorrelated at all spatial distances.

**(d)** 
1. **Decoupling of the Cokriging System:**
In the transformed variable space, we seek to predict $Y_j(\mathbf{s}_0)$ from the observed vectors $\mathbf{Y}(\mathbf{s}_1), \dots, \mathbf{Y}(\mathbf{s}_n)$.
Because $\operatorname{Cov}(Y_j(\mathbf{s}), Y_k(\mathbf{s}')) = 0$ for all $k \ne j$ across all pairs of sample sites, the cross-variable covariance blocks in the joint Ordinary Cokriging system are identically zero:
$$
\mathbf{V}_{jk} = \mathbf{0} \quad \text{and} \quad \mathbf{c}_{jk, 0} = \mathbf{0} \quad \forall k \ne j
$$
Under isotopic sampling, the unbiasedness constraints $\mathbf{1}'\boldsymbol{\lambda}_{jk} = \delta_{jk}$ ensure that the cross-variable weight vectors are identically zero:
$$
\boldsymbol{\lambda}_{jk} = \mathbf{0} \quad \forall k \ne j
$$
Hence, the Ordinary Cokriging predictor of $Y_j(\mathbf{s}_0)$ reduces strictly to univariate Ordinary Kriging using only observations of $Y_j$:
$$
\widehat{Y}_j(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i^{(j)} Y_j(\mathbf{s}_i) = (\boldsymbol{\lambda}^{(j)})' \mathbf{Y}_j
$$
where $\boldsymbol{\lambda}^{(j)}$ solves the univariate Ordinary Kriging system:
$$
\begin{pmatrix} \lambda_j \mathbf{R} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda}^{(j)} \\ m_j \end{pmatrix} = \begin{pmatrix} \lambda_j \mathbf{r}_0 \\ 1 \end{pmatrix}
$$
where $\mathbf{R} = [\rho(\mathbf{s}_i - \mathbf{s}_{i'})]_{i,i'=1}^n \in \mathbb{R}^{n \times n}$ and $\mathbf{r}_0 = [\rho(\mathbf{s}_i - \mathbf{s}_0)]_{i=1}^n \in \mathbb{R}^n$.
Dividing the first block equation by the scalar eigenvalue $\lambda_j > 0$ and redefining the Lagrange multiplier $\widetilde{m}_j = m_j / \lambda_j$:
$$
\begin{pmatrix} \mathbf{R} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda}^{(j)} \\ \widetilde{m}_j \end{pmatrix} = \begin{pmatrix} \mathbf{r}_0 \\ 1 \end{pmatrix}
$$
Notice that $\lambda_j$ cancels completely from the system! The weight vector $\boldsymbol{\lambda}^{(j)}$ depends solely on the spatial correlation matrix $\mathbf{R}$ and vector $\mathbf{r}_0$. Consequently, the kriging weight vector is identical across all $m$ principal components:
$$
\boldsymbol{\lambda}^{(1)} = \boldsymbol{\lambda}^{(2)} = \dots = \boldsymbol{\lambda}^{(m)} = \boldsymbol{\lambda}_{\text{univ}}
$$

2. **Equivalence of Back-Transformed Predictor:**
Stacking the $m$ univariate predictions into vector form:
$$
\widehat{\mathbf{Y}}(\mathbf{s}_0) = \begin{pmatrix} \widehat{Y}_1(\mathbf{s}_0) \\ \vdots \\ \widehat{Y}_m(\mathbf{s}_0) \end{pmatrix} = \begin{pmatrix} \sum_{i=1}^n \lambda_{\text{univ}, i} Y_1(\mathbf{s}_i) \\ \vdots \\ \sum_{i=1}^n \lambda_{\text{univ}, i} Y_m(\mathbf{s}_i) \end{pmatrix} = \sum_{i=1}^n \lambda_{\text{univ}, i} \mathbf{Y}(\mathbf{s}_i)
$$
Applying the back-transformation:
$$
\widehat{\mathbf{Z}}(\mathbf{s}_0) = \mathbf{Q} \widehat{\mathbf{Y}}(\mathbf{s}_0) = \mathbf{Q} \left(\sum_{i=1}^n \lambda_{\text{univ}, i} \mathbf{Y}(\mathbf{s}_i)\right) = \sum_{i=1}^n \lambda_{\text{univ}, i} \mathbf{Q}\mathbf{Y}(\mathbf{s}_i) = \sum_{i=1}^n \lambda_{\text{univ}, i} \mathbf{Z}(\mathbf{s}_i)
$$
By the Autokrigeability Theorem (Wackernagel 2003; S&G Section 6.3), when variables share an intrinsic correlation LMC under isotopic sampling, full multivariate Ordinary Cokriging produces weights identical to univariate Ordinary Kriging: $\widehat{\mathbf{Z}}_{\text{OCK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_{\text{univ}, i} \mathbf{Z}(\mathbf{s}_i)$.
Thus:
$$
\widehat{\mathbf{Z}}(\mathbf{s}_0) \equiv \widehat{\mathbf{Z}}_{\text{OCK}}(\mathbf{s}_0)
$$
The Principal Component Kriging predictor is mathematically and numerically identical to full multivariate Ordinary Cokriging.

3. **FLOP Complexity and Computational Speedup:**
- **Full Ordinary Cokriging:**
  The full system matrix has dimension $N_{\text{full}} = mn + m \approx mn$.
  Solving via Cholesky or LU factorization requires:
  $$
  \text{FLOPs}_{\text{OCK}} \approx \frac{1}{3}(mn)^3 = \frac{1}{3} m^3 n^3
  $$
- **Principal Component Kriging:**
  1. Spectral decomposition of $m \times m$ matrix $\mathbf{B}$: $\mathcal{O}(m^3)$ operations.
  2. Data transformation $\mathbf{Y}(\mathbf{s}_i) = \mathbf{Q}'\mathbf{Z}(\mathbf{s}_i)$ for $i=1,\dots,n$: $m^2 n$ operations.
  3. Factorization of the single $n \times n$ correlation matrix $\mathbf{R}$: $\frac{1}{3} n^3$ operations.
  4. Forward/back substitution to solve for $\boldsymbol{\lambda}_{\text{univ}}$: $2n^2$ operations.
  5. Computation of $\widehat{Y}_j(\mathbf{s}_0)$ and back-transformation $\mathbf{Q}\widehat{\mathbf{Y}}(\mathbf{s}_0)$: $\mathcal{O}(m n + m^2)$ operations per prediction point.
  Total training FLOPs:
  $$
  \text{FLOPs}_{\text{PCK}} \approx \frac{1}{3} n^3 + m^2 n + \mathcal{O}(m^3)
  $$
**Quantitative Comparison:**
For a realistic geochemical survey with $m = 10$ elemental assays at $n = 2{,}000$ sampling sites:
- Full OCK solves a system of size $20{,}000 \times 20{,}000$:
$$
\text{FLOPs}_{\text{OCK}} \approx \frac{1}{3}(20{,}000)^3 \approx 2.67 \times 10^{12} \text{ FLOPs} \quad (\approx 2.67 \text{ TeraFLOPs})
$$
- Principal Component Kriging factorizes a single $2{,}000 \times 2{,}000$ matrix:
$$
\text{FLOPs}_{\text{PCK}} \approx \frac{1}{3}(2{,}000)^3 + (10)^2(2{,}000) \approx 2.67 \times 10^9 \text{ FLOPs} \quad (\approx 2.67 \text{ GigaFLOPs})
$$
The computational speedup factor is:
$$
\text{Speedup} = \frac{\frac{1}{3} m^3 n^3}{\frac{1}{3} n^3} = m^3 = 10^3 = 1{,}000\times
$$
Principal Component Kriging yields an exact, lossless $1{,}000$-fold computational acceleration. $\blacksquare$

---

## Tier I: Core Theoretical Re-Derivations

### Exercise 7.5: Cross-Covariance Transpose Symmetry and Lag Reflection
Let $\{\mathbf{Z}(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$ be a second-order stationary bivariate random field:
$$
\mathbf{Z}(\mathbf{s}) = \begin{pmatrix} Z_1(\mathbf{s}) \\ Z_2(\mathbf{s}) \end{pmatrix}, \quad \operatorname{E}[\mathbf{Z}(\mathbf{s})] = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}
$$
The cross-covariance between variables $i, j \in \{1, 2\}$ at spatial lag $\mathbf{h} \in \mathbb{R}^d$ is defined as:
$$
C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s})) = \operatorname{E}[(Z_i(\mathbf{s}+\mathbf{h}) - \mu_i)(Z_j(\mathbf{s}) - \mu_j)]
$$
**(a)** Prove that auto-covariance functions are even functions of lag: $C_{ii}(\mathbf{h}) = C_{ii}(-\mathbf{h})$.
**(b)** Prove that cross-covariance functions satisfy the **lag reflection transpose identity**:
$$
C_{ij}(\mathbf{h}) = C_{ji}(-\mathbf{h}) \quad \forall \mathbf{h} \in \mathbb{R}^d
$$
and express this in terms of the matrix covariance kernel $\mathbf{C}(\mathbf{h})$ as:
$$
\mathbf{C}(\mathbf{h})' = \mathbf{C}(-\mathbf{h})
$$
**(c)** Give a physical example (e.g., pollutant dispersal downwind from a smoke stack, or groundwater contamination downstream from a chemical plant) where $C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})$ (cross-covariance asymmetry), explaining why cross-covariances are not generally even functions of lag.

##### Solution to Exercise 7.5
**(a)** By second-order stationarity, the covariance depends only on the spatial displacement. For the auto-covariance of variable $i$:
$$
C_{ii}(-\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}-\mathbf{h}), Z_i(\mathbf{s}))
$$
Let $\mathbf{u} = \mathbf{s} - \mathbf{h}$. Then $\mathbf{s} = \mathbf{u} + \mathbf{h}$, so:
$$
C_{ii}(-\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{u}), Z_i(\mathbf{u}+\mathbf{h})) = \operatorname{Cov}(Z_i(\mathbf{u}+\mathbf{h}), Z_i(\mathbf{u})) = C_{ii}(\mathbf{h})
$$
Thus, auto-covariance kernels are strictly symmetric under lag reversal: $C_{ii}(\mathbf{h}) = C_{ii}(-\mathbf{h})$.

**(b)** For the cross-covariance between variables $i$ and $j$:
$$
C_{ji}(-\mathbf{h}) = \operatorname{Cov}(Z_j(\mathbf{s}-\mathbf{h}), Z_i(\mathbf{s}))
$$
Let $\mathbf{u} = \mathbf{s} - \mathbf{h}$, which implies $\mathbf{s} = \mathbf{u} + \mathbf{h}$. By stationarity:
$$
C_{ji}(-\mathbf{h}) = \operatorname{Cov}(Z_j(\mathbf{u}), Z_i(\mathbf{u}+\mathbf{h})) = \operatorname{Cov}(Z_i(\mathbf{u}+\mathbf{h}), Z_j(\mathbf{u})) = C_{ij}(\mathbf{h})
$$
Hence:
$$
C_{ij}(\mathbf{h}) = C_{ji}(-\mathbf{h})
$$
For the full $p \times p$ matrix covariance kernel $\mathbf{C}(\mathbf{h}) = [C_{ij}(\mathbf{h})]_{i,j=1}^p$:
The $(j, i)$-th entry of the transpose matrix is:
$$
[\mathbf{C}(\mathbf{h})']_{ji} = C_{ij}(\mathbf{h}) = C_{ji}(-\mathbf{h}) = [\mathbf{C}(-\mathbf{h})]_{ji}
$$
Therefore:
$$
\mathbf{C}(\mathbf{h})' = \mathbf{C}(-\mathbf{h})
$$

**(c)** Physical manifestation of cross-covariance asymmetry:
Consider an industrial smoke stack emitting sulfur dioxide ($Z_1$) and fine particulates ($Z_2$). Prevailing atmospheric winds blow steadily eastward from west to east along the $x$-axis.
- As the plume travels downwind, chemical reactions transform $Z_1$ into sulfate aerosols ($Z_2$), meaning that particulate concentration $Z_2$ peaks downwind (eastward) of high gas concentrations $Z_1$.
- Let $\mathbf{h} = (h_x, 0)'$ with $h_x > 0$ representing an eastward displacement.
- Then $C_{12}(\mathbf{h}) = \operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s}))$ measures the covariance between gas *eastward* and particulates *westward*.
- Conversely, $C_{12}(-\mathbf{h}) = \operatorname{Cov}(Z_1(\mathbf{s}-\mathbf{h}), Z_2(\mathbf{s}))$ measures the covariance between gas *westward* (upstream) and particulates *eastward* (downstream).
Because physical advection and reaction kinetics operate directionally along the wind vector, the correlation between upstream gas and downstream particulates is substantially higher than between downstream gas and upstream particulates:
$$
C_{12}(-\mathbf{h}) \gg C_{12}(\mathbf{h})
$$
Hence $C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})$. The individual cross-covariance function is asymmetric with respect to origin reflection $\mathbf{h} \mapsto -\mathbf{h}$, although the matrix relation $\mathbf{C}(\mathbf{h})' = \mathbf{C}(-\mathbf{h})$ is strictly preserved. $\blacksquare$

---

### Exercise 7.6: Ordinary Cokriging Lagrangian System Derivation
Let $\{\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), Z_2(\mathbf{s}))'\}$ be a bivariate second-order stationary spatial random field with unknown constant means $\operatorname{E}[Z_1(\mathbf{s})] = \mu_1$ and $\operatorname{E}[Z_2(\mathbf{s})] = \mu_2$.
Suppose we observe variable 1 at $n_1$ locations $\mathbf{s}_1^{(1)}, \dots, \mathbf{s}_{n_1}^{(1)}$ and variable 2 at $n_2$ locations $\mathbf{s}_1^{(2)}, \dots, \mathbf{s}_{n_2}^{(2)}$.
The goal is to predict the primary variable $Z_1(\mathbf{s}_0)$ at an unsampled location $\mathbf{s}_0 \in D$ using a linear cokriging predictor:
$$
\widehat{Z}_1(\mathbf{s}_0) = \sum_{i=1}^{n_1} \lambda_{1,i} Z_1(\mathbf{s}_i^{(1)}) + \sum_{j=1}^{n_2} \lambda_{2,j} Z_2(\mathbf{s}_j^{(2)}) = \boldsymbol{\lambda}_1'\mathbf{Z}_1 + \boldsymbol{\lambda}_2'\mathbf{Z}_2
$$
**(a)** Prove that unbiasedness $\operatorname{E}[\widehat{Z}_1(\mathbf{s}_0)] = \operatorname{E}[Z_1(\mathbf{s}_0)] = \mu_1$ for all unknown means $(\mu_1, \mu_2) \in \mathbb{R}^2$ requires the two simultaneous constraints:
$$
\sum_{i=1}^{n_1} \lambda_{1,i} = \mathbf{1}'\boldsymbol{\lambda}_1 = 1 \quad \text{and} \quad \sum_{j=1}^{n_2} \lambda_{2,j} = \mathbf{1}'\boldsymbol{\lambda}_2 = 0
$$
**(b)** Formulate the Mean Squared Prediction Error $\sigma_{\text{OCK}}^2(\mathbf{s}_0) = \operatorname{E}[(Z_1(\mathbf{s}_0) - \widehat{Z}_1(\mathbf{s}_0))^2]$ in block matrix notation:
$$
\operatorname{Var}(Z_1(\mathbf{s}_0) - \widehat{Z}_1(\mathbf{s}_0)) = C_{11}(\mathbf{0}) - 2 \begin{pmatrix} \boldsymbol{\lambda}_1' & \boldsymbol{\lambda}_2' \end{pmatrix} \begin{pmatrix} \mathbf{c}_{11,0} \\ \mathbf{c}_{21,0} \end{pmatrix} + \begin{pmatrix} \boldsymbol{\lambda}_1' & \boldsymbol{\lambda}_2' \end{pmatrix} \begin{pmatrix} \mathbf{V}_{11} & \mathbf{V}_{12} \\ \mathbf{V}_{21} & \mathbf{V}_{22} \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda}_1 \\ \boldsymbol{\lambda}_2 \end{pmatrix}
$$
**(c)** Introduce two Lagrange multipliers $2m_1$ and $2m_2$ to enforce the constraints, compute the partial gradients, and construct the $(n_1 + n_2 + 2) \times (n_1 + n_2 + 2)$ Ordinary Cokriging block system:
$$
\begin{pmatrix}
\mathbf{V}_{11} & \mathbf{V}_{12} & \mathbf{1}_{n_1} & \mathbf{0} \\
\mathbf{V}_{21} & \mathbf{V}_{22} & \mathbf{0} & \mathbf{1}_{n_2} \\
\mathbf{1}_{n_1}' & \mathbf{0} & 0 & 0 \\
\mathbf{0} & \mathbf{1}_{n_2}' & 0 & 0
\end{pmatrix}
\begin{pmatrix} \boldsymbol{\lambda}_1 \\ \boldsymbol{\lambda}_2 \\ m_1 \\ m_2 \end{pmatrix}
=
\begin{pmatrix} \mathbf{c}_{11,0} \\ \mathbf{c}_{21,0} \\ 1 \\ 0 \end{pmatrix}
$$
**(d)** Derive the Ordinary Cokriging prediction variance $\sigma_{\text{OCK}}^2(\mathbf{s}_0)$ and prove that:
$$
\sigma_{\text{OCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - \boldsymbol{\lambda}_1'\mathbf{c}_{11,0} - \boldsymbol{\lambda}_2'\mathbf{c}_{21,0} - m_1
$$

##### Solution to Exercise 7.6
**(a)** Compute the expectation of the linear predictor:
$$
\begin{aligned}
\operatorname{E}[\widehat{Z}_1(\mathbf{s}_0)] &= \sum_{i=1}^{n_1} \lambda_{1,i} \operatorname{E}[Z_1(\mathbf{s}_i^{(1)})] + \sum_{j=1}^{n_2} \lambda_{2,j} \operatorname{E}[Z_2(\mathbf{s}_j^{(2)})] \\
&= \mu_1 \sum_{i=1}^{n_1} \lambda_{1,i} + \mu_2 \sum_{j=1}^{n_2} \lambda_{2,j}
\end{aligned}
$$
The target expectation is $\operatorname{E}[Z_1(\mathbf{s}_0)] = \mu_1$.
The prediction bias is:
$$
\operatorname{E}[\widehat{Z}_1(\mathbf{s}_0) - Z_1(\mathbf{s}_0)] = \mu_1 \left( \sum_{i=1}^{n_1} \lambda_{1,i} - 1 \right) + \mu_2 \left( \sum_{j=1}^{n_2} \lambda_{2,j} \right)
$$
Because $\mu_1$ and $\mu_2$ are independent, unknown scalar constants, this bias must vanish identically for all $(\mu_1, \mu_2) \in \mathbb{R}^2$.
Setting the coefficient of $\mu_1$ to zero requires:
$$
\sum_{i=1}^{n_1} \lambda_{1,i} = 1 \iff \mathbf{1}'\boldsymbol{\lambda}_1 = 1
$$
Setting the coefficient of $\mu_2$ to zero requires:
$$
\sum_{j=1}^{n_2} \lambda_{2,j} = 0 \iff \mathbf{1}'\boldsymbol{\lambda}_2 = 0
$$

**(b)** Under these unbiasedness constraints:
$$
Z_1(\mathbf{s}_0) - \widehat{Z}_1(\mathbf{s}_0) = (Z_1(\mathbf{s}_0) - \mu_1) - \boldsymbol{\lambda}_1'(\mathbf{Z}_1 - \mu_1 \mathbf{1}) - \boldsymbol{\lambda}_2'(\mathbf{Z}_2 - \mu_2 \mathbf{1})
$$
Let $\boldsymbol{\lambda} = \begin{pmatrix} \boldsymbol{\lambda}_1 \\ \boldsymbol{\lambda}_2 \end{pmatrix}$, $\mathbf{Z} = \begin{pmatrix} \mathbf{Z}_1 \\ \mathbf{Z}_2 \end{pmatrix}$, $\mathbf{V} = \begin{pmatrix} \mathbf{V}_{11} & \mathbf{V}_{12} \\ \mathbf{V}_{21} & \mathbf{V}_{22} \end{pmatrix}$, and $\mathbf{c}_0 = \begin{pmatrix} \mathbf{c}_{11,0} \\ \mathbf{c}_{21,0} \end{pmatrix}$.
The variance of the prediction error is:
$$
\begin{aligned}
\operatorname{Var}(Z_1(\mathbf{s}_0) - \widehat{Z}_1(\mathbf{s}_0)) &= \operatorname{Var}(Z_1(\mathbf{s}_0)) - 2\operatorname{Cov}(Z_1(\mathbf{s}_0), \boldsymbol{\lambda}'\mathbf{Z}) + \operatorname{Var}(\boldsymbol{\lambda}'\mathbf{Z}) \\
&= C_{11}(\mathbf{0}) - 2 \boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} \\
&= C_{11}(\mathbf{0}) - 2(\boldsymbol{\lambda}_1'\mathbf{c}_{11,0} + \boldsymbol{\lambda}_2'\mathbf{c}_{21,0}) + \boldsymbol{\lambda}_1'\mathbf{V}_{11}\boldsymbol{\lambda}_1 + 2\boldsymbol{\lambda}_1'\mathbf{V}_{12}\boldsymbol{\lambda}_2 + \boldsymbol{\lambda}_2'\mathbf{V}_{22}\boldsymbol{\lambda}_2
\end{aligned}
$$

**(c)** Adjoin the two constraints with multipliers $2m_1$ and $2m_2$:
$$
\mathcal{L}(\boldsymbol{\lambda}_1, \boldsymbol{\lambda}_2, m_1, m_2) = C_{11}(\mathbf{0}) - 2\boldsymbol{\lambda}_1'\mathbf{c}_{11,0} - 2\boldsymbol{\lambda}_2'\mathbf{c}_{21,0} + \boldsymbol{\lambda}_1'\mathbf{V}_{11}\boldsymbol{\lambda}_1 + 2\boldsymbol{\lambda}_1'\mathbf{V}_{12}\boldsymbol{\lambda}_2 + \boldsymbol{\lambda}_2'\mathbf{V}_{22}\boldsymbol{\lambda}_2 + 2m_1(\mathbf{1}'\boldsymbol{\lambda}_1 - 1) + 2m_2(\mathbf{1}'\boldsymbol{\lambda}_2)
$$
Differentiating with respect to the parameter vectors:
$$
\frac{1}{2}\frac{\partial \mathcal{L}}{\partial \boldsymbol{\lambda}_1} = -\mathbf{c}_{11,0} + \mathbf{V}_{11}\boldsymbol{\lambda}_1 + \mathbf{V}_{12}\boldsymbol{\lambda}_2 + m_1 \mathbf{1}_{n_1} = \mathbf{0} \implies \mathbf{V}_{11}\boldsymbol{\lambda}_1 + \mathbf{V}_{12}\boldsymbol{\lambda}_2 + m_1 \mathbf{1}_{n_1} = \mathbf{c}_{11,0}
$$
$$
\frac{1}{2}\frac{\partial \mathcal{L}}{\partial \boldsymbol{\lambda}_2} = -\mathbf{c}_{21,0} + \mathbf{V}_{21}\boldsymbol{\lambda}_1 + \mathbf{V}_{22}\boldsymbol{\lambda}_2 + m_2 \mathbf{1}_{n_2} = \mathbf{0} \implies \mathbf{V}_{21}\boldsymbol{\lambda}_1 + \mathbf{V}_{22}\boldsymbol{\lambda}_2 + m_2 \mathbf{1}_{n_2} = \mathbf{c}_{21,0}
$$
$$
\frac{1}{2}\frac{\partial \mathcal{L}}{\partial m_1} = \mathbf{1}_{n_1}'\boldsymbol{\lambda}_1 - 1 = 0 \implies \mathbf{1}_{n_1}'\boldsymbol{\lambda}_1 = 1
$$
$$
\frac{1}{2}\frac{\partial \mathcal{L}}{\partial m_2} = \mathbf{1}_{n_2}'\boldsymbol{\lambda}_2 = 0 \implies \mathbf{1}_{n_2}'\boldsymbol{\lambda}_2 = 0
$$
Assembling these in block matrix form gives the $(n_1 + n_2 + 2) \times (n_1 + n_2 + 2)$ system:
$$
\begin{pmatrix}
\mathbf{V}_{11} & \mathbf{V}_{12} & \mathbf{1}_{n_1} & \mathbf{0} \\
\mathbf{V}_{21} & \mathbf{V}_{22} & \mathbf{0} & \mathbf{1}_{n_2} \\
\mathbf{1}_{n_1}' & \mathbf{0} & 0 & 0 \\
\mathbf{0} & \mathbf{1}_{n_2}' & 0 & 0
\end{pmatrix}
\begin{pmatrix} \boldsymbol{\lambda}_1 \\ \boldsymbol{\lambda}_2 \\ m_1 \\ m_2 \end{pmatrix}
=
\begin{pmatrix} \mathbf{c}_{11,0} \\ \mathbf{c}_{21,0} \\ 1 \\ 0 \end{pmatrix}
$$

**(d)** Multiply the first block equation by $\boldsymbol{\lambda}_1'$ and the second block equation by $\boldsymbol{\lambda}_2'$:
$$
\boldsymbol{\lambda}_1'\mathbf{V}_{11}\boldsymbol{\lambda}_1 + \boldsymbol{\lambda}_1'\mathbf{V}_{12}\boldsymbol{\lambda}_2 + m_1 \boldsymbol{\lambda}_1'\mathbf{1}_{n_1} = \boldsymbol{\lambda}_1'\mathbf{c}_{11,0}
$$
$$
\boldsymbol{\lambda}_2'\mathbf{V}_{21}\boldsymbol{\lambda}_1 + \boldsymbol{\lambda}_2'\mathbf{V}_{22}\boldsymbol{\lambda}_2 + m_2 \boldsymbol{\lambda}_2'\mathbf{1}_{n_2} = \boldsymbol{\lambda}_2'\mathbf{c}_{21,0}
$$
Adding these two equations:
$$
\boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} + m_1(\mathbf{1}'\boldsymbol{\lambda}_1) + m_2(\mathbf{1}'\boldsymbol{\lambda}_2) = \boldsymbol{\lambda}_1'\mathbf{c}_{11,0} + \boldsymbol{\lambda}_2'\mathbf{c}_{21,0}
$$
Since $\mathbf{1}'\boldsymbol{\lambda}_1 = 1$ and $\mathbf{1}'\boldsymbol{\lambda}_2 = 0$:
$$
\boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} + m_1(1) + m_2(0) = \boldsymbol{\lambda}_1'\mathbf{c}_{11,0} + \boldsymbol{\lambda}_2'\mathbf{c}_{21,0} \implies \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} = \boldsymbol{\lambda}_1'\mathbf{c}_{11,0} + \boldsymbol{\lambda}_2'\mathbf{c}_{21,0} - m_1
$$
Substitute $\boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda}$ into the variance equation from part **(b)**:
$$
\begin{aligned}
\sigma_{\text{OCK}}^2(\mathbf{s}_0) &= C_{11}(\mathbf{0}) - 2(\boldsymbol{\lambda}_1'\mathbf{c}_{11,0} + \boldsymbol{\lambda}_2'\mathbf{c}_{21,0}) + (\boldsymbol{\lambda}_1'\mathbf{c}_{11,0} + \boldsymbol{\lambda}_2'\mathbf{c}_{21,0} - m_1) \\
&= C_{11}(\mathbf{0}) - \boldsymbol{\lambda}_1'\mathbf{c}_{11,0} - \boldsymbol{\lambda}_2'\mathbf{c}_{21,0} - m_1
\end{aligned}
$$
Notice that the second multiplier $m_2$ does not appear in the variance expression because $\mathbf{1}'\boldsymbol{\lambda}_2 = 0$. $\blacksquare$

---

### Exercise 7.7: The Autokrigeability Theorem via Kronecker Product Algebra
One of the most profound theoretical results in multivariate geostatistics is the **Autokrigeability Theorem** (Matheron, 1979; Wackernagel, 2003): under isotopic sampling and an intrinsic correlation model, secondary variables provide zero additional information for predicting a primary variable, and Cokriging collapses identically to univariate Ordinary Kriging.
Let $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_p(\mathbf{s}))'$ be a second-order stationary $p$-variate spatial process.
Suppose the spatial sampling is **isotopic**: all $p$ variables are measured at the exact same $n$ locations $\mathbf{s}_1, \dots, \mathbf{s}_n$.
Suppose the matrix covariance function follows the **intrinsic correlation model** (proportional covariance model):
$$
\mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(\mathbf{h})
$$
where $\mathbf{B} \in \mathbb{R}^{p \times p}$ is a symmetric positive definite coregionalization matrix and $\rho(\mathbf{h})$ is a valid scalar correlation function ($\rho(\mathbf{0}) = 1$).
**(a)** Express the $pn \times pn$ data covariance matrix $\mathbf{V}$ as a Kronecker product:
$$
\mathbf{V} = \mathbf{B} \otimes \mathbf{R}
$$
where $\mathbf{R} = [\rho(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}$.
**(b)** Express the $pn \times 1$ target covariance vector $\mathbf{c}_0$ as a Kronecker product:
$$
\mathbf{c}_0 = \mathbf{b}_1 \otimes \mathbf{r}_0
$$
where $\mathbf{b}_1 = (B_{11}, B_{12}, \dots, B_{1p})'$ is the first column of $\mathbf{B}$ and $\mathbf{r}_0 = (\rho(\mathbf{s}_1 - \mathbf{s}_0), \dots, \rho(\mathbf{s}_n - \mathbf{s}_0))'$.
**(c)** For Simple Cokriging, the system is $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$. Using the mixed-product property of Kronecker products:
$$
(\mathbf{A} \otimes \mathbf{C})(\mathbf{B} \otimes \mathbf{D}) = (\mathbf{A}\mathbf{B}) \otimes (\mathbf{C}\mathbf{D})
$$
and $(\mathbf{B} \otimes \mathbf{R})^{-1} = \mathbf{B}^{-1} \otimes \mathbf{R}^{-1}$, solve analytically for the Cokriging weight vector $\boldsymbol{\lambda} \in \mathbb{R}^{pn}$.
**(d)** Prove that the secondary variable weights are identically zero:
$$
\boldsymbol{\lambda}_1 = \mathbf{R}^{-1}\mathbf{r}_0, \quad \boldsymbol{\lambda}_2 = \mathbf{0}, \quad \dots, \quad \boldsymbol{\lambda}_p = \mathbf{0}
$$
thereby proving the Autokrigeability Theorem.

##### Solution to Exercise 7.7
**(a)** Ordering observations by variable then location ($\mathbf{Z} = (\mathbf{Z}_1', \mathbf{Z}_2', \dots, \mathbf{Z}_p')'$):
The block covariance between $\mathbf{Z}_i$ and $\mathbf{Z}_j$ is:
$$
\mathbf{V}_{ij} = [C_{ij}(\mathbf{s}_k - \mathbf{s}_l)]_{k,l=1}^n = [B_{ij} \rho(\mathbf{s}_k - \mathbf{s}_l)]_{k,l=1}^n = B_{ij} \mathbf{R}
$$
Therefore, the full data covariance matrix is:
$$
\mathbf{V} = \begin{pmatrix}
B_{11}\mathbf{R} & B_{12}\mathbf{R} & \cdots & B_{1p}\mathbf{R} \\
B_{21}\mathbf{R} & B_{22}\mathbf{R} & \cdots & B_{2p}\mathbf{R} \\
\vdots & \vdots & \ddots & \vdots \\
B_{p1}\mathbf{R} & B_{p2}\mathbf{R} & \cdots & B_{pp}\mathbf{R}
\end{pmatrix}
= \mathbf{B} \otimes \mathbf{R}
$$

**(b)** The target covariance vector between $\mathbf{Z}$ and $Z_1(\mathbf{s}_0)$ is partitioned into $p$ sub-vectors:
$$
\mathbf{c}_{0, i} = \operatorname{Cov}(\mathbf{Z}_i, Z_1(\mathbf{s}_0)) = [C_{i1}(\mathbf{s}_k - \mathbf{s}_0)]_{k=1}^n = [B_{i1} \rho(\mathbf{s}_k - \mathbf{s}_0)]_{k=1}^n = B_{i1} \mathbf{r}_0
$$
Since $\mathbf{B}$ is symmetric, $B_{i1} = B_{1i}$.
Thus:
$$
\mathbf{c}_0 = \begin{pmatrix} B_{11}\mathbf{r}_0 \\ B_{21}\mathbf{r}_0 \\ \vdots \\ B_{p1}\mathbf{r}_0 \end{pmatrix} = \begin{pmatrix} B_{11} \\ B_{21} \\ \vdots \\ B_{p1} \end{pmatrix} \otimes \mathbf{r}_0 = \mathbf{b}_1 \otimes \mathbf{r}_0
$$

**(c)** The Simple Cokriging normal equations are:
$$
\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0 \implies (\mathbf{B} \otimes \mathbf{R}) \boldsymbol{\lambda} = \mathbf{b}_1 \otimes \mathbf{r}_0
$$
Because $\mathbf{B}$ and $\mathbf{R}$ are invertible, the inverse of their Kronecker product is:
$$
\mathbf{V}^{-1} = (\mathbf{B} \otimes \mathbf{R})^{-1} = \mathbf{B}^{-1} \otimes \mathbf{R}^{-1}
$$
Multiplying $\mathbf{V}^{-1}$ by $\mathbf{c}_0$:
$$
\boldsymbol{\lambda} = (\mathbf{B}^{-1} \otimes \mathbf{R}^{-1}) (\mathbf{b}_1 \otimes \mathbf{r}_0)
$$
Apply the mixed-product property $(\mathbf{A} \otimes \mathbf{C})(\mathbf{B} \otimes \mathbf{D}) = (\mathbf{A}\mathbf{B}) \otimes (\mathbf{C}\mathbf{D})$:
$$
\boldsymbol{\lambda} = (\mathbf{B}^{-1} \mathbf{b}_1) \otimes (\mathbf{R}^{-1} \mathbf{r}_0)
$$

**(d)** Now evaluate $\mathbf{B}^{-1} \mathbf{b}_1$:
Notice that $\mathbf{b}_1$ is the *first column* of matrix $\mathbf{B}$.
Therefore:
$$
\mathbf{b}_1 = \mathbf{B} \mathbf{e}_1
$$
where $\mathbf{e}_1 = (1, 0, \dots, 0)' \in \mathbb{R}^p$ is the first elementary basis vector.
Multiplying by $\mathbf{B}^{-1}$:
$$
\mathbf{B}^{-1} \mathbf{b}_1 = \mathbf{B}^{-1} (\mathbf{B} \mathbf{e}_1) = (\mathbf{B}^{-1} \mathbf{B}) \mathbf{e}_1 = \mathbf{I}_p \mathbf{e}_1 = \mathbf{e}_1
$$
Substituting this back into the expression for $\boldsymbol{\lambda}$:
$$
\boldsymbol{\lambda} = \mathbf{e}_1 \otimes (\mathbf{R}^{-1} \mathbf{r}_0) = \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix} \otimes (\mathbf{R}^{-1}\mathbf{r}_0) = \begin{pmatrix} \mathbf{R}^{-1}\mathbf{r}_0 \\ \mathbf{0} \\ \vdots \\ \mathbf{0} \end{pmatrix}
$$
Partitioning $\boldsymbol{\lambda} = (\boldsymbol{\lambda}_1', \boldsymbol{\lambda}_2', \dots, \boldsymbol{\lambda}_p')'$:
$$
\boldsymbol{\lambda}_1 = \mathbf{R}^{-1}\mathbf{r}_0 = \boldsymbol{\lambda}_{\text{SK}}
$$
$$
\boldsymbol{\lambda}_2 = \mathbf{0}, \quad \boldsymbol{\lambda}_3 = \mathbf{0}, \quad \dots, \quad \boldsymbol{\lambda}_p = \mathbf{0}
$$
Every secondary variable receives a weight vector identically equal to zero!
The Simple Cokriging predictor collapses exactly to univariate Simple Kriging:
$$
\widehat{Z}_1(\mathbf{s}_0) = \boldsymbol{\lambda}_1'\mathbf{Z}_1 + \sum_{k=2}^p \mathbf{0}'\mathbf{Z}_k = \boldsymbol{\lambda}_{\text{SK}}'\mathbf{Z}_1
$$
This establishes the Autokrigeability Theorem. (Note: for Ordinary Cokriging, the exact same result holds when the means are unknown, as the Lagrange multipliers for secondary variables $m_2 = \dots = m_p = 0$). $\blacksquare$

---

### Exercise 7.8: Markov Models MM1 and MM2 in Collocated Cokriging
When secondary data are exhaustively available across a dense raster (e.g., satellite imagery or seismic acoustic impedance), full cokriging requires inverting a massive system of size $n_1 + n_2 \approx 10^6$. **Collocated Cokriging (CCK)** retains only the secondary observation located directly at the target location $\mathbf{s}_0$: $Z_2(\mathbf{s}_0)$.
To avoid modeling the cross-variogram, André Journel proposed the Markov Model 1 (MM1):
$$
\operatorname{E}[Z_1(\mathbf{s}) \mid Z_2(\mathbf{s}) = z_2, Z_2(\mathbf{s}+\mathbf{h}) = z_2'] = \operatorname{E}[Z_1(\mathbf{s}) \mid Z_2(\mathbf{s}) = z_2]
$$
**(a)** Prove that under MM1, the cross-covariance is proportional to the secondary auto-covariance:
$$
C_{12}(\mathbf{h}) = \frac{C_{12}(\mathbf{0})}{C_{22}(\mathbf{0})} C_{22}(\mathbf{h}) = \rho_{12}(\mathbf{0}) \sqrt{\frac{C_{11}(\mathbf{0})}{C_{22}(\mathbf{0})}} C_{22}(\mathbf{h})
$$
**(b)** In Markov Model 2 (MM2), the assumption is reversed: $Z_2(\mathbf{s})$ is assumed to be a filtered/degraded version of $Z_1(\mathbf{s})$, yielding:
$$
C_{12}(\mathbf{h}) = \frac{C_{12}(\mathbf{0})}{C_{11}(\mathbf{0})} C_{11}(\mathbf{h})
$$
Under what physical conditions is MM2 superior to MM1?
**(c)** What is the **variance screening paradox** in Collocated Cokriging? Why does omitting secondary data at sample locations $\mathbf{s}_i \ne \mathbf{s}_0$ cause Collocated Cokriging to underestimate the true prediction error variance?

##### Solution to Exercise 7.8
**(a)** Proof of MM1 Cross-Covariance Formula:
Assume without loss of generality that variables are standardized to mean zero and unit variance: $\operatorname{E}[Z_1] = \operatorname{E}[Z_2] = 0$, $C_{11}(0) = C_{22}(0) = 1$. Let $\rho_0 = \rho_{12}(\mathbf{0}) = \operatorname{Cov}(Z_1(\mathbf{s}), Z_2(\mathbf{s}))$.
Under MM1, the regression of $Z_1(\mathbf{s})$ on the collocated secondary variable $Z_2(\mathbf{s})$ satisfies:
$$
Z_1(\mathbf{s}) = \rho_0 Z_2(\mathbf{s}) + \eta(\mathbf{s})
$$
where the residual $\eta(\mathbf{s})$ is conditionally independent of $Z_2(\mathbf{s}+\mathbf{h})$ for all $\mathbf{h} \in \mathbb{R}^d$: $\operatorname{Cov}(\eta(\mathbf{s}), Z_2(\mathbf{s}+\mathbf{h})) = 0$.
Now compute the cross-covariance $C_{12}(\mathbf{h})$:
$$
\begin{aligned}
C_{12}(\mathbf{h}) &= \operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s})) \\
&= \operatorname{Cov}(\rho_0 Z_2(\mathbf{s}+\mathbf{h}) + \eta(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s})) \\
&= \rho_0 \operatorname{Cov}(Z_2(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s})) + \operatorname{Cov}(\eta(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s})) \\
&= \rho_0 C_{22}(\mathbf{h}) + 0 \\
&= \rho_{12}(\mathbf{0}) C_{22}(\mathbf{h})
\end{aligned}
$$
Restoring unstandardized variances $C_{11}(\mathbf{0})$ and $C_{22}(\mathbf{0})$:
$$
C_{12}(\mathbf{h}) = \frac{C_{12}(\mathbf{0})}{C_{22}(\mathbf{0})} C_{22}(\mathbf{h})
$$

**(b)** Physical Context for MM2:
Under MM2, $C_{12}(\mathbf{h}) \propto C_{11}(\mathbf{h})$, meaning the cross-covariance reflects the spatial continuity of the *primary variable* rather than the secondary.
- MM1 assumes $Z_1$ is a noisy measurement of $Z_2$. This fails when the primary variable $Z_1$ is much smoother than the secondary variable $Z_2$ (e.g., $Z_1$ is elevation or hydraulic head, while $Z_2$ is high-frequency radar reflectivity or noisy surface roughness).
- MM2 is appropriate when the secondary variable $Z_2$ represents an indirect, blurred, or lower-resolution physical footprint of $Z_1$ (e.g., $Z_1$ is true subsurface permeability, and $Z_2$ is a smoothed, volume-averaged seismic velocity attribute). Under MM2, the cross-structure correctly inherits the smoother range of $Z_1$.

**(c)** The Variance Screening Paradox in Collocated Cokriging:
In Collocated Cokriging, the system uses only $\{Z_1(\mathbf{s}_1), \dots, Z_1(\mathbf{s}_n)\}$ and the single collocated point $Z_2(\mathbf{s}_0)$. All other secondary observations $\{Z_2(\mathbf{s}_1), \dots, Z_2(\mathbf{s}_n)\}$ are discarded.
- In reality, if $Z_2$ is known everywhere, observing $Z_1(\mathbf{s}_i)$ and $Z_2(\mathbf{s}_i)$ at the same point allows the filter to separate the common signal from the specific residual $\eta(\mathbf{s}_i)$.
- When we retain only $Z_2(\mathbf{s}_0)$ and ignore the other secondary points, the system falsely treats $Z_2(\mathbf{s}_0)$ as if it were conditionally independent of the primary data errors.
- As a result, the collocated secondary variable $Z_2(\mathbf{s}_0)$ receives an artificially inflated weight $\lambda_2(\mathbf{s}_0)$, and the reported cokriging variance $\sigma_{\text{CCK}}^2(\mathbf{s}_0)$ is systematically *underestimated* relative to full cokriging. Furthermore, as sample points $\mathbf{s}_i$ get close to $\mathbf{s}_0$, the weights can destabilize into violent positive/negative oscillations. $\blacksquare$

---

## Tier II: Pencil-and-Paper Mechanics

### Exercise 7.9: Pencil-and-Paper 2-Variable, 3-Point Cokriging System
Consider a 1D spatial domain with two variables $Z_1$ and $Z_2$.
Observations are taken heterotopically:
- Variable 1 (primary) is observed at one location: $s_1 = 0.0$ with $Z_1(0) = 10.0$.
- Variable 2 (secondary) is observed at two locations: $s_1 = 0.0$ with $Z_2(0) = 5.0$, and $s_2 = 1.0$ with $Z_2(1) = 8.0$.
We wish to predict $Z_1(s_0)$ at $s_0 = 0.5$.
The bivariate random field follows the intrinsic correlation model $\mathbf{C}(h) = \mathbf{B} e^{-|h|}$ with:
$$
\mathbf{B} = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix} = \begin{pmatrix} 4.0 & 2.0 \\ 2.0 & 1.0 \end{pmatrix}^*
$$
*(Note: to ensure strict positive definiteness, let $B_{22} = 1.25$, so $\det(\mathbf{B}) = 4(1.25) - 2^2 = 5 - 4 = 1.0 > 0$.)*
**(a)** Compute all elements of the covariance matrices $\mathbf{V}_{11} \in \mathbb{R}^{1 \times 1}$, $\mathbf{V}_{12} \in \mathbb{R}^{1 \times 2}$, $\mathbf{V}_{22} \in \mathbb{R}^{2 \times 2}$ and target vectors $\mathbf{c}_{11,0} \in \mathbb{R}^1$, $\mathbf{c}_{21,0} \in \mathbb{R}^2$.
**(b)** Formulate the $5 \times 5$ Ordinary Cokriging system:
$$
\begin{pmatrix}
V_{11} & V_{12}^{(1)} & V_{12}^{(2)} & 1 & 0 \\
V_{21}^{(1)} & V_{22}^{(11)} & V_{22}^{(12)} & 0 & 1 \\
V_{21}^{(2)} & V_{22}^{(21)} & V_{22}^{(22)} & 0 & 1 \\
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0
\end{pmatrix}
\begin{pmatrix} \lambda_{1,1} \\ \lambda_{2,1} \\ \lambda_{2,2} \\ m_1 \\ m_2 \end{pmatrix}
=
\begin{pmatrix} c_{11,0} \\ c_{21,0}^{(1)} \\ c_{21,0}^{(2)} \\ 1 \\ 0 \end{pmatrix}
$$
**(c)** Solve the system by hand to determine all weights and Lagrange multipliers.
**(d)** Compute the point prediction $\widehat{Z}_1(0.5)$ and prediction variance $\sigma_{\text{OCK}}^2(0.5)$.

##### Solution to Exercise 7.9
**(a)** Spatial distances:
- Primary sample: $s_1^{(1)} = 0$.
- Secondary samples: $s_1^{(2)} = 0, s_2^{(2)} = 1$.
- Target: $s_0 = 0.5$.
- Distance $|s_1^{(1)} - s_0| = 0.5$.
- Distance $|s_1^{(2)} - s_0| = 0.5$, $|s_2^{(2)} - s_0| = 0.5$.
- Distance $|s_1^{(2)} - s_2^{(2)}| = 1.0$.
Values of $\rho(h) = e^{-|h|}$:
$\rho(0) = 1$, $\rho(0.5) = e^{-0.5} \approx 0.6065$, $\rho(1.0) = e^{-1} \approx 0.3679$.

Matrix components ($\mathbf{C}(h) = \mathbf{B}\rho(h)$ with $B_{11}=4.0, B_{12}=2.0, B_{22}=1.25$):
- $\mathbf{V}_{11} = [B_{11} \rho(0)] = [4.0]$.
- $\mathbf{V}_{12} = [B_{12} \rho(s_1^{(1)} - s_1^{(2)}), B_{12} \rho(s_1^{(1)} - s_2^{(2)})] = [2.0 \rho(0), 2.0 \rho(1)] = [2.0, 2.0 e^{-1}] \approx [2.0000, 0.7358]$.
- $\mathbf{V}_{21} = \mathbf{V}_{12}' = \begin{pmatrix} 2.0000 \\ 0.7358 \end{pmatrix}$.
- $\mathbf{V}_{22} = B_{22} \begin{pmatrix} \rho(0) & \rho(1) \\ \rho(1) & \rho(0) \end{pmatrix} = 1.25 \begin{pmatrix} 1.0 & e^{-1} \\ e^{-1} & 1.0 \end{pmatrix} \approx \begin{pmatrix} 1.2500 & 0.4598 \\ 0.4598 & 1.2500 \end{pmatrix}$.
Target covariance vectors:
- $\mathbf{c}_{11,0} = [B_{11} \rho(0.5)] = [4.0 e^{-0.5}] \approx [2.4261]$.
- $\mathbf{c}_{21,0} = \begin{pmatrix} B_{21} \rho(0.5) \\ B_{21} \rho(0.5) \end{pmatrix} = \begin{pmatrix} 2.0 e^{-0.5} \\ 2.0 e^{-0.5} \end{pmatrix} \approx \begin{pmatrix} 1.2131 \\ 1.2131 \end{pmatrix}$.

**(b)** The $5 \times 5$ Ordinary Cokriging block system is:
$$
\begin{pmatrix}
4.0000 & 2.0000 & 0.7358 & 1.0 & 0.0 \\
2.0000 & 1.2500 & 0.4598 & 0.0 & 1.0 \\
0.7358 & 0.4598 & 1.2500 & 0.0 & 1.0 \\
1.0000 & 0.0000 & 0.0000 & 0.0 & 0.0 \\
0.0000 & 1.0000 & 1.0000 & 0.0 & 0.0
\end{pmatrix}
\begin{pmatrix} \lambda_{1,1} \\ \lambda_{2,1} \\ \lambda_{2,2} \\ m_1 \\ m_2 \end{pmatrix}
=
\begin{pmatrix} 2.4261 \\ 1.2131 \\ 1.2131 \\ 1.0000 \\ 0.0000 \end{pmatrix}
$$

**(c)** Solving by hand:
From row 4 (unbiasedness on primary variable):
$$
1.0 \cdot \lambda_{1,1} = 1.0 \implies \lambda_{1,1} = 1.0000
$$
From row 5 (unbiasedness on secondary variable):
$$
\lambda_{2,1} + \lambda_{2,2} = 0 \implies \lambda_{2,2} = -\lambda_{2,1}
$$
Now substitute $\lambda_{1,1} = 1$ and $\lambda_{2,2} = -\lambda_{2,1}$ into rows 2 and 3:
Row 2:
$$
2.0(1) + 1.25 \lambda_{2,1} + 0.4598(-\lambda_{2,1}) + m_2 = 1.2131
$$
$$
2.0 + (1.25 - 0.4598)\lambda_{2,1} + m_2 = 1.2131 \implies 0.7902 \lambda_{2,1} + m_2 = -0.7869 \quad \text{(Eq A)}
$$
Row 3:
$$
0.7358(1) + 0.4598 \lambda_{2,1} + 1.25(-\lambda_{2,1}) + m_2 = 1.2131
$$
$$
0.7358 - (1.25 - 0.4598)\lambda_{2,1} + m_2 = 1.2131 \implies -0.7902 \lambda_{2,1} + m_2 = 0.4773 \quad \text{(Eq B)}
$$
Subtract (Eq B) from (Eq A):
$$
(0.7902 - (-0.7902))\lambda_{2,1} = -0.7869 - 0.4773
$$
$$
1.5804 \lambda_{2,1} = -1.2642 \implies \lambda_{2,1} = \frac{-1.2642}{1.5804} \approx -0.8000
$$
Then:
$$
\lambda_{2,2} = -\lambda_{2,1} \approx 0.8000
$$
Now find $m_2$ using (Eq A):
$$
0.7902(-0.8000) + m_2 = -0.7869 \implies -0.6322 + m_2 = -0.7869 \implies m_2 \approx -0.1547
$$
Finally, find $m_1$ using Row 1:
$$
4.0(1) + 2.0(-0.8) + 0.7358(0.8) + m_1 = 2.4261
$$
$$
4.0 - 1.6 + 0.5886 + m_1 = 2.4261 \implies 2.9886 + m_1 = 2.4261 \implies m_1 \approx -0.5625
$$

**(d)** Point prediction and variance:
Observations: $Z_1(0) = 10.0$, $Z_2(0) = 5.0$, $Z_2(1) = 8.0$.
$$
\begin{aligned}
\widehat{Z}_1(0.5) &= \lambda_{1,1} Z_1(0) + \lambda_{2,1} Z_2(0) + \lambda_{2,2} Z_2(1) \\
&= 1.0(10.0) + (-0.8000)(5.0) + (0.8000)(8.0) \\
&= 10.0 - 4.0 + 6.4 = 12.4000
\end{aligned}
$$
Notice the physical elegance: between $s=0$ and $s=1$, the secondary variable climbed from $5.0$ to $8.0$ (a rise of $+3.0$).
Cokriging uses the local gradient of the secondary variable to adjust the primary prediction:
$$
\widehat{Z}_1(0.5) = Z_1(0) + 0.8(Z_2(1) - Z_2(0)) = 10.0 + 0.8(3.0) = 12.4
$$
Prediction variance:
$$
\begin{aligned}
\sigma_{\text{OCK}}^2(0.5) &= C_{11}(0) - \boldsymbol{\lambda}_1'\mathbf{c}_{11,0} - \boldsymbol{\lambda}_2'\mathbf{c}_{21,0} - m_1 \\
&= 4.0 - (1.0)(2.4261) - [(-0.8)(1.2131) + (0.8)(1.2131)] - (-0.5625) \\
&= 4.0 - 2.4261 - 0.0 + 0.5625 = 2.1364
\end{aligned}
$$
Notice that $\boldsymbol{\lambda}_2'\mathbf{c}_{21,0} = 0$ because $\lambda_{2,1} = -\lambda_{2,2}$ and $c_{21,0}^{(1)} = c_{21,0}^{(2)}$.
Without the secondary variable, univariate Ordinary Kriging with a single observation $Z_1(0)$ would have given $\widehat{Z}_1(0.5) = 10.0$ and variance $C_{11}(0) - c_{11,0} - m_1 = 4.0 - 2.4261 + 1.5739 = 3.1478$.
Cokriging reduced the prediction variance from $3.15$ down to $2.14$ by incorporating the secondary data slope! $\blacksquare$

---

### Exercise 7.10: The Collocated Cokriging Weight Redistribution
Suppose a primary variable $Z_1$ is observed at a single spatial location $s_1 = 1$, and an auxiliary secondary variable $Z_2$ is observed collocated with the target at $s_0 = 0$.
The variables are standardized ($\sigma_1^2 = \sigma_2^2 = 1$), with primary auto-correlation $\rho_1(h) = e^{-|h|}$ and collocated cross-correlation $\rho = \operatorname{Corr}(Z_1(s), Z_2(s)) \in (-1, 1)$. Under MM1, the cross-correlation function is $C_{12}(h) = \rho e^{-|h|}$.
Consider Simple Cokriging with known mean zero:
$$
\widehat{Z}_1(0) = \lambda_1 Z_1(1) + \lambda_2 Z_2(0)
$$
**(a)** Formulate the $2 \times 2$ Simple Cokriging system:
$$
\begin{pmatrix} C_{11}(0) & C_{12}(1) \\ C_{21}(-1) & C_{22}(0) \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} C_{11}(1) \\ C_{12}(0) \end{pmatrix}
$$
**(b)** Substitute $C_{11}(0) = C_{22}(0) = 1$, $C_{11}(1) = e^{-1}$, $C_{12}(0) = \rho$, and $C_{12}(1) = C_{21}(-1) = \rho e^{-1}$ to write the matrix system.
**(c)** Solve analytically for $\lambda_1$ and $\lambda_2$ as explicit functions of $\rho$.
**(d)** Analyze the behavior of $\lambda_1$ and $\lambda_2$ in the two limits $\rho \to 0$ and $\rho \to 1$.

##### Solution to Exercise 7.10
**(a)** The covariance between the two predictors $Z_1(1)$ and $Z_2(0)$ is:
- $\operatorname{Var}(Z_1(1)) = C_{11}(0) = 1$.
- $\operatorname{Var}(Z_2(0)) = C_{22}(0) = 1$.
- $\operatorname{Cov}(Z_1(1), Z_2(0)) = C_{12}(1 - 0) = C_{12}(1) = \rho e^{-1}$.
- The target is $Z_1(0)$:
  - $\operatorname{Cov}(Z_1(1), Z_1(0)) = C_{11}(1) = e^{-1}$.
  - $\operatorname{Cov}(Z_2(0), Z_1(0)) = C_{21}(0) = C_{12}(0) = \rho$.

**(b)** The system is:
$$
\begin{pmatrix} 1 & \rho e^{-1} \\ \rho e^{-1} & 1 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} e^{-1} \\ \rho \end{pmatrix}
$$

**(c)** The determinant of the matrix is:
$$
\det = 1 - (\rho e^{-1})^2 = 1 - \rho^2 e^{-2}
$$
Since $|\rho| < 1$ and $e^{-2} \approx 0.1353$, $\det > 0$.
The inverse matrix is:
$$
\begin{pmatrix} 1 & \rho e^{-1} \\ \rho e^{-1} & 1 \end{pmatrix}^{-1} = \frac{1}{1 - \rho^2 e^{-2}} \begin{pmatrix} 1 & -\rho e^{-1} \\ -\rho e^{-1} & 1 \end{pmatrix}
$$
Multiplying by the target vector $\begin{pmatrix} e^{-1} \\ \rho \end{pmatrix}$:
$$
\lambda_1 = \frac{e^{-1} - (\rho e^{-1})\rho}{1 - \rho^2 e^{-2}} = \frac{e^{-1}(1 - \rho^2)}{1 - \rho^2 e^{-2}}
$$
$$
\lambda_2 = \frac{-(\rho e^{-1})e^{-1} + \rho}{1 - \rho^2 e^{-2}} = \frac{\rho(1 - e^{-2})}{1 - \rho^2 e^{-2}}
$$

**(d)** Limiting behavior:
- **Case 1: No cross-correlation ($\rho \to 0$):**
  $$
  \lambda_1 \to \frac{e^{-1}(1 - 0)}{1 - 0} = e^{-1} \approx 0.3679 = \lambda_{\text{SK}}
  $$
  $$
  \lambda_2 \to \frac{0(1 - e^{-2})}{1 - 0} = 0
  $$
  As expected, the secondary variable receives zero weight, and the primary weight collapses to univariate Simple Kriging.
- **Case 2: Perfect cross-correlation ($\rho \to 1$):**
  $$
  \lambda_1 \to \frac{e^{-1}(1 - 1)}{1 - e^{-2}} = 0
  $$
  $$
  \lambda_2 \to \frac{1(1 - e^{-2})}{1 - e^{-2}} = 1
  $$
  When the collocated secondary variable is perfectly correlated with the target, it receives $100\%$ of the weight ($\lambda_2 = 1$), completely screening out the spatial observation $Z_1(1)$ ($\lambda_1 = 0$). The collocated proxy acts as an exact surrogate for the target. $\blacksquare$

---

### Exercise 7.11: Asymmetric Cross-Covariance and Directional Shift
Consider an advective transport process where an upstream chemical indicator $Z_1$ drifts along a river and converts into a secondary metabolite $Z_2$ at a downstream displacement of $\delta = 2.0$ km.
The cross-covariance function is modeled as a shifted exponential:
$$
C_{12}(h) = \rho \sigma_1 \sigma_2 \exp\left(-\frac{|h - 2.0|}{\phi}\right)
$$
with sill $\sigma_1^2 = \sigma_2^2 = 1.0$, correlation strength $\rho = 0.8$, and spatial range $\phi = 2.0$.
**(a)** Compute $C_{12}(+2.0)$ and $C_{12}(-2.0)$. Show that $C_{12}(+2.0) \ne C_{12}(-2.0)$.
**(b)** What is $C_{21}(+2.0)$ and $C_{21}(-2.0)$ using the lag reflection identity $C_{21}(h) = C_{12}(-h)$?
**(c)** Explain why a standard cross-variogram $\gamma_{12}(h) = \frac{1}{2}\operatorname{E}[(Z_1(s+h) - Z_1(s))(Z_2(s+h) - Z_2(s))]$ fails to capture this directional lag shift $\delta = 2.0$, and why the pseudo-cross-variogram must be used instead.

##### Solution to Exercise 7.11
**(a)** Evaluating $C_{12}(h) = 0.8 \exp(-|h - 2.0| / 2.0)$:
- At $h = +2.0$:
  $$
  C_{12}(+2.0) = 0.8 \exp\left(-\frac{|2.0 - 2.0|}{2.0}\right) = 0.8 e^0 = 0.8000
  $$
- At $h = -2.0$:
  $$
  C_{12}(-2.0) = 0.8 \exp\left(-\frac{|-2.0 - 2.0|}{2.0}\right) = 0.8 \exp\left(-\frac{4.0}{2.0}\right) = 0.8 e^{-2} \approx 0.8(0.1353) \approx 0.1083
  $$
Clearly, $C_{12}(+2.0) = 0.8000 \ne 0.1083 = C_{12}(-2.0)$.
The maximum cross-correlation does not occur at lag zero, but at lag $h = +2.0$, reflecting the physical advection time of the river plume.

**(b)** Using the fundamental transpose identity $C_{21}(h) = C_{12}(-h)$:
- At $h = +2.0$:
  $$
  C_{21}(+2.0) = C_{12}(-2.0) \approx 0.1083
  $$
- At $h = -2.0$:
  $$
  C_{21}(-2.0) = C_{12}(+2.0) = 0.8000
  $$

**(c)** Failure of the classical cross-variogram:
Recall the definition of the classical cross-variogram:
$$
\gamma_{12}(h) = \frac{1}{2} \operatorname{E}[(Z_1(s+h) - Z_1(s))(Z_2(s+h) - Z_2(s))]
$$
Expanding the product:
$$
\gamma_{12}(h) = C_{12}(0) - \frac{1}{2}[C_{12}(h) + C_{12}(-h)]
$$
Notice that $\gamma_{12}(h)$ depends *only on the symmetric sum* $C_{12}(h) + C_{12}(-h)$!
Because the cross-variogram averages $C_{12}(h)$ and $C_{12}(-h)$, it is strictly symmetric: $\gamma_{12}(h) = \gamma_{12}(-h)$.
It completely averages out and obscures the directional shift $\delta = 2.0$!
To preserve the asymmetric shift, one must use the **pseudo-cross-variogram**:
$$
g_{12}(h) = \frac{1}{2} \operatorname{Var}(Z_1(s+h) - Z_2(s)) = \frac{1}{2}[C_{11}(0) + C_{22}(0)] - C_{12}(h)
$$
Because $g_{12}(h)$ directly isolates $C_{12}(h)$ without averaging it with $C_{12}(-h)$, it directly identifies the lag $h = 2.0$ where $g_{12}(h)$ reaches its absolute minimum. $\blacksquare$

---

## Tier III: Algorithmic Translation & Labs in Pure Python

### Exercise 7.12: Pure Python Multivariate Block Matrix Cokriging Engine
Implement an Ordinary Cokriging engine in pure Python (standard library only) that supports heterotopic sampling of two variables.
**(a)** Implement a Gauss-Jordan elimination solver with partial pivoting.
**(b)** Implement `ordinary_cokriging_predict(coords_1, z1_data, coords_2, z2_data, target_coord, B_mat, phi)`:
- `coords_1`: list of $(x, y)$ tuples for variable 1 ($n_1$ points).
- `z1_data`: list of observed values for variable 1.
- `coords_2`: list of $(x, y)$ tuples for variable 2 ($n_2$ points).
- `z2_data`: list of observed values for variable 2.
- `target_coord`: $(x_0, y_0)$ target location.
- `B_mat`: $2 \times 2$ list representing the coregionalization matrix $\mathbf{B}$.
- `phi`: spatial correlation range $\phi$ for $\rho(h) = e^{-h/\phi}$.
**(c)** The solver must construct the full $(n_1 + n_2 + 2) \times (n_1 + n_2 + 2)$ augmented matrix, solve for primary weights $\boldsymbol{\lambda}_1$, secondary weights $\boldsymbol{\lambda}_2$, multipliers $m_1, m_2$, and return:
1. Point prediction $\widehat{Z}_1(\mathbf{s}_0)$.
2. Cokriging prediction variance $\sigma_{\text{OCK}}^2(\mathbf{s}_0)$.
3. Weight vectors $\boldsymbol{\lambda}_1$ and $\boldsymbol{\lambda}_2$.
**(d)** Test your code on the 3-point dataset from Exercise 7.9 and verify that the outputs match the pencil-and-paper derivations.

##### Solution to Exercise 7.12
Here is the complete, self-contained Python script:

```python
import math

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve_linear_system(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        max_row = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                max_row = r
        if max_val < 1e-12:
            raise ValueError(f"Singular matrix encountered at pivot {i}.")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for c in range(i, n + 1):
            M[i][c] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(i, n + 1):
                    M[r][c] -= factor * M[i][c]
    return [M[r][n] for r in range(n)]

def ordinary_cokriging_predict(coords_1, z1_data, coords_2, z2_data, target_coord, B_mat, phi):
    """
    Ordinary Cokriging predictor for bivariate heterotopic data.
    """
    n1 = len(z1_data)
    n2 = len(z2_data)
    dim = n1 + n2 + 2
    
    A = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    
    # Block V11 (n1 x n1)
    for i in range(n1):
        for j in range(n1):
            h = euclidean_dist(coords_1[i], coords_1[j])
            A[i][j] = B_mat[0][0] * math.exp(-h / phi)
            
    # Block V12 (n1 x n2) and V21 (n2 x n1)
    for i in range(n1):
        for j in range(n2):
            h = euclidean_dist(coords_1[i], coords_2[j])
            cov = B_mat[0][1] * math.exp(-h / phi)
            A[i][n1 + j] = cov
            A[n1 + j][i] = cov
            
    # Block V22 (n2 x n2)
    for i in range(n2):
        for j in range(n2):
            h = euclidean_dist(coords_2[i], coords_2[j])
            A[n1 + i][n1 + j] = B_mat[1][1] * math.exp(-h / phi)
            
    # Constraints:
    # 1' lambda1 = 1
    for i in range(n1):
        A[i][n1 + n2] = 1.0
        A[n1 + n2][i] = 1.0
        
    # 1' lambda2 = 0
    for j in range(n2):
        A[n1 + j][n1 + n2 + 1] = 1.0
        A[n1 + n2 + 1][n1 + j] = 1.0
        
    # Target covariance vector c0
    c11_0 = [0.0] * n1
    for i in range(n1):
        h = euclidean_dist(coords_1[i], target_coord)
        cov = B_mat[0][0] * math.exp(-h / phi)
        c11_0[i] = cov
        b[i] = cov
        
    c21_0 = [0.0] * n2
    for j in range(n2):
        h = euclidean_dist(coords_2[j], target_coord)
        cov = B_mat[1][0] * math.exp(-h / phi)
        c21_0[j] = cov
        b[n1 + j] = cov
        
    b[n1 + n2] = 1.0
    b[n1 + n2 + 1] = 0.0
    
    # Solve system
    sol = solve_linear_system(A, b)
    w1 = sol[:n1]
    w2 = sol[n1:n1 + n2]
    m1 = sol[n1 + n2]
    m2 = sol[n1 + n2 + 1]
    
    # Point prediction
    z1_hat = sum(w * z for w, z in zip(w1, z1_data)) + sum(w * z for w, z in zip(w2, z2_data))
    
    # Prediction variance
    c11_zero = B_mat[0][0]
    w1_c11 = sum(w * c for w, c in zip(w1, c11_0))
    w2_c21 = sum(w * c for w, c in zip(w2, c21_0))
    var_ock = c11_zero - w1_c11 - w2_c21 - m1
    
    return z1_hat, var_ock, w1, w2, m1, m2

# Test on Exercise 7.9 configuration
c1 = [(0.0, 0.0)]
z1 = [10.0]
c2 = [(0.0, 0.0), (1.0, 0.0)]
z2 = [5.0, 8.0]
target = (0.5, 0.0)
B = [[4.0, 2.0], [2.0, 1.25]]
phi = 1.0

pred, var, w1, w2, m1, m2 = ordinary_cokriging_predict(c1, z1, c2, z2, target, B, phi)

print("Exercise 7.9 Verification:")
print(f"  Predicted Z1(0.5) : {pred:.4f}")
print(f"  OCK Variance      : {var:.4f}")
print(f"  Primary Weights   : {[round(w, 4) for w in w1]}")
print(f"  Secondary Weights : {[round(w, 4) for w in w2]}")
print(f"  Multiplier m1     : {m1:.4f}")
print(f"  Multiplier m2     : {m2:.4f}")
```

Output:
```text
Exercise 7.9 Verification:
  Predicted Z1(0.5) : 12.4000
  OCK Variance      : 2.1364
  Primary Weights   : [1.0]
  Secondary Weights : [-0.8, 0.8]
  Multiplier m1     : -0.5625
  Multiplier m2     : -0.1547
```

**Discussion:**
The computational output matches the pencil-and-paper derivations from Exercise 7.9 to four decimal places:
- Primary weight: $\lambda_{1,1} = 1.0000$.
- Secondary weights: $\lambda_{2,1} = -0.8000$, $\lambda_{2,2} = +0.8000$.
- Point prediction: $12.4000$.
- Prediction variance: $2.1364$. $\blacksquare$

---

### Exercise 7.13: Simulation Lab: Empirical Demonstration of the Autokrigeability Invariance
In this simulation lab, you will empirically verify the Autokrigeability Theorem (Exercise 7.7).
**(a)** Construct a synthetic 2D grid of 9 isotopic sampling locations in a $3 \times 3$ grid on $[0, 2] \times [0, 2]$.
**(b)** Assign synthetic observations to both primary variable $Z_1$ and secondary variable $Z_2$ at all 9 points.
**(c)** Define an intrinsic coregionalization matrix $\mathbf{B} = \begin{pmatrix} 5.0 & 3.5 \\ 3.5 & 4.0 \end{pmatrix}$ and exponential correlation range $\phi = 2.0$.
**(d)** Predict $Z_1(\mathbf{s}_0)$ at the center point $(0.5, 0.5)$ using:
1. Univariate Ordinary Kriging on $Z_1$ alone.
2. Bivariate Ordinary Cokriging using both $Z_1$ and $Z_2$.
**(e)** Confirm that:
1. Every secondary weight $\lambda_{2,j}$ in Cokriging is identically zero ($|\lambda_{2,j}| < 10^{-14}$).
2. The primary weights $\lambda_{1,i}$ are identical between Kriging and Cokriging.
3. The point predictions and prediction variances match to machine precision.

##### Solution to Exercise 7.13
Here is the Python verification script:

```python
import math

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve_linear_system(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        max_row = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                max_row = r
        if max_val < 1e-12:
            raise ValueError(f"Singular matrix encountered at pivot {i}.")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for c in range(i, n + 1):
            M[i][c] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(i, n + 1):
                    M[r][c] -= factor * M[i][c]
    return [M[r][n] for r in range(n)]

def ordinary_cokriging_predict(coords_1, z1_data, coords_2, z2_data, target_coord, B_mat, phi):
    """Ordinary Cokriging predictor for bivariate heterotopic data."""
    n1 = len(z1_data)
    n2 = len(z2_data)
    dim = n1 + n2 + 2
    A = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    for i in range(n1):
        for j in range(n1):
            h = euclidean_dist(coords_1[i], coords_1[j])
            A[i][j] = B_mat[0][0] * math.exp(-h / phi)
    for i in range(n1):
        for j in range(n2):
            h = euclidean_dist(coords_1[i], coords_2[j])
            cov = B_mat[0][1] * math.exp(-h / phi)
            A[i][n1 + j] = cov
            A[n1 + j][i] = cov
    for i in range(n2):
        for j in range(n2):
            h = euclidean_dist(coords_2[i], coords_2[j])
            A[n1 + i][n1 + j] = B_mat[1][1] * math.exp(-h / phi)
    for i in range(n1):
        A[i][n1 + n2] = 1.0
        A[n1 + n2][i] = 1.0
    for j in range(n2):
        A[n1 + j][n1 + n2 + 1] = 1.0
        A[n1 + n2 + 1][n1 + j] = 1.0
    c11_0 = [0.0] * n1
    for i in range(n1):
        h = euclidean_dist(coords_1[i], target_coord)
        cov = B_mat[0][0] * math.exp(-h / phi)
        c11_0[i] = cov
        b[i] = cov
    c21_0 = [0.0] * n2
    for j in range(n2):
        h = euclidean_dist(coords_2[j], target_coord)
        cov = B_mat[1][0] * math.exp(-h / phi)
        c21_0[j] = cov
        b[n1 + j] = cov
    b[n1 + n2] = 1.0
    b[n1 + n2 + 1] = 0.0
    sol = solve_linear_system(A, b)
    w1 = sol[:n1]
    w2 = sol[n1:n1 + n2]
    m1 = sol[n1 + n2]
    m2 = sol[n1 + n2 + 1]
    z1_hat = sum(w * z for w, z in zip(w1, z1_data)) + sum(w * z for w, z in zip(w2, z2_data))
    c11_zero = B_mat[0][0]
    w1_c11 = sum(w * c for w, c in zip(w1, c11_0))
    w2_c21 = sum(w * c for w, c in zip(w2, c21_0))
    var_ock = c11_zero - w1_c11 - w2_c21 - m1
    return z1_hat, var_ock, w1, w2, m1, m2

# 1. Setup 3x3 isotopic grid
grid_pts = []
for x in [0.0, 1.0, 2.0]:
    for y in [0.0, 1.0, 2.0]:
        grid_pts.append((x, y))

# Arbitrary isotopic values
z1_vals = [12.1, 14.3, 11.8, 15.2, 16.0, 13.9, 10.5, 12.8, 14.1]
z2_vals = [25.4, 28.1, 23.9, 29.8, 31.2, 27.5, 22.0, 26.5, 28.9]

target_pt = (0.5, 0.5)
B_matrix = [[5.0, 3.5], [3.5, 4.0]]
phi_val = 2.0

# 2. Ordinary Cokriging
pred_ck, var_ck, w1_ck, w2_ck, m1_ck, m2_ck = ordinary_cokriging_predict(
    grid_pts, z1_vals, grid_pts, z2_vals, target_pt, B_matrix, phi_val
)

# 3. Univariate Ordinary Kriging (using Ex 5.8 formulation with sill = B[0][0])
def univariate_ok(coords, z_data, target, sill, phi):
    n = len(z_data)
    A = [[0.0]*(n+1) for _ in range(n+1)]
    b = [0.0]*(n+1)
    for i in range(n):
        for j in range(n):
            h = euclidean_dist(coords[i], coords[j])
            A[i][j] = sill * math.exp(-h / phi)
        A[i][n] = 1.0
        A[n][i] = 1.0
        h0 = euclidean_dist(coords[i], target)
        cov0 = sill * math.exp(-h0 / phi)
        b[i] = cov0
    b[n] = 1.0
    sol = solve_linear_system(A, b)
    w = sol[:n]
    m = sol[n]
    z_hat = sum(wi * zi for wi, zi in zip(w, z_data))
    c0_zero = sill
    var_ok = c0_zero - sum(wi * bi for wi, bi in zip(w, b[:n])) - m
    return z_hat, var_ok, w, m

pred_ok, var_ok, w_ok, m_ok = univariate_ok(grid_pts, z1_vals, target_pt, B_matrix[0][0], phi_val)

print("=== Autokrigeability Verification ===")
print(f"Cokriging Point Prediction : {pred_ck:.10f}")
print(f"Kriging Point Prediction   : {pred_ok:.10f}")
print(f"Difference in Prediction   : {abs(pred_ck - pred_ok):.2e}\n")

print(f"Cokriging Prediction Var   : {var_ck:.10f}")
print(f"Kriging Prediction Var     : {var_ok:.10f}")
print(f"Difference in Variance     : {abs(var_ck - var_ok):.2e}\n")

max_sec_w = max(abs(w) for w in w2_ck)
print(f"Maximum Secondary Weight   : {max_sec_w:.2e}")
max_w_diff = max(abs(w1 - w2) for w1, w2 in zip(w1_ck, w_ok))
print(f"Max Primary Weight Diff    : {max_w_diff:.2e}")
```

Output:
```text
=== Autokrigeability Verification ===
Cokriging Point Prediction : 14.1611115598
Kriging Point Prediction   : 14.1611115598
Difference in Prediction   : 0.00e+00

Cokriging Prediction Var   : 1.2842109271
Kriging Prediction Var     : 1.2842109271
Difference in Variance     : 0.00e+00

Maximum Secondary Weight   : 1.11e-16
Max Primary Weight Diff    : 1.11e-16
```

**Diagnostic Reflection:**
The simulation confirms the analytical proof:
1. Every secondary weight in Cokriging vanishes to machine epsilon ($1.11 \times 10^{-16} \approx 0$).
2. The primary weights match univariate Ordinary Kriging perfectly.
3. Prediction and prediction variance differences are strictly $0.00$.
Whenever sampling is isotopic and the linear model of coregionalization contains only a single spatial structure (intrinsic correlation), gathering secondary data adds exactly zero predictive value. Secondary data is only useful when sampling is heterotopic (primary data is missing where secondary is present) or when multiple spatial structures with distinct ranges are present. $\blacksquare$

---

## Tier IV: Edge-Case Diagnostics & Conceptual Reflections

### Exercise 7.14: The Collocated Cokriging Variance Inflation Paradox
Consider a collocated cokriging setup where an inexpensive secondary variable $Z_2$ is available at millions of seismic pixels, while primary well-log porosity $Z_1$ is sampled at only 30 boreholes.
In simplified Collocated Cokriging, software packages retain only the primary data $Z_1(\mathbf{s}_1), \dots, Z_1(\mathbf{s}_n)$ and the collocated secondary point $Z_2(\mathbf{s}_0)$, intentionally discarding the secondary observations at the borehole locations $Z_2(\mathbf{s}_1), \dots, Z_2(\mathbf{s}_n)$.
**(a)** Explain why discarding $Z_2(\mathbf{s}_i)$ at the borehole locations causes the model to overestimate the independent information content of $Z_2(\mathbf{s}_0)$.
**(b)** What happens when an unsampled target $\mathbf{s}_0$ approaches very close to an existing borehole $\mathbf{s}_1$ ($\|\mathbf{s}_0 - \mathbf{s}_1\| \to 0$)? Explain why the weights $\lambda_1(\mathbf{s}_1)$ and $\lambda_2(\mathbf{s}_0)$ begin to oscillate wildly.
**(c)** How does the **Multi-Collocated Cokriging** formulation resolve this instability?

##### Solution to Exercise 7.14
**(a)** Overestimation of Information Content:
When secondary observations at the boreholes $Z_2(\mathbf{s}_1), \dots, Z_2(\mathbf{s}_n)$ are discarded:
- The system cannot calibrate the local residual error between $Z_1$ and $Z_2$ at known points.
- The predictor assumes that the collocated value $Z_2(\mathbf{s}_0)$ is conditionally independent of the primary data $Z_1(\mathbf{s}_1), \dots, Z_1(\mathbf{s}_n)$ given the unobserved value $Z_1(\mathbf{s}_0)$.
- Consequently, $Z_2(\mathbf{s}_0)$ receives an excessively large weight, and the calculated cokriging variance $\sigma_{\text{CCK}}^2(\mathbf{s}_0)$ is artificially deflated, giving a false sense of certainty.

**(b)** Instability and Weight Oscillation as $\mathbf{s}_0 \to \mathbf{s}_1$:
As the prediction target $\mathbf{s}_0$ approaches the borehole location $\mathbf{s}_1$:
- The distance $\|\mathbf{s}_0 - \mathbf{s}_1\| \to 0$, which implies $C_{11}(\mathbf{s}_1 - \mathbf{s}_0) \to C_{11}(\mathbf{0})$ and $C_{12}(\mathbf{s}_1 - \mathbf{s}_0) \to C_{12}(\mathbf{0})$.
- The predictor contains two highly redundant terms: $Z_1(\mathbf{s}_1)$ (the primary observation at $\mathbf{s}_1$) and $Z_2(\mathbf{s}_0)$ (which is virtually collocated with $\mathbf{s}_1$).
- Because the system does *not* include $Z_2(\mathbf{s}_1)$, it fails to account for the cross-correlation between $Z_1(\mathbf{s}_1)$ and $Z_2(\mathbf{s}_1)$.
- Inverting this ill-conditioned block causes the weights to destabilize: $\lambda_1(\mathbf{s}_1)$ can swing to $+2.5$ while $\lambda_2(\mathbf{s}_0)$ swings to $-1.5$, producing severe edge artifacts and discontinuous spikes in the prediction map near boreholes.

**(c)** Resolution via Multi-Collocated Cokriging:
Multi-Collocated Cokriging retains:
1. All primary data: $\{Z_1(\mathbf{s}_1), \dots, Z_1(\mathbf{s}_n)\}$.
2. The collocated secondary point: $Z_2(\mathbf{s}_0)$.
3. The secondary data *at all primary sample locations*: $\{Z_2(\mathbf{s}_1), \dots, Z_2(\mathbf{s}_n)\}$.
By including the secondary measurements at the boreholes, the system properly conditions on the observed discrepancy $Z_1(\mathbf{s}_i) - Z_2(\mathbf{s}_i)$, strictly eliminating collinearity, suppressing weight oscillations as $\mathbf{s}_0 \to \mathbf{s}_1$, and restoring continuous, well-calibrated prediction variances. $\blacksquare$

---

### Exercise 7.15: Cramér's Theorem and Admissibility of Cross-Covariances
In univariate spatial statistics, Bochner's Theorem guarantees that any positive definite function is a valid covariance. In multivariate spatial statistics, Cramér's Theorem establishes the conditions for matrix positive definiteness.
**(a)** State Cramér's Theorem for matrix covariance kernels: what condition must the matrix spectral density function $\mathbf{f}(\boldsymbol{\omega})$ satisfy for all spatial frequencies $\boldsymbol{\omega} \in \mathbb{R}^d$?
**(b)** Prove the **Cauchy-Schwarz Cross-Covariance Bound**: for any valid second-order stationary bivariate process:
$$
|C_{12}(\mathbf{h})|^2 \le C_{11}(\mathbf{0}) C_{22}(\mathbf{0}) \quad \forall \mathbf{h} \in \mathbb{R}^d
$$
**(c)** Give a counterexample showing why independently choosing two valid univariate covariances $C_{11}(h) = \exp(-|h|)$ and $C_{22}(h) = \exp(-|h|/3)$ with a naive cross-covariance $C_{12}(h) = 0.9 \exp(-|h|/2)$ can violate Cramér's condition and fail to be positive definite.

##### Solution to Exercise 7.15
**(a)** Cramér's Theorem (1940):
A continuous matrix function $\mathbf{C}(\mathbf{h}) = [C_{ij}(\mathbf{h})]_{i,j=1}^p$ is a valid, positive definite multivariate cross-covariance function on $\mathbb{R}^d$ if and only if it admits the continuous Fourier spectral representation:
$$
\mathbf{C}(\mathbf{h}) = \int_{\mathbb{R}^d} e^{i \boldsymbol{\omega}'\mathbf{h}} \mathbf{f}(\boldsymbol{\omega}) d\boldsymbol{\omega}
$$
where $\mathbf{f}(\boldsymbol{\omega})$ is a continuous $p \times p$ Hermitian positive semi-definite spectral density matrix function for all spatial frequencies $\boldsymbol{\omega} \in \mathbb{R}^d$:
$$
\mathbf{f}(\boldsymbol{\omega}) = \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} e^{-i \boldsymbol{\omega}'\mathbf{h}} \mathbf{C}(\mathbf{h}) d\mathbf{h}
$$
satisfying the condition:
$$
\mathbf{a}^* \mathbf{f}(\boldsymbol{\omega}) \mathbf{a} \ge 0 \quad \forall \mathbf{a} \in \mathbb{C}^p, \ \forall \boldsymbol{\omega} \in \mathbb{R}^d
$$

**(b)** Cauchy-Schwarz Bound:
By definition of covariance, for any real random variables $X$ and $Y$, $|\operatorname{Cov}(X, Y)|^2 \le \operatorname{Var}(X) \operatorname{Var}(Y)$.
Let $X = Z_1(\mathbf{s}+\mathbf{h})$ and $Y = Z_2(\mathbf{s})$.
Then $\operatorname{Cov}(X, Y) = C_{12}(\mathbf{h})$.
By second-order stationarity:
$$
\operatorname{Var}(X) = \operatorname{Var}(Z_1(\mathbf{s}+\mathbf{h})) = C_{11}(\mathbf{0})
$$
$$
\operatorname{Var}(Y) = \operatorname{Var}(Z_2(\mathbf{s})) = C_{22}(\mathbf{0})
$$
Applying Cauchy-Schwarz:
$$
|C_{12}(\mathbf{h})|^2 = |\operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s}))|^2 \le \operatorname{Var}(Z_1(\mathbf{s}+\mathbf{h})) \operatorname{Var}(Z_2(\mathbf{s})) = C_{11}(\mathbf{0}) C_{22}(\mathbf{0})
$$

**(c)** Counterexample to naive cross-covariance modeling:
Let $d=1$. Consider candidate univariate covariances:
- $C_{11}(h) = e^{-|h|}$, whose 1D Fourier transform is $f_{11}(\omega) = \frac{1}{\pi} \frac{1}{1 + \omega^2}$.
- $C_{22}(h) = e^{-|h|/3}$, whose Fourier transform is $f_{22}(\omega) = \frac{1}{\pi} \frac{3}{1 + 9\omega^2}$.
Suppose a practitioner chooses $C_{12}(h) = 0.9 e^{-|h|/2}$.
At the origin, $|C_{12}(0)| = 0.9 \le \sqrt{1 \times 1} = 1$, which satisfies Cauchy-Schwarz.
However, its Fourier transform is $f_{12}(\omega) = 0.9 \frac{1}{\pi} \frac{2}{1 + 4\omega^2} = \frac{1}{\pi} \frac{1.8}{1 + 4\omega^2}$.
For the spectral density matrix $\mathbf{f}(\omega) = \begin{pmatrix} f_{11}(\omega) & f_{12}(\omega) \\ f_{12}(\omega) & f_{22}(\omega) \end{pmatrix}$ to be positive semi-definite at all frequencies $\omega$, its determinant must be non-negative:
$$
\det \mathbf{f}(\omega) = f_{11}(\omega) f_{22}(\omega) - [f_{12}(\omega)]^2 \ge 0 \quad \forall \omega \in \mathbb{R}
$$
Evaluate at high frequency, say $\omega = 2$:
- $f_{11}(2) = \frac{1}{\pi} \frac{1}{1 + 4} = \frac{1}{5\pi} = 0.200/\pi$.
- $f_{22}(2) = \frac{1}{\pi} \frac{3}{1 + 36} = \frac{3}{37\pi} \approx 0.0811/\pi$.
- Product: $f_{11}(2) f_{22}(2) \approx \frac{0.0162}{\pi^2}$.
- Cross-term: $f_{12}(2) = \frac{1}{\pi} \frac{1.8}{1 + 16} = \frac{1.8}{17\pi} \approx 0.1059/\pi$.
- Squared: $[f_{12}(2)]^2 \approx \frac{0.0112}{\pi^2}$. (Still positive here).
Now evaluate at $\omega = 0.5$:
- $f_{11}(0.5) = \frac{1}{\pi} \frac{1}{1 + 0.25} = \frac{0.80}{\pi}$.
- $f_{22}(0.5) = \frac{1}{\pi} \frac{3}{1 + 2.25} = \frac{3}{3.25\pi} \approx \frac{0.923}{\pi}$.
- Product: $f_{11}(0.5) f_{22}(0.5) \approx \frac{0.738}{\pi^2}$.
- Cross-term: $f_{12}(0.5) = \frac{1}{\pi} \frac{1.8}{1 + 1} = \frac{0.90}{\pi}$.
- Squared: $[f_{12}(0.5)]^2 = \frac{0.810}{\pi^2}$.
Now compute the spectral determinant at $\omega = 0.5$:
$$
\det \mathbf{f}(0.5) = \frac{0.738 - 0.810}{\pi^2} = -\frac{0.072}{\pi^2} < 0
$$
The spectral determinant is strictly negative!
Thus, $\mathbf{C}(h)$ is **mathematically invalid**: placing points on a line can produce negative prediction error variances or non-invertible covariance matrices. This demonstrates why the **Linear Model of Coregionalization (LMC)** is essential to guarantee admissible cross-covariances. $\blacksquare$

---

### Exercise 7.16: Conceptual Reflection: Cokriging vs. Kriging with External Drift
In practice, environmental scientists often must choose between **Cokriging (CK)** and **Kriging with External Drift (KED)** when auxiliary variables are present. Compare these two paradigms.
**(a)** Compare how the auxiliary variable is conceptualized mathematically in CK vs. KED (stochastic field vs. deterministic covariate).
**(b)** Compare their data sampling requirements: can KED be applied if the auxiliary variable is only measured at 10 sparse boreholes? Can CK be applied?
**(c)** Compare their computational cost when predicting across a raster of $10^6$ cells.
**(d)** Provide a definitive decision tree for when a practitioner should choose Cokriging over KED, and vice versa.

##### Solution to Exercise 7.16
**(a)** Stochastic vs. Deterministic Conceptualization:
- **Cokriging (CK):** Models both the primary variable $Z_1(\mathbf{s})$ and secondary variable $Z_2(\mathbf{s})$ as **stochastic random fields**. Cokriging accounts for spatial cross-correlations at different spatial lags ($C_{12}(\mathbf{h})$) and respects the mutual uncertainty in both fields.
- **Kriging with External Drift (KED):** Treats the primary variable as a random field whose expectation is governed by a **deterministic explanatory covariate**: $\operatorname{E}[Y(\mathbf{s})] = \beta_0 + \beta_1 x(\mathbf{s})$. The auxiliary variable $x(\mathbf{s})$ is assumed to be known without error.

**(b)** Data Availability Requirements:
- If the auxiliary variable is sampled at only 10 sparse locations:
  - **KED fails completely:** KED requires knowing the auxiliary variable $x(\mathbf{s}_0)$ at every single target prediction location $\mathbf{s}_0$. If $x$ is unknown at $\mathbf{s}_0$, KED cannot form the drift vector $\mathbf{x}_0 = (1, x(\mathbf{s}_0))'$.
  - **Cokriging succeeds:** CK easily handles sparse, heterotopic sampling. Even if $Z_2$ is only sampled at 10 points, CK incorporates whatever mutual covariance exists between those 10 points and the primary points.

**(c)** Computational Complexity:
- When predicting across $M = 10^6$ raster nodes with exhaustively known secondary data:
  - **Full Cokriging:** Inverting the joint system of size $(n_1 + 10^6)$ is computationally impossible without local neighborhood approximations. Even with local search ($k=50$), local CK requires inverting $(2k + 2) \times (2k + 2)$ matrices $10^6$ times.
  - **KED:** Requires inverting only an $(n_1 + 2) \times (n_1 + 2)$ matrix (or solving local $(k+2) \times (k+2)$ systems), which is computationally orders of magnitude faster.

**(d)** Practical Decision Framework:
1. **Choose Kriging with External Drift (KED) when:**
   - The auxiliary variable is an exhaustive, continuous raster known everywhere across the domain (e.g., elevation from DEM, distance to coast, satellite vegetation index NDVI).
   - The physical relationship between $Y$ and $x$ is primarily a large-scale deterministic drift (trend) rather than a lag-dependent cross-autocorrelation.
   - You want to avoid the mathematical burden of fitting and validating a full Linear Model of Coregionalization (LMC).
2. **Choose Cokriging (CK) when:**
   - The auxiliary variable is sampled sparsely or irregularly, and is missing at many target prediction locations.
   - You want to simultaneously predict both variables across the domain.
   - The cross-dependence exhibits complex spatial lag dynamics (e.g., directional advection shifts, different ranges of spatial correlation, or phase lags).
   - The secondary variable itself is subject to substantial measurement noise that must be filtered out stochastically. $\blacksquare$
