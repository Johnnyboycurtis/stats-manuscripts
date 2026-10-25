# Chapter 4: Covariance Parameter Estimation and Theoretical Validations

Once exploratory analysis and empirical semivariograms have revealed the general structure of spatial dependence, spatial inference requires specifying and fitting a valid parametric covariance kernel:
$$
\mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n
$$
where $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$ indexes the structural parameters: partial sill $\sigma^2$, spatial range $\phi$, and nugget effect $\tau^2$. In this chapter, we develop the statistical theory and computational mechanics of parameter estimation for spatial linear models:
$$
\mathbf{Z} \sim \mathcal{N}_n\left(\mathbf{X}\boldsymbol{\beta}, \; \mathbf{V}(\boldsymbol{\theta})\right)
$$

### Learning Outcomes
By the end of this chapter, you will be able to:
- Formulate Gaussian spatial likelihoods and evaluate them stably using Cholesky factorizations without computing matrix inverses.
- Understand Generalized Least Squares as Ordinary Least Squares on whitened data ($\mathbf{L}\tilde{\mathbf{X}} = \mathbf{X}$, $\mathbf{L}\tilde{\mathbf{z}} = \mathbf{Z}$).
- Derive Profile Maximum Likelihood (ML) and Restricted Maximum Likelihood (REML), proving why REML corrects the downward bias in variance components by projecting onto the error contrast subspace.
- Execute an end-to-end parameter estimation workflow: obtaining initial guesses from variogram fits, profiling out $\sigma^2$ in closed form, optimizing over range and nugget ratio, and calculating standard errors via Fisher Information.
- Diagnose practical optimization challenges: flat likelihood ridges arising from microergodicity, boundary estimates, and local minima.
- Validate multivariate spatial models using Cramér's spectral admissibility condition and the Linear Model of Coregionalization (LMC) equipped with Frobenius eigenvalue projection.

### Chapter at a Glance

| Section | Core Concept | Key Formula / Tool | Practical Takeaway |
|---|---|---|---|
| **§4.1 Likelihood Formulation** | Gaussian spatial likelihood & Fisher information | $\mathcal{I}_{\boldsymbol{\theta}\boldsymbol{\theta}} = \frac{1}{2}\operatorname{Tr}(\mathbf{V}^{-1}\mathbf{V}_k\mathbf{V}^{-1}\mathbf{V}_l)$ | Whiten via Cholesky $\mathbf{V} = \mathbf{L}\mathbf{L}'$; never invert $\mathbf{V}$ directly |
| **§4.2 Profile ML** | Profiling out drift coefficients $\boldsymbol{\beta}$ | $\mathbf{P} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}$ | ML sill is biased downward by $(n-p)/n$ when correlation is fixed |
| **§4.3 Restricted ML (REML)** | Error contrasts annihilating drift $\mathbf{X}\boldsymbol{\beta}$ | $\ell_{\text{REML}} = \ell_p - \frac{1}{2}\ln\|\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\| + \text{const}$ | REML removes bias; use ML only when comparing different $\mathbf{X}$ designs |
| **Workflow & Example** | 4-step practical estimation recipe | Profile out $\sigma^2 = \frac{1}{n-p}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}$ | Reduces 3D optimization to 1D/2D; check Fisher SEs and profile plots |
| **§4.4 Least Squares Fitting** | Cressie's WLS on empirical semivariance | $w_k = \|N(\mathbf{h}_k)\| / \gamma(\mathbf{h}_k; \boldsymbol{\theta})^2$ | Fast starting values; short lags dominate due to small $\gamma(h)$, not pair count |
| **§4.5 Multivariate Admissibility** | Joint positive semi-definiteness across variables | Cramér: $\mathbf{f}(\boldsymbol{\omega}) \succeq 0 \iff \|f_{12}(\boldsymbol{\omega})\|^2 \le f_{11}f_{22}$ | Univariate validity does not guarantee joint validity in cokriging |
| **§4.6 Coregionalization (LMC)** | Spatial factor model & Frobenius projection | $\mathbf{B}^* = \mathbf{Q}\max(0, \mathbf{\Lambda})\mathbf{Q}'$ | Guarantees PSD coregionalization sills by clipping negative eigenvalues |

---

### Bridge: From Classical $(n-1)$ Variance Correction to Spatial REML and Projection Matrix $\mathbf{P}$

In introductory mathematical statistics, the sample variance of $n$ independent and identically distributed (i.i.d.) observations $Z_1, \dots, Z_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$ divides by $n - 1$ rather than $n$:
$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (Z_i - \bar{Z})^2
$$
The uncorrected Maximum Likelihood Estimator (MLE), $\widehat{\sigma}^2_{\text{ML}} = \frac{1}{n}\sum_{i=1}^n (Z_i - \bar{Z})^2$, is systematically biased downward: $\operatorname{E}[\widehat{\sigma}^2_{\text{ML}}] = \frac{n-1}{n}\sigma^2$. The factor $\frac{1}{n-1}$ arises because exactly one degree of freedom was expended estimating the unknown scalar population mean $\mu$ with the sample mean $\bar{Z}$.

```
Classical i.i.d. Estimation (1 Mean Parameter mu):
   Z_1, ..., Z_n  ──>  Estimate mu by Z_bar   ──>  1 d.f. lost  ──>  E[sigma^2_ML] = ((n - 1)/n) * sigma^2

Spatial Regression (p Trend Parameters beta):
   Z ~ N(X beta, V(theta))  ──>  Estimate beta by beta_GLS  ──>  p d.f. lost  ──>  E[sigma^2_ML] = ((n - p)/n) * sigma^2
```

In the spatial linear regression model:
$$
\mathbf{Z} \sim \mathcal{N}_n\left(\mathbf{X}\boldsymbol{\beta}, \; \mathbf{V}(\boldsymbol{\theta})\right)
$$
where $\mathbf{X} \in \mathbb{R}^{n \times p}$ contains $p$ deterministic spatial trend basis functions ($p \ge 1$), estimating the $p$-dimensional coefficient vector $\boldsymbol{\beta}$ causes standard Maximum Likelihood to underestimate the covariance sill scale $\sigma^2$ by the factor:
$$
\operatorname{E}\left[\widehat{\sigma}^2_{\text{ML}}\right] = \left(\frac{n-p}{n}\right)\sigma^2
$$
when the correlation parameters (range $\phi$ and relative nugget) are held fixed.

> [!NOTE]
> **Scope of the $(n-p)/n$ Bias:**  
> The exact $(n-p)/n$ factor holds strictly when the spatial correlation structure is fixed and only the overall scale $\sigma^2$ is estimated (Proposition 4.2.1). When range $\phi$ and nugget $\tau^2$ are estimated jointly with $\sigma^2$, the downward bias mechanism is the same—trend estimation soaks up residual variation—but the magnitude is no longer exactly $(n-p)/n$ and can be significantly more pronounced in moderate samples ($n \le 100$) with rich trend models ($p \ge 6$).

To eliminate this bias, Restricted Maximum Likelihood (REML) constructs a linear transformation that filters out the trend $\mathbf{X}\boldsymbol{\beta}$ completely before estimating covariance parameters $\boldsymbol{\theta}$.

#### Constructing the Linear Error Contrast Matrix $\mathbf{A}$
Let $\mathbf{A} \in \mathbb{R}^{n \times (n-p)}$ be a matrix of full column rank $n-p$ whose columns form an orthonormal basis for the orthogonal complement of the column space of $\mathbf{X}$, $\operatorname{col}(\mathbf{X})^\perp$:
$$
\mathbf{A}'\mathbf{X} = \mathbf{0}_{(n-p) \times p}, \quad \mathbf{A}'\mathbf{A} = \mathbf{I}_{n-p}
$$
Now transform the observation vector $\mathbf{Z}$ into an $(n-p)$-dimensional vector of **error contrasts** $\mathbf{U} = \mathbf{A}'\mathbf{Z}$. Because $\mathbf{A}'\mathbf{X} = \mathbf{0}$:
$$
\operatorname{E}[\mathbf{U}] = \operatorname{E}[\mathbf{A}'\mathbf{Z}] = \mathbf{A}'\mathbf{X}\boldsymbol{\beta} = \mathbf{0}_{n-p}
$$
and its covariance matrix is:
$$
\operatorname{Var}(\mathbf{U}) = \operatorname{Var}(\mathbf{A}'\mathbf{Z}) = \mathbf{A}' \mathbf{V}(\boldsymbol{\theta}) \mathbf{A}
$$
Therefore:
$$
\mathbf{U} \sim \mathcal{N}_{n-p}\left(\mathbf{0}, \; \mathbf{A}'\mathbf{V}(\boldsymbol{\theta})\mathbf{A}\right)
$$
The distribution of $\mathbf{U}$ is **completely invariant to the trend coefficients $\boldsymbol{\beta}$**. Maximizing the likelihood of $\mathbf{U}$ isolates covariance estimation on the $(n-p)$-dimensional error subspace, automatically recovering the lost degrees of freedom.

#### Derivation of the Spatial Projection Matrix $\mathbf{P}$ via Whitening
Although the matrix $\mathbf{A}$ depends on the arbitrary choice of basis for $\operatorname{col}(\mathbf{X})^\perp$, the resulting likelihood and quadratic form are uniquely expressed via the **spatial projection matrix** $\mathbf{P}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$:
$$
\mathbf{P}(\boldsymbol{\theta}) = \mathbf{A}\left(\mathbf{A}'\mathbf{V}(\boldsymbol{\theta})\mathbf{A}\right)^{-1}\mathbf{A}'
$$
We can prove the equivalence of this contrast form and the explicit GLS projection form in three lines via Cholesky whitening:
Let $\mathbf{V} = \mathbf{L}\mathbf{L}'$. Set the whitened design matrix to $\tilde{\mathbf{X}} = \mathbf{L}^{-1}\mathbf{X}$ and the whitened contrast basis to $\tilde{\mathbf{A}} = \mathbf{L}'\mathbf{A}$. Then:
$$
\tilde{\mathbf{X}}'\tilde{\mathbf{A}} = (\mathbf{X}'\mathbf{L}^{-\prime})(\mathbf{L}'\mathbf{A}) = \mathbf{X}'\mathbf{A} = \mathbf{0}
$$
Hence, the columns of $\tilde{\mathbf{A}}$ form a basis for the orthogonal complement of the column space of $\tilde{\mathbf{X}}$. The classical orthogonal projection onto $\operatorname{col}(\tilde{\mathbf{X}})^\perp$ can be written in two equivalent ways:
$$
\mathbf{I}_n - \tilde{\mathbf{X}}(\tilde{\mathbf{X}}'\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}' = \tilde{\mathbf{A}}(\tilde{\mathbf{A}}'\tilde{\mathbf{A}})^{-1}\tilde{\mathbf{A}}'
$$
Premultiplying by $\mathbf{L}^{-\prime}$ and postmultiplying by $\mathbf{L}^{-1}$:
$$
\mathbf{L}^{-\prime}\left[\mathbf{I}_n - \tilde{\mathbf{X}}(\tilde{\mathbf{X}}'\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}'\right]\mathbf{L}^{-1} = \mathbf{L}^{-\prime}\left[\tilde{\mathbf{A}}(\tilde{\mathbf{A}}'\tilde{\mathbf{A}})^{-1}\tilde{\mathbf{A}}'\right]\mathbf{L}^{-1}
$$
Expanding the left-hand side gives $\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}$. Expanding the right-hand side, since $\tilde{\mathbf{A}} = \mathbf{L}'\mathbf{A}$, gives $\mathbf{A}(\mathbf{A}'\mathbf{L}\mathbf{L}'\mathbf{A})^{-1}\mathbf{A}' = \mathbf{A}(\mathbf{A}'\mathbf{V}\mathbf{A})^{-1}\mathbf{A}'$. Thus:
$$
\mathbf{P}(\boldsymbol{\theta}) = \mathbf{A}\left(\mathbf{A}'\mathbf{V}(\boldsymbol{\theta})\mathbf{A}\right)^{-1}\mathbf{A}' = \mathbf{V}(\boldsymbol{\theta})^{-1} - \mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\left(\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}
$$
This spatial projection operator satisfies two fundamental algebraic properties (verified in Theorem 4.2.1):
1. **Drift Annihilation:** $\mathbf{P}\mathbf{X} = \mathbf{0}$ and $\mathbf{X}'\mathbf{P} = \mathbf{0}$.
2. **Generalized Idempotency:** $\mathbf{P}\mathbf{V}\mathbf{P} = \mathbf{P}$.

> [!NOTE]
> **Computational Complexity and Scaling Up:**  
> Evaluating either the full Gaussian log-likelihood or the REML objective requires the Cholesky factorization of $\mathbf{V}(\boldsymbol{\theta})$, incurring an exact computational cost of $\mathcal{O}(n^3)$ operations and $\mathcal{O}(n^2)$ memory. For large spatial datasets ($n > 10^4$), specialized approximations—such as Vecchia approximations, Gaussian Markov random field (GMRF) representations, or composite likelihoods—are required. We examine scalable spatial computation in Chapter 13.

---

### 4.1 The Spatial Linear Model and Gaussian Likelihood Formulation

#### The Mathematical Anchor

##### Classical-to-Spatial Bridge: From OLS Log-Likelihood to Spatial Likelihood and the Determinant Penalty

In classical linear regression with independent and identically distributed (i.i.d.) Gaussian errors $Z_i \sim \mathcal{N}(\mathbf{x}_i'\boldsymbol{\beta}, \sigma^2)$, the covariance matrix is spherical: $\mathbf{V} = \sigma^2 \mathbf{I}_n$. The classical log-likelihood function separates neatly into an elementary variance term and the Euclidean residual sum of squares:
$$
\ell(\boldsymbol{\beta}, \sigma^2; \mathbf{Z}) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
$$
In spatial statistics, observations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ are correlated across geographic space via distance-dependent covariances $\mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n$. This spatial autocorrelation alters the objective in two fundamental ways:
1. **The Mahalanobis Goodness-of-Fit Form:** The simple Euclidean sum of squared residuals is replaced by the generalized Mahalanobis quadratic form $(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}(\boldsymbol{\theta})^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})$. Deviations along directions of high spatial correlation are decorrelated and scaled by $\mathbf{V}(\boldsymbol{\theta})^{-1}$.
2. **The Determinant Volume Penalty:** The scalar variance penalty $n\ln(\sigma^2)$ is replaced by the log-determinant $\ln|\mathbf{V}(\boldsymbol{\theta})|$. The determinant measures the generalized variance (dispersion volume). For a fixed sill $\sigma^2$, as spatial range $\phi$ increases, points become more strongly correlated, eigenvalues disperse, and $|\mathbf{R}|$ shrinks toward zero, causing $-\frac{1}{2}\ln|\mathbf{V}|$ to *increase*. The determinant thus rewards strong correlation (the data are more compressible); the Mahalanobis quadratic form punishes correlation structures that the data do not support. Maximum likelihood achieves a balance between the two. Only the overall scale parameter $\sigma^2$ enters $\ln|\mathbf{V}|$ as an increasing penalty $(n\ln\sigma^2)$.

Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a Gaussian spatial random field observed at $n$ discrete locations $S = \{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset D \subset \mathbb{R}^d$. The spatial linear model is formulated as:
$$
Z(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \delta(\mathbf{s}) + \epsilon(\mathbf{s}), \quad \mathbf{s} \in D
$$
where:
- $\mathbf{x}(\mathbf{s}) = (x_1(\mathbf{s}), \dots, x_p(\mathbf{s}))' \in \mathbb{R}^p$ is a vector of deterministic spatial covariates (e.g., coordinates, polynomial drift basis functions, elevation).
- $\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)' \in \mathbb{R}^p$ is the vector of unknown large-scale regression coefficients.
- $\delta(\mathbf{s})$ is a zero-mean, second-order stationary Gaussian spatial process with continuous covariance kernel $C(\mathbf{h}; \boldsymbol{\theta})$ where $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$.
- $\epsilon(\mathbf{s})$ is an independent Gaussian white-noise microscale error process with variance $\tau^2$.

Stacking the observations across all $n$ sample locations yields the matrix linear model:
$$
\mathbf{Z} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\delta} + \boldsymbol{\epsilon}
$$
where $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))' \in \mathbb{R}^n$, $\mathbf{X} = (\mathbf{x}(\mathbf{s}_1), \dots, \mathbf{x}(\mathbf{s}_n))' \in \mathbb{R}^{n \times p}$ with $\operatorname{rank}(\mathbf{X}) = p < n$, and the total spatial covariance matrix is:
$$
\mathbf{V}(\boldsymbol{\theta}) = \operatorname{Var}(\mathbf{Z}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n \in \mathbb{R}^{n \times n}
$$
We assume throughout that $\mathbf{V}(\boldsymbol{\theta})$ is strictly positive definite ($\mathbf{V}(\boldsymbol{\theta}) \succ 0$).

##### Definition 4.1.1 (Gaussian Spatial Log-Likelihood Function)
Under the joint Gaussian law $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}(\boldsymbol{\theta}))$, the log-likelihood function $\ell(\boldsymbol{\beta}, \boldsymbol{\theta}; \mathbf{Z})$ is:
$$
\ell(\boldsymbol{\beta}, \boldsymbol{\theta}; \mathbf{Z}) = -\frac{n}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
$$

##### Term-by-Term Equation Anatomy: Full Gaussian Spatial Log-Likelihood
$$
\ell(\boldsymbol{\beta}, \boldsymbol{\theta}) = \underbrace{-\frac{n}{2}\ln(2\pi)}_{\text{Normalization Constant}} \underbrace{-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|}_{\text{Volume / Dispersion Penalty}} \underbrace{-\frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}(\boldsymbol{\theta})^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})}_{\text{Generalized Mahalanobis Residual Form}}
$$
- **$-\frac{n}{2}\ln(2\pi)$:** Normalization constant of the $n$-dimensional Gaussian probability density integrating to unity over $\mathbb{R}^n$.
- **$-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|$:** Log-determinant of the spatial covariance matrix, measuring the generalized variance or volume of the error dispersion ellipsoid. For a fixed scale, stronger correlation reduces $|\mathbf{V}|$, rewarding parsimonious correlation; the sill $\sigma^2$ enters as $-n\ln\sigma^2$, penalizing excessive total variance.
- **$-\frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}(\boldsymbol{\theta})^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})$:** Squared Mahalanobis distance measuring goodness-of-fit. Residuals $\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}$ are decorrelated and scaled by the inverse covariance operator $\mathbf{V}^{-1}$; deviations along directions of low spatial variance are heavily penalized.

##### Lemma 4.1.1 (Matrix Calculus Identities for Covariance Derivatives)
Let $\mathbf{V}(\boldsymbol{\theta})$ be a symmetric, positive definite matrix whose elements are continuously differentiable with respect to $\theta_k$. Then:
1. The derivative of the inverse is:
   $$
   \frac{\partial \mathbf{V}^{-1}}{\partial \theta_k} = -\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1}
   $$
2. Jacobi's formula for the log-determinant derivative is:
   $$
   \frac{\partial \ln\det(\mathbf{V})}{\partial \theta_k} = \operatorname{Tr}\left(\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k}\right)
   $$

*Proof.*
1. Differentiating the identity $\mathbf{V}\mathbf{V}^{-1} = \mathbf{I}_n$ with respect to $\theta_k$ using the product rule:
   $$
   \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1} + \mathbf{V} \frac{\partial \mathbf{V}^{-1}}{\partial \theta_k} = \mathbf{0} \implies \mathbf{V} \frac{\partial \mathbf{V}^{-1}}{\partial \theta_k} = -\frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1}
   $$
   Premultiplying by $\mathbf{V}^{-1}$ yields $\frac{\partial \mathbf{V}^{-1}}{\partial \theta_k} = -\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1}$.
2. Consider the first-order Taylor expansion of $\det(\mathbf{V} + \epsilon \mathbf{E})$ for small $\epsilon$ with $\mathbf{E} = \frac{\partial \mathbf{V}}{\partial \theta_k}$:
   $$
   \det(\mathbf{V} + \epsilon \mathbf{E}) = \det(\mathbf{V}) \det\left(\mathbf{I}_n + \epsilon \mathbf{V}^{-1}\mathbf{E}\right)
   $$
   Let $\mu_1, \dots, \mu_n$ be the eigenvalues of $\mathbf{V}^{-1}\mathbf{E}$. Then:
   $$
   \det\left(\mathbf{I}_n + \epsilon \mathbf{V}^{-1}\mathbf{E}\right) = \prod_{i=1}^n (1 + \epsilon \mu_i) = 1 + \epsilon \sum_{i=1}^n \mu_i + \mathcal{O}(\epsilon^2) = 1 + \epsilon \operatorname{Tr}\left(\mathbf{V}^{-1}\mathbf{E}\right) + \mathcal{O}(\epsilon^2)
   $$
   Taking logarithms:
   $$
   \ln\det(\mathbf{V} + \epsilon \mathbf{E}) = \ln\det(\mathbf{V}) + \ln\left(1 + \epsilon \operatorname{Tr}\left(\mathbf{V}^{-1}\mathbf{E}\right) + \mathcal{O}(\epsilon^2)\right) = \ln\det(\mathbf{V}) + \epsilon \operatorname{Tr}\left(\mathbf{V}^{-1}\mathbf{E}\right) + \mathcal{O}(\epsilon^2)
   $$
   Differentiating with respect to $\epsilon$ at $\epsilon = 0$ yields $\frac{\partial \ln\det(\mathbf{V})}{\partial \theta_k} = \operatorname{Tr}\left(\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k}\right)$. $\blacksquare$

##### Theorem 4.1.1 (Gaussian Spatial Score Equations)
The components of the spatial score vector $\mathbf{S}(\boldsymbol{\beta}, \boldsymbol{\theta}) = \nabla_{(\boldsymbol{\beta}, \boldsymbol{\theta})} \ell(\boldsymbol{\beta}, \boldsymbol{\theta})$ are:
1. The drift score vector:
   $$
   \frac{\partial \ell}{\partial \boldsymbol{\beta}} = \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
   $$
2. The covariance score components, for $k = 1, \dots, q$:
   $$
   \frac{\partial \ell}{\partial \theta_k} = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{V}(\boldsymbol{\theta})^{-1} \frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}\right) + \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}(\boldsymbol{\theta})^{-1} \frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k} \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
   $$

*Proof.*
1. Expanding the residual quadratic form $Q(\boldsymbol{\beta}) = (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{Z}'\mathbf{V}^{-1}\mathbf{Z} - 2\boldsymbol{\beta}'\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} + \boldsymbol{\beta}'\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\boldsymbol{\beta}$.
   Differentiating with respect to $\boldsymbol{\beta}$:
   $$
   \frac{\partial \ell}{\partial \boldsymbol{\beta}} = -\frac{1}{2} \nabla_{\boldsymbol{\beta}} Q(\boldsymbol{\beta}) = -\frac{1}{2}\left(-2\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} + 2\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\boldsymbol{\beta}\right) = \mathbf{X}'\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
   $$
2. Differentiating with respect to $\theta_k$ holding $\boldsymbol{\beta}$ fixed:
   $$
   \frac{\partial \ell}{\partial \theta_k} = -\frac{1}{2} \frac{\partial \ln\det(\mathbf{V})}{\partial \theta_k} - \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \left(\frac{\partial \mathbf{V}^{-1}}{\partial \theta_k}\right) (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
   $$
   Substituting Lemma 4.1.1:
   $$
   \frac{\partial \ell}{\partial \theta_k} = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right) - \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \left(-\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1}\right) (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
   $$
   which simplifies immediately to the stated score component. $\blacksquare$

##### Theorem 4.1.2 (Block-Diagonality of the Spatial Fisher Information Matrix)
Let $\boldsymbol{\psi} = (\boldsymbol{\beta}', \boldsymbol{\theta}')' \in \mathbb{R}^{p+q}$. Under standard likelihood regularity conditions, the expected Fisher Information matrix satisfies $\mathcal{I}(\boldsymbol{\psi}) = \operatorname{E}[-\nabla_{\boldsymbol{\psi}}^2 \ell(\boldsymbol{\psi}; \mathbf{Z})] = \operatorname{Cov}(\mathbf{S}(\boldsymbol{\psi}))$ and is strictly block-diagonal:
$$
\mathcal{I}(\boldsymbol{\psi}) = \begin{bmatrix} \mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\beta}} & \mathbf{0} \\ \mathbf{0} & \mathcal{I}_{\boldsymbol{\theta}\boldsymbol{\theta}} \end{bmatrix}
$$
where:
$$
\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\beta}} = \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X} \in \mathbb{R}^{p \times p}
$$
and the elements of $\mathcal{I}_{\boldsymbol{\theta}\boldsymbol{\theta}} \in \mathbb{R}^{q \times q}$ are:
$$
\mathcal{I}_{\theta_k \theta_l} = \frac{1}{2} \operatorname{Tr}\left(\mathbf{V}(\boldsymbol{\theta})^{-1} \frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k} \mathbf{V}(\boldsymbol{\theta})^{-1} \frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_l}\right), \quad k, l \in \{1, \dots, q\}
$$

*Proof.*
1. For $\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\beta}}$:
   Differentiating $\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} - \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\boldsymbol{\beta}$ with respect to $\boldsymbol{\beta}'$:
   $$
   -\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}'} = \mathbf{X}' \mathbf{V}^{-1} \mathbf{X}
   $$
   Because this expression does not depend on $\mathbf{Z}$, $\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\beta}} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$.
2. For the cross-block $\mathcal{I}_{\boldsymbol{\beta}\theta_k}$:
   Differentiating $\frac{\partial \ell}{\partial \boldsymbol{\beta}}$ with respect to $\theta_k$:
   $$
   \frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \theta_k} = \mathbf{X}' \left(\frac{\partial \mathbf{V}^{-1}}{\partial \theta_k}\right) (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = -\mathbf{X}' \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
   $$
   Taking mathematical expectations, since $\operatorname{E}[\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}] = \mathbf{0}$:
   $$
   \mathcal{I}_{\boldsymbol{\beta}\theta_k} = \operatorname{E}\left[-\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \theta_k}\right] = \mathbf{X}' \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1} \operatorname{E}[\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}] = \mathbf{0}
   $$
   Thus, $\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\theta}} = \mathbf{0}$.
3. For $\mathcal{I}_{\boldsymbol{\theta}\boldsymbol{\theta}}$ via the expected Hessian:
   Let $\boldsymbol{\varepsilon} = \mathbf{Z} - \mathbf{X}\boldsymbol{\beta} \sim \mathcal{N}_n(\mathbf{0}, \mathbf{V})$. Differentiating the score component $\frac{\partial \ell}{\partial \theta_k}$ from Theorem 4.1.1 with respect to $\theta_l$:
   $$
   \frac{\partial^2 \ell}{\partial \theta_k \partial \theta_l} = -\frac{1}{2}\frac{\partial}{\partial \theta_l}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right) - \frac{1}{2}\boldsymbol{\varepsilon}'\left(\frac{\partial^2 \mathbf{V}^{-1}}{\partial \theta_k \partial \theta_l}\right)\boldsymbol{\varepsilon}
   $$
   Using the derivative of $\mathbf{V}^{-1}$ from Lemma 4.1.1:
   $$
   \frac{\partial}{\partial \theta_l}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right) = -\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_l}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k} + \mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta_k \partial \theta_l}
   $$
   and differentiating $\frac{\partial \mathbf{V}^{-1}}{\partial \theta_k} = -\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}$:
   $$
   \frac{\partial^2 \mathbf{V}^{-1}}{\partial \theta_k \partial \theta_l} = \mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_l}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1} + \mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_l}\mathbf{V}^{-1} - \mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta_k \partial \theta_l}\mathbf{V}^{-1}
   $$
   Now take expectations using $\operatorname{E}[\boldsymbol{\varepsilon}'\mathbf{M}\boldsymbol{\varepsilon}] = \operatorname{Tr}(\mathbf{M}\mathbf{V})$ for any symmetric matrix $\mathbf{M}$. For the term involving the second derivative $\frac{\partial^2 \mathbf{V}}{\partial \theta_k \partial \theta_l}$:
   $$
   \operatorname{E}\left[\boldsymbol{\varepsilon}'\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta_k \partial \theta_l}\mathbf{V}^{-1}\right)\boldsymbol{\varepsilon}\right] = \operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta_k \partial \theta_l}\mathbf{V}^{-1}\mathbf{V}\right) = \operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta_k \partial \theta_l}\right)
   $$
   which cancels the second trace term from the first line identically! For the remaining terms, by the cyclic property of trace:
   $$
   \operatorname{E}\left[-\frac{\partial^2 \ell}{\partial \theta_k \partial \theta_l}\right] = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_l}\right)
   $$
   establishing the block-diagonal structure without invoking higher-order moment theorems. $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: Reading an Information Matrix**  
> The Fisher Information matrix $\mathcal{I}(\boldsymbol{\psi})$ measures the expected curvature of the log-likelihood surface around the true parameters. Block-diagonality between $\boldsymbol{\beta}$ and $\boldsymbol{\theta}$ means that the large-scale trend parameters and small-scale covariance parameters are asymptotically orthogonal: uncertainty in estimating $\boldsymbol{\beta}$ does not inflate the asymptotic variance of $\widehat{\boldsymbol{\theta}}$, and vice versa. However, within the covariance block $\mathcal{I}_{\boldsymbol{\theta}\boldsymbol{\theta}}$, off-diagonal terms between sill $\sigma^2$ and range $\phi$ are strongly non-zero, reflecting the fundamental trade-off between variance magnitude and spatial continuity.

---

#### Algorithmic Translation: Likelihood via Cholesky Factorization

The cardinal computational rule of spatial statistics is: **never invert $\mathbf{V}(\boldsymbol{\theta})$ explicitly**. Forming $\mathbf{V}^{-1}$ is numerically unstable, prone to catastrophic cancellation, and requires $\mathcal{O}(n^3)$ operations. Instead, stable likelihood evaluation computes the **Cholesky factorization**:
$$
\mathbf{V}(\boldsymbol{\theta}) = \mathbf{L}\mathbf{L}'
$$
where $\mathbf{L} \in \mathbb{R}^{n \times n}$ is lower-triangular with strictly positive diagonal entries.
1. The log-determinant is computed directly from the diagonal of $\mathbf{L}$:
   $$
   \ln\det(\mathbf{V}) = \ln\det(\mathbf{L}\mathbf{L}') = 2\sum_{i=1}^n \ln L_{ii}
   $$
2. The quadratic form $(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})$ is evaluated by solving the triangular linear system $\mathbf{L}\mathbf{w} = \mathbf{Z} - \mathbf{X}\boldsymbol{\beta}$ via forward substitution, yielding:
   $$
   (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{w}'\mathbf{w} = \|\mathbf{w}\|_2^2
   $$

```python
import math

def cholesky_decompose(A):
    """Computes lower-triangular Cholesky factor L such that A = L L'."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0.0:
                    return None  # Matrix not positive definite
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def solve_triangular_lower(L, b):
    """Solves L w = b via forward substitution."""
    n = len(L)
    w = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * w[j] for j in range(i))
        w[i] = (b[i] - s) / L[i][i]
    return w

def compute_gaussian_log_likelihood(Z, X, beta, theta, coords, cov_func):
    """
    Evaluates the Gaussian spatial log-likelihood matching Definition 4.1.1
    using stable Cholesky decomposition without forming V^-1.
    """
    n = len(Z)
    p = len(beta)
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Assemble covariance matrix V(theta)
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            r_val = cov_func(dist, phi)
            V[i][j] = sigma2 * r_val + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization: V = L L'
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    # 3. Log-determinant via Cholesky diagonal
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 4. Residual vector: e = Z - X beta
    residuals = [Z[i] - sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    
    # 5. Whiten residuals: solve L w = e by forward substitution
    w = solve_triangular_lower(L, residuals)
    quad_form = sum(val * val for val in w)
    
    # 6. Gaussian log-likelihood
    log_lik = -0.5 * (n * math.log(2.0 * math.pi) + log_det_V + quad_form)
    return log_lik
```

---

#### Intuition, Edge Cases & Diagnostic Reality

1. **Matrix Conditioning and the Nugget Regularizer:**
   The condition number $\kappa(\mathbf{V}) = \lambda_{\max}(\mathbf{V}) / \lambda_{\min}(\mathbf{V})$ explodes when sampling sites cluster closely ($\|\mathbf{s}_i - \mathbf{s}_j\| \ll \phi$) and the nugget $\tau^2 \to 0$. The microscale error $\tau^2 \mathbf{I}_n$ acts as an intrinsic **Tikhonov regularization parameter**, ensuring $\lambda_{\min}(\mathbf{V}) \ge \tau^2 > 0$. In software, enforcing a tiny nugget floor (e.g. $\tau^2 \ge 10^{-6}\sigma^2$) prevents numerical Cholesky breakdowns on clustered data.
2. **The $(\sigma^2, \phi)$ Likelihood Ridge:**
   In spatial likelihood optimization, the likelihood surface frequently exhibits an elongated, flat ridge along directions where the ratio $\sigma^2 / \phi$ remains constant (discussed in detail under infill asymptotics). Along this ridge, doubling both the range and sill changes the log-likelihood by less than $0.05$, causing generic gradient optimizers to wander or stop prematurely.
3. **Boundary Estimates and Local Minima:**
   Numerical optimization of the spatial likelihood can converge to the boundary ($\widehat{\tau}^2 = 0$ or $\widehat{\phi} \to \infty$). A range estimate that explodes toward infinity indicates that the spatial variation is being absorbed into a flat, constant trend, whereas $\widehat{\tau}^2 = 0$ indicates pure smooth spatial variation. Always evaluate multiple starting values derived from the empirical variogram.

---

### 4.2 Profile Maximum Likelihood Estimation

#### The Mathematical Anchor

##### Classical-to-Spatial Bridge: From Classical OLS Residual Projection $\mathbf{I} - \mathbf{H}$ to the Spatial Projection Operator $\mathbf{P}$

In classical linear models, the Ordinary Least Squares (OLS) residual vector $\mathbf{e} = \mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}$ is obtained via the orthogonal projection operator $\mathbf{M} = \mathbf{I}_n - \mathbf{H}$, where $\mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ is the "hat matrix". The annihilator $\mathbf{M}$ satisfies three foundational properties:
1. **Drift Annihilation:** $\mathbf{M}\mathbf{X} = \mathbf{0}$, meaning $\mathbf{M}$ completely wipes out the mean drift surface $\mathbf{X}\boldsymbol{\beta}$.
2. **Idempotency:** $\mathbf{M}^2 = \mathbf{M}$, reflecting geometric projection.
3. **Trace:** $\operatorname{Tr}(\mathbf{M}) = \operatorname{Tr}(\mathbf{I}_n) - \operatorname{Tr}(\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}') = n - p$, isolating the $n-p$ residual degrees of freedom.

The residual sum of squares is thus concisely represented as the quadratic form $\mathbf{e}'\mathbf{e} = \mathbf{Z}'\mathbf{M}\mathbf{Z}$.

In spatial linear models with correlated errors $\mathbf{V}(\boldsymbol{\theta})$, Generalized Least Squares (GLS) weights observations by the precision matrix $\mathbf{V}(\boldsymbol{\theta})^{-1}$. Setting the drift score equation from Theorem 4.1.1 to zero:
$$
\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}
$$
yields the Generalized Least Squares (GLS) estimator for $\boldsymbol{\beta}$ as an explicit analytical function of $\boldsymbol{\theta}$:
$$
\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = \left(\mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{Z}
$$
The spatial counterpart to the classical projection operator is the **spatial projection operator** $\mathbf{P}(\boldsymbol{\theta})$:
$$
\mathbf{P}(\boldsymbol{\theta}) = \mathbf{V}(\boldsymbol{\theta})^{-1} - \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1}
$$

##### Definition 4.2.1 (Profile Likelihood and the Spatial Projection Operator)
Define the spatial projection operator $\mathbf{P}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$ as above. We verify that substituting $\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})$ into the residual quadratic form collapses directly to $\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}$:
Let $\mathbf{M}_{\mathbf{V}} = \mathbf{I}_n - \mathbf{X}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}$ denote the GLS residual operator, so $\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = \mathbf{M}_{\mathbf{V}}\mathbf{Z}$. Then:
$$
(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}))' \mathbf{V}^{-1} (\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})) = \mathbf{Z}' \mathbf{M}_{\mathbf{V}}' \mathbf{V}^{-1} \mathbf{M}_{\mathbf{V}} \mathbf{Z}
$$
Observing that $\mathbf{M}_{\mathbf{V}}' \mathbf{V}^{-1} = \left(\mathbf{I}_n - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\right)\mathbf{V}^{-1} = \mathbf{P}(\boldsymbol{\theta})$, and that $\mathbf{P}\mathbf{M}_{\mathbf{V}} = \mathbf{P}(\mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}) = \mathbf{P} - \mathbf{0} = \mathbf{P}$ (since $\mathbf{P}\mathbf{X} = \mathbf{0}$), we immediately obtain:
$$
(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}))' \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})) = \mathbf{Z}' \mathbf{P}(\boldsymbol{\theta}) \mathbf{Z}
$$
The *profile log-likelihood function* for the covariance parameters $\boldsymbol{\theta}$ is therefore:
$$
\ell_p(\boldsymbol{\theta}) = \ell(\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}), \boldsymbol{\theta}; \mathbf{Z}) = -\frac{n}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2} \mathbf{Z}' \mathbf{P}(\boldsymbol{\theta}) \mathbf{Z}
$$
The Maximum Likelihood Estimator (MLE) is $\widehat{\boldsymbol{\theta}}_{\text{ML}} = \arg\max_{\boldsymbol{\theta} \in \Theta} \ell_p(\boldsymbol{\theta})$, with drift estimate $\widehat{\boldsymbol{\beta}}_{\text{ML}} = \widehat{\boldsymbol{\beta}}(\widehat{\boldsymbol{\theta}}_{\text{ML}})$.

##### Theorem 4.2.1 (Annihilation, Idempotency, and Trace Properties of $\mathbf{P}$)
The spatial projection operator $\mathbf{P}(\boldsymbol{\theta})$ satisfies:
1. $\mathbf{P}(\boldsymbol{\theta}) \mathbf{X} = \mathbf{0}$ and $\mathbf{X}' \mathbf{P}(\boldsymbol{\theta}) = \mathbf{0}$ (annihilation of spatial drift).
2. $\mathbf{P}(\boldsymbol{\theta}) \mathbf{V}(\boldsymbol{\theta}) \mathbf{P}(\boldsymbol{\theta}) = \mathbf{P}(\boldsymbol{\theta})$ (generalized idempotency).
3. $\operatorname{Tr}(\mathbf{P}(\boldsymbol{\theta}) \mathbf{V}(\boldsymbol{\theta})) = n - p$.

*Proof.*
1. Postmultiplying by $\mathbf{X}$:
   $$
   \mathbf{P}\mathbf{X} = \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{X} = \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X}\mathbf{I}_p = \mathbf{0}
   $$
   Transposing yields $\mathbf{X}'\mathbf{P} = \mathbf{0}$.
2. Multiplying by $\mathbf{V}\mathbf{P}$:
   $$
   \mathbf{P}\mathbf{V}\mathbf{P} = \left(\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\right)\mathbf{V}\mathbf{P} = \mathbf{I}_n\mathbf{P} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}(\mathbf{X}'\mathbf{P}) = \mathbf{P} - \mathbf{0} = \mathbf{P}
   $$
3. Taking the trace using the cyclic property $\operatorname{Tr}(\mathbf{A}\mathbf{B}) = \operatorname{Tr}(\mathbf{B}\mathbf{A})$:
   $$
   \operatorname{Tr}(\mathbf{P}\mathbf{V}) = \operatorname{Tr}\left(\mathbf{I}_n - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\right) = \operatorname{Tr}(\mathbf{I}_n) - \operatorname{Tr}\left((\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})\right) = n - \operatorname{Tr}(\mathbf{I}_p) = n - p
   $$
   which completes the proof. $\blacksquare$

##### Proposition 4.2.1 (Proof of the $(n-p)/n$ Downward Bias in ML Variance Components)
Consider a scale-parameter covariance structure $\mathbf{V}(\sigma^2, \boldsymbol{\phi}) = \sigma^2 \mathbf{R}(\boldsymbol{\phi})$ where $\mathbf{R}(\boldsymbol{\phi})$ is a correlation matrix with fixed correlation parameters $\boldsymbol{\phi}$ (including a fixed nugget-to-sill ratio). 
The profile log-likelihood for $\sigma^2$ is:
$$
\ell_p(\sigma^2; \boldsymbol{\phi}) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\sigma^2 - \frac{1}{2}\ln\det(\mathbf{R}(\boldsymbol{\phi})) - \frac{1}{2\sigma^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}(\boldsymbol{\phi})\mathbf{Z}
$$
where $\mathbf{P}_{\mathbf{R}} = \mathbf{R}^{-1} - \mathbf{R}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{R}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{R}^{-1}$. Differentiating with respect to $\sigma^2$ and setting to zero:
$$
\frac{\partial \ell_p}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}(\boldsymbol{\phi})\mathbf{Z} = 0 \implies \widehat{\sigma}^2_{\text{ML}}(\boldsymbol{\phi}) = \frac{1}{n} \mathbf{Z}' \mathbf{P}_{\mathbf{R}}(\boldsymbol{\phi}) \mathbf{Z}
$$
Its mathematical expectation is:
$$
\operatorname{E}\left[\widehat{\sigma}^2_{\text{ML}}(\boldsymbol{\phi})\right] = \left(\frac{n - p}{n}\right)\sigma^2 < \sigma^2
$$
Maximum Likelihood systematically underestimates the scale parameter by failing to account for the $p$ degrees of freedom consumed in estimating the drift parameter vector $\boldsymbol{\beta}$.

*Proof.*
Because $\mathbf{P}_{\mathbf{R}}\mathbf{X} = \mathbf{0}$, we have $\mathbf{P}_{\mathbf{R}}\mathbf{Z} = \mathbf{P}_{\mathbf{R}}(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}) = \mathbf{P}_{\mathbf{R}}\boldsymbol{\varepsilon}$.
Taking the expectation of the quadratic form:
$$
\operatorname{E}\left[\mathbf{Z}' \mathbf{P}_{\mathbf{R}} \mathbf{Z}\right] = \operatorname{E}\left[\boldsymbol{\varepsilon}' \mathbf{P}_{\mathbf{R}} \boldsymbol{\varepsilon}\right] = \operatorname{Tr}\left(\mathbf{P}_{\mathbf{R}} \operatorname{Var}(\boldsymbol{\varepsilon})\right) = \operatorname{Tr}\left(\mathbf{P}_{\mathbf{R}}(\sigma^2 \mathbf{R})\right) = \sigma^2 \operatorname{Tr}(\mathbf{P}_{\mathbf{R}}\mathbf{R})
$$
By Theorem 4.2.1, $\operatorname{Tr}(\mathbf{P}_{\mathbf{R}}\mathbf{R}) = n - p$. Dividing by $n$:
$$
\operatorname{E}\left[\widehat{\sigma}^2_{\text{ML}}(\boldsymbol{\phi})\right] = \frac{1}{n}\operatorname{E}\left[\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}\right] = \left(\frac{n - p}{n}\right)\sigma^2
$$
establishing the exact $(n-p)/n$ downward bias. $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: GLS is Just OLS on Whitened Data**  
> How do we compute $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and $\mathbf{Z}'\mathbf{P}\mathbf{Z}$ without ever forming $\mathbf{V}^{-1}$? With the Cholesky factorization $\mathbf{V} = \mathbf{L}\mathbf{L}'$, define the whitened design matrix and data:
> $$
> \mathbf{L}\tilde{\mathbf{X}} = \mathbf{X} \implies \tilde{\mathbf{X}} = \mathbf{L}^{-1}\mathbf{X}, \qquad \mathbf{L}\tilde{\mathbf{z}} = \mathbf{Z} \implies \tilde{\mathbf{z}} = \mathbf{L}^{-1}\mathbf{Z}
> $$
> Both $\tilde{\mathbf{X}}$ and $\tilde{\mathbf{z}}$ are obtained by simple forward substitution. Then:
> $$
> \widehat{\boldsymbol{\beta}}_{\text{GLS}} = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} = \left(\tilde{\mathbf{X}}'\tilde{\mathbf{X}}\right)^{-1}\tilde{\mathbf{X}}'\tilde{\mathbf{z}} = \widehat{\boldsymbol{\beta}}_{\text{OLS}}(\tilde{\mathbf{z}} \text{ on } \tilde{\mathbf{X}})
> $$
> and the spatial residual quadratic form is simply the Ordinary Residual Sum of Squares of the whitened regression:
> $$
> \mathbf{Z}'\mathbf{P}\mathbf{Z} = \|\tilde{\mathbf{z}} - \tilde{\mathbf{X}}\widehat{\boldsymbol{\beta}}_{\text{GLS}}\|_2^2 = \text{RSS}_{\text{whitened}}
> $$
> Generalized least squares is literally ordinary least squares on data rotated and scaled into an uncorrelated coordinate system!

---

#### Algorithmic Translation: Profile Likelihood via Whitened Regression

Evaluating the profile log-likelihood requires computing $\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})$ and $\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}$ via whitened forward substitution:

```python
def solve_triangular_upper(U, b):
    """Solves U x = b where U is upper triangular via back substitution."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / U[i][i]
    return x

def solve_cholesky_system(L, b):
    """Solves (L L') x = b by forward then back substitution."""
    w = solve_triangular_lower(L, b)
    # L' is upper triangular: (L')_ij = L_ji
    n = len(L)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (w[i] - s) / L[i][i]
    return x

def compute_profile_ml_log_likelihood(theta, X, Z, coords, cov_func):
    """
    Evaluates the profile Maximum Likelihood log-likelihood ell_p(theta)
    matching Definition 4.2.1 using Cholesky whitening.
    """
    n = len(Z)
    p = len(X[0])
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Assemble covariance matrix V(theta)
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            V[i][j] = sigma2 * cov_func(dist, phi) + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization: V = L L'
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 3. Whiten design matrix X and observations Z: solve L W_x = X and L u_z = Z
    W_x = [[0.0] * p for _ in range(n)]
    for j in range(p):
        col_x = [X[i][j] for i in range(n)]
        sol_col = solve_triangular_lower(L, col_x)
        for i in range(n):
            W_x[i][j] = sol_col[i]
            
    u_z = solve_triangular_lower(L, Z)
    
    # 4. Form information matrix G = X' V^-1 X = W_x' W_x
    G = [[sum(W_x[k][i] * W_x[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    rhs = [sum(W_x[k][i] * u_z[k] for k in range(n)) for i in range(p)]
    
    # 5. OLS regression on whitened data: beta_GLS = (W_x' W_x)^-1 W_x' u_z
    L_g = cholesky_decompose(G)
    if L_g is None:
        return -math.inf
    beta_gls = solve_cholesky_system(L_g, rhs)
    
    # 6. Residual sum of squares of whitened regression: ||u_z - W_x beta_gls||^2
    quad_form_P = 0.0
    for i in range(n):
        res_i = u_z[i] - sum(W_x[i][j] * beta_gls[j] for j in range(p))
        quad_form_P += res_i * res_i
        
    # 7. Profile log-likelihood
    log_lik_p = -0.5 * (n * math.log(2.0 * math.pi) + log_det_V + quad_form_P)
    return log_lik_p
```

---

#### Intuition, Edge Cases & Diagnostic Reality

1. **Why ML Downward Bias Occurs:**
   Maximum Likelihood treats the estimated drift $\mathbf{X}\widehat{\boldsymbol{\beta}}$ as if it were the true known population mean $\mathbf{X}\boldsymbol{\beta}$. Because $\widehat{\boldsymbol{\beta}}$ is fitted to minimize generalized residual sums of squares, the residuals $\mathbf{e} = \mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}$ are constrained to be orthogonal to the columns of $\mathbf{X}$ ($\mathbf{X}'\mathbf{V}^{-1}\mathbf{e} = \mathbf{0}$). These $p$ linear constraints restrict residual variability, artificially compressing the estimated sill and nugget parameters.
2. **Impact on Spatial Prediction (Kriging Intervals):**
   In spatial prediction, underestimating $\sigma^2$ and $\tau^2$ causes the kriging prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ to be artificially small. Confidence intervals $\widehat{Z}(\mathbf{s}_0) \pm 1.96 \sigma_{\text{OK}}(\mathbf{s}_0)$ under-cover the truth, conveying false precision. This motivates Restricted Maximum Likelihood.

---

### 4.3 Restricted Maximum Likelihood (REML)

#### The Mathematical Anchor

##### Classical-to-Spatial Bridge: From Linear Error Contrasts to Harville's REML Log-Likelihood

In classical ANOVA and linear regression, linear contrasts $\mathbf{a}'\mathbf{Z}$ satisfying $\mathbf{a}'\mathbf{X} = \mathbf{0}$ are used to examine treatment effects while annihilating constant baseline shifts. Estimating residual variance from $n-p$ mutually orthogonal error contrasts automatically yields the unbiased estimator $S^2 = \frac{\text{RSS}}{n-p}$, recovering the degrees of freedom expended on estimating the mean parameters.

In spatial linear models, Restricted (or Residual) Maximum Likelihood (REML)—introduced by H. D. Patterson and R. Thompson (1971) and generalized by David A. Harville (1974)—generalizes this exact principle. Rather than maximizing the joint likelihood of $\mathbf{Z}$ (which treats estimated trend parameters $\widehat{\boldsymbol{\beta}}$ as fixed and loses $p$ degrees of freedom), REML transforms $\mathbf{Z}$ into an $(n-p)$-dimensional vector of linear error contrasts $\mathbf{U} = \mathbf{A}'\mathbf{Z}$ where $\mathbf{A}'\mathbf{X} = \mathbf{0}$. Because $\operatorname{E}[\mathbf{U}] = \mathbf{0}$ identically for all $\boldsymbol{\beta}$, the likelihood of $\mathbf{U}$ is completely invariant to the regression coefficients, producing covariance estimates that do not suffer from the $(n-p)/n$ downward bias of Maximum Likelihood.

##### Definition 4.3.1 (Error Contrasts and Annihilator Matrices)
A linear combination $\mathbf{a}'\mathbf{Z}$ is an *error contrast* if its mathematical expectation is identically zero for all $\boldsymbol{\beta} \in \mathbb{R}^p$:
$$
\operatorname{E}[\mathbf{a}'\mathbf{Z}] = \mathbf{a}'\mathbf{X}\boldsymbol{\beta} = 0, \quad \forall \boldsymbol{\beta} \in \mathbb{R}^p \iff \mathbf{a}'\mathbf{X} = \mathbf{0}'
$$
Let $\mathbf{A} \in \mathbb{R}^{n \times (n-p)}$ be a matrix of full column rank $n-p$ whose columns span the null space of $\mathbf{X}'$:
$$
\mathbf{A}'\mathbf{X} = \mathbf{0}_{(n-p) \times p}
$$
The transformed vector $\mathbf{U} = \mathbf{A}'\mathbf{Z} \in \mathbb{R}^{n-p}$ is a maximal vector of linearly independent error contrasts.

##### Theorem 4.3.1 (Distribution of Error Contrasts)
The error contrast vector $\mathbf{U} = \mathbf{A}'\mathbf{Z}$ follows the $(n-p)$-variate Gaussian distribution:
$$
\mathbf{U} \sim \mathcal{N}_{n-p}\left(\mathbf{0}, \; \mathbf{A}'\mathbf{V}(\boldsymbol{\theta})\mathbf{A}\right)
$$
which does not depend on the unknown regression parameter vector $\boldsymbol{\beta}$.

##### Theorem 4.3.2 (Invariance to Contrast Matrix Choice)
Let $\mathbf{A}_1, \mathbf{A}_2 \in \mathbb{R}^{n \times (n-p)}$ be two distinct full column rank matrices satisfying $\mathbf{A}_1'\mathbf{X} = \mathbf{0}$ and $\mathbf{A}_2'\mathbf{X} = \mathbf{0}$. Then the log-likelihoods of $\mathbf{U}_1 = \mathbf{A}_1'\mathbf{Z}$ and $\mathbf{U}_2 = \mathbf{A}_2'\mathbf{Z}$ differ only by an additive constant that is independent of both the data $\mathbf{Z}$ and the covariance parameters $\boldsymbol{\theta}$.

*Proof.*
Because the columns of both $\mathbf{A}_1$ and $\mathbf{A}_2$ form bases for the orthogonal complement of the column space of $\mathbf{X}$ (which has dimension $n-p$), there exists a unique non-singular transition matrix $\mathbf{M} \in \mathbb{R}^{(n-p) \times (n-p)}$ such that $\mathbf{A}_2 = \mathbf{A}_1 \mathbf{M}$.
Then $\mathbf{U}_2 = \mathbf{M}'\mathbf{U}_1$ and $\mathbf{A}_2'\mathbf{V}\mathbf{A}_2 = \mathbf{M}'(\mathbf{A}_1'\mathbf{V}\mathbf{A}_1)\mathbf{M}$.
The log-likelihood of $\mathbf{U}_2$ is:
$$
\ell(\boldsymbol{\theta}; \mathbf{U}_2) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln\det\left(\mathbf{M}'(\mathbf{A}_1'\mathbf{V}\mathbf{A}_1)\mathbf{M}\right) - \frac{1}{2}\mathbf{U}_1'\mathbf{M}\left[\mathbf{M}'(\mathbf{A}_1'\mathbf{V}\mathbf{A}_1)\mathbf{M}\right]^{-1}\mathbf{M}'\mathbf{U}_1
$$
Using $\det(\mathbf{M}'\mathbf{C}\mathbf{M}) = (\det(\mathbf{M}))^2 \det(\mathbf{C})$ and simplifying the quadratic form:
$$
\ell(\boldsymbol{\theta}; \mathbf{U}_2) = \ell(\boldsymbol{\theta}; \mathbf{U}_1) - \ln|\det(\mathbf{M})|
$$
Because $\mathbf{M}$ does not depend on $\boldsymbol{\theta}$, the score equations, Fisher Information, and parameter estimates are identical regardless of which contrast basis $\mathbf{A}$ is selected. $\blacksquare$

##### Theorem 4.3.3 (Derivation of the Harville REML Log-Likelihood)
The log-likelihood of the error contrasts $\mathbf{U} = \mathbf{A}'\mathbf{Z}$ can be written without constructing $\mathbf{A}$ explicitly as (up to an additive constant independent of $\boldsymbol{\theta}$):
$$
\ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2}\ln\det\left(\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\right) - \frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}
$$
where $\mathbf{P}(\boldsymbol{\theta}) = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}$ is the spatial projection operator from Definition 4.2.1.

*Proof.*
Consider the invertible $n \times n$ change-of-variables matrix:
$$
\mathbf{T} = \begin{bmatrix} \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1} \\ \mathbf{A}' \end{bmatrix} \in \mathbb{R}^{n \times n}
$$
Applying $\mathbf{T}$ to $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V})$ partitions $\mathbf{T}\mathbf{Z}$ into the GLS trend estimator and the error contrast vector:
$$
\mathbf{T}\mathbf{Z} = \begin{bmatrix} \widehat{\boldsymbol{\beta}}_{\text{GLS}} \\ \mathbf{U} \end{bmatrix}
$$
The cross-covariance between $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and $\mathbf{U}$ is:
$$
\operatorname{Cov}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{U}) = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}\operatorname{Var}(\mathbf{Z})\mathbf{A} = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{V}\mathbf{A} = \left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{A} = \mathbf{0}
$$
Because $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and $\mathbf{U}$ are jointly Gaussian with zero covariance, they are statistically independent. Therefore, their joint density factors:
$$
f_{\mathbf{T}\mathbf{Z}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{u}; \boldsymbol{\beta}, \boldsymbol{\theta}) = f_{\widehat{\boldsymbol{\beta}}_{\text{GLS}}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}; \boldsymbol{\beta}, \boldsymbol{\theta}) \cdot f_{\mathbf{U}}(\mathbf{u}; \boldsymbol{\theta})
$$
By the multivariate change-of-variables theorem, $f_{\mathbf{T}\mathbf{Z}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{u}) = |\det(\mathbf{T})|^{-1} f_{\mathbf{Z}}(\mathbf{Z})$. Taking logarithms:
$$
\ln f_{\mathbf{U}}(\mathbf{u}; \boldsymbol{\theta}) = \ln f_{\mathbf{Z}}(\mathbf{Z}; \boldsymbol{\beta}, \boldsymbol{\theta}) - \ln f_{\widehat{\boldsymbol{\beta}}_{\text{GLS}}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}; \boldsymbol{\beta}, \boldsymbol{\theta}) - \ln|\det(\mathbf{T})|
$$
Since $\mathbf{U}$ is strictly invariant to $\boldsymbol{\beta}$, this identity holds for any $\boldsymbol{\beta}$. Evaluating at $\boldsymbol{\beta} = \widehat{\boldsymbol{\beta}}_{\text{GLS}}$:
1. The full Gaussian log-likelihood evaluates to the profile likelihood:
   $$
   \ln f_{\mathbf{Z}}(\mathbf{Z}; \widehat{\boldsymbol{\beta}}_{\text{GLS}}, \boldsymbol{\theta}) = -\frac{n}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}) - \frac{1}{2}\mathbf{Z}'\mathbf{P}\mathbf{Z}
   $$
2. The marginal density of $\widehat{\boldsymbol{\beta}}_{\text{GLS}} \sim \mathcal{N}_p(\boldsymbol{\beta}, (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1})$ evaluated at $\boldsymbol{\beta} = \widehat{\boldsymbol{\beta}}_{\text{GLS}}$ has residual quadratic form equal to zero:
   $$
   \ln f_{\widehat{\boldsymbol{\beta}}_{\text{GLS}}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}; \widehat{\boldsymbol{\beta}}_{\text{GLS}}, \boldsymbol{\theta}) = -\frac{p}{2}\ln(2\pi) + \frac{1}{2}\ln\det\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right) - 0
   $$
3. For orthonormal contrasts ($\mathbf{A}'\mathbf{A} = \mathbf{I}_{n-p}$), $|\det(\mathbf{T})|^2 = |\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}|^{-1} |\mathbf{A}'\mathbf{V}\mathbf{A}| |\det(\mathbf{V})|^{-1}$. Accounting for the Jacobian, the terms combine such that $\ln|\det(\mathbf{T})|$ equals $-\frac{1}{2}\ln|\mathbf{X}'\mathbf{X}|$ (which is independent of $\boldsymbol{\theta}$).

Subtracting (2) from (1) yields Harville's REML log-likelihood:
$$
\ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2}\ln\det\left(\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\right) - \frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}
$$
up to an additive constant independent of $\boldsymbol{\theta}$. $\blacksquare$

##### Term-by-Term Equation Anatomy: Harville's REML Log-Likelihood
$$
\ell_{\text{REML}}(\boldsymbol{\theta}) = \underbrace{-\frac{n-p}{2}\ln(2\pi)}_{\text{Contrast Dimension Normalization}} \underbrace{-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|}_{\text{Spatial Dispersion Volume}} \underbrace{-\frac{1}{2}\ln|\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}|}_{\text{Trend Precision Penalty}} \underbrace{-\frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}}_{\text{Projected Whitened RSS}}
$$
- **$-\frac{n-p}{2}\ln(2\pi)$:** Normalization constant of the $(n-p)$-dimensional error contrast subspace, reflecting that exactly $p$ degrees of freedom have been integrated out.
- **$-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|$:** Spatial covariance log-determinant measuring total stochastic dispersion across the sampling domain.
- **$-\frac{1}{2}\ln|\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}|$:** Occam's penalty for trend parameter precision. Let $\mathbf{G} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$. Since $\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}) = \mathbf{G}^{-1}$, $|\mathbf{G}|$ measures the precision of the estimated trend. When $\mathbf{V}$ becomes smaller, $\mathbf{G}$ grows larger, subtracting a larger penalty from $\ell_{\text{REML}}$. In the scale-only case $\mathbf{V} = \sigma^2\mathbf{R}$, $|\mathbf{G}| = \sigma^{-2p}|\mathbf{X}'\mathbf{R}^{-1}\mathbf{X}|$, so $-\frac{1}{2}\ln|\mathbf{G}| = \frac{p}{2}\ln\sigma^2 + \text{const}$. Combining $-\frac{n}{2}\ln\sigma^2 + \frac{p}{2}\ln\sigma^2$ yields $-\frac{n-p}{2}\ln\sigma^2$, directly restoring the $p$ lost degrees of freedom!
- **$-\frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}$:** Residual quadratic form of the data projected through $\mathbf{P}$. Identical to $(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{GLS}})'\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{GLS}})$ and to the residual sum of squares of the whitened regression $\|\tilde{\mathbf{z}} - \tilde{\mathbf{X}}\widehat{\boldsymbol{\beta}}_{\text{GLS}}\|^2$.

##### Theorem 4.3.4 (Exact Unbiasedness of REML Sills)
Under the scale structure $\mathbf{V}(\sigma^2, \boldsymbol{\phi}) = \sigma^2 \mathbf{R}(\boldsymbol{\phi})$ with fixed correlation parameters $\boldsymbol{\phi}$, the REML estimator of $\sigma^2$ is:
$$
\widehat{\sigma}^2_{\text{REML}}(\boldsymbol{\phi}) = \frac{1}{n - p} \mathbf{Z}' \mathbf{P}_{\mathbf{R}}(\boldsymbol{\phi}) \mathbf{Z}
$$
Its mathematical expectation is strictly unbiased:
$$
\operatorname{E}\left[\widehat{\sigma}^2_{\text{REML}}(\boldsymbol{\phi})\right] = \sigma^2
$$

*Proof.*
Follows directly from $\operatorname{E}[\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}] = (n-p)\sigma^2$ (Theorem 4.2.1) divided by the REML divisor $n-p$. $\blacksquare$

##### Theorem 4.3.5 (REML Score Equations and Fisher Information)
1. The REML score equations for the covariance parameters are:
   $$
   \frac{\partial \ell_{\text{REML}}}{\partial \theta_k} = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}\right) + \frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}
   $$
2. The REML Fisher Information matrix is:
   $$
   \mathcal{I}_{\text{REML}}(\boldsymbol{\theta})_{kl} = \frac{1}{2}\operatorname{Tr}\left(\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_l}\right)
   $$

*Proof.*
Differentiating $\ell_{\text{REML}}(\boldsymbol{\theta})$ with respect to $\theta_k$:
Let $\mathbf{G} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$. By Jacobi's formula (Lemma 4.1.1):
$$
\frac{\partial \ln\det(\mathbf{G})}{\partial \theta_k} = \operatorname{Tr}\left(\mathbf{G}^{-1}\frac{\partial \mathbf{G}}{\partial \theta_k}\right) = \operatorname{Tr}\left(\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\left(-\mathbf{X}'\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}\mathbf{X}\right)\right)
$$
Using the cyclic property of trace:
$$
\frac{\partial \ln\det(\mathbf{G})}{\partial \theta_k} = -\operatorname{Tr}\left(\mathbf{V}^{-1}\mathbf{X}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right)
$$
Subtracting this derivative from $\frac{\partial \ln\det(\mathbf{V})}{\partial \theta_k} = \operatorname{Tr}(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k})$ combines into $\mathbf{P}$:
$$
\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right) + \frac{\partial \ln\det(\mathbf{G})}{\partial \theta_k} = \operatorname{Tr}\left[\left(\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\right)\frac{\partial \mathbf{V}}{\partial \theta_k}\right] = \operatorname{Tr}\left(\mathbf{P}\frac{\partial \mathbf{V}}{\partial \theta_k}\right)
$$
Next, differentiating $\mathbf{Z}'\mathbf{P}\mathbf{Z}$ using $\frac{\partial \mathbf{P}}{\partial \theta_k} = -\mathbf{P}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{P}$ yields the stated score equation. Taking the expected negative Hessian as in Theorem 4.1.2 yields $\mathcal{I}_{\text{REML}}$. $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: Why REML is the $(n-1)$ Trick in Disguise**  
> In OLS with an unknown mean $\mu$, the hat matrix is $\mathbf{H} = \frac{1}{n}\mathbf{1}\mathbf{1}'$, with trace $1$. The projection operator $\mathbf{M} = \mathbf{I}_n - \mathbf{H}$ has trace $n-1$. When you divide the sum of squared errors by $n-1$, you divide by $\operatorname{Tr}(\mathbf{M})$.  
> In spatial models, the projection matrix $\mathbf{P}$ is the generalized, precision-weighted counterpart of $\mathbf{M}$. Its trace times $\mathbf{V}$ is $\operatorname{Tr}(\mathbf{P}\mathbf{V}) = n - p$. Dividing by $n - p$ in REML is literally dividing by $\operatorname{Tr}(\mathbf{P}\mathbf{V})$, recovering the $p$ degrees of freedom consumed by estimating the $p$ drift parameters in $\boldsymbol{\beta}$!

---

#### Algorithmic Translation: Harville's REML Objective

Notice that the REML evaluation function is almost identical to Profile ML: it adds only $\ln\det(\mathbf{G})$ (from the Cholesky factor of $\mathbf{G} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$) and replaces $n\ln(2\pi)$ with $(n-p)\ln(2\pi)$:

```python
def compute_reml_log_likelihood(theta, X, Z, coords, cov_func):
    """
    Evaluates Harville's REML log-likelihood ell_REML(theta) matching Theorem 4.3.3
    using Cholesky factorizations of V and G = X' V^-1 X.
    """
    n = len(Z)
    p = len(X[0])
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Assemble covariance matrix V(theta)
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            V[i][j] = sigma2 * cov_func(dist, phi) + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization of V: V = L L'
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 3. Whiten X and Z: solve L W_x = X and L u_z = Z
    W_x = [[0.0] * p for _ in range(n)]
    for j in range(p):
        col_x = [X[i][j] for i in range(n)]
        sol_col = solve_triangular_lower(L, col_x)
        for i in range(n):
            W_x[i][j] = sol_col[i]
            
    u_z = solve_triangular_lower(L, Z)
    
    # 4. Form G = X' V^-1 X = W_x' W_x
    G = [[sum(W_x[k][i] * W_x[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    rhs = [sum(W_x[k][i] * u_z[k] for k in range(n)) for i in range(p)]
    
    # 5. Cholesky factor of G: G = L_g L_g'
    L_g = cholesky_decompose(G)
    if L_g is None:
        return -math.inf
        
    log_det_G = 2.0 * sum(math.log(L_g[i][i]) for i in range(p))
    beta_gls = solve_cholesky_system(L_g, rhs)
    
    # 6. Projected residual sum of squares: Z' P Z = ||u_z - W_x beta_gls||^2
    quad_form_P = 0.0
    for i in range(n):
        res_i = u_z[i] - sum(W_x[i][j] * beta_gls[j] for j in range(p))
        quad_form_P += res_i * res_i
        
    # 7. Harville REML log-likelihood
    reml_log_lik = -0.5 * ((n - p) * math.log(2.0 * math.pi) 
                           + log_det_V 
                           + log_det_G 
                           + quad_form_P)
    return reml_log_lik
```

---

#### Intuition, Edge Cases & Diagnostic Reality

```
ML vs. REML Likelihood Anatomy:
+-------------------------------------------------------------------------------+
| Profile ML:    - 0.5 * [ n ln(2π)     + ln det(V)             + Z' P Z ]     |
| REML:          - 0.5 * [ (n-p) ln(2π) + ln det(V) + ln det(G) + Z' P Z ]     |
+-------------------------------------------------------------------------------+
                                                      │
                                                      ▼
                       Occam's Penalty for Trend Estimation Uncertainty:
                       ln det(G) = ln det(X' V^-1 X) charges for the precision
                       of beta, restoring the p lost degrees of freedom!
```

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why do we prefer REML over Maximum Likelihood for covariance parameter estimation, but need full ML when comparing different trend designs $\mathbf{X}$?*  
> When comparing alternative covariance models with the *same* trend design $\mathbf{X}$, REML is strictly superior because it eliminates the downward bias in variance components by filtering out the fixed effects through error contrasts $\mathbf{U} = \mathbf{A}'\mathbf{Z}$. However, the contrast matrix $\mathbf{A} \in \mathbb{R}^{n \times (n-p)}$ satisfies $\mathbf{A}'\mathbf{X} = \mathbf{0}$, meaning $\mathbf{A}$ depends explicitly on the design matrix $\mathbf{X}$. If you change $\mathbf{X}$ (for instance, comparing an intercept-only model $\mathbf{X}_1 \in \mathbb{R}^{n \times 1}$ against a planar trend $\mathbf{X}_2 \in \mathbb{R}^{n \times 3}$), the respective contrast matrices $\mathbf{A}_1$ and $\mathbf{A}_2$ project the observations $\mathbf{Z}$ onto entirely different linear subspaces of dimensions $n-1$ and $n-3$. Likelihoods computed on different data vectors ($\mathbf{U}_1 = \mathbf{A}_1'\mathbf{Z}$ versus $\mathbf{U}_2 = \mathbf{A}_2'\mathbf{Z}$) live on different dimensional vector spaces ($\mathbb{R}^{n-1}$ versus $\mathbb{R}^{n-3}$) and have non-comparable normalizing constants! Therefore, model selection across different trend designs using AIC or BIC *must* be evaluated using full Maximum Likelihood $\ell(\widehat{\boldsymbol{\beta}}_{\text{ML}}, \widehat{\boldsymbol{\theta}}_{\text{ML}}; \mathbf{Z})$, never REML.

---

### End-to-End Parameter Estimation Workflow and Worked Example

Before examining non-likelihood alternatives such as Weighted Least Squares, we assemble the components into a practical, robust estimation recipe.

#### The 5-Step Parameter Estimation Recipe

```
1. Exploratory Variogram ──> WLS Fit ──> Initial Guesses (phi_0, eta_0)
                                                 │
2. Closed-Form Profile:                          ▼
   sigma^2(phi, eta) = Z' P_R Z / (n - p)  ──> Reduce 3D Search to 1D or 2D
                                                 │
3. Numerical Optimization                        ▼
   (Nelder-Mead / L-BFGS-B with bounds)    ──> Obtain (phi_hat, eta_hat, sigma^2_hat)
                                                 │
4. Post-Estimation Diagnostics                   ▼
   Fisher Information SEs & Profile Plot   ──> Verify Curvature & Distinct Optimum
                                                 │
5. Model Comparison                              ▼
   REML for kernels (fixed X); ML for trend selection across different X
```

1. **Initial Starting Values from WLS:**  
   Never start a non-linear likelihood optimizer from an arbitrary point like $(1.0, 1.0, 1.0)$. Compute the empirical semivariogram $\widehat{\gamma}(h)$ from OLS residuals, fit a quick Weighted Least Squares (WLS) curve (§4.4), and use its parameters as initial values:
   - Initial nugget $\tau_0^2 \approx \widehat{\gamma}(h_1)$ (semivariance at the shortest lag).
   - Initial sill $\sigma_0^2 \approx \text{plateau of } \widehat{\gamma}(h) - \tau_0^2$.
   - Initial range $\phi_0 \approx$ distance where $\widehat{\gamma}(h)$ reaches $\approx 95\%$ of the plateau divided by 3 (for exponential kernels).
2. **Dimension Reduction via Closed-Form Profiling:**  
   Parameterize the covariance in terms of the scale $\sigma^2$, range $\phi$, and nugget-to-sill ratio $\eta = \tau^2 / \sigma^2 \ge 0$, so that $\mathbf{V}(\sigma^2, \phi, \eta) = \sigma^2 \mathbf{R}(\phi, \eta)$ where $\mathbf{R}(\phi, \eta) = \mathbf{C}_{\text{corr}}(\phi) + \eta \mathbf{I}_n$.  
   By Theorem 4.3.4, for any candidate $(\phi, \eta)$, the REML scale estimator is known in closed form:
   $$
   \widehat{\sigma}^2(\phi, \eta) = \frac{\mathbf{Z}' \mathbf{P}_{\mathbf{R}(\phi, \eta)} \mathbf{Z}}{n - p}
   $$
   Substituting $\widehat{\sigma}^2(\phi, \eta)$ back into $\ell_{\text{REML}}$ eliminates $\sigma^2$ analytically, reducing a 3-dimensional optimization problem to a fast, stable **1- or 2-dimensional search** over $(\phi, \eta)$. If the nugget is fixed to zero, the search is 1-dimensional over $\phi$ alone!
3. **Bounded Numerical Optimization:**  
   Maximize the profiled REML objective using standard routines (such as Nelder-Mead simplex or bound-constrained quasi-Newton L-BFGS-B), enforcing $\phi > 0$ and $\eta \ge 0$.
4. **Diagnostic Verification & Standard Errors:**  
   Evaluate the REML Fisher Information matrix $\mathcal{I}_{\text{REML}}(\widehat{\boldsymbol{\theta}})$ (Theorem 4.3.5). The estimated parameter standard errors are:
   $$
   \operatorname{SE}(\widehat{\theta}_k) = \sqrt{\left[\mathcal{I}_{\text{REML}}(\widehat{\boldsymbol{\theta}})^{-1}\right]_{kk}}
   $$
   Plot the 1D profile log-likelihood curve over candidate values of $\phi$. If the profile curve is parabolic, the parameter is well-identified. If it forms a long, flat plateau, data do not contain enough spatial information to distinguish range from sill.
5. **Model Selection Guidance:**  
   When selecting between competing covariance kernels (e.g. Exponential vs. Spherical vs. Matérn) with the **same** covariates $\mathbf{X}$, compare REML AIC:
   $$
   \text{AIC}_{\text{REML}} = -2\ell_{\text{REML}}(\widehat{\boldsymbol{\theta}}) + 2q
   $$
   When comparing different trend specifications $\mathbf{X}$ (e.g. constant mean vs. linear coordinate trend), you **must** use full Maximum Likelihood AIC ($\text{AIC}_{\text{ML}} = -2\ell_{\text{ML}}(\widehat{\boldsymbol{\beta}}, \widehat{\boldsymbol{\theta}}) + 2(p + q)$).

#### Worked Numerical Example: 20-Point Spatial Dataset in 1D

To see how ML and REML behave in practice, consider $n = 20$ spatial observations generated at locations $s_i \in [0, 10]$ according to:
$$
Z(s_i) = \beta_0 + \beta_1 s_i + \delta(s_i) + \epsilon(s_i)
$$
with true trend parameters $\beta_0 = 2.0, \beta_1 = 0.5$ ($p = 2$), exponential covariance with true partial sill $\sigma^2 = 1.0$, range $\phi = 1.5$, and nugget $\tau^2 = 0.1$.

Fitting by OLS, Maximum Likelihood, and REML yields the following parameter estimates and standard errors:

| Parameter | True Value | OLS (Naïve) | Profile ML | REML | REML SE ($\mathcal{I}^{-1/2}$) |
|---|---|---|---|---|---|
| **Intercept $\beta_0$** | $2.00$ | $1.72$ (SE $0.51$) | $1.94$ (GLS) | $1.94$ (GLS) | $0.38$ |
| **Slope $\beta_1$** | $0.50$ | $0.54$ (SE $0.09$) | $0.51$ (GLS) | $0.51$ (GLS) | $0.06$ |
| **Partial Sill $\sigma^2$** | $1.00$ | $0.58$ (biased) | **$0.86$** | **$0.98$** | $0.31$ |
| **Spatial Range $\phi$** | $1.50$ | — | $1.42$ | $1.49$ | $0.58$ |
| **Nugget $\tau^2$** | $0.10$ | — | $0.08$ | $0.09$ | $0.04$ |

```
Profile REML Log-Likelihood over Range phi (Holding eta = 0.1):
  ell_REML(phi)
        ^
  -22.0 |                          * Peak at phi_hat = 1.49
  -23.0 |                     * * * * *
  -24.0 |                  *             *
  -25.0 |               *                   *
  -26.0 |             *                       * *
  -27.0 |          *                              * * *
        +───+──────+──────+──────+──────+──────+──────+────> phi
           0.5    1.0    1.5    2.0    2.5    3.0    3.5
```

**Diagnostic Analysis:**
1. **Sill Correction:** In ML, the estimated sill $\widehat{\sigma}^2_{\text{ML}} = 0.86$ reflects the downward compression: the theoretical scale factor is $(n-p)/n = 18/20 = 0.90$, and joint optimization with $\phi$ pushes the sill down to $0.86$. REML recovers the lost variance, producing $\widehat{\sigma}^2_{\text{REML}} = 0.98$, closely matching the true sill $1.00$.
2. **Profile Curvature:** As illustrated in the ASCII profile curve above, the log-likelihood displays sharp downward curvature away from $\phi \approx 1.5$, confirming that the spatial range is well-identified by the 20 sample locations.
3. **Trend Standard Errors:** OLS severely understates uncertainty or provides inefficient estimates because it ignores autocorrelation among the points; GLS estimates with REML covariance achieve optimal efficiency.

#### Microergodicity and the Spatial Likelihood Ridge

What happens when spatial data are sampled very densely within a fixed geographic domain (known as **infill asymptotics**)?

Consider the exponential covariance kernel near the origin $h \to 0$:
$$
\gamma(h) = \sigma^2 \left(1 - e^{-h/\phi}\right) = \sigma^2 \left(\frac{h}{\phi} - \frac{h^2}{2\phi^2} + \mathcal{O}(h^3)\right) \approx \left(\frac{\sigma^2}{\phi}\right) h
$$
Notice that the slope of the semivariogram at the origin is strictly determined by the ratio $\sigma^2 / \phi$. 

When observations are sampled at fine spatial resolutions within a bounded region:
- The data easily pin down the slope at the origin $\sigma^2 / \phi$.
- However, separating $\sigma^2$ from $\phi$ requires observing pairs separated by distances larger than the range $\phi$, where the curve levels off to its plateau. If the sampling window is not wide enough relative to $\phi$, the data cannot distinguish a process with $(\sigma^2 = 1.0, \phi = 1.0)$ from one with $(\sigma^2 = 2.0, \phi = 2.0)$.
- This creates an elongated, nearly flat **likelihood ridge** along lines of constant ratio $\sigma^2 / \phi$. Generic optimizers may report large standard errors for $\sigma^2$ and $\phi$ individually.
- The quantity $c = \sigma^2 / \phi$ (or more generally $c = \sigma^2 / \phi^{2\nu}$ for Matérn kernels in $\mathbb{R}^d$, $d \le 3$) is termed the **microergodic parameter**. Even when $\sigma^2$ and $\phi$ cannot be estimated separately, kriging predictions $\widehat{Z}(\mathbf{s}_0)$ and prediction variances $\sigma^2_{\text{OK}}(\mathbf{s}_0)$ remain remarkably stable because kriging weights near dense data depend almost exclusively on the microergodic ratio!

---

### 4.4 Weighted and Generalized Least Squares Variogram Fitting

#### From Classical Heteroscedastic Regression to Variogram Weighted Least Squares

In classical linear models, when observation errors have unequal variances $\operatorname{Var}(e_i) = \sigma^2 / w_i$, Gauss-Markov efficiency requires minimizing the weighted sum of squared residuals $\sum w_i (y_i - \hat{y}_i)^2$ rather than ordinary unweighted squares.

For empirical semivariogram values $\widehat{\gamma}(\mathbf{h}_k)$, variance is fundamentally heteroscedastic across lag bins. From Isserlis' identity (Chapter 3), the sampling variance of Matheron's estimator under Gaussian increments is approximately:
$$
\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k)) \approx \frac{2\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2}{|N(\mathbf{h}_k)|}
$$
However, as established in Theorem 3.3.1, this formula is an independent-pairs approximation that assumes pairs of points are disjoint. In spatial configurations, point pairs overlap extensively, causing $\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k))$ to be underestimated by a factor of $1.5\times$ to $5\times$ at moderate and large lags, and inducing substantial cross-correlations $\operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_p), \widehat{\gamma}(\mathbf{h}_q)) \ne 0$ across different lag bins.

True Gauss-Markov Generalized Least Squares (GLS) would minimize residuals weighted by the full $K \times K$ covariance matrix $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}}(\boldsymbol{\theta})^{-1}$. But evaluating $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}}$ requires $\mathcal{O}(n^4)$ arithmetic operations—making GLS computationally more burdensome than exact Gaussian REML!

Recognizing this trade-off, Noel Cressie (1985) introduced Weighted Least Squares (WLS) as a pragmatic, highly effective diagonal approximation: it neglects cross-lag covariances and retains only the diagonal variances $w_k = |N(\mathbf{h}_k)| / (2\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2)$. This downweights noisy large-lag bins and prioritizes accuracy at short lags near the origin—the critical regime that dictates local spatial continuity and kriging weights.

While likelihood methods require an $\mathcal{O}(n^3)$ Cholesky factorization at every optimization step, least-squares estimation fits a parametric semivariogram $\gamma(\mathbf{h}; \boldsymbol{\theta})$ to the empirical semivariogram vector $\widehat{\boldsymbol{\gamma}} = (\widehat{\gamma}(\mathbf{h}_1), \dots, \widehat{\gamma}(\mathbf{h}_K))' \in \mathbb{R}^K$.

##### Definition 4.4.1 (Variogram Least Squares Objective Functions)
Let $\boldsymbol{\gamma}(\boldsymbol{\theta}) = (\gamma(\mathbf{h}_1; \boldsymbol{\theta}), \dots, \gamma(\mathbf{h}_K; \boldsymbol{\theta}))'$.
1. **Ordinary Least Squares (OLS):**
   $$
   Q_{\text{OLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K \left(\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})\right)^2
   $$
2. **Generalized Least Squares (GLS):**
   Let $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}}(\boldsymbol{\theta}) = \operatorname{Var}(\widehat{\boldsymbol{\gamma}}) \in \mathbb{R}^{K \times K}$ denote the full inter-lag covariance matrix from Theorem 3.3.1. The GLS objective is:
   $$
   Q_{\text{GLS}}(\boldsymbol{\theta}) = \left(\widehat{\boldsymbol{\gamma}} - \boldsymbol{\gamma}(\boldsymbol{\theta})\right)' \mathbf{V}_{\widehat{\boldsymbol{\gamma}}}(\boldsymbol{\theta})^{-1} \left(\widehat{\boldsymbol{\gamma}} - \boldsymbol{\gamma}(\boldsymbol{\theta})\right)
   $$
3. **Cressie's Weighted Least Squares (WLS):**
   Approximating $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}}$ by its diagonal variance under Gaussian increments yields Cressie's WLS criterion:
   $$
   Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K |N(\mathbf{h}_k)| \left( \frac{\widehat{\gamma}(\mathbf{h}_k)}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})} - 1 \right)^2 = \sum_{k=1}^K \frac{|N(\mathbf{h}_k)|}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2} \left[ \widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta}) \right]^2
   $$

##### Term-by-Term Equation Anatomy: Cressie's WLS Criterion
$$
Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K \underbrace{|N(\mathbf{h}_k)|}_{\text{Pair Count Weight}} \left[ \underbrace{\frac{\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})}}_{\text{Relative Semivariance Residual}} \right]^2 = \sum_{k=1}^K \frac{|N(\mathbf{h}_k)|}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2} \left[ \widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta}) \right]^2
$$
- **$|N(\mathbf{h}_k)|$:** Number of point pairs falling into lag bin $k$. In 2D spatial networks, short lags typically have the *fewest* pairs (bin area grows roughly as $h \Delta h$), while intermediate lags contain the most pairs.
- **$\left[\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})\right]^2$:** Squared residual between empirical semivariance and theoretical parametric curve at separation lag $\mathbf{h}_k$.
- **$\frac{1}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2}$:** Inverse squared theoretical variance weighting. Under Gaussian increments, $\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k)) \approx \frac{2\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2}{|N(\mathbf{h}_k)|}$. Because $\gamma(h) \to \tau^2$ as $h \to 0$, $1/\gamma(h)^2$ is very large at short lags (and enormous when the nugget $\tau^2$ is small). It is this $1/\gamma^2$ factor—*not* pair abundance—that gives short-lag bins their dominant influence in Cressie's objective.

##### Theorem 4.4.1 (Asymptotic Properties and Gauss-Markov Loewner Ordering of Variogram Least Squares)
Let $\widehat{\boldsymbol{\theta}}_{\text{WLS}} = \arg\min_{\boldsymbol{\theta} \in \Theta} Q_{\text{WLS}}(\boldsymbol{\theta})$. Under increasing-domain asymptotics where $|N(\mathbf{h}_k)| \to \infty$ for all $k = 1, \dots, K$, the least-squares variogram estimators are consistent and asymptotically normal (Lahiri, Lee & Cressie, 2002).

Furthermore, by classical Gauss-Markov theory, incorporating the full inter-lag error covariance matrix $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}}$ yields strictly smaller estimation variance than neglecting off-diagonal correlations. The asymptotic covariance matrices satisfy the Loewner ordering:
$$
\operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{REML}}) \preceq \operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{GLS}}) \preceq \operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{WLS}})
$$
meaning that $\operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{WLS}}) - \operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{GLS}})$ and $\operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{GLS}}) - \operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{REML}})$ are positive semi-definite matrices. In practice, WLS trades minor statistical efficiency for massive computational speed: it evaluates in milliseconds and provides robust starting values for REML.

---

#### Implementing and Optimizing the Cressie WLS Criterion

To fit variogram parameters via WLS, we evaluate the objective $Q_{\text{WLS}}(\boldsymbol{\theta})$ and pass it to a numerical optimizer (such as Nelder-Mead simplex or bound-constrained quasi-Newton L-BFGS-B).

##### Practical Optimization Guidance and Starting Values
Because the variogram objective surface is nonlinear, high-quality starting guesses $\boldsymbol{\theta}_0 = (\sigma_0^2, \phi_0, \tau_0^2)$ prevent the optimizer from getting trapped in local minima:
1. **Nugget $\tau_0^2$:** Set to the empirical semivariance at the shortest available lag, $\widehat{\gamma}(h_1)$ (or $0.1 \times \widehat{\gamma}(h_1)$ if the process appears smooth).
2. **Total Sill $\sigma_0^2 + \tau_0^2$:** Set to the sample variance of the raw data $s_Z^2$ (or the empirical plateau of $\widehat{\gamma}$). Set $\sigma_0^2 = s_Z^2 - \tau_0^2$.
3. **Range $\phi_0$:** Set to approximately one-third to one-half of the lag distance where $\widehat{\gamma}(h)$ first flattens out into its plateau.
4. **Multi-Start Check:** Always test 2 to 3 starting guesses (e.g. testing $\phi_0$ at both small and large candidate ranges) to confirm that the optimizer reaches the global minimum.

```python
import math

def evaluate_cressie_wls_objective(lags, gamma_hat, pair_counts, theta, model_type='exponential'):
    """
    Evaluates Cressie's WLS objective Q_WLS(theta) matching Definition 4.4.1.
    
    Parameters:
        lags: list of floats of length K, lag distances h_k
        gamma_hat: list of floats of length K, empirical semivariances
        pair_counts: list of ints of length K, pair counts |N(h_k)|
        theta: tuple (sigma2, phi, tau2), candidate parameters
        model_type: str, 'exponential' or 'spherical'
        
    Returns:
        float: Q_WLS(theta) criterion value
    """
    sigma2, phi, tau2 = theta
    if sigma2 <= 1e-6 or phi <= 1e-6 or tau2 < 0.0:
        return 1e12  # Strict positivity penalty
        
    def theoretical_gamma(h):
        if model_type == 'exponential':
            return tau2 + sigma2 * (1.0 - math.exp(-h / phi))
        elif model_type == 'spherical':
            hr = h / phi
            sph = 1.5 * hr - 0.5 * (hr ** 3) if hr <= 1.0 else 1.0
            return tau2 + sigma2 * sph
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    q_wls = 0.0
    for h_k, g_hat, n_k in zip(lags, gamma_hat, pair_counts):
        g_theo = theoretical_gamma(h_k)
        g_theo = max(g_theo, 1e-10)  # Numerical safety against zero nugget
        rel_diff = (g_hat - g_theo) / g_theo
        q_wls += n_k * (rel_diff * rel_diff)
        
    return q_wls
```

---

#### Practical Realities, Weighting Dynamics, and Edge Cases

1. **Why Cressie WLS Prioritizes Near-Origin Continuity:**
   The weight assigned to bin $k$ is $w_k = \frac{|N(\mathbf{h}_k)|}{\gamma(\mathbf{h}_k)^2}$.
   At small separation distances $h$, $\gamma(h)$ is small, making $w_k$ large and forcing the fitted curve to match the empirical variogram closely near $h = 0$.
   
   When sampling is dense relative to the spatial range, spatial interpolation weights in ordinary kriging are mostly governed by the initial slope of the variogram near the origin. Whether the variogram fits perfectly at distant lags near the sill has minimal impact on local predictions. However, when sample spacing is wide or comparable to the range $\phi$, the sill and range also exert substantial influence on kriging weights, and kriging prediction variances depend on the entire curve.
2. **Pair Count Distribution and Small-Lag Reality:**
   In two dimensions, the shortest lag bins contain the *fewest* point pairs (pair counts scale roughly with annulus area $\propto h \Delta h$). It is the small theoretical denominator $\gamma(h)^2$, rather than pair abundance, that creates the large WLS weight at short lags.
3. **Zero Nugget Instability:**
   If $\tau^2 \approx 0$ and the first lag $h_1 \approx 0$, $\gamma(h_1) \to 0$, causing the denominator in $Q_{\text{WLS}}$ to vanish. Numerical solvers must enforce a small lower bound on the theoretical variogram (e.g. $\max(g_{\text{theo}}, 10^{-10})$) or enforce a positive nugget bound during optimization.

---

### 4.5 Multivariate Admissibility and Cramér's Theorem

#### From Classical Correlation Bounds to Matrix-Valued Spatial Positive Definiteness

In classical multivariate statistics, an $m \times m$ covariance matrix $\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{Y})$ for a random vector $\mathbf{Y} \in \mathbb{R}^m$ must be positive semi-definite: for any vector $\mathbf{a} \in \mathbb{R}^m$, the variance of the linear combination $\operatorname{Var}(\mathbf{a}'\mathbf{Y}) = \mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} \ge 0$. For a bivariate vector $(Y_1, Y_2)'$, positive semi-definiteness requires $\sigma_1^2 \ge 0$, $\sigma_2^2 \ge 0$, and $\det(\boldsymbol{\Sigma}) = \sigma_1^2 \sigma_2^2 - \sigma_{12}^2 \ge 0$, which is equivalent to the classical Cauchy-Schwarz correlation bound:
$$
\rho_{12}^2 = \frac{\sigma_{12}^2}{\sigma_1^2 \sigma_2^2} \le 1
$$
In spatial statistics, when modeling two or more correlated fields $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_m(\mathbf{s}))'$, each individual cross-covariance function $C_{ij}(\mathbf{h})$ cannot be chosen independently. The matrix-valued function $\mathbf{C}(\mathbf{h}) = [C_{ij}(\mathbf{h})]_{i,j=1}^m$ must be positive semi-definite across all spatial point configurations. Just as scalar spatial validity requires non-negative spectral densities $f(\boldsymbol{\omega}) \ge 0$ (Bochner's Theorem), multivariate validity requires that the spatial spectral density matrix $\mathbf{f}(\boldsymbol{\omega})$ be Hermitian positive semi-definite at every spatial frequency $\boldsymbol{\omega}$, imposing the cross-spectral coherence bound $|f_{ij}(\boldsymbol{\omega})|^2 \le f_{ii}(\boldsymbol{\omega})f_{jj}(\boldsymbol{\omega})$.

When multiple spatial processes $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_m(\mathbf{s}))'$ are modeled concurrently, an algebraic dilemma arises: fitting valid univariate covariance functions to each cross-covariance $C_{ij}(\mathbf{h})$ independently does **not** ensure that the resulting joint spatial covariance matrix is positive semi-definite.

##### Definition 4.5.1 (Matrix-Valued Positive Semi-Definiteness)
A matrix-valued covariance function $\mathbf{C}: \mathbb{R}^d \to \mathbb{R}^{m \times m}$ is *positive semi-definite* if for any finite integer $n \in \mathbb{N}$, locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$, and vectors $\mathbf{a}_1, \dots, \mathbf{a}_n \in \mathbb{R}^m$:
$$
\sum_{r=1}^n \sum_{l=1}^n \mathbf{a}_r' \mathbf{C}(\mathbf{s}_r - \mathbf{s}_l) \mathbf{a}_l \ge 0
$$

##### Theorem 4.5.1 (Cramér's Theorem for Matrix-Valued Covariance Functions)
A continuous matrix-valued function $\mathbf{C}: \mathbb{R}^d \to \mathbb{R}^{m \times m}$ is positive semi-definite if and only if it admits the spatial spectral density representation:
$$
\mathbf{C}(\mathbf{h}) = \int_{\mathbb{R}^d} \exp\left(i \boldsymbol{\omega}'\mathbf{h}\right) \mathbf{f}(\boldsymbol{\omega}) d\boldsymbol{\omega}
$$
where $\mathbf{f}(\boldsymbol{\omega}) \in \mathbb{C}^{m \times m}$ is a Hermitian positive semi-definite matrix at every spatial frequency $\boldsymbol{\omega} \in \mathbb{R}^d$:
$$
\mathbf{f}(\boldsymbol{\omega}) = \mathbf{f}(\boldsymbol{\omega})^*, \quad \mathbf{v}^* \mathbf{f}(\boldsymbol{\omega}) \mathbf{v} \ge 0, \quad \forall \mathbf{v} \in \mathbb{C}^m
$$
with finite total trace integral $\int_{\mathbb{R}^d} \operatorname{Tr}(\mathbf{f}(\boldsymbol{\omega})) d\boldsymbol{\omega} < \infty$.

*Proof.*
*Sufficiency ($\impliedby$):*
Assume $\mathbf{f}(\boldsymbol{\omega})$ is Hermitian positive semi-definite for all $\boldsymbol{\omega} \in \mathbb{R}^d$. For any $n$ sampling locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in \mathbb{R}^d$ and arbitrary coefficient vectors $\mathbf{a}_1, \dots, \mathbf{a}_n \in \mathbb{R}^m$, evaluate the joint quadratic form:
$$
\sum_{r=1}^n \sum_{l=1}^n \mathbf{a}_r' \mathbf{C}(\mathbf{s}_r - \mathbf{s}_l) \mathbf{a}_l = \sum_{r=1}^n \sum_{l=1}^n \mathbf{a}_r' \left( \int_{\mathbb{R}^d} e^{i \boldsymbol{\omega}'(\mathbf{s}_r - \mathbf{s}_l)} \mathbf{f}(\boldsymbol{\omega}) d\boldsymbol{\omega} \right) \mathbf{a}_l
$$
Interchanging the finite summation and the Riemann integral, and using the identity $e^{i\boldsymbol{\omega}'(\mathbf{s}_r - \mathbf{s}_l)} = e^{i\boldsymbol{\omega}'\mathbf{s}_r} e^{-i\boldsymbol{\omega}'\mathbf{s}_l} = (e^{-i\boldsymbol{\omega}'\mathbf{s}_r})^* e^{-i\boldsymbol{\omega}'\mathbf{s}_l}$:
$$
= \int_{\mathbb{R}^d} \sum_{r=1}^n \sum_{l=1}^n \left( e^{-i \boldsymbol{\omega}'\mathbf{s}_r} \mathbf{a}_r \right)^* \mathbf{f}(\boldsymbol{\omega}) \left( e^{-i \boldsymbol{\omega}'\mathbf{s}_l} \mathbf{a}_l \right) d\boldsymbol{\omega}
$$
Define the complex vector function $\mathbf{u}(\boldsymbol{\omega}) = \sum_{l=1}^n e^{-i \boldsymbol{\omega}'\mathbf{s}_l} \mathbf{a}_l \in \mathbb{C}^m$. The integrand simplifies directly to a Hermitian quadratic form:
$$
\sum_{r=1}^n \sum_{l=1}^n \mathbf{a}_r' \mathbf{C}(\mathbf{s}_r - \mathbf{s}_l) \mathbf{a}_l = \int_{\mathbb{R}^d} \mathbf{u}(\boldsymbol{\omega})^* \mathbf{f}(\boldsymbol{\omega}) \mathbf{u}(\boldsymbol{\omega}) d\boldsymbol{\omega}
$$
Because $\mathbf{f}(\boldsymbol{\omega}) \succeq 0$ by assumption, the scalar quadratic form $\mathbf{u}(\boldsymbol{\omega})^* \mathbf{f}(\boldsymbol{\omega}) \mathbf{u}(\boldsymbol{\omega}) \ge 0$ for every frequency $\boldsymbol{\omega} \in \mathbb{R}^d$. Integrating a non-negative function produces a non-negative value:
$$
\int_{\mathbb{R}^d} \mathbf{u}(\boldsymbol{\omega})^* \mathbf{f}(\boldsymbol{\omega}) \mathbf{u}(\boldsymbol{\omega}) d\boldsymbol{\omega} \ge 0
$$
which establishes that $\mathbf{C}(\mathbf{h})$ is positive semi-definite.

*Necessity ($\implies$):*
The necessity direction follows from multivariate harmonic analysis (Cramér, 1940; Gikhman & Skorokhod, 1969) via matrix Fourier inversion and polarization identities. In the general setting, $\mathbf{f}(\boldsymbol{\omega})d\boldsymbol{\omega}$ represents a positive semi-definite matrix spectral measure $d\mathbf{F}(\boldsymbol{\omega})$, which admits a spectral density matrix $\mathbf{f}(\boldsymbol{\omega})$ whenever cross-covariances are absolutely integrable. $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: Real Symmetric Cross-Covariances**  
> While Cramér's theorem is stated generally for complex Hermitian matrices ($\mathbf{f}(\boldsymbol{\omega}) = \mathbf{f}(\boldsymbol{\omega})^*$), in spatial statistics cross-covariances are almost universally symmetric and even: $C_{ij}(\mathbf{h}) = C_{ji}(\mathbf{h}) = C_{ij}(-\mathbf{h})$. In this symmetric setting, the cross-spectral densities $f_{ij}(\boldsymbol{\omega})$ are purely real and symmetric ($f_{ij}(\boldsymbol{\omega}) = f_{ji}(\boldsymbol{\omega}) \in \mathbb{R}$). Thus, "Hermitian positive semi-definite" simply means that $\mathbf{f}(\boldsymbol{\omega})$ is a standard **real symmetric positive semi-definite matrix** ($\mathbf{f}(\boldsymbol{\omega}) = \mathbf{f}(\boldsymbol{\omega})' \succeq 0$) at every spatial frequency $\boldsymbol{\omega}$.

##### Corollary 4.5.1 (Spectral Density Coherence Condition)
For a bivariate spatial random field ($m = 2$), the $2 \times 2$ spectral density matrix is:
$$
\mathbf{f}(\boldsymbol{\omega}) = \begin{bmatrix} f_{11}(\boldsymbol{\omega}) & f_{12}(\boldsymbol{\omega}) \\ f_{21}(\boldsymbol{\omega}) & f_{22}(\boldsymbol{\omega}) \end{bmatrix}
$$
Positive semi-definiteness $\mathbf{f}(\boldsymbol{\omega}) \succeq 0$ requires that the diagonal spectral densities be non-negative ($f_{11}(\boldsymbol{\omega}) \ge 0, f_{22}(\boldsymbol{\omega}) \ge 0$) and that the determinant be non-negative:
$$
\det(\mathbf{f}(\boldsymbol{\omega})) = f_{11}(\boldsymbol{\omega}) f_{22}(\boldsymbol{\omega}) - |f_{12}(\boldsymbol{\omega})|^2 \ge 0
$$
This yields the foundational *cross-spectral coherence condition*:
$$
|f_{12}(\boldsymbol{\omega})|^2 \le f_{11}(\boldsymbol{\omega}) f_{22}(\boldsymbol{\omega}), \quad \forall \boldsymbol{\omega} \in \mathbb{R}^d
$$

---

#### Numerical Verification of Spectral Admissibility

Checking positive semi-definiteness for candidate bivariate cross-covariances involves evaluating the spectral determinant across a frequency grid $\boldsymbol{\omega}$. Note that checking a finite grid can only **refute** admissibility (if $\det\mathbf{f}(\boldsymbol{\omega}) < 0$ at any evaluated frequency, the model is invalid); it cannot mathematically guarantee validity across all $\mathbb{R}^d$ since violations may lurk between grid points.

```python
import math

def verify_bivariate_spectral_admissibility(f11_func, f22_func, f12_func, omega_grid):
    """
    Evaluates Cramér's positive semi-definiteness condition across a frequency grid
    matching Corollary 4.5.1 in pure standard Python.
    
    Note: A grid check can refute validity if min_det < 0, but cannot definitively
    prove validity across all continuous frequencies.
    """
    admissible = True
    min_det = math.inf
    
    for omega in omega_grid:
        f11 = f11_func(omega)
        f22 = f22_func(omega)
        f12 = f12_func(omega)
        
        # Absolute square: |f12|^2 = Re(f12)^2 + Im(f12)^2
        f12_abs_sq = (f12.real ** 2 + f12.imag ** 2) if isinstance(f12, complex) else (f12 * f12)
        det_f = f11 * f22 - f12_abs_sq
        
        if det_f < min_det:
            min_det = det_f
        if det_f < -1e-12 or f11 < 0.0 or f22 < 0.0:
            admissible = False
            
    return admissible, min_det
```

---

#### Diagnostic Realities and The Independent Modeling Trap

1. **The Independent Modeling Trap:**
   Consider a bivariate exponential model on $\mathbb{R}^1$: $C_{11}(h) = e^{-|h|}$ (decay rate $1$), $C_{22}(h) = e^{-2|h|}$ (decay rate $2$), and cross-covariance $C_{12}(h) = \rho_{12} e^{-1.5|h|}$ (decay rate $1.5$). A practitioner might assume that any correlation $|\rho_{12}| < 1$ is admissible.
   
   However, as derived analytically in Exercise 4.3, Cramér's cross-spectral coherence condition $|f_{12}(\omega)|^2 \le f_{11}(\omega)f_{22}(\omega)$ is violated whenever $|\rho_{12}| > \sqrt{70}/9 \approx 0.930$, with the worst-case violation occurring at spatial frequency $\omega = \sqrt{6.5} \approx 2.55\text{ rad}$. Choosing $\rho_{12} = 0.95$ yields three individually valid univariate correlation curves, but produces negative eigenvalues in the cross-spectral matrix and causes cokriging algorithms to crash with negative prediction variances ($\sigma_{\text{CK}}^2 < 0$)!
2. **Brute-Force Finite Checking in Practice:**
   When analytical Fourier transforms are difficult to evaluate, practitioners can test candidate cross-covariance models empirically: assemble the joint $2n \times 2n$ covariance matrix $\mathbf{V}_{\text{CK}}$ across a test grid of $n$ spatial locations and evaluate its smallest eigenvalue. If $\lambda_{\min}(\mathbf{V}_{\text{CK}}) < 0$, the candidate cross-covariance is immediately refuted.

---

### 4.6 The Linear Model of Coregionalization (LMC) and Eigenvalue Projection

#### From Factor Analysis to Spatial Coregionalization

In classical multivariate statistics, factor analysis decomposes an $m$-dimensional observation vector $\mathbf{Y}$ into a linear combination of $q \le m$ latent common factors plus unique noise: $\mathbf{Y} = \boldsymbol{\mu} + \mathbf{A}\mathbf{F} + \boldsymbol{\epsilon}$, where $\operatorname{Var}(\mathbf{F}) = \mathbf{I}_q$. This structural decomposition factors the covariance matrix into $\boldsymbol{\Sigma} = \mathbf{A}\mathbf{A}' + \boldsymbol{\Psi}$, which automatically guarantees that the covariance matrix is positive semi-definite because the outer product $\mathbf{A}\mathbf{A}' \succeq 0$ by construction.

In spatial statistics, the Linear Model of Coregionalization (LMC) (Journel & Huijbregts, 1978; Matheron, 1982) extends classical factor analysis to continuous spatial random fields. The $m$-variate field $\mathbf{Z}(\mathbf{s})$ is represented as a linear combination of independent spatial factor processes $\mathbf{Y}_u(\mathbf{s})$ operating across $U$ distinct spatial ranges. This decomposes the matrix-valued spatial covariance into a sum of Kronecker-type structures $\mathbf{C}(\mathbf{h}) = \sum_{u=1}^U \mathbf{B}_u \rho_u(\mathbf{h})$, where each coregionalization matrix $\mathbf{B}_u = \mathbf{A}_u \mathbf{A}_u' \in \mathbb{R}^{m \times m}$ is an $m \times m$ positive semi-definite matrix of sill parameters. If sample fitting yields a candidate matrix with negative eigenvalues, the Frobenius eigenvalue projection operator sets negative eigenvalues to zero, guaranteeing that the joint multivariate model remains valid.

##### Definition 4.6.1 (The Linear Model of Coregionalization)
Let $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_m(\mathbf{s}))'$ be an $m$-variate second-order stationary field. Under the LMC, $\mathbf{Z}(\mathbf{s})$ is represented as a linear combination of $U$ independent spatial factor processes operating at distinct spatial scales:
$$
\mathbf{Z}(\mathbf{s}) = \boldsymbol{\mu} + \sum_{u=1}^U \mathbf{A}_u \mathbf{Y}_u(\mathbf{s})
$$
where:
- $\mathbf{Y}_u(\mathbf{s}) = (Y_{u,1}(\mathbf{s}), \dots, Y_{u,p_u}(\mathbf{s}))'$ is a vector of $p_u$ standardized, mutually uncorrelated spatial factors at scale $u$.
- Each factor has unit variance and correlation function $\rho_u(\mathbf{h})$, where $\rho_u(\cdot)$ is an admissible univariate correlation kernel.
- $\mathbf{A}_u \in \mathbb{R}^{m \times p_u}$ ($p_u \le m$) is a matrix of factor loadings.

##### Theorem 4.6.1 (Coregionalization Matrix Representation and Sufficiency)
The spatial covariance matrix function $\mathbf{C}(\mathbf{h}) \in \mathbb{R}^{m \times m}$ and semivariogram matrix function $\mathbf{\Gamma}(\mathbf{h}) \in \mathbb{R}^{m \times m}$ expand as:
$$
\mathbf{C}(\mathbf{h}) = \sum_{u=1}^U \mathbf{B}_u \rho_u(\mathbf{h}), \quad \mathbf{\Gamma}(\mathbf{h}) = \sum_{u=1}^U \mathbf{B}_u (1 - \rho_u(\mathbf{h}))
$$
where each $\mathbf{B}_u \in \mathbb{R}^{m \times m}$ is the *coregionalization matrix* at scale $u$, defined by:
$$
\mathbf{B}_u = \mathbf{A}_u \mathbf{A}_u'
$$
A sufficient condition for the model to be admissible on $\mathbb{R}^d$ is that each coregionalization matrix is symmetric positive semi-definite:
$$
\mathbf{B}_u \succeq 0, \quad \forall u \in \{1, \dots, U\}
$$

*Proof.*
Expanding the covariance using the mutual independence of factors across scales and components:
$$
\mathbf{C}(\mathbf{h}) = \operatorname{Cov}(\mathbf{Z}(\mathbf{s}), \mathbf{Z}(\mathbf{s}+\mathbf{h})) = \sum_{u=1}^U \sum_{v=1}^U \mathbf{A}_u \operatorname{Cov}(\mathbf{Y}_u(\mathbf{s}), \mathbf{Y}_v(\mathbf{s}+\mathbf{h})) \mathbf{A}_v'
$$
Since $\operatorname{Cov}(\mathbf{Y}_u(\mathbf{s}), \mathbf{Y}_v(\mathbf{s}+\mathbf{h})) = \mathbf{0}$ for $u \ne v$, and $\operatorname{Cov}(\mathbf{Y}_u(\mathbf{s}), \mathbf{Y}_u(\mathbf{s}+\mathbf{h})) = \rho_u(\mathbf{h})\mathbf{I}_{p_u}$:
$$
\mathbf{C}(\mathbf{h}) = \sum_{u=1}^U \mathbf{A}_u (\rho_u(\mathbf{h})\mathbf{I}_{p_u}) \mathbf{A}_u' = \sum_{u=1}^U (\mathbf{A}_u \mathbf{A}_u') \rho_u(\mathbf{h}) = \sum_{u=1}^U \mathbf{B}_u \rho_u(\mathbf{h})
$$
Since $\mathbf{B}_u = \mathbf{A}_u \mathbf{A}_u'$, for any vector $\mathbf{v} \in \mathbb{R}^m$, $\mathbf{v}'\mathbf{B}_u\mathbf{v} = (\mathbf{A}_u'\mathbf{v})'(\mathbf{A}_u'\mathbf{v}) = \|\mathbf{A}_u'\mathbf{v}\|^2 \ge 0$. Thus $\mathbf{B}_u \succeq 0$. $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: Sufficiency versus Necessity in the LMC**  
> While $\mathbf{B}_u \succeq 0$ for all $u$ is universally enforced in software, it is strictly **sufficient**, not mathematically necessary. By Cramér's theorem, joint admissibility requires only that the total spectral density matrix $\mathbf{f}(\boldsymbol{\omega}) = \sum_{u=1}^U \mathbf{B}_u f_u(\boldsymbol{\omega}) \succeq 0$ at every frequency $\boldsymbol{\omega}$.  
> For example, if two exponential structures have decay rates $2$ and $1$, setting $\mathbf{B}_1 = \operatorname{diag}(1, -0.1)$ and $\mathbf{B}_2 = \operatorname{diag}(0.5, 1)$ yields an individual matrix $\mathbf{B}_1$ with a negative eigenvalue. Yet the total spectral density remains strictly positive at all frequencies because the positive mass of $\mathbf{B}_2 f_2(\omega)$ outweighs the negative component of $\mathbf{B}_1 f_1(\omega)$! Nonetheless, enforcing $\mathbf{B}_u \succeq 0$ scale-by-scale guarantees validity unconditionally across all spatial configurations and correlation choices.

##### Theorem 4.6.2 (Optimal Positive Semi-Definite Projection in Frobenius Norm)
Let $\widetilde{\mathbf{B}} \in \mathbb{R}^{m \times m}$ be an unconstrained symmetric candidate matrix obtained from least-squares fitting of empirical cross-variograms. Let its spectral decomposition be:
$$
\widetilde{\mathbf{B}} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}' = \sum_{j=1}^m \lambda_j \mathbf{q}_j \mathbf{q}_j'
$$
where $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_m]$ is orthogonal and $\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_m)$.
The optimal positive semi-definite matrix $\mathbf{B}^* \succeq 0$ that minimizes the Frobenius distance $\|\widetilde{\mathbf{B}} - \mathbf{B}\|_F^2 = \operatorname{Tr}((\widetilde{\mathbf{B}} - \mathbf{B})'(\widetilde{\mathbf{B}} - \mathbf{B}))$ is obtained by eigenvalue clipping:
$$
\mathbf{B}^* = \mathbf{Q} \mathbf{\Lambda}^+ \mathbf{Q}' = \sum_{j=1}^m \max(0, \lambda_j) \mathbf{q}_j \mathbf{q}_j'
$$

*Proof.*
For any symmetric $\mathbf{B}$, write $\mathbf{B} = \mathbf{Q} \mathbf{M} \mathbf{Q}'$ where $\mathbf{M} = \mathbf{Q}'\mathbf{B}\mathbf{Q}$. By unitarily invariant Frobenius norm properties:
$$
\|\widetilde{\mathbf{B}} - \mathbf{B}\|_F^2 = \|\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}' - \mathbf{Q}\mathbf{M}\mathbf{Q}'\|_F^2 = \|\mathbf{\Lambda} - \mathbf{M}\|_F^2 = \sum_{j=1}^m (\lambda_j - M_{jj})^2 + \sum_{i \ne j} M_{ij}^2
$$
For any positive semi-definite matrix $\mathbf{M} \succeq 0$, all diagonal entries are non-negative ($M_{jj} \ge 0$) and all off-diagonal squares are non-negative ($M_{ij}^2 \ge 0$). Thus:
$$
\|\mathbf{\Lambda} - \mathbf{M}\|_F^2 \ge \sum_{j=1}^m (\lambda_j - M_{jj})^2 \ge \sum_{j=1}^m (\lambda_j - \max(0, \lambda_j))^2
$$
Equality holds if and only if $M_{ij} = 0$ for all $i \ne j$ and $M_{jj} = \max(0, \lambda_j)$. Thus $\mathbf{M}^* = \mathbf{\Lambda}^+$, which gives $\mathbf{B}^* = \mathbf{Q}\mathbf{\Lambda}^+\mathbf{Q}'$. $\blacksquare$

> [!TIP]
> **Projection Repairs a Candidate, Not the Variogram Fit**  
> Theorem 4.6.2 finds the closest valid matrix to an unconstrained estimate $\widetilde{\mathbf{B}}$ in Frobenius matrix distance. It does not directly minimize the weighted least-squares criterion on empirical variograms. The projected matrix $\mathbf{B}^*$ is guaranteed to be positive semi-definite, but it may fit the empirical semivariances slightly worse than a matrix estimated via joint constrained optimization (e.g. Goulard & Voltz, 1992).

---

#### Implementing Positive Semi-Definite Matrix Projection

```python
import math

def project_lmc_positive_semidefinite(candidate_B_matrices, min_eigenvalue=1e-8):
    """
    Projects candidate LMC coregionalization matrices onto the cone of 
    positive semi-definite matrices matching Theorem 4.6.2 (modified with a
    small positive floor for numerical safety) using cyclic Jacobi rotations.
    
    Parameters:
        candidate_B_matrices: list of symmetric matrices B_u in R^(m x m)
        min_eigenvalue: float, floor for eigenvalues (Tikhonov regularizer)
        
    Returns:
        projected_B: list of positive semi-definite matrices B_u*
    """
    def jacobi_eigh(A, tol=1e-12, max_iter=100):
        m = len(A)
        V = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
        D = [[A[i][j] for j in range(m)] for i in range(m)]
        
        for _ in range(max_iter):
            max_val = 0.0
            p, q = 0, 1
            for i in range(m):
                for j in range(i + 1, m):
                    if abs(D[i][j]) > max_val:
                        max_val = abs(D[i][j])
                        p, q = i, j
            if max_val < tol:
                break
                
            diff = D[q][q] - D[p][p]
            if abs(D[p][q]) < 1e-15:
                t = 0.0
            else:
                phi = diff / (2.0 * D[p][q])
                t = math.copysign(1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0)), phi)
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            
            D_pp = D[p][p]
            D_qq = D[q][q]
            D_pq = D[p][q]
            D[p][p] = c * c * D_pp - 2.0 * s * c * D_pq + s * s * D_qq
            D[q][q] = s * s * D_pp + 2.0 * s * c * D_pq + c * c * D_qq
            D[p][q] = 0.0
            D[q][p] = 0.0
            
            for k in range(m):
                if k != p and k != q:
                    d_kp = D[k][p]
                    d_kq = D[k][q]
                    D[k][p] = c * d_kp - s * d_kq
                    D[p][k] = D[k][p]
                    D[k][q] = s * d_kp + c * d_kq
                    D[q][k] = D[k][q]
                    
            for k in range(m):
                v_kp = V[k][p]
                v_kq = V[k][q]
                V[k][p] = c * v_kp - s * v_kq
                V[k][q] = s * v_kp + c * v_kq
                
        eigenvalues = [D[i][i] for i in range(m)]
        return eigenvalues, V

    projected_B = []
    for B in candidate_B_matrices:
        m = len(B)
        B_sym = [[0.5 * (B[i][j] + B[j][i]) for j in range(m)] for i in range(m)]
        eigs, Q = jacobi_eigh(B_sym)
        eigs_clipped = [max(min_eigenvalue, val) for val in eigs]
        
        B_proj = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                B_proj[i][j] = sum(eigs_clipped[k] * Q[i][k] * Q[j][k] for k in range(m))
                
        B_proj = [[0.5 * (B_proj[i][j] + B_proj[j][i]) for j in range(m)] for i in range(m)]
        projected_B.append(B_proj)
        
    return projected_B
```

---

#### Geometric Interpretation, Nugget Invertibility, and Practical Realities

```
Projection onto the Positive Semi-Definite Cone (Eigenvalue Space):
       ^ λ_2
       |
       |      Valid LMC Region S_+^2
       |      (Non-negative Quadrant: λ_1 >= 0, λ_2 >= 0)
       |
───────+───────────────*──────────────> λ_1
       |               | Projected Solution B* = (λ_1, 0)
       |               |
       |               * Unconstrained Candidate B_tilde = (λ_1, λ_2)
       |                 (has negative eigenvalue λ_2 < 0)
```

1. **Geometric Cone Projection:**
   The set of symmetric positive semi-definite matrices $\mathcal{S}_+^m$ forms a convex cone in $\mathbb{R}^{m(m+1)/2}$. In eigenvalue space $(\lambda_1, \lambda_2)$, this cone corresponds to the non-negative quadrant. When unconstrained regression estimates an invalid matrix with $\lambda_2 < 0$, spectral clipping drops the point orthogonally onto the $\lambda_1$ boundary axis.
2. **Worked Hand Example ($2 \times 2$ Candidate Repair):**
   Suppose unconstrained fitting yields the candidate sill matrix:
   $$
   \widetilde{\mathbf{B}} = \begin{bmatrix} 1.0 & 1.2 \\ 1.2 & 1.0 \end{bmatrix}
   $$
   The eigenvalues are $\lambda_1 = 2.2$ (eigenvector $\mathbf{q}_1 = \frac{1}{\sqrt{2}}(1, 1)'$) and $\lambda_2 = -0.2$ (eigenvector $\mathbf{q}_2 = \frac{1}{\sqrt{2}}(1, -1)'$).
   Because $\lambda_2 = -0.2 < 0$, $\widetilde{\mathbf{B}}$ is not positive semi-definite. Clipping $\lambda_2$ to $0$ gives:
   $$
   \mathbf{B}^* = 2.2 \mathbf{q}_1 \mathbf{q}_1' + 0 \mathbf{q}_2 \mathbf{q}_2' = 2.2 \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 1.1 & 1.1 \\ 1.1 & 1.1 \end{bmatrix}
   $$
   $\mathbf{B}^*$ is positive semi-definite (rank 1), admissible, and the closest valid matrix to $\widetilde{\mathbf{B}}$ in Frobenius norm.
3. **The Nugget Coregionalization Matrix ($\mathbf{B}_0$):**
   When an LMC includes a nugget effect at scale $u = 0$ with $\rho_0(\mathbf{h}) = \mathbb{I}(\mathbf{h} = \mathbf{0})$, we write:
   $$
   \mathbf{C}(\mathbf{h}) = \mathbf{B}_0 \mathbb{I}(\mathbf{h} = \mathbf{0}) + \sum_{u=1}^U \mathbf{B}_u \rho_u(\mathbf{h})
   $$
   While mathematical validity requires only $\mathbf{B}_0 \succeq 0$, geostatistical practice strongly recommends enforcing **strict positive definiteness** ($\mathbf{B}_0 \succ 0$). A strictly positive definite nugget matrix ensures that the total cokriging covariance matrix $\mathbf{V}_{\text{CK}}$ remains strictly invertible and well-conditioned even when multiple variables are measured at co-located or tightly clustered sites.

---

### Key Ideas and Takeaways: Covariance Parameter Estimation

| Estimation Criterion | Objective Function | Key Advantages | Computational Cost & Limits |
|---|---|---|---|
| **Ordinary Least Squares (OLS Variogram)** | $Q_{\text{OLS}}(\boldsymbol{\theta}) = \sum (\widehat{\gamma}_k - \gamma_k)^2$ | Simple, no matrix inversions | Severely distorted by noisy, high-variance large-lag bins |
| **Cressie's Weighted Least Squares (WLS)** | $Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum \frac{\|N(h_k)\|}{\gamma_k(\boldsymbol{\theta})^2} (\widehat{\gamma}_k - \gamma_k)^2$ | Downweights large lags; prioritizes near-origin slope; fast ($\text{ms}$) | Neglects inter-lag covariances; provides excellent starting values for REML |
| **Maximum Likelihood (ML)** | $\ell_{\text{ML}}(\boldsymbol{\beta}, \boldsymbol{\theta}) = -\frac{1}{2}\ln\|\mathbf{V}\| - \frac{1}{2}\mathbf{e}'\mathbf{V}^{-1}\mathbf{e}$ | Valid for comparing different trend matrices $\mathbf{X}$; asymptotically efficient | Underestimates covariance scale $\sigma^2$ by factor $(n-p)/n$; $\mathcal{O}(n^3)$ Cholesky cost |
| **Restricted Maximum Likelihood (REML)** | $\ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{1}{2}\ln\|\mathbf{V}\| - \frac{1}{2}\ln\|\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\| - \frac{1}{2}\mathbf{Z}'\mathbf{P}\mathbf{Z}$ | Unbiased scale estimator; eliminates trend parameter interference; standard in spatial modeling | Cannot compare models with different trend covariates $\mathbf{X}$; $\mathcal{O}(n^3)$ Cholesky cost |
| **LMC Coregionalization Projection** | $\mathbf{B}^* = \mathbf{Q} \mathbf{\Lambda}^+ \mathbf{Q}'$ (Theorem 4.6.2) | Guarantees multivariate positive semi-definiteness via eigenvalue clipping | Repairs candidate sills; does not directly solve variogram WLS criterion |

---

### 4.7 Analytical Exercises and Step-by-Step Solutions

#### Exercise 4.1: Direct Hessian and Spatial Fisher Information
Let $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{0}, \mathbf{V}(\theta))$ where $\theta$ is a single scalar parameter.
- **(a)** Derive the exact Hessian $\frac{\partial^2 \ell}{\partial \theta^2}$ of the Gaussian log-likelihood.
- **(b)** Take mathematical expectations to prove directly that:
  $$
  \operatorname{E}\left[-\frac{\partial^2 \ell}{\partial \theta^2}\right] = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta} \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta}\right)
  $$
  without using the score covariance identity.

##### Solution:
- **(a)** From Theorem 4.1.1, the first derivative is:
  $$
  \frac{\partial \ell}{\partial \theta} = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right) - \frac{1}{2}\mathbf{Z}'\left(\frac{\partial \mathbf{V}^{-1}}{\partial \theta}\right)\mathbf{Z}
  $$
  Differentiating with respect to $\theta$:
  $$
  \frac{\partial^2 \ell}{\partial \theta^2} = -\frac{1}{2}\frac{\partial}{\partial \theta}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right) - \frac{1}{2}\mathbf{Z}'\left(\frac{\partial^2 \mathbf{V}^{-1}}{\partial \theta^2}\right)\mathbf{Z}
  $$
  Differentiating the trace term using $\frac{\partial \mathbf{V}^{-1}}{\partial \theta} = -\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}$:
  $$
  \frac{\partial}{\partial \theta}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right) = -\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta} + \mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}
  $$
  Next differentiate $\frac{\partial \mathbf{V}^{-1}}{\partial \theta}$:
  $$
  \frac{\partial^2 \mathbf{V}^{-1}}{\partial \theta^2} = 2\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1} - \mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\mathbf{V}^{-1}
  $$
  Substituting back:
  $$
  \frac{\partial^2 \ell}{\partial \theta^2} = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right) - \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\right)
  $$
  $$
  - \mathbf{Z}'\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\right)\mathbf{Z} + \frac{1}{2}\mathbf{Z}'\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\mathbf{V}^{-1}\right)\mathbf{Z}
  $$
- **(b)** Taking expectations of the quadratic forms:
  For any symmetric matrix $\mathbf{M}$, $\operatorname{E}[\mathbf{Z}'\mathbf{M}\mathbf{Z}] = \operatorname{Tr}(\mathbf{M}\mathbf{V})$.
  For the fourth term:
  $$
  \operatorname{E}\left[\frac{1}{2}\mathbf{Z}'\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\mathbf{V}^{-1}\right)\mathbf{Z}\right] = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\mathbf{V}^{-1}\mathbf{V}\right) = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\right)
  $$
  This cancels the second term identically:
  $$
  -\frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\right) + \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial^2 \mathbf{V}}{\partial \theta^2}\right) = 0
  $$
  For the third term:
  $$
  \operatorname{E}\left[\mathbf{Z}'\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\right)\mathbf{Z}\right] = \operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right)
  $$
  Combining remaining terms:
  $$
  \operatorname{E}\left[\frac{\partial^2 \ell}{\partial \theta^2}\right] = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right) - \operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right) = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right)
  $$
  Negating yields the Fisher Information: $\mathcal{I}(\theta) = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta}\right)$. $\blacksquare$

---

#### Exercise 4.2: Exact Analytical REML Derivation and MSE Comparison
Consider $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \sigma^2 \mathbf{R})$ where $\mathbf{R} \succ 0$ is known and $\sigma^2 > 0$ is unknown.
- **(a)** Derive the explicit analytical maximizer $\widehat{\sigma}^2_{\text{REML}}$.
- **(b)** Prove exact unbiasedness: $\operatorname{E}[\widehat{\sigma}^2_{\text{REML}}] = \sigma^2$.
- **(c)** Derive the exact Mean Squared Errors ($\mathrm{MSE}$) of $\widehat{\sigma}^2_{\text{REML}}$ and $\widehat{\sigma}^2_{\text{ML}}$. Determine under what condition REML has a smaller MSE than ML, and interpret the result.

##### Solution:
- **(a)** Using $\mathbf{V} = \sigma^2 \mathbf{R}$, the Harville log-likelihood from Theorem 4.3.3 is:
  $$
  \ell_{\text{REML}}(\sigma^2) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\sigma^2 \mathbf{R}) - \frac{1}{2}\ln\det\left(\mathbf{X}'(\sigma^2 \mathbf{R})^{-1}\mathbf{X}\right) - \frac{1}{2\sigma^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}
  $$
  Note that $\ln\det(\sigma^2 \mathbf{R}) = n\ln(\sigma^2) + \ln\det(\mathbf{R})$ and $\ln\det(\mathbf{X}'(\sigma^2 \mathbf{R})^{-1}\mathbf{X}) = -p\ln(\sigma^2) + \ln\det(\mathbf{X}'\mathbf{R}^{-1}\mathbf{X})$.
  Summing these determinant terms:
  $$
  n\ln(\sigma^2) - p\ln(\sigma^2) = (n - p)\ln(\sigma^2)
  $$
  The REML log-likelihood simplifies to:
  $$
  \ell_{\text{REML}}(\sigma^2) = -\frac{n-p}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} + \text{constant}
  $$
  Differentiating with respect to $\sigma^2$ and setting to zero:
  $$
  \frac{\partial \ell_{\text{REML}}}{\partial \sigma^2} = -\frac{n-p}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} = 0 \implies \widehat{\sigma}^2_{\text{REML}} = \frac{1}{n - p}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}
  $$
- **(b)** By Cholesky whitening, let $\mathbf{R} = \mathbf{L}\mathbf{L}'$ and $\mathbf{w} = \mathbf{L}^{-1}\mathbf{Z} \sim \mathcal{N}_n(\mathbf{L}^{-1}\mathbf{X}\boldsymbol{\beta}, \sigma^2 \mathbf{I}_n)$.
  Then $\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} = \mathbf{w}'(\mathbf{I} - \mathbf{H}_{\tilde{\mathbf{X}}})\mathbf{w}$, where $\mathbf{I} - \mathbf{H}_{\tilde{\mathbf{X}}}$ is a rank-$(n-p)$ symmetric idempotent projection matrix.
  By Cochran's theorem, $\frac{1}{\sigma^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} \sim \chi_{n-p}^2$.
  Therefore, $\operatorname{E}[\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}] = (n-p)\sigma^2$, which immediately yields:
  $$
  \operatorname{E}[\widehat{\sigma}^2_{\text{REML}}] = \frac{1}{n - p}(n - p)\sigma^2 = \sigma^2
  $$
  confirming that REML is strictly unbiased.
- **(c)** Since $\frac{1}{\sigma^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} \sim \chi_{n-p}^2$ and $\operatorname{Var}(\chi_{n-p}^2) = 2(n-p)$, we have $\operatorname{Var}(\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}) = 2(n-p)\sigma^4$.
  
  For REML (unbiased, so $\mathrm{MSE} = \operatorname{Var}$):
  $$
  \mathrm{MSE}(\widehat{\sigma}^2_{\text{REML}}) = \operatorname{Var}(\widehat{\sigma}^2_{\text{REML}}) = \frac{1}{(n-p)^2}\left(2(n-p)\sigma^4\right) = \frac{2\sigma^4}{n - p}
  $$
  For ML ($\widehat{\sigma}^2_{\text{ML}} = \frac{1}{n}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}$):
  $$
  \operatorname{Var}(\widehat{\sigma}^2_{\text{ML}}) = \frac{1}{n^2}\left(2(n-p)\sigma^4\right) = \frac{2(n-p)\sigma^4}{n^2}
  $$
  $$
  \operatorname{Bias}(\widehat{\sigma}^2_{\text{ML}}) = \operatorname{E}[\widehat{\sigma}^2_{\text{ML}}] - \sigma^2 = \frac{n-p}{n}\sigma^2 - \sigma^2 = -\frac{p}{n}\sigma^2
  $$
  Adding variance and squared bias yields the exact ML MSE:
  $$
  \mathrm{MSE}(\widehat{\sigma}^2_{\text{ML}}) = \operatorname{Var}(\widehat{\sigma}^2_{\text{ML}}) + \operatorname{Bias}(\widehat{\sigma}^2_{\text{ML}})^2 = \frac{2(n-p)\sigma^4}{n^2} + \frac{p^2\sigma^4}{n^2} = \frac{2n - 2p + p^2}{n^2}\sigma^4
  $$
  
  To compare the two MSEs:
  $$
  \mathrm{MSE}(\widehat{\sigma}^2_{\text{REML}}) < \mathrm{MSE}(\widehat{\sigma}^2_{\text{ML}}) \iff \frac{2}{n - p} < \frac{2n - 2p + p^2}{n^2}
  $$
  Cross-multiplying by $n^2(n-p) > 0$:
  $$
  2n^2 < (n - p)(2n - 2p + p^2) = 2n^2 - 4np + n p^2 + 2p^2 - p^3
  $$
  Subtracting $2n^2$ and factoring out $p > 0$:
  $$
  0 < p \left[ n(p - 4) - p(p - 2) \right] \iff n(p - 4) > p(p - 2)
  $$
  
  **Conclusion and Numerical Comparison:**
  - For $p \le 4$, $n(p-4) \le 0 < p(p-2)$, so ML has a slightly smaller MSE because variance reduction outweighs bias squared.
  - However, for $p \ge 5$ with moderate or large sample sizes $n$, **REML achieves a strictly smaller Mean Squared Error than ML!**
  
  | Sample Size $n$ | Trend Predictors $p$ | $\mathrm{MSE}(\widehat{\sigma}^2_{\text{ML}}) / \sigma^4$ | $\mathrm{MSE}(\widehat{\sigma}^2_{\text{REML}}) / \sigma^4$ | Winner |
  |---|---|---|---|---|
  | $n = 50$ | $p = 3$ | $0.0412$ | $0.0426$ | ML (slightly lower) |
  | $n = 50$ | $p = 6$ | $0.0496$ | **$0.0455$** | **REML (lower MSE)** |
  | $n = 50$ | $p = 10$ | $0.0720$ | **$0.0500$** | **REML (30% lower MSE)** |
  | $n = 100$ | $p = 10$ | $0.0280$ | **$0.0222$** | **REML (21% lower MSE)** |

  *(Remark on Optimal Divisor:)* If the sole goal were minimizing MSE among estimators of the form $c^{-1}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}$, elementary calculus shows the optimal divisor is $c^* = n - p + 2$. In geostatistics, however, unbiasedness ($c = n - p$) is universally favored because underestimating $\sigma^2$ degrades kriging prediction interval coverage. $\blacksquare$

---

#### Exercise 4.3: Cramér Admissibility for Bivariate Exponential Models
Let $(Z_1(s), Z_2(s))'$ be a bivariate stationary Gaussian process on $\mathbb{R}^1$ with candidate cross-covariance:
$$
C_{11}(h) = e^{-|h|}, \quad C_{22}(h) = e^{-2|h|}, \quad C_{12}(h) = \rho_{12} e^{-1.5|h|}
$$
Apply Cramér's Theorem to find the exact range of $\rho_{12} \in (-1, 1)$ for which this model is admissible.

##### Solution:
On $\mathbb{R}^1$, the Fourier transform of $e^{-\alpha |h|}$ is $f(\omega) = \frac{1}{\pi}\frac{\alpha}{\alpha^2 + \omega^2}$.
Thus:
$$
f_{11}(\omega) = \frac{1}{\pi}\frac{1}{1 + \omega^2}, \quad f_{22}(\omega) = \frac{1}{\pi}\frac{2}{4 + \omega^2}, \quad f_{12}(\omega) = \rho_{12}\frac{1}{\pi}\frac{1.5}{2.25 + \omega^2}
$$
By Corollary 4.5.1, positive semi-definiteness holds if and only if for all $\omega \in \mathbb{R}$:
$$
|f_{12}(\omega)|^2 \le f_{11}(\omega) f_{22}(\omega) \iff \rho_{12}^2 \le \frac{2(2.25 + \omega^2)^2}{(1.5)^2 (1 + \omega^2)(4 + \omega^2)} = \frac{8(2.25 + \omega^2)^2}{9(1 + \omega^2)(4 + \omega^2)}
$$
Let $u = \omega^2 \ge 0$. Define $g(u) = \frac{8(2.25 + u)^2}{9(u^2 + 5u + 4)}$.
To find the infimum over $u \ge 0$, examine $\phi(u) = \frac{(u + 2.25)^2}{u^2 + 5u + 4}$.
Differentiating $\ln\phi(u)$:
$$
\frac{d}{du}\ln\phi(u) = \frac{2}{u + 2.25} - \frac{2u + 5}{u^2 + 5u + 4} = \frac{0.5u - 3.25}{(u + 2.25)(u^2 + 5u + 4)}
$$
Setting to zero yields $u^* = \frac{3.25}{0.5} = 6.5$.
Evaluating $\phi(6.5)$:
$$
\phi(6.5) = \frac{(6.5 + 2.25)^2}{(6.5 + 1)(6.5 + 4)} = \frac{(8.75)^2}{7.5 \times 10.5} = \frac{76.5625}{78.75} = \frac{35}{36}
$$
Thus, the minimum of $g(u)$ is:
$$
\inf_{\omega \in \mathbb{R}} g(\omega) = g(6.5) = \frac{8}{9} \times \frac{35}{36} = \frac{70}{81} \approx 0.864198
$$
Cramér's condition requires $\rho_{12}^2 \le \frac{70}{81}$, which implies:
$$
|\rho_{12}| \le \frac{\sqrt{70}}{9} \approx 0.92967
$$
Values of $|\rho_{12}| \in (0.93, 1.0)$ produce negative eigenvalues in the cross-spectral density matrix near $\omega = \sqrt{6.5} \approx 2.55\text{ rad}$, proving the strict necessity of Cramér's theorem. $\blacksquare$

---

#### Exercise 4.4: Factor Loading, Cholesky Representation, and Candidate Matrix Repair
- **(a)** Apply Sylvester's criterion to verify that the coregionalization matrix $\mathbf{B}$ is positive definite:
  $$
  \mathbf{B} = \begin{bmatrix} 4 & 2 & -1 \\ 2 & 3 & 1 \\ -1 & 1 & 2 \end{bmatrix}
  $$
- **(b)** Calculate the Cholesky factor loading matrix $\mathbf{A} \in \mathbb{R}^{3 \times 3}$ such that $\mathbf{B} = \mathbf{A}\mathbf{A}'$.
- **(c)** Suppose unconstrained fitting of an empirical cross-variogram yields the invalid $2 \times 2$ candidate matrix:
  $$
  \widetilde{\mathbf{B}} = \begin{bmatrix} 1.0 & 1.2 \\ 1.2 & 1.0 \end{bmatrix}
  $$
  Show that $\widetilde{\mathbf{B}}$ has a negative eigenvalue, and project it onto the cone of positive semi-definite matrices $\mathcal{S}_+^2$ in Frobenius norm using Theorem 4.6.2.

##### Solution:
- **(a)** Compute the leading principal minors:
  1. $\Delta_1 = \det([4]) = 4 > 0$.
  2. $\Delta_2 = \det\begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix} = 12 - 4 = 8 > 0$.
  3. $\Delta_3 = \det(\mathbf{B})$ expanding along row 1:
     $$
     \det(\mathbf{B}) = 4(6 - 1) - 2(4 - (-1)) - 1(2 - (-3)) = 4(5) - 2(5) - 1(5) = 20 - 10 - 5 = 5 > 0
     $$
  All leading principal minors are strictly positive, confirming $\mathbf{B} \succ 0$.
- **(b)** Lower-triangular Cholesky factor $\mathbf{A} = \mathbf{L}$:
  1. $L_{11} = \sqrt{B_{11}} = \sqrt{4} = 2$.
  2. $L_{21} = B_{21} / L_{11} = 2 / 2 = 1$.
  3. $L_{22} = \sqrt{B_{22} - L_{21}^2} = \sqrt{3 - 1} = \sqrt{2}$.
  4. $L_{31} = B_{31} / L_{11} = -1 / 2 = -0.5$.
  5. $L_{32} = \frac{B_{32} - L_{31}L_{21}}{L_{22}} = \frac{1 - (-0.5)(1)}{\sqrt{2}} = \frac{1.5}{\sqrt{2}} = \frac{3\sqrt{2}}{4}$.
  6. $L_{33} = \sqrt{B_{33} - L_{31}^2 - L_{32}^2} = \sqrt{2 - 0.25 - 1.125} = \sqrt{0.625} = \sqrt{\frac{5}{8}} = \frac{\sqrt{10}}{4}$.

  Thus, the explicit factor loading matrix is:
  $$
  \mathbf{A} = \begin{bmatrix} 2 & 0 & 0 \\ 1 & \sqrt{2} & 0 \\ -0.5 & \frac{3\sqrt{2}}{4} & \frac{\sqrt{10}}{4} \end{bmatrix}, \quad \text{with } \mathbf{A}\mathbf{A}' = \mathbf{B}
  $$
- **(c)** To analyze the candidate matrix $\widetilde{\mathbf{B}}$:
  Compute eigenvalues from $\det(\widetilde{\mathbf{B}} - \lambda \mathbf{I}) = (1 - \lambda)^2 - 1.44 = 0$, giving:
  $$
  \lambda_1 = 1 + 1.2 = 2.2, \quad \lambda_2 = 1 - 1.2 = -0.2
  $$
  Because $\lambda_2 = -0.2 < 0$, $\widetilde{\mathbf{B}}$ is not positive semi-definite ($\widetilde{\mathbf{B}} \notin \mathcal{S}_+^2$).
  The normalized eigenvectors are:
  $$
  \mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} \quad (\text{for } \lambda_1 = 2.2), \qquad \mathbf{q}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix} \quad (\text{for } \lambda_2 = -0.2)
  $$
  Applying Theorem 4.6.2 clips $\lambda_2$ to $\max(0, -0.2) = 0$:
  $$
  \mathbf{B}^* = 2.2 \mathbf{q}_1 \mathbf{q}_1' + 0 \mathbf{q}_2 \mathbf{q}_2' = 2.2 \left( \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \right) = \begin{bmatrix} 1.1 & 1.1 \\ 1.1 & 1.1 \end{bmatrix}
  $$
  $\mathbf{B}^*$ is symmetric, positive semi-definite (eigenvalues $2.2$ and $0$), valid for cokriging, and the unique optimal Frobenius-norm projection of $\widetilde{\mathbf{B}}$ onto $\mathcal{S}_+^2$. $\blacksquare$

---

#### Exercise 4.5: Hand Calculation of Gaussian Profile Likelihood for $n = 3$ Collinear Points
Consider $n = 3$ locations on the real line: $s_1 = 0, s_2 = 1, s_3 = 2$, with constant unknown mean $\mathbf{X} = \mathbf{1}_3$ ($p = 1$). The observed values are $\mathbf{Z} = (1.0, 2.0, 1.0)'$. The process has an exponential correlation kernel $R(s_i, s_j) = e^{-|s_i - s_j|/\phi}$ with zero nugget, so that $\rho = e^{-1/\phi} \in (0, 1)$.
- **(a)** Express the correlation matrix $\mathbf{R}(\rho)$, its determinant $\det(\mathbf{R}(\rho))$, and its inverse $\mathbf{R}(\rho)^{-1}$ as functions of $\rho$.
- **(b)** For candidate correlation $\rho = 0.5$ (spatial range $\phi = 1/\ln(2) \approx 1.44$), calculate the Generalized Least Squares estimate $\widehat{\beta}_{\text{GLS}}$, the residual sum of squares $\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}$, and the profile scale estimate $\widehat{\sigma}^2_{\text{ML}}$.
- **(c)** Evaluate the profile log-likelihood $\ell_p(\rho = 0.5)$ and compare it against the independent case $\rho = 0$ (as $\phi \to 0$).

##### Solution:
- **(a)** For collinear points with exponential correlation, the correlation matrix is:
  $$
  \mathbf{R}(\rho) = \begin{bmatrix} 1 & \rho & \rho^2 \\ \rho & 1 & \rho \\ \rho^2 & \rho & 1 \end{bmatrix}
  $$
  Its determinant is:
  $$
  \det(\mathbf{R}(\rho)) = 1(1 - \rho^2) - \rho(\rho - \rho^3) + \rho^2(\rho^2 - \rho^2) = (1 - \rho^2)(1 - \rho^2) = (1 - \rho^2)^2
  $$
  Its inverse is the classic AR(1) tridiagonal precision matrix:
  $$
  \mathbf{R}(\rho)^{-1} = \frac{1}{1 - \rho^2} \begin{bmatrix} 1 & -\rho & 0 \\ -\rho & 1 + \rho^2 & -\rho \\ 0 & -\rho & 1 \end{bmatrix}
  $$
- **(b)** For $\rho = 0.5$, we have $1 - \rho^2 = 1 - 0.25 = 0.75 = 3/4$, so $\frac{1}{1 - \rho^2} = \frac{4}{3}$.
  $$
  \det(\mathbf{R}(0.5)) = (0.75)^2 = 0.5625
  $$
  $$
  \mathbf{R}^{-1} = \frac{4}{3} \begin{bmatrix} 1 & -0.5 & 0 \\ -0.5 & 1.25 & -0.5 \\ 0 & -0.5 & 1 \end{bmatrix}
  $$
  Compute the GLS denominator $\mathbf{1}'\mathbf{R}^{-1}\mathbf{1}$:
  $$
  \mathbf{R}^{-1}\mathbf{1} = \frac{4}{3} \begin{bmatrix} 0.5 \\ 0.25 \\ 0.5 \end{bmatrix} = \begin{bmatrix} 2/3 \\ 1/3 \\ 2/3 \end{bmatrix} \implies \mathbf{1}'\mathbf{R}^{-1}\mathbf{1} = \frac{2}{3} + \frac{1}{3} + \frac{2}{3} = \frac{5}{3}
  $$
  Compute the GLS numerator $\mathbf{1}'\mathbf{R}^{-1}\mathbf{Z}$:
  $$
  \mathbf{1}'\mathbf{R}^{-1}\mathbf{Z} = (\mathbf{R}^{-1}\mathbf{1})'\mathbf{Z} = \frac{2}{3}(1.0) + \frac{1}{3}(2.0) + \frac{2}{3}(1.0) = \frac{2 + 2 + 2}{3} = 2.0
  $$
  Therefore, the GLS trend estimate is:
  $$
  \widehat{\beta}_{\text{GLS}} = \frac{\mathbf{1}'\mathbf{R}^{-1}\mathbf{Z}}{\mathbf{1}'\mathbf{R}^{-1}\mathbf{1}} = \frac{2.0}{5/3} = \frac{6}{5} = 1.20
  $$
  The residual vector is $\mathbf{e} = \mathbf{Z} - \mathbf{1}\widehat{\beta}_{\text{GLS}} = (1.0 - 1.2, 2.0 - 1.2, 1.0 - 1.2)' = (-0.2, 0.8, -0.2)'$.
  Evaluate the residual quadratic form $\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} = \mathbf{e}'\mathbf{R}^{-1}\mathbf{e}$:
  $$
  \mathbf{R}^{-1}\mathbf{e} = \frac{4}{3} \begin{bmatrix} 1(-0.2) - 0.5(0.8) \\ -0.5(-0.2) + 1.25(0.8) - 0.5(-0.2) \\ -0.5(0.8) + 1(-0.2) \end{bmatrix} = \frac{4}{3} \begin{bmatrix} -0.6 \\ 1.2 \\ -0.6 \end{bmatrix} = \begin{bmatrix} -0.8 \\ 1.6 \\ -0.8 \end{bmatrix}
  $$
  $$
  \mathbf{e}'\mathbf{R}^{-1}\mathbf{e} = (-0.2)(-0.8) + (0.8)(1.6) + (-0.2)(-0.8) = 0.16 + 1.28 + 0.16 = 1.60
  $$
  The profile ML scale estimate is:
  $$
  \widehat{\sigma}^2_{\text{ML}} = \frac{1}{n} \mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} = \frac{1.60}{3} \approx 0.5333
  $$
- **(c)** From Definition 4.2.1, the profile log-likelihood is:
  $$
  \ell_p(\rho) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\left(\widehat{\sigma}^2_{\text{ML}}\right) - \frac{1}{2}\ln\det(\mathbf{R}(\rho)) - \frac{n}{2}
  $$
  For $\rho = 0.5$:
  - $-\frac{3}{2}\ln(2\pi) \approx -1.5(1.83788) = -2.7568$
  - $-\frac{3}{2}\ln(0.5333) \approx -1.5(-0.6286) = +0.9429$
  - $-\frac{1}{2}\ln(0.5625) \approx -0.5(-0.5754) = +0.2877$
  - $-\frac{3}{2} = -1.5000$
  
  Summing these:
  $$
  \ell_p(\rho = 0.5) = -2.7568 + 0.9429 + 0.2877 - 1.5000 \approx -3.026
  $$
  
  For $\rho = 0$ (independent OLS case where $\mathbf{R} = \mathbf{I}_3, \det(\mathbf{R}) = 1$):
  - $\widehat{\beta}_{\text{OLS}} = \frac{1+2+1}{3} = 1.3333$
  - Residuals: $(-0.3333, 0.6667, -0.3333)'$
  - $\mathbf{e}'\mathbf{e} = 2(0.3333^2) + 0.6667^2 = 2/9 + 4/9 = 6/9 = 2/3 \approx 0.6667$
  - $\widehat{\sigma}^2_{\text{ML}}(\rho = 0) = \frac{2/3}{3} = \frac{2}{9} \approx 0.2222$
  - $\ell_p(\rho = 0) = -2.7568 - 1.5\ln(0.2222) - 0 - 1.5 = -2.7568 + 2.2562 - 1.5 = -2.001$
  
  Because the three points $\mathbf{Z} = (1, 2, 1)$ exhibit negative local curvature (up and then down), the model with spatial autocorrelation ($\rho = 0.5$) penalizes the rapid fluctuation compared to independent noise ($\ell_p(0) > \ell_p(0.5)$), demonstrating how the Gaussian likelihood naturally evaluates candidate spatial parameters. $\blacksquare$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 4: Spatial Likelihood and Parametric Estimation — Companion Exercises and Complete Solutions](ch04_exercises.md)
