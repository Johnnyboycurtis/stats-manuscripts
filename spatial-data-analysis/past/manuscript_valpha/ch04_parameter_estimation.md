# Chapter 4: Covariance Parameter Estimation and Theoretical Validations

Once exploratory analysis and empirical semivariograms have revealed the general structure of spatial dependence, spatial inference requires specifying and fitting a valid parametric covariance kernel:
$$
\mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n
$$
where $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$ indexes the structural parameters: partial sill $\sigma^2$, spatial range $\phi$, and nugget effect $\tau^2$. In this chapter, we develop the statistical theory of parameter estimation for spatial linear models:
$$
\mathbf{Z} \sim \mathcal{N}_n\left(\mathbf{X}\boldsymbol{\beta}, \; \mathbf{V}(\boldsymbol{\theta})\right)
$$
Following the pedagogical philosophy of **Maria L. Rizzo** (Beat 1: Mathematical Anchor; Beat 2: Algorithmic Translation; Beat 3: Geometric Intuition and Diagnostic Reality) and the semantic linear-model discipline of **Oliver Schabenberger & Carol A. Gotway**, we examine Gaussian spatial likelihoods, profile Maximum Likelihood (ML), Restricted Maximum Likelihood (REML), Weighted Least Squares (WLS), and the multivariate Linear Model of Coregionalization (LMC).

---

### Classical-to-Spatial Bridge: From the Classical $(n-1)$ Variance Degrees of Freedom Correction to REML Error Contrasts and the Spatial Projection Operator $\mathbf{P}$

In introductory mathematical statistics, students are taught that the sample variance of $n$ independent and identically distributed (i.i.d.) observations $Y_1, \dots, Y_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$ divides by $n - 1$, rather than $n$:
$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2
$$
The uncorrected Maximum Likelihood Estimator (MLE), $\widehat{\sigma}^2_{\text{ML}} = \frac{1}{n}\sum_{i=1}^n (Y_i - \bar{Y})^2$, is systematically biased downward: $\operatorname{E}[\widehat{\sigma}^2_{\text{ML}}] = \frac{n-1}{n}\sigma^2$. The factor $\frac{1}{n-1}$ arises because exactly one degree of freedom was expended estimating the unknown scalar population mean $\mu$ using the sample mean $\bar{Y}$.

```
Classical i.i.d. Estimation (1 Mean Parameter mu):
   Y_1, ..., Y_n  ──>  Estimate mu by Y_bar  ──>  1 d.f. lost  ──>  E[sigma^2_ML] = ((n - 1)/n) * sigma^2

Spatial Regression (p Trend Parameters beta):
   Y ~ N(X beta, V(theta))  ──>  Estimate beta by beta_GLS  ──>  p d.f. lost  ──>  E[sigma^2_ML] = ((n - p)/n) * sigma^2
```

In the spatial linear regression model:
$$
\mathbf{Y} \sim \mathcal{N}_n\left(\mathbf{X}\boldsymbol{\beta}, \; \mathbf{V}(\boldsymbol{\theta})\right)
$$
where $\mathbf{X} \in \mathbb{R}^{n \times p}$ contains $p$ deterministic spatial trend basis functions ($p \ge 1$), estimating the $p$-dimensional coefficient vector $\boldsymbol{\beta}$ causes standard Maximum Likelihood to underestimate the covariance sill $\sigma^2$ by the factor:
$$
\operatorname{E}\left[\widehat{\sigma}^2_{\text{ML}}\right] = \left(\frac{n-p}{n}\right)\sigma^2
$$
When $p$ is small relative to large $n$, this downward bias may seem modest. But in spatial studies with moderate sample sizes ($n = 50$ or $100$) and polynomial drift surfaces or multiple physical covariates ($p = 6$ or $10$), ML underestimates spatial variability by $10\%$ to $20\%$, producing falsely overconfident prediction intervals in subsequent kriging.

To eliminate this bias, Restricted Maximum Likelihood (REML) constructs a linear transformation that filters out the trend $\mathbf{X}\boldsymbol{\beta}$ completely before estimating covariance parameters $\boldsymbol{\theta}$.

#### Constructing the Linear Error Contrast Matrix $\mathbf{A}$
Let $\mathbf{A} \in \mathbb{R}^{n \times (n-p)}$ be a matrix of full column rank $n-p$ whose columns form an orthonormal basis for the orthogonal complement of the column space of $\mathbf{X}$, $\operatorname{col}(\mathbf{X})^\perp$:
$$
\mathbf{A}'\mathbf{X} = \mathbf{0}_{(n-p) \times p}, \quad \mathbf{A}'\mathbf{A} = \mathbf{I}_{n-p}
$$
Now transform the observation vector $\mathbf{Y}$ into an $(n-p)$-dimensional vector of **error contrasts** $\mathbf{U} = \mathbf{A}'\mathbf{Y}$. Because $\mathbf{A}'\mathbf{X} = \mathbf{0}$:
$$
\operatorname{E}[\mathbf{U}] = \operatorname{E}[\mathbf{A}'\mathbf{Y}] = \mathbf{A}'\mathbf{X}\boldsymbol{\beta} = \mathbf{0}_{n-p}
$$
and its covariance matrix is:
$$
\operatorname{Var}(\mathbf{U}) = \operatorname{Var}(\mathbf{A}'\mathbf{Y}) = \mathbf{A}' \mathbf{V}(\boldsymbol{\theta}) \mathbf{A}
$$
Therefore:
$$
\mathbf{U} \sim \mathcal{N}_{n-p}\left(\mathbf{0}, \; \mathbf{A}'\mathbf{V}(\boldsymbol{\theta})\mathbf{A}\right)
$$
The distribution of $\mathbf{U}$ is **completely invariant to the trend coefficients $\boldsymbol{\beta}$**! Maximizing the likelihood of $\mathbf{U}$ isolates covariance estimation on the $(n-p)$-dimensional error subspace, automatically recovering the lost degrees of freedom.

#### Derivation of the Spatial Projection Matrix $\mathbf{P}$
Although the matrix $\mathbf{A}$ depends on the arbitrary choice of basis for $\operatorname{col}(\mathbf{X})^\perp$, the resulting likelihood and quadratic form are uniquely expressed via the **spatial projection matrix** $\mathbf{P}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$:
$$
\mathbf{P}(\boldsymbol{\theta}) = \mathbf{A}\left(\mathbf{A}'\mathbf{V}(\boldsymbol{\theta})\mathbf{A}\right)^{-1}\mathbf{A}'
$$
By projecting $\mathbf{Y}$ through $\mathbf{V}$, one can show via generalized least squares and projection algebra (see Theorem 4.2.1 and Theorem 4.3.3) that $\mathbf{P}(\boldsymbol{\theta})$ admits the closed-form representation without needing to construct $\mathbf{A}$ explicitly:
$$
\mathbf{P}(\boldsymbol{\theta}) = \mathbf{V}(\boldsymbol{\theta})^{-1} - \mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\left(\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}
$$

##### Proposition (Fundamental Invariance Properties of $\mathbf{P}$)
The spatial projection matrix $\mathbf{P} = \mathbf{P}(\boldsymbol{\theta})$ satisfies:
1. **Drift Annihilation:** $\mathbf{P}\mathbf{X} = \mathbf{0}$ and $\mathbf{X}'\mathbf{P} = \mathbf{0}$.
2. **Generalized Idempotency:** $\mathbf{P}\mathbf{V}\mathbf{P} = \mathbf{P}$.

*Proof.*
1. Postmultiplying the closed-form definition by $\mathbf{X}$:
$$
\mathbf{P}\mathbf{X} = \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right)^{-1}\left(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\right) = \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X}\mathbf{I}_p = \mathbf{0}
$$
Transposing yields $\mathbf{X}'\mathbf{P} = \mathbf{0}$.
2. Postmultiplying by $\mathbf{V}\mathbf{P}$:
$$
\mathbf{P}\mathbf{V}\mathbf{P} = \left(\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\right)\mathbf{V}\mathbf{P} = \mathbf{I}_n\mathbf{P} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\underbrace{(\mathbf{X}'\mathbf{P})}_{=\mathbf{0}} = \mathbf{P} - \mathbf{0} = \mathbf{P}
$$
$\blacksquare$

#### Computational Scalability and the Bridge to Composite Likelihood
Evaluating either the full Gaussian log-likelihood or the REML objective requires the Cholesky factorization of the $n \times n$ dense spatial covariance matrix $\mathbf{V}(\boldsymbol{\theta})$, which incurs an exact computational cost of $\mathcal{O}(n^3)$ floating-point operations and $\mathcal{O}(n^2)$ memory storage. When $n > 10^4$, computing $|\mathbf{V}(\boldsymbol{\theta})|$ and $\mathbf{V}(\boldsymbol{\theta})^{-1}$ becomes computationally prohibitive on standard workstations.

To circumvent this bottleneck, **Composite Likelihood (CL)** approximates the intractable joint likelihood of all $n$ locations by the product of lower-dimensional marginal or conditional likelihoods over subsets of data points:
$$
\mathcal{L}_{\text{CL}}(\boldsymbol{\theta}; \mathbf{Y}) = \prod_{i=1}^{n-1}\prod_{j=i+1}^n \left[f(Y(\mathbf{s}_i), Y(\mathbf{s}_j); \boldsymbol{\theta})\right]^{w_{ij}}
$$
where $w_{ij} \in \{0, 1\}$ is a screening indicator that retains only pairs separated by distances below a cutoff radius $\|\mathbf{s}_i - \mathbf{s}_j\| \le d_{\max}$. Because each pair involves only a $2 \times 2$ covariance matrix $\mathbf{V}_{ij}$, computing the pairwise composite log-likelihood requires no large matrix inversions or determinants, scaling linearly with the number of active pairs $\mathcal{O}(n \cdot K)$ where $K$ is the average number of neighbors within $d_{\max}$. Composite likelihood estimates $\widehat{\boldsymbol{\theta}}_{\text{CL}}$ are strongly consistent and asymptotically normal under increasing-domain asymptotics, offering a scalable surrogate when evaluating $|\mathbf{V}|$ is prohibitive.

---

### 4.1 The Spatial Linear Model and Gaussian Likelihood Formulation

#### Beat 1: The Mathematical Anchor (The "Why")

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
Under the joint Gaussian law $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}(\boldsymbol{\theta}))$ (or equivalently denoted $\mathbf{Y} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}(\boldsymbol{\theta}))$), the log-likelihood function $\ell(\boldsymbol{\beta}, \boldsymbol{\theta}; \mathbf{Z})$ is:
$$
\ell(\boldsymbol{\beta}, \boldsymbol{\theta}; \mathbf{Z}) = -\frac{n}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
$$

##### Term-by-Term Equation Anatomy: Full Gaussian Spatial Log-Likelihood
$$
\ell(\boldsymbol{\beta}, \boldsymbol{\theta}) = \underbrace{-\frac{n}{2}\ln(2\pi)}_{\text{Normalization Constant}} \underbrace{-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|}_{\text{Volume / Dispersion Penalty}} \underbrace{-\frac{1}{2}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}(\boldsymbol{\theta})^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})}_{\text{Generalized Mahalanobis Residual Form}}
$$
- **$-\frac{n}{2}\ln(2\pi)$:** Geometric normalization factor of the $n$-dimensional Gaussian probability density integrating to unity over $\mathbb{R}^n$.
- **$-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|$:** Log-determinant of the spatial covariance matrix, measuring the generalized variance or volume of the error dispersion ellipsoid. As spatial correlation increases or variance shrinks, $|\mathbf{V}|$ decreases, pushing this term upward; simultaneously, it penalizes over-parameterization by growing large if variance or range are inflated.
- **$-\frac{1}{2}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}(\boldsymbol{\theta})^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$:** Squared Mahalanobis distance measuring goodness-of-fit. Residuals $\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}$ (or $\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}$) are decorrelated and scaled by the inverse covariance operator $\mathbf{V}^{-1}$; deviations along directions of low spatial variance are heavily penalized.

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
2. Let $\lambda_1, \dots, \lambda_n$ denote the strictly positive eigenvalues of $\mathbf{V}$. Since $\det(\mathbf{V}) = \prod_{i=1}^n \lambda_i$, we have $\ln\det(\mathbf{V}) = \sum_{i=1}^n \ln \lambda_i = \operatorname{Tr}(\ln \mathbf{V})$. Taking differentials:
   $$
   d \ln\det(\mathbf{V}) = \operatorname{Tr}(\mathbf{V}^{-1} d\mathbf{V}) \implies \frac{\partial \ln\det(\mathbf{V})}{\partial \theta_k} = \operatorname{Tr}\left(\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k}\right)
   $$
   which completes the proof. $\blacksquare$

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
Let $\boldsymbol{\psi} = (\boldsymbol{\beta}', \boldsymbol{\theta}')' \in \mathbb{R}^{p+q}$. The expected Fisher Information matrix:
$$
\mathcal{I}(\boldsymbol{\psi}) = \operatorname{E}\left[ -\nabla_{\boldsymbol{\psi}}^2 \ell(\boldsymbol{\psi}; \mathbf{Z}) \right] = \operatorname{Cov}\left(\mathbf{S}(\boldsymbol{\psi})\right)
$$
is strictly block-diagonal:
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
3. For $\mathcal{I}_{\boldsymbol{\theta}\boldsymbol{\theta}}$:
   Let $\boldsymbol{\varepsilon} = \mathbf{Z} - \mathbf{X}\boldsymbol{\beta} \sim \mathcal{N}_n(\mathbf{0}, \mathbf{V})$. Define $\mathbf{A}_k = \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1}$.
   The score component is $S_{\theta_k} = \frac{1}{2}(\boldsymbol{\varepsilon}'\mathbf{A}_k \boldsymbol{\varepsilon} - \operatorname{Tr}(\mathbf{V}\mathbf{A}_k))$.
   Using the Gaussian quadratic form covariance theorem $\operatorname{Cov}(\boldsymbol{\varepsilon}'\mathbf{A}\boldsymbol{\varepsilon}, \boldsymbol{\varepsilon}'\mathbf{B}\boldsymbol{\varepsilon}) = 2\operatorname{Tr}(\mathbf{A}\mathbf{V}\mathbf{B}\mathbf{V})$:
   $$
   \mathcal{I}_{\theta_k \theta_l} = \operatorname{Cov}(S_{\theta_k}, S_{\theta_l}) = \frac{1}{4} \cdot 2\operatorname{Tr}(\mathbf{A}_k \mathbf{V} \mathbf{A}_l \mathbf{V}) = \frac{1}{2} \operatorname{Tr}\left(\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_k} \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta_l}\right)
   $$
   establishing block-diagonality. $\blacksquare$

---

#### Beat 2: Algorithmic Translation (The "How")

Direct inversion of $\mathbf{V}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$ is numerically unstable and computationally inefficient ($\mathcal{O}(n^3)$ flops). Stable likelihood evaluation relies on the **Cholesky factorization**:
$$
\mathbf{V}(\boldsymbol{\theta}) = \mathbf{L}\mathbf{L}'
$$
where $\mathbf{L} \in \mathbb{R}^{n \times n}$ is lower-triangular with strictly positive diagonal entries.
1. The log-determinant is computed from the diagonal of $\mathbf{L}$:
   $$
   \ln\det(\mathbf{V}) = \ln\det(\mathbf{L}\mathbf{L}') = 2\sum_{i=1}^n \ln L_{ii}
   $$
2. The quadratic form $(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})$ is evaluated by solving the triangular linear system $\mathbf{L}\mathbf{w} = \mathbf{Z} - \mathbf{X}\boldsymbol{\beta}$ via forward substitution, yielding:
   $$
   (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})' \mathbf{V}^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{w}'\mathbf{w} = \|\mathbf{w}\|_2^2
   $$

```python
def compute_gaussian_log_likelihood(Z, X, beta, theta, coords, cov_func):
    """
    Evaluates the Gaussian spatial log-likelihood matching Definition 4.1.1
    using stable pure Python Cholesky decomposition.
    
    Parameters:
        Z: list of floats, shape (n,), observations
        X: list of lists, shape (n, p), design matrix
        beta: list of floats, shape (p,), drift parameters
        theta: tuple (sigma2, phi, tau2), covariance parameters
        coords: list of lists, shape (n, d), spatial sites s_i
        cov_func: callable, C(h, phi) returning correlation for scalar lag h
        
    Returns:
        log_lik: float, Gaussian log-likelihood value
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
    
    # 5. Solve L w = e by forward substitution
    w = solve_triangular_lower(L, residuals)
    quad_form = sum(val * val for val in w)
    
    # 6. Gaussian log-likelihood
    log_lik = -0.5 * (n * math.log(2.0 * math.pi) + log_det_V + quad_form)
    return log_lik
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

1. **Condition Numbers and Matrix Conditioning:**
   The numerical condition number of the spatial covariance matrix is $\kappa(\mathbf{V}) = \lambda_{\max}(\mathbf{V}) / \lambda_{\min}(\mathbf{V})$.
   When sampling points $\mathbf{s}_i$ and $\mathbf{s}_j$ are closely clustered ($\|\mathbf{s}_i - \mathbf{s}_j\| \ll \phi$), rows $i$ and $j$ of $\mathbf{V}$ become nearly identical. If $\tau^2 = 0$ (no nugget), $\kappa(\mathbf{V})$ frequently exceeds $10^{16}$, causing standard Cholesky decomposition to fail due to machine-precision singularity.
2. **The Microscale Nugget as Tikhonov Regularization:**
   The microscale measurement error $\tau^2 \mathbf{I}_n$ shifts every eigenvalue of $\mathbf{C}(\boldsymbol{\theta})$ upward by $\tau^2$:
   $$
   \lambda_i(\mathbf{V}) = \lambda_i(\mathbf{C}) + \tau^2 \ge \tau^2 > 0
   $$
   The nugget parameter acts mathematically as an intrinsic **Tikhonov regularization parameter**, bounding $\lambda_{\min}(\mathbf{V}) \ge \tau^2$ and ensuring that $\mathbf{V}$ remains strictly invertible regardless of how tightly clustered sample locations are.

---

### 4.2 Profile Maximum Likelihood Estimation

#### Beat 1: The Mathematical Anchor (The "Why")

Setting the drift score equation to zero:
$$
\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}
$$
yields the Generalized Least Squares (GLS) estimator for $\boldsymbol{\beta}$ as an explicit analytical function of $\boldsymbol{\theta}$:
$$
\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = \left(\mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{Z}
$$

##### Definition 4.2.1 (Profile Likelihood and the Spatial Projection Operator)
Define the spatial projection operator $\mathbf{P}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$:
$$
\mathbf{P}(\boldsymbol{\theta}) = \mathbf{V}(\boldsymbol{\theta})^{-1} - \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X} \left(\mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1} \mathbf{X}\right)^{-1} \mathbf{X}' \mathbf{V}(\boldsymbol{\theta})^{-1}
$$
Substituting $\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})$ into the residual quadratic form:
$$
(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}))' \mathbf{V}(\boldsymbol{\theta})^{-1} (\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})) = \mathbf{Z}' \mathbf{P}(\boldsymbol{\theta}) \mathbf{Z}
$$
The *profile log-likelihood function* for the covariance parameters $\boldsymbol{\theta}$ is:
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
Consider a scale-parameter covariance structure $\mathbf{V}(\sigma^2, \boldsymbol{\phi}) = \sigma^2 \mathbf{R}(\boldsymbol{\phi})$ where $\mathbf{R}(\boldsymbol{\phi})$ is a correlation matrix. For any fixed $\boldsymbol{\phi}$, the profile MLE of $\sigma^2$ is:
$$
\widehat{\sigma}^2_{\text{ML}}(\boldsymbol{\phi}) = \frac{1}{n} \mathbf{Z}' \mathbf{P}_{\mathbf{R}}(\boldsymbol{\phi}) \mathbf{Z}
$$
where $\mathbf{P}_{\mathbf{R}} = \mathbf{R}^{-1} - \mathbf{R}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{R}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{R}^{-1}$. Its mathematical expectation is:
$$
\operatorname{E}\left[\widehat{\sigma}^2_{\text{ML}}(\boldsymbol{\phi})\right] = \left(\frac{n - p}{n}\right)\sigma^2 < \sigma^2
$$
Maximum Likelihood systematically underestimates the variance components by failing to account for the $p$ degrees of freedom consumed in estimating the drift parameter vector $\boldsymbol{\beta}$.

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

---

#### Beat 2: Algorithmic Translation (The "How")

Evaluating the profile log-likelihood requires computing $\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})$ and $\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}$ without forming $\mathbf{V}^{-1}$ explicitly:

```python
def compute_profile_ml_log_likelihood(theta, X, Z, coords, cov_func):
    """
    Evaluates the profile Maximum Likelihood log-likelihood ell_p(theta)
    matching Definition 4.2.1 in pure standard Python.
    """
    n = len(Z)
    p = len(X[0])
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Covariance matrix assembly
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            V[i][j] = sigma2 * cov_func(dist, phi) + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization of V
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 3. Solve L W_x = X and L u_z = Z
    # W_x has dimensions (n, p), columns solved separately
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
    
    # 5. Factorize G and solve for beta_GLS
    L_g = cholesky_decompose(G)
    if L_g is None:
        return -math.inf
    beta_gls = solve_cholesky_system(L_g, rhs)
    
    # 6. Residual quadratic form: ||u_z - W_x beta_gls||^2
    quad_form_P = 0.0
    for i in range(n):
        res_i = u_z[i] - sum(W_x[i][j] * beta_gls[j] for j in range(p))
        quad_form_P += res_i * res_i
        
    # 7. Profile log-likelihood
    log_lik_p = -0.5 * (n * math.log(2.0 * math.pi) + log_det_V + quad_form_P)
    return log_lik_p
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

1. **Why ML Downward Bias Occurs:**
   Maximum Likelihood treats the estimated drift $\mathbf{X}\widehat{\boldsymbol{\beta}}$ as if it were the true known population mean $\mathbf{X}\boldsymbol{\beta}$. Because $\widehat{\boldsymbol{\beta}}$ is fitted to minimize generalized residual sums of squares, the residuals $\mathbf{e} = \mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}$ are constrained to be orthogonal to the columns of $\mathbf{X}$ ($\mathbf{X}'\mathbf{V}^{-1}\mathbf{e} = \mathbf{0}$). These $p$ linear constraints restrict residual variability, artificially compressing the estimated sill and nugget parameters.
2. **Kriging Impact:**
   In spatial prediction, underestimating $\sigma^2$ and $\tau^2$ causes the kriging prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ to be artificially small. Confidence intervals $\widehat{Z}(\mathbf{s}_0) \pm 1.96 \sigma_{\text{OK}}(\mathbf{s}_0)$ under-cover the truth, conveying false precision.

---

### 4.3 Restricted Maximum Likelihood (REML)

#### Beat 1: The Mathematical Anchor (The "Why")

To eliminate the downward bias of ML, H. D. Patterson and R. Thompson (1971), and later David A. Harville (1974), developed Restricted (or Residual) Maximum Likelihood (REML). The foundational idea is to transform $\mathbf{Z}$ into a vector of linear combinations (error contrasts) whose distribution is completely free of the nuisance drift parameter vector $\boldsymbol{\beta}$.

##### Definition 4.3.1 (Error Contrasts and Annihilator Matrices)
A linear combination $\mathbf{a}'\mathbf{Z}$ is an *error contrast* if its mathematical expectation is identically zero for all $\boldsymbol{\beta} \in \mathbb{R}^p$:
$$
\operatorname{E}[\mathbf{a}'\mathbf{Z}] = \mathbf{a}'\mathbf{X}\boldsymbol{\beta} = 0, \quad \forall \boldsymbol{\beta} \in \mathbb{R}^p \iff \mathbf{a}'\mathbf{X} = \mathbf{0}'
$$
Let $\mathbf{K} \in \mathbb{R}^{n \times (n-p)}$ be a matrix of full column rank $n-p$ whose columns span the null space of $\mathbf{X}'$:
$$
\mathbf{K}'\mathbf{X} = \mathbf{0}_{(n-p) \times p}
$$
The transformed vector $\mathbf{W} = \mathbf{K}'\mathbf{Z} \in \mathbb{R}^{n-p}$ is a maximal vector of linearly independent error contrasts.

##### Theorem 4.3.1 (Distribution of Error Contrasts)
The error contrast vector $\mathbf{W} = \mathbf{K}'\mathbf{Z}$ follows the $(n-p)$-variate Gaussian distribution:
$$
\mathbf{W} \sim \mathcal{N}_{n-p}\left(\mathbf{0}, \; \mathbf{K}'\mathbf{V}(\boldsymbol{\theta})\mathbf{K}\right)
$$
which does not depend on the unknown regression parameter $\boldsymbol{\beta}$.

##### Theorem 4.3.2 (Invariance to Contrast Matrix Choice)
Let $\mathbf{K}_1, \mathbf{K}_2 \in \mathbb{R}^{n \times (n-p)}$ be two distinct full column rank matrices satisfying $\mathbf{K}_1'\mathbf{X} = \mathbf{0}$ and $\mathbf{K}_2'\mathbf{X} = \mathbf{0}$. Then the log-likelihoods of $\mathbf{W}_1 = \mathbf{K}_1'\mathbf{Z}$ and $\mathbf{W}_2 = \mathbf{K}_2'\mathbf{Z}$ differ by a constant that is independent of both the data $\mathbf{Z}$ and the covariance parameters $\boldsymbol{\theta}$.

*Proof.*
Because the columns of both $\mathbf{K}_1$ and $\mathbf{K}_2$ form bases for the orthogonal complement of the column space of $\mathbf{X}$ (which has dimension $n-p$), there exists a unique non-singular matrix $\mathbf{M} \in \mathbb{R}^{(n-p) \times (n-p)}$ such that $\mathbf{K}_2 = \mathbf{K}_1 \mathbf{M}$.
Then $\mathbf{W}_2 = \mathbf{M}'\mathbf{W}_1$ and $\mathbf{K}_2'\mathbf{V}\mathbf{K}_2 = \mathbf{M}'(\mathbf{K}_1'\mathbf{V}\mathbf{K}_1)\mathbf{M}$.
The log-likelihood of $\mathbf{W}_2$ is:
$$
\ell(\boldsymbol{\theta}; \mathbf{W}_2) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln\det\left(\mathbf{M}'(\mathbf{K}_1'\mathbf{V}\mathbf{K}_1)\mathbf{M}\right) - \frac{1}{2}\mathbf{W}_1'\mathbf{M}\left[\mathbf{M}'(\mathbf{K}_1'\mathbf{V}\mathbf{K}_1)\mathbf{M}\right]^{-1}\mathbf{M}'\mathbf{W}_1
$$
Using $\det(\mathbf{M}'\mathbf{A}\mathbf{M}) = (\det(\mathbf{M}))^2 \det(\mathbf{A})$:
$$
\ell(\boldsymbol{\theta}; \mathbf{W}_2) = \ell(\boldsymbol{\theta}; \mathbf{W}_1) - \ln|\det(\mathbf{M})|
$$
Because $\mathbf{M}$ does not depend on $\boldsymbol{\theta}$, the score equations, Fisher Information, and parameter estimates are identical. $\blacksquare$

##### Theorem 4.3.3 (Derivation of the Harville REML Log-Likelihood)
The log-likelihood of the error contrasts $\mathbf{W} = \mathbf{K}'\mathbf{Z}$ can be written without explicit construction of the matrix $\mathbf{K}$ as:
$$
\ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2}\ln\det\left(\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\right) - \frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}
$$
where $\mathbf{P}(\boldsymbol{\theta})$ is the spatial projection operator from Definition 4.2.1.

*Proof.*
Consider the full-rank linear transformation matrix $\mathbf{T} = [(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}, \mathbf{K}]' \in \mathbb{R}^{n \times n}$ where $\mathbf{K} \in \mathbb{R}^{n \times (n-p)}$ satisfies $\mathbf{K}'\mathbf{X} = \mathbf{0}$ and $\operatorname{rank}(\mathbf{K}) = n - p$.
Applying $\mathbf{T}$ to the random field vector $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V})$:
$$
\mathbf{T}\mathbf{Z} = \begin{bmatrix} (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} \\ \mathbf{K}'\mathbf{Z} \end{bmatrix} = \begin{bmatrix} \widehat{\boldsymbol{\beta}}_{\text{GLS}} \\ \mathbf{W} \end{bmatrix}
$$
We evaluate the cross-covariance between the GLS drift estimator and the error contrast vector $\mathbf{W}$:
$$
\operatorname{Cov}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{W}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\operatorname{Var}(\mathbf{Z})\mathbf{K} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{V}\mathbf{K} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{K} = \mathbf{0}
$$
Because $\widehat{\boldsymbol{\beta}}_{\text{GLS}}$ and $\mathbf{W}$ are jointly Gaussian with zero cross-covariance, they are statistically independent. The marginal distribution of the GLS estimator is $\widehat{\boldsymbol{\beta}}_{\text{GLS}} \sim \mathcal{N}_p(\boldsymbol{\beta}, (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1})$, which contains all information regarding the mean drift coefficients $\boldsymbol{\beta}$. Consequently, the joint density factors as the product of independent marginal distributions:
$$
f_{\mathbf{T}\mathbf{Z}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}, \mathbf{w}; \boldsymbol{\beta}, \boldsymbol{\theta}) = f_{\widehat{\boldsymbol{\beta}}_{\text{GLS}}}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}; \boldsymbol{\beta}, \boldsymbol{\theta}) \cdot f_{\mathbf{W}}(\mathbf{w}; \boldsymbol{\theta})
$$
Applying the Jacobian change-of-variables theorem with determinant identity $|\det(\mathbf{T})| = |\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}|^{-1/2} |\mathbf{K}'\mathbf{V}\mathbf{K}|^{1/2} |\det(\mathbf{V})|^{-1/2}$ and factoring out the profile drift evaluated at $\boldsymbol{\beta} = \widehat{\boldsymbol{\beta}}_{\text{GLS}}$, the log-likelihood of the error contrasts simplifies to:
$$
\ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2}\ln\det\left(\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}\right) - \frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}
$$
which completes the derivation. $\blacksquare$

##### Term-by-Term Equation Anatomy: REML Restricted Log-Likelihood
$$
\ell_R(\boldsymbol{\theta}) = \underbrace{-\frac{n-p}{2}\ln(2\pi)}_{\text{Contrast Subspace Normalization}} \underbrace{-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|}_{\text{Spatial Dispersion Volume}} \underbrace{-\frac{1}{2}\ln|\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}|}_{\text{Trend Uncertainty Penalty}} \underbrace{-\frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}}_{\text{Projected Error Contrast Form}}
$$
- **$-\frac{n-p}{2}\ln(2\pi)$:** Normalization constant corresponding to the dimension of the error contrast subspace $\mathbb{R}^{n-p}$, reflecting that exactly $p$ degrees of freedom have been integrated out.
- **$-\frac{1}{2}\ln|\mathbf{V}(\boldsymbol{\theta})|$:** Standard spatial covariance log-determinant measuring total stochastic dispersion across the sampling domain.
- **$-\frac{1}{2}\ln|\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X}|$:** Occam's penalty for large-scale trend estimation uncertainty. Because the Generalized Least Squares estimator of drift has covariance $\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$, the determinant $|\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}|$ equals $1 / |\operatorname{Var}(\widehat{\boldsymbol{\beta}}_{\text{GLS}})|$. A candidate covariance matrix $\mathbf{V}(\boldsymbol{\theta})$ that attributes too much variance to the deterministic trend causes $|\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}|$ to inflate, which subtracts a larger penalty from $\ell_R(\boldsymbol{\theta})$. This prevents the likelihood from soaking up real spatial correlation into spurious trend confidence.
- **$-\frac{1}{2}\mathbf{Y}'\mathbf{P}(\boldsymbol{\theta})\mathbf{Y}$:** Quadratic form of the data vector $\mathbf{Y}$ (or $\mathbf{Z}$) projected onto the orthogonal complement of the trend space via $\mathbf{P}$. Because $\mathbf{P}\mathbf{X} = \mathbf{0}$, this term is identical to $(\mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{GLS}})'\mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}_{\text{GLS}})$, evaluating fit purely on spatial residuals.

##### Theorem 4.3.4 (Exact Unbiasedness of REML Sills)
Under the scale structure $\mathbf{V}(\sigma^2, \boldsymbol{\phi}) = \sigma^2 \mathbf{R}(\boldsymbol{\phi})$, the REML estimator of $\sigma^2$ is:
$$
\widehat{\sigma}^2_{\text{REML}}(\boldsymbol{\phi}) = \frac{1}{n - p} \mathbf{Z}' \mathbf{P}_{\mathbf{R}}(\boldsymbol{\phi}) \mathbf{Z}
$$
Its mathematical expectation is strictly unbiased:
$$
\operatorname{E}\left[\widehat{\sigma}^2_{\text{REML}}(\boldsymbol{\phi})\right] = \sigma^2
$$

*Proof.*
Follows directly from $\operatorname{E}[\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}] = (n-p)\sigma^2$ (Theorem 4.2.1) divided by $n-p$. $\blacksquare$

##### Theorem 4.3.5 (REML Score Equations and Fisher Information)
1. The REML score equations for the covariance parameters are:
   $$
   \frac{\partial \ell_{\text{REML}}}{\partial \theta_k} = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}\right) + \frac{1}{2}\mathbf{Z}'\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}\mathbf{P}(\boldsymbol{\theta})\mathbf{Z}
   $$
2. The REML Fisher Information matrix is:
   $$
   \mathcal{I}_{\text{REML}}(\boldsymbol{\theta})_{kl} = \frac{1}{2}\operatorname{Tr}\left(\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_k}\mathbf{P}(\boldsymbol{\theta})\frac{\partial \mathbf{V}(\boldsymbol{\theta})}{\partial \theta_l}\right)
   $$

---

#### Beat 2: Algorithmic Translation (The "How")

The following pseudocode implements the REML objective function via Cholesky factorizations of both $\mathbf{V}$ and $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$:

```python
def compute_reml_log_likelihood(theta, X, Z, coords, cov_func):
    """
    Evaluates Harville's REML log-likelihood ell_REML(theta) matching Theorem 4.3.3
    in pure standard Python.
    """
    n = len(Z)
    p = len(X[0])
    d = len(coords[0])
    sigma2, phi, tau2 = theta
    
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return -math.inf
        
    # 1. Covariance matrix assembly
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            V[i][j] = sigma2 * cov_func(dist, phi) + (tau2 if i == j else 0.0)
            
    # 2. Cholesky factorization of V
    L = cholesky_decompose(V)
    if L is None:
        return -math.inf
        
    log_det_V = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    
    # 3. Solve L W_x = X and L u_z = Z
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
    
    # 5. Factorize G to compute log|G| and beta_GLS
    L_g = cholesky_decompose(G)
    if L_g is None:
        return -math.inf
        
    log_det_G = 2.0 * sum(math.log(L_g[i][i]) for i in range(p))
    beta_gls = solve_cholesky_system(L_g, rhs)
    
    # 6. Projected residual quadratic form: Z' P Z = ||u_z - W_x beta_gls||^2
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

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

```
ML vs. REML Likelihood Anatomy:
+-------------------------------------------------------------------------------+
| Profile ML:    - 0.5 * [ n ln(2π)     + ln det(V)             + Z' P Z ]     |
| REML:          - 0.5 * [ (n-p) ln(2π) + ln det(V) + ln det(G) + Z' P Z ]     |
+-------------------------------------------------------------------------------+
                                                      │
                                                      ▼
                       Occam's Penalty for Trend Estimation Uncertainty:
                       ln det(G) = ln det(X' V^-1 X) restores the p lost
                       degrees of freedom, eliminating variance downward bias!
```

1. **The Role of the Log-Determinant Penalty:**
   The REML objective function augments the profile likelihood with $+\frac{1}{2}\ln\det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1} = -\frac{1}{2}\ln\det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})$.
   This term represents the volume of the confidence ellipsoid for the drift parameters $\boldsymbol{\beta}$. When a candidate covariance matrix $\mathbf{V}$ attributes too much variation to the trend, $\det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})$ inflates, penalizing the likelihood.
2. **Degrees-of-Freedom Compensation:**
   By evaluating likelihood only on the $(n-p)$-dimensional subspace orthogonal to the drift basis $\mathbf{X}$, REML automatically inflates the variance estimates by the factor $\frac{n}{n-p}$, exactly eliminating the systematic downward bias of ML.

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why do we prefer REML over Maximum Likelihood for covariance parameter estimation, but need full ML when comparing different trend designs $\mathbf{X}$?*  
> When comparing alternative covariance models with the *same* trend design $\mathbf{X}$, REML is strictly superior because it eliminates the $(n-p)/n$ downward bias in variance components by filtering out the fixed effects through error contrasts $\mathbf{U} = \mathbf{A}'\mathbf{Y}$. However, the contrast matrix $\mathbf{A} \in \mathbb{R}^{n \times (n-p)}$ satisfies $\mathbf{A}'\mathbf{X} = \mathbf{0}$, meaning $\mathbf{A}$ depends explicitly on the design matrix $\mathbf{X}$. If you change $\mathbf{X}$ (for instance, comparing an intercept-only model $\mathbf{X}_1 \in \mathbb{R}^{n \times 1}$ against a planar trend $\mathbf{X}_2 \in \mathbb{R}^{n \times 3}$), the respective contrast matrices $\mathbf{A}_1$ and $\mathbf{A}_2$ project the observations $\mathbf{Y}$ onto entirely different linear subspaces of dimensions $n-1$ and $n-3$. Likelihoods computed on different data vectors ($\mathbf{U}_1 = \mathbf{A}_1'\mathbf{Y}$ versus $\mathbf{U}_2 = \mathbf{A}_2'\mathbf{Y}$) live on different measure spaces and have non-comparable normalizing constants! Therefore, model selection across different trend designs using AIC or BIC *must* be evaluated using full Maximum Likelihood $\ell(\widehat{\boldsymbol{\beta}}_{\text{ML}}, \widehat{\boldsymbol{\theta}}_{\text{ML}}; \mathbf{Y})$, never REML.

---

### 4.4 Weighted and Generalized Least Squares Variogram Fitting

#### Beat 1: The Mathematical Anchor (The "Why")

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
   Recognizing that computing $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}}$ at every step is as demanding as REML, Noel Cressie (1985) proposed approximating $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}}$ by its diagonal variance under Gaussian increments:
   $$
   \operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k)) \approx \frac{2\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2}{|N(\mathbf{h}_k)|}
   $$
   This yields Cressie's WLS criterion:
   $$
   Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K |N(\mathbf{h}_k)| \left( \frac{\widehat{\gamma}(\mathbf{h}_k)}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})} - 1 \right)^2 = \sum_{k=1}^K |N(\mathbf{h}_k)| \left[ \frac{\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})} \right]^2
   $$

##### Term-by-Term Equation Anatomy: Cressie's WLS Criterion
$$
Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K \underbrace{|N(\mathbf{h}_k)|}_{\text{Pair Count Weight}} \left[ \underbrace{\frac{\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})}}_{\text{Relative Semivariance Residual}} \right]^2 = \sum_{k=1}^K \frac{|N(\mathbf{h}_k)|}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2} \left[ \widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta}) \right]^2
$$
- **$|N(\mathbf{h}_k)|$:** Number of point pairs falling into lag bin $k$. Bins with dense sample pairs have smaller empirical variance and thus receive proportionally greater influence in parameter estimation.
- **$\left[\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})\right]^2$:** Standard squared error between empirical semivariance and theoretical parametric curve at separation lag $\mathbf{h}_k$.
- **$\frac{1}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2}$:** Inverse squared theoretical variance weighting. Under Gaussian increments, $\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k)) \approx \frac{2\gamma(\mathbf{h}_k; \boldsymbol{\theta})^2}{|N(\mathbf{h}_k)|}$. Weighting by $1 / \gamma^2$ heavily penalizes discrepancies at short lags where $\gamma$ is small, ensuring the fitted semivariogram matches the crucial near-origin behavior (nugget effect and initial slope) which dictates kriging weights.

##### Theorem 4.4.1 (Asymptotic Properties and Efficiency of WLS)
Under increasing-domain asymptotics where $|N(\mathbf{h}_k)| \to \infty$ for all $k = 1, \dots, K$:
1. The WLS estimator $\widehat{\boldsymbol{\theta}}_{\text{WLS}} = \arg\min_{\boldsymbol{\theta} \in \Theta} Q_{\text{WLS}}(\boldsymbol{\theta})$ is strongly consistent: $\widehat{\boldsymbol{\theta}}_{\text{WLS}} \xrightarrow{a.s.} \boldsymbol{\theta}_0$.
2. The asymptotic distribution is:
   $$
   \sqrt{n}\left(\widehat{\boldsymbol{\theta}}_{\text{WLS}} - \boldsymbol{\theta}_0\right) \xrightarrow{d} \mathcal{N}_q\left(\mathbf{0}, \; \mathbf{H}^{-1}\mathbf{J}\mathbf{H}^{-1}\right)
   $$
   where $\mathbf{H} = \sum_{k=1}^K \frac{|N(\mathbf{h}_k)|}{\gamma(\mathbf{h}_k; \boldsymbol{\theta}_0)^2} (\nabla_{\boldsymbol{\theta}}\gamma(\mathbf{h}_k)) (\nabla_{\boldsymbol{\theta}}\gamma(\mathbf{h}_k))'$ and $\mathbf{J} = \lim_{n \to \infty} \operatorname{Var}(\nabla_{\boldsymbol{\theta}} Q_{\text{WLS}})$.
3. Because WLS neglects off-diagonal covariances $\operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_p), \widehat{\gamma}(\mathbf{h}_q)) \ne 0$, its asymptotic covariance satisfies the Loewner ordering:
   $$
   \operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{REML}}) \le \operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{GLS}}) \le \operatorname{Var}(\widehat{\boldsymbol{\theta}}_{\text{WLS}})
   $$

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Wait—What is microergodicity, and why can't we estimate $\sigma^2$ and $\phi$ separately under infill asymptotics?*  
> In spatial statistics, asymptotic behavior depends fundamentally on whether sampling occurs on an expanding region (increasing-domain asymptotics) or more densely within a fixed bounded region (infill asymptotics). Under infill asymptotics in $\mathbb{R}^d$ ($d \le 3$), two Gaussian measures with Matérn covariance kernels having different partial sills $\sigma^2$ and range parameters $\phi$ are mutually absolutely continuous (equivalent) if and only if the ratio $\sigma^2 / \phi^{2\nu}$ (or $\sigma^2 / \phi$ for the exponential kernel $\nu = 1/2$) remains identical! Because the likelihood ratio tends to a non-degenerate random variable rather than concentrating at a single point, $\sigma^2$ and $\phi$ are not individually consistently estimable from a single realization in a bounded domain. Only the combined parameter ratio $c = \sigma^2 / \phi^{2\nu}$—termed the **microergodic parameter**—can be estimated consistently. In practical terms, attempting to separately estimate $\sigma^2$ and $\phi$ with fine-scale infill data yields severe parameter trade-offs and elongated likelihood ridges, whereas the kriging predictions themselves remain entirely stable because they depend strictly on the microergodic ratio!

---

#### Beat 2: Algorithmic Translation (The "How")

```python
import math

def fit_variogram_cressie_wls(lags, gamma_hat, pair_counts, initial_theta, model_type='exponential'):
    """
    Fits parametric semivariogram parameters via Cressie's WLS criterion
    matching Definition 4.4.1 using a pure Python Nelder-Mead simplex optimizer.
    
    Parameters:
        lags: list of floats of length K, lag distances h_k
        gamma_hat: list of floats of length K, empirical semivariances
        pair_counts: list of ints of length K, pair counts |N(h_k)|
        initial_theta: tuple (sigma2_0, phi_0, tau2_0), starting guess
        model_type: str, 'exponential' or 'spherical'
        
    Returns:
        opt_theta: tuple (sigma2, phi, tau2), fitted parameters
    """
    def theoretical_gamma(h, theta):
        sigma2, phi, tau2 = theta
        if model_type == 'exponential':
            return tau2 + sigma2 * (1.0 - math.exp(-h / phi))
        elif model_type == 'spherical':
            hr = h / phi
            sph = 1.5 * hr - 0.5 * (hr ** 3) if hr <= 1.0 else 1.0
            return tau2 + sigma2 * sph
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def wls_objective(theta):
        sigma2, phi, tau2 = theta
        if sigma2 <= 1e-6 or phi <= 1e-6 or tau2 < 0.0:
            return 1e12  # Strict positivity penalty
            
        wls_val = 0.0
        for h_k, g_hat, n_k in zip(lags, gamma_hat, pair_counts):
            g_theo = theoretical_gamma(h_k, theta)
            g_theo = max(g_theo, 1e-10)
            rel_diff = (g_hat / g_theo) - 1.0
            wls_val += n_k * (rel_diff * rel_diff)
        return wls_val

    # Pure Python Nelder-Mead simplex optimization
    dim = len(initial_theta)
    sim = [list(initial_theta)]
    for i in range(dim):
        p = list(initial_theta)
        p[i] = p[i] * 1.1 if p[i] != 0.0 else 0.05
        sim.append(p)
        
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(500):
        sim.sort(key=wls_objective)
        if abs(wls_objective(sim[-1]) - wls_objective(sim[0])) < 1e-7:
            break
            
        x_o = [sum(sim[i][d] for i in range(dim)) / dim for d in range(dim)]
        x_r = [x_o[d] + alpha * (x_o[d] - sim[-1][d]) for d in range(dim)]
        f_r = wls_objective(x_r)
        
        if wls_objective(sim[0]) <= f_r < wls_objective(sim[-2]):
            sim[-1] = x_r
            continue
        if f_r < wls_objective(sim[0]):
            x_e = [x_o[d] + gamma * (x_r[d] - x_o[d]) for d in range(dim)]
            sim[-1] = x_e if wls_objective(x_e) < f_r else x_r
            continue
        x_c = [x_o[d] + rho * (sim[-1][d] - x_o[d]) for d in range(dim)]
        if wls_objective(x_c) < wls_objective(sim[-1]):
            sim[-1] = x_c
            continue
        x_best = sim[0]
        for i in range(1, len(sim)):
            sim[i] = [x_best[d] + sigma * (sim[i][d] - x_best[d]) for d in range(dim)]
            
    return tuple(sim[0])
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

1. **Why Cressie WLS Is Ideal for Spatial Prediction:**
   The weight assigned to bin $k$ is $w_k = \frac{|N(\mathbf{h}_k)|}{\gamma(\mathbf{h}_k)^2}$.
   At small separation distances $h$, $\gamma(h)$ is small and pair count $|N(h)|$ is typically large. Consequently, $w_k$ is enormous at small lags and decays rapidly as $h$ increases.
   
   This aligns with kriging theory: **spatial interpolation weights are governed almost exclusively by the slope of the variogram near the origin**. Whether the variogram fits well at lag $h = 50\text{ km}$ near the sill has almost zero impact on kriging predictions at unobserved sites. WLS forces the numerical optimizer to match the small-lag behavior with extreme fidelity.
2. **Diagnostic Edge Case (Zero Nugget Instability):**
   If $\tau^2 \approx 0$ and the first lag $h_1 \approx 0$, $\gamma(h_1) \to 0$, causing the denominator in $Q_{\text{WLS}}$ to vanish. Numerical solvers must enforce a lower bound on the nugget or use soft clamping to prevent overflow.

---

### 4.5 Multivariate Admissibility and Cramér's Theorem

#### Beat 1: The Mathematical Anchor (The "Why")

When multiple spatial processes $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_m(\mathbf{s}))'$ are modeled concurrently, an algebraic dilemma arises: fitting valid univariate covariance functions to each cross-covariance $C_{ij}(\mathbf{h})$ independently does **not** ensure that the resulting joint spatial covariance matrix is positive semi-definite.

##### Definition 4.5.1 (Matrix-Valued Positive Semi-Definiteness)
A matrix-valued covariance function $\mathbf{C}: \mathbb{R}^d \to \mathbb{R}^{m \times m}$ is *positive semi-definite* if for any finite integer $n \in \mathbb{N}$, locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$, and vectors $\mathbf{a}_1, \dots, \mathbf{a}_n \in \mathbb{R}^m$:
$$
\sum_{r=1}^n \sum_{l=1}^n \mathbf{a}_r' \mathbf{C}(\mathbf{s}_r - \mathbf{s}_l) \mathbf{a}_l \ge 0
$$

##### Theorem 4.5.1 (Cramér's Theorem for Matrix-Valued Covariance Functions)
A continuous matrix-valued function $\mathbf{C}: \mathbb{R}^d \to \mathbb{R}^{m \times m}$ is positive semi-definite if and only if it admits the spectral representation:
$$
\mathbf{C}(\mathbf{h}) = \int_{\mathbb{R}^d} \exp\left(i \boldsymbol{\omega}'\mathbf{h}\right) d\mathbf{F}(\boldsymbol{\omega})
$$
where $\mathbf{F}(\cdot)$ is an $m \times m$ matrix of complex-valued Borel measures such that for any Borel set $E \in \mathcal{B}(\mathbb{R}^d)$, the matrix $\mathbf{F}(E)$ is Hermitian positive semi-definite:
$$
\mathbf{F}(E) = \mathbf{F}(E)^*, \quad \mathbf{v}^* \mathbf{F}(E) \mathbf{v} \ge 0, \quad \forall \mathbf{v} \in \mathbb{C}^m
$$
and has finite total trace variation $\operatorname{Tr}(\mathbf{F}(\mathbb{R}^d)) < \infty$.

*Proof.*
*Necessity ($\implies$):*
Assume $\mathbf{C}(\mathbf{h})$ is positive semi-definite. For any fixed $\mathbf{v} \in \mathbb{C}^m$, define the scalar function $g_{\mathbf{v}}(\mathbf{h}) = \mathbf{v}^* \mathbf{C}(\mathbf{h}) \mathbf{v}$. For any points $\mathbf{s}_1, \dots, \mathbf{s}_n$ and complex scalars $\alpha_1, \dots, \alpha_n$:
$$
\sum_{r=1}^n \sum_{l=1}^n \overline{\alpha_r} \alpha_l g_{\mathbf{v}}(\mathbf{s}_r - \mathbf{s}_l) = \sum_{r=1}^n \sum_{l=1}^n (\alpha_r \mathbf{v})^* \mathbf{C}(\mathbf{s}_r - \mathbf{s}_l) (\alpha_l \mathbf{v}) \ge 0
$$
by matrix positive semi-definiteness. By Bochner's Theorem, there exists a unique positive measure $F_{\mathbf{v}}$ such that $g_{\mathbf{v}}(\mathbf{h}) = \int_{\mathbb{R}^d} e^{i \boldsymbol{\omega}'\mathbf{h}} dF_{\mathbf{v}}(\boldsymbol{\omega})$. Polarization identities establish the existence of the matrix measure $\mathbf{F}$ satisfying $\mathbf{v}^* \mathbf{F}(E) \mathbf{v} \ge 0$.

*Sufficiency ($\impliedby$):*
Assume $\mathbf{F}(E) \succeq 0$. For any $\mathbf{a}_1, \dots, \mathbf{a}_n \in \mathbb{R}^m$:
$$
\sum_{r=1}^n \sum_{l=1}^n \mathbf{a}_r' \mathbf{C}(\mathbf{s}_r - \mathbf{s}_l) \mathbf{a}_l = \int_{\mathbb{R}^d} \left(\sum_{r=1}^n e^{i\boldsymbol{\omega}'\mathbf{s}_r}\mathbf{a}_r\right)^* d\mathbf{F}(\boldsymbol{\omega}) \left(\sum_{l=1}^n e^{i\boldsymbol{\omega}'\mathbf{s}_l}\mathbf{a}_l\right) \ge 0
$$
since the integrand is non-negative everywhere. $\blacksquare$

##### Corollary 4.5.1 (Spectral Density Coherence Condition)
When $d\mathbf{F}(\boldsymbol{\omega}) = \mathbf{f}(\boldsymbol{\omega}) d\boldsymbol{\omega}$, validity requires that the spatial spectral density matrix $\mathbf{f}(\boldsymbol{\omega}) \in \mathbb{C}^{m \times m}$ be Hermitian positive semi-definite for almost all $\boldsymbol{\omega} \in \mathbb{R}^d$. In particular, the cross-spectral coherence must satisfy:
$$
|f_{ij}(\boldsymbol{\omega})|^2 \le f_{ii}(\boldsymbol{\omega}) f_{jj}(\boldsymbol{\omega}), \quad \forall \boldsymbol{\omega} \in \mathbb{R}^d
$$

---

#### Beat 2: Algorithmic Translation (The "How")

Checking positive semi-definiteness for candidate bivariate cross-covariances involves evaluating the spectral determinant across a frequency grid $\boldsymbol{\omega}$:

```python
import math

def verify_bivariate_spectral_admissibility(f11_func, f22_func, f12_func, omega_grid):
    """
    Verifies Cramér's positive semi-definiteness condition across frequencies
    matching Corollary 4.5.1 in pure standard Python.
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

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

1. **The Independent Modeling Trap:**
   A practitioner fits an Exponential model to $Z_1$ ($a_1 = 10$), an Exponential model to $Z_2$ ($a_2 = 2$), and an Exponential model to the cross-continuity ($a_{12} = 5$) with high correlation $\rho_{12} = 0.9$.
   
   Although all three individual curves are mathematically valid univariate models, the cross-spectral coherence $|f_{12}(\omega)|^2$ exceeds $f_{11}(\omega)f_{22}(\omega)$ at intermediate frequencies (see Exercise 4.3). When cokriging is executed, the resulting cokriging matrix possesses negative eigenvalues, leading to negative estimation variances ($\sigma_{\text{CK}}^2 < 0$) and algorithm crashes.

---

### 4.6 The Linear Model of Coregionalization (LMC) and Eigenvalue Projection

#### Beat 1: The Mathematical Anchor (The "Why")

To guarantee multivariate positive semi-definiteness without evaluating multidimensional Cramér integrals, geostatistics relies on the **Linear Model of Coregionalization (LMC)** (Georges Matheron, 1982).

##### Definition 4.6.1 (The Linear Model of Coregionalization)
Let $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_m(\mathbf{s}))'$ be an $m$-variate second-order stationary field. Under the LMC, $\mathbf{Z}(\mathbf{s})$ is represented as a linear combination of $U$ independent spatial factor processes operating at distinct spatial scales:
$$
\mathbf{Z}(\mathbf{s}) = \boldsymbol{\mu} + \sum_{u=1}^U \mathbf{A}_u \mathbf{Y}_u(\mathbf{s})
$$
where:
- $\mathbf{Y}_u(\mathbf{s}) = (Y_{u,1}(\mathbf{s}), \dots, Y_{u,p_u}(\mathbf{s}))'$ is a vector of $p_u$ standardized, mutually uncorrelated spatial factors at scale $u$.
- Each factor has unit variance and correlation function $\rho_u(\mathbf{h})$, where $\rho_u(\cdot)$ is an admissible univariate correlation kernel.
- $\mathbf{A}_u \in \mathbb{R}^{m \times p_u}$ ($p_u \le m$) is a matrix of factor loadings.

##### Theorem 4.6.1 (Coregionalization Matrix Representation)
The spatial covariance matrix function $\mathbf{C}(\mathbf{h}) \in \mathbb{R}^{m \times m}$ and semivariogram matrix function $\mathbf{\Gamma}(\mathbf{h}) \in \mathbb{R}^{m \times m}$ expand as:
$$
\mathbf{C}(\mathbf{h}) = \sum_{u=1}^U \mathbf{B}_u \rho_u(\mathbf{h}), \quad \mathbf{\Gamma}(\mathbf{h}) = \sum_{u=1}^U \mathbf{B}_u (1 - \rho_u(\mathbf{h}))
$$
where each $\mathbf{B}_u \in \mathbb{R}^{m \times m}$ is the *coregionalization matrix* at scale $u$, defined by:
$$
\mathbf{B}_u = \mathbf{A}_u \mathbf{A}_u'
$$
The model is admissible on $\mathbb{R}^d$ if and only if each coregionalization matrix is symmetric positive semi-definite:
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
To minimize this sum subject to $\mathbf{B} \succeq 0 \iff \mathbf{M} \succeq 0$, we must set off-diagonal entries $M_{ij} = 0$ for $i \ne j$, and choose diagonal entries $M_{jj} \ge 0$ to minimize $(\lambda_j - M_{jj})^2$. The unconstrained minimizer is $M_{jj} = \lambda_j$. Projecting onto the non-negative half-line gives $M_{jj}^* = \max(0, \lambda_j)$. Thus $\mathbf{B}^* = \mathbf{Q}\mathbf{\Lambda}^+\mathbf{Q}'$. $\blacksquare$

---

#### Beat 2: Algorithmic Translation (The "How")

```python
import math

def project_lmc_positive_semidefinite(candidate_B_matrices, min_eigenvalue=1e-8):
    """
    Projects candidate LMC coregionalization matrices onto the cone of 
    positive semi-definite matrices matching Theorem 4.6.2 using a pure Python
    cyclic Jacobi eigenvalue decomposition.
    
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
        # V columns are the orthonormal eigenvectors
        return eigenvalues, V

    projected_B = []
    for B in candidate_B_matrices:
        m = len(B)
        # Symmetrize
        B_sym = [[0.5 * (B[i][j] + B[j][i]) for j in range(m)] for i in range(m)]
        
        # Spectral decomposition: B = Q Lambda Q'
        eigs, Q = jacobi_eigh(B_sym)
        
        # Eigenvalue clipping: lambda_j^+ = max(min_eigenvalue, lambda_j)
        eigs_clipped = [max(min_eigenvalue, val) for val in eigs]
        
        # Reconstruct B* = Q Lambda^+ Q' = sum lambda_k q_k q_k'
        B_proj = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                B_proj[i][j] = sum(eigs_clipped[k] * Q[i][k] * Q[j][k] for k in range(m))
                
        # Symmetrize reconstruction
        B_proj = [[0.5 * (B_proj[i][j] + B_proj[j][i]) for j in range(m)] for i in range(m)]
        projected_B.append(B_proj)
        
    return projected_B
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

```
Projection onto the Positive Semi-Definite Cone:
  Eigenvalue Space (λ_1, λ_2):
         ^ λ_2
         |     Valid LMC Region:
         |     Positive Semi-Definite Cone S_+^2
         |       /
         |      /  * Projected Solution B*
         |     /   |
         |    /    |
  ───────+───/─────|────────> λ_1
         |  /      |
         | /       * Unconstrained Candidate B_tilde
         |/          (has negative eigenvalue λ_2 < 0)
```

1. **Geometric Cone Projection:**
   The set of symmetric positive semi-definite matrices $\mathcal{S}_+^m$ forms a convex cone in $\mathbb{R}^{m(m+1)/2}$. When a practitioner fits coregionalization parameters by unconstrained linear regression, the estimate $\widetilde{\mathbf{B}}_u$ often lands outside the cone due to sampling noise. Spectral clipping orthogonally projects the invalid estimate back onto the closest point on the boundary of $\mathcal{S}_+^m$.
2. **Nugget Coregionalization Matrix ($\mathbf{B}_0$ Strict Definiteness):**
   While spatial structures at long ranges ($\mathbf{B}_u$ for $u \ge 1$) can be rank-deficient (semi-definite with $\operatorname{rank}(\mathbf{B}_u) < m$), the nugget coregionalization matrix $\mathbf{B}_0$ **must be strictly positive definite** ($\mathbf{B}_0 \succ 0$). A strictly positive definite nugget guarantees that the total cokriging covariance matrix $\mathbf{V}_{\text{CK}}$ remains strictly invertible.

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
- **(c)** Compare the Mean Squared Errors ($\mathrm{MSE}$) of $\widehat{\sigma}^2_{\text{REML}}$ and $\widehat{\sigma}^2_{\text{ML}}$.

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
- **(b)** By Theorem 4.2.1, $\operatorname{E}[\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}] = \sigma^2 \operatorname{Tr}(\mathbf{P}_{\mathbf{R}}\mathbf{R}) = (n - p)\sigma^2$.
  Thus:
  $$
  \operatorname{E}[\widehat{\sigma}^2_{\text{REML}}] = \frac{1}{n - p}(n - p)\sigma^2 = \sigma^2
  $$
- **(c)** Note that $\frac{1}{\sigma^2}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z} \sim \chi_{n-p}^2$. Since $\operatorname{Var}(\chi_{n-p}^2) = 2(n-p)$, we have $\operatorname{Var}(\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}) = 2(n-p)\sigma^4$.
  For REML:
  $$
  \operatorname{Var}(\widehat{\sigma}^2_{\text{REML}}) = \frac{1}{(n-p)^2}(2(n-p)\sigma^4) = \frac{2\sigma^4}{n - p} \implies \mathrm{MSE}(\widehat{\sigma}^2_{\text{REML}}) = \frac{2\sigma^4}{n - p}
  $$
  For ML: $\widehat{\sigma}^2_{\text{ML}} = \frac{1}{n}\mathbf{Z}'\mathbf{P}_{\mathbf{R}}\mathbf{Z}$.
  $$
  \operatorname{Var}(\widehat{\sigma}^2_{\text{ML}}) = \frac{2(n-p)\sigma^4}{n^2}, \quad \operatorname{Bias}(\widehat{\sigma}^2_{\text{ML}}) = -\frac{p}{n}\sigma^2
  $$
  $$
  \mathrm{MSE}(\widehat{\sigma}^2_{\text{ML}}) = \frac{2(n-p)\sigma^4}{n^2} + \frac{p^2\sigma^4}{n^2} = \frac{2n - p}{n^2}\sigma^4
  $$
  Notice that $\frac{2n - p}{n^2} < \frac{2}{n - p}$. While ML achieves a slightly lower MSE due to variance contraction, it incurs systematic downward bias, distorting kriging prediction variance. $\blacksquare$

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

#### Exercise 4.4: Factor Loading and Cholesky Representation of an LMC Matrix
Consider the estimated $3 \times 3$ candidate coregionalization matrix:
$$
\mathbf{B} = \begin{bmatrix} 4 & 2 & -1 \\ 2 & 3 & 1 \\ -1 & 1 & 2 \end{bmatrix}
$$
- **(a)** Apply Sylvester's criterion to verify positive definiteness.
- **(b)** Calculate the Cholesky factor loading matrix $\mathbf{A} \in \mathbb{R}^{3 \times 3}$ such that $\mathbf{B} = \mathbf{A}\mathbf{A}'$.

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
  \mathbf{A} = \begin{bmatrix} 2 & 0 & 0 \\ 1 & \sqrt{2} & 0 \\ -0.5 & \frac{3\sqrt{2}}{4} & \frac{\sqrt{10}}{4} \end{bmatrix}
  $$
  Verification:
  $$
  \mathbf{A}\mathbf{A}' = \begin{bmatrix} 2 & 0 & 0 \\ 1 & \sqrt{2} & 0 \\ -0.5 & \frac{3\sqrt{2}}{4} & \frac{\sqrt{10}}{4} \end{bmatrix} \begin{bmatrix} 2 & 1 & -0.5 \\ 0 & \sqrt{2} & \frac{3\sqrt{2}}{4} \\ 0 & 0 & \frac{\sqrt{10}}{4} \end{bmatrix} = \begin{bmatrix} 4 & 2 & -1 \\ 2 & 3 & 1 \\ -1 & 1 & 2 \end{bmatrix} = \mathbf{B}
  $$
  confirming the exact factor representation. $\blacksquare$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 4: Spatial Likelihood and Parametric Estimation — Companion Exercises and Complete Solutions](ch04_exercises.md)
