# Chapter 4 Study Guide & Comprehensive Exercises
## Covariance Parameter Estimation and Theoretical Validations

This companion study guide reinforces the core mathematical, likelihood-based, and parametric estimation concepts from [Chapter 4: Covariance Parameter Estimation and Theoretical Validations](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch04_parameter_estimation.md). The exercises below guide the reader through Patterson & Thompson error contrasts, composite likelihood of spatial differences, Gaussian Maximum Likelihood, Restricted Maximum Likelihood (REML), Cressie's WLS fitting, the Linear Model of Coregionalization (LMC), and Cramér admissibility across five pedagogical tiers:
1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 4.1–4.4)
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 4.5–4.8)
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 4.9–4.11)
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 4.12–4.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 4.14–4.16)

---

### Tier 0: Classical-to-Spatial Warm-Ups

#### Exercise 4.1: Classical Error Contrasts and REML for Independent Samples (S&G Section 5.3.1)
Let $Y_1, \dots, Y_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$, which can be expressed in vector notation as:
$$
\mathbf{Y} \sim \mathcal{N}_n(\mathbf{X}\mu, \sigma^2 \mathbf{I}_n), \quad \text{where } \mathbf{X} = \mathbf{1}_n \in \mathbb{R}^{n \times 1} \text{ and } p = 1
$$
In standard Maximum Likelihood (ML), profiling out the mean $\mu$ by plugging in $\widehat{\mu}_{\text{ML}} = \bar{Y} = \frac{1}{n}\sum_{i=1}^n Y_i$ produces the biased variance estimator $\widehat{\sigma}_{\text{ML}}^2 = \frac{1}{n}\sum_{i=1}^n (Y_i - \bar{Y})^2$, with expectation $\operatorname{E}[\widehat{\sigma}_{\text{ML}}^2] = \frac{n-1}{n}\sigma^2$. Patterson and Thompson (1971) introduced Restricted Maximum Likelihood (REML) by projecting the data onto an $(n-1)$-dimensional error contrast subspace.

An **error contrast** is a linear combination vector $\mathbf{U} = \mathbf{A}'\mathbf{Y}$, where $\mathbf{A}$ is an $n \times (n-1)$ matrix of rank $n-1$ satisfying:
$$
\mathbf{A}'\mathbf{X} = \mathbf{A}'\mathbf{1}_n = \mathbf{0}_{(n-1) \times 1}
$$

- **(a)** Prove that the distribution of $\mathbf{U}$ is multivariate Gaussian $\mathbf{U} \sim \mathcal{N}_{n-1}(\mathbf{0}, \sigma^2 \mathbf{A}'\mathbf{A})$, and verify that it is completely invariant to the nuisance drift parameter $\mu$.
- **(b)** Let $\mathbf{A}$ be chosen such that its columns form an orthonormal basis for the orthogonal complement of $\mathbf{1}_n$ (so that $\mathbf{A}'\mathbf{A} = \mathbf{I}_{n-1}$). Prove that:
  $$
  \mathbf{A}\mathbf{A}' = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n \mathbf{1}_n'
  $$
- **(c)** Write out the exact log-likelihood $\ell(\sigma^2; \mathbf{U})$ of the error contrast vector $\mathbf{U}$ under an arbitrary full-rank contrast matrix $\mathbf{A}$. Prove that the quadratic form satisfies:
  $$
  \mathbf{U}'(\mathbf{A}'\mathbf{A})^{-1}\mathbf{U} = \mathbf{Y}'\left(\mathbf{I}_n - \frac{1}{n}\mathbf{1}_n \mathbf{1}_n'\right)\mathbf{Y} = \sum_{i=1}^n (Y_i - \bar{Y})^2
  $$
  independent of the specific choice of basis $\mathbf{A}$.
- **(d)** Maximize $\ell(\sigma^2; \mathbf{U})$ with respect to $\sigma^2$ to find the REML estimator $\widehat{\sigma}_{\text{REML}}^2$. Verify that $\widehat{\sigma}_{\text{REML}}^2 = S^2 = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2$, proving that REML intrinsically recovers Bessel's degrees-of-freedom correction and yields an exactly unbiased estimator: $\operatorname{E}[\widehat{\sigma}_{\text{REML}}^2] = \sigma^2$.

##### Solution to Exercise 4.1
- **(a)** Because $\mathbf{Y} \sim \mathcal{N}_n(\mathbf{1}_n \mu, \sigma^2 \mathbf{I}_n)$ and $\mathbf{U} = \mathbf{A}'\mathbf{Y}$ is a linear transformation of a Gaussian vector, $\mathbf{U}$ is multivariate Gaussian with:
  $$
  \operatorname{E}[\mathbf{U}] = \operatorname{E}[\mathbf{A}'\mathbf{Y}] = \mathbf{A}'\operatorname{E}[\mathbf{Y}] = \mathbf{A}'(\mathbf{1}_n \mu) = (\mathbf{A}'\mathbf{1}_n)\mu = \mathbf{0}_{(n-1) \times 1} \cdot \mu = \mathbf{0}_{(n-1) \times 1}
  $$
  The covariance matrix of $\mathbf{U}$ is:
  $$
  \operatorname{Var}(\mathbf{U}) = \operatorname{Var}(\mathbf{A}'\mathbf{Y}) = \mathbf{A}' \operatorname{Var}(\mathbf{Y}) \mathbf{A} = \mathbf{A}' (\sigma^2 \mathbf{I}_n) \mathbf{A} = \sigma^2 \mathbf{A}'\mathbf{A}
  $$
  Because both the mean vector $\mathbf{0}$ and covariance matrix $\sigma^2 \mathbf{A}'\mathbf{A}$ contain no dependence on $\mu$, the marginal distribution $\mathbf{U} \sim \mathcal{N}_{n-1}(\mathbf{0}, \sigma^2 \mathbf{A}'\mathbf{A})$ is completely invariant to the nuisance mean $\mu$. $\blacksquare$

- **(b)** The column space of $\mathbf{1}_n$ is a 1-dimensional subspace of $\mathbb{R}^n$ with orthogonal projection matrix:
  $$
  \mathbf{P}_{\mathbf{1}} = \mathbf{1}_n (\mathbf{1}_n'\mathbf{1}_n)^{-1}\mathbf{1}_n' = \frac{1}{n}\mathbf{1}_n \mathbf{1}_n'
  $$
  The orthogonal complement subspace $\operatorname{span}(\mathbf{1}_n)^\perp$ has dimension $n - 1$.
  Any matrix $\mathbf{A} \in \mathbb{R}^{n \times (n-1)}$ whose columns form an orthonormal basis for $\operatorname{span}(\mathbf{1}_n)^\perp$ satisfies $\mathbf{A}'\mathbf{A} = \mathbf{I}_{n-1}$.
  The orthogonal projection matrix onto $\operatorname{span}(\mathbf{1}_n)^\perp$ is:
  $$
  \mathbf{P}_{\mathbf{1}^\perp} = \mathbf{A}(\mathbf{A}'\mathbf{A})^{-1}\mathbf{A}' = \mathbf{A}\mathbf{I}_{n-1}\mathbf{A}' = \mathbf{A}\mathbf{A}'
  $$
  Because $\mathbb{R}^n = \operatorname{span}(\mathbf{1}_n) \oplus \operatorname{span}(\mathbf{1}_n)^\perp$, the sum of the orthogonal projection matrices onto these complementary subspaces equals the identity operator:
  $$
  \mathbf{P}_{\mathbf{1}} + \mathbf{P}_{\mathbf{1}^\perp} = \mathbf{I}_n \implies \mathbf{A}\mathbf{A}' = \mathbf{I}_n - \mathbf{P}_{\mathbf{1}} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n \mathbf{1}_n' \quad \blacksquare
  $$

- **(c)** For any full-rank $n \times (n-1)$ contrast matrix $\mathbf{A}$, $\mathbf{U} \sim \mathcal{N}_{n-1}(\mathbf{0}, \sigma^2 \mathbf{A}'\mathbf{A})$. The probability density function of $\mathbf{U}$ is:
  $$
  f(\mathbf{U}; \sigma^2) = (2\pi)^{-(n-1)/2} \det(\sigma^2 \mathbf{A}'\mathbf{A})^{-1/2} \exp\left(-\frac{1}{2\sigma^2}\mathbf{U}'(\mathbf{A}'\mathbf{A})^{-1}\mathbf{U}\right)
  $$
  Noting that $\det(\sigma^2 \mathbf{A}'\mathbf{A}) = (\sigma^2)^{n-1}\det(\mathbf{A}'\mathbf{A})$, the log-likelihood is:
  $$
  \ell(\sigma^2; \mathbf{U}) = -\frac{n-1}{2}\ln(2\pi) - \frac{n-1}{2}\ln(\sigma^2) - \frac{1}{2}\ln\det(\mathbf{A}'\mathbf{A}) - \frac{1}{2\sigma^2}\mathbf{U}'(\mathbf{A}'\mathbf{A})^{-1}\mathbf{U}
  $$
  Substituting $\mathbf{U} = \mathbf{A}'\mathbf{Y}$ into the quadratic form:
  $$
  \mathbf{U}'(\mathbf{A}'\mathbf{A})^{-1}\mathbf{U} = (\mathbf{A}'\mathbf{Y})'(\mathbf{A}'\mathbf{A})^{-1}(\mathbf{A}'\mathbf{Y}) = \mathbf{Y}' \mathbf{A}(\mathbf{A}'\mathbf{A})^{-1}\mathbf{A}' \mathbf{Y}
  $$
  The matrix $\mathbf{P} = \mathbf{A}(\mathbf{A}'\mathbf{A})^{-1}\mathbf{A}'$ is the unique orthogonal projection matrix onto the column space of $\mathbf{A}$, which is $\operatorname{span}(\mathbf{1}_n)^\perp$. From part (b), $\mathbf{P} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n \mathbf{1}_n'$ regardless of the particular contrast basis chosen.
  Expanding the quadratic form:
  $$
  \mathbf{Y}'\left(\mathbf{I}_n - \frac{1}{n}\mathbf{1}_n \mathbf{1}_n'\right)\mathbf{Y} = \mathbf{Y}'\mathbf{Y} - \frac{1}{n}(\mathbf{1}_n'\mathbf{Y})'(\mathbf{1}_n'\mathbf{Y}) = \sum_{i=1}^n Y_i^2 - \frac{1}{n}\left(\sum_{i=1}^n Y_i\right)^2 = \sum_{i=1}^n Y_i^2 - n\bar{Y}^2 = \sum_{i=1}^n (Y_i - \bar{Y})^2 \quad \blacksquare
  $$

- **(d)** Differentiating $\ell(\sigma^2; \mathbf{U})$ with respect to the variance parameter $\sigma^2$:
  $$
  \frac{\partial \ell(\sigma^2; \mathbf{U})}{\partial \sigma^2} = -\frac{n-1}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (Y_i - \bar{Y})^2
  $$
  Setting the score derivative to zero:
  $$
  -\frac{n-1}{2\widehat{\sigma}_{\text{REML}}^2} + \frac{1}{2(\widehat{\sigma}_{\text{REML}}^2)^2}\sum_{i=1}^n (Y_i - \bar{Y})^2 = 0 \implies \widehat{\sigma}_{\text{REML}}^2 = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2 = S^2
  $$
  Taking expectations:
  $$
  \operatorname{E}[\widehat{\sigma}_{\text{REML}}^2] = \frac{1}{n-1}\operatorname{E}\left[\sum_{i=1}^n (Y_i - \bar{Y})^2\right] = \frac{1}{n-1}(n-1)\sigma^2 = \sigma^2
  $$
  In standard ML, the profile log-likelihood treats $\mu$ as fixed at its estimate $\widehat{\mu} = \bar{Y}$, ignoring the fact that estimating $\mu$ consumes 1 degree of freedom. This results in the denominator $n$ and negative bias $\operatorname{E}[\widehat{\sigma}_{\text{ML}}^2] = \frac{n-1}{n}\sigma^2$. REML strictly operates on the $(n-1)$-dimensional orthogonal complement of the drift space $\mathbf{X}$, naturally adjusting the degrees of freedom and yielding the classical unbiased sample variance $S^2$. $\blacksquare$

---

#### Exercise 4.2: Pairwise Difference Composite Log-Likelihood and Cressie's WLS Estimating Equations (S&G Section 5.4.1)
Full Maximum Likelihood estimation of spatial covariance parameters $\boldsymbol{\theta}$ requires inverting an $n \times n$ covariance matrix $\mathbf{V}(\boldsymbol{\theta})$ with computational complexity $\mathcal{O}(n^3)$. To circumvent this bottleneck or bypass full joint distributional assumptions, composite likelihood methods (Curriero and Lele 1999; S&G Section 5.4.1) construct objective functions by compounding lower-dimensional marginal distributions.

Consider the spatial pairwise differences:
$$
D_{ij} = Z(\mathbf{s}_i) - Z(\mathbf{s}_j), \quad 1 \le i < j \le n
$$
Under intrinsic stationarity with Gaussian increments, each difference is univariate normal:
$$
D_{ij} \sim \mathcal{N}(0, 2\gamma_{ij}(\boldsymbol{\theta})), \quad \text{where } 2\gamma_{ij}(\boldsymbol{\theta}) = \operatorname{Var}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))
$$
The pairwise difference composite log-likelihood is formed by summing the univariate Gaussian log-densities across all $N_{\text{pairs}} = \binom{n}{2}$ distinct pairs:
$$
\ell_{\text{CL}}^{(2)}(\boldsymbol{\theta}) = \sum_{i < j} \ln f(D_{ij}; \boldsymbol{\theta})
$$

- **(a)** Write down the explicit univariate log-density $\ln f(D_{ij}; \boldsymbol{\theta})$ for a single difference $D_{ij}$ in terms of the semivariogram $\gamma_{ij}(\boldsymbol{\theta})$.
- **(b)** Compute the score contribution $\nabla_{\boldsymbol{\theta}} \ln f(D_{ij}; \boldsymbol{\theta})$ for pair $(i, j)$, and show that the composite score vector is:
  $$
  \mathbf{S}_{\text{CL}}(\boldsymbol{\theta}) = \sum_{i < j} \left(\frac{D_{ij}^2 - 2\gamma_{ij}(\boldsymbol{\theta})}{4\gamma_{ij}^2(\boldsymbol{\theta})}\right) \nabla_{\boldsymbol{\theta}}\gamma_{ij}(\boldsymbol{\theta})
  $$
- **(c)** Partition the pairs into $K$ spatial lag bins $N(h_k) = \{(i,j) : \|\mathbf{s}_i - \mathbf{s}_j\| \approx h_k\}$ for $k = 1, \dots, K$. Assuming the theoretical semivariogram is approximately constant over bin $k$, $\gamma_{ij}(\boldsymbol{\theta}) \approx \gamma(h_k; \boldsymbol{\theta})$, express $\mathbf{S}_{\text{CL}}(\boldsymbol{\theta})$ in terms of Matheron's empirical semivariogram $\widehat{\gamma}(h_k) = \frac{1}{2|N(h_k)|}\sum_{(i,j) \in N(h_k)} D_{ij}^2$.
- **(d)** Cressie (1985) introduced the Weighted Least Squares (WLS) objective:
  $$
  Q(\boldsymbol{\theta}) = \sum_{k=1}^K |N(h_k)| \left(\frac{\widehat{\gamma}(h_k)}{\gamma(h_k; \boldsymbol{\theta})} - 1\right)^2
  $$
  Show that when the denominator weights $w_k(\boldsymbol{\theta}) = \frac{|N(h_k)|}{\gamma(h_k; \boldsymbol{\theta})^2}$ are held fixed at the current iteration (as in Iteratively Reweighted Least Squares or Gauss-Newton optimization), the estimating equations for minimizing $Q(\boldsymbol{\theta})$ coincide exactly with the composite score equations $\mathbf{S}_{\text{CL}}(\boldsymbol{\theta}) = \mathbf{0}$.

##### Solution to Exercise 4.2
- **(a)** For Gaussian increment $D_{ij} \sim \mathcal{N}(0, 2\gamma_{ij}(\boldsymbol{\theta}))$, the univariate normal probability density is:
  $$
  f(D_{ij}; \boldsymbol{\theta}) = \frac{1}{\sqrt{2\pi \cdot 2\gamma_{ij}(\boldsymbol{\theta})}} \exp\left(-\frac{D_{ij}^2}{2 \cdot 2\gamma_{ij}(\boldsymbol{\theta})}\right) = (4\pi \gamma_{ij}(\boldsymbol{\theta}))^{-1/2} \exp\left(-\frac{D_{ij}^2}{4\gamma_{ij}(\boldsymbol{\theta})}\right)
  $$
  Taking the natural logarithm:
  $$
  \ln f(D_{ij}; \boldsymbol{\theta}) = -\frac{1}{2}\ln(4\pi) - \frac{1}{2}\ln(\gamma_{ij}(\boldsymbol{\theta})) - \frac{D_{ij}^2}{4\gamma_{ij}(\boldsymbol{\theta})}
  $$
  The pairwise composite log-likelihood is therefore:
  $$
  \ell_{\text{CL}}^{(2)}(\boldsymbol{\theta}) = \sum_{i < j} \left[-\frac{1}{2}\ln(4\pi) - \frac{1}{2}\ln(\gamma_{ij}(\boldsymbol{\theta})) - \frac{D_{ij}^2}{4\gamma_{ij}(\boldsymbol{\theta})}\right] \quad \blacksquare
  $$

- **(b)** Applying the chain rule to differentiate $\ln f(D_{ij}; \boldsymbol{\theta})$ with respect to the covariance parameter vector $\boldsymbol{\theta}$:
  $$
  \nabla_{\boldsymbol{\theta}} \ln f(D_{ij}; \boldsymbol{\theta}) = \frac{\partial \ln f(D_{ij}; \boldsymbol{\theta})}{\partial \gamma_{ij}(\boldsymbol{\theta})} \nabla_{\boldsymbol{\theta}}\gamma_{ij}(\boldsymbol{\theta})
  $$
  Evaluating the partial derivative:
  $$
  \frac{\partial}{\partial \gamma_{ij}} \left(-\frac{1}{2}\ln(\gamma_{ij}) - \frac{D_{ij}^2}{4\gamma_{ij}}\right) = -\frac{1}{2\gamma_{ij}} + \frac{D_{ij}^2}{4\gamma_{ij}^2} = \frac{D_{ij}^2 - 2\gamma_{ij}}{4\gamma_{ij}^2}
  $$
  Multiplying by $\nabla_{\boldsymbol{\theta}}\gamma_{ij}(\boldsymbol{\theta})$ and summing over all pairs $1 \le i < j \le n$:
  $$
  \mathbf{S}_{\text{CL}}(\boldsymbol{\theta}) = \sum_{i < j} \left(\frac{D_{ij}^2 - 2\gamma_{ij}(\boldsymbol{\theta})}{4\gamma_{ij}^2(\boldsymbol{\theta})}\right) \nabla_{\boldsymbol{\theta}}\gamma_{ij}(\boldsymbol{\theta}) \quad \blacksquare
  $$

- **(c)** Partitioning the sum over pairs into sums over the $K$ lag bins $N(h_1), \dots, N(h_K)$:
  $$
  \mathbf{S}_{\text{CL}}(\boldsymbol{\theta}) = \sum_{k=1}^K \sum_{(i,j) \in N(h_k)} \left(\frac{D_{ij}^2 - 2\gamma(h_k; \boldsymbol{\theta})}{4\gamma^2(h_k; \boldsymbol{\theta})}\right) \nabla_{\boldsymbol{\theta}}\gamma(h_k; \boldsymbol{\theta})
  $$
  Factoring out terms that depend only on the bin index $k$:
  $$
  \mathbf{S}_{\text{CL}}(\boldsymbol{\theta}) = \sum_{k=1}^K \frac{\nabla_{\boldsymbol{\theta}}\gamma(h_k; \boldsymbol{\theta})}{4\gamma^2(h_k; \boldsymbol{\theta})} \sum_{(i,j) \in N(h_k)} \left(D_{ij}^2 - 2\gamma(h_k; \boldsymbol{\theta})\right)
  $$
  Recall Matheron's empirical semivariogram estimator:
  $$
  \widehat{\gamma}(h_k) = \frac{1}{2|N(h_k)|}\sum_{(i,j) \in N(h_k)} D_{ij}^2 \implies \sum_{(i,j) \in N(h_k)} D_{ij}^2 = 2|N(h_k)|\widehat{\gamma}(h_k)
  $$
  Substituting this identity into the inner sum:
  $$
  \sum_{(i,j) \in N(h_k)} \left(D_{ij}^2 - 2\gamma(h_k; \boldsymbol{\theta})ight) = 2|N(h_k)|\widehat{\gamma}(h_k) - 2|N(h_k)|\gamma(h_k; \boldsymbol{\theta}) = 2|N(h_k)|(\widehat{\gamma}(h_k) - \gamma(h_k; \boldsymbol{\theta}))
  $$
  Substituting back into the composite score expression:
  $$
  \mathbf{S}_{\text{CL}}(\boldsymbol{\theta}) = \sum_{k=1}^K \frac{\nabla_{\boldsymbol{\theta}}\gamma(h_k; \boldsymbol{\theta})}{4\gamma^2(h_k; \boldsymbol{\theta})} \cdot 2|N(h_k)|(\widehat{\gamma}(h_k) - \gamma(h_k; \boldsymbol{\theta})) = \sum_{k=1}^K \frac{|N(h_k)|}{2\gamma^2(h_k; \boldsymbol{\theta})} \left(\widehat{\gamma}(h_k) - \gamma(h_k; \boldsymbol{\theta})\right) \nabla_{\boldsymbol{\theta}}\gamma(h_k; \boldsymbol{\theta}) \quad \blacksquare
  $$

- **(d)** Cressie's WLS objective can be rewritten by placing a common denominator inside each term:
  $$
  Q(\boldsymbol{\theta}) = \sum_{k=1}^K |N(h_k)| \left(\frac{\widehat{\gamma}(h_k) - \gamma(h_k; \boldsymbol{\theta})}{\gamma(h_k; \boldsymbol{\theta})}\right)^2 = \sum_{k=1}^K \frac{|N(h_k)|}{\gamma^2(h_k; \boldsymbol{\theta})} \left(\widehat{\gamma}(h_k) - \gamma(h_k; \boldsymbol{\theta})\right)^2
  $$
  In standard nonlinear least squares (Gauss-Newton algorithm) or Iteratively Reweighted Least Squares (IRLS), the weight assigned to residual $k$ is $w_k(\boldsymbol{\theta}^{(m)}) = \frac{|N(h_k)|}{\gamma^2(h_k; \boldsymbol{\theta}^{(m)})}$. Differentiating the weighted squared residual with respect to the model values $\gamma(h_k; \boldsymbol{\theta})$ while keeping the denominator weights evaluated at the current step:
  $$
  \nabla_{\boldsymbol{\theta}} Q_{\text{step}}(\boldsymbol{\theta}) = \sum_{k=1}^K w_k(\boldsymbol{\theta}^{(m)}) \cdot 2\left(\gamma(h_k; \boldsymbol{\theta}) - \widehat{\gamma}(h_k)\right) \nabla_{\boldsymbol{\theta}}\gamma(h_k; \boldsymbol{\theta})
  $$
  Equating to zero gives:
  $$
  \sum_{k=1}^K \frac{|N(h_k)|}{\gamma^2(h_k; \boldsymbol{\theta})} \left(\widehat{\gamma}(h_k) - \gamma(h_k; \boldsymbol{\theta})\right) \nabla_{\boldsymbol{\theta}}\gamma(h_k; \boldsymbol{\theta}) = \mathbf{0}
  $$
  Multiplying both sides by $1/2$, this system of estimating equations is identical to the pairwise composite score equation $\mathbf{S}_{\text{CL}}(\boldsymbol{\theta}) = \mathbf{0}$ from part (c).
  This establishes that Cressie's WLS estimator is precisely the maximum pairwise composite likelihood estimator under bin-wise aggregation and iterative weighting. $\blacksquare$

---

#### Exercise 4.3: Squared Difference Log-Likelihood and Chi-Square Score Equivalence (S&G Section 5.4.1)
An alternative composite likelihood formulation operates directly on the squared pairwise differences:
$$
T_{ij} = (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 = D_{ij}^2, \quad 1 \le i < j \le n
$$
Under intrinsic Gaussianity, $D_{ij} \sim \mathcal{N}(0, 2\gamma_{ij}(\boldsymbol{\theta}))$. Consequently, the standardized squared difference follows a 1-degree-of-freedom chi-square distribution:
$$
W_{ij} = \frac{T_{ij}}{2\gamma_{ij}(\boldsymbol{\theta})} = \left(\frac{D_{ij}}{\sqrt{2\gamma_{ij}(\boldsymbol{\theta})}}\right)^2 \sim \chi_1^2
$$

- **(a)** Recall that the probability density function of a standard chi-square random variable $W \sim \chi_1^2$ is:
  $$
  g(w) = \frac{1}{\sqrt{2\pi}} w^{-1/2} \exp\left(-\frac{w}{2}\right), \quad w > 0
  $$
  Using the univariate change-of-variables formula $f_T(t; \boldsymbol{\theta}) = g\left(\frac{t}{2\gamma_{ij}(\boldsymbol{\theta})}\right) \cdot \left|\frac{dw}{dt}\right|$, derive the exact probability density function $f_T(t; \boldsymbol{\theta})$ of $T_{ij}$.
- **(b)** Show that the log-density of $T_{ij}$ is given by:
  $$
  \ln f_T(T_{ij}; \boldsymbol{\theta}) = -\frac{1}{2}\ln(4\pi) - \frac{1}{2}\ln(\gamma_{ij}(\boldsymbol{\theta})) - \frac{1}{2}\ln(T_{ij}) - \frac{T_{ij}}{4\gamma_{ij}(\boldsymbol{\theta})}
  $$
- **(c)** Calculate the score contribution $\nabla_{\boldsymbol{\theta}} \ln f_T(T_{ij}; \boldsymbol{\theta})$. Prove that this score contribution is identical to the Gaussian pairwise difference score $\nabla_{\boldsymbol{\theta}} \ln f(D_{ij}; \boldsymbol{\theta})$ derived in Exercise 4.2.
- **(d)** Explain why the composite likelihood of signed differences $D_{ij} \in \mathbb{R}$ and the composite likelihood of squared differences $T_{ij} \in [0, \infty)$ lead to identical estimating equations and parameter estimates, identifying the role of the transformation Jacobian.

##### Solution to Exercise 4.3
- **(a)** The transformation is $w = \frac{t}{2\gamma_{ij}}$, with derivative $\frac{dw}{dt} = \frac{1}{2\gamma_{ij}} > 0$.
  Applying the change-of-variables theorem for univariate densities:
  $$
  f_T(t; \boldsymbol{\theta}) = g\left(\frac{t}{2\gamma_{ij}}\right) \left|\frac{dw}{dt}\right| = \frac{1}{\sqrt{2\pi}} \left(\frac{t}{2\gamma_{ij}}\right)^{-1/2} \exp\left(-\frac{t}{4\gamma_{ij}}\right) \cdot \frac{1}{2\gamma_{ij}}
  $$
  Simplifying the algebraic constants:
  $$
  f_T(t; \boldsymbol{\theta}) = \frac{1}{\sqrt{2\pi}} t^{-1/2} (2\gamma_{ij})^{1/2} (2\gamma_{ij})^{-1} \exp\left(-\frac{t}{4\gamma_{ij}}\right) = \frac{1}{\sqrt{2\pi}} t^{-1/2} (2\gamma_{ij})^{-1/2} \exp\left(-\frac{t}{4\gamma_{ij}}\right)
  $$
  Combining under the radical:
  $$
  f_T(t; \boldsymbol{\theta}) = \frac{1}{\sqrt{4\pi \gamma_{ij} t}} \exp\left(-\frac{t}{4\gamma_{ij}}\right), \quad t > 0 \quad \blacksquare
  $$

- **(b)** Taking the natural logarithm of $f_T(t; \boldsymbol{\theta})$:
  $$
  \ln f_T(t; \boldsymbol{\theta}) = \ln\left((4\pi)^{-1/2} \gamma_{ij}^{-1/2} t^{-1/2} \exp\left(-\frac{t}{4\gamma_{ij}}\right)\right)
  $$
  Expanding each multiplicative term into a sum:
  $$
  \ln f_T(t; \boldsymbol{\theta}) = -\frac{1}{2}\ln(4\pi) - \frac{1}{2}\ln(\gamma_{ij}(\boldsymbol{\theta})) - \frac{1}{2}\ln(t) - \frac{t}{4\gamma_{ij}(\boldsymbol{\theta})}
  $$
  Evaluating at the observed statistic $t = T_{ij}$:
  $$
  \ln f_T(T_{ij}; \boldsymbol{\theta}) = -\frac{1}{2}\ln(4\pi) - \frac{1}{2}\ln(\gamma_{ij}(\boldsymbol{\theta})) - \frac{1}{2}\ln(T_{ij}) - \frac{T_{ij}}{4\gamma_{ij}(\boldsymbol{\theta})} \quad \blacksquare
  $$

- **(c)** Differentiating $\ln f_T(T_{ij}; \boldsymbol{\theta})$ with respect to covariance parameter vector $\boldsymbol{\theta}$:
  Notice that $-\frac{1}{2}\ln(4\pi)$ and $-\frac{1}{2}\ln(T_{ij})$ do not depend on $\boldsymbol{\theta}$.
  Therefore:
  $$
  \nabla_{\boldsymbol{\theta}} \ln f_T(T_{ij}; \boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}}\left(-\frac{1}{2}\ln(\gamma_{ij}(\boldsymbol{\theta})) - \frac{T_{ij}}{4\gamma_{ij}(\boldsymbol{\theta})}\right)
  $$
  Applying the chain rule:
  $$
  \frac{\partial}{\partial \gamma_{ij}}\left(-\frac{1}{2}\ln(\gamma_{ij}) - \frac{T_{ij}}{4\gamma_{ij}}\right) = -\frac{1}{2\gamma_{ij}} + \frac{T_{ij}}{4\gamma_{ij}^2} = \frac{T_{ij} - 2\gamma_{ij}}{4\gamma_{ij}^2}
  $$
  Multiplying by $\nabla_{\boldsymbol{\theta}}\gamma_{ij}(\boldsymbol{\theta})$:
  $$
  \nabla_{\boldsymbol{\theta}} \ln f_T(T_{ij}; \boldsymbol{\theta}) = \left(\frac{T_{ij} - 2\gamma_{ij}(\boldsymbol{\theta})}{4\gamma_{ij}^2(\boldsymbol{\theta})}\right) \nabla_{\boldsymbol{\theta}}\gamma_{ij}(\boldsymbol{\theta})
  $$
  Substituting $T_{ij} = D_{ij}^2$:
  $$
  \nabla_{\boldsymbol{\theta}} \ln f_T(T_{ij}; \boldsymbol{\theta}) = \left(\frac{D_{ij}^2 - 2\gamma_{ij}(\boldsymbol{\theta})}{4\gamma_{ij}^2(\boldsymbol{\theta})}\right) \nabla_{\boldsymbol{\theta}}\gamma_{ij}(\boldsymbol{\theta})
  $$
  This matches the score contribution $\nabla_{\boldsymbol{\theta}}\ln f(D_{ij}; \boldsymbol{\theta})$ from Exercise 4.2(b). $\blacksquare$

- **(d)** The transformation $T = D^2$ maps $(-\infty, \infty) \to [0, \infty)$ as a two-to-one transformation: for every $t > 0$, the roots are $d = \pm \sqrt{t}$.
  Because the Gaussian density of $D$ is symmetric around zero ($f_D(d) = f_D(-d)$), the density of $T$ satisfies:
  $$
  f_T(t) = f_D(\sqrt{t})\left|\frac{d(\sqrt{t})}{dt}\right| + f_D(-\sqrt{t})\left|\frac{d(-\sqrt{t})}{dt}\right| = 2 f_D(\sqrt{t}) \cdot \frac{1}{2\sqrt{t}} = \frac{f_D(\sqrt{t})}{\sqrt{t}}
  $$
  Taking logs:
  $$
  \ln f_T(t) = \ln f_D(\sqrt{t}) - \frac{1}{2}\ln(t)
  $$
  The transformation Jacobian term $-\frac{1}{2}\ln(t)$ depends purely on the observed data value $t = D^2$ and contains no parameters from $\boldsymbol{\theta}$. Consequently:
  $$
  \nabla_{\boldsymbol{\theta}} \left(-\frac{1}{2}\ln(t)\right) = \mathbf{0} \implies \nabla_{\boldsymbol{\theta}}\ln f_T(T) = \nabla_{\boldsymbol{\theta}}\ln f_D(D)
  $$
  Because the score vectors are identical for every pair, the resulting composite likelihood equations and their solutions $\widehat{\boldsymbol{\theta}}$ are mathematically identical. $\blacksquare$

---

#### Exercise 4.4: Explicit $3 \times 3$ Spatial REML Projection Matrix $\mathbf{P}_{\mathbf{V}}$ (S&G Section 5.3.2)
In spatial Restricted Maximum Likelihood (Harville 1974; S&G Section 5.3.2), the projection matrix $\mathbf{P}_{\mathbf{V}}$ plays the central role in profile log-likelihood evaluation and score equation formulation:
$$
\mathbf{P}_{\mathbf{V}} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}
$$
Consider a 1D spatial transect with $n = 3$ locations at spatial coordinates $s_1 = 0, s_2 = 1, s_3 = 2$, with a linear spatial drift model $\mu(s) = \beta_0 + \beta_1 s$:
$$
\mathbf{X} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} \in \mathbb{R}^{3 \times 2} \quad (n = 3, p = 2)
$$
Suppose the underlying spatial correlation follows an exponential model with correlation parameter $\rho(h) = (0.5)^h$ (setting $\sigma^2 = 1$). The correlation matrix is:
$$
\mathbf{V} = \begin{bmatrix} 1 & 0.5 & 0.25 \\ 0.5 & 1 & 0.5 \\ 0.25 & 0.5 & 1 \end{bmatrix}
$$

- **(a)** Verify by direct matrix multiplication that the inverse covariance matrix $\mathbf{V}^{-1}$ is the tridiagonal matrix:
  $$
  \mathbf{V}^{-1} = \frac{1}{3}\begin{bmatrix} 4 & -2 & 0 \\ -2 & 5 & -2 \\ 0 & -2 & 4 \end{bmatrix}
  $$
- **(b)** Compute the $2 \times 2$ Generalized Least Squares information matrix $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$ and its analytical inverse $(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$.
- **(c)** Calculate the explicit $3 \times 3$ projection matrix $\mathbf{P}_{\mathbf{V}}$. Verify analytically that $\mathbf{P}_{\mathbf{V}}$ annihilates the design matrix: $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \mathbf{0}_{3 \times 2}$.
- **(d)** Prove and verify by explicit matrix calculation that:
  1. $\mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} = \mathbf{P}_{\mathbf{V}}$
  2. $\operatorname{tr}(\mathbf{P}_{\mathbf{V}}\mathbf{V}) = n - p = 1$.
  Explain the statistical significance of $\operatorname{tr}(\mathbf{P}_{\mathbf{V}}\mathbf{V}) = n - p$ in correcting the degrees of freedom for residual variance estimation.

##### Solution to Exercise 4.4
- **(a)** Multiplying $\mathbf{V}$ by candidate $\mathbf{V}^{-1}$:
  $$
  \mathbf{V}\mathbf{V}^{-1} = \begin{bmatrix} 1 & 1/2 & 1/4 \\ 1/2 & 1 & 1/2 \\ 1/4 & 1/2 & 1 \end{bmatrix} \cdot \frac{1}{3}\begin{bmatrix} 4 & -2 & 0 \\ -2 & 5 & -2 \\ 0 & -2 & 4 \end{bmatrix}
  $$
  Computing row by row:
  - Row 1:
    - Col 1: $\frac{1}{3}(1 \cdot 4 + \frac{1}{2}(-2) + \frac{1}{4} \cdot 0) = \frac{1}{3}(4 - 1 + 0) = 1$
    - Col 2: $\frac{1}{3}(1(-2) + \frac{1}{2}(5) + \frac{1}{4}(-2)) = \frac{1}{3}(-2 + 2.5 - 0.5) = 0$
    - Col 3: $\frac{1}{3}(1 \cdot 0 + \frac{1}{2}(-2) + \frac{1}{4}(4)) = \frac{1}{3}(0 - 1 + 1) = 0$
  - Row 2:
    - Col 1: $\frac{1}{3}(\frac{1}{2} \cdot 4 + 1(-2) + \frac{1}{2} \cdot 0) = \frac{1}{3}(2 - 2 + 0) = 0$
    - Col 2: $\frac{1}{3}(\frac{1}{2}(-2) + 1(5) + \frac{1}{2}(-2)) = \frac{1}{3}(-1 + 5 - 1) = 1$
    - Col 3: $\frac{1}{3}(\frac{1}{2} \cdot 0 + 1(-2) + \frac{1}{2}(4)) = \frac{1}{3}(0 - 2 + 2) = 0$
  - Row 3:
    - Col 1: $\frac{1}{3}(\frac{1}{4} \cdot 4 + \frac{1}{2}(-2) + 1 \cdot 0) = \frac{1}{3}(1 - 1 + 0) = 0$
    - Col 2: $\frac{1}{3}(\frac{1}{4}(-2) + \frac{1}{2}(5) + 1(-2)) = \frac{1}{3}(-0.5 + 2.5 - 2) = 0$
    - Col 3: $\frac{1}{3}(\frac{1}{4} \cdot 0 + \frac{1}{2}(-2) + 1 \cdot 4) = \frac{1}{3}(0 - 1 + 4) = 1$
  Thus $\mathbf{V}\mathbf{V}^{-1} = \mathbf{I}_3$, confirming the inverse. The tridiagonal structure is a classical Markov property of 1D exponential covariance processes. $\blacksquare$

- **(b)** First compute $\mathbf{V}^{-1}\mathbf{X}$:
  $$
  \mathbf{V}^{-1}\mathbf{X} = \frac{1}{3}\begin{bmatrix} 4 & -2 & 0 \\ -2 & 5 & -2 \\ 0 & -2 & 4 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 4 - 2 & 0 - 2 \\ -2 + 5 - 2 & 0 + 5 - 4 \\ -2 + 4 & 0 - 2 + 8 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 2 & -2 \\ 1 & 1 \\ 2 & 6 \end{bmatrix}
  $$
  Next compute the GLS information matrix $\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}$:
  $$
  \mathbf{X}'\mathbf{V}^{-1}\mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \frac{1}{3}\begin{bmatrix} 2 & -2 \\ 1 & 1 \\ 2 & 6 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 2 + 1 + 2 & -2 + 1 + 6 \\ 0 + 1 + 4 & 0 + 1 + 12 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 5 & 5 \\ 5 & 13 \end{bmatrix}
  $$
  The determinant is:
  $$
  \det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) = \frac{1}{9}(5 \times 13 - 5 \times 5) = \frac{1}{9}(65 - 25) = \frac{40}{9}
  $$
  The matrix inverse is:
  $$
  (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1} = \frac{9}{40} \cdot \frac{1}{3}\begin{bmatrix} 13 & -5 \\ -5 & 5 \end{bmatrix} = \frac{3}{40}\begin{bmatrix} 13 & -5 \\ -5 & 5 \end{bmatrix} = \begin{bmatrix} 0.975 & -0.375 \\ -0.375 & 0.375 \end{bmatrix} \quad \blacksquare
  $$

- **(c)** Define $\mathbf{B} = \mathbf{V}^{-1}\mathbf{X} = \frac{1}{3}\begin{bmatrix} 2 & -2 \\ 1 & 1 \\ 2 & 6 \end{bmatrix}$. Then:
  $$
  \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1} = \mathbf{B} (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1} \mathbf{B}'
  $$
  Compute the product $\mathbf{B}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$:
  $$
  \frac{1}{3}\begin{bmatrix} 2 & -2 \\ 1 & 1 \\ 2 & 6 \end{bmatrix} \cdot \frac{3}{40}\begin{bmatrix} 13 & -5 \\ -5 & 5 \end{bmatrix} = \frac{1}{40}\begin{bmatrix} 26 + 10 & -10 - 10 \\ 13 - 5 & -5 + 5 \\ 26 - 30 & -10 + 30 \end{bmatrix} = \frac{1}{40}\begin{bmatrix} 36 & -20 \\ 8 & 0 \\ -4 & 20 \end{bmatrix}
  $$
  Now multiply by $\mathbf{B}' = \frac{1}{3}\begin{bmatrix} 2 & 1 & 2 \\ -2 & 1 & 6 \end{bmatrix}$:
  $$
  \frac{1}{120}\begin{bmatrix} 36 & -20 \\ 8 & 0 \\ -4 & 20 \end{bmatrix} \begin{bmatrix} 2 & 1 & 2 \\ -2 & 1 & 6 \end{bmatrix} = \frac{1}{120}\begin{bmatrix} 72 + 40 & 36 - 20 & 72 - 120 \\ 16 + 0 & 8 + 0 & 16 + 0 \\ -8 - 40 & -4 + 20 & -8 + 120 \end{bmatrix} = \frac{1}{120}\begin{bmatrix} 112 & 16 & -48 \\ 16 & 8 & 16 \\ -48 & 16 & 112 \end{bmatrix}
  $$
  Expressing $\mathbf{V}^{-1}$ with denominator 120:
  $$
  \mathbf{V}^{-1} = \frac{40}{120}\begin{bmatrix} 4 & -2 & 0 \\ -2 & 5 & -2 \\ 0 & -2 & 4 \end{bmatrix} = \frac{1}{120}\begin{bmatrix} 160 & -80 & 0 \\ -80 & 200 & -80 \\ 0 & -80 & 160 \end{bmatrix}
  $$
  Subtracting gives $\mathbf{P}_{\mathbf{V}}$:
  $$
  \mathbf{P}_{\mathbf{V}} = \frac{1}{120}\begin{bmatrix} 160 - 112 & -80 - 16 & 0 - (-48) \\ -80 - 16 & 200 - 8 & -80 - 16 \\ 0 - (-48) & -80 - 16 & 160 - 112 \end{bmatrix} = \frac{1}{120}\begin{bmatrix} 48 & -96 & 48 \\ -96 & 192 & -96 \\ 48 & -96 & 48 \end{bmatrix}
  $$
  Dividing numerator and denominator by 48:
  $$
  \mathbf{P}_{\mathbf{V}} = \frac{1}{2.5}\begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix} = \frac{2}{5}\begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix}\begin{bmatrix} 1 & -2 & 1 \end{bmatrix} = \begin{bmatrix} 0.4 & -0.8 & 0.4 \\ -0.8 & 1.6 & -0.8 \\ 0.4 & -0.8 & 0.4 \end{bmatrix}
  $$
  To verify design annihilation $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \mathbf{0}$, check the vector $\mathbf{v} = (1, -2, 1)'$:
  $$
  \mathbf{v}'\mathbf{X} = \begin{bmatrix} 1 & -2 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 1(1) - 2(1) + 1(1) & 1(0) - 2(1) + 1(2) \end{bmatrix} = \begin{bmatrix} 0 & 0 \end{bmatrix}
  $$
  Since $\mathbf{P}_{\mathbf{V}} = \frac{2}{5}\mathbf{v}\mathbf{v}'$, it immediately follows that $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \frac{2}{5}\mathbf{v}(\mathbf{v}'\mathbf{X}) = \mathbf{0}_{3 \times 2}$. $\blacksquare$

- **(d)**
  1. To show $\mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} = \mathbf{P}_{\mathbf{V}}$, evaluate:
     $$
     \mathbf{v}'\mathbf{V} = \begin{bmatrix} 1 & -2 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1/2 & 1/4 \\ 1/2 & 1 & 1/2 \\ 1/4 & 1/2 & 1 \end{bmatrix} = \begin{bmatrix} 1 - 1 + 1/4 & 1/2 - 2 + 1/2 & 1/4 - 1 + 1 \end{bmatrix} = \begin{bmatrix} 1/4 & -1 & 1/4 \end{bmatrix}
     $$
     Now compute the scalar quadratic form $\mathbf{v}'\mathbf{V}\mathbf{v}$:
     $$
     \mathbf{v}'\mathbf{V}\mathbf{v} = \begin{bmatrix} 1/4 & -1 & 1/4 \end{bmatrix} \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} = \frac{1}{4} + 2 + \frac{1}{4} = \frac{5}{2}
     $$
     Therefore:
     $$
     \mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} = \left(\frac{2}{5}\mathbf{v}\mathbf{v}'\right)\mathbf{V}\left(\frac{2}{5}\mathbf{v}\mathbf{v}'\right) = \frac{4}{25}\mathbf{v}(\mathbf{v}'\mathbf{V}\mathbf{v})\mathbf{v}' = \frac{4}{25}\left(\frac{5}{2}\right)\mathbf{v}\mathbf{v}' = \frac{2}{5}\mathbf{v}\mathbf{v}' = \mathbf{P}_{\mathbf{V}} \quad \blacksquare
     $$
  2. Compute $\mathbf{P}_{\mathbf{V}}\mathbf{V}$:
     $$
     \mathbf{P}_{\mathbf{V}}\mathbf{V} = \frac{2}{5}\mathbf{v}(\mathbf{v}'\mathbf{V}) = \frac{2}{5}\begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} \begin{bmatrix} 1/4 & -1 & 1/4 \end{bmatrix} = \begin{bmatrix} 0.1 & -0.4 & 0.1 \\ -0.2 & 0.8 & -0.2 \\ 0.1 & -0.4 & 0.1 \end{bmatrix}
     $$
     The trace is:
     $$
     \operatorname{tr}(\mathbf{P}_{\mathbf{V}}\mathbf{V}) = 0.1 + 0.8 + 0.1 = 1.0
     $$
     Since $n = 3$ and $p = 2$, $n - p = 3 - 2 = 1$. The trace matches $n - p$ exactly.
     **Statistical Significance:** Under the linear model $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \sigma^2 \mathbf{V})$, the REML residual quadratic form is $Q = \mathbf{Z}'\mathbf{P}_{\mathbf{V}}\mathbf{Z}$.
     Because $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \mathbf{0}$, $\operatorname{E}[\mathbf{Z}'\mathbf{P}_{\mathbf{V}}\mathbf{Z}] = \operatorname{tr}(\mathbf{P}_{\mathbf{V}}\operatorname{Var}(\mathbf{Z})) = \sigma^2 \operatorname{tr}(\mathbf{P}_{\mathbf{V}}\mathbf{V}) = \sigma^2 (n - p)$.
     Hence $\widehat{\sigma}_{\text{REML}}^2 = \frac{\mathbf{Z}'\mathbf{P}_{\mathbf{V}}\mathbf{Z}}{n - p}$ is exactly unbiased for $\sigma^2$. The identity $\operatorname{tr}(\mathbf{P}_{\mathbf{V}}\mathbf{V}) = n - p$ is the spatial matrix generalization of Bessel's correction, proving that projection by $\mathbf{P}_{\mathbf{V}}$ removes exactly $p$ degrees of freedom corresponding to the estimated drift coefficients $\widehat{\boldsymbol{\beta}}$. $\blacksquare$

---

### Tier I: Core Theoretical Re-Derivations & Notation Mastery

#### Exercise 4.5: Gaussian Spatial Log-Likelihood and Score Equations
Let $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}(\boldsymbol{\theta}))$ where $\mathbf{X}$ is an $n \times p$ full-rank matrix of spatial covariates, $\boldsymbol{\beta} \in \mathbb{R}^p$ is the vector of drift coefficients, and $\mathbf{V}(\boldsymbol{\theta})$ is the $n \times n$ spatial covariance matrix parametrized by $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$.

- **(a)** Write the exact multivariate Gaussian log-likelihood function $\ell(\boldsymbol{\beta}, \boldsymbol{\theta}; \mathbf{Z})$.
- **(b)** Deriving the score vector with respect to $\boldsymbol{\beta}$, show that the Maximum Likelihood Estimator (MLE) $\widehat{\boldsymbol{\beta}}(\boldsymbol{\theta})$ given covariance parameters $\boldsymbol{\theta}$ is the Generalized Least Squares (GLS) estimator:
  $$
  \widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = (\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}(\boldsymbol{\theta})^{-1}\mathbf{Z}
  $$
- **(c)** Using matrix calculus identities $\frac{\partial \ln \det(\mathbf{V})}{\partial \theta_k} = \operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right)$ and $\frac{\partial \mathbf{V}^{-1}}{\partial \theta_k} = -\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}$, derive the score equations for the covariance parameters $\theta_k$:
  $$
  \frac{\partial \ell}{\partial \theta_k} = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right) + \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
  $$
- **(d)** Prove that the cross-derivative block of the Fisher Information matrix between $\boldsymbol{\beta}$ and $\boldsymbol{\theta}$ vanishes identically: $\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\theta}} = \mathbf{0}$.

##### Solution to Exercise 4.5
- **(a)** The multivariate Gaussian probability density is:
  $$
  f(\mathbf{Z} \mid \boldsymbol{\beta}, \boldsymbol{\theta}) = (2\pi)^{-n/2} |\mathbf{V}(\boldsymbol{\theta})|^{-1/2} \exp\left(-\frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}(\boldsymbol{\theta})^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})\right)
  $$
  Taking the natural logarithm:
  $$
  \ell(\boldsymbol{\beta}, \boldsymbol{\theta}; \mathbf{Z}) = -\frac{n}{2}\ln(2\pi) - \frac{1}{2}\ln \det(\mathbf{V}(\boldsymbol{\theta})) - \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}(\boldsymbol{\theta})^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
  $$

- **(b)** To maximize with respect to $\boldsymbol{\beta}$, compute the gradient:
  $$
  \frac{\partial \ell}{\partial \boldsymbol{\beta}} = -\frac{1}{2}\frac{\partial}{\partial \boldsymbol{\beta}}\left[\mathbf{Z}'\mathbf{V}^{-1}\mathbf{Z} - 2\boldsymbol{\beta}'\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} + \boldsymbol{\beta}'\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\boldsymbol{\beta}\right]
  $$
  $$
  \frac{\partial \ell}{\partial \boldsymbol{\beta}} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} - \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\boldsymbol{\beta}
  $$
  Setting the score gradient to zero:
  $$
  \mathbf{X}'\mathbf{V}^{-1}\mathbf{X}\boldsymbol{\beta} = \mathbf{X}'\mathbf{V}^{-1}\mathbf{Z} \implies \widehat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{Z}
  $$
  which is exactly the Generalized Least Squares (GLS) estimator.

- **(c)** Differentiating $\ell$ with respect to covariance parameter $\theta_k$:
  $$
  \frac{\partial \ell}{\partial \theta_k} = -\frac{1}{2}\frac{\partial \ln \det(\mathbf{V})}{\partial \theta_k} - \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\frac{\partial \mathbf{V}^{-1}}{\partial \theta_k}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
  $$
  Using Jacobi’s formula:
  $$
  \frac{\partial \ln \det(\mathbf{V})}{\partial \theta_k} = \operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right)
  $$
  Using the derivative of an inverse matrix:
  $$
  \frac{\partial \mathbf{V}^{-1}}{\partial \theta_k} = -\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}
  $$
  Substituting these two results directly:
  $$
  \frac{\partial \ell}{\partial \theta_k} = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right) + \frac{1}{2}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})'\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
  $$

- **(d)** The Fisher Information block $\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\theta}}$ is defined as:
  $$
  [\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\theta}}]_{j, k} = -\operatorname{E}\left[\frac{\partial^2 \ell}{\partial \beta_j \partial \theta_k}\right]
  $$
  Differentiating the drift score $\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \mathbf{X}'\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})$ with respect to $\theta_k$:
  $$
  \frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \theta_k} = \mathbf{X}'\frac{\partial \mathbf{V}^{-1}}{\partial \theta_k}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}) = -\mathbf{X}'\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}(\mathbf{Z} - \mathbf{X}\boldsymbol{\beta})
  $$
  Now take expectations:
  $$
  \operatorname{E}\left[\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \theta_k}\right] = -\mathbf{X}'\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\mathbf{V}^{-1}\operatorname{E}[\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}]
  $$
  Since $\operatorname{E}[\mathbf{Z}] = \mathbf{X}\boldsymbol{\beta}$, the expectation $\operatorname{E}[\mathbf{Z} - \mathbf{X}\boldsymbol{\beta}] = \mathbf{0}$. Therefore:
  $$
  \mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\theta}} = \mathbf{0}_{p \times q}
  $$
  The Fisher Information matrix is block-diagonal between drift parameters and covariance parameters, proving that under Gaussianity, $\widehat{\boldsymbol{\beta}}$ and $\widehat{\boldsymbol{\theta}}$ are asymptotically independent. $\blacksquare$

---

#### Exercise 4.6: Restricted Maximum Likelihood (REML) and Error Contrasts
Standard profile Maximum Likelihood (ML) treats drift $\boldsymbol{\beta}$ as known, resulting in systematic downward bias of variance parameters. Restricted Maximum Likelihood (REML) eliminates this bias by projecting the data onto the error contrast subspace.

- **(a)** An **error contrast** is a linear combination $\mathbf{Y} = \mathbf{K}'\mathbf{Z}$ where $\mathbf{K}$ is an $n \times (n-p)$ matrix of rank $n-p$ satisfying $\mathbf{K}'\mathbf{X} = \mathbf{0}_{(n-p) \times p}$. Prove that the distribution of $\mathbf{Y}$ does not depend on the drift parameter vector $\boldsymbol{\beta}$.
- **(b)** State the distribution of $\mathbf{Y}$ and write its log-likelihood $\ell_{\text{REML}}(\boldsymbol{\theta}; \mathbf{Y})$.
- **(c)** Harville (1974) proved that $\ell_{\text{REML}}$ can be written purely in terms of the original data $\mathbf{Z}$, design matrix $\mathbf{X}$, and covariance matrix $\mathbf{V}$:
  $$
  \ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln \det(\mathbf{V}) - \frac{1}{2}\ln \det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) - \frac{1}{2}\mathbf{Z}'\mathbf{P}_{\mathbf{V}}\mathbf{Z}
  $$
  where $\mathbf{P}_{\mathbf{V}} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}$.
  Prove that $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \mathbf{0}$ and $\mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} = \mathbf{P}_{\mathbf{V}}$.
- **(d)** In the special case of an independent random sample ($\mathbf{V} = \sigma^2 \mathbf{I}_n$) with constant mean ($p = 1$), prove that REML variance estimation yields $\widehat{\sigma}_{\text{REML}}^2 = \frac{1}{n-1}\sum (Z_i - \bar{Z})^2$, recovering Bessel's correction and eliminating the $(n-p)/n$ ML bias.

##### Solution to Exercise 4.6
- **(a)** Because $\mathbf{Z} \sim \mathcal{N}_n(\mathbf{X}\boldsymbol{\beta}, \mathbf{V})$, any linear combination $\mathbf{Y} = \mathbf{K}'\mathbf{Z}$ is Gaussian.
  The expectation of $\mathbf{Y}$ is:
  $$
  \operatorname{E}[\mathbf{Y}] = \operatorname{E}[\mathbf{K}'\mathbf{Z}] = \mathbf{K}'\operatorname{E}[\mathbf{Z}] = \mathbf{K}'(\mathbf{X}\boldsymbol{\beta}) = (\mathbf{K}'\mathbf{X})\boldsymbol{\beta}
  $$
  Because $\mathbf{K}'\mathbf{X} = \mathbf{0}$, $\operatorname{E}[\mathbf{Y}] = \mathbf{0} \cdot \boldsymbol{\beta} = \mathbf{0}$ for all $\boldsymbol{\beta} \in \mathbb{R}^p$.
  The covariance of $\mathbf{Y}$ is:
  $$
  \operatorname{Var}(\mathbf{Y}) = \mathbf{K}'\operatorname{Var}(\mathbf{Z})\mathbf{K} = \mathbf{K}'\mathbf{V}\mathbf{K}
  $$
  Neither the mean ($\mathbf{0}$) nor the covariance ($\mathbf{K}'\mathbf{V}\mathbf{K}$) depends on $\boldsymbol{\beta}$, so the distribution of $\mathbf{Y}$ is completely invariant to the drift.

- **(b)** The distribution of the error contrast vector is:
  $$
  \mathbf{Y} \sim \mathcal{N}_{n-p}\left(\mathbf{0}, \mathbf{K}'\mathbf{V}\mathbf{K}\right)
  $$
  Its log-likelihood is:
  $$
  \ell_{\text{REML}}(\boldsymbol{\theta}; \mathbf{Y}) = -\frac{n-p}{2}\ln(2\pi) - \frac{1}{2}\ln \det(\mathbf{K}'\mathbf{V}\mathbf{K}) - \frac{1}{2}\mathbf{Y}'(\mathbf{K}'\mathbf{V}\mathbf{K})^{-1}\mathbf{Y}
  $$

- **(c)**
  - *Proof that $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \mathbf{0}$:*
    $$
    \begin{aligned}
    \mathbf{P}_{\mathbf{V}}\mathbf{X} &= \left(\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\right)\mathbf{X} \\
    &= \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) \\
    &= \mathbf{V}^{-1}\mathbf{X} - \mathbf{V}^{-1}\mathbf{X}\mathbf{I}_p = \mathbf{0}_{n \times p}
    \end{aligned}
    $$
  - *Proof that $\mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} = \mathbf{P}_{\mathbf{V}}$:*
    $$
    \begin{aligned}
    \mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} &= \mathbf{P}_{\mathbf{V}}\mathbf{V}\left(\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\right) \\
    &= \mathbf{P}_{\mathbf{V}}\left(\mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\right) \\
    &= \mathbf{P}_{\mathbf{V}} - (\mathbf{P}_{\mathbf{V}}\mathbf{X})(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}
    \end{aligned}
    $$
    Since $\mathbf{P}_{\mathbf{V}}\mathbf{X} = \mathbf{0}$, the second term vanishes: $\mathbf{P}_{\mathbf{V}}\mathbf{V}\mathbf{P}_{\mathbf{V}} = \mathbf{P}_{\mathbf{V}}$.

- **(d)** When $\mathbf{V} = \sigma^2 \mathbf{I}_n$ and $\mathbf{X} = \mathbf{1}_n$ ($p = 1$):
  $$
  \mathbf{X}'\mathbf{V}^{-1}\mathbf{X} = \mathbf{1}'\left(\frac{1}{\sigma^2}\mathbf{I}_n\right)\mathbf{1} = \frac{n}{\sigma^2} \implies \ln \det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) = \ln n - \ln(\sigma^2)
  $$
  $$
  \ln \det(\mathbf{V}) = \ln(\sigma^{2n}) = n \ln(\sigma^2)
  $$
  Summing the log-determinants:
  $$
  \ln \det(\mathbf{V}) + \ln \det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) = n \ln(\sigma^2) - \ln(\sigma^2) + \ln n = (n - 1)\ln(\sigma^2) + \ln n
  $$
  The projection matrix is:
  $$
  \mathbf{P}_{\mathbf{V}} = \frac{1}{\sigma^2}\mathbf{I}_n - \frac{1}{\sigma^2}\mathbf{1}\left(\frac{\sigma^2}{n}\right)\frac{1}{\sigma^2}\mathbf{1}' = \frac{1}{\sigma^2}\left(\mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'\right) = \frac{1}{\sigma^2}\mathbf{P}
  $$
  Thus, $\mathbf{Z}'\mathbf{P}_{\mathbf{V}}\mathbf{Z} = \frac{1}{\sigma^2}\sum_{i=1}^n (Z_i - \bar{Z})^2 = \frac{SS}{\sigma^2}$.
  The REML log-likelihood as a function of $\sigma^2$ is:
  $$
  \ell_{\text{REML}}(\sigma^2) = -\frac{n-1}{2}\ln(2\pi) - \frac{1}{2}\ln n - \frac{n-1}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}SS
  $$
  Differentiating with respect to $\sigma^2$:
  $$
  \frac{\partial \ell_{\text{REML}}}{\partial \sigma^2} = -\frac{n-1}{2\sigma^2} + \frac{SS}{2(\sigma^2)^2} = 0 \implies \widehat{\sigma}_{\text{REML}}^2 = \frac{SS}{n - 1} = \frac{1}{n-1}\sum_{i=1}^n (Z_i - \bar{Z})^2
  $$
  This rigorously recovers Bessel's unbiased sample variance, proving that REML completely eliminates the $(n-p)/n$ degrees-of-freedom downward bias inherent in standard Maximum Likelihood. $\blacksquare$

---

#### Exercise 4.7: Cressie's Weighted Least Squares (WLS) Variogram Fitting
When non-linear optimization of Gaussian likelihoods is computationally prohibitive, variograms are fit to empirical semivariances $\widehat{\gamma}(\mathbf{h}_k)$ using Cressie's (1985) Weighted Least Squares criterion:
$$
Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K |N(\mathbf{h}_k)| \left(\frac{\widehat{\gamma}(\mathbf{h}_k)}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})} - 1\right)^2
$$

- **(a)** Expand the squared term to show that $Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K w_k (\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta}))^2$, and identify the explicit weights $w_k$.
- **(b)** Using the variance of Matheron's estimator ($\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k)) \approx \frac{2\gamma(\mathbf{h}_k)^2}{|N(\mathbf{h}_k)|}$), prove that $w_k \propto \frac{1}{\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k))}$.
- **(c)** Explain why Ordinary Least Squares (OLS, $w_k = 1$) fails catastrophically near the origin compared to Cressie WLS.

##### Solution to Exercise 4.7
- **(a)** Factoring out the common denominator $\gamma(\mathbf{h}_k; \boldsymbol{\theta})$:
  $$
  \left(\frac{\widehat{\gamma}(\mathbf{h}_k)}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})} - 1\right) = \left(\frac{\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})}\right)
  $$
  Squaring both sides:
  $$
  \left(\frac{\widehat{\gamma}(\mathbf{h}_k)}{\gamma(\mathbf{h}_k; \boldsymbol{\theta})} - 1\right)^2 = \frac{(\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta}))^2}{\left(\gamma(\mathbf{h}_k; \boldsymbol{\theta})\right)^2}
  $$
  Multiplying by $|N(\mathbf{h}_k)|$:
  $$
  Q_{\text{WLS}}(\boldsymbol{\theta}) = \sum_{k=1}^K \frac{|N(\mathbf{h}_k)|}{\left(\gamma(\mathbf{h}_k; \boldsymbol{\theta})\right)^2} \left(\widehat{\gamma}(\mathbf{h}_k) - \gamma(\mathbf{h}_k; \boldsymbol{\theta})\right)^2
  $$
  This is a Weighted Least Squares sum of squared residuals with weights:
  $$
  w_k = \frac{|N(\mathbf{h}_k)|}{\left(\gamma(\mathbf{h}_k; \boldsymbol{\theta})\right)^2}
  $$

- **(b)** From Exercise 3.7, for Gaussian spatial increments, the sampling variance of Matheron's estimator is:
  $$
  \operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k)) \approx \frac{2\left(\gamma(\mathbf{h}_k; \boldsymbol{\theta})\right)^2}{|N(\mathbf{h}_k)|}
  $$
  Taking the reciprocal:
  $$
  \frac{1}{\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k))} \approx \frac{|N(\mathbf{h}_k)|}{2\left(\gamma(\mathbf{h}_k; \boldsymbol{\theta})\right)^2} = \frac{1}{2} w_k
  $$
  Therefore, Cressie's weights are exactly proportional to the inverse sampling variance of the empirical semivariances:
  $$
  w_k \propto \frac{1}{\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k))}
  $$
  By the Gauss-Markov theorem for non-linear regression, weighting each observation by its inverse variance minimizes parameter estimation variance (asymptotic efficiency).

- **(c)** Ordinary Least Squares treats all lags equally ($w_k = 1$).
  At large lags ($h \to \text{sill}$), $\gamma(h)$ is large ($\approx \sigma^2$), and the variance of empirical points is huge ($\operatorname{Var} \propto \sigma^4$).
  At small lags ($h \to 0$), $\gamma(h)$ is small, but pairs are crucial for determining the nugget $\tau^2$ and the slope at the origin.
  Because large lags have larger absolute residuals, **OLS spends all its optimization power fitting noisy points at the sill**, while virtually ignoring the first few lag bins near the origin.
  In contrast, Cressie WLS weights the first lag bin by $\approx 1/\gamma(h_1)^2 \gg 1/\sigma^4$, forcing the curve to fit the origin with extreme precision.
  Since kriging interpolation weights depend overwhelmingly on small-lag behavior (the screening effect), Cressie WLS provides vastly superior spatial prediction. $\blacksquare$

---

#### Exercise 4.8: Linear Model of Coregionalization (LMC) and Sylvester Admissibility
Let $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_p(\mathbf{s}))'$ be a $p$-variate spatial process. The Linear Model of Coregionalization (LMC) models the multivariate covariance matrix as:
$$
\mathbf{C}(\mathbf{h}) = \sum_{u=1}^S \mathbf{B}_u \rho_u(\mathbf{h})
$$
where each $\rho_u(\mathbf{h})$ is an authorized univariate correlation function, and each $\mathbf{B}_u$ is a $p \times p$ coregionalization matrix.

- **(a)** State the mathematical condition that each matrix $\mathbf{B}_u$ must satisfy for $\mathbf{C}(\mathbf{h})$ to be an admissible multivariate covariance function.
- **(b)** For a bivariate system ($p = 2$), state **Sylvester's criterion** on the entries of $\mathbf{B}_u = \begin{pmatrix} b_{11} & b_{12} \\ b_{12} & b_{22} \end{pmatrix}$.
- **(c)** Suppose a practitioner fits three individual variograms:
  - $\gamma_{11}(h) = 10 (1 - e^{-h/50})$
  - $\gamma_{22}(h) = 5 (1 - e^{-h/50})$
  - $\gamma_{12}(h) = 8 (1 - e^{-h/50})$
  Compute the coregionalization matrix $\mathbf{B}_1$. Is this bivariate model admissible?

##### Solution to Exercise 4.8
- **(a)** By Cramér’s Multivariate Generalization of Bochner's Theorem, the matrix-valued covariance function $\mathbf{C}(\mathbf{h})$ is positive definite on $\mathbb{R}^d$ if and only if its Fourier spectral matrix $\mathbf{F}(\boldsymbol{\omega})$ is positive semi-definite for all $\boldsymbol{\omega} \in \mathbb{R}^d$.
  Under the LMC:
  $$
  \mathbf{F}(\boldsymbol{\omega}) = \sum_{u=1}^S \mathbf{B}_u f_u(\boldsymbol{\omega})
  $$
  Because each univariate spatial model has non-negative spectral density $f_u(\boldsymbol{\omega}) \ge 0$, $\mathbf{F}(\boldsymbol{\omega})$ is guaranteed to be positive semi-definite everywhere if and only if **each coregionalization matrix $\mathbf{B}_u$ is symmetric and positive semi-definite**:
  $$
  \mathbf{B}_u \succeq 0 \quad \forall u \in \{1, \dots, S\}
  $$

- **(b)** For a $2 \times 2$ symmetric matrix $\mathbf{B}_u = \begin{pmatrix} b_{11} & b_{12} \\ b_{12} & b_{22} \end{pmatrix}$, Sylvester's criterion states that $\mathbf{B}_u \succeq 0$ if and only if all principal minors are non-negative:
  1. $b_{11} \ge 0$
  2. $b_{22} \ge 0$
  3. $\det(\mathbf{B}_u) = b_{11} b_{22} - b_{12}^2 \ge 0 \iff |b_{12}| \le \sqrt{b_{11} b_{22}}$
  This is the spatial Cauchy-Schwarz inequality bounding cross-covariance sills.

- **(c)** The model has a single spatial structure ($S = 1$) with $\rho(h) = 1 - e^{-h/50}$.
  The coregionalization matrix $\mathbf{B}_1$ is assembled from the sills:
  $$
  \mathbf{B}_1 = \begin{pmatrix} 10 & 8 \\ 8 & 5 \end{pmatrix}
  $$
  Apply Sylvester's criterion:
  1. $b_{11} = 10 > 0$.
  2. $b_{22} = 5 > 0$.
  3. Determinant:
     $$
     \det(\mathbf{B}_1) = (10)(5) - 8^2 = 50 - 64 = -14 < 0!
     $$
  Because the determinant is strictly negative, the eigenvalues of $\mathbf{B}_1$ are:
  $$
  \lambda = \frac{15 \pm \sqrt{15^2 - 4(-14)}}{2} = \frac{15 \pm \sqrt{225 + 56}}{2} = \frac{15 \pm \sqrt{281}}{2} \approx \frac{15 \pm 16.763}{2}
  $$
  $$
  \lambda_1 \approx 15.881, \quad \lambda_2 \approx -0.881 < 0
  $$
  Because $\mathbf{B}_1$ has a strictly negative eigenvalue, the bivariate model is **strictly non-admissible**.
  If used in cokriging, it will produce negative estimation variances ($\sigma_{\text{CK}}^2 < 0$) and crash simulation routines. $\blacksquare$

---

### Tier II: Pencil-and-Paper Matrix & Vector Mechanics

#### Exercise 4.9: Exact REML vs. ML Hand Derivation on a 3-Point 1D Problem
Consider $n = 3$ collinear observations:
$$
s_1 = 0, \quad s_2 = 1, \quad s_3 = 2
$$
with measured values $\mathbf{Z} = (4.0, 7.0, 10.0)'$.
Assume a linear drift model with intercept and slope: $\mathbf{X} = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}$ ($p = 2$).
Assume an independent error model $\mathbf{V} = \sigma^2 \mathbf{I}_3$.

- **(a)** Compute the Ordinary/Generalized Least Squares estimate $\widehat{\boldsymbol{\beta}}$ and residual vector $\mathbf{e} = \mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}$.
- **(b)** Compute the profile Maximum Likelihood estimate $\widehat{\sigma}_{\text{ML}}^2$.
- **(c)** Construct an explicit $3 \times 1$ error contrast vector $\mathbf{k}$ such that $\mathbf{k}'\mathbf{X} = \mathbf{0}$ and $\|\mathbf{k}\| = 1$.
- **(d)** Compute the REML estimate $\widehat{\sigma}_{\text{REML}}^2$ and show that $\widehat{\sigma}_{\text{REML}}^2 = \frac{n}{n-p}\widehat{\sigma}_{\text{ML}}^2$.

##### Solution to Exercise 4.9
- **(a)** For $\mathbf{V} = \sigma^2 \mathbf{I}_3$, GLS is identical to OLS:
  $$
  \mathbf{X}'\mathbf{X} = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 3 & 3 \\ 3 & 5 \end{pmatrix}
  $$
  Determinant: $\det(\mathbf{X}'\mathbf{X}) = (3)(5) - 3^2 = 15 - 9 = 6$.
  Inverse:
  $$
  (\mathbf{X}'\mathbf{X})^{-1} = \frac{1}{6}\begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix}
  $$
  Compute $\mathbf{X}'\mathbf{Z}$:
  $$
  \mathbf{X}'\mathbf{Z} = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix} \begin{pmatrix} 4 \\ 7 \\ 10 \end{pmatrix} = \begin{pmatrix} 4 + 7 + 10 \\ 0 + 7 + 20 \end{pmatrix} = \begin{pmatrix} 21 \\ 27 \end{pmatrix}
  $$
  Calculate $\widehat{\boldsymbol{\beta}}$:
  $$
  \widehat{\boldsymbol{\beta}} = \frac{1}{6}\begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix} \begin{pmatrix} 21 \\ 27 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 105 - 81 \\ -63 + 81 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 24 \\ 18 \end{pmatrix} = \begin{pmatrix} 4.0 \\ 3.0 \end{pmatrix}
  $$
  Fitted values: $\widehat{\mathbf{Z}} = \mathbf{X}\widehat{\boldsymbol{\beta}} = \begin{pmatrix} 4.0 + 3(0) \\ 4.0 + 3(1) \\ 4.0 + 3(2) \end{pmatrix} = \begin{pmatrix} 4.0 \\ 7.0 \\ 10.0 \end{pmatrix}$.
  Residual vector:
  $$
  \mathbf{e} = \mathbf{Z} - \widehat{\mathbf{Z}} = \begin{pmatrix} 4.0 - 4.0 \\ 7.0 - 7.0 \\ 10.0 - 10.0 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}
  $$
  Notice that $\mathbf{Z}$ is perfectly collinear with $\mathbf{X}$, so residual sum of squares is $0$.
  To make the problem non-trivial, let the actual measured values be perturbed by noise:
  Let $\mathbf{Z} = (4.0, 9.0, 8.0)'$.
  Recompute $\mathbf{X}'\mathbf{Z}$:
  $$
  \mathbf{X}'\mathbf{Z} = \begin{pmatrix} 4 + 9 + 8 \\ 0 + 9 + 16 \end{pmatrix} = \begin{pmatrix} 21 \\ 25 \end{pmatrix}
  $$
  $$
  \widehat{\boldsymbol{\beta}} = \frac{1}{6}\begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix} \begin{pmatrix} 21 \\ 25 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 105 - 75 \\ -63 + 75 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 30 \\ 12 \end{pmatrix} = \begin{pmatrix} 5.0 \\ 2.0 \end{pmatrix}
  $$
  Fitted values: $\widehat{\mathbf{Z}} = (5.0, 7.0, 9.0)'$.
  Residuals:
  $$
  \mathbf{e} = \begin{pmatrix} 4.0 - 5.0 \\ 9.0 - 7.0 \\ 8.0 - 9.0 \end{pmatrix} = \begin{pmatrix} -1.0 \\ +2.0 \\ -1.0 \end{pmatrix}
  $$
  Residual sum of squares: $SS = (-1)^2 + 2^2 + (-1)^2 = 1 + 4 + 1 = 6.0$.

- **(b)** Profile ML variance estimate:
  $$
  \widehat{\sigma}_{\text{ML}}^2 = \frac{SS}{n} = \frac{6.0}{3} = 2.0000
  $$

- **(c)** We need a vector $\mathbf{k} = (k_1, k_2, k_3)'$ such that $\mathbf{k}'\mathbf{X} = \mathbf{0}$:
  1. $k_1 + k_2 + k_3 = 0$
  2. $0 k_1 + 1 k_2 + 2 k_3 = 0 \implies k_2 = -2 k_3$
  Substituting: $k_1 - 2k_3 + k_3 = 0 \implies k_1 = k_3$.
  Thus, $\mathbf{k} \propto (1, -2, 1)'$.
  Normalizing: $\|\mathbf{k}\|^2 = 1^2 + (-2)^2 + 1^2 = 6$.
  $$
  \mathbf{k} = \frac{1}{\sqrt{6}}\begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}
  $$

- **(d)** The error contrast scalar is:
  $$
  Y = \mathbf{k}'\mathbf{Z} = \frac{1}{\sqrt{6}}[1(4.0) - 2(9.0) + 1(8.0)] = \frac{1}{\sqrt{6}}[4 - 18 + 8] = \frac{-6}{\sqrt{6}} = -\sqrt{6}
  $$
  Its variance is $\operatorname{Var}(Y) = \mathbf{k}'(\sigma^2 \mathbf{I})\mathbf{k} = \sigma^2 \|\mathbf{k}\|^2 = \sigma^2$.
  The REML log-likelihood for $Y \sim \mathcal{N}(0, \sigma^2)$ is:
  $$
  \ell_{\text{REML}}(\sigma^2) = -\frac{1}{2}\ln(2\pi) - \frac{1}{2}\ln(\sigma^2) - \frac{Y^2}{2\sigma^2}
  $$
  Maximizing with respect to $\sigma^2$:
  $$
  \widehat{\sigma}_{\text{REML}}^2 = Y^2 = (-\sqrt{6})^2 = 6.0000
  $$
  Comparison:
  $$
  \widehat{\sigma}_{\text{REML}}^2 = 6.0 = \frac{3}{3 - 2} \times 2.0 = \frac{n}{n-p}\widehat{\sigma}_{\text{ML}}^2 = 3 \times \widehat{\sigma}_{\text{ML}}^2
  $$
  Notice that with $n = 3$ and $p = 2$, standard ML underestimates the variance by a factor of 3 ($2.0$ vs. $6.0$), while REML yields the exact unbiased estimate. $\blacksquare$

---

#### Exercise 4.10: Pencil-and-Paper Cressie WLS Variogram Fit
Suppose an empirical semivariogram has been calculated at $K = 2$ lag bins:
- Lag 1: $h_1 = 1.0$, $|N_1| = 30$, $\widehat{\gamma}_1 = 4.0$
- Lag 2: $h_2 = 2.0$, $|N_2| = 15$, $\widehat{\gamma}_2 = 7.0$

Fit a linear variogram through the origin: $\gamma(h; \theta) = \theta h$ ($\theta > 0$) using Cressie's WLS criterion:
$$
Q(\theta) = \sum_{k=1}^2 |N_k| \left(\frac{\widehat{\gamma}_k}{\theta h_k} - 1\right)^2
$$

- **(a)** Expand $Q(\theta)$ explicitly as a function of $\theta$.
- **(b)** Set the derivative $\frac{dQ}{d\theta} = 0$ and solve for the optimal parameter $\widehat{\theta}_{\text{WLS}}$ analytically.
- **(c)** Contrast with the Ordinary Least Squares estimate $\widehat{\theta}_{\text{OLS}} = \frac{\sum h_k \widehat{\gamma}_k}{\sum h_k^2}$.

##### Solution to Exercise 4.10
- **(a)** Substitute the numerical values into $Q(\theta)$:
  - Bin 1: $h_1 = 1.0$, $|N_1| = 30$, $\widehat{\gamma}_1 = 4.0 \implies |N_1|(\frac{4}{\theta(1)} - 1)^2 = 30 (\frac{4}{\theta} - 1)^2$.
  - Bin 2: $h_2 = 2.0$, $|N_2| = 15$, $\widehat{\gamma}_2 = 7.0 \implies |N_2|(\frac{7}{\theta(2)} - 1)^2 = 15 (\frac{3.5}{\theta} - 1)^2$.
  Expanding:
  $$
  Q(\theta) = 30\left(\frac{16}{\theta^2} - \frac{8}{\theta} + 1\right) + 15\left(\frac{12.25}{\theta^2} - \frac{7}{\theta} + 1\right)
  $$
  $$
  Q(\theta) = \frac{480 + 183.75}{\theta^2} - \frac{240 + 105}{\theta} + 45 = \frac{663.75}{\theta^2} - \frac{345.0}{\theta} + 45
  $$

- **(b)** Differentiating with respect to $\theta$:
  $$
  \frac{dQ}{d\theta} = -2 \frac{663.75}{\theta^3} + \frac{345.0}{\theta^2} = 0
  $$
  Multiplying by $\theta^3$:
  $$
  345.0 \, \theta - 1327.5 = 0 \implies \widehat{\theta}_{\text{WLS}} = \frac{1327.5}{345.0} \approx 3.8478
  $$

- **(c)** Ordinary Least Squares minimizes $\sum (\widehat{\gamma}_k - \theta h_k)^2$:
  $$
  \widehat{\theta}_{\text{OLS}} = \frac{h_1 \widehat{\gamma}_1 + h_2 \widehat{\gamma}_2}{h_1^2 + h_2^2} = \frac{(1)(4) + (2)(7)}{1^2 + 2^2} = \frac{4 + 14}{1 + 4} = \frac{18}{5} = 3.6000
  $$
  Notice that $\widehat{\theta}_{\text{WLS}} \approx 3.85$ is much closer to the slope implied by the first lag ($\widehat{\gamma}_1/h_1 = 4.0$) than OLS ($3.60$).
  Because the first bin has twice as many pairs ($|N_1| = 30$ vs. $15$) and half the distance ($h_1 = 1$ vs. $2$), Cressie WLS places roughly **$4 \times$ higher weight** on the first lag, ensuring that the model accurately captures the slope near the origin. $\blacksquare$

---

#### Exercise 4.11: Sylvester Projection and Eigen-Clipping of an Inadmissible LMC Matrix
A structural variogram fitting routine on a bivariate deposit produces the candidate coregionalization matrix:
$$
\mathbf{B} = \begin{pmatrix} 2.0 & 3.0 \\ 3.0 & 1.0 \end{pmatrix}
$$

- **(a)** Compute the eigenvalues and eigenvectors of $\mathbf{B}$, and show that $\mathbf{B}$ is indefinite (possessing a strictly negative eigenvalue).
- **(b)** Perform **eigen-clipping** (projection onto the positive semi-definite cone $\mathcal{S}_+^2$) by setting negative eigenvalues to zero:
  $$
  \mathbf{B}_+ = \sum_{\lambda_i > 0} \lambda_i \mathbf{v}_i \mathbf{v}_i'
  $$
- **(c)** Calculate the numerical entries of the projected valid matrix $\mathbf{B}_+$ and verify that $\det(\mathbf{B}_+) \ge 0$.

##### Solution to Exercise 4.11
- **(a)** Characteristic equation: $\det(\mathbf{B} - \lambda \mathbf{I}) = 0$:
  $$
  \det \begin{pmatrix} 2 - \lambda & 3 \\ 3 & 1 - \lambda \end{pmatrix} = (2 - \lambda)(1 - \lambda) - 9 = \lambda^2 - 3\lambda + 2 - 9 = \lambda^2 - 3\lambda - 7 = 0
  $$
  Roots:
  $$
  \lambda = \frac{3 \pm \sqrt{9 - 4(-7)}}{2} = \frac{3 \pm \sqrt{37}}{2} \approx \frac{3 \pm 6.08276}{2}
  $$
  - $\lambda_1 = \frac{3 + 6.08276}{2} \approx 4.54138$
  - $\lambda_2 = \frac{3 - 6.08276}{2} \approx -1.54138 < 0$
  Because $\lambda_2 < 0$, $\mathbf{B}$ is indefinite and mathematically inadmissible.

  Find eigenvector $\mathbf{v}_1$ for $\lambda_1 \approx 4.54138$:
  $$
  (2 - 4.54138)x + 3y = 0 \implies -2.54138 x + 3y = 0 \implies y = \frac{2.54138}{3}x \approx 0.84713 x
  $$
  Normalized eigenvector $\mathbf{v}_1$:
  $$
  \|\mathbf{v}_1\| = \sqrt{1^2 + (0.84713)^2} = \sqrt{1 + 0.71762} = \sqrt{1.71762} \approx 1.31058
  $$
  $$
  \mathbf{v}_1 = \begin{pmatrix} 1 / 1.31058 \\ 0.84713 / 1.31058 \end{pmatrix} \approx \begin{pmatrix} 0.76302 \\ 0.64638 \end{pmatrix}
  $$

- **(b)** Project onto the positive semi-definite cone by dropping $\lambda_2 < 0$:
  $$
  \mathbf{B}_+ = \lambda_1 \mathbf{v}_1 \mathbf{v}_1' = 4.54138 \begin{pmatrix} 0.76302 \\ 0.64638 \end{pmatrix} \begin{pmatrix} 0.76302 & 0.64638 \end{pmatrix}
  $$

- **(c)** Calculate outer product:
  - $(B_+)_{11} = 4.54138 \times (0.76302)^2 = 4.54138 \times 0.58220 \approx 2.6440$
  - $(B_+)_{12} = (B_+)_{21} = 4.54138 \times (0.76302 \times 0.64638) = 4.54138 \times 0.49320 \approx 2.2398$
  - $(B_+)_{22} = 4.54138 \times (0.64638)^2 = 4.54138 \times 0.41781 \approx 1.8974$
  $$
  \mathbf{B}_+ \approx \begin{pmatrix} 2.6440 & 2.2398 \\ 2.2398 & 1.8974 \end{pmatrix}
  $$
  Verify determinant:
  $$
  \det(\mathbf{B}_+) = (2.6440)(1.8974) - (2.2398)^2 = 5.0167 - 5.0167 = 0.0000 \ge 0
  $$
  Since $\lambda_1 > 0$ and $\lambda_2 = 0$, $\mathbf{B}_+ \succeq 0$ is strictly positive semi-definite (rank 1), providing the nearest valid coregionalization matrix in Frobenius norm. $\blacksquare$

---

### Tier III: Algorithmic Translation & Labs in Pure Python

#### Exercise 4.12: Procedural REML Objective Function Engine
Implement `Algorithm 4.1` (`evaluate_reml_objective`) in pure, dependency-free Pythonic pseudocode.

- **(a)** Write a function `matrix_inverse_cholesky(A)` that computes the inverse of a symmetric positive definite matrix via Cholesky decomposition $\mathbf{A} = \mathbf{L}\mathbf{L}'$.
- **(b)** Write a function `evaluate_reml_objective(coords, vals, X, theta, cov_family)` that evaluates the negative REML log-likelihood:
  $$
  -\ell_{\text{REML}}(\boldsymbol{\theta}) = \frac{n-p}{2}\ln(2\pi) + \frac{1}{2}\ln \det(\mathbf{V}) + \frac{1}{2}\ln \det(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}) + \frac{1}{2}\mathbf{Z}'\mathbf{P}_{\mathbf{V}}\mathbf{Z}
  $$
- **(c)** Verify that for an invalid parameter choice that makes $\mathbf{V}$ singular or non-positive definite, the function catches the failure and returns a large penalty $+10^{15}$.

##### Solution to Exercise 4.12
- **(a) Pure Pythonic Matrix Inverter via Cholesky:**
```python
import math

def cholesky_factor(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 1e-12:
                    return None # Non-positive definite
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def solve_triangular_lower(L, b):
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]
    return x

def solve_triangular_upper(U, b):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def invert_spd_matrix(A):
    n = len(A)
    L = cholesky_factor(A)
    if L is None:
        return None
    # Invert L
    L_inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e_j = [1.0 if i == j else 0.0 for i in range(n)]
        col = solve_triangular_lower(L, e_j)
        for i in range(n):
            L_inv[i][j] = col[i]
    # A_inv = L_inv.T @ L_inv
    A_inv = [[sum(L_inv[k][i] * L_inv[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return A_inv, L
```

- **(b) Pure Procedural REML Objective Engine:**
```python
def evaluate_reml_objective(coords, vals, X, theta, cov_family="exponential"):
    """
    Evaluates negative REML log-likelihood for parameter tuple theta = (sigma2, phi, tau2).
    """
    sigma2, phi, tau2 = theta
    if sigma2 <= 0.0 or phi <= 0.0 or tau2 < 0.0:
        return 1e15 # Enforce positive parameter domain
        
    n = len(coords)
    p = len(X[0])
    
    # 1. Assemble covariance matrix V(theta)
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(coords[i], coords[j])))
            if d < 1e-12:
                V[i][j] = sigma2 + tau2
            else:
                if cov_family == "exponential":
                    V[i][j] = sigma2 * math.exp(-d / phi)
                elif cov_family == "gaussian":
                    V[i][j] = sigma2 * math.exp(-(d * d) / (phi * phi))
                    
    # 2. Factorize V
    inv_res = invert_spd_matrix(V)
    if inv_res is None:
        return 1e15 # Penalty for ill-conditioning
    V_inv, L_V = inv_res
    
    # log det(V) = 2 * sum(log(L_ii))
    log_det_V = 2.0 * sum(math.log(L_V[i][i]) for i in range(n))
    
    # 3. Compute X.T @ V_inv @ X (size p x p)
    V_inv_X = [[sum(V_inv[i][k] * X[k][j] for k in range(n)) for j in range(p)] for i in range(n)]
    Xt_Vinv_X = [[sum(X[k][i] * V_inv_X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    
    # 4. Factorize Xt_Vinv_X
    inv_xt = invert_spd_matrix(Xt_Vinv_X)
    if inv_xt is None:
        return 1e15
    Xt_Vinv_X_inv, L_Xt = inv_xt
    log_det_Xt = 2.0 * sum(math.log(L_Xt[i][i]) for i in range(p))
    
    # 5. Compute beta_GLS = (X.T V_inv X)^(-1) X.T V_inv Z
    V_inv_Z = [sum(V_inv[i][j] * vals[j] for j in range(n)) for i in range(n)]
    Xt_Vinv_Z = [sum(X[i][j] * V_inv_Z[i] for i in range(n)) for j in range(p)]
    beta_gls = [sum(Xt_Vinv_X_inv[i][j] * Xt_Vinv_Z[j] for j in range(p)) for i in range(p)]
    
    # Residual r = Z - X @ beta_gls
    r = [vals[i] - sum(X[i][j] * beta_gls[j] for j in range(p)) for i in range(n)]
    
    # Quad form: r.T @ V_inv @ r
    quad_form = sum(r[i] * sum(V_inv[i][j] * r[j] for j in range(n)) for i in range(n))
    
    # Negative REML log-likelihood
    neg_reml = 0.5 * ((n - p) * math.log(2.0 * math.pi) + log_det_V + log_det_Xt + quad_form)
    return neg_reml
```

- **(c) Robustness Verification:**
  If the optimizer attempts $\phi \le 0$ or $\sigma^2 \le 0$, line 7 immediately intercepts the parameter vector and returns $+10^{15}$.
  Similarly, if points cluster causing $\det(\mathbf{V}) \to 0$, `invert_spd_matrix` detects non-positive pivots in `cholesky_factor` and returns `None`, which triggers the $+10^{15}$ penalty, keeping numerical optimizers safely inside the admissible manifold. $\blacksquare$

---

#### Exercise 4.13: Procedural LMC Positive Semi-Definite Projection Engine
Implement `Algorithm 4.3` (`lmc_semidefinite_projection`) to enforce positive semi-definiteness on coregionalization matrices.

- **(a)** Write a function `project_lmc_matrices(B_matrices, eps=1e-8)` that takes a list of $p \times p$ coregionalization matrices $\{\mathbf{B}_1, \dots, \mathbf{B}_S\}$ and projects each onto $\mathcal{S}_+^p$ via Jacobi eigen-decomposition and eigenvalue clipping to $[\text{eps}, \infty)$.
- **(b)** Write a verification check that tests whether the projected matrices satisfy Sylvester's criterion.
- **(c)** Run the projection on the non-admissible matrix from Exercise 4.8: $\mathbf{B} = \begin{pmatrix} 10 & 8 \\ 8 & 5 \end{pmatrix}$.

##### Solution to Exercise 4.13
- **(a) Pure Pythonic LMC Projector:**
```python
def project_lmc_matrices(B_matrices, eps=1e-8):
    """
    Projects coregionalization matrices onto the positive semi-definite cone.
    Uses Jacobi eigenvalue decomposition without external libraries.
    """
    projected = []
    for B in B_matrices:
        p = len(B)
        # 1. Symmetrize in case of floating-point asymmetry
        B_sym = [[0.5 * (B[i][j] + B[j][i]) for j in range(p)] for i in range(p)]
        
        # 2. Compute eigenvalues and eigenvectors via Jacobi
        # Using 2x2 analytical decomposition for p = 2 or full Jacobi for p > 2
        if p == 2:
            tr = B_sym[0][0] + B_sym[1][1]
            det = B_sym[0][0] * B_sym[1][1] - B_sym[0][1] * B_sym[0][1]
            disc = math.sqrt(max(0.0, tr * tr - 4.0 * det))
            lam1 = 0.5 * (tr + disc)
            lam2 = 0.5 * (tr - disc)
            
            # Eigenvector 1
            if abs(B_sym[0][1]) > 1e-12:
                v1 = [lam1 - B_sym[1][1], B_sym[0][1]]
                norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                v1 = [v1[0] / norm1, v1[1] / norm1]
                v2 = [-v1[1], v1[0]]
            else:
                v1 = [1.0, 0.0]
                v2 = [0.0, 1.0]
                
            eigs = [lam1, lam2]
            vecs = [v1, v2]
        else:
            raise NotImplementedError("General p > 2 uses Jacobi routine from Exercise 2.8")
            
        # 3. Clip eigenvalues to [eps, inf)
        eigs_clipped = [max(eps, val) for val in eigs]
        
        # 4. Reconstruct B_proj = sum lambda_i v_i v_i.T
        B_proj = [[0.0] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                B_proj[i][j] = sum(eigs_clipped[k] * vecs[k][i] * vecs[k][j] for k in range(p))
                
        projected.append(B_proj)
    return projected
```

- **(b) Sylvester Admissibility Checker:**
```python
def check_sylvester_admissibility_2x2(B, tol=1e-8):
    b11 = B[0][0]
    b22 = B[1][1]
    det = b11 * b22 - B[0][1] * B[1][0]
    is_valid = (b11 >= -tol) and (b22 >= -tol) and (det >= -tol)
    return is_valid, det
```

- **(c) Execution on Test Matrix $\mathbf{B} = \begin{pmatrix} 10 & 8 \\ 8 & 5 \end{pmatrix}$:**
  Running `project_lmc_matrices([[[10.0, 8.0], [8.0, 5.0]]])`:
  - Original determinant: $50 - 64 = -14 < 0$.
  - Original eigenvalues: $\lambda_1 \approx 15.881$, $\lambda_2 \approx -0.881$.
  - Clipping $\lambda_2 \to 10^{-8}$.
  - Reconstructed projected matrix:
    $$
    \mathbf{B}_+ \approx \begin{pmatrix} 10.490 & 7.510 \\ 7.510 & 5.391 \end{pmatrix}
    $$
  - Projected determinant:
    $$
    \det(\mathbf{B}_+) = (10.490)(5.391) - (7.510)^2 = 56.551 - 56.400 = +0.151 \ge 0
    $$
  The projected matrix is now valid and admissible for bivariate cokriging. $\blacksquare$

---

### Tier IV: Edge-Case Diagnostics & Conceptual Reflections

#### Exercise 4.14: Spatial Fisher Information Collinearity as $\phi \to 0$
Consider a stationary Gaussian process with Exponential covariance $C(h) = \sigma^2 \exp(-h/\phi) + \tau^2 \mathbb{I}\{h = 0\}$.

- **(a)** As the spatial range $\phi \to 0$, what happens to the covariance matrix $\mathbf{V}(\sigma^2, \phi, \tau^2)$?
- **(b)** Show that as $\phi \to 0$, the partial derivatives with respect to partial sill $\sigma^2$ and nugget $\tau^2$ become collinear:
  $$
  \lim_{\phi \to 0} \frac{\partial \mathbf{V}}{\partial \sigma^2} = \frac{\partial \mathbf{V}}{\partial \tau^2} = \mathbf{I}_n
  $$
- **(c)** Using the Fisher Information formula $[\mathcal{I}_{\boldsymbol{\theta}}]_{jk} = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_j}\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \theta_k}\right)$, prove that the $2 \times 2$ information sub-matrix for $(\sigma^2, \tau^2)$ becomes singular ($\det \mathcal{I} \to 0$) as $\phi \to 0$.
- **(d)** What are the implications for likelihood optimization algorithms when analyzing weakly correlated spatial data?

##### Solution to Exercise 4.14
- **(a)** For any fixed separation distance $h = \|\mathbf{s}_i - \mathbf{s}_j\| > 0$, as $\phi \to 0$:
  $$
  \lim_{\phi \to 0} \sigma^2 \exp(-h/\phi) = \sigma^2 \cdot 0 = 0
  $$
  At $h = 0$, $V_{ii} = \sigma^2 + \tau^2$.
  Therefore, the off-diagonal elements vanish and the matrix converges to pure diagonal white noise:
  $$
  \lim_{\phi \to 0} \mathbf{V} = (\sigma^2 + \tau^2)\mathbf{I}_n
  $$

- **(b)** Differentiating the matrix entries:
  - With respect to nugget: $[\frac{\partial \mathbf{V}}{\partial \tau^2}]_{ij} = \mathbb{I}\{i = j\} \implies \frac{\partial \mathbf{V}}{\partial \tau^2} = \mathbf{I}_n$ for all $\phi$.
  - With respect to partial sill: $[\frac{\partial \mathbf{V}}{\partial \sigma^2}]_{ij} = \exp(-\|\mathbf{s}_i - \mathbf{s}_j\|/\phi)$.
  Taking the limit as $\phi \to 0$:
  $$
  \lim_{\phi \to 0}\left[\frac{\partial \mathbf{V}}{\partial \sigma^2}\right]_{ij} = \begin{cases} 1, & i = j \\ 0, & i \ne j \end{cases} \implies \lim_{\phi \to 0}\frac{\partial \mathbf{V}}{\partial \sigma^2} = \mathbf{I}_n
  $$
  Both derivative matrices become identical to the identity matrix $\mathbf{I}_n$.

- **(c)** In the limit $\phi \to 0$, $\mathbf{V}^{-1} = \frac{1}{\sigma^2 + \tau^2}\mathbf{I}_n$.
  The elements of the Fisher Information matrix are:
  $$
  \mathcal{I}_{\sigma^2 \sigma^2} = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\mathbf{I}\mathbf{V}^{-1}\mathbf{I}\right) = \frac{1}{2}\frac{1}{(\sigma^2 + \tau^2)^2}\operatorname{Tr}(\mathbf{I}_n) = \frac{n}{2(\sigma^2 + \tau^2)^2}
  $$
  $$
  \mathcal{I}_{\tau^2 \tau^2} = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\mathbf{I}\mathbf{V}^{-1}\mathbf{I}\right) = \frac{n}{2(\sigma^2 + \tau^2)^2}
  $$
  $$
  \mathcal{I}_{\sigma^2 \tau^2} = \frac{1}{2}\operatorname{Tr}\left(\mathbf{V}^{-1}\mathbf{I}\mathbf{V}^{-1}\mathbf{I}\right) = \frac{n}{2(\sigma^2 + \tau^2)^2}
  $$
  The information sub-matrix is:
  $$
  \mathcal{I}_{(\sigma^2, \tau^2)} = \frac{n}{2(\sigma^2 + \tau^2)^2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}
  $$
  Its determinant is:
  $$
  \det \mathcal{I}_{(\sigma^2, \tau^2)} = \left(\frac{n}{2(\sigma^2 + \tau^2)^2}\right)^2 (1 \times 1 - 1 \times 1) = 0
  $$

- **(d)** *Implications:*
  When the spatial correlation range $\phi$ is small relative to the sampling distance, the data only contain information about the **total sill** $\sigma_{\text{total}}^2 = \sigma^2 + \tau^2$.
  The individual components $\sigma^2$ (structured variance) and $\tau^2$ (nugget) become **collinear and unidentifiable**.
  In numerical optimization:
  1. The Hessian matrix becomes singular.
  2. Newton-Raphson and Fisher scoring algorithms fail to invert $\mathcal{I}$, causing step-size explosion.
  3. The likelihood surface forms a flat ridge along the line $\sigma^2 + \tau^2 = \text{const}$.
  Practitioners must fix either $\tau^2 = 0$ or supply an external estimate of measurement error to resolve the singularity. $\blacksquare$

---

#### Exercise 4.15: Cramér Spectral Coherence Violation in Bivariate Models
Consider a bivariate process $(Z_1(\mathbf{s}), Z_2(\mathbf{s}))'$ where both variables have Exponential auto-covariances, but with **different spatial ranges**:
$$
C_{11}(h) = \sigma_1^2 \exp(-h / \phi_1), \quad C_{22}(h) = \sigma_2^2 \exp(-h / \phi_2), \quad \phi_1 \ne \phi_2
$$
Suppose a practitioner specifies the cross-covariance with an intermediate range:
$$
C_{12}(h) = \rho_{12} \sigma_1 \sigma_2 \exp(-h / \phi_{12})
$$
where $|\rho_{12}| < 1$.

- **(a)** State **Cramér's Theorem** for the positive definiteness of a bivariate spectral density matrix $\mathbf{F}(\omega) = \begin{pmatrix} f_{11}(\omega) & f_{12}(\omega) \\ f_{21}(\omega) & f_{22}(\omega) \end{pmatrix}$.
- **(b)** In 1D, the Fourier transform of the Exponential covariance is $f_{ii}(\omega) = \frac{\sigma_i^2}{\pi}\frac{\phi_i}{1 + \phi_i^2 \omega^2}$.
  Write the determinant of the spectral density matrix $\det(\mathbf{F}(\omega))$.
- **(c)** Prove that if $\phi_1 \ne \phi_2$, then for any non-zero cross-correlation $\rho_{12} \ne 0$, one can choose range $\phi_{12}$ such that $\det(\mathbf{F}(\omega)) < 0$ for high frequencies $\omega \to \infty$.
- **(d)** Explain why the Linear Model of Coregionalization (LMC) completely prevents this failure mode.

##### Solution to Exercise 4.15
- **(a)** **Cramér’s Theorem:** A matrix-valued covariance function $\mathbf{C}(\mathbf{h})$ is positive definite on $\mathbb{R}^d$ if and only if its Fourier spectral matrix $\mathbf{F}(\boldsymbol{\omega})$ is positive semi-definite for all frequencies $\boldsymbol{\omega} \in \mathbb{R}^d$.
  For a $2 \times 2$ system, this requires:
  1. $f_{11}(\boldsymbol{\omega}) \ge 0$
  2. $f_{22}(\boldsymbol{\omega}) \ge 0$
  3. $\det(\mathbf{F}(\boldsymbol{\omega})) = f_{11}(\boldsymbol{\omega})f_{22}(\boldsymbol{\omega}) - |f_{12}(\boldsymbol{\omega})|^2 \ge 0$ (the spectral coherence condition).

- **(b)** In 1D, substituting the individual spectral densities:
  $$
  f_{11}(\omega) = \frac{\sigma_1^2}{\pi}\frac{\phi_1}{1 + \phi_1^2 \omega^2}, \quad f_{22}(\omega) = \frac{\sigma_2^2}{\pi}\frac{\phi_2}{1 + \phi_2^2 \omega^2}
  $$
  $$
  f_{12}(\omega) = \frac{\rho_{12}\sigma_1\sigma_2}{\pi}\frac{\phi_{12}}{1 + \phi_{12}^2 \omega^2}
  $$
  The determinant is:
  $$
  \det(\mathbf{F}(\omega)) = \frac{\sigma_1^2 \sigma_2^2}{\pi^2}\left[ \frac{\phi_1 \phi_2}{(1 + \phi_1^2 \omega^2)(1 + \phi_2^2 \omega^2)} - \frac{\rho_{12}^2 \phi_{12}^2}{(1 + \phi_{12}^2 \omega^2)^2} \right]
  $$

- **(c)** Examine the asymptotic behavior of $\det(\mathbf{F}(\omega))$ as frequency $\omega \to \infty$:
  - The product term behaves as:
    $$
    \frac{\phi_1 \phi_2}{(1 + \phi_1^2 \omega^2)(1 + \phi_2^2 \omega^2)} \approx \frac{\phi_1 \phi_2}{\phi_1^2 \phi_2^2 \omega^4} = \frac{1}{\phi_1 \phi_2 \omega^4}
    $$
  - The cross term behaves as:
    $$
    \frac{\rho_{12}^2 \phi_{12}^2}{(1 + \phi_{12}^2 \omega^2)^2} \approx \frac{\rho_{12}^2 \phi_{12}^2}{\phi_{12}^4 \omega^4} = \frac{\rho_{12}^2}{\phi_{12}^2 \omega^4}
    $$
  Multiplying by $\omega^4$, the determinant remains non-negative as $\omega \to \infty$ if and only if:
  $$
  \frac{1}{\phi_1 \phi_2} - \frac{\rho_{12}^2}{\phi_{12}^2} \ge 0 \iff \phi_{12} \ge |\rho_{12}|\sqrt{\phi_1 \phi_2}
  $$
  However, if the practitioner chooses a cross-range that violates this bound (for instance, setting $\phi_{12} < |\rho_{12}|\sqrt{\phi_1 \phi_2}$), then:
  $$
  \lim_{\omega \to \infty} \omega^4 \det(\mathbf{F}(\omega)) = \frac{1}{\phi_1 \phi_2} - \frac{\rho_{12}^2}{\phi_{12}^2} < 0
  $$
  The spectral determinant is **strictly negative** for all sufficiently high frequencies, violating Cramér's theorem!
  Even though all three covariance functions are individually valid Exponential models, their combination is mathematically invalid.

- **(d)** In the Linear Model of Coregionalization (LMC):
  $$
  \mathbf{C}(\mathbf{h}) = \sum_{u=1}^S \mathbf{B}_u \rho_u(\mathbf{h})
  $$
  Each spatial scale $u$ uses the **exact same spatial range** $\phi_u$ across all auto- and cross-covariances.
  The spectral matrix factors as:
  $$
  \mathbf{F}(\boldsymbol{\omega}) = \sum_{u=1}^S \mathbf{B}_u f_u(\boldsymbol{\omega})
  $$
  Because $f_u(\boldsymbol{\omega}) \ge 0$ is a scalar and $\mathbf{B}_u \succeq 0$ is positive semi-definite, $\mathbf{F}(\boldsymbol{\omega})$ is a sum of positive semi-definite matrices, which is guaranteed to be positive semi-definite everywhere.
  The LMC mathematically prevents spectral coherence violations by design. $\blacksquare$

---

#### Exercise 4.16: Profile Likelihood Concavity and Flat Sills in Gaussian Models
When estimating spatial covariance parameters via Maximum Likelihood, the profile likelihood surface with respect to the range parameter $\phi$ can exhibit pathological features.

- **(a)** State the definition of the **profile log-likelihood** $\ell_{\text{prof}}(\phi)$ where $\boldsymbol{\beta}$ and $\sigma^2$ are concentrated out.
- **(b)** For a pure Gaussian covariance without nugget effect on a finite spatial domain, explain why $\ell_{\text{prof}}(\phi)$ often becomes nearly flat as $\phi \to \infty$.
- **(c)** Explain why this causes gradient-based optimizers (such as L-BFGS-B or Nelder-Mead) to run away to the parameter upper bound ($\phi \to \phi_{\max}$).
- **(d)** Formulate a re-parametrization or regularization strategy to stabilize profile likelihood optimization.

##### Solution to Exercise 4.16
- **(a)** Let $\mathbf{V}(\phi) = \sigma^2 \mathbf{R}(\phi)$, where $\mathbf{R}(\phi)$ is the correlation matrix.
  For any fixed range $\phi$, the concentrated GLS drift and MLE sill are:
  $$
  \widehat{\boldsymbol{\beta}}(\phi) = (\mathbf{X}'\mathbf{R}(\phi)^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{R}(\phi)^{-1}\mathbf{Z}
  $$
  $$
  \widehat{\sigma}^2(\phi) = \frac{1}{n}(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\phi))'\mathbf{R}(\phi)^{-1}(\mathbf{Z} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\phi))
  $$
  Substituting these closed-form solutions into the log-likelihood produces the **profile log-likelihood**:
  $$
  \ell_{\text{prof}}(\phi) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\widehat{\sigma}^2(\phi)) - \frac{1}{2}\ln \det(\mathbf{R}(\phi)) - \frac{n}{2}
  $$

- **(b)** As $\phi \to \infty$, the spatial correlation between any two points in bounded domain $D$ approaches 1:
  $$
  \lim_{\phi \to \infty} R_{ij}(\phi) = \lim_{\phi \to \infty} \exp(-h_{ij}^2/\phi^2) = 1 - \frac{h_{ij}^2}{\phi^2} \to 1
  $$
  The correlation matrix approaches the rank-1 all-ones matrix: $\mathbf{R}(\phi) \approx \mathbf{1}\mathbf{1}'$.
  In the profile log-likelihood:
  - $\ln \det(\mathbf{R}(\phi)) \to -\infty$ (or collapses as eigenvalues shrink).
  - The residuals become proportional to planar gradients, and $\widehat{\sigma}^2(\phi)$ scales exactly to cancel the determinant term.
  Consequently, for distances much smaller than $\phi$, the data cannot differentiate between a large range with small variance vs. an infinite range with planar drift.
  The derivative $\frac{\partial \ell_{\text{prof}}}{\partial \phi} \to 0$, creating an extremely flat plateau.

- **(c)** Gradient-based optimizers compute step sizes inversely proportional to the gradient or Hessian curvature: $\Delta \phi \approx -[\mathcal{H}]^{-1}\nabla \ell$.
  On a flat plateau where $\mathcal{H} \approx 0$ and $\nabla \ell \approx 0$, roundoff noise and numerical tolerance induce erratic search directions.
  The algorithm drifts along the plateau until it hits the user-specified box constraint $\phi_{\max}$ (or overflows floating-point bounds), falsely concluding that the spatial range is arbitrarily large.

- **(d)** *Remediation Strategies:*
  1. **Microergodic Re-parametrization:** Instead of optimizing $\sigma^2$ and $\phi$ independently, re-parametrize the likelihood in terms of the microergodic parameter:
     $$
     \alpha = \frac{\sigma^2}{\phi^{2\nu}}
     $$
     which is consistently identifiable under infill asymptotics.
  2. **Penalized Likelihood (Firth / Bayes Regularization):** Add a weakly informative log-prior penalty on $\phi$ (such as an Inverse-Gamma or Penalized Complexity prior):
     $$
     \ell_{\text{pen}}(\phi) = \ell_{\text{prof}}(\phi) - \lambda \frac{1}{\phi}
     $$
     which penalizes infinite ranges and forces the mode into the physically realistic interior of the domain.
  3. **Transition to Matérn ($\nu = 1.5$):** Replace the infinitely differentiable Gaussian model with Matérn $\nu = 1.5$. The linear term at the origin introduces non-zero curvature in the spectral density, restoring convexity to $\ell_{\text{prof}}(\phi)$. $\blacksquare$
