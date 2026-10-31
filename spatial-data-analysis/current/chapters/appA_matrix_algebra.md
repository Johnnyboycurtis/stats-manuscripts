# Appendix A: Matrix Algebra and Multivariate Normal Calculus

Spatial statistics is fundamentally a matrix discipline. A spatial model is a specification of a covariance matrix $\boldsymbol{\Sigma}$, a precision matrix $\mathbf{Q}$, or an interaction between the two. Computing a generalized least squares trend, predicting an unobserved value by kriging, testing for autocorrelation with Moran's $I$, fitting a spatial generalized linear mixed model by Laplace approximation, or scaling to large data with Vecchia's algorithm all reduce to operations on structured matrices.

This appendix provides a self-contained, mathematically rigorous reference for the matrix calculus and multivariate normal theory used throughout the book. Every major identity is derived step by step, with intermediate algebraic steps displayed in full so that no manipulation is left as an unexplained mystery.

---

## A.0 Review of the Five Tools of Chapter 0

Chapter 0 introduced five core tools that recur throughout the book. This section reviews each tool in depth, establishing the foundational formulas and connecting them directly to the warm-up exercises of Section 0.7.

### A.0.1 Tool 1: Linear Transformations of Random Vectors

Let $\mathbf{Y}$ be an $n$-dimensional random vector with mean vector $\boldsymbol{\mu} = \operatorname{E}[\mathbf{Y}]$ and $n \times n$ covariance matrix $\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{Y}) = \operatorname{E}[(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})']$. Let $\mathbf{B}$ be a fixed, non-random $m \times n$ matrix, and let $\mathbf{b}$ be a fixed $m$-dimensional vector.

The linear transformation $\mathbf{W} = \mathbf{B}\mathbf{Y} + \mathbf{b}$ has expectation:

$$
\operatorname{E}[\mathbf{W}] = \operatorname{E}[\mathbf{B}\mathbf{Y} + \mathbf{b}] = \mathbf{B}\operatorname{E}[\mathbf{Y}] + \mathbf{b} = \mathbf{B}\boldsymbol{\mu} + \mathbf{b}
$$

and covariance matrix:

$$
\begin{aligned}
\operatorname{Var}(\mathbf{W}) &= \operatorname{E}\!\left[(\mathbf{W} - \operatorname{E}[\mathbf{W}])(\mathbf{W} - \operatorname{E}[\mathbf{W}])'\right] \\
&= \operatorname{E}\!\left[(\mathbf{B}(\mathbf{Y} - \boldsymbol{\mu}))(\mathbf{B}(\mathbf{Y} - \boldsymbol{\mu}))'\right] \\
&= \operatorname{E}\!\left[\mathbf{B}(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})'\mathbf{B}'\right] \\
&= \mathbf{B}\operatorname{E}\!\left[(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})'\right]\mathbf{B}' \\
&= \mathbf{B}\boldsymbol{\Sigma}\mathbf{B}'
\end{aligned}
$$

**Linear combinations.** When $\mathbf{B}$ is a single row vector $\mathbf{a}'$, the transformation is a scalar linear combination $W = \mathbf{a}'\mathbf{Y} = \sum_{i=1}^n a_i Y_i$. Its variance is the quadratic form:

$$
\operatorname{Var}(\mathbf{a}'\mathbf{Y}) = \mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} = \sum_{i=1}^n \sum_{j=1}^n a_i a_j \Sigma_{ij}
$$

Setting $\mathbf{a} = \frac{1}{n}\mathbf{1}_n$ gives the variance of the sample mean $\bar{Y} = \frac{1}{n}\sum_{i=1}^n Y_i$:

$$
\operatorname{Var}(\bar{Y}) = \left(\frac{1}{n}\mathbf{1}_n'\right)\boldsymbol{\Sigma}\left(\frac{1}{n}\mathbf{1}_n\right) = \frac{1}{n^2}\mathbf{1}_n'\boldsymbol{\Sigma}\mathbf{1}_n = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \Sigma_{ij}
$$

When observations are independent with variance $\sigma^2$, $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}_n$ and $\operatorname{Var}(\bar{Y}) = \frac{\sigma^2}{n}$. When observations share a common correlation $r > 0$ across all pairs, the off-diagonal entries contribute $n(n-1)r\sigma^2$, yielding:

$$
\operatorname{Var}(\bar{Y}) = \frac{\sigma^2 [n + n(n-1)r]}{n^2} = \frac{\sigma^2 [1 + (n-1)r]}{n}
$$

As $n \to \infty$, this variance approaches $r\sigma^2 > 0$, demonstrating that no finite or infinite number of equicorrelated observations can ever reduce the estimation variance of the mean to zero (Section 0.7, Exercise 2).

**Field simulation via matrix factors.** Let $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ be a vector of independent standard normal random variables. If $\mathbf{L}$ is any $n \times n$ matrix satisfying $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$, then by Tool 1:

$$
\operatorname{Var}(\boldsymbol{\mu} + \mathbf{L}\mathbf{Z}) = \mathbf{L}\operatorname{Var}(\mathbf{Z})\mathbf{L}' = \mathbf{L}\mathbf{I}_n\mathbf{L}' = \mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}
$$

Choosing $\mathbf{L}$ to be the unique lower-triangular Cholesky factor allows exact simulation of arbitrary Gaussian random fields in $\mathcal{O}(n^3)$ operations (§0.3.2, §2.7.1).

---

### A.0.2 Tool 2: Quadratic Forms and Positive Semi-Definiteness

A quadratic form in $\mathbf{y} \in \mathbb{R}^n$ associated with an $n \times n$ matrix $\mathbf{M}$ is the scalar function:

$$
Q(\mathbf{y}) = \mathbf{y}'\mathbf{M}\mathbf{y} = \sum_{i=1}^n\sum_{j=1}^n M_{ij}y_i y_j
$$

Without loss of generality, $\mathbf{M}$ can always be assumed symmetric. If $\mathbf{M}$ is not symmetric, $\mathbf{y}'\mathbf{M}\mathbf{y} = \mathbf{y}'\left(\frac{\mathbf{M} + \mathbf{M}'}{2}\right)\mathbf{y}$, because the skew-symmetric component $\frac{\mathbf{M} - \mathbf{M}'}{2}$ evaluates to zero for all $\mathbf{y}$.

**Centering and residuals.** Common regression quantities are quadratic forms:
- The centered sum of squares is $\sum_{i=1}^n (y_i - \bar{y})^2 = \mathbf{y}'\mathbf{P}_c\mathbf{y}$, where $\mathbf{P}_c = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n\mathbf{1}_n'$ is the orthogonal centering projection matrix.
- The ordinary least squares residual sum of squares is $\mathbf{e}'\mathbf{e} = \mathbf{y}'(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{y}$, where $\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$.

**Positive semi-definiteness as non-negative variance.** A symmetric matrix $\boldsymbol{\Sigma}$ is positive semi-definite (written $\boldsymbol{\Sigma} \succeq \mathbf{0}$) if:

$$
\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} \ge 0 \quad \text{for all } \mathbf{a} \in \mathbb{R}^n
$$

and strictly positive definite ($\boldsymbol{\Sigma} \succ \mathbf{0}$) if $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} > 0$ for all $\mathbf{a} \ne \mathbf{0}$.

This is not a technical mathematical assumption; it is an inescapable consequence of probability theory. Because $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} = \operatorname{Var}(\mathbf{a}'\mathbf{Y})$, and variances cannot be negative, every valid covariance matrix must be positive semi-definite. If a candidate function yields a negative eigenvalue $\lambda_k < 0$ with eigenvector $\mathbf{v}_k$, the linear combination $W = \mathbf{v}_k'\mathbf{Y}$ would have variance $\operatorname{Var}(W) = \mathbf{v}_k'\boldsymbol{\Sigma}\mathbf{v}_k = \lambda_k < 0$, which is physically impossible (§0.4.2, §1.6.4).

---

### A.0.3 Tool 3: The Multivariate Normal Density

Let $\mathbf{Y} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ be an $n$-dimensional Gaussian vector with positive definite covariance matrix $\boldsymbol{\Sigma}$. Its probability density function is:

$$
f(\mathbf{y}) = (2\pi)^{-n/2}\lvert\boldsymbol{\Sigma}\rvert^{-1/2}\exp\!\left(-\frac{1}{2}(\mathbf{y} - \boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \boldsymbol{\mu})\right)
$$

The log-likelihood of $\mathbf{y}$ decomposes into two competing terms:

$$
\log f(\mathbf{y}) = -\frac{n}{2}\log(2\pi) \;-\; \underbrace{\frac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert}_{\text{determinant penalty}} \;-\; \underbrace{\frac{1}{2}(\mathbf{y} - \boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \boldsymbol{\mu})}_{\text{Mahalanobis distance penalty}}
$$

1. **The determinant penalty** $-\frac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert$ penalizes models that allow excessive variability. The generalized variance $\lvert\boldsymbol{\Sigma}\rvert$ measures the squared volume of the ellipsoid of uncertainty.
2. **The Mahalanobis distance** $(\mathbf{y} - \boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \boldsymbol{\mu})$ measures the departure of the observed data from the mean surface in units standardized by correlation.

**Bivariate intuition.** For $n = 2$ with $\boldsymbol{\mu} = \mathbf{0}$, $\operatorname{Var}(Y_1) = \operatorname{Var}(Y_2) = \sigma^2$, and correlation $r$:

$$
\boldsymbol{\Sigma} = \sigma^2\begin{bmatrix} 1 & r \\ r & 1 \end{bmatrix}, \qquad \lvert\boldsymbol{\Sigma}\rvert = \sigma^4(1 - r^2), \qquad \boldsymbol{\Sigma}^{-1} = \frac{1}{\sigma^2(1 - r^2)}\begin{bmatrix} 1 & -r \\ -r & 1 \end{bmatrix}
$$

The Mahalanobis quadratic form expands to:

$$
\mathbf{y}'\boldsymbol{\Sigma}^{-1}\mathbf{y} = \frac{y_1^2 - 2r y_1 y_2 + y_2^2}{\sigma^2(1 - r^2)} = \frac{(y_1 - y_2)^2}{2\sigma^2(1 - r)} + \frac{(y_1 + y_2)^2}{2\sigma^2(1 + r)}
$$

When $r \to 1$, the first term heavily penalizes any discrepancy between neighbors ($y_1 \ne y_2$), while the second term tolerates common shifts ($y_1 \approx y_2$). A spatial model rewards configurations whose smoothness matches the correlation function (§0.4.3).

---

### A.0.4 Tool 4: Maximum Likelihood by Score

Let $\ell(\boldsymbol{\theta}) = \log L(\boldsymbol{\theta})$ be the log-likelihood function of parameter vector $\boldsymbol{\theta}$.
- The **score vector** is the gradient $\mathbf{s}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta})$. Maximum likelihood estimates $\hat{\boldsymbol{\theta}}$ satisfy the score equations $\mathbf{s}(\hat{\boldsymbol{\theta}}) = \mathbf{0}$.
- The **observed information matrix** is the negative Hessian $\mathbf{J}(\boldsymbol{\theta}) = -\nabla_{\boldsymbol{\theta}}^2\ell(\boldsymbol{\theta})$.
- The **expected (Fisher) information matrix** is $\mathcal{I}(\boldsymbol{\theta}) = \operatorname{E}[\mathbf{J}(\boldsymbol{\theta})] = \operatorname{Var}(\mathbf{s}(\boldsymbol{\theta}))$.

**Concentrated scale parameter.** Suppose $\mathbf{Y} \sim N(\mathbf{0}, \sigma^2\mathbf{R})$ where correlation matrix $\mathbf{R}$ is known. Differentiating with respect to $\sigma^2$:

$$
\ell(\sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2}\log\lvert\mathbf{R}\rvert - \frac{\mathbf{y}'\mathbf{R}^{-1}\mathbf{y}}{2\sigma^2}
$$

$$
\frac{\partial\ell}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{\mathbf{y}'\mathbf{R}^{-1}\mathbf{y}}{2\sigma^4} = 0 \quad\Longrightarrow\quad \hat{\sigma}^2 = \frac{\mathbf{y}'\mathbf{R}^{-1}\mathbf{y}}{n}
$$

Plugging $\hat{\sigma}^2$ back into $\ell$ yields the *profile (concentrated) log-likelihood*, eliminating $\sigma^2$ from numerical optimization (§0.4.4, §5.2).

---

### A.0.5 Tool 5: The Normal Equations and Projections

In linear regression $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$, Ordinary Least Squares (OLS) minimizes the Euclidean sum of squared residuals $S(\boldsymbol{\beta}) = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$. Setting the gradient $-2\mathbf{X}'(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$ to zero gives the classical normal equations:

$$
\mathbf{X}'\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\mathbf{Y} \quad\Longrightarrow\quad \hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}
$$

Fitted values are the orthogonal projection $\hat{\mathbf{Y}} = \mathbf{P}_{\mathbf{X}}\mathbf{Y}$, with projection matrix $\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$, satisfying $\mathbf{P}_{\mathbf{X}}^2 = \mathbf{P}_{\mathbf{X}}$ and $\mathbf{P}_{\mathbf{X}}' = \mathbf{P}_{\mathbf{X}}$.

**Generalized Least Squares (GLS).** When $\operatorname{Var}(\boldsymbol{\varepsilon}) = \boldsymbol{\Sigma}$, Gauss–Markov theory requires minimizing the Mahalanobis distance $(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$. Differentiating:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) = -2\mathbf{X}'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}
$$

which yields the generalized normal equations:

$$
\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}} = \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y} \quad\Longrightarrow\quad \hat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y}
$$

with variance-covariance matrix $\operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}}) = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}$ (§0.4.5, §1.3.5).

---

## A.1 Partitioned Matrices, the Schur Complement, and Inversion Identities

Partitioned matrices arise whenever data are divided into subsets, such as observed versus unobserved sites in kriging (§4.3.4), or fixed effects versus spatial random effects in spatial mixed models (§9.4.2).

### A.1.1 Block Elimination and the Schur Complement

Consider an invertible square matrix partitioned into four blocks:

$$
\mathbf{M} = \begin{bmatrix} \mathbf{M}_{11} & \mathbf{M}_{12} \\ \mathbf{M}_{21} & \mathbf{M}_{22} \end{bmatrix}
$$

where $\mathbf{M}_{11}$ is $p \times p$, $\mathbf{M}_{22}$ is $q \times q$, and $\mathbf{M}_{11}$ is non-singular.

We eliminate the lower-left block $\mathbf{M}_{21}$ using block Gaussian elimination. Multiplying $\mathbf{M}$ on the left by an elementary lower block-triangular matrix:

$$
\begin{bmatrix} \mathbf{I}_p & \mathbf{0} \\ -\mathbf{M}_{21}\mathbf{M}_{11}^{-1} & \mathbf{I}_q \end{bmatrix} \begin{bmatrix} \mathbf{M}_{11} & \mathbf{M}_{12} \\ \mathbf{M}_{21} & \mathbf{M}_{22} \end{bmatrix} = \begin{bmatrix} \mathbf{M}_{11} & \mathbf{M}_{12} \\ \mathbf{0} & \mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12} \end{bmatrix}
$$

The matrix remaining in the lower-right block is the **Schur complement** of $\mathbf{M}_{11}$ in $\mathbf{M}$:

$$
\mathbf{F} = \mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}
$$

Eliminating the upper-right block $\mathbf{M}_{12}$ by multiplying on the right by an upper block-triangular matrix yields the complete block LDU factorization of $\mathbf{M}$:

$$
\mathbf{M} = \begin{bmatrix} \mathbf{I}_p & \mathbf{0} \\ \mathbf{M}_{21}\mathbf{M}_{11}^{-1} & \mathbf{I}_q \end{bmatrix} \begin{bmatrix} \mathbf{M}_{11} & \mathbf{0} \\ \mathbf{0} & \mathbf{F} \end{bmatrix} \begin{bmatrix} \mathbf{I}_p & \mathbf{M}_{11}^{-1}\mathbf{M}_{12} \\ \mathbf{0} & \mathbf{I}_q \end{bmatrix}
$$

---

### A.1.2 Full Verification of the Partitioned Inverse

Inverting each factor in the LDU decomposition:

$$
\mathbf{M}^{-1} = \begin{bmatrix} \mathbf{I}_p & -\mathbf{M}_{11}^{-1}\mathbf{M}_{12} \\ \mathbf{0} & \mathbf{I}_q \end{bmatrix} \begin{bmatrix} \mathbf{M}_{11}^{-1} & \mathbf{0} \\ \mathbf{0} & \mathbf{F}^{-1} \end{bmatrix} \begin{bmatrix} \mathbf{I}_p & \mathbf{0} \\ -\mathbf{M}_{21}\mathbf{M}_{11}^{-1} & \mathbf{I}_q \end{bmatrix}
$$

Carrying out the two matrix multiplications step by step:

$$
\begin{aligned}
\mathbf{M}^{-1} &= \begin{bmatrix} \mathbf{M}_{11}^{-1} & -\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1} \\ \mathbf{0} & \mathbf{F}^{-1} \end{bmatrix} \begin{bmatrix} \mathbf{I}_p & \mathbf{0} \\ -\mathbf{M}_{21}\mathbf{M}_{11}^{-1} & \mathbf{I}_q \end{bmatrix} \\
&= \begin{bmatrix} \mathbf{M}_{11}^{-1} + \mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} & -\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1} \\ -\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} & \mathbf{F}^{-1} \end{bmatrix}
\end{aligned}
$$

**Direct verification by multiplication.** We multiply $\mathbf{M}\mathbf{M}^{-1}$ block by block to verify that it produces the identity matrix $\mathbf{I}_{p+q}$:

1. **Upper-left block:**
   $$
   \mathbf{M}_{11}\left(\mathbf{M}_{11}^{-1} + \mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1}\right) + \mathbf{M}_{12}\left(-\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1}\right) = \mathbf{I}_p + \mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} - \mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} = \mathbf{I}_p
   $$
2. **Upper-right block:**
   $$
   \mathbf{M}_{11}\left(-\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\right) + \mathbf{M}_{12}\mathbf{F}^{-1} = -\mathbf{M}_{12}\mathbf{F}^{-1} + \mathbf{M}_{12}\mathbf{F}^{-1} = \mathbf{0}
   $$
3. **Lower-left block:**
   $$
   \begin{aligned}
   \mathbf{M}_{21}\left(\mathbf{M}_{11}^{-1} + \mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1}\right) + \mathbf{M}_{22}\left(-\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1}\right) &= \left(\mathbf{M}_{21}\mathbf{M}_{11}^{-1} + \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1}\right) - \mathbf{M}_{22}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} \\
   &= \mathbf{M}_{21}\mathbf{M}_{11}^{-1} - \left(\mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\right)\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} \\
   &= \mathbf{M}_{21}\mathbf{M}_{11}^{-1} - \mathbf{F}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} = \mathbf{0}
   \end{aligned}
   $$
4. **Lower-right block:**
   $$
   \mathbf{M}_{21}\left(-\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\right) + \mathbf{M}_{22}\mathbf{F}^{-1} = \left(\mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\right)\mathbf{F}^{-1} = \mathbf{F}\mathbf{F}^{-1} = \mathbf{I}_q
   $$

This verifies the formula in full (§4.3.4).

**Both forms of the upper-left block.** Swapping the roles of blocks 1 and 2 in the derivation reveals that when $\mathbf{M}_{22}$ is invertible, the upper-left block of $\mathbf{M}^{-1}$ can be written in two mathematically identical forms:

$$
(\mathbf{M}^{-1})_{11} = \mathbf{M}_{11}^{-1} + \mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} = \left(\mathbf{M}_{11} - \mathbf{M}_{12}\mathbf{M}_{22}^{-1}\mathbf{M}_{21}\right)^{-1}
$$

---

### A.1.3 The Block Determinant Formula

Taking the determinant of both sides of the block LDU decomposition:

$$
\lvert\mathbf{M}\rvert = \left|\begin{bmatrix} \mathbf{I}_p & \mathbf{0} \\ \mathbf{M}_{21}\mathbf{M}_{11}^{-1} & \mathbf{I}_q \end{bmatrix}\right| \cdot \left|\begin{bmatrix} \mathbf{M}_{11} & \mathbf{0} \\ \mathbf{0} & \mathbf{F} \end{bmatrix}\right| \cdot \left|\begin{bmatrix} \mathbf{I}_p & \mathbf{M}_{11}^{-1}\mathbf{M}_{12} \\ \mathbf{0} & \mathbf{I}_q \end{bmatrix}\right|
$$

Because unit triangular matrices have determinant 1, and block-diagonal matrices have determinant equal to the product of the block determinants:

$$
\lvert\mathbf{M}\rvert = \lvert\mathbf{M}_{11}\rvert \cdot \lvert\mathbf{F}\rvert = \lvert\mathbf{M}_{11}\rvert \cdot \lvert\mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\rvert
$$

Furthermore, because $\mathbf{F}^{-1} = (\mathbf{M}^{-1})_{22}$, we have $\lvert\mathbf{F}\rvert = 1/\lvert(\mathbf{M}^{-1})_{22}\rvert$, which establishes the reciprocal determinant formula used in Chapter 5 (§5.3.4):

$$
\lvert\mathbf{M}\rvert = \frac{\lvert\mathbf{M}_{11}\rvert}{\lvert(\mathbf{M}^{-1})_{22}\rvert}
$$

---

### A.1.4 The Sherman–Morrison–Woodbury Identity

The Woodbury identity expresses the inverse of a matrix subjected to a low-rank or structured update $(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1}$ in terms of inverses of the smaller core matrices $\mathbf{A}$ and $\mathbf{C}$.

Consider the partitioned matrix:

$$
\mathbf{N} = \begin{bmatrix} \mathbf{A} & \mathbf{U} \\ -\mathbf{V} & \mathbf{C}^{-1} \end{bmatrix}
$$

(Notice the negative sign on $\mathbf{V}$, which is an algebraic device ensuring that the Schur complement subtraction turns into an addition, §13.2.2).

We compute the upper-left block $(\mathbf{N}^{-1})_{11}$ in the two equivalent forms of Section A.1.2:
1. Using the Schur complement of $\mathbf{C}^{-1}$, which is $\mathbf{A} - \mathbf{U}(\mathbf{C}^{-1})^{-1}(-\mathbf{V}) = \mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V}$:
   $$
   (\mathbf{N}^{-1})_{11} = (\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1}
   $$
2. Using the Schur complement of $\mathbf{A}$, which is $\mathbf{F} = \mathbf{C}^{-1} - (-\mathbf{V})\mathbf{A}^{-1}\mathbf{U} = \mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}$:
   $$
   (\mathbf{N}^{-1})_{11} = \mathbf{A}^{-1} + \mathbf{A}^{-1}\mathbf{U}\mathbf{F}^{-1}(-\mathbf{V})\mathbf{A}^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}
   $$

Equating the two expressions yields the **Sherman–Morrison–Woodbury identity**:

$$
(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}
$$

When $\mathbf{u}, \mathbf{v}$ are vectors and $c = 1$, this simplifies to the rank-one **Sherman–Morrison formula**:

$$
(\mathbf{A} + \mathbf{u}\mathbf{v}')^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}'\mathbf{A}^{-1}}{1 + \mathbf{v}'\mathbf{A}^{-1}\mathbf{u}}
$$

**The Matrix Determinant Lemma.** Applying the block determinant formula to $\mathbf{N}$ in both directions:
1. Expanding along $\mathbf{A}$: $\lvert\mathbf{N}\rvert = \lvert\mathbf{A}\rvert \cdot \lvert\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\rvert$.
2. Expanding along $\mathbf{C}^{-1}$: $\lvert\mathbf{N}\rvert = \lvert\mathbf{C}^{-1}\rvert \cdot \lvert\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V}\rvert$.

Equating the two expressions and multiplying by $\lvert\mathbf{C}\rvert$:

$$
\lvert\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V}\rvert = \lvert\mathbf{A}\rvert \cdot \lvert\mathbf{C}\rvert \cdot \lvert\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\rvert
$$

For a rank-one update $\mathbf{A} + \mathbf{u}\mathbf{v}'$, this yields $\lvert\mathbf{A} + \mathbf{u}\mathbf{v}'\rvert = \lvert\mathbf{A}\rvert(1 + \mathbf{v}'\mathbf{A}^{-1}\mathbf{u})$ (§13.2.2).

**The Henderson / Mixed-Model Form.** In spatial GLMMs, Henderson's equations require inverting $\mathbf{V} - \mathbf{V}(\mathbf{V} + \mathbf{Q})^{-1}\mathbf{V}$ where $\mathbf{V}$ is a diagonal matrix of Poisson working variances and $\mathbf{Q}$ is a spatial precision matrix. Applying the Woodbury identity with $\mathbf{A} = \mathbf{V}$, $\mathbf{U} = \mathbf{I}$, $\mathbf{C} = -\mathbf{Q}^{-1}$, and $\mathbf{V} = \mathbf{I}$:

$$
\mathbf{V} - \mathbf{V}(\mathbf{V} + \mathbf{Q})^{-1}\mathbf{V} = (\mathbf{V}^{-1} + \mathbf{Q}^{-1})^{-1}
$$

which directly proves the equivalence between penalized mixed-model back-fitting and generalized least squares (§9.4.1, Chapter 9 Exercise 8).

---

### A.1.5 The LOOCV Projection Matrix $\Pi$

In Chapter 4 (§4.7.4), the cross-validation shortcut computes the leave-one-out prediction error without refitting the model. This shortcut rests on the residual projection matrix:

$$
\boldsymbol{\Pi} = \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}
$$

Consider the partitioned generalized least squares normal system:

$$
\mathbf{K} = \begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix}
$$

The Schur complement of $\boldsymbol{\Sigma}$ is $\mathbf{F} = \mathbf{0} - \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X} = -\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}$. Applying the partitioned inverse formula:

$$
(\mathbf{K}^{-1})_{11} = \boldsymbol{\Sigma}^{-1} + \boldsymbol{\Sigma}^{-1}\mathbf{X}\left(-\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Pi}
$$

Because $\boldsymbol{\Pi}$ is the upper-left block of $\mathbf{K}^{-1}$, partitioning $\mathbf{K}$ into observation $i$ and all remaining observations $-i$ shows by block Gaussian elimination that:

$$
e_{-i} = y_i - \hat{y}_{-i} = -\frac{[\boldsymbol{\Pi}\mathbf{y}]_i}{\Pi_{ii}}, \qquad \operatorname{Var}(Y_i - \hat{Y}_{-i}) = \frac{1}{\Pi_{ii}}
$$

Thus, all $n$ leave-one-out residuals and kriging standard errors are obtained simultaneously from a single solve (§4.7.2).

---

## A.2 Vector and Matrix Derivatives

Maximum likelihood estimation and Fisher information scoring require differentiating scalar objective functions with respect to vectors and matrices.

### A.2.1 Gradients of Linear and Quadratic Forms

Let $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{c} \in \mathbb{R}^n$, and let $\mathbf{A}$ be an $n \times n$ matrix.

1. **Linear form $\mathbf{c}'\mathbf{x}$:**
   $$
   \frac{\partial}{\partial\mathbf{x}}(\mathbf{c}'\mathbf{x}) = \frac{\partial}{\partial\mathbf{x}}\left(\sum_{j=1}^n c_j x_j\right) = \begin{bmatrix} c_1 \\ \vdots \\ c_n \end{bmatrix} = \mathbf{c}
   $$
2. **Quadratic form $\mathbf{x}'\mathbf{A}\mathbf{x}$:**
   Expanding $\mathbf{x}'\mathbf{A}\mathbf{x} = \sum_{j=1}^n\sum_{k=1}^n A_{jk}x_j x_k$ and differentiating with respect to $x_i$:
   $$
   \frac{\partial}{\partial x_i}\sum_{j=1}^n\sum_{k=1}^n A_{jk}x_j x_k = \sum_{k=1}^n A_{ik}x_k + \sum_{j=1}^n A_{ji}x_j = [\mathbf{A}\mathbf{x}]_i + [\mathbf{A}'\mathbf{x}]_i
   $$
   Assembling into a vector:
   $$
   \frac{\partial}{\partial\mathbf{x}}(\mathbf{x}'\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}')\mathbf{x}
   $$
   When $\mathbf{A}$ is symmetric ($\mathbf{A} = \mathbf{A}'$):
   $$
   \frac{\partial}{\partial\mathbf{x}}(\mathbf{x}'\mathbf{A}\mathbf{x}) = 2\mathbf{A}\mathbf{x}
   $$
   which is the vector analog of $\frac{d}{dx}(ax^2) = 2ax$ (§4.2.1).

---

### A.2.2 Derivative of a Matrix Inverse

Let $\boldsymbol{\Sigma}(\theta)$ be a matrix whose entries depend smoothly on a scalar parameter $\theta$. Differentiating both sides of the identity $\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1} = \mathbf{I}$ using the matrix product rule:

$$
\frac{\partial\boldsymbol{\Sigma}}{\partial\theta}\boldsymbol{\Sigma}^{-1} + \boldsymbol{\Sigma}\frac{\partial\boldsymbol{\Sigma}^{-1}}{\partial\theta} = \mathbf{0}
$$

Multiplying on the left by $\boldsymbol{\Sigma}^{-1}$ and rearranging yields:

$$
\frac{\partial\boldsymbol{\Sigma}^{-1}}{\partial\theta} = -\boldsymbol{\Sigma}^{-1}\frac{\partial\boldsymbol{\Sigma}}{\partial\theta}\boldsymbol{\Sigma}^{-1}
$$

which is the matrix counterpart of $\frac{d}{d\theta}(x^{-1}) = -x^{-2}\frac{dx}{d\theta}$ (§5.4.2).

---

### A.2.3 Derivative of a Log-Determinant (Jacobi's Formula)

Let $\boldsymbol{\Sigma}(\theta)$ be symmetric and positive definite. Consider a perturbation of $\theta$ by a small scalar $t$. By Taylor expansion:

$$
\boldsymbol{\Sigma}(\theta + t) = \boldsymbol{\Sigma} + t\dot{\boldsymbol{\Sigma}} + \mathcal{O}(t^2) = \boldsymbol{\Sigma}\left(\mathbf{I} + t\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}\right) + \mathcal{O}(t^2)
$$

where $\dot{\boldsymbol{\Sigma}} = \partial\boldsymbol{\Sigma}/\partial\theta$.

Taking the determinant:

$$
\lvert\boldsymbol{\Sigma}(\theta + t)\rvert = \lvert\boldsymbol{\Sigma}\rvert \cdot \lvert\mathbf{I} + t\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}\rvert + \mathcal{O}(t^2)
$$

Let $\lambda_1, \dots, \lambda_n$ be the eigenvalues of $\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}$. The eigenvalues of $\mathbf{I} + t\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}$ are $1 + t\lambda_i$. Therefore:

$$
\lvert\mathbf{I} + t\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}\rvert = \prod_{i=1}^n (1 + t\lambda_i) = 1 + t\sum_{i=1}^n \lambda_i + \mathcal{O}(t^2) = 1 + t\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}) + \mathcal{O}(t^2)
$$

Taking natural logarithms and using $\log(1 + u) = u + \mathcal{O}(u^2)$:

$$
\log\lvert\boldsymbol{\Sigma}(\theta + t)\rvert = \log\lvert\boldsymbol{\Sigma}\rvert + t\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}) + \mathcal{O}(t^2)
$$

Subtracting $\log\lvert\boldsymbol{\Sigma}\rvert$, dividing by $t$, and taking the limit $t \to 0$ gives **Jacobi's formula**:

$$
\frac{\partial\log\lvert\boldsymbol{\Sigma}\rvert}{\partial\theta} = \operatorname{tr}\!\left(\boldsymbol{\Sigma}^{-1}\frac{\partial\boldsymbol{\Sigma}}{\partial\theta}\right)
$$

---

### A.2.4 Derivatives of Projections and Fisher Information

**Derivative of $\boldsymbol{\Pi}$.** Recall $\boldsymbol{\Pi} = \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}$. Using the product rule and $\partial\boldsymbol{\Sigma}^{-1}/\partial\theta_j = -\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}_j\boldsymbol{\Sigma}^{-1}$:

$$
\frac{\partial\boldsymbol{\Pi}}{\partial\theta_j} = -\boldsymbol{\Pi}\dot{\boldsymbol{\Sigma}}_j\boldsymbol{\Pi}
$$

**The Expected Fisher Information in REML.** In REML estimation (§5.4.6), the profile log-likelihood is:

$$
\ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert - \frac{1}{2}\log\lvert\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\rvert - \frac{1}{2}\mathbf{Y}'\boldsymbol{\Pi}\mathbf{Y}
$$

The score is:

$$
\frac{\partial\ell_{\text{REML}}}{\partial\theta_j} = -\frac{1}{2}\operatorname{tr}(\boldsymbol{\Pi}\dot{\boldsymbol{\Sigma}}_j) + \frac{1}{2}\mathbf{Y}'\boldsymbol{\Pi}\dot{\boldsymbol{\Sigma}}_j\boldsymbol{\Pi}\mathbf{Y}
$$

Differentiating a second time involves second derivatives $\ddot{\boldsymbol{\Sigma}}_{jk} = \partial^2\boldsymbol{\Sigma}/(\partial\theta_j\partial\theta_k)$. However, when taking expectations to compute the Fisher information matrix $\mathcal{I}(\boldsymbol{\theta}) = -\operatorname{E}[\nabla^2\ell_{\text{REML}}]$:

$$
\operatorname{E}\!\left[\mathbf{Y}'\boldsymbol{\Pi}\ddot{\boldsymbol{\Sigma}}_{jk}\boldsymbol{\Pi}\mathbf{Y}\right] = \operatorname{tr}(\boldsymbol{\Pi}\ddot{\boldsymbol{\Sigma}}_{jk}\boldsymbol{\Pi}\boldsymbol{\Sigma}) = \operatorname{tr}(\boldsymbol{\Pi}\ddot{\boldsymbol{\Sigma}}_{jk})
$$

because $\boldsymbol{\Pi}\boldsymbol{\Sigma}\boldsymbol{\Pi} = \boldsymbol{\Pi}$. This expectation *identically cancels* the second-derivative term arising from the log-determinant. As a result, the expected Fisher information depends exclusively on first derivatives:

$$
\mathcal{I}_{jk}(\boldsymbol{\theta}) = \frac{1}{2}\operatorname{tr}\!\left(\boldsymbol{\Pi}\dot{\boldsymbol{\Sigma}}_j\boldsymbol{\Pi}\dot{\boldsymbol{\Sigma}}_k\right)
$$

This remarkable cancellation guarantees that Fisher scoring algorithms do not require expensive second derivatives of the covariance function (§5.4.6).

---

## A.3 Moments of Quadratic Forms in Normal Vectors

Quadratic forms $\mathbf{Y}'\mathbf{M}\mathbf{Y}$ drive test statistics and variance estimators throughout the text, including the sample variance $s^2$ (Chapter 1), Moran's $I$ (Chapter 6), and Lagrange multiplier tests (Chapter 7).

### A.3.1 Expectation of a Quadratic Form

**Theorem.** Let $\mathbf{Y} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ be an $n$-dimensional random vector, and let $\mathbf{M}$ be an $n \times n$ symmetric matrix. Then:

$$
\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}
$$

*Proof.* Write $\mathbf{Y} = (\mathbf{Y} - \boldsymbol{\mu}) + \boldsymbol{\mu}$. Expanding the quadratic form:

$$
\mathbf{Y}'\mathbf{M}\mathbf{Y} = (\mathbf{Y} - \boldsymbol{\mu})'\mathbf{M}(\mathbf{Y} - \boldsymbol{\mu}) + 2\boldsymbol{\mu}'\mathbf{M}(\mathbf{Y} - \boldsymbol{\mu}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}
$$

Taking expectations, the middle term vanishes because $\operatorname{E}[\mathbf{Y} - \boldsymbol{\mu}] = \mathbf{0}$. For the first term, because $(\mathbf{Y} - \boldsymbol{\mu})'\mathbf{M}(\mathbf{Y} - \boldsymbol{\mu})$ is a $1 \times 1$ scalar, it equals its own trace:

$$
\begin{aligned}
\operatorname{E}\!\left[(\mathbf{Y} - \boldsymbol{\mu})'\mathbf{M}(\mathbf{Y} - \boldsymbol{\mu})\right] &= \operatorname{E}\!\left[\operatorname{tr}\left((\mathbf{Y} - \boldsymbol{\mu})'\mathbf{M}(\mathbf{Y} - \boldsymbol{\mu})\right)\right] \\
&= \operatorname{E}\!\left[\operatorname{tr}\left(\mathbf{M}(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})'\right)\right] \\
&= \operatorname{tr}\left(\mathbf{M}\operatorname{E}[(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})']\right) \\
&= \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma})
\end{aligned}
$$

where we used the cyclic invariance of the trace: $\operatorname{tr}(\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A})$. Adding $\boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}$ completes the proof. $\square$

---

### A.3.2 Variance of a Quadratic Form

**Theorem.** Under the same conditions:

$$
\operatorname{Var}(\mathbf{Y}'\mathbf{M}\mathbf{Y}) = 2\operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}\boldsymbol{\Sigma}) + 4\boldsymbol{\mu}'\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}\boldsymbol{\mu}
$$

*Proof.* Without loss of generality, let $\boldsymbol{\mu} = \mathbf{0}$ (the mean-shift term $4\boldsymbol{\mu}'\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}\boldsymbol{\mu}$ follows by expanding the cross-product terms).

Let $\mathbf{Y} = \boldsymbol{\Sigma}^{1/2}\mathbf{Z}$ where $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$. Then:

$$
\mathbf{Y}'\mathbf{M}\mathbf{Y} = \mathbf{Z}'\boldsymbol{\Sigma}^{1/2}\mathbf{M}\boldsymbol{\Sigma}^{1/2}\mathbf{Z} = \mathbf{Z}'\mathbf{A}\mathbf{Z}
$$

where $\mathbf{A} = \boldsymbol{\Sigma}^{1/2}\mathbf{M}\boldsymbol{\Sigma}^{1/2}$ is symmetric. By the spectral theorem (Section A.6.1), $\mathbf{A} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ where $\mathbf{U}$ is orthogonal and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ contains the eigenvalues of $\mathbf{A}$.

Let $\mathbf{W} = \mathbf{U}'\mathbf{Z}$. Because $\mathbf{U}$ is orthogonal, $\mathbf{W} \sim N(\mathbf{0}, \mathbf{I}_n)$, meaning $W_1, \dots, W_n$ are independent standard normal variables. Then:

$$
\mathbf{Z}'\mathbf{A}\mathbf{Z} = \mathbf{W}'\boldsymbol{\Lambda}\mathbf{W} = \sum_{i=1}^n \lambda_i W_i^2
$$

Because the $W_i$ are independent, the variance of the sum is the sum of the variances:

$$
\operatorname{Var}(\mathbf{Z}'\mathbf{A}\mathbf{Z}) = \sum_{i=1}^n \lambda_i^2 \operatorname{Var}(W_i^2)
$$

For a standard normal variable $W_i$, $\operatorname{E}[W_i^2] = 1$ and $\operatorname{E}[W_i^4] = 3$. Therefore:

$$
\operatorname{Var}(W_i^2) = \operatorname{E}[W_i^4] - (\operatorname{E}[W_i^2])^2 = 3 - 1 = 2
$$

Substituting this variance:

$$
\operatorname{Var}(\mathbf{Z}'\mathbf{A}\mathbf{Z}) = \sum_{i=1}^n \lambda_i^2 (2) = 2\sum_{i=1}^n \lambda_i^2 = 2\operatorname{tr}(\boldsymbol{\Lambda}^2) = 2\operatorname{tr}(\mathbf{A}^2)
$$

Finally, using cyclic invariance of the trace:

$$
\operatorname{tr}(\mathbf{A}^2) = \operatorname{tr}\!\left(\boldsymbol{\Sigma}^{1/2}\mathbf{M}\boldsymbol{\Sigma}^{1/2}\boldsymbol{\Sigma}^{1/2}\mathbf{M}\boldsymbol{\Sigma}^{1/2}\right) = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}\boldsymbol{\Sigma})
$$

which yields $\operatorname{Var}(\mathbf{Y}'\mathbf{M}\mathbf{Y}) = 2\operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}\boldsymbol{\Sigma})$. $\square$

**Generalization to non-symmetric matrices.** When $\mathbf{A}$ is not symmetric (as in the SAR lag score test of Chapter 7 §7.6.4), applying the decomposition to $\frac{\mathbf{A} + \mathbf{A}'}{2}$ yields:

$$
\operatorname{Var}(\mathbf{Y}'\mathbf{A}\mathbf{Y}) = 2\operatorname{tr}\!\left(\left(\frac{\mathbf{A} + \mathbf{A}'}{2}\right)\boldsymbol{\Sigma}\left(\frac{\mathbf{A} + \mathbf{A}'}{2}\right)\boldsymbol{\Sigma}\right) = \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}\boldsymbol{\Sigma}) + \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}'\boldsymbol{\Sigma})
$$

Under independent errors $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}_n$, this simplifies to $\sigma^4\operatorname{tr}(\mathbf{A}^2 + \mathbf{A}\mathbf{A}')$.

---

### A.3.3 The Pitman–Koopmans Theorem

In Chapter 6 (§6.5.2), Moran's $I$ is defined as a ratio of quadratic forms:

$$
I = \frac{n}{S_0} \frac{\mathbf{Y}'\mathbf{M}_1\mathbf{Y}}{\mathbf{Y}'\mathbf{M}_2\mathbf{Y}} = \frac{n}{S_0} \frac{\mathbf{Y}'\mathbf{P}_c\mathbf{W}\mathbf{P}_c\mathbf{Y}}{\mathbf{Y}'\mathbf{P}_c\mathbf{Y}}
$$

Computing the expectation and variance of a ratio of random variables is notoriously difficult in general because $\operatorname{E}[A/B] \ne \operatorname{E}[A]/\operatorname{E}[B]$. The Pitman–Koopmans theorem resolves this under normality.

**Theorem (Pitman–Koopmans).** Let $\mathbf{Y} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$. Let $\mathbf{M}_1$ be a symmetric matrix and $\mathbf{M}_2$ an idempotent matrix ($\mathbf{M}_2^2 = \mathbf{M}_2$) of rank $k$. If $\mathbf{M}_1\mathbf{M}_2 = \mathbf{M}_1$, then:
1. The ratio $R = \frac{\mathbf{Y}'\mathbf{M}_1\mathbf{Y}}{\mathbf{Y}'\mathbf{M}_2\mathbf{Y}}$ is statistically independent of the denominator $\mathbf{Y}'\mathbf{M}_2\mathbf{Y}$.
2. The expectation of the ratio equals the ratio of expectations:
   $$
   \operatorname{E}[R] = \frac{\operatorname{E}[\mathbf{Y}'\mathbf{M}_1\mathbf{Y}]}{\operatorname{E}[\mathbf{Y}'\mathbf{M}_2\mathbf{Y}]} = \frac{\operatorname{tr}(\mathbf{M}_1)}{\operatorname{tr}(\mathbf{M}_2)} = \frac{\operatorname{tr}(\mathbf{M}_1)}{k}
   $$

*Proof.* Under the null hypothesis, $\mathbf{Y}$ has a spherically symmetric normal distribution. The denominator $\mathbf{Y}'\mathbf{M}_2\mathbf{Y}$ depends only on the length of the projected vector $\mathbf{P}_c\mathbf{Y}$, while the ratio $R$ is completely scale-invariant: replacing $\mathbf{Y}$ with $c\mathbf{Y}$ for any $c > 0$ leaves $R$ unchanged. By Basu's theorem, any scale-invariant statistic is independent of the complete, sufficient statistic for the scale parameter $\sigma^2$ (which is the quadratic denominator $\mathbf{Y}'\mathbf{M}_2\mathbf{Y}$).

Because $R$ and $B = \mathbf{Y}'\mathbf{M}_2\mathbf{Y}$ are independent:

$$
\operatorname{E}[A] = \operatorname{E}[R \cdot B] = \operatorname{E}[R] \cdot \operatorname{E}[B] \quad\Longrightarrow\quad \operatorname{E}[R] = \frac{\operatorname{E}[A]}{\operatorname{E}[B]}
$$

This guarantees that under Gaussian independent errors, moments of Moran's $I$ match the exact analytical formulas derived in Chapter 6 (§6.5.3). $\square$

---

## A.4 Partitioned Multivariate Normal Conditionals and Precision Forms

The relationship between joint normal distributions and their full conditional distributions is the conceptual bridge connecting geostatistical kriging (Part II) to Markov random fields (Part III).

### A.4.1 The Covariance Form

Let $\mathbf{Y}$ be partitioned into two subvectors:

$$
\begin{bmatrix} \mathbf{Y}_1 \\ \mathbf{Y}_2 \end{bmatrix} \sim N\!\left(\begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix}\right)
$$

We construct an auxiliary vector $\mathbf{W} = \mathbf{Y}_1 - \mathbf{A}\mathbf{Y}_2$ and choose the matrix $\mathbf{A}$ such that $\mathbf{W}$ and $\mathbf{Y}_2$ are uncorrelated:

$$
\operatorname{Cov}(\mathbf{W}, \mathbf{Y}_2) = \operatorname{Cov}(\mathbf{Y}_1 - \mathbf{A}\mathbf{Y}_2, \mathbf{Y}_2) = \boldsymbol{\Sigma}_{12} - \mathbf{A}\boldsymbol{\Sigma}_{22} = \mathbf{0} \quad\Longrightarrow\quad \mathbf{A} = \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}
$$

Because $\mathbf{W}$ and $\mathbf{Y}_2$ are jointly normal, zero covariance implies statistical independence. Thus, $\mathbf{W} = \mathbf{Y}_1 - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{Y}_2$ is independent of $\mathbf{Y}_2$.

Computing the mean and variance of $\mathbf{W}$:

$$
\operatorname{E}[\mathbf{W}] = \boldsymbol{\mu}_1 - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\mu}_2
$$

$$
\begin{aligned}
\operatorname{Var}(\mathbf{W}) &= \operatorname{Var}(\mathbf{Y}_1 - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{Y}_2) \\
&= \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{22}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} \\
&= \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}
\end{aligned}
$$

Because $\mathbf{Y}_1 = \mathbf{W} + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{Y}_2$, conditioning on $\mathbf{Y}_2 = \mathbf{y}_2$ treats $\mathbf{Y}_2$ as a constant:

$$
\mathbf{Y}_1 \mid (\mathbf{Y}_2 = \mathbf{y}_2) \sim N\!\left(\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{y}_2 - \boldsymbol{\mu}_2),\, \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}\right)
$$

When $\mathbf{Y}_1 = \tilde{Y}(\mathbf{s}_0)$ is a scalar target and $\mathbf{Y}_2 = \mathbf{Y}$ is the observed data vector, this yields the simple kriging equations:

$$
\hat{Y}(\mathbf{s}_0) = \mu_0 + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu}), \qquad \sigma^2_{\text{SK}} = \sigma^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0
$$

confirming that simple kriging is the exact conditional expectation under normality (§4.5.3).

---

### A.4.2 The Precision Form and Brook's Lemma

While covariance matrices characterize marginal properties, precision matrices $\mathbf{Q} = \boldsymbol{\Sigma}^{-1}$ characterize conditional properties.

Consider the exponent of the multivariate normal density expressed in terms of $\mathbf{Q}$:

$$
-\frac{1}{2}(\mathbf{y} - \boldsymbol{\mu})'\mathbf{Q}(\mathbf{y} - \boldsymbol{\mu}) = -\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n Q_{ij}(y_i - \mu_i)(y_j - \mu_j)
$$

Isolating a single variable $y_i$ by grouping all terms involving $y_i$:

$$
-\frac{1}{2}\left[ Q_{ii}(y_i - \mu_i)^2 + 2(y_i - \mu_i)\sum_{j \ne i} Q_{ij}(y_j - \mu_j) \right] + C(\mathbf{y}_{-i})
$$

Completing the square in $y_i$:

$$
-\frac{Q_{ii}}{2}\left[ (y_i - \mu_i) + \sum_{j \ne i}\frac{Q_{ij}}{Q_{ii}}(y_j - \mu_j) \right]^2 + C^*(\mathbf{y}_{-i})
$$

Recognizing this as the exponent of a univariate normal density reveals the full conditional distribution:

$$
y_i \mid \mathbf{y}_{-i} \sim N\!\left(\mu_i - \sum_{j \ne i}\frac{Q_{ij}}{Q_{ii}}(y_j - \mu_j),\, \frac{1}{Q_{ii}}\right)
$$

This fundamental identity establishes that:
1. The conditional variance is the reciprocal of the diagonal entry: $\operatorname{Var}(Y_i \mid \mathbf{Y}_{-i}) = 1/Q_{ii}$.
2. The conditional regression coefficient of $Y_i$ on $Y_j$ given all other observations is $-Q_{ij}/Q_{ii}$.
3. **Conditional Independence:** $Y_i$ and $Y_j$ are conditionally independent given all remaining observations if and only if $Q_{ij} = 0$.

Thus, specifying a sparse precision matrix $\mathbf{Q}$ defines a Gaussian Markov Random Field (GMRF) directly on a network graph (§8.1.2).

---

### A.4.3 Vecchia's Approximation as a Gaussian DAG

Vecchia's approximation (§13.4.1) factors the joint density sequentially:

$$
f(\mathbf{y}) = f(y_1)\prod_{i=2}^n f(y_i \mid y_{i-1}, \dots, y_1) \approx f(y_1)\prod_{i=2}^n f(y_i \mid \mathbf{y}_{\mathcal{C}_i})
$$

where $\mathcal{C}_i \subset \{1, \dots, i-1\}$ contains at most $k \ll n$ conditioning neighbors.

By Section A.4.1, each univariate conditional distribution is normal:

$$
y_i \mid \mathbf{y}_{\mathcal{C}_i} \sim N\!\left(\mu_i + \mathbf{b}_i'(\mathbf{y}_{\mathcal{C}_i} - \boldsymbol{\mu}_{\mathcal{C}_i}),\, F_{ii}\right)
$$

where $\mathbf{b}_i = \boldsymbol{\Sigma}_{\mathcal{C}_i}^{-1}\boldsymbol{\Sigma}_{\mathcal{C}_i, i}$ are the simple kriging weights and $F_{ii} = \Sigma_{ii} - \boldsymbol{\Sigma}_{i, \mathcal{C}_i}\boldsymbol{\Sigma}_{\mathcal{C}_i}^{-1}\boldsymbol{\Sigma}_{\mathcal{C}_i, i}$ is the conditional kriging variance.

This sequential conditioning defines a linear Gaussian **Directed Acyclic Graph (DAG)**:

$$
\mathbf{y} - \boldsymbol{\mu} = \boldsymbol{\Gamma}(\mathbf{y} - \boldsymbol{\mu}) + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \mathbf{F})
$$

where $\mathbf{F} = \operatorname{diag}(F_{11}, \dots, F_{nn})$ is diagonal, and $\boldsymbol{\Gamma}$ is strictly lower-triangular with nonzeros $\Gamma_{ij} = b_{ij}$ only when $j \in \mathcal{C}_i$.

Rearranging the structural equation:

$$
(\mathbf{I}_n - \boldsymbol{\Gamma})(\mathbf{y} - \boldsymbol{\mu}) = \boldsymbol{\varepsilon} \quad\Longrightarrow\quad \mathbf{y} - \boldsymbol{\mu} = (\mathbf{I}_n - \boldsymbol{\Gamma})^{-1}\boldsymbol{\varepsilon}
$$

Because $\boldsymbol{\Gamma}$ is strictly lower-triangular, $\boldsymbol{\Gamma}^n = \mathbf{0}$. The matrix Neumann series terminates as an exact, finite polynomial:

$$
(\mathbf{I}_n - \boldsymbol{\Gamma})^{-1} = \mathbf{I}_n + \boldsymbol{\Gamma} + \boldsymbol{\Gamma}^2 + \dots + \boldsymbol{\Gamma}^{n-1}
$$

Entry $(i, j)$ of $\boldsymbol{\Gamma}^m$ sums the product of kriging weights along all directed paths of length $m$ from node $j$ to node $i$ in the DAG (Chapter 13 Exercise 13).

Taking the covariance of $(\mathbf{I}_n - \boldsymbol{\Gamma})(\mathbf{y} - \boldsymbol{\mu}) = \boldsymbol{\varepsilon}$:

$$
(\mathbf{I}_n - \boldsymbol{\Gamma})\boldsymbol{\Sigma}(\mathbf{I}_n - \boldsymbol{\Gamma})' = \mathbf{F}
$$

Inverting both sides produces the sparse Cholesky factorization of the Vecchia precision matrix:

$$
\widetilde{\mathbf{Q}} = \boldsymbol{\Sigma}^{-1} = (\mathbf{I}_n - \boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I}_n - \boldsymbol{\Gamma})
$$

with sparse triangular factor $\mathbf{L}^{-1} = \mathbf{F}^{-1/2}(\mathbf{I}_n - \boldsymbol{\Gamma})$. In graphical modeling, $\widetilde{\mathbf{Q}}$ is the adjacency matrix of the **moral graph** formed by adding undirected edges between parents sharing a common child (§13.4.1).

---

## A.5 Kronecker Products and the vec Operator

When spatial observations are collected across regular lattices, separable space-time regimes, or multivariate channels, covariance matrices exhibit Kronecker product structure (§13.6).

### A.5.1 Definitions and Fundamental Algebraic Rules

Let $\mathbf{A}$ be an $m \times n$ matrix and $\mathbf{B}$ a $p \times q$ matrix. The **Kronecker product** $\mathbf{A} \otimes \mathbf{B}$ is the $mp \times nq$ block matrix:

$$
\mathbf{A} \otimes \mathbf{B} = \begin{bmatrix}
A_{11}\mathbf{B} & \cdots & A_{1n}\mathbf{B} \\
\vdots & \ddots & \vdots \\
A_{m1}\mathbf{B} & \cdots & A_{mn}\mathbf{B}
\end{bmatrix}
$$

The Kronecker product satisfies four essential operational identities:

1. **The Mixed-Product Rule:** If the matrix products $\mathbf{A}\mathbf{C}$ and $\mathbf{B}\mathbf{D}$ exist:
   $$
   (\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A}\mathbf{C}) \otimes (\mathbf{B}\mathbf{D})
   $$
   *Proof.* Multiplying block by block:
   $$
   [(\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D})]_{ij} = \sum_{k} (A_{ik}\mathbf{B})(C_{kj}\mathbf{D}) = \left(\sum_k A_{ik}C_{kj}\right)(\mathbf{B}\mathbf{D}) = [\mathbf{A}\mathbf{C}]_{ij}(\mathbf{B}\mathbf{D}) = [(\mathbf{A}\mathbf{C}) \otimes (\mathbf{B}\mathbf{D})]_{ij} \quad \square
   $$

2. **Inverse Rule:** If $\mathbf{A}$ and $\mathbf{B}$ are invertible:
   $$
   (\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}
   $$
   *Proof.* By the mixed-product rule:
   $$
   (\mathbf{A} \otimes \mathbf{B})(\mathbf{A}^{-1} \otimes \mathbf{B}^{-1}) = (\mathbf{A}\mathbf{A}^{-1}) \otimes (\mathbf{B}\mathbf{B}^{-1}) = \mathbf{I} \otimes \mathbf{I} = \mathbf{I} \quad \square
   $$

3. **Transpose Rule:**
   $$
   (\mathbf{A} \otimes \mathbf{B})' = \mathbf{A}' \otimes \mathbf{B}'
   $$

4. **Determinant Exponents:** If $\mathbf{A}$ is $m \times m$ and $\mathbf{B}$ is $n \times n$:
   $$
   \lvert\mathbf{A} \otimes \mathbf{B}\rvert = \lvert\mathbf{A}\rvert^n \cdot \lvert\mathbf{B}\rvert^m
   $$
   *Proof.* Let $\lambda_1, \dots, \lambda_m$ be eigenvalues of $\mathbf{A}$ and $\mu_1, \dots, \mu_n$ eigenvalues of $\mathbf{B}$. By the mixed-product rule, if $\mathbf{A}\mathbf{u}_i = \lambda_i\mathbf{u}_i$ and $\mathbf{B}\mathbf{v}_j = \mu_j\mathbf{v}_j$:
   $$
   (\mathbf{A} \otimes \mathbf{B})(\mathbf{u}_i \otimes \mathbf{v}_j) = (\mathbf{A}\mathbf{u}_i) \otimes (\mathbf{B}\mathbf{v}_j) = (\lambda_i\mu_j)(\mathbf{u}_i \otimes \mathbf{v}_j)
   $$
   Thus, the $mn$ eigenvalues of $\mathbf{A} \otimes \mathbf{B}$ are all products $\lambda_i\mu_j$. The determinant is their product:
   $$
   \lvert\mathbf{A} \otimes \mathbf{B}\rvert = \prod_{i=1}^m\prod_{j=1}^n (\lambda_i\mu_j) = \left(\prod_{i=1}^m \lambda_i^n\right)\left(\prod_{j=1}^n \mu_j^m\right) = \lvert\mathbf{A}\rvert^n \lvert\mathbf{B}\rvert^m \quad \square
   $$

---

### A.5.2 The vec Operator and Fast Linear Algebra

The vectorization operator $\operatorname{vec}(\mathbf{X})$ stacks the columns of an $m \times n$ matrix $\mathbf{X}$ into an $mn$-dimensional column vector:

$$
\mathbf{X} = [\mathbf{x}_1 \ \ \mathbf{x}_2 \ \ \cdots \ \ \mathbf{x}_n] \quad\Longrightarrow\quad \operatorname{vec}(\mathbf{X}) = \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \\ \vdots \\ \mathbf{x}_n \end{bmatrix}
$$

**The Matrix-Vector Vectorization Identity.** For matrices $\mathbf{A} \in \mathbb{R}^{p \times m}$, $\mathbf{X} \in \mathbb{R}^{m \times n}$, and $\mathbf{B} \in \mathbb{R}^{n \times q}$:

$$
\operatorname{vec}(\mathbf{A}\mathbf{X}\mathbf{B}) = (\mathbf{B}' \otimes \mathbf{A})\operatorname{vec}(\mathbf{X})
$$

*Proof.* Let $\mathbf{b}_j$ be the $j$th column of $\mathbf{B}$. The $j$th column of $\mathbf{A}\mathbf{X}\mathbf{B}$ is $\mathbf{A}\mathbf{X}\mathbf{b}_j = \mathbf{A}\left(\sum_{k=1}^n B_{kj}\mathbf{x}_k\right) = \sum_{k=1}^n B_{kj}\mathbf{A}\mathbf{x}_k$.

Writing this linear combination as a block matrix product:

$$
\sum_{k=1}^n B_{kj}\mathbf{A}\mathbf{x}_k = \begin{bmatrix} B_{1j}\mathbf{A} & B_{2j}\mathbf{A} & \cdots & B_{nj}\mathbf{A} \end{bmatrix} \begin{bmatrix} \mathbf{x}_1 \\ \vdots \\ \mathbf{x}_n \end{bmatrix} = (\mathbf{b}_j' \otimes \mathbf{A})\operatorname{vec}(\mathbf{X})
$$

Stacking all columns $j = 1, \dots, q$ yields:

$$
\operatorname{vec}(\mathbf{A}\mathbf{X}\mathbf{B}) = \begin{bmatrix} \mathbf{b}_1' \otimes \mathbf{A} \\ \vdots \\ \mathbf{b}_q' \otimes \mathbf{A} \end{bmatrix}\operatorname{vec}(\mathbf{X}) = (\mathbf{B}' \otimes \mathbf{A})\operatorname{vec}(\mathbf{X}) \quad \square
$$

**Computational Payoff.** In separable space-time models with $n_S$ locations and $n_T$ time points ($N = n_S n_T$ total observations), the covariance is $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$.
- Storing $\boldsymbol{\Sigma}$ explicitly requires $N^2$ memory entries; factoring it requires $\mathcal{O}(N^3) = \mathcal{O}(n_S^3 n_T^3)$ floating-point operations.
- Using the vec identity, we factor $\boldsymbol{\Sigma}_S = \mathbf{L}_S\mathbf{L}_S'$ in $\mathcal{O}(n_S^3)$ and $\boldsymbol{\Sigma}_T = \mathbf{L}_T\mathbf{L}_T'$ in $\mathcal{O}(n_T^3)$. Simulating a field $\mathbf{Y}$ via:
  $$
  \operatorname{vec}(\mathbf{Y}) = (\mathbf{L}_T \otimes \mathbf{L}_S)\operatorname{vec}(\mathbf{Z}) = \operatorname{vec}(\mathbf{L}_S \mathbf{Z} \mathbf{L}_T')
  $$
  takes only $\mathcal{O}(n_S n_T(n_S + n_T))$ operations and zero extra matrix storage (§13.6.2, Chapter 13 Exercise 12).

---

## A.6 Supporting Matrix Theorems and Algebraic Identities

This section gathers additional core matrix theorems referenced across the book.

### A.6.1 The Spectral Theorem for Symmetric Matrices

Every real symmetric $n \times n$ matrix $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}'$ possesses an orthogonal eigendecomposition:

$$
\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}' = \sum_{i=1}^n \lambda_i \mathbf{v}_i\mathbf{v}_i'
$$

where:
1. The eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$ are strictly real.
2. The eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_n$ form an orthonormal basis of $\mathbb{R}^n$, so $\mathbf{U}'\mathbf{U} = \mathbf{U}\mathbf{U}' = \mathbf{I}_n$.
3. The trace equals the sum of eigenvalues: $\operatorname{tr}(\boldsymbol{\Sigma}) = \sum_{i=1}^n \lambda_i$.
4. The determinant equals the product of eigenvalues: $\lvert\boldsymbol{\Sigma}\rvert = \prod_{i=1}^n \lambda_i$.
5. **Matrix powers and functions:** For any function $g: \mathbb{R} \to \mathbb{R}$, $g(\boldsymbol{\Sigma}) = \mathbf{U}\operatorname{diag}(g(\lambda_1), \dots, g(\lambda_n))\mathbf{U}'$.
6. **Symmetric matrix square root:** If $\boldsymbol{\Sigma} \succeq \mathbf{0}$, its unique symmetric positive semi-definite square root is:
   $$
   \boldsymbol{\Sigma}^{1/2} = \mathbf{U}\boldsymbol{\Lambda}^{1/2}\mathbf{U}', \qquad \boldsymbol{\Sigma}^{-1/2} = \mathbf{U}\boldsymbol{\Lambda}^{-1/2}\mathbf{U}'
   $$
   The symmetric inverse square root $\boldsymbol{\Sigma}^{-1/2}$ decorrelates $\mathbf{Y}$ along principal axis coordinates, in contrast to the lower-triangular Cholesky factor $\mathbf{L}^{-1}$ which decorrelates sequentially (§1.3.3, Chapter 1 Exercise 11).

---

### A.6.2 Existence and Uniqueness of the Cholesky Factorization

**Theorem.** If $\boldsymbol{\Sigma}$ is a symmetric positive definite $n \times n$ matrix, there exists a unique lower-triangular matrix $\mathbf{L}$ with strictly positive diagonal entries ($L_{ii} > 0$) such that:

$$
\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}'
$$

*Proof.* By induction on matrix dimension $n$. For $n = 1$, $\Sigma_{11} > 0$, so $L_{11} = \sqrt{\Sigma_{11}} > 0$ is uniquely determined. Assume the theorem holds for matrices of dimension $n - 1$. Partition $\boldsymbol{\Sigma}$ as:

$$
\boldsymbol{\Sigma} = \begin{bmatrix} \boldsymbol{\Sigma}_{n-1} & \mathbf{s} \\ \mathbf{s}' & \sigma_{nn} \end{bmatrix}
$$

We seek a partitioned lower-triangular factor:

$$
\mathbf{L} = \begin{bmatrix} \mathbf{L}_{n-1} & \mathbf{0} \\ \boldsymbol{\ell}' & l_{nn} \end{bmatrix}
$$

Multiplying $\mathbf{L}\mathbf{L}'$:

$$
\begin{bmatrix} \mathbf{L}_{n-1}\mathbf{L}_{n-1}' & \mathbf{L}_{n-1}\boldsymbol{\ell} \\ \boldsymbol{\ell}'\mathbf{L}_{n-1}' & \boldsymbol{\ell}'\boldsymbol{\ell} + l_{nn}^2 \end{bmatrix} = \begin{bmatrix} \boldsymbol{\Sigma}_{n-1} & \mathbf{s} \\ \mathbf{s}' & \sigma_{nn} \end{bmatrix}
$$

1. $\boldsymbol{\Sigma}_{n-1} = \mathbf{L}_{n-1}\mathbf{L}_{n-1}'$: By the induction hypothesis, $\mathbf{L}_{n-1}$ exists and is unique.
2. $\mathbf{L}_{n-1}\boldsymbol{\ell} = \mathbf{s}$: Because $\mathbf{L}_{n-1}$ is non-singular, $\boldsymbol{\ell} = \mathbf{L}_{n-1}^{-1}\mathbf{s}$ is uniquely determined.
3. $\sigma_{nn} - \boldsymbol{\ell}'\boldsymbol{\ell} = l_{nn}^2$: Note that $\sigma_{nn} - \boldsymbol{\ell}'\boldsymbol{\ell} = \sigma_{nn} - \mathbf{s}'(\mathbf{L}_{n-1}\mathbf{L}_{n-1}')^{-1}\mathbf{s} = \sigma_{nn} - \mathbf{s}'\boldsymbol{\Sigma}_{n-1}^{-1}\mathbf{s}$, which is the Schur complement of $\boldsymbol{\Sigma}_{n-1}$ in $\boldsymbol{\Sigma}$. Because $\boldsymbol{\Sigma}$ is positive definite, its Schur complement is strictly positive. Hence, $l_{nn} = \sqrt{\sigma_{nn} - \boldsymbol{\ell}'\boldsymbol{\ell}} > 0$ exists and is uniquely determined. $\square$

---

### A.6.3 Idempotent Projection Matrices

A square matrix $\mathbf{P}$ is **idempotent** if $\mathbf{P}^2 = \mathbf{P}$.

1. **Eigenvalues:** If $\mathbf{P}\mathbf{v} = \lambda\mathbf{v}$, then $\mathbf{P}^2\mathbf{v} = \lambda^2\mathbf{v} = \lambda\mathbf{v}$, which implies $\lambda(\lambda - 1) = 0$. Hence, all eigenvalues of an idempotent matrix are either $0$ or $1$.
2. **Trace equals rank:** By the spectral theorem:
   $$
   \operatorname{tr}(\mathbf{P}) = \sum_{i=1}^n \lambda_i = \operatorname{rank}(\mathbf{P})
   $$
3. **Orthogonal complement:** The matrix $\mathbf{I}_n - \mathbf{P}$ is also idempotent:
   $$
   (\mathbf{I}_n - \mathbf{P})^2 = \mathbf{I}_n - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I}_n - 2\mathbf{P} + \mathbf{P} = \mathbf{I}_n - \mathbf{P}
   $$
   and projects onto the null space of $\mathbf{P}$ (§1.1.2).

---

### A.6.4 Lagrange Multipliers for Constrained Spatial Optimization

In ordinary kriging (§4.3.1), we seek kriging weights $\boldsymbol{\lambda} \in \mathbb{R}^n$ that minimize the prediction variance subject to the unbiasedness constraint $\boldsymbol{\lambda}'\mathbf{1}_n = 1$:

$$
\min_{\boldsymbol{\lambda}} \left[ \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2 \right] \quad \text{subject to} \quad \mathbf{1}_n'\boldsymbol{\lambda} = 1
$$

We introduce a scalar Lagrange multiplier $2m$ (the factor 2 simplifies differentiation):

$$
\mathcal{L}(\boldsymbol{\lambda}, m) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2 + 2m(\mathbf{1}_n'\boldsymbol{\lambda} - 1)
$$

Setting vector gradients to zero:

$$
\frac{\partial\mathcal{L}}{\partial\boldsymbol{\lambda}} = 2\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\mathbf{c}_0 + 2m\mathbf{1}_n = \mathbf{0} \quad\Longrightarrow\quad \boldsymbol{\Sigma}\boldsymbol{\lambda} + m\mathbf{1}_n = \mathbf{c}_0
$$

$$
\frac{\partial\mathcal{L}}{\partial m} = 2(\mathbf{1}_n'\boldsymbol{\lambda} - 1) = 0 \quad\Longrightarrow\quad \mathbf{1}_n'\boldsymbol{\lambda} = 1
$$

Assembling into a bordered linear system:

$$
\begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ m \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
$$

Solving this system via the Schur complement yields the optimal ordinary kriging weights and Lagrange multiplier $m = -\operatorname{Cov}(\hat{Y}(\mathbf{s}_0), \hat{\mu}_{\text{GLS}})$ (§4.3.4).

---

### A.6.5 The Moore–Penrose Pseudo-Inverse and Generalized Determinants

The Intrinsic Conditional Autoregressive (ICAR) model (§8.3.4) specifies a rank-deficient precision matrix equal to the graph Laplacian: $\mathbf{Q} = \sigma^{-2}\mathcal{L} = \sigma^{-2}(\mathbf{D} - \mathbf{A})$. Because rows sum to zero, $\mathbf{Q}\mathbf{1}_n = \mathbf{0}$, meaning $\lambda_1 = 0$ and ordinary inversion fails.

1. **Moore–Penrose Pseudo-Inverse $\mathbf{Q}^+$:**
   By the spectral theorem, $\mathbf{Q} = \sum_{i=2}^n \lambda_i \mathbf{v}_i\mathbf{v}_i'$. The Moore–Penrose pseudo-inverse inverts the non-zero eigenvalues:
   $$
   \mathbf{Q}^+ = \sum_{i=2}^n \frac{1}{\lambda_i} \mathbf{v}_i\mathbf{v}_i'
   $$
   It satisfies $\mathbf{Q}\mathbf{Q}^+\mathbf{Q} = \mathbf{Q}$, $\mathbf{Q}^+\mathbf{Q}\mathbf{Q}^+ = \mathbf{Q}^+$, and $\mathbf{Q}\mathbf{Q}^+ = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n\mathbf{1}_n' = \mathbf{P}_c$. Under the constraint $\mathbf{1}_n'\mathbf{u} = 0$, $\mathbf{Q}^+$ acts as the exact covariance matrix on the constrained subspace (§8.3.4).

2. **Generalized Determinant $\lvert\mathbf{Q}\rvert^*$:**
   The generalized determinant is the product of strictly positive eigenvalues:
   $$
   \lvert\mathbf{Q}\rvert^* = \prod_{\lambda_i > 0} \lambda_i
   $$
   By **Kirchhoff's Matrix Tree Theorem**, the number of spanning trees in a connected network graph equals $\frac{1}{n}\prod_{i=2}^n \lambda_i(\mathcal{L}) = \frac{1}{n}\lvert\mathcal{L}\rvert^*$ (§8.3.4).

---

### A.6.6 The Neumann Series for Matrix Inversion

Let $\mathbf{W}$ be an $n \times n$ matrix with spectral radius $\rho(\mathbf{W}) = \max_i |\lambda_i| < 1$.

**Theorem.** The matrix inverse $(\mathbf{I}_n - \mathbf{W})^{-1}$ can be expanded as the absolutely convergent infinite power series:

$$
(\mathbf{I}_n - \mathbf{W})^{-1} = \sum_{k=0}^\infty \mathbf{W}^k = \mathbf{I}_n + \mathbf{W} + \mathbf{W}^2 + \mathbf{W}^3 + \dots
$$

*Proof.* Consider the partial sum $\mathbf{S}_m = \sum_{k=0}^m \mathbf{W}^k$. Multiplying by $\mathbf{I}_n - \mathbf{W}$:

$$
(\mathbf{I}_n - \mathbf{W})\mathbf{S}_m = (\mathbf{I}_n - \mathbf{W})(\mathbf{I}_n + \mathbf{W} + \dots + \mathbf{W}^m) = \mathbf{I}_n - \mathbf{W}^{m+1}
$$

Because all eigenvalues satisfy $|\lambda_i| < 1$, $\lim_{m \to \infty}\mathbf{W}^{m+1} = \mathbf{0}$. Taking limits:

$$
(\mathbf{I}_n - \mathbf{W})\lim_{m \to \infty}\mathbf{S}_m = \mathbf{I}_n \quad\Longrightarrow\quad \lim_{m \to \infty}\mathbf{S}_m = (\mathbf{I}_n - \mathbf{W})^{-1} \quad \square
$$

In Spatial Autoregressive (SAR) models (§7.3.2), setting $\mathbf{W} \to \rho\mathbf{W}$ decomposes the spatial multiplier into direct effects ($\mathbf{I}_n$), immediate neighbor spillovers ($\rho\mathbf{W}$), second-order spillovers ($\rho^2\mathbf{W}^2$), and higher-order network reverberations.

---

### A.6.7 The Multivariate Gaussian Integral

In the Laplace approximation for spatial GLMMs (§9.5.1), integrating out Gaussian random effects requires evaluating integrals of quadratic exponentials.

**Theorem.** For any symmetric positive definite matrix $\mathbf{H} \in \mathbb{R}^{n \times n}$ and vector $\mathbf{b} \in \mathbb{R}^n$:

$$
\int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2}\mathbf{u}'\mathbf{H}\mathbf{u} + \mathbf{b}'\mathbf{u}\right) d\mathbf{u} = (2\pi)^{n/2}\lvert\mathbf{H}\rvert^{-1/2}\exp\!\left(\frac{1}{2}\mathbf{b}'\mathbf{H}^{-1}\mathbf{b}\right)
$$

*Proof.* Completing the square in the exponent:

$$
-\frac{1}{2}\mathbf{u}'\mathbf{H}\mathbf{u} + \mathbf{b}'\mathbf{u} = -\frac{1}{2}(\mathbf{u} - \mathbf{H}^{-1}\mathbf{b})'\mathbf{H}(\mathbf{u} - \mathbf{H}^{-1}\mathbf{b}) + \frac{1}{2}\mathbf{b}'\mathbf{H}^{-1}\mathbf{b}
$$

Factoring out the constant scalar exponential:

$$
\int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2}\mathbf{u}'\mathbf{H}\mathbf{u} + \mathbf{b}'\mathbf{u}\right) d\mathbf{u} = \exp\!\left(\frac{1}{2}\mathbf{b}'\mathbf{H}^{-1}\mathbf{b}\right) \int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2}(\mathbf{u} - \mathbf{H}^{-1}\mathbf{b})'\mathbf{H}(\mathbf{u} - \mathbf{H}^{-1}\mathbf{b})\right) d\mathbf{u}
$$

Let $\mathbf{v} = \mathbf{u} - \mathbf{H}^{-1}\mathbf{b}$. By Cholesky factorization, $\mathbf{H} = \mathbf{L}\mathbf{L}'$. Substitute $\mathbf{z} = \mathbf{L}'\mathbf{v}$, with Jacobian determinant $d\mathbf{v} = \lvert\mathbf{L}'\rvert^{-1}d\mathbf{z} = \lvert\mathbf{H}\rvert^{-1/2}d\mathbf{z}$:

$$
\int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2}\mathbf{v}'\mathbf{H}\mathbf{v}\right) d\mathbf{v} = \lvert\mathbf{H}\rvert^{-1/2}\int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2}\mathbf{z}'\mathbf{z}\right) d\mathbf{z} = \lvert\mathbf{H}\rvert^{-1/2}\prod_{i=1}^n \left(\int_{-\infty}^\infty e^{-z_i^2/2}dz_i\right)
$$

Because each 1D integral equals $\sqrt{2\pi}$, the product evaluates to $(2\pi)^{n/2}\lvert\mathbf{H}\rvert^{-1/2}$, completing the derivation. $\square$

---

### A.6.8 The Cauchy–Schwarz Inequality and Effective Sample Sizes

For any vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ with inner product $\langle\mathbf{a}, \mathbf{b}\rangle = \mathbf{a}'\mathbf{b}$:

$$
(\mathbf{a}'\mathbf{b})^2 \le (\mathbf{a}'\mathbf{a})(\mathbf{b}'\mathbf{b})
$$

with equality if and only if $\mathbf{a}$ and $\mathbf{b}$ are collinear.

**Application to Effective Sample Size.** In Chapter 1 (§1.4.4, Exercise 7), the effective sample size for estimating a mean under correlation matrix $\mathbf{R}$ is bounded by setting $\mathbf{a} = \mathbf{R}^{1/2}\mathbf{1}_n$ and $\mathbf{b} = \mathbf{R}^{-1/2}\mathbf{1}_n$:

$$
(\mathbf{a}'\mathbf{b})^2 = \left((\mathbf{R}^{1/2}\mathbf{1}_n)'(\mathbf{R}^{-1/2}\mathbf{1}_n)\right)^2 = (\mathbf{1}_n'\mathbf{1}_n)^2 = n^2
$$

$$
(\mathbf{a}'\mathbf{a}) = \mathbf{1}_n'\mathbf{R}\mathbf{1}_n, \qquad (\mathbf{b}'\mathbf{b}) = \mathbf{1}_n'\mathbf{R}^{-1}\mathbf{1}_n
$$

Applying the Cauchy–Schwarz inequality:

$$
n^2 \le (\mathbf{1}_n'\mathbf{R}\mathbf{1}_n)(\mathbf{1}_n'\mathbf{R}^{-1}\mathbf{1}_n) \quad\Longrightarrow\quad \mathbf{1}_n'\mathbf{R}^{-1}\mathbf{1}_n \ge \frac{n^2}{\mathbf{1}_n'\mathbf{R}\mathbf{1}_n}
$$

Dividing by $n$, this proves that the GLS effective sample size $n_{\text{eff}}(\text{GLS}) = \mathbf{1}_n'\mathbf{R}^{-1}\mathbf{1}_n$ always equals or exceeds the naive sample-mean effective sample size $n_{\text{eff}}(\text{OLS}) = n^2 / (\mathbf{1}_n'\mathbf{R}\mathbf{1}_n)$, confirming that GLS extracts strictly more information from correlated spatial data than OLS (§1.4.4).

---

### A.6.9 The Hadamard (Schur) Product Theorem

Let $\mathbf{A} \circ \mathbf{B}$ denote the entrywise (Hadamard) product of two $n \times n$ matrices: $[\mathbf{A} \circ \mathbf{B}]_{ij} = A_{ij}B_{ij}$.

**Theorem (Schur Product Theorem).** If $\mathbf{A} \succeq \mathbf{0}$ and $\mathbf{B} \succeq \mathbf{0}$ are positive semi-definite matrices, then their Hadamard product is also positive semi-definite:

$$
\mathbf{A} \circ \mathbf{B} \succeq \mathbf{0}
$$

*Proof.* By the spectral theorem, decompose $\mathbf{B} = \sum_{k=1}^n \lambda_k \mathbf{v}_k\mathbf{v}_k'$ where $\lambda_k \ge 0$. Then entry $B_{ij} = \sum_k \lambda_k v_{ki} v_{kj}$.

For any vector $\mathbf{x} \in \mathbb{R}^n$:

$$
\mathbf{x}'(\mathbf{A} \circ \mathbf{B})\mathbf{x} = \sum_{i=1}^n\sum_{j=1}^n x_i x_j A_{ij} B_{ij} = \sum_{k=1}^n \lambda_k \sum_{i=1}^n\sum_{j=1}^n (x_i v_{ki})(x_j v_{kj})A_{ij}
$$

Let $\mathbf{y}^{(k)} \in \mathbb{R}^n$ have entries $y_i^{(k)} = x_i v_{ki}$. Then:

$$
\mathbf{x}'(\mathbf{A} \circ \mathbf{B})\mathbf{x} = \sum_{k=1}^n \lambda_k \left(\mathbf{y}^{(k)\prime}\mathbf{A}\mathbf{y}^{(k)}\right)
$$

Because $\mathbf{A}$ is positive semi-definite, $\mathbf{y}^{(k)\prime}\mathbf{A}\mathbf{y}^{(k)} \ge 0$ for each $k$. Because $\lambda_k \ge 0$, the sum is a non-negative combination of non-negative terms. Therefore, $\mathbf{x}'(\mathbf{A} \circ \mathbf{B})\mathbf{x} \ge 0$ for all $\mathbf{x}$, which proves $\mathbf{A} \circ \mathbf{B} \succeq \mathbf{0}$. $\square$

In Chapter 2 (§2.4.3) and Chapter 13 (§13.3.1), the Schur product theorem provides the mathematical foundation for covariance tapering: multiplying a dense covariance matrix $\boldsymbol{\Sigma}$ entrywise by a compactly supported positive definite Wendland correlation matrix $\mathbf{K}_{\text{Wendland}}$ produces a valid, positive definite sparse covariance matrix.
