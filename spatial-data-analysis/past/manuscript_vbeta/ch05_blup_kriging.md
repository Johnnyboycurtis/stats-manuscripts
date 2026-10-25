# Chapter 5: Best Linear Unbiased Prediction (BLUP) Framework

This chapter establishes the geometric and statistical foundations of optimal spatial prediction, known in geostatistics as *kriging*. Spatial prediction addresses the challenge of estimating an unobserved continuous physical field $Z(\mathbf{s}_0)$ at an unsampled location $\mathbf{s}_0 \in D \subset \mathbb{R}^d$ from a discrete set of spatially autocorrelated observations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$.

By the end of this chapter, you will be able to:
1. Construct kriging systems by hand for small datasets and solve Simple Kriging (SK) via orthogonal projection and Ordinary Kriging (OK) via Lagrange multipliers.
2. Explain why kriging weights can be negative, when negative weights reflect local slope extrapolation, and why smoother kernels invite them.
3. Diagnose when kriging behaves as an exact interpolator versus a smoothing filter in the presence of a nugget effect.
4. Distinguish between exact spatial screening (which holds for Simple Kriging under 1D exponential covariance) and approximate screening (which holds for Ordinary Kriging due to mean estimation).
5. Compute leave-one-out cross-validation residuals and variances across all $n$ data sites in a single matrix inversion pass.
6. Interpret what the kriging prediction variance measures—and what it leaves out when covariance parameters are plugged in from sample estimates.

| Section | Classical Anchor | Spatial Problem | Core Mathematical Object | Computational Reality |
|---|---|---|---|---|
| **5.1 Vector Space Geometry** | OLS column-space projection | Orthogonal projection of random variables | Inner product $\langle X, Y \rangle = \operatorname{Cov}(X, Y)$ | Pythagorean energy conservation; singular $\mathbf{V}$ |
| **5.2 Simple Kriging (SK)** | Bivariate normal conditioning | Spatial MMSE with known mean $\mu(\mathbf{s})$ | Dual normal system $\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0$ | Shrinkage to mean; LOO PRESS shortcut; nugget smoothing |
| **5.3 Ordinary Kriging (OK)** | OLS with unknown intercept | Spatial BLUP with unknown constant mean $\mu$ | Bordered system with multiplier $m_{\text{OK}}$ | Filtered mean; exact interpolation; variance deficit |
| **5.4 Spatial Screening & Weights** | $\operatorname{AR}(1)$ Markov independence | 1D and grid screening; negative weights | Exponential factorization; separable Kronecker product | Exact SK vs. approximate OK screening; slope extrapolation |
| **5.5 Local vs. Global Kriging** | Rolling-window regression | Computational intractability at large $n$ | Moving neighborhoods $\mathcal{S}(\mathbf{s}_0)$; covariance tapering | Edge seam jumps; octant balancing; sparse Cholesky |

---

### Classical-to-Spatial Bridge: From Bivariate Normal Conditional Expectation and Classical Regression to Best Linear Unbiased Prediction (BLUP)

In classical probability theory, consider two correlated Gaussian random variables:
$$
\begin{bmatrix} Y_0 \\ Y_1 \end{bmatrix} \sim \mathcal{N}_2\left(\begin{bmatrix} \mu_0 \\ \mu_1 \end{bmatrix}, \begin{bmatrix} \sigma_0^2 & \rho \sigma_0 \sigma_1 \\ \rho \sigma_0 \sigma_1 & \sigma_1^2 \end{bmatrix}\right)
$$
Suppose we observe $Y_1 = y_1$ and wish to predict the unobserved quantity $Y_0$. The optimal minimum mean-squared error (MMSE) predictor is the conditional expectation:
$$
\operatorname{E}[Y_0 \mid Y_1 = y_1] = \mu_0 + \rho \frac{\sigma_0}{\sigma_1} (y_1 - \mu_1)
$$
with conditional prediction variance:
$$
\operatorname{Var}(Y_0 \mid Y_1 = y_1) = \sigma_0^2 \left(1 - \rho^2\right)
$$
Notice the intuitive three-part structure of this classical 1D result:
1. The baseline starting guess is the unconditioned mean $\mu_0$.
2. The observation $y_1$ updates this guess proportionally to its deviation $(y_1 - \mu_1)$, scaled by the regression coefficient $\beta = \rho \frac{\sigma_0}{\sigma_1} = \frac{\operatorname{Cov}(Y_0, Y_1)}{\operatorname{Var}(Y_1)}$.
3. Observing $Y_1$ reduces the baseline uncertainty $\sigma_0^2$ by the variance reduction term $\rho^2 \sigma_0^2 = \frac{[\operatorname{Cov}(Y_0, Y_1)]^2}{\operatorname{Var}(Y_1)}$.

```
Classical Bivariate Conditioning:
   E[Y_0 | Y_1] = mu_0 + Cov(Y_0, Y_1) * Var(Y_1)^{-1} * (Y_1 - mu_1)
   Var(Y_0 | Y_1) = Var(Y_0) - Cov(Y_0, Y_1) * Var(Y_1)^{-1} * Cov(Y_1, Y_0)

Multivariate Spatial Extension (Simple Kriging):
   E[Z(s_0) | Z] = mu(s_0) + c_0' V^{-1} (Z - mu)
   Var(Z(s_0) | Z) = C(0) - c_0' V^{-1} c_0
```

#### Transition to Multivariate Spatial Prediction (Simple Kriging)
In spatial statistics, we observe not one scalar $Y_1$, but a vector of $n$ correlated spatial observations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ across geographic space, and we seek to predict the field at an unsampled target site $Z(\mathbf{s}_0)$.
Assuming joint Gaussianity:
$$
\begin{bmatrix} Z(\mathbf{s}_0) \\ \mathbf{Z} \end{bmatrix} \sim \mathcal{N}_{1+n}\left(\begin{bmatrix} \mu(\mathbf{s}_0) \\ \boldsymbol{\mu} \end{bmatrix}, \begin{bmatrix} \sigma^2 & \mathbf{c}_0' \\ \mathbf{c}_0 & \mathbf{V} \end{bmatrix}\right)
$$
where $\sigma^2 = C(\mathbf{0})$, $\mathbf{c}_0 = (C(\mathbf{s}_0 - \mathbf{s}_1), \dots, C(\mathbf{s}_0 - \mathbf{s}_n))' \in \mathbb{R}^n$, and $\mathbf{V} = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}$.
Extending bivariate Gaussian conditioning to the multivariate partitioned covariance matrix immediately yields:
$$
\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] = \mu(\mathbf{s}_0) + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \boldsymbol{\mu})
$$
with conditional variance:
$$
\operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z}) = \sigma^2 - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0
$$
This expression is **Simple Kriging (SK)**! The spatial weight vector $\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0$ is the direct multidimensional generalization of the classical bivariate regression slope $\rho \frac{\sigma_0}{\sigma_1}$, accounting for both point-to-target cross-correlations $\mathbf{c}_0$ and inter-point data redundancy $\mathbf{V}^{-1}$.

#### When the Mean is Unknown: The Bridge to Ordinary Kriging
In almost all real-world applications, the spatial mean $\mu$ is unknown. If we assume a constant unknown mean $\mu(\mathbf{s}) \equiv \mu$, we cannot evaluate $\mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \mu\mathbf{1})$ directly. Instead, we must replace $\mu$ with its optimal spatial estimator.

The Best Linear Unbiased Estimator (BLUE) of the constant mean $\mu$ under spatial covariance $\mathbf{V}$ is the Generalized Least Squares (GLS) estimator:
$$
\widehat{\mu}_{\text{GLS}} = \left(\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}\right)^{-1} \mathbf{1}'\mathbf{V}^{-1}\mathbf{Z}
$$
Plugging $\widehat{\mu}_{\text{GLS}}$ into the Simple Kriging formula yields the **Ordinary Kriging (OK)** predictor:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \widehat{\mu}_{\text{GLS}} + \mathbf{c}_0'\mathbf{V}^{-1}\left(\mathbf{Z} - \widehat{\mu}_{\text{GLS}}\mathbf{1}\right)
$$
Ordinary Kriging is thus revealed not as an ad-hoc algorithm, but as the natural two-stage BLUP: it first computes the optimal spatial GLS trend estimate $\widehat{\mu}_{\text{GLS}}$ and then applies Simple Kriging to the resulting spatial residuals!

---

### 5.1 Vector Space Geometry of Random Variables and Orthogonal Prediction

#### The Geometric Anchor: Random Variables as Vectors

In classical Ordinary Least Squares (OLS) regression $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$, students learn that finding the best linear estimator is fundamentally a geometric problem. The observation vector $\mathbf{Y} \in \mathbb{R}^n$ lives in $n$-dimensional Euclidean space, and the fitted values $\widehat{\mathbf{Y}} = \mathbf{X}\widehat{\boldsymbol{\beta}}$ represent the unique orthogonal projection of $\mathbf{Y}$ onto the column space $\mathcal{C}(\mathbf{X})$. The residual vector $\mathbf{e} = \mathbf{Y} - \widehat{\mathbf{Y}}$ drops a perpendicular plumb line to $\mathcal{C}(\mathbf{X})$, satisfying the normal equations $\mathbf{X}'(\mathbf{Y} - \mathbf{X}\widehat{\boldsymbol{\beta}}) = \mathbf{0}$, which guarantees that $\mathbf{e}$ is orthogonal to every predictor column.

In spatial prediction, we perform the exact same geometric projection—with one conceptual difference: **the vectors are random variables themselves!**

Instead of working with vectors of numbers in $\mathbb{R}^n$, we consider the collection of all random variables with finite variance:
$$
\mathcal{V} = \{X : \operatorname{Var}(X) < \infty\}
$$
This collection forms a real vector space governed by standard column-space geometry:
1. **Adding Vectors:** Adding two random variables $aX + bY$ corresponds to forming a linear portfolio or linear predictor.
2. **Inner Product (Dot Product):** The dot product between two centered random variables $X$ and $Y$ (with $\operatorname{E}[X] = \operatorname{E}[Y] = 0$) is their covariance:
   $$
   \langle X, Y \rangle = \operatorname{Cov}(X, Y) = \operatorname{E}[XY]
   $$
3. **Vector Length (Norm):** The length of a centered random variable $X$ is its standard deviation:
   $$
   \|X\| = \sqrt{\langle X, X \rangle} = \sqrt{\operatorname{Var}(X)} = \operatorname{SD}(X)
   $$
4. **Distance:** The distance between two random variables $X$ and $Y$ is the root-mean-squared prediction error (RMSPE):
   $$
   d(X, Y) = \|X - Y\| = \sqrt{\operatorname{E}[(X - Y)^2]}
   $$
5. **Orthogonality:** Two random variables are orthogonal ($X \perp Y$) if and only if their inner product is zero:
   $$
   \langle X, Y \rangle = 0 \iff \operatorname{Cov}(X, Y) = 0
   $$
   Orthogonality is simply statistical uncorrelatedness!

*(Note: In advanced probability, this space of finite-variance random variables equipped with covariance as an inner product is formally referred to as the $L^2$ Hilbert space; for master's-level linear models, we require only standard finite-dimensional projections.)*

#### Definition 5.1.1 (Spatial Prediction Setting)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a real-valued second-order spatial random field indexed by a spatial domain $D \subset \mathbb{R}^d$ ($d \ge 1$), such that $\operatorname{Var}(Z(\mathbf{s})) < \infty$ for all $\mathbf{s} \in D$. Suppose the field has been observed at $n$ distinct spatial locations:
$$
\mathbf{s}_1, \mathbf{s}_2, \dots, \mathbf{s}_n \in D, \quad \mathbf{s}_i \ne \mathbf{s}_j \text{ for } i \ne j
$$
yielding the random vector of observations:
$$
\mathbf{Z} = (Z(\mathbf{s}_1), Z(\mathbf{s}_2), \dots, Z(\mathbf{s}_n))' \in \mathbb{R}^n
$$
Let $\mathbf{s}_0 \in D$ denote an arbitrary target location (typically $\mathbf{s}_0 \notin \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$). The fundamental objective of spatial prediction is to construct an estimator $\widehat{Z}(\mathbf{s}_0)$ based on the data $\mathbf{Z}$ that minimizes the mean-squared prediction error (MSPE):
$$
\operatorname{MSPE}(\widehat{Z}(\mathbf{s}_0)) = \operatorname{E}\left[\left(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)\right)^2\right]
$$
over an admissible linear subspace of predictors $\mathcal{M} = \operatorname{span}\{Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n)\}$.

#### Theorem 5.1.1 (Orthogonal Projection Theorem in the Vector Space of Random Variables)
Let $\mathcal{V}$ be the vector space of finite-variance random variables and let $\mathcal{M} = \operatorname{span}\{Y_1, \dots, Y_n\} \subset \mathcal{V}$ be the linear subspace spanned by centered observations $Y_i = Z(\mathbf{s}_i) - \operatorname{E}[Z(\mathbf{s}_i)]$, with non-singular covariance matrix $\mathbf{V} = [\operatorname{Cov}(Y_i, Y_j)]_{i,j=1}^n$. Let $X = Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0)]$ be the centered target random variable.
Then:
1. There exists a unique linear predictor $\widehat{X} = \boldsymbol{\lambda}'\mathbf{Y} \in \mathcal{M}$ that minimizes the mean-squared prediction error:
   $$
   \operatorname{E}\left[(X - \widehat{X})^2\right] = \min_{Y \in \mathcal{M}} \operatorname{E}\left[(X - Y)^2\right]
   $$
2. A necessary and sufficient condition for $\widehat{X} \in \mathcal{M}$ to achieve this minimum is that the prediction error $X - \widehat{X}$ is orthogonal (uncorrelated) to every random variable in $\mathcal{M}$:
   $$
   \operatorname{Cov}(X - \widehat{X}, Y_i) = \operatorname{E}[(X - \widehat{X}) Y_i] = 0, \quad \forall i = 1, \dots, n
   $$
3. The optimal weight vector is given explicitly by the normal equations:
   $$
   \mathbf{V}\boldsymbol{\lambda} = \mathbf{v}_0 \iff \boldsymbol{\lambda} = \mathbf{V}^{-1}\mathbf{v}_0
   $$
   where $\mathbf{v}_0 = (\operatorname{Cov}(X, Y_1), \dots, \operatorname{Cov}(X, Y_n))'$.
4. The variance of the target decomposes according to the Pythagorean identity:
   $$
   \operatorname{Var}(X) = \operatorname{Var}(\widehat{X}) + \operatorname{Var}(X - \widehat{X})
   $$

*Proof.*
1. **Quadratic Form Minimization:** Any candidate predictor $Y \in \mathcal{M}$ can be written as $Y = \boldsymbol{\lambda}'\mathbf{Y} = \sum_{i=1}^n \lambda_i Y_i$. The mean-squared prediction error is:
   $$
   \begin{aligned}
   Q(\boldsymbol{\lambda}) &= \operatorname{E}\left[\left(X - \boldsymbol{\lambda}'\mathbf{Y}\right)^2\right] \\
   &= \operatorname{E}[X^2] - 2\boldsymbol{\lambda}'\operatorname{E}[\mathbf{Y}X] + \boldsymbol{\lambda}'\operatorname{E}[\mathbf{Y}\mathbf{Y}']\boldsymbol{\lambda} \\
   &= \operatorname{Var}(X) - 2\boldsymbol{\lambda}'\mathbf{v}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda}
   \end{aligned}
   $$
   Since $\mathbf{V}$ is symmetric positive definite ($\mathbf{V} \succ 0$), $Q(\boldsymbol{\lambda})$ is a strictly convex quadratic function of $\boldsymbol{\lambda}$. Setting its gradient to zero:
   $$
   \nabla_{\boldsymbol{\lambda}} Q(\boldsymbol{\lambda}) = -2\mathbf{v}_0 + 2\mathbf{V}\boldsymbol{\lambda} = \mathbf{0} \implies \mathbf{V}\boldsymbol{\lambda} = \mathbf{v}_0
   $$
   Because $\mathbf{V}$ is non-singular, the unique global minimizer is $\boldsymbol{\lambda}^* = \mathbf{V}^{-1}\mathbf{v}_0$.

2. **Orthogonality Condition:**
   - *Sufficiency:* Suppose $\operatorname{Cov}(X - \widehat{X}, Y) = 0$ for all $Y \in \mathcal{M}$. For any competing predictor $Y \in \mathcal{M}$, decompose $X - Y = (X - \widehat{X}) + (\widehat{X} - Y)$. Because $\widehat{X} - Y \in \mathcal{M}$, the cross-term vanishes:
     $$
     \begin{aligned}
     \operatorname{E}[(X - Y)^2] &= \operatorname{E}\left[((X - \widehat{X}) + (\widehat{X} - Y))^2\right] \\
     &= \operatorname{E}[(X - \widehat{X})^2] + \operatorname{E}[(\widehat{X} - Y)^2] + 2\operatorname{E}[(X - \widehat{X})(\widehat{X} - Y)] \\
     &= \operatorname{E}[(X - \widehat{X})^2] + \operatorname{E}[(\widehat{X} - Y)^2] \ge \operatorname{E}[(X - \widehat{X})^2]
     \end{aligned}
     $$
     Equality holds if and only if $\operatorname{E}[(\widehat{X} - Y)^2] = 0 \iff Y = \widehat{X}$ (in mean square).
   - *Necessity:* For any scalar $t \in \mathbb{R}$ and any $Y \in \mathcal{M}$, $\widehat{X} + tY \in \mathcal{M}$. Define $\phi(t) = \operatorname{E}[(X - (\widehat{X} + tY))^2] = \operatorname{E}[(X - \widehat{X})^2] - 2t\operatorname{Cov}(X - \widehat{X}, Y) + t^2 \operatorname{Var}(Y)$. If $\widehat{X}$ is the minimizer, then $\phi'(0) = -2\operatorname{Cov}(X - \widehat{X}, Y) = 0$, so $(X - \widehat{X}) \perp \mathcal{M}$.

3. **Normal Equations:** In vector notation, $\operatorname{E}[(X - \boldsymbol{\lambda}'\mathbf{Y})\mathbf{Y}'] = \mathbf{c}_0' - \boldsymbol{\lambda}'\mathbf{V} = \mathbf{0}' \iff \mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$.

4. **Pythagorean Identity:** Write $X = \widehat{X} + (X - \widehat{X})$. Since $\operatorname{Cov}(\widehat{X}, X - \widehat{X}) = 0$ by orthogonality, taking variances on both sides yields:
   $$
   \operatorname{Var}(X) = \operatorname{Var}(\widehat{X}) + \operatorname{Var}(X - \widehat{X}) + 2\operatorname{Cov}(\widehat{X}, X - \widehat{X}) = \operatorname{Var}(\widehat{X}) + \operatorname{Var}(X - \widehat{X})
   $$
   This completes the proof. $\blacksquare$

#### Corollary 5.1.2 (Orthogonal Projection onto Affine Subspaces and the Geometric Route to Ordinary Kriging)
Let $\mathcal{M}_{\text{aff}} = X_0 + \mathcal{M}_0$ be an affine subspace of random variables, where $X_0 \in \mathcal{V}$ is a fixed random variable and $\mathcal{M}_0$ is a linear subspace of centered random variables. Then for any $X \in \mathcal{V}$, there exists a unique $\widehat{X} \in \mathcal{M}_{\text{aff}}$ minimizing $\operatorname{E}[(X - Y)^2]$ over $Y \in \mathcal{M}_{\text{aff}}$, characterized by:
$$
\operatorname{Cov}(X - \widehat{X}, Y_0) = 0, \quad \forall Y_0 \in \mathcal{M}_0
$$
Equivalently, $\widehat{X} = X_0 + \Pi_{\mathcal{M}_0}(X - X_0)$, where $\Pi_{\mathcal{M}_0}$ is the orthogonal projection onto $\mathcal{M}_0$.

*Geometric Connection to Ordinary Kriging:*  
When the constant spatial mean $\mu$ is unknown, we cannot form centered data directly. Instead, we require linear predictors $\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$ subject to the unbiasedness constraint $\sum_{i=1}^n \lambda_i = 1$. The set of all such linear combinations forms an affine hyperplane:
$$
\mathcal{M}_{\text{aff}} = Z(\mathbf{s}_1) + \left\{\sum_{i=1}^n w_i Z(\mathbf{s}_i) : \sum_{i=1}^n w_i = 0\right\}
$$
where the direction space $\mathcal{M}_0$ is the linear subspace of spatial contrasts (zero-sum weights). Corollary 5.1.2 guarantees that Ordinary Kriging is nothing other than the orthogonal projection of $Z(\mathbf{s}_0)$ onto this affine contrast hyperplane. Section 5.3 derives the exact same result using Lagrange multipliers. $\blacksquare$

#### Algorithmic Translation: From Projections to Normal Equations

The Orthogonal Projection Theorem translates directly into a system of normal equations. When the subspace $\mathcal{M}$ is spanned by a finite collection of centered random variables $\mathbf{Y} = (Y_1, \dots, Y_n)'$, any element in $\mathcal{M}$ can be expressed as a linear combination $\boldsymbol{\lambda}'\mathbf{Y} = \sum_{i=1}^n \lambda_i Y_i$.

```
Geometric Orthogonal Projection in the Vector Space of Random Variables:

                      Z(s_0) - μ(s_0)
                            /|
                           / |
             Hypotenuse   /  |  Error Vector e_0 ⊥ M
                         /   |  Var(e_0) = σ_{SK}^2(s_0)
                        /    |
                       /     |
                      /      | [Right Angle]
                     +-------+--------------------------->
                     0     Z_SK(s_0) - μ(s_0) ∈ M
                     \_________________ _________________/
                                       v
                             Base Subspace M = span{Z(s_1)-μ, ..., Z(s_n)-μ}
```

Procedural breakdown of orthogonal projection onto finite-dimensional subspaces:
1. **Identify the Basis:** Construct the vector of centered random variables $\mathbf{Y} = \mathbf{Z} - \operatorname{E}[\mathbf{Z}]$.
2. **Compute Inner Products:** Form the covariance matrix $\mathbf{V} = [\operatorname{Cov}(Y_i, Y_j)]_{i,j=1}^n = \operatorname{Var}(\mathbf{Z})$.
3. **Compute Target Projections:** Form the cross-covariance vector $\mathbf{c}_0 = (\operatorname{Cov}(Z(\mathbf{s}_0), Y_i))_{i=1}^n$.
4. **Solve Normal Equations:** Solve the linear system $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$ for the optimal weights using triangular Cholesky factorization.
5. **Evaluate Minimized Error:** Compute the minimized prediction variance $\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)) = \operatorname{Var}(Z(\mathbf{s}_0)) - \boldsymbol{\lambda}'\mathbf{c}_0$.

#### Diagnostic Realities and Linear Model Geometry

> [!NOTE]
> **From the Lecture Hall: Kriging is Linear Regression with Data as Predictors and Covariances as the Design**  
> In classical OLS regression, the dependent variable $Y$ is predicted from fixed explanatory columns $\mathbf{X}_1, \dots, \mathbf{X}_p$, and the normal equations $\mathbf{X}'\mathbf{X}\widehat{\boldsymbol{\beta}} = \mathbf{X}'\mathbf{Y}$ invert the sample Gram matrix. In kriging, the *data values themselves* $Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n)$ act as the predictors! The target $Z(\mathbf{s}_0)$ is projected onto the $n$-dimensional linear span of the data. The data covariance matrix $\mathbf{V}$ plays the role of the Gram matrix $\mathbf{X}'\mathbf{X}$, and the cross-covariance vector $\mathbf{c}_0$ plays the role of $\mathbf{X}'\mathbf{Y}$. The kriging weights $\boldsymbol{\lambda} = \mathbf{V}^{-1}\mathbf{c}_0$ are simply the regression coefficients of the unobserved target regressed onto the observed sample field.

- **Pythagorean Energy Conservation:** The identity $\operatorname{Var}(X) = \operatorname{Var}(\widehat{X}) + \operatorname{Var}(X - \widehat{X})$ states that the total variance of the unobserved random variable decomposes into the variance of the predictor plus the variance of the prediction error. Because the error variance is non-negative, the predictor $\widehat{X}$ can *never* have more variance than the true state $X$ under Simple Kriging.
- **Orthogonality Buys Zero Residual Signal:** Because $X - \widehat{X} \perp \mathcal{M}$, the prediction error is uncorrelated with every single observation $Z(\mathbf{s}_i)$: $\operatorname{Cov}(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0), Z(\mathbf{s}_i)) = 0$ for all $i=1,\dots,n$. There is no linear spatial signal left over in the data that could improve the prediction.
- **Singular Covariance Matrices:** If the sample locations contain exact duplicates ($\mathbf{s}_i = \mathbf{s}_j$) or collinear linear dependencies, $\mathbf{V}$ becomes singular ($\det(\mathbf{V}) = 0$). In vector space terms, the spanning set $\{Y_1, \dots, Y_n\}$ is linearly dependent. The projection $\widehat{X}$ remains unique and well-defined, but the coefficient vector $\boldsymbol{\lambda}$ is no longer unique. In practical software, never invert $\mathbf{V}$ explicitly; solve $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$ via Cholesky decomposition $\mathbf{V} = \mathbf{L}\mathbf{L}'$. If near-singularity occurs due to extreme spatial clustering, a tiny numerical jitter (e.g., $10^{-10}\mathbf{I}$) stabilizes the triangular solve without altering the physical model. (Do not confuse numerical jitter with a physical nugget effect $\tau^2 > 0$, which fundamentally alters the spatial process by adding non-zero measurement noise; see §5.2.3.)

---

### 5.2 Simple Kriging (SK)

#### The Theoretical Baseline: Known Mean and Orthogonal Projection

Simple Kriging addresses spatial prediction under the foundational assumption that the spatial mean function $\mu(\mathbf{s})$ is completely known across the domain $D$. While a known mean is idealized in practice, Simple Kriging provides the theoretical benchmark and building block against which all other kriging variants are measured.

#### Definition 5.2.1 (Simple Kriging Framework)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a spatial random field whose residual field $Y(\mathbf{s}) = Z(\mathbf{s}) - \mu(\mathbf{s})$ is second-order stationary with zero mean and continuous, strictly positive definite spatial covariance kernel:
$$
C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s} + \mathbf{h})) = \operatorname{E}\left[(Z(\mathbf{s}) - \mu(\mathbf{s}))(Z(\mathbf{s} + \mathbf{h}) - \mu(\mathbf{s} + \mathbf{h}))\right]
$$
where $\mu(\mathbf{s}) = \operatorname{E}[Z(\mathbf{s})]$ is a known deterministic spatial trend function. Let $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ denote the observed data vector, and $\boldsymbol{\mu} = (\mu(\mathbf{s}_1), \dots, \mu(\mathbf{s}_n))' \in \mathbb{R}^n$ the vector of known means at the sample locations. The $n \times n$ data covariance matrix is:
$$
\mathbf{V} = \operatorname{Var}(\mathbf{Z}) = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}
$$
and the $n \times 1$ vector of cross-covariances between the target site $\mathbf{s}_0$ and the sample sites is:
$$
\mathbf{c}_0 = (C(\mathbf{s}_0 - \mathbf{s}_1), C(\mathbf{s}_0 - \mathbf{s}_2), \dots, C(\mathbf{s}_0 - \mathbf{s}_n))' \in \mathbb{R}^n
$$
Because the observation locations are distinct and $C(\mathbf{h})$ is strictly positive definite, $\mathbf{V}$ is symmetric positive definite and non-singular.

#### Definition 5.2.2 (Simple Kriging Predictor)
The **Simple Kriging (SK) predictor** is an inhomogeneous linear combination of the observations:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = k_0 + \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = k_0 + \boldsymbol{\lambda}'\mathbf{Z}
$$
where $k_0 \in \mathbb{R}$ and $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)' \in \mathbb{R}^n$ are deterministic constants selected to satisfy:
1. **Unbiasedness:** $\operatorname{E}[\widehat{Z}_{\text{SK}}(\mathbf{s}_0)] = \operatorname{E}[Z(\mathbf{s}_0)] = \mu(\mathbf{s}_0)$.
2. **Minimum Mean-Squared Error:** The prediction error variance $\operatorname{E}[(Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0))^2]$ is minimized.

#### Theorem 5.2.1 (Simple Kriging Normal Equations and Prediction Variance)
Under Definition 5.2.1, the unique Best Linear Unbiased Predictor $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ is:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + \boldsymbol{\lambda}_{\text{SK}}' (\mathbf{Z} - \boldsymbol{\mu}) = \mu(\mathbf{s}_0) + \mathbf{c}_0' \mathbf{V}^{-1} (\mathbf{Z} - \boldsymbol{\mu})
$$
where the weight vector $\boldsymbol{\lambda}_{\text{SK}} \in \mathbb{R}^n$ satisfies the **Simple Kriging normal equations**:
$$
\mathbf{V} \boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0 \iff \boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1} \mathbf{c}_0
$$
The minimized prediction error variance, termed the **Simple Kriging variance** $\sigma_{\text{SK}}^2(\mathbf{s}_0)$, is:
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = \operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \boldsymbol{\lambda}_{\text{SK}}' \mathbf{c}_0 = C(\mathbf{0}) - \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{c}_0
$$

*Proof.*
First, evaluate the unbiasedness condition:
$$
\operatorname{E}[\widehat{Z}_{\text{SK}}(\mathbf{s}_0)] = k_0 + \boldsymbol{\lambda}' \boldsymbol{\mu} = \mu(\mathbf{s}_0) \implies k_0 = \mu(\mathbf{s}_0) - \boldsymbol{\lambda}' \boldsymbol{\mu}
$$
Substituting $k_0$ into the predictor produces the centered formulation:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) - \mu(\mathbf{s}_0) = \boldsymbol{\lambda}' (\mathbf{Z} - \boldsymbol{\mu}) = \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) - \mu(\mathbf{s}_i))
$$
The prediction error is:
$$
Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = (Z(\mathbf{s}_0) - \mu(\mathbf{s}_0)) - \boldsymbol{\lambda}' (\mathbf{Z} - \boldsymbol{\mu}) = X - \boldsymbol{\lambda}'\mathbf{Y}
$$
where $X = Z(\mathbf{s}_0) - \mu(\mathbf{s}_0)$ and $\mathbf{Y} = \mathbf{Z} - \boldsymbol{\mu}$. By the Orthogonal Projection Theorem (Theorem 5.1.1), the unique linear combination $\boldsymbol{\lambda}'\mathbf{Y}$ minimizing $\operatorname{E}[(X - \boldsymbol{\lambda}'\mathbf{Y})^2]$ satisfies the normal equations $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$, yielding $\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0$.

Substituting $\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0$ into the quadratic error expansion $\operatorname{MSPE}(\boldsymbol{\lambda}) = C(\mathbf{0}) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda}$ immediately yields:
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - 2\boldsymbol{\lambda}_{\text{SK}}'\mathbf{c}_0 + \boldsymbol{\lambda}_{\text{SK}}'\mathbf{c}_0 = C(\mathbf{0}) - \boldsymbol{\lambda}_{\text{SK}}'\mathbf{c}_0 = C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0
$$
This establishes the theorem. $\blacksquare$

##### Term-by-Term Equation Anatomy: Simple Kriging System and Prediction Variance
$$
\underbrace{\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0}_{\text{Normal Dual System}} \iff \boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0, \qquad \sigma_{\text{SK}}^2(\mathbf{s}_0) = \underbrace{\sigma^2}_{\text{Prior Field Variance}} - \underbrace{\mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0}_{\text{Spatial Information Reduction}}
$$
- **$\mathbf{V} \in \mathbb{R}^{n \times n}$:** Data-to-data covariance matrix measuring spatial autocorrelation and clustering among the $n$ sample points. Inverting $\mathbf{V}$ automatically penalizes redundant clustered points (geometric declustering).
- **$\mathbf{c}_0 \in \mathbb{R}^n$:** Target-to-data cross-covariance vector $[C(\mathbf{s}_0 - \mathbf{s}_i)]_{i=1}^n$, prioritizing sample points geographically proximate to the prediction site $\mathbf{s}_0$.
- **$\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0$:** Vector of optimal linear weights assigned to centered sample observations $(Z(\mathbf{s}_i) - \mu(\mathbf{s}_i))$.
- **$\sigma^2 = C(\mathbf{0})$:** Total stationary a priori variance of the spatial process before any sampling data are collected.
- **$-\mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0 = -\boldsymbol{\lambda}_{\text{SK}}'\mathbf{c}_0$:** Quadratic variance reduction gained by conditioning on the $n$ spatial observations. Because $\mathbf{V}^{-1} \succ 0$, this quadratic form is strictly non-negative, proving that observing spatial data never increases prediction uncertainty: $\sigma_{\text{SK}}^2(\mathbf{s}_0) \le \sigma^2$.

#### Theorem 5.2.2 (Equivalence to Conditional Expectation Under Gaussianity)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a Gaussian random field with continuous covariance kernel $C(\mathbf{h})$ and mean function $\mu(\mathbf{s})$. Then the joint distribution of $(Z(\mathbf{s}_0), \mathbf{Z}')'$ is multivariate normal:
$$
\begin{bmatrix} Z(\mathbf{s}_0) \\ \mathbf{Z} \end{bmatrix} \sim \mathcal{N}_{1+n}\left(\begin{bmatrix} \mu(\mathbf{s}_0) \\ \boldsymbol{\mu} \end{bmatrix}, \begin{bmatrix} C(\mathbf{0}) & \mathbf{c}_0' \\ \mathbf{c}_0 & \mathbf{V} \end{bmatrix}\right)
$$
Under this Gaussian specification:
1. The conditional distribution of $Z(\mathbf{s}_0)$ given the observations $\mathbf{Z}$ is univariate Gaussian:
   $$
   Z(\mathbf{s}_0) \mid \mathbf{Z} \sim \mathcal{N}\left(\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}], \operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z})\right)
   $$
2. The conditional expectation is identically equal to the Simple Kriging predictor:
   $$
   \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] = \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + \mathbf{c}_0' \mathbf{V}^{-1} (\mathbf{Z} - \boldsymbol{\mu})
   $$
3. The conditional variance is identically equal to the Simple Kriging prediction variance:
   $$
   \operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z}) = \sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{c}_0
   $$
4. $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ is the globally optimal predictor of $Z(\mathbf{s}_0)$ among *all* finite-variance functions $g(\mathbf{Z})$ (linear or non-linear) with $\operatorname{Var}(g(\mathbf{Z})) < \infty$, minimizing $\operatorname{E}[(Z(\mathbf{s}_0) - g(\mathbf{Z}))^2]$.

*Proof.*
Define the linear transformation $W = Z(\mathbf{s}_0) - \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{Z}$. Because $(Z(\mathbf{s}_0), \mathbf{Z}')'$ is jointly Gaussian, $(W, \mathbf{Z}')'$ is also jointly Gaussian. Evaluate the cross-covariance between $W$ and $\mathbf{Z}$:
$$
\operatorname{Cov}(W, \mathbf{Z}) = \operatorname{Cov}(Z(\mathbf{s}_0), \mathbf{Z}) - \mathbf{c}_0' \mathbf{V}^{-1} \operatorname{Cov}(\mathbf{Z}, \mathbf{Z}) = \mathbf{c}_0' - \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{V} = \mathbf{c}_0' - \mathbf{c}_0' = \mathbf{0}'
$$
In a multivariate Gaussian distribution, zero covariance implies statistical independence: $W \perp \mathbf{Z}$.
Consequently, the conditional distribution of $W$ given $\mathbf{Z}$ equals its marginal distribution:
$$
\operatorname{E}[W \mid \mathbf{Z}] = \operatorname{E}[W] = \mu(\mathbf{s}_0) - \mathbf{c}_0' \mathbf{V}^{-1} \boldsymbol{\mu}
$$
$$
\operatorname{Var}(W \mid \mathbf{Z}) = \operatorname{Var}(W) = \operatorname{Var}(Z(\mathbf{s}_0)) - 2\mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0 + \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{V}\mathbf{V}^{-1}\mathbf{c}_0 = C(\mathbf{0}) - \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{c}_0
$$
Since $Z(\mathbf{s}_0) = W + \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{Z}$, by linearity of conditional expectation:
$$
\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] = \operatorname{E}[W \mid \mathbf{Z}] + \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{Z} = \mu(\mathbf{s}_0) + \mathbf{c}_0' \mathbf{V}^{-1} (\mathbf{Z} - \boldsymbol{\mu}) = \widehat{Z}_{\text{SK}}(\mathbf{s}_0)
$$
and its conditional variance is $\operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z}) = \operatorname{Var}(W \mid \mathbf{Z}) = \sigma_{\text{SK}}^2(\mathbf{s}_0)$.

Finally, for any function $g(\mathbf{Z})$ with finite variance $\operatorname{Var}(g(\mathbf{Z})) < \infty$, the law of total expectation gives:
$$
\operatorname{E}[(Z(\mathbf{s}_0) - g(\mathbf{Z}))^2] = \operatorname{E}\left[\operatorname{E}\left[(Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] + \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] - g(\mathbf{Z}))^2 \;\Big|\; \mathbf{Z}\right]\right]
$$
The conditional cross-term vanishes because $\operatorname{E}[(Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]) \mid \mathbf{Z}] = 0$. Thus:
$$
\operatorname{E}[(Z(\mathbf{s}_0) - g(\mathbf{Z}))^2] = \operatorname{E}\left[(Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}])^2\right] + \operatorname{E}\left[(\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] - g(\mathbf{Z}))^2\right] \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)
$$
with equality if and only if $g(\mathbf{Z}) = \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]$ almost surely. $\blacksquare$

#### Algorithmic Translation: Solving the Simple Kriging System

The Simple Kriging procedure consists of three computational steps:
1. Assembly of the $n \times n$ data covariance matrix $\mathbf{V}$ and the $n \times 1$ cross-covariance vector $\mathbf{c}_0$.
2. Solution of the symmetric positive definite linear system $\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0$. In production software, this is executed via Cholesky decomposition ($\mathbf{V} = \mathbf{L}\mathbf{L}'$, followed by forward and back triangular substitution); the implementation below uses explicit row reduction with partial pivoting to remain completely dependency-free.
3. Linear combination of centered observations and evaluation of the quadratic variance reduction.

```python
def simple_kriging_predict(sample_coords, sample_vals, target_coord, 
                           mean_val, cov_kernel, jitter=0.0):
    """
    Computes the Simple Kriging prediction and prediction variance.

    Parameters:
    -----------
    sample_coords : list of tuples/lists of float, shape (n, d)
        Coordinates of the n observed spatial locations s_1, ..., s_n.
    sample_vals : list of float, shape (n,)
        Observed field values Z(s_1), ..., Z(s_n).
    target_coord : tuple/list of float, shape (d,)
        Coordinates of the target prediction site s_0.
    mean_val : float
        Known constant spatial mean mu = E[Z(s)].
    cov_kernel : callable
        Covariance function C(h) returning scalar covariance for distance lag h.
    jitter : float, optional
        Tiny numerical regularization added to diagonal (default 0.0).
        Note: This is strictly for numerical conditioning, not a physical nugget effect.

    Returns:
    --------
    z_hat : float
        Simple Kriging prediction Z_SK(s_0).
    sk_var : float
        Simple Kriging prediction error variance sigma_SK^2(s_0).
    weights : list of float
        Optimal Simple Kriging weights lambda_1, ..., lambda_n.
    """
    n = len(sample_coords)
    d = len(target_coord)

    # Step 1: Assemble the n x n data covariance matrix V
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_sq = sum((sample_coords[i][k] - sample_coords[j][k]) ** 2 for k in range(d))
            dist = dist_sq ** 0.5
            V[i][j] = cov_kernel(dist)
            if i == j and jitter > 0.0:
                V[i][j] += jitter

    # Step 2: Assemble the n x 1 target-to-data covariance vector c_0
    c0_vec = [0.0] * n
    for i in range(n):
        dist_sq = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(d))
        dist = dist_sq ** 0.5
        c0_vec[i] = cov_kernel(dist)

    # Step 3: Solve the symmetric positive definite linear system V * lambda = c_0
    # Gaussian elimination with partial pivoting (explicit for pedagogical transparency)
    A = [row[:] for row in V]
    b = c0_vec[:]

    for p in range(n):
        max_row = p
        max_val = abs(A[p][p])
        for r in range(p + 1, n):
            if abs(A[r][p]) > max_val:
                max_val = abs(A[r][p])
                max_row = r
        A[p], A[max_row] = A[max_row], A[p]
        b[p], b[max_row] = b[max_row], b[p]

        pivot = A[p][p]
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError("Covariance matrix V is numerically singular.")

        for c in range(p, n):
            A[p][c] /= pivot
        b[p] /= pivot

        for r in range(n):
            if r != p:
                factor = A[r][p]
                for c in range(p, n):
                    A[r][c] -= factor * A[p][c]
                b[r] -= factor * b[p]

    weights = b

    # Step 4: Evaluate the Simple Kriging prediction
    residual_sum = sum(weights[i] * (sample_vals[i] - mean_val) for i in range(n))
    z_hat = mean_val + residual_sum

    # Step 5: Evaluate the Simple Kriging variance
    prior_var = cov_kernel(0.0)
    var_reduction = sum(weights[i] * c0_vec[i] for i in range(n))
    sk_var = prior_var - var_reduction

    # Diagnostic check: negative variance signals an invalid covariance model
    if sk_var < -1e-10:
        raise ValueError(
            f"Negative kriging variance ({sk_var:.6e}) detected! "
            "The covariance kernel violates positive definiteness."
        )
    sk_var = max(0.0, sk_var)

    return z_hat, sk_var, weights
```

#### Practical Realities, Cross-Validation Shortcuts, and Nugget Smoothing

##### 1. Shrinkage Towards the Prior Mean
In Simple Kriging, the sum of weights $\sum_{i=1}^n \lambda_i$ is not constrained to equal 1. When the target site $\mathbf{s}_0$ is located far from all observation sites ($\|\mathbf{s}_0 - \mathbf{s}_i\| \gg \phi$), cross-covariances decay to zero: $\mathbf{c}_0 \to \mathbf{0}$. Consequently, the weights decay to zero ($\boldsymbol{\lambda}_{\text{SK}} \to \mathbf{0}$), and the predictor gracefully reverts to the prior population mean:
$$
\lim_{\min_i \|\mathbf{s}_0 - \mathbf{s}_i\| \to \infty} \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu(\mathbf{s}_0)
$$
Simultaneously, the variance reduction $\mathbf{c}_0' \mathbf{V}^{-1} \mathbf{c}_0$ vanishes, and the prediction variance reaches its theoretical maximum, the total sill $C(\mathbf{0})$:
$$
\lim_{\min_i \|\mathbf{s}_0 - \mathbf{s}_i\| \to \infty} \sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0})
$$

##### 2. The Spatial PRESS Shortcut: Exact Leave-One-Out (LOO) Cross-Validation
In linear regression, students learn that leave-one-out cross-validation residuals $e_{(-i)} = Y_i - \widehat{Y}_{(-i)}$ can be computed from the full OLS fit using the diagonal leverage values: $e_{(-i)} = e_i / (1 - h_{ii})$. 

In spatial statistics, an identical analytical miracle exists! For Simple Kriging, let $\widehat{Z}_{(-i)}(\mathbf{s}_i)$ denote the predictor at site $\mathbf{s}_i$ constructed using all sample points *except* $\mathbf{s}_i$. Remarkably, both the leave-one-out prediction error and its prediction variance can be evaluated from the full inverted covariance matrix $\mathbf{V}^{-1}$ without refitting the model:
$$
e_{(-i)} = Z(\mathbf{s}_i) - \widehat{Z}_{(-i)}(\mathbf{s}_i) = \frac{[\mathbf{V}^{-1}(\mathbf{Z} - \boldsymbol{\mu})]_i}{[\mathbf{V}^{-1}]_{ii}}, \qquad \operatorname{Var}(e_{(-i)}) = \frac{1}{[\mathbf{V}^{-1}]_{ii}}
$$
This single-pass identity provides the premier model diagnostic in geostatistics: the standardized cross-validation residuals:
$$
r_i = \frac{e_{(-i)}}{\sqrt{\operatorname{Var}(e_{(-i)})}} = \frac{[\mathbf{V}^{-1}(\mathbf{Z} - \boldsymbol{\mu})]_i}{\sqrt{[\mathbf{V}^{-1}]_{ii}}}
$$
should behave as exchangeable standard normal variates $\mathcal{N}(0, 1)$ if the covariance model and mean function are correctly specified.

##### 3. Kriging with a Nugget Effect: Smoothing vs. Exact Interpolation
A critical distinction in spatial prediction is whether the sample data contain microscale measurement error (a nugget effect $\tau^2 > 0$). Consider the classical decomposition:
$$
Z(\mathbf{s}) = S(\mathbf{s}) + \epsilon(\mathbf{s})
$$
where $S(\mathbf{s})$ is the smooth latent physical process with continuous covariance $C_S(\mathbf{h})$, and $\epsilon(\mathbf{s}) \stackrel{\text{i.i.d.}}{\sim} (0, \tau^2)$ represents microscale measurement noise. The observed data covariance is $C_Z(\mathbf{h}) = C_S(\mathbf{h}) + \tau^2 \mathbb{I}_{\{\mathbf{h} = \mathbf{0}\}}$, so the data matrix is $\mathbf{V} = \mathbf{V}_S + \tau^2 \mathbf{I}_n$.

When predicting at an unobserved site $\mathbf{s}_0$, investigators must distinguish between two distinct prediction targets:
1. **Predicting the Latent Surface $S(\mathbf{s}_0)$:** The target-to-data cross-covariance is $\mathbf{c}_0 = [C_S(\mathbf{s}_0 - \mathbf{s}_i)]_{i=1}^n$ (which contains no nugget!). The prediction is $\widehat{S}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \boldsymbol{\mu})$, with error variance $\operatorname{Var}(S(\mathbf{s}_0) - \widehat{S}(\mathbf{s}_0)) = C_S(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0$.
2. **Predicting a New Noisy Measurement $Z(\mathbf{s}_0)$:** The target adds independent noise $\epsilon(\mathbf{s}_0)$, giving the same predictor $\widehat{Z}(\mathbf{s}_0) = \widehat{S}(\mathbf{s}_0)$, but the error variance increases by $\tau^2$: $\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)) = C_S(\mathbf{0}) + \tau^2 - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0$.

**The Smoothing Phenomenon:** When $\tau^2 > 0$, kriging is **no longer an exact interpolator** of the observed data! If we evaluate the latent predictor $\widehat{S}(\mathbf{s}_k)$ at an observation location $\mathbf{s}_k$, the cross-covariance vector $\mathbf{c}_0 = \mathbf{C}_{S, \bullet k}$ does not equal the $k$-th column of $\mathbf{V} = \mathbf{V}_S + \tau^2 \mathbf{I}$. Consequently, the weight vector does not collapse to $\mathbf{e}_k$; instead, weight spreads across neighboring observations. The predictor $\widehat{S}(\mathbf{s}_k)$ acts as an optimal Wiener smoothing filter that filters out measurement noise, yielding $\widehat{S}(\mathbf{s}_k) \ne Z(\mathbf{s}_k)$ and a non-zero latent error variance $\sigma^2(\mathbf{s}_k) > 0$.

##### 4. What the Kriging Variance Leaves Out
Students frequently treat the kriging variance $\sigma_{\text{SK}}^2(\mathbf{s}_0)$ as the definitive posterior uncertainty of the spatial prediction. However, standard kriging variance comes with two crucial caveats:
- **Plug-In Parameter Bias:** The formula $C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0$ treats the covariance parameters $(\sigma^2, \phi, \tau^2)$ as *known constants*. In practice, these parameters are estimated from sample data (Chapter 4). Plugging in point estimates $\widehat{\boldsymbol{\theta}}$ ignores covariance parameter uncertainty, meaning nominal $95\%$ prediction intervals $\widehat{Z}(\mathbf{s}_0) \pm 1.96\,\sigma(\mathbf{s}_0)$ are systematically somewhat too narrow.
- **Gaussian Ceiling:** The BLUP is the minimum mean-squared error predictor among *linear* combinations for any finite-variance field. However, interpreting $\pm 1.96\,\sigma(\mathbf{s}_0)$ as a 95% probability coverage interval is mathematically valid only under joint Gaussianity. For skewed or heavy-tailed fields, optimal prediction intervals require non-linear geostatistics (Chapter 9) or simulation (Chapter 10).

---

---

### 5.3 Ordinary Kriging (OK)

#### From OLS with an Unknown Constant Mean to Ordinary Kriging Lagrange Multipliers

In classical linear regression with an unknown intercept $Y_i = \beta_0 + \epsilon_i$, an affine linear estimator $\widehat{\beta}_0 = \sum_{i=1}^n \lambda_i Y_i$ is unbiased if and only if the weights sum to unity:
$$
\operatorname{E}\left[\sum_{i=1}^n \lambda_i Y_i\right] = \beta_0 \sum_{i=1}^n \lambda_i = \beta_0 \iff \sum_{i=1}^n \lambda_i = 1
$$
In spatial statistics, the true regional mean $\mu$ is almost never known a priori. Simple Kriging assumes $\mu$ is known, but if an investigator uses an arbitrary or guessed mean $\mu^* \ne \mu$, the Simple Kriging predictor suffers from persistent systematic bias: $\operatorname{E}[\widehat{Z}_{\text{SK}}(\mathbf{s}_0) - Z(\mathbf{s}_0)] = (1 - \sum \lambda_i)(\mu^* - \mu)$.

Ordinary Kriging overcomes this vulnerability by assuming that the mean is an **unknown constant** across the spatial domain:
$$
\operatorname{E}[Z(\mathbf{s})] \equiv \mu \in \mathbb{R}, \quad \forall \mathbf{s} \in D
$$
and enforcing the exact same affine constraint $\sum_{i=1}^n \lambda_i = 1$. This constraint filters out the unknown parameter $\mu$ entirely, guaranteeing unbiasedness $\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0)] = \mu$ regardless of the numerical value of $\mu$. To minimize the quadratic prediction error variance subject to this linear equality constraint, we introduce a single scalar Lagrange multiplier $m_{\text{OK}}$, exactly as in constrained linear model optimization.

#### Definition 5.3.1 (Ordinary Kriging Predictor)
The **Ordinary Kriging (OK) predictor** is a linear combination of the observations:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = \boldsymbol{\lambda}' \mathbf{Z}
$$
To ensure that prediction errors are zero-mean regardless of the unknown parameter $\mu$, we impose the uniform unbiasedness criterion:
$$
\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0) - Z(\mathbf{s}_0)] = 0 \iff \mu \sum_{i=1}^n \lambda_i - \mu = 0 \iff \mu \left(\sum_{i=1}^n \lambda_i - 1\right) = 0, \quad \forall \mu \in \mathbb{R}
$$
This condition holds for all $\mu \in \mathbb{R}$ if and only if the weights satisfy the **affine constraint**:
$$
\sum_{i=1}^n \lambda_i = \mathbf{1}' \boldsymbol{\lambda} = 1
$$
where $\mathbf{1} = (1, 1, \dots, 1)' \in \mathbb{R}^n$.

#### Theorem 5.3.1 (Ordinary Kriging System via Lagrange Multipliers)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a second-order stationary random field with unknown constant mean $\mu$ and covariance kernel $C(\mathbf{h})$. The weight vector $\boldsymbol{\lambda} \in \mathbb{R}^n$ that minimizes the prediction error variance $\operatorname{E}[(Z(\mathbf{s}_0) - \boldsymbol{\lambda}'\mathbf{Z})^2]$ subject to $\mathbf{1}'\boldsymbol{\lambda} = 1$ is the unique solution to the augmented linear system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ -m_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
$$
where $m_{\text{OK}} \in \mathbb{R}$ is the Lagrange multiplier associated with the affine unbiasedness constraint.

In expanded scalar form, the system comprises $n+1$ linear equations:
$$
\begin{cases}
\sum_{j=1}^n \lambda_j C(\mathbf{s}_i - \mathbf{s}_j) - m_{\text{OK}} = C(\mathbf{s}_0 - \mathbf{s}_i), & i = 1, \dots, n \\
\sum_{j=1}^n \lambda_j = 1 &
\end{cases}
$$
The minimized prediction error variance, termed the **Ordinary Kriging variance**, is:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{c}_0 + m_{\text{OK}} = C(\mathbf{0}) - \sum_{i=1}^n \lambda_i C(\mathbf{s}_0 - \mathbf{s}_i) + m_{\text{OK}}
$$

*Proof.*
Under the constraint $\sum_{i=1}^n \lambda_i = 1$, the prediction error satisfies:
$$
Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0) = (Z(\mathbf{s}_0) - \mu) - \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) - \mu)
$$
Because this error has zero expectation, its second moment equals its variance:
$$
\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{c}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda}
$$
We formulate the Lagrangian objective function with multiplier $2m_{\text{OK}}$:
$$
\mathcal{L}(\boldsymbol{\lambda}, m_{\text{OK}}) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{c}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} + 2 m_{\text{OK}} \left(1 - \mathbf{1}' \boldsymbol{\lambda}\right)
$$
Differentiating with respect to $\boldsymbol{\lambda}$ and setting the vector gradient to zero:
$$
\nabla_{\boldsymbol{\lambda}} \mathcal{L} = -2 \mathbf{c}_0 + 2 \mathbf{V} \boldsymbol{\lambda} - 2 m_{\text{OK}} \mathbf{1} = \mathbf{0} \implies \mathbf{V} \boldsymbol{\lambda} - m_{\text{OK}} \mathbf{1} = \mathbf{c}_0
$$
Differentiating with respect to $m_{\text{OK}}$ recovers the affine constraint:
$$
\frac{\partial \mathcal{L}}{\partial m_{\text{OK}}} = 2(1 - \mathbf{1}' \boldsymbol{\lambda}) = 0 \implies \mathbf{1}' \boldsymbol{\lambda} = 1
$$
Writing these two matrix relations in block form yields the $(n+1) \times (n+1)$ Ordinary Kriging linear system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ -m_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
$$
To evaluate the minimized prediction error variance, pre-multiply the first block equation $\mathbf{V}\boldsymbol{\lambda} - m_{\text{OK}}\mathbf{1} = \mathbf{c}_0$ by $\boldsymbol{\lambda}'$:
$$
\boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} - m_{\text{OK}} \boldsymbol{\lambda}' \mathbf{1} = \boldsymbol{\lambda}' \mathbf{c}_0 \implies \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} = \boldsymbol{\lambda}' \mathbf{c}_0 + m_{\text{OK}}
$$
where we utilized $\boldsymbol{\lambda}'\mathbf{1} = 1$. Substituting this quadratic form back into the error variance expression:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{c}_0 + (\boldsymbol{\lambda}' \mathbf{c}_0 + m_{\text{OK}}) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{c}_0 + m_{\text{OK}}
$$
This establishes the theorem. $\blacksquare$

#### Proposition 5.3.1 (Invertibility of the Bordered Kriging Matrix via Schur Complement)
Let $\mathbf{V} \in \mathbb{R}^{n \times n}$ be symmetric positive definite. Then the bordered matrix
$$
\mathbf{K} = \begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix}
$$
is non-singular, and its inverse is given by block inversion:
$$
\mathbf{K}^{-1} = \begin{bmatrix} 
\mathbf{V}^{-1} - \frac{\mathbf{V}^{-1}\mathbf{1}\mathbf{1}'\mathbf{V}^{-1}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} & \frac{\mathbf{V}^{-1}\mathbf{1}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} \\
\frac{\mathbf{1}'\mathbf{V}^{-1}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} & -\frac{1}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}}
\end{bmatrix}
$$

*Proof.*
The Schur complement of the block $\mathbf{V}$ in $\mathbf{K}$ is $S = 0 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{1} = -\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}$. Because $\mathbf{V}$ is strictly positive definite, $\mathbf{V}^{-1}$ is also strictly positive definite. For any non-zero vector $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{x}'\mathbf{V}^{-1}\mathbf{x} > 0$; in particular, with $\mathbf{x} = \mathbf{1} \ne \mathbf{0}$:
$$
\mathbf{1}'\mathbf{V}^{-1}\mathbf{1} > 0 \implies S = -\mathbf{1}'\mathbf{V}^{-1}\mathbf{1} \ne 0
$$
By the block determinant formula:
$$
\det(\mathbf{K}) = \det(\mathbf{V}) \cdot S = -\det(\mathbf{V})(\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}) \ne 0
$$
Because $\det(\mathbf{V}) > 0$ and $\mathbf{1}'\mathbf{V}^{-1}\mathbf{1} > 0$, the determinant is strictly negative ($\det(\mathbf{K}) < 0$), confirming that $\mathbf{K}$ is non-singular with exactly $n$ positive eigenvalues and 1 negative eigenvalue (a symmetric saddle point). Multiplying $\mathbf{K}$ by the explicit block inverse formula directly verifies $\mathbf{K} \mathbf{K}^{-1} = \mathbf{I}_{n+1}$. $\blacksquare$

#### Proposition 5.3.2 (Decomposition of Ordinary Kriging as Simple Kriging with Estimated GLS Mean)
The Ordinary Kriging predictor decomposes exactly into Simple Kriging evaluated at the Generalized Least Squares (GLS) estimator of the mean:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \widehat{\mu}_{\text{GLS}} + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \widehat{\mu}_{\text{GLS}}\mathbf{1})
$$
where $\widehat{\mu}_{\text{GLS}} = (\mathbf{1}'\mathbf{V}^{-1}\mathbf{1})^{-1}\mathbf{1}'\mathbf{V}^{-1}\mathbf{Z}$ is the BLUE of the unknown constant mean $\mu$. Furthermore, the Ordinary Kriging variance equals the Simple Kriging variance inflated by the estimation variance of the GLS mean:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \frac{(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)
$$

*Proof.*
From the block inverse in Proposition 5.3.1, the Ordinary Kriging weights are:
$$
\boldsymbol{\lambda}_{\text{OK}} = \left(\mathbf{V}^{-1} - \frac{\mathbf{V}^{-1}\mathbf{1}\mathbf{1}'\mathbf{V}^{-1}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}}\right)\mathbf{c}_0 + \frac{\mathbf{V}^{-1}\mathbf{1}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} \cdot 1 = \mathbf{V}^{-1}\mathbf{c}_0 + \frac{\mathbf{V}^{-1}\mathbf{1}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}}\left(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0\right)
$$
Inner-producting with $\mathbf{Z}$ yields:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{Z} + \widehat{\mu}_{\text{GLS}}\left(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0\right) = \widehat{\mu}_{\text{GLS}} + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \widehat{\mu}_{\text{GLS}}\mathbf{1})
$$
Now subtract the Simple Kriging predictor $\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \mu\mathbf{1})$:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = (\widehat{\mu}_{\text{GLS}} - \mu)\left(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0\right)
$$
Because $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ is the orthogonal projection of $Z(\mathbf{s}_0)$ onto $\mathcal{M}_{\text{data}}$, the Simple Kriging error $Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ is uncorrelated with any linear combination of $\mathbf{Z}$, and specifically uncorrelated with $\widehat{\mu}_{\text{GLS}} - \mu = (\mathbf{1}'\mathbf{V}^{-1}\mathbf{1})^{-1}\mathbf{1}'\mathbf{V}^{-1}(\mathbf{Z} - \mu\mathbf{1})$. By the Pythagorean Theorem for orthogonal linear random variables:
$$
\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = \operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0)) + \operatorname{Var}\left((\widehat{\mu}_{\text{GLS}} - \mu)(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)\right)
$$
Recognizing that $\operatorname{Var}(\widehat{\mu}_{\text{GLS}}) = (\mathbf{1}'\mathbf{V}^{-1}\mathbf{1})^{-1}$ produces the variance inflation identity immediately:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \frac{(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}}
$$
Because the second term is a square over a positive quantity, $\sigma_{\text{OK}}^2(\mathbf{s}_0) \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)$ everywhere. $\blacksquare$

##### Term-by-Term Equation Anatomy: Ordinary Kriging Bordered Lagrangian Matrix System
$$
\begin{bmatrix} 
\mathbf{V} & \mathbf{1} \\ 
\mathbf{1}' & 0 
\end{bmatrix} 
\begin{bmatrix} 
\boldsymbol{\lambda} \\ 
-m_{\text{OK}} 
\end{bmatrix} 
= 
\begin{bmatrix} 
\mathbf{c}_0 \\ 
1 
\end{bmatrix}
$$
- **$\mathbf{V} \in \mathbb{R}^{n \times n}$:** Spatial covariance matrix of observed points. The off-diagonal entries quantify mutual spatial clustering; inversion of this block automatically downweights redundant clusters (spatial declustering).
- **$\mathbf{1} \in \mathbb{R}^n$:** Vector of ones acting as the design vector for the unknown constant spatial mean $\mu(\mathbf{s}) \equiv \mu$.
- **$\mathbf{1}' \boldsymbol{\lambda} = 1$:** Affine sum-to-one constraint enforcing exact mean unbiasedness: $\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0)] = \mu \sum_i \lambda_i = \mu$.
- **$-m_{\text{OK}} \in \mathbb{R}$:** The negative of the scalar Lagrange multiplier $m_{\text{OK}}$, serving as the shadow price of the unbiasedness constraint.
- **$\mathbf{c}_0 \in \mathbb{R}^n$:** Cross-covariance vector connecting the target site $\mathbf{s}_0$ to each of the $n$ sample points.
- **$1$:** Right-hand side scalar enforcing that the weights sum exactly to unity.

##### Term-by-Term Equation Anatomy: Ordinary Kriging Prediction Variance
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = \underbrace{C(\mathbf{0})}_{\text{Prior Process Variance}} - \underbrace{\boldsymbol{\lambda}'\mathbf{c}_0}_{\text{Direct Spatial Information}} + \underbrace{m_{\text{OK}}}_{\text{Lagrange Multiplier Adjustment}}
$$
- **$C(\mathbf{0})$:** Total stationary prior variance of the spatial field before incorporating sample data.
- **$-\boldsymbol{\lambda}'\mathbf{c}_0$:** Direct variance reduction contributed by the weighted spatial cross-covariances with the observation sites.
- **$+m_{\text{OK}}$:** Variance adjustment representing the uncertainty cost of estimating the unknown mean $\mu$. By Proposition 5.3.2, Ordinary Kriging variance strictly exceeds Simple Kriging variance by $\frac{(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} \ge 0$.

> [!NOTE]
> **From the Lecture Hall: What is a Lagrange Multiplier? The Price of the Sum-to-One Constraint**  
> In constrained optimization, a Lagrange multiplier represents the rate of change of the optimal objective function with respect to a relaxation of the constraint. In Ordinary Kriging, the constraint is $\sum_{i=1}^n \lambda_i = 1$.  
> Simple Kriging minimizes prediction error variance freely without this constraint, yielding weights whose sum $\sum \lambda_{\text{SK}, i} = \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0$ is typically less than 1 (shrinking toward zero in sparse regions). Ordinary Kriging forces the weights to sum to 1 to guarantee mean-unbiasedness. The multiplier $m_{\text{OK}}$ is the exact mathematical penalty—the "price"—paid for this rigidity:
> $$
> m_{\text{OK}} = \frac{1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} - \boldsymbol{\lambda}_{\text{SK}}'\mathbf{c}_0 + \boldsymbol{\lambda}_{\text{OK}}'\mathbf{c}_0
> $$
> When the target site $\mathbf{s}_0$ is surrounded by dense, balanced data such that the unconstrained weights naturally sum close to 1 ($\mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0 \approx 1$), $m_{\text{OK}} \approx 0$ and OK variance is virtually identical to SK variance. In sparse or extrapolation regions where $\mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0 \ll 1$, $m_{\text{OK}} > 0$ inflates the prediction error variance to reflect the substantial uncertainty in estimating the regional mean.

#### Theorem 5.3.2 (Exact Interpolation Property in the Absence of a Nugget)
Suppose the spatial process has no microscale nugget effect ($\tau^2 = 0$), so that $C(\mathbf{0}) = \lim_{\|\mathbf{h}\| \to \mathbf{0}} C(\mathbf{h})$ is continuous at the origin. Let $\mathbf{s}_0 = \mathbf{s}_k \in \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ be one of the observation locations. Then:
1. The Ordinary Kriging weight vector collapses to the $k$-th canonical basis vector:
   $$
   \lambda_k = 1, \quad \text{and} \quad \lambda_j = 0 \; \text{ for all } j \ne k
   $$
2. The Ordinary Kriging predictor reproduces the observed datum identically:
   $$
   \widehat{Z}_{\text{OK}}(\mathbf{s}_k) = Z(\mathbf{s}_k)
   $$
3. The Ordinary Kriging prediction variance vanishes at the observation site:
   $$
   \sigma_{\text{OK}}^2(\mathbf{s}_k) = 0
   $$
4. The Lagrange multiplier vanishes at the observation site:
   $$
   m_{\text{OK}}(\mathbf{s}_k) = 0
   $$

*Proof.*
When $\mathbf{s}_0 = \mathbf{s}_k$ and $\tau^2 = 0$, the cross-covariance vector $\mathbf{c}_0$ is identical to the $k$-th column of $\mathbf{V}$, denoted $\mathbf{V}_{\bullet k} = \mathbf{V} \mathbf{e}_k$, where $\mathbf{e}_k = (0, \dots, 1, \dots, 0)'$ is the $k$-th standard basis vector in $\mathbb{R}^n$:
$$
\mathbf{c}_0 = \mathbf{V} \mathbf{e}_k
$$
Substitute the candidate solution $\boldsymbol{\lambda} = \mathbf{e}_k$ and $-m_{\text{OK}} = 0$ into the left-hand side of the Ordinary Kriging block system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \mathbf{e}_k \\ 0 \end{bmatrix} = \begin{bmatrix} \mathbf{V} \mathbf{e}_k + \mathbf{1} \cdot 0 \\ \mathbf{1}' \mathbf{e}_k \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
$$
Because the augmented Ordinary Kriging matrix is non-singular (Proposition 5.3.1), this solution is unique. Therefore:
$$
\boldsymbol{\lambda} = \mathbf{e}_k, \quad m_{\text{OK}} = 0
$$
The kriging prediction is $\widehat{Z}_{\text{OK}}(\mathbf{s}_k) = \mathbf{e}_k' \mathbf{Z} = Z(\mathbf{s}_k)$.
The prediction variance becomes:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_k) = C(\mathbf{0}) - \mathbf{e}_k' \mathbf{c}_0 + m_{\text{OK}} = C(\mathbf{0}) - C(\mathbf{s}_k - \mathbf{s}_k) + 0 = C(\mathbf{0}) - C(\mathbf{0}) = 0
$$
Thus, Ordinary Kriging is an exact interpolator when $\tau^2 = 0$. (When a measurement error nugget $\tau^2 > 0$ is present, as established in Section 5.2.3, the predictor acts as a smoothing filter and $\sigma_{\text{OK}}^2(\mathbf{s}_k) > 0$). $\blacksquare$

#### Theorem 5.3.3 (Equivalence of Covariance and Semivariogram Formulations)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an intrinsically stationary random field with semivariogram $\gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))$.
1. The Ordinary Kriging system can be formulated entirely in terms of the semivariogram matrix $\mathbf{\Gamma} = [\gamma(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$ and point-to-data semivariogram vector $\boldsymbol{\gamma}_0 = (\gamma(\mathbf{s}_0 - \mathbf{s}_1), \dots, \gamma(\mathbf{s}_0 - \mathbf{s}_n))'$:
   $$
   \begin{bmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ m_{\text{OK}}^{(\gamma)} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}
   $$
   or in expanded scalar form:
   $$
   \begin{cases}
   \sum_{j=1}^n \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + m_{\text{OK}}^{(\gamma)} = \gamma(\mathbf{s}_0 - \mathbf{s}_i), & i = 1, \dots, n \\
   \sum_{j=1}^n \lambda_j = 1 &
   \end{cases}
   $$
2. When the process is second-order stationary with $C(\mathbf{h}) = C(\mathbf{0}) - \gamma(\mathbf{h})$, the weight vector $\boldsymbol{\lambda}$ produced by the semivariogram system is **identical** to that of the covariance system, and the numerical values of the Lagrange multipliers are **identical**:
   $$
   m_{\text{OK}}^{(\gamma)} = m_{\text{OK}}^{(C)}
   $$
3. The Ordinary Kriging variance expressed in semivariogram terms is:
   $$
   \sigma_{\text{OK}}^2(\mathbf{s}_0) = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 + m_{\text{OK}}^{(\gamma)} = \sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) + m_{\text{OK}}^{(\gamma)}
   $$
4. The semivariogram formulation is strictly more general than the covariance formulation: it remains valid for intrinsically stationary processes with unbounded semivariograms (such as fractional Brownian fields where $\gamma(\mathbf{h}) \propto \|\mathbf{h}\|^\alpha$ with $0 < \alpha < 2$) for which no covariance function exists.

*Proof.*
Under intrinsic stationarity, the variance of any linear combination $\sum_{i=0}^n w_i Z(\mathbf{s}_i)$ with $\sum_{i=0}^n w_i = 0$ satisfies:
$$
\operatorname{Var}\left(\sum_{i=0}^n w_i Z(\mathbf{s}_i)\right) = -\sum_{i=0}^n \sum_{j=0}^n w_i w_j \gamma(\mathbf{s}_i - \mathbf{s}_j)
$$
For the prediction error $Z(\mathbf{s}_0) - \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$, set $w_0 = 1$ and $w_i = -\lambda_i$ for $i = 1, \dots, n$. The zero-sum condition holds because $w_0 + \sum_{i=1}^n w_i = 1 - \sum_{i=1}^n \lambda_i = 1 - 1 = 0$. Expanding the quadratic form:
$$
\begin{aligned}
\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0)) &= -\left( w_0^2 \gamma(\mathbf{0}) + 2 w_0 \sum_{i=1}^n w_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) + \sum_{i=1}^n \sum_{j=1}^n w_i w_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \right) \\
&= -\left( 0 - 2 \sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) + \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \right) \\
&= 2 \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 - \boldsymbol{\lambda}' \mathbf{\Gamma} \boldsymbol{\lambda}
\end{aligned}
$$
Formulate the Lagrangian to minimize this variance subject to $\mathbf{1}'\boldsymbol{\lambda} = 1$:
$$
\mathcal{L}^{(\gamma)}(\boldsymbol{\lambda}, m_{\text{OK}}^{(\gamma)}) = 2 \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 - \boldsymbol{\lambda}' \mathbf{\Gamma} \boldsymbol{\lambda} - 2 m_{\text{OK}}^{(\gamma)} \left(\mathbf{1}' \boldsymbol{\lambda} - 1\right)
$$
Differentiating with respect to $\boldsymbol{\lambda}$:
$$
\nabla_{\boldsymbol{\lambda}} \mathcal{L}^{(\gamma)} = 2 \boldsymbol{\gamma}_0 - 2 \mathbf{\Gamma} \boldsymbol{\lambda} - 2 m_{\text{OK}}^{(\gamma)} \mathbf{1} = \mathbf{0} \implies \mathbf{\Gamma} \boldsymbol{\lambda} + m_{\text{OK}}^{(\gamma)} \mathbf{1} = \boldsymbol{\gamma}_0
$$
Combining this with $\mathbf{1}'\boldsymbol{\lambda} = 1$ gives the semivariogram block system:
$$
\begin{bmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ m_{\text{OK}}^{(\gamma)} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}
$$
Now assume second-order stationarity holds, so that $\mathbf{\Gamma} = C(\mathbf{0})\mathbf{1}\mathbf{1}' - \mathbf{V}$ and $\boldsymbol{\gamma}_0 = C(\mathbf{0})\mathbf{1} - \mathbf{c}_0$. Substitute these into $\mathbf{\Gamma}\boldsymbol{\lambda} + m_{\text{OK}}^{(\gamma)}\mathbf{1} = \boldsymbol{\gamma}_0$:
$$
(C(\mathbf{0})\mathbf{1}\mathbf{1}' - \mathbf{V})\boldsymbol{\lambda} + m_{\text{OK}}^{(\gamma)}\mathbf{1} = C(\mathbf{0})\mathbf{1} - \mathbf{c}_0
$$
Because $\mathbf{1}'\boldsymbol{\lambda} = 1$, the first term simplifies: $C(\mathbf{0})\mathbf{1}(\mathbf{1}'\boldsymbol{\lambda}) = C(\mathbf{0})\mathbf{1}$. Thus:
$$
C(\mathbf{0})\mathbf{1} - \mathbf{V}\boldsymbol{\lambda} + m_{\text{OK}}^{(\gamma)}\mathbf{1} = C(\mathbf{0})\mathbf{1} - \mathbf{c}_0
$$
Subtracting $C(\mathbf{0})\mathbf{1}$ and multiplying by $-1$:
$$
\mathbf{V}\boldsymbol{\lambda} - m_{\text{OK}}^{(\gamma)}\mathbf{1} = \mathbf{c}_0
$$
Comparing this with the covariance normal equation $\mathbf{V}\boldsymbol{\lambda} - m_{\text{OK}}^{(C)}\mathbf{1} = \mathbf{c}_0$ (from Theorem 5.3.1) confirms that both systems have the exact same weights $\boldsymbol{\lambda}$, and the multipliers are identical:
$$
m_{\text{OK}}^{(\gamma)} = m_{\text{OK}}^{(C)}
$$
For the prediction variance, pre-multiply $\mathbf{\Gamma}\boldsymbol{\lambda} + m_{\text{OK}}^{(\gamma)}\mathbf{1} = \boldsymbol{\gamma}_0$ by $\boldsymbol{\lambda}'$:
$$
\boldsymbol{\lambda}' \mathbf{\Gamma} \boldsymbol{\lambda} + m_{\text{OK}}^{(\gamma)} = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 \implies \boldsymbol{\lambda}' \mathbf{\Gamma} \boldsymbol{\lambda} = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 - m_{\text{OK}}^{(\gamma)}
$$
Substituting this into $\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda}$:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - (\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - m_{\text{OK}}^{(\gamma)}) = \boldsymbol{\lambda}'\boldsymbol{\gamma}_0 + m_{\text{OK}}^{(\gamma)}
$$
This establishes the theorem. $\blacksquare$

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Wait—Do the covariance and semivariogram Ordinary Kriging systems produce different Lagrange multipliers?*  
> A persistent myth in literature claims that because covariance measures spatial *affinity* while semivariance measures *dissimilarity*, the Lagrange multiplier must flip signs. This confusion is purely an artifact of inconsistent bookkeeping!  
> Notice our symmetric formulation:
> - In the covariance system: the multiplier block is $-m_{\text{OK}}^{(C)}$, and variance is $C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 + m_{\text{OK}}^{(C)}$.
> - In the semivariogram system: the multiplier block is $+m_{\text{OK}}^{(\gamma)}$, and variance is $\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 + m_{\text{OK}}^{(\gamma)}$.  
> Under this standard formulation, the numerical values are **strictly identical**: $m_{\text{OK}}^{(\gamma)} = m_{\text{OK}}^{(C)}$. The exact same scalar appears in both systems!

#### Algorithmic Translation: Ordinary Kriging

The Ordinary Kriging solver constructs the augmented $(n+1) \times (n+1)$ system using either covariance or semivariogram functions, solves for weights and the Lagrange multiplier, and returns the BLUP prediction and prediction variance.

```python
def ordinary_kriging_predict(sample_coords, sample_vals, target_coord, 
                             variogram_kernel):
    """
    Computes Ordinary Kriging prediction and prediction variance using the semivariogram.

    Parameters:
    -----------
    sample_coords : list of tuples/lists of float, shape (n, d)
        Coordinates of observed locations s_1, ..., s_n.
    sample_vals : list of float, shape (n,)
        Observed values Z(s_1), ..., Z(s_n).
    target_coord : tuple/list of float, shape (d,)
        Coordinates of the target location s_0.
    variogram_kernel : callable
        Semivariogram function gamma(h) returning semivariance for distance lag h.
        Must satisfy gamma(0.0) == 0.0.

    Returns:
    --------
    z_hat : float
        Ordinary Kriging prediction Z_OK(s_0).
    ok_var : float
        Ordinary Kriging prediction error variance sigma_OK^2(s_0).
    weights : list of float
        Ordinary Kriging weights lambda_1, ..., lambda_n.
    m_ok : float
        Lagrange multiplier m_OK associated with the unbiasedness constraint.
    """
    n = len(sample_coords)
    d = len(target_coord)

    # Step 1: Assemble the (n+1) x (n+1) augmented semivariogram matrix Gamma_aug
    # [ Gamma   1 ]
    # [   1'    0 ]
    Gamma_aug = [[0.0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(n):
            if i == j:
                Gamma_aug[i][j] = variogram_kernel(0.0)  # == 0.0 for continuous variogram
            else:
                dist = sum((sample_coords[i][k] - sample_coords[j][k]) ** 2 for k in range(d)) ** 0.5
                Gamma_aug[i][j] = variogram_kernel(dist)

    for i in range(n):
        Gamma_aug[i][n] = 1.0
        Gamma_aug[n][i] = 1.0
    Gamma_aug[n][n] = 0.0

    # Step 2: Assemble the (n+1) x 1 right-hand side vector gamma_0_aug
    # [ gamma_0 ]
    # [    1    ]
    gamma_0_aug = [0.0] * (n + 1)
    for i in range(n):
        dist = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(d)) ** 0.5
        gamma_0_aug[i] = variogram_kernel(dist)
    gamma_0_aug[n] = 1.0

    # Step 3: Solve the linear system Gamma_aug * [lambda, m_ok]' = gamma_0_aug
    A = [row[:] for row in Gamma_aug]
    b = gamma_0_aug[:]
    m = n + 1

    for p in range(m):
        max_row = p
        max_val = abs(A[p][p])
        for r in range(p + 1, m):
            if abs(A[r][p]) > max_val:
                max_val = abs(A[r][p])
                max_row = r
        A[p], A[max_row] = A[max_row], A[p]
        b[p], b[max_row] = b[max_row], b[p]

        pivot = A[p][p]
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError(f"Singular kriging matrix at pivot {p}.")

        for c in range(p, m):
            A[p][c] /= pivot
        b[p] /= pivot

        for r in range(m):
            if r != p:
                factor = A[r][p]
                for c in range(p, m):
                    A[r][c] -= factor * A[p][c]
                b[r] -= factor * b[p]

    weights = b[:n]
    m_ok = b[n]

    # Step 4: Compute the Ordinary Kriging prediction
    z_hat = sum(weights[i] * sample_vals[i] for i in range(n))

    # Step 5: Compute the Ordinary Kriging variance
    # sigma_OK^2(s_0) = sum_{i=1}^n lambda_i * gamma(s_0 - s_i) + m_ok
    ok_var = sum(weights[i] * gamma_0_aug[i] for i in range(n)) + m_ok

    # Diagnostic check: negative variance indicates an invalid variogram model
    if ok_var < -1e-10:
        raise ValueError(
            f"Negative Ordinary Kriging variance ({ok_var:.6e}) detected! "
            "The variogram model violates conditional negative definiteness."
        )
    ok_var = max(0.0, ok_var)

    return z_hat, ok_var, weights, m_ok
```

#### Intuition, Edge Cases & Diagnostic Verification

- **Partition of Unity and Scale/Shift Invariance:** Because $\sum_{i=1}^n \lambda_i = 1$, the Ordinary Kriging weights form an affine *partition of unity*. If the sample values are shifted by a global constant $c \in \mathbb{R}$ ($Z(\mathbf{s}_i) \to Z(\mathbf{s}_i) + c$), the predictor shifts by the exact same constant:
  $$
  \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) + c) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) + c \sum_{i=1}^n \lambda_i = \widehat{Z}_{\text{OK}}(\mathbf{s}_0) + c
  $$
- **Variance Deficit and Kriging Smoothing:** Ordinary Kriging is a minimum-variance estimator, meaning that it intentionally eliminates extreme fluctuations to minimize expected squared error. The variance of the OK predictor satisfies:
  $$
  \operatorname{Var}(\widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \sigma_{\text{OK}}^2(\mathbf{s}_0) + 2m_{\text{OK}}
  $$
  In interior interpolation regions, $\sigma_{\text{OK}}^2(\mathbf{s}_0) - 2m_{\text{OK}} > 0$, so $\operatorname{Var}(\widehat{Z}_{\text{OK}}(\mathbf{s}_0)) < C(\mathbf{0})$: the variance of the predicted surface is systematically smaller than the true physical dispersion variance $C(\mathbf{0})$. Kriging maps act as low-pass spatial filters: high values are smoothed downward, low values are smoothed upward, and the sample histogram variance is compressed.  
  *The Extrapolation Caveat:* While the predictor variance is suppressed in interpolation, in extreme extrapolation regions—especially with infinitely differentiable kernels like the Gaussian covariance—the Lagrange multiplier can grow large enough that $\operatorname{Var}(\widehat{Z}_{\text{OK}}(\mathbf{s}_0))$ actually exceeds $C(\mathbf{0})$. For instance, with data at $s \in \{0, 0.2\}$ and Gaussian covariance $C(h) = e^{-h^2}$, the OK predictor variance at $s_0 = 0.5$ reaches $1.2132 > 1.0$.

#### A Worked Numerical Benchmark: Simple vs. Ordinary Kriging on a 1D Transect

To solidify the algebraic mechanics, consider four observation locations along a 1D transect with unit Exponential covariance $C(h) = \exp(-|h|)$:
$$
\mathbf{s} = (0.0, 1.0, 2.0, 3.0)', \quad \mathbf{Z} = (1.0, 2.0, 1.5, 3.0)'
$$
We predict at the midpoint target location $s_0 = 1.5$.
1. **Simple Kriging (assuming known prior mean $\mu = 0$):**
   Solving $\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0$ yields:
   $$
   \boldsymbol{\lambda}_{\text{SK}} = (0.0, 0.4434, 0.4434, 0.0)', \quad \sum_{i=1}^4 \lambda_{\text{SK}, i} = 0.8868
   $$
   The prediction is $\widehat{Z}_{\text{SK}}(1.5) = 0 + 0.4434(2.0) + 0.4434(1.5) = 1.5519$.  
   The prediction variance is $\sigma_{\text{SK}}^2(1.5) = 1.0 - \boldsymbol{\lambda}_{\text{SK}}'\mathbf{c}_0 = 0.4621$.
2. **GLS Mean Estimation:**
   The Generalized Least Squares estimator of the spatial mean across the four points is:
   $$
   \widehat{\mu}_{\text{GLS}} = \frac{\mathbf{1}'\mathbf{V}^{-1}\mathbf{Z}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} = 1.9032, \quad \operatorname{Var}(\widehat{\mu}_{\text{GLS}}) = \frac{1}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} = 0.4190
   $$
3. **Ordinary Kriging (unknown constant mean):**
   Solving the bordered augmented system yields:
   $$
   \boldsymbol{\lambda}_{\text{OK}} = (0.0347, 0.4653, 0.4653, 0.0347)', \quad \sum_{i=1}^4 \lambda_{\text{OK}, i} = 1.0, \quad m_{\text{OK}} = 0.0474
   $$
   The OK prediction is $\widehat{Z}_{\text{OK}}(1.5) = 1.7673$.  
   The prediction variance is:
   $$
   \sigma_{\text{OK}}^2(1.5) = C(0) - \boldsymbol{\lambda}_{\text{OK}}'\mathbf{c}_0 + m_{\text{OK}} = 1.0 - 0.5799 + 0.0474 = 0.4675
   $$
   Notice that the variance inflation formula from Proposition 5.3.2 holds to machine precision:
   $$
   \sigma_{\text{SK}}^2(1.5) + \frac{(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} = 0.4621 + (1 - 0.8868)^2 \times 0.4190 = 0.4621 + 0.0054 = 0.4675
   $$
   Ordinary Kriging inflates the prediction error variance by exactly $+0.0054$ to account for the uncertainty in estimating the regional mean $\mu$ from the data!

---

---

### 5.4 Geometric Properties of Kriging Weights and Spatial Screening

#### Classical-to-Spatial Bridge: From Time-Series AR(1) Recursive Subtraction to 1D Spatial Screening

In classical discrete time series, an autoregressive process of order 1, $Z_t = 
ho Z_{t-1} + \epsilon_t$, has the fundamental property that given the immediate past $Z_{t-1}$, all earlier observations $Z_{t-2}, Z_{t-3}, \dots$ provide zero additional linear predictive information for $Z_t$. 

In spatial statistics on the real line $\mathbb{R}^1$, the continuous analogue is a stationary process with Exponential covariance $C(h) = \sigma^2 \exp(-|h|/\phi)$. When predicting $Z(t_0)$ at an interior location between two sample points $t_k < t_0 < t_{k+1}$, spatial autocorrelation creates a bilateral screening effect: the two flanking observations $Z(t_k)$ and $Z(t_{k+1})$ screen $Z(t_0)$ completely from all external points $t_1, \dots, t_{k-1}$ and $t_{k+2}, \dots, t_n$. Consequently, the optimal Simple Kriging weights for all non-flanking points vanish identically ($\lambda_i = 0$ for all $i 
otin \{k, k+1\}$).

Unlike inverse-distance weighting (IDW) schemes where weights are strictly positive heuristic functions of distance, kriging weights are determined rigorously by the *geometry* of the sample configuration: relative distances to the target, mutual clustering of observation points, and the smoothness of the underlying covariance kernel. Under specific geometric configurations, nearby observations completely screen distant observations.

#### Theorem 5.4.1 (Exact Screening in One Dimension for Exponential Covariance)
Let $\{Z(t) : t \in \mathbb{R}\}$ be a stationary random process on the real line with Exponential covariance function:
$$
C(h) = \sigma^2 \exp\left(-rac{|h|}{\phi}
ight), \quad \phi > 0, \; \sigma^2 > 0
$$
Suppose the process is observed at $n$ ordered locations:
$$
t_1 < t_2 < \dots < t_k < t_0 < t_{k+1} < \dots < t_n
$$
where $t_0 \in (t_k, t_{k+1})$ is the unsampled target prediction location.
Then:
1. The Simple Kriging weight vector $oldsymbol{\lambda}_{	ext{SK}} \in \mathbb{R}^n$ assigns non-zero weights strictly to the two immediate flanking neighbors $t_k$ and $t_{k+1}$:
   $$
   \lambda_i = 0, \quad orall i 
otin \{k, k+1\}
   $$
2. The optimal weights assigned to the flanking neighbors are:
   $$
   \lambda_k = rac{\sinh\left(rac{t_{k+1} - t_0}{\phi}
ight)}{\sinh\left(rac{t_{k+1} - t_k}{\phi}
ight)}, \quad \lambda_{k+1} = rac{\sinh\left(rac{t_0 - t_k}{\phi}
ight)}{\sinh\left(rac{t_{k+1} - t_k}{\phi}
ight)}
   $$
   where $\sinh(x) = rac{1}{2}(e^x - e^{-x})$.

*Proof.*
Without loss of generality, assume $\mu = 0$ and $\sigma^2 = 1$. The Simple Kriging normal equations $\mathbf{V}oldsymbol{\lambda} = \mathbf{c}_0$ consist of $n$ scalar linear equations:
$$
\sum_{j=1}^n \exp\left(-rac{|t_i - t_j|}{\phi}
ight) \lambda_j = \exp\left(-rac{|t_0 - t_i|}{\phi}
ight), \quad i = 1, \dots, n
$$
Consider an external point to the right: $t_m$ with $m \ge k+2$. For any observation point $t_j \le t_{k+1} < t_m$, the distance satisfies $|t_m - t_j| = (t_m - t_{k+1}) + (t_{k+1} - t_j)$. By the multiplicative property of the exponential function:
$$
C(t_m - t_j) = \exp\left(-rac{t_m - t_j}{\phi}
ight) = \exp\left(-rac{t_m - t_{k+1}}{\phi}
ight) \exp\left(-rac{t_{k+1} - t_j}{\phi}
ight) = e^{-\Delta/\phi} C(t_{k+1} - t_j)
$$
where $\Delta = t_m - t_{k+1} > 0$. Similarly, for the target location $t_0 < t_{k+1} < t_m$:
$$
C(t_0 - t_m) = e^{-\Delta/\phi} C(t_0 - t_{k+1})
$$
Now, write the $m$-th and $(k+1)$-th normal equations:
$$
\sum_{j=1}^n C(t_m - t_j)\lambda_j = C(t_0 - t_m), \qquad \sum_{j=1}^n C(t_{k+1} - t_j)\lambda_j = C(t_0 - t_{k+1})
$$
Multiplying the $(k+1)$-th equation by $e^{-\Delta/\phi}$ and subtracting it from the $m$-th equation, all terms corresponding to $j \le k+1$ cancel identically because $C(t_m - t_j) - e^{-\Delta/\phi}C(t_{k+1} - t_j) = 0$. Applying this triangular elimination recursively across all external indices $i 
otin \{k, k+1\}$ demonstrates that the unique linear solution requires:
$$
\lambda_i = 0, \quad orall i 
otin \{k, k+1\}
$$
(This exact cancellation also appears directly in the 3-point system derived in Exercise 5.1).  
The system thus reduces strictly to the $2 	imes 2$ system for the flanking pair:
$$
egin{bmatrix} 1 & e^{-\Delta_0/\phi} \ e^{-\Delta_0/\phi} & 1 \end{bmatrix} egin{bmatrix} \lambda_k \ \lambda_{k+1} \end{bmatrix} = egin{bmatrix} e^{-h_1/\phi} \ e^{-h_2/\phi} \end{bmatrix}
$$
where $h_1 = t_0 - t_k > 0$, $h_2 = t_{k+1} - t_0 > 0$, and $\Delta_0 = t_{k+1} - t_k = h_1 + h_2$. The determinant is $D = 1 - e^{-2\Delta_0/\phi}$. Applying Cramer's rule to solve for $\lambda_k$:
$$
egin{aligned}
\lambda_k &= rac{e^{-h_1/\phi} - e^{-\Delta_0/\phi} e^{-h_2/\phi}}{1 - e^{-2\Delta_0/\phi}} = rac{e^{-h_1/\phi} - e^{-(h_1+2h_2)/\phi}}{1 - e^{-2(h_1+h_2)/\phi}} \
&= rac{e^{-(h_1+h_2)/\phi} \left(e^{h_2/\phi} - e^{-h_2/\phi}
ight)}{e^{-(h_1+h_2)/\phi} \left(e^{(h_1+h_2)/\phi} - e^{-(h_1+h_2)/\phi}
ight)} = rac{\sinh(h_2/\phi)}{\sinh(\Delta_0/\phi)} = rac{\sinh\left(rac{t_{k+1} - t_0}{\phi}
ight)}{\sinh\left(rac{t_{k+1} - t_k}{\phi}
ight)}
\end{aligned}
$$
By symmetry, interchanging $h_1$ and $h_2$ yields:
$$
\lambda_{k+1} = rac{\sinh(h_1/\phi)}{\sinh(\Delta_0/\phi)} = rac{\sinh\left(rac{t_0 - t_k}{\phi}
ight)}{\sinh\left(rac{t_{k+1} - t_k}{\phi}
ight)}
$$
This establishes exact 1D screening without requiring Gaussianity. $lacksquare$

```
Screening Effect in One Dimension (Exponential Covariance):
       λ_1=0       λ_{k-1}=0       λ_k > 0        Target t_0      λ_{k+1} > 0      λ_n=0
---------*-------------*--------------*---------------*---------------*--------------*-------> t
        t_1           t_{k-1}        t_k                              t_{k+1}        t_n
                                      \_______________ ________________/
                                                      v
                                        Exact Bilateral Screening:
                               Only t_k and t_{k+1} receive non-zero weights
```

#### Theorem 5.4.2 (Tensor Factorization of Kriging on Product Grids)
Let $D = \mathbb{R}^2$. Suppose the spatial covariance kernel is separable across orthogonal coordinate axes:
$$
C(\mathbf{h}) = C(h_x, h_y) = C_1(h_x) C_2(h_y)
$$
where $C_1$ and $C_2$ are valid one-dimensional covariance functions. Suppose observations are placed on a regular Cartesian product grid:
$$
\mathcal{S} = \{\mathbf{s}_{ij} = (x_i, y_j)' : i = 1, \dots, n_x, \; j = 1, \dots, n_y\}
$$
and let the target location be $\mathbf{s}_0 = (x_0, y_0)'$.
Then:
1. The $n 	imes n$ covariance matrix $\mathbf{V}$ (where $n = n_x n_y$) factors into the Kronecker product:
   $$
   \mathbf{V} = \mathbf{V}_x \otimes \mathbf{V}_y
   $$
   where $\mathbf{V}_x = [C_1(x_i - x_{i'})]_{i,i'=1}^{n_x}$ and $\mathbf{V}_y = [C_2(y_j - y_{j'})]_{j,j'=1}^{n_y}$.
2. The cross-covariance vector $\mathbf{c}_0$ factors as:
   $$
   \mathbf{c}_0 = \mathbf{c}_{0,x} \otimes \mathbf{c}_{0,y}
   $$
   where $\mathbf{c}_{0,x} = (C_1(x_0 - x_1), \dots, C_1(x_0 - x_{n_x}))'$ and $\mathbf{c}_{0,y} = (C_2(y_0 - y_1), \dots, C_2(y_0 - y_{n_y}))'$.
3. The Simple Kriging weight vector $oldsymbol{\lambda}_{	ext{SK}} \in \mathbb{R}^{n_x n_y}$ factorizes into the Kronecker product of the separate 1D weight vectors:
   $$
   oldsymbol{\lambda}_{	ext{SK}} = oldsymbol{\lambda}_x \otimes oldsymbol{\lambda}_y
   $$
   where $\mathbf{V}_x oldsymbol{\lambda}_x = \mathbf{c}_{0,x}$ and $\mathbf{V}_y oldsymbol{\lambda}_y = \mathbf{c}_{0,y}$. In particular, the weight $\lambda_{ij}$ assigned to grid node $(x_i, y_j)'$ is $\lambda_{ij} = \lambda_i^{(x)} \lambda_j^{(y)}$.
4. If $C_1$ and $C_2$ are both Exponential covariance functions, exact screening occurs simultaneously along both axes: only the four bounding vertices of the grid rectangle enclosing $(x_0, y_0)'$ receive non-zero weights.

*Proof.*
1. By definition of separability, $\operatorname{Cov}(Z(x_i, y_j), Z(x_{i'}, y_{j'})) = C_1(x_i - x_{i'}) C_2(y_j - y_{j'})$. In lexicographical ordering, the entry at $((i-1)n_y + j, (i'-1)n_y + j')$ is $[\mathbf{V}_x]_{i,i'} [\mathbf{V}_y]_{j,j'}$, which is the definition of the Kronecker product $\mathbf{V}_x \otimes \mathbf{V}_y$.
2. Similarly, the covariance with $\mathbf{s}_0 = (x_0, y_0)'$ is $C_1(x_0 - x_i) C_2(y_0 - y_j) = [\mathbf{c}_{0,x}]_i [\mathbf{c}_{0,y}]_j$, so $\mathbf{c}_0 = \mathbf{c}_{0,x} \otimes \mathbf{c}_{0,y}$.
3. By the mixed-product property of the Kronecker product, $(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}$ and $(\mathbf{A} \otimes \mathbf{B})(\mathbf{u} \otimes \mathbf{v}) = (\mathbf{A}\mathbf{u}) \otimes (\mathbf{B}\mathbf{v})$. Therefore:
   $$
   oldsymbol{\lambda}_{	ext{SK}} = \mathbf{V}^{-1} \mathbf{c}_0 = (\mathbf{V}_x \otimes \mathbf{V}_y)^{-1} (\mathbf{c}_{0,x} \otimes \mathbf{c}_{0,y}) = (\mathbf{V}_x^{-1} \mathbf{c}_{0,x}) \otimes (\mathbf{V}_y^{-1} \mathbf{c}_{0,y}) = oldsymbol{\lambda}_x \otimes oldsymbol{\lambda}_y
   $$
4. When $C_1$ and $C_2$ are Exponential, Theorem 5.4.1 guarantees that $oldsymbol{\lambda}_x$ has only 2 non-zero entries (flanking indices $k, k+1$) and $oldsymbol{\lambda}_y$ has only 2 non-zero entries (flanking indices $l, l+1$). The Kronecker product $oldsymbol{\lambda}_x \otimes oldsymbol{\lambda}_y$ has exactly $2 	imes 2 = 4$ non-zero entries, corresponding to the four corners of the grid rectangle enclosing $\mathbf{s}_0$. $lacksquare$

> [!IMPORTANT]
> **Separability and Grid Geometry:**
> Notice that Theorem 5.4.2 requires two strict conditions: (1) the covariance kernel must be separable ($C(\mathbf{h}) = C_1(h_x)C_2(h_y)$), and (2) the data must lie on a Cartesian product grid. For isotropic Euclidean covariances in $\mathbb{R}^2$ ($C(\mathbf{h}) = C(\|\mathbf{h}\|)$) or irregularly scattered observations, exact screening to a finite subset does *not* hold in general, though screening remains approximately true for points shielded behind closer data.

#### Exact vs. Approximate Screening: Simple Kriging vs. Ordinary Kriging

A critical conceptual distinction is that **exact screening** holds for **Simple Kriging** (where the mean $\mu$ is known). In **Ordinary Kriging**, screening is **approximate**:
- As established in Proposition 5.3.2, $\widehat{Z}_{	ext{OK}}(\mathbf{s}_0) = \widehat{\mu}_{	ext{GLS}} + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \widehat{\mu}_{	ext{GLS}}\mathbf{1})$.
- The residual prediction $\mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \widehat{\mu}_{	ext{GLS}}\mathbf{1})$ exhibits exact screening because $\mathbf{c}_0'\mathbf{V}^{-1}$ zeroes out non-flanking observations.
- However, the GLS trend estimate $\widehat{\mu}_{	ext{GLS}} = (\mathbf{1}'\mathbf{V}^{-1}\mathbf{1})^{-1}\mathbf{1}'\mathbf{V}^{-1}\mathbf{Z}$ depends on **all** observations across the domain.
- Consequently, Ordinary Kriging assigns small non-zero weights (typically $\sim 1	ext{--}8\%$) to distant observations purely to pool information for estimating the regional mean $\mu$.

#### Origins and Geometric Interpretation of Negative Kriging Weights

Unlike convex combination schemes (such as inverse-distance weighting) where weights are constrained to the interval $[0, 1]$, kriging systems routinely assign **negative weights** ($\lambda_i < 0$). These negative weights are not numerical errors; they are mathematically required to achieve minimum mean-squared prediction error.

Negative weights emerge from two primary geometric and analytic mechanisms:
1. **Directional Gradient and Curvature Extrapolation:** When predicting at a location $\mathbf{s}_0$ that lies outside the convex hull of the data, or when an observation site $\mathbf{s}_2$ lies shielded directly behind another site $\mathbf{s}_1$ relative to $\mathbf{s}_0$, the kriging system behaves as a numerical difference operator. The linear combination $\lambda_1 Z(\mathbf{s}_1) + \lambda_2 Z(\mathbf{s}_2)$ with $\lambda_1 > 0$ and $\lambda_2 < 0$ approximates a spatial difference:
   $$
   \lambda_1 Z(\mathbf{s}_1) + \lambda_2 Z(\mathbf{s}_2) = (\lambda_1 + \lambda_2) Z(\mathbf{s}_1) - \lambda_2 \left(Z(\mathbf{s}_1) - Z(\mathbf{s}_2)
ight)
   $$
   This subtracts redundant information from the shielded point $\mathbf{s}_2$ and extrapolates the local directional gradient toward $\mathbf{s}_0$.
2. **Kernel Differentiability at the Origin:** Negative weights are heavily influenced by the analytical smoothness of the covariance function.
   - For rough, non-differentiable kernels (such as the Exponential kernel $
u = 1/2$), sample paths are nowhere differentiable, screening is nearly total, and negative weights are rare and near zero.
   - For infinitely differentiable kernels (such as the Gaussian kernel $C(h) = \exp(-h^2/\phi^2)$) or smooth Matérn models ($
u > 1$), sample paths possess continuous spatial derivatives. The linear predictor exploits this regularity via spatial Taylor approximations, which inherently require alternating finite-difference weights.

```python
# Numerical demonstration: 1D Screening and Negative Weights
# Extrapolation from 2 sample points: s_1 = 1.0, s_2 = 2.0 to target s_0 = 0.0
import math

coords = [1.0, 2.0]
s0 = 0.0

# 1. Exponential Covariance C(h) = exp(-|h|) (rough, non-differentiable)
# V = [[1, e^-1], [e^-1, 1]], c0 = [e^-1, e^-2]
# Weights: lambda_1 = e^-1 = 0.3679, lambda_2 = 0.0 (exact screening!)
w_exp_1 = math.exp(-1.0)
w_exp_2 = 0.0

# 2. Gaussian Covariance C(h) = exp(-h^2) (infinitely smooth)
# V = [[1, e^-1], [e^-1, 1]], c0 = [e^-1, e^-4]
# Solving V lambda = c0 yields:
det = 1.0 - math.exp(-2.0)
w_gau_1 = (math.exp(-1.0) - math.exp(-1.0) * math.exp(-4.0)) / det
w_gau_2 = (math.exp(-4.0) - math.exp(-1.0) * math.exp(-1.0)) / det

print(f"Exponential Weights: lambda_1 = {w_exp_1:.4f}, lambda_2 = {w_exp_2:.4f}")
print(f"Gaussian Weights:    lambda_1 = {w_gau_1:.4f}, lambda_2 = {w_gau_2:.4f}")
# Output:
# Exponential Weights: lambda_1 = 0.3679, lambda_2 = 0.0000
# Gaussian Weights:    lambda_1 = 0.4177, lambda_2 = -0.1353
```

In the Gaussian case, the geometry is identical, yet the second weight is strictly negative ($\lambda_2 = -0.1353$). The kriging system uses the difference $Z(s_1) - Z(s_2)$ to extrapolate the smooth trajectory toward $s_0$.

> [!WARNING]
> **Physical Plausibility and Out-of-Range Predictions:**  
> Because Ordinary Kriging weights can be negative and are not constrained to $[0, 1]$, the resulting spatial prediction $\widehat{Z}(\mathbf{s}_0) = \sum \lambda_i Z(\mathbf{s}_i)$ is not guaranteed to lie within the range of the observed data $[\min_i Z(\mathbf{s}_i), \max_i Z(\mathbf{s}_i)]$.  
> When modeling non-negative physical quantities (such as ore grades, pollutant concentrations, or rainfall), large negative weights combined with steep gradients can produce **negative predicted values** ($\widehat{Z} < 0$). In such applications, practitioners either post-process by clipping predictions, fit rougher covariance kernels (e.g., Matérn with $\nu \le 3/2$), or model the log-transformed field $\log Z(\mathbf{s})$ (lognormal kriging).

> [!NOTE]
> **From the Lecture Hall: Why Does a Smoother Kernel Invite Negative Weights?**  
> Students often assume that a smoother covariance kernel (like Gaussian) must be more well-behaved than a rough kernel (like Exponential). In kriging geometry, the exact opposite occurs!  
> A process with an Exponential covariance behaves like Brownian motion: past changes provide no forecast of future slope, so extrapolation is impossible and the predictor simply damps toward the mean (weights stay non-negative and screen). A process with Gaussian covariance has smooth, parabolic curves. The kriging system knows that a high value at $s_1$ coupled with a lower value at $s_2$ implies a positive incoming slope. It uses a negative weight on $s_2$ to project that tangent forward. However, this high sensitivity to local slopes makes Gaussian kriging prone to severe ringing, overshoot, and ill-conditioned matrices.

---

### 5.5 Local vs. Global Neighborhood Kriging

#### Beat 1: The Mathematical Anchor (The "Why")

##### 5.5.1 Classical-to-Spatial Bridge: From Rolling-Window Moving Averages to Local Spatial Neighborhood Kriging
In classical time series and nonparametric regression, when sample size $n$ is very large, fitting a global spline or full covariance matrix across the entire history becomes computationally prohibitive ($O(n^3)$ operations) and practically uninformative for local dynamics. Practitioners replace the global fit with rolling-window moving averages or local linear regression (LOESS) that restrict estimation to the $k$ nearest temporal neighbors. 

In spatial geostatistics, computing global kriging requires inverting or factorizing an $n \times n$ covariance matrix $\mathbf{V}$, which becomes intractable when $n > 10^4$ or when generating high-resolution continuous predictions across an evaluation grid of $M$ target sites. Local neighborhood kriging mimics rolling windows by restricting prediction at $\mathbf{s}_0$ to a local subset $\mathcal{S}(\mathbf{s}_0) \subset \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ of size $n_0 \ll n$ (typically $n_0 \in [16, 64]$). Thanks to the spatial screening effect, the prediction loss from discarding distant observations is negligible, but local selection introduces subtle theoretical trade-offs, directional clustering challenges, and geometric boundary artifacts.

#### Proposition 5.5.1 (Sub-Optimality of Local Neighborhood Kriging)
Let $\mathcal{M}_{\text{local}} = \operatorname{span}\{Z(\mathbf{s}) : \mathbf{s} \in \mathcal{S}(\mathbf{s}_0)\}$ and $\mathcal{M}_{\text{global}} = \operatorname{span}\{Z(\mathbf{s}) : \mathbf{s} \in \mathcal{S}\}$. Because $\mathcal{S}(\mathbf{s}_0) \subset \mathcal{S}$, the local subspace is nested within the global subspace: $\mathcal{M}_{\text{local}} \subset \mathcal{M}_{\text{global}}$.

For **Simple Kriging**, by the Orthogonal Projection Theorem:
$$
\sigma_{\text{SK, global}}^2(\mathbf{s}_0) = \inf_{Y \in \mathcal{M}_{\text{global}}} \operatorname{E}\left[(Z(\mathbf{s}_0) - Y)^2\right] \le \inf_{Y \in \mathcal{M}_{\text{local}}} \operatorname{E}\left[(Z(\mathbf{s}_0) - Y)^2\right] = \sigma_{\text{SK, local}}^2(\mathbf{s}_0)
$$

For **Ordinary Kriging**, the linear predictor $\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \sum_{i \in \mathcal{S}(\mathbf{s}_0)} \lambda_i Z(\mathbf{s}_i)$ subject to $\sum_{i \in \mathcal{S}(\mathbf{s}_0)} \lambda_i = 1$ is equivalent to a global predictor whose weights on all discarded sites $\mathbf{s} \in \mathcal{S} \setminus \mathcal{S}(\mathbf{s}_0)$ are constrained to zero. Because the feasible set of local unbiased weights is a constrained subset of the global feasible set $\{\boldsymbol{\lambda} \in \mathbb{R}^n : \mathbf{1}'\boldsymbol{\lambda} = 1\}$, the minimum mean squared prediction error over the global set is bounded above by the minimum over the local subset:
$$
\sigma_{\text{OK, global}}^2(\mathbf{s}_0) \le \sigma_{\text{OK, local}}^2(\mathbf{s}_0)
$$
Thus, local neighborhood kriging always yields a prediction error variance greater than or equal to that of global kriging.

#### Proposition 5.5.2 (Discontinuity Across Search Boundaries)
Let $\mathcal{B} \subset D$ denote the geometric boundary across which the composition of the local neighborhood $\mathcal{S}(\mathbf{s}_0)$ changes (e.g., an observation point enters or exits the search radius).
Then:
1. Across $\mathcal{B}$, as an observation site enters or exits the active neighborhood set, its kriging weight generically jumps discontinuously between $0$ and a non-zero value $\lambda^* \ne 0$. (The only exception occurs in the knife-edge scenario where screening or geometric symmetry happens to assign exactly zero weight to that site at the transition boundary.)
2. Consequently, the local kriging predicted surface $\mathbf{s}_0 \mapsto \widehat{Z}_{\text{local}}(\mathbf{s}_0)$ and the local kriging variance surface $\mathbf{s}_0 \mapsto \sigma_{\text{local}}^2(\mathbf{s}_0)$ exhibit artificial step-discontinuities across neighborhood transition boundaries.

#### Beat 2: Algorithmic Translation (The "How")

The following dependency-free Python implementation provides both quadrant-aware neighborhood selection to counter directional clustering (shadowing) and local Ordinary Kriging solving the $(n_0 + 1) \times (n_0 + 1)$ bordered system via Gaussian elimination with partial pivoting.

```python
def select_moving_neighborhood(sample_coords, target_coord, max_neighbors=16, 
                                max_per_quadrant=None, search_radius=None):
    """
    Selects a moving spatial neighborhood of sample coordinates around target_coord,
    with optional directional quadrant balancing to prevent directional shadowing.

    Parameters:
    -----------
    sample_coords : list of tuple of float
        Observation locations [(x1, y1), ...].
    target_coord : tuple of float
        Target location (x0, y0).
    max_neighbors : int
        Total maximum number of neighbors n_0 to retain.
    max_per_quadrant : int or None
        Maximum neighbors allowed per spatial quadrant (for 2D coords).
    search_radius : float or None
        Maximum allowable distance threshold.

    Returns:
    --------
    selected_indices : list of int
        Indices of selected sample coordinates.
    """
    x0, y0 = target_coord[0], target_coord[1]
    candidates = []

    for idx, (xi, yi) in enumerate(sample_coords):
        dx = xi - x0
        dy = yi - y0
        dist = (dx**2 + dy**2) ** 0.5
        if search_radius is not None and dist > search_radius:
            continue
        
        # Determine quadrant: 0: NE (+,+), 1: NW (-,+), 2: SW (-,-), 3: SE (+,-)
        if dx >= 0 and dy >= 0:
            quad = 0
        elif dx < 0 and dy >= 0:
            quad = 1
        elif dx < 0 and dy < 0:
            quad = 2
        else:
            quad = 3
        candidates.append((dist, quad, idx))

    # Sort by distance
    candidates.sort(key=lambda item: item[0])

    if max_per_quadrant is None:
        return [idx for _, _, idx in candidates[:max_neighbors]]

    quad_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    selected_indices = []
    for dist, quad, idx in candidates:
        if len(selected_indices) >= max_neighbors:
            break
        if quad_counts[quad] < max_per_quadrant:
            selected_indices.append(idx)
            quad_counts[quad] += 1

    return selected_indices


def local_ordinary_kriging_predict(sample_coords, sample_vals, target_coord,
                                   variogram_kernel, max_neighbors=16, 
                                   max_per_quadrant=None, search_radius=None):
    """
    Performs Local Ordinary Kriging prediction at a target location using
    a moving neighborhood selection.

    Parameters:
    -----------
    sample_coords : list of tuples of float
        Spatial coordinates of sample observations.
    sample_vals : list of float
        Observed response values at sample locations.
    target_coord : tuple of float
        Prediction coordinate s_0.
    variogram_kernel : callable
        Semivariogram function gamma(h).
    max_neighbors : int
        Maximum number of local neighbors n_0 to include.
    max_per_quadrant : int or None
        Maximum neighbors per quadrant (octant in 3D) to avoid shadowing.
    search_radius : float or None
        Maximum distance threshold; points beyond this distance are excluded.

    Returns:
    --------
    z_hat : float
        Local Ordinary Kriging prediction at s_0.
    ok_var : float
        Local Ordinary Kriging prediction error variance.
    neighbor_indices : list of int
        Indices of the sample points selected in the local neighborhood.
    weights : list of float
        Ordinary Kriging weights assigned to the selected neighbors.
    """
    d = len(target_coord)

    # Step 1: Select neighborhood
    neighbor_indices = select_moving_neighborhood(
        sample_coords, target_coord, max_neighbors=max_neighbors,
        max_per_quadrant=max_per_quadrant, search_radius=search_radius
    )

    if not neighbor_indices:
        raise ValueError("No sample points found within search radius.")

    n_0 = len(neighbor_indices)
    local_coords = [sample_coords[idx] for idx in neighbor_indices]
    local_vals = [sample_vals[idx] for idx in neighbor_indices]

    # Step 2: Build the local (n_0 + 1) x (n_0 + 1) augmented system
    m = n_0 + 1
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m

    for i in range(n_0):
        for j in range(n_0):
            hij = sum((local_coords[i][k] - local_coords[j][k]) ** 2 for k in range(d)) ** 0.5
            A[i][j] = variogram_kernel(hij)
        A[i][n_0] = 1.0
        A[n_0][i] = 1.0
        h0i = sum((target_coord[k] - local_coords[i][k]) ** 2 for k in range(d)) ** 0.5
        b[i] = variogram_kernel(h0i)

    A[n_0][n_0] = 0.0
    b[n_0] = 1.0

    gamma_0 = [b[i] for i in range(n_0)]

    # Step 3: Solve via Gaussian elimination with partial pivoting
    for p in range(m):
        max_row = p
        max_val = abs(A[p][p])
        for r in range(p + 1, m):
            if abs(A[r][p]) > max_val:
                max_val = abs(A[r][p])
                max_row = r
        A[p], A[max_row] = A[max_row], A[p]
        b[p], b[max_row] = b[max_row], b[p]

        pivot = A[p][p]
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError("Singular local kriging matrix.")

        for c in range(p, m):
            A[p][c] /= pivot
        b[p] /= pivot

        for r in range(m):
            if r != p:
                factor = A[r][p]
                for c in range(p, m):
                    A[r][c] -= factor * A[p][c]
                b[r] -= factor * b[p]

    weights = b[:n_0]
    m_ok = b[n_0]

    # Step 4: Compute prediction and variance
    z_hat = sum(weights[i] * local_vals[i] for i in range(n_0))
    ok_var = sum(weights[i] * gamma_0[i] for i in range(n_0)) + m_ok
    ok_var = max(0.0, ok_var)

    return z_hat, ok_var, neighbor_indices, weights
```

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Computational Complexity and Speedup:** Global Ordinary Kriging requires factorizing an $n \times n$ matrix once ($O(n^3)$ operations, or $\approx \frac{1}{3} n^3$ FLOPs) and solving for $M$ grid points ($O(M n^2)$). When $n = 50{,}000$, storing a full double-precision covariance matrix requires $2.5 \times 10^9 \times 8 \text{ bytes} \approx 20\text{ GB}$ of RAM, and computing its Cholesky factorization consumes over $4 \times 10^{13}$ FLOPs—impractical on a standard laptop and memory-intensive even on a large workstation. In contrast, local neighborhood kriging solves $M$ independent small systems of dimension $(n_0 + 1)$, requiring $M \times O(n_0^3)$ operations. With $n_0 = 32$, $(n_0 + 1)^3 \approx 3.6 \times 10^4$ operations per grid site, making evaluation effortlessly parallelizable and memory-light.
- **Edge Seam Discontinuities:** Proposition 5.5.2 reveals that local kriging maps are not mathematically smooth everywhere: as target $\mathbf{s}_0$ moves continuously, sample points cross into or out of the local search neighborhood, causing the assigned weights to jump discontinuously. This generates subtle "patchwork" or "seam" lines in the interpolated surface and variance contours. In high-precision mapping, increasing $n_0$ (e.g., from 16 to 48) or smoothing neighborhood transitions suppresses these artifacts.
- **Directional Shadowing and Octant Search:** In clustered sampling designs (such as boreholes drilled along linear exploration transects), simple Euclidean distance search gathers points strictly from the nearest cluster, leaving other directions unrepresented. Enforcing quadrant (2D) or octant (3D) search limits the number of points per directional sector (e.g., at most 4 points per quadrant), ensuring omnidirectional angular coverage and preventing severe extrapolation bias.
- **Covariance Tapering as a Seamless Global Alternative:** To eliminate seam discontinuities while preserving computational efficiency, modern spatial statistics employs **covariance tapering** (Furrer, Genton, & Nychka, 2006). A compactly supported positive definite Wendland polynomial kernel $K(h)$ tapers the true spatial covariance: $C_{\text{taper}}(h) = C(h) K(h/r)$. By the **Schur Product Theorem**, the elementwise product of two positive semidefinite covariance matrices is guaranteed to be positive semidefinite, so $C_{\text{taper}}(h)$ is a valid covariance function. Because $C_{\text{taper}}(h) = 0$ for $h \ge r$, the resulting global covariance matrix $\mathbf{V}_{\text{taper}}$ is mathematically sparse and positive definite. Direct sparse Cholesky factorization achieves typical $O(n^{3/2})$ scaling on planar 2D problems with a good fill-reducing ordering, entirely eliminating local neighborhood boundary jumps.

---

### Key Ideas and Takeaways: Spatial Prediction (BLUP)

The following reference table synthesizes the core mathematical formulations, properties, and practical diagnostics established across Chapter 5:

| Property / Feature | Simple Kriging (SK) | Ordinary Kriging (OK) |
|---|---|---|
| **Mean Assumption** | Known constant mean $\mu$ (or stationary residual with $\mu = 0$) | Unknown constant mean $\mu$ |
| **Linear Predictor** | $\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \mu\mathbf{1})$ | $\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \boldsymbol{\lambda}_{\text{OK}}'\mathbf{Z} = \widehat{\mu}_{\text{GLS}} + \mathbf{c}_0'\mathbf{V}^{-1}(\mathbf{Z} - \widehat{\mu}_{\text{GLS}}\mathbf{1})$ |
| **Normal Equations** | $\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0$ | $\begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix}\begin{bmatrix} \boldsymbol{\lambda}_{\text{OK}} \\ -m_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}$ (Covariance) or $\begin{bmatrix} \boldsymbol{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix}\begin{bmatrix} \boldsymbol{\lambda}_{\text{OK}} \\ m_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}$ |
| **Prediction Variance** | $\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0$ | $\sigma_{\text{OK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \frac{(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} = \boldsymbol{\lambda}_{\text{OK}}'\boldsymbol{\gamma}_0 + m_{\text{OK}}$ |
| **Relative Precision** | Minimizes MSPE over all linear predictors | $\sigma_{\text{OK}}^2(\mathbf{s}_0) \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)$ (penalty for estimating $\mu$) |
| **Data Interpolation** | Exact interpolator if nugget $\tau^2 = 0$: $\widehat{Z}(\mathbf{s}_i) = Z(\mathbf{s}_i)$ | Exact interpolator if nugget $\tau^2 = 0$: $\widehat{Z}(\mathbf{s}_i) = Z(\mathbf{s}_i)$ |
| **Screening Effect** | **Exact screening** in 1D Markov/exponential fields | **Approximate screening** (flanking points take primary weight; outer points take small weights to estimate $\mu$) |
| **Negative Weights** | Possible under smooth kernels ($C \in C^2$, Gaussian, Matérn $\nu > 1$) | Possible under smooth kernels and geometric clustering |
| **Fast Leave-One-Out** | $e_{(-i)} = [\mathbf{V}^{-1}(\mathbf{Z}-\mu\mathbf{1})]_i / [\mathbf{V}^{-1}]_{ii}$, $\operatorname{Var}(e_{(-i)}) = 1/[\mathbf{V}^{-1}]_{ii}$ | Analogous block-inverse formula using bordered system |
| **Local vs. Global** | Nested subspaces: $\sigma_{\text{SK, local}}^2 \ge \sigma_{\text{SK, global}}^2$ | Nested constraint sets: $\sigma_{\text{OK, local}}^2 \ge \sigma_{\text{OK, global}}^2$ |

---

### 5.6 Analytical Exercises with Step-by-Step Complete Solutions

#### Exercise 5.1 (Exact Markovian Screening in One Dimension)
Let $\{Z(t) : t \in \mathbb{R}\}$ be a stationary Gaussian random process on the line with mean zero and Exponential covariance:
$$
C(h) = \sigma^2 \exp\left(-\frac{|h|}{\phi}\right), \quad \phi > 0
$$
Suppose observations are obtained at three ordered locations $t_1 < t_2 < t_3$, and we predict $Z(t_0)$ at an interior point $t_0 \in (t_1, t_2)$.
- **(a)** Formulate the $3 \times 3$ Simple Kriging normal equations $\mathbf{V} \boldsymbol{\lambda} = \mathbf{v}_0$.
- **(b)** Perform explicit Gaussian elimination or block inversion to solve for $\lambda_1, \lambda_2, \lambda_3$.
- **(c)** Prove analytically that $\lambda_3 \equiv 0$, confirming that $t_2$ completely screens $t_3$ from $t_0$.

*Solution.*
- **(a)** Without loss of generality, scale $\sigma^2 = 1$. Let the coordinate distances be:
  $$
  d_1 = t_0 - t_1 > 0, \quad d_2 = t_2 - t_0 > 0, \quad \Delta_{12} = t_2 - t_1 = d_1 + d_2, \quad \Delta_{23} = t_3 - t_2 > 0
  $$
  Then $t_3 - t_1 = \Delta_{12} + \Delta_{23}$, and $t_3 - t_0 = d_2 + \Delta_{23}$. Define the correlation factors:
  $$
  \rho_1 = e^{-d_1/\phi}, \quad \rho_2 = e^{-d_2/\phi}, \quad a = e^{-\Delta_{12}/\phi} = \rho_1 \rho_2, \quad b = e^{-\Delta_{23}/\phi}
  $$
  The covariance matrix $\mathbf{V}$ and cross-covariance vector $\mathbf{v}_0$ are:
  $$
  \mathbf{V} = \begin{bmatrix} 1 & a & ab \\ a & 1 & b \\ ab & b & 1 \end{bmatrix}, \quad \mathbf{v}_0 = \begin{bmatrix} \rho_1 \\ \rho_2 \\ \rho_2 b \end{bmatrix}
  $$
  Notice that $C(t_3 - t_0) = e^{-(d_2 + \Delta_{23})/\phi} = e^{-d_2/\phi} e^{-\Delta_{23}/\phi} = \rho_2 b$.

- **(b)** The system of linear equations is:
  $$
  \begin{cases}
  (1) & \lambda_1 + a \lambda_2 + ab \lambda_3 = \rho_1 \\
  (2) & a \lambda_1 + \lambda_2 + b \lambda_3 = \rho_2 \\
  (3) & ab \lambda_1 + b \lambda_2 + \lambda_3 = \rho_2 b
  \end{cases}
  $$
  Multiply equation (2) by $b$:
  $$
  ab \lambda_1 + b \lambda_2 + b^2 \lambda_3 = \rho_2 b
  $$
  Subtract this from equation (3):
  $$
  (ab \lambda_1 - ab \lambda_1) + (b \lambda_2 - b \lambda_2) + (1 - b^2) \lambda_3 = \rho_2 b - \rho_2 b = 0
  $$
  This simplifies to:
  $$
  (1 - b^2) \lambda_3 = 0
  $$

- **(c)** Because $t_3 > t_2$, we have $\Delta_{23} > 0$, so $b = e^{-\Delta_{23}/\phi} < 1$, which implies $1 - b^2 > 0$.
  Dividing by $(1 - b^2)$ yields:
  $$
  \lambda_3 \equiv 0
  $$
  Since $\lambda_3 = 0$, equations (1) and (2) reduce to the $2 \times 2$ system:
  $$
  \begin{bmatrix} 1 & a \\ a & 1 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \end{bmatrix} = \begin{bmatrix} \rho_1 \\ \rho_2 \end{bmatrix}
  $$
  Inverting the $2 \times 2$ matrix:
  $$
  \begin{bmatrix} \lambda_1 \\ \lambda_2 \end{bmatrix} = \frac{1}{1 - a^2} \begin{bmatrix} 1 & -a \\ -a & 1 \end{bmatrix} \begin{bmatrix} \rho_1 \\ \rho_2 \end{bmatrix} = \frac{1}{1 - a^2} \begin{bmatrix} \rho_1 - a \rho_2 \\ \rho_2 - a \rho_1 \end{bmatrix}
  $$
  Substituting $a = \rho_1 \rho_2$:
  $$
  \lambda_1 = \frac{\rho_1 - \rho_1 \rho_2^2}{1 - \rho_1^2 \rho_2^2} = \frac{\rho_1(1 - \rho_2^2)}{1 - (\rho_1 \rho_2)^2}, \quad \lambda_2 = \frac{\rho_2 - \rho_2 \rho_1^2}{1 - \rho_1^2 \rho_2^2} = \frac{\rho_2(1 - \rho_1^2)}{1 - (\rho_1 \rho_2)^2}
  $$
  This confirms analytically that $t_2$ completely screens $t_3$, so $\lambda_3 \equiv 0$. $\blacksquare$

---

#### Exercise 5.2 (Analytical Derivation of Two-Point Ordinary Kriging System)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an intrinsically stationary random field with semivariogram $\gamma(h)$. Suppose observations are obtained at two locations $\mathbf{s}_1$ and $\mathbf{s}_2$ with separation $h_{12} = \|\mathbf{s}_1 - \mathbf{s}_2\|$, and we predict at location $\mathbf{s}_0$ with distances $h_{01} = \|\mathbf{s}_0 - \mathbf{s}_1\|$ and $h_{02} = \|\mathbf{s}_0 - \mathbf{s}_2\|$.
- **(a)** Write down the complete $3 \times 3$ Ordinary Kriging system in semivariogram form.
- **(b)** Solve for the weights $\lambda_1, \lambda_2$ and the Lagrange multiplier $m_{\text{OK}}$.
- **(c)** Derive the exact expression for the prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$.
- **(d)** Specialize to the midpoint $\mathbf{s}_0 = (\mathbf{s}_1 + \mathbf{s}_2)/2$ under an isotropic power variogram $\gamma(h) = \theta h^\alpha$ ($0 < \alpha < 2$), and evaluate the variance as $\alpha \to 1$ and $\alpha \to 2$. Explain the significance of the $\alpha \le 2$ boundary for conditional negative definiteness.

*Solution.*
- **(a)** The Ordinary Kriging semivariogram system is:
  $$
  \begin{bmatrix} 0 & \gamma(h_{12}) & 1 \\ \gamma(h_{12}) & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \\ m_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \gamma(h_{01}) \\ \gamma(h_{02}) \\ 1 \end{bmatrix}
  $$

- **(b)** From the constraint equation, $\lambda_1 + \lambda_2 = 1 \implies \lambda_2 = 1 - \lambda_1$.
  The first two equations are:
  $$
  \begin{cases}
  \lambda_2 \gamma(h_{12}) + m_{\text{OK}} = \gamma(h_{01}) \\
  \lambda_1 \gamma(h_{12}) + m_{\text{OK}} = \gamma(h_{02})
  \end{cases}
  $$
  Subtracting the second equation from the first:
  $$
  (\lambda_2 - \lambda_1) \gamma(h_{12}) = \gamma(h_{01}) - \gamma(h_{02}) \implies (1 - 2\lambda_1) \gamma(h_{12}) = \gamma(h_{01}) - \gamma(h_{02})
  $$
  Solving for $\lambda_1$ and $\lambda_2$:
  $$
  \lambda_1 = \frac{1}{2} + \frac{\gamma(h_{02}) - \gamma(h_{01})}{2 \gamma(h_{12})}
  $$
  $$
  \lambda_2 = 1 - \lambda_1 = \frac{1}{2} + \frac{\gamma(h_{01}) - \gamma(h_{02})}{2 \gamma(h_{12})}
  $$
  Adding the two equations yields:
  $$
  (\lambda_1 + \lambda_2) \gamma(h_{12}) + 2m_{\text{OK}} = \gamma(h_{01}) + \gamma(h_{02}) \implies m_{\text{OK}} = \frac{\gamma(h_{01}) + \gamma(h_{02}) - \gamma(h_{12})}{2}
  $$

- **(c)** By Theorem 5.3.3, the prediction variance is $\sigma_{\text{OK}}^2(\mathbf{s}_0) = \lambda_1 \gamma(h_{01}) + \lambda_2 \gamma(h_{02}) + m_{\text{OK}}$:
  $$
  \begin{aligned}
  \sigma_{\text{OK}}^2(\mathbf{s}_0) &= \left(\frac{1}{2} + \frac{\gamma(h_{02}) - \gamma(h_{01})}{2\gamma(h_{12})}\right) \gamma(h_{01}) + \left(\frac{1}{2} + \frac{\gamma(h_{01}) - \gamma(h_{02})}{2\gamma(h_{12})}\right) \gamma(h_{02}) + \frac{\gamma(h_{01}) + \gamma(h_{02}) - \gamma(h_{12})}{2} \\
  &= \frac{\gamma(h_{01}) + \gamma(h_{02})}{2} - \frac{(\gamma(h_{01}) - \gamma(h_{02}))^2}{2\gamma(h_{12})} + \frac{\gamma(h_{01}) + \gamma(h_{02}) - \gamma(h_{12})}{2} \\
  &= \gamma(h_{01}) + \gamma(h_{02}) - \frac{\gamma(h_{12})}{2} - \frac{(\gamma(h_{01}) - \gamma(h_{02}))^2}{2\gamma(h_{12})}
  \end{aligned}
  $$

- **(d)** At the midpoint $\mathbf{s}_0 = (\mathbf{s}_1 + \mathbf{s}_2)/2$, symmetry dictates $h_{01} = h_{02} = h_{12}/2$.
  Then $\gamma(h_{01}) = \gamma(h_{02})$, so $\lambda_1 = \lambda_2 = 1/2$. The squared difference term vanishes:
  $$
  \sigma_{\text{OK}}^2(\mathbf{s}_0) = 2 \gamma\left(\frac{h_{12}}{2}\right) - \frac{\gamma(h_{12})}{2}
  $$
  For the power variogram $\gamma(h) = \theta h^\alpha$:
  $$
  \sigma_{\text{OK}}^2(\mathbf{s}_0) = 2 \theta \left(\frac{h_{12}}{2}\right)^\alpha - \frac{1}{2} \theta h_{12}^\alpha = \theta h_{12}^\alpha \left( 2^{1-\alpha} - \frac{1}{2} \right)
  $$
  - As $\alpha \to 1$ (linear variogram / Brownian motion):
    $$
    \sigma_{\text{OK}}^2(\mathbf{s}_0) = \theta h_{12} \left(2^0 - \frac{1}{2}\right) = \frac{1}{2} \theta h_{12}
    $$
  - As $\alpha \to 2$ (parabolic variogram, smooth differentiable field):
    $$
    2^{1-2} - \frac{1}{2} = 2^{-1} - \frac{1}{2} = 0 \implies \sigma_{\text{OK}}^2(\mathbf{s}_0) \to 0
    $$
    The prediction error variance vanishes because $\alpha = 2$ corresponds to a degenerate process whose realizations are almost surely straight lines ($Z(s) = A + B s$); thus the midpoint is determined exactly by linear interpolation without error.
  - **Validity boundary:** As established in Chapter 2, the power variogram $\gamma(h) = \theta h^\alpha$ is conditionally negative definite if and only if $0 < \alpha \le 2$. For $\alpha > 2$, $\gamma(h)$ is no longer conditionally negative definite, and kriging variance formulas can produce negative variances, violating the fundamental properties of a metric space. $\blacksquare$

---

#### Exercise 5.3 (Geometric Origins of Negative Kriging Weights)
Consider three collinear points on the real line: $s_1 = -1$, $s_2 = 0$, $s_3 = 1$. Let the target point be $s_0 = 2$, so that $s_0$ lies outside the convex hull of the data (extrapolation). Suppose the random field is stationary with covariance kernel $C(h)$.
- **(a)** Set up the Ordinary Kriging system for predicting $Z(s_0)$.
- **(b)** Let $C(h) = \exp(-|h|)$. Compute the analytical weights $\lambda_1, \lambda_2, \lambda_3$ and evaluate them numerically. Explain why this configuration does not exhibit screening.
- **(c)** Replace the covariance with the Gaussian covariance $C(h) = \exp(-h^2)$. Verify which weight becomes negative, and explain what the weight signs reveal about gradient/curvature extrapolation.

*Solution.*
- **(a)** The distances among points are:
  - Data-to-data: $|s_1 - s_2| = 1$, $|s_2 - s_3| = 1$, $|s_1 - s_3| = 2$.
  - Target-to-data: $|s_0 - s_3| = 1$, $|s_0 - s_2| = 2$, $|s_0 - s_1| = 3$.
  The covariance matrix $\mathbf{V}$ and cross-covariance vector $\mathbf{c}_0$ are:
  $$
  \mathbf{V} = \begin{bmatrix} C(0) & C(1) & C(2) \\ C(1) & C(0) & C(1) \\ C(2) & C(1) & C(0) \end{bmatrix}, \quad \mathbf{c}_0 = \begin{bmatrix} C(3) \\ C(2) \\ C(1) \end{bmatrix}
  $$
  The Ordinary Kriging system in covariance form is:
  $$
  \begin{bmatrix} C(0) & C(1) & C(2) & 1 \\ C(1) & C(0) & C(1) & 1 \\ C(2) & C(1) & C(0) & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ -m_{\text{OK}} \end{bmatrix} = \begin{bmatrix} C(3) \\ C(2) \\ C(1) \\ 1 \end{bmatrix}
  $$

- **(b)** For $C(h) = \exp(-|h|)$, let $\rho = e^{-1} \approx 0.367879$. Then $C(0) = 1$, $C(1) = \rho$, $C(2) = \rho^2$, $C(3) = \rho^3$.
  The system is:
  $$
  \begin{cases}
  \lambda_1 + \rho \lambda_2 + \rho^2 \lambda_3 - m_{\text{OK}} = \rho^3 \\
  \rho \lambda_1 + \lambda_2 + \rho \lambda_3 - m_{\text{OK}} = \rho^2 \\
  \rho^2 \lambda_1 + \rho \lambda_2 + \lambda_3 - m_{\text{OK}} = \rho \\
  \lambda_1 + \lambda_2 + \lambda_3 = 1
  \end{cases}
  $$
  Subtracting equation (2) from equation (1):
  $$
  (1 - \rho)\lambda_1 - (1 - \rho)\lambda_2 + \rho(\rho - 1)\lambda_3 = \rho^2(\rho - 1) \implies \lambda_1 - \lambda_2 - \rho \lambda_3 = -\rho^2
  $$
  Subtracting equation (2) from equation (3):
  $$
  \rho(\rho - 1)\lambda_1 - (1 - \rho)\lambda_2 + (1 - \rho)\lambda_3 = \rho(1 - \rho) \implies -\rho \lambda_1 - \lambda_2 + \lambda_3 = \rho
  $$
  Subtracting the first reduced equation from the second:
  $$
  (1 + \rho)(\lambda_3 - \lambda_1) = \rho(1 + \rho) \implies \lambda_3 - \lambda_1 = \rho \implies \lambda_3 = \lambda_1 + \rho
  $$
  Substituting into $\lambda_2 = \lambda_1 - \rho \lambda_3 + \rho^2$:
  $$
  \lambda_2 = \lambda_1 - \rho(\lambda_1 + \rho) + \rho^2 = (1 - \rho)\lambda_1
  $$
  Substituting into $\lambda_1 + \lambda_2 + \lambda_3 = 1$:
  $$
  \lambda_1 + (1 - \rho)\lambda_1 + (\lambda_1 + \rho) = 1 \implies (3 - \rho)\lambda_1 = 1 - \rho \implies \lambda_1 = \frac{1 - \rho}{3 - \rho}
  $$
  Thus:
  $$
  \lambda_1 = \frac{1 - \rho}{3 - \rho} \approx \frac{0.632121}{2.632121} \approx +0.2402
  $$
  $$
  \lambda_2 = \frac{(1 - \rho)^2}{3 - \rho} \approx \frac{(0.632121)^2}{2.632121} \approx +0.1518
  $$
  $$
  \lambda_3 = \lambda_1 + \rho \approx 0.2402 + 0.3679 \approx +0.6080
  $$
  All weights are strictly positive under the Exponential covariance: $(\lambda_1, \lambda_2, \lambda_3) \approx (0.2402, 0.1518, 0.6080)$.
  *Why is this not screening?* In Simple Kriging, $s_3$ would completely screen $s_2$ and $s_1$ during extrapolation, giving $\boldsymbol{\lambda}_{\text{SK}} = (0, 0, \rho) = (0, 0, 0.3679)$. In Ordinary Kriging, however, because the mean $\mu$ is unknown, the system must enforce $\sum \lambda_i = 1$ to remain unbiased. It allocates positive weights to $s_1$ and $s_2$ because they contribute valuable information toward estimating the unknown mean $\mu$ across the entire dataset.

- **(c)** Now let $C(h) = \exp(-h^2)$. The covariances are:
  $$
  C(0) = 1, \quad C(1) = e^{-1} \approx 0.367879, \quad C(2) = e^{-4} \approx 0.018316, \quad C(3) = e^{-9} \approx 0.000123
  $$
  Solving the augmented system numerically yields:
  $$
  \lambda_1 \approx +0.3285, \quad \lambda_2 \approx -0.0316, \quad \lambda_3 \approx +0.7031, \quad m_{\text{OK}} \approx -0.3296
  $$
  Notice that $\lambda_2 = -0.0316 < 0$ is strictly negative, while the weights sum exactly to $\sum \lambda_i = 0.3285 - 0.0316 + 0.7031 = 1.0$.
  *Geometric and physical interpretation:* With a smooth, infinitely differentiable Gaussian kernel, Ordinary Kriging assigns a small negative weight to the intermediate point $s_2$ relative to $s_3$ to estimate the local slope and curvature of the realization leading out toward $s_0$. The closest point $s_3$ receives the majority weight ($0.7031$), $s_2$ receives a negative weight (acting as a finite-difference contrast against $s_3$), and the distant point $s_1$ retains positive weight ($0.3285$) to satisfy the unbiasedness constraint $\sum \lambda_i = 1$ and anchor the estimate of $\mu$. $\blacksquare$

---

#### Exercise 5.4 (Variance Deficit and Low-Pass Filtering)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^2\}$ be a second-order stationary isotropic Gaussian random field with mean $\mu = 0$, variance $\sigma^2 = 1$, and covariance $C(\mathbf{h}) = \exp(-\|\mathbf{h}\|)$. Suppose observations are taken on a regular square lattice with grid spacing $\Delta$:
$$
\mathcal{S} = \{(i\Delta, j\Delta)' : i, j \in \mathbb{Z}\}
$$
Let $\mathbf{s}_0 = (\Delta/2, \Delta/2)'$ be the center of a grid cell.
- **(a)** Express the Simple Kriging variance $\sigma_{\text{SK}}^2(\mathbf{s}_0)$ as a function of grid spacing $\Delta$ using the 4 nearest grid vertices.
- **(b)** Compute the ratio of the predictor's variance to the true process variance:
  $$
  R(\Delta) = \frac{\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))}{\operatorname{Var}(Z(\mathbf{s}_0))}
  $$
- **(c)** Evaluate the asymptotic behavior of $R(\Delta)$ as $\Delta \to 0$ (dense infill) and as $\Delta \to \infty$ (sparse data).

*Solution.*
- **(a)** The four closest grid vertices to $\mathbf{s}_0 = (\Delta/2, \Delta/2)'$ are:
  $$
  \mathbf{s}_1 = (0, 0)', \quad \mathbf{s}_2 = (\Delta, 0)', \quad \mathbf{s}_3 = (\Delta, \Delta)', \quad \mathbf{s}_4 = (0, \Delta)'
  $$
  The distance from $\mathbf{s}_0$ to each vertex is:
  $$
  \|\mathbf{s}_0 - \mathbf{s}_i\| = \sqrt{\left(\frac{\Delta}{2}\right)^2 + \left(\frac{\Delta}{2}\right)^2} = \frac{\Delta}{\sqrt{2}}, \quad i = 1, 2, 3, 4
  $$
  The cross-covariance vector is $\mathbf{c}_0 = c_0 \mathbf{1}$, where $c_0 = \exp(-\Delta/\sqrt{2})$.
  The pairwise distances among the four vertices are:
  - 4 edges of length $\Delta$, with covariance $\rho_1 = \exp(-\Delta)$.
  - 2 diagonals of length $\sqrt{2}\Delta$, with covariance $\rho_2 = \exp(-\sqrt{2}\Delta)$.
  The $4 \times 4$ covariance matrix is:
  $$
  \mathbf{V} = \begin{bmatrix} 1 & \rho_1 & \rho_2 & \rho_1 \\ \rho_1 & 1 & \rho_1 & \rho_2 \\ \rho_2 & \rho_1 & 1 & \rho_1 \\ \rho_1 & \rho_2 & \rho_1 & 1 \end{bmatrix}
  $$
  By four-fold rotational symmetry, the all-ones vector $\mathbf{1}$ is an eigenvector of $\mathbf{V}$ with eigenvalue:
  $$
  \kappa = 1 + 2\rho_1 + \rho_2 = 1 + 2\exp(-\Delta) + \exp(-\sqrt{2}\Delta)
  $$
  Therefore $\mathbf{V}^{-1} \mathbf{1} = \frac{1}{\kappa} \mathbf{1}$. The Simple Kriging weight vector is:
  $$
  \boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1} \mathbf{c}_0 = c_0 \mathbf{V}^{-1} \mathbf{1} = \frac{c_0}{\kappa} \mathbf{1} = \frac{\exp(-\Delta/\sqrt{2})}{1 + 2\exp(-\Delta) + \exp(-\sqrt{2}\Delta)} \mathbf{1}
  $$
  The Simple Kriging prediction variance is:
  $$
  \sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{c}_0' \mathbf{V}^{-1} \mathbf{c}_0 = 1 - c_0^2 \mathbf{1}' \left(\frac{1}{\kappa} \mathbf{1}\right) = 1 - \frac{4 c_0^2}{\kappa} = 1 - \frac{4 \exp(-\sqrt{2}\Delta)}{1 + 2\exp(-\Delta) + \exp(-\sqrt{2}\Delta)}
  $$

- **(b)** By Theorem 5.2.1, the variance of the predictor is:
  $$
  \operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \sigma_{\text{SK}}^2(\mathbf{s}_0) = \frac{4 \exp(-\sqrt{2}\Delta)}{1 + 2\exp(-\Delta) + \exp(-\sqrt{2}\Delta)}
  $$
  Since $\operatorname{Var}(Z(\mathbf{s}_0)) = C(\mathbf{0}) = 1$, the variance ratio is:
  $$
  R(\Delta) = \frac{\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))}{\operatorname{Var}(Z(\mathbf{s}_0))} = \frac{4 \exp(-\sqrt{2}\Delta)}{1 + 2\exp(-\Delta) + \exp(-\sqrt{2}\Delta)}
  $$

- **(c)** As $\Delta \to 0$ (dense infill sampling):
  $$
  \exp(-\Delta) \to 1, \quad \exp(-\sqrt{2}\Delta) \to 1
  $$
  The numerator tends to $4(1) = 4$, and the denominator tends to $1 + 2(1) + 1 = 4$. Thus:
  $$
  \lim_{\Delta \to 0} R(\Delta) = \frac{4}{4} = 1
  $$
  In the dense infill limit, the prediction error variance vanishes ($\sigma_{\text{SK}}^2(\mathbf{s}_0) \to 0$), and the predictor preserves 100% of the true field variance: no smoothing deficit occurs.

  As $\Delta \to \infty$ (sparse data limit):
  The exponential terms decay to zero:
  $$
  \lim_{\Delta \to \infty} R(\Delta) = \lim_{\Delta \to \infty} \frac{4 \exp(-\sqrt{2}\Delta)}{1 + 0 + 0} = 0
  $$
  In the sparse data limit, the observations carry zero information regarding $\mathbf{s}_0$. The predictor reverts to the constant mean $\mu = 0$, which has zero variance. The variance deficit is 100%, and the prediction error variance equals the total sill $\sigma_{\text{SK}}^2(\mathbf{s}_0) \to 1$. $\blacksquare$

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 5: Spatial Prediction and Kriging Theory (BLUP) — Companion Exercises and Complete Solutions](ch05_exercises.md)
