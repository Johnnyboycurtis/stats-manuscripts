# Chapter 5: Best Linear Unbiased Prediction (BLUP) Framework

This chapter establishes the functional-analytic and statistical foundations of optimal spatial prediction, known in geostatistics as *kriging*. Spatial prediction addresses the challenge of estimating an unobserved continuous physical field $Z(\mathbf{s}_0)$ at an unsampled location $\mathbf{s}_0 \in D \subset \mathbb{R}^d$ from a discrete set of spatially autocorrelated observations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$.

Following the pedagogical philosophy of Maria L. Rizzo and the semantic linear-model discipline of Oliver Schabenberger & Carol A. Gotway, we develop the theory of spatial prediction through the coordinated Three-Beat pedagogical framework. We formulate spatial prediction within the geometry of the Hilbert space $L^2(\Omega, \mathcal{F}, P)$, derive Simple Kriging (SK) via the Orthogonal Projection Theorem, construct Ordinary Kriging (OK) under an unknown constant mean via constrained optimization, analyze the variance deficit and low-pass filtering properties, and dissect the geometric phenomena of spatial screening and negative kriging weights.

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

### 5.1 Hilbert Space Geometry and the Orthogonal Projection Theorem

#### Beat 1: The Mathematical Anchor (The "Why")

In classical mathematical statistics, estimation typically focuses on unknown deterministic parameters $\boldsymbol{\theta} \in \Theta$. In spatial geostatistics, the primary target of inference is a random variable $Z(\mathbf{s}_0)$ evaluated at an unobserved geographic location $\mathbf{s}_0 \in D$. The fundamental question is: *What constitutes an optimal predictor of an unobserved random variable, and within what mathematical space does this optimization occur?*

The approximation theory of square-integrable random variables provides the answer. Let $(\Omega, \mathcal{F}, P)$ be a complete probability space, and let $\mathcal{H} = L^2(\Omega, \mathcal{F}, P)$ denote the real Hilbert space of equivalence classes of real-valued random variables with finite second moments:
$$
\operatorname{E}[|X|^2] = \int_\Omega |X(\omega)|^2 dP(\omega) < \infty
$$
The space $\mathcal{H}$ is equipped with the inner product:
$$
\langle X, Y \rangle = \operatorname{E}[XY] = \int_\Omega X(\omega) Y(\omega) dP(\omega)
$$
which induces the norm $\|X\|_{L^2} = \sqrt{\langle X, X \rangle} = \sqrt{\operatorname{E}[X^2]}$. The distance between two random variables $X, Y \in \mathcal{H}$ is the root-mean-square metric:
$$
d(X, Y) = \|X - Y\|_{L^2} = \sqrt{\operatorname{E}[(X - Y)^2]}
$$

#### Definition 5.1.1 (Spatial Prediction Setting)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a real-valued second-order spatial random field indexed by a spatial domain $D \subset \mathbb{R}^d$ ($d \ge 1$), such that $Z(\mathbf{s}) \in \mathcal{H}$ for all $\mathbf{s} \in D$. Suppose the field has been observed at $n$ distinct spatial locations:
$$
\mathbf{s}_1, \mathbf{s}_2, \dots, \mathbf{s}_n \in D, \quad \mathbf{s}_i \ne \mathbf{s}_j \text{ for } i \ne j
$$
yielding the random vector of observations:
$$
\mathbf{Z} = (Z(\mathbf{s}_1), Z(\mathbf{s}_2), \dots, Z(\mathbf{s}_n))' \in \mathbb{R}^n
$$
Let $\mathbf{s}_0 \in D$ denote an arbitrary target location (typically $\mathbf{s}_0 \notin \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$). The fundamental objective of spatial prediction is to construct an estimator $\widehat{Z}(\mathbf{s}_0)$ based on the data $\mathbf{Z}$ that minimizes the mean-squared prediction error (MSPE):
$$
\operatorname{MSPE}(\widehat{Z}(\mathbf{s}_0)) = \operatorname{E}\left[\left(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)\right)^2\right] = \|Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)\|_{L^2}^2
$$
over an admissible class of estimators $\mathcal{M} \subset \mathcal{H}$.

#### Theorem 5.1.1 (Orthogonal Projection Theorem in Hilbert Space)
Let $\mathcal{H}$ be a real Hilbert space and let $\mathcal{M}$ be a closed linear subspace of $\mathcal{H}$. Then:
1. For every element $X \in \mathcal{H}$, there exists a unique element $\widehat{X} \in \mathcal{M}$ such that:
   $$
   \|X - \widehat{X}\|_{L^2} = \inf_{Y \in \mathcal{M}} \|X - Y\|_{L^2}
   $$
2. A necessary and sufficient condition for $\widehat{X} \in \mathcal{M}$ to achieve this minimum is that the error vector $X - \widehat{X}$ is orthogonal to the subspace $\mathcal{M}$, denoted $(X - \widehat{X}) \perp \mathcal{M}$:
   $$
   \langle X - \widehat{X}, Y \rangle = 0, \quad \forall Y \in \mathcal{M}
   $$
3. The mapping $\Pi_{\mathcal{M}}: \mathcal{H} \to \mathcal{M}$ defined by $\Pi_{\mathcal{M}} X = \widehat{X}$ is a linear, self-adjoint, idempotent projection operator ($\Pi_{\mathcal{M}}^2 = \Pi_{\mathcal{M}}$) with operator norm $\|\Pi_{\mathcal{M}}\| = 1$ (if $\mathcal{M} \ne \{\mathbf{0}\}$), satisfying the Pythagorean identity:
   $$
   \|X\|_{L^2}^2 = \|\widehat{X}\|_{L^2}^2 + \|X - \widehat{X}\|_{L^2}^2
   $$

*Proof.*
1. **Existence and Uniqueness:** Let $d = \inf_{Y \in \mathcal{M}} \|X - Y\|_{L^2}$. By definition of the infimum, there exists a sequence $\{Y_k\}_{k=1}^\infty \subset \mathcal{M}$ such that $\lim_{k \to \infty} \|X - Y_k\|_{L^2} = d$. By the parallelogram law in inner product spaces:
   $$
   \|(X - Y_k) - (X - Y_m)\|_{L^2}^2 + \|(X - Y_k) + (X - Y_m)\|_{L^2}^2 = 2\|X - Y_k\|_{L^2}^2 + 2\|X - Y_m\|_{L^2}^2
   $$
   Simplifying the terms:
   $$
   \|Y_m - Y_k\|_{L^2}^2 + 4 \left\|X - \frac{Y_k + Y_m}{2}\right\|_{L^2}^2 = 2\|X - Y_k\|_{L^2}^2 + 2\|X - Y_m\|_{L^2}^2
   $$
   Since $\mathcal{M}$ is a linear subspace, $\frac{1}{2}(Y_k + Y_m) \in \mathcal{M}$, which implies $\|X - \frac{1}{2}(Y_k + Y_m)\|_{L^2} \ge d$. Thus:
   $$
   \|Y_m - Y_k\|_{L^2}^2 \le 2\|X - Y_k\|_{L^2}^2 + 2\|X - Y_m\|_{L^2}^2 - 4d^2 \xrightarrow{k,m \to \infty} 2d^2 + 2d^2 - 4d^2 = 0
   $$
   Hence $\{Y_k\}$ is a Cauchy sequence in $\mathcal{H}$. Because $\mathcal{H}$ is complete and $\mathcal{M}$ is closed, there exists a unique limit $\widehat{X} = \lim_{k \to \infty} Y_k \in \mathcal{M}$ satisfying $\|X - \widehat{X}\|_{L^2} = d$.
   
   If another element $\widetilde{X} \in \mathcal{M}$ also satisfies $\|X - \widetilde{X}\|_{L^2} = d$, applying the parallelogram law to $\widehat{X}$ and $\widetilde{X}$ yields $\|\widehat{X} - \widetilde{X}\|_{L^2}^2 \le 2d^2 + 2d^2 - 4d^2 = 0$, establishing uniqueness: $\widehat{X} = \widetilde{X}$.

2. **Orthogonality Condition:**
   - *Sufficiency:* Suppose $\langle X - \widehat{X}, Y \rangle = 0$ for all $Y \in \mathcal{M}$. For any $Y \in \mathcal{M}$, write $X - Y = (X - \widehat{X}) + (\widehat{X} - Y)$. Because $\widehat{X} - Y \in \mathcal{M}$, the cross-term vanishes:
     $$
     \|X - Y\|_{L^2}^2 = \|(X - \widehat{X}) + (\widehat{X} - Y)\|_{L^2}^2 = \|X - \widehat{X}\|_{L^2}^2 + \|\widehat{X} - Y\|_{L^2}^2 \ge \|X - \widehat{X}\|_{L^2}^2
     $$
     with equality if and only if $\|\widehat{X} - Y\|_{L^2} = 0 \iff Y = \widehat{X}$.
   - *Necessity:* Suppose $\widehat{X}$ minimizes $\|X - Y\|_{L^2}^2$ over $Y \in \mathcal{M}$. For any fixed $Y \in \mathcal{M}$ and any scalar $t \in \mathbb{R}$, $\widehat{X} + tY \in \mathcal{M}$. Define the quadratic real function:
     $$
     \phi(t) = \|X - (\widehat{X} + tY)\|_{L^2}^2 = \|X - \widehat{X}\|_{L^2}^2 - 2t \langle X - \widehat{X}, Y \rangle + t^2 \|Y\|_{L^2}^2
     $$
     Since $\phi(t)$ achieves its minimum at $t = 0$, its derivative at zero must vanish:
     $$
     \phi'(0) = -2 \langle X - \widehat{X}, Y \rangle = 0 \implies \langle X - \widehat{X}, Y \rangle = 0
     $$
     Because $Y \in \mathcal{M}$ was arbitrary, $(X - \widehat{X}) \perp \mathcal{M}$.

3. **Pythagorean Identity:** Setting $Y = \mathbf{0} \in \mathcal{M}$ in the orthogonal expansion $X = \widehat{X} + (X - \widehat{X})$ yields $\|X\|_{L^2}^2 = \|\widehat{X}\|_{L^2}^2 + \|X - \widehat{X}\|_{L^2}^2$. Linearity and idempotence follow from the uniqueness of the orthogonal projection. $\blacksquare$

#### Corollary 5.1.2 (Orthogonal Projection onto Closed Affine Subspaces)
Let $\mathcal{M}_{\text{aff}} = X_0 + \mathcal{M}_0$ be a closed affine subspace of $\mathcal{H}$, where $X_0 \in \mathcal{H}$ and $\mathcal{M}_0$ is a closed linear subspace of $\mathcal{H}$. Then for any $X \in \mathcal{H}$, there exists a unique $\widehat{X} \in \mathcal{M}_{\text{aff}}$ minimizing $\|X - Y\|_{L^2}$ over $Y \in \mathcal{M}_{\text{aff}}$, characterized by:
$$
\langle X - \widehat{X}, Y_0 \rangle = 0, \quad \forall Y_0 \in \mathcal{M}_0
$$
Equivalently, $\widehat{X} = X_0 + \Pi_{\mathcal{M}_0}(X - X_0)$.

*Proof.*
Any $Y \in \mathcal{M}_{\text{aff}}$ can be written as $Y = X_0 + Y_0$ for some $Y_0 \in \mathcal{M}_0$. Then:
$$
\|X - Y\|_{L^2}^2 = \|(X - X_0) - Y_0\|_{L^2}^2
$$
By Theorem 5.1.1, the unique element minimizing this norm over $Y_0 \in \mathcal{M}_0$ is $\widehat{Y}_0 = \Pi_{\mathcal{M}_0}(X - X_0)$, which is uniquely characterized by $\langle (X - X_0) - \widehat{Y}_0, V \rangle = 0$ for all $V \in \mathcal{M}_0$. Setting $\widehat{X} = X_0 + \widehat{Y}_0 \in \mathcal{M}_{\text{aff}}$ gives $\langle X - \widehat{X}, V \rangle = 0$ for all $V \in \mathcal{M}_0$. $\blacksquare$

#### Beat 2: Algorithmic Translation (The "How")

The Orthogonal Projection Theorem translates directly into a system of normal equations. When the subspace $\mathcal{M}$ is spanned by a finite collection of random variables $\mathbf{Y} = (Y_1, \dots, Y_n)' \in \mathcal{H}^n$, any element in $\mathcal{M}$ can be expressed as a linear combination $\boldsymbol{\lambda}'\mathbf{Y} = \sum_{i=1}^n \lambda_i Y_i$.

```
Geometric Orthogonal Projection in L^2(Ω, F, P):
                     Z(s_0) - μ(s_0)
                          ^
                          | \
                          |   \   Error Vector e_0 ⊥ M
                          |     \  ||e_0||^2 = σ_{SK}^2(s_0)
                          |       \
                          +--------> \
                          0          Z_SK(s_0) - μ(s_0) ∈ M
                         /           /
                        /           /
                       /___________/
               Subspace M = span{Z(s_1)-μ(s_1), ..., Z(s_n)-μ(s_n)}
```

Procedural breakdown of orthogonal projection onto finite-dimensional subspaces:
1. **Identify the Basis:** Construct the vector of centered random variables $\mathbf{Y} = \mathbf{Z} - \operatorname{E}[\mathbf{Z}]$.
2. **Compute Inner Products:** Form the Gram matrix $\mathbf{V} = [\langle Y_i, Y_j \rangle]_{i,j=1}^n = \operatorname{Var}(\mathbf{Z})$.
3. **Compute Target Projections:** Form the cross-covariance vector $\mathbf{v}_0 = (\langle Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0)], Y_i \rangle)_{i=1}^n$.
4. **Solve Normal Equations:** Solve the linear system $\mathbf{V}\boldsymbol{\lambda} = \mathbf{v}_0$ for the optimal weights.
5. **Evaluate Minimized Error:** Compute the squared orthogonal distance $\|Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)\|_{L^2}^2 = \operatorname{Var}(Z(\mathbf{s}_0)) - \boldsymbol{\lambda}'\mathbf{v}_0$.

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Pythagorean Energy Conservation:** The identity $\|X\|_{L^2}^2 = \|\widehat{X}\|_{L^2}^2 + \|X - \widehat{X}\|_{L^2}^2$ states that the total variance of the unobserved random variable decomposes into the variance of the predictor plus the variance of the prediction error. Because the error variance is non-negative, the predictor $\widehat{X}$ can *never* have more variance than the true state $X$.
- **Singular Gram Matrices:** If the sample locations contain duplicates ($\mathbf{s}_i = \mathbf{s}_j$) or if one observation is a deterministic linear combination of others, the Gram matrix $\mathbf{V}$ becomes singular ($\det(\mathbf{V}) = 0$). In Hilbert space terms, the spanning set $\{Y_1, \dots, Y_n\}$ is linearly dependent. The projection $\widehat{X}$ remains unique and well-defined, but the coefficient vector $\boldsymbol{\lambda}$ is no longer unique. Adding a microscale nugget effect $\tau^2 > 0$ shifts the spectrum $\mathbf{V} \mapsto \mathbf{V} + \tau^2 \mathbf{I}_n$, strictly restoring invertibility.

---

### 5.2 Simple Kriging (SK)

#### Beat 1: The Mathematical Anchor (The "Why")

Simple Kriging addresses spatial prediction under the foundational assumption that the mean function of the spatial random field is completely known across the domain $D$. While this assumption is idealized, Simple Kriging provides the theoretical benchmark against which all other kriging variants are measured.

#### Definition 5.2.1 (Simple Kriging Framework)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a second-order stationary spatial random field with known deterministic spatial mean function:
$$
\mu(\mathbf{s}) = \operatorname{E}[Z(\mathbf{s})], \quad \mathbf{s} \in D
$$
and continuous, positive definite spatial covariance kernel:
$$
C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s} + \mathbf{h})) = \operatorname{E}\left[(Z(\mathbf{s}) - \mu(\mathbf{s}))(Z(\mathbf{s} + \mathbf{h}) - \mu(\mathbf{s} + \mathbf{h}))\right]
$$
Let $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ denote the observed data vector, and $\boldsymbol{\mu} = (\mu(\mathbf{s}_1), \dots, \mu(\mathbf{s}_n))' \in \mathbb{R}^n$ the vector of known means at the sample locations. The $n \times n$ data covariance matrix is:
$$
\mathbf{V} = \operatorname{Var}(\mathbf{Z}) = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}
$$
and the $n \times 1$ vector of cross-covariances between the target site $\mathbf{s}_0$ and the sample sites is:
$$
\mathbf{v}_0 = (C(\mathbf{s}_0 - \mathbf{s}_1), C(\mathbf{s}_0 - \mathbf{s}_2), \dots, C(\mathbf{s}_0 - \mathbf{s}_n))' \in \mathbb{R}^n
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
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + \boldsymbol{\lambda}_{\text{SK}}' (\mathbf{Z} - \boldsymbol{\mu}) = \mu(\mathbf{s}_0) + \mathbf{v}_0' \mathbf{V}^{-1} (\mathbf{Z} - \boldsymbol{\mu})
$$
where the weight vector $\boldsymbol{\lambda}_{\text{SK}} \in \mathbb{R}^n$ satisfies the **Simple Kriging normal equations**:
$$
\mathbf{V} \boldsymbol{\lambda}_{\text{SK}} = \mathbf{v}_0 \iff \boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1} \mathbf{v}_0
$$
The minimized prediction error variance, termed the **Simple Kriging variance** $\sigma_{\text{SK}}^2(\mathbf{s}_0)$, is:
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = \operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \boldsymbol{\lambda}_{\text{SK}}' \mathbf{v}_0 = C(\mathbf{0}) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0
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
Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = (Z(\mathbf{s}_0) - \mu(\mathbf{s}_0)) - \boldsymbol{\lambda}' (\mathbf{Z} - \boldsymbol{\mu})
$$
Define the centered random variables $Y_0 = Z(\mathbf{s}_0) - \mu(\mathbf{s}_0)$ and $\mathbf{Y} = \mathbf{Z} - \boldsymbol{\mu}$. The mean-squared prediction error is:
$$
\begin{aligned}
\operatorname{MSPE}(\boldsymbol{\lambda}) &= \operatorname{E}\left[\left(Y_0 - \boldsymbol{\lambda}' \mathbf{Y}\right)^2\right] \\
&= \operatorname{E}[Y_0^2] - 2 \boldsymbol{\lambda}' \operatorname{E}[\mathbf{Y} Y_0] + \boldsymbol{\lambda}' \operatorname{E}[\mathbf{Y} \mathbf{Y}'] \boldsymbol{\lambda} \\
&= C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{v}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda}
\end{aligned}
$$
Because $\mathbf{V}$ is symmetric positive definite, $\operatorname{MSPE}(\boldsymbol{\lambda})$ is a strictly convex quadratic functional. Differentiating with respect to the vector $\boldsymbol{\lambda}$ and setting the gradient to zero:
$$
\nabla_{\boldsymbol{\lambda}} \operatorname{MSPE}(\boldsymbol{\lambda}) = -2 \mathbf{v}_0 + 2 \mathbf{V} \boldsymbol{\lambda} = \mathbf{0} \implies \mathbf{V} \boldsymbol{\lambda} = \mathbf{v}_0
$$
Since $\mathbf{V}$ is non-singular, the unique global minimizer is $\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1} \mathbf{v}_0$.

To compute the minimized prediction variance, substitute $\mathbf{V} \boldsymbol{\lambda}_{\text{SK}} = \mathbf{v}_0$ into the quadratic expansion:
$$
\begin{aligned}
\sigma_{\text{SK}}^2(\mathbf{s}_0) &= C(\mathbf{0}) - 2 \boldsymbol{\lambda}_{\text{SK}}' \mathbf{v}_0 + \boldsymbol{\lambda}_{\text{SK}}' \mathbf{V} \boldsymbol{\lambda}_{\text{SK}} \\
&= C(\mathbf{0}) - 2 \boldsymbol{\lambda}_{\text{SK}}' \mathbf{v}_0 + \boldsymbol{\lambda}_{\text{SK}}' \mathbf{v}_0 \\
&= C(\mathbf{0}) - \boldsymbol{\lambda}_{\text{SK}}' \mathbf{v}_0 = C(\mathbf{0}) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0
\end{aligned}
$$
This completes the analytical proof. $\blacksquare$

##### Term-by-Term Equation Anatomy: Simple Kriging System and Prediction Variance
$$
\underbrace{\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{c}_0}_{\text{Normal Dual System}} \iff \boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0, \qquad \sigma_{\text{SK}}^2(\mathbf{s}_0) = \underbrace{\sigma^2}_{\text{Prior Field Variance}} - \underbrace{\mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0}_{\text{Spatial Information Reduction}}
$$
- **$\mathbf{V} \in \mathbb{R}^{n \times n}$:** Data-to-data covariance matrix measuring spatial autocorrelation and clustering among the $n$ sample points. Inverting $\mathbf{V}$ automatically penalizes redundant clustered points (geometric declustering).
- **$\mathbf{c}_0 \in \mathbb{R}^n$ (also denoted $\mathbf{v}_0$):** Target-to-data cross-covariance vector $[C(\mathbf{s}_0 - \mathbf{s}_i)]_{i=1}^n$, prioritizing sample points that are geographically proximate to the prediction site $\mathbf{s}_0$.
- **$\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0$:** Vector of optimal linear weights assigned to centered sample observations $(Z(\mathbf{s}_i) - \mu(\mathbf{s}_i))$.
- **$\sigma^2 = C(\mathbf{0})$:** Total stationary a priori variance of the spatial process before any sampling data are collected.
- **$-\mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0 = -\boldsymbol{\lambda}_{\text{SK}}'\mathbf{c}_0$:** Quadratic variance reduction gained by conditioning on the $n$ spatial observations. Because $\mathbf{V}^{-1} \succ 0$, this quadratic form is strictly non-negative, proving that observing spatial data never increases prediction uncertainty: $\sigma_{\text{SK}}^2(\mathbf{s}_0) \le \sigma^2$.

#### Theorem 5.2.2 (Equivalence to Conditional Expectation Under Gaussianity)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a Gaussian random field with continuous covariance kernel $C(\mathbf{h})$ and mean function $\mu(\mathbf{s})$. Then the joint distribution of $(Z(\mathbf{s}_0), \mathbf{Z}')'$ is multivariate normal:
$$
\begin{bmatrix} Z(\mathbf{s}_0) \\ \mathbf{Z} \end{bmatrix} \sim \mathcal{N}_{1+n}\left(\begin{bmatrix} \mu(\mathbf{s}_0) \\ \boldsymbol{\mu} \end{bmatrix}, \begin{bmatrix} C(\mathbf{0}) & \mathbf{v}_0' \\ \mathbf{v}_0 & \mathbf{V} \end{bmatrix}\right)
$$
Under this Gaussian specification:
1. The conditional distribution of $Z(\mathbf{s}_0)$ given the observations $\mathbf{Z}$ is univariate Gaussian:
   $$
   Z(\mathbf{s}_0) \mid \mathbf{Z} \sim \mathcal{N}\left(\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}], \operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z})\right)
   $$
2. The conditional expectation is identically equal to the Simple Kriging predictor:
   $$
   \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] = \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + \mathbf{v}_0' \mathbf{V}^{-1} (\mathbf{Z} - \boldsymbol{\mu})
   $$
3. The conditional variance is identically equal to the Simple Kriging prediction variance:
   $$
   \operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z}) = \sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0
   $$
4. $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ is the globally optimal predictor of $Z(\mathbf{s}_0)$ among *all* Borel measurable functions $g(\mathbf{Z}) \in L^2(\Omega, \sigma(\mathbf{Z}), P)$, not merely linear predictors.

*Proof.*
Define the linear transformation $W = Z(\mathbf{s}_0) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{Z}$. Because $(Z(\mathbf{s}_0), \mathbf{Z}')'$ is jointly Gaussian, $(W, \mathbf{Z}')'$ is also jointly Gaussian. Evaluate the cross-covariance between $W$ and $\mathbf{Z}$:
$$
\operatorname{Cov}(W, \mathbf{Z}) = \operatorname{Cov}(Z(\mathbf{s}_0), \mathbf{Z}) - \mathbf{v}_0' \mathbf{V}^{-1} \operatorname{Cov}(\mathbf{Z}, \mathbf{Z}) = \mathbf{v}_0' - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{V} = \mathbf{v}_0' - \mathbf{v}_0' = \mathbf{0}'
$$
In a multivariate Gaussian distribution, zero covariance implies statistical independence: $W \perp \mathbf{Z}$.
Consequently, the conditional distribution of $W$ given $\mathbf{Z}$ equals its marginal distribution:
$$
\operatorname{E}[W \mid \mathbf{Z}] = \operatorname{E}[W] = \mu(\mathbf{s}_0) - \mathbf{v}_0' \mathbf{V}^{-1} \boldsymbol{\mu}
$$
$$
\operatorname{Var}(W \mid \mathbf{Z}) = \operatorname{Var}(W) = \operatorname{Var}(Z(\mathbf{s}_0)) - 2\mathbf{v}_0'\mathbf{V}^{-1}\mathbf{v}_0 + \mathbf{v}_0'\mathbf{V}^{-1}\mathbf{V}\mathbf{V}^{-1}\mathbf{v}_0 = C(\mathbf{0}) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0
$$
Since $Z(\mathbf{s}_0) = W + \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{Z}$, by linearity of conditional expectation:
$$
\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] = \operatorname{E}[W \mid \mathbf{Z}] + \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{Z} = \mu(\mathbf{s}_0) + \mathbf{v}_0' \mathbf{V}^{-1} (\mathbf{Z} - \boldsymbol{\mu}) = \widehat{Z}_{\text{SK}}(\mathbf{s}_0)
$$
and its conditional variance is $\operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Z}) = \operatorname{Var}(W \mid \mathbf{Z}) = \sigma_{\text{SK}}^2(\mathbf{s}_0)$.

Finally, for any $\sigma(\mathbf{Z})$-measurable function $g(\mathbf{Z}) \in L^2$, the law of total expectation gives:
$$
\operatorname{E}[(Z(\mathbf{s}_0) - g(\mathbf{Z}))^2] = \operatorname{E}\left[\operatorname{E}\left[(Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] + \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] - g(\mathbf{Z}))^2 \;\Big|\; \mathbf{Z}\right]\right]
$$
The conditional cross-term vanishes because $\operatorname{E}[(Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]) \mid \mathbf{Z}] = 0$. Thus:
$$
\operatorname{E}[(Z(\mathbf{s}_0) - g(\mathbf{Z}))^2] = \operatorname{E}\left[(Z(\mathbf{s}_0) - \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}])^2\right] + \operatorname{E}\left[(\operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}] - g(\mathbf{Z}))^2\right] \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)
$$
with equality if and only if $g(\mathbf{Z}) = \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Z}]$ almost surely. $\blacksquare$

#### Beat 2: Algorithmic Translation (The "How")

The Simple Kriging procedure consists of three computational steps:
1. Assembly of the $n \times n$ data covariance matrix $\mathbf{V}$ and the $n \times 1$ cross-covariance vector $\mathbf{v}_0$.
2. Solution of the symmetric positive definite linear system $\mathbf{V}\boldsymbol{\lambda}_{\text{SK}} = \mathbf{v}_0$ via Cholesky decomposition.
3. Linear combination of centered observations and evaluation of the quadratic variance reduction.

```python
def simple_kriging_predict(sample_coords, sample_vals, target_coord, 
                           mean_val, cov_kernel, nugget=0.0):
    """
    Computes the Simple Kriging prediction and variance at an unsampled location.

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
    nugget : float, optional
        Microscale variance (nugget effect) added to diagonal for regularization.

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
            if i == j:
                V[i][j] += nugget

    # Step 2: Assemble the n x 1 target-to-data covariance vector v_0
    v0 = [0.0] * n
    for i in range(n):
        dist_sq = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(d))
        dist = dist_sq ** 0.5
        v0[i] = cov_kernel(dist)

    # Step 3: Solve the symmetric positive definite linear system V * lambda = v_0
    # Using Gaussian elimination with partial pivoting (explicit for pedagogical clarity)
    A = [row[:] for row in V]
    b = v0[:]

    for p in range(n):
        # Pivot selection
        max_row = p
        max_val = abs(A[p][p])
        for r in range(p + 1, n):
            if abs(A[r][p]) > max_val:
                max_val = abs(A[r][p])
                max_row = r
        A[p], A[max_row] = A[max_row], A[p]
        b[p], b[max_row] = b[max_row], b[p]

        pivot = A[p][p]
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
    # z_hat = mu + sum_{i=1}^n lambda_i * (Z(s_i) - mu)
    residual_sum = sum(weights[i] * (sample_vals[i] - mean_val) for i in range(n))
    z_hat = mean_val + residual_sum

    # Step 5: Evaluate the Simple Kriging variance
    # sigma_SK^2(s_0) = C(0) - sum_{i=1}^n lambda_i * v_0[i]
    c0 = cov_kernel(0.0) + nugget
    var_reduction = sum(weights[i] * v0[i] for i in range(n))
    sk_var = max(0.0, c0 - var_reduction)

    return z_hat, sk_var, weights
```

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Shrinkage Towards the Prior Mean:** In Simple Kriging, the sum of weights $\sum_{i=1}^n \lambda_i$ is not constrained to equal 1. When the target site $\mathbf{s}_0$ is located far from all observation sites ($\|\mathbf{s}_0 - \mathbf{s}_i\| \gg \phi$), the cross-covariances decay to zero: $\mathbf{v}_0 \to \mathbf{0}$. Consequently, the weights decay to zero ($\boldsymbol{\lambda}_{\text{SK}} \to \mathbf{0}$), and the predictor gracefully reverts to the prior population mean:
  $$
  \lim_{\min_i \|\mathbf{s}_0 - \mathbf{s}_i\| \to \infty} \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu(\mathbf{s}_0)
  $$
  Simultaneously, the variance reduction $\mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0$ vanishes, and the prediction variance reaches its theoretical maximum, the total sill $C(\mathbf{0})$:
  $$
  \lim_{\min_i \|\mathbf{s}_0 - \mathbf{s}_i\| \to \infty} \sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0})
  $$
- **Variance Deficit Under Interpolation:** The variance of the predictor itself satisfies $\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0 = C(\mathbf{0}) - \sigma_{\text{SK}}^2(\mathbf{s}_0) \le C(\mathbf{0})$. Whenever there is prediction uncertainty ($\sigma_{\text{SK}}^2(\mathbf{s}_0) > 0$), the predictor systematically possesses *less* spatial variability than the true physical surface $Z(\mathbf{s}_0)$.

---

### 5.3 Ordinary Kriging (OK)

#### Beat 1: The Mathematical Anchor (The "Why")

In practical geostatistical applications, the true population mean $\mu(\mathbf{s})$ is virtually never known. Assuming an arbitrary mean introduces severe systematic bias into Simple Kriging. Ordinary Kriging resolves this fundamental dilemma by assuming that the mean is an **unknown constant** $\mu \in \mathbb{R}$ across the local spatial neighborhood:
$$
\operatorname{E}[Z(\mathbf{s})] \equiv \mu, \quad \forall \mathbf{s} \in D
$$
The objective is to derive an optimal linear predictor whose weights filter out the unknown parameter $\mu$ entirely.

#### Definition 5.3.1 (Ordinary Kriging Predictor)
The **Ordinary Kriging (OK) predictor** is a purely linear combination of the observations:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = \boldsymbol{\lambda}' \mathbf{Z}
$$
To ensure that prediction errors are zero-mean regardless of the true value of $\mu$, we impose the uniform unbiasedness criterion:
$$
\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0)] = \operatorname{E}[Z(\mathbf{s}_0)] \iff \operatorname{E}\left[\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right] = \mu \iff \mu \sum_{i=1}^n \lambda_i = \mu, \quad \forall \mu \in \mathbb{R}
$$
This condition holds for all $\mu \in \mathbb{R}$ if and only if the weights satisfy the **affine constraint**:
$$
\sum_{i=1}^n \lambda_i = \mathbf{1}' \boldsymbol{\lambda} = 1
$$
where $\mathbf{1} = (1, 1, \dots, 1)' \in \mathbb{R}^n$.

#### Theorem 5.3.1 (Ordinary Kriging System via Lagrange Multipliers)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a second-order stationary random field with unknown mean $\mu$ and covariance kernel $C(\mathbf{h})$. The weight vector $\boldsymbol{\lambda} \in \mathbb{R}^n$ that minimizes the prediction error variance $\operatorname{E}[(Z(\mathbf{s}_0) - \boldsymbol{\lambda}'\mathbf{Z})^2]$ subject to $\mathbf{1}'\boldsymbol{\lambda} = 1$ is the unique solution to the augmented linear system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ -\mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{v}_0 \\ 1 \end{bmatrix}
$$
where $\mu_{\text{OK}} \in \mathbb{R}$ is the Lagrange multiplier associated with the unbiasedness constraint.

In expanded scalar form, the system comprises $n+1$ linear equations:
$$
\begin{cases}
\sum_{j=1}^n \lambda_j C(\mathbf{s}_i - \mathbf{s}_j) - \mu_{\text{OK}} = C(\mathbf{s}_0 - \mathbf{s}_i), & i = 1, \dots, n \\
\sum_{j=1}^n \lambda_j = 1 &
\end{cases}
$$
The minimized prediction error variance, termed the **Ordinary Kriging variance**, is:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{v}_0 + \mu_{\text{OK}} = C(\mathbf{0}) - \sum_{i=1}^n \lambda_i C(\mathbf{s}_0 - \mathbf{s}_i) + \mu_{\text{OK}}
$$

*Proof.*
Under the constraint $\sum_{i=1}^n \lambda_i = 1$, the prediction error satisfies:
$$
Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0) = Z(\mathbf{s}_0) - \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = (Z(\mathbf{s}_0) - \mu) - \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) - \mu)
$$
Because the error is centered, its second moment equals its variance:
$$
\begin{aligned}
\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0)) &= \operatorname{Var}(Z(\mathbf{s}_0)) - 2 \sum_{i=1}^n \lambda_i \operatorname{Cov}(Z(\mathbf{s}_0), Z(\mathbf{s}_i)) + \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) \\
&= C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{v}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda}
\end{aligned}
$$
We formulate the Lagrangian objective function with multiplier $2\mu_{\text{OK}}$:
$$
\mathcal{L}(\boldsymbol{\lambda}, \mu_{\text{OK}}) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{v}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} + 2 \mu_{\text{OK}} \left(1 - \mathbf{1}' \boldsymbol{\lambda}\right)
$$
Differentiating with respect to $\boldsymbol{\lambda}$ and setting the vector gradient to zero:
$$
\nabla_{\boldsymbol{\lambda}} \mathcal{L} = -2 \mathbf{v}_0 + 2 \mathbf{V} \boldsymbol{\lambda} - 2 \mu_{\text{OK}} \mathbf{1} = \mathbf{0} \implies \mathbf{V} \boldsymbol{\lambda} - \mu_{\text{OK}} \mathbf{1} = \mathbf{v}_0
$$
Differentiating with respect to $\mu_{\text{OK}}$ recovers the affine constraint:
$$
\frac{\partial \mathcal{L}}{\partial \mu_{\text{OK}}} = 2(1 - \mathbf{1}' \boldsymbol{\lambda}) = 0 \implies \mathbf{1}' \boldsymbol{\lambda} = 1
$$
Combining these two matrix equations into block form yields the $(n+1) \times (n+1)$ Ordinary Kriging linear system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ -\mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{v}_0 \\ 1 \end{bmatrix}
$$
To evaluate the minimized prediction error variance, pre-multiply the first block equation $\mathbf{V}\boldsymbol{\lambda} - \mu_{\text{OK}}\mathbf{1} = \mathbf{v}_0$ by $\boldsymbol{\lambda}'$:
$$
\boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} - \mu_{\text{OK}} \boldsymbol{\lambda}' \mathbf{1} = \boldsymbol{\lambda}' \mathbf{v}_0 \implies \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} = \boldsymbol{\lambda}' \mathbf{v}_0 + \mu_{\text{OK}}
$$
where we utilized $\boldsymbol{\lambda}'\mathbf{1} = 1$. Substituting this quadratic form back into the variance expression:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{v}_0 + (\boldsymbol{\lambda}' \mathbf{v}_0 + \mu_{\text{OK}}) = C(\mathbf{0}) - \boldsymbol{\lambda}' \mathbf{v}_0 + \mu_{\text{OK}}
$$
This establishes the theorem. $\blacksquare$

##### Term-by-Term Equation Anatomy: Ordinary Kriging Bordered Lagrangian Matrix System
$$
\begin{bmatrix} 
\mathbf{V} & \mathbf{1} \\ 
\mathbf{1}' & 0 
\end{bmatrix} 
\begin{bmatrix} 
\boldsymbol{\lambda} \\ 
\mu 
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
- **$\mu \in \mathbb{R}$:** Scalar Lagrange multiplier (written as $\mu \equiv -\mu_{\text{OK}}$ in the symmetric system), serving as the shadow price of the unbiasedness constraint.
- **$\mathbf{c}_0 \in \mathbb{R}^n$ (or $\mathbf{v}_0$):** Cross-covariance vector connecting the target site $\mathbf{s}_0$ to each of the $n$ sample points.
- **$1$:** Right-hand side scalar enforcing that the weights sum exactly to unity.

##### Term-by-Term Equation Anatomy: Ordinary Kriging Prediction Variance
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = \underbrace{\sigma^2}_{\text{Prior Process Variance}} - \underbrace{\boldsymbol{\lambda}'\mathbf{c}_0}_{\text{Direct Spatial Information}} - \underbrace{\mu}_{\text{Lagrange Multiplier Adjustment}}
$$
- **$\sigma^2 = C(\mathbf{0})$:** Total stationary prior variance of the spatial field before incorporating sample data.
- **$-\boldsymbol{\lambda}'\mathbf{c}_0$:** Direct variance reduction contributed by the weighted spatial cross-covariances with the observation sites.
- **$-\mu$ (or $+\mu_{\text{OK}}$ where $\mu = -\mu_{\text{OK}}$):** Positive variance inflation penalty representing the uncertainty cost of estimating the unknown mean $\mu$. Because $\mu$ is unknown, Ordinary Kriging variance strictly exceeds Simple Kriging variance: $\sigma_{\text{OK}}^2(\mathbf{s}_0) = \sigma_{\text{SK}}^2(\mathbf{s}_0) + \frac{(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} \ge \sigma_{\text{SK}}^2(\mathbf{s}_0)$.

#### Theorem 5.3.2 (Exact Interpolation Property)
Let $\mathbf{s}_0 = \mathbf{s}_k \in \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ be one of the observation locations. Then:
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
   \mu_{\text{OK}}(\mathbf{s}_k) = 0
   $$

*Proof.*
When $\mathbf{s}_0 = \mathbf{s}_k$, the cross-covariance vector $\mathbf{v}_0$ is identical to the $k$-th column of $\mathbf{V}$, denoted $\mathbf{V}_{\bullet k} = \mathbf{V} \mathbf{e}_k$, where $\mathbf{e}_k = (0, \dots, 1, \dots, 0)'$ is the $k$-th standard basis vector in $\mathbb{R}^n$:
$$
\mathbf{v}_0 = \mathbf{V} \mathbf{e}_k
$$
Substitute the candidate solution $\boldsymbol{\lambda} = \mathbf{e}_k$ and $-\mu_{\text{OK}} = 0$ into the left-hand side of the Ordinary Kriging block system:
$$
\begin{bmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \mathbf{e}_k \\ 0 \end{bmatrix} = \begin{bmatrix} \mathbf{V} \mathbf{e}_k + \mathbf{1} \cdot 0 \\ \mathbf{1}' \mathbf{e}_k \end{bmatrix} = \begin{bmatrix} \mathbf{v}_0 \\ 1 \end{bmatrix}
$$
Because the augmented Ordinary Kriging matrix is non-singular, this solution is unique. Therefore:
$$
\boldsymbol{\lambda} = \mathbf{e}_k, \quad \mu_{\text{OK}} = 0
$$
The kriging prediction is $\widehat{Z}_{\text{OK}}(\mathbf{s}_k) = \mathbf{e}_k' \mathbf{Z} = Z(\mathbf{s}_k)$.
The prediction variance becomes:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_k) = C(\mathbf{0}) - \mathbf{e}_k' \mathbf{v}_0 + \mu_{\text{OK}} = C(\mathbf{0}) - C(\mathbf{s}_k - \mathbf{s}_k) + 0 = C(\mathbf{0}) - C(\mathbf{0}) = 0
$$
Thus, Ordinary Kriging is an exact interpolator. $\blacksquare$

#### Theorem 5.3.3 (Equivalence of Covariance and Semivariogram Formulations)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an intrinsically stationary random field with semivariogram $\gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))$.
1. The Ordinary Kriging system can be formulated entirely in terms of the semivariogram matrix $\mathbf{\Gamma} = [\gamma(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$ and point-to-data semivariogram vector $\boldsymbol{\gamma}_0 = (\gamma(\mathbf{s}_0 - \mathbf{s}_1), \dots, \gamma(\mathbf{s}_0 - \mathbf{s}_n))'$:
   $$
   \begin{bmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \mu_{\text{OK}}^{(\gamma)} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}
   $$
   or in expanded scalar form:
   $$
   \begin{cases}
   \sum_{j=1}^n \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + \mu_{\text{OK}}^{(\gamma)} = \gamma(\mathbf{s}_0 - \mathbf{s}_i), & i = 1, \dots, n \\
   \sum_{j=1}^n \lambda_j = 1 &
   \end{cases}
   $$
2. When the process is second-order stationary with $C(\mathbf{h}) = C(\mathbf{0}) - \gamma(\mathbf{h})$, the weight vector $\boldsymbol{\lambda}$ produced by the semivariogram system is identical to that of the covariance system, and the Lagrange multipliers satisfy:
   $$
   \mu_{\text{OK}}^{(\gamma)} = -\mu_{\text{OK}}^{(C)}
   $$
3. The Ordinary Kriging variance expressed in semivariogram terms is:
   $$
   \sigma_{\text{OK}}^2(\mathbf{s}_0) = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 + \mu_{\text{OK}}^{(\gamma)} = \sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) + \mu_{\text{OK}}^{(\gamma)}
   $$
4. The semivariogram formulation is strictly more general than the covariance formulation: it remains valid for intrinsically stationary processes with unbounded semivariograms (e.g., fractional Brownian fields where $\gamma(\mathbf{h}) \propto \|\mathbf{h}\|^\alpha$ with $0 < \alpha < 2$) for which no covariance function exists.

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
\mathcal{L}^{(\gamma)}(\boldsymbol{\lambda}, \mu_{\text{OK}}^{(\gamma)}) = 2 \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 - \boldsymbol{\lambda}' \mathbf{\Gamma} \boldsymbol{\lambda} - 2 \mu_{\text{OK}}^{(\gamma)} \left(\mathbf{1}' \boldsymbol{\lambda} - 1\right)
$$
Differentiating with respect to $\boldsymbol{\lambda}$:
$$
\nabla_{\boldsymbol{\lambda}} \mathcal{L}^{(\gamma)} = 2 \boldsymbol{\gamma}_0 - 2 \mathbf{\Gamma} \boldsymbol{\lambda} - 2 \mu_{\text{OK}}^{(\gamma)} \mathbf{1} = \mathbf{0} \implies \mathbf{\Gamma} \boldsymbol{\lambda} + \mu_{\text{OK}}^{(\gamma)} \mathbf{1} = \boldsymbol{\gamma}_0
$$
Combining this with $\mathbf{1}'\boldsymbol{\lambda} = 1$ gives the semivariogram block system.

Now assume second-order stationarity holds, so that $\mathbf{\Gamma} = C(\mathbf{0})\mathbf{1}\mathbf{1}' - \mathbf{V}$ and $\boldsymbol{\gamma}_0 = C(\mathbf{0})\mathbf{1} - \mathbf{v}_0$. Substitute these into $\mathbf{\Gamma}\boldsymbol{\lambda} + \mu_{\text{OK}}^{(\gamma)}\mathbf{1} = \boldsymbol{\gamma}_0$:
$$
(C(\mathbf{0})\mathbf{1}\mathbf{1}' - \mathbf{V})\boldsymbol{\lambda} + \mu_{\text{OK}}^{(\gamma)}\mathbf{1} = C(\mathbf{0})\mathbf{1} - \mathbf{v}_0
$$
Because $\mathbf{1}'\boldsymbol{\lambda} = 1$, the first term simplifies: $C(\mathbf{0})\mathbf{1}(\mathbf{1}'\boldsymbol{\lambda}) = C(\mathbf{0})\mathbf{1}$. Thus:
$$
C(\mathbf{0})\mathbf{1} - \mathbf{V}\boldsymbol{\lambda} + \mu_{\text{OK}}^{(\gamma)}\mathbf{1} = C(\mathbf{0})\mathbf{1} - \mathbf{v}_0
$$
Subtracting $C(\mathbf{0})\mathbf{1}$ and multiplying by $-1$:
$$
\mathbf{V}\boldsymbol{\lambda} - \mu_{\text{OK}}^{(\gamma)}\mathbf{1} = \mathbf{v}_0
$$
Comparing this with the covariance normal equation $\mathbf{V}\boldsymbol{\lambda} - \mu_{\text{OK}}^{(C)}\mathbf{1} = \mathbf{v}_0$ confirms that the solutions $\boldsymbol{\lambda}$ are identical, with $\mu_{\text{OK}}^{(\gamma)} = -\mu_{\text{OK}}^{(C)}$.

For the prediction variance, pre-multiply $\mathbf{\Gamma}\boldsymbol{\lambda} + \mu_{\text{OK}}^{(\gamma)}\mathbf{1} = \boldsymbol{\gamma}_0$ by $\boldsymbol{\lambda}'$:
$$
\boldsymbol{\lambda}' \mathbf{\Gamma} \boldsymbol{\lambda} + \mu_{\text{OK}}^{(\gamma)} = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 \implies \boldsymbol{\lambda}' \mathbf{\Gamma} \boldsymbol{\lambda} = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 - \mu_{\text{OK}}^{(\gamma)}
$$
Substituting this into $\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda}$:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - (\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \mu_{\text{OK}}^{(\gamma)}) = \boldsymbol{\lambda}'\boldsymbol{\gamma}_0 + \mu_{\text{OK}}^{(\gamma)}
$$
This establishes the theorem. $\blacksquare$

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Wait—Why does the Lagrange multiplier $\mu$ enter with opposite signs when setting up kriging with covariances vs. semivariograms?*  
> When setting up the Ordinary Kriging Lagrangian using covariance functions, the prediction error variance expands to $\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}) = \sigma^2 - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda}$. Minimizing this variance while enforcing $\mathbf{1}'\boldsymbol{\lambda} = 1$ leads to the Lagrangian:
> $$
> \mathcal{L}_{\text{cov}}(\boldsymbol{\lambda}, \mu) = \sigma^2 - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} - 2\mu\left(\boldsymbol{\lambda}'\mathbf{1} - 1\right)
> $$
> Here, larger covariance $\boldsymbol{\lambda}'\mathbf{c}_0$ *reduces* error variance, so we are maximizing cross-covariance while penalizing covariance redundancy. Conversely, in the semivariogram formulation, the error variance expands to $\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda}$. The Lagrangian becomes:
> $$
> \mathcal{L}_{\text{var}}(\boldsymbol{\lambda}, \mu) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda} + 2\mu\left(\boldsymbol{\lambda}'\mathbf{1} - 1\right)
> $$
> Because semivariance $\gamma(h) = \sigma^2 - C(h)$ represents spatial *dissimilarity* (variance distance) rather than affinity, the sign of the cross-terms is flipped! Setting the respective gradients to zero produces identical optimal weights $\boldsymbol{\lambda}^*$, but the equilibrium shadow price flips sign: $\mu_{\text{var}} = -\mu_{\text{cov}}$.

#### Beat 2: Algorithmic Translation (The "How")

The Ordinary Kriging solver constructs an augmented $(n+1) \times (n+1)$ system using either covariance or semivariogram functions, solves for weights and Lagrange multipliers, and returns the BLUP prediction and prediction variance.

```python
def ordinary_kriging_predict(sample_coords, sample_vals, target_coord, 
                             variogram_kernel):
    """
    Computes Ordinary Kriging prediction and prediction variance.

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

    Returns:
    --------
    z_hat : float
        Ordinary Kriging prediction Z_OK(s_0).
    ok_var : float
        Ordinary Kriging prediction error variance sigma_OK^2(s_0).
    weights : list of float
        Ordinary Kriging weights lambda_1, ..., lambda_n.
    mu_ok : float
        Lagrange multiplier associated with unbiasedness constraint.
    """
    n = len(sample_coords)
    d = len(target_coord)

    # Step 1: Assemble the (n+1) x (n+1) augmented semivariogram matrix Gamma_aug
    # [ Gamma   1 ]
    # [   1'    0 ]
    Gamma_aug = [[0.0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(n):
            dist_sq = sum((sample_coords[i][k] - sample_coords[j][k]) ** 2 for k in range(d))
            dist = dist_sq ** 0.5
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
        dist_sq = sum((target_coord[k] - sample_coords[i][k]) ** 2 for k in range(d))
        dist = dist_sq ** 0.5
        gamma_0_aug[i] = variogram_kernel(dist)
    gamma_0_aug[n] = 1.0

    # Step 3: Solve the linear system Gamma_aug * [lambda, mu]' = gamma_0_aug
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
    mu_ok = b[n]

    # Step 4: Compute the Ordinary Kriging prediction
    z_hat = sum(weights[i] * sample_vals[i] for i in range(n))

    # Step 5: Compute the Ordinary Kriging variance
    # sigma_OK^2(s_0) = sum_{i=1}^n lambda_i * gamma(s_0 - s_i) + mu_ok
    ok_var = sum(weights[i] * gamma_0_aug[i] for i in range(n)) + mu_ok
    ok_var = max(0.0, ok_var)

    return z_hat, ok_var, weights, mu_ok
```

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Partition of Unity and Scale/Shift Invariance:** Because $\sum_{i=1}^n \lambda_i = 1$, the Ordinary Kriging weights form an affine *partition of unity*. If the sample values are shifted by a global constant $c \in \mathbb{R}$ ($Z(\mathbf{s}_i) \to Z(\mathbf{s}_i) + c$), the predictor shifts by the exact same constant:
  $$
  \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) + c) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) + c \sum_{i=1}^n \lambda_i = \widehat{Z}_{\text{OK}}(\mathbf{s}_0) + c
  $$
- **Variance Deficit and Kriging Smoothing:** Ordinary Kriging is a minimum-variance estimator, meaning that it intentionally eliminates extreme fluctuations to minimize expected squared error. The variance of the OK predictor satisfies:
  $$
  \operatorname{Var}(\widehat{Z}_{\text{OK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \sigma_{\text{OK}}^2(\mathbf{s}_0) + 2\mu_{\text{OK}}
  $$
  Because $\sigma_{\text{OK}}^2(\mathbf{s}_0) > 0$ at unsampled sites, the variance of the predicted surface is systematically smaller than the true physical dispersion variance $C(\mathbf{0})$. Kriging maps act as low-pass spatial filters: high values are underestimated, low values are overestimated, and histogram variance is compressed. Consequently, evaluating non-linear threshold functions (e.g., ore grade cut-offs $\mathbb{I}_{\{Z(\mathbf{s}_0) \ge z_c\}}$) on kriged maps produces severe bias, motivating conditional geostatistical simulation (Chapter 10).

---

### 5.4 Geometric Properties of Kriging Weights and Spatial Screening

#### Beat 1: The Mathematical Anchor (The "Why")

Unlike inverse-distance weighting (IDW) schemes where weights are strictly positive functions of distance, kriging weights are determined by the *geometry* of the sample configuration: distances to the target, mutual clustering of observation points, and the smoothness of the underlying covariance kernel. Under specific geometric configurations, nearby observations completely screen distant observations.

#### Theorem 5.4.1 (Exact Markovian Screening in One Dimension)
Let $\{Z(t) : t \in \mathbb{R}\}$ be a stationary Gaussian random process on the real line with Exponential covariance function:
$$
C(h) = \sigma^2 \exp\left(-\frac{|h|}{\phi}\right), \quad \phi > 0, \; \sigma^2 > 0
$$
Suppose the process is observed at $n$ ordered locations:
$$
t_1 < t_2 < \dots < t_k < t_0 < t_{k+1} < \dots < t_n
$$
where $t_0 \in (t_k, t_{k+1})$ is the unsampled target prediction location.
Then:
1. The Simple Kriging weight vector $\boldsymbol{\lambda}_{\text{SK}} \in \mathbb{R}^n$ assigns non-zero weights strictly to the two immediate flanking neighbors $t_k$ and $t_{k+1}$:
   $$
   \lambda_i = 0, \quad \forall i \notin \{k, k+1\}
   $$
2. The optimal weights assigned to the flanking neighbors are:
   $$
   \lambda_k = \frac{\sinh\left(\frac{t_{k+1} - t_0}{\phi}\right)}{\sinh\left(\frac{t_{k+1} - t_k}{\phi}\right)}, \quad \lambda_{k+1} = \frac{\sinh\left(\frac{t_0 - t_k}{\phi}\right)}{\sinh\left(\frac{t_{k+1} - t_k}{\phi}\right)}
   $$
   where $\sinh(x) = \frac{1}{2}(e^x - e^{-x})$.

*Proof.*
A stationary Gaussian process with Exponential covariance on $\mathbb{R}$ is the classical Ornstein-Uhlenbeck process, which is a continuous-parameter Markov process. Let $\mathcal{F}_{\le t_k} = \sigma(\{Z(t) : t \le t_k\})$ and $\mathcal{F}_{\ge t_{k+1}} = \sigma(\{Z(t) : t \ge t_{k+1}\})$ denote the past and future $\sigma$-algebras. By the Markov property, conditionally on the boundary states $Z(t_k)$ and $Z(t_{k+1})$, the interior variable $Z(t_0)$ is conditionally independent of all external variables:
$$
Z(t_0) \perp \{Z(t_1), \dots, Z(t_{k-1}), Z(t_{k+2}), \dots, Z(t_n)\} \;\Big|\; \left(Z(t_k), Z(t_{k+1})\right)
$$
By Theorem 5.2.2, Simple Kriging under Gaussianity coincides with conditional expectation:
$$
\widehat{Z}_{\text{SK}}(t_0) = \operatorname{E}[Z(t_0) \mid Z(t_1), \dots, Z(t_n)] = \operatorname{E}[Z(t_0) \mid Z(t_k), Z(t_{k+1})]
$$
Because this projection lies entirely in $\operatorname{span}\{Z(t_k) - \mu, Z(t_{k+1}) - \mu\}$, the uniqueness of the orthogonal projection (Theorem 5.1.1) requires that all external weights vanish identically:
$$
\lambda_i = 0, \quad \forall i \notin \{k, k+1\}
$$
To determine $\lambda_k$ and $\lambda_{k+1}$, set $\mu = 0$ and $\sigma^2 = 1$ without loss of generality. Let $h_1 = t_0 - t_k > 0$, $h_2 = t_{k+1} - t_0 > 0$, and $\Delta = t_{k+1} - t_k = h_1 + h_2$. The $2 \times 2$ normal equations for the active flanking pair are:
$$
\begin{bmatrix} 1 & e^{-\Delta/\phi} \\ e^{-\Delta/\phi} & 1 \end{bmatrix} \begin{bmatrix} \lambda_k \\ \lambda_{k+1} \end{bmatrix} = \begin{bmatrix} e^{-h_1/\phi} \\ e^{-h_2/\phi} \end{bmatrix}
$$
The determinant of this matrix is $D = 1 - e^{-2\Delta/\phi}$. Applying Cramer's rule to solve for $\lambda_k$:
$$
\begin{aligned}
\lambda_k &= \frac{e^{-h_1/\phi} - e^{-\Delta/\phi} e^{-h_2/\phi}}{1 - e^{-2\Delta/\phi}} = \frac{e^{-h_1/\phi} - e^{-(h_1+2h_2)/\phi}}{1 - e^{-2(h_1+h_2)/\phi}} \\
&= \frac{e^{-(h_1+h_2)/\phi} \left(e^{h_2/\phi} - e^{-h_2/\phi}\right)}{e^{-(h_1+h_2)/\phi} \left(e^{(h_1+h_2)/\phi} - e^{-(h_1+h_2)/\phi}\right)} = \frac{\sinh(h_2/\phi)}{\sinh(\Delta/\phi)} = \frac{\sinh\left(\frac{t_{k+1} - t_0}{\phi}\right)}{\sinh\left(\frac{t_{k+1} - t_k}{\phi}\right)}
\end{aligned}
$$
By symmetry, interchanging $h_1$ and $h_2$ yields:
$$
\lambda_{k+1} = \frac{\sinh(h_1/\phi)}{\sinh(\Delta/\phi)} = \frac{\sinh\left(\frac{t_0 - t_k}{\phi}\right)}{\sinh\left(\frac{t_{k+1} - t_k}{\phi}\right)}
$$
This establishes exact 1D Markov screening. $\blacksquare$

```
Screening Effect in One Dimension (Exponential Covariance):
       λ_1=0       λ_{k-1}=0       λ_k > 0        Target t_0      λ_{k+1} > 0      λ_n=0
---------*-------------*--------------*---------------*---------------*--------------*-------> t
        t_1           t_{k-1}        t_k                              t_{k+1}        t_n
                                      \_______________ ________________/
                                                      v
                                        Exact Markov Screening:
                               Only t_k and t_{k+1} receive non-zero weights
```

#### Theorem 5.4.2 (Tensor Factorization of Kriging on Product Grids)
Let $D = \mathbb{R}^2$. Suppose the spatial covariance kernel is separable:
$$
C(\mathbf{h}) = C(h_x, h_y) = C_1(h_x) C_2(h_y)
$$
where $C_1$ and $C_2$ are valid one-dimensional covariance functions. Suppose observations are placed on a regular Cartesian product grid:
$$
\mathcal{S} = \{\mathbf{s}_{ij} = (x_i, y_j)' : i = 1, \dots, n_x, \; j = 1, \dots, n_y\}
$$
and let the target location be $\mathbf{s}_0 = (x_0, y_0)'$.
Then:
1. The $n \times n$ covariance matrix $\mathbf{V}$ (where $n = n_x n_y$) factors into the Kronecker product:
   $$
   \mathbf{V} = \mathbf{V}_x \otimes \mathbf{V}_y
   $$
   where $\mathbf{V}_x = [C_1(x_i - x_{i'})]_{i,i'=1}^{n_x}$ and $\mathbf{V}_y = [C_2(y_j - y_{j'})]_{j,j'=1}^{n_y}$.
2. The cross-covariance vector $\mathbf{v}_0$ factors as:
   $$
   \mathbf{v}_0 = \mathbf{v}_{0,x} \otimes \mathbf{v}_{0,y}
   $$
   where $\mathbf{v}_{0,x} = (C_1(x_0 - x_1), \dots, C_1(x_0 - x_{n_x}))'$ and $\mathbf{v}_{0,y} = (C_2(y_0 - y_1), \dots, C_2(y_0 - y_{n_y}))'$.
3. The Simple Kriging weight vector $\boldsymbol{\lambda}_{\text{SK}} \in \mathbb{R}^{n_x n_y}$ factorizes into the Kronecker product of the separate 1D weight vectors:
   $$
   \boldsymbol{\lambda}_{\text{SK}} = \boldsymbol{\lambda}_x \otimes \boldsymbol{\lambda}_y
   $$
   where $\mathbf{V}_x \boldsymbol{\lambda}_x = \mathbf{v}_{0,x}$ and $\mathbf{V}_y \boldsymbol{\lambda}_y = \mathbf{v}_{0,y}$. In particular, the weight $\lambda_{ij}$ assigned to grid node $(x_i, y_j)'$ is $\lambda_{ij} = \lambda_i^{(x)} \lambda_j^{(y)}$.
4. If $C_1$ and $C_2$ are both Exponential covariance functions, exact screening occurs simultaneously along both axes: only the four bounding vertices of the grid rectangle enclosing $(x_0, y_0)'$ receive non-zero weights.

*Proof.*
1. By definition of separability, $\operatorname{Cov}(Z(x_i, y_j), Z(x_{i'}, y_{j'})) = C_1(x_i - x_{i'}) C_2(y_j - y_{j'})$. In lexicographical ordering, the entry at $((i-1)n_y + j, (i'-1)n_y + j')$ is $[\mathbf{V}_x]_{i,i'} [\mathbf{V}_y]_{j,j'}$, which is the definition of the Kronecker product $\mathbf{V}_x \otimes \mathbf{V}_y$.
2. Similarly, the covariance with $\mathbf{s}_0 = (x_0, y_0)'$ is $C_1(x_0 - x_i) C_2(y_0 - y_j) = [\mathbf{v}_{0,x}]_i [\mathbf{v}_{0,y}]_j$, so $\mathbf{v}_0 = \mathbf{v}_{0,x} \otimes \mathbf{v}_{0,y}$.
3. By the mixed-product property of the Kronecker product, $(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}$ and $(\mathbf{A} \otimes \mathbf{B})(\mathbf{u} \otimes \mathbf{v}) = (\mathbf{A}\mathbf{u}) \otimes (\mathbf{B}\mathbf{v})$. Therefore:
   $$
   \boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1} \mathbf{v}_0 = (\mathbf{V}_x \otimes \mathbf{V}_y)^{-1} (\mathbf{v}_{0,x} \otimes \mathbf{v}_{0,y}) = (\mathbf{V}_x^{-1} \mathbf{v}_{0,x}) \otimes (\mathbf{V}_y^{-1} \mathbf{v}_{0,y}) = \boldsymbol{\lambda}_x \otimes \boldsymbol{\lambda}_y
   $$
4. When $C_1$ and $C_2$ are Exponential, Theorem 5.4.1 guarantees that $\boldsymbol{\lambda}_x$ has only 2 non-zero entries (flanking indices $k, k+1$) and $\boldsymbol{\lambda}_y$ has only 2 non-zero entries (flanking indices $l, l+1$). The Kronecker product $\boldsymbol{\lambda}_x \otimes \boldsymbol{\lambda}_y$ has exactly $2 \times 2 = 4$ non-zero entries, corresponding to the four corners of the grid cell enclosing $\mathbf{s}_0$. $\blacksquare$

#### Beat 2: Algorithmic Translation (The "How")

Because global matrix inversion scales as $O(n^3)$, practical implementations on large spatial datasets utilize local moving neighborhoods $\mathcal{S}(\mathbf{s}_0) \subset \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ containing $n_0 \ll n$ points. To prevent directional clustering from biasing the neighborhood, quadrant or octant sector constraints are enforced.

```python
def select_moving_neighborhood(sample_coords, target_coord, 
                               max_points=32, max_per_quadrant=8, 
                               search_radius=float('inf')):
    """
    Selects a balanced spatial neighborhood around a target location using quadrant search.

    Parameters:
    -----------
    sample_coords : list of tuples of float, shape (n, 2)
        List of 2D coordinates (x, y) for all sample sites.
    target_coord : tuple of float, shape (2,)
        Target coordinate (x_0, y_0).
    max_points : int
        Maximum total number of points to include in neighborhood.
    max_per_quadrant : int
        Maximum number of points allowed per spatial quadrant.
    search_radius : float
        Maximum allowable Euclidean distance from target.

    Returns:
    --------
    selected_indices : list of int
        Indices of chosen sample points in sample_coords.
    """
    x0, y0 = target_coord

    # Step 1: Compute distance and assign quadrant for all points within radius
    # Quadrants: 0: NE (+,+), 1: NW (-,+), 2: SW (-,-), 3: SE (+,-)
    candidates = []
    for idx, (x, y) in enumerate(sample_coords):
        dx = x - x0
        dy = y - y0
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= search_radius:
            if dx >= 0 and dy >= 0:
                quad = 0
            elif dx < 0 and dy >= 0:
                quad = 1
            elif dx < 0 and dy < 0:
                quad = 2
            else:
                quad = 3
            candidates.append((dist, quad, idx))

    # Step 2: Sort candidates by distance from target
    candidates.sort(key=lambda item: item[0])

    # Step 3: Greedily select points respecting quadrant and total capacity
    quad_counts = [0, 0, 0, 0]
    selected_indices = []

    for dist, quad, idx in candidates:
        if len(selected_indices) >= max_points:
            break
        if quad_counts[quad] < max_per_quadrant:
            selected_indices.append(idx)
            quad_counts[quad] += 1

    return selected_indices
```

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

#### Origins and Geometric Interpretation of Negative Kriging Weights
Unlike convex combination schemes (such as inverse-distance weighting) where weights are constrained to the interval $[0, 1]$, kriging systems routinely assign **negative weights** ($\lambda_i < 0$). These negative weights are not numerical errors; they are mathematically required to achieve minimum mean-squared prediction error.

Negative weights emerge from three physical and geometric mechanisms:
1. **Directional Derivative Estimation (Partial Screening):** When an observation $\mathbf{s}_2$ lies behind an intervening observation $\mathbf{s}_1$ along a ray emanating from $\mathbf{s}_0$, the variables $Z(\mathbf{s}_1)$ and $Z(\mathbf{s}_2)$ are mutually correlated. The kriging system acts as a numerical differentiator: it assigns a positive weight $\lambda_1 > 1$ to the closer point and a negative weight $\lambda_2 < 0$ to the distant point. The linear combination $\lambda_1 Z(\mathbf{s}_1) + \lambda_2 Z(\mathbf{s}_2) \approx Z(\mathbf{s}_1) + \alpha (Z(\mathbf{s}_1) - Z(\mathbf{s}_2))$ extrapolates the local spatial gradient toward $\mathbf{s}_0$.
2. **Curvature Matching and Non-Convex Configurations (Poisson Analogy):** When the prediction point $\mathbf{s}_0$ lies outside the convex hull of the data (spatial extrapolation), the optimal linear projection matches the local curvature of the random surface. In numerical analysis, second-order finite differences require alternating signs ($f''(x) \approx f(x+h) - 2f(x) + f(x-h)$). The kriging system assigns negative weights to outer points to match the second derivative of the field.
3. **Smooth Covariance Kernels:** Negative weights are significantly more pronounced when the covariance kernel is infinitely differentiable at the origin, such as the Gaussian covariance $C(\mathbf{h}) = \sigma^2 \exp(-\|\mathbf{h}\|^2/\phi^2)$ or Matérn kernels with smoothness parameter $\nu \ge 2$. Highly differentiable sample paths permit high-precision extrapolation via spatial Taylor expansions, demanding negative difference weights. Conversely, rough processes with non-differentiable covariance (such as the Exponential model where $\nu = 1/2$) exhibit strong screening and minimal or no negative weights.

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Wait—Why do kriging weights sometimes come out negative, and does that violate physical reality?*  
> Intuition from classical convex combinations (like inverse-distance weighting) suggests that all interpolation weights should be strictly non-negative: $\lambda_i \ge 0$. However, kriging does not calculate physical mixture proportions; it solves an orthogonal projection problem in Hilbert space $L^2$. Negative weights occur primarily due to the **screening effect** and **geometric declustering**. When an observation site $\mathbf{s}_2$ is situated directly behind another sample site $\mathbf{s}_1$ relative to the prediction location $\mathbf{s}_0$, the closer site $\mathbf{s}_1$ already captures the majority of the spatial signal. Because $\mathbf{s}_1$ and $\mathbf{s}_2$ are themselves highly correlated, the kriging system uses the difference $Z(\mathbf{s}_1) - Z(\mathbf{s}_2)$ to approximate the local spatial gradient (directional derivative). The closer point receives a weight $\lambda_1 > 1$, while the shielded point receives a negative weight $\lambda_2 < 0$ to subtract redundant information and extrapolate the trend toward $\mathbf{s}_0$. Negative weights are mathematically optimal and preserve physical reality by declustering redundant spatial data!

---

### 5.5 Local vs. Global Neighborhood Kriging

While global kriging incorporates all $n$ sample points, moving local neighborhoods $\mathcal{S}(\mathbf{s}_0)$ containing $n_0 \ll n$ points are standard in large-scale geostatistics. This computational approximation induces distinct theoretical trade-offs:

#### Proposition 5.5.1 (Sub-Optimality of Local Neighborhood Kriging)
Let $\mathcal{M}_{\text{local}} = \operatorname{span}\{Z(\mathbf{s}) : \mathbf{s} \in \mathcal{S}(\mathbf{s}_0)\}$ and $\mathcal{M}_{\text{global}} = \operatorname{span}\{Z(\mathbf{s}) : \mathbf{s} \in \mathcal{S}\}$. Because $\mathcal{S}(\mathbf{s}_0) \subset \mathcal{S}$, the local subspace is nested within the global subspace: $\mathcal{M}_{\text{local}} \subset \mathcal{M}_{\text{global}}$.
By the Orthogonal Projection Theorem:
$$
\sigma_{\text{global}}^2(\mathbf{s}_0) = \inf_{Y \in \mathcal{M}_{\text{global}}} \|Z(\mathbf{s}_0) - Y\|_{L^2}^2 \le \inf_{Y \in \mathcal{M}_{\text{local}}} \|Z(\mathbf{s}_0) - Y\|_{L^2}^2 = \sigma_{\text{local}}^2(\mathbf{s}_0)
$$
Thus, local neighborhood kriging always yields a prediction error variance greater than or equal to that of global kriging.

#### Proposition 5.5.2 (Discontinuity Across Search Boundaries)
Let $\mathcal{B} \subset D$ denote the geometric boundary across which the composition of the local neighborhood $\mathcal{S}(\mathbf{s}_0)$ changes (e.g., an observation point enters or exits the search radius).
Then:
1. Across $\mathcal{B}$, the weight assigned to the entering or exiting point jumps discontinuously from 0 to a non-zero value $\lambda^* \ne 0$ unless the covariance kernel satisfies $C(\|\mathbf{s}_i - \mathbf{s}_0\|) = 0$ at the boundary.
2. Consequently, the local kriging predicted surface $\mathbf{s}_0 \mapsto \widehat{Z}_{\text{local}}(\mathbf{s}_0)$ and the local kriging variance surface $\mathbf{s}_0 \mapsto \sigma_{\text{local}}^2(\mathbf{s}_0)$ exhibit artificial step-discontinuities across neighborhood transition boundaries.

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
- **(b)** Solve for the weights $\lambda_1, \lambda_2$ and the Lagrange multiplier $\mu_{\text{OK}}$.
- **(c)** Derive the exact expression for the prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$.
- **(d)** Specialize to the midpoint $\mathbf{s}_0 = (\mathbf{s}_1 + \mathbf{s}_2)/2$ under an isotropic power variogram $\gamma(h) = \theta h^\alpha$ ($0 < \alpha < 2$), and evaluate the variance as $\alpha \to 1$ and $\alpha \to 2$.

*Solution.*
- **(a)** The Ordinary Kriging semivariogram system is:
  $$
  \begin{bmatrix} 0 & \gamma(h_{12}) & 1 \\ \gamma(h_{12}) & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \\ \mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \gamma(h_{01}) \\ \gamma(h_{02}) \\ 1 \end{bmatrix}
  $$

- **(b)** From the constraint equation, $\lambda_1 + \lambda_2 = 1 \implies \lambda_2 = 1 - \lambda_1$.
  The first two equations are:
  $$
  \begin{cases}
  \lambda_2 \gamma(h_{12}) + \mu_{\text{OK}} = \gamma(h_{01}) \\
  \lambda_1 \gamma(h_{12}) + \mu_{\text{OK}} = \gamma(h_{02})
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
  (\lambda_1 + \lambda_2) \gamma(h_{12}) + 2\mu_{\text{OK}} = \gamma(h_{01}) + \gamma(h_{02}) \implies \mu_{\text{OK}} = \frac{\gamma(h_{01}) + \gamma(h_{02}) - \gamma(h_{12})}{2}
  $$

- **(c)** By Theorem 5.3.3, the prediction variance is $\sigma_{\text{OK}}^2(\mathbf{s}_0) = \lambda_1 \gamma(h_{01}) + \lambda_2 \gamma(h_{02}) + \mu_{\text{OK}}$:
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
    The prediction error variance vanishes because $\alpha = 2$ corresponds to an exact linear trend where the midpoint is determined without error. $\blacksquare$

---

#### Exercise 5.3 (Geometric Origins of Negative Kriging Weights)
Consider three collinear points on the real line: $s_1 = -1$, $s_2 = 0$, $s_3 = 1$. Let the target point be $s_0 = 2$, so that $s_0$ lies outside the convex hull of the data (extrapolation). Suppose the random field is stationary with covariance kernel $C(h)$.
- **(a)** Set up the Ordinary Kriging system for predicting $Z(s_0)$.
- **(b)** Let $C(h) = \exp(-|h|)$. Compute the analytical weights $\lambda_1, \lambda_2, \lambda_3$ and evaluate them numerically.
- **(c)** Replace the covariance with the Gaussian covariance $C(h) = \exp(-h^2)$. Verify which weight becomes negative, and explain why curvature matching produces negative weights.

*Solution.*
- **(a)** The distances among points are:
  - Data-to-data: $|s_1 - s_2| = 1$, $|s_2 - s_3| = 1$, $|s_1 - s_3| = 2$.
  - Target-to-data: $|s_0 - s_3| = 1$, $|s_0 - s_2| = 2$, $|s_0 - s_1| = 3$.
  The covariance matrix $\mathbf{V}$ and cross-covariance vector $\mathbf{v}_0$ are:
  $$
  \mathbf{V} = \begin{bmatrix} C(0) & C(1) & C(2) \\ C(1) & C(0) & C(1) \\ C(2) & C(1) & C(0) \end{bmatrix}, \quad \mathbf{v}_0 = \begin{bmatrix} C(3) \\ C(2) \\ C(1) \end{bmatrix}
  $$
  The Ordinary Kriging system is:
  $$
  \begin{bmatrix} C(0) & C(1) & C(2) & 1 \\ C(1) & C(0) & C(1) & 1 \\ C(2) & C(1) & C(0) & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ -\mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} C(3) \\ C(2) \\ C(1) \\ 1 \end{bmatrix}
  $$

- **(b)** For $C(h) = \exp(-|h|)$, let $\rho = e^{-1} \approx 0.367879$. Then $C(0) = 1$, $C(1) = \rho$, $C(2) = \rho^2$, $C(3) = \rho^3$.
  The system is:
  $$
  \begin{cases}
  \lambda_1 + \rho \lambda_2 + \rho^2 \lambda_3 - \mu_{\text{OK}} = \rho^3 \\
  \rho \lambda_1 + \lambda_2 + \rho \lambda_3 - \mu_{\text{OK}} = \rho^2 \\
  \rho^2 \lambda_1 + \rho \lambda_2 + \lambda_3 - \mu_{\text{OK}} = \rho \\
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
  All weights are strictly positive under the Exponential covariance.

- **(c)** Now let $C(h) = \exp(-h^2)$. The covariances are:
  $$
  C(0) = 1, \quad C(1) = e^{-1} \approx 0.367879, \quad C(2) = e^{-4} \approx 0.018316, \quad C(3) = e^{-9} \approx 0.000123
  $$
  Solving the augmented system $\mathbf{A} \mathbf{x} = \mathbf{b}$ numerically yields:
  $$
  \lambda_1 \approx +0.3285, \quad \lambda_2 \approx -0.0316, \quad \lambda_3 \approx +0.7031, \quad \mu_{\text{OK}} \approx +0.3296
  $$
  Notice that $\lambda_2 = -0.0316 < 0$ is strictly negative, while the weights sum exactly to $\sum \lambda_i = 0.3285 - 0.0316 + 0.7031 = 1.0$.
  The negative weight on $s_2$ combined with $\lambda_3 > 1$ represents a numerical forward-difference extrapolation:
  $$
  \widehat{Z}(s_0) \approx Z(s_3) + \alpha (Z(s_3) - Z(s_2)) = (1 + \alpha) Z(s_3) - \alpha Z(s_2)
  $$
  Because the Gaussian covariance produces infinitely differentiable sample paths, the kriging system uses the slope across $s_2$ and $s_3$ to extrapolate toward $s_0$, which inherently requires a negative weight on the trailing point $s_2$. $\blacksquare$

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
  The cross-covariance vector is $\mathbf{v}_0 = c_0 \mathbf{1}$, where $c_0 = \exp(-\Delta/\sqrt{2})$.
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
  \boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1} \mathbf{v}_0 = c_0 \mathbf{V}^{-1} \mathbf{1} = \frac{c_0}{\kappa} \mathbf{1} = \frac{\exp(-\Delta/\sqrt{2})}{1 + 2\exp(-\Delta) + \exp(-\sqrt{2}\Delta)} \mathbf{1}
  $$
  The Simple Kriging prediction variance is:
  $$
  \sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{v}_0' \mathbf{V}^{-1} \mathbf{v}_0 = 1 - c_0^2 \mathbf{1}' \left(\frac{1}{\kappa} \mathbf{1}\right) = 1 - \frac{4 c_0^2}{\kappa} = 1 - \frac{4 \exp(-\sqrt{2}\Delta)}{1 + 2\exp(-\Delta) + \exp(-\sqrt{2}\Delta)}
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


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 5: Spatial Prediction and Kriging Theory (BLUP) — Companion Exercises and Complete Solutions](ch05_exercises.md)
