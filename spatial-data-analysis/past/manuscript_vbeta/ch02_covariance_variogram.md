# Chapter 2: Spatial Covariance and Variogram Theory

Spatial statistics distinguishes itself from classical multivariate statistics by treating geographic coordinates not merely as external indexing labels, but as the fundamental geometric substrate governing stochastic dependence. In Chapter 1, we established that a continuous spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$ admits the structural linear-model decomposition:
$$
Z(\mathbf{s}) = \mu(\mathbf{s}) + \delta(\mathbf{s}) + \epsilon(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \delta(\mathbf{s}) + \epsilon(\mathbf{s})
$$
where $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ is the deterministic large-scale mean drift, $\delta(\mathbf{s})$ is a zero-mean, continuous, spatially correlated stochastic process, and $\epsilon(\mathbf{s})$ represents an uncorrelated white-noise micro-scale or measurement error process.

When observing the field at $n$ discrete sample locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in D$, the random observation vector $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ has covariance matrix:
$$
\operatorname{Var}(\mathbf{Z}) = \mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n
$$
where $\boldsymbol{\theta} = (\sigma^2, \phi, \tau^2)'$ denotes the structural parameter vector comprising the partial sill $\sigma^2 = \operatorname{Var}(\delta(\mathbf{s}))$, the spatial range parameter $\phi$, and the nugget effect $\tau^2 = \operatorname{Var}(\epsilon(\mathbf{s}))$.

This chapter develops the analytical theory and computational mechanics governing the spatial covariance kernel $C(\mathbf{h})$ and the semivariogram $\gamma(\mathbf{h})$ for lag separation vectors $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j \in \mathbb{R}^d$. By the end of this chapter, you will be able to:
- Verify whether a candidate spatial covariance or semivariogram function is mathematically admissible using positive definiteness, Bochner's spectral representation, and conditional negative definiteness.
- Navigate the canonical parametric families—Exponential, Gaussian, Spherical, and the general Matérn class—understanding their exact spatial ranges, closed forms, and dimension restrictions.
- Diagnose local sample regularity (mean-square continuity vs. differentiability) by analyzing origin curvature ($\gamma(h)$ as $h \to 0$).
- Construct mathematically valid 2D and 3D anisotropic covariance models via geometric coordinate transformations and zonal nested structures.

> [!NOTE]
> **Chapter at a Glance: Covariance vs. Semivariogram Duality**
>
> | Dimension / Property | Covariance Kernel $C(\mathbf{h})$ | Semivariogram $\gamma(\mathbf{h})$ |
> |---|---|---|
> | **Physical Meaning** | Bivariate persistence / co-fluctuation | Spatial dispersion / increment variance |
> | **Required Condition** | Positive semi-definite (PSD) on $\mathbb{R}^d$ | Conditionally negative definite (CND) on $\mathbf{1}'\boldsymbol{\lambda} = 0$ |
> | **Behavior at Origin** | $C(\mathbf{0}) = \sigma^2$ (or $\sigma^2 + \tau^2$) | $\gamma(\mathbf{0}) = 0$, with $\lim_{h \to 0^+} \gamma(h) = \tau^2$ (nugget) |
> | **Asymptotic Behavior** | $C(\mathbf{h}) \to 0$ as $\|\mathbf{h}\| \to \infty$ | $\gamma(\mathbf{h}) \to \sigma^2 + \tau^2$ (sill) or grows sub-quadratically |
> | **Spectral Admissibility** | Non-negative Fourier transform $f(\boldsymbol{\omega}) \ge 0$ | Schoenberg transform $\exp(-t\gamma(\mathbf{h}))$ is PSD for all $t > 0$ |

---

### 2.1 Positive Definite Functions and Bochner's Theorem in $\mathbb{R}^d$

#### 2.1.1 Classical-to-Spatial Bridge: From Scalar Variance of Linear Combinations to Spatial Positive Definiteness

To understand why spatial statistics demands positive definite covariance functions, we begin with elementary probability theory.

##### Elementary Linear Combinations of Random Variables
Consider two scalar random variables $X$ and $Y$ with variances $\sigma_X^2, \sigma_Y^2$ and covariance $\sigma_{XY} = \operatorname{Cov}(X, Y)$. For any real constants $a, b \in \mathbb{R}$, the variance of the linear combination $aX + bY$ is:
$$
\operatorname{Var}(aX + bY) = a^2 \sigma_X^2 + b^2 \sigma_Y^2 + 2ab \sigma_{XY}
$$
Because the variance of any real-valued random variable is defined as the expectation of a squared deviation $\operatorname{E}[(W - \operatorname{E}[W])^2]$, variance can never be negative:
$$
\operatorname{Var}(aX + bY) \ge 0 \quad \forall a, b \in \mathbb{R}
$$
In matrix notation, letting $\mathbf{c} = (a, b)'$ and $\mathbf{\Sigma} = \begin{pmatrix} \sigma_X^2 & \sigma_{XY} \\ \sigma_{XY} & \sigma_Y^2 \end{pmatrix}$, this non-negativity is expressed as a quadratic form:
$$
\mathbf{c}' \mathbf{\Sigma} \mathbf{c} \ge 0 \quad \forall \mathbf{c} \in \mathbb{R}^2
$$
which implies that the $2 \times 2$ covariance matrix $\mathbf{\Sigma}$ must be positive semi-definite.

##### Generalization to Spatial Configurations: Positive Definiteness
In spatial statistics, we observe not two variables, but $n$ spatial observations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ sampled at arbitrary locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in \mathbb{R}^d$. Any linear predictor (such as an ordinary kriging estimator) takes the form of a linear combination $Y = \sum_{i=1}^n c_i Z(\mathbf{s}_i) = \mathbf{c}'\mathbf{Z}$. Its variance is:
$$
\operatorname{Var}(Y) = \operatorname{Var}\left(\sum_{i=1}^n c_i Z(\mathbf{s}_i)\right) = \sum_{i=1}^n \sum_{j=1}^n c_i c_j \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = \mathbf{c}' \mathbf{V} \mathbf{c} = \sum_{i=1}^n \sum_{j=1}^n c_i c_j C(\mathbf{s}_i - \mathbf{s}_j)
$$
Because $\operatorname{Var}(Y) \ge 0$ must hold for *any* sample size $n \ge 1$, *any* arbitrary geographic configuration of sites $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$, and *any* arbitrary vector of real weights $\mathbf{c} \in \mathbb{R}^n$, the continuous covariance function $C(\mathbf{h})$ must satisfy:
$$
\sum_{i=1}^n \sum_{j=1}^n c_i c_j C(\mathbf{s}_i - \mathbf{s}_j) \ge 0 \quad \forall \mathbf{c} \in \mathbb{R}^n
$$
This is the formal definition of a **positive definite function** on $\mathbb{R}^d$.

#### 2.1.2 Why We Need This: Non-Negative Prediction Variances and Spectral Representations

In ordinary linear algebra, any symmetric matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ with non-negative eigenvalues is positive semi-definite. However, in spatial statistics, the covariance matrix $\mathbf{V}(\boldsymbol{\theta})$ must be generated by sampling a continuous spatial kernel $C : \mathbb{R}^d \to \mathbb{R}$ at an arbitrary number $n$ of arbitrary locations $\mathbf{s}_1, \dots, \mathbf{s}_n \in \mathbb{R}^d$. 

If a candidate function $C(\mathbf{h})$ permitted even a single spatial configuration of sample locations for which $\mathbf{V}$ had a negative eigenvalue, there would exist a linear combination of spatial observations $Y = \sum_{i=1}^n a_i Z(\mathbf{s}_i) = \mathbf{a}'\mathbf{Z}$ with negative variance:
$$
\operatorname{Var}(Y) = \operatorname{Var}(\mathbf{a}'\mathbf{Z}) = \mathbf{a}' \mathbf{V}(\boldsymbol{\theta}) \mathbf{a} < 0
$$
Because variance is defined as the expectation of a squared deviation $\operatorname{E}[(Y - \operatorname{E}[Y])^2] \ge 0$, a negative variance is an impossible contradiction: an average of squared real values cannot be negative. Therefore, spatial covariance kernels cannot be arbitrarily chosen curves; they must belong to the mathematical class of *positive semi-definite functions*.

> [!IMPORTANT]
> **Definition 2.1: Positive Semi-Definite Function on $\mathbb{R}^d$**
>
> A continuous real-valued function $C : \mathbb{R}^d \to \mathbb{R}$ is said to be **positive semi-definite** (commonly called *positive definite* in spatial statistics and geostatistics) on $\mathbb{R}^d$ if for every integer $n \ge 1$, every set of distinct spatial locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$, and every vector of real coefficients $\mathbf{a} = (a_1, \dots, a_n)' \in \mathbb{R}^n$:
> $$
> \sum_{i=1}^n \sum_{j=1}^n a_i a_j C(\mathbf{s}_i - \mathbf{s}_j) = \mathbf{a}' \mathbf{C} \mathbf{a} \ge 0
> $$
> It is **strictly positive definite** if the quadratic form equals zero only when $\mathbf{a} = \mathbf{0}$.

Verifying the double summation $\mathbf{a}'\mathbf{C}\mathbf{a} \ge 0$ directly for all possible configurations of points $\{\mathbf{s}_i\}_{i=1}^n$ and all $n \in \mathbb{N}$ is combinatorially impossible. The foundational tool that transforms this intractable geometric constraint into an operable analytical condition is Salomon Bochner's celebrated 1932 theorem connecting positive definiteness to Fourier analysis and characteristic functions.

In mathematical statistics, the characteristic function of a random vector $\mathbf{W}$ with probability density $f(\boldsymbol{\omega})$ is $\phi_{\mathbf{W}}(\mathbf{h}) = \operatorname{E}[e^{i \mathbf{W}'\mathbf{h}}]$. When the density of $\mathbf{W}$ is symmetric about the origin ($f(-\boldsymbol{\omega}) = f(\boldsymbol{\omega})$), the imaginary sine component integrates to zero, leaving a pure cosine expectation: $\operatorname{E}[\cos(\mathbf{W}'\mathbf{h})]$. Bochner's theorem establishes that continuous spatial covariance functions are precisely the cosine transforms of non-negative spectral densities!

> [!IMPORTANT]
> **Theorem 2.1: Bochner's Theorem in $\mathbb{R}^d$ (Sufficiency via Spectral Density)**
>
> A continuous, symmetric, real-valued function $C : \mathbb{R}^d \to \mathbb{R}$ is positive semi-definite if it admits the spectral representation:
> $$
> C(\mathbf{h}) = \int_{\mathbb{R}^d} \cos(\boldsymbol{\omega}' \mathbf{h}) f(\boldsymbol{\omega}) d\boldsymbol{\omega}
> $$
> where $f(\boldsymbol{\omega}) \ge 0$ is an integrable, symmetric spatial spectral density function on $\mathbb{R}^d$.

*Note on Full Measure Extension:* The complete if-and-only-if version of Bochner's theorem states that $C(\mathbf{h})$ is positive semi-definite if and only if it is the Fourier transform of a finite, non-negative *spectral measure* $F(d\boldsymbol{\omega})$. While this general formulation encompasses discrete point masses (such as the periodic covariance $C(h) = \cos(\omega_0 h)$ whose spectral measure concentrates entirely at $\pm \omega_0$), all standard continuous covariance families in spatial modeling possess proper, integrable spectral densities $f(\boldsymbol{\omega})$.

> [!NOTE]
> **From the Lecture Hall: Bridging the Prerequisite Gap**
> **A Covariance Is a Mixture of Cosines**
>
> If you have studied characteristic functions or Fourier analysis, Bochner's theorem should feel intuitive rather than mysterious. 
>
> Consider a pure cosine wave $C_k(h) = \cos(\omega_k h)$. Is it a valid covariance function in $\mathbb{R}^1$? Yes: it is the covariance of the random oscillation $Z(s) = A \cos(\omega_k s) + B \sin(\omega_k s)$, where $A$ and $B$ are uncorrelated zero-mean random variables with equal variance $\sigma_k^2$. The variance of any linear combination of $Z(s)$ is guaranteed non-negative.
>
> Bochner's theorem simply generalizes this single cosine to an infinite, continuous superposition (mixture) of harmonic plane waves $\cos(\boldsymbol{\omega}'\mathbf{h})$, each weighted by a non-negative energy weight $f(\boldsymbol{\omega}) d\boldsymbol{\omega} \ge 0$. Because every individual cosine is positive semi-definite, and non-negative mixtures of PSD functions remain PSD, the resulting integral $C(\mathbf{h})$ is guaranteed to be a valid covariance.

##### Term-by-Term Equation Anatomy: Bochner's Spectral Representation
$$
C(\mathbf{h}) = \int_{\mathbb{R}^d} \cos(\boldsymbol{\omega}' \mathbf{h}) f(\boldsymbol{\omega}) d\boldsymbol{\omega}
$$
- $C(\mathbf{h})$: The continuous spatial covariance function evaluated at separation lag vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j \in \mathbb{R}^d$.
- $\int_{\mathbb{R}^d} \cdots d\boldsymbol{\omega}$: The ordinary multidimensional Riemann integral over all spatial frequency components $\boldsymbol{\omega} = (\omega_1, \dots, \omega_d)' \in \mathbb{R}^d$.
- $\cos(\boldsymbol{\omega}' \mathbf{h})$: The harmonic cosine plane wave oscillating at spatial frequency vector $\boldsymbol{\omega}$. Because $f(\boldsymbol{\omega})$ is symmetric about the origin ($f(-\boldsymbol{\omega}) = f(\boldsymbol{\omega})$), the imaginary component $\int \sin(\boldsymbol{\omega}'\mathbf{h}) f(\boldsymbol{\omega}) d\boldsymbol{\omega}$ vanishes identically.
- $\boldsymbol{\omega}'\mathbf{h}$: The Euclidean inner product $\sum_{k=1}^d \omega_k h_k$, modulating wave phase according to spatial orientation and lag distance.
- $f(\boldsymbol{\omega})$: The non-negative spatial spectral density ($f(\boldsymbol{\omega}) \ge 0$), representing the variance power allotted to spatial frequency $\boldsymbol{\omega}$. The total variance of the process is the integral of the spectral density: $C(\mathbf{0}) = \int_{\mathbb{R}^d} f(\boldsymbol{\omega}) d\boldsymbol{\omega} = \sigma^2$. Positivity of $f(\boldsymbol{\omega})$ guarantees non-negative variance for any linear combination.

#### Analytical Proof of Bochner's Theorem ($\Leftarrow$ Direction)
We prove that if $C(\mathbf{h})$ is the cosine transform of an integrable, non-negative spectral density $f(\boldsymbol{\omega}) \ge 0$, it is guaranteed to be positive semi-definite for any spatial configuration.

Let $n \ge 1$ be arbitrary, let $\mathbf{s}_1, \dots, \mathbf{s}_n \in \mathbb{R}^d$ be arbitrary locations, and let $\mathbf{a} = (a_1, \dots, a_n)' \in \mathbb{R}^n$. Construct the quadratic form:
$$
\begin{aligned}
Q &= \sum_{i=1}^n \sum_{j=1}^n a_i a_j C(\mathbf{s}_i - \mathbf{s}_j) \\
  &= \sum_{i=1}^n \sum_{j=1}^n a_i a_j \int_{\mathbb{R}^d} \cos(\boldsymbol{\omega}' (\mathbf{s}_i - \mathbf{s}_j)) f(\boldsymbol{\omega}) d\boldsymbol{\omega}
\end{aligned}
$$
Using the trigonometric identity $\cos(\alpha - \beta) = \operatorname{Re}(e^{i(\alpha - \beta)}) = \operatorname{Re}(e^{i\alpha} e^{-i\beta})$, we can write the finite double summation inside the integral:
$$
\begin{aligned}
Q &= \int_{\mathbb{R}^d} \operatorname{Re}\left( \sum_{i=1}^n \sum_{j=1}^n a_i a_j e^{i \boldsymbol{\omega}' \mathbf{s}_i} e^{-i \boldsymbol{\omega}' \mathbf{s}_j} \right) f(\boldsymbol{\omega}) d\boldsymbol{\omega} \\
  &= \int_{\mathbb{R}^d} \operatorname{Re}\left[ \left( \sum_{i=1}^n a_i e^{i \boldsymbol{\omega}' \mathbf{s}_i} \right) \left( \sum_{j=1}^n a_j e^{-i \boldsymbol{\omega}' \mathbf{s}_j} \right) \right] f(\boldsymbol{\omega}) d\boldsymbol{\omega}
\end{aligned}
$$
Noting that $\sum_{j=1}^n a_j e^{-i \boldsymbol{\omega}' \mathbf{s}_j}$ is the complex conjugate of $\sum_{i=1}^n a_i e^{i \boldsymbol{\omega}' \mathbf{s}_i}$, the product is the squared modulus of a complex scalar, which is strictly real and non-negative:
$$
\left( \sum_{i=1}^n a_i e^{i \boldsymbol{\omega}' \mathbf{s}_i} \right) \overline{\left( \sum_{i=1}^n a_i e^{i \boldsymbol{\omega}' \mathbf{s}_i} \right)} = \left| \sum_{i=1}^n a_i e^{i \boldsymbol{\omega}' \mathbf{s}_i} \right|^2 \ge 0
$$
Since the integrand is a squared magnitude multiplied by a non-negative spectral density $f(\boldsymbol{\omega}) \ge 0$, the integral is guaranteed to be non-negative:
$$
Q = \int_{\mathbb{R}^d} \left| \sum_{i=1}^n a_i e^{i \boldsymbol{\omega}' \mathbf{s}_i} \right|^2 f(\boldsymbol{\omega}) d\boldsymbol{\omega} \ge 0
$$
Thus, $C(\mathbf{h})$ is positive semi-definite on $\mathbb{R}^d$. $\blacksquare$

The Fourier inversion formula recovers the spectral density from the spatial covariance kernel:
$$
f(\boldsymbol{\omega}) = \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} \cos(\boldsymbol{\omega}' \mathbf{h}) C(\mathbf{h}) d\mathbf{h}
$$
Under **isotropy**, spatial correlation depends solely on the Euclidean scalar distance $h = \|\mathbf{h}\|$. By symmetry, the spectral density depends only on the scalar radial frequency $\rho = \|\boldsymbol{\omega}\|$, denoted $f_0(\rho)$.

> [!IMPORTANT]
> **Theorem 2.2: Admissibility Criterion via Spectral Positivity**
>
> An isotropic candidate function $C_0(h)$ is a valid spatial covariance function in $\mathbb{R}^d$ if and only if its $d$-dimensional isotropic spectral density is non-negative for all radial frequencies:
> $$
> f_0(\rho) \ge 0 \quad \forall \rho \in [0, \infty)
> $$

---

#### 2.1.3 How to Compute It: Spatial Covariance Matrix Assembly

To translate positive definite covariance theory into computational practice, we implement explicit procedural logic that takes an arbitrary set of $n$ spatial coordinates $\mathbf{S} = (\mathbf{s}_1, \dots, \mathbf{s}_n)' \in \mathbb{R}^{n \times d}$ and evaluates the spatial covariance matrix $\mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n$. 

*(Note: We demonstrate below using the elementary Exponential covariance $C(h) = \sigma^2 \exp(-h/\phi)$; Section 2.3 systematically covers the Gaussian, Spherical, and Matérn families.)*

The procedure executes in three distinct steps:
1. **Pairwise Coordinate Differencing:** Construct the full $n \times n$ Euclidean distance matrix $\mathbf{H} = [h_{ij}]$ where $h_{ij} = \|\mathbf{s}_i - \mathbf{s}_j\|$.
2. **Parametric Kernel Evaluation:** Apply the scalar covariance kernel $C(h; \boldsymbol{\theta})$ elementwise to each off-diagonal entry $h_{ij}$.
3. **Nugget Regularization:** Set diagonal entries to the total sill $C(0) = \sigma^2 + \tau^2$.

```python
import math

def assemble_spatial_covariance_matrix(coords, theta, family="exponential"):
    """
    Constructs the n x n spatial covariance matrix V(theta) for a given set
    of spatial coordinates and covariance model parameters.
    
    Parameters:
        coords: List of coordinates [s_1, s_2, ..., s_n], where each s_i is a vector in R^d.
        theta:  Parameter vector (sigma2, phi, tau2) where:
                sigma2 = partial sill (variance of continuous spatial process delta)
                phi    = spatial range parameter
                tau2   = nugget effect (variance of micro-scale error epsilon)
        family: Covariance family string: "exponential", "gaussian", or "spherical"
        
    Returns:
        V: n x n symmetric covariance matrix (positive semi-definite for distinct
           locations under admissible parameter settings).
    """
    sigma2, phi, tau2 = theta[0], theta[1], theta[2]
    n = len(coords)
    d = len(coords[0])
    
    # Step 1: Initialize n x n matrix with zeros
    V = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Step 2: Compute pairwise Euclidean distances and kernel values
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal entry: C(0) = partial sill + nugget effect
                V[i][j] = sigma2 + tau2
            else:
                # Compute Euclidean distance: h = sqrt(sum_k (s_ik - s_jk)^2)
                sum_sq_diff = 0.0
                for k in range(d):
                    diff = coords[i][k] - coords[j][k]
                    sum_sq_diff += diff * diff
                h = math.sqrt(sum_sq_diff)
                
                # Evaluate parametric covariance kernel C(h)
                if family == "exponential":
                    # C(h) = sigma2 * exp(-h / phi)
                    val = sigma2 * math.exp(-h / phi)
                elif family == "gaussian":
                    # C(h) = sigma2 * exp(-(h / phi)^2)
                    val = sigma2 * math.exp(-((h / phi) ** 2))
                elif family == "spherical":
                    # C(h) = sigma2 * (1 - 1.5*(h/phi) + 0.5*(h/phi)^3) for h <= phi, else 0
                    if h <= phi:
                        ratio = h / phi
                        val = sigma2 * (1.0 - 1.5 * ratio + 0.5 * (ratio ** 3))
                    else:
                        val = 0.0
                else:
                    raise ValueError("Unsupported covariance family: " + family)
                
                # Exploit matrix symmetry: V_ij = V_ji
                V[i][j] = val
                V[j][i] = val
                
    return V
```

---

#### 2.1.4 What Can Go Wrong: Singular Matrices, Clustered Points, and Ridge Regularization

Why does positive definiteness fail in practice, and why do spatial analysts encounter singular matrix errors (`LinAlgError: matrix is singular or not positive definite`)? In empirical workflows, this error stems from two fundamentally different sources:
1. **Mathematical Inadmissibility:** Choosing an invalid candidate function whose spectral density goes negative, which creates true negative eigenvalues on certain spatial configurations.
2. **Numerical Ill-Conditioning:** Using a valid covariance function on closely spaced or clustered sample points, which drives the smallest eigenvalue $\lambda_{\min}(\mathbf{V})$ below floating-point machine precision.

Let us inspect the numerical mechanism in detail:

1. **The Geometry of Clustered Points:**
   If two observation locations $\mathbf{s}_1$ and $\mathbf{s}_2$ are separated by a tiny distance $\Delta = \|\mathbf{s}_1 - \mathbf{s}_2\| \approx 0$, their correlation is nearly $1$:
   $$
   C(\Delta) \approx \sigma^2 \left(1 - \frac{\Delta}{\phi}\right) \approx \sigma^2 = C(0)
   $$
   The first two rows of $\mathbf{V}$ become nearly collinear: row 1 is approximately row 2. The smallest eigenvalue $\lambda_{\min}(\mathbf{V})$ collapses toward zero, and the condition number explodes:
   $$
   \kappa(\mathbf{V}) = \frac{\lambda_{\max}(\mathbf{V})}{\lambda_{\min}(\mathbf{V})} \longrightarrow \infty
   $$
   In 64-bit IEEE double-precision arithmetic, when $\kappa(\mathbf{V}) > 10^{16}$, standard matrix inversion routines (such as Cholesky or LU decomposition) cannot distinguish the system from a non-invertible, singular operator.

2. **The Nugget Effect as a Tikhonov Regularizer:**
   As introduced in Chapter 1 (§1.3), the nugget parameter $\tau^2 > 0$ plays a crucial stabilizing role. In linear model terms, $\mathbf{V}(\boldsymbol{\theta}) = \mathbf{C} + \tau^2 \mathbf{I}_n$. Because $\mathbf{C}$ is positive semi-definite, its eigenvalues satisfy $\lambda_i(\mathbf{C}) \ge 0$, and adding $\tau^2 \mathbf{I}_n$ shifts every eigenvalue of $\mathbf{V}$ upward by exactly $\tau^2$:
   $$
   \lambda_i(\mathbf{V}) = \lambda_i(\mathbf{C}) + \tau^2 \ge \tau^2 > 0 \quad \forall i=1, \dots, n
   $$
   Furthermore, because all eigenvalues of $\mathbf{C}$ are non-negative, the maximum eigenvalue is bounded above by the trace of $\mathbf{C}$:
   $$
   \lambda_{\max}(\mathbf{C}) \le \sum_{i=1}^n \lambda_i(\mathbf{C}) = \operatorname{tr}(\mathbf{C}) = n C(0) = n \sigma^2
   $$
   Consequently, the maximum eigenvalue of $\mathbf{V}$ is bounded by $\lambda_{\max}(\mathbf{V}) \le n \sigma^2 + \tau^2$. This yields an explicit upper bound on the spectral condition number:
   $$
   \kappa(\mathbf{V}) = \frac{\lambda_{\max}(\mathbf{V})}{\lambda_{\min}(\mathbf{V})} \le \frac{n \sigma^2 + \tau^2}{\tau^2} = 1 + n \frac{\sigma^2}{\tau^2}
   $$
   The nugget effect thus acts as a physically grounded form of ridge regression (Tikhonov regularization), improving numerical conditioning in proportion to the ratio $\tau^2 / (n\sigma^2)$.

> [!TIP]
> **Worked Numerical Example: Condition Number Capping**
>
> Consider $n = 50$ points with partial sill $\sigma^2 = 1.0$:
> - **Without nugget ($\tau^2 = 0$):** If two points lie within distance $\Delta = 10^{-4}\phi$, the smallest eigenvalue drops to $\lambda_{\min} \approx 10^{-5}$, and with tighter clusters it quickly breaches $\lambda_{\min} < 10^{-16}$, yielding an uninvertible matrix ($\kappa > 10^{16}$).
> - **With modest nugget ($\tau^2 = 0.05$):** The minimum eigenvalue is guaranteed to be $\lambda_{\min} \ge 0.05$, and the condition number is strictly bounded:
>   $$
>   \kappa(\mathbf{V}) \le 1 + 50 \times \frac{1.0}{0.05} = 1 + 1000 = 1001 \approx 10^3
>   $$
>   Even with extreme spatial clustering, the linear solver remains well within the numerical stability zone ($\kappa \ll 10^{16}$). Note, however, that when $n$ is very large (e.g., $n = 50{,}000$) or the nugget is microscopic ($\tau^2 = 10^{-6}$), the upper bound $1 + n\sigma^2/\tau^2 \approx 5 \times 10^{10}$ remains sizable; the nugget aids conditioning, but cannot eliminate ill-conditioning on massive clustered designs without spatial subsetting or domain decomposition.

---

### 2.2 Semivariogram Theory and Conditionally Negative Definite (CND) Functions

#### 2.2.1 Why We Need This: Intrinsic Stationarity and Zero-Sum Contrasts

While second-order stationary processes require a finite, constant variance $\operatorname{Var}(Z(\mathbf{s})) = C(\mathbf{0}) < \infty$, many natural spatial phenomena (such as elevation profiles, geologic head, turbulent flow velocities, and mineral grades) exhibit unrestricted variability over expanding domains. For these processes, the absolute variance is infinite or undefined, but their spatial differences or increments remain stationary.

##### The Shift to Increments and ANOVA Contrast Intuition
In classical linear models and ANOVA, an experimental contrast is a linear combination of treatment means with zero-sum weights $\sum_{i=1}^n c_i \mu_i$ where $\sum_{i=1}^n c_i = 0$. Because the weights sum to zero, any constant baseline intercept $\mu_0$ cancels out completely:
$$
\sum_{i=1}^n c_i (\mu_0 + \tau_i) = \mu_0 \underbrace{\sum_{i=1}^n c_i}_{= 0} + \sum_{i=1}^n c_i \tau_i = \sum_{i=1}^n c_i \tau_i
$$
In spatial statistics, many natural processes exhibit spatial drift or unbounded variance where a global mean $\mu$ is ill-defined or non-stationary. By imposing the zero-sum contrast constraint $\sum_{i=1}^n \lambda_i = 0$ on spatial linear combinations $Y = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$, the unknown constant mean cancels identically:
$$
\operatorname{E}[Y] = \operatorname{E}\left[\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right] = \mu \sum_{i=1}^n \lambda_i = 0
$$
Furthermore, anchoring to an arbitrary pivot location $\mathbf{s}_1$, the linear combination can be rewritten entirely as a sum of pairwise differences:
$$
Y = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) - Z(\mathbf{s}_1))
$$
As established in Chapter 1, a random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is **intrinsically stationary** of order 0 if for any lag vector $\mathbf{h} \in \mathbb{R}^d$, the first difference increment $Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})$ satisfies:
1. $\operatorname{E}[Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})] = 0$
2. $\operatorname{Var}(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})) = 2 \gamma(\mathbf{h})$

The quantity $2\gamma(\mathbf{h})$ is the **variogram**, and $\gamma(\mathbf{h})$ is the **semivariogram**. Under second-order stationarity, the relationship between covariance and semivariogram is exact:
$$
\begin{aligned}
2\gamma(\mathbf{h}) &= \operatorname{Var}(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})) \\
&= \operatorname{Var}(Z(\mathbf{s} + \mathbf{h})) + \operatorname{Var}(Z(\mathbf{s})) - 2\operatorname{Cov}(Z(\mathbf{s} + \mathbf{h}), Z(\mathbf{s})) \\
&= C(\mathbf{0}) + C(\mathbf{0}) - 2C(\mathbf{h}) \\
&= 2[C(\mathbf{0}) - C(\mathbf{h})]
\end{aligned}
$$
Dividing by 2 gives the fundamental identity:
$$
\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})
$$
When covariance exists, the semivariogram mirrors it: as distance increases and spatial autocorrelation decays ($C(\mathbf{h}) \to 0$), the semivariogram rises from zero toward an asymptotic plateau known as the **sill** ($\lim_{\|\mathbf{h}\| \to \infty} \gamma(\mathbf{h}) = C(\mathbf{0})$).

```
  Semivariance gamma(h)
      ▲
      │                                     Total Sill (sigma^2 + tau^2)
      │- - - - - - - - - - - - - - - - - - -┌─────────────────────────────
      │                       .  . - - - ~ ~│
      │                .  - ~               │ Partial Sill (sigma^2)
      │          . - ~                      │
      │      . ~                            │
      │   .~                                │
tau^2 ┼──* (Nugget Effect)                  │
      │  │                                  │
      └──┴──────────────────────────────────┴─────────────────────────────►
        h=0                             h = Practical Range    Distance ||h||
```

> [!NOTE]
> **Wait—Why Did We Do That?**
> **FAQ 1: Why do we define the semivariogram with a factor of $\frac{1}{2}$ ($\gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))$) instead of just the variance of differences?**
>
> The raw variance of difference $2\gamma(\mathbf{h}) = \operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))$ is formally termed the **variogram**. Yet throughout spatial statistics and geostatistics, practitioners almost universally work with the **semivariogram** $\gamma(\mathbf{h})$, which carries the factor of $\frac{1}{2}$. Why inject this extra factor?
>
> The answer lies in **mathematical duality with the covariance function**. Under second-order stationarity:
> $$
> \operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})) = \operatorname{Var}(Z(\mathbf{s}+\mathbf{h})) + \operatorname{Var}(Z(\mathbf{s})) - 2\operatorname{Cov}(Z(\mathbf{s}+\mathbf{h}), Z(\mathbf{s})) = 2C(\mathbf{0}) - 2C(\mathbf{h})
> $$
> Dividing through by 2 produces the clean, symmetric relation:
> $$
> \gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})
> $$
> This definition has two profound aesthetic and computational benefits:
> 1. **Sill Alignment:** At infinite separation distance ($\|\mathbf{h}\| \to \infty$) where spatial autocorrelation vanishes ($C(\mathbf{h}) \to 0$), the semivariogram plateau equals the total variance of the process: $\lim_{\|\mathbf{h}\| \to \infty} \gamma(\mathbf{h}) = C(\mathbf{0}) = \sigma^2 + \tau^2$. Without the factor of $\frac{1}{2}$, the sill would equal $2C(\mathbf{0}) = 2\sigma^2 + 2\tau^2$, forcing practitioners to carry awkward factor-of-two corrections through every linear model and kriging equation.
> 2. **Kriging System Symmetry:** In Ordinary Kriging (Chapter 5), under the standard convention $\boldsymbol{\Gamma}\boldsymbol{\lambda} + \mu\mathbf{1} = \boldsymbol{\gamma}_0$, the system matrix formulated with $\gamma(\mathbf{h})$ yields identical numerical prediction weights and identical Lagrange multipliers as the covariance-based system $\mathbf{V}\boldsymbol{\lambda} - \mu\mathbf{1} = \mathbf{v}_0$, ensuring the semivariogram is a drop-in replacement for covariance without rescaling.

##### Derivation of the Conditionally Negative Definite (CND) Condition
Now consider the variance of any zero-sum contrast $Y = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)$ with $\sum_{i=1}^n \lambda_i = 0$. Expanding the squared pairwise differences explicitly:
$$
\begin{aligned}
\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 &= \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \left[ Z(\mathbf{s}_i)^2 - 2 Z(\mathbf{s}_i) Z(\mathbf{s}_j) + Z(\mathbf{s}_j)^2 \right] \\
&= \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)^2 \underbrace{\left(\sum_{j=1}^n \lambda_j\right)}_{= 0} - 2 \left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) \left(\sum_{j=1}^n \lambda_j Z(\mathbf{s}_j)\right) + \sum_{j=1}^n \lambda_j Z(\mathbf{s}_j)^2 \underbrace{\left(\sum_{i=1}^n \lambda_i\right)}_{= 0} \\
&= 0 - 2 \left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right)^2 + 0 \\
&= -2 \left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right)^2
\end{aligned}
$$
Because the weights sum to zero, the individual variance terms $Z(\mathbf{s}_i)^2$ and $Z(\mathbf{s}_j)^2$ are completely annihilated! Taking expectations of both sides:
$$
\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \underbrace{\operatorname{E}[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2]}_{= 2\gamma(\mathbf{s}_i - \mathbf{s}_j)} = -2 \operatorname{E}\left[\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right)^2\right] = -2 \operatorname{Var}\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right)
$$
Dividing by $-2$:
$$
\operatorname{Var}\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) = -\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j)
$$
Because the variance of any real random variable must be non-negative:
$$
\operatorname{Var}\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) \ge 0 \implies -\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \ge 0 \iff \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \le 0
$$
The minus sign in the increment variance expansion inverts the inequality: while covariances must be positive semi-definite for all vectors $\mathbf{c}$, semivariograms must be negative semi-definite on the orthogonal subspace $\mathbf{1}'\boldsymbol{\lambda} = 0$. This defines the class of **conditionally negative definite (CND)** functions.

> [!IMPORTANT]
> **Definition 2.2: Conditionally Negative Definite (CND) Functions**
>
> A real-valued function $\gamma : \mathbb{R}^d \to \mathbb{R}$ with $\gamma(\mathbf{0}) = 0$, continuous away from the origin (allowing for a possible nugget jump at $\mathbf{h} \to \mathbf{0}$), is **conditionally negative definite** (strictly, conditionally negative semi-definite) if for all integers $n \ge 1$, all distinct locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$, and all vectors $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)' \in \mathbb{R}^n$ satisfying the zero-sum constraint $\sum_{i=1}^n \lambda_i = \mathbf{1}'\boldsymbol{\lambda} = 0$:
> $$
> \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \le 0
> $$

> [!NOTE]
> **From the Lecture Hall: Bridging the Prerequisite Gap**
> **Why Variograms Need a Zero-Sum Constraint**
>
> Students encountering CND functions often ask: why define a function that is only negative definite on an orthogonal subspace, rather than everywhere?
>
> In ANOVA, testing $H_0: \mu_1 = \mu_2$ uses the contrast vector $\mathbf{c} = (1, -1)'$. Notice that $\sum c_i = 0$. If an experiment has an unknown overall laboratory calibration offset $\mu_0$, that offset affects every treatment equally: $Y_{ij} = \mu_0 + \tau_i + \epsilon_{ij}$. The contrast completely eliminates $\mu_0$ because $\sum c_i \mu_0 = \mu_0(1 - 1) = 0$.
>
> In spatial statistics, intrinsic stationarity assumes the field may drift without bound across the landscape—no constant global mean $\mu$ needs to exist. But for any linear combination whose weights sum to zero, any constant baseline cancels out. The semivariogram $\gamma(\mathbf{h})$ measures the variance of these differenced contrasts. The zero-sum constraint is not a mathematical trick; it is the exact algebraic prerequisite that allows us to model spatial correlation when the overall process mean is undefined.

#### Analytical Proof: Sub-Quadratic Growth Bound of the Semivariogram
A common misconception is that an unbounded semivariogram can grow at any arbitrary rate. Here we prove a cornerstone theorem of spatial statistics: a semivariogram can grow at most quadratically with lag distance.

> [!IMPORTANT]
> **Theorem 2.3: At Most Quadratic Growth of Semivariograms**
>
> If $\gamma(\mathbf{h})$ is a conditionally negative definite function on $\mathbb{R}^d$, then for any integer $m \ge 1$:
> $$
> \gamma(m\mathbf{h}) \le m^2 \gamma(\mathbf{h})
> $$
> Consequently, the semivariogram grows at most quadratically:
> $$
> \limsup_{\|\mathbf{h}\| \to \infty} \frac{\gamma(\mathbf{h})}{\|\mathbf{h}\|^2} \le c < \infty
> $$
> Moreover, exact quadratic growth ($\lim_{\|\mathbf{h}\| \to \infty} \gamma(\mathbf{h})/\|\mathbf{h}\|^2 = c > 0$) occurs if and only if the random field contains a random linear trend $Z(\mathbf{s}) = \mathbf{A}'\mathbf{s} + B$, where $\mathbf{A}$ is a zero-mean random slope vector. For any intrinsically stationary field with no random trend, $\lim_{\|\mathbf{h}\| \to \infty} \gamma(\mathbf{h})/\|\mathbf{h}\|^2 = 0$.

**Step-by-Step Proof:**
1. Consider collinear locations along any ray $\mathbf{h} \in \mathbb{R}^d$. Let $\mathbf{s}_k = k \mathbf{h}$ for $k = 0, 1, \dots, m$.
2. Define the increment $D_k = Z(\mathbf{s}_k) - Z(\mathbf{s}_{k-1}) = Z(k\mathbf{h}) - Z((k-1)\mathbf{h})$. Under intrinsic stationarity, each increment $D_k$ has zero mean, variance $\operatorname{Var}(D_k) = 2\gamma(\mathbf{h})$, and standard deviation $\operatorname{SD}(D_k) = \sqrt{\operatorname{Var}(D_k)} = \sqrt{2\gamma(\mathbf{h})}$.
3. The total displacement over distance $m\mathbf{h}$ is the telescoping sum of the $m$ step increments:
   $$
   Z(m\mathbf{h}) - Z(\mathbf{0}) = \sum_{k=1}^m D_k
   $$
4. Compute the variance of the total displacement. By the Cauchy-Schwarz covariance inequality ($\operatorname{Cov}(X, Y) \le \operatorname{SD}(X)\operatorname{SD}(Y)$):
   $$
   \operatorname{Var}\left(\sum_{k=1}^m D_k\right) = \sum_{j=1}^m \sum_{k=1}^m \operatorname{Cov}(D_j, D_k) \le \sum_{j=1}^m \sum_{k=1}^m \operatorname{SD}(D_j)\operatorname{SD}(D_k) = \left(\sum_{k=1}^m \operatorname{SD}(D_k)\right)^2
   $$
   Taking the square root of both sides gives the standard deviation inequality:
   $$
   \operatorname{SD}\left(\sum_{k=1}^m D_k\right) \le \sum_{k=1}^m \operatorname{SD}(D_k)
   $$
5. Substituting $\operatorname{SD}(D_k) = \sqrt{2\gamma(\mathbf{h})}$ and $\operatorname{SD}(Z(m\mathbf{h}) - Z(\mathbf{0})) = \sqrt{2\gamma(m\mathbf{h})}$:
   $$
   \sqrt{2\gamma(m\mathbf{h})} \le \sum_{k=1}^m \sqrt{2\gamma(\mathbf{h})} = m \sqrt{2\gamma(\mathbf{h})}
   $$
6. Squaring both sides and dividing by 2:
   $$
   \gamma(m\mathbf{h}) \le m^2 \gamma(\mathbf{h}) \quad \forall m \in \mathbb{N}
   $$
   This proves that $\gamma(h)$ cannot grow faster than $h^2$. 
7. To understand the boundary case $\gamma(\mathbf{h}) = c \|\mathbf{h}\|^2$, observe that the second difference $\Delta_{\mathbf{h}}^2 Z(\mathbf{s}) = Z(\mathbf{s} + 2\mathbf{h}) - 2Z(\mathbf{s}+\mathbf{h}) + Z(\mathbf{s})$ has zero-sum weights $\boldsymbol{\lambda} = (1, -2, 1)'$. Evaluating its variance under $\gamma(h) = c h^2$:
   $$
   \begin{aligned}
   \operatorname{Var}(\Delta_{\mathbf{h}}^2 Z(\mathbf{s})) &= -\sum_{i=1}^3 \sum_{j=1}^3 \lambda_i \lambda_j \gamma(s_i - s_j) \\
   &= - \left[ -4 c h^2 + 2 c (2h)^2 - 4 c h^2 \right] = - \left[ -4 c h^2 + 8 c h^2 - 4 c h^2 \right] = 0
   \end{aligned}
   $$
   A variance of zero implies that all second differences vanish almost surely: $\Delta_{\mathbf{h}}^2 Z(\mathbf{s}) \equiv 0$. The field must therefore be a linear plane $Z(\mathbf{s}) = \mathbf{A}'\mathbf{s} + B$, where the slope $\mathbf{A}$ is a random vector with covariance $\operatorname{Var}(\mathbf{A}) = 2c\mathbf{I}_d$. For any genuine stochastic field lacking a random linear slope, the ratio satisfies $\lim_{\|\mathbf{h}\| \to \infty} \gamma(\mathbf{h})/\|\mathbf{h}\|^2 = 0$. $\blacksquare$

> [!IMPORTANT]
> **Theorem 2.4: Schoenberg's Equivalence Theorem (1938)**
>
> Let $\gamma : \mathbb{R}^d \to \mathbb{R}$ satisfy $\gamma(\mathbf{0}) = 0$. The function $\gamma(\mathbf{h})$ is conditionally negative definite if and only if for every positive real scaling parameter $t > 0$, the transformed kernel:
> $$
> C_t(\mathbf{h}) = \exp(-t \gamma(\mathbf{h}))
> $$
> is positive semi-definite on $\mathbb{R}^d$.

**Proof of Sufficiency ($\Leftarrow$ via Taylor Series):**
Suppose $C_t(\mathbf{h}) = \exp(-t\gamma(\mathbf{h}))$ is positive semi-definite for all $t > 0$. Let $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)'$ satisfy the zero-sum constraint $\sum_{i=1}^n \lambda_i = 0$. Expanding the quadratic form in a first-order Taylor series around $t = 0$:
$$
\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \exp(-t \gamma(\mathbf{s}_i - \mathbf{s}_j)) = \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \left[ 1 - t \gamma(\mathbf{s}_i - \mathbf{s}_j) + \mathcal{O}(t^2) \right] \ge 0
$$
Distributing the sum:
$$
\underbrace{\left(\sum_{i=1}^n \lambda_i\right) \left(\sum_{j=1}^n \lambda_j\right)}_{= 0} - t \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + \mathcal{O}(t^2) \ge 0
$$
Dividing by $t > 0$ and taking the limit as $t \to 0^+$:
$$
-\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \ge 0 \implies \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \le 0
$$
Hence $\gamma(\mathbf{h})$ is conditionally negative definite. (The converse direction follows from the Lévy-Khinchin theorem; see Schoenberg, 1938). $\blacksquare$

---

#### 2.2.2 How to Compute It: Projected CND Matrix Verification

How do we computationally verify whether a candidate semivariogram matrix $\boldsymbol{\Gamma} = [\gamma(\mathbf{s}_i - \mathbf{s}_j)]$ is conditionally negative definite on a sample set?

Because the test vector $\boldsymbol{\lambda}$ must lie in the $(n-1)$-dimensional linear subspace orthogonal to the vector of ones $\mathbf{1}_n = (1, \dots, 1)'$, we apply the symmetric **centering projection operator**:
$$
\mathbf{P} = \mathbf{I}_n - \frac{1}{n} \mathbf{1}_n \mathbf{1}_n'
$$
For any vector $\mathbf{u} \in \mathbb{R}^n$, the projected vector $\boldsymbol{\lambda} = \mathbf{P}\mathbf{u}$ automatically satisfies $\mathbf{1}'\boldsymbol{\lambda} = \mathbf{1}'\mathbf{P}\mathbf{u} = \mathbf{0}'\mathbf{u} = 0$. 
Thus, verifying conditional negative definiteness is mathematically equivalent to showing that the projected matrix:
$$
\mathbf{Q} = -\mathbf{P} \boldsymbol{\Gamma} \mathbf{P}
$$
is positive semi-definite (i.e., all eigenvalues of $\mathbf{Q}$ are non-negative: $\lambda_i(\mathbf{Q}) \ge -\epsilon_{\text{tol}}$). Note that $\mathbf{Q}$ always possesses at least one zero eigenvalue corresponding to the null vector $\mathbf{P}\mathbf{1}_n = \mathbf{0}$.

```python
def verify_conditionally_negative_definite(coords, semivariogram_func, tol=1e-8):
    """
    Verifies the Conditionally Negative Definite (CND) property of a candidate
    semivariogram function on a finite configuration of spatial coordinates.
    
    Parameters:
        coords: List of coordinates [s_1, ..., s_n], s_i in R^d.
        semivariogram_func: Callable gamma(h) taking scalar distance h.
        tol: Numerical tolerance for zero/negative eigenvalues.
        
    Returns:
        is_cnd: Boolean (True if all projected pivots/eigenvalues >= -tol)
        min_pivot: Float, smallest diagonal pivot of projected matrix
        Q: Projected matrix Q = -P * Gamma * P
    """
    n = len(coords)
    d = len(coords[0])
    
    # Step 1: Compute n x n distance matrix and evaluate Semivariogram Matrix Gamma
    Gamma = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_sq = 0.0
            for k in range(d):
                diff = coords[i][k] - coords[j][k]
                dist_sq += diff * diff
            h = dist_sq ** 0.5
            Gamma[i][j] = semivariogram_func(h)
            
    # Step 2: Construct the Orthogonal Projection Operator P = I - (1/n) * 1 * 1'
    P = [[-1.0 / n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        P[i][i] += 1.0
        
    # Step 3: Compute Q = -P * Gamma * P via matrix multiplication
    # T = Gamma * P
    T = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[i][j] += Gamma[i][k] * P[k][j]
                
    # Q = -P * T
    Q = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(n):
                val += P[i][k] * T[k][j]
            Q[i][j] = -val
            
    # Step 4: Spectral diagnostic via LDL' decomposition with regularized pivot check
    # Since Q has rank at most n-1 (Q * 1 = 0), we check that no pivot is significantly negative.
    def check_psd_pivots(A, tolerance):
        m = len(A)
        L = [[0.0] * m for _ in range(m)]
        D = [0.0] * m
        min_p = float("inf")
        for i in range(m):
            L[i][i] = 1.0
            for j in range(i):
                s = sum(L[i][k] * L[j][k] * D[k] for k in range(j))
                if abs(D[j]) > 1e-12:
                    L[i][j] = (A[i][j] - s) / D[j]
            s = sum(L[i][k] ** 2 * D[k] for k in range(i))
            D[i] = A[i][i] - s
            if D[i] < min_p:
                min_p = D[i]
            if D[i] < -tolerance:
                return False, min_p
        return True, min_p

    is_cnd, min_pivot = check_psd_pivots(Q, tol)
    return is_cnd, min_pivot, Q
```

> [!TIP]
> **Worked Numerical Example: Testing Candidate Variograms on Three Points**
>
> Consider three collinear points $s_1 = 0$, $s_2 = 1$, $s_3 = 2$ in $\mathbb{R}^1$:
> - **Valid linear model $\gamma(h) = h$:** The projected matrix $\mathbf{Q} = -\mathbf{P}\boldsymbol{\Gamma}\mathbf{P}$ has eigenvalues $\{0.50, 0.0, 0.0\}$. All eigenvalues are non-negative; `is_cnd = True`.
> - **Invalid cubic model $\gamma(h) = h^3$:** The projected matrix $\mathbf{Q}$ has eigenvalues $\{-1.333, 0.0, 0.0\}$. The negative eigenvalue $\lambda = -1.333$ immediately flags invalidity; `is_cnd = False`.
> - **Invalid fractional power $\gamma(h) = h^{2.5}$:** The projected matrix $\mathbf{Q}$ has minimum eigenvalue $\lambda \approx -0.552 < 0$; `is_cnd = False`.

---

#### 2.2.3 What Can Go Wrong: The Polynomial Pitfall and the Parabolic Trap

Why can we not simply fit an arbitrary higher-degree polynomial $\gamma(h) = a + b h^2 + c h^4$ or a fractional power like $\gamma(h) = b h^{2.5}$ to empirical variogram clouds?

##### 1. The Parabolic Trap and Negative Prediction Variances
If an analyst fits a higher-order polynomial or a power law with exponent $\alpha > 2$, the resulting candidate semivariogram violates the conditionally negative definite condition for certain point configurations. When plugged into Ordinary Kriging (Chapter 5), the kriging prediction variance equation:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_i - \mathbf{s}_0) + \mu
$$
can evaluate to a **negative number**: $\sigma_{\text{OK}}^2(\mathbf{s}_0) < 0$. A negative variance is fatal: it prevents the calculation of standard errors, causes confidence intervals to collapse into complex numbers, and crashes spatial simulation algorithms.

##### 2. Explicit Proof of Failure for $\alpha > 2$ via Second Differences
To see directly why any power $\alpha > 2$ fails CND, consider the elementary second-difference contrast on three unit-spaced points $s_1 = 0, s_2 = 1, s_3 = 2$:
$$
\boldsymbol{\lambda} = (1, -2, 1)' \implies \sum_{i=1}^3 \lambda_i = 1 - 2 + 1 = 0
$$
The variance of the linear contrast $Y = Z(0) - 2Z(1) + Z(2)$ under candidate variogram $\gamma(h) = b h^\alpha$ ($b > 0$) is:
$$
\begin{aligned}
\operatorname{Var}(Y) &= -\sum_{i=1}^3 \sum_{j=1}^3 \lambda_i \lambda_j \gamma(|s_i - s_j|) \\
&= -b \left[ 2\lambda_1\lambda_2 \cdot 1^\alpha + 2\lambda_2\lambda_3 \cdot 1^\alpha + 2\lambda_1\lambda_3 \cdot 2^\alpha \right] \\
&= -b \left[ 2(1)(-2) \cdot 1 + 2(-2)(1) \cdot 1 + 2(1)(1) \cdot 2^\alpha \right] \\
&= -b \left[ -4 - 4 + 2 \cdot 2^\alpha \right] \\
&= -2b (2^\alpha - 4)
\end{aligned}
$$
Now inspect $\operatorname{Var}(Y)$ as a function of the power $\alpha$:
- For $\alpha = 1$ (linear variogram): $\operatorname{Var}(Y) = -2b(2 - 4) = +4b > 0$ (valid!).
- For $\alpha = 2$ (parabolic boundary): $\operatorname{Var}(Y) = -2b(4 - 4) = 0$ (valid boundary: deterministic linear trend).
- For any $\alpha > 2$: Since $2^\alpha > 2^2 = 4$, we have $2^\alpha - 4 > 0$, which yields:
  $$
  \operatorname{Var}(Y) = -2b(2^\alpha - 4) < 0 \quad \text{for all } \alpha > 2
  $$
Because a variance can never be negative, **the power model $\gamma(h) = b h^\alpha$ is mathematically invalid for all $\alpha > 2$ in every Euclidean dimension $\mathbb{R}^d$**.

##### 3. Admissible Power-Law Parameter Range
The valid power-law semivariogram is $\gamma(h) = b h^\alpha$ ($b > 0$), which is conditionally negative definite **if and only if $0 < \alpha \le 2$**:
- For $0 < \alpha < 2$: The model corresponds to an intrinsically stationary process with stationary increments and fractal paths (fractional Brownian field).
- At $\alpha = 2$: The model forms the degenerate parabolic boundary corresponding to a random linear slope.
- For $\alpha > 2$: Mathematically invalid; violates the sub-quadratic growth theorem and produces negative contrast variances.

---

### 2.3 Admissible Parametric Covariance and Semivariogram Families

#### 2.3.1 Why We Need This: The Permissible Model Zoo

To ensure strict positive definiteness across all spatial configurations, spatial statistics relies on admissible parametric families. Below is the structural taxonomy of valid isotropic models in $\mathbb{R}^d$.

| Family Name | Covariance Function $C(h)$ | Valid Dimensions | Practical Range $h_{\text{eff}}$ ($C \approx 0.05\sigma^2$) | Mean-Square Smoothness |
| :--- | :--- | :--- | :--- | :--- |
| **Exponential** | $\sigma^2 \exp\left(-\frac{h}{\phi}\right)$ | All $d \ge 1$ | $3\phi$ | Continuous, nowhere MS-differentiable ($\nu = 1/2$) |
| **Gaussian** | $\sigma^2 \exp\left(-\frac{h^2}{\phi^2}\right)$ | All $d \ge 1$ | $\sqrt{3}\phi \approx 1.732\phi$ | Infinitely MS-differentiable ($\nu \to \infty$) |
| **Spherical** | $\sigma^2\left[1-\frac{3h}{2a}+\frac{h^3}{2a^3}\right]\mathbb{I}_{\{h \le a\}}$ | $d = 1, 2, 3$ **ONLY** | Exact range $a$ ($C(a) = 0$) | Continuous, nowhere MS-differentiable (linear origin) |
| **Matérn Class** | $\sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}h}{\phi}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}h}{\phi}\right)$ | All $d \ge 1$ | $\approx 3\phi$ (for all $\nu$) | Order $\lceil\nu\rceil-1$ MS-differentiable |
| **Power-Law** | No $C(h)$; $\gamma(h) = b h^\alpha$ ($0 < \alpha \le 2$) | All $d \ge 1$ | $\infty$ (intrinsically stationary) | Continuous for $\alpha \in (0, 2)$; linear drift for $\alpha = 2$ |

> [!NOTE]
> **Matérn vs. Gaussian Limit and Range Parameter Scaling:**
> With the standard $\sqrt{2\nu}$ scaling in the Matérn argument, taking the limit as $\nu \to \infty$ yields:
> $$
> \lim_{\nu \to \infty} C_{\text{Matérn}}(h; \nu, \phi) = \sigma^2 \exp\left(-\frac{h^2}{2\phi^2}\right)
> $$
> When compared to the conventional Gaussian covariance formula $C_{\text{Gauss}}(h) = \sigma^2 \exp\left(-\frac{h^2}{\phi_{\text{Gauss}}^2}\right)$, the two parameterizations match via $\phi_{\text{Gauss}} = \sqrt{2}\phi_{\text{Matérn}}$. Under this matching, the practical range (where correlation drops to 5%) is $h_{\text{eff}} = \sqrt{3}\phi_{\text{Gauss}} = \sqrt{6}\phi_{\text{Matérn}} \approx 2.45\phi_{\text{Matérn}}$.

#### The Matérn Covariance Class: From Accessible Polynomials to the Unified Family
In applied spatial statistics and machine learning, practitioners almost universally work with the closed-form **half-integer** cases of the Matérn class ($\nu = p + 1/2$ for integer $p \ge 0$). For these values, the modified Bessel function simplifies into an elementary polynomial multiplied by an exponential decay factor, completely eliminating the need for special-function software libraries.

Before introducing the general parent formula, we examine the three canonical workhorse models:
1. **Rough / Non-Differentiable ($\nu = 1/2$, Exponential Model):**
   Setting $\nu = 1/2$ produces the classical Exponential covariance:
   $$
   C(h; \nu=1/2) = \sigma^2 \exp\left(-\frac{h}{\phi}\right)
   $$
   Trajectories generated by this kernel are continuous but nowhere mean-square differentiable, capturing jagged, rough physical landscapes (such as Ornstein-Uhlenbeck processes).
2. **Smooth / Once Mean-Square Differentiable ($\nu = 3/2$):**
   Setting $\nu = 3/2$ yields an elementary linear polynomial multiplied by an exponential decay:
   $$
   C(h; \nu=3/2) = \sigma^2 \left( 1 + \frac{\sqrt{3} h}{\phi} \right) \exp\left(-\frac{\sqrt{3} h}{\phi}\right)
   $$
   Trajectories are once mean-square differentiable, providing continuous spatial velocity and gradients while avoiding numerical stiffness.
3. **Very Smooth / Twice Mean-Square Differentiable ($\nu = 5/2$):**
   Setting $\nu = 5/2$ yields a quadratic polynomial multiplied by an exponential decay:
   $$
   C(h; \nu=5/2) = \sigma^2 \left( 1 + \frac{\sqrt{5} h}{\phi} + \frac{5 h^2}{3\phi^2} \right) \exp\left(-\frac{\sqrt{5} h}{\phi}\right)
   $$
   Trajectories are twice mean-square differentiable, possessing continuous spatial curvature.
4. **Infinitely Differentiable ($\nu \to \infty$, Gaussian Model):**
   As $\nu \to \infty$, the kernel converges uniformly to the Gaussian covariance:
   $$
   \lim_{\nu \to \infty} C(h; \nu) = \sigma^2 \exp\left(-\frac{h^2}{2\phi^2}\right)
   $$

The general Matérn class unifies all these cases into a single continuous parametric family, allowing the smoothness parameter $\nu > 0$ to be estimated directly from empirical data via maximum likelihood:
$$
C(h) = \frac{\sigma^2}{2^{\nu-1}\Gamma(\nu)} \left( \frac{\sqrt{2\nu} h}{\phi} \right)^\nu K_\nu\left( \frac{\sqrt{2\nu} h}{\phi} \right)
$$

##### Term-by-Term Equation Anatomy: The Matérn Covariance Kernel
- $C(h)$: The isotropic covariance between two spatial locations separated by scalar Euclidean distance $h = \|\mathbf{h}\| \ge 0$.
- $\sigma^2$: The partial sill (variance of the continuous spatial random process $\delta(\mathbf{s})$), setting the baseline amplitude at zero distance: $C(0) = \sigma^2$.
- $\phi$: The spatial range parameter ($\phi > 0$). Scales distance to determine the spatial rate of correlation decay; larger $\phi$ indicates longer-range spatial persistence.
- $\nu$: The smoothness parameter ($\nu > 0$). Controls the mean-square differentiability and fractal roughness of the process: $\{Z(\mathbf{s})\}$ is $m$-times mean-square differentiable if and only if $\nu > m$.
- $\frac{1}{2^{\nu-1}\Gamma(\nu)}$: The exact analytical normalization constant incorporating Euler's gamma function $\Gamma(\nu)$ to ensure $C(0) = \sigma^2$ identically across all choices of $\nu > 0$.
- $\frac{\sqrt{2\nu} h}{\phi}$: The dimensionless scaled distance argument, parameterized with $\sqrt{2\nu}$ so that $\phi$ maintains a consistent practical range interpretation ($h_{\text{eff}} \approx 3\phi$) across varying $\nu$.
- $K_\nu(\cdot)$: The modified Bessel function of the second kind of order $\nu$. Solves the modified Bessel differential equation, governing both the polynomial behavior of the covariance near the origin and its exponential decay at long distances.

> [!NOTE]
> **From the Lecture Hall: What $K_\nu$ Is and Why You Rarely Need It**
>
> Students are often intimidated by the modified Bessel function $K_\nu(z)$. Formally, $K_\nu(z)$ is a solution to the modified Bessel differential equation $z^2 y'' + z y' - (z^2 + \nu^2) y = 0$ that decays exponentially as $z \to \infty$:
> $$
> K_\nu(z) = \int_0^\infty e^{-z \cosh t} \cosh(\nu t) \, dt, \quad z > 0
> $$
> But here is the practical secret: **you almost never need to compute $K_\nu(z)$ directly in production code.** Whenever $\nu = p + 1/2$ is a half-integer ($p = 0, 1, 2, \dots$), $K_\nu(z)$ reduces by elementary algebra to $\sqrt{\pi / (2z)} \, e^{-z} \sum_{k=0}^p \frac{(p+k)!}{k!(p-k)! (2z)^k}$.
>
> In practice, selecting $\nu \in \{0.5, 1.5, 2.5\}$ gives you rough, smooth, and very smooth fields with pure elementary functions (polynomials times exponentials). You only need general $\nu$ numerical evaluation if you are doing continuous gradient-based maximum likelihood estimation of $\nu$ over $\mathbb{R}^+$.

The isotropic $d$-dimensional spectral density of the Matérn covariance is known in closed form:
$$
f(\boldsymbol{\omega}) = \frac{\sigma^2 \Gamma\left(\nu + \frac{d}{2}\right) (2\nu)^\nu}{\pi^{d/2} \Gamma(\nu) \phi^{2\nu}} \left( \frac{2\nu}{\phi^2} + \|\boldsymbol{\omega}\|^2 \right)^{-\left(\nu + \frac{d}{2}\right)}
$$
Because $\sigma^2 > 0$, $\nu > 0$, and $\phi > 0$, this spectral density is strictly positive for all radial frequencies $\rho = \|\boldsymbol{\omega}\| \in [0, \infty)$ and in all dimensions $d \ge 1$. Hence, by Bochner's theorem, **the Matérn covariance is strictly positive definite in all Euclidean dimensions $\mathbb{R}^d$**.

#### The Spherical Model Dimension Barrier (Valid Only for $d \le 3$)
The Spherical model is derived as the normalized volume of intersection between two solid spheres of diameter $a$ in $\mathbb{R}^d$ separated by distance $h$. With nugget effect $\tau^2$, partial sill $\sigma^2$, and range $a$, the semivariogram is:
$$
\gamma(h) = \begin{cases}
0, & h = 0 \\
\tau^2 + \sigma^2 \left[ \frac{3}{2}\left(\frac{h}{a}\right) - \frac{1}{2}\left(\frac{h}{a}\right)^3 \right], & 0 < h \le a \\
\tau^2 + \sigma^2, & h > a
\end{cases}
$$
and corresponding covariance in $\mathbb{R}^3$ for $h \le a$ is:
$$
C(h) = \sigma^2 \left( 1 - \frac{3h}{2a} + \frac{h^3}{2a^3} \right) \mathbb{I}_{\{h \le a\}}
$$

##### Term-by-Term Equation Anatomy: The Spherical Semivariogram
- $\gamma(h)$: The isotropic semivariance at scalar Euclidean separation distance $h = \|\mathbf{h}\|$.
- $\tau^2$: The nugget effect ($\tau^2 \ge 0$). An instantaneous jump at infinitesimal separation ($h \to 0^+$) reflecting microscale variation and measurement error.
- $\sigma^2$: The partial sill (variance of the continuous spatial field). The total sill is the sum $\tau^2 + \sigma^2$.
- $a$: The strict practical range ($a > 0$). The separation distance at which spatial correlation becomes exactly zero and the semivariogram reaches its plateau.
- $\frac{3}{2}\left(\frac{h}{a}\right) = 1.5(h/a)$: The linear ramp term. Dominates behavior near the origin ($h \to 0$), establishing a strictly positive tangent slope $\gamma'(0^+) = \frac{3\sigma^2}{2a}$. This linear origin implies that sample realizations are mean-square continuous but nowhere mean-square differentiable.
- $-\frac{1}{2}\left(\frac{h}{a}\right)^3 = -0.5(h/a)^3$: The cubic correction term. Arises geometrically from the 3D spherical cap intersection volume, curving the function smoothly into the horizontal sill so that its derivative vanishes smoothly at the range: $\gamma'(a^-) = 0$.

##### Why the Spherical Model Fails for $d \ge 4$: The Nesting Principle and Matrix Indefiniteness
A foundational property of spatial covariance functions is the **nesting principle**: if a function $C(h)$ is positive definite in $\mathbb{R}^d$, it is automatically positive definite in any lower-dimensional subspace $\mathbb{R}^k$ ($k < d$), because any configuration of $n$ points in $\mathbb{R}^k$ can be embedded isometrically into $\mathbb{R}^d$ by appending zero coordinates.

However, the converse is **false**: a function valid in $\mathbb{R}^3$ is *not* guaranteed to remain positive definite in $\mathbb{R}^4$. The 3-dimensional spherical polynomial $C(h) = \sigma^2 (1 - 1.5(h/a) + 0.5(h/a)^3)$ represents the overlap volume of two spheres in $\mathbb{R}^3$. In $\mathbb{R}^4$, the intersection volume of two 4D hyperspheres is governed by an entirely different polynomial.

When the 3D spherical covariance formula is evaluated on $\mathbb{R}^d$ for $d \ge 4$, its Fourier transform (spectral density) develops negative values at high radial frequencies (Gneiting 1999; Wendland 2005). By Bochner's theorem, this spectral negativity means the function is **not positive definite in $\mathbb{R}^d$ for any $d \ge 4$**.

We can understand why compact support creates indefiniteness through an intuitive finite star configuration:
Consider a central point $\mathbf{s}_0 = \mathbf{0}$ and $2d$ peripheral points placed along the positive and negative coordinate axes at distance $r = a / \sqrt{2} \approx 0.707a$ from the origin:
$$
\mathbf{s}_{2m-1} = r \mathbf{e}_m, \quad \mathbf{s}_{2m} = -r \mathbf{e}_m \quad (m = 1, \dots, d)
$$
1. The distance between the central point and any peripheral point is $\|\mathbf{s}_0 - \mathbf{s}_i\| = r < a$. Each peripheral point retains positive correlation with the center:
   $$
   \rho = \frac{C(r)}{\sigma^2} = 1 - \frac{3}{2\sqrt{2}} + \frac{1}{4\sqrt{2}} = 1 - \frac{5}{4\sqrt{2}} \approx 0.11612 > 0
   $$
   so $\operatorname{Cov}(Z(\mathbf{s}_0), Z(\mathbf{s}_i)) = \rho \sigma^2$.
2. The distance between any two distinct peripheral points on different coordinate axes is $\sqrt{r^2 + r^2} = r\sqrt{2} = a$. Between opposite points on the same axis it is $2r = a\sqrt{2} > a$.
3. Because the spherical covariance has compact support ($C(h) = 0$ for $h \ge a$), all off-diagonal covariances between peripheral points are identically zero: $\operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = 0$ for all $i \ne j \in \{1, \dots, 2d\}$. The peripheral points form an orthogonal (uncorrelated) coordinate frame.
4. Now consider the linear predictor of the central value from the peripheral points:
   $$
   Y = Z(\mathbf{s}_0) - \rho \sum_{i=1}^{2d} Z(\mathbf{s}_i)
   $$
   Expanding the variance of $Y$:
   $$
   \begin{aligned}
   \operatorname{Var}(Y) &= \operatorname{Var}(Z(\mathbf{s}_0)) - 2\rho \sum_{i=1}^{2d} \operatorname{Cov}(Z(\mathbf{s}_0), Z(\mathbf{s}_i)) + \rho^2 \sum_{i=1}^{2d} \operatorname{Var}(Z(\mathbf{s}_i)) \\
   &= \sigma^2 - 2\rho [2d \cdot \rho \sigma^2] + \rho^2 [2d \cdot \sigma^2] \\
   &= \sigma^2 [1 - 2d \rho^2]
   \end{aligned}
   $$
5. For the covariance kernel to be valid, $\operatorname{Var}(Y) \ge 0$ must hold for all configurations, requiring $2d \rho^2 \le 1$, or:
   $$
   2d \le \frac{1}{\rho^2} = \frac{1}{(0.11612)^2} \approx 74.2 \implies d \le 37
   $$
   This star configuration proves that compact support of this geometry cannot hold for arbitrary dimensions, failing by $d \ge 38$. When combined with the continuous spectral analysis of Gneiting (1999), the exact threshold is even sharper: the negative Fourier ripples appear immediately at **$d = 4$**.

Therefore, **the 3D spherical model is strictly invalid for $d \ge 4$**. Never use the spherical model for 4D spatio-temporal datasets $(x, y, z, t)$.

> [!NOTE]
> **Wait—Why Did We Do That?**
> **FAQ 2: Why is the spherical model valid only in dimensions $d \le 3$, and what happens in $\mathbb{R}^4$?**
>
> In mining and petroleum geostatistics, the spherical model is by far the most popular empirical semivariogram. Why does it abruptly fail if applied in 4 dimensions (for example, in spatio-temporal modeling where a coordinate vector is $\mathbf{s} = (s_x, s_y, s_z, t)' \in \mathbb{R}^4$)?
>
> The reason lies in the geometry of spherical intersections and positive definiteness:
> 1. **Geometric Intersection Volume:** In 3D space, the spherical covariance is derived as the normalized volume of overlap between two solid spheres of diameter $a$ separated by distance $h$. In $\mathbb{R}^3$, this intersection volume is the polynomial $1 - 1.5(h/a) + 0.5(h/a)^3$.
> 2. **Spectral Positivity and Eigenvalues:** An isotropic covariance function is positive definite on $\mathbb{R}^d$ if and only if every finite covariance matrix $\mathbf{V}$ has non-negative eigenvalues. In $\mathbb{R}^1, \mathbb{R}^2$, and $\mathbb{R}^3$, the 3D spherical polynomial guarantees non-negative quadratic forms for all configurations. But in $\mathbb{R}^4$, geometric packing allows points to be sufficiently separated that their mutual correlations vanish while their correlations with a central point remain positive.
> 3. **The Dimension Barrier in $\mathbb{R}^4$:** As shown in the spectral density and finite configuration analyses above, this causes the assembled spatial covariance matrix $\mathbf{V}$ to develop negative eigenvalues for space-time data configurations.
> 
> If an analyst applies the 3D spherical model to a 4D spatio-temporal dataset, ordinary kriging systems will produce negative prediction variances ($\sigma_{\text{OK}}^2 < 0$), and Cholesky simulation algorithms will crash with numerical factorization errors. For $d \ge 4$, analysts must use models that are valid in all dimensions, such as the Matérn, Exponential, or Gaussian families.

---

#### 2.3.2 How to Compute It: Evaluating Matérn Covariances

The following dependency-free Python procedure evaluates the generalized Matérn covariance. It uses exact closed-form algebraic expressions for the standard half-integers $\nu \in \{0.5, 1.5, 2.5\}$, and computes arbitrary non-half-integer values of $\nu > 0$ via robust trapezoidal quadrature of the integral definition of $K_\nu(z)$.

```python
import math

def matern_covariance_evaluator(h, sigma2, phi, nu=1.5):
    """
    Evaluates the Matérn covariance kernel C(h) for arbitrary distances.
    Includes explicit analytical implementations for half-integers nu = 0.5, 1.5, 2.5,
    and standard-library numerical quadrature of K_nu(z) for arbitrary positive nu.
    
    Parameters:
        h:      Scalar Euclidean distance (h >= 0)
        sigma2: Partial sill variance (sigma2 > 0)
        phi:    Spatial range parameter (phi > 0)
        nu:     Smoothness parameter (nu > 0)
        
    Returns:
        C: Covariance value at lag h.
    """
    if h == 0.0:
        return sigma2
        
    scaled_h = h / phi
    
    if abs(nu - 0.5) < 1e-6:
        # Nu = 1/2: Exponential model (continuous, nowhere differentiable)
        # C(h) = sigma2 * exp(-h / phi)
        return sigma2 * math.exp(-scaled_h)
        
    elif abs(nu - 1.5) < 1e-6:
        # Nu = 3/2: Once mean-square differentiable
        # C(h) = sigma2 * (1 + sqrt(3)*h/phi) * exp(-sqrt(3)*h/phi)
        sqrt3 = math.sqrt(3.0)
        arg = sqrt3 * scaled_h
        return sigma2 * (1.0 + arg) * math.exp(-arg)
        
    elif abs(nu - 2.5) < 1e-6:
        # Nu = 5/2: Twice mean-square differentiable
        # C(h) = sigma2 * (1 + sqrt(5)*h/phi + 5*h^2 / (3*phi^2)) * exp(-sqrt(5)*h/phi)
        sqrt5 = math.sqrt(5.0)
        arg = sqrt5 * scaled_h
        poly = 1.0 + arg + (5.0 * (scaled_h ** 2) / 3.0)
        return sigma2 * poly * math.exp(-arg)
        
    else:
        # General nu > 0: Numerical quadrature of the modified Bessel function K_nu(z)
        # Integral representation: K_nu(z) = integral_0^infinity exp(-z*cosh(t)) * cosh(nu*t) dt
        z = math.sqrt(2.0 * nu) * scaled_h
        
        # Trapezoidal quadrature up to t_max = 12 (integrand decays exponentially)
        n_steps = 400
        t_max = 12.0
        dt = t_max / n_steps
        
        integral = 0.5 * (math.exp(-z) + math.exp(-z * math.cosh(t_max)) * math.cosh(nu * t_max))
        for step in range(1, n_steps):
            t = step * dt
            integral += math.exp(-z * math.cosh(t)) * math.cosh(nu * t)
        k_nu_z = integral * dt
        
        # Prefactor: sigma2 * 2^(1 - nu) / Gamma(nu) * z^nu * K_nu(z)
        coeff = (sigma2 * (2.0 ** (1.0 - nu)) / math.gamma(nu)) * (z ** nu)
        return coeff * k_nu_z
```

---

#### 2.3.3 What Can Go Wrong: The Gaussian Hazard and the Dimension Barrier

```
Smoothness Spectrum:
nu = 0.5 (Exponential)  ========> Continuous but nowhere differentiable (Jagged, rough)
nu = 1.5 (Matérn 3/2)   ========> Once mean-square differentiable (Continuous velocity)
nu = 2.5 (Matérn 5/2)   ========> Twice mean-square differentiable (Continuous curvature)
nu -> inf (Gaussian)    ========> Infinitely differentiable (Analytical, hyper-smooth)
```

**The Gaussian Hazard:**
Practitioners frequently choose the Gaussian covariance model $C(h) = \sigma^2 \exp(-h^2/\phi^2)$ assuming its smoothness is an asset. In truth, the Gaussian model is pathologically over-smooth for most physical phenomena:
- Because its spectral density decays at an extreme Gaussian rate ($f(\boldsymbol{\omega}) \propto e^{-\phi^2 \|\boldsymbol{\omega}\|^2 / 4}$ in angular frequency), the random field is mean-square analytic. Knowing the field and its derivatives in an arbitrarily small spatial neighborhood completely determines the field everywhere.
- In numerical prediction, the rows of $\mathbf{V}$ become virtually identical when sample points are clustered. The spatial covariance matrix condition number can easily exceed $\kappa(\mathbf{V}) > 10^{16}$, causing standard Cholesky and LU matrix solvers to produce nonsensical, wildly oscillating negative kriging weights.
- Michael Stein (1999) famously cautioned against the uncritical use of the Gaussian covariance for environmental and geophysical fields. In real-world data, the Matérn family with $\nu = 1.5$ or $\nu = 2.5$ provides the ideal balance of realistic physical differentiability and well-conditioned numerical stability.

> [!TIP]
> **Which Model Should I Use? A Practical Decision Guide**
>
> 1. **Default Recommendation: Matérn $\nu = 1.5$ or $\nu = 2.5$.**
>    For continuous environmental, geological, or atmospheric fields, start with Matérn $\nu = 1.5$ (once differentiable) or $\nu = 2.5$ (twice differentiable). They capture realistic smoothness without numerical instability.
> 2. **Rough Processes: Exponential ($\nu = 0.5$).**
>    Use when the phenomenon exhibits sharp fractures, jagged variations, or high microscale irregularity where spatial velocity cannot be assumed continuous.
> 3. **Smooth Physics: Gaussian ($\nu \to \infty$).**
>    Use *only* when the underlying physical mechanism is governed by smooth differential equations (e.g., barometric pressure or temperature across smooth continental scales), and verify that inter-point sampling distances are large enough to avoid $\kappa(\mathbf{V}) > 10^{14}$.
> 4. **Strict Compact Support: Spherical.**
>    Use only in dimensions $d \le 3$, and only when a strict cut-off distance $a$ beyond which points are completely uncorrelated is physically justified and desired for sparse matrix computations. Never use in 4D spatio-temporal modeling.

---

### 2.4 Regularity at the Origin and Mean-Square Differentiability

#### 2.4.1 Why We Need This: How the Origin Dictates Spatial Smoothness

The behavior of the semivariogram $\gamma(\mathbf{h})$ as $\|\mathbf{h}\| \to 0$ governs the local continuity and differentiability of the random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$.

> [!NOTE]
> **THEOREM 2.5: Mean-Square Continuity Criterion**
>
> A second-order stationary random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ is mean-square continuous at every location $\mathbf{s} \in D$ if and only if its covariance function $C(\mathbf{h})$ is continuous at the origin $\mathbf{h} = \mathbf{0}$, or equivalently:
> $$
> \lim_{\|\mathbf{h}\| \to 0} \gamma(\mathbf{h}) = 0
> $$

**Analytical Proof:**
Recall from Chapter 1 (Theorem 1.4.1) that the mean-square increment variance of a stationary field satisfies:
$$
\operatorname{E}[(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s}))^2] = \operatorname{Var}(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})) = 2\gamma(\mathbf{h}) = 2[C(\mathbf{0}) - C(\mathbf{h})]
$$
By definition of $L^2$ convergence, $Z(\mathbf{s} + \mathbf{h}) \xrightarrow{\text{m.s.}} Z(\mathbf{s})$ as $\|\mathbf{h}\| \to 0$ if and only if $\lim_{\|\mathbf{h}\| \to 0} \operatorname{E}[(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s}))^2] = 0$. This holds if and only if $\lim_{\|\mathbf{h}\| \to 0} \gamma(\mathbf{h}) = 0$, which is equivalent to $\lim_{\|\mathbf{h}\| \to 0} C(\mathbf{h}) = C(\mathbf{0})$. Thus, continuity of the single deterministic scalar function $C(\mathbf{h})$ at the single point $\mathbf{h} = \mathbf{0}$ guarantees mean-square continuity everywhere across $\mathbb{R}^d$. $\blacksquare$

> [!NOTE]
> **THEOREM 2.6: Mean-Square Differentiability via Second-Order Covariance Derivatives**
>
> Let $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ be a stationary random field with covariance function $C(\mathbf{h})$. The directional mean-square derivative:
> $$
> \frac{\partial Z(\mathbf{s})}{\partial s_k} = \lim_{h \to 0} \frac{Z(\mathbf{s} + h \mathbf{e}_k) - Z(\mathbf{s})}{h}
> $$
> exists in mean square if and only if the second partial derivative $\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}$ exists and is finite, or equivalently:
> $$
> \lim_{h \to 0} \frac{\gamma(h\mathbf{e}_k)}{h^2} < \infty
> $$

**Analytical Proof:**
Let $D_h = \frac{Z(\mathbf{s} + h\mathbf{e}_k) - Z(\mathbf{s})}{h}$ denote the spatial difference quotient along coordinate axis $k$. A family of random variables $D_h$ converges in mean square as $h \to 0$ to some random variable in $L^2$ if and only if it satisfies the **Cauchy criterion in $L^2$**:
$$
\lim_{h, h' \to 0} \operatorname{E}\left[(D_h - D_{h'})^2\right] = 0
$$
Because the field has constant mean $\mu$, $\operatorname{E}[D_h] = 0$ for all $h \ne 0$. Expanding the mean-square difference:
$$
\operatorname{E}\left[(D_h - D_{h'})^2\right] = \operatorname{Var}(D_h) + \operatorname{Var}(D_{h'}) - 2\operatorname{Cov}(D_h, D_{h'})
$$
First, compute the variance of $D_h$:
$$
\operatorname{Var}(D_h) = \frac{1}{h^2} \operatorname{Var}(Z(\mathbf{s} + h\mathbf{e}_k) - Z(\mathbf{s})) = \frac{2\gamma(h\mathbf{e}_k)}{h^2} = -\frac{C(h\mathbf{e}_k) - 2C(\mathbf{0}) + C(-h\mathbf{e}_k)}{h^2}
$$
By the symmetric definition of the second derivative, as $h \to 0$, if $\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}$ exists and is finite, then:
$$
\lim_{h \to 0} \operatorname{Var}(D_h) = -\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}
$$
Next, evaluate the covariance between increments at different step sizes $h$ and $h'$:
$$
\begin{aligned}
\operatorname{Cov}(D_h, D_{h'}) &= \frac{1}{h h'} \operatorname{Cov}\left(Z(\mathbf{s} + h\mathbf{e}_k) - Z(\mathbf{s}), \; Z(\mathbf{s} + h'\mathbf{e}_k) - Z(\mathbf{s})\right) \\
&= \frac{1}{h h'} \left[ C((h - h')\mathbf{e}_k) - C(h\mathbf{e}_k) - C(-h'\mathbf{e}_k) + C(\mathbf{0}) \right]
\end{aligned}
$$
Using the Taylor expansion of $C(\delta \mathbf{e}_k)$ near the origin: since $C$ is an even function ($C(\mathbf{h}) = C(-\mathbf{h})$), any first derivative vanishes ($C'(0) = 0$). Thus $C(\delta \mathbf{e}_k) = C(\mathbf{0}) + \frac{1}{2} C_k''(0) \delta^2 + o(\delta^2)$, where $C_k''(0) = \frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}$. Substituting this into the bracketed numerator:
$$
\begin{aligned}
C((h-h')\mathbf{e}_k) - C(h\mathbf{e}_k) - C(-h'\mathbf{e}_k) + C(\mathbf{0}) &= \frac{1}{2} C_k''(0) \left[ (h-h')^2 - h^2 - (h')^2 \right] + o(h^2 + (h')^2) \\
&= \frac{1}{2} C_k''(0) \left[ -2hh' \right] + o(hh') \\
&= -h h' C_k''(0) + o(hh')
\end{aligned}
$$
Dividing by $h h'$ and taking $h, h' \to 0$:
$$
\lim_{h, h' \to 0} \operatorname{Cov}(D_h, D_{h'}) = -C_k''(0) = -\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}
$$
Substituting all three terms into the Cauchy expansion:
$$
\lim_{h, h' \to 0} \operatorname{E}\left[(D_h - D_{h'})^2\right] = \left(-\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}\right) + \left(-\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}\right) - 2\left(-\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}\right) = 0
$$
Therefore, $\{D_h\}$ is a Cauchy sequence in the complete Hilbert space $L^2$, converging to a mean-square limit if and only if the second partial derivative $\frac{\partial^2 C(\mathbf{0})}{\partial h_k^2}$ exists and is finite. $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: Smoothness Lives in the Second Difference**
>
> Why does differentiability of the random field require *two* derivatives of the covariance function?
>
> In ordinary calculus, checking whether a single deterministic curve $f(x)$ has a derivative at $x_0$ requires only that $f'(x_0)$ exists. But a random field $Z(\mathbf{s})$ is an infinite collection of random variables. When we compute the *variance* of the difference quotient $(Z(\mathbf{s}+h) - Z(\mathbf{s}))/h$, we square the denominator:
> $$
> \operatorname{Var}\left(\frac{\Delta Z}{h}\right) = \frac{\operatorname{Var}(\Delta Z)}{h^2} = \frac{2\gamma(h)}{h^2} = -\frac{C(h) - 2C(0) + C(-h)}{h^2}
> $$
> Notice the numerator: $C(h) - 2C(0) + C(-h)$ is a **second central difference** of $C$. Dividing by $h^2$ produces the exact definition of the second derivative $C''(0)$. In the geometry of Hilbert spaces, computing the variance of a first-order rate of change requires second-order differentiation of the inner product!

#### Curvature Classification at the Origin
Expanding the isotropic semivariogram $\gamma(h)$ in a power series near $h=0$:
$$
\gamma(h) = \tau^2 + c h^\alpha + o(h^\alpha) \quad \text{as } h \to 0^+
$$
1. **Discontinuous Origin ($\tau^2 > 0$):**
   $\lim_{h \to 0^+} \gamma(h) = \tau^2 > 0$. The field is not even mean-square continuous. Trajectories exhibit jagged white-noise micro-fluctuations.
2. **Linear Origin ($\alpha = 1$, $\tau^2 = 0$):**
   $\gamma(h) \approx c h$. The one-sided first derivative $\gamma'(0^+) = c > 0$, but the ratio $\gamma(h)/h^2 \to \infty$. The field is mean-square continuous, but **nowhere mean-square differentiable** (e.g., Exponential model, Matérn $\nu = 0.5$). Trajectories resemble continuous Brownian / Ornstein-Uhlenbeck paths with sharp fractal-like edges.
3. **Sub-Parabolic Origin ($1 < \alpha < 2$, $\tau^2 = 0$):**
   $\gamma(h) \approx c h^\alpha$ with $1 < \alpha < 2$. The field is mean-square continuous with $\gamma'(0^+) = 0$, but because $\gamma(h)/h^2 \propto h^{\alpha - 2} \to \infty$ as $h \to 0$, the second derivative does not exist. The field remains **nowhere mean-square differentiable** (e.g., power-law variograms with $1 < \alpha < 2$, or Matérn with $0.5 < \nu < 1.0$).
4. **Parabolic Origin ($\alpha = 2$, $\tau^2 = 0$):**
   $\gamma(h) \approx c h^2$. Here $\gamma'(0) = 0$ and $\lim_{h \to 0} \gamma(h)/h^2 = c < \infty$. The covariance function is twice differentiable at $\mathbf{h} = \mathbf{0}$. The random field is **mean-square differentiable** with continuous, flowing surface trajectories (e.g., Matérn with $\nu > 1$, including $\nu = 1.5, 2.5$, and the Gaussian model).

---

#### 2.4.2 How to Compute It: Diagnosing Origin Curvature

The following diagnostic procedure fits a log-log regression to empirical semivariances at the shortest sampling lags to estimate the local power $\alpha$, classifying the physical origin behavior.

```python
def empirical_origin_curvature_diagnostic(lags, semivariances):
    """
    Diagnoses the origin curvature of an empirical semivariogram cloud.
    Estimates the exponent alpha in: gamma(h) ~= c * h^alpha as h -> 0.
    
    Caveats and Diagnostic Best Practices:
      1. Shortest Lags: Regression must be restricted to the innermost lags
         (h << practical range) where higher-order polynomial terms are negligible.
      2. Nugget Handling: If a nugget effect tau^2 > 0 is present, log-log
         estimation on raw semivariances yields alpha -> 0. Subtracting the
         estimated nugget (gamma(h) - tau^2) is essential prior to power fitting.
      3. Sparse Data Noise: With small sample sizes, empirical semivariance at
         the innermost bin has high sampling variance; bin counts should be checked.
    
    Parameters:
        lags: Sorted list of small lag distances [h_1, h_2, ..., h_k].
        semivariances: Corresponding empirical semivariances [g_1, g_2, ..., g_k].
        
    Returns:
        alpha: Estimated local power exponent.
        classification: String ("Discontinuous", "Linear", "Intermediate", or "Parabolic").
    """
    import math
    
    # Restrict to the smallest positive lags (first 5 lags)
    k = min(5, len(lags))
    
    # Detect substantial nugget effect at the first lag
    if semivariances[0] > 0.15 * semivariances[-1]:
        return 0.0, "Discontinuous (Substantial Nugget Effect tau^2 > 0)"
        
    log_h = []
    log_g = []
    
    for i in range(k):
        if lags[i] > 1e-6 and semivariances[i] > 1e-6:
            log_h.append(math.log(lags[i]))
            log_g.append(math.log(semivariances[i]))
            
    m = len(log_h)
    if m < 2:
        return 1.0, "Insufficient short lags to diagnose curvature"
        
    # Ordinary Least Squares slope on log-log coordinates:
    # log(g) = log(c) + alpha * log(h)
    mean_x = sum(log_h) / m
    mean_y = sum(log_g) / m
    
    ss_xx = sum((log_h[i] - mean_x) ** 2 for i in range(m))
    ss_xy = sum((log_h[i] - mean_x) * (log_g[i] - mean_y) for i in range(m))
    
    alpha = ss_xy / ss_xx if ss_xx != 0 else 1.0
    
    if alpha < 1.2:
        classification = "Linear Origin (MS-Continuous, Non-Differentiable)"
    elif alpha < 1.8:
        classification = "Intermediate Origin (Continuous, Non-Differentiable, 1 < alpha < 2)"
    else:
        classification = "Parabolic Origin (Mean-Square Differentiable)"
        
    return alpha, classification
```

---

#### 2.4.3 What Can Go Wrong: The Nugget Effect and Measurement Error

In physical core sampling (e.g., mineral grades or soil contaminants), real observations taken just centimeters apart rarely match identically. This discontinuity at the origin is known as the **nugget effect** $\tau^2 = \lim_{h \to 0^+} \gamma(h)$.

The nugget effect arises from two distinct physical phenomena:
$$
\tau^2 = \sigma_{\text{micro}}^2 + \sigma_{\text{measurement}}^2
$$
1. **Micro-Scale Spatial Variation ($\sigma_{\text{micro}}^2$):** Real physical processes contain high-frequency structures (e.g., mineralized veinlets, fine sediment fissures) operating at spatial scales smaller than the shortest inter-sample spacing $h_{\min} = \min_{i \ne j} \|\mathbf{s}_i - \mathbf{s}_j\|$. Because our sampling grid cannot resolve these wavelengths, the spatial autocorrelation drops to zero before the first lag bin, appearing as an instantaneous step discontinuity.
2. **Measurement Error ($\sigma_{\text{measurement}}^2$):** Analytical assay imprecision, instrument noise, and physical core handling errors introduce zero-mean uncorrelated perturbations $\epsilon(\mathbf{s}) \sim (0, \sigma_{\text{meas}}^2)$.

Distinguishing between micro-scale variance and measurement error is impossible from single-sample spatial designs alone; it requires repeated duplicate assays collected at identical locations $\mathbf{s}_i$.

---

### 2.5 Spatial Anisotropy: Geometric, Zonal, and General Linear

#### 2.5.1 Why We Need This: When Direction Matters

Natural spatial processes rarely disperse uniformly in all directions. Geological stratification, prevailing wind currents, fluvial deposition, and tectonic stress introduce directional dependence, termed **spatial anisotropy**.

> [!NOTE]
> **DEFINITION 2.3: Spatial Isotropy vs. Anisotropy**
>
> A spatial covariance function $C(\mathbf{h})$ is:
> 1. **Isotropic** if it depends solely on Euclidean length: $C(\mathbf{h}) = C_0(\|\mathbf{h}\|)$.
> 2. **Anisotropic** if it depends on both distance and orientation: $C(\mathbf{h}) \ne C(\mathbf{R}\mathbf{h})$ for some rotation matrix $\mathbf{R} \in \mathrm{SO}(d)$.

#### 1. Geometric Anisotropy
In geometric anisotropy, the directional semivariograms exhibit the **same sill $\sigma^2 + \tau^2$ in all directions**, but have direction-dependent range parameters $\phi(\mathbf{u})$. The contours of equal spatial correlation form concentric ellipsoids in $\mathbb{R}^d$.

> [!NOTE]
> **THEOREM 2.7: Metric Tensor Transformation for Geometric Anisotropy**
>
> Let $C_0(h)$ be an isotropic covariance function that is positive definite on $\mathbb{R}^d$. Let $\mathbf{A} \in \mathbb{R}^{d \times d}$ be a symmetric, positive definite metric matrix. Then the anisotropic kernel:
> $$
> C(\mathbf{h}) = C_0\left( \sqrt{\mathbf{h}' \mathbf{A} \mathbf{h}} \right)
> $$
> is guaranteed to be positive definite on $\mathbb{R}^d$.

**Analytical Proof:**
Because $\mathbf{A}$ is symmetric and positive definite, by the spectral theorem it admits the orthogonal decomposition:
$$
\mathbf{A} = \mathbf{R} \boldsymbol{\Lambda} \mathbf{R}' = \mathbf{R} \operatorname{diag}(a_1^{-2}, \dots, a_d^{-2}) \mathbf{R}'
$$
where $\mathbf{R} \in \mathrm{SO}(d)$ is a rotation matrix of eigenvectors and $a_k > 0$ are the directional spatial ranges. Define the linear coordinate transformation matrix $\mathbf{M} \in \mathbb{R}^{d \times d}$:
$$
\mathbf{M} = \boldsymbol{\Lambda}^{1/2} \mathbf{R}' = \operatorname{diag}(a_1^{-1}, \dots, a_d^{-1}) \mathbf{R}'
$$
Then the quadratic form in the argument of $C_0$ can be written as:
$$
\mathbf{h}' \mathbf{A} \mathbf{h} = \mathbf{h}' \mathbf{M}' \mathbf{M} \mathbf{h} = \|\mathbf{M} \mathbf{h}\|^2
$$
Thus, $C(\mathbf{h}) = C_0(\|\mathbf{M}\mathbf{h}\|)$. 
Now let $n \ge 1$, let $\mathbf{s}_1, \dots, \mathbf{s}_n \in \mathbb{R}^d$ be arbitrary locations, and let $\mathbf{a} \in \mathbb{R}^n$. The quadratic form is:
$$
\sum_{i=1}^n \sum_{j=1}^n a_i a_j C(\mathbf{s}_i - \mathbf{s}_j) = \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_0(\|\mathbf{M}(\mathbf{s}_i - \mathbf{s}_j)\|) = \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_0(\|\mathbf{u}_i - \mathbf{u}_j\|)
$$
where $\mathbf{u}_i = \mathbf{M}\mathbf{s}_i \in \mathbb{R}^d$ are transformed spatial coordinates. Because $C_0$ is positive definite on $\mathbb{R}^d$, this quadratic form is non-negative for any set of vectors $\{\mathbf{u}_i\}_{i=1}^n$. Since $\mathbf{M}$ is non-singular ($\det(\mathbf{M}) = \prod a_k^{-1} \ne 0$), the transformation is an isomorphism. Therefore, $C(\mathbf{h})$ is strictly positive definite on $\mathbb{R}^d$. $\blacksquare$

#### 2. Zonal (Sill) Anisotropy
In zonal anisotropy, the directional semivariograms exhibit **different sills** in different spatial directions. Zonal anisotropy cannot be resolved by coordinate stretching because no linear transformation of coordinates can change the limiting value $\lim_{h \to \infty} \gamma(h \mathbf{u})$.

To preserve positive definiteness, zonal anisotropy must be modeled as a sum of nested, independent spatial sub-processes with different dimensional supports:
$$
Z(\mathbf{s}) = Z_1(\mathbf{s}) + Z_2(\mathbf{P}_z \mathbf{s})
$$
where $Z_1$ is an isotropic random field on $\mathbb{R}^3$, and $Z_2$ is an independent random field defined only on a lower-dimensional subspace (e.g., vertical depth $s_z$). The resulting valid covariance is additive:
$$
C(\mathbf{h}) = C_1(\|\mathbf{h}\|) + C_2(|h_z|)
$$
To verify that this sum is positive definite, note that for any linear combination $\sum_{i=1}^n a_i Z(\mathbf{s}_i)$, the variance decomposes as $\mathbf{a}'\mathbf{V}_1\mathbf{a} + \mathbf{a}'\mathbf{V}_2\mathbf{a} \ge 0 + 0 = 0$ because both sub-processes are independent second-order stationary fields.

---

#### 2.5.2 How to Compute It: Coordinate Transforms for Geometric Anisotropy

The following procedure implements the coordinate rotation and scaling transformation for 2D geometric anisotropy. In geographic conventions, $x$ represents Easting and $y$ represents Northing, and azimuth $\theta_{\text{az}}$ is measured in degrees clockwise from North. 

After applying this transformation, distances between transformed coordinates can be evaluated using a standard isotropic covariance model whose range parameter is set to `range_major`, because the minor axis coordinates have been scaled up by the factor `range_major / range_minor`.

```python
def transform_anisotropic_coordinates_2d(coords, azimuth_degrees, range_major, range_minor):
    """
    Transforms 2D spatial coordinates (Easting, Northing) to isotropic space.
    
    After transformation, distances should be evaluated using an isotropic
    covariance function with range parameter set to range_major.
    
    Parameters:
        coords: List of 2D points [(x_1, y_1), ..., (x_n, y_n)] where
                x = Easting, y = Northing.
        azimuth_degrees: Major axis strike angle in degrees clockwise from North.
        range_major: Spatial range along major axis (a_maj > 0).
        range_minor: Spatial range along minor axis (a_min > 0).
        
    Returns:
        transformed_coords: List of [u_iso, v_iso] coordinates in isotropic space.
    """
    import math
    
    # Convert azimuth (degrees clockwise from North) to standard Cartesian angle theta:
    # North is +y (90 deg Cartesian), East is +x (0 deg Cartesian)
    theta = math.radians(90.0 - azimuth_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Anisotropy ratio: stretches the minor axis to match the major range
    aniso_ratio = range_major / range_minor
    
    transformed_coords = []
    for pt in coords:
        x, y = pt[0], pt[1]
        
        # Step 1: Rotate coordinate frame to align with principal continuity axes
        # u_rot aligns with the major axis; v_rot aligns with the minor axis
        u_rot = x * cos_t + y * sin_t
        v_rot = -x * sin_t + y * cos_t
        
        # Step 2: Scale minor axis by the anisotropy ratio
        u_iso = u_rot
        v_iso = v_rot * aniso_ratio
        
        transformed_coords.append([u_iso, v_iso])
        
    return transformed_coords
```

---

#### 2.5.3 What Can Go Wrong: Inadmissible Directional Modeling

```
                             North (Azimuth 0°)
                                     ▲
        Minor Axis                   │                   Major Axis
    (Azimuth 315° / 135°)            │                  (Azimuth 45°)
              \                      │                      /
               \                     │                     /
                \                    │                    /
                 \                   │                   /
                  \                  │                  /
        ───────────\─────────────────┼─────────────────/───────────► East (Azimuth 90°)
                    \                │                /
                     \               │               /
                      \              │              /
                       \             │             /
                                     │
```

**The Danger of Inadmissible Anisotropy Corrections:**
A naive approach to anisotropy is to fit independent 1D semivariograms in four separate directions ($0^\circ, 45^\circ, 90^\circ, 135^\circ$), obtain directional ranges $\phi(\theta)$ and sills $\sigma^2(\theta)$, and interpolate them using an ad-hoc trigonometric formula like $\phi(\theta) = \phi_{\min} + (\phi_{\max} - \phi_{\min})|\cos\theta|$.
- **Why this fails:** While the 1D directional curves fit the sample points, their combination does not correspond to a valid $d$-dimensional Fourier transform. The assembled spatial covariance matrix $\mathbf{V}$ can produce negative eigenvalues for diagonal point configurations, causing kriging systems to crash.
- **The Golden Rule:** Always construct anisotropy through coordinate transformation matrices $\mathbf{M} = \boldsymbol{\Lambda}^{1/2}\mathbf{R}^\prime$ applied to an established positive definite isotropic kernel.

---

### Key Ideas and Takeaways: Covariance and Variogram Theory

| Concept / Object | Mathematical Condition / Definition | Key Practical Implication & Pitfall |
| :--- | :--- | :--- |
| **Positive Definite Covariance** $C(\mathbf{h})$ | $\mathbf{a}^\prime \mathbf{V} \mathbf{a} \ge 0$ for all $\mathbf{a} \in \mathbb{R}^n$; $f(\boldsymbol{\omega}) \ge 0$ (Bochner) | Guarantees $\operatorname{Var}(\sum a_i Z(\mathbf{s}_i)) \ge 0$. Ad-hoc smooth kernels often fail positive definiteness and crash solvers. |
| **Conditionally Negative Definite Variogram** $\gamma(\mathbf{h})$ | $\boldsymbol{\lambda}^\prime \boldsymbol{\Gamma} \boldsymbol{\lambda} \le 0$ for all $\sum \lambda_i = 0$; $\lim_{h \to \infty} \frac{\gamma(h)}{h^2} = 0$ | Governs spatial contrasts without estimating the global mean $\mu$. Semivariogram growth cannot exceed parabolic rate ($h^2$). |
| **Stationarity Hierarchy** | Second-order: $C(\mathbf{h}) = \sigma^2 - \gamma(\mathbf{h})$; Intrinsic: only $\gamma(\mathbf{h})$ exists | Intrinsic stationarity accommodates unbounded processes (e.g. fractional Brownian motion) where covariance and sill do not exist. |
| **Matérn Covariance Class** | $C_\nu(h) = \frac{\sigma^2}{2^{\nu-1}\Gamma(\nu)}\left(\frac{\sqrt{2\nu}h}{\phi}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}h}{\phi}\right)$ | Decouples spatial range $\phi$ from sample-path smoothness $\nu$. Unifies Exponential ($\nu=0.5$) and Gaussian ($\nu \to \infty$). |
| **Mean-Square Differentiability** | Differentiable $m$ times $\iff C^{(2m)}(0)$ exists and is finite $\iff \nu > m$ | Linear origin curvature ($h^1$) yields continuous but nowhere-differentiable paths; parabolic ($h^2$) yields smooth gradients. |
| **Geometric vs. Zonal Anisotropy** | Geometric: $C(\mathbf{h}) = C_{\text{iso}}(\|\mathbf{M}\mathbf{h}\|)$; Zonal: $C_1(\mathbf{h}) + C_2(\mathbf{h})$ | Model directional dependence via linear coordinate transforms $\mathbf{M}$ or covariance sums; never interpolate directional 1D curves ad-hoc. |
| **Nugget Regularization** $\tau^2$ | $\mathbf{V} = \mathbf{C} + \tau^2 \mathbf{I}_n$; condition bound $\kappa(\mathbf{V}) \le 1 + \frac{n\sigma^2}{\tau^2}$ | Absorbs measurement noise and microscale variance; strictly bounds matrix conditioning and prevents numerical singularity. |

---

### 2.6 Analytical Exercises and Step-by-Step Solutions

#### Exercise 2.1: Failure of Compactly Supported Covariance Models in High Dimensions
**Problem Statement:**
Prove that compactly supported spherical-type covariance models cannot remain positive definite across arbitrarily high dimensions $\mathbb{R}^d$, and explain why the 3-dimensional spherical covariance model:
$$
C(h) = \sigma^2 \left( 1 - \frac{3}{2}\frac{h}{\phi} + \frac{1}{2}\frac{h^3}{\phi^3} \right) \mathbb{I}_{\{h \le \phi\}}
$$
fails positive definiteness in higher dimensions.

**Complete Step-by-Step Solution:**
1. A candidate covariance function $C(h)$ is valid in $\mathbb{R}^d$ if and only if for all sample configurations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$ and all weight vectors $\mathbf{a} \in \mathbb{R}^n$, the variance of the linear combination $Y = \mathbf{a}^\prime\mathbf{Z}$ is non-negative:
   $$
   \operatorname{Var}(Y) = \mathbf{a}^\prime\mathbf{V}\mathbf{a} \ge 0
   $$
2. Construct a star configuration in $\mathbb{R}^d$ consisting of a central point $\mathbf{s}_0 = \mathbf{0}$ and $2d$ peripheral points placed along the positive and negative coordinate axes at distance $c = \phi / \sqrt{2}$ from the origin:
   $$
   \mathbf{s}_{2m-1} = c \mathbf{e}_m, \quad \mathbf{s}_{2m} = -c \mathbf{e}_m \quad (m = 1, \dots, d)
   $$
3. Evaluate pairwise distances between distinct peripheral points:
   - Between opposite points on the same coordinate axis: $\|\mathbf{s}_{2m-1} - \mathbf{s}_{2m}\| = 2c = \sqrt{2}\phi > \phi$.
   - Between points on different coordinate axes: $\|\pm c\mathbf{e}_j \mp c\mathbf{e}_k\| = \sqrt{c^2 + c^2} = c\sqrt{2} = \phi$.
   Since $C(h) = 0$ for all $h \ge \phi$, the covariance between every pair of distinct peripheral points is identically zero:
   $$
   \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = 0 \quad \forall i \ne j \in \{1, \dots, 2d\}
   $$
   Therefore, the $2d \times 2d$ covariance matrix of the peripheral points is the diagonal matrix $\sigma^2 \mathbf{I}_{2d}$.
4. Each peripheral point is separated from the central point by $c = \phi / \sqrt{2} < \phi$, yielding positive correlation:
   $$
   C(c) = \sigma^2 \left( 1 - \frac{3}{2\sqrt{2}} + \frac{1}{4\sqrt{2}} \right) = \sigma^2 \left( 1 - \frac{5}{4\sqrt{2}} \right) \approx 0.11612 \, \sigma^2 > 0
   $$
5. Consider the linear combination $Y = a_0 Z(\mathbf{s}_0) - \sum_{i=1}^{2d} Z(\mathbf{s}_i)$. Expanding its variance using matrix algebra:
   $$
   \operatorname{Var}(Y) = a_0^2 \sigma^2 + 2d \sigma^2 - 2 a_0 (2d) C(c)
   $$
   Setting the derivative with respect to $a_0$ to zero yields the optimal weight $a_0 = 2d \frac{C(c)}{\sigma^2}$. Substituting this value minimizes the quadratic form:
   $$
   \operatorname{Var}(Y) = 2d \sigma^2 \left[ 1 - 2d \left(\frac{C(c)}{\sigma^2}\right)^2 \right]
   $$
6. If the model were valid in arbitrary dimension $d$, $\operatorname{Var}(Y) \ge 0$ would hold for all $d$. However:
   $$
   \operatorname{Var}(Y) < 0 \iff 2d > \left(\frac{\sigma^2}{C(c)}\right)^2 \approx \frac{1}{(0.11612)^2} \approx 74.2 \implies 2d \ge 76 \implies d \ge 38
   $$
   This simple star configuration proves using only variance algebra (without Bessel transforms) that the 3D spherical model strictly violates positive definiteness for all dimensions $d \ge 38$.
7. **The Exact Dimension Threshold:** More dense spherical point packings drive this dimension lower. In fact, computing the continuous radial Fourier transform (the spectral density $f_d(\omega)$) reveals that $f_d(\omega)$ develops negative lobes whenever $d \ge 4$ (see Gneiting, 1999). Thus, the standard spherical model is valid only in dimensions $d = 1, 2, 3$. $\blacksquare$

---

#### Exercise 2.2: Mean-Square Differentiability of the Matérn Class via Taylor Expansion
**Problem Statement:**
Using Taylor series expansions of the Matérn covariance kernel near the origin ($h \to 0$), determine the mean-square differentiability of a stationary random field $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ for the canonical smoothness values $\nu = 0.5$, $\nu = 1.5$, and $\nu = 2.5$.

**Complete Step-by-Step Solution:**
1. By Theorem 2.6 (and the spectral characterization of mean-square calculus), a stationary random field $\{Z(\mathbf{s})\}$ is $m$-times mean-square differentiable if and only if the $2m$-th derivative of the isotropic covariance function $C(h)$ exists and is finite at $h = 0$. In terms of Taylor series, $C(h)$ must admit an even-power expansion up to order $h^{2m}$ without lower-order odd powers or fractional powers.
2. **Case 1: $\nu = 0.5$ (Exponential model):**
   The covariance function is $C(h) = \sigma^2 \exp(-h/\phi)$.
   Expanding the exponential in a Taylor series for $h \ge 0$:
   $$
   C(h) = \sigma^2 \left( 1 - \frac{h}{\phi} + \frac{h^2}{2\phi^2} - \mathcal{O}(h^3) \right)
   $$
   The corresponding semivariogram near the origin is:
   $$
   \gamma(h) = C(0) - C(h) = \sigma^2 \frac{h}{\phi} - \sigma^2 \frac{h^2}{2\phi^2} + \mathcal{O}(h^3)
   $$
   The one-sided derivative at the origin is $\gamma^\prime(0^+) = \sigma^2/\phi > 0$. Because the origin is linear ($h^1$) rather than parabolic ($h^2$), the two-sided second derivative does not exist (the ratio $\lim_{h \to 0} \frac{2\gamma(h)}{h^2} = \lim_{h \to 0} \frac{2\sigma^2}{\phi h} = \infty$ diverges).
   Therefore, for $\nu = 0.5$, the random field is mean-square continuous but **nowhere mean-square differentiable** ($m = 0$).
3. **Case 2: $\nu = 1.5$:**
   The covariance function is $C(h) = \sigma^2 \left( 1 + \frac{\sqrt{3} h}{\phi} \right) \exp\left(-\frac{\sqrt{3} h}{\phi}\right)$.
   Let $u = \frac{\sqrt{3} h}{\phi}$. Expanding $(1 + u)\exp(-u)$ using $\exp(-u) = 1 - u + \frac{u^2}{2} - \frac{u^3}{6} + \mathcal{O}(u^4)$:
   $$
   \begin{aligned}
   (1 + u)\left(1 - u + \frac{u^2}{2} - \frac{u^3}{6} + \mathcal{O}(u^4)\right) &= 1 - u + \frac{u^2}{2} - \frac{u^3}{6} + u - u^2 + \frac{u^3}{2} + \mathcal{O}(u^4) \\
   &= 1 - \frac{u^2}{2} + \frac{u^3}{3} + \mathcal{O}(u^4)
   \end{aligned}
   $$
   Substituting $u = \frac{\sqrt{3} h}{\phi}$:
   $$
   C(h) = \sigma^2 \left[ 1 - \frac{3h^2}{2\phi^2} + \frac{\sqrt{3}h^3}{\phi^3} + \mathcal{O}(h^4) \right]
   $$
   The semivariogram is:
   $$
   \gamma(h) = C(0) - C(h) = \sigma^2 \left[ \frac{3h^2}{2\phi^2} - \frac{\sqrt{3}h^3}{\phi^3} + \mathcal{O}(h^4) \right]
   $$
   The second derivative exists: $\gamma^{\prime\prime}(0) = \frac{3\sigma^2}{\phi^2} < \infty$, establishing that the first mean-square directional derivative $\frac{\partial Z(\mathbf{s})}{\partial s_k}$ exists. However, the odd power $h^3$ prevents the existence of a fourth derivative at the origin.
   Therefore, for $\nu = 1.5$, the field is **once mean-square differentiable** ($m = 1$).
4. **Case 3: $\nu = 2.5$:**
   The covariance function is $C(h) = \sigma^2 \left( 1 + \frac{\sqrt{5} h}{\phi} + \frac{5h^2}{3\phi^2} \right) \exp\left(-\frac{\sqrt{5} h}{\phi}\right)$.
   Let $w = \frac{\sqrt{5} h}{\phi}$. Expanding $\left(1 + w + \frac{w^2}{3}\right)\exp(-w)$ using $\exp(-w) = 1 - w + \frac{w^2}{2} - \frac{w^3}{6} + \frac{w^4}{24} - \frac{w^5}{120} + \mathcal{O}(w^6)$:
   Multiplying the polynomials degree by degree:
   - Degree 0: $1 \cdot 1 = 1$
   - Degree 1: $1(-1) + 1(1) = 0$
   - Degree 2: $1(1/2) + 1(-1) + (1/3)(1) = 1/2 - 1 + 1/3 = -1/6$
   - Degree 3: $1(-1/6) + 1(1/2) + (1/3)(-1) = -1/6 + 1/2 - 1/3 = 0$
   - Degree 4: $1(1/24) + 1(-1/6) + (1/3)(1/2) = 1/24 - 1/6 + 1/6 = 1/24$
   - Degree 5: $1(-1/120) + 1(1/24) + (1/3)(-1/6) = -1/120 + 1/24 - 1/18 = -1/45$
   $$
   \left(1 + w + \frac{w^2}{3}\right)\exp(-w) = 1 - \frac{w^2}{6} + \frac{w^4}{24} - \frac{w^5}{45} + \mathcal{O}(w^6)
   $$
   Substituting $w = \frac{\sqrt{5} h}{\phi}$:
   $$
   C(h) = \sigma^2 \left[ 1 - \frac{5h^2}{6\phi^2} + \frac{25h^4}{24\phi^4} - \frac{25\sqrt{5}h^5}{45\phi^5} + \mathcal{O}(w^6) \right]
   $$
   Both the second derivative ($C^{\prime\prime}(0) = -\frac{5\sigma^2}{3\phi^2}$) and fourth derivative ($C^{(4)}(0) = \frac{25\sigma^2}{\phi^4}$) exist and are finite at $h=0$. However, the odd power $h^5$ appears at order 5, preventing the existence of a sixth derivative.
   Therefore, for $\nu = 2.5$, the field is **twice mean-square differentiable** ($m = 2$).
5. **General Rule:** In general, a Matérn random field is $m$-times mean-square differentiable if and only if $\nu > m$. $\blacksquare$

---

#### Exercise 2.3: Failure of Conditional Negative Definiteness for Power Variograms with Exponent $\alpha > 2$
**Problem Statement:**
Prove that the candidate semivariogram $\gamma(h) = b h^\alpha$ ($b > 0$) is **not** conditionally negative definite for any $\alpha > 2$.

**Complete Step-by-Step Solution:**
1. A candidate function $\gamma(h)$ is conditionally negative definite if and only if for all configurations $\{\mathbf{s}_i\}_{i=1}^n$ and all contrast weight vectors satisfying $\sum_{i=1}^n \lambda_i = 0$, the quadratic form:
   $$
   Q = \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\|\mathbf{s}_i - \mathbf{s}_j\|) \le 0
   $$
2. It suffices to find a single 1D counterexample configuration on $\mathbb{R}^1$ where $Q > 0$.
3. Choose $n=3$ collinear points on $\mathbb{R}^1$:
   $$
   s_1 = 0, \quad s_2 = 1, \quad s_3 = 2
   $$
4. Choose the second-difference contrast weight vector:
   $$
   \boldsymbol{\lambda} = (1, -2, 1)^\prime
   $$
   Verify the zero-sum contrast condition: $\sum_{i=1}^3 \lambda_i = 1 - 2 + 1 = 0$.
5. Expand the quadratic form $Q(\alpha) = \sum_{i=1}^3 \sum_{j=1}^3 \lambda_i \lambda_j |s_i - s_j|^\alpha$ (setting $b = 1$ without loss of generality):
   - Diagonal terms ($i = j$): $|s_i - s_j| = 0$, contributing 0.
   - Pairs at distance $|s_i - s_j| = 1$: The pairs are $(1,2), (2,1), (2,3), (3,2)$:
     $$
     2 [\lambda_1 \lambda_2 + \lambda_2 \lambda_3] \cdot 1^\alpha = 2 [(1)(-2) + (-2)(1)] \cdot 1 = 2[-2 - 2] = -8
     $$
   - Pairs at distance $|s_i - s_j| = 2$: The pairs are $(1,3)$ and $(3,1)$:
     $$
     2 [\lambda_1 \lambda_3] \cdot 2^\alpha = 2 [(1)(1)] \cdot 2^\alpha = 2 \cdot 2^\alpha
     $$
6. Summing all components:
   $$
   Q(\alpha) = 2 \cdot 2^\alpha - 8 = 2(2^\alpha - 4)
   $$
7. Evaluate $Q(\alpha)$ across different values of $\alpha$:
   - At $\alpha = 1$ (linear): $Q(1) = 2(2^1 - 4) = -4 \le 0$ (valid CND).
   - At $\alpha = 2$ (parabolic boundary): $Q(2) = 2(2^2 - 4) = 0 \le 0$. This represents the limiting boundary of intrinsic stationarity, corresponding to a deterministic linear trend with random slope $Z(s) = A + B s$.
   - For any $\alpha > 2$: Because $2^\alpha$ is strictly increasing, $2^\alpha > 2^2 = 4$ for all $\alpha > 2$, which implies:
     $$
     Q(\alpha) = 2(2^\alpha - 4) > 0 \quad \forall \alpha > 2
     $$
8. Since $Q(\alpha) > 0$, the variance of the linear contrast is strictly negative:
   $$
   \operatorname{Var}\left(Z(0) - 2Z(1) + Z(2)\right) = -b Q(\alpha) = -2b(2^\alpha - 4) < 0
   $$
   This yields an impossible negative variance for every $\alpha > 2$.
9. Because $\mathbb{R}^1$ embeds isometrically into $\mathbb{R}^d$ for any $d \ge 1$, failure of CND on $\mathbb{R}^1$ implies failure in all $\mathbb{R}^d$. Hence, $\gamma(h) = b h^\alpha$ is strictly invalid for all $\alpha > 2$. $\blacksquare$

---

#### Exercise 2.4: Pencil-and-Paper Verification of Conditional Negative Definiteness via Contrast Projection
**Problem Statement:**
Consider three equally spaced locations on the real line: $s_1 = 0, s_2 = 1, s_3 = 2$.
Any zero-sum contrast vector $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \lambda_3)^\prime \in \mathbb{R}^3$ satisfying $\sum_{i=1}^3 \lambda_i = 0$ can be parameterized in terms of its first two components: $\lambda_3 = -\lambda_1 - \lambda_2$. This defines the contrast subspace projection matrix $\mathbf{P} \in \mathbb{R}^{3 \times 2}$:
$$
\boldsymbol{\lambda} = \mathbf{P}\mathbf{v} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ -1 & -1 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix}
$$
A candidate semivariogram matrix $\boldsymbol{\Gamma} \in \mathbb{R}^{3 \times 3}$ is conditionally negative definite on this 3-point configuration if and only if the $2 \times 2$ filtered matrix:
$$
\mathbf{Q} = -\mathbf{P}^\prime \boldsymbol{\Gamma} \mathbf{P}
$$
is positive semi-definite (i.e. has all non-negative eigenvalues).

By explicit pencil-and-paper matrix multiplication, evaluate $\mathbf{Q}$, its determinant, and its eigenvalues for:
1. The candidate linear semivariogram $\gamma_1(h) = h$.
2. The candidate quartic semivariogram $\gamma_2(h) = h^4$.
Confirm that $\gamma_1$ is valid on this configuration while $\gamma_2$ is invalid, and find a specific contrast vector $\boldsymbol{\lambda}$ for which $\boldsymbol{\lambda}^\prime \boldsymbol{\Gamma}_2 \boldsymbol{\lambda} > 0$.

**Complete Step-by-Step Solution:**
1. **Case 1: Linear candidate $\gamma_1(h) = h$:**
   Assemble the $3 \times 3$ distance-based variogram matrix $\boldsymbol{\Gamma}_1$:
   $$
   \boldsymbol{\Gamma}_1 = \begin{pmatrix} 
   \gamma_1(|0-0|) & \gamma_1(|0-1|) & \gamma_1(|0-2|) \\
   \gamma_1(|1-0|) & \gamma_1(|1-1|) & \gamma_1(|1-2|) \\
   \gamma_1(|2-0|) & \gamma_1(|2-1|) & \gamma_1(|2-2|)
   \end{pmatrix} = \begin{pmatrix} 0 & 1 & 2 \\ 1 & 0 & 1 \\ 2 & 1 & 0 \end{pmatrix}
   $$
   Compute $\mathbf{P}^\prime \boldsymbol{\Gamma}_1$:
   $$
   \mathbf{P}^\prime \boldsymbol{\Gamma}_1 = \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \end{pmatrix} \begin{pmatrix} 0 & 1 & 2 \\ 1 & 0 & 1 \\ 2 & 1 & 0 \end{pmatrix} = \begin{pmatrix} -2 & 0 & 2 \\ -1 & -1 & 1 \end{pmatrix}
   $$
   Multiply by $\mathbf{P}$:
   $$
   \mathbf{P}^\prime \boldsymbol{\Gamma}_1 \mathbf{P} = \begin{pmatrix} -2 & 0 & 2 \\ -1 & -1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ -1 & -1 \end{pmatrix} = \begin{pmatrix} -2 - 2 & 0 - 2 \\ -1 - 1 & -1 - 1 \end{pmatrix} = \begin{pmatrix} -4 & -2 \\ -2 & -2 \end{pmatrix}
   $$
   Negate to obtain $\mathbf{Q}_1 = -\mathbf{P}^\prime \boldsymbol{\Gamma}_1 \mathbf{P}$:
   $$
   \mathbf{Q}_1 = \begin{pmatrix} 4 & 2 \\ 2 & 2 \end{pmatrix}
   $$
   Evaluate definiteness using the principal minors:
   - First leading minor: $M_1 = 4 > 0$.
   - Determinant: $\det(\mathbf{Q}_1) = (4)(2) - (2)(2) = 8 - 4 = 4 > 0$.
   - Trace: $\operatorname{tr}(\mathbf{Q}_1) = 4 + 2 = 6$.
   Eigenvalues solve $\lambda^2 - 6\lambda + 4 = 0$:
   $$
   \lambda = \frac{6 \pm \sqrt{36 - 16}}{2} = 3 \pm \sqrt{5} \approx 5.236, \, 0.764
   $$
   Both eigenvalues are strictly positive. Thus $\mathbf{v}^\prime \mathbf{Q}_1 \mathbf{v} > 0$ for all $\mathbf{v} \ne \mathbf{0}$, which guarantees $\boldsymbol{\lambda}^\prime \boldsymbol{\Gamma}_1 \boldsymbol{\lambda} < 0$. The linear variogram is conditionally negative definite on this configuration.

2. **Case 2: Quartic candidate $\gamma_2(h) = h^4$:**
   Assemble $\boldsymbol{\Gamma}_2$:
   $$
   \boldsymbol{\Gamma}_2 = \begin{pmatrix} 0^4 & 1^4 & 2^4 \\ 1^4 & 0^4 & 1^4 \\ 2^4 & 1^4 & 0^4 \end{pmatrix} = \begin{pmatrix} 0 & 1 & 16 \\ 1 & 0 & 1 \\ 16 & 1 & 0 \end{pmatrix}
   $$
   Compute $\mathbf{P}^\prime \boldsymbol{\Gamma}_2$:
   $$
   \mathbf{P}^\prime \boldsymbol{\Gamma}_2 = \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \end{pmatrix} \begin{pmatrix} 0 & 1 & 16 \\ 1 & 0 & 1 \\ 16 & 1 & 0 \end{pmatrix} = \begin{pmatrix} -16 & 0 & 16 \\ -15 & -1 & 1 \end{pmatrix}
   $$
   Multiply by $\mathbf{P}$:
   $$
   \mathbf{P}^\prime \boldsymbol{\Gamma}_2 \mathbf{P} = \begin{pmatrix} -16 & 0 & 16 \\ -15 & -1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ -1 & -1 \end{pmatrix} = \begin{pmatrix} -16 - 16 & 0 - 16 \\ -15 - 1 & -1 - 1 \end{pmatrix} = \begin{pmatrix} -32 & -16 \\ -16 & -2 \end{pmatrix}
   $$
   Negate to obtain $\mathbf{Q}_2 = -\mathbf{P}^\prime \boldsymbol{\Gamma}_2 \mathbf{P}$:
   $$
   \mathbf{Q}_2 = \begin{pmatrix} 32 & 16 \\ 16 & 2 \end{pmatrix}
   $$
   Evaluate the determinant:
   $$
   \det(\mathbf{Q}_2) = (32)(2) - (16)(16) = 64 - 256 = -192 < 0
   $$
   Because $\det(\mathbf{Q}_2) < 0$, the eigenvalues have opposite signs:
   $$
   \lambda^2 - 34\lambda - 192 = 0 \implies \lambda = \frac{34 \pm \sqrt{1156 + 768}}{2} = \frac{34 \pm \sqrt{1924}}{2} \approx 38.93, \, -4.93
   $$
   The presence of a negative eigenvalue ($\lambda_2 \approx -4.93 < 0$) proves that $\mathbf{Q}_2$ is indefinite.

3. **Constructing the Violating Contrast:**
   Choose the second-difference contrast $\mathbf{v} = (1, -2)^\prime$, which corresponds to:
   $$
   \boldsymbol{\lambda} = \mathbf{P} \begin{pmatrix} 1 \\ -2 \end{pmatrix} = \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}
   $$
   Evaluating the quadratic form with $\mathbf{Q}_2$:
   $$
   \mathbf{v}^\prime \mathbf{Q}_2 \mathbf{v} = \begin{pmatrix} 1 & -2 \end{pmatrix} \begin{pmatrix} 32 & 16 \\ 16 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ -2 \end{pmatrix} = \begin{pmatrix} 1 & -2 \end{pmatrix} \begin{pmatrix} 0 \\ 12 \end{pmatrix} = -24 < 0
   $$
   Since $\mathbf{v}^\prime \mathbf{Q}_2 \mathbf{v} = -\boldsymbol{\lambda}^\prime \boldsymbol{\Gamma}_2 \boldsymbol{\lambda} = -24$, we have:
   $$
   \boldsymbol{\lambda}^\prime \boldsymbol{\Gamma}_2 \boldsymbol{\lambda} = +24 > 0 \implies \operatorname{Var}(Z(0) - 2Z(1) + Z(2)) = -24 < 0
   $$
   The candidate quartic variogram $\gamma_2(h) = h^4$ produces an impossible negative variance and is strictly inadmissible. $\blacksquare$

---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 2: Spatial Covariance and Variogram Theory — Companion Exercises and Complete Solutions](ch02_exercises.md)
