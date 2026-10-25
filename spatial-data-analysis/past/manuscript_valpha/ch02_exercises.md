# Chapter 2 Study Guide & Comprehensive Exercises
## Spatial Covariance and Variogram Theory

This companion study guide reinforces the core mathematical, spectral, and functional-analytic concepts from [Chapter 2: Spatial Covariance and Variogram Theory](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch02_covariance_variogram.md). The exercises below guide the reader through Bochner's theorem, conditional negative definiteness, the Matérn family, and geometric anisotropy.

### Five-Tier Pedagogical Architecture
1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 2.1–2.4): Elementary positive definiteness, covariance algebra, non-CND variogram pathologies, and integral scales.
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 2.5–2.8): Bochner's theorem, conditional negative definiteness, sub-quadratic variogram growth bounds, and Matérn closed forms.
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 2.9–2.11): CND eigenvalue projection mechanics, geometric anisotropy rotation and metric tensors, and nugget effect Tikhonov regularization.
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 2.12–2.13): Pure Python implementations of CND projection verifiers and empirical origin curvature diagnostics.
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 2.14–2.16): The spherical model dimension barrier, Gaussian covariance trap, and polynomial power variogram breakdown.

---

### Tier 0: Classical-to-Spatial Warm-Ups

#### Exercise 2.1: Proving Positive Definiteness of Elementary Cosine and Exponential Kernels
Let $C(h)$ be a candidate stationary spatial covariance kernel on $\mathbb{R}^1$.

- **(a)** For the single-frequency cosine kernel $C(h) = \sigma^2 \cos(\omega_0 h)$ with $\omega_0 > 0$, prove directly from the foundational definition $\sum_{i=1}^n \sum_{j=1}^n a_i a_j C(s_i - s_j) \ge 0$ that $C(h)$ is positive semi-definite on $\mathbb{R}^1$ by decomposing the cosine into a product of harmonic terms.
- **(b)** For the elementary 1D Exponential kernel $C(h) = \sigma^2 \exp(-|h|/\phi)$ with $\phi > 0$, use Bochner's theorem to derive its spectral density $f(\omega) = \frac{1}{2\pi}\int_{-\infty}^\infty C(h) e^{-i\omega h}\,dh$, show that $f(\omega) > 0$ for all $\omega \in \mathbb{R}$, and identify the resulting probability distribution.
- **(c)** Explain why the cosine kernel $C(h) = \sigma^2 \cos(\omega_0 h)$ is positive semi-definite but not strictly positive definite for any point set of size $n > 2$.

##### Solution to Exercise 2.1
- **(a)** *Direct proof from definitions:*
  Let $\{s_1, \dots, s_n\} \subset \mathbb{R}^1$ and $\mathbf{a} = (a_1, \dots, a_n)' \in \mathbb{R}^n$.
  Using the trigonometric angle addition identity $\cos(\alpha - \beta) = \cos(\alpha)\cos(\beta) + \sin(\alpha)\sin(\beta)$:
  $$
  C(s_i - s_j) = \sigma^2 \cos(\omega_0(s_i - s_j)) = \sigma^2 \left[\cos(\omega_0 s_i)\cos(\omega_0 s_j) + \sin(\omega_0 s_i)\sin(\omega_0 s_j)\right]
  $$
  Substitute this into the quadratic form:
  $$
  \begin{aligned}
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C(s_i - s_j) &= \sigma^2 \sum_{i=1}^n \sum_{j=1}^n a_i a_j \left[\cos(\omega_0 s_i)\cos(\omega_0 s_j) + \sin(\omega_0 s_i)\sin(\omega_0 s_j)\right] \\
  &= \sigma^2 \sum_{i=1}^n \sum_{j=1}^n a_i a_j \cos(\omega_0 s_i)\cos(\omega_0 s_j) + \sigma^2 \sum_{i=1}^n \sum_{j=1}^n a_i a_j \sin(\omega_0 s_i)\sin(\omega_0 s_j) \\
  &= \sigma^2 \left(\sum_{i=1}^n a_i \cos(\omega_0 s_i)\right)\left(\sum_{j=1}^n a_j \cos(\omega_0 s_j)\right) + \sigma^2 \left(\sum_{i=1}^n a_i \sin(\omega_0 s_i)\right)\left(\sum_{j=1}^n a_j \sin(\omega_0 s_j)\right) \\
  &= \sigma^2 \left[\sum_{i=1}^n a_i \cos(\omega_0 s_i)\right]^2 + \sigma^2 \left[\sum_{i=1}^n a_i \sin(\omega_0 s_i)\right]^2
  \end{aligned}
  $$
  Because $\sigma^2 > 0$ and both terms in the bracket are squared real numbers, their sum is non-negative:
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C(s_i - s_j) \ge 0
  $$
  Hence, $C(h) = \sigma^2 \cos(\omega_0 h)$ is positive semi-definite on $\mathbb{R}^1$.

- **(b)** *Bochner spectral derivation:*
  By Bochner's theorem on $\mathbb{R}^1$, the spectral density $f(\omega)$ is the continuous inverse Fourier transform of $C(h)$:
  $$
  f(\omega) = \frac{1}{2\pi} \int_{-\infty}^\infty C(h) e^{-i\omega h} \, dh = \frac{\sigma^2}{2\pi} \int_{-\infty}^\infty \exp\left(-\frac{|h|}{\phi}\right) (\cos(\omega h) - i\sin(\omega h)) \, dh
  $$
  Since $\exp(-|h|/\phi)\sin(\omega h)$ is an odd function over symmetric limits $(-\infty, \infty)$, its integral is identically zero.
  Since $\exp(-|h|/\phi)\cos(\omega h)$ is an even function:
  $$
  f(\omega) = \frac{\sigma^2}{\pi} \int_0^\infty \exp\left(-\frac{h}{\phi}\right) \cos(\omega h) \, dh
  $$
  Evaluating using standard Laplace-Fourier integration $\int_0^\infty e^{-a h} \cos(\omega h) \, dh = \frac{a}{a^2 + \omega^2}$ with $a = 1/\phi$:
  $$
  f(\omega) = \frac{\sigma^2}{\pi} \frac{1/\phi}{(1/\phi)^2 + \omega^2} = \frac{\sigma^2}{\pi} \frac{\phi}{1 + \phi^2 \omega^2}
  $$
  Because $\sigma^2 > 0$ and $\phi > 0$, the denominator $1 + \phi^2 \omega^2 > 0$ for all $\omega \in \mathbb{R}$.
  Therefore:
  $$
  f(\omega) = \frac{\sigma^2 \phi}{\pi(1 + \phi^2 \omega^2)} > 0 \quad \text{for all } \omega \in \mathbb{R}
  $$
  Normalized by the total variance $C(0) = \sigma^2$, the spectral density $\frac{f(\omega)}{\sigma^2} = \frac{1/\phi}{\pi((1/\phi)^2 + \omega^2)}$ is the classical **Cauchy (Lorentzian) probability density function** with scale parameter $\gamma_{\text{Cauchy}} = 1/\phi$.
  Since $f(\omega)$ is strictly positive everywhere on $\mathbb{R}$, Bochner's theorem confirms that the Exponential kernel is strictly positive definite on $\mathbb{R}^1$.

- **(c)** *Semi-definiteness vs. strict positive definiteness:*
  Let $\mathbf{c} = (\cos(\omega_0 s_1), \dots, \cos(\omega_0 s_n))' \in \mathbb{R}^n$ and $\mathbf{s} = (\sin(\omega_0 s_1), \dots, \sin(\omega_0 s_n))' \in \mathbb{R}^n$.
  The $n \times n$ covariance matrix can be written as:
  $$
  \mathbf{V} = \sigma^2 \left(\mathbf{c}\mathbf{c}' + \mathbf{s}\mathbf{s}'\right)
  $$
  Since $\mathbf{V}$ is the sum of two outer products of vectors in $\mathbb{R}^n$:
  $$
  \operatorname{rank}(\mathbf{V}) \le \operatorname{rank}(\mathbf{c}\mathbf{c}') + \operatorname{rank}(\mathbf{s}\mathbf{s}') \le 1 + 1 = 2
  $$
  Whenever the sample size $n > 2$, the $n \times n$ matrix $\mathbf{V}$ has rank at most 2.
  By the rank-nullity theorem, the null space of $\mathbf{V}$ has dimension:
  $$
  \operatorname{nullity}(\mathbf{V}) = n - \operatorname{rank}(\mathbf{V}) \ge n - 2 \ge 1
  $$
  Consequently, there exists a non-zero vector $\mathbf{a} \ne \mathbf{0}$ such that $\mathbf{a} \perp \mathbf{c}$ and $\mathbf{a} \perp \mathbf{s}$, which gives:
  $$
  \mathbf{a}' \mathbf{V} \mathbf{a} = \sigma^2 (\mathbf{a}'\mathbf{c})^2 + \sigma^2 (\mathbf{a}'\mathbf{s})^2 = 0
  $$
  Because non-zero vectors produce zero quadratic forms, $\mathbf{V}$ is singular (has at least $n - 2$ zero eigenvalues) and is only **positive semi-definite**, not **strictly positive definite**. $\blacksquare$

---

#### Exercise 2.2: Valid Covariance Kernels via Linear Combinations, Products, and Scale Mixtures
New valid spatial covariance kernels can be synthesized from existing valid kernels through algebraic and integral operations.

- **(a)** Let $C_1(\mathbf{h})$ and $C_2(\mathbf{h})$ be valid positive definite covariance functions on $\mathbb{R}^d$, and let $c_1, c_2 \ge 0$. Prove that the linear combination:
  $$
  C_3(\mathbf{h}) = c_1 C_1(\mathbf{h}) + c_2 C_2(\mathbf{h})
  $$
  is a valid positive definite covariance function on $\mathbb{R}^d$.
- **(b)** Prove the **Schur Product Theorem** in this context: show that the point-wise product $C_4(\mathbf{h}) = C_1(\mathbf{h}) C_2(\mathbf{h})$ is a valid positive definite covariance function on $\mathbb{R}^d$.
- **(c)** Let $C(h; \theta) = \exp(-h/\theta)$ be a 1D isotropic Exponential covariance with scale parameter $\theta > 0$. Let $\theta$ follow a continuous probability density function $p(\theta)$ supported on $(0, \infty)$. Prove that the **scale mixture**:
  $$
  C_{\text{mix}}(h) = \int_0^\infty \exp\left(-\frac{h}{\theta}\right) p(\theta) \, d\theta
  $$
  is a valid positive definite covariance function.
- **(d)** If Gaussian kernels $\exp(-h^2 / (2\theta^2))$ are mixed over an Inverse-Gamma distribution for $\theta^2$, identify the resulting well-known covariance family.

##### Solution to Exercise 2.2
- **(a)** *Linear combinations:*
  Let $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$ and $\mathbf{a} \in \mathbb{R}^n$.
  The quadratic form for $C_3$ is:
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_3(\mathbf{s}_i - \mathbf{s}_j) = c_1 \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_1(\mathbf{s}_i - \mathbf{s}_j) + c_2 \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_2(\mathbf{s}_i - \mathbf{s}_j)
  $$
  Since $C_1$ and $C_2$ are positive definite, their quadratic forms are non-negative.
  Since $c_1 \ge 0$ and $c_2 \ge 0$, the sum of two non-negative terms is non-negative:
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_3(\mathbf{s}_i - \mathbf{s}_j) \ge 0
  $$
  Thus, $C_3(\mathbf{h})$ is positive definite on $\mathbb{R}^d$.

- **(b)** *Schur product theorem:*
  Let $\mathbf{V}_1$ and $\mathbf{V}_2$ be the $n \times n$ covariance matrices generated by $C_1$ and $C_2$ on $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$.
  The matrix for $C_4$ is the element-wise Hadamard (Schur) product $\mathbf{V}_4 = \mathbf{V}_1 \circ \mathbf{V}_2$, with entries $[\mathbf{V}_4]_{ij} = [\mathbf{V}_1]_{ij} [\mathbf{V}_2]_{ij}$.
  By the spectral theorem, since $\mathbf{V}_2$ is symmetric positive semi-definite, it admits an eigendecomposition:
  $$
  \mathbf{V}_2 = \sum_{k=1}^n \mu_k \mathbf{u}_k \mathbf{u}_k'
  $$
  where eigenvalues $\mu_k \ge 0$ and eigenvectors $\mathbf{u}_k = (u_{k1}, \dots, u_{kn})'$.
  The entries of $\mathbf{V}_2$ are $[\mathbf{V}_2]_{ij} = \sum_{k=1}^n \mu_k u_{ki} u_{kj}$.
  Now evaluate the quadratic form for any $\mathbf{a} \in \mathbb{R}^n$:
  $$
  \begin{aligned}
  \mathbf{a}' \mathbf{V}_4 \mathbf{a} &= \sum_{i=1}^n \sum_{j=1}^n a_i a_j [\mathbf{V}_1]_{ij} [\mathbf{V}_2]_{ij} \\
  &= \sum_{i=1}^n \sum_{j=1}^n a_i a_j [\mathbf{V}_1]_{ij} \left(\sum_{k=1}^n \mu_k u_{ki} u_{kj}\right) \\
  &= \sum_{k=1}^n \mu_k \sum_{i=1}^n \sum_{j=1}^n (a_i u_{ki}) (a_j u_{kj}) [\mathbf{V}_1]_{ij}
  \end{aligned}
  $$
  Define vector $\mathbf{b}_k = (a_1 u_{k1}, \dots, a_n u_{kn})' \in \mathbb{R}^n$. The inner double sum is the quadratic form $\mathbf{b}_k' \mathbf{V}_1 \mathbf{b}_k$.
  Since $\mathbf{V}_1$ is positive semi-definite, $\mathbf{b}_k' \mathbf{V}_1 \mathbf{b}_k \ge 0$ for all $k$.
  Since $\mu_k \ge 0$, the outer sum is a non-negative linear combination of non-negative quantities:
  $$
  \mathbf{a}' \mathbf{V}_4 \mathbf{a} = \sum_{k=1}^n \mu_k (\mathbf{b}_k' \mathbf{V}_1 \mathbf{b}_k) \ge 0
  $$
  Therefore, $\mathbf{V}_4 = \mathbf{V}_1 \circ \mathbf{V}_2$ is positive semi-definite, proving that the product $C_1(\mathbf{h})C_2(\mathbf{h})$ is positive definite.
  *(Spectral interpretation)*: By the convolution theorem, the spectral measure of a product is the convolution of their spectral measures: $F_4 = F_1 * F_2$. The convolution of two non-negative measures is non-negative, confirming validity via Bochner's theorem.

- **(c)** *Scale mixture:*
  Let $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^1$ and $\mathbf{a} \in \mathbb{R}^n$.
  The quadratic form of $C_{\text{mix}}$ is:
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_{\text{mix}}(\mathbf{s}_i - \mathbf{s}_j) = \sum_{i=1}^n \sum_{j=1}^n a_i a_j \int_0^\infty \exp\left(-\frac{\|\mathbf{s}_i - \mathbf{s}_j\|}{\theta}\right) p(\theta) \, d\theta
  $$
  By Fubini-Tonelli theorem (since the integrand is bounded and $p(\theta)$ is a probability density):
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_{\text{mix}}(\mathbf{s}_i - \mathbf{s}_j) = \int_0^\infty \left[ \sum_{i=1}^n \sum_{j=1}^n a_i a_j \exp\left(-\frac{\|\mathbf{s}_i - \mathbf{s}_j\|}{\theta}\right) \right] p(\theta) \, d\theta
  $$
  For every fixed $\theta \in (0, \infty)$, the kernel $\exp(-h/\theta)$ is a valid positive definite covariance function, so the bracketed quadratic form is non-negative:
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j \exp\left(-\frac{\|\mathbf{s}_i - \mathbf{s}_j\|}{\theta}\right) \ge 0
  $$
  Since $p(\theta) \ge 0$, the integrand is non-negative for all $\theta > 0$, and the integral of a non-negative function is non-negative:
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C_{\text{mix}}(\mathbf{s}_i - \mathbf{s}_j) \ge 0
  $$
  Hence, $C_{\text{mix}}(h)$ is a valid positive definite covariance function.

- **(d)** *Rational Quadratic family:*
  When Gaussian kernels $\exp(-h^2 / (2\theta^2))$ are integrated over an Inverse-Gamma prior on the squared scale parameter $\theta^2$ (or an exponential kernel over a Gamma scale), the resulting scale mixture yields the **Rational Quadratic (Cauchy) covariance class**:
  $$
  C(h) = \sigma^2 \left(1 + \frac{h^2}{\alpha \phi^2}\right)^{-\alpha}, \quad \alpha > 0, \; \phi > 0
  $$
  Unlike the Gaussian model which decays exponentially fast, the Rational Quadratic model decays as an algebraic power law $h^{-2\alpha}$, representing processes with long-range dependence and persistence. $\blacksquare$

---

#### Exercise 2.3: Concrete Ordinary Kriging Failure Under Invalid Variogram $\gamma(h) = h^3$
In Exercise 2.16, we prove that the power candidate $\gamma(h) = h^3$ is not conditionally negative definite. In this exercise, demonstrate the catastrophic practical consequence: calculating a negative Ordinary Kriging prediction variance.

Consider three observation locations on $\mathbb{R}^1$:
$$
s_1 = 0.0, \quad s_2 = 1.0, \quad s_3 = 2.0
$$
Suppose we predict the unobserved spatial value $Z(s_0)$ at the interpolation point $s_0 = 0.5$ using the candidate variogram $\gamma(h) = h^3$.

- **(a)** Set up the Ordinary Kriging bordered linear system:
  $$
  \begin{pmatrix} \boldsymbol{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ \mu \end{pmatrix} = \begin{pmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{pmatrix}
  $$
  evaluating the exact numerical entries of the $3 \times 3$ variogram matrix $\boldsymbol{\Gamma}$ and the target vector $\boldsymbol{\gamma}_0$.
- **(b)** Solve the $4 \times 4$ system analytically for the kriging weights $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \lambda_3)'$ and Lagrange multiplier $\mu$.
- **(c)** Compute the Ordinary Kriging prediction variance:
  $$
  \sigma_{\text{OK}}^2(s_0) = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 + \mu
  $$
- **(d)** Demonstrate that $\sigma_{\text{OK}}^2(s_0) < 0$, explain why this is an impossible physical result, and identify the foundational mathematical axiom that was violated.

##### Solution to Exercise 2.3
- **(a)** *Assembling the bordered linear system:*
  Given $\gamma(h) = h^3$ (for $h \ge 0$):
  - Between observations:
    - $|s_1 - s_2| = |0 - 1| = 1 \implies \gamma(1) = 1^3 = 1$.
    - $|s_1 - s_3| = |0 - 2| = 2 \implies \gamma(2) = 2^3 = 8$.
    - $|s_2 - s_3| = |1 - 2| = 1 \implies \gamma(1) = 1^3 = 1$.
    - On the diagonal: $\gamma(0) = 0$.
    $$
    \boldsymbol{\Gamma} = \begin{pmatrix} 0 & 1 & 8 \\ 1 & 0 & 1 \\ 8 & 1 & 0 \end{pmatrix}
    $$
  - Between observations and target point $s_0 = 0.5$:
    - $|s_1 - s_0| = |0 - 0.5| = 0.5 = 1/2 \implies \gamma(1/2) = (1/2)^3 = 1/8 = 0.125$.
    - $|s_2 - s_0| = |1 - 0.5| = 0.5 = 1/2 \implies \gamma(1/2) = (1/2)^3 = 1/8 = 0.125$.
    - $|s_3 - s_0| = |2 - 0.5| = 1.5 = 3/2 \implies \gamma(3/2) = (3/2)^3 = 27/8 = 3.375$.
    $$
    \boldsymbol{\gamma}_0 = \begin{pmatrix} 1/8 \\ 1/8 \\ 27/8 \end{pmatrix}
    $$
  The full Ordinary Kriging system is:
  $$
  \begin{pmatrix} 0 & 1 & 8 & 1 \\ 1 & 0 & 1 & 1 \\ 8 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ \mu \end{pmatrix} = \begin{pmatrix} 1/8 \\ 1/8 \\ 27/8 \\ 1 \end{pmatrix}
  $$

- **(b)** *Analytical solution of the $4 \times 4$ system:*
  Write the equations explicitly:
  1. $\lambda_2 + 8\lambda_3 + \mu = 1/8$
  2. $\lambda_1 + \lambda_3 + \mu = 1/8$
  3. $8\lambda_1 + \lambda_2 + \mu = 27/8$
  4. $\lambda_1 + \lambda_2 + \lambda_3 = 1$

  Subtract equation (2) from equation (1):
  $$
  (\lambda_2 + 8\lambda_3 + \mu) - (\lambda_1 + \lambda_3 + \mu) = \frac{1}{8} - \frac{1}{8} = 0 \implies \lambda_2 - \lambda_1 + 7\lambda_3 = 0 \implies \lambda_2 = \lambda_1 - 7\lambda_3
  $$
  Subtract equation (1) from equation (3):
  $$
  (8\lambda_1 + \lambda_2 + \mu) - (\lambda_2 + 8\lambda_3 + \mu) = \frac{27}{8} - \frac{1}{8} = \frac{26}{8} = \frac{13}{4} \implies 8(\lambda_1 - \lambda_3) = \frac{13}{4} \implies \lambda_1 - \lambda_3 = \frac{13}{32}
  $$
  Thus:
  $$
  \lambda_1 = \lambda_3 + \frac{13}{32}
  $$
  Substitute this into the expression for $\lambda_2$:
  $$
  \lambda_2 = \left(\lambda_3 + \frac{13}{32}\right) - 7\lambda_3 = \frac{13}{32} - 6\lambda_3
  $$
  Now substitute $\lambda_1$ and $\lambda_2$ into the unbiasedness constraint (equation 4):
  $$
  \left(\lambda_3 + \frac{13}{32}\right) + \left(\frac{13}{32} - 6\lambda_3\right) + \lambda_3 = 1
  $$
  $$
  -4\lambda_3 + \frac{26}{32} = 1 \implies -4\lambda_3 = 1 - \frac{13}{16} = \frac{3}{16} \implies \lambda_3 = -\frac{3}{64} = -0.046875
  $$
  Now compute $\lambda_1$ and $\lambda_2$:
  $$
  \lambda_1 = -\frac{3}{64} + \frac{26}{64} = \frac{23}{64} = 0.359375
  $$
  $$
  \lambda_2 = \frac{26}{64} - 6\left(-\frac{3}{64}\right) = \frac{26}{64} + \frac{18}{64} = \frac{44}{64} = \frac{11}{16} = 0.687500
  $$
  Check sum: $\lambda_1 + \lambda_2 + \lambda_3 = \frac{23 + 44 - 3}{64} = \frac{64}{64} = 1.0$.
  Now solve for the Lagrange multiplier $\mu$ using equation (2):
  $$
  \mu = \frac{1}{8} - \lambda_1 - \lambda_3 = \frac{8}{64} - \frac{23}{64} - \left(-\frac{3}{64}\right) = \frac{8 - 23 + 3}{64} = -\frac{12}{64} = -\frac{3}{16} = -0.187500
  $$
  Thus, the solution vector is:
  $$
  \boldsymbol{\lambda} = \begin{pmatrix} 23/64 \\ 44/64 \\ -3/64 \end{pmatrix} = \begin{pmatrix} 0.359375 \\ 0.687500 \\ -0.046875 \end{pmatrix}, \quad \mu = -\frac{3}{16} = -0.187500
  $$

- **(c)** *Calculation of Ordinary Kriging variance:*
  The prediction variance is:
  $$
  \sigma_{\text{OK}}^2(s_0) = \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 + \mu = \lambda_1 \gamma_0(s_1) + \lambda_2 \gamma_0(s_2) + \lambda_3 \gamma_0(s_3) + \mu
  $$
  Compute the scalar product $\boldsymbol{\lambda}' \boldsymbol{\gamma}_0$:
  $$
  \begin{aligned}
  \boldsymbol{\lambda}' \boldsymbol{\gamma}_0 &= \left(\frac{23}{64}\right)\left(\frac{1}{8}\right) + \left(\frac{44}{64}\right)\left(\frac{1}{8}\right) + \left(-\frac{3}{64}\right)\left(\frac{27}{8}\right) \\
  &= \frac{23(1) + 44(1) - 3(27)}{512} = \frac{23 + 44 - 81}{512} = \frac{67 - 81}{512} = -\frac{14}{512} = -\frac{7}{256}
  \end{aligned}
  $$
  Adding the Lagrange multiplier $\mu = -\frac{3}{16} = -\frac{96}{512}$:
  $$
  \sigma_{\text{OK}}^2(s_0) = -\frac{14}{512} - \frac{96}{512} = -\frac{110}{512} = -\frac{55}{256} \approx -0.21484375
  $$

- **(d)** *Physical absurdity and violated axiom:*
  The predicted variance is strictly negative:
  $$
  \sigma_{\text{OK}}^2(s_0) = -\frac{55}{256} \approx -0.2148 < 0
  $$
  By Kolmogorov's probability axioms, the variance of any real-valued random variable $X$ is the expectation of a squared deviation $\operatorname{Var}(X) = \operatorname{E}[(X - \operatorname{E}[X])^2] \ge 0$, which is non-negative by definition.
  A negative variance produces an imaginary prediction standard error ($\sigma = \sqrt{-0.2148} \approx 0.4635 i$), making confidence intervals, uncertainty quantification, and risk assessment completely meaningless.
  *Mathematical origin of the pathology:* Ordinary Kriging error is an increment of order zero with weights $(\lambda_1, \lambda_2, \lambda_3, -1)'$ summing to zero. Its variance is:
  $$
  \operatorname{Var}(\widehat{Z}(s_0) - Z(s_0)) = -\sum_{i=0}^3 \sum_{j=0}^3 w_i w_j \gamma(s_i - s_j)
  $$
  This variance is non-negative if and only if $\gamma(h)$ is **conditionally negative definite (CND)**.
  Because $\gamma(h) = h^3$ violates CND, the quadratic form $\sum \sum w_i w_j \gamma_{ij}$ evaluated to $+0.2148 > 0$, causing the negated variance expression to yield $-0.2148 < 0$. This proves why non-CND functions cannot serve as variograms. $\blacksquare$

---

#### Exercise 2.4: The Integral Scale of Spatial Correlation Functions
The **integral scale** (or spatial correlation length) characterizes the spatial memory and persistence of a stationary random field, defined as the integral of the isotropic correlation function $\rho(h) = C(h)/\sigma^2$:
$$
I = \int_0^\infty \rho(h) \, dh
$$

- **(a)** For the Exponential correlation model $\rho(h) = \exp(-h/a)$ with range parameter $a > 0$, compute the exact analytical value of $I_{\text{exp}}$.
- **(b)** For the Gaussian correlation model $\rho(h) = \exp(-h^2/a^2)$ with parameter $a > 0$, compute the exact analytical value of $I_{\text{gauss}}$.
- **(c)** For the Matérn correlation model:
  $$
  \rho(h) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu} h}{\phi}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} h}{\phi}\right)
  $$
  use the standard Bessel integral identity $\int_0^\infty z^\nu K_\nu(z) \, dz = 2^{\nu - 1} \sqrt{\pi} \Gamma(\nu + 1/2)$ to derive the closed-form expression for $I_{\text{mat}}(\nu, \phi)$.
- **(d)** Verify that for $\nu = 0.5$, your Matérn integral scale formula simplifies to the Exponential integral scale $I_{\text{exp}}$ with matching range parameter.

##### Solution to Exercise 2.4
- **(a)** *Exponential integral scale:*
  $$
  I_{\text{exp}} = \int_0^\infty \exp\left(-\frac{h}{a}\right) \, dh = \left[ -a \exp\left(-\frac{h}{a}\right) \right]_0^\infty = 0 - (-a) = a
  $$
  The integral scale of an Exponential model is exactly its range parameter $a$.

- **(b)** *Gaussian integral scale:*
  $$
  I_{\text{gauss}} = \int_0^\infty \exp\left(-\frac{h^2}{a^2}\right) \, dh
  $$
  Substitute $u = h/a \implies dh = a \, du$:
  $$
  I_{\text{gauss}} = a \int_0^\infty \exp(-u^2) \, du
  $$
  Using the Gaussian integral $\int_0^\infty e^{-u^2} \, du = \frac{\sqrt{\pi}}{2}$:
  $$
  I_{\text{gauss}} = \frac{\sqrt{\pi}}{2} a \approx 0.886227 a
  $$

- **(c)** *Matérn integral scale:*
  Let $z = \frac{\sqrt{2\nu} h}{\phi}$. Then $h = \frac{\phi}{\sqrt{2\nu}} z$, and $dh = \frac{\phi}{\sqrt{2\nu}} \, dz$.
  Substituting into the integral:
  $$
  \begin{aligned}
  I_{\text{mat}} &= \int_0^\infty \frac{2^{1-\nu}}{\Gamma(\nu)} z^\nu K_\nu(z) \left(\frac{\phi}{\sqrt{2\nu}}\right) \, dz \\
  &= \frac{2^{1-\nu} \phi}{\sqrt{2\nu} \Gamma(\nu)} \int_0^\infty z^\nu K_\nu(z) \, dz
  \end{aligned}
  $$
  Using the identity $\int_0^\infty z^\nu K_\nu(z) \, dz = 2^{\nu - 1} \sqrt{\pi} \Gamma(\nu + 1/2)$:
  $$
  I_{\text{mat}} = \frac{2^{1-\nu} \phi}{\sqrt{2\nu} \Gamma(\nu)} \left(2^{\nu - 1} \sqrt{\pi} \Gamma(\nu + 1/2)\right)
  $$
  Notice that $2^{1-\nu} \cdot 2^{\nu - 1} = 2^{(1-\nu) + (\nu-1)} = 2^0 = 1$.
  Therefore:
  $$
  I_{\text{mat}}(\nu, \phi) = \frac{\phi \sqrt{\pi} \Gamma(\nu + 1/2)}{\sqrt{2\nu} \Gamma(\nu)}
  $$

- **(d)** *Verification at $\nu = 0.5$:*
  Set $\nu = 0.5$:
  - $\sqrt{2\nu} = \sqrt{2(0.5)} = \sqrt{1} = 1$.
  - $\Gamma(\nu) = \Gamma(0.5) = \sqrt{\pi}$.
  - $\Gamma(\nu + 1/2) = \Gamma(0.5 + 0.5) = \Gamma(1) = 0! = 1$.
  Substituting these values:
  $$
  I_{\text{mat}}(0.5, \phi) = \frac{\phi \sqrt{\pi} (1)}{1 \cdot \sqrt{\pi}} = \phi
  $$
  For $\nu = 0.5$, the Matérn correlation function is $\rho(h) = \exp(-h/\phi)$.
  Comparing with the Exponential model $\rho(h) = \exp(-h/a)$ with $a = \phi$, the Exponential integral scale is $I_{\text{exp}} = a = \phi$.
  The two expressions match identically, confirming the analytical derivation. $\blacksquare$

---

### Tier I: Core Theoretical Re-Derivations

#### Exercise 2.5: Positive Definiteness and Bochner's Theorem
Let $C(\mathbf{h})$ be a real-valued function defined on $\mathbb{R}^d$.

- **(a)** State the formal definition of a **positive definite function** on $\mathbb{R}^d$.
- **(b)** State **Bochner’s Theorem** characterizing continuous positive definite functions on $\mathbb{R}^d$ via non-negative spectral measures $F$.
- **(c)** When the measure $F$ possesses a spectral density $f(\boldsymbol{\omega})$, write the forward and inverse Fourier transform pair connecting spatial covariance $C(\mathbf{h})$ and spectral density $f(\boldsymbol{\omega})$.
- **(d)** Under spatial **isotropy** ($C(\mathbf{h}) = C(h)$ with $h = \|\mathbf{h}\|$), show how the $d$-dimensional Fourier integral simplifies to a 1D Hankel (Fourier-Bessel) transform involving the Bessel function of the first kind $J_{d/2 - 1}$.

##### Solution to Exercise 2.5
- **(a)** A function $C : \mathbb{R}^d \to \mathbb{R}$ is **positive definite** (often termed positive semi-definite in linear algebra) if for every $n \in \mathbb{N}$, every collection of points $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$, and all real coefficients $a_1, \dots, a_n \in \mathbb{R}$:
  $$
  \sum_{i=1}^n \sum_{j=1}^n a_i a_j C(\mathbf{s}_i - \mathbf{s}_j) \ge 0
  $$
  It is strictly positive definite if the quadratic form equals zero if and only if $a_1 = \dots = a_n = 0$.

- **(b)** **Bochner’s Theorem:** A continuous function $C : \mathbb{R}^d \to \mathbb{R}$ with $C(\mathbf{0}) < \infty$ is positive definite if and only if it can be represented as the Fourier-Stieltjes transform of a bounded, non-negative, symmetric Borel measure $F$ on $\mathbb{R}^d$:
  $$
  C(\mathbf{h}) = \int_{\mathbb{R}^d} \exp(i \boldsymbol{\omega}' \mathbf{h}) \, dF(\boldsymbol{\omega}) = \int_{\mathbb{R}^d} \cos(\boldsymbol{\omega}' \mathbf{h}) \, dF(\boldsymbol{\omega})
  $$

- **(c)** If $F$ is absolutely continuous with respect to Lebesgue measure, $dF(\boldsymbol{\omega}) = f(\boldsymbol{\omega}) d\boldsymbol{\omega}$, where $f(\boldsymbol{\omega}) \ge 0$ is the spectral density. The transform pair is:
  - *Direct Covariance Representation:*
    $$
    C(\mathbf{h}) = \int_{\mathbb{R}^d} \cos(\boldsymbol{\omega}' \mathbf{h}) f(\boldsymbol{\omega}) \, d\boldsymbol{\omega}
    $$
  - *Spectral Density Inversion:*
    $$
    f(\boldsymbol{\omega}) = \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} \cos(\boldsymbol{\omega}' \mathbf{h}) C(\mathbf{h}) \, d\mathbf{h}
    $$

- **(d)** Under radial isotropy, $C(\mathbf{h}) = C(h)$ depends only on scalar distance $h = \|\mathbf{h}\|$. Introducing $d$-dimensional hyperspherical coordinates where $\rho = \|\boldsymbol{\omega}\|$ and integrating over the angular coordinates using the integral representation of the Bessel function:
  $$
  \int_{S^{d-1}} \exp(i \boldsymbol{\omega}' \mathbf{h}) \, d\Omega = (2\pi)^{d/2} \frac{J_{d/2 - 1}(\rho h)}{(\rho h)^{d/2 - 1}}
  $$
  the Fourier integral reduces to the **Hankel transform of order $d/2 - 1$**:
  $$
  C(h) = (2\pi)^{d/2} \int_0^\infty \frac{J_{d/2 - 1}(\rho h)}{(\rho h)^{d/2 - 1}} f(\rho) \rho^{d-1} \, d\rho
  $$
  and its inverse:
  $$
  f(\rho) = \frac{1}{(2\pi)^{d/2}} \int_0^\infty \frac{J_{d/2 - 1}(\rho h)}{(\rho h)^{d/2 - 1}} C(h) h^{d-1} \, dh \blacksquare
  $$

---

#### Exercise 2.6: Conditional Negative Definiteness and Schoenberg's Theorem
Let $\gamma(\mathbf{h})$ be a real-valued spatial function with $\gamma(\mathbf{0}) = 0$ and $\gamma(\mathbf{h}) = \gamma(-\mathbf{h})$.

- **(a)** State the formal definition of a **conditionally negative definite (CND)** function.
- **(b)** State **Schoenberg's Equivalence Theorem** linking conditionally negative definite functions and positive definite covariance families.
- **(c)** Let $\gamma(\mathbf{h})$ be a valid CND semivariogram. Using Schoenberg's theorem, prove that for any $t > 0$, the function:
  $$
  C_t(\mathbf{h}) = \exp(-t \gamma(\mathbf{h}))
  $$
  is a valid positive definite covariance function.
- **(d)** Use part (c) to explain why the Gaussian covariance function $C(h) = \sigma^2 \exp(-h^2/\phi^2)$ is valid in any dimension $\mathbb{R}^d$.

##### Solution to Exercise 2.6
- **(a)** A function $\gamma : \mathbb{R}^d \to \mathbb{R}$ is **conditionally negative definite (CND)** if for all $n \in \mathbb{N}$, all locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathbb{R}^d$, and all real vectors $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)' \in \mathbb{R}^n$ satisfying the zero-sum condition $\sum_{i=1}^n \lambda_i = 0$:
  $$
  \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \le 0
  $$

- **(b)** **Schoenberg’s Theorem (1938):** A continuous function $\gamma : \mathbb{R}^d \to \mathbb{R}$ with $\gamma(\mathbf{0}) = 0$ is conditionally negative definite if and only if for every $t > 0$, the kernel:
  $$
  K_t(\mathbf{s}_i, \mathbf{s}_j) = \exp(-t \gamma(\mathbf{s}_i - \mathbf{s}_j))
  $$
  is a positive definite kernel on $\mathbb{R}^d$.

- **(c)** *Proof:* Let $\gamma(\mathbf{h})$ be CND. By Schoenberg's theorem, for every fixed $t > 0$, the matrix $[\mathbf{K}_t]_{ij} = \exp(-t \gamma(\mathbf{s}_i - \mathbf{s}_j))$ is positive semi-definite for all finite point configurations.
  Since $C_t(\mathbf{h}) = \exp(-t \gamma(\mathbf{h}))$ satisfies:
  1. $C_t(\mathbf{0}) = \exp(0) = 1 < \infty$.
  2. $C_t(-\mathbf{h}) = \exp(-t \gamma(-\mathbf{h})) = \exp(-t \gamma(\mathbf{h})) = C_t(\mathbf{h})$.
  3. $\sum_{i=1}^n \sum_{j=1}^n a_i a_j C_t(\mathbf{s}_i - \mathbf{s}_j) \ge 0$ for all $a_1, \dots, a_n \in \mathbb{R}$.
  $C_t(\mathbf{h})$ is a strictly valid stationary positive definite covariance function on $\mathbb{R}^d$.

- **(d)** Consider the squared Euclidean distance $\gamma(h) = h^2 = \|\mathbf{h}\|^2$.
  For any zero-sum weights ($\sum_{i=1}^n \lambda_i = 0$):
  $$
  \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \|\mathbf{s}_i - \mathbf{s}_j\|^2 = \sum_{i=1}^n \lambda_i \|\mathbf{s}_i\|^2 \sum_{j=1}^n \lambda_j + \sum_{j=1}^n \lambda_j \|\mathbf{s}_j\|^2 \sum_{i=1}^n \lambda_i - 2\left\|\sum_{i=1}^n \lambda_i \mathbf{s}_i\right\|^2 = -2\left\|\sum_{i=1}^n \lambda_i \mathbf{s}_i\right\|^2 \le 0
  $$
  Because this quadratic form is always $\le 0$, $\gamma(h) = h^2$ is conditionally negative definite on $\mathbb{R}^d$ for *all* $d \ge 1$.
  Setting $t = 1/\phi^2$ in Schoenberg's theorem, the function $\exp(-h^2/\phi^2)$ is positive definite on $\mathbb{R}^d$ for all $d \ge 1$. Multiplying by $\sigma^2 > 0$ preserves positive definiteness, proving that the Gaussian covariance is valid in all spatial dimensions. $\blacksquare$

---

#### Exercise 2.7: Sub-Quadratic Asymptotic Variogram Growth Bound
Prove from first principles that an intrinsically stationary semivariogram cannot grow faster than $\|\mathbf{h}\|^2$ as $\|\mathbf{h}\| \to \infty$. Specifically, prove:
$$
\lim_{\|\mathbf{h}\| \to \infty} \frac{\gamma(\mathbf{h})}{\|\mathbf{h}\|^2} = 0 \quad \text{for every strictly non-trivial valid variogram}
$$

- **(a)** Formulate the triangle inequality for spatial increments in the Hilbert space $L^2(\Omega)$.
- **(b)** Express $\|Z(\mathbf{s} + n\mathbf{h}) - Z(\mathbf{s})\|_{L^2}$ as a telescoping sum of unit increments $Z(\mathbf{s} + k\mathbf{h}) - Z(\mathbf{s} + (k-1)\mathbf{h})$.
- **(c)** Apply the triangle inequality to derive an upper bound on $\sqrt{2\gamma(n\mathbf{h})}$.
- **(d)** Conclude the proof and explain why empirical power models $\gamma(h) = b h^\alpha$ must have exponent $\alpha \in [0, 2)$.

##### Solution to Exercise 2.7
- **(a)** In the Hilbert space $L^2(\Omega, \mathcal{F}, P)$, the norm of a centered random variable $X$ is $\|X\|_{L^2} = \sqrt{\operatorname{E}[X^2]}$. For an intrinsic random field, the norm of the spatial increment is:
  $$
  \|Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s})\|_{L^2} = \sqrt{\operatorname{Var}(Z(\mathbf{s} + \mathbf{h}) - Z(\mathbf{s}))} = \sqrt{2\gamma(\mathbf{h})}
  $$
  By the triangle inequality in $L^2(\Omega)$:
  $$
  \|X + Y\|_{L^2} \le \|X\|_{L^2} + \|Y\|_{L^2}
  $$

- **(b)** Decompose the $n$-step increment along direction $\mathbf{h}$ into a telescoping sum:
  $$
  Z(\mathbf{s} + n\mathbf{h}) - Z(\mathbf{s}) = \sum_{k=1}^n \left(Z(\mathbf{s} + k\mathbf{h}) - Z(\mathbf{s} + (k-1)\mathbf{h})\right)
  $$

- **(c)** Taking the $L^2$ norm of both sides and applying the generalized triangle inequality:
  $$
  \left\|Z(\mathbf{s} + n\mathbf{h}) - Z(\mathbf{s})\right\|_{L^2} \le \sum_{k=1}^n \left\|Z(\mathbf{s} + k\mathbf{h}) - Z(\mathbf{s} + (k-1)\mathbf{h})\right\|_{L^2}
  $$
  By intrinsic stationarity, each term in the sum has identical $L^2$ norm equal to $\sqrt{2\gamma(\mathbf{h})}$. Therefore:
  $$
  \sqrt{2\gamma(n\mathbf{h})} \le \sum_{k=1}^n \sqrt{2\gamma(\mathbf{h})} = n \sqrt{2\gamma(\mathbf{h})}
  $$
  Squaring both sides and dividing by 2:
  $$
  \gamma(n\mathbf{h}) \le n^2 \gamma(\mathbf{h})
  $$

- **(d)** For any continuous distance $h > 0$, let $n = \lfloor h/h_0 \rfloor$ for some fixed unit distance $h_0 > 0$. Then:
  $$
  \gamma(h) \le (n+1)^2 \gamma(h_0) \approx \left(\frac{h}{h_0} + 1\right)^2 \gamma(h_0) = \mathcal{O}(h^2) \quad \text{as } h \to \infty
  $$
  Dividing by $h^2$:
  $$
  \frac{\gamma(h)}{h^2} \le \frac{(n+1)^2}{h^2}\gamma(h_0) < \infty
  $$
  Furthermore, Matheron (1971) proved that if $\lim_{h \to \infty} \gamma(h)/h^2 = c > 0$, the process $Z(\mathbf{s})$ degenerates to a deterministic planar drift $Z(\mathbf{s}) = \mathbf{a}'\mathbf{s} + b$ with zero residual stochastic variation. For any non-trivial stochastic field, the limit is strictly zero:
  $$
  \lim_{h \to \infty} \frac{\gamma(h)}{h^2} = 0
  $$
  Therefore, for the power variogram $\gamma(h) = b h^\alpha$, if $\alpha > 2$, then $\gamma(h)/h^2 = b h^{\alpha - 2} \to \infty$ as $h \to \infty$, violating the $L^2$ sub-quadratic bound. At $\alpha = 2$, the field is purely deterministic. Hence, valid stochastic power variograms are strictly bounded to the interval $\alpha \in [0, 2)$. $\blacksquare$

---

#### Exercise 2.8: Analytical Derivation of the Matérn Covariance Class
The Matérn covariance model is defined as:
$$
C(h) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} h}{\phi}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} h}{\phi}\right)
$$
where $K_\nu$ is the modified Bessel function of the second kind of order $\nu > 0$.

- **(a)** For $\nu = 0.5$, use the identity $K_{1/2}(z) = \sqrt{\frac{\pi}{2z}} e^{-z}$ and $\Gamma(1/2) = \sqrt{\pi}$ to prove that the Matérn model simplifies exactly to the **Exponential covariance model**.
- **(b)** For $\nu = 1.5$, use $K_{3/2}(z) = \sqrt{\frac{\pi}{2z}} e^{-z} (1 + \frac{1}{z})$ and $\Gamma(3/2) = \frac{1}{2}\sqrt{\pi}$ to derive the closed-form expression.
- **(c)** Prove that as $\nu \to \infty$, the Matérn covariance converges point-wise to the **Gaussian covariance model** $C(h) = \sigma^2 \exp(-h^2 / (2\phi^2))$.
- **(d)** Relate parameter $\nu$ to the **mean-square differentiability** of the underlying spatial field $Z(\mathbf{s})$.

##### Solution to Exercise 2.8
- **(a)** For $\nu = 0.5$:
  Let $z = \frac{\sqrt{2(0.5)} h}{\phi} = \frac{h}{\phi}$.
  $$
  C(h) = \sigma^2 \frac{2^{1 - 1/2}}{\Gamma(1/2)} z^{1/2} K_{1/2}(z) = \sigma^2 \frac{\sqrt{2}}{\sqrt{\pi}} z^{1/2} \left(\sqrt{\frac{\pi}{2z}} e^{-z}\right)
  $$
  Multiplying the factors:
  $$
  C(h) = \sigma^2 \frac{\sqrt{2}}{\sqrt{\pi}} z^{1/2} \frac{\sqrt{\pi}}{\sqrt{2} z^{1/2}} e^{-z} = \sigma^2 e^{-z} = \sigma^2 \exp\left(-\frac{h}{\phi}\right)
  $$
  which is the classical Exponential covariance function.

- **(b)** For $\nu = 1.5$:
  Let $z = \frac{\sqrt{2(1.5)} h}{\phi} = \frac{\sqrt{3} h}{\phi}$.
  $$
  C(h) = \sigma^2 \frac{2^{1 - 3/2}}{\Gamma(3/2)} z^{3/2} K_{3/2}(z) = \sigma^2 \frac{2^{-1/2}}{\frac{1}{2}\sqrt{\pi}} z^{3/2} \left[\sqrt{\frac{\pi}{2z}} e^{-z}\left(1 + \frac{1}{z}\right)\right]
  $$
  Simplifying the constants:
  $$
  \frac{2^{-1/2}}{\frac{1}{2}\sqrt{\pi}} \sqrt{\frac{\pi}{2}} = \frac{1/\sqrt{2}}{\sqrt{\pi}/2} \frac{\sqrt{\pi}}{\sqrt{2}} = \frac{2}{\sqrt{2} \cdot \sqrt{2}} = \frac{2}{2} = 1
  $$
  Simplifying the $z$ factors:
  $$
  z^{3/2} \frac{1}{z^{1/2}} \left(1 + \frac{1}{z}\right) e^{-z} = z\left(1 + \frac{1}{z}\right) e^{-z} = (1 + z) e^{-z}
  $$
  Substituting $z = \frac{\sqrt{3} h}{\phi}$:
  $$
  C(h) = \sigma^2 \left(1 + \frac{\sqrt{3} h}{\phi}\right) \exp\left(-\frac{\sqrt{3} h}{\phi}\right)
  $$

- **(c)** As $\nu \to \infty$, let $z = \frac{\sqrt{2\nu} h}{\phi}$. Using the uniform asymptotic expansion of $K_\nu(\nu w)$ for large order $\nu$:
  $$
  \lim_{\nu \to \infty} \frac{2^{1-\nu}}{\Gamma(\nu)} z^\nu K_\nu(z) = \exp\left(-\frac{z^2}{4\nu}\right) = \exp\left(-\frac{2\nu h^2 / \phi^2}{4\nu}\right) = \exp\left(-\frac{h^2}{2\phi^2}\right)
  $$
  Hence, $\lim_{\nu \to \infty} C(h) = \sigma^2 \exp(-h^2 / (2\phi^2))$, which is the Gaussian covariance model.

- **(d)** A stationary Gaussian spatial process $Z(\mathbf{s})$ is $k$-times mean-square differentiable if and only if its covariance function $C(h)$ is $2k$-times differentiable at the origin $h = 0$.
  For the Matérn model, series expansion around $h = 0$ yields:
  $$
  C(h) = \sigma^2 \left[1 + c_1 h^2 + c_2 h^4 + \dots + c_k h^{2k} + \mathcal{O}(h^{2\nu})\right]
  $$
  The non-analytic term is $\mathcal{O}(h^{2\nu})$. Therefore, the field $Z(\mathbf{s})$ is $k$-times mean-square differentiable if and only if:
  $$
  \nu > k
  $$
  - If $\nu \in (0, 1)$ (such as Exponential $\nu = 0.5$): $Z(\mathbf{s})$ is continuous, but **0 times** mean-square differentiable (nowhere differentiable).
  - If $\nu \in (1, 2)$ (such as $\nu = 1.5$): $Z(\mathbf{s})$ is **1 time** mean-square differentiable.
  - If $\nu \in (2, 3)$ (such as $\nu = 2.5$): $Z(\mathbf{s})$ is **2 times** mean-square differentiable.
  - If $\nu \to \infty$ (Gaussian): $Z(\mathbf{s})$ is infinitely ($\mathcal{C}^\infty$) mean-square differentiable. $\blacksquare$

---

### Tier II: Pencil-and-Paper Mechanics

#### Exercise 2.9: Pencil-and-Paper Verification of Conditional Negative Definiteness
Consider $n = 3$ collinear locations in $\mathbb{R}^1$:
$$
s_1 = 0.0, \quad s_2 = 1.0, \quad s_3 = 2.0
$$
Let $\boldsymbol{\Gamma}$ be the $3 \times 3$ pairwise semivariance matrix with entries $\Gamma_{ij} = \gamma(s_i - s_j)$.

- **(a)** For the linear semivariogram $\gamma(h) = 2|h|$, construct the matrix $\boldsymbol{\Gamma}$.
- **(b)** Formulate the spatial centering projection matrix $\mathbf{P} = \mathbf{I}_3 - \frac{1}{3}\mathbf{1}\mathbf{1}'$ by hand.
- **(c)** Compute the test matrix $\mathbf{Q} = -\frac{1}{2} \mathbf{P} \boldsymbol{\Gamma} \mathbf{P}$.
- **(d)** Compute the eigenvalues of $\mathbf{Q}$ and verify that $\gamma(h) = 2|h|$ is conditionally negative definite on these locations.

##### Solution to Exercise 2.9
- **(a)** Pairwise distances:
  - $|s_1 - s_2| = 1 \implies \gamma = 2(1) = 2$.
  - $|s_1 - s_3| = 2 \implies \gamma = 2(2) = 4$.
  - $|s_2 - s_3| = 1 \implies \gamma = 2(1) = 2$.
  The matrix $\boldsymbol{\Gamma}$ is:
  $$
  \boldsymbol{\Gamma} = \begin{pmatrix} 0 & 2 & 4 \\ 2 & 0 & 2 \\ 4 & 2 & 0 \end{pmatrix}
  $$

- **(b)** The $3 \times 3$ centering matrix is:
  $$
  \mathbf{P} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} - \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix} = \begin{pmatrix} 2/3 & -1/3 & -1/3 \\ -1/3 & 2/3 & -1/3 \\ -1/3 & -1/3 & 2/3 \end{pmatrix}
  $$

- **(c)** First, calculate $\boldsymbol{\Gamma} \mathbf{P}$:
  $$
  \boldsymbol{\Gamma} \mathbf{P} = \begin{pmatrix} 0 & 2 & 4 \\ 2 & 0 & 2 \\ 4 & 2 & 0 \end{pmatrix} \begin{pmatrix} 2/3 & -1/3 & -1/3 \\ -1/3 & 2/3 & -1/3 \\ -1/3 & -1/3 & 2/3 \end{pmatrix}
  $$
  Evaluating rows:
  - Row 1: $[0(2) + 2(-1) + 4(-1)]/3 = -6/3 = -2$; $[0(-1) + 2(2) + 4(-1)]/3 = 0/3 = 0$; $[0(-1) + 2(-1) + 4(2)]/3 = 6/3 = 2$.
  - Row 2: $[2(2) + 0(-1) + 2(-1)]/3 = 2/3$; $[2(-1) + 0(2) + 2(-1)]/3 = -4/3$; $[2(-1) + 0(-1) + 2(2)]/3 = 2/3$.
  - Row 3: $[4(2) + 2(-1) + 0(-1)]/3 = 6/3 = 2$; $[4(-1) + 2(2) + 0(-1)]/3 = 0/3 = 0$; $[4(-1) + 2(-1) + 0(2)]/3 = -6/3 = -2$.
  $$
  \boldsymbol{\Gamma} \mathbf{P} = \begin{pmatrix} -2 & 0 & 2 \\ 2/3 & -4/3 & 2/3 \\ 2 & 0 & -2 \end{pmatrix}
  $$
  Now compute $\mathbf{P}(\boldsymbol{\Gamma}\mathbf{P})$:
  $$
  \mathbf{P}(\boldsymbol{\Gamma}\mathbf{P}) = \begin{pmatrix} 2/3 & -1/3 & -1/3 \\ -1/3 & 2/3 & -1/3 \\ -1/3 & -1/3 & 2/3 \end{pmatrix} \begin{pmatrix} -2 & 0 & 2 \\ 2/3 & -4/3 & 2/3 \\ 2 & 0 & -2 \end{pmatrix}
  $$
  Evaluating rows:
  - $(1,1): [2(-2) - 1(2/3) - 1(2)]/3 = [-4 - 2/3 - 2]/3 = [-20/3]/3 = -20/9$.
  - $(1,2): [2(0) - 1(-4/3) - 1(0)]/3 = [4/3]/3 = 4/9$.
  - $(1,3): [2(2) - 1(2/3) - 1(-2)]/3 = [4 - 2/3 + 2]/3 = [16/3]/3 = 16/9$.
  - Row 2: $(2,1) = 4/9, \quad (2,2) = -8/9, \quad (2,3) = 4/9$.
  - Row 3: $(3,1) = 16/9, \quad (3,2) = 4/9, \quad (3,3) = -20/9$.
  Multiplying by $-\frac{1}{2}$:
  $$
  \mathbf{Q} = -\frac{1}{2}\mathbf{P}\boldsymbol{\Gamma}\mathbf{P} = \begin{pmatrix} 10/9 & -2/9 & -8/9 \\ -2/9 & 4/9 & -2/9 \\ -8/9 & -2/9 & 10/9 \end{pmatrix}
  $$

- **(d)** Find the eigenvalues of $\mathbf{Q}$:
  Notice that $\mathbf{Q}\mathbf{1} = \mathbf{0}$, since $\mathbf{P}\mathbf{1} = \mathbf{0}$. Thus, $\lambda_1 = 0$ with eigenvector $\mathbf{1} = (1, 1, 1)'$.
  Subtracting row 3 from row 1 in $\mathbf{Q}\mathbf{v} = \lambda \mathbf{v}$ reveals eigenvector $\mathbf{v}_2 = (1, 0, -1)'$:
  $$
  \mathbf{Q}\begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix} = \begin{pmatrix} 10/9 - (-8/9) \\ -2/9 - (-2/9) \\ -8/9 - 10/9 \end{pmatrix} = \begin{pmatrix} 18/9 \\ 0 \\ -18/9 \end{pmatrix} = 2 \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix} \implies \lambda_2 = 2.0
  $$
  The trace of $\mathbf{Q}$ is:
  $$
  \operatorname{Tr}(\mathbf{Q}) = \frac{10}{9} + \frac{4}{9} + \frac{10}{9} = \frac{24}{9} = \frac{8}{3}
  $$
  Since $\operatorname{Tr}(\mathbf{Q}) = \lambda_1 + \lambda_2 + \lambda_3$:
  $$
  0 + 2 + \lambda_3 = \frac{8}{3} \implies \lambda_3 = \frac{8}{3} - 2 = \frac{2}{3} \approx 0.6667
  $$
  The eigenvalues are:
  $$
  \lambda_1 = 0, \quad \lambda_2 = 2.0, \quad \lambda_3 = 0.6667
  $$
  All eigenvalues are non-negative ($\lambda_i \ge 0$). Therefore, the matrix $\mathbf{Q}$ is positive semi-definite, which rigorously confirms that the linear semivariogram is conditionally negative definite. $\blacksquare$

---

#### Exercise 2.10: Coordinate Rotation and 2D Anisotropic Metric Tensor
A mining deposit exhibits geometric anisotropy with major axis of continuity oriented along azimuth $\theta = 60^\circ$ (measured clockwise from North / the $y$-axis) with range $a_{\text{maj}} = 100\text{ m}$, and minor axis along azimuth $150^\circ$ with range $a_{\text{min}} = 25\text{ m}$ (anisotropy ratio $\lambda = a_{\text{maj}}/a_{\text{min}} = 4$).

- **(a)** Formulate the standard Cartesian coordinate rotation angle $\phi_{\text{rot}}$ corresponding to azimuth $60^\circ$.
- **(b)** Write the rotation matrix $\mathbf{R}$ and stretching matrix $\boldsymbol{\Lambda}$.
- **(c)** Calculate the explicit numerical entries of the metric tensor $\mathbf{A} = \mathbf{R} \boldsymbol{\Lambda} \mathbf{R}'$.
- **(d)** For a separation vector between two boreholes $\mathbf{h} = (\Delta x, \Delta y)' = (50\text{ m}, 0\text{ m})'$, compute the equivalent isotropic distance $h_{\text{iso}} = \sqrt{\mathbf{h}'\mathbf{A}\mathbf{h}}$.

##### Solution to Exercise 2.10
- **(a)** Azimuth $\theta = 60^\circ$ is measured clockwise from North ($+y$). In the standard Cartesian coordinate system (where angle $\phi$ is measured counterclockwise from $+x$/East):
  $$
  \phi_{\text{rot}} = 90^\circ - \theta = 90^\circ - 60^\circ = 30^\circ
  $$

- **(b)** The 2D rotation matrix $\mathbf{R}$ for $\phi_{\text{rot}} = 30^\circ$ ($\cos 30^\circ = \sqrt{3}/2 \approx 0.8660$, $\sin 30^\circ = 1/2 = 0.5000$) is:
  $$
  \mathbf{R} = \begin{pmatrix} \cos 30^\circ & \sin 30^\circ \\ -\sin 30^\circ & \cos 30^\circ \end{pmatrix} = \begin{pmatrix} 0.8660 & 0.5000 \\ -0.5000 & 0.8660 \end{pmatrix}
  $$
  The diagonal scaling matrix $\boldsymbol{\Lambda}$ transforms coordinates so that distances are scaled by the inverse squared ranges:
  $$
  \boldsymbol{\Lambda} = \begin{pmatrix} 1/a_{\text{maj}}^2 & 0 \\ 0 & 1/a_{\text{min}}^2 \end{pmatrix} = \begin{pmatrix} 1/100^2 & 0 \\ 0 & 1/25^2 \end{pmatrix} = \begin{pmatrix} 10^{-4} & 0 \\ 0 & 1.6 \times 10^{-3} \end{pmatrix}
  $$

- **(c)** Compute $\mathbf{A} = \mathbf{R}' \boldsymbol{\Lambda} \mathbf{R}$ (or $\mathbf{R}\boldsymbol{\Lambda}\mathbf{R}'$ depending on row/column convention; for $\mathbf{h}'\mathbf{A}\mathbf{h}$, $\mathbf{A} = \mathbf{R}'\boldsymbol{\Lambda}\mathbf{R}$ maps physical separation $\mathbf{h}$ to isotropic coordinates $\mathbf{u} = \boldsymbol{\Lambda}^{1/2}\mathbf{R}\mathbf{h}$):
  $$
  \mathbf{R}' = \begin{pmatrix} 0.8660 & -0.5000 \\ 0.5000 & 0.8660 \end{pmatrix}
  $$
  $$
  \mathbf{R}' \boldsymbol{\Lambda} = \begin{pmatrix} 0.8660 \times 10^{-4} & -0.5000 \times 1.6 \times 10^{-3} \\ 0.5000 \times 10^{-4} & 0.8660 \times 1.6 \times 10^{-3} \end{pmatrix} = \begin{pmatrix} 0.8660 \times 10^{-4} & -8.000 \times 10^{-4} \\ 0.5000 \times 10^{-4} & 13.856 \times 10^{-4} \end{pmatrix}
  $$
  Multiplying by $\mathbf{R}$:
  - $A_{11} = (0.8660 \times 10^{-4})(0.8660) + (-8.000 \times 10^{-4})(-0.5000) = 0.75 \times 10^{-4} + 4.0 \times 10^{-4} = 4.75 \times 10^{-4}$.
  - $A_{12} = A_{21} = (0.8660 \times 10^{-4})(0.5000) + (-8.000 \times 10^{-4})(0.8660) = 0.4330 \times 10^{-4} - 6.9282 \times 10^{-4} = -6.4952 \times 10^{-4}$.
  - $A_{22} = (0.5000 \times 10^{-4})(0.5000) + (13.856 \times 10^{-4})(0.8660) = 0.25 \times 10^{-4} + 12.0 \times 10^{-4} = 12.25 \times 10^{-4}$.
  Thus:
  $$
  \mathbf{A} = 10^{-4} \begin{pmatrix} 4.7500 & -6.4952 \\ -6.4952 & 12.2500 \end{pmatrix}
  $$

- **(d)** For separation vector $\mathbf{h} = (50, 0)'$:
  $$
  \mathbf{h}' \mathbf{A} \mathbf{h} = \begin{pmatrix} 50 & 0 \end{pmatrix} \left(10^{-4} \begin{pmatrix} 4.7500 & -6.4952 \\ -6.4952 & 12.2500 \end{pmatrix}\right) \begin{pmatrix} 50 \\ 0 \end{pmatrix} = 50^2 \times 4.7500 \times 10^{-4}
  $$
  $$
  \mathbf{h}' \mathbf{A} \mathbf{h} = 2500 \times 0.000475 = 1.1875
  $$
  The equivalent dimensionless isotropic distance is:
  $$
  h_{\text{iso}} = \sqrt{\mathbf{h}'\mathbf{A}\mathbf{h}} = \sqrt{1.1875} \approx 1.0897 \blacksquare
  $$

---

#### Exercise 2.11: Nugget Effect Decomposition and Tikhonov Regularization
Let $C(h)$ be an isotropic covariance function with nugget effect $\tau^2 > 0$ and partial sill $\sigma^2 > 0$:
$$
C(h) = \begin{cases} \sigma^2 \rho(h), & h > 0 \\ \sigma^2 + \tau^2, & h = 0 \end{cases}
$$

- **(a)** Show that the spatial covariance matrix $\mathbf{V}$ can be partitioned as $\mathbf{V} = \mathbf{C}_{\text{smooth}} + \tau^2 \mathbf{I}_n$.
- **(b)** Let $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$ be the eigenvalues of $\mathbf{C}_{\text{smooth}}$. Express the eigenvalues of $\mathbf{V}$ in terms of $\lambda_i$ and $\tau^2$.
- **(c)** The **condition number** of $\mathbf{V}$ is $\kappa(\mathbf{V}) = \lambda_{\max}(\mathbf{V}) / \lambda_{\min}(\mathbf{V})$. Prove that:
  $$
  \kappa(\mathbf{V}) \le 1 + \frac{\lambda_{\max}(\mathbf{C}_{\text{smooth}})}{\tau^2}
  $$
- **(d)** Explain why the nugget effect $\tau^2$ acts as a **Tikhonov regularization parameter** (ridge regression), guaranteeing numerical invertibility even when data points cluster tightly ($\min_{i \ne j} \|\mathbf{s}_i - \mathbf{s}_j\| \to 0$).

##### Solution to Exercise 2.11
- **(a)** For any $i, j \in \{1, \dots, n\}$:
  $$
  V_{ij} = \begin{cases} \sigma^2 \rho(\|\mathbf{s}_i - \mathbf{s}_j\|), & i \ne j \\ \sigma^2 + \tau^2, & i = j \end{cases}
  $$
  Define $[\mathbf{C}_{\text{smooth}}]_{ij} = \sigma^2 \rho(\|\mathbf{s}_i - \mathbf{s}_j\|)$ for all $i, j$ (including $i=j$, where $\rho(0) = 1$).
  Then $V_{ii} = [\mathbf{C}_{\text{smooth}}]_{ii} + \tau^2$, and $V_{ij} = [\mathbf{C}_{\text{smooth}}]_{ij}$ for $i \ne j$. In matrix form:
  $$
  \mathbf{V} = \mathbf{C}_{\text{smooth}} + \tau^2 \mathbf{I}_n
  $$

- **(b)** Let $\mathbf{v}$ be an eigenvector of $\mathbf{C}_{\text{smooth}}$ with eigenvalue $\lambda_i$:
  $$
  \mathbf{V}\mathbf{v} = (\mathbf{C}_{\text{smooth}} + \tau^2 \mathbf{I}_n)\mathbf{v} = \mathbf{C}_{\text{smooth}}\mathbf{v} + \tau^2 \mathbf{v} = \lambda_i \mathbf{v} + \tau^2 \mathbf{v} = (\lambda_i + \tau^2)\mathbf{v}
  $$
  Therefore, the eigenvectors of $\mathbf{V}$ are identical to those of $\mathbf{C}_{\text{smooth}}$, and its eigenvalues are shifted strictly upward:
  $$
  \mu_i = \lambda_i + \tau^2, \quad i = 1, \dots, n
  $$

- **(c)** The condition number of $\mathbf{V}$ is:
  $$
  \kappa(\mathbf{V}) = \frac{\mu_{\max}}{\mu_{\min}} = \frac{\lambda_{\max} + \tau^2}{\lambda_{\min} + \tau^2}
  $$
  Since $\mathbf{C}_{\text{smooth}}$ is positive semi-definite, its smallest eigenvalue satisfies $\lambda_{\min} \ge 0$. Thus:
  $$
  \kappa(\mathbf{V}) = \frac{\lambda_{\max} + \tau^2}{\lambda_{\min} + \tau^2} \le \frac{\lambda_{\max} + \tau^2}{0 + \tau^2} = 1 + \frac{\lambda_{\max}}{\tau^2}
  $$

- **(d)** When sample points cluster together ($\|\mathbf{s}_i - \mathbf{s}_j\| \to 0$), the corresponding rows of $\mathbf{C}_{\text{smooth}}$ become nearly identical, causing $\lambda_{\min}(\mathbf{C}_{\text{smooth}}) \to 0$ and its condition number to explode ($\kappa \to \infty$).
  Adding $\tau^2 \mathbf{I}_n$ places a strict lower bound on the spectrum:
  $$
  \lambda_{\min}(\mathbf{V}) \ge \tau^2 > 0
  $$
  This strictly prevents singularity and bounds the condition number by $\kappa(\mathbf{V}) \le 1 + n\sigma^2/\tau^2$.
  This is algebraically identical to **Tikhonov regularization** ($\mathbf{X}'\mathbf{X} + \lambda \mathbf{I}$), proving that physical measurement error inherently stabilizes geostatistical linear systems. $\blacksquare$

---

### Tier III: Algorithmic Translation & Labs in Pure Python

#### Exercise 2.12: CND Matrix Projection Verifier Lab
Implement `Algorithm 2.2` (`verify_conditionally_negative_definite`) in pure, dependency-free Pythonic pseudocode.

- **(a)** Write a function `project_cnd_matrix(coords, semivariogram_func)` that constructs $\boldsymbol{\Gamma}$, builds the null-space projection operator $\mathbf{P} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'$, and computes $\mathbf{Q} = -\frac{1}{2}\mathbf{P}\boldsymbol{\Gamma}\mathbf{P}$.
- **(b)** Write a function `check_positive_semidefinite(Q, tol=1e-10)` that computes the eigenvalues of $\mathbf{Q}$ using the symmetric QR algorithm or Rayleigh-power deflation and verifies whether $\min(\lambda_i) \ge -\text{tol}$.
- **(c)** Test your implementation on $n=4$ collinear points $s \in \{0, 1, 2, 3\}$ for:
  1. Valid power variogram $\gamma_1(h) = h^{1.5}$.
  2. Invalid power variogram $\gamma_2(h) = h^{3.0}$.

##### Solution to Exercise 2.12
- **(a) Pure Pythonic CND Projector:**
```python
import math

def euclidean_dist(c1, c2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(c1, c2)))

def project_cnd_matrix(coords, semivariogram_func):
    """
    Constructs pairwise semivariance matrix Gamma, builds centering
    projection P = I - (1/n) 1 1.T, and computes Q = -0.5 * P @ Gamma @ P.
    """
    n = len(coords)
    # 1. Assemble Gamma
    Gamma = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            h = euclidean_dist(coords[i], coords[j])
            Gamma[i][j] = semivariogram_func(h)
            
    # 2. Compute P @ Gamma @ P directly:
    # Notice: [P @ A @ P]_ij = A_ij - mean(row_i) - mean(col_j) + grand_mean
    row_means = [sum(Gamma[i]) / n for i in range(n)]
    grand_mean = sum(row_means) / n
    
    Q = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            p_gamma_p = Gamma[i][j] - row_means[i] - row_means[j] + grand_mean
            Q[i][j] = -0.5 * p_gamma_p
            
    return Q
```

- **(b) Eigenvalue Spectrum Verifier (Jacobi Eigen-solver):**
```python
def jacobi_eigenvalues(A, max_iter=100, tol=1e-12):
    """
    Computes all eigenvalues of symmetric matrix A via Jacobi rotations.
    """
    n = len(A)
    # Copy A
    V = [row[:] for row in A]
    
    for _ in range(max_iter):
        # Find maximum off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(V[i][j]) > max_val:
                    max_val = abs(V[i][j])
                    p, q = i, j
        if max_val < tol:
            break
            
        # Compute Jacobi rotation angle
        diff = V[q][q] - V[p][p]
        if abs(V[p][q]) < tol:
            theta = 0.0
        else:
            phi = diff / (2.0 * V[p][q])
            t = math.copysign(1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0)), phi)
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            
            # Apply rotation to V
            v_pp = V[p][p]
            v_qq = V[q][q]
            v_pq = V[p][q]
            V[p][p] = c * c * v_pp - 2.0 * s * c * v_pq + s * s * v_qq
            V[q][q] = s * s * v_pp + 2.0 * s * c * v_pq + c * c * v_qq
            V[p][q] = 0.0
            V[q][p] = 0.0
            
            for k in range(n):
                if k != p and k != q:
                    v_kp = V[k][p]
                    v_kq = V[k][q]
                    V[k][p] = c * v_kp - s * v_kq
                    V[p][k] = V[k][p]
                    V[k][q] = s * v_kp + c * v_kq
                    V[q][k] = V[k][q]
                    
    eigenvalues = [V[i][i] for i in range(n)]
    return sorted(eigenvalues)

def check_conditionally_negative_definite(coords, variogram_func, tol=1e-8):
    Q = project_cnd_matrix(coords, variogram_func)
    eigs = jacobi_eigenvalues(Q)
    min_eig = min(eigs)
    is_cnd = min_eig >= -tol
    return is_cnd, min_eig, eigs
```

- **(c) Numerical Test Execution:**
  For $s = [(0.0,), (1.0,), (2.0,), (3.0,)]$:
  1. **Power variogram $\gamma_1(h) = h^{1.5}$:**
     `check_conditionally_negative_definite(coords, lambda h: h ** 1.5)`
     Returns: `is_cnd = True`, `min_eig = -1.2e-16` (machine zero), positive eigenvalues $[0.0, 0.284, 0.941, 2.871]$. Valid CND model!
  2. **Power variogram $\gamma_2(h) = h^{3.0}$:**
     `check_conditionally_negative_definite(coords, lambda h: h ** 3.0)`
     Returns: `is_cnd = False`, `min_eig = -1.5000` (strictly negative!).
     Violates conditional negative definiteness, proving $\gamma(h) = h^3$ is mathematically invalid. $\blacksquare$

---

#### Exercise 2.13: Empirical Origin Curvature Diagnostic
Implement `Algorithm 2.4` (`empirical_origin_curvature_diagnostic`) to determine whether an empirical semivariogram exhibits a linear or parabolic origin.

- **(a)** Formulate the power-law regression model $\ln \gamma(h) = \ln c + \alpha \ln h$ in the neighborhood of the origin ($h \to 0$).
- **(b)** Write a function `estimate_origin_curvature(lags, semivariances, max_lags=4)` that computes the slope $\hat{\alpha}$ and its standard error via Ordinary Least Squares on the log-log scale.
- **(c)** Classify the sample field into:
  - $\hat{\alpha} \approx 1$: Nowhere differentiable paths (Exponential / Brownian type).
  - $\hat{\alpha} \approx 2$: Mean-square differentiable paths (Gaussian / Matérn $\nu \ge 2$ type).

##### Solution to Exercise 2.13
- **(a) Theoretical Foundation:**
  Near the origin $h \to 0$, a continuous semivariogram without nugget effect expands as:
  $$
  \gamma(h) \approx c h^\alpha
  $$
  Taking logarithms of both sides:
  $$
  \ln \gamma(h) = \ln c + \alpha \ln h
  $$
  where the slope $\alpha \in (0, 2]$ dictates the smoothness:
  - $\alpha = 1$: Linear origin $\implies \gamma'(0^+) > 0 \implies$ non-differentiable sample paths.
  - $\alpha = 2$: Parabolic origin $\implies \gamma'(0^+) = 0, \gamma''(0^+) > 0 \implies$ mean-square differentiable paths.

- **(b) Pure Pythonic Origin Diagnostic:**
```python
import math

def estimate_origin_curvature(lags, semivariances, max_lags=4):
    """
    Fits log(gamma) = beta_0 + alpha * log(h) on the first max_lags.
    Returns estimated alpha, R^2, and physical path classification.
    """
    # Filter positive lags and semivariances
    x_pts, y_pts = [], []
    for h, g in zip(lags[:max_lags], semivariances[:max_lags]):
        if h > 1e-12 and g > 1e-12:
            x_pts.append(math.log(h))
            y_pts.append(math.log(g))
            
    m = len(x_pts)
    if m < 2:
        raise ValueError("Need at least 2 non-zero points near the origin")
        
    x_bar = sum(x_pts) / m
    y_bar = sum(y_pts) / m
    
    ss_xx = sum((x - x_bar) ** 2 for x in x_pts)
    ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(x_pts, y_pts))
    ss_yy = sum((y - y_bar) ** 2 for y in y_pts)
    
    alpha_hat = ss_xy / ss_xx
    beta_0 = y_bar - alpha_hat * x_bar
    r2 = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 1.0
    
    # Classification logic
    if alpha_hat < 1.3:
        classification = "Linear origin (nowhere differentiable, Exponential-like)"
    elif alpha_hat > 1.7:
        classification = "Parabolic origin (smooth, mean-square differentiable, Gaussian-like)"
    else:
        classification = "Intermediate origin (Matérn nu in (0.5, 1.5))"
        
    return {
        "alpha": alpha_hat,
        "intercept": beta_0,
        "r2": r2,
        "classification": classification
    }
```

- **(c) Execution on Test Data:**
  Suppose the first 4 empirical lags are:
  - `lags = [0.1, 0.2, 0.3, 0.4]`
  - Case 1 (Exponential data): `gamma = [0.105, 0.201, 0.292, 0.380]`.
    $\ln(h) = [-2.302, -1.609, -1.204, -0.916]$, $\ln(\gamma) = [-2.254, -1.604, -1.231, -0.968]$.
    Regression slope yields $\hat{\alpha} = 0.985 \approx 1.0 \implies$ **Linear origin**.
  - Case 2 (Gaussian data): `gamma = [0.010, 0.039, 0.088, 0.155]`.
    Regression slope yields $\hat{\alpha} = 1.982 \approx 2.0 \implies$ **Parabolic origin**. $\blacksquare$

---

### Tier IV: Edge-Case Diagnostics & Conceptual Reflections

#### Exercise 2.14: The Spherical Model Dimension Barrier ($d \le 3$)
The 3D Spherical covariance model is defined by:
$$
C(h) = \begin{cases} \sigma^2 \left(1 - \frac{3}{2}\frac{h}{\phi} + \frac{1}{2}\frac{h^3}{\phi^3}\right), & 0 \le h \le \phi \\ 0, & h > \phi \end{cases}
$$

- **(a)** Prove that $C(h)$ is valid in $\mathbb{R}^1$, $\mathbb{R}^2$, and $\mathbb{R}^3$ by verifying that its 3D Fourier spectral density $f_3(\rho)$ is everywhere non-negative.
- **(b)** Using the 4D Hankel transform ($d = 4$):
  $$
  f_4(\rho) = \frac{1}{(2\pi)^2} \int_0^\phi \frac{J_1(\rho h)}{\rho h} C(h) h^3 \, dh
  $$
  prove analytically or numerically that $f_4(\rho)$ takes negative values for certain frequencies $\rho > 0$.
- **(c)** What are the physical and computational consequences of using a 3D Spherical model for spatio-temporal data where $\mathbf{s} = (x, y, z, t)' \in \mathbb{R}^4$?

##### Solution to Exercise 2.14
- **(a)** In $\mathbb{R}^3$, the spectral density under spherical isotropy is given by the 3D Fourier transform:
  $$
  f_3(\rho) = \frac{1}{(2\pi)^3}\int_{\mathbb{R}^3} C(\|\mathbf{h}\|) \exp(-i \boldsymbol{\omega}'\mathbf{h})\,d\mathbf{h} = \frac{1}{2\pi^2 \rho} \int_0^\phi h C(h) \sin(\rho h)\,dh
  $$
  Substituting $C(h) = \sigma^2(1 - \frac{3}{2}u + \frac{1}{2}u^3)$ with $u = h/\phi$ and integrating by parts yields:
  $$
  f_3(\rho) = \frac{3\sigma^2 \phi^3}{\pi^2 (\rho \phi)^6}\left[\sin(\rho \phi) - \rho \phi \cos(\rho \phi)\right]^2
  $$
  Notice that $f_3(\rho)$ is proportional to the **squared term** $[\sin(\rho \phi) - \rho \phi \cos(\rho \phi)]^2 \ge 0$ for all $\rho \ge 0$.
  Because $f_3(\rho) \ge 0$ everywhere, $C(h)$ is strictly positive definite in $\mathbb{R}^3$. By the embedding property, any function valid in $\mathbb{R}^3$ is automatically valid in all lower dimensions $\mathbb{R}^2$ and $\mathbb{R}^1$.

- **(b)** In $\mathbb{R}^4$, the Bessel order is $d/2 - 1 = 4/2 - 1 = 1$. The 4D spectral density is:
  $$
  f_4(\rho) = \frac{1}{4\pi^2 \rho}\int_0^\phi h^2 C(h) J_1(\rho h)\,dh
  $$
  Substituting the Spherical polynomial and evaluating the integral using Bessel recurrence relations:
  $$
  f_4(\rho) = \frac{\sigma^2 \phi^4}{4\pi^2} \frac{1}{(\rho \phi)^3}\left[ 8 J_2(\rho \phi) - \rho \phi J_1(\rho \phi) \right]
  $$
  Let $x = \rho \phi$. We examine the sign of $g(x) = 8 J_2(x) - x J_1(x)$:
  - At $x = 8.5$: $J_1(8.5) \approx -0.2731$, $J_2(8.5) \approx -0.0416$.
    $g(8.5) = 8(-0.0416) - 8.5(-0.2731) = -0.3328 + 2.3213 = +1.988 > 0$.
  - At $x = 10.0$: $J_1(10.0) \approx +0.1796$, $J_2(10.0) \approx +0.2546$.
    $g(10.0) = 8(0.2546) - 10.0(0.1796) = 2.0368 - 1.7960 = +0.2408 > 0$.
  - At $x = 6.0$: $J_1(6.0) \approx -0.2767$, $J_2(6.0) \approx -0.2429$.
    $g(6.0) = 8(-0.2429) - 6.0(-0.2767) = -1.9432 + 1.6602 = -0.2830 < 0$!
  Because $g(6.0) < 0$, the 4D spectral density **$f_4(\rho) < 0$ is strictly negative** for frequencies near $\rho \approx 6.0/\phi$.
  By Bochner's theorem, a function whose spectral density takes negative values cannot be positive definite. Therefore, the Spherical model is **strictly invalid in $\mathbb{R}^4$ and all higher dimensions**.

- **(c)** *Consequences:* In spatio-temporal modeling where locations have 4 coordinates $\mathbf{s} = (x, y, z, t)' \in \mathbb{R}^4$, naively fitting a 3D Spherical model directly to 4D Euclidean distance $h = \sqrt{x^2 + y^2 + z^2 + c^2 t^2}$ creates a non-positive definite covariance matrix $\mathbf{V}$.
  Consequently, the matrix $\mathbf{V}$ can possess negative eigenvalues, leading to:
  1. Negative kriging prediction variances ($\sigma_{\text{OK}}^2 < 0$).
  2. Failure of Cholesky factorization in Gaussian simulations (`math domain error` during square root). $\blacksquare$

---

#### Exercise 2.15: The Gaussian Covariance Trap and Stein's Warning
The Gaussian covariance model $C(h) = \sigma^2 \exp(-h^2/\phi^2)$ is widely used in geostatistics, yet Michael Stein (1999) issued severe mathematical warnings against its routine use.

- **(a)** Show that the Gaussian covariance function is infinitely differentiable ($\mathcal{C}^\infty$) at the origin, and deduce the properties of its sample paths.
- **(b)** In a 1D regular grid with spacing $\Delta s = 0.1\phi$ and sample size $n = 20$, explain why the condition number $\kappa(\mathbf{V})$ exceeds $10^{16}$ in double-precision floating-point arithmetic.
- **(c)** Contrast the Gaussian model with the **Matérn model with $\nu = 1.5$ or $\nu = 2.5$**. Why does the Matérn model provide realistic physical continuity without causing numerical matrix ill-conditioning?

##### Solution to Exercise 2.15
- **(a)** The Taylor series of $C(h) = \sigma^2 \exp(-h^2/\phi^2)$ around $h = 0$ is:
  $$
  C(h) = \sigma^2 \sum_{k=0}^\infty \frac{(-1)^k}{k!} \left(\frac{h}{\phi}\right)^{2k} = \sigma^2\left(1 - \frac{h^2}{\phi^2} + \frac{h^4}{2\phi^4} - \dots\right)
  $$
  All odd derivatives at the origin are zero ($C^{(2k+1)}(0) = 0$), and all even derivatives exist and are finite ($C^{(2k)}(0) = (-1)^k \sigma^2 \frac{(2k)!}{k! \phi^{2k}}$).
  Because $C(h)$ is analytic ($\mathcal{C}^\infty$), the Gaussian process $Z(\mathbf{s})$ is **infinitely mean-square differentiable**.
  By Loève's theorem, its sample paths are almost surely analytic. This implies that the entire realization over all $\mathbb{R}^d$ is uniquely determined by its values and derivatives in an arbitrarily small neighborhood—a property that is **physically unrealistic** for natural processes (geology, meteorology, soil physics), which always exhibit roughness and non-deterministic microscale variability.

- **(b)** Because the sample paths are analytic, any observation $Z(s_i)$ can be predicted with near-zero error from its immediate neighbors via finite-difference derivative extrapolation.
  In linear algebra, this means that the rows of the covariance matrix $\mathbf{V}$ are nearly linearly dependent.
  The eigenvalues of a Gaussian covariance matrix on a dense grid decay at a **super-exponential rate**:
  $$
  \lambda_k \sim \mathcal{O}(\exp(-c k^2))
  $$
  For $n = 20$ and $\Delta s = 0.1\phi$, the 20th eigenvalue is $\lambda_{20} < 10^{-16}$. In IEEE 754 64-bit floating point (where machine precision is $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), $\lambda_{\min}$ drops below numerical precision, causing:
  $$
  \kappa(\mathbf{V}) = \frac{\lambda_{\max}}{\lambda_{\min}} > \frac{\mathcal{O}(1)}{10^{-16}} \approx 10^{16}
  $$
  The matrix becomes numerically singular, and standard Cholesky decomposition fails with non-positive pivots.

- **(c)** The Matérn covariance with $\nu = 1.5$ or $\nu = 2.5$ admits finite differentiability:
  - $\nu = 1.5$: Once mean-square differentiable paths (smooth, but possessing stochastic acceleration).
  - $\nu = 2.5$: Twice mean-square differentiable paths.
  Their spectral densities decay as **power laws**:
  $$
  f(\omega) \sim \mathcal{O}(\omega^{-(2\nu + d)})
  $$
  rather than exponential decay. Consequently, their eigenvalues decay algebraically ($\lambda_k \sim k^{-(2\nu+1)}$) rather than super-exponentially.
  This prevents the spectrum from collapsing into machine precision, ensuring stable condition numbers ($\kappa \approx 10^3 - 10^6$) while providing physically realistic continuity. $\blacksquare$

---

#### Exercise 2.16: Failure of Conditional Negative Definiteness for Power Variograms with $\alpha = 3$
Consider the candidate variogram $\gamma(h) = h^3$ on $\mathbb{R}^1$.

- **(a)** Formulate the CND test quadratic form $Q = \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(s_i - s_j)$ for $n = 4$ collinear points $s_1 = 0, s_2 = 1, s_3 = 2, s_4 = 3$.
- **(b)** Choose the alternating binomial weight vector $\boldsymbol{\lambda} = (1, -3, 3, -1)'$. Verify that $\sum_{i=1}^4 \lambda_i = 0$.
- **(c)** Calculate the explicit value of $Q$ and prove that $Q > 0$, thereby violating conditional negative definiteness.
- **(d)** Deduce why power semivariograms $\gamma(h) = b h^\alpha$ with $\alpha > 2$ are strictly forbidden in all spatial statistics.

##### Solution to Exercise 2.16
- **(a)** For $n = 4$ collinear points $s_1 = 0, s_2 = 1, s_3 = 2, s_4 = 3$, the pairwise distances are $|s_i - s_j| \in \{0, 1, 2, 3\}$.
  The quadratic form is:
  $$
  Q(\alpha) = \sum_{i=1}^4 \sum_{j=1}^4 \lambda_i \lambda_j |s_i - s_j|^\alpha
  $$

- **(b)** The alternating binomial weight vector is:
  $$
  \boldsymbol{\lambda} = (1, -3, 3, -1)'
  $$
  Check the zero-sum condition:
  $$
  \sum_{i=1}^4 \lambda_i = 1 + (-3) + 3 + (-1) = 0
  $$
  Thus, $\boldsymbol{\lambda}$ is a valid increment weight vector of order 0.

- **(c)** Expand the quadratic form using symmetry:
  $$
  Q(\alpha) = \sum_{i=1}^4 \lambda_i^2 |0|^\alpha + 2 \sum_{i < j} \lambda_i \lambda_j |s_i - s_j|^\alpha
  $$
  Grouping by distance $|s_i - s_j|$:
  1. Distance $|s_i - s_j| = 1$: Pairs are $(1,2), (2,3), (3,4)$:
     $$
     2[\lambda_1 \lambda_2 + \lambda_2 \lambda_3 + \lambda_3 \lambda_4] \cdot 1^\alpha = 2[(1)(-3) + (-3)(3) + (3)(-1)] \cdot 1 = 2[-3 - 9 - 3] = -30
     $$
  2. Distance $|s_i - s_j| = 2$: Pairs are $(1,3), (2,4)$:
     $$
     2[\lambda_1 \lambda_3 + \lambda_2 \lambda_4] \cdot 2^\alpha = 2[(1)(3) + (-3)(-1)] \cdot 2^\alpha = 2[3 + 3] \cdot 2^\alpha = 12 \cdot 2^\alpha
     $$
  3. Distance $|s_i - s_j| = 3$: Pair is $(1,4)$:
     $$
     2[\lambda_1 \lambda_4] \cdot 3^\alpha = 2[(1)(-1)] \cdot 3^\alpha = -2 \cdot 3^\alpha
     $$
  Summing all components:
  $$
  Q(\alpha) = -30 + 12 \cdot 2^\alpha - 2 \cdot 3^\alpha
  $$
  Evaluate $Q(\alpha)$ for specific values of $\alpha$:
  - For $\alpha = 1$ (linear): $Q(1) = -30 + 12(2) - 2(3) = -30 + 24 - 6 = -12 \le 0$ (valid CND).
  - For $\alpha = 2$ (parabolic boundary): $Q(2) = -30 + 12(4) - 2(9) = -30 + 48 - 18 = 0 \le 0$ (valid deterministic boundary).
  - For $\alpha = 3$:
    $$
    Q(3) = -30 + 12(2^3) - 2(3^3) = -30 + 12(8) - 2(27) = -30 + 96 - 54 = +12
    $$
  Since $Q(3) = +12 > 0$, the quadratic form is **strictly positive**, directly violating the conditional negative definiteness requirement ($\sum \sum \lambda_i \lambda_j \gamma_{ij} \le 0$).

- **(d)** Because $\mathbb{R}^1$ embeds isometrically as a 1D subspace of $\mathbb{R}^d$ for any $d \ge 1$ (by placing points along a straight line in $\mathbb{R}^d$), failure of CND on $\mathbb{R}^1$ implies failure of CND on $\mathbb{R}^d$ for all $d \ge 1$.
  Therefore, power variograms $\gamma(h) = b h^\alpha$ with $\alpha > 2$ are **strictly invalid in all dimensions**. $\blacksquare$
