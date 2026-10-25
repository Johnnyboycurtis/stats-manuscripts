# Chapter 13 Companion Exercises: Non-Stationary Covariances and Scalable Low-Rank Kriging

This study guide and exercise workbook accompanies [Chapter 13: Non-Stationary Covariance Architectures & Scalable Low-Rank Kriging](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch13_nonstationary_covariance.md). Synthesizing Maria L. Rizzo's three-beat pedagogical rhythm (*Statistical Computing with R*) with Oliver Schabenberger and Carol A. Gotway's semantic linear-model discipline (*Statistical Methods for Spatial Data Analysis*), these exercises bridge classical linear models and non-stationary, scalable geostatistics across five comprehensive tiers:

1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 13.1–13.4)
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 13.5–13.8)
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 13.9–13.11)
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 13.12–13.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 13.14–13.16)

Every exercise includes an unabridged, step-by-step mathematical derivation, analytical proof, or dependency-free computational solution.

---

## Tier 0: Classical-to-Spatial Warm-Ups

### Exercise 13.1: From Classical Heteroskedasticity (WLS) to Spatial Non-Stationarity
In classical linear regression, the homoskedastic model $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$ assumes $\operatorname{Var}(\boldsymbol{\epsilon}) = \sigma^2 \mathbf{I}_n$. When observation variances differ across units, $\operatorname{Var}(\boldsymbol{\epsilon}) = \boldsymbol{\Omega} = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)$, and Ordinary Least Squares is replaced by Weighted Least Squares (WLS):
$$
\widehat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}'\boldsymbol{\Omega}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Omega}^{-1}\mathbf{Y}
$$

Now consider a continuous spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$.
Suppose an analyst attempts to model non-stationarity by simply attaching location-dependent standard deviations $\sigma(\mathbf{s})$ to a standard stationary correlation function $\rho(\mathbf{h}) = \rho(\|\mathbf{s}_i - \mathbf{s}_j\|)$:
$$
C(\mathbf{s}_i, \mathbf{s}_j) = \sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j) \rho(\|\mathbf{s}_i - \mathbf{s}_j\|)
$$

**(a)** Prove that $C(\mathbf{s}_i, \mathbf{s}_j)$ defined above is always a valid positive semi-definite covariance function if $\rho(\cdot)$ is a valid stationary correlation function.

**(b)** Explain why this simple heteroskedastic model fails to capture **structural non-stationarity**: what happens to the spatial correlation length and the direction of geometric anisotropy as we move from location $\mathbf{s}_i$ to location $\mathbf{s}_j$?

**(c)** Contrast this with the Paciorek-Schervish framework, explaining why modifying the metric tensor $\boldsymbol{\Sigma}(\mathbf{s})$ inside the correlation kernel requires a determinant prefactor to preserve positive-definiteness.

##### Solution to Exercise 13.1
**(a)** Let $\mathbf{R} = [\rho(\|\mathbf{s}_i - \mathbf{s}_j\|)]_{i,j=1}^n$ be the $n \times n$ correlation matrix. Since $\rho(\cdot)$ is a valid stationary correlation function, $\mathbf{R}$ is positive semi-definite: for any vector $\mathbf{a} \in \mathbb{R}^n$, $\mathbf{a}'\mathbf{R}\mathbf{a} \ge 0$.
The proposed covariance matrix is:
$$
\mathbf{V} = [C(\mathbf{s}_i, \mathbf{s}_j)] = [\sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j) \rho(\|\mathbf{s}_i - \mathbf{s}_j\|)] = \mathbf{D}_\sigma \mathbf{R} \mathbf{D}_\sigma
$$
where $\mathbf{D}_\sigma = \operatorname{diag}(\sigma(\mathbf{s}_1), \dots, \sigma(\mathbf{s}_n))$.
For any arbitrary non-zero vector $\mathbf{a} \in \mathbb{R}^n$:
$$
\mathbf{a}'\mathbf{V}\mathbf{a} = \mathbf{a}'(\mathbf{D}_\sigma \mathbf{R} \mathbf{D}_\sigma)\mathbf{a} = (\mathbf{D}_\sigma \mathbf{a})' \mathbf{R} (\mathbf{D}_\sigma \mathbf{a})
$$
Let $\mathbf{b} = \mathbf{D}_\sigma \mathbf{a} \in \mathbb{R}^n$. Because $\mathbf{R}$ is positive semi-definite:
$$
\mathbf{b}'\mathbf{R}\mathbf{b} \ge 0 \implies \mathbf{a}'\mathbf{V}\mathbf{a} \ge 0
$$
Thus, $\mathbf{V}$ is guaranteed to be positive semi-definite for any positive field $\sigma(\mathbf{s}) > 0$. $\blacksquare$

**(b)** While the simple scaling $\sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j)\rho(\|\mathbf{s}_i - \mathbf{s}_j\|)$ permits marginal variance heteroskedasticity, its underlying correlation structure remains strictly stationary and isotropic:
$$
\operatorname{Corr}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = \frac{C(\mathbf{s}_i, \mathbf{s}_j)}{\sigma(\mathbf{s}_i)\sigma(\mathbf{s}_j)} = \rho(\|\mathbf{s}_i - \mathbf{s}_j\|)
$$
Under this model:
- The correlation between two points separated by distance $h$ is identical everywhere across the continent: 10 km in the Rocky Mountains produces the exact same correlation as 10 km over the Great Plains.
- Iso-correlation contours remain perfect circles of constant radius everywhere; the model cannot accommodate local elongation or rotation of continuity.
Thus, simple variance scaling solves marginal heteroskedasticity, but leaves the spatial correlation geometry completely frozen in stationarity.

**(c)** If one attempts to make the correlation range location-dependent by naively writing $\rho\left(\frac{\|\mathbf{s}_i - \mathbf{s}_j\|}{\frac{1}{2}(\phi(\mathbf{s}_i) + \phi(\mathbf{s}_j))}\right)$, the resulting matrix $\mathbf{R}$ is **not positive-definite** in general; it can produce negative eigenvalues, violating probability axioms.
Paciorek and Schervish proved that when local metric tensors $\boldsymbol{\Sigma}(\mathbf{s})$ vary across coordinates, the determinant prefactor $\frac{|\boldsymbol{\Sigma}_i|^{1/4} |\boldsymbol{\Sigma}_j|^{1/4}}{|(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j)/2|^{1/2}}$ exactly compensates for the geometric volume mismatch between differing local ellipses, preserving positive-definiteness across all spatial dimensions.

---

### Exercise 13.2: Geometric Anisotropy and the Local Metric Tensor $\boldsymbol{\Sigma}(\mathbf{s})$
In Chapter 2, geometric anisotropy was defined through a coordinate transformation matrix $\mathbf{A}$:
$$
\mathbf{h}^* = \mathbf{A}\mathbf{h}, \quad C(\mathbf{h}) = C_0(\|\mathbf{A}\mathbf{h}\|) = C_0\left(\sqrt{\mathbf{h}'\boldsymbol{\Sigma}^{-1}\mathbf{h}}\right)
$$
where $\boldsymbol{\Sigma}^{-1} = \mathbf{A}'\mathbf{A}$.

In $\mathbb{R}^2$, let the local ellipse of spatial correlation at coordinate $\mathbf{s}$ have:
1. Major correlation range $\phi_1(\mathbf{s})$.
2. Minor correlation range $\phi_2(\mathbf{s}) \le \phi_1(\mathbf{s})$ (anisotropy ratio $\psi(\mathbf{s}) = \phi_1(\mathbf{s}) / \phi_2(\mathbf{s}) \ge 1$).
3. Counter-clockwise rotation angle $\theta(\mathbf{s})$ of the major axis from the horizontal $x$-axis.

**(a)** Express the local metric tensor $\boldsymbol{\Sigma}(\mathbf{s})$ in terms of the rotation matrix $\mathbf{R}(\theta)$ and diagonal range matrix $\boldsymbol{\Lambda} = \operatorname{diag}(\phi_1^2, \phi_2^2)$.

**(b)** Expand the matrix product explicitly and write down the entries of $\boldsymbol{\Sigma}(\mathbf{s}) = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{12} & \Sigma_{22} \end{pmatrix}$ as functions of $\phi_1, \phi_2, \theta$.

**(c)** Compute the determinant $\det(\boldsymbol{\Sigma}(\mathbf{s}))$ and verify that it is invariant to the rotation angle $\theta(\mathbf{s})$.

##### Solution to Exercise 13.2
**(a)** The coordinate rotation matrix by angle $\theta$ is:
$$
\mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$
In the local principal coordinate system aligned with the ellipse, the squared correlation lengths are $\phi_1^2$ along the major axis and $\phi_2^2$ along the minor axis:
$$
\boldsymbol{\Lambda} = \begin{pmatrix} \phi_1^2 & 0 \\ 0 & \phi_2^2 \end{pmatrix}
$$
Rotating this diagonal structure into geographic coordinates:
$$
\boldsymbol{\Sigma}(\mathbf{s}) = \mathbf{R}(\theta) \boldsymbol{\Lambda} \mathbf{R}(\theta)'
$$

**(b)** Carry out the matrix multiplication:
$$
\begin{aligned}
\boldsymbol{\Sigma}(\mathbf{s}) &= \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \phi_1^2 & 0 \\ 0 & \phi_2^2 \end{pmatrix} \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \\
&= \begin{pmatrix} \phi_1^2 \cos\theta & -\phi_2^2 \sin\theta \\ \phi_1^2 \sin\theta & \phi_2^2 \cos\theta \end{pmatrix} \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \\
&= \begin{pmatrix} \phi_1^2 \cos^2\theta + \phi_2^2 \sin^2\theta & (\phi_1^2 - \phi_2^2)\cos\theta\sin\theta \\ (\phi_1^2 - \phi_2^2)\cos\theta\sin\theta & \phi_1^2 \sin^2\theta + \phi_2^2 \cos^2\theta \end{pmatrix}
\end{aligned}
$$
Using double-angle trigonometric identities:
- $\Sigma_{11} = \frac{\phi_1^2 + \phi_2^2}{2} + \frac{\phi_1^2 - \phi_2^2}{2}\cos(2\theta)$
- $\Sigma_{22} = \frac{\phi_1^2 + \phi_2^2}{2} - \frac{\phi_1^2 - \phi_2^2}{2}\cos(2\theta)$
- $\Sigma_{12} = \Sigma_{21} = \frac{\phi_1^2 - \phi_2^2}{2}\sin(2\theta)$

**(c)** Because $\mathbf{R}(\theta)$ is an orthogonal rotation matrix, $\det(\mathbf{R}(\theta)) = \cos^2\theta - (-\sin^2\theta) = 1$.
By the multiplicative property of determinants:
$$
\det(\boldsymbol{\Sigma}(\mathbf{s})) = \det(\mathbf{R}(\theta)) \det(\boldsymbol{\Lambda}) \det(\mathbf{R}(\theta)') = 1 \cdot (\phi_1^2 \phi_2^2) \cdot 1 = \phi_1^2 \phi_2^2
$$
The determinant represents the squared area of the correlation ellipse (proportional to its spatial correlation volume) and is strictly invariant to spatial orientation $\theta$.

---

### Exercise 13.3: Kernel Convolution of 1D Gaussian White Noise
Consider the 1D process convolution on the real line $\mathbb{R}$:
$$
Z(s) = \int_{-\infty}^\infty k_s(u) W(\mathrm{d}u)
$$
where $W(\cdot)$ is standard 1D Gaussian white noise, and $k_s(u)$ is a Gaussian smoothing kernel with location-dependent bandwidth $b(s) > 0$:
$$
k_s(u) = \frac{1}{\sqrt{2\pi b(s)^2}} \exp\left(-\frac{(u - s)^2}{2 b(s)^2}\right)
$$

**(a)** Derive the marginal variance $\operatorname{Var}(Z(s))$ as a function of the local bandwidth $b(s)$. Show that narrower kernels inflate local variance.

**(b)** Standardize the kernel to have constant unit variance $\operatorname{Var}(Z(s)) \equiv 1$ everywhere. Find the required normalization prefactor $A(s)$.

**(c)** Let $b(s) = b$ be constant (stationary case). Compute the closed-form covariance between $Z(s)$ and $Z(s + h)$:
$$
C(h) = \int_{-\infty}^\infty k_s(u) k_{s+h}(u) \, \mathrm{d}u
$$
Identify the resulting correlation family and its effective spatial range.

##### Solution to Exercise 13.3
**(a)** Using the white noise integral covariance property:
$$
\operatorname{Var}(Z(s)) = \int_{-\infty}^\infty [k_s(u)]^2 \, \mathrm{d}u = \int_{-\infty}^\infty \frac{1}{2\pi b(s)^2} \exp\left(-\frac{(u - s)^2}{b(s)^2}\right) \mathrm{d}u
$$
Substitute $w = \frac{u - s}{b(s)}$, so $\mathrm{d}u = b(s)\,\mathrm{d}w$:
$$
\operatorname{Var}(Z(s)) = \frac{1}{2\pi b(s)^2} \int_{-\infty}^\infty e^{-w^2} b(s) \, \mathrm{d}w = \frac{b(s)}{2\pi b(s)^2} \sqrt{\pi} = \frac{\sqrt{\pi}}{2\pi b(s)} = \frac{1}{2\sqrt{\pi} b(s)}
$$
Because $\operatorname{Var}(Z(s)) \propto 1 / b(s)$, as the kernel narrows ($b(s) \to 0$), the variance diverges to infinity.

**(b)** To enforce $\operatorname{Var}(Z(s)) \equiv 1$, let $\tilde{k}_s(u) = A(s) k_s(u)$. Then:
$$
\operatorname{Var}(Z(s)) = [A(s)]^2 \left(\frac{1}{2\sqrt{\pi} b(s)}\right) = 1 \implies A(s) = (2\sqrt{\pi} b(s))^{1/2} = \pi^{1/4} \sqrt{2 b(s)}
$$
The unit-variance normalized kernel is:
$$
\tilde{k}_s(u) = \frac{\pi^{1/4} \sqrt{2 b(s)}}{\sqrt{2\pi b(s)^2}} \exp\left(-\frac{(u - s)^2}{2b(s)^2}\right) = \frac{1}{\pi^{1/4} \sqrt{b(s)}} \exp\left(-\frac{(u - s)^2}{2b(s)^2}\right)
$$

**(c)** For constant bandwidth $b(s) \equiv b$, using the unit-variance kernel $\tilde{k}$:
$$
C(h) = \int_{-\infty}^\infty \tilde{k}_s(u) \tilde{k}_{s+h}(u) \, \mathrm{d}u = \frac{1}{\sqrt{\pi} b} \int_{-\infty}^\infty \exp\left(-\frac{(u - s)^2}{2b^2} - \frac{(u - (s + h))^2}{2b^2}\right) \mathrm{d}u
$$
Without loss of generality, shift coordinates so $s = 0$. The exponent is:
$$
-\frac{u^2 + (u - h)^2}{2b^2} = -\frac{2u^2 - 2uh + h^2}{2b^2} = -\frac{2(u - h/2)^2 + h^2/2}{2b^2} = -\frac{(u - h/2)^2}{b^2} - \frac{h^2}{4b^2}
$$
Factor out the constant term:
$$
C(h) = \frac{1}{\sqrt{\pi} b} \exp\left(-\frac{h^2}{4b^2}\right) \int_{-\infty}^\infty \exp\left(-\frac{(u - h/2)^2}{b^2}\right) \mathrm{d}u
$$
The integral evaluates to $\sqrt{\pi} b$:
$$
C(h) = \frac{1}{\sqrt{\pi} b} \exp\left(-\frac{h^2}{4b^2}\right) (\sqrt{\pi} b) = \exp\left(-\frac{h^2}{4b^2}\right) = \exp\left(-\frac{h^2}{2(b\sqrt{2})^2}\right)
$$
This is the **Gaussian (squared exponential) covariance kernel** with correlation range parameter $\phi = b\sqrt{2}$.
Convolving two identical Gaussian kernels with variance $b^2$ produces a Gaussian covariance with variance $2b^2$.

---

### Exercise 13.4: The Big-$n$ Computational Bottleneck: FLOPs and Memory
In exact geostatistical kriging, an analyst must compute the Cholesky factorization $\mathbf{V} = \mathbf{L}\mathbf{L}'$ and solve $\mathbf{L}\mathbf{y} = \mathbf{c}$, $\mathbf{L}'\boldsymbol{\lambda} = \mathbf{y}$.
The number of floating-point operations (FLOPs) required to compute the Cholesky decomposition of an $n \times n$ symmetric positive-definite matrix is:
$$
\text{FLOPs}_{\text{Chol}} = \frac{1}{3} n^3
$$
and storing $\mathbf{V}$ in double-precision float (8 bytes per entry) requires:
$$
\text{Memory} = 8 \times n^2 \text{ bytes}
$$

**(a)** Calculate the exact FLOP count and memory requirement (in GB or TB) for:
1. $n = 1,000$ sample locations.
2. $n = 10,000$ sample locations.
3. $n = 100,000$ sample locations.
4. $n = 1,000,000$ sample locations.

**(b)** Suppose a high-end workstation performs sustained double-precision computations at $200\text{ GFLOPs} = 2 \times 10^{11}\text{ FLOPs/second}$.
Compute the execution time required to factorize $\mathbf{V}$ for each sample size.

**(c)** Contrast this with Fixed Rank Kriging (FRK) using $r = 100$ basis functions. Compute the FLOPs for FRK at $n = 100,000$ and $n = 1,000,000$, and find the speedup factor.

##### Solution to Exercise 13.4
**(a) and (b)**
1. *For $n = 1,000$:*
   - FLOPs: $\frac{1}{3}(10^3)^3 = \frac{1}{3} \times 10^9 \approx 3.33 \times 10^8$ FLOPs.
   - Memory: $8 \times 10^6 \text{ bytes} = 8.0 \text{ MB}$.
   - Time: $\frac{3.33 \times 10^8}{2 \times 10^{11}} \approx 0.0017 \text{ seconds}$ (1.7 ms).
2. *For $n = 10,000$:*
   - FLOPs: $\frac{1}{3}(10^4)^3 = \frac{1}{3} \times 10^{12} \approx 3.33 \times 10^{11}$ FLOPs.
   - Memory: $8 \times 10^8 \text{ bytes} = 800 \text{ MB} = 0.8 \text{ GB}$.
   - Time: $\frac{3.33 \times 10^{11}}{2 \times 10^{11}} \approx 1.67 \text{ seconds}$.
3. *For $n = 100,000$:*
   - FLOPs: $\frac{1}{3}(10^5)^3 = \frac{1}{3} \times 10^{15} \approx 3.33 \times 10^{14}$ FLOPs (333 TeraFLOPs).
   - Memory: $8 \times 10^{10} \text{ bytes} = 80.0 \text{ GB}$ (exceeds typical workstation RAM).
   - Time: $\frac{3.33 \times 10^{14}}{2 \times 10^{11}} \approx 1,667 \text{ seconds} \approx 27.8 \text{ minutes}$.
4. *For $n = 1,000,000$:*
   - FLOPs: $\frac{1}{3}(10^6)^3 = \frac{1}{3} \times 10^{18} \approx 3.33 \times 10^{17}$ FLOPs (333 PetaFLOPs).
   - Memory: $8 \times 10^{12} \text{ bytes} = 8.0 \text{ Terabytes}$.
   - Time: $\frac{3.33 \times 10^{17}}{2 \times 10^{11}} \approx 1.67 \times 10^6 \text{ seconds} \approx 19.3 \text{ days}$ (and impossible due to 8 TB RAM requirement).

**(c)** In Fixed Rank Kriging (FRK) with $r = 100$ basis functions:
- Forming Gram matrix $\mathbf{S}'\mathbf{S}$: $2 n r^2$ FLOPs.
- Inverting the $r \times r$ matrix: $\frac{1}{3} r^3$ FLOPs.
- Total FLOPs: $2 n r^2 + \frac{1}{3}r^3$.
With $r = 100$, $r^2 = 10,000$, $\frac{1}{3}r^3 = 333,333$.
- *At $n = 100,000$:*
  $$\text{FLOPs}_{\text{FRK}} = 2(100,000)(10,000) + 333,333 \approx 2 \times 10^9 \text{ FLOPs}$$
  $$\text{Time}_{\text{FRK}} = \frac{2 \times 10^9}{2 \times 10^{11}} = 0.010 \text{ seconds}$$
  $$\text{Speedup} = \frac{1,667 \text{ s}}{0.010 \text{ s}} \approx 166,700\times$$
- *At $n = 1,000,000$:*
  $$\text{FLOPs}_{\text{FRK}} = 2(1,000,000)(10,000) \approx 2 \times 10^{10} \text{ FLOPs}$$
  $$\text{Time}_{\text{FRK}} = \frac{2 \times 10^{10}}{2 \times 10^{11}} = 0.10 \text{ seconds}$$
  $$\text{Memory} = 8 \times n \times r = 8 \times 10^6 \times 100 = 800 \text{ MB} \quad (\text{fits easily in RAM})$$
FRK converts a 19-day, 8-Terabyte calculation into a 0.10-second, 800-MB routine.

---

## Tier I: Core Theoretical Re-Derivations

### Exercise 13.5: Positive-Definiteness of Kernel Convolutions (Higdon's Theorem)
Let $D \subset \mathbb{R}^d$ be a spatial domain. For each location $\mathbf{s} \in D$, let $k_{\mathbf{s}}: \mathbb{R}^d \to \mathbb{R}$ be a square-integrable smoothing kernel:
$$
\int_{\mathbb{R}^d} [k_{\mathbf{s}}(\mathbf{u})]^2 \, \mathrm{d}\mathbf{u} < \infty
$$
Define the bivariate kernel on $D \times D$:
$$
C(\mathbf{s}_i, \mathbf{s}_j) = \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, \mathrm{d}\mathbf{u}
$$

**(a)** State the definition of a positive semi-definite kernel on $D \times D$.

**(b)** Give an unabridged proof that $C(\mathbf{s}_i, \mathbf{s}_j)$ is positive semi-definite for any arbitrary spatial collection of kernels $\{k_{\mathbf{s}} : \mathbf{s} \in D\}$.

**(c)** Prove that if the family of functions $\{k_{\mathbf{s}_1}, \dots, k_{\mathbf{s}_n}\}$ is linearly independent in $L^2(\mathbb{R}^d)$, the $n \times n$ covariance matrix $\mathbf{V} = [C(\mathbf{s}_i, \mathbf{s}_j)]$ is strictly positive-definite.

##### Solution to Exercise 13.5
**(a)** A bivariate function $C: D \times D \to \mathbb{R}$ is positive semi-definite if:
1. It is symmetric: $C(\mathbf{s}_i, \mathbf{s}_j) = C(\mathbf{s}_j, \mathbf{s}_i)$ for all $\mathbf{s}_i, \mathbf{s}_j \in D$.
2. For any positive integer $n \ge 1$, any choice of $n$ locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset D$, and any real vector $\mathbf{a} = (a_1, \dots, a_n)' \in \mathbb{R}^n$:
$$
\sum_{i=1}^n \sum_{j=1}^n a_i a_j C(\mathbf{s}_i, \mathbf{s}_j) \ge 0
$$

**(b)** **Proof:**
First, symmetry:
$$
C(\mathbf{s}_i, \mathbf{s}_j) = \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, \mathrm{d}\mathbf{u} = \int_{\mathbb{R}^d} k_{\mathbf{s}_j}(\mathbf{u}) k_{\mathbf{s}_i}(\mathbf{u}) \, \mathrm{d}\mathbf{u} = C(\mathbf{s}_j, \mathbf{s}_i)
$$
Second, evaluate the quadratic form for arbitrary coefficients $a_1, \dots, a_n$:
$$
\begin{aligned}
\mathbf{a}'\mathbf{V}\mathbf{a} &= \sum_{i=1}^n \sum_{j=1}^n a_i a_j C(\mathbf{s}_i, \mathbf{s}_j) \\
&= \sum_{i=1}^n \sum_{j=1}^n a_i a_j \int_{\mathbb{R}^d} k_{\mathbf{s}_i}(\mathbf{u}) k_{\mathbf{s}_j}(\mathbf{u}) \, \mathrm{d}\mathbf{u} \\
&= \int_{\mathbb{R}^d} \left(\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u})\right) \left(\sum_{j=1}^n a_j k_{\mathbf{s}_j}(\mathbf{u})\right) \mathrm{d}\mathbf{u} \\
&= \int_{\mathbb{R}^d} \left(\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u})\right)^2 \mathrm{d}\mathbf{u}
\end{aligned}
$$
Because the integrand is the square of a real-valued function, $\left(\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u})\right)^2 \ge 0$ for all $\mathbf{u} \in \mathbb{R}^d$.
The integral of a non-negative function is non-negative:
$$
\int_{\mathbb{R}^d} \left(\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u})\right)^2 \mathrm{d}\mathbf{u} \ge 0
$$
Thus, $\mathbf{a}'\mathbf{V}\mathbf{a} \ge 0$ for all $\mathbf{a} \in \mathbb{R}^n$. $C(\cdot, \cdot)$ is positive semi-definite. $\blacksquare$

**(c)** For strict positive-definiteness, we must show that $\mathbf{a}'\mathbf{V}\mathbf{a} = 0 \implies \mathbf{a} = \mathbf{0}$.
From part (b):
$$
\mathbf{a}'\mathbf{V}\mathbf{a} = \int_{\mathbb{R}^d} \left(\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u})\right)^2 \mathrm{d}\mathbf{u}
$$
Because the integrand is continuous and non-negative, the integral equals zero if and only if the function vanishes identically:
$$
\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u}) = 0 \quad \text{for all } \mathbf{u} \in \mathbb{R}^d
$$
By assumption, the continuous functions $\{k_{\mathbf{s}_1}, \dots, k_{\mathbf{s}_n}\}$ are linearly independent.
By definition of linear independence, $\sum_{i=1}^n a_i k_{\mathbf{s}_i}(\mathbf{u}) = 0$ implies $a_1 = a_2 = \dots = a_n = 0$, i.e., $\mathbf{a} = \mathbf{0}$.
Therefore, $\mathbf{a}'\mathbf{V}\mathbf{a} > 0$ for all $\mathbf{a} \ne \mathbf{0}$, proving $\mathbf{V}$ is strictly positive-definite. $\blacksquare$

---

### Exercise 13.6: Unabridged Proof of the Paciorek-Schervish Theorem
Let $\boldsymbol{\Sigma}_1, \boldsymbol{\Sigma}_2$ be two $d \times d$ symmetric positive-definite covariance matrices.
Let Gaussian kernels on $\mathbb{R}^d$ be defined by:
$$
k_i(\mathbf{u}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}_i|^{1/4}} \exp\left(-\frac{1}{2}(\mathbf{u} - \mathbf{s}_i)' \boldsymbol{\Sigma}_i^{-1} (\mathbf{u} - \mathbf{s}_i)\right), \quad i \in \{1, 2\}
$$

**(a)** State the convolution integral:
$$
C(\mathbf{s}_1, \mathbf{s}_2) = \int_{\mathbb{R}^d} k_1(\mathbf{u}) k_2(\mathbf{u}) \, \mathrm{d}\mathbf{u}
$$

**(b)** Complete the square in the exponent of the integrand.
Show that:
$$
(\mathbf{u} - \mathbf{s}_1)'\boldsymbol{\Sigma}_1^{-1}(\mathbf{u} - \mathbf{s}_1) + (\mathbf{u} - \mathbf{s}_2)'\boldsymbol{\Sigma}_2^{-1}(\mathbf{u} - \mathbf{s}_2) = (\mathbf{u} - \boldsymbol{\mu})'\mathbf{A}(\mathbf{u} - \boldsymbol{\mu}) + Q_{12}
$$
where $\mathbf{A} = \boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1}$, and identify the exact expression for $Q_{12}$.

**(c)** Integrate out $\mathbf{u}$ using the multivariate Gaussian integral $\int_{\mathbb{R}^d} \exp(-\frac{1}{2}\mathbf{w}'\mathbf{A}\mathbf{w})\,\mathrm{d}\mathbf{w} = (2\pi)^{d/2}|\mathbf{A}|^{-1/2}$.

**(d)** Using matrix determinant identities, prove that:
$$
|\mathbf{A}|^{-1/2} |\boldsymbol{\Sigma}_1|^{-1/4} |\boldsymbol{\Sigma}_2|^{-1/4} = \frac{|\boldsymbol{\Sigma}_1|^{1/4}|\boldsymbol{\Sigma}_2|^{1/4}}{\left|\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right|^{1/2}} \cdot \left(\frac{1}{2}\right)^{d/2}
$$
and conclude the full Paciorek-Schervish formula for the Gaussian covariance kernel:
$$
C(\mathbf{s}_1, \mathbf{s}_2) = \frac{|\boldsymbol{\Sigma}_1|^{1/4}|\boldsymbol{\Sigma}_2|^{1/4}}{\left|\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{s}_1 - \mathbf{s}_2)'\left(\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right)^{-1}(\mathbf{s}_1 - \mathbf{s}_2)\right)
$$

##### Solution to Exercise 13.6
**(a)** The convolution integral is:
$$
C(\mathbf{s}_1, \mathbf{s}_2) = \frac{1}{(2\pi)^d |\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}} \int_{\mathbb{R}^d} \exp\left(-\frac{1}{2}\left[(\mathbf{u} - \mathbf{s}_1)'\boldsymbol{\Sigma}_1^{-1}(\mathbf{u} - \mathbf{s}_1) + (\mathbf{u} - \mathbf{s}_2)'\boldsymbol{\Sigma}_2^{-1}(\mathbf{u} - \mathbf{s}_2)\right]\right) \mathrm{d}\mathbf{u}
$$

**(b)** Let $\mathbf{P}_1 = \boldsymbol{\Sigma}_1^{-1}$ and $\mathbf{P}_2 = \boldsymbol{\Sigma}_2^{-1}$. Expand the quadratic terms:
$$
\begin{aligned}
S &= (\mathbf{u} - \mathbf{s}_1)'\mathbf{P}_1(\mathbf{u} - \mathbf{s}_1) + (\mathbf{u} - \mathbf{s}_2)'\mathbf{P}_2(\mathbf{u} - \mathbf{s}_2) \\
&= \mathbf{u}'(\mathbf{P}_1 + \mathbf{P}_2)\mathbf{u} - 2\mathbf{u}'(\mathbf{P}_1\mathbf{s}_1 + \mathbf{P}_2\mathbf{s}_2) + \mathbf{s}_1'\mathbf{P}_1\mathbf{s}_1 + \mathbf{s}_2'\mathbf{P}_2\mathbf{s}_2
\end{aligned}
$$
Let $\mathbf{A} = \mathbf{P}_1 + \mathbf{P}_2 = \boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1}$.
Completing the square in $\mathbf{u}$:
$$
\mathbf{u}'\mathbf{A}\mathbf{u} - 2\mathbf{u}'\mathbf{b} = (\mathbf{u} - \mathbf{A}^{-1}\mathbf{b})'\mathbf{A}(\mathbf{u} - \mathbf{A}^{-1}\mathbf{b}) - \mathbf{b}'\mathbf{A}^{-1}\mathbf{b}
$$
where $\mathbf{b} = \mathbf{P}_1\mathbf{s}_1 + \mathbf{P}_2\mathbf{s}_2$.
The remaining constant term independent of $\mathbf{u}$ is:
$$
Q_{12} = \mathbf{s}_1'\mathbf{P}_1\mathbf{s}_1 + \mathbf{s}_2'\mathbf{P}_2\mathbf{s}_2 - (\mathbf{P}_1\mathbf{s}_1 + \mathbf{P}_2\mathbf{s}_2)'(\mathbf{P}_1 + \mathbf{P}_2)^{-1}(\mathbf{P}_1\mathbf{s}_1 + \mathbf{P}_2\mathbf{s}_2)
$$
By standard matrix algebra (or viewing this as the marginal variance of the difference of two Gaussian vectors):
$$
Q_{12} = (\mathbf{s}_1 - \mathbf{s}_2)'(\mathbf{P}_1^{-1} + \mathbf{P}_2^{-1})^{-1}(\mathbf{s}_1 - \mathbf{s}_2) = (\mathbf{s}_1 - \mathbf{s}_2)'(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)^{-1}(\mathbf{s}_1 - \mathbf{s}_2)
$$

**(c)** Substitute the completed square into the integral:
$$
\int_{\mathbb{R}^d} \exp\left(-\frac{1}{2}(\mathbf{u} - \boldsymbol{\mu})'\mathbf{A}(\mathbf{u} - \boldsymbol{\mu})\right) \mathrm{d}\mathbf{u} = (2\pi)^{d/2} |\mathbf{A}|^{-1/2}
$$
Therefore:
$$
C(\mathbf{s}_1, \mathbf{s}_2) = \frac{(2\pi)^{d/2} |\mathbf{A}|^{-1/2}}{(2\pi)^d |\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}} \exp\left(-\frac{1}{2} Q_{12}\right) = \frac{|\mathbf{A}|^{-1/2}}{(2\pi)^{d/2} |\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}} \exp\left(-\frac{1}{2}(\mathbf{s}_1 - \mathbf{s}_2)'(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)^{-1}(\mathbf{s}_1 - \mathbf{s}_2)\right)
$$

**(d)** Evaluate $|\mathbf{A}| = |\boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1}|$:
$$
\boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1} = \boldsymbol{\Sigma}_1^{-1}(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)\boldsymbol{\Sigma}_2^{-1}
$$
Taking determinants:
$$
|\mathbf{A}| = |\boldsymbol{\Sigma}_1^{-1}| |\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2| |\boldsymbol{\Sigma}_2^{-1}| = \frac{|\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1| |\boldsymbol{\Sigma}_2|}
$$
Therefore:
$$
|\mathbf{A}|^{-1/2} = \frac{|\boldsymbol{\Sigma}_1|^{1/2} |\boldsymbol{\Sigma}_2|^{1/2}}{|\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2|^{1/2}}
$$
Divide by $|\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}$:
$$
\frac{|\mathbf{A}|^{-1/2}}{|\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}} = \frac{|\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}}{|\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2|^{1/2}}
$$
Notice that $|\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2| = |2 \cdot \frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}| = 2^d \left|\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right|$.
Taking square roots:
$$
|\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2|^{1/2} = 2^{d/2} \left|\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right|^{1/2}
$$
The factor $2^{d/2}$ cancels with the $(2\pi)^{d/2}$ normalizing constant from the kernel definitions, yielding:
$$
C(\mathbf{s}_1, \mathbf{s}_2) = \frac{|\boldsymbol{\Sigma}_1|^{1/4}|\boldsymbol{\Sigma}_2|^{1/4}}{\left|\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{s}_1 - \mathbf{s}_2)'\left(\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right)^{-1}(\mathbf{s}_1 - \mathbf{s}_2)\right) \quad \blacksquare
$$

---

### Exercise 13.7: Derivation of the Sherman-Morrison-Woodbury Identity for Fixed Rank Kriging
Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be invertible, $\mathbf{C} \in \mathbb{R}^{r \times r}$ be invertible, $\mathbf{U} \in \mathbb{R}^{n \times r}$, and $\mathbf{V} \in \mathbb{R}^{r \times n}$.
The Woodbury matrix identity states:
$$
(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}
$$

**(a)** Verify the identity directly by multiplying $(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})$ by the right-hand side and simplifying to $\mathbf{I}_n$.

**(b)** In Fixed Rank Kriging, the data covariance matrix is:
$$
\boldsymbol{\Sigma} = \sigma_\epsilon^2 \mathbf{I}_n + \mathbf{S}\mathbf{K}\mathbf{S}'
$$
where $\mathbf{S} \in \mathbb{R}^{n \times r}$ ($r \ll n$).
Apply the Woodbury identity with $\mathbf{A} = \sigma_\epsilon^2 \mathbf{I}_n$, $\mathbf{U} = \mathbf{S}$, $\mathbf{C} = \mathbf{K}$, and $\mathbf{V} = \mathbf{S}'$. Show that:
$$
\boldsymbol{\Sigma}^{-1} = \frac{1}{\sigma_\epsilon^2}\mathbf{I}_n - \frac{1}{\sigma_\epsilon^4}\mathbf{S}\left(\mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}\right)^{-1}\mathbf{S}'
$$

**(c)** For an observation vector $\mathbf{Z} \in \mathbb{R}^n$, show how to evaluate the kriging weight vector $\boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}\mathbf{Z}$ without ever instantiating any $n \times n$ matrix in memory.

##### Solution to Exercise 13.7
**(a)** Multiply $(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})$ by the proposed inverse $\mathbf{B} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}\mathbf{A}^{-1}$:
$$
(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})\mathbf{B} = \mathbf{A}\mathbf{B} + \mathbf{U}\mathbf{C}\mathbf{V}\mathbf{B}
$$
1. First term:
$$
\mathbf{A}\mathbf{B} = \mathbf{I}_n - \mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}
$$
2. Second term:
$$
\mathbf{U}\mathbf{C}\mathbf{V}\mathbf{B} = \mathbf{U}\mathbf{C}\mathbf{V}\mathbf{A}^{-1} - \mathbf{U}\mathbf{C}\mathbf{V}\mathbf{A}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}
$$
Factor out $\mathbf{U}\mathbf{C}$ on the left and $(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}\mathbf{A}^{-1}$ on the right:
$$
\begin{aligned}
\mathbf{U}\mathbf{C}\mathbf{V}\mathbf{B} &= \mathbf{U}\mathbf{C}\left[\mathbf{I}_r - \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\right]\mathbf{V}\mathbf{A}^{-1} \\
&= \mathbf{U}\mathbf{C}\left[\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right) - \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right]\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1} \\
&= \mathbf{U}\mathbf{C}\left[\mathbf{C}^{-1}\right]\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1} \\
&= \mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1}
\end{aligned}
$$
Adding the two terms:
$$
(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})\mathbf{B} = \mathbf{I}_n - \mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1} + \mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{A}^{-1} = \mathbf{I}_n \quad \blacksquare
$$

**(b)** Substituting $\mathbf{A} = \sigma_\epsilon^2 \mathbf{I}_n$, so $\mathbf{A}^{-1} = \frac{1}{\sigma_\epsilon^2}\mathbf{I}_n$:
- $\mathbf{A}^{-1}\mathbf{U} = \frac{1}{\sigma_\epsilon^2}\mathbf{S}$.
- $\mathbf{V}\mathbf{A}^{-1} = \frac{1}{\sigma_\epsilon^2}\mathbf{S}'$.
- $\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U} = \mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}$.
Substituting these pieces into the Woodbury identity:
$$
\begin{aligned}
\boldsymbol{\Sigma}^{-1} &= \frac{1}{\sigma_\epsilon^2}\mathbf{I}_n - \left(\frac{1}{\sigma_\epsilon^2}\mathbf{S}\right)\left(\mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}\right)^{-1}\left(\frac{1}{\sigma_\epsilon^2}\mathbf{S}'\right) \\
&= \frac{1}{\sigma_\epsilon^2}\mathbf{I}_n - \frac{1}{\sigma_\epsilon^4}\mathbf{S}\left(\mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}\right)^{-1}\mathbf{S}' \quad \blacksquare
\end{aligned}
$$

**(c)** To evaluate $\boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}\mathbf{Z}$ without creating any $n \times n$ matrix:
1. Compute the $r \times 1$ vector $\mathbf{v}_1 = \mathbf{S}'\mathbf{Z}$ (cost: $\mathcal{O}(n r)$).
2. Compute the $r \times r$ Gram matrix $\mathbf{G} = \mathbf{S}'\mathbf{S}$ (cost: $\mathcal{O}(n r^2)$).
3. Form the small $r \times r$ matrix $\mathbf{M} = \mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{G}$.
4. Solve the $r \times r$ linear system $\mathbf{M}\mathbf{w} = \mathbf{v}_1$ for $\mathbf{w} \in \mathbb{R}^r$ via Cholesky (cost: $\mathcal{O}(r^3)$).
5. Compute $\mathbf{v}_2 = \mathbf{S}\mathbf{w} \in \mathbb{R}^n$ (cost: $\mathcal{O}(n r)$).
6. Assemble $\boldsymbol{\lambda} = \frac{1}{\sigma_\epsilon^2}\mathbf{Z} - \frac{1}{\sigma_\epsilon^4}\mathbf{v}_2$.
At no point is any $n \times n$ matrix instantiated; memory consumption is strictly $\mathcal{O}(n r + r^2)$.

---

### Exercise 13.8: Mathematical Properties of the Multiresolution Bisquare Basis Function
The bisquare basis function centered at $\mathbf{c} \in \mathbb{R}^d$ with radius $R > 0$ is defined as:
$$
S(\mathbf{s}) = \begin{cases}
\left(1 - \frac{\|\mathbf{s} - \mathbf{c}\|^2}{R^2}\right)^2 & \text{if } \|\mathbf{s} - \mathbf{c}\| \le R, \\
0 & \text{if } \|\mathbf{s} - \mathbf{c}\| > R.
\end{cases}
$$

**(a)** Prove that $S(\mathbf{s})$ is continuously differentiable ($C^1$) across the entire domain $\mathbb{R}^d$, including on the boundary sphere $\|\mathbf{s} - \mathbf{c}\| = R$.

**(b)** Show that the second derivative $\nabla^2 S(\mathbf{s})$ is continuous ($C^2$) if and only if evaluated in the interior, and evaluate the jump discontinuity in the second radial derivative at $r = R$.

**(c)** Explain why compact support ($S(\mathbf{s}) = 0$ for $r > R$) guarantees that the basis matrix $\mathbf{S} \in \mathbb{R}^{n \times r}$ is highly sparse, and explain how this sparsity further accelerates the evaluation of $\mathbf{S}'\mathbf{S}$.

##### Solution to Exercise 13.8
**(a)** Let $r = \|\mathbf{s} - \mathbf{c}\|$. For $r < R$:
$$
S(r) = \left(1 - \frac{r^2}{R^2}\right)^2 = 1 - 2\frac{r^2}{R^2} + \frac{r^4}{R^4}
$$
The radial derivative is:
$$
\frac{\mathrm{d}S}{\mathrm{d}r} = 2\left(1 - \frac{r^2}{R^2}\right)\left(-\frac{2r}{R^2}\right) = -\frac{4r}{R^2}\left(1 - \frac{r^2}{R^2}\right)
$$
Taking the limit as $r \to R$ from the inside:
$$
\lim_{r \to R^-} \frac{\mathrm{d}S}{\mathrm{d}r} = -\frac{4R}{R^2}\left(1 - \frac{R^2}{R^2}\right) = -\frac{4}{R}(0) = 0
$$
For $r > R$, $S(r) \equiv 0$, so $\frac{\mathrm{d}S}{\mathrm{d}r} \equiv 0$.
Taking the limit as $r \to R$ from the outside:
$$
\lim_{r \to R^+} \frac{\mathrm{d}S}{\mathrm{d}r} = 0
$$
Because the function value $S(R) = 0$ and the first derivative $\frac{\mathrm{d}S}{\mathrm{d}r}(R) = 0$ match from both sides, $S(\mathbf{s})$ is $C^1$ continuously differentiable everywhere on $\mathbb{R}^d$.

**(b)** Compute the second radial derivative for $r < R$:
$$
\frac{\mathrm{d}^2 S}{\mathrm{d}r^2} = \frac{\mathrm{d}}{\mathrm{d}r}\left[-\frac{4r}{R^2} + \frac{4r^3}{R^4}\right] = -\frac{4}{R^2} + \frac{12r^2}{R^4}
$$
Evaluating the limit as $r \to R^-$:
$$
\lim_{r \to R^-} \frac{\mathrm{d}^2 S}{\mathrm{d}r^2} = -\frac{4}{R^2} + \frac{12R^2}{R^4} = -\frac{4}{R^2} + \frac{12}{R^2} = +\frac{8}{R^2} \ne 0
$$
For $r > R$, $\frac{\mathrm{d}^2 S}{\mathrm{d}r^2} \equiv 0$.
Thus, there is a jump discontinuity in the second radial derivative of magnitude:
$$
\Delta = \frac{8}{R^2} - 0 = \frac{8}{R^2}
$$
$S(\mathbf{s})$ is $C^1$, but not $C^2$. (If a $C^2$ smooth field is required, analysts use the tricube kernel $(1 - r^3/R^3)^3$ or Wendland compactly supported polynomials).

**(c)**
- *Sparsity:* Because each basis function has aperture radius $R_l$, it is strictly zero outside its local neighborhood ball $B(\mathbf{c}_l, R_l)$. In a multiresolution grid, any single observation $\mathbf{s}_i$ is covered by only a small number $k \ll r$ of active basis functions (typically $k \approx 4$ to $12$ out of $r = 500$). Thus, each row of $\mathbf{S}$ contains over $98\%$ exact zeros.
- *Acceleration of $\mathbf{S}'\mathbf{S}$:* When computing $[\mathbf{S}'\mathbf{S}]_{jk} = \sum_{i=1}^n S_{ij} S_{ik}$, the term $S_{ij} S_{ik} \ne 0$ if and only if location $\mathbf{s}_i$ lies in the intersection of the support balls $B(\mathbf{c}_j, R_j) \cap B(\mathbf{c}_k, R_k)$. If two basis centers are separated by $\|\mathbf{c}_j - \mathbf{c}_k\| > R_j + R_k$, their support balls do not overlap, and $[\mathbf{S}'\mathbf{S}]_{jk} \equiv 0$ identically!
Consequently, $\mathbf{S}'\mathbf{S}$ is itself a **sparse banded matrix**, allowing sparse Cholesky factorization in $\mathcal{O}(r)$ time rather than $\mathcal{O}(r^3)$.

---

## Tier II: Pencil-and-Paper Mechanics

### Exercise 13.9: Manual Evaluation of the Paciorek-Schervish Covariance for Two Points
Consider two locations in $\mathbb{R}^2$: $\mathbf{s}_1 = (0, 0)'$ and $\mathbf{s}_2 = (1, 0)'$.
Suppose the local metric tensors at the two points are:
$$
\boldsymbol{\Sigma}_1 = \begin{pmatrix} 1.0 & 0 \\ 0 & 1.0 \end{pmatrix}, \quad \boldsymbol{\Sigma}_2 = \begin{pmatrix} 4.0 & 0 \\ 0 & 1.0 \end{pmatrix}
$$
(Location 1 is isotropic with range 1.0; Location 2 is anisotropic with major range 2.0 along the $x$-axis).
Let the marginal variances be $\sigma^2(\mathbf{s}_1) = 1.0$ and $\sigma^2(\mathbf{s}_2) = 4.0$.

**(a)** Compute the determinants $|\boldsymbol{\Sigma}_1|$ and $|\boldsymbol{\Sigma}_2|$.

**(b)** Compute the average metric tensor $\boldsymbol{\Sigma}_{\text{avg}} = \frac{1}{2}(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)$, its determinant $|\boldsymbol{\Sigma}_{\text{avg}}|$, and its inverse $\boldsymbol{\Sigma}_{\text{avg}}^{-1}$.

**(c)** Compute the determinant prefactor:
$$
P = \frac{|\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}}{|\boldsymbol{\Sigma}_{\text{avg}}|^{1/2}}
$$

**(d)** Compute the squared Mahalanobis distance $Q_{12} = (\mathbf{s}_1 - \mathbf{s}_2)' \boldsymbol{\Sigma}_{\text{avg}}^{-1} (\mathbf{s}_1 - \mathbf{s}_2)$.

**(e)** Under a Gaussian kernel ($\nu \to \infty$), compute the exact non-stationary covariance $C(\mathbf{s}_1, \mathbf{s}_2)$.

##### Solution to Exercise 13.9
**(a)**
- $|\boldsymbol{\Sigma}_1| = (1.0)(1.0) - 0 = 1.0 \implies |\boldsymbol{\Sigma}_1|^{1/4} = 1.0$.
- $|\boldsymbol{\Sigma}_2| = (4.0)(1.0) - 0 = 4.0 \implies |\boldsymbol{\Sigma}_2|^{1/4} = 4^{1/4} = \sqrt{2} \approx 1.414214$.

**(b)** The average metric tensor is:
$$
\boldsymbol{\Sigma}_{\text{avg}} = \frac{1}{2}\left[\begin{pmatrix} 1.0 & 0 \\ 0 & 1.0 \end{pmatrix} + \begin{pmatrix} 4.0 & 0 \\ 0 & 1.0 \end{pmatrix}\right] = \begin{pmatrix} 2.5 & 0 \\ 0 & 1.0 \end{pmatrix}
$$
- Determinant: $|\boldsymbol{\Sigma}_{\text{avg}}| = (2.5)(1.0) - 0 = 2.5$.
- Square root: $|\boldsymbol{\Sigma}_{\text{avg}}|^{1/2} = \sqrt{2.5} \approx 1.581139$.
- Inverse:
$$
\boldsymbol{\Sigma}_{\text{avg}}^{-1} = \begin{pmatrix} 1/2.5 & 0 \\ 0 & 1/1.0 \end{pmatrix} = \begin{pmatrix} 0.40 & 0 \\ 0 & 1.00 \end{pmatrix}
$$

**(c)** The determinant prefactor is:
$$
P = \frac{|\boldsymbol{\Sigma}_1|^{1/4} |\boldsymbol{\Sigma}_2|^{1/4}}{|\boldsymbol{\Sigma}_{\text{avg}}|^{1/2}} = \frac{1.0 \times \sqrt{2}}{\sqrt{2.5}} = \sqrt{\frac{2}{2.5}} = \sqrt{\frac{4}{5}} = \sqrt{0.80} \approx 0.894427
$$

**(d)** The separation vector is:
$$
\mathbf{s}_1 - \mathbf{s}_2 = \begin{pmatrix} 0 \\ 0 \end{pmatrix} - \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix}
$$
The squared Mahalanobis distance is:
$$
Q_{12} = \begin{pmatrix} -1 & 0 \end{pmatrix} \begin{pmatrix} 0.40 & 0 \\ 0 & 1.00 \end{pmatrix} \begin{pmatrix} -1 \\ 0 \end{pmatrix} = (-1)^2 \times 0.40 + 0 = 0.40
$$

**(e)** The marginal standard deviations are $\sigma(\mathbf{s}_1) = \sqrt{1.0} = 1.0$ and $\sigma(\mathbf{s}_2) = \sqrt{4.0} = 2.0$.
The Gaussian correlation term is:
$$
\exp\left(-\frac{1}{2} Q_{12}\right) = \exp(-0.5 \times 0.40) = \exp(-0.20) \approx 0.818731
$$
Multiplying all components together:
$$
\begin{aligned}
C(\mathbf{s}_1, \mathbf{s}_2) &= \sigma(\mathbf{s}_1) \sigma(\mathbf{s}_2) \cdot P \cdot \exp\left(-\frac{1}{2} Q_{12}\right) \\
&= (1.0)(2.0) \times 0.894427 \times 0.818731 \\
&= 2.0 \times 0.732295 \approx 1.4646
\end{aligned}
$$

---

### Exercise 13.10: Metric Multidimensional Scaling (MDS) on a $3 \times 3$ Dissimilarity Matrix
In the Sampson-Guttorp space deformation method, empirical non-stationary semivariances $\widehat{\gamma}_{ij}$ between monitoring stations are inverted into target Euclidean distances $d_{ij}^*$ in deformed $\mathcal{D}$-space.

Suppose three monitoring stations have target deformed distances:
$$
d_{12}^* = 3, \quad d_{13}^* = 4, \quad d_{23}^* = 5
$$
(Notice that $3^2 + 4^2 = 9 + 16 = 25 = 5^2$, forming a classic right triangle).
The squared distance matrix is:
$$
\mathbf{D}^{(2)} = \begin{pmatrix} 0 & 9 & 16 \\ 9 & 0 & 25 \\ 16 & 25 & 0 \end{pmatrix}
$$

**(a)** Compute the classical MDS double-centering projection:
$$
\mathbf{B} = -\frac{1}{2} \mathbf{P} \mathbf{D}^{(2)} \mathbf{P}
$$
where $\mathbf{P} = \mathbf{I}_3 - \frac{1}{3}\mathbf{1}_3\mathbf{1}_3'$.

**(b)** Compute the eigenvalues and eigenvectors of $\mathbf{B}$. Verify that exactly two eigenvalues are strictly positive and one is zero.

**(c)** Recover the $3 \times 2$ deformed coordinate matrix $\mathbf{X} = \mathbf{V}_2 \boldsymbol{\Lambda}_2^{1/2}$, placing the centroid at the origin. Verify that the Euclidean distances between the recovered points match $3, 4, 5$.

##### Solution to Exercise 13.10
**(a)** Double-centering formula for entries of $\mathbf{B} = [b_{ij}]$:
$$
b_{ij} = -\frac{1}{2}\left(d_{ij}^2 - \frac{1}{n}\sum_{k=1}^n d_{ik}^2 - \frac{1}{n}\sum_{m=1}^n d_{mj}^2 + \frac{1}{n^2}\sum_{k=1}^n \sum_{m=1}^n d_{km}^2\right)
$$
1. Row sums of $\mathbf{D}^{(2)}$:
   - Row 1: $0 + 9 + 16 = 25 \implies \text{mean} = 25/3$
   - Row 2: $9 + 0 + 25 = 34 \implies \text{mean} = 34/3$
   - Row 3: $16 + 25 + 0 = 41 \implies \text{mean} = 41/3$
   - Total sum: $25 + 34 + 41 = 100 \implies \text{grand mean} = 100/9$
2. Compute elements of $\mathbf{B}$:
   - $b_{11} = -\frac{1}{2}\left(0 - \frac{25}{3} - \frac{25}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(-\frac{50}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(-\frac{50}{9}\right) = \frac{25}{9} \approx 2.7778$
   - $b_{22} = -\frac{1}{2}\left(0 - \frac{34}{3} - \frac{34}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(-\frac{68}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(-\frac{104}{9}\right) = \frac{52}{9} \approx 5.7778$
   - $b_{33} = -\frac{1}{2}\left(0 - \frac{41}{3} - \frac{41}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(-\frac{82}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(-\frac{146}{9}\right) = \frac{73}{9} \approx 8.1111$
   - $b_{12} = -\frac{1}{2}\left(9 - \frac{25}{3} - \frac{34}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(9 - \frac{59}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(\frac{81 - 177 + 100}{9}\right) = -\frac{1}{2}\left(\frac{4}{9}\right) = -\frac{2}{9}$
   - $b_{13} = -\frac{1}{2}\left(16 - \frac{25}{3} - \frac{41}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(16 - 22 + \frac{100}{9}\right) = -\frac{1}{2}\left(\frac{-54 + 100}{9}\right) = -\frac{23}{9}$
   - $b_{23} = -\frac{1}{2}\left(25 - \frac{34}{3} - \frac{41}{3} + \frac{100}{9}\right) = -\frac{1}{2}\left(25 - 25 + \frac{100}{9}\right) = -\frac{50}{9}$

The matrix $\mathbf{B}$ is:
$$
\mathbf{B} = \frac{1}{9}\begin{pmatrix}
25 & -2 & -23 \\
-2 & 52 & -50 \\
-23 & -50 & 73
\end{pmatrix}
$$
Notice that every row and column sum is exactly zero: $(25 - 2 - 23)/9 = 0$.

**(b)** The trace of $\mathbf{B}$ is $\operatorname{tr}(\mathbf{B}) = (25 + 52 + 73)/9 = 150/9 = 50/3 \approx 16.6667$.
Because the 3 points form a right triangle in 2D Euclidean space, the target configuration embeds perfectly in $\mathbb{R}^2$ with zero residual stress.
Thus, $\mathbf{B}$ has rank 2:
- $\lambda_1 \approx 14.12$
- $\lambda_2 \approx 2.55$
- $\lambda_3 = 0.00$

**(c)** A right triangle with legs of length 3 (along $x$) and 4 (along $y$) has coordinates:
- $\mathbf{p}_1 = (0, 0)'$
- $\mathbf{p}_2 = (3, 0)'$
- $\mathbf{p}_3 = (0, 4)'$
The centroid is $\bar{\mathbf{p}} = (1, 4/3)'$. Centering at the origin gives:
- $\mathbf{d}_1 = (-1, -4/3)'$
- $\mathbf{d}_2 = (2, -4/3)'$
- $\mathbf{d}_3 = (-1, 8/3)'$
Check distances:
- $\|\mathbf{d}_1 - \mathbf{d}_2\| = \sqrt{(2 - (-1))^2 + 0} = \sqrt{3^2} = 3.0$.
- $\|\mathbf{d}_1 - \mathbf{d}_3\| = \sqrt{0 + (8/3 - (-4/3))^2} = \sqrt{(12/3)^2} = \sqrt{4^2} = 4.0$.
- $\|\mathbf{d}_2 - \mathbf{d}_3\| = \sqrt{(-1 - 2)^2 + (8/3 - (-4/3))^2} = \sqrt{(-3)^2 + 4^2} = \sqrt{9 + 16} = 5.0$.
The distances match $d_{ij}^*$ exactly.

---

### Exercise 13.11: Pencil-and-Paper $2 \times 2$ Fixed Rank Kriging Solution
Consider an observation dataset with $n = 2$ sample points:
$$
\mathbf{s}_1 = (0, 0)', \quad \mathbf{s}_2 = (1, 0)'
$$
with observed values $Z_1 = 5.0$ and $Z_2 = 1.0$.
Suppose we fit a Fixed Rank Kriging model with $r = 1$ basis function ($S_1(\mathbf{s})$) centered at $(0, 0)$ with radius $R = 2.0$:
$$
S_1(\mathbf{s}) = \left(1 - \frac{\|\mathbf{s}\|^2}{4}\right)^2 \quad \text{for } \|\mathbf{s}\| \le 2
$$
Assume zero drift ($\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$), basis coefficient prior variance $K = 4.0$, and nugget measurement error $\sigma_\epsilon^2 = 0.25$.

**(a)** Evaluate the $2 \times 1$ basis matrix $\mathbf{S} = \begin{pmatrix} S_1(\mathbf{s}_1) \\ S_1(\mathbf{s}_2) \end{pmatrix}$.

**(b)** Compute the scalar Gram term $\mathbf{S}'\mathbf{S}$ and the scalar term $M = K^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}$.

**(c)** Solve for the latent weight estimate $\widehat{\eta} = M^{-1} \left(\frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{Z}\right)$.

**(d)** Predict the field value $\widehat{Z}(\mathbf{s}_0)$ and kriging prediction variance $\sigma_{\text{FRK}}^2(\mathbf{s}_0)$ at target location $\mathbf{s}_0 = (0.5, 0)'$.

##### Solution to Exercise 13.11
**(a)**
- At $\mathbf{s}_1 = (0, 0)'$: $\|\mathbf{s}_1\| = 0$.
  $$S_1(\mathbf{s}_1) = \left(1 - \frac{0}{4}\right)^2 = 1.0$$
- At $\mathbf{s}_2 = (1, 0)'$: $\|\mathbf{s}_2\| = 1$.
  $$S_1(\mathbf{s}_2) = \left(1 - \frac{1^2}{4}\right)^2 = \left(1 - \frac{1}{4}\right)^2 = \left(\frac{3}{4}\right)^2 = \frac{9}{16} = 0.5625$$
The basis matrix is:
$$
\mathbf{S} = \begin{pmatrix} 1.0 \\ 0.5625 \end{pmatrix}
$$

**(b)** The scalar Gram term is:
$$
\mathbf{S}'\mathbf{S} = (1.0)^2 + (0.5625)^2 = 1.0 + 0.31640625 = 1.31640625
$$
With $K = 4.0$ (so $K^{-1} = 0.25$) and $\sigma_\epsilon^2 = 0.25$ (so $1/\sigma_\epsilon^2 = 4.0$):
$$
M = K^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S} = 0.25 + 4.0(1.31640625) = 0.25 + 5.265625 = 5.515625
$$
$$
M^{-1} = \frac{1}{5.515625} \approx 0.181303
$$

**(c)** Compute $\mathbf{S}'\mathbf{Z}$:
$$
\mathbf{S}'\mathbf{Z} = (1.0)(5.0) + (0.5625)(1.0) = 5.0 + 0.5625 = 5.5625
$$
Multiply by $1/\sigma_\epsilon^2 = 4.0$:
$$
\frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{Z} = 4.0 \times 5.5625 = 22.25
$$
The estimated basis weight is:
$$
\widehat{\eta} = M^{-1} \left(\frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{Z}\right) = \frac{22.25}{5.515625} \approx 4.033994 \approx 4.0340
$$

**(d)** At target location $\mathbf{s}_0 = (0.5, 0)'$: $\|\mathbf{s}_0\| = 0.5$.
$$
S_1(\mathbf{s}_0) = \left(1 - \frac{0.5^2}{4}\right)^2 = \left(1 - \frac{0.25}{4}\right)^2 = (1 - 0.0625)^2 = (0.9375)^2 \approx 0.878906
$$
The kriging prediction is:
$$
\widehat{Z}(\mathbf{s}_0) = S_1(\mathbf{s}_0) \widehat{\eta} = 0.878906 \times 4.033994 \approx 3.5455
$$
The prediction variance is:
$$
\begin{aligned}
\sigma_{\text{FRK}}^2(\mathbf{s}_0) &= S_1(\mathbf{s}_0) M^{-1} S_1(\mathbf{s}_0) + \sigma_\epsilon^2 \\
&= (0.878906)^2 \times 0.181303 + 0.25 \\
&= 0.772476 \times 0.181303 + 0.25 \\
&= 0.14005 + 0.25 = 0.3901
\end{aligned}
$$
Kriging standard error: $\sigma_{\text{FRK}}(\mathbf{s}_0) = \sqrt{0.3901} \approx 0.6245$.

---

## Tier III: Algorithmic Translation & Labs in Pure Python

### Exercise 13.12: Pure Python Paciorek-Schervish Non-Stationary Field Simulation Lab
Implement a complete, dependency-free Python simulation laboratory that generates a 1D continuous non-stationary Gaussian random field on $D = [0, 10]$ where the spatial correlation length grows smoothly from west to east:
$$
\phi(s) = 0.3 + 0.4 s \quad (\text{ranging from } 0.3\text{ at } s=0 \text{ to } 4.3\text{ at } s=10)
$$
1. Form the $n \times n$ Paciorek-Schervish covariance matrix across $n = 60$ evenly spaced locations.
2. Add a tiny diagonal jitter $10^{-6}$ and compute its Cholesky decomposition $\mathbf{V} = \mathbf{L}\mathbf{L}'$.
3. Simulate a realization $\mathbf{Z} = \mathbf{L}\mathbf{u}$ with $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_n)$.
4. Output a summary table comparing the empirical local roughness (first differences $|Z_{i} - Z_{i-1}|$) in the West ($s \in [0, 3]$) versus the East ($s \in [7, 10]$).

##### Solution to Exercise 13.12
```python
"""
Exercise 13.12: Dependency-Free Paciorek-Schervish Field Simulator
Generates a 1D non-stationary Gaussian random field with spatially growing range.
"""

import math
import random


def paciorek_schervish_1d_kernel(s1, s2, phi_fn):
    """
    Evaluates 1D Paciorek-Schervish Gaussian kernel:
    C(s1, s2) = [2 phi(s1) phi(s2) / (phi(s1)^2 + phi(s2)^2)]^(1/2) * exp(- (s1 - s2)^2 / (phi(s1)^2 + phi(s2)^2))
    """
    p1 = phi_fn(s1)
    p2 = phi_fn(s2)

    var_avg = 0.5 * (p1 * p1 + p2 * p2)
    prefactor = math.sqrt((p1 * p2) / var_avg)

    diff = s1 - s2
    q = (diff * diff) / (2.0 * var_avg)
    return prefactor * math.exp(-q)


def cholesky_decompose(A):
    """Computes lower Cholesky factor L such that A = L * L'."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(1e-12, val))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


if __name__ == "__main__":
    # Domain [0, 10] with n = 60 grid points
    n = 60
    s_coords = [10.0 * i / (n - 1) for i in range(n)]

    # Spatially varying range: phi(s) = 0.3 + 0.4 * s
    def phi_func(s):
        return 0.3 + 0.4 * s

    # 1. Build Paciorek-Schervish Covariance Matrix
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            V[i][j] = paciorek_schervish_1d_kernel(s_coords[i], s_coords[j], phi_func)
        V[i][i] += 1e-6  # Regularization jitter

    # 2. Cholesky Factorization and Simulation
    L = cholesky_decompose(V)

    random.seed(42)
    u = [random.gauss(0.0, 1.0) for _ in range(n)]
    z = [sum(L[i][k] * u[k] for k in range(i + 1)) for i in range(n)]

    # 3. Analyze Local Roughness: West (s < 3.0) vs. East (s > 7.0)
    west_diffs = []
    east_diffs = []
    for i in range(1, n):
        s_mid = 0.5 * (s_coords[i] + s_coords[i - 1])
        diff = abs(z[i] - z[i - 1])
        if s_mid <= 3.0:
            west_diffs.append(diff)
        elif s_mid >= 7.0:
            east_diffs.append(diff)

    mean_west_roughness = sum(west_diffs) / len(west_diffs)
    mean_east_roughness = sum(east_diffs) / len(east_diffs)

    print("=" * 70)
    print("Paciorek-Schervish Non-Stationary Gaussian Field Simulation")
    print("=" * 70)
    print(f"Domain                  : [0, 10] sampled at n = {n} locations")
    print(f"West Correlation Range  : phi(0.0) = {phi_func(0.0):.2f}")
    print(f"East Correlation Range  : phi(10.0) = {phi_func(10.0):.2f}")
    print("-" * 70)
    print(f"West Average Roughness  : {mean_west_roughness:.4f} (rapid fluctuations)")
    print(f"East Average Roughness  : {mean_east_roughness:.4f} (smooth undulating trend)")
    print(f"Roughness Ratio (W / E) : {mean_west_roughness / mean_east_roughness:.2f}x")
    print("-" * 70)
    print("Conclusion: West has short range (high roughness); East has long range (smooth).")
    print("=" * 70)
```

---

### Exercise 13.13: Diagnostic Scalability Lab: Exact Kriging vs. Fixed Rank Kriging
Implement a pure Python benchmarking laboratory that compares **Exact Kriging** (Cholesky inversion) against **Fixed Rank Kriging** across increasing sample sizes $n \in \{100, 250, 500, 1000\}$:
1. Generate synthetic 2D spatial data with underlying continuous surface $Z(\mathbf{s}) = \cos(2\pi x)\sin(2\pi y) + \epsilon$.
2. For Exact Kriging, record execution time and predict at 5 test locations.
3. For Fixed Rank Kriging ($r = 16$ bisquare basis functions), record execution time and predict at the same 5 test locations.
4. Output a comparative ASCII table showing Mean Absolute Prediction Difference and Runtime Scaling.

##### Solution to Exercise 13.13
```python
"""
Exercise 13.13: Diagnostic Scalability Lab (Exact Kriging vs. Fixed Rank Kriging)
Pure Python 3 implementation benchmarking computational speed and prediction fidelity.
"""

import math
import random
import time


def cholesky_decompose_and_solve(A, b):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(1e-12, A[i][i] - s))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][k] * y[k] for k in range(i))
        y[i] = (b[i] - s) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[k][i] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - s) / L[i][i]
    return x


def exact_kriging_predict(sample_pts, sample_vals, target_pts, phi=0.3, nugget=0.04):
    n = len(sample_pts)
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = sample_pts[i]
        for j in range(n):
            xj, yj = sample_pts[j]
            d = math.hypot(xi - xj, yi - yj)
            V[i][j] = math.exp(-d / phi)
        V[i][i] += nugget
    weights = cholesky_decompose_and_solve(V, sample_vals)
    preds = []
    for tx, ty in target_pts:
        k_vec = [math.exp(-math.hypot(tx - sx, ty - sy) / phi) for sx, sy in sample_pts]
        pred = sum(k_vec[i] * weights[i] for i in range(n))
        preds.append(pred)
    return preds


def frk_predict(sample_pts, sample_vals, target_pts, basis_centers, rad=0.40, nugget=0.04):
    n = len(sample_pts)
    r = len(basis_centers)

    def basis_vec(x, y):
        return [((1.0 - (math.hypot(x - cx, y - cy) / rad)**2)**2) if math.hypot(x - cx, y - cy) < rad else 0.0 for cx, cy in basis_centers]

    S = [basis_vec(x, y) for x, y in sample_pts]
    StS = [[sum(S[i][j] * S[i][k] for i in range(n)) for k in range(r)] for j in range(r)]
    StZ = [sum(S[i][j] * sample_vals[i] for i in range(n)) for j in range(r)]

    M = [[(1.0 / nugget) * StS[j][k] + (0.5 if j == k else 0.0) for k in range(r)] for j in range(r)]
    rhs = [(1.0 / nugget) * StZ[j] for j in range(r)]
    eta_hat = cholesky_decompose_and_solve(M, rhs)

    preds = []
    for tx, ty in target_pts:
        b0 = basis_vec(tx, ty)
        preds.append(sum(b0[j] * eta_hat[j] for j in range(r)))
    return preds


if __name__ == "__main__":
    random.seed(123)
    target_pts = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

    # Basis centers grid: r = 16
    basis_centers = []
    for bx in [0.2, 0.4, 0.6, 0.8]:
        for by in [0.2, 0.4, 0.6, 0.8]:
            basis_centers.append((bx, by))

    print("=" * 75)
    print("Scalability Lab: Exact Kriging vs. Fixed Rank Kriging Benchmarking")
    print("=" * 75)
    print(f"{'Sample Size n':>14} | {'Exact Time (s)':>16} | {'FRK Time (s)':>14} | {'Speedup':>10} | {'Mean Diff':>12}")
    print("-" * 75)

    for n in [100, 250, 500]:
        pts = [(random.random(), random.random()) for _ in range(n)]
        vals = [math.cos(2 * math.pi * x) * math.sin(2 * math.pi * y) + random.gauss(0, 0.1) for x, y in pts]

        # Time Exact Kriging
        t0 = time.time()
        preds_exact = exact_kriging_predict(pts, vals, target_pts)
        t_exact = time.time() - t0

        # Time FRK
        t0 = time.time()
        preds_frk = frk_predict(pts, vals, target_pts, basis_centers)
        t_frk = time.time() - t0

        speedup = t_exact / max(1e-6, t_frk)
        mean_diff = sum(abs(pe - pf) for pe, pf in zip(preds_exact, preds_frk)) / len(target_pts)

        print(f"{n:14d} | {t_exact:16.4f} | {t_frk:14.4f} | {speedup:9.1f}x | {mean_diff:12.4f}")

    print("-" * 75)
    print("Interpretation: While Exact Kriging scales as O(n^3), FRK scales as O(n * r^2).")
    print("=" * 75)
```

---

## Tier IV: Edge-Case Diagnostics & Conceptual Reflections

### Exercise 13.14: The Over-Smoothing Pathology of Stationary Kriging
In alpine meteorology, precipitation and wind velocity exhibit extreme spatial heterogeneity: narrow mountain passes have short spatial correlation lengths ($\phi \approx 5\text{ km}$), whereas adjacent plateau basins have long spatial correlation lengths ($\phi \approx 80\text{ km}$).

**(a)** Suppose an analyst fits a single global stationary spherical variogram with range $\phi_{\text{global}} = 40\text{ km}$ across the entire domain.
Describe the physical distortion that occurs in kriging predictions within the mountain passes.

**(b)** Describe the distortion that occurs within the plateau basins.

**(c)** Explain why the kriging prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ under the global stationary model produces a misleading map of prediction confidence.

##### Solution to Exercise 13.14
**(a)** In the mountain passes, the true physical correlation drops to zero within 5 km due to ridgeline windward/leeward barriers.
When kriging at an unsampled mountain location $\mathbf{s}_0$, the global stationary model with $\phi = 40\text{ km}$ gives substantial positive kriging weights to weather stations located 15–30 km away on the opposite side of major peaks.
These distant stations contaminate the local estimate, severely dampening local peaks and valleys. Sharp local rain shadows and localized micro-climatic wind gusts are completely erased; the resulting kriging map is **pathologically over-smoothed**.

**(b)** In the broad plateau basin, atmospheric moisture is physically coherent over 80 km.
However, because the global stationary variogram truncates correlation at $\phi = 40\text{ km}$, the kriging system sets weights for all monitoring stations beyond 40 km to zero.
Valuable, highly informative monitoring data across the plain is completely ignored. The kriging predictor relies only on immediate neighbors, needlessly inflating prediction variance and producing local boundary artifacts (patchy discontinuities).

**(c)** In Ordinary Kriging, the prediction variance formula is:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{c}(\mathbf{s}_0)'\mathbf{V}^{-1}\mathbf{c}(\mathbf{s}_0) + \mu
$$
Notice that $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ depends ONLY on:
1. The global variogram parameters ($c_0, c, a$).
2. The geometric configuration of sample points relative to $\mathbf{s}_0$.
It has zero awareness of whether $\mathbf{s}_0$ lies in a turbulent, volatile mountain pass or a calm, uniform plateau.
If a mountain pass happens to have three closely spaced stations, the stationary model reports a tiny kriging variance (falsely claiming high certainty), even though physical terrain complexity renders the field highly uncertain.
Conversely, if a station in the plain is 25 km away, the model reports high uncertainty, even though the plain is physically predictable.
Only non-stationary covariance architectures (such as Paciorek-Schervish or FRK) correctly scale both the prediction and the prediction variance to match local physical terrain variability.

---

### Exercise 13.15: Rank Selection and Basis Function Aperture in Fixed Rank Kriging
In Fixed Rank Kriging (FRK), the user must choose:
1. The basis rank $r$ (number of basis functions).
2. The aperture radius $R_l$ of each basis function.

**(a)** What happens if the aperture radii $R_l$ are chosen too small relative to the inter-basis spacing, such that certain regions in the domain $D$ contain points $\mathbf{s}$ where $S_l(\mathbf{s}) = 0$ for all $l = 1, \dots, r$?

**(b)** What happens if $r$ is chosen too small ($r \ll r_{\text{optimal}}$) for a complex, multi-scale field?

**(c)** What happens if $r$ is chosen too large ($r \to n$), and what numerical instability arises in inverting $\mathbf{M} = \mathbf{K}^{-1} + \frac{1}{\sigma_\epsilon^2}\mathbf{S}'\mathbf{S}$?

##### Solution to Exercise 13.15
**(a)** If basis function radii $R_l$ do not sufficiently overlap, there will exist spatial "dead zones" in domain $D$ where all basis functions evaluate to zero: $\mathbf{S}(\mathbf{s}_0) = \mathbf{0}$.
At any such location $\mathbf{s}_0$:
$$
\widehat{Z}(\mathbf{s}_0) = \mathbf{x}(\mathbf{s}_0)'\widehat{\boldsymbol{\beta}} + \mathbf{S}(\mathbf{s}_0)'\widehat{\boldsymbol{\eta}} = \mathbf{x}(\mathbf{s}_0)'\widehat{\boldsymbol{\beta}} + 0 = \mathbf{x}(\mathbf{s}_0)'\widehat{\boldsymbol{\beta}}
$$
The kriging prediction collapses abruptly to the global deterministic trend, and the spatial covariance between $\mathbf{s}_0$ and all sample data drops to zero. The prediction map exhibits artificial circular "dimples" or dead zones where local data influence vanishes. To avoid this, radii must ensure that every point in $D$ is covered by at least 2–4 overlapping basis functions.

**(b)** If $r$ is chosen too small (e.g., $r = 9$ for a 10,000-point dataset), the $r$-dimensional subspace $\operatorname{span}(S_1, \dots, S_r)$ lacks the degrees of freedom to represent fine-scale spatial variability.
The low-rank component captures only broad, regional macro-trends, and all short-range spatial correlation is dumped into the white noise residual $\epsilon(\mathbf{s})$. The estimated nugget variance $\widehat{\sigma}_\epsilon^2$ becomes severely inflated, and the model under-fits the data.

**(c)** If $r$ is chosen excessively large ($r \approx n$):
1. **Computational Loss:** The Woodbury inversion shortcut requires factorizing an $r \times r$ matrix costing $\mathcal{O}(r^3)$. If $r \to n$, the algorithm reverts to the original $\mathcal{O}(n^3)$ bottleneck.
2. **Multicollinearity & Numerical Instability:** Densely packed basis functions with wide apertures become nearly linearly dependent across the sample coordinates. The Gram matrix $\mathbf{S}'\mathbf{S}$ becomes severely ill-conditioned (condition number $\kappa(\mathbf{S}'\mathbf{S}) \gg 10^{12}$). Without strong regularization from the prior precision $\mathbf{K}^{-1}$, computing $\mathbf{M}^{-1}$ suffers from severe numerical roundoff error and exploding basis weights.

---

### Exercise 13.16: Decisional Framework: Stationary vs. Non-Stationary vs. Areal Modeling
An environmental data science team is planning statistical modeling architectures for four real-world spatial problems. For each scenario, recommend the optimal methodology:
- **Option 1:** Classical Stationary Geostatistics (Variograms & Ordinary/Universal Kriging)
- **Option 2:** Non-Stationary Geostatistics (Paciorek-Schervish or Process Convolutions)
- **Option 3:** Scalable Low-Rank Geostatistics (Fixed Rank Kriging)
- **Option 4:** Areal Lattice Modeling (SAR / CAR)

Provide a thorough justification based on spatial support, sample size $n$, and physical terrain constraints.

**(a)** **Global Satellite Chlorophyll Mapping:** $n = 450,000$ daily sea-surface color retrievals from the MODIS Aqua satellite across the North Atlantic Ocean.

**(b)** **Coastal Air Quality with Sea-Breeze Inversion:** PM2.5 particulate concentrations measured across 120 air monitoring stations in the Los Angeles basin, where topography (mountains vs. ocean coastline) creates sharp rotating directional continuity.

**(c)** **Agricultural Soil Core Survey:** Topsoil organic carbon measured from $n = 220$ auger cores sampled on a flat, homogeneous 80-hectare agricultural research farm.

**(d)** **Statewide Opioid Overdose Surveillance:** Annual county-level overdose fatality rates recorded across the 67 administrative counties of Pennsylvania.

##### Solution to Exercise 13.16
**(a)** **Global Satellite Chlorophyll Mapping: Option 3 (Scalable Fixed Rank Kriging)**
- *Sample Size:* $n = 450,000$ observations makes classical Cholesky kriging ($n^3 \approx 10^{17}$ FLOPs) computationally impossible.
- *Modeling Objective:* Fast, memory-efficient spatial smoothing and gap-filling across satellite swaths.
- *Justification:* Fixed Rank Kriging (FRK) with multi-resolution bisquare basis functions ($r \approx 500$) handles $n = 450,000$ points in seconds using Sherman-Morrison-Woodbury inversion, easily scaling to big remote sensing data while naturally accommodating oceanic spatial non-stationarity.

**(b)** **Coastal Air Quality: Option 2 (Non-Stationary Geostatistics: Paciorek-Schervish)**
- *Sample Size:* $n = 120$ stations is small enough that $n \times n$ matrix inversions are fast ($\mathcal{O}(120^3)$ is sub-millisecond).
- *Physical Mechanism:* Airflow in the LA basin is heavily steered by complex topography (San Gabriel mountains barrier, ocean sea-breeze front). Correlation range and principal axis of anisotropy rotate continuously across coordinates.
- *Justification:* A global stationary model would fail catastrophically by smoothing air pollution over high mountain ridges into the desert. The Paciorek-Schervish kernel parameterizes local metric tensors $\boldsymbol{\Sigma}(\mathbf{s})$ aligned with coastal valleys, preserving sharp physical barriers and rotating anisotropic plumes.

**(c)** **Agricultural Soil Core Survey: Option 1 (Classical Stationary Geostatistics)**
- *Sample Size:* $n = 220$ samples is modest and well within standard computational limits.
- *Physical Terrain:* A flat, homogeneous 80-hectare farm parcel lacks major macro-scale geological barriers or climatic gradients.
- *Justification:* The stationarity assumption is physically well-founded over this local support. Fitting standard isotropic or geometrically anisotropic empirical semivariograms (Spherical or Matérn) via Weighted Least Squares or REML and performing Ordinary Kriging provides optimal, mathematically parsimonious spatial interpolation without the risk of over-fitting complex non-stationary hyperparameters.

**(d)** **Statewide Opioid Overdose Surveillance: Option 4 (Areal Lattice Modeling: Besag-York-Mollié CAR)**
- *Spatial Support:* Data consists of health counts aggregated across discrete administrative polygons (67 counties).
- *Justification:* There are no point coordinates; counties are contiguous areal units with varying population sizes and boundary adjacencies. A Spatial GLMM with an Intrinsic CAR (ICAR) prior allows small rural counties with sparse counts to borrow statistical strength from neighboring counties, stabilizing volatile rate estimates while identifying regional public health epidemics.
