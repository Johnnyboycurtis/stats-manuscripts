# Appendix B: Proofs of Positive Definiteness and CND Admissibility

This appendix establishes the rigorous mathematical proofs of positive definiteness and conditionally negative definite (CND) admissibility for spatial correlation models in Euclidean space $\mathbb{R}^d$. We provide analytical derivations for the Exponential, Gaussian, Spherical, and Matérn covariance families, prove Schoenberg's embedding and characterization theorems, and evaluate radial Hankel inversion alongside Pólya's convexity criterion.

---

### B.1 Rigorous Proofs of Positive Definiteness for Parametric Covariance Models

By Bochner's Theorem ([Chapter 2, Theorem 2.1.1](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch02_covariance_variogram.md)), an integrable isotropic covariance function $C(h) = C_0(\|h\|)$ is positive definite in $\mathbb{R}^d$ if and only if its $d$-dimensional Fourier transform (spectral density) is non-negative everywhere:
$$
f(\omega) = \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} C(h) dh \ge 0, \quad \forall \omega \in \mathbb{R}^d
$$

---

#### B.1.1 Exponential Covariance Model

#### Theorem B.1.1 (Positive Definiteness and Spectral Density of the Exponential Model)
The Exponential covariance model:
$$
C(h) = \sigma^2 \exp\left(-\frac{\|h\|}{\phi}\right), \quad \sigma^2 > 0, \; \phi > 0
$$
is positive definite in Euclidean space of all dimensions $\mathbb{R}^d$ ($d \ge 1$), and its spectral density is:
$$
f(\omega) = \frac{\sigma^2 \Gamma\left(\frac{d+1}{2}\right) \phi^d}{\pi^{(d+1)/2} \left(1 + \phi^2 \|\omega\|^2\right)^{(d+1)/2}} > 0, \quad \forall \omega \in \mathbb{R}^d
$$

*Proof.*
We present two complementary proofs: first via Gaussian mixture subordination, and second via direct Fourier evaluation.

**Method 1: Gaussian Subordination.**
We recall the classical integral identity for the exponential function via the gamma distribution:
$$
e^{-u} = \frac{1}{\sqrt{\pi}} \int_0^\infty \frac{e^{-t}}{\sqrt{t}} \exp\left(-\frac{u^2}{4t}\right) dt, \quad u \ge 0
$$
Substituting $u = \frac{\|h\|}{\phi}$:
$$
C(h) = \sigma^2 \exp\left(-\frac{\|h\|}{\phi}\right) = \frac{\sigma^2}{\sqrt{\pi}} \int_0^\infty \frac{e^{-t}}{\sqrt{t}} \exp\left(-\frac{\|h\|^2}{4\phi^2 t}\right) dt
$$
Notice that for each fixed $t > 0$, the kernel $h \mapsto \exp\left(-\frac{\|h\|^2}{4\phi^2 t}\right)$ is a Gaussian covariance function with variance 1 and range parameter $2\phi\sqrt{t}$, which is strictly positive definite in any dimension $\mathbb{R}^d$. 

Because $C(h)$ is represented as a mixture of positive definite kernels weighted by the positive measure $d\mu(t) = \frac{\sigma^2}{\sqrt{\pi}} \frac{e^{-t}}{\sqrt{t}} dt$ on $(0, \infty)$, $C(h)$ is positive definite in $\mathbb{R}^d$ for all $d \ge 1$.

**Method 2: Analytical Calculation of the Spectral Density.**
Applying Fubini's Theorem to interchange the spatial Fourier integral and the subordination integral:
$$
\begin{aligned}
f(\omega) &= \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} \left[\frac{\sigma^2}{\sqrt{\pi}} \int_0^\infty \frac{e^{-t}}{\sqrt{t}} \exp\left(-\frac{\|h\|^2}{4\phi^2 t}\right) dt\right] dh \\
&= \frac{\sigma^2}{(2\pi)^d \sqrt{\pi}} \int_0^\infty \frac{e^{-t}}{\sqrt{t}} \left[\int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} \exp\left(-\frac{\|h\|^2}{4\phi^2 t}\right) dh\right] dt
\end{aligned}
$$
The inner integral is the Fourier transform of a $d$-dimensional Gaussian function with variance parameter $\alpha = 2\phi^2 t$:
$$
\int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} \exp\left(-\frac{\|h\|^2}{2\alpha}\right) dh = (2\pi \alpha)^{d/2} \exp\left(-\frac{\alpha \|\omega\|^2}{2}\right) = \left(4\pi \phi^2 t\right)^{d/2} \exp\left(-\phi^2 t \|\omega\|^2\right)
$$
Substituting this result:
$$
\begin{aligned}
f(\omega) &= \frac{\sigma^2}{(2\pi)^d \sqrt{\pi}} \int_0^\infty \frac{e^{-t}}{\sqrt{t}} \left(4\pi \phi^2 t\right)^{d/2} \exp\left(-\phi^2 t \|\omega\|^2\right) dt \\
&= \frac{\sigma^2 2^d \pi^{d/2} \phi^d}{(2\pi)^d \sqrt{\pi}} \int_0^\infty t^{\frac{d-1}{2}} \exp\left(-t\left(1 + \phi^2 \|\omega\|^2\right)\right) dt \\
&= \frac{\sigma^2 \phi^d}{\pi^{(d+1)/2}} \int_0^\infty t^{\frac{d+1}{2} - 1} \exp\left(-t\left(1 + \phi^2 \|\omega\|^2\right)\right) dt
\end{aligned}
$$
Using the substitution $v = t(1 + \phi^2 \|\omega\|^2)$, $dt = \frac{dv}{1 + \phi^2 \|\omega\|^2}$:
$$
f(\omega) = \frac{\sigma^2 \phi^d}{\pi^{(d+1)/2} \left(1 + \phi^2 \|\omega\|^2\right)^{(d+1)/2}} \int_0^\infty v^{\frac{d+1}{2} - 1} e^{-v} dv = \frac{\sigma^2 \Gamma\left(\frac{d+1}{2}\right) \phi^d}{\pi^{(d+1)/2} \left(1 + \phi^2 \|\omega\|^2\right)^{(d+1)/2}}
$$
Because $\Gamma((d+1)/2) > 0$ and $1 + \phi^2 \|\omega\|^2 > 0$, $f(\omega) > 0$ for all $\omega \in \mathbb{R}^d$. By Bochner's Theorem, $C(h)$ is strictly positive definite in any Euclidean dimension $d \ge 1$. $\blacksquare$

---

#### B.1.2 Gaussian Covariance Model

#### Theorem B.1.2 (Positive Definiteness and Spectral Density of the Gaussian Model)
The Gaussian covariance model:
$$
C(h) = \sigma^2 \exp\left(-\frac{\|h\|^2}{a^2}\right), \quad \sigma^2 > 0, \; a > 0
$$
is strictly positive definite in Euclidean space of all dimensions $\mathbb{R}^d$ ($d \ge 1$), with spectral density:
$$
f(\omega) = \frac{\sigma^2 a^d}{(2\sqrt{\pi})^d} \exp\left(-\frac{a^2 \|\omega\|^2}{4}\right) > 0, \quad \forall \omega \in \mathbb{R}^d
$$

*Proof.*
Because the Euclidean norm squared decomposes as $\|h\|^2 = \sum_{k=1}^d h_k^2$ and the inner product as $\langle \omega, h \rangle = \sum_{k=1}^d \omega_k h_k$, the $d$-dimensional Fourier integral factors into a product of $d$ one-dimensional integrals:
$$
\begin{aligned}
f(\omega) &= \frac{\sigma^2}{(2\pi)^d} \int_{\mathbb{R}^d} \exp\left(-i \sum_{k=1}^d \omega_k h_k - \sum_{k=1}^d \frac{h_k^2}{a^2}\right) dh \\
&= \frac{\sigma^2}{(2\pi)^d} \prod_{k=1}^d \int_{-\infty}^\infty \exp\left(-i \omega_k h_k - \frac{h_k^2}{a^2}\right) dh_k
\end{aligned}
$$
Completing the square in the exponent of each univariate integral:
$$
-\frac{h_k^2}{a^2} - i \omega_k h_k = -\frac{1}{a^2} \left(h_k^2 + i a^2 \omega_k h_k\right) = -\frac{1}{a^2} \left(h_k + \frac{i a^2 \omega_k}{2}\right)^2 - \frac{a^2 \omega_k^2}{4}
$$
By contour integration and Cauchy's Integral Theorem, shifting the integration contour along the imaginary axis leaves the integral invariant:
$$
\int_{-\infty}^\infty \exp\left(-\frac{1}{a^2} \left(h_k + \frac{i a^2 \omega_k}{2}\right)^2\right) dh_k = \int_{-\infty}^\infty \exp\left(-\frac{u^2}{a^2}\right) du = a \sqrt{\pi}
$$
Multiplying the terms across all $d$ dimensions:
$$
f(\omega) = \frac{\sigma^2}{(2\pi)^d} \prod_{k=1}^d \left[a \sqrt{\pi} \exp\left(-\frac{a^2 \omega_k^2}{4}\right)\right] = \frac{\sigma^2 a^d \pi^{d/2}}{(2\pi)^d} \exp\left(-\frac{a^2 \sum_{k=1}^d \omega_k^2}{4}\right) = \frac{\sigma^2 a^d}{(2\sqrt{\pi})^d} \exp\left(-\frac{a^2 \|\omega\|^2}{4}\right)
$$
Because $f(\omega) > 0$ everywhere on $\mathbb{R}^d$ and is integrable, $C(h)$ is strictly positive definite for all $d \ge 1$. $\blacksquare$

---

#### B.1.3 Spherical Covariance Model

#### Theorem B.1.3 (Geometric Construction and Dimension Restriction of the Spherical Model)
Let $C_{\text{sph}}: \mathbb{R}^d \to \mathbb{R}$ denote the Spherical covariance function with range $a > 0$:
$$
C_{\text{sph}}(h) = \begin{cases}
\sigma^2 \left[1 - \frac{3}{2}\left(\frac{\|h\|}{a}\right) + \frac{1}{2}\left(\frac{\|h\|}{a}\right)^3\right], & 0 \le \|h\| \le a \\
0, & \|h\| > a
\end{cases}
$$
1. $C_{\text{sph}}(h)$ is positive definite in $\mathbb{R}^1$, $\mathbb{R}^2$, and $\mathbb{R}^3$.
2. $C_{\text{sph}}(h)$ **fails to be positive definite** in $\mathbb{R}^d$ for all $d \ge 4$.

*Proof.*
**Part 1: Positive Definiteness in $\mathbb{R}^3$ via Geometric Covariogram.**
Let $r = a/2$. Let $B_r = \{x \in \mathbb{R}^3 : \|x\| \le r\}$ denote the closed Euclidean ball of radius $r$ centered at the origin in $\mathbb{R}^3$, with volume $|B_r| = \frac{4}{3}\pi r^3 = \frac{\pi a^3}{6}$. Let $g(x) = \mathbb{I}_{B_r}(x)$.

Consider the spatial convolution of $g$ with its reflection $\widetilde{g}(x) = g(-x) = g(x)$:
$$
K(h) = (g * g)(h) = \int_{\mathbb{R}^3} \mathbb{I}_{B_r}(u) \mathbb{I}_{B_r}(u + h) du = |B_r \cap (B_r - h)|
$$
This represents the geometric volume of intersection of two spheres of radius $r$ whose centers are separated by distance $\|h\|$. 

```
          Sphere 1                 Sphere 2
       (Center: -h/2)           (Center: +h/2)
            /-----\                  /-----\
           /       \                /       \
          |         |              |         |
          |       (============)   |         |
          |         |  Overlap |   |         |
           \       /   Region   \  /        /
            \-----/                  \-----/
               <-------- ||h|| -------->
```

When $\|h\| \ge 2r = a$, the spheres are disjoint, so $K(h) = 0$. 

When $\|h\| < 2r = a$, the overlap consists of two symmetric spherical caps of height:
$$
H = r - \frac{\|h\|}{2}
$$
The volume of a single spherical cap of height $H$ in a sphere of radius $r$ is:
$$
V_{\text{cap}} = \frac{\pi H^2}{3} (3r - H) = \frac{\pi}{3} \left(r - \frac{\|h\|}{2}\right)^2 \left(3r - \left(r - \frac{\|h\|}{2}\right)\right) = \frac{\pi}{3} \left(r - \frac{\|h\|}{2}\right)^2 \left(2r + \frac{\|h\|}{2}\right)
$$
Expanding the product:
$$
\begin{aligned}
V_{\text{cap}} &= \frac{\pi}{3} \left(r^2 - r \|h\| + \frac{\|h\|^2}{4}\right) \left(2r + \frac{\|h\|}{2}\right) \\
&= \frac{\pi}{3} \left(2r^3 - 2r^2 \|h\| + \frac{r \|h\|^2}{2} + \frac{r^2 \|h\|}{2} - \frac{r \|h\|^2}{2} + \frac{\|h\|^3}{8}\right) \\
&= \frac{\pi}{3} \left(2r^3 - \frac{3}{2} r^2 \|h\| + \frac{\|h\|^3}{8}\right)
\end{aligned}
$$
The total volume of intersection is $K(h) = 2 V_{\text{cap}}$:
$$
K(h) = \frac{2\pi}{3} \left(2r^3 - \frac{3}{2} r^2 \|h\| + \frac{\|h\|^3}{8}\right) = \frac{4\pi r^3}{3} \left(1 - \frac{3 \|h\|}{4r} + \frac{\|h\|^3}{16 r^3}\right)
$$
Substituting $r = a/2$:
$$
K(h) = |B_r| \left[1 - \frac{3\|h\|}{2a} + \frac{1}{2}\left(\frac{\|h\|}{a}\right)^3\right]
$$
Therefore:
$$
C_{\text{sph}}(h) = \frac{\sigma^2}{|B_{a/2}|} (g * g)(h)
$$
Taking the $3$-dimensional Fourier transform and applying the Convolution Theorem:
$$
\widehat{C}_{\text{sph}}(\omega) = \frac{\sigma^2}{|B_{a/2}|} |\widehat{g}(\omega)|^2 \ge 0, \quad \forall \omega \in \mathbb{R}^3
$$
Because the Fourier transform is identically non-negative, $C_{\text{sph}}$ is positive definite in $\mathbb{R}^3$. By restriction to lower-dimensional subspaces, it is positive definite in $\mathbb{R}^2$ and $\mathbb{R}^1$.

**Part 2: Inadmissibility in $\mathbb{R}^d$ for $d \ge 4$.**
To evaluate positive definiteness in dimension $d$, we compute the isotropic spectral density via the radial Hankel transform ([Theorem 2.1.2](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch02_covariance_variogram.md)):
$$
f_d(\rho) = \frac{1}{(2\pi)^{d/2} \rho^{(d-2)/2}} \int_0^a r^{d/2} J_{(d-2)/2}(\rho r) \left[1 - \frac{3r}{2a} + \frac{r^3}{2a^3}\right] dr
$$
For $d = 4$, the Bessel order is $(4-2)/2 = 1$. The 4D radial Hankel transform evaluates in closed form to:
$$
f_4(\rho) = \frac{\sigma^2 a^4}{4\pi^2} \frac{1}{(\rho a)^3} \left[ 8 J_2(\rho a) - \rho a J_1(\rho a) \right]
$$
Let $x = \rho a$ and define the oscillation factor $g(x) = 8 J_2(x) - x J_1(x)$. Evaluating $g(x)$ at the dimensionless frequency $x = 6.0$:
$$
J_1(6.0) \approx -0.2767, \quad J_2(6.0) \approx -0.2429
$$
$$
g(6.0) = 8(-0.2429) - 6.0(-0.2767) = -1.9432 + 1.6602 = -0.2830 < 0
$$
Because $g(6.0) < 0$, the spectral density $f_4(\rho) < 0$ takes strictly negative values over periodic intervals of radial frequency $\rho$.

More generally, a foundational theorem of Matheron (1971) establishes that a radial covariance function $\psi(r)$ possessing compact support $[0, a]$ with linear behavior near the origin ($\psi'(0^+) < 0$) belongs to the Schoenberg validity class $\Phi_d$ if and only if $d \le 3$. For any $d \ge 4$, the Hankel transform inevitably oscillates below zero.

By Bochner's Theorem, because the spectral density takes strictly negative values, the Spherical model is **not positive definite in $\mathbb{R}^d$ for any $d \ge 4$**. $\blacksquare$

---

#### B.1.4 Matérn Covariance Model

#### Theorem B.1.4 (Positive Definiteness and Spectral Density of the Matérn Class)
The Matérn covariance model:
$$
C(h) = \frac{\sigma^2}{2^{\nu-1}\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\|h\|}{\phi}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|h\|}{\phi}\right), \quad \sigma^2, \phi, \nu > 0
$$
is positive definite in Euclidean space of all dimensions $\mathbb{R}^d$ ($d \ge 1$). Its spectral density in $\mathbb{R}^d$ is:
$$
f(\omega) = \frac{\sigma^2 \Gamma\left(\nu + \frac{d}{2}\right) (2\nu)^\nu}{\pi^{d/2} \Gamma(\nu) \phi^{2\nu}} \left(\frac{2\nu}{\phi^2} + \|\omega\|^2\right)^{-\left(\nu + \frac{d}{2}\right)} > 0, \quad \forall \omega \in \mathbb{R}^d
$$

*Proof.*
We utilize the integral representation of the modified Bessel function of the second kind:
$$
K_\nu(z) = \frac{1}{2} \left(\frac{z}{2}\right)^\nu \int_0^\infty t^{-(\nu+1)} \exp\left(-t - \frac{z^2}{4t}\right) dt
$$
Substitute $z = \frac{\sqrt{2\nu}\|h\|}{\phi}$ into $C(h)$:
$$
\begin{aligned}
C(h) &= \frac{\sigma^2}{2^{\nu-1}\Gamma(\nu)} z^\nu \left[\frac{1}{2} \left(\frac{z}{2}\right)^\nu \int_0^\infty t^{-(\nu+1)} \exp\left(-t - \frac{z^2}{4t}\right) dt\right] \\
&= \frac{\sigma^2}{\Gamma(\nu)} \left(\frac{z^2}{4}\right)^\nu \int_0^\infty t^{-(\nu+1)} \exp\left(-t - \frac{z^2}{4t}\right) dt
\end{aligned}
$$
Make the change of variables $u = \frac{z^2}{4t}$, so $t = \frac{z^2}{4u}$ and $dt = -\frac{z^2}{4u^2} du$:
Alternatively and more cleanly, keep $t$ as the integration variable:
$$
C(h) = \frac{\sigma^2}{\Gamma(\nu)} \int_0^\infty t^{-(\nu+1)} e^{-t} \left(\frac{z^2}{4}\right)^\nu \exp\left(-\frac{z^2}{4t}\right) dt
$$
By using the alternative integral identity:
$$
\frac{z^\nu K_\nu(z)}{2^{\nu-1}} = \int_0^\infty s^{\nu-1} e^{-s} \exp\left(-\frac{z^2}{4s}\right) ds
$$
Substituting $z^2 = \frac{2\nu \|h\|^2}{\phi^2}$:
$$
C(h) = \frac{\sigma^2}{\Gamma(\nu)} \int_0^\infty s^{\nu-1} e^{-s} \exp\left(-\frac{\nu \|h\|^2}{2\phi^2 s}\right) ds
$$
This represents $C(h)$ as a continuous mixture of Gaussian covariance functions with variance parameter $\alpha(s) = \frac{\phi^2 s}{\nu} > 0$. 

Now, compute the $d$-dimensional Fourier transform:
$$
\begin{aligned}
f(\omega) &= \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} \left[\frac{\sigma^2}{\Gamma(\nu)} \int_0^\infty s^{\nu-1} e^{-s} \exp\left(-\frac{\nu \|h\|^2}{2\phi^2 s}\right) ds\right] dh \\
&= \frac{\sigma^2}{(2\pi)^d \Gamma(\nu)} \int_0^\infty s^{\nu-1} e^{-s} \left[\int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} \exp\left(-\frac{\|h\|^2}{2 (\phi^2 s / \nu)}\right) dh\right] ds
\end{aligned}
$$
Evaluating the inner $d$-dimensional Gaussian integral:
$$
\int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} \exp\left(-\frac{\|h\|^2}{2 (\phi^2 s / \nu)}\right) dh = \left(2\pi \frac{\phi^2 s}{\nu}\right)^{d/2} \exp\left(-\frac{\phi^2 s \|\omega\|^2}{2\nu}\right)
$$
Substituting this into the $s$-integral:
$$
\begin{aligned}
f(\omega) &= \frac{\sigma^2}{(2\pi)^d \Gamma(\nu)} \int_0^\infty s^{\nu-1} e^{-s} \left(\frac{2\pi \phi^2}{\nu}\right)^{d/2} s^{d/2} \exp\left(-\frac{\phi^2 \|\omega\|^2}{2\nu} s\right) ds \\
&= \frac{\sigma^2 \phi^d}{(2\pi)^{d/2} \nu^{d/2} \Gamma(\nu)} \int_0^\infty s^{\nu + \frac{d}{2} - 1} \exp\left(-s\left(1 + \frac{\phi^2 \|\omega\|^2}{2\nu}\right)\right) ds
\end{aligned}
$$
Using the gamma integral $\int_0^\infty s^{\alpha-1} e^{-\beta s} ds = \Gamma(\alpha) \beta^{-\alpha}$:
$$
f(\omega) = \frac{\sigma^2 \phi^d}{(2\pi)^{d/2} \nu^{d/2} \Gamma(\nu)} \Gamma\left(\nu + \frac{d}{2}\right) \left(1 + \frac{\phi^2 \|\omega\|^2}{2\nu}\right)^{-\left(\nu + \frac{d}{2}\right)}
$$
Factoring out $\left(\frac{\phi^2}{2\nu}\right)^{-\left(\nu + d/2\right)}$:
$$
\begin{aligned}
f(\omega) &= \frac{\sigma^2 \phi^d \Gamma\left(\nu + \frac{d}{2}\right)}{(2\pi)^{d/2} \nu^{d/2} \Gamma(\nu)} \left(\frac{\phi^2}{2\nu}\right)^{-\left(\nu + \frac{d}{2}\right)} \left(\frac{2\nu}{\phi^2} + \|\omega\|^2\right)^{-\left(\nu + \frac{d}{2}\right)} \\
&= \frac{\sigma^2 \phi^d \Gamma\left(\nu + \frac{d}{2}\right)}{(2\pi)^{d/2} \nu^{d/2} \Gamma(\nu)} \frac{(2\nu)^{\nu + \frac{d}{2}}}{\phi^{2\nu + d}} \left(\frac{2\nu}{\phi^2} + \|\omega\|^2\right)^{-\left(\nu + \frac{d}{2}\right)} \\
&= \frac{\sigma^2 \Gamma\left(\nu + \frac{d}{2}\right) (2\nu)^\nu}{\pi^{d/2} \Gamma(\nu) \phi^{2\nu}} \left(\frac{2\nu}{\phi^2} + \|\omega\|^2\right)^{-\left(\nu + \frac{d}{2}\right)}
\end{aligned}
$$
Because $\nu > 0$, $\phi > 0$, and $\sigma^2 > 0$, the prefactor is strictly positive, and $\frac{2\nu}{\phi^2} + \|\omega\|^2 > 0$ for all $\omega \in \mathbb{R}^d$. 

Hence, $f(\omega) > 0$ everywhere on $\mathbb{R}^d$ for every dimension $d \ge 1$. By Bochner's Theorem, the Matérn covariance model is strictly positive definite in all Euclidean dimensions $\mathbb{R}^d$. $\blacksquare$

---

### B.2 Schoenberg's Embedding Theorem and Completely Monotonic Functions

#### Definition B.2.1 (Completely Monotonic Functions)
A function $\psi: (0, \infty) \to \mathbb{R}$ is said to be **completely monotonic** if it possesses derivatives of all orders and satisfies:
$$
(-1)^k \psi^{(k)}(t) \ge 0, \quad \forall t > 0, \; \forall k \in \{0, 1, 2, \dots\}
$$

#### Theorem B.2.1 (Bernstein's Theorem on Completely Monotonic Functions)
A function $\psi: (0, \infty) \to \mathbb{R}$ is completely monotonic if and only if it can be represented as the Laplace-Stieltjes transform of a non-negative Borel measure $\mu$ on $[0, \infty)$:
$$
\psi(t) = \int_0^\infty e^{-st} d\mu(s), \quad \forall t > 0
$$
where the integral converges for all $t > 0$.

*Proof.*
This is Bernstein's classic characterization of completely monotonic functions via the Hausdorff moment problem. $\blacksquare$

#### Theorem B.2.2 (Schoenberg's Representation Theorem, 1938)
A continuous radial function $C(h) = \psi(\|h\|^2)$ is positive definite in Euclidean space of all dimensions $\mathbb{R}^d$ for every $d \ge 1$ ($C \in \Phi_\infty$) if and only if $\psi(t)$ is completely monotonic on $(0, \infty)$.

*Proof.*
**Sufficiency:**
Suppose $\psi(t)$ is completely monotonic on $(0, \infty)$. By Bernstein's Theorem (Theorem B.2.1):
$$
\psi(t) = \int_0^\infty e^{-st} d\mu(s)
$$
for a non-negative measure $\mu$ on $[0, \infty)$ with $\mu([0, \infty)) = \psi(0) = C(0) < \infty$. 

Setting $t = \|h\|^2$:
$$
C(h) = \int_0^\infty \exp\left(-s \|h\|^2\right) d\mu(s)
$$
By Theorem B.1.2, for each fixed $s > 0$, the Gaussian kernel $h \mapsto \exp(-s\|h\|^2)$ is positive definite on $\mathbb{R}^d$ for every $d \ge 1$. Because $\mu$ is a non-negative measure, for any $n \in \mathbb{N}$, locations $\{s_1, \dots, s_n\} \subset \mathbb{R}^d$, and weights $\mathbf{c} \in \mathbb{R}^n$:
$$
\sum_{i=1}^n \sum_{j=1}^n c_i c_j C(s_i - s_j) = \int_0^\infty \left[\sum_{i=1}^n \sum_{j=1}^n c_i c_j \exp\left(-s \|s_i - s_j\|^2\right)\right] d\mu(s) \ge 0
$$
Thus, $C(h)$ is positive definite in $\mathbb{R}^d$ for every $d \ge 1$.

**Necessity:**
Suppose $C(h) = \psi(\|h\|^2)$ is positive definite in $\mathbb{R}^d$ for every $d \ge 1$. By the radial spectral representation ([Chapter 2, Theorem 2.1.2](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch02_covariance_variogram.md)), in dimension $d$:
$$
\psi(r^2) = \int_0^\infty \Omega_d(\rho r) dG_d(\rho)
$$
where $G_d$ is a bounded, non-decreasing radial spectral measure, and:
$$
\Omega_d(u) = \Gamma\left(\frac{d}{2}\right) \left(\frac{2}{u}\right)^{(d-2)/2} J_{(d-2)/2}(u)
$$
Using the asymptotic expansion of the Bessel function as order $\alpha = (d-2)/2 \to \infty$:
$$
\lim_{d \to \infty} \Omega_d\left(\rho \sqrt{\frac{2t}{d}}\right) = \exp\left(-\frac{\rho^2 t}{2d} \dots\right)
$$
Through the change of variable $s = \frac{\rho^2}{2d}$, as $d \to \infty$, the kernel $\Omega_d$ converges uniformly on compact sets to the exponential kernel:
$$
\lim_{d \to \infty} \Omega_d(\rho r) = \exp(-s r^2) = \exp(-st)
$$
By Helly's Selection Theorem on the family of measures $G_d$, there exists a limiting measure $\mu$ such that:
$$
\psi(t) = \int_0^\infty e^{-st} d\mu(s)
$$
By Bernstein's Theorem, $\psi(t)$ is completely monotonic on $(0, \infty)$. $\blacksquare$

---

#### Theorem B.2.3 (Complete Proof of Schoenberg's CND Characterization)
Let $\gamma: \mathbb{R}^d \to \mathbb{R}$ be a continuous function with $\gamma(0) = 0$ and $\gamma(-h) = \gamma(h)$. Then $\gamma(h)$ is conditionally negative definite if and only if for every $t > 0$, the function:
$$
C_t(h) = \exp\left(-t \gamma(h)\right)
$$
is positive definite on $\mathbb{R}^d$.

*Proof.*
**Sufficiency ($\Leftarrow$):**
Assume that for every $t > 0$, $C_t(h) = \exp(-t \gamma(h))$ is positive definite. 

Let $n \in \mathbb{N}$, $\{s_1, \dots, s_n\} \subset \mathbb{R}^d$, and $\boldsymbol{\lambda} \in \mathbb{R}^n$ with $\sum_{i=1}^n \lambda_i = 0$. 

Because $C_t$ is positive definite:
$$
\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \exp(-t \gamma(s_i - s_j)) \ge 0
$$
Since $\sum_{i=1}^n \lambda_i = 0$, we have $\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j = \left(\sum_{i=1}^n \lambda_i\right)^2 = 0$. Subtracting this zero term:
$$
\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \left(\exp(-t \gamma(s_i - s_j)) - 1\right) \ge 0
$$
Divide both sides by $-t < 0$, which reverses the inequality:
$$
\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \frac{1 - \exp(-t \gamma(s_i - s_j))}{t} \le 0
$$
Now take the limit as $t \downarrow 0$. By L'Hôpital's rule or Taylor expansion:
$$
\lim_{t \downarrow 0} \frac{1 - \exp(-t \gamma(h))}{t} = \gamma(h)
$$
Passing the limit into the finite sum:
$$
\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(s_i - s_j) = \lim_{t \downarrow 0} \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \frac{1 - \exp(-t \gamma(s_i - s_j))}{t} \le 0
$$
Thus, $\gamma(h)$ is conditionally negative definite.

**Necessity ($\Rightarrow$):**
Assume $\gamma(h)$ is conditionally negative definite. Fix an arbitrary origin location $s_0 \in \mathbb{R}^d$, and define the bivariate kernel:
$$
K(s_i, s_j) = \gamma(s_i - s_0) + \gamma(s_j - s_0) - \gamma(s_i - s_j)
$$
We claim that $K(s_i, s_j)$ is positive semi-definite on $\mathbb{R}^d \times \mathbb{R}^d$. 

To prove this, let $\{s_1, \dots, s_n\} \subset \mathbb{R}^d$ and $c_1, \dots, c_n \in \mathbb{R}$. Define an auxiliary $(n+1)$-th point $s_{n+1} = s_0$, and assign it weight $c_{n+1} = -\sum_{i=1}^n c_i$. 

Then $\sum_{k=1}^{n+1} c_k = \sum_{i=1}^n c_i + c_{n+1} = 0$. Because $\gamma$ is CND:
$$
\sum_{k=1}^{n+1} \sum_{l=1}^{n+1} c_k c_l \gamma(s_k - s_l) \le 0
$$
Expanding the sum by separating the index $n+1$:
$$
\begin{aligned}
\sum_{k=1}^{n+1} \sum_{l=1}^{n+1} c_k c_l \gamma(s_k - s_l) &= \sum_{i=1}^n \sum_{j=1}^n c_i c_j \gamma(s_i - s_j) + 2 \sum_{i=1}^n c_i c_{n+1} \gamma(s_i - s_0) + c_{n+1}^2 \gamma(0) \\
&= \sum_{i=1}^n \sum_{j=1}^n c_i c_j \gamma(s_i - s_j) - 2 \sum_{i=1}^n c_i \left(\sum_{j=1}^n c_j\right) \gamma(s_i - s_0) \\
&= \sum_{i=1}^n \sum_{j=1}^n c_i c_j \left[\gamma(s_i - s_j) - \gamma(s_i - s_0) - \gamma(s_j - s_0)\right] \\
&= -\sum_{i=1}^n \sum_{j=1}^n c_i c_j K(s_i, s_j) \le 0
\end{aligned}
$$
Multiplying by $-1$:
$$
\sum_{i=1}^n \sum_{j=1}^n c_i c_j K(s_i, s_j) \ge 0
$$
Thus, $K(s_i, s_j)$ is a positive semi-definite kernel.

By the **Schur Product Theorem**, if a kernel $K$ is positive semi-definite, then its Hadamard powers $K^m(s_i, s_j) = (K(s_i, s_j))^m$ are positive semi-definite for all $m \in \mathbb{N}$. 

For any $t > 0$, expanding the exponential function as a power series:
$$
\exp(t K(s_i, s_j)) = \sum_{m=0}^\infty \frac{t^m}{m!} K(s_i, s_j)^m
$$
Because each summand is positive semi-definite and the series converges uniformly on compact sets, the kernel $(s_i, s_j) \mapsto \exp(t K(s_i, s_j))$ is positive semi-definite.

Now rewrite the target function $\exp(-t \gamma(s_i - s_j))$ using the definition of $K$:
$$
-t \gamma(s_i - s_j) = t K(s_i, s_j) - t \gamma(s_i - s_0) - t \gamma(s_j - s_0)
$$
Exponentiating both sides:
$$
\exp(-t \gamma(s_i - s_j)) = \exp(-t \gamma(s_i - s_0)) \exp(t K(s_i, s_j)) \exp(-t \gamma(s_j - s_0))
$$
Define $a(s) = \exp(-t \gamma(s - s_0)) > 0$. The kernel matrix is:
$$
[\exp(-t \gamma(s_i - s_j))]_{i,j=1}^n = \mathbf{D} [\exp(t K(s_i, s_j))]_{i,j=1}^n \mathbf{D}
$$
where $\mathbf{D} = \operatorname{diag}(a(s_1), \dots, a(s_n))$ is a diagonal matrix with positive diagonal entries. 

For any $\mathbf{c} \in \mathbb{R}^n$:
$$
\mathbf{c}^T [\exp(-t \gamma(s_i - s_j))] \mathbf{c} = (\mathbf{D} \mathbf{c})^T [\exp(t K(s_i, s_j))] (\mathbf{D} \mathbf{c}) \ge 0
$$
Because this holds for any collection of points $\{s_1, \dots, s_n\}$, the function $C_t(h) = \exp(-t \gamma(h))$ is positive definite on $\mathbb{R}^d$ for every $t > 0$. $\blacksquare$

---

### B.3 Radial Fourier-Hankel Inversion in $\mathbb{R}^d$ and Pólya's Convexity Criterion

#### B.3.1 Radial Fourier-Hankel Inversion in $\mathbb{R}^d$

When modeling isotropic spatial dependence $C(h) = C_0(\|h\|)$, the $d$-dimensional Fourier integral simplifies via integration over the unit $(d-1)$-sphere $S^{d-1}$:
$$
f_0(\rho) = \frac{1}{(2\pi)^d} \int_0^\infty r^{d-1} C_0(r) \left[\int_{S^{d-1}} e^{-i \rho r \cos\theta_1} d\sigma(\xi)\right] dr
$$
The surface integral evaluates to the Bessel kernel:
$$
\int_{S^{d-1}} e^{-i \rho r \cos\theta_1} d\sigma(\xi) = (2\pi)^{d/2} \frac{J_{(d-2)/2}(\rho r)}{(\rho r)^{(d-2)/2}}
$$
yielding the **radial Hankel transform pair**:
$$
f_0(\rho) = \frac{1}{(2\pi)^{d/2} \rho^{(d-2)/2}} \int_0^\infty r^{d/2} J_{(d-2)/2}(\rho r) C_0(r) dr
$$
$$
C_0(r) = \frac{(2\pi)^{d/2}}{r^{(d-2)/2}} \int_0^\infty \rho^{d/2} J_{(d-2)/2}(\rho r) f_0(\rho) d\rho
$$
The necessary and sufficient criterion for $C_0(\|h\|)$ to be positive definite in $\mathbb{R}^d$ is:
$$
f_0(\rho) \ge 0, \quad \forall \rho \ge 0
$$

---

#### B.3.2 Pólya's Convexity Criterion on the Real Line $\mathbb{R}^1$

On the real line $\mathbb{R}^1$, the Hankel transform reduces to the classical cosine Fourier transform:
$$
f(\omega) = \frac{1}{\pi} \int_0^\infty C(r) \cos(\omega r) dr
$$
A remarkably powerful geometric criterion discovered by George Pólya allows verifying positive definiteness without computing the Fourier integral explicitly.

#### Theorem B.3.1 (Pólya's Convexity Criterion in $\mathbb{R}^1$)
Let $C: \mathbb{R} \to [0, \infty)$ be an even, continuous function ($C(-r) = C(r)$) satisfying:
1. $C(0) < \infty$.
2. $\lim_{r \to \infty} C(r) = 0$.
3. $C(r)$ is **convex** on $[0, \infty)$, meaning that for all $r_1, r_2 \ge 0$ and $\lambda \in [0, 1]$:
   $$
   C(\lambda r_1 + (1-\lambda) r_2) \le \lambda C(r_1) + (1-\lambda) C(r_2)
   $$
Then $C(h)$ is positive definite on $\mathbb{R}^1$.

*Proof.*
We present an analytical proof via double integration by parts, followed by a geometric proof via triangular basis decomposition.

**Method 1: Integration by Parts.**
Because $C(r)$ is convex on $[0, \infty)$ and bounded, its first derivative $C'(r)$ exists almost everywhere, is non-decreasing, and satisfies $C'(r) \le 0$ for all $r > 0$. Moreover, the second distributional derivative $dC'(r)$ is a non-negative Borel measure on $(0, \infty)$.

Compute the cosine Fourier transform for $\omega > 0$:
$$
f(\omega) = \frac{1}{\pi} \int_0^\infty C(r) \cos(\omega r) dr
$$
Integrate by parts with $u = C(r)$ and $dv = \cos(\omega r) dr$ ($v = \frac{\sin(\omega r)}{\omega}$):
$$
f(\omega) = \frac{1}{\pi} \left[ C(r) \frac{\sin(\omega r)}{\omega} \right]_0^\infty - \frac{1}{\pi \omega} \int_0^\infty C'(r) \sin(\omega r) dr
$$
Because $\lim_{r \to \infty} C(r) = 0$ and $C(0) \sin(0) = 0$, the boundary term vanishes:
$$
f(\omega) = -\frac{1}{\pi \omega} \int_0^\infty C'(r) \sin(\omega r) dr
$$
Integrate by parts a second time with $u = C'(r)$ and $dv = \sin(\omega r) dr$ ($v = \frac{1 - \cos(\omega r)}{\omega}$):
$$
f(\omega) = -\frac{1}{\pi \omega} \left[ C'(r) \frac{1 - \cos(\omega r)}{\omega} \right]_0^\infty + \frac{1}{\pi \omega^2} \int_0^\infty (1 - \cos(\omega r)) dC'(r)
$$
Because $C(r)$ is convex and tends to zero at infinity, $C'(r) \to 0$ as $r \to \infty$, and $1 - \cos(\omega \cdot 0) = 0$, so the boundary term vanishes. 

We obtain:
$$
f(\omega) = \frac{1}{\pi \omega^2} \int_0^\infty (1 - \cos(\omega r)) dC'(r)
$$
Since $1 - \cos(\omega r) \ge 0$ for all $r, \omega \in \mathbb{R}$, and convexity implies $dC'(r) \ge 0$, the integrand and the measure are both non-negative. 

Therefore, $f(\omega) \ge 0$ for all $\omega > 0$. By continuity, $f(0) \ge 0$. By Bochner's Theorem, $C(h)$ is positive definite on $\mathbb{R}^1$.

**Method 2: Triangular Function Decomposition.**
For any parameter $a > 0$, define the triangular (tent) function:
$$
T_a(r) = \max\left(0, 1 - \frac{|r|}{a}\right)
$$
Notice that $T_a(r)$ is the autocorrelation (covariogram) of a rectangular pulse:
$$
T_a(r) = \frac{1}{a} \int_{-\infty}^\infty \mathbb{I}_{[-a/2, a/2]}(u) \mathbb{I}_{[-a/2, a/2]}(u + r) du
$$
By the Convolution Theorem, its Fourier transform is:
$$
\widehat{T}_a(\omega) = a \left(\frac{\sin(\omega a / 2)}{\omega a / 2}\right)^2 \ge 0
$$
Any continuous, convex, decaying function $C(r)$ on $[0, \infty)$ with $\lim_{r \to \infty} C(r) = 0$ can be represented uniquely as a mixture of triangular functions:
$$
C(r) = \int_0^\infty T_a(r) a \, dC'(a)
$$
Because $dC'(a) \ge 0$ by convexity, $C(r)$ is a positive mixture of positive definite functions, hence positive definite on $\mathbb{R}^1$. $\blacksquare$

---

#### B.3.3 Failure of Pólya's Criterion in Higher Dimensions $\mathbb{R}^d$ ($d \ge 2$)

> [!CAUTION] Non-Transferability of Pólya's Criterion to $\mathbb{R}^d$
> Convexity of $C_0(r)$ on $[0, \infty)$ is **not sufficient** for positive definiteness in dimension $d \ge 2$.

To demonstrate this failure analytically, consider the triangular function $C(h) = \max(0, 1 - \|h\|/a)$ in $\mathbb{R}^2$ ($d = 2$). The function is convex on $[0, \infty)$ and satisfies all hypotheses of Pólya's Theorem on $\mathbb{R}^1$.

However, in $\mathbb{R}^2$, its spectral density is given by the Hankel transform with Bessel function $J_0$:
$$
f_2(\rho) = \frac{1}{2\pi} \int_0^a r J_0(\rho r) \left(1 - \frac{r}{a}\right) dr
$$
Using the identity $\int_0^z x J_0(x) dx = z J_1(z)$ and integrating by parts:
$$
f_2(\rho) = \frac{a^2}{2\pi (\rho a)^3} \int_0^{\rho a} z J_1(z) dz - \dots = \frac{a^2}{2\pi (\rho a)^3} \left[ \pi \rho a \left(J_1(\rho a) \mathbf{H}_0(\rho a) - J_0(\rho a) \mathbf{H}_1(\rho a)\right) \right]
$$
where $\mathbf{H}_\alpha$ is the Struve function. As $\rho a$ increases, $f_2(\rho)$ oscillates between positive and negative values. For example, for $\rho a \approx 4.5$, $f_2(\rho) < 0$.

Because the spectral density assumes negative values, the triangular covariance function is **not positive definite in $\mathbb{R}^2$ or $\mathbb{R}^3$**.

#### Theorem B.3.2 (Williamson's $k$-Convexity Theorem for $\mathbb{R}^d$)
Let $d \ge 2$. An isotropic function $C(h) = C_0(\|h\|)$ is positive definite in $\mathbb{R}^d$ for all $h \in \mathbb{R}^d$ if and only if $C_0(\sqrt{t})$ satisfies Williamson's $k$-convexity condition with $k = \lfloor d/2 \rfloor$:
$$
(-1)^j \frac{d^j}{dt^j} \left(C_0(\sqrt{t})\right) \ge 0, \quad \forall j = 0, 1, \dots, k
$$
and $(-1)^k \frac{d^k}{dt^k} (C_0(\sqrt{t}))$ is non-increasing and convex on $(0, \infty)$.

As $d \to \infty$, $k \to \infty$, requiring derivatives of all orders to alternate in sign, which recovers Schoenberg's complete monotonicity theorem (Theorem B.2.2). $\blacksquare$