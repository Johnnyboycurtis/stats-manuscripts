# Appendix D: Integral Geometry Relations for Spatial Regularization in $\mathbb{R}^2$ and $\mathbb{R}^3$

This appendix develops the mathematical machinery of integral geometry and geometric probability required for spatial regularization, support transitions, and block dispersion computations. We formulate the geometric covariogram of compact Euclidean bodies, prove the dimensional reduction theorem that collapses $2d$-fold volume integrals to single $d$-fold convolutions, establish the Cauchy-Crofton chord-length relations, derive closed-form representations for segments, rectangles, and cuboids, and provide analytical formulations for Matheron's classical geostatistical auxiliary functions.

---

### D.1 The Geometric Covariogram and Dimensional Reduction

Evaluating regularized covariances $\overline{C}(v_1, v_2)$ or within-block semivariograms $\overline{\gamma}(v, v)$ requires computing integrals of the form:
$$
I = \int_{v_1} \int_{v_2} g(x - y) dx dy
$$
In Euclidean space $\mathbb{R}^d$, this represents an integration over $\mathbb{R}^{2d}$ ($4$-fold integration in $\mathbb{R}^2$; $6$-fold integration in $\mathbb{R}^3$). Integral geometry enables the reduction of this problem to a single $d$-fold integral over the difference vector $h = x - y$.

```
Geometric Interpretation of the Covariogram
Set v                                Translate v_h = v + h
+-------------+                      +-------------+
|             |                      |             |
|        +----+--------+             |             |
|        |/////|       |             +-------------+
|        |/v \cap v_h /|
+--------+----+        |
         |             |
         +-------------+
         Overlap Area: K_v(h) = \operatorname{Leb}(v \cap (v + h))
```

#### Definition D.1.1 (The Geometric Covariogram)
Let $v \subset \mathbb{R}^d$ be a compact set with positive Lebesgue measure $|v| = \operatorname{Leb}(v) > 0$. Let $\mathbb{I}_v: \mathbb{R}^d \to \{0, 1\}$ denote its indicator function.
1. The **geometric covariogram** $K_v: \mathbb{R}^d \to [0, \infty)$ of $v$ is the Lebesgue measure of the intersection of $v$ with its translate by vector $h$:
   $$
   K_v(h) = \operatorname{Leb}(v \cap (v - h)) = \operatorname{Leb}(v \cap (v + h)) = \int_{\mathbb{R}^d} \mathbb{I}_v(x) \mathbb{I}_v(x + h) dx
   $$
2. In terms of spatial convolution, defining $\check{\mathbb{I}}_v(x) = \mathbb{I}_v(-x)$:
   $$
   K_v(h) = (\mathbb{I}_v * \check{\mathbb{I}}_v)(h)
   $$
3. For two distinct compact sets $v_1, v_2 \subset \mathbb{R}^d$, the **cross-geometric covariogram** is:
   $$
   K_{v_1, v_2}(h) = \operatorname{Leb}(v_1 \cap (v_2 - h)) = \int_{\mathbb{R}^d} \mathbb{I}_{v_1}(x) \mathbb{I}_{v_2}(x + h) dx = (\mathbb{I}_{v_1} * \check{\mathbb{I}}_{v_2})(h)
   $$

#### Theorem D.1.1 (Structural Properties of the Geometric Covariogram)
For any compact set $v \subset \mathbb{R}^d$:
1. **Value at the Origin:** $K_v(0) = |v|$.
2. **Symmetry:** $K_v(-h) = K_v(h)$ for all $h \in \mathbb{R}^d$.
3. **Compact Support:** The support of $K_v$ is the Minkowski difference:
   $$
   \operatorname{supp}(K_v) = v \ominus v = \{x - y : x \in v, \, y \in v\}
   $$
   If $\operatorname{diam}(v) = D$, then $K_v(h) = 0$ for all $\|h\| > D$.
4. **Total Mass:**
   $$
   \int_{\mathbb{R}^d} K_v(h) dh = |v|^2
   $$
5. **Continuity and Boundary Behavior:**
   $K_v$ is uniformly continuous on $\mathbb{R}^d$. Moreover, if $v$ has a $C^2$-smooth boundary $\partial v$, the directional derivative at the origin in direction $\mathbf{u} \in S^{d-1}$ satisfies Matheron's boundary formula:
   $$
   \lim_{t \to 0^+} \frac{K_v(0) - K_v(t\mathbf{u})}{t} = \operatorname{Proj}_{\mathbf{u}^\perp}(v)
   $$
   where $\operatorname{Proj}_{\mathbf{u}^\perp}(v)$ is the orthogonal projection area of $v$ onto the hyperplane perpendicular to $\mathbf{u}$.

*Proof.*
1. $K_v(0) = \int_{\mathbb{R}^d} \mathbb{I}_v(x)^2 dx = \int_{\mathbb{R}^d} \mathbb{I}_v(x) dx = |v|$.
2. Making the change of variables $y = x + h$ with $dy = dx$:
   $$
   K_v(h) = \int_{\mathbb{R}^d} \mathbb{I}_v(x) \mathbb{I}_v(x + h) dx = \int_{\mathbb{R}^d} \mathbb{I}_v(y - h) \mathbb{I}_v(y) dy = K_v(-h)
   $$
3. The integrand $\mathbb{I}_v(x) \mathbb{I}_v(x+h)$ is strictly non-zero if and only if there exists $x \in v$ such that $x + h \in v$, which implies $h = y - x \in v \ominus v$.
4. By Fubini-Tonelli:
   $$
   \int_{\mathbb{R}^d} K_v(h) dh = \int_{\mathbb{R}^d} \left(\int_{\mathbb{R}^d} \mathbb{I}_v(x) \mathbb{I}_v(x + h) dx\right) dh = \int_{\mathbb{R}^d} \mathbb{I}_v(x) \left(\int_{\mathbb{R}^d} \mathbb{I}_v(x + h) dh\right) dx = \int_{\mathbb{R}^d} \mathbb{I}_v(x) |v| dx = |v|^2
   $$
5. The function $K_v = \mathbb{I}_v * \check{\mathbb{I}}_v$ is the convolution of two $L^2$ functions, hence it is uniformly continuous. The directional derivative follows from the divergence theorem applied to the cylinder formed by the translation of $v$ along $\mathbf{u}$. $\blacksquare$

#### Theorem D.1.2 (The Fundamental Dimensional Reduction Theorem)
Let $v_1, v_2 \subset \mathbb{R}^d$ be compact domains of positive measure, and let $g: \mathbb{R}^d \to \mathbb{R}$ be a locally integrable function. Then:
$$
\int_{v_1} \int_{v_2} g(x - y) dx dy = \int_{\mathbb{R}^d} K_{v_2, v_1}(h) g(h) dh
$$
In particular, for $v_1 = v_2 = v$:
$$
\int_v \int_v g(x - y) dx dy = \int_{\mathbb{R}^d} K_v(h) g(h) dh
$$
Consequently, the within-block average semivariogram is:
$$
\overline{\gamma}(v, v) = \frac{1}{|v|^2} \int_{\mathbb{R}^d} K_v(h) \gamma(h) dh
$$
reducing a $2d$-dimensional integral to a single $d$-dimensional integral over $\operatorname{supp}(K_v)$.

*Proof.*
Express the double integral over indicator functions on $\mathbb{R}^d$:
$$
\int_{v_1} \int_{v_2} g(x - y) dx dy = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} \mathbb{I}_{v_1}(x) \mathbb{I}_{v_2}(y) g(x - y) dx dy
$$
Introduce the change of variables $h = x - y \in \mathbb{R}^d$ and keep $y \in \mathbb{R}^d$ as the second variable. The Jacobian of the linear transformation $(x, y) \mapsto (h, y)$ is:
$$
\mathbf{J} = \det \begin{pmatrix} \mathbf{I}_d & \mathbf{I}_d \\ \mathbf{0} & \mathbf{I}_d \end{pmatrix} = 1
$$
Applying Fubini's theorem:
$$
\begin{aligned}
\int_{v_1} \int_{v_2} g(x - y) dx dy &= \int_{\mathbb{R}^d} dh \, g(h) \left(\int_{\mathbb{R}^d} \mathbb{I}_{v_1}(y + h) \mathbb{I}_{v_2}(y) dy\right) \\
&= \int_{\mathbb{R}^d} dh \, g(h) \operatorname{Leb}(v_2 \cap (v_1 - h)) \\
&= \int_{\mathbb{R}^d} K_{v_2, v_1}(h) g(h) dh
\end{aligned}
$$
When $v_1 = v_2 = v$, $K_{v, v}(h) = K_v(h)$, completing the proof. $\blacksquare$

#### Corollary D.1.1 (Radial Reduction for Isotropic Structures)
If the kernel is isotropic, $g(h) = g(\|h\|) = g(r)$, define the **direction-averaged geometric covariogram**:
$$
\overline{K}_v(r) = \frac{1}{|S^{d-1}|} \int_{S^{d-1}} K_v(r \mathbf{u}) d\sigma(\mathbf{u})
$$
where $S^{d-1}$ is the unit sphere in $\mathbb{R}^d$ and $|S^{d-1}| = \frac{2\pi^{d/2}}{\Gamma(d/2)}$.
Then:
$$
\int_v \int_v g(\|x - y\|) dx dy = |S^{d-1}| \int_0^{D_{\max}} r^{d-1} \overline{K}_v(r) g(r) dr
$$
1. In $\mathbb{R}^2$ ($|S^1| = 2\pi$):
   $$
   \overline{\gamma}(v, v) = \frac{2\pi}{|v|^2} \int_0^{D_{\max}} r \overline{K}_v(r) \gamma(r) dr
   $$
2. In $\mathbb{R}^3$ ($|S^2| = 4\pi$):
   $$
   \overline{\gamma}(v, v) = \frac{4\pi}{|v|^2} \int_0^{D_{\max}} r^2 \overline{K}_v(r) \gamma(r) dr
   $$

---

### D.2 Cauchy-Crofton Formula and Chord-Length Distributions

Integral geometry characterizes convex bodies through their intersections with affine lines. Crofton's theorems establish an exact link between the geometric covariogram and the distribution of chord lengths produced by random lines intersecting $v$.

#### Definition D.2.1 (Kinematic Measure on Affine Lines)
Let $\mathcal{A}(1, d)$ denote the affine Grassmannian of all unoriented 1D affine lines $L$ in $\mathbb{R}^d$. Any line $L \in \mathcal{A}(1, d)$ is uniquely parameterized by its direction $\mathbf{u} \in S^{d-1}$ (modulo sign $\pm$) and its orthogonal displacement $p \in \mathbf{u}^\perp \cong \mathbb{R}^{d-1}$.
The rigid-motion invariant Kinematic Measure $d\mu(L)$ on $\mathcal{A}(1, d)$ is defined as:
$$
d\mu(L) = d\sigma(\mathbf{u}) dp
$$
where $d\sigma(\mathbf{u})$ is the uniform spherical measure on the hemisphere and $dp$ is the Lebesgue measure on the hyperplane $\mathbf{u}^\perp$.

#### Theorem D.2.1 (The Cauchy-Crofton Formula for Volume and Surface Area)
Let $v \subset \mathbb{R}^d$ be a bounded convex body with surface area $\operatorname{Area}(\partial v)$ and volume $|v|$. Let $\ell(L \cap v) = \operatorname{Leb}_1(L \cap v)$ denote the length of the chord formed by the intersection of line $L$ with $v$.
1. **Total Measure of Intersecting Lines:**
   The measure of all affine lines intersecting $v$ is proportional to the surface area of $v$:
   $$
   \int_{\{L : L \cap v \ne \emptyset\}} d\mu(L) = \frac{|S^{d-2}|}{d-1} \operatorname{Area}(\partial v)
   $$
   Specifically:
   - In $\mathbb{R}^2$: The measure of lines intersecting a convex set of perimeter $U$ is $U$.
   - In $\mathbb{R}^3$: The measure of lines intersecting a convex body of surface area $A$ is $\pi A$.
2. **Mean Chord Length:**
   The mean chord length $\bar{\ell}$ with respect to the kinematic measure is:
   $$
   \bar{\ell} = \frac{\int_{\{L : L \cap v \ne \emptyset\}} \ell(L \cap v) d\mu(L)}{\int_{\{L : L \cap v \ne \emptyset\}} d\mu(L)} = \frac{|S^{d-1}|}{|S^{d-2}|} \frac{|v|}{\operatorname{Area}(\partial v)}
   $$
   - In $\mathbb{R}^2$: $\bar{\ell} = \pi \frac{|v|}{U}$.
   - In $\mathbb{R}^3$: $\bar{\ell} = 4 \frac{|v|}{A}$.

#### Theorem D.2.2 (Crofton's Second Theorem: Covariogram-Chord Length Identity)
Let $F_\Lambda(r) = P(\ell(L \cap v) \le r)$ denote the cumulative distribution function of chord lengths generated by random uniform lines intersecting $v$, and let $f_\Lambda(r) = F_\Lambda'(r)$ be its probability density.
The isotropic geometric covariogram $\overline{K}_v(r)$ and the chord-length density $f_\Lambda(r)$ satisfy the differential relation:
$$
\frac{d^2}{dr^2}\left[\frac{\overline{K}_v(r)}{|v|}\right] = \frac{1}{\bar{\ell}} f_\Lambda(r)
$$
for all $r > 0$.
Integrating twice yields the representation:
$$
\frac{\overline{K}_v(r)}{|v|} = 1 - \frac{r}{\bar{\ell}} + \frac{1}{\bar{\ell}} \int_0^r (r - t) f_\Lambda(t) dt
$$

*Proof.*
Consider a random line $L$ with direction $\mathbf{u}$ and position $p \in \mathbf{u}^\perp$. If $L \cap v \ne \emptyset$, the intersection is a line segment of length $\ell(p, \mathbf{u})$.
The volume of the intersection of $v$ with its translate $v + r\mathbf{u}$ along the line $L$ is given by the 1D overlap of segments:
$$
\operatorname{Leb}_1\left((L \cap v) \cap ((L \cap v) + r\mathbf{u})\right) = \max\left(0, \, \ell(p, \mathbf{u}) - r\right) = (\ell(p, \mathbf{u}) - r)^+
$$
Integrating over all lines parallel to $\mathbf{u}$ across the projection hyperplane $\mathbf{u}^\perp$:
$$
K_v(r \mathbf{u}) = \int_{\mathbf{u}^\perp} (\ell(p, \mathbf{u}) - r)^+ dp
$$
Averaging over all unit directions $\mathbf{u} \in S^{d-1}$:
$$
\overline{K}_v(r) = \frac{1}{|S^{d-1}|} \int_{S^{d-1}} \left(\int_{\mathbf{u}^\perp} (\ell(p, \mathbf{u}) - r)^+ dp\right) d\sigma(\mathbf{u}) = \frac{1}{|S^{d-1}|} \int_{\{L : L \cap v \ne \emptyset\}} (\ell(L \cap v) - r)^+ d\mu(L)
$$
By definition of the chord-length density $f_\Lambda(t)$ and total line measure $M = \frac{|v|}{\bar{\ell}}|S^{d-1}|$:
$$
\overline{K}_v(r) = \frac{M}{|S^{d-1}|} \int_r^\infty (t - r) f_\Lambda(t) dt = \frac{|v|}{\bar{\ell}} \int_r^\infty (t - r) f_\Lambda(t) dt
$$
Dividing by $|v|$:
$$
\frac{\overline{K}_v(r)}{|v|} = \frac{1}{\bar{\ell}} \int_r^\infty (t - r) f_\Lambda(t) dt
$$
Differentiating with respect to $r$ via Leibniz's rule:
$$
\frac{d}{dr}\left[\frac{\overline{K}_v(r)}{|v|}\right] = \frac{1}{\bar{\ell}} \left[ -(r - r)f_\Lambda(r) + \int_r^\infty (-1) f_\Lambda(t) dt \right] = -\frac{1}{\bar{\ell}} \int_r^\infty f_\Lambda(t) dt = -\frac{1}{\bar{\ell}}[1 - F_\Lambda(r)]
$$
At $r = 0$, this confirms $\left.\frac{d}{dr}\frac{\overline{K}_v(r)}{|v|}\right|_{r=0} = -\frac{1}{\bar{\ell}}$.
Differentiating a second time:
$$
\frac{d^2}{dr^2}\left[\frac{\overline{K}_v(r)}{|v|}\right] = -\frac{1}{\bar{\ell}} \left(-f_\Lambda(r)\right) = \frac{1}{\bar{\ell}} f_\Lambda(r)
$$
This completes the proof. $\blacksquare$

---

### D.3 Explicit Geometric Covariograms for Canonical Geometries

For standard rectangular and block geometries, the multi-dimensional geometric covariogram factors into a product of one-dimensional covariograms.

#### Proposition D.3.1 (1D Line Segment)
Let $v = [0, L] \subset \mathbb{R}$ be a segment of length $L > 0$.
1. The geometric covariogram is:
   $$
   K_L(h) = (L - |h|)^+ = \begin{cases} L - |h|, & |h| \le L \\ 0, & |h| > L \end{cases}
   $$
2. The within-segment average semivariogram is:
   $$
   \overline{\gamma}(L, L) = \frac{2}{L^2} \int_0^L (L - h) \gamma(h) dh
   $$

#### Proposition D.3.2 (2D Rectangle)
Let $v = [0, a] \times [0, b] \subset \mathbb{R}^2$ be a rectangle with side lengths $a, b > 0$ and area $|v| = ab$.
1. The geometric covariogram factors as the Cartesian product of two 1D covariograms:
   $$
   K_v(h_1, h_2) = K_a(h_1) K_b(h_2) = (a - |h_1|)^+ (b - |h_2|)^+
   $$
2. The within-block average semivariogram is given by the double integral:
   $$
   \overline{\gamma}(v, v) = \frac{4}{a^2 b^2} \int_0^a \int_0^b (a - h_1)(b - h_2) \gamma\left(\sqrt{h_1^2 + h_2^2}\right) dh_1 dh_2
   $$
3. For point-to-block regularization, where point $s_0 = (0, 0)$ is the corner of $v$:
   $$
   \overline{\gamma}(s_0, v) = \frac{1}{ab} \int_0^a \int_0^b \gamma\left(\sqrt{u_1^2 + u_2^2}\right) du_1 du_2
   $$

#### Proposition D.3.3 (3D Cuboid / Parallelepiped)
Let $v = [0, a] \times [0, b] \times [0, c] \subset \mathbb{R}^3$ be a rectangular cuboid with volume $|v| = abc$.
1. The geometric covariogram is:
   $$
   K_v(h_1, h_2, h_3) = (a - |h_1|)^+ (b - |h_2|)^+ (c - |h_3|)^+
   $$
2. The within-block average semivariogram is:
   $$
   \overline{\gamma}(v, v) = \frac{8}{a^2 b^2 c^2} \int_0^a \int_0^b \int_0^c (a - h_1)(b - h_2)(c - h_3) \gamma\left(\sqrt{h_1^2 + h_2^2 + h_3^2}\right) dh_1 dh_2 dh_3
   $$

#### Proposition D.3.4 (2D Circular Disk)
Let $v = B(0, R) = \{x \in \mathbb{R}^2 : \|x\| \le R\}$ be a circular disk of radius $R$ and area $|v| = \pi R^2$.
1. The geometric covariogram is isotropic and equals the intersection area of two disks separated by distance $r = \|h\|$:
   $$
   K_v(r) = \begin{cases} 2R^2 \arccos\left(\frac{r}{2R}\right) - r\sqrt{R^2 - \frac{r^2}{4}}, & 0 \le r \le 2R \\ 0, & r > 2R \end{cases}
   $$
2. The within-disk average semivariogram is:
   $$
   \overline{\gamma}(v, v) = \frac{2\pi}{\pi^2 R^4} \int_0^{2R} r K_v(r) \gamma(r) dr = \frac{2}{\pi R^4} \int_0^{2R} r \left[ 2R^2 \arccos\left(\frac{r}{2R}\right) - r\sqrt{R^2 - \frac{r^2}{4}} \right] \gamma(r) dr
   $$

---

### D.4 Matheron's Auxiliary Functions

Matheron formalized spatial regularization using parameterized auxiliary functions that tabulate and analytically solve standard integral geometry integrals.

#### Definition D.4.1 (The Linear Auxiliary Function $\chi(L)$)
For a 1D segment $L$, the auxiliary function $\chi(L)$ is the within-line average semivariogram:
$$
\chi(L) = \overline{\gamma}(L, L) = \frac{2}{L^2} \int_0^L (L - h) \gamma(h) dh
$$

#### Definition D.4.2 (The Point-to-Line Auxiliary Function $H(L)$)
The average semivariogram between an endpoint of a line segment and the line segment itself is:
$$
H(L) = \overline{\gamma}(0, L) = \frac{1}{L} \int_0^L \gamma(h) dh
$$
The relation between $\chi(L)$ and $H(L)$ is:
$$
\chi(L) = \frac{2}{L^2} \int_0^L u H(u) du
$$

#### Definition D.4.3 (The Planar Block Auxiliary Function $F(a, b)$)
For a 2D rectangle $v = [0, a] \times [0, b]$:
$$
F(a, b) = \overline{\gamma}(v, v) = \frac{4}{a^2 b^2} \int_0^a \int_0^b (a - x)(b - y) \gamma\left(\sqrt{x^2 + y^2}\right) dx dy
$$

#### Theorem D.4.1 (Analytical Auxiliary Values for the Linear Semivariogram)
Let $\gamma(h) = \alpha \|h\|$ be a linear semivariogram with slope $\alpha > 0$.
1. **Line Segment:**
   $$
   H(L) = \frac{1}{L} \int_0^L \alpha h dh = \frac{\alpha L}{2}
   $$
   $$
   \chi(L) = \frac{2}{L^2} \int_0^L (L - h) \alpha h dh = \frac{2\alpha}{L^2} \left[ \frac{L^3}{2} - \frac{L^3}{3} \right] = \frac{\alpha L}{3}
   $$
2. **Rectangle $a \times b$:**
   $$
   F(a, b) = \frac{\alpha}{15} \left[ \frac{a^3}{b^2} + \frac{b^3}{a^2} + \left(3 - \frac{a^2}{b^2} - \frac{b^2}{a^2}\right)\sqrt{a^2 + b^2} + \frac{5}{2}\left(\frac{b^2}{a}\operatorname{arcsinh}\left(\frac{a}{b}\right) + \frac{a^2}{b}\operatorname{arcsinh}\left(\frac{b}{a}\right)\right) \right]
   $$
   For a square ($a = b$):
   $$
   F(a, a) = \frac{\alpha a}{15} \left[ 2 + \sqrt{2} + 5 \ln(1 + \sqrt{2}) \right] \approx 0.521405 \, \alpha a
   $$
3. **Circular Disk of Radius $R$:**
   $$
   \overline{\gamma}(v, v) = \frac{256}{45\pi} \alpha R \approx 1.81069 \, \alpha R
   $$

*Proof of (1).*
Direct integration:
$$
\chi(L) = \frac{2\alpha}{L^2} \int_0^L (L h - h^2) dh = \frac{2\alpha}{L^2} \left[ \frac{L h^2}{2} - \frac{h^3}{3} \right]_0^L = \frac{2\alpha}{L^2} \left( \frac{L^3}{6} \right) = \frac{\alpha L}{3}
$$
The evaluation of (2) and (3) follows by converting to polar coordinates $(r, \theta)$ and using the standard integrals for $\int r^2 \sqrt{r^2} dr$ and the covariogram representation. $\blacksquare$

---

### D.5 Analytical Exercises and Complete Solutions

#### Exercise D.1: Average Covariance of a Disk under a Gaussian Covariance
Let $v = B(0, R) \subset \mathbb{R}^2$ be a disk of radius $R$. Consider a stationary Gaussian random field with isotropic Gaussian covariance $C(h) = \sigma^2 \exp(-\|h\|^2 / a^2)$.
**(a)** Formulate the within-disk average covariance $\overline{C}(v, v)$ as a 4-fold integral over Cartesian coordinates.
**(b)** Use Theorem D.1.2 to reduce $\overline{C}(v, v)$ to a 1D integral using the geometric covariogram $K_v(r)$.
**(c)** Alternatively, compute the double integral by transforming to the product of independent 1D integrals in Cartesian coordinates using the identity $\exp(-(x^2+y^2)/a^2) = \exp(-x^2/a^2)\exp(-y^2/a^2)$, and evaluate the asymptotic limit as $R/a \to 0$.

##### Solution:
**(a)** In Cartesian coordinates, the disk is $v = \{(x_1, x_2) : x_1^2 + x_2^2 \le R^2\}$.
The 4-fold integral is:
$$
\overline{C}(v, v) = \frac{1}{(\pi R^2)^2} \iint_{x_1^2 + x_2^2 \le R^2} \iint_{y_1^2 + y_2^2 \le R^2} \sigma^2 \exp\left(-\frac{(x_1 - y_1)^2 + (x_2 - y_2)^2}{a^2}\right) dx_1 dx_2 dy_1 dy_2
$$

**(b)** Using the geometric covariogram $K_v(r)$ of a disk of radius $R$ from Proposition D.3.4:
$$
K_v(r) = 2R^2 \arccos\left(\frac{r}{2R}\right) - r\sqrt{R^2 - \frac{r^2}{4}}, \quad 0 \le r \le 2R
$$
By Corollary D.1.1, the 4-fold integral collapses to the 1D integral:
$$
\overline{C}(v, v) = \frac{2\sigma^2}{\pi R^4} \int_0^{2R} r \left[ 2R^2 \arccos\left(\frac{r}{2R}\right) - r\sqrt{R^2 - \frac{r^2}{4}} \right] \exp\left(-\frac{r^2}{a^2}\right) dr
$$
Making the substitution $u = \frac{r}{2R}$ ($r = 2R u$, $dr = 2R du$):
$$
\overline{C}(v, v) = \frac{2\sigma^2}{\pi R^4} \int_0^1 (2Ru) \left[ 2R^2 \arccos(u) - 2Ru \cdot R\sqrt{1 - u^2} \right] \exp\left(-\frac{4R^2 u^2}{a^2}\right) (2R du)
$$
$$
= \frac{16\sigma^2}{\pi} \int_0^1 u \left[ \arccos(u) - u\sqrt{1 - u^2} \right] \exp\left(-\frac{4R^2 u^2}{a^2}\right) du
$$

**(c)** As $R/a \to 0$, expand the exponential term in Taylor series:
$$
\exp\left(-\frac{4R^2 u^2}{a^2}\right) = 1 - \frac{4R^2 u^2}{a^2} + \mathcal{O}\left(\left(\frac{R}{a}\right)^4\right)
$$
For the leading term:
$$
\int_0^1 u \left[ \arccos(u) - u\sqrt{1 - u^2} \right] du
$$
Evaluate each part:
- $\int_0^1 u \arccos(u) du$: Integrate by parts with $U = \arccos(u)$, $dV = u du \implies V = u^2/2$:
  $$
  \left[ \frac{u^2}{2}\arccos(u) \right]_0^1 + \int_0^1 \frac{u^2}{2\sqrt{1 - u^2}} du = 0 + \frac{1}{2}\int_0^{\pi/2} \sin^2 \theta d\theta = \frac{1}{2}\left(\frac{\pi}{4}\right) = \frac{\pi}{8}
  $$
- $\int_0^1 u^2 \sqrt{1 - u^2} du$: Substitute $u = \sin \theta$:
  $$
  \int_0^{\pi/2} \sin^2 \theta \cos^2 \theta d\theta = \frac{1}{4}\int_0^{\pi/2} \sin^2(2\theta) d\theta = \frac{1}{4}\left(\frac{\pi}{4}\right) = \frac{\pi}{16}
  $$
Subtracting the two terms:
$$
\frac{\pi}{8} - \frac{\pi}{16} = \frac{\pi}{16}
$$
Multiplying by $\frac{16\sigma^2}{\pi}$:
$$
\overline{C}(v, v) = \frac{16\sigma^2}{\pi} \left( \frac{\pi}{16} \right) = \sigma^2
$$
As $R/a \to 0$, $\overline{C}(v, v) \to \sigma^2 = C(0)$, recovering the point variance exactly.

---

#### Exercise D.2: Chord-Length Density of a 2D Rectangle
Consider a square block $v = [0, L] \times [0, L] \subset \mathbb{R}^2$.
**(a)** State the perimeter $U$ and area $|v|$, and compute the mean chord length $\bar{\ell}$ from Crofton's First Theorem.
**(b)** Compute the directional geometric covariogram $K_v(r \mathbf{u})$ for $\mathbf{u} = (1, 0)^T$ (horizontal direction) and $\mathbf{u} = (1/\sqrt{2}, 1/\sqrt{2})^T$ (diagonal direction).
**(c)** Verify Crofton's derivative identity $\left.\frac{d}{dr}\left[\frac{\overline{K}_v(r)}{|v|}\right]\right|_{r=0} = -\frac{1}{\bar{\ell}}$.

##### Solution:
**(a)** For a square of side $L$:
Area $|v| = L^2$, perimeter $U = 4L$.
By Theorem D.2.1:
$$
\bar{\ell} = \pi \frac{|v|}{U} = \pi \frac{L^2}{4L} = \frac{\pi L}{4}
$$

**(b)**
1. **Horizontal direction $\mathbf{u} = (1, 0)^T$:**
   $h = (r, 0)^T$. From Proposition D.3.2:
   $$
   K_v(r, 0) = (L - r)^+ (L - 0)^+ = L(L - r)^+ = \begin{cases} L(L - r), & 0 \le r \le L \\ 0, & r > L \end{cases}
   $$
2. **Diagonal direction $\mathbf{u} = (1/\sqrt{2}, 1/\sqrt{2})^T$:**
   $h = (r/\sqrt{2}, r/\sqrt{2})^T$.
   $$
   K_v\left(\frac{r}{\sqrt{2}}, \frac{r}{\sqrt{2}}\right) = \left(L - \frac{r}{\sqrt{2}}\right)^+ \left(L - \frac{r}{\sqrt{2}}\right)^+ = \left(L - \frac{r}{\sqrt{2}}\right)^2
   $$
   for $0 \le r \le \sqrt{2}L$, and $0$ for $r > \sqrt{2}L$.

**(c)** In $\mathbb{R}^2$, the directional derivative at $r=0$ in direction $\mathbf{u} = (\cos \theta, \sin \theta)^T$ is:
$$
\lim_{r \to 0^+} \frac{K_v(0) - K_v(r\mathbf{u})}{r} = \operatorname{Proj}_{\mathbf{u}^\perp}(v)
$$
For a square of side $L$, the projection of the square onto the line perpendicular to $\theta$ is:
$$
\operatorname{Proj}_{\mathbf{u}^\perp}(v) = L |\cos \theta| + L |\sin \theta|
$$
Averaging over $\theta \in [0, 2\pi)$:
$$
\overline{\operatorname{Proj}}(v) = \frac{1}{2\pi} \int_0^{2\pi} L (|\cos \theta| + |\sin \theta|) d\theta = \frac{4L}{2\pi} \int_0^{\pi/2} (\cos \theta + \sin \theta) d\theta = \frac{2L}{\pi} [1 + 1] = \frac{4L}{\pi} = \frac{U}{\pi}
$$
Therefore:
$$
\left.\frac{d}{dr}\overline{K}_v(r)\right|_{r=0} = -\frac{U}{\pi} = -\frac{4L}{\pi}
$$
Dividing by $|v| = L^2$:
$$
\left.\frac{d}{dr}\left[\frac{\overline{K}_v(r)}{|v|}\right]\right|_{r=0} = -\frac{4L}{\pi L^2} = -\frac{4}{\pi L}
$$
Comparing with $\bar{\ell} = \frac{\pi L}{4}$:
$$
-\frac{1}{\bar{\ell}} = -\frac{1}{\frac{\pi L}{4}} = -\frac{4}{\pi L}
$$
The two values match identically, confirming Crofton's formula.

---

#### Exercise D.3: Analytical Evaluation of the Spherical Semivariogram over a Line Segment
Let $\{Z(s) : s \in \mathbb{R}\}$ be a 1D intrinsically stationary process with spherical semivariogram:
$$
\gamma(h) = \begin{cases} c\left(\frac{3h}{2a} - \frac{h^3}{2a^3}\right), & 0 \le h \le a \\ c, & h > a \end{cases}
$$
Compute the explicit closed-form expression for the linear auxiliary function $\chi(L) = \overline{\gamma}(L, L)$ for a segment of length $L \le a$.

##### Solution:
For $L \le a$, the separation distance $h$ between any two points in the segment $[0, L]$ satisfies $0 \le h \le L \le a$. Therefore, the semivariogram is polynomial across the entire integration domain:
$$
\chi(L) = \frac{2}{L^2} \int_0^L (L - h) \gamma(h) dh = \frac{2c}{L^2} \int_0^L (L - h) \left(\frac{3h}{2a} - \frac{h^3}{2a^3}\right) dh
$$
Expand the integrand:
$$
(L - h)\left(\frac{3h}{2a} - \frac{h^3}{2a^3}\right) = \frac{3L}{2a}h - \frac{3}{2a}h^2 - \frac{L}{2a^3}h^3 + \frac{1}{2a^3}h^4
$$
Integrate each power of $h$ from $0$ to $L$:
$$
\int_0^L h dh = \frac{L^2}{2}
$$
$$
\int_0^L h^2 dh = \frac{L^3}{3}
$$
$$
\int_0^L h^3 dh = \frac{L^4}{4}
$$
$$
\int_0^L h^4 dh = \frac{L^5}{5}
$$
Substitute these into the integral:
$$
\begin{aligned}
\int_0^L (L - h)\gamma(h) dh &= c \left[ \frac{3L}{2a}\left(\frac{L^2}{2}\right) - \frac{3}{2a}\left(\frac{L^3}{3}\right) - \frac{L}{2a^3}\left(\frac{L^4}{4}\right) + \frac{1}{2a^3}\left(\frac{L^5}{5}\right) \right] \\
&= c \left[ \frac{3L^3}{4a} - \frac{L^3}{2a} - \frac{L^5}{8a^3} + \frac{L^5}{10a^3} \right] \\
&= c \left[ \left(\frac{3}{4} - \frac{2}{4}\right)\frac{L^3}{a} + \left(-\frac{5}{40} + \frac{4}{40}\right)\frac{L^5}{a^3} \right] \\
&= c \left[ \frac{L^3}{4a} - \frac{L^5}{40a^3} \right]
\end{aligned}
$$
Now multiply by $\frac{2}{L^2}$:
$$
\chi(L) = \frac{2c}{L^2} \left[ \frac{L^3}{4a} - \frac{L^5}{40a^3} \right] = c \left[ \frac{L}{2a} - \frac{L^3}{20a^3} \right]
$$
This is Matheron's exact auxiliary function $\chi(L)$ for the spherical semivariogram when $L \le a$.
