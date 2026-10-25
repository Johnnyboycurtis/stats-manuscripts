# Appendix A: Spectral Theory of Stationary Random Fields

This appendix presents the rigorous harmonic analysis of spatial random fields in Euclidean space $\mathbb{R}^d$. We develop the theory of orthogonal stochastic measures and Hilbert-space-valued stochastic integration, construct the unitary shift group and derive the spectral representation of weakly stationary random fields, provide an unabridged, step-by-step measure-theoretic proof of Bochner's Theorem in $\mathbb{R}^d$, and formulate the Cramér-Khinchin spectral representation for intrinsic random fields of order 0.

---

### A.1 Orthogonal Stochastic Measures and Stochastic Integration in Hilbert Space

Throughout this appendix, $(\Omega, \mathcal{F}, P)$ denotes a complete probability space, and $L^2(\Omega, \mathcal{F}, P)$ denotes the complex Hilbert space of equivalence classes of square-integrable complex random variables equipped with the inner product:
$$
\langle X, Y \rangle_{L^2} = \operatorname{E}\left[X \overline{Y}\right] = \int_\Omega X(\omega) \overline{Y(\omega)} dP(\omega)
$$
and induced norm $\|X\|_{L^2} = \sqrt{\operatorname{E}[|X|^2]}$.

#### Definition A.1.1 (Orthogonal Stochastic Measure)
Let $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$ denote $d$-dimensional Euclidean space equipped with its Borel $\sigma$-algebra. Let $\mathcal{B}_0(\mathbb{R}^d) = \{B \in \mathcal{B}(\mathbb{R}^d) : \overline{B} \text{ is compact}\}$ denote the ring of bounded Borel sets.

An **orthogonal stochastic measure** (or **orthogonally scattered measure**) is a set function:
$$
Z: \mathcal{B}_0(\mathbb{R}^d) \to L^2(\Omega, \mathcal{F}, P)
$$
satisfying the following axioms:
1. **Centering:** For every $A \in \mathcal{B}_0(\mathbb{R}^d)$:
   $$
   \operatorname{E}[Z(A)] = 0
   $$
2. **Countable Additivity in $L^2$:** If $\{A_n\}_{n=1}^\infty \subset \mathcal{B}_0(\mathbb{R}^d)$ is a sequence of pairwise disjoint sets whose union $A = \bigcup_{n=1}^\infty A_n$ belongs to $\mathcal{B}_0(\mathbb{R}^d)$, then the series converges in $L^2$-norm:
   $$
   \lim_{N \to \infty} \left\|Z(A) - \sum_{n=1}^N Z(A_n)\right\|_{L^2} = 0
   $$
   which we write as $Z\left(\bigcup_{n=1}^\infty A_n\right) = \sum_{n=1}^\infty Z(A_n)$ in $L^2$.
3. **Orthogonality:** There exists a deterministic, non-negative, locally finite Borel measure $F$ on $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$—termed the **spectral measure** or **structure measure**—such that for all $A, B \in \mathcal{B}_0(\mathbb{R}^d)$:
   $$
   \operatorname{E}\left[Z(A) \overline{Z(B)}\right] = F(A \cap B)
   $$
   In particular, if $A \cap B = \emptyset$, then $Z(A) \perp Z(B)$ in $L^2(\Omega, \mathcal{F}, P)$:
   $$
   \langle Z(A), Z(B) \rangle_{L^2} = 0
   $$
4. **Reality (Conjugate Symmetry):** If the stochastic measure represents a real-valued random field, it satisfies the conjugate symmetry condition:
   $$
   Z(-A) = \overline{Z(A)}, \quad \forall A \in \mathcal{B}_0(\mathbb{R}^d)
   $$
   which implies that $F$ is a symmetric measure: $F(-B) = F(B)$ for all $B \in \mathcal{B}(\mathbb{R}^d)$.

---

#### A.1.2 Construction of the Stochastic Integral

We construct the stochastic integral of deterministic functions $\phi: \mathbb{R}^d \to \mathbb{C}$ with respect to the orthogonal stochastic measure $Z$:
$$
I(\phi) = \int_{\mathbb{R}^d} \phi(\omega) dZ(\omega)
$$

**Step 1: Integration of Simple Functions.**
Let $\mathcal{S}(\mathbb{R}^d)$ denote the linear space of complex-valued simple Borel functions with bounded support:
$$
\phi(\omega) = \sum_{j=1}^m c_j \mathbb{I}_{A_j}(\omega)
$$
where $m \in \mathbb{N}$, $c_j \in \mathbb{C}$, and $\{A_j\}_{j=1}^m \subset \mathcal{B}_0(\mathbb{R}^d)$ are pairwise disjoint. 

Define the stochastic integral of $\phi$ by the linear operator:
$$
I(\phi) = \sum_{j=1}^m c_j Z(A_j) \in L^2(\Omega, \mathcal{F}, P)
$$
By the additivity of $Z$, this mapping is well-defined and independent of the chosen representation of $\phi$.

We compute the $L^2$ inner product between two simple functions $\phi = \sum_{j=1}^m c_j \mathbb{I}_{A_j}$ and $\psi = \sum_{k=1}^l d_k \mathbb{I}_{B_k}$. Without loss of generality, we may refine the partitions so that $\{A_j\} = \{B_k\}$:
$$
\begin{aligned}
\langle I(\phi), I(\psi) \rangle_{L^2} &= \operatorname{E}\left[\left(\sum_{j=1}^m c_j Z(A_j)\right) \overline{\left(\sum_{k=1}^m d_k Z(A_k)\right)}\right] \\
&= \sum_{j=1}^m \sum_{k=1}^m c_j \overline{d_k} \operatorname{E}\left[Z(A_j) \overline{Z(A_k)}\right] \\
&= \sum_{j=1}^m \sum_{k=1}^m c_j \overline{d_k} F(A_j \cap A_k) \\
&= \sum_{j=1}^m c_j \overline{d_j} F(A_j) \\
&= \int_{\mathbb{R}^d} \phi(\omega) \overline{\psi(\omega)} dF(\omega) = \langle \phi, \psi \rangle_{L^2(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d), F)}
\end{aligned}
$$
In particular, the norm is preserved:
$$
\|I(\phi)\|_{L^2(P)}^2 = \operatorname{E}\left[|I(\phi)|^2\right] = \int_{\mathbb{R}^d} |\phi(\omega)|^2 dF(\omega) = \|\phi\|_{L^2(F)}^2
$$
Thus, $I$ is an isometric linear operator from the dense subspace $\mathcal{S}(\mathbb{R}^d) \subset L^2(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d), F)$ into $L^2(\Omega, \mathcal{F}, P)$.

**Step 2: Extension to $L^2(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d), F)$.**
Because simple functions $\mathcal{S}(\mathbb{R}^d)$ are dense in the Hilbert space $L^2(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d), F)$, for every $\phi \in L^2(F)$ there exists a Cauchy sequence of simple functions $\{\phi_n\}_{n=1}^\infty \subset \mathcal{S}(\mathbb{R}^d)$ such that $\lim_{n \to \infty} \|\phi_n - \phi\|_{L^2(F)} = 0$.

Because $I$ is an isometry:
$$
\|I(\phi_n) - I(\phi_m)\|_{L^2(P)} = \|\phi_n - \phi_m\|_{L^2(F)} \xrightarrow{n, m \to \infty} 0
$$
Hence $\{I(\phi_n)\}_{n=1}^\infty$ is a Cauchy sequence in the complete Hilbert space $L^2(\Omega, \mathcal{F}, P)$. By completeness, there exists a unique limit element in $L^2(P)$, which we define as the stochastic integral:
$$
I(\phi) = \int_{\mathbb{R}^d} \phi(\omega) dZ(\omega) = \lim_{n \to \infty} I(\phi_n)
$$

#### Theorem A.1.1 (Kolmogorov's Isometric Isomorphism)
The stochastic integration operator:
$$
I: L^2(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d), F) \to \mathcal{H}_Z \subset L^2(\Omega, \mathcal{F}, P)
$$
where $\mathcal{H}_Z = \overline{\operatorname{sp}}\{Z(A) : A \in \mathcal{B}_0(\mathbb{R}^d)\}$ is the closed linear subspace spanned by the stochastic measure, is an isometric isomorphism of Hilbert spaces. Specifically:
1. **Linearity:** $I(a \phi + b \psi) = a I(\phi) + b I(\psi)$ for all $a, b \in \mathbb{C}$ and $\phi, \psi \in L^2(F)$.
2. **Isometric Identity:** For all $\phi, \psi \in L^2(F)$:
   $$
   \operatorname{E}\left[\left(\int_{\mathbb{R}^d} \phi(\omega) dZ(\omega)\right) \overline{\left(\int_{\mathbb{R}^d} \psi(\omega) dZ(\omega)\right)}\right] = \int_{\mathbb{R}^d} \phi(\omega) \overline{\psi(\omega)} dF(\omega)
   $$
3. **Surjectivity:** $I\left(L^2(F)\right) = \mathcal{H}_Z$.

---

### A.2 Spectral Representation of Weakly Stationary Random Fields

Let $\{Z(s) : s \in \mathbb{R}^d\}$ be a centered ($\operatorname{E}[Z(s)] = 0$), second-order stationary, mean-square continuous random field on $\mathbb{R}^d$. Let $\mathcal{H}_Z = \overline{\operatorname{sp}}\{Z(s) : s \in \mathbb{R}^d\} \subset L^2(\Omega, \mathcal{F}, P)$ denote the Gaussian/spatial Hilbert space generated by the random field.

#### Definition A.2.1 (The Unitary Shift Group)
For each spatial shift vector $s \in \mathbb{R}^d$, define the linear operator $U_s$ on the linear span $\operatorname{sp}\{Z(t) : t \in \mathbb{R}^d\}$ by:
$$
U_s \left(\sum_{k=1}^n c_k Z(t_k)\right) = \sum_{k=1}^n c_k Z(t_k + s)
$$
We verify that $U_s$ preserves the inner product in $L^2$:
$$
\begin{aligned}
\left\langle U_s \sum_{j=1}^n c_j Z(t_j), U_s \sum_{k=1}^m d_k Z(r_k) \right\rangle_{L^2} &= \sum_{j=1}^n \sum_{k=1}^m c_j \overline{d_k} \operatorname{E}\left[Z(t_j + s) \overline{Z(r_k + s)}\right] \\
&= \sum_{j=1}^n \sum_{k=1}^m c_j \overline{d_k} C(t_j - r_k) \\
&= \left\langle \sum_{j=1}^n c_j Z(t_j), \sum_{k=1}^m d_k Z(r_k) \right\rangle_{L^2}
\end{aligned}
$$
Thus, $U_s$ extends uniquely to an isometric linear bijection on $\mathcal{H}_Z$, making $U_s$ a **unitary operator** satisfying $U_s^* = U_s^{-1} = U_{-s}$.

The family $\{U_s : s \in \mathbb{R}^d\}$ constitutes an abelian group of unitary operators:
1. $U_0 = I$ (the identity operator).
2. $U_{s_1 + s_2} = U_{s_1} U_{s_2}$ for all $s_1, s_2 \in \mathbb{R}^d$.
3. Mean-square continuity of $Z(s)$ implies that the map $s \mapsto U_s$ is strongly continuous:
   $$
   \lim_{s \to 0} \|U_s Z(t) - Z(t)\|_{L^2} = \lim_{s \to 0} \sqrt{2 C(0) - 2 \operatorname{Re}(C(s))} = 0
   $$

#### Theorem A.2.1 (Generalized Stone's Theorem for Unitary Representations of $\mathbb{R}^d$)
Let $\{U_s : s \in \mathbb{R}^d\}$ be a strongly continuous group of unitary operators acting on a separable complex Hilbert space $\mathcal{H}$. Then there exists a unique projection-valued measure (spectral family of orthogonal projections) $E: \mathcal{B}(\mathbb{R}^d) \to \mathcal{B}(\mathcal{H})$ such that:
$$
U_s = \int_{\mathbb{R}^d} e^{i \langle \omega, s \rangle} dE(\omega), \quad \forall s \in \mathbb{R}^d
$$
where the integral converges in the strong operator topology of $\mathcal{H}$.

*Proof.*
This is the classic Stone-Naimark-Ambrose-Godement (SNAG) theorem for locally compact abelian groups applied to $G = (\mathbb{R}^d, +)$, whose Pontryagin dual group is $\widehat{G} \cong \mathbb{R}^d$ under the pairing $\chi_\omega(s) = e^{i \langle \omega, s \rangle}$. $\blacksquare$

#### Theorem A.2.2 (Spectral Representation of the Random Field)
Every centered, second-order stationary, mean-square continuous random field $\{Z(s) : s \in \mathbb{R}^d\}$ admits the spectral representation:
$$
Z(s) = \int_{\mathbb{R}^d} e^{i \langle \omega, s \rangle} dZ(\omega)
$$
where $Z(\cdot)$ is an orthogonal stochastic measure on $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$ satisfying:
$$
\operatorname{E}\left[Z(A) \overline{Z(B)}\right] = F(A \cap B)
$$
for a unique finite, non-negative Borel measure $F$ on $\mathbb{R}^d$.

*Proof.*
Apply Stone's Theorem (Theorem A.2.1) to the unitary shift group $U_s$ on $\mathcal{H}_Z$. For any Borel set $A \in \mathcal{B}(\mathbb{R}^d)$, define:
$$
Z(A) = E(A) Z(0) \in \mathcal{H}_Z
$$
We verify the properties of $Z(A)$:
1. **Centering:** $\operatorname{E}[Z(A)] = \operatorname{E}[E(A) Z(0)] = 0$ because $E(A)$ is a linear operator on centered variables.
2. **Orthogonality and Spectral Measure:** Because $E(\cdot)$ is a projection-valued measure ($E(A) E(B) = E(A \cap B)$ and $E(A)^* = E(A)$):
   $$
   \begin{aligned}
   \operatorname{E}\left[Z(A) \overline{Z(B)}\right] &= \langle Z(A), Z(B) \rangle_{L^2} \\
   &= \langle E(A) Z(0), E(B) Z(0) \rangle_{L^2} \\
   &= \langle E(B)^* E(A) Z(0), Z(0) \rangle_{L^2} \\
   &= \langle E(B) E(A) Z(0), Z(0) \rangle_{L^2} \\
   &= \langle E(A \cap B) Z(0), Z(0) \rangle_{L^2}
   \end{aligned}
   $$
   Define the set function $F(A) = \langle E(A) Z(0), Z(0) \rangle_{L^2} = \|E(A) Z(0)\|_{L^2}^2$. By the countable additivity of $E(\cdot)$, $F$ is a countably additive, non-negative Borel measure on $\mathbb{R}^d$, with total mass:
   $$
   F(\mathbb{R}^d) = \|E(\mathbb{R}^d) Z(0)\|_{L^2}^2 = \|I Z(0)\|_{L^2}^2 = \|Z(0)\|_{L^2}^2 = C(0) < \infty
   $$
3. **Representation of $Z(s)$:** By Stone's Theorem, acting on $Z(0)$:
   $$
   Z(s) = U_s Z(0) = \left(\int_{\mathbb{R}^d} e^{i \langle \omega, s \rangle} dE(\omega)\right) Z(0) = \int_{\mathbb{R}^d} e^{i \langle \omega, s \rangle} d(E(\omega) Z(0)) = \int_{\mathbb{R}^d} e^{i \langle \omega, s \rangle} dZ(\omega)
   $$
   This completes the proof. $\blacksquare$

#### Corollary A.2.1 (Spectral Representation of the Stationary Covariance)
The covariance kernel $C(h) = \operatorname{Cov}(Z(s+h), Z(s))$ admits the spectral representation:
$$
C(h) = \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} dF(\omega)
$$
For a real-valued random field, $C(h) = \int_{\mathbb{R}^d} \cos(\langle \omega, h \rangle) dF(\omega)$.

*Proof.*
Using the isometric property of stochastic integration (Theorem A.1.1):
$$
\begin{aligned}
C(h) = \operatorname{E}\left[Z(s+h) \overline{Z(s)}\right] &= \operatorname{E}\left[\left(\int_{\mathbb{R}^d} e^{i \langle \omega_1, s+h \rangle} dZ(\omega_1)\right) \overline{\left(\int_{\mathbb{R}^d} e^{i \langle \omega_2, s \rangle} dZ(\omega_2)\right)}\right] \\
&= \int_{\mathbb{R}^d} e^{i \langle \omega, s+h \rangle} \overline{e^{i \langle \omega, s \rangle}} dF(\omega) \\
&= \int_{\mathbb{R}^d} e^{i \langle \omega, s+h \rangle} e^{-i \langle \omega, s \rangle} dF(\omega) \\
&= \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} dF(\omega)
\end{aligned}
$$
If $Z(s)$ is real-valued, $C(h)$ is real and even ($C(-h) = C(h)$). Decomposing $e^{i \langle \omega, h \rangle} = \cos(\langle \omega, h \rangle) + i \sin(\langle \omega, h \rangle)$:
$$
C(h) = \int_{\mathbb{R}^d} \cos(\langle \omega, h \rangle) dF(\omega) + i \int_{\mathbb{R}^d} \sin(\langle \omega, h \rangle) dF(\omega)
$$
Because $F$ is symmetric ($F(-B) = F(B)$) and $\sin(\cdot)$ is odd, the imaginary integral vanishes identically, yielding $C(h) = \int_{\mathbb{R}^d} \cos(\langle \omega, h \rangle) dF(\omega)$. $\blacksquare$

---

### A.3 Complete Measure-Theoretic Proof of Bochner's Theorem in $\mathbb{R}^d$

We now provide an independent, purely functional-analytic proof of Bochner's Theorem without appealing to operator spectral theory.

#### Theorem A.3.1 (Bochner's Theorem in $\mathbb{R}^d$)
A continuous function $C: \mathbb{R}^d \to \mathbb{C}$ is positive definite if and only if there exists a finite, non-negative Borel measure $F$ on $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$ such that:
$$
C(h) = \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} dF(\omega), \quad \forall h \in \mathbb{R}^d
$$

*Proof.*
The proof proceeds in six rigorous steps.

**Step 1: Necessity.**
Suppose there exists a finite non-negative Borel measure $F$ on $\mathbb{R}^d$ such that $C(h) = \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} dF(\omega)$. 

Continuity of $C(h)$ follows from the Dominated Convergence Theorem, since $|e^{i \langle \omega, h \rangle}| \le 1$ and $F(\mathbb{R}^d) < \infty$. 

To verify positive definiteness, let $n \in \mathbb{N}$, $s_1, \dots, s_n \in \mathbb{R}^d$, and $c_1, \dots, c_n \in \mathbb{C}$. Then:
$$
\begin{aligned}
\sum_{j=1}^n \sum_{k=1}^n c_j \overline{c_k} C(s_j - s_k) &= \sum_{j=1}^n \sum_{k=1}^n c_j \overline{c_k} \int_{\mathbb{R}^d} e^{i \langle \omega, s_j - s_k \rangle} dF(\omega) \\
&= \int_{\mathbb{R}^d} \left(\sum_{j=1}^n c_j e^{i \langle \omega, s_j \rangle}\right) \left(\sum_{k=1}^n \overline{c_k} e^{-i \langle \omega, s_k \rangle}\right) dF(\omega) \\
&= \int_{\mathbb{R}^d} \left|\sum_{j=1}^n c_j e^{i \langle \omega, s_j \rangle}\right|^2 dF(\omega) \ge 0
\end{aligned}
$$
Because the integrand is non-negative and $F$ is a non-negative measure, the integral is non-negative. This establishes necessity.

**Step 2: Continuous Extension of Positive Definiteness to $L^1$ Functions.**
Now assume $C: \mathbb{R}^d \to \mathbb{C}$ is continuous and positive definite. 

Let $\psi \in C_c(\mathbb{R}^d)$ be a continuous complex-valued function with compact support. By Riemann sum approximations of the double integral, for any partition of the support of $\psi$ into small disjoint cells $\{B_j\}$ with sample points $s_j \in B_j$ and volumes $\Delta v_j$:
$$
\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} C(x - y) \psi(x) \overline{\psi(y)} dx dy = \lim_{\max \Delta v_j \to 0} \sum_{j} \sum_{k} C(s_j - s_k) \psi(s_j) \overline{\psi(s_k)} \Delta v_j \Delta v_k \ge 0
$$
Since $C_c(\mathbb{R}^d)$ is dense in $L^1(\mathbb{R}^d)$ and $|C(x-y)| \le C(0)$, the inequality:
$$
\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} C(x - y) \psi(x) \overline{\psi(y)} dx dy \ge 0
$$
holds for all $\psi \in L^1(\mathbb{R}^d)$.

**Step 3: Gaussian Regularization and Candidate Spectral Density.**
For any parameter $\varepsilon > 0$, define the Gaussian mollified kernel:
$$
C_\varepsilon(h) = C(h) \exp\left(-\frac{\varepsilon \|h\|^2}{2}\right)
$$
Because the Gaussian function $h \mapsto \exp(-\varepsilon \|h\|^2 / 2)$ is positive definite (its Fourier transform is positive), and by the Schur Product Theorem the product of positive definite kernels is positive definite, $C_\varepsilon(h)$ is positive definite on $\mathbb{R}^d$.

Moreover, since $|C(h)| \le C(0)$, $C_\varepsilon(h)$ decays exponentially as $\|h\| \to \infty$, ensuring $C_\varepsilon \in L^1(\mathbb{R}^d) \cap L^2(\mathbb{R}^d)$. 

Define the Fourier transform of $C_\varepsilon$:
$$
f_\varepsilon(\omega) = \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} e^{-i \langle \omega, h \rangle} C_\varepsilon(h) dh
$$

**Step 4: Analytical Proof that $f_\varepsilon(\boldsymbol{\omega}) \ge 0$.**
Let $\delta > 0$. Consider the test function:
$$
\psi_\delta(\mathbf{x}) = \exp(-i \boldsymbol{\omega}'\mathbf{x}) \exp\left(-\frac{\delta \|\mathbf{x}\|^2}{2}\right) \in L^1(\mathbb{R}^d)
$$
Applying the continuous positive definiteness from Step 2 to $C_\varepsilon$:
$$
0 \le \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} C_\varepsilon(\mathbf{x} - \mathbf{y}) \exp(-i \boldsymbol{\omega}'(\mathbf{x} - \mathbf{y})) \exp\left(-\frac{\delta \|\mathbf{x}\|^2}{2}\right) \exp\left(-\frac{\delta \|\mathbf{y}\|^2}{2}\right) d\mathbf{x} d\mathbf{y}
$$
Let $h = x - y$ and $u = \frac{x + y}{2}$. The Jacobian of the transformation $(x, y) \mapsto (h, u)$ is 1. The Gaussian damping factor becomes:
$$
\exp\left(-\frac{\delta \|x\|^2}{2}\right) \exp\left(-\frac{\delta \|y\|^2}{2}\right) = \exp\left(-\delta \|u\|^2\right) \exp\left(-\frac{\delta \|h\|^2}{4}\right)
$$
Substituting these coordinates:
$$
0 \le \int_{\mathbb{R}^d} C_\varepsilon(h) e^{-i \langle \omega, h \rangle} \exp\left(-\frac{\delta \|h\|^2}{4}\right) \left[\int_{\mathbb{R}^d} \exp(-\delta \|u\|^2) du\right] dh
$$
The inner integral evaluates to:
$$
\int_{\mathbb{R}^d} \exp(-\delta \|u\|^2) du = \left(\frac{\pi}{\delta}\right)^{d/2}
$$
Dividing by $\left(\frac{\pi}{\delta}\right)^{d/2} (2\pi)^d$:
$$
0 \le \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} C_\varepsilon(h) e^{-i \langle \omega, h \rangle} \exp\left(-\frac{\delta \|h\|^2}{4}\right) dh
$$
Because $C_\varepsilon \in L^1(\mathbb{R}^d)$, we take the limit as $\delta \downarrow 0$. By Lebesgue's Dominated Convergence Theorem:
$$
f_\varepsilon(\omega) = \lim_{\delta \downarrow 0} \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} C_\varepsilon(h) e^{-i \langle \omega, h \rangle} \exp\left(-\frac{\delta \|h\|^2}{4}\right) dh \ge 0
$$
Thus, $f_\varepsilon(\omega) \ge 0$ for all $\omega \in \mathbb{R}^d$.

**Step 5: Fourier Inversion for $C_\varepsilon$ and Total Mass Conservation.**
Because $C_\varepsilon \in L^1(\mathbb{R}^d)$ is continuous and positive definite, by the classical Fourier Inversion Theorem:
$$
C_\varepsilon(h) = \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} f_\varepsilon(\omega) d\omega
$$
Evaluating at $h = 0$:
$$
\int_{\mathbb{R}^d} f_\varepsilon(\omega) d\omega = C_\varepsilon(0) = C(0) \exp(0) = C(0)
$$
Define the sequence of non-negative Borel measures:
$$
dF_\varepsilon(\omega) = f_\varepsilon(\omega) d\omega
$$
For every $\varepsilon > 0$, $F_\varepsilon(\mathbb{R}^d) = C(0) < \infty$. The family of measures $\{F_\varepsilon\}_{\varepsilon > 0}$ has uniformly bounded total mass.

**Step 6: Tightness and Weak Convergence to the Spectral Measure $F$.**
To extract a convergent subsequence of measures on $\mathbb{R}^d$, we must establish **tightness** (no mass escapes to $\|\omega\| = \infty$ as $\varepsilon \to 0$).

Let $\sigma > 0$. Integrate $C(0) - C_\varepsilon(h)$ against the Gaussian probability density $g_\sigma(h) = (2\pi \sigma^2)^{-d/2} \exp(-\|h\|^2 / (2\sigma^2))$:
$$
\begin{aligned}
\int_{\mathbb{R}^d} (C(0) - C_\varepsilon(h)) g_\sigma(h) dh &= \int_{\mathbb{R}^d} \left[\int_{\mathbb{R}^d} (1 - e^{i \langle \omega, h \rangle}) dF_\varepsilon(\omega)\right] g_\sigma(h) dh \\
&= \int_{\mathbb{R}^d} \left[1 - \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} g_\sigma(h) dh\right] dF_\varepsilon(\omega) \\
&= \int_{\mathbb{R}^d} \left(1 - \exp\left(-\frac{\sigma^2 \|\omega\|^2}{2}\right)\right) dF_\varepsilon(\omega)
\end{aligned}
$$
For any radius $R > 0$, when $\|\omega\| \ge R$, we have $1 - \exp(-\sigma^2 \|\omega\|^2 / 2) \ge 1 - \exp(-\sigma^2 R^2 / 2)$. Therefore:
$$
F_\varepsilon(\{\omega : \|\omega\| \ge R\}) \le \frac{1}{1 - \exp(-\sigma^2 R^2 / 2)} \int_{\mathbb{R}^d} (C(0) - C_\varepsilon(h)) g_\sigma(h) dh
$$
Because $C$ is continuous at the origin, for any $\eta > 0$ there exists $\delta > 0$ such that $|C(0) - C(h)| < \eta$ whenever $\|h\| < \delta$. Choosing $\sigma$ sufficiently small and then $R$ sufficiently large, we obtain:
$$
\sup_{\varepsilon > 0} F_\varepsilon(\{\omega : \|\omega\| \ge R\}) < \eta
$$
Thus, the family of measures $\{F_\varepsilon\}_{\varepsilon > 0}$ is tight on $\mathbb{R}^d$.

By **Prokhorov's Theorem**, there exists a sequence $\varepsilon_k \downarrow 0$ and a finite, non-negative Borel measure $F$ on $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$ such that $F_{\varepsilon_k} \Rightarrow F$ weakly as $k \to \infty$.

For any fixed $h \in \mathbb{R}^d$, the function $\omega \mapsto e^{i \langle \omega, h \rangle}$ is continuous and bounded. By weak convergence:
$$
\lim_{k \to \infty} \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} dF_{\varepsilon_k}(\omega) = \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} dF(\omega)
$$
On the other hand:
$$
\lim_{k \to \infty} C_{\varepsilon_k}(h) = \lim_{k \to \infty} C(h) \exp\left(-\frac{\varepsilon_k \|h\|^2}{2}\right) = C(h)
$$
Equating limits:
$$
C(h) = \int_{\mathbb{R}^d} e^{i \langle \omega, h \rangle} dF(\omega)
$$
Uniqueness of $F$ is guaranteed by the Stone-Weierstrass theorem, since linear combinations of characters $\{e^{i \langle \omega, h \rangle} : h \in \mathbb{R}^d\}$ are dense in $C_0(\mathbb{R}^d)$. This completes the proof of Bochner's Theorem. $\blacksquare$

---

### A.4 Cramér-Khinchin Theorem and Spectral Representation of Intrinsic Random Fields of Order 0

We now extend spectral theory beyond second-order stationary fields to intrinsic random fields of order 0, whose semivariograms $\gamma(h)$ may grow without bound as $\|h\| \to \infty$.

#### Definition A.4.1 (Spatial Increment Filter)
Let $\{Z(s) : s \in \mathbb{R}^d\}$ be an intrinsically stationary random field of order 0 with semivariogram $\gamma(h)$. For any fixed lag vector $h \in \mathbb{R}^d$, define the **spatial increment field**:
$$
Y_h(s) = \Delta_h Z(s) = Z(s+h) - Z(s)
$$
The increment field $Y_h(s)$ is a centered, second-order stationary random field on $\mathbb{R}^d$ with stationary covariance kernel:
$$
\begin{aligned}
C_{Y_h}(u) &= \operatorname{Cov}(Y_h(s+u), Y_h(s)) \\
&= \operatorname{E}\left[(Z(s+u+h) - Z(s+u))(Z(s+h) - Z(s))\right] \\
&= \frac{1}{2}\operatorname{E}\left[(Z(s+u+h) - Z(s))^2\right] + \frac{1}{2}\operatorname{E}\left[(Z(s+u) - Z(s+h))^2\right] \\
&\quad - \frac{1}{2}\operatorname{E}\left[(Z(s+u+h) - Z(s+h))^2\right] - \frac{1}{2}\operatorname{E}\left[(Z(s+u) - Z(s))^2\right] \\
&= \gamma(u+h) + \gamma(u-h) - 2\gamma(u)
\end{aligned}
$$
At $u = 0$:
$$
C_{Y_h}(0) = \operatorname{Var}(Z(s+h) - Z(s)) = 2\gamma(h)
$$

#### Lemma A.4.1 (Spectral Filtering of Increments)
By Bochner's Theorem applied to the stationary increment field $Y_h(s)$, there exists a finite symmetric Borel measure $F_h$ on $\mathbb{R}^d$ such that:
$$
C_{Y_h}(u) = \int_{\mathbb{R}^d} e^{i \langle \omega, u \rangle} dF_h(\omega)
$$
Evaluating at $u = 0$:
$$
2\gamma(h) = \int_{\mathbb{R}^d} dF_h(\omega)
$$
The measures $F_h$ for varying $h \in \mathbb{R}^d$ are mutually consistent, and there exists a unique symmetric Borel measure $\Lambda$ on $\mathbb{R}^d \setminus \{0\}$ such that:
$$
dF_h(\omega) = \left|e^{i \langle \omega, h \rangle} - 1\right|^2 d\Lambda(\omega) = 2\left(1 - \cos(\langle \omega, h \rangle)\right) d\Lambda(\omega)
$$

*Proof.*
Consider the joint increment structure for two distinct lags $h_1, h_2 \in \mathbb{R}^d$. The cross-covariance between $Y_{h_1}(s)$ and $Y_{h_2}(s)$ is stationary:
$$
\operatorname{Cov}(Y_{h_1}(s+u), Y_{h_2}(s)) = \gamma(u + h_1) + \gamma(u - h_2) - \gamma(u) - \gamma(u + h_1 - h_2)
$$
By the spectral representation of multivariate stationary fields, there exists a complex cross-spectral measure $F_{h_1, h_2}$. 

Notice that the difference operator $\Delta_h$ corresponds in harmonic space to multiplication by the transfer function $e^{i \langle \omega, h \rangle} - 1$. For any disjoint collection of Borel sets bounded away from the origin, the ratio:
$$
\Lambda(B) = \int_B \frac{dF_h(\omega)}{|e^{i \langle \omega, h \rangle} - 1|^2}
$$
is independent of the choice of $h \in \mathbb{R}^d$ for which $|e^{i \langle \omega, h \rangle} - 1| \ne 0$ on $B$. 

Extending to all Borel subsets of $\mathbb{R}^d \setminus \{0\}$, this defines a unique Borel measure $\Lambda$ on $\mathbb{R}^d \setminus \{0\}$. $\blacksquare$

#### Theorem A.4.1 (Cramér-Khinchin Representation for Intrinsic Random Fields of Order 0)
A continuous function $\gamma: \mathbb{R}^d \to [0, \infty)$ with $\gamma(0) = 0$ is conditionally negative definite if and only if it admits the representation:
$$
\gamma(h) = \int_{\mathbb{R}^d \setminus \{0\}} \left(1 - \cos(\langle \omega, h \rangle)\right) d\Lambda(\omega) + h^T \mathbf{A} h
$$
where:
1. $\mathbf{A} \in \mathbb{R}^{d \times d}$ is a symmetric, positive semi-definite matrix.
2. $\Lambda$ is a symmetric Borel measure on $\mathbb{R}^d \setminus \{0\}$ ($\Lambda(-B) = \Lambda(B)$) satisfying the **admissibility integrability condition**:
   $$
   \int_{\mathbb{R}^d \setminus \{0\}} \frac{\|\omega\|^2}{1 + \|\omega\|^2} d\Lambda(\omega) < \infty
   $$
If the intrinsic random field possesses no deterministic linear drift component, then $\mathbf{A} = \mathbf{0}$, and:
$$
\gamma(h) = \int_{\mathbb{R}^d \setminus \{0\}} \left(1 - \cos(\langle \omega, h \rangle)\right) d\Lambda(\omega)
$$

*Proof.*
**Sufficiency:**
Suppose $\gamma(h)$ admits the integral representation. We verify conditional negative definiteness. 

Let $n \in \mathbb{N}$, $\{s_1, \dots, s_n\} \subset \mathbb{R}^d$, and $\boldsymbol{\lambda} \in \mathbb{R}^n$ with $\sum_{j=1}^n \lambda_j = 0$. 

First, consider the quadratic term $h^T \mathbf{A} h$:
$$
\sum_{j=1}^n \sum_{k=1}^n \lambda_j \lambda_k (s_j - s_k)^T \mathbf{A} (s_j - s_k) = -2 \left(\sum_{j=1}^n \lambda_j s_j\right)^T \mathbf{A} \left(\sum_{k=1}^n \lambda_k s_k\right) \le 0
$$
because $\mathbf{A}$ is positive semi-definite.

Second, consider the integral term:
$$
\begin{aligned}
\sum_{j=1}^n \sum_{k=1}^n \lambda_j \lambda_k \gamma(s_j - s_k) &= \int_{\mathbb{R}^d \setminus \{0\}} \sum_{j=1}^n \sum_{k=1}^n \lambda_j \lambda_k \left(1 - \cos(\langle \omega, s_j - s_k \rangle)\right) d\Lambda(\omega) \\
&= \int_{\mathbb{R}^d \setminus \{0\}} \left[ \left(\sum_{j=1}^n \lambda_j\right) \left(\sum_{k=1}^n \lambda_k\right) - \sum_{j=1}^n \sum_{k=1}^n \lambda_j \lambda_k \operatorname{Re}\left(e^{i \langle \omega, s_j \rangle} e^{-i \langle \omega, s_k \rangle}\right) \right] d\Lambda(\omega)
\end{aligned}
$$
Because $\sum_{j=1}^n \lambda_j = 0$, the first term vanishes identically:
$$
\sum_{j=1}^n \sum_{k=1}^n \lambda_j \lambda_k \gamma(s_j - s_k) = -\int_{\mathbb{R}^d \setminus \{0\}} \left|\sum_{j=1}^n \lambda_j e^{i \langle \omega, s_j \rangle}\right|^2 d\Lambda(\omega) \le 0
$$
Because the integrand is non-negative and $\Lambda$ is a non-negative measure, the integral is non-negative, and the negative sign ensures the sum is $\le 0$. Thus $\gamma$ is conditionally negative definite.

**Necessity:**
Let $\gamma(h)$ be continuous and CND. By Lemma A.4.1, for each $h \in \mathbb{R}^d$, $2\gamma(h) = \int_{\mathbb{R}^d \setminus \{0\}} 2(1 - \cos(\langle \omega, h \rangle)) d\Lambda(\omega) + Q(h)$, where $Q(h)$ represents any singular contribution at $\omega = 0$. 

Expanding the integrand for small $\|\omega\| \to 0$:
$$
1 - \cos(\langle \omega, h \rangle) = \frac{\langle \omega, h \rangle^2}{2} + \mathcal{O}(\|\omega\|^4)
$$
For $\gamma(h)$ to be finite for every $h \in \mathbb{R}^d$, the measure $\Lambda$ must satisfy:
$$
\int_{\|\omega\| \le 1} \|\omega\|^2 d\Lambda(\omega) < \infty
$$
For large $\|\omega\| \ge 1$, since $1 - \cos(\langle \omega, h \rangle) \le 2$, integrability requires:
$$
\int_{\|\omega\| > 1} d\Lambda(\omega) < \infty
$$
Combining both regimes into a single algebraic criterion yields:
$$
\int_{\mathbb{R}^d \setminus \{0\}} \frac{\|\omega\|^2}{1 + \|\omega\|^2} d\Lambda(\omega) < \infty
$$
Any residual quadratic form $Q(h) = \lim_{\|\omega\| \to 0} \frac{1 - \cos(\langle \omega, h \rangle)}{\|\omega\|^2} \Lambda(\{\omega\})$ evaluates to a positive semi-definite quadratic form $h^T \mathbf{A} h$, completing the proof. $\blacksquare$

#### Corollary A.4.1 (Harmonic Representation of the Intrinsic Random Field)
An intrinsic random field of order 0 with semivariogram $\gamma(h) = \int_{\mathbb{R}^d \setminus \{0\}} (1 - \cos(\langle \omega, h \rangle)) d\Lambda(\omega)$ admits the stochastic integral representation:
$$
Z(s) = Z_0 + \int_{\mathbb{R}^d \setminus \{0\}} \left(e^{i \langle \omega, s \rangle} - 1\right) dZ(\omega)
$$
where $Z_0$ is a random variable representing the baseline level $Z(0)$, and $Z(\cdot)$ is an orthogonal stochastic measure on $\mathbb{R}^d \setminus \{0\}$ whose structure measure is $\Lambda$:
$$
\operatorname{E}\left[Z(A) \overline{Z(B)}\right] = \Lambda(A \cap B), \quad \forall A, B \in \mathcal{B}_0(\mathbb{R}^d \setminus \{0\})
$$
The subtraction of $1 = e^{i \langle \omega, 0 \rangle}$ in the integrand provides the necessary regularization at the origin $\omega = 0$, guaranteeing square-integrability:
$$
\operatorname{E}\left[(Z(s) - Z_0)^2\right] = \int_{\mathbb{R}^d \setminus \{0\}} \left|e^{i \langle \omega, s \rangle} - 1\right|^2 d\Lambda(\omega) = 2 \int_{\mathbb{R}^d \setminus \{0\}} (1 - \cos(\langle \omega, s \rangle)) d\Lambda(\omega) = 2\gamma(s) < \infty
$$