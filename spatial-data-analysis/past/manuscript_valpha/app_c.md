# Appendix C: Hilbert Space Projections and Block-Matrix Derivations

This appendix establishes the functional-analytic and linear algebraic foundations of spatial prediction theory. We provide complete, rigorous proofs of the Hilbert Space Orthogonal Projection Theorem in $L^2(\Omega, \mathcal{F}, P)$, derive closed-form block-matrix inversions via Schur complements for Ordinary Kriging, Universal Kriging, and Multivariate Cokriging systems, and prove the structural equivalence between covariance-based and semivariogram-based kriging formulations.

---

### C.1 The Orthogonal Projection Theorem in $L^2(\Omega, \mathcal{F}, P)$

Let $(\Omega, \mathcal{F}, P)$ be a complete probability space, and let $\mathcal{H} = L^2(\Omega, \mathcal{F}, P)$ be the real Hilbert space of equivalence classes of square-integrable random variables equipped with the inner product:
$$
\langle X, Y \rangle = \operatorname{E}[XY]
$$
and induced norm $\|X\|_{L^2} = \sqrt{\langle X, X \rangle} = \sqrt{\operatorname{E}[X^2]}$.

#### Lemma C.1.1 (Parallelogram Law)
For all $X, Y \in \mathcal{H}$:
$$
\|X + Y\|_{L^2}^2 + \|X - Y\|_{L^2}^2 = 2\|X\|_{L^2}^2 + 2\|Y\|_{L^2}^2
$$

*Proof.*
Expanding the inner products:
$$
\|X + Y\|_{L^2}^2 = \langle X + Y, X + Y \rangle = \|X\|_{L^2}^2 + 2\langle X, Y \rangle + \|Y\|_{L^2}^2
$$
$$
\|X - Y\|_{L^2}^2 = \langle X - Y, X - Y \rangle = \|X\|_{L^2}^2 - 2\langle X, Y \rangle + \|Y\|_{L^2}^2
$$
Adding the two equations cancels the cross-terms $2\langle X, Y \rangle$, yielding the result. $\blacksquare$

#### Theorem C.1.1 (Orthogonal Projection Theorem)
Let $\mathcal{M}$ be a closed linear subspace of $\mathcal{H}$. 
1. For every element $X \in \mathcal{H}$, there exists a unique element $\widehat{X} \in \mathcal{M}$ such that:
   $$
   \|X - \widehat{X}\|_{L^2} = \inf_{Y \in \mathcal{M}} \|X - Y\|_{L^2}
   $$
2. An element $\widehat{X} \in \mathcal{M}$ satisfies this minimization property if and only if the error vector $X - \widehat{X}$ is orthogonal to the subspace $\mathcal{M}$:
   $$
   \langle X - \widehat{X}, Y \rangle = 0, \quad \forall Y \in \mathcal{M}
   $$
3. The mapping $\Pi_{\mathcal{M}}: \mathcal{H} \to \mathcal{M}$ defined by $\Pi_{\mathcal{M}} X = \widehat{X}$ is a linear, self-adjoint, idempotent operator satisfying:
   $$
   \|\Pi_{\mathcal{M}} X\|_{L^2} \le \|X\|_{L^2}, \quad \forall X \in \mathcal{H}
   $$
   with $\|\Pi_{\mathcal{M}}\| = 1$ if $\mathcal{M} \ne \{0\}$.

*Proof.*
**Step 1: Existence of the Minimizer.**
Let $d = \inf_{Y \in \mathcal{M}} \|X - Y\|_{L^2}$. By definition of the infimum, there exists a sequence $\{Y_n\}_{n=1}^\infty \subset \mathcal{M}$ such that:
$$
\lim_{n \to \infty} \|X - Y_n\|_{L^2} = d
$$
We show that $\{Y_n\}$ is a Cauchy sequence in $\mathcal{H}$.
Apply the Parallelogram Law (Lemma C.1.1) to $u = X - Y_n$ and $v = X - Y_m$:
$$
\|u + v\|_{L^2}^2 + \|u - v\|_{L^2}^2 = 2\|u\|_{L^2}^2 + 2\|v\|_{L^2}^2
$$
Notice that:
$$
u - v = (X - Y_n) - (X - Y_m) = Y_m - Y_n
$$
$$
u + v = (X - Y_n) + (X - Y_m) = 2X - (Y_n + Y_m) = 2\left(X - \frac{Y_n + Y_m}{2}\right)
$$
Because $\mathcal{M}$ is a linear subspace, $\frac{Y_n + Y_m}{2} \in \mathcal{M}$, which implies:
$$
\left\|X - \frac{Y_n + Y_m}{2}\right\|_{L^2} \ge d \implies \|u + v\|_{L^2}^2 = 4\left\|X - \frac{Y_n + Y_m}{2}\right\|_{L^2}^2 \ge 4d^2
$$
Substituting this into the parallelogram identity:
$$
\|Y_m - Y_n\|_{L^2}^2 = 2\|X - Y_n\|_{L^2}^2 + 2\|X - Y_m\|_{L^2}^2 - 4\left\|X - \frac{Y_n + Y_m}{2}\right\|_{L^2}^2 \le 2\|X - Y_n\|_{L^2}^2 + 2\|X - Y_m\|_{L^2}^2 - 4d^2
$$
As $n, m \to \infty$, $\|X - Y_n\|_{L^2}^2 \to d^2$ and $\|X - Y_m\|_{L^2}^2 \to d^2$. Thus:
$$
\lim_{n, m \to \infty} \|Y_m - Y_n\|_{L^2}^2 \le 2d^2 + 2d^2 - 4d^2 = 0
$$
Therefore, $\{Y_n\}$ is a Cauchy sequence in $\mathcal{H}$. Because $\mathcal{H}$ is complete and $\mathcal{M}$ is closed, $\{Y_n\}$ converges to a limit $\widehat{X} \in \mathcal{M}$.
By continuity of the norm:
$$
\|X - \widehat{X}\|_{L^2} = \lim_{n \to \infty} \|X - Y_n\|_{L^2} = d
$$
which establishes existence.

**Step 2: Uniqueness of the Minimizer.**
Suppose there exists another minimizer $\widehat{X}' \in \mathcal{M}$ such that $\|X - \widehat{X}'\|_{L^2} = d$.
Apply the parallelogram law to $X - \widehat{X}$ and $X - \widehat{X}'$:
$$
\|\widehat{X} - \widehat{X}'\|_{L^2}^2 = 2\|X - \widehat{X}\|_{L^2}^2 + 2\|X - \widehat{X}'\|_{L^2}^2 - 4\left\|X - \frac{\widehat{X} + \widehat{X}'}{2}\right\|_{L^2}^2
$$
Since $\frac{\widehat{X} + \widehat{X}'}{2} \in \mathcal{M}$, $\left\|X - \frac{\widehat{X} + \widehat{X}'}{2}\right\|_{L^2}^2 \ge d^2$. Thus:
$$
\|\widehat{X} - \widehat{X}'\|_{L^2}^2 \le 2d^2 + 2d^2 - 4d^2 = 0 \implies \|\widehat{X} - \widehat{X}'\|_{L^2} = 0 \implies \widehat{X} = \widehat{X}'
$$
This establishes uniqueness.

**Step 3: Equivalence with Orthogonality.**
($\implies$) Suppose $\widehat{X} \in \mathcal{M}$ minimizes $\|X - Y\|_{L^2}$ over $Y \in \mathcal{M}$.
Let $Y \in \mathcal{M}$ be arbitrary, and let $t \in \mathbb{R}$. Then $\widehat{X} + tY \in \mathcal{M}$.
Define the quadratic function:
$$
g(t) = \|X - (\widehat{X} + tY)\|_{L^2}^2 = \|(X - \widehat{X}) - tY\|_{L^2}^2 = \|X - \widehat{X}\|_{L^2}^2 - 2t \langle X - \widehat{X}, Y \rangle + t^2 \|Y\|_{L^2}^2
$$
Because $\widehat{X}$ minimizes the distance, $g(t)$ achieves its global minimum at $t = 0$. Therefore, its derivative at $t = 0$ must vanish:
$$
g'(0) = -2\langle X - \widehat{X}, Y \rangle = 0 \implies \langle X - \widehat{X}, Y \rangle = 0
$$
Since $Y \in \mathcal{M}$ was arbitrary, $(X - \widehat{X}) \perp \mathcal{M}$.

($\impliedby$) Conversely, suppose $\widehat{X} \in \mathcal{M}$ satisfies $\langle X - \widehat{X}, Y \rangle = 0$ for all $Y \in \mathcal{M}$.
For any $Y \in \mathcal{M}$, write $X - Y = (X - \widehat{X}) + (\widehat{X} - Y)$.
Because $\widehat{X} - Y \in \mathcal{M}$, we have $(X - \widehat{X}) \perp (\widehat{X} - Y)$.
By the Pythagorean theorem:
$$
\|X - Y\|_{L^2}^2 = \|(X - \widehat{X}) + (\widehat{X} - Y)\|_{L^2}^2 = \|X - \widehat{X}\|_{L^2}^2 + \|\widehat{X} - Y\|_{L^2}^2 \ge \|X - \widehat{X}\|_{L^2}^2
$$
with equality if and only if $\|\widehat{X} - Y\|_{L^2} = 0$, i.e., $Y = \widehat{X}$. Thus $\widehat{X}$ is the unique minimizer.

**Step 4: Operator Properties.**
1. **Linearity:** Let $X_1, X_2 \in \mathcal{H}$ and $\alpha, \beta \in \mathbb{R}$.
   For any $Y \in \mathcal{M}$:
   $$
   \langle (\alpha X_1 + \beta X_2) - (\alpha \Pi_{\mathcal{M}} X_1 + \beta \Pi_{\mathcal{M}} X_2), Y \rangle = \alpha \langle X_1 - \Pi_{\mathcal{M}} X_1, Y \rangle + \beta \langle X_2 - \Pi_{\mathcal{M}} X_2, Y \rangle = \alpha(0) + \beta(0) = 0
   $$
   Since $\alpha \Pi_{\mathcal{M}} X_1 + \beta \Pi_{\mathcal{M}} X_2 \in \mathcal{M}$, by uniqueness of the projection, $\Pi_{\mathcal{M}}(\alpha X_1 + \beta X_2) = \alpha \Pi_{\mathcal{M}} X_1 + \beta \Pi_{\mathcal{M}} X_2$.
2. **Idempotence:** If $X \in \mathcal{M}$, then $X - X = 0 \perp \mathcal{M}$, so $\Pi_{\mathcal{M}} X = X$. Hence $\Pi_{\mathcal{M}}^2 = \Pi_{\mathcal{M}}$.
3. **Self-Adjointness:** For any $X, W \in \mathcal{H}$:
   $$
   \langle \Pi_{\mathcal{M}} X, W \rangle = \langle \Pi_{\mathcal{M}} X, \Pi_{\mathcal{M}} W + (W - \Pi_{\mathcal{M}} W) \rangle = \langle \Pi_{\mathcal{M}} X, \Pi_{\mathcal{M}} W \rangle + 0 = \langle X - (X - \Pi_{\mathcal{M}} X), \Pi_{\mathcal{M}} W \rangle = \langle X, \Pi_{\mathcal{M}} W \rangle
   $$
4. **Norm:** By the Pythagorean identity, $\|X\|_{L^2}^2 = \|\Pi_{\mathcal{M}} X\|_{L^2}^2 + \|X - \Pi_{\mathcal{M}} X\|_{L^2}^2 \ge \|\Pi_{\mathcal{M}} X\|_{L^2}^2$. Thus $\|\Pi_{\mathcal{M}} X\|_{L^2} \le \|X\|_{L^2}$, so $\|\Pi_{\mathcal{M}}\| \le 1$. If $\mathcal{M} \ne \{0\}$, choose $X \in \mathcal{M} \setminus \{0\}$, then $\|\Pi_{\mathcal{M}} X\|_{L^2} = \|X\|_{L^2}$, so $\|\Pi_{\mathcal{M}}\| = 1$. $\blacksquare$

---

### C.2 Block-Matrix Inversion Derivations via Schur Complements

In spatial prediction, the constrained optimization problem generates partitioned symmetric linear systems. We derive the explicit matrix inverses using the algebraic theory of Schur complements.

#### Lemma C.2.1 (Schur Complement Inversion Lemma)
Let $\mathbf{M}$ be a partitioned matrix of the form:
$$
\mathbf{M} = \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^T & \mathbf{D} \end{bmatrix}
$$
where $\mathbf{A} \in \mathbb{R}^{n \times n}$ is non-singular and $\mathbf{D} \in \mathbb{R}^{p \times p}$.
1. The **Schur complement** of $\mathbf{A}$ in $\mathbf{M}$ is the $p \times p$ matrix:
   $$
   \mathbf{S} = \mathbf{D} - \mathbf{B}^T \mathbf{A}^{-1} \mathbf{B}
   $$
2. $\mathbf{M}$ is invertible if and only if $\mathbf{S}$ is invertible.
3. When $\mathbf{S}$ is invertible, the inverse $\mathbf{M}^{-1}$ is given by:
   $$
   \mathbf{M}^{-1} = \begin{bmatrix} \mathbf{A}^{-1} + \mathbf{A}^{-1}\mathbf{B}\mathbf{S}^{-1}\mathbf{B}^T\mathbf{A}^{-1} & -\mathbf{A}^{-1}\mathbf{B}\mathbf{S}^{-1} \\ -\mathbf{S}^{-1}\mathbf{B}^T\mathbf{A}^{-1} & \mathbf{S}^{-1} \end{bmatrix}
   $$

*Proof.*
We perform the block LDU (lower-diagonal-upper) factorization of $\mathbf{M}$.
Consider the matrix identity:
$$
\begin{bmatrix} \mathbf{I}_n & \mathbf{0} \\ -\mathbf{B}^T\mathbf{A}^{-1} & \mathbf{I}_p \end{bmatrix} \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^T & \mathbf{D} \end{bmatrix} \begin{bmatrix} \mathbf{I}_n & -\mathbf{A}^{-1}\mathbf{B} \\ \mathbf{0} & \mathbf{I}_p \end{bmatrix} = \begin{bmatrix} \mathbf{A} & \mathbf{0} \\ \mathbf{0} & \mathbf{D} - \mathbf{B}^T\mathbf{A}^{-1}\mathbf{B} \end{bmatrix} = \begin{bmatrix} \mathbf{A} & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}
$$
Let $\mathbf{L} = \begin{bmatrix} \mathbf{I}_n & \mathbf{0} \\ \mathbf{B}^T\mathbf{A}^{-1} & \mathbf{I}_p \end{bmatrix}$ and $\mathbf{U} = \begin{bmatrix} \mathbf{I}_n & \mathbf{A}^{-1}\mathbf{B} \\ \mathbf{0} & \mathbf{I}_p \end{bmatrix}$. Notice that $\mathbf{L}^{-1} = \begin{bmatrix} \mathbf{I}_n & \mathbf{0} \\ -\mathbf{B}^T\mathbf{A}^{-1} & \mathbf{I}_p \end{bmatrix}$ and $\mathbf{U}^{-1} = \begin{bmatrix} \mathbf{I}_n & -\mathbf{A}^{-1}\mathbf{B} \\ \mathbf{0} & \mathbf{I}_p \end{bmatrix}$.
Therefore:
$$
\mathbf{M} = \mathbf{L} \begin{bmatrix} \mathbf{A} & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix} \mathbf{U}
$$
By the product rule for determinants:
$$
\det(\mathbf{M}) = \det(\mathbf{L}) \det\left(\begin{bmatrix} \mathbf{A} & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}\right) \det(\mathbf{U}) = 1 \cdot (\det(\mathbf{A})\det(\mathbf{S})) \cdot 1 = \det(\mathbf{A})\det(\mathbf{S})
$$
Since $\mathbf{A}$ is non-singular ($\det(\mathbf{A}) \ne 0$), $\mathbf{M}$ is invertible if and only if $\det(\mathbf{S}) \ne 0$, i.e., $\mathbf{S}$ is invertible.

To obtain the inverse, invert the LDU decomposition:
$$
\begin{aligned}
\mathbf{M}^{-1} &= \mathbf{U}^{-1} \begin{bmatrix} \mathbf{A}^{-1} & \mathbf{0} \\ \mathbf{0} & \mathbf{S}^{-1} \end{bmatrix} \mathbf{L}^{-1} \\
&= \begin{bmatrix} \mathbf{I}_n & -\mathbf{A}^{-1}\mathbf{B} \\ \mathbf{0} & \mathbf{I}_p \end{bmatrix} \begin{bmatrix} \mathbf{A}^{-1} & \mathbf{0} \\ \mathbf{0} & \mathbf{S}^{-1} \end{bmatrix} \begin{bmatrix} \mathbf{I}_n & \mathbf{0} \\ -\mathbf{B}^T\mathbf{A}^{-1} & \mathbf{I}_p \end{bmatrix} \\
&= \begin{bmatrix} \mathbf{A}^{-1} & -\mathbf{A}^{-1}\mathbf{B}\mathbf{S}^{-1} \\ \mathbf{0} & \mathbf{S}^{-1} \end{bmatrix} \begin{bmatrix} \mathbf{I}_n & \mathbf{0} \\ -\mathbf{B}^T\mathbf{A}^{-1} & \mathbf{I}_p \end{bmatrix} \\
&= \begin{bmatrix} \mathbf{A}^{-1} + \mathbf{A}^{-1}\mathbf{B}\mathbf{S}^{-1}\mathbf{B}^T\mathbf{A}^{-1} & -\mathbf{A}^{-1}\mathbf{B}\mathbf{S}^{-1} \\ -\mathbf{S}^{-1}\mathbf{B}^T\mathbf{A}^{-1} & \mathbf{S}^{-1} \end{bmatrix}
\end{aligned}
$$
This completes the proof. $\blacksquare$

---

#### C.2.1 Inversion of the Ordinary Kriging Block Matrix
The Ordinary Kriging system (Theorem 5.3.1) in block matrix form is:
$$
\begin{bmatrix} \mathbf{\Sigma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ -\mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
$$
Here $\mathbf{A} = \mathbf{\Sigma} \in \mathbb{R}^{n \times n}$, $\mathbf{B} = \mathbf{1} \in \mathbb{R}^{n \times 1}$, and $\mathbf{D} = 0 \in \mathbb{R}^{1 \times 1}$.

#### Theorem C.2.1 (Explicit Ordinary Kriging Inverse and Solution)
1. The scalar Schur complement is:
   $$
   S = 0 - \mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1} = -\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1} < 0
   $$
   Its inverse is $S^{-1} = -\frac{1}{\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1}}$.
2. The inverse of the Ordinary Kriging augmented matrix is:
   $$
   \begin{bmatrix} \mathbf{\Sigma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix}^{-1} = \begin{bmatrix} \mathbf{\Sigma}^{-1} - \frac{\mathbf{\Sigma}^{-1}\mathbf{1}\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} & \frac{\mathbf{\Sigma}^{-1}\mathbf{1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \\ \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} & -\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \end{bmatrix}
   $$
3. The Ordinary Kriging weight vector $\boldsymbol{\lambda}_{\text{OK}}$ is:
   $$
   \boldsymbol{\lambda}_{\text{OK}} = \mathbf{\Sigma}^{-1} \mathbf{c}_0 + \left(\frac{1 - \mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{c}_0}{\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1}}\right) \mathbf{\Sigma}^{-1} \mathbf{1}
   $$
4. The Lagrange multiplier $\mu_{\text{OK}}$ is:
   $$
   \mu_{\text{OK}} = \frac{1 - \mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{c}_0}{\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1}}
   $$

*Proof.*
1. Direct application of Lemma C.2.1 with $\mathbf{D} = 0$ gives $S = -\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1}$. Because $\mathbf{\Sigma}$ is positive definite, $\mathbf{\Sigma}^{-1}$ is positive definite, so $\mathbf{1}^T \mathbf{\Sigma}^{-1} \mathbf{1} > 0$, guaranteeing that $S < 0$ is non-zero and invertible.
2. Substituting $\mathbf{A} = \mathbf{\Sigma}$, $\mathbf{B} = \mathbf{1}$, and $S^{-1} = -\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$ into the Schur formula:
   - Upper-left block: $\mathbf{\Sigma}^{-1} + \mathbf{\Sigma}^{-1}\mathbf{1}\left(-\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}\right)\mathbf{1}^T\mathbf{\Sigma}^{-1} = \mathbf{\Sigma}^{-1} - \frac{\mathbf{\Sigma}^{-1}\mathbf{1}\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$.
   - Upper-right block: $-\mathbf{\Sigma}^{-1}\mathbf{1}\left(-\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}\right) = \frac{\mathbf{\Sigma}^{-1}\mathbf{1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$.
   - Lower-left block: $-\left(-\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}\right)\mathbf{1}^T\mathbf{\Sigma}^{-1} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$.
   - Lower-right block: $S^{-1} = -\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}$.
3. Multiply the inverse matrix by the right-hand side vector $(\mathbf{c}_0^T, 1)^T$:
   $$
   \begin{bmatrix} \boldsymbol{\lambda}_{\text{OK}} \\ -\mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \mathbf{\Sigma}^{-1} - \frac{\mathbf{\Sigma}^{-1}\mathbf{1}\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} & \frac{\mathbf{\Sigma}^{-1}\mathbf{1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \\ \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} & -\frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \end{bmatrix} \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
   $$
   Computing the top component:
   $$
   \begin{aligned}
   \boldsymbol{\lambda}_{\text{OK}} &= \left(\mathbf{\Sigma}^{-1} - \frac{\mathbf{\Sigma}^{-1}\mathbf{1}\mathbf{1}^T\mathbf{\Sigma}^{-1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}\right)\mathbf{c}_0 + \frac{\mathbf{\Sigma}^{-1}\mathbf{1}}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \cdot 1 \\
   &= \mathbf{\Sigma}^{-1}\mathbf{c}_0 - \frac{(\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0)}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \mathbf{\Sigma}^{-1}\mathbf{1} + \frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \mathbf{\Sigma}^{-1}\mathbf{1} \\
   &= \mathbf{\Sigma}^{-1}\mathbf{c}_0 + \left(\frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}\right) \mathbf{\Sigma}^{-1}\mathbf{1}
   \end{aligned}
   $$
4. Computing the bottom scalar component:
   $$
   -\mu_{\text{OK}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} - \frac{1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} = \frac{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}} \implies \mu_{\text{OK}} = \frac{1 - \mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}^T\mathbf{\Sigma}^{-1}\mathbf{1}}
   $$
   This completes the derivation. $\blacksquare$

---

#### C.2.2 Inversion of the Universal Kriging Block Matrix
The Universal Kriging system (Theorem 6.3.1) in block matrix form is:
$$
\begin{bmatrix} \mathbf{\Sigma} & \mathbf{X} \\ \mathbf{X}^T & \mathbf{0} \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \boldsymbol{\mu}_{\text{UK}} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ \mathbf{x}_0 \end{bmatrix}
$$
Here $\mathbf{A} = \mathbf{\Sigma} \in \mathbb{R}^{n \times n}$, $\mathbf{B} = \mathbf{X} \in \mathbb{R}^{n \times p}$, and $\mathbf{D} = \mathbf{0} \in \mathbb{R}^{p \times p}$.

#### Theorem C.2.2 (Explicit Universal Kriging Inverse and Solution)
1. The $p \times p$ Schur complement matrix is:
   $$
   \mathbf{S} = \mathbf{0} - \mathbf{X}^T \mathbf{\Sigma}^{-1} \mathbf{X} = -\left(\mathbf{X}^T \mathbf{\Sigma}^{-1} \mathbf{X}\right)
   $$
   Because $\mathbf{X}$ has full column rank $p$ and $\mathbf{\Sigma}$ is positive definite, $\mathbf{S}$ is strictly negative definite, and its inverse is:
   $$
   \mathbf{S}^{-1} = -\left(\mathbf{X}^T \mathbf{\Sigma}^{-1} \mathbf{X}\right)^{-1}
   $$
2. The inverse of the Universal Kriging augmented matrix is:
   $$
   \begin{bmatrix} \mathbf{\Sigma} & \mathbf{X} \\ \mathbf{X}^T & \mathbf{0} \end{bmatrix}^{-1} = \begin{bmatrix} \mathbf{\Sigma}^{-1} - \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1} & \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \\ \left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1} & -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \end{bmatrix}
   $$
3. The Universal Kriging weight vector $\boldsymbol{\lambda}_{\text{UK}}$ is:
   $$
   \boldsymbol{\lambda}_{\text{UK}} = \mathbf{\Sigma}^{-1}\mathbf{c}_0 + \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \left(\mathbf{x}_0 - \mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0\right)
   $$
4. The Lagrange multiplier vector $\boldsymbol{\mu}_{\text{UK}}$ is:
   $$
   \boldsymbol{\mu}_{\text{UK}} = -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \left(\mathbf{x}_0 - \mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0\right)
   $$

*Proof.*
Apply Lemma C.2.1 directly with $\mathbf{A} = \mathbf{\Sigma}$, $\mathbf{B} = \mathbf{X}$, $\mathbf{D} = \mathbf{0}$, and $\mathbf{S}^{-1} = -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}$:
- Block $(1, 1)$: $\mathbf{\Sigma}^{-1} + \mathbf{\Sigma}^{-1}\mathbf{X}\left(-\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\right)\mathbf{X}^T\mathbf{\Sigma}^{-1} = \mathbf{\Sigma}^{-1} - \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1}$.
- Block $(1, 2)$: $-\mathbf{\Sigma}^{-1}\mathbf{X}\left(-\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\right) = \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}$.
- Block $(2, 1)$: $-\left(-\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\right)\mathbf{X}^T\mathbf{\Sigma}^{-1} = \left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1}$.
- Block $(2, 2)$: $\mathbf{S}^{-1} = -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}$.

Multiplying this block inverse matrix by the vector $(\mathbf{c}_0^T, \mathbf{x}_0^T)^T$:
$$
\begin{bmatrix} \boldsymbol{\lambda}_{\text{UK}} \\ \boldsymbol{\mu}_{\text{UK}} \end{bmatrix} = \begin{bmatrix} \mathbf{\Sigma}^{-1} - \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1} & \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \\ \left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1} & -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \end{bmatrix} \begin{bmatrix} \mathbf{c}_0 \\ \mathbf{x}_0 \end{bmatrix}
$$
Top block:
$$
\begin{aligned}
\boldsymbol{\lambda}_{\text{UK}} &= \left(\mathbf{\Sigma}^{-1} - \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1}\right)\mathbf{c}_0 + \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{x}_0 \\
&= \mathbf{\Sigma}^{-1}\mathbf{c}_0 + \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \left(\mathbf{x}_0 - \mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0\right)
\end{aligned}
$$
Bottom block:
$$
\boldsymbol{\mu}_{\text{UK}} = \left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0 - \left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{x}_0 = -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \left(\mathbf{x}_0 - \mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{c}_0\right)
$$
This establishes the explicit expressions. $\blacksquare$

---

#### C.2.3 Inversion of the Multivariate Cokriging Block Matrix
In standard Ordinary Cokriging (Theorem 7.3.2), the system matrix is:
$$
\mathbf{M}_{\text{OCK}} = \begin{bmatrix} \mathbf{\Sigma} & \mathbf{X} \\ \mathbf{X}^T & \mathbf{0} \end{bmatrix} \in \mathbb{R}^{(N+p) \times (N+p)}
$$
where $N = \sum_{i=1}^p n_i$, $\mathbf{\Sigma} \in \mathbb{R}^{N \times N}$ is the block cross-covariance matrix, and $\mathbf{X} = \operatorname{diag}(\mathbf{1}_{n_1}, \dots, \mathbf{1}_{n_p}) \in \mathbb{R}^{N \times p}$.

#### Theorem C.2.3 (Multivariate Cokriging Inversion)
1. The $p \times p$ Schur complement of $\mathbf{\Sigma}$ in $\mathbf{M}_{\text{OCK}}$ is:
   $$
   \mathbf{S} = -\mathbf{X}^T \mathbf{\Sigma}^{-1} \mathbf{X}
   $$
   Its $(i, j)$-th entry is:
   $$
   [\mathbf{S}]_{ij} = -\mathbf{1}_{n_i}^T [\mathbf{\Sigma}^{-1}]_{ij} \mathbf{1}_{n_j}
   $$
   where $[\mathbf{\Sigma}^{-1}]_{ij} \in \mathbb{R}^{n_i \times n_j}$ is the $(i, j)$-th sub-block of the inverse covariance matrix $\mathbf{\Sigma}^{-1}$.
2. The full block inverse is:
   $$
   \mathbf{M}_{\text{OCK}}^{-1} = \begin{bmatrix} \mathbf{\Sigma}^{-1} - \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1} & \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \\ \left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{\Sigma}^{-1} & -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \end{bmatrix}
   $$
3. The cokriging weights and Lagrange multipliers for target variable $Z_1(s_0)$ are:
   $$
   \boldsymbol{\lambda} = \mathbf{\Sigma}^{-1} \mathbf{c}_{0,1} + \mathbf{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \left(\mathbf{e}_1 - \mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{c}_{0,1}\right)
   $$
   $$
   \boldsymbol{\mu}_{\text{OCK}} = -\left(\mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{X}\right)^{-1} \left(\mathbf{e}_1 - \mathbf{X}^T\mathbf{\Sigma}^{-1}\mathbf{c}_{0,1}\right)
   $$

*Proof.*
The proof follows from Theorem C.2.2 by replacing the UK design matrix $\mathbf{X}$ with the block-diagonal indicator matrix $\mathbf{X} = \operatorname{diag}(\mathbf{1}_{n_1}, \dots, \mathbf{1}_{n_p})$ and replacing the target basis vector $\mathbf{x}_0$ with the unit coordinate vector $\mathbf{e}_1 = (1, 0, \dots, 0)^T \in \mathbb{R}^p$. $\blacksquare$

---

### C.3 Formal Equivalence of Covariance and Semivariogram Kriging Systems

In the theoretical literature and empirical applications, Ordinary Kriging is formulated either in terms of the covariance function $C(h)$ or the semivariogram $\gamma(h)$. Here we provide the complete algebraic proof of their exact equivalence under second-order stationarity, and establish the conditions under which the semivariogram formulation generalizes to non-covariance processes.

#### Theorem C.3.1 (Algebraic Equivalence of OK Systems)
Let $\{Z(s) : s \in \mathcal{D}\}$ be a second-order stationary random field with covariance function $C(h)$ and semivariogram $\gamma(h) = C(0) - C(h)$.
Let:
- $\mathbf{\Sigma} = [C(s_i - s_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}$ and $\mathbf{c}_0 = (C(s_0 - s_1), \dots, C(s_0 - s_n))^T \in \mathbb{R}^n$.
- $\mathbf{\Gamma} = [\gamma(s_i - s_j)]_{i,j=1}^n \in \mathbb{R}^{n \times n}$ and $\boldsymbol{\gamma}_0 = (\gamma(s_0 - s_1), \dots, \gamma(s_0 - s_n))^T \in \mathbb{R}^n$.
- $\mathbf{J} = \mathbf{1}\mathbf{1}^T \in \mathbb{R}^{n \times n}$ be the all-ones matrix.

Consider the two linear systems:
$$
\text{(Covariance System):} \quad \begin{bmatrix} \mathbf{\Sigma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda}^{(C)} \\ -\mu_{\text{OK}}^{(C)} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
$$
$$
\text{(Semivariogram System):} \quad \begin{bmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda}^{(\gamma)} \\ \mu_{\text{OK}}^{(\gamma)} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}
$$
Then:
1. The weight vectors are identical:
   $$
   \boldsymbol{\lambda}^{(C)} \equiv \boldsymbol{\lambda}^{(\gamma)}
   $$
2. The Lagrange multipliers satisfy the transformation:
   $$
   \mu_{\text{OK}}^{(\gamma)} = C(0) - \mu_{\text{OK}}^{(C)} \quad \left(\text{or } \mu_{\text{OK}}^{(\gamma)} = -\mu_{\text{OK}}^{(C)} \text{ if using the system with } -\mu \mathbf{1}\right)
   $$
3. The prediction variances evaluated by the two systems are identically equal:
   $$
   \sigma_{\text{OK}}^2(s_0) = C(0) - (\boldsymbol{\lambda}^{(C)})^T \mathbf{c}_0 + \mu_{\text{OK}}^{(C)} \equiv (\boldsymbol{\lambda}^{(\gamma)})^T \boldsymbol{\gamma}_0 + \mu_{\text{OK}}^{(\gamma)}
   $$

*Proof.*
**Step 1: Matrix Representation.**
Under the identity $\gamma(h) = C(0) - C(h)$, we have:
$$
\mathbf{\Gamma} = C(0)\mathbf{J} - \mathbf{\Sigma} = C(0)\mathbf{1}\mathbf{1}^T - \mathbf{\Sigma}
$$
$$
\boldsymbol{\gamma}_0 = C(0)\mathbf{1} - \mathbf{c}_0
$$
Substitute these identities into the semivariogram system:
$$
\begin{bmatrix} C(0)\mathbf{1}\mathbf{1}^T - \mathbf{\Sigma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda}^{(\gamma)} \\ \mu_{\text{OK}}^{(\gamma)} \end{bmatrix} = \begin{bmatrix} C(0)\mathbf{1} - \mathbf{c}_0 \\ 1 \end{bmatrix}
$$
The first matrix equation is:
$$
\left(C(0)\mathbf{1}\mathbf{1}^T - \mathbf{\Sigma}\right) \boldsymbol{\lambda}^{(\gamma)} + \mu_{\text{OK}}^{(\gamma)} \mathbf{1} = C(0)\mathbf{1} - \mathbf{c}_0
$$
Expand the left side:
$$
C(0)\mathbf{1} \left(\mathbf{1}^T \boldsymbol{\lambda}^{(\gamma)}\right) - \mathbf{\Sigma} \boldsymbol{\lambda}^{(\gamma)} + \mu_{\text{OK}}^{(\gamma)} \mathbf{1} = C(0)\mathbf{1} - \mathbf{c}_0
$$
The second equation of the system is the constraint $\mathbf{1}^T \boldsymbol{\lambda}^{(\gamma)} = 1$. Substituting this constraint into the first term:
$$
C(0)\mathbf{1}(1) - \mathbf{\Sigma} \boldsymbol{\lambda}^{(\gamma)} + \mu_{\text{OK}}^{(\gamma)} \mathbf{1} = C(0)\mathbf{1} - \mathbf{c}_0
$$
Subtract $C(0)\mathbf{1}$ from both sides:
$$
-\mathbf{\Sigma} \boldsymbol{\lambda}^{(\gamma)} + \mu_{\text{OK}}^{(\gamma)} \mathbf{1} = -\mathbf{c}_0
$$
Multiplying the entire equation by $-1$:
$$
\mathbf{\Sigma} \boldsymbol{\lambda}^{(\gamma)} - \mu_{\text{OK}}^{(\gamma)} \mathbf{1} = \mathbf{c}_0
$$
Comparing this directly with the first equation of the covariance system:
$$
\mathbf{\Sigma} \boldsymbol{\lambda}^{(C)} - \mu_{\text{OK}}^{(C)} \mathbf{1} = \mathbf{c}_0
$$
Because the augmented matrix is invertible, the solution is unique. Hence:
$$
\boldsymbol{\lambda}^{(\gamma)} \equiv \boldsymbol{\lambda}^{(C)} \quad \text{and} \quad \mu_{\text{OK}}^{(\gamma)} = \mu_{\text{OK}}^{(C)} \quad \text{(when both write the term as } -\mu\mathbf{1}\text{)}
$$
If the semivariogram system is written with $+\mu_{\text{OK}}^{(\gamma)}\mathbf{1} = \boldsymbol{\gamma}_0$, then substituting $-\mathbf{\Sigma} \boldsymbol{\lambda} + \mu \mathbf{1} = -\mathbf{c}_0$ yields the corresponding sign change. In either case, the weight vectors $\boldsymbol{\lambda}$ are mathematically identical.

**Step 2: Equivalence of the Prediction Variances.**
Now evaluate the semivariogram prediction variance using $\boldsymbol{\lambda}^{(\gamma)} = \boldsymbol{\lambda}$ and $\boldsymbol{\gamma}_0 = C(0)\mathbf{1} - \mathbf{c}_0$:
$$
\begin{aligned}
\sigma_{\text{OK},(\gamma)}^2(s_0) &= \boldsymbol{\lambda}^T \boldsymbol{\gamma}_0 + \mu_{\text{OK}} \\
&= \boldsymbol{\lambda}^T \left(C(0)\mathbf{1} - \mathbf{c}_0\right) + \mu_{\text{OK}} \\
&= C(0) (\boldsymbol{\lambda}^T \mathbf{1}) - \boldsymbol{\lambda}^T \mathbf{c}_0 + \mu_{\text{OK}}
\end{aligned}
$$
Because $\boldsymbol{\lambda}^T \mathbf{1} = 1$:
$$
\sigma_{\text{OK},(\gamma)}^2(s_0) = C(0)(1) - \boldsymbol{\lambda}^T \mathbf{c}_0 + \mu_{\text{OK}} = C(0) - \boldsymbol{\lambda}^T \mathbf{c}_0 + \mu_{\text{OK}} = \sigma_{\text{OK},(C)}^2(s_0)
$$
The two variances are identically equal. $\blacksquare$

#### Corollary C.3.2 (Extension to Intrinsically Stationary Processes)
While the covariance system requires that $C(0) = \operatorname{Var}(Z(s)) < \infty$ exists, the semivariogram system:
$$
\begin{bmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ \mu_{\text{OK}} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}
$$
is well-defined and strictly solvable for all intrinsically stationary random fields where $\gamma(h)$ is conditionally negative definite of order 0, even when $\lim_{\|h\| \to \infty} \gamma(h) = \infty$ (such as power semivariograms $\gamma(h) = \theta \|h\|^\alpha$ with $0 < \alpha < 2$). For such fields, no covariance function $C(h)$ exists, rendering the semivariogram formulation the fundamental general representation of Ordinary Kriging.
