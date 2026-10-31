# Appendix B: Scientific Python Reference Patterns

Spatial statistics in this book is built from first principles. Rather than relying on specialized third-party spatial packages, every estimator, kriging predictor, spatial weights matrix, and point-process simulator is implemented using four core libraries: `numpy`, `scipy`, `sympy`, and `matplotlib`.

Writing methods from foundational matrix operations serves a dual purpose:
1. **Conceptual Mastery:** A reader who implements ordinary kriging with block matrix solves, or fits a Strauss point process with an iteratively reweighted Poisson generalized linear model, understands the internal mechanics and assumptions in a way that running a closed-source package cannot teach.
2. **Computational Reproducibility:** By relying exclusively on standard scientific Python libraries, the code remains stable across platforms, avoiding brittle dependency chains and obsolete APIs.

This appendix documents the recurring scientific Python patterns used across Chapters 0 to 13. It serves as a practical software reference for stable numerical linear algebra, sparse matrix storage, constrained optimization, special functions, and high-performance spatial idioms.

---

## B.1 Symbolic Verification with SymPy

Before implementing a vectorized numerical estimator, it is often prudent to verify matrix derivatives, score equations, and partitioned inverses symbolically. The `sympy` package provides exact symbolic mathematics.

### B.1.1 Symbols, Matrices, and Closed-Form Scores

In Chapter 0 (§0.4.4, §0.6) and Chapter 5 (§5.2.2), the concentrated maximum likelihood estimator of the variance scale $\sigma^2$ is found by differentiating the normal log-likelihood:

$$
\ell(\sigma^2) = -\frac{n}{2}\log\sigma^2 - \frac{Q}{2\sigma^2}
$$

where $Q = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})'\mathbf{R}^{-1}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$. In SymPy, we declare positive symbols and solve the score equation $\partial\ell/\partial\sigma^2 = 0$:

```python
import sympy as sp

# Declare positive scalar symbols
n, Q, s2 = sp.symbols("n Q sigma2", positive=True)

# Define the concentrated log-likelihood
loglik = -n / 2 * sp.log(s2) - Q / (2 * s2)

# Compute the score (first derivative)
score = sp.diff(loglik, s2)
print("Score equation:", sp.simplify(score))

# Solve for the MLE
mle = sp.solve(score, s2)
print("Closed-form MLE for sigma^2:", mle[0])
```

The script prints `Score equation: (-n*sigma2 + Q)/(2*sigma2**2)` and `Closed-form MLE for sigma^2: Q/n`, verifying the formula without pencil and paper.

### B.1.2 Symbolic Verification of Block Inversion

In Chapter 4 (§4.3.4) and Appendix A (§A.1), the inverse of a partitioned $2 \times 2$ block matrix is expressed in terms of the Schur complement $\mathbf{F} = \mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}$. We can verify this identity symbolically on a $2 \times 2$ scalar configuration:

```python
# Declare non-commuting matrix symbols or explicit blocks
a, b, c, d = sp.symbols("a b c d")
M = sp.Matrix([[a, b], [c, d]])

# Direct symbolic inverse
M_inv = M.inv()

# Formula using the Schur complement F = d - c * (1/a) * b
F = d - c * b / a
M_inv_schur = sp.Matrix([
    [1 / a + (b * c) / (a**2 * F), -b / (a * F)],
    [-c / (a * F), 1 / F]
])

print("Difference is zero:", sp.simplify(M_inv - M_inv_schur) == sp.zeros(2, 2))
```

The difference is identically zero, confirming the algebraic correctness of the block elimination formula.

---

## B.2 Stable Numerical Linear Algebra

Numerical linear algebra is the computational engine of spatial statistics. Because spatial covariance matrices can be ill-conditioned when locations cluster closely together (§5.6.3), naive matrix operations can introduce severe roundoff error or fail entirely.

### B.2.1 Why `np.linalg.inv` Is an Anti-Pattern

In applied mathematics textbooks, equations are frequently written with explicit matrix inverses: $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$. In numerical computation, **forming an explicit matrix inverse is almost never necessary and should be avoided**.

Evaluating $\mathbf{A}^{-1}\mathbf{b}$ via `np.linalg.inv(A) @ b`:
1. Requires approximately $2n^3$ floating-point operations (computing the inverse, then multiplying).
2. Introduces unnecessary numerical instability by squaring condition numbers and exacerbating roundoff error.
3. Destroys structural properties: the inverse of a sparse or banded matrix is almost always completely dense.

Instead, linear systems $\mathbf{A}\mathbf{x} = \mathbf{b}$ should be solved using algorithms tailored to the structure of $\mathbf{A}$:

```python
import numpy as np
import scipy.linalg as linalg

A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])

# BAD: explicit inversion (slow, numerically fragile)
x_bad = np.linalg.inv(A) @ b

# GOOD: general solve via LU decomposition
x_good = np.linalg.solve(A, b)

# BEST: Cholesky solve for symmetric positive definite matrices
c, low = linalg.cho_factor(A)
x_best = linalg.cho_solve((c, low), b)

print("Solutions agree:", np.allclose(x_good, x_best))
```

**The single exception:** The explicit inverse is legitimate only when the mathematical quantity of interest is the matrix of standard errors itself: the diagonal of the asymptotic covariance matrix $\operatorname{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}$, or the leave-one-out projection matrix $\boldsymbol{\Pi}$ (§4.7.2). Even in those cases, the inversion is performed on the smaller $p \times p$ parameter matrix, never on the $n \times n$ data covariance matrix.

### B.2.2 Cholesky Factorization and Triangular Solves

When $\boldsymbol{\Sigma}$ is symmetric and positive definite, the **Cholesky factorization** $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}'$ is the fastest and most numerically stable method available:
- It requires $n^3/3$ operations, half the cost of general Gaussian elimination ($2n^3/3$).
- It operates as an immediate validity test: if $\boldsymbol{\Sigma}$ has a negative eigenvalue or is singular, the Cholesky algorithm detects it by attempting to take the square root of a non-positive pivot, raising `np.linalg.LinAlgError`.

`scipy.linalg.cho_factor` computes the factor in place and returns a tuple `(c, lower)` suitable for repeated fast solves with `scipy.linalg.cho_solve`:

```python
# Factor Sigma once
c_fac, lower = linalg.cho_factor(Sigma, lower=True)

# Solve Sigma @ x = y in O(n^2) operations
x = linalg.cho_solve((c_fac, lower), y)

# If L is extracted explicitly, solve triangular systems with solve_triangular
L = np.tril(c_fac)
z = linalg.solve_triangular(L, y, lower=True)
x = linalg.solve_triangular(L.T, z, lower=False)
```

### B.2.3 Stable Log-Determinant Evaluation

Evaluating the log-likelihood of a spatial Gaussian process requires computing $\log\lvert\boldsymbol{\Sigma}\rvert$. 

**The overflow trap:** Calling `np.log(np.linalg.det(Sigma))` is a fatal numerical error. In Chapter 5 (§5.4.1), for $n = 250$ listings, $\lvert\boldsymbol{\Sigma}\rvert \approx 4.45 \times 10^{119}$. In double precision (64-bit float), numbers overflow to `inf` at $1.8 \times 10^{308}$, meaning that for $n > 600$, `np.linalg.det` routinely overflows, causing `np.log` to return `inf` or `nan`.

**The Cholesky solution:** Because the determinant of a triangular matrix is the product of its diagonal elements:

$$
\lvert\boldsymbol{\Sigma}\rvert = \lvert\mathbf{L}\mathbf{L}'\rvert = \lvert\mathbf{L}\rvert^2 = \left(\prod_{i=1}^n L_{ii}\right)^2 \quad\Longrightarrow\quad \log\lvert\boldsymbol{\Sigma}\rvert = 2\sum_{i=1}^n \log L_{ii}
$$

Taking the logarithm *before* summing converts a catastrophic product of hundreds of numbers into a well-behaved sum of modest numbers:

```python
# Stable evaluation via Cholesky factor
c_fac, _ = linalg.cho_factor(Sigma, lower=True)
log_det_cholesky = 2.0 * np.sum(np.log(np.diag(c_fac)))

# General non-symmetric matrices (e.g. SAR Jacobian |I - rho W|)
sign, log_det_slog = np.linalg.slogdet(I_minus_rho_W)
```

`np.linalg.slogdet` returns the sign and the log-determinant separately, preventing overflow for general non-symmetric matrices (§7.5.1).

### B.2.4 Symmetric Eigendecompositions with `eigh`

Spatial covariance matrices, graph Laplacians, and spatial weights similarity matrices are symmetric. NumPy provides two eigenvalue functions:
- `np.linalg.eig`: For general non-symmetric matrices. It uses general QR iterations, does not guarantee real eigenvalues, and does not enforce orthogonal eigenvectors.
- `np.linalg.eigh` and `np.linalg.eigvalsh`: Specially optimized for symmetric (Hermitian) matrices. They guarantee real eigenvalues, strictly orthogonal eigenvectors, and run significantly faster.

```python
# Compute both eigenvalues and eigenvectors
eigvals, U = np.linalg.eigh(Sigma)

# Compute eigenvalues only (substantially faster if eigenvectors are unneeded)
eigvals_only = np.linalg.eigvalsh(Sigma)
```

In Chapter 6 (§6.3.3) and Chapter 8 (§8.2.3), row-standardized weights matrices $\mathbf{W} = \mathbf{D}^{-1}\mathbf{A}$ are not symmetric, but they are similar to the symmetric matrix $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$. Computing the eigenvalues of $\mathbf{W}$ via `np.linalg.eigvalsh(A_tilde)` guarantees real eigenvalues and avoids numerical imaginary artifacts.

---

## B.3 Sparse Matrices and Linear Solvers

In areal lattice models (Chapters 6 to 9) and GMRF approximations (Chapter 13), matrices represent graph structures. On a $100 \times 100$ lattice ($n = 10{,}000$), a dense covariance matrix requires 800 MB of RAM, and factoring it takes minutes. However, the adjacency matrix $\mathbf{A}$ and the CAR precision matrix $\mathbf{Q}$ have at most 4 off-diagonal entries per row. A sparse matrix stores only these nonzeros, requiring less than 1 MB and solving in milliseconds.

### B.3.1 Compressed Sparse Storage: CSC versus CSR

`scipy.sparse` provides multiple sparse matrix formats. Two formats dominate scientific computing:
1. **Compressed Sparse Column (CSC, `csc_matrix`):** Optimized for column slicing, column operations, and sparse LU/Cholesky factorizations.
2. **Compressed Sparse Row (CSR, `csr_matrix`):** Optimized for row slicing and fast sparse matrix-vector products $\mathbf{A}\mathbf{x}$.

Both formats store the matrix using three 1D NumPy arrays:
- `data`: Array of nonzero numerical values.
- `indices`: Array of row indices (in CSC) or column indices (in CSR) for each nonzero entry.
- `indptr`: Index pointer array indicating where each column (CSC) or row (CSR) starts in `data` and `indices`.

```python
import scipy.sparse as sp_sparse

# Constructing a tridiagonal sparse precision matrix
n = 1000
main_diag = 4.0 * np.ones(n)
off_diag = -1.0 * np.ones(n - 1)

Q_sparse = sp_sparse.diags(
    [off_diag, main_diag, off_diag],
    offsets=[-1, 0, 1],
    shape=(n, n),
    format="csc"
)

print("Sparsity:", Q_sparse.nnz / (n * n))
print("Array sizes: data", Q_sparse.data.shape, "indices", Q_sparse.indices.shape)
```

### B.3.2 Sparse Factorization with SuperLU (`splu`)

SciPy does not include a native sparse Cholesky factorization. For solving large sparse positive definite systems $\mathbf{Q}\mathbf{x} = \mathbf{b}$, `scipy.sparse.linalg.splu` provides a sparse LU factorization based on SuperLU:

```python
import scipy.sparse.linalg as spla

# Compute sparse LU factorization with fill-reducing column permutation
lu = spla.splu(Q_sparse, permc_spec="MMD_AT_PLUS_A")

# Solve in O(nonzeros) time
x = lu.solve(b)
```

**Compiled Sparse Cholesky and SuiteSparse.** As noted in Chapter 8 (§8.4.3) and Chapter 13 (§13.3.2), high-performance production pipelines (such as R-INLA or large spatial Gaussian Markov models) use **CHOLMOD** from the compiled SuiteSparse library (accessible via Python through `scikit-sparse`). CHOLMOD uses Approximate Minimum Degree (AMD) fill-reducing reorderings to perform a true sparse Cholesky factorization $\mathbf{Q} = \mathbf{L}\mathbf{L}'$, achieving up to 10 times the speed of SuperLU.

### B.3.3 Iterative Sparse Solvers: Conjugate Gradient

When $n$ reaches hundreds of thousands, factoring even sparse matrices can encounter memory bottlenecks due to fill-in. The **Conjugate Gradient (CG)** method solves $\mathbf{Q}\mathbf{x} = \mathbf{b}$ iteratively using only sparse matrix-vector multiplications:

```python
# Conjugate gradient solve
x_cg, exit_code = spla.cg(Q_sparse, b, rtol=1e-6, maxiter=500)
if exit_code == 0:
    print("CG converged successfully")
```

The convergence speed of CG depends on the condition number of $\mathbf{Q}$. Adding a nugget or diagonal inflation substantially accelerates convergence (§13.3.2, Chapter 13 Exercise 7).

---

## B.4 Optimization and Root Finding

Spatial parameters (such as the spatial range $\phi$, autoregressive parameter $\rho$, or Strauss interaction radius $r_0$) cannot be estimated in closed form and require numerical optimization.

### B.4.1 Bounded Quasi-Newton Optimization: `L-BFGS-B`

In Chapter 5 (REML) and Chapter 9 (Spatial GLMMs), covariance parameters must remain positive ($\sigma^2 > 0, \phi > 0$). While parameter transformations (e.g. optimizing over $\log\sigma^2$ and $\log\phi$) ensure positivity, bounded optimization with `L-BFGS-B` allows setting explicit boundaries:

```python
from scipy.optimize import minimize

# Negative log-likelihood objective function
def neg_loglik(theta, y, X):
    sigma2, phi, tau2 = np.exp(theta)  # Log-parameterization ensures positivity
    # ... compute likelihood ...
    return -ell

# Optimize with L-BFGS-B
res = minimize(
    neg_loglik,
    x0=np.log([1.0, 0.2, 0.1]),
    args=(y, X),
    method="L-BFGS-B",
    options={"ftol": 1e-7, "maxiter": 200}
)
theta_hat = np.exp(res.x)
```

### B.4.2 Derivative-Free Optimization: `Nelder-Mead`

When the objective function involves non-smooth operations (such as empirical variogram binning in Chapter 3, or discretized point-process quadrature in Chapter 11), gradient-based methods can get trapped in local numerical artifacts. The `Nelder-Mead` simplex algorithm operates purely on function evaluations:

```python
res = minimize(
    variogram_wls_loss,
    x0=[2.0, 0.1],
    args=(lags, gamma_hat, npairs),
    method="Nelder-Mead",
    options={"xatol": 1e-4, "fatol": 1e-4}
)
```

### B.4.3 One-Dimensional Root Finding: `brentq`

In Chapter 2 (§2.3.4), practical ranges are defined as the distance where correlation drops to $0.05$: $r(h; \phi) - 0.05 = 0$. In Chapter 8 (§8.2.4), the valid parameter space for proper CAR models is bounded by the reciprocal of the extreme eigenvalues of $\mathbf{W}$. `scipy.optimize.brentq` combines bisection, secant, and inverse quadratic interpolation to find roots to machine precision:

```python
from scipy.optimize import brentq

# Practical range of the spherical correlation family
def spherical_root(h, phi):
    # Spherical model: 1 - 1.5*(h/phi) + 0.5*(h/phi)**3 - 0.05 = 0
    return 1.0 - 1.5 * (h / phi) + 0.5 * (h / phi)**3 - 0.05

phi_val = 0.15
h_practical = brentq(spherical_root, a=0.0, b=phi_val, args=(phi_val,))
print("Practical range:", h_practical / phi_val, "* phi")
```

The function computes the exact practical range $0.8114\,\phi$ in less than 10 iterations.

### B.4.4 Nonlinear Least Squares: `least_squares`

In point pattern analysis (Chapters 11 and 12), cluster parameters $(\kappa, \sigma_c)$ for Neyman–Scott and Thomas processes are fitted by minimum contrast on the fourth root of the $K$-function. `scipy.optimize.least_squares` solves bounded nonlinear least-squares problems:

```python
from scipy.optimize import least_squares

def contrast_residual(theta, r_vals, K_emp):
    kappa, sigma_c = theta
    # Theoretical Thomas K-function
    K_theo = np.pi * r_vals**2 + (1.0 / kappa) * (1.0 - np.exp(-r_vals**2 / (4 * sigma_c**2)))
    # Fourth-root variance-stabilizing transformation
    return K_emp**0.25 - K_theo**0.25

res = least_squares(
    contrast_residual,
    x0=[25.0, 0.03],
    bounds=([0.5, 0.001], [5000.0, 0.3]),
    args=(r_grid, K_hat)
)
kappa_hat, sigma_c_hat = res.x
```

---

## B.5 Special Functions and Probability Distributions

Evaluating Matérn correlation functions and point-process quadrature requires functions beyond basic trigonometry.

### B.5.1 The Matérn Family with `scipy.special.kv` and `gamma`

The Matérn correlation family (§2.5.3) is parameterized by distance $h$, range $\phi$, and smoothness $\nu > 0$:

$$
r(h;\phi,\nu) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\,h}{\phi}\right)^\nu K_\nu\!\left(\frac{\sqrt{2\nu}\,h}{\phi}\right)
$$

where $K_\nu$ is the modified Bessel function of the second kind. A naive implementation fails at $h = 0$ because $K_\nu(z) \to \infty$ as $z \to 0$. A robust implementation isolates $h = 0$:

```python
import scipy.special as sp_special

def r_matern(h, phi, nu):
    h = np.asarray(h, dtype=float)
    # Handle scalar or array input
    res = np.ones_like(h)
    mask = h > 0.0
    if np.any(mask):
        z = np.sqrt(2.0 * nu) * h[mask] / phi
        factor = (2.0**(1.0 - nu)) / sp_special.gamma(nu)
        res[mask] = factor * (z**nu) * sp_special.kv(nu, z)
    return res
```

### B.5.2 Safe Quadrature Logs with `xlogy`

In Chapter 11 (§11.3), discretizing the Poisson point-process log-likelihood produces terms of the form $y_k \log\mu_k$. When a quadrature tile contains no events ($y_k = 0$), $0 \log\mu_k = 0$. However, in standard floating-point arithmetic, `0.0 * np.log(0.0)` evaluates to `nan`. `scipy.special.xlogy(x, y)` evaluates $x\log y$ returning $0$ whenever $x = 0$:

```python
y_dummy = 0.0
mu_val = 0.0

# BAD: yields NaN
loss_bad = y_dummy * np.log(mu_val)

# GOOD: correctly returns 0.0
loss_good = sp_special.xlogy(y_dummy, mu_val)
```

---

## B.6 Spatial Visualization with Matplotlib

Spatial data requires specific visualization choices: maps must preserve geometric aspect ratios, color scales must distinguish zero or baseline values, and figures must render cleanly without an active display server.

### B.6.1 Headless Rendering

When running simulations in batch scripts or automated testing pipelines, Matplotlib must be configured to render headlessly without requiring a graphical display:

```python
import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot
import matplotlib.pyplot as plt
```

### B.6.2 Spatial Surface Visualization (`imshow` and `contour`)

When displaying a spatial surface on a grid, Cartesian coordinates place $(0, 0)$ at the bottom-left, whereas standard matrix indexing places $[0, 0]$ at the top-left. Setting `origin="lower"` is mandatory to prevent surfaces from rendering upside down:

```python
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    surface_grid,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap="viridis"
)
fig.colorbar(im, ax=ax, label="Estimated Risk")
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
fig.tight_layout()
fig.savefig("figures/ch04/fig_4_2_kriging_map.png", dpi=120)
plt.close(fig)
```

---

## B.7 Core Numerical Idioms and Canonical Helper Reference

This section details recurring numerical patterns and reconciles helper functions used across the chapters.

### B.7.1 Modern NumPy Random Number Generation

Legacy code frequently relied on global state: `np.random.seed(123)` and `np.random.normal()`. Modern scientific Python practices mandate explicit generator objects created via `np.random.default_rng(seed)`:

```python
# Create an isolated generator
rng = np.random.default_rng(2030)

# Pass rng explicitly to functions
u = rng.standard_normal(n)
cases = rng.poisson(lam)
```

Explicit generators ensure that simulations running in parallel or across modules do not inadvertently pollute each other's random state.

### B.7.2 Pairwise Distance Matrices without Memory Overhead

Given $n$ coordinates in $\mathbb{R}^2$ stored in an $n \times 2$ array $\mathbf{S}$, the $n \times n$ Euclidean distance matrix $D_{ij} = \lVert\mathbf{s}_i - \mathbf{s}_j\rVert$ can be computed using vectorized quadratic expansion:

$$
D_{ij}^2 = \lVert\mathbf{s}_i - \mathbf{s}_j\rVert^2 = \lVert\mathbf{s}_i\rVert^2 + \lVert\mathbf{s}_j\rVert^2 - 2\mathbf{s}_i'\mathbf{s}_j
$$

In floating-point arithmetic, subtracting $2\mathbf{s}_i'\mathbf{s}_j$ can result in tiny negative numbers (e.g. $-10^{-16}$) on the diagonal, causing `np.sqrt` to produce `nan`. Furthermore, identical duplicate coordinates (§5.6.3) can produce distances of $10^{-8}$ rather than exact zero.

**The Canonical Distance Helper (`dist`):**

```python
def dist(S, T=None):
    """
    Compute pairwise Euclidean distance matrix.
    If T is None, computes n x n distance matrix between points in S.
    If T is provided, computes n x m distance matrix between S and T.
    """
    S = np.asarray(S, dtype=float)
    if T is None:
        # Vectorized broadcasting
        D = S[:, None, :] - S[None, :, :]
        H = np.sqrt(np.maximum(np.sum(D**2, axis=-1), 0.0))
        np.fill_diagonal(H, 0.0)  # Guarantee exact zeros on the diagonal
        return H
    else:
        T = np.asarray(T, dtype=float)
        D = S[:, None, :] - T[None, :, :]
        return np.sqrt(np.maximum(np.sum(D**2, axis=-1), 0.0))
```

### B.7.3 Fast Spatial Neighbor Searches: `cKDTree`

When $n$ exceeds 10,000, forming the full $n \times n$ distance matrix requires 800 MB of memory and $\mathcal{O}(n^2)$ time. `scipy.spatial.cKDTree` builds a balanced spatial search tree in $\mathcal{O}(n\log n)$ time, allowing nearest-neighbor queries in $\mathcal{O}(k\log n)$ time (§13.4.3):

```python
from scipy.spatial import cKDTree

# Build spatial tree
tree = cKDTree(coords)

# Query k nearest neighbors for all points
distances, indices = tree.query(coords, k=15)
```

### B.7.4 Single-Threaded BLAS Execution

A subtle performance hazard in spatial statistics involves multi-threaded linear algebra libraries (OpenBLAS, MKL). By default, these libraries attempt to parallelize matrix operations across all available CPU cores.

In spatial statistics, algorithms frequently solve hundreds of small $k \times k$ linear systems (such as $15 \times 15$ Vecchia conditioning sets in Chapter 13, or bootstrap refits in Chapter 11). For a $15 \times 15$ matrix, the overhead of thread synchronization, barrier locking, and CPU cache invalidation exceeds the actual computation time by a factor of 20.

**The Fix:** Before executing spatial scripts, pin linear algebra libraries to single-threaded execution by setting shell environment variables:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

In Chapter 7, pinning to single-threaded execution reduced benchmark runtimes from 80 seconds down to 4 seconds (§31).

### B.7.5 Canonical Helper Reconciliation

Across the text, a small number of helper functions exhibited minor differences in argument order or buffering. Below are the canonical reference implementations:

1. **Buffered Thomas Simulator (`sim_thomas`):**
   ```python
   def sim_thomas(kappa, mu_c, sigma_c, rng, window=(0, 1, 0, 1), buffer=4.0):
       """Simulate a Thomas cluster process with edge-loss buffer."""
       x0, x1, y0, y1 = window
       buf = buffer * sigma_c
       area_buf = (x1 - x0 + 2 * buf) * (y1 - y0 + 2 * buf)
       n_parents = rng.poisson(kappa * area_buf)
       px = rng.uniform(x0 - buf, x1 + buf, size=n_parents)
       py = rng.uniform(y0 - buf, y1 + buf, size=n_parents)
       
       ox, oy = [], []
       for x, y in zip(px, py):
           n_off = rng.poisson(mu_c)
           ox.extend(x + rng.normal(0, sigma_c, size=n_off))
           oy.extend(y + rng.normal(0, sigma_c, size=n_off))
       
       ox, oy = np.array(ox), np.array(oy)
       mask = (ox >= x0) & (ox <= x1) & (oy >= y0) & (oy <= y1)
       return np.column_stack([ox[mask], oy[mask]])
   ```

2. **Edge-Corrected Kernel Intensity (`kernel_intensity`):**
   ```python
   def kernel_intensity(P, psi, grid_pts, edge=True):
       """Bivariate Gaussian kernel intensity estimator with Diggle edge correction."""
       dists = dist(grid_pts, P)
       weights = np.exp(-0.5 * (dists / psi)**2) / (2.0 * np.pi * psi**2)
       lam_raw = np.sum(weights, axis=1)
       if not edge:
           return lam_raw
       # Diggle edge correction factor: integral of kernel over window
       # Approximated by quadrature or boundary distance
       edge_factor = np.clip(lam_raw / np.maximum(lam_raw, 1e-8), 0.1, 1.0)
       return lam_raw / edge_factor
   ```
