# Chapter 13: Matrix Approximations and Fast Algorithms for Large $n$

Every method in Parts II to IV ends with the same two computations. To fit a spatial model we evaluate the Gaussian log-likelihood, which needs $\log\lvert\boldsymbol{\Sigma}\rvert$ and the quadratic form $\mathbf{r}'\boldsymbol{\Sigma}^{-1}\mathbf{r}$ of a residual vector $\mathbf{r}$. To predict, we solve $\boldsymbol{\Sigma}\mathbf{x} = \mathbf{r}$ and multiply by a covariance vector. Chapter 5 did both with a Cholesky factorization of the $n \times n$ covariance matrix, and for the 250 listings of the running rent example that cost next to nothing.

Now picture the same model for a metropolitan area with tens of thousands of rental listings. The covariance matrix of 20,000 listings has 400 million entries and takes 3.2 GB in double precision. Factoring it takes time that grows with the cube of $n$, and a maximum likelihood fit needs the factorization hundreds of times. Nothing in the statistics changes at that size. The difficulty is purely computational, and it has a computational answer: find structure in $\boldsymbol{\Sigma}$ or in $\boldsymbol{\Sigma}^{-1}$ and use it.

This chapter presents four kinds of structure. A covariance matrix can be close to **low rank**, because a smooth neighborhood effect is determined by its values at a modest number of knot locations (Section 13.2). It can be close to **sparse**, if we multiply the correlation function by one that vanishes beyond some distance (Section 13.3). Its inverse, or the factor of its inverse, can be sparse, because each observation depends on only a few earlier neighbors once we condition on them (Sections 13.4 and 13.5). And on a grid in space and time it can have **Kronecker** structure, which is exact instead of approximate (Section 13.6). Three of the four are approximations, so we measure how good they are against the exact answer on data sets small enough to compute it, and then we run them at sizes where the exact answer is out of reach.

The tools are ones we already have. The Woodbury identity of Section 13.2 follows from the block inversion of Chapter 4. The tapering of Section 13.3 is the product rule for covariance functions of Chapter 2. Vecchia's approximation is the kriging weights of Chapter 4 used as regression coefficients, and the conditional independence it imposes is the idea behind the precision matrices of Chapter 8. The bridge of Section 13.5 shows that a grid version of the SAR model of Chapter 7 is a Matérn field, which connects the covariance models of Part II with the lattice models of Part III. This is the last chapter of the book, so it also gathers threads from every part.

> **Chapter at a Glance**
>
> - **The Bottleneck (§13.1):** What one exact likelihood evaluation costs in time and memory, measured up to $n = 5{,}000$ and extrapolated beyond, and the four structures the rest of the chapter exploits.
> - **Low-Rank Models (§13.2):** A random effect built from $q \ll n$ knots, the Woodbury identity and determinant lemma derived from block inversion, a four-listing hand example, likelihood and kriging in $O(nq^2)$ time, and what a low-rank model loses.
> - **Covariance Tapering (§13.3):** Multiplying by a compactly supported correlation keeps the matrix valid and makes it sparse. Sparse factorization with SciPy, and an honest account of when it pays.
> - **Vecchia's Approximation (§13.4):** The likelihood as a product of conditionals on a few neighbors, the sparse factor it induces, the effect of the ordering, and a fit to 20,000 listings.
> - **Matérn Fields as Markov Fields (§13.5):** A grid precision matrix that is a SAR model and also a Matérn field, checked numerically, and the Woodbury identity again with a sparse precision.
> - **Kronecker Products (§13.6):** Space-time covariances that factor, the determinant and solve in closed form, and what separability assumes.
> - **Walkthrough and Lab (§13.7):** A benchmark of the four methods against the exact computation, timings at 20,000 and 50,000 listings, and four figures.

---

## 13.1 The $O(n^3)$ Bottleneck

### 13.1.1 What One Likelihood Evaluation Costs

Suppose $\mathbf{Y} \sim N(\mathbf{X}\boldsymbol{\beta}, \boldsymbol{\Sigma})$ with $\boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n$, the rent model of Chapters 3 to 5. For fixed $\boldsymbol{\theta} = (\sigma^2, \tau^2, \phi)'$ the profile log-likelihood of Section 5.2 takes four steps.

1. **Build** the $n \times n$ matrix $\boldsymbol{\Sigma}$ from the distances. This costs $O(n^2)$ operations and $n^2$ stored numbers.
2. **Factor** $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}'$ with the Cholesky factorization of Section 2.7. This costs about $n^3/3$ multiplications and additions, and it dominates everything else.
3. **Solve** $\boldsymbol{\Sigma}\mathbf{x} = \mathbf{b}$ for the $p + 1$ right-hand sides we need (the residual and the columns of $\mathbf{X}$), by two triangular solves each. Each solve costs $O(n^2)$.
4. **Read off** the log-determinant as $2\sum_i\log L_{ii}$ in $O(n)$ operations.

The factorization is the problem. Doubling $n$ multiplies its cost by eight. The memory is a second problem that arrives sooner: the matrix takes $8n^2$ bytes, so it stops fitting in the memory of an ordinary computer well before the time becomes unreasonable.

An optimizer needs many evaluations. Fisher scoring from a poor start needed 12 iterations in Chapter 5, and the Nelder-Mead searches of this chapter take about 200 evaluations. Every evaluation pays the cost of Step 2 again, because $\boldsymbol{\Sigma}$ changes with $\boldsymbol{\theta}$.

### 13.1.2 Measuring It

The code below times the Cholesky factorization for random listings in the unit square, at six sample sizes up to $n = 5{,}000$. It also defines the helper functions used throughout the chapter: a distance matrix, the exponential and Matérn correlation functions (Section 2.5), and a timer. Each timing is the best of three runs. The timings depend on the machine and on what else it is running, so treat the seconds as indicative and the ratios between sizes as the finding. The code imports `scipy.spatial` (the k-d tree `cKDTree` and the pairwise distance function `cdist`), which is part of SciPy and so within the book's rule of using no spatial packages; it is used here for speed and for nearest-neighbor search at sizes where a hand-built search would be slow, and the `dist` function of Chapter 1 would work at the smaller sizes.

```python
import time
import numpy as np
from scipy import linalg, optimize, sparse, special
from scipy.sparse import linalg as spla
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

sigma2, tau2, phi = 4.0, 1.0, 0.15      # the rent model of Chapters 3 to 5


def dist(S):
    """Matrix of Euclidean distances between the rows of S."""
    q = np.sum(S**2, axis=1)
    H2 = q[:, None] - 2 * S @ S.T + q[None, :]
    H = np.sqrt(np.maximum(H2, 0.0))
    np.fill_diagonal(H, 0.0)
    return H


def r_exp(h, ph):
    """Exponential correlation."""
    return np.exp(-h / ph)


def r_m52(h, ph):
    """Matern correlation with smoothness 5/2."""
    x = h / ph
    return (1.0 + x + x**2 / 3.0) * np.exp(-x)


def exp_cov(H, s2=sigma2, t2=tau2, ph=phi):
    """Covariance matrix of the rents: exponential plus nugget."""
    return s2 * r_exp(H, ph) + t2 * np.eye(len(H))


def design(S):
    """Design matrix: intercept and the east coordinate."""
    return np.column_stack([np.ones(len(S)), S[:, 0]])


def timed(f, *args, repeat=1, **kwargs):
    """Result of f and the best wall-clock time over repeat calls."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = f(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return out, best


rng = np.random.default_rng(2110)
ns = [500, 1000, 2000, 3000, 4000, 5000]
chol_times = []
for n in ns:
    S = rng.uniform(size=(n, 2))
    Sigma = exp_cov(dist(S))
    _, t = timed(np.linalg.cholesky, Sigma, repeat=3)
    chol_times.append(t)
    print(f"n={n:5d}  matrix={8 * n**2 / 1e6:6.1f} MB  "
          f"Cholesky={t:6.3f} s")
chol_times = np.array(chol_times)
big = np.array(ns) >= 2000
slope = np.polyfit(np.log(np.array(ns)[big]), np.log(chol_times[big]), 1)[0]
print(f"fitted exponent for n >= 2000: {slope:.2f}")

c3 = chol_times[-1] / ns[-1]**3           # seconds per n^3 at n = 5,000
for n in (10_000, 20_000, 50_000, 100_000):
    print(f"n={n:7d}  matrix={8 * n**2 / 1e9:7.1f} GB  "
          f"Cholesky (extrapolated) = {c3 * n**3 / 60:8.1f} min")
```

The fitted exponent is about 3, as the operation count predicts. A log-log plot makes the same point (Figure 13.1, left): measured times follow a line of slope 3, and the matrix size grows with slope 2 (Figure 13.1, right).

![Figure 13.1. Left: measured Cholesky time against the number of listings on logarithmic axes, with a reference line of slope 3. Right: the memory of a dense covariance matrix. The measured points are at most 5,000 listings; the larger sizes are extrapolated from the formula $8n^2$ bytes.](../figures/ch13/fig_13_1_scaling.png)

The last four lines of output are **extrapolations**, not measurements. They assume that the cubic law continues and ignore everything that changes at large sizes: cache effects, memory limits, and virtual memory. Memory reaches its limit first. At $n = 20{,}000$ the matrix needs 3.2 GB, which many laptops can hold, and the extrapolated factorization takes a little over a minute. At $n = 50{,}000$ the matrix needs 20 GB and the extrapolated factorization takes about 18.5 minutes, for one likelihood evaluation. At $n = 100{,}000$ the matrix needs 80 GB. A metropolitan data set with tens of thousands of listings is therefore beyond the exact method on a laptop, and a fit with hundreds of evaluations is beyond it on nearly any machine.

### 13.1.3 Four Structures

The remaining sections each exploit one structure of the model. The table says what is approximated and what the cost becomes.

| Structure | What is approximated | Cost of a likelihood evaluation | Section |
| :--- | :--- | :--- | :--- |
| Low rank | The random effect, by its values at $q$ knots | $O(nq^2 + q^3)$ | 13.2 |
| Sparse covariance | The covariance matrix, by tapering its long-range entries | Sparse factorization; depends on the fill-in | 13.3 |
| Sparse factor of the precision | The joint density, by conditioning each point on $k$ earlier neighbors | $O(nk^3)$ | 13.4 |
| Sparse precision | The Matérn field, by a Markov field on a grid | Sparse factorization of the precision | 13.5 |
| Kronecker | Nothing, if the covariance is separable | $O(n_S^3 + n_T^3 + n_Sn_T(n_S + n_T))$ | 13.6 |

The listings in the story come from the rent model of Chapters 3 to 5. The benchmarks use the exact simulation up to $n = 5{,}000$, because only there can we compute the true answer. For the 20,000 listings of Section 13.4.4 the data cannot be simulated exactly, and the text says how they are generated.

---

## 13.2 Low-Rank Spatial Models

### 13.2.1 The Idea

Two listings on the same block carry nearly the same information about the unexplained neighborhood effect $u$. If $u$ is smooth, then its values at 20,000 listing locations are determined, to a good approximation, by its values at a few hundred well-spread locations. A low-rank model builds that into the covariance: it writes the random effect as a combination of $q \ll n$ quantities.

Choose $q$ **knot** locations $\boldsymbol{\kappa}_1, \dots, \boldsymbol{\kappa}_q$ in the study region. (In this section $\boldsymbol{\kappa}_j$ is a knot location. The symbol $\kappa$ is also the inverse range in Section 13.5 and the parent intensity of Chapters 11 and 12.) Let $\boldsymbol{\alpha}$ be the vector of $u$ at the knots, with covariance matrix $\boldsymbol{\Sigma}_\alpha = \sigma^2\mathbf{R}_K$, where $\mathbf{R}_K$ is the $q \times q$ correlation matrix of the knots. Approximate the effect at any location by kriging it from the knots, in the sense of Chapter 4. By Section 4.2, the best linear predictor of $u(\mathbf{s})$ from $\boldsymbol{\alpha}$ is $\mathbf{r}(\mathbf{s})'\mathbf{R}_K^{-1}\boldsymbol{\alpha}$, where $\mathbf{r}(\mathbf{s})$ is the vector of correlations between $\mathbf{s}$ and the knots. Stacking the $n$ listings gives

$$
\mathbf{u} \approx \mathbf{B}\boldsymbol{\alpha}, \qquad \mathbf{B} = \mathbf{R}_{nK}\mathbf{R}_K^{-1}
$$

where $\mathbf{R}_{nK}$ is the $n \times q$ matrix of correlations between listings and knots. The matrix $\mathbf{B}$ is an $n \times q$ matrix of kriging weights. The model for the rents becomes

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{B}\boldsymbol{\alpha} + \boldsymbol{\varepsilon}, \qquad \operatorname{Var}(\boldsymbol{\alpha}) = \boldsymbol{\Sigma}_\alpha, \qquad \operatorname{Var}(\boldsymbol{\varepsilon}) = \tau^2\mathbf{I}_n
$$

and the covariance matrix of $\mathbf{Y}$ is

$$
\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'
$$

This is the linear mixed model of Section 2.1.5 with $q$ random effects instead of $n$: the matrix $\mathbf{B}$ plays the role of the matrix $\mathbf{Z}_b$ there. The matrix $\mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}' = \sigma^2\mathbf{R}_{nK}\mathbf{R}_K^{-1}\mathbf{R}_{Kn}$ has rank at most $q$. It is positive semi-definite by the Gramian rule of Section 2.4.3, which Chapter 2 promised to use here. This particular choice of $\mathbf{B}$ and $\boldsymbol{\Sigma}_\alpha$ is called the **predictive process** [@banerjee2008]. The matrices $\mathbf{B}$ and $\boldsymbol{\alpha}$ here have the local meaning just defined; they are unrelated to the generic matrix $\mathbf{B}$ of Chapter 1 and the kernel ridge coefficients of Chapter 4.

More general low-rank models let $\mathbf{B}$ be any matrix of basis functions evaluated at the listings, such as Gaussian bumps or splines, with $\boldsymbol{\Sigma}_\alpha$ a free covariance matrix to be estimated. That is called fixed-rank kriging [@cressie2008]. Everything in the rest of this section holds for any $\mathbf{B}$ and $\boldsymbol{\Sigma}_\alpha$, and Exercise 4 builds a bump basis. We use the predictive process because its covariance is tied to the exponential model of Chapters 3 to 5, which lets us compare with the exact answer.

For knots we use a regular $g \times g$ grid over the unit square, so that $q = g^2$.

### 13.2.2 A Tool We Need: the Woodbury Identity and the Determinant Lemma

We need $\boldsymbol{\Sigma}^{-1}$ and $\log\lvert\boldsymbol{\Sigma}\rvert$ for a matrix of the form "diagonal plus low rank". It is useful to allow a general diagonal matrix $\boldsymbol{\Omega} = \operatorname{diag}(d_1, \dots, d_n)$ in place of $\tau^2\mathbf{I}_n$, because Section 13.2.5 needs that generality:

$$
\boldsymbol{\Sigma} = \boldsymbol{\Omega} + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'
$$

**Block inversion, restated.** Section 4.3.4 inverted a partitioned matrix with the Schur complement of the upper-left block, and noted that exchanging the roles of the two blocks gives the same formula. For a matrix $\mathbf{N}$ with blocks $\mathbf{N}_{11}, \mathbf{N}_{12}, \mathbf{N}_{21}, \mathbf{N}_{22}$, the upper-left block of $\mathbf{N}^{-1}$ is therefore given by the formula with the roles exchanged (first form) and by the formula of Section 4.3.4 (second form):

$$
\left(\mathbf{N}^{-1}\right)_{11} = \left(\mathbf{N}_{11} - \mathbf{N}_{12}\mathbf{N}_{22}^{-1}\mathbf{N}_{21}\right)^{-1} = \mathbf{N}_{11}^{-1} + \mathbf{N}_{11}^{-1}\mathbf{N}_{12}\left(\mathbf{N}_{22} - \mathbf{N}_{21}\mathbf{N}_{11}^{-1}\mathbf{N}_{12}\right)^{-1}\mathbf{N}_{21}\mathbf{N}_{11}^{-1}
$$

The first form eliminates the lower-right block, the second eliminates the upper-left block, which is the form of Section 4.3.4. Both are the same matrix, and Appendix A.1 will state both forms and verify them by multiplication.

**The identity.** Apply this to the $(n + q) \times (n + q)$ matrix

$$
\mathbf{N} = \begin{bmatrix} \boldsymbol{\Omega} & \mathbf{B} \\ -\mathbf{B}' & \boldsymbol{\Sigma}_\alpha^{-1} \end{bmatrix}
$$

with $\mathbf{N}_{11} = \boldsymbol{\Omega}$, $\mathbf{N}_{12} = \mathbf{B}$, $\mathbf{N}_{21} = -\mathbf{B}'$, and $\mathbf{N}_{22} = \boldsymbol{\Sigma}_\alpha^{-1}$. The first form gives $(\boldsymbol{\Omega} + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}')^{-1} = \boldsymbol{\Sigma}^{-1}$. The second gives $\boldsymbol{\Omega}^{-1} - \boldsymbol{\Omega}^{-1}\mathbf{B}\left(\boldsymbol{\Sigma}_\alpha^{-1} + \mathbf{B}'\boldsymbol{\Omega}^{-1}\mathbf{B}\right)^{-1}\mathbf{B}'\boldsymbol{\Omega}^{-1}$. Equating them is the **Sherman-Morrison-Woodbury identity**:

$$
\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Omega}^{-1} - \boldsymbol{\Omega}^{-1}\mathbf{B}\,\mathbf{M}^{-1}\mathbf{B}'\boldsymbol{\Omega}^{-1}, \qquad \mathbf{M} = \boldsymbol{\Sigma}_\alpha^{-1} + \mathbf{B}'\boldsymbol{\Omega}^{-1}\mathbf{B}
$$

The matrix $\mathbf{M}$ is only $q \times q$. With $\boldsymbol{\Omega} = \tau^2\mathbf{I}_n$ the identity becomes $\boldsymbol{\Sigma}^{-1} = \tau^{-2}\mathbf{I}_n - \tau^{-4}\mathbf{B}\left(\boldsymbol{\Sigma}_\alpha^{-1} + \tau^{-2}\mathbf{B}'\mathbf{B}\right)^{-1}\mathbf{B}'$. In words: inverting an $n \times n$ matrix is replaced by inverting a diagonal matrix, which is free, and a $q \times q$ matrix.

**The determinant lemma.** A block matrix factors as

$$
\begin{bmatrix} \mathbf{N}_{11} & \mathbf{N}_{12} \\ \mathbf{N}_{21} & \mathbf{N}_{22} \end{bmatrix} = \begin{bmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{N}_{21}\mathbf{N}_{11}^{-1} & \mathbf{I} \end{bmatrix}\begin{bmatrix} \mathbf{N}_{11} & \mathbf{0} \\ \mathbf{0} & \mathbf{N}_{22} - \mathbf{N}_{21}\mathbf{N}_{11}^{-1}\mathbf{N}_{12} \end{bmatrix}\begin{bmatrix} \mathbf{I} & \mathbf{N}_{11}^{-1}\mathbf{N}_{12} \\ \mathbf{0} & \mathbf{I} \end{bmatrix}
$$

(multiply the three matrices to check). The outer factors are triangular with ones on the diagonal, so their determinants are 1, and $\lvert\mathbf{N}\rvert = \lvert\mathbf{N}_{11}\rvert\,\lvert\mathbf{N}_{22} - \mathbf{N}_{21}\mathbf{N}_{11}^{-1}\mathbf{N}_{12}\rvert$. Eliminating the other block gives, in the same way, $\lvert\mathbf{N}\rvert = \lvert\mathbf{N}_{22}\rvert\,\lvert\mathbf{N}_{11} - \mathbf{N}_{12}\mathbf{N}_{22}^{-1}\mathbf{N}_{21}\rvert$. For the matrix $\mathbf{N}$ above, the first expression is $\lvert\boldsymbol{\Omega}\rvert\,\lvert\boldsymbol{\Sigma}_\alpha^{-1} + \mathbf{B}'\boldsymbol{\Omega}^{-1}\mathbf{B}\rvert = \lvert\boldsymbol{\Omega}\rvert\,\lvert\mathbf{M}\rvert$, and the second is $\lvert\boldsymbol{\Sigma}_\alpha^{-1}\rvert\,\lvert\boldsymbol{\Omega} + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'\rvert = \lvert\boldsymbol{\Sigma}_\alpha\rvert^{-1}\lvert\boldsymbol{\Sigma}\rvert$. Equating them,

$$
\lvert\boldsymbol{\Sigma}\rvert = \lvert\boldsymbol{\Omega}\rvert\,\lvert\boldsymbol{\Sigma}_\alpha\rvert\,\lvert\mathbf{M}\rvert, \qquad \log\lvert\boldsymbol{\Sigma}\rvert = \sum_{i=1}^n\log d_i + \log\lvert\boldsymbol{\Sigma}_\alpha\rvert + \log\lvert\mathbf{M}\rvert
$$

With $\boldsymbol{\Omega} = \tau^2\mathbf{I}_n$ the first term is $n\log\tau^2$. Both $\log\lvert\boldsymbol{\Sigma}_\alpha\rvert$ and $\log\lvert\mathbf{M}\rvert$ come from $q \times q$ Cholesky factors.

### 13.2.3 By Hand: Four Listings, Two Knots

Take $n = 4$ listings and $q = 2$ knots with

$$
\mathbf{B} = \begin{bmatrix} 1 & 0 \\ 0.8 & 0.2 \\ 0.2 & 0.8 \\ 0 & 1 \end{bmatrix}, \qquad \boldsymbol{\Sigma}_\alpha = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \qquad \tau^2 = 1
$$

Each row of $\mathbf{B}$ says how much a listing leans on each knot: the first listing sits on knot 1 and the last on knot 2. The code builds $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_4 + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'$, forms the $2 \times 2$ matrix $\mathbf{M}$, and compares the Woodbury inverse with the dense inverse.

```python
np.set_printoptions(precision=4, suppress=True)
B4 = np.array([[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]])
Sa4 = np.array([[2.0, 1.0], [1.0, 2.0]])
t24 = 1.0
Sigma4 = t24 * np.eye(4) + B4 @ Sa4 @ B4.T
print("Sigma =\n", Sigma4)
Sa_inv4 = np.linalg.inv(Sa4)
M4 = Sa_inv4 + B4.T @ B4 / t24
print("Sa^-1 =\n", Sa_inv4)
print("B'B / tau2 =\n", B4.T @ B4 / t24)
print("M = Sa^-1 + B'B / tau2 =\n", M4)
W4 = np.eye(4) / t24 - B4 @ np.linalg.inv(M4) @ B4.T / t24**2
print("Woodbury inverse =\n", W4)
print("dense inverse =\n", np.linalg.inv(Sigma4))
print("largest difference:", np.abs(W4 - np.linalg.inv(Sigma4)).max())
print("|Sa| =", round(np.linalg.det(Sa4), 4), " |M| =",
      round(np.linalg.det(M4), 4), " |Sigma| =",
      round(np.linalg.det(Sigma4), 4))
print("log|Sigma|, dense:", round(np.linalg.slogdet(Sigma4)[1], 6),
      " lemma:", round(4 * np.log(t24) + np.log(np.linalg.det(Sa4))
                       + np.log(np.linalg.det(M4)), 6))
```

The matrix $\mathbf{M} = \boldsymbol{\Sigma}_\alpha^{-1} + \mathbf{B}'\mathbf{B}/\tau^2$ has $2.3467$ on its diagonal and $-0.0133$ off it. Its inverse, applied through the formula $\boldsymbol{\Sigma}^{-1} = \mathbf{I} - \mathbf{B}\mathbf{M}^{-1}\mathbf{B}'$ (here $\tau^2 = 1$), reproduces the dense $4 \times 4$ inverse to machine precision. The determinants confirm the lemma: $\lvert\boldsymbol{\Sigma}_\alpha\rvert = 3$ and $\lvert\mathbf{M}\rvert = 5.5067$, so $\lvert\boldsymbol{\Sigma}\rvert = 1 \times 3 \times 5.5067 = 16.52$, and $\log\lvert\boldsymbol{\Sigma}\rvert = 2.804572$ both ways.

### 13.2.4 Likelihood and Kriging in $O(nq^2)$ Time

With $\boldsymbol{\Sigma}^{-1}$ and $\log\lvert\boldsymbol{\Sigma}\rvert$ available cheaply, the profile likelihood of Section 5.2 goes through unchanged. Count the cost. Forming $\mathbf{B}'\boldsymbol{\Omega}^{-1}\mathbf{B}$ takes $O(nq^2)$ operations, factoring $\mathbf{M}$ takes $O(q^3)$, and each solve $\boldsymbol{\Sigma}^{-1}\mathbf{v}$ costs $O(nq)$ because it multiplies by $\mathbf{B}'$, solves with $\mathbf{M}$, and multiplies by $\mathbf{B}$. In all, a likelihood evaluation costs $O(nq^2 + q^3)$ time and $O(nq)$ memory, instead of $O(n^3)$ and $O(n^2)$.

Kriging also carries over. The noise-free value at a new location $\mathbf{s}_0$ is $\mu(\mathbf{s}_0) + \mathbf{b}_0'\boldsymbol{\alpha}$, where $\mathbf{b}_0 = \mathbf{R}_K^{-1}\mathbf{r}(\mathbf{s}_0)$ are the kriging weights for $\mathbf{s}_0$. Its covariance with the data is $\mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{b}_0$. By Section 4.4.3 the predictor is $\mathbf{x}_0'\hat{\boldsymbol{\beta}} + \mathbf{b}_0'\boldsymbol{\Sigma}_\alpha\mathbf{B}'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})$, that is, $\mathbf{b}_0'\hat{\boldsymbol{\alpha}}$ with $\hat{\boldsymbol{\alpha}} = \boldsymbol{\Sigma}_\alpha\mathbf{B}'\boldsymbol{\Sigma}^{-1}\mathbf{r}$. So the knot values are estimated once, and every prediction is then a product of a $1 \times q$ and a $q \times 1$ vector.

The code implements the class `LowRank`, which stores the Cholesky factor of $\mathbf{M}$ and applies the Woodbury formula, and the function `lowrank_fit`, which evaluates the likelihood and kriging. It also computes the quantity `deficit` that Section 13.2.5 explains. We compare against the exact computation on the $n = 2{,}000$ rent data set of Section 13.1, with 200 held-out listings. The exact answer is from the function `exact_fit` (the Chapter 5 likelihood with GLS and kriging), defined in the block below together with the simulator. The simulator `simulate_rents` draws data and held-out listings jointly from the rent model with exact Cholesky simulation (Section 2.7), so we know the true noise-free rent at the held-out listings.

```python
def simulate_rents(n, n_test, seed, ph=phi):
    """Exact simulation at n + n_test sites (n data, n_test held out)."""
    rng = np.random.default_rng(seed)
    S_all = rng.uniform(size=(n + n_test, 2))
    H = dist(S_all)
    L = np.linalg.cholesky(sigma2 * r_exp(H, ph) + 1e-8 * np.eye(len(H)))
    u = L @ rng.standard_normal(len(H))
    eps = np.sqrt(tau2) * rng.standard_normal(len(H))
    mu = 18.0 + 6.0 * S_all[:, 0]
    y = mu + u + eps
    return S_all[:n], y[:n], S_all[n:], (mu + u)[n:]


def exact_fit(S, y, X, S0, X0, corr=r_exp, s2=sigma2, t2=tau2, ph=phi):
    """Gaussian log-likelihood (beta profiled out by GLS) and kriging."""
    Sigma = s2 * corr(dist(S), ph) + t2 * np.eye(len(S))
    c = linalg.cho_factor(Sigma)
    Sx, Sy = linalg.cho_solve(c, X), linalg.cho_solve(c, y)
    beta = np.linalg.solve(X.T @ Sx, X.T @ Sy)
    r = y - X @ beta
    Sr = linalg.cho_solve(c, r)
    logdet = 2.0 * np.sum(np.log(np.diag(c[0])))
    ll = -0.5 * (len(y) * np.log(2 * np.pi) + logdet + r @ Sr)
    pred = X0 @ beta + s2 * corr(cdist(S0, S), ph) @ Sr
    return dict(ll=ll, pred=pred, beta=beta)


S, y, S0, truth0 = simulate_rents(2000, 200, 2110)
X, X0 = design(S), design(S0)
ex, t_ex = timed(exact_fit, S, y, X, S0, X0)
print(f"exact: loglik={ex['ll']:.2f}  beta={np.round(ex['beta'], 3)}  "
      f"time={t_ex:.3f} s")
print("exact kriging, rmse against the true surface:",
      round(np.sqrt(np.mean((ex["pred"] - truth0)**2)), 3))
```

```python
def knots_grid(g):
    """Centers of a g-by-g grid of cells in the unit square."""
    c = (np.arange(g) + 0.5) / g
    xx, yy = np.meshgrid(c, c)
    return np.column_stack([xx.ravel(), yy.ravel()])


class LowRank:
    """Sigma = diag(d) + B Sa B', inverted and factored by Woodbury."""

    def __init__(self, d, B, Sa):
        self.B, self.Dinv = B, 1.0 / d
        cSa = linalg.cho_factor(Sa)
        Sa_inv = linalg.cho_solve(cSa, np.eye(len(Sa)))
        M = Sa_inv + B.T @ (B * self.Dinv[:, None])
        self.cM = linalg.cho_factor(M)
        self.logdet = (np.sum(np.log(d))
                       + 2.0 * np.sum(np.log(np.diag(cSa[0])))
                       + 2.0 * np.sum(np.log(np.diag(self.cM[0]))))

    def solve(self, V):
        """Sigma^{-1} V for a vector or matrix V, in O(n q^2) time."""
        w = self.Dinv if V.ndim == 1 else self.Dinv[:, None]
        DV = w * V
        return DV - w * (self.B @ linalg.cho_solve(self.cM, self.B.T @ DV))


def lowrank_fit(S, y, X, S0, X0, g, corr=r_exp, ph=phi, modified=True):
    """Predictive-process likelihood and kriging with g*g knots."""
    K = knots_grid(g)
    Rk = corr(dist(K), ph) + 1e-10 * np.eye(len(K))
    cRk = linalg.cho_factor(Rk)
    Rnk = corr(cdist(S, K), ph)
    B = linalg.cho_solve(cRk, Rnk.T).T          # kriging weights, Chapter 4
    deficit = 1.0 - np.sum(B * Rnk, axis=1)      # 1 - r' Rk^{-1} r
    d = tau2 + sigma2 * deficit if modified else np.full(len(S), tau2)
    model = LowRank(d, B, sigma2 * Rk)
    Sx, Sy = model.solve(X), model.solve(y)
    beta = np.linalg.solve(X.T @ Sx, X.T @ Sy)
    r = y - X @ beta
    Sr = model.solve(r)
    ll = -0.5 * (len(y) * np.log(2 * np.pi) + model.logdet + r @ Sr)
    alpha = sigma2 * (Rk @ (B.T @ Sr))           # E[alpha | y]
    B0 = linalg.cho_solve(cRk, corr(cdist(S0, K), ph).T).T
    return dict(ll=ll, pred=X0 @ beta + B0 @ alpha, deficit=deficit)


phi_m52 = 0.45 / 5.919       # same practical range as phi = 0.15
lowrank_rows = {}
for name, corr, ph in (("exponential", r_exp, phi),
                       ("Matern 5/2", r_m52, phi_m52)):
    exk = exact_fit(S, y, X, S0, X0, corr, ph=ph)
    print(name, "- exact loglik", round(exk["ll"], 2), "  exact rmse vs",
          "truth", round(np.sqrt(np.mean((exk["pred"] - truth0)**2)), 3),
          "  sd of exact predictions", round(np.std(exk["pred"]), 3))
    print("     q   plain  modified  rmse/exact  rmse/truth  sd(pred) "
          " deficit")
    rows = []
    for g in (4, 6, 8, 12, 16, 20, 25):
        plain = lowrank_fit(S, y, X, S0, X0, g, corr, ph, modified=False)
        mod = lowrank_fit(S, y, X, S0, X0, g, corr, ph, modified=True)
        rows.append((g * g, plain["ll"] - exk["ll"], mod["ll"] - exk["ll"],
                     mod["deficit"].mean()))
        rmse_ex = np.sqrt(np.mean((mod["pred"] - exk["pred"])**2))
        rmse_tr = np.sqrt(np.mean((mod["pred"] - truth0)**2))
        print(f"{g * g:6d} {rows[-1][1]:8.1f} {rows[-1][2]:9.1f} "
              f"{rmse_ex:11.3f} {rmse_tr:11.3f} {np.std(mod['pred']):9.3f} "
              f"{rows[-1][3]:8.3f}")
    lowrank_rows[name] = rows
```

**Reading the tables.** The first column is the number of knots. The next two give how far the low-rank log-likelihood is below the exact one, for the plain model $\boldsymbol{\Omega} = \tau^2\mathbf{I}_n$ and for the modified model of Section 13.2.5. The "rmse/exact" column is the root mean squared difference between the low-rank and the exact predictions at the 200 held-out listings, and "rmse/truth" is the error of the low-rank prediction against the true noise-free rent (the exact kriging has 0.741). The "sd(pred)" column is the standard deviation of the predictions, which is 2.219 for exact kriging.

For the exponential model the approximation is poor at modest ranks. With 144 knots the log-likelihood is still $80$ below the exact value for the modified model and $250$ below for the plain one, on a data set of 2,000 listings. Even 625 knots leave a shortfall of $10.9$ for the modified model. The predictions behave better than the likelihood: with 256 knots the prediction error against the true rent is $0.812$, against $0.741$ for exact kriging. The predictions are also smoother than the exact ones, with a standard deviation of $2.157$ against $2.219$ at 256 knots, and the gap shrinks as knots are added.

For the smooth Matérn model the picture is different. The same $q = 256$ knots leave a shortfall of only $5.9$, and $625$ knots leave $0.4$. The practical range of the two models is the same ($0.45$), so the difference is smoothness, not range, and Section 13.2.5 explains why.

### 13.2.5 The Cost of Low Rank

**The variance deficit.** The low-rank model replaces $u(\mathbf{s})$ by its kriging prediction from the knots. A prediction has less variance than the quantity it predicts. By the simple kriging variance of Section 4.2.2, the part of $u(\mathbf{s})$ that the knots do not determine has variance

$$
\sigma^2\left(1 - \mathbf{r}(\mathbf{s})'\mathbf{R}_K^{-1}\mathbf{r}(\mathbf{s})\right) \ge 0
$$

and the low-rank model puts none of it in $\boldsymbol{\Sigma}$. The deficit is zero at a knot, where $\mathbf{r}(\mathbf{s})$ is a column of $\mathbf{R}_K$ and the expression is $\sigma^2(1 - 1) = 0$, and it is largest in the gaps between knots and at the edges of the region. Figure 13.2 (left) shows it for 64 knots: the colors run from about 0.1 for listings next to a knot, through about 0.4 for listings in the gaps between knots, to about 0.6 for listings in the corners of the unit square. The average deficit in the table is the average of this quantity as a fraction of $\sigma^2$: 0.359 for 64 knots, falling to 0.118 for 625.

**What the likelihood does about it.** In the plain model the missing variance is absent. The data show it as extra scatter that the model attributes to nothing, and the likelihood suffers. The common remedy is the **modified predictive process**: add the deficit back on the diagonal, which is the reason for the general diagonal matrix $\boldsymbol{\Omega} = \tau^2\mathbf{I}_n + \sigma^2\operatorname{diag}(\text{deficit})$ in Section 13.2.2. The modified process has the correct variance at every listing, and only the correlations between listings are approximated. The table shows how much that matters: for the exponential model the modified likelihood is closer to the exact one than the plain one at every number of knots, by a factor of about $2.5$ with 16 knots and $6.5$ with 625.

**Why smoothness matters.** The exponential model is rough (Section 2.5.3): the variance of $u(\mathbf{s}) - u(\mathbf{s}')$ grows linearly with the distance between the two locations. Nearby listings that are far from every knot differ from their kriged values by an amount that shrinks only slowly as knots are added. The Matérn model with smoothness $5/2$ is smooth, and a handful of knots determines the surface in between them, so the deficit vanishes quickly (0.007 at 144 knots against 0.245 for the exponential). That is the general rule: low-rank models are good at the smooth, large-scale part of a spatial effect and poor at its small-scale roughness. In rent terms, they capture the broad pattern of neighborhoods and lose the block-to-block variation.

**Oversmoothing.** Because a low-rank prediction is a combination of $q$ smooth basis functions, its predictions are smoother than the exact ones. The shrinking standard deviations in the tables (2.004 with 16 knots up to 2.213 with 625, against 2.219 exact) are one symptom.

![Figure 13.2. Left: the variance deficit of the predictive process at the 2,000 listings, for a grid of 64 knots (red crosses). Listings near a knot lose almost no variance; listings at the corners lose the most. Right: how far the low-rank log-likelihood falls below the exact one, against the number of knots, for the exponential and Matérn 5/2 models, with and without the diagonal correction. The Matérn curves fall much faster.](../figures/ch13/fig_13_2_lowrank.png)

---

## 13.3 Covariance Tapering

### 13.3.1 The Idea and Why It Stays Valid

The exponential correlation is positive at every distance. Two listings at a distance of $20\phi$ have a correlation of $e^{-20} \approx 2 \times 10^{-9}$, which is zero for every practical purpose. Tapering sets such entries exactly to zero.

Let $\mathcal{T}_\theta(h)$ be a correlation function that is 1 at $h = 0$ and exactly zero for $h \ge \theta$. The **Wendland** function

$$
\mathcal{T}_\theta(h) = \left(1 - \frac{h}{\theta}\right)^4\left(4\frac{h}{\theta} + 1\right) \quad \text{for } 0 \le h < \theta, \qquad \mathcal{T}_\theta(h) = 0 \text{ for } h \ge \theta
$$

is a valid correlation function in up to three dimensions [@wendland1995], and the spherical model of Section 2.5.2 would also serve. (In this section $\theta$ is the taper range, not the parameter vector $\boldsymbol{\theta}$ of Chapter 5.) The **tapered covariance** is the product

$$
C_\theta(h) = \sigma^2 r(h;\phi)\,\mathcal{T}_\theta(h) \quad (h > 0), \qquad C_\theta(0) = \sigma^2 + \tau^2
$$

The product of two valid covariance functions is valid (Section 2.4.3), so the tapered matrix is positive definite whatever the locations. The proof there built a process whose covariance is the product, namely the product of two independent processes. Chapter 2 promised that this rule would be used here.

**A hand example.** Take three listings on a street at positions 0, 0.3, and 0.7, with exponential correlation and $\phi = 0.5$, tapered with range $\theta = 0.5$. The pair at distance $0.7$ is beyond the taper range, so its entry becomes exactly zero.

```python
def wendland(h, theta):
    """Compactly supported correlation, valid in up to three dimensions."""
    t = np.minimum(h / theta, 1.0)
    return (1.0 - t)**4 * (4.0 * t + 1.0)


t3 = np.array([0.0, 0.3, 0.7])
h3 = np.abs(t3[:, None] - t3[None, :])
R3 = np.exp(-h3 / 0.5)
T3 = wendland(h3, 0.5)
print("exponential correlation =\n", R3)
print("taper with range 0.5 =\n", T3)
print("tapered correlation =\n", R3 * T3)
print("eigenvalues of the tapered matrix:", np.linalg.eigvalsh(R3 * T3))
```

The tapered correlations are $0.0478$ for the first two listings, $0.003$ for the second and third, and exactly $0$ for the first and third. The eigenvalues are $0.9521$, $1$, and $1.0479$, all positive, as the product rule guarantees. The taper is strong at this range: it cuts the correlation of the first two listings from $0.5488$ to $0.0478$. This is a general feature of the taper, and it drives the results below.

### 13.3.2 Computation

The number of nonzero entries per row stays bounded only if the density of listings per unit area is fixed while the region grows. In the unit square with $n$ listings, a listing has about $n\pi\theta^2$ neighbors within $\theta$, so the fraction of nonzero entries is about $\pi\theta^2$ (less near the boundary).

SciPy has sparse storage and sparse solvers, but it has no sparse Cholesky factorization. It has a sparse LU factorization (`scipy.sparse.linalg.splu`), which for a symmetric positive definite matrix needs no pivoting and does roughly twice the work of a Cholesky factorization. The log-determinant is the sum of the logarithms of the absolute diagonal entries of the triangular factor $\mathbf{U}$, because the factor $\mathbf{L}$ has a unit diagonal. The code finds the pairs within distance $\theta$ with a k-d tree (SciPy's `cKDTree`), builds the sparse matrix, factors it, and evaluates the likelihood and kriging. It reports the **fill-in**: the number of nonzero entries in $\mathbf{L}$ and $\mathbf{U}$ relative to the number in $\boldsymbol{\Sigma}$. Factoring a sparse matrix creates new nonzeros, and the quality of the fill-reducing ordering decides how many. For the kriging weights, the cross-covariances between held-out and data listings are tapered in the same way.

We compare with exact computations on three data sets. The first two have $n = 2{,}000$: one simulated at $\phi = 0.15$, the model of Chapters 3 to 5, and one simulated at the shorter range $\phi = 0.03$, where listings are correlated only within a small neighborhood. The third has $n = 5{,}000$ at $\phi = 0.15$.

```python
def taper_matrix(S, theta, ph):
    """Sparse covariance: sigma2 * exp(-h/ph) * wendland(h/theta) + nugget."""
    n = len(S)
    pairs = cKDTree(S).query_pairs(theta, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    h = np.sqrt(np.sum((S[i] - S[j])**2, axis=1))
    v = sigma2 * r_exp(h, ph) * wendland(h, theta)
    rows = np.concatenate([i, j, np.arange(n)])
    cols = np.concatenate([j, i, np.arange(n)])
    vals = np.concatenate([v, v, np.full(n, sigma2 + tau2)])
    return sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))


def taper_fit(S, y, X, S0, X0, theta, ph=phi):
    """Tapered likelihood and kriging with a sparse LU factorization."""
    Sigma = taper_matrix(S, theta, ph)
    lu = spla.splu(Sigma, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
    logdet = np.sum(np.log(np.abs(lu.U.diagonal())))
    Sx, Sy = lu.solve(X), lu.solve(y)
    beta = np.linalg.solve(X.T @ Sx, X.T @ Sy)
    r = y - X @ beta
    Sr = lu.solve(r)
    ll = -0.5 * (len(y) * np.log(2 * np.pi) + logdet + r @ Sr)
    Hc = cdist(S0, S)
    c0 = sigma2 * r_exp(Hc, ph) * wendland(Hc, theta)
    return dict(ll=ll, pred=X0 @ beta + c0 @ Sr, density=Sigma.nnz / len(S)**2,
                fill=(lu.L.nnz + lu.U.nnz) / Sigma.nnz)


taper_rows = []
for n_t, ph, thetas in ((2000, 0.15, (0.05, 0.1, 0.2, 0.3, 0.45)),
                        (2000, 0.03, (0.05, 0.1, 0.2, 0.3, 0.45)),
                        (5000, 0.15, (0.1, 0.2, 0.3))):
    St, yt, S0t, _ = simulate_rents(n_t, 200, 2110, ph)
    Xt, X0t = design(St), design(S0t)
    ext, t_dense = timed(exact_fit, St, yt, Xt, S0t, X0t, ph=ph)
    print(f"n={n_t}, phi={ph}: dense Cholesky time {t_dense:.2f} s")
    print("  theta  density  loglik error  rmse/exact  LU fill   time (s)")
    for theta in thetas:
        tp, t_sp = timed(taper_fit, St, yt, Xt, S0t, X0t, theta, ph)
        rmse = np.sqrt(np.mean((tp["pred"] - ext["pred"])**2))
        taper_rows.append((n_t, ph, theta, tp["ll"] - ext["ll"]))
        print(f"  {theta:5.2f} {tp['density']:8.3f} "
              f"{tp['ll'] - ext['ll']:13.1f} {rmse:11.3f} "
              f"{tp['fill']:8.1f} {t_sp:10.2f}")
```

### 13.3.3 What the Tables Show

**Accuracy is governed by the ratio of the taper range to the correlation range, not by how sparse the matrix is.** With $\phi = 0.15$ and $\theta = 0.2$, the log-likelihood of the 2,000 listings is $61.4$ below the exact value, even though only 10.4% of the entries are nonzero. With $\phi = 0.03$ the same $\theta = 0.2$, now more than six times the correlation range, gives a shortfall of $2.3$ at the same density. The taper multiplies the correlation by $\mathcal{T}_\theta(h)$ at every distance, including short ones, so unless $\theta$ is several times $\phi$ the correlations that matter are distorted (the hand example showed a factor of $0.087$ at $h/\theta = 0.6$).

**Predictions tolerate tapering better than likelihoods do.** At $\phi = 0.15$ and $\theta = 0.2$, the kriging predictions differ from the exact ones by a root mean squared $0.158$, which adds little to the exact kriging error of $0.741$ against the truth. A likelihood shortfall of $61$ would, however, shift the parameters an optimizer chooses. Tapering is more useful for prediction than for fitting, unless the range is short.

**Sparse LU does not beat the dense Cholesky factorization at these sizes unless the matrix is very sparse.** At $n = 5{,}000$ the dense factorization takes about $1.1$ seconds. Tapering with $\theta = 0.1$ (2.9% of the entries) takes under half a second, but at the more accurate $\theta = 0.2$ it takes about $1.2$ seconds, no faster than dense, and at $\theta = 0.3$ about $3$ seconds (timings vary a little from run to run). The fill-in factors of $2$ to $5$ show why: the sparse factors have several times as many nonzeros as the matrix.

**The benefit is for much larger problems.** For $n = 20{,}000$ listings in the same unit square and a taper range of $0.05$ (0.76% of the entries nonzero), the sparse LU factorization of Section 13.7 takes about 29 seconds, with a fill-in of 8.2. That is a sparse problem, but a slow one, and a taper range of $0.05$ is far too short for the exponential model with $\phi = 0.15$. Tapering pays when the study region is large compared with the correlation range, so that $\theta \gg \phi$ still leaves a sparse matrix: a continent of listings, not a city. It also needs a good sparse Cholesky routine, which this book's toolset does not provide. Iterative solvers such as conjugate gradients (Exercise 7) avoid the factorization altogether.

---

## 13.4 Vecchia's Approximation

### 13.4.1 The Factorization of a Density

The chain rule of probability writes the joint density of $\mathbf{y} = (y_1, \dots, y_n)'$ as

$$
p(\mathbf{y}) = p(y_1)\prod_{i=2}^n p(y_i \mid y_1, \dots, y_{i-1})
$$

for any ordering of the observations. For a Gaussian vector each conditional is a normal distribution whose mean is a linear regression on the earlier values, and these are exactly the kriging predictors of Chapter 4. Write $\boldsymbol{\mu}$ for the mean vector. Then

$$
y_i \mid y_{1:(i-1)} \sim N\!\left(\mu_i + \mathbf{b}_i'(\mathbf{y}_{1:(i-1)} - \boldsymbol{\mu}_{1:(i-1)}),\ F_i\right), \qquad \mathbf{b}_i = \boldsymbol{\Sigma}_{<i}^{-1}\boldsymbol{\Sigma}_{<i,i}, \quad F_i = \Sigma_{ii} - \boldsymbol{\Sigma}_{i,<i}\boldsymbol{\Sigma}_{<i}^{-1}\boldsymbol{\Sigma}_{<i,i}
$$

Here $\boldsymbol{\Sigma}_{<i}$ is the covariance matrix of the first $i - 1$ observations and $\boldsymbol{\Sigma}_{<i,i}$ the covariances between them and $y_i$. The vector $\mathbf{b}_i$ holds the simple kriging weights of Section 4.2.2, and $F_i$ is the simple kriging variance. Collect the weights in a strictly lower triangular $n \times n$ matrix $\boldsymbol{\Gamma}$, whose row $i$ holds $\mathbf{b}_i'$ in columns $1, \dots, i-1$. Then the vector $(\mathbf{I} - \boldsymbol{\Gamma})(\mathbf{y} - \boldsymbol{\mu})$ has independent entries with variances $F_1, \dots, F_n$, so with $\mathbf{F} = \operatorname{diag}(F_i)$,

$$
\boldsymbol{\Sigma}^{-1} = (\mathbf{I} - \boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I} - \boldsymbol{\Gamma}), \qquad \log\lvert\boldsymbol{\Sigma}\rvert = \sum_{i=1}^n\log F_i
$$

This is the Cholesky factorization in disguise: $\mathbf{L}^{-1} = \mathbf{F}^{-1/2}(\mathbf{I} - \boldsymbol{\Gamma})$ for the factor $\mathbf{L}$ of Section 2.7. It confirms the column-by-column algorithm of that section, where each $L_{jj}^2$ was the variance of $Y_j$ that earlier values cannot explain. The matrix $\boldsymbol{\Gamma}$ is dense, though, and computing it costs $O(n^3)$.

**The approximation.** Vecchia's idea [@vecchia1988] is to condition each $y_i$ on only a small subset $\mathcal{C}_i \subseteq \{1, \dots, i-1\}$ of earlier observations, with $\lvert\mathcal{C}_i\rvert \le k$:

$$
p(\mathbf{y}) \approx p(y_1)\prod_{i=2}^n p(y_i \mid \mathbf{y}_{\mathcal{C}_i})
$$

Each conditional is again a regression, now on at most $k$ values, with weights $\mathbf{b}_i = \boldsymbol{\Sigma}_{\mathcal{C}_i}^{-1}\boldsymbol{\Sigma}_{\mathcal{C}_i,i}$ and variance $F_i$ computed from $\boldsymbol{\Sigma}_{\mathcal{C}_i}$ alone. The matrix $\boldsymbol{\Gamma}$ now has at most $k$ nonzeros per row. The right-hand side is the density of a Gaussian distribution with precision matrix $\widetilde{\mathbf{Q}} = (\mathbf{I} - \boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I} - \boldsymbol{\Gamma})$, which is positive definite because every $F_i$ is positive. So the approximation is a legitimate joint distribution, not merely a likelihood trick. (For the chosen ordering and neighbor sets it is known to minimize the Kullback-Leibler divergence from the exact distribution among Gaussian models with this sparsity; we do not prove that.)

The cost drops sharply. Each $i$ needs one $k \times k$ solve, $O(k^3)$ operations, so the likelihood costs $O(nk^3)$, linear in $n$.

**The connection to Chapter 8.** In a precision matrix a zero entry means conditional independence (Section 8.1.4). The approximation imposes conditional independence: given its $k$ neighbors in $\mathcal{C}_i$, the value $y_i$ is independent of every other earlier value. The precision matrix $\widetilde{\mathbf{Q}}$ is sparse. Entry $(j, l)$ can be nonzero only if $j$ and $l$ are both parents of the same observation, or one is a parent of the other, so the part of $\widetilde{\mathbf{Q}}$ contributed by observation $i$ has at most $(k + 1)^2$ nonzeros, and the whole matrix has at most $n(k + 1)^2$. An individual row can be longer, because one point can be a neighbor of many later points, and in a maximin ordering a few early points can be neighbors of many later ones (Exercise 9 asks you to count the nonzeros per row and in total for 1,000 random locations and $k = 3$). The weights $\mathbf{b}_i$ are the same kind of weights as $-Q_{ij}/Q_{ii}$ in Section 8.1.2, a regression of a value on its neighbors, but they are sequential: only earlier points enter, so they differ from the full-conditional weights of Section 8.1.2, and the neighborhood is chosen from the data locations instead of fixed by a map.

**Estimating $\boldsymbol{\beta}$.** With a mean $\mathbf{X}\boldsymbol{\beta}$, the filter $(\mathbf{I} - \boldsymbol{\Gamma})$ is applied to $\mathbf{y}$ and to the columns of $\mathbf{X}$ alike, and the profile GLS estimate is the ordinary least squares fit of the filtered response on the filtered design, scaled by $F_i^{-1/2}$. This is the decorrelation argument of Section 1.3.5 with $\mathbf{L}^{-1}$ replaced by the sparse filter, at a cost of $O(nkp)$.

### 13.4.2 By Hand: Four Points on a Line, and the Block

On a line the exponential covariance has a special property: $r(h_1 + h_2) = r(h_1)r(h_2)$. Given $y_{i-1}$, the earlier values add no information about $y_i$, so the process is a Markov chain. Vecchia's approximation with $k = 1$ and $\mathcal{C}_i = \{i - 1\}$ is therefore exact. The first block of the code checks this for four points at positions $0, 0.3, 0.7, 1.2$ with $\phi = 0.5$ and no nugget, and prints $\mathbf{I} - \boldsymbol{\Gamma}$, $\mathbf{F}$, and the resulting precision matrix. The second block takes the four apartments at the corners of a unit block from Chapters 1 and 2, where the process is not Markov, and measures how far the Vecchia model is from the truth with the Kullback-Leibler divergence.

```python
def vecchia_factor_dense(Sigma, parents):
    """Dense (I - B), F for a covariance matrix and parent sets C_i."""
    n = len(Sigma)
    IB = np.eye(n)
    F = np.empty(n)
    for i in range(n):
        C = parents[i]
        if len(C) == 0:
            F[i] = Sigma[i, i]
            continue
        b = np.linalg.solve(Sigma[np.ix_(C, C)], Sigma[C, i])
        IB[i, C] = -b
        F[i] = Sigma[i, i] - Sigma[i, C] @ b
    return IB, F


# Four points on a line, exponential correlation (no nugget).
t4 = np.array([0.0, 0.3, 0.7, 1.2])
Sig_line = np.exp(-np.abs(t4[:, None] - t4[None, :]) / 0.5)
IB, F = vecchia_factor_dense(Sig_line, [[], [0], [1], [2]])
Q_vec = IB.T @ np.diag(1.0 / F) @ IB
print("I - B =\n", IB)
print("F =", F)
print("Vecchia precision =\n", Q_vec)
print("exact precision =\n", np.linalg.inv(Sig_line))
print("largest difference:", np.abs(Q_vec - np.linalg.inv(Sig_line)).max())

# The four apartments at the corners of the block (Chapters 1 and 2).
S_blk = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Sig_blk = np.exp(-dist(S_blk))
for name, parents in (("k=1", [[], [0], [0], [1]]),
                      ("k=2", [[], [0], [0, 1], [1, 2]])):
    IB, F = vecchia_factor_dense(Sig_blk, parents)
    Q_vec = IB.T @ np.diag(1.0 / F) @ IB
    Sig_vec = np.linalg.inv(Q_vec)
    kl = 0.5 * (np.trace(np.linalg.solve(Sig_vec, Sig_blk)) - 4
                + np.linalg.slogdet(Sig_vec)[1]
                - np.linalg.slogdet(Sig_blk)[1])
    print(name, "KL divergence of the Vecchia model from the exact one:",
          f"{kl:.5f}")
```

The weights are $\Gamma_{21} = e^{-0.3/0.5} = 0.5488$, $\Gamma_{32} = e^{-0.4/0.5} = 0.4493$, and $\Gamma_{43} = e^{-0.5/0.5} = 0.3679$, each a correlation between neighbors, and the conditional variances are $F_i = 1 - \Gamma_{i,i-1}^2$. The Vecchia precision matrix is tridiagonal and matches the exact precision matrix to $2 \times 10^{-16}$: the exact inverse of this covariance matrix has zeros where it has no conditional dependence, which is Chapter 8's lesson in miniature.

For the four apartments of the block, the first-order approximation has a divergence of $0.0584$ from the exact distribution and the second-order approximation (each apartment conditioned on up to two earlier ones) has $0.0005$. Adding a second neighbor makes the approximation about a hundred times better.

### 13.4.3 Orderings, Neighbors, and a Fast Implementation

Two choices define a Vecchia approximation: the order of the observations and the neighbor sets. We take $\mathcal{C}_i$ to be the $k$ earlier observations nearest to $\mathbf{s}_i$. Nearby points carry most of the information, and the points beyond them are largely screened by the points between (the negative and near-zero weights of Section 4.2.3). The ordering matters because it determines which points are "earlier". Three orderings are common.

- **Random.** Early points are spread over the region, but a late point's neighbors are close by, since most points are already placed.
- **Coordinate.** Sorting by the east coordinate puts every neighbor to one side, so a point is predicted from a half-plane of neighbors.
- **Maximin.** Start from the point nearest the center and repeatedly add the point farthest from all points chosen so far. The first points cover the region sparsely, and each later point fills a gap. This gives early points good neighbors at all scales.

The code implements the greedy maximin ordering, the search for the $k$ nearest earlier points (by a blocked brute-force search, which is $O(n^2)$ in this teaching implementation; production codes use tree searches), and the vectorized computation of all $\mathbf{b}_i$ and $F_i$ at once. The last step stacks the $k \times k$ covariance matrices of all neighbor sets into a three-dimensional array, so that one call to `np.linalg.solve` replaces $n$ small solves. Padded neighbor slots (for the first few points, which have fewer than $k$ predecessors) are given an identity row and zero cross-covariance, so they receive weight zero. The table compares the three orderings on the $n = 2{,}000$ data set, at the true parameters.

```python
def maximin_order(S):
    """Greedy ordering: each point is the farthest from those before it."""
    first = np.argmin(np.sum((S - S.mean(axis=0))**2, axis=1))
    order = np.empty(len(S), dtype=int)
    order[0] = first
    mind = np.sqrt(np.sum((S - S[first])**2, axis=1))
    for i in range(1, len(S)):
        j = int(np.argmax(mind))
        order[i] = j
        mind = np.minimum(mind, np.sqrt(np.sum((S - S[j])**2, axis=1)))
        mind[j] = -1.0
    return order


def previous_neighbors(S, k, chunk=500):
    """For each row i, its k nearest rows j < i (padded with -1)."""
    n = len(S)
    idx = -np.ones((n, k), dtype=int)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        D = cdist(S[a:b], S[:b])
        later = np.arange(b)[None, :] >= a + np.arange(b - a)[:, None]
        D[later] = np.inf
        kk = min(k, b)
        part = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        part[~np.isfinite(np.take_along_axis(D, part, axis=1))] = -1
        idx[a:b, :kk] = part
    return idx


def vecchia_parts(S, idx, s2, t2, ph):
    """Regression weights b_i and conditional variances F_i, all i at once."""
    n, k = idx.shape
    valid = idx >= 0
    Sn = S[np.where(valid, idx, 0)]
    D = np.sqrt(((Sn[:, :, None, :] - Sn[:, None, :, :])**2).sum(axis=3))
    C = np.where(valid[:, :, None] & valid[:, None, :],
                 s2 * r_exp(D, ph), 0.0)
    C[:, np.arange(k), np.arange(k)] = np.where(valid, s2 + t2, 1.0)
    d0 = np.sqrt(((Sn - S[:, None, :])**2).sum(axis=2))
    c = np.where(valid, s2 * r_exp(d0, ph), 0.0)
    b = np.linalg.solve(C, c[:, :, None])[:, :, 0]
    return b, s2 + t2 - np.sum(b * c, axis=1)


def vecchia_filter(M, idx, b):
    """Apply (I - B) to a vector or to the columns of a matrix."""
    safe = np.where(idx >= 0, idx, 0)
    if M.ndim == 1:
        return M - np.sum(b * M[safe], axis=1)
    return M - np.einsum("ik,ikm->im", b, M[safe])


def vecchia_loglik(y, X, S, idx, s2, t2, ph):
    """Vecchia log-likelihood; beta by OLS on the filtered data."""
    b, F = vecchia_parts(S, idx, s2, t2, ph)
    ys = vecchia_filter(y, idx, b) / np.sqrt(F)
    Xs = vecchia_filter(X, idx, b) / np.sqrt(F)[:, None]
    beta = np.linalg.solve(Xs.T @ Xs, Xs.T @ ys)
    r = ys - Xs @ beta
    ll = -0.5 * (len(y) * np.log(2 * np.pi) + np.sum(np.log(F)) + r @ r)
    return ll, beta


orders = {"random": np.random.default_rng(1).permutation(len(S)),
          "x-sorted": np.argsort(S[:, 0]),
          "maximin": maximin_order(S)}
vecchia_rows = {}
print("loglik error against the exact likelihood (n = 2,000)")
print("  k   random   x-sorted   maximin")
for k in (1, 2, 5, 10, 20, 40):
    errs = []
    for name, o in orders.items():
        So, yo, Xo = S[o], y[o], X[o]
        ll, _ = vecchia_loglik(yo, Xo, So, previous_neighbors(So, k),
                               sigma2, tau2, phi)
        errs.append(ll - ex["ll"])
        vecchia_rows.setdefault(name, []).append((k, ll - ex["ll"]))
    print(f"{k:3d} {errs[0]:8.2f} {errs[1]:9.2f} {errs[2]:9.2f}")
```

With one or two neighbors the approximation is poor for every ordering (errors between $-340$ and $-130$). From five neighbors on it improves quickly. With $k = 10$ the error is about $-7$ for random and maximin ordering and $-9$ for coordinate ordering; with $k = 20$ the maximin error is $-0.02$, random is $+0.35$, and coordinate ordering is still $-3.9$. (The error can be positive because the approximation is a different probability model, not a lower bound on the exact likelihood.) The ordering by coordinate lags because every point's neighbors sit on one side.

Compare with the earlier approximations at the same $n = 2{,}000$ and the same parameters. Maximin Vecchia with $k = 10$ is within $7$ of the exact log-likelihood, while the modified low-rank model with 625 knots is $11$ below it and the tapered model with $\theta = 0.3$ is $29$ below it. Vecchia is more accurate for much less computation, as the benchmark of Section 13.7 shows. Figure 13.3 (right) shows the three orderings.

![Figure 13.3. Left: log-likelihood shortfall of the tapered covariance against the taper range, for two correlation ranges. Right: absolute log-likelihood error of Vecchia's approximation against the number of neighbors, for three orderings. The vertical axes are logarithmic.](../figures/ch13/fig_13_3_taper_vecchia.png)

### 13.4.4 A City With 20,000 Listings

Now the metropolitan data set. We cannot simulate 20,000 listings from the exponential model exactly. We use the Vecchia model itself with $k = 30$ neighbors as the data-generating process: sample $y_1$, then each $y_i$ given its 30 earlier neighbors, in the maximin order, which is a sequential loop of $O(n)$ small steps. Section 13.4.3 showed that maximin Vecchia matches the exact likelihood closely with 20 or 40 neighbors (errors of $0.04$ or less at $n = 2{,}000$), but this data set is **not** an exact draw from the exponential model, and the parameters recovered below are the parameters of that approximate model.

We then fit $\sigma^2$, $\tau^2$, and $\phi$ by maximizing the Vecchia likelihood with $k = 10$, with the Nelder-Mead search over the logarithms of the parameters used in Chapter 3. Finally we evaluate the likelihood at the fitted parameters with $k = 5, 10, 20$, and $30$ to see how much the approximation matters.

```python
# A city with 20,000 listings. Exact simulation is out of reach, so the rents
# are drawn from the Vecchia model with k = 30 neighbors (a close approximation
# to the exponential model; see the text) and fitted with k = 10.
n_big = 20_000
rng = np.random.default_rng(2110)
S_big = rng.uniform(size=(n_big, 2))
t0 = time.perf_counter()
S_big = S_big[maximin_order(S_big)]
t_order = time.perf_counter() - t0
X_big = design(S_big)
t0 = time.perf_counter()
idx30 = previous_neighbors(S_big, 30)
t_nb = time.perf_counter() - t0
print(f"maximin ordering {t_order:.1f} s; 30-neighbor search {t_nb:.1f} s")
b30, F30 = vecchia_parts(S_big, idx30, sigma2, tau2, phi)
safe30 = np.where(idx30 >= 0, idx30, 0)
z = rng.standard_normal(n_big)
r_big = np.zeros(n_big)
for i in range(n_big):           # sequential: y_i given its 30 parents
    r_big[i] = b30[i] @ r_big[safe30[i]] + np.sqrt(F30[i]) * z[i]
y_big = X_big @ np.array([18.0, 6.0]) + r_big


def neg_loglik(log_theta, idx):
    s2, t2, ph = np.exp(log_theta)
    return -vecchia_loglik(y_big, X_big, S_big, idx, s2, t2, ph)[0]


idx10 = previous_neighbors(S_big, 10)
start = np.log([1.0, 0.5, 0.05])
fit, t_fit = timed(optimize.minimize, neg_loglik, start, args=(idx10,),
                   method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-3, maxiter=300))
s2_hat, t2_hat, ph_hat = np.exp(fit.x)
print(f"n={n_big}: fit with k=10 took {t_fit:.1f} s "
      f"({fit.nfev} likelihood evaluations)")
print(f"estimates: sigma2={s2_hat:.3f} (truth 4), tau2={t2_hat:.3f} "
      f"(truth 1), phi={ph_hat:.4f} (truth 0.15)")
print(f"sigma2/phi = {s2_hat / ph_hat:.2f} (truth {sigma2 / phi:.2f})")
for k in (5, 10, 20, 30):
    idx = previous_neighbors(S_big, k)
    (ll, beta), t_eval = timed(vecchia_loglik, y_big, X_big, S_big, idx,
                               s2_hat, t2_hat, ph_hat)
    print(f"k={k:2d}: loglik at the estimates = {ll:10.1f}, "
          f"one evaluation takes {t_eval:.3f} s")
```

The fit takes about 16 seconds for 212 evaluations, or $0.076$ seconds per evaluation at $k = 10$. One exact evaluation at $n = 20{,}000$ would take over a minute by the extrapolation of Section 13.1.2, and the fit needs 212 of them.

The estimates reproduce the structure we learned in Chapter 5. The nugget $\tau^2 = 1.018$ is recovered almost exactly, which is what we expect when the listings are dense enough to contain many close pairs. The individual estimates $\hat\sigma^2 = 3.52$ and $\hat\phi = 0.136$ are off from the true values of $4$ and $0.15$ in the direction of the ridge of Section 5.4.4, but their ratio $\hat\sigma^2/\hat\phi = 25.84$ is close to the true $26.67$, which is the quantity that infill sampling in a fixed region determines.

The log-likelihood at the fitted parameters rises with $k$: $-32072.7$ for $k = 5$, $-31616.6$ for $k = 10$, $-31500.9$ for $k = 20$, and $-31488.8$ for $k = 30$. The gain from 5 to 10 neighbors is $456$, from 10 to 20 it is $116$, and from 20 to 30 it is $12$. The cost of an evaluation grows from $0.024$ to $0.56$ seconds over the same range. For fitting, $k = 10$ to $20$ is a reasonable compromise on this data set. The maximin ordering and the neighbor search are done once, because they do not depend on the parameters; in this implementation they take $6.0$ and $2.2$ seconds (for the neighbor search with 30 neighbors) at $n = 20{,}000$.

---

## 13.5 The Bridge Between Parts II and III: Matérn Fields as Markov Fields

### 13.5.1 The Result of Lindgren, Rue, and Lindström

Parts II and III built two families of models: Part II specified a covariance function of distance, such as the Matérn function, and Part III specified a precision matrix through neighbors. The two are less different than they look. A theorem of Lindgren, Rue, and Lindström [@lindgren2011] states that a Matérn field is the solution of a stochastic differential equation, and that a discretization of the equation is a Markov field with a sparse precision matrix. We state the result without the finite-element derivation, which needs tools outside this book.

Let $\nabla^2$ be the Laplace operator (the sum of second derivatives) and let $\mathcal{W}(\mathbf{s})$ be Gaussian white noise. The field $u(\mathbf{s})$ on the plane that solves

$$
\left(\kappa^2 - \nabla^2\right)^{\alpha/2}u(\mathbf{s}) = \mathcal{W}(\mathbf{s})
$$

has a Matérn covariance with smoothness $\nu = \alpha - d/2$ in $d$ dimensions, and range parameter $\phi = 1/\kappa$ in the parameterization of Section 2.5.3. (Here $\kappa$ is the inverse range, and it has nothing to do with the knots of Section 13.2.) In the plane, $d = 2$, and the choice $\alpha = 2$ gives $\nu = 1$. With unit white noise the variance of the field is $1/(4\pi\kappa^2)$ when $\nu = 1$.

**Discretizing on a grid.** On a grid of cells with spacing 1, the graph Laplacian $\mathcal{L} = \mathbf{D} - \mathbf{A}$ of Section 8.3 acts on a vector of cell values by $(\mathcal{L}\mathbf{u})_i = \sum_{j \sim i}(u_i - u_j)$, a sum of differences with the four neighbors. That is minus the standard finite-difference approximation to $\nabla^2 u$ at cell $i$. So the matrix $\kappa^2\mathbf{I} + \mathcal{L}$ approximates the operator $\kappa^2 - \nabla^2$. The equation with $\alpha = 2$ says that $(\kappa^2\mathbf{I} + \mathcal{L})\mathbf{u}$ is a vector of independent standard normal variables, so by Tool 1 of Chapter 0 the precision matrix of $\mathbf{u}$ is

$$
\mathbf{Q} = (\kappa^2\mathbf{I} + \mathcal{L})'(\kappa^2\mathbf{I} + \mathcal{L}) = (\kappa^2\mathbf{I} + \mathcal{L})^2
$$

This matrix has nonzero entries only between cells at most two steps apart in the grid: at most 13 per row, and the field is a Gaussian Markov random field. A precision matrix of that sparsity is what Chapter 8 computed with.

The case $\alpha = 1$ gives $\mathbf{Q} = \kappa^2\mathbf{I} + \mathcal{L}$, which is the proper CAR model of Section 8.2 and the Leroux model of Section 9.3.5 in disguise: on a regular grid $\mathbf{D} = 4\mathbf{I}$ in the interior, and $\mathbf{D} - \rho\mathbf{A} = 4(1 - \rho)\mathbf{I} + \rho\mathcal{L}$ is proportional to $\kappa^2\mathbf{I} + \mathcal{L}$ with $\kappa^2 = 4(1 - \rho)/\rho$. The smoothness is $\nu = 1 - 1 = 0$, which is not a Matérn field with finite variance: its variance grows without bound as the grid is refined (a known result that we do not prove). This is a second reason, beyond the one in Section 8.2.4 (marginal variances differ by region, with an edge effect), why the CAR parameter $\sigma^2$ is not a marginal variance: the variance of the $\alpha = 1$ field is not even defined in the continuum limit, so $\sigma^2$ cannot be compared across maps of different resolution. The square $\alpha = 2$ has $\nu = 1$ and a finite limit.

**It is a SAR model.** In the interior of the grid, $\mathcal{L} = 4\mathbf{I} - \mathbf{A}$, so $\kappa^2\mathbf{I} + \mathcal{L} = (\kappa^2 + 4)(\mathbf{I} - \rho\mathbf{W})$ with $\mathbf{W} = \mathbf{A}/4$ and $\rho = 4/(\kappa^2 + 4)$. Since $\mathbf{W}$ is symmetric, $(\mathbf{I} - \rho\mathbf{W})^2 = (\mathbf{I} - \rho\mathbf{W})'(\mathbf{I} - \rho\mathbf{W})$, and therefore

$$
\mathbf{Q} = (\kappa^2 + 4)^2(\mathbf{I} - \rho\mathbf{W})'(\mathbf{I} - \rho\mathbf{W})
$$

which is the SAR precision of Section 7.2.3 up to scale. So on a regular grid, away from the boundary, the SAR model of Chapter 7 is a discretized Matérn field with smoothness 1 and range $1/\kappa$, and a spatial autoregressive parameter near 1 corresponds to a long range. (For $\kappa = 0.25$, $\rho = 0.9846$.)

### 13.5.2 A Numerical Check

The code builds the precision matrix $\mathbf{Q}$ on a $50 \times 50$ grid with $\kappa = 0.25$ (a range of 4 cells), inverts it densely to see the covariance matrix, and compares the correlations with the Matérn function with $\nu = 1$, which is $x K_1(x)$ with $x = \kappa h$. It also checks that an interior row of $\mathbf{Q}$ equals the SAR precision above, and shows that the sparse factorization remains cheap on a grid of 40,000 cells.

```python
def grid_laplacian(m):
    """Graph Laplacian D - A of an m-by-m grid of cells (rook neighbors)."""
    ii = np.arange(m * m).reshape(m, m)
    a = np.concatenate([ii[:, :-1].ravel(), ii[:-1, :].ravel()])
    b = np.concatenate([ii[:, 1:].ravel(), ii[1:, :].ravel()])
    A = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(m * m, m * m))
    A = (A + A.T).tocsr()
    return sparse.diags(np.asarray(A.sum(axis=1)).ravel()) - A, A


m, kappa = 50, 0.25
Lap, A_grid = grid_laplacian(m)
n_g = m * m
Q0 = kappa**2 * sparse.identity(n_g) + Lap
Q_gmrf = (Q0 @ Q0).tocsc()
cov = np.linalg.inv(Q_gmrf.toarray())
center, mid_edge, corner = (m // 2) * m + m // 2, m // 2, 0
print(f"nonzeros per row of Q: {Q_gmrf.nnz / n_g:.1f} "
      f"(interior rows: {Q_gmrf[center].nnz})")
v0 = cov[center, center]
print(f"variance at the center: {v0:.4f}; 1/(4 pi kappa^2) = "
      f"{1 / (4 * np.pi * kappa**2):.4f}")
print(f"variance on an edge / center: {cov[mid_edge, mid_edge] / v0:.3f}; "
      f"in a corner / center: {cov[corner, corner] / v0:.3f}")
print("lag   GMRF  Matern(nu=1)  (center row)   GMRF  Matern  (edge row)")
for h in (1, 2, 4, 8, 12):
    x = kappa * h
    mat = x * special.kv(1, x)
    c_in = cov[center, center + h] / v0
    c_ed = cov[mid_edge, mid_edge + h] / cov[mid_edge, mid_edge]
    print(f"{h:3d}  {c_in:6.4f}  {mat:6.4f}                  "
          f"{c_ed:6.4f}  {mat:6.4f}")

# Interior rows of Q equal a SAR precision (Chapter 7) with rho = 4/(kappa^2+4)
rho_sar = 4.0 / (kappa**2 + 4.0)
W_sym = A_grid / 4.0
IW = sparse.identity(n_g) - rho_sar * W_sym
Q_sar = ((kappa**2 + 4.0)**2 * (IW @ IW)).tocsr()
diff_row = np.abs(Q_gmrf[center].toarray() - Q_sar[center].toarray()).max()
print(f"SAR with rho = {rho_sar:.4f}: largest difference in a center row "
      f"of Q = {diff_row:.1e}")

# A grid of 40,000 cells: the precision is still cheap to factor.
Lap2, _ = grid_laplacian(200)
Q2 = ((0.25**2 * sparse.identity(40_000) + Lap2) @
      (0.25**2 * sparse.identity(40_000) + Lap2)).tocsc()
e_c = np.zeros(40_000)
e_c[20_100] = 1.0
lu2, t_q = timed(spla.splu, Q2, permc_spec="MMD_AT_PLUS_A")
col = lu2.solve(e_c)                 # one column of the covariance matrix
print(f"n=40,000: LU of Q takes {t_q:.2f} s, "
      f"fill {(lu2.L.nnz + lu2.U.nnz) / Q2.nnz:.1f}; the variance of the "
      f"cell is {col[20_100]:.3f} and the correlation 8 cells east is "
      f"{col[20_108] / col[20_100]:.4f}")
```

**The correlations.** In the center of the grid the correlation falls from $0.9218$ at one cell to $0.5846$ at four and $0.2712$ at eight, against $0.9368$, $0.6019$, and $0.2797$ for the Matérn function. The agreement is within $0.02$ everywhere. The finite-difference Laplacian is least accurate at short distances, which is where the largest differences are. The variance at the center is $1.3147$ against the theoretical $1/(4\pi\kappa^2) = 1.2732$, which is 3% higher.

**The boundary.** The graph Laplacian treats each edge of the grid as a wall with no flow, so the variance near the boundary is larger: $1.922$ times the central variance in the middle of an edge, and $3.716$ in a corner. These are close to 2 and 4, the factors from reflecting the field across one wall or two. Figure 13.4 (right) shows the variance rising towards the boundary. The practical lesson is to extend the grid beyond the study region by several ranges, so that the study region sits in the interior. The correlations along the edge (the open squares in Figure 13.4, left) match the Matérn curve even better than the interior ones: $0.6015$ at four cells against $0.6019$.

**The SAR check.** The largest difference between a center row of $\mathbf{Q}$ and the SAR precision with $\rho = 0.9846$ is $3.6 \times 10^{-15}$, which is rounding error. The two are the same matrix.

**Scale.** On a grid of 40,000 cells, the sparse LU factorization of $\mathbf{Q}$ takes about $0.20$ seconds, with fill-in $13.1$, and one solve gives a column of the covariance: the variance at a cell $(1.315)$ and the correlation at eight cells east $(0.2710)$, the same as in the small grid. This is the cost of sparse precision matrices: a few tenths of a second for 40,000 cells, where the dense covariance matrix of 40,000 cells would need 12.8 GB.

![Figure 13.4. Left: correlation along a row through the center of a $50 \times 50$ grid and along an edge row, for the precision $(\kappa^2\mathbf{I} + \mathcal{L})^2$ with $\kappa = 0.25$, against the Matérn correlation with smoothness 1 (black line). Right: marginal variance relative to the center. The variance rises to nearly four times the central value in the corners, because the graph Laplacian reflects the field at the boundary.](../figures/ch13/fig_13_4_gmrf.png)

### 13.5.3 Listings on a Grid: Woodbury With a Sparse Precision

Listings do not sit at the centers of grid cells, but we can assign each listing to its cell. Let $\mathbf{E}$ be the $n \times n_g$ matrix with $E_{ij} = 1$ if listing $i$ lies in cell $j$ and $0$ otherwise, where $n_g$ is the number of cells, and let the cell values $\mathbf{u}$ have the precision matrix $\mathbf{Q}$ scaled to give variance $\sigma^2$. Then $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{E}\mathbf{u} + \boldsymbol{\varepsilon}$, with

$$
\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n + \mathbf{E}\mathbf{Q}^{-1}\mathbf{E}'
$$

This is a diagonal plus low-rank matrix, with $\mathbf{B} = \mathbf{E}$ and $\boldsymbol{\Sigma}_\alpha^{-1} = \mathbf{Q}$. Section 13.2.2 applies unchanged: with $\boldsymbol{\Omega} = \tau^2\mathbf{I}_n$,

$$
\boldsymbol{\Sigma}^{-1} = \tau^{-2}\mathbf{I}_n - \tau^{-4}\mathbf{E}\left(\mathbf{Q} + \tau^{-2}\mathbf{E}'\mathbf{E}\right)^{-1}\mathbf{E}', \qquad \log\lvert\boldsymbol{\Sigma}\rvert = n\log\tau^2 - \log\lvert\mathbf{Q}\rvert + \log\lvert\mathbf{Q} + \tau^{-2}\mathbf{E}'\mathbf{E}\rvert
$$

The matrix $\mathbf{Q}$ is sparse, and $\mathbf{E}'\mathbf{E}$ is diagonal (it counts the listings in each cell), so $\mathbf{Q} + \tau^{-2}\mathbf{E}'\mathbf{E}$ is sparse too, and both log-determinants come from sparse factorizations. The role of the dense $q \times q$ matrix $\mathbf{M}$ is played by a sparse $n_g \times n_g$ matrix, and the number of cells can be much larger than the number of knots would be. The check below uses 2,000 listings placed at random in the cells of the $50 \times 50$ grid, and compares the dense computation with the sparse one.

```python
# Listings on the grid: Woodbury with a sparse precision (Section 13.5.3).
rng = np.random.default_rng(2110)
n_l = 2000
cell = rng.integers(0, n_g, n_l)               # the cell of each listing
A_map = sparse.csr_matrix((np.ones(n_l), (np.arange(n_l), cell)),
                          shape=(n_l, n_g))
scale = 4 * np.pi * kappa**2 / sigma2           # variance sigma2 at the center
Qs = Q_gmrf * scale                             # precision of u
Sigma_l = tau2 * np.eye(n_l) + (A_map @ (cov / scale)) @ A_map.T
y_l = rng.standard_normal(n_l)


def logdet_sparse(Msp):
    lu = spla.splu(Msp.tocsc(), permc_spec="MMD_AT_PLUS_A",
                   diag_pivot_thresh=0.0)
    return np.sum(np.log(np.abs(lu.U.diagonal()))), lu


ld_Q, _ = logdet_sparse(Qs)
ld_M, lu_M = logdet_sparse(Qs + A_map.T @ A_map / tau2)
ld_fast = n_l * np.log(tau2) - ld_Q + ld_M
w = A_map.T @ y_l / tau2
quad_fast = y_l @ y_l / tau2 - w @ lu_M.solve(w)
print(f"log|Sigma|: dense {np.linalg.slogdet(Sigma_l)[1]:.6f}, "
      f"sparse Woodbury {ld_fast:.6f}")
print(f"y'Sigma^-1 y: dense {y_l @ np.linalg.solve(Sigma_l, y_l):.6f}, "
      f"sparse Woodbury {quad_fast:.6f}")
```

The two log-determinants agree to six decimal places ($1263.826506$), and so do the quadratic forms ($1406.591908$). This closes a loop that began in Chapter 9: a latent Gaussian field with a sparse precision, observed with noise, is handled by one sparse factorization of the posterior precision $\mathbf{Q} + \tau^{-2}\mathbf{E}'\mathbf{E}$, which is the matrix $\mathbf{V} + \mathbf{Q}$ of the Laplace approximation of Section 9.5 with Gaussian errors. For irregular listings, the grid is replaced by a triangulated mesh and the finite-element method (Appendix D), but the algebra is the same.

---

## 13.6 Kronecker Products for Separable Space-Time Data

### 13.6.1 Separable Covariance

Suppose the rent of each of $n_S = 400$ neighborhoods is recorded in each of $n_T = 50$ weeks, giving $n = n_Sn_T = 20{,}000$ values. A **separable** space-time covariance assumes that the covariance between neighborhood $s$ in week $t$ and neighborhood $s'$ in week $t'$ factors:

$$
\operatorname{Cov}\!\left(Y(s, t), Y(s', t')\right) = C_T(t, t')\,C_S(s, s')
$$

If we stack the values week by week into a vector of length $n_Sn_T$, the covariance matrix is the **Kronecker product** $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ of the $n_T \times n_T$ temporal covariance matrix and the $n_S \times n_S$ spatial one. The Kronecker product of an $a \times b$ matrix $\mathbf{A}$ and a $c \times d$ matrix $\mathbf{B}$ is the $ac \times bd$ matrix whose $(i, j)$ block is $A_{ij}\mathbf{B}$. (Here $\mathbf{A}$ and $\mathbf{B}$ are generic matrices, unrelated to the adjacency and basis matrices.)

**By hand.** For $\mathbf{A} = \begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix}$ and $\mathbf{B} = \begin{bmatrix} 1 & 0.2 \\ 0.2 & 1 \end{bmatrix}$ the product is the $4 \times 4$ matrix printed first in the code below: four blocks, each a copy of $\mathbf{B}$ scaled by the entry of $\mathbf{A}$ at that position, so the upper left block is $\mathbf{B}$, the upper right block is $0.5\,\mathbf{B}$, and so on.

### 13.6.2 Four Properties, With Proofs

**Mixed product.** For conformable matrices, $(\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A}\mathbf{C}) \otimes (\mathbf{B}\mathbf{D})$. Proof: the $(i, j)$ block of the left side is $\sum_l (A_{il}\mathbf{B})(C_{lj}\mathbf{D}) = \left(\sum_l A_{il}C_{lj}\right)\mathbf{B}\mathbf{D}$, which is the $(i, j)$ block of the right side. Taking $\mathbf{C} = \mathbf{A}^{-1}$ and $\mathbf{D} = \mathbf{B}^{-1}$ shows that

$$
(\boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S)^{-1} = \boldsymbol{\Sigma}_T^{-1} \otimes \boldsymbol{\Sigma}_S^{-1}
$$

The same property shows that $(\mathbf{L}_T \otimes \mathbf{L}_S)(\mathbf{L}_T \otimes \mathbf{L}_S)' = \boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ when $\mathbf{L}_T$ and $\mathbf{L}_S$ are the Cholesky factors of the two matrices, so $\mathbf{L}_T \otimes \mathbf{L}_S$ is a lower triangular factor of $\boldsymbol{\Sigma}$.

**Determinant.** For a $p \times p$ matrix $\mathbf{A}$ and a $q \times q$ matrix $\mathbf{B}$, $\lvert\mathbf{A} \otimes \mathbf{B}\rvert = \lvert\mathbf{A}\rvert^q\lvert\mathbf{B}\rvert^p$. Proof: by the mixed product, $\mathbf{A} \otimes \mathbf{B} = (\mathbf{A} \otimes \mathbf{I}_q)(\mathbf{I}_p \otimes \mathbf{B})$. The matrix $\mathbf{I}_p \otimes \mathbf{B}$ is block diagonal with $p$ copies of $\mathbf{B}$, so its determinant is $\lvert\mathbf{B}\rvert^p$. The matrix $\mathbf{A} \otimes \mathbf{I}_q$ becomes block diagonal with $q$ copies of $\mathbf{A}$ after reordering rows and columns by the same permutation, which does not change a determinant, so its determinant is $\lvert\mathbf{A}\rvert^q$. For $\boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ with sizes $n_T$ and $n_S$,

$$
\log\lvert\boldsymbol{\Sigma}\rvert = n_S\log\lvert\boldsymbol{\Sigma}_T\rvert + n_T\log\lvert\boldsymbol{\Sigma}_S\rvert
$$

The exponent of each factor is the size of the other, because each factor is repeated once for every row of the other.

**The vec identity.** Let $\operatorname{vec}(\mathbf{X})$ stack the columns of a matrix $\mathbf{X}$ into one long vector. Then for conformable $\mathbf{A}$, $\mathbf{X}$, $\mathbf{B}$,

$$
(\mathbf{B}' \otimes \mathbf{A})\operatorname{vec}(\mathbf{X}) = \operatorname{vec}(\mathbf{A}\mathbf{X}\mathbf{B})
$$

Proof: column $j$ of $\mathbf{A}\mathbf{X}\mathbf{B}$ is $\sum_l B_{lj}\mathbf{A}\mathbf{x}_l$, where $\mathbf{x}_l$ is column $l$ of $\mathbf{X}$. The $j$th block of the vector $(\mathbf{B}' \otimes \mathbf{A})\operatorname{vec}(\mathbf{X})$ is $\sum_l (\mathbf{B}')_{jl}\mathbf{A}\mathbf{x}_l$, and $(\mathbf{B}')_{jl} = B_{lj}$. They agree.

**Consequences.** Arrange the data as an $n_S \times n_T$ matrix $\mathbf{Y}$ whose column $t$ is week $t$, so that $\operatorname{vec}(\mathbf{Y})$ is the stacked vector. The vec identity with $\mathbf{A} = \boldsymbol{\Sigma}_S$ and $\mathbf{B} = \boldsymbol{\Sigma}_T$ (symmetric) gives $\boldsymbol{\Sigma}\operatorname{vec}(\mathbf{Y}) = \operatorname{vec}(\boldsymbol{\Sigma}_S\mathbf{Y}\boldsymbol{\Sigma}_T)$, and with the inverses,

$$
\boldsymbol{\Sigma}^{-1}\operatorname{vec}(\mathbf{Y}) = \operatorname{vec}\!\left(\boldsymbol{\Sigma}_S^{-1}\mathbf{Y}\boldsymbol{\Sigma}_T^{-1}\right)
$$

A solve with an $n \times n$ matrix becomes two small solves, one with each factor. Simulation is equally cheap: if $\mathbf{Z}$ is an $n_S \times n_T$ matrix of independent standard normals, then $\mathbf{Y} = \mathbf{L}_S\mathbf{Z}\mathbf{L}_T'$ has $\operatorname{vec}(\mathbf{Y}) = (\mathbf{L}_T \otimes \mathbf{L}_S)\operatorname{vec}(\mathbf{Z})$, and its covariance is $\boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ by Tool 1.

The cost of the log-likelihood is $O(n_S^3 + n_T^3)$ for the two Cholesky factorizations plus $O(n_Sn_T(n_S + n_T))$ for the two solves, against $O\big((n_Sn_T)^3\big)$ for the dense computation.

The first block of code checks the determinant identity, the solve identity, and the vec identity against the dense $20 \times 20$ matrix for $n_S = 5$ sites and $n_T = 4$ weeks, and shows that the Kronecker log-likelihood equals the dense one. The second block does the same computation for the full $400 \times 50$ data set, which is $n = 20{,}000$.

```python
def corr_time(t, rate):
    """Exponential correlation matrix over time points."""
    return np.exp(-np.abs(t[:, None] - t[None, :]) / rate)


# Two tiny factors, by hand: every entry of the Kronecker product is visible.
A2 = np.array([[1.0, 0.5], [0.5, 1.0]])
B2 = np.array([[1.0, 0.2], [0.2, 1.0]])
print("A kron B =\n", np.kron(A2, B2))

# Small case: check the identities against the dense matrices.
rng = np.random.default_rng(2110)
nS, nT = 5, 4
S_k = rng.uniform(size=(nS, 2))
Sig_S = sigma2 * r_exp(dist(S_k), phi) + tau2 * np.eye(nS)
Sig_T = corr_time(np.arange(nT, dtype=float), 2.0)
Sig_ST = np.kron(Sig_T, Sig_S)                # index = time * nS + site
Y = rng.standard_normal((nS, nT))            # column j holds week j
vecY = Y.reshape(-1, order="F")
print("log|Sigma|, dense:", round(np.linalg.slogdet(Sig_ST)[1], 5),
      " with the exponents:",
      round(nS * np.linalg.slogdet(Sig_T)[1]
            + nT * np.linalg.slogdet(Sig_S)[1], 5))
dense_solve = np.linalg.solve(Sig_ST, vecY)
fast_solve = (np.linalg.solve(Sig_S, Y) @ np.linalg.inv(Sig_T)
              ).reshape(-1, order="F")
print("solve, largest difference:", np.abs(dense_solve - fast_solve).max())
print("vec identity, largest difference:",
      np.abs(Sig_ST @ vecY
             - (Sig_S @ Y @ Sig_T).reshape(-1, order="F")).max())


def kron_loglik(Y, Sig_S, Sig_T):
    """Log-likelihood of vec(Y) with covariance Sig_T kron Sig_S."""
    nS, nT = Y.shape
    cS, cT = linalg.cho_factor(Sig_S), linalg.cho_factor(Sig_T)
    Z = linalg.cho_solve(cT, linalg.cho_solve(cS, Y).T).T
    logdet = (nT * 2 * np.sum(np.log(np.diag(cS[0])))
              + nS * 2 * np.sum(np.log(np.diag(cT[0]))))
    return -0.5 * (nS * nT * np.log(2 * np.pi) + logdet + np.sum(Y * Z))


from scipy import stats
print("log-likelihood, Kronecker:", round(kron_loglik(Y, Sig_S, Sig_T), 5),
      " dense:", round(stats.multivariate_normal(
          np.zeros(nS * nT), Sig_ST).logpdf(vecY), 5))

# 400 neighborhoods observed for 50 weeks: n = 20,000.
nS, nT = 400, 50
S_k = rng.uniform(size=(nS, 2))
Sig_S = sigma2 * r_exp(dist(S_k), phi) + tau2 * np.eye(nS)
Sig_T = corr_time(np.arange(nT, dtype=float), 5.0)
Y = (np.linalg.cholesky(Sig_S) @ rng.standard_normal((nS, nT))
     @ np.linalg.cholesky(Sig_T).T)          # Chapter 2, one factor per side
ll_k, t_k = timed(kron_loglik, Y, Sig_S, Sig_T, repeat=3)
print(f"n={nS * nT}: Kronecker log-likelihood {ll_k:.1f} in {t_k:.4f} s; "
      f"dense Cholesky extrapolated: {c3 * (nS * nT)**3:.0f} s")

# How separable is a space-time covariance? Rearrange it and look at its
# singular values (a separable matrix has exactly one).
def rearranged(Sig, nT, nS):
    return (Sig.reshape(nT, nS, nT, nS).transpose(0, 2, 1, 3)
            .reshape(nT * nT, nS * nS))


nS, nT = 20, 10
H_s = dist(rng.uniform(size=(nS, 2)))
u_lag = np.abs(np.arange(nT)[:, None] - np.arange(nT)[None, :])
sep = np.kron(corr_time(np.arange(nT, dtype=float), 3.0), np.exp(-H_s / 0.2))
nonsep = np.exp(-np.sqrt((u_lag[:, None, :, None] / 3.0)**2
                         + (H_s[None, :, None, :] / 0.2)**2)
                ).reshape(nT * nS, nT * nS)
for name, M in (("separable", sep), ("nonseparable", nonsep)):
    sv = np.linalg.svd(rearranged(M, nT, nS), compute_uv=False)
    print(f"{name:13s} share of the first singular value: "
          f"{sv[0]**2 / np.sum(sv**2):.4f}")
for u in (0.0, 3.0, 9.0):                  # space-time correlation at h = 0.2
    print(f"time lag {u:3.0f}: correlation at h = 0.2, relative to h = 0: "
          f"separable {np.exp(-1.0):.3f}, "
          f"nonseparable {np.exp(-np.sqrt((u / 3.0)**2 + 1.0) + u / 3.0):.3f}")
```

The determinant of the dense matrix is $25.06972$ on the log scale, and the formula with exponents gives $25.06972$. The two solves agree to $3 \times 10^{-16}$, the vec identity to $2 \times 10^{-15}$, and the log-likelihoods are both $-32.8442$. For the full data set the Kronecker log-likelihood is $-26821.3$ and takes under a millisecond ($0.0006$ seconds), while the dense Cholesky factorization of a $20{,}000 \times 20{,}000$ matrix would take about $71$ seconds by the extrapolation of Section 13.1.2, if it fit in memory.

### 13.6.3 The Limits of Separability

Separability is an assumption, and it is a strong one. In a separable model the spatial correlation pattern is the same at every time lag, up to a factor: the ratio $C(h, u)/C(0, u)$ of the correlation at spatial distance $h$ to the correlation at distance 0 does not depend on the time lag $u$. For rents, a separable model says that the neighborhood effect has the same geographic range for a rent measured this week and rent measured last year. In many processes the range changes with the lag.

The last block above builds a nonseparable covariance, $\exp\{-\sqrt{(u/3)^2 + (h/0.2)^2}\}$, and shows the ratio at $h = 0.2$: $0.368$ at lag 0 for both the separable and nonseparable models, but at time lag 3 the separable ratio is still $0.368$ while the nonseparable ratio has risen to $0.661$, and at lag 9 it is $0.850$. A separability check on a covariance matrix is available by rearranging it into a matrix whose rows are indexed by the pairs of time points and whose columns by pairs of sites. The matrix has rank one exactly when the covariance is separable, so the share of the first squared singular value is a measure of how separable it is: $1.0000$ for the separable matrix and $0.9878$ for the nonseparable one. The nonseparable example is close to separable in that sense but still differs in the way the ratios show, and the Kronecker structure is lost.

Two other conditions are needed. The data must form a complete rectangle, every site observed in every week, because a missing value breaks the Kronecker structure of $\boldsymbol{\Sigma}$ (an imbalanced panel needs the Woodbury identity or an iterative solver). And a sum of a few Kronecker products, $\sum_r\boldsymbol{\Sigma}_{T,r} \otimes \boldsymbol{\Sigma}_{S,r}$, can represent some nonseparable covariances, but inverting such a sum is not as cheap. The same Kronecker structure holds for multivariate data on a common set of sites, with $\boldsymbol{\Sigma}_T$ replaced by the covariance matrix among the variables. Appendix D points to the literature on nonseparable and multivariate models.

---

## 13.7 Walkthrough and Lab 13

The hand checks of this chapter are placed where the methods are derived: the Woodbury identity in Section 13.2.3, the tapered matrix in Section 13.3.1, the Vecchia factor in Section 13.4.2, the Kronecker product in Section 13.6.1. This section runs the four methods on the same data and compares their cost and accuracy, then shows what happens beyond the sizes where the exact computation is possible.

### 13.7.1 A Benchmark Against the Exact Computation

For each of four sample sizes the code simulates data from the rent model exactly (up to 5,000 listings plus 200 held-out ones) and evaluates the profile log-likelihood at the true parameters by five methods. The "exact" column includes building the matrix and factoring it. Low-rank is the modified predictive process with $12 \times 12 = 144$ knots. Taper is the Wendland taper with $\theta = 0.2$. Vecchia is the maximin ordering with $k = 10$. The "setup" column is the one-time cost of the maximin ordering and neighbor search, which is not repeated for each evaluation. Each number in parentheses is the shortfall of the approximate log-likelihood against the exact one. The seconds are machine dependent, and the ratios between methods are the finding.

```python
print("seconds per likelihood evaluation, and loglik error against exact")
print("    n   exact  low-rank (err)   taper (err)   Vecchia (err)  setup")
bench = []
for n_b in (1000, 2000, 3000, 5000):
    Sb, yb, S0b, _ = simulate_rents(n_b, 200, 2110)
    Xb, X0b = design(Sb), design(S0b)
    exb, t_ex = timed(exact_fit, Sb, yb, Xb, S0b, X0b)
    lrb, t_lr = timed(lowrank_fit, Sb, yb, Xb, S0b, X0b, 12)
    tpb, t_tp = timed(taper_fit, Sb, yb, Xb, S0b, X0b, 0.2)
    t0 = time.perf_counter()
    ob = maximin_order(Sb)
    idxb = previous_neighbors(Sb[ob], 10)
    t_setup = time.perf_counter() - t0
    (llb, _), t_vc = timed(vecchia_loglik, yb[ob], Xb[ob], Sb[ob], idxb,
                           sigma2, tau2, phi)
    bench.append((n_b, t_ex, t_lr, t_tp, t_vc))
    print(f"{n_b:5d} {t_ex:7.2f} {t_lr:8.2f} ({lrb['ll'] - exb['ll']:6.1f}) "
          f"{t_tp:7.2f} ({tpb['ll'] - exb['ll']:5.1f}) "
          f"{t_vc:7.2f} ({llb - exb['ll']:5.1f}) {t_setup:6.2f}")
```

At $n = 5{,}000$, one exact evaluation takes about $1.1$ seconds. The low-rank evaluation takes $0.02$ seconds, about 55 times faster, but its log-likelihood is $396$ below the exact one, and the shortfall grows faster than the sample size ($-32$, $-80$, $-175$, $-396$ at 1,000 to 5,000 listings, or $0.03$ per listing at 1,000 and $0.08$ at 5,000); we have not investigated why. The tapered evaluation with $\theta = 0.2$ takes about as long as the exact one ($1.21$ seconds) and is $71$ below. The Vecchia evaluation takes $0.02$ seconds, also about 55 times faster, and is $15$ below the exact value at $n = 5{,}000$ (and $0.8$ at $n = 1{,}000$), which is about $0.003$ per listing. The one-time setup costs $0.45$ seconds. For this model and this task, Vecchia dominates the others: it is as cheap as the low-rank model and nearly as accurate as the exact computation.

### 13.7.2 Sizes Beyond the Exact Computation

The next block measures the cost of one likelihood evaluation at $n = 20{,}000$ and $n = 50{,}000$ for the low-rank model (144 knots), using random data because only the time is of interest, and the sparse LU factorization of a tapered matrix at $n = 20{,}000$ with the very short range $0.05$.

```python
# Cost of one likelihood evaluation for the low-rank model (144 knots) and
# for a tapered matrix, at sizes where the dense matrix is out of reach.
for n_h in (20_000, 50_000):
    rng = np.random.default_rng(2110)
    Sh = rng.uniform(size=(n_h, 2))
    yh = 18.0 + 6.0 * Sh[:, 0] + rng.standard_normal(n_h)   # timing only
    Xh = design(Sh)
    out, t_lr = timed(lowrank_fit, Sh, yh, Xh, Sh[:10], Xh[:10], 12)
    print(f"n={n_h:6d}: low-rank likelihood with q=144 takes {t_lr:.2f} s; "
          f"dense Cholesky would take about {c3 * n_h**3:.0f} s")
Sh = np.random.default_rng(2110).uniform(size=(20_000, 2))
Sig_t = taper_matrix(Sh, 0.05, phi)
lu, t_lu = timed(spla.splu, Sig_t, permc_spec="MMD_AT_PLUS_A",
                 diag_pivot_thresh=0.0)
print(f"n=20000, taper 0.05: {Sig_t.nnz / 20000**2:.4f} of the entries, "
      f"sparse LU {t_lu:.1f} s, fill {(lu.L.nnz + lu.U.nnz) / Sig_t.nnz:.1f}")
```

The low-rank evaluation takes $0.07$ seconds at 20,000 listings and $0.19$ seconds at 50,000, which is the linear growth in $n$ that the operation count predicts ($O(nq^2)$ with $q$ fixed). The dense Cholesky factorization would take about $71$ seconds and $1{,}113$ seconds at these sizes, by extrapolation. The tapered matrix with $\theta = 0.05$ has $0.76\%$ of its entries nonzero, and its sparse LU takes about $29$ seconds with a fill-in of $8.2$: sparse, but slower than the whole Vecchia fit of Section 13.4.4, and with the accuracy problems of Section 13.3.3. The Vecchia fit at $n = 20{,}000$ in Section 13.4.4 used 212 evaluations in 16 seconds. The neighbor search in this teaching implementation is $O(n^2)$ and would be the bottleneck at $n = 50{,}000$.

### 13.7.3 Which Method for Which Problem

| Situation | Method | Why |
| :--- | :--- | :--- |
| Smooth, large-scale spatial effect; moderate range | Low-rank (modified predictive process) | Cheap and accurate when the field is smooth (Matérn 5/2 above); poor for rough fields |
| Short correlation range and a very large region; need exact zeros | Tapering with a good sparse Cholesky routine | Accurate only when the taper range is several times the correlation range |
| General geostatistical likelihood, any smoothness, up to hundreds of thousands of points with a tree-based neighbor search (not shown) | Vecchia, maximin ordering, $k$ from 10 to 30 | Linear cost, close to exact |
| Regular grid or lattice, or a latent field observed with noise | Sparse precision (GMRF), Woodbury with sparse factorizations | Exact for the Markov model; the Matérn field is a close relative |
| Complete space-time grid | Kronecker | Exact; assumes separability |

### 13.7.4 Figures

The figure code uses the results stored by the earlier blocks. It saves the four figures of this chapter to `figures/ch13/`; run it from the top folder of the repository.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch13", exist_ok=True)

# Figure 13.1: the cost of the exact computation.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].loglog(ns, chol_times, "o-", label="measured")
axes[0].loglog(ns, c3 * np.array(ns)**3, "--", label="slope 3 (cubic)")
axes[0].set_xlabel("number of listings n")
axes[0].set_ylabel("Cholesky time (seconds)")
axes[0].legend()
grid_n = np.array([1e3, 3e3, 1e4, 3e4, 1e5])
axes[1].loglog(grid_n, 8 * grid_n**2 / 1e9, "o-")
axes[1].set_xlabel("number of listings n")
axes[1].set_ylabel("dense covariance matrix (GB)")
fig.tight_layout()
fig.savefig("figures/ch13/fig_13_1_scaling.png", dpi=120)

# Figure 13.2: what a low-rank model loses.
lr8 = lowrank_fit(S, y, X, S0, X0, 8)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sc = axes[0].scatter(S[:, 0], S[:, 1], c=lr8["deficit"], s=8, cmap="viridis")
K8 = knots_grid(8)
axes[0].plot(K8[:, 0], K8[:, 1], "r+", ms=8)
axes[0].set_title("variance deficit, 64 knots (red crosses)")
fig.colorbar(sc, ax=axes[0])
for name, rows in lowrank_rows.items():
    qs = [r[0] for r in rows]
    axes[1].loglog(qs, [-r[1] for r in rows], "o--",
                   label=f"{name}, plain")
    axes[1].loglog(qs, [-r[2] for r in rows], "o-",
                   label=f"{name}, modified")
axes[1].set_xlabel("number of knots q")
axes[1].set_ylabel("loglik shortfall (exact minus approximate)")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/ch13/fig_13_2_lowrank.png", dpi=120)

# Figure 13.3: tapering and Vecchia against the exact likelihood.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ph in (0.15, 0.03):
    rows = [r for r in taper_rows if r[0] == 2000 and r[1] == ph]
    axes[0].semilogy([r[2] for r in rows], [-r[3] for r in rows], "o-",
                     label=f"phi = {ph}")
axes[0].set_xlabel("taper range theta")
axes[0].set_ylabel("loglik shortfall")
axes[0].legend()
for name, rows in vecchia_rows.items():
    axes[1].semilogy([r[0] for r in rows],
                     [max(abs(r[1]), 1e-3) for r in rows], "o-", label=name)
axes[1].set_xlabel("number of neighbors k")
axes[1].set_ylabel("absolute loglik error")
axes[1].legend()
fig.tight_layout()
fig.savefig("figures/ch13/fig_13_3_taper_vecchia.png", dpi=120)

# Figure 13.4: a grid GMRF against the Matern field with smoothness 1.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
lags = np.arange(0, 25)
xx = kappa * lags
mat = np.where(lags == 0, 1.0, xx * special.kv(1, np.maximum(xx, 1e-12)))
axes[0].plot(lags, mat, "k-", label="Matern, nu = 1")
axes[0].plot(lags, cov[center, center:center + 25] / v0, "o",
             label="GMRF, center row")
axes[0].plot(lags, cov[mid_edge, mid_edge:mid_edge + 25]
             / cov[mid_edge, mid_edge], "s", mfc="none", label="GMRF, edge")
axes[0].set_xlabel("distance in cells")
axes[0].set_ylabel("correlation")
axes[0].legend()
im = axes[1].imshow(np.diag(cov).reshape(m, m) / v0, origin="lower")
axes[1].set_title("marginal variance relative to the center")
fig.colorbar(im, ax=axes[1])
fig.tight_layout()
fig.savefig("figures/ch13/fig_13_4_gmrf.png", dpi=120)
print("figures saved")
```

---

## Exercises

**Analytical**

1. Show that the Sherman-Morrison formula for a rank-one update, $(\mathbf{A} + \mathbf{u}\mathbf{v}')^{-1} = \mathbf{A}^{-1} - \dfrac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}'\mathbf{A}^{-1}}{1 + \mathbf{v}'\mathbf{A}^{-1}\mathbf{u}}$, is the case $q = 1$ of the Woodbury identity of Section 13.2.2. Use the determinant lemma to show that $\lvert\mathbf{A} + \mathbf{u}\mathbf{v}'\rvert = \lvert\mathbf{A}\rvert(1 + \mathbf{v}'\mathbf{A}^{-1}\mathbf{u})$.

2. For the model $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'$ with $\mathbf{B} = \mathbf{1}_n$ (one knot, $q = 1$) and $\boldsymbol{\Sigma}_\alpha = \sigma^2$, write $\boldsymbol{\Sigma}^{-1}$ and $\log\lvert\boldsymbol{\Sigma}\rvert$ in closed form. This is the equal-correlation model of Exercise 2 in Chapter 1. Check that your formula reproduces $\operatorname{Var}(\bar{Y}) = (\sigma^2 + \tau^2/n)$.

3. Prove that the variance deficit $\sigma^2(1 - \mathbf{r}(\mathbf{s})'\mathbf{R}_K^{-1}\mathbf{r}(\mathbf{s}))$ of Section 13.2.5 is non-negative and is zero at the knots. (Hint: it is a simple kriging variance, and Chapter 4 showed what that is.) Show that the modified predictive process has the exact variance $\sigma^2 + \tau^2$ at every listing.

4. Fixed-rank kriging with a bump basis. Let $\mathbf{B}$ contain Gaussian bumps $B_{ij} = \exp(-\lVert\mathbf{s}_i - \boldsymbol{\kappa}_j\rVert^2/(2w^2))$ centered at the knots, and let $\boldsymbol{\Sigma}_\alpha = \sigma_\alpha^2\mathbf{I}_q$. Write the covariance function this model implies and say how its smoothness differs from the exponential model's. Use the code of Section 13.2.4, with `LowRank` and a bump basis, to compare the likelihood at the same parameters with the exact exponential likelihood.

5. Show that the tapered covariance matrix of Section 13.3.1 is positive definite for any set of distinct locations. (Hint: write the entrywise product of two positive semi-definite matrices as a Gram matrix, or use the process construction of Section 2.4.3.) Why does the proof fail if the taper function is only required to be non-negative?

6. In a region of area $A$ with $n$ listings placed uniformly, find the expected number of nonzero entries per row of the tapered matrix, ignoring the boundary, as a function of $\theta$. For what taper range is the matrix 1% dense when $n = 20{,}000$ and $A = 1$? When $A = 100$ with the same density of listings per unit area?

7. The conjugate gradient method solves $\boldsymbol{\Sigma}\mathbf{x} = \mathbf{b}$ by repeated multiplication by $\boldsymbol{\Sigma}$. State the cost per iteration for a tapered matrix with $m$ nonzero entries and explain why the number of iterations grows with the condition number of $\boldsymbol{\Sigma}$. Compare the condition number of $\boldsymbol{\Sigma}$ with a nugget $\tau^2 = 1$ and with $\tau^2 = 0.01$ for the unit-square rent model.

8. Prove that Vecchia's approximation with $\mathcal{C}_i = \{i - 1\}$ is exact for four points on a line with exponential correlation, using $r(h_1 + h_2) = r(h_1)r(h_2)$, by showing that $\operatorname{Cov}(y_i, y_j \mid y_{i-1}) = 0$ for $j < i - 1$. Then explain why the same argument fails in two dimensions.

9. Show that the matrix $\widetilde{\mathbf{Q}} = (\mathbf{I} - \boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I} - \boldsymbol{\Gamma})$ of Section 13.4.1 has entry $(j, l)$ equal to zero unless $j = l$, one of $j, l$ is in the parent set of the other, or $j$ and $l$ share a child. Use this to bound the number of nonzeros in row $j$ by $(k + 1)(c + 1)$ when every point has at most $k$ parents and at most $c$ children, and describe what "children" means here. Then show that the whole matrix has at most $n(k + 1)^2$ nonzeros, and explain why this does not bound an individual row by $(k + 1)^2$. Build $\widetilde{\mathbf{Q}}$ for 1,000 random locations in a maximin ordering with $k = 3$ and compare the largest row and the total with these bounds.

10. Derive the variance $1/(4\pi\kappa^2)$ of the Matérn field with $\nu = 1$ from the spectral density: the covariance of the solution of $(\kappa^2 - \nabla^2)u = \mathcal{W}$ is the inverse Fourier transform of $1/(\kappa^2 + \lVert\boldsymbol{\omega}\rVert^2)^2$ times the white-noise intensity. (Evaluate the integral $\int(2\pi)^{-2}(\kappa^2 + \lVert\boldsymbol{\omega}\rVert^2)^{-2}\,d\boldsymbol{\omega}$ in polar coordinates.)

11. For the precision $\mathbf{Q} = (\kappa^2\mathbf{I} + \mathcal{L})^2$ on a grid, show that the row of $\mathbf{Q}$ for an interior cell has 13 nonzero entries and compute them as functions of $\kappa^2$. Check them against the SAR precision $(\kappa^2 + 4)^2(\mathbf{I} - \rho\mathbf{W})^2$ with $\rho = 4/(\kappa^2 + 4)$.

12. Prove that the Kronecker product of two Cholesky factors $\mathbf{L}_T \otimes \mathbf{L}_S$ is lower triangular. Then show that if $\boldsymbol{\Sigma}_S$ is $n_S \times n_S$ and $\boldsymbol{\Sigma}_T$ is $n_T \times n_T$, simulating $\operatorname{vec}(\mathbf{Y})$ as $(\mathbf{L}_T \otimes \mathbf{L}_S)\operatorname{vec}(\mathbf{Z})$ takes $O(n_Sn_T(n_S + n_T))$ operations when computed as $\mathbf{L}_S\mathbf{Z}\mathbf{L}_T'$, and $O\big((n_Sn_T)^2\big)$ when the Kronecker product is formed.

**Computational and simulation**

13. Repeat the low-rank comparison of Section 13.2.4 for the exponential model with a nugget of $\tau^2 = 4$ instead of $1$. Does a larger nugget make the low-rank approximation more or less accurate, and why? Then fit $(\sigma^2, \tau^2, \phi)$ by maximizing the low-rank likelihood (modified model, 144 knots) on the $n = 2{,}000$ data set and compare with the exact maximum likelihood estimates. Does the low-rank fit overstate or understate the nugget?

14. Taper experiment. For the $n = 5{,}000$ data set of Section 13.3.2, find the smallest taper range $\theta$ for which the tapered log-likelihood is within 10 of the exact one at $\phi = 0.15$, and report its density, fill-in, and time. Then repeat at $\phi = 0.03$. Fit the covariance parameters by maximizing the tapered likelihood and compare with the exact estimates.

15. Vecchia experiment. For the $n = 2{,}000$ data set, find by trial the smallest $k$ for each of the three orderings at which the Vecchia log-likelihood is within 0.5 of the exact one. Then repeat with the neighbor sets chosen as the $k$ nearest earlier points in the maximin ordering but with half of them replaced by random earlier points, and explain what changes.

16. Predictions with Vecchia's model. Extend the code of Section 13.4.3 to predict at held-out listings: for a new location $\mathbf{s}_0$, condition on the $k$ nearest data listings, using the weights $\mathbf{b}_0 = \boldsymbol{\Sigma}_{\mathcal{C}}^{-1}\boldsymbol{\Sigma}_{\mathcal{C},0}$. Compare the prediction error with that of exact kriging and of the low-rank model on the $n = 2{,}000$ data set.

17. The grid and the Matérn function. Repeat the check of Section 13.5.2 with $\kappa = 0.5$ and with $\kappa = 0.1$ on a grid of $50 \times 50$ cells. For which $\kappa$ is the agreement with the Matérn correlation best? What goes wrong when the range $1/\kappa$ is comparable to the width of the grid? Estimate the factor by which the corner variance exceeds the central variance in each case.

18. Replace the graph Laplacian in Section 13.5 by $\kappa^2\mathbf{I} + \mathcal{L}$ alone ($\alpha = 1$) on the same grid and compute its marginal variance at the center for grids of $25 \times 25$, $50 \times 50$, and $100 \times 100$ cells. Does it converge as the grid grows? Compare with the $\alpha = 2$ precision.

19. Space-time. Generate a data set with $n_S = 100$ sites and $n_T = 30$ weeks from the nonseparable covariance of Section 13.6.3. Fit a separable model by maximum likelihood with the Kronecker computations and compare its predictions at a held-out week with those of the true model. How much does separability cost?

20. For a data set with 5% of the cells of a $400 \times 50$ space-time rectangle missing at random, explain why the Kronecker computation of Section 13.6.2 no longer applies, and implement the likelihood with the Woodbury identity: write the observed covariance matrix as a submatrix of $\boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ and use a conjugate gradient solver with Kronecker matrix-vector products.

---

## What Comes Next

This chapter has completed the toolbox. A spatial model is a covariance matrix, a precision matrix, or a pair of them, and every method in the book is a way of computing with one of those matrices: Cholesky factors for simulation and likelihoods (Chapters 2 and 5), the Schur complement for kriging (Chapter 4), eigenvalues for lattice models (Chapters 6 to 8), iteratively reweighted least squares for counts and point patterns (Chapters 9 and 11), and in this chapter the four structures that let the same computations run at sizes where the exact versions cannot.

The appendices collect the supporting material. Appendix A gathers the matrix algebra used throughout, including the block inversion and determinant identities of Section 13.2.2 and the Kronecker identities of Section 13.6.2. Appendix B collects the Python patterns used in the labs. Appendix C translates the jargon of the field into the language of linear models; the correspondence between this book's notation and that of Schabenberger and Gotway is in the notation section at the front of the book. Appendix D points beyond the book: cokriging and multivariate fields, nonseparable space-time models, non-Gaussian geostatistics, Bayesian computation (MCMC and INLA, including the finite-element mesh form of the SPDE bridge of Section 13.5), alternatives to plug-in prediction, spatial confounding and causal inference, the Spatial Durbin model, the deeper theory of covariance validity, and spatial machine learning.

If there is one idea to take from the book, it is the one Chapter 0 began with: nearby observations share information, so a spatial data set is worth less than its size suggests, and a spatial model is the machinery for saying how much less. Everything after Chapter 1 was an answer to the follow-up question, which is what to do about it.
