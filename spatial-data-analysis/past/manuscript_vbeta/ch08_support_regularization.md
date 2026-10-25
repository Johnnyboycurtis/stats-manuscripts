# Chapter 8: Change of Support and Spatial Regularization

In physical, environmental, and earth science applications, spatial measurements are virtually never collected at dimensionless geometric points. Instead, physical observations and operational decisions occur over finite, non-zero geometric domains: diamond drill cores ($5\text{ cm}$ diameter cylinders), soil cores ($10\text{ cm}$ depth increments), satellite pixels ($10\text{ m} \times 10\text{ m}$ to $1\text{ km} \times 1\text{ km}$ footprints), air-quality monitoring sensor cones, or selective mining excavation units ($10\text{ m} \times 10\text{ m} \times 5\text{ m}$ blocks). The geometric volume, shape, and spatial orientation over which a physical attribute is measured or averaged is termed its **support**.

When transitioning from point-support observations $Z(\mathbf{s})$ to a volumetric average $Z_v(\mathbf{s})$, spatial variability collapses, spatial continuity regularizes, extreme statistical tails attenuate, and the marginal probability distribution undergoes a non-linear transformation. In this chapter, we develop the linear-model and second-order stochastic theory of spatial regularization, establish the spatial Analysis of Variance (ANOVA) decomposition culminating in Krige's Additivity Theorem, prove Cartier's conditional expectation relation, analyze analytical change-of-support models, and derive the Ordinary Block Kriging system with explicit computational implementations.

### Learning Outcomes
1. **Model Volume Regularization:** Formulate the spatial averaging of second-order stationary random fields across compact support volumes, demonstrating how expectation is preserved while spatial variance is dampened.
2. **Compute Regularized Covariances and Semivariograms:** Apply spatial double integration and numerical grid quadrature to evaluate within-block and between-block covariance and variogram functions.
3. **Establish Spatial ANOVA and Krige's Additivity:** Derive dispersion variance from first principles and prove Krige's Additivity Theorem using the finite-sample ANOVA identity.
4. **Evaluate Support Effects on Cut-Off Economics:** Quantify how volume support changes recovered tonnage, metal quantity, and mean recovered grade using Cartier's relation and Jensen's inequality.
5. **Apply Analytical Change-of-Support Transformations:** Implement the Affine Correction and Indirect Lognormal Correction models to forecast block-support distributions from point data.
6. **Implement Ordinary Block Kriging:** Derive the block kriging system, establish its relationship with integrated point kriging, and diagnose the information effect and conditional bias in spatial predictions.

---

### Chapter at a Glance: Change of Support and Regularization

| Concept / Tool | Mathematical Definition | Physical Meaning | Linear Models Connection |
| :--- | :--- | :--- | :--- |
| **Regularized Field** $Z_v(\mathbf{s})$ | $\frac{1}{\vert v\vert}\int_{v(\mathbf{s})} Z(\mathbf{u})\,d\mathbf{u}$ | Volume average over support $v$ | Continuous analog of sample mean $\bar{Z} = \frac{1}{n}\mathbf{1}'\mathbf{Z}$ |
| **Within-Block Covariance** $\overline{C}(v, v)$ | $\frac{1}{\vert v\vert^2}\int_v\int_v C(\mathbf{u}-\mathbf{w})\,d\mathbf{u}d\mathbf{w}$ | Expected covariance between two uniform random points in $v$ | Continuous analog of $\frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1}$ |
| **Within-Block Semivariance** $\overline{\gamma}(v, v)$ | $\frac{1}{\vert v\vert^2}\int_v\int_v \gamma(\mathbf{u}-\mathbf{w})\,d\mathbf{u}d\mathbf{w}$ | Internal point-pair variability; variance reduction $C(\mathbf{0}) - \overline{C}(v,v)$ | Sample variance of points inside block: $D^2(\mathbf{0} \mid v)$ |
| **Dispersion Variance** $D^2(v \mid V)$ | $\overline{C}(v, v) - \overline{C}(V, V) = \overline{\gamma}(V, V) - \overline{\gamma}(v, v)$ | Spatial variability of small units $v$ within larger panels $V$ | Within-group variance in one-way ANOVA |
| **Krige's Additivity** | $D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)$ | Total deposit variability = Within-panel + Between-panel | $\text{SS}_{\text{Total}} = \text{SS}_{\text{Within}} + \text{SS}_{\text{Between}}$ |
| **Cartier's Relation** | $\operatorname{E}[Z_v \mid Z_V] = Z_V$ for random $v \subset V$ | Local conservation of mass across support scales | Orthogonal projection in linear conditional expectation |
| **Block Kriging System** | $\begin{pmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda}_v \\ \mu_{\text{OK}} \end{pmatrix} = \begin{pmatrix} \overline{\boldsymbol{\gamma}}_v \\ 1 \end{pmatrix}$ | BLUE predictor of volume average from point data | Standard OK system with RHS averaged over target volume |

---

### 8.1 The Volume-Support Problem and Spatial Regularization

#### Classical-to-Spatial Bridge: From Chapter 1's Sample Mean to Continuous Volume Averaging

In [Chapter 1](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch01_foundations.md), we established the foundational variance formula for the arithmetic mean of $n$ correlated spatial observations $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ with covariance matrix $\mathbf{V} = \operatorname{Var}(\mathbf{Z})$:
$$
\operatorname{Var}(\bar{Z}) = \operatorname{Var}\left(\frac{1}{n}\mathbf{1}'\mathbf{Z}\right) = \frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1} = \frac{\sigma^2}{n^2} \sum_{i=1}^n \sum_{j=1}^n \rho(\mathbf{s}_i - \mathbf{s}_j)
$$
When observations are independent ($\rho(\mathbf{s}_i - \mathbf{s}_j) = 0$ for $i \ne j$), this reduces to the classical elementary variance $\sigma^2/n$. In the presence of positive spatial autocorrelation, the off-diagonal entries of $\mathbf{V}$ are strictly positive, causing the variance of the average to shrink at a rate slower than $1/n$. We captured this behavior through the **effective sample size**:
$$
n_{\text{eff}} = \frac{n}{1 + (n - 1)\bar{\rho}} \le n
$$
where $\bar{\rho}$ is the average correlation among the sample pairs.

Now consider physical measurements collected over a continuous geometric volume $v \subset \mathbb{R}^d$ (such as an extraction block, a diamond drill core, or an atmospheric grid cell). The volume-averaged variable $Z_v$ is mathematically nothing other than the sample mean $\bar{Z}$ evaluated over a spatial continuum of points! 
Replacing the discrete summation $\frac{1}{n}\sum_{i=1}^n (\cdot)$ with continuous spatial integration $\frac{1}{\vert v\vert}\int_v (\cdot)\,d\mathbf{u}$, the variance of the volume average is:
$$
\operatorname{Var}(Z_v) = \frac{1}{\vert v\vert^2} \int_v \int_v C(\mathbf{u} - \mathbf{w})\,d\mathbf{u}\,d\mathbf{w}
$$
This spatial double integral is the exact continuous counterpart to $\frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1}$.

For blocks whose dimensions are large relative to the correlation range of the process, the double integral can be evaluated in terms of the **integral range** $A$:
$$
A = \int_{\mathbb{R}^d} \rho(\mathbf{h})\,d\mathbf{h}
$$
The integral range $A$ has units of volume ($[\text{Length}]^d$) and represents the continuous analog of the correlation sum $1 + (n-1)\bar{\rho}$. For a large block volume $\vert v\vert \gg A$, the variance collapses as:
$$
\operatorname{Var}(Z_v) \approx \sigma^2 \frac{A}{\vert v\vert} = \frac{\sigma^2}{N_{\text{eff}}}
$$
where $N_{\text{eff}} = \vert v\vert / A$ is the effective number of independent spatial cells contained within the volume $v$. 

> [!NOTE]
> **From the Lecture Hall: A Block Average is Chapter 1's Sample Mean over a Continuum**  
> Students often wonder whether "change of support" requires an entirely new branch of probability theory. It does not. Every result in spatial regularization is the continuous counterpart of the variance of a sample mean:
> - In Chapter 1, summing $n$ correlated discrete random variables gives variance $\frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1}$.
> - In Chapter 8, integrating over a volume $v$ gives variance $\frac{1}{\vert v\vert^2}\int_v\int_v C(\mathbf{u}-\mathbf{w})\,d\mathbf{u}d\mathbf{w}$.
> - In Chapter 1, independent points shrink variance at rate $1/n$.
> - In Chapter 8, large blocks shrink variance at rate $A/\vert v\vert$, where $A$ is the integral range.
> 
> The core intuition is identical: aggregating spatial values cancels out opposing local fluctuations, dampening the variance while leaving the expected value unchanged.

##### Definition 8.1.1 (Spatial Support and Regularized Random Field)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$ be a second-order stationary spatial random field with mean $\operatorname{E}[Z(\mathbf{s})] = \mu$ and continuous covariance function $C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h}))$.
1. A **spatial support** $v \subset \mathbb{R}^d$ is a compact, bounded subset of Euclidean space with non-zero Lebesgue measure $\vert v\vert = \int_v 1\,d\mathbf{u} > 0$.
2. For any reference location $\mathbf{s} \in D$, the translate of support $v$ centered at $\mathbf{s}$ is:
   $$
   v(\mathbf{s}) = \mathbf{s} + v = \{\mathbf{s} + \mathbf{u} : \mathbf{u} \in v\}
   $$
3. The **regularized spatial random field** $Z_v(\mathbf{s})$ is the volume average of $Z$ across the support $v(\mathbf{s})$:
   $$
   Z_v(\mathbf{s}) = \frac{1}{\vert v\vert} \int_{v(\mathbf{s})} Z(\mathbf{u})\,d\mathbf{u} = \frac{1}{\vert v\vert} \int_v Z(\mathbf{s} + \mathbf{u})\,d\mathbf{u}
   $$

##### Theorem 8.1.1 (Linearity and Second-Order Properties of $Z_v$)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ have continuous covariance $C(\mathbf{h})$ on compact domain $D$, and let $v \subset \mathbb{R}^d$ be a compact support:
1. **Preservation of the Mean:** The expectation operator commutes with spatial integration, preserving the mean:
   $$
   \operatorname{E}[Z_v(\mathbf{s})] = \frac{1}{\vert v\vert} \int_v \operatorname{E}[Z(\mathbf{s} + \mathbf{u})]\,d\mathbf{u} = \frac{1}{\vert v\vert} \int_v \mu\,d\mathbf{u} = \mu
   $$
2. **Second-Order Covariance Function:** The regularized field $\{Z_v(\mathbf{s})\}$ is second-order stationary with well-defined covariance function:
   $$
   C_v(\mathbf{h}) = \operatorname{Cov}(Z_v(\mathbf{s}), Z_v(\mathbf{s}+\mathbf{h})) = \frac{1}{\vert v\vert^2} \int_v \int_v C(\mathbf{h} + \mathbf{u}' - \mathbf{u})\,d\mathbf{u}\,d\mathbf{u}'
   $$
3. **Variance Dampening:** The point variance of the regularized field satisfies:
   $$
   \operatorname{Var}(Z_v(\mathbf{s})) = C_v(\mathbf{0}) = \frac{1}{\vert v\vert^2}\int_v\int_v C(\mathbf{u}' - \mathbf{u})\,d\mathbf{u}\,d\mathbf{u}' \le C(\mathbf{0}) = \operatorname{Var}(Z(\mathbf{s}))
   $$
   with strict inequality whenever $C(\mathbf{h}) < C(\mathbf{0})$ on a set of positive measure within the support differences $v - v$.

*Proof.*
1. Because the point mean function $\mu(\mathbf{s}) = \mu$ is continuous on the compact set $v$, the integral of the expectation is well-defined. Linearity of expectation allows exchanging expectation and integration:
   $$
   \operatorname{E}\left[\frac{1}{\vert v\vert}\int_v Z(\mathbf{s}+\mathbf{u})\,d\mathbf{u}\right] = \frac{1}{\vert v\vert}\int_v \operatorname{E}[Z(\mathbf{s}+\mathbf{u})]\,d\mathbf{u} = \frac{1}{\vert v\vert}\int_v \mu\,d\mathbf{u} = \mu
   $$
2. By definition of covariance:
   $$
   \begin{aligned}
   C_v(\mathbf{h}) &= \operatorname{E}[(Z_v(\mathbf{s}) - \mu)(Z_v(\mathbf{s}+\mathbf{h}) - \mu)] \\
   &= \operatorname{E}\left[\left(\frac{1}{\vert v\vert}\int_v (Z(\mathbf{s}+\mathbf{u}) - \mu)\,d\mathbf{u}\right) \left(\frac{1}{\vert v\vert}\int_v (Z(\mathbf{s}+\mathbf{h}+\mathbf{u}') - \mu)\,d\mathbf{u}'\right)\right] \\
   &= \frac{1}{\vert v\vert^2} \int_v \int_v \operatorname{E}[(Z(\mathbf{s}+\mathbf{u}) - \mu)(Z(\mathbf{s}+\mathbf{h}+\mathbf{u}') - \mu)]\,d\mathbf{u}\,d\mathbf{u}' \\
   &= \frac{1}{\vert v\vert^2} \int_v \int_v C(\mathbf{s}+\mathbf{h}+\mathbf{u}' - (\mathbf{s}+\mathbf{u}))\,d\mathbf{u}\,d\mathbf{u}' = \frac{1}{\vert v\vert^2} \int_v \int_v C(\mathbf{h} + \mathbf{u}' - \mathbf{u})\,d\mathbf{u}\,d\mathbf{u}'
   \end{aligned}
   $$
   Notice that $C_v(\mathbf{h})$ depends only on the spatial lag $\mathbf{h}$ and the fixed support geometry $v$, proving second-order stationarity of $Z_v(\mathbf{s})$.
3. At lag $\mathbf{h} = \mathbf{0}$, $C_v(\mathbf{0}) = \frac{1}{\vert v\vert^2}\int_v\int_v C(\mathbf{u}' - \mathbf{u})\,d\mathbf{u}\,d\mathbf{u}'$. Since positive semi-definiteness ensures $C(\mathbf{u}' - \mathbf{u}) \le C(\mathbf{0})$ for all pairs $\mathbf{u}, \mathbf{u}'$,
   $$
   C_v(\mathbf{0}) \le \frac{1}{\vert v\vert^2} \int_v \int_v C(\mathbf{0})\,d\mathbf{u}\,d\mathbf{u}' = C(\mathbf{0}) \frac{\vert v\vert^2}{\vert v\vert^2} = C(\mathbf{0})
   $$
   This establishes that spatial averaging never increases variance. $\blacksquare$

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why does spatial regularization eliminate variance while preserving the spatial mean?*  
> The expectation operator $\operatorname{E}[\cdot]$ and spatial integration $\frac{1}{\vert v\vert}\int_v (\cdot)\,d\mathbf{u}$ are both linear operators. They commute freely, meaning that integrating over physical space cannot create, destroy, or translate the expected mass of the spatial attribute: $\operatorname{E}[Z_v] = \mu$.
> 
> Variance, however, is a quadratic functional of deviations: $\operatorname{Var}(Z_v) = \operatorname{E}[(Z_v - \mu)^2]$. When we average over a positive physical volume $\vert v\vert > 0$, positive stochastic fluctuations from rich zones and negative stochastic fluctuations from lean zones inside the same block cancel each other out. The larger the block volume, the greater the opportunity for internal spatial cancellation, collapsing the variance of block averages toward zero.

#### Algorithmic Translation: Block Discretization via Spatial Quadrature

Analytical evaluation of the double integral $\frac{1}{\vert v\vert^2}\int_v\int_v C(\mathbf{u}'-\mathbf{u})\,d\mathbf{u}\,d\mathbf{u}'$ is possible only for simple geometric domains (e.g., 1D segments or rectangles) under specific covariance models. In operational applications, the block support $v$ is discretized into a regular lattice of $M$ internal quadrature nodes $\{\mathbf{u}_k\}_{k=1}^M \subset v$, each representing an elemental volume $\Delta v = \vert v\vert / M$:

> **Geometric Discretization:**  
> The block $v$ is discretized into a regular lattice of $M$ internal quadrature nodes $\mathbf{u}_k \in v$, each representing an elemental volume $\Delta v = |v|/M$:
> $$
> \begin{pmatrix}
> \mathbf{u}_1 & \mathbf{u}_2 & \mathbf{u}_3 & \mathbf{u}_4 \\
> \mathbf{u}_5 & \mathbf{u}_6 & \mathbf{u}_7 & \mathbf{u}_8 \\
> \mathbf{u}_9 & \mathbf{u}_{10} & \mathbf{u}_{11} & \mathbf{u}_{12} \\
> \mathbf{u}_{13} & \mathbf{u}_{14} & \mathbf{u}_{15} & \mathbf{u}_{16}
> \end{pmatrix} \subset v
> $$

The double integral is approximated by the arithmetic mean over all $M^2$ pairwise distances between internal lattice nodes:
$$
\overline{C}(v, v) \approx \frac{1}{M^2} \sum_{j=1}^M \sum_{k=1}^M C(\mathbf{u}_j - \mathbf{u}_k)
$$
Similarly, the regularized covariance between two translated blocks $v$ and $v_{\mathbf{h}} = v + \mathbf{h}$ is approximated by:
$$
\overline{C}(v, v_{\mathbf{h}}) \approx \frac{1}{M^2} \sum_{j=1}^M \sum_{k=1}^M C(\mathbf{u}_j - (\mathbf{u}_k + \mathbf{h}))
$$

```python
import numpy as np
import itertools
from typing import Callable, Sequence

def compute_regularized_covariance_grid(
    block_dims: Sequence[float],
    n_disc: Sequence[int],
    cov_func: Callable[[np.ndarray], np.ndarray],
    h_lag: Sequence[float] | None = None
) -> tuple[float, float]:
    """
    Computes within-block covariance C_bar(v, v) and between-block
    lagged covariance C_bar(v, v_h) via regular grid quadrature.
    
    Parameters:
        block_dims: Side lengths [dx, dy, ...] of compact block v
        n_disc: Number of discretization steps [mx, my, ...] per axis
        cov_func: Vectorized callable accepting Euclidean distance array
        h_lag: Spatial translation lag vector h (defaults to zero vector)
        
    Returns:
        (c_bar_vv, c_bar_vvh): within-block and lagged block covariances
    """
    dim = len(block_dims)
    if h_lag is None:
        h_lag = [0.0] * dim
        
    # Step 1: Generate regular grid coordinates centered inside block v
    coords_per_axis = []
    for d in range(dim):
        step = block_dims[d] / n_disc[d]
        start = -block_dims[d] / 2.0 + step / 2.0
        coords_per_axis.append([start + k * step for k in range(n_disc[d])])
        
    # Cartesian product to form M discretization points: shape (M, dim)
    u_nodes = np.array(list(itertools.product(*coords_per_axis)))
    M = len(u_nodes)
    
    # Step 2: Compute all pairwise difference vectors for C_bar(v, v)
    diffs_vv = u_nodes[:, None, :] - u_nodes[None, :, :]  # Shape: (M, M, dim)
    dists_vv = np.linalg.norm(diffs_vv, axis=-1)
    c_bar_vv = float(np.mean(cov_func(dists_vv)))
    
    # Step 3: Compute pairwise differences between v and translated block v_h
    u_nodes_h = u_nodes + np.array(h_lag)
    diffs_vvh = u_nodes[:, None, :] - u_nodes_h[None, :, :]
    dists_vvh = np.linalg.norm(diffs_vvh, axis=-1)
    c_bar_vvh = float(np.mean(cov_func(dists_vvh)))
    
    return c_bar_vv, c_bar_vvh
```

#### Quadrature Error vs. Grid Density

A common misconception among practitioners is that a coarse grid (such as $4 \times 4 \times 4 = 64$ points) is universally accurate to within $1\%$. In truth, quadrature error depends directly on the ratio of block size to the correlation range. Because the diagonal node pairs ($j = k$) evaluate at zero distance, contributing $C(\mathbf{0})$ to the sum, regular grid quadrature **systematically overestimates** within-block covariance $\overline{C}(v, v)$.

The following empirical benchmark measures the relative percentage overestimation of $\overline{C}(v, v)$ for an isotropic exponential model $C(h) = \exp(-h/a)$ in a cubic block, compared to a converged 4-million-point reference quadrature:

| Cube Side ($L$), Range ($a$) | $2^3 = 8$ Nodes | $4^3 = 64$ Nodes | $8^3 = 512$ Nodes |
| :--- | :--- | :--- | :--- |
| **Side $L = 1$, Range $a = 1.0$** | $+10.9\%$ | $+2.0\%$ | $+0.44\%$ |
| **Side $L = 1$, Range $a = 0.3$** | $+55.0\%$ | $+7.0\%$ | $+1.30\%$ |
| **Side $L = 3$, Range $a = 1.0$** | $+46.0\%$ | $+6.1\%$ | $+1.20\%$ |

Notice that when the block is large relative to the correlation range ($L/a \ge 3$), a $4^3$ grid still overestimates within-block covariance by $6\%$ to $7\%$. The operational rule of thumb is: **refine the discretization grid until calculated average covariances stabilize within engineering tolerance**.

> [!WARNING]
> **Discretization and the Nugget Effect:**  
> When a point model includes a nugget effect $\tau^2$, the quadrature formula evaluates diagonal pairs at $h = 0$, where $C(0) = \sigma^2 + \tau^2$, while all off-diagonal pairs evaluate at $h > 0$, where the nugget contributes zero. As a result, the discretized sum assigns a nugget contribution of exactly $\tau^2 / M$ to $\overline{C}(v, v)$. This artificial grid-dependent contribution vanishes as $M \to \infty$, reflecting the fact that pure point-scale white noise averages out over a spatial continuum.

---

### 8.2 Regularized Covariances and Semivariograms

How does volumetric integration alter the spatial continuity and semivariogram structure? We express regularized covariances and semivariograms in terms of spatial cross-averaging operators.

##### Definition 8.2.1 (Average Covariance and Semivariogram Operators)
Let $A, B \subset \mathbb{R}^d$ be two compact domains with non-zero measures $\vert A\vert > 0, \vert B\vert > 0$.
1. The **average covariance** between domains $A$ and $B$ is:
   $$
   \overline{C}(A, B) = \frac{1}{\vert A\vert \vert B\vert} \int_A \int_B C(\mathbf{u} - \mathbf{w})\,d\mathbf{u}\,d\mathbf{w}
   $$
2. The **average semivariance** between domains $A$ and $B$ is:
   $$
   \overline{\gamma}(A, B) = \frac{1}{\vert A\vert \vert B\vert} \int_A \int_B \gamma(\mathbf{u} - \mathbf{w})\,d\mathbf{u}\,d\mathbf{w}
   $$
3. When $A = B = v$, $\overline{C}(v, v)$ and $\overline{\gamma}(v, v)$ denote the **within-block average covariance** and **within-block average semivariance**.
4. When $A = \{\mathbf{s}_i\}$ is a point and $B = v$, $\overline{C}(\mathbf{s}_i, v) = \frac{1}{\vert v\vert}\int_v C(\mathbf{s}_i - \mathbf{u})\,d\mathbf{u}$ denotes the **point-to-block average covariance**.

##### Term-by-Term Equation Anatomy: Within-Block Semivariance Integral
$$
\overline{\gamma}(v, v) = \underbrace{\frac{1}{\vert v\vert^2}}_{\substack{\text{Normalization by} \\ \text{Squared Volume}}} \underbrace{\int_v \int_v}_{\substack{\text{Double Spatial} \\ \text{Integral}}} \underbrace{\gamma(\mathbf{u} - \mathbf{w})}_{\substack{\text{Point Semivariance} \\ \text{between Internal Nodes}}} \, d\mathbf{u} \, d\mathbf{w}
$$
- **$\vert v\vert^2$:** Normalizes the double volume integral so that $\overline{\gamma}(v, v)$ has the units of variance ($[Z]^2$).
- **$\mathbf{u}, \mathbf{w} \in v$:** Two coordinates independently traversing the compact geometry $v$. The integral evaluates the semivariance across all possible internal spatial vectors $\mathbf{h} = \mathbf{u} - \mathbf{w}$.
- **Physical Interpretation:** $\overline{\gamma}(v, v)$ represents the expected semivariance between two points chosen uniformly at random within the block $v$. As the volume $\vert v\vert$ expands, the expected distance between internal points increases, causing $\overline{\gamma}(v, v)$ to rise monotonically toward the sill $C(\mathbf{0})$.

##### Theorem 8.2.1 (Structure of the Regularized Semivariogram)
Let $\{Z(\mathbf{s})\}$ be second-order stationary with continuous covariance $C(\mathbf{h})$ and semivariogram $\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$. Let $v \subset \mathbb{R}^d$ be a compact support. Then:
1. The within-block average covariance and semivariance satisfy:
   $$
   \overline{C}(v, v) = C(\mathbf{0}) - \overline{\gamma}(v, v)
   $$
2. The regularized semivariogram $\gamma_v(\mathbf{h})$ between blocks separated by lag $\mathbf{h}$ is:
   $$
   \gamma_v(\mathbf{h}) = \overline{\gamma}(v, v_{\mathbf{h}}) - \overline{\gamma}(v, v)
   $$
3. The regularized sill (the variance of block averages) is:
   $$
   \sigma_v^2 = C_v(\mathbf{0}) = C(\mathbf{0}) - \overline{\gamma}(v, v) \le C(\mathbf{0})
   $$

*Proof.*
1. Substitute the relation $C(\mathbf{u}-\mathbf{w}) = C(\mathbf{0}) - \gamma(\mathbf{u}-\mathbf{w})$ into Definition 8.2.1:
   $$
   \overline{C}(v, v) = \frac{1}{\vert v\vert^2}\int_v\int_v [C(\mathbf{0}) - \gamma(\mathbf{u}-\mathbf{w})]\,d\mathbf{u}d\mathbf{w} = C(\mathbf{0}) - \overline{\gamma}(v, v)
   $$
2. By definition of the semivariogram of $Z_v$:
   $$
   \gamma_v(\mathbf{h}) = \frac{1}{2} \operatorname{E}[(Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s}))^2] = C_v(\mathbf{0}) - C_v(\mathbf{h})
   $$
   Express the regularized covariance in semivariance form:
   $$
   \begin{aligned}
   C_v(\mathbf{h}) &= \frac{1}{\vert v\vert^2}\int_v \int_v C(\mathbf{h} + \mathbf{u}' - \mathbf{u})\,d\mathbf{u}d\mathbf{u}' \\
   &= \frac{1}{\vert v\vert^2}\int_v \int_v [C(\mathbf{0}) - \gamma(\mathbf{h} + \mathbf{u}' - \mathbf{u})]\,d\mathbf{u}d\mathbf{u}' = C(\mathbf{0}) - \overline{\gamma}(v, v_{\mathbf{h}})
   \end{aligned}
   $$
   Subtracting $C_v(\mathbf{h})$ from $C_v(\mathbf{0}) = C(\mathbf{0}) - \overline{\gamma}(v, v)$:
   $$
   \gamma_v(\mathbf{h}) = [C(\mathbf{0}) - \overline{\gamma}(v, v)] - [C(\mathbf{0}) - \overline{\gamma}(v, v_{\mathbf{h}})] = \overline{\gamma}(v, v_{\mathbf{h}}) - \overline{\gamma}(v, v)
   $$
   Alternatively, consider the algebraic polarization identity: for any four random variables $X_1, X_2, X_3, X_4$,
   $$
   (X_1 - X_2)(X_3 - X_4) = \frac{1}{2}[(X_1 - X_4)^2 + (X_2 - X_3)^2 - (X_1 - X_3)^2 - (X_2 - X_4)^2]
   $$
   Setting $X_1 = Z(\mathbf{u}+\mathbf{h})$, $X_2 = Z(\mathbf{u})$, $X_3 = Z(\mathbf{w}+\mathbf{h})$, $X_4 = Z(\mathbf{w})$, taking expectations, and integrating over $\mathbf{u}, \mathbf{w} \in v$ yields identically $\gamma_v(\mathbf{h}) = \overline{\gamma}(v, v_{\mathbf{h}}) - \overline{\gamma}(v, v)$. $\blacksquare$

##### Proposition 8.2.1 (Spatial Filtering, Nugget Effects, and Smoothness at the Origin)
In geostatistical applications, the point nugget effect $\tau^2$ consists of two physically distinct phenomena:
1. **Micro-scale Spatial Variation ($\tau_{\text{micro}}^2$):** Real spatial structure whose correlation length $a_0$ is smaller than the minimum sample spacing. Over a support volume $v$ with $\vert v\vert \gg a_0^d$, this component is spatially averaged out, contributing an attenuated variance $\tau_{\text{micro}}^2 A / \vert v\vert$ to the block variance (where $A \approx a_0^d$ is its micro-scale integral range).
2. **Measurement / Assay Error ($\tau_{\text{meas}}^2$):** Laboratory assay error, sensor noise, or sample preparation variability. When a physical bulk sample of volume $v$ is assayed, this measurement error is reintroduced by the laboratory assay protocol and is **not** eliminated by the geometric support volume.

Furthermore, regarding the smoothness of the regularized field:
- **Continuous Point Covariances:** If the underlying continuous point field has continuous covariance without micro-scale white noise, the regularized semivariogram $\gamma_v(\mathbf{h})$ has **zero nugget effect** and leaves the origin **parabolically**:
  $$
  \gamma_v(\mathbf{h}) = \mathcal{O}(\Vert\mathbf{h}\Vert^2) \quad \text{as } \Vert\mathbf{h}\Vert \to 0
  $$
  This implies that the regularized random field $Z_v(\mathbf{s})$ is mean-square differentiable in all spatial directions, even if the point field $Z(\mathbf{s})$ was rough and non-differentiable (such as an exponential covariance model).
- **Micro-scale Spatial Components:** If an unresolved micro-scale component is present, its volume-regularized covariance is proportional to the geometric covariogram (the intersection volume $\vert v \cap (v + \mathbf{h})\vert$). Because the intersection volume of a solid block decreases linearly with displacement $\Vert\mathbf{h}\Vert$, its regularized semivariogram leaves the origin with a linear slope:
  $$
  \gamma_{v, \text{micro}}(\mathbf{h}) = \mathcal{O}(\Vert\mathbf{h}\Vert) \quad \text{as } \Vert\mathbf{h}\Vert \to 0
  $$

#### Practical Properties of Regularized Semivariograms

| Geometric Property | Point Support $\gamma(\mathbf{h})$ | Block Support $\gamma_v(\mathbf{h})$ | Mathematical Mechanism |
| :--- | :--- | :--- | :--- |
| **Nugget / Origin** | Jump discontinuity $\tau^2 > 0$ | Parabolic $\mathcal{O}(\|\mathbf{h}\|^2)$ departure | Microscale noise filtered by volume averaging |
| **Sill** | Total variance $C(\mathbf{0}) = \sigma^2 + \tau^2$ | Dampened variance $\sigma_v^2 = C(\mathbf{0}) - \overline{\gamma}(v, v)$ | Within-block variability $\overline{\gamma}(v, v)$ absorbed internally |
| **Effective Range** | Base spatial range $a$ | Extended range $a_v \approx a + \text{diam}(v)$ | Blocks must separate by point range plus block width |

From a modeling perspective, changing support produces three universal geometric modifications to the semivariogram:
1. **Lower Sill:** The regularized sill is reduced by exactly the within-block semivariance: $\sigma_v^2 = C(\mathbf{0}) - \overline{\gamma}(v, v)$.
2. **Longer Range:** The effective correlation range of the regularized field increases by approximately the block dimension along that direction ($a_v \approx a + L$). Two blocks must be separated by their own width plus the point correlation range before they become uncorrelated.
3. **Smooth Departure at the Origin:** Discontinuities and high-frequency fluctuations are smoothed away by volume integration.

To illustrate sill reduction concretely, consider a 1D segment of length $L$ under an isotropic exponential covariance $C(h) = \sigma^2 \exp(-\vert h\vert / a)$. As derived in [Exercise 8.1](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch08_support_regularization.md#exercise-81-regularized-covariance-of-a-1d-segment-under-an-exponential-covariance), the normalized block variance $\overline{C}(v, v) / \sigma^2$ contracts monotonically as block length $L$ expands relative to correlation range $a$:

| Segment Ratio $L / a$ | Within-Block Semivariance $\overline{\gamma}(v, v) / \sigma^2$ | Regularized Sill $\sigma_v^2 / \sigma^2 = \overline{C}(v, v) / \sigma^2$ |
| :--- | :--- | :--- |
| **$0.1$ (Point-like)** | $0.048$ | $0.952$ |
| **$0.5$** | $0.213$ | $0.787$ |
| **$1.0$ (Block side = Range)** | $0.368$ | $0.632$ |
| **$2.0$** | $0.570$ | $0.430$ |
| **$5.0$** | $0.760$ | $0.240$ |
| **$10.0$ (Large Block)** | $0.865$ | $0.135$ |

---

### 8.3 Dispersion Variance and Krige's Additivity Theorem

How do we quantify the variability of small extraction units within large geological panels? In spatial statistics, the variance of small support units within a larger enclosing domain is formalized through the **dispersion variance**.

##### Definition 8.3.1 (Dispersion Variance)
Let $v \subset V \subset \mathbb{R}^d$ be two nested compact supports ($0 < \vert v\vert \le \vert V\vert$).
1. The **dispersion variance of points within volume $V$**, denoted $D^2(\mathbf{0} \mid V)$, is the spatial variance of point values $Z(\mathbf{s})$ around the volume average $Z_V$:
   $$
   D^2(\mathbf{0} \mid V) = \frac{1}{\vert V\vert} \int_V \operatorname{Var}(Z(\mathbf{u}) - Z_V)\,d\mathbf{u} = \overline{\gamma}(V, V) = C(\mathbf{0}) - \overline{C}(V, V)
   $$
2. The **dispersion variance of units $v$ within volume $V$**, denoted $D^2(v \mid V)$, is the spatial variance of block averages $Z_v$ as $v$ traverses all positions inside $V$:
   $$
   D^2(v \mid V) = \overline{\gamma}(V, V) - \overline{\gamma}(v, v) = \overline{C}(v, v) - \overline{C}(V, V)
   $$

*Derivation of the Formula for $D^2(v \mid V)$.*  
Consider the variance of the difference between a small sub-block average $Z_v$ and the encompassing panel average $Z_V$:
$$
\operatorname{Var}(Z_v - Z_V) = \operatorname{Var}(Z_v) - 2\operatorname{Cov}(Z_v, Z_V) + \operatorname{Var}(Z_V) = \overline{C}(v, v) - 2\overline{C}(v, V) + \overline{C}(V, V)
$$
When we average this variance over all positions of $v$ that tile $V$, the cross-covariance $\overline{C}(v, V)$ evaluates to $\overline{C}(V, V)$, because the union of all translates of $v$ reconstructs $V$. Substituting $\overline{C}(v, V) = \overline{C}(V, V)$ yields:
$$
D^2(v \mid V) = \overline{C}(v, v) - 2\overline{C}(V, V) + \overline{C}(V, V) = \overline{C}(v, v) - \overline{C}(V, V)
$$
Using $\overline{C}(A, A) = C(\mathbf{0}) - \overline{\gamma}(A, A)$, this can be written equivalently in semivariogram notation as:
$$
D^2(v \mid V) = [C(\mathbf{0}) - \overline{\gamma}(v, v)] - [C(\mathbf{0}) - \overline{\gamma}(V, V)] = \overline{\gamma}(V, V) - \overline{\gamma}(v, v)
$$

##### Theorem 8.3.1 (Krige's Additivity Theorem)
Let $v \subset V \subset W \subset \mathbb{R}^d$ be three nested spatial supports:
$$
v \subset V \subset W \quad \text{with } 0 < \vert v\vert \le \vert V\vert \le \vert W\vert
$$
Then the total dispersion variance of small blocks $v$ within the global domain $W$ decomposes additively into:
$$
D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)
$$

*Proof via the Finite Spatial ANOVA Identity.*  
We establish the proof using the classical Analysis of Variance (ANOVA) identity for a fixed spatial realization, followed by taking expectations across the random field.

Partition the global domain $W$ into $N$ non-overlapping intermediate panels $\{V_k\}_{k=1}^N$ congruent to $V$, such that $\vert W\vert = N\vert V\vert$. Further partition each panel $V_k$ into $M$ non-overlapping sub-blocks $\{v_{kj}\}_{j=1}^M$ congruent to $v$, such that $\vert V\vert = M\vert v\vert$.
For any realization of the spatial field, let:
- $z_{kj}$ denote the realization value of sub-block $v_{kj}$;
- $\bar{z}_k = \frac{1}{M}\sum_{j=1}^M z_{kj}$ denote the average value of panel $V_k$;
- $\bar{z} = \frac{1}{NM}\sum_{k=1}^N \sum_{j=1}^M z_{kj} = \frac{1}{N}\sum_{k=1}^N \bar{z}_k$ denote the grand deposit average over $W$.

In one-way ANOVA, the Total Sum of Squares decomposes algebraically into Within-Group and Between-Group sums of squares:
$$
\sum_{k=1}^N \sum_{j=1}^M (z_{kj} - \bar{z})^2 = \sum_{k=1}^N \sum_{j=1}^M (z_{kj} - \bar{z}_k)^2 + \sum_{k=1}^N M(\bar{z}_k - \bar{z})^2
$$
Dividing this identity by the total number of sub-blocks $NM$:
$$
\frac{1}{NM}\sum_{k=1}^N \sum_{j=1}^M (z_{kj} - \bar{z})^2 = \frac{1}{N}\sum_{k=1}^N \left(\frac{1}{M}\sum_{j=1}^M (z_{kj} - \bar{z}_k)^2\right) + \frac{1}{N}\sum_{k=1}^N (\bar{z}_k - \bar{z})^2
$$
Now take the mathematical expectation over the distribution of the random field:
1. The left-hand side is the expected spatial variance of sub-blocks $v$ across the domain $W$, which is by definition $D^2(v \mid W)$.
2. In the first term on the right, $\frac{1}{M}\sum_{j=1}^M (z_{kj} - \bar{z}_k)^2$ is the internal spatial variance of sub-blocks inside panel $V_k$. Averaged across all panels $k=1,\dots, N$, its expectation is by definition $D^2(v \mid V)$.
3. The second term on the right is the spatial variance among panel averages $\bar{z}_k$ across the global domain $W$, whose expectation is by definition $D^2(V \mid W)$.

Therefore:
$$
D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)
$$
Alternatively, substituting Definition 8.3.1 into the right-hand side confirms the telescoping cancellation:
$$
[\overline{\gamma}(V, V) - \overline{\gamma}(v, v)] + [\overline{\gamma}(W, W) - \overline{\gamma}(V, V)] = \overline{\gamma}(W, W) - \overline{\gamma}(v, v) = D^2(v \mid W)
$$
This completes the proof. $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: Spatial ANOVA and the Law of Total Variance in Concrete Volumes**  
> Krige's Additivity Theorem is the spatial, geometric realization of the classical Law of Total Variance:
> $$
> \operatorname{Var}(Y) = \operatorname{E}[\operatorname{Var}(Y \mid X)] + \operatorname{Var}(\operatorname{E}[Y \mid X])
> $$
> When we select a small unit $v$ at random from a deposit $W$:
> - Its unconditional variability across the entire deposit is $D^2(v \mid W)$.
> - Given that unit $v$ lies inside panel $V_k$, its local variability is the conditional variance $\operatorname{Var}(Z_v \mid V_k) = D^2(v \mid V)$.
> - The variability among the panel conditional expectations $\operatorname{E}[Z_v \mid V_k] = Z_{V_k}$ is the between-panel variance $D^2(V \mid W)$.
> 
> Krige's additivity proves that total geological variability cannot be destroyed; it can only be partitioned between internal block homogenization ($D^2(v \mid V)$) and panel-to-panel variation ($D^2(V \mid W)$).

#### Concrete Numerical Benchmark: Dispersion Variance in a Stratified Deposit

To see Krige's additivity in action, consider a 1D continuous borehole of length $W = 12\text{ m}$ partitioned into $N = 3$ panels of length $V = 4\text{ m}$, each consisting of $M = 4$ drill core samples of length $v = 1\text{ m}$. Under a spherical semivariogram with sill $c = 10.0$ and range $a = 6.0\text{ m}$:
- Internal sample semivariance: $\overline{\gamma}(v, v) = 0.8310$;
- Internal panel semivariance: $\overline{\gamma}(V, V) = 3.1852$;
- Internal deposit semivariance: $\overline{\gamma}(W, W) = 6.7500$.

Evaluating the dispersion variances yields:
$$
\begin{aligned}
D^2(v \mid V) &= \overline{\gamma}(V, V) - \overline{\gamma}(v, v) = 3.1852 - 0.8310 = 2.3542 \\
D^2(V \mid W) &= \overline{\gamma}(W, W) - \overline{\gamma}(V, V) = 6.7500 - 3.1852 = 3.5648 \\
D^2(v \mid W) &= \overline{\gamma}(W, W) - \overline{\gamma}(v, v) = 6.7500 - 0.8310 = 5.9190
\end{aligned}
$$
Summing the within-panel and between-panel components:
$$
D^2(v \mid V) + D^2(V \mid W) = 2.3542 + 3.5648 = 5.9190 = D^2(v \mid W)
$$
The identity holds with mathematical precision.

---

### 8.4 Cartier's Relation and Variance Reduction Across Supports

When aggregating spatial measurements into larger blocks, what happens to our expectation of the fine-scale process? Cartier's relation formalizes the property that volume averaging acts as an orthogonal conditional expectation operator.

##### Theorem 8.4.1 (Cartier's Relation)
Let $V \subset \mathbb{R}^d$ be a compact domain, and let $Z_V = \frac{1}{\vert V\vert}\int_V Z(\mathbf{u})\,d\mathbf{u}$ denote the block average.
1. **Randomly Chosen Point:** Let $S$ be a coordinate distributed uniformly in $V$ with spatial density $p_V(\mathbf{s}) = \frac{1}{\vert V\vert}$, independently of the field $Z$. Then:
   $$
   \operatorname{E}[Z(S) \mid Z_V] = Z_V
   $$
2. **Randomly Positioned Sub-Block:** Let $v \subset V$ be a smaller support ($0 < \vert v\vert < \vert V\vert$). If $v$ is placed uniformly at random within $V$, the conditional expectation of the sub-block average satisfies:
   $$
   \operatorname{E}[Z_v \mid Z_V] = Z_V
   $$
   Equivalently, if $V$ is partitioned into $m$ disjoint sub-blocks $v_1, \dots, v_m$ of equal volume, the average of their conditional expectations satisfies:
   $$
   \frac{1}{m}\sum_{j=1}^m \operatorname{E}[Z_{v_j} \mid Z_V] = Z_V
   $$

*Proof.*
1. Conditioning on the realization of the random field and taking the expectation over the uniform spatial distribution of $S$:
   $$
   \operatorname{E}_S[Z(S) \mid \text{field}] = \int_V Z(\mathbf{s}) \frac{1}{\vert V\vert}\,d\mathbf{s} = Z_V
   $$
   Taking expectations conditional on $Z_V$ yields $\operatorname{E}[Z(S) \mid Z_V] = Z_V$ by the tower property of conditional expectation.
2. For sub-block $v$, let $\mathbf{u}$ denote a random translation vector distributed uniformly such that $v(\mathbf{u}) \subset V$. Because spatial integration is linear, exchanging the expectation over position and the spatial volume integral yields $\operatorname{E}[Z_v \mid Z_V] = Z_V$. $\blacksquare$

> [!CAUTION]
> **Cartier's Relation Does NOT Hold for a Fixed Spatial Sub-Block!**  
> It is tempting to assume that for *any fixed* sub-block $v \subset V$, $\operatorname{E}[Z_v \mid Z_V] = Z_V$. This is mathematically false!  
> For a stationary Gaussian field, the conditional expectation of a fixed sub-block $Z_v$ given the larger block $Z_V$ is the classical linear regression:
> $$
> \operatorname{E}[Z_v \mid Z_V] = \mu + \frac{\operatorname{Cov}(Z_v, Z_V)}{\operatorname{Var}(Z_V)}(Z_V - \mu)
> $$
> The regression slope $\beta = \operatorname{Cov}(Z_v, Z_V) / \operatorname{Var}(Z_V)$ is not equal to $1$ for an individual fixed sub-block:
> - For an edge sub-block, covariance with the broader block is dampened by boundary effects, yielding a slope $\beta < 1$ (e.g., $0.948$ for $[0, 0.25]$ in $[0, 1]$ under an exponential model).
> - For a central sub-block, covariance with the broader block is stronger, yielding a slope $\beta > 1$ (e.g., $1.065$ for $[0.375, 0.625]$).
> 
> The regression slopes average to exactly $1.0$ across all positions in $V$. Cartier's relation holds on average across the domain, not at every fixed internal coordinate.

##### Corollary 8.4.1 (Variance Monotonicity across Supports)
For nested supports $v \subset V \subset W$:
$$
\operatorname{Var}(Z_W) \le \operatorname{Var}(Z_V) \le \operatorname{Var}(Z_v) \le \operatorname{Var}(Z(\mathbf{s}))
$$
with equality if and only if the field is spatially constant (correlation identically 1.0).

*Proof.*
By Krige's Additivity Theorem (Theorem 8.3.1) applied with point support $\mathbf{0}$:
$$
D^2(\mathbf{0} \mid V) = \operatorname{Var}(Z(\mathbf{s})) - \operatorname{Var}(Z_V) \ge 0 \implies \operatorname{Var}(Z_V) \le \operatorname{Var}(Z(\mathbf{s}))
$$
Similarly, for any two supports $v \subset V$, $D^2(v \mid V) = \operatorname{Var}(Z_v) - \operatorname{Var}(Z_V) \ge 0$, because dispersion variance is the expectation of a squared deviation and is therefore non-negative. Hence $\operatorname{Var}(Z_V) \le \operatorname{Var}(Z_v)$. $\blacksquare$

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *What is the practical purpose of Cartier's relation in resource modeling?*  
> Cartier's relation establishes **local mass conservation**: if an ore block has an average grade of $2.5\text{ g/t}$, the expected grade of drill cores drilled inside that block must equal $2.5\text{ g/t}$. Spatial averaging neither creates nor destroys physical mass.  
> 
> Cartier's relation is an intrinsic property of true block values, not a mathematical constraint that an engineer "enforces." Rather, analytical change-of-support models (such as the Affine Correction and the Indirect Lognormal Correction) are explicitly *designed* to honor Cartier's relation so that forecasted block distributions preserve the true regional mean.

---

### 8.5 Distributional Transformations under Change of Support

When spatial support coarsens from point measurements $Z(\mathbf{s})$ to volumetric blocks $Z_V(\mathbf{s})$, the marginal probability distribution undergoes three fundamental transformations:
1. **Variance Reduction:** The dispersion variance contracts monotonically from $\sigma_0^2$ to $\sigma_V^2 = \overline{C}(V, V) < \sigma_0^2$.
2. **Central Limit Effect toward Normality:** Volume averaging combines multiple weakly correlated spatial sub-regions, causing the block distribution to approach a Gaussian shape.
3. **Attenuation of Skewness:** Asymmetry in the point histogram (e.g., highly skewed mineral or pollutant grades) is dampened.

#### The Spatial Central Limit Effect and the Integral Range

In classical statistics, the Central Limit Theorem dictates that the arithmetic mean of $n$ independent observations has skewness shrinking as $n^{-1/2}$ and approaches normality. In spatial continuous fields, the same phenomenon governs volume regularization through the **integral range** $A = \int_{\mathbb{R}^d} \rho(\mathbf{h})\,d\mathbf{h}$.

When a block $V$ is large relative to the integral range ($|V| \gg A$), the block can be viewed heuristically as averaging $N_{\text{eff}} = |V| / A$ approximately independent spatial sub-domains. Under standard spatial mixing conditions (meaning that spatial autocorrelation decays sufficiently rapidly with distance, such that $\int \vert C(\mathbf{h})\vert\,d\mathbf{h} < \infty$):
1. **Asymptotic Normality:** The regularized random variable $Z_V$ is approximately normally distributed:
   $$
   Z_V \;\dot\sim\; \mathcal{N}\left(\mu, \sigma_V^2\right) \quad \text{with } \sigma_V^2 \approx \sigma^2 \frac{A}{|V|}
   $$
2. **Skewness Shrinkage:** The coefficient of skewness $\gamma_1(V) = \operatorname{E}[(Z_V - \mu)^3] / \sigma_V^3$ attenuates in proportion to the inverse square root of the effective sample size:
   $$
   \gamma_1(V) = \mathcal{O}\left(\sqrt{\frac{A}{|V|}}\right) \to 0 \quad \text{as } |V| \to \infty
   $$

The following table demonstrates this transformation for a lognormal point process ($m = 2.72\text{ g/t}$, point variance $\sigma_0^2 = 12.70$, point skewness $\gamma_1 = 6.18$) across increasing block supports:

| Support Scale | Effective Cells $N_{\text{eff}} = |V|/A$ | Block Variance $\sigma_V^2$ | Block Skewness $\gamma_1(V)$ | Distribution Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Point Support ($v_0$)** | $1$ | $12.70$ | $6.18$ | Heavily skewed lognormal |
| **Small Block ($V_1$)** | $5$ | $2.54$ | $2.76$ | Moderately skewed |
| **Bench Panel ($V_2$)** | $25$ | $0.51$ | $1.24$ | Mildly skewed |
| **Large Stope ($V_3$)** | $100$ | $0.13$ | $0.62$ | Near-Gaussian |

---

### 8.6 Selectivity Curves and Cut-off Economics

In mining, environmental remediation, and agricultural management, decisions are governed by non-linear economic thresholds. A mining block $V$ is mined as ore if its estimated grade exceeds a cut-off threshold $z_c$; otherwise it is discarded as barren waste rock. An environmental site is excavated if soil toxicity exceeds a legal remediation standard $z_c$.

##### Definition 8.6.1 (Selectivity Functions)
Let $Z_v$ denote the grade of a selective unit of support $v$, distributed according to cumulative distribution function $F_v(z) = P(Z_v \le z)$. For any economic cut-off grade $z_c \ge 0$:
1. The **recovered tonnage fraction** (proportion of units above cut-off) is:
   $$
   T_v(z_c) = P(Z_v \ge z_c) = 1 - F_v(z_c) = \int_{z_c}^\infty f_v(z)\,dz
   $$
2. The **quantity of recovered metal / attribute** (per unit mass of deposit) is:
   $$
   Q_v(z_c) = \operatorname{E}[Z_v \cdot \mathbb{I}\{Z_v \ge z_c\}] = \int_{z_c}^\infty z f_v(z)\,dz
   $$
3. The **mean recovered grade** of selected units is:
   $$
   m_v(z_c) = \frac{Q_v(z_c)}{T_v(z_c)} = \operatorname{E}[Z_v \mid Z_v \ge z_c]
   $$

Integration by parts reveals the fundamental relationship between recovered metal and tonnage:
$$
Q_v(z_c) = \int_{z_c}^\infty z \, dF_v(z) = z_c [1 - F_v(z_c)] + \int_{z_c}^\infty [1 - F_v(z)]\,dz = z_c T_v(z_c) + \int_{z_c}^\infty T_v(z)\,dz
$$
The quantity $Q_v(z_c) - z_c T_v(z_c) = \int_{z_c}^\infty T_v(z)\,dz = \operatorname{E}[(Z_v - z_c)^+]$ represents the **excess metal** above cut-off.

> [!IMPORTANT]
> **The Ergodic Identification in Selectivity:**  
> Identifying the operational fraction of an ore deposit exceeding cut-off $z_c$ with the marginal probability $P(Z_v \ge z_c)$ assumes spatial second-order stationarity and **ergodicity** (Chapter 1). The spatial histogram of block grades across the deposit is assumed to match the theoretical marginal distribution $F_v(z)$.

> [!NOTE]
> **From the Lecture Hall: Reading a Cut-off: Tonnage, Metal, and Recovered Grade**  
> Beginners frequently confuse the three selectivity curves:
> - **Tonnage $T(z_c)$:** The fraction of total rock excavated and sent to the processing plant. It is a probability: $0 \le T(z_c) \le 1$.
> - **Quantity of Metal $Q(z_c)$:** The total recovered gold, copper, or contaminant mass per ton of total rock. At $z_c = 0$, $Q(0) = m$ (the global average grade). As $z_c$ rises, $Q(z_c)$ falls monotonically toward zero.
> - **Mean Recovered Grade $m(z_c) = Q(z_c) / T(z_c)$:** The average concentration delivered to the mill. As cut-off rises, low-grade rock is discarded, so $m(z_c)$ increases monotonically: $m(z_c) \ge z_c$.

> **The Support Effect across Cut-Off Regimes:**  
> Because volume averaging preserves the expectation $m$ while contracting dispersion ($\sigma_V^2 < \sigma_v^2$), the block density $f_V(z)$ clusters tightly around $m$, intersecting the broader point density $f_v(z)$ at a crossing point $z^*$:
> - **Low Cut-Off ($z_c < m$):** Most blocks exceed $z_c$. Block recovery exceeds point recovery: $T_V(z_c) > T_v(z_c)$.
> - **Intermediate Cut-Off ($m \le z_c \le z^*$):** Block density concentrates mass in the shoulder, so block tonnage remains higher than point tonnage ($T_V > T_v$), but with heavily diluted average grade.
> - **High Cut-Off ($z_c > z^*$):** Extreme rich grades are smoothed out. Point support overstates tonnage and grade: $T_v(z_c) > T_V(z_c)$ and $m_v(z_c) > m_V(z_c)$.

##### Theorem 8.6.1 (The Support Effect on Selectivity)
Let $v \subset V$ denote small and large supports sharing identical global mean $m$, with ordered variances $\sigma_V^2 < \sigma_v^2$. For unimodal distributions that cross at a unique threshold $z^* > m$:
1. **Excess Metal Deficit:** For every cut-off $z_c$, the excess metal above cut-off satisfies:
   $$
   \operatorname{E}[(Z_V - z_c)^+] \le \operatorname{E}[(Z_v - z_c)^+]
   $$
   with strict inequality for all internal cut-offs where $0 < F_v(z_c) < 1$.
2. **High Cut-offs ($z_c > z^*$):** Volume averaging eliminates high-grade tails, reducing both tonnage and metal:
   $$
   T_V(z_c) < T_v(z_c) \quad \text{and} \quad Q_V(z_c) < Q_v(z_c)
   $$
3. **Moderate Cut-offs ($m < z_c < z^*$):** Because block grades cluster tightly around the mean $m$, block tonnage **exceeds** point tonnage:
   $$
   T_V(z_c) > T_v(z_c)
   $$
4. **Tail Grade Attenuation:** For all high cut-offs $z_c > z^*$, the mean recovered grade on block support is strictly lower than that on point support:
   $$
   m_V(z_c) < m_v(z_c)
   $$

*Proof.*
1. The function $\varphi(z) = (z - z_c)^+ = \max(0, z - z_c)$ is convex for any fixed $z_c \in \mathbb{R}$. By Cartier's relation, the large-block value is the conditional expectation of the small-block value: $Z_V = \operatorname{E}[Z_v \mid Z_V]$. Applying conditional Jensen's inequality:
   $$
   \varphi(Z_V) = \varphi(\operatorname{E}[Z_v \mid Z_V]) \le \operatorname{E}[\varphi(Z_v) \mid Z_V]
   $$
   Taking the unconditional expectation across both sides:
   $$
   \operatorname{E}[(Z_V - z_c)^+] \le \operatorname{E}[(Z_v - z_c)^+]
   $$
   Since $Q(z_c) - z_c T(z_c) = \operatorname{E}[(Z - z_c)^+]$, the excess metal delivered by blocks is strictly smaller than that evaluated on point support.
2. For $z_c > z^*$, the cumulative distribution of the contracted variable satisfies $F_V(z_c) > F_v(z_c)$, which implies immediately that $T_V(z_c) = 1 - F_V(z_c) < 1 - F_v(z_c) = T_v(z_c)$. Since $Q_V(z_c) = z_c T_V(z_c) + \int_{z_c}^\infty T_V(z)\,dz$, integrating the strictly smaller upper tail gives $Q_V(z_c) < Q_v(z_c)$.
3. For $m < z_c < z^*$, the density of $Z_V$ exceeds that of $Z_v$, causing $F_V(z_c) < F_v(z_c)$ and hence $T_V(z_c) > T_v(z_c)$.
4. In the upper tail $z_c > z^*$, the extreme point values that elevate the point tail expectation $m_v(z_c)$ are smoothed into the interior of mining blocks, diluting the tail grade and establishing $m_V(z_c) < m_v(z_c)$. $\blacksquare$

> [!WARNING]
> **The Crossing Point Myth:**  
> A widespread misconception in introductory texts states that the crossing point $z^*$ between point and block distributions always equals the arithmetic mean $m$. For symmetric Gaussian fields, $z^* = m$. But for positively skewed mineral grades (lognormal or Pareto distributions), the density crossing point $z^*$ lies **significantly above the mean** ($z^* > m$)!  
> As demonstrated in [Exercise 8.4](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch08_support_regularization.md#exercise-84-indirect-lognormal-change-of-support-for-mining-selectivity), for a lognormal deposit with $m = 2.72\text{ g/t}$, the crossing point occurs at $z^* = 3.56\text{ g/t}$. At an intermediate cut-off of $z_c = 3.00\text{ g/t}$, block tonnage is $32.54\%$ while point tonnage is only $27.47\%$. More blocks qualify, but their mean grade is drastically lower ($4.47\text{ g/t}$ vs. $6.49\text{ g/t}$).

---

### 8.7 Analytical Change-of-Support Models

Given point-support data $Z$ with mean $m$ and variance $\sigma_0^2$, how can we predict the block distribution $F_v(z)$ when only the regularized variance $\sigma_v^2 = \overline{C}(v, v)$ is known? We examine the two most widely applied analytical models.

#### 8.7.1 The Affine Correction Model

The affine correction assumes the standardized shape of the distribution is perfectly invariant under change of support, contracting linearly toward the mean.

##### Definition 8.7.1 (Affine Correction)
Let point variable $Z$ have mean $m$ and variance $\sigma_0^2$. The **affine correction** models the block variable $Z_v$ as:
$$
Z_v = r(Z - m) + m = (1 - r)m + rZ
$$
where $r \in (0, 1]$ is the **variance reduction factor**:
$$
r = \frac{\sigma_v}{\sigma_0} = \sqrt{\frac{\overline{C}(v, v)}{C(\mathbf{0})}} = \sqrt{1 - \frac{\overline{\gamma}(v, v)}{\sigma_0^2}}
$$
Quantiles transform linearly according to $z_{v, p} = r(z_p - m) + m$, and the recovered tonnage satisfies:
$$
T_v(z_c) = T_0\left(m + \frac{z_c - m}{r}\right)
$$

> [!WARNING]
> **Limitations of the Affine Model:**  
> 1. **Preservation of Skewness:** Because $Z_v$ is a linear transformation of $Z$, the third standardized moment is identical: $\gamma_1(Z_v) = \gamma_1(Z)$. This contradicts the physical reality that spatial volume averaging dampens skewness.  
> 2. **Downscaling Hazard:** For forward regularization ($r \le 1$), $Z_v$ is a convex combination of $m$ and $Z \ge 0$, which can **never** become negative. However, if the affine formula is inverted for downscaling ($r > 1$), it predicts impossible negative grades ($z = m + \frac{1}{r}(Z_v - m) < 0$) whenever $Z_v < (1 - r)m$.

#### 8.7.2 The Indirect Lognormal Correction Model

When point observations are strictly positive and highly skewed, the lognormal distribution provides a much more realistic physical approximation.

##### Theorem 8.7.2 (Indirect Lognormal Correction Equations)
Let the point variable be lognormal: $Z = \exp(Y)$ with $Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$, mean $m = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2)$, and variance $\sigma_0^2 = m^2(\exp(\sigma_Y^2) - 1)$.  
Assume the block variable $Z_v = \exp(Y_v)$ is also lognormal with $Y_v \sim \mathcal{N}(\mu_{Y_v}, \sigma_{Y_v}^2)$. Requiring that $Z_v$ preserve the true arithmetic mean $\operatorname{E}[Z_v] = m$ and match the regularized variance $\operatorname{Var}(Z_v) = \sigma_v^2 = \overline{C}(v, v)$:
1. The **block logarithmic variance** is:
   $$
   \sigma_{Y_v}^2 = \ln\left(1 + \frac{\sigma_v^2}{m^2}\right) = \ln\left(1 + \frac{\overline{C}(v, v)}{m^2}\right)
   $$
2. The **block logarithmic mean** is:
   $$
   \mu_{Y_v} = \ln(m) - \frac{1}{2}\sigma_{Y_v}^2 = \ln(m) - \frac{1}{2}\ln\left(1 + \frac{\sigma_v^2}{m^2}\right)
   $$
3. The **recovered tonnage fraction** at cut-off $z_c > 0$ is:
   $$
   T_v(z_c) = \Phi\left(\frac{\mu_{Y_v} - \ln(z_c)}{\sigma_{Y_v}}\right)
   $$
4. The **recovered quantity of metal** is:
   $$
   Q_v(z_c) = m \cdot \Phi\left(\frac{\mu_{Y_v} + \sigma_{Y_v}^2 - \ln(z_c)}{\sigma_{Y_v}}\right)
   $$
   where $\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-t^2/2}\,dt$ is the standard normal cumulative distribution function.

*Proof.*
1. For any lognormal variable $W = \exp(X)$ with $X \sim \mathcal{N}(\mu_X, \sigma_X^2)$, the variance formula is $\operatorname{Var}(W) = (\operatorname{E}[W])^2[\exp(\sigma_X^2) - 1]$. Applying this to $Z_v$:
   $$
   \operatorname{Var}(Z_v) = m^2 [\exp(\sigma_{Y_v}^2) - 1] = \sigma_v^2 \implies \exp(\sigma_{Y_v}^2) = 1 + \frac{\sigma_v^2}{m^2} \implies \sigma_{Y_v}^2 = \ln\left(1 + \frac{\sigma_v^2}{m^2}\right)
   $$
2. Since $\operatorname{E}[Z_v] = \exp(\mu_{Y_v} + \frac{1}{2}\sigma_{Y_v}^2) = m$, taking natural logarithms yields $\mu_{Y_v} = \ln(m) - \frac{1}{2}\sigma_{Y_v}^2$.
3. The recovered tonnage is:
   $$
   T_v(z_c) = P(\exp(Y_v) \ge z_c) = P(Y_v \ge \ln z_c) = 1 - \Phi\left(\frac{\ln z_c - \mu_{Y_v}}{\sigma_{Y_v}}\right) = \Phi\left(\frac{\mu_{Y_v} - \ln z_c}{\sigma_{Y_v}}\right)
   $$
4. Evaluating $Q_v(z_c) = \int_{\ln z_c}^\infty e^y \frac{1}{\sqrt{2\pi}\sigma_{Y_v}} \exp\left(-\frac{(y - \mu_{Y_v})^2}{2\sigma_{Y_v}^2}\right)\,dy$: completing the square in the exponent factors out $\exp(\mu_{Y_v} + \frac{1}{2}\sigma_{Y_v}^2) = m$, leaving $m \Phi\left(\frac{\mu_{Y_v} + \sigma_{Y_v}^2 - \ln z_c}{\sigma_{Y_v}}\right)$. $\blacksquare$

> [!NOTE]
> **The Indirect Lognormal Model as a Marginal Approximation:**  
> The sum or integral of lognormal random variables is not strictly lognormal. The Indirect Lognormal Correction is an analytical marginal approximation that matches the first two moments ($m$ and $\sigma_v^2$) while preserving the rank order of quantiles.  
> 
> Notice also the fundamental limitation of both analytical change-of-support models: **they incorporate spatial covariance exclusively through the single scalar parameter $\overline{C}(v, v)$**. Neither model accounts for local conditioning data or spatial clustering; they predict global deposit-wide recovery curves.

```python
import numpy as np
from scipy.stats import norm

def indirect_lognormal_change_of_support(
    mean_grade: float,
    point_var: float,
    block_var: float,
    cutoffs: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Computes selective mining recovery curves under the Indirect Lognormal
    Change-of-Support model.
    
    Parameters:
        mean_grade: Deposit arithmetic mean grade m
        point_var: Point-support variance sigma_0^2
        block_var: Block-support variance sigma_v^2 = C_bar(v, v)
        cutoffs: 1D array of economic cut-off thresholds z_c
        
    Returns:
        dict containing tonnage (T), metal (Q), and mean recovered grade (m)
        for both point and block support.
    """
    # Point lognormal parameters
    sigma_Y2 = np.log(1.0 + point_var / (mean_grade**2))
    sigma_Y = np.sqrt(sigma_Y2)
    mu_Y = np.log(mean_grade) - 0.5 * sigma_Y2
    
    # Block lognormal parameters
    sigma_Yv2 = np.log(1.0 + block_var / (mean_grade**2))
    sigma_Yv = np.sqrt(sigma_Yv2)
    mu_Yv = np.log(mean_grade) - 0.5 * sigma_Yv2
    
    # Point selectivity curves
    d1_0 = (mu_Y - np.log(cutoffs)) / sigma_Y
    d2_0 = d1_0 + sigma_Y
    T_0 = norm.cdf(d1_0)
    Q_0 = mean_grade * norm.cdf(d2_0)
    m_0 = np.where(T_0 > 1e-9, Q_0 / T_0, 0.0)
    
    # Block selectivity curves
    d1_v = (mu_Yv - np.log(cutoffs)) / sigma_Yv
    d2_v = d1_v + sigma_Yv
    T_v = norm.cdf(d1_v)
    Q_v = mean_grade * norm.cdf(d2_v)
    m_v = np.where(T_v > 1e-9, Q_v / T_v, 0.0)
    
    return {
        'cutoffs': cutoffs,
        'T_point': T_0, 'Q_point': Q_0, 'm_point': m_0,
        'T_block': T_v, 'Q_block': Q_v, 'm_block': m_v
    }
```

---

### 8.8 Ordinary Block Kriging

How do we predict the volumetric average $Z_v(\mathbf{s}_0) = \frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} Z(\mathbf{u})\,d\mathbf{u}$ from $n$ discrete point measurements $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$? Ordinary Block Kriging provides the Best Linear Unbiased Predictor (BLUP) on block support.

##### Theorem 8.8.1 (Ordinary Block Kriging System)
Let $\{Z(\mathbf{s})\}$ be an intrinsically stationary random field with semivariogram $\gamma(\mathbf{h})$. Given observations $\mathbf{Z}$, consider the linear block predictor:
$$
\widehat{Z}_v(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = \boldsymbol{\lambda}'\mathbf{Z}
$$
Under the unbiasedness constraint $\sum_{i=1}^n \lambda_i = 1$:
1. The optimal kriging weight vector $\boldsymbol{\lambda} \in \mathbb{R}^n$ and Lagrange multiplier $\mu_{\text{OK}}$ satisfy:
   $$
   \begin{pmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix}
   \begin{pmatrix} \boldsymbol{\lambda} \\ \mu_{\text{OK}} \end{pmatrix}
   =
   \begin{pmatrix} \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) \\ 1 \end{pmatrix}
   $$
   where $\mathbf{\Gamma}_{ij} = \gamma(\mathbf{s}_i - \mathbf{s}_j)$, $\mathbf{1} = (1, \dots, 1)'$, and $\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)$ is the vector of point-to-block average semivariances:
   $$
   [\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)]_i = \overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0)) = \frac{1}{\vert v\vert} \int_{v(\mathbf{s}_0)} \gamma(\mathbf{s}_i - \mathbf{u})\,d\mathbf{u}, \quad i = 1, \dots, n
   $$
2. The Ordinary Block Kriging prediction error variance is:
   $$
   \sigma_{\text{OK}, v}^2(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i \overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0)) + \mu_{\text{OK}} - \overline{\gamma}(v, v) = \boldsymbol{\lambda}' \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) + \mu_{\text{OK}} - \overline{\gamma}(v, v)
   $$
3. In covariance notation under second-order stationarity:
   $$
   \begin{pmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix}
   \begin{pmatrix} \boldsymbol{\lambda} \\ -\mu_C \end{pmatrix}
   =
   \begin{pmatrix} \overline{\mathbf{c}}_v(\mathbf{s}_0) \\ 1 \end{pmatrix}
   $$
   $$
   \sigma_{\text{OK}, v}^2(\mathbf{s}_0) = \overline{C}(v, v) - \boldsymbol{\lambda}' \overline{\mathbf{c}}_v(\mathbf{s}_0) + \mu_C
   $$
   where $\mathbf{V}_{ij} = C(\mathbf{s}_i - \mathbf{s}_j)$ and $[\overline{\mathbf{c}}_v(\mathbf{s}_0)]_i = \overline{C}(\mathbf{s}_i, v(\mathbf{s}_0))$.

*Proof.*
The estimation error is $e = \widehat{Z}_v(\mathbf{s}_0) - Z_v(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) - \frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} Z(\mathbf{u})\,d\mathbf{u}$.
Expressing the expected squared error in semivariogram form:
$$
\begin{aligned}
\sigma_E^2 &= \operatorname{E}[(e)^2] = -\frac{1}{2}\operatorname{E}\left[\left(\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) - \frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} Z(\mathbf{u})\,d\mathbf{u}\right)^2\right] \\
&= -\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + 2\sum_{i=1}^n \lambda_i \left(\frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} \gamma(\mathbf{s}_i - \mathbf{u})\,d\mathbf{u}\right) - \frac{1}{\vert v\vert^2}\int_{v(\mathbf{s}_0)}\int_{v(\mathbf{s}_0)} \gamma(\mathbf{u}-\mathbf{w})\,d\mathbf{u}d\mathbf{w} \\
&= -\boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda} + 2\boldsymbol{\lambda}'\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) - \overline{\gamma}(v, v)
\end{aligned}
$$
Formulate the Lagrangian with unbiasedness multiplier $2\mu_{\text{OK}}$:
$$
\mathcal{L}(\boldsymbol{\lambda}, \mu_{\text{OK}}) = -\boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda} + 2\boldsymbol{\lambda}'\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) - \overline{\gamma}(v, v) + 2\mu_{\text{OK}}(\boldsymbol{\lambda}'\mathbf{1} - 1)
$$
Differentiating with respect to $\lambda_i$ and setting to zero:
$$
\frac{\partial \mathcal{L}}{\partial \lambda_i} = -2\sum_{j=1}^n \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + 2\overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0)) + 2\mu_{\text{OK}} = 0
$$
$$
\sum_{j=1}^n \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + \mu_{\text{OK}} = \overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0))
$$
Multiplying by $\lambda_i$, summing over $i$, and using $\sum \lambda_i = 1$:
$$
\boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda} + \mu_{\text{OK}} = \boldsymbol{\lambda}'\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) \implies \boldsymbol{\lambda}'\mathbf{\Gamma}\boldsymbol{\lambda} = \boldsymbol{\lambda}'\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) - \mu_{\text{OK}}
$$
Substituting this back into the error variance expression yields:
$$
\sigma_{\text{OK}, v}^2(\mathbf{s}_0) = -[\boldsymbol{\lambda}'\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) - \mu_{\text{OK}}] + 2\boldsymbol{\lambda}'\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) - \overline{\gamma}(v, v) = \boldsymbol{\lambda}'\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) + \mu_{\text{OK}} - \overline{\gamma}(v, v)
$$
This completes the derivation. $\blacksquare$

##### Term-by-Term Equation Anatomy: Ordinary Block Kriging System
$$
\underbrace{\begin{pmatrix} \mathbf{\Gamma} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{pmatrix}}_{\substack{\text{Data-to-Data Semivariogram Matrix } \\ (n+1) \times (n+1) \text{ (Independent of Block Size!)}}}
\underbrace{\begin{pmatrix} \boldsymbol{\lambda} \\ \mu_{\text{OK}} \end{pmatrix}}_{\substack{\text{Block Weights} \\ \text{\& Multiplier}}}
=
\underbrace{\begin{pmatrix} \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) \\ 1 \end{pmatrix}}_{\substack{\text{Point-to-Block} \\ \text{Semivariance Vector}}}
$$
and the minimized prediction variance:
$$
\sigma_{\text{OK}, v}^2(\mathbf{s}_0) = \underbrace{\boldsymbol{\lambda}' \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)}_{\substack{\text{Weighted Point-to-Block} \\ \text{Semivariance}}} + \underbrace{\mu_{\text{OK}}}_{\substack{\text{Unbiasedness} \\ \text{Penalty}}} - \underbrace{\overline{\gamma}(v, v)}_{\substack{\text{Within-Block Dispersion} \\ \text{(Variance Reduction of Support)}}}
$$
- **Data Matrix $\mathbf{\Gamma}$:** Depends purely on sample coordinates $\mathbf{s}_1, \dots, \mathbf{s}_n$. It is inverted or factored once and reused for all target blocks!
- **Right-Hand Side $\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)$:** Averages the point semivariances over the target volume $v(\mathbf{s}_0)$.
- **Subtrahend $-\overline{\gamma}(v, v)$:** Explicitly removes internal block dispersion variance. Point kriging predicts an erratic single point; block kriging predicts an aggregated volume whose internal fluctuations cancel out.

##### Corollary 8.8.1 (Equivalence with Integrated Point Kriging and Corrected Variance Identity)
1. **Weights Equivalence:** The Ordinary Block Kriging weight vector $\boldsymbol{\lambda}_v$ is identically equal to the volume average of the point kriging weight vectors $\boldsymbol{\lambda}(\mathbf{u})$ evaluated across the block:
   $$
   \boldsymbol{\lambda}_v = \frac{1}{\vert v\vert} \int_{v(\mathbf{s}_0)} \boldsymbol{\lambda}(\mathbf{u})\,d\mathbf{u}
   $$
2. **Corrected Variance Identity:** The average of point kriging estimation variances across the block satisfies:
   $$
   \frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} \sigma_{\text{OK}}^2(\mathbf{u})\,d\mathbf{u} = \sigma_{\text{OK}, v}^2(\mathbf{s}_0) + \overline{\gamma}(v, v) - \frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} \operatorname{Var}(\widehat{Z}(\mathbf{u}) - \widehat{Z}_v)\,d\mathbf{u}
   $$
   where $\operatorname{Var}(\widehat{Z}(\mathbf{u}) - \widehat{Z}_v)$ is the spatial dispersion of the point kriged surface across the target block.

*Proof.*
1. Point kriging at coordinate $\mathbf{u} \in v(\mathbf{s}_0)$ satisfies $\begin{pmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda}(\mathbf{u}) \\ \mu(\mathbf{u}) \end{pmatrix} = \begin{pmatrix} \boldsymbol{\gamma}(\mathbf{u}) \\ 1 \end{pmatrix}$. Integrating both sides across $\mathbf{u} \in v(\mathbf{s}_0)$:
   $$
   \begin{pmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \left[\frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} \begin{pmatrix} \boldsymbol{\lambda}(\mathbf{u}) \\ \mu(\mathbf{u}) \end{pmatrix}\,d\mathbf{u}\right] = \begin{pmatrix} \frac{1}{\vert v\vert}\int_{v(\mathbf{s}_0)} \boldsymbol{\gamma}(\mathbf{u})\,d\mathbf{u} \\ 1 \end{pmatrix} = \begin{pmatrix} \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) \\ 1 \end{pmatrix}
   $$
   Because the system matrix is non-singular, its unique solution is the block kriging weight vector $\boldsymbol{\lambda}_v$.
2. Decompose the point error $Z(\mathbf{u}) - \widehat{Z}(\mathbf{u})$ around block averages $Z_v$ and $\widehat{Z}_v$:
   $$
   Z(\mathbf{u}) - \widehat{Z}(\mathbf{u}) = (Z_v - \widehat{Z}_v) + (Z(\mathbf{u}) - Z_v) - (\widehat{Z}(\mathbf{u}) - \widehat{Z}_v)
   $$
   Squaring, averaging over $\mathbf{u} \in v$, and taking expectations eliminates the cross terms by the orthogonality of kriging residuals, yielding the stated identity. The equality holds strictly without the dispersion correction term if and only if the kriged predictor is constant across the block (e.g., when a single observation is available, as in Exercise 8.3, or when the block is far from all data). $\blacksquare$

> [!NOTE]
> **From the Lecture Hall: The Information Effect: Why Kriged Block Estimates are Smoother than True Blocks**  
> Even though block kriging predicts the block average $Z_v$, the predicted values $\widehat{Z}_v$ do not have the same variance as the true block values $Z_v$!  
> Because kriging is a conditional expectation operator, the variance of predictions is strictly smaller than the true block variance:
> $$
> \operatorname{Var}(\widehat{Z}_v) = \operatorname{Var}(Z_v) - \sigma_{\text{OK}, v}^2 < \operatorname{Var}(Z_v)
> $$
> Applying an economic cut-off $z_c$ directly to kriged estimates $\widehat{Z}_v$ introduces **conditional bias** (the "information effect"): because the kriged map is overly smoothed, high-grade blocks are underestimated and low-grade blocks are overestimated. This is why non-linear geostatistics and conditional simulation (Chapters 9–10) are required for accurate mine planning!

```python
import numpy as np
import itertools
from typing import Callable, Sequence

def ordinary_block_kriging(
    data_coords: np.ndarray,
    data_values: np.ndarray,
    target_block_center: np.ndarray,
    block_dims: np.ndarray,
    n_disc: Sequence[int],
    gamma_func: Callable[[np.ndarray], np.ndarray]
) -> tuple[float, float, np.ndarray, float]:
    """
    Performs Ordinary Block Kriging for a rectangular block support.
    
    Parameters:
        data_coords: Array of shape (n, dim) with sample coordinates
        data_values: Array of shape (n,) with sample measurements
        target_block_center: Center coordinate of target block [x0, y0, ...]
        block_dims: Side lengths [dx, dy, ...] of target block
        n_disc: Quadrature discretization steps [mx, my, ...] per axis
        gamma_func: Callable computing semivariance from distance array
        
    Returns:
        (z_pred, ok_var, weights, mu_ok)
    """
    n = len(data_values)
    dim = len(block_dims)
    
    # Step 1: Discretize target block into M internal quadrature nodes
    coords_per_axis = []
    for d in range(dim):
        step = block_dims[d] / n_disc[d]
        start = target_block_center[d] - block_dims[d] / 2.0 + step / 2.0
        coords_per_axis.append([start + k * step for k in range(n_disc[d])])
    u_nodes = np.array(list(itertools.product(*coords_per_axis)))
    M = len(u_nodes)
    
    # Step 2: Compute within-block average semivariance gamma_bar(v, v)
    diffs_vv = u_nodes[:, None, :] - u_nodes[None, :, :]
    dists_vv = np.linalg.norm(diffs_vv, axis=-1)
    gamma_bar_vv = float(np.mean(gamma_func(dists_vv)))
    
    # Step 3: Compute point-to-block semivariance vector gamma_bar(s_i, v)
    diffs_iv = data_coords[:, None, :] - u_nodes[None, :, :]  # (n, M, dim)
    dists_iv = np.linalg.norm(diffs_iv, axis=-1)
    gamma_bar_iv = np.mean(gamma_func(dists_iv), axis=1)     # (n,)
    
    # Step 4: Assemble Ordinary Kriging system matrix
    diffs_data = data_coords[:, None, :] - data_coords[None, :, :]
    dists_data = np.linalg.norm(diffs_data, axis=-1)
    Gamma = gamma_func(dists_data)
    
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = Gamma
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    
    rhs = np.zeros(n + 1)
    rhs[:n] = gamma_bar_iv
    rhs[n] = 1.0
    
    # Step 5: Solve augmented linear system
    sol = np.linalg.solve(K, rhs)
    weights = sol[:n]
    mu_ok = sol[n]
    
    # Step 6: Evaluate predicted block value and block kriging variance
    z_pred = float(np.dot(weights, data_values))
    ok_var = float(np.dot(weights, gamma_bar_iv) + mu_ok - gamma_bar_vv)
    
    return z_pred, ok_var, weights, mu_ok
```

> [!TIP]
> **Data Points Inside the Target Block:**  
> When a drill hole falls directly inside the target block $v(\mathbf{s}_0)$, point kriging assigns it a weight of $1.0$ and returns zero error variance. In block kriging:
> - If only a single borehole exists ($n = 1$), it receives weight $\lambda_1 = 1.0$, but the block kriging variance remains strictly positive: $\sigma_{\text{OK}, V}^2 = 2\overline{\gamma}(\mathbf{s}_1, V) - \overline{\gamma}(V, V) > 0$ (Exercise 8.3).
> - When other data points are present in the neighborhood ($n > 1$), an interior borehole **does not receive weight 1.0** (typically $\lambda_1 \in [0.4, 0.7]$). Surrounding exterior samples contribute valuable information about the spatial trend and boundary grade of the block.

---

### Key Ideas and Takeaways: Change of Support and Spatial Regularization

| Principle | Formula / Characterization | Operational Implication | Common Pitfall to Avoid |
| :--- | :--- | :--- | :--- |
| **Mean Preservation** | $\operatorname{E}[Z_v(\mathbf{s})] = \mu$ | Volume averaging preserves the spatial mean grade exactly | Assuming regional drift or trend is preserved pointwise |
| **Variance Collapse** | $\sigma_v^2 = C(\mathbf{0}) - \overline{\gamma}(v, v)$ | Larger extraction units exhibit smaller variance | Believing drill-core variance applies to shovel buckets |
| **Integral Range Scaling** | $\operatorname{Var}(Z_V) \approx \sigma^2 \frac{A}{\vert V\vert}$ | Block variance shrinks at rate $A/\vert V\vert$ for large blocks | Assuming $1/n$ independence scaling without checking $A$ |
| **Regularized Semivariogram** | $\gamma_v(\mathbf{h}) = \overline{\gamma}(v, v_{\mathbf{h}}) - \overline{\gamma}(v, v)$ | Sill decreases by $\overline{\gamma}(v, v)$; range increases by block size | Forgetting that block variograms leave origin parabolically |
| **Spatial ANOVA Decomposition** | $D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)$ | Deposit variance = Within-panel variance + Between-panel variance | Believing dispersion variance is independent of panel size |
| **Cartier's Relation** | $\operatorname{E}[Z_v \mid Z_V] = Z_V$ (random $v$) | Fine-scale units have expectation equal to encompassing block | Assuming slope is 1.0 for a fixed edge sub-block |
| **Support Effect on Recovery** | $\operatorname{E}[(Z_V - z_c)^+] \le \operatorname{E}[(Z_v - z_c)^+]$ | Blocks recover less metal than points at high economic cut-offs | Using point histograms to forecast mine production |
| **Crossing Point Threshold** | $z^* > m$ for skewed grades | Block tonnage exceeds point tonnage for cut-offs between $m$ and $z^*$ | Assuming crossing point occurs at the mean for skewed data |
| **Block Kriging BLUE** | $\mathbf{K} \begin{pmatrix} \boldsymbol{\lambda} \\ \mu \end{pmatrix} = \begin{pmatrix} \overline{\boldsymbol{\gamma}}_v \\ 1 \end{pmatrix}$ | System matrix $\mathbf{\Gamma}$ inverted once; RHS averaged over block | Integrating point kriging variances without dispersion correction |

---

### 8.9 Analytical Exercises and Complete Step-by-Step Solutions

#### Exercise 8.1: Regularized Covariance of a 1D Segment under an Exponential Covariance
Let $\{Z(s) : s \in \mathbb{R}\}$ be a 1D second-order stationary random field with exponential covariance $C(h) = \sigma^2 \exp(-\vert h\vert / a)$, where $a > 0$ is the correlation range. Let $v = [0, L]$ be a line segment of length $L > 0$.
**(a)** Compute the exact analytical within-block average covariance $\overline{C}(v, v)$.  
**(b)** Deduce the dispersion variance of points within the segment $D^2(0 \mid v)$.  
**(c)** Derive the asymptotic limits of $\overline{C}(v, v)$ as $L/a \to 0$ and $L/a \to \infty$, connecting the result to the integral range $A$ and Chapter 1's effective sample size.

##### Solution:
**(a)** By Definition 8.2.1:
$$
\overline{C}(v, v) = \frac{1}{L^2} \int_0^L \int_0^L \sigma^2 \exp\left(-\frac{\vert u - u'\vert}{a}\right)\,du\,du'
$$
Exploiting symmetry across the diagonal of the square $[0, L] \times [0, L]$:
$$
\overline{C}(v, v) = \frac{2\sigma^2}{L^2} \int_0^L du \int_0^u \exp\left(-\frac{u - u'}{a}\right)\,du'
$$
Evaluating the inner integral:
$$
\int_0^u \exp\left(\frac{u' - u}{a}\right)\,du' = \left[ a \exp\left(\frac{u' - u}{a}\right) \right]_{u'=0}^u = a \left(1 - \exp\left(-\frac{u}{a}\right)\right)
$$
Now evaluate the outer integral:
$$
\begin{aligned}
\int_0^L a \left(1 - \exp\left(-\frac{u}{a}\right)\right)\,du &= a \left[ u + a \exp\left(-\frac{u}{a}\right) \right]_0^L \\
&= a \left( L + a e^{-L/a} - a \right) = a L - a^2 (1 - e^{-L/a})
\end{aligned}
$$
Dividing by $L^2$:
$$
\overline{C}(v, v) = \frac{2\sigma^2}{L^2} \left[ a L - a^2 (1 - e^{-L/a}) \right] = 2\sigma^2 \left(\frac{a}{L}\right) \left[ 1 - \left(\frac{a}{L}\right) (1 - e^{-L/a}) \right]
$$
Defining the dimensionless ratio $\theta = L / a$:
$$
\overline{C}(v, v) = 2\sigma^2 \left(\frac{1}{\theta}\right) \left[ 1 - \frac{1 - e^{-\theta}}{\theta} \right] = \frac{2\sigma^2}{\theta^2} (\theta - 1 + e^{-\theta})
$$

**(b)** The dispersion variance of points within the segment is:
$$
D^2(0 \mid v) = C(0) - \overline{C}(v, v) = \sigma^2 \left[ 1 - \frac{2}{\theta^2}(\theta - 1 + e^{-\theta}) \right] = \overline{\gamma}(v, v)
$$

**(c)** Asymptotic limits:
1. **Infinitesimal Segment ($L/a = \theta \to 0$):** Expand $e^{-\theta} = 1 - \theta + \frac{\theta^2}{2} - \frac{\theta^3}{6} + \mathcal{O}(\theta^4)$:
   $$
   \theta - 1 + e^{-\theta} = \theta - 1 + \left(1 - \theta + \frac{\theta^2}{2} - \frac{\theta^3}{6} + \dots\right) = \frac{\theta^2}{2} - \frac{\theta^3}{6} + \dots
   $$
   Substituting into $\overline{C}(v, v)$:
   $$
   \lim_{\theta \to 0} \overline{C}(v, v) = \lim_{\theta \to 0} \frac{2\sigma^2}{\theta^2} \left(\frac{\theta^2}{2} - \frac{\theta^3}{6}\right) = \sigma^2 \left(1 - \frac{\theta}{3}\right) = \sigma^2
   $$
   As $L \to 0$, within-block variance matches point variance $\sigma^2$, and $D^2(0 \mid v) \to 0$.
2. **Infinite Segment ($L/a = \theta \to \infty$):** The exponential term $e^{-\theta}$ vanishes, yielding:
   $$
   \overline{C}(v, v) \approx \frac{2\sigma^2}{\theta} = \sigma^2 \frac{2a}{L}
   $$
   Notice that $2a = \int_{-\infty}^\infty \exp(-\vert h\vert / a)\,dh$ is identically the **integral range** $A$ of the 1D exponential covariance! Thus $\overline{C}(v, v) \approx \sigma^2 A / L$, which is Chapter 1's effective sample size formula $\sigma^2 / N_{\text{eff}}$ with $N_{\text{eff}} = L / A = L / (2a)$.

Evaluating the ratio $\overline{C}(v, v) / (2\sigma^2 a / L) = 1 - \frac{1 - e^{-\theta}}{\theta}$ confirms rapid convergence to the asymptotic limit:
- At $L/a = 1$: ratio is $1 - (1 - e^{-1})/1 = e^{-1} \approx 0.368$
- At $L/a = 5$: ratio is $1 - (1 - e^{-5})/5 \approx 0.801$
- At $L/a = 20$: ratio is $1 - (1 - e^{-20})/20 \approx 0.950$
- At $L/a = 100$: ratio is $1 - 1/100 = 0.990$

---

#### Exercise 8.2: Dispersion Variance Additivity in a Stratified Deposit
A continuous borehole of length $W = 12\text{ m}$ is partitioned into $N = 3$ panels of length $V = 4\text{ m}$, each of which is cut into $M = 4$ assay samples of length $v = 1\text{ m}$.
The spatial semivariogram along the borehole is spherical:
$$
\gamma(h) = \begin{cases} c \left(\frac{3\vert h\vert}{2a} - \frac{\vert h\vert^3}{2a^3}\right), & \vert h\vert \le a \\ c, & \vert h\vert > a \end{cases}
$$
with sill $c = 10.0$ and range $a = 6.0\text{ m}$.
The auxiliary function $\chi(L) = \overline{\gamma}(L, L)$ for a segment of length $L \le a$ is:
$$
\chi(L) = c \left(\frac{L}{2a} - \frac{L^3}{20a^3}\right)
$$
and for $L > a$:
$$
\chi(L) = c \left(1 - \frac{3a}{4L} + \frac{a^2}{5L^2}\right)
$$
**(a)** Compute $\overline{\gamma}(v, v)$ and $\overline{\gamma}(V, V)$.  
**(b)** Compute $\overline{\gamma}(W, W)$.  
**(c)** Verify Krige's Additivity Theorem: $D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)$ and provide an independent computational verification.

##### Solution:
**(a)** For sample support $v = 1\text{ m} \le a = 6\text{ m}$:
$$
\overline{\gamma}(v, v) = \chi(1) = 10 \left(\frac{1}{2(6)} - \frac{1^3}{20(6)^3}\right) = 10 \left(\frac{1}{12} - \frac{1}{4320}\right) = 10 \left(\frac{359}{4320}\right) = \frac{359}{432} \approx 0.8310
$$
For panel support $V = 4\text{ m} \le a = 6\text{ m}$:
$$
\overline{\gamma}(V, V) = \chi(4) = 10 \left(\frac{4}{12} - \frac{4^3}{20(216)}ight) = 10 \left(\frac{1}{3} - \frac{64}{4320}\right) = 10 \left(\frac{1440 - 64}{4320}\right) = \frac{1376}{432} \approx 3.1852
$$

**(b)** For total deposit support $W = 12\text{ m} > a = 6\text{ m}$, use the formula for $L > a$:
$$
\begin{aligned}
\overline{\gamma}(W, W) &= \chi(12) = 10 \left(1 - \frac{3(6)}{4(12)} + \frac{6^2}{5(12)^2}\right) \\
&= 10 \left(1 - \frac{18}{48} + \frac{36}{5 \times 144}\right) = 10 \left(1 - \frac{3}{8} + \frac{1}{20}\right) \\
&= 10 \left(\frac{5}{8} + \frac{1}{20}\right) = 10 \left(\frac{25 + 2}{40}\right) = 10 \left(\frac{27}{40}\right) = \frac{27}{4} = 6.7500
\end{aligned}
$$
In common denominator $432$: $6.75 = \frac{6.75 \times 432}{432} = \frac{2916}{432}$.

**(c)** Compute the three dispersion variances:
$$
\begin{aligned}
D^2(v \mid V) &= \overline{\gamma}(V, V) - \overline{\gamma}(v, v) = \frac{1376 - 359}{432} = \frac{1017}{432} \approx 2.3542 \\
D^2(V \mid W) &= \overline{\gamma}(W, W) - \overline{\gamma}(V, V) = \frac{2916 - 1376}{432} = \frac{1540}{432} \approx 3.5648 \\
D^2(v \mid W) &= \overline{\gamma}(W, W) - \overline{\gamma}(v, v) = \frac{2916 - 359}{432} = \frac{2557}{432} \approx 5.9190
\end{aligned}
$$
Summing the components:
$$
D^2(v \mid V) + D^2(V \mid W) = \frac{1017 + 1540}{432} = \frac{2557}{432} \approx 5.9190 = D^2(v \mid W)
$$
Krige's Additivity Theorem is verified analytically.

**Independent Computational Verification:**  
To confirm that this identity is not merely an artifact of definitions, we evaluate the 1-million-pair Monte Carlo numerical integration of $\frac{1}{L^2}\int_0^L \int_0^L \gamma(u - u')\,du\,du'$:
- $\overline{\gamma}(W, W)_{\text{num}} = 6.744 \approx 6.750$;
- $\overline{\gamma}(V, V)_{\text{num}} = 3.185 \approx 3.185$;
- $\overline{\gamma}(v, v)_{\text{num}} = 0.830 \approx 0.831$.

Furthermore, discretizing the 12 one-meter blocks into $M = 50$ nodes each and evaluating the discrete sample ANOVA sum of squares across the block covariances yields:
$$
D^2(v \mid V) = 2.341, \quad D^2(V \mid W) = 3.562 \implies \text{Sum} = 5.903 \approx 5.919 = D^2(v \mid W)
$$
confirming the exactness of the physical variance decomposition.

---

#### Exercise 8.3: Bias and Variance in Block Kriging with a Single Point Observation
Let $\{Z(\mathbf{s})\}$ be a 2D second-order stationary random field with unknown mean $\mu$ and continuous covariance $C(\mathbf{h})$. A mining block $V$ is estimated from a single point sample $Z(\mathbf{s}_1)$ located at the geometric center of $V$.
**(a)** Formulate the Ordinary Block Kriging estimator $\widehat{Z}_V = \lambda_1 Z(\mathbf{s}_1)$ and determine $\lambda_1$.  
**(b)** Derive the block kriging error variance $\sigma_{\text{OK}, V}^2$.  
**(c)** Show that the average point prediction error variance across $V$ exceeds $\sigma_{\text{OK}, V}^2$ by exactly $D^2(\mathbf{0} \mid V)$, and explain why the dispersion correction term from Corollary 8.8.1 vanishes here.

##### Solution:
**(a)** The Ordinary Kriging unbiasedness condition requires:
$$
\sum_{i=1}^1 \lambda_i = 1 \implies \lambda_1 = 1.0
$$
The linear predictor is $\widehat{Z}_V = Z(\mathbf{s}_1)$.

**(b)** The Ordinary Block Kriging system in covariance form for $n = 1$ is:
$$
\lambda_1 C(\mathbf{0}) - \mu_C = \overline{C}(\mathbf{s}_1, V) \implies \mu_C = C(\mathbf{0}) - \overline{C}(\mathbf{s}_1, V)
$$
The block kriging estimation variance is:
$$
\sigma_{\text{OK}, V}^2 = \overline{C}(V, V) - \lambda_1 \overline{C}(\mathbf{s}_1, V) + \mu_C = C(\mathbf{0}) - 2\overline{C}(\mathbf{s}_1, V) + \overline{C}(V, V)
$$
Substituting $C(\mathbf{h}) = C(\mathbf{0}) - \gamma(\mathbf{h})$:
$$
\begin{aligned}
\sigma_{\text{OK}, V}^2 &= C(\mathbf{0}) - 2[C(\mathbf{0}) - \overline{\gamma}(\mathbf{s}_1, V)] + [C(\mathbf{0}) - \overline{\gamma}(V, V)] \\
&= 2\overline{\gamma}(\mathbf{s}_1, V) - \overline{\gamma}(V, V)
\end{aligned}
$$

**(c)** If $Z(\mathbf{s}_1)$ is used to predict point value $Z(\mathbf{u})$ at $\mathbf{u} \in V$, the point estimation error variance is:
$$
\sigma^2_{\text{point}}(\mathbf{u}) = \operatorname{E}[(Z(\mathbf{s}_1) - Z(\mathbf{u}))^2] = 2\gamma(\mathbf{s}_1 - \mathbf{u})
$$
Averaging this point error variance over all locations $\mathbf{u} \in V$:
$$
\overline{\sigma^2_{\text{point}}} = \frac{1}{\vert V\vert}\int_V 2\gamma(\mathbf{s}_1 - \mathbf{u})\,d\mathbf{u} = 2\overline{\gamma}(\mathbf{s}_1, V)
$$
Subtracting the block kriging variance:
$$
\overline{\sigma^2_{\text{point}}} - \sigma_{\text{OK}, V}^2 = 2\overline{\gamma}(\mathbf{s}_1, V) - [2\overline{\gamma}(\mathbf{s}_1, V) - \overline{\gamma}(V, V)] = \overline{\gamma}(V, V) = D^2(\mathbf{0} \mid V)
$$
Hence:
$$
\overline{\sigma^2_{\text{point}}} = \sigma_{\text{OK}, V}^2 + D^2(\mathbf{0} \mid V)
$$
In Corollary 8.8.1, the general variance identity is:
$$
\overline{\sigma^2_{\text{point}}} = \sigma_{\text{OK}, V}^2 + \overline{\gamma}(V, V) - \operatorname{Var}_V(\widehat{Z}(\mathbf{u}))
$$
With a single datum $Z(\mathbf{s}_1)$, the point kriging predictor at any target node $\mathbf{u} \in V$ is $\widehat{Z}(\mathbf{u}) = 1.0 \cdot Z(\mathbf{s}_1)$, which is spatially constant across the block. Consequently, the spatial dispersion of the kriged surface across $V$ is identically zero: $\operatorname{Var}_V(\widehat{Z}(\mathbf{u})) = 0$. This confirms why the simple sum holds exactly for a single datum.

---

#### Exercise 8.4: Indirect Lognormal Change of Support for Mining Selectivity
A gold deposit has point-support drill-core grades $Z = \exp(Y)$ with $Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$, where $\mu_Y = 0.5$ and $\sigma_Y^2 = 1.0$.
Mining will proceed using Selective Mining Units (blocks $v$) for which numerical quadrature yields within-block average covariance $\overline{C}(v, v) = 2.50$.
The economic cut-off grade is $z_c = 3.0\text{ g/t}$.
**(a)** Compute the arithmetic mean grade $m = \operatorname{E}[Z]$ and point variance $\sigma_0^2 = \operatorname{Var}(Z)$.  
**(b)** Using the Indirect Lognormal Correction model, compute the block logarithmic parameters $\sigma_{Y_v}^2$ and $\mu_{Y_v}$.  
**(c)** Calculate the recovered tonnage fraction $T_v(z_c)$, quantity of metal $Q_v(z_c)$, and mean grade above cut-off $m_v(z_c)$ for block support $v$.  
**(d)** Compare $T_v(z_c)$, $Q_v(z_c)$, and $m_v(z_c)$ with naive point-support selection $T_0(z_c)$, $Q_0(z_c)$, and $m_0(z_c)$, identifying the crossing point $z^*$ and interpreting the operational consequences.

##### Solution:
**(a)** Point support parameters:
$$
m = \exp\left(\mu_Y + \frac{1}{2}\sigma_Y^2\right) = \exp(0.5 + 0.5) = \exp(1.0) \approx 2.7183\text{ g/t}
$$
$$
\sigma_0^2 = m^2 [\exp(\sigma_Y^2) - 1] = (2.7183)^2 [\exp(1.0) - 1] = 7.3891 \times 1.7183 \approx 12.6966\text{ (g/t)}^2
$$

**(b)** Block logarithmic parameters:
The block variance is $\sigma_v^2 = \overline{C}(v, v) = 2.50$:
$$
\sigma_{Y_v}^2 = \ln\left(1 + \frac{\sigma_v^2}{m^2}\right) = \ln\left(1 + \frac{2.50}{7.3891}\right) = \ln(1 + 0.33834) = \ln(1.33834) \approx 0.2914
$$
$$
\sigma_{Y_v} = \sqrt{0.2914} \approx 0.5398
$$
$$
\mu_{Y_v} = \ln(m) - \frac{1}{2}\sigma_{Y_v}^2 = 1.0 - \frac{1}{2}(0.2914) = 1.0 - 0.1457 = 0.8543
$$

**(c)** Recovered block quantities at cut-off $z_c = 3.0\text{ g/t}$ (with $\ln z_c = \ln 3.0 \approx 1.0986$):
$$
d_1 = \frac{\mu_{Y_v} - \ln z_c}{\sigma_{Y_v}} = \frac{0.8543 - 1.0986}{0.5398} = \frac{-0.2443}{0.5398} \approx -0.4526
$$
Recovered tonnage fraction:
$$
T_v(z_c) = \Phi(-0.4526) \approx 0.3254 \quad (32.54\%)
$$
Recovered metal quantity:
$$
d_2 = d_1 + \sigma_{Y_v} = -0.4526 + 0.5398 = 0.0872
$$
$$
Q_v(z_c) = m \cdot \Phi(d_2) = 2.7183 \cdot \Phi(0.0872) \approx 2.7183 \times 0.5347 \approx 1.4536\text{ g/t}
$$
Mean recovered block grade:
$$
m_v(z_c) = \frac{Q_v(z_c)}{T_v(z_c)} = \frac{1.4536}{0.3254} \approx 4.467\text{ g/t}
$$

**(d)** Point support selection at cut-off $z_c = 3.0\text{ g/t}$:
$$
d_{1, 0} = \frac{\mu_Y - \ln z_c}{\sigma_Y} = \frac{0.5 - 1.0986}{1.0} = -0.5986
$$
$$
T_0(z_c) = \Phi(-0.5986) \approx 0.2747 \quad (27.47\%)
$$
$$
d_{2, 0} = d_{1, 0} + \sigma_Y = -0.5986 + 1.0 = 0.4014
$$
$$
Q_0(z_c) = m \cdot \Phi(0.4014) = 2.7183 \times 0.6559 \approx 1.7830\text{ g/t}
$$
$$
m_0(z_c) = \frac{Q_0(z_c)}{T_0(z_c)} = \frac{1.7830}{0.2747} \approx 6.491\text{ g/t}
$$

**Detailed Operational Comparison:**
1. **Grade Overestimation:** Naive point-support evaluation forecasts a delivered grade of $6.491\text{ g/t}$, whereas the true block grade delivered by mining shovels is $4.467\text{ g/t}$. Naive point support overstates delivered grade by:
   $$
   \frac{6.491 - 4.467}{4.467} \times 100\% = +45.3\%
   $$
   Equivalently, the realistic block grade is $\frac{4.467 - 6.491}{6.491} = -31.2\%$ lower than naive point forecasts.
2. **Metal Overstatement:** Point support forecasts $1.783\text{ g/t}$ of recoverable metal, whereas block support delivers $1.454\text{ g/t}$—a **$+22.7\%$ overstatement of recoverable metal**.
3. **The Crossing Point Phenomenon:** Notice that at cut-off $z_c = 3.0\text{ g/t}$, block tonnage is **higher** than point tonnage ($32.54\%$ vs. $27.47\%$), even though $z_c = 3.0 > m = 2.72\text{ g/t}$!  
   Equating the point and block lognormal densities reveals that the unique crossing point is:
   $$
   z^* = \exp\left( \frac{\sigma_{Y_v}\mu_Y - \sigma_Y \mu_{Y_v}}{\sigma_{Y_v} - \sigma_Y} \right) \approx 3.56\text{ g/t}
   $$
   Because $z_c = 3.0 < z^* = 3.56$, the cut-off falls in the intermediate window where the contracted block distribution concentrates higher density than the diffuse point distribution. More mining blocks qualify as ore, but their diluted quality produces a catastrophic drop in recovered grade if mill circuits were engineered for $6.5\text{ g/t}$ feed!

---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:  
👉 [Chapter 8: Change of Support and Volume Regularization — Companion Exercises and Complete Solutions](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch08_exercises.md)
