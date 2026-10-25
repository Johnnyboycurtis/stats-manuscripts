# Chapter 8: Change of Support and Spatial Regularization

This chapter develops the mathematical theory of the change-of-support problem in spatial statistics and geostatistics. In physical and earth sciences, measurements are rarely collected at dimensionless geometric points; rather, observations and engineering decisions occur over physical volumes: drill cores, satellite pixels, environmental sensor cones, or mining blocks. The geometric domain over which an attribute is measured or averaged is termed its **support**. 

When moving from a point-support observation process $Z(\mathbf{s})$ to a volumetric average $Z_v(\mathbf{s})$, the spatial variance collapses, spatial continuity regularizes, extreme tails attenuate, and the underlying probability law undergoes a fundamental transformation. Following the pedagogical architecture of Maria L. Rizzo and the semantic linear-model discipline of Oliver Schabenberger & Carol A. Gotway, this chapter establishes the measure-theoretic foundation of spatial regularization, proves Krige's Additivity Theorem and Cartier's spatial martingale relation, develops analytical change-of-support models, and derives the Ordinary Block Kriging system with explicit numerical algorithms.

---

### 8.1 The Volume-Support Problem and Mean-Square Regularization

#### Beat 1: The Mathematical Anchor (The "Why")

In classical linear models, independent observations share a common identity regardless of aggregation geometry. In spatial linear models, however, the target variable depends intrinsically on the Lebesgue measure of the spatial support. Consider the three-layer structural decomposition (Schabenberger & Gotway):
$$
Z(\mathbf{s}) = \mu(\mathbf{s}) + \delta(\mathbf{s}) + \epsilon(\mathbf{s})
$$
where $\mathbf{s} \in D \subset \mathbb{R}^d$ ($d \ge 1$) represents a continuous Euclidean coordinate vector, $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ is the deterministic drift, $\delta(\mathbf{s})$ is a zero-mean second-order stationary spatial process with covariance kernel $C(\mathbf{h}) = \operatorname{Cov}(\delta(\mathbf{s}), \delta(\mathbf{s}+\mathbf{h}))$, and $\epsilon(\mathbf{s})$ is independent microscale white noise with variance $\tau^2$ (the nugget effect).

#### Classical-to-Spatial Bridge: From ANOVA Variance Decomposition and the Central Limit Theorem to Krige's Additivity Theorem

In classical statistics, consider a one-way Analysis of Variance (ANOVA) design where $N$ independent observations $Y_{ij}$ are partitioned into $k$ groups of size $m$, such that $N = k m$:
$$
Y_{ij} = \mu + \alpha_i + \epsilon_{ij}, \quad i=1,\dots, k, \quad j=1,\dots, m
$$
where $\alpha_i$ is the group effect and $\epsilon_{ij} \overset{\text{iid}}{\sim} (0, \sigma^2)$ is residual error. The Total Sum of Squares decomposes orthogonally into Between-Group and Within-Group variations:
$$
\text{SS}_{\text{Total}} = \text{SS}_{\text{Between}} + \text{SS}_{\text{Within}}
$$
Dividing by degrees of freedom reveals the fundamental variance decomposition: the overall sample variance partitions into the variance of group means plus the average variance within individual groups. Furthermore, by the classical Central Limit Theorem, the variance of an arithmetic group mean shrinks in strict inverse proportion to group sample size $m$:
$$
\operatorname{Var}(\bar{Y}_i) = \frac{\sigma^2}{m}
$$
As sample size $m$ grows, the variability among group means collapses toward zero, while the expected value remains centered at the grand population mean: $\operatorname{E}[\bar{Y}_i] = \mu$.

Now transition to spatial statistics and geostatistics. Observations are not isolated discrete samples; they are collected over continuous physical volumes or supports. Let $v$ denote a small sample support (e.g., a diamond drill-core cylinder of diameter $5\text{ cm}$), nested inside an intermediate extraction or mining block $V$ (e.g., a $10\text{ m} \times 10\text{ m} \times 5\text{ m}$ mining bench panel), which is in turn nested inside the entire deposit or study field $W$:
$$
v \subset V \subset W \quad \text{with } 0 < |v| \le |V| \le |W|
$$
In spatial continuous fields, spatial dispersion variance replaces discrete ANOVA sums of squares. The dispersion variance $D^2(A \mid B)$ quantifies the variability of spatial units of support $A$ within a larger geometric domain $B$.

The spatial generalization of the ANOVA sum of squares is **Krige's Additivity Theorem**:
$$
D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)
$$
This theorem proves that the total dispersion variance of point or drill-core samples $v$ across the entire field $W$ decomposes additively into:
1. $D^2(v \mid V)$: The internal dispersion variance of small supports $v$ *within* the blocks $V$ (the spatial analog of $\text{SS}_{\text{Within}}$).
2. $D^2(V \mid W)$: The dispersion variance of block averages $V$ *across* the field $W$ (the spatial analog of $\text{SS}_{\text{Between}}$).

Unlike the classical i.i.d. setting where $\operatorname{Var}(\bar{Y}) = \sigma^2/m$, spatial data are autocorrelated. Spatial volume regularization acts as a **spatial low-pass convolution filter**:
$$
Z_V(\mathbf{s}) = (Z * k_V)(\mathbf{s}) = \frac{1}{|V|} \int_V Z(\mathbf{s} + \mathbf{u}) d\mathbf{u}
$$
where $k_V(\mathbf{u}) = \frac{1}{|V|}\mathbb{I}_V(\mathbf{u})$ is a spatial moving-average box filter. In the spatial frequency domain, convolution with the box kernel $k_V$ multiplies the spatial spectral density by a squared sinc-type transfer function $|\widehat{k}_V(\boldsymbol{\omega})|^2$. This transfer function acts as an aggressive low-pass filter:
- High spatial frequencies (short-scale stochastic chatter, microscale heterogeneity, and measurement error $\tau^2$) are strongly attenuated.
- Low spatial frequencies (regional trends, grand mean) are preserved exactly: $\operatorname{E}[Z_V(\mathbf{s})] = \operatorname{E}[Z(\mathbf{s})] = \mu$.
- The dispersion variance of block supports $D^2(V \mid W) = \operatorname{Var}(Z_V)$ is strictly smaller than the point-support dispersion variance $D^2(v \mid W) = \operatorname{Var}(Z_v)$, but it does not decay as $1/|V|$ because spatial autocorrelation maintains covariance between neighboring points within $V$.

When a physical sensor or extraction tool samples a finite volume $v \subset D$ centered at $\mathbf{s}$, it does not record $Z(\mathbf{s})$; it records an integrated spatial average.

##### Definition 8.1.1 (Spatial Support and Regularized Random Field)
Let $(\Omega, \mathcal{F}, P)$ be a complete probability space, and let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a real-valued second-order spatial random field on $D \subset \mathbb{R}^d$ with mean $\mu(\mathbf{s})$ and continuous covariance kernel $C(\mathbf{s}_1, \mathbf{s}_2)$.
1. A **block support** $v \subset \mathbb{R}^d$ is a non-empty compact subset possessing strictly positive and finite Lebesgue measure:
   $$
   |v| = \operatorname{Leb}(v) = \int_v d\mathbf{u} \in (0, \infty)
   $$
2. For any reference location $\mathbf{s} \in D$, the translate of $v$ centered at $\mathbf{s}$ is $v(\mathbf{s}) = \mathbf{s} + v = \{\mathbf{s} + \mathbf{u} : \mathbf{u} \in v\}$.
3. The **regularized spatial random field** $Z_v(\mathbf{s})$ is the volume average of $Z$ over $v(\mathbf{s})$:
   $$
   Z_v(\mathbf{s}) = \frac{1}{|v|} \int_{v(\mathbf{s})} Z(\mathbf{u}) d\mathbf{u} = \frac{1}{|v|} \int_v Z(\mathbf{s} + \mathbf{u}) d\mathbf{u}
   $$
   where the integral is defined in the mean-square (Riemann-Bochner) sense in the Hilbert space $L^2(\Omega, \mathcal{F}, P)$.

##### Theorem 8.1.1 (Existence, Linearity, and Continuity of $Z_v$)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a second-order random field with continuous covariance kernel $C: D \times D \to \mathbb{R}$. Then for any compact support $v \subset \mathbb{R}^d$ such that $v(\mathbf{s}) \subset D$:
1. The mean-square integral defining $Z_v(\mathbf{s})$ exists as a unique element in $L^2(\Omega, \mathcal{F}, P)$.
2. The expectation operator commutes with spatial integration:
   $$
   \operatorname{E}[Z_v(\mathbf{s})] = \frac{1}{|v|} \int_v \operatorname{E}[Z(\mathbf{s} + \mathbf{u})] d\mathbf{u}
   $$
   Under constant mean $\mu(\mathbf{s}) \equiv \mu$, $\operatorname{E}[Z_v(\mathbf{s})] = \mu$.
3. If $\{Z(\mathbf{s})\}$ is mean-square continuous, $\{Z_v(\mathbf{s})\}$ is mean-square continuous and possesses $P$-almost surely continuous sample paths.

*Proof.*
1. Let $\mathcal{P}_m = \{v_{m,1}, \dots, v_{m,k_m}\}$ be a sequence of Borel partitions of $v$ whose mesh diameter $\delta(\mathcal{P}_m) = \max_j \operatorname{diam}(v_{m,j}) \to 0$ as $m \to \infty$. Choose arbitrary evaluation points $\mathbf{u}_{m,j} \in v_{m,j}$ and construct the Riemann sum in $L^2(\Omega, \mathcal{F}, P)$:
   $$
   S_m = \frac{1}{|v|} \sum_{j=1}^{k_m} Z(\mathbf{s} + \mathbf{u}_{m,j}) |v_{m,j}|
   $$
   For two partitions $\mathcal{P}_m$ and $\mathcal{P}_l$, evaluate the mean-square distance:
   $$
   \|S_m - S_l\|_{L^2}^2 = \operatorname{E}\left[(S_m - S_l)^2\right] = \operatorname{E}[S_m^2] - 2\operatorname{E}[S_m S_l] + \operatorname{E}[S_l^2]
   $$
   Expanding the bilinear product expectation:
   $$
   \operatorname{E}[S_m S_l] = \frac{1}{|v|^2} \sum_{j=1}^{k_m} \sum_{k=1}^{k_l} \operatorname{E}\left[Z(\mathbf{s} + \mathbf{u}_{m,j}) Z(\mathbf{s} + \mathbf{u}_{l,k})\right] |v_{m,j}| |v_{l,k}|
   $$
   Because the covariance kernel $C(\mathbf{s}_1, \mathbf{s}_2)$ and mean $\mu(\mathbf{s})$ are continuous, the bivariate function $(\mathbf{u}, \mathbf{u}') \mapsto \operatorname{E}[Z(\mathbf{s}+\mathbf{u}) Z(\mathbf{s}+\mathbf{u}')]$ is continuous on the compact domain $v \times v$. Hence it is Riemann integrable:
   $$
   \lim_{m, l \to \infty} \operatorname{E}[S_m S_l] = \frac{1}{|v|^2} \int_v \int_v \operatorname{E}[Z(\mathbf{s}+\mathbf{u}) Z(\mathbf{s}+\mathbf{u}')] d\mathbf{u} d\mathbf{u}'
   $$
   Thus, $\|S_m - S_l\|_{L^2}^2 \to 0$ as $m, l \to \infty$. Completeness of $L^2(\Omega, \mathcal{F}, P)$ guarantees the existence of a unique limit $Z_v(\mathbf{s}) \in L^2(\Omega, \mathcal{F}, P)$, independent of partition choice.
2. For any $X \in L^2(\Omega, \mathcal{F}, P)$, the inner product $\langle Z_v(\mathbf{s}), X \rangle = \operatorname{E}[Z_v(\mathbf{s}) X]$ is a continuous linear functional. Setting $X \equiv 1$ yields:
   $$
   \operatorname{E}[Z_v(\mathbf{s})] = \lim_{m \to \infty} \operatorname{E}[S_m] = \lim_{m \to \infty} \frac{1}{|v|} \sum_{j=1}^{k_m} \operatorname{E}[Z(\mathbf{s}+\mathbf{u}_{m,j})] |v_{m,j}| = \frac{1}{|v|} \int_v \operatorname{E}[Z(\mathbf{s}+\mathbf{u})] d\mathbf{u}
   $$
   Under a constant mean $\mu$, $\operatorname{E}[Z_v(\mathbf{s})] = \frac{1}{|v|} \int_v \mu \, d\mathbf{u} = \mu$.
3. Mean-square continuity follows from bounding $\|Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s})\|_{L^2}^2$ by the modulus of continuity of $C$ on compact sets, which vanishes as $\|\mathbf{h}\| \to 0$. $\blacksquare$

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why does the volume support effect reduce variance while preserving the spatial mean?*  
> The expectation operator $\operatorname{E}[\cdot]$ and the spatial integral $\frac{1}{|V|}\int_V (\cdot) d\mathbf{u}$ are both linear operators. By Fubini's theorem (Theorem 8.1.1), they commute:
> $$
> \operatorname{E}[Z_V] = \operatorname{E}\left[\frac{1}{|V|}\int_V Z(\mathbf{u}) d\mathbf{u}\right] = \frac{1}{|V|}\int_V \operatorname{E}[Z(\mathbf{u})] d\mathbf{u} = \frac{1}{|V|}\int_V \mu \, d\mathbf{u} = \mu
> $$
> Because expectation is linear, integrating over volume does not add, destroy, or displace the expected mass of the regionalized variable.
> 
> In contrast, variance is a quadratic functional of deviations from the mean. When we integrate over a positive support volume $|V| > 0$, positive stochastic fluctuations from rich zones and negative stochastic fluctuations from depleted zones inside the same block cancel each other out. Mathematically, the block variance is:
> $$
> \operatorname{Var}(Z_V) = \frac{1}{|V|^2}\int_V \int_V C(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w} \le C(\mathbf{0}) = \operatorname{Var}(Z(\mathbf{s}))
> $$
> The Cauchy-Schwarz inequality guarantees that unless the spatial field is perfectly correlated ($C(\mathbf{h}) \equiv C(\mathbf{0})$), the double integral over distinct points $\mathbf{u} \ne \mathbf{w}$ strictly damps the variance. High-grade and low-grade extremes are physically smoothed into intermediate block values, compressing the tails of the probability density function toward the spatial mean.

---

#### Beat 2: Algorithmic Translation (The "How")

In operational geostatistics, analytical integration of $C(\mathbf{u} - \mathbf{u}')$ over arbitrary polygons or polyhedra is generally intractable. Instead, the support volume $v$ is discretized into $M$ regular or quasi-Monte Carlo points $\{\mathbf{u}_k\}_{k=1}^M \subset v$, each representing an elemental volume $\Delta v = |v| / M$.

```
Discretization of Block Support v into M Internal Nodes:
+-------------------------------------------------------+
|   * u_1          * u_2          * u_3          * u_4  |
|                                                       |
|   * u_5          * u_6          * u_7          * u_8  |
|                                                       |
|   * u_9          * u_10         * u_11         * u_12 |
|                                                       |
|   * u_13         * u_14         * u_15         * u_16 |
+-------------------------------------------------------+
| Block volume |v|; elemental volume for each node = |v|/M|
+-------------------------------------------------------+
```

```python
def compute_regularized_covariance_mc(
    block_dims: list[float],
    n_disc: list[int],
    cov_func,
    h_lag: list[float]
) -> tuple[float, float]:
    """
    Computes within-block covariance C_bar(v, v) and between-block
    lagged covariance C_bar(v, v_h) via regular grid discretization.
    
    Parameters:
        block_dims: [dx, dy, dz] side lengths of the compact block v
        n_disc: [mx, my, mz] number of discretization steps per axis
        cov_func: callable C(h_vector) returning scalar point covariance
        h_lag: [hx, hy, hz] spatial translation lag vector h
        
    Returns:
        (c_bar_vv, c_bar_vvh): within-block and lagged block covariances
    """
    dim = len(block_dims)
    # Step 1: Generate regular grid coordinates centered inside block v
    coords_per_axis = []
    for d in range(dim):
        step = block_dims[d] / n_disc[d]
        start = -block_dims[d] / 2.0 + step / 2.0
        coords_per_axis.append([start + k * step for k in range(n_disc[d])])
    
    # Cartesian product to form M discretization points
    import itertools
    u_nodes = list(itertools.product(*coords_per_axis))
    M = len(u_nodes)
    
    # Step 2: Double loop over discretization points for C_bar(v, v)
    sum_cov_vv = 0.0
    for j in range(M):
        for k in range(M):
            # Separation vector between internal points u_j and u_k
            sep = [u_nodes[j][d] - u_nodes[k][d] for d in range(dim)]
            sum_cov_vv += cov_func(sep)
    c_bar_vv = sum_cov_vv / (M * M)
    
    # Step 3: Double loop for lagged covariance C_bar(v, v_h)
    sum_cov_vvh = 0.0
    for j in range(M):
        for k in range(M):
            # Point in v is u_j; point in v_h is u_k + h_lag
            sep_h = [u_nodes[j][d] - (u_nodes[k][d] + h_lag[d]) for d in range(dim)]
            sum_cov_vvh += cov_func(sep_h)
    c_bar_vvh = sum_cov_vvh / (M * M)
    
    return c_bar_vv, c_bar_vvh
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **The Drill-Core vs. Mining Shovel Dilemma:**
  Consider diamond drill-core samples of diameter $5 \text{ cm}$ versus a mining excavation shovel with a bucket capacity of $40 \text{ m}^3$ (a Selective Mining Unit, SMU). The drill-core data measure point support $Z(\mathbf{s})$; the shovel excavates block support $Z_v(\mathbf{s})$. 
  If drill-core assays show grades ranging from $0.1 \text{ g/t}$ to $80.0 \text{ g/t}$ gold, a naive engineer might design a processing mill expecting $80 \text{ g/t}$ feed. But the shovel cannot selectively extract a $5 \text{ cm}$ core: it extracts the entire $40 \text{ m}^3$ block, which averages rich veins with surrounding barren waste rock. As a consequence, the extreme high grades are physically averaged away. Ignoring support volume leads to severe financial misallocation.
- **Quadrature Error vs. Grid Density:**
  The Monte Carlo approximation error of $\overline{C}(v, v)$ scales as $\mathcal{O}(M^{-1/d})$ for uniform lattices in $d$ dimensions. In practice, a $4 \times 4 \times 4$ discretization ($M = 64$ points) achieves $< 1\%$ error for isotropic exponential models, whereas an inadequate $2 \times 2 \times 2$ grid systematically overestimates within-block variance.

---

### 8.2 Regularized Covariances and Semivariograms

#### Beat 1: The Mathematical Anchor (The "Why")

How does volumetric integration alter the spatial continuity structure? We express regularized covariances and semivariograms in terms of spatial cross-averaging operators.

##### Definition 8.2.1 (Average Covariance and Semivariogram Operators)
Let $A, B \subset \mathbb{R}^d$ be two compact subsets with $|A| > 0$ and $|B| > 0$.
1. The **average covariance** $\overline{C}(A, B)$ is:
   $$
   \overline{C}(A, B) = \frac{1}{|A||B|} \int_A \int_B C(\mathbf{u} - \mathbf{u}') d\mathbf{u} d\mathbf{u}'
   $$
2. The **average semivariogram** $\overline{\gamma}(A, B)$ is:
   $$
   \overline{\gamma}(A, B) = \frac{1}{|A||B|} \int_A \int_B \gamma(\mathbf{u} - \mathbf{u}') d\mathbf{u} d\mathbf{u}'
   $$
3. For $A = B = v$, the within-block terms are $\overline{C}(v, v)$ and $\overline{\gamma}(v, v)$.

##### Term-by-Term Equation Anatomy: Average Covariance and Within-Block Semivariance Integrals

For two arbitrary spatial supports $v$ and $V$, the block-to-block average covariance integral is:
$$
\underbrace{\overline{C}(v, V)}_{\substack{\text{Average Covariance} \\ \text{between Supports } v \text{ and } V}} = \underbrace{\frac{1}{|v||V|}}_{\substack{\text{Bilinear Volume} \\ \text{Normalization Factor}}} \int_v \int_V \underbrace{C(\mathbf{u}, \mathbf{w})}_{\substack{\text{Point-Support Covariance} \\ \text{between Points } \mathbf{u} \text{ and } \mathbf{w}}} d\mathbf{u} \, d\mathbf{w}
$$
and for within-block average semivariance of block $V$:
$$
\underbrace{\overline{\gamma}(V, V)}_{\substack{\text{Within-Block Average} \\ \text{Semivariance of Support } V}} = \underbrace{\frac{1}{|V|^2}}_{\substack{\text{Quadratic Volume} \\ \text{Normalization Factor}}} \int_V \int_V \underbrace{\gamma(\mathbf{u} - \mathbf{w})}_{\substack{\text{Point Semivariogram} \\ \text{at Internal Lag } \mathbf{u}-\mathbf{w}}} d\mathbf{u} \, d\mathbf{w}
$$
- **$\overline{C}(v, V)$:** The expected cross-covariance between the regularized random variable on support $v$ and that on support $V$: $\overline{C}(v, V) = \operatorname{Cov}(Z_v, Z_V)$. When $v = \{\mathbf{s}_i\}$ represents an infinitesimal point coordinate, this simplifies to the point-to-block covariance $\overline{C}(\mathbf{s}_i, V) = \frac{1}{|V|}\int_V C(\mathbf{s}_i, \mathbf{w}) d\mathbf{w}$.
- **$\frac{1}{|v||V|}$:** The bilinear normalization factor dividing by the product of the Lebesgue volumes. It scales the double volume integral so that $\overline{C}(v, V)$ represents the continuous statistical expectation of covariance over independent uniform spatial draws $\mathbf{U} \sim \operatorname{Uniform}(v)$ and $\mathbf{W} \sim \operatorname{Uniform}(V)$.
- **$\int_v \int_V (\cdot) d\mathbf{u} \, d\mathbf{w}$:** The double Lebesgue integral sweeping coordinate vector $\mathbf{u}$ over all points in compact domain $v$ and coordinate vector $\mathbf{w}$ over all points in compact domain $V$.
- **$C(\mathbf{u}, \mathbf{w})$:** The continuous point-support spatial covariance kernel evaluated between internal points $\mathbf{u} \in v$ and $\mathbf{w} \in V$. Under second-order stationarity, $C(\mathbf{u}, \mathbf{w}) = C(\mathbf{u} - \mathbf{w})$.
- **$\overline{\gamma}(V, V)$:** The within-block dispersion semivariance of support $V$. It represents the expected semivariance between two points chosen at random inside the exact same block volume $V$: $\overline{\gamma}(V, V) = D^2(\mathbf{0} \mid V)$.
- **$\frac{1}{|V|^2}$:** The quadratic normalization factor scaling the double integral over $V \times V$ into an average semivariance.
- **$\gamma(\mathbf{u} - \mathbf{w})$:** The point-support semivariogram evaluated at internal separation lag vector $\mathbf{u} - \mathbf{w}$. As block volume collapses to point support ($|V| \to 0$), $\overline{\gamma}(V, V) \to 0$; as block volume expands infinitely ($|V| \to \infty$), $\overline{\gamma}(V, V)$ approaches the total point sill $C(\mathbf{0})$.

##### Theorem 8.2.1 (Structure of the Regularized Semivariogram)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ be an intrinsically stationary random field with semivariogram $\gamma(\mathbf{h})$. Let $v \subset \mathbb{R}^d$ be a compact support, and let $v_{\mathbf{h}} = v + \mathbf{h}$ denote the translate of $v$ by lag vector $\mathbf{h} \in \mathbb{R}^d$.
1. The covariance function of the regularized field $Z_v(\mathbf{s})$ is:
   $$
   C_v(\mathbf{h}) = \operatorname{Cov}(Z_v(\mathbf{s}), Z_v(\mathbf{s}+\mathbf{h})) = \overline{C}(v, v_{\mathbf{h}})
   $$
2. The regularized semivariogram $\gamma_v(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s}))$ satisfies the identity:
   $$
   \gamma_v(\mathbf{h}) = \overline{\gamma}(v, v_{\mathbf{h}}) - \overline{\gamma}(v, v)
   $$
3. Under second-order stationarity where $C(\mathbf{h}) = C(\mathbf{0}) - \gamma(\mathbf{h})$, the variance of the block variable is:
   $$
   \operatorname{Var}(Z_v(\mathbf{s})) = C_v(\mathbf{0}) = \overline{C}(v, v) = C(\mathbf{0}) - \overline{\gamma}(v, v)
   $$

*Proof.*
1. By Definition 8.1.1 and Fubini's theorem:
   $$
   \begin{aligned}
   C_v(\mathbf{h}) &= \operatorname{Cov}\left(\frac{1}{|v|}\int_v Z(\mathbf{s}+\mathbf{u}) d\mathbf{u}, \, \frac{1}{|v|}\int_v Z(\mathbf{s}+\mathbf{h}+\mathbf{u}') d\mathbf{u}'\right) \\
   &= \frac{1}{|v|^2} \int_v \int_v \operatorname{Cov}(Z(\mathbf{s}+\mathbf{u}), Z(\mathbf{s}+\mathbf{h}+\mathbf{u}')) d\mathbf{u} d\mathbf{u}' \\
   &= \frac{1}{|v|^2} \int_v \int_v C(\mathbf{u} - \mathbf{u}' - \mathbf{h}) d\mathbf{u} d\mathbf{u}'
   \end{aligned}
   $$
   Setting $\mathbf{w} = \mathbf{u}' + \mathbf{h} \in v_{\mathbf{h}}$, with $d\mathbf{u}' = d\mathbf{w}$, we obtain $\frac{1}{|v||v_{\mathbf{h}}|} \int_v \int_{v_{\mathbf{h}}} C(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w} = \overline{C}(v, v_{\mathbf{h}})$.
2. For the semivariogram, write the increment directly:
   $$
   \gamma_v(\mathbf{h}) = \frac{1}{2} \operatorname{E}\left[\left(Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s})\right)^2\right] = \frac{1}{2} \operatorname{E}\left[\left(\frac{1}{|v|}\int_v [Z(\mathbf{s}+\mathbf{h}+\mathbf{u}) - Z(\mathbf{s}+\mathbf{u})] \, d\mathbf{u}\right)^2\right]
   $$
   Expanding the square as a double integral over dummy coordinates $\mathbf{u}, \mathbf{w} \in v$ and dividing by $2|v|^2$:
   $$
   \gamma_v(\mathbf{h}) = \frac{1}{2|v|^2} \int_v \int_v \operatorname{E}\left[\left(Z(\mathbf{s}+\mathbf{h}+\mathbf{u}) - Z(\mathbf{s}+\mathbf{u})\right)\left(Z(\mathbf{s}+\mathbf{h}+\mathbf{w}) - Z(\mathbf{s}+\mathbf{w})\right)\right] \, d\mathbf{u} \, d\mathbf{w}
   $$
   Using the algebraic polarization identity $(X_1 - X_2)(X_3 - X_4) = -\frac{1}{2}(X_1 - X_3)^2 - \frac{1}{2}(X_2 - X_4)^2 + \frac{1}{2}(X_1 - X_4)^2 + \frac{1}{2}(X_2 - X_3)^2$ with $X_1 = Z(\mathbf{s}+\mathbf{h}+\mathbf{u})$, $X_2 = Z(\mathbf{s}+\mathbf{u})$, $X_3 = Z(\mathbf{s}+\mathbf{h}+\mathbf{w})$, $X_4 = Z(\mathbf{s}+\mathbf{w})$:
   $$
   \operatorname{E}[(X_1 - X_2)(X_3 - X_4)] = -\gamma(\mathbf{u} - \mathbf{w}) - \gamma(\mathbf{u} - \mathbf{w}) + \gamma(\mathbf{u} - \mathbf{w} + \mathbf{h}) + \gamma(\mathbf{u} - \mathbf{w} - \mathbf{h})
   $$
   Integrating over $\mathbf{u}, \mathbf{w} \in v$:
   The first two terms each integrate to $-|v|^2 \overline{\gamma}(v, v)$. The third and fourth terms, by symmetry $\gamma(-\mathbf{x}) = \gamma(\mathbf{x})$, each integrate to $|v|^2 \overline{\gamma}(v, v_{\mathbf{h}})$. Dividing by $2|v|^2$:
   $$
   \gamma_v(\mathbf{h}) = \frac{1}{2|v|^2}\left[-2|v|^2 \overline{\gamma}(v, v) + 2|v|^2 \overline{\gamma}(v, v_{\mathbf{h}})\right] = \overline{\gamma}(v, v_{\mathbf{h}}) - \overline{\gamma}(v, v)
   $$
3. Setting $\mathbf{h} = \mathbf{0}$ in the covariance relation:
   $$
   C_v(\mathbf{0}) = \overline{C}(v, v) = \frac{1}{|v|^2}\int_v \int_v [C(\mathbf{0}) - \gamma(\mathbf{u} - \mathbf{u}')] d\mathbf{u} d\mathbf{u}' = C(\mathbf{0}) - \overline{\gamma}(v, v)
   $$
   Since $\gamma(\mathbf{h}) \ge 0$, $\overline{\gamma}(v, v) \ge 0$, proving $\operatorname{Var}(Z_v(\mathbf{s})) \le \operatorname{Var}(Z(\mathbf{s}))$. $\blacksquare$

##### Proposition 8.2.1 (Nugget Filtering and Mean-Square Differentiability)
Let $\{Z(\mathbf{s})\}$ possess a microscale nugget effect. To preserve measure-theoretic consistency under Lebesgue integration, the nugget effect is represented as continuous spatial white noise $\epsilon(\mathbf{s})$ with Dirac delta covariance $\operatorname{Cov}(\epsilon(\mathbf{s}), \epsilon(\mathbf{s}')) = \tau_0^2 \delta(\mathbf{s} - \mathbf{s}')$, where $\tau_0^2 = \tau^2 \cdot a_0^d$ represents the microscale integral capacity ($a_0$ is the microscale correlation length and $\tau^2$ is the point-scale variance):

1. The within-block average covariance filters the spatial white noise inversely with volume:
   $$
   \overline{C}(v, v) = \frac{\tau_0^2}{|v|} + \overline{C}_0(v, v)
   $$
   *Proof.*
   $$
   \frac{1}{|v|^2} \int_v \int_v \tau_0^2 \delta(\mathbf{u} - \mathbf{u}') \, d\mathbf{u} \, d\mathbf{u}' = \frac{\tau_0^2}{|v|^2} \int_v 1 \, d\mathbf{u} = \frac{\tau_0^2}{|v|}
   $$
   The nugget variance within a block of support volume $|v|$ shrinks at rate $O(|v|^{-1})$, reflecting the classical law of large numbers for uncorrelated spatial increments.
2. The cross-covariance between disjoint blocks $v_i \cap v_j = \emptyset$ filters the nugget effect entirely:
   $$
   \overline{C}(v_i, v_j) = \overline{C}_0(v_i, v_j) \quad \text{for } v_i \cap v_j = \emptyset
   $$
   Because the Dirac delta $\delta(\mathbf{u} - \mathbf{u}') = 0$ for all $\mathbf{u} \in v_i$ and $\mathbf{u}' \in v_j$, the nugget effect contributes strictly zero to cross-block covariances.
3. For any support of strictly positive measure $|v| > 0$, the discontinuity at the origin is completely eliminated:
   $$
   \lim_{\|\mathbf{h}\| \to 0} \gamma_v(\mathbf{h}) = 0 = \gamma_v(\mathbf{0})
   $$
4. The regularized field $Z_v(\mathbf{s})$ is mean-square differentiable in all directions, even if the point field $Z(\mathbf{s})$ is non-differentiable.

*Proof.*
The geometric covariogram of $v$ is $K_v(\mathbf{x}) = \operatorname{Leb}(v \cap (v - \mathbf{x})) = \int_{\mathbb{R}^d} \mathbb{I}_v(\mathbf{u}) \mathbb{I}_v(\mathbf{u} + \mathbf{x}) d\mathbf{u}$. 
The regularized covariance is the convolution $C_v(\mathbf{h}) = \frac{1}{|v|^2} (K_v * C)(\mathbf{h})$. Because $K_v$ is Lipschitz continuous and has compact support, convolution with any locally integrable kernel $C$ produces a continuously differentiable function. Hence $\lim_{\|\mathbf{h}\| \to 0} \gamma_v(\mathbf{h}) = 0$. $\blacksquare$

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

```
Comparison of Point vs. Block Semivariogram:
Semivariance \gamma(h)
  ^
  |        Point Semivariogram \gamma(h)
  |       /----------------------------- Total Sill C(0) = \sigma^2 + \tau^2
  |      /
  |     /  Block Semivariogram \gamma_v(h)
  |    /  /----------------------------- Block Sill \sigma_v^2 = C(0) - \bar{\gamma}(v, v)
  |   /  /
\tau^2+ /
  |  / /
  | / /
  |/ /
  +-------------------------------------> Lag h = ||h||
  0
  (Block semivariogram has ZERO nugget effect and parabolic behavior at origin)
```

- **Why the Nugget Disappears:** The nugget effect $\tau^2$ represents spatially uncorrelated microscale noise ($\operatorname{Cov}(\epsilon(\mathbf{s}), \epsilon(\mathbf{s}+\mathbf{h})) = 0$ for $\mathbf{h} \ne \mathbf{0}$). In an integral $\int_v \epsilon(\mathbf{u}) d\mathbf{u}$, uncorrelated points cancel each other out by the continuous Law of Large Numbers. The nugget variance decays as $\tau^2 / |v|$, smoothing out sharp measurement chatter.
- **Curvature at the Origin:** Point semivariograms often exhibit linear behavior near the origin ($\gamma(h) \sim c_1 h$), corresponding to rough, non-differentiable sample paths (such as the Exponential model). In contrast, regularized semivariograms always leave the origin parabolically ($\gamma_v(h) \sim c_2 h^2$), reflecting the smooth, continuous sample paths produced by volumetric averaging.

---

### 8.3 Dispersion Variance and Krige's Additivity Theorem

#### Beat 1: The Mathematical Anchor (The "Why")

How does the variability of small blocks within a mining deposit relate to the overall deposit variability? Danie G. Krige (1951) discovered empirically that variances over nested geometric supports obey a strict additive law. Georges Matheron (1962) provided the theoretical proof through dispersion variance.

##### Definition 8.3.1 (Dispersion Variance)
Let $V \subset \mathbb{R}^d$ be a bounded domain of positive volume $|V| > 0$, and let $v \subset V$ denote a smaller block support ($0 < |v| \le |V|$).
1. The **dispersion variance of points within $V$**, denoted $D^2(\mathbf{0} \mid V)$, is the spatial variance of points randomly located within $V$:
   $$
   D^2(\mathbf{0} \mid V) = \frac{1}{|V|} \int_V \operatorname{Var}(Z(\mathbf{s}) - Z_V) d\mathbf{s} = \overline{\gamma}(V, V)
   $$
2. The **dispersion variance of blocks $v$ within $V$**, denoted $D^2(v \mid V)$, is the spatial variance of block averages $Z_v(\mathbf{s})$ as $v$ ranges over all possible positions inside $V$:
   $$
   D^2(v \mid V) = \overline{\gamma}(V, V) - \overline{\gamma}(v, v)
   $$
3. In covariance notation under second-order stationarity:
   $$
   D^2(v \mid V) = \overline{C}(v, v) - \overline{C}(V, V)
   $$

##### Theorem 8.3.1 (Krige's Additivity Theorem)
Let $v, V, W \subset \mathbb{R}^d$ be three nested spatial supports:
$$
v \subset V \subset W \quad \text{with } 0 < |v| \le |V| \le |W|
$$
Then the dispersion variance of small blocks $v$ within the large domain $W$ decomposes additively into the dispersion variance of $v$ within intermediate blocks $V$ plus the dispersion variance of $V$ within $W$:
$$
D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)
$$

*Proof.*
We present two complementary proofs: an algebraic proof and a measure-theoretic proof.

**Algebraic Proof:**
By Definition 8.3.1, express each dispersion variance as the difference of within-support average semivariograms:
$$
\begin{aligned}
D^2(v \mid W) &= \overline{\gamma}(W, W) - \overline{\gamma}(v, v) \\
D^2(v \mid V) &= \overline{\gamma}(V, V) - \overline{\gamma}(v, v) \\
D^2(V \mid W) &= \overline{\gamma}(W, W) - \overline{\gamma}(V, V)
\end{aligned}
$$
Summing $D^2(v \mid V)$ and $D^2(V \mid W)$:
$$
\begin{aligned}
D^2(v \mid V) + D^2(V \mid W) &= [\overline{\gamma}(V, V) - \overline{\gamma}(v, v)] + [\overline{\gamma}(W, W) - \overline{\gamma}(V, V)] \\
&= \overline{\gamma}(W, W) - \overline{\gamma}(v, v) + [\overline{\gamma}(V, V) - \overline{\gamma}(V, V)] \\
&= \overline{\gamma}(W, W) - \overline{\gamma}(v, v) = D^2(v \mid W)
\end{aligned}
$$
The intermediate term $\overline{\gamma}(V, V)$ cancels identically.

**Probabilistic Proof via Total Variance:**
Partition the domain $W$ into $N$ non-overlapping intermediate blocks $\{V_k\}_{k=1}^N$ congruent to $V$, so $|W| = N|V|$. Further partition each $V_k$ into $M$ non-overlapping small blocks $\{v_{k,j}\}_{j=1}^M$ congruent to $v$, so $|V| = M|v|$.
Select a small block $v$ uniformly at random from $W$. This defines a hierarchical random process:
- Stage 1: Select block $K \in \{1, \dots, N\}$ with $P(K = k) = 1/N$.
- Stage 2: Given $K = k$, select sub-block $J \in \{1, \dots, M\}$ with $P(J = j \mid K = k) = 1/M$.

Let $Z_v = Z(v_{K, J})$ and $Z_V = Z(V_K) = \frac{1}{M}\sum_{j=1}^M Z(v_{K, j})$. By the Law of Total Variance:
$$
\operatorname{Var}(Z_v) = \operatorname{E}\left[\operatorname{Var}(Z_v \mid K)\right] + \operatorname{Var}\left(\operatorname{E}[Z_v \mid K]\right)
$$
1. The left-hand side $\operatorname{Var}(Z_v)$ is the dispersion variance of small blocks $v$ within $W$: $\operatorname{Var}(Z_v) = D^2(v \mid W)$.
2. The conditional variance $\operatorname{Var}(Z_v \mid K = k)$ is the dispersion variance of blocks $v$ inside block $V_k$. Taking the expectation over $K$:
   $$
   \operatorname{E}[\operatorname{Var}(Z_v \mid K)] = \frac{1}{N}\sum_{k=1}^N D^2(v \mid V_k) = D^2(v \mid V)
   $$
3. The conditional expectation is $\operatorname{E}[Z_v \mid K = k] = Z(V_k)$. Its variance across all blocks in $W$ is:
   $$
   \operatorname{Var}(\operatorname{E}[Z_v \mid K]) = \operatorname{Var}(Z_V) = D^2(V \mid W)
   $$
Substituting these results yields $D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)$. $\blacksquare$

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **The ANOVA Analogy:** Krige's Additivity Theorem is the spatial continuous analog of the Analysis of Variance (ANOVA) identity:
  $$
  \text{Total Sum of Squares} = \text{Within-Group Sum of Squares} + \text{Between-Group Sum of Squares}
  $$
  $D^2(v \mid W)$ represents total variation across the deposit; $D^2(v \mid V)$ is the variability *within* extraction panels; $D^2(V \mid W)$ is the variability *between* panel averages.
- **The Limit of Point Support:** Setting $v = \mathbf{0}$ yields:
  $$
  D^2(\mathbf{0} \mid W) = D^2(\mathbf{0} \mid V) + D^2(V \mid W)
  $$
  This demonstrates that the total point variance within a basin $W$ is partitioned into the local microscale variance inside block $V$ plus the macro-scale variance between blocks.

---

### 8.4 Cartier's Relation and the Spatial Martingale Property

#### Beat 1: The Mathematical Anchor (The "Why")

When aggregating spatial measurements into larger blocks, what happens to our expectation of the fine-scale process? Cartier's relation shows that spatial regularization acts as an orthogonal conditional expectation operator.

##### Theorem 8.4.1 (Cartier's Relation)
Let $V \subset \mathbb{R}^d$ be a compact domain, and let $S$ be a random coordinate distributed uniformly in $V$ with probability measure $P_V(d\mathbf{s}) = \frac{d\mathbf{s}}{|V|}$. Let $Z_V = \frac{1}{|V|}\int_V Z(\mathbf{u}) d\mathbf{u}$.
1. For the point variable $Z(S)$:
   $$
   \operatorname{E}[Z(S) \mid Z_V] = Z_V \quad P\text{-a.s.}
   $$
2. Let $\mathcal{P}_1$ and $\mathcal{P}_2$ be two nested partitions of a domain $D$ such that $\mathcal{P}_2$ is coarser than $\mathcal{P}_1$ (every block $V \in \mathcal{P}_2$ is a union of disjoint sub-blocks $v \in \mathcal{P}_1$). Let $\mathcal{F}_{\mathcal{P}_2} \subset \mathcal{F}_{\mathcal{P}_1}$ denote the generated $\sigma$-algebras. Then:
   $$
   \operatorname{E}[Z_{\mathcal{P}_1} \mid \mathcal{F}_{\mathcal{P}_2}] = Z_{\mathcal{P}_2}
   $$
   The sequence of regularized fields forms a **spatial reverse martingale** under successive coarsening of support.

*Proof.*
1. To verify the conditional expectation, let $g: \mathbb{R} \to \mathbb{R}$ be any bounded Borel test function. Integrating over the uniform spatial distribution $P_V$:
   $$
   \int_V \operatorname{E}[Z(\mathbf{s}) g(Z_V)] \frac{d\mathbf{s}}{|V|} = \operatorname{E}\left[\left(\frac{1}{|V|}\int_V Z(\mathbf{s}) d\mathbf{s}\right) g(Z_V)\right] = \operatorname{E}[Z_V g(Z_V)]
   $$
   Since $Z_V$ is measurable with respect to $\sigma(Z_V)$, this satisfies the Kolmogorov definition of conditional expectation: $\operatorname{E}[Z(S) \mid Z_V] = Z_V$.
2. For any $V \in \mathcal{P}_2$ with $V = \bigcup_{j=1}^m v_j$ ($v_j \in \mathcal{P}_1$ disjoint):
   $$
   Z_V = \frac{1}{|V|}\int_V Z(\mathbf{u}) d\mathbf{u} = \sum_{j=1}^m \frac{|v_j|}{|V|} Z_{v_j}
   $$
   Conditioning on $\mathcal{F}_{\mathcal{P}_2}$:
   $$
   \operatorname{E}\left[\sum_{j=1}^m \frac{|v_j|}{|V|} Z_{v_j} \, \Bigg| \, \mathcal{F}_{\mathcal{P}_2}\right] = Z_V
   $$
   This establishes the martingale property. $\blacksquare$

##### Corollary 8.4.1 (Variance Monotonicity across Supports)
For nested supports $v \subset V$:
$$
\operatorname{Var}(Z_V) \le \operatorname{Var}(Z_v) \le \operatorname{Var}(Z(\mathbf{s}))
$$
with equality if and only if the field is degenerate (spatially constant $P$-a.s.).

*Proof.*
By Cartier's relation, $Z_V = \operatorname{E}[Z_v \mid Z_V]$. By the conditional variance formula:
$$
\operatorname{Var}(Z_v) = \operatorname{Var}(\operatorname{E}[Z_v \mid Z_V]) + \operatorname{E}[\operatorname{Var}(Z_v \mid Z_V)] = \operatorname{Var}(Z_V) + \operatorname{E}[\operatorname{Var}(Z_v \mid Z_V)]
$$
Because conditional variance is non-negative, $\operatorname{Var}(Z_V) \le \operatorname{Var}(Z_v)$. $\blacksquare$

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

Cartier's relation guarantees **mass conservation**: if an ore block has an average grade of $2.5 \text{ g/t}$, the expected grade of any random drill core drilled inside that block is exactly $2.5 \text{ g/t}$. Averaging never creates or destroys total mass. However, conditional variance $\operatorname{Var}(Z_v \mid Z_V) > 0$ means individual core samples will scatter widely around $2.5 \text{ g/t}$.

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *What is Cartier's relation and why does it prevent bias when predicting non-linear mining recovery on block supports?*  
> Cartier's relation establishes that the expected value of a fine-support point or sample $Z(v)$ conditioned on the surrounding block average $Z(V)$ is identically equal to the block average:
> $$
> \operatorname{E}[Z(v) \mid Z(V)] = Z(V) \quad P\text{-a.s.}
> $$
> In probability theory, this states that the sequence of spatial regularizations under coarsening supports forms a **spatial reverse martingale**.
> 
> Why is this property vital in mining and environmental economics? Mining selectivity is strictly non-linear: an operator extracts block $V$ if and only if its estimated grade exceeds an economic cut-off $z_c$, capturing recovered tonnage $T(z_c) = \mathbb{I}\{Z(V) \ge z_c\}$.
> If an analyst naively uses point-support sample data $Z(v)$ to forecast block recovery without enforcing Cartier's conditional expectation:
> 1. Point samples have much wider variance than blocks ($\sigma_v^2 \gg \sigma_V^2$).
> 2. The high-grade tail of point samples $P(Z(v) \ge z_c)$ is substantially fatter than that of blocks $P(Z(V) \ge z_c)$ for high cut-offs $z_c > \mu$.
> 3. Naively evaluating block recovery from point distributions predicts non-existent high-grade mining blocks, leading to massive over-estimation of recoverable metal and severe financial failure.
> 
> Cartier's relation guarantees that change-of-support models (such as the Discrete Gaussian Model or Indirect Lognormal Correction) satisfy the condition $\operatorname{E}[Z(v) \mid Z(V)] = Z(V)$, preserving total mass and preventing biased over-estimation of recoverable tonnage above economic cut-offs.

---

### 8.5 Distributional Transformations under Change of Support

#### Beat 1: The Mathematical Anchor (The "Why")

As support volume increases, how does the entire probability density function $f_v(z)$ evolve? It undergoes three concurrent transformations:
1. **Variance reduction:** $\sigma_v^2 = \overline{C}(v, v) < \sigma_0^2$.
2. **Central Limit Gaussianization:** As $|v| \to \infty$, $Z_v$ converges in distribution to normal.
3. **Skewness attenuation:** High positive skewness is rapidly suppressed.

##### Theorem 8.5.1 (Spatial Central Limit Theorem for Block Regularization)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ be a strictly stationary, centered random field with finite variance $\sigma^2 = C(\mathbf{0}) < \infty$. Assume $\{Z(\mathbf{s})\}$ satisfies the $\alpha$-mixing condition with mixing coefficients satisfying $\int_0^\infty r^{d-1}[\alpha(r)]^{\frac{\delta}{2+\delta}} dr < \infty$ for some $\delta > 0$, and $\operatorname{E}[|Z(\mathbf{s})|^{2+\delta}] < \infty$.
Let $v_n = n v_0 = \{n \mathbf{x} : \mathbf{x} \in v_0\}$ be homothetically expanding compact domains ($n \to \infty$), where $v_0$ has non-empty interior. Let:
$$
\sigma_\infty^2 = \int_{\mathbb{R}^d} C(\mathbf{h}) d\mathbf{h} \in (0, \infty)
$$
Then as $n \to \infty$:
$$
\sqrt{|v_n|} \, Z_{v_n}(\mathbf{s}) = \frac{1}{\sqrt{|v_n|}} \int_{v_n(\mathbf{s})} Z(\mathbf{u}) d\mathbf{u} \xrightarrow{d} \mathcal{N}\left(0, \, \sigma_\infty^2\right)
$$
and the standardized block variable satisfies:
$$
\frac{Z_{v_n}(\mathbf{s})}{\sqrt{\operatorname{Var}(Z_{v_n}(\mathbf{s}))}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

*Proof.*
The unnormalized integral variance is:
$$
\operatorname{Var}\left(\int_{v_n} Z(\mathbf{u}) d\mathbf{u}\right) = \int_{v_n} \int_{v_n} C(\mathbf{u} - \mathbf{u}') d\mathbf{u} d\mathbf{u}' = \int_{\mathbb{R}^d} K_{v_n}(\mathbf{h}) C(\mathbf{h}) d\mathbf{h}
$$
Dividing by $|v_n|$:
$$
\frac{1}{|v_n|} \operatorname{Var}\left(\int_{v_n} Z(\mathbf{u}) d\mathbf{u}\right) = \int_{\mathbb{R}^d} \frac{K_{v_n}(\mathbf{h})}{|v_n|} C(\mathbf{h}) d\mathbf{h}
$$
For every fixed $\mathbf{h}$, $\lim_{n \to \infty} \frac{K_{v_n}(\mathbf{h})}{|v_n|} = 1$. By dominated convergence under the mixing condition, the limit is $\int_{\mathbb{R}^d} C(\mathbf{h}) d\mathbf{h} = \sigma_\infty^2$. Asymptotic normality follows from Bernstein's spatial blocking technique. $\blacksquare$

##### Proposition 8.5.1 (Skewness Attenuation)
Let $\gamma_1(0) = \operatorname{E}[(Z(\mathbf{s}) - \mu)^3] / \sigma_0^3$ be the point-support skewness. For regularized blocks of volume $|v|$:
$$
\gamma_1(v) = \frac{\operatorname{E}[(Z_v - \mu)^3]}{\sigma_v^3} = \mathcal{O}\left(|v|^{-1/2}\right) \quad \text{as } |v| \to \infty
$$
Skewness decays inversely with the square root of support volume.

---

### 8.6 Selectivity Curves and Cut-off Economics

#### Beat 1: The Mathematical Anchor (The "Why")

In resource evaluation, an engineer must decide whether each parcel of rock or soil should be processed or discarded based on a threshold cut-off $z_c$. The economic yield depends critically on support volume.

```
Point Support Distribution (High Variance, Long Tails)
Probability Density
  ^
  |        /\
  |       /  \
  |      /    \
--+-----+------+-------\--------+-----> Grade z
 0     z_c    Mean   z*       Extreme Highs
        <---- Tonnage ---->

Block Support Distribution (Low Variance, Compact)
Probability Density
  ^
  |          /\
  |         /  \
  |        /    \
--+-------+--/\--+--------------> Grade z
 0       z_c Mean z*
```

##### Definition 8.6.1 (Selectivity Functions)
Let $F_v(z) = P(Z_v \le z)$ be the CDF of regularized variable $Z_v$, and let $z_c \in \mathbb{R}$ be a cut-off threshold.
1. The **recovered tonnage fraction** $T_v(z_c)$ is:
   $$
   T_v(z_c) = P(Z_v \ge z_c) = 1 - F_v(z_c) = \int_{z_c}^\infty dF_v(z)
   $$
2. The **quantity of metal** $Q_v(z_c)$ is the truncated first moment:
   $$
   Q_v(z_c) = \operatorname{E}[Z_v \cdot \mathbb{I}\{Z_v \ge z_c\}] = \int_{z_c}^\infty z \, dF_v(z)
   $$
3. The **mean recovered grade** $m_v(z_c)$ is:
   $$
   m_v(z_c) = \operatorname{E}[Z_v \mid Z_v \ge z_c] = \frac{Q_v(z_c)}{T_v(z_c)}
   $$

##### Theorem 8.6.1 (The Support Effect on Selectivity)
Let $v \subset V$ be two supports with identical mean $\operatorname{E}[Z_v] = \operatorname{E}[Z_V] = m$, but ordered variances $\sigma_V^2 < \sigma_v^2$. If the distributions cross at a unique point $z^* \approx m$:
1. For high cut-offs $z_c > z^*$:
   $$
   T_V(z_c) < T_v(z_c) \quad \text{and} \quad Q_V(z_c) < Q_v(z_c)
   $$
2. For low cut-offs $z_c < z^*$:
   $$
   T_V(z_c) > T_v(z_c)
   $$
3. For all $z_c > m$, the mean recovered grade is strictly decreasing in support volume:
   $$
   m_V(z_c) < m_v(z_c)
   $$

*Proof.*
Integrating $Q(z_c)$ by parts:
$$
Q(z_c) = \int_{z_c}^\infty z \, dF(z) = z_c [1 - F(z_c)] + \int_{z_c}^\infty [1 - F(z)] dz = z_c T(z_c) + \int_{z_c}^\infty T(z) dz
$$
Because $Z_V$ has smaller variance than $Z_v$ while sharing the same mean $m$, $F_V$ is a mean-preserving contraction of $F_v$ (second-order stochastic dominance). For $z_c > z^*$, $F_V(z_c) > F_v(z_c)$, so $T_V(z_c) < T_v(z_c)$ and $Q_V(z_c) < Q_v(z_c)$. 

Differentiating $m(z_c) = Q(z_c)/T(z_c)$ with respect to $z_c$ yields $\frac{d m}{d z_c} = \frac{f(z_c)}{T(z_c)}[m(z_c) - z_c] > 0$. Because probability mass on $V$ is concentrated tightly near $m$, the tail conditional expectation for $z_c > m$ is pulled down closer to $m$, proving $m_V(z_c) < m_v(z_c)$. $\blacksquare$

---

### 8.7 Analytical Change-of-Support Models

#### 8.7.1 The Affine Correction Model

The affine correction assumes the standardized distribution shape is invariant under change of support, contracting linearly toward the mean.

##### Definition 8.7.1 (Affine Correction)
Let $Z$ be point support with mean $m$ and variance $\sigma_0^2$. The **affine correction** models the block variable $Z_v$ as:
$$
Z_v = r(Z - m) + m
$$
where $r \in (0, 1]$ is the variance reduction factor:
$$
r = \frac{\sigma_v}{\sigma_0} = \sqrt{\frac{\overline{C}(v, v)}{C(\mathbf{0})}} = \sqrt{1 - \frac{\overline{\gamma}(v, v)}{\sigma_0^2}}
$$
Quantiles transform according to $z_{v, p} = r(z_p - m) + m$, and tonnage satisfies $T_v(z_c) = T_0\left(m + \frac{z_c - m}{r}\right)$.

> [!WARNING]
> The affine model preserves skewness identically ($\gamma_1(v) = \gamma_1(0)$) and can predict impossible negative grades ($z_{v, \min} = r(z_{\min} - m) + m < 0$) when $r < 1 - \frac{m}{m - z_{\min}}$.

---

#### 8.7.2 The Indirect Lognormal Correction Model

When the point distribution is skewed and strictly positive, the lognormal distribution provides a much more realistic model.

##### Theorem 8.7.2 (Indirect Lognormal Correction Equations)
Let point variable $Z = \exp(Y)$ with $Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$, mean $m = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2)$, and variance $\sigma_0^2 = m^2(\exp(\sigma_Y^2) - 1)$.
Assume the block variable $Z_v = \exp(Y_v)$ is also lognormal with $Y_v \sim \mathcal{N}(\mu_{Y_v}, \sigma_{Y_v}^2)$. Requiring $\operatorname{E}[Z_v] = m$ and $\operatorname{Var}(Z_v) = \sigma_v^2 = \overline{C}(v, v)$:
1. The logarithmic variance of the block support is:
   $$
   \sigma_{Y_v}^2 = \ln\left(1 + \frac{\sigma_v^2}{m^2}\right) = \ln\left(1 + \frac{\overline{C}(v, v)}{m^2}\right)
   $$
2. The logarithmic mean of the block support is:
   $$
   \mu_{Y_v} = \ln(m) - \frac{1}{2}\sigma_{Y_v}^2 = \ln(m) - \frac{1}{2}\ln\left(1 + \frac{\sigma_v^2}{m^2}\right)
   $$
3. Recovered tonnage at cut-off $z_c > 0$ is:
   $$
   T_v(z_c) = 1 - \Phi\left(\frac{\ln(z_c) - \mu_{Y_v}}{\sigma_{Y_v}}\right) = \Phi\left(\frac{\mu_{Y_v} - \ln(z_c)}{\sigma_{Y_v}}\right)
   $$
4. Recovered metal quantity is:
   $$
   Q_v(z_c) = m \cdot \Phi\left(\frac{\mu_{Y_v} + \sigma_{Y_v}^2 - \ln(z_c)}{\sigma_{Y_v}}\right)
   $$
   where $\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-t^2/2} dt$.

*Proof.*
1. For any lognormal variable $W = \exp(X)$ with $X \sim \mathcal{N}(\mu_X, \sigma_X^2)$, $\operatorname{Var}(W) = (\operatorname{E}[W])^2[\exp(\sigma_X^2) - 1]$. Applying this to $Z_v$:
   $$
   \operatorname{Var}(Z_v) = m^2 [\exp(\sigma_{Y_v}^2) - 1] \implies \exp(\sigma_{Y_v}^2) - 1 = \frac{\sigma_v^2}{m^2} \implies \sigma_{Y_v}^2 = \ln\left(1 + \frac{\sigma_v^2}{m^2}\right)
   $$
2. Since $\operatorname{E}[Z_v] = \exp(\mu_{Y_v} + \frac{1}{2}\sigma_{Y_v}^2) = m$, taking natural logs gives $\mu_{Y_v} = \ln(m) - \frac{1}{2}\sigma_{Y_v}^2$.
3. $T_v(z_c) = P(\exp(Y_v) \ge z_c) = P\left(\frac{Y_v - \mu_{Y_v}}{\sigma_{Y_v}} \ge \frac{\ln z_c - \mu_{Y_v}}{\sigma_{Y_v}}\right) = \Phi\left(\frac{\mu_{Y_v} - \ln z_c}{\sigma_{Y_v}}\right)$.
4. Evaluating $Q_v(z_c) = \int_{\ln z_c}^\infty e^y \frac{1}{\sqrt{2\pi}\sigma_{Y_v}} \exp\left(-\frac{(y - \mu_{Y_v})^2}{2\sigma_{Y_v}^2}\right) dy$: completing the square in the exponent factors out $\exp(\mu_{Y_v} + \frac{1}{2}\sigma_{Y_v}^2) = m$, leaving $m \Phi\left(\frac{\mu_{Y_v} + \sigma_{Y_v}^2 - \ln z_c}{\sigma_{Y_v}}\right)$. $\blacksquare$

---

### 8.8 Block Kriging

#### Beat 1: The Mathematical Anchor (The "Why")

How do we predict the volume average $Z_v(\mathbf{s}_0) = \frac{1}{|v|}\int_{v(\mathbf{s}_0)} Z(\mathbf{u}) d\mathbf{u}$ from $n$ discrete point measurements $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$?

##### Theorem 8.8.1 (Ordinary Block Kriging System)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an intrinsically stationary random field with semivariogram $\gamma(\mathbf{h})$. Given observations $\mathbf{Z}$, consider the linear predictor:
$$
\widehat{Z}_v(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = \boldsymbol{\lambda}'\mathbf{Z}
$$
Under the unbiasedness condition $\sum_{i=1}^n \lambda_i = 1$:
1. The kriging weight vector $\boldsymbol{\lambda} \in \mathbb{R}^n$ and Lagrange multiplier $\mu_{\text{OK}}$ satisfy:
   $$
   \begin{pmatrix} \mathbf{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix}
   \begin{pmatrix} \boldsymbol{\lambda} \\ \mu_{\text{OK}} \end{pmatrix}
   =
   \begin{pmatrix} \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) \\ 1 \end{pmatrix}
   $$
   where $\mathbf{\Gamma}_{ij} = \gamma(\mathbf{s}_i - \mathbf{s}_j)$, $\mathbf{1} = (1, \dots, 1)'$, and $\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)$ is the vector of point-to-block average semivariances:
   $$
   [\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)]_i = \overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0)) = \frac{1}{|v|} \int_{v(\mathbf{s}_0)} \gamma(\mathbf{s}_i - \mathbf{u}) d\mathbf{u}, \quad i = 1, \dots, n
   $$
2. The Ordinary Block Kriging error variance is:
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

##### Term-by-Term Equation Anatomy: Ordinary Block Kriging System and Prediction Variance
$$
\underbrace{\begin{pmatrix} \mathbf{\Gamma} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{pmatrix}}_{\substack{\text{Augmented Sample-to-Sample} \\ \text{Semivariogram Matrix } (n+1) \times (n+1)}}
\underbrace{\begin{pmatrix} \boldsymbol{\lambda} \\ \mu_{\text{OK}} \end{pmatrix}}_{\substack{\text{Block Kriging Weights} \\ \text{\& Lagrange Multiplier}}}
=
\underbrace{\begin{pmatrix} \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) \\ 1 \end{pmatrix}}_{\substack{\text{Point-to-Block Semivariance Vector} \\ \text{\& Unbiasedness Constant}}}
$$
and its minimized prediction error variance:
$$
\sigma_{\text{OK}, v}^2(\mathbf{s}_0) = \underbrace{\boldsymbol{\lambda}' \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)}_{\substack{\text{Weighted Point-to-Block} \\ \text{Semivariance}}} + \underbrace{\mu_{\text{OK}}}_{\substack{\text{Unbiasedness} \\ \text{Penalty}}} - \underbrace{\overline{\gamma}(v, v)}_{\substack{\text{Within-Block Dispersion Variance} \\ \text{(Variance Reduction of Support)}}}
$$
- **$\mathbf{\Gamma} \in \mathbb{R}^{n \times n}$:** The symmetric matrix of point-to-point semivariances between sample locations: $\mathbf{\Gamma}_{ij} = \gamma(\mathbf{s}_i - \mathbf{s}_j)$. Notice that this entire $n \times n$ matrix is purely a function of observation coordinates and is completely independent of the size, orientation, or geometry of the target block $v$.
- **$\mathbf{1}_n$ and $\mathbf{1}_n'$:** Column and row vectors of ones enforcing the Ordinary Kriging unbiasedness constraint $\sum_{i=1}^n \lambda_i = 1$, which filters out the unknown stationary spatial mean $\mu$.
- **$\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)' \in \mathbb{R}^n$:** The optimal linear weights assigned to discrete point samples $Z(\mathbf{s}_i)$ to predict the volume average $Z_v(\mathbf{s}_0)$.
- **$\mu_{\text{OK}} \in \mathbb{R}$:** The scalar Lagrange multiplier associated with the unbiasedness constraint $\mathbf{1}'\boldsymbol{\lambda} = 1$.
- **$\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) \in \mathbb{R}^n$:** The vector of point-to-block average semivariances between sample points $\mathbf{s}_i$ and the entire target volume $v(\mathbf{s}_0)$: $[\overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)]_i = \frac{1}{|v|}\int_{v(\mathbf{s}_0)} \gamma(\mathbf{s}_i - \mathbf{u}) d\mathbf{u}$. All geometric properties of the target block (support size, aspect ratio, orientation) enter the kriging system exclusively through this right-hand side vector.
- **$\boldsymbol{\lambda}' \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) + \mu_{\text{OK}}$:** The raw linear interpolation uncertainty term, identical in algebraic structure to point kriging.
- **Decoding $-\overline{\gamma}(v, v)$ (The Subtracted Within-Block Variance):**
  Why is $\overline{\gamma}(v, v)$ *subtracted* from the block kriging variance?
  In point kriging, the target is an infinitesimal point $Z(\mathbf{s}_0)$, which fluctuates with full point-support variance $\operatorname{Var}(Z(\mathbf{s}_0)) = C(\mathbf{0})$.
  In block kriging, the target is a spatial volume average $Z_v(\mathbf{s}_0) = \frac{1}{|v|}\int_{v(\mathbf{s}_0)} Z(\mathbf{u}) d\mathbf{u}$. Its inherent physical variance is diminished by internal spatial averaging:
  $$
  \operatorname{Var}(Z_v(\mathbf{s}_0)) = C(\mathbf{0}) - \overline{\gamma}(v, v)
  $$
  Because the target block variable has inherently smaller variance than a point variable, predicting the block is fundamentally *less uncertain* than predicting an isolated point coordinate. The subtrahend $-\overline{\gamma}(v, v) = -D^2(\mathbf{0} \mid v)$ explicitly deducts the internal microscale fluctuations that are averaged away by the physical support volume. If an analyst mistakenly omits $-\overline{\gamma}(v, v)$, they severely overestimate block prediction uncertainty.

*Proof.*
The estimation error is $\varepsilon_v = \widehat{Z}_v(\mathbf{s}_0) - Z_v(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) - \frac{1}{|v|}\int_{v(\mathbf{s}_0)} Z(\mathbf{u}) d\mathbf{u}$.
Unbiasedness requires $\operatorname{E}[\varepsilon_v] = \mu(\sum_{i=1}^n \lambda_i - 1) = 0 \iff \sum_{i=1}^n \lambda_i = 1$.
Squaring $\varepsilon_v$ and taking expectations:
$$
\operatorname{E}[\varepsilon_v^2] = \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) - 2\sum_{i=1}^n \lambda_i \operatorname{Cov}(Z(\mathbf{s}_i), Z_v(\mathbf{s}_0)) + \operatorname{Var}(Z_v(\mathbf{s}_0))
$$
Substituting $\operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = C(\mathbf{0}) - \gamma(\mathbf{s}_i - \mathbf{s}_j)$, $\operatorname{Cov}(Z(\mathbf{s}_i), Z_v(\mathbf{s}_0)) = C(\mathbf{0}) - \overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0))$, and $\operatorname{Var}(Z_v(\mathbf{s}_0)) = C(\mathbf{0}) - \overline{\gamma}(v, v)$, and using $\sum \lambda_i = 1$:
$$
\sigma_E^2(\boldsymbol{\lambda}) = -\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + 2\sum_{i=1}^n \lambda_i \overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0)) - \overline{\gamma}(v, v)
$$
Form the Lagrangian with multiplier $2\mu_{\text{OK}}$:
$$
L(\boldsymbol{\lambda}, \mu_{\text{OK}}) = \sigma_E^2(\boldsymbol{\lambda}) + 2\mu_{\text{OK}}\left(\sum_{i=1}^n \lambda_i - 1\right)
$$
Differentiating with respect to $\lambda_i$ and setting to zero yields $\sum_{j=1}^n \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) + \mu_{\text{OK}} = \overline{\gamma}(\mathbf{s}_i, v(\mathbf{s}_0))$. Combining with $\sum \lambda_i = 1$ yields the system. Multiplying the $i$-th equation by $\lambda_i$, summing over $i$, and substituting into $\sigma_E^2$ produces $\sigma_{\text{OK}, v}^2(\mathbf{s}_0) = \boldsymbol{\lambda}' \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) + \mu_{\text{OK}} - \overline{\gamma}(v, v)$. $\blacksquare$

##### Corollary 8.8.1 (Equivalence with Average Point Kriging)
Let $\widehat{Z}(\mathbf{u})$ be the Ordinary Point Kriging predictor at $\mathbf{u} \in v(\mathbf{s}_0)$ based on $\mathbf{Z}$. Then:
$$
\frac{1}{|v|} \int_{v(\mathbf{s}_0)} \widehat{Z}(\mathbf{u}) d\mathbf{u} = \widehat{Z}_v(\mathbf{s}_0)
$$
However, the average of point kriging estimation variances strictly overestimates the block kriging variance by the within-block dispersion variance:
$$
\frac{1}{|v|} \int_{v(\mathbf{s}_0)} \sigma_{\text{OK}}^2(\mathbf{u}) d\mathbf{u} = \sigma_{\text{OK}, v}^2(\mathbf{s}_0) + D^2(\mathbf{0} \mid v) > \sigma_{\text{OK}, v}^2(\mathbf{s}_0)
$$

*Proof.*
Point kriging weights satisfy $\mathbf{\Gamma}\boldsymbol{\lambda}(\mathbf{u}) + \mu(\mathbf{u})\mathbf{1} = \boldsymbol{\gamma}(\mathbf{u})$. Integrating over $\mathbf{u} \in v(\mathbf{s}_0)$ and dividing by $|v|$:
$$
\mathbf{\Gamma}\left(\frac{1}{|v|}\int_{v(\mathbf{s}_0)} \boldsymbol{\lambda}(\mathbf{u}) d\mathbf{u}\right) + \left(\frac{1}{|v|}\int_{v(\mathbf{s}_0)} \mu(\mathbf{u}) d\mathbf{u}\right)\mathbf{1} = \frac{1}{|v|}\int_{v(\mathbf{s}_0)} \boldsymbol{\gamma}(\mathbf{u}) d\mathbf{u} = \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0)
$$
By uniqueness of the solution, the integrated weights match the block kriging weights $\boldsymbol{\lambda}_v = \frac{1}{|v|}\int_{v(\mathbf{s}_0)} \boldsymbol{\lambda}(\mathbf{u}) d\mathbf{u}$. By linearity, $\widehat{Z}_v(\mathbf{s}_0) = \frac{1}{|v|}\int_{v(\mathbf{s}_0)} \widehat{Z}(\mathbf{u}) d\mathbf{u}$.
For variances, averaging $\sigma_{\text{OK}}^2(\mathbf{u}) = \boldsymbol{\lambda}(\mathbf{u})'\boldsymbol{\gamma}(\mathbf{u}) + \mu(\mathbf{u})$ gives $\boldsymbol{\lambda}_v' \overline{\boldsymbol{\gamma}}_v(\mathbf{s}_0) + \mu_{\text{OK}} = \sigma_{\text{OK}, v}^2(\mathbf{s}_0) + \overline{\gamma}(v, v)$. Since $\overline{\gamma}(v, v) = D^2(\mathbf{0} \mid v) > 0$, the point variance exceeds block variance. $\blacksquare$

---

#### Beat 2: Algorithmic Translation (The "How")

```python
def solve_ordinary_block_kriging(
    data_coords: list[list[float]],
    data_values: list[float],
    target_block_center: list[float],
    block_dims: list[float],
    n_disc: list[int],
    gamma_func
) -> tuple[float, float, list[float]]:
    """
    Solves the Ordinary Block Kriging system for a rectangular target block.
    
    Parameters:
        data_coords: n x d list of sample coordinate vectors [s_1, ..., s_n]
        data_values: list of observed values [z(s_1), ..., z(s_n)]
        target_block_center: coordinate vector s_0 of block center
        block_dims: [dx, dy, ...] dimensions of the target block v
        n_disc: [mx, my, ...] discretization grid per dimension
        gamma_func: callable gamma(h_vector) returning scalar semivariance
        
    Returns:
        (z_pred, ok_var, weights): predicted block grade, variance, weights
    """
    import itertools
    n = len(data_values)
    dim = len(block_dims)
    
    # Step 1: Discretize the target block v centered at target_block_center
    coords_per_axis = []
    for d in range(dim):
        step = block_dims[d] / n_disc[d]
        start = target_block_center[d] - block_dims[d]/2.0 + step/2.0
        coords_per_axis.append([start + k * step for k in range(n_disc[d])])
    
    u_nodes = list(itertools.product(*coords_per_axis))
    M = len(u_nodes)
    
    # Step 2: Compute within-block average semivariogram gamma_bar(v, v)
    sum_gamma_vv = 0.0
    for j in range(M):
        for k in range(M):
            h_jk = [u_nodes[j][d] - u_nodes[k][d] for d in range(dim)]
            sum_gamma_vv += gamma_func(h_jk)
    gamma_bar_vv = sum_gamma_vv / (M * M)
    
    # Step 3: Compute point-to-block average semivariances gamma_bar(s_i, v)
    gamma_bar_iv = [0.0] * n
    for i in range(n):
        sum_gamma_iv = 0.0
        for k in range(M):
            h_ik = [data_coords[i][d] - u_nodes[k][d] for d in range(dim)]
            sum_gamma_iv += gamma_func(h_ik)
        gamma_bar_iv[i] = sum_gamma_iv / M
        
    # Step 4: Assemble the (n+1) x (n+1) Ordinary Kriging matrix K
    K = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            h_ij = [data_coords[i][d] - data_coords[j][d] for d in range(dim)]
            K[i][j] = gamma_func(h_ij)
        K[i][n] = 1.0
        K[n][i] = 1.0
    K[n][n] = 0.0
    
    # Right-hand side vector k_0
    k_0 = [gamma_bar_iv[i] for i in range(n)] + [1.0]
    
    # Step 5: Solve K * [lambda; mu] = k_0 using Gauss-Jordan elimination
    # (Augment matrix [K | k_0])
    A = [row[:] + [k_0[i]] for i, row in enumerate(K)]
    size = n + 1
    for col in range(size):
        # Pivot search
        pivot = col
        for row in range(col + 1, size):
            if abs(A[row][col]) > abs(A[pivot][col]):
                pivot = row
        A[col], A[pivot] = A[pivot], A[col]
        
        pivot_val = A[col][col]
        for c in range(col, size + 1):
            A[col][c] /= pivot_val
            
        for row in range(size):
            if row != col:
                factor = A[row][col]
                for c in range(col, size + 1):
                    A[row][c] -= factor * A[col][c]
                    
    solution = [A[row][size] for row in range(size)]
    weights = solution[:n]
    mu_ok = solution[n]
    
    # Step 6: Compute predicted block value
    z_pred = sum(weights[i] * data_values[i] for i in range(n))
    
    # Step 7: Compute block kriging variance
    ok_var = sum(weights[i] * gamma_bar_iv[i] for i in range(n)) + mu_ok - gamma_bar_vv
    
    return z_pred, ok_var, weights
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

- **Why Point Kriging Overestimates Block Uncertainty:**
  Notice the subtrahend $-\overline{\gamma}(v, v)$ in the block kriging variance formula. Point kriging predicts an infinitesimally small coordinate $Z(\mathbf{s}_0)$, which fluctuates erratically. Block kriging predicts an entire spatial average $Z_v(\mathbf{s}_0)$, whose variance is dampened by within-block averaging. Averaging point kriging variances across the block mistakenly includes internal point fluctuations ($D^2(\mathbf{0} \mid v)$) as prediction error.
- **Data Point Inside Target Block:**
  If an exploration borehole falls directly inside the target block $v(\mathbf{s}_0)$, point kriging would assign it a weight $\lambda = 1.0$ and return zero error variance. In block kriging, however, a single borehole inside a $20 \text{ m} \times 20 \text{ m}$ block does *not* receive weight 1.0, and the block kriging variance remains strictly positive. A single drill core does not eliminate uncertainty about the rest of the block!

---

### 8.9 Analytical Exercises and Complete Step-by-Step Solutions

#### Exercise 8.1: Regularized Covariance of a 1D Segment under an Exponential Covariance
Let $Z(s)$ be a 1D second-order stationary random field on $\mathbb{R}$ with exponential covariance $C(h) = \sigma^2 \exp(-|h|/a)$, where $a > 0$ is the correlation range. Let $v = [0, L]$ be a line segment of length $L > 0$.
**(a)** Compute the exact analytical within-block average covariance $\overline{C}(v, v)$.
**(b)** Deduce the dispersion variance of points within the segment $D^2(0 \mid v)$.
**(c)** Derive the asymptotic limits of $\overline{C}(v, v)$ as $L/a \to 0$ and $L/a \to \infty$.

##### Solution:
**(a)** By Definition 8.2.1:
$$
\overline{C}(v, v) = \frac{1}{L^2} \int_0^L \int_0^L \sigma^2 \exp\left(-\frac{|u - u'|}{a}\right) du du'
$$
Exploiting symmetry over the square $[0, L] \times [0, L]$:
$$
\overline{C}(v, v) = \frac{2\sigma^2}{L^2} \int_0^L du \int_0^u \exp\left(-\frac{u - u'}{a}\right) du'
$$
Evaluate the inner integral:
$$
\int_0^u \exp\left(\frac{u' - u}{a}\right) du' = \left[ a \exp\left(\frac{u' - u}{a}\right) \right]_{u'=0}^u = a \left(1 - \exp\left(-\frac{u}{a}\right)\right)
$$
Now evaluate the outer integral:
$$
\begin{aligned}
\int_0^L a \left(1 - \exp\left(-\frac{u}{a}\right)\right) du &= a \left[ u + a \exp\left(-\frac{u}{a}\right) \right]_0^L \\
&= a \left(L + a \exp\left(-\frac{L}{a}\right) - a\right) = a L - a^2 \left(1 - e^{-L/a}\right)
\end{aligned}
$$
Multiplying by $\frac{2\sigma^2}{L^2}$:
$$
\overline{C}(v, v) = \frac{2\sigma^2 a}{L} - \frac{2\sigma^2 a^2}{L^2} \left(1 - e^{-L/a}\right) = \frac{2\sigma^2 a^2}{L^2} \left(\frac{L}{a} - 1 + e^{-L/a}\right)
$$

**(b)** The dispersion variance of points within $v$ is:
$$
D^2(0 \mid v) = C(0) - \overline{C}(v, v) = \sigma^2 \left[ 1 - \frac{2a^2}{L^2}\left(\frac{L}{a} - 1 + e^{-L/a}\right) \right]
$$

**(c)** Let $\theta = L/a$.
As $\theta \to 0$, Taylor-expand $e^{-\theta} = 1 - \theta + \frac{\theta^2}{2} - \frac{\theta^3}{6} + \mathcal{O}(\theta^4)$:
$$
\theta - 1 + e^{-\theta} = \frac{\theta^2}{2} - \frac{\theta^3}{6} + \mathcal{O}(\theta^4)
$$
Substituting into $\overline{C}(v, v)$:
$$
\overline{C}(v, v) = \frac{2\sigma^2}{\theta^2}\left(\frac{\theta^2}{2} - \frac{\theta^3}{6} + \mathcal{O}(\theta^4)\right) = \sigma^2 \left(1 - \frac{\theta}{3} + \mathcal{O}(\theta^2)\right) \xrightarrow{\theta \to 0} \sigma^2 = C(0)
$$
As $\theta = L/a \to \infty$:
$$
e^{-\theta} \to 0 \implies \overline{C}(v, v) \approx \frac{2\sigma^2 a}{L} \to 0
$$
The regularized variance decays as $\mathcal{O}(L^{-1})$, verifying the continuous ergodic theorem in dimension $d=1$.

---

#### Exercise 8.2: Dispersion Variance Additivity in a Stratified Deposit
A continuous borehole of length $W = 12 \text{ m}$ is partitioned into 3 drill runs of length $V = 4 \text{ m}$, each of which is cut into 4 assay samples of length $v = 1 \text{ m}$.
The spatial semivariogram along the borehole is spherical:
$$
\gamma(h) = \begin{cases} c \left(\frac{3|h|}{2a} - \frac{|h|^3}{2a^3}\right), & |h| \le a \\ c, & |h| > a \end{cases}
$$
with sill $c = 10.0$ and range $a = 6 \text{ m}$.
The auxiliary function $\chi(L) = \overline{\gamma}(L, L)$ for a segment of length $L \le a$ is:
$$
\chi(L) = c \left(\frac{L}{2a} - \frac{L^3}{20a^3}\right)
$$
and for $L > a$:
$$
\chi(L) = c \left(1 - \frac{3a}{4L} + \frac{a^3}{20L^3}\right)
$$
**(a)** Compute $\overline{\gamma}(v, v)$ and $\overline{\gamma}(V, V)$.
**(b)** Compute $\overline{\gamma}(W, W)$.
**(c)** Verify Krige's Additivity Theorem: $D^2(v \mid W) = D^2(v \mid V) + D^2(V \mid W)$.

##### Solution:
**(a)** For $v = 1 \text{ m} \le a = 6 \text{ m}$:
$$
\overline{\gamma}(v, v) = \chi(1) = 10 \left(\frac{1}{2(6)} - \frac{1^3}{20(6)^3}\right) = 10 \left(\frac{1}{12} - \frac{1}{4320}\right) = 10 \left(\frac{359}{4320}\right) = \frac{359}{432} \approx 0.8310
$$
For $V = 4 \text{ m} \le a = 6 \text{ m}$:
$$
\overline{\gamma}(V, V) = \chi(4) = 10 \left(\frac{4}{12} - \frac{4^3}{20(216)}\right) = 10 \left(\frac{1}{3} - \frac{64}{4320}\right) = 10 \left(\frac{1440 - 64}{4320}\right) = \frac{1376}{432} \approx 3.1852
$$

**(b)** For $W = 12 \text{ m} > a = 6 \text{ m}$:
$$
\begin{aligned}
\overline{\gamma}(W, W) &= \chi(12) = 10 \left(1 - \frac{3(6)}{4(12)} + \frac{6^3}{20(12)^3}\right) \\
&= 10 \left(1 - \frac{18}{48} + \frac{216}{34560}\right) = 10 \left(\frac{5}{8} + \frac{1}{160}\right) = 10 \left(\frac{101}{160}\right) = \frac{101}{16} = 6.3125
\end{aligned}
$$
In common denominator $432$: $\frac{101}{16} = \frac{101 \times 27}{432} = \frac{2727}{432}$.

**(c)** Calculate dispersion variances:
$$
D^2(v \mid V) = \overline{\gamma}(V, V) - \overline{\gamma}(v, v) = \frac{1376 - 359}{432} = \frac{1017}{432} \approx 2.3542
$$
$$
D^2(V \mid W) = \overline{\gamma}(W, W) - \overline{\gamma}(V, V) = \frac{2727 - 1376}{432} = \frac{1351}{432} \approx 3.1273
$$
Sum of the two components:
$$
D^2(v \mid V) + D^2(V \mid W) = \frac{1017 + 1351}{432} = \frac{2368}{432} \approx 5.4815
$$
Compute $D^2(v \mid W)$ directly:
$$
D^2(v \mid W) = \overline{\gamma}(W, W) - \overline{\gamma}(v, v) = \frac{2727 - 359}{432} = \frac{2368}{432} \approx 5.4815
$$
The two sides match identically, confirming Krige's Additivity Theorem.

---

#### Exercise 8.3: Bias and Variance in Block Kriging with a Single Point Observation
Let $Z(\mathbf{s})$ be a 2D stationary random field with unknown mean $\mu$ and covariance $C(\mathbf{h})$. A mining block $V$ is estimated from a single point sample $Z(\mathbf{s}_1)$ at the center of $V$.
**(a)** Formulate the Ordinary Block Kriging estimator $\widehat{Z}_V = \lambda_1 Z(\mathbf{s}_1)$ and determine $\lambda_1$.
**(b)** Derive the kriging variance $\sigma_{\text{OK}, V}^2$.
**(c)** Show that the average point prediction error across $V$ exceeds $\sigma_{\text{OK}, V}^2$ by $D^2(\mathbf{0} \mid V)$.

##### Solution:
**(a)** Unbiasedness requires $\sum_{i=1}^1 \lambda_i = 1$, which uniquely fixes $\lambda_1 = 1$. The estimator is $\widehat{Z}_V = Z(\mathbf{s}_1)$.

**(b)** The block kriging system for $n=1$ is:
$$
\lambda_1 C(\mathbf{0}) - \mu_C = \overline{C}(\mathbf{s}_1, V) \implies \mu_C = C(\mathbf{0}) - \overline{C}(\mathbf{s}_1, V)
$$
The block kriging estimation variance is:
$$
\sigma_{\text{OK}, V}^2 = \overline{C}(V, V) - \lambda_1 \overline{C}(\mathbf{s}_1, V) + \mu_C = C(\mathbf{0}) - 2\overline{C}(\mathbf{s}_1, V) + \overline{C}(V, V)
$$
In semivariogram notation, using $C(\mathbf{h}) = C(\mathbf{0}) - \gamma(\mathbf{h})$:
$$
\sigma_{\text{OK}, V}^2 = 2\overline{\gamma}(\mathbf{s}_1, V) - \overline{\gamma}(V, V)
$$

**(c)** If $Z(\mathbf{s}_1)$ is used to predict point value $Z(\mathbf{u})$ at $\mathbf{u} \in V$, the point error variance is:
$$
\sigma^2(\mathbf{u}) = \operatorname{E}[(Z(\mathbf{s}_1) - Z(\mathbf{u}))^2] = 2\gamma(\mathbf{s}_1 - \mathbf{u})
$$
Averaging over all points $\mathbf{u} \in V$:
$$
\overline{\sigma^2_{\text{point}}} = \frac{1}{|V|} \int_V 2\gamma(\mathbf{s}_1 - \mathbf{u}) d\mathbf{u} = 2\overline{\gamma}(\mathbf{s}_1, V)
$$
Subtracting the block kriging variance:
$$
\overline{\sigma^2_{\text{point}}} - \sigma_{\text{OK}, V}^2 = 2\overline{\gamma}(\mathbf{s}_1, V) - [2\overline{\gamma}(\mathbf{s}_1, V) - \overline{\gamma}(V, V)] = \overline{\gamma}(V, V) = D^2(\mathbf{0} \mid V)
$$
Hence:
$$
\overline{\sigma^2_{\text{point}}} = \sigma_{\text{OK}, V}^2 + D^2(\mathbf{0} \mid V)
$$
Point kriging variance systematically exceeds block prediction uncertainty by the within-block dispersion variance.

---

#### Exercise 8.4: Indirect Lognormal Change of Support for Mining Selectivity
A gold prospect has point-support drill core grades $Z = \exp(Y)$ with $Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$, where $\mu_Y = 0.5$ and $\sigma_Y^2 = 1.0$.
Mining will proceed using Selective Mining Units (blocks $v$) for which within-block average covariance calculation yields $\overline{C}(v, v) = 2.50$.
The economic cut-off grade is $z_c = 3.0 \text{ g/t}$.
**(a)** Compute the arithmetic mean grade $m = \operatorname{E}[Z]$ and point variance $\sigma_0^2 = \operatorname{Var}(Z)$.
**(b)** Using the indirect lognormal correction model, compute the block logarithmic parameters $\sigma_{Y_v}^2$ and $\mu_{Y_v}$.
**(c)** Calculate the recovered tonnage fraction $T_v(z_c)$ and mean grade above cut-off $m_v(z_c)$ for block support $v$.
**(d)** Compare $T_v(z_c)$ and $m_v(z_c)$ with the naive point-support selection $T_0(z_c)$ and $m_0(z_c)$.

##### Solution:
**(a)** Point support parameters:
$$
m = \exp\left(\mu_Y + \frac{1}{2}\sigma_Y^2\right) = \exp\left(0.5 + 0.5\right) = \exp(1.0) \approx 2.7183 \text{ g/t}
$$
$$
\sigma_0^2 = m^2 [\exp(\sigma_Y^2) - 1] = (2.7183)^2 [\exp(1.0) - 1] = 7.3891 \times 1.7183 \approx 12.6966 (\text{g/t})^2
$$

**(b)** Block logarithmic parameters:
The block variance is $\sigma_v^2 = \overline{C}(v, v) = 2.50$.
$$
\sigma_{Y_v}^2 = \ln\left(1 + \frac{\sigma_v^2}{m^2}\right) = \ln\left(1 + \frac{2.50}{7.3891}\right) = \ln(1 + 0.33834) = \ln(1.33834) \approx 0.2914
$$
$$
\sigma_{Y_v} = \sqrt{0.2914} \approx 0.5398
$$
$$
\mu_{Y_v} = \ln(m) - \frac{1}{2}\sigma_{Y_v}^2 = 1.0 - \frac{1}{2}(0.2914) = 1.0 - 0.1457 = 0.8543
$$

**(c)** Recovered block quantities at cut-off $z_c = 3.0 \text{ g/t}$ ($\ln z_c = \ln 3.0 \approx 1.0986$):
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
Q_v(z_c) = m \cdot \Phi(d_2) = 2.7183 \cdot \Phi(0.0872) \approx 2.7183 \times 0.5347 \approx 1.4535 \text{ g/t}
$$
Mean recovered block grade:
$$
m_v(z_c) = \frac{Q_v(z_c)}{T_v(z_c)} = \frac{1.4535}{0.3254} \approx 4.4668 \text{ g/t}
$$

**(d)** Point support selection at cut-off $z_c = 3.0 \text{ g/t}$:
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
Q_0(z_c) = m \cdot \Phi(0.4014) = 2.7183 \times 0.6559 \approx 1.7830 \text{ g/t}
$$
$$
m_0(z_c) = \frac{Q_0(z_c)}{T_0(z_c)} = \frac{1.7830}{0.2747} \approx 6.4907 \text{ g/t}
$$

**Critical Comparison:**
Notice that naive point-support evaluation forecasts a mean grade of $6.49 \text{ g/t}$, whereas the realistic block grade delivered by mining shovels is only $4.47 \text{ g/t}$—a **31.2% over-estimation of grade**! Assuming point support in mine design would result in catastrophic shortfall in recovered metal and plant underperformance.


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 8: Change of Support and Volume Regularization — Companion Exercises and Complete Solutions](ch08_exercises.md)
