# Chapter 7: Multivariate Spatial Prediction and Cokriging

In many environmental, geophysical, geological, and ecological investigations, scientists rarely sample a single spatial variable in isolation. Mining engineers evaluate gold alongside silver and copper; hydrologists analyze piezometric water table depth alongside topography, precipitation, and soil permeability; soil scientists measure organic matter alongside electrical conductivity and clay content. 

In spatial linear models, when multiple cross-correlated variables are observed across a spatial domain $D \subset \mathbb{R}^d$, predicting a primary variable $Z_1(\mathbf{s}_0)$ can be significantly improved by incorporating secondary auxiliary variables $\{Z_2, \dots, Z_p\}$. The collection of optimal linear spatial prediction techniques for multivariate random fields is termed **cokriging**.

Following the pedagogical philosophy of Maria L. Rizzo and the semantic linear-model discipline of Oliver Schabenberger & Carol A. Gotway, this chapter establishes the rigorous theory of multivariate spatial prediction through the coordinated Three-Beat framework. We formulate vector random fields and cross-covariance kernels, analyze isotopic and heterotopic sampling geometries, derive Simple and Ordinary Cokriging via Lagrangian optimization, resolve the collocated cokriging dilemma, prove the celebrated Autokrigeability Theorem via Kronecker product algebra, and provide transparent Pythonic pseudocode algorithms matching the mathematical mechanics line for line.

---

### 7.1 Multivariate Spatial Processes and Cross-Continuity

#### Beat 1: The Mathematical Anchor (The "Why")

Let $(\Omega, \mathcal{F}, P)$ be a complete probability space. A **multivariate spatial random field** is a vector-valued stochastic process:
$$
\mathbf{Z}(\mathbf{s}) = \begin{bmatrix} Z_1(\mathbf{s}) \\ Z_2(\mathbf{s}) \\ \vdots \\ Z_p(\mathbf{s}) \end{bmatrix}, \quad \mathbf{s} \in D \subset \mathbb{R}^d
$$
where each component $Z_i(\mathbf{s}) \in L^2(\Omega, \mathcal{F}, P)$ is a square-integrable scalar random variable.

Following Schabenberger & Gotway's linear-model decomposition, each component process admits the structural representation:
$$
Z_i(\mathbf{s}) = \mu_i(\mathbf{s}) + \delta_i(\mathbf{s}) + \epsilon_i(\mathbf{s}) = \mathbf{x}_i(\mathbf{s})'\boldsymbol{\beta}_i + \delta_i(\mathbf{s}) + \epsilon_i(\mathbf{s}), \quad i=1, \dots, p
$$
where $\mu_i(\mathbf{s}) = \operatorname{E}[Z_i(\mathbf{s})]$ is the deterministic mean trend, $\delta_i(\mathbf{s})$ is a zero-mean continuous spatial sub-process, and $\epsilon_i(\mathbf{s})$ represents micro-scale variation or measurement error.

#### Classical-to-Spatial Bridge: From Classical Multivariate Regression and Pearson Cross-Correlations to the Spatial Cross-Covariance Tensor

In classical multivariate statistics, an investigator analyzes $m$ distinct quantitative response variables observed across $n$ independent experimental units. The classical multivariate linear regression model is expressed in matrix form as:
$$
\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}
$$
where $\mathbf{Y} \in \mathbb{R}^{n \times m}$ is the response matrix containing $m$ features observed across $n$ subjects, $\mathbf{X} \in \mathbb{R}^{n \times q}$ is the design matrix of $q$ explanatory covariates, $\mathbf{B} \in \mathbb{R}^{q \times m}$ is the matrix of regression coefficients, and $\mathbf{E} \in \mathbb{R}^{n \times m}$ is the random error matrix.

Under the classical assumption of independent observational units, the rows $\mathbf{e}_i'$ of $\mathbf{E}$ are i.i.d. random vectors with mean $\mathbf{0}$ and an $m \times m$ cross-covariance matrix:
$$
\operatorname{Var}(\mathbf{e}_i) = \boldsymbol{\Sigma} = \begin{bmatrix}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1m} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{m1} & \sigma_{m2} & \cdots & \sigma_{mm}
\end{bmatrix} \in \mathbb{R}^{m \times m}
$$
In this classical i.i.d. framework:
1. **Cross-Unit Independence:** Distinct units are uncorrelated: $\operatorname{Cov}(\mathbf{e}_i, \mathbf{e}_j) = \mathbf{0}$ for $i \ne j$.
2. **Static Symmetric Co-variation:** The covariance matrix $\boldsymbol{\Sigma}$ is strictly symmetric: $\sigma_{jk} = \operatorname{Cov}(E_{ij}, E_{ik}) = \sigma_{kj}$ for all attribute pairs $(j, k)$. The classical Pearson cross-correlation $\rho_{jk} = \sigma_{jk} / (\sigma_j \sigma_k)$ measures co-variation strictly at the same observational unit.

Now transition to spatial statistics. When $m$ continuous attributes are observed across a continuous geographic domain $D \subset \mathbb{R}^d$, the observation at coordinate vector $\mathbf{s} \in D$ is an $m$-dimensional vector process:
$$
\mathbf{Z}(\mathbf{s}) = \begin{bmatrix} Z_1(\mathbf{s}) \\ Z_2(\mathbf{s}) \\ \vdots \\ Z_m(\mathbf{s}) \end{bmatrix} \in \mathbb{R}^m
$$
Spatial cross-dependence can no longer be characterized by a single static matrix $\boldsymbol{\Sigma}$. Instead, the cross-covariance between attribute $j$ and attribute $k$ varies continuously with the spatial separation lag vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j \in \mathbb{R}^d$, generating the **spatial cross-covariance function**:
$$
C_{jk}(\mathbf{h}) = \operatorname{Cov}(Z_j(\mathbf{s}), Z_k(\mathbf{s}+\mathbf{h})) = \operatorname{E}[(Z_j(\mathbf{s}) - \mu_j)(Z_k(\mathbf{s}+\mathbf{h}) - \mu_k)]
$$
This transition from classical regression to spatial multivariate prediction exposes two critical mathematical departures:
- **Directional Asymmetry ($C_{jk}(\mathbf{h}) \ne C_{jk}(-\mathbf{h})$):** In classical statistics, covariance between two variables is inherently symmetric. In spatial processes, however, physical directional phenomena (e.g., river currents, atmospheric dispersion, tectonic dip) introduce spatial time-lags or delays: measuring variable $j$ at upstream coordinate $\mathbf{s}$ and variable $k$ at downstream coordinate $\mathbf{s} + \mathbf{h}$ does *not* equal measuring variable $j$ at downstream coordinate $\mathbf{s}$ and variable $k$ at upstream coordinate $\mathbf{s} - \mathbf{h}$. Thus, in general:
  $$
  C_{jk}(\mathbf{h}) \ne C_{jk}(-\mathbf{h}) \quad \text{for } j \ne k
  $$
  Instead, spatial cross-covariances satisfy **transpose symmetry**:
  $$
  C_{jk}(\mathbf{h}) = C_{kj}(-\mathbf{h}) \iff \mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})'
  $$
  where $\mathbf{C}(\mathbf{h}) \in \mathbb{R}^{m \times m}$ is the spatial cross-covariance matrix function.
- **Joint Positive Definiteness:** In classical regression, any positive semi-definite matrix $\boldsymbol{\Sigma} \succeq 0$ is mathematically admissible. In spatial multivariate prediction, the matrix-valued function $\mathbf{C}(\mathbf{h})$ must be **jointly positive definite across all configurations of spatial points and lags**:
  $$
  \sum_{i=1}^n \sum_{j=1}^n \mathbf{a}_i' \mathbf{C}(\mathbf{s}_i - \mathbf{s}_j) \mathbf{a}_j \ge 0 \quad \forall n \ge 1, \; \mathbf{s}_i \in D, \; \mathbf{a}_i \in \mathbb{R}^m
  $$
  Arbitrarily combining valid univariate covariance functions almost universally violates this joint condition. This mathematical requirement leads directly to the **Linear Model of Coregionalization (LMC)**, which constructs $\mathbf{C}(\mathbf{h})$ as a superposition of valid scalar correlation functions weighted by positive semi-definite coregionalization matrices.

```
+-------------------------------------------------------------------------------------------------+
| DEFINITION 7.1: Second-Order Stationarity of Multivariate Random Fields                         |
| A multivariate random field $\{\mathbf{Z}(\mathbf{s}) : \mathbf{s} \in D\}$ is second-order   |
| stationary if:                                                                                  |
| 1. The mean vector is constant across space:                                                    |
|    $$                                                                                           |
|    \operatorname{E}[\mathbf{Z}(\mathbf{s})] = \boldsymbol{\mu} = (\mu_1, \dots, \mu_p)' \quad \forall \mathbf{s} \in D |
|    $$                                                                                           |
| 2. The cross-covariance between any two components $Z_i(\mathbf{s})$ and $Z_j(\mathbf{s}+\mathbf{h})$|
|    depends only on the lag separation vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j \in \mathbb{R}^d$:|
|    $$                                                                                           |
|    C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s})) = \operatorname{E}[(Z_i(\mathbf{s}+\mathbf{h}) - \mu_i)(Z_j(\mathbf{s}) - \mu_j)] |
|    $$                                                                                           |
+-------------------------------------------------------------------------------------------------+
```

The collection of all auto- and cross-covariances forms the **matrix covariance function**:
$$
\mathbf{C}(\mathbf{h}) = \begin{bmatrix}
C_{11}(\mathbf{h}) & C_{12}(\mathbf{h}) & \cdots & C_{1p}(\mathbf{h}) \\
C_{21}(\mathbf{h}) & C_{22}(\mathbf{h}) & \cdots & C_{2p}(\mathbf{h}) \\
\vdots & \vdots & \ddots & \vdots \\
C_{p1}(\mathbf{h}) & C_{p2}(\mathbf{h}) & \cdots & C_{pp}(\mathbf{h})
\end{bmatrix} \in \mathbb{R}^{p \times p}
$$
where $C_{ii}(\mathbf{h})$ is the univariate auto-covariance of variable $i$, and $C_{ij}(\mathbf{h})$ ($i \ne j$) is the cross-covariance between variables $i$ and $j$.

```
+-------------------------------------------------------------------------------------------------+
| PROPOSITION 7.1: Fundamental Properties of Cross-Covariances                                   |
| 1. **Cross-Symmetry Identity:** For all lag vectors $\mathbf{h} \in \mathbb{R}^d$:              |
|    $$                                                                                           |
|    C_{ij}(\mathbf{h}) = C_{ji}(-\mathbf{h}) \iff \mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})' |
|    $$                                                                                           |
| 2. **Asymmetry in General:** Unlike auto-covariances, cross-covariances are generally NOT even:  |
|    $$                                                                                           |
|    C_{ij}(\mathbf{h}) \ne C_{ij}(-\mathbf{h}) \quad \text{for } i \ne j                        |
|    $$                                                                                           |
| 3. **Cauchy-Schwarz Cross-Correlation Bound:** For all $\mathbf{h} \in \mathbb{R}^d$:           |
|    $$                                                                                           |
|    |C_{ij}(\mathbf{h})| \le \sqrt{C_{ii}(\mathbf{0}) C_{jj}(\mathbf{0})}                       |
|    $$                                                                                           |
+-------------------------------------------------------------------------------------------------+
```

**Proof of Cross-Symmetry:**
By definition:
$$
C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s}))
$$
Setting $\mathbf{u} = \mathbf{s} + \mathbf{h}$, so that $\mathbf{s} = \mathbf{u} - \mathbf{h}$:
$$
C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{u}), Z_j(\mathbf{u}-\mathbf{h})) = \operatorname{Cov}(Z_j(\mathbf{u}-\mathbf{h}), Z_i(\mathbf{u})) = C_{ji}(-\mathbf{h})
$$
In matrix notation, $\mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})'$. $\blacksquare$

```
+-------------------------------------------------------------------------------------------------+
| THEOREM 7.1: Cramér's Multivariate Spectral Theorem                                             |
| A matrix function $\mathbf{C} : \mathbb{R}^d \to \mathbb{R}^{p \times p}$ is a valid covariance |
| matrix function if and only if it admits the spectral representation:                           |
| $$                                                                                              |
| \mathbf{C}(\mathbf{h}) = \int_{\mathbb{R}^d} e^{i \boldsymbol{\omega}'\mathbf{h}} d\mathbf{F}(\boldsymbol{\omega}) |
| $$                                                                                              |
| where $\mathbf{F}$ is a $p \times p$ Hermitian positive semi-definite matrix of Borel measures  |
| on $\mathbb{R}^d$, meaning that for any Borel set $E \subset \mathbb{R}^d$ and any $\mathbf{a} \in \mathbb{C}^p$, |
| $\mathbf{a}^* \mathbf{F}(E) \mathbf{a} \ge 0$.                                                  |
+-------------------------------------------------------------------------------------------------+
```

#### The Linear Model of Coregionalization (LMC)
Constructing matrix covariance functions that satisfy Cramér's theorem is challenging because choosing arbitrary valid univariate models for each $C_{ij}(\mathbf{h})$ almost always results in a matrix that violates positive semi-definiteness. 

The standard analytical construction that guarantees mathematical admissibility in geostatistics is the **Linear Model of Coregionalization (LMC)** (Journel & Huijbregts, 1978). In the LMC, each variable $Z_i(\mathbf{s})$ is modeled as a linear combination of $K$ independent, orthogonal spatial sub-processes $\{Y_k(\mathbf{s})\}_{k=1}^K$:
$$
\mathbf{Z}(\mathbf{s}) = \sum_{k=1}^K \mathbf{A}_k \mathbf{Y}_k(\mathbf{s})
$$
where each vector process $\mathbf{Y}_k(\mathbf{s})$ has spatial correlation function $\rho_k(\mathbf{h})$ (with $\rho_k(\mathbf{0})=1$) and uncorrelated components. The resulting matrix covariance is:
$$
\mathbf{C}(\mathbf{h}) = \sum_{k=1}^K \mathbf{B}_k \rho_k(\mathbf{h})
$$
where $\mathbf{B}_k = \mathbf{A}_k \mathbf{A}_k' \in \mathbb{R}^{p \times p}$ is called the **coregionalization matrix** associated with spatial structure $k$.

##### Term-by-Term Equation Anatomy: The Linear Model of Coregionalization (LMC)
$$
\underbrace{\mathbf{C}(\mathbf{h})}_{\substack{\text{Spatial Cross-Covariance} \\ \text{Matrix Kernel } (m \times m)}} = \sum_{k=1}^K \underbrace{\mathbf{B}_k}_{\substack{\text{Coregionalization Matrix} \\ \text{Positive Semi-Definite } (m \times m)}} \cdot \underbrace{\rho_k(\mathbf{h})}_{\substack{\text{Authorized Univariate} \\ \text{Spatial Correlation Function}}}
$$
- **$\mathbf{C}(\mathbf{h}) \in \mathbb{R}^{m \times m}$:** The matrix-valued spatial cross-covariance kernel evaluated at separation lag vector $\mathbf{h} \in \mathbb{R}^d$. The diagonal elements $C_{jj}(\mathbf{h})$ represent the univariate auto-covariances of each variable $j \in \{1, \dots, m\}$, while the off-diagonal entries $C_{jk}(\mathbf{h})$ ($j \ne k$) represent the cross-covariances between pairs of distinct variables $j$ and $k$.
- **$\sum_{k=1}^K$:** The discrete superposition over $K$ distinct, orthogonal spatial scales or physical processes (e.g., $k=1$ represents nugget/microscale variation, $k=2$ captures short-range diffusion, and $k=3$ represents long-range geological drift).
- **$\mathbf{B}_k \in \mathbb{R}^{m \times m}$:** The coregionalization matrix for spatial scale $k$. To guarantee that the joint spatial covariance is mathematically admissible, each $\mathbf{B}_k = [b_{ij}^k]_{i,j=1}^m$ must be symmetric ($\mathbf{B}_k = \mathbf{B}_k'$) and positive semi-definite ($\mathbf{B}_k \succeq 0$). The diagonal entry $b_{jj}^k \ge 0$ is the partial sill (variance contribution) of variable $j$ at scale $k$, while the off-diagonal entry $b_{jk}^k$ is the co-sill (cross-covariance contribution) linking variables $j$ and $k$ at scale $k$.
- **$\rho_k(\mathbf{h})$:** An authorized, valid univariate spatial correlation function on $\mathbb{R}^d$ (e.g., Spherical, Exponential, Gaussian, or Matérn) satisfying $\rho_k(\mathbf{0}) = 1$ and positive definiteness. It governs the spatial range and geometric rate of spatial decay for scale $k$, shared across all variables active at that scale.

```
+-------------------------------------------------------------------------------------------------+
| THEOREM 7.2: Admissibility of the Linear Model of Coregionalization                             |
| The matrix covariance function $\mathbf{C}(\mathbf{h}) = \sum_{k=1}^K \mathbf{B}_k \rho_k(\mathbf{h})$ |
| is strictly positive semi-definite on $\mathbb{R}^d$ if and only if:                            |
| 1. Every spatial correlation function $\rho_k(\mathbf{h})$ is valid and positive definite on $\mathbb{R}^d$.|
| 2. Every coregionalization matrix $\mathbf{B}_k$ is symmetric and positive semi-definite:       |
|    $$                                                                                           |
|    \mathbf{B}_k = \mathbf{B}_k' \quad \text{and} \quad \mathbf{u}' \mathbf{B}_k \mathbf{u} \ge 0 \quad \forall \mathbf{u} \in \mathbb{R}^p, \quad \forall k=1,\dots, K |
|    $$                                                                                           |
+-------------------------------------------------------------------------------------------------+
```

For a bivariate system ($p=2$), $\mathbf{B}_k = \begin{bmatrix} b_{11}^k & b_{12}^k \\ b_{12}^k & b_{22}^k \end{bmatrix} \succeq 0$ if and only if:
$$
b_{11}^k \ge 0, \quad b_{22}^k \ge 0, \quad \text{and} \quad b_{11}^k b_{22}^k - (b_{12}^k)^2 \ge 0
$$

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why can't we simply fit independent parametric variograms to each primary variable and cross-variable without checking positive definiteness?*  
> If an analyst independently fits parametric models to empirical semivariograms (e.g., choosing a spherical model for $\gamma_{11}(\mathbf{h})$, an exponential model for $\gamma_{22}(\mathbf{h})$, and a Gaussian model for the cross-semivariogram $\gamma_{12}(\mathbf{h})$), the unconstrained combination will almost certainly violate the spatial Cauchy-Schwarz bound:
> $$
> |\gamma_{12}(\mathbf{h})| \le \sqrt{\gamma_{11}(\mathbf{h}) \gamma_{22}(\mathbf{h})} \quad \forall \mathbf{h} \in \mathbb{R}^d
> $$
> Even if the bound holds at isolated lags, the assembled multivariate cokriging block matrix across all sample pairs:
> $$
> \boldsymbol{\Gamma} = \begin{bmatrix} \boldsymbol{\Gamma}_{11} & \boldsymbol{\Gamma}_{12} \\ \boldsymbol{\Gamma}_{21} & \boldsymbol{\Gamma}_{22} \end{bmatrix}
> $$
> can easily fail to be conditionally negative definite (or equivalently, the global covariance matrix $\boldsymbol{\Sigma}$ will possess negative eigenvalues). When inverted during cokriging, this mathematical invalidity leads to catastrophic failures: non-invertible or ill-conditioned linear systems, nonsensical negative prediction variances ($\sigma_{\text{OCK}}^2(\mathbf{s}_0) < 0$), and severe violations of the axioms of probability. The LMC construction mathematically guarantees that the joint covariance matrix is positive semi-definite everywhere by construction.

---

#### Beat 2: Algorithmic Translation (The "How")

The following procedure evaluates the full $N \times N$ block covariance matrix for an arbitrary set of sample points and variables under a Linear Model of Coregionalization.

```python
def assemble_multivariate_covariance_block_matrix(sample_coords_by_var, B_matrices, rho_models):
    """
    Assembles the global N x N multivariate covariance matrix Sigma for an LMC.
    
    Parameters:
        sample_coords_by_var: List of length p, where sample_coords_by_var[i] is a list
                              of n_i spatial coordinates for variable i.
        B_matrices: List of K coregionalization matrices [B_1, ..., B_K], each p x p.
        rho_models: List of K correlation functions [rho_1, ..., rho_K], where rho_k(h)
                    evaluates the univariate correlation at scalar distance h.
                    
    Returns:
        Sigma: Global N x N symmetric positive semi-definite covariance matrix,
               where N = sum(n_i).
        block_indices: List of start and end row indices for each variable block.
    """
    p = len(sample_coords_by_var)
    K = len(B_matrices)
    n_counts = [len(coords) for coords in sample_coords_by_var]
    N = sum(n_counts)
    
    # Track index boundaries for each variable block
    block_indices = []
    current_idx = 0
    for count in n_counts:
        block_indices.append((current_idx, current_idx + count))
        current_idx += count
        
    # Initialize global N x N matrix with zeros
    Sigma = [[0.0 for _ in range(N)] for _ in range(N)]
    
    # Populate each (i, j) variable block
    for i in range(p):
        start_i, end_i = block_indices[i]
        coords_i = sample_coords_by_var[i]
        n_i = n_counts[i]
        
        for j in range(i, p):
            start_j, end_j = block_indices[j]
            coords_j = sample_coords_by_var[j]
            n_j = n_counts[j]
            
            # Fill block Sigma_ij of size n_i x n_j
            for alpha in range(n_i):
                s_alpha = coords_i[alpha]
                row = start_i + alpha
                
                for beta in range(n_j):
                    s_beta = coords_j[beta]
                    col = start_j + beta
                    
                    # If i == j, enforce symmetry to avoid redundant distance calculations
                    if i == j and beta < alpha:
                        Sigma[row][col] = Sigma[col][row]
                        continue
                        
                    # Compute Euclidean distance: h = ||s_alpha - s_beta||
                    diff_sq = sum((s_alpha[d] - s_beta[d]) ** 2 for d in range(len(s_alpha)))
                    h = diff_sq ** 0.5
                    
                    # Evaluate LMC cross-covariance: C_ij(h) = sum_k B_k[i][j] * rho_k(h)
                    val = 0.0
                    for k in range(K):
                        val += B_matrices[k][i][j] * rho_models[k](h)
                        
                    Sigma[row][col] = val
                    if i != j:
                        # Cross-symmetry: Sigma_ji[beta, alpha] = Sigma_ij[alpha, beta]
                        Sigma[col][row] = val
                        
    return Sigma, block_indices
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

1. **The Physical Meaning of Cross-Covariance Asymmetry:**
   Why do natural phenomena exhibit $C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})$?
   - Consider a river flowing East ($\mathbf{h} = (h_x, 0)$). Industrial effluent discharges lead ($Z_1$) and zinc ($Z_2$) into the water. Chemical reactions cause lead to precipitate out within $500\text{ m}$, while zinc remains suspended for $3000\text{ m}$.
   - Evaluating cross-covariance along the downstream direction shows high correlation between lead at $\mathbf{s}$ and zinc at $\mathbf{s} + 1500\text{ m}$ ($C_{12}(\mathbf{h}) > 0$).
   - However, zinc upstream cannot be caused by lead downstream, so $C_{12}(-\mathbf{h}) \approx 0$.
   - Symmetric models like the standard LMC cannot capture this directional convection/transport; modeling asymmetry requires generalized vector autoregressive (VAR) spatial kernels or advection-diffusion covariance operators.
2. **Sylvester Violation Warning:**
   If a user fits cross-variograms independently and obtains empirical sills with $b_{12} = 5.0$ when $b_{11} = 2.0$ and $b_{22} = 3.0$, the determinant is $2(3) - 5^2 = 6 - 25 = -19 < 0$. The resulting cokriging matrix will possess large negative eigenvalues, causing kriging variance to become negative.

---

### 7.2 Multivariate Sampling Geometries

#### Beat 1: The Mathematical Anchor (The "Why")

In multivariate geostatistics, the relative spatial arrangement of observations profoundly impacts computational complexity and prediction performance. We classify sampling configurations into three fundamental geometries:

```
+-------------------------------------------------------------------------------------------------+
| DEFINITION 7.2: Multivariate Sampling Geometries                                                |
| Let $S_i = \{\mathbf{s}_{i,1}, \dots, \mathbf{s}_{i,n_i}\} \subset D$ denote the set of sample    |
| locations where variable $Z_i$ is observed ($i=1, \dots, p$).                                   |
| 1. **Isotopic Configuration:** All variables are measured at the exact same spatial locations:   |
|    $$                                                                                           |
|    S_1 = S_2 = \cdots = S_p = S, \quad \text{with } n_1 = n_2 = \cdots = n_p = n               |
|    $$                                                                                           |
| 2. **Heterotopic Configuration:** The sampling locations differ across variables ($S_i \ne S_j$).|
| 3. **Partially Heterotopic (Collocated) Configuration:** The primary variable $Z_1$ is sampled   |
|    sparsely ($n_1$ is small), while secondary variables $\{Z_2, \dots, Z_p\}$ are sampled       |
|    densely or exhaustively across the entire domain ($n_2 \gg n_1$).                            |
+-------------------------------------------------------------------------------------------------+
```

```
Multivariate Sampling Configurations:

Isotopic (All variables co-located):   Partially Heterotopic (Collocated secondary):
+-----------------------------------+  +-----------------------------------+
|  (Z1,Z2)                 (Z1,Z2)  |  |  (Z1,Z2)       Z2        (Z1,Z2)  |
|                                   |  |       Z2             Z2            |
|              (Z1,Z2)              |  |              (Z1,Z2)          Z2   |
|                                   |  |  Z2                  Z2            |
|  (Z1,Z2)                 (Z1,Z2)  |  |  (Z1,Z2)       Z2        (Z1,Z2)  |
+-----------------------------------+  +-----------------------------------+
```

Let $N = \sum_{i=1}^p n_i$ denote the total number of scalar measurements. The global observation vector is stacked blockwise:
$$
\mathbf{Z} = \begin{bmatrix} \mathbf{Z}_1 \\ \mathbf{Z}_2 \\ \vdots \\ \mathbf{Z}_p \end{bmatrix} \in \mathbb{R}^N, \quad \text{where } \mathbf{Z}_i = \begin{bmatrix} Z_i(\mathbf{s}_{i,1}) \\ \vdots \\ Z_i(\mathbf{s}_{i,n_i}) \end{bmatrix} \in \mathbb{R}^{n_i}
$$
The global $N \times N$ covariance matrix $\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{Z})$ is partitioned into $p \times p$ sub-matrices:
$$
\boldsymbol{\Sigma} = \begin{bmatrix}
\boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} & \cdots & \boldsymbol{\Sigma}_{1p} \\
\boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} & \cdots & \boldsymbol{\Sigma}_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
\boldsymbol{\Sigma}_{p1} & \boldsymbol{\Sigma}_{p2} & \cdots & \boldsymbol{\Sigma}_{pp}
\end{bmatrix} \in \mathbb{R}^{N \times N}
$$
where each block $[\boldsymbol{\Sigma}_{ij}]_{\alpha, \beta} = C_{ij}(\mathbf{s}_{i,\alpha} - \mathbf{s}_{j,\beta}) \in \mathbb{R}^{n_i \times n_j}$.

---

### 7.3 Simple and Ordinary Cokriging Systems

#### Beat 1: The Mathematical Anchor (The "Why")

Let $\mathbf{s}_0 \in D$ be an unsampled location where we seek to predict the primary target variable $Z_1(\mathbf{s}_0)$. The general linear multivariate predictor is:
$$
\widehat{Z}_1(\mathbf{s}_0) = \sum_{i=1}^p \sum_{\alpha=1}^{n_i} \lambda_{i\alpha} Z_i(\mathbf{s}_{i\alpha}) = \boldsymbol{\lambda}' \mathbf{Z}
$$
where $\boldsymbol{\lambda} = (\boldsymbol{\lambda}_1', \dots, \boldsymbol{\lambda}_p')' \in \mathbb{R}^N$ is the vector of cokriging weights, with $\boldsymbol{\lambda}_i = (\lambda_{i1}, \dots, \lambda_{in_i})'$.

#### 1. Simple Cokriging (SCK)
When the mean vector $\boldsymbol{\mu} = (\mu_1, \dots, \mu_p)'$ is completely known, the error is $e(\mathbf{s}_0) = \widehat{Z}_1(\mathbf{s}_0) - Z_1(\mathbf{s}_0)$. Working with zero-mean centered variables $Y_i(\mathbf{s}) = Z_i(\mathbf{s}) - \mu_i$, the predictor is:
$$
\widehat{Z}_{1,\text{SCK}}(\mathbf{s}_0) = \mu_1 + \sum_{i=1}^p \sum_{\alpha=1}^{n_i} \lambda_{i\alpha}^{\text{SCK}} (Z_i(\mathbf{s}_{i\alpha}) - \mu_i) = \mu_1 + \boldsymbol{\lambda}_{\text{SCK}}' (\mathbf{Z} - \mathbf{X}\boldsymbol{\mu})
$$
Minimizing the mean-squared prediction error $\sigma_{\text{SCK}}^2(\mathbf{s}_0) = \operatorname{E}[(\widehat{Z}_1(\mathbf{s}_0) - Z_1(\mathbf{s}_0))^2]$ leads directly to the **Simple Cokriging normal equations**:
$$
\boldsymbol{\Sigma} \boldsymbol{\lambda}_{\text{SCK}} = \mathbf{c}_{0,1}
$$
where $\mathbf{c}_{0,1} = (\mathbf{c}_{0,11}', \mathbf{c}_{0,12}', \dots, \mathbf{c}_{0,1p}')' \in \mathbb{R}^N$ is the cross-covariance vector between all observed data and the target variable $Z_1(\mathbf{s}_0)$, with $[\mathbf{c}_{0,1i}]_\alpha = C_{1i}(\mathbf{s}_0 - \mathbf{s}_{i\alpha}) = C_{i1}(\mathbf{s}_{i\alpha} - \mathbf{s}_0)$.

The minimized Simple Cokriging variance is:
$$
\sigma_{\text{SCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - \boldsymbol{\lambda}_{\text{SCK}}' \mathbf{c}_{0,1} = C_{11}(\mathbf{0}) - \mathbf{c}_{0,1}' \boldsymbol{\Sigma}^{-1} \mathbf{c}_{0,1}
$$

#### 2. Ordinary Cokriging (OCK)
When the constant means $\mu_1, \dots, \mu_p$ are unknown, the predictor must satisfy the unbiasedness condition:
$$
\operatorname{E}[\widehat{Z}_1(\mathbf{s}_0) - Z_1(\mathbf{s}_0)] = \operatorname{E}\left[ \sum_{i=1}^p \sum_{\alpha=1}^{n_i} \lambda_{i\alpha} Z_i(\mathbf{s}_{i\alpha}) - Z_1(\mathbf{s}_0) \right] = \sum_{i=1}^p \mu_i \left( \sum_{\alpha=1}^{n_i} \lambda_{i\alpha} \right) - \mu_1 = 0
$$
Because this identity must hold for all arbitrary, unknown mean values $(\mu_1, \dots, \mu_p) \in \mathbb{R}^p$, the coefficients of each $\mu_i$ must vanish independently. This establishes the **Ordinary Cokriging unbiasedness constraints**:
$$
\sum_{\alpha=1}^{n_1} \lambda_{1\alpha} = 1, \quad \text{and} \quad \sum_{\alpha=1}^{n_j} \lambda_{j\alpha} = 0 \quad \forall j = 2, \dots, p
$$

```
+-------------------------------------------------------------------------------------------------+
| THEOREM 7.3: Ordinary Cokriging BLUP System                                                     |
| The Best Linear Unbiased Predictor (BLUP) weights $\boldsymbol{\lambda}_{\text{OCK}}$ and       |
| Lagrange multipliers $\boldsymbol{\mu}_{\text{OCK}} = (\mu_1, \dots, \mu_p)'$ satisfy the       |
| $(N+p) \times (N+p)$ linear system:                                                             |
| $$                                                                                              |
| \begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix}     |
| \begin{bmatrix} \boldsymbol{\lambda}_{\text{OCK}} \\ -\boldsymbol{\mu}_{\text{OCK}} \end{bmatrix} |
| = \begin{bmatrix} \mathbf{c}_{0,1} \\ \mathbf{e}_1 \end{bmatrix}                                |
| $$                                                                                              |
| where $\mathbf{e}_1 = (1, 0, \dots, 0)' \in \mathbb{R}^p$, and $\mathbf{X} \in \mathbb{R}^{N \times p}$|
| is the block-diagonal constraint matrix:                                                        |
| $$                                                                                              |
| \mathbf{X} = \begin{bmatrix} \mathbf{1}_{n_1} & \mathbf{0} & \cdots & \mathbf{0} \\ \mathbf{0} & \mathbf{1}_{n_2} & \cdots & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{0} & \mathbf{0} & \cdots & \mathbf{1}_{n_p} \end{bmatrix} |
| $$                                                                                              |
+-------------------------------------------------------------------------------------------------+
```

For a bivariate system ($p=2$ attributes: primary target $Z_1$ and secondary auxiliary variable $Z_2$), geostatisticians typically express the Ordinary Cokriging system in semivariogram block form:
$$
\begin{bmatrix}
\boldsymbol{\Gamma}_{11} & \boldsymbol{\Gamma}_{12} & \mathbf{1} & \mathbf{0} \\
\boldsymbol{\Gamma}_{21} & \boldsymbol{\Gamma}_{22} & \mathbf{0} & \mathbf{1} \\
\mathbf{1}' & \mathbf{0}' & 0 & 0 \\
\mathbf{0}' & \mathbf{1}' & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{\lambda}_1 \\
\boldsymbol{\lambda}_2 \\
\mu_1 \\
\mu_2
\end{bmatrix}
=
\begin{bmatrix}
\boldsymbol{\gamma}_{11}(\mathbf{s}_0) \\
\boldsymbol{\gamma}_{12}(\mathbf{s}_0) \\
1 \\
0
\end{bmatrix}
$$

##### Term-by-Term Equation Anatomy: Bivariate Ordinary Cokriging Block Matrix System
$$
\underbrace{\begin{bmatrix}
\boldsymbol{\Gamma}_{11} & \boldsymbol{\Gamma}_{12} & \mathbf{1}_{n_1} & \mathbf{0}_{n_1} \\
\boldsymbol{\Gamma}_{21} & \boldsymbol{\Gamma}_{22} & \mathbf{0}_{n_2} & \mathbf{1}_{n_2} \\
\mathbf{1}_{n_1}' & \mathbf{0}_{n_2}' & 0 & 0 \\
\mathbf{0}_{n_1}' & \mathbf{1}_{n_2}' & 0 & 0
\end{bmatrix}}_{\substack{\text{Augmented Block Variogram System Matrix} \\ (n_1+n_2+2) \times (n_1+n_2+2)}}
\underbrace{\begin{bmatrix}
\boldsymbol{\lambda}_1 \\
\boldsymbol{\lambda}_2 \\
\mu_1 \\
\mu_2
\end{bmatrix}}_{\substack{\text{Cokriging Weights} \\ \text{\& Lagrange Multipliers}}}
=
\underbrace{\begin{bmatrix}
\boldsymbol{\gamma}_{11}(\mathbf{s}_0) \\
\boldsymbol{\gamma}_{12}(\mathbf{s}_0) \\
1 \\
0
\end{bmatrix}}_{\substack{\text{Target Semivariance Vectors} \\ \text{\& Unbiasedness Constants}}}
$$
- **$\boldsymbol{\Gamma}_{11} \in \mathbb{R}^{n_1 \times n_1}$ (Primary Auto-Semivariogram Block):** Quantifies spatial autocorrelation and sample redundancy among the $n_1$ primary sample coordinates. The $(\alpha, \beta)$-th entry is $[\boldsymbol{\Gamma}_{11}]_{\alpha, \beta} = \gamma_{11}(\mathbf{s}_{1,\alpha} - \mathbf{s}_{1,\beta})$. Diagonal entries are zero ($\gamma_{11}(\mathbf{0}) = 0$).
- **$\boldsymbol{\Gamma}_{12} \in \mathbb{R}^{n_1 \times n_2}$ and $\boldsymbol{\Gamma}_{21} = \boldsymbol{\Gamma}_{12}' \in \mathbb{R}^{n_2 \times n_1}$ (Primary-Secondary Cross-Semivariogram Blocks):** Quantifies the spatial cross-continuity connecting primary and secondary observations across sample locations: $[\boldsymbol{\Gamma}_{12}]_{\alpha, \beta} = \gamma_{12}(\mathbf{s}_{1,\alpha} - \mathbf{s}_{2,\beta})$. Off-diagonal cross-blocks transfer secondary spatial structure into the primary prediction.
- **$\boldsymbol{\Gamma}_{22} \in \mathbb{R}^{n_2 \times n_2}$ (Secondary Auto-Semivariogram Block):** Quantifies spatial redundancy among the $n_2$ secondary sample coordinates: $[\boldsymbol{\Gamma}_{22}]_{\alpha, \beta} = \gamma_{22}(\mathbf{s}_{2,\alpha} - \mathbf{s}_{2,\beta})$. Prevents densely clustered secondary samples from over-influencing the prediction.
- **$\mathbf{1}_{n_1}, \mathbf{0}_{n_1}, \mathbf{0}_{n_2}, \mathbf{1}_{n_2}$ (Unbiasedness Design Blocks):** Constraint vectors enforcing the dual unbiasedness requirements:
  - $\mathbf{1}_{n_1}'\boldsymbol{\lambda}_1 = \sum_{\alpha=1}^{n_1} \lambda_{1\alpha} = 1$: Forces primary weights to sum to unity so the expected primary prediction equals the unknown primary mean $\mu_1$.
  - $\mathbf{1}_{n_2}'\boldsymbol{\lambda}_2 = \sum_{\beta=1}^{n_2} \lambda_{2\beta} = 0$: Forces secondary weights to sum strictly to zero, ensuring the prediction is unaffected by the unknown secondary mean $\mu_2$ or arbitrary shift in secondary measurement units.
- **$\boldsymbol{\lambda}_1 \in \mathbb{R}^{n_1}$ and $\boldsymbol{\lambda}_2 \in \mathbb{R}^{n_2}$:** The optimal linear cokriging weights assigned to primary observations $Z_1(\mathbf{s}_{1,\alpha})$ and secondary observations $Z_2(\mathbf{s}_{2,\beta})$.
- **$\mu_1, \mu_2 \in \mathbb{R}$:** Scalar Lagrange multipliers enforcing the primary and secondary unbiasedness constraints, respectively.
- **$\boldsymbol{\gamma}_{11}(\mathbf{s}_0) \in \mathbb{R}^{n_1}$:** Point-to-target auto-semivariance vector between each primary sample location and target location $\mathbf{s}_0$, $[\boldsymbol{\gamma}_{11}(\mathbf{s}_0)]_\alpha = \gamma_{11}(\mathbf{s}_{1,\alpha} - \mathbf{s}_0)$.
- **$\boldsymbol{\gamma}_{12}(\mathbf{s}_0) \in \mathbb{R}^{n_2}$:** Point-to-target cross-semivariance vector between each secondary sample location and target location $\mathbf{s}_0$, $[\boldsymbol{\gamma}_{12}(\mathbf{s}_0)]_\beta = \gamma_{12}(\mathbf{s}_{2,\beta} - \mathbf{s}_0)$.
- **$1, 0$:** Scalar right-hand side unbiasedness targets, corresponding to $\mathbf{1}_{n_1}'\boldsymbol{\lambda}_1 = 1$ and $\mathbf{1}_{n_2}'\boldsymbol{\lambda}_2 = 0$.

#### Analytical Proof of Ordinary Cokriging System and Variance
We construct the Lagrangian function to minimize prediction error variance subject to the $p$ unbiasedness constraints:
$$
L(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \operatorname{Var}\left( \boldsymbol{\lambda}'\mathbf{Z} - Z_1(\mathbf{s}_0) \right) + 2\mu_1 \left( 1 - \mathbf{1}_{n_1}'\boldsymbol{\lambda}_1 \right) + 2\sum_{j=2}^p \mu_j \left( 0 - \mathbf{1}_{n_j}'\boldsymbol{\lambda}_j \right)
$$
Expanding the error variance:
$$
\operatorname{Var}(\boldsymbol{\lambda}'\mathbf{Z} - Z_1(\mathbf{s}_0)) = \boldsymbol{\lambda}' \boldsymbol{\Sigma} \boldsymbol{\lambda} - 2\boldsymbol{\lambda}' \mathbf{c}_{0,1} + C_{11}(\mathbf{0})
$$
In matrix notation, the $p$ constraints are $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{e}_1$. The Lagrangian is:
$$
L(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \boldsymbol{\lambda}' \boldsymbol{\Sigma} \boldsymbol{\lambda} - 2\boldsymbol{\lambda}' \mathbf{c}_{0,1} + C_{11}(\mathbf{0}) + 2\boldsymbol{\mu}' (\mathbf{e}_1 - \mathbf{X}'\boldsymbol{\lambda})
$$
Differentiating with respect to the weight vector $\boldsymbol{\lambda}$ and setting to zero:
$$
\frac{\partial L}{\partial \boldsymbol{\lambda}} = 2\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\mathbf{c}_{0,1} - 2\mathbf{X}\boldsymbol{\mu} = \mathbf{0} \implies \boldsymbol{\Sigma}\boldsymbol{\lambda} - \mathbf{X}\boldsymbol{\mu} = \mathbf{c}_{0,1}
$$
Differentiating with respect to the multiplier vector $\boldsymbol{\mu}$:
$$
\frac{\partial L}{\partial \boldsymbol{\mu}} = 2(\mathbf{e}_1 - \mathbf{X}'\boldsymbol{\lambda}) = \mathbf{0} \implies \mathbf{X}'\boldsymbol{\lambda} = \mathbf{e}_1
$$
Combining these two vector equations yields the $(N+p) \times (N+p)$ block system.

To derive the minimized Ordinary Cokriging prediction variance:
$$
\begin{aligned}
\sigma_{\text{OCK}}^2(\mathbf{s}_0) &= \boldsymbol{\lambda}' \boldsymbol{\Sigma} \boldsymbol{\lambda} - 2\boldsymbol{\lambda}' \mathbf{c}_{0,1} + C_{11}(\mathbf{0})
\end{aligned}
$$
From the first system equation, $\boldsymbol{\Sigma}\boldsymbol{\lambda} = \mathbf{c}_{0,1} + \mathbf{X}\boldsymbol{\mu}$. Multiplying by $\boldsymbol{\lambda}'$:
$$
\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} = \boldsymbol{\lambda}'\mathbf{c}_{0,1} + \boldsymbol{\lambda}'\mathbf{X}\boldsymbol{\mu} = \boldsymbol{\lambda}'\mathbf{c}_{0,1} + (\mathbf{X}'\boldsymbol{\lambda})'\boldsymbol{\mu}
$$
Since $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{e}_1$, we have $(\mathbf{X}'\boldsymbol{\lambda})'\boldsymbol{\mu} = \mathbf{e}_1'\boldsymbol{\mu} = \mu_1$.
Substituting back into the variance formula:
$$
\begin{aligned}
\sigma_{\text{OCK}}^2(\mathbf{s}_0) &= (\boldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1) - 2\boldsymbol{\lambda}'\mathbf{c}_{0,1} + C_{11}(\mathbf{0}) \\
&= C_{11}(\mathbf{0}) - \boldsymbol{\lambda}_{\text{OCK}}' \mathbf{c}_{0,1} + \mu_1
\end{aligned}
$$
This completes the proof. $\blacksquare$

---

#### Beat 2: Algorithmic Translation (The "How")

The following self-contained algorithm solves the Ordinary Cokriging system for a bivariate dataset without third-party black-box solvers, performing explicit block assembly and matrix inversion.

```python
def ordinary_cokriging_solver_bivariate(coords_1, vals_1, coords_2, vals_2, s0, 
                                        B_matrices, rho_models):
    """
    Solves the Ordinary Cokriging (OCK) system to predict Z_1(s0) from
    primary observations (coords_1, vals_1) and secondary observations (coords_2, vals_2).
    
    Parameters:
        coords_1: List of n_1 coordinates for primary variable.
        vals_1:   List of n_1 values for primary variable.
        coords_2: List of n_2 coordinates for secondary variable.
        vals_2:   List of n_2 values for secondary variable.
        s0:       Target coordinate vector for prediction.
        B_matrices: List of K coregionalization matrices (each 2x2).
        rho_models: List of K correlation functions rho_k(h).
        
    Returns:
        z1_hat: Predicted value of primary variable at s0.
        sigma2_ock: Ordinary Cokriging prediction variance.
        weights_1: Primary weights vector lambda_1 (sums to 1.0).
        weights_2: Secondary weights vector lambda_2 (sums to 0.0).
    """
    n1 = len(coords_1)
    n2 = len(coords_2)
    N = n1 + n2
    p = 2
    M = N + p  # Total matrix dimension
    
    # Step 1: Assemble the N x N covariance block matrix Sigma
    coords_by_var = [coords_1, coords_2]
    # Reuse assemble_multivariate_covariance_block_matrix logic
    K = len(B_matrices)
    Sigma = [[0.0 for _ in range(N)] for _ in range(N)]
    
    for i in range(2):
        start_i = 0 if i == 0 else n1
        n_i = n1 if i == 0 else n2
        c_i = coords_by_var[i]
        
        for j in range(2):
            start_j = 0 if j == 0 else n1
            n_j = n1 if j == 0 else n2
            c_j = coords_by_var[j]
            
            for a in range(n_i):
                for b in range(n_j):
                    # Distance between sample a of var i and sample b of var j
                    diff_sq = sum((c_i[a][d] - c_j[b][d])**2 for d in range(len(s0)))
                    h = diff_sq ** 0.5
                    val = sum(B_matrices[k][i][j] * rho_models[k](h) for k in range(K))
                    Sigma[start_i + a][start_j + b] = val
                    
    # Step 2: Build the augmented (N + 2) x (N + 2) system matrix A
    A = [[0.0 for _ in range(M)] for _ in range(M)]
    for r in range(N):
        for c in range(N):
            A[r][c] = Sigma[r][c]
            
    # Add constraint blocks: X and X^T
    # Var 1 constraint: sum(lambda_1) = 1
    for a in range(n1):
        A[a][N] = 1.0      # Column for -mu_1
        A[N][a] = 1.0      # Row for sum(lambda_1)
        
    # Var 2 constraint: sum(lambda_2) = 0
    for b in range(n2):
        A[n1 + b][N + 1] = 1.0  # Column for -mu_2
        A[N + 1][n1 + b] = 1.0  # Row for sum(lambda_2)
        
    # Step 3: Build the right-hand side vector rhs
    rhs = [0.0 for _ in range(M)]
    
    # c_0,11: Covariance between primary samples and target Z_1(s0)
    for a in range(n1):
        diff_sq = sum((coords_1[a][d] - s0[d])**2 for d in range(len(s0)))
        h = diff_sq ** 0.5
        rhs[a] = sum(B_matrices[k][0][0] * rho_models[k](h) for k in range(K))
        
    # c_0,12: Cross-covariance between secondary samples and target Z_1(s0)
    for b in range(n2):
        diff_sq = sum((coords_2[b][d] - s0[d])**2 for d in range(len(s0)))
        h = diff_sq ** 0.5
        rhs[n1 + b] = sum(B_matrices[k][1][0] * rho_models[k](h) for k in range(K))
        
    rhs[N] = 1.0      # Primary unbiasedness target
    rhs[N + 1] = 0.0  # Secondary unbiasedness target
    
    # Step 4: Solve A * sol = rhs via Gaussian elimination with partial pivoting
    # Make augmented copy for elimination
    aug = [row[:] + [rhs[i]] for i, row in enumerate(A)]
    for col in range(M):
        # Pivot selection
        max_row = col
        max_val = abs(aug[col][col])
        for r in range(col + 1, M):
            if abs(aug[r][col]) > max_val:
                max_val = abs(aug[r][col])
                max_row = r
        aug[col], aug[max_row] = aug[max_row], aug[col]
        
        pivot = aug[col][col]
        for c in range(col, M + 1):
            aug[col][c] /= pivot
            
        for r in range(M):
            if r != col:
                factor = aug[r][col]
                for c in range(col, M + 1):
                    aug[r][c] -= factor * aug[col][c]
                    
    sol = [aug[r][M] for r in range(M)]
    
    # Extract weights and multipliers
    weights_1 = sol[:n1]
    weights_2 = sol[n1:N]
    mu_1 = -sol[N]
    mu_2 = -sol[N + 1]
    
    # Step 5: Compute prediction and prediction variance
    z1_hat = sum(weights_1[a] * vals_1[a] for a in range(n1)) + \
             sum(weights_2[b] * vals_2[b] for b in range(n2))
             
    c00 = sum(B_matrices[k][0][0] * rho_models[k](0.0) for k in range(K))
    dot_lam_c0 = sum(sol[i] * rhs[i] for i in range(N))
    sigma2_ock = c00 - dot_lam_c0 - mu_1
    
    return z1_hat, sigma2_ock, weights_1, weights_2
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

1. **Why Secondary Weights Must Sum to Zero:**
   In Ordinary Cokriging, the constraint $\sum_{\beta=1}^{n_2} \lambda_{2\beta} = 0$ is unavoidable when secondary mean $\mu_2$ is unknown. If the secondary weights did not sum to zero, a shift in the measurement scale of the secondary variable (e.g., measuring elevation in centimeters instead of meters) would alter the primary prediction $\widehat{Z}_1(\mathbf{s}_0)$!
2. **The Mechanism of Negative Secondary Weights:**
   Because the secondary weights sum to zero, **they cannot all be non-negative** (unless they are all identically zero). The secondary weights inevitably form dipoles: positive weights on secondary data close to $\mathbf{s}_0$, counterbalanced by negative weights on secondary data located at primary sampling sites.
   In essence, the secondary variable acts as a **spatial gradient corrector**:
   $$
   \sum_{\beta=1}^{n_2} \lambda_{2\beta} Z_2(\mathbf{s}_{2\beta}) \approx c \cdot \nabla Z_2(\mathbf{s}_0) \cdot \Delta \mathbf{s}
   $$
   It transfers the local slope of the secondary variable to refine the primary estimate.

---

### 7.4 Collocated Cokriging and Markov Screening Approximations

#### Beat 1: The Mathematical Anchor (The "Why")

When secondary information is derived from geophysical surveys or satellite imaging, the number of secondary observations $n_2$ can exceed $10^5$ to $10^7$ points. Solving an Ordinary Cokriging system of size $(n_1 + n_2 + 2) \times (n_1 + n_2 + 2)$ becomes computationally intractable.

To circumvent this curse of dimensionality, geostatisticians use **Collocated Cokriging**, which retains only a small neighborhood of primary data along with a single collocated secondary measurement $Z_2(\mathbf{s}_0)$ evaluated directly at the prediction location $\mathbf{s}_0$.

However, retaining only $Z_2(\mathbf{s}_0)$ creates a theoretical paradox when using standard Ordinary Cokriging.

```
+-------------------------------------------------------------------------------------------------+
| THEOREM 7.4: The Collocated Ordinary Cokriging Paradox                                          |
| If Ordinary Cokriging is applied using $n_1$ primary data and a single collocated secondary     |
| datum $Z_2(\mathbf{s}_0)$, the secondary weight $\lambda_2(\mathbf{s}_0)$ is forced to be:     |
| $$                                                                                              |
| \lambda_2(\mathbf{s}_0) \equiv 0                                                               |
| $$                                                                                              |
| Standard Ordinary Cokriging completely ignores the collocated secondary datum!                  |
+-------------------------------------------------------------------------------------------------+
```

**Proof:**
The secondary unbiasedness constraint requires:
$$
\sum_{\beta=1}^{n_2} \lambda_{2\beta} = 0
$$
When $n_2 = 1$ (only the collocated point $Z_2(\mathbf{s}_0)$ is used), the summation contains only a single term:
$$
\lambda_2(\mathbf{s}_0) = 0
$$
Thus, the single secondary weight is algebraically forced to zero. The system reduces identically to univariate Ordinary Kriging of $Z_1$. $\blacksquare$

#### Resolving the Paradox: Standardized Collocated Simple Cokriging
To utilize collocated secondary data effectively, both variables must be standardized to zero mean and unit variance:
$$
Y_1(\mathbf{s}) = \frac{Z_1(\mathbf{s}) - \mu_1}{\sigma_1}, \quad Y_2(\mathbf{s}) = \frac{Z_2(\mathbf{s}) - \mu_2}{\sigma_2}
$$
Under **Markov Model 1 (MM1)** (Journel, 1999), we assume that the collocated secondary datum $Y_2(\mathbf{s}_0)$ screens the target $Y_1(\mathbf{s}_0)$ from all other secondary data $Y_2(\mathbf{s})$ ($\mathbf{s} \ne \mathbf{s}_0$):
$$
\operatorname{Cov}(Y_1(\mathbf{s}_0), Y_2(\mathbf{s}) \mid Y_2(\mathbf{s}_0)) = 0 \quad \forall \mathbf{s} \ne \mathbf{s}_0
$$
This conditional independence hypothesis implies that the cross-correlation function is strictly proportional to the secondary auto-correlation:
$$
\rho_{12}(\mathbf{h}) = \rho_{12}(\mathbf{0}) \rho_{22}(\mathbf{h})
$$
The standardized collocated Simple Cokriging system size is $(n_1 + 1) \times (n_1 + 1)$:
$$
\begin{bmatrix}
\mathbf{R}_{11} & \mathbf{r}_{12}(\mathbf{s}_0) \\
\mathbf{r}_{12}(\mathbf{s}_0)' & 1
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{\lambda}_1 \\
\lambda_2(\mathbf{s}_0)
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{r}_{0,11} \\
\rho_{12}(\mathbf{0})
\end{bmatrix}
$$
Because there is no zero-sum constraint on $\lambda_2(\mathbf{s}_0)$, $\lambda_2(\mathbf{s}_0) \ne 0$, and the secondary information is successfully incorporated.

---

#### Beat 2: Algorithmic Translation (The "How")

```python
def standardized_collocated_cokriging(coords_1, vals_1, s0, val_2_s0,
                                      rho11_func, rho22_func, rho12_0, 
                                      mu1, sigma1, mu2, sigma2):
    """
    Standardized Collocated Simple Cokriging using Markov Model 1 (MM1).
    
    Parameters:
        coords_1: Primary sample locations [s_1, ..., s_n1].
        vals_1:   Primary sample measurements.
        s0:       Target prediction location.
        val_2_s0: Collocated secondary measurement Z_2(s0).
        rho11_func: Correlation function for primary variable.
        rho22_func: Correlation function for secondary variable.
        rho12_0:  Collocated correlation coefficient Cor(Z_1(s), Z_2(s)).
        mu1, sigma1: Known or estimated mean and std dev of primary variable.
        mu2, sigma2: Known or estimated mean and std dev of secondary variable.
        
    Returns:
        z1_pred: Predicted primary value at s0.
        pred_var: Collocated cokriging prediction variance.
    """
    n1 = len(coords_1)
    
    # Step 1: Standardize observations to zero mean and unit variance
    y1 = [(vals_1[i] - mu1) / sigma1 for i in range(n1)]
    y2_0 = (val_2_s0 - mu2) / sigma2
    
    # Step 2: Build the (n1 + 1) x (n1 + 1) correlation matrix R
    R = [[0.0 for _ in range(n1 + 1)] for _ in range(n1 + 1)]
    
    # Block R_11 (n1 x n1)
    for i in range(n1):
        for j in range(n1):
            h = sum((coords_1[i][d] - coords_1[j][d])**2 for d in range(len(s0))) ** 0.5
            R[i][j] = rho11_func(h)
            
    # Vector r_12(s0): cross-correlation between primary samples and collocated secondary
    # Under MM1: rho_12(h) = rho_12(0) * rho_22(h)
    for i in range(n1):
        h = sum((coords_1[i][d] - s0[d])**2 for d in range(len(s0))) ** 0.5
        cross_corr = rho12_0 * rho22_func(h)
        R[i][n1] = cross_corr
        R[n1][i] = cross_corr
        
    # Diagonal entry for collocated secondary datum
    R[n1][n1] = 1.0
    
    # Step 3: Build the right-hand side vector rhs
    rhs = [0.0 for _ in range(n1 + 1)]
    for i in range(n1):
        h = sum((coords_1[i][d] - s0[d])**2 for d in range(len(s0))) ** 0.5
        rhs[i] = rho11_func(h)
    rhs[n1] = rho12_0  # Corr(Y_1(s0), Y_2(s0))
    
    # Step 4: Solve R * lam = rhs (Gaussian elimination)
    M = n1 + 1
    aug = [row[:] + [rhs[i]] for i, row in enumerate(R)]
    for col in range(M):
        max_r = col
        for r in range(col + 1, M):
            if abs(aug[r][col]) > abs(aug[max_r][col]):
                max_r = r
        aug[col], aug[max_r] = aug[max_r], aug[col]
        
        pivot = aug[col][col]
        for c in range(col, M + 1):
            aug[col][c] /= pivot
        for r in range(M):
            if r != col:
                f = aug[r][col]
                for c in range(col, M + 1):
                    aug[r][c] -= f * aug[col][c]
                    
    lam = [aug[r][M] for r in range(M)]
    
    # Step 5: Compute standardized prediction and back-transform
    y1_pred = sum(lam[i] * y1[i] for i in range(n1)) + lam[n1] * y2_0
    z1_pred = mu1 + sigma1 * y1_pred
    
    # Standardized prediction variance: 1 - lam' * rhs
    norm_var = 1.0 - sum(lam[i] * rhs[i] for i in range(M))
    pred_var = (sigma1 ** 2) * max(0.0, norm_var)
    
    return z1_pred, pred_var
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

1. **The "Variance Bug" of Collocated Cokriging:**
   In collocated cokriging, the secondary datum is always evaluated at distance zero from the target ($h=0$). Thus, its correlation with the target is always maximal ($\rho_{12}(0)$), regardless of whether $\mathbf{s}_0$ is surrounded by dense primary data or located in an unsampled desert. 
   Consequently, the collocated cokriging variance $\sigma_{\text{collocated}}^2(\mathbf{s}_0)$ displays an artificial global depression across the entire domain, failing to reflect true spatial information voids.
2. **Matrix Singularity near Primary Data:**
   As target location $\mathbf{s}_0$ approaches a primary data site $\mathbf{s}_{1,k}$ ($\|\mathbf{s}_0 - \mathbf{s}_{1,k}\| \to 0$), the collocated datum $Z_2(\mathbf{s}_0)$ becomes co-located with $Z_1(\mathbf{s}_{1,k})$. If $\rho_{12}(0)$ is large, column $k$ and column $n_1+1$ in $\mathbf{R}$ become highly collinear, causing numerical instability.

---

### 7.5 The Autokrigeability Theorem

#### Beat 1: The Mathematical Anchor (The "Why")

A central foundational question in multivariate geostatistics is: *Under what mathematical conditions does the inclusion of secondary variables provide zero additional predictive benefit over univariate kriging?*

The celebrated **Autokrigeability Theorem** provides the exact necessary and sufficient conditions.

```
+-------------------------------------------------------------------------------------------------+
| THEOREM 7.5: The Autokrigeability Theorem (Wackernagel, 2003)                                   |
| Suppose a multivariate random field $\{\mathbf{Z}(\mathbf{s}) : \mathbf{s} \in D\}$ satisfies: |
| 1. **Intrinsic Coregionalization Model (ICM):** All auto- and cross-covariances share the same   |
|    spatial correlation structure:                                                               |
|    $$                                                                                           |
|    \mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(\mathbf{h})                                         |
|    $$                                                                                           |
|    where $\mathbf{B} \in \mathbb{R}^{p \times p}$ is positive semi-definite and $\rho(\mathbf{h})$ is valid.|
| 2. **Isotopic Sampling:** All $p$ variables are measured at the identical set of $n$ locations: |
|    $$                                                                                           |
|    \mathbf{s}_{1,\alpha} = \mathbf{s}_{2,\alpha} = \cdots = \mathbf{s}_{p,\alpha} = \mathbf{s}_\alpha \quad \forall \alpha=1, \dots, n |
|    $$                                                                                           |
| Then for any target location $\mathbf{s}_0 \in D$:                                              |
| 1. The Ordinary Cokriging weights on all secondary variables are IDENTICALLY ZERO:             |
|    $$                                                                                           |
|    \boldsymbol{\lambda}_j^{\text{OCK}} = \mathbf{0} \quad \forall j = 2, \dots, p               |
|    $$                                                                                           |
| 2. The Ordinary Cokriging weights on the primary variable equal the univariate Ordinary Kriging weights: |
|    $$                                                                                           |
|    \boldsymbol{\lambda}_1^{\text{OCK}} = \boldsymbol{\lambda}_{\text{OK}}^*                     |
|    $$                                                                                           |
| 3. The Ordinary Cokriging prediction variance is IDENTICAL to the univariate Ordinary Kriging variance: |
|    $$                                                                                           |
|    \sigma_{\text{OCK}}^2(\mathbf{s}_0) = \sigma_{\text{OK}}^2(\mathbf{s}_0)                    |
|    $$                                                                                           |
+-------------------------------------------------------------------------------------------------+
```

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *What is "Autokrigeability" and why does secondary data completely drop out when variables share the same spatial correlation structure?*  
> When all attributes are sampled at the exact same spatial locations (isotopic sampling) and share an identical spatial continuity pattern—meaning all coregionalization matrices $\mathbf{B}_k$ are proportional to a single matrix $\mathbf{B}$, or the system is governed by a single correlation function $\rho(\mathbf{h})$ (an Intrinsic Coregionalization Model)—the covariance matrix factorizes into a Kronecker product $\boldsymbol{\Sigma} = \mathbf{B} \otimes \mathbf{R}$. Because the spatial distance configuration between sample locations and the target $\mathbf{s}_0$ is identical for both primary and secondary variables, the spatial information conveyed by the secondary measurements is completely redundant with the primary measurements. When the Ordinary Cokriging Lagrangian system is solved under the constraint $\sum_{\beta=1}^n \lambda_{j\beta} = 0$, the secondary weights solve uniquely to:
> $$
> \boldsymbol{\lambda}_j^{\text{OCK}} = \mathbf{0} \quad \forall j \ge 2
> $$
> Secondary measurements receive exactly zero weight! Cokriging collapses identically to univariate Ordinary Kriging on the primary variable ($\boldsymbol{\lambda}_1^{\text{OCK}} = \boldsymbol{\lambda}_{\text{OK}}^*$), yielding the exact same prediction and estimation variance ($\sigma_{\text{OCK}}^2(\mathbf{s}_0) = \sigma_{\text{OK}}^2(\mathbf{s}_0)$). Secondary data can only improve predictions when:
> 1. Sampling is **heterotopic** (secondary data exists in spatial voids where primary data was not measured), or
> 2. The cross-correlation structure is **multi-scale** (variables exhibit different spatial ranges or negative cross-correlation at certain spatial scales, breaking proportionality).

#### Analytical Proof via Kronecker Product Algebra
We prove the theorem using the algebraic properties of the Kronecker product $\otimes$.

**Step 1: Express the Multivariate System in Kronecker Form**
Under the Intrinsic Coregionalization Model and isotopic sampling:
- The $n \times n$ spatial correlation matrix among the $n$ sampling locations is $\mathbf{R} = [\rho(\mathbf{s}_\alpha - \mathbf{s}_\beta)]$.
- The global $pn \times pn$ covariance matrix $\boldsymbol{\Sigma}$ factorizes into the Kronecker product:
  $$
  \boldsymbol{\Sigma} = \mathbf{B} \otimes \mathbf{R}
  $$
- The constraint matrix $\mathbf{X} \in \mathbb{R}^{pn \times p}$ factorizes as:
  $$
  \mathbf{X} = \mathbf{I}_p \otimes \mathbf{1}_n
  $$
- The cross-covariance vector with target $Z_1(\mathbf{s}_0)$ is:
  $$
  \mathbf{c}_{0,1} = (\mathbf{B}\mathbf{e}_1) \otimes \mathbf{r}_0
  $$
  where $\mathbf{e}_1 = (1, 0, \dots, 0)' \in \mathbb{R}^p$ and $\mathbf{r}_0 = (\rho(\mathbf{s}_1 - \mathbf{s}_0), \dots, \rho(\mathbf{s}_n - \mathbf{s}_0))'$.

**Step 2: Propose Candidate Solution**
Let $\boldsymbol{\lambda}^* \in \mathbb{R}^n$ and $\mu^* \in \mathbb{R}$ be the unique solution to the univariate Ordinary Kriging system for variable 1:
$$
\begin{bmatrix} B_{11} \mathbf{R} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda}^* \\ -\mu^* \end{bmatrix} = \begin{bmatrix} B_{11} \mathbf{r}_0 \\ 1 \end{bmatrix}
$$
Dividing by $B_{11} > 0$ and defining $\nu^* = \mu^* / B_{11}$:
$$
\mathbf{R} \boldsymbol{\lambda}^* - \nu^* \mathbf{1}_n = \mathbf{r}_0, \quad \text{and} \quad \mathbf{1}_n' \boldsymbol{\lambda}^* = 1
$$
We propose the candidate multivariate solution:
$$
\boldsymbol{\lambda} = \mathbf{e}_1 \otimes \boldsymbol{\lambda}^*, \quad \text{and} \quad \boldsymbol{\mu}_{\text{OCK}} = \nu^* \mathbf{B} \mathbf{e}_1
$$
In block form, this candidate sets $\boldsymbol{\lambda}_1 = \boldsymbol{\lambda}^*$ and $\boldsymbol{\lambda}_j = \mathbf{0}$ for all $j \ge 2$.

**Step 3: Verify the Candidate Satisfies the Full Ordinary Cokriging System**
Recall the mixed-product property of Kronecker products: $(\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A}\mathbf{C}) \otimes (\mathbf{B}\mathbf{D})$.

1. Evaluate the first block equation $\boldsymbol{\Sigma}\boldsymbol{\lambda} - \mathbf{X}\boldsymbol{\mu}_{\text{OCK}}$:
   $$
   \begin{aligned}
   \boldsymbol{\Sigma} \boldsymbol{\lambda} &= (\mathbf{B} \otimes \mathbf{R})(\mathbf{e}_1 \otimes \boldsymbol{\lambda}^*) = (\mathbf{B}\mathbf{e}_1) \otimes (\mathbf{R}\boldsymbol{\lambda}^*) \\
   \mathbf{X}\boldsymbol{\mu}_{\text{OCK}} &= (\mathbf{I}_p \otimes \mathbf{1}_n)(\nu^* \mathbf{B}\mathbf{e}_1) = (\mathbf{B}\mathbf{e}_1) \otimes (\nu^* \mathbf{1}_n)
   \end{aligned}
   $$
   Subtracting the two terms:
   $$
   \boldsymbol{\Sigma}\boldsymbol{\lambda} - \mathbf{X}\boldsymbol{\mu}_{\text{OCK}} = (\mathbf{B}\mathbf{e}_1) \otimes (\mathbf{R}\boldsymbol{\lambda}^* - \nu^* \mathbf{1}_n)
   $$
   By the univariate OK equation, $\mathbf{R}\boldsymbol{\lambda}^* - \nu^*\mathbf{1}_n = \mathbf{r}_0$. Therefore:
   $$
   \boldsymbol{\Sigma}\boldsymbol{\lambda} - \mathbf{X}\boldsymbol{\mu}_{\text{OCK}} = (\mathbf{B}\mathbf{e}_1) \otimes \mathbf{r}_0 = \mathbf{c}_{0,1}
   $$
   The first equation is satisfied exactly!

2. Evaluate the constraint equation $\mathbf{X}'\boldsymbol{\lambda}$:
   $$
   \mathbf{X}'\boldsymbol{\lambda} = (\mathbf{I}_p \otimes \mathbf{1}_n')(\mathbf{e}_1 \otimes \boldsymbol{\lambda}^*) = (\mathbf{I}_p \mathbf{e}_1) \otimes (\mathbf{1}_n' \boldsymbol{\lambda}^*) = \mathbf{e}_1 \otimes 1 = \mathbf{e}_1
   $$
   The constraint equation is satisfied exactly!

Because the Ordinary Cokriging matrix is non-singular, the solution is unique. Hence, the candidate is the unique solution:
$$
\boldsymbol{\lambda}_1 = \boldsymbol{\lambda}^*, \quad \text{and} \quad \boldsymbol{\lambda}_j = \mathbf{0} \quad \forall j \ge 2
$$

**Step 4: Verify Equivalence of Prediction Variances**
The Ordinary Cokriging variance is:
$$
\sigma_{\text{OCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1
$$
Substituting $\boldsymbol{\lambda} = \mathbf{e}_1 \otimes \boldsymbol{\lambda}^*$ and $\mu_1 = [\boldsymbol{\mu}_{\text{OCK}}]_1 = \nu^* B_{11} = \mu^*$:
$$
\boldsymbol{\lambda}'\mathbf{c}_{0,1} = (\mathbf{e}_1 \otimes \boldsymbol{\lambda}^*)' ((\mathbf{B}\mathbf{e}_1) \otimes \mathbf{r}_0) = (\mathbf{e}_1' \mathbf{B} \mathbf{e}_1) (\boldsymbol{\lambda}^{*\prime} \mathbf{r}_0) = B_{11} (\boldsymbol{\lambda}^{*\prime}\mathbf{r}_0)
$$
Thus:
$$
\sigma_{\text{OCK}}^2(\mathbf{s}_0) = B_{11}\rho(\mathbf{0}) - B_{11}\boldsymbol{\lambda}^{*\prime}\mathbf{r}_0 + \mu^* = \sigma_{\text{OK}}^2(\mathbf{s}_0)
$$
This completes the unabridged proof. $\blacksquare$

---

#### Beat 2: Algorithmic Translation (The "How")

The following diagnostic procedure tests whether a proposed multivariate configuration satisfies the conditions of the Autokrigeability Theorem before the analyst wastes computational resources on large-scale cokriging.

```python
def autokrigeability_diagnostic(B_matrices, rho_models, is_isotopic):
    """
    Diagnoses whether an Ordinary Cokriging problem satisfies the
    Autokrigeability Theorem, rendering secondary data redundant.
    
    Parameters:
        B_matrices: List of K coregionalization matrices.
        rho_models: List of K correlation functions.
        is_isotopic: Boolean indicating if all variables are sampled at identical sites.
        
    Returns:
        is_autokrigeable: Boolean.
        explanation: Diagnostic explanation string.
    """
    if not is_isotopic:
        return False, "Sampling is heterotopic; secondary data provides spatial coverage in voids."
        
    K = len(B_matrices)
    if K == 1:
        # Single spatial structure: strictly an Intrinsic Coregionalization Model (ICM)
        return True, "SYSTEM IS AUTOKRIGEABLE: Pure ICM with isotopic data. " \
                     "Secondary weights will be 0. Univariate kriging is mathematically optimal."
                     
    # Check if multiple structures have proportional coregionalization matrices
    # B_k = c_k * B_1
    p = len(B_matrices[0])
    is_proportional = True
    for k in range(1, K):
        # Estimate scaling constant c_k = B_k[0,0] / B_0[0,0]
        if B_matrices[0][0][0] == 0:
            is_proportional = False
            break
        c_k = B_matrices[k][0][0] / B_matrices[0][0][0]
        for i in range(p):
            for j in range(p):
                if abs(B_matrices[k][i][j] - c_k * B_matrices[0][i][j]) > 1e-6:
                    is_proportional = False
                    break
                    
    if is_proportional:
        return True, "SYSTEM IS AUTOKRIGEABLE: Proportional coregionalization matrices. " \
                     "Univariate kriging yields identical estimates."
    else:
        return False, "System is NOT autokrigeable: Multiple spatial scales with differing " \
                      "correlation structures. Cokriging will improve prediction."
```

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification (The "Reality")

```
                            THE AUTOKRIGEABILITY DILEMMA
           Does Cokriging improve upon univariate Ordinary Kriging?

                     Is sampling configuration Isotopic?
                               /             \
                             YES              NO  ====>  COKRIGING ADDS VALUE
                             /                           (Secondary data fills voids)
              Do variables share a single
              spatial correlation range?
                    /             \
                  YES              NO  =============>  COKRIGING ADDS VALUE
                  /                                     (Different spatial scales)
    AUTOKRIGEABILITY HOLDS!
    Secondary weights = 0
    Cokriging == Univariate OK
```

**The Practitioner's Fallacy:**
A common blunder among spatial analysts is to spend weeks fitting a complex Linear Model of Coregionalization to isotopic geochemical data where gold, silver, and copper share the exact same variogram range, only to discover that the cokriging map is pixel-for-pixel identical to the univariate kriging map!
- **When is Cokriging truly worth the effort?**
  1. **Heterotopic Sampling:** When the secondary variable is sampled at thousands of locations where the primary variable is absent (e.g., densely sampled digital elevation models guiding sparse rainfall gauges).
  2. **Multi-Scale Coregionalization:** When variables interact differently across spatial scales (e.g., high correlation at regional scales due to geology, but negative correlation at local scales due to geochemical scavenging).

---

### 7.6 Analytical Exercises and Step-by-Step Solutions

#### Exercise 7.1: Sylvester's Criterion and Coregionalization Admissibility
**Problem Statement:**
A geostatistician fits an empirical bivariate LMC with a single spherical structure:
$$
\mathbf{C}(\mathbf{h}) = \mathbf{B} \rho(h), \quad \text{where } \mathbf{B} = \begin{bmatrix} 4.0 & 5.0 \\ 5.0 & 6.0 \end{bmatrix}
$$
where $\rho(h) = (1 - 1.5(h/10) + 0.5(h/10)^3)\mathbb{I}_{\{h \le 10\}}$.
**(a)** Determine whether this matrix covariance model is mathematically admissible in $\mathbb{R}^2$.
**(b)** Construct a spatial point configuration where this model produces a negative prediction variance.

**Complete Step-by-Step Solution:**
**(a)** 
1. By Theorem 7.2, an LMC model $\mathbf{C}(\mathbf{h}) = \mathbf{B}\rho(h)$ is positive semi-definite if and only if $\rho(h)$ is a valid correlation function on $\mathbb{R}^2$ and $\mathbf{B}$ is positive semi-definite ($\mathbf{B} \succeq 0$).
2. The spherical model in $\mathbb{R}^2$ is valid since $d=2 \le 3$.
3. We test positive semi-definiteness of $\mathbf{B}$ via Sylvester's Criterion:
   - Principal minor 1: $M_1 = B_{11} = 4.0 > 0$.
   - Principal minor 2: $M_2 = \det(\mathbf{B}) = B_{11} B_{22} - (B_{12})^2 = (4.0)(6.0) - (5.0)^2 = 24.0 - 25.0 = -1.0 < 0$.
4. Because the determinant $\det(\mathbf{B}) = -1.0 < 0$, $\mathbf{B}$ has one positive eigenvalue and one negative eigenvalue:
   $$
   \operatorname{Tr}(\mathbf{B}) = 4 + 6 = 10, \quad \det(\mathbf{B}) = -1 \implies \lambda = \frac{10 \pm \sqrt{100 - 4(1)(-1)}}{2} = \frac{10 \pm \sqrt{104}}{2} \approx 5 \pm 5.099
   $$
   $\lambda_1 \approx 10.099 > 0$, $\lambda_2 \approx -0.099 < 0$.
5. Because $\mathbf{B}$ has a negative eigenvalue, $\mathbf{C}(\mathbf{h})$ is **NOT positive semi-definite**, and is strictly mathematically inadmissible. $\blacksquare$

**(b)**
1. To obtain a negative variance, we construct a linear combination of co-located observations $Y = a_1 Z_1(\mathbf{s}) + a_2 Z_2(\mathbf{s}) = \mathbf{a}'\mathbf{Z}(\mathbf{s})$ using the eigenvector $\mathbf{v}_2$ corresponding to the negative eigenvalue $\lambda_2 \approx -0.099$.
2. Solving $(\mathbf{B} - \lambda_2 \mathbf{I})\mathbf{v}_2 = \mathbf{0}$:
   $$
   (4.0 - (-0.1)) a_1 + 5.0 a_2 = 0 \implies 4.1 a_1 + 5.0 a_2 = 0 \implies a_1 = 5.0, \quad a_2 = -4.1
   $$
3. Compute the variance of $Y = 5.0 Z_1(\mathbf{s}) - 4.1 Z_2(\mathbf{s})$:
   $$
   \begin{aligned}
   \operatorname{Var}(Y) &= a_1^2 \operatorname{Var}(Z_1) + a_2^2 \operatorname{Var}(Z_2) + 2 a_1 a_2 \operatorname{Cov}(Z_1, Z_2) \\
   &= (5.0)^2 (4.0) + (-4.1)^2 (6.0) + 2(5.0)(-4.1)(5.0) \\
   &= 25.0(4.0) + 16.81(6.0) - 205.0 \\
   &= 100.0 + 100.86 - 205.0 = -4.14 < 0
   \end{aligned}
   $$
4. The variance is $-4.14$, which violates Kolmogorov Axiom 2. $\blacksquare$

---

#### Exercise 7.2: Unabridged Proof of the Autokrigeability Theorem for a Bivariate System ($p=2$)
**Problem Statement:**
Let $\{Z_1(\mathbf{s}), Z_2(\mathbf{s})\}$ be an intrinsically coregionalized bivariate field with $\mathbf{C}(\mathbf{h}) = \mathbf{B}\rho(\mathbf{h})$, observed at identical locations $\mathbf{s}_1, \dots, \mathbf{s}_n$. 
Prove algebraically from the scalar Ordinary Cokriging equations that the secondary weights satisfy $\lambda_{2\alpha} = 0$ for all $\alpha = 1, \dots, n$.

**Complete Step-by-Step Solution:**
1. The bivariate Ordinary Cokriging system consists of two vector equations and two scalar constraint equations:
   $$
   \begin{aligned}
   B_{11} \mathbf{R} \boldsymbol{\lambda}_1 + B_{12} \mathbf{R} \boldsymbol{\lambda}_2 - \mu_1 \mathbf{1}_n &= B_{11} \mathbf{r}_0 \quad \text{--- (Eq. 1)} \\
   B_{21} \mathbf{R} \boldsymbol{\lambda}_1 + B_{22} \mathbf{R} \boldsymbol{\lambda}_2 - \mu_2 \mathbf{1}_n &= B_{21} \mathbf{r}_0 \quad \text{--- (Eq. 2)} \\
   \mathbf{1}_n' \boldsymbol{\lambda}_1 &= 1 \quad \text{--- (Eq. 3)} \\
   \mathbf{1}_n' \boldsymbol{\lambda}_2 &= 0 \quad \text{--- (Eq. 4)}
   \end{aligned}
   $$
   where $\mathbf{R} = [\rho(\mathbf{s}_\alpha - \mathbf{s}_\beta)] \in \mathbb{R}^{n \times n}$ and $\mathbf{r}_0 = (\rho(\mathbf{s}_1 - \mathbf{s}_0), \dots, \rho(\mathbf{s}_n - \mathbf{s}_0))'$.
2. Multiply Eq. 1 by $B_{21}$ and Eq. 2 by $B_{11}$:
   $$
   \begin{aligned}
   B_{21} B_{11} \mathbf{R} \boldsymbol{\lambda}_1 + B_{21} B_{12} \mathbf{R} \boldsymbol{\lambda}_2 - B_{21} \mu_1 \mathbf{1}_n &= B_{21} B_{11} \mathbf{r}_0 \\
   B_{11} B_{21} \mathbf{R} \boldsymbol{\lambda}_1 + B_{11} B_{22} \mathbf{R} \boldsymbol{\lambda}_2 - B_{11} \mu_2 \mathbf{1}_n &= B_{11} B_{21} \mathbf{r}_0
   \end{aligned}
   $$
3. Subtract the first equation from the second:
   $$
   (B_{11} B_{22} - B_{12}^2) \mathbf{R} \boldsymbol{\lambda}_2 - (B_{11} \mu_2 - B_{12} \mu_1) \mathbf{1}_n = \mathbf{0}
   $$
4. Let $\det(\mathbf{B}) = B_{11} B_{22} - B_{12}^2 > 0$ (assuming $\mathbf{B}$ is strictly positive definite). Define scalar constant $c = \frac{B_{11} \mu_2 - B_{12} \mu_1}{\det(\mathbf{B})}$. The equation simplifies to:
   $$
   \mathbf{R} \boldsymbol{\lambda}_2 = c \mathbf{1}_n
   $$
5. Multiply by $\mathbf{R}^{-1}$:
   $$
   \boldsymbol{\lambda}_2 = c \mathbf{R}^{-1} \mathbf{1}_n
   $$
6. Apply the secondary unbiasedness constraint (Eq. 4: $\mathbf{1}_n' \boldsymbol{\lambda}_2 = 0$):
   $$
   \mathbf{1}_n' \boldsymbol{\lambda}_2 = c (\mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n) = 0
   $$
7. Because $\mathbf{R}$ is a strictly positive definite matrix, its inverse $\mathbf{R}^{-1}$ is also strictly positive definite. For any non-zero vector, the quadratic form is strictly positive:
   $$
   \mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n > 0
   $$
8. Therefore, the only way $c (\mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n) = 0$ can hold is if:
   $$
   c = 0
   $$
9. Substituting $c = 0$ back into Step 5:
   $$
   \boldsymbol{\lambda}_2 = 0 \cdot \mathbf{R}^{-1} \mathbf{1}_n = \mathbf{0}
   $$
10. Since $\boldsymbol{\lambda}_2 = \mathbf{0}$, Eq. 1 reduces to $B_{11} \mathbf{R} \boldsymbol{\lambda}_1 - \mu_1 \mathbf{1}_n = B_{11} \mathbf{r}_0$, which is identically the univariate Ordinary Kriging system. $\blacksquare$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 7: Multivariate Geostatistics and Cokriging Theory — Companion Exercises and Complete Solutions](ch07_exercises.md)
