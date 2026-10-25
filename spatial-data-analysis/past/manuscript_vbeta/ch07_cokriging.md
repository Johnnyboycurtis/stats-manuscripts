# Chapter 7: Multivariate Spatial Prediction and Cokriging

In many environmental, geophysical, geological, and ecological investigations, scientists rarely sample a single spatial variable in isolation. Mining engineers evaluate gold alongside silver and copper; hydrologists analyze piezometric water table depth alongside topography, precipitation, and soil permeability; soil scientists measure organic matter alongside electrical conductivity and clay content. 

In spatial linear models, when multiple cross-correlated variables are observed across a spatial domain $D \subset \mathbb{R}^d$, predicting an expensive or sparsely sampled primary variable $Z_1(\mathbf{s}_0)$ can often be substantially improved by incorporating auxiliary secondary variables $\{Z_2, \dots, Z_p\}$. The collection of optimal linear spatial prediction techniques for multivariate random fields is termed **cokriging**.

---

### Learning Outcomes
After studying this chapter, you will be able to:
1. **Differentiate Cross-Covariance Structures:** Distinguish between univariate auto-covariance and multivariate cross-covariance functions, verifying transpose symmetry $\mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})'$ and Cauchy–Schwarz bounds.
2. **Construct Valid Coregionalization Models:** Specify admissible matrix covariance functions using the Linear Model of Coregionalization (LMC), testing positive semidefiniteness via Sylvester's criterion and spectral constraints.
3. **Unify Cokriging with Generalized Linear Models:** Formulate Simple and Ordinary Cokriging systems as block-stacked extensions of Simple Kriging and Universal Kriging with block-indicator drift matrices.
4. **Solve and Diagnose Cokriging Systems:** Assemble augmented block systems, extract prediction weights and Lagrange multipliers, and compute exact prediction error variances with correct multiplier sign conventions.
5. **Evaluate Collocated Screening:** Analyze collocated cokriging and Markov Model 1 (MM1) screening approximations, verifying correlation consistency and understanding the mathematical basis of variance floors.
6. **Apply the Autokrigeability Theorem:** Diagnose whether isotopic sampling and intrinsic coregionalization render secondary data redundant, proving the vanishing of secondary weights via elementary matrix algebra and Kronecker product structures.

---

### Chapter at a Glance

| Section | Core Concept | Mathematical Tool | Pedagogical Takeaway |
| :--- | :--- | :--- | :--- |
| **§7.1 Multivariate Spatial Processes** | Vector random fields & cross-continuity | Matrix covariance $\mathbf{C}(\mathbf{h})$, transpose symmetry $\mathbf{C}(-\mathbf{h})'$ | Cross-covariances need not be symmetric ($C_{ij}(\mathbf{h}) 
e C_{ij}(-\mathbf{h})$), and joint positive definiteness restricts the entire matrix function. |
| **§7.2 Multivariate Sampling Geometries** | Observation arrangements | Isotopic ($S_i = S$), heterotopic ($S_i 
e S_j$), partially heterotopic | Cokriging delivers massive gains when secondary data is dense in primary voids, but negligible gains when data is isotopic. |
| **§7.3 Simple and Ordinary Cokriging** | BLUP on stacked data vectors | Block augmented system, $\sigma_{	ext{OCK}}^2 = C_{11}(\mathbf{0}) - oldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1$ | Ordinary Cokriging is Universal Kriging with block-indicator drift; secondary weights sum to zero to protect against unknown baseline offsets. |
| **§7.4 Collocated Cokriging & MM1** | Dimensionality reduction via screening | Markov Model 1: $
ho_{12}(\mathbf{h}) = 
ho_{12}(\mathbf{0})
ho_{22}(\mathbf{h})$ | Standard OCK ignores a lone collocated datum; standardized Simple Cokriging incorporates it, providing a legitimate variance floor $1 - 
ho_{12}(\mathbf{0})^2$. |
| **§7.5 The Autokrigeability Theorem** | Redundancy of secondary data | Kronecker product $\mathbf{B} \otimes \mathbf{R}$, null secondary weights $oldsymbol{\lambda}_2 = \mathbf{0}$ | When variables share the same spatial correlation scale under isotopic sampling, secondary measurements receive zero weight. |

---

### 7.1 Multivariate Spatial Processes and Cross-Continuity

#### 7.1.1 The Mathematical Anchor: Vector Random Fields and Cross-Covariance

A **multivariate spatial random field** is a vector-valued collection of finite-variance random variables indexed across a continuous geographic domain $D \subset \mathbb{R}^d$:
$$
\mathbf{Z}(\mathbf{s}) = egin{bmatrix} Z_1(\mathbf{s}) \ Z_2(\mathbf{s}) \ dots \ Z_p(\mathbf{s}) \end{bmatrix}, \quad \mathbf{s} \in D \subset \mathbb{R}^d
$$
where each component $Z_i(\mathbf{s})$ has finite variance $\operatorname{Var}(Z_i(\mathbf{s})) < \infty$.

Following the spatial linear model decomposition, each component process admits the structural representation:
$$
Z_i(\mathbf{s}) = \mu_i(\mathbf{s}) + \delta_i(\mathbf{s}) + \epsilon_i(\mathbf{s}) = \mathbf{x}_i(\mathbf{s})'oldsymbol{eta}_i + \delta_i(\mathbf{s}) + \epsilon_i(\mathbf{s}), \quad i=1, \dots, p
$$
where $\mu_i(\mathbf{s}) = \operatorname{E}[Z_i(\mathbf{s})]$ is the deterministic mean trend, $\delta_i(\mathbf{s})$ is a zero-mean continuous spatial sub-process, and $\epsilon_i(\mathbf{s})$ represents micro-scale variation or measurement error.

#### Classical-to-Spatial Bridge: From Classical Multivariate Regression to the Spatial Cross-Covariance Tensor

In classical multivariate regression, an investigator analyzes $p$ quantitative response variables observed across $n$ independent experimental units:
$$
\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}
$$
where $\mathbf{Y} \in \mathbb{R}^{n 	imes p}$, $\mathbf{X} \in \mathbb{R}^{n 	imes q}$, $\mathbf{B} \in \mathbb{R}^{q 	imes p}$, and $\mathbf{E} \in \mathbb{R}^{n 	imes p}$. Under classical i.i.d. assumptions:
1. **Cross-Unit Independence:** Distinct observational units are uncorrelated: $\operatorname{Cov}(\mathbf{e}_i, \mathbf{e}_j) = \mathbf{0}$ for $i 
e j$.
2. **Static Symmetric Covariance:** The error covariance matrix $oldsymbol{\Sigma} = \operatorname{Var}(\mathbf{e}_i) \in \mathbb{R}^{p 	imes p}$ is strictly symmetric ($\sigma_{jk} = \sigma_{kj}$), measuring co-variation strictly at the same observational unit.

When transitioning to spatial domains where $p$ continuous attributes are observed across $D \subset \mathbb{R}^d$, two profound mathematical departures occur:
1. **Spatial Lag Dependency & Directional Asymmetry:** Covariation depends continuously on the spatial separation lag vector $\mathbf{h} = \mathbf{s}_1 - \mathbf{s}_2 \in \mathbb{R}^d$. Furthermore, cross-covariances need not be symmetric functions of lag: measuring attribute $j$ at $\mathbf{s}$ and attribute $k$ at $\mathbf{s}+\mathbf{h}$ does *not* necessarily equal measuring attribute $j$ at $\mathbf{s}+\mathbf{h}$ and attribute $k$ at $\mathbf{s}$. Thus, in general:
   $$
   C_{jk}(\mathbf{h}) 
e C_{jk}(-\mathbf{h}) \quad 	ext{for } j 
e k
   $$
   Instead, cross-covariances satisfy **transpose symmetry**:
   $$
   C_{jk}(\mathbf{h}) = C_{kj}(-\mathbf{h}) \iff \mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})'
   $$
2. **Joint Positive Definiteness:** In classical regression, any positive semidefinite matrix $oldsymbol{\Sigma} \succeq 0$ is admissible. In spatial multivariate prediction, the matrix-valued function $\mathbf{C}(\mathbf{h})$ must be **jointly positive semidefinite across all configurations of spatial locations**:
   $$
   \sum_{lpha=1}^n \sum_{eta=1}^n \mathbf{a}_lpha' \mathbf{C}(\mathbf{s}_lpha - \mathbf{s}_eta) \mathbf{a}_eta \ge 0 \quad orall n \ge 1, \; \mathbf{s}_lpha \in D, \; \mathbf{a}_lpha \in \mathbb{R}^p
   $$

> [!NOTE]
> **Lag Convention for Cross-Covariance:**  
> Throughout this book, we maintain the standard convention:
> $$
> C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s})) = \operatorname{E}[(Z_i(\mathbf{s}+\mathbf{h}) - \mu_i)(Z_j(\mathbf{s}) - \mu_j)]
> $$
> In this convention, variable $i$ is displaced by lag $\mathbf{h}$ relative to variable $j$. Notice that:
> $$
> C_{ji}(-\mathbf{h}) = \operatorname{Cov}(Z_j(\mathbf{s}-\mathbf{h}), Z_i(\mathbf{s})) = \operatorname{Cov}(Z_j(\mathbf{s}'), Z_i(\mathbf{s}'+\mathbf{h})) = C_{ij}(\mathbf{h})
> $$
> This confirms transpose symmetry $\mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})'$ without ambiguity.

#### Definition 7.1.1 (Second-Order Multivariate Stationarity)
A multivariate random field $\{\mathbf{Z}(\mathbf{s}) : \mathbf{s} \in D\}$ is second-order stationary if:
1. The mean vector is spatially invariant:
   $$
   \operatorname{E}[\mathbf{Z}(\mathbf{s})] = oldsymbol{\mu} = (\mu_1, \dots, \mu_p)' \quad orall \mathbf{s} \in D
   $$
2. The cross-covariance between any two components $Z_i(\mathbf{s}+\mathbf{h})$ and $Z_j(\mathbf{s})$ depends solely on the separation lag vector $\mathbf{h} \in \mathbb{R}^d$:
   $$
   C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s}))
   $$

The collection of all auto- and cross-covariances forms the **matrix covariance function**:
$$
\mathbf{C}(\mathbf{h}) = egin{bmatrix}
C_{11}(\mathbf{h}) & C_{12}(\mathbf{h}) & \cdots & C_{1p}(\mathbf{h}) \
C_{21}(\mathbf{h}) & C_{22}(\mathbf{h}) & \cdots & C_{2p}(\mathbf{h}) \
dots & dots & \ddots & dots \
C_{p1}(\mathbf{h}) & C_{p2}(\mathbf{h}) & \cdots & C_{pp}(\mathbf{h})
\end{bmatrix} \in \mathbb{R}^{p 	imes p}
$$

#### Proposition 7.1.1 (Properties of Cross-Covariances)
1. **Transpose Symmetry:** For all $\mathbf{h} \in \mathbb{R}^d$, $C_{ij}(\mathbf{h}) = C_{ji}(-\mathbf{h})$, or equivalently $\mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})'$.
2. **Directional Asymmetry:** In general, $C_{ij}(\mathbf{h}) 
e C_{ij}(-\mathbf{h})$ for $i 
e j$.
3. **Cauchy–Schwarz Bound:** For all $\mathbf{h} \in \mathbb{R}^d$, $|C_{ij}(\mathbf{h})| \le \sqrt{C_{ii}(\mathbf{0}) C_{jj}(\mathbf{0})}$.
4. **Lag-Zero Covariance:** The matrix at lag zero, $\mathbf{C}(\mathbf{0}) = \operatorname{Var}(\mathbf{Z}(\mathbf{s})) \in \mathbb{R}^{p 	imes p}$, is symmetric and positive semidefinite: $\mathbf{C}(\mathbf{0}) = \mathbf{C}(\mathbf{0})' \succeq 0$.

*Proof of Cauchy–Schwarz Bound.*
By the classical Cauchy–Schwarz inequality for any two random variables $X = Z_i(\mathbf{s}+\mathbf{h})$ and $Y = Z_j(\mathbf{s})$:
$$
|\operatorname{Cov}(X, Y)| \le \sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}
$$
Under second-order stationarity, $\operatorname{Var}(Z_i(\mathbf{s}+\mathbf{h})) = C_{ii}(\mathbf{0})$ and $\operatorname{Var}(Z_j(\mathbf{s})) = C_{jj}(\mathbf{0})$. Substituting these directly gives $|C_{ij}(\mathbf{h})| \le \sqrt{C_{ii}(\mathbf{0})C_{jj}(\mathbf{0})}$. $lacksquare$

#### 7.1.2 The Linear Model of Coregionalization (LMC)

Because arbitrary combinations of valid univariate covariance functions almost never produce a jointly positive semidefinite matrix function $\mathbf{C}(\mathbf{h})$, spatial statisticians construct admissible multivariate models via the **Linear Model of Coregionalization (LMC)** (Journel & Huijbregts, 1978).

In the LMC, each observed variable $Z_i(\mathbf{s})$ is modeled as a linear combination of $K$ mutually orthogonal, zero-mean spatial factors $\{Y_k(\mathbf{s})\}_{k=1}^K$, where each factor exhibits a characteristic spatial correlation structure $
ho_k(\mathbf{h})$:
$$
\mathbf{C}(\mathbf{h}) = \sum_{k=1}^K \mathbf{B}_k 
ho_k(\mathbf{h})
$$
where:
- $\mathbf{B}_k \in \mathbb{R}^{p 	imes p}$ is the symmetric $p 	imes p$ **coregionalization matrix** governing the cross-covariances at spatial scale $k$.
- $
ho_k(\mathbf{h})$ is an authorized univariate spatial correlation function on $\mathbb{R}^d$ ($
ho_k(\mathbf{0}) = 1$).

#### Theorem 7.1.1 (Admissibility of the Linear Model of Coregionalization)
The matrix covariance function $\mathbf{C}(\mathbf{h}) = \sum_{k=1}^K \mathbf{B}_k 
ho_k(\mathbf{h})$ is valid and positive semidefinite on $\mathbb{R}^d$ if:
1. Every spatial correlation function $
ho_k(\mathbf{h})$ is valid and positive definite on $\mathbb{R}^d$.
2. Every coregionalization matrix $\mathbf{B}_k$ is symmetric and positive semidefinite: $\mathbf{B}_k = \mathbf{B}_k' \succeq 0$ for all $k=1,\dots, K$.

*Proof.*
For any number of sample points $n$, locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$, and arbitrary vectors $\mathbf{a}_1, \dots, \mathbf{a}_n \in \mathbb{R}^p$, evaluate the global quadratic form:
$$
Q = \sum_{lpha=1}^n \sum_{eta=1}^n \mathbf{a}_lpha' \mathbf{C}(\mathbf{s}_lpha - \mathbf{s}_eta) \mathbf{a}_eta = \sum_{k=1}^K \sum_{lpha=1}^n \sum_{eta=1}^n \mathbf{a}_lpha' \mathbf{B}_k \mathbf{a}_eta 
ho_k(\mathbf{s}_lpha - \mathbf{s}_eta)
$$
Because each $\mathbf{B}_k \succeq 0$, it admits a spectral factorization $\mathbf{B}_k = \sum_{m=1}^p \lambda_{k,m} \mathbf{v}_{k,m} \mathbf{v}_{k,m}'$ with eigenvalues $\lambda_{k,m} \ge 0$. Substituting this yields:
$$
Q = \sum_{k=1}^K \sum_{m=1}^p \lambda_{k,m} \left( \sum_{lpha=1}^n \sum_{eta=1}^n (\mathbf{v}_{k,m}' \mathbf{a}_lpha) (\mathbf{v}_{k,m}' \mathbf{a}_eta) 
ho_k(\mathbf{s}_lpha - \mathbf{s}_eta) 
ight)
$$
For each $(k, m)$, let $w_lpha = \mathbf{v}_{k,m}' \mathbf{a}_lpha \in \mathbb{R}$. The inner double sum is $\mathbf{w}' \mathbf{R}_k \mathbf{w}$, where $\mathbf{R}_k = [
ho_k(\mathbf{s}_lpha - \mathbf{s}_eta)] \in \mathbb{R}^{n 	imes n}$. Because $
ho_k$ is a valid univariate correlation function, $\mathbf{R}_k$ is positive semidefinite, so $\mathbf{w}' \mathbf{R}_k \mathbf{w} \ge 0$. Since $\lambda_{k,m} \ge 0$, $Q$ is a sum of non-negative terms, hence $Q \ge 0$. Thus $\mathbf{C}(\mathbf{h})$ is positive semidefinite. $lacksquare$

> [!WARNING]
> **Sufficiency vs. Necessity:**  
> The condition $\mathbf{B}_k \succeq 0$ for all $k$ is **sufficient** for the validity of $\mathbf{C}(\mathbf{h})$, but it is not strictly necessary. As shown in Chapter 4, it is possible for individual $\mathbf{B}_k$ to have slight negative eigenvalues while the sum $\sum \mathbf{B}_k 
ho_k(\mathbf{h})$ remains globally positive semidefinite across all lags. However, in practice, enforcing $\mathbf{B}_k \succeq 0$ for each structural component $k$ guarantees validity by construction regardless of spatial sampling design, making it the universal standard in applied geostatistics.

#### 7.1.3 Algorithmic Translation: Assembling Block Covariance Matrices

The following function evaluates the global $N 	imes N$ block covariance matrix for an arbitrary multivariate sampling dataset under an LMC:

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
    
    block_indices = []
    current_idx = 0
    for count in n_counts:
        block_indices.append((current_idx, current_idx + count))
        current_idx += count
        
    Sigma = [[0.0 for _ in range(N)] for _ in range(N)]
    
    for i in range(p):
        start_i, end_i = block_indices[i]
        coords_i = sample_coords_by_var[i]
        n_i = n_counts[i]
        
        for j in range(i, p):
            start_j, end_j = block_indices[j]
            coords_j = sample_coords_by_var[j]
            n_j = n_counts[j]
            
            for alpha in range(n_i):
                s_alpha = coords_i[alpha]
                row = start_i + alpha
                
                for beta in range(n_j):
                    s_beta = coords_j[beta]
                    col = start_j + beta
                    
                    if i == j and beta < alpha:
                        Sigma[row][col] = Sigma[col][row]
                        continue
                        
                    diff_sq = sum((s_alpha[d] - s_beta[d]) ** 2 for d in range(len(s_alpha)))
                    h = diff_sq ** 0.5
                    
                    val = 0.0
                    for k in range(K):
                        val += B_matrices[k][i][j] * rho_models[k](h)
                        
                    Sigma[row][col] = val
                    if i != j:
                        Sigma[col][row] = val
                        
    return Sigma, block_indices
```

#### 7.1.4 Intuition, Edge Cases & Diagnostic Verification

- **Physical Meaning of Asymmetry (The Shifted Process):** Natural processes with advective transport (wind, groundwater flow) exhibit directional cross-asymmetry. For example, if process $Z_2(\mathbf{s})$ is an advected copy of process $Z_1(\mathbf{s})$ with spatial displacement $\mathbf{v} \in \mathbb{R}^d$ and independent noise $\eta(\mathbf{s})$:
  $$
  Z_2(\mathbf{s}) = lpha Z_1(\mathbf{s} - \mathbf{v}) + \eta(\mathbf{s})
  $$
  then the cross-covariance is:
  $$
  C_{12}(\mathbf{h}) = \operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s})) = lpha C_{11}(\mathbf{h} + \mathbf{v})
  $$
  Notice that $C_{12}(\mathbf{h})$ reaches its maximum at lag $\mathbf{h} = -\mathbf{v}$, while $C_{12}(-\mathbf{h}) = lpha C_{11}(-\mathbf{h} + \mathbf{v})$ peaks at $\mathbf{h} = +\mathbf{v}$. Thus $C_{12}(\mathbf{h}) 
e C_{12}(-\mathbf{h})$, exhibiting pronounced asymmetry reflecting physical downstream transport.
- **Checking Sylvester's Criterion at Lag Zero:** Before fitting complex spatial structures, an analyst must always check the empirical cross-covariance matrix at lag zero: $\mathbf{C}(\mathbf{0}) \in \mathbb{R}^{p 	imes p}$. For a bivariate system ($p=2$), $\mathbf{C}(\mathbf{0}) = egin{bmatrix} C_{11}(\mathbf{0}) & C_{12}(\mathbf{0}) \ C_{12}(\mathbf{0}) & C_{22}(\mathbf{0}) \end{bmatrix}$. If $\det(\mathbf{C}(\mathbf{0})) = C_{11}(\mathbf{0})C_{22}(\mathbf{0}) - C_{12}(\mathbf{0})^2 < 0$, the correlation coefficient exceeds 1 in magnitude, which is physically impossible and will cause negative prediction error variances.

---


### 7.2 Multivariate Sampling Geometries

#### 7.2.1 The Mathematical Anchor: Geometrical Configurations

In multivariate spatial prediction, the relative spatial arrangement of measurements across the $p$ variables profoundly dictates both computational feasibility and prediction gain.

#### Definition 7.2.1 (Sampling Geometries)
Let $S_i = \{\mathbf{s}_{i,1}, \dots, \mathbf{s}_{i,n_i}\} \subset D$ denote the set of sample locations where attribute $Z_i$ is observed ($i = 1, \dots, p$).
1. **Isotopic Configuration:** All $p$ attributes are measured at the exact same spatial locations:
   $$
   S_1 = S_2 = \cdots = S_p = S, \quad 	ext{with } n_1 = n_2 = \cdots = n_p = n
   $$
   Total observations $N = p \cdot n$.
2. **Heterotopic Configuration:** The sampling locations differ across variables ($S_i 
e S_j$). Sampling may be completely disjoint ($S_i \cap S_j = \emptyset$) or partially overlapping.
3. **Partially Heterotopic (Collocated) Configuration:** The primary variable $Z_1$ is sampled sparsely ($n_1$ is small, expensive lab assays), while secondary variables $\{Z_2, \dots, Z_p\}$ are sampled densely or exhaustively across the continuous domain ($n_2 \gg n_1$, cheap satellite imagery or seismic surveys).

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

#### 7.2.2 The Stacked Data Vector and Augmented Block Structure

Let $N = \sum_{i=1}^p n_i$ denote the total number of scalar observations. In the linear model formulation of cokriging, all measurements are arranged into a single stacked vector:
$$
\mathbf{Z} = egin{bmatrix} \mathbf{Z}_1 \ \mathbf{Z}_2 \ dots \ \mathbf{Z}_p \end{bmatrix} \in \mathbb{R}^N, \quad 	ext{where } \mathbf{Z}_i = egin{bmatrix} Z_i(\mathbf{s}_{i,1}) \ dots \ Z_i(\mathbf{s}_{i,n_i}) \end{bmatrix} \in \mathbb{R}^{n_i}
$$
The global $N 	imes N$ spatial covariance matrix $oldsymbol{\Sigma} = \operatorname{Var}(\mathbf{Z})$ is partitioned into $p 	imes p$ blocks:
$$
oldsymbol{\Sigma} = egin{bmatrix}
oldsymbol{\Sigma}_{11} & oldsymbol{\Sigma}_{12} & \cdots & oldsymbol{\Sigma}_{1p} \
oldsymbol{\Sigma}_{21} & oldsymbol{\Sigma}_{22} & \cdots & oldsymbol{\Sigma}_{2p} \
dots & dots & \ddots & dots \
oldsymbol{\Sigma}_{p1} & oldsymbol{\Sigma}_{p2} & \cdots & oldsymbol{\Sigma}_{pp}
\end{bmatrix} \in \mathbb{R}^{N 	imes N}
$$
where each sub-matrix $[oldsymbol{\Sigma}_{ij}]_{lpha, eta} = C_{ij}(\mathbf{s}_{i,lpha} - \mathbf{s}_{j,eta}) \in \mathbb{R}^{n_i 	imes n_j}$.

Notice the dimensional implications:
- Under **isotopic sampling** ($n_i = n$), $oldsymbol{\Sigma}$ has size $pn 	imes pn$. If the coregionalization model is intrinsic ($\mathbf{C}(\mathbf{h}) = \mathbf{B} 
ho(\mathbf{h})$), $oldsymbol{\Sigma}$ factors cleanly into a Kronecker product $\mathbf{B} \otimes \mathbf{R}$, enabling fast algebra.
- Under **heterotopic sampling** with large secondary surveys ($n_1 = 50, n_2 = 100{,}000$), the full matrix size is $(100{,}050) 	imes (100{,}050)$. Inverting this dense matrix directly requires $pprox 10^{15}$ operations—an impossible computation without moving neighborhoods or collocated approximations (§7.4).

#### 7.2.3 Intuition, Edge Cases & Diagnostic Verification: When Does Geometry Matter?

> [!NOTE]
> **From the Lecture Hall: Why Cokriging Needs Heterotopic Data**  
> Students often ask: *If variables are correlated with $
ho_{12} = 0.8$, won't cokriging always give much better predictions than univariate kriging?*  
> The surprising answer is **no**. If the data is *isotopic* (measured at the same boreholes) and governed by a single spatial scale, secondary data gives **zero** improvement over univariate kriging (Theorem 7.5).  
> Cokriging delivers its dramatic value precisely in **heterotopic** settings, where the secondary variable acts as a high-resolution spatial scaffold across spatial information voids.

---


### 7.3 Simple and Ordinary Cokriging Systems

#### 7.3.1 The Mathematical Anchor: The Linear Model Unification

Before diving into block derivations, we establish the central conceptual bridge that unifies cokriging with Chapters 5 and 6:
> [!NOTE]
> **The Unifying Linear Model Bridge:**  
> **Cokriging is simply kriging on a stacked data vector.**  
> - **Simple Cokriging (SCK)** is identically Simple Kriging (Theorem 5.2.1) applied to the $N$-dimensional stacked observation vector $\mathbf{Z} \in \mathbb{R}^N$.
> - **Ordinary Cokriging (OCK)** is identically Universal Kriging (Theorem 6.3.1) applied to $\mathbf{Z}$ where the design matrix $\mathbf{X} \in \mathbb{R}^{N 	imes p}$ contains block indicators for the $p$ unknown means, and the target drift vector is $\mathbf{x}_0 = \mathbf{e}_1 = (1, 0, \dots, 0)' \in \mathbb{R}^p$.

Let $\mathbf{s}_0 \in D$ be an unsampled location where we seek to predict the primary target variable $Z_1(\mathbf{s}_0)$. The general linear multivariate predictor is:
$$
\widehat{Z}_1(\mathbf{s}_0) = \sum_{i=1}^p \sum_{lpha=1}^{n_i} \lambda_{ilpha} Z_i(\mathbf{s}_{ilpha}) = oldsymbol{\lambda}' \mathbf{Z}
$$
where $oldsymbol{\lambda} = (oldsymbol{\lambda}_1', \dots, oldsymbol{\lambda}_p')' \in \mathbb{R}^N$ is the vector of cokriging weights, with $oldsymbol{\lambda}_i = (\lambda_{i1}, \dots, \lambda_{in_i})' \in \mathbb{R}^{n_i}$.

#### 1. Simple Cokriging (SCK)
When the constant mean vector $oldsymbol{\mu} = (\mu_1, \dots, \mu_p)'$ is completely known, working with mean-centered variables $Y_i(\mathbf{s}) = Z_i(\mathbf{s}) - \mu_i$ yields the predictor:
$$
\widehat{Z}_{1,	ext{SCK}}(\mathbf{s}_0) = \mu_1 + \sum_{i=1}^p \sum_{lpha=1}^{n_i} \lambda_{ilpha}^{	ext{SCK}} (Z_i(\mathbf{s}_{ilpha}) - \mu_i) = \mu_1 + oldsymbol{\lambda}_{	ext{SCK}}' (\mathbf{Z} - \mathbf{X}oldsymbol{\mu})
$$
Minimizing the mean-squared prediction error $\sigma_{	ext{SCK}}^2(\mathbf{s}_0) = \operatorname{E}[(\widehat{Z}_1(\mathbf{s}_0) - Z_1(\mathbf{s}_0))^2]$ yields the **Simple Cokriging normal equations**:
$$
oldsymbol{\Sigma} oldsymbol{\lambda}_{	ext{SCK}} = \mathbf{c}_{0,1}
$$
where $\mathbf{c}_{0,1} = (\mathbf{c}_{0,11}', \mathbf{c}_{0,12}', \dots, \mathbf{c}_{0,1p}')' \in \mathbb{R}^N$ is the cross-covariance vector between all observed data and target $Z_1(\mathbf{s}_0)$, with $[\mathbf{c}_{0,1i}]_lpha = C_{1i}(\mathbf{s}_0 - \mathbf{s}_{ilpha}) = C_{i1}(\mathbf{s}_{ilpha} - \mathbf{s}_0)$.

The minimized Simple Cokriging variance is:
$$
\sigma_{	ext{SCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - oldsymbol{\lambda}_{	ext{SCK}}' \mathbf{c}_{0,1} = C_{11}(\mathbf{0}) - \mathbf{c}_{0,1}' oldsymbol{\Sigma}^{-1} \mathbf{c}_{0,1}
$$

#### 2. Ordinary Cokriging (OCK)
When the constant means $\mu_1, \dots, \mu_p$ are unknown, the predictor must satisfy the unbiasedness condition:
$$
\operatorname{E}[\widehat{Z}_1(\mathbf{s}_0) - Z_1(\mathbf{s}_0)] = \operatorname{E}\left[ \sum_{i=1}^p \sum_{lpha=1}^{n_i} \lambda_{ilpha} Z_i(\mathbf{s}_{ilpha}) - Z_1(\mathbf{s}_0) 
ight] = \sum_{i=1}^p \mu_i \left( \sum_{lpha=1}^{n_i} \lambda_{ilpha} 
ight) - \mu_1 = 0
$$
Because this identity must hold for **all arbitrary, unknown values of the means** $(\mu_1, \dots, \mu_p) \in \mathbb{R}^p$, the coefficient multiplying each $\mu_i$ must vanish independently. This establishes the **Ordinary Cokriging unbiasedness constraints**:
$$
\sum_{lpha=1}^{n_1} \lambda_{1lpha} = 1, \quad 	ext{and} \quad \sum_{lpha=1}^{n_j} \lambda_{jlpha} = 0 \quad orall j = 2, \dots, p
$$

#### Theorem 7.3.1 (Ordinary Cokriging BLUP System)
The Best Linear Unbiased Predictor (BLUP) weights $oldsymbol{\lambda}_{	ext{OCK}} \in \mathbb{R}^N$ and Lagrange multipliers $oldsymbol{\mu} = (\mu_1, \dots, \mu_p)' \in \mathbb{R}^p$ satisfy the $(N+p) 	imes (N+p)$ linear system:
$$
egin{bmatrix} oldsymbol{\Sigma} & \mathbf{X} \ \mathbf{X}' & \mathbf{0} \end{bmatrix}
egin{bmatrix} oldsymbol{\lambda}_{	ext{OCK}} \ -oldsymbol{\mu} \end{bmatrix}
=
egin{bmatrix} \mathbf{c}_{0,1} \ \mathbf{e}_1 \end{bmatrix}
$$
where $\mathbf{e}_1 = (1, 0, \dots, 0)' \in \mathbb{R}^p$, and $\mathbf{X} \in \mathbb{R}^{N 	imes p}$ is the block-diagonal indicator matrix:
$$
\mathbf{X} = egin{bmatrix} \mathbf{1}_{n_1} & \mathbf{0} & \cdots & \mathbf{0} \ \mathbf{0} & \mathbf{1}_{n_2} & \cdots & \mathbf{0} \ dots & dots & \ddots & dots \ \mathbf{0} & \mathbf{0} & \cdots & \mathbf{1}_{n_p} \end{bmatrix} \in \mathbb{R}^{N 	imes p}
$$
The minimized Ordinary Cokriging prediction variance is:
$$
\sigma_{	ext{OCK}}^2(\mathbf{s}_0) = C_{11}(\mathbf{0}) - oldsymbol{\lambda}_{	ext{OCK}}' \mathbf{c}_{0,1} + \mu_1
$$

*Proof.*
The prediction error variance is:
$$
\sigma^2 = \operatorname{Var}(oldsymbol{\lambda}'\mathbf{Z} - Z_1(\mathbf{s}_0)) = oldsymbol{\lambda}' oldsymbol{\Sigma} oldsymbol{\lambda} - 2oldsymbol{\lambda}' \mathbf{c}_{0,1} + C_{11}(\mathbf{0})
$$
We minimize $\sigma^2$ subject to the $p$ linear constraints $\mathbf{X}'oldsymbol{\lambda} = \mathbf{e}_1$. We formulate the Lagrangian:
$$
\mathcal{L}(oldsymbol{\lambda}, oldsymbol{\mu}) = oldsymbol{\lambda}' oldsymbol{\Sigma} oldsymbol{\lambda} - 2oldsymbol{\lambda}' \mathbf{c}_{0,1} + C_{11}(\mathbf{0}) - 2 oldsymbol{\mu}' (\mathbf{X}'oldsymbol{\lambda} - \mathbf{e}_1)
$$
Setting partial derivatives to zero:
$$
rac{\partial \mathcal{L}}{\partial oldsymbol{\lambda}} = 2 oldsymbol{\Sigma} oldsymbol{\lambda} - 2 \mathbf{c}_{0,1} - 2 \mathbf{X} oldsymbol{\mu} = \mathbf{0} \implies oldsymbol{\Sigma} oldsymbol{\lambda} - \mathbf{X} oldsymbol{\mu} = \mathbf{c}_{0,1}
$$
$$
rac{\partial \mathcal{L}}{\partial oldsymbol{\mu}} = -2 (\mathbf{X}'oldsymbol{\lambda} - \mathbf{e}_1) = \mathbf{0} \implies \mathbf{X}'oldsymbol{\lambda} = \mathbf{e}_1
$$
Writing this in bordered block matrix form yields:
$$
egin{bmatrix} oldsymbol{\Sigma} & \mathbf{X} \ \mathbf{X}' & \mathbf{0} \end{bmatrix}
egin{bmatrix} oldsymbol{\lambda} \ -oldsymbol{\mu} \end{bmatrix}
=
egin{bmatrix} \mathbf{c}_{0,1} \ \mathbf{e}_1 \end{bmatrix}
$$
To obtain the minimized prediction variance, multiply the top equation $oldsymbol{\Sigma}oldsymbol{\lambda} = \mathbf{c}_{0,1} + \mathbf{X}oldsymbol{\mu}$ by $oldsymbol{\lambda}'$:
$$
oldsymbol{\lambda}' oldsymbol{\Sigma} oldsymbol{\lambda} = oldsymbol{\lambda}' \mathbf{c}_{0,1} + (\mathbf{X}'oldsymbol{\lambda})' oldsymbol{\mu} = oldsymbol{\lambda}' \mathbf{c}_{0,1} + \mathbf{e}_1' oldsymbol{\mu} = oldsymbol{\lambda}' \mathbf{c}_{0,1} + \mu_1
$$
Substitute this back into the error variance definition:
$$
\sigma_{	ext{OCK}}^2(\mathbf{s}_0) = (oldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1) - 2oldsymbol{\lambda}'\mathbf{c}_{0,1} + C_{11}(\mathbf{0}) = C_{11}(\mathbf{0}) - oldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1
$$
This completes the proof. $lacksquare$

#### 7.3.2 Algorithmic Translation: Bivariate Ordinary Cokriging Solver

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
    aug = [row[:] + [rhs[i]] for i, row in enumerate(A)]
    for col in range(M):
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
    # In augmented matrix A, columns N and N+1 correspond to -mu_1 and -mu_2.
    # Therefore, sol[N] = -mu_1, which means mu_1 = -sol[N].
    weights_1 = sol[:n1]
    weights_2 = sol[n1:N]
    mu_1 = -sol[N]
    mu_2 = -sol[N + 1]
    
    # Step 5: Compute prediction and exact prediction variance:
    # sigma^2 = C_11(0) - lambda' * c_0 + mu_1
    z1_hat = sum(weights_1[a] * vals_1[a] for a in range(n1)) +              sum(weights_2[b] * vals_2[b] for b in range(n2))
             
    c00 = sum(B_matrices[k][0][0] * rho_models[k](0.0) for k in range(K))
    dot_lam_c0 = sum(sol[i] * rhs[i] for i in range(N))
    sigma2_ock = c00 - dot_lam_c0 + mu_1
    
    return z1_hat, sigma2_ock, weights_1, weights_2
```

#### 7.3.3 Intuition, Edge Cases & Diagnostic Verification

- **Why Secondary Weights Must Sum to Zero:**
  In Ordinary Cokriging, the constraint $\sum_{eta=1}^{n_2} \lambda_{2eta} = 0$ protects against an **unknown secondary baseline offset**. If an analyst measures water table elevation relative to an unknown local sea-level datum, adding an arbitrary constant offset $C$ to all secondary measurements changes $Z_2(\mathbf{s})$ to $Z_2(\mathbf{s}) + C$. Because the secondary weights satisfy $\sum \lambda_{2eta} = 0$, the offset contribution is $\sum \lambda_{2eta} C = C \sum \lambda_{2eta} = 0$, leaving the primary prediction $\widehat{Z}_1(\mathbf{s}_0)$ invariant!
- **The Dipole Mechanism: Secondary Data as a Gradient Corrector:**
  Because secondary weights must sum to zero, they cannot all be positive. Instead, they form **spatial dipoles**: a positive weight on a secondary datum near the target $\mathbf{s}_0$, counterbalanced by negative weights on secondary data located near primary sample sites.
  
  Consider a concrete 1D transect with primary data at $x = 0$ and $x = 6$, and secondary data available at all integers $x \in \{0, 1, 2, 3, 4, 5, 6\}$. We predict at midpoint $x_0 = 3$ with cross-correlation $
ho_{12} = 0.9$ under an exponential covariance kernel:

  | Location $x$ | Primary Weight $\lambda_{1}(x)$ | Secondary Weight $\lambda_{2}(x)$ | Role in Prediction |
  | :--- | :--- | :--- | :--- |
  | $x = 0$ | $+0.50$ | $-0.45$ | Negative counterbalance anchor |
  | $x = 1, 2$ | — | $0.00$ | Screened out by neighbors |
  | $x = 3$ | — | $+0.90$ | Positive local proxy at target |
  | $x = 4, 5$ | — | $0.00$ | Screened out by neighbors |
  | $x = 6$ | $+0.50$ | $-0.45$ | Negative counterbalance anchor |
  | **Sum** | **$+1.00$** | **$0.00$** | Unbiasedness satisfied |

  The secondary term acts as an empirical spatial slope:
  $$
  \sum \lambda_{2eta} Z_2(x_eta) = 0.90 \left[ Z_2(3) - rac{Z_2(0) + Z_2(6)}{2} 
ight]
  $$
  It measures the local anomaly of the secondary field at the target relative to the secondary values at the primary sites, adding that calibrated anomaly directly into the primary prediction!

- **Concrete Numerical Benchmark: When Does Cokriging Pay?**
  Consider two primary observations at $x = -1$ and $x = +1$ with variance $C_{11}(0) = 1.0$, predicting at $x_0 = 0$. A secondary variable is observed along a dense grid of 21 points in $[-2, 2]$.
  - **Primary Ordinary Kriging alone:** $\sigma^2 = 0.8319$ ($\mu_1 = 0.1998$).
  - **Cokriging with $
ho_{12} = 0.5$:** $\sigma^2$ drops to **$0.6239$** (a $25\%$ variance reduction).
  - **Cokriging with $
ho_{12} = 0.9$:** $\sigma^2$ plunges to **$0.1581$** (an **$81\%$ variance reduction**!).
  
  When high-correlation secondary data is available in spatial voids where primary data is absent, cokriging provides dramatic accuracy gains.

---


### 7.4 Collocated Cokriging and Markov Screening Approximations

#### 7.4.1 The Mathematical Anchor: The Collocated Paradox and Markov Model 1

When auxiliary secondary information is derived from satellite imaging, digital elevation models (DEMs), or 3D seismic reflection cubes, the secondary sample size $n_2$ frequently reaches $10^5$ to $10^7$ observations. Building and inverting the full $(n_1 + n_2 + 2) 	imes (n_1 + n_2 + 2)$ Ordinary Cokriging system across millions of grid cells is computationally impossible.

To circumvent this curse of dimensionality, geostatisticians employ **Collocated Cokriging**, which retains a local neighborhood of primary observations along with a single auxiliary secondary datum $Z_2(\mathbf{s}_0)$ evaluated directly at the target location $\mathbf{s}_0$.

However, retaining only a single secondary datum creates an immediate mathematical dilemma under standard Ordinary Cokriging.

> [!IMPORTANT]
> **Remark 7.4.1 (The Collocated Ordinary Cokriging Dilemma):**  
> If standard Ordinary Cokriging is applied using $n_1$ primary data and a single collocated secondary datum $Z_2(\mathbf{s}_0)$, the secondary unbiasedness constraint requires:
> $$
> \sum_{eta=1}^{n_2} \lambda_{2eta} = 0 \implies \lambda_2(\mathbf{s}_0) = 0
> $$
> Because the sum contains only a single term, the collocated secondary weight is forced to be **identically zero**. Standard Ordinary Cokriging completely ignores a lone collocated secondary datum!

In applied practice, geostatisticians resolve this dilemma through two alternative formulations:
1. **Standardized Ordinary Cokriging:** The secondary variable is standardized to have the same mean as the primary variable, and a single overall constraint $\sum \lambda_{1lpha} + \sum \lambda_{2eta} = 1$ is enforced (Goovaerts, 1998).
2. **Standardized Collocated Simple Cokriging:** Both variables are standardized to zero mean and unit variance:
   $$
   Y_1(\mathbf{s}) = rac{Z_1(\mathbf{s}) - \mu_1}{\sigma_1}, \quad Y_2(\mathbf{s}) = rac{Z_2(\mathbf{s}) - \mu_2}{\sigma_2}
   $$
   and Simple Cokriging is performed without zero-sum constraints.

#### Markov Model 1 (MM1) Screening Hypothesis
Under **Markov Model 1 (MM1)** (Journel, 1999), we hypothesize that the collocated secondary datum $Y_2(\mathbf{s}_0)$ conditionally screens the target variable $Y_1(\mathbf{s}_0)$ from all other secondary measurements $Y_2(\mathbf{s})$ ($\mathbf{s} 
e \mathbf{s}_0$):
$$
\operatorname{Cov}(Y_1(\mathbf{s}_0), Y_2(\mathbf{s}) \mid Y_2(\mathbf{s}_0)) = 0 \quad orall \mathbf{s} 
e \mathbf{s}_0
$$

#### Proposition 7.4.1 (Cross-Correlation Under MM1)
Under the MM1 conditional screening hypothesis, the cross-correlation function is strictly proportional to the secondary auto-correlation function:
$$

ho_{12}(\mathbf{h}) = 
ho_{12}(\mathbf{0}) 
ho_{22}(\mathbf{h})
$$

*Proof.*
Decompose $Y_1(\mathbf{s}_0)$ via linear regression onto $Y_2(\mathbf{s}_0)$:
$$
Y_1(\mathbf{s}_0) = 
ho_{12}(\mathbf{0}) Y_2(\mathbf{s}_0) + R(\mathbf{s}_0)
$$
where $R(\mathbf{s}_0)$ is the regression residual, satisfying $\operatorname{E}[R(\mathbf{s}_0)] = 0$ and $\operatorname{Cov}(R(\mathbf{s}_0), Y_2(\mathbf{s}_0)) = 0$.
The MM1 hypothesis asserts that $R(\mathbf{s}_0)$ is conditionally uncorrelated with every secondary measurement across the domain: $\operatorname{Cov}(R(\mathbf{s}_0), Y_2(\mathbf{s})) = 0$ for all $\mathbf{s} \in D$.
Therefore, evaluating the spatial cross-covariance at lag $\mathbf{h} = \mathbf{s}_0 - \mathbf{s}$:
$$
egin{aligned}
\operatorname{Cov}(Y_1(\mathbf{s}_0), Y_2(\mathbf{s})) &= \operatorname{Cov}\Big(
ho_{12}(\mathbf{0}) Y_2(\mathbf{s}_0) + R(\mathbf{s}_0), \, Y_2(\mathbf{s})\Big) \
&= 
ho_{12}(\mathbf{0}) \operatorname{Cov}(Y_2(\mathbf{s}_0), Y_2(\mathbf{s})) + \operatorname{Cov}(R(\mathbf{s}_0), Y_2(\mathbf{s})) \
&= 
ho_{12}(\mathbf{0}) 
ho_{22}(\mathbf{h}) + 0 = 
ho_{12}(\mathbf{0}) 
ho_{22}(\mathbf{h})
\end{aligned}
$$
This establishes the MM1 cross-correlation relation. $lacksquare$

> [!WARNING]
> **Admissibility Condition for MM1:**  
> MM1 does not hold for arbitrary combinations of $
ho_{11}$ and $
ho_{22}$. Because the residual process $R(\mathbf{s})$ has correlation:
> $$
> 
ho_R(\mathbf{h}) = rac{
ho_{11}(\mathbf{h}) - 
ho_{12}(\mathbf{0})^2 
ho_{22}(\mathbf{h})}{1 - 
ho_{12}(\mathbf{0})^2}
> $$
> $
ho_R(\mathbf{h})$ must itself be a valid, positive semidefinite correlation function on $\mathbb{R}^d$. When $|
ho_{12}(\mathbf{0})|$ is large, this requires $
ho_{11}$ and $
ho_{22}$ to have very similar spatial ranges. If $
ho_{22}$ has a much longer range than $
ho_{11}$ under high correlation, the joint system becomes non-positive definite.

The standardized collocated Simple Cokriging system size is $(n_1 + 1) 	imes (n_1 + 1)$:
$$
egin{bmatrix}
\mathbf{R}_{11} & \mathbf{r}_{12}(\mathbf{s}_0) \
\mathbf{r}_{12}(\mathbf{s}_0)' & 1
\end{bmatrix}
egin{bmatrix}
oldsymbol{\lambda}_1 \
\lambda_2(\mathbf{s}_0)
\end{bmatrix}
=
egin{bmatrix}
\mathbf{r}_{0,11} \

ho_{12}(\mathbf{0})
\end{bmatrix}
$$
where $[\mathbf{r}_{12}(\mathbf{s}_0)]_lpha = 
ho_{12}(\mathbf{0}) 
ho_{22}(\mathbf{s}_{1lpha} - \mathbf{s}_0)$.

#### 7.4.2 Algorithmic Translation: Standardized Collocated Cokriging

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
            
    # Vector r_12(s0): under MM1, rho_12(h) = rho_12(0) * rho_22(h)
    for i in range(n1):
        h = sum((coords_1[i][d] - s0[d])**2 for d in range(len(s0))) ** 0.5
        cross_corr = rho12_0 * rho22_func(h)
        R[i][n1] = cross_corr
        R[n1][i] = cross_corr
        
    R[n1][n1] = 1.0
    
    # Step 3: Build the right-hand side vector rhs
    rhs = [0.0 for _ in range(n1 + 1)]
    for i in range(n1):
        h = sum((coords_1[i][d] - s0[d])**2 for d in range(len(s0))) ** 0.5
        rhs[i] = rho11_func(h)
    rhs[n1] = rho12_0
    
    # Step 4: Solve R * lam = rhs via Gaussian elimination
    M = n1 + 1
    aug = [row[:] + [rhs[i]] for i, row in enumerate(R)]
    for col in range(M):
        max_r = col
        for r in range(col + 1, M):
            if abs(aug[r][col]) > abs(aug[max_r][col]):
                max_r = r
        aug[col], aug[max_r] = aug[max_r], aug[col]
        
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError(f"Singular collocated matrix at pivot {col}. Verify MM1 model validity.")
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

#### 7.4.3 Intuition, Edge Cases & Diagnostic Verification

- **Understanding the Prediction Variance Floor:**
  Analysts sometimes refer to the "variance floor" of collocated cokriging as a defect. In reality, **it is a mathematically legitimate consequence of the model**:
  If the secondary variable is observed directly at $\mathbf{s}_0$ and shares correlation $
ho_{12}(\mathbf{0})$ with the target, the simple linear regression of $Y_1(\mathbf{s}_0)$ onto $Y_2(\mathbf{s}_0)$ alone yields residual variance $1 - 
ho_{12}(\mathbf{0})^2$. Even if there are zero primary observations nearby, knowing $Y_2(\mathbf{s}_0)$ guarantees that the standardized uncertainty cannot exceed $1 - 
ho_{12}(\mathbf{0})^2$.
  
  The real cautions with collocated cokriging are:
  1. **Plug-In Mean Uncertainty:** In practice, means $\mu_1, \mu_2$ are estimated rather than known, introducing unmodeled parameter estimation variance.
  2. **Screening Violations:** If MM1 is violated (e.g., surrounding secondary data carry additional spatial information beyond the collocated datum), the collocated predictor is slightly sub-optimal compared to full cokriging.

---

### 7.5 The Autokrigeability Theorem

#### 7.5.1 The Mathematical Anchor: When Is Cokriging Redundant?

A central question in spatial statistics is: *Under what mathematical conditions does incorporating cross-correlated secondary variables provide exactly zero predictive benefit over univariate kriging?*

The celebrated **Autokrigeability Theorem** (Wackernagel, 2003) answers this question definitively.

We first present the elementary, pencil-and-paper proof for a bivariate system ($p=2$), which directly exposes the algebraic mechanism, followed by the general Kronecker product formulation.

#### Theorem 7.5.1 (Autokrigeability for Bivariate Systems, $p=2$)
Suppose a bivariate spatial field $\{Z_1(\mathbf{s}), Z_2(\mathbf{s})\}$ satisfies:
1. **Intrinsic Coregionalization Model (ICM):** Both auto- and cross-covariances share the identical spatial correlation function:
   $$
   \mathbf{C}(\mathbf{h}) = \mathbf{B} 
ho(\mathbf{h}) = egin{bmatrix} B_{11} & B_{12} \ B_{21} & B_{22} \end{bmatrix} 
ho(\mathbf{h})
   $$
   with $\mathbf{B} \succ 0$ (strictly positive definite) and $
ho(\mathbf{h})$ valid on $\mathbb{R}^d$.
2. **Isotopic Sampling:** Both variables are observed at the identical set of $n$ locations: $\mathbf{s}_{1,lpha} = \mathbf{s}_{2,lpha} = \mathbf{s}_lpha$ for $lpha = 1, \dots, n$.

Then at any prediction site $\mathbf{s}_0 \in D$:
1. All secondary weights are identically zero: $oldsymbol{\lambda}_2^{	ext{OCK}} = \mathbf{0}$.
2. The primary weights equal the univariate Ordinary Kriging weights: $oldsymbol{\lambda}_1^{	ext{OCK}} = oldsymbol{\lambda}_{	ext{OK}}^*$.
3. The Ordinary Cokriging variance equals the univariate Ordinary Kriging variance: $\sigma_{	ext{OCK}}^2(\mathbf{s}_0) = \sigma_{	ext{OK}}^2(\mathbf{s}_0)$.

*Proof.*
The bivariate Ordinary Cokriging system consists of two vector equations and two scalar constraint equations:
$$
egin{aligned}
B_{11} \mathbf{R} oldsymbol{\lambda}_1 + B_{12} \mathbf{R} oldsymbol{\lambda}_2 - \mu_1 \mathbf{1}_n &= B_{11} \mathbf{r}_0 \quad 	ext{--- (Eq. 1)} \
B_{21} \mathbf{R} oldsymbol{\lambda}_1 + B_{22} \mathbf{R} oldsymbol{\lambda}_2 - \mu_2 \mathbf{1}_n &= B_{21} \mathbf{r}_0 \quad 	ext{--- (Eq. 2)} \
\mathbf{1}_n' oldsymbol{\lambda}_1 &= 1 \quad 	ext{--- (Eq. 3)} \
\mathbf{1}_n' oldsymbol{\lambda}_2 &= 0 \quad 	ext{--- (Eq. 4)}
\end{aligned}
$$
where $\mathbf{R} = [
ho(\mathbf{s}_lpha - \mathbf{s}_eta)] \in \mathbb{R}^{n 	imes n}$ and $\mathbf{r}_0 = (
ho(\mathbf{s}_1 - \mathbf{s}_0), \dots, 
ho(\mathbf{s}_n - \mathbf{s}_0))'$.
Multiply Eq. 1 by $B_{21}$ and Eq. 2 by $B_{11}$:
$$
B_{21} B_{11} \mathbf{R} oldsymbol{\lambda}_1 + B_{21} B_{12} \mathbf{R} oldsymbol{\lambda}_2 - B_{21} \mu_1 \mathbf{1}_n = B_{21} B_{11} \mathbf{r}_0
$$
$$
B_{11} B_{21} \mathbf{R} oldsymbol{\lambda}_1 + B_{11} B_{22} \mathbf{R} oldsymbol{\lambda}_2 - B_{11} \mu_2 \mathbf{1}_n = B_{11} B_{21} \mathbf{r}_0
$$
Subtract the first equation from the second:
$$
(B_{11} B_{22} - B_{12}^2) \mathbf{R} oldsymbol{\lambda}_2 - (B_{11} \mu_2 - B_{12} \mu_1) \mathbf{1}_n = \mathbf{0}
$$
Since $\mathbf{B} \succ 0$, $\det(\mathbf{B}) = B_{11} B_{22} - B_{12}^2 > 0$. Defining scalar constant $c = rac{B_{11}\mu_2 - B_{12}\mu_1}{\det(\mathbf{B})}$, we have:
$$
\mathbf{R} oldsymbol{\lambda}_2 = c \mathbf{1}_n \implies oldsymbol{\lambda}_2 = c \mathbf{R}^{-1} \mathbf{1}_n
$$
Now apply the secondary unbiasedness constraint $\mathbf{1}_n' oldsymbol{\lambda}_2 = 0$:
$$
\mathbf{1}_n' oldsymbol{\lambda}_2 = c (\mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n) = 0
$$
Because $\mathbf{R}$ is strictly positive definite, $\mathbf{R}^{-1}$ is also strictly positive definite, so the quadratic form is strictly positive: $\mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n > 0$.
Therefore, we must have $c = 0$, which immediately implies:
$$
oldsymbol{\lambda}_2 = \mathbf{0}
$$
Substituting $oldsymbol{\lambda}_2 = \mathbf{0}$ back into Eq. 1:
$$
B_{11} \mathbf{R} oldsymbol{\lambda}_1 - \mu_1 \mathbf{1}_n = B_{11} \mathbf{r}_0
$$
Dividing by $B_{11} > 0$ yields $\mathbf{R}oldsymbol{\lambda}_1 - (\mu_1 / B_{11})\mathbf{1}_n = \mathbf{r}_0$, which is identically the univariate Ordinary Kriging system. Thus $oldsymbol{\lambda}_1 = oldsymbol{\lambda}_{	ext{OK}}^*$ and $\sigma_{	ext{OCK}}^2 = \sigma_{	ext{OK}}^2$. $lacksquare$

#### 7.5.2 General Kronecker Formulation ($p \ge 2$)

For an arbitrary number of variables $p \ge 2$, under isotopic sampling and an Intrinsic Coregionalization Model $\mathbf{C}(\mathbf{h}) = \mathbf{B} 
ho(\mathbf{h})$ with $\mathbf{B} \succ 0$:
- The covariance matrix factors into the Kronecker product $oldsymbol{\Sigma} = \mathbf{B} \otimes \mathbf{R}$.
- The target cross-covariance vector factors into $\mathbf{c}_{0,1} = (\mathbf{B}\mathbf{e}_1) \otimes \mathbf{r}_0$.
- The constraint matrix factors into $\mathbf{X} = \mathbf{I}_p \otimes \mathbf{1}_n$.

The candidate solution:
$$
oldsymbol{\lambda} = \mathbf{e}_1 \otimes oldsymbol{\lambda}_{	ext{OK}}^*, \quad oldsymbol{\mu} = (\mathbf{B}\mathbf{e}_1) \mu_{	ext{OK}}^*
$$
satisfies the Ordinary Cokriging block equations identically:
$$
egin{aligned}
oldsymbol{\Sigma} oldsymbol{\lambda} - \mathbf{X}oldsymbol{\mu} &= (\mathbf{B} \otimes \mathbf{R})(\mathbf{e}_1 \otimes oldsymbol{\lambda}_{	ext{OK}}^*) - (\mathbf{I}_p \otimes \mathbf{1}_n)((\mathbf{B}\mathbf{e}_1) \mu_{	ext{OK}}^*) \
&= (\mathbf{B}\mathbf{e}_1) \otimes (\mathbf{R}oldsymbol{\lambda}_{	ext{OK}}^*) - (\mathbf{B}\mathbf{e}_1) \otimes (\mu_{	ext{OK}}^* \mathbf{1}_n) \
&= (\mathbf{B}\mathbf{e}_1) \otimes (\mathbf{R}oldsymbol{\lambda}_{	ext{OK}}^* - \mu_{	ext{OK}}^* \mathbf{1}_n) = (\mathbf{B}\mathbf{e}_1) \otimes \mathbf{r}_0 = \mathbf{c}_{0,1}
\end{aligned}
$$
By uniqueness of the solution to the non-singular augmented linear system, this candidate is the unique BLUP solution! Secondary weights are identically zero.

#### 7.5.3 Algorithmic Translation: Autokrigeability Diagnostic

```python
def autokrigeability_diagnostic(B_matrices, rho_models, is_isotopic):
    """
    Diagnoses whether an Ordinary Cokriging problem satisfies the
    Autokrigeability Theorem, rendering secondary data redundant.
    
    Parameters:
        B_matrices: List of K coregionalization matrices (p x p).
        rho_models: List of K correlation functions rho_k(h).
        is_isotopic: Boolean indicating if all variables are sampled at identical sites.
        
    Returns:
        is_autokrigeable: Boolean.
        explanation: Diagnostic explanation string.
    """
    if not is_isotopic:
        return False, "Sampling is heterotopic; secondary data provides spatial coverage in voids."
        
    K = len(B_matrices)
    p = len(B_matrices[0])
    
    # Check if all correlation models are identical by evaluating at multiple test lags
    test_lags = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    identical_rhos = True
    for k in range(1, K):
        for h in test_lags:
            if abs(rho_models[k](h) - rho_models[0](h)) > 1e-6:
                identical_rhos = False
                break
        if not identical_rhos:
            break
            
    if identical_rhos:
        # Sum of structures with same correlation function collapses to a single intrinsic structure
        return True, "SYSTEM IS AUTOKRIGEABLE: All structures share the same correlation function " \
                     "under isotopic sampling. Secondary weights will be 0."
                     
    # Check if cross-covariances involving the primary variable are proportional across structures
    # B_k[0, j] = c_k * B_0[0, j] for all j
    primary_proportional = True
    for k in range(1, K):
        if B_matrices[0][0][0] == 0:
            primary_proportional = False
            break
        scale = B_matrices[k][0][0] / B_matrices[0][0][0]
        for j in range(p):
            if abs(B_matrices[k][0][j] - scale * B_matrices[0][0][j]) > 1e-6:
                primary_proportional = False
                break
                
    if primary_proportional:
        return True, "SYSTEM IS AUTOKRIGEABLE: Primary auto- and cross-covariances are proportional " \
                     "across all spatial scales. Secondary data will receive zero weight."
                     
    return False, "System is NOT autokrigeable: Multi-scale spatial coregionalization with distinct " \
                  "correlation scales. Cokriging will add predictive value."
```

#### 7.5.4 Intuition, Edge Cases & Diagnostic Verification

```
                            THE AUTOKRIGEABILITY DECISION FLOW
           Does Cokriging improve upon univariate Ordinary Kriging?

                     Is sampling configuration Isotopic?
                               /                                          YES              NO  ====>  COKRIGING ADDS SUBSTANTIAL VALUE
                             /                           (Secondary data fills spatial voids)
              Do variables share identical
              spatial correlation functions?
                    /                               YES              NO  =============>  COKRIGING ADDS VALUE
                  /                                     (Different spatial scales)
    AUTOKRIGEABILITY HOLDS!
    Secondary weights = 0
    Cokriging == Univariate OK
```

**The Practitioner's Fallacy:**
A frequent mistake among spatial practitioners is spending weeks fitting complex multivariate cross-variograms to geochemical borehole assays where gold, silver, and copper share the exact same spatial correlation structure, only to discover that the cokriging map is pixel-for-pixel identical to univariate kriging!
- **When is Cokriging Worth the Computational Effort?**
  1. **Heterotopic Sampling (Massive Gain):** When the secondary variable is observed at thousands of locations where the primary variable is absent (e.g., $81\%$ variance reduction).
  2. **Multi-Scale Isotopic Coregionalization (Modest Gain):** When variables interact differently across spatial scales (e.g., positive correlation at regional scales due to geology, but negative correlation at short scales due to geochemical scavenging). However, the variance reduction under isotopic sampling is typically modest (often $0.5\%$ to $3\%$).

---


### Key Ideas and Takeaways: Multivariate Spatial Prediction and Cokriging

| Concept | Mathematical Form | Master's-Level Interpretation | Practical Diagnostic Rule |
| :--- | :--- | :--- | :--- |
| **Matrix Cross-Covariance** | $\mathbf{C}(\mathbf{h}) = [C_{ij}(\mathbf{h})]_{i,j=1}^p$ | Cross-covariances are generally asymmetric ($C_{ij}(\mathbf{h}) 
e C_{ij}(-\mathbf{h})$), satisfying transpose symmetry $\mathbf{C}(\mathbf{h}) = \mathbf{C}(-\mathbf{h})'$. | Must satisfy Cauchy–Schwarz bound $|C_{ij}(\mathbf{h})| \le \sqrt{C_{ii}(\mathbf{0})C_{jj}(\mathbf{0})}$ and $\mathbf{C}(\mathbf{0}) \succeq 0$. |
| **Linear Model of Coregionalization (LMC)** | $\mathbf{C}(\mathbf{h}) = \sum_{k=1}^K \mathbf{B}_k 
ho_k(\mathbf{h})$ | Models vector field as superposition of orthogonal spatial factors; guarantees joint positive semidefiniteness. | Check that every coregionalization matrix $\mathbf{B}_k \succeq 0$ via Sylvester's criterion. |
| **Linear Model Unification** | Cokriging on stacked $\mathbf{Z} \in \mathbb{R}^N$ | SCK is Simple Kriging on $\mathbf{Z}$; OCK is Universal Kriging on $\mathbf{Z}$ with block-indicator design matrix $\mathbf{X}$. | Solved once using standard linear algebra without ad-hoc rules. |
| **Zero-Sum Secondary Constraints** | $\sum_{eta=1}^{n_j} \lambda_{jeta} = 0 \; (j \ge 2)$ | Protects the primary prediction against unknown constant offsets in secondary variable baselines. | Forces secondary weights to form spatial dipoles (positive proxy at target, negative anchor at primary sites). |
| **Ordinary Cokriging Variance** | $\sigma_{	ext{OCK}}^2 = C_{11}(\mathbf{0}) - oldsymbol{\lambda}'\mathbf{c}_{0,1} + \mu_1$ | Exact error variance derived from Lagrangian; multiplier $\mu_1$ carries a positive sign. | Code must use $+ \mu_1$ (with $oldsymbol{\Sigma}oldsymbol{\lambda} - \mathbf{X}oldsymbol{\mu} = \mathbf{c}_{0,1}$). |
| **Markov Model 1 (MM1)** | $
ho_{12}(\mathbf{h}) = 
ho_{12}(\mathbf{0})
ho_{22}(\mathbf{h})$ | Assumes collocated secondary datum conditionally screens target from all other secondary data. | Standardized Simple Cokriging yields legitimate variance floor $1 - 
ho_{12}(\mathbf{0})^2$; requires similar ranges. |
| **Autokrigeability Theorem** | $\mathbf{C}(\mathbf{h}) = \mathbf{B}
ho(\mathbf{h})$ with $S_i = S$ | Under isotopic sampling and identical correlation scales, secondary measurements receive zero weight. | Cokriging only pays under heterotopic sampling (massive gain) or multi-scale coregionalization (modest gain). |

---

### 7.6 Analytical Exercises and Step-by-Step Solutions

#### Exercise 7.1: Sylvester's Criterion, Lag-Zero Consistency, and Coregionalization Admissibility
**Problem Statement:**
A geostatistician proposes an empirical bivariate LMC with a single spherical correlation structure:
$$
\mathbf{C}(\mathbf{h}) = \mathbf{B} 
ho(h), \quad 	ext{where } \mathbf{B} = egin{bmatrix} 4.0 & 5.0 \ 5.0 & 6.0 \end{bmatrix}
$$
where $
ho(h) = (1 - 1.5(h/10) + 0.5(h/10)^3)\mathbb{1}_{\{h \le 10\}}$.
- **(a)** Using the necessary condition that the lag-zero covariance matrix $\mathbf{C}(\mathbf{0})$ must be a valid covariance matrix, determine whether this model is mathematically admissible in $\mathbb{R}^2$.
- **(b)** Construct a linear combination of co-located observations $Y = a_1 Z_1(\mathbf{s}) + a_2 Z_2(\mathbf{s})$ that produces a strictly negative variance, and explain why this contradicts the fundamental definition of variance as an expected squared deviation.

**Complete Step-by-Step Solution:**
- **(a)** 
  1. For any stationary multivariate random field $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), Z_2(\mathbf{s}))'$, the lag-zero matrix is the variance-covariance matrix of the co-located random vector:
     $$
     \mathbf{C}(\mathbf{0}) = \operatorname{Var}(\mathbf{Z}(\mathbf{s})) = egin{bmatrix} \operatorname{Var}(Z_1(\mathbf{s})) & \operatorname{Cov}(Z_1(\mathbf{s}), Z_2(\mathbf{s})) \ \operatorname{Cov}(Z_2(\mathbf{s}), Z_1(\mathbf{s})) & \operatorname{Var}(Z_2(\mathbf{s})) \end{bmatrix}
     $$
  2. Because $
ho(0) = 1$, the lag-zero covariance matrix for this model is $\mathbf{C}(\mathbf{0}) = \mathbf{B} 
ho(0) = \mathbf{B}$.
  3. Every covariance matrix must be positive semidefinite ($\mathbf{B} \succeq 0$). By Sylvester's Criterion, a $2 	imes 2$ symmetric matrix is positive semidefinite if and only if its diagonal entries are non-negative and its determinant is non-negative:
     $$
     \det(\mathbf{B}) = B_{11} B_{22} - B_{12}^2 = (4.0)(6.0) - (5.0)^2 = 24.0 - 25.0 = -1.0
     $$
  4. Because $\det(\mathbf{B}) = -1.0 < 0$, $\mathbf{B}$ is indefinite, possessing one positive eigenvalue and one negative eigenvalue:
     $$
     \lambda = rac{\operatorname{tr}(\mathbf{B}) \pm \sqrt{\operatorname{tr}(\mathbf{B})^2 - 4\det(\mathbf{B})}}{2} = rac{10 \pm \sqrt{100 - 4(-1)}}{2} = rac{10 \pm \sqrt{104}}{2} pprox 5 \pm 5.0990
     $$
     giving $\lambda_1 pprox 10.0990 > 0$ and $\lambda_2 pprox -0.0990 < 0$.
  5. Because $\mathbf{C}(\mathbf{0})$ has a strictly negative eigenvalue, it cannot be a valid covariance matrix. Therefore, the proposed model is strictly mathematically inadmissible. $lacksquare$

- **(b)**
  1. We construct a linear combination $Y = a_1 Z_1(\mathbf{s}) + a_2 Z_2(\mathbf{s}) = \mathbf{a}'\mathbf{Z}(\mathbf{s})$ along the eigenvector $\mathbf{v}_2$ corresponding to the negative eigenvalue $\lambda_2 pprox -0.0990$.
  2. The eigenvector satisfies $(\mathbf{B} - \lambda_2 \mathbf{I})\mathbf{v}_2 = \mathbf{0}$:
     $$
     (4.0 - (-0.0990)) a_1 + 5.0 a_2 = 0 \implies 4.099 a_1 + 5.0 a_2 = 0 \implies \mathbf{a} pprox egin{bmatrix} 5.0 \ -4.1 \end{bmatrix}
     $$
  3. Evaluating the variance of $Y = 5.0 Z_1(\mathbf{s}) - 4.1 Z_2(\mathbf{s})$:
     $$
     egin{aligned}
     \operatorname{Var}(Y) &= a_1^2 \operatorname{Var}(Z_1) + a_2^2 \operatorname{Var}(Z_2) + 2 a_1 a_2 \operatorname{Cov}(Z_1, Z_2) \
     &= (5.0)^2 (4.0) + (-4.1)^2 (6.0) + 2(5.0)(-4.1)(5.0) \
     &= 25.0(4.0) + 16.81(6.0) - 205.0 \
     &= 100.0 + 100.86 - 205.0 = -4.14 < 0
     \end{aligned}
     $$
  4. The calculated variance is $-4.14$. In probability theory, variance is defined as the expectation of a non-negative squared deviation:
     $$
     \operatorname{Var}(Y) = \operatorname{E}[(Y - \operatorname{E}[Y])^2] \ge 0
     $$
     Because $(Y - \operatorname{E}[Y])^2$ is non-negative almost surely, its expectation cannot be strictly negative. A negative variance is a direct mathematical contradiction resulting from an inadmissible covariance model. $lacksquare$

---

#### Exercise 7.2: Unabridged Algebraic Proof of the Autokrigeability Theorem for a Bivariate System ($p=2$)
**Problem Statement:**
Let $\{Z_1(\mathbf{s}), Z_2(\mathbf{s})\}$ be an intrinsically coregionalized bivariate field with $\mathbf{C}(\mathbf{h}) = \mathbf{B}
ho(\mathbf{h})$, observed at identical locations $\mathbf{s}_1, \dots, \mathbf{s}_n$. 
Prove algebraically from the scalar Ordinary Cokriging equations that the secondary weights satisfy $\lambda_{2lpha} = 0$ for all $lpha = 1, \dots, n$.

**Complete Step-by-Step Solution:**
1. The bivariate Ordinary Cokriging system consists of two vector equations and two scalar constraint equations:
   $$
   egin{aligned}
   B_{11} \mathbf{R} oldsymbol{\lambda}_1 + B_{12} \mathbf{R} oldsymbol{\lambda}_2 - \mu_1 \mathbf{1}_n &= B_{11} \mathbf{r}_0 \quad 	ext{--- (Eq. 1)} \
   B_{21} \mathbf{R} oldsymbol{\lambda}_1 + B_{22} \mathbf{R} oldsymbol{\lambda}_2 - \mu_2 \mathbf{1}_n &= B_{21} \mathbf{r}_0 \quad 	ext{--- (Eq. 2)} \
   \mathbf{1}_n' oldsymbol{\lambda}_1 &= 1 \quad 	ext{--- (Eq. 3)} \
   \mathbf{1}_n' oldsymbol{\lambda}_2 &= 0 \quad 	ext{--- (Eq. 4)}
   \end{aligned}
   $$
   where $\mathbf{R} = [
ho(\mathbf{s}_lpha - \mathbf{s}_eta)] \in \mathbb{R}^{n 	imes n}$ and $\mathbf{r}_0 = (
ho(\mathbf{s}_1 - \mathbf{s}_0), \dots, 
ho(\mathbf{s}_n - \mathbf{s}_0))'$.
2. Multiply Eq. 1 by $B_{21}$ and Eq. 2 by $B_{11}$:
   $$
   egin{aligned}
   B_{21} B_{11} \mathbf{R} oldsymbol{\lambda}_1 + B_{21} B_{12} \mathbf{R} oldsymbol{\lambda}_2 - B_{21} \mu_1 \mathbf{1}_n &= B_{21} B_{11} \mathbf{r}_0 \
   B_{11} B_{21} \mathbf{R} oldsymbol{\lambda}_1 + B_{11} B_{22} \mathbf{R} oldsymbol{\lambda}_2 - B_{11} \mu_2 \mathbf{1}_n &= B_{11} B_{21} \mathbf{r}_0
   \end{aligned}
   $$
3. Subtract the first equation from the second:
   $$
   (B_{11} B_{22} - B_{12}^2) \mathbf{R} oldsymbol{\lambda}_2 - (B_{11} \mu_2 - B_{12} \mu_1) \mathbf{1}_n = \mathbf{0}
   $$
4. Let $\det(\mathbf{B}) = B_{11} B_{22} - B_{12}^2 > 0$ (assuming $\mathbf{B}$ is strictly positive definite). Define scalar constant $c = rac{B_{11} \mu_2 - B_{12} \mu_1}{\det(\mathbf{B})}$. The equation simplifies to:
   $$
   \mathbf{R} oldsymbol{\lambda}_2 = c \mathbf{1}_n
   $$
5. Multiply by $\mathbf{R}^{-1}$:
   $$
   oldsymbol{\lambda}_2 = c \mathbf{R}^{-1} \mathbf{1}_n
   $$
6. Apply the secondary unbiasedness constraint (Eq. 4: $\mathbf{1}_n' oldsymbol{\lambda}_2 = 0$):
   $$
   \mathbf{1}_n' oldsymbol{\lambda}_2 = c (\mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n) = 0
   $$
7. Because $\mathbf{R}$ is a strictly positive definite matrix, its inverse $\mathbf{R}^{-1}$ is also strictly positive definite. For the non-zero vector $\mathbf{1}_n$, the quadratic form is strictly positive:
   $$
   \mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n > 0
   $$
8. Therefore, the only way $c (\mathbf{1}_n' \mathbf{R}^{-1} \mathbf{1}_n) = 0$ can hold is if:
   $$
   c = 0
   $$
9. Substituting $c = 0$ back into Step 5:
   $$
   oldsymbol{\lambda}_2 = 0 \cdot \mathbf{R}^{-1} \mathbf{1}_n = \mathbf{0}
   $$
10. Since $oldsymbol{\lambda}_2 = \mathbf{0}$, Eq. 1 reduces to $B_{11} \mathbf{R} oldsymbol{\lambda}_1 - \mu_1 \mathbf{1}_n = B_{11} \mathbf{r}_0$, which is identically the univariate Ordinary Kriging system. $lacksquare$


---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 7: Multivariate Geostatistics and Cokriging Theory — Companion Exercises and Complete Solutions](ch07_exercises.md)
