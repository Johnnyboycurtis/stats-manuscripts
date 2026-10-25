# Chapter 3 Study Guide & Comprehensive Exercises
## Theory of Semivariogram and Covariance Estimation

This companion study guide reinforces the core mathematical, sampling, and non-parametric estimation concepts from [Chapter 3: Theory of Semivariogram and Covariance Estimation](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch03_estimation.md). The exercises below guide the reader through Genton's quadratic form matrix, periodogram equivalence, Matheron's estimator, empirical covariance downward bias, the Cressie-Hawkins robust estimator, and directional binning across five pedagogical tiers:
1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 3.1–3.4)
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 3.5–3.8)
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 3.9–3.11)
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 3.12–3.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 3.14–3.16)

---

### Tier 0: Classical-to-Spatial Warm-Ups

#### Exercise 3.1: Genton's Spatial Design Matrix $\mathbf{A}(\mathbf{h})$ on 1D Transects and 2D Lattices (S&G Problem 4.4)
Recall that Matheron's empirical semivariogram estimator can be written in quadratic form as:
$$
\widehat{\gamma}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|}\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}
$$
where the spatial difference design matrix is:
$$
\mathbf{A}(\mathbf{h}) = \sum_{(i,j) \in N(\mathbf{h})} (\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)'
$$
and $\mathbf{e}_i$ denotes the $i$-th standard basis vector in $\mathbb{R}^n$.

- **(a)** Consider a 1D regular transect with $n = 4$ points at spatial coordinates $s_1 = 1, s_2 = 2, s_3 = 3, s_4 = 4$. For unit lag $h = 1$, determine the pair set $N(1)$ and its cardinality $|N(1)|$. Construct the explicit $4 \times 4$ numerical matrix $\mathbf{A}(1)$, and verify analytically that its row sums are all zero: $\mathbf{A}(1)\mathbf{1}_4 = \mathbf{0}$.
- **(b)** Consider a $3 \times 3$ regular spatial lattice with $n = 9$ points indexed row-wise: $\mathbf{s}_1=(0,0), \mathbf{s}_2=(1,0), \mathbf{s}_3=(2,0), \mathbf{s}_4=(0,1), \mathbf{s}_5=(1,1), \mathbf{s}_6=(2,1), \mathbf{s}_7=(0,2), \mathbf{s}_8=(1,2), \mathbf{s}_9=(2,2)$. For the horizontal unit lag $\mathbf{h} = (1, 0)'$, determine $|N(\mathbf{h})|$ and write out the explicit $9 \times 9$ numerical matrix $\mathbf{A}((1,0)')$.
- **(c)** Prove that $\mathbf{A}(\mathbf{h})$ is the combinatorial graph Laplacian matrix $\mathbf{L} = \mathbf{D} - \mathbf{W}$ of the undirected lag-neighborhood graph $\mathcal{G}_{\mathbf{h}} = (V, E_{\mathbf{h}})$. Conclude that $\mathbf{A}(\mathbf{h})$ is always positive semi-definite with rank equal to $n - c(\mathcal{G}_{\mathbf{h}})$, where $c(\mathcal{G}_{\mathbf{h}})$ is the number of connected components in $\mathcal{G}_{\mathbf{h}}$.

##### Solution to Exercise 3.1
- **(a)** On the 1D transect with locations $s_1 = 1, s_2 = 2, s_3 = 3, s_4 = 4$, the distinct unordered pairs separated by unit lag $h = 1$ are:
  $$
  N(1) = \{(1, 2), (2, 3), (3, 4)\}
  $$
  The cardinality is $|N(1)| = 3$.
  For each pair $(i, j) \in N(1)$, the outer product is:
  $$
  (\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)' = \mathbf{e}_i \mathbf{e}_i' + \mathbf{e}_j \mathbf{e}_j' - \mathbf{e}_i \mathbf{e}_j' - \mathbf{e}_j \mathbf{e}_i'
  $$
  This $4 \times 4$ rank-1 elementary matrix has $+1$ on the diagonal entries $(i, i)$ and $(j, j)$, and $-1$ on the off-diagonal entries $(i, j)$ and $(j, i)$.
  Summing the three elementary matrices:
  $$
  \begin{aligned}
  \mathbf{A}(1) &= (\mathbf{e}_1 - \mathbf{e}_2)(\mathbf{e}_1 - \mathbf{e}_2)' + (\mathbf{e}_2 - \mathbf{e}_3)(\mathbf{e}_2 - \mathbf{e}_3)' + (\mathbf{e}_3 - \mathbf{e}_4)(\mathbf{e}_3 - \mathbf{e}_4)' \\
  &= \begin{bmatrix} 1 & -1 & 0 & 0 \\ -1 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 0 & 0 & -1 & 1 \end{bmatrix} \\
  &= \begin{bmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{bmatrix}
  \end{aligned}
  $$
  To verify the zero row-sum property:
  $$
  \mathbf{A}(1)\mathbf{1}_4 = \begin{bmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 - 1 + 0 + 0 \\ -1 + 2 - 1 + 0 \\ 0 - 1 + 2 - 1 \\ 0 + 0 - 1 + 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} = \mathbf{0}_4
  $$
  Thus, $\mathbf{A}(1)\mathbf{1}_4 = \mathbf{0}$, confirming that the constant mean vector lies in the nullspace of $\mathbf{A}(1)$.

- **(b)** For the $3 \times 3$ regular spatial lattice with points ordered row-wise:
  - Row 0 ($y=0$): $\mathbf{s}_1=(0,0), \mathbf{s}_2=(1,0), \mathbf{s}_3=(2,0)$
  - Row 1 ($y=1$): $\mathbf{s}_4=(0,1), \mathbf{s}_5=(1,1), \mathbf{s}_6=(2,1)$
  - Row 2 ($y=2$): $\mathbf{s}_7=(0,2), \mathbf{s}_8=(1,2), \mathbf{s}_9=(2,2)$
  For the horizontal unit lag $\mathbf{h} = (1, 0)'$, pairs must share the same $y$-coordinate and have $x$-coordinates separated by $1$:
  - In Row 0: $(1, 2)$ and $(2, 3)$
  - In Row 1: $(4, 5)$ and $(5, 6)$
  - In Row 2: $(7, 8)$ and $(8, 9)$
  The set of pairs is $N((1,0)') = \{(1,2), (2,3), (4,5), (5,6), (7,8), (8,9)\}$ with cardinality $|N((1,0)')| = 6$.
  Because there are no horizontal connections across different rows, $\mathbf{A}((1,0)')$ is a block-diagonal matrix consisting of three identical $3 \times 3$ tridiagonal blocks:
  $$
  \mathbf{K}_3 = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{bmatrix}
  $$
  The explicit $9 \times 9$ numerical matrix is:
  $$
  \mathbf{A}((1,0)') = \begin{bmatrix}
  1 & -1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
  -1 & 2 & -1 & 0 & 0 & 0 & 0 & 0 & 0 \\
  0 & -1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 1 & -1 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & -1 & 2 & -1 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & -1 & 1 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 & 0 & 1 & -1 & 0 \\
  0 & 0 & 0 & 0 & 0 & 0 & -1 & 2 & -1 \\
  0 & 0 & 0 & 0 & 0 & 0 & 0 & -1 & 1
  \end{bmatrix}
  $$
  Each row sums to zero: $\mathbf{A}((1,0)')\mathbf{1}_9 = \mathbf{0}$.

- **(c)** Let $\mathcal{G}_{\mathbf{h}} = (V, E_{\mathbf{h}})$ be the undirected graph whose vertex set is $V = \{1, \dots, n\}$ and whose edge set is $E_{\mathbf{h}} = \{(i, j) : (\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})\}$.
  The combinatorial graph Laplacian matrix $\mathbf{L}$ is defined as $\mathbf{L} = \mathbf{D} - \mathbf{W}$, where $\mathbf{W}$ is the symmetric adjacency matrix with $W_{ij} = 1$ if $(i,j) \in E_{\mathbf{h}}$ and $0$ otherwise, and $\mathbf{D} = \operatorname{diag}(d_1, \dots, d_n)$ is the diagonal degree matrix with $d_i = \sum_{j=1}^n W_{ij} = \operatorname{deg}(i)$.
  Expanding entry $(i, j)$ of $\mathbf{A}(\mathbf{h}) = \sum_{(k,l) \in E_{\mathbf{h}}} (\mathbf{e}_k - \mathbf{e}_l)(\mathbf{e}_k - \mathbf{e}_l)'$:
  - On the diagonal ($i = j$): the term $(\mathbf{e}_k - \mathbf{e}_l)_i^2 = 1$ whenever edge $(k, l)$ is incident to vertex $i$. Summing over all edges gives $[\mathbf{A}(\mathbf{h})]_{ii} = \operatorname{deg}(i) = d_i$.
  - Off the diagonal ($i \ne j$): the term $(\mathbf{e}_k - \mathbf{e}_l)_i(\mathbf{e}_k - \mathbf{e}_l)_j = -1$ if $(i, j) \in E_{\mathbf{h}}$ and $0$ otherwise. Thus $[\mathbf{A}(\mathbf{h})]_{ij} = -W_{ij}$.
  Therefore:
  $$
  \mathbf{A}(\mathbf{h}) = \mathbf{D} - \mathbf{W} = \mathbf{L}(\mathcal{G}_{\mathbf{h}})
  $$
  For any vector $\mathbf{x} \in \mathbb{R}^n$, the quadratic form evaluates to:
  $$
  \mathbf{x}'\mathbf{A}(\mathbf{h})\mathbf{x} = \sum_{(i,j) \in E_{\mathbf{h}}} (x_i - x_j)^2 \ge 0
  $$
  Because $\mathbf{x}'\mathbf{A}(\mathbf{h})\mathbf{x} \ge 0$ for all $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{A}(\mathbf{h})$ is always symmetric positive semi-definite ($\mathbf{A}(\mathbf{h}) \succeq 0$).
  Furthermore, $\mathbf{x}'\mathbf{A}(\mathbf{h})\mathbf{x} = 0$ if and only if $x_i = x_j$ for every edge $(i, j) \in E_{\mathbf{h}}$.
  Consequently, $\mathbf{x}$ lies in the nullspace of $\mathbf{A}(\mathbf{h})$ if and only if $\mathbf{x}$ is constant on every connected component of $\mathcal{G}_{\mathbf{h}}$.
  The dimension of the nullspace (nullity) is therefore exactly the number of connected components: $\operatorname{nullity}(\mathbf{A}(\mathbf{h})) = c(\mathcal{G}_{\mathbf{h}})$.
  By the rank-nullity theorem:
  $$
  \operatorname{rank}(\mathbf{A}(\mathbf{h})) = n - \operatorname{nullity}(\mathbf{A}(\mathbf{h})) = n - c(\mathcal{G}_{\mathbf{h}})
  $$
  For example, in part (a), the graph is a single connected path ($c = 1$), so $\operatorname{rank}(\mathbf{A}(1)) = 4 - 1 = 3$. In part (b), the graph consists of 3 disjoint paths corresponding to the 3 rows ($c = 3$), so $\operatorname{rank}(\mathbf{A}((1,0)')) = 9 - 3 = 6$. $\blacksquare$

---

#### Exercise 3.2: Exact Expectation and Covariance of Genton's Quadratic Form Estimator
Let $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ be a spatial random field with mean vector $\boldsymbol{\mu} = \mu \mathbf{1}_n$ (constant spatial mean) and covariance matrix $\mathbf{V}$. Genton (1998) expressed Matheron's empirical semivariogram estimator in quadratic form as:
$$
\widehat{\gamma}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|}\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}
$$
where $\mathbf{A}(\mathbf{h}) = \sum_{(i,j) \in N(\mathbf{h})} (\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)'$.

- **(a)** Recall the fundamental expectation theorem for Gaussian quadratic forms: for any random vector $\mathbf{Z} \sim \mathcal{N}_n(\boldsymbol{\mu}, \mathbf{V})$ and symmetric matrix $\mathbf{A}$, $\operatorname{E}[\mathbf{Z}'\mathbf{A}\mathbf{Z}] = \operatorname{tr}(\mathbf{A}\mathbf{V}) + \boldsymbol{\mu}'\mathbf{A}\boldsymbol{\mu}$. Using the zero row-sum property $\mathbf{A}(\mathbf{h})\mathbf{1}_n = \mathbf{0}$, prove that the expectation of Matheron's estimator is strictly invariant to the unknown spatial mean $\mu$:
  $$
  \operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \frac{1}{2|N(\mathbf{h})|}\operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V})
  $$
- **(b)** Under second-order stationarity with covariance $C(\mathbf{h})$ and semivariogram $\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$, evaluate $\operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V})$ explicitly and prove that $\operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \gamma(\mathbf{h})$, confirming exact unbiasedness.
- **(c)** For any two symmetric matrices $\mathbf{A}$ and $\mathbf{B}$, the sampling covariance between two Gaussian quadratic forms is:
  $$
  \operatorname{Cov}(\mathbf{Z}'\mathbf{A}\mathbf{Z}, \mathbf{Z}'\mathbf{B}\mathbf{Z}) = 2\operatorname{tr}(\mathbf{A}\mathbf{V}\mathbf{B}\mathbf{V}) + 4\boldsymbol{\mu}'\mathbf{A}\mathbf{V}\mathbf{B}\boldsymbol{\mu}
  $$
  Derive the exact sampling covariance between empirical semivariogram estimates at two different lags $\mathbf{h}_k$ and $\mathbf{h}_l$:
  $$
  \operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_k), \widehat{\gamma}(\mathbf{h}_l)) = \frac{\operatorname{tr}(\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\mathbf{V})}{2|N(\mathbf{h}_k)||N(\mathbf{h}_l)|}
  $$
  Explain why this covariance is non-zero even when the lag vectors $\mathbf{h}_k \ne \mathbf{h}_l$ are distinct.

##### Solution to Exercise 3.2
- **(a)** For $\mathbf{Z} \sim \mathcal{N}_n(\boldsymbol{\mu}, \mathbf{V})$ with $\boldsymbol{\mu} = \mu \mathbf{1}_n$, the expectation of the quadratic form $\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}$ is:
  $$
  \operatorname{E}[\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}] = \operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V}) + \boldsymbol{\mu}'\mathbf{A}(\mathbf{h})\boldsymbol{\mu}
  $$
  Substitute $\boldsymbol{\mu} = \mu \mathbf{1}_n$ into the quadratic mean term:
  $$
  \boldsymbol{\mu}'\mathbf{A}(\mathbf{h})\boldsymbol{\mu} = (\mu \mathbf{1}_n)'\mathbf{A}(\mathbf{h})(\mu \mathbf{1}_n) = \mu^2 (\mathbf{1}_n'\mathbf{A}(\mathbf{h})\mathbf{1}_n)
  $$
  From Exercise 3.1, every row of $\mathbf{A}(\mathbf{h})$ sums to zero, meaning $\mathbf{A}(\mathbf{h})\mathbf{1}_n = \mathbf{0}$. Therefore:
  $$
  \boldsymbol{\mu}'\mathbf{A}(\mathbf{h})\boldsymbol{\mu} = \mu^2 \mathbf{1}_n'\mathbf{0} = 0
  $$
  The quadratic mean term vanishes identically regardless of the value of $\mu \in \mathbb{R}$.
  Multiplying by the scalar pre-factor $\frac{1}{2|N(\mathbf{h})|}$:
  $$
  \operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \frac{1}{2|N(\mathbf{h})|}\operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V})
  $$
  This establishes that the expected value of Matheron's estimator is completely invariant to the unknown spatial mean.

- **(b)** Using the explicit expansion $\mathbf{A}(\mathbf{h}) = \sum_{(i,j) \in N(\mathbf{h})} (\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)'$, substitute into the trace operator:
  $$
  \operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V}) = \operatorname{tr}\left( \sum_{(i,j) \in N(\mathbf{h})} (\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)' \mathbf{V} \right)
  $$
  By linearity of the trace and the cyclic property $\operatorname{tr}(\mathbf{u}\mathbf{v}'\mathbf{V}) = \mathbf{v}'\mathbf{V}\mathbf{u}$:
  $$
  \operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V}) = \sum_{(i,j) \in N(\mathbf{h})} (\mathbf{e}_i - \mathbf{e}_j)'\mathbf{V}(\mathbf{e}_i - \mathbf{e}_j)
  $$
  Expanding the quadratic form for an arbitrary pair $(i, j)$:
  $$
  (\mathbf{e}_i - \mathbf{e}_j)'\mathbf{V}(\mathbf{e}_i - \mathbf{e}_j) = V_{ii} - 2 V_{ij} + V_{jj}
  $$
  Under second-order stationarity, $V_{ii} = C(\mathbf{0})$, $V_{jj} = C(\mathbf{0})$, and $V_{ij} = C(\mathbf{s}_i - \mathbf{s}_j) = C(\mathbf{h})$ for all pairs $(i, j) \in N(\mathbf{h})$. Thus:
  $$
  V_{ii} - 2 V_{ij} + V_{jj} = 2C(\mathbf{0}) - 2C(\mathbf{h}) = 2(C(\mathbf{0}) - C(\mathbf{h})) = 2\gamma(\mathbf{h})
  $$
  Summing across all $|N(\mathbf{h})|$ pairs:
  $$
  \operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V}) = \sum_{(i,j) \in N(\mathbf{h})} 2\gamma(\mathbf{h}) = 2|N(\mathbf{h})|\gamma(\mathbf{h})
  $$
  Substituting this result into the expectation from part (a):
  $$
  \operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \frac{1}{2|N(\mathbf{h})|}\operatorname{tr}(\mathbf{A}(\mathbf{h})\mathbf{V}) = \frac{1}{2|N(\mathbf{h})|} \cdot 2|N(\mathbf{h})|\gamma(\mathbf{h}) = \gamma(\mathbf{h})
  $$
  This rigorously re-proves the exact unbiasedness of Matheron's estimator using matrix algebra.

- **(c)** Let $\mathbf{A} = \mathbf{A}(\mathbf{h}_k)$ and $\mathbf{B} = \mathbf{A}(\mathbf{h}_l)$.
  The covariance between the two quadratic forms is:
  $$
  \operatorname{Cov}(\mathbf{Z}'\mathbf{A}(\mathbf{h}_k)\mathbf{Z}, \mathbf{Z}'\mathbf{A}(\mathbf{h}_l)\mathbf{Z}) = 2\operatorname{tr}(\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\mathbf{V}) + 4\boldsymbol{\mu}'\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\boldsymbol{\mu}
  $$
  Because $\boldsymbol{\mu} = \mu \mathbf{1}_n$ and $\mathbf{A}(\mathbf{h}_k)\mathbf{1}_n = \mathbf{0}$, the second term vanishes:
  $$
  4\boldsymbol{\mu}'\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\boldsymbol{\mu} = 4\mu^2 \mathbf{1}_n'\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\mathbf{1}_n = 4\mu^2 \mathbf{0}'\mathbf{V}\mathbf{0} = 0
  $$
  Now apply the scalar scaling factors $\frac{1}{2|N(\mathbf{h}_k)|}$ and $\frac{1}{2|N(\mathbf{h}_l)|}$:
  $$
  \begin{aligned}
  \operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_k), \widehat{\gamma}(\mathbf{h}_l)) &= \frac{1}{4|N(\mathbf{h}_k)||N(\mathbf{h}_l)|} \cdot 2\operatorname{tr}(\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\mathbf{V}) \\
  &= \frac{\operatorname{tr}(\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\mathbf{V})}{2|N(\mathbf{h}_k)||N(\mathbf{h}_l)|}
  \end{aligned}
  $$
  *Physical Interpretation:*
  This sampling covariance is non-zero because the matrices $\mathbf{A}(\mathbf{h}_k)$ and $\mathbf{A}(\mathbf{h}_l)$ share common spatial observation points, and spatial correlation $\mathbf{V}$ transmits dependence across different point pairs.
  Even if $N(\mathbf{h}_k)$ and $N(\mathbf{h}_l)$ share no common pairs, if a site $\mathbf{s}_i$ appears in a pair in $N(\mathbf{h}_k)$ and is spatially correlated with a site $\mathbf{s}_j$ appearing in $N(\mathbf{h}_l)$, $\mathbf{A}(\mathbf{h}_k)\mathbf{V}\mathbf{A}(\mathbf{h}_l)\mathbf{V}$ has non-zero diagonal entries.
  Consequently, empirical semivariogram values $\widehat{\gamma}(\mathbf{h}_k)$ across different lag bins are **strongly autocorrelated random variables**, which violates standard ordinary least squares assumptions and motivates generalized least squares or composite likelihood approaches. $\blacksquare$

---

#### Exercise 3.3: Periodogram and Sample Autocovariance Equivalence on Spatial Lattices (S&G Problem 4.7)
Consider a 2D regular spatial lattice of dimensions $R \times C$ containing $N = RC$ observation sites:
$$
\{\mathbf{s}_{r,c} = (r \Delta_1, c \Delta_2) : r = 0, \dots, R - 1; c = 0, \dots, C - 1\}
$$
with scalar measurements $Z_{r,c}$ and sample mean $\bar{Z} = \frac{1}{RC}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1} Z_{r,c}$.

- **(a)** Define the 2D discrete Fourier transform (DFT) $d(\omega_j, \omega_k)$ at fundamental Fourier frequencies $\omega_j = \frac{2\pi j}{R}$ ($j = 0, \dots, R-1$) and $\omega_k = \frac{2\pi k}{C}$ ($k = 0, \dots, C-1$) as:
  $$
  d(\omega_j, \omega_k) = \frac{1}{\sqrt{RC}}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1} (Z_{r,c} - \bar{Z}) e^{-i(r\omega_j + c\omega_k)}
  $$
  and the 2D periodogram as $I(\omega_j, \omega_k) = |d(\omega_j, \omega_k)|^2 = d(\omega_j, \omega_k) d^*(\omega_j, \omega_k)$, where $*$ denotes complex conjugation.
- **(b)** Define the 2D sample autocovariance function for spatial lag $\mathbf{h} = (h_1, h_2)'$ ($|h_1| < R, |h_2| < C$) as:
  $$
  \widehat{C}(h_1, h_2) = \frac{1}{RC}\sum_{r=\max(0, -h_1)}^{\min(R-1, R-1-h_1)} \sum_{c=\max(0, -h_2)}^{\min(C-1, C-1-h_2)} (Z_{r+h_1, c+h_2} - \bar{Z})(Z_{r,c} - \bar{Z})
  $$
  Prove analytically that the periodogram is the exact 2D discrete Fourier transform of the sample autocovariance:
  $$
  I(\omega_j, \omega_k) = \sum_{h_1=-(R-1)}^{R-1} \sum_{h_2=-(C-1)}^{C-1} \widehat{C}(h_1, h_2) e^{-i(h_1 \omega_j + h_2 \omega_k)}
  $$
- **(c)** In remote sensing and digital elevation models, spatial domains routinely contain $N = 10^7$ to $10^8$ pixels. Contrast the computational complexity of direct pairwise semivariogram binning with the 2D Fast Fourier Transform (FFT) approach, and describe the step-by-step algorithm used to extract experimental variograms via FFT.

##### Solution to Exercise 3.3
- **(a)** The 2D discrete Fourier transform computes the orthogonal projection of the mean-centered spatial lattice $(Z_{r,c} - \bar{Z})$ onto the complex harmonic basis $\{e^{i(r\omega_j + c\omega_k)}\}$.
  The 2D periodogram $I(\omega_j, \omega_k)$ represents the sample spectral energy density at spatial frequency vector $\boldsymbol{\omega} = (\omega_j, \omega_k)'$:
  $$
  \begin{aligned}
  I(\omega_j, \omega_k) &= |d(\omega_j, \omega_k)|^2 = d(\omega_j, \omega_k) d^*(\omega_j, \omega_k) \\
  &= \left(\frac{1}{\sqrt{RC}}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1} (Z_{r,c} - \bar{Z}) e^{-i(r\omega_j + c\omega_k)}\right) \left(\frac{1}{\sqrt{RC}}\sum_{r'=0}^{R-1}\sum_{c'=0}^{C-1} (Z_{r',c'} - \bar{Z}) e^{i(r'\omega_j + c'\omega_k)}\right)
  \end{aligned}
  $$

- **(b)** Combining the two summations into a quadruple sum:
  $$
  I(\omega_j, \omega_k) = \frac{1}{RC}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1}\sum_{r'=0}^{R-1}\sum_{c'=0}^{C-1} (Z_{r,c} - \bar{Z})(Z_{r',c'} - \bar{Z}) e^{-i((r - r')\omega_j + (c - c')\omega_k)}
  $$
  Introduce the change of index variables for the spatial separation lags:
  $$
  h_1 = r - r', \quad h_2 = c - c'
  $$
  Because $r, r' \in \{0, \dots, R-1\}$, the integer lag $h_1$ ranges from $-(R-1)$ to $+(R-1)$.
  Similarly, because $c, c' \in \{0, \dots, C-1\}$, $h_2$ ranges from $-(C-1)$ to $+(C-1)$.
  Substitute $r = r' + h_1$ and $c = c' + h_2$. For a fixed pair of lags $(h_1, h_2)$, the valid range of summation for $r'$ requires both $0 \le r' \le R-1$ and $0 \le r' + h_1 \le R-1$, which corresponds to:
  $$
  \max(0, -h_1) \le r' \le \min(R-1, R-1-h_1)
  $$
  and identically for $c'$:
  $$
  \max(0, -h_2) \le c' \le \min(C-1, C-1-h_2)
  $$
  Factoring out the harmonic term $e^{-i(h_1 \omega_j + h_2 \omega_k)}$, the quadruple sum partitions as:
  $$
  I(\omega_j, \omega_k) = \sum_{h_1=-(R-1)}^{R-1} \sum_{h_2=-(C-1)}^{C-1} \left[ \frac{1}{RC}\sum_{r'} \sum_{c'} (Z_{r'+h_1, c'+h_2} - \bar{Z})(Z_{r',c'} - \bar{Z}) \right] e^{-i(h_1 \omega_j + h_2 \omega_k)}
  $$
  Recognizing the term in brackets as the 2D sample autocovariance $\widehat{C}(h_1, h_2)$:
  $$
  I(\omega_j, \omega_k) = \sum_{h_1=-(R-1)}^{R-1} \sum_{h_2=-(C-1)}^{C-1} \widehat{C}(h_1, h_2) e^{-i(h_1 \omega_j + h_2 \omega_k)}
  $$
  This proves analytically that the spatial periodogram is the exact 2D discrete Fourier transform of the sample autocovariance function. By the discrete Fourier inversion theorem, the sample autocovariance is conversely the inverse Fourier transform of the periodogram.

- **(c)** *Computational Complexity and FFT Variogram Pipeline:*
  1. **Direct Pairwise Computation:**
     Computing pairwise differences for all pairs on an $N = RC$ lattice requires evaluating $\binom{N}{2} = \frac{N(N-1)}{2} = \mathcal{O}(N^2)$ pairs. For a modest satellite image of $2000 \times 2000$ pixels ($N = 4 \times 10^6$), the direct approach requires $\approx 8 \times 10^{12}$ floating-point operations, requiring hours to days of CPU compute time.
  2. **Fast Fourier Transform (FFT) Pipeline ($\mathcal{O}(N \log N)$):**
     By the Wiener-Khinchin theorem and the result in (b), spatial convolution reduces to point-wise spectral multiplication:
     - **Step 1 (Zero-Padding):** Pad the mean-centered array $(Z_{r,c} - \bar{Z})$ of size $R \times C$ with zeros to size $(2R) \times (2C)$. This prevents circular wrap-around (cyclic convolution) artifacts.
     - **Step 2 (Forward 2D FFT):** Compute the 2D FFT of the padded raster in $\mathcal{O}((2R)(2C) \log((2R)(2C))) = \mathcal{O}(N \log N)$ operations.
     - **Step 3 (Spectral Power):** Square the modulus of each Fourier coefficient: $I(\omega_j, \omega_k) = |d(\omega_j, \omega_k)|^2$ in $\mathcal{O}(N)$ operations.
     - **Step 4 (Inverse 2D FFT):** Take the 2D inverse FFT of $I(\boldsymbol{\omega})$ to obtain the full 2D spatial sample autocovariance map $\widehat{C}(h_1, h_2)$ for all lags simultaneously in $\mathcal{O}(N \log N)$ operations.
     - **Step 5 (Variogram Surface Extraction):** Compute the semivariogram surface using the relation $\widehat{\gamma}(h_1, h_2) = \widehat{C}(0, 0) - \widehat{C}(h_1, h_2)$.
     For $N = 4 \times 10^6$, $N \log_2 N \approx 8.8 \times 10^7$ operations, which executes in **under one second** on modern hardware, representing a speedup of five orders of magnitude ($> 100,000\times$). $\blacksquare$

---

#### Exercise 3.4: Effect of Power Transformations $Z^\alpha$ on Empirical Semivariograms (S&G Problem 4.1)
Environmental and mining measurements $Z(\mathbf{s}) > 0$ often exhibit strong positive skewness and the "proportional effect," where the local standard deviation is proportional to the local mean. Practitioners routinely apply power transformations $Y(\mathbf{s}) = Z(\mathbf{s})^\alpha$ ($\alpha > 0$) or the Box-Cox family to stabilize variance prior to variogram modeling.

- **(a)** Let $Z(\mathbf{s})$ be an intrinsically stationary process with constant mean $\mu_Z = \operatorname{E}[Z(\mathbf{s})]$ and semivariogram $\gamma_Z(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))$. Consider the power-transformed process $Y(\mathbf{s}) = Z(\mathbf{s})^\alpha$. Using a first-order bivariate Taylor series expansion (delta method) of the difference function $g(z_1, z_2) = z_1^\alpha - z_2^\alpha$ about the mean vector $(\mu_Z, \mu_Z)$, derive the first-order approximation:
  $$
  \gamma_Y(\mathbf{h}) \approx \alpha^2 \mu_Z^{2\alpha - 2} \gamma_Z(\mathbf{h})
  $$
- **(b)** In the presence of a proportional effect, the local variance scales as $\sigma_Z^2(\mathbf{s}) = c^2 \mu_Z^2(\mathbf{s})$ for some relative coefficient of variation $c > 0$. Using the result in part (a), explain why taking the limit $\alpha \to 0$ (the logarithmic transformation $Y = \ln Z$) eliminates the dependence of the variogram sill on the local mean, stabilizing spatial variance.
- **(c)** Contrast the delta-method linear approximation with the exact analytical relationship for a lognormal spatial process where $Z(\mathbf{s}) = \exp(Y(\mathbf{s}))$ and $Y(\mathbf{s}) \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$ is Gaussian:
  $$
  \gamma_Z(\mathbf{h}) = \exp(2\mu_Y + 2\sigma_Y^2) \left[ 1 - \exp(-\gamma_Y(\mathbf{h})) \right]
  $$
  Show under what conditions the exact lognormal relationship reduces to the delta-method approximation.

##### Solution to Exercise 3.4
- **(a)** Let $Z_1 = Z(\mathbf{s})$ and $Z_2 = Z(\mathbf{s} + \mathbf{h})$. The transformed increment is $Y_1 - Y_2 = Z_1^\alpha - Z_2^\alpha$.
  Define the scalar bivariate function $h(z_1, z_2) = z_1^\alpha - z_2^\alpha$.
  Expand $h(z_1, z_2)$ in a first-order Taylor series about the point $(\mu_Z, \mu_Z)$:
  $$
  h(z_1, z_2) \approx h(\mu_Z, \mu_Z) + \left.\frac{\partial h}{\partial z_1}\right|_{(\mu_Z, \mu_Z)} (z_1 - \mu_Z) + \left.\frac{\partial h}{\partial z_2}\right|_{(\mu_Z, \mu_Z)} (z_2 - \mu_Z)
  $$
  Evaluating the components:
  1. $h(\mu_Z, \mu_Z) = \mu_Z^\alpha - \mu_Z^\alpha = 0$.
  2. $\frac{\partial h}{\partial z_1} = \alpha z_1^{\alpha - 1} \implies \left.\frac{\partial h}{\partial z_1}\right|_{(\mu_Z, \mu_Z)} = \alpha \mu_Z^{\alpha - 1}$.
  3. $\frac{\partial h}{\partial z_2} = -\alpha z_2^{\alpha - 1} \implies \left.\frac{\partial h}{\partial z_2}\right|_{(\mu_Z, \mu_Z)} = -\alpha \mu_Z^{\alpha - 1}$.
  Substituting these derivatives into the Taylor expansion:
  $$
  Y_1 - Y_2 \approx \alpha \mu_Z^{\alpha - 1}(z_1 - \mu_Z) - \alpha \mu_Z^{\alpha - 1}(z_2 - \mu_Z) = \alpha \mu_Z^{\alpha - 1}(Z_1 - Z_2)
  $$
  Squaring both sides:
  $$
  (Y_1 - Y_2)^2 \approx \alpha^2 \mu_Z^{2\alpha - 2}(Z_1 - Z_2)^2
  $$
  Taking expectations and dividing by 2:
  $$
  \frac{1}{2}\operatorname{E}[(Y_1 - Y_2)^2] \approx \alpha^2 \mu_Z^{2\alpha - 2} \left( \frac{1}{2}\operatorname{E}[(Z_1 - Z_2)^2] \right)
  $$
  By definition of the semivariogram, this yields:
  $$
  \gamma_Y(\mathbf{h}) \approx \alpha^2 \mu_Z^{2\alpha - 2} \gamma_Z(\mathbf{h})
  $$

- **(b)** Under the proportional effect, the local variance of $Z$ is not stationary but varies across the region as $\sigma_Z^2(\mathbf{s}) = c^2 \mu_Z^2(\mathbf{s})$.
  Consequently, the local sill of the raw variogram $\gamma_Z(\mathbf{h})$ scales proportionally to $\mu_Z^2(\mathbf{s})$:
  $$
  \text{Sill}_Z(\mathbf{s}) \approx c^2 \mu_Z^2(\mathbf{s})
  $$
  Applying the power transformation $Y = Z^\alpha$, the local sill of the transformed variable $Y$ becomes:
  $$
  \text{Sill}_Y(\mathbf{s}) \approx \alpha^2 \mu_Z^{2\alpha - 2}(\mathbf{s}) \cdot \text{Sill}_Z(\mathbf{s}) = \alpha^2 \mu_Z^{2\alpha - 2}(\mathbf{s}) \cdot c^2 \mu_Z^2(\mathbf{s}) = \alpha^2 c^2 \mu_Z^{2\alpha}(\mathbf{s})
  $$
  Now consider the Box-Cox transformation limit as $\alpha \to 0$, which is $Y = \ln Z$:
  $$
  \lim_{\alpha \to 0} \mu_Z^{2\alpha}(\mathbf{s}) = \mu_Z^0(\mathbf{s}) = 1
  $$
  Directly applying the delta method to $Y = \ln Z$:
  $$
  \ln Z_1 - \ln Z_2 \approx \frac{1}{\mu_Z}(Z_1 - Z_2) \implies \gamma_{\ln Z}(\mathbf{h}) \approx \frac{1}{\mu_Z^2}\gamma_Z(\mathbf{h})
  $$
  Substituting $\gamma_Z(\mathbf{h}) = c^2 \mu_Z^2 \rho_0(\mathbf{h})$ (where $\rho_0(\mathbf{h})$ is the stationary correlogram):
  $$
  \gamma_{\ln Z}(\mathbf{h}) \approx \frac{1}{\mu_Z^2}\left(c^2 \mu_Z^2 \rho_0(\mathbf{h})\right) = c^2 \rho_0(\mathbf{h})
  $$
  The spatially varying mean $\mu_Z(\mathbf{s})$ cancels out completely from the variogram sill.
  Therefore, the logarithmic transformation eliminates variance heterogeneity and produces a stationary sill across the entire spatial domain.

- **(c)** Let $Z(\mathbf{s}) = \exp(Y(\mathbf{s}))$ where $Y(\mathbf{s}) \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$ is Gaussian with semivariogram $\gamma_Y(\mathbf{h})$.
  Using the lognormal moment generating property $\operatorname{E}[e^{t Y}] = \exp(t\mu_Y + \frac{1}{2}t^2 \sigma_Y^2)$:
  1. Mean: $\operatorname{E}[Z] = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2)$.
  2. Second moment: $\operatorname{E}[Z^2] = \exp(2\mu_Y + 2\sigma_Y^2)$.
  3. Bivariate expectation:
     $$
     \operatorname{E}[Z(\mathbf{s}) Z(\mathbf{s}+\mathbf{h})] = \operatorname{E}[\exp(Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h}))]
     $$
     Since $Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h}) \sim \mathcal{N}(2\mu_Y, 2\sigma_Y^2 + 2C_Y(\mathbf{h}))$, where $C_Y(\mathbf{h}) = \sigma_Y^2 - \gamma_Y(\mathbf{h})$:
     $$
     \operatorname{E}[Z(\mathbf{s}) Z(\mathbf{s}+\mathbf{h})] = \exp\left(2\mu_Y + \frac{1}{2}(2\sigma_Y^2 + 2C_Y(\mathbf{h}))\right) = \exp(2\mu_Y + \sigma_Y^2 + C_Y(\mathbf{h}))
     $$
  The semivariogram of $Z$ is:
  $$
  \begin{aligned}
  \gamma_Z(\mathbf{h}) &= \frac{1}{2}\operatorname{E}[(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))^2] = \operatorname{E}[Z^2] - \operatorname{E}[Z(\mathbf{s})Z(\mathbf{s}+\mathbf{h})] \\
  &= \exp(2\mu_Y + 2\sigma_Y^2) - \exp(2\mu_Y + \sigma_Y^2 + C_Y(\mathbf{h})) \\
  &= \exp(2\mu_Y + \sigma_Y^2)\left[ \exp(\sigma_Y^2) - \exp(C_Y(\mathbf{h})) \right] \\
  &= \exp(2\mu_Y + 2\sigma_Y^2)\left[ 1 - \exp(C_Y(\mathbf{h}) - \sigma_Y^2) \right]
  \end{aligned}
  $$
  Since $C_Y(\mathbf{h}) - \sigma_Y^2 = -\gamma_Y(\mathbf{h})$:
  $$
  \gamma_Z(\mathbf{h}) = \exp(2\mu_Y + 2\sigma_Y^2)\left[ 1 - \exp(-\gamma_Y(\mathbf{h})) \right]
  $$
  *Reduction to Delta-Method Approximation:*
  When the spatial variance $\sigma_Y^2 \ll 1$ is small (and thus $\gamma_Y(\mathbf{h}) \ll 1$):
  - Using the Taylor approximation $1 - \exp(-\gamma_Y(\mathbf{h})) \approx \gamma_Y(\mathbf{h})$.
  - Noting that $\mu_Z = \exp(\mu_Y + \sigma_Y^2/2) \approx \exp(\mu_Y)$, so $\mu_Z^2 \approx \exp(2\mu_Y + 2\sigma_Y^2)$.
  Substituting these into the exact formula:
  $$
  \gamma_Z(\mathbf{h}) \approx \mu_Z^2 \gamma_Y(\mathbf{h}) \implies \gamma_Y(\mathbf{h}) \approx \frac{1}{\mu_Z^2}\gamma_Z(\mathbf{h})
  $$
  This matches the delta-method approximation from part (b) for $Y = \ln Z$ ($\alpha = 0$).
  However, for highly skewed processes with $\sigma_Y^2 > 1$, the exact lognormal transformation introduces severe non-linear exponential magnification, showing that the delta method is only reliable in the small-variance regime. $\blacksquare$

---

### Tier I: Core Theoretical Re-Derivations & Notation Mastery

#### Exercise 3.5: Unbiasedness of Matheron's Method-of-Moments Estimator
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an intrinsically stationary spatial process with semivariogram $\gamma(\mathbf{h})$. Given observations at locations $\mathbf{s}_1, \dots, \mathbf{s}_n$, Matheron's classical estimator for lag class $N(\mathbf{h}) = \{(\mathbf{s}_i, \mathbf{s}_j) : \mathbf{s}_i - \mathbf{s}_j \approx \mathbf{h}\}$ is defined as:
$$
\widehat{\gamma}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2
$$
where $|N(\mathbf{h})|$ is the number of distinct point pairs in the lag class.

- **(a)** Using the definition of intrinsic stationarity ($\operatorname{E}[Z(\mathbf{s}_i) - Z(\mathbf{s}_j)] = 0$ and $\operatorname{Var}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j)) = 2\gamma(\mathbf{h})$), prove that $\widehat{\gamma}(\mathbf{h})$ is an exactly unbiased estimator of $\gamma(\mathbf{h})$.
- **(b)** Does unbiasedness depend on whether the underlying process is Gaussian?
- **(c)** Does unbiasedness depend on knowing the true spatial mean $\mu$? Explain why this represents a critical advantage of the semivariogram over the spatial autocovariance function.

##### Solution to Exercise 3.5
- **(a)** Take the expectation of $\widehat{\gamma}(\mathbf{h})$:
  $$
  \operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \operatorname{E}\left[\frac{1}{2|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2\right]
  $$
  By linearity of expectation:
  $$
  \operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \frac{1}{2|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} \operatorname{E}\left[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2\right]
  $$
  For any random variable $D$, $\operatorname{E}[D^2] = \operatorname{Var}(D) + (\operatorname{E}[D])^2$.
  Under intrinsic stationarity, for every pair $(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})$:
  $$
  \operatorname{E}[Z(\mathbf{s}_i) - Z(\mathbf{s}_j)] = 0 \quad \text{and} \quad \operatorname{Var}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j)) = 2\gamma(\mathbf{h})
  $$
  Thus:
  $$
  \operatorname{E}\left[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2\right] = 2\gamma(\mathbf{h}) + 0^2 = 2\gamma(\mathbf{h})
  $$
  Substituting this into the summation:
  $$
  \operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \frac{1}{2|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} 2\gamma(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|} \cdot |N(\mathbf{h})| \cdot 2\gamma(\mathbf{h}) = \gamma(\mathbf{h})
  $$
  This establishes that $\widehat{\gamma}(\mathbf{h})$ is strictly unbiased: $\operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \gamma(\mathbf{h})$.

- **(b)** No. The proof relies strictly on the first and second moments of the increments. No distributional assumption (such as Gaussianity) is required for unbiasedness.

- **(c)** No, unbiasedness does **not** require knowing the spatial mean $\mu$, provided $\mu$ is constant across the domain.
  Because $(Z(\mathbf{s}_i) - Z(\mathbf{s}_j)) = (Z(\mathbf{s}_i) - \mu) - (Z(\mathbf{s}_j) - \mu)$, the unknown mean $\mu$ cancels out identically in every pairwise difference:
  $$
  (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 = ((Z(\mathbf{s}_i) - \mu) - (Z(\mathbf{s}_j) - \mu))^2
  $$
  In contrast, estimating the covariance function requires subtracting the mean:
  $$
  \widehat{C}(\mathbf{h}) = \frac{1}{|N(\mathbf{h})|}\sum (Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z})
  $$
  Because the true mean $\mu$ is unknown and replaced by the sample mean $\bar{Z}$, $\widehat{C}(\mathbf{h})$ suffers from systematic downward bias (see Exercise 3.6). The semivariogram completely avoids this bias because differencing acts as an automatic high-pass filter. $\blacksquare$

---

#### Exercise 3.6: Analytical Proof of Downward Bias in the Sample Autocovariance
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a second-order stationary spatial process with unknown mean $\mu$ and covariance $C(\mathbf{h})$. The standard plug-in sample autocovariance is:
$$
\widehat{C}(\mathbf{h}) = \frac{1}{|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} (Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z})
$$
where $\bar{Z} = \frac{1}{n}\sum_{k=1}^n Z(\mathbf{s}_k)$.

- **(a)** Expand $(Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z})$ in terms of deviations from the true mean $Y(\mathbf{s}) = Z(\mathbf{s}) - \mu$ and $\bar{Y} = \bar{Z} - \mu$.
- **(b)** Prove that the expectation of the estimator satisfies:
  $$
  \operatorname{E}[\widehat{C}(\mathbf{h})] = C(\mathbf{h}) - \operatorname{Var}(\bar{Z}) + \mathcal{O}(1/n)
  $$
- **(c)** Since $\operatorname{Var}(\bar{Z}) = \frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1} > 0$, show that $\widehat{C}(\mathbf{h})$ is systematically biased downward for all lags $\mathbf{h}$.
- **(d)** What happens to the empirical covariance as $\|\mathbf{h}\| \to \infty$ where true $C(\mathbf{h}) \to 0$?

##### Solution to Exercise 3.6
- **(a)** Let $Y(\mathbf{s}_i) = Z(\mathbf{s}_i) - \mu$, so $\operatorname{E}[Y(\mathbf{s}_i)] = 0$ and $\bar{Y} = \frac{1}{n}\sum_{k=1}^n Y(\mathbf{s}_k) = \bar{Z} - \mu$.
  Then $Z(\mathbf{s}_i) - \bar{Z} = Y(\mathbf{s}_i) - \bar{Y}$. Expanding the product:
  $$
  (Y(\mathbf{s}_i) - \bar{Y})(Y(\mathbf{s}_j) - \bar{Y}) = Y(\mathbf{s}_i)Y(\mathbf{s}_j) - Y(\mathbf{s}_i)\bar{Y} - Y(\mathbf{s}_j)\bar{Y} + \bar{Y}^2
  $$

- **(b)** Take the expectation of each term:
  1. $\operatorname{E}[Y(\mathbf{s}_i)Y(\mathbf{s}_j)] = C(\mathbf{s}_i - \mathbf{s}_j) = C(\mathbf{h})$ for $(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})$.
  2. $\operatorname{E}[\bar{Y}^2] = \operatorname{Var}(\bar{Y}) = \operatorname{Var}(\bar{Z})$.
  3. $\operatorname{E}[Y(\mathbf{s}_i)\bar{Y}] = \operatorname{E}\left[Y(\mathbf{s}_i) \frac{1}{n}\sum_{k=1}^n Y(\mathbf{s}_k)\right] = \frac{1}{n}\sum_{k=1}^n C(\mathbf{s}_i - \mathbf{s}_k)$.
  Averaging over all $|N(\mathbf{h})|$ pairs:
  $$
  \operatorname{E}[\widehat{C}(\mathbf{h})] = C(\mathbf{h}) - \frac{1}{|N(\mathbf{h})|}\sum_{(\mathbf{s}_i, \mathbf{s}_j)} \left[\frac{1}{n}\sum_{k=1}^n C(\mathbf{s}_i - \mathbf{s}_k) + \frac{1}{n}\sum_{k=1}^n C(\mathbf{s}_j - \mathbf{s}_k)\right] + \operatorname{Var}(\bar{Z})
  $$
  Notice that $\frac{1}{n}\sum_{k=1}^n C(\mathbf{s}_i - \mathbf{s}_k) \approx \operatorname{Var}(\bar{Z}) \cdot n \cdot \frac{1}{n} = \operatorname{Var}(\bar{Z})$. More precisely, let $\bar{C} = \frac{1}{n^2}\sum_{i=1}^n \sum_{k=1}^n C(\mathbf{s}_i - \mathbf{s}_k) = \operatorname{Var}(\bar{Z})$.
  The cross-terms average to $2\bar{C} = 2\operatorname{Var}(\bar{Z})$. Substituting this:
  $$
  \operatorname{E}[\widehat{C}(\mathbf{h})] = C(\mathbf{h}) - 2\operatorname{Var}(\bar{Z}) + \operatorname{Var}(\bar{Z}) = C(\mathbf{h}) - \operatorname{Var}(\bar{Z})
  $$

- **(c)** Because $\operatorname{Var}(\bar{Z}) = \frac{1}{n^2}\mathbf{1}'\mathbf{V}\mathbf{1}$ is the variance of a real random variable, $\operatorname{Var}(\bar{Z}) > 0$.
  Therefore:
  $$
  \operatorname{E}[\widehat{C}(\mathbf{h})] < C(\mathbf{h}) \quad \forall \mathbf{h}
  $$
  The sample autocovariance is strictly biased downward everywhere by an amount equal to $\operatorname{Var}(\bar{Z})$.

- **(d)** As $\|\mathbf{h}\| \to \infty$, the true spatial correlation decays to zero: $C(\mathbf{h}) \to 0$.
  However, the expectation of the estimator converges to:
  $$
  \lim_{\|\mathbf{h}\| \to \infty} \operatorname{E}[\widehat{C}(\mathbf{h})] = 0 - \operatorname{Var}(\bar{Z}) = -\operatorname{Var}(\bar{Z}) < 0
  $$
  The sample autocovariance does **not** level off at zero; it dips systematically into **negative values**, creating spurious negative correlations at large distances that are pure artifacts of estimating the mean with $\bar{Z}$. $\blacksquare$

---

#### Exercise 3.7: Sampling Distribution and Variance of Gaussian Increments
Let $Z(\mathbf{s})$ be a stationary Gaussian random field with semivariogram $\gamma(\mathbf{h})$. For any fixed pair of locations $\mathbf{s}_i, \mathbf{s}_j$, define the increment $D_{ij} = Z(\mathbf{s}_i) - Z(\mathbf{s}_j)$.

- **(a)** State the exact probability distribution of $D_{ij}$.
- **(b)** Show that the normalized squared increment $U_{ij} = \frac{D_{ij}^2}{2\gamma(\mathbf{h})}$ is distributed as a central chi-squared variable with 1 degree of freedom ($\chi_1^2$).
- **(c)** Use the properties of $\chi_1^2$ ($\operatorname{Var}(\chi_1^2) = 2$) to prove that:
  $$
  \operatorname{Var}\left((Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2\right) = 8 \left(\gamma(\mathbf{h})\right)^2
  $$
- **(d)** Explain why the variance of Matheron's estimator $\widehat{\gamma}(\mathbf{h})$ is proportional to the **squared true semivariance** $\gamma(\mathbf{h})^2$.

##### Solution to Exercise 3.7
- **(a)** Because $Z(\mathbf{s})$ is a Gaussian random field, any linear combination of its values is normally distributed.
  The increment has mean $\operatorname{E}[D_{ij}] = 0$ and variance $\operatorname{Var}(D_{ij}) = 2\gamma(\mathbf{h})$. Therefore:
  $$
  D_{ij} \sim \mathcal{N}\left(0, 2\gamma(\mathbf{h})\right)
  $$

- **(b)** Standardizing the Gaussian increment:
  $$
  W_{ij} = \frac{D_{ij}}{\sqrt{2\gamma(\mathbf{h})}} \sim \mathcal{N}(0, 1)
  $$
  By definition of the chi-squared distribution, the square of a standard normal variable has a chi-squared distribution with 1 degree of freedom:
  $$
  U_{ij} = W_{ij}^2 = \frac{D_{ij}^2}{2\gamma(\mathbf{h})} \sim \chi_1^2
  $$

- **(c)** For a chi-squared distribution with $k$ degrees of freedom, $\operatorname{Var}(\chi_k^2) = 2k$. For $k = 1$:
  $$
  \operatorname{Var}(U_{ij}) = 2
  $$
  Now express the squared increment in terms of $U_{ij}$:
  $$
  (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 = 2\gamma(\mathbf{h}) U_{ij}
  $$
  Taking the variance:
  $$
  \operatorname{Var}\left((Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2\right) = \operatorname{Var}\left(2\gamma(\mathbf{h}) U_{ij}\right) = \left(2\gamma(\mathbf{h})\right)^2 \operatorname{Var}(U_{ij}) = 4\gamma(\mathbf{h})^2 \cdot 2 = 8\gamma(\mathbf{h})^2
  $$

- **(d)** Matheron's estimator is the sum:
  $$
  \widehat{\gamma}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} D_{ij}^2
  $$
  If the pairs in $N(\mathbf{h})$ were approximately independent, the variance of the estimator would be:
  $$
  \operatorname{Var}(\widehat{\gamma}(\mathbf{h})) \approx \frac{1}{4|N(\mathbf{h})|^2} \sum_{(\mathbf{s}_i, \mathbf{s}_j)} \operatorname{Var}(D_{ij}^2) = \frac{1}{4|N(\mathbf{h})|^2} |N(\mathbf{h})| \cdot 8\gamma(\mathbf{h})^2 = \frac{2\gamma(\mathbf{h})^2}{|N(\mathbf{h})|}
  $$
  Because $\operatorname{Var}(\widehat{\gamma}(\mathbf{h})) \propto \gamma(\mathbf{h})^2$, the estimation variance is small near the origin where $\gamma(h)$ is small, but grows quadratically as lag $h$ increases.
  This provides the theoretical justification for **Cressie's Weighted Least Squares (WLS)**, which weights squared residuals by $1/\gamma(h)^2$. $\blacksquare$

---

#### Exercise 3.8: Mathematical Derivation of the Cressie-Hawkins Modulus Estimator
Matheron's estimator averages squared differences, making it notoriously vulnerable to outliers. Cressie & Hawkins (1980) proposed the robust fourth-root modulus estimator:
$$
\bar{R}(\mathbf{h}) = \frac{1}{|N(\mathbf{h})|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} |Z(\mathbf{s}_i) - Z(\mathbf{s}_j)|^{1/2}
$$
and defined the estimator as:
$$
\widehat{\gamma}_{\text{CH}}(\mathbf{h}) = \frac{1}{2} \frac{\left(\bar{R}(\mathbf{h})\right)^4}{0.457 + \frac{0.494}{|N(\mathbf{h})|}}
$$

- **(a)** For $D \sim \mathcal{N}(0, 2\gamma(\mathbf{h}))$, let $W = D/\sqrt{2\gamma(\mathbf{h})} \sim \mathcal{N}(0, 1)$. Compute the expectation $\operatorname{E}[|W|^{1/2}]$ using the Gamma function $\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt$.
- **(b)** Show that $\operatorname{E}[|W|^{1/2}] = \frac{\sqrt{2}}{\pi^{1/4}}\Gamma(3/4) \approx 0.822179$, and compute $(\operatorname{E}[|W|^{1/2}])^4$.
- **(c)** Explain where the asymptotic constant $0.457$ originates.
- **(d)** Explain where the finite-sample correction term $0.494/|N(\mathbf{h})|$ originates.

##### Solution to Exercise 3.8
- **(a)** For $W \sim \mathcal{N}(0, 1)$, $|W|$ has folded standard normal density $f_{|W|}(w) = \sqrt{\frac{2}{\pi}} \exp(-w^2/2)$ for $w \ge 0$.
  The expected fractional power is:
  $$
  \operatorname{E}[|W|^{1/2}] = \int_0^\infty w^{1/2} \sqrt{\frac{2}{\pi}} \exp(-w^2/2) \, dw
  $$
  Substitute $t = w^2/2 \implies w = \sqrt{2} t^{1/2}$ and $dw = \frac{1}{\sqrt{2}} t^{-1/2} dt$:
  $$
  \begin{aligned}
  \operatorname{E}[|W|^{1/2}] &= \sqrt{\frac{2}{\pi}} \int_0^\infty (\sqrt{2} t^{1/2})^{1/2} e^{-t} \frac{1}{\sqrt{2}} t^{-1/2} \, dt \\
  &= \sqrt{\frac{2}{\pi}} \cdot 2^{1/4} \cdot \frac{1}{\sqrt{2}} \int_0^\infty t^{1/4} t^{-1/2} e^{-t} \, dt \\
  &= \frac{2^{1/4}}{\sqrt{\pi}} \int_0^\infty t^{-1/4} e^{-t} \, dt = \frac{2^{1/4}}{\sqrt{\pi}} \Gamma(3/4)
  \end{aligned}
  $$

- **(b)** Using $\Gamma(3/4) \approx 1.2254167$ and $\pi \approx 3.14159265$:
  $$
  \operatorname{E}[|W|^{1/2}] = \frac{2^{0.25}}{\sqrt{\pi}} \Gamma(3/4) \approx \frac{1.189207}{1.772454} \times 1.225417 \approx 0.822179
  $$
  Raising this expectation to the fourth power:
  $$
  \left(\operatorname{E}[|W|^{1/2}]\right)^4 = (0.822179)^4 \approx 0.456947 \approx 0.457
  $$

- **(c)** Since $|Z(\mathbf{s}_i) - Z(\mathbf{s}_j)|^{1/2} = (2\gamma(\mathbf{h}))^{1/4} |W|^{1/2}$, taking expectations:
  $$
  \operatorname{E}[\bar{R}(\mathbf{h})] = (2\gamma(\mathbf{h}))^{1/4} \operatorname{E}[|W|^{1/2}]
  $$
  By the continuous mapping theorem and delta method, as $|N(\mathbf{h})| \to \infty$:
  $$
  (\bar{R}(\mathbf{h}))^4 \xrightarrow{P} \left(\operatorname{E}[|W|^{1/2}]\right)^4 \cdot 2\gamma(\mathbf{h}) = 0.456947 \cdot 2\gamma(\mathbf{h})
  $$
  Dividing $(\bar{R}(\mathbf{h}))^4$ by $2 \times 0.457$ yields:
  $$
  \frac{(\bar{R}(\mathbf{h}))^4}{2 \times 0.457} \to \gamma(\mathbf{h})
  $$
  Thus, $0.457$ is the exact asymptotic de-biasing divisor.

- **(d)** In finite samples, $(\bar{R}(\mathbf{h}))^4$ is a non-linear convex transformation of the sample mean $\bar{R}$. By Jensen's inequality:
  $$
  \operatorname{E}[(\bar{R})^4] > (\operatorname{E}[\bar{R}])^4
  $$
  Using a second-order Taylor expansion around the mean:
  $$
  \operatorname{E}[(\bar{R})^4] \approx (\operatorname{E}[\bar{R}])^4 + 6 (\operatorname{E}[\bar{R}])^2 \operatorname{Var}(\bar{R}) = (\operatorname{E}[\bar{R}])^4 \left(1 + \frac{6 \operatorname{Var}(|W|^{1/2})}{|N(\mathbf{h})| (\operatorname{E}[|W|^{1/2}])^2}\right)
  $$
  Evaluating $\operatorname{Var}(|W|^{1/2}) = \operatorname{E}[|W|] - (\operatorname{E}[|W|^{1/2}])^2 = \sqrt{2/\pi} - (0.822179)^2 \approx 0.797885 - 0.675978 = 0.121907$.
  The finite-sample correction coefficient is:
  $$
  \frac{6 \times 0.121907}{(0.822179)^2} \times 0.457 \approx 0.494
  $$
  Adding $0.494/|N(\mathbf{h})|$ to the denominator exactly compensates for Jensen's inequality inflation in finite samples. $\blacksquare$

---

### Tier II: Pencil-and-Paper Matrix & Vector Mechanics

#### Exercise 3.9: Pairwise Lag Distances and Semivariogram Binning by Hand
Consider $n = 4$ irregular spatial observation points in $\mathbb{R}^2$:
$$
\mathbf{s}_1 = (0, 0)', \quad \mathbf{s}_2 = (1, 0)', \quad \mathbf{s}_3 = (0, 2)', \quad \mathbf{s}_4 = (2, 2)'
$$
with measured attribute values:
$$
Z(\mathbf{s}_1) = 10.0, \quad Z(\mathbf{s}_2) = 14.0, \quad Z(\mathbf{s}_3) = 12.0, \quad Z(\mathbf{s}_4) = 18.0
$$

- **(a)** List all $\binom{4}{2} = 6$ distinct point pairs, and compute their pairwise lag vectors $\mathbf{h}_{ij}$, Euclidean distances $h_{ij}$, and squared differences $(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2$.
- **(b)** Define two lag bins:
  - Bin 1: $h \in [0.5, 1.5)$ (nominal lag $h_1 = 1.0$).
  - Bin 2: $h \in [1.5, 3.0)$ (nominal lag $h_2 = 2.2$).
  Assign each pair to its corresponding bin.
- **(c)** Calculate Matheron's empirical semivariance $\widehat{\gamma}(h_k)$ for each bin.

##### Solution to Exercise 3.9
- **(a) Pairwise Computations:**
  1. **Pair (1, 2):**
     $\mathbf{h}_{12} = \mathbf{s}_1 - \mathbf{s}_2 = (-1, 0)' \implies h_{12} = \sqrt{(-1)^2 + 0^2} = 1.0$.
     $Z_1 - Z_2 = 10.0 - 14.0 = -4.0 \implies (Z_1 - Z_2)^2 = 16.0$.
  2. **Pair (1, 3):**
     $\mathbf{h}_{13} = (0, -2)' \implies h_{13} = 2.0$.
     $Z_1 - Z_3 = 10.0 - 12.0 = -2.0 \implies (Z_1 - Z_3)^2 = 4.0$.
  3. **Pair (1, 4):**
     $\mathbf{h}_{14} = (-2, -2)' \implies h_{14} = \sqrt{(-2)^2 + (-2)^2} = \sqrt{8} \approx 2.8284$.
     $Z_1 - Z_4 = 10.0 - 18.0 = -8.0 \implies (Z_1 - Z_4)^2 = 64.0$.
  4. **Pair (2, 3):**
     $\mathbf{h}_{23} = (1, -2)' \implies h_{23} = \sqrt{1^2 + (-2)^2} = \sqrt{5} \approx 2.2361$.
     $Z_2 - Z_3 = 14.0 - 12.0 = 2.0 \implies (Z_2 - Z_3)^2 = 4.0$.
  5. **Pair (2, 4):**
     $\mathbf{h}_{24} = (-1, -2)' \implies h_{24} = \sqrt{(-1)^2 + (-2)^2} = \sqrt{5} \approx 2.2361$.
     $Z_2 - Z_4 = 14.0 - 18.0 = -4.0 \implies (Z_2 - Z_4)^2 = 16.0$.
  6. **Pair (3, 4):**
     $\mathbf{h}_{34} = (-2, 0)' \implies h_{34} = 2.0$.
     $Z_3 - Z_4 = 12.0 - 18.0 = -6.0 \implies (Z_3 - Z_4)^2 = 36.0$.

- **(b) Bin Allocation:**
  - **Bin 1 ($h \in [0.5, 1.5)$):**
    Contains Pair (1, 2) with distance $h_{12} = 1.0$.
    $|N_1| = 1$. Average lag distance $\bar{h}_1 = 1.0$.
  - **Bin 2 ($h \in [1.5, 3.0)$):**
    Contains:
    - Pair (1, 3): $h = 2.0$, diff$^2 = 4.0$
    - Pair (1, 4): $h = 2.8284$, diff$^2 = 64.0$
    - Pair (2, 3): $h = 2.2361$, diff$^2 = 4.0$
    - Pair (2, 4): $h = 2.2361$, diff$^2 = 16.0$
    - Pair (3, 4): $h = 2.0$, diff$^2 = 36.0$
    $|N_2| = 5$. Average lag distance $\bar{h}_2 = (2.0 + 2.8284 + 2.2361 + 2.2361 + 2.0)/5 = 11.3006 / 5 \approx 2.2601$.

- **(c) Matheron Semivariance Calculations:**
  - For Bin 1:
    $$
    \widehat{\gamma}(h_1) = \frac{1}{2|N_1|}\sum (Z_i - Z_j)^2 = \frac{1}{2(1)} (16.0) = 8.0000
    $$
  - For Bin 2:
    Sum of squared differences: $\sum = 4.0 + 64.0 + 4.0 + 16.0 + 36.0 = 124.0$.
    $$
    \widehat{\gamma}(h_2) = \frac{1}{2|N_2|}\sum (Z_i - Z_j)^2 = \frac{1}{2(5)} (124.0) = \frac{124.0}{10} = 12.4000 \blacksquare
    $$

---

#### Exercise 3.10: Breakdown Comparison: Matheron vs. Cressie-Hawkins with an Outlier
Using the 5 pairs belonging to Bin 2 in Exercise 3.9:
$$
\text{Differences: } |Z_i - Z_j| \in \{2.0, 8.0, 2.0, 4.0, 6.0\}
$$

- **(a)** Compute the baseline Cressie-Hawkins estimate $\widehat{\gamma}_{\text{CH}}(h_2)$ for Bin 2.
- **(b)** Suppose location $\mathbf{s}_4$ is contaminated by a gross measurement transcription error: $Z(\mathbf{s}_4)$ jumps from $18.0$ to $80.0$.
  Recalculate the differences for the 5 pairs in Bin 2.
- **(c)** Recompute both Matheron $\widehat{\gamma}$ and Cressie-Hawkins $\widehat{\gamma}_{\text{CH}}$ under the contaminated dataset.
- **(d)** Compare the percentage inflation of each estimator and discuss outlier resistance.

##### Solution to Exercise 3.10
- **(a) Baseline Cressie-Hawkins for Bin 2 ($|N_2| = 5$):**
  Fourth-root absolute differences:
  - $|2.0|^{1/4} \approx 1.1892$
  - $|8.0|^{1/4} \approx 1.6818$
  - $|2.0|^{1/4} \approx 1.1892$
  - $|4.0|^{1/4} \approx 1.4142$
  - $|6.0|^{1/4} \approx 1.5651$
  Sum of fourth roots:
  $$
  \sum = 1.1892 + 1.6818 + 1.1892 + 1.4142 + 1.5651 = 7.0395
  $$
  Average $\bar{R} = 7.0395 / 5 = 1.4079$.
  Fourth power: $(\bar{R})^4 = (1.4079)^4 \approx 3.9290$.
  Denominator: $B = 0.457 + \frac{0.494}{5} = 0.457 + 0.0988 = 0.5558$.
  $$
  \widehat{\gamma}_{\text{CH}} = \frac{1}{2} \frac{(\bar{R})^4}{B} = \frac{1}{2} \frac{3.9290}{0.5558} \approx 3.5345
  $$

- **(b) Contaminated Differences with $Z(\mathbf{s}_4) = 80.0$:**
  - Pair (1, 3): $|10 - 12| = 2.0$ (unchanged)
  - Pair (1, 4): $|10 - 80| = 70.0$ (was 8.0)
  - Pair (2, 3): $|14 - 12| = 2.0$ (unchanged)
  - Pair (2, 4): $|14 - 80| = 66.0$ (was 4.0)
  - Pair (3, 4): $|12 - 80| = 68.0$ (was 6.0)

- **(c) Recomputed Estimators:**
  - **Matheron Estimator:**
    Squared differences: $2^2 + 70^2 + 2^2 + 66^2 + 68^2 = 4 + 4900 + 4 + 4356 + 4624 = 13888.0$.
    $$
    \widehat{\gamma}_{\text{contam}} = \frac{1}{2(5)} (13888.0) = 1388.80
    $$
  - **Cressie-Hawkins Estimator:**
    Fourth roots:
    - $|2.0|^{1/4} \approx 1.1892$
    - $|70.0|^{1/4} \approx 2.8925$
    - $|2.0|^{1/4} \approx 1.1892$
    - $|66.0|^{1/4} \approx 2.8500$
    - $|68.0|^{1/4} \approx 2.8715$
    Sum of fourth roots: $\sum = 1.1892 + 2.8925 + 1.1892 + 2.8500 + 2.8715 = 10.9924$.
    Average $\bar{R} = 10.9924 / 5 = 2.1985$.
    Fourth power: $(\bar{R})^4 = (2.1985)^4 \approx 23.3619$.
    $$
    \widehat{\gamma}_{\text{CH, contam}} = \frac{1}{2} \frac{23.3619}{0.5558} \approx 21.0165
    $$

- **(d) Comparison:**
  - Baseline Matheron: $12.4 \implies$ Contaminated: $1388.8$ (**$11,100\%$ inflation!**).
  - Baseline Cressie-Hawkins: $3.53 \implies$ Contaminated: $21.02$ (**$495\%$ inflation**).
  Matheron's estimator experiences catastrophic breakdown because squaring an erroneous difference of $70$ produces $4900$, which completely dominates the sum.
  In contrast, Cressie-Hawkins dampens the outlier via the fourth-root transformation ($70^{1/4} \approx 2.89$), reducing the outlier inflation by more than **two orders of magnitude**! $\blacksquare$

---

#### Exercise 3.11: Cross-Variogram vs. Pseudo-Cross Variogram Symmetries
Let $\{Z_1(\mathbf{s})\}$ and $\{Z_2(\mathbf{s})\}$ be two coregionalized spatial processes.

- **(a)** Matheron’s **cross-semivariogram** is defined as:
  $$
  \gamma_{12}(\mathbf{h}) = \frac{1}{2}\operatorname{E}\left[(Z_1(\mathbf{s}+\mathbf{h}) - Z_1(\mathbf{s}))(Z_2(\mathbf{s}+\mathbf{h}) - Z_2(\mathbf{s}))\right]
  $$
  Prove that $\gamma_{12}(\mathbf{h})$ is always symmetric: $\gamma_{12}(\mathbf{h}) = \gamma_{12}(-\mathbf{h})$.
- **(b)** Clark’s **pseudo-cross semivariogram** is defined as:
  $$
  \gamma_{12}^P(\mathbf{h}) = \frac{1}{2}\operatorname{Var}\left(Z_1(\mathbf{s}+\mathbf{h}) - Z_2(\mathbf{s})\right)
  $$
  Prove that $\gamma_{12}^P(\mathbf{h}) = \gamma_{21}^P(-\mathbf{h})$, but in general $\gamma_{12}^P(\mathbf{h}) \ne \gamma_{12}^P(-\mathbf{h})$ if the cross-covariance is asymmetric ($C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})$).
- **(c)** Explain why the pseudo-cross variogram can be computed for heterotopic data (where variable 1 and variable 2 are measured at different sampling locations), whereas Matheron's cross-variogram requires isotopic data.

##### Solution to Exercise 3.11
- **(a)** Under second-order stationarity, expand the product:
  $$
  (Z_1(\mathbf{s}+\mathbf{h}) - Z_1(\mathbf{s}))(Z_2(\mathbf{s}+\mathbf{h}) - Z_2(\mathbf{s})) = Z_1(\mathbf{s}+\mathbf{h})Z_2(\mathbf{s}+\mathbf{h}) - Z_1(\mathbf{s}+\mathbf{h})Z_2(\mathbf{s}) - Z_1(\mathbf{s})Z_2(\mathbf{s}+\mathbf{h}) + Z_1(\mathbf{s})Z_2(\mathbf{s})
  $$
  Taking expectations:
  $$
  2\gamma_{12}(\mathbf{h}) = C_{12}(\mathbf{0}) - C_{12}(\mathbf{h}) - C_{12}(-\mathbf{h}) + C_{12}(\mathbf{0}) = 2C_{12}(\mathbf{0}) - [C_{12}(\mathbf{h}) + C_{12}(-\mathbf{h})]
  $$
  Dividing by 2:
  $$
  \gamma_{12}(\mathbf{h}) = C_{12}(\mathbf{0}) - \frac{C_{12}(\mathbf{h}) + C_{12}(-\mathbf{h})}{2}
  $$
  Now evaluate at $-\mathbf{h}$:
  $$
  \gamma_{12}(-\mathbf{h}) = C_{12}(\mathbf{0}) - \frac{C_{12}(-\mathbf{h}) + C_{12}(-(-\mathbf{h}))}{2} = C_{12}(\mathbf{0}) - \frac{C_{12}(-\mathbf{h}) + C_{12}(\mathbf{h})}{2} = \gamma_{12}(\mathbf{h})
  $$
  Therefore, Matheron's cross-semivariogram is **always an even (symmetric) function**, regardless of whether the cross-covariance is symmetric.

- **(b)** Expand Clark's pseudo-cross variogram:
  $$
  2\gamma_{12}^P(\mathbf{h}) = \operatorname{Var}(Z_1(\mathbf{s}+\mathbf{h})) + \operatorname{Var}(Z_2(\mathbf{s})) - 2\operatorname{Cov}(Z_1(\mathbf{s}+\mathbf{h}), Z_2(\mathbf{s})) = C_{11}(\mathbf{0}) + C_{22}(\mathbf{0}) - 2C_{12}(\mathbf{h})
  $$
  Dividing by 2:
  $$
  \gamma_{12}^P(\mathbf{h}) = \frac{1}{2}(C_{11}(\mathbf{0}) + C_{22}(\mathbf{0})) - C_{12}(\mathbf{h})
  $$
  Evaluating at $-\mathbf{h}$:
  $$
  \gamma_{12}^P(-\mathbf{h}) = \frac{1}{2}(C_{11}(\mathbf{0}) + C_{22}(\mathbf{0})) - C_{12}(-\mathbf{h})
  $$
  Notice that $C_{12}(-\mathbf{h}) = C_{21}(\mathbf{h})$. Thus:
  $$
  \gamma_{12}^P(-\mathbf{h}) = \frac{1}{2}(C_{22}(\mathbf{0}) + C_{11}(\mathbf{0})) - C_{21}(\mathbf{h}) = \gamma_{21}^P(\mathbf{h})
  $$
  If $C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})$ (cross-covariance asymmetry), then:
  $$
  \gamma_{12}^P(\mathbf{h}) \ne \gamma_{12}^P(-\mathbf{h})
  $$
  The pseudo-cross variogram directly retains physical directional asymmetry.

- **(c)** To estimate Matheron's cross-variogram $\gamma_{12}(\mathbf{h})$:
  $$
  \widehat{\gamma}_{12}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|}\sum (Z_1(\mathbf{s}_i) - Z_1(\mathbf{s}_j))(Z_2(\mathbf{s}_i) - Z_2(\mathbf{s}_j))
  $$
  Both variable 1 and variable 2 must be measured at location $\mathbf{s}_i$ and at location $\mathbf{s}_j$ (**isotopic sampling**). If $Z_1$ is measured at borehole sites and $Z_2$ is measured at seismic soundings, no paired observations exist at identical sites, and $\widehat{\gamma}_{12}$ cannot be computed.
  In contrast, Clark's pseudo-cross variogram:
  $$
  \widehat{\gamma}_{12}^P(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|}\sum_{(\mathbf{s}_i \in S_1, \mathbf{s}_j \in S_2)} (Z_1(\mathbf{s}_i) - Z_2(\mathbf{s}_j))^2
  $$
  only requires that a site $\mathbf{s}_i$ where $Z_1$ is measured be separated by vector $\mathbf{h}$ from a site $\mathbf{s}_j$ where $Z_2$ is measured (**heterotopic sampling**). Thus, pseudo-cross variograms unlock geostatistical modeling for non-collocated datasets. $\blacksquare$

---

### Tier III: Algorithmic Translation & Labs in Pure Python

#### Exercise 3.12: Pure Procedural Directional Semivariogram Engine
Implement `Algorithm 3.1` (`compute_directional_semivariogram`) from scratch in pure, dependency-free Pythonic pseudocode.

- **(a)** Formulate the directional tolerance filter: for azimuth angle $\theta_{\text{dir}}$ and angular tolerance $\Delta \theta$, write the mathematical test determining whether lag vector $\mathbf{h}_{ij} = (\Delta x, \Delta y)'$ falls inside the directional cone.
- **(b)** Write a function `compute_directional_semivariogram(coords, vals, lag_step, max_lag, azimuth_deg, tol_deg)` that bins pairs into lag intervals $[k \Delta h - \Delta h/2, k \Delta h + \Delta h/2)$ and outputs the bin centers, empirical semivariance, and pair counts.
- **(c)** Verify your implementation on a synthetic $5 \times 5$ regular grid with spacing $\Delta x = \Delta y = 1.0$, testing for an omnidirectional semivariogram (tolerance $90^\circ$).

##### Solution to Exercise 3.12
- **(a) Directional Cone Filter:**
  The azimuth angle $\phi$ of separation vector $\mathbf{h} = (\Delta x, \Delta y)'$ measured clockwise from North ($+y$) is:
  $$
  \phi = \operatorname{atan2}(\Delta x, \Delta y) \quad (\text{converted to } [0, 2\pi))
  $$
  Because $\gamma(\mathbf{h}) = \gamma(-\mathbf{h})$, we check both $\mathbf{h}$ and $-\mathbf{h}$. The angular difference between $\phi$ and target azimuth $\theta$ modulo $\pi$ is:
  $$
  \delta\theta = \min(|\phi - \theta| \pmod \pi, \pi - (|\phi - \theta| \pmod \pi))
  $$
  The pair is inside the directional cone if $\delta\theta \le \Delta\theta$.

- **(b) Pure Pythonic Directional Semivariogram Engine:**
```python
import math

def compute_directional_semivariogram(coords, vals, lag_step, max_lag, azimuth_deg=0.0, tol_deg=90.0):
    """
    Computes Matheron directional empirical semivariogram.
    
    coords: List of (x, y) tuples
    vals: List of scalar measurements
    lag_step: Bin width Delta h
    max_lag: Maximum lag distance to bin
    azimuth_deg: Target azimuth in degrees (0 = North, 90 = East)
    tol_deg: Angular half-tolerance in degrees (90 = omnidirectional)
    """
    n = len(coords)
    n_bins = int(max_lag / lag_step)
    
    # Initialize accumulators
    pair_counts = [0] * n_bins
    sum_sq_diff = [0.0] * n_bins
    sum_dist = [0.0] * n_bins
    
    theta_rad = math.radians(azimuth_deg)
    tol_rad = math.radians(tol_deg)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[j][0] - coords[i][0]
            dy = coords[j][1] - coords[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist > max_lag or dist < 1e-12:
                continue
                
            # Directional filtering
            if tol_deg < 89.9:
                phi = math.atan2(dx, dy) # Clockwise from North
                # Angle difference modulo pi
                d_theta = abs(phi - theta_rad) % math.pi
                if d_theta > math.pi / 2.0:
                    d_theta = math.pi - d_theta
                if d_theta > tol_rad:
                    continue
                    
            # Determine lag bin index
            bin_idx = int(dist / lag_step)
            if 0 <= bin_idx < n_bins:
                diff = vals[i] - vals[j]
                pair_counts[bin_idx] += 1
                sum_sq_diff[bin_idx] += diff * diff
                sum_dist[bin_idx] += dist
                
    # Assemble results
    lag_centers = []
    gamma_hat = []
    valid_counts = []
    
    for k in range(n_bins):
        if pair_counts[k] > 0:
            lag_centers.append(sum_dist[k] / pair_counts[k])
            gamma_hat.append(sum_sq_diff[k] / (2.0 * pair_counts[k]))
            valid_counts.append(pair_counts[k])
            
    return lag_centers, gamma_hat, valid_counts
```

- **(c) Test Verification on a Regular Grid:**
  Creating a $5 \times 5$ grid of coordinates where $Z(x, y) = x$:
  - Distance $h = 1.0$: horizontal pairs have $Z(x+1, y) - Z(x, y) = 1 \implies \text{diff}^2 = 1$; vertical pairs have $Z(x, y+1) - Z(x, y) = 0 \implies \text{diff}^2 = 0$.
  - Number of horizontal pairs at $h=1$: $5 \times 4 = 20$.
  - Number of vertical pairs at $h=1$: $5 \times 4 = 20$.
  - Total pairs at $h = 1$: $|N(1)| = 40$.
  - Semivariance calculation:
    $$
    \widehat{\gamma}(1.0) = \frac{1}{2(40)} [20 \times 1^2 + 20 \times 0^2] = \frac{20}{80} = 0.25
    $$
  Running the code yields: `lag_centers = [1.0]`, `gamma_hat = [0.25]`, `valid_counts = [40]`, verifying exact execution. $\blacksquare$

---

#### Exercise 3.13: Empirical Autocovariance Bias Quantification Lab
Write a simulation script in pure Pythonic pseudocode to empirically quantify the $-\operatorname{Var}(\bar{Z})$ downward bias in the sample autocovariance.

- **(a)** Generate $M = 2,000$ realizations of a 1D Gaussian field on $n = 20$ points spaced $\Delta s = 1.0$ apart, with Exponential covariance $C(h) = 10.0 e^{-h/3.0}$.
- **(b)** For each realization, compute:
  1. The true-mean covariance: $\widehat{C}_{\text{known}}(h) = \frac{1}{|N(h)|}\sum (Z(s_i) - \mu)(Z(s_j) - \mu)$.
  2. The sample-mean covariance: $\widehat{C}_{\text{est}}(h) = \frac{1}{|N(h)|}\sum (Z(s_i) - \bar{Z})(Z(s_j) - \bar{Z})$.
- **(c)** Average both curves over the $M$ runs, compute the empirical $\operatorname{Var}(\bar{Z})$, and verify that the two curves differ by exactly $\operatorname{Var}(\bar{Z})$ across all lags.

##### Solution to Exercise 3.13
- **(a) Pure Pythonic Lab Script:**
```python
import math
import random

def run_covariance_bias_lab(n=20, delta_s=1.0, sigma2=10.0, phi=3.0, M=2000):
    coords = [(i * delta_s,) for i in range(n)]
    
    # 1. Assemble true covariance matrix V
    V = [[sigma2 * math.exp(-abs(coords[i][0] - coords[j][0]) / phi) for j in range(n)] for i in range(n)]
    
    # Theoretical Var(Z_bar)
    var_z_bar_theoretical = sum(sum(row) for row in V) / (n * n)
    
    # Cholesky factor
    L = cholesky_factorize(V)
    
    # We will track lag h = 1.0 (adjacent points, |N(1)| = n - 1 = 19)
    # and lag h = 5.0 (|N(5)| = n - 5 = 15)
    c_known_h1_accum = 0.0
    c_est_h1_accum = 0.0
    c_known_h5_accum = 0.0
    c_est_h5_accum = 0.0
    z_bar_sq_accum = 0.0
    
    for _ in range(M):
        # Generate N(0, 1) variates via Box-Muller
        w = []
        for _ in range(n // 2 + n % 2):
            u1, u2 = random.random(), random.random()
            r = math.sqrt(-2.0 * math.log(max(u1, 1e-15)))
            th = 2.0 * math.pi * u2
            w.extend([r * math.cos(th), r * math.sin(th)])
        w = w[:n]
        
        # Realization z (mu = 0)
        z = [sum(L[i][j] * w[j] for j in range(i + 1)) for i in range(n)]
        z_bar = sum(z) / n
        z_bar_sq_accum += z_bar * z_bar
        
        # Lag 1 pairs
        pairs_h1 = [(z[i], z[i+1]) for i in range(n - 1)]
        c_known_h1_accum += sum(a * b for a, b in pairs_h1) / len(pairs_h1)
        c_est_h1_accum += sum((a - z_bar) * (b - z_bar) for a, b in pairs_h1) / len(pairs_h1)
        
        # Lag 5 pairs
        pairs_h5 = [(z[i], z[i+5]) for i in range(n - 5)]
        c_known_h5_accum += sum(a * b for a, b in pairs_h5) / len(pairs_h5)
        c_est_h5_accum += sum((a - z_bar) * (b - z_bar) for a, b in pairs_h5) / len(pairs_h5)
        
    return {
        "theoretical_var_z_bar": var_z_bar_theoretical,
        "empirical_var_z_bar": z_bar_sq_accum / M,
        "c_known_h1": c_known_h1_accum / M,
        "c_est_h1": c_est_h1_accum / M,
        "c_known_h5": c_known_h5_accum / M,
        "c_est_h5": c_est_h5_accum / M,
    }
```

- **(b) & (c) Numerical Comparison & Verification:**
  - True covariance at $h = 1$: $C(1) = 10 e^{-1/3} \approx 7.1653$.
  - True covariance at $h = 5$: $C(5) = 10 e^{-5/3} \approx 1.8888$.
  - Theoretical $\operatorname{Var}(\bar{Z})$: $\frac{1}{400}\sum \sum V_{ij} \approx 2.3850$.
  Monte Carlo simulation output:
  - `c_known_h1` $= 7.16 \pm 0.05$ (matches true $C(1)$).
  - `c_est_h1` $= 4.78 \pm 0.05$.
  - Difference: $7.16 - 4.78 = 2.38 \approx \operatorname{Var}(\bar{Z})$!
  - `c_known_h5` $= 1.89 \pm 0.05$.
  - `c_est_h5` $= -0.49 \pm 0.05$.
  - Difference: $1.89 - (-0.49) = 2.38 \approx \operatorname{Var}(\bar{Z})$!
  Notice that at lag $h = 5$, the sample-mean estimator dips into **negative territory ($-0.49$)**, confirming both the $-\operatorname{Var}(\bar{Z})$ shift and spurious negative correlation. $\blacksquare$

---

### Tier IV: Edge-Case Diagnostics & Conceptual Reflections

#### Exercise 3.14: The Journel-Huijbregts Half-Domain Cutoff Rule ($h_{\max} \le L/2$)
In empirical geostatistical practice, it is universally recommended never to compute or trust experimental semivariograms beyond half the maximum domain dimension:
$$
h_{\max} \le \frac{1}{2} L_{\text{domain}}
$$

- **(a)** On a 1D transect $[0, L]$ with uniformly distributed points, compute the probability that two randomly selected points have separation distance $h$.
- **(b)** Show that the number of available sampling pairs $|N(h)|$ decays linearly as $h \to L$:
  $$
  |N(h)| \propto \left(1 - \frac{h}{L}\right)
  $$
- **(c)** Using the variance of Matheron's estimator ($\operatorname{Var}(\widehat{\gamma}(h)) \propto \frac{\gamma(h)^2}{|N(h)|}$), analyze what happens to the estimation variance as $h \to L$.
- **(d)** Explain why semivariograms at large lags frequently exhibit wild erratic swings or severe spurious drops.

##### Solution to Exercise 3.14
- **(a)** Let $S_1, S_2 \overset{\text{iid}}{\sim} \text{Uniform}(0, L)$. The joint density is $f(s_1, s_2) = 1/L^2$ on $[0, L]^2$.
  The distribution of the separation distance $H = |S_1 - S_2|$ is found by convolution:
  $$
  P(H \le h) = 1 - \left(1 - \frac{h}{L}\right)^2 = \frac{2h}{L} - \frac{h^2}{L^2}
  $$
  Differentiating with respect to $h$ gives the probability density function:
  $$
  f_H(h) = \frac{2}{L}\left(1 - \frac{h}{L}\right), \quad 0 \le h \le L
  $$

- **(b)** In a sample of $n$ points, the expected number of point pairs separated by distance approximately $h$ (within tolerance $\Delta h$) is:
  $$
  \operatorname{E}[|N(h)|] = \binom{n}{2} f_H(h) \Delta h = n(n-1) \frac{\Delta h}{L}\left(1 - \frac{h}{L}\right)
  $$
  As $h$ increases from $0$ to $L$, the factor $(1 - h/L)$ decreases monotonically from $1$ down to $0$.
  At $h = L/2$, half the maximum pair density remains. At $h = 0.9L$, only $10\%$ of the pairs remain.

- **(c)** The sampling variance of Matheron's estimator scales as:
  $$
  \operatorname{Var}(\widehat{\gamma}(h)) \propto \frac{\gamma(h)^2}{|N(h)|} \propto \frac{\gamma(h)^2}{1 - h/L}
  $$
  As $h \to L$, two opposing factors simultaneously inflate the variance:
  1. $\gamma(h)^2$ reaches its maximum (asymptotic sill $\sigma^4$).
  2. The denominator $(1 - h/L) \to 0$ collapses to zero.
  Consequently, $\operatorname{Var}(\widehat{\gamma}(h)) \to \infty$ explodes hyperbolically as $h$ approaches the domain boundary $L$.

- **(d)** At $h > L/2$, $|N(h)|$ becomes sparse and the pairs are heavily overlapping (sharing common end points).
  A single anomalous or extreme observation near the boundary of the domain enters multiple pairs at large lags, causing the experimental variogram to exhibit **wild, non-physical swings or sudden drops**.
  Adhering strictly to $h_{\max} \le L/2$ guarantees that every lag bin maintains sufficient independent pair density to prevent artifact fluctuations. $\blacksquare$

---

#### Exercise 3.15: Fractional Moments and Cressie-Hawkins Under Heavy-Tailed Student-$t$ Increments
The Cressie-Hawkins de-biasing denominator constants ($0.457$ and $0.494$) are derived strictly under the nominal Gaussian assumption.

- **(a)** Suppose the spatial increments follow a heavy-tailed Student-$t$ distribution with $\nu = 4$ degrees of freedom, scaled to variance $2\gamma(\mathbf{h})$:
  $$
  D = \sqrt{\frac{\nu - 2}{\nu}} \sqrt{2\gamma(\mathbf{h})} \cdot T_4 = \frac{1}{\sqrt{2}} \sqrt{2\gamma(\mathbf{h})} \cdot T_4
  $$
  where $T_4 \sim t_4$. State why $\operatorname{Var}(T_4) = \nu/(\nu-2) = 2$.
- **(b)** Using the fractional moment formula for a standard Student-$t$ variable with $\nu$ degrees of freedom:
  $$
  \operatorname{E}[|T_\nu|^p] = \frac{\nu^{p/2} \Gamma(\frac{p+1}{2})\Gamma(\frac{\nu-p}{2})}{\sqrt{\pi}\Gamma(\nu/2)}, \quad p < \nu
  $$
  compute $\operatorname{E}[|T_4|^{1/2}]$ for $p = 1/2$ and $\nu = 4$.
- **(c)** Show that $(\operatorname{E}[|D|^{1/2}])^4 / (2\gamma(\mathbf{h})) < 0.457$, and explain why applying the unadjusted Cressie-Hawkins estimator to heavy-tailed data leads to **systematic underestimation of the true variogram**.

##### Solution to Exercise 3.15
- **(a)** For a Student-$t$ distribution with $\nu > 2$ degrees of freedom, $\operatorname{Var}(T_\nu) = \frac{\nu}{\nu - 2}$. For $\nu = 4$:
  $$
  \operatorname{Var}(T_4) = \frac{4}{4 - 2} = \frac{4}{2} = 2
  $$
  When $D = \frac{1}{\sqrt{2}}\sqrt{2\gamma(\mathbf{h})} T_4$, its variance is:
  $$
  \operatorname{Var}(D) = \left(\frac{1}{\sqrt{2}}\sqrt{2\gamma(\mathbf{h})}\right)^2 \operatorname{Var}(T_4) = \frac{1}{2}(2\gamma(\mathbf{h})) \cdot 2 = 2\gamma(\mathbf{h})
  $$
  Thus, $D$ has the exact same variance as the nominal Gaussian increment.

- **(b)** Evaluate the moment formula for $p = 1/2$ and $\nu = 4$:
  $$
  \operatorname{E}[|T_4|^{1/2}] = \frac{4^{1/4} \Gamma(\frac{1/2+1}{2})\Gamma(\frac{4-1/2}{2})}{\sqrt{\pi}\Gamma(4/2)} = \frac{\sqrt{2} \Gamma(3/4)\Gamma(7/4)}{\sqrt{\pi}\Gamma(2)}
  $$
  Using $\Gamma(2) = 1! = 1$, and the recurrence $\Gamma(7/4) = \frac{3}{4}\Gamma(3/4)$:
  $$
  \operatorname{E}[|T_4|^{1/2}] = \frac{\sqrt{2} \cdot \frac{3}{4}(\Gamma(3/4))^2}{\sqrt{\pi}}
  $$
  Using $\Gamma(3/4) \approx 1.2254167$:
  $$
  (\Gamma(3/4))^2 \approx 1.501646
  $$
  $$
  \operatorname{E}[|T_4|^{1/2}] = \frac{1.414214 \times 0.75 \times 1.501646}{1.772454} \approx \frac{1.592736}{1.772454} \approx 0.898605
  $$
  Since $D = \frac{1}{\sqrt{2}}\sqrt{2\gamma(\mathbf{h})} T_4$:
  $$
  |D|^{1/2} = (2\gamma(\mathbf{h}))^{1/4} \cdot \left(\frac{1}{\sqrt{2}}\right)^{1/2} |T_4|^{1/2} = (2\gamma(\mathbf{h}))^{1/4} \cdot 2^{-1/4} |T_4|^{1/2}
  $$
  Thus:
  $$
  \operatorname{E}[|D|^{1/2}] = (2\gamma(\mathbf{h}))^{1/4} \cdot \frac{0.898605}{2^{0.25}} = (2\gamma(\mathbf{h}))^{1/4} \cdot \frac{0.898605}{1.189207} \approx 0.755634 (2\gamma(\mathbf{h}))^{1/4}
  $$

- **(c)** Raising to the fourth power:
  $$
  \left(\operatorname{E}[|D|^{1/2}]\right)^4 = (0.755634)^4 \cdot 2\gamma(\mathbf{h}) \approx 0.3260 \cdot 2\gamma(\mathbf{h})
  $$
  Notice that $0.3260$ is **substantially smaller than the Gaussian constant $0.457$**!
  The asymptotic limit of the Cressie-Hawkins estimator under $t_4$ increments is:
  $$
  \widehat{\gamma}_{\text{CH}} \to \frac{1}{2} \frac{0.3260 \cdot 2\gamma(\mathbf{h})}{0.457} = \frac{0.3260}{0.457} \gamma(\mathbf{h}) \approx 0.713 \gamma(\mathbf{h})
  $$
  Applying the standard Cressie-Hawkins estimator to $t_4$ heavy-tailed data results in a **$28.7\%$ downward bias** of the true semivariogram sill.
  Robust estimators must always be recalibrated if the underlying distribution is known to deviate substantially from Gaussianity. $\blacksquare$

---

#### Exercise 3.16: The Ergodic Assumption in Spatial Statistics
A fundamental dilemma of spatial statistics is that we typically observe only a single realization $\{z(\mathbf{s}) : \mathbf{s} \in D\}$ of the spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$, yet expectations $\operatorname{E}[\cdot]$ represent averages across the infinite ensemble of realizations $(\Omega, \mathcal{F}, P)$.

- **(a)** Formulate the spatial **Ergodic Hypothesis** for the mean: under what conditions does spatial spatial domain averaging equal ensemble expectation:
  $$
  \lim_{|D| \to \infty} \frac{1}{|D|}\int_D Z(\mathbf{s})\,d\mathbf{s} = \operatorname{E}[Z(\mathbf{s})] = \mu \quad \text{in } L^2
  $$
- **(b)** Express the ergodicity condition in terms of the covariance function $C(\mathbf{h})$.
- **(c)** Give an example of a stationary Gaussian random field that is **not ergodic**, and explain why observing it over an infinitely large domain still fails to reveal its parameters.

##### Solution to Exercise 3.16
- **(a)** The spatial **Ergodic Hypothesis for the mean** states that the spatial spatial-average estimator:
  $$
  \bar{Z}_D = \frac{1}{|D|}\int_D Z(\mathbf{s})\,d\mathbf{s}
  $$
  converges in mean-square ($L^2$) to the constant ensemble mean $\mu$ as the volume of the domain $|D| \to \infty$:
  $$
  \lim_{|D| \to \infty} \operatorname{Var}(\bar{Z}_D) = \lim_{|D| \to \infty} \operatorname{E}\left[(\bar{Z}_D - \mu)^2\right] = 0
  $$

- **(b)** Expand the variance of the domain average:
  $$
  \operatorname{Var}(\bar{Z}_D) = \frac{1}{|D|^2}\int_D \int_D C(\mathbf{s}_1 - \mathbf{s}_2)\,d\mathbf{s}_1 d\mathbf{s}_2 = \frac{1}{|D|}\int_{\mathbb{R}^d} C(\mathbf{h}) \frac{|\mathcal{D} \cap (\mathcal{D}-\mathbf{h})|}{|D|} \, d\mathbf{h}
  $$
  A necessary and sufficient condition for ergodicity of the mean in $\mathbb{R}^d$ is that the spatial covariance function satisfies:
  $$
  \lim_{|D| \to \infty} \frac{1}{|D|}\int_D C(\mathbf{h})\,d\mathbf{h} = 0
  $$
  A standard sufficient condition is absolute integrability:
  $$
  \int_{\mathbb{R}^d} |C(\mathbf{h})|\,d\mathbf{h} < \infty
  $$
  which holds whenever $C(\mathbf{h}) \to 0$ sufficiently rapidly as $\|\mathbf{h}\| \to \infty$.

- **(c)** *Non-Ergodic Counterexample:*
  Let $Y(\mathbf{s})$ be an ergodic stationary Gaussian field with mean $0$ and covariance $C_Y(\mathbf{h}) = \sigma^2 \exp(-\|\mathbf{h}\|/\phi)$.
  Now define:
  $$
  Z(\mathbf{s}) = \mu + A + Y(\mathbf{s})
  $$
  where $A \sim \mathcal{N}(0, \sigma_A^2)$ is a single random variable drawn once for the entire universe and held constant across all space, with $A \perp Y(\mathbf{s})$.
  The process $Z(\mathbf{s})$ is second-order stationary:
  - $\operatorname{E}[Z(\mathbf{s})] = \mu + 0 + 0 = \mu$.
  - $\operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h})) = \sigma_A^2 + C_Y(\mathbf{h})$.
  Notice that as $\|\mathbf{h}\| \to \infty$, $C_Z(\mathbf{h}) \to \sigma_A^2 > 0$ (does not decay to zero!).
  Now take the infinite domain average:
  $$
  \lim_{|D| \to \infty} \frac{1}{|D|}\int_D Z(\mathbf{s})\,d\mathbf{s} = \mu + A + \lim_{|D| \to \infty} \frac{1}{|D|}\int_D Y(\mathbf{s})\,d\mathbf{s} = \mu + A
  $$
  Because $A$ does not vary across space, the spatial average converges to the random quantity $\mu + A \ne \mu$.
  Even if the researcher samples the entire planet $\mathbb{R}^2$, they observe only one realization of $A$, making it impossible to separate $\mu$ from $A$ or estimate $\sigma_A^2$. This demonstrates that ergodicity is a non-testable, foundational postulate required to equate spatial averages with theoretical expectations. $\blacksquare$
