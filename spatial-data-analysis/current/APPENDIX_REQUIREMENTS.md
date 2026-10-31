# Requirements for the Appendices

Compiled on 2026-10-25 from a read-through of Chapters 0 to 13 (review F, with additions from reviews A to E in `reviews/`), and updated on 2026-10-31 following the completion and approval of Chapters 10 to 13. Each item names where a chapter expects it. Line numbers refer to chapter files and may drift. Use this list when drafting the appendices, and ensure `OUTLINE.md` (Appendices section) remains in perfect sync.

### Settled Architectural Decisions
1. **Appendix A Structure:** Appendix A opens with **A.0: Review of the Five Tools of Chapter 0** (linear transformations, quadratic forms, multivariate normal density, maximum likelihood, and normal equations/GLS), followed by A.1 to A.5, and closes with **A.6: Matrix Facts Used in the Text** (spectral theorem, Cholesky, idempotent projections, pseudo-inverse, Neumann series, Gaussian integral, Cauchy-Schwarz, etc.).
2. **Concordance Home:** The Schabenberger and Gotway (2005) notation concordance and master symbol table remain in `NOTATION.md` in the front matter. **Appendix C** is strictly the conceptual **Jargon Rosetta Stone** (translating domain jargon into linear models and GLMs), with a brief opening cross-reference to `NOTATION.md`.
3. **Appendix D Scope:** Appendix D gathers all advanced, skippable, or graduate-level frontiers deferred across Chapters 0–13 (finite-element SPDE/INLA buffer meshes, Spatial Durbin model, alternatives to plug-in prediction, Bochner covariance validity, spatial confounding/RSR, non-Gaussian geostatistics, and spatial ML).

---

### Appendix A: Matrix Algebra and Multivariate Normal Calculus

Expected by the chapters (location of each expectation in parentheses):

0. **A.0 Review of the five tools** (`ch00:225`, `:334`, `:415`):
   - Linear transformations of random vectors ($\operatorname{E}[\mathbf{B}\mathbf{Y}] = \mathbf{B}\boldsymbol{\mu}$, $\operatorname{Var}(\mathbf{B}\mathbf{Y}) = \mathbf{B}\boldsymbol{\Sigma}\mathbf{B}'$).
   - Quadratic forms $\mathbf{y}'\mathbf{M}\mathbf{y}$ and positive semi-definiteness as non-negative variance ($\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} \ge 0$).
   - Multivariate normal density: determinant penalty term $-\frac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert$ versus Mahalanobis distance term $-\frac{1}{2}(\mathbf{y}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{y}-\boldsymbol{\mu})$.
   - Maximum likelihood by score: score equations, Fisher information, and concentrated/profile log-likelihoods.
   - OLS normal equations $\mathbf{X}'\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\mathbf{y}$, orthogonal projections $\mathbf{P}_{\mathbf{X}}$, and the generalized normal equations $\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{y}$.
   - Keyed directly to the warm-up exercises of §0.7.

1. **A.1 Block-matrix inversion and the Schur complement:**
   - Full 4-block verification of partitioned inverse by block Gaussian elimination ("Appendix A.1 does this in full", `ch04:378`, §4.3.4).
   - Both forms of the upper-left block: $(\mathbf{M}_{11} - \mathbf{M}_{12}\mathbf{M}_{22}^{-1}\mathbf{M}_{21})^{-1}$ and $\mathbf{M}_{11}^{-1} + \mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1}$.
   - Block determinant identity: $\lvert\mathbf{M}\rvert = \lvert\mathbf{M}_{11}\rvert\lvert\mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\rvert = \lvert\mathbf{M}_{11}\rvert / \lvert(\mathbf{M}^{-1})_{22}\rvert$ (`ch05:421`, §5.3.4).
   - Sherman–Morrison–Woodbury identity and matrix determinant lemma derived from block elimination (`ch13` §13.2.2, `ch13:1253`).
   - Special Woodbury form used in Henderson equations: $\mathbf{V} - \mathbf{V}(\mathbf{V}+\mathbf{Q})^{-1}\mathbf{V} = (\mathbf{V}^{-1}+\mathbf{Q}^{-1})^{-1}$ (`ch09:647`, §9.4.1; `ch09` Exercise 8).
   - Schur complement derivation of LOOCV shortcut matrix $\boldsymbol{\Pi} = \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}$ (`ch04` §4.7.4; `ch05` §5.3.4).

2. **A.2 Vector and matrix derivatives:**
   - Gradients of linear and quadratic forms: $\partial(\boldsymbol{\lambda}'\mathbf{c})/\partial\boldsymbol{\lambda} = \mathbf{c}$, $\partial(\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda})/\partial\boldsymbol{\lambda} = 2\boldsymbol{\Sigma}\boldsymbol{\lambda}$ ("Appendix A.2 collects them", `ch04:118`, §4.2.1).
   - Derivative of matrix inverse: $\partial\boldsymbol{\Sigma}^{-1}/\partial\theta = -\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}}\boldsymbol{\Sigma}^{-1}$ ("Appendix A.2 collects them", `ch05:463`, §5.4.2).
   - Derivative of log-determinant: $\partial\log\lvert\boldsymbol{\Sigma}\rvert/\partial\theta = \operatorname{tr}(\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}})$ (`ch05:463`, §5.4.2).
   - Derivative of projection/residual precision: $\partial\boldsymbol{\Pi}/\partial\theta_j = -\boldsymbol{\Pi}\dot{\boldsymbol{\Sigma}}_j\boldsymbol{\Pi}$ (`ch05` Exercise 6).
   - Second derivatives for observed and expected Fisher information matrices (`ch05` §5.4.6, `ch07` §7.5).

3. **A.3 Moments of quadratic forms in normal vectors:**
   - First moment: $\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}$ (`ch01` §1.1.1; used in Chapters 5, 6, 7).
   - Second moment / variance: $\operatorname{Var}(\mathbf{Y}'\mathbf{M}\mathbf{Y}) = 2\operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}\boldsymbol{\Sigma}) + 4\boldsymbol{\mu}'\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}\boldsymbol{\mu}$, reducing to $2\sigma^4\operatorname{tr}(\mathbf{M}^2)$ under the null for symmetric $\mathbf{M}$ (`ch06` §6.5.3).
   - Non-symmetric quadratic forms: $\operatorname{Var}(\mathbf{Y}'\mathbf{A}\mathbf{Y}) = \sigma^4\operatorname{tr}(\mathbf{A}^2 + \mathbf{A}\mathbf{A}')$ (`ch07` §7.6.4).
   - Trace properties: cyclic invariance $\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{tr}(\mathbf{C}\mathbf{A}\mathbf{B})$ and linearity.
   - Pitman–Koopmans theorem on the independence of quadratic ratios and scale invariance under normality (`ch06` §6.5.2).

4. **A.4 Conditional distributions of partitioned normals, and precision-matrix forms:**
   - Covariance form: $\mathbf{Y}_1 \mid \mathbf{Y}_2 \sim N(\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{Y}_2 - \boldsymbol{\mu}_2),\, \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21})$ (`ch04:622`, §4.5.3).
   - Precision form via completing the square: $y_i \mid \mathbf{y}_{-i} \sim N(\mu_i - \sum_{j \ne i}\frac{Q_{ij}}{Q_{ii}}(y_j - \mu_j),\, Q_{ii}^{-1})$ (`ch08` §8.1.2).
   - Sparse Cholesky factorization of precision: $\boldsymbol{\Sigma}^{-1} = (\mathbf{I}-\boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I}-\boldsymbol{\Gamma})$ and $\mathbf{L}^{-1} = \mathbf{F}^{-1/2}(\mathbf{I}-\boldsymbol{\Gamma})$ (`ch13` §13.4.1).
   - Vecchia Directed Acyclic Graph (DAG) structural equation: $\mathbf{y} = \boldsymbol{\mu} + \boldsymbol{\Gamma}(\mathbf{y}-\boldsymbol{\mu}) + \boldsymbol{\varepsilon}$, strictly lower-triangular $\boldsymbol{\Gamma}$, exact terminating Neumann series $(\mathbf{I}-\boldsymbol{\Gamma})^{-1} = \sum_{m=0}^{n-1}\boldsymbol{\Gamma}^m$, and path interpretations in DAGs (`ch13` §13.4.1, Exercise 13).

5. **A.5 Kronecker products and the vec operator:**
   - Mixed-product rule: $(\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A}\mathbf{C}) \otimes (\mathbf{B}\mathbf{D})$ (`ch13` §13.6.2).
   - Inverse rule: $(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}$.
   - Determinant exponents: $\lvert\mathbf{A} \otimes \mathbf{B}\rvert = \lvert\mathbf{A}\rvert^{n_B}\lvert\mathbf{B}\rvert^{n_A}$ (`ch13` §13.6.2).
   - Vectorization identity: $\operatorname{vec}(\mathbf{A}\mathbf{X}\mathbf{B}) = (\mathbf{B}' \otimes \mathbf{A})\operatorname{vec}(\mathbf{X})$ (`ch13` §13.6.2).
   - Triangular and Cholesky preservation: $\mathbf{L}_T \otimes \mathbf{L}_S$ is lower-triangular Cholesky factor of $\boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ (`ch13` Exercise 12).

6. **A.6 Facts about matrices used in the text:**
   - Spectral theorem for symmetric matrices: real eigenvalues, orthonormal eigenvectors $\mathbf{U}'\mathbf{U} = \mathbf{I}_n$, $\operatorname{tr}(\boldsymbol{\Sigma}) = \sum \lambda_i$, $\lvert\boldsymbol{\Sigma}\rvert = \prod \lambda_i$ (`ch01:254`, §1.3.3; `ch02` §2.6).
   - Symmetric square root $\boldsymbol{\Sigma}^{1/2} = \mathbf{U}\boldsymbol{\Lambda}^{1/2}\mathbf{U}'$ and inverse square root decorrelation $\boldsymbol{\Sigma}^{-1/2}$ versus lower-triangular Cholesky $\mathbf{L}^{-1}$ (`ch01` Exercise 11).
   - Cholesky factorization: existence, uniqueness, and positive-definiteness check (`ch02` §2.7.1).
   - Idempotent projection matrices: $\mathbf{P}^2 = \mathbf{P}$, $\operatorname{tr}(\mathbf{P}) = \operatorname{rank}(\mathbf{P})$, orthogonal complement $\mathbf{I} - \mathbf{P}$ (`ch01` §1.1.2).
   - Lagrange multipliers for equality constraints (`ch04` §4.3.1).
   - Moore–Penrose pseudo-inverse $\mathbf{Q}^+$ and generalized determinant $\lvert\mathbf{Q}\rvert^* = \prod_{\lambda_i > 0}\lambda_i$ (`ch08` §8.3.4).
   - Neumann series for $(\mathbf{I} - \rho\mathbf{W})^{-1} = \sum_{k=0}^\infty \rho^k\mathbf{W}^k$ for $\lvert\rho\lambda_{\max}\rvert < 1$ (`ch07` §7.3.2).
   - Multivariate Gaussian integral: $\int_{\mathbb{R}^n} \exp(-\frac{1}{2}\mathbf{u}'\mathbf{H}\mathbf{u})\,d\mathbf{u} = (2\pi)^{n/2}\lvert\mathbf{H}\rvert^{-1/2}$ (`ch09` §9.5.1).
   - Cauchy–Schwarz inequality for inner products and quadratic forms: $(\mathbf{a}'\mathbf{b})^2 \le (\mathbf{a}'\mathbf{a})(\mathbf{b}'\mathbf{b})$ and $(\mathbf{1}'\mathbf{R}^{-1}\mathbf{1})(\mathbf{1}'\mathbf{R}\mathbf{1}) \ge n^2$ (`ch01` Exercise 7).
   - Hadamard (Schur) product rule: entrywise product $\mathbf{A} \circ \mathbf{B}$ of positive semi-definite matrices is positive semi-definite (`ch02` §2.4.3; `ch13` §13.3.1).

---

### Appendix B: Scientific Python Reference Patterns

Patterns the chapters point to or reuse (chapters in parentheses):

- **B.1 SymPy for analytical verification:**
  - `sp.symbols`, `sp.Matrix`, symbolic derivatives, symbolic matrix inverses, and score solving (Chapters 0, 4, 5).
- **B.2 Stable numerical linear algebra:**
  - Stable Cholesky solves: `scipy.linalg.cho_factor` and `cho_solve` (Chapters 5, 13); `solve_triangular` (5, 8).
  - Standard solves: `np.linalg.solve` and `np.linalg.cholesky`.
  - Log-determinants: `np.linalg.slogdet` and $2\sum\log L_{ii}$ via Cholesky (5, 7, 9, 12, 13).
  - Symmetric eigenvalue problems: `np.linalg.eigh` and `np.linalg.eigvalsh` (1, 2, 5, 6, 7, 8, 9, 13).
  - Least squares: `np.linalg.lstsq` (4, 5, 7).
  - Tensor contractions: `np.einsum` (2, 3, 13).
  - Explicit `np.linalg.inv` policy: why direct inversion is avoided except when extracting analytical asymptotic variance diagonals (e.g. `(X'X)^{-1}`, `Pi`).
- **B.3 Sparse matrices and sparse linear algebra:**
  - Compressed Sparse Column (CSC) and Compressed Sparse Row (CSR) storage formats: three-array representation (`data`, `indices`, `indptr`) ("compressed sparse column ... Appendix B.3", `ch08:679`, §8.4.3).
  - Diagonal creation: `scipy.sparse.diags`.
  - Sparse factorization: `scipy.sparse.linalg.splu` (SuperLU) with fill-reducing column permutation orderings (Chapters 8, 13).
  - Compiled sparse Cholesky: why SciPy lacks native sparse Cholesky, and the role of CHOLMOD / SuiteSparse (`scikit-sparse`) with Approximate Minimum Degree (AMD) ordering (`ch08` §8.4.3, `ch13` §13.3.2).
  - Iterative sparse solvers: conjugate gradient `scipy.sparse.linalg.cg` (`ch08` Exercise 17, `ch13` Exercise 7).
  - Graph connectivity: `scipy.sparse.csgraph.connected_components` (`ch09` §9.3.4).
- **B.4 Optimization and root finding:**
  - `scipy.optimize.minimize` with `L-BFGS-B` for bounded and log-transformed variance parameters (5, 9, 13).
  - `Nelder-Mead` simplex search for derivative-free variogram and likelihood fitting (3, 9, 11, 13).
  - Scalar root finding: `scipy.optimize.brentq` for practical ranges and interval bounds (2, 8).
  - Nonlinear least squares: `scipy.optimize.least_squares` for point process minimum contrast on summary functions (11, 12).
  - Profile log-likelihood loops and numerical Hessians (`ch05`, `ch07`, `ch12`).
- **B.5 Special functions and distributions:**
  - `scipy.special.kv` (modified Bessel of the second kind) and `scipy.special.gamma` for Matérn covariance and spectral densities (2, 13).
  - `scipy.special.xlogy` for numerically stable $x\log y$ evaluation with empty quadrature cells (11).
  - `scipy.stats` distributions: `t.ppf`, `chi2.ppf`, `norm.cdf`, `pearsonr`.
- **B.6 Spatial visualization with matplotlib:**
  - Headless rendering setup: `matplotlib.use("Agg")`.
  - Grid imaging: `imshow(..., origin="lower")` (11 chapters).
  - Contour lines: `contour` (5, 9).
  - Spatial point plots: `scatter` (6, 9, 11, 12, 13).
  - Consistent layout and export: `tight_layout()`, `savefig("figures/chNN/...")`.
- **Recurring conventions and performance patterns:**
  - Modern RNG seeding: `np.random.default_rng(seed)` passed explicitly across all functions.
  - Pairwise Euclidean distance matrices: vectorized quadratic expansion with `np.fill_diagonal(D, 0.0)` and `np.maximum(..., 0.0)` to eliminate floating-point precision hazards (`ch01` §1.6.1, `ch05` §5.6.3).
  - Single-threaded BLAS execution: pinning `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` to eliminate multi-threading thrashing during hundreds of small matrix operations (`ch07`, `scripts/run_chapter_code.py`).
  - High-precision wall-clock benchmarking: `time.perf_counter()` (8, 13).
  - Fast spatial neighbor search: `scipy.spatial.cKDTree` and `cdist` for large-$n$ scaling (`ch13` §13.4.3).
  - Kronecker matrix operations: `np.kron` and matrix-reshaping tricks for Kronecker-vector products $(\mathbf{B}' \otimes \mathbf{A})\operatorname{vec}(\mathbf{X}) = \operatorname{vec}(\mathbf{A}\mathbf{X}\mathbf{B})$ (`ch13` §13.6).

---

### Appendix C: A Jargon Rosetta Stone

Translates domain-specific vocabulary into standard linear-model, multivariate normal, and GLM concepts (chapter in parentheses):

- **Geostatistics (Chapters 2 to 5):**
  - Kriging (simple, ordinary, universal) = Best Linear Unbiased Prediction (BLUP) with known mean, unknown constant mean, and linear regression mean.
  - Nugget ($\tau^2$) = Independent observation error variance (white noise).
  - Partial sill ($\sigma^2$) = Variance scale of the spatially correlated random effect.
  - Sill ($\sigma^2 + \tau^2$) = Total marginal variance of the observation process.
  - Range ($\phi$) / Practical range = Spatial correlation length scale (distance where correlation drops to $0.05$).
  - Semivariogram $\gamma(h)$ = Half the variance of pairwise differences: $\frac{1}{2}\operatorname{Var}(Y(\mathbf{s}) - Y(\mathbf{s}+\mathbf{h}))$.
  - Stationarity (second-order) = Constant mean and covariance depending only on displacement vector $\mathbf{h}$.
  - Isotropy = Covariance depending only on Euclidean separation distance $h = \lVert\mathbf{h}\rVert$.
  - Screening effect = Conditional independence property where intervening points screen off correlation from distant points.
  - Block kriging / Change of support = BLUP of a linear functional (spatial average) of the random field.
  - REML (Restricted Maximum Likelihood) = Maximum likelihood on linearly transformed error contrasts orthogonal to the regression mean $\mathbf{X}$.
  - Spatial confounding = Collinearity between smooth spatial covariates and smooth spatial random effects.
  - LOOCV shortcut residual = Scaled residual from the projection/precision matrix: $e_{-i} = -[\boldsymbol{\Pi}\mathbf{y}]_i / \Pi_{ii}$.

- **Areal and Lattice Models (Chapters 6 to 9):**
  - Modifiable Areal Unit Problem (MAUP) = Aggregation distortion arising from scale and zoning of continuous processes.
  - Ecological fallacy = Drawing individual-level causal inferences from aggregate area-level regressions.
  - Spatial weights matrix ($\mathbf{W}$) = Adjacency / interaction graph matrix, usually row-standardized.
  - Moran's $I$ = Ratio of quadratic forms in centered data / residuals, measuring spatial autocorrelation.
  - Geary's $C$ = Normalized sum of squared pairwise differences along graph edges.
  - SAR (Spatial Autoregressive model) = Simultaneous autoregression generating spatial covariance through the matrix inverse $(\mathbf{I}-\rho\mathbf{W})^{-1}$.
  - Feedback matrix $\boldsymbol{\Phi} = (\mathbf{I}-\rho\mathbf{W})^{-1}$ and Impacts (direct, indirect, total) = Infinite Neumann series of spillover shock propagation across network edges.
  - CAR (Conditional Autoregressive model) = Spatial Markov random field specified through full conditional distributions and sparse precision matrix $\mathbf{Q}$.
  - Proper CAR vs. ICAR = Full-rank precision $\sigma^{-2}(\mathbf{D} - \rho\mathbf{A})$ versus rank-deficient graph Laplacian $\sigma^{-2}(\mathbf{D} - \mathbf{A})$ with improper flat prior on the intercept.
  - BYM / BYM2 / Leroux = Convolution models combining structured spatial random effects with unstructured independent exchangeable random effects.
  - Henderson's equations = Penalized linear mixed-model equations for jointly solving fixed effects and spatial random effects.
  - Laplace approximation = Second-order Taylor expansion around the mode to integrate out non-Gaussian random effects.

- **Point Patterns (Chapters 10 to 12):**
  - Complete Spatial Randomness (CSR) = Homogeneous spatial Poisson point process.
  - First-order intensity $\lambda(\mathbf{s})$ = Expected number of points per unit area.
  - Ripley's $K(r)$ / Besag's $L(r)$ = Cumulative second-order summary of expected points within radius $r$ of a typical point, normalized by intensity.
  - Pair correlation function $g(r)$ = Ratio of joint second-order density to the product of marginal densities ($g(r) = 1$ under CSR).
  - Berman–Turner device = Discretization of continuous Poisson log-likelihood into a weighted Poisson GLM with quadrature weights and dummy points.
  - Logistic bridge = Transforming an inhomogeneous case-control point pattern into an ordinary binary logistic regression.
  - Scan statistic = Maximized Poisson likelihood ratio across candidate spatial scanning windows.
  - Minimum contrast estimation = Nonlinear least squares (moment-matching M-estimation / GMM) fitting theoretical summary functions ($K$ or $g$) to empirical estimates.
  - Neyman–Scott / Thomas / Matérn cluster processes = Cox processes where Poisson offspring cluster around unobserved Poisson parents.
  - Log-Gaussian Cox Process (LGCP) = Poisson process whose log-intensity is a continuous Gaussian random field ($\log\Lambda(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + u(\mathbf{s})$).
  - Papangelou conditional intensity $\lambda^*(\mathbf{u})$ = Conditional rate of an event occurring at $\mathbf{u}$ given the remaining point configuration.
  - Strauss process = Gibbs point process with pairwise inhibition potential $\gamma = e^\zeta \le 1$.
  - Ruelle stability failure = Explosive density collapse when $\gamma > 1$ without saturation bounds.
  - Pseudolikelihood = Objective function replacing intractable joint normalizing constants by the product of full Papangelou conditional densities.

- **Large Data and Matrix Approximations (Chapter 13):**
  - Low-rank spatial model / Predictive process = Projecting a continuous spatial process onto $q \ll n$ knots via conditional expectations, resulting in diagonal-plus-low-rank covariance.
  - Variance deficit = Unaccounted simple kriging variance $\sigma^2(1 - \mathbf{r}'\mathbf{R}_K^{-1}\mathbf{r})$ in plain predictive processes, repaired by modified predictive processes.
  - Covariance tapering = Multiplying candidate covariance matrices entrywise by compactly supported Wendland correlation functions to induce structural sparsity.
  - Vecchia approximation = Factoring the joint multivariate normal density into univariate conditionals, restricted to $k \ll n$ nearest previous neighbors.
  - Vecchia DAG and moral graph = Representing Vecchia's model as a Gaussian Directed Acyclic Graph whose undirected moral graph defines a sparse Cholesky factor of the precision.
  - SPDE bridge = Equivalence established by Lindgren et al. (2011) between continuous Matérn Gaussian fields and discrete GMRFs satisfying stochastic partial differential equations on triangulated meshes.
  - Neumann boundary reflection ("hall of mirrors") = Variance inflation (up to $2\times$ at edges, $4\times$ at corners) on bounded grid Laplacians with zero-flux boundary conditions.
  - Separable space-time covariance = Kronecker product structure $\boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ assuming spatial and temporal correlation structures decouple multiplicatively.

---

### Appendix D: Where to Go Next

Frontiers and advanced literature connections deferred by the text:

1. **Spatial sampling design** (`ch04:154`, §4.2.2): Optimal location placement to minimize mean squared prediction error (MSPE), exploiting the property that kriging variance depends only on configuration geometry, not observation values $\mathbf{y}$.
2. **Alternatives to plug-in prediction** (`ch05:689`, §5.4.5): Accounting for covariance-parameter uncertainty via the parametric bootstrap, restricted likelihood predictive distributions, and fully Bayesian posterior predictive averaging.
3. **Spatial confounding and causal inference** (`ch05:899`, §5.6.2; `ch09` §9.6.5): Resolving variance inflation and parameter bias between smooth spatial covariates and spatial random effects via Restricted Spatial Regression (RSR) [@reich2006; @hughes2013; @hanks2015] and modern spatial causal inference frameworks.
4. **The full SPDE approach and finite-element meshes** (`ch13:896`, `:1253`):
   - Triangulated irregular meshes for continuous domains.
   - Solving stochastic partial differential equations via finite elements.
   - Using expanded buffer zones in R-INLA meshes to eliminate the Neumann boundary reflection / variance inflation analyzed in Chapter 13 §13.5.2.
   - Bayesian spatial inference via INLA (Integrated Nested Laplace Approximations) and MCMC.
5. **Nonseparable and multivariate space-time models** (`ch13:1043`, `:1253`):
   - Gneiting's nonseparable space-time covariance class [@gneiting2002].
   - Cokriging, coregionalization models, and multivariate spatial random fields.
6. **Non-Gaussian geostatistics and trans-Gaussian kriging** (`ch04` §4.5.4, `ch13:1253`):
   - Copula-based spatial models, skew-normal fields, and Tukey $g$-and-$h$ transformations beyond lognormal corrections.
7. **Deeper theory of covariance validity and spectral representations** (`ch00:355`, `ch02:295`):
   - Bochner's theorem for positive-definite functions in Euclidean space.
   - Schoenberg's theorem for valid isotropic correlation functions.
   - Spectral representation theorem and continuous Fourier transforms of stationary random fields.
8. **Covariance models on the sphere for global data** (`ch01` §1.6.4):
   - Validity constraints using great-circle distance versus chordal distance [@gneiting2013].
   - Strictly positive definite kernel constructions on spheres $\mathbb{S}^2$ and Riemannian manifolds.
9. **The Spatial Durbin Model and higher-order spatial econometrics** (`ch07:471`, `:1253`):
   - Spatial Durbin model $\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \mathbf{W}\mathbf{X}\boldsymbol{\gamma} + \boldsymbol{\varepsilon}$.
   - Common-factor hypothesis testing ($\boldsymbol{\gamma} = -\rho\boldsymbol{\beta}$) linking the Durbin model to the spatial error model.
10. **Marked point patterns and space-time point processes** (`ch10` §10.7, `ch13:1253`):
    - Continuous mark distributions, Stoyan's mark correlation function $k_{mm}(r)$, and space-time Hawkes self-exciting point processes.
11. **GMM and composite likelihood for spatial point processes** (`ch11` §11.8, `ch12` §12.2.5):
    - Diggle (2003) generalized method of moments for minimum contrast on summary functions.
    - Second-order composite likelihood estimation when full likelihoods are intractable.
12. **Geographically Weighted Regression and spatial machine learning** (`ch04` §4.5.3, `ch13:1253`):
    - Spatial random forests, spatial neural operators, and physics-informed neural networks (PINNs) connecting continuous SPDEs to modern machine learning.
