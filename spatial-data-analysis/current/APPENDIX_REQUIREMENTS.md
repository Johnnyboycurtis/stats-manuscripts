# Requirements for the Appendices

Compiled on 2026-10-25 from a read-through of Chapters 0 to 13 (review F, with additions from reviews A to E in `reviews/`). Each item names where a chapter expects it. Line numbers refer to the chapter files on that date and will drift. Use this list when drafting the appendices, and update `OUTLINE.md` (Appendices section) to match what is decided.

Decisions the list depends on (see `CONSISTENCY_REVIEW.md`, section 6): whether Appendix A also reviews the five tools of Chapter 0, and whether the Schabenberger and Gotway concordance stays in `NOTATION.md` or moves to Appendix C.

### Appendix A: Matrix Algebra and Multivariate Normal Calculus

Expected by the chapters (location of each expectation in parentheses):

0. **(new) Review of the five tools** (`ch00:225`, `:334`, `:415`): linear transformations of random vectors ($\operatorname{E}[\mathbf{B}\mathbf{Y}]$, $\operatorname{Var}(\mathbf{B}\mathbf{Y})$); quadratic forms and positive semi-definiteness as non-negative variance; the multivariate normal density (determinant term and Mahalanobis term); maximum likelihood by score; the OLS normal equations and projection. Each with the warm-up exercise it supports.
1. **A.1 Block-matrix inversion and the Schur complement.** Verified by multiplication ("Appendix A.1 does this in full", `ch04:378`, §4.3.4); the block determinant formula (`ch05:421`, §5.3.4); the Woodbury identity by name (`ch09:647`, §9.4.1 and `ch09:1326`, §9.6.5 say it "follows from the block inversion formula of Appendix A.1"); the matrix determinant lemma and Sylvester identity (derived in `ch13` §13.2.2, and Chapter 13's closing says Appendix A gathers them, `ch13:1253`); the Schur-complement check of conditional normals (`ch08` §8.1.2).
2. **A.2 Vector and matrix derivatives.** Gradients of linear and quadratic forms ("Appendix A.2 collects them", `ch04:118`, §4.2.1); $\partial\log\lvert\boldsymbol{\Sigma}\rvert/\partial\theta = \operatorname{tr}(\boldsymbol{\Sigma}^{-1}\dot{\boldsymbol{\Sigma}})$ and $\partial\boldsymbol{\Sigma}^{-1}/\partial\theta$ ("Appendix A.2 collects them", `ch05:463`, §5.4.2); second derivatives for the observed information (§5.4.6).
3. **A.3 Moments of quadratic forms in normal vectors.** $\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}$ (derived in `ch01` §1.1.1; used in Chapters 5, 6, 7); $\operatorname{Var}(\mathbf{Y}'\mathbf{M}\mathbf{Y}) = 2\sigma^4\operatorname{tr}(\mathbf{M}^2)$ and the second moments used for Moran's $I$ (§6.5.3); the trace properties (cyclic, linear); the Pitman-Koopmans independence statement (§6.5.2).
4. **A.4 Conditional distributions of partitioned normals, and the precision-matrix forms.** Cited at `ch04:622` (§4.5.3); the completing-the-square derivation in `ch08` §8.1.2; the Vecchia factorization $\boldsymbol{\Sigma}^{-1} = (\mathbf{I}-\boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I}-\boldsymbol{\Gamma})$ (`ch13` §13.4.1).
5. **A.5 Kronecker products and the vec operator.** Mixed-product rule, determinant exponents $\lvert\mathbf{A}\otimes\mathbf{B}\rvert = \lvert\mathbf{A}\rvert^q\lvert\mathbf{B}\rvert^p$, the vec identity (`ch13` §13.6.2; proofs are in the chapter, so the appendix may only state them).
6. **Facts "collected in Appendix A" without a numbered home:** the spectral theorem for symmetric matrices, orthonormal eigenvectors, $\operatorname{tr} = \sum\lambda$, $\lvert\cdot\rvert = \prod\lambda$, symmetric square roots (`ch01:254`, §1.3.3; `ch02` §2.6; Chapter 1 Exercise 7); existence and uniqueness of the Cholesky factor (`ch02` §2.7.1); idempotent projection matrices and their trace (`ch01` §1.1.2); the Moore-Penrose pseudo-inverse and generalized determinant (`ch08` §8.3.4); the Neumann series for $(\mathbf{I}-\rho\mathbf{W})^{-1}$ (`ch07` §7.3.2); the Gaussian integral (`ch09` §9.5.1); Cauchy-Schwarz (`ch01` Exercise 7).

Statements chapters take from the literature without proof, which the appendix may cite rather than prove: Bochner's theorem and validity of the named correlation families (`ch00:355`, `ch02:295`, §2.4.4); the representer theorem (`ch04:642`); Zhang (2004) on estimability (`ch05:669`); Kirchhoff's matrix-tree theorem (`ch08:510`); the sequential-adsorption limit 0.5472 (`ch12:892`); consistency of the pseudolikelihood estimator (`ch12:1038`); the SPDE result, the unbounded variance of the CAR field, and the Kullback-Leibler optimality of Vecchia's model (`ch13:528,751,769`).

### Appendix B: Scientific Python Reference Patterns

Patterns the chapters point to or reuse (chapters in parentheses):

- **B.1 SymPy:** `sp.symbols`, `sp.Matrix`, symbolic derivation checks (Chapters 0, 4, 5).
- **B.2 Stable solves:** `scipy.linalg.cho_factor` and `cho_solve` (Chapters 5, 13); `solve_triangular` (5, 8); `np.linalg.solve` (11 chapters); `np.linalg.cholesky` (12 chapters); `slogdet` for log-determinants (5, 7, 9, 12, 13); `eigh` and `eigvalsh` (1, 2, 5, 6, 7, 8, 9, 13); `lstsq` (4, 5, 7); `np.einsum` (2, 3, 13); the explicit `np.linalg.inv` exceptions (see F24).
- **B.3 Sparse matrices:** CSC and CSR storage ("compressed sparse column ... Appendix B.3", `ch08:679`, §8.4.3); `sparse.diags`; `splu` (8, 13); no sparse Cholesky in SciPy; `scipy.sparse.csgraph.connected_components` (9); conjugate gradient (Chapter 13 Exercise 7 only describes it).
- **B.4 Optimization:** `scipy.optimize.minimize` with `L-BFGS-B` over log parameters (5, 9) and `Nelder-Mead` (3, 9, 11, 13); `brentq` (2, 8); `least_squares` (12); profile likelihood patterns; numerical Hessians (Chapter 7).
- **B.5 Special functions:** `scipy.special.kv` and `gamma` for the Matérn family (2, 13); `special.xlogy` for empty windows (11); `scipy.stats` distributions (`t.ppf`, `chi2`, `pearsonr`).
- **B.6 Plotting:** `matplotlib.use("Agg")`, `imshow` (11 chapters), `contour` (5, 9), `scatter` (6, 9, 11, 12, 13), `savefig` to `figures/chNN/`, `tight_layout`.
- **Recurring, not in the outline:** seeding with `np.random.default_rng` and passing `rng` (all chapters); the vectorized distance matrix with `np.fill_diagonal` (10 chapters; Chapter 1 §1.6.1 explains the rounding issue); one-thread BLAS (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; Chapter 7's code took 4 s with one thread and over 80 s under load, and `scripts/run_chapter_code.py` sets it); timing with `time.perf_counter` (8, 13); `scipy.spatial` `cKDTree` and `cdist` (13 only); `np.kron` (13).

Helper inconsistencies to reconcile or document (function: chapters: difference):

- `dist`: Chapter 0 (no diagonal fix: `sqrt(max(...))` only, so the diagonal is about $10^{-8}$); Chapters 1, 2, 3, 5, 6, 13 (quadratic expansion with `fill_diagonal`); Chapter 4 (two-argument version, `dist(S, T=None)`); Chapter 10 (two-argument broadcasting version with a different algorithm); Chapter 11 (one-line version of the quadratic expansion).
- `gamma_exp`: Chapter 2 returns 0 at $h=0$; Chapter 3 is the formula for $h>0$ only.
- `kernel_intensity`: Chapter 10 `(P, psi, pts, edge=True, leave_out=False)` with edge correction; Chapter 11 `(P_data, grid, psi)` without edge correction, argument order differs.
- `sim_thomas`: Chapter 10 (fixed 4-sigma buffer, no buffer argument) versus Chapter 12 (`buffer=4.0` argument); same algorithm.
- `fit_bt`: Chapter 11 `(P_data, m, cov_fn, offset_fn, quad)` versus Chapter 12 `(X, ind, a, iters)`; different interfaces for the same Berman-Turner fit.
- Identical across chapters (no action): `rent_district`, `county_map`, `road_distance`, `pop_density`, `excess_risk`, `grid`.

### Appendix C: Jargon Rosetta Stone (plus where the notation concordance lives, see F9)

Terms the chapters actually define, to translate or cross-reference (chapter in parentheses):

- Geostatistics (2 to 5): nugget, sill, partial sill, range, practical range, semivariogram, isotropy, geometric and zonal anisotropy, stationarity (second-order and intrinsic), Matheron's estimator, Cressie-Hawkins estimator, lag tolerance, kriging (simple, ordinary, universal, block), BLUP, screening effect, change of support, universal kriging variance, REML, plug-in prediction, spatial confounding.
- Areal (6 to 9): modifiable areal unit problem, ecological fallacy, rook and queen contiguity, row-standardized weights, Moran's $I$, Geary's $C$, local indicators, permutation test, SAR (lag and error), feedback matrix and impacts (direct, indirect, total), LM tests, CAR (proper, intrinsic), Gaussian Markov random field, graph Laplacian, generalized determinant, Brook's lemma, SMR, funnel plot, overdispersion, quasi-Poisson, empirical Bayes, BYM, BYM2, Leroux, penalized quasi-likelihood, Henderson's equations, Laplace approximation.
- Point patterns (10 to 12): CSR, intensity, quadrats, kernel intensity, bandwidth, $G$, $F$, $J$, Ripley's $K$, $L$, pair correlation, edge corrections (guard area, Hanisch, Ripley, translation), envelopes, relative risk, random labelling, Berman-Turner quadrature, offset, scan statistic, minimum contrast, Neyman-Scott, Thomas, Matérn cluster, Cox and log-Gaussian Cox process, hard-core Types I and II, simple sequential inhibition, Gibbs and Strauss processes, pseudolikelihood.
- Large data (13): low rank, knot, predictive process, tapering and Wendland function, Vecchia, SPDE, separable covariance, Kronecker.
- Translations the chapters already state in prose and that Appendix C should collect: kriging = BLUP; nugget = independent noise variance; sill = $\sigma^2+\tau^2$; CAR = a precision matrix; ICAR = Laplacian precision; SAR = covariance through $(\mathbf{I}-\rho\mathbf{W})^{-1}$; Moran's $I$ = a ratio of quadratic forms; Berman-Turner = weighted Poisson regression; relative risk = logistic odds; scan statistic = maximized Poisson likelihood ratio; Strauss = Poisson GLM with a neighbor-count covariate; minimum contrast = nonlinear least squares on a summary function; kriging = Gaussian process regression = kernel ridge regression (`ch04` §4.5.3); LOO residual = conditional mean from the precision (`ch04` §4.7, `ch08` §8.1.2). Each already has an entry in OUTLINE Appendix C except the last three.

### Appendix D: Where to Go Next

Items the chapters defer, each with its origin:

- Spatial sampling design (`ch04:154`, §4.2.2).
- Alternatives to plug-in prediction: parametric bootstrap and Bayesian averaging over $\boldsymbol{\theta}$ (`ch05:689`, §5.4.5).
- Spatial confounding and causal inference (`ch05:899`, §5.6.2; also OUTLINE).
- The finite-element mesh for irregular locations, and the full SPDE approach, INLA and MCMC (`ch13:896,1253`).
- Nonseparable and multivariate space-time models (`ch13:1043`) and cokriging (OUTLINE only; no chapter pointer).
- Non-Gaussian geostatistics and trans-Gaussian kriging beyond the log scale (OUTLINE; `ch04` §4.5.4 gives only the simple-kriging correction).
- The deeper theory of covariance validity: Bochner's theorem and spectral characterizations (`ch00:355`, `ch02:295`).
- Covariance models on the sphere (`ch01` §1.6.4; OUTLINE).
- The Spatial Durbin model (`ch07:471`: "which we do not develop here"; no pointer to Appendix D, and OUTLINE Appendix D does not list it).
- Marked point patterns and mark correlation functions, space-time point processes (OUTLINE only; Chapter 10 covers multitype patterns).
- Geographically weighted regression and spatial machine learning (OUTLINE; `ch04` §4.5.3 touches Gaussian processes and kernel ridge regression).

Missing from the current OUTLINE Appendix D but promised by chapters: the finite-element form of the SPDE bridge, the Spatial Durbin model, alternatives to plug-in prediction, and the deeper theory of covariance validity.
