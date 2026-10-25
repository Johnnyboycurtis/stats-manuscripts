# Spatial Data Analysis: A Linear Models and Matrix Approach

## About This Book

### Audience and Prerequisites

Written for first-year Master's students and strong senior undergraduates in statistics, data science, and related fields. Readers should have completed:

* **Multivariate calculus** (gradients, Lagrange multipliers, Taylor expansions)
* **Linear algebra** (rank, eigenvalues, quadratic forms, positive definiteness)
* **Applied linear regression**, including matrix notation for the linear model
* **Undergraduate mathematical statistics** (likelihood, expectations, the multivariate normal distribution)

Chapter 0 reviews the five tools from these courses that the book relies on most. Generalized linear models are reviewed in Chapter 9, so prior GLM coursework is helpful but not required.

### Design Principles

1. **Mathematical anti-gatekeeping.** Spatial statistics carries terminology inherited from mining and geography (*kriging, nugget, sill, range, intrinsic autoregression*). This book keeps the terms, because students need them to read the literature and use software, and translates each one into the language of linear models and generalized linear models: GLS, BLUP, projection matrices, precision matrices, and IRLS.
2. **One model equation.** Geostatistical, areal, and count models are all written as $Y = \text{mean} + u + \varepsilon$, a linear mixed model with a spatial random effect $u$. Each new method is presented as a change to one of those three terms.
3. **A complete analysis cycle in every part.** Each part takes the reader through exploring, fitting, inferring, predicting or interpreting, and checking, so no data type ends at a hypothesis test.
4. **Step-by-step derivations and $n=4$ walkthroughs.** In the spirit of Dobson & Barnett and Faraway, no step is skipped with "it is trivial to show." Every major matrix result is paired with a tiny configuration of 3 to 5 locations, printed in full, so the reader can verify each operation by hand.
5. **Known-truth simulation instead of data wrangling.** In the spirit of Rizzo, Cholesky simulation is introduced in Chapter 2. Students generate synthetic fields with known parameters $(\boldsymbol{\beta}, \sigma^2, \tau^2, \phi)$ and use Monte Carlo experiments to see why naive methods fail and how spatial estimators recover the truth. Simulations also include the problems real data bring: duplicate locations, skewed responses, outliers, and intensity trends.
6. **The scientific Python stack.** Computation uses `NumPy`, `SciPy`, and `SymPy`, with `matplotlib` for maps, variograms, and envelopes, and no GIS or spatial packages. Symbolic derivations are verified in `SymPy`. Numerical linear algebra, optimization, and special functions are built from first principles with `NumPy` and `SciPy`.

### Scope

**Included:** geostatistics (variograms, kriging, change of support, likelihood inference and model selection), areal and lattice models (spatial weights, SAR, CAR, BYM, spatial GLMMs, and the modifiable areal unit problem), point patterns (intensity estimation, second-order summaries, inhomogeneous Poisson, cluster, and Gibbs models), and computational methods for large $n$.

**Deliberately excluded:** measure-theoretic probability, $\sigma$-algebras, spectral measures, finite-element methods, and infinite-dimensional functional analysis. A random field is treated through its finite-dimensional distributions, and covariance validity is checked through eigenvalues, Cholesky factors, and closure properties, with validity proofs for named families taken from the literature. Appendix D lists directions beyond the book.

## Notation

### Conventions

* Bold lowercase letters are vectors ($\mathbf{s}, \boldsymbol{\beta}, \boldsymbol{\lambda}$), bold uppercase letters are matrices ($\mathbf{X}, \boldsymbol{\Sigma}, \mathbf{Q}$), and italic letters are scalars.
* Uppercase denotes random quantities and lowercase their observed values: $\mathbf{Y}$ is the random response vector and $\mathbf{y}$ the data, as in Dobson & Barnett.
* The operator $\operatorname{E}[\cdot]$ is set in roman type; italic $E_i$ denotes expected counts in Chapter 9.
* Notation follows Schabenberger & Gotway (2005) where that does not cost clarity. Appendix C gives a symbol-by-symbol concordance.

### The Model Used Throughout

$$Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s}), \qquad \mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$$

Here $u(\mathbf{s})$ is the spatial random effect, with $\operatorname{Cov}(u(\mathbf{s}_i), u(\mathbf{s}_j)) = \sigma^2 r(h_{ij};\phi)$, and $\varepsilon(\mathbf{s})$ is the nugget term with variance $\tau^2$. The noise-free value $\tilde{Y}(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s})$ is the usual prediction target.

### Symbols

| Symbol | Dimension | Meaning |
| :--- | :--- | :--- |
| $\mathcal{D} \subset \mathbb{R}^2$ | Continuous | Bounded study domain, with area $\operatorname{area}(\mathcal{D})$ |
| $\mathbf{s},\ \mathbf{s}_0$ | $2 \times 1$ | A location, and a prediction location |
| $\mathbf{S}$ | $n \times 2$ | Coordinate matrix with rows $\mathbf{s}_i'$ |
| $\mathbf{h}_{ij},\ h_{ij}$ | $2 \times 1$, scalar | Lag vector $\mathbf{s}_i - \mathbf{s}_j$ and distance $h_{ij} = \lVert \mathbf{h}_{ij} \rVert$ |
| $\mathbf{H}$ | $n \times n$ | Matrix of pairwise distances $h_{ij}$ |
| $\mathbf{Y},\ \mathbf{y}$ | $n \times 1$ | Random response vector and its observed value |
| $\mathbf{X},\ \boldsymbol{\beta}$ | $n \times p$, $p \times 1$ | Design matrix and fixed effects |
| $u(\mathbf{s}),\ \mathbf{u}$ | Process, $n \times 1$ | Spatial random effect |
| $\varepsilon(\mathbf{s}),\ \boldsymbol{\varepsilon}$ | Process, $n \times 1$ | Nugget (measurement error plus micro-scale variation) |
| $\tilde{Y}(\mathbf{s}_0),\ \tilde{Y}(B)$ | Scalars | Noise-free target $\mu(\mathbf{s}_0) + u(\mathbf{s}_0)$, and its average over a block $B$ |
| $\boldsymbol{\Sigma},\ \boldsymbol{\theta}$ | $n \times n$, vector | Covariance matrix of $\mathbf{Y}$ and its parameters |
| $C(\mathbf{h}),\ \gamma(\mathbf{h})$ | Functions | Covariance function and semivariogram of $Y$ |
| $\sigma^2,\ \tau^2,\ \phi$ | Scalars | Partial sill (spatial variance scale), nugget, range parameter |
| $\delta$ | Scalar | Noise-to-signal ratio $\tau^2/\sigma^2$ |
| $\nu$ | Scalar | Matérn smoothness |
| $r(h;\phi),\ \mathbf{R}(\phi)$ | Function, $n \times n$ | Correlation function of $u$ and its matrix, $r_{ij} = r(h_{ij};\phi)$ |
| $\boldsymbol{\Psi}(\phi,\delta)$ | $n \times n$ | $\mathbf{R}(\phi) + \delta\mathbf{I}_n$, so that $\boldsymbol{\Sigma} = \sigma^2\boldsymbol{\Psi}$ |
| $\mathbf{G}$ | $2 \times 2$ | Positive-definite matrix for geometric anisotropy |
| $\lambda_0,\ \boldsymbol{\lambda}$ | Scalar, $n \times 1$ | Constant term and weights of a linear predictor $\lambda_0 + \boldsymbol{\lambda}'\mathbf{Y}$ |
| $m$ (or $\mathbf{m}$) | Scalar (or $p \times 1$) | Lagrange multiplier(s) in the kriging systems |
| $\mathbf{c}_0,\ \mathbf{c}_B$ | $n \times 1$ | $\operatorname{Cov}(\mathbf{Y}, \tilde{Y}(\mathbf{s}_0))$ and $\operatorname{Cov}(\mathbf{Y}, \tilde{Y}(B))$ |
| $\mathbf{P}_{\mathbf{X}},\ \mathbf{P}_c$ | $n \times n$ | OLS projection $\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ and centering matrix $\mathbf{I}_n - \tfrac{1}{n}\mathbf{1}\mathbf{1}'$ |
| $\boldsymbol{\Pi}$ | $n \times n$ | Residual precision $\boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}$ |
| $\mathbf{T}$ | $n \times (n-p)$ | REML error-contrast matrix with $\mathbf{T}'\mathbf{X} = \mathbf{0}$ and $\mathbf{T}\mathbf{T}' = \mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$ |
| $\mathbf{A},\ \mathbf{D}$ | $n \times n$ | Binary symmetric adjacency matrix and diagonal degree matrix, $d_i = \sum_j A_{ij}$ |
| $\mathbf{W},\ w_{ij},\ S_0$ | $n \times n$, scalars | Spatial weights matrix, its entries, and their sum $S_0 = \sum_i\sum_j w_{ij}$ |
| $\mathbf{y}_c,\ \mathbf{M}$ | $n \times 1$, $n \times n$ | Centered data $\mathbf{P}_c\mathbf{y}$ and the symmetrized matrix in Moran's $I$ |
| $\rho$ | Scalar | Autoregressive parameter in SAR, CAR, and Leroux models, and the BYM2 mixing parameter (scales differ between models) |
| $\mathbf{Q}$ | $n \times n$ | Precision matrix of a Gaussian Markov random field |
| $c_{ij},\ \sigma_i^2$ | Scalars | CAR conditional regression coefficients and conditional variances |
| $\mathbf{u}^{\text{icar}},\ \mathbf{v}$ | $n \times 1$ | Structured (ICAR) and unstructured (i.i.d.) components in BYM models |
| $\boldsymbol{\eta},\ \boldsymbol{\mu}$ | $n \times 1$ | GLM linear predictor and mean |
| $\mathbf{z},\ \mathbf{V}$ | $n \times 1$, $n \times n$ | IRLS working response and diagonal working weights (Dobson & Barnett write $\mathbf{W}$) |
| $E_i$ | Scalar | Expected count for region $i$ |
| $N(B),\ \lambda(\mathbf{s})$ | Count, function | Number of events in a set $B$, and point process intensity |
| $\hat{\lambda}(\mathbf{s}),\ \mathcal{K}_\psi,\ \psi$ | Function, kernel, scalar | Kernel intensity estimate, smoothing kernel, and bandwidth |
| $\lambda_1(\mathbf{s}),\ \lambda_0(\mathbf{s}),\ \operatorname{RR}(\mathbf{s})$ | Functions | Case and control intensities, and spatial relative risk $\lambda_1/\lambda_0$ |
| $\lambda_{\text{pop}}(\mathbf{s})$ | Function | Baseline population intensity used as an offset |
| $K_{12}(r),\ G_{12}(r)$ | Functions | Cross-type $K$ and $G$ functions |
| $n_w,\ e_w$ | Scalars | Observed and expected counts in scan window $w$ |
| $\kappa,\ \mu_c,\ \sigma_c$ | Scalars | Parent intensity, mean offspring per parent, and offspring spread in cluster processes |
| $r_0$ | Scalar | Interaction radius (hard core, Strauss, Matérn cluster disc) |
| $t(\mathbf{u}, \mathbf{S}),\ \zeta$ | Count, scalar | Number of points within $r_0$ of $\mathbf{u}$, and the Strauss interaction parameter |
| $b(\mathbf{s}, r)$ | Set | Disc of radius $r$ centered at $\mathbf{s}$ |
| $K(r),\ L(r),\ g(r)$ | Functions | Ripley's $K$, Besag's $L = \sqrt{K/\pi}$, and the pair correlation function |
| $K_{\text{inhom}}(r)$ | Function | Inhomogeneous $K$-function |
| $G(r),\ F(r),\ J(r)$ | Functions | Nearest-neighbor, empty-space, and $J$ functions |
| $\omega_{ij}$ | Scalars | Edge-correction weights |
| $\mathbf{s}_k^{*},\ a_k,\ \mathbb{1}_k,\ n_d$ | Locations, scalars | Quadrature points, their area weights, data-point indicators, and number of dummy points |
| $n_q$ | Scalar | Number of quadrats |
| $\mathbf{B},\ \boldsymbol{\alpha},\ \boldsymbol{\Sigma}_\alpha,\ q$ | $n \times q$, $q \times 1$, $q \times q$, scalar | Basis matrix, basis coefficients, their covariance, and number of knots |
| $\mathcal{C}_i$ | Set | Conditioning set of location $i$ in Vecchia's approximation |
| $\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}',\ \lambda_i$ | $n \times n$, scalars | Eigendecomposition and eigenvalues |
| $\mathbf{L}\mathbf{L}'$ | $n \times n$ | Cholesky factorization |
| $\mathbf{Z}$ | $n \times 1$ | Standard normal vector used in simulation |
| $\mathbf{a}$ | $n \times 1$ | Generic coefficient vector (e.g. in positive-definiteness tests) |

### Intentional Overloads

These follow strong conventions in the literature, and typography or context keeps them apart:

* $\lambda$: bold $\boldsymbol{\lambda}$ and $\lambda_0$ for kriging (Part II), scalar $\lambda_i$ for eigenvalues, and $\lambda(\mathbf{s})$ for intensity (Part IV).
* $L$, $G$: the functions $L(r)$ and $G(r)$ in Part IV versus the Cholesky factor $\mathbf{L}$ and anisotropy matrix $\mathbf{G}$.
* $g$: the GLM link function $g(\cdot)$ in Chapter 9 and the pair correlation function $g(r)$ in Part IV.
* $\sigma^2$: always the spatial variance scale. It is the partial sill in geostatistics and the innovation variance in SAR and CAR models, where it is not the marginal variance.
* $\boldsymbol{\mu}$: the mean, whether the mean function $\mu(\mathbf{s})$ or the GLM mean vector.

## Table of Contents

## Getting Started

### Chapter 0: The Big Picture and the Math You Already Know

* 0.1 What is spatial statistics?
    * 0.1.1 When rows of a data set are not independent: nearby observations carry overlapping information
    * 0.1.2 One realization: we observe one map of one world, so replication has to come from assumptions about how the process behaves across space (made precise as stationarity in Chapter 2)
    * 0.1.3 The three questions spatial data asks
        * Continuous space (geostatistics): what is the value where we did not measure?
        * Regions (lattice and areal data): why does this county differ from its neighbors?
        * Point events (point patterns): why do events cluster here and avoid each other there?
* 0.2 One equation for the whole book, in words: an observation is a large-scale trend, plus a spatially smooth random effect, plus independent noise, $Y = \mu + u + \varepsilon$
* 0.3 Three ways classical statistics misleads in space, shown with simulated maps rather than derivations
    * 0.3.1 Inflated sample size: a thousand sensors packed close together are not a thousand independent measurements (derived in Chapter 1)
    * 0.3.2 Spurious correlation: two independent but smooth surfaces often show a large sample correlation, and a naive test calls it significant (derived in Chapter 1)
    * 0.3.3 Small-number maps: regions with small populations dominate a map of raw rates even when the true risk is flat everywhere (developed in Chapter 9)
* 0.4 The math inventory: five tools from earlier courses that do most of the work in this book
    * 0.4.1 Linear transformations of random vectors: $\operatorname{E}[\mathbf{B}\mathbf{Y}] = \mathbf{B}\boldsymbol{\mu}$ and $\operatorname{Var}(\mathbf{B}\mathbf{Y}) = \mathbf{B}\boldsymbol{\Sigma}\mathbf{B}'$
    * 0.4.2 Quadratic forms $\mathbf{y}'\mathbf{M}\mathbf{y}$, and why positive semi-definiteness says that no variance can be negative: $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} = \operatorname{Var}(\mathbf{a}'\mathbf{Y}) \ge 0$
    * 0.4.3 The multivariate normal density read in two parts: a determinant term that measures spread, and a Mahalanobis distance that measures fit
    * 0.4.4 Maximum likelihood: differentiate the log-likelihood and set the score to zero
    * 0.4.5 The OLS normal equations $\mathbf{X}'\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\mathbf{y}$
    * 0.4.6 What the book builds on top of these, as it needs them: Cholesky and eigen decompositions, block matrix inversion, Lagrange multipliers, and IRLS (collected in Appendix A)
* 0.5 How this book works
    * 0.5.1 What is left out, and why: measure theory, functional analysis, and black-box spatial software
    * 0.5.2 The rhythm of each chapter: a linear-model bridge, a full derivation, a hand-sized example with 3 to 5 locations, then code and a known-truth simulation
* 0.6 Setting up: a short environment check for `NumPy`, `SciPy`, `SymPy`, and `matplotlib`
* 0.7 *Warm-up exercises: one problem on each of the five tools, with pointers to Appendix A for any that feel unfamiliar*

---

## PART I: Matrix Foundations and Spatial Covariance

### Chapter 1: Spatial Dependence Through the Lens of Linear Models

* 1.1 Reviewing the classical linear model: $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}_n$
* 1.2 What is a spatial process? From i.i.d. scalars to correlated random vectors $\mathbf{Y} = [Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_n)]'$
* 1.3 Why we cannot ignore space: OLS when $\operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma}$
    * 1.3.1 Unbiasedness is preserved: $\operatorname{E}[\hat{\boldsymbol{\beta}}_{\text{OLS}}] = \boldsymbol{\beta}$
    * 1.3.2 The true variance of OLS: $\operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{OLS}}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}$
    * 1.3.3 Why positive spatial correlation typically understates standard errors and inflates Type I error, most severely when the covariates are themselves spatially smooth
    * 1.3.4 Loss of efficiency relative to the GLS variance $(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}$
* 1.4 Effective sample size: how redundant information reduces $n$ to $n_{\text{eff}}$
* 1.5 The three spatial data structures as matrix geometries
    * 1.5.1 Continuous geostatistical fields (dense distance matrices $\mathbf{H}$)
    * 1.5.2 Discrete areal lattices (sparse adjacency matrices $\mathbf{A}$)
    * 1.5.3 Spatial point patterns (random coordinate sets and discretized grid counts)
* 1.6 Coordinates and distance
    * 1.6.1 Vectorized distance matrices via quadratic expansion:

      $$h_{ij}^2 = (\mathbf{s}_i - \mathbf{s}_j)'(\mathbf{s}_i - \mathbf{s}_j) = \mathbf{s}_i'\mathbf{s}_i - 2\mathbf{s}_i'\mathbf{s}_j + \mathbf{s}_j'\mathbf{s}_j$$

    * 1.6.2 Why Euclidean distance on raw latitude and longitude is wrong, and simple map projections for small regions
    * 1.6.3 Great-circle (haversine) and chordal distance for large regions
    * 1.6.4 A warning carried into Chapter 2: some covariance functions are not valid on the sphere with great-circle distance (e.g. Matérn with $\nu > 1/2$), while chordal distance preserves validity
* 1.7 *Walkthrough and Lab 1: an $n=4$ paper-and-pencil variance distortion example; a 1,000-run Monte Carlo coverage experiment in NumPy, run once with i.i.d. covariates and once with a spatially smooth covariate; comparing Euclidean, projected, and great-circle distances for the same points*

### Chapter 2: The Spatial Linear Model, Covariance, and Simulation

* 2.1 One model equation: $Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})$
    * 2.1.1 The large-scale mean $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$
    * 2.1.2 The spatial random effect $u(\mathbf{s})$, with covariance $\sigma^2 r(h;\phi)$
    * 2.1.3 The nugget term $\varepsilon(\mathbf{s})$: measurement error plus variation finer than the sampling spacing (how this merges the micro-scale and measurement-error terms of the Cressie and Schabenberger-Gotway decomposition)
    * 2.1.4 Deriving the covariance matrix: $\boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n = \sigma^2\boldsymbol{\Psi}(\phi,\delta)$
    * 2.1.5 Reading the model as a linear mixed model
* 2.2 Stationarity expressed with moments
    * 2.2.1 Constant mean vector: $\operatorname{E}[\mathbf{Y}] = \mu\mathbf{1}_n$
    * 2.2.2 Second-order stationarity: $\operatorname{Cov}(Y(\mathbf{s}_i), Y(\mathbf{s}_j)) = C(\mathbf{h}_{ij})$
    * 2.2.3 Intrinsic stationarity: $\operatorname{Var}(Y(\mathbf{s}_i) - Y(\mathbf{s}_j)) = 2\gamma(\mathbf{h}_{ij})$
    * 2.2.4 The bridge between them: $\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$
* 2.3 Reading the variogram: decoding the jargon on the curve
    * 2.3.1 *Nugget* $\tau^2$: the jump of $\gamma$ just above the origin
    * 2.3.2 *Sill* $C(\mathbf{0}) = \sigma^2 + \tau^2$: the level $\gamma$ approaches at large distances, the total variance of $Y(\mathbf{s})$
    * 2.3.3 *Partial sill* $\sigma^2$: the rise from nugget to sill, the variance of $u(\mathbf{s})$
    * 2.3.4 *Range* $\phi$: a distance scale, not a decay rate (some texts write $\exp(-\phi h)$, the reciprocal convention), and the practical range, e.g. $3\phi$ for the exponential model
    * 2.3.5 Isotropy: when $C$ and $\gamma$ depend only on $h = \lVert\mathbf{h}\rVert$
* 2.4 What makes a covariance matrix valid?
    * 2.4.1 Positive semi-definiteness: $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} \ge 0$ for all $\mathbf{a} \in \mathbb{R}^n$
    * 2.4.2 Why naive functions fail: a $3 \times 3$ linear-decay counterexample with a negative eigenvalue
    * 2.4.3 Algebraic closure properties: sums, products, and Gramians $\mathbf{B}\mathbf{B}'$
    * 2.4.4 What finite checks can and cannot show: a negative eigenvalue proves a function invalid, but no set of eigenvalue checks proves it valid; validity of the named families is taken from the literature
* 2.5 Parametric correlation families $r(h;\phi)$
    * 2.5.1 Exponential and Gaussian models: smoothness versus numerical ill-conditioning (condition numbers, jitter)
    * 2.5.2 The spherical model: exact zeros and compact support
    * 2.5.3 The Matérn family: general form and the closed-form cases $\nu = 1/2,\ 3/2,\ 5/2$
* 2.6 Geometric anisotropy through coordinate transformations: $h_{ij}(\mathbf{G}) = \sqrt{(\mathbf{s}_i - \mathbf{s}_j)'\mathbf{G}(\mathbf{s}_i - \mathbf{s}_j)}$
* 2.7 Simulating spatial random fields from first principles
    * 2.7.1 Cholesky simulation: $\mathbf{Y} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$ with $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}'$
    * 2.7.2 Spectral simulation $\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ when $\boldsymbol{\Sigma}$ is nearly singular
    * 2.7.3 Simulating $u$ and $\varepsilon$ separately, so the noise-free surface is known
* 2.8 *Walkthrough and Lab 2: eigenvalues of valid and invalid $4 \times 4$ matrices; simulating synthetic spatial worlds with known parameters in NumPy and SciPy; plotting simulated surfaces with matplotlib*

---

## PART II: Geostatistical Models and Generalized Least Squares

### Chapter 3: Empirical Variography and Weighted Least Squares Fitting

* 3.1 Estimating the empirical semivariogram
    * 3.1.1 Matheron's method-of-moments estimator: $\hat{\gamma}(h_k) = \dfrac{1}{2\lvert N(h_k)\rvert}\sum_{(i,j) \in N(h_k)} (Y_i - Y_j)^2$, where $N(h_k)$ is the set of pairs in lag bin $k$
    * 3.1.2 Vectorized computation with the pairwise difference matrix $\mathbf{Y}\mathbf{1}' - \mathbf{1}\mathbf{Y}'$
    * 3.1.3 Distance binning, lag tolerances, and why pairs beyond about half the domain diameter are unreliable
    * 3.1.4 Directional variograms: angular tolerances, the rose diagram, detecting geometric anisotropy, and a note on zonal anisotropy (sills that differ by direction)
* 3.2 Robust estimation: the Cressie-Hawkins fourth-root estimator, and why outliers inflate Matheron's estimator
* 3.3 Fitting parametric curves to empirical variograms
    * 3.3.1 Why ordinary nonlinear least squares is suboptimal: bins are correlated and heteroscedastic
    * 3.3.2 The approximation $\operatorname{Var}(\hat{\gamma}(h_k)) \approx \dfrac{2[\gamma(h_k)]^2}{\lvert N(h_k)\rvert}$
    * 3.3.3 Cressie's weighted least squares objective and iterative fitting
* 3.4 The trend-contamination pitfall: how an unmodeled linear trend $\mathbf{X}\boldsymbol{\beta}$ produces a variogram that grows like $h^2$; detrending with OLS residuals $\mathbf{e} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{Y}$ and why that induces bias (resolved by GLS and REML in Chapter 5)
* 3.5 *Walkthrough and Lab 3: hand-computing a binned $\hat{\gamma}(h)$ on a 5-point configuration; fitting exponential and Matérn variograms by WLS in SciPy; directional variograms of an anisotropic simulated field; a contaminated simulation comparing the Matheron and Cressie-Hawkins estimators*

### Chapter 4: Spatial Prediction: Kriging as Linear Model BLUP

* 4.1 Demystifying "kriging": it is best linear unbiased prediction
    * 4.1.1 The prediction target: the noise-free value $\tilde{Y}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + u(\mathbf{s}_0)$, with $\mathbf{c}_0 = \operatorname{Cov}(\mathbf{Y}, \tilde{Y}(\mathbf{s}_0))$
    * 4.1.2 Linear predictors $\hat{Y}(\mathbf{s}_0) = \lambda_0 + \boldsymbol{\lambda}'\mathbf{Y}$, and why mean squared prediction error is variance plus squared bias
    * 4.1.3 For unbiased predictors, $\operatorname{E}[\hat{Y}(\mathbf{s}_0)] = \operatorname{E}[\tilde{Y}(\mathbf{s}_0)]$, the MSPE is a quadratic form:

      $$\operatorname{MSPE}(\boldsymbol{\lambda}) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2$$

* 4.2 Simple kriging (known mean $\mu$)
    * 4.2.1 Choosing $\lambda_0 = \mu(1 - \boldsymbol{\lambda}'\mathbf{1}_n)$ for unbiasedness, then setting $\partial\operatorname{MSPE}/\partial\boldsymbol{\lambda} = \mathbf{0}$ to obtain $\boldsymbol{\Sigma}\boldsymbol{\lambda} = \mathbf{c}_0$
    * 4.2.2 The predictor $\hat{Y}_{\text{SK}}(\mathbf{s}_0) = \mu + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mu\mathbf{1}_n)$ and variance $\sigma^2_{\text{SK}}(\mathbf{s}_0) = \sigma^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$, which depends on the locations but not on $\mathbf{y}$ (the basis of sampling design, Appendix D)
    * 4.2.3 Two geometric pictures: kriging as projection of $\tilde{Y}(\mathbf{s}_0)$ onto the span of the data under the covariance inner product, and GLS fitted values as an orthogonal projection of $\mathbf{Y}$ under the $\boldsymbol{\Sigma}^{-1}$ inner product
* 4.3 Ordinary kriging (unknown constant mean $\mu$)
    * 4.3.1 With $\lambda_0 = 0$, unbiasedness $\operatorname{E}[\boldsymbol{\lambda}'\mathbf{Y}] = \mu \iff \boldsymbol{\lambda}'\mathbf{1}_n = 1$, enforced by a Lagrange multiplier $m$
    * 4.3.2 The block-matrix system:

      $$\begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ m \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}$$

    * 4.3.3 Analytic block inversion: how $m$ adds the cost of estimating $\hat{\mu}_{\text{GLS}}$ to the prediction variance
* 4.4 Universal kriging (spatial regression with covariates $\mathbf{X}\boldsymbol{\beta}$)
    * 4.4.1 Unbiasedness constraints $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{x}_0$ and the multiplier vector $\mathbf{m} \in \mathbb{R}^p$
    * 4.4.2 The partitioned universal kriging system
    * 4.4.3 The two-stage decomposition (GLS trend plus simple kriging of residuals):

      $$\hat{Y}_{\text{UK}}(\mathbf{s}_0) = \mathbf{x}_0'\hat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})$$

    * 4.4.4 Universal kriging variance: interpolation uncertainty plus $\hat{\boldsymbol{\beta}}_{\text{GLS}}$ uncertainty
* 4.5 Two targets, and interpolation versus smoothing
    * 4.5.1 Predicting $\tilde{Y}$: $\hat{Y}(\mathbf{s}_i) = y_i$ at every data location exactly when $\tau^2 = 0$; for $\tau^2 > 0$ kriging smooths
    * 4.5.2 Predicting a new observation $Y(\mathbf{s}_0)$: the same predictor, with $\tau^2$ added to the prediction variance
    * 4.5.3 Connection to Gaussian process regression and kernel ridge regression
    * 4.5.4 Skewed responses: kriging $\log Y$ and the bias of back-transforming the prediction
* 4.6 Change of support: block kriging
    * 4.6.1 Predicting an areal average $\tilde{Y}(B) = \dfrac{1}{\operatorname{area}(B)}\displaystyle\int_B \tilde{Y}(\mathbf{s})\,d\mathbf{s}$, a linear functional of the same process
    * 4.6.2 The only change from point kriging: $\mathbf{c}_0$ becomes the averaged covariance vector $\mathbf{c}_B$, computed by averaging over a grid of points in $B$
    * 4.6.3 Why block predictions have smaller variance than point predictions
* 4.7 Evaluating predictions: leave-one-out cross-validation from one matrix computation

  $$Y_i - \hat{Y}_{-i} = \frac{(\boldsymbol{\Pi}\mathbf{Y})_i}{\Pi_{ii}}, \qquad \boldsymbol{\Pi}\mathbf{Y} = \boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})$$

  The spatial analogue of the regression shortcut $e_i/(1 - [\mathbf{P}_{\mathbf{X}}]_{ii})$. With a known mean and no covariates, $\boldsymbol{\Pi}$ reduces to $\boldsymbol{\Sigma}^{-1}$.
* 4.8 *Walkthrough and Lab 4: symbolic Lagrange-multiplier derivation in SymPy; solving an $n=4$ kriging problem by hand; surface prediction, block kriging, and LOOCV in NumPy; kriging a simulated lognormal field on the log scale*

### Chapter 5: Likelihood Inference, Model Selection, and Diagnostics

* 5.1 The multivariate normal log-likelihood for $\mathbf{Y} \sim N(\mathbf{X}\boldsymbol{\beta}, \boldsymbol{\Sigma}(\boldsymbol{\theta}))$:

  $$\ell(\boldsymbol{\beta}, \boldsymbol{\theta}) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\log\lvert\boldsymbol{\Sigma}(\boldsymbol{\theta})\rvert - \frac{1}{2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}(\boldsymbol{\theta})^{-1}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

* 5.2 Profile maximum likelihood
    * 5.2.1 Factoring $\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \sigma^2\boldsymbol{\Psi}(\phi,\delta)$ with $\delta = \tau^2/\sigma^2$
    * 5.2.2 Closed-form GLS substitution for $\hat{\boldsymbol{\beta}}(\phi,\delta)$ and $\hat{\sigma}^2(\phi,\delta)$, reducing the search to two dimensions
* 5.3 Restricted maximum likelihood (REML)
    * 5.3.1 Why ML underestimates variance parameters in small samples
    * 5.3.2 Filtering out $\mathbf{X}\boldsymbol{\beta}$ with the contrast matrix $\mathbf{T}$, where $\mathbf{T}'\mathbf{X} = \mathbf{0}$, and the identity $\boldsymbol{\Pi} = \mathbf{T}(\mathbf{T}'\boldsymbol{\Sigma}\mathbf{T})^{-1}\mathbf{T}'$
    * 5.3.3 The REML log-likelihood, unique up to an additive constant that does not depend on the choice of $\mathbf{T}$:

      $$\ell_{\text{REML}}(\boldsymbol{\theta}) = -\frac{n-p}{2}\log(2\pi) - \frac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert - \frac{1}{2}\log\lvert\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\rvert - \frac{1}{2}\mathbf{y}'\boldsymbol{\Pi}\mathbf{y}$$

* 5.4 Numerical optimization and asymptotic standard errors
    * 5.4.1 Evaluating $\log\lvert\boldsymbol{\Sigma}\rvert$ and $\boldsymbol{\Sigma}^{-1}\mathbf{e}$ safely with Cholesky, without forming an inverse
    * 5.4.2 Score equations and expected information for $\boldsymbol{\theta} = [\sigma^2, \tau^2, \phi]'$:

      $$\mathcal{I}_{\text{ML}}(\boldsymbol{\theta})_{jk} = \tfrac{1}{2}\operatorname{tr}\!\left(\boldsymbol{\Sigma}^{-1}\frac{\partial\boldsymbol{\Sigma}}{\partial\theta_j}\boldsymbol{\Sigma}^{-1}\frac{\partial\boldsymbol{\Sigma}}{\partial\theta_k}\right), \qquad \mathcal{I}_{\text{REML}}(\boldsymbol{\theta})_{jk} = \tfrac{1}{2}\operatorname{tr}\!\left(\boldsymbol{\Pi}\frac{\partial\boldsymbol{\Sigma}}{\partial\theta_j}\boldsymbol{\Pi}\frac{\partial\boldsymbol{\Sigma}}{\partial\theta_k}\right)$$

    * 5.4.3 Fisher scoring and standard errors from the diagonal of $[\mathcal{I}(\hat{\boldsymbol{\theta}})]^{-1}$; Newton-Raphson with the observed Hessian for comparison
    * 5.4.4 Practical identifiability: why $\sigma^2$ and $\phi$ are strongly correlated under fixed-domain sampling, and why only a combination of them is well determined
    * 5.4.5 Plug-in prediction: kriging variances ignore uncertainty in $\hat{\boldsymbol{\theta}}$
* 5.5 Model selection and tests for spatial structure
    * 5.5.1 Comparing covariance families with AIC and BIC, including a profile over Matérn smoothness $\nu$, and why REML likelihoods are comparable only for the same $\mathbf{X}$
    * 5.5.2 Testing for spatial structure ($\sigma^2 = 0$) or a nugget ($\tau^2 = 0$): the parameter lies on the boundary, so the likelihood ratio statistic follows $\tfrac{1}{2}\chi^2_0 + \tfrac{1}{2}\chi^2_1$ rather than $\chi^2_1$
    * 5.5.3 Checking boundary results by parametric bootstrap
* 5.6 Spatial regression diagnostics and confounding
    * 5.6.1 Decorrelated residuals $\mathbf{r}^* = \mathbf{L}^{-1}(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})$
    * 5.6.2 Spatial confounding: how smooth covariates in $\mathbf{X}$ align with the leading eigenvectors of $\boldsymbol{\Sigma}$, and why $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ and $\hat{\boldsymbol{\beta}}_{\text{GLS}}$ can differ
    * 5.6.3 Duplicate and near-duplicate locations: why $\boldsymbol{\Sigma}$ becomes singular when $\tau^2 = 0$, and what the nugget does about it
* 5.7 *Walkthrough and Lab 5: deriving the information traces in SymPy; a Cholesky-based REML optimizer and Fisher scoring routine in SciPy; a boundary likelihood ratio test checked by simulation; a simulated dataset with duplicate locations*

---

## PART III: Discrete Spatial Models (Lattice and Areal Data)

### Chapter 6: Areal Units, Spatial Weights, and Quadratic Form Tests

* 6.1 Areal units are a choice: the modifiable areal unit problem
    * 6.1.1 The scale effect and the zoning effect: how aggregating the same data differently changes correlations, regression coefficients, and Moran's $I$, sometimes reversing their sign
    * 6.1.2 The ecological fallacy: area-level relationships are not individual-level relationships
    * 6.1.3 What this means for every model in Part III
* 6.2 Building a spatial weights matrix
    * 6.2.1 Contiguity: rook and queen adjacency
    * 6.2.2 Distance bands, inverse-distance weights, and $k$-nearest-neighbor weights
    * 6.2.3 Why $k$-nearest-neighbor weights are asymmetric, and which results that breaks (the similarity argument in 6.3.3 and the CAR symmetry condition in 8.1.3)
    * 6.2.4 Sensitivity of conclusions to the choice of $\mathbf{W}$
* 6.3 Representing regions as graphs
    * 6.3.1 The symmetric binary adjacency matrix $\mathbf{A}$ and degree matrix $\mathbf{D} = \operatorname{diag}(\mathbf{A}\mathbf{1}_n)$
    * 6.3.2 Row-standardized weights $\mathbf{W} = \mathbf{D}^{-1}\mathbf{A}$, with $w_{ij} = A_{ij}/d_i$: why $\mathbf{W}\mathbf{1}_n = \mathbf{1}_n$ and why $\mathbf{W}$ is not symmetric
    * 6.3.3 A similarity transformation: $\mathbf{W}$ has the same eigenvalues as the symmetric matrix $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, so all of them are real
* 6.4 Moran's $I$ as a ratio of quadratic forms
    * 6.4.1 The centering matrix $\mathbf{P}_c$ and the centered data $\mathbf{y}_c = \mathbf{P}_c\mathbf{y}$
    * 6.4.2 The symmetrized central matrix $\mathbf{M} = \tfrac{1}{2}\mathbf{P}_c(\mathbf{W} + \mathbf{W}')\mathbf{P}_c$
    * 6.4.3 The ratio form, with $S_0 = \sum_i\sum_j w_{ij}$:

      $$I = \frac{n}{S_0}\,\frac{\mathbf{y}_c'\mathbf{W}\mathbf{y}_c}{\mathbf{y}_c'\mathbf{y}_c} = \frac{n}{S_0}\,\frac{\mathbf{Y}'\mathbf{M}\mathbf{Y}}{\mathbf{Y}'\mathbf{P}_c\mathbf{Y}}$$

* 6.5 Exact null moments of $I$ under normality, $\mathbf{Y} \sim N(\mu\mathbf{1}_n, \sigma^2\mathbf{I}_n)$
    * 6.5.1 Independence of $I$ and its denominator $\mathbf{Y}'\mathbf{P}_c\mathbf{Y}$ (the Pitman-Koopmans argument), and why $\mathbf{M}\mathbf{1}_n = \mathbf{0}$ removes $\mu$
    * 6.5.2 Deriving $\operatorname{E}[I] = -\dfrac{1}{n-1}$ from $\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \sigma^2\operatorname{tr}(\mathbf{M})$
    * 6.5.3 Deriving $\operatorname{Var}(I)$ from the second moments $\operatorname{E}[(\mathbf{Y}'\mathbf{M}\mathbf{Y})^2]$ and $\operatorname{E}[(\mathbf{Y}'\mathbf{P}_c\mathbf{Y})^2]$, using $\operatorname{Var}(\mathbf{Y}'\mathbf{M}\mathbf{Y}) = 2\sigma^4\operatorname{tr}(\mathbf{M}^2)$ for symmetric $\mathbf{M}$
* 6.6 Moran's $I$ on regression residuals: the same derivation with $\mathbf{P}_c$ replaced by $\mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$, giving

  $$\operatorname{E}[I] = \frac{n}{S_0}\,\frac{\operatorname{tr}\!\left((\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{W}\right)}{n - p}$$

* 6.7 Geary's $C$ as a graph-Laplacian quadratic form; local indicators of spatial association (LISA) as region-level contributions
* 6.8 Distribution-free inference: Monte Carlo permutation tests by shuffling vector indices, and why permuting residuals needs care
* 6.9 *Walkthrough and Lab 6: hand-computing $\mathbf{A}, \mathbf{D}, \mathbf{W}, \mathbf{M}$ and the exact moments of $I$ on a 4-region map; analytical z-tests versus permutation distributions in NumPy; a MAUP simulation that aggregates one fine-grid field two ways and tracks $\hat{\boldsymbol{\beta}}$ and $I$*

### Chapter 7: Simultaneous Autoregressive Models: Spatial Regression with Feedback

* 7.1 The simultaneous idea: each region's value is regressed on its neighbors' values, all at once
* 7.2 The spatial error model
    * 7.2.1 The model: $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$, $\mathbf{u} = \rho\mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$
    * 7.2.2 The reduced form $\mathbf{u} = (\mathbf{I}_n - \rho\mathbf{W})^{-1}\boldsymbol{\varepsilon}$
    * 7.2.3 The induced covariance $\operatorname{Var}(\mathbf{u}) = \sigma^2\left[(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})\right]^{-1}$ and precision $\mathbf{Q}_{\text{SAR}} = \sigma^{-2}(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})$
    * 7.2.4 GLS for $\boldsymbol{\beta}$ given $\rho$: the error model is Chapter 5 with a new $\boldsymbol{\Sigma}$
* 7.3 The spatial lag model
    * 7.3.1 The model $\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ and its reduced form $\mathbf{Y} = (\mathbf{I}_n - \rho\mathbf{W})^{-1}(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon})$
    * 7.3.2 The feedback multiplier $(\mathbf{I}_n - \rho\mathbf{W})^{-1} = \mathbf{I}_n + \rho\mathbf{W} + \rho^2\mathbf{W}^2 + \cdots$: effects travel to neighbors, neighbors of neighbors, and back
    * 7.3.3 Why OLS is inconsistent: $\mathbf{W}\mathbf{Y}$ is correlated with $\boldsymbol{\varepsilon}$
* 7.4 The parameter space: $\mathbf{I}_n - \rho\mathbf{W}$ is singular only at $\rho = 1/\lambda_i$; the conventional space is the interval containing 0, $\rho \in (\lambda_{\min}^{-1}, \lambda_{\max}^{-1})$
* 7.5 Maximum likelihood with Ord's eigenvalue determinant identity
    * 7.5.1 The Jacobian term $\log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert$ in both likelihoods
    * 7.5.2 Computing it once from the eigenvalues $\lambda_i$ of $\mathbf{W}$:

      $$\log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert = \sum_{i=1}^n\log(1 - \rho\lambda_i)$$

    * 7.5.3 Profiling out $\boldsymbol{\beta}$ and $\sigma^2$ to a one-dimensional search over $\rho$
* 7.6 Choosing between the error and lag models
    * 7.6.1 Lagrange multiplier tests (LM-error, LM-lag, and their robust versions) computed from OLS residuals
    * 7.6.2 Likelihood ratio tests between nested specifications
* 7.7 Interpreting the lag model: direct, indirect, and total impacts
    * 7.7.1 Why $\beta_k$ is not the marginal effect: the effect matrix $(\mathbf{I}_n - \rho\mathbf{W})^{-1}\beta_k$
    * 7.7.2 Direct impact (the average diagonal), total impact (the average row sum), and indirect impact (the difference)
* 7.8 *Walkthrough and Lab 7: expanding the feedback multiplier by hand on a 4-region map; eigenvalue-based likelihood optimization for the error and lag models in NumPy; LM tests and impact estimates on a simulated lag model*

### Chapter 8: Conditional Autoregressive Models and Sparse Precision Matrices

* 8.1 The conditional idea: specify each region given its neighbors, and let the joint distribution follow
    * 8.1.1 Full conditionals: $u_i \mid \mathbf{u}_{-i} \sim N\!\left(\sum_{j \ne i} c_{ij}u_j,\ \sigma_i^2\right)$
    * 8.1.2 From conditionals to the joint density through the precision matrix:

      $$\operatorname{E}[u_i \mid \mathbf{u}_{-i}] = -\sum_{j \ne i}\frac{Q_{ij}}{Q_{ii}}u_j, \qquad \operatorname{Var}(u_i \mid \mathbf{u}_{-i}) = \frac{1}{Q_{ii}}$$

    * 8.1.3 The symmetry requirement $c_{ij}\sigma_j^2 = c_{ji}\sigma_i^2$
    * 8.1.4 Gaussian Markov random fields: $Q_{ij} = 0$ exactly when $u_i$ and $u_j$ are conditionally independent given the rest
* 8.2 The proper CAR model
    * 8.2.1 The precision $\mathbf{Q}_{\text{CAR}} = \sigma^{-2}(\mathbf{D} - \rho\mathbf{A})$ and conditional variances $\sigma^2/d_i$
    * 8.2.2 Validity for $\rho$ between the reciprocals of the extreme eigenvalues of $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$
    * 8.2.3 Why $\sigma^2$ is not the marginal variance, and why marginal variances differ by region
* 8.3 The intrinsic CAR (ICAR) model
    * 8.3.1 Setting $\rho = 1$: the graph-Laplacian precision $\mathbf{Q}_{\text{ICAR}} = \sigma^{-2}(\mathbf{D} - \mathbf{A})$
    * 8.3.2 Why the quadratic form is a sum of squared neighbor differences:

      $$\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \tfrac{1}{2}\sum_{i=1}^n\sum_{j=1}^n A_{ij}(u_i - u_j)^2$$

    * 8.3.3 Rank deficiency: $(\mathbf{D} - \mathbf{A})\mathbf{1}_n = \mathbf{0}$, with one zero eigenvalue per connected component
    * 8.3.4 Identifiability through one sum-to-zero constraint per connected component, and the generalized (pseudo) determinant
* 8.4 Computing with precision matrices
    * 8.4.1 The CAR log-determinant from eigenvalues $\tilde{\lambda}_i$ of $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$:

      $$\log\lvert\mathbf{D} - \rho\mathbf{A}\rvert = \sum_{i=1}^n\log d_i + \sum_{i=1}^n\log(1 - \rho\tilde{\lambda}_i)$$

    * 8.4.2 Simulating a GMRF from a Cholesky factor of $\mathbf{Q}$ rather than of $\boldsymbol{\Sigma}$
    * 8.4.3 Sparse storage and sparse solves
* 8.5 SAR versus CAR
    * 8.5.1 Simultaneous versus conditional specification, covariance versus precision
    * 8.5.2 Zeros in $\mathbf{Q}_{\text{CAR}}$ against the second-order neighbors created by $\mathbf{W}'\mathbf{W}$ in $\mathbf{Q}_{\text{SAR}}$
    * 8.5.3 Why $\rho$ is not comparable across the two models
* 8.6 *Walkthrough and Lab 8: deriving the joint precision from the conditionals on a 4-region map; comparing the sparsity patterns of $\mathbf{Q}_{\text{SAR}}$ and $\mathbf{Q}_{\text{CAR}}$ on a $3 \times 3$ grid; CAR likelihood optimization and GMRF simulation with `scipy.sparse`*

### Chapter 9: Generalized Linear Mixed Models for Spatial Count Data

* 9.1 Reviewing GLMs in the style of Dobson & Barnett: exponential family, link functions $g(\boldsymbol{\mu}) = \boldsymbol{\eta}$, score equations, and IRLS
* 9.2 Spatial Poisson regression for regional counts
    * 9.2.1 The model equation again, on the log scale with an offset: $Y_i \mid u_i \sim \text{Poisson}\!\left(E_i\exp(\mathbf{x}_i'\boldsymbol{\beta} + u_i)\right)$
    * 9.2.2 Why raw standardized mortality ratios $Y_i/E_i$ have unstable variance in small-population regions
* 9.3 Shrinkage and spatial random effects
    * 9.3.1 Empirical Bayes shrinkage (Clayton and Kaldor): the gamma-Poisson model and why small regions are pulled toward the overall rate
    * 9.3.2 A spatial CAR distribution for the random effect: $\mathbf{u} \sim N(\mathbf{0}, \mathbf{Q}^{-1})$
    * 9.3.3 The Besag-York-Mollié (BYM) model: $\mathbf{u} = \mathbf{u}^{\text{icar}} + \mathbf{v}$, an ICAR component plus an i.i.d. component, and why the two are only weakly identified
    * 9.3.4 BYM2: scaling the ICAR component so that the mixing parameter $\rho$ is the share of variance that is spatially structured
    * 9.3.5 The Leroux precision $\mathbf{Q}(\sigma^2,\rho) = \sigma^{-2}\left[\rho(\mathbf{D} - \mathbf{A}) + (1 - \rho)\mathbf{I}_n\right]$, the same idea in a single precision matrix
    * 9.3.6 Comparing BYM, BYM2, and Leroux
* 9.4 Fitting spatial GLMMs by penalized quasi-likelihood (a teaching device for penalized IRLS)
    * 9.4.1 The linearized working response, with the offset removed: $\mathbf{z}^{(t)} = \boldsymbol{\eta}^{(t)} - \log\mathbf{E} + \mathbf{V}^{-1}(\mathbf{y} - \boldsymbol{\mu}^{(t)})$ and working weights $\mathbf{V} = \operatorname{diag}(\boldsymbol{\mu}^{(t)})$
    * 9.4.2 The coupled mixed-model (Henderson) equations:

      $$\begin{bmatrix} \mathbf{X}'\mathbf{V}\mathbf{X} & \mathbf{X}'\mathbf{V} \\ \mathbf{V}\mathbf{X} & \mathbf{V} + \mathbf{Q} \end{bmatrix} \begin{bmatrix} \boldsymbol{\beta}^{(t+1)} \\ \mathbf{u}^{(t+1)} \end{bmatrix} = \begin{bmatrix} \mathbf{X}'\mathbf{V}\mathbf{z}^{(t)} \\ \mathbf{V}\mathbf{z}^{(t)} \end{bmatrix}$$

    * 9.4.3 Estimating $\sigma^2$ and $\rho$ by REML on the working vector $\mathbf{z}$
* 9.5 Laplace approximation of the marginal likelihood, using the Hessian $\mathbf{V} + \mathbf{Q}$ at the mode of $\mathbf{u}$
* 9.6 *Walkthrough and Lab 9: one hand-computed IRLS iteration on a 3-region Poisson map; a penalized IRLS plus Laplace approximation solver in NumPy and SciPy; empirical Bayes versus Leroux smoothing of simulated disease counts*

---

## PART IV: Spatial Point Patterns

### Chapter 10: Exploring Point Patterns: Intensity, Randomness, and Interaction

* 10.1 What changes in a point pattern: the coordinate matrix $\mathbf{S} \in \mathbb{R}^{n \times 2}$ is now the random outcome
* 10.2 The homogeneous Poisson process (complete spatial randomness, CSR)
    * 10.2.1 Deriving $N(B) \sim \text{Poisson}(\lambda\operatorname{area}(B))$ as the limit of Bernoulli counts on a refining pixel grid
    * 10.2.2 Conditional uniformity: given $N(\mathcal{D}) = n$, locations are i.i.d. uniform on $\mathcal{D}$
    * 10.2.3 Quadrat counts: testing $X^2 = \sum_{k=1}^{n_q} (n_k - \bar{n})^2/\bar{n}$ against $\chi^2_{n_q-1}$, and the index of dispersion $s_n^2/\bar{n}$
* 10.3 Simulating point patterns from first principles
    * 10.3.1 Homogeneous patterns: a Poisson count, then uniform coordinates
    * 10.3.2 Lewis-Shedler thinning: generate candidates at rate $\lambda_{\max}$ and keep $\mathbf{s}_i$ when $U_i < \lambda(\mathbf{s}_i)/\lambda_{\max}$
    * 10.3.3 Why simulation comes first: every test in Part IV compares the data with simulated patterns
* 10.4 First-order exploration: kernel intensity estimation
    * 10.4.1 The estimator $\hat{\lambda}(\mathbf{s}) = \sum_{i=1}^n \mathcal{K}_\psi(\mathbf{s} - \mathbf{s}_i)$, evaluated on a grid through a grid-to-point distance matrix, and its relation to kernel density estimation
    * 10.4.2 Edge correction: dividing by the kernel mass inside $\mathcal{D}$
    * 10.4.3 Bandwidth selection: mean squared error criteria and likelihood cross-validation
* 10.5 Distance methods with the distance matrix $\mathbf{H}$
    * 10.5.1 Event-to-event nearest-neighbor distances $d_i^{\text{nn}} = \min_{j \ne i} h_{ij}$ and their distribution $G(r)$
    * 10.5.2 Deriving $G(r) = 1 - \exp(-\lambda\pi r^2)$ under CSR from the void probability $\Pr(N(b(\mathbf{s}, r)) = 0)$
    * 10.5.3 Point-to-event empty-space distances $F(r)$, computed from grid-to-point distances
    * 10.5.4 The $J$-function $J(r) = \dfrac{1 - G(r)}{1 - F(r)}$: a ratio of survival functions that equals 1 under CSR, falls below 1 for clustering, and rises above 1 for regularity
* 10.6 Second-order summaries
    * 10.6.1 Ripley's $K$-function: $\lambda K(r) = \operatorname{E}[\text{number of extra events within distance } r \text{ of a typical event}]$, with $K(r) = \pi r^2$ under CSR
    * 10.6.2 The estimator with the indicator matrix $\mathbb{1}(\mathbf{H} \le r)$:

      $$\hat{K}(r) = \frac{\operatorname{area}(\mathcal{D})}{n(n-1)}\sum_{i=1}^n\sum_{j \ne i}\omega_{ij}\,\mathbb{1}(h_{ij} \le r)$$

    * 10.6.3 Besag's $L(r) = \sqrt{K(r)/\pi}$, plotted as $L(r) - r$ so that CSR is the horizontal line at zero
    * 10.6.4 The pair correlation function $g(r) = \dfrac{K'(r)}{2\pi r}$: a non-cumulative summary that is easier to read at each distance, estimated by kernel smoothing of pair distances
* 10.7 Multitype patterns
    * 10.7.1 Categorical marks: cases and controls, or two species
    * 10.7.2 Cross-type summaries $K_{12}(r)$ and $G_{12}(r)$: attraction versus segregation
    * 10.7.3 Spatial relative risk as a ratio of kernel estimates, $\widehat{\operatorname{RR}}(\mathbf{s}) = \hat{\lambda}_1(\mathbf{s})/\hat{\lambda}_0(\mathbf{s})$, and why both estimates should share one bandwidth
* 10.8 Boundary truncation (edge effects)
    * 10.8.1 The guard-area (minus-sampling) method
    * 10.8.2 Hanisch and Kaplan-Meier (reduced-sample) estimators for $G$ and $F$
    * 10.8.3 Geometric derivation of Ripley's isotropic weight $\omega_{ij} = \dfrac{2\pi h_{ij}}{\text{length of } \partial b(\mathbf{s}_i, h_{ij}) \cap \mathcal{D}} \ge 1$ on a rectangle
* 10.9 Monte Carlo envelopes: under CSR for $G$, $F$, $L$, and $g$, and under random labelling for multitype summaries
* 10.10 *Walkthrough and Lab 10: printing the coordinates, the full $4 \times 4$ distance matrix $\mathbf{H}$, the nearest-neighbor distances $d_i^{\text{nn}}$, and $\hat{G}(r)$ for an $n=4$ pattern; homogeneous and thinned simulation in NumPy; kernel intensity maps with several bandwidths; vectorized $\hat{G}$, $\hat{F}$, $\hat{K}$, and $\hat{g}$ with envelopes for simulated homogeneous, clustered, and inhibited patterns (the models behind these simulations are derived in Chapter 12); a kernel relative-risk map for simulated cases and controls*

### Chapter 11: Point Processes as Generalized Linear Models

* 11.1 Intensity as a regression function: $\lambda(\mathbf{s}) = \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$
* 11.2 Deriving the exact log-likelihood from a refining grid of Bernoulli trials:

  $$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n\mathbf{x}(\mathbf{s}_i)'\boldsymbol{\beta} - \int_{\mathcal{D}}\exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})\,d\mathbf{s}$$

* 11.3 The Berman-Turner quadrature device: point processes as Poisson GLMs
    * 11.3.1 Quadrature points: the $n$ data points plus $n_d$ dummy points $\mathbf{s}_k^{*}$, with area weights $a_k$ from pixels or Voronoi tiles
    * 11.3.2 Data-point indicators $\mathbb{1}_k$, pseudo-responses $y_k = \mathbb{1}_k/a_k$, and the approximation

      $$\ell(\boldsymbol{\beta}) \approx \sum_{k=1}^{n+n_d} a_k\left(y_k\log\lambda(\mathbf{s}_k^{*}) - \lambda(\mathbf{s}_k^{*})\right)$$

    * 11.3.3 Why this is a weighted Poisson regression with prior weights $a_k$, solved by the IRLS algorithm of Chapter 9
* 11.4 Offsets: modeling excess over a known baseline
    * 11.4.1 A population intensity $\lambda_{\text{pop}}(\mathbf{s})$ entering as an offset: $\log\lambda(\mathbf{s}) = \log\lambda_{\text{pop}}(\mathbf{s}) + \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$
    * 11.4.2 Reading $\boldsymbol{\beta}$ as relative excess risk, the point-pattern version of the offset $\log E_i$ in Chapter 9
* 11.5 Case-control data and the logistic regression bridge
    * 11.5.1 Pooling cases and controls into one coordinate matrix $\mathbf{S}$, with labels $Y_i \in \{0, 1\}$
    * 11.5.2 Deriving $\Pr(Y_i = 1 \mid \mathbf{s}_i) = \dfrac{\lambda_1(\mathbf{s}_i)}{\lambda_1(\mathbf{s}_i) + \lambda_0(\mathbf{s}_i)}$, so that

      $$\operatorname{logit}\Pr(Y_i = 1 \mid \mathbf{s}_i) = \log\operatorname{RR}(\mathbf{s}_i)$$

      and the unknown population density cancels
    * 11.5.3 Modeling $\log\operatorname{RR}(\mathbf{s}) = \beta_0 + \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ by binomial IRLS, with the intercept absorbing the case-to-control sampling ratio
* 11.6 Where is the excess? Spatial scan statistics
    * 11.6.1 Circular windows $w$ centered on a grid, over a range of radii, with observed counts $n_w$ and expected counts $e_w$ from the baseline model
    * 11.6.2 The Poisson log-likelihood ratio for a window:

      $$\operatorname{LLR}_w = \left[n_w\log\frac{n_w}{e_w} + (n - n_w)\log\frac{n - n_w}{n - e_w}\right]\mathbb{1}(n_w > e_w)$$

      evaluated for all windows at once in NumPy, with `scipy.special.xlogy` handling empty windows
    * 11.6.3 Inference for the most likely cluster: the maximum over many overlapping windows has no $\chi^2$ reference distribution, so its p-value comes from Monte Carlo simulation under the baseline
* 11.7 Separating trend from interaction
    * 11.7.1 Why a varying intensity makes the ordinary $K$-function show clustering when points do not interact: the point-pattern version of the variogram trend contamination in §3.4
    * 11.7.2 The inhomogeneous $K$-function, which reweights the indicator matrix elementwise by the outer product of inverse intensities:

      $$\hat{K}_{\text{inhom}}(r) = \frac{1}{\operatorname{area}(\mathcal{D})}\sum_{i=1}^n\sum_{j \ne i}\frac{\omega_{ij}\,\mathbb{1}(h_{ij} \le r)}{\hat{\lambda}(\mathbf{s}_i)\hat{\lambda}(\mathbf{s}_j)}$$

    * 11.7.3 Where $\hat{\lambda}$ comes from (a kernel estimate or the fitted GLM), and the bias of estimating intensity and interaction from the same data
* 11.8 Inference under clustering
    * 11.8.1 Why Poisson GLM standard errors for $\boldsymbol{\beta}$ are too small when points cluster: the Chapter 1 failure of OLS in a new setting
    * 11.8.2 Corrected standard errors from a sandwich estimator and from a parametric bootstrap
* 11.9 *Walkthrough and Lab 11: assembling the Berman-Turner design matrix, area weights $a_k$, and pseudo-responses by hand for $n = 3$ data points and $n_d = 4$ dummy points; fitting an inhomogeneous Poisson process by IRLS in NumPy; a case-control logistic fit compared with the kernel relative-risk map; a scan statistic with a Monte Carlo p-value; ordinary versus inhomogeneous $K$ on a trend-only pattern; a coverage experiment for $\hat{\boldsymbol{\beta}}$ under clustering*

### Chapter 12: Models for Clustering and Inhibition

* 12.1 Three explanations for a departure from CSR: heterogeneity (a trend in intensity), clustering (points attract), and inhibition (points repel)
* 12.2 Cluster processes
    * 12.2.1 Neyman-Scott processes: Poisson parents with intensity $\kappa$, and a Poisson number of offspring with mean $\mu_c$ around each parent
    * 12.2.2 The Thomas process (Gaussian offspring with spread $\sigma_c$) and the Matérn cluster process (offspring uniform on a disc of radius $r_0$)
    * 12.2.3 Simulation with a buffer around $\mathcal{D}$ (e.g. $4\sigma_c$) so that clusters whose parents fall outside the window are not lost
    * 12.2.4 Deriving the closed-form $K$-function of the Thomas process:

      $$K(r) = \pi r^2 + \frac{1}{\kappa}\left(1 - \exp\!\left(-\frac{r^2}{4\sigma_c^2}\right)\right)$$

    * 12.2.5 Fitting by minimum contrast: nonlinear least squares between $\hat{K}(r)^{1/4}$ (or $\hat{g}$) and the model, the same routine used to fit variograms in Chapter 3
    * 12.2.6 Inhomogeneous cluster models: a GLM for intensity plus a cluster model for interaction
* 12.3 The log-Gaussian Cox process, $\log\lambda(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + u(\mathbf{s})$: the model equation once more. On a pixel grid it combines the covariance models of Part II, the GMRF precisions of Part III, and the Poisson GLM of Chapter 11
* 12.4 Inhibition
    * 12.4.1 Matérn hard-core processes: Type I deletes every point with a neighbor closer than $r_0$; Type II gives each point a random time mark and deletes a point when an earlier point lies within $r_0$; why both cap the attainable intensity
    * 12.4.2 Simple sequential inhibition: adding points one at a time until the window saturates
    * 12.4.3 Gibbs processes and the conditional intensity. The Strauss model:

      $$\log\lambda(\mathbf{u} \mid \mathbf{S}) = \mathbf{x}(\mathbf{u})'\boldsymbol{\beta} + \zeta\, t(\mathbf{u}, \mathbf{S}), \qquad \zeta \le 0$$

      where $t(\mathbf{u}, \mathbf{S})$ counts points within $r_0$ of $\mathbf{u}$; $\zeta \le 0$ is required for a valid model, and $\zeta \to -\infty$ gives a hard core
    * 12.4.4 The matrix view: $t$ is a row sum of an indicator matrix (excluding the point itself at data points), so it enters the Berman-Turner design matrix as one more covariate column
    * 12.4.5 Maximum pseudolikelihood with the same IRLS; why GLM standard errors are not valid here and are replaced by a parametric bootstrap; choosing $r_0$ by profiling
* 12.5 Checking a fitted point process model
    * 12.5.1 Residuals: smoothed residual intensity fields $\hat{\lambda}(\mathbf{s}) - \lambda(\mathbf{s}; \hat{\boldsymbol{\theta}})$
    * 12.5.2 Goodness of fit with envelopes simulated from the fitted model
    * 12.5.3 Choosing between a trend-only, cluster, Cox, and Gibbs model
* 12.6 *Walkthrough and Lab 12: simulating Thomas, Matérn hard-core I and II, and sequential inhibition patterns in NumPy; fitting the Thomas model by minimum contrast; fitting a Strauss model by pseudolikelihood with the extra covariate column; fitted-model envelopes*

---

## PART V: Matrix Scalability for Large Spatial Systems

### Chapter 13: Matrix Approximations and Fast Algorithms for Large $n$

* 13.1 The $\mathcal{O}(n^3)$ Cholesky bottleneck and the memory cost of dense $n \times n$ matrices
* 13.2 Low-rank spatial models (fixed-rank kriging and basis expansions)
    * 13.2.1 A random effect built from $q \ll n$ knots: $\mathbf{u} = \mathbf{B}\boldsymbol{\alpha}$, so $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{B}\boldsymbol{\alpha} + \boldsymbol{\varepsilon}$, with $\operatorname{Var}(\boldsymbol{\alpha}) = \boldsymbol{\Sigma}_\alpha$ and $\operatorname{Var}(\boldsymbol{\varepsilon}) = \tau^2\mathbf{I}_n$
    * 13.2.2 Inverting $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'$ in $\mathcal{O}(nq^2)$ time with the Sherman-Morrison-Woodbury identity:

      $$\boldsymbol{\Sigma}^{-1} = \tau^{-2}\mathbf{I}_n - \tau^{-4}\mathbf{B}\left(\boldsymbol{\Sigma}_\alpha^{-1} + \tau^{-2}\mathbf{B}'\mathbf{B}\right)^{-1}\mathbf{B}'$$

    * 13.2.3 Fast log-determinants with the matrix determinant lemma:

      $$\log\lvert\tau^2\mathbf{I}_n + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'\rvert = n\log\tau^2 + \log\lvert\boldsymbol{\Sigma}_\alpha\rvert + \log\lvert\boldsymbol{\Sigma}_\alpha^{-1} + \tau^{-2}\mathbf{B}'\mathbf{B}\rvert$$

    * 13.2.4 The cost of low rank: small-scale variation is lost and predictions oversmooth
* 13.3 Covariance tapering: multiplying by a compactly supported correlation (§2.5.2) to make $\boldsymbol{\Sigma}$ sparse while keeping it valid
* 13.4 Sparse precision from nearest-neighbor conditioning (Vecchia's approximation)
    * 13.4.1 Factoring the density into ordered conditionals: $p(\mathbf{y}) = p(y_1)\prod_{i=2}^n p(y_i \mid \mathbf{y}_{\mathcal{C}_i})$
    * 13.4.2 How conditioning sets with $\lvert\mathcal{C}_i\rvert \le k$ give a factor with at most $k$ non-zeros per row
* 13.5 The bridge between Parts II and III: Matérn fields as Gaussian Markov random fields
    * 13.5.1 The result of Lindgren, Rue, and Lindström, stated without the finite-element derivation: a Matérn field is approximately a GMRF
    * 13.5.2 A grid example: a precision of the form $(a\mathbf{I} + \mathbf{D} - \mathbf{A})^2$, built from the ICAR graph Laplacian, behaves like a Matérn field with $\nu = 1$
* 13.6 Kronecker products for separable space-time and multivariate grids
    * 13.6.1 Separable covariance $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$, with $\boldsymbol{\Sigma}_T$ of size $n_T \times n_T$ and $\boldsymbol{\Sigma}_S$ of size $n_S \times n_S$
    * 13.6.2 Fast inversion $\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Sigma}_T^{-1} \otimes \boldsymbol{\Sigma}_S^{-1}$, determinants $\lvert\boldsymbol{\Sigma}\rvert = \lvert\boldsymbol{\Sigma}_T\rvert^{n_S}\lvert\boldsymbol{\Sigma}_S\rvert^{n_T}$, and Cholesky factors
* 13.7 *Walkthrough and Lab 13: an $n=4$, $q=2$ Woodbury check by hand; benchmarking exact Cholesky (up to $n=5{,}000$) against low-rank kriging, tapering, and Vecchia factors in NumPy and SciPy; comparing a grid GMRF with a Matérn field*

---

## Appendices

### Appendix A: Matrix Algebra and Multivariate Normal Calculus

* A.1 Partitioned (block) matrix inversion and the Schur complement
* A.2 Vector and matrix derivatives: gradients of linear and quadratic forms, $\dfrac{\partial\log\lvert\boldsymbol{\Sigma}\rvert}{\partial\theta} = \operatorname{tr}\!\left(\boldsymbol{\Sigma}^{-1}\dfrac{\partial\boldsymbol{\Sigma}}{\partial\theta}\right)$, and $\dfrac{\partial\boldsymbol{\Sigma}^{-1}}{\partial\theta} = -\boldsymbol{\Sigma}^{-1}\dfrac{\partial\boldsymbol{\Sigma}}{\partial\theta}\boldsymbol{\Sigma}^{-1}$
* A.3 Moments of quadratic forms in normal vectors: for symmetric $\mathbf{M}$ and $\mathbf{Y} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, $\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}$ and the corresponding variance
* A.4 Conditional distributions of partitioned multivariate normals and their precision-matrix forms
* A.5 Kronecker products and the $\operatorname{vec}$ operator

### Appendix B: Scientific Python Reference Patterns

* B.1 Symbolic verification of matrix identities and gradients with `sympy.matrices`
* B.2 Stable solvers: `scipy.linalg.cho_factor` and `cho_solve` instead of `numpy.linalg.inv`
* B.3 Sparse matrices (`scipy.sparse.csc_matrix`) and sparse solvers (`spsolve`, `splu`, conjugate gradient)
* B.4 Constrained optimization of profile and REML log-likelihoods with `scipy.optimize.minimize` (L-BFGS-B)
* B.5 The Matérn function with `scipy.special.kv`
* B.6 Plotting with `matplotlib`: point maps, kriged surfaces, variograms, and simulation envelopes

### Appendix C: A Jargon Rosetta Stone and Notation Concordance

**C.1 Terms.** Spatial terminology translated into linear-model terms:

* Kriging, simple, ordinary, and universal: BLUP with a known mean, an unknown constant mean, and a regression mean estimated by GLS
* Block kriging and change of support: BLUP of a linear functional (an average) of the process
* Nugget, partial sill, sill, range: nugget variance $\tau^2$, variance of the spatial random effect, total variance, correlation length scale
* Semivariogram: half the variance of a difference, as a function of the lag
* Stationarity and isotropy: constant mean and a covariance that depends only on the lag, and only on distance
* SAR and CAR: random effects specified through the covariance (SAR) or the precision matrix (CAR)
* ICAR: a precision matrix equal to the graph Laplacian, rank deficient by construction
* BYM: a structured plus an unstructured random effect
* Moran's $I$: a normalized quadratic form in the centered data or the residuals
* Impacts: averages of the effect matrix $(\mathbf{I}_n - \rho\mathbf{W})^{-1}\beta_k$
* CSR and Berman-Turner: the homogeneous Poisson process, and a weighted Poisson GLM
* Spatial relative risk: the odds in a logistic regression on case-control labels
* Scan statistic: the largest Poisson likelihood ratio over many candidate windows
* Strauss model: a Poisson GLM, fitted by pseudolikelihood, with a neighbor-count covariate
* Minimum contrast: least squares between an estimated and a model summary function

**C.2 Notation concordance with Schabenberger & Gotway (2005).**

| Schabenberger & Gotway | This book | Note |
| :--- | :--- | :--- |
| $Z(\mathbf{s})$ | $Y(\mathbf{s})$ | $Y$ keeps the regression convention; $\mathbf{Z}$ is the simulation vector here |
| $D \subset \mathbb{R}^d$ | $\mathcal{D} \subset \mathbb{R}^2$ | $\mathbf{D}$ is the degree matrix here |
| $\mu(\mathbf{s}) + W(\mathbf{s}) + \eta(\mathbf{s}) + \varepsilon(\mathbf{s})$ | $\mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})$ | $u$ is the smooth small-scale term; micro-scale variation is absorbed into the nugget $\varepsilon$ |
| $C(\mathbf{h})$, $\gamma(\mathbf{h})$, $\mathbf{h}$, $\lVert\mathbf{h}\rVert$ | Same | |
| $\boldsymbol{\Sigma}$ or $\mathbf{V}$ | $\boldsymbol{\Sigma}$ | $\mathbf{V}$ is reserved for IRLS weights |
| $\boldsymbol{\lambda}$ (kriging weights) | $\boldsymbol{\lambda}$ | |
| $\mathbf{X}$, $\boldsymbol{\beta}$, $\boldsymbol{\theta}$ | Same | |
| $\mathbf{W}$, $w_{ij}$, $S_0$ | Same | |

### Appendix D: Where to Go Next

* Spatial sampling design: choosing locations to minimize kriging variance, which does not depend on the data (§4.2.2)
* Point patterns with continuous marks and mark correlation functions
* Space-time point processes
* Cokriging and multivariate fields; general (non-separable) space-time covariance models
* Non-Gaussian geostatistics and trans-Gaussian kriging beyond the log scale
* Covariance models on the sphere for global data
* Bayesian computation (MCMC, INLA) and the full SPDE approach
* Spatial confounding and causal inference
* Geographically weighted regression and spatial machine learning

---

## How Each Chapter Reads

1. **Linear model bridge.** The chapter opens by connecting its spatial problem to a familiar linear-model or GLM equation, translating any domain jargon into standard statistical terms.
2. **Step-by-step matrix derivation.** The estimator, predictor, or test statistic is derived with clear matrix algebra and calculus, showing every intermediate step.
3. **A concrete $n=4$ example.** A miniature configuration of 3 to 5 locations, with explicit numbers, lets the student see the exact entries of $\mathbf{H}, \boldsymbol{\Sigma}, \mathbf{W}, \mathbf{Q}$ and verify the arithmetic by hand.
4. **Algorithm and simulation.** The matrix equations are translated into `NumPy` and `SciPy` code, and a simulation with known parameters shows how the method performs.
5. **Graded exercises.** Analytical derivations (matrix proofs and `SymPy` checks), then computational and simulation problems (implementing estimators and running Monte Carlo studies).
