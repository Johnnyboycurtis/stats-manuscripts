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
5. **Known-truth simulation instead of data wrangling.** In the spirit of Rizzo, Cholesky simulation is used from Chapter 0 and explained in Chapter 2. Students generate synthetic fields with known parameters $(\boldsymbol{\beta}, \sigma^2, \tau^2, \phi)$ and use Monte Carlo experiments to see why naive methods fail and how spatial estimators recover the truth. Simulations also include the problems real data bring: duplicate locations, skewed responses, outliers, and intensity trends.
6. **The scientific Python stack.** Computation uses `NumPy`, `SciPy`, and `SymPy`, with `matplotlib` for maps, variograms, and envelopes, and no GIS or spatial packages. Where a symbolic check helps (Chapters 0, 4, and 5), derivations are verified in `SymPy`. Numerical linear algebra, optimization, and special functions are built from first principles with `NumPy` and `SciPy`.

### Scope

**Included:** geostatistics (variograms, kriging, change of support, likelihood inference and model selection), areal and lattice models (spatial weights, SAR, CAR, BYM, spatial GLMMs, and the modifiable areal unit problem), point patterns (intensity estimation, second-order summaries, inhomogeneous Poisson, cluster, and Gibbs models), and computational methods for large $n$. Bayesian ideas appear only as empirical Bayes shrinkage (Chapter 9), Gaussian Markov random fields and the SPDE bridge (Chapter 13), and scan statistics (Chapter 11); full Bayesian computation is left to Appendix D.

**Deliberately excluded:** measure-theoretic probability, $\sigma$-algebras, spectral measures, finite-element methods, and infinite-dimensional functional analysis. A random field is treated through its finite-dimensional distributions, and covariance validity is checked through eigenvalues, Cholesky factors, and closure properties, with validity proofs for named families taken from the literature. Appendix D lists directions beyond the book.

## Notation

All notation, the typographic conventions, the model used throughout, and the concordance with Schabenberger and Gotway (2005) are in `NOTATION.md`. It is the single source for symbols. Its "First introduced" column says where each symbol is defined, and no chapter may use a symbol before that point.

## Table of Contents

## Getting Started

### Chapter 0: The Big Picture and the Math You Already Know

*Running stories:* apartment rents, census tracts, and traffic crashes (opening); noise monitors and a district average (§0.3.1); neighborhood rents and childhood asthma rates (§0.3.2); county overdose deaths (§0.3.3).

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
    * 0.3.2 Spurious correlation: two independent but smooth surfaces usually show only a modest sample correlation, yet a naive test calls it significant far more often than 5% of the time (derived in Chapter 1)
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
    * 0.5.3 A map of the book: one paragraph per part
* 0.6 Setting up: a short environment check for `NumPy`, `SciPy`, `SymPy`, and `matplotlib`
* 0.7 *Warm-up exercises: one problem on each of the five tools, with pointers to Appendix A for any that feel unfamiliar*

---

## PART I: Matrix Foundations and Spatial Covariance

### Chapter 1: Spatial Dependence Through the Lens of Linear Models

*Running stories:* rent regressed on square footage (§1.1); nearby rents sharing an unexplained neighborhood effect (§1.2); four apartments at the corners of a block (§1.3.2); rent on distance to transit versus floor number (§1.3.4, §1.7.2); noise monitors along a highway (§1.4.2); listings that arrive with latitude and longitude (§1.6).

* 1.1 Reviewing the classical linear model: $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\operatorname{Var}(\boldsymbol{\varepsilon}) = \tau^2 \mathbf{I}_n$ (written $\tau^2$ because $\sigma^2$ is reserved for the spatial variance)
    * 1.1.1 A tool we need: the expected value of a quadratic form, $\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}$, derived with the trace
    * 1.1.2 Estimating the error variance, and the three places where independence is used
* 1.2 What is a spatial process? From i.i.d. scalars to correlated random vectors $\mathbf{Y} = [Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_n)]'$; the covariance function $C(h)$ and the variance $C(0)$
* 1.3 Why we cannot ignore space: OLS when $\operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma}$
    * 1.3.1 Unbiasedness is preserved: $\operatorname{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$
    * 1.3.2 The true variance of OLS, $(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}$, the downward bias of $s^2$, and a four-apartment example worked by hand
    * 1.3.3 The patterns inside $\boldsymbol{\Sigma}$: eigenvectors and eigenvalues, from a two-location example to the general case
    * 1.3.4 When the standard error is too small: the variance ratio as a weighted average of eigenvalues, most severe when covariates are spatially smooth
    * 1.3.5 Generalized least squares as OLS on decorrelated data, and how much efficiency OLS loses
    * 1.3.6 Proof of Aitken's theorem (Optional)
* 1.4 Effective sample size
    * 1.4.1 Definition for the sample mean: $n_{\text{eff}} = n^2/\mathbf{1}'\mathbf{R}\mathbf{1}$
    * 1.4.2 A transect: the closed form $n_{\text{eff}} \approx n(1 - r_1)/(1 + r_1)$
    * 1.4.3 Infill: more points in the same region, and the ceiling on $n_{\text{eff}}$
    * 1.4.4 Effective sample size depends on the question
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
* 1.7 Walkthrough and Lab 1
    * 1.7.1 *Checking the four-apartment calculations in code, and adding an alternating covariate on the same block*
    * 1.7.2 *A 1,000-run Monte Carlo coverage experiment: rent on distance to transit (smooth) and on floor number (unrelated to location)*
    * 1.7.3 *Comparing Euclidean-on-degrees, projected, great-circle, and chordal distances*

### Chapter 2: The Spatial Linear Model, Covariance, and Simulation

*Running stories:* a city's rent surface, with $u$ as unexplained neighborhood desirability; lead in tap water as a large-nugget example, since lead depends on each house's own service line; simulated rent fields.

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
    * 2.4.2 Why naive functions fail: a three-location step-function counterexample (correlation 0.9 within 1 km, none beyond) with a negative eigenvalue
    * 2.4.3 Algebraic closure properties: sums, products, and Gramians $\mathbf{B}\mathbf{B}'$
    * 2.4.4 What finite checks can and cannot show: the tent function passes small checks and on a line but fails on a grid in the plane; validity of the named families is taken from the literature
* 2.5 Parametric correlation families $r(h;\phi)$
    * 2.5.1 Exponential and Gaussian models: smoothness versus numerical ill-conditioning (condition numbers, jitter)
    * 2.5.2 The spherical model: exact zeros and compact support
    * 2.5.3 The Matérn family: general form and the closed-form cases $\nu = 1/2,\ 3/2,\ 5/2$
* 2.6 Geometric anisotropy through coordinate transformations: $h_{ij}(\mathbf{G}) = \sqrt{(\mathbf{s}_i - \mathbf{s}_j)'\mathbf{G}(\mathbf{s}_i - \mathbf{s}_j)}$
* 2.7 Simulating spatial random fields from first principles
    * 2.7.1 The Cholesky factorization: two locations by hand, the column-by-column algorithm, and a check against 100,000 simulated draws
    * 2.7.2 Spectral simulation $\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ when $\boldsymbol{\Sigma}$ is nearly singular
    * 2.7.3 Simulating $u$ and $\varepsilon$ separately, so the noise-free surface is known
* 2.8 Walkthrough and Lab 2
    * 2.8.1 *Eigenvalues of valid and invalid matrices: the exponential block, the step function, and the tent on a line and on a grid*
    * 2.8.2 *A simulated rent district in three layers (trend, neighborhood effect, nugget), with matplotlib figures of the layers and of exponential versus Matérn smoothness*

---

## PART II: Geostatistical Models and Generalized Least Squares

### Chapter 3: Empirical Variography and Weighted Least Squares Fitting

*Running stories:* variograms of noise-monitor readings; anisotropy along a river or highway corridor; outliers from data-entry errors in listings.

* 3.1 Estimating the empirical semivariogram
    * 3.1.1 Matheron's method-of-moments estimator, why it is unbiased under intrinsic stationarity, and a five-monitor example by hand
    * 3.1.2 Vectorized computation with the pairwise difference matrix $\mathbf{y}\mathbf{1}' - \mathbf{1}\mathbf{y}'$; the shared simulated rent-listing data set (250 listings) used in Chapters 3 to 5
    * 3.1.3 Distance bins, lag tolerances, and why pairs beyond about half the largest distance are unreliable
    * 3.1.4 Directional variograms: angular tolerances, a riverfront field with three-to-one anisotropy, the rose diagram, and zonal anisotropy
* 3.2 Robust estimation
    * 3.2.1 Why outliers inflate Matheron's estimator: five bad listings
    * 3.2.2 The Cressie-Hawkins fourth-root estimator
    * 3.2.3 Where the constants 0.457 and 0.494 come from (Optional)
* 3.3 Fitting parametric curves to empirical variograms
    * 3.3.1 Why ordinary nonlinear least squares is suboptimal: bins are correlated and heteroscedastic
    * 3.3.2 The approximation $\operatorname{Var}(\hat{\gamma}(h_k)) \approx 2[\gamma(h_k)]^2/\lvert N(h_k)\rvert$, derived from $\operatorname{E}[Z^4] = 3$, and a simulation showing it understates the true variance by a factor of about 9.5
    * 3.3.3 Cressie's weighted least squares objective and iterative fitting
* 3.4 The trend-contamination pitfall
    * 3.4.1 What an ignored trend does: the extra term $\tfrac{1}{2}(\mu_i - \mu_j)^2$, growing like $h^2$, and a runaway exponential fit
    * 3.4.2 Detrending with OLS residuals $\mathbf{e} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{Y}$
    * 3.4.3 Why residuals have a biased variogram: $\boldsymbol{\Sigma}_e = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\boldsymbol{\Sigma}(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})$ (resolved by REML in Chapter 5)
* 3.5 Walkthrough and Lab 3
    * 3.5.1 *From raw listings to a fitted variogram: exponential and Matérn ($\nu = 3/2$) fits by Cressie's WLS, with Figure 3.1*
    * 3.5.2 *Directional variograms of the riverfront field (Figure 3.2)*
    * 3.5.3 *Matheron versus Cressie-Hawkins under contamination (Figure 3.3)*

### Chapter 4: Spatial Prediction: Kriging as Linear Model BLUP

*Running stories:* predicting the rent of an unlisted building; the average rent over a neighborhood (block kriging; housing agency benchmark); kriging log rents.

* 4.1 Demystifying "kriging": it is best linear unbiased prediction
    * 4.1.1 The prediction target: the noise-free value $\tilde{Y}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + u(\mathbf{s}_0)$, with $\mathbf{c}_0 = \operatorname{Cov}(\mathbf{Y}, \tilde{Y}(\mathbf{s}_0))$
    * 4.1.2 Linear predictors $\hat{Y}(\mathbf{s}_0) = \lambda_0 + \boldsymbol{\lambda}'\mathbf{Y}$, and mean squared prediction error as variance plus squared bias
    * 4.1.3 The variance part is a quadratic form: $\operatorname{MSPE}(\boldsymbol{\lambda}) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2$ for unbiased predictors
* 4.2 Simple kriging (known mean)
    * 4.2.1 A tool we need: derivatives with respect to a vector
    * 4.2.2 The simple kriging equations $\boldsymbol{\Sigma}\boldsymbol{\lambda} = \mathbf{c}_0$, the predictor, and the variance $\sigma^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$, which does not depend on $\mathbf{y}$ (sampling design, Appendix D)
    * 4.2.3 By hand: the center of the four-apartment block, and a location off the block (eigenvector solve, negative weights, the screening effect)
    * 4.2.4 Two geometric pictures: kriging as projection under the covariance inner product; GLS fitted values as projection under the $\boldsymbol{\Sigma}^{-1}$ inner product
* 4.3 Ordinary kriging (unknown constant mean)
    * 4.3.1 Unbiasedness as the constraint $\boldsymbol{\lambda}'\mathbf{1}_n = 1$, enforced by a Lagrange multiplier $m$
    * 4.3.2 The block-matrix system with the kriging matrix $\mathbf{K}$, solved by elimination: ordinary kriging is GLS for the mean plus simple kriging
    * 4.3.3 What the Lagrange multiplier costs: $\sigma^2_{\text{OK}} = \sigma^2_{\text{SK}} + (1 - \mathbf{1}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0)^2\operatorname{Var}(\hat{\mu}_{\text{GLS}})$, worked by hand off the block
    * 4.3.4 A tool we need: inverting a block matrix (the Schur complement)
* 4.4 Universal kriging (spatial regression with covariates)
    * 4.4.1 Unbiasedness for every $\boldsymbol{\beta}$: $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{x}_0$
    * 4.4.2 The universal kriging system with multipliers $\mathbf{m}$
    * 4.4.3 The two-stage decomposition: GLS trend plus simple kriging of residuals
    * 4.4.4 The universal kriging variance: simple kriging variance plus the cost of estimating the trend; the unlisted building in the rent district
* 4.5 Two targets, and interpolation versus smoothing
    * 4.5.1 Predicting the noise-free value at a data location: interpolation when $\tau^2 = 0$, smoothing when $\tau^2 > 0$ (with Figure 4.1)
    * 4.5.2 Predicting a new observation: the same predictor, with $\tau^2$ added to the variance
    * 4.5.3 Kriging as a conditional mean, Gaussian process regression, and kernel ridge regression (Optional)
    * 4.5.4 Skewed responses: kriging $\log Y$ and the back-transformation correction $e^{\sigma^2_{\text{SK}}/2}$
* 4.6 Change of support: block kriging
    * 4.6.1 Predicting an areal average $\tilde{Y}(B)$
    * 4.6.2 The only change is $\mathbf{c}_B$ (and $\sigma^2_B$, $\mathbf{x}_B$); the block prediction is the average of point predictions
    * 4.6.3 Why block predictions are more precise
* 4.7 Evaluating predictions: leave-one-out cross-validation
    * 4.7.1 The idea, and the cost of doing it directly
    * 4.7.2 The shortcut $Y_i - \hat{Y}_{-i} = (\boldsymbol{\Pi}\mathbf{Y})_i/\Pi_{ii}$, with variance $1/\Pi_{ii}$, and the regression special case $e_i/(1 - h_{ii})$
    * 4.7.3 Using the residuals: standardized leave-one-out residuals as a check
    * 4.7.4 Proof of the shortcut (Optional)
* 4.8 Walkthrough and Lab 4
    * 4.8.1 *The ordinary kriging equations in SymPy*
    * 4.8.2 *The four-apartment block in code*
    * 4.8.3 *Do the intervals cover the truth? A 200-replicate coverage experiment against the true noise-free rents*
    * 4.8.4 *A rent map with kriging standard errors (Figure 4.2)*
    * 4.8.5 *How the kriging standard error depends on the number of listings: 20 random location sets at n = 25, 100, 250, 1000, against the effective sample size of the mean*

### Chapter 5: Likelihood Inference, Model Selection, and Diagnostics

*Running stories:* choosing a covariance model for rents; several units in one building (duplicate locations).

* 5.1 The multivariate normal log-likelihood $\ell(\boldsymbol{\beta}, \boldsymbol{\theta})$, derived from the density; the four-apartment block evaluated by hand with eigenvectors
* 5.2 Profile maximum likelihood
    * 5.2.1 Factoring $\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \sigma^2\boldsymbol{\Psi}(\phi,\delta)$ with $\delta = \tau^2/\sigma^2$
    * 5.2.2 Closed-form GLS substitution for $\hat{\boldsymbol{\beta}}(\phi,\delta)$ and $\hat{\sigma}^2(\phi,\delta) = \mathrm{RSS}_\Psi/n$, the profile log-likelihood, and ML and REML fits of the rent district
* 5.3 Restricted maximum likelihood (REML)
    * 5.3.1 Why ML underestimates variance parameters: the $n/(n-1)$ example and $\operatorname{E}[\mathrm{RSS}_\Psi] = \sigma^2(n - p)$
    * 5.3.2 Filtering out $\mathbf{X}\boldsymbol{\beta}$ with error contrasts $\mathbf{T}$, $\mathbf{T}'\mathbf{X} = \mathbf{0}$; the identities $\boldsymbol{\Pi} = \mathbf{T}(\mathbf{T}'\boldsymbol{\Sigma}\mathbf{T})^{-1}\mathbf{T}'$ and $\lvert\mathbf{T}'\boldsymbol{\Sigma}\mathbf{T}\rvert = \lvert\boldsymbol{\Sigma}\rvert\lvert\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\rvert/\lvert\mathbf{X}'\mathbf{X}\rvert$
    * 5.3.3 The REML log-likelihood: why the choice of $\mathbf{T}$ only changes a constant, the form in terms of $\boldsymbol{\Sigma}$, and profiling with $\hat{\sigma}^2 = \mathrm{RSS}_\Psi/(n - p)$
    * 5.3.4 Proofs of the two identities (Optional)
* 5.4 Numerical optimization and asymptotic standard errors
    * 5.4.1 Evaluating $\log\lvert\boldsymbol{\Sigma}\rvert$ and $\boldsymbol{\Sigma}^{-1}\mathbf{e}$ safely with Cholesky, without forming an inverse
    * 5.4.2 The score and the expected information: derivatives of $\boldsymbol{\Sigma}^{-1}$ and $\log\lvert\boldsymbol{\Sigma}\rvert$, the ML and REML formulas, and the derivative matrices for the exponential model
    * 5.4.3 Fisher scoring from a poor start, standard errors from expected and observed information
    * 5.4.4 Practical identifiability: the ridge in $(\sigma^2, \phi)$ and why only $\sigma^2/\phi$ is well determined under fixed-domain sampling
    * 5.4.5 Plug-in prediction: kriging variances from ML, REML, and true parameters, and what plug-in ignores
    * 5.4.6 Deriving the information formula (Optional)
* 5.5 Model selection and tests for spatial structure
    * 5.5.1 AIC and BIC across Matérn smoothness $\nu = 1/2, 3/2, 5/2$, and why REML likelihoods are comparable only for the same $\mathbf{X}$
    * 5.5.2 Testing $\tau^2 = 0$ and $\sigma^2 = 0$ on the boundary: the $\tfrac{1}{2}\chi^2_0 + \tfrac{1}{2}\chi^2_1$ mixture, explained
    * 5.5.3 Checking the boundary test by parametric bootstrap (200 replicates)
* 5.6 Spatial regression diagnostics and confounding
    * 5.6.1 Decorrelated residuals $\mathbf{r}^* = \mathbf{L}^{-1}(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})$
    * 5.6.2 Spatial confounding: OLS versus GLS for a smooth covariate, and its share on the leading eigenvectors of $\boldsymbol{\Sigma}$
    * 5.6.3 Duplicate locations: several units in one building, why $\boldsymbol{\Sigma}$ is singular when $\tau^2 = 0$, and why duplicates are the best data for the nugget
* 5.7 Walkthrough and Lab 5
    * 5.7.1 *Checking the information formula symbolically in SymPy*
    * 5.7.2 *ML versus REML in 200 small simulated data sets*
    * 5.7.3 *Figures: the profile REML surface and its ridge, and a normal quantile plot of decorrelated residuals*

---

## PART III: Discrete Spatial Models (Lattice and Areal Data)

### Chapter 6: Areal Units, Spatial Weights, and Quadratic Form Tests

*Running stories:* overdose death rates by county; census tracts versus ZIP codes (MAUP); spatial transcriptomics spots as an aside.

* 6.1 Areal units are a choice: the modifiable areal unit problem
    * 6.1.1 The scale effect and the zoning effect: a four-tract hand example (slope $+0.2$ for tracts, $+0.5$ and $-1.0$ for two zonings), why aggregation changes a slope (a variance-weighted average of a local and a regional effect), and a 200-world simulation
    * 6.1.2 The ecological fallacy: three counties and nine people (within-county slope $+1$, county-mean slope $-1.5$, pooled slope $-1.0$)
    * 6.1.3 What this means for every model in Part III
* 6.2 Building a spatial weights matrix
    * 6.2.1 Contiguity: rook and queen, with spatial transcriptomics spots on a hexagonal lattice as an aside
    * 6.2.2 Distance bands, inverse distance, and $k$ nearest neighbors
    * 6.2.3 Why nearest-neighbor weights are not symmetric (three rural clinics on a highway), and which results that breaks
    * 6.2.4 Sensitivity of conclusions to the choice of $\mathbf{W}$: a strong and a weak simulated response, six weights
* 6.3 Representing regions as graphs
    * 6.3.1 The adjacency matrix $\mathbf{A}$ and the degree matrix $\mathbf{D}$, with the four-region hub map
    * 6.3.2 Row-standardized weights $\mathbf{W} = \mathbf{D}^{-1}\mathbf{A}$, $S_0 = n$, and why $\mathbf{W}$ is not symmetric
    * 6.3.3 A similarity transformation: $\mathbf{W}$ has the real eigenvalues of $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$; where they lie ($\lvert\lambda\rvert \le 1$, and $-1$ occurs exactly for a bipartite map); the hub map by hand
* 6.4 Moran's $I$ as a ratio of quadratic forms
    * 6.4.1 The centering matrix $\mathbf{P}_c$ and the centered data $\mathbf{y}_c$
    * 6.4.2 The symmetrized weights $\mathbf{W}_s$ and the central matrix $\mathbf{M} = \mathbf{P}_c\mathbf{W}_s\mathbf{P}_c$
    * 6.4.3 The ratio form $I = \frac{n}{S_0}\frac{\mathbf{y}'\mathbf{M}\mathbf{y}}{\mathbf{y}'\mathbf{P}_c\mathbf{y}}$; $I$ as the slope of a regression of neighbor averages on values (the Moran scatterplot); the hub map by hand, and binary weights giving a different answer
    * 6.4.4 Reading the value of $I$: a weighted average of the eigenvalues of $\mathbf{W}_s$
* 6.5 Exact null moments of $I$ under normality
    * 6.5.1 A tool we need: quadratic forms in normal vectors as weighted sums of squared normals
    * 6.5.2 Independence of $I$ and its denominator (the Pitman-Koopmans argument via rotation invariance)
    * 6.5.3 The mean: $\operatorname{E}[I] = -1/(n-1)$
    * 6.5.4 The variance from the second moments of both quadratic forms; the hub map by hand ($\operatorname{Var}(I) = 3/40$)
    * 6.5.5 Checking the formulas by simulation
    * 6.5.6 The textbook variance formula with $S_0$, $S_1$, $S_2$ (Optional)
* 6.6 Moran's $I$ on regression residuals
    * 6.6.1 The same derivation with $\mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$ in place of $\mathbf{P}_c$; the hub map by hand ($\operatorname{E}[I_e] = -71/120$)
    * 6.6.2 An application to the simulated overdose rates: $I$ falls from $0.615$ to $0.348$ after regressing on deprivation
* 6.7 Geary's $C$ and local indicators
    * 6.7.1 Geary's $C$ as a graph-Laplacian quadratic form, with $\operatorname{E}[C] = 1$, the relation to $I$, and the hub map by hand
    * 6.7.2 Local indicators of spatial association: they sum to $S_0 I$; conditional permutation tests; what 144 local tests do on maps with no structure
* 6.8 Distribution-free inference: permutation tests
    * 6.8.1 The idea, and why the $+1$ p-value is valid
    * 6.8.2 The mean of the permutation distribution is $-1/(n-1)$ for every $\mathbf{W}$
    * 6.8.3 Exact enumeration on four regions (6 distinct arrangements, exact $p = 1/3$)
    * 6.8.4 Monte Carlo inference on the county map
    * 6.8.5 When permuting the data gives the wrong answer: unequal variances, and residuals (with the Freedman-Lane projection)
* 6.9 Walkthrough and Lab 6
    * 6.9.1 The hub map collected: every hand calculation next to its code check
    * 6.9.2 Weights and conclusions on the county map (z-scores and permutation p-values for six weights and two responses; Figures 6.3 and 6.4)
    * 6.9.3 A simulated study of the modifiable areal unit problem (200 worlds, six aggregations; Figures 6.1 and 6.2)

### Chapter 7: Simultaneous Autoregressive Models: Spatial Regression with Feedback

*Running stories:* property tax rates of neighboring cities (lag model); minimum-wage spillovers across county lines (error model); R&D spillovers between regions (impacts).

* 7.1 The simultaneous idea: each region's value regressed on its neighbors', all at once; a two-region example; the lag and error models side by side
* 7.2 The spatial error model
    * 7.2.1 The model: $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$, $\mathbf{u} = \rho\mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$, and the minimum-wage story
    * 7.2.2 The reduced form and the feedback matrix $\boldsymbol{\Phi}(\rho) = (\mathbf{I}_n - \rho\mathbf{W})^{-1}$
    * 7.2.3 The induced covariance $\sigma^2\boldsymbol{\Phi}\boldsymbol{\Phi}' = \sigma^2\left[(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})\right]^{-1}$ and precision $\mathbf{Q}_{\text{SAR}}$, which reaches two steps across the map; the four-region example by hand
    * 7.2.4 GLS for $\boldsymbol{\beta}$ given $\rho$ as OLS on spatially filtered data
* 7.3 The spatial lag model
    * 7.3.1 The model, its reduced form, and why its covariance equals the error model's while its mean differs
    * 7.3.2 The feedback multiplier as a Neumann series and as walks on the map; the four-region example by hand
    * 7.3.3 Why OLS fails: omitting the lag (omitted-variable bias) and including it ($\operatorname{Cov}(\mathbf{W}\mathbf{Y}, \boldsymbol{\varepsilon}) = \sigma^2\mathbf{W}\boldsymbol{\Phi}$ and the approximate bias)
* 7.4 The parameter space: the determinant as a product of eigenvalues, the eigenvalues of $\mathbf{W}$ (at most 1 in absolute value; $-1$ for bipartite maps), and the interval $(\lambda_{\min}^{-1}, \lambda_{\max}^{-1})$
* 7.5 Maximum likelihood
    * 7.5.1 The likelihood and the Jacobian term (change of variables); maximum likelihood is OLS plus the Jacobian; a hand evaluation
    * 7.5.2 Ord's eigenvalue identity
    * 7.5.3 Profiling out $\boldsymbol{\beta}$ and $\sigma^2$ to a search over $\rho$; testing $\rho = 0$ (an interior null)
* 7.6 Choosing between the error and lag models
    * 7.6.1 The idea of a score test
    * 7.6.2 The four statistics (LM-error, LM-lag, and the robust versions) and the decision rule
    * 7.6.3 Likelihood ratio tests and AIC
    * 7.6.4 Deriving the LM statistics (Optional)
* 7.7 Interpreting the lag model: direct, indirect, and total impacts
    * 7.7.1 Why $\beta_k$ is not the marginal effect: the effect matrix $\boldsymbol{\Phi}\beta_k$
    * 7.7.2 Direct, indirect, and total impacts, computed from eigenvalues; the four-region example by hand; where the OLS slope falls
    * 7.7.3 Uncertainty in the impacts by simulation
    * 7.7.4 Checking by simulation: raise a covariate in one county
* 7.8 Walkthrough and Lab 7
    * 7.8.1 *The four regions in code*
    * 7.8.2 *The county map and its eigenvalues*
    * 7.8.3 *A simulated lag model: how badly does OLS do?*
    * 7.8.4 *Maximum likelihood and the profile likelihood (Figure 7.1)*
    * 7.8.5 *The sampling distribution of the estimators (Figure 7.2)*
    * 7.8.6 *Lagrange multiplier tests: one data set and rejection rates under three truths*
    * 7.8.7 *Impacts with intervals, and a simulation check*
    * 7.8.8 *The error model: naive versus ML intervals*
    * 7.8.9 *Figures (Figure 7.3 shows spillovers for three values of $\rho$)*

### Chapter 8: Conditional Autoregressive Models and Sparse Precision Matrices

*Running stories:* county health rates; gene expression on a tissue-slide spot grid.

* 8.1 The conditional idea: specify each region given its neighbors, and let the joint distribution follow
    * 8.1.1 Full conditionals: $u_i \mid \mathbf{u}_{-i} \sim N\!\left(\sum_{j \ne i} c_{ij}u_j,\ \sigma_i^2\right)$, and the four-region map (hub A, triangle A-B-C, pendant D) used throughout the chapter
    * 8.1.2 A tool we need: conditional distributions from the precision matrix, derived by completing the square and checked with the Schur complement of Chapter 4; the weights $-Q_{ij}/Q_{ii}$ are kriging weights, and the leave-one-out residual of Chapter 4 is $[\mathbf{Q}(\mathbf{y} - \boldsymbol{\mu})]_i/Q_{ii}$
    * 8.1.3 Reconstructing the joint distribution: Brook's lemma by hand for three regions, the symmetry requirement $c_{ij}\sigma_j^2 = c_{ji}\sigma_i^2$, and $\mathbf{Q} = \boldsymbol{\Sigma}_c^{-1}(\mathbf{I}_n - \mathbf{C})$
    * 8.1.4 Gaussian Markov random fields: $Q_{ij} = 0$ exactly when $u_i$ and $u_j$ are conditionally independent given the rest, and why that is not independence
* 8.2 The proper CAR model
    * 8.2.1 Equal weights for neighbors: $c_{ij} = \rho A_{ij}/d_i$, $\sigma_i^2 = \sigma^2/d_i$, and $\mathbf{Q}_{\text{CAR}} = \sigma^{-2}(\mathbf{D} - \rho\mathbf{A})$
    * 8.2.2 A four-region map by hand: $\mathbf{Q}$, the columns of $\boldsymbol{\Sigma}$ solved by hand ($12/29$, $32/29$, and so on), conditional versus marginal variances
    * 8.2.3 Validity for $\rho$ between $1/\tilde{\lambda}_{\min}$ and 1, with the normalized adjacency matrix $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$
    * 8.2.4 Why $\sigma^2$ is not the marginal variance, and why marginal variances differ by region (the edge effect)
    * 8.2.5 No eigenvalue of $\tilde{\mathbf{A}}$ exceeds 1 in absolute value (Optional)
* 8.3 The intrinsic CAR (ICAR) model
    * 8.3.1 Setting $\rho = 1$: a region is the average of its neighbors plus noise
    * 8.3.2 The quadratic form as a sum of squared neighbor differences, the incidence matrix ($\mathbf{D} - \mathbf{A} = \mathbf{O}'\mathbf{O}$), ICAR as a roughness penalty and as the random walk on a path, and the proper CAR as a blend of a ridge penalty and a roughness penalty:

      $$\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \tfrac{1}{2}\sum_{i=1}^n\sum_{j=1}^nA_{ij}(u_i - u_j)^2$$

    * 8.3.3 Rank deficiency: $(\mathbf{D} - \mathbf{A})\mathbf{1}_n = \mathbf{0}$, and the null space is the vectors constant on each connected component
    * 8.3.4 Sum-to-zero constraints, the pseudo-inverse covariance, spectral simulation, the generalized determinant, and Kirchhoff's matrix-tree check
    * 8.3.5 Why the pseudo-inverse and the generalized determinant (Optional)
* 8.4 Computing with precision matrices
    * 8.4.1 The CAR likelihood: the log-determinant from the eigenvalues of $\tilde{\mathbf{A}}$, the profile log-likelihood of $\rho$, and the sums of squares that are computed once:

      $$\log\lvert\mathbf{D} - \rho\mathbf{A}\rvert = \sum_{i=1}^n\log d_i + \sum_{k=1}^n\log(1 - \rho\tilde{\lambda}_k)$$

    * 8.4.2 Simulating from a precision matrix: factor $\mathbf{Q} = \mathbf{L}\mathbf{L}'$ and solve $\mathbf{L}'\mathbf{u} = \mathbf{z}$
    * 8.4.3 Sparse storage and sparse solves on a 4,992-spot hexagonal grid; dense versus sparse timings; a sampler that uses the incidence matrix in place of a Cholesky factor
* 8.5 SAR versus CAR
    * 8.5.1 Simultaneous versus conditional: $\mathbf{Q}_{\text{CAR}} = \sigma^{-2}\mathbf{D}(\mathbf{I}_n - \rho\mathbf{W})$ against $\mathbf{Q}_{\text{SAR}} = \sigma^{-2}(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})$; every SAR is a CAR with a richer neighborhood
    * 8.5.2 Sparsity: zeros in $\mathbf{Q}$ versus zeros in $\boldsymbol{\Sigma}$ ($3 \times 3$ patterns, $12 \times 12$ counts)
    * 8.5.3 Why $\rho$ is not comparable: neighbor correlations under CAR and SAR, and the SAR $\rho$ that matches the CAR at 0.9
* 8.6 Walkthrough and Lab 8
    * 8.6.1 *Simulating a proper CAR field on the 144-county map and recovering its parameters by profile likelihood*
    * 8.6.2 *How well are $\rho$, $\sigma^2$, and the slope recovered? 500 replicates*
    * 8.6.3 *Figures 8.1 to 8.4: simulated fields, variances, sparsity patterns, and the profile likelihood*

### Chapter 9: Generalized Linear Mixed Models for Spatial Count Data

*Running stories:* overdose deaths and vaccination rates by county (disease mapping).

* 9.1 Reviewing GLMs in the style of Dobson & Barnett
    * 9.1.1 The exponential family: mean and variance from $b(\vartheta)$; Gaussian, Poisson, and binomial
    * 9.1.2 The linear predictor and the link function $g(\mu_i) = \eta_i$; canonical links
    * 9.1.3 The score equations, and $\mathbf{X}'(\mathbf{y} - \boldsymbol{\mu}) = \mathbf{0}$ for a canonical link
    * 9.1.4 Fisher information $\mathbf{X}'\mathbf{V}\mathbf{X}$ and the weights ($v_i = \mu_i$ for the Poisson)
    * 9.1.5 IRLS is Fisher scoring (Newton-Raphson for a canonical link): the working response $\mathbf{z}$
    * 9.1.6 By hand: three counties, five iterations, checked against direct maximization
* 9.2 Spatial Poisson regression for regional counts
    * 9.2.1 The model equation again: $Y_i \mid u_i \sim \text{Poisson}\!\left(E_i\exp(\mathbf{x}_i'\boldsymbol{\beta} + u_i)\right)$ with expected counts as an offset; relative risk and the SMR
    * 9.2.2 Why the SMR misleads: $\operatorname{Var}(\text{SMR}_i) = \theta_i/E_i$, and the exact Poisson limits
    * 9.2.3 What a Poisson GLM that ignores $u$ gets wrong: overdispersion and the quasi-Poisson correction
* 9.3 Shrinkage and spatial random effects
    * 9.3.1 Empirical Bayes (Clayton and Kaldor): the gamma-Poisson model, the posterior mean as a weighted average, the negative binomial marginal likelihood, and three counties by hand
    * 9.3.2 A CAR distribution for the random effect: ICAR conditionals as neighbor averages
    * 9.3.3 The Besag-York-Mollié (BYM) model: $\mathbf{u} = \mathbf{u}^{\text{icar}} + \mathbf{v}$, and why the two components are only weakly identified
    * 9.3.4 BYM2: the constrained generalized inverse, the scaling constant $c$ as a geometric mean, a path of three regions by hand, and how $c$ grows with the map
    * 9.3.5 The Leroux precision: positive definiteness for $\rho < 1$ and its conditional distributions
    * 9.3.6 Comparing BYM, BYM2, and Leroux: eigenvalue forms, BYM = BYM2, prior marginal variances
    * 9.3.7 The constrained generalized inverse (Optional)
* 9.4 Fitting spatial GLMMs by penalized quasi-likelihood
    * 9.4.1 The penalized log-likelihood, its gradient, and its Hessian
    * 9.4.2 Henderson's mixed-model equations, from Newton's method and as penalized weighted least squares
    * 9.4.3 By hand: one penalized IRLS step on three regions
    * 9.4.4 Estimating the hyperparameters by REML on the working vector, $\boldsymbol{\Sigma}_z = \mathbf{V}^{-1} + \mathbf{Q}^{-1}$
    * 9.4.5 Where PQL is weak
* 9.5 The Laplace approximation of the marginal likelihood
    * 9.5.1 The derivation, with the Gaussian integral as a tool
    * 9.5.2 A one-region check against numerical integration
    * 9.5.3 Using it: maximum likelihood and likelihood ratio tests on the boundary
* 9.6 Walkthrough and Lab 9
    * 9.6.1 *The county data and the raw SMRs, with the funnel plot (Figure 9.1)*
    * 9.6.2 *The Poisson GLM and empirical Bayes*
    * 9.6.3 *The spatial models by PQL, with RMSE comparison and maps (Figure 9.2)*
    * 9.6.4 *Laplace maximum likelihood, the likelihood ratio test, and weak identification (Figure 9.3)*
    * 9.6.5 *Does the standard error cover? A 100-replicate experiment*

---

## PART IV: Spatial Point Patterns

### Chapter 10: Exploring Point Patterns: Intensity, Randomness, and Interaction

*Running stories:* immune cells on a tumor slide; traffic crashes; basketball shot charts with made and missed shots.

* 10.1 What changes in a point pattern: the coordinate matrix $\mathbf{S} \in \mathbb{R}^{n \times 2}$ is now the random outcome; first-order, second-order, and multitype questions; intensity versus density; the window
* 10.2 The homogeneous Poisson process (complete spatial randomness, CSR)
    * 10.2.1 Deriving $N(B) \sim \text{Poisson}(\lambda\operatorname{area}(B))$ as the limit of Bernoulli counts on a refining pixel grid, with a generating-function proof and a numerical check
    * 10.2.2 Conditional uniformity: given $N(\mathcal{D}) = n$, locations are i.i.d. uniform on $\mathcal{D}$ (derived through the multinomial)
    * 10.2.3 Quadrat counts: Pearson's $X^2$ with $\operatorname{E}[X^2] = n_q - 1$ derived, the index of dispersion, a four-event hand example, and a simulated check of the null distribution
* 10.3 Simulating point patterns from first principles
    * 10.3.1 Homogeneous patterns: a Poisson count, then uniform coordinates
    * 10.3.2 Lewis-Shedler thinning: the algorithm, why it works, an eight-candidate table by hand, and a check of the counts against the integral of the crash intensity
    * 10.3.3 Why simulation comes first: Monte Carlo tests and exchangeability
    * 10.3.4 Two patterns that are not random (a Thomas-type clustered pattern and a Matérn type II inhibited pattern, simulated inline), and the quadrat test's blind spot for small-scale regularity
* 10.4 First-order exploration: kernel intensity estimation
    * 10.4.1 The estimator $\hat{\lambda}(\mathbf{s}) = \sum_{i=1}^n \mathcal{K}_\psi(\mathbf{s} - \mathbf{s}_i)$ on a grid through the grid-to-event distance matrix; bias $\approx (\psi^2/2)\nabla^2\lambda$ and variance $\approx \lambda/(4\pi\psi^2)$, checked by simulation
    * 10.4.2 Edge correction: dividing by the kernel mass $\mathcal{E}_\psi(\mathbf{s})$ inside the window, checked at a point 0.02 from the boundary
    * 10.4.3 Choosing the bandwidth: the normal-reference rule, integrated squared error, and likelihood cross-validation; the crash example (Figure 10.2)
* 10.5 Distance methods
    * 10.5.1 Nearest-neighbor distances $d_i^{\text{nn}}$ and $\hat{G}(r)$, with the four-event hand example (coordinates, $\mathbf{H}$, $d_i^{\text{nn}}$, $\hat{G}$)
    * 10.5.2 Deriving $G(r) = 1 - \exp(-\lambda\pi r^2)$ from the void probability, and $\operatorname{E}[d^{\text{nn}}] = 1/(2\sqrt{\lambda})$
    * 10.5.3 The empty-space function $F(r)$ from a grid of test points
    * 10.5.4 The $J$-function $J(r) = (1 - G(r))/(1 - F(r))$: equal to 1 under CSR, below 1 for clustering, above 1 for inhibition; the three patterns compared
* 10.6 Second-order summaries
    * 10.6.1 Ripley's $K$-function and $K(r) = \pi r^2$ under CSR
    * 10.6.2 The estimator with Ripley's edge weights $\omega_{ij}$, the exact `arc_inside` function, and a four-event hand example of the weights and $\hat{K}(0.55)$
    * 10.6.3 Besag's $L(r) = \sqrt{K(r)/\pi}$, plotted as $L(r) - r$, and why it stabilizes the variance (delta method, checked by simulation)
    * 10.6.4 The pair correlation function $g(r) = K'(r)/(2\pi r)$, estimated by kernel smoothing of pair distances and checked against the Thomas formula
* 10.7 Multitype patterns
    * 10.7.1 Marks, and two kinds of question: interaction versus different intensities
    * 10.7.2 Cross-type summaries $K_{12}(r)$ and $G_{12}(r)$; independence, attraction, and segregation compared
    * 10.7.3 The spatial relative risk $\operatorname{RR}(\mathbf{s}) = \lambda_1(\mathbf{s})/\lambda_0(\mathbf{s})$, why the two kernel estimates should share a bandwidth (a bias calculation with no noise), and how much data a relative risk map needs (shared size versus ten times the data)
* 10.8 Boundary truncation
    * 10.8.1 The exact mean of the naive $\hat{K}$, $\pi r^2 - \tfrac{8}{3}r^3 + \tfrac{1}{2}r^4$, derived, and the guard-area estimator
    * 10.8.2 Reduced-sample estimators for $G$ and $F$: censoring at the boundary distance, Kaplan-Meier, and Hanisch; the four-event example in which every distance is censored
    * 10.8.3 Ripley's weight on a rectangle: the single-edge arc, corner overlaps by inclusion and exclusion, and why the weighted estimator is unbiased
    * 10.8.4 Comparing the fixes on 300 CSR patterns
* 10.9 Monte Carlo envelopes and tests
    * 10.9.1 The pointwise envelope and its level $2/(m + 1)$ at one distance
    * 10.9.2 Why a pointwise envelope is not a test, and the maximum-deviation test; a study with 100 CSR data sets
    * 10.9.3 Envelopes for multitype summaries: random labelling
* 10.10 Walkthrough and Lab 10
    * 10.10.1 *Random, clustered, and inhibited patterns: $G$, $F$, $J$, $L$, and $g$ with envelopes and the global test (Figures 10.3 and 10.4)*
    * 10.10.2 *Case-control data: relative risk maps at two sample sizes and random labelling envelopes for $K_{12}$ and $G_{12}$ (Figure 10.5)*

### Chapter 11: Point Processes as Generalized Linear Models

*Running stories:* crashes and road features; disease cases near an industrial plant (case-control); outbreak detection (scan statistic).

* 11.1 Intensity as a regression function: $\lambda(\mathbf{s}) = \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$
    * 11.1.1 The model and its parts: the Poisson process restated, the log link, multiplicative effects
    * 11.1.2 The crash data: the shared Part IV setting, thinning, and 403 simulated crashes
    * 11.1.3 The link to Chapter 9: a count regression with offset $\log a$ as the regions shrink
* 11.2 The exact likelihood from a refining grid
    * 11.2.1 Bernoulli trials on a pixel grid: the limit of the pixel log-likelihood is $\sum_i\log\lambda(\mathbf{s}_i) - \int\lambda$, with a three-event numerical check
    * 11.2.2 The log-likelihood for a log-linear intensity: score, information $\mathcal{I}(\boldsymbol{\beta}) = \int\mathbf{x}\mathbf{x}'\lambda$, and concavity
    * 11.2.3 By hand: the homogeneous case
* 11.3 The Berman-Turner quadrature device
    * 11.3.1 The integral is the obstacle: quadrature points $\mathbf{s}_k^{*}$ with weights $a_k$, and why the data points belong among them
    * 11.3.2 Data-point indicators $\mathbb{1}_k$ and pseudo-responses $y_k = \mathbb{1}_k/a_k$; counting weights and tile weights
    * 11.3.3 A weighted Poisson regression, and IRLS with prior weights and an offset; the BT score is the exact score with the integral replaced by quadrature
    * 11.3.4 By hand: three data points and four dummy points (every matrix printed, one IRLS step, check by direct maximization)
    * 11.3.5 Choosing the dummy points; counting versus Dirichlet-tile weights; the areal GLM of Chapter 9 as a special case of the device
    * 11.3.6 The crash pattern (Figure 11.1)
* 11.4 Offsets: modeling excess over a known baseline
    * 11.4.1 The offset model $\log\lambda = \log\lambda_{\text{pop}} + \mathbf{x}'\boldsymbol{\beta}$ and its place in the working response
    * 11.4.2 Reading the coefficients as relative excess risk
    * 11.4.3 The plant story: one year versus five years of cases, likelihood ratio test, profile over the plume reach $h$
* 11.5 Case-control data and the logistic bridge
    * 11.5.1 Pooling cases and controls
    * 11.5.2 Deriving the bridge: $\operatorname{logit}\Pr(Y = 1 \mid \mathbf{s}) = \log\operatorname{RR}(\mathbf{s})$, and the baseline cancels
    * 11.5.3 Binomial IRLS and the sampling ratio (the intercept moves by $\log k$)
    * 11.5.4 Against the kernel map (Figure 11.2)
* 11.6 Where is the excess? Spatial scan statistics
    * 11.6.1 Windows and expected counts
    * 11.6.2 The log-likelihood ratio of a window, derived conditionally on $n$, with a hand check
    * 11.6.3 Computing all windows at once with `scipy.special.xlogy`
    * 11.6.4 Inference: why the maximum needs a simulation (Figure 11.3)
    * 11.6.5 Cautions
* 11.7 Separating trend from interaction
    * 11.7.1 Why the ordinary $K$-function reports clustering: inflation by $1 + \operatorname{CV}^2$
    * 11.7.2 The inhomogeneous $K$-function, why it is unbiased under no interaction, and translation edge weights for the unit square
    * 11.7.3 In practice: envelopes, and the bias of estimating $\hat{\lambda}$ from the same data (Figure 11.4)
* 11.8 Inference under clustering
    * 11.8.1 What clustering does to counts: the inhomogeneous Thomas process by thinning, the variance of a sum over events, the variance-to-mean ratio of counts (Figure 11.5)
    * 11.8.2 The pair correlation of the Thomas process (Optional)
    * 11.8.3 The sandwich for a point process: $\mathcal{I}^{-1}(\mathcal{I} + \boldsymbol{\Xi})\mathcal{I}^{-1}$
    * 11.8.4 Estimating the middle: pair-correlation plug-in with minimum contrast, and block scores
    * 11.8.5 The parametric bootstrap
    * 11.8.6 One clustered pattern: every standard error
* 11.9 Walkthrough and Lab 11 (the repeated-sampling experiments)
    * 11.9.1 The price of an unknown baseline
    * 11.9.2 Size and power of the scan test
    * 11.9.3 Plug-in intensities and the inhomogeneous $K$
    * 11.9.4 Coverage under clustering
    * 11.9.5 The bootstrap over repeated patterns

### Chapter 12: Models for Clustering and Inhibition

*Running stories:* immune cells clustering around tumor cells (cluster processes); the spacing of tumor cells (inhibition).

* 12.1 Three explanations for a departure from CSR: heterogeneity (a trend in intensity), clustering (points attract), and inhibition (points repel); the table of fingerprints in $g(r)$ and $K(r)$, and the translation-corrected $\hat{K}$ used throughout
* 12.2 Cluster processes
    * 12.2.1 Neyman-Scott processes: parents, Poisson offspring counts, dispersal; the intensity $\kappa\mu_c$ from Campbell's formula, and the count variance $\lambda(1 + \mu_c)$
    * 12.2.2 The Thomas process (Gaussian dispersal with spread $\sigma_c$) and the Matérn cluster process (uniform on a disc of radius $R$)
    * 12.2.3 Simulation with a buffer: with and without it, mean counts and edge-strip intensity
    * 12.2.4 Deriving the $K$-function of the Thomas process: the second-order product density, same-parent and different-parent pairs, $g(\mathbf{h}) = 1 + f_2(\mathbf{h})/\kappa$, and $K(r) = \pi r^2 + \operatorname{P}(\lVert\mathbf{D}_1 - \mathbf{D}_2\rVert \le r)/\kappa$
    * 12.2.5 Fitting by minimum contrast: the contrast, the power $c = 1/4$ and its parallel with Chapter 3, a two-radius hand example, one slide, and 200 slides for three powers
    * 12.2.6 Inhomogeneous cluster models: independent thinning, the Berman-Turner fit of the trend, ordinary versus inhomogeneous $K$ for the cluster fit, and the naive standard error that is $3.9$ times too small
    * 12.2.7 The $K$-function of the Matérn cluster process (Optional)
* 12.3 The log-Gaussian Cox process
    * 12.3.1 The model equation once more: $\log\Lambda(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + u(\mathbf{s})$, the lognormal mean $\lambda = e^{\beta_0 + \sigma^2/2}$, the derivation of $g(r) = \exp(C_u(r))$, and the Thomas process as a Cox process
    * 12.3.2 Simulating and fitting an LGCP: pixel-grid simulation, the numerical $K$, minimum contrast, and a Thomas-versus-LGCP comparison on both kinds of pattern
    * 12.3.3 A pixel grid: Parts II and III meet the Poisson GLM (penalized IRLS with a Leroux precision, Laplace choice of $\xi$)
* 12.4 Inhibition
    * 12.4.1 Matérn hard-core processes: Type I and Type II, the derived intensities $\lambda_{\text{par}}e^{-\lambda_{\text{par}}\pi r_0^2}$ and $(1 - e^{-\lambda_{\text{par}}\pi r_0^2})/(\pi r_0^2)$, and why both cap the intensity
    * 12.4.2 Simple sequential inhibition and the jamming limit
    * 12.4.3 Gibbs processes and the Strauss model: the density, why $\gamma \le 1$, why $Z$ is intractable, the conditional intensity, and birth-death Metropolis-Hastings with its acceptance probabilities
    * 12.4.4 The matrix view: one more column; a four-point hand example with the indicator matrix printed
    * 12.4.5 Maximum pseudolikelihood: the cell-by-cell derivation, the Berman-Turner fit, why GLM standard errors fail, a parametric bootstrap, and choosing $r_0$ by profiling
* 12.5 Checking a fitted point process model
    * 12.5.1 Residuals: smoothed residual fields
    * 12.5.2 Goodness of fit with envelopes from the fitted model
    * 12.5.3 Choosing among trend, cluster, Cox, and Gibbs models: a table of Monte Carlo tests for three patterns and five fitted models
* 12.6 Walkthrough and Lab 12: seven figures (three explanations; minimum contrast; inhibition mechanisms; the Strauss slide and profile; envelopes; residual fields; the LGCP and its pixel GMRF fit); the lab is organized by figure rather than numbered subsections

---

## PART V: Matrix Scalability for Large Spatial Systems

### Chapter 13: Matrix Approximations and Fast Algorithms for Large $n$

*Running stories:* a metropolitan rent data set with tens of thousands of listings.

* 13.1 The $\mathcal{O}(n^3)$ bottleneck
    * 13.1.1 What one likelihood evaluation costs in time and memory
    * 13.1.2 Measuring it: Cholesky time and matrix size up to $n = 5{,}000$, a fitted exponent near 3, and extrapolations (labelled as such) to $n = 100{,}000$
    * 13.1.3 Four structures: low rank, sparse covariance, sparse factor of the precision, Kronecker
* 13.2 Low-rank spatial models (fixed-rank kriging and basis expansions)
    * 13.2.1 The idea: $\mathbf{u} \approx \mathbf{B}\boldsymbol{\alpha}$ with $q \ll n$ knots, the predictive process $\mathbf{B} = \mathbf{R}_{nK}\mathbf{R}_K^{-1}$, $\boldsymbol{\Sigma}_\alpha = \sigma^2\mathbf{R}_K$, and $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'$
    * 13.2.2 A tool we need: the Sherman-Morrison-Woodbury identity and the determinant lemma, derived from block inversion and the block determinant, for a general diagonal part $\boldsymbol{\Omega}$
    * 13.2.3 By hand: four listings, two knots (dense inverse against Woodbury, $\lvert\boldsymbol{\Sigma}\rvert = \lvert\boldsymbol{\Omega}\rvert\lvert\boldsymbol{\Sigma}_\alpha\rvert\lvert\mathbf{M}\rvert = 16.52$)
    * 13.2.4 Likelihood and kriging in $\mathcal{O}(nq^2)$ time, compared with the exact computation on $n = 2{,}000$ listings (exponential and Matérn 5/2)
    * 13.2.5 The cost of low rank: the variance deficit, the modified predictive process, why smoothness matters, oversmoothing (Figure 13.2)
* 13.3 Covariance tapering
    * 13.3.1 The idea and why it stays valid (the product rule of §2.4.3), the Wendland taper, a three-listing hand example
    * 13.3.2 Computation: sparse storage, sparse LU (SciPy has no sparse Cholesky), fill-in
    * 13.3.3 What the tables show: accuracy depends on the ratio of the taper range to the correlation range, predictions tolerate tapering better than likelihoods, and sparse LU does not beat dense Cholesky at these sizes
* 13.4 Sparse precision from nearest-neighbor conditioning (Vecchia's approximation)
    * 13.4.1 The factorization of a density: kriging weights as regression coefficients, $\boldsymbol{\Sigma}^{-1} = (\mathbf{I} - \boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I} - \boldsymbol{\Gamma})$, the approximation with $\lvert\mathcal{C}_i\rvert \le k$, its sparse precision, the link to Chapter 8, and $\boldsymbol{\beta}$ by OLS on filtered data
    * 13.4.2 By hand: four points on a line (exact with $k = 1$ because the exponential process is Markov) and the four-apartment block (Kullback-Leibler divergence 0.0584 for $k = 1$, 0.0005 for $k = 2$)
    * 13.4.3 Orderings (random, coordinate, maximin), neighbor sets, and a vectorized implementation; accuracy against the exact likelihood (Figure 13.3)
    * 13.4.4 A city with 20,000 listings: data from the Vecchia model with $k = 30$, a fit with $k = 10$, and the likelihood as $k$ grows
* 13.5 The bridge between Parts II and III: Matérn fields as Gaussian Markov random fields
    * 13.5.1 The result of Lindgren, Rue, and Lindström, stated without the finite-element derivation; the grid precision $(\kappa^2\mathbf{I} + \mathcal{L})^2$; the case $\alpha = 1$ as the proper CAR and Leroux models; the interior of the grid as a SAR model with $\rho = 4/(\kappa^2 + 4)$
    * 13.5.2 A numerical check on a $50 \times 50$ grid: correlations against the Matérn function with $\nu = 1$, boundary variance inflation (about 2 on an edge, 4 in a corner), the SAR identity, and a 40,000-cell factorization (Figure 13.4)
    * 13.5.3 Listings on a grid: the Woodbury identity with a sparse precision, checked against the dense computation
* 13.6 Kronecker products for separable space-time and multivariate grids
    * 13.6.1 Separable covariance $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$, with a $2 \times 2$ by $2 \times 2$ hand example
    * 13.6.2 Four properties with proofs (mixed product, determinant exponents, the vec identity, the consequences for solves and simulation) and a check against dense matrices; 400 sites by 50 weeks
    * 13.6.3 The limits of separability: a nonseparable covariance, the rearrangement and its singular values, and the requirement of a complete rectangle
* 13.7 Walkthrough and Lab 13
    * 13.7.1 A benchmark against the exact computation at $n = 1{,}000$ to $5{,}000$: cost per evaluation and likelihood error for low rank, tapering, and Vecchia
    * 13.7.2 Sizes beyond the exact computation: low rank at $n = 20{,}000$ and $50{,}000$, and a tapered LU at $n = 20{,}000$
    * 13.7.3 Which method for which problem (a decision table)
    * 13.7.4 Figures 13.1 to 13.4

---

## Appendices

### Appendix A: Matrix Algebra and Multivariate Normal Calculus

* A.0 A review of the five tools of Chapter 0, one subsection each: linear transformations of random vectors, quadratic forms and positive semi-definiteness, the multivariate normal density, maximum likelihood, and the normal equations, keyed to the warm-up exercises of §0.7
* A.1 Partitioned (block) matrix inversion and the Schur complement, with both forms of the blocks and the block determinant; the Woodbury identity and the matrix determinant lemma, stated by name and derived from block inversion
* A.2 Vector and matrix derivatives: gradients of linear and quadratic forms, $\dfrac{\partial\log\lvert\boldsymbol{\Sigma}\rvert}{\partial\theta} = \operatorname{tr}\!\left(\boldsymbol{\Sigma}^{-1}\dfrac{\partial\boldsymbol{\Sigma}}{\partial\theta}\right)$, $\dfrac{\partial\boldsymbol{\Sigma}^{-1}}{\partial\theta} = -\boldsymbol{\Sigma}^{-1}\dfrac{\partial\boldsymbol{\Sigma}}{\partial\theta}\boldsymbol{\Sigma}^{-1}$, and $\partial\boldsymbol{\Pi}/\partial\theta_j$
* A.3 Moments of quadratic forms in normal vectors: for symmetric $\mathbf{M}$ and $\mathbf{Y} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, $\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}$, the variance, the trace properties, and the Pitman-Koopmans independence statement
* A.4 Conditional distributions of partitioned multivariate normals, in covariance and precision form (the precision form is derived in §8.1.2, to which this section points)
* A.5 Kronecker products and the $\operatorname{vec}$ operator
* A.6 Facts about matrices used in the text: the spectral theorem for symmetric matrices (orthonormal eigenvectors, trace and determinant as sums and products of eigenvalues, symmetric square roots), existence and uniqueness of the Cholesky factor, idempotent projections and their traces, Lagrange multipliers for equality constraints, the Moore-Penrose pseudo-inverse and generalized determinant, the Neumann series, and the Gaussian integral

### Appendix B: Scientific Python Reference Patterns

* B.1 Symbolic verification of matrix identities and gradients with `sympy.matrices`
* B.2 Stable solvers: `scipy.linalg.cho_factor` and `cho_solve` instead of `numpy.linalg.inv`
* B.3 Sparse matrices (`scipy.sparse.csc_matrix`) and sparse solvers (`spsolve`, `splu`, conjugate gradient)
* B.4 Constrained optimization of profile and REML log-likelihoods with `scipy.optimize.minimize` (L-BFGS-B)
* B.5 The Matérn function with `scipy.special.kv`
* B.6 Plotting with `matplotlib`: point maps, kriged surfaces, variograms, and simulation envelopes

### Appendix C: A Jargon Rosetta Stone

Spatial terminology translated into linear-model terms:

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

The notation guide and the concordance with Schabenberger and Gotway (2005) are not part of this appendix; they are in the book's notation section (`NOTATION.md`).

### Appendix D: Where to Go Next

* Spatial sampling design: choosing locations to minimize kriging variance, which does not depend on the data (§4.2.2)
* Point patterns with continuous marks and mark correlation functions
* Space-time point processes
* Cokriging and multivariate fields; general (non-separable) space-time covariance models
* Non-Gaussian geostatistics and trans-Gaussian kriging beyond the log scale
* Covariance models on the sphere for global data
* Bayesian computation (MCMC, INLA), and alternatives to plug-in prediction that average over uncertainty in the covariance parameters
* The full SPDE approach, including the finite-element mesh for irregular locations
* The Spatial Durbin model (a spatial lag of the covariates in the lag model of Chapter 7)
* The deeper theory behind the validity of covariance functions: Bochner's theorem and the spectral representation of random fields
* Spatial confounding and causal inference
* Geographically weighted regression and spatial machine learning

---

## How Each Chapter Reads

1. **Linear model bridge.** The chapter opens by connecting its spatial problem to a familiar linear-model or GLM equation, translating any domain jargon into standard statistical terms.
2. **Step-by-step matrix derivation.** The estimator, predictor, or test statistic is derived with clear matrix algebra and calculus, showing every intermediate step.
3. **A concrete $n=4$ example.** A miniature configuration of 3 to 5 locations, with explicit numbers, lets the student see the exact entries of $\mathbf{H}, \boldsymbol{\Sigma}, \mathbf{W}, \mathbf{Q}$ and verify the arithmetic by hand.
4. **Algorithm and simulation.** The matrix equations are translated into `NumPy` and `SciPy` code, and a simulation with known parameters shows how the method performs.
5. **Graded exercises.** Analytical derivations (matrix proofs and, where useful, `SymPy` checks), then computational and simulation problems (implementing estimators and running Monte Carlo studies).

