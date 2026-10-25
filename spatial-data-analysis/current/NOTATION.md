# Notation

This section collects the notation used in the book. Every symbol is defined in the text where it first appears; the "First introduced" column tells you where to look. If you meet a symbol you do not recognize, this is the place to check.

---

## Conventions

- **Vectors and matrices.** Bold lowercase letters are vectors ($\mathbf{s}$, $\mathbf{y}$, $\boldsymbol{\beta}$). Bold uppercase letters are matrices ($\mathbf{X}$, $\boldsymbol{\Sigma}$, $\mathbf{Q}$). Italic letters are scalars ($n$, $\sigma^2$, $\phi$).
- **Random and observed.** Uppercase letters denote random quantities and lowercase letters their observed values, as in Dobson and Barnett: $\mathbf{Y}$ is the random response vector and $\mathbf{y}$ is the data.
- **Transpose.** A prime denotes the transpose: $\mathbf{X}'$.
- **Operators.** Expectation, variance, covariance, and trace are set in roman type: $\operatorname{E}[\cdot]$, $\operatorname{Var}(\cdot)$, $\operatorname{Cov}(\cdot, \cdot)$, $\operatorname{tr}(\cdot)$.
- **Chapter at a Glance boxes** preview each chapter and may name symbols that the chapter defines later; the rule that no symbol appears before its first introduction applies from the first section onward.
- **Special vectors and matrices.** $\mathbf{1}$ is a vector of ones and $\mathbf{I}_n$ is the $n \times n$ identity matrix. The size is dropped when it is clear from context.

## The Model Used Throughout

$$
Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s}), \qquad \mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}
$$

An observation is a trend, plus a spatial random effect, plus independent noise. The spatial random effect $u(\mathbf{s})$ has variance $\sigma^2$ and correlation $r(h;\phi)$ between locations a distance $h$ apart. The noise $\varepsilon(\mathbf{s})$ has variance $\tau^2$. At $n$ locations,

$$
\operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n
$$

**A note on $\sigma^2$ and $\tau^2$.** Regression texts write the error variance of the classical linear model as $\sigma^2$. In this book $\sigma^2$ always means the variance of the spatial random effect, so the classical error variance is written $\tau^2$: the classical model is the spatial model with no spatial random effect.

---

## Linear Models and General Tools

| Symbol | Meaning | First introduced |
| :--- | :--- | :--- |
| $n$, $p$ | Number of observations; number of regression coefficients | §0.2, §1.1 |
| $\mathbf{Y}$, $\mathbf{y}$ | Random response vector; its observed value | §0.2 |
| $\mathbf{X}$, $\boldsymbol{\beta}$ | Design matrix ($n \times p$); regression coefficients | §0.4.5 |
| $\hat{\boldsymbol{\beta}}$, $\hat{\boldsymbol{\beta}}_{\text{GLS}}$ | OLS estimate; generalized least squares estimate | §0.4.5 |
| $\mathbf{e}$, $s^2$ | OLS residuals; residual variance estimate $\mathbf{e}'\mathbf{e}/(n - p)$ | §1.1 |
| $\mathbf{P}_{\mathbf{X}}$ | Projection onto the columns of $\mathbf{X}$: $\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ | §0.4.5 |
| $\mathbf{P}_c$ | Centering matrix $\mathbf{I}_n - \tfrac{1}{n}\mathbf{1}\mathbf{1}'$ | §0.4.2 |
| $\mathbf{a}$, $\mathbf{B}$, $\mathbf{M}$ | A generic fixed vector, linear transformation, and symmetric matrix in a quadratic form | §0.4.1, §0.4.2 |
| $\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$, $\mathbf{v}_k$, $\lambda_k$ | Eigendecomposition of a symmetric matrix; eigenvectors (the columns of $\mathbf{U}$); eigenvalues | §1.3.3 |
| $\mathbf{L}$ | A matrix with $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$, usually the Cholesky factor | §0.4.1 (Cholesky: §2.7) |
| $\mathbf{Z}$ | Vector of independent standard normal variables used in simulation | §0.4.1 |
| $n_{\text{eff}}$ | Effective sample size | §0.3.1 |
| $\mathbb{1}(\cdot)$ | Indicator: 1 if the condition holds, 0 otherwise | §10.5.1 |

## Spatial Locations and Covariance

| Symbol | Meaning | First introduced |
| :--- | :--- | :--- |
| $\mathcal{D}$ | The study region, a subset of $\mathbb{R}^2$; its area is $\operatorname{area}(\mathcal{D})$ | §1.2 (area: §4.6.1) |
| $\mathbf{s}$, $\mathbf{s}_i$, $\mathbf{s}_0$ | A location; the $i$th data location; a prediction location | §0.2, §1.2, §4.1 |
| $\mathbf{S}$ | Coordinate matrix ($n \times 2$) with rows $\mathbf{s}_i'$ | §1.5.1 |
| $h_{ij}$, $\mathbf{H}$ | Distance between $\mathbf{s}_i$ and $\mathbf{s}_j$; the matrix of these distances | §1.2, §1.5.1 |
| $\mathbf{h}$ | Lag vector $\mathbf{s}_i - \mathbf{s}_j$ | §2.2 |
| $\mu(\mathbf{s})$, $\mathbf{x}(\mathbf{s})$ | Mean at $\mathbf{s}$; covariates at $\mathbf{s}$ | §0.2 |
| $u(\mathbf{s})$, $\mathbf{u}$ | Spatial random effect | §0.2 |
| $\varepsilon(\mathbf{s})$, $\boldsymbol{\varepsilon}$ | Nugget: measurement error plus variation finer than the sampling spacing | §0.2 (nugget: §2.1.3) |
| $\sigma^2$, $\tau^2$ | Variance of $u$ (partial sill); variance of $\varepsilon$ (nugget) | §0.2 |
| $\phi$ | Range parameter: how quickly correlation decays with distance | §0.2 |
| $r(h;\phi)$, $\mathbf{R}(\phi)$ | Correlation function of $u$; its matrix, $R_{ij} = r(h_{ij};\phi)$ | §0.3, §0.2 |
| $\boldsymbol{\Sigma}$ | Covariance matrix of $\mathbf{Y}$ | §0.2 |
| $C(h)$, $C(0)$ | Covariance between $Y$ at two locations a distance $h$ apart; the variance $\sigma^2 + \tau^2$. From §2.2 also written $C(\mathbf{h})$, $C(\mathbf{0})$ with the lag vector, which depends on direction as well as distance | §1.2 (§2.2) |
| $\gamma(\mathbf{h})$ | Semivariogram | §2.2 |
| $\delta$, $\boldsymbol{\Psi}(\phi,\delta)$ | Noise-to-signal ratio $\tau^2/\sigma^2$; $\mathbf{R}(\phi) + \delta\mathbf{I}_n$, so $\boldsymbol{\Sigma} = \sigma^2\boldsymbol{\Psi}$ | §2.1.4 |
| $\nu$ | Matérn smoothness | §1.6.4 (developed in §2.5.3) |
| $\mathbf{G}$ | Matrix defining geometric anisotropy | §2.6 |
| $\boldsymbol{\theta}$, $\gamma(h;\boldsymbol{\theta})$ | Vector of covariance parameters, such as $(\sigma^2, \tau^2, \phi)'$; the model semivariogram | §3.3 |
| $N(h_k)$, $\lvert N(h_k)\rvert$ | Set of pairs $(i, j)$ with $i < j$ in distance bin $k$; number of such pairs | §3.1.1 |
| $h_k$ | Average distance of the pairs in bin $k$ (the bin's lag) | §3.1.1 |
| $\hat{\gamma}(h_k)$ | Matheron's estimate of the semivariogram in bin $k$ | §3.1.1 |

## Prediction and Likelihood

| Symbol | Meaning | First introduced |
| :--- | :--- | :--- |
| $\tilde{Y}(\mathbf{s}_0)$, $\tilde{Y}(B)$ | Noise-free target $\mu(\mathbf{s}_0) + u(\mathbf{s}_0)$; its average over a block $B$ | §4.1, §4.6 |
| $\lambda_0$, $\boldsymbol{\lambda}$ | Constant and weights of a linear predictor $\lambda_0 + \boldsymbol{\lambda}'\mathbf{Y}$ | §4.1 |
| $m$, $\mathbf{m}$ | Lagrange multiplier(s) in the kriging equations | §4.3 |
| $\mathbf{c}_0$, $\mathbf{c}_B$ | Covariances between $\mathbf{Y}$ and the prediction target | §4.1, §4.6 |
| $\boldsymbol{\Pi}$ | Residual precision $\boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}$ | §4.7 |
| $\ell$, $\ell_{\text{REML}}$ | Log-likelihood (first used in §0.4.4); restricted log-likelihood | §0.4.4, §5.3 |
| $\mathbf{T}$ | Error-contrast matrix for REML, with $\mathbf{T}'\mathbf{X} = \mathbf{0}$ | §5.3 |
| $\mathcal{I}(\boldsymbol{\theta})$ | Fisher information | §5.4 |
| $\mu_0$, $\mathbf{x}_0$ | Mean and covariates at the prediction location $\mathbf{s}_0$ | §4.1.1 |
| $\sigma^2_{\text{SK}}$, $\sigma^2_{\text{OK}}$, $\sigma^2_{\text{UK}}$ | Simple, ordinary, and universal kriging variances | §4.2.2, §4.3.3, §4.4.4 |
| $\hat{\mu}_{\text{GLS}}$ | GLS estimate of a constant mean | §1.4.4 |
| $\mathbf{K}$ | Kriging matrix $\begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix}$ | §4.3.2 |
| $\sigma^2_B$, $\mathbf{x}_B$ | Variance of the block average of $u$; block-averaged covariates | §4.6.1 |
| $\dot{\boldsymbol{\Sigma}}_j$ | Derivative $\partial\boldsymbol{\Sigma}/\partial\theta_j$ | §5.4.2 |
| $\hat{\ell}$, AIC, BIC | Maximized log-likelihood; Akaike and Bayesian information criteria | §5.5.1 |
| $\mathbf{r}^*$ | Decorrelated residuals $\mathbf{L}^{-1}(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})$ | §5.6.1 |

## Areal Data

| Symbol | Meaning | First introduced |
| :--- | :--- | :--- |
| $\mathbf{A}$ | Binary adjacency matrix: $A_{ij} = 1$ if regions $i$ and $j$ are neighbors | §1.5.2 |
| $\mathbf{D}$, $d_i$ | Diagonal degree matrix; number of neighbors of region $i$ | §6.3.1 |
| $\mathbf{W}$, $w_{ij}$, $S_0$ | Spatial weights matrix; its entries; their sum | §6.2 ($S_0$: §6.3.2) |
| $\mathbf{y}_c$ | Centered data $\mathbf{P}_c\mathbf{y}$ | §6.4 |
| $I$ | Moran's $I$ | §6.4 |
| $\rho$ | Autoregressive parameter in SAR, CAR, and Leroux models (its scale differs between models); also the BYM2 mixing parameter, the share of the variance that is spatial | §7.1, §8.2, §9.3 |
| $\mathbf{Q}$ | Precision matrix (inverse covariance) of a Gaussian Markov random field | §7.2.3 |
| $c_{ij}$, $\sigma_i^2$ | CAR conditional regression coefficients and conditional variances | §8.1.1 |
| $\mathbf{u}^{\text{icar}}$, $\mathbf{v}$ | Structured and unstructured components of a BYM random effect | §9.3.3 |
| $E_i$ | Expected count in region $i$ | §9.2.1 |
| $g(\cdot)$, $\boldsymbol{\eta}$, $\boldsymbol{\mu}$ | GLM link function; linear predictor; mean | §9.1.2 |
| $\mathbf{z}$, $\mathbf{V}$ | IRLS working response; diagonal working weights (Dobson and Barnett write $\mathbf{W}$) | §9.1.5, §9.1.4 |
| $\mathbf{W}_s$ | Symmetrized weights $\tfrac{1}{2}(\mathbf{W} + \mathbf{W}')$ | §6.4.2 |
| $\mathbf{M}$ | Central matrix of Moran's $I$, $\mathbf{P}_c\mathbf{W}_s\mathbf{P}_c$ | §6.4.2 |
| $I_e$, $\mathbf{M}_{\mathbf{X}}$ | Moran's $I$ of OLS residuals; $(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{W}_s(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})$ | §6.6.1 |
| $C$ | Geary's $C$ (non-bold scalar; not the covariance function $C(h)$) | §6.7.1 |
| $\mathbf{D}_s$ | Degree matrix of the symmetrized weights, $\operatorname{diag}(\mathbf{W}_s\mathbf{1}_n)$ | §6.7.1 |
| $I_i$ | Local Moran indicator of region $i$ | §6.7.2 |
| $\boldsymbol{\Phi}(\rho)$ | Feedback matrix $(\mathbf{I}_n - \rho\mathbf{W})^{-1}$ | §7.2.2 |
| $\mathbf{Y}^*$, $\mathbf{X}^*$ | Spatially filtered data $(\mathbf{I}_n - \rho\mathbf{W})\mathbf{Y}$ and $(\mathbf{I}_n - \rho\mathbf{W})\mathbf{X}$ | §7.2.4 |
| $T_W$, $J$ | Constants in the LM statistics: $\operatorname{tr}(\mathbf{W}^2 + \mathbf{W}'\mathbf{W})$ and the information for the lag parameter | §7.6.2 |
| $g_{\text{err}}$, $g_{\text{lag}}$ | Scores for the error and lag parameters at the OLS fit | §7.6.2 |
| $\text{LM}_{\text{err}}$, $\text{LM}_{\text{lag}}$, $\text{RLM}_{\text{err}}$, $\text{RLM}_{\text{lag}}$ | Lagrange multiplier statistics and their robust versions | §7.6.2 |
| $M_{\text{dir}}$, $M_{\text{ind}}$, $M_{\text{tot}}$ | Direct, indirect, and total impacts of a covariate | §7.7.2 |
| $\tilde{\mathbf{A}}$, $\tilde{\lambda}_k$ | Normalized adjacency $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ and its eigenvalues (those of $\mathbf{W}$) | §6.3.3 (used in Chapter 8) |
| $\mathbf{O}$ | Edge-by-region incidence matrix, with $\mathbf{O}'\mathbf{O} = \mathbf{D} - \mathbf{A}$ | §8.3.2 |
| $\xi_k$ | Eigenvalues of the graph Laplacian $\mathbf{D} - \mathbf{A}$ (zero for each connected component) | §8.3.4 |
| $\lvert\cdot\rvert_*$, $^{+}$ | Generalized determinant (product of nonzero eigenvalues); Moore-Penrose pseudo-inverse | §8.3.4 |
| $\theta_i$, $\text{SMR}_i$ | Relative risk $\operatorname{E}[Y_i \mid u_i]/E_i$; standardized mortality ratio $Y_i/E_i$ | §9.2.1 |
| $\mathbf{u}^{\star}$, $c$, $\boldsymbol{\Sigma}^{\star}$ | Unscaled ICAR effect; BYM2 scaling constant (geometric mean of the marginal variances of $\boldsymbol{\Sigma}^{\star}$); constrained generalized inverse of $\mathbf{D} - \mathbf{A}$ | §9.3.3 (scaling constant: §9.3.4) |
| $\mathbf{o}$ | Offset vector $\log\mathbf{E}$ | §9.4.1 |
| $\ell_{\text{pen}}$ | Penalized log-likelihood | §9.4.1 |
| $\boldsymbol{\Sigma}_z$, $\boldsymbol{\Pi}_z$ | Working covariance $\mathbf{V}^{-1} + \mathbf{Q}^{-1}$ and its residual precision | §9.4.4 |
| $\hat{\mathbf{u}}$ | Mode of the random effects for fixed $\boldsymbol{\beta}$ and hyperparameters | §9.4.2 (used in §9.5.1) |

## Point Patterns

| Symbol | Meaning | First introduced |
| :--- | :--- | :--- |
| $N(B)$ | Number of events in a set $B$ | §10.2.1 |
| $\lambda(\mathbf{s})$ | Intensity: expected number of events per unit area at $\mathbf{s}$ | §10.1 |
| $n_q$ | Number of quadrats | §10.2.3 |
| $\hat{\lambda}(\mathbf{s})$, $\mathcal{K}_\psi$, $\psi$ | Kernel intensity estimate; smoothing kernel; bandwidth | §10.4.1 |
| $d_i^{\text{nn}}$ | Distance from event $i$ to its nearest neighbor | §10.5.1 |
| $b(\mathbf{s}, r)$ | Disc of radius $r$ centered at $\mathbf{s}$ | §10.5.2 |
| $G(r)$, $F(r)$, $J(r)$ | Nearest-neighbor, empty-space, and $J$ functions | §10.5.1, §10.5.3, §10.5.4 |
| $K(r)$, $L(r)$, $g(r)$ | Ripley's $K$; Besag's $L = \sqrt{K/\pi}$; pair correlation function | §10.6.1, §10.6.3, §10.6.4 |
| $\omega_{ij}$ | Edge-correction weight for the pair $(i, j)$: Ripley's isotropic weight in Chapter 10, the translation weight $1/[(1 - \lvert\Delta x\rvert)(1 - \lvert\Delta y\rvert)]$ for the unit square in Chapters 11 and 12 | §10.6.2 |
| $\lambda_1(\mathbf{s})$, $\lambda_0(\mathbf{s})$, $\operatorname{RR}(\mathbf{s})$ | Case and control intensities; spatial relative risk $\lambda_1/\lambda_0$ | §10.7.1, §10.7.3 |
| $K_{12}(r)$, $G_{12}(r)$ | Cross-type $K$ and $G$ functions | §10.7.2 |
| $\mathbf{s}_k^{*}$, $a_k$, $\mathbb{1}_k$, $n_d$ | Quadrature points; their area weights; indicators that a pixel or quadrature point holds an event; number of dummy points | §11.3.1 ($\mathbb{1}_k$: §11.2.1) |
| $\lambda_{\text{pop}}(\mathbf{s})$ | Baseline population intensity used as an offset | §11.4.1 |
| $n_w$, $e_w$ | Observed and expected counts in scan window $w$ | §11.6.1 |
| $K_{\text{inhom}}(r)$ | Inhomogeneous $K$-function | §11.7.2 |
| $\mathbf{U}(\boldsymbol{\beta})$ | Score of the point process log-likelihood, $\sum_i\mathbf{x}(\mathbf{s}_i) - \int\mathbf{x}\lambda$ | §11.2.2 |
| $\mathcal{I}(\boldsymbol{\beta})$ | Point process information $\int\mathbf{x}\mathbf{x}'\lambda$ (extends $\mathcal{I}(\boldsymbol{\theta})$ of §5.4) | §11.2.2 |
| $\boldsymbol{\Xi}$ | Interaction term in the variance of the score, $\int\!\!\int\mathbf{x}(\mathbf{s})\mathbf{x}(\mathbf{t})'\lambda(\mathbf{s})\lambda(\mathbf{t})(g - 1)\,d\mathbf{s}\,d\mathbf{t}$ | §11.8.3 |
| $\lambda_{\text{bg}}(\mathbf{s})$ | Baseline intensity in the scan statistic | §11.6.1 |
| $\operatorname{LLR}_w$ | Log-likelihood ratio of scan window $w$ | §11.6.2 |
| $\operatorname{CV}^2$ | Squared coefficient of variation of the intensity over the region | §11.7.1 |
| $\mathcal{N}_c(\mathbf{h})$ | Density of the difference of two offspring displacements, $\exp(-\lVert\mathbf{h}\rVert^2/(4\sigma_c^2))/(4\pi\sigma_c^2)$ | §11.8.2 |
| $K_{\text{Thomas}}(r)$ | $K$-function of the Thomas process | §11.8.4 |
| $\mathbf{U}_b$, $J$ | Score contribution of block $b$; number of blocks per side | §11.8.4 |
| $\kappa$, $\mu_c$, $\sigma_c$ | Parent intensity; mean offspring per parent; offspring spread | §11.8.1 (developed in §12.2.1) |
| $r_0$ | Interaction radius (hard core, Strauss) | §12.4.1 |
| $t(\mathbf{u}, \mathbf{S})$, $\zeta$ | Number of points within $r_0$ of $\mathbf{u}$; Strauss interaction parameter | §12.4.3 |
| $\gamma$, $b$ | Strauss penalty per close pair, $\zeta = \log\gamma \le 0$; Strauss base intensity (so $\log b$ is the intercept) | §12.4.3 |
| $\lambda_{\text{par}}$ | Parent intensity in Matérn hard-core processes | §12.4.1 |
| $\Lambda(\mathbf{s})$ | Random intensity surface of a Cox process | §12.3.1 |
| $C_u(h)$ | Covariance function of the spatial random effect $u$ | §12.3.1 |
| $f$, $\mathbf{d}$ | Dispersal density of offspring and a displacement | §12.2.1 |

## Large Data

| Symbol | Meaning | First introduced |
| :--- | :--- | :--- |
| $\mathbf{B}$, $\boldsymbol{\alpha}$, $\boldsymbol{\Sigma}_\alpha$, $q$ | Basis matrix; basis coefficients; their covariance; number of knots (bold $\boldsymbol{\alpha}$ is first used in §4.5.3 for kernel ridge coefficients) | §13.2.1 |
| $\boldsymbol{\kappa}_j$, $\mathbf{R}_K$, $\mathbf{R}_{nK}$ | Location of knot $j$ (bold, subscripted); correlation matrix of the knots; $n \times q$ correlations between listings and knots | §13.2.1 |
| $\boldsymbol{\Omega}$ | Diagonal part $\operatorname{diag}(d_1, \dots, d_n)$ of a diagonal-plus-low-rank covariance matrix | §13.2.2 |
| $\mathbf{b}_0$ | Kriging weights of a new location on the knots, $\mathbf{R}_K^{-1}\mathbf{r}(\mathbf{s}_0)$ | §13.2.4 |
| $\mathcal{T}_\theta(h)$, $\theta$, $C_\theta(h)$ | Compactly supported (Wendland) taper with range $\theta$; tapered covariance function | §13.3.1 |
| $\mathcal{C}_i$ | Conditioning set of location $i$ in Vecchia's approximation | §13.4.1 |
| $\boldsymbol{\Gamma}$, $F_i$, $\mathbf{F}$ | Strictly lower triangular matrix of kriging weights on earlier observations; conditional variances and their diagonal matrix | §13.4.1 |
| $\widetilde{\mathbf{Q}}$ | Precision matrix of the Vecchia model, $(\mathbf{I} - \boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I} - \boldsymbol{\Gamma})$ | §13.4.1 |
| $\kappa$, $\alpha$, $\nu$, $\mathcal{W}$, $\nabla^2$ | Inverse range, exponent, and smoothness of the SPDE $(\kappa^2 - \nabla^2)^{\alpha/2}u = \mathcal{W}$ with $\nu = \alpha - d/2$; Gaussian white noise; Laplace operator | §13.5.1 |
| $\mathbf{E}$, $n_g$ | Listing-to-cell indicator matrix ($n \times n_g$); number of grid cells | §13.5.3 |
| $\boldsymbol{\Sigma}_T$, $\boldsymbol{\Sigma}_S$, $n_T$, $n_S$ | Temporal and spatial factors of a separable covariance, and their sizes | §13.6.1 ($n_T$, $n_S$ appear first in §13.1.3) |
| $\otimes$, $\operatorname{vec}$, $\mathbf{L}_T$, $\mathbf{L}_S$, $C_T$, $C_S$ | Kronecker product; column-stacking operator; Cholesky factors of $\boldsymbol{\Sigma}_T$ and $\boldsymbol{\Sigma}_S$; temporal and spatial covariance functions | §13.6.1, §13.6.2 |

---

## Local Symbols

These are defined where they are used. Most appear in one chapter only; a few are reused in a neighboring chapter with the same meaning, and the chapter column says where.

| Symbol | Meaning | Where |
| :--- | :--- | :--- |
| $\theta$ | Common risk in the small-population simulation | §0.3.3 |
| $\text{pop}_i$ | Population of region $i$ | §0.3.3 |
| $a$, $b$ | Correlations $e^{-1}$ and $e^{-\sqrt{2}}$ in the four-apartment example | §1.3.2 |
| $\mathbf{x}_c$ | A covariate centered at its mean | §1.3.4 |
| $\mathbf{B}_{\text{GLS}}$ | The matrix that produces the GLS estimate, $\hat{\boldsymbol{\beta}}_{\text{GLS}} = \mathbf{B}_{\text{GLS}}\mathbf{Y}$ | §1.3.6 |
| $r$ | Correlation between two locations in a two-location example (first used in §0.4.3 and the Chapter 0 exercises) | §0.4.3, §1.3.3, §2.7.1 |
| $\mathbf{Z}_b$ | Random-effects design matrix of a linear mixed model (bold $\mathbf{Z}$ with a subscript; bold $\mathbf{Z}$ without one is the standard normal vector) | §2.1.5 |
| $e_{\text{true}}(\mathbf{s})$, $\mathbf{e}_{\text{true}}$ | The true error process $u + \varepsilon$ and its vector at the data locations (not the OLS residuals $\mathbf{e}$) | §3.4.1, §3.4.3 |
| $Y_t$, $Z_t$ | A random walk and its independent steps | §2.2.3 |
| $\Gamma(\cdot)$, $K_\nu$ | Gamma function; modified Bessel function of the second kind, in the Matérn formula | §2.5.3 |
| $x$ | Scaled distance $h/\phi$ | §2.5.3 |
| $\mathbf{G}^{1/2}$ | Symmetric square root of $\mathbf{G}$ | §2.6 |
| $L_{ij}$ | Entries of the Cholesky factor $\mathbf{L}$ | §2.7.1 |
| $\mathbf{L}_u$, $\mathbf{Z}_1$, $\mathbf{Z}_2$ | Cholesky factor of $\sigma^2\mathbf{R}$, and the two independent normal vectors used to simulate $\mathbf{u}$ and $\boldsymbol{\varepsilon}$ | §2.7.3 |
| $s_{\text{east}}$ | East coordinate of a location | §2.8.2 |
| $\vartheta_{ij}$ | Direction of the lag vector of pair $(i, j)$, measured from east | §3.1.4 |
| $\bar{\gamma}(h_k)$, $\bar{a}_k$ | Cressie-Hawkins estimate; average of $\lvert Y_i - Y_j\rvert^{1/2}$ in bin $k$ | §3.2.2 |
| $m$ | $\operatorname{E}\lvert Z\rvert^{1/2}$ in the Cressie-Hawkins derivation | §3.2.2 |
| $d$ | Deviation of $\bar{a}_k$ from its mean | §3.2.3 |
| $\gamma_e$ | Semivariogram of the error process $u + \varepsilon$ | §3.4.1 |
| $\Delta_{ij}$ | East-west separation of a pair | §3.4.1 |
| $\boldsymbol{\Sigma}_e$ | Covariance matrix of OLS residuals | §3.4.3 |
| $\Delta\mu_{ij}$ | Mean of $Y_i - Y_j$ when the mean is not constant | Chapter 3 exercises |
| $a_0$, $q_0$ | $\mathbf{1}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$ and $\mathbf{1}'\boldsymbol{\Sigma}^{-1}\mathbf{1}$ | §4.3.3 |
| $\mathbf{M}_{11}$, $\mathbf{M}_{12}$, $\mathbf{M}_{21}$, $\mathbf{M}_{22}$, $\mathbf{F}$ | Blocks of a partitioned matrix and its Schur complement | §4.3.4 |
| $\mathbf{d}_0$ | $\mathbf{x}_0 - \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$ | §4.4.4 |
| $\mathbf{e}_i$ | Standard basis vector with 1 in position $i$ | §4.5.1 |
| $\mathbf{C}_u$, $\boldsymbol{\alpha}$ | $\sigma^2\mathbf{R}$; kernel ridge coefficients | §4.5.3 |
| $\langle\cdot,\cdot\rangle$ | Inner product | §4.2.4 |
| $V$, $a$, $v$ | A normal variable, its mean, and its variance, in the lognormal mean | §4.5.4 |
| $n_B$, $\mathbf{s}^B_k$ | Number and locations of grid points inside a block | §4.6.1 |
| $D$, $D_k$, $v_k$ | Prediction errors and point kriging variances | §4.1.2, §4.6.3 |
| $\text{LOO}_i$ | Leave-one-out residual | §4.7.3 |
| $\mathbf{b}$, $\mathbf{K}_{-1}$, $f$, $\mathbf{y}^{\circ}$ | Pieces of the optional proof of the leave-one-out shortcut | §4.7.4 |
| $\mathrm{RSS}_\Psi(\phi, \delta)$ | Generalized residual sum of squares $(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})'\boldsymbol{\Psi}^{-1}(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})$ | §5.2.2 |
| $\ell_p$, $\ell_{\text{REML},p}$ | Profile ML and profile REML log-likelihoods | §5.2.2, §5.3.3 |
| $\boldsymbol{\Pi}_\Psi$ | Residual precision built from $\boldsymbol{\Psi}$ | §5.3.1 |
| $\mathbf{F}$ | Invertible matrix relating two contrast matrices | §5.3.3 |
| $\mathbf{J}_1$, $\mathbf{J}_2$, $\mathbf{M}_2$ | Matrices in the optional proofs | §5.3.4 |
| $\ddot{\boldsymbol{\Sigma}}_{jk}$ | Second derivative of $\boldsymbol{\Sigma}$ | §5.4.6 |
| $k$ | Number of estimated parameters in AIC and BIC | §5.5.1 |
| $t$ | Small perturbation of a parameter in the derivative of a log-determinant | §5.4.2 |

| $\mathbf{N}$ | A generic invertible matrix (§6.3.3); the stacked matrix with $\mathbf{N}'\mathbf{N} = \mathbf{D} - \rho\mathbf{A}$ (§8.4.3); the Newton matrix of $\ell_{\text{pen}}$ (§9.4.2) | Chapters 6, 8, 9 |
| $v_a$, $v_b$ | Variances across units of the aggregated local and regional parts | §6.1.1 |
| $z_k$ | An independent standard normal $\mathbf{v}_k'\mathbf{Z}$ in a quadratic form (§6.5.1); also the independent standard normals of the ICAR simulation (§8.3.4, §9.3.7) | §6.5.1 |
| $q_1$, $q_2$ | Numerator $\mathbf{Y}'\mathbf{M}\mathbf{Y}$ and denominator $\mathbf{Y}'\mathbf{P}_c\mathbf{Y}$ of $I$ | §6.5.2 |
| $\mathbf{U}_{\perp}$, $\mathbf{w}$, $\mathbf{O}$ | Orthonormal basis of vectors perpendicular to $\mathbf{1}$; $\mathbf{U}_{\perp}'\mathbf{Y}$; a rotation matrix (§6.5.2) | §6.5.2 |
| $S_1$, $S_2$ | Sums of weights in the textbook variance of $I$ | §6.5.6 |
| $r_i$, $w_{\cdot i}$ | Row and column sums of $\mathbf{W}$ | §6.7.1 |
| $m_2$ | $\sum_i y_{c,i}^2/n$ | §6.7.2 |
| $B$ | Number of random rearrangements in a permutation test | §6.8.1 |
| $\mathbf{e}_\pi$ | A rearranged residual vector | §6.8.5 |
| $\mathbf{X}_L$ | $[\mathbf{W}\mathbf{Y},\ \mathbf{X}]$, the regressors when the lag is included in OLS | §7.3.3 |
| $\mathbf{e}_y$, $\mathbf{e}_{Wy}$, $\hat{\boldsymbol{\beta}}_y$, $\hat{\boldsymbol{\beta}}_{Wy}$ | OLS residuals and coefficients from regressing $\mathbf{y}$ and $\mathbf{W}\mathbf{y}$ on $\mathbf{X}$ | §7.5.3 |
| $\boldsymbol{\varepsilon}_0$ | True innovations at the null in the LM derivation | §7.6.4 |
| $\mathbf{C}$, $\boldsymbol{\Sigma}_c$ | Matrix of the $c_{ij}$; diagonal matrix of conditional variances $\sigma_i^2$ | §8.1.3 |
| $\mathbf{g}$ | A column of $\boldsymbol{\Sigma}$ in the by-hand solve of the four-region map | §8.2.2 |
| $\hat{\boldsymbol{\beta}}(\rho)$, $\hat{\sigma}^2(\rho)$, $\mathrm{RSS}(\rho)$, $\ell_p(\rho)$ | Profile quantities of the CAR likelihood | §8.4.1 |
| $\mathbf{P}_{\text{comp}}$, $\mathbf{1}_k$, $n_k$ | Projection onto sum-to-zero within each connected component; component indicator and size | §8.3.4 |
| $t_k$ | Coordinates of $\mathbf{u}$ in the Laplacian eigenbasis | §8.3.5 |
| $\vartheta$, $b(\vartheta)$, $f_0(y)$ | Natural parameter; cumulant function; base function of the exponential family | §9.1.1 |
| $\mathbf{r}$ | Residuals on the linear-predictor scale, $r_i = (y_i - \mu_i)\,\partial\eta_i/\partial\mu_i$ | §9.1.5 |
| $s_P^2$ | Pearson dispersion | §9.2.3 |
| $a$, $w_i$, $\text{Gamma}(a, b)$ | Gamma shape (rate $b$); empirical Bayes shrinkage weight $E_i/(E_i + a)$ | §9.3.1 |
| $\mathcal{L}$ | The graph Laplacian $\mathbf{D} - \mathbf{A}$ written as one symbol | §9.3.7 |
| $\varsigma_k$ | Eigenvalues of $\operatorname{Var}(\mathbf{u})$ in each model (BYM, BYM2, Leroux); the Laplacian eigenvalues of §8.3.4 keep the symbol $\xi_k$ | §9.3.6 |
| $\mathcal{E}_\psi(\mathbf{s})$, $\Phi$, $\operatorname{LCV}(\psi)$ | Kernel mass inside the window; standard normal distribution function; likelihood cross-validation score | §10.4.2, §10.4.3 |
| $\hat{G}_{\text{KM}}$, $\hat{G}_{\text{H}}$, $b_i$, $\mathcal{D}_{\ominus d}$ | Kaplan-Meier and Hanisch estimates of $G$; distance from event $i$ to the boundary; the window eroded by $d$ | §10.8 |
| $f_{ij}$ | Fraction of the circle of radius $h_{ij}$ about $\mathbf{s}_i$ inside the window, so $\omega_{ij} = 1/f_{ij}$ | §10.6.2 |
| $n_k$, $\bar{n}$, $X^2$ | Quadrat counts, their mean, and Pearson's statistic | §10.2.3 |
| $N_B$, $z$, $p$ | Pixels in $B$; argument of a probability generating function; Bernoulli probability | §10.2.1 |
| $m$ | Grid size (§10.2.1); number of simulations (§10.9) | Chapter 10 |
| $n_g$, $\mathbf{q}_k$ | Number of grid or test points; test points in the estimator of $F$ | §10.4.1, §10.5.3 |
| $\mathbf{u}$ | A dummy location in kernel integrals (§10.4); a generic location (§12.4) | Chapters 10, 12 |
| $\mathcal{K}_b^{(1)}$, $b$ | One-dimensional Gaussian kernel and its bandwidth for smoothing pair distances | §10.6.4 |
| $(\Delta_1, \Delta_2)$, $v$, $\theta$ | Components of the difference of two uniform points; polar radius; polar angle | §10.8.1 |
| $t_i$, $\delta_i$ | Censored time $\min(d_i^{\text{nn}}, b_i)$ and the indicator that it is observed | §10.8.2 |
| $\alpha_L$, $\alpha_R$, $\alpha_B$, $\alpha_T$, $a_L$, $a_R$ | Half-angles of the arcs cut off by the four edges; distances to the left and right edges | §10.8.3 |
| $T$ | Maximum-deviation statistic $\max_r\lvert\hat{L}(r) - r\rvert$ | §10.9.2 |
| $n_1$, $n_2$, $n_0$ | Numbers of type-1 and type-2 events; numbers of cases and controls ($n_1$, $n_0$) | §10.7.2, §11.5.1 |
| $M$, $c_M$, $p_j$, $\mathbb{1}_j$, $\mathbf{s}_{(i)}$ | Number of pixels; pixel area; Bernoulli probability, indicator, and center of the pixel of event $i$ | §11.2.1 |
| $y_k$, $v_k$, $z_k$, $\mu_k$, $\eta_k$, $o_k$ | Berman-Turner pseudo-response, working weight, working response, mean, linear predictor, and offset at quadrature point $k$ | §11.3, §11.4.1 |
| $\text{bump}_h(\mathbf{s})$, $h$ | Gaussian bump around the plant and its reach | §11.4.3 |
| $\kappa_1$, $\kappa_0$, $\theta(\mathbf{s})$ | Constants for how many cases and controls were collected; excess risk | §11.5.2 |
| $T$, $f$ (Chapter 11) | A sum $\sum_if(\mathbf{s}_i)$ over events and the summed function; offspring count $M$ in §11.8.2 | §11.8 |
| $R$ | Disc radius in the Matérn cluster process | §12.2.2 |
| $\mathbf{c}$, $q(\mathbf{c})$ | Parent location; test function in Campbell's formula | §12.2.1 |
| $\rho^{(2)}(\mathbf{s}, \mathbf{s}')$, $f_2(\mathbf{h})$, $\mathbf{D}_1 - \mathbf{D}_2$ | Second-order product density; density of the difference of two independent displacements; the difference itself | §12.2.4 |
| $P_{\text{sib}}(r)$, $z$ | Distribution function of the distance between two siblings; $z = r/(2R)$ | §12.2.7 |
| $c$, $D(\kappa, \sigma_c)$ | Power in the minimum contrast (usually $1/4$); the minimum contrast criterion | §12.2.5 |
| $\operatorname{pairs}(\mathbf{S})$, $Z(b, \gamma)$ | Number of pairs within $r_0$; normalizing constant of the Strauss density | §12.4.3 |
| $A_0$, $I_c$ | Area of a small cell; indicator that cell $c$ has a point | §12.4.5 |
| $R(\mathbf{s})$ | Smoothed residual field | §12.5.1 |
| $\bar{K}$ | Mean of simulated $\hat{K}^{1/4}$ values in a Monte Carlo test | §12.5.3 |
| $\mathbf{M}$ (§13.2.2) | The $q \times q$ matrix $\boldsymbol{\Sigma}_\alpha^{-1} + \mathbf{B}'\boldsymbol{\Omega}^{-1}\mathbf{B}$ in the Woodbury identity | §13.2.2 |
| $\boldsymbol{\Sigma}_{<i}$, $\boldsymbol{\Sigma}_{<i,i}$ | Covariances among earlier observations, and between them and observation $i$ | §13.4.1 |
| $r_1$, $\Delta$ | Correlation between neighbors on a transect; spacing between them | §1.4.2 |
| $\boldsymbol{\Delta}$ | Difference between a linear unbiased estimator and GLS, in the proof of Aitken's theorem | §1.3.6 |
| $R_E$, $\text{lat}$, $\text{lon}$ | Earth's mean radius (6371 km); latitude; longitude | §1.6.2 |
| $d_{\text{gc}}$, $d_{\text{ch}}$ | Great-circle and chordal distance | §1.6.3 |

## Symbols With More Than One Use

A few symbols follow strong conventions in different parts of the literature. Typography or context keeps them apart.

- **$\lambda$.** Bold $\boldsymbol{\lambda}$ and $\lambda_0$ are kriging weights (Part II); $\lambda_k$ are eigenvalues; $\lambda(\mathbf{s})$ is a point process intensity (Part IV).
- **$L$ and $G$.** The functions $L(r)$ and $G(r)$ in Part IV; the bold matrices $\mathbf{L}$ (Cholesky factor) and $\mathbf{G}$ (anisotropy).
- **$K$.** $K_\nu$ with a subscript is the Bessel function in the Matérn formula (Chapter 2); $K(r)$ is Ripley's $K$-function (Part IV).
- **$g$.** The GLM link $g(\cdot)$ in Chapter 9; the pair correlation function $g(r)$ in Part IV.
- **$\mathbf{M}$ and $\mathbf{B}$.** Generic matrices in Chapters 0 and 1 ($\mathbf{M}$ in quadratic forms, $\mathbf{B}$ for linear transformations). Later, $\mathbf{M}$ is the matrix in Moran's $I$ (Chapter 6) and $\mathbf{B}$ is a basis matrix (Chapter 13).
- **$\sigma^2$.** Always the variance scale of the spatial random effect: the partial sill in geostatistics, and the innovation variance in SAR and CAR models, where it is not the marginal variance.
- **$\boldsymbol{\mu}$.** The mean, whether the mean function $\mu(\mathbf{s})$ or the GLM mean vector.
- **Bold $\mathbf{K}$.** The kriging matrix (Chapter 4). The non-bold $K(r)$ and $K_\nu$ are Ripley's function and the Bessel function.
- **$\mathbf{e}$.** The OLS residual vector (Chapter 1 onward); $\mathbf{e}_i$ with a subscript is a standard basis vector (§4.5.1 only).
- **$\boldsymbol{\alpha}$.** Kernel ridge coefficients in §4.5.3 (first use); basis coefficients in Chapter 13.
- **$m$.** The Lagrange multiplier in kriging (Chapter 4); a constant $\operatorname{E}\lvert Z\rvert^{1/2}$ in the Cressie-Hawkins derivation (§3.2.2 only).
- **$\Delta$.** Transect spacing in §1.4.2; east-west separation $\Delta_{ij}$ in §3.4.1; both local.
- **$\mathbf{F}$.** A Schur complement in §4.3.4 and a contrast-relating matrix in §5.3.3; both local.
- **Bold $\mathbf{Z}$ and $\mathbf{Z}_b$.** Bold $\mathbf{Z}$ is always the vector of independent standard normals used in simulation; the mixed-model design matrix of §2.1.5 is $\mathbf{Z}_b$.
- **$\xi$, $\varsigma$, $z$.** $\xi_k$ are the eigenvalues of the graph Laplacian (Chapters 8 and 9), $\varsigma_k$ the eigenvalues of $\operatorname{Var}(\mathbf{u})$ (§9.3.6), and $z_k$ independent standard normals (Chapters 6, 8, 9). In Chapter 6, italic $z$ without a subscript is a standardized test statistic, and bold $\mathbf{z}$ is the IRLS working response (Chapter 9).
- **$J$.** The LM information for the lag parameter (§7.6.2); the number of blocks per side (§11.8.4); the $J$-function (§10.5.4); the proof matrices $\mathbf{J}_1$, $\mathbf{J}_2$ (§5.3.4).
- **$c$.** The BYM2 scaling constant (§9.3.4); the minimum-contrast power (§12.2.5); and $c_{ij}$ (CAR coefficients), $c_i$ no longer used for column sums (now $w_{\cdot i}$).
- **$r_0$.** The interaction radius (§12.4.1); also a rate multiplier in Exercise 8 of Chapter 11.
- **$k$.** The number of estimated parameters in AIC and BIC (§5.5.1); the number of neighbors in Vecchia's approximation (§13.4) and in nearest-neighbor weights (§6.2.2).
- **$\theta$.** The common risk of §0.3.3; the relative risk $\theta_i$ of §9.2.1; the excess risk $\theta(\mathbf{s})$ of §11.5.2; the taper range of §13.3.1; and the parameter vector $\boldsymbol{\theta}$.
- **Bold $\boldsymbol{\omega}$** (three uses in Chapter 13) is local to the passages that define it.
- **$f$.** The dispersal density of §12.2.1; the Strauss density $f(\mathbf{S})$ of §12.4.3.
- **$\nabla^2$.** The Hessian of the log-likelihood (§5.4.3); the Laplace operator (§13.5.1).
- **$\mathbf{C}_u$ and $C_u(h)$.** The matrix $\sigma^2\mathbf{R}$ (§4.5.3); the covariance function of $u$ (§12.3.1).
- **Block matrices.** $\mathbf{M}_{11}, \dots, \mathbf{M}_{22}$ (§4.3.4) and $\mathbf{M}_2$ (§5.3.4) are generic blocks, distinct from Moran's $\mathbf{M}$ (Chapter 6).
- **$\delta$.** The noise-to-signal ratio (§2.1.4); the censoring indicator $\delta_i$ (§10.8.2).
- **$\mathbf{O}$.** The edge-by-region incidence matrix (§8.3.2); a rotation matrix (§6.5.2).
- **Relative risk.** In Chapter 9, the mean count over the expected count; in Chapters 10 and 11, the ratio of case intensity to control intensity, or to a baseline intensity (§11.4).
- **Part V symbols.** $\kappa$ is a knot location $\boldsymbol{\kappa}_j$ (bold, §13.2), the inverse range of the SPDE (scalar, §13.5), and the parent intensity (Chapters 11 and 12). $\boldsymbol{\alpha}$ (bold) is the vector of knot values in §13.2 and the kernel ridge coefficients in §4.5.3; scalar $\alpha$ is the SPDE exponent in §13.5.1. $\mathbf{B}$ is the $n \times q$ weight matrix in §13.2 and a generic matrix in §13.6.2. $\theta$ (scalar) is the taper range in §13.3, distinct from the parameter vector $\boldsymbol{\theta}$ and the relative risk $\theta_i$. $\boldsymbol{\Gamma}$ (bold) is the lower triangular weight matrix of §13.4, not the gamma function $\Gamma(\cdot)$ of Chapter 2.
- **Part IV symbols.** $\gamma$ is the Strauss penalty in Chapter 12 and the semivariogram in Part II; $Z$ in italics is the Strauss normalizing constant, while bold $\mathbf{Z}$ is a standard normal vector; $R$ is a disc radius (§12.2.2), the correlation matrix $\mathbf{R}(\phi)$, and the residual field $R(\mathbf{s})$ (§12.5.1); $\mathbf{d}$ is a displacement, $d_i$ a degree; $\mathbf{c}$ is a parent location, $\mathbf{c}_0$ a kriging covariance vector; $T$ and $B$ in italics are a test statistic and a number of simulations, distinct from the matrices $\mathbf{T}$ and $\mathbf{B}$; $\mathbf{U}(\boldsymbol{\beta})$ with an argument is a score vector, distinct from the eigenvector matrix $\mathbf{U}$; $\mathbf{S}$ in §12.4 is the pattern itself (coordinate matrix with rows $\mathbf{s}_i'$); the sibling-distance distribution function is $P_{\text{sib}}$ because $F$ is the empty-space function.
- **$\rho$.** The autoregressive parameter of Chapters 7 and 8; the BYM2 mixing parameter in Chapter 9 (§9.3.4); the Leroux weight on the Laplacian (§9.3.5). Its scale differs between models.
- **$\mathbf{M}$ in Part III.** Moran's central matrix $\mathbf{P}_c\mathbf{W}_s\mathbf{P}_c$ (Chapter 6); $\mathbf{D} - \rho\mathbf{A}$ in the CAR likelihood (§8.4.1); a positive definite matrix in a Gaussian integral (§9.5.1). All local.
- **$M_{\text{dir}}$, $T_W$.** Italic scalars with subscripts (Chapter 7), distinct from the matrices $\mathbf{M}$ and $\mathbf{T}$.
- **$\boldsymbol{\Sigma}$ in Chapters 8 and 9.** In Chapter 8, $\boldsymbol{\Sigma} = \mathbf{Q}^{-1}$ is the covariance of the random effect $\mathbf{u}$ (stated in §8.2.1), not of $\mathbf{Y}$.
- **$s^2$.** $\mathbf{e}'\mathbf{e}/(n - p)$ in §1.1.2; $\mathbf{e}'\mathbf{e}/n$ in the LM statistics (§7.6.2).
- **$z$ and $\mathbf{z}$.** Italic $z$ is a standardized test statistic (Chapter 6); bold $\mathbf{z}$ is the IRLS working response (Chapter 9). Chapter 6 uses no bold $\mathbf{z}$.
- **$C$.** Geary's $C$ (non-bold scalar, §6.7.1); the covariance function $C(h)$; the matrix $\mathbf{C}$ of CAR coefficients (§8.1.3) and $\mathbf{C}_u$ (§4.5.3).
- **$\vartheta$.** The natural parameter (§9.1.1) and the direction of a lag vector $\vartheta_{ij}$ (§3.1.4); both local.
- **$w_i$.** With one subscript, a shrinkage weight (§9.3.1); $w_{ij}$ is a spatial weight.
- **$\mathbf{v}$.** With a subscript, $\mathbf{v}_k$ is an eigenvector (Chapter 1 onward); without one, $\mathbf{v}$ is the unstructured random effect of a BYM model (Chapter 9).

---

## Concordance With Schabenberger and Gotway (2005)

Readers moving between this book and *Statistical Methods for Spatial Data Analysis* can translate notation with these tables.

### 1. General Conventions and Core Data Structures

| Concept | Schabenberger and Gotway (2005) | This book | Editorial note and rationale |
| :--- | :--- | :--- | :--- |
| **Spatial response process** | $Z(\mathbf{s})$ or $Z_i$ | $Y(\mathbf{s})$ or $\mathbf{Y}$ | We use $Y$ to maintain standard linear model and regression conventions ($\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$). Bold $\mathbf{Z}$ is reserved exclusively for standard normal simulation vectors. |
| **Study domain** | $D \subset \mathbb{R}^d$ | $\mathcal{D} \subset \mathbb{R}^2$ | Uses script $\mathcal{D}$ to avoid conflict with the degree matrix ($\mathbf{D}$) used in areal lattice models. |
| **Master model equation** | $\mu(\mathbf{s}) + W(\mathbf{s}) + \eta(\mathbf{s}) + \varepsilon(\mathbf{s})$ | $\mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})$ | We streamline S&G's four-part decomposition into trend ($\mu$), spatial random effect ($u$), and nugget ($\varepsilon$, absorbing micro-scale variation and measurement error). |
| **Observations vector** | $\mathbf{Z} = [Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n)]'$ | $\mathbf{Y} = [Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_n)]'$ | Consistent with the shift from $Z$ to $Y$. |

### 2. Geostatistics and Covariance Structures

| Concept | Schabenberger and Gotway (2005) | This book | Editorial note and rationale |
| :--- | :--- | :--- | :--- |
| **Covariance matrix** | $\boldsymbol{\Sigma}$ or $\mathbf{V}$ | $\boldsymbol{\Sigma}$ | We reserve $\mathbf{V}$ strictly for GLM and IRLS working weights (Chapters 9 and 11) to avoid symbol collisions. |
| **Spatial variance (partial sill)** | Varies (often sill or $\sigma^2$) | $\sigma^2$ | Always denotes the variance scale of the spatial random effect $u(\mathbf{s})$. |
| **Nugget variance** | Varies ($\theta_0, \sigma^2_\varepsilon$, etc.) | $\tau^2$ | Standardizes the classical error variance notation so it does not conflict with the spatial variance $\sigma^2$. |
| **Noise-to-signal ratio** | Not explicitly a single letter | $\delta = \tau^2/\sigma^2$ | Used in Part II for profiling likelihoods ($\boldsymbol{\Psi} = \mathbf{R} + \delta\mathbf{I}_n$). |
| **Range parameter** | $\theta$ or $\alpha$ or $\phi$ | $\phi$ | Represents how quickly correlation decays with distance (a distance scale parameter). |
| **Semivariogram** | $\gamma(\mathbf{h})$ or $2\gamma(\mathbf{h})$ (mixed) | $\gamma(\mathbf{h})$ | We strictly define $\gamma(\mathbf{h})$ as the semivariogram ($\tfrac{1}{2}\operatorname{Var}(Y(\mathbf{s}+\mathbf{h}) - Y(\mathbf{s}))$) and $2\gamma(\mathbf{h})$ as the variogram to avoid confusion. |
| **Matérn smoothness** | $\nu$ | $\nu$ | Identical usage for the Matérn smoothness parameter. |
| **Kriging weights** | $\boldsymbol{\lambda}$ | $\boldsymbol{\lambda}$ | Identical vector of prediction weights. |

### 3. Areal and Lattice Data

| Concept | Schabenberger and Gotway (2005) | This book | Editorial note and rationale |
| :--- | :--- | :--- | :--- |
| **Spatial weights matrix** | $\mathbf{W}, w_{ij}$ | $\mathbf{W}, w_{ij}$ | Identical. Lowercase $w_{ij}$ for entries, uppercase $\mathbf{W}$ for the matrix. |
| **Sum of weights** | $S_0$ | $S_0$ | Identical (used in Moran's $I$ scaling). |
| **Degree matrix** | $\mathbf{D}$ (sometimes implied) | $\mathbf{D}$ | Diagonal matrix of neighbor counts ($\operatorname{diag}(\mathbf{W}\mathbf{1})$). |
| **Adjacency matrix** | $\mathbf{A}$ | $\mathbf{A}$ | Binary contiguity matrix ($A_{ij} = 1$ if neighbors). |
| **SAR / CAR parameter** | $\rho$ | $\rho$ | Used across SAR, CAR, and Leroux models (with explicit notes that valid parameter ranges differ by model). |
| **Precision matrix (GMRF)** | $\mathbf{Q}$ or $\mathbf{V}^{-1}$ | $\mathbf{Q}$ | Inverse covariance matrix of a Gaussian Markov random field. |
| **BYM random effects** | Unstructured + structured | $\mathbf{u}^{\text{icar}} + \mathbf{v}$ | Explicitly separated into ICAR (structured) and Gaussian (unstructured) components, updated to BYM2 scaling in Chapter 9. |

### 4. Point Patterns

| Concept | Schabenberger and Gotway (2005) | This book | Editorial note and rationale |
| :--- | :--- | :--- | :--- |
| **Point coordinates** | $\mathbf{s}_i$ or events in $D$ | $\mathbf{S} \in \mathbb{R}^{n \times 2}$ | Coordinate matrix with rows $\mathbf{s}_i'$. |
| **Intensity function** | $\lambda(s)$ | $\lambda(\mathbf{s})$ | Bold $\mathbf{s}$ emphasizes spatial coordinate vectors in regression-style intensity models (Part IV). |
| **Ripley's $K$-function** | $K(h)$ | $K(r)$ | Uses $r$ for the search radius argument to cleanly separate distance lag $h$ (geostatistics) from search radii $r$. |
| **Besag's transformation** | $L(h) = \sqrt{K(h)/\pi}$ | $L(r) = \sqrt{K(r)/\pi}$ | Uses radius $r$. |
| **Edge correction weights** | Various ($w(\mathbf{s}_i, \mathbf{s}_j)$) | $\omega_{ij}$ | Standardized notation avoiding collision with spatial weights $w_{ij}$. |

### 5. Linear Models and Matrix Tools

| Concept | Schabenberger and Gotway (2005) | This book | Editorial note and rationale |
| :--- | :--- | :--- | :--- |
| **Design matrix** | $\mathbf{X}$ | $\mathbf{X}$ | Identical. |
| **Regression coefficients** | $\boldsymbol{\beta}$ or $\boldsymbol{\alpha}$ | $\boldsymbol{\beta}$ | $\boldsymbol{\beta}$ for linear and GLM predictors; $\boldsymbol{\alpha}$ reserved for basis and kernel ridge coefficients. |
| **Projection matrices** | Various | $\mathbf{P}_{\mathbf{X}}, \mathbf{P}_c$ | $\mathbf{P}_{\mathbf{X}}$ for OLS projection; $\mathbf{P}_c = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'$ for the centering matrix. |
| **Information matrix** | $\mathcal{I}(\boldsymbol{\theta})$ | $\mathcal{I}(\boldsymbol{\theta})$ | Fisher information for likelihood and REML inference. |

