# Appendix D: Where to Go Next

This book has organized spatial statistics around the linear model, generalized linear models, and multivariate normal theory. To build an accessible and rigorous foundation with fully self-contained Python implementations and complete matrix derivations, we intentionally focused on standard Gaussian geostatistics, lattice models with simple contiguity graphs, Poisson and Gibbs point processes on planar windows, and tractable large-$n$ matrix approximations.

Real spatial analysis, however, frequently encounters boundaries that standard models cannot cross: non-Gaussian responses, irregularly bounded non-Euclidean geometries, unobserved spatial confounders that distort causal inferences, and multivariate or space-time dependencies that strain basic separability assumptions.

This appendix provides a roadmap for what lies beyond this book. Each section connects a specific limitation or simplifying assumption from Chapters 0 to 13 to the modern research literature, showing you where spatial statistics is heading and providing the key equations and foundational references to guide your next steps.

---

## D.1 Optimal Spatial Sampling Design

In Section 4.2.2, we established one of the most remarkable properties of the Gaussian linear model: the kriging variance (mean squared prediction error) at an unobserved location $\mathbf{s}_0$,
$$
\sigma^2_K(\mathbf{s}_0) = \sigma^2 + \tau^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 + (\mathbf{x}_0 - \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0)'(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}(\mathbf{x}_0 - \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0),
$$
depends *only* on the spatial configuration of the sample locations $\mathbf{S} = (\mathbf{s}_1, \dots, \mathbf{s}_n)'$, the target location $\mathbf{s}_0$, the covariate values $\mathbf{X}$ and $\mathbf{x}_0$, and the covariance parameters $\boldsymbol{\theta} = (\sigma^2, \tau^2, \phi)'$. It does *not* depend on the observed response values $\mathbf{y} = (y_1, \dots, y_n)'$.

This algebraic decoupling makes prospective **spatial experimental design** possible: an investigator can optimize the geographic placement of monitoring stations, boreholes, or air quality sensors *before collecting a single observation*.

### Design Criteria

Depending on the substantive objective, optimal spatial design literature classifies sampling schemes into distinct formal criteria:

1. **A-Optimality (Average Prediction Variance):** When the goal is to predict the continuous surface across an entire domain $\mathcal{D}$, the design chooses locations $\mathbf{S}^* = \arg\min_{\mathbf{S}} \int_{\mathcal{D}} \sigma^2_K(\mathbf{s}_0 \mid \mathbf{S})\,d\mathbf{s}_0$. Numerically, this integral is approximated over a fine prediction grid. A-optimal designs distribute sampling locations in regular, space-filling geometric patterns (such as triangular or hexagonal lattices), maximizing the minimum distance between neighboring points.
2. **G-Optimality (Minimax Prediction Variance):** In risk assessment and environmental monitoring, avoiding catastrophic blind spots is more critical than average precision. The minimax criterion minimizes the maximum kriging variance across the domain:
   $$
   \mathbf{S}^* = \arg\min_{\mathbf{S}} \max_{\mathbf{s}_0 \in \mathcal{D}} \sigma^2_K(\mathbf{s}_0 \mid \mathbf{S}).
   $$
   This strategy aggressively places sensors in spatial gaps, penalizing large unmonitored holes.
3. **D-Optimality (Covariance Parameter Identification):** Here lies the central dilemma of spatial design. While space-filling lattices minimize prediction error when parameters $\boldsymbol{\theta}$ are known, they are exceptionally poor at *estimating* $\boldsymbol{\theta}$. As shown in Chapter 3, identifying the nugget effect $\tau^2$ and the short-range correlation slope requires pairs of points at very small separation distances $h \to 0$. D-optimal designs maximize the determinant of the Fisher information matrix $\log |\mathcal{I}(\boldsymbol{\theta})|$ derived in Section 5.3.3. D-optimal configurations force points to cluster closely together in space.

### The Hybrid Compromise

Practical spatial sampling requires a compromise between prediction coverage and parameter identification. Diggle and Lophaven (2006) proposed augmenting a regular space-filling lattice with a fraction (e.g., 10% to 20%) of close-pair clusters or random infill points. Zimmerman (2006) provides a comprehensive comparison of spatial design strategies under misspecified covariance models. Müller (2007) reviews computational search algorithms, including simulated annealing and genetic algorithms, used to navigate the combinatorial landscape of candidate spatial configurations.

---

## D.2 Beyond Plug-in Prediction: Parameter Uncertainty in Geostatistics

Throughout Chapter 4 and Chapter 5, spatial prediction relied on the **plug-in predictor**: we estimated covariance parameters $\hat{\boldsymbol{\theta}} = (\hat{\sigma}^2, \hat{\tau}^2, \hat{\phi})'$ via profile REML or maximum likelihood (§5.3), substituted them directly into $\boldsymbol{\Sigma}(\hat{\boldsymbol{\theta}})$ and $\mathbf{c}_0(\hat{\boldsymbol{\theta}})$, and treated the resulting kriging equations as if the parameters were known constants without sampling error.

As flagged in Section 5.4.5, this plug-in convention introduces systematic optimism. The nominal kriging variance $\sigma^2_K(\mathbf{s}_0 \mid \hat{\boldsymbol{\theta}})$ measures only the conditional uncertainty of the Gaussian process given $\boldsymbol{\theta}$; it completely ignores the sampling variance $\operatorname{Var}(\hat{\boldsymbol{\theta}})$. Consequently, nominal 95% plug-in prediction intervals routinely achieve empirical coverage rates of only 88% to 92%, particularly in moderate sample sizes ($n < 200$) or when correlation parameters lie on shallow likelihood ridges.

### Modern Solutions to Plug-in Optimism

Three primary methodologies account for parameter uncertainty:

1. **The Parametric Bootstrap:** One simulates $B$ synthetic realizations $\mathbf{y}^{(b)}$ from $\mathcal{N}(\mathbf{X}\hat{\boldsymbol{\beta}}, \boldsymbol{\Sigma}(\hat{\boldsymbol{\theta}}))$, re-fits the spatial model to each realization to obtain $\hat{\boldsymbol{\theta}}^{(b)}$, and simulates target predictions $Y^{(b)}(\mathbf{s}_0)$ conditional on $\mathbf{y}^{(b)}$ and $\hat{\boldsymbol{\theta}}^{(b)}$. The empirical quantiles of $\{Y^{(b)}(\mathbf{s}_0)\}_{b=1}^B$ yield prediction intervals that directly incorporate parameter variability.
2. **Analytical Second-Order Corrections:** Harville and Jeske (1992) developed a second-order Taylor series expansion of the actual mean squared prediction error:
   $$
   \operatorname{MSPE}(\hat{Y}(\mathbf{s}_0; \hat{\boldsymbol{\theta}})) \approx \sigma^2_K(\mathbf{s}_0 \mid \boldsymbol{\theta}) + \operatorname{tr}\left( \operatorname{Var}(\hat{\boldsymbol{\theta}}) \operatorname{Cov}\left( \left.\frac{\partial \hat{Y}(\mathbf{s}_0)}{\partial \boldsymbol{\theta}}\right|_{\boldsymbol{\theta}} \right) \right).
   $$
   Evaluating this correction requires the asymptotic covariance matrix $\operatorname{Var}(\hat{\boldsymbol{\theta}}) \approx \mathcal{I}_{\text{REML}}^{-1}$ from Section 5.3.3 and the gradients of the kriging weights with respect to $\boldsymbol{\theta}$.
3. **Fully Bayesian Posterior Predictive Averaging:** The most coherent framework for uncertainty quantification is fully Bayesian. Rather than conditioning on a point estimate $\hat{\boldsymbol{\theta}}$, Bayesian geostatistics integrates predictions over the full joint posterior distribution $p(\boldsymbol{\beta}, \boldsymbol{\theta} \mid \mathbf{y})$:
   $$
   p(y_0 \mid \mathbf{y}) = \int_{\boldsymbol{\Theta}} \int_{\mathbb{R}^p} p(y_0 \mid \mathbf{y}, \boldsymbol{\beta}, \boldsymbol{\theta}) \, p(\boldsymbol{\beta}, \boldsymbol{\theta} \mid \mathbf{y}) \, d\boldsymbol{\beta} \, d\boldsymbol{\theta}.
   $$
   When Markov chain Monte Carlo (MCMC) samples $(\boldsymbol{\beta}^{(s)}, \boldsymbol{\theta}^{(s)})$ are drawn from the posterior, generating $y_0^{(s)} \sim p(y_0 \mid \mathbf{y}, \boldsymbol{\beta}^{(s)}, \boldsymbol{\theta}^{(s)})$ produces an exact predictive sample. Diggle and Ribeiro (2007) develop this model-based geostatistical approach in depth.

---

## D.3 Spatial Confounding, Restricted Spatial Regression, and Causal Inference

In Section 5.6.2 and Section 9.6.5, we warned of a subtle trap known as **spatial confounding**. When studying the relationship between an exposure covariate $\mathbf{x}$ and an outcome $\mathbf{y}$, analysts frequently add a spatial random effect $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ to "control for spatial autocorrelation." 

However, if the exposure $\mathbf{x}$ itself varies smoothly across the landscape (e.g., regional air pollution, median income, or temperature), $\mathbf{x}$ will be collinear with the synthetic spatial basis vectors formed by the leading eigenvectors of $\boldsymbol{\Sigma}$.

### The Confounding Dilemma

This collinearity causes two severe pathologies:
1. **Variance Inflation:** The standard error of the regression coefficient $\hat{\beta}_1$ inflates dramatically—often by factors of 3 to 10—destroying statistical power to detect genuine treatment effects.
2. **Sign Reversal and Attenuation:** The spatial random effect $\mathbf{u}$ acts as a statistical sponge, soaking up variation that properly belongs to $\mathbf{x}$. In extreme cases, $\hat{\beta}_1$ switches sign relative to standard OLS, producing counterintuitive scientific conclusions.

### Restricted Spatial Regression (RSR)

To prevent the spatial random effect from competing with the fixed-effect covariates, Reich, Hodges, and Zadnik (2006) and Hughes and Haran (2013) proposed **Restricted Spatial Regression (RSR)**. RSR projects the spatial random effect onto the subspace orthogonal to the design matrix $\mathbf{X}$.

Let $\mathbf{P}_{\mathbf{X}}^\perp = \mathbf{I}_n - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ be the projection matrix onto the orthogonal complement of the column space $\mathcal{C}(\mathbf{X})$. The restricted model reparameterizes the spatial random effect as:
$$
\mathbf{u} = \mathbf{P}_{\mathbf{X}}^\perp \boldsymbol{\gamma},
$$
where $\boldsymbol{\gamma}$ is an unconstrained spatial vector. Because $\mathbf{X}'\mathbf{u} = \mathbf{X}'\mathbf{P}_{\mathbf{X}}^\perp \boldsymbol{\gamma} = \mathbf{0}$ by construction, the spatial random effect is strictly orthogonal to the covariates. Consequently, the point estimate $\hat{\boldsymbol{\beta}}$ in an RSR model is identical to the ordinary least squares estimate $\hat{\boldsymbol{\beta}}_{\text{OLS}}$, while the restricted random effect models spatial correlation strictly in the residual subspace.

### The Hanks et al. Critique and Modern Spatial Causality

While RSR stabilizes variance, Hanks et al. (2015) and Khan and Calder (2020) demonstrated that RSR is not a free lunch: if an unobserved spatial confounder *does* exist in nature and correlates with both $\mathbf{x}$ and $\mathbf{y}$, forcing $\mathbf{X}'\mathbf{u} = \mathbf{0}$ deliberately induces omitted-variable bias. 

Modern spatial statistics has therefore turned toward formal **spatial causal inference**:
* **Potential Outcomes under Spatial Interference:** Classical causal inference relies on the Stable Unit Treatment Value Assumption (SUTVA), which posits that one unit's treatment assignment does not affect another unit's potential outcomes. In spatial settings, this assumption fails routinely due to spatial spillover effects (e.g., mosquito abatement in one district protects neighboring untreated districts). Papadogeorgou et al. (2019) formulate causal estimands under bipartite network interference.
* **Spatial Matching and Confounder Buffers:** To estimate treatment effects without parametric random-effect artifacts, observational units are matched with controls that share identical covariate profiles but sit outside the geographic radius of spatial spillover.

---

## D.4 Finite-Element SPDEs, Buffer Meshes, and Bayesian Computation (INLA / MCMC)

In Section 13.5, we derived the celebrated stochastic partial differential equation (SPDE) bridge: a stationary Gaussian random field $u(\mathbf{s})$ with Matérn covariance satisfies the fractional elliptic SPDE
$$
(\kappa^2 - \Delta)^{\alpha/2} u(\mathbf{s}) = \mathcal{W}(\mathbf{s}),
$$
where $\kappa = 1/\phi$, $\Delta = \frac{\partial^2}{\partial s_1^2} + \frac{\partial^2}{\partial s_2^2}$ is the continuous Laplacian, and $\alpha = \nu + d/2$. We showed that on a regular grid, discretizing $(\kappa^2 - \Delta)$ produces a second-order grid Laplacian precision matrix $\mathbf{Q} = (\kappa^2\mathbf{I} + \mathcal{L})^2$ that is analytically identical to a Spatial Autoregressive (SAR) precision from Chapter 7.

### The Finite-Element Mesh on Irregular Domains

In practical geostatistics, observation locations are rarely arranged on regular Cartesian grids. Lindgren, Rue, and Lindström (2011) generalized this SPDE bridge to arbitrary continuous domains by applying the **finite element method (FEM)** over a Constrained Delaunay Triangulation.

The study domain $\mathcal{D}$ is partitioned into non-overlapping triangles defined by $m$ vertex nodes $\{\mathbf{v}_k\}_{k=1}^m$. The continuous spatial process $u(\mathbf{s})$ is approximated by a linear combination of piecewise linear pyramid basis functions $\psi_k(\mathbf{s})$:
$$
u(\mathbf{s}) \approx \sum_{k=1}^m \psi_k(\mathbf{s}) u_k,
$$
where $\psi_k(\mathbf{v}_j) = 1$ if $j = k$ and $0$ otherwise, and $\mathbf{u} = (u_1, \dots, u_m)'$ is the vector of field values at the mesh vertices. 

Under the FEM weak formulation with test functions, the precision matrix $\mathbf{Q}$ of the vertex vector $\mathbf{u}$ is given explicitly by sparse matrix products:
$$
\mathbf{Q}_{\alpha=1} = \kappa^2 \mathbf{C} + \mathbf{G}, \qquad \mathbf{Q}_{\alpha=2} = (\kappa^2 \mathbf{C} + \mathbf{G}) \mathbf{C}^{-1} (\kappa^2 \mathbf{C} + \mathbf{G}),
$$
where $\mathbf{C}$ is a diagonal lumped mass matrix ($C_{kk} = \int \psi_k(\mathbf{s})\,d\mathbf{s}$) and $\mathbf{G}$ is a sparse stiffness matrix ($G_{jk} = \int \nabla \psi_j(\mathbf{s}) \cdot \nabla \psi_k(\mathbf{s})\,d\mathbf{s}$). Because basis functions have local compact support (overlapping only on adjacent triangles), $\mathbf{Q}$ is extremely sparse: each row contains non-zero entries only for immediately connected mesh neighbors.

### The Neumann Boundary Reflection and the Outer Buffer Mesh

In Section 13.5.2, we discovered a crucial physical artifact: imposing zero-flux (Neumann) boundary conditions on a finite grid causes the field to reflect across boundaries like light in a hall of mirrors. The marginal variance doubles ($2\times$) along boundaries and quadruples ($4\times$) in domain corners.

In applied FEM and INLA workflows, this boundary inflation is solved by extending the computational mesh into an **outer buffer zone**. Practitioners construct an irregular triangulation that extends at least one to two practical correlation ranges ($3\phi$) beyond the geographic boundary of the study area $\mathcal{D}$. The mesh is generated with fine, dense triangles inside $\mathcal{D}$ (where data reside) and progressively coarser, larger triangles throughout the outer buffer. After evaluating the GMRF precision $\mathbf{Q}$ and estimating the model, the boundary nodes in the outer buffer are simply discarded, leaving the study region untouched by reflection distortions.

### Integrated Nested Laplace Approximations (INLA)

Solving high-dimensional spatial GLMMs (Chapter 9) via traditional Markov chain Monte Carlo (MCMC) algorithms can require days of computation, with Gibbs and Metropolis samplers suffering from severe autocorrelation. 

Rue, Martino, and Chopin (2009) introduced **Integrated Nested Laplace Approximations (INLA)** as a deterministic alternative to MCMC for latent Gaussian models. By exploiting the extreme sparsity of the FEM precision matrix $\mathbf{Q}$ and applying nested Laplace expansions (introduced conceptually in Section 9.5), INLA computes accurate marginal posterior distributions for fixed effects $p(\beta_j \mid \mathbf{y})$, hyperparameters $p(\theta_k \mid \mathbf{y})$, and latent spatial fields $p(u(\mathbf{s}_i) \mid \mathbf{y})$ in minutes rather than days. Blangiardo and Cameletti (2015) provide an applied guide to spatio-temporal modeling with the `R-INLA` package.

---

## D.5 Nonseparable Space-Time Covariances and Multivariate Geostatistics (Cokriging)

In Section 13.6, we developed the computational mechanics of **separable space-time models**, where the joint covariance factors into a Kronecker product $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$. We proved that this structure reduces the computational cost of factorizing an $(n \cdot T) \times (n \cdot T)$ matrix from $\mathcal{O}(n^3 T^3)$ to $\mathcal{O}(n^3 + T^3)$.

However, we also highlighted the substantive limitation of separability: in a separable model, the spatial correlation function $R_S(\mathbf{h})$ is forced to be identical across all temporal lags. Separable models cannot capture physical processes exhibiting **space-time interaction**, prevailing directional advection (e.g., wind transporting atmospheric particulates over time), or temporal scales that vary across space.

### Gneiting's Nonseparable Covariance Class

Gneiting (2002) introduced a general mathematical class of valid, nonseparable, stationary space-time covariance functions on $\mathbb{R}^d \times \mathbb{R}$:
$$
C(\mathbf{h}, u) = \frac{\sigma^2}{\left(a |u|^{2\alpha} + 1\right)^{\beta d / 2}} \exp\left( - \frac{c \|\mathbf{h}\|^{2\gamma}}{\left(a |u|^{2\alpha} + 1\right)^{\beta \gamma}} \right),
$$
where $\mathbf{h} \in \mathbb{R}^d$ is the spatial displacement vector, $u \in \mathbb{R}$ is the temporal lag, $a, c > 0$ are scaling parameters, $\alpha, \gamma \in (0, 1]$ are smoothness parameters, and $\beta \in [0, 1]$ is the **space-time interaction parameter**.

The substantive power of Gneiting's formulation lies in the parameter $\beta$:
* When $\beta = 0$, the function factors into $C_S(\mathbf{h}) \cdot C_T(u)$, recovering a separable covariance.
* When $\beta > 0$, the spatial correlation range expands or contracts dynamically as a function of temporal lag $u$. The parameter $\beta$ can be estimated directly by maximum likelihood, providing a formal hypothesis test ($H_0: \beta = 0$) for whether data justify the separable approximation.

### Multivariate Fields and Cokriging

In environmental and geochemical surveys, investigators often measure several interdependent variables at each spatial location—for example, soil concentrations of cadmium, copper, lead, and zinc. Analyzing each metal with an independent univariate kriging model discards the cross-variable correlations that allow a densely sampled auxiliary variable (e.g., zinc) to improve predictions for an expensively sampled target variable (e.g., cadmium).

**Cokriging** generalizes kriging to vector-valued spatial processes $\mathbf{Y}(\mathbf{s}) = (Y_1(\mathbf{s}), \dots, Y_m(\mathbf{s}))'$. The core mathematical challenge is constructing a valid matrix-valued cross-covariance function:
$$
\mathbf{C}(\mathbf{h}) = \begin{bmatrix} C_{11}(\mathbf{h}) & C_{12}(\mathbf{h}) & \cdots & C_{1m}(\mathbf{h}) \\ C_{21}(\mathbf{h}) & C_{22}(\mathbf{h}) & \cdots & C_{2m}(\mathbf{h}) \\ \vdots & \vdots & \ddots & \vdots \\ C_{m1}(\mathbf{h}) & C_{m2}(\mathbf{h}) & \cdots & C_{mm}(\mathbf{h}) \end{bmatrix},
$$
such that for any arbitrary set of locations and vectors, the resulting $m n \times m n$ covariance matrix is strictly positive definite.

Two primary constructions dominate the literature:
1. **The Linear Model of Coregionalization (LMC):** The multivariate process is modeled as a linear transformation of $p \le m$ independent, orthogonal univariate Gaussian processes $w_k(\mathbf{s}) \sim \mathcal{GP}(0, \rho_k(\mathbf{h}))$:
   $$
   \mathbf{Y}(\mathbf{s}) = \mathbf{A} \mathbf{w}(\mathbf{s}) + \boldsymbol{\varepsilon}(\mathbf{s}) \implies \mathbf{C}(\mathbf{h}) = \sum_{k=1}^p \mathbf{V}_k \, \rho_k(\mathbf{h}),
   $$
   where $\mathbf{V}_k = \mathbf{a}_k \mathbf{a}_k'$ are positive semi-definite coregionalization matrices (Wackernagel 2003).
2. **The Multivariate Matérn Class:** Gneiting, Kleiber, and Schlather (2010) established explicit parameter constraints ensuring that setting each direct covariance $C_{ii}(\mathbf{h})$ and cross-covariance $C_{ij}(\mathbf{h})$ to a Matérn function produces a valid, strictly positive definite cross-covariance matrix.

---

## D.6 Non-Gaussian Geostatistics, Trans-Gaussian Kriging, and Copulas

Standard kriging theory assumes that the spatial random field is Gaussian. In applied ecological, hydrological, and financial data, empirical distributions routinely violate Gaussianity: precipitation exhibits exact zeros and heavy upper tails; air pollutants exhibit extreme positive skewness; and soil attributes (such as pH or fractional sand content) have strict physical lower and upper bounds.

### Trans-Gaussian Kriging and Nonlinear Transformations

In Section 4.5.4, we discussed lognormal kriging, noting that when modeling $Z(\mathbf{s}) = \log Y(\mathbf{s})$, exponentiating the kriging prediction $\exp(\hat{Z}(\mathbf{s}_0))$ systematically underestimates the expected original response because $\operatorname{E}[\exp(Z)] = \exp(\mu + \frac{1}{2}\sigma^2) > \exp(\mu)$. 

For a general monotonic transformation $g(Y(\mathbf{s})) = Z(\mathbf{s}) \sim \mathcal{GP}(\mathbf{X}\boldsymbol{\beta}, \boldsymbol{\Sigma})$, **trans-Gaussian kriging** (Cressie 1993) uses a second-order Taylor expansion to compute an unbiased back-transformed point prediction:
$$
\hat{Y}(\mathbf{s}_0) \approx g^{-1}(\hat{Z}(\mathbf{s}_0)) + \frac{1}{2} \left. \frac{d^2 g^{-1}(z)}{dz^2} \right|_{z = \hat{Z}(\mathbf{s}_0)} \sigma^2_K(\mathbf{s}_0),
$$
where $\sigma^2_K(\mathbf{s}_0)$ is the kriging variance on the transformed scale. 

While trans-Gaussian corrections repair the expected value, they cannot resolve the fundamental rigidity of Gaussian dependence: in a transformed Gaussian field, the spatial dependence structure remains conditionally symmetric, meaning extreme highs and extreme lows exhibit identical joint clustering patterns.

### Spatial Copulas and Skew-Gaussian Processes

To decouple marginal distributions from spatial dependence, modern geostatistics employs **spatial copulas** (Bárdossy 2006; Kazianka and Pilz 2010). By Sklar's theorem, any continuous multivariate distribution $F(y_1, \dots, y_n)$ can be expressed uniquely through its univariate marginals $F_i(y_i)$ and a copula function $\mathcal{C}$:
$$
F(y_1, \dots, y_n) = \mathcal{C}\left( F_1(y_1), \dots, F_n(y_n) \right).
$$

Under a Gaussian copula, each observation $Y(\mathbf{s}_i)$ can follow an arbitrary marginal distribution (e.g., Gamma, Generalized Extreme Value, or Zero-Inflated Poisson). The data are mapped to standard normal scores via probability integral transformations:
$$
Z_i = \Phi^{-1}\left( F_i(Y(\mathbf{s}_i)) \right),
$$
where $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}(\boldsymbol{\theta}))$ inherits a spatial correlation matrix. 

When extreme events cluster more intensely than moderate events (as in catastrophic rainfall or flooding), analysts replace the Gaussian copula with an **extreme-value copula** or model the response using **max-stable processes** (Davison, Padoan, and Ribatet 2012), providing a mathematically consistent framework for spatial extreme-value analysis.

---

## D.7 Deeper Harmonic Analysis: Bochner's Theorem, Schoenberg's Theorem, and Spectral Representations

In Section 0.5.1 and Section 2.4.4, we emphasized that one cannot simply invent an arbitrary function of distance $C(h)$ and declare it a covariance function: the function must ensure that for any arbitrary finite set of locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$, the resulting matrix $\boldsymbol{\Sigma}_{ij} = C(\|\mathbf{s}_i - \mathbf{s}_j\|)$ is positive semi-definite ($\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} \ge 0$ for all non-zero vectors $\mathbf{a}$).

Why do seemingly plausible functions (such as a linear variogram with a sill) fail to be positive definite? The definitive answers belong to harmonic analysis.

### Bochner's Theorem (Positive Definiteness in Euclidean Space)

Bochner's theorem (1932) provides the foundational characterization of covariance functions in continuous Euclidean space $\mathbb{R}^d$:

> **Bochner's Theorem:** A continuous, complex-valued function $C(\mathbf{h})$ on $\mathbb{R}^d$ is positive definite if and only if it can be represented as the Fourier transform of a finite, non-negative spectral measure $F(d\boldsymbol{\omega})$:
> $$
> C(\mathbf{h}) = \int_{\mathbb{R}^d} e^{i \boldsymbol{\omega}' \mathbf{h}} \, dF(\boldsymbol{\omega}).
> $$

When the spectral measure is absolutely continuous with respect to Lebesgue measure, $dF(\boldsymbol{\omega}) = f(\boldsymbol{\omega})\,d\boldsymbol{\omega}$, where $f(\boldsymbol{\omega}) \ge 0$ is the **spatial spectral density**. 

Bochner's theorem provides the ultimate operational test for covariance validity: to determine whether a candidate covariance function $C(\mathbf{h})$ is valid in $d$ dimensions, compute its inverse Fourier transform:
$$
f(\boldsymbol{\omega}) = \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} e^{-i \boldsymbol{\omega}' \mathbf{h}} C(\mathbf{h}) \, d\mathbf{h}.
$$
If $f(\boldsymbol{\omega}) \ge 0$ for all frequencies $\boldsymbol{\omega} \in \mathbb{R}^d$, the covariance is valid; if $f(\boldsymbol{\omega}) < 0$ for any $\boldsymbol{\omega}$, the function is mathematically invalid and will produce negative prediction variances on certain spatial configurations.

### Schoenberg's Theorem (Isotropic Covariances Across Dimensions)

When spatial dependence depends solely on Euclidean distance $h = \|\mathbf{h}\|$, the covariance is isotropic. Schoenberg (1938) addressed the question: *Which isotropic correlation functions remain valid across any arbitrary dimension $d$?*

> **Schoenberg's Theorem:** A continuous function $C(h)$ with $C(0) = 1$ is positive definite in $\mathbb{R}^d$ for *all* $d = 1, 2, 3, \dots$ if and only if it can be represented as a scale mixture of squared exponential (Gaussian) covariances:
> $$
> C(h) = \int_0^\infty \exp\left( -r h^2 \right) \, dG(r),
> $$
> where $G(r)$ is a cumulative distribution function on $[0, \infty)$.

Schoenberg's theorem explains deep empirical behaviors:
* The **Gaussian covariance** $C(h) = \exp(-h^2/\phi^2)$ corresponds to a point mass $G(r)$ at $r = 1/\phi^2$, and is therefore valid in every spatial dimension $d \ge 1$.
* The **Matérn covariance** can be represented as an inverse-gamma mixture of Gaussians, proving that the Matérn family is valid in all dimensions.
* By contrast, the **spherical covariance** (§2.3.4) is valid only up to $d = 3$; in $d \ge 4$, its Fourier transform drops below zero, causing positive definiteness to fail. Stein (1999) provides the authoritative statistical treatment of spectral representations and asymptotic kriging theory.

---

## D.8 Geostatistics on Spheres and Manifolds

Throughout this text, spatial coordinates $\mathbf{s} = (s_1, s_2)'$ were assumed to lie in a flat planar domain $\mathcal{D} \subset \mathbb{R}^2$, with distances measured via the Pythagorean Euclidean metric $\|\mathbf{s}_i - \mathbf{s}_j\|$. In Section 1.6.4, we warned that when analyzing continental or planetary datasets (such as global climate models, satellite remote sensing, or international disease tracking), the Earth's curvature cannot be flattened onto a Cartesian plane without introducing distortion.

### Chordal Distance versus Great-Circle Distance

To model spatial correlation on the sphere $\mathbb{S}^2 = \{\mathbf{s} \in \mathbb{R}^3 : \|\mathbf{s}\| = R\}$, two metrics present themselves:

1. **Chordal Distance:** The straight-line Euclidean distance cutting directly through the interior of the Earth:
   $$
   d_{\text{chord}}(\mathbf{s}_1, \mathbf{s}_2) = \|\mathbf{s}_1 - \mathbf{s}_2\|_{\mathbb{R}^3} = 2 R \sin\left( \frac{\theta}{2} \right),
   $$
   where $\theta \in [0, \pi]$ is the central angle between the two locations. Because chordal distance is an ordinary Euclidean distance in $\mathbb{R}^3$, plugging $d_{\text{chord}}$ into any valid 3D planar covariance model (such as a 3D Matérn) guarantees a positive definite covariance matrix on the sphere. However, chordal distance is physically unnatural: it treats correlation as traveling through the Earth's mantle rather than across its atmosphere or oceans.
2. **Great-Circle Distance:** The intrinsic geodesic distance along the curved spherical surface:
   $$
   d_{\text{gc}}(\mathbf{s}_1, \mathbf{s}_2) = R \, \theta = R \arccos\left( \frac{\mathbf{s}_1 \cdot \mathbf{s}_2}{R^2} \right).
   $$

### The Planar-to-Spherical Pitfall

A common trap among applied analysts is taking a standard planar covariance model $C_{\text{planar}}(h)$ and simply plugging in great-circle distance: $C(\theta) = C_{\text{planar}}(R\theta)$.

Gneiting (2013) demonstrated that **this practice is mathematically hazardous**: standard covariance families that are positive definite in Euclidean space $\mathbb{R}^2$ can fail to be positive definite when evaluated using great-circle distance on the sphere $\mathbb{S}^2$. For example, the spherical covariance model with great-circle distance fails positive definiteness on the sphere.

### Schoenberg's Characterization on the Sphere

Schoenberg (1942) proved that a continuous function $C(\theta)$ is strictly positive definite on the sphere $\mathbb{S}^2$ if and only if it admits an expansion in **Legendre polynomials** $P_\ell(\cos\theta)$ with strictly positive coefficients:
$$
C(\theta) = \sum_{\ell=0}^\infty a_\ell P_\ell(\cos\theta), \qquad a_\ell \ge 0, \quad \sum_{\ell=0}^\infty a_\ell < \infty.
$$

Valid covariance functions designed specifically for spherical geometry include spherical Wendland functions, the multiquadric family, and specialized spherical Matérn formulations (Guinness and Fuentes 2016; Jeong and Jun 2015).

---

## D.9 The Spatial Durbin Model and Higher-Order Spatial Econometrics

In Chapter 7, we developed the Spatial Autoregressive (SAR) lag model,
$$
\mathbf{Y} = \rho \mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
$$
and proved that the global spillover multiplier $(\mathbf{I}_n - \rho\mathbf{W})^{-1}$ generates non-zero indirect impacts across the entire network. In Section 7.6.3, we noted that the SAR lag model imposes a strong structural constraint: the ratio of indirect (spillover) impacts to direct impacts is forced to be identical for *every* explanatory variable in the model.

### The Spatial Durbin Model (SDM)

To relax this rigid proportional spillover constraint, spatial econometrics relies on the **Spatial Durbin Model (SDM)** (LeSage and Pace 2009):
$$
\mathbf{Y} = \rho \mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \mathbf{W}\mathbf{X}\boldsymbol{\gamma} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n),
$$
where $\mathbf{W}\mathbf{X}$ is the spatial lag of the explanatory variables, and $\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_p)'$ measures the direct impact of neighboring covariates on the local outcome.

The SDM provides several crucial advantages:
1. **Flexible Spillovers:** By allowing each covariate $X_k$ to have both an individual direct parameter $\beta_k$ and a neighbor parameter $\gamma_k$, the model permits some predictors to generate strong global spillovers while others remain purely local.
2. **Robustness to Omitted Spatial Covariates:** LeSage and Pace (2009) proved that if an omitted variable is spatially correlated and correlates with an included predictor $\mathbf{x}_k$, the SDM estimates unbiased fixed-effect parameters where ordinary SAR or OLS models suffer from omitted-variable bias.

### The Common Factor Hypothesis

The Spatial Durbin Model serves as an overarching umbrella uniting the SAR lag model and the Spatial Error Model (SEM) from Chapter 7. 

Recall that the Spatial Error Model specifies $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$ with $\mathbf{u} = \rho\mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}$. Multiplying both sides by $(\mathbf{I}_n - \rho\mathbf{W})$ yields:
$$
\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} - \rho\mathbf{W}\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}.
$$
Comparing this to the Spatial Durbin equation reveals that the Spatial Error Model is a restricted special case of the Spatial Durbin Model under the non-linear parameter restriction:
$$
H_0: \boldsymbol{\gamma} = -\rho \boldsymbol{\beta}.
$$
This restriction is known as the **Common Factor Hypothesis**. Rather than guessing whether to estimate a spatial lag model or a spatial error model, analysts can fit the unrestricted Spatial Durbin Model and perform a formal likelihood ratio or Wald test of $H_0: \boldsymbol{\gamma} = -\rho\boldsymbol{\beta}$. If the restriction is rejected, the spatial error specification is misspecified.

### Impact Metrics in the Spatial Durbin Model

The marginal derivative matrix in the SDM expands to:
$$
\mathbf{S}_k(\mathbf{W}) = \frac{\partial \mathbf{Y}}{\partial \mathbf{x}_k} = (\mathbf{I}_n - \rho\mathbf{W})^{-1} \left( \beta_k \mathbf{I}_n + \gamma_k \mathbf{W} \right).
$$
The direct impact is the average of the diagonal entries $n^{-1}\operatorname{tr}(\mathbf{S}_k(\mathbf{W}))$, while the total impact is the average row sum $n^{-1}\mathbf{1}_n'\mathbf{S}_k(\mathbf{W})\mathbf{1}_n = \frac{\beta_k + \gamma_k}{1 - \rho}$. Elhorst (2014) provides a detailed taxonomy of higher-order spatial econometric specifications.

---

## D.10 Marked Point Patterns and Space-Time Point Processes

In Chapters 10, 11, and 12, our study of spatial point patterns focused primarily on unmarked planar points (where each event is simply a location $\mathbf{s}_i$) and discrete multitype patterns (Section 10.7, where points carried categorical labels such as tree species or case vs. control).

In many applied domains, each event $\mathbf{s}_i$ carries a **continuous quantitative mark** $m_i \in \mathbb{R}$—for example, the diameter or timber volume of a tree, the magnitude and depth of an earthquake, or the transaction price of a real estate sale.

### Continuous Marked Point Processes and Mark Correlation Functions

To explore how quantitative attributes depend on geographic separation, point process theory constructs **mark summary functions** (Illian et al. 2008). 

Let $m(\mathbf{s})$ denote the mark attached to an event at location $\mathbf{s}$, and let $\mu_m = \operatorname{E}[m]$ and $\sigma_m^2 = \operatorname{Var}(m)$ denote the marginal mean and variance across all marks. **Stoyan's mark correlation function** is defined as:
$$
k_{mm}(r) = \frac{\operatorname{E}_{0r}[m(\mathbf{0}) \, m(\mathbf{r})]}{\mu_m^2},
$$
where $\operatorname{E}_{0r}$ denotes the conditional expectation given that two events occur separated by distance $r$.

The substantive interpretation of $k_{mm}(r)$ provides immediate physical insight:
* Under the null hypothesis of **independent marking** (where marks are assigned to locations independently of spatial geometry), $k_{mm}(r) = 1$ for all distances $r$.
* When $k_{mm}(r) > 1$, neighboring events tend to carry mutually elevated marks (e.g., clustering of high-yield crops or mutually supportive tree canopies).
* When $k_{mm}(r) < 1$, neighboring events suppress one another's marks (e.g., resource competition where trees growing within 3 meters of one another have smaller trunk diameters).

Complementary summaries include the **mark variogram** $\gamma_m(r) = \frac{1}{2}\operatorname{E}_{0r}[(m(\mathbf{0}) - m(\mathbf{r}))^2]$ and Schlather's $I_{mm}(r)$.

### Spatio-Temporal Point Processes and Hawkes Self-Exciting Models

When events possess both geographic locations and timestamps $(\mathbf{s}_i, t_i) \in \mathcal{D} \times [0, T]$, the pattern forms a **spatio-temporal point process**. 

The process is characterized by its **space-time conditional intensity** $\lambda^*(\mathbf{s}, t \mid \mathcal{H}_t)$, measuring the conditional rate of an event occurring at location $\mathbf{s}$ and time $t$ given the entire preceding history of events $\mathcal{H}_t = \{(\mathbf{s}_j, t_j) : t_j < t\}$.

A prominent framework is the **spatio-temporal Hawkes process** (or self-exciting point process):
$$
\lambda^*(\mathbf{s}, t \mid \mathcal{H}_t) = \mu(\mathbf{s}) + \sum_{j : t_j < t} g(\mathbf{s} - \mathbf{s}_j, \, t - t_j),
$$
where $\mu(\mathbf{s})$ is a baseline background Poisson intensity, and $g(\mathbf{h}, u) \ge 0$ is a spatio-temporal triggering kernel governing how a past event induces secondary events in its geographic and temporal vicinity.

Hawkes processes form the mathematical foundation of:
* **Earthquake Seismology:** The Epidemic-Type Aftershock Sequence (ETAS) model (Ogata 1988), where each seismic shock triggers a cascade of secondary aftershocks whose rate decays as a power law in time (Omori's law) and distance.
* **Urban Crime Modeling:** Predictive policing models (Mohler et al. 2011), where residential burglaries or street crimes exhibit short-term "near-repeat" contagion effects within several city blocks and days following an initial incident. Diggle (2013) provides a comprehensive treatise on spatio-temporal point process models.

---

## D.11 Generalized Method of Moments and Composite Likelihood for Complex Spatial Models

Throughout Chapter 12 (§12.2.5 and §12.6), we confronted the computational barrier of Gibbs point processes: the normalizing constant (partition function)
$$
Z(\boldsymbol{\theta}) = \int_{\mathcal{D}^n} \exp\left( -U(\mathbf{s}_1, \dots, \mathbf{s}_n; \boldsymbol{\theta}) \right) \, d\mathbf{s}_1 \cdots d\mathbf{s}_n
$$
is an intractable high-dimensional integral over variable point counts. To circumvent $Z(\boldsymbol{\theta})$, we introduced Besag's pseudolikelihood, which replaces the joint density with the product of full conditional Papangelou intensities.

However, pseudolikelihood is known to suffer from substantial efficiency loss when spatial interactions are moderate to strong. Two advanced estimation frameworks provide rigorous alternatives.

### Generalized Method of Moments (GMM) for Point Processes

In Section 12.2.5, we applied **minimum contrast estimation**, fitting theoretical summary curves $K(r; \boldsymbol{\theta})$ or $g(r; \boldsymbol{\theta})$ to empirical estimates $\hat{K}(r)$ or $\hat{g}(r)$ via non-linear least squares.

Diggle (2003) and Guan (2006) formalized minimum contrast as an optimal **Generalized Method of Moments (GMM)** estimator. Let $\mathbf{v} = (\hat{K}(r_1), \dots, \hat{K}(r_m))'$ be a vector of empirical summaries evaluated across $m$ distance lags, and let $\boldsymbol{\mu}_K(\boldsymbol{\theta})$ denote their theoretical expectation. The GMM estimator solves:
$$
\hat{\boldsymbol{\theta}}_{\text{GMM}} = \arg\min_{\boldsymbol{\theta}} \left( \mathbf{v} - \boldsymbol{\mu}_K(\boldsymbol{\theta}) \right)' \mathbf{C}_{\mathbf{v}}^{-1} \left( \mathbf{v} - \boldsymbol{\mu}_K(\boldsymbol{\theta}) \right),
$$
where $\mathbf{C}_{\mathbf{v}} = \operatorname{Var}(\mathbf{v})$ is the covariance matrix of the empirical summary vector. Guan (2006) proved that estimating $\mathbf{C}_{\mathbf{v}}$ via spatial block-resampling yields $\sqrt{n}$-consistent and asymptotically normal parameter estimates, placing minimum contrast on solid inferential footing.

### Composite Likelihood Estimation

**Composite likelihood** (Varin, Reid, and Firth 2011) constructs an objective function by multiplying together a collection of valid, low-dimensional marginal or conditional event densities, bypassing the intractable joint distribution:
$$
\mathcal{L}_{\text{comp}}(\boldsymbol{\theta}) = \prod_{i=1}^{n-1} \prod_{j=i+1}^n p(\mathbf{s}_i, \mathbf{s}_j \mid \boldsymbol{\theta}).
$$
In spatial point patterns and max-stable extreme-value fields, the **pairwise composite likelihood** evaluates the joint probability of pairs of points located within a threshold distance $r_{\max}$.

Because each individual pairwise score function has mean zero, composite likelihood estimators $\hat{\boldsymbol{\theta}}_{\text{CL}}$ are guaranteed to be consistent and asymptotically normal:
$$
\sqrt{n}(\hat{\boldsymbol{\theta}}_{\text{CL}} - \boldsymbol{\theta}_0) \xrightarrow{d} \mathcal{N}\left( \mathbf{0}, \, \mathcal{G}^{-1}(\boldsymbol{\theta}_0) \right),
$$
where $\mathcal{G}(\boldsymbol{\theta})$ is the **Godambe information matrix** (or sandwich variance estimator):
$$
\mathcal{G}(\boldsymbol{\theta}) = \mathbf{H}(\boldsymbol{\theta}) \, \mathbf{J}(\boldsymbol{\theta})^{-1} \, \mathbf{H}(\boldsymbol{\theta}).
$$
Here $\mathbf{H}(\boldsymbol{\theta}) = \operatorname{E}[-\nabla^2 \log \mathcal{L}_{\text{comp}}(\boldsymbol{\theta})]$ is the expected Hessian matrix (sensitivity matrix) and $\mathbf{J}(\boldsymbol{\theta}) = \operatorname{Var}(\nabla \log \mathcal{L}_{\text{comp}}(\boldsymbol{\theta}))$ is the covariance matrix of the composite score vector. Baddeley, Coeurjolly, and Rubak (2014) develop composite likelihood methods specifically for Gibbs and cluster point processes.

---

## D.12 Spatial Machine Learning and Geographically Weighted Regression

In Section 4.5.3 and Section 13.7, we observed that simple kriging is mathematically identical to Gaussian process regression, while ordinary kriging is equivalent to kernel ridge regression. As spatial datasets grow to millions of records, the boundaries between classical spatial statistics and modern machine learning have blurred.

### Geographically Weighted Regression (GWR)

In standard spatial regression, the relationship between $X$ and $Y$ is assumed to be spatially homogeneous: a one-unit change in $X$ produces an identical expected change $\beta$ everywhere in the domain. 

**Geographically Weighted Regression (GWR)** (Brunsdon, Fotheringham, and Charlton 1996; Fotheringham, Brunsdon, and Charlton 2002) relaxes this assumption by allowing regression coefficients to vary continuously across space:
$$
Y(\mathbf{s}) = \beta_0(\mathbf{s}) + \sum_{k=1}^p \beta_k(\mathbf{s}) X_k(\mathbf{s}) + \varepsilon(\mathbf{s}).
$$
To estimate $\boldsymbol{\beta}(\mathbf{s}_0)$ at an arbitrary target location $\mathbf{s}_0$, GWR solves a locally weighted least squares problem:
$$
\hat{\boldsymbol{\beta}}(\mathbf{s}_0) = \left( \mathbf{X}' \mathbf{W}(\mathbf{s}_0) \mathbf{X} \right)^{-1} \mathbf{X}' \mathbf{W}(\mathbf{s}_0) \mathbf{Y},
$$
where $\mathbf{W}(\mathbf{s}_0) = \operatorname{diag}(K(\|\mathbf{s}_1 - \mathbf{s}_0\| / b), \dots, K(\|\mathbf{s}_n - \mathbf{s}_0\| / b))$ is a diagonal weight matrix defined by a continuous spatial kernel $K(\cdot)$ with bandwidth $b$.

*GWR vs. Spatially Varying Coefficients (SVC):* While GWR is an algorithmic local smoother, modern spatial statistics generally prefers **Spatially Varying Coefficient (SVC) models** (Gelfand et al. 2003). SVC places formal Gaussian process priors on the coefficient surfaces $\beta_k(\mathbf{s}) \sim \mathcal{GP}(0, \boldsymbol{\Sigma}_k)$, embedding local parameter variation directly inside a coherent Bayesian hierarchical model with rigorous uncertainty quantification.

### Spatial Machine Learning and Spatial Cross-Validation

Standard machine learning algorithms (Random Forests, Gradient Boosting, Deep Neural Networks) assume that training samples are independent and identically distributed (i.i.d.). When applied to spatial data, feeding geographic coordinates $(s_1, s_2)$ as raw features into standard models produces severe pitfalls:
1. **Over-Optimistic Performance under Random Cross-Validation:** If observations are randomly assigned to cross-validation folds, training and test sets contain points located millimeters apart. Due to spatial autocorrelation, the algorithm "memorizes" spatial proximity rather than learning generalizable functional relationships. Roberts et al. (2017) demonstrate that evaluating spatial models requires **spatial block cross-validation**, where entire geographic regions are held out during training.
2. **Physics-Informed Neural Networks (PINNs) and Neural Operators:** For massive spatio-temporal systems (e.g., climate dynamics and fluid flow), modern deep learning embeds the SPDEs of Section 13.5 directly into neural loss functions. Physics-Informed Neural Networks (Raissi, Perdikaris, and Karniadakis 2019) and Fourier Neural Operators (Kovachki et al. 2023) solve continuous spatial partial differential equations on arbitrary domains, linking classical spatial calculus directly to deep learning architectures.

---

## Summary Guide to Further Reading

| Frontier Topic | Foundational Starting Papers | Authoritative Advanced Textbooks |
| :--- | :--- | :--- |
| **Spatial Design (§D.1)** | Diggle & Lophaven (2006); Zimmerman (2006) | Müller (2007) *Spatial Design* |
| **Beyond Plug-in (§D.2)** | Harville & Jeske (1992) | Diggle & Ribeiro (2007) *Model-based Geostatistics* |
| **Spatial Confounding (§D.3)** | Reich et al. (2006); Hanks et al. (2015) | Papadogeorgou et al. (2019) |
| **SPDE & INLA (§D.4)** | Lindgren et al. (2011); Rue et al. (2009) | Blangiardo & Cameletti (2015) *Spatial and Spatio-temporal Models with R-INLA* |
| **Space-Time & Cokriging (§D.5)** | Gneiting (2002); Gneiting et al. (2010) | Wackernagel (2003) *Multivariate Geostatistics* |
| **Non-Gaussian & Copulas (§D.6)** | Bárdossy (2006); Kazianka & Pilz (2010) | Davison, Padoan, & Ribatet (2012) *Statistical Modeling of Spatial Extremes* |
| **Harmonic Analysis (§D.7)** | Bochner (1932); Schoenberg (1938) | Stein (1999) *Interpolation of Spatial Data* |
| **Spherical Geostatistics (§D.8)** | Gneiting (2013); Jeong & Jun (2015) | Guinness & Fuentes (2016) |
| **Spatial Durbin (§D.9)** | LeSage & Pace (2009); Elhorst (2014) | LeSage & Pace (2009) *Introduction to Spatial Econometrics* |
| **Marked & Hawkes (§D.10)** | Ogata (1988); Mohler et al. (2011) | Illian et al. (2008) *Statistical Analysis and Modelling of Spatial Point Patterns* |
| **GMM & Composite Likelihood (§D.11)**| Guan (2006); Varin, Reid, & Firth (2011) | Baddeley, Coeurjolly, & Rubak (2014) |
| **Spatial ML & GWR (§D.12)** | Brunsdon et al. (1996); Raissi et al. (2019) | Fotheringham, Brunsdon, & Charlton (2002) *Geographically Weighted Regression* |
