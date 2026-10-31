# Appendix C: A Jargon Rosetta Stone

Spatial statistics has developed across several distinct applied fields: mining engineering (geostatistics), regional science and econometrics (areal data and spatial autoregression), forestry and plant ecology (point processes), and climate science (space-time fields). Each community invented its own vocabulary, often giving novel names to standard concepts in applied linear models, multivariate analysis, and generalized linear models.

This appendix serves as a translator. Its goal is anti-gatekeeping: to demystify domain-specific spatial jargon by mapping every term directly to its underlying linear-model or GLM counterpart. When you encounter a term in the literature, use this guide to identify what matrix equation, projection, random effect, or likelihood construction is actually being described.

The reader-facing notation guide and the concordance between this book's notation and that of Schabenberger and Gotway (2005) are located in `NOTATION.md` at the front of the book.

---

## C.1 Geostatistics and Continuous Fields

Continuous spatial processes model a response $Y(\mathbf{s})$ observed at geographic coordinates $\mathbf{s} \in \mathcal{D} \subset \mathbb{R}^2$, anchored by the core decomposition $Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})$.

| Spatial Term | Linear Model / GLM Translation | Mathematical Representation | First Introduced |
| :--- | :--- | :--- | :--- |
| **Kriging (Simple)** | Best Linear Unbiased Predictor (BLUP) with known mean | $\hat{Y}(\mathbf{s}_0) = \mu_0 + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu})$ | §4.2 |
| **Kriging (Ordinary)** | BLUP with an unknown constant mean $\mu$ | $\hat{Y}(\mathbf{s}_0) = \hat{\mu}_{\text{GLS}} + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \hat{\mu}_{\text{GLS}}\mathbf{1}_n)$ | §4.3 |
| **Kriging (Universal)** | BLUP with linear regression mean $\mathbf{X}\boldsymbol{\beta}$ estimated by GLS | $\hat{Y}(\mathbf{s}_0) = \mathbf{x}_0'\hat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})$ | §4.4 |
| **Kriging Variance** | Mean squared prediction error (MSPE) of the BLUP | $\operatorname{Var}(\hat{Y}(\mathbf{s}_0) - \tilde{Y}(\mathbf{s}_0))$ | §4.2.2 |
| **Block Kriging** | BLUP of a linear functional (spatial average) of the process | $\frac{1}{\operatorname{area}(B)}\int_B Y(\mathbf{s})\,d\mathbf{s}$ | §4.6 |
| **Nugget ($\tau^2$)** | Variance of independent observation error (white noise) | $\operatorname{Var}(\varepsilon(\mathbf{s})) = \tau^2$ | §2.1.3 |
| **Partial Sill ($\sigma^2$)** | Variance scale of the spatially correlated random effect | $\operatorname{Var}(u(\mathbf{s})) = \sigma^2$ | §2.1.2 |
| **Sill ($\sigma^2 + \tau^2$)** | Total marginal variance of the observation process | $\operatorname{Var}(Y(\mathbf{s})) = \sigma^2 + \tau^2$ | §2.1.3 |
| **Range ($\phi$)** | Correlation length-scale parameter | Distance governing exponential decay rate $e^{-h/\phi}$ | §2.3.1 |
| **Practical Range** | Distance at which spatial correlation falls to $0.05$ | $3\phi$ (exponential), $\sqrt{3}\phi$ (Gaussian), $\approx 0.811\phi$ (spherical) | §2.3.4 |
| **Semivariogram ($\gamma(h)$)** | Half the variance of a pairwise difference | $\gamma(h) = \frac{1}{2}\operatorname{Var}(Y(\mathbf{s}) - Y(\mathbf{s}+\mathbf{h}))$ | §2.2.1 |
| **Second-Order Stationarity** | Constant mean and covariance depending only on displacement $\mathbf{h}$ | $\operatorname{Cov}(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h})) = C(\mathbf{h})$ | §2.2.2 |
| **Intrinsic Stationarity** | Constant mean and variance of differences depending only on $\mathbf{h}$ | $\frac{1}{2}\operatorname{Var}(Y(\mathbf{s}) - Y(\mathbf{s}+\mathbf{h})) = \gamma(\mathbf{h})$ | §2.2.3 |
| **Isotropy** | Spatial dependence invariant to direction, depending only on distance $h$ | $C(\mathbf{h}) = C(\lVert\mathbf{h}\rVert)$ | §2.2.4 |
| **Anisotropy (Geometric)** | Elliptical correlation contours transformed by a linear coordinate map | $C(\mathbf{h}) = C_0(\sqrt{\mathbf{h}'\mathbf{G}\mathbf{h}})$ for positive definite $\mathbf{G}$ | §2.6 |
| **Screening Effect** | Conditional independence property of spatial Markov fields | Intervening observations screen off correlation from distant points | §4.2.3 |
| **Change of Support** | Aggregation or change in spatial resolution of the response variable | The Modifiable Areal Unit Problem (MAUP) in continuous space | §4.6.1 |
| **REML** | Maximum likelihood on error contrasts orthogonal to $\mathbf{X}$ | Maximizing likelihood on $\mathbf{T}'\mathbf{Y}$ where $\mathbf{T}'\mathbf{X} = \mathbf{0}$ | §5.3 |
| **Spatial Confounding** | Collinearity between smooth spatial covariates and spatial random effects | $\mathbf{x}$ collinear with the dominant eigenvectors of $\boldsymbol{\Sigma}$ | §5.6.2 |
| **LOOCV Shortcut Residual** | Scaled leave-one-out residual computed directly from $\boldsymbol{\Pi}$ | $e_{-i} = -[\boldsymbol{\Pi}\mathbf{y}]_i / \Pi_{ii}$ | §4.7.2 |

### Detailed Concepts and Translations

**Kriging as Best Linear Unbiased Prediction (BLUP).** Mining engineers named spatial interpolation after Danie Krige. In statistical theory, kriging is identical to Gauss–Markov best linear unbiased prediction under a spatial covariance structure. When the process is Gaussian, kriging is also the conditional expectation $\operatorname{E}[\tilde{Y}(\mathbf{s}_0) \mid \mathbf{Y}]$, which is the minimum mean-squared-error predictor of any form. In modern machine learning, simple kriging is known as **Gaussian process regression (GPR)**, and ordinary kriging is equivalent to **kernel ridge regression** with an intercept constraint (§4.5.3).

**Nugget, Sill, and Range.** Geostatistical variogram terms reflect gold-grade assays:
- The *nugget effect* originally referred to microscopic gold nuggets causing coin-toss assay variability between adjacent core samples. In linear models, the nugget is simply independent observation error variance $\tau^2$ (or measurement error plus micro-scale variation below the shortest sampling distance).
- The *sill* is the horizontal plateau of the variogram, corresponding to the total marginal variance $\operatorname{Var}(Y(\mathbf{s})) = \sigma^2 + \tau^2$.
- The *partial sill* is the variance component $\sigma^2$ explained by the smooth spatial random effect $u(\mathbf{s})$.
- The *range* is the distance where spatial dependence decays to negligible levels. Because the exponential covariance function approaches zero asymptotically without hitting it, the *practical range* is defined as the distance where correlation drops to $0.05$.

**The Semivariogram.** Matheron defined the semivariogram as $\gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Y(\mathbf{s}) - Y(\mathbf{s}+\mathbf{h}))$. The prefix *semi-* ("half") accounts for the factor of $1/2$, which ensures that for stationary processes, $\gamma(h) = C(0) - C(h)$. The variogram can exist even when the marginal variance $C(0)$ is infinite (e.g. for intrinsic random walks and Brownian motion), which is why geostatisticians historically preferred variograms to covariances.

**The Screening Effect.** When predicting at a new site $\mathbf{s}_0$, the nearest surrounding observations receive large kriging weights, while observations behind them receive weights close to zero, even if they are strongly correlated with $\mathbf{s}_0$. In graphical models, this is conditional independence: intervening points form a Markov blanket that screens off distant points (§4.2.3).

**Spatial Confounding.** When an applied researcher fits a spatial regression with smooth geographical covariates (such as east-west coordinate or distance to coast), the estimated regression slope often changes drastically compared to OLS, and its standard error inflates. This is not a pathology of spatial statistics; it is multicollinearity. A smooth covariate projects heavily onto the top eigenvectors of $\boldsymbol{\Sigma}$, competing directly with the smooth spatial random effect $\mathbf{u}$ for the exact same variance (§5.6.2).

---

## C.2 Areal Data, Spatial Weights, and Lattice Models

Areal models analyze observations aggregated into discrete geographic regions (polygons, counties, census tracts) represented as nodes on an adjacency graph.

| Spatial Term | Linear Model / GLM Translation | Mathematical Representation | First Introduced |
| :--- | :--- | :--- | :--- |
| **MAUP** | Aggregation distortion due to grouping and zoning of spatial units | Variance reduction and correlation inflation under spatial averaging | §6.1.1 |
| **Ecological Fallacy** | Conflating aggregate group-level associations with individual-level relationships | $\operatorname{Cov}(\bar{X}, \bar{Y}) \ne \operatorname{Cov}(X_i, Y_i)$ | §6.1.2 |
| **Spatial Weights Matrix ($\mathbf{W}$)** | Graph adjacency matrix, typically row-standardized | $w_{ij} = A_{ij} / d_i$ where $\sum_j w_{ij} = 1$ | §6.2.1 |
| **Spatial Lag ($\mathbf{W}\mathbf{Y}$)** | Local neighborhood moving average | $[\mathbf{W}\mathbf{Y}]_i = \sum_j w_{ij}Y_j$ | §6.2.2 |
| **Moran's $I$** | Ratio of quadratic forms measuring autocorrelation of data or residuals | $I = \frac{n}{S_0}\frac{\mathbf{z}'\mathbf{W}\mathbf{z}}{\mathbf{z}'\mathbf{z}}$ | §6.4.1 |
| **Geary's $C$** | Normalized sum of squared differences across graph edges | $C = \frac{n-1}{2S_0}\frac{\sum_{i,j}A_{ij}(y_i - y_j)^2}{\sum_i(y_i - \bar{y})^2}$ | §6.4.2 |
| **Local Moran ($I_i$)** | Decomposition of global Moran's $I$ into region-specific components | $I_i = z_i \sum_j w_{ij} z_j / s^2$ | §6.7.1 |
| **SAR (Spatial Lag Model)** | Autoregressive simultaneous equation with endogenous spatial lag | $\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ | §7.1 |
| **SAR (Spatial Error Model)** | Linear regression with autoregressive spatial error term | $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}, \quad \mathbf{u} = \rho\mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}$ | §7.1 |
| **Feedback Matrix ($\boldsymbol{\Phi}$)** | Spatial multiplier matrix expanding neighborhood spillovers | $\boldsymbol{\Phi} = (\mathbf{I}_n - \rho\mathbf{W})^{-1} = \sum_{k=0}^\infty \rho^k\mathbf{W}^k$ | §7.3.1 |
| **Impacts (Direct, Indirect, Total)** | Summary averages of the matrix of partial derivatives $\partial Y_i / \partial X_{jk}$ | Average diagonal, off-diagonal, and row sums of $\boldsymbol{\Phi}\beta_k$ | §7.7.1 |
| **CAR (Proper)** | Gaussian Markov random field specified by full conditional normals | $Y_i \mid \mathbf{Y}_{-i} \sim N(\mu_i + \rho\sum_j w_{ij}(Y_j - \mu_j),\, \sigma^2/d_i)$ | §8.1.1 |
| **ICAR (Intrinsic CAR)** | Rank-deficient GMRF with graph Laplacian precision matrix | $\mathbf{Q} = \sigma^{-2}(\mathbf{D} - \mathbf{A})$, rank $n - 1$ | §8.3.1 |
| **Graph Laplacian ($\mathcal{L}$)** | Difference between degree matrix and adjacency matrix | $\mathcal{L} = \mathbf{D} - \mathbf{A}$ | §8.3.1 |
| **BYM Model** | Convolution mixed model: structured spatial effect + independent effect | $\eta_i = \mathbf{x}_i'\boldsymbol{\beta} + u_i + v_i$ ($u_i$ ICAR, $v_i$ i.i.d.) | §9.3.3 |
| **BYM2 Model** | Scaled, orthogonalized reparameterization of the BYM model | $\mathbf{u}^* = \sigma \left(\sqrt{\rho/s_{\text{geom}}}\,\mathbf{u} + \sqrt{1-\rho}\,\mathbf{v}\right)$ | §9.3.4 |
| **Leroux Model** | Convex combination precision bridging independence and ICAR | $\mathbf{Q} = \sigma^{-2}\left[\rho(\mathbf{D} - \mathbf{A}) + (1 - \rho)\mathbf{I}_n\right]$ | §9.3.5 |
| **Henderson's Equations** | Penalized mixed-model equations for fixed and random effects | $\begin{bmatrix} \mathbf{X}'\mathbf{V}\mathbf{X} & \mathbf{X}'\mathbf{V} \\ \mathbf{V}\mathbf{X} & \mathbf{V} + \mathbf{Q} \end{bmatrix} \begin{bmatrix} \hat{\boldsymbol{\beta}} \\ \hat{\mathbf{u}} \end{bmatrix} = \begin{bmatrix} \mathbf{X}'\mathbf{V}\mathbf{z} \\ \mathbf{V}\mathbf{z} \end{bmatrix}$ | §9.4.2 |
| **Laplace Approximation** | Second-order Taylor series integration over non-Gaussian random effects | Approximating $\int \exp(\ell(\mathbf{u}))\,d\mathbf{u}$ by Gaussian curvature at mode | §9.5.1 |

### Detailed Concepts and Translations

**Moran's $I$ as a Normalized Ratio of Quadratic Forms.** In spatial analysis, Moran's $I$ is presented as the spatial analog of Pearson's correlation coefficient. In matrix terms, it is a ratio of quadratic forms in centered data $\mathbf{z}$:

$$
I = \frac{n}{S_0} \frac{\mathbf{z}'\mathbf{W}\mathbf{z}}{\mathbf{z}'\mathbf{z}} = \frac{n}{S_0} \frac{\mathbf{Y}'\mathbf{P}_c\mathbf{W}\mathbf{P}_c\mathbf{Y}}{\mathbf{Y}'\mathbf{P}_c\mathbf{Y}}
$$

Because the numerator and denominator share the projection matrix $\mathbf{P}_c$, Pitman–Koopmans theory guarantees that under Gaussian independent errors, the ratio is statistically independent of its denominator, allowing exact analytical variance derivations (§6.5.2).

**SAR versus CAR: Covariance versus Precision.** Students often confuse Spatial Autoregressive (SAR) and Conditional Autoregressive (CAR) models because both use the letter $\mathbf{W}$ and an autocorrelation parameter $\rho$:
- A **SAR model** specifies the generative process *simultaneously*: $\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \boldsymbol{\varepsilon}$. Inverting the system gives $\mathbf{Y} = (\mathbf{I}_n - \rho\mathbf{W})^{-1}\boldsymbol{\varepsilon}$. Its covariance matrix is dense: $\boldsymbol{\Sigma} = \sigma^2(\mathbf{I}_n - \rho\mathbf{W})^{-1}(\mathbf{I}_n - \rho\mathbf{W}')^{-1}$.
- A **CAR model** specifies the process through *conditional distributions*: $Y_i \mid \mathbf{Y}_{-i}$. By Brook's lemma, this implies a joint distribution whose *precision matrix* $\mathbf{Q} = \boldsymbol{\Sigma}^{-1}$ is sparse: $Q_{ij} = 0$ whenever regions $i$ and $j$ are not neighbors.
- In short: SAR models define spatial structure through the **covariance matrix**; CAR models define spatial structure through the **precision matrix** (§8.5).

**The ICAR Model as an Improper Laplacian Prior.** The Intrinsic Conditional Autoregressive (ICAR) model sets $\mathbf{Q} = \sigma^{-2}(\mathbf{D} - \mathbf{A}) = \sigma^{-2}\mathcal{L}$. Because every row of the graph Laplacian sums to zero, $\mathcal{L}\mathbf{1}_n = \mathbf{0}$, meaning $\mathbf{Q}$ has a zero eigenvalue and cannot be inverted. In Bayesian linear models, ICAR is an improper prior on spatial random effects with an undefined global intercept. Adding a sum-to-zero constraint $\sum_i u_i = 0$ projects the model onto the orthogonal complement of $\mathbf{1}_n$, allowing unique posterior modes (§8.3).

**Impacts in Spatial Lag Models.** In classical regression, the effect of covariate $X_k$ on $Y_i$ is simply the scalar $\beta_k$. In a spatial lag model, raising $X_{ik}$ directly changes $Y_i$, which spills over to increase neighbor values $Y_j$ through $\rho\mathbf{W}\mathbf{Y}$, which loops back to further increase $Y_i$. The total effect is captured by the spatial multiplier matrix $\boldsymbol{\Phi}\beta_k = (\mathbf{I}_n - \rho\mathbf{W})^{-1}\beta_k$. Econometricians summarize this dense $n \times n$ matrix into three scalar quantities: *Direct Impact* (average diagonal element), *Indirect Impact* (average off-diagonal row sum, measuring spillovers), and *Total Impact* (sum of direct and indirect, identically equal to $\beta_k / (1 - \rho)$) (§7.7).

---

## C.3 Spatial Point Patterns and Interaction Processes

Point pattern analysis studies configurations of points $\mathbf{S} = \{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathcal{D}$ where the observation locations themselves are stochastic.

| Spatial Term | Linear Model / GLM Translation | Mathematical Representation | First Introduced |
| :--- | :--- | :--- | :--- |
| **CSR** | Homogeneous spatial Poisson point process (constant intensity null) | $\Lambda(B) = \lambda\operatorname{area}(B), \quad N(B) \sim \operatorname{Poisson}(\lambda\operatorname{area}(B))$ | §10.2 |
| **First-Order Intensity ($\lambda(\mathbf{s})$)** | Expected event count per unit area (Poisson regression rate) | $\operatorname{E}[N(B)] = \int_B \lambda(\mathbf{s})\,d\mathbf{s}$ | §10.2 |
| **Ripley's $K(r)$** | Normalized cumulative expected count of pairs within distance $r$ | $K(r) = \frac{1}{\lambda}\operatorname{E}[\text{events within } r \text{ of a typical event}]$ | §10.6 |
| **Besag's $L(r)$** | Variance-stabilizing transformation of Ripley's $K$ | $L(r) = \sqrt{K(r)/\pi}$ (under CSR, $L(r) - r = 0$) | §10.6.3 |
| **Pair Correlation ($g(r)$)** | Ratio of joint second-order density to squared first-order density | $g(r) = K'(r)/(2\pi r)$ ($g(r) = 1$ under CSR) | §10.6.4 |
| **Berman–Turner Device** | Discretization of Poisson point-process log-likelihood into a weighted GLM | Log-likelihood approximated by $\sum_k w_k(y_k\log\mu_k - \mu_k)$ | §11.3 |
| **Logistic Bridge** | Case-control point pattern modeled as binary logistic regression | $\Pr(\text{case} \mid \mathbf{s}) = \frac{\exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})}{1 + \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})}$ | §11.4 |
| **Scan Statistic** | Maximized Poisson GLM likelihood ratio test across windows | $\operatorname{LLR}_w = n_w\log\frac{n_w}{e_w} + (n-n_w)\log\frac{n-n_w}{n-e_w}$ | §11.6 |
| **Minimum Contrast** | Nonlinear least squares (moment matching / GMM) on summary functions | $\min_{\boldsymbol{\theta}}\int_{r_{\min}}^{r_{\max}} \left(\hat{K}(r)^c - K(r;\boldsymbol{\theta})^c\right)^2 dr$ | §11.8, §12.2.5 |
| **Neyman–Scott Process** | Clustered Poisson cluster process with random offspring per parent | Unobserved parent Poisson process thinned and replaced by offspring | §12.2.1 |
| **Thomas Process** | Neyman–Scott process with isotropic Gaussian offspring dispersal | Dispersal density $f(\mathbf{d}) = \frac{1}{2\pi\sigma_c^2}\exp(-\lVert\mathbf{d}\rVert^2/(2\sigma_c^2))$ | §12.2.2 |
| **Matérn Cluster Process** | Neyman–Scott process with uniform disc offspring dispersal | Offspring uniform on $b(\mathbf{0}, R)$ | §12.2.2 |
| **LGCP** | Poisson regression with continuous spatial Gaussian random field prior | $\log\Lambda(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + u(\mathbf{s}), \quad u \sim \operatorname{GP}(0, C_u)$ | §12.3.1 |
| **Hard-Core Process** | Point process with strict minimum inter-event exclusion distance $r_0$ | Type I (independent deletion) or Type II (ordered deletion) | §12.4.1 |
| **Sequential Inhibition (SSI)** | Random sequential packing process with hard-core exclusion | Generates configurations up to the jamming limit ($0.5472$) | §12.4.2 |
| **Strauss Process** | Gibbs point process with pairwise interaction parameter $\gamma = e^\zeta \le 1$ | Likelihood density $f(\mathbf{S}) \propto b^{n(\mathbf{S})}\gamma^{\operatorname{pairs}(\mathbf{S})}$ | §12.4.3 |
| **Papangelou Intensity** | Conditional intensity of an event at $\mathbf{u}$ given existing pattern $\mathbf{S}$ | $\lambda^*(\mathbf{u} \mid \mathbf{S}) = \frac{f(\mathbf{S}\cup\{\mathbf{u}\})}{f(\mathbf{S})} = b\gamma^{t(\mathbf{u}, \mathbf{S})}$ | §12.4.3 |
| **Ruelle Stability Failure** | Explosive probability mass collapse when $\gamma > 1$ | Divergent partition function $Z \to \infty$ for clustering Gibbs models | §12.4.3 |
| **Pseudolikelihood** | Approximate composite likelihood product of Papangelou conditionals | $\operatorname{PL}(\mathbf{S}) = \prod_i \lambda^*(\mathbf{s}_i \mid \mathbf{S}\setminus\mathbf{s}_i)\exp\left(-\int_{\mathcal{D}}\lambda^*(\mathbf{u}\mid\mathbf{S})d\mathbf{u}\right)$ | §12.4.5 |

### Detailed Concepts and Translations

**The Berman–Turner Quadrature Device as a Weighted GLM.** The continuous log-likelihood of an inhomogeneous Poisson process is $\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \log\lambda(\mathbf{s}_i) - \int_{\mathcal{D}}\lambda(\mathbf{u})\,d\mathbf{u}$. The integral has no analytical form when $\lambda$ depends on complex spatial covariates. Berman and Turner (1992) discretized the domain into $m$ quadrature tiles of area $a_k$, creating a synthetic data set of observed events plus dummy quadrature points. Setting pseudo-response $y_k = \mathbb{1}_k / a_k$ and quadrature weights $w_k = a_k$, the likelihood becomes:

$$
\ell(\boldsymbol{\beta}) \approx \sum_{k=1}^{n+m} w_k \left[ y_k \log\lambda(\mathbf{s}_k) - \lambda(\mathbf{s}_k) \right]
$$

This is algebraically identical to the log-likelihood of an iteratively reweighted Poisson generalized linear model with log link, weights $w_k$, and offset $\log w_k$. Thus, point process models are fitted with standard GLM code (§11.3).

**The Logistic Bridge for Case-Control Data.** Epidemiologists frequently collect disease case coordinates alongside healthy control locations. If cases arise from intensity $\lambda_1(\mathbf{s}) = \lambda_{\text{pop}}(\mathbf{s})\exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$ and controls from baseline population density $\lambda_0(\mathbf{s}) = \lambda_{\text{pop}}(\mathbf{s})$, the baseline intensity $\lambda_{\text{pop}}(\mathbf{s})$ cancels out of the conditional probability:

$$
\Pr(\text{case} \mid \text{event at } \mathbf{s}) = \frac{\lambda_1(\mathbf{s})}{\lambda_0(\mathbf{s}) + \lambda_1(\mathbf{s})} = \frac{\exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})}{1 + \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})}
$$

Conditioning on event locations converts an intractable inhomogeneous point process problem into ordinary binary logistic regression (§11.4).

**Minimum Contrast as Moment-Matching GMM.** Neyman–Scott and Thomas cluster models have intractable likelihoods because parent locations are unobserved. Minimum contrast bypasses the likelihood by matching the empirical summary function $\hat{K}(r)$ to its analytical theoretical counterpart $K(r;\boldsymbol{\theta})$. In econometric terminology, this is a generalized method of moments (GMM) or M-estimator. Because $\operatorname{Var}(\hat{K}(r)) \propto K(r)^2$, using the fourth-root power transformation $c = 1/4$ stabilizes the variance across integration radii, exactly as Cressie and Hawkins' fourth-root estimator stabilizes variogram variance in Chapter 3 (§12.2.5).

**The Strauss Model as a Weighted GLM with a Neighbor Count.** Gibbs models specify interactions through pairwise potentials. The Strauss density $f(\mathbf{S}) \propto b^n \gamma^{\operatorname{pairs}(\mathbf{S})}$ yields the Papangelou conditional intensity $\lambda^*(\mathbf{u} \mid \mathbf{S}) = b\gamma^{t(\mathbf{u}, \mathbf{S})}$, where $t(\mathbf{u}, \mathbf{S})$ is the number of points in $\mathbf{S}$ within distance $r_0$ of $\mathbf{u}$. Taking logarithms:

$$
\log\lambda^*(\mathbf{u} \mid \mathbf{S}) = \log b + \zeta\,t(\mathbf{u}, \mathbf{S}), \qquad \zeta = \log\gamma
$$

Fitting a Strauss model by maximum pseudolikelihood is identical to running the Berman–Turner Poisson GLM with one additional covariate column containing the neighbor counts $t(\mathbf{s}_k, \mathbf{S})$ (§12.4.4). When $\gamma > 1$ ($\zeta > 0$), adding points increases the conditional rate of adding further points, causing an explosive mathematical collapse onto a single coordinate (Ruelle stability failure).

---

## C.4 Large-Scale Spatial Approximations

When sample size $n$ exceeds several thousand, exact geostatistical likelihood evaluation requires $\mathcal{O}(n^3)$ operations for Cholesky factorization and $\mathcal{O}(n^2)$ memory to store $\boldsymbol{\Sigma}$. Chapter 13 classifies large-$n$ approximations by *which structural matrix they target*.

| Approximation Strategy | Targeted Matrix Structure | Core Mechanism | Computational Cost | First Introduced |
| :--- | :--- | :--- | :--- | :--- |
| **Low-Rank / Predictive Process** | Low rank in covariance: $\boldsymbol{\Sigma} \approx \tau^2\mathbf{I}_n + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'$ | Project onto $q \ll n$ knots via conditional expectations; invert via Woodbury | $\mathcal{O}(nq^2 + q^3)$ | §13.2 |
| **Variance Deficit Correction** | Diagonal correction in low-rank models | Add diagonal variance deficit $\operatorname{diag}(\sigma^2 - \mathbf{b}_i'\boldsymbol{\Sigma}_\alpha\mathbf{b}_i)$ to repair coverage | $\mathcal{O}(nq^2)$ | §13.2.5 |
| **Covariance Tapering** | Structural sparsity in covariance: $\boldsymbol{\Sigma} \circ \mathbf{K}_{\text{Wendland}}$ | Multiply entrywise by compactly supported taper; factor with sparse Cholesky | Sparse Cholesky / LU | §13.3 |
| **Vecchia Approximation** | Sparsity in Cholesky factor of precision: $\mathbf{L}^{-1}$ | Condition each point on at most $k \ll n$ nearest previous neighbors | $\mathcal{O}(nk^3)$ | §13.4 |
| **Vecchia DAG & Moral Graph** | Gaussian Directed Acyclic Graph | Ordered conditioning yields directed DAG; precision $\widetilde{\mathbf{Q}}$ is undirected moral graph | $\mathcal{O}(nk^3)$ | §13.4.1 |
| **SPDE / GMRF Bridge** | Structural sparsity in precision: $\mathbf{Q} \approx (\kappa^2\mathbf{I} + \mathcal{L})^\alpha$ | Equivalence between continuous Matérn fields and discrete GMRFs | Sparse Cholesky / band solver | §13.5 |
| **Neumann Boundary Reflection** | Boundary variance inflation in GMRF grids | Zero-flux boundary conditions act as a "hall of mirrors" doubling edge variance | Diagnostic | §13.5.2 |
| **Separable Kronecker Product** | Multiplicative factorization: $\boldsymbol{\Sigma}_T \otimes \boldsymbol{\Sigma}_S$ | Space and time decorrelate; factor space and time matrices independently | $\mathcal{O}(n_S^3 + n_T^3)$ | §13.6 |

### Detailed Concepts and Translations

**Low-Rank Models and the Variance Deficit.** Low-rank methods (predictive processes, fixed-rank kriging) approximate the infinite-dimensional spatial field by projecting it onto $q \ll n$ spatial knots: $\mathbf{u} \approx \mathbf{B}\boldsymbol{\alpha}$. Because the number of basis functions is small, the covariance becomes diagonal-plus-low-rank: $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n + \mathbf{B}\boldsymbol{\Sigma}_\alpha\mathbf{B}'$. The Sherman–Morrison–Woodbury identity flips inversion from size $n \times n$ to $q \times q$. However, projection acts as a conditional expectation, which strictly discards variance: $\operatorname{Var}(u(\mathbf{s}) \mid \boldsymbol{\alpha}) = \sigma^2(1 - \mathbf{r}'\mathbf{R}_K^{-1}\mathbf{r}) \ge 0$. This **variance deficit** causes plain predictive processes to produce severely overconfident prediction intervals in knot-sparse regions. The *modified predictive process* (Banerjee et al., 2008) adds this discarded variance back into the diagonal, restoring valid 95% interval coverage (§13.2.5).

**Vecchia's Approximation as a Gaussian DAG.** Vecchia (1988) factors the joint multivariate normal density into a cascade of univariate conditionals:

$$
f(\mathbf{y}) = f(y_1)\prod_{i=2}^n f(y_i \mid y_{i-1}, \dots, y_1) \approx f(y_1)\prod_{i=2}^n f(y_i \mid \mathbf{y}_{\mathcal{C}_i})
$$

where $\mathcal{C}_i \subset \{1, \dots, i-1\}$ contains the $k \ll n$ nearest predecessors of point $i$. This approximation corresponds to a Gaussian Directed Acyclic Graph (DAG) with linear structural equations $\mathbf{y} = \boldsymbol{\mu} + \boldsymbol{\Gamma}(\mathbf{y}-\boldsymbol{\mu}) + \boldsymbol{\varepsilon}$, where $\boldsymbol{\Gamma}$ is strictly lower-triangular with nonzeros only in conditioning columns $\mathcal{C}_i$. Inverting the structural equation yields the sparse triangular Cholesky factor of the precision matrix:

$$
\mathbf{L}^{-1} = \mathbf{F}^{-1/2}(\mathbf{I}_n - \boldsymbol{\Gamma}), \qquad \widetilde{\mathbf{Q}} = \boldsymbol{\Sigma}^{-1} = (\mathbf{I}_n - \boldsymbol{\Gamma})'\mathbf{F}^{-1}(\mathbf{I}_n - \boldsymbol{\Gamma})
$$

In probabilistic graphical model theory, $\widetilde{\mathbf{Q}}$ is the adjacency matrix of the **moral graph** obtained by marrying co-parents in the DAG (§13.4.1).

**The SPDE Bridge: Continuous Fields Meet Discrete Precision.** Lindgren, Rue, and Lindström (2011) proved that a continuous Matérn Gaussian field $u(\mathbf{s})$ satisfies the stochastic partial differential equation $(\kappa^2 - \Delta)^{\alpha/2}u(\mathbf{s}) = \mathcal{W}(\mathbf{s})$. When discretized onto a regular grid or triangulated mesh, the continuous differential operator translates into a sparse graph Laplacian matrix $\mathcal{L}$. For smoothness $\nu = 1$ in two dimensions ($\alpha = 2$), the precision matrix takes the explicit sparse form $\mathbf{Q} = (\kappa^2\mathbf{I}_n + \mathcal{L})^2$. On an interior grid, this precision is identical to a second-order Spatial Autoregressive (SAR) precision with spatial parameter $\rho = 4 / (\kappa^2 + 4)$. This demonstrates that geostatistical random fields and lattice autoregressive models are not competing frameworks, but continuous and discrete representations of the exact same underlying spatial reality (§13.5).
