# Chapter 9: Generalized Linear Mixed Models for Spatial Count Data

Chapter 0 ended its first tour of spatial pitfalls with a map of county overdose rates that could not be trusted. Counties with few residents filled both ends of the scale, even though every county had the same true risk, because a small count is a noisy count. Chapters 7 and 8 gave us models for lattice data, but they were models for Gaussian responses. Counts of deaths, cases, and events are not Gaussian, and a Gaussian model with constant variance has no way to say that small counts are noisier counts.

This chapter builds the count version of everything in Part III. It has two ingredients, both familiar. The first is the generalized linear model (GLM): the Poisson regression you may have met in Dobson and Barnett or in a regression course, fitted by iteratively reweighted least squares (IRLS). The second is the spatial random effect $u$ of Chapter 2, now given a conditional autoregressive (CAR) distribution from Chapter 8. Put the two together and you have a generalized linear mixed model (GLMM), in which the model equation of this book,

$$
Y = \text{mean} + u + \varepsilon,
$$

lives on the log scale and the independent noise $\varepsilon$ is replaced by Poisson sampling variability.

The running story is disease mapping. A health department has five-year overdose death counts for 144 counties and wants to know which counties are truly worse off, and whether a county-level deprivation index explains the pattern. We will see why the raw ratio of observed to expected deaths misleads, how shrinkage repairs it, how spatial models borrow strength from neighbors, and how all of it is computed. As in earlier chapters, the data are simulated, so we can always compare the estimates with the truth.

> **Chapter at a Glance**
>
> - **Reviewing GLMs (§9.1):** The exponential family, link functions, score equations, and IRLS as Fisher scoring, ending with one iteration on three counties worked by hand.
> - **Spatial Poisson Regression (§9.2):** The model with expected counts as an offset, and why the raw standardized mortality ratio has a variance that depends on the county's size.
> - **Shrinkage and Spatial Random Effects (§9.3):** The gamma-Poisson model of empirical Bayes, then the BYM, BYM2, and Leroux models, with the scaling constant that makes BYM2's parameters interpretable.
> - **Penalized Quasi-Likelihood (§9.4):** Penalized IRLS and Henderson's equations, REML on the working vector, and one iteration by hand.
> - **The Laplace Approximation (§9.5):** An approximate marginal likelihood for the hyperparameters, checked against numerical integration.
> - **Walkthrough and Lab (§9.6):** Raw, shrunken, and spatial estimates of county risk compared with the truth, a likelihood ratio test, weak identification of the mixing parameter, and a coverage experiment that revisits Chapter 1.

---

## 9.1 Reviewing GLMs in the Style of Dobson and Barnett

You have fitted GLMs before, and probably used them as black boxes: choose a family, choose a link, call a function. This section opens the box. We need its parts in matrix form, because the spatial models of this chapter are built by adding one term to the same equations.

### 9.1.1 The Exponential Family

A random variable $Y$ belongs to the (one-parameter) **exponential family** in canonical form if its density or probability function can be written

$$
f(y;\vartheta) = f_0(y)\exp\!\left\{y\vartheta - b(\vartheta)\right\}
$$

for a function $b(\cdot)$ of the **natural parameter** $\vartheta$ and a function $f_0(\cdot)$ that does not involve $\vartheta$. (Dobson and Barnett write the natural parameter as $\theta$; this book keeps $\theta$ for the relative risk below. Families with a free dispersion parameter, such as the Gaussian with unknown variance, add one more function; we do not need it.)

The mean and variance follow from $b$ alone. Write the log-density as $\ell(\vartheta) = \log f_0(y) + y\vartheta - b(\vartheta)$ and differentiate:

$$
\frac{\partial\ell}{\partial\vartheta} = y - b'(\vartheta), \qquad \frac{\partial^2\ell}{\partial\vartheta^2} = -b''(\vartheta)
$$

Two facts hold for any regular density: the expected value of the score $\partial\ell/\partial\vartheta$ is zero, and the expected squared score equals minus the expected second derivative. (Both come from differentiating $\int f\,dy = 1$ once and twice, as in a mathematical statistics course.) The first fact gives

$$
0 = \operatorname{E}[Y] - b'(\vartheta) \quad\Longrightarrow\quad \operatorname{E}[Y] = b'(\vartheta)
$$

and the second gives $\operatorname{E}[(Y - b'(\vartheta))^2] = b''(\vartheta)$, that is,

$$
\operatorname{Var}(Y) = b''(\vartheta)
$$

The variance is the derivative of the mean with respect to the natural parameter: $\operatorname{Var}(Y) = d\mu/d\vartheta$. That is the whole content of the claim that a GLM's variance is a function of its mean.

Three examples cover most of what we use.

| Family | $b(\vartheta)$ | Mean $\mu = b'(\vartheta)$ | Variance $b''(\vartheta)$ | Canonical link $\vartheta = g(\mu)$ |
| :--- | :--- | :--- | :--- | :--- |
| Gaussian, variance 1 | $\vartheta^2/2$ | $\vartheta$ | $1$ | identity |
| Poisson | $e^{\vartheta}$ | $e^{\vartheta}$ | $\mu$ | $\log\mu$ |
| Binomial, $m$ trials | $m\log(1 + e^{\vartheta})$ | $m\,\dfrac{e^{\vartheta}}{1 + e^{\vartheta}}$ | $m\,p(1 - p)$ | $\operatorname{logit}(p)$ |

For the Poisson, the probability function $\mu^y e^{-\mu}/y! = \exp\{y\log\mu - \mu - \log y!\}$ has $\vartheta = \log\mu$, $b(\vartheta) = e^{\vartheta}$, and $f_0(y) = 1/y!$, so mean and variance are both $e^{\vartheta} = \mu$. The binomial row is what we would use for vaccination rates by county, where each county has $m$ residents of whom $Y$ are vaccinated; Exercise 12 adapts this chapter's code to it.

### 9.1.2 The Linear Predictor and the Link

A GLM relates the means of $n$ observations to covariates through a **linear predictor** $\boldsymbol{\eta} = \mathbf{X}\boldsymbol{\beta}$ and a **link function** $g$:

$$
g(\mu_i) = \eta_i = \mathbf{x}_i'\boldsymbol{\beta}
$$

where $\mathbf{x}_i'$ is row $i$ of $\mathbf{X}$. The linear model of Chapter 1 is the Gaussian case with the identity link. For counts we use the log link, which keeps means positive and turns effects into ratios: adding 1 to $x$ multiplies the mean by $e^{\beta_1}$. The link is called **canonical** when it equals the function in the last column of the table, so that $\vartheta_i = \eta_i$. For the Poisson, the log link is canonical. The canonical link has a convenient property that we use below.

### 9.1.3 The Score Equations

Maximum likelihood chooses $\boldsymbol{\beta}$ to maximize $\ell(\boldsymbol{\beta}) = \sum_i\ell_i$. The derivative of $\ell_i$ with respect to $\beta_j$ passes through three links of a chain, $\beta_j \to \eta_i \to \mu_i \to \vartheta_i \to \ell_i$:

$$
\frac{\partial\ell_i}{\partial\beta_j} = \frac{\partial\ell_i}{\partial\vartheta_i}\,\frac{\partial\vartheta_i}{\partial\mu_i}\,\frac{\partial\mu_i}{\partial\eta_i}\,\frac{\partial\eta_i}{\partial\beta_j}
$$

The pieces are $\partial\ell_i/\partial\vartheta_i = y_i - \mu_i$ (from Section 9.1.1), $\partial\vartheta_i/\partial\mu_i = 1/b''(\vartheta_i) = 1/\operatorname{Var}(Y_i)$ (since $\mu = b'(\vartheta)$ gives $d\mu/d\vartheta = b''(\vartheta)$), and $\partial\eta_i/\partial\beta_j = x_{ij}$. Therefore the **score equations** are

$$
\frac{\partial\ell}{\partial\beta_j} = \sum_{i=1}^n\frac{y_i - \mu_i}{\operatorname{Var}(Y_i)}\,\frac{\partial\mu_i}{\partial\eta_i}\,x_{ij} = 0, \qquad j = 1, \dots, p
$$

For a canonical link, $\mu_i = b'(\eta_i)$, so $\partial\mu_i/\partial\eta_i = b''(\eta_i) = \operatorname{Var}(Y_i)$ and the equations collapse to

$$
\mathbf{X}'(\mathbf{y} - \boldsymbol{\mu}) = \mathbf{0}
$$

The residuals are orthogonal to every column of $\mathbf{X}$, exactly as in the normal equations of Chapter 0 (Tool 5). For Poisson regression with an intercept this says $\sum_i y_i = \sum_i\hat{\mu}_i$: the fitted counts add up to the observed counts.

### 9.1.4 Fisher Information and the Weights

The expected information about $\boldsymbol{\beta}$ is the covariance matrix of the score. The score is a sum over independent observations, so for entries $j$ and $k$,

$$
\mathcal{I}(\boldsymbol{\beta})_{jk} = \sum_{i=1}^n\frac{x_{ij}x_{ik}}{\operatorname{Var}(Y_i)}\left(\frac{\partial\mu_i}{\partial\eta_i}\right)^2, \qquad\text{that is,}\qquad \mathcal{I}(\boldsymbol{\beta}) = \mathbf{X}'\mathbf{V}\mathbf{X}
$$

with the diagonal **weight matrix** $\mathbf{V} = \operatorname{diag}(v_i)$, $v_i = (\partial\mu_i/\partial\eta_i)^2/\operatorname{Var}(Y_i)$. For the Poisson with the log link, $\partial\mu_i/\partial\eta_i = \mu_i$ and $\operatorname{Var}(Y_i) = \mu_i$, so

$$
v_i = \mu_i
$$

Dobson and Barnett call this matrix $\mathbf{W}$. This book uses $\mathbf{V}$, because $\mathbf{W}$ is the spatial weights matrix of Chapters 6 and 7. The asymptotic covariance of $\hat{\boldsymbol{\beta}}$ is $\mathcal{I}(\hat{\boldsymbol{\beta}})^{-1} = (\mathbf{X}'\mathbf{V}\mathbf{X})^{-1}$. Compare this with the OLS variance $\tau^2(\mathbf{X}'\mathbf{X})^{-1}$ of Chapter 1: the same formula, with weights.

For the canonical link there is a further simplification. The second derivative of the log-likelihood is $\partial^2\ell/\partial\boldsymbol{\beta}\,\partial\boldsymbol{\beta}' = -\mathbf{X}'\mathbf{V}\mathbf{X}$ with no $\mathbf{y}$ in it, so the observed and expected information coincide.

### 9.1.5 IRLS Is Fisher Scoring

Fisher scoring updates the estimate by adding the inverse expected information times the score:

$$
\boldsymbol{\beta}^{\text{new}} = \boldsymbol{\beta} + \mathcal{I}(\boldsymbol{\beta})^{-1}\,\frac{\partial\ell}{\partial\boldsymbol{\beta}}
$$

(For a canonical link this is the Newton-Raphson method, by the last remark of Section 9.1.4.) Multiply both sides by $\mathcal{I}(\boldsymbol{\beta}) = \mathbf{X}'\mathbf{V}\mathbf{X}$:

$$
\mathbf{X}'\mathbf{V}\mathbf{X}\,\boldsymbol{\beta}^{\text{new}} = \mathbf{X}'\mathbf{V}\mathbf{X}\boldsymbol{\beta} + \frac{\partial\ell}{\partial\boldsymbol{\beta}}
$$

Define the residuals on the scale of the linear predictor, $r_i = (y_i - \mu_i)\,\partial\eta_i/\partial\mu_i$. Since $v_i\,\partial\eta_i/\partial\mu_i = (\partial\mu_i/\partial\eta_i)/\operatorname{Var}(Y_i)$, the score equations of Section 9.1.3 read $\partial\ell/\partial\beta_j = \sum_iv_ix_{ij}r_i$, that is, $\partial\ell/\partial\boldsymbol{\beta} = \mathbf{X}'\mathbf{V}\mathbf{r}$. Substituting, and using $\mathbf{X}\boldsymbol{\beta} = \boldsymbol{\eta}$,

$$
\mathbf{X}'\mathbf{V}\mathbf{X}\,\boldsymbol{\beta}^{\text{new}} = \mathbf{X}'\mathbf{V}(\mathbf{X}\boldsymbol{\beta} + \mathbf{r}) = \mathbf{X}'\mathbf{V}\mathbf{z}, \qquad z_i = \eta_i + (y_i - \mu_i)\,\frac{\partial\eta_i}{\partial\mu_i}
$$

These are the normal equations of a **weighted least squares** problem with weights $\mathbf{V}$ and a **working response** $\mathbf{z}$. Each iteration computes $\boldsymbol{\mu}$, $\mathbf{V}$, and $\mathbf{z}$ from the current $\boldsymbol{\beta}$, solves a weighted regression, and repeats; that is **iteratively reweighted least squares**. For the Poisson with the log link,

$$
z_i = \eta_i + \frac{y_i - \mu_i}{\mu_i}, \qquad v_i = \mu_i
$$

The working response is the current linear predictor plus a first-order correction: $\log y_i \approx \log\mu_i + (y_i - \mu_i)/\mu_i$.

### 9.1.6 By Hand: Three Counties

Take three counties with covariate values $x = -1, 0, 1$ and death counts $y = 2, 3, 7$, and fit the model $\log\mu_i = \beta_0 + \beta_1x_i$. (There is no offset yet; Section 9.2 adds one.) Start at $\boldsymbol{\beta} = (\log 4, 0)'$, where $4$ is the mean count, so every $\mu_i = 4$.

At this start $\mathbf{V} = 4\,\mathbf{I}$ and the working response is $z_i = \log 4 + (y_i - 4)/4 = 1.3863 + (-0.5, -0.25, 0.75)$, that is $\mathbf{z} = (0.886, 1.136, 2.136)'$. With $\mathbf{X}$ having columns $\mathbf{1}$ and $(-1, 0, 1)'$,

$$
\mathbf{X}'\mathbf{V}\mathbf{X} = 4\begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 12 & 0 \\ 0 & 8 \end{bmatrix}, \qquad \mathbf{X}'\mathbf{V}\mathbf{z} = 4\begin{bmatrix} 0.886 + 1.136 + 2.136 \\ -0.886 + 0 + 2.136 \end{bmatrix} = \begin{bmatrix} 16.636 \\ 5 \end{bmatrix}
$$

so $\boldsymbol{\beta}^{\text{new}} = (16.636/12,\ 5/8)' = (1.3863, 0.625)'$. The code repeats this step and prints the first five iterations.

```python
import numpy as np
from scipy import optimize, special, stats, integrate
from scipy.sparse.csgraph import connected_components

x3 = np.array([-1.0, 0.0, 1.0])
y3 = np.array([2.0, 3.0, 7.0])
X3 = np.column_stack([np.ones(3), x3])
beta = np.array([np.log(y3.mean()), 0.0])
for it in range(1, 6):
    eta = X3 @ beta
    mu = np.exp(eta)
    z = eta + (y3 - mu) / mu
    XtVX = (X3.T * mu) @ X3
    XtVz = (X3.T * mu) @ z
    beta_new = np.linalg.solve(XtVX, XtVz)
    print(f"iteration {it}: mu={np.round(mu, 3)}  z={np.round(z, 3)}")
    print(f"  X'VX={np.round(XtVX, 3).tolist()}  X'Vz={np.round(XtVz, 3)}")
    print(f"  beta: {np.round(beta, 4)} -> {np.round(beta_new, 4)}")
    beta = beta_new
```

The coefficients settle at $(1.2416, 0.6709)$ by the fourth iteration, and each iteration changes the working response and weights only slightly after the second. The general function below does the same for any design matrix and an optional offset, and its standard errors come from $(\mathbf{X}'\mathbf{V}\mathbf{X})^{-1}$. As a check, we maximize the Poisson log-likelihood directly with a general-purpose optimizer.

```python
def poisson_irls(y, X, offset=None, iters=50, tol=1e-10):
    """Poisson log-link GLM by IRLS; returns beta, its covariance, mu."""
    offset = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / np.exp(offset).sum())
    for _ in range(iters):
        mu = np.exp(offset + X @ beta)
        z = X @ beta + (y - mu) / mu
        XtV = X.T * mu
        beta_new = np.linalg.solve(XtV @ X, XtV @ z)
        done = np.max(np.abs(beta_new - beta)) < tol
        beta = beta_new
        if done:
            break
    mu = np.exp(offset + X @ beta)
    return beta, np.linalg.inv((X.T * mu) @ X), mu


beta_hat, cov_hat, _ = poisson_irls(y3, X3)
print("IRLS estimate:", np.round(beta_hat, 4),
      " standard errors:", np.round(np.sqrt(np.diag(cov_hat)), 4))


def poisson_negll(b):
    eta = X3 @ b
    return -np.sum(y3 * eta - np.exp(eta) - special.gammaln(y3 + 1))


direct = optimize.minimize(poisson_negll, np.zeros(2), method="BFGS")
print("direct maximization:", np.round(direct.x, 4))
```

The two routes agree: IRLS is maximum likelihood, computed as a sequence of weighted regressions. The slope is $0.6709$ with standard error $0.394$: with three counties, we cannot say much. Chapter 11 reuses this exact algorithm to fit point patterns.

---

## 9.2 Spatial Poisson Regression for Regional Counts

### 9.2.1 The Model Equation Again

Let $Y_i$ be the number of overdose deaths in county $i$ over five years. A county with more people has more deaths even if its risk per person is the same, so we compare each count with an **expected count** $E_i$: the number of deaths that county would have if it had the overall rate. With the overall rate $\hat{r} = \sum_iY_i/\sum_i\text{pop}_i$,

$$
E_i = \hat{r}\cdot\text{pop}_i
$$

(This is called indirect standardization; real analyses also adjust for age and sex, which only changes how $E_i$ is computed.) The **relative risk** $\theta_i$ of county $i$ is the ratio of its mean count to $E_i$, and the **standardized mortality ratio** (SMR) is its raw estimate:

$$
\theta_i = \frac{\operatorname{E}[Y_i \mid u_i]}{E_i}, \qquad \text{SMR}_i = \frac{Y_i}{E_i}
$$

A relative risk of 1.5 means a county has 50% more deaths than expected from its size. (This chapter's relative risk compares a county's mean count with its expected count. Chapters 10 and 11 use the same words for a different ratio: the intensity of cases divided by the intensity of controls at a location. Both are ratios of observed to expected risk, but they are estimated differently.)

The spatial regression model for counts is

$$
Y_i \mid u_i \sim \text{Poisson}\!\left(E_i\exp(\mathbf{x}_i'\boldsymbol{\beta} + u_i)\right), \qquad \mathbf{u} \sim N(\mathbf{0}, \mathbf{Q}^{-1})
$$

On the log scale the mean is $\log\operatorname{E}[Y_i \mid u_i] = \log E_i + \mathbf{x}_i'\boldsymbol{\beta} + u_i$. This is the model equation of the book. The term $\log E_i$ is a known constant added to the linear predictor, called an **offset**. The trend is $\mathbf{x}_i'\boldsymbol{\beta}$ and the spatial random effect $u_i$ has the precision matrix $\mathbf{Q}$ (Sections 7.2.3 and 8.1) (for the ICAR component, $\mathbf{Q}^{-1}$ is read as the constrained generalized inverse of Section 9.3.4). The nugget $\varepsilon$ has not disappeared: its role, independent noise around the smooth surface, is played by the Poisson variation of $Y_i$ around its mean. In the Gaussian model that noise has a free variance $\tau^2$ that we estimate; here it is tied to the mean, because the Poisson variance equals its mean. When the extra variation in the data exceeds that, the random effect $u$ absorbs it, which is one reason $u$ is useful even without spatial structure.

In this model the relative risk is exactly $\theta_i = \exp(\mathbf{x}_i'\boldsymbol{\beta} + u_i)$: the covariates and the random effect act multiplicatively on the ratio of deaths to expected deaths.

Two cautions from Chapter 6 carry over to this model. First, the coefficient $\boldsymbol{\beta}$ describes how risk differs between counties, not how it differs between the people inside them. If counties with higher deprivation have higher overdose risk, that does not show that more deprived individuals are at higher risk, and a county-level coefficient can differ in size or even in sign from the individual-level one (the ecological fallacy of Section 6.1.2). Second, the offset $\log E_i$ is what lets the model handle counties of very different size: the unequal variances that made rates non-exchangeable in Section 6.8.5 enter through the Poisson variance $E_i\theta_i$, so a small county is given little weight automatically instead of being treated as an observation as precise as a large one. The next subsection shows what goes wrong without this.

### 9.2.2 Why the SMR Misleads

Given the relative risk, $Y_i$ is Poisson with mean $E_i\theta_i$ and so has variance $E_i\theta_i$. The SMR divides the count by the constant $E_i$, so

$$
\operatorname{E}[\text{SMR}_i] = \theta_i, \qquad \operatorname{Var}(\text{SMR}_i) = \frac{E_i\theta_i}{E_i^2} = \frac{\theta_i}{E_i}
$$

The SMR is unbiased, but its variance is inversely proportional to the expected count. This is the result of Section 0.3.3, now with the offset: there, with $E_i = \theta\cdot\text{pop}_i$, the variance of the rate was $\theta/\text{pop}_i$.

When every county has relative risk 1, the standard deviation of the SMR is $1/\sqrt{E_i}$. The table shows it for four sizes, together with the 2.5% and 97.5% points of the SMR, computed from the exact Poisson distribution.

```python
print("  E     sd(SMR)   2.5% and 97.5% of SMR when theta = 1")
for E_i in (1, 10, 100, 1000):
    lo = stats.poisson.ppf(0.025, E_i) / E_i
    hi = stats.poisson.ppf(0.975, E_i) / E_i
    print(f"{E_i:5d}  {1 / np.sqrt(E_i):7.3f}   {lo:6.3f}  {hi:6.3f}")
```

| Expected count $E$ | SD of SMR | 2.5% point | 97.5% point |
| :--- | :--- | :--- | :--- |
| 1 | 1.000 | 0.000 | 3.000 |
| 10 | 0.316 | 0.400 | 1.700 |
| 100 | 0.100 | 0.810 | 1.200 |
| 1,000 | 0.032 | 0.938 | 1.062 |

A county expecting one death will show an SMR of 3 or more about 2.5% of the time even though nothing is wrong, and zero deaths, an SMR of 0, far more often than that. A county expecting a thousand deaths will not stray beyond $0.94$ to $1.06$. A map of raw SMRs therefore puts small counties at both extremes. The **funnel plot** of Section 9.6.1 shows this directly: SMR against $E$, with the exact limits as a funnel.

### 9.2.3 What a Poisson GLM That Ignores $u$ Gets Wrong

The simplest alternative to the SMR is a Poisson GLM with covariates but no random effect: $\log\mu_i = \log E_i + \mathbf{x}_i'\boldsymbol{\beta}$, fitted by the IRLS of Section 9.1. It uses the covariates to smooth the estimates, and it is the right model when the counts vary around $\mathbf{x}_i'\boldsymbol{\beta}$ with Poisson noise only.

When the true risks also vary for reasons the covariates miss, which is the setting of this chapter, the counts are more variable than Poisson. This is **overdispersion**. The GLM standard errors, $(\mathbf{X}'\mathbf{V}\mathbf{X})^{-1}$, assume Poisson variation and come out too small. A common repair is the **quasi-Poisson** correction: multiply the covariance by the Pearson dispersion $s_P^2 = \frac{1}{n - p}\sum_i(y_i - \hat{\mu}_i)^2/\hat{\mu}_i$, the Poisson analogue of the residual variance estimate $s^2$ of Chapter 1. That corrects for extra variation that is independent between counties, but not for variation that is spatially correlated, and Chapter 1 has already shown what spatially correlated variation does to a covariate that is itself spatially smooth. Section 9.6.5 measures how well each approach covers.

---

## 9.3 Shrinkage and Spatial Random Effects

The SMR uses only county $i$'s own count. When $E_i$ is small, that count carries little information, and a better estimate borrows from other counties. This section develops two ways to borrow. The first borrows from all counties equally, by shrinking toward the overall rate. The second borrows mostly from neighbors, through the spatial random effect.

### 9.3.1 Empirical Bayes Shrinkage: The Gamma-Poisson Model

**A tool we need: the gamma distribution.** A positive random variable $\theta$ has a **gamma distribution** with shape $a > 0$ and rate $b > 0$, written $\theta \sim \text{Gamma}(a, b)$, if its density is

$$
p(\theta) = \frac{b^a}{\Gamma(a)}\,\theta^{a-1}e^{-b\theta}, \qquad \theta > 0
$$

Its mean is $a/b$ and its variance is $a/b^2$. We take $a = b$, so the mean is 1 and the variance is $1/a$: the relative risks scatter around the overall rate, with a spread controlled by $a$. A large $a$ means the counties are nearly alike.

The **gamma-Poisson model** (Clayton and Kaldor, 1987 [@clayton1987]) says that each county's relative risk is a draw from this distribution, and its count is Poisson given the risk:

$$
\theta_i \sim \text{Gamma}(a, a), \qquad Y_i \mid \theta_i \sim \text{Poisson}(E_i\theta_i)
$$

**The posterior.** Multiply the prior density by the Poisson probability, keeping only the factors that involve $\theta_i$:

$$
p(\theta_i \mid y_i) \propto \theta_i^{a-1}e^{-a\theta_i}\cdot(E_i\theta_i)^{y_i}e^{-E_i\theta_i} \propto \theta_i^{a + y_i - 1}e^{-(a + E_i)\theta_i}
$$

This is the kernel of a gamma density with shape $a + y_i$ and rate $a + E_i$. So $\theta_i \mid y_i \sim \text{Gamma}(a + y_i,\ a + E_i)$, with mean and standard deviation

$$
\operatorname{E}[\theta_i \mid y_i] = \frac{a + y_i}{a + E_i}, \qquad \operatorname{SD}(\theta_i \mid y_i) = \frac{\sqrt{a + y_i}}{a + E_i}
$$

**The posterior mean is a weighted average.** Write $y_i = E_i\cdot\text{SMR}_i$ and split the fraction:

$$
\frac{a + y_i}{a + E_i} = \frac{E_i}{a + E_i}\cdot\text{SMR}_i + \frac{a}{a + E_i}\cdot 1 = w_i\,\text{SMR}_i + (1 - w_i)\cdot 1, \qquad w_i = \frac{E_i}{E_i + a}
$$

The estimate is a compromise between the county's own SMR and the overall rate, 1. The weight $w_i$ on the SMR rises toward 1 as the expected count grows and falls toward 0 as it shrinks, and the constant $a$ says how many expected deaths it takes for a county to trust its own data: at $E_i = a$ the weight is one half. This is **shrinkage**, and it is the Bayesian answer to Section 9.2.2.

**Estimating $a$.** In **empirical Bayes**, the prior parameter is estimated from the data. Integrating $\theta_i$ out of the model gives the marginal distribution of $Y_i$. With the gamma density above,

$$
P(Y_i = y) = \int_0^\infty\frac{(E_i\theta)^ye^{-E_i\theta}}{y!}\,\frac{a^a\theta^{a-1}e^{-a\theta}}{\Gamma(a)}\,d\theta = \frac{\Gamma(y + a)}{\Gamma(a)\,y!}\left(\frac{a}{a + E_i}\right)^a\left(\frac{E_i}{a + E_i}\right)^y
$$

(the integrand is again a gamma kernel, so the integral is a ratio of gamma functions). This is the **negative binomial** probability function, with mean $E_i$ and variance $E_i + E_i^2/a$, greater than the Poisson variance $E_i$: the gamma mixing produces overdispersion. The likelihood of $a$ is the product over counties of these probabilities, which we maximize numerically.

```python
def nb_negloglik(log_a, y, E):
    """Negative marginal log-likelihood of the gamma-Poisson model."""
    a = np.exp(log_a)
    return -np.sum(special.gammaln(y + a) - special.gammaln(a)
                   - special.gammaln(y + 1) + a * np.log(a / (a + E))
                   + y * np.log(E / (a + E)))


def eb_gamma_poisson(y, E):
    """Empirical Bayes shrinkage of the SMR toward 1."""
    fit = optimize.minimize_scalar(nb_negloglik, bounds=(-5, 8),
                                   args=(y, E), method="bounded")
    a = np.exp(fit.x)
    w = E / (E + a)
    return a, w, w * y / E + (1 - w), np.sqrt(a + y) / (a + E)


E3 = np.array([1.0, 10.0, 100.0])
y3c = np.array([3.0, 12.0, 105.0])
a3 = 10.0
w3 = E3 / (E3 + a3)
print("SMR:", y3c / E3)
print("weights:", np.round(w3, 4))
print("EB estimates:", np.round(w3 * y3c / E3 + (1 - w3), 4))
print("posterior sd:", np.round(np.sqrt(a3 + y3c) / (a3 + E3), 4))
```

**By hand.** Take three counties with expected counts $E = (1, 10, 100)$ and observed counts $y = (3, 12, 105)$, so the SMRs are $(3, 1.2, 1.05)$, and suppose $a = 10$. The weights are $w = E/(E + 10) = (0.0909, 0.5, 0.9091)$, and the estimates

$$
\frac{a + y_i}{a + E_i} = \frac{13}{11},\ \frac{22}{20},\ \frac{115}{110} = 1.1818,\ 1.1,\ 1.0455
$$

The county that expects one death has an SMR of 3, which shrinks to 1.18, because a count of 3 against an expectation of 1 is weak evidence. The county that expects 100 deaths barely moves. The posterior standard deviations, $\sqrt{13}/11 = 0.3278$ and so on, are $0.3278$, $0.2345$, and $0.0975$: the large county's estimate is also more certain.

Empirical Bayes shrinks every county toward the same value, 1. That is a limitation. If neighboring counties tend to resemble each other, then a better target for a small county is its neighbors' average, not the statewide rate. That is what a spatial random effect provides.

### 9.3.2 A CAR Distribution for the Random Effect

Replace the independent gamma draws with a Gaussian random effect on the log scale that has the spatial structure of Chapter 8. The **proper CAR** model has precision $\mathbf{Q} = \sigma^{-2}(\mathbf{D} - \rho\mathbf{A})$, where $\mathbf{A}$ is the adjacency matrix and $\mathbf{D}$ the diagonal matrix of neighbor counts. The **intrinsic CAR** (ICAR) model sets $\rho = 1$, giving the graph Laplacian $\mathbf{Q} = \sigma^{-2}(\mathbf{D} - \mathbf{A})$, whose quadratic form is a sum of squared differences between neighbors,

$$
\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \tfrac{1}{2}\sum_i\sum_jA_{ij}(u_i - u_j)^2
$$

and which is singular: $(\mathbf{D} - \mathbf{A})\mathbf{1} = \mathbf{0}$, with one zero eigenvalue for each connected component of the map. The conditional distributions of the ICAR model are the key to its behavior. By Chapter 8,

$$
\operatorname{E}[u_i \mid \mathbf{u}_{-i}] = \frac{1}{d_i}\sum_{j}A_{ij}u_j, \qquad \operatorname{Var}(u_i \mid \mathbf{u}_{-i}) = \frac{\sigma^2}{d_i}
$$

Given its neighbors, a county's effect is expected to equal their average, with a variance that shrinks as the number of neighbors $d_i$ grows. Combined with a Poisson likelihood, the fitted $u_i$ is a compromise between the county's own SMR and its neighbors' average, weighted by how much each is worth. That is borrowing strength from neighbors, and it is what the hand example of Section 9.4.3 shows in numbers.

Three variants differ in how they split the variance between a spatially structured part and an independent part. The next three subsections introduce them; Section 9.3.6 compares them.

### 9.3.3 The BYM Model

The **Besag-York-Mollié** (BYM) model [@besag1991] writes the random effect as the sum of two components:

$$
\mathbf{u} = \mathbf{u}^{\text{icar}} + \mathbf{v}, \qquad \mathbf{u}^{\text{icar}} \sim \text{ICAR with precision } \sigma_u^{-2}(\mathbf{D} - \mathbf{A}), \qquad \mathbf{v} \sim N(\mathbf{0}, \sigma_v^2\mathbf{I}_n)
$$

The structured part $\mathbf{u}^{\text{icar}}$ carries what neighboring counties share, and the unstructured part $\mathbf{v}$ carries what each county has on its own, which is the overdispersion the Poisson model cannot absorb. The two variances $\sigma_u^2$ and $\sigma_v^2$ are the hyperparameters.

The covariance of $\mathbf{u}$ is the sum of the two covariances, $\operatorname{Var}(\mathbf{u}) = \sigma_u^2\boldsymbol{\Sigma}^{\star} + \sigma_v^2\mathbf{I}_n$, where $\boldsymbol{\Sigma}^{\star}$ is the generalized inverse of $\mathbf{D} - \mathbf{A}$ with a sum-to-zero constraint on each connected component, defined in Section 9.3.4. Each county has one observation, and that observation contributes to $\mathbf{u}^{\text{icar}}_i + v_i$ only through the sum. The data can say how large $\mathbf{u}$ is and how it is spatially arranged, but splitting $\mathbf{u}$ into a structured and an unstructured part has to rely on the arrangement alone. The two components are therefore **weakly identified**, and Section 9.6.4 shows how weakly.

BYM has a second, subtler problem. The precision $\sigma_u^{-2}(\mathbf{D} - \mathbf{A})$ is a conditional precision: it describes the effect given its neighbors, not the variance of the effect itself. The marginal variance of the ICAR model depends on the map, as Section 9.3.4 shows, so $\sigma_u^2$ and $\sigma_v^2$ are on different scales, and the same numbers mean different things on different maps.

### 9.3.4 BYM2: Scaling So That the Parameters Mean Something

The **BYM2** model [@riebler2016] reparameterizes BYM with a total scale $\sigma$ and a **mixing parameter** $\rho \in [0, 1]$ that is the share of the variance that is spatially structured:

$$
\mathbf{u} = \sigma\left(\sqrt{\frac{\rho}{c}}\,\mathbf{u}^{\star} + \sqrt{1 - \rho}\,\mathbf{v}\right), \qquad \mathbf{u}^{\star} \sim \text{ICAR with precision } \mathbf{D} - \mathbf{A}, \quad \mathbf{v} \sim N(\mathbf{0}, \mathbf{I}_n)
$$

Here $\mathbf{u}^{\star}$ is the unscaled ICAR effect (precision $\mathbf{D} - \mathbf{A}$, with the sum-to-zero constraint), and the **scaling constant** $c$ is chosen so that the structured part has a marginal variance of about one. Then each county's effect has variance close to $\sigma^2[\rho + (1 - \rho)] = \sigma^2$, of which a fraction $\rho$ is spatial. Here $\rho$ is a mixing parameter, not the autoregressive parameter of Chapter 8, though both lie between 0 and 1.

**The marginal variances of the ICAR.** The ICAR precision is singular, so it has no ordinary inverse. Adding the sum-to-zero constraint, $\sum_iu^{\star}_i = 0$ within each connected component, selects a unique covariance matrix for the remaining directions. For a connected map with $n$ regions this is

$$
\boldsymbol{\Sigma}^{\star} = \left(\mathbf{D} - \mathbf{A} + \tfrac{1}{n}\mathbf{1}\mathbf{1}'\right)^{-1} - \tfrac{1}{n}\mathbf{1}\mathbf{1}'
$$

which equals the Moore-Penrose pseudo-inverse of $\mathbf{D} - \mathbf{A}$ (Section 9.3.7 proves it). For a map with several connected components, the same formula is applied to each component separately, which is the meaning of "one constraint per component". The diagonal of $\boldsymbol{\Sigma}^{\star}$ gives the marginal variance of each region.

Marginal variances differ from region to region. **A path of three regions** shows how. Regions 1, 2, and 3 lie in a row, so $\mathbf{D} - \mathbf{A} = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{bmatrix}$. Its inverse after the adjustment above is

$$
\boldsymbol{\Sigma}^{\star} = \frac{1}{9}\begin{bmatrix} 5 & -1 & -4 \\ -1 & 2 & -1 \\ -4 & -1 & 5 \end{bmatrix}
$$

(the rows sum to zero, as the constraint requires). The marginal variances are $5/9$, $2/9$, and $5/9$: the middle region, which has two neighbors to be tied to, is more tightly pinned than the two end regions.

**The scaling constant is the geometric mean.** Because the variances differ across regions, a single constant cannot make them all equal to one. Following Sørbye and Rue [@sorbye2014], $c$ is the **geometric mean** of the marginal variances:

$$
c = \exp\!\left\{\frac{1}{n}\sum_{i=1}^n\log\Sigma^{\star}_{ii}\right\}
$$

so that $\mathbf{u}^{\star}/\sqrt{c}$ has a geometric-mean marginal variance of exactly one. The geometric mean is used instead of the arithmetic mean because it is not dominated by a few regions with unusually large variances, and because it is the natural average on the log scale, where prior scales are usually reasoned about. For the path of three regions,

$$
c = \left(\tfrac{5}{9}\cdot\tfrac{2}{9}\cdot\tfrac{5}{9}\right)^{1/3} = \left(\tfrac{50}{729}\right)^{1/3} = 0.4093
$$

The code computes $\boldsymbol{\Sigma}^{\star}$ and $c$ for any map, one connected component at a time, and shows how $c$ grows with the size of the map.

```python
def scaling_constant(A):
    """Geometric mean of the marginal variances of the ICAR model.

    Uses the generalized inverse with a sum-to-zero constraint on each
    connected component; one-region components are ignored."""
    n_comp, label = connected_components(A, directed=False)
    marg = np.zeros(len(A))
    for k in range(n_comp):
        idx = np.where(label == k)[0]
        if len(idx) == 1:
            continue
        Ak = A[np.ix_(idx, idx)]
        Lk = np.diag(Ak.sum(axis=1)) - Ak
        Jk = np.ones((len(idx), len(idx))) / len(idx)
        marg[idx] = np.diag(np.linalg.inv(Lk + Jk) - Jk)
    pos = marg > 0
    return np.exp(np.mean(np.log(marg[pos]))), marg


A3 = np.array([[0.0, 1, 0], [1, 0, 1], [0, 1, 0]])
L3 = np.diag(A3.sum(axis=1)) - A3
J3 = np.ones((3, 3)) / 3
Sigma_star = np.linalg.inv(L3 + J3) - J3
print("9 * Sigma_star:\n", np.round(9 * Sigma_star, 4))
print("matches the pseudo-inverse:",
      np.allclose(Sigma_star, np.linalg.pinv(L3)))
c3, marg3 = scaling_constant(A3)
print("marginal variances * 9:", np.round(9 * marg3, 4))
print(f"scaling constant c = {c3:.4f}")


def rook_grid(m):
    """Binary adjacency matrix of an m-by-m grid of square regions."""
    A = np.zeros((m * m, m * m))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            if i + 1 < m:
                A[k, k + m] = A[k + m, k] = 1
            if j + 1 < m:
                A[k, k + 1] = A[k + 1, k] = 1
    return A


for m in (3, 6, 12, 24, 48):
    c_m, _ = scaling_constant(rook_grid(m))
    print(f"{m:2d} x {m:2d} grid ({m * m:4d} regions): c = {c_m:.3f}")
```

The output reproduces the matrix above, the marginal variances $5/9, 2/9, 5/9$ (times 9: 5, 2, 5), and $c = 0.4093$. Square grids of 9, 36, 144, 576, and 2,304 regions have $c$ of 0.410, 0.551, 0.677, 0.796, and 0.912: the typical marginal variance of the ICAR grows as the map grows. This is why the unscaled variance $\sigma_u^2$ of BYM has no fixed meaning. A prior or an estimate for $\sigma_u^2$ that suits a 36-county state is on a different scale from one for a 576-county state, while BYM2's $\sigma$ means the same thing on both.

### 9.3.5 The Leroux Model

The **Leroux** model [@leroux2000] mixes independence and ICAR smoothing in the precision matrix itself:

$$
\mathbf{Q}(\sigma^2, \rho) = \sigma^{-2}\left[\rho(\mathbf{D} - \mathbf{A}) + (1 - \rho)\mathbf{I}_n\right], \qquad 0 \le \rho < 1
$$

At $\rho = 0$ the effects are independent with variance $\sigma^2$; as $\rho \to 1$ the precision approaches the ICAR. For $\rho < 1$ the matrix is positive definite, with no constraint needed: $\mathbf{D} - \mathbf{A}$ is positive semi-definite (its quadratic form is a sum of squares) and $(1 - \rho)\mathbf{I}_n$ is positive definite, so their positive combination is positive definite. The proper CAR model of Chapter 8 needs a bound on $\rho$ from the eigenvalues of $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$; the Leroux model needs only $\rho < 1$.

The conditional distributions are

$$
\operatorname{E}[u_i \mid \mathbf{u}_{-i}] = \frac{\rho\sum_jA_{ij}u_j}{\rho d_i + 1 - \rho}, \qquad \operatorname{Var}(u_i \mid \mathbf{u}_{-i}) = \frac{\sigma^2}{\rho d_i + 1 - \rho}
$$

which follow from the precision matrix entries $Q_{ii} = \sigma^{-2}(\rho d_i + 1 - \rho)$ and $Q_{ij} = -\sigma^{-2}\rho A_{ij}$ and the formulas of Chapter 8. Compared with the ICAR, the pull toward neighbors is weaker: the neighbor sum is divided by $\rho d_i + 1 - \rho$ instead of $d_i$, so the conditional mean is a shrunken neighbor average, the more so as $\rho$ falls.

The Leroux model is attractive because it has a single sparse precision matrix. It shares the BYM scale problem: $\sigma^2$ is not the marginal variance, and the marginal variance changes with $\rho$.

### 9.3.6 Comparing BYM, BYM2, and Leroux

All three models are functions of the eigendecomposition of the graph Laplacian. Write $\mathbf{D} - \mathbf{A} = \mathbf{U}\operatorname{diag}(\xi_1, \dots, \xi_n)\mathbf{U}'$, with $\xi_1 = 0$ (for a connected map) belonging to the constant vector and $\xi_2, \dots, \xi_n > 0$. Then the covariance of $\mathbf{u}$ is $\mathbf{U}\operatorname{diag}(\varsigma_1, \dots, \varsigma_n)\mathbf{U}'$ with

| Model | Eigenvalue $\varsigma_k$ of $\operatorname{Var}(\mathbf{u})$, $k \ge 2$ | Eigenvalue $\varsigma_1$ (constant direction) |
| :--- | :--- | :--- |
| Leroux | $\sigma^2/(\rho\xi_k + 1 - \rho)$ | $\sigma^2/(1 - \rho)$ |
| BYM | $\sigma_u^2/\xi_k + \sigma_v^2$ | $\sigma_v^2$ |
| BYM2 | $\sigma^2\left[(\rho/c)/\xi_k + 1 - \rho\right]$ | $\sigma^2(1 - \rho)$ |

The covariance and the precision of any of them are $\mathbf{U}\operatorname{diag}(\varsigma_k)\mathbf{U}'$ and $\mathbf{U}\operatorname{diag}(1/\varsigma_k)\mathbf{U}'$, and the log-determinant is a sum of logarithms, all cheap for the 144 counties of this chapter's lab. (Large maps use the sparse forms of Chapter 8.) The function below builds all three and is used throughout the lab.

The table also shows that BYM and BYM2 are the same model written in two ways. Matching the eigenvalues,

$$
\sigma_u^2 = \frac{\sigma^2\rho}{c}, \qquad \sigma_v^2 = \sigma^2(1 - \rho)
$$

so any BYM fit is a BYM2 fit and the reverse; only the meaning of the parameters differs. The practical differences are therefore about interpretation and about priors, not about the fitted surface.

The code compares prior marginal variances on the 12 by 12 map: how much does the typical variance of $u_i$ move with $\rho$ when $\sigma^2$ is held at one?

```python
def make_prior(A, kind):
    """Return prior(par) -> (C, Q, logdet Q, hyperparameters).

    All three models are functions of the eigenvalues of D - A, so the
    covariance is C = U diag(lam) U' and the precision Q = U diag(1/lam) U'.
    kind 'leroux' and 'bym2': par = (log sigma2, logit rho).
    kind 'bym': par = (log sigma_u2, log sigma_v2)."""
    n = len(A)
    e, U = np.linalg.eigh(np.diag(A.sum(axis=1)) - A)
    c, _ = scaling_constant(A)

    def prior(par):
        lam = np.empty(n)
        if kind == "bym":
            su2, sv2 = np.exp(par)
            lam[1:] = su2 / e[1:] + sv2
            lam[0] = sv2
            hyp = (su2, sv2)
        else:
            s2 = np.exp(par[0])
            rho = 1 / (1 + np.exp(-np.clip(par[1], -7, 7)))
            if kind == "leroux":
                lam = s2 / (rho * e + 1 - rho)
            else:
                lam[1:] = s2 * ((rho / c) / e[1:] + 1 - rho)
                lam[0] = s2 * (1 - rho)
            hyp = (s2, rho)
        return (U * lam) @ U.T, (U / lam) @ U.T, -np.sum(np.log(lam)), hyp
    return prior


A12 = rook_grid(12)
print("prior marginal variances on the 12 x 12 map (mean, smallest, largest)")
print("rho    Leroux (sigma2 = 1)      BYM2 (sigma2 = 1)")
for rho in (0.1, 0.5, 0.9, 0.99):
    row = []
    for kind in ("leroux", "bym2"):
        par = np.array([0.0, np.log(rho / (1 - rho))])
        v = np.diag(make_prior(A12, kind)(par)[0])
        row.append(f"{v.mean():5.2f} {v.min():5.2f} {v.max():5.2f}")
    print(f"{rho:4.2f}   {row[0]}          {row[1]}")
v = np.diag(make_prior(A12, "bym")(np.zeros(2))[0])
print(f"BYM (both variances 1): {v.mean():5.2f} {v.min():5.2f} {v.max():5.2f}")
```

| $\rho$ | Leroux ($\sigma^2 = 1$): mean, min, max | BYM2 ($\sigma^2 = 1$): mean, min, max |
| :--- | :--- | :--- |
| 0.10 | 0.81, 0.79, 0.92 | 1.00, 0.97, 1.11 |
| 0.50 | 0.56, 0.51, 0.84 | 1.02, 0.83, 1.54 |
| 0.90 | 0.63, 0.51, 1.12 | 1.04, 0.70, 1.96 |
| 0.99 | 1.37, 1.14, 2.03 | 1.04, 0.66, 2.06 |

In the Leroux model, the mean marginal variance with $\sigma^2 = 1$ moves from 0.81 to 0.56 to 1.37 as $\rho$ rises, so $\sigma^2$ is not the variance of the effects and its meaning depends on $\rho$. In BYM2 the mean marginal variance stays at 1.00 to 1.04 for every $\rho$: $\sigma^2$ is the total variance and $\rho$ moves it between the two components without changing the total. BYM with both variances equal to one has a mean marginal variance of 1.71, and that figure would change with the map. A summary:

| | BYM | BYM2 | Leroux |
| :--- | :--- | :--- | :--- |
| Parameters | $\sigma_u^2$, $\sigma_v^2$ | $\sigma^2$, $\rho$ | $\sigma^2$, $\rho$ |
| What $\rho$ means | none | share of variance that is spatial | weight on the Laplacian in the precision |
| Is $\sigma^2$ the marginal variance? | no ($\sigma_u^2$ depends on the map) | yes, approximately | no (depends on $\rho$) |
| Sparse precision | as a pair of components | as a pair of components | directly |
| Scaling needed | yes, to compare maps | built in | not built in |

### 9.3.7 The Constrained Generalized Inverse (Optional)

*This subsection proves the formula for $\boldsymbol{\Sigma}^{\star}$ used in Section 9.3.4. It can be skipped on a first reading.*

Let $\mathcal{L} = \mathbf{D} - \mathbf{A}$ for a connected map, with eigenvalues $0 = \xi_1 < \xi_2 \le \dots \le \xi_n$ and orthonormal eigenvectors $\mathbf{v}_1 = \mathbf{1}/\sqrt{n}, \mathbf{v}_2, \dots, \mathbf{v}_n$. (The zero eigenvalue is simple because the map is connected: $\mathbf{a}'\mathcal{L}\mathbf{a} = \tfrac{1}{2}\sum_{ij}A_{ij}(a_i - a_j)^2 = 0$ forces $a_i = a_j$ across every adjacent pair, hence for the whole map.)

The ICAR density, restricted to the subspace $\sum_iu^{\star}_i = 0$, which is spanned by $\mathbf{v}_2, \dots, \mathbf{v}_n$, is a proper Gaussian density with precision $\mathcal{L}$ acting on that subspace. Its covariance acts on the same subspace and equals $\sum_{k\ge2}\xi_k^{-1}\mathbf{v}_k\mathbf{v}_k'$, which is the Moore-Penrose pseudo-inverse $\mathcal{L}^{+}$.

Now compute the matrix in the formula. Since $\tfrac{1}{n}\mathbf{1}\mathbf{1}' = \mathbf{v}_1\mathbf{v}_1'$ and the eigenvectors are orthonormal,

$$
\mathcal{L} + \tfrac{1}{n}\mathbf{1}\mathbf{1}' = \sum_{k\ge2}\xi_k\mathbf{v}_k\mathbf{v}_k' + 1\cdot\mathbf{v}_1\mathbf{v}_1'
$$

has eigenvalues $1, \xi_2, \dots, \xi_n$, all positive, so it is invertible, with inverse $\mathbf{v}_1\mathbf{v}_1' + \sum_{k\ge2}\xi_k^{-1}\mathbf{v}_k\mathbf{v}_k'$. Subtracting $\tfrac{1}{n}\mathbf{1}\mathbf{1}' = \mathbf{v}_1\mathbf{v}_1'$ leaves $\sum_{k\ge2}\xi_k^{-1}\mathbf{v}_k\mathbf{v}_k' = \mathcal{L}^{+}$. $\square$

The same eigenvector expansion explains how to simulate the ICAR: draw $\mathbf{u}^{\star} = \sum_{k\ge2}\xi_k^{-1/2}z_k\mathbf{v}_k$ with independent standard normal $z_k$. By Tool 1 its covariance is $\sum_{k\ge2}\xi_k^{-1}\mathbf{v}_k\mathbf{v}_k' = \mathcal{L}^{+}$, and it satisfies the constraint because each $\mathbf{v}_k$ with $k \ge 2$ is orthogonal to $\mathbf{1}$. The lab uses this to generate the true risk surface.

---

## 9.4 Fitting Spatial GLMMs by Penalized Quasi-Likelihood

For known hyperparameters (the variance and mixing parameters, which fix $\mathbf{Q}$), the model has two kinds of unknowns: the regression coefficients $\boldsymbol{\beta}$ and the random effects $\mathbf{u}$. This section shows how to estimate both with a small extension of IRLS, and how to estimate the hyperparameters with the REML of Chapter 5. The method is called **penalized quasi-likelihood** (PQL; Breslow and Clayton, 1993 [@breslow1993]). We present it as a teaching device: it is simple, fast, and shows why the model behaves as it does, and Section 9.5 supplies a more careful likelihood.

### 9.4.1 The Penalized Log-Likelihood

Collect the log offset into $\mathbf{o} = \log\mathbf{E}$ and write $\boldsymbol{\eta} = \mathbf{o} + \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$ and $\boldsymbol{\mu} = e^{\boldsymbol{\eta}}$ (entrywise). The joint log-density of the counts and the random effects, as a function of $(\boldsymbol{\beta}, \mathbf{u})$ with the hyperparameters fixed, is

$$
\ell_{\text{pen}}(\boldsymbol{\beta}, \mathbf{u}) = \sum_{i=1}^n\left(y_i\eta_i - e^{\eta_i} - \log y_i!\right) - \tfrac{1}{2}\mathbf{u}'\mathbf{Q}\mathbf{u} + \text{constant}
$$

The first term is the Poisson log-likelihood of Section 9.1. The second is the log of the Gaussian density of $\mathbf{u}$, and it acts as a **penalty**: it subtracts more from the log-likelihood the rougher $\mathbf{u}$ is. For the ICAR, $\tfrac{1}{2}\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \tfrac{1}{4}\sum_{ij}A_{ij}(u_i - u_j)^2$ penalizes differences between neighbors. The constant (the log-determinant of $\mathbf{Q}$) does not depend on $(\boldsymbol{\beta}, \mathbf{u})$ and matters only for the hyperparameters, in Section 9.5.

The first derivatives are, from Section 9.1.3 and the derivative of a quadratic form,

$$
\frac{\partial\ell_{\text{pen}}}{\partial\boldsymbol{\beta}} = \mathbf{X}'(\mathbf{y} - \boldsymbol{\mu}), \qquad \frac{\partial\ell_{\text{pen}}}{\partial\mathbf{u}} = (\mathbf{y} - \boldsymbol{\mu}) - \mathbf{Q}\mathbf{u}
$$

and the second derivatives, with $\mathbf{V} = \operatorname{diag}(\boldsymbol{\mu})$ as in Section 9.1.4, are

$$
-\frac{\partial^2\ell_{\text{pen}}}{\partial(\boldsymbol{\beta}, \mathbf{u})\,\partial(\boldsymbol{\beta}, \mathbf{u})'} = \begin{bmatrix} \mathbf{X}'\mathbf{V}\mathbf{X} & \mathbf{X}'\mathbf{V} \\ \mathbf{V}\mathbf{X} & \mathbf{V} + \mathbf{Q} \end{bmatrix}
$$

(The Poisson part contributes $\mathbf{V}$ through $\boldsymbol{\eta} = \ldots + \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$, and the penalty adds $\mathbf{Q}$ to the $\mathbf{u}$ block.) When $\mathbf{Q}$ is positive definite and $\mathbf{X}$ has full column rank, this matrix is positive definite. The $\mathbf{u}$ block $\mathbf{V} + \mathbf{Q}$ is, and the Schur complement for the $\boldsymbol{\beta}$ block is $\mathbf{X}'\left[\mathbf{V} - \mathbf{V}(\mathbf{V} + \mathbf{Q})^{-1}\mathbf{V}\right]\mathbf{X} = \mathbf{X}'(\mathbf{V}^{-1} + \mathbf{Q}^{-1})^{-1}\mathbf{X}$ (by the Woodbury identity, which follows from the block inversion formula of Appendix A.1 and returns in Chapter 13), which is also positive definite. The same combination $\mathbf{V}^{-1} + \mathbf{Q}^{-1}$ returns in Section 9.4.4.

### 9.4.2 Henderson's Equations

Newton's method for maximizing $\ell_{\text{pen}}$ adds the inverse of this matrix times the gradient to the current $(\boldsymbol{\beta}, \mathbf{u})$. Write $\mathbf{N}$ for the matrix above. Multiplying the update by $\mathbf{N}$ gives

$$
\mathbf{N}\begin{bmatrix}\boldsymbol{\beta}^{\text{new}} \\ \mathbf{u}^{\text{new}}\end{bmatrix} = \mathbf{N}\begin{bmatrix}\boldsymbol{\beta} \\ \mathbf{u}\end{bmatrix} + \begin{bmatrix}\mathbf{X}'(\mathbf{y} - \boldsymbol{\mu}) \\ (\mathbf{y} - \boldsymbol{\mu}) - \mathbf{Q}\mathbf{u}\end{bmatrix}
$$

The product on the right is $\mathbf{N}(\boldsymbol{\beta}, \mathbf{u})' = \left(\mathbf{X}'\mathbf{V}(\mathbf{X}\boldsymbol{\beta} + \mathbf{u}),\ \mathbf{V}(\mathbf{X}\boldsymbol{\beta} + \mathbf{u}) + \mathbf{Q}\mathbf{u}\right)'$. Adding the gradient, the $\mathbf{Q}\mathbf{u}$ terms cancel and we can factor out $\mathbf{V}$:

$$
\mathbf{X}'\mathbf{V}(\mathbf{X}\boldsymbol{\beta} + \mathbf{u}) + \mathbf{X}'(\mathbf{y} - \boldsymbol{\mu}) = \mathbf{X}'\mathbf{V}\mathbf{z}, \qquad \mathbf{V}(\mathbf{X}\boldsymbol{\beta} + \mathbf{u}) + (\mathbf{y} - \boldsymbol{\mu}) = \mathbf{V}\mathbf{z}
$$

where $\mathbf{z} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u} + \mathbf{V}^{-1}(\mathbf{y} - \boldsymbol{\mu})$ is the working response of Section 9.1.5 **with the offset removed**: it equals $\boldsymbol{\eta} - \mathbf{o} + \mathbf{V}^{-1}(\mathbf{y} - \boldsymbol{\mu})$, the linear predictor without its known part. The Newton step is therefore the solution of **Henderson's mixed-model equations**:

$$
\begin{bmatrix} \mathbf{X}'\mathbf{V}\mathbf{X} & \mathbf{X}'\mathbf{V} \\ \mathbf{V}\mathbf{X} & \mathbf{V} + \mathbf{Q} \end{bmatrix}\begin{bmatrix} \boldsymbol{\beta}^{\text{new}} \\ \mathbf{u}^{\text{new}} \end{bmatrix} = \begin{bmatrix} \mathbf{X}'\mathbf{V}\mathbf{z} \\ \mathbf{V}\mathbf{z} \end{bmatrix}
$$

The same equations arise from a weighted least squares view. They are the normal equations for minimizing

$$
(\mathbf{z} - \mathbf{X}\boldsymbol{\beta} - \mathbf{u})'\mathbf{V}(\mathbf{z} - \mathbf{X}\boldsymbol{\beta} - \mathbf{u}) + \mathbf{u}'\mathbf{Q}\mathbf{u}
$$

with $\mathbf{z}$ and $\mathbf{V}$ held fixed. Setting the derivative with respect to $\boldsymbol{\beta}$ to zero gives $\mathbf{X}'\mathbf{V}(\mathbf{z} - \mathbf{X}\boldsymbol{\beta} - \mathbf{u}) = \mathbf{0}$, the first block row, and setting the derivative with respect to $\mathbf{u}$ to zero gives $\mathbf{V}(\mathbf{z} - \mathbf{X}\boldsymbol{\beta} - \mathbf{u}) - \mathbf{Q}\mathbf{u} = \mathbf{0}$, the second. So **penalized IRLS is IRLS with a ridge-type penalty on $\mathbf{u}$**: each iteration is a weighted regression of $\mathbf{z}$ on the covariates and on an identity design for $\mathbf{u}$, in which $\mathbf{u}$ is penalized by $\mathbf{Q}$.

Two properties are worth recording.

- **Identifiability.** If $\mathbf{Q}$ is singular, as for the ICAR, and $\mathbf{X}$ contains an intercept, the system is singular, because adding a constant to $\mathbf{u}$ and subtracting it from the intercept changes nothing. The sum-to-zero constraint of Section 9.3.4 removes this. Leroux with $\rho < 1$, BYM, and BYM2 (whose $\mathbf{u}$ includes the unstructured part) have a positive definite $\mathbf{Q}$, so this chapter's code needs no constraint.
- **A link back to GLS.** Eliminating $\mathbf{u}$ from the equations shows that $\hat{\boldsymbol{\beta}}$ is the generalized least squares estimate for the working model $\mathbf{z} \sim N(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}^{-1} + \mathbf{Q}^{-1})$, and that $\hat{\mathbf{u}}$ is the kriging-style predictor of Chapter 4 (Exercise 8). So the PQL standard error of $\hat{\boldsymbol{\beta}}$ is $\left(\mathbf{X}'(\mathbf{V}^{-1} + \mathbf{Q}^{-1})^{-1}\mathbf{X}\right)^{-1}$, the GLS formula of Section 1.3.5.

### 9.4.3 By Hand: One Penalized IRLS Step on Three Regions

Take three regions in a row, with expected counts $E = (1, 10, 100)$ and observed counts $y = (3, 12, 105)$ (the counties of Section 9.3.1), and the Leroux precision with $\sigma^2 = 1$ and $\rho = 0.5$. The path of three regions has $\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$ and $\mathbf{D} = \operatorname{diag}(1, 2, 1)$, so

$$
\mathbf{Q} = 0.5(\mathbf{D} - \mathbf{A}) + 0.5\,\mathbf{I}_3 = \begin{bmatrix} 1 & -0.5 & 0 \\ -0.5 & 1.5 & -0.5 \\ 0 & -0.5 & 1 \end{bmatrix}
$$

There is only an intercept in $\mathbf{X}$, which is a column of ones. Start at $\beta = \log(\sum y/\sum E) = \log(120/111) = 0.078$ and $\mathbf{u} = \mathbf{0}$. Then $\boldsymbol{\mu} = E\,e^{0.078} = (1.081, 10.811, 108.108)$, which is also $\operatorname{diag}(\mathbf{V})$, and the working response is $z_i = 0.078 + (y_i - \mu_i)/\mu_i$:

$$
\mathbf{z} = (0.078 + 1.775,\ 0.078 + 0.110,\ 0.078 - 0.029)' = (1.853, 0.188, 0.0492)'
$$

The first county's count of 3 is almost three times its mean of 1.081, so its working response is far above the intercept. Henderson's equations have a $4 \times 4$ left-hand side. Its top-left entry is $\mathbf{1}'\mathbf{V}\mathbf{1} = 120$; the rest of the first row is $\mathbf{1}'\mathbf{V} = (1.081, 10.811, 108.108)$; and the lower-right block is $\mathbf{V} + \mathbf{Q}$, which adds the diagonal $(1.081, 10.811, 108.108)$ to $\mathbf{Q}$:

$$
\begin{bmatrix} 120 & 1.081 & 10.811 & 108.108 \\ 1.081 & 2.081 & -0.5 & 0 \\ 10.811 & -0.5 & 12.311 & -0.5 \\ 108.108 & 0 & -0.5 & 109.108 \end{bmatrix}\begin{bmatrix}\beta \\ u_1 \\ u_2 \\ u_3\end{bmatrix} = \begin{bmatrix} 9.3554 \\ 2.0032 \\ 2.0320 \\ 5.3202 \end{bmatrix}
$$

The right-hand side is $(\mathbf{1}'\mathbf{V}\mathbf{z},\ \mathbf{V}\mathbf{z})$; for example, $108.108 \times 0.04921 = 5.320$. The solution is $\beta = 0.472$, $\mathbf{u} = (0.660, -0.240, -0.420)'$, and you can check any row by multiplying it out. The fitted relative risks $e^{\beta + u_i}$ after one step are $(3.101, 1.262, 1.053)$.

The first step overshoots for the first county (3.10, against an SMR of 3), because the linearization at $\mathbf{u} = \mathbf{0}$ is poor when the count is nearly three times the mean. Further iterations fix that. The code below does the first step by hand-built matrices, confirms that the general penalized IRLS function agrees, and runs to convergence.

```python
def pirls_mixed(y, X, offset, Q, beta=None, u=None, iters=100, tol=1e-9):
    """Penalized IRLS: solve Henderson's equations until convergence.

    Returns beta, u, mu, the working response z, and iterations used."""
    n, p = X.shape
    if beta is None:
        beta = np.zeros(p)
        beta[0] = np.log(y.sum() / np.exp(offset).sum())
    u = np.zeros(n) if u is None else u
    for it in range(1, iters + 1):
        mu = np.exp(offset + X @ beta + u)
        z = X @ beta + u + (y - mu) / mu
        XtV = X.T * mu
        lhs = np.block([[XtV @ X, XtV], [XtV.T, np.diag(mu) + Q]])
        sol = np.linalg.solve(lhs, np.concatenate([XtV @ z, mu * z]))
        step = np.max(np.abs(sol - np.concatenate([beta, u])))
        beta, u = sol[:p], sol[p:]
        if step < tol:
            break
    mu = np.exp(offset + X @ beta + u)
    return beta, u, mu, X @ beta + u + (y - mu) / mu, it
```

```python
A_path = np.array([[0.0, 1, 0], [1, 0, 1], [0, 1, 0]])
D_path = np.diag(A_path.sum(axis=1))
rho_h = 0.5
Q_h = rho_h * (D_path - A_path) + (1 - rho_h) * np.eye(3)
X_h = np.ones((3, 1))
E_h, y_h = np.array([1.0, 10.0, 100.0]), np.array([3.0, 12.0, 105.0])
b0 = np.log(y_h.sum() / E_h.sum())
mu_h = E_h * np.exp(b0)
z_h = b0 + (y_h - mu_h) / mu_h
XtV = X_h.T * mu_h
lhs_h = np.block([[XtV @ X_h, XtV], [XtV.T, np.diag(mu_h) + Q_h]])
rhs_h = np.concatenate([XtV @ z_h, mu_h * z_h])
sol_h = np.linalg.solve(lhs_h, rhs_h)
print("starting beta:", round(b0, 4), " mu:", np.round(mu_h, 3))
print("working response z:", np.round(z_h, 4))
print("Q:\n", Q_h)
print("left-hand side:\n", np.round(lhs_h, 4))
print("right-hand side:", np.round(rhs_h, 4))
print("solution (beta, u1, u2, u3):", np.round(sol_h, 4))
print("fitted relative risks:", np.round(np.exp(sol_h[0] + sol_h[1:]), 4))

one_step = pirls_mixed(y_h, X_h, np.log(E_h), Q_h, iters=1)
print("one step with pirls_mixed agrees:",
      np.allclose(np.concatenate([one_step[0], one_step[1]]), sol_h))
b_c, u_c, mu_c, _, n_it = pirls_mixed(y_h, X_h, np.log(E_h), Q_h)
print(f"converged after {n_it} iterations: beta={np.round(b_c, 4)}, "
      f"u={np.round(u_c, 4)}")
print("SMR:", np.round(y_h / E_h, 4),
      " fitted relative risks:", np.round(mu_c / E_h, 4))
```

After six iterations the fitted relative risks are $(2.414, 1.234, 1.052)$, against SMRs of $(3, 1.2, 1.05)$. County 1, which expects one death and had three, is pulled down from 3.00 to 2.41, because its two-county neighborhood is not as high: its neighbor has a ratio of about 1.2. County 3 has an expected count of 100 and does not move. County 2 rises slightly from 1.2 to 1.234, pulled up by its high neighbor. The weights decide how much each county trusts its own count, which in Henderson's equations is the size of $\mathbf{V}$ against $\mathbf{Q}$.

### 9.4.4 Estimating the Hyperparameters by REML on the Working Vector

Penalized IRLS needs $\mathbf{Q}$, which depends on the hyperparameters. PQL estimates them from the **working model** that the final iteration implies. The working response is the linear predictor plus the correction $(\mathbf{y} - \boldsymbol{\mu})/\boldsymbol{\mu}$, whose entries have variance $\mu_i/\mu_i^2 = 1/\mu_i$ and are approximately independent. So

$$
\mathbf{z} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u} + \mathbf{e}, \qquad \mathbf{u} \sim N(\mathbf{0}, \mathbf{Q}^{-1}), \qquad \mathbf{e} \sim N(\mathbf{0}, \mathbf{V}^{-1}), \qquad \text{so}\qquad \mathbf{z} \sim N(\mathbf{X}\boldsymbol{\beta},\ \boldsymbol{\Sigma}_z)
$$

with $\boldsymbol{\Sigma}_z = \mathbf{V}^{-1} + \mathbf{Q}^{-1}$. This is a Gaussian linear model with a known part $\mathbf{V}^{-1}$ and an unknown part $\mathbf{Q}^{-1}$ that depends on the hyperparameters, which is exactly the setting of Chapter 5. The REML log-likelihood of Section 5.3.3, with $\boldsymbol{\Sigma}_z$ in place of $\boldsymbol{\Sigma}$ and $\mathbf{z}$ in place of $\mathbf{y}$, is

$$
\ell_{\text{REML}} = -\tfrac{1}{2}\log\lvert\boldsymbol{\Sigma}_z\rvert - \tfrac{1}{2}\log\lvert\mathbf{X}'\boldsymbol{\Sigma}_z^{-1}\mathbf{X}\rvert - \tfrac{1}{2}\mathbf{z}'\boldsymbol{\Pi}_z\mathbf{z} + \text{constant}
$$

where $\boldsymbol{\Pi}_z = \boldsymbol{\Sigma}_z^{-1} - \boldsymbol{\Sigma}_z^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}_z^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}_z^{-1}$. PQL alternates two steps until the hyperparameters stop changing: run penalized IRLS with the current hyperparameters, then maximize $\ell_{\text{REML}}$ over the hyperparameters with $\mathbf{z}$ and $\mathbf{V}$ held fixed. The code below does this, using Cholesky factors as in Chapter 5 and the Nelder-Mead optimizer for the one- or two-dimensional hyperparameter search.

```python
def reml_negloglik(par, prior, z, mu, X):
    """Negative REML log-likelihood of z ~ N(X beta, V^-1 + C), up to a
    constant, as in Chapter 5 with Sigma_z = V^-1 + C."""
    C = prior(par)[0]
    Sz = C + np.diag(1 / mu)
    L = np.linalg.cholesky(Sz)
    XtSiX = X.T @ np.linalg.solve(Sz, X)
    beta = np.linalg.solve(XtSiX, X.T @ np.linalg.solve(Sz, z))
    r = z - X @ beta
    return 0.5 * (2 * np.sum(np.log(np.diag(L)))
                  + np.linalg.slogdet(XtSiX)[1]
                  + r @ np.linalg.solve(Sz, r))


def fit_pql(y, X, offset, prior, start, outer=40, tol=1e-5):
    """PQL: alternate penalized IRLS with REML on the working vector."""
    par = np.array(start, float)
    beta = u = None
    for k in range(1, outer + 1):
        Q = prior(par)[1]
        beta, u, mu, z, _ = pirls_mixed(y, X, offset, Q, beta, u)
        res = optimize.minimize(reml_negloglik, par, method="Nelder-Mead",
                                args=(prior, z, mu, X),
                                options={"xatol": 1e-4, "fatol": 1e-9})
        change = np.max(np.abs(res.x - par))
        par = res.x
        if change < tol:
            break
    C, Q, _, hyp = prior(par)
    beta, u, mu, z, _ = pirls_mixed(y, X, offset, Q, beta, u)
    cov_beta = np.linalg.inv(X.T @ np.linalg.solve(C + np.diag(1 / mu), X))
    return dict(beta=beta, u=u, mu=mu, hyper=hyp, par=par, cov=cov_beta,
                outer=k)
```

### 9.4.5 Where PQL Is Weak

The working model replaces the Poisson noise by Gaussian noise with variance $1/\mu_i$. That is accurate when counts are large and crude when they are small: for a count of 0 the working response $\eta - 1$ is not close to a normal variate. Breslow and Clayton noted that PQL tends to underestimate the variance of the random effects when counts are small. The lab counties have a median expected count of about five, so we use the Laplace likelihood of the next section as a check.

---

## 9.5 The Laplace Approximation of the Marginal Likelihood

The hyperparameters ought to be estimated by maximizing the **marginal likelihood**, which averages the Poisson likelihood over the random effects:

$$
L(\boldsymbol{\beta}, \text{hyperparameters}) = \int p(\mathbf{y} \mid \mathbf{u}, \boldsymbol{\beta})\,p(\mathbf{u})\,d\mathbf{u}
$$

This is an $n$-dimensional integral, with no closed form. The **Laplace approximation** replaces the integrand by a Gaussian with the same mode and curvature, whose integral is known.

### 9.5.1 The Derivation

**A tool we need: the Gaussian integral.** For a positive definite $n \times n$ matrix $\mathbf{M}$,

$$
\int\exp\!\left\{-\tfrac{1}{2}(\mathbf{u} - \mathbf{m})'\mathbf{M}(\mathbf{u} - \mathbf{m})\right\}d\mathbf{u} = (2\pi)^{n/2}\lvert\mathbf{M}\rvert^{-1/2}
$$

This is the normalizing constant of the multivariate normal density with covariance $\mathbf{M}^{-1}$, which we met in Chapter 0 (Tool 3).

Write the integrand as $\exp\{h(\mathbf{u})\}$, where

$$
h(\mathbf{u}) = \sum_i\left(y_i\eta_i - e^{\eta_i} - \log y_i!\right) - \tfrac{1}{2}\mathbf{u}'\mathbf{Q}\mathbf{u} + \tfrac{1}{2}\log\lvert\mathbf{Q}\rvert - \tfrac{n}{2}\log 2\pi
$$

with $\boldsymbol{\eta} = \mathbf{o} + \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$. Fix $\boldsymbol{\beta}$ and the hyperparameters. Let $\hat{\mathbf{u}}$ be the mode of $h$, the solution of $(\mathbf{y} - \boldsymbol{\mu}) - \mathbf{Q}\mathbf{u} = \mathbf{0}$, found by Newton's method on $\mathbf{u}$ alone with the step $(\mathbf{V} + \mathbf{Q})^{-1}\left[(\mathbf{y} - \boldsymbol{\mu}) - \mathbf{Q}\mathbf{u}\right]$, which is the $\mathbf{u}$ block of Section 9.4.1. At the mode the gradient vanishes, and the Hessian of $h$ is $-(\hat{\mathbf{V}} + \mathbf{Q})$ with $\hat{\mathbf{V}} = \operatorname{diag}(\hat{\boldsymbol{\mu}})$. Expanding $h$ to second order around $\hat{\mathbf{u}}$,

$$
h(\mathbf{u}) \approx h(\hat{\mathbf{u}}) - \tfrac{1}{2}(\mathbf{u} - \hat{\mathbf{u}})'(\hat{\mathbf{V}} + \mathbf{Q})(\mathbf{u} - \hat{\mathbf{u}})
$$

Exponentiating and integrating with the Gaussian integral gives $L \approx \exp\{h(\hat{\mathbf{u}})\}(2\pi)^{n/2}\lvert\hat{\mathbf{V}} + \mathbf{Q}\rvert^{-1/2}$. Take the logarithm; the constant $\tfrac{n}{2}\log 2\pi$ cancels:

$$
\log L \approx \sum_i\left(y_i\hat{\eta}_i - \hat{\mu}_i - \log y_i!\right) - \tfrac{1}{2}\hat{\mathbf{u}}'\mathbf{Q}\hat{\mathbf{u}} + \tfrac{1}{2}\log\lvert\mathbf{Q}\rvert - \tfrac{1}{2}\log\lvert\hat{\mathbf{V}} + \mathbf{Q}\rvert
$$

The last two terms combine: $\lvert\hat{\mathbf{V}} + \mathbf{Q}\rvert = \lvert\mathbf{Q}\rvert\,\lvert\mathbf{I} + \mathbf{Q}^{-1}\hat{\mathbf{V}}\rvert$, so $\tfrac{1}{2}\log\lvert\mathbf{Q}\rvert - \tfrac{1}{2}\log\lvert\hat{\mathbf{V}} + \mathbf{Q}\rvert = -\tfrac{1}{2}\log\lvert\mathbf{I} + \mathbf{Q}^{-1}\hat{\mathbf{V}}\rvert$. This term grows with the weights: it is the price the likelihood pays for the data pinning down the random effects. If the response were Gaussian, $h$ would be exactly quadratic and the approximation would be exact, which is how the marginal likelihood of Chapter 5 was computed.

The approximate log-likelihood depends on $\boldsymbol{\beta}$ and the hyperparameters, both directly and through $\hat{\mathbf{u}}$. The functions below find $\hat{\mathbf{u}}$ by Newton's method and evaluate the approximation for given parameters; the lab maximizes it over all of them with a general-purpose optimizer. This differs from PQL, which holds $\mathbf{V}$ and $\mathbf{z}$ fixed in the hyperparameter step.

```python
def inner_u(y, Xb, offset, Q, u0, iters=50, tol=1e-10):
    """Mode of the random effects for fixed beta and hyperparameters."""
    u = u0.copy()
    for _ in range(iters):
        mu = np.exp(offset + Xb + u)
        step = np.linalg.solve(np.diag(mu) + Q, (y - mu) - Q @ u)
        u = u + step
        if np.max(np.abs(step)) < tol:
            break
    return u


def laplace_negloglik(theta, y, X, offset, prior):
    """Negative Laplace approximation of the marginal log-likelihood.

    theta = (beta, hyperparameter vector for `prior`)."""
    p = X.shape[1]
    Q, logdet_Q = prior(theta[p:])[1:3]
    Xb = X @ theta[:p]
    u = inner_u(y, Xb, offset, Q, np.zeros(len(y)))
    eta = offset + Xb + u
    mu = np.exp(eta)
    loglik = np.sum(y * eta - mu - special.gammaln(y + 1))
    return -(loglik - 0.5 * u @ Q @ u + 0.5 * logdet_Q
             - 0.5 * np.linalg.slogdet(np.diag(mu) + Q)[1])
```

### 9.5.2 A One-Region Check

How good is the approximation? Take a single region with one random effect, $Y \mid u \sim \text{Poisson}(e^u)$ and $u \sim N(0, \sigma^2)$, so that the integral is one-dimensional and can be computed to high accuracy by numerical integration. The table gives the relative error of the Laplace approximation for several counts and several variances.

```python
def laplace_check(y_i, s2, E_i=1.0, b=0.0):
    """One region, one random effect: Laplace versus numerical integral."""
    def integrand(u):
        eta = np.log(E_i) + b + u
        return (np.exp(y_i * eta - np.exp(eta) - special.gammaln(y_i + 1))
                * np.exp(-u**2 / (2 * s2)) / np.sqrt(2 * np.pi * s2))
    exact = integrate.quad(integrand, -15, 15, epsabs=1e-16,
                           epsrel=1e-12, limit=200)[0]
    u = 0.0
    for _ in range(100):
        mu = E_i * np.exp(b + u)
        step = ((y_i - mu) - u / s2) / (mu + 1 / s2)
        u += step
        if abs(step) < 1e-13:
            break
    mu = E_i * np.exp(b + u)
    eta = np.log(E_i) + b + u
    log_l = (y_i * eta - mu - special.gammaln(y_i + 1) - u**2 / (2 * s2)
             - 0.5 * np.log(s2) - 0.5 * np.log(mu + 1 / s2))
    return exact, np.exp(log_l)


print("relative error of the Laplace approximation (Laplace / exact - 1)")
print("  y     s2=0.5     s2=2      s2=8")
for y_i in (0, 1, 2, 5, 20, 100):
    errs = []
    for s2 in (0.5, 2.0, 8.0):
        exact, lap = laplace_check(y_i, s2)
        errs.append(f"{100 * (lap / exact - 1):7.2f}%")
    print(f"{y_i:4d}  " + "  ".join(errs))
```

| $y$ | $\sigma^2 = 0.5$ | $\sigma^2 = 2$ | $\sigma^2 = 8$ |
| :--- | :--- | :--- | :--- |
| 0 | 0.65% | 1.26% | 0.23% |
| 1 | 0.59% | -0.46% | -4.04% |
| 2 | 0.46% | -1.12% | -3.04% |
| 5 | 0.03% | -1.11% | -1.52% |
| 20 | -0.30% | -0.40% | -0.41% |
| 100 | -0.08% | -0.08% | -0.08% |

The error is small whenever the random effect is modest: for $\sigma^2 = 0.5$ it stays under 0.7% even at a count of zero. It is largest, about 4%, when a large variance ($\sigma^2 = 8$) meets a small count, where the posterior of $u$ is skewed and a Gaussian is a poor stand-in, and it shrinks as the count grows, to under 0.1% at $y = 100$. The county data of the lab have counts around five and spatial variance around 0.15, which is in the comfortable corner of the table.

### 9.5.3 Using It

The Laplace likelihood supports the same inference as any likelihood: maximum likelihood estimates, comparison of models by their maximized values, and likelihood ratio tests. To test whether a spatial random effect is needed at all, compare it with the plain Poisson GLM, which is the case $\sigma^2 = 0$. As in Section 5.5.2, the null value lies on the boundary of the parameter space, so the reference distribution is the mixture $\tfrac{1}{2}\chi^2_0 + \tfrac{1}{2}\chi^2_1$. (In BYM2 the mixing parameter $\rho$ disappears under the null, which makes even that reference approximate. The lab's statistic is so large that the conclusion does not depend on it.)

---

## 9.6 Walkthrough and Lab 9

### 9.6.1 The County Data and the Raw SMRs

The lab uses the 144-county map shared by Part III: a $12 \times 12$ grid of square counties with rook adjacency, lognormal populations (the same median, about 5,000, as in Section 0.3.3, but a smaller spread: the standard deviation of the log population is 1.0 here and 1.2 there), and a smooth standardized deprivation index $x$. The responses are simulated from a BYM2 model so that we know the truth:

$$
Y_i \sim \text{Poisson}\!\left(0.001\cdot\text{pop}_i\cdot\exp(0.3\,x_i + u_i)\right), \qquad \mathbf{u} \text{ BYM2 with } \sigma^2 = 0.15,\ \rho = 0.6
$$

The covariate has coefficient 0.3, so each standard deviation of deprivation raises the risk by a factor of $e^{0.3} = 1.35$. The random effect has standard deviation $\sqrt{0.15} = 0.39$ and 60% of its variance is spatial. It is simulated with the eigenvector expansion of Section 9.3.7, using the scaling constant $c$ of Section 9.3.4. This is the BYM2 model's own prior, a friendly truth for BYM2; Exercise 11 uses a different one. The baseline rate is 0.001 deaths per resident over five years, as in Chapter 0. After the counts are drawn, the analyst's expected counts are $E_i = \hat{r}\cdot\text{pop}_i$ with $\hat{r}$ the overall rate, and the **true relative risk** is $\theta_i = (\text{true mean count})/E_i$.

```python
def county_map(m=12):
    """A grid of square 'counties': centroids, rook adjacency, populations,
    and a smooth standardized covariate (a deprivation index)."""
    g = (np.arange(m) + 0.5) / m
    xx, yy = np.meshgrid(g, g)
    S = np.column_stack([xx.ravel(), yy.ravel()])
    n = m * m
    A = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            if i + 1 < m:
                A[k, k + m] = A[k + m, k] = 1
            if j + 1 < m:
                A[k, k + 1] = A[k + 1, k] = 1
    rng = np.random.default_rng(2031)
    pop = np.round(np.exp(rng.normal(np.log(5000), 1.0, n))).astype(int)
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2))
    L = np.linalg.cholesky(np.exp(-d / 0.25) + 1e-8 * np.eye(n))
    x = L @ np.random.default_rng(2032).standard_normal(n)
    return S, A, pop, (x - x.mean()) / x.std()


S, A, pop, x = county_map()
n = len(S)
X = np.column_stack([np.ones(n), x])
e_val, U_vec = np.linalg.eigh(np.diag(A.sum(axis=1)) - A)
c_scale, _ = scaling_constant(A)

sigma2_true, rho_true, beta1_true, rate0 = 0.15, 0.6, 0.3, 0.001
rng = np.random.default_rng(2070)
u_star = U_vec[:, 1:] @ (rng.standard_normal(n - 1) / np.sqrt(e_val[1:]))
v_iid = rng.standard_normal(n)
u_true = np.sqrt(sigma2_true) * (np.sqrt(rho_true / c_scale) * u_star
                                 + np.sqrt(1 - rho_true) * v_iid)
mean_count = rate0 * pop * np.exp(beta1_true * x + u_true)
y = rng.poisson(mean_count)

rate_hat = y.sum() / pop.sum()
E = rate_hat * pop
offset = np.log(E)
theta_true = mean_count / E
smr = y / E

print(f"{n} counties; populations {pop.min()} to {pop.max()}, "
      f"median {int(np.median(pop))}")
print(f"total deaths {y.sum()}, median county {int(np.median(y))}, "
      f"{(y == 0).sum()} counties with none")
print(f"overall rate {rate_hat:.5f}; expected counts E: "
      f"{E.min():.2f} to {E.max():.1f}, median {np.median(E):.2f}")
print(f"true relative risks: {theta_true.min():.2f} to {theta_true.max():.2f}")
sp_var = np.var(np.sqrt(sigma2_true * rho_true / c_scale) * u_star)
iid_var = np.var(np.sqrt(sigma2_true * (1 - rho_true)) * v_iid)
print(f"realized variances: spatial {sp_var:.3f}, i.i.d. {iid_var:.3f}, "
      f"spatial share {sp_var / (sp_var + iid_var):.2f}")

order = np.argsort(smr)
smallest = E <= np.quantile(E, 0.25)
print(f"10 highest SMRs from the smallest quarter of counties: "
      f"{smallest[order[-10:]].mean():.0%}")
print(f"10 lowest SMRs from the smallest quarter of counties: "
      f"{smallest[order[:10]].mean():.0%}")
print(f"variance of the SMR: smallest quarter "
      f"{np.var(smr[smallest]):.3f}, largest quarter "
      f"{np.var(smr[E >= np.quantile(E, 0.75)]):.3f}")
print(f"largest expected count among counties with no deaths: "
      f"{E[y == 0].max():.2f}")
print("expected counts of the 10 highest SMRs:", np.round(E[order[-10:]], 1))
print(f"correlation of the deprivation index with the true effect: "
      f"{np.corrcoef(x, u_true)[0, 1]:.2f}")


def rmse(est, mask=None):
    mask = np.ones(n, bool) if mask is None else mask
    return np.sqrt(np.mean((est[mask] - theta_true[mask])**2))


largest = E >= np.quantile(E, 0.75)
```

The output describes a hard data set. The 144 counties have populations from 369 to 225,262 and a total of 1,476 deaths, with a median of 5 per county and 14 counties with none. Expected counts run from 0.40 to 245.5, median 5.40, and the true relative risks from 0.32 to 2.38. In this realization the spatial share of the variance is 0.40, below the 0.6 it was designed to have: one draw of a spatially correlated field can differ from its design, as Chapter 1 explained.

The raw SMRs behave as Section 9.2.2 predicts. Of the ten highest SMRs, 50% belong to the smallest quarter of counties, where 25% would be expected if size did not matter, and 90% of the ten lowest do. (Chapter 0 found 55% and 80% in a different simulation: 100 counties with a larger spread in population and a constant risk, and averaged over 1,000 repetitions, not one map. The two setups agree in direction and roughly in size.) The variance of the SMR is 0.969 in the smallest quarter of counties and 0.199 in the largest. Figure 9.1 shows the funnel plot.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

os.makedirs("figures/ch09", exist_ok=True)

# Figure 9.1: the funnel plot of the SMR
fig, ax = plt.subplots(figsize=(6.5, 4))
grid_E = np.logspace(np.log10(E.min() * 0.9), np.log10(E.max() * 1.1), 200)
lo = stats.poisson.ppf(0.025, grid_E) / grid_E
hi = stats.poisson.ppf(0.975, grid_E) / grid_E
ax.fill_between(grid_E, lo, hi, color="0.9", step="mid",
                label="95% limits if every county had relative risk 1")
ax.scatter(E, smr, s=14, color="tab:blue", label="county SMR")
top = order[-10:]
ax.scatter(E[top], smr[top], s=40, facecolors="none", edgecolors="tab:red",
           label="10 highest SMRs")
ax.axhline(1.0, color="0.4", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("expected count E (log scale)")
ax.set_ylabel("standardized mortality ratio")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig("figures/ch09/fig_9_1_funnel.png", dpi=120)
```

![Figure 9.1. The SMR of each county against its expected count, on a log scale. The shaded funnel gives the 95% limits for a county whose true relative risk is 1. The ten highest SMRs are circled.](../figures/ch09/fig_9_1_funnel.png)

In Figure 9.1, the points scatter widely at small expected counts and narrow toward 1 as the expected count grows, with most points inside the funnel. The funnel's outline is jagged because its limits are quantiles of a discrete distribution. Nine of the ten highest SMRs (circled) have expected counts under 8; the exception, with an expected count of 55 and an SMR of 2.3, lies far above the funnel, and that excess is real. All 14 counties with no deaths sit at the bottom edge with expected counts under 3. A reader who looked only at a map of SMRs would not be able to tell the one real excess from the nine accidents of small numbers; a reader with the funnel can.

### 9.6.2 The Poisson GLM and Empirical Bayes

The next block fits the Poisson GLM with the deprivation index, using the IRLS function of Section 9.1.6, and the gamma-Poisson empirical Bayes estimates of Section 9.3.1. It also fits the three spatial models by PQL with REML for the hyperparameters, which Section 9.6.3 discusses.

```python
beta_g, cov_g, mu_g = poisson_irls(y, X, offset)
disp = np.sum((y - mu_g)**2 / mu_g) / (n - 2)
print(f"Poisson GLM: slope {beta_g[1]:.3f}, naive SE "
      f"{np.sqrt(cov_g[1, 1]):.4f}, Pearson dispersion {disp:.2f}, "
      f"quasi-Poisson SE {np.sqrt(cov_g[1, 1] * disp):.4f}")

a_hat, w_eb, eb, eb_sd = eb_gamma_poisson(y, E)
print(f"empirical Bayes: a = {a_hat:.2f} (prior sd of the relative risk "
      f"{1 / np.sqrt(a_hat):.2f}); weight on the SMR from "
      f"{w_eb.min():.2f} to {w_eb.max():.2f}")

priors = {k: make_prior(A, k) for k in ("leroux", "bym2", "bym")}
starts = {"leroux": (np.log(0.2), 0.0), "bym2": (np.log(0.2), 0.0),
          "bym": (np.log(0.05), np.log(0.1))}
pql = {k: fit_pql(y, X, offset, priors[k], starts[k]) for k in priors}
for k, f in pql.items():
    print(f"{k:7s} PQL-REML: beta1 = {f['beta'][1]:.3f} "
          f"(SE {np.sqrt(f['cov'][1, 1]):.4f}), hyperparameters "
          f"({f['hyper'][0]:.4f}, {f['hyper'][1]:.4f}), "
          f"{f['outer']} outer iterations")
s2, rho = pql["bym2"]["hyper"]
print(f"BYM2 mapped to BYM: sigma_u2 = {s2 * rho / c_scale:.4f}, "
      f"sigma_v2 = {s2 * (1 - rho):.4f}")

print("\nRMSE against the true relative risks")
print("method            all   smallest quarter   largest quarter")
methods = {"SMR": smr, "Poisson GLM": mu_g / E, "empirical Bayes": eb,
           "Leroux": pql["leroux"]["mu"] / E, "BYM2": pql["bym2"]["mu"] / E}
for name, est in methods.items():
    print(f"{name:15s}  {rmse(est):.3f}   {rmse(est, smallest):.3f}"
          f"              {rmse(est, largest):.3f}")

print("\nrank correlation with the true relative risks")
for name, est in methods.items():
    print(f"{name:15s}  {stats.spearmanr(est, theta_true)[0]:.3f}")
```

The GLM estimates a slope of 0.177 with a naive standard error of 0.0242. The truth is 0.3, which is more than five of these standard errors away. The Pearson dispersion is 2.58, so the counts vary about two and a half times more than Poisson, and the quasi-Poisson standard error is 0.0389. Two things are at work. The naive standard error is too small, as the dispersion alone shows. And the estimate itself is low: in this realization the correlation between the deprivation index and the true random effect happens to be $-0.15$, so the unexplained spatial effect partly offsets the covariate. One data set cannot separate the two, and Section 9.6.5 repeats the experiment 100 times to do so.

Empirical Bayes estimates $\hat{a} = 5.20$, a prior standard deviation of $1/\sqrt{5.20} = 0.44$ for the relative risks. The weights on the SMR run from 0.07 for the smallest county to 0.98 for the largest, so the smallest counties are almost entirely replaced by the overall rate, and the largest keep their own estimates.

### 9.6.3 The Spatial Models by PQL

The three spatial models give consistent answers. Leroux estimates the slope as 0.210 (standard error 0.0507) with hyperparameters $(\sigma^2, \rho) = (0.2193, 0.2201)$. BYM2 estimates it as 0.226 (0.0520) with $(\sigma^2, \rho) = (0.1524, 0.2125)$. BYM, which is BYM2 under another name, gives exactly the same slope and standard error, with variances $(\sigma_u^2, \sigma_v^2) = (0.0478, 0.1200)$; mapping the BYM2 estimates through $\sigma_u^2 = \sigma^2\rho/c$ and $\sigma_v^2 = \sigma^2(1 - \rho)$ gives the same $(0.0478, 0.1200)$, as Section 9.3.6 predicted.

The BYM2 estimate of the total variance, 0.152, is very close to the true 0.15. The mixing parameter is estimated as 0.21, against a design value of 0.6 and a realized spatial share of 0.40; Section 9.6.4 shows how little the data say about it.

The accuracy of the estimates of the true relative risks is summarized by the root mean squared error:

| Method | All counties | Smallest quarter | Largest quarter |
| :--- | :--- | :--- | :--- |
| SMR | 0.523 | 0.778 | 0.202 |
| Poisson GLM with deprivation | 0.358 | 0.387 | 0.310 |
| Empirical Bayes | 0.310 | 0.377 | 0.160 |
| Leroux | 0.268 | 0.330 | 0.124 |
| BYM2 | 0.267 | 0.320 | 0.126 |

The ordering is the one the chapter has been building toward. The raw SMR is worst overall (0.523), and nearly four times worse in the smallest counties than in the largest (0.778 against 0.202), which is the funnel plot in a single column. The GLM uses the deprivation index and helps a lot in the small counties (0.387), but it hurts in the large ones (0.310 against the SMR's 0.202), because a model with no random effect cannot track the county-level risk where the data are informative. Empirical Bayes keeps what the large counties know (0.160) and shrinks the small ones (0.377). The spatial models do best in every column, with an overall error about half the SMR's, and BYM2 and Leroux are nearly equal, as the earlier comparison suggested. The rank correlation between the estimates and the true relative risks tells the same story from another angle: 0.600 for the SMR, 0.604 for the Poisson GLM, 0.630 for empirical Bayes, and 0.755 and 0.756 for Leroux and BYM2. Shrinkage toward a single value changes the ranking of counties only a little; borrowing from neighbors changes it a great deal, and for the better.

```python
# Figure 9.2: maps of the true relative risk and five estimates
panels = [("True relative risk", theta_true), ("SMR", smr),
          ("Empirical Bayes", eb), ("Poisson GLM with x", mu_g / E),
          ("Leroux", pql["leroux"]["mu"] / E),
          ("BYM2", pql["bym2"]["mu"] / E)]
fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
for ax, (title, val) in zip(axes.ravel(), panels):
    im = ax.imshow(val.reshape(12, 12), origin="lower", cmap="viridis",
                   norm=LogNorm(0.4, 2.5), extent=(0, 1, 0, 1))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.8, label="relative risk (log scale)")
fig.savefig("figures/ch09/fig_9_2_maps.png", dpi=120, bbox_inches="tight")
```

![Figure 9.2. True relative risk and five estimates of it, on one logarithmic color scale. White squares in the SMR map are counties with no deaths, where the logarithm is undefined.](../figures/ch09/fig_9_2_maps.png)

Figure 9.2 shows the maps. The SMR map is the noisiest, with isolated bright and dark counties and white squares where there were no deaths. Empirical Bayes is much smoother, since it shrinks every county toward 1, and it keeps only the stronger signals. The Poisson GLM map is nearly flat with only a slight gradient, because the deprivation index explains a small part of the variation. The Leroux and BYM2 maps are close to each other and show patches of high and low risk with softer edges than the truth, which is what shrinking toward neighbors produces; the rank correlations above say that these patches line up with the true ones better than any other column does.

### 9.6.4 Laplace Maximum Likelihood and Weak Identification

Now fit the same models by maximizing the Laplace likelihood of Section 9.5, over the coefficients and hyperparameters together, and test whether a random effect is needed at all.

```python
lap = {}
for k in priors:
    start = np.concatenate([[0.0, 0.0], starts[k]])
    lap[k] = optimize.minimize(laplace_negloglik, start, method="L-BFGS-B",
                               args=(y, X, offset, priors[k]))
    r = lap[k]
    print(f"{k:7s} Laplace ML: beta = {np.round(r.x[:2], 3)}, hyperparameters "
          f"{np.round(priors[k](r.x[2:])[3], 4)}, "
          f"negative log-likelihood {r.fun:.3f}")

nll_glm = -np.sum(y * np.log(mu_g) - mu_g - special.gammaln(y + 1))
lr = 2 * (nll_glm - lap["bym2"].fun)
print(f"\nPoisson GLM negative log-likelihood {nll_glm:.3f}; "
      f"likelihood ratio statistic for sigma2 = 0: {lr:.1f}")
print(f"mixture p-value: {0.5 * stats.chi2.sf(lr, 1):.2e}")

print("\nprofile of the Laplace likelihood in rho (BYM2)")
best = lap["bym2"]
for rho_p in (0.02, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 0.999):
    logit = np.log(rho_p / (1 - rho_p))
    prof = optimize.minimize(
        lambda q: laplace_negloglik(np.array([q[0], q[1], q[2], logit]),
                                    y, X, offset, priors["bym2"]),
        best.x[:3], method="L-BFGS-B")
    print(f"rho = {rho_p:5.3f}: loss of log-likelihood "
          f"{prof.fun - best.fun:5.2f}, sigma2 = {np.exp(prof.x[2]):.3f}")
```

The Laplace estimates agree with PQL where it matters. For BYM2 the slope is 0.227 against PQL's 0.226, and the hyperparameters are $(0.1538, 0.2045)$ against $(0.1524, 0.2125)$. For Leroux the slope is 0.209 against 0.210, and $\sigma^2$ is 0.172 against 0.219; the estimates of $\rho$ are 0.064 and 0.220, which say little except that $\rho$ is small and uncertain. BYM again matches BYM2, with the same maximized log-likelihood (negative log-likelihood 355.386) and variances $(0.0465, 0.1224)$. The counts here are not so small that PQL fails.

The Poisson GLM's negative log-likelihood is 413.701, against 355.386 for BYM2, so the likelihood ratio statistic for $\sigma^2 = 0$ is 116.6. Against the mixture reference of Section 9.5.3, the p-value is about $1.7 \times 10^{-27}$. There is overwhelming evidence of variation in risk beyond what the Poisson and the deprivation index explain.

The question is how much of that variation is spatial. The profile in the last table of the output holds $\rho$ fixed at each value and maximizes over the coefficients and $\sigma^2$.

| $\rho$ | 0.02 | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 | 0.95 | 0.999 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Loss of log-likelihood | 0.31 | 0.07 | 0.00 | 0.18 | 0.70 | 1.67 | 2.89 | 3.44 |
| $\sigma^2$ | 0.148 | 0.150 | 0.154 | 0.164 | 0.179 | 0.202 | 0.227 | 0.238 |

The loss never exceeds 1.67 for any $\rho$ from 0.02 to 0.8, so every one of these values is inside the 95% likelihood region, whose cutoff for one parameter is 1.92 (half of 3.84, the 95% point of a $\chi^2_1$ distribution). The data cannot tell a mostly independent random effect from one that is 80% spatial. The truth, 0.6, is comfortably inside. The total variance $\sigma^2$, in contrast, moves little: from 0.148 to 0.238 across the whole range of $\rho$. This is the BYM weak identification of Section 9.3.3, seen from the BYM2 side: the total is determined, the split is not.

```python
# Figure 9.3: weak identification of the BYM variance split
fixed = lap["bym"].x[:2]
grid_u = np.linspace(np.log(0.003), np.log(1.0), 25)
surface = np.array([[laplace_negloglik(np.array([*fixed, a, b]), y, X,
                                       offset, priors["bym"])
                     for a in grid_u] for b in grid_u])
surface -= lap["bym"].fun
rhos = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 0.999])
loss = []
for rho_p in rhos:
    logit = np.log(rho_p / (1 - rho_p))
    prof = optimize.minimize(
        lambda q: laplace_negloglik(np.array([q[0], q[1], q[2], logit]),
                                    y, X, offset, priors["bym2"]),
        lap["bym2"].x[:3], method="L-BFGS-B")
    loss.append(prof.fun - lap["bym2"].fun)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cs = axes[0].contourf(np.exp(grid_u), np.exp(grid_u), surface,
                      levels=[0, 1, 2, 4, 8, 16, 32, 64], cmap="viridis_r")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("structured variance (ICAR part)")
axes[0].set_ylabel("unstructured variance")
fig.colorbar(cs, ax=axes[0], label="loss of log-likelihood")
axes[1].plot(rhos, loss, "o-")
axes[1].axhline(1.92, color="0.4", ls="--", lw=0.8)
axes[1].axvline(rho_true, color="tab:red", lw=0.8)
axes[1].set_xlabel("mixing parameter rho")
axes[1].set_ylabel("loss of log-likelihood")
fig.tight_layout()
fig.savefig("figures/ch09/fig_9_3_identification.png", dpi=120)
print(f"cells within 2 units of the maximum: {(surface < 2).sum()} of "
      f"{surface.size}; within 4 units: {(surface < 4).sum()}")
near = surface < 2
print(f"within 2 units: structured variance "
      f"{np.exp(grid_u[near.any(axis=0)]).min():.3f} to "
      f"{np.exp(grid_u[near.any(axis=0)]).max():.3f}; unstructured "
      f"{np.exp(grid_u[near.any(axis=1)]).min():.3f} to "
      f"{np.exp(grid_u[near.any(axis=1)]).max():.3f}")
print(f"loss at the smallest structured variance on the grid "
      f"({np.exp(grid_u[0]):.3f}): {surface[:, 0].min():.2f}")
print("figures saved")
```

![Figure 9.3. Left: loss of log-likelihood for the two BYM variances, with the coefficients held at their estimates. Right: the profile of the BYM2 likelihood in the mixing parameter rho; the dashed line is the 95% cutoff and the vertical line marks the design value 0.6.](../figures/ch09/fig_9_3_identification.png)

In the left panel of Figure 9.3 the bright region where the loss is under two units covers 81 of the 625 grid cells, and 161 cells have a loss under four. It is a band, long in the direction of the structured variance and narrow in the direction of the unstructured variance. The unstructured variance is determined to within a factor of about five: the cells with a loss under two have it between 0.034 and 0.184. The structured variance is not: it can be anywhere from the smallest value on the grid, 0.003, to roughly 0.23 with little loss, and setting it to 0.003 costs only 0.40 units of log-likelihood. On this data set, in other words, we cannot prove that there is any spatially structured variation beyond the unstructured part, although there is some. The right panel shows the BYM2 profile in $\rho$: it is flat near its minimum at 0.2 and crosses the dashed 95% line only beyond $\rho = 0.8$.

The reason is the same as in Chapter 5, where only a combination of $\sigma^2$ and $\phi$ was well determined: 144 counties with a median of five deaths each carry little information about how a risk surface is arranged, as opposed to how large it is. More counties or larger counts sharpen it; Exercise 15 tries a map with 576 counties.

### 9.6.5 Does the Standard Error Cover?

The last experiment repeats Chapter 1's. Simulate 100 new data sets from the same truth, with new random effects and new counts each time and the same covariate and populations. For each, fit the Poisson GLM, the quasi-Poisson correction, and the BYM2 model by PQL, and record whether the 95% interval for the slope contains the true 0.3.

```python
def simulate_counts(rng):
    """New spatial effects and counts from the same BYM2 truth."""
    ustar = U_vec[:, 1:] @ (rng.standard_normal(n - 1) / np.sqrt(e_val[1:]))
    u = np.sqrt(sigma2_true) * (np.sqrt(rho_true / c_scale) * ustar
                                + np.sqrt(1 - rho_true)
                                * rng.standard_normal(n))
    return rng.poisson(rate0 * pop * np.exp(beta1_true * x + u))


rng = np.random.default_rng(2071)
reps = 100
est = np.zeros((reps, 2))
se = np.zeros((reps, 3))
for r in range(reps):
    yr = simulate_counts(rng)
    off_r = np.log(yr.sum() / pop.sum() * pop)
    b_r, cov_r, mu_r = poisson_irls(yr, X, off_r)
    disp_r = max(np.sum((yr - mu_r)**2 / mu_r) / (n - 2), 1.0)
    f_r = fit_pql(yr, X, off_r, priors["bym2"], (np.log(0.15), 0.0))
    est[r] = b_r[1], f_r["beta"][1]
    se[r] = (np.sqrt(cov_r[1, 1]), np.sqrt(cov_r[1, 1] * disp_r),
             np.sqrt(f_r["cov"][1, 1]))
print("method              mean estimate   SD of estimates   mean SE   "
      "coverage")
for j, (name, col) in enumerate((("Poisson GLM", 0),
                                 ("quasi-Poisson", 0),
                                 ("BYM2 (PQL)", 1))):
    cover = np.mean(np.abs(est[:, col] - beta1_true) < 1.96 * se[:, j])
    print(f"{name:18s}  {est[:, col].mean():12.3f}  "
          f"{est[:, col].std():15.3f}  {se[:, j].mean():8.3f}  "
          f"{cover:8.0%}")
```

| Method | Mean estimate | SD of estimates | Mean SE | Coverage |
| :--- | :--- | :--- | :--- | :--- |
| Poisson GLM | 0.282 | 0.100 | 0.023 | 35% |
| quasi-Poisson | 0.282 | 0.100 | 0.038 | 46% |
| BYM2 (PQL) | 0.292 | 0.048 | 0.052 | 96% |

All three estimators are close to the truth on average (0.282 and 0.292 against 0.3). The difference is in their uncertainty. The Poisson GLM's slope varies from data set to data set with a standard deviation of 0.100, but its standard error says 0.023: more than four times too small, and a 95% interval that covers 35% of the time. The quasi-Poisson correction multiplies the standard error by the square root of the dispersion, which raises it to 0.038, still about two and a half times too small, with 46% coverage. It corrects for variation that is independent between counties, but the extra variation here is spatial, and the deprivation index is itself spatially smooth, which is the combination that Chapter 1 showed to be worst.

The BYM2 model's standard error, 0.052, matches the actual spread of 0.048, and the interval covers 96% of the time. Its estimates are also more precise: a standard deviation of 0.048 against 0.100, so the GLM's variance is about four times larger. This is the efficiency gain of generalized least squares (Section 1.3.5), now for counts.

---

## Exercises

**Analytical**

1. For the binomial with $m$ trials and success probability $p$, show that $b(\vartheta) = m\log(1 + e^{\vartheta})$ with $\vartheta = \operatorname{logit}(p)$, verify that $b'(\vartheta) = mp$ and $b''(\vartheta) = mp(1 - p)$, and find the IRLS weights $v_i$ and working response $z_i$ for the logit link.

2. For a Poisson regression with the identity link $\mu_i = \mathbf{x}_i'\boldsymbol{\beta}$, which is not canonical, find $\partial\mu_i/\partial\eta_i$, the weights $v_i$, and the working response $z_i$. Show that the observed information, which involves $\mathbf{y}$, differs from the expected information $\mathbf{X}'\mathbf{V}\mathbf{X}$.

3. Show that $\operatorname{Var}(\text{SMR}_i \mid \theta_i) = \theta_i/E_i$, and find the expected count at which the coefficient of variation of the SMR is 10% when $\theta = 1$ and when $\theta = 2$. Why does a county with a higher true risk have a more precisely estimated relative risk?

4. In the gamma-Poisson model of Section 9.3.1, show that the marginal mean and variance of $Y_i$ are $E_i$ and $E_i + E_i^2/a$. Show that the shrinkage weight $w_i = E_i/(E_i + a)$ tends to 1 as $E_i \to \infty$, and that, averaged over the gamma prior and the Poisson noise, the mean squared error of the SMR is $1/E_i$ while that of the posterior mean is $1/(E_i + a)$.

5. Suppose the prior is $\theta_i \sim \text{Gamma}(a, a/m_i)$ with mean $m_i$. Show that the posterior mean is $w_i\,\text{SMR}_i + (1 - w_i)m_i$ with the same weight $w_i$, so the estimate shrinks toward $m_i$. How would you choose $m_i$ from a fitted Poisson regression, and what is the interpretation?

6. For the Leroux model, verify the conditional mean and variance in Section 9.3.5 from the entries of $\mathbf{Q}$. Show that the model's precision is positive definite for $\rho < 1$, and find the conditional distribution of $u_i$ as $\rho \to 1$.

7. For a path of four regions, find $\boldsymbol{\Sigma}^{\star}$, the marginal variances, and the scaling constant $c$, by hand, and check them with the code of Section 9.3.4. Then add an isolated fifth region with no neighbors. Show that the formula applied to the whole map fails, and that applying it to each connected component and ignoring the isolated region gives the same $c$ as before.

8. Eliminate $\mathbf{u}$ from Henderson's equations. Use the identity $\mathbf{V} - \mathbf{V}(\mathbf{V} + \mathbf{Q})^{-1}\mathbf{V} = (\mathbf{V}^{-1} + \mathbf{Q}^{-1})^{-1}$ (a form of the Woodbury identity, derived from the block inversion formula of Appendix A.1) to show that $\hat{\boldsymbol{\beta}}$ solves the generalized least squares equations for $\mathbf{z} \sim N(\mathbf{X}\boldsymbol{\beta}, \mathbf{V}^{-1} + \mathbf{Q}^{-1})$. Then show that $\hat{\mathbf{u}} = \mathbf{Q}^{-1}(\mathbf{V}^{-1} + \mathbf{Q}^{-1})^{-1}(\mathbf{z} - \mathbf{X}\hat{\boldsymbol{\beta}})$, and interpret this as the kriging formula of Section 4.4.3.

9. For the one-region model of Section 9.5.2, write the equation for the mode $\hat{u}$ and the Laplace approximation of the integral. Evaluate them for $y = 0$ and $\sigma^2 = 0.5$ using a calculator and a few Newton steps, and compare with the first entry of the table. Show that the approximation is exact if $y \mid u$ were $N(u, 1)$.

10. Show that the BYM2 model with $(\sigma^2, \rho)$ and the BYM model with $(\sigma_u^2, \sigma_v^2) = (\sigma^2\rho/c,\ \sigma^2(1 - \rho))$ have the same covariance matrix. Express $\rho$ in terms of $\sigma_u^2$, $\sigma_v^2$, and $c$.

**Computational and simulation**

11. Replace the truth of Section 9.6.1 by a smooth Gaussian field with exponential correlation (range 0.25) of variance 0.10 plus independent noise of variance 0.05. Refit the models and compare the RMSE table and the estimate of $\rho$. Which model, if any, is hurt by the change of truth?

12. Simulate vaccination counts: county $i$ has $m_i = \text{pop}_i$ residents of whom $Y_i \sim \text{Binomial}(m_i, p_i)$ are vaccinated, with $\operatorname{logit}(p_i) = 0.8 - 0.4\,x_i + u_i$ and $\mathbf{u}$ drawn from BYM2. Modify `poisson_irls` and `pirls_mixed` for the logit link, using the weights and working response of Exercise 1, and fit the model.

13. Repeat the Laplace and PQL fits of the BYM2 model on 50 simulated data sets for baseline rates of 0.00025, 0.001, and 0.004 (expected counts one quarter, one, and four times as large). How do the two estimates of $\sigma^2$ compare with the truth as the counts change? Do you see the PQL bias that Section 9.4.5 describes?

14. Add an isolated county to the map (a row and column of zeros appended to $\mathbf{A}$). Compute the scaling constant with `scaling_constant` and with a naive pseudo-inverse of the whole Laplacian (`np.linalg.pinv`), and compare. Fit the BYM2 model to simulated data and explain what the isolated county's estimated effect is.

15. Rerun the profile of $\rho$ (Section 9.6.4) on a map of $24 \times 24$ counties, generated with `county_map(24)` and with simulated responses from the same BYM2 truth. How does the width of the 95% region for $\rho$ change relative to the 144-county map?

16. Modify `eb_gamma_poisson` so that each county shrinks toward $\exp(\mathbf{x}_i'\hat{\boldsymbol{\beta}})$ from the Poisson GLM (Exercise 5) and estimate $a$ from the negative binomial with the corresponding means. Compare its RMSE with the table of Section 9.6.3.

---

## What Comes Next

This chapter completes Part III. For areal data we now have the full path from the raw ratio to a fitted spatial model: why small counties mislead, how shrinkage and neighbors repair it, how the model is estimated, and how its uncertainty compares with the naive one. Part IV changes what is random. In a point pattern the locations of the events are the data, and there are no regions to count in. But if we lay a fine grid of pixels over the study region, the number of events in each pixel is a Poisson count, with a mean that depends on location. Chapter 10 explores such patterns, Chapter 11 fits them with exactly the Poisson regression and IRLS of Section 9.1, using pixel areas as offsets, and Chapter 12 returns to this chapter's mixed model when the log intensity gets a Gaussian random effect, the log-Gaussian Cox process.
