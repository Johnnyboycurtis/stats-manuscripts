# Chapter 11: Point Processes as Generalized Linear Models

Chapter 10 described point patterns with summaries: kernel maps of the intensity, distance functions, and Ripley's $K$. Those tools say what a pattern looks like. This chapter asks the questions of a regression course. Which covariates explain where the events occur? By how much? How uncertain is that answer? Is there an excess of events near a suspected source, and where is it?

The answer is a pleasant surprise. A point process with a log-linear intensity is a generalized linear model in disguise. After a short derivation, fitting it means running the iteratively reweighted least squares (IRLS) algorithm of Chapter 9 on a data set built from the events and a grid of extra points. Case-control data turn into logistic regression. A known baseline becomes an offset. Almost nothing needs to be learned from scratch; the work is in seeing why these familiar tools apply.

The same machinery also shows where the familiar tools stop working. Chapters 1 and 9 each ended with a warning: when observations are dependent, the standard errors of a model that ignores the dependence are too small. Point patterns repeat the warning in a new form. If events cluster, the standard errors from the Poisson GLM are too small by a factor of two or more. We end the chapter by measuring the damage and repairing it.

Three running stories carry the chapter. A traffic safety office has the locations of crashes and wants to know how road layout and population explain them. A health department has the addresses of disease cases and of a sample of healthy controls, and a suspected industrial plant. An outbreak surveillance team has a stream of event locations and wants to know whether some small area has too many. As in earlier chapters, every data set is simulated, so each method can be checked against the truth.

> **Chapter at a Glance**
>
> - **Intensity as a Regression Function (§11.1):** The log-linear model $\lambda(\mathbf{s}) = \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$, its link to the count regressions of Chapter 9, and the crash data.
> - **The Exact Likelihood (§11.2):** The point process log-likelihood derived as the limit of Bernoulli trials on a finer and finer grid, with its score equations and information.
> - **The Berman-Turner Device (§11.3):** Replacing the integral by quadrature turns the likelihood into a weighted Poisson regression. A three-event example is worked by hand, and the crash pattern is fitted.
> - **Offsets (§11.4):** Modeling the excess over a known baseline, with the plant story.
> - **Case-Control Data (§11.5):** Why the log-odds of being a case is the log relative risk, so that logistic regression fits the problem and the unknown baseline drops out.
> - **Scan Statistics (§11.6):** A likelihood ratio for every circular window, and why the p-value of the best window must come from simulation.
> - **Trend Versus Interaction (§11.7):** Why Ripley's $K$ reports clustering for a pattern whose only structure is a trend, and the inhomogeneous $K$ that fixes it.
> - **Inference Under Clustering (§11.8):** Why Poisson GLM standard errors fail for clustered events, and three repairs: a pair-correlation sandwich, a block sandwich, and a parametric bootstrap.
> - **Walkthrough and Lab (§11.9):** Repeated-sampling experiments for each of the claims above.

---

## 11.1 Intensity as a Regression Function

### 11.1.1 The Model and Its Parts

A point pattern is a random collection of locations in a study region $\mathcal{D}$. In this chapter $\mathcal{D}$ is the unit square, so $\operatorname{area}(\mathcal{D}) = 1$. For a subset $B$ of the region, $N(B)$ is the number of events that fall in $B$, and the **intensity** $\lambda(\mathbf{s})$ is the expected number of events per unit area near the location $\mathbf{s}$. Chapter 10 introduced these as the language of point patterns, and its basic model is the one we use here. In a **Poisson process** with intensity $\lambda(\mathbf{s})$:

- the count $N(B)$ in any region $B$ has a Poisson distribution with mean $\int_B\lambda(\mathbf{s})\,d\mathbf{s}$, and
- counts in regions that do not overlap are independent.

The second property is the point-pattern version of independent errors. It says that where one event falls tells us nothing about where another falls, beyond what the intensity says. It is the property that will fail in Section 11.8.

To explain the intensity with covariates, write it as a log-linear function:

$$
\lambda(\mathbf{s}) = \exp\!\left(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}\right)
$$

Here $\mathbf{x}(\mathbf{s})$ is a vector of covariates evaluated at location $\mathbf{s}$, such as an intercept, the distance from $\mathbf{s}$ to a highway, and the population density at $\mathbf{s}$, and $\boldsymbol{\beta}$ is a vector of coefficients. The exponential keeps the intensity positive, and it makes effects multiplicative: raising a covariate by one unit multiplies the rate of events by $\exp(\beta_j)$. This is the log link of Poisson regression, applied to a surface instead of a table of counts.

### 11.1.2 The Crash Data

A city traffic safety office has the locations of crashes in a square district and two covariates measured everywhere on the map: the distance to the main highway and the local population density. The highway is the line $y = 0.3 + 0.4x$, and the population is concentrated downtown, around the point $(0.7, 0.7)$. We simulate crashes with intensity

$$
\lambda(\mathbf{s}) = \exp\!\left(\beta_0 + \beta_1\,\text{road distance}(\mathbf{s}) + \beta_2\,\text{population density}(\mathbf{s})\right)
$$

with $\beta_1 = -3$ and $\beta_2 = 1.5$, and with $\beta_0$ set so that the expected number of crashes is 400. The three functions below define the setting that Chapters 10 to 12 share, so that results connect across the part. The crash pattern itself is a new simulation from a richer model than the crash example of Chapter 10, which used the intensity $600\exp\{-6\operatorname{dist}(\mathbf{s})\}$ (road distance only, about 199 expected crashes). Here the intensity also depends on population density and the expected count is 400, and Chapter 10's pattern is not reused.

```python
import numpy as np
from scipy import optimize, special, stats


def road_distance(P):
    """Distance from points P (n x 2) to the 'highway' y = 0.3 + 0.4 x."""
    return np.abs(0.4 * P[:, 0] - P[:, 1] + 0.3) / np.sqrt(0.4**2 + 1.0)


def pop_density(P):
    """Relative population density: a downtown bump on a flat base."""
    d2 = (P[:, 0] - 0.7)**2 + (P[:, 1] - 0.7)**2
    return 0.3 + 0.7 * np.exp(-d2 / (2 * 0.2**2))


def excess_risk(P):
    """Relative risk around an industrial plant at (0.3, 0.6)."""
    d2 = (P[:, 0] - 0.3)**2 + (P[:, 1] - 0.6)**2
    return np.exp(np.exp(-d2 / (2 * 0.1**2)))
```

A coefficient of $-3$ for road distance means that moving 0.1 units farther from the highway multiplies the crash rate by $e^{-0.3} = 0.74$. A coefficient of $1.5$ for population density means that an increase of 0.1 multiplies the rate by $e^{0.15} = 1.16$.

To simulate an inhomogeneous Poisson process we use **thinning**, which Chapter 10 introduced as the Lewis-Shedler algorithm. Propose events from a homogeneous Poisson process whose rate $\lambda_{\max}$ is at least the largest intensity, then keep an event at $\mathbf{s}$ with probability $\lambda(\mathbf{s})/\lambda_{\max}$. Independent thinning of a Poisson process gives a Poisson process whose intensity is the original rate times the retention probability, which is $\lambda(\mathbf{s})$.

```python
def grid_points(m):
    """Centers of an m-by-m grid of cells in the unit square."""
    g = (np.arange(m) + 0.5) / m
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


def design(P):
    """Covariates [1, road distance, population density]."""
    return np.column_stack([np.ones(len(P)), road_distance(P),
                            pop_density(P)])


def simulate_poisson(rate_fn, rate_max, rng):
    """Inhomogeneous Poisson process on the unit square, by thinning."""
    P = rng.uniform(size=(rng.poisson(rate_max), 2))
    return P[rng.uniform(size=len(P)) < rate_fn(P) / rate_max]


G = grid_points(200)
X_G = design(G)
b1, b2 = -3.0, 1.5
b0 = np.log(400) - np.log(np.mean(np.exp(X_G[:, 1:] @ [b1, b2])))
beta_true = np.array([b0, b1, b2])
rate = lambda P: np.exp(design(P) @ beta_true)
rate_max = 1.05 * rate(G).max()
print("true coefficients:", np.round(beta_true, 3))
print("expected number of crashes:", round(rate(G).mean(), 1))
rng = np.random.default_rng(2090)
P_crash = simulate_poisson(rate, rate_max, rng)
print("simulated crashes:", len(P_crash))
```

The true coefficients are $(5.846, -3, 1.5)$ and the expected count is 400. The simulation produced 403 crashes.

### 11.1.3 The Link to Chapter 9

Chapter 9 modeled a count $Y_i$ in region $i$ as Poisson with mean $E_i\exp(\mathbf{x}_i'\boldsymbol{\beta} + u_i)$, where the expected count $E_i$ entered as an offset. A region is only one way to cut up the map. Under the Poisson process, the number of events in a small region of area $a$ around $\mathbf{s}$ is Poisson with mean close to $a\lambda(\mathbf{s})$, so

$$
\log\operatorname{E}[N(B)] \approx \log a + \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}
$$

That is exactly a Poisson regression with offset $\log a$. A point process is what this model becomes when the regions shrink until each holds at most one event. The next section makes that limit precise, and the section after it shows that the limit does not require any new fitting algorithm.

---

## 11.2 The Exact Likelihood From a Refining Grid

### 11.2.1 Bernoulli Trials on a Pixel Grid

To fit $\boldsymbol{\beta}$ by maximum likelihood we need the probability of the observed pattern. Rather than state it, we derive it, using only independent coin flips.

Cut the region into $M$ square pixels of area $c_M = \operatorname{area}(\mathcal{D})/M$, with centers $\mathbf{s}_1, \dots, \mathbf{s}_M$. Under a Poisson process with intensity $\lambda$, the number of events in pixel $j$ is Poisson with mean $\int_{\text{pixel } j}\lambda \approx \lambda(\mathbf{s}_j)c_M$. For a Poisson variable with a small mean $\mu$, the probability of one event is $\mu e^{-\mu} = \mu + O(\mu^2)$, the probability of two or more is $O(\mu^2)$, and the probability of none is $1 - \mu + O(\mu^2)$. So when the pixels are small each holds zero or one event, and we can record it with an indicator $\mathbb{1}_j$ that equals 1 when pixel $j$ holds an event. The indicators are independent across pixels, because counts in disjoint regions are independent. The pixel contains an event with probability

$$
p_j = \lambda(\mathbf{s}_j)\,c_M
$$

up to terms of order $c_M^2$. The log-likelihood of the pixel indicators is the log-likelihood of independent Bernoulli trials:

$$
\ell_M(\boldsymbol{\beta}) = \sum_{j=1}^M\left[\mathbb{1}_j\log p_j + (1 - \mathbb{1}_j)\log(1 - p_j)\right]
$$

Now let the pixels shrink. Split the sum into the pixels that hold events and the pixels that do not.

**Pixels with events.** There are $n$ of them, one per event. If event $i$ lies in the pixel centered at $\mathbf{s}_{(i)}$, its term is $\log p = \log\lambda(\mathbf{s}_{(i)}) + \log c_M$. Summing over events,

$$
\sum_j\mathbb{1}_j\log p_j = \sum_{i=1}^n\log\lambda(\mathbf{s}_{(i)}) + n\log c_M
$$

As $c_M \to 0$ the pixel center $\mathbf{s}_{(i)}$ approaches the event $\mathbf{s}_i$, so the first sum tends to $\sum_i\log\lambda(\mathbf{s}_i)$.

**Pixels without events.** The second part of the log-likelihood is $\sum_j(1 - \mathbb{1}_j)\log(1 - p_j)$. Compare it with the sum over all pixels, $\sum_j\log(1 - p_j)$. The two differ by the $n$ terms for pixels that hold events, each of size at most $2\max_jp_j$, which tends to zero. For $p \le \tfrac{1}{2}$ we have $-p - p^2 \le \log(1 - p) \le -p$, so

$$
-\sum_j p_j - \sum_j p_j^2 \;\le\; \sum_j\log(1 - p_j) \;\le\; -\sum_j p_j
$$

The sum $\sum_jp_j = \sum_j\lambda(\mathbf{s}_j)c_M$ is a Riemann sum and tends to $\int_{\mathcal{D}}\lambda(\mathbf{s})\,d\mathbf{s}$. The sum $\sum_jp_j^2 \le (\max_jp_j)\sum_jp_j$ tends to zero because $\max_jp_j = O(c_M)$. So the second part of $\ell_M$ tends to $-\int\lambda$.

**Putting the parts together.** The term $n\log c_M$ does not involve $\boldsymbol{\beta}$, so it does not affect the maximizer. Subtracting it, we get

$$
\ell_M(\boldsymbol{\beta}) - n\log c_M \;\longrightarrow\; \ell(\boldsymbol{\beta}) = \sum_{i=1}^n\log\lambda(\mathbf{s}_i) - \int_{\mathcal{D}}\lambda(\mathbf{s})\,d\mathbf{s}
$$

This is the **log-likelihood of a point process**: the log-intensity summed over the events, minus the total expected number of events. Its two terms have a plain reading. The first rewards intensities that are high where events occurred. The second penalizes intensities that are high anywhere, because that predicts more events than we saw.

A small example makes the limit visible. Three events lie at $(0.1, 0.2)$, $(0.4, 0.3)$, and $(0.8, 0.9)$, and the intensity is $\lambda(\mathbf{s}) = \exp(1 + 0.5\,s_1)$, where $s_1$ is the east coordinate. The integral is $e(e^{0.5} - 1)/0.5 = 3.527$ and the three log-intensities sum to $3 + 0.5(0.1 + 0.4 + 0.8) = 3.65$, so the point process log-likelihood is $3.65 - 3.527 = 0.123$. The code builds the Bernoulli log-likelihood on $4 \times 4$, $8 \times 8$, up to $128 \times 128$ pixels and subtracts $n\log c_M$.

```python
pts3 = np.array([[0.1, 0.2], [0.4, 0.3], [0.8, 0.9]])
beta_ex = np.array([1.0, 0.5])
lam_ex = lambda P: np.exp(beta_ex[0] + beta_ex[1] * P[:, 0])
exact = np.sum(np.log(lam_ex(pts3))) - lam_ex(grid_points(400)).mean()
print("point process log-likelihood:", round(exact, 4))
for m in (4, 8, 16, 32, 64, 128):
    cell = (np.floor(pts3[:, 1] * m) * m + np.floor(pts3[:, 0] * m))
    z = np.zeros(m * m)
    z[cell.astype(int)] = 1
    c_m = 1.0 / m**2
    p = lam_ex(grid_points(m)) * c_m
    bern = np.sum(z * np.log(p) + (1 - z) * np.log(1 - p))
    print(f"M={m * m:6d}  Bernoulli log-lik={bern:9.4f}  "
          f"minus n log c_M={bern - 3 * np.log(c_m):8.4f}")
```

| Pixels $M$ | 16 | 64 | 256 | 1,024 | 4,096 | 16,384 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\ell_M - n\log c_M$ | 0.4280 | 0.1919 | 0.1290 | 0.1254 | 0.1265 | 0.1238 |

The values approach the exact value of 0.1232. They do not decrease steadily because the event sits at a different offset from its pixel center at each resolution, so $\log\lambda(\mathbf{s}_{(i)})$ differs from $\log\lambda(\mathbf{s}_i)$ by a little more or a little less each time.

### 11.2.2 The Log-Likelihood for a Log-Linear Intensity

Put $\lambda(\mathbf{s}) = \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$ into $\ell(\boldsymbol{\beta})$. The log-intensity is linear in $\boldsymbol{\beta}$, so

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n\mathbf{x}(\mathbf{s}_i)'\boldsymbol{\beta} - \int_{\mathcal{D}}\exp\!\left(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}\right)d\mathbf{s}
$$

Differentiate with respect to $\boldsymbol{\beta}$, moving the derivative inside the integral. The score vector is

$$
\mathbf{U}(\boldsymbol{\beta}) = \frac{\partial\ell}{\partial\boldsymbol{\beta}} = \sum_{i=1}^n\mathbf{x}(\mathbf{s}_i) - \int_{\mathcal{D}}\mathbf{x}(\mathbf{s})\,\lambda(\mathbf{s})\,d\mathbf{s}
$$

and the maximum likelihood estimate solves $\mathbf{U}(\hat{\boldsymbol{\beta}}) = \mathbf{0}$. In words, **the total of the covariate vector over the observed events equals the total expected under the fitted model.** This is the point-pattern version of the score equation $\mathbf{X}'(\mathbf{y} - \boldsymbol{\mu}) = \mathbf{0}$ for a GLM with a canonical link (Section 9.1.3): the residual vanishes when summed against each covariate. With an intercept in the model, the first component says the fitted intensity integrates to the observed number of events, $\int\hat{\lambda} = n$.

Differentiate once more:

$$
\frac{\partial^2\ell}{\partial\boldsymbol{\beta}\,\partial\boldsymbol{\beta}'} = -\int_{\mathcal{D}}\mathbf{x}(\mathbf{s})\mathbf{x}(\mathbf{s})'\,\lambda(\mathbf{s})\,d\mathbf{s} = -\mathcal{I}(\boldsymbol{\beta})
$$

This matrix is not random: it does not involve the events. For any vector $\mathbf{a}$, $\mathbf{a}'\mathcal{I}(\boldsymbol{\beta})\mathbf{a} = \int(\mathbf{a}'\mathbf{x}(\mathbf{s}))^2\lambda(\mathbf{s})\,d\mathbf{s} \ge 0$, so the log-likelihood is concave and, when the covariates are not collinear, has a unique maximum, which IRLS (Newton's method here) finds in a handful of steps. The matrix $\mathcal{I}(\boldsymbol{\beta})$ is the Fisher information, as in Chapters 5 and 9, and for a Poisson process $\operatorname{Var}(\hat{\boldsymbol{\beta}}) \approx \mathcal{I}(\boldsymbol{\beta})^{-1}$.

### 11.2.3 By Hand: The Homogeneous Case

Take only an intercept, $\mathbf{x} = 1$, so $\lambda$ is a constant. The score equation reads $n = \lambda\operatorname{area}(\mathcal{D})$, so $\hat{\lambda} = n/\operatorname{area}(\mathcal{D})$: the estimated intensity is the number of events per unit area. The information for $\beta_0 = \log\lambda$ is $\lambda\operatorname{area}(\mathcal{D})$, the expected count, so $\operatorname{Var}(\hat{\beta}_0) \approx 1/\operatorname{E}[n]$. This agrees with a delta-method calculation for the Poisson count: $\operatorname{Var}(\log n) \approx \operatorname{Var}(n)/(\operatorname{E}n)^2 = 1/\operatorname{E}n$. The likelihood framework reproduces the answer from a first course, as a check should.

---

## 11.3 The Berman-Turner Quadrature Device

### 11.3.1 The Integral Is the Obstacle

The likelihood of Section 11.2 has one difficulty: the integral $\int\exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})\,d\mathbf{s}$. For covariates like road distance and a population surface it has no closed form, and it changes every time $\boldsymbol{\beta}$ changes. The standard remedy is **quadrature**: approximate the integral by a weighted sum over a set of points,

$$
\int_{\mathcal{D}}f(\mathbf{s})\,d\mathbf{s} \approx \sum_k a_kf(\mathbf{s}_k^{*})
$$

where the **quadrature points** $\mathbf{s}_k^{*}$ are locations and the **weights** $a_k \ge 0$ are areas that sum to $\operatorname{area}(\mathcal{D})$. Each weight is the area of the part of the region that its point stands for.

Berman and Turner's idea was to choose the quadrature points cleverly: use the $n$ data points together with $n_d$ extra **dummy points**, usually on a regular grid. Putting the data points in the set pays off in the first term of the log-likelihood. Every data point is a quadrature point, so the sum of log-intensities over events is a sum over quadrature points as well. Both terms of $\ell$ then live on the same set of locations, and that is what turns the likelihood into a regression.

### 11.3.2 Data-Point Indicators and Pseudo-Responses

Number the $n + n_d$ quadrature points $k = 1, \dots, n + n_d$, and let $\mathbb{1}_k = 1$ if point $k$ is a data point and $\mathbb{1}_k = 0$ if it is a dummy point. The first term of the log-likelihood is exactly

$$
\sum_{i=1}^n\log\lambda(\mathbf{s}_i) = \sum_{k}\mathbb{1}_k\log\lambda(\mathbf{s}_k^{*})
$$

and the second is approximately $\sum_ka_k\lambda(\mathbf{s}_k^{*})$. Define the **pseudo-response** $y_k = \mathbb{1}_k/a_k$, so that $a_ky_k = \mathbb{1}_k$. Then

$$
\ell(\boldsymbol{\beta}) \approx \sum_{k=1}^{n+n_d}a_k\left(y_k\log\lambda(\mathbf{s}_k^{*}) - \lambda(\mathbf{s}_k^{*})\right)
$$

The pseudo-response is large (the reciprocal of a small area) at data points and zero at dummy points. It is not a count and not an integer. It is a device that makes the algebra come out right.

**Choosing the weights.** Two rules are common. For **counting weights**, lay a grid over the region; the area of each grid cell is shared equally among the quadrature points, data or dummy, that fall in the cell. If a cell holds only its own dummy point, that point gets the whole cell. For **tile weights**, give each quadrature point the area of the region closer to it than to any other point: its Dirichlet (Voronoi) tile. Both rules give weights that are positive and sum to the area. Section 11.3.5 compares them.

### 11.3.3 A Weighted Poisson Regression, and IRLS

For a response $y$ with a Poisson mean $\mu$ and a prior weight $a$, the expression $a(y\log\mu - \mu)$ is the Poisson log-likelihood contribution, except for terms that do not involve $\mu$, of $a$ identical observations: each contributes $y\log\mu - \mu$ minus $\log y!$, and the weight multiplies it. So the approximate log-likelihood above is the log-likelihood of a **weighted Poisson regression** with responses $y_k$, means $\mu_k = \lambda(\mathbf{s}_k^{*}) = \exp(\mathbf{x}_k'\boldsymbol{\beta})$, and prior weights $a_k$. That is a problem for any GLM routine with a weights argument. The responses are not integers, but the algorithm only needs the objective and its derivatives.

Recall from Section 9.1.5 that IRLS updates $\boldsymbol{\beta}$ by solving a weighted least squares problem with working weights and a working response. With prior weights, for the Poisson with log link, the working weights are $v_k = a_k\mu_k$ and the working response is $z_k = \eta_k + (y_k - \mu_k)/\mu_k$, where $\eta_k = \mathbf{x}_k'\boldsymbol{\beta}$:

$$
\mathbf{X}'\mathbf{V}\mathbf{X}\,\boldsymbol{\beta}^{\text{new}} = \mathbf{X}'\mathbf{V}\mathbf{z}, \qquad \mathbf{V} = \operatorname{diag}(a_k\mu_k)
$$

where $\mathbf{X}$ now has one row for each quadrature point.

Two checks confirm that this is the same problem as Section 11.2. The score of the weighted regression is $\mathbf{X}'\operatorname{diag}(a)(\mathbf{y} - \boldsymbol{\mu})$, whose $p$th component is $\sum_ka_kx_{kp}(y_k - \mu_k) = \sum_{i}x_p(\mathbf{s}_i) - \sum_ka_kx_{kp}\mu_k$, using $a_ky_k = \mathbb{1}_k$. This is the exact score of Section 11.2.2 with the integral replaced by quadrature. Likewise, $\mathbf{X}'\mathbf{V}\mathbf{X} = \sum_ka_k\mu_k\mathbf{x}_k\mathbf{x}_k'$ is the quadrature approximation of $\mathcal{I}(\boldsymbol{\beta}) = \int\mathbf{x}\mathbf{x}'\lambda$. So the standard errors from the GLM, the square roots of the diagonal of $(\mathbf{X}'\mathbf{V}\mathbf{X})^{-1}$, estimate the standard errors of the point process estimate, **provided the process is Poisson**. Section 11.8 shows what happens when it is not.

The function below implements IRLS with prior weights and an optional offset (Section 11.4). The working response excludes the offset, as it did in Chapter 9.

```python
def irls_poisson(X, y, a, offset=None, tol=1e-10, maxit=50):
    """Weighted Poisson regression by IRLS: mean mu_k = exp(offset_k +
    x_k'b), prior weights a_k, response y_k. Returns the estimate, its
    covariance, the log-likelihood sum a_k (y_k log mu_k - mu_k), and the
    number of iterations."""
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    b[0] = np.log(np.sum(a * y) / np.sum(a * np.exp(off)))
    for it in range(1, maxit + 1):
        eta = off + X @ b
        mu = np.exp(eta)
        v = a * mu
        z = (eta - off) + (y - mu) / mu
        info = X.T @ (v[:, None] * X)
        b_new = np.linalg.solve(info, X.T @ (v * z))
        done = np.max(np.abs(b_new - b)) < tol
        b = b_new
        if done:
            break
    mu = np.exp(off + X @ b)
    ll = np.sum(a * (y * (off + X @ b) - mu))
    cov = np.linalg.inv(X.T @ ((a * mu)[:, None] * X))
    return b, cov, ll, it


def quadrature_counting(P_data, m):
    """Data points plus an m-by-m dummy grid. Each grid cell's area is
    shared equally by the points (data and dummy) that fall in it."""
    pts = np.vstack([P_data, grid_points(m)])
    cell = np.floor(pts[:, 1] * m) * m + np.floor(pts[:, 0] * m)
    cell = cell.astype(int)
    count = np.bincount(cell, minlength=m * m)
    a = (1.0 / m**2) / count[cell]
    ind = np.r_[np.ones(len(P_data)), np.zeros(m * m)]
    return pts, a, ind
```

### 11.3.4 By Hand: Three Data Points and Four Dummy Points

Use the three events of Section 11.2.1, at $(0.1, 0.2)$, $(0.4, 0.3)$, and $(0.8, 0.9)$, and a single covariate, the east coordinate, so $\mathbf{X}$ has an intercept column and an east column. Lay a $2 \times 2$ grid over the unit square, with cells of area $0.25$, and put one dummy point at the center of each cell: $(0.25, 0.25)$, $(0.75, 0.25)$, $(0.25, 0.75)$, and $(0.75, 0.75)$. Use counting weights.

The first two events and the dummy point $(0.25, 0.25)$ all fall in the south-west cell. Three points share an area of 0.25, so each gets $0.25/3 = 0.0833$. The third event and the dummy point $(0.75, 0.75)$ share the north-east cell and get $0.125$ each. The dummy points $(0.75, 0.25)$ and $(0.25, 0.75)$ are alone in their cells and get $0.25$ each. The weights sum to 1. The pseudo-response is $y = 1/a$ for the events, which gives $12$, $12$, and $8$, and $0$ for the dummy points.

| $k$ | Point | $a_k$ | $\mathbb{1}_k$ | $y_k$ |
| :--- | :--- | :--- | :--- | :--- |
| 1 | event $(0.1, 0.2)$ | 0.0833 | 1 | 12 |
| 2 | event $(0.4, 0.3)$ | 0.0833 | 1 | 12 |
| 3 | event $(0.8, 0.9)$ | 0.125 | 1 | 8 |
| 4 | dummy $(0.25, 0.25)$ | 0.0833 | 0 | 0 |
| 5 | dummy $(0.75, 0.25)$ | 0.25 | 0 | 0 |
| 6 | dummy $(0.25, 0.75)$ | 0.25 | 0 | 0 |
| 7 | dummy $(0.75, 0.75)$ | 0.125 | 0 | 0 |

Start from the homogeneous fit $\boldsymbol{\beta} = (\log 3, 0)'$: three events per unit area. Then $\mu_k = 3$ for every point, $\eta_k = \log 3 = 1.0986$, and the working weights are $v_k = 3a_k = (0.25, 0.25, 0.375, 0.25, 0.75, 0.75, 0.375)$. The working responses are $z_k = 1.0986 + (y_k - 3)/3$. For the events with $y = 12$ this is $1.0986 + 3 = 4.0986$; for the event with $y = 8$ it is $1.0986 + 1.6667 = 2.7653$; for the dummy points it is $1.0986 - 1 = 0.0986$. So

$$
\mathbf{z} = (4.0986,\ 4.0986,\ 2.7653,\ 0.0986,\ 0.0986,\ 0.0986,\ 0.0986)'
$$

The score at the start has intercept component $\sum a_k(y_k - 3) = 3 - 3\sum a_k = 0$, as it should for a fit that matches the event count, and east component $\sum_ix_i - 3\sum_ka_kx_k = 1.3 - 3(0.50625) = -0.2188$. The weighted normal equations are

$$
\mathbf{X}'\mathbf{V}\mathbf{X} = \begin{bmatrix} 3 & 1.5188 \\ 1.5188 & 0.9778 \end{bmatrix}, \qquad \mathbf{X}'\mathbf{V}\mathbf{z} = \begin{bmatrix} 3.2958 \\ 1.4498 \end{bmatrix}
$$

and solving them gives $\boldsymbol{\beta}^{\text{new}} = (1.6286, -1.0469)'$. The code repeats the step to convergence and checks the answer by maximizing the quadrature log-likelihood directly with a general-purpose optimizer.

```python
np.set_printoptions(precision=4, suppress=True)
pts, a, ind = quadrature_counting(pts3, 2)
y = ind / a
X = np.column_stack([np.ones(len(pts)), pts[:, 0]])
print("columns: x1, x2, weight a, indicator, pseudo-response y")
print(np.column_stack([pts, a, ind, y]))
print("weights sum to", a.sum())
b = np.array([np.log(3.0), 0.0])
eta = X @ b
mu = np.exp(eta)
v = a * mu
z = eta + (y - mu) / mu
info = X.T @ (v[:, None] * X)
rhs = X.T @ (v * z)
print("score at the start:", X.T @ (a * (y - mu)))
print("working response z:", z)
print("weights v = a mu:", v)
print("X'VX:\n", info)
print("X'Vz:", rhs)
print("one step:", np.linalg.solve(info, rhs))
b_hat3, cov3, ll3, it3 = irls_poisson(X, y, a)
print("IRLS limit:", b_hat3, "after", it3, "iterations")
fun = lambda b: -np.sum(a * (y * (X @ b) - np.exp(X @ b)))
opt = optimize.minimize(fun, [np.log(3.0), 0.0], method="BFGS")
print("direct maximization:", opt.x)
```

IRLS converges in 5 iterations to $(1.5915, -1.0495)'$, and the direct maximization gives the same numbers. The events are slightly to the west of the center of the square, and the fitted intensity falls toward the east.

### 11.3.5 Choosing the Dummy Points, and the Areal GLM as a Special Case

The quadrature is an approximation, so the number of dummy points $n_d$ matters. The block below defines `fit_bt`, which assembles the quadrature, calls IRLS, and returns the estimate and its covariance (Chapter 12 defines a `fit_bt` with a different interface, taking a ready-made design matrix, indicators, and weights, because its design matrix has an extra column), and `quadrature_tiles`, which computes tile weights by assigning every point of a fine $200 \times 200$ grid to its nearest quadrature point; a point's weight is the share of the fine grid that it owns. It then fits the crash pattern three ways: with counting weights, with tile weights, and by direct maximization of the exact likelihood with a $300 \times 300$ pixel rule, which serves as the reference.

```python
def fit_bt(P_data, m, cov_fn=design, offset_fn=None, quad=None):
    """Fit lambda(s) = exp(offset + x(s)'b) by Berman-Turner quadrature."""
    pts, a, ind = quadrature_counting(P_data, m) if quad is None else quad
    off = None if offset_fn is None else offset_fn(pts)
    return irls_poisson(cov_fn(pts), ind / a, a, off)


def quadrature_tiles(P_data, m, fine=200, chunk=4000):
    """Data points plus an m-by-m dummy grid; the weight of each point is
    the area of its nearest-point (Dirichlet) tile, found on a fine grid."""
    pts = np.vstack([P_data, grid_points(m)])
    F = grid_points(fine)
    owner = np.empty(len(F), dtype=int)
    for s in range(0, len(F), chunk):
        d2 = ((F[s:s + chunk, None, :] - pts[None, :, :])**2).sum(axis=2)
        owner[s:s + chunk] = d2.argmin(axis=1)
    a = np.bincount(owner, minlength=len(pts)) / fine**2
    ind = np.r_[np.ones(len(P_data)), np.zeros(m * m)]
    return pts, a, ind


b_hat, cov, ll, it = fit_bt(P_crash, 40)
se = np.sqrt(np.diag(cov))
print("counting weights, m = 40:", np.round(b_hat, 3), "in", it, "steps")
print("model-based SE:          ", np.round(se, 3))
print("truth:                   ", np.round(beta_true, 3))
print("z-scores of the error:   ", np.round((b_hat - beta_true) / se, 2))
b_tile = fit_bt(P_crash, 40, quad=quadrature_tiles(P_crash, 40))[0]
print("Dirichlet tiles, m = 40: ", np.round(b_tile, 3))

G2 = grid_points(300)
X_G2 = design(G2)
sumx = design(P_crash).sum(axis=0)
negll = lambda b: -(sumx @ b - np.mean(np.exp(X_G2 @ b)))
opt = optimize.minimize(negll, np.r_[np.log(400), 0, 0], method="BFGS")
print("exact likelihood, 300 x 300 pixel rule:", np.round(opt.x, 3))

print("   m    n_d     b0      b1      b2   SE(b1)")
for m in (5, 10, 20, 40, 80):
    bm, cm = fit_bt(P_crash, m)[:2]
    print(f"{m:4d} {m * m:6d} {bm[0]:7.3f} {bm[1]:7.3f} {bm[2]:7.3f}"
          f"   {np.sqrt(cm[1, 1]):.3f}")

m = 20
cell = np.floor(P_crash[:, 1] * m) * m + np.floor(P_crash[:, 0] * m)
counts = np.bincount(cell.astype(int), minlength=m * m)
c_m = 1.0 / m**2
b_pix = irls_poisson(design(grid_points(m)), counts / c_m,
                     np.full(m * m, c_m))[0]
print("Poisson GLM on 20 x 20 pixel counts:", np.round(b_pix, 3))
```

With counting weights and a $40 \times 40$ dummy grid ($n_d = 1{,}600$) the estimate is $\hat{\boldsymbol{\beta}} = (5.584, -2.267, 1.758)$ with standard errors $(0.183, 0.412, 0.239)$. The truth is $(5.846, -3, 1.5)$, so the errors are $-1.43$, $1.78$, and $1.08$ standard errors: within what chance allows. Tile weights give $(5.581, -2.259, 1.761)$, and the near-exact likelihood gives $(5.581, -2.260, 1.762)$. The three agree to within $0.01$.

How many dummy points are enough?

| Grid $m \times m$ | $n_d$ | $\hat{\beta}_0$ | $\hat{\beta}_1$ | $\hat{\beta}_2$ |
| :--- | :--- | :--- | :--- | :--- |
| $5 \times 5$ | 25 | 5.618 | $-1.982$ | 1.537 |
| $10 \times 10$ | 100 | 5.576 | $-2.150$ | 1.719 |
| $20 \times 20$ | 400 | 5.571 | $-2.256$ | 1.781 |
| $40 \times 40$ | 1,600 | 5.584 | $-2.267$ | 1.758 |
| $80 \times 80$ | 6,400 | 5.581 | $-2.261$ | 1.762 |

The estimates stop changing in the second decimal at about 400 dummy points. A useful rule is to use a grid fine enough that the covariates are close to constant within a cell. The standard error of $\hat{\beta}_1$ is $0.412$ for every grid with 100 or more dummy points: it is determined by the 403 events, not by the quadrature.

The last two lines of the block show something that links this chapter to the last one. Count the events in each cell of a $20 \times 20$ grid and fit a Poisson GLM with offset $\log c_M$, where $c_M$ is the cell area, with the covariates evaluated at the cell centers. This is the regional count model of Chapter 9 with $E_i$ equal to the cell area. The estimate is $(5.569, -2.268, 1.784)$, close to the Berman-Turner fit. In the code the cell counts enter as response $\text{count}_j/c_M$ with prior weight $c_M$, which is the Berman-Turner form with the cell centers as the quadrature points and no data points. So **the areal GLM is a special case of the Berman-Turner device**, and the Berman-Turner device is the limit of the areal GLM as the regions shrink. Putting the data points in the quadrature set keeps the first term of the log-likelihood exact, because the events are not binned.

### 11.3.6 The Crash Pattern

Figure 11.1 shows the 403 crashes, the true intensity, and the fitted intensity on the same color scale. The fit recovers the main features of the truth: the ridge along the highway, and the peak where the highway passes through the populated downtown.

![Figure 11.1. Left: the 403 simulated crashes and the highway. Middle: the true crash intensity. Right: the intensity fitted by Berman-Turner IRLS. The fit recovers the ridge along the highway and the downtown peak.](../figures/ch11/fig_11_1_crash_fit.png)

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch11", exist_ok=True)
G60 = grid_points(60)
lam_true = rate(G60).reshape(60, 60)
lam_fit = np.exp(design(G60) @ b_hat).reshape(60, 60)
vmax = max(lam_true.max(), lam_fit.max())
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
axes[0].scatter(P_crash[:, 0], P_crash[:, 1], s=6, color="black")
xs = np.array([0.0, 1.0])
axes[0].plot(xs, 0.3 + 0.4 * xs, color="tab:red", lw=1.5, label="highway")
axes[0].set_title("Simulated crashes")
axes[0].legend(loc="lower right", fontsize=8)
for ax, field, title in ((axes[1], lam_true, "True intensity"),
                         (axes[2], lam_fit, "Fitted intensity")):
    im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1),
                   cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title)
for ax in axes:
    ax.set_xlabel("east")
axes[0].set_ylabel("north")
fig.colorbar(im, ax=axes[1:], label="crashes per unit area")
fig.savefig("figures/ch11/fig_11_1_crash_fit.png", dpi=120,
            bbox_inches="tight")
```

---

## 11.4 Offsets: Modeling Excess Over a Known Baseline

### 11.4.1 The Offset Model

Often the question is not where events occur but where there are more of them than expected. A health department does not ask why cases cluster downtown; the people live downtown. It asks whether there are more cases than the population explains.

Suppose a baseline intensity $\lambda_{\text{pop}}(\mathbf{s})$, proportional to the number of people at risk near $\mathbf{s}$, is known. Model the intensity of cases as the baseline times a relative risk:

$$
\lambda(\mathbf{s}) = \lambda_{\text{pop}}(\mathbf{s})\exp\!\left(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}\right), \qquad \log\lambda(\mathbf{s}) = \log\lambda_{\text{pop}}(\mathbf{s}) + \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}
$$

The term $\log\lambda_{\text{pop}}(\mathbf{s})$ has no coefficient to estimate, so it is an **offset**. In the Berman-Turner regression it enters the mean, $\mu_k = \exp(o_k + \mathbf{x}_k'\boldsymbol{\beta})$ with $o_k = \log\lambda_{\text{pop}}(\mathbf{s}_k^{*})$, but not the working response $z_k = \eta_k - o_k + (y_k - \mu_k)/\mu_k$, which is the same bookkeeping as in Section 9.4.2. The function `irls_poisson` already accepts an offset.

This is the point-pattern version of the expected count $E_i$ of Chapter 9. There, $E_i$ was the baseline risk integrated over region $i$; here the baseline enters at each point and the regions have shrunk away.

### 11.4.2 Reading the Coefficients

The factor $\exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$ is the relative risk at $\mathbf{s}$: the intensity of cases there compared with what the baseline predicts. (This is a ratio of intensities, like the case-to-control ratio of Chapter 10; Chapter 9's relative risk was a county's mean count divided by its expected count.) With an intercept $\beta_0$ and a covariate that vanishes far from the source, $\exp(\beta_0)$ is the background level, the number of cases per unit of baseline. The other coefficients describe the excess.

### 11.4.3 The Plant Story

A suspected industrial plant stands at $(0.3, 0.6)$. If it raises the risk of a disease, the excess should be largest at the plant and fade with distance. We model this with a Gaussian bump covariate with a reach $h$,

$$
\text{bump}_h(\mathbf{s}) = \exp\!\left(-\frac{\lVert\mathbf{s} - (0.3, 0.6)'\rVert^2}{2h^2}\right)
$$

so that the model is $\log\lambda = \log\lambda_{\text{pop}} + \beta_0 + \beta_1\text{bump}_h(\mathbf{s})$. In the simulation the cases have intensity $120\,\lambda_{\text{pop}}(\mathbf{s})\,\text{excess}(\mathbf{s})$ per year, where the population baseline is the density defined in Section 11.1 (`pop_density` in the code), and the excess (`excess_risk` in the code) is $\exp(\text{bump}_{0.1}(\mathbf{s}))$. The truth is therefore $\beta_0 = \log 120 = 4.787$ and $\beta_1 = 1$ with $h = 0.1$: the risk at the plant is $e^1 = 2.72$ times the baseline. The controls, a sample of the population, have intensity $800\,\lambda_{\text{pop}}(\mathbf{s})$.

Two data sets are simulated. The first is a single year of cases, and the second pools five years, which multiplies the case intensity by 5. The pooled data set has five times the information, and its intercept should move by $\log 5$ to $\log 600 = 6.397$.

For each data set the code fits the offset model with $h = 0.1$, taking the plume's reach as known from the plant's emissions report. It tests the excess against a model with the intercept only, using a likelihood ratio test. For the pooled data it then profiles the log-likelihood over $h$.

```python
def bump(P, h):
    """Gaussian bump of spread h around the plant at (0.3, 0.6)."""
    d2 = (P[:, 0] - 0.3)**2 + (P[:, 1] - 0.6)**2
    return np.exp(-d2 / (2 * h**2))


rate_ctrl = lambda P: 800 * pop_density(P)
rate_case = lambda P: 120 * pop_density(P) * excess_risk(P)
max_case = 1.05 * (120 * pop_density(G) * excess_risk(G)).max()
P_ctrl = simulate_poisson(rate_ctrl, 800.0, rng)
yearly = [simulate_poisson(rate_case, max_case, rng) for _ in range(5)]
P_case1, P_case5 = yearly[0], np.vstack(yearly)
print("controls:", len(P_ctrl), " cases in one year:", len(P_case1),
      " cases in five years:", len(P_case5))

log_pop = lambda P: np.log(pop_density(P))
cov_bump = lambda h: (lambda P: np.column_stack([np.ones(len(P)),
                                                 bump(P, h)]))
cov_one = lambda P: np.ones((len(P), 1))
print("truth: b0 = log(120 k) =", np.round(np.log([120, 600]), 3),
      " b1 = 1")
for name, Pc in (("one year", P_case1), ("five years", P_case5)):
    b_o, cov_o, ll_o, _ = fit_bt(Pc, 30, cov_bump(0.1), log_pop)
    ll_0 = fit_bt(Pc, 30, cov_one, log_pop)[2]
    lr = 2 * (ll_o - ll_0)
    print(f"{name:10s} b = {np.round(b_o, 3)}  SE = "
          f"{np.round(np.sqrt(np.diag(cov_o)), 3)}  "
          f"LR = {lr:6.2f}  p = {stats.chi2.sf(lr, 1):.4f}")

print("profile over the plume spread h (five years):")
print("    h   log-lik      b0     b1")
profile = {}
for h in (0.05, 0.075, 0.1, 0.15, 0.2, 0.3):
    profile[h] = fit_bt(P_case5, 30, cov_bump(h), log_pop)
    print(f"{h:5.3f} {profile[h][2]:9.2f} {profile[h][0][0]:7.3f} "
          f"{profile[h][0][1]:6.3f}")
```

The one-year data have 58 cases and the pooled data 296 cases, and there are 385 controls (363 expected). These are new draws, not the sample of Section 10.7.3, which had 46 cases and 349 controls in one year; here a second, five-year pooled data set is added because one year gives too few cases to say anything about the plant.

| Data | $\hat{\beta}_0$ | $\hat{\beta}_1$ | SE($\hat{\beta}_1$) | LR statistic | p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| One year (58 cases) | 4.815 | 0.573 | 0.716 | 0.58 | 0.448 |
| Five years (296 cases) | 6.406 | 1.053 | 0.276 | 12.08 | 0.0005 |

With one year of data the estimated excess is $0.573$ (a factor of 1.77 at the plant) with a standard error of $0.716$: the data cannot distinguish this from no excess, even though the simulated excess is real. Five years of data estimate the amplitude at $1.053$ (a factor of 2.87, against a true 2.72) with a standard error of $0.276$, and reject no excess firmly. The lesson is about power, not about the method: 58 cases is too few to see a bump of this size, and that is useful to know before spending money to look for one. The intercept shifts as predicted: $6.406 - 4.815 = 1.591$, close to $\log 5 = 1.609$.

The profile over $h$ is the second lesson. The log-likelihood is 1424.96 at $h = 0.05$, 1425.16 at $h = 0.075$, and 1424.97 at $h = 0.1$, and the amplitude estimate falls from $1.665$ to $1.053$ across that range. A narrow, tall bump and a wider, lower one explain the data about equally well. The reach of the plume cannot be read off these data, and any p-value that selects $h$ by maximizing the likelihood is optimistic, because the test statistic is no longer a $\chi^2$ with one degree of freedom. We fixed $h$ in advance to avoid this.

---

## 11.5 Case-Control Data and the Logistic Bridge

### 11.5.1 Pooling Cases and Controls

The offset model needs the baseline $\lambda_{\text{pop}}(\mathbf{s})$ at every location, which means a population map. Often there is none. What there is, instead, is a sample of **controls**: locations of people who did not have the disease, drawn so that they follow the population. If the controls follow the population, their density is the baseline, up to a constant, and we can use them in place of the map. Case-control data are of this form.

Pool the cases and the controls into a single coordinate matrix $\mathbf{S}$ with $n_1 + n_0$ rows, and attach a label $Y_i = 1$ for a case and $Y_i = 0$ for a control. The locations of the controls carry information about the baseline, and the labels carry information about whether cases and controls are spread differently.

### 11.5.2 Deriving the Bridge

Let the cases form a Poisson process with intensity $\lambda_1(\mathbf{s})$ and the controls an independent Poisson process with intensity $\lambda_0(\mathbf{s})$. Chapter 10 called the ratio $\operatorname{RR}(\mathbf{s}) = \lambda_1(\mathbf{s})/\lambda_0(\mathbf{s})$ the relative risk. Return to the pixel picture of Section 11.2.1. In a small pixel of area $c$ around $\mathbf{s}$, the probability of a case is $\lambda_1(\mathbf{s})c$, the probability of a control is $\lambda_0(\mathbf{s})c$, and the probability of both is of order $c^2$. Given that the pixel contains an event of either kind, the probability that it is a case is

$$
\Pr(Y = 1 \mid \text{event at }\mathbf{s}) = \frac{\lambda_1(\mathbf{s})c}{\lambda_1(\mathbf{s})c + \lambda_0(\mathbf{s})c} = \frac{\lambda_1(\mathbf{s})}{\lambda_1(\mathbf{s}) + \lambda_0(\mathbf{s})}
$$

The pixel size cancels. Take the log-odds:

$$
\operatorname{logit}\Pr(Y = 1 \mid \mathbf{s}) = \log\frac{\lambda_1(\mathbf{s})}{\lambda_0(\mathbf{s})} = \log\operatorname{RR}(\mathbf{s})
$$

**The log-odds that an event at $\mathbf{s}$ is a case is the log relative risk at $\mathbf{s}$.** Now suppose the cases have intensity $\lambda_1(\mathbf{s}) = \kappa_1\lambda_{\text{pop}}(\mathbf{s})\theta(\mathbf{s})$ and the controls $\lambda_0(\mathbf{s}) = \kappa_0\lambda_{\text{pop}}(\mathbf{s})$, where $\theta(\mathbf{s})$ is the excess risk and $\kappa_1$, $\kappa_0$ are constants that reflect how many cases and controls were collected. The unknown baseline cancels from the ratio:

$$
\log\operatorname{RR}(\mathbf{s}) = \log\frac{\kappa_1}{\kappa_0} + \log\theta(\mathbf{s})
$$

If we model the log relative risk as $\beta_0 + \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$, we have an ordinary logistic regression of the label on the covariates. We never need to estimate or even know $\lambda_{\text{pop}}$. The population density, the thing that made cases cluster downtown in the first place, drops out.

In our simulation, $\lambda_1/\lambda_0 = (120/800)\exp(\text{bump}_{0.1}(\mathbf{s}))$, so the true logit is $\log 0.15 + \text{bump}_{0.1}(\mathbf{s})$, with intercept $-1.897$ and slope 1.

The logistic likelihood is a conditional likelihood: it is the probability of the labels given the locations of all the events. The other factor of the full likelihood, the probability of the pooled locations, depends on $\lambda_1 + \lambda_0$ and carries little information about the relative risk once the baseline is unknown. Section 11.9.1 measures what it costs to throw that factor away.

### 11.5.3 Binomial IRLS, and the Sampling Ratio

The logistic model is a GLM with binomial responses and a logit link, and IRLS applies as in Section 9.1.5. With $p_i = 1/(1 + e^{-\eta_i})$, the working weights are $v_i = p_i(1 - p_i)$ and the working response is $z_i = \eta_i + (y_i - p_i)/(p_i(1 - p_i))$.

```python
def irls_logistic(X, y, tol=1e-10, maxit=50):
    """Logistic regression by IRLS: weights p(1 - p), working response
    z = eta + (y - p) / (p (1 - p))."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.mean() / (1 - y.mean()))
    for it in range(1, maxit + 1):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        v = p * (1 - p)
        z = eta + (y - p) / v
        info = X.T @ (v[:, None] * X)
        b_new = np.linalg.solve(info, X.T @ (v * z))
        done = np.max(np.abs(b_new - b)) < tol
        b = b_new
        if done:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return b, np.linalg.inv(info), ll, it


def pooled(P_cases, P_controls):
    P = np.vstack([P_cases, P_controls])
    return P, np.r_[np.ones(len(P_cases)), np.zeros(len(P_controls))]


print("truth: b0 = log(120 k / 800) =", np.round(np.log([120 / 800,
                                                         600 / 800]), 3))
for name, Pc in (("one year", P_case1), ("five years", P_case5)):
    P_all, label = pooled(Pc, P_ctrl)
    X_l = cov_bump(0.1)(P_all)
    b_l, cov_l, ll_l, _ = irls_logistic(X_l, label)
    ll_0 = irls_logistic(X_l[:, :1], label)[2]
    lr = 2 * (ll_l - ll_0)
    print(f"{name:10s} b = {np.round(b_l, 3)}  SE = "
          f"{np.round(np.sqrt(np.diag(cov_l)), 3)}  "
          f"LR = {lr:6.2f}  p = {stats.chi2.sf(lr, 1):.4f}")

P_all, label = pooled(P_case5, P_ctrl)
b_l5 = irls_logistic(cov_bump(0.1)(P_all), label)[0]
```

The intercept absorbs the sampling ratio $\kappa_1/\kappa_0$. Pooling five years of cases multiplies $\kappa_1$ by 5, so the true intercept moves from $\log 0.15 = -1.897$ to $\log 0.75 = -0.288$, while the slope stays at 1.

| Data | $\hat{\beta}_0$ | $\hat{\beta}_1$ | SE($\hat{\beta}_1$) | LR statistic | p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| One year | $-1.926$ | 0.523 | 0.783 | 0.42 | 0.519 |
| Five years | $-0.336$ | 1.012 | 0.428 | 5.78 | 0.016 |

The fitted intercept moves by $1.590$, against $\log 5 = 1.609$, and the slope barely changes: it is $1.012$ for the pooled data against a true $1$. This is why the intercept of a case-control model has no interpretation as a risk. Its value reflects how many controls were sampled per case, a decision made by the investigator. The slopes are the interpretable part.

Compare the standard error with the offset model of Section 11.4.3: $0.428$ for the logistic fit against $0.276$ for the offset fit with five years of cases. The logistic regression pays for not knowing the baseline.

### 11.5.4 Against the Kernel Map

Chapter 10 estimated the relative risk without a model, as a ratio of two kernel intensity estimates, $\hat{\lambda}_1(\mathbf{s})/\hat{\lambda}_0(\mathbf{s})$ with a shared bandwidth. The code does the same here, with bandwidth $0.08$ (its `kernel_intensity` has no edge correction and a different argument order from Chapter 10's, which divides by the kernel mass inside the window; the plant is more than three bandwidths from the border, so the difference does not matter here), for the pooled data, and compares it with two logistic fits: a cubic polynomial in the coordinates (an intercept and nine terms), which is a flexible smooth surface that does not use the plant's location, and the bump model, which does use the plant's location and reach.

```python
def kernel_intensity(P_data, grid, psi):
    """Gaussian kernel intensity estimate, no edge correction."""
    d2 = ((grid[:, None, :] - P_data[None, :, :])**2).sum(axis=2)
    return np.exp(-d2 / (2 * psi**2)).sum(axis=1) / (2 * np.pi * psi**2)


def cubic(P):
    """Intercept and all monomials x^i y^j with 1 <= i + j <= 3."""
    cols = [np.ones(len(P))]
    cols += [P[:, 0]**i * P[:, 1]**j for i in range(4) for j in range(4 - i)
             if 0 < i + j]
    return np.column_stack(cols)


G50 = grid_points(50)
log_rr_true = np.log(600 / 800) + bump(G50, 0.1)
log_rr_kernel = np.log(kernel_intensity(P_case5, G50, 0.08)
                       / kernel_intensity(P_ctrl, G50, 0.08))
b_cub = irls_logistic(cubic(P_all), label)[0]
log_rr_cubic = cubic(G50) @ b_cub
log_rr_bump = b_l5[0] + b_l5[1] * bump(G50, 0.1)
rmse = lambda e: np.sqrt(np.mean(e**2))
const = np.full(len(G50), log_rr_true.mean())
print("no spatial pattern (best constant): RMSE =",
      f"{rmse(const - log_rr_true):.3f}")
for name, f in (("kernel ratio", log_rr_kernel),
                ("cubic logistic", log_rr_cubic),
                ("bump logistic", log_rr_bump)):
    print(f"{name:15s} RMSE of log RR = {rmse(f - log_rr_true):.3f}  "
          f"correlation with truth = {np.corrcoef(f, log_rr_true)[0, 1]:.3f}")

for psi in (0.04, 0.08, 0.16):
    f = np.log(kernel_intensity(P_case5, G50, psi)
               / kernel_intensity(P_ctrl, G50, psi))
    print(f"kernel bandwidth {psi:4.2f}: RMSE = {rmse(f - log_rr_true):.3f}"
          f"  correlation = {np.corrcoef(f, log_rr_true)[0, 1]:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
panels = ((log_rr_true, "True log relative risk"),
          (log_rr_kernel, "Kernel ratio (bandwidth 0.08)"),
          (log_rr_cubic, "Logistic, cubic in location"),
          (log_rr_bump, "Logistic, plume bump"))
lims = (-1.0, 0.8)
for ax, (f, title) in zip(axes.ravel(), panels):
    im = ax.imshow(f.reshape(50, 50), origin="lower", extent=(0, 1, 0, 1),
                   cmap="viridis", vmin=lims[0], vmax=lims[1])
    ax.plot(0.3, 0.6, marker="+", color="white", ms=10)
    ax.set_title(title, fontsize=10)
fig.colorbar(im, ax=axes, extend="min",
             label="log relative risk (cases to controls)")
fig.savefig("figures/ch11/fig_11_2_case_control.png", dpi=120,
            bbox_inches="tight")
```

Figure 11.2 shows the true log relative risk and the three estimates. The color scale is limited to $[-1, 0.8]$ so the true bump is visible.

![Figure 11.2. The true log relative risk (top left), the ratio of kernel estimates (top right), a logistic fit with a cubic polynomial in the coordinates (bottom left), and the logistic fit with the plume bump (bottom right), all for five years of cases and the controls. The white cross marks the plant. Only the model that knows the shape finds it.](../figures/ch11/fig_11_2_case_control.png)

The result is sobering. A map with no spatial pattern at all, the constant equal to the average of the truth, has an RMSE of $0.166$ against the true log relative risk. The kernel ratio has an RMSE of $0.462$ and the cubic logistic fit $0.390$, both worse than the constant. Their correlations with the truth are low: $0.292$ and $0.259$. Changing the kernel bandwidth does not help: the correlation is $0.210$ at $0.04$, $0.292$ at $0.08$, and $0.252$ at $0.16$, while the RMSE falls with smoothing only because heavily smoothed maps approach the constant. With 296 cases a local ratio is mostly noise. The bump model has an RMSE of $0.048$ and, since its shape is the truth's shape, a correlation of $1.000$.

This is not an argument for the bump model over the kernel map. The bump model succeeds because we gave it the right shape and the right center. A bump placed in the wrong spot, or a true excess shaped like a ridge, would do badly (Exercise 17). The honest summary is that parametric structure buys power if it is right, and that the nonparametric map is the safer first look but needs far more data to find a localized excess.

---

## 11.6 Where Is the Excess? Spatial Scan Statistics

A regression like the one in Section 11.4 tests a specific alternative: excess near a known source. Outbreak surveillance asks a different question. Event locations arrive, and the team wants to know whether any small area has more events than usual, without saying in advance where. The **spatial scan statistic** answers by testing every area of a certain shape and size.

### 11.6.1 Windows and Expected Counts

A **window** is a disc of radius $r$ around a center. The scan considers many centers on a grid and several radii, a few thousand windows in all. For each window, we compare the number of events inside, $n_w$, with the number $e_w$ we would expect inside if the events followed a known baseline. If $n$ events were observed in total and $\lambda_{\text{bg}}(\mathbf{s})$ is the baseline intensity, then

$$
e_w = n\,\frac{\int_w\lambda_{\text{bg}}(\mathbf{s})\,d\mathbf{s}}{\int_{\mathcal{D}}\lambda_{\text{bg}}(\mathbf{s})\,d\mathbf{s}}
$$

This conditions on the total count: the baseline determines only the shape of the intensity, and the scale comes from the data.

### 11.6.2 The Log-Likelihood Ratio

Condition on $n$. Under the baseline, each event independently falls inside window $w$ with probability $p_w = e_w/n$. Under the alternative that the window holds an excess, each event falls inside with some larger probability $q$. Within the window and outside it, the distribution of the locations is still the baseline's shape, so the shape terms cancel in the ratio of likelihoods, and

$$
\frac{L_1}{L_0} = \frac{q^{n_w}(1 - q)^{n - n_w}}{p_w^{n_w}(1 - p_w)^{n - n_w}}
$$

The maximum over $q$ is at $\hat{q} = n_w/n$, as for any binomial proportion. Substituting, and using $p_w = e_w/n$, gives the log-likelihood ratio of the window:

$$
\operatorname{LLR}_w = n_w\log\frac{n_w}{e_w} + (n - n_w)\log\frac{n - n_w}{n - e_w}, \qquad \text{kept only if } n_w > e_w
$$

and set to zero otherwise, because we look only for excesses. This is the same expression one obtains from a Poisson model with the rate multiplied by a constant inside the window and a different constant outside: the $-\text{mean}$ terms of the Poisson log-likelihood cancel because both fits have the same total.

A small check: with $n = 100$ events, a window that was expected to hold $e_w = 10$ and holds $n_w = 20$ has $\operatorname{LLR} = 20\log 2 + 80\log(80/90) = 13.863 - 9.423 = 4.440$. (Papers on scan statistics usually report $\operatorname{LLR}$ itself. A likelihood ratio statistic compared with a $\chi^2$ distribution is twice this value, so we compare $2\operatorname{LLR}$ with $\chi^2$ quantiles.)

### 11.6.3 Computing All Windows at Once

The statistic has two logarithms of counts that can be zero. When a window holds no events, $n_w\log(n_w/e_w)$ is $0\log 0$, which should be zero and which NumPy reports as `nan`. The function `scipy.special.xlogy(x, y)` computes $x\log y$ and returns 0 when $x = 0$. With it, the whole table of statistics for every center and radius is one vectorized expression on a table of counts.

The outbreak story: a surveillance system sees $n = 350$ events over a district. The background intensity is proportional to population density, $600\,\lambda_{\text{pop}}$. The simulated data have 315 background events plus 35 extra events in a tight Gaussian cluster, spread $0.04$, centered at $(0.25, 0.2)$, a sparsely populated corner. The scan uses 441 centers on a $21 \times 21$ grid and 7 radii from $0.05$ to $0.3$: 3,087 windows. The expected counts integrate the baseline over each window on a fine grid. The code, which appears in the next subsection together with the p-value computation, scans all windows and reports the most likely cluster.

### 11.6.4 Inference: Why the Maximum Needs a Simulation

For one window fixed in advance, the one-sided LLR has a known large-sample distribution: $2\operatorname{LLR}$ is zero when $n_w \le e_w$, which happens about half the time, and otherwise follows a $\chi^2$ distribution with one degree of freedom. So a single window is a boundary test like those of Section 5.5.2, and one rejects at 5% when $2\operatorname{LLR}$ exceeds the $\chi^2_1$ quantile $2.71$ (the 0.90 quantile, because only half of the mass is positive).

The scan does not test one window. It tests 3,087 overlapping windows and reports the largest statistic. The maximum of many statistics is larger than a typical one, exactly as the largest of many $t$ statistics is, and the windows overlap, so the statistics are strongly dependent and no tidy reference distribution exists. The reference distribution has to be simulated: generate data sets from the baseline with the same $n$, compute the largest $2\operatorname{LLR}$ of each, and see where the observed maximum falls. The Monte Carlo p-value is $(1 + \#\{\max_{\text{sim}} \ge \max_{\text{obs}}\})/(B + 1)$, which is valid for the reasons given in Section 6.8.1.

The code simulates $B = 999$ data sets from the baseline.

```python
def llr_table(n_w, e_w, n):
    """Log-likelihood ratio for elevated-risk windows (zero otherwise)."""
    out = (special.xlogy(n_w, n_w / e_w)
           + special.xlogy(n - n_w, (n - n_w) / (n - e_w)))
    return np.where(n_w > e_w, out, 0.0)


print("hand check, n_w = 20, e_w = 10, n = 100:",
      round(float(llr_table(20.0, 10.0, 100.0)), 4))

cg = np.linspace(0.0, 1.0, 21)
centers = np.column_stack([a.ravel() for a in np.meshgrid(cg, cg)])
radii = np.array([0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])
F = grid_points(100)
lam_bg = lambda P: 600 * pop_density(P)
d2_F = ((centers[:, None, :] - F[None, :, :])**2).sum(axis=2)
mass = np.stack([((d2_F <= r**2) * lam_bg(F)).sum(axis=1) / len(F)
                 for r in radii], axis=1)
mass_total = lam_bg(F).mean()


def scan(P):
    """LLR of every (center, radius) window; expected counts use the
    baseline mass and the total number of events n."""
    n = len(P)
    d2 = ((centers[:, None, :] - P[None, :, :])**2).sum(axis=2)
    n_w = np.stack([(d2 <= r**2).sum(axis=1) for r in radii], axis=1)
    e_w = n * mass / mass_total
    return llr_table(n_w.astype(float), e_w, n), n_w, e_w


def sample_baseline(n, rng):
    """n events from the baseline density, proportional to pop_density."""
    out = np.empty((0, 2))
    while len(out) < n:
        C = rng.uniform(size=(2 * n, 2))
        out = np.vstack([out, C[rng.uniform(size=2 * n) < pop_density(C)]])
    return out[:n]


def inject(n_bg, n_extra, rng):
    """Baseline events plus a Gaussian cluster of extra events."""
    P = sample_baseline(n_bg, rng)
    if n_extra:
        extra = rng.normal([0.25, 0.2], 0.04, (n_extra, 2))
        P = np.vstack([P, np.clip(extra, 0, 1)])
    return P


P_out = inject(315, 35, rng)
llr, n_w, e_w = scan(P_out)
i, j = np.unravel_index(llr.argmax(), llr.shape)
print("events:", len(P_out), " windows scanned:", llr.size)
print("best window: center", centers[i], "radius", radii[j],
      " n_w =", n_w[i, j], " e_w =", np.round(e_w[i, j], 2),
      " LLR =", np.round(llr[i, j], 2))

B = 999
null_max = np.empty(B)
null_one = np.empty(B)
mid = np.argmin(np.abs(centers - 0.5).sum(axis=1))
for k in range(B):
    lk = scan(sample_baseline(len(P_out), rng))[0]
    null_max[k] = lk.max()
    null_one[k] = lk[mid, 3]
p_mc = (1 + np.sum(null_max >= llr.max())) / (B + 1)
print("Monte Carlo p-value:", p_mc)
print("2 * LLR, maximum over windows: median",
      np.round(2 * np.median(null_max), 2), " 95th percentile",
      np.round(2 * np.quantile(null_max, 0.95), 2))
print("2 * LLR, one fixed window:     median",
      np.round(2 * np.median(null_one), 2), " 95th percentile",
      np.round(2 * np.quantile(null_one, 0.95), 2))
print("fixed window: e_w =", np.round(scan(P_out)[2][mid, 3], 1),
      " share of 2 * LLR above 3.84:",
      np.round(np.mean(2 * null_one > 3.84), 3))
print("share of data sets whose maximum 2 * LLR exceeds 3.84:",
      np.round(np.mean(2 * null_max > 3.84), 3))
print("chi-square(1) quantiles: median", np.round(stats.chi2.ppf(0.5, 1), 2),
      " 95th percentile", np.round(stats.chi2.ppf(0.95, 1), 2))
print("naive chi-square p-value of the observed maximum:",
      f"{stats.chi2.sf(2 * llr.max(), 1):.1e}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(P_out[:, 0], P_out[:, 1], s=6, color="black")
axes[0].add_patch(plt.Circle(centers[i], radii[j], fill=False,
                             color="tab:red", lw=2))
axes[0].plot(0.25, 0.2, marker="+", color="tab:blue", ms=12)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_aspect("equal")
axes[0].set_title("Events and the most likely cluster")
axes[1].hist(2 * null_max, bins=30, color="lightgray", edgecolor="gray")
axes[1].axvline(stats.chi2.ppf(0.95, 1), color="tab:blue", ls="--",
                label="chi-square(1), 95%")
axes[1].set_xlim(0, 30)
axes[1].annotate(f"observed: {2 * llr.max():.0f}\n(off the scale)",
                 xy=(29.5, 5), ha="right", color="tab:red")
axes[1].set_xlabel("2 * largest LLR over all windows, no cluster present")
axes[1].set_ylabel("simulated data sets")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/ch11/fig_11_3_scan.png", dpi=120)
```

The most likely cluster is centered at $(0.25, 0.2)$, the true center, with radius $0.05$: it holds $n_w = 31$ events where $e_w = 1.87$ were expected, with $\operatorname{LLR} = 59.17$ ($2\operatorname{LLR} = 118.3$). None of the 999 simulated data sets has a maximum that large, so the Monte Carlo p-value is $0.001$, the smallest value 999 simulations can give. A $\chi^2_1$ p-value for the same statistic would be $1.5 \times 10^{-27}$, which is meaningless.

Under the baseline, the maximum of $2\operatorname{LLR}$ over all windows has median $9.77$ and 95th percentile $15.0$. A $\chi^2_1$ variable has median $0.45$ and 95th percentile $3.84$. The reference distribution of the maximum sits far above the one for a single window. For one fixed window near the center of the map, with $e_w = 30.7$ expected events, the share of simulations with $2\operatorname{LLR} > 3.84$ is $0.013$. That is below the $0.025$ the boundary theory predicts, because conditioning on $n$ and the discreteness of counts make the single-window test a little conservative, but it is in the right region, and nowhere near the behavior of the maximum. Using the single-window critical value for the scan's maximum would flag every data set: in all 999 simulated data sets with no cluster, the largest $2\operatorname{LLR}$ exceeds $3.84$.

Figure 11.3 shows the events with the most likely cluster circled and the null distribution of the maximum.

![Figure 11.3. Left: 350 events, with the most likely cluster circled in red and the true cluster center marked in blue. Right: the distribution of the largest 2 LLR over all windows in 999 data sets with no cluster. The chi-square 95% point for one window (dashed) is far to the left of this distribution, and the observed value of 118 is off the scale.](../figures/ch11/fig_11_3_scan.png)

### 11.6.5 Cautions

The scan tests departures from the baseline, and its power depends on how good the baseline is. If the baseline leaves out something real, such as a hospital that makes nearby cases more likely to be reported, the scan finds it and reports a cluster. If the baseline is wrong because the population map is outdated, the same thing happens (Exercise 18). The most likely cluster is a hypothesis to investigate, not a finding. Section 11.9.2 measures the size and power of the scan by repeated simulation.

---

## 11.7 Separating Trend From Interaction

### 11.7.1 Why the Ordinary $K$-Function Reports Clustering

In Section 3.4 we saw a trend in the mean produce a variogram that looked like spatial dependence. Point patterns have the same trap. Ripley's $K$-function, from Chapter 10, measures interaction by counting extra events near a typical event. The estimator reads: the expected number of other events within distance $r$ of a typical event, divided by the intensity. It assumes that the intensity is the same everywhere. If it is not, then a typical event is more likely to be in a dense area, where its neighbors are also more numerous, and $K$ mistakes this for attraction.

To see how large the effect is, take an inhomogeneous Poisson process with no interaction at all. Pairs of events at locations $\mathbf{s}$ and $\mathbf{t}$ occur at rate $\lambda(\mathbf{s})\lambda(\mathbf{t})$. The ordinary estimator divides by the square of the average intensity $\bar{\lambda}$ and counts all pairs closer than $r$, so its expected value is roughly

$$
K_{\text{ord}}(r) \approx \frac{1}{\bar{\lambda}^2\operatorname{area}(\mathcal{D})}\int\!\!\int_{\lVert\mathbf{s} - \mathbf{t}\rVert \le r}\lambda(\mathbf{s})\lambda(\mathbf{t})\,d\mathbf{s}\,d\mathbf{t} \approx \pi r^2\,\frac{\operatorname{E}[\lambda^2]}{\bar{\lambda}^2} = \pi r^2\left(1 + \operatorname{CV}^2\right)
$$

for small $r$. The middle expression replaces $\lambda(\mathbf{t})$ by $\lambda(\mathbf{s})$ inside a small disc, and $\operatorname{CV}^2 = \operatorname{Var}(\lambda)/\bar{\lambda}^2$ is the squared coefficient of variation of the intensity over the map, with the average taken over the region. Under no interaction, the ordinary $K$ is inflated by $1 + \operatorname{CV}^2$, and this has nothing to do with the events attracting each other. It is only the variation in the intensity.

For the crash intensity, $\operatorname{CV}^2 = 0.433$. Then $K_{\text{ord}} \approx 1.433\pi r^2$, and $L(r) - r = \sqrt{K/\pi} - r$ should be about $(\sqrt{1.433} - 1)r = 0.197\,r$ for small $r$, positive and growing in proportion to $r$. Section 11.7.3 tests the prediction.

### 11.7.2 The Inhomogeneous $K$-Function

The remedy is to weight each pair by the reciprocal of the intensities at its two events:

$$
\hat{K}_{\text{inhom}}(r) = \frac{1}{\operatorname{area}(\mathcal{D})}\sum_{i=1}^n\sum_{j \ne i}\frac{\omega_{ij}\,\mathbb{1}(h_{ij} \le r)}{\hat{\lambda}(\mathbf{s}_i)\hat{\lambda}(\mathbf{s}_j)}
$$

where $h_{ij}$ is the distance between events $i$ and $j$ and $\omega_{ij} \ge 1$ is an edge-correction weight. A pair in a dense area is worth less, because pairs there are expected to be frequent. If the intensity were constant and equal to $n/\operatorname{area}(\mathcal{D})$, this would reduce to an ordinary $K$ estimator (with $n^2$ in the denominator in place of $n(n-1)$).

Why is it right? For a Poisson process, the expected sum over ordered pairs of any function $f(\mathbf{s}_i, \mathbf{s}_j)$ is $\int\!\!\int f(\mathbf{s}, \mathbf{t})\lambda(\mathbf{s})\lambda(\mathbf{t})\,d\mathbf{s}\,d\mathbf{t}$, because pairs occur at rate $\lambda(\mathbf{s})\lambda(\mathbf{t})$. Take $f = \omega\,\mathbb{1}(\lVert\mathbf{s} - \mathbf{t}\rVert \le r)/(\lambda(\mathbf{s})\lambda(\mathbf{t}))$. The intensities cancel, leaving $\int\!\!\int\omega\,\mathbb{1}(\cdot)\,d\mathbf{s}\,d\mathbf{t}$, which the edge weights make equal to $\operatorname{area}(\mathcal{D})\,\pi r^2$. So $\operatorname{E}[\hat{K}_{\text{inhom}}(r)] = \pi r^2$ whatever the intensity, as long as there is no interaction. Departures of $\hat{K}_{\text{inhom}}$ from $\pi r^2$ are then evidence of interaction.

**Edge weights.** Chapter 10 corrected edges with Ripley's weights, built from the length of a circle that lies inside the region. (The estimator below also normalizes differently: it divides each pair by the product of the intensities, so with a constant intensity it is smaller than Chapter 10's $\hat{K}$ by the factor $(n - 1)/n$, in addition to the difference in edge weights, and values at small $r$ differ slightly.) For a rectangle there is a simpler correct choice, the **translation correction**: $\omega_{ij} = \operatorname{area}(\mathcal{D})/\operatorname{area}(\mathcal{D} \cap (\mathcal{D} + \mathbf{h}_{ij}))$, where $\mathcal{D} + \mathbf{h}$ is the region shifted by the vector $\mathbf{h}_{ij}$ from event $j$ to event $i$. For the unit square the overlap of the square with its shifted copy is $(1 - \lvert\Delta x\rvert)(1 - \lvert\Delta y\rvert)$, where $\Delta x$ and $\Delta y$ are the coordinate differences, so $\omega_{ij} = 1/((1 - \lvert\Delta x\rvert)(1 - \lvert\Delta y\rvert))$. Why does this make the expectation come out right? The integral $\int\!\!\int_{\mathcal{D} \times \mathcal{D}}\omega(\mathbf{s}, \mathbf{t})\mathbb{1}(\lVert\mathbf{s} - \mathbf{t}\rVert \le r)\,d\mathbf{s}\,d\mathbf{t}$ can be rewritten by setting $\mathbf{h} = \mathbf{t} - \mathbf{s}$. For fixed $\mathbf{h}$, the pairs $(\mathbf{s}, \mathbf{s} + \mathbf{h})$ with both points in $\mathcal{D}$ sweep out exactly $\operatorname{area}(\mathcal{D} \cap (\mathcal{D} - \mathbf{h}))$ of starting points, the weight $\omega$ is the reciprocal of this overlap times $\operatorname{area}(\mathcal{D})$, and the product is $\operatorname{area}(\mathcal{D})$ for every $\mathbf{h}$ in the disc. Integrating over the disc of radius $r$ gives $\operatorname{area}(\mathcal{D})\,\pi r^2$. Both edge corrections work; this one is easier to code.

### 11.7.3 In Practice

The estimator needs $\hat{\lambda}(\mathbf{s}_i)$ at the events. For the crash pattern we use the fitted GLM intensity of Section 11.3.6. We compare $L(r) - r$ for the ordinary and the inhomogeneous estimators against envelopes: for the ordinary $K$, the envelope comes from 99 patterns with complete spatial randomness and the same $n$; for the inhomogeneous $K$, from 99 patterns simulated from the fitted intensity, with the intensity re-fitted to each simulated pattern, so the envelope includes the effect of estimating $\hat{\lambda}$.

```python
def dist(S):
    """Matrix of Euclidean distances between the rows of S."""
    q = np.sum(S**2, axis=1)
    H = np.sqrt(np.maximum(q[:, None] - 2 * S @ S.T + q[None, :], 0.0))
    np.fill_diagonal(H, 0.0)
    return H


def k_hat(P, r_grid, lam):
    """Inhomogeneous K on the unit square with translation edge weights
    omega_ij = 1 / ((1 - |dx|)(1 - |dy|)). A constant lam = n gives the
    ordinary K."""
    H = dist(P)
    dx = np.abs(P[:, 0][:, None] - P[:, 0][None, :])
    dy = np.abs(P[:, 1][:, None] - P[:, 1][None, :])
    W = 1.0 / ((1 - dx) * (1 - dy)) / np.outer(lam, lam)
    np.fill_diagonal(W, 0.0)
    return np.array([W[H <= r].sum() for r in r_grid])


l_minus_r = lambda K, r: np.sqrt(K / np.pi) - r

r_grid = np.linspace(0.01, 0.15, 15)
n = len(P_crash)
lam_fit = np.exp(design(P_crash) @ b_hat)
lm_ord = l_minus_r(k_hat(P_crash, r_grid, np.full(n, float(n))), r_grid)
lm_inh = l_minus_r(k_hat(P_crash, r_grid, lam_fit), r_grid)

env_ord, env_inh = [], []
for s in range(99):
    Pu = rng.uniform(size=(n, 2))
    env_ord.append(l_minus_r(k_hat(Pu, r_grid, np.full(n, float(n))),
                             r_grid))
    r_fit = lambda P: np.exp(design(P) @ b_hat)
    Ps = simulate_poisson(r_fit, 1.05 * r_fit(G).max(), rng)
    bs = fit_bt(Ps, 20)[0]
    env_inh.append(l_minus_r(k_hat(Ps, r_grid,
                                   np.exp(design(Ps) @ bs)), r_grid))
env_ord, env_inh = np.array(env_ord), np.array(env_inh)
print("   r   ordinary L - r (CSR band)       inhomogeneous L - r (band)")
for q in (1, 3, 5, 7, 9, 14):
    print(f"{r_grid[q]:5.2f} {lm_ord[q]:8.4f} "
          f"({env_ord[:, q].min():7.4f}, {env_ord[:, q].max():7.4f})   "
          f"{lm_inh[q]:8.4f} "
          f"({env_inh[:, q].min():7.4f}, {env_inh[:, q].max():7.4f})")

cv2 = np.mean(rate(G)**2) / np.mean(rate(G))**2 - 1
print("squared coefficient of variation of the intensity:", round(cv2, 3))
print("predicted (L - r) / r for small r:", round(np.sqrt(1 + cv2) - 1, 3))
print("observed (L - r) / r at r = 0.04, 0.08:",
      np.round(lm_ord[[3, 7]] / r_grid[[3, 7]], 3))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, obs, env, title in ((axes[0], lm_ord, env_ord, "Ordinary K"),
                            (axes[1], lm_inh, env_inh,
                             "Inhomogeneous K, fitted intensity")):
    ax.fill_between(r_grid, env.min(axis=0), env.max(axis=0),
                    color="lightgray", label="envelope, 99 simulations")
    ax.plot(r_grid, obs, color="tab:red", lw=2, label="crash pattern")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("distance r")
    ax.set_title(title)
axes[0].set_ylabel("L(r) - r")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/ch11/fig_11_4_trend_interaction.png", dpi=120)
```

| $r$ | Ordinary $L - r$ | CSR envelope | Inhomogeneous $L - r$ | Envelope |
| :--- | :--- | :--- | :--- | :--- |
| 0.02 | 0.0040 | $(-0.0025, 0.0026)$ | 0.0005 | $(-0.0046, 0.0041)$ |
| 0.06 | 0.0135 | $(-0.0019, 0.0023)$ | $-0.0004$ | $(-0.0038, 0.0046)$ |
| 0.10 | 0.0231 | $(-0.0032, 0.0049)$ | 0.0022 | $(-0.0045, 0.0058)$ |
| 0.15 | 0.0336 | $(-0.0047, 0.0066)$ | 0.0040 | $(-0.0062, 0.0056)$ |

The ordinary $K$ is far outside the envelope at every distance: it reports clustering. The inhomogeneous $K$ is inside the envelope at every distance: it reports nothing. The pattern was generated with no interaction, so the second answer is the right one. The prediction of Section 11.7.1 is close: $(L - r)/r$ is $0.206$ at $r = 0.04$ and $0.232$ at $r = 0.08$, against the predicted $0.197$.

![Figure 11.4. Left: L(r) - r for the crash pattern, estimated with the ordinary K, against an envelope from complete spatial randomness. Right: the same pattern with the inhomogeneous K and the fitted intensity, against an envelope from the fitted model. The pattern has a trend and no interaction.](../figures/ch11/fig_11_4_trend_interaction.png)

**Estimating the intensity and the interaction from the same data.** The inhomogeneous $K$ uses $\hat{\lambda}$ from the same events whose interaction it assesses. A flexible $\hat{\lambda}$ can absorb the clusters it is supposed to detect: an estimate that follows every bump of the pattern will explain interaction away. How much this matters depends on the flexibility. Section 11.9.3 compares estimates in 100 Poisson patterns. The estimator is nearly unbiased with a GLM fit of three parameters, and biased downward, sometimes severely, when the intensity is a kernel estimate that includes the event itself.

---

## 11.8 Inference Under Clustering

### 11.8.1 What Clustering Does to Counts

The Poisson GLM of Section 11.3 assumes that counts in disjoint regions are independent. Crashes violate this when they cluster at dangerous intersections: one bad intersection produces many crashes in a small area. We model this with a **Thomas process**, a standard cluster model, in its inhomogeneous form:

1. Parents form a homogeneous Poisson process with intensity $\kappa$ (parents per unit area).
2. Each parent has a Poisson number of offspring with mean $\mu_c$, placed around it with independent Gaussian displacements of standard deviation $\sigma_c$ in each coordinate.
3. Every offspring is kept with probability $\lambda(\mathbf{s})/(\kappa\mu_c)$, so that the events we observe have intensity $\lambda(\mathbf{s})$. This requires $\kappa\mu_c \ge \max\lambda$.

Chapter 12 develops this model, with a tissue-slide story. Here the clusters are the dangerous intersections, and the thinning in step 3 gives them the trend of the crash intensity.

We take $\kappa = 100$ and $\sigma_c = 0.03$, with the crash intensity of Section 11.1.2 as the trend. (This is the same family as the clustered pattern of Section 10.3.4, with its own parameters: 20 parents per unit area there, 100 here, and here an inhomogeneous trend.) Each parent has about $\mu_c = 14$ offspring before thinning, the expected number of events is again about 400, and the counts are more variable than Poisson counts. The function below simulates the process. Parents are placed on a window enlarged by $4\sigma_c$ on every side, so that offspring of parents just outside the square can still land inside it.

```python
def simulate_cluster(rate_fn, rate_max, kappa, sigma_c, rng):
    """Inhomogeneous Thomas process on the unit square, by thinning.
    Parents have intensity kappa on an enlarged window; each has
    Poisson(rate_max / kappa) offspring, Gaussian with spread sigma_c;
    offspring inside the square are kept with probability rate / rate_max."""
    margin = 4 * sigma_c
    lo, hi = -margin, 1 + margin
    n_par = rng.poisson(kappa * (hi - lo)**2)
    parents = rng.uniform(lo, hi, size=(n_par, 2))
    counts = rng.poisson(rate_max / kappa, size=n_par)
    kids = np.repeat(parents, counts, axis=0)
    kids = kids + sigma_c * rng.standard_normal(kids.shape)
    kids = kids[np.all((kids >= 0) & (kids <= 1), axis=1)]
    return kids[rng.uniform(size=len(kids)) < rate_fn(kids) / rate_max]


kappa, sigma_c = 100.0, 0.03
rate_max_c = 1.05 * rate(grid_points(300) * 1.24 - 0.12).max()
sim_p = lambda: simulate_poisson(rate, rate_max, rng)
sim_c = lambda: simulate_cluster(rate, rate_max_c, kappa, sigma_c, rng)
print("offspring per parent before thinning:",
      round(rate_max_c / kappa, 1))
```

**The variance of a sum over events.** Chapter 10 defined the pair correlation function $g(h)$: the expected number of ordered pairs of events in small areas $d\mathbf{s}$ and $d\mathbf{t}$, at distance $h$ apart, is $\lambda(\mathbf{s})\lambda(\mathbf{t})g(h)\,d\mathbf{s}\,d\mathbf{t}$. For a Poisson process $g \equiv 1$, and $g > 1$ at distance $h$ means pairs at that distance are more common than chance. (With a constant intensity this is the $g$ of Chapter 10; here the intensity may vary, and $g$ compares pairs with what independence would give at the local intensities.) For any function $f$, consider the sum $T = \sum_if(\mathbf{s}_i)$ over the events. Its mean is $\int f\lambda$. Its second moment separates into the terms with $i = j$ and with $i \ne j$:

$$
\operatorname{E}[T^2] = \operatorname{E}\sum_if(\mathbf{s}_i)^2 + \operatorname{E}\sum_{i \ne j}f(\mathbf{s}_i)f(\mathbf{s}_j) = \int f^2\lambda + \int\!\!\int f(\mathbf{s})f(\mathbf{t})\lambda(\mathbf{s})\lambda(\mathbf{t})g\,d\mathbf{s}\,d\mathbf{t}
$$

Subtracting the square of the mean,

$$
\operatorname{Var}(T) = \int f^2\lambda + \int\!\!\int f(\mathbf{s})f(\mathbf{t})\lambda(\mathbf{s})\lambda(\mathbf{t})\left(g - 1\right)d\mathbf{s}\,d\mathbf{t}
$$

The first term is the Poisson variance. The second is the effect of interaction: it is positive when $g > 1$ (clustering) and negative when $g < 1$ (inhibition). With $f \equiv 1$, $T = N(\mathcal{D})$ and the second term is the extra variance of the count.

For the Thomas process,

$$
g(h) = 1 + \frac{\exp\!\left(-h^2/(4\sigma_c^2)\right)}{4\pi\kappa\sigma_c^2}
$$

The next subsection derives this, and it can be skipped. The function starts at $1 + 1/(4\pi\kappa\sigma_c^2) = 1.88$ at $h = 0$, so pairs of events very close together are almost twice as common as chance, and falls to 1 over a distance of about $2\sigma_c$. Because $\int(g - 1)\,d\mathbf{h} = 1/\kappa$, the count in a region much larger than a cluster has variance $\int\lambda + (1/\kappa)\int\lambda^2$, a variance-to-mean ratio of $1 + \int\lambda^2/(\kappa\int\lambda)$. The code computes this ratio, and a second version that respects the boundary of the square (a pair with one event outside the square is not counted), and compares both with the variance-to-mean ratio of the number of events in 300 simulated clustered patterns. The boundary version is a double sum over grid cells, which separates into a product over the two coordinates and so costs very little.

```python
def pair_integral(A, sigma, m=60):
    """c^2 * sum_ij A_ip A_jq phi(s_i - s_j) over the m x m grid, where
    phi(h) = exp(-h^2 / (4 sigma^2)) / (4 pi sigma^2). Separable in x, y."""
    g = (np.arange(m) + 0.5) / m
    K1 = np.exp(-(g[:, None] - g[None, :])**2 / (4 * sigma**2))
    K1 = K1 / np.sqrt(4 * np.pi * sigma**2)
    A3 = A.reshape(m, m, -1)
    out = np.empty((A3.shape[2], A3.shape[2]))
    for p in range(A3.shape[2]):
        T = K1 @ A3[:, :, p] @ K1.T
        for q in range(A3.shape[2]):
            out[p, q] = np.sum(A3[:, :, q] * T) / m**4
    return out


sizes = np.array([len(sim_c()) for _ in range(300)])
lam60 = rate(grid_points(60))
mean_n = np.mean(lam60)
exact_var = mean_n + pair_integral(lam60[:, None], sigma_c)[0, 0] / kappa
print("clustered counts: mean", round(sizes.mean(), 1), " variance",
      round(sizes.var(ddof=1), 1), " variance / mean",
      round(sizes.var(ddof=1) / sizes.mean(), 2))
print("formula, ignoring the boundary:",
      round(1 + np.mean(rate(G)**2) / np.mean(rate(G)) / kappa, 2),
      " formula with the boundary:", round(exact_var / mean_n, 2))
```

The simulated variance-to-mean ratio of the count is $6.58$, against $6.73$ for the formula that ignores the boundary and $6.51$ for the formula that respects it. A Poisson process has a ratio of 1. The count of crashes in the district has more than six times the Poisson variance, so the effective sample size for the overall rate is about a sixth of $n$. This is the same effect as the effective sample size of Chapter 1.

Figure 11.5 compares a Poisson pattern and a clustered pattern with the same trend and about the same number of events. The clustered pattern looks lumpy.

```python
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, sim, title in ((axes[0], sim_p, "Poisson"),
                       (axes[1], sim_c, "Clustered")):
    P = sim()
    ax.scatter(P[:, 0], P[:, 1], s=6, color="black")
    ax.set_title(f"{title}: {len(P)} crashes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("figures/ch11/fig_11_5_patterns.png", dpi=120)
```

![Figure 11.5. A Poisson pattern (left, 419 crashes) and a clustered pattern (right, 438 crashes) with the same trend. The clustered pattern has tight groups of events, which a Poisson process with the same intensity does not.](../figures/ch11/fig_11_5_patterns.png)

### 11.8.2 The Pair Correlation of the Thomas Process (Optional)

*This subsection derives the formula for $g$ used above. It can be skipped on a first reading.*

Two offspring of the same parent are displaced from it by independent $N(\mathbf{0}, \sigma_c^2\mathbf{I})$ vectors, so the difference between their locations is $N(\mathbf{0}, 2\sigma_c^2\mathbf{I})$, with density $\mathcal{N}_c(\mathbf{h}) = \exp(-\lVert\mathbf{h}\rVert^2/(4\sigma_c^2))/(4\pi\sigma_c^2)$. Before thinning, a parent has $M$ offspring (here $M$ is the number of offspring of one parent, not the pixel count of Section 11.2), where $M$ is Poisson with mean $\mu_c$, so the number of ordered pairs of offspring of one parent is $M(M - 1)$, with mean $\operatorname{E}[M(M-1)] = \mu_c^2$. The parents occur at rate $\kappa$ per unit area, so the expected number of ordered pairs from a common parent, per unit area and per unit of displacement density, is $\kappa\mu_c^2\mathcal{N}_c(\mathbf{h})$. Pairs with offspring of different parents occur at rate $(\kappa\mu_c)^2$, as in a Poisson process, because the parents are independent. Therefore the total pair rate is $(\kappa\mu_c)^2 + \kappa\mu_c^2\mathcal{N}_c(\mathbf{h})$, and dividing by the squared intensity $(\kappa\mu_c)^2$,

$$
g(h) = 1 + \frac{\kappa\mu_c^2\mathcal{N}_c}{(\kappa\mu_c)^2} = 1 + \frac{\mathcal{N}_c}{\kappa}
$$

Independent thinning with retention probability $p(\mathbf{s})$ multiplies the intensity at $\mathbf{s}$ by $p(\mathbf{s})$ and the pair rate at $(\mathbf{s}, \mathbf{t})$ by $p(\mathbf{s})p(\mathbf{t})$, so it leaves $g$ unchanged. This is why the thinned process has a pair correlation that does not depend on location, and why the trend $\lambda(\mathbf{s})$ and the clustering can be modeled separately.

### 11.8.3 The Sandwich for a Point Process

The same calculation gives the variance of $\hat{\boldsymbol{\beta}}$. The estimating equation is $\mathbf{U}(\hat{\boldsymbol{\beta}}) = \mathbf{0}$ with the score $\mathbf{U}(\boldsymbol{\beta}) = \sum_i\mathbf{x}(\mathbf{s}_i) - \int\mathbf{x}\lambda$ of Section 11.2.2. Expand around the true $\boldsymbol{\beta}$. The derivative of the score is $-\mathcal{I}(\boldsymbol{\beta})$ and does not involve the events, so to first order

$$
\mathbf{0} = \mathbf{U}(\hat{\boldsymbol{\beta}}) \approx \mathbf{U}(\boldsymbol{\beta}) - \mathcal{I}(\boldsymbol{\beta})(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}) \qquad \Longrightarrow \qquad \hat{\boldsymbol{\beta}} - \boldsymbol{\beta} \approx \mathcal{I}^{-1}\mathbf{U}(\boldsymbol{\beta})
$$

The score has mean zero, and applying the variance formula with $f = \mathbf{a}'\mathbf{x}$ for each vector $\mathbf{a}$ gives its covariance matrix:

$$
\operatorname{Var}(\mathbf{U}) = \mathcal{I}(\boldsymbol{\beta}) + \boldsymbol{\Xi}, \qquad \boldsymbol{\Xi} = \int\!\!\int\mathbf{x}(\mathbf{s})\mathbf{x}(\mathbf{t})'\lambda(\mathbf{s})\lambda(\mathbf{t})\left(g - 1\right)d\mathbf{s}\,d\mathbf{t}
$$

Therefore

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}) \approx \mathcal{I}^{-1}\left(\mathcal{I} + \boldsymbol{\Xi}\right)\mathcal{I}^{-1}
$$

This is the sandwich of Section 1.3.2 in a new setting. The outer pieces are what the Poisson GLM computes, and the middle has an extra term for the dependence. If $g \equiv 1$ then $\boldsymbol{\Xi} = \mathbf{0}$ and the sandwich collapses to $\mathcal{I}^{-1}$, the GLM's answer. For the Thomas process, $g - 1 = \mathcal{N}_c/\kappa$ is a positive definite function, so $\boldsymbol{\Xi}$ is positive semi-definite and the true variance is at least the naive one in every direction. The inflation is largest for covariates that are smooth, because a smooth covariate takes similar values on the events of a cluster, so its sum over the cluster does not average out. These are the same two conditions, positive dependence and a smooth covariate, that made the OLS standard errors of Chapter 1 too small.

### 11.8.4 Estimating the Middle

The matrix $\boldsymbol{\Xi}$ involves the unknown $g$. There are two routes.

**A pair-correlation estimate.** Choose a parametric form for $g$ and plug in estimates. For the Thomas process $g - 1 = \mathcal{N}_c/\kappa$, and $\boldsymbol{\Xi}$ becomes $(1/\kappa)\int\!\!\int\mathbf{x}(\mathbf{s})\mathbf{x}(\mathbf{t})'\lambda(\mathbf{s})\lambda(\mathbf{t})\mathcal{N}_c(\mathbf{s} - \mathbf{t})\,d\mathbf{s}\,d\mathbf{t}$: a double integral over a grid, computed with the separable trick of Section 11.8.1. The parameters $\kappa$ and $\sigma_c$ are estimated by **minimum contrast**: fit the model's $K$-function to the observed inhomogeneous $K$ by least squares, in the way that a parametric variogram is fitted to a binned one in Chapter 3, as Chapter 12 develops in full. The model's $K$-function is

$$
K_{\text{Thomas}}(r) = \pi r^2 + \frac{1}{\kappa}\left(1 - e^{-r^2/(4\sigma_c^2)}\right)
$$

and we compare the fourth roots of the observed and fitted curves, which stabilizes the variance.

**A block estimate.** Cut the region into $J \times J$ blocks and compute the score contribution of each block, $\mathbf{U}_b = \sum_{i \in b}\mathbf{x}(\mathbf{s}_i) - \int_b\mathbf{x}\hat{\lambda}$. If blocks were independent, $\operatorname{Var}(\mathbf{U}) = \sum_b\operatorname{Var}(\mathbf{U}_b)$, which is estimated by $\sum_b\mathbf{U}_b\mathbf{U}_b'$ with a small-sample factor $J^2/(J^2 - 1)$. This needs no model for the clustering. Its weakness is that clusters straddle block boundaries, which makes neighboring blocks dependent and, we expect, the estimate too small, and with few blocks the estimate is noisy.

### 11.8.5 The Parametric Bootstrap

A third way avoids formulas. Fit the Thomas model: the trend by the GLM and $(\kappa, \sigma_c)$ by minimum contrast. Simulate many patterns from the fitted model, refit the GLM to each, and take the standard deviation of the refitted coefficients as the standard error. The bootstrap must simulate **clusters**. A bootstrap that simulates a Poisson process from the fitted trend reproduces the naive standard errors, because it has no clustering to reproduce (Exercise 20).

### 11.8.6 One Clustered Pattern

The code defines the block and pair-correlation sandwiches and the minimum contrast fit, then computes every standard error for one clustered pattern, with 299 bootstrap replicates for the last.

```python
def block_sandwich(P, b, m=30, J=4):
    """Covariance of the Berman-Turner fit with a block-score 'meat'."""
    pts, a, ind = quadrature_counting(P, m)
    X = design(pts)
    mu = np.exp(X @ b)
    info = X.T @ ((a * mu)[:, None] * X)
    contrib = X * (ind - a * mu)[:, None]
    blk = np.floor(pts[:, 1] * J) * J + np.floor(pts[:, 0] * J)
    U = np.zeros((J * J, X.shape[1]))
    np.add.at(U, blk.astype(int), contrib)
    inv = np.linalg.inv(info)
    return inv @ ((J * J / (J * J - 1)) * U.T @ U) @ inv


def pc_sandwich(b, kappa, sigma, m=60):
    """Covariance when pairs of events have pair correlation
    g(h) = 1 + exp(-h^2 / (4 sigma^2)) / (4 pi kappa sigma^2)."""
    Gm = grid_points(m)
    X = design(Gm)
    lam = np.exp(X @ b)
    info = X.T @ (lam[:, None] * X) / m**2
    A = X * lam[:, None]
    inv = np.linalg.inv(info)
    return inv @ (info + pair_integral(A, sigma, m) / kappa) @ inv


def thomas_k(r, kappa, sigma):
    return np.pi * r**2 + (1 - np.exp(-r**2 / (4 * sigma**2))) / kappa


def min_contrast(P, b, r_grid):
    """Fit (kappa, sigma) of an inhomogeneous Thomas process by matching
    K_inhom^(1/4) to the model's K^(1/4)."""
    k_obs = k_hat(P, r_grid, np.exp(design(P) @ b))
    obj = lambda t: np.sum((k_obs**0.25 - thomas_k(
        r_grid, np.exp(t[0]), np.exp(t[1]))**0.25)**2)
    res = optimize.minimize(obj, np.log([50.0, 0.05]), method="Nelder-Mead")
    return np.exp(res.x)


P_cl = sim_c()
b_cl, cov_cl, _, _ = fit_bt(P_cl, 30)
kap_hat, sig_hat = min_contrast(P_cl, b_cl, r_grid)
r_hat = lambda P: np.exp(design(P) @ b_cl)
max_hat = 1.05 * r_hat(grid_points(300) * 1.24 - 0.12).max()
boot = np.array([fit_bt(simulate_cluster(r_hat, max_hat, kap_hat, sig_hat,
                                         rng), 30)[0] for _ in range(299)])
ses = {"naive Poisson GLM": np.sqrt(np.diag(cov_cl)),
       "block sandwich": np.sqrt(np.diag(block_sandwich(P_cl, b_cl))),
       "pair correlation, true (kappa, sigma)":
           np.sqrt(np.diag(pc_sandwich(b_cl, kappa, sigma_c))),
       "pair correlation, estimated":
           np.sqrt(np.diag(pc_sandwich(b_cl, kap_hat, sig_hat))),
       "parametric bootstrap": boot.std(axis=0, ddof=1)}
print("one clustered pattern: n =", len(P_cl), " estimate",
      np.round(b_cl, 3), " truth", np.round(beta_true, 3))
print("minimum contrast: kappa =", round(kap_hat, 1), " sigma =",
      round(sig_hat, 4), " (true 100 and 0.03)")
for name, v in ses.items():
    print(f"{name:38s} SE = {np.round(v, 3)}")
```

The 424-event pattern gives $\hat{\boldsymbol{\beta}} = (5.842, -3.312, 1.698)$ against the truth $(5.846, -3, 1.5)$. Minimum contrast estimates $\hat{\kappa} = 181.7$ and $\hat{\sigma}_c = 0.0323$, against the true $100$ and $0.03$. The cluster size is right, but the parent density is too high, which makes the clustering look weaker than it is.

| Method | SE($\hat{\beta}_0$) | SE($\hat{\beta}_1$) | SE($\hat{\beta}_2$) |
| :--- | :--- | :--- | :--- |
| Naive Poisson GLM | 0.174 | 0.419 | 0.228 |
| Block sandwich ($4 \times 4$ blocks) | 0.311 | 0.480 | 0.559 |
| Pair correlation, true $(\kappa, \sigma_c)$ | 0.436 | 0.920 | 0.648 |
| Pair correlation, estimated | 0.342 | 0.733 | 0.502 |
| Parametric bootstrap | 0.331 | 0.777 | 0.492 |

The naive standard errors are too small by a factor of 2.2 to 2.8 against the pair-correlation values with the true parameters. The estimated-parameter versions fall in between: the overestimate of $\kappa$ shrinks the correction. The block estimate is erratic, as expected with 16 blocks. A single pattern cannot tell which method is right; the repeated experiment of Section 11.9.4 can.

---

## 11.9 Walkthrough and Lab 11

The sections above each ended with a worked example. This section collects the repeated-sampling experiments behind the claims made along the way: that the logistic fit pays a price for an unknown baseline, that the scan test has the stated size and power, that estimating the intensity from the same data biases the $K$-function, and that the corrected standard errors cover.

### 11.9.1 The Price of an Unknown Baseline

For 200 simulated data sets, fit the plume amplitude $\beta_1$ with the offset model (the baseline known) and with logistic regression (the baseline unknown, estimated by controls), for one year of cases and for five. The true amplitude is 1.

```python
rows = []
for r in range(200):
    P0 = simulate_poisson(rate_ctrl, 800.0, rng)
    cases = [simulate_poisson(rate_case, max_case, rng) for _ in range(5)]
    row = []
    for Pc in (cases[0], np.vstack(cases)):
        b_o = fit_bt(Pc, 30, cov_bump(0.1), log_pop)[0]
        Pa, ya = pooled(Pc, P0)
        b_l = irls_logistic(cov_bump(0.1)(Pa), ya)[0]
        row += [b_o[1], b_l[1]]
    rows.append(row)
rows = np.array(rows)
print("200 replications, true b1 = 1")
for k, name in enumerate(("one year, offset", "one year, logistic",
                          "five years, offset", "five years, logistic")):
    print(f"{name:21s} mean {rows[:, k].mean():.3f}  "
          f"sd {rows[:, k].std(ddof=1):.3f}")
```

| Cases | Method | Mean of $\hat{\beta}_1$ | SD |
| :--- | :--- | :--- | :--- |
| One year | Offset (baseline known) | 0.963 | 0.780 |
| One year | Logistic | 1.012 | 0.860 |
| Five years | Offset (baseline known) | 1.008 | 0.282 |
| Five years | Logistic | 1.055 | 0.475 |

Both methods are close to unbiased. The logistic fit loses 10% in standard deviation with one year of cases and 68% with five. The reason is that the baseline is estimated from 385 controls, and the logistic variance has two parts, one from the cases and one from the controls. With 58 cases, the cases' part dominates. With 296 cases, the controls' part is as large as the cases'. More controls would close the gap (Exercise 16).

### 11.9.2 Size and Power of the Scan Test

Using the reference distribution of Section 11.6.4, run 200 data sets for each of four cluster sizes, with $n = 350$ events in all, and record how often the Monte Carlo p-value is below $0.05$, and, among the rejections, how often the best window's center lies within $0.1$ of the true center.

```python
print("extra  reject  localized")
for extra in (0, 10, 20, 35):
    rej = loc = 0
    for rep in range(200):
        lk = scan(inject(350 - extra, extra, rng))[0]
        i, j = np.unravel_index(lk.argmax(), lk.shape)
        if (1 + np.sum(null_max >= lk.max())) / (B + 1) < 0.05:
            rej += 1
            loc += np.hypot(*(centers[i] - [0.25, 0.2])) < 0.1
    print(f"{extra:5d} {rej / 200:7.3f} {loc / max(rej, 1):9.3f}")
```

| Extra events | Rejection rate | Best window near true center |
| :--- | :--- | :--- |
| 0 | 0.065 | not applicable |
| 10 | 0.535 | 0.869 |
| 20 | 1.000 | 0.990 |
| 35 | 1.000 | 1.000 |

With no cluster the test rejects in 6.5% of the 200 data sets, near the nominal 5% (a standard error of about 1.5%). Ten extra events in a window that expects about two are detected half the time, and twenty are always detected and nearly always located correctly. These numbers describe this cluster size and this baseline, and say how much power one can expect, not that the method is always this sensitive.

### 11.9.3 Plug-In Intensities and the Inhomogeneous $K$

In 100 Poisson crash patterns, compute $\hat{K}_{\text{inhom}}(r)/(\pi r^2)$ at $r = 0.04$ and $0.08$ with six intensity estimates. The correct value is 1. The kernel estimates are edge corrected and use a Gaussian kernel of bandwidth $0.10$ or $0.03$, either including the event itself or leaving it out.

```python
def kernel_at_points(P, psi, leave_out):
    """Kernel intensity estimate at the data points, edge corrected."""
    phi = np.exp(-dist(P)**2 / (2 * psi**2)) / (2 * np.pi * psi**2)
    if leave_out:
        np.fill_diagonal(phi, 0.0)
    edge = lambda c: special.ndtr((1 - c) / psi) - special.ndtr(-c / psi)
    return phi.sum(axis=1) / (edge(P[:, 0]) * edge(P[:, 1]))


r_chk = np.array([0.04, 0.08])
ratio = {k: [] for k in ("true intensity", "fitted GLM",
                         "kernel 0.10, own point", "kernel 0.10, left out",
                         "kernel 0.03, own point", "kernel 0.03, left out")}
for s in range(100):
    Ps = sim_p()
    bs = fit_bt(Ps, 20)[0]
    lam_options = {"true intensity": rate(Ps),
                   "fitted GLM": np.exp(design(Ps) @ bs),
                   "kernel 0.10, own point": kernel_at_points(Ps, .10, False),
                   "kernel 0.10, left out": kernel_at_points(Ps, .10, True),
                   "kernel 0.03, own point": kernel_at_points(Ps, .03, False),
                   "kernel 0.03, left out": kernel_at_points(Ps, .03, True)}
    for k, lam in lam_options.items():
        ratio[k].append(k_hat(Ps, r_chk, lam) / (np.pi * r_chk**2))
print("mean K_inhom / (pi r^2) at r = 0.04, 0.08 (1 is correct):")
for k, v in ratio.items():
    print(f"  {k:24s}", np.round(np.mean(v, axis=0), 3))
```

| Intensity estimate | $r = 0.04$ | $r = 0.08$ |
| :--- | :--- | :--- |
| True intensity | 1.012 | 0.999 |
| Fitted GLM | 1.008 | 0.995 |
| Kernel 0.10, includes the event | 0.794 | 0.801 |
| Kernel 0.10, event left out | 0.885 | 0.893 |
| Kernel 0.03, includes the event | 0.358 | 0.453 |
| Kernel 0.03, event left out | 0.806 | 2.016 |

The GLM estimate, with three parameters, gives the same answer as the true intensity. A kernel estimate that includes the event itself is biased downward, because every event contributes to its own estimated intensity; the smaller the bandwidth, the larger the contribution and the bias. Leaving the event out reduces the bias but does not remove it, and with a small bandwidth the estimate at an isolated event can be close to zero, which makes some pair weights enormous and the result unstable, as the value $2.016$ shows. A parsimonious parametric intensity is the safer plug-in.

### 11.9.4 Coverage Under Clustering

Simulate 300 clustered patterns. For each, fit the Poisson GLM and compute four standard errors: the naive one, the block sandwich, and the pair-correlation sandwich with the true and with the estimated $(\kappa, \sigma_c)$. A 95% interval covers when $\lvert\hat{\beta} - \beta\rvert < 1.96\,\text{SE}$.

```python
rows = []
for r in range(300):
    P = sim_c()
    b, cv, _, _ = fit_bt(P, 30)
    kh, sh = min_contrast(P, b, r_grid)
    se = [np.sqrt(np.diag(cv)), np.sqrt(np.diag(block_sandwich(P, b))),
          np.sqrt(np.diag(pc_sandwich(b, kappa, sigma_c))),
          np.sqrt(np.diag(pc_sandwich(b, kh, sh)))]
    rows.append(np.concatenate([b] + se))
rows = np.array(rows)
bh, sd = rows[:, :3], rows[:, :3].std(axis=0, ddof=1)
print("300 clustered patterns: mean estimate", np.round(bh.mean(axis=0), 3),
      " sd", np.round(sd, 3))
for k, name in enumerate(("naive Poisson", "block sandwich",
                          "pair corr., true", "pair corr., estimated")):
    se = rows[:, 3 + 3 * k:6 + 3 * k]
    cover = (np.abs(bh - beta_true) < 1.96 * se).mean(axis=0)
    print(f"{name:22s} mean SE {np.round(se.mean(axis=0), 3)}  "
          f"coverage {np.round(cover, 3)}")
```

For comparison, 300 Poisson patterns from the same trend:

```python
p_rows = []
for r in range(300):
    P = sim_p()
    b, cv, _, _ = fit_bt(P, 30)
    p_rows.append(np.concatenate([b, np.sqrt(np.diag(cv))]))
p_rows = np.array(p_rows)
cover = (np.abs(p_rows[:, :3] - beta_true) < 1.96 * p_rows[:, 3:]).mean(axis=0)
print("300 Poisson patterns: mean estimate",
      np.round(p_rows[:, :3].mean(axis=0), 3), " sd",
      np.round(p_rows[:, :3].std(axis=0, ddof=1), 3), " naive coverage",
      np.round(cover, 3))
```

With Poisson data, the estimates are centered on the truth, $(5.854, -3.024, 1.487)$, the standard deviations are $(0.172, 0.420, 0.237)$, the naive standard errors are right, and the intervals cover 95.7%, 95.3%, and 96.3% of the time. With clustered data, the estimates are still centered on the truth, at $(5.858, -3.034, 1.435)$, so **clustering does not bias the coefficients**. It makes them more variable: the standard deviations are $(0.400, 0.856, 0.639)$, more than twice the Poisson values.

| Method | Mean SE | Coverage of 95% intervals |
| :--- | :--- | :--- |
| Naive Poisson | $(0.181, 0.425, 0.243)$ | $(0.620, 0.680, 0.600)$ |
| Block sandwich | $(0.331, 0.734, 0.456)$ | $(0.867, 0.877, 0.803)$ |
| Pair correlation, true $(\kappa, \sigma_c)$ | $(0.437, 0.906, 0.649)$ | $(0.970, 0.957, 0.940)$ |
| Pair correlation, estimated | $(0.423, 0.881, 0.628)$ | $(0.960, 0.947, 0.937)$ |

The naive intervals cover only 60% to 68% of the time. The block sandwich helps but is still too small. The pair-correlation sandwich is almost exactly right, whether the cluster parameters are known or estimated by minimum contrast, because the formula $\mathcal{I}^{-1}(\mathcal{I} + \boldsymbol{\Xi})\mathcal{I}^{-1}$ is right when $g$ is. The price is that one must model the interaction.

### 11.9.5 The Bootstrap Over Repeated Patterns

For 50 clustered patterns, run the parametric bootstrap of Section 11.8.5 with 99 resamples each, and compare its mean standard error with the spread of the estimates.

```python
boot_rows = []
for r in range(50):
    P = sim_c()
    b = fit_bt(P, 30)[0]
    kh, sh = min_contrast(P, b, r_grid)
    rh = lambda Q, b=b: np.exp(design(Q) @ b)
    mh = 1.05 * rh(grid_points(300) * 1.24 - 0.12).max()
    bb = np.array([fit_bt(simulate_cluster(rh, mh, kh, sh, rng), 30)[0]
                   for _ in range(99)])
    boot_rows.append(np.concatenate([b, bb.std(axis=0, ddof=1)]))
boot_rows = np.array(boot_rows)
cover = (np.abs(boot_rows[:, :3] - beta_true)
         < 1.96 * boot_rows[:, 3:]).mean(axis=0)
print("50 clustered patterns, bootstrap: mean SE",
      np.round(boot_rows[:, 3:].mean(axis=0), 3), " sd of estimates",
      np.round(boot_rows[:, :3].std(axis=0, ddof=1), 3),
      " coverage", np.round(cover, 3))
```

The mean bootstrap standard errors are $(0.432, 0.894, 0.651)$, close to the pair-correlation values and to the standard deviations of the estimates over the 50 patterns, $(0.479, 0.808, 0.792)$, which are noisy with only 50 patterns. Coverage is $0.90$, $0.98$, and $0.92$, with a standard error of 3% to 4% on each. The bootstrap is slower than the sandwich and needs a model for the clustering, and in return it makes no first-order approximation.

---

## Exercises

**Analytical**

1. Show that for a homogeneous Poisson process on $\mathcal{D}$, conditioning on $N(\mathcal{D}) = n$ makes the locations $n$ independent uniform points, so the locations carry no information about $\lambda$. Which factor of the likelihood carries all of it?

2. For $\lambda(s) = \exp(\beta_0 + \beta_1 s)$ on $[0, 1]$ (one dimension), compute $\mathcal{I}(\boldsymbol{\beta})$ in closed form. Show that it is positive definite when $\beta_1$ is any real number, and find its determinant.

3. Verify the second component of the score at the starting value in Section 11.3.4: $\sum_ix_i - 3\sum_ka_kx_k = 1.3 - 1.51875 = -0.2188$. Compute $\sum_ka_kx_k$ by hand from the table.

4. Show that counting weights are nonnegative and sum to $\operatorname{area}(\mathcal{D})$. If no data points fall in a cell, what rule do the weights reduce to?

5. Show that if $a$ is a positive integer, $a(y\log\mu - \mu)$ equals the sum of the Poisson log-likelihood contributions of $a$ observations, each equal to $y$, up to a constant that does not involve $\mu$.

6. With only an intercept and an offset $\log\lambda_{\text{pop}}(\mathbf{s})$, show that $\hat{\beta}_0 = \log\bigl(n/\int\lambda_{\text{pop}}\bigr)$. Explain why this is the maximum likelihood estimate of the overall relative risk.

7. Let cases have intensity $\kappa_1\lambda_{\text{pop}}(\mathbf{s})\theta(\mathbf{s})$ and controls $\kappa_0\lambda_{\text{pop}}(\mathbf{s})$. Show that multiplying the number of years of cases by $k$ adds $\log k$ to the true intercept of the logistic model and leaves the slopes unchanged.

8. Derive the scan statistic from a Poisson model: inside window $w$ the intensity is multiplied by $r_1$, outside by $r_0$, with expected counts $e_w$ and $n_{\text{tot}} - e_w$ under the baseline. Maximize over $(r_1, r_0)$ and show the maximized log-likelihood minus the null log-likelihood is $\operatorname{LLR}_w$ when the baseline's total expected count equals $n$.

9. For a fixed window, expand $\operatorname{LLR}_w$ around $n_w = e_w$ and show that $2\operatorname{LLR}_w \approx (n_w - e_w)^2/e_w$ when $n$ is large and $e_w/n$ is small. Why does this imply that $2\operatorname{LLR}_w$ is approximately $\chi^2_1$ for a fixed window?

10. For the unit square, derive the translation weight $\omega_{ij} = 1/((1 - \lvert\Delta x\rvert)(1 - \lvert\Delta y\rvert))$ by computing the area of $\mathcal{D} \cap (\mathcal{D} + \mathbf{h})$.

11. Derive $K_{\text{ord}}(r) \approx \pi r^2(1 + \operatorname{CV}^2)$ for an inhomogeneous Poisson process. For $\lambda(\mathbf{s}) = \bar{\lambda}(1 + a\cos(2\pi s_1))$, compute $\operatorname{CV}^2$ and the predicted $(L - r)/r$.

12. Show that for the Thomas process before thinning, $\int(g(h) - 1)\,d\mathbf{h} = 1/\kappa$, and conclude that the variance-to-mean ratio of the count in a very large region is $1 + \lambda/\kappa = 1 + \mu_c$.

13. For an intercept-only log-linear model fitted to a clustered process in a large region, show that $\operatorname{Var}(\hat{\beta}_0) \approx (1 + \lambda/\kappa)/(\lambda\operatorname{area}(\mathcal{D}))$. By what factor are the naive standard errors too small when $\lambda = 400$ and $\kappa = 100$?

14. In the sandwich $\mathcal{I}^{-1}(\mathcal{I} + \boldsymbol{\Xi})\mathcal{I}^{-1}$ for an intercept-only model, show that $\boldsymbol{\Xi}$ is negative when $g < 1$ everywhere (inhibition), so that the naive standard error is too large. Chapter 12 develops models with $g < 1$.

**Computational and simulation**

15. Add a covariate with no effect (the north coordinate) to the crash model and fit it to 500 Poisson patterns. What fraction of the z-tests for its coefficient reject at 5%? Repeat with 500 clustered patterns.

16. In the case-control experiment of Section 11.9.1, replace the 385 controls (363 expected) by about 4,000 expected controls (intensity $8{,}000\,\lambda_{\text{pop}}$) and repeat the five-year comparison. Does the standard deviation of the logistic fit approach that of the offset fit?

17. Fit the bump logistic model with the wrong center, $(0.5, 0.5)$ instead of $(0.3, 0.6)$, and with the right center but a ridge-shaped true excess along a line. How well does each recover the pattern, and what does the likelihood ratio test say?

18. Run the scan statistic on data from the baseline of Section 11.6.3 but with a homogeneous baseline $\lambda_{\text{bg}} \equiv$ constant. Where does the scan find clusters, and how often does it reject at 5%?

19. Make a figure of $\hat{K}_{\text{inhom}}(r)/(\pi r^2)$ at $r = 0.08$ against the kernel bandwidth, from $0.02$ to $0.3$, with and without leaving the event out, for 100 Poisson patterns.

20. Replace the parametric bootstrap of Section 11.8.5 by one that simulates a Poisson process from the fitted trend. Show that its standard errors are close to the naive Poisson GLM standard errors and that its coverage under clustering is poor.

21. Vary the cluster strength: $\kappa \in \{50, 100, 400, 1{,}600\}$ with the same trend. For each, plot the ratio of the standard deviation of $\hat{\beta}_1$ to its naive standard error against the variance-to-mean ratio of the counts. Does the ratio of standard errors track the square root of the variance-to-mean ratio?

22. Vary the number of blocks $J$ in the block sandwich from 2 to 8 and report the coverage of the intervals for $\beta_1$ under clustering. Is there a $J$ that works?

---

## What Comes Next

This chapter treated the structure of a point pattern as known or as a nuisance. Trends were modeled and clustering was corrected for, but the clusters themselves were never described. Chapter 12 turns to them. It develops the Thomas process that we used here and the Matérn cluster process, and fits them by minimum contrast, as a preview did in Section 11.8. It introduces the log-Gaussian Cox process, which puts a spatial random effect $u(\mathbf{s})$ into the log-intensity and so connects point patterns with the geostatistics of Part II. And it treats inhibition with hard-core and Strauss processes, where the interaction term enters the Berman-Turner design matrix as one more column, so that the IRLS of this chapter fits them as well.
