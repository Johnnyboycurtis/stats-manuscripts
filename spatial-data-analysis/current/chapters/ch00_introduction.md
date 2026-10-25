# Chapter 0: The Big Picture and the Math You Already Know

Most statistics courses rest on one quiet assumption: the rows of a data set are independent. Each observation is a fresh draw, unrelated to the one before it. That assumption is what makes $\sigma^2/n$ the variance of a sample mean, what makes regression standard errors correct, and what makes a p-value mean what it says. (Here $\sigma^2$ is just the variance of one observation. From Chapter 1 on, this book writes the variance of independent errors as $\tau^2$ and keeps $\sigma^2$ for the variance of the spatial random effect introduced in Section 0.2.)

Spatial data breaks that assumption in a predictable way. Two apartments on the same block rent for similar amounts, because they share schools, transit, and street noise. Two neighboring census tracts share employers, clinics, and commuters. Traffic crashes pile up at a few badly designed intersections instead of scattering at random. When observations near each other are related, the familiar formulas still produce numbers, but the numbers overstate how much the data know.

This chapter has two jobs. The first is to show, with small simulations and no heavy algebra, what spatial statistics is and why it is needed. The second is to take stock of the mathematics you already have. Almost everything in this book is built from five tools you met in linear algebra, regression, and mathematical statistics. If you can use those five tools, you can follow every derivation in the chapters ahead.

> **Chapter at a Glance**
>
> - **What Spatial Statistics Is (§0.1):** Statistics for data where location matters and nearby observations are dependent. We meet the three kinds of spatial data and the question each one asks.
> - **One Equation for the Whole Book (§0.2):** Every model in the book is a version of $Y = \mu + u + \varepsilon$: a trend, a spatial random effect, and independent noise.
> - **Three Ways Classical Statistics Misleads (§0.3):** Simulations show inflated sample sizes, correlation between unrelated maps, and rate maps dominated by small populations.
> - **The Math Inventory (§0.4):** Five tools you already know, each shown at work on a spatial problem, and a list of what the book builds on top of them.
> - **How the Book Works (§0.5–§0.7):** What we leave out and why, the rhythm of each chapter, a map of the book, a software check, and warm-up exercises.

---

## 0.1 What Is Spatial Statistics?

### 0.1.1 When Rows Are Not Independent

Picture a data frame of apartment listings. Each row is an apartment: an identifier, two coordinates, and a monthly rent. In a regression course you would fit a model to this table and treat the rows as independent. Row 4 and row 5 would be unrelated draws.

Now suppose apartments 4 and 5 are in the same building. They share a location, a school zone, and a walk to the same train station. If apartment 4 rents for more than its size and condition predict, because the block has quietly become desirable, apartment 5 probably does too. Their errors are not independent. They share information.

This is the observation behind what geographers call Tobler's first law: everything is related to everything else, but near things are more related than distant things. Spatial statistics is the part of statistics that takes this seriously. It treats location as part of the data, describes how dependence changes with distance, and adjusts estimation, prediction, and testing to account for it.

The cost of ignoring dependence is not that estimates become wrong on average. A regression coefficient fitted by ordinary least squares is still unbiased when the errors are spatially correlated. The cost is that we misjudge how precise our estimates are. Positively correlated observations repeat each other, so a data set of $n$ correlated rows carries less information than $n$ independent rows. Formulas that assume independence count every row as new information, and so they report standard errors that are too small and p-values that are too optimistic. Section 0.3 shows how large the effect can be.

### 0.1.2 One World, One Map

Classical statistics leans on replication. To learn about a distribution, we observe many independent draws from it. Spatial data rarely offers that. We observe one set of rents in one city in one month, one arrangement of immune cells on one tissue slide, one set of overdose counts for one set of counties. There is no second planet to sample.

So where does replication come from? The answer, developed in Chapter 2, is an assumption about the process: that the rules generating the data are the same everywhere in the region. If the relationship between two locations depends only on the separation between them (how far apart they are and in which direction), and not on where they are, then every pair of locations 100 meters apart in the same direction is a replicate of "two locations 100 meters apart." A single map then contains many replicates of each separation, and that is enough to estimate how dependence decays. Chapter 2 refines this by separating a version in which direction matters from one in which only distance matters.

This assumption is called stationarity. It is useful and it is checkable, but it is an assumption. Much of the practical skill in spatial statistics lies in deciding which parts of a map can reasonably be treated as following the same rules, and in modeling the parts that cannot (for example, with a trend in the mean).

### 0.1.3 Three Kinds of Spatial Data, Three Questions

Spatial data comes in three broad forms. They differ in what is random, and so they lead to different questions and different models.

| Data type | What is random | Example | The central question | Where in the book |
| :--- | :--- | :--- | :--- | :--- |
| Geostatistical (continuous space) | The measured value, at fixed locations | Rents at listed apartments; noise levels at street monitors | What is the value where we did not measure, and how sure are we? | Part II (Chapters 3–5) |
| Areal or lattice (regions) | The value attached to each region of a fixed map | Overdose deaths by county; median income by census tract | Does a region differ from its neighbors by more than chance, and what explains it? | Part III (Chapters 6–9) |
| Point pattern (events) | The locations themselves | Immune cells on a tumor slide; addresses of disease cases; traffic crashes | Are events clustered, regular, or random, and what drives their density? | Part IV (Chapters 10–12) |

The first type is about a quantity that varies smoothly across space, like the rent level of a city or the noise level of its streets, and that we have measured at a finite number of points. The second is about quantities that only make sense for whole regions, like a county's mortality rate. The third is different in kind: there is no measured value at all, only a list of where things happened. The same three types turn up far from geography. Gene expression measured on a grid of spots across a tissue slide is areal data. The locations of a basketball player's shot attempts are a point pattern. The book treats all three with one modeling framework, which is the subject of the next section.

---

## 0.2 One Equation for the Whole Book

Every model in this book is a variation on a single equation. For a location $\mathbf{s}$,

$$
Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})
$$

In words, an observation is a trend, plus a spatial random effect, plus independent noise.

- **The trend $\mu(\mathbf{s})$** is the part explained by covariates, written as a regression: $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$. For rents, square footage, distance to a transit stop, or the coordinates themselves can all enter $\mathbf{x}(\mathbf{s})$. This is the linear model you already know.
- **The spatial random effect $u(\mathbf{s})$** is the part that is not explained by covariates but is shared by nearby locations. It is random, it has mean zero, and its values at two locations are correlated, more strongly the closer the locations are. Its variance is written $\sigma^2$.
- **The noise $\varepsilon(\mathbf{s})$** is independent from location to location. It includes measurement error and any variation at scales finer than the spacing of the data. Its variance is written $\tau^2$.

If you have seen mixed models, $u$ will feel familiar. In a study of students nested in schools, each school gets a random intercept, and two students in the same school are correlated because they share it. A spatial random effect works the same way, with one change: there are no discrete schools. Every location has its own effect, and the effects of nearby locations are correlated, with the correlation fading smoothly with distance. A spatial model is a mixed model in which the random effect is a surface.

Collect $n$ observations into a vector $\mathbf{Y}$. Because $u$ and $\varepsilon$ are independent of each other, their variances add, and the whole spatial structure of the data sits in one covariance matrix:

$$
\operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n
$$

Here $\mathbf{R}(\phi)$ holds the correlations of the random effect between every pair of locations, and $\phi$ sets how quickly correlation fades with distance. If $\sigma^2 = 0$, this collapses to $\tau^2\mathbf{I}_n$, the independent-errors model of ordinary regression. Everything spatial in this book is about what happens when $\sigma^2 > 0$: how to estimate $\boldsymbol{\Sigma}$, how to do regression with it, and how to predict with it.

The parts of the book change one piece of the equation at a time:

- **Part II** treats $u$ as a continuous surface and describes it through a covariance function of distance.
- **Part III** defines $u$ on a map of regions, describing each region's effect through its neighbors. Chapter 9 places the same equation inside a generalized linear model for counts.
- **Part IV** uses the equation for the logarithm of the density of events, connecting point patterns back to everything before.
- **Part V** keeps the equation and asks how to compute with it when $n$ is in the tens of thousands.

---

## 0.3 Three Ways Classical Statistics Misleads in Space

This section uses three small simulations to show what goes wrong when spatial dependence is ignored. You do not need to follow the simulation code in detail yet. Chapter 2 explains exactly how it works, using a Cholesky factorization of $\boldsymbol{\Sigma}$. For now, treat it as a way to draw random maps whose true properties we know.

All three simulations use a regular grid of locations in the unit square and the exponential correlation function $r(h;\phi) = \exp(-h/\phi)$, where $h$ is the distance between two locations. A larger $\phi$ means correlation persists over longer distances. The helper functions below build the grid and the matrix of pairwise distances.

```python
import numpy as np
from scipy import stats

def grid(m):
    """Centers of an m-by-m grid of cells in the unit square, as (m*m, 2)."""
    g = (np.arange(m) + 0.5) / m
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])

def dist(S):
    """Matrix of Euclidean distances between the rows of S."""
    q = np.sum(S**2, axis=1)
    H2 = q[:, None] - 2 * S @ S.T + q[None, :]
    H = np.sqrt(np.maximum(H2, 0.0))  # rounding fix, see Section 1.6.1
    np.fill_diagonal(H, 0.0)
    return H
```

### 0.3.1 More Data Is Not Always More Information

Suppose a city installs noise monitors on an $m \times m$ grid across a district, here the unit square, to estimate the district's average noise level. For simplicity take $\mu$ constant, $\sigma^2 = 1$, and $\tau^2 = 0$. The estimate is the sample mean $\bar{Y} = \frac{1}{n}\mathbf{1}'\mathbf{Y}$.

The variance of $\bar{Y}$ follows from Tool 1 in Section 0.4: for a fixed vector $\mathbf{a}$, $\operatorname{Var}(\mathbf{a}'\mathbf{Y}) = \mathbf{a}'\boldsymbol{\Sigma}\mathbf{a}$. With $\mathbf{a} = \frac{1}{n}\mathbf{1}$,

$$
\operatorname{Var}(\bar{Y}) = \frac{1}{n^2}\,\mathbf{1}'\boldsymbol{\Sigma}\mathbf{1} = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \Sigma_{ij}
$$

If the observations were independent, $\boldsymbol{\Sigma} = \mathbf{I}_n$ and this would be the familiar $1/n$. With correlation, every off-diagonal entry adds to the variance. A useful summary is the **effective sample size**, the number of independent observations that would give the same precision:

$$
n_{\text{eff}} = \frac{\sigma^2}{\operatorname{Var}(\bar{Y})}
$$

The code below computes $\operatorname{Var}(\bar{Y})$ and $n_{\text{eff}}$ exactly, with no simulation needed, for grids of 100, 400, and 1,600 locations and three values of $\phi$.

```python
for m in (10, 20, 40):
    S = grid(m)
    n = len(S)
    H = dist(S)
    for phi in (0.05, 0.1, 0.2):
        R = np.exp(-H / phi)
        var_mean = R.sum() / n**2
        print(f"n={n:5d}  phi={phi:4.2f}  Var(mean)={var_mean:.4f}  "
              f"iid={1/n:.5f}  n_eff={1/var_mean:6.1f}")
```

| $n$ | $\phi = 0.05$ | $\phi = 0.1$ | $\phi = 0.2$ |
| :--- | :--- | :--- | :--- |
| 100 | $n_{\text{eff}} = 54.5$ | $n_{\text{eff}} = 19.6$ | $n_{\text{eff}} = 6.9$ |
| 400 | $n_{\text{eff}} = 69.3$ | $n_{\text{eff}} = 20.6$ | $n_{\text{eff}} = 7.0$ |
| 1,600 | $n_{\text{eff}} = 72.1$ | $n_{\text{eff}} = 20.8$ | $n_{\text{eff}} = 7.0$ |

Two things stand out. First, the losses are large. With $\phi = 0.2$, a grid of 1,600 monitors estimates the district average about as well as 7 independent measurements would. A confidence interval built on $n = 1{,}600$ would be roughly $\sqrt{1600/7} \approx 15$ times too narrow.

Second, adding monitors inside the same district barely helps. Going from 400 to 1,600 monitors quadruples the data and leaves $n_{\text{eff}}$ almost unchanged. New monitors placed between old ones mostly repeat what their neighbors already report.

This does not mean dense sampling is useless. It depends on the question. For predicting the noise level at an unmonitored address, a nearby monitor is very valuable, and Section 4.8.5 shows that the prediction error falls steadily as sampling gets denser. What dense sampling cannot do is make a district average more precise than the field's own variability allows. Chapter 1 develops this argument and extends it to regression coefficients.

### 0.3.2 Correlation Out of Nothing

Now simulate two maps on the same $20 \times 20$ grid, completely independently of each other, each with $\phi = 0.2$. Then do what an analyst might do with two real maps, such as neighborhood rents and childhood asthma rates: compute the correlation across the 400 locations and test whether it differs from zero with the usual Pearson test.

```python
rng = np.random.default_rng(2026)
S = grid(20)
n = len(S)
# tiny jitter on the diagonal; explained in Section 2.5.1
L = np.linalg.cholesky(np.exp(-dist(S) / 0.2) + 1e-10 * np.eye(n))

reps = 1000
rs = np.empty(reps)
rejections = 0
for b in range(reps):
    y1 = L @ rng.standard_normal(n)   # first map
    y2 = L @ rng.standard_normal(n)   # second, independent map
    r, p = stats.pearsonr(y1, y2)
    rs[b] = r
    rejections += p < 0.05

print("median |r|:", np.median(np.abs(rs)))
print("share with |r| > 0.3:", np.mean(np.abs(rs) > 0.3))
print("naive test rejects:", rejections / reps)

# The same test on independent, non-spatial data
rej_iid = sum(
    stats.pearsonr(rng.standard_normal(n), rng.standard_normal(n))[1] < 0.05
    for _ in range(reps)
)
print("naive test rejects (independent data):", rej_iid / reps)
```

The two maps are unrelated by construction, so a test at the 5% level should reject about 5% of the time. It rejects in 59.4% of the 1,000 simulations. (Repeating the experiment with independent, non-spatial data gives a rejection rate of 5.1%, as it should.)

The sample correlations themselves are usually modest: the median $\lvert r \rvert$ is 0.12, and only 9.2% exceed 0.3. The problem is the test, not the size of $r$. The Pearson test assumes 400 independent pairs and judges $r$ against the tiny sampling variability that 400 independent pairs would have. But a smooth map has only a handful of independent "ups and downs" across the region, as Section 0.3.1 showed, so two smooth maps line up by chance far more often than the test expects.

The same mechanism affects regression. Chapter 1 shows that ordinary least squares standard errors fail most severely when a covariate is itself spatially smooth, which is the usual situation with real covariates such as income, air quality, or distance to downtown.

### 0.3.3 Small Populations Make Loud Maps

The third simulation involves no spatial correlation at all, and it still misleads. Take 100 counties with very different populations, as real counties have. In this simulation populations range from 244 to 55,143, with a median of 4,349. Count overdose deaths over five years, and give every county exactly the same risk, $\theta = 0.001$ (one death per 1,000 residents), so the true rate is identical everywhere. Draw each county's count from a Poisson distribution with mean $\theta \cdot \text{pop}_i$, compute the raw rate $Y_i / \text{pop}_i$, and ask which counties look most extreme.

```python
rng = np.random.default_rng(7)
n_counties = 100
pop = np.round(np.exp(rng.normal(np.log(5000), 1.2, n_counties))).astype(int)
smallest_quarter = pop <= np.quantile(pop, 0.25)
theta = 0.001

reps = 1000
top_small = bottom_small = 0.0
for b in range(reps):
    y = rng.poisson(theta * pop)
    order = np.argsort(y / pop)
    top_small += smallest_quarter[order[-10:]].mean()
    bottom_small += smallest_quarter[order[:10]].mean()

print("top 10 rates from the smallest quarter:", top_small / reps)
print("bottom 10 rates from the smallest quarter:", bottom_small / reps)
```

If population size had nothing to do with which counties look extreme, about 25% of the ten highest rates would come from the smallest quarter of counties. In fact, on average 55% of the ten highest rates and 80% of the ten lowest come from that quarter. Many of the lowest are small counties with zero deaths.

The reason is the variance of a rate. Since $\operatorname{Var}(Y_i) = \theta \cdot \text{pop}_i$ for a Poisson count,

$$
\operatorname{Var}\!\left(\frac{Y_i}{\text{pop}_i}\right) = \frac{\theta}{\text{pop}_i}
$$

A county with 300 people has a rate about 180 times more variable than a county with 55,000. A map of raw rates therefore highlights small counties at both ends, and a reader looking for hot spots will find them in the least populated places, even though the true risk is flat. Chapter 9 fixes this by borrowing strength: estimating each county's rate partly from its own data and partly from its neighbors, using the spatial random effect $u$ from Section 0.2.

---

## 0.4 The Math Inventory: Five Tools You Already Know

Most derivations in this book use some combination of five tools from earlier courses. This section reviews each one briefly and shows it doing spatial work. If a tool feels rusty, the warm-up exercises in Section 0.7 and the review of the five tools in Appendix A will help.

### 0.4.1 Tool 1: Linear Transformations of Random Vectors

If $\mathbf{Y}$ is a random vector with mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$, and $\mathbf{B}$ is a fixed matrix, then

$$
\operatorname{E}[\mathbf{B}\mathbf{Y}] = \mathbf{B}\boldsymbol{\mu}, \qquad \operatorname{Var}(\mathbf{B}\mathbf{Y}) = \mathbf{B}\boldsymbol{\Sigma}\mathbf{B}'
$$

When $\mathbf{B}$ is a single row $\mathbf{a}'$, the second formula gives the variance of a linear combination: $\operatorname{Var}(\mathbf{a}'\mathbf{Y}) = \mathbf{a}'\boldsymbol{\Sigma}\mathbf{a}$. That is how Section 0.3.1 found the variance of the sample mean.

This one rule does an enormous amount of work in the book. Here is a preview of how it explains simulation. Let $\mathbf{Z}$ be a vector of independent standard normals, so $\operatorname{Var}(\mathbf{Z}) = \mathbf{I}_n$, and let $\mathbf{L}$ be any matrix with $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$. Then

$$
\operatorname{Var}(\mathbf{L}\mathbf{Z}) = \mathbf{L}\,\mathbf{I}_n\,\mathbf{L}' = \boldsymbol{\Sigma}
$$

So multiplying independent noise by $\mathbf{L}$ produces a vector with exactly the covariance we want. That is the entire idea behind the line `L @ rng.standard_normal(n)` in Section 0.3.2. Chapter 2 finds $\mathbf{L}$ with the Cholesky factorization.

The same rule gives the variance of OLS estimates under spatial correlation (Chapter 1) and the error of a kriging prediction (Chapter 4).

### 0.4.2 Tool 2: Quadratic Forms, and Why Variances Cannot Be Negative

A quadratic form is a scalar of the form

$$
\mathbf{y}'\mathbf{M}\mathbf{y} = \sum_{i=1}^n\sum_{j=1}^n M_{ij}\,y_i\,y_j
$$

for a symmetric matrix $\mathbf{M}$. You have used them without the name. The residual sum of squares is one. So is the centered sum of squares: with the centering matrix $\mathbf{P}_c = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'$, we have $\mathbf{y}'\mathbf{P}_c\mathbf{y} = \sum_i (y_i - \bar{y})^2$. In spatial statistics, Moran's $I$ (Chapter 6) and the smoothness penalty of the intrinsic CAR model (Chapter 8) are both quadratic forms, and much of their behavior follows from that fact.

Quadratic forms also explain a requirement that can look like a technicality. For any covariance matrix $\boldsymbol{\Sigma}$ and any vector $\mathbf{a}$, Tool 1 says $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} = \operatorname{Var}(\mathbf{a}'\mathbf{Y})$. A variance cannot be negative, so

$$
\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} \ge 0 \quad \text{for every } \mathbf{a}
$$

This is what "positive semi-definite" means. It is not an extra assumption. It is a property every genuine covariance matrix has.

It also rules out matrices that look reasonable. Suppose we claim that variable 1 is strongly positively correlated with variables 2 and 3 (correlation 0.9 each), while variables 2 and 3 are strongly negatively correlated with each other (correlation $-0.9$):

$$
\boldsymbol{\Sigma} = \begin{bmatrix} 1 & 0.9 & 0.9 \\ 0.9 & 1 & -0.9 \\ 0.9 & -0.9 & 1 \end{bmatrix}
$$

Take $\mathbf{a} = (1, -1, -1)'$. Then $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} = -2.4$, a negative "variance." No three random variables can have these correlations. Chapter 2 shows that the same trap catches natural-looking functions of distance, which is why spatial statistics works with a short list of proven covariance functions.

### 0.4.3 Tool 3: The Multivariate Normal Density, Read in Two Parts

If $\mathbf{Y} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with $n$ components, its log density is

$$
\log f(\mathbf{y}) = -\frac{n}{2}\log(2\pi) \;-\; \frac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert \;-\; \frac{1}{2}(\mathbf{y} - \boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \boldsymbol{\mu})
$$

The formula is easier to read as two competing parts.

- **The determinant term** $-\frac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert$ measures spread. A covariance matrix that allows a lot of variability has a large determinant, and this term penalizes it.
- **The distance term** $-\frac{1}{2}(\mathbf{y} - \boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \boldsymbol{\mu})$ measures fit. It is a squared distance between the data and the mean, measured in units that account for the variances and correlations in $\boldsymbol{\Sigma}$. This is the Mahalanobis distance.

When we fit a spatial model by maximum likelihood (Chapter 5), these two parts pull against each other. Making $\boldsymbol{\Sigma}$ larger makes any data look closer to the mean, which improves the distance term, but the determinant term charges for it. The estimate balances the two.

A two-location example shows the role of correlation. With $\boldsymbol{\mu} = \mathbf{0}$ and

$$
\boldsymbol{\Sigma} = \sigma^2\begin{bmatrix} 1 & r_{12} \\ r_{12} & 1 \end{bmatrix}, \qquad \lvert\boldsymbol{\Sigma}\rvert = \sigma^4(1 - r_{12}^2), \qquad \mathbf{y}'\boldsymbol{\Sigma}^{-1}\mathbf{y} = \frac{y_1^2 - 2r_{12}\,y_1y_2 + y_2^2}{\sigma^2(1 - r_{12}^2)}
$$

When the correlation $r_{12}$ is close to 1, data with $y_1 \approx y_2$ give a small distance, because two strongly correlated sites are expected to agree. Data with $y_1 \approx -y_2$ give a very large distance, because they contradict the correlation. The density rewards the model whose correlation structure matches the pattern in the data.

### 0.4.4 Tool 4: Maximum Likelihood

The recipe is the one from mathematical statistics: write the log-likelihood, differentiate with respect to the parameter, set the derivative to zero, and solve.

Here it is on a spatial problem. Suppose $\mathbf{Y} \sim N(\mathbf{0}, \sigma^2\mathbf{R})$, where the correlation matrix $\mathbf{R}$ is known and only the variance $\sigma^2$ is unknown. Since $\lvert\sigma^2\mathbf{R}\rvert = (\sigma^2)^n\lvert\mathbf{R}\rvert$ and $(\sigma^2\mathbf{R})^{-1} = \mathbf{R}^{-1}/\sigma^2$, Tool 3 gives

$$
\ell(\sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2}\log\lvert\mathbf{R}\rvert - \frac{\mathbf{y}'\mathbf{R}^{-1}\mathbf{y}}{2\sigma^2}
$$

Differentiating with respect to $\sigma^2$ and setting the result to zero,

$$
\frac{\partial\ell}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{\mathbf{y}'\mathbf{R}^{-1}\mathbf{y}}{2\sigma^4} = 0 \quad\Longrightarrow\quad \hat{\sigma}^2 = \frac{\mathbf{y}'\mathbf{R}^{-1}\mathbf{y}}{n}
$$

If the observations are independent, $\mathbf{R} = \mathbf{I}_n$ and this is the familiar $\frac{1}{n}\sum_i y_i^2$. With correlation, the sum of squares is replaced by a quadratic form in $\mathbf{R}^{-1}$, which down-weights observations that merely repeat their neighbors. Chapter 5 uses exactly this step to remove $\sigma^2$ from the likelihood, leaving a much smaller problem for the remaining parameters. Maximum likelihood also drives the GLM fitting of Chapter 9 and the point process models of Chapter 11.

### 0.4.5 Tool 5: The Normal Equations

Ordinary least squares chooses $\boldsymbol{\beta}$ to minimize the residual sum of squares $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})'(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$. Setting the gradient $-2\mathbf{X}'(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ to zero gives the normal equations

$$
\mathbf{X}'\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\mathbf{y}
$$

Geometrically, they say the residual vector is perpendicular to every column of $\mathbf{X}$. The fitted values are the projection $\mathbf{P}_{\mathbf{X}}\mathbf{y}$, with $\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$.

Spatial regression changes one thing. When the errors have covariance $\boldsymbol{\Sigma}$, the right criterion weights the residuals by $\boldsymbol{\Sigma}^{-1}$, minimizing $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$. The same calculus gives the generalized least squares equations

$$
\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{y}
$$

That inserted $\boldsymbol{\Sigma}^{-1}$ runs through Chapters 4 and 5, where it turns out to be the key both to spatial regression and to spatial prediction.

### 0.4.6 What the Book Builds on Top of These

The five tools carry most of the load, but not all of it. The book introduces the following as they are needed, always with a full derivation and a small numerical example. The table is not exhaustive. Appendix A collects the matrix tools (Cholesky and eigendecomposition facts, Lagrange multipliers, block inversion and the Woodbury identity, derivatives of $\log\lvert\boldsymbol{\Sigma}\rvert$ and $\boldsymbol{\Sigma}^{-1}$, Kronecker products) for reference; sparse matrices are collected in Appendix B, and iteratively reweighted least squares is developed in Chapter 9.

| Tool | What it does | First needed |
| :--- | :--- | :--- |
| Cholesky factorization $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}'$ | Simulates fields and evaluates likelihoods stably | Used from Chapter 0 (Section 0.3.2); explained in Chapter 2 |
| Expected value of a quadratic form | Finds the average of sums of squares such as $s^2$ | Chapter 1 |
| Eigendecomposition $\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ | Splits data into uncorrelated patterns; checks validity; computes determinants quickly | Chapter 1; Chapters 2 and 6 to 8 |
| Lagrange multipliers | Optimizes subject to an unbiasedness constraint | Chapter 4 |
| Block matrix inversion | Solves partitioned systems such as the kriging equations | Chapter 4 |
| Woodbury identity and matrix determinant lemma | Invert and take the determinant of a diagonal-plus-low-rank matrix cheaply | Chapter 13 (Section 13.2.2); Appendix A.1 |
| Derivatives of $\log\lvert\boldsymbol{\Sigma}\rvert$ and $\boldsymbol{\Sigma}^{-1}$ | Score equations and standard errors for covariance parameters | Chapter 5 |
| Newton-Raphson and Fisher scoring | Find maximum likelihood estimates iteratively | Chapter 5 (Section 5.4.3); Chapter 9 (Section 9.1.5) |
| Neumann series | Expands $(\mathbf{I} - \rho\mathbf{W})^{-1}$ as a sum of powers | Chapter 7 (Section 7.3.2) |
| Laplace approximation | Approximates an integral over random effects | Chapter 9 (Section 9.5) |
| Kronecker products and the $\operatorname{vec}$ operator | Handle separable space-time and multivariate covariances | Chapter 13 (Section 13.6.2); Appendix A.5 |
| Sparse matrices | Stores and solves large neighborhood models efficiently | Concept in Chapter 1 (Section 1.5.2); methods in Chapter 8 and Appendix B.3 |
| Iteratively reweighted least squares (IRLS) | Fits generalized linear models as a sequence of weighted regressions | Chapter 9 |
---

## 0.5 How This Book Works

### 0.5.1 What We Leave Out, and Why

Many treatments of spatial statistics begin with measure-theoretic probability, spectral theory, and functional analysis. This book does not. Every quantity we compute comes from a finite number of locations, and finite collections of random variables are fully described by vectors, matrices, and the multivariate normal distribution. That is enough to derive every estimator, predictor, and test in the book.

The price is generality in a few places. The clearest case is the validity of covariance functions: we can show that a function fails by finding a negative eigenvalue, but proving that a function works for every possible set of locations needs tools we do not develop. In those places the book says so plainly and takes the result from the literature. Appendix D points to where the deeper theory lives.

We also do not use spatial software packages. Every method is built from matrix operations in NumPy and SciPy. This is a teaching choice, not a judgment about software: after building a method yourself, you will understand what a package does with your data, what its options mean, and when its output should not be trusted. Nothing stops you from using spatial packages afterward.

### 0.5.2 The Rhythm of a Chapter

Each chapter follows the same sequence.

1. **A linear-model bridge.** The spatial problem is connected to a regression or GLM idea you already know, and any new terminology is translated into those terms.
2. **A full derivation.** The estimator, predictor, or test is derived with matrix algebra and calculus. No steps are skipped.
3. **A hand-sized example.** A configuration of 3 to 5 locations is worked through with every matrix printed in full, so you can check each step with a calculator.
4. **Code and a known-truth simulation.** The equations become a few lines of NumPy, and a simulation with known parameters shows whether the method recovers the truth.
5. **Exercises.** Analytical problems come first, then computational and simulation problems.

On a first reading of a chapter, it is reasonable to read the bridge, the hand-sized example, and the simulation, then return to the derivation. The derivations are complete so that you can always find out why a result holds, not so that you must absorb every line before moving on.

### 0.5.3 A Map of the Book

- **Part I (Chapters 1–2)** shows why ordinary regression fails with spatial data and builds the spatial linear model: covariance functions, the variogram, and simulation.
- **Part II (Chapters 3–5)** covers continuous spatial data: estimating the variogram, predicting by kriging, and fitting models by maximum likelihood and REML.
- **Part III (Chapters 6–9)** covers data on regions: neighborhood weights and Moran's $I$, simultaneous and conditional autoregressive models, and spatial GLMs for counts.
- **Part IV (Chapters 10–12)** covers point patterns: exploring them, fitting them as Poisson and logistic GLMs, and modeling clustering and inhibition.
- **Part V (Chapter 13)** shows how to compute with large spatial data sets by approximating $\boldsymbol{\Sigma}$ or its inverse.

The appendices collect the matrix algebra (Appendix A), Python patterns (Appendix B), a translation table between spatial terminology and linear-model terminology (Appendix C), and directions for further study, including pointers to the deeper theory this book leaves out (Appendix D). The notation guide is in the notation section at the front of the book.

---

## 0.6 Setting Up

The code in this book needs four libraries: NumPy for arrays and linear algebra, SciPy for optimization, special functions, and sparse matrices, SymPy for symbolic algebra, and matplotlib for plots. Run the following check. It prints the library versions, verifies Tool 1's simulation identity by building a small covariance matrix and its Cholesky factor, and uses SymPy to redo the maximum likelihood calculation of Section 0.4.4.

```python
import numpy as np
import scipy
import sympy as sp
import matplotlib

print("NumPy", np.__version__, "| SciPy", scipy.__version__,
      "| SymPy", sp.__version__, "| matplotlib", matplotlib.__version__)

# Tool 1: a Cholesky factor L satisfies L L' = Sigma
S = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
H = np.sqrt(((S[:, None, :] - S[None, :, :])**2).sum(axis=2))
Sigma = np.exp(-H / 0.5)
L = np.linalg.cholesky(Sigma)
print("L L' equals Sigma:", np.allclose(L @ L.T, Sigma))

# Tool 4: maximize l(s2) = -(n/2) log s2 - Q / (2 s2), where Q = y' R^{-1} y
n, Q, s2 = sp.symbols("n Q sigma2", positive=True)
loglik = -n / 2 * sp.log(s2) - Q / (2 * s2)
print("MLE of sigma^2:", sp.solve(sp.diff(loglik, s2), s2))
```

If the script prints `True` and `[Q/n]`, your environment is ready.

---

## 0.7 Warm-Up Exercises

Each exercise practices one of the five tools. If one gives you trouble, the matching section of Appendix A reviews the background.

1. **(Tool 1)** Two observations have $\operatorname{Var}(Y_1) = \operatorname{Var}(Y_2) = \sigma^2$ (here just the common variance of an observation) and correlation $r$. Show that $\operatorname{Var}\!\left(\frac{Y_1 + Y_2}{2}\right) = \frac{\sigma^2(1 + r)}{2}$ and that $n_{\text{eff}} = \frac{2}{1 + r}$. What are $n_{\text{eff}}$ for $r = 0$ and $r = 1$, and why do those values make sense?

2. **(Tool 1)** Now take $n$ observations with common variance $\sigma^2$ (again the generic variance of one observation) and the same correlation $r > 0$ between every pair. Show that $\operatorname{Var}(\bar{Y}) = \frac{\sigma^2\left(1 + (n - 1)r\right)}{n}$. Find the limit as $n \to \infty$ and interpret it: what is the most information about the mean that any number of such observations can provide?

3. **(Tool 2)** For the $3 \times 3$ matrix in Section 0.4.2, verify by hand that $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} = -2.4$ for $\mathbf{a} = (1, -1, -1)'$. Then compute its eigenvalues with `np.linalg.eigvalsh` and connect the negative eigenvalue to your answer.

4. **(Tool 2)** Show that the centering matrix $\mathbf{P}_c = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'$ is symmetric and satisfies $\mathbf{P}_c\mathbf{P}_c = \mathbf{P}_c$, and that $\mathbf{y}'\mathbf{P}_c\mathbf{y} = \sum_i (y_i - \bar{y})^2$.

5. **(Tool 3)** Using the two-location example in Section 0.4.3 with $\sigma^2 = 1$, evaluate the log density at $\mathbf{y} = (1, 1)'$ and at $\mathbf{y} = (1, -1)'$ for $r_{12} = 0, 0.5, 0.9$. Explain the pattern you see in terms of the determinant and distance terms.

6. **(Tool 4)** For the estimator $\hat{\sigma}^2 = \mathbf{y}'\mathbf{R}^{-1}\mathbf{y}/n$ of Section 0.4.4, use $\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] = \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma})$ for a mean-zero vector (derived in Section 1.1.1, so you may take it as given here) to show that $\hat{\sigma}^2$ is unbiased. Confirm the derivative calculation with SymPy.

7. **(Tool 5)** Let $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}'$. Multiply the model $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$, $\operatorname{Var}(\boldsymbol{\varepsilon}) = \boldsymbol{\Sigma}$, on the left by $\mathbf{L}^{-1}$. Use Tool 1 to show that the transformed errors are independent with variance 1, then apply the ordinary normal equations to the transformed model. Show that the result is the generalized least squares equations of Section 0.4.5.

8. **(Simulation)** Rerun the experiment of Section 0.3.2 with $\phi = 0.05$ and $\phi = 0.4$. Report the false rejection rate in each case and explain the trend using the effective sample sizes of Section 0.3.1.

---

## What Comes Next

Chapter 1 makes Section 0.3 precise. It derives the variance of the OLS estimator when the errors are spatially correlated, shows when and why standard errors are too small, and defines the effective sample size for the sample mean and shows that it depends on the question. Chapter 2 then builds the spatial linear model of Section 0.2 piece by piece and shows how to simulate from it. From there, each part of the book takes the same equation, $Y = \mu + u + \varepsilon$, to a new kind of data.
