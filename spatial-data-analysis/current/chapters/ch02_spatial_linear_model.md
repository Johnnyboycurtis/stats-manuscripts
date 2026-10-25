# Chapter 2: The Spatial Linear Model, Covariance, and Simulation

Chapter 1 took the covariance matrix $\boldsymbol{\Sigma}$ as given and showed what goes wrong when it is ignored. This chapter builds $\boldsymbol{\Sigma}$ from the ground up. Every later chapter fits, predicts with, or approximates this matrix, so it is worth understanding where each part of it comes from and which choices are allowed.

The chapter answers four questions. What assumptions turn the model $Y = \mu + u + \varepsilon$ into the formula $\boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n$? How can one map, observed once, tell us anything about how values at different locations are related? Which functions of distance can serve as covariance functions, and why do some reasonable-looking ones fail? And how do we turn independent random numbers into a simulated spatial field with exactly the covariance we want?

Along the way, the jargon of geostatistics (nugget, sill, range) gets translated into the variance components of the model, and the variogram appears as a picture of those components.

> **Chapter at a Glance**
>
> - **One Model Equation (§2.1):** The trend, the spatial random effect, and the nugget, and the assumptions that produce $\boldsymbol{\Sigma}$. The model is a linear mixed model.
> - **Stationarity (§2.2):** How assumptions about the process let one map act like many replicates, and the semivariogram $\gamma(\mathbf{h})$.
> - **Reading the Variogram (§2.3):** Nugget, sill, partial sill, and range, read off a curve.
> - **Validity (§2.4):** Why a covariance function must give a positive semi-definite matrix for every set of locations, a three-location counterexample, and ways to build valid models from valid pieces.
> - **Correlation Families (§2.5):** Exponential, Gaussian, spherical, and Matérn, with their practical ranges, smoothness, and numerical behavior.
> - **Anisotropy (§2.6):** When correlation reaches farther in one direction than another.
> - **Simulation (§2.7):** The Cholesky factorization by hand and in code, and the eigendecomposition as a fallback.
> - **Walkthrough and Lab (§2.8):** Valid and invalid matrices, and a simulated rent district.

---

## 2.1 One Model Equation

Chapter 1 used the model

$$
Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s})
$$

and the covariance matrix it produces. This section looks at each term in turn, states the assumptions each one carries, and derives $\boldsymbol{\Sigma}$ from them. The running example is the rent of apartments across a city.

### 2.1.1 The Large-Scale Mean

The first term is the part of the response explained by covariates:

$$
\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}
$$

This is ordinary regression. For rents, $\mathbf{x}(\mathbf{s})$ might contain an intercept, the apartment's size, its distance to downtown, or the coordinates of $\mathbf{s}$ themselves, which lets the mean rise smoothly in one direction across the city. The coefficients $\boldsymbol{\beta}$ are fixed, unknown numbers.

The mean captures variation that we can name and measure. Whatever spatial pattern is left over goes into the next term. Where to draw the line is a modeling choice. A smooth east-west rise in rents can be put in the mean, as a coordinate covariate, or left to the spatial random effect. Putting it in the mean usually makes the remaining spatial structure simpler, and Chapter 3 shows what goes wrong when a real trend is left out.

### 2.1.2 The Spatial Random Effect

The second term, $u(\mathbf{s})$, is everything spatial that the covariates miss. For rents, think of it as the unexplained desirability of a neighborhood: a block that has quietly become fashionable, a street with a good bakery, a view. These things are not in the data, but they are shared by nearby apartments.

We model $u$ as a random process with

$$
\operatorname{E}[u(\mathbf{s})] = 0, \qquad \operatorname{Cov}\!\left(u(\mathbf{s}_i), u(\mathbf{s}_j)\right) = \sigma^2\,r(h_{ij};\phi)
$$

The mean is zero because any average level belongs in $\mu$. The variance $\sigma^2$ says how large the neighborhood effects are, and the correlation function $r(h;\phi)$, with $r(0;\phi) = 1$, says how similar the effects are at two locations a distance $h$ apart. Chapter 1 used the exponential choice $r(h;\phi) = e^{-h/\phi}$; Section 2.5 introduces others.

### 2.1.3 The Nugget

The third term, $\varepsilon(\mathbf{s})$, is independent from location to location, with

$$
\operatorname{E}[\varepsilon(\mathbf{s})] = 0, \qquad \operatorname{Var}(\varepsilon(\mathbf{s})) = \tau^2, \qquad \operatorname{Cov}\!\left(\varepsilon(\mathbf{s}_i), \varepsilon(\mathbf{s}_j)\right) = 0 \text{ for } i \ne j
$$

It has two sources, and the model does not need to separate them.

- **Measurement error.** A listed rent may differ from what a tenant actually pays; a sensor has instrument noise.
- **Variation at scales finer than the data.** Two apartments in the same building differ in floor, light, and renovation. If our data have at most one apartment per block, nothing in them can show whether those differences are correlated over a few meters. At the scale of the data they look like independent noise.

In geostatistics this term is called the **nugget**, a name from gold mining, where a sample that happens to contain a nugget differs sharply from a sample taken right next to it.

A useful example of a large nugget is lead in household tap water. Much of the lead comes from each house's own service line, the pipe between the water main and the house, so two neighboring houses can have very different readings. Some of the variation is shared, because neighborhoods built in the same era used similar pipes, and that part belongs to $u$. But the house-to-house part, which is large, is nugget.

Schabenberger and Gotway, following Cressie, split the response into four terms: the mean, a smooth small-scale process, a micro-scale process, and measurement error. Our $u$ corresponds to their smooth small-scale process, and our $\varepsilon$ combines the last two. With data at the usual spacing the micro-scale and measurement-error parts cannot be told apart, so we treat them as one. The notation section gives the full correspondence.

### 2.1.4 Deriving the Covariance Matrix

At $n$ locations, the model in vector form is $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u} + \boldsymbol{\varepsilon}$. We add one more assumption: **$\mathbf{u}$ and $\boldsymbol{\varepsilon}$ are independent of each other.** The noise in a listing does not depend on how desirable the neighborhood is.

The variance of a sum of two independent random vectors is the sum of their variances, since the cross-covariance terms are zero. The mean $\mathbf{X}\boldsymbol{\beta}$ is fixed and adds nothing. So

$$
\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{Y}) = \operatorname{Var}(\mathbf{u}) + \operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n
$$

where $\mathbf{R}(\phi)$ has entries $R_{ij} = r(h_{ij};\phi)$. The spatial random effect fills the whole matrix; the nugget adds only to the diagonal.

**A hand-sized example.** Put three apartments on a street at positions 0, 0.1, and 0.5 (in kilometers), with $\sigma^2 = 1$, $\tau^2 = 0.2$, and the exponential correlation with $\phi = 0.2$. The distances are 0.1, 0.5, and 0.4, so the correlations of $u$ are $e^{-0.5} = 0.607$, $e^{-2.5} = 0.082$, and $e^{-2} = 0.135$, and

$$
\boldsymbol{\Sigma} = \begin{bmatrix} 1.2 & 0.607 & 0.082 \\ 0.607 & 1.2 & 0.135 \\ 0.082 & 0.135 & 1.2 \end{bmatrix}
$$

Each diagonal entry is $\sigma^2 + \tau^2 = 1.2$. The off-diagonal entries come from $u$ alone. Section 2.7 simulates from this matrix and checks the result.

It is often convenient to pull $\sigma^2$ out as a common factor. Define the **noise-to-signal ratio** $\delta = \tau^2/\sigma^2$. Then

$$
\boldsymbol{\Sigma} = \sigma^2\left(\mathbf{R}(\phi) + \delta\mathbf{I}_n\right) = \sigma^2\boldsymbol{\Psi}(\phi, \delta)
$$

where $\boldsymbol{\Psi}(\phi, \delta) = \mathbf{R}(\phi) + \delta\mathbf{I}_n$. In the example, $\delta = 0.2$. This form separates the overall scale $\sigma^2$ from the shape of the covariance, which depends only on $\phi$ and $\delta$. Chapter 5 uses it to estimate $\sigma^2$ in closed form, the same way Section 0.4.4 did.

### 2.1.5 Reading the Model as a Linear Mixed Model

If you have seen linear mixed models, this model is one. The standard mixed model is $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}_b\mathbf{b} + \boldsymbol{\varepsilon}$, with fixed effects $\boldsymbol{\beta}$, random effects $\mathbf{b}$, and a matrix $\mathbf{Z}_b$ that says which random effects each observation receives. (The subscript $b$ keeps this design matrix apart from the vector $\mathbf{Z}$ of independent standard normals that Section 2.7 uses for simulation.) In a study of students in schools, $\mathbf{Z}_b$ assigns each student the random intercept of their school, and the random intercepts are independent.

The spatial model is the case where each observation has its own random effect, so $\mathbf{Z}_b = \mathbf{I}_n$ and $\mathbf{b} = \mathbf{u}$, and the random effects are correlated, with covariance $\sigma^2\mathbf{R}(\phi)$, instead of independent. Everything familiar from mixed models applies: generalized least squares for $\boldsymbol{\beta}$ (Section 1.3.5), prediction of the random effects (Chapter 4), and restricted maximum likelihood for the variance parameters (Chapter 5).

---

## 2.2 Stationarity Expressed With Moments

Chapter 0 raised a puzzle. Classical statistics learns about a distribution from many independent draws, but we observe one map. How can one map tell us how values at two locations are related?

The answer is an assumption: that the rules generating the process are the same everywhere in the region. Then every pair of locations 100 meters apart is a draw from the same relationship, "two locations 100 meters apart," and one map contains many replicates of each separation. This section states the assumption precisely, in terms of means and covariances only.

Stationarity is an assumption about the random part of the model. Rents are not stationary: their mean changes with the covariates. The assumption applies to what remains after the mean is removed, $u(\mathbf{s}) + \varepsilon(\mathbf{s})$. For simplicity the definitions below are written for a process with constant mean.

### 2.2.1 Constant Mean

The first requirement is that the mean is the same everywhere:

$$
\operatorname{E}[Y(\mathbf{s})] = \mu \text{ for all } \mathbf{s}, \qquad \text{so} \qquad \operatorname{E}[\mathbf{Y}] = \mu\mathbf{1}_n
$$

For a model with covariates, this is the statement that $\operatorname{E}[Y(\mathbf{s}) - \mu(\mathbf{s})] = 0$ everywhere, which holds by construction.

### 2.2.2 Second-Order Stationarity

Write $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$ for the **lag vector** between two locations: the displacement from one to the other, which carries both a distance and a direction. A process is **second-order stationary** if its mean is constant and the covariance between any two locations depends only on the lag between them:

$$
\operatorname{Cov}\!\left(Y(\mathbf{s}), Y(\mathbf{s} + \mathbf{h})\right) = C(\mathbf{h}) \quad \text{for all } \mathbf{s}
$$

The function $C(\mathbf{h})$ is the **covariance function**. It does not matter where in the region a pair sits, only how far apart and in which direction. Setting $\mathbf{h} = \mathbf{0}$ gives $C(\mathbf{0}) = \operatorname{Var}(Y(\mathbf{s}))$, the same variance at every location.

In Chapter 1 the covariance depended only on the distance $h = \lVert\mathbf{h}\rVert$, not the direction. That stronger assumption is **isotropy** (Section 2.3.5), and Chapter 1's $C(h)$ is the isotropic case of $C(\mathbf{h})$.

### 2.2.3 Intrinsic Stationarity

A weaker assumption looks at differences instead of individual values. A process is **intrinsically stationary** if its increments have mean zero and a variance that depends only on the lag:

$$
\operatorname{E}\!\left[Y(\mathbf{s} + \mathbf{h}) - Y(\mathbf{s})\right] = 0, \qquad \operatorname{Var}\!\left(Y(\mathbf{s} + \mathbf{h}) - Y(\mathbf{s})\right) = 2\gamma(\mathbf{h})
$$

The function $\gamma(\mathbf{h})$ is the **semivariogram**: half the variance of the difference between two values separated by $\mathbf{h}$. (The factor of one half makes the formulas in the next subsection tidy. The full $2\gamma(\mathbf{h})$ is called the variogram, and in practice the two words are used loosely.)

Intrinsic stationarity is weaker because a process can have well-behaved differences while its own variance is not constant. The simplest example is a random walk along a line. Let $Y_0 = 0$ and $Y_t = Y_{t-1} + Z_t$, where the steps $Z_t$ are independent with mean 0 and variance 1. Then for $t > s$,

$$
Y_t - Y_s = Z_{s+1} + \dots + Z_t, \qquad \operatorname{Var}(Y_t - Y_s) = t - s
$$

so $\gamma(h) = h/2$ depends only on the lag $h = t - s$, and the process is intrinsically stationary. But $\operatorname{Var}(Y_t) = t$ grows with $t$, so no single covariance function $C(h)$ describes it, and it is not second-order stationary. Processes like this wander without returning to a fixed level. They are rare in the applications of this book, but they explain why some variograms rise forever instead of leveling off.

### 2.2.4 The Bridge Between Covariance and Variogram

For a second-order stationary process, the two descriptions carry the same information. Expanding the variance of a difference,

$$
\begin{aligned}
2\gamma(\mathbf{h}) &= \operatorname{Var}\!\left(Y(\mathbf{s} + \mathbf{h}) - Y(\mathbf{s})\right) \\
&= \operatorname{Var}(Y(\mathbf{s} + \mathbf{h})) + \operatorname{Var}(Y(\mathbf{s})) - 2\operatorname{Cov}\!\left(Y(\mathbf{s} + \mathbf{h}), Y(\mathbf{s})\right) \\
&= C(\mathbf{0}) + C(\mathbf{0}) - 2C(\mathbf{h})
\end{aligned}
$$

and dividing by 2,

$$
\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})
$$

The semivariogram is the covariance function turned upside down: it starts at zero and rises toward $C(\mathbf{0})$ as $C(\mathbf{h})$ decays toward zero.

Why work with the variogram at all, if the covariance function says the same thing? Because a variogram can be estimated without knowing the mean. Differences $Y(\mathbf{s}_i) - Y(\mathbf{s}_j)$ cancel any constant mean, so averaging squared differences over pairs at similar lags estimates $\gamma$ directly. That is the subject of Chapter 3, and it is how the replication described at the start of this section is put to work.

---

## 2.3 Reading the Variogram: Decoding the Jargon on the Curve

For the model of Section 2.1 with isotropic correlation, the semivariogram is

$$
\gamma(h) = \begin{cases} 0 & h = 0 \\ \tau^2 + \sigma^2\left(1 - r(h;\phi)\right) & h > 0 \end{cases}
$$

This follows from $\gamma(h) = C(0) - C(h)$ with $C(0) = \sigma^2 + \tau^2$ and $C(h) = \sigma^2 r(h;\phi)$ for $h > 0$. The geostatistical terms describe features of this curve. The table below evaluates it for the exponential model with $\sigma^2 = 1$, $\tau^2 = 0.2$, and $\phi = 0.2$.

```python
import numpy as np
from scipy import special, optimize


def gamma_exp(h, sigma2, tau2, phi):
    """Semivariogram of the exponential model with a nugget."""
    h = np.asarray(h, dtype=float)
    return np.where(h > 0, tau2 + sigma2 * (1 - np.exp(-h / phi)), 0.0)


for h in (0.0, 0.01, 0.1, 0.2, 0.6, 1.0):
    print(f"h={h:4.2f}  gamma={gamma_exp(h, 1.0, 0.2, 0.2):.4f}")
```

| $h$ | 0 | 0.01 | 0.1 | 0.2 | 0.6 | 1.0 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\gamma(h)$ | 0 | 0.2488 | 0.5935 | 0.8321 | 1.1502 | 1.1933 |

### 2.3.1 Nugget: The Jump at the Origin

At $h = 0$ the semivariogram is 0, since a value does not differ from itself. But just above zero it is already near $\tau^2 = 0.2$: at $h = 0.01$ it is 0.2488. The curve jumps at the origin, and the height of the jump is the **nugget** $\tau^2$. Two measurements taken very close together still differ, by the independent noise of Section 2.1.3.

### 2.3.2 Sill: Where the Curve Levels Off

At large distances $r(h;\phi) \to 0$, so $\gamma(h) \to \tau^2 + \sigma^2 = C(0)$. This level is the **sill**: the total variance of $Y$ at a single location. In the table, $\gamma$ approaches 1.2.

### 2.3.3 Partial Sill: The Rise Above the Nugget

The rise from the nugget to the sill is $\sigma^2$, the **partial sill**: the variance of the spatial random effect $u$. The ratio of partial sill to sill, here $1/1.2 \approx 0.83$, is the share of the variance that is spatially structured.

### 2.3.4 Range: How Far Correlation Reaches

The **range parameter** $\phi$ sets the distance scale over which correlation decays. In this book $\phi$ is always a distance: doubling $\phi$ doubles the reach of the correlation. Some texts write the exponential as $e^{-\phi h}$ instead, so that $\phi$ is a rate of decay and larger values mean shorter reach. When reading other sources, check which convention they use.

Because many correlation functions approach zero without reaching it, a more practical summary is the **practical range**: the distance at which the correlation falls to 0.05, so that the semivariogram has covered 95% of the climb from nugget to sill. For the exponential model, $e^{-h/\phi} = 0.05$ gives $h = \phi\ln 20 \approx 3\phi$. In the table, $3\phi = 0.6$ and $\gamma(0.6) = 1.1502 = 0.2 + 0.95 \times 1$. Section 2.5 gives the practical range of each family.

### 2.3.5 Isotropy

So far the semivariogram has depended only on distance. A process whose covariance depends only on $h = \lVert\mathbf{h}\rVert$, not on the direction of $\mathbf{h}$, is **isotropic**. Isotropy is a convenient default, but it fails when, for example, rents stay similar for a long way along a riverfront but change quickly as you move away from the river. Section 2.6 handles that case, and Chapter 3 shows how to detect it from data.

---

## 2.4 What Makes a Covariance Matrix Valid?

Any function that decreases from 1 toward 0 looks like a plausible correlation function. This section shows that plausibility is not enough.

### 2.4.1 The Requirement

Section 0.4.2 showed that every covariance matrix is positive semi-definite: for any vector $\mathbf{a}$, $\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} = \operatorname{Var}(\mathbf{a}'\mathbf{Y}) \ge 0$. A covariance function must therefore produce a positive semi-definite matrix **for every possible set of locations**, not just for the locations in one data set. If some configuration of points gives a matrix with a negative eigenvalue, then some linear combination of the values has a negative "variance," and the function cannot be the covariance of any process.

### 2.4.2 Why Naive Functions Fail: A Three-Location Counterexample

Suppose we propose a simple rule for rents: apartments within 1 km of each other have correlation 0.9, and apartments farther apart are uncorrelated. In symbols, $r(h) = 0.9$ for $0 < h < 1$ and $r(h) = 0$ for $h \ge 1$.

Place three apartments A, B, C on a street at 0, 0.6, and 1.2 km. A and B are 0.6 km apart, as are B and C, but A and C are 1.2 km apart. The rule gives

$$
\mathbf{R} = \begin{bmatrix} 1 & 0.9 & 0 \\ 0.9 & 1 & 0.9 \\ 0 & 0.9 & 1 \end{bmatrix}
$$

Take $\mathbf{a} = (1, -\sqrt{2}, 1)'$. Then

$$
\mathbf{a}'\mathbf{R}\mathbf{a} = (1 + 2 + 1) + 2(1)(-\sqrt{2})(0.9) + 2(-\sqrt{2})(1)(0.9) + 2(1)(1)(0) = 4 - 3.6\sqrt{2} \approx -1.09
$$

a negative variance. The eigenvalues of $\mathbf{R}$ are $1 - 0.9\sqrt{2} \approx -0.27$, $1$, and $1 + 0.9\sqrt{2} \approx 2.27$, and $\mathbf{a}$ is (up to scaling) the eigenvector of the negative one.

The intuition is a kind of transitivity. If A is strongly tied to B, and B is strongly tied to C, then A and C cannot be unrelated. The rule demands that they are, and no random process can satisfy all three demands at once. Exercise 4 shows the rule fails whenever its correlation is above $1/\sqrt{2}$.

```python
C_step = np.array([[1.0, 0.9, 0.0],
                   [0.9, 1.0, 0.9],
                   [0.0, 0.9, 1.0]])
print("eigenvalues:", np.round(np.linalg.eigvalsh(C_step), 4))
a = np.array([1.0, -np.sqrt(2), 1.0])
print("a' C a =", round(a @ C_step @ a, 4))
```

### 2.4.3 Building Valid Models From Valid Pieces

Checking every configuration of points is impossible, so how do we know any function is valid? One route is to build new covariance functions from ones already known to be valid, using operations that preserve validity. Each operation below is justified by exhibiting a process that has the new covariance, which is enough: a covariance of an actual process is automatically positive semi-definite.

- **Sums.** If $u_1$ and $u_2$ are independent processes with covariance functions $C_1$ and $C_2$, then $u_1 + u_2$ has covariance $C_1 + C_2$, by the argument of Section 2.1.4. So the sum of two valid covariance functions is valid. The model $\sigma^2 r(h;\phi)$ plus a nugget is an example.
- **Products.** If $u_1$ and $u_2$ are independent with mean zero, the product $u_1(\mathbf{s})u_2(\mathbf{s})$ has covariance

  $$
  \operatorname{E}\!\left[u_1(\mathbf{s}_i)u_2(\mathbf{s}_i)u_1(\mathbf{s}_j)u_2(\mathbf{s}_j)\right] = \operatorname{E}\!\left[u_1(\mathbf{s}_i)u_1(\mathbf{s}_j)\right]\operatorname{E}\!\left[u_2(\mathbf{s}_i)u_2(\mathbf{s}_j)\right] = C_1 C_2
  $$

  So the product of two valid covariance functions is valid. Chapter 13 uses this to make covariance matrices sparse.
- **Positive multiples.** If $C$ is valid and $c > 0$, then $cC$ is the covariance of $\sqrt{c}\,u$.
- **Gramians.** For any matrix $\mathbf{B}$, the matrix $\mathbf{B}\mathbf{B}'$ is positive semi-definite, because $\mathbf{a}'\mathbf{B}\mathbf{B}'\mathbf{a} = \lVert\mathbf{B}'\mathbf{a}\rVert^2 \ge 0$. It is the covariance of $\mathbf{B}\mathbf{Z}$ with $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$, by Tool 1. Chapter 13 builds large-scale models this way.
- **Changing the coordinates.** If $C(\mathbf{h})$ is valid in the plane, then so is $C(\mathbf{B}\mathbf{h})$ for any fixed $2 \times 2$ matrix $\mathbf{B}$: it is the covariance of the process $u(\mathbf{B}\mathbf{s})$, observed at transformed locations. Section 2.6 uses this.

### 2.4.4 What Finite Checks Can and Cannot Show

Computing the eigenvalues of $\mathbf{R}$ for a particular set of locations is a one-sided test. A negative eigenvalue proves a function invalid, as in Section 2.4.2. A set of nonnegative eigenvalues proves only that this configuration is fine.

The "tent" function $r(h) = \max(0, 1 - h)$ shows how misleading a few checks can be. It is a valid correlation function for points on a line. In the plane, small configurations of points usually pass the eigenvalue check, yet on a $12 \times 12$ grid with spacing 0.35 the smallest eigenvalue is $-0.13$ (Section 2.8.1). The function is not valid in two dimensions, and only a large enough configuration reveals it.

Proving that a function is valid for every configuration requires a characterization of valid covariance functions through their Fourier transforms (Bochner's theorem), which is outside the scope of this book. The families in Section 2.5 are all proven valid in the plane, and we take those results from the literature. In practice: use a named family, combine families with the operations of Section 2.4.3, and treat any home-made function with suspicion.

---

## 2.5 Parametric Correlation Families

This section introduces the correlation functions used in the rest of the book. Each is valid in two and three dimensions. Each has a range parameter $\phi$, but $\phi$ has a different meaning in each family, so comparisons between families are best made through the practical range.

The code below defines all four families and computes their practical ranges. For the Matérn family it uses `scipy.special.kv`, the modified Bessel function of the second kind, and `scipy.special.gamma`, the gamma function.

```python
def r_exponential(h, phi):
    return np.exp(-h / phi)


def r_gaussian(h, phi):
    return np.exp(-(h / phi)**2)


def r_spherical(h, phi):
    x = np.minimum(h / phi, 1.0)
    return 1 - 1.5 * x + 0.5 * x**3


def r_matern(h, phi, nu):
    """Matern correlation in the h/phi parameterization."""
    x = np.asarray(h, dtype=float) / phi
    out = np.ones_like(x)
    pos = x > 0
    xp = x[pos]
    out[pos] = 2**(1 - nu) / special.gamma(nu) * xp**nu * special.kv(nu, xp)
    return out


h = np.linspace(0.0, 2.0, 401)
closed = {0.5: np.exp(-h),
          1.5: (1 + h) * np.exp(-h),
          2.5: (1 + h + h**2 / 3) * np.exp(-h)}
for nu, r_closed in closed.items():
    err = np.max(np.abs(r_matern(h, 1.0, nu) - r_closed))
    print(f"nu={nu}: largest difference from closed form = {err:.1e}")

families = {"exponential": lambda h: r_exponential(h, 1.0),
            "gaussian": lambda h: r_gaussian(h, 1.0),
            "spherical": lambda h: r_spherical(h, 1.0),
            "matern 3/2": lambda h: r_matern(np.array([h]), 1.0, 1.5)[0],
            "matern 5/2": lambda h: r_matern(np.array([h]), 1.0, 2.5)[0]}
for name, r in families.items():
    hr = optimize.brentq(lambda h: r(h) - 0.05, 1e-6, 10.0)
    print(f"{name:12s} practical range = {hr:.3f} phi")
```

| Family | Correlation $r(h;\phi)$ | Practical range |
| :--- | :--- | :--- |
| Exponential | $e^{-h/\phi}$ | $2.996\,\phi$ |
| Gaussian | $e^{-(h/\phi)^2}$ | $1.731\,\phi$ |
| Spherical | $1 - \tfrac{3}{2}(h/\phi) + \tfrac{1}{2}(h/\phi)^3$ for $h \le \phi$; 0 beyond | $0.811\,\phi$ (the true range is $\phi$: exactly zero beyond) |
| Matérn, $\nu = 3/2$ | $(1 + h/\phi)\,e^{-h/\phi}$ | $4.744\,\phi$ |
| Matérn, $\nu = 5/2$ | $\left(1 + h/\phi + \tfrac{1}{3}(h/\phi)^2\right)e^{-h/\phi}$ | $5.919\,\phi$ |

### 2.5.1 Exponential and Gaussian: Smoothness Versus Numerical Trouble

The **exponential** model has been our default since Chapter 1. Near $h = 0$, $1 - e^{-h/\phi} \approx h/\phi$, so the semivariogram rises linearly from the origin. As Section 2.5.3 explains, that linear start means the simulated surfaces are rough: continuous, but jagged at every scale.

The **Gaussian** model, $e^{-(h/\phi)^2}$, starts flat: $1 - e^{-(h/\phi)^2} \approx (h/\phi)^2$, a quadratic rise. Its surfaces are extremely smooth, smoother than almost any real phenomenon, and the smoothness causes a numerical problem. When locations are close together relative to $\phi$, the rows of $\mathbf{R}$ for neighboring locations are nearly identical, and the matrix is nearly singular.

The **condition number** of a matrix, the ratio of its largest to smallest eigenvalue, measures how close it is to singular. Double-precision arithmetic carries about 16 significant digits, so a condition number near $10^{16}$ means the smallest eigenvalue is lost in rounding error. The code compares the two models on the $20 \times 20$ grid of Chapter 1 with $\phi = 0.2$.

```python
def dist(S):
    """Matrix of Euclidean distances between the rows of S."""
    q = np.sum(S**2, axis=1)
    H2 = q[:, None] - 2 * S @ S.T + q[None, :]
    H = np.sqrt(np.maximum(H2, 0.0))
    np.fill_diagonal(H, 0.0)
    return H


def grid(m):
    """Centers of an m-by-m grid of cells in the unit square."""
    g = (np.arange(m) + 0.5) / m
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


H = dist(grid(20))
for name, R in (("exponential", r_exponential(H, 0.2)),
                ("gaussian", r_gaussian(H, 0.2))):
    print(f"{name:12s} condition number = {np.linalg.cond(R):.2e}")
    try:
        np.linalg.cholesky(R)
        print("  Cholesky succeeded")
    except np.linalg.LinAlgError:
        print("  Cholesky failed")
R_g = r_gaussian(H, 0.2)
for jitter in (1e-10, 1e-8, 1e-6):
    try:
        np.linalg.cholesky(R_g + jitter * np.eye(len(H)))
        ok = "succeeded"
    except np.linalg.LinAlgError:
        ok = "failed"
    print(f"jitter {jitter:.0e}: Cholesky {ok}")
```

The exponential matrix has condition number $5.82 \times 10^2$. The Gaussian matrix has condition number $4.95 \times 10^{17}$, and the Cholesky factorization of Section 2.7 fails on it. Adding a tiny multiple of the identity, a **jitter** of $10^{-10}$, is enough to make it succeed. A jitter is a small artificial nugget. A real nugget $\tau^2 > 0$ does the same job, which is one practical reason most fitted models include one.

Because of its unrealistic smoothness and its numerical fragility, the Gaussian model is rarely the best choice for real data. When a smooth model is wanted, the Matérn family with $\nu = 3/2$ or $5/2$ is better behaved.

### 2.5.2 The Spherical Model: Exactly Zero Beyond the Range

The **spherical** model

$$
r(h;\phi) = \begin{cases} 1 - \tfrac{3}{2}\dfrac{h}{\phi} + \tfrac{1}{2}\left(\dfrac{h}{\phi}\right)^3 & 0 \le h \le \phi \\ 0 & h > \phi \end{cases}
$$

reaches zero exactly at $h = \phi$ and stays there. Here $\phi$ is the true range: locations farther apart than $\phi$ are uncorrelated. Its practical range, where the correlation falls to 0.05, is shorter, $0.811\,\phi$, because the correlation reaches 0.05 well before it reaches zero (the table in this section gives the value, found by solving $1 - \tfrac{3}{2}x + \tfrac{1}{2}x^3 = 0.05$ numerically). Covariance functions that vanish beyond a finite distance are said to have **compact support**. Their covariance matrices contain exact zeros, which Chapter 13 exploits for large data sets.

The spherical model is valid in one, two, and three dimensions, but not in higher dimensions. As Section 2.4.4 showed with the tent function, validity can depend on the dimension, and the spherical model is a safe choice only because maps have two dimensions.

### 2.5.3 The Matérn Family

The **Matérn** family adds a parameter $\nu > 0$ that controls smoothness. In the parameterization used in this book,

$$
r(h;\phi,\nu) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{h}{\phi}\right)^{\nu}K_\nu\!\left(\frac{h}{\phi}\right)
$$

where $\Gamma(\cdot)$ is the gamma function and $K_\nu$ is the modified Bessel function of the second kind. We do not need the theory of Bessel functions. SciPy evaluates $K_\nu$, and for the three values of $\nu$ used most often, the formula reduces to elementary functions:

$$
\begin{aligned}
\nu = \tfrac{1}{2}: &\quad r = e^{-h/\phi} \\
\nu = \tfrac{3}{2}: &\quad r = \left(1 + \tfrac{h}{\phi}\right)e^{-h/\phi} \\
\nu = \tfrac{5}{2}: &\quad r = \left(1 + \tfrac{h}{\phi} + \tfrac{1}{3}\left(\tfrac{h}{\phi}\right)^2\right)e^{-h/\phi}
\end{aligned}
$$

The code above confirms these closed forms against the Bessel-function formula; the largest difference over $0 \le h \le 2\phi$ is below $10^{-15}$. The case $\nu = 1/2$ is the exponential model, so the Matérn family contains our default as a special case.

**What $\nu$ controls.** The smoothness of a simulated surface is governed by how fast the semivariogram rises near the origin. If $1 - r(h)$ grows like $h$ for small $h$, as for the exponential, then the expected squared difference between values a small distance apart is proportional to that distance, and the surface is jagged. If $1 - r(h)$ grows like $h^2$, nearby values differ much less, and the surface is smooth. Expanding $e^{-x} = 1 - x + \tfrac{1}{2}x^2 - \tfrac{1}{6}x^3 + \dots$ with $x = h/\phi$,

$$
\begin{aligned}
\nu = \tfrac{1}{2}: &\quad 1 - r = 1 - e^{-x} = x - \tfrac{1}{2}x^2 + \dots \\
\nu = \tfrac{3}{2}: &\quad 1 - r = 1 - (1 + x)e^{-x} = \tfrac{1}{2}x^2 - \tfrac{1}{3}x^3 + \dots
\end{aligned}
$$

The linear term disappears for $\nu = 3/2$, and larger $\nu$ makes the start flatter still. Figure 2.2 in Section 2.8.2 shows the result: an exponential transect is jagged, while a Matérn transect with $\nu = 5/2$ and the same practical range is smooth.

**Choosing $\nu$.** Values between $1/2$ and $5/2$ cover most applications. Estimating $\nu$ from data is possible but difficult, and Chapter 5 compares fits across a few fixed values. Recall also the warning of Section 1.6.4: with great-circle distance on the sphere, only $\nu \le 1/2$ is safe.

---

## 2.6 Geometric Anisotropy Through Coordinate Transformations

Suppose rents along a river stay similar for several kilometers, because the whole riverfront is desirable, but drop off quickly as you move inland. Correlation then reaches farther east-west, along the river, than north-south. A single distance cannot capture this.

The fix is to measure distance differently in different directions. For a symmetric positive-definite $2 \times 2$ matrix $\mathbf{G}$, define the distance between $\mathbf{s}_i$ and $\mathbf{s}_j$ as

$$
h_{ij}(\mathbf{G}) = \sqrt{(\mathbf{s}_i - \mathbf{s}_j)'\mathbf{G}(\mathbf{s}_i - \mathbf{s}_j)}
$$

and use it in place of ordinary distance in any correlation function. With $\mathbf{G} = \mathbf{I}_2$ this is ordinary distance. With $\mathbf{G} = \operatorname{diag}(1, 4)$, a north-south separation counts double, so correlation decays twice as fast in that direction. This is **geometric anisotropy**: the contours of equal correlation are ellipses instead of circles.

```python
def dist_aniso(S, G):
    """Distances sqrt((s_i - s_j)' G (s_i - s_j)) between rows of S."""
    D = S[:, None, :] - S[None, :, :]
    return np.sqrt(np.einsum("ijk,kl,ijl->ij", D, G, D))


G = np.diag([1.0, 4.0])
pts = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]])
Ha = dist_aniso(pts, G)
print("distance east:", round(Ha[0, 1], 3), " north:", round(Ha[0, 2], 3))
print("correlation east:", round(r_exponential(Ha[0, 1], 0.2), 4),
      " north:", round(r_exponential(Ha[0, 2], 0.2), 4))
```

Two apartments 0.2 km apart east-west have distance 0.2 and, with the exponential model and $\phi = 0.2$, correlation 0.3679. Two apartments 0.2 km apart north-south have distance 0.4 and correlation 0.1353.

**Why this stays valid.** Write the eigendecomposition $\mathbf{G} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ from Section 1.3.3, with positive eigenvalues, and let $\mathbf{G}^{1/2} = \mathbf{U}\boldsymbol{\Lambda}^{1/2}\mathbf{U}'$. Then $h_{ij}(\mathbf{G}) = \lVert\mathbf{G}^{1/2}\mathbf{s}_i - \mathbf{G}^{1/2}\mathbf{s}_j\rVert$, the ordinary distance between transformed locations. So an anisotropic model is an isotropic model applied after rotating the map to line up with the directions $\mathbf{U}$ and stretching it by $\boldsymbol{\Lambda}^{1/2}$. By the last bullet of Section 2.4.3, validity is preserved.

The eigenvectors of $\mathbf{G}$ point along the directions of slowest and fastest decay, and the eigenvalues set the stretching. A $\mathbf{G}$ that is not diagonal describes an ellipse tilted away from the coordinate axes, for a river that does not run due east. Chapter 3 shows how to detect anisotropy in data with directional variograms.

---

## 2.7 Simulating Spatial Random Fields From First Principles

Chapters 0 and 1 simulated spatial data with the line `L @ rng.standard_normal(n)` and promised an explanation. Here it is.

### 2.7.1 The Cholesky Factorization

By Tool 1, if $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$, then $\boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$ has mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$. Any $\mathbf{L}$ with $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$ works. The **Cholesky factorization** finds the unique such $\mathbf{L}$ that is lower triangular with positive diagonal entries. It exists whenever $\boldsymbol{\Sigma}$ is positive definite.

**Two locations by hand.** With $\boldsymbol{\Sigma} = \begin{bmatrix} 1 & r \\ r & 1 \end{bmatrix}$, write $\mathbf{L} = \begin{bmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{bmatrix}$ and match entries of

$$
\mathbf{L}\mathbf{L}' = \begin{bmatrix} L_{11}^2 & L_{11}L_{21} \\ L_{11}L_{21} & L_{21}^2 + L_{22}^2 \end{bmatrix} = \begin{bmatrix} 1 & r \\ r & 1 \end{bmatrix}
$$

The first entry gives $L_{11} = 1$, the off-diagonal gives $L_{21} = r$, and the last gives $L_{22} = \sqrt{1 - r^2}$. The simulated values are

$$
Y_1 = Z_1, \qquad Y_2 = rZ_1 + \sqrt{1 - r^2}\,Z_2
$$

The second value copies a fraction $r$ of the first and adds fresh noise, scaled so that its variance is still 1. With $r = 0.6$, $L_{21} = 0.6$ and $L_{22} = 0.8$.

**The general algorithm.** The same matching, done one column at a time, gives every entry of $\mathbf{L}$:

$$
L_{jj} = \sqrt{\Sigma_{jj} - \sum_{k<j}L_{jk}^2}, \qquad L_{ij} = \frac{1}{L_{jj}}\left(\Sigma_{ij} - \sum_{k<j}L_{ik}L_{jk}\right) \quad \text{for } i > j
$$

Each value $Y_j$ is built from the earlier values plus fresh noise. The quantity under the square root is the variance of $Y_j$ that the earlier values cannot explain, which must be positive; that is where the factorization fails if $\boldsymbol{\Sigma}$ is not positive definite, or is so close to singular that rounding makes it look otherwise. The work grows like $n^3/3$ arithmetic operations, which is fast for a few thousand locations and slow beyond that (Chapter 13).

The code below checks the two-location case, then simulates 100,000 draws for the three apartments of Section 2.1.4 and compares the sample covariance with $\boldsymbol{\Sigma}$.

```python
r = 0.6
Sigma2 = np.array([[1.0, r], [r, 1.0]])
print(np.round(np.linalg.cholesky(Sigma2), 4))

rng = np.random.default_rng(42)
S3 = np.array([[0.0, 0.0], [0.1, 0.0], [0.5, 0.0]])
Sigma3 = 1.0 * r_exponential(dist(S3), 0.2) + 0.2 * np.eye(3)
L3 = np.linalg.cholesky(Sigma3)
Ysim = (L3 @ rng.standard_normal((3, 100_000))).T
print("target:\n", np.round(Sigma3, 3))
print("sample covariance:\n", np.round(np.cov(Ysim, rowvar=False), 3))

lam, U = np.linalg.eigh(R_g)
print("smallest eigenvalue of the Gaussian R:", f"{lam.min():.1e}")
L_spec = U * np.sqrt(np.maximum(lam, 0.0))
print("L L' reproduces R:", np.allclose(L_spec @ L_spec.T, R_g, atol=1e-8))
```

The sample covariance agrees with the target to within simulation error: for example, 0.610 against 0.607 for the first pair and 0.141 against 0.135 for the second and third.

### 2.7.2 Spectral Simulation When $\boldsymbol{\Sigma}$ Is Nearly Singular

When the Cholesky factorization fails because $\boldsymbol{\Sigma}$ is nearly singular, as for the Gaussian model in Section 2.5.1, the eigendecomposition gives another square root. If $\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$, then $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}^{1/2}$ satisfies $\mathbf{L}\mathbf{L}' = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}' = \boldsymbol{\Sigma}$.

The advantage is that we can see and repair the problem. For the Gaussian matrix of Section 2.5.1, the smallest computed eigenvalue is $-4.2 \times 10^{-15}$: slightly negative, which is impossible for a true covariance matrix and is pure rounding error. Setting such values to zero before taking square roots gives a valid factor, and the code confirms that it reproduces $\mathbf{R}$. The cost is speed: an eigendecomposition takes several times longer than a Cholesky factorization. In practice, a small jitter or nugget with Cholesky is the usual fix, and the spectral method is the fallback.

### 2.7.3 Simulating the Pieces Separately

Simulating $\mathbf{Y}$ directly from $\boldsymbol{\Sigma} = \sigma^2\mathbf{R} + \tau^2\mathbf{I}_n$ gives the right distribution, but it hides the two random parts. For teaching, and for checking the prediction methods of Chapter 4, it is better to simulate them separately:

$$
\mathbf{u} = \mathbf{L}_u\mathbf{Z}_1, \qquad \boldsymbol{\varepsilon} = \tau\,\mathbf{Z}_2, \qquad \mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u} + \boldsymbol{\varepsilon}
$$

where $\mathbf{L}_u\mathbf{L}_u' = \sigma^2\mathbf{R}(\phi)$ and $\mathbf{Z}_1$, $\mathbf{Z}_2$ are independent standard normal vectors. The result has exactly the same distribution, by Section 2.1.4, and we also know the noise-free surface $\mathbf{X}\boldsymbol{\beta} + \mathbf{u}$. Chapter 4 predicts that surface, and having the truth lets us check the predictions.

---

## 2.8 Walkthrough and Lab 2

### 2.8.1 Valid and Invalid Matrices

The first part of the lab collects the validity checks of Section 2.4. The four-apartment block of Chapter 1, with the exponential model, gives a valid matrix. The step-function rule of Section 2.4.2, applied to four apartments 0.6 km apart along a street, gives an invalid one. The tent function passes on a line and fails on a grid.

```python
S4 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
print("block, exponential:",
      np.round(np.linalg.eigvalsh(r_exponential(dist(S4), 1.0)), 4))
t4 = np.array([0.0, 0.6, 1.2, 1.8])
C4 = np.where(np.abs(t4[:, None] - t4[None, :]) < 1.0, 0.9, 0.0)
np.fill_diagonal(C4, 1.0)
print("line, step function:", np.round(np.linalg.eigvalsh(C4), 4))


def r_tent(h):
    return np.maximum(0.0, 1.0 - h)


t = np.arange(50) * 0.3
print("tent on a line, smallest eigenvalue:",
      round(np.linalg.eigvalsh(r_tent(np.abs(t[:, None] - t[None, :])))[0], 4))
g = np.arange(12) * 0.35
xx, yy = np.meshgrid(g, g)
S_grid = np.column_stack([xx.ravel(), yy.ravel()])
print("tent on a 12 x 12 grid, smallest eigenvalue:",
      round(np.linalg.eigvalsh(r_tent(dist(S_grid)))[0], 4))
```

| Matrix | Eigenvalues (smallest first) | Valid? |
| :--- | :--- | :--- |
| Block of four, exponential | 0.5074, 0.7569, 0.7569, 1.9789 | Yes |
| Four on a line, step function | $-0.4562$, 0.4438, 1.5562, 2.4562 | No |
| Tent, 50 points on a line | smallest 0.084 | Passes |
| Tent, $12 \times 12$ grid | smallest $-0.1331$ | No |

### 2.8.2 A Simulated Rent District

Now put the pieces together. Simulate monthly rents, in hundreds of dollars, at 900 apartments on a $30 \times 30$ grid covering a district (rescaled to the unit square). The trend rises toward downtown in the east:

$$
\mu(\mathbf{s}) = 18 + 6\,s_{\text{east}}
$$

where $s_{\text{east}}$ is the east coordinate. The neighborhood effect has $\sigma^2 = 4$ (a standard deviation of \$200) with exponential correlation and $\phi = 0.15$, and the nugget has $\tau^2 = 1$ (a standard deviation of \$100). The rents are simulated, with the pieces kept separate as in Section 2.7.3.

```python
rng = np.random.default_rng(7)
S = grid(30)
n = len(S)
H = dist(S)
beta0, beta1 = 18.0, 6.0
sigma2, tau2, phi = 4.0, 1.0, 0.15
mu = beta0 + beta1 * S[:, 0]
L_u = np.linalg.cholesky(sigma2 * r_exponential(H, phi) + 1e-10 * np.eye(n))
u = L_u @ rng.standard_normal(n)
eps = np.sqrt(tau2) * rng.standard_normal(n)
y = mu + u + eps
print(f"mean of u: {u.mean():.3f}  sd of u: {u.std():.3f}  "
      f"sd of eps: {eps.std():.3f}  mean rent: {y.mean():.2f}")
print(f"rent range: {y.min():.2f} to {y.max():.2f}")
```

The output shows the three layers behaving as designed, with one lesson from Chapter 1 built in. The nugget has standard deviation 1.030, close to its true value of 1. The neighborhood effect has standard deviation 1.804 across the district, a little below its true value of 2, and its average over the district is $-2.056$, not 0. In this realization the whole district happens to sit below its trend by about \$200. That is not a bug. A spatially correlated effect averaged over one region has a large variance, exactly as Section 1.4 showed, so one map can sit well above or below its long-run average. The simulated rents range from 10.62 to 26.71, that is, from about \$1,060 to \$2,670 a month.

The next block draws the three layers and a comparison of smoothness. It saves the figures to `figures/ch02/`; run it from the top folder of the repository.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch02", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, field, title in zip(axes, (mu, mu + u, y),
                            ("Trend", "Trend + spatial effect", "Observed")):
    im = ax.imshow(field.reshape(30, 30), origin="lower",
                   extent=(0, 1, 0, 1), cmap="viridis", vmin=10, vmax=28)
    ax.set_title(title)
    ax.set_xlabel("east")
fig.colorbar(im, ax=axes, label="rent (hundreds of dollars)")
fig.savefig("figures/ch02/fig_2_1_rent_layers.png", dpi=120)

rng = np.random.default_rng(11)
x = np.linspace(0, 1, 400)
Hx = np.abs(x[:, None] - x[None, :])
fig, ax = plt.subplots(figsize=(7, 3.2))
for nu, phi_nu, label in ((0.5, 0.1, "exponential, nu = 1/2"),
                          (2.5, 0.05, "Matern, nu = 5/2")):
    R = r_matern(Hx, phi_nu, nu) + 1e-8 * np.eye(len(x))
    ax.plot(x, np.linalg.cholesky(R) @ rng.standard_normal(len(x)),
            label=label)
ax.set_xlabel("position along transect")
ax.set_ylabel("simulated u")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ch02/fig_2_2_smoothness.png", dpi=120)
print("figures saved")
```

![Figure 2.1. A simulated rent district in three layers: the trend rising to the east, the trend plus the neighborhood effect, and the observed rents with the nugget added. The neighborhood effect creates patches of high and low rent; the nugget adds speckle.](../figures/ch02/fig_2_1_rent_layers.png)

In Figure 2.1, the left panel is the smooth eastward rise in rents. The middle panel adds the neighborhood effect, which breaks the district into patches a few tenths of a unit across, in line with the practical range $3\phi = 0.45$. The right panel adds the nugget, which scatters apartment-to-apartment speckle over the patches without changing their shape.

![Figure 2.2. One simulated transect from each of two Matérn models with the same practical range of 0.3. The exponential model is jagged at every scale; the model with smoothness 5/2 is smooth.](../figures/ch02/fig_2_2_smoothness.png)

Figure 2.2 compares an exponential transect ($\phi = 0.1$, practical range about 0.30) with a Matérn transect with $\nu = 5/2$ ($\phi = 0.05$, practical range about 0.30). The two wander over similar distances, but the exponential path is jagged at every scale while the Matérn path is smooth, as the expansion in Section 2.5.3 predicts.

---

## Exercises

**Analytical**

1. In Section 2.1.4, the derivation of $\boldsymbol{\Sigma}$ assumed that $\mathbf{u}$ and $\boldsymbol{\varepsilon}$ are independent. Suppose instead that their cross-covariance matrix $\operatorname{Cov}(\mathbf{u}, \boldsymbol{\varepsilon})$ is not zero. Find $\operatorname{Var}(\mathbf{Y})$, and describe a rent setting where such a correlation might arise.

2. For the random walk of Section 2.2.3, show that $\operatorname{Cov}(Y_s, Y_t) = \min(s, t)$. Explain why this covariance cannot be written as a function of $t - s$ alone.

3. Show that for a second-order stationary process, $0 \le \gamma(\mathbf{h}) \le 2C(\mathbf{0})$. When can $\gamma(\mathbf{h})$ exceed $C(\mathbf{0})$, and what does that say about $C(\mathbf{h})$?

4. Generalize Section 2.4.2: three locations on a line at 0, $d$, and $2d$, with a rule giving correlation $c$ to pairs closer than some cutoff between $d$ and $2d$ and 0 to pairs beyond it. Show that the eigenvalues are $1 - c\sqrt{2}$, $1$, and $1 + c\sqrt{2}$, and conclude that the rule is invalid whenever $c > 1/\sqrt{2}$.

5. In the product argument of Section 2.4.3, where exactly was independence of $u_1$ and $u_2$ used? Show by example that the product of a process with itself, $u(\mathbf{s})^2$, does not have covariance $C(\mathbf{h})^2$.

6. Verify that the spherical correlation function is continuous at $h = \phi$ and that its first derivative is also zero there. Compute its practical range exactly, by solving a cubic, and compare it with the true range $\phi$ and with the value $0.811\,\phi$ in the table of Section 2.5.

7. Using the expansion in Section 2.5.3, find the leading term of $1 - r(h)$ for the Matérn model with $\nu = 5/2$. Compare the small-$h$ behavior of the exponential, Matérn $3/2$, Matérn $5/2$, and Gaussian models, and order them from roughest to smoothest.

8. Compute the Cholesky factor of the $3 \times 3$ matrix $\boldsymbol{\Sigma}$ in Section 2.1.4 by hand, using the column-by-column formulas of Section 2.7.1. Check your answer with `np.linalg.cholesky`.

9. A river runs at 30 degrees north of east, and correlation along the river reaches three times as far as correlation across it. Write down $\mathbf{G}$ as $\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ with $\mathbf{U}$ a rotation matrix, and compute the anisotropic distance between $(0, 0)$ and $(1, 0)$.

**Computational and simulation**

10. Simulate 500 rent districts as in Section 2.8.2, recording the average of $\mathbf{u}$ over each. Compare the standard deviation of these averages with the theoretical value $\sqrt{\mathbf{1}'\sigma^2\mathbf{R}\mathbf{1}}/n$ from Chapter 1, and compute the effective sample size of the district.

11. Time `np.linalg.cholesky` and `np.linalg.eigh` on exponential covariance matrices for grids of 500, 1,000, and 2,000 locations. How does the ratio of their times change with $n$?

12. Find the smallest square grid (number of points per side and spacing) on which the tent function of Section 2.4.4 produces a negative eigenvalue. Try spacings between 0.2 and 1.0.

13. For the $20 \times 20$ grid of Section 2.5.1, compute the condition number of $\sigma^2\mathbf{R} + \tau^2\mathbf{I}$ for the Gaussian model with $\sigma^2 = 1$ and $\tau^2 = 10^{-6}$, $10^{-3}$, and $10^{-1}$. Then repeat with the Matérn model with $\nu = 5/2$ and the same practical range. Which model needs less nugget to be numerically safe?

---

## What Comes Next

This chapter assumed we know $\sigma^2$, $\tau^2$, $\phi$, and the correlation family. Chapter 3 begins estimating them from data. It turns the semivariogram of Section 2.2 into an estimator by averaging squared differences over pairs of locations, fits the families of Section 2.5 to the result, and shows what happens when a trend in the mean is mistaken for spatial correlation.
