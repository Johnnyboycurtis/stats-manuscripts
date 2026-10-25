# Chapter 12: Models for Clustering and Inhibition

Chapter 10 gave us tools for looking at a point pattern: kernel intensity maps, the nearest-neighbor and empty-space functions, Ripley's $K$-function and the pair correlation function, and Monte Carlo envelopes. Chapter 11 turned the first-order part of the story into a regression: if the intensity depends on covariates, a Poisson GLM, fitted through the Berman-Turner device, estimates how. What neither chapter did is *model the way points relate to each other*. When the envelope for $L(r) - r$ is left behind, the pattern is saying something, and so far all we can do is describe it.

This chapter supplies models for that something. A pattern can depart from complete spatial randomness in three ways. The intensity can vary from place to place, which Chapter 11 handled. Points can attract each other, so that they come in clumps. Or points can repel each other, so that they keep their distance. The chapter builds one family of models for attraction (cluster processes, and their cousin the log-Gaussian Cox process) and one for repulsion (hard-core processes and Gibbs processes, of which the Strauss model is the simplest). For each family we ask the same questions: how is it simulated, what does it imply for $K(r)$, how are its parameters estimated, and how do we check that it fits.

Two threads from earlier chapters come together here. The model equation $Y = \mu + u + \varepsilon$ returns in Section 12.3 as the log-intensity of a Cox process, where the spatial random effect $u$ of Part II creates the clustering. And the estimation methods are old friends in new clothes: fitting a cluster process is the nonlinear least squares of Chapter 3 applied to $\hat{K}$ in place of a variogram, and fitting an inhibition process is the weighted Poisson regression of Chapter 11 with one extra column in the design matrix.

> **Chapter at a Glance**
>
> - **Three Explanations (§12.1):** Trend, clustering, and inhibition, and what each predicts for $g(r)$ and $K(r)$.
> - **Cluster Processes (§12.2):** Neyman-Scott processes (parents, offspring), the Thomas and Matérn cluster models, simulation with a buffer, the closed-form $K$ of the Thomas process, fitting by minimum contrast, and inhomogeneous cluster models.
> - **The Log-Gaussian Cox Process (§12.3):** The model equation once more, with the pair correlation function $g(r) = \exp(C_u(r))$ and a pixel-grid fit that combines Parts II and III with the Poisson GLM.
> - **Inhibition (§12.4):** Matérn hard-core processes and sequential inhibition, the Strauss model as a Gibbs process, the neighbor count as one more covariate column, and maximum pseudolikelihood.
> - **Checking a Fitted Model (§12.5):** Smoothed residual fields, envelopes simulated from the fitted model, and what a table of goodness-of-fit tests can and cannot tell apart.
> - **Walkthrough and Lab (§12.6):** Seven figures that put the pieces side by side.

---

## 12.1 Three Explanations for a Departure From CSR

Suppose you have a slide of cells and an envelope plot in which $L(r) - r$ rises above the Monte Carlo band. What does that mean? Three different stories produce the same picture, and the rest of the chapter exists to tell them apart.

- **Heterogeneity.** The intensity is not constant. A region of the slide is more densely populated than another, and points are placed independently given that intensity. Pairs of points are close to each other mostly because both sit in the dense region, not because one attracts the other. This is the situation of Chapter 11, where the intensity $\lambda(\mathbf{s}) = \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$ is explained by covariates.
- **Clustering.** Points attract each other. Given the position of one point, a second point is more likely to be nearby than a uniformly placed point would be. Immune cells gather around tumor nests; seedlings grow near parent trees; disease cases spread from a source.
- **Inhibition.** Points repel each other. Given the position of one point, a second is *less* likely to be nearby. Cells cannot overlap, shops avoid opening next door to each other, and tumor cells compete for space.

Recall from Chapter 10 the pair correlation function $g(r)$, which compares the density of pairs at distance $r$ with what independence would give, and its cumulative version $K(r) = \int_0^r 2\pi s\,g(s)\,ds$. Under complete spatial randomness $g(r) = 1$ and $K(r) = \pi r^2$. The three explanations leave different fingerprints.

| Explanation | How the points arise | $g(r)$ after accounting for $\lambda(\mathbf{s})$ | Ordinary $K(r)$ against $\pi r^2$ | Model, and where |
| :--- | :--- | :--- | :--- | :--- |
| Heterogeneity | Independent given a varying intensity | Equal to 1 | Above (an artifact) | Inhomogeneous Poisson, Chapter 11 |
| Clustering | Points attract | Above 1 at short distances | Above | Cluster processes (§12.2), Cox processes (§12.3) |
| Inhibition | Points repel | Below 1 at short distances | Below | Hard-core and Gibbs processes (§12.4) |

Two cautions come with this table. First, the third column is the one that distinguishes heterogeneity from clustering, and it requires knowing $\lambda(\mathbf{s})$. Chapter 11 showed that an ordinary $K$-function applied to a pattern with a trend looks clustered, which is why the inhomogeneous $K$-function reweights each pair by the inverse intensities. We will see the same trap again in Section 12.5.3.

Second, the line between heterogeneity and clustering is partly a matter of what you measured. If the dense region is explained by a covariate you observed, it is a trend. If it comes from a driver you did not observe and cannot name, the model has to treat it as random, and a random intensity looks exactly like clustering. That observation is the idea behind the Cox processes of Section 12.3, and it is why the cluster models of Section 12.2 can be read either mechanistically (points really do gather around parents) or as a convenient description of unexplained heterogeneity.

Figure 12.1 in Section 12.6 puts the three side by side: three patterns, with the *same ordinary* $L(r) - r$ plotted below each. The first two rise above zero and the third falls below it.

The tools we carry over are the ones from Chapters 10 and 11. We write $\lambda(\mathbf{s})$ for the intensity and $\mathcal{D}$ for the unit square (so $\operatorname{area}(\mathcal{D}) = 1$). The estimator is

$$
\hat{K}(r) = \frac{1}{n(n-1)}\sum_{i=1}^n\sum_{j \ne i}\omega_{ij}\,\mathbb{1}(h_{ij} \le r)
$$

where $h_{ij}$ is the distance between points $i$ and $j$ and $\omega_{ij}$ corrects for edges. Chapter 10 derived Ripley's isotropic weights. On a rectangle there is a simpler choice, the *translation correction*, which gives each pair the weight $\omega_{ij} = 1/\left[(1 - \lvert\Delta x_{ij}\rvert)(1 - \lvert\Delta y_{ij}\rvert)\right]$, where $\Delta x_{ij}$ and $\Delta y_{ij}$ are the coordinate differences. The weight is the reciprocal of the fraction of the window that stays inside the window when it is shifted by the pair's separation vector, and it can be computed in one line. We use it throughout this chapter.

---

## 12.2 Cluster Processes

### 12.2.1 Neyman-Scott Processes

A **Neyman-Scott process** builds a clustered pattern in three steps.

1. **Parents.** Place unobserved *parent* points as a homogeneous Poisson process with intensity $\kappa$.
2. **Offspring counts.** Give each parent an independent number of offspring, Poisson with mean $\mu_c$.
3. **Dispersal.** Place each offspring at the parent's location plus an independent random displacement with a fixed density $f(\mathbf{d})$ that is symmetric around the origin.

The observed pattern is the set of offspring only. The parents are never seen.

Our running story is a slide of tissue from a tumor. The parents are small tumor nests that the stain did not reveal. The offspring are immune cells, which gather around the nests. The unit square is a $1\text{ mm} \times 1\text{ mm}$ field of view, so a displacement spread of $0.03$ is $30\ \mu\text{m}$. (The data are simulated; the numbers are chosen for readability and are not measurements.)

**The intensity of the offspring process.** The process of offspring is stationary, and its intensity is $\lambda = \kappa\mu_c$. The derivation is a short calculation and a first example of an argument we will use again for $K$.

*A tool: Campbell's formula for a Poisson process.* If $\Phi$ is a Poisson process with intensity $\kappa$ and $q$ is a function, then the expected sum of $q$ over the points of $\Phi$ is $\operatorname{E}\left[\sum_{\mathbf{c} \in \Phi} q(\mathbf{c})\right] = \kappa\int q(\mathbf{c})\,d\mathbf{c}$. In words, the expected sum is the intensity times the integral, because the expected number of points in a small area $d\mathbf{c}$ is $\kappa\,d\mathbf{c}$.

Let $B$ be a region and $N(B)$ the number of offspring in $B$. A parent at $\mathbf{c}$ contributes, on average, $\mu_c$ offspring, each landing in $B$ with probability $\int \mathbb{1}(\mathbf{c} + \mathbf{d} \in B)\,f(\mathbf{d})\,d\mathbf{d}$. Writing the sum over parents and applying Campbell's formula to $q(\mathbf{c}) = \mu_c\int \mathbb{1}(\mathbf{c} + \mathbf{d} \in B)\,f(\mathbf{d})\,d\mathbf{d}$,

$$
\operatorname{E}[N(B)] = \kappa\mu_c\int\!\!\int \mathbb{1}(\mathbf{c} + \mathbf{d} \in B)\,f(\mathbf{d})\,d\mathbf{d}\,d\mathbf{c} = \kappa\mu_c\int f(\mathbf{d})\left[\int \mathbb{1}(\mathbf{c} + \mathbf{d} \in B)\,d\mathbf{c}\right]d\mathbf{d} = \kappa\mu_c\operatorname{area}(B)
$$

The inner integral is the area of $B$ shifted by $-\mathbf{d}$, which is $\operatorname{area}(B)$ whatever $\mathbf{d}$ is, and then $\int f = 1$. So the intensity is $\lambda = \kappa\mu_c$: parents per unit area times offspring per parent.

**The variance of the count.** The number of offspring in the whole window is a compound Poisson sum: $M \sim \text{Poisson}(\kappa\operatorname{area}(\mathcal{D}))$ parents, each with an independent $\text{Poisson}(\mu_c)$ number of offspring (ignoring offspring that fall outside the window). For a compound sum, $\operatorname{Var}\left(\sum_{i=1}^M N_i\right) = \operatorname{E}[M]\operatorname{Var}(N_i) + \operatorname{Var}(M)\operatorname{E}[N_i]^2$, which gives

$$
\operatorname{Var}(N(\mathcal{D})) = \kappa\mu_c + \kappa\mu_c^2 = \lambda(1 + \mu_c)
$$

on the unit square. A Poisson process has variance equal to the mean, $\lambda$. The cluster process has variance $(1 + \mu_c)$ times as large: the more points per cluster, the more the count varies from slide to slide. For $\kappa = 25$ and $\mu_c = 8$, the mean is $200$ and the standard deviation is $\sqrt{200 \times 9} \approx 42$. That is why two slides from the same process can have very different numbers of cells, and why a single slide cannot pin down $\kappa$ and $\mu_c$ separately very well; there are only about $25$ clusters on it.

### 12.2.2 The Thomas and Matérn Cluster Processes

The two standard Neyman-Scott models differ only in the dispersal density $f$.

- **The Thomas process** displaces each offspring by an independent bivariate normal, $\mathbf{d} \sim N(\mathbf{0}, \sigma_c^2\mathbf{I}_2)$. Its parameters are the parent intensity $\kappa$, the mean cluster size $\mu_c$, and the spread $\sigma_c$. Clusters have no sharp boundary: most offspring fall within about $2\sigma_c$ of the parent, and a few fall much farther.
- **The Matérn cluster process** places each offspring uniformly on a disc of radius $R$ around its parent. Its parameters are $\kappa$, $\mu_c$, and $R$. Clusters are discs with a sharp edge.

In this chapter $\sigma_c$ always denotes the spread of the Gaussian dispersal, and $R$ the radius of the disc. The Matérn cluster process shares its name with the Matérn covariance family of Chapter 2 and the Matérn hard-core processes of Section 12.4, and has nothing mathematical in common with them except the originator.

### 12.2.3 Simulation With a Buffer

To simulate a Neyman-Scott process on the unit square, a first idea is: draw parents in the square, draw offspring around them, and keep the offspring. This is wrong near the edges. A parent just *outside* the square can have offspring *inside* it, and those offspring are lost if parents are only drawn inside. The pattern ends up with too few points near the boundary, and the intensity is not constant.

The fix is to draw parents on a buffered window that extends beyond the square by a margin large enough that parents farther out have a negligible chance of reaching in. For the Thomas process, a displacement has standard deviation $\sigma_c$ in each coordinate, so the chance that a normal variable exceeds $4\sigma_c$ in magnitude is about $6 \times 10^{-5}$. A buffer of $4\sigma_c$ is plenty. For the Matérn cluster process the buffer is exactly $R$.

The code below implements both simulators and checks the buffer. With $\kappa = 25$, $\mu_c = 8$, and $\sigma_c = 0.03$, the intensity is $200$. We simulate many patterns with and without the buffer and compare mean counts, and we count points in the strip within $0.03$ of an edge. Without a buffer, the expected fraction of offspring that stay inside is $1 - 4\sigma_c/\sqrt{2\pi}$, because each edge loses a strip of area equivalent to $\sigma_c/\sqrt{2\pi}$ per unit length of edge (the mean of $\max(0, Z)\sigma_c$ for a standard normal $Z$), so the four edges lose $4\sigma_c/\sqrt{2\pi}$ of the total.

```python
import numpy as np
from scipy import optimize


def sim_thomas(kappa, mu_c, sigma_c, rng, buffer=4.0):
    """Thomas process on the unit square; parents live in a buffered window.

    The simulator of Section 10.3.4 with the buffer width as an argument."""
    b = buffer * sigma_c
    n_par = rng.poisson(kappa * (1 + 2 * b) ** 2)
    parents = rng.uniform(-b, 1 + b, size=(n_par, 2))
    counts = rng.poisson(mu_c, size=n_par)
    centers = np.repeat(parents, counts, axis=0)
    pts = centers + sigma_c * rng.standard_normal(centers.shape)
    return pts[np.all((pts >= 0) & (pts <= 1), axis=1)]


def sim_matern_cluster(kappa, mu_c, R, rng):
    """Matern cluster process: offspring uniform on discs of radius R."""
    n_par = rng.poisson(kappa * (1 + 2 * R) ** 2)
    parents = rng.uniform(-R, 1 + R, size=(n_par, 2))
    counts = rng.poisson(mu_c, size=n_par)
    centers = np.repeat(parents, counts, axis=0)
    rad = R * np.sqrt(rng.uniform(size=len(centers)))
    ang = rng.uniform(0, 2 * np.pi, size=len(centers))
    pts = centers + np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    return pts[np.all((pts >= 0) & (pts <= 1), axis=1)]
```

Now the check, with $6000$ patterns of each kind.

```python
rng = np.random.default_rng(2100)
kappa, mu_c, sigma_c = 25.0, 8.0, 0.03
n_with = [len(sim_thomas(kappa, mu_c, sigma_c, rng)) for _ in range(6000)]
n_without = [len(sim_thomas(kappa, mu_c, sigma_c, rng, buffer=0.0))
             for _ in range(6000)]
print(f"expected intensity kappa*mu_c = {kappa * mu_c:.0f}")
print(f"mean count with buffer:    {np.mean(n_with):6.1f} "
      f"(se {np.std(n_with) / np.sqrt(6000):.1f})")
print(f"mean count without buffer: {np.mean(n_without):6.1f} "
      f"(se {np.std(n_without) / np.sqrt(6000):.1f})")
loss = 4 * sigma_c / np.sqrt(2 * np.pi)
print(f"predicted loss without buffer: {100 * loss:.1f}% "
      f"-> {kappa * mu_c * (1 - loss):.1f}")
# edge profile: points within 0.03 of an edge, per unit area
def edge_density(P):
    near = np.minimum.reduce([P[:, 0], 1 - P[:, 0], P[:, 1], 1 - P[:, 1]])
    return np.sum(near < 0.03)
ring = 1 - (1 - 0.06) ** 2
e_with = np.mean([edge_density(sim_thomas(kappa, mu_c, sigma_c, rng))
                  for _ in range(500)]) / ring
e_without = np.mean([edge_density(sim_thomas(kappa, mu_c, sigma_c, rng, 0.0))
                     for _ in range(500)]) / ring
print(f"intensity in the outer 0.03 strip: with {e_with:.0f}, "
      f"without {e_without:.0f}")
```

With the buffer, the mean count over $6000$ patterns is $199.6$ (standard error $0.5$), indistinguishable from the expected $200$. Without it the mean is $190.0$, matching the predicted $4.8\%$ loss ($190.4$). The damage is concentrated at the edges: the intensity in the strip within $0.03$ of an edge is $194$ with the buffer (within simulation error of $200$) and only $131$ without it. An unbuffered pattern is not stationary, and a $K$-function estimated from it would be biased. We use the buffer from here on.

### 12.2.4 Deriving the $K$-Function of the Thomas Process

The goal is the closed form

$$
K(r) = \pi r^2 + \frac{1}{\kappa}\left(1 - \exp\!\left(-\frac{r^2}{4\sigma_c^2}\right)\right)
$$

It contains the Poisson value $\pi r^2$ plus an excess that grows from $0$ at $r = 0$ to $1/\kappa$ at large $r$. The derivation goes through the pairs of points, so we start with a tool.

**A tool we need: the second-order product density.** Let $\rho^{(2)}(\mathbf{s}, \mathbf{s}')\,d\mathbf{s}\,d\mathbf{s}'$ be the expected number of *ordered pairs* of distinct points with one in the small area $d\mathbf{s}$ and the other in $d\mathbf{s}'$. For a homogeneous Poisson process the two points are independent, so $\rho^{(2)} = \lambda^2$. For a stationary process $\rho^{(2)}$ depends only on the separation $\mathbf{h} = \mathbf{s} - \mathbf{s}'$, and the pair correlation function is the ratio to the Poisson value,

$$
g(\mathbf{h}) = \frac{\rho^{(2)}(\mathbf{h})}{\lambda^2}
$$

This is the definition of $g$ from Chapter 10 seen through pairs: $g > 1$ means pairs at that separation are more common than independence predicts. Given a point at the origin, the intensity of other points at $\mathbf{h}$ is $\rho^{(2)}(\mathbf{h})/\lambda$, so the expected number of other points within distance $r$ of a typical point is $\int_{\lVert\mathbf{h}\rVert \le r}\rho^{(2)}(\mathbf{h})/\lambda\,d\mathbf{h}$. By the definition of $K$, $\lambda K(r)$ is that expected number,

$$
K(r) = \int_{\lVert\mathbf{h}\rVert \le r} g(\mathbf{h})\,d\mathbf{h} = \int_0^r 2\pi s\,g(s)\,ds
$$

where the last form assumes $g$ depends only on distance.

**Step 1: split the pairs.** An ordered pair of distinct offspring comes either from two *different* parents or from the *same* parent. We find the expected number of each kind.

*Different parents.* A parent at $\mathbf{c}$ produces offspring with intensity function $\mu_c f(\mathbf{s} - \mathbf{c})$: the mean number of offspring times the density of where each lands. Two different parents are distinct points of a Poisson process, and the second-order version of Campbell's formula says that the expected sum over ordered pairs of distinct points of a Poisson process is $\kappa^2$ times the double integral. So the expected number of such pairs in $d\mathbf{s}\,d\mathbf{s}'$ is

$$
\kappa^2\int\!\!\int\mu_c f(\mathbf{s} - \mathbf{c}_1)\,\mu_c f(\mathbf{s}' - \mathbf{c}_2)\,d\mathbf{c}_1\,d\mathbf{c}_2 = \kappa^2\mu_c^2\cdot 1\cdot 1 = \lambda^2
$$

because each integral of a density is 1. Pairs from different parents behave exactly like pairs from a Poisson process.

*The same parent.* A parent at $\mathbf{c}$ has $N$ offspring, Poisson with mean $\mu_c$. The number of ordered pairs of distinct offspring is $N(N-1)$, with expectation $\operatorname{E}[N(N-1)] = \operatorname{E}[N^2] - \operatorname{E}[N] = (\mu_c + \mu_c^2) - \mu_c = \mu_c^2$. Each pair's two displacements are independent, so a pair lands in $d\mathbf{s}\,d\mathbf{s}'$ with density $f(\mathbf{s} - \mathbf{c})f(\mathbf{s}' - \mathbf{c})$. Summing over parents with Campbell's formula,

$$
\kappa\mu_c^2\int f(\mathbf{s} - \mathbf{c})\,f(\mathbf{s}' - \mathbf{c})\,d\mathbf{c} = \kappa\mu_c^2\,f_2(\mathbf{s} - \mathbf{s}')
$$

where $f_2(\mathbf{h}) = \int f(\mathbf{d})f(\mathbf{d} - \mathbf{h})\,d\mathbf{d}$ is the density of the *difference* $\mathbf{D}_1 - \mathbf{D}_2$ of two independent displacements.

**Step 2: add and divide.** The two contributions give $\rho^{(2)}(\mathbf{h}) = \lambda^2 + \kappa\mu_c^2 f_2(\mathbf{h})$, and since $\lambda^2 = \kappa^2\mu_c^2$,

$$
g(\mathbf{h}) = 1 + \frac{\kappa\mu_c^2 f_2(\mathbf{h})}{\kappa^2\mu_c^2} = 1 + \frac{f_2(\mathbf{h})}{\kappa}
$$

Integrating over a disc of radius $r$,

$$
K(r) = \pi r^2 + \frac{1}{\kappa}\operatorname{P}\!\left(\lVert\mathbf{D}_1 - \mathbf{D}_2\rVert \le r\right)
$$

This holds for every Neyman-Scott process with Poisson offspring counts: the excess over $\pi r^2$ is $1/\kappa$ times the probability that two siblings are within $r$ of each other. Only the dispersal density enters, through the distribution of the distance between two siblings.

**Step 3: the Thomas case.** If $\mathbf{D}_1$ and $\mathbf{D}_2$ are independent $N(\mathbf{0}, \sigma_c^2\mathbf{I}_2)$, their difference is $N(\mathbf{0}, 2\sigma_c^2\mathbf{I}_2)$. The distance of a bivariate normal $N(\mathbf{0}, s^2\mathbf{I}_2)$ from the origin has $\operatorname{P}(\text{distance} \le r) = 1 - e^{-r^2/(2s^2)}$ (the squared distance is $s^2$ times a $\chi^2_2$ variable, which is exponential with mean $2s^2$). With $s^2 = 2\sigma_c^2$ this is $1 - e^{-r^2/(4\sigma_c^2)}$, which is the claimed formula. The pair correlation function is $g(r) = 1 + \dfrac{1}{4\pi\kappa\sigma_c^2}e^{-r^2/(4\sigma_c^2)}$.

Two readings of the result help later.

- **The long-range limit.** As $r \to \infty$, $K(r) - \pi r^2 \to 1/\kappa$, so $\lambda K(r) - \lambda\pi r^2 \to \lambda/\kappa = \mu_c$. A typical point has, on average, $\mu_c$ more neighbors than a Poisson process would give it, and these are its siblings.
- **What the data can identify.** $K(r)$ depends on $\kappa$ and $\sigma_c$ but not on $\mu_c$. The mean cluster size is recovered from the intensity, $\hat{\mu}_c = \hat{\lambda}/\hat{\kappa}$, as in Section 12.2.5.

**A numerical check.** The code below defines $\hat{K}$ with the translation correction of Section 12.1, then simulates $300$ Thomas patterns and compares the average of $\hat{K}(r)$ with the formula at four radii. The Matérn cluster process is checked the same way in Section 12.2.7.

```python
def khat(P, r):
    """Translation-corrected K-hat for the unit square at radii r.

    The weight of a pair is 1 / [(1 - |dx|)(1 - |dy|)], the reciprocal of
    the share of the window that a shift by the pair's separation keeps.
    """
    n = len(P)
    if n < 2:
        return np.zeros(len(r))
    dx = np.abs(P[:, None, 0] - P[None, :, 0])
    dy = np.abs(P[:, None, 1] - P[None, :, 1])
    h = np.hypot(dx, dy)
    w = 1.0 / ((1 - dx) * (1 - dy))
    off = ~np.eye(n, dtype=bool)
    h, w = h[off], w[off]
    order = np.argsort(h)
    cw = np.concatenate([[0.0], np.cumsum(w[order])])
    idx = np.searchsorted(h[order], r, side="right")
    return cw[idx] / (n * (n - 1))


def K_thomas(r, kappa, sigma_c):
    return np.pi * r**2 + (1 - np.exp(-r**2 / (4 * sigma_c**2))) / kappa
```

```python
r_grid = np.linspace(0.005, 0.15, 59)
K_runs = np.array([khat(sim_thomas(kappa, mu_c, sigma_c, rng), r_grid)
                   for _ in range(300)])
print("   r     pi r^2   mean K-hat   Thomas K")
for rr in (0.02, 0.05, 0.10, 0.15):
    i = np.argmin(np.abs(r_grid - rr))
    print(f"{r_grid[i]:5.3f}  {np.pi * r_grid[i]**2:7.4f}  "
          f"{K_runs[:, i].mean():10.4f}  "
          f"{K_thomas(r_grid[i], kappa, sigma_c):8.4f}")
```

| $r$ | $\pi r^2$ | mean $\hat{K}(r)$ | Thomas $K(r)$ |
| :--- | :--- | :--- | :--- |
| 0.02 | 0.0013 | 0.0056 | 0.0055 |
| 0.05 | 0.0079 | 0.0286 | 0.0279 |
| 0.10 | 0.0314 | 0.0695 | 0.0689 |
| 0.15 | 0.0707 | 0.1108 | 0.1106 |

The Thomas $K$-function is unbiased in the sense that matters here: the average of $\hat{K}$ over many simulated patterns tracks the formula at every radius, to within $0.001$, on a scale where $K$ itself ranges from $0.001$ to $0.11$. (The ratio estimator $\hat{K}$ has a small positive bias for $n$ near $200$, visible at $r = 0.05$ as $0.0286$ against $0.0279$.) The excess over $\pi r^2$ is large: at $r = 0.05$ the Thomas $K$ is $3.5$ times the Poisson value, and by $r = 0.15$ it is still $1.6$ times.

### 12.2.5 Fitting by Minimum Contrast

The parameters $\kappa$ and $\sigma_c$ can be recovered from an estimated $K$-function by choosing the values whose theoretical $K$ comes closest to $\hat{K}$. This is the **minimum contrast** method, and if it sounds familiar it should: it is what Chapter 3 did with the variogram. There we estimated $\hat{\gamma}(h_k)$ in bins, chose a parametric family $\gamma(h;\boldsymbol{\theta})$, and fitted $\boldsymbol{\theta}$ by weighted nonlinear least squares. Here we have an estimated summary function $\hat{K}(r)$ on a grid of radii, a parametric family $K(r;\kappa,\sigma_c)$, and we choose the parameters to minimize the **contrast**

$$
D(\kappa, \sigma_c) = \sum_{k}\left[\hat{K}(r_k)^{c} - K(r_k;\kappa,\sigma_c)^{c}\right]^2\Delta r
$$

where $r_1 < \dots < r_M$ is a grid of radii with spacing $\Delta r$. The power $c$ is the same kind of power transformation as the fourth root in the Cressie-Hawkins estimator of Chapter 3, although the motivation here is different: $\hat{K}(r)$ is far more variable at large $r$ than at small $r$, because it counts more pairs, and raising to a power below one pulls the large values in, so that the contrast is not dominated by the largest radii. The usual choice is $c = 1/4$.

Two choices have to be made before fitting.

- **The radii.** The lower limit should avoid the smallest distances, where $\hat{K}$ is built from a handful of pairs. The upper limit should stay short enough that $\hat{K}$ is still reliable, with a common guideline of about one quarter of the shorter side of the window, and long enough to see the cluster structure: here $0.15$ is five times $\sigma_c$. We use $59$ radii from $0.005$ to $0.15$.
- **What is fitted.** For the Thomas process $K$ depends on $\kappa$ and $\sigma_c$ only. The third parameter follows from the intensity: since $\lambda = \kappa\mu_c$, $\hat{\mu}_c = \hat{\lambda}/\hat{\kappa}$ with $\hat{\lambda} = n/\operatorname{area}(\mathcal{D})$.

**A hand-sized example.** Take the simulated slide of tumor-nest immune cells and evaluate the contrast at the true parameters at two radii, $r = 0.05$ and $r = 0.10$. At $r = 0.05$, $r^2/(4\sigma_c^2) = 0.0025/0.0036 = 0.694$, so $1 - e^{-0.694} = 0.501$, the excess is $0.501/25 = 0.0200$, and $K = \pi(0.05)^2 + 0.0200 = 0.00785 + 0.0200 = 0.0279$, whose fourth root is $0.4086$. At $r = 0.10$, $r^2/(4\sigma_c^2) = 2.78$, $1 - e^{-2.78} = 0.938$, the excess is $0.0375$, $K = 0.0314 + 0.0375 = 0.0689$, and the fourth root is $0.5124$. The code reports these model values next to the slide's own $\hat{K}$, and the squared differences that enter $D$.

```python
slide = sim_thomas(kappa, mu_c, sigma_c, rng)
n_slide = len(slide)
K_obs = khat(slide, r_grid)
print(f"slide: {n_slide} immune cells")

# two radii by hand
print("\n   r   pi r^2  exp-term  K model  K^(1/4)   K-hat  "
      "K-hat^(1/4)  sq. diff")
for rr in (0.05, 0.10):
    i = np.argmin(np.abs(r_grid - rr))
    e = 1 - np.exp(-r_grid[i]**2 / (4 * sigma_c**2))
    km = K_thomas(r_grid[i], kappa, sigma_c)
    print(f"{r_grid[i]:5.3f} {np.pi * r_grid[i]**2:7.4f} {e / kappa:8.4f} "
          f"{km:8.4f} {km**0.25:8.4f} {K_obs[i]:8.4f} {K_obs[i]**0.25:10.4f} "
          f"{(K_obs[i]**0.25 - km**0.25)**2:10.2e}")
```

The slide has $137$ cells, well below the expected $200$ (the standard deviation of the count is $42$, Section 12.2.1). At $r = 0.05$ the contrast contribution is tiny, $7.8 \times 10^{-6}$; at $r = 0.10$ it is $2.7 \times 10^{-4}$, because this slide has fewer pairs than the truth predicts at larger distances. Now fit.

```python
def contrast(params, K_hat, r, c=0.25):
    """Minimum contrast residuals for the Thomas model."""
    kap, sig = np.exp(params)
    return K_hat**c - K_thomas(r, kap, sig)**c


def fit_thomas(K_hat, r, n_obs, c=0.25, start=(10.0, 0.05)):
    sol = optimize.least_squares(contrast, np.log(start), args=(K_hat, r, c),
                                 bounds=(np.log([0.5, 0.002]),
                                         np.log([5000.0, 0.2])))
    kap, sig = np.exp(sol.x)
    return kap, sig, n_obs / kap, np.sum(sol.fun**2) * (r[1] - r[0])


k_hat, s_hat, m_hat, D_min = fit_thomas(K_obs, r_grid, n_slide)
print(f"\nfit: kappa={k_hat:.1f} (truth {kappa:.0f}), "
      f"sigma_c={s_hat:.4f} (truth {sigma_c}), "
      f"mu_c={m_hat:.2f} (truth {mu_c:.0f}), contrast={D_min:.2e}")
D_true = np.sum(contrast(np.log([kappa, sigma_c]), K_obs, r_grid)**2)
print("contrast at the true parameters: "
      f"{D_true * (r_grid[1] - r_grid[0]):.2e}")
for start in ((3.0, 0.2), (50.0, 0.01), (100.0, 0.1)):
    kk, ss, _, _ = fit_thomas(K_obs, r_grid, n_slide, start=start)
    print(f"start {start}: kappa={kk:.1f}, sigma_c={ss:.4f}")
```

The fitted parameters are $\hat\kappa = 33.5$, $\hat\sigma_c = 0.0248$, and $\hat\mu_c = 137/33.5 = 4.09$, against the truth of $25$, $0.03$, and $8$. The minimum contrast is $1.76 \times 10^{-6}$, an order of magnitude below the value at the true parameters, $2.15 \times 10^{-5}$: on this slide the fitted curve is closer to $\hat{K}$ than the truth is, which is what minimization should do. Three very different starting values all converge to the same answer, so the objective has a single basin here.

The estimates look poor, and the reason is not the method. A slide of this process has about $25$ clusters. Estimating a cluster intensity from $25$ clusters, and a cluster size from the ratio of two noisy quantities, is inherently imprecise, and the particular slide has fewer points than average. Figure 12.2 in Section 12.6 shows how little the curves differ even though the parameters do: many parameter pairs give nearly the same $K^{1/4}$.

**How good is the method?** The honest answer needs many slides, not one. The next block simulates $200$ slides from the true model, fits each with $c = 1/4$, $1/2$, and $1$, and summarizes the estimates by their median and by the root-mean-square error of the logarithm of each estimate (a relative error: $0.3$ means typically about 30% off).

```python
# which power c?
true_vals = np.array([kappa, sigma_c, mu_c])
fits = {c: [] for c in (0.25, 0.5, 1.0)}
for _ in range(200):
    P = sim_thomas(kappa, mu_c, sigma_c, rng)
    Kh = khat(P, r_grid)
    for c in fits:
        fits[c].append(fit_thomas(Kh, r_grid, len(P), c=c)[:3])
print("\n  c   median kappa  median sigma_c  median mu_c   rms log error "
      "(kappa, sigma_c, mu_c)")
for c, v in fits.items():
    v = np.array(v)
    rms = np.sqrt(np.mean(np.log(v / true_vals)**2, axis=0))
    med = np.median(v, axis=0)
    print(f"{c:4.2f}  {med[0]:10.1f}  {med[1]:13.4f}  {med[2]:10.2f}   "
          f"{rms[0]:.3f} {rms[1]:.3f} {rms[2]:.3f}")
```

| $c$ | median $\hat\kappa$ | median $\hat\sigma_c$ | median $\hat\mu_c$ | rms log error of $\hat\kappa$, $\hat\sigma_c$, $\hat\mu_c$ |
| :--- | :--- | :--- | :--- | :--- |
| 0.25 | 25.9 | 0.0296 | 7.58 | 0.310, 0.142, 0.296 |
| 0.50 | 26.2 | 0.0295 | 7.50 | 0.319, 0.156, 0.305 |
| 1.00 | 26.3 | 0.0294 | 7.46 | 0.333, 0.192, 0.320 |

The medians are close to the truth: $25.9$, $0.0296$, and $7.6$ against $25$, $0.03$, and $8$. The typical relative error of $\hat\kappa$ and $\hat\mu_c$ is about $30\%$, and that of $\hat\sigma_c$ about $14\%$. A plausible reason for the smaller spread of $\hat\sigma_c$ is that the location of the shoulder of $K$ is determined by the many sibling pairs at short distances, while $\kappa$ rests on the height of the plateau, which depends on the number of clusters.

The power $c = 1/4$ gives the smallest errors for all three parameters, but not by much; $c = 1$, which is plain least squares on $\hat{K}$, is about $35\%$ worse for $\sigma_c$. The gain is real but small: transforming the summary function before fitting helps a little.

Minimum contrast gives point estimates and no standard errors. The contrast is a descriptive criterion, not a likelihood, and the values $\hat{K}(r_k)$ at neighboring radii are strongly dependent, so the Hessian of $D$ says nothing reliable about uncertainty. Uncertainty comes from simulation: simulate from the fitted model, refit each pattern, and read the spread of the refitted estimates (Exercise 18). This is the same parametric bootstrap that Chapter 11 recommends for the Poisson GLM under clustering.

### 12.2.6 Inhomogeneous Cluster Models

Real slides have trends *and* clusters. Immune cells are denser in some regions of the slide, and within those regions they still gather around nests. A cluster model with a trend should have two parts: a model for the first-order intensity, fitted as in Chapter 11, and a model for the interaction, fitted as in Section 12.2.5.

The simplest such model is an **independently thinned Neyman-Scott process**. Simulate a stationary cluster process with intensity $\kappa\mu_c$, then keep each point $\mathbf{s}$ independently with probability $p(\mathbf{s})$. The observed intensity is $\lambda(\mathbf{s}) = \kappa\mu_c\,p(\mathbf{s})$. Because thinning is independent, it does not change the *pair correlation function*: by Exercise 4 the thinned process has the same $g$ as the unthinned one, and therefore the same inhomogeneous $K$-function: Chapter 11's $K_{\text{inhom}}$ divides each pair by $\lambda(\mathbf{s}_i)\lambda(\mathbf{s}_j)$, and for a thinned process its expectation is the $K$ of the unthinned one. So the two parts separate cleanly:

1. Fit $\lambda(\mathbf{s}) = \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$ by the weighted Poisson regression of Chapter 11. This estimates $\boldsymbol{\beta}$ consistently *even though the points cluster*. (The estimating equation is unbiased whatever the interaction, so the estimate is consistent; its *standard error* is another matter, as the experiment below shows.)
2. Estimate the inhomogeneous $K$-function with $\hat\lambda(\mathbf{s}_i)$ plugged in, and fit $\kappa$ and $\sigma_c$ to it by minimum contrast.

We use the relative density `pop_density` of the shared setting as the covariate: a slide with a densely cellular region centered near $(0.7, 0.7)$ on a sparser background. (The name comes from the shared setting of Part IV; here read it as the local density of tissue.) A point at $\mathbf{s}$ is kept with probability $p(\mathbf{s}) = \text{pop\_density}(\mathbf{s})^2$, so the observed intensity is $\lambda(\mathbf{s}) = 750\,\text{pop\_density}(\mathbf{s})^2$ with $\kappa = 25$, $\mu_c = 30$, and $\sigma_c = 0.03$. Taking logs, $\log\lambda(\mathbf{s}) = \log 750 + 2\log\text{pop\_density}(\mathbf{s})$: a log-linear model with $\beta_0 = \log 750 = 6.62$ and $\beta_1 = 2$ on the covariate $\log\text{pop\_density}$.

The code below restates the pieces from earlier chapters that we need. The Berman-Turner device of Chapter 11 writes the Poisson log-likelihood as a sum over $n$ data points and $n_d$ dummy points $\mathbf{s}_k^*$ on a grid, with weights $a_k$. Here the weights split the area of each grid cell equally among the quadrature points in it. The function `fit_bt` is the IRLS of Chapter 9 with prior weights $a_k$, with a simpler interface than the `fit_bt` of Section 11.3 because here the caller builds the design matrix (and, in Section 12.4, appends the Strauss column): with working response $z_k = \eta_k + (y_k - \mu_k)/\mu_k$ and working weights $a_k\mu_k$, where $y_k = \mathbb{1}_k/a_k$. The function `khat_inhom` reweights each pair by $1/(\hat\lambda_i\hat\lambda_j)$ in place of the constant $1/\hat\lambda^2$.

```python
def pop_density(P):
    """Relative density: a bump centered at (0.7, 0.7) on a flat base."""
    d2 = (P[:, 0] - 0.7)**2 + (P[:, 1] - 0.7)**2
    return 0.3 + 0.7 * np.exp(-d2 / (2 * 0.2**2))


def quad_scheme(P, m=30):
    """Data points plus an m-by-m grid of dummy points, counting weights."""
    g = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(g, g)
    dummy = np.column_stack([gx.ravel(), gy.ravel()])
    ij = np.minimum((P * m).astype(int), m - 1)
    cell_data = ij[:, 1] * m + ij[:, 0]
    count = np.bincount(cell_data, minlength=m * m)
    cell = np.concatenate([cell_data, np.arange(m * m)])
    a = (1.0 / m**2) / (1 + count[cell])
    ind = np.concatenate([np.ones(len(P)), np.zeros(len(dummy))])
    return np.vstack([P, dummy]), ind, a


def fit_bt(X, ind, a, iters=40):
    """IRLS for the Berman-Turner weighted Poisson regression.

    The fit of Section 11.3 with a simpler interface: the caller builds X."""
    y = ind / a
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(ind.sum())
    for _ in range(iters):
        eta = X @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        W = a * mu
        new = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        done = np.max(np.abs(new - beta)) < 1e-10
        beta = new
        if done:
            break
    mu = np.exp(X @ beta)
    cov = np.linalg.inv(X.T @ ((a * mu)[:, None] * X))
    loglik = np.sum(ind * (X @ beta)) - np.sum(a * mu)
    return beta, cov, loglik


def sim_thinned_thomas(kappa, mu_c, sigma_c, rng):
    P = sim_thomas(kappa, mu_c, sigma_c, rng)
    return P[rng.uniform(size=len(P)) < pop_density(P)**2]


def khat_inhom(P, r, lam):
    """K-hat with each pair weighted by 1 / (lam_i lam_j); unit area."""
    n = len(P)
    dx = np.abs(P[:, None, 0] - P[None, :, 0])
    dy = np.abs(P[:, None, 1] - P[None, :, 1])
    h = np.hypot(dx, dy)
    w = 1.0 / ((1 - dx) * (1 - dy) * lam[:, None] * lam[None, :])
    off = ~np.eye(n, dtype=bool)
    h, w = h[off], w[off]
    order = np.argsort(h)
    cw = np.concatenate([[0.0], np.cumsum(w[order])])
    return cw[np.searchsorted(h[order], r, side="right")]


def fit_trend(P, m=30):
    """Log-linear intensity in log pop_density; returns beta, cov, lam_i."""
    Q, ind, a = quad_scheme(P, m)
    X = np.column_stack([np.ones(len(Q)), np.log(pop_density(Q))])
    beta, cov, _ = fit_bt(X, ind, a)
    lam = np.exp(beta[0] + beta[1] * np.log(pop_density(P)))
    return beta, cov, lam
```

First one pattern.

```python
kap2, mu2 = 25.0, 30.0                 # kappa * mu_c = 750 before thinning
P = sim_thinned_thomas(kap2, mu2, sigma_c, rng)
beta_h, cov_h, lam_i = fit_trend(P)
print(f"\nthinned Thomas pattern: n = {len(P)}")
print(f"beta0 = {beta_h[0]:.3f} (truth {np.log(kap2 * mu2):.3f}), "
      f"beta1 = {beta_h[1]:.3f} (truth 2), naive se(beta1) = "
      f"{np.sqrt(cov_h[1, 1]):.3f}")
Kh_ord = khat(P, r_grid)
Kh_inh = khat_inhom(P, r_grid, lam_i)
for name, Kh in (("ordinary K-hat", Kh_ord), ("K-hat inhom ", Kh_inh)):
    kk, ss, _, _ = fit_thomas(Kh, r_grid, len(P))
    print(f"{name}: kappa={kk:.1f}, sigma_c={ss:.4f}")
```

The log-linear fit recovers the trend: $\hat\beta_0 = 6.58$ (truth $6.62$) and $\hat\beta_1 = 2.24$ (truth $2$), with a naive standard error for $\hat\beta_1$ of $0.207$, so the estimate is about one standard error from the truth. For the clustering, fitting the ordinary $\hat{K}$ gives $\hat\kappa = 5.6$ and $\hat\sigma_c = 0.038$, while fitting $\hat{K}_{\text{inhom}}$ gives $\hat\kappa = 13.0$ and $\hat\sigma_c = 0.034$. They differ by a factor of $2$ in $\kappa$ and the truth is $25$, but a single pattern with $153$ points is a small sample. Repeat $100$ times.

```python
bs, naive, kk_o, kk_i, ss_o, ss_i = [], [], [], [], [], []
for _ in range(100):
    P = sim_thinned_thomas(kap2, mu2, sigma_c, rng)
    b, cv, lam_i = fit_trend(P)
    bs.append(b[1]); naive.append(np.sqrt(cv[1, 1]))
    ko, so, _, _ = fit_thomas(khat(P, r_grid), r_grid, len(P))
    ki, si, _, _ = fit_thomas(khat_inhom(P, r_grid, lam_i), r_grid, len(P))
    kk_o.append(ko); kk_i.append(ki); ss_o.append(so); ss_i.append(si)
print(f"\n100 patterns: mean beta1 = {np.mean(bs):.3f}, "
      f"sd = {np.std(bs):.3f}, "
      f"mean naive se = {np.mean(naive):.3f}")
print(f"median kappa: ordinary {np.median(kk_o):.1f}, inhomogeneous "
      f"{np.median(kk_i):.1f} (truth {kap2:.0f})")
print(f"median sigma_c: ordinary {np.median(ss_o):.4f}, inhomogeneous "
      f"{np.median(ss_i):.4f} (truth {sigma_c})")
```

| | Ordinary $\hat{K}$ | $\hat{K}_{\text{inhom}}$ | Truth |
| :--- | :--- | :--- | :--- |
| median $\hat\kappa$ | 9.5 | 24.5 | 25 |
| median $\hat\sigma_c$ | 0.0356 | 0.0298 | 0.03 |

The Poisson regression over the same $100$ patterns: mean $\hat\beta_1 = 1.913$ (truth $2$), standard deviation $0.775$, average reported standard error $0.201$.

Two things stand out.

- **The cluster fit needs $K_{\text{inhom}}$.** The median $\hat\kappa$ from the *ordinary* $K$ is $9.5$: the trend is mistaken for additional clustering, and the fit reports fewer, larger clusters ($\hat\sigma_c = 0.036$ against the truth $0.030$). The median from the *inhomogeneous* $K$ is $24.5$ and $0.0298$, almost exactly the truth of $25$ and $0.030$. This is Chapter 11's trap seen from the modeling side: a model fitted to the wrong summary is wrong in a specific way.
- **The naive standard error of $\hat\beta_1$ is far too small.** The estimates average $1.91$ (consistent for $\beta_1 = 2$), but their standard deviation across patterns is $0.775$, while the standard error reported by the Poisson regression averages $0.201$. The true variability is $3.9$ times what the GLM reports, because the GLM assumes independent points and the cluster process has positively correlated ones. This is the failure of Chapter 1, in a new setting: a correct estimate, with a standard error from a formula that does not apply. Chapter 11 offered a sandwich estimator and a parametric bootstrap as fixes; the inhomogeneous cluster model supplies the bootstrap's simulator.

### 12.2.7 The $K$-Function of the Matérn Cluster Process (Optional)

*This subsection gives the closed form of $K$ for the Matérn cluster process and checks it by simulation. It can be skipped on a first reading.*

The derivation of Section 12.2.4 holds for any dispersal density: $K(r) = \pi r^2 + P_{\text{sib}}(r)/\kappa$, where $P_{\text{sib}}(r)$ is the probability that two siblings are within $r$ of each other. For the Matérn cluster process each sibling is uniform on a disc of radius $R$ around the parent, so two siblings have a displacement-difference density proportional to the area of overlap of two discs of radius $R$ whose centers are $\lVert\mathbf{h}\rVert$ apart. That overlap (a "lens") has area $2R^2\arccos\!\left(\dfrac{\lVert\mathbf{h}\rVert}{2R}\right) - \dfrac{\lVert\mathbf{h}\rVert}{2}\sqrt{4R^2 - \lVert\mathbf{h}\rVert^2}$, and the density of $\mathbf{X}_1 - \mathbf{X}_2$ at $\mathbf{h}$ is the lens area divided by $(\pi R^2)^2$. Integrating over $\lVert\mathbf{h}\rVert \le r$ in polar coordinates and substituting $u = s/(2R)$ gives, with $z = r/(2R)$,

$$
P_{\text{sib}}(r) = \frac{16}{\pi}\int_0^z\left(u\arccos u - u^2\sqrt{1 - u^2}\right)du = 1 + \frac{2}{\pi}\left[(4z^2 - 1)\arccos z - z(1 + 2z^2)\sqrt{1 - z^2}\right]
$$

for $z \le 1$, and $P_{\text{sib}}(r) = 1$ for $r \ge 2R$ (siblings are never more than $2R$ apart). The second form follows from elementary integration (by parts for the first term). Two checks: at $z = 0$ the bracket is $-\pi/2$ and $P_{\text{sib}} = 0$, and at $z = 1$ the bracket is $0$ and $P_{\text{sib}} = 1$; and for small $z$, $P_{\text{sib}} \approx 4z^2 = r^2/R^2$, the chance that a sibling falls within $r$ of a given point of the disc, which is the ratio of areas $\pi r^2/(\pi R^2)$. The code compares the formula with simulation for $R = 0.06$.

```python
def K_matern_cluster(r, kappa, R):
    """K of the Matern cluster process; P_sib is the sibling-distance cdf."""
    z = np.minimum(r / (2 * R), 1.0)
    P_sib = 1 + (2 / np.pi) * ((4 * z**2 - 1) * np.arccos(z)
                               - z * (1 + 2 * z**2) * np.sqrt(1 - z**2))
    return np.pi * r**2 + P_sib / kappa


R_disc = 0.06
K_mat = np.array([khat(sim_matern_cluster(kappa, mu_c, R_disc, rng), r_grid)
                  for _ in range(300)])
print("   r     pi r^2   mean K-hat   Matern cluster K")
for rr in (0.02, 0.05, 0.10, 0.15):
    i = np.argmin(np.abs(r_grid - rr))
    print(f"{r_grid[i]:5.3f}  {np.pi * r_grid[i]**2:7.4f}  "
          f"{K_mat[:, i].mean():10.4f}  "
          f"{K_matern_cluster(r_grid[i], kappa, R_disc):12.4f}")
```

| $r$ | $\pi r^2$ | mean $\hat{K}(r)$ | Matérn cluster $K(r)$ |
| :--- | :--- | :--- | :--- |
| 0.02 | 0.0013 | 0.0052 | 0.0051 |
| 0.05 | 0.0079 | 0.0263 | 0.0260 |
| 0.10 | 0.0314 | 0.0694 | 0.0699 |
| 0.15 | 0.0707 | 0.1080 | 0.1107 |

The formula tracks the simulated average to the accuracy seen for the Thomas process, and the cluster boundary shows in $K$: for $r \ge 2R = 0.12$ the excess is exactly $1/\kappa = 0.04$, a plateau that the Thomas $K$ reaches only in the limit.

---

## 12.3 The Log-Gaussian Cox Process

The cluster processes of Section 12.2 are *mechanistic*: there are parents, and offspring gather around them. A second way to produce clustering needs no parents at all. Let the intensity itself be random.

### 12.3.1 The Model Equation Once More

A **Cox process** is a Poisson process whose intensity function $\Lambda(\mathbf{s})$ is itself a random surface. First the surface is drawn, then, given the surface, points are placed as an inhomogeneous Poisson process with that intensity. The **log-Gaussian Cox process** (LGCP) takes the log of the surface to be a Gaussian random field:

$$
\log\Lambda(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + u(\mathbf{s})
$$

This is the model equation of the book, $Y = \mu + u + \varepsilon$, with $Y$ the log-intensity, $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ the trend of Chapter 11, and $u(\mathbf{s})$ the spatial random effect of Part II, a mean-zero Gaussian process with covariance function $C_u(h)$. There is no separate nugget term $\varepsilon$, because the Poisson variation of the point counts plays that role. In this section we take the trend to be a constant, $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta} = \beta_0$, and give $u$ the exponential covariance $C_u(h) = \sigma^2 e^{-h/\phi}$ of Chapter 2.

An LGCP clusters for a reason you already know. Where $u$ is large the intensity is high, and points pile up; where $u$ is small there are few points. Nearby locations have similar values of $u$, so points close together tend to sit in the same high-intensity patch.

**The intensity.** Given the surface, the expected number of points in a small area is $\Lambda(\mathbf{s})\,d\mathbf{s}$. The unconditional intensity is the average of $\Lambda$ over the randomness of $u$. For a normal variable $U \sim N(0, v)$, the lognormal mean is $\operatorname{E}[e^U] = e^{v/2}$, so

$$
\lambda = \operatorname{E}[\Lambda(\mathbf{s})] = e^{\beta_0}\operatorname{E}[e^{u(\mathbf{s})}] = e^{\beta_0 + \sigma^2/2}
$$

The mean intensity is *not* $e^{\beta_0}$. This is the same lognormal correction we met in Chapter 4 when back-transforming log rents.

**The pair correlation function.** Given the surface, two distinct points at $\mathbf{x}$ and $\mathbf{y}$ are independent with intensities $\Lambda(\mathbf{x})$ and $\Lambda(\mathbf{y})$, so the expected number of ordered pairs is $\operatorname{E}[\Lambda(\mathbf{x})\Lambda(\mathbf{y})]$, averaged over the surface. Writing $\Lambda(\mathbf{x})\Lambda(\mathbf{y}) = e^{2\beta_0}e^{u(\mathbf{x}) + u(\mathbf{y})}$:

$$
\rho^{(2)}(\mathbf{x}, \mathbf{y}) = e^{2\beta_0}\operatorname{E}\!\left[e^{u(\mathbf{x}) + u(\mathbf{y})}\right]
$$

The sum $u(\mathbf{x}) + u(\mathbf{y})$ is normal with mean $0$ and variance $\sigma^2 + \sigma^2 + 2C_u(h) = 2\sigma^2 + 2C_u(h)$, where $h = \lVert\mathbf{x} - \mathbf{y}\rVert$. Applying the lognormal mean again with $v = 2\sigma^2 + 2C_u(h)$,

$$
\rho^{(2)} = e^{2\beta_0}\,e^{\sigma^2 + C_u(h)} = \left(e^{\beta_0 + \sigma^2/2}\right)^2 e^{C_u(h)} = \lambda^2 e^{C_u(h)}
$$

and therefore

$$
g(r) = \exp\!\big(C_u(r)\big), \qquad K(r) = \int_0^r 2\pi s\,\exp\!\big(\sigma^2 e^{-s/\phi}\big)\,ds
$$

Three things to read off. At $r = 0$, $g(0) = e^{\sigma^2} > 1$. Far apart, $C_u \to 0$ and $g \to 1$, so the process looks Poisson at large distances. And the clustering has the same covariance structure as the latent field: the range $\phi$ of the field is the range of the clustering. The integral for $K$ has no elementary closed form for the exponential covariance, but it is a one-dimensional integral that is easy to evaluate numerically, as the code in Section 12.3.2 does with the trapezoid rule.

**The Thomas process is also a Cox process.** Given the parent locations, the offspring of a Thomas process are independent points with intensity $\Lambda(\mathbf{s}) = \mu_c\sum_{\mathbf{c}}f(\mathbf{s} - \mathbf{c})$, where $f$ is the bivariate normal density with spread $\sigma_c$ of Section 12.2.2, summed over the parents. (Poisson offspring numbers with independent placements are the same as an inhomogeneous Poisson process with this intensity.) So the Thomas process is a Cox process whose random surface is a sum of bumps, one at each parent. The LGCP is a Cox process whose surface is the exponential of a Gaussian field. Both are random intensities; they differ in how the randomness is generated.

### 12.3.2 Simulating and Fitting an LGCP

An LGCP is simulated on a pixel grid. Divide the square into $m \times m$ pixels. Draw the field $u$ at the pixel centers with the Cholesky method of Chapter 2, set $\Lambda_k = \exp\!\big(\log\lambda - \sigma^2/2 + u_k\big)$ so that the mean intensity is $\lambda$, draw a Poisson count in each pixel with mean $\Lambda_k/m^2$, and scatter each pixel's points uniformly inside it. The discretization makes the intensity constant within a pixel, which slightly lowers $K$ at distances comparable to the pixel width; with $m = 40$ the pixel width is $0.025$, well below the field's range.

Our running story is the same tissue slide. Immune cells sometimes respond to a smooth, unobserved gradient of inflammatory signals instead of to discrete nests. Take mean intensity $\lambda = 200$, log-intensity variance $\sigma^2 = 1$, and range $\phi = 0.06$ (a practical range of $3\phi = 0.18$).

Fitting is minimum contrast again, with the LGCP $K$ in place of the Thomas $K$. The code below checks the formula against simulation, fits the two parameters, and then asks a question that will matter in Section 12.5: if we fit *both* models to *both* kinds of pattern, which fits better?

```python
def pixel_centers(m):
    g = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel()])


def exp_chol(m, sigma2, phi):
    C = pixel_centers(m)
    d = np.hypot(C[:, None, 0] - C[None, :, 0], C[:, None, 1] - C[None, :, 1])
    return np.linalg.cholesky(sigma2 * np.exp(-d / phi) + 1e-8 * np.eye(m * m))


def sim_lgcp(lam, L, rng, m=40, with_field=False):
    """LGCP on m x m pixels with mean intensity lam; L = chol(Var(u))."""
    s2 = np.sum(L[0]**2)                       # Var(u) at one pixel
    u = L @ rng.standard_normal(m * m)
    Lam = np.exp(np.log(lam) - s2 / 2 + u)
    counts = rng.poisson(Lam / m**2)
    cell = np.repeat(np.arange(m * m), counts)
    P = np.column_stack([(cell % m + rng.uniform(size=len(cell))) / m,
                         (cell // m + rng.uniform(size=len(cell))) / m])
    return (P, Lam) if with_field else P


def K_lgcp(r, sigma2, phi):
    """K for an exponential-covariance LGCP: integral of 2 pi s exp(C(s))."""
    s = np.linspace(0.0, r.max(), 2001)
    f = 2 * np.pi * s * np.exp(sigma2 * np.exp(-s / phi))
    cum = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(s))])
    return np.interp(r, s, cum)
```

**What the functions do.** `sim_lgcp` takes the Cholesky factor `L` of the covariance matrix of $\mathbf{u}$ on the pixel grid (Chapter 2) and returns the points, and optionally the intensity surface used to draw them, so that the truth is available for checking. `K_lgcp` evaluates the integral for $K(r)$ by the trapezoid rule on a fine grid of distances. Now the first check: does the average of $\hat{K}$ over simulated LGCP patterns follow the integral?

```python
lam_c, s2_c, phi_c, m_c = 200.0, 1.0, 0.06, 40
L_c = exp_chol(m_c, s2_c, phi_c)
K_lg = np.array([khat(sim_lgcp(lam_c, L_c, rng), r_grid) for _ in range(200)])
print("\n   r   pi r^2   mean K-hat   LGCP K   Thomas K (kappa=25)")
for rr in (0.02, 0.05, 0.10, 0.15):
    i = np.argmin(np.abs(r_grid - rr))
    print(f"{r_grid[i]:5.3f} {np.pi * r_grid[i]**2:7.4f} "
          f"{K_lg[:, i].mean():10.4f} "
          f"{K_lgcp(r_grid, s2_c, phi_c)[i]:8.4f} "
          f"{K_thomas(r_grid[i], kappa, sigma_c):9.4f}")
```

| $r$ | $\pi r^2$ | mean $\hat{K}(r)$ | LGCP $K(r)$ | Thomas $K(r)$ ($\kappa = 25$) |
| :--- | :--- | :--- | :--- | :--- |
| 0.02 | 0.0013 | 0.0027 | 0.0028 | 0.0055 |
| 0.05 | 0.0079 | 0.0136 | 0.0142 | 0.0279 |
| 0.10 | 0.0314 | 0.0442 | 0.0455 | 0.0689 |
| 0.15 | 0.0707 | 0.0877 | 0.0900 | 0.1106 |

The LGCP formula follows the simulated average closely: at $r = 0.05$, $0.0142$ against $0.0136$; at $r = 0.15$, $0.0900$ against $0.0877$. The simulation sits a few percent lower, as expected from the piecewise-constant intensity. The Thomas $K$ for the same intensity is about twice as large at short distances, because the Thomas parameters $\kappa = 25$ and $\sigma_c = 0.03$ were chosen for a much tighter kind of clustering.

Now fit. The contrast is the same as before, with $K_{\text{LGCP}}(r;\sigma^2,\phi)$ in place of $K_{\text{Thomas}}$. The block also fits both models to both kinds of pattern, to answer a question that matters for Section 12.5: if the data are clustered, can the $K$-function tell a Thomas process from a log-Gaussian Cox process?

```python
def fit_lgcp(K_hat, r, c=0.25):
    def res(p):
        return K_hat**c - K_lgcp(r, *np.exp(p))**c
    sol = optimize.least_squares(res, np.log([0.5, 0.05]),
                                 bounds=(np.log([0.01, 0.005]),
                                         np.log([10.0, 1.0])))
    return np.exp(sol.x), np.sum(sol.fun**2) * (r[1] - r[0])


est = np.array([fit_lgcp(Kh, r_grid)[0] for Kh in K_lg[:100]])
print(f"100 patterns: median sigma2 = {np.median(est[:, 0]):.2f} "
      f"(truth {s2_c}), median phi = {np.median(est[:, 1]):.3f} "
      f"(truth {phi_c})")

# the same two models fitted to each kind of pattern
P_lg, Lam_true = sim_lgcp(lam_c, L_c, rng, with_field=True)
K_lgp = khat(P_lg, r_grid)
K_thp = khat(slide, r_grid)
print("\npattern        model     minimum contrast")
for pname, Kh, npts in (("Thomas slide", K_thp, n_slide),
                        ("LGCP slide", K_lgp, len(P_lg))):
    D_t = fit_thomas(Kh, r_grid, npts)[3]
    D_l = fit_lgcp(Kh, r_grid)[1]
    print(f"{pname:13s}  Thomas    {D_t:.2e}")
    print(f"{'':13s}  LGCP      {D_l:.2e}")
```

| Pattern | Thomas contrast | LGCP contrast |
| :--- | :--- | :--- |
| Thomas slide | $1.76 \times 10^{-6}$ | $1.55 \times 10^{-5}$ |
| LGCP slide | $6.02 \times 10^{-6}$ | $5.15 \times 10^{-6}$ |

Across $100$ simulated patterns, the median estimates are $\hat\sigma^2 = 0.91$ and $\hat\phi = 0.058$, against the truth $1$ and $0.06$: the same slight underestimate of the variance as in the Thomas fit, and the range essentially right.

The comparison of the two models is the surprise. On the Thomas slide, the Thomas model has the smaller contrast, $1.8 \times 10^{-6}$ against $1.6 \times 10^{-5}$, a ratio of nearly nine. That fits the truth. On the LGCP slide, though, the contrast is $6.0 \times 10^{-6}$ for the Thomas model and $5.2 \times 10^{-6}$ for the LGCP: the true model wins by a margin of about $15\%$, which is within the noise of a single pattern. With about $200$ points and a two-parameter fit to $K$, the Gaussian-cluster and Gaussian-field descriptions are hard to tell apart. This should not be surprising: the Thomas process is a Cox process too (Section 12.3.1), and two random surfaces with similar length scales and variability produce similar pair statistics. What separates them is higher-order structure, such as nearest-neighbor distances, which the $K$-function does not see.

### 12.3.3 A Pixel Grid: Parts II and III Meet the Poisson GLM

The LGCP is the place where three parts of the book meet. On a grid of $m \times m$ pixels, let $y_k$ be the number of points in pixel $k$ and $|\text{pixel}| = 1/m^2$ its area. The LGCP says that given the field,

$$
y_k \mid u_k \sim \text{Poisson}\!\left(|\text{pixel}|\,\exp(\beta_0 + u_k)\right)
$$

which is the spatial Poisson GLMM of Chapter 9 with the pixel area playing the role of the expected count $E_k$. The random effect $\mathbf{u}$ can be given a prior in two ways. In Part II's way, $\mathbf{u} \sim N(\mathbf{0}, \boldsymbol{\Sigma})$ with the exponential covariance, and fitting needs the dense $\boldsymbol{\Sigma}$ and its inverse. In Part III's way, $\mathbf{u}$ gets a Gaussian Markov random field prior with sparse precision $\mathbf{Q}$, for example the Leroux precision $\mathbf{Q} = \sigma^{-2}\left[\rho(\mathbf{D} - \mathbf{A}) + (1 - \rho)\mathbf{I}\right]$ of Chapter 9 on the rook adjacency of the pixels. The GMRF is only an approximation to the exponential covariance field, but it is a good one for smooth fields and it leads to fast computation. It is the $\alpha = 1$ member of the family of Section 13.5: a practical smoother, not a Matérn field with finite variance as the grid is refined, and Chapter 13 explains what a closer Matérn match needs. Fitting is the penalized IRLS of Chapter 9: with working weights $\mathbf{V} = \operatorname{diag}(\mu_k)$ and working response $z_k = \eta_k + (y_k - \mu_k)/\mu_k$, one solves Henderson's equations

$$
\begin{bmatrix}\mathbf{1}'\mathbf{V}\mathbf{1} & \mathbf{1}'\mathbf{V} \\ \mathbf{V}\mathbf{1} & \mathbf{V} + \mathbf{Q}\end{bmatrix}\begin{bmatrix}\beta_0 \\ \mathbf{u}\end{bmatrix} = \begin{bmatrix}\mathbf{1}'\mathbf{V}\mathbf{z} \\ \mathbf{V}\mathbf{z}\end{bmatrix}
$$

repeatedly, and the variance scale $\sigma^2$ is chosen by maximizing the Laplace approximation of the marginal likelihood of Chapter 9, $\ell(\hat\beta_0, \hat{\mathbf{u}}) - \tfrac{1}{2}\hat{\mathbf{u}}'\mathbf{Q}\hat{\mathbf{u}} + \tfrac{1}{2}\log\lvert\mathbf{Q}\rvert - \tfrac{1}{2}\log\lvert\mathbf{V} + \mathbf{Q}\rvert$, with the constants that do not depend on $\sigma^2$ dropped. The code restates these two pieces, then fits the LGCP slide of the previous block on a $20 \times 20$ grid of pixels, with $\rho = 0.95$ fixed. The truth was simulated on a $40 \times 40$ grid, so each fitting pixel contains four of the finer pixels and the target is the log of their average intensity.

```python
# the same field on a coarser pixel grid, as a Poisson GLMM with a GMRF
g_fit = 20
ij = np.minimum((P_lg * g_fit).astype(int), g_fit - 1)
y_pix = np.bincount(ij[:, 1] * g_fit + ij[:, 0],
                    minlength=g_fit**2).astype(float)
true_pix = Lam_true.reshape(g_fit, m_c // g_fit, g_fit, m_c // g_fit)
true_log = np.log(true_pix.mean(axis=(1, 3)).ravel())


def rook_laplacian(g):
    n = g * g
    A = np.zeros((n, n))
    for i in range(g):
        for j in range(g):
            k = i * g + j
            if i + 1 < g:
                A[k, k + g] = A[k + g, k] = 1
            if j + 1 < g:
                A[k, k + 1] = A[k + 1, k] = 1
    return np.diag(A.sum(axis=1)) - A


def fit_pixel_glmm(y, Lap, sigma2, rho=0.95, iters=60):
    """Penalized IRLS for log mu = log(area) + b0 + u, Leroux precision Q."""
    n = len(y)
    area = 1.0 / n
    Q = (rho * Lap + (1 - rho) * np.eye(n)) / sigma2
    b0, u = np.log(y.sum() / n / area), np.zeros(n)
    for _ in range(iters):
        eta = b0 + u
        mu = area * np.exp(eta)
        z = eta + (y - mu) / mu
        H = np.zeros((n + 1, n + 1))
        H[0, 0], H[0, 1:], H[1:, 0] = mu.sum(), mu, mu
        H[1:, 1:] = np.diag(mu) + Q
        sol = np.linalg.solve(H, np.concatenate([[mu @ z], mu * z]))
        done = np.max(np.abs(sol[0] - b0)) + np.max(np.abs(sol[1:] - u)) < 1e-9
        b0, u = sol[0], sol[1:]
        if done:
            break
    eta = b0 + u
    mu = area * np.exp(eta)
    # Laplace approximation to the log marginal likelihood (up to constants)
    _, logdet_Q = np.linalg.slogdet(Q)
    _, logdet_H = np.linalg.slogdet(np.diag(mu) + Q)
    marg = (np.sum(y * eta - mu) - 0.5 * u @ Q @ u
            + 0.5 * logdet_Q - 0.5 * logdet_H)
    return b0, u, marg


Lap = rook_laplacian(g_fit)
print(f"\n{len(P_lg)} points, {g_fit * g_fit} pixels, "
      f"{np.mean(y_pix == 0):.0%} of pixels empty")
print("sigma2   Laplace log marginal   corr with true log intensity")
best = None
for s2 in (0.1, 0.3, 1.0, 3.0, 10.0):
    b0, u, marg = fit_pixel_glmm(y_pix, Lap, s2)
    corr = np.corrcoef(b0 + u, true_log)[0, 1]
    print(f"{s2:6.1f}   {marg:19.2f}   {corr:20.3f}")
    if best is None or marg > best[0]:
        best = (marg, s2, b0, u)
raw_corr = np.corrcoef(np.log(y_pix + 0.5), true_log)[0, 1]
print(f"best sigma2 = {best[1]}; "
      f"raw log(count + 0.5) correlates {raw_corr:.3f}")
```

| $\sigma^2$ | Laplace log marginal | correlation with true log intensity |
| :--- | :--- | :--- |
| 0.1 | 978.82 | 0.619 |
| 0.3 | 989.16 | 0.667 |
| 1.0 | 995.51 | 0.721 |
| 3.0 | 983.69 | 0.722 |
| 10.0 | 940.29 | 0.672 |

The slide has $220$ points on $400$ pixels, and $62\%$ of the pixels are empty, so each pixel carries little information. The Laplace marginal likelihood is largest at $\sigma^2 = 1.0$ (log marginal $995.5$, against $989.2$ at $0.3$ and $983.7$ at $3.0$), and at that value the fitted log-intensity has a correlation of $0.72$ with the true one. The raw counts, $\log(y_k + 0.5)$, correlate only $0.53$: borrowing strength from neighbors through the GMRF prior improves the reconstruction substantially, as Chapter 9's disease maps did. The correlation is highest near the marginal-likelihood optimum, $0.721$ at $\sigma^2 = 1$ and $0.722$ at $\sigma^2 = 3$, and falls to $0.62$ at $\sigma^2 = 0.1$ when the small variance over-smooths the fit; the marginal likelihood picks a good value without knowing the truth. Figure 12.7 in Section 12.6 shows the result: the fitted surface finds the high-intensity patches in the upper left, along the right edge, and in the lower left, but it is smoother and lower in resolution than the truth.

---

## 12.4 Inhibition

Clustering says that points attract. Inhibition says the opposite: given a point at some location, other points are *less* likely to appear nearby. Tumor cells cannot overlap, so a slide of them has a minimum spacing. Trees compete for light and water, so seedlings near a mature tree die. Chain restaurants avoid opening next door to each other. In each case $g(r) < 1$ for small $r$, and $K(r) < \pi r^2$.

Here our story is the tumor cells themselves. Immune cells (Section 12.2) gather; the tumor cells they gather around keep their distance from each other. All the examples in this section are simulated, with interaction distances in millimeters on the same $1\text{ mm} \times 1\text{ mm}$ field.

### 12.4.1 Matérn Hard-Core Processes

The simplest inhibition model starts with a Poisson pattern and *deletes* points that are too close to one another. Fix a **hard-core distance** $r_0$. Start from a homogeneous Poisson process with intensity $\lambda_{\text{par}}$, the **parent** pattern (a name borrowed from Section 12.2, though nothing here has offspring). Matérn proposed two ways to delete.

- **Type I.** Delete *every* point that has another point within distance $r_0$. If two points are close, both go.
- **Type II.** Give each point an independent random mark $t$ uniform on $(0, 1)$, think of it as an arrival time. Delete a point when another point lies within $r_0$ *and has an earlier mark*. Of a close pair, only the later arrival is deleted, and a point survives if no earlier point is within $r_0$ of it.

Both leave a pattern in which no two points are closer than $r_0$: a hard core. They differ in how many points survive, and we can compute it.

**The intensity of Type I.** A parent point survives if no other parent is within $r_0$. The other parents form a Poisson process, so the number of them in a disc of area $\pi r_0^2$ is Poisson with mean $\lambda_{\text{par}}\pi r_0^2$, and the chance of none is $e^{-\lambda_{\text{par}}\pi r_0^2}$. The intensity of survivors is therefore

$$
\lambda_I = \lambda_{\text{par}}\,e^{-\lambda_{\text{par}}\pi r_0^2}
$$

**The intensity of Type II.** A parent with mark $t$ survives if no *earlier* parent is within $r_0$. The earlier parents, those with mark below $t$, form a thinned Poisson process with intensity $\lambda_{\text{par}} t$. So the survival probability given $t$ is $e^{-\lambda_{\text{par}} t\pi r_0^2}$. Averaging over $t \sim \text{Uniform}(0, 1)$ with $a = \lambda_{\text{par}}\pi r_0^2$,

$$
\int_0^1 e^{-at}\,dt = \frac{1 - e^{-a}}{a}, \qquad\text{so}\qquad \lambda_{II} = \lambda_{\text{par}}\cdot\frac{1 - e^{-a}}{a} = \frac{1 - e^{-\lambda_{\text{par}}\pi r_0^2}}{\pi r_0^2}
$$

**Why both cap the intensity.** However many parents we start with, the number of survivors is limited.

- $\lambda_I = \lambda_{\text{par}} e^{-a}$ with $a = \lambda_{\text{par}}\pi r_0^2$ rises, peaks, and then *falls*: it is maximal at $\lambda_{\text{par}} = 1/(\pi r_0^2)$ with value $1/(e\pi r_0^2)$. Past that, so many points are crowded that almost every one has a neighbor, and Type I deletes nearly everything. Starting with more points gives fewer points.
- $\lambda_{II} = (1 - e^{-a})/(\pi r_0^2)$ increases with $\lambda_{\text{par}}$ and approaches the ceiling $1/(\pi r_0^2)$ without reaching it.

Neither comes close to the densest possible arrangement of discs, hexagonal packing with intensity $2/(\sqrt{3}r_0^2)$: points with a minimum distance $r_0$ can be at most that dense. Matérn's thinnings are far from packing, because they are *random* and never rearrange points to make room.

We take $r_0 = 0.05$. The code simulates both types on the unit square, buffered by $r_0$ so that points near the edge are treated like the rest, and compares the average number of survivors with the formulas.

```python
def sim_matern_hc(lam0, r0, rng, kind=1):
    """Matern hard-core I (kind=1) or II (kind=2) on the unit square."""
    lo, hi = -r0, 1 + r0                      # buffer so edges are not special
    n = rng.poisson(lam0 * (hi - lo) ** 2)
    P = rng.uniform(lo, hi, size=(n, 2))
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    close = (d < r0) & ~np.eye(n, dtype=bool)
    if kind == 1:
        keep = ~close.any(axis=1)             # any neighbor: delete
    else:
        marks = rng.uniform(size=n)           # earlier neighbor: delete
        keep = ~(close & (marks[None, :] < marks[:, None])).any(axis=1)
    P = P[keep]
    return P[np.all((P >= 0) & (P <= 1), axis=1)]


def sim_ssi(r0, attempts, rng):
    """Simple sequential inhibition on the unit square."""
    P = np.empty((attempts, 2))
    n = 0
    for u in rng.uniform(size=(attempts, 2)):
        if n == 0 or np.all(np.hypot(P[:n, 0] - u[0], P[:n, 1] - u[1]) >= r0):
            P[n] = u
            n += 1
    return P[:n]
```

```python
r0_hc = 0.05
disc = np.pi * r0_hc**2
print("\nlam0   type I: mean n   formula    type II: mean n   formula")
for lam0 in (100, 300, 1000, 2000):
    n1 = np.mean([len(sim_matern_hc(lam0, r0_hc, rng, 1)) for _ in range(100)])
    n2 = np.mean([len(sim_matern_hc(lam0, r0_hc, rng, 2)) for _ in range(100)])
    print(f"{lam0:5d}  {n1:12.1f}  {lam0 * np.exp(-lam0 * disc):8.1f}  "
          f"{n2:15.1f}  {(1 - np.exp(-lam0 * disc)) / disc:8.1f}")
print(f"largest possible type I intensity  1/(e pi r0^2) = "
      f"{1 / (np.e * disc):.1f}")
print(f"supremum for type II               1/(pi r0^2)   = {1 / disc:.1f}")
print(f"densest packing of discs           2/(sqrt(3) r0^2) = "
      f"{2 / (np.sqrt(3) * r0_hc**2):.1f}")
```

| $\lambda_{\text{par}}$ | Type I, simulated | Type I, formula | Type II, simulated | Type II, formula |
| :--- | :--- | :--- | :--- | :--- |
| 100 | 44.3 | 45.6 | 68.6 | 69.3 |
| 300 | 28.5 | 28.4 | 115.4 | 115.3 |
| 1000 | 0.3 | 0.4 | 127.4 | 127.3 |
| 2000 | 0.0 | 0.0 | 126.7 | 127.3 |

The simulated mean counts follow the formulas. The two types then go in opposite directions as $\lambda_{\text{par}}$ grows. Type I has already peaked: the largest intensity it can reach is $1/(e\pi r_0^2) = 46.8$, near $\lambda_{\text{par}} = 1/(\pi r_0^2) = 127$, and by $\lambda_{\text{par}} = 1000$ almost everything is deleted, and by $2000$ nothing survives. Type II climbs toward its ceiling of $1/(\pi r_0^2) = 127.3$. Hexagonal packing would allow $461.9$, more than three times as many: Matérn's thinnings leave a great deal of room unused.

### 12.4.2 Simple Sequential Inhibition

A third process fills space more efficiently. **Simple sequential inhibition** (SSI) proposes points one at a time, uniformly in the window, and *accepts* a proposal only if it is at least $r_0$ from every point accepted so far. Rejected proposals are discarded. It continues for a fixed number of proposals, or until nothing more can be added.

The intensity depends on how long we keep trying. Early proposals are nearly always accepted. As the window fills, gaps that are big enough for a new disc get scarcer, and acceptance falls toward zero. The pattern saturates at the **jamming limit**, a famous constant of random sequential packing: for discs of diameter $r_0$ placed at random, the covered fraction approaches $0.5472$ in the plane, so the intensity approaches $0.5472/(\pi r_0^2/4) = 0.6967/r_0^2$. (This is a result from the literature, which we quote and check by simulation.) In a finite window, points near the edge have fewer neighbors to avoid, and a few more points fit; the count at the limit is a little higher than the planar formula.

The code below counts the points kept after $500$ up to $100{,}000$ proposals.

```python
print("\nattempts   points kept (mean of 5)")
for att in (500, 2000, 10000, 50000, 100000):
    kept = np.mean([len(sim_ssi(r0_hc, att, rng)) for _ in range(5)])
    print(f"{att:8d}   {kept:8.1f}")
print(f"random sequential adsorption limit 0.5472/(pi r0^2/4) = "
      f"{0.5472 / (np.pi * r0_hc**2 / 4):.1f}")
```

| Proposals | 500 | 2000 | 10,000 | 50,000 | 100,000 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Points kept (mean of 5) | 171.4 | 230.4 | 266.8 | 278.8 | 287.0 |

The count grows quickly at first and then flattens. The planar jamming limit is $278.7$. The finite window lets the count creep a few percent above it ($287.0$ is $3\%$ over) because points near an edge have fewer neighbors to avoid. At about $280$ to $290$ points, sequential inhibition packs more than twice as many points as Matérn Type II can reach ($127$), and still less than two thirds of the hexagonal $461.9$.

### 12.4.3 Gibbs Processes and the Strauss Model

Matérn's hard-core processes are defined by a *recipe* (thin a Poisson process). A **Gibbs process** is defined by a *probability density* and so can be fitted by a likelihood-type method. The **Strauss process** is the simplest Gibbs process with soft repulsion.

For a configuration $\mathbf{S} = \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ of $n$ points in $\mathcal{D}$, let

$$
\operatorname{pairs}(\mathbf{S}) = \text{the number of pairs } \{i, j\} \text{ with } \lVert\mathbf{s}_i - \mathbf{s}_j\rVert \le r_0
$$

be the number of *close pairs*. The Strauss process has density, with respect to a unit-intensity Poisson process,

$$
f(\mathbf{S}) = \frac{1}{Z(b, \gamma)}\,b^{n}\,\gamma^{\operatorname{pairs}(\mathbf{S})}, \qquad 0 \le \gamma \le 1
$$

where $b > 0$ controls the overall number of points and $\gamma$ is the penalty per close pair. The normalizing constant is $Z(b, \gamma) = \sum_{n \ge 0}\dfrac{b^n}{n!}\int_{\mathcal{D}^n}\gamma^{\operatorname{pairs}(\mathbf{S})}\,d\mathbf{s}_1\cdots d\mathbf{s}_n$. We write $\zeta = \log\gamma \le 0$ for the **interaction parameter**.

Three special cases show what $\gamma$ does. With $\gamma = 1$ the density is $b^n/Z$ and the process is Poisson with intensity $b$. With $0 < \gamma < 1$ each close pair multiplies the density by $\gamma$, so patterns with many close pairs are rarer. With $\gamma = 0$ any pattern with a close pair has density zero: a hard core. Letting $\zeta \to -\infty$ approaches the hard-core process.

**Why $\gamma \le 1$.** Suppose $\gamma > 1$. Consider configurations with all $n$ points inside one disc of diameter $r_0$ (which fits in $\mathcal{D}$). Every pair is close, so $\operatorname{pairs}(\mathbf{S}) = n(n-1)/2$ and these configurations contribute at least $\dfrac{b^n}{n!}\gamma^{n(n-1)/2}(\pi r_0^2/4)^n$ to $Z$. For large $n$, $\gamma^{n(n-1)/2}$, which grows like $\gamma^{n^2/2}$, overwhelms the factorial $n!$, so the series for $Z$ diverges and no probability distribution exists. A model that rewards close pairs would run away into one giant clump. That is why $\zeta \le 0$ is a requirement of the model, not just a convention, and why the Strauss model describes inhibition only. (Clustering needs the models of Sections 12.2 and 12.3.)

**Why $Z$ is intractable.** To evaluate $Z$ we would integrate $\gamma^{\operatorname{pairs}(\mathbf{S})}$ over all positions of $n$ points, for every $n$. The close-pair count couples all the points together, so the integral does not separate into $n$ one-dimensional integrals. With $n = 100$ points it is a $200$-dimensional integral of a function that jumps whenever a pair crosses distance $r_0$. There is no closed form, and numerical evaluation is hopeless. Maximizing $\log f(\mathbf{S}) = n\log b + \operatorname{pairs}(\mathbf{S})\zeta - \log Z$ over $(b, \zeta)$ would need $\log Z$ at every trial value.

**The conditional intensity.** The device that avoids $Z$ is to describe the process through *ratios* of densities. For a configuration $\mathbf{S}$ and a new location $\mathbf{u} \notin \mathbf{S}$, adding $\mathbf{u}$ adds one point and creates as many new close pairs as there are points of $\mathbf{S}$ within $r_0$ of $\mathbf{u}$. Call that number

$$
t(\mathbf{u}, \mathbf{S}) = \sum_{j}\mathbb{1}\big(\lVert\mathbf{u} - \mathbf{s}_j\rVert \le r_0\big)
$$

Then $\operatorname{pairs}(\mathbf{S} \cup \mathbf{u}) = \operatorname{pairs}(\mathbf{S}) + t(\mathbf{u}, \mathbf{S})$, and the normalizing constant cancels in the ratio:

$$
\lambda(\mathbf{u} \mid \mathbf{S}) = \frac{f(\mathbf{S} \cup \mathbf{u})}{f(\mathbf{S})} = b\,\gamma^{\,t(\mathbf{u}, \mathbf{S})}, \qquad \log\lambda(\mathbf{u} \mid \mathbf{S}) = \log b + \zeta\,t(\mathbf{u}, \mathbf{S})
$$

This **conditional intensity** is the intensity of a point at $\mathbf{u}$ *given the rest of the pattern*. Read it as a Poisson rule with a penalty: with no neighbors within $r_0$ the intensity is the base value $b$; each neighbor multiplies it by $\gamma$. When $\mathbf{u}$ is itself a point of the pattern we use the rest of the pattern, $\lambda(\mathbf{s}_i \mid \mathbf{S} \setminus \mathbf{s}_i)$, so a point is never its own neighbor.

To let the base intensity depend on covariates, replace $\log b$ by $\mathbf{x}(\mathbf{u})'\boldsymbol{\beta}$:

$$
\log\lambda(\mathbf{u} \mid \mathbf{S}) = \mathbf{x}(\mathbf{u})'\boldsymbol{\beta} + \zeta\,t(\mathbf{u}, \mathbf{S}), \qquad \zeta \le 0
$$

This inhomogeneous Strauss model is the one fitted, with a constant base intensity, in Section 12.4.5.

**Simulating a Strauss process.** Because $Z$ is unknown, we cannot draw from $f$ directly. A Markov chain can: the **birth-death Metropolis-Hastings** algorithm moves between patterns by adding or removing one point at a time. At each step, with probability $1/2$ propose a *birth*, a new point $\mathbf{u}$ uniform on $\mathcal{D}$, or with probability $1/2$ propose a *death*, removing one of the $n$ current points chosen uniformly. To find acceptance probabilities, consider a pattern $\mathbf{S}$ with $n$ points and the pattern $\mathbf{S} \cup \mathbf{u}$ with $n + 1$. Detailed balance requires

$$
f(\mathbf{S})\cdot\tfrac{1}{2}\cdot\frac{1}{\operatorname{area}(\mathcal{D})}\cdot a_{\text{birth}} = f(\mathbf{S} \cup \mathbf{u})\cdot\tfrac{1}{2}\cdot\frac{1}{n + 1}\cdot a_{\text{death}}
$$

which is solved by accepting a birth with probability $\min\!\Big\{1, \dfrac{\lambda(\mathbf{u} \mid \mathbf{S})\operatorname{area}(\mathcal{D})}{n + 1}\Big\}$ and a death of $\mathbf{s}_i$ with probability $\min\!\Big\{1, \dfrac{n}{\operatorname{area}(\mathcal{D})\,\lambda(\mathbf{s}_i \mid \mathbf{S} \setminus \mathbf{s}_i)}\Big\}$. Only the conditional intensity appears, so $Z$ is never needed. After enough steps the chain's pattern is a draw from the Strauss process.

### 12.4.4 The Matrix View: One More Column

Section 12.4.3 defined $t(\mathbf{u}, \mathbf{S})$ as a count. In matrix terms it is a row sum. Let $\mathbf{H}$ be the matrix of distances between the $n$ data points, as in Chapter 10, and let

$$
\mathbf{I}_{r_0} = \big[\mathbb{1}(h_{ij} \le r_0)\big]_{i \ne j}
$$

be the indicator matrix of close pairs, with zeros on the diagonal. Row $i$ lists the neighbors of point $i$, and the row sum $t_i = \sum_j (\mathbf{I}_{r_0})_{ij}$ is the neighbor count at data point $i$, excluding the point itself. For a dummy point $\mathbf{u}$ not in the pattern, the count is the row sum of the *cross* indicator matrix between $\mathbf{u}$ and the data points.

Now recall the Berman-Turner device of Chapter 11. The Poisson log-likelihood $\sum_i\log\lambda(\mathbf{s}_i) - \int_{\mathcal{D}}\lambda(\mathbf{u})\,d\mathbf{u}$ is approximated on a set of $n + n_d$ quadrature points $\mathbf{s}_k^*$ (the data points plus $n_d$ dummy points on a grid) with area weights $a_k$, and maximizing

$$
\sum_{k=1}^{n + n_d}a_k\Big(y_k\log\lambda(\mathbf{s}_k^*) - \lambda(\mathbf{s}_k^*)\Big), \qquad y_k = \frac{\mathbb{1}_k}{a_k}
$$

($\mathbb{1}_k = 1$ for data points, $0$ for dummy points) is a weighted Poisson regression, solved by IRLS. In Chapter 11 the log-intensity was $\mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ with covariates known at every location. For the Strauss model the log-intensity is $\log b + \zeta\,t(\mathbf{s}_k^*, \mathbf{S})$, and for every quadrature point $t$ is a number we can compute from the data and $r_0$. So $t$ is *one more covariate column*:

$$
\mathbf{X}_{\text{quad}} = \begin{bmatrix} 1 & t(\mathbf{s}_1^*, \mathbf{S}) \\ \vdots & \vdots \\ 1 & t(\mathbf{s}_{n + n_d}^*, \mathbf{S}) \end{bmatrix}, \qquad \log\boldsymbol{\lambda} = \mathbf{X}_{\text{quad}}\begin{bmatrix}\log b \\ \zeta\end{bmatrix}
$$

Nothing else changes. The interaction is not a new kind of model that needs new software; it is a covariate that happens to be computed from the response. Covariates $\mathbf{x}(\mathbf{u})$ for an inhomogeneous Strauss model would be added as further columns.

**A hand-sized example.** Four points at $(0.20, 0.20)$, $(0.23, 0.22)$, $(0.26, 0.18)$, and $(0.70, 0.60)$, and $r_0 = 0.06$. The distances are $0.036$ between the first two points, $0.050$ between the second and third, and $0.063$ between the first and third, which is just over $r_0$; the fourth point is far from the others. So the first and second points are neighbors, and so are the second and third. Add two dummy points, $(0.24, 0.20)$ in the middle of the group and $(0.50, 0.50)$ in empty space. The code builds the indicator matrix, the neighbor counts, and the relative conditional intensities $\gamma^t$ for $\gamma = 0.15$.

```python
S4 = np.array([[0.20, 0.20], [0.23, 0.22], [0.26, 0.18], [0.70, 0.60]])
r0_toy = 0.06
H4 = np.hypot(S4[:, None, 0] - S4[None, :, 0], S4[:, None, 1] - S4[None, :, 1])
I4 = (H4 <= r0_toy) & ~np.eye(4, dtype=bool)
print("\nindicator matrix of pairs within 0.06 (diagonal excluded):")
print(I4.astype(int))
print("t at the data points (row sums):", I4.sum(axis=1))
dummy_toy = np.array([[0.24, 0.20], [0.50, 0.50]])
Hd = np.hypot(dummy_toy[:, None, 0] - S4[None, :, 0],
              dummy_toy[:, None, 1] - S4[None, :, 1])
print("t at two dummy points:", (Hd <= r0_toy).sum(axis=1))
gamma_toy = 0.15
print("conditional intensity / beta = gamma^t:",
      np.round(gamma_toy ** np.concatenate([I4.sum(axis=1),
                                            (Hd <= r0_toy).sum(axis=1)]), 4))
```

The indicator matrix has a $1$ for the pair of points $1$ and $2$ ($0.036$ apart) and for the pair $2$ and $3$ ($0.050$ apart). The first and third points are $0.063$ apart, just over $r_0$, so they are not neighbors, and the fourth point is alone. The row sums are the neighbor counts $t = (1, 2, 1, 0)$. The dummy point in the middle of the group has all three nearby points within $0.06$, so $t = 3$; the dummy point in empty space has $t = 0$. The last line prints $\gamma^t$, the multiplier on the base intensity $b$: at the second point, which has two neighbors, it is $0.15^2 = 0.0225$; at the dummy point in the middle of the group it is $0.15^3 = 0.0034$. The model says that a new point at the center of that group is about $300$ times less likely than a point in open space. That is what inhibition means in the fitted intensity.

The function below builds the design matrix for a whole pattern. It reuses `quad_scheme` from Section 12.2.6 and computes the neighbor counts for every quadrature point, subtracting $1$ at the data points, where the distance to themselves is zero and would otherwise count.

```python
def strauss_X(P, r0, m=30):
    """Berman-Turner design matrix [1, t] for the Strauss model."""
    Q, ind, a = quad_scheme(P, m)
    d = np.hypot(Q[:, None, 0] - P[None, :, 0], Q[:, None, 1] - P[None, :, 1])
    t = (d <= r0).sum(axis=1) - ind           # a point is not its own neighbor
    return np.column_stack([np.ones(len(Q)), t]), ind, a
```

### 12.4.5 Maximum Pseudolikelihood

**The idea.** The full likelihood of a Strauss pattern is $\log f(\mathbf{S}) = n\log b + \zeta\operatorname{pairs}(\mathbf{S}) - \log Z$, and $Z$ cannot be computed. The **pseudolikelihood** replaces the likelihood by a product of conditional probabilities in which $Z$ cancels. To see it, divide $\mathcal{D}$ into many small cells $c$ of area $A_0$ around locations $\mathbf{u}_c$, and let $I_c = 1$ if the pattern has a point in cell $c$ and $0$ otherwise. Given the rest of the pattern, the chance that cell $c$ holds a point is, for small $A_0$, approximately $\lambda(\mathbf{u}_c \mid \mathbf{S})\,A_0$. This conditional probability involves only the ratio $f(\mathbf{S} \cup \mathbf{u}_c)/f(\mathbf{S})$ of Section 12.4.3, in which $Z$ cancels. Multiplying the conditional probabilities over cells,

$$
\log\text{PL} = \sum_c\Big[I_c\log\big(\lambda(\mathbf{u}_c \mid \mathbf{S})A_0\big) + (1 - I_c)\log\big(1 - \lambda(\mathbf{u}_c \mid \mathbf{S})A_0\big)\Big]
$$

Cells with a point contribute $\log\lambda + \log A_0$, and the second term for empty cells is $\approx -\lambda A_0$. Letting $A_0 \to 0$, and dropping the constant $n\log A_0$,

$$
\log\text{PL}(\boldsymbol{\beta}, \zeta) = \sum_{i=1}^n\log\lambda\big(\mathbf{s}_i \mid \mathbf{S}\setminus\mathbf{s}_i\big) - \int_{\mathcal{D}}\lambda(\mathbf{u} \mid \mathbf{S})\,d\mathbf{u}
$$

This is the Poisson log-likelihood of Chapter 11, $\sum_i\log\lambda(\mathbf{s}_i) - \int\lambda$, with the *conditional intensity* $\lambda(\mathbf{u} \mid \mathbf{S})$ in the place of the intensity. When $\zeta = 0$ the conditional intensity does not depend on the pattern and $\log\text{PL}$ is exactly the Poisson log-likelihood. That is the sense in which the pseudolikelihood is "Poisson-type": it treats the conditional intensity as if it were a fixed intensity function.

**Fitting is the Berman-Turner regression with the extra column.** The integral is approximated on the quadrature points exactly as in Section 12.4.4, and the pseudolikelihood is maximized by the same IRLS, with the design matrix $[\mathbf{1}, t]$. No new algorithm is needed. Under standard conditions the maximizer is consistent as the window grows (a result we take from the literature, Exercise 9 gives the estimating equation).

**What the pseudolikelihood is not.** It is not a likelihood. The conditional probabilities it multiplies are not independent, so the Fisher information of the Poisson regression, the matrix $\mathbf{X}'\operatorname{diag}(a_k\mu_k)\mathbf{X}$ from which the GLM reports standard errors, does not describe the uncertainty of $\hat\zeta$. The experiment below shows how far off it is. Differences in $\log\text{PL}$ cannot be used for likelihood-ratio tests either. Uncertainty comes from simulation: simulate from the fitted model with the birth-death algorithm of Section 12.4.3, refit each simulated pattern, and read the spread. This is a **parametric bootstrap**.

We simulate a slide of tumor cells with $b = 300$, $\gamma = 0.15$ (so $\zeta = \log 0.15 = -1.90$), and $r_0 = 0.06$ (a $60\ \mu\text{m}$ interaction distance), fit the model with the true $r_0$, run a parametric bootstrap with $60$ replicates, and then run $100$ fresh slides from the true model to learn the actual sampling variability.

```python
def sim_strauss(base, gamma, r0, rng, steps=10000, start=None):
    """Birth-death Metropolis-Hastings for a Strauss process on [0,1]^2."""
    P = np.empty((0, 2)) if start is None else start.copy()
    for _ in range(steps):
        n = len(P)
        if rng.uniform() < 0.5:                       # propose a birth
            u = rng.uniform(size=2)
            t = 0 if n == 0 else np.sum(np.hypot(*(P - u).T) <= r0)
            if rng.uniform() < base * gamma**t / (n + 1):
                P = np.vstack([P, u])
        elif n > 0:                                   # propose a death
            i = rng.integers(n)
            others = np.delete(P, i, axis=0)
            t = np.sum(np.hypot(*(others - P[i]).T) <= r0)
            if rng.uniform() < n / (base * gamma**t):
                P = others
    return P
```

```python
base_s, gamma_s, r0_s = 300.0, 0.15, 0.06
zeta_s = np.log(gamma_s)
tumor = sim_strauss(base_s, gamma_s, r0_s, rng)
X, ind, a = strauss_X(tumor, r0_s)
b_hat, cov_hat, ll_hat = fit_bt(X, ind, a)
n_pairs = int(X[:len(tumor), 1].sum() / 2)
print(f"\ntumor slide: {len(tumor)} cells, {n_pairs} pairs within {r0_s}")
print(f"b-hat = {np.exp(b_hat[0]):.0f} (truth {base_s:.0f}), "
      f"gamma-hat = {np.exp(b_hat[1]):.3f} (truth {gamma_s}), "
      f"zeta-hat = {b_hat[1]:.3f} (truth {zeta_s:.3f})")
print(f"naive se(zeta-hat) = {np.sqrt(cov_hat[1, 1]):.3f}")

boot = []
for _ in range(60):
    Pb = sim_strauss(np.exp(b_hat[0]), np.exp(b_hat[1]), r0_s, rng, steps=8000)
    Xb, ib, ab = strauss_X(Pb, r0_s)
    boot.append(fit_bt(Xb, ib, ab)[0])
boot = np.array(boot)
print(f"parametric bootstrap (60): sd(zeta-hat) = {boot[:, 1].std():.3f}, "
      f"sd(log b-hat) = {boot[:, 0].std():.3f}")

rep = []
for _ in range(100):
    Pr = sim_strauss(base_s, gamma_s, r0_s, rng, steps=8000)
    Xr, ir, ar = strauss_X(Pr, r0_s)
    br, cr, _ = fit_bt(Xr, ir, ar)
    rep.append((br[1], np.sqrt(cr[1, 1])))
rep = np.array(rep)
print(f"100 fresh slides: median zeta-hat = {np.median(rep[:, 0]):.3f}, "
      f"sd = {rep[:, 0].std():.3f}, "
      f"median naive se = {np.median(rep[:, 1]):.3f}")
```

The simulated slide has $91$ cells and $12$ pairs within $0.06$ of each other. The fit gives $\hat\gamma = 0.218$ and $\hat\zeta = -1.52$, against the truth $\gamma = 0.15$ and $\zeta = -1.90$. The estimate of the base intensity is $\hat{b} = 205$ against $300$: the two parameters trade off, since a weaker penalty can be partly compensated by a lower base intensity, and one slide with $12$ close pairs cannot pin both down.

The key numbers are the three standard deviations. The GLM's own standard error for $\hat\zeta$ is $0.215$. The parametric bootstrap gives $0.266$. And the standard deviation of $\hat\zeta$ over $100$ fresh slides from the true model is $0.327$, while the GLM standard error averages (median) $0.211$ over those slides. So the naive standard error understates the real variability by about a third, the bootstrap recovers most of the gap on this slide, and the lesson of Chapter 1 appears again: a correct point estimate with an uncertainty from a formula that assumes independence. The median $\hat\zeta$ over fresh slides is $-1.76$, a little closer to zero than the truth: the pseudolikelihood slightly understates the strength of inhibition when there are few close pairs, which is what to expect when $\zeta$ is estimated from a dozen pairs. When a slide has *no* pair closer than $r_0$ at all, the estimate of $\zeta$ runs off to $-\infty$, the hard-core limit.

**Choosing $r_0$.** The distance $r_0$ is not smooth in the likelihood: moving it changes the neighbor counts in jumps. The standard approach is to profile: for each $r_0$ on a grid, fit the model and record the maximized log pseudolikelihood, then choose the best.

```python
print("\n   r0    zeta-hat   log pseudolikelihood")
prof = []
for r0 in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
    Xp, ip, ap = strauss_X(tumor, r0)
    bp, _, llp = fit_bt(Xp, ip, ap)
    prof.append((r0, bp[1], llp))
    print(f"{r0:5.2f}  {bp[1]:9.3f}  {llp:12.2f}")
```

| $r_0$ | $\hat\zeta$ | log pseudolikelihood |
| :--- | :--- | :--- |
| 0.02 | $-1.178$ | 321.54 |
| 0.03 | $-1.651$ | 328.55 |
| 0.04 | $-2.028$ | 342.03 |
| 0.05 | $-2.013$ | 354.92 |
| 0.06 | $-1.524$ | 356.65 |
| 0.07 | $-0.868$ | 340.16 |
| 0.08 | $-0.453$ | 327.44 |
| 0.10 | $-0.486$ | 334.12 |

The log pseudolikelihood climbs from $321.5$ at $r_0 = 0.02$ to $356.7$ at $r_0 = 0.06$ and falls to $340.2$ at $0.07$ and $327.4$ at $0.08$. The maximizer, $0.06$, is the true interaction distance. The value at $0.05$ is close behind ($354.9$), so the data cannot separate $0.05$ from $0.06$. The profile rises again at $r_0 = 0.10$ ($334.1$, above the values at $0.08$): when the distance is large and the interaction weak, the Strauss model absorbs the overall regularity of the pattern, and a profile should be examined over its whole range instead of hill-climbed from one starting value. Because the log pseudolikelihood is not a log-likelihood, we use the profile only to pick $r_0$, not to build a confidence interval.

---

## 12.5 Checking a Fitted Point Process Model

A fitted model is a claim about how the data were generated. This section gives three ways of testing the claim, in increasing order of what they can detect: residual fields (is the intensity right?), envelopes (are the pair statistics right?), and a table of goodness-of-fit tests (which model families survive?).

### 12.5.1 Residuals: Smoothed Residual Fields

For a regression, the residual is the data minus the fitted value, and a map of residuals shows where the model misses. For a point pattern, the analogue compares the observed points with the fitted intensity, but points are not numbers on a grid. We make them comparable by smoothing both with the same kernel. With a Gaussian kernel $\mathcal{K}_\psi$ of bandwidth $\psi$ (as in Chapter 10), the **smoothed residual field** is

$$
R(\mathbf{s}) = \sum_{i=1}^n \mathcal{K}_\psi(\mathbf{s} - \mathbf{s}_i) - \int_{\mathcal{D}}\mathcal{K}_\psi(\mathbf{s} - \mathbf{u})\,\hat\lambda(\mathbf{u})\,d\mathbf{u}
$$

The first term is the kernel estimate of the intensity of Chapter 10 *without* edge correction; the second is the same kernel applied to the fitted intensity. Using the same kernel on both means that smoothing bias cancels, edge effects included, and if the fitted intensity is correct the expectation of $R(\mathbf{s})$ is zero at every $\mathbf{s}$: the expected value of a kernel sum over the points is the kernel integrated against the true intensity, whatever the interaction between points. The residual field therefore estimates the difference $\lambda(\mathbf{s}) - \hat\lambda(\mathbf{s})$, smoothed.

For a Poisson process with intensity $\lambda$ the kernel sum has variance $\lambda\int \mathcal{K}_\psi^2 = \lambda/(4\pi\psi^2)$, so under a correct Poisson model $R$ has a standard deviation of about $\sqrt{n/(4\pi\psi^2)}$ at each location. We report residuals in multiples of this "noise standard deviation".

We compute fields for three fits with bandwidth $\psi = 0.08$. The first is a pattern with a pure trend: an inhomogeneous Poisson process with intensity $750\,\text{pop\_density}(\mathbf{s})^2$ (a dense region around $(0.7, 0.7)$), fitted by a *constant* intensity. The second is the same pattern fitted with the correct log-linear model of Chapter 11. The third is the Thomas slide of Section 12.2.5, fitted by a constant intensity, which is the right first-order model for it.

```python
def sim_inhom_poisson(intensity, rng, n_grid=50):
    """Poisson process with intensity function, by thinning (unit square)."""
    G = pixel_centers(n_grid)
    lam_max = 1.02 * intensity(G).max()
    n = rng.poisson(lam_max)
    P = rng.uniform(size=(n, 2))
    return P[rng.uniform(size=n) < intensity(P) / lam_max]


trend = sim_inhom_poisson(lambda P: 750 * pop_density(P)**2, rng)
print(f"\ntrend pattern: {len(trend)} points")
```

```python
def smoothed_residual(P, intensity, h=0.08, n_grid=40):
    """Kernel-smoothed (data - fitted) on a grid; same kernel on both."""
    G = pixel_centers(n_grid)
    def kern(A, B):
        d2 = ((A[:, None, 0] - B[None, :, 0])**2
              + (A[:, None, 1] - B[None, :, 1])**2)
        return np.exp(-d2 / (2 * h**2)) / (2 * np.pi * h**2)
    data_part = kern(G, P).sum(axis=1)
    model_part = kern(G, G) @ intensity(G) / n_grid**2
    return (data_part - model_part).reshape(n_grid, n_grid)


beta_t, cov_t, _ = fit_trend(trend)
fits = {
    "trend pattern, constant intensity":
        (trend, lambda G: np.full(len(G), len(trend))),
    "trend pattern, log-linear in log density":
        (trend, lambda G: np.exp(beta_t[0]
                                 + beta_t[1] * np.log(pop_density(G)))),
    "Thomas slide, constant intensity":
        (slide, lambda G: np.full(len(G), n_slide)),
}
res_fields = {}
for name, (P, f) in fits.items():
    R = smoothed_residual(P, f)
    res_fields[name] = R
    noise = np.sqrt(len(P) / (4 * np.pi * 0.08**2))
    print(f"{name:42s} rms = {np.sqrt(np.mean(R**2)) / noise:4.2f}, "
          f"max |R| = {np.abs(R).max() / noise:4.2f} (in noise sd units)")
print(f"log-linear fit: beta0 = {beta_t[0]:.3f} (truth {np.log(750):.3f}), "
      f"beta1 = {beta_t[1]:.3f} (truth 2), naive se = "
      f"{np.sqrt(cov_t[1, 1]):.3f}")
```

The Poisson fit with the trend recovers the truth: $\hat\beta_0 = 6.65$ (truth $6.62$) and $\hat\beta_1 = 1.89$ (truth $2$), with naive standard error $0.176$ (the standard error is valid here, since the pattern is Poisson).

- **Trend, constant intensity.** The residual field has root-mean-square $3.0$ noise standard deviations and a maximum of $10.2$. Figure 12.6 shows a single large positive blob where the dense region is: the model has put too few points there and too many elsewhere.
- **Trend, correct model.** The root-mean-square falls to $0.79$ and the maximum to $1.9$: the field looks like noise. (A root-mean-square below $1$ is typical, because smoothing correlates neighboring values and the maximum is the extreme of a smooth field.)
- **Thomas slide, constant intensity.** The first-order model is right, but the field has a root-mean-square of $2.0$ noise standard deviations and a maximum of $4.5$. There is no single blob; Figure 12.6 shows a scatter of positive patches at the cluster locations. The excess variability is the signature of clustering: a cluster process has more variance in its smoothed intensity than a Poisson process of the same intensity (Section 12.2.1 gave the factor $1 + \mu_c$ for counts).

So a residual field detects a wrong *trend* reliably. It cannot say whether excess variability is clustering, and it cannot detect inhibition at all, since inhibition leaves the first-order intensity untouched and reduces variance only slightly. For those we need second-order summaries.

### 12.5.2 Goodness of Fit With Envelopes From the Fitted Model

Chapter 10 compared $\hat{L}(r) - r$ to an envelope from *complete spatial randomness*, which asks whether the pattern is Poisson. The same recipe works for any fitted model: simulate $B$ patterns from the fitted model, compute $\hat{L}(r) - r$ for each, and plot the pointwise minimum and maximum. If the model is adequate, the observed curve should stay inside the envelope; if the model is inadequate in a way that $K$ can see, the curve leaves it.

The next block computes four envelopes with $B = 99$, using $\hat{L}(r) - r$ with $L(r) = \sqrt{K(r)/\pi}$ as in Chapter 10: the Thomas slide against a Poisson process and against the fitted Thomas process, and the tumor slide of Section 12.4.5 against a Poisson process and against the fitted Strauss process. For each it reports the fraction of the $59$ radii at which the observed curve is outside the envelope.

```python
def l_minus_r(P, r):
    return np.sqrt(khat(P, r) / np.pi) - r


def envelope(sim, r, B):
    """Pointwise min and max of L(r) - r over B simulated patterns."""
    runs = np.array([l_minus_r(sim(), r) for _ in range(B)])
    return runs.min(axis=0), runs.max(axis=0)


n_t = len(tumor)
env = {
    "Thomas slide vs. Poisson": envelope(
        lambda: rng.uniform(size=(rng.poisson(n_slide), 2)), r_grid, 99),
    "Thomas slide vs. fitted Thomas": envelope(
        lambda: sim_thomas(k_hat, m_hat, s_hat, rng), r_grid, 99),
    "Tumor slide vs. Poisson": envelope(
        lambda: rng.uniform(size=(rng.poisson(n_t), 2)), r_grid, 99),
    "Tumor slide vs. fitted Strauss": envelope(
        lambda: sim_strauss(np.exp(b_hat[0]), np.exp(b_hat[1]), r0_s, rng,
                            steps=8000), r_grid, 99),
}
obs_L = {"Thomas slide": l_minus_r(slide, r_grid),
         "Tumor slide": l_minus_r(tumor, r_grid)}
for name, (lo, hi) in env.items():
    key = name.split(" vs.")[0]
    L_o = obs_L[key]
    outside = np.mean((L_o < lo) | (L_o > hi))
    print(f"{name:32s} observed outside the envelope at "
          f"{outside:4.0%} of radii")
```

Against the Poisson envelope, the Thomas slide is outside at $98\%$ of radii and the tumor slide at $58\%$. Against the fitted-model envelopes, they are outside at $0\%$ and $2\%$ (one radius of $59$) of the radii. Figure 12.5 shows the four panels. Two features stand out. The fitted-Thomas envelope is *much wider* than the Poisson envelope, about $0.06$ wide at $r = 0.1$ against $0.02$: a cluster process has more variable pair counts, because the number and size of clusters vary from slide to slide. And the tumor slide's curve dips below the Poisson band for $r$ between about $0.02$ and $0.07$, where pairs are missing, precisely the range of $r_0 = 0.06$.

A caution about reading envelopes. Each envelope is pointwise: at a single radius, a $99$-simulation envelope has a $2\%$ chance of excluding a curve from the true model, and over $59$ correlated radii the chance of crossing *somewhere* is considerably higher. Envelopes are for looking; they are not tests. A formal test uses a single statistic for the whole curve, which is the subject of the next subsection. And because the parameters were estimated from the same pattern, the fitted-model envelope is optimistic: the data are, by construction, close to the fitted model.

### 12.5.3 Choosing Among Trend, Cluster, Cox, and Gibbs Models

With four model families available, a table of goodness-of-fit tests can show which are consistent with a given pattern. For each pattern and each fitted model we compute a Monte Carlo p-value from a single statistic over the whole range of radii. The statistic compares $\hat{K}^{1/4}$ with the average of $\hat{K}^{1/4}$ across the simulations:

$$
T = \sum_k\left[\hat{K}(r_k)^{1/4} - \bar{K}(r_k)^{1/4}\right]^2\Delta r
$$

where $\bar{K}^{1/4}$ is the mean of the simulated values. Simulate $B$ patterns from the fitted model, compute $T$ for each against the same mean, and let $p = (1 + \#\{T_b \ge T_{\text{obs}}\})/(B + 1)$. With $B = 39$ the smallest possible p-value is $1/40 = 0.025$. (This statistic is in the family of tests proposed by Diggle, Cressie, and by Loosmore and Ford; we name them only for orientation.)

The five columns are:

- **Poisson with covariate, ordinary $K$:** the log-linear model of Chapter 11, simulated by thinning, compared on the ordinary $K$-function.
- **Poisson with covariate, $K_{\text{inhom}}$:** the same model, compared on the inhomogeneous $K$-function, re-fitting the intensity inside every simulation so that the estimation is accounted for.
- **Thomas:** fitted by minimum contrast (Section 12.2.5), with $\kappa \in [0.5, 5000]$ and $\sigma_c \in [0.002, 0.2]$ so that the fit stays in a sensible range.
- **LGCP:** fitted by minimum contrast (Section 12.3.2).
- **Strauss:** pseudolikelihood fit with $r_0$ chosen by profiling (Section 12.4.5), with $\gamma$ capped at $1$.

The three patterns are the Thomas slide, the tumor slide of Section 12.4.5, and the trend pattern of Section 12.5.1. We know the truth for each: clustering, inhibition, and a trend with no interaction.

```python
def mc_pvalue(stat_obs, stat_fn, sim, B):
    """Monte Carlo p-value of sum_r (K^(1/4) - mean of simulated)^2."""
    runs = np.array([stat_fn(sim()) for _ in range(B)])
    ref = runs.mean(axis=0)
    t_sim = np.sum((runs - ref)**2, axis=1)
    t_obs = np.sum((stat_obs - ref)**2)
    return (1 + np.sum(t_sim >= t_obs)) / (B + 1)


def k4(P):
    return khat(P, r_grid)**0.25


def best_strauss(P):
    """Strauss pseudolikelihood fit with r0 chosen by profiling."""
    best = None
    for r0 in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
        Xp, ip, ap = strauss_X(P, r0)
        bp, _, llp = fit_bt(Xp, ip, ap)
        if best is None or llp > best[0]:
            best = (llp, r0, bp)
    return best


def p_values(P, B=39):
    n = len(P)
    out = {}
    # Poisson with covariate, ordinary K and inhomogeneous K (re-fitted)
    b, _, _ = fit_trend(P)
    lam_fn = lambda G: np.exp(b[0] + b[1] * np.log(pop_density(G)))
    sim = lambda: sim_inhom_poisson(lam_fn, rng)
    out["Poisson + covariate, K"] = mc_pvalue(k4(P), k4, sim, B)

    def kinh4(Q):
        _, _, lam = fit_trend(Q)
        return khat_inhom(Q, r_grid, lam)**0.25
    out["Poisson + covariate, K-inhom"] = mc_pvalue(kinh4(P), kinh4, sim, B)
    # Thomas
    kp, sg, mc, _ = fit_thomas(khat(P, r_grid), r_grid, n)
    out["Thomas"] = mc_pvalue(
        k4(P), k4, lambda: sim_thomas(kp, mc, sg, rng), B)
    # LGCP
    (s2, ph), _ = fit_lgcp(khat(P, r_grid), r_grid)
    L_f = exp_chol(m_c, s2, ph)
    out["LGCP"] = mc_pvalue(k4(P), k4, lambda: sim_lgcp(n, L_f, rng), B)
    # Strauss
    _, r0b, bs = best_strauss(P)
    g_hat = min(np.exp(bs[1]), 1.0)
    out["Strauss"] = mc_pvalue(
        k4(P), k4, lambda: sim_strauss(np.exp(bs[0]), g_hat, r0b, rng,
                                       steps=8000), B)
    return out
```

```python
patterns = {"Thomas slide": slide, "Tumor slide (Strauss)": tumor,
            "Trend pattern": trend}
table = {name: p_values(P) for name, P in patterns.items()}
cols = list(next(iter(table.values())).keys())
print("\n" + " " * 24 + "".join(f"{c:>30s}" for c in cols))
for name, row in table.items():
    print(f"{name:24s}" + "".join(f"{row[c]:30.3f}" for c in cols))

print("\nThomas fits: kappa, sigma_c, mu_c")
for name, P in patterns.items():
    kp, sg, mc, _ = fit_thomas(khat(P, r_grid), r_grid, len(P))
    print(f"{name:24s} {kp:8.1f} {sg:8.4f} {mc:8.2f}")
```

| Pattern (truth) | Poisson + covariate, $K$ | Poisson + covariate, $K_{\text{inhom}}$ | Thomas | LGCP | Strauss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Thomas slide (clustering) | 0.025 | 0.025 | 0.950 | 0.650 | 0.025 |
| Tumor slide (inhibition) | 0.025 | 0.025 | 0.025 | 0.025 | 0.525 |
| Trend pattern (trend only) | 1.000 | 0.450 | 0.800 | 0.125 | 0.025 |

Read the table by rows.

- **Clustering.** The Poisson models are rejected by either $K$ ($p = 0.025$, the smallest possible), so even after the covariate there is interaction. The Strauss model is also rejected. The Thomas model (the truth) and the LGCP both fit.
- **Inhibition.** Every model except Strauss is rejected. The Thomas and LGCP fits degenerate to Poisson patterns in disguise (the printed Thomas fit has $\hat\kappa = 5000$ and $\hat\mu_c = 0.02$: almost every parent has no offspring), because $K$ below $\pi r^2$ is something a cluster model can only mimic by switching clustering off.
- **Trend only.** The correct model, Poisson with the covariate, passes with both $K$ ($p = 1.000$) and $K_{\text{inhom}}$ ($p = 0.45$). The Strauss model is rejected. But the **Thomas model also fits** ($p = 0.80$), and the LGCP is borderline ($p = 0.125$). The printed Thomas fit shows how: $\hat\kappa = 2.4$ and $\hat\sigma_c = 0.2$ (the upper bound), with $\hat\mu_c = 83$. That is a few very large clusters, which is how a clustering model explains a single dense region. The $K$-function of that fit is close enough to the trend pattern's $K$ that the test cannot tell them apart.

The last row is the same trap that Chapter 11 and Section 12.1 warned about, now in the form of a model choice. A clustering model can fit a pattern whose only feature is a trend, because $K$ does not know about the covariate. What separated the models was not a Thomas fit but the *covariate-aware* column: for the trend pattern, the Poisson model with a covariate passes the $K_{\text{inhom}}$ test, so no interaction is needed; for the Thomas slide it fails the same test, so interaction remains after the trend is removed. When a trend is plausible, the right order is: fit the first-order model, compute $K_{\text{inhom}}$ with it, and only then ask whether an interaction model is needed.

Two further caveats. First, for the Thomas, LGCP, and Strauss columns the parameters were estimated from the data being tested, and the Monte Carlo reference distribution does not account for that. Such tests are conservative: p-values tend to be larger than they should be, and a model is rejected less often than the nominal rate. Exercise 19 asks you to measure this. (The $K_{\text{inhom}}$ column re-fits the intensity within each simulation and avoids this problem for the first-order part.) Second, $K$ summarizes only part of a pattern. Passing a $K$-based test does not show a model is correct, as the Thomas row for the trend pattern demonstrates.

---

## 12.6 Walkthrough and Lab 12

Most of the code of this chapter appeared next to the derivations that motivate it. The lab collects its output in seven figures, each produced from objects already computed above, so that the three explanations, the three kinds of fit, and the checking tools can be seen side by side. Run the code from the top folder of the repository so that the figures are written to `figures/ch12/`.

**Three explanations (Figure 12.1).** The first block also sets up the plotting helper used by all the figures.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch12", exist_ok=True)


def draw_points(ax, P, title, color="tab:blue"):
    ax.scatter(P[:, 0], P[:, 1], s=9, color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=10)


# Figure 12.1: three explanations, same ordinary L(r) - r
fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
three = (("Trend (Poisson)", trend), ("Clustering (Thomas)", slide),
         ("Inhibition (Strauss)", tumor))
for j, (name, P) in enumerate(three):
    draw_points(axes[0, j], P, f"{name}, n = {len(P)}")
    axes[1, j].plot(r_grid, l_minus_r(P, r_grid), color="tab:red")
    axes[1, j].axhline(0, color="gray", lw=1)
    axes[1, j].set_ylim(-0.04, 0.07)
    axes[1, j].set_xlabel("r")
    axes[1, j].set_ylabel("L(r) - r")
fig.tight_layout()
fig.savefig("figures/ch12/fig_12_1_three_explanations.png", dpi=120)
plt.close(fig)
```

![Figure 12.1. Three patterns with the same ordinary L(r) - r plotted beneath. Top: an inhomogeneous Poisson pattern (n = 199), a Thomas pattern (n = 137), and a Strauss pattern (n = 91). Bottom: L(r) - r for each, with the zero line for complete spatial randomness.](../figures/ch12/fig_12_1_three_explanations.png)

In Figure 12.1 the first two curves rise above zero and the third falls below it. The trend pattern's curve climbs steadily, almost linearly, to about $0.05$ at $r = 0.15$ with no plateau: a pattern whose points crowd into one region has more pairs at *every* distance up to the size of the region. The Thomas curve rises faster, reaches about $0.045$ by $r = 0.06$, and then declines slowly: the excess pairs are within-cluster pairs, and their share falls once $r$ exceeds the cluster size. The Strauss curve dips to $-0.033$ near $r = 0.055$, where pairs are missing, and then recovers part of the way ($-0.011$ at $r = 0.08$, about $-0.01$ to $-0.02$ at larger $r$). The shapes distinguish the stories, but only when there is enough data to see them, and the trend pattern is the least distinctive of the three (Section 12.5.3 showed that $K$ alone cannot separate it from clustering).

**Minimum contrast (Figure 12.2).**

```python
# Figure 12.2: minimum contrast fit of the Thomas model
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
draw_points(axes[0], slide, f"Simulated slide, n = {n_slide}")
axes[1].plot(r_grid, K_obs**0.25, color="black", label="K-hat")
axes[1].plot(r_grid, K_thomas(r_grid, k_hat, s_hat)**0.25,
             color="tab:red", label="fitted Thomas")
axes[1].plot(r_grid, K_thomas(r_grid, kappa, sigma_c)**0.25,
             color="tab:green", ls="--", label="true Thomas")
axes[1].plot(r_grid, (np.pi * r_grid**2)**0.25, color="gray", ls=":",
             label="complete spatial randomness")
axes[1].set_xlabel("r")
axes[1].set_ylabel("K(r)^(1/4)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/ch12/fig_12_2_min_contrast.png", dpi=120)
plt.close(fig)
```

![Figure 12.2. The simulated immune-cell slide (n = 137) and its minimum contrast fit. Right: K-hat to the power 1/4 (black), the fitted Thomas curve (red), the true Thomas curve (green dashed), and complete spatial randomness (gray dotted).](../figures/ch12/fig_12_2_min_contrast.png)

The slide on the left shows clear groups of cells. On the right the fitted Thomas curve lies almost exactly on $\hat{K}^{1/4}$, and the true Thomas curve lies a little above both at larger distances (about $0.577$ against $0.558$ at $r = 0.15$): this slide has fewer pairs than the process averages. The parameters behind the fitted and true curves differ substantially ($\hat\kappa = 33.5$ against $25$, $\hat\sigma_c = 0.0248$ against $0.03$), and yet the two curves are close. That is the practical meaning of imprecise cluster parameters: many parameter pairs give nearly the same $K^{1/4}$. The CSR curve, far below, shows how strong the clustering is.

**Inhibition mechanisms (Figure 12.3).**

```python
# Figure 12.3: three inhibition mechanisms
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (name, P) in zip(axes, (
        ("Matern I, lambda0 = 100", sim_matern_hc(100, r0_hc, rng, 1)),
        ("Matern II, lambda0 = 300", sim_matern_hc(300, r0_hc, rng, 2)),
        ("Sequential, 3000 attempts", sim_ssi(r0_hc, 3000, rng)))):
    draw_points(ax, P, f"{name}, n = {len(P)}")
    for p in P:
        ax.add_patch(plt.Circle(p, r0_hc / 2, fill=False, lw=0.4,
                                color="tab:blue"))
fig.tight_layout()
fig.savefig("figures/ch12/fig_12_3_inhibition.png", dpi=120)
plt.close(fig)
```

![Figure 12.3. Three inhibition mechanisms with the hard-core distance r0 = 0.05; each point carries a disc of radius r0/2, so discs never overlap. Left: Matérn Type I from a Poisson parent pattern with intensity 100 (n = 43). Middle: Matérn Type II with intensity 300 (n = 110). Right: sequential inhibition after 3000 proposals (n = 245).](../figures/ch12/fig_12_3_inhibition.png)

In Figure 12.3 no two discs overlap, in any panel. Type I (left) leaves large empty patches: whenever two parents are close, both are removed, so the survivors are sparse and the gaps are where pairs were. Type II (middle) is denser, since one of each close pair stays. Sequential inhibition (right) is the densest, with $245$ points and discs that almost tile the square; it keeps trying until the window is nearly full.

**The Strauss slide and the profile (Figure 12.4).**

```python
# Figure 12.4: the Strauss slide and the profile pseudolikelihood
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
draw_points(axes[0], tumor, f"Tumor cells, n = {len(tumor)}")
dd = np.hypot(tumor[:, None, 0] - tumor[None, :, 0],
              tumor[:, None, 1] - tumor[None, :, 1])
for i, j in zip(*np.where(np.triu(dd <= r0_s, 1))):
    axes[0].plot(*tumor[[i, j]].T, color="tab:red", lw=1.5)
pr = np.array(prof)
axes[1].plot(pr[:, 0], pr[:, 2], "o-", color="tab:blue")
axes[1].axvline(r0_s, color="tab:green", ls="--", label="true r0")
axes[1].set_xlabel("r0")
axes[1].set_ylabel("log pseudolikelihood")
axes[1].legend()
fig.tight_layout()
fig.savefig("figures/ch12/fig_12_4_strauss_profile.png", dpi=120)
plt.close(fig)
```

![Figure 12.4. Left: the simulated tumor-cell slide (n = 91) with the 12 pairs within r0 = 0.06 joined by red segments. Right: log pseudolikelihood against r0, with the true r0 marked.](../figures/ch12/fig_12_4_strauss_profile.png)

On the left of Figure 12.4, the $12$ close pairs (red) are short segments, with a few chains of three cells; the rest of the cells are well separated. On the right, the log pseudolikelihood peaks at $r_0 = 0.06$ (the true value, marked), with $r_0 = 0.05$ nearly as high, then drops, and shows a smaller second rise at $0.10$.

**Envelopes (Figure 12.5).**

```python
# Figure 12.5: envelopes from the Poisson and from the fitted models
fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
for ax, (name, (lo, hi)) in zip(axes.ravel(), env.items()):
    ax.fill_between(r_grid, lo, hi, color="lightgray")
    ax.plot(r_grid, obs_L[name.split(" vs.")[0]], color="tab:red")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("L(r) - r")
for ax in axes[1]:
    ax.set_xlabel("r")
fig.tight_layout()
fig.savefig("figures/ch12/fig_12_5_envelopes.png", dpi=120)
plt.close(fig)
```

![Figure 12.5. Observed L(r) - r (red) and 99-simulation envelopes (gray). Top: the Thomas slide against a Poisson process and against the fitted Thomas process. Bottom: the tumor slide against a Poisson process and against the fitted Strauss process.](../figures/ch12/fig_12_5_envelopes.png)

In Figure 12.5 the top left panel shows the Thomas slide's curve far above a narrow Poisson band at every radius. In the top right, the same curve sits inside the much wider fitted-Thomas band. In the bottom left the tumor slide's curve dips below the Poisson band for $r$ from about $0.02$ to $0.07$ and then runs along its lower edge. In the bottom right the curve is inside the fitted-Strauss band except for a brief touch of its lower edge near $r = 0.055$.

**Residual fields (Figure 12.6).**

```python
# Figure 12.6: smoothed residual fields
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
vmax = max(np.abs(R).max() for R in res_fields.values())
for ax, (name, R) in zip(axes, res_fields.items()):
    im = ax.imshow(R, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(name, fontsize=8)
fig.colorbar(im, ax=axes, label="smoothed residual intensity")
fig.savefig("figures/ch12/fig_12_6_residual_fields.png", dpi=120,
            bbox_inches="tight")
plt.close(fig)
```

![Figure 12.6. Smoothed residual fields (bandwidth 0.08) on a common color scale. Left: the trend pattern fitted by a constant intensity. Middle: the same pattern fitted by the correct log-linear model. Right: the Thomas slide fitted by a constant intensity.](../figures/ch12/fig_12_6_residual_fields.png)

In Figure 12.6 the left panel has a large red region near $(0.7, 0.75)$ on a pale blue background: a trend left unmodeled. The middle panel is almost uniformly pale, with small patches of both signs. The right panel shows scattered positive and negative patches of moderate size and no large single region, the signature of excess variability without a trend.

**LGCP and the pixel GMRF (Figure 12.7).**

```python
# Figure 12.7: LGCP, truth and the pixel GMRF fit
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
log_true_fine = np.log(Lam_true).reshape(m_c, m_c)
for ax, (name, img) in zip(axes[[0, 2]], (
        ("True log intensity", log_true_fine),
        ("Fitted log intensity (GMRF)",
         (best[2] + best[3]).reshape(g_fit, g_fit)))):
    im = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap="viridis",
                   vmin=log_true_fine.min(), vmax=log_true_fine.max())
    ax.set_title(name, fontsize=10)
draw_points(axes[1], P_lg, f"Simulated points, n = {len(P_lg)}")
fig.colorbar(im, ax=axes, label="log intensity")
fig.savefig("figures/ch12/fig_12_7_lgcp.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("figures saved")
```

![Figure 12.7. The LGCP simulation. Left: the true log intensity on the 40 x 40 grid. Middle: the 220 simulated points. Right: the GMRF fit on the 20 x 20 pixel grid.](../figures/ch12/fig_12_7_lgcp.png)

In Figure 12.7 the middle panel shows points concentrated in a few places: the upper left, a band near the right edge, and a patch in the lower left. The true log intensity (left) has high values in those places and a low region around $(0.1, 0.4)$. The fitted surface (right) reproduces the large-scale pattern, with highs in the upper left, along the right edge, and in the lower left, but as a much smoother, coarser picture: each $0.05 \times 0.05$ pixel holds on average $0.55$ points, so the data support only a smoothed reconstruction.

**Which tool for which question.** The chapter's models answer different questions:

| Question | Model | Estimated by |
| :--- | :--- | :--- |
| Does intensity depend on covariates? | Inhomogeneous Poisson | Weighted Poisson GLM (Chapter 11) |
| Do points gather around unobserved parents? | Thomas, Matérn cluster | Minimum contrast on $K$ or $K_{\text{inhom}}$ |
| Is there a smooth unobserved driver of intensity? | LGCP | Minimum contrast; or a pixel GLMM with a GMRF prior |
| Do points keep their distance? | Hard-core, Strauss | Pseudolikelihood (weighted Poisson GLM with a neighbor-count column) |
| Is the fitted model adequate? | Any | Residual fields, envelopes, Monte Carlo tests from the fitted model |

---

## Exercises

**Analytical**

1. Show that for any Neyman-Scott process with Poisson offspring counts and any dispersal density, the intensity is $\kappa\mu_c$, and that the number of offspring in a window of area $A$ (ignoring edge effects) has variance $\kappa A\mu_c(1 + \mu_c)$. The ratio of variance to mean is called the Fano factor. What does a Fano factor of $1 + \mu_c$ imply for the index of dispersion of quadrat counts in Chapter 10?

2. Using the difference of two independent $N(\mathbf{0}, \sigma_c^2\mathbf{I}_2)$ displacements, derive the pair correlation function of the Thomas process and confirm that integrating $2\pi s\,g(s)$ from $0$ to $r$ gives the $K$-function of Section 12.2.4. Evaluate $g(0)$ for $\kappa = 25$ and $\sigma_c = 0.03$ and explain what the number means in terms of pairs.

3. Suppose the number of offspring per parent is not Poisson but has mean $\mu_c$ and variance $v$. Redo the "same parent" step of Section 12.2.4 with $\operatorname{E}[N(N-1)] = v + \mu_c^2 - \mu_c$ and show that

   $$
   g(\mathbf{h}) = 1 + \frac{\operatorname{E}[N(N-1)]}{\kappa\mu_c^2}f_2(\mathbf{h})
   $$

   What is $\operatorname{E}[N(N-1)]$ when every parent has exactly $m$ offspring? Why does the Poisson assumption make $\mu_c$ unidentifiable from $K(r)$ alone?

4. Let $\Phi$ be a stationary point process with pair correlation function $g$, and thin it independently with retention probability $p(\mathbf{s})$. Show that the thinned process has intensity $p(\mathbf{s})\lambda$ and the same pair correlation function $g$. Why does this make the inhomogeneous $K$-function of Section 12.2.6 the right summary for a thinned Thomas process?

5. For the Matérn cluster process, the sibling-distance distribution function is $P_{\text{sib}}(r) = (16/\pi)\int_0^{z}\left(u\arccos u - u^2\sqrt{1-u^2}\right)du$ with $z = r/(2R)$. Show that $P_{\text{sib}} \approx 4z^2 = r^2/R^2$ for small $z$ and explain this with the area of the overlap of two discs. Verify that $P_{\text{sib}} = 1$ at $z = 1$.

6. For a log-Gaussian Cox process with constant trend, the count in a region $B$ has variance $\operatorname{Var}(N(B)) = \lambda\operatorname{area}(B) + \lambda^2\int_B\int_B\left(e^{C_u(\lVert\mathbf{s} - \mathbf{s}'\rVert)} - 1\right)d\mathbf{s}\,d\mathbf{s}'$. Derive it from the law of total variance and the pair correlation function $g = e^{C_u}$. Why is the variance larger than the Poisson value $\lambda\operatorname{area}(B)$?

7. For a Matérn Type I process with parent intensity $\lambda_{\text{par}}$, show that the intensity is maximized at $\lambda_{\text{par}} = 1/(\pi r_0^2)$, where it equals $1/(e\pi r_0^2)$, and that at that value only a fraction $1/e$ of parents survive. For Type II, show that $\lambda_{II}$ increases with $\lambda_{\text{par}}$ and tends to $1/(\pi r_0^2)$.

8. For $n = 2$ points on the unit square, write the normalizing constant of the Strauss process as $Z_2 = 1 + b\cdot 1 + \dfrac{b^2}{2}\left[1 - (1 - \gamma)\,p(r_0)\right]$, where $p(r_0)$ is the probability that two uniform points are within $r_0$. (The terms are $n = 0$, $n = 1$, and $n = 2$ of the series for $Z$, truncated.) Compute $p(r_0)$ approximately for small $r_0$, ignoring edge effects, and explain why the same calculation for $n = 100$ is not feasible.

9. Show that the score equation of the pseudolikelihood in $\zeta$ is $\sum_{i=1}^n t(\mathbf{s}_i, \mathbf{S}\setminus\mathbf{s}_i) = \int_{\mathcal{D}}t(\mathbf{u}, \mathbf{S})\,\lambda(\mathbf{u} \mid \mathbf{S})\,d\mathbf{u}$. Interpret it: the observed total neighbor count is matched by the count the fitted conditional intensity expects. Check that it reduces to a familiar equation in the Poisson case $\gamma = 1$.

10. Verify the acceptance probability for a death move in Section 12.4.3 from detailed balance, and check that for $n = 0$ and $n = 1$ the chain moves between the empty pattern and a single point with the right stationary ratio $b\operatorname{area}(\mathcal{D})$.

**Computational and simulation**

11. Keep $\lambda = \kappa\mu_c = 200$ and $\sigma_c = 0.03$ fixed and vary $\kappa \in \{10, 25, 50, 100\}$ (so $\mu_c$ changes). For each, simulate $200$ slides, fit by minimum contrast, and plot the root-mean-square log error of $\hat\kappa$ and $\hat\sigma_c$ against $\kappa$. Explain the pattern in terms of the number of clusters on a slide.

12. Refit the slide of Section 12.2.5 using fitting ranges $[0.005, 0.08]$, $[0.005, 0.15]$, and $[0.005, 0.25]$. Report the three estimates of $(\kappa, \sigma_c)$ and discuss why the upper limit matters.

13. Fit a Matérn cluster process by minimum contrast to a Thomas pattern, and a Thomas process to a Matérn cluster pattern, using the $K$-functions of Sections 12.2.4 and 12.2.7. Compare the minimum contrast values over $200$ patterns of each kind. Can the $K$-function tell Gaussian dispersal from disc dispersal?

14. Use the function `road_distance` of the shared setting (Section 11.1.2) as a covariate: simulate an inhomogeneous Poisson process with intensity $500\exp(-6\,\text{road\_distance}(\mathbf{s}))$, fit it with the Berman-Turner code of Section 12.2.6 (replacing the covariate), and compare the ordinary and inhomogeneous $K$-functions of the pattern. Then thin a Thomas process with the same retention probability and repeat the cluster-parameter estimation.

15. Write an inhomogeneous Strauss simulator by changing the birth acceptance probability to use $b(\mathbf{u}) = \exp(\beta_0 + \beta_1\log\text{pop\_density}(\mathbf{u}))$. Fit it with the design matrix $[1, \log\text{pop\_density}, t]$ and recover $(\beta_0, \beta_1, \zeta)$ from $100$ simulated slides. How do the three estimates correlate across slides?

16. Fit the Strauss model, with $r_0$ chosen by profiling, to $100$ Matérn Type II patterns with $\lambda_{\text{par}} = 300$ and $r_0 = 0.05$. Report the median $\hat\zeta$, the profile maximizer $\hat{r}_0$, and the fraction of patterns for which $\hat\zeta < -5$ (a near hard core). Does the pseudolikelihood fit know it is looking at a hard core?

17. Repeat the pixel-grid GMRF fit of Section 12.3.3 with $10 \times 10$, $20 \times 20$, and $40 \times 40$ pixels. For each size, report the Laplace-optimal $\sigma^2$ and the correlation with the true log-intensity. What limits the accuracy at the finest grid?

18. Build a parametric bootstrap for the minimum contrast estimates of the slide in Section 12.2.5: simulate $200$ Thomas patterns from the fitted parameters, refit each, and report standard deviations of $\hat\kappa$, $\hat\sigma_c$, and $\hat\mu_c$. Compare with the standard deviations across $200$ fresh slides from the true model.

19. Check the calibration of the goodness-of-fit test in Section 12.5.3: simulate $100$ Thomas slides from the true model, fit the Thomas model to each, and run the Monte Carlo test (with $B = 39$) using the fitted parameters. What fraction of tests reject at the 5% level? Explain why it is below 5%, and describe how re-fitting the model within each simulation would fix it.

---

## What Comes Next

This chapter closes Part IV. A pattern of points can now be explored (Chapter 10), related to covariates through a Poisson or logistic GLM (Chapter 11), and modeled for clustering or inhibition, with the fit checked against simulation (this chapter). The same pattern of ideas runs through all of Parts II to IV: write down a model with a spatial random effect or an interaction, find the function that summarizes it (the variogram, the covariance, $K$), fit by matching or by a likelihood-type criterion, and check the fit against simulated data from the fitted model.

Part V turns to a different limit: what happens to these methods when $n$ is not hundreds but hundreds of thousands. Chapter 13 shows how the dense covariance matrix $\boldsymbol{\Sigma}$ of Part II, and its sparse cousin the precision matrix $\mathbf{Q}$ of Part III, can be approximated so that the likelihoods of Chapter 5 and the predictions of Chapter 4 remain computable.
