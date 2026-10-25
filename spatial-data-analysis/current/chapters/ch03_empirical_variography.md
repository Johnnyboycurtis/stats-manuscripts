# Chapter 3: Empirical Variography and Weighted Least Squares Fitting

Chapter 2 described a spatial process through its semivariogram $\gamma(\mathbf{h})$: half the variance of the difference between two values separated by the lag $\mathbf{h}$. It treated the variogram as known. In practice it is not. We have one map of data, and we must estimate how dependence decays with distance from that map alone.

This chapter builds the standard tools for doing so. The idea is the one promised in Section 2.2: if the process follows the same rules everywhere, then every pair of locations at a given separation is a replicate, and averaging squared differences over many such pairs estimates $\gamma$. We turn that idea into an estimator, compute it efficiently, study how reliable it is, make it resistant to bad data, and fit the parametric families of Section 2.5 to it.

The chapter ends with a warning that runs through the rest of the book. A variogram is a statement about the random part of the data. If the mean is not constant and we ignore that, the variogram absorbs the trend and tells a misleading story. Removing the trend first helps, but the most common way of removing it introduces a smaller bias of its own, which Chapter 5 finally fixes.

> **Chapter at a Glance**
>
> - **The Empirical Semivariogram (§3.1):** Matheron's estimator, why it is unbiased, how to compute it for thousands of pairs at once, how to choose distance bins, and how to look for direction-dependent correlation.
> - **Robust Estimation (§3.2):** Why a few bad data points inflate the estimator, and the Cressie-Hawkins alternative, with its constants derived from the normal distribution.
> - **Fitting a Model (§3.3):** Why ordinary least squares is a poor way to fit a variogram curve, the approximate variance of each bin, and Cressie's weighted least squares.
> - **The Trend Pitfall (§3.4):** How an ignored trend makes the variogram grow without limit, and why removing the trend by OLS biases the variogram downward.
> - **Walkthrough and Lab (§3.5):** The rent district from raw listings to a fitted variogram, directional variograms, and outliers, with figures.

---

## 3.1 Estimating the Empirical Semivariogram

### 3.1.1 Matheron's Estimator

Recall from Section 2.2.3 that a process is intrinsically stationary when its increments have mean zero and a variance that depends only on the lag:

$$
\operatorname{E}\!\left[Y(\mathbf{s}_i) - Y(\mathbf{s}_j)\right] = 0, \qquad \operatorname{Var}\!\left(Y(\mathbf{s}_i) - Y(\mathbf{s}_j)\right) = 2\gamma(\mathbf{s}_i - \mathbf{s}_j)
$$

For a single pair of observations, the squared difference has expectation

$$
\operatorname{E}\!\left[(Y_i - Y_j)^2\right] = \operatorname{Var}(Y_i - Y_j) + \left(\operatorname{E}[Y_i - Y_j]\right)^2 = 2\gamma(\mathbf{s}_i - \mathbf{s}_j) + 0
$$

using the one-variable version of the result in Section 1.1.1: the mean of a square is the variance plus the squared mean. So half the squared difference of any pair is an unbiased estimate of the semivariogram at that pair's lag. One pair gives a very noisy estimate. Averaging over many pairs at the same lag keeps the estimate unbiased and reduces its variance.

Exactly repeated lags are rare in irregular data, so we group pairs by distance. Divide distances into **bins**, and let $N(h_k)$ be the set of pairs $(i, j)$, with $i < j$, whose distance falls in bin $k$. Write $\lvert N(h_k) \rvert$ for the number of such pairs and $h_k$ for their average distance, which serves as the bin's lag. **Matheron's estimator** is

$$
\hat{\gamma}(h_k) = \frac{1}{2\lvert N(h_k) \rvert}\sum_{(i,j) \in N(h_k)} (Y_i - Y_j)^2
$$

It is called a method-of-moments estimator because it replaces an expectation with an average over the data, as the sample variance does.

Two assumptions are hiding in this formula. First, it uses only distance, so it assumes isotropy: pairs separated east-west and north-south are pooled. Section 3.1.4 relaxes this. Second, within a bin, pairs have slightly different distances, so $\hat{\gamma}(h_k)$ estimates the average of $\gamma$ over those distances, not $\gamma(h_k)$ exactly. When $\gamma$ is close to linear across a bin, the two agree. Narrow bins make this bias small but leave few pairs per bin; wide bins do the opposite. Section 3.1.3 discusses the trade-off.

**A hand-sized example: five noise monitors.** Five noise monitors sit along a highway at 0, 1, 2, 3, and 4 km, recording average daytime noise levels of 62, 65, 63, 69, and 66 decibels. With monitors exactly 1 km apart, each lag is its own bin. At lag 1 there are four pairs, with squared differences $(65 - 62)^2 = 9$, $(63 - 65)^2 = 4$, $(69 - 63)^2 = 36$, and $(66 - 69)^2 = 9$, so

$$
\hat{\gamma}(1) = \frac{9 + 4 + 36 + 9}{2 \times 4} = \frac{58}{8} = 7.25
$$

The code repeats this for every lag.

```python
import numpy as np
from scipy import optimize, special

t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y5 = np.array([62.0, 65.0, 63.0, 69.0, 66.0])
for lag in (1, 2, 3, 4):
    i = np.arange(5 - lag)
    sq = (y5[i] - y5[i + lag])**2
    print(f"lag {lag}: pairs={len(i)}  squared diffs={sq}  "
          f"gamma_hat={sq.sum() / (2 * len(i)):.3f}")
```

| Lag (km) | Pairs | Squared differences | $\hat{\gamma}$ |
| :--- | :--- | :--- | :--- |
| 1 | 4 | 9, 4, 36, 9 | 7.250 |
| 2 | 3 | 1, 16, 9 | 4.333 |
| 3 | 2 | 49, 1 | 12.500 |
| 4 | 1 | 16 | 8.000 |

The estimates jump around: the semivariance at lag 2 is lower than at lag 1, and lag 3 is the highest of all. With so few pairs, a single large difference (the jump from 63 to 69 between the third and fourth monitors) dominates. Real data sets have thousands of pairs per bin, and the rest of this section shows how to compute and organize them.

### 3.1.2 Vectorized Computation

For $n$ locations there are $n(n-1)/2$ pairs. Looping over them in Python is slow, but the same broadcasting idea as the distance matrix of Section 1.6.1 computes every squared difference at once. The matrix of differences is

$$
\mathbf{y}\mathbf{1}' - \mathbf{1}\mathbf{y}'
$$

whose $(i, j)$ entry is $y_i - y_j$. Squaring it entrywise gives every squared difference, and the distance matrix $\mathbf{H}$ says which bin each pair belongs to. Because both matrices are symmetric, only the entries above the diagonal are needed: each pair once, and no pair of a location with itself.

The function `empirical_variogram` below does this. It also defines the shared data set used in Chapters 3, 4, and 5: **simulated rent listings** in a district, rescaled to the unit square. Rents are in hundreds of dollars, the trend rises to the east as $\mu(\mathbf{s}) = 18 + 6\,s_{\text{east}}$, the neighborhood effect $u$ has $\sigma^2 = 4$ with exponential correlation and $\phi = 0.15$, and the nugget has $\tau^2 = 1$. These are the same values as the rent district of Section 2.8.2, but the 250 listings are now at random locations, as real listings are.

```python
def dist(S):
    """Matrix of Euclidean distances between the rows of S."""
    q = np.sum(S**2, axis=1)
    H2 = q[:, None] - 2 * S @ S.T + q[None, :]
    H = np.sqrt(np.maximum(H2, 0.0))
    np.fill_diagonal(H, 0.0)
    return H


def rent_district(n=250, seed=2030):
    """Simulated rent listings: locations, trend, effect, noise, rent."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(size=(n, 2))
    mu = 18.0 + 6.0 * S[:, 0]
    R = np.exp(-dist(S) / 0.15)
    L = np.linalg.cholesky(4.0 * R + 1e-10 * np.eye(n))
    u = L @ rng.standard_normal(n)
    eps = rng.standard_normal(n)
    return S, mu, u, eps, mu + u + eps


def empirical_variogram(S, y, edges):
    """Matheron estimator on distance bins given by edges."""
    H = dist(S)
    D2 = (y[:, None] - y[None, :])**2
    iu = np.triu_indices(len(y), k=1)
    h, d2 = H[iu], D2[iu]
    centers, gam, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h > lo) & (h <= hi)
        centers.append(h[m].mean())
        gam.append(d2[m].sum() / (2 * m.sum()))
        counts.append(m.sum())
    return np.array(centers), np.array(gam), np.array(counts)


def gamma_exp(h, sigma2, tau2, phi):
    """Semivariogram of the exponential model with a nugget (as in Ch. 2)."""
    h = np.asarray(h, dtype=float)
    return np.where(h > 0, tau2 + sigma2 * (1 - np.exp(-h / phi)), 0.0)


S, mu, u, eps, y = rent_district()
n = len(y)
H = dist(S)
print("pairs:", n * (n - 1) // 2, " largest distance:", round(H.max(), 3))
edges = np.linspace(0.0, 0.6, 13)
hc, g_oracle, cnt = empirical_variogram(S, u + eps, edges)
for k in range(len(hc)):
    print(f"bin {edges[k]:.2f}-{edges[k+1]:.2f}: h={hc[k]:.3f} "
          f"pairs={cnt[k]:4d} gamma_hat={g_oracle[k]:.3f} "
          f"true={gamma_exp(hc[k], 4.0, 1.0, 0.15):.3f}")
far = np.linspace(0.6, 1.4, 9)
hf, gf, cf = empirical_variogram(S, u + eps, far)
for k in range(len(hf)):
    print(f"far bin {far[k]:.1f}-{far[k+1]:.1f}: pairs={cf[k]:4d} "
          f"gamma_hat={gf[k]:.3f}")
```

In this section we estimate the variogram of $u + \varepsilon$, the part of the rent left after subtracting the true trend. A real analyst does not know the trend; we do, because the data are simulated, and using it lets us study the estimator by itself. Section 3.4 removes this advantage. The 250 listings give 31,125 pairs, and the largest distance between two listings is 1.358.

### 3.1.3 Distance Bins, Lag Tolerances, and the Half-Diameter Rule

The code uses 12 bins of width 0.05, from 0 to 0.6. Half the width of a bin, here 0.025, is called the **lag tolerance**: pairs within that distance of the bin's center are pooled.

| Bin | $h_k$ | Pairs $\lvert N(h_k)\rvert$ | $\hat{\gamma}(h_k)$ | True $\gamma(h_k)$ |
| :--- | :--- | :--- | :--- | :--- |
| 0.00 to 0.05 | 0.034 | 223 | 1.844 | 1.803 |
| 0.05 to 0.10 | 0.078 | 692 | 2.553 | 2.617 |
| 0.10 to 0.15 | 0.127 | 1104 | 3.488 | 3.285 |
| 0.15 to 0.20 | 0.176 | 1473 | 3.931 | 3.765 |
| 0.20 to 0.25 | 0.226 | 1769 | 4.239 | 4.113 |
| 0.25 to 0.30 | 0.275 | 1920 | 4.258 | 4.359 |
| 0.30 to 0.35 | 0.325 | 2080 | 4.154 | 4.543 |
| 0.35 to 0.40 | 0.375 | 2238 | 4.579 | 4.672 |
| 0.40 to 0.45 | 0.425 | 2341 | 4.736 | 4.765 |
| 0.45 to 0.50 | 0.475 | 2186 | 4.847 | 4.832 |
| 0.50 to 0.55 | 0.525 | 2173 | 5.215 | 4.879 |
| 0.55 to 0.60 | 0.575 | 2180 | 5.382 | 4.913 |

The estimates follow the true curve: they start near the nugget-plus-a-little at short distances and rise toward the sill of 5. The first bin has the fewest pairs (223), because few listings happen to be very close together, and the bins in the middle have the most.

**Choosing bins.** Three practical considerations guide the choice.

- **Enough pairs.** A common rule of thumb asks for at least 30 pairs per bin. With 250 listings this is easy except at the shortest distances, which are also the most important for estimating the nugget and the shape near the origin.
- **Narrow enough to keep the shape.** Wide bins average $\gamma$ over a range of distances and blur the curve near the origin, where it changes fastest.
- **Not too far.** Use distances only up to about half the largest distance in the data.

**Why stop at half the largest distance?** The second half of the code's output computes the estimator in bins beyond 0.6:

| Bin | 0.6 to 0.7 | 0.7 to 0.8 | 0.8 to 0.9 | 0.9 to 1.0 | 1.0 to 1.1 | 1.1 to 1.2 | 1.2 to 1.3 | 1.3 to 1.4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pairs | 4016 | 3025 | 1964 | 1118 | 438 | 137 | 36 | 12 |
| $\hat{\gamma}$ | 5.341 | 6.172 | 7.169 | 7.874 | 7.874 | 7.407 | 6.602 | 9.695 |

The true semivariogram is flat at 5 out here, yet the estimates climb to nearly 8 and then wander. Two things go wrong at long distances. Pairs become scarce, as the counts show. More importantly, the long pairs all join opposite edges or corners of the district, so they compare a few large regions with each other instead of sampling the whole district. Within one realization, $u$ has large-scale ups and downs, and the long pairs measure the difference between whichever large regions happen to be high and low. In this realization $u$ happens to drift upward to the east (Section 3.4 measures the drift), so east-west pairs far apart differ more than the model says. The half-diameter rule keeps us in the range where many pairs from all over the district contribute to each bin.

### 3.1.4 Directional Variograms

Matheron's estimator as written pools pairs in all directions. If correlation reaches farther in one direction than another, as in the riverfront example of Section 2.6, pooling hides it. A **directional variogram** uses only pairs whose lag vector points within some angle of a chosen direction.

For a pair $(i, j)$, the direction of the lag vector is the angle $\vartheta_{ij}$ it makes with the east axis. Because the pair $(j, i)$ points the opposite way, directions are taken modulo 180 degrees, so 10 degrees and 190 degrees count as the same direction. The **angular tolerance** is the half-width of the wedge of directions pooled; 22.5 degrees is common, which splits the half-circle into four wedges centered on east-west, north-south, and the two diagonals.

To see the method work, we simulate a field we know to be anisotropic: 300 listings along a riverfront that runs east-west, with the anisotropy matrix $\mathbf{G} = \operatorname{diag}(1, 9)$ of Section 2.6, so that correlation reaches three times farther east-west than north-south. The other parameters match the rent district.

```python
def directional_variogram(S, y, edges, angle, tol):
    """Matheron estimator using pairs whose direction is within tol
    degrees of angle (measured from east, counterclockwise)."""
    H = dist(S)
    iu = np.triu_indices(len(y), k=1)
    dx = S[iu[1], 0] - S[iu[0], 0]
    dy = S[iu[1], 1] - S[iu[0], 1]
    theta = np.degrees(np.arctan2(dy, dx)) % 180.0
    off = np.abs(theta - angle)
    keep = np.minimum(off, 180.0 - off) <= tol
    h = H[iu][keep]
    d2 = ((y[iu[0]] - y[iu[1]])**2)[keep]
    gam, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h > lo) & (h <= hi)
        gam.append(d2[m].sum() / (2 * m.sum()))
        counts.append(m.sum())
    return np.array(gam), np.array(counts)


rng = np.random.default_rng(404)
S_r = rng.uniform(size=(300, 2))
D = S_r[:, None, :] - S_r[None, :, :]
G = np.diag([1.0, 9.0])
H_aniso = np.sqrt(np.einsum("ijk,kl,ijl->ij", D, G, D))
L_r = np.linalg.cholesky(4.0 * np.exp(-H_aniso / 0.15)
                         + 1.0 * np.eye(300))
y_r = L_r @ rng.standard_normal(300)
dir_edges = np.linspace(0.0, 0.4, 9)
for angle, name in ((0.0, "east-west"), (90.0, "north-south")):
    g_dir, c_dir = directional_variogram(S_r, y_r, dir_edges, angle, 22.5)
    print(f"{name:12s}", np.round(g_dir, 2), " pairs:", c_dir.min(),
          "to", c_dir.max())
```

| Distance bin | 0 to 0.05 | 0.05 to 0.10 | 0.10 to 0.15 | 0.15 to 0.20 | 0.20 to 0.25 | 0.25 to 0.30 | 0.30 to 0.35 | 0.35 to 0.40 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| East-west | 1.52 | 3.07 | 3.55 | 3.80 | 4.01 | 4.54 | 4.74 | 4.47 |
| North-south | 2.76 | 3.83 | 4.05 | 4.00 | 4.42 | 4.41 | 4.45 | 5.27 |

At short distances the north-south variogram is higher: 2.76 against 1.52 in the first bin and 3.83 against 3.07 in the second. North-south pairs lose their correlation faster, as designed. By about 0.25 both curves are near the same level, and beyond that they cross back and forth, because each directional bin has only a quarter of the pairs (between 80 and 737 here) and is correspondingly noisier. Figure 3.2 in Section 3.5 plots both curves.

Two refinements are worth knowing.

- **The rose diagram.** Computing directional variograms in many directions and recording, for each direction, the distance at which the variogram reaches a fixed level gives a set of points that, under geometric anisotropy, lie on an ellipse. Plotting them (often as a "rose" around the origin) shows the directions of longest and shortest reach, which are the eigenvectors of $\mathbf{G}$ in Section 2.6. Exercise 15 builds one.
- **Zonal anisotropy.** Geometric anisotropy changes how fast the variogram rises in each direction but not the sill it reaches. Sometimes the sills themselves differ. If rents vary by neighborhood across the river but are similar for long stretches along it, the north-south variogram may keep climbing to a higher level than the east-west one. That is **zonal anisotropy**, and it cannot be produced by stretching coordinates. It is usually modeled by adding a second process that varies in one direction only (Exercise 7).

---

## 3.2 Robust Estimation: The Cressie-Hawkins Estimator

### 3.2.1 Why Outliers Inflate Matheron's Estimator

Matheron's estimator squares differences, so large differences dominate it. One bad data point is worse than it looks, because it takes part in $n - 1$ pairs and contaminates every bin.

Rental listings contain bad data points. A landlord may include parking and utilities in the rent, or a typo may add a digit. To see the effect, take the rent district's $u + \varepsilon$ and add 12 (that is, \$1,200) to five randomly chosen listings, two percent of the data.

### 3.2.2 The Cressie-Hawkins Idea

Cressie and Hawkins proposed averaging a quantity that is less sensitive to large differences: the square root of the absolute difference, $\lvert Y_i - Y_j \rvert^{1/2}$. A difference of 12 contributes $12^2 = 144$ to Matheron's sum but only $\sqrt{12} \approx 3.5$ to theirs. The question is how to turn an average of square roots back into an estimate of $\gamma$.

The answer uses normality. If the data are Gaussian, the difference $Y_i - Y_j$ is normal with mean 0 and variance $2\gamma$, so we can write it as $\sqrt{2\gamma}\,Z$ with $Z \sim N(0, 1)$. Then

$$
\lvert Y_i - Y_j \rvert^{1/2} = (2\gamma)^{1/4}\lvert Z \rvert^{1/2}, \qquad \operatorname{E}\!\left[\lvert Y_i - Y_j \rvert^{1/2}\right] = (2\gamma)^{1/4}\,m, \qquad m = \operatorname{E}\!\left[\lvert Z \rvert^{1/2}\right]
$$

Raising the expectation to the fourth power gives $2\gamma\,m^4$. So if $\bar{a}_k$ is the average of $\lvert Y_i - Y_j \rvert^{1/2}$ over the pairs in bin $k$, then $\bar{a}_k^4/(2m^4)$ estimates $\gamma(h_k)$. Section 3.2.3 shows that $m^4 \approx 0.457$.

Raising an average to the fourth power introduces a small upward bias, for the same reason that the square of a sample mean overestimates the square of the true mean. Cressie and Hawkins corrected for it, and their estimator is

$$
\bar{\gamma}(h_k) = \frac{\bar{a}_k^4}{2\left(0.457 + 0.494/\lvert N(h_k) \rvert\right)}, \qquad \bar{a}_k = \frac{1}{\lvert N(h_k) \rvert}\sum_{(i,j) \in N(h_k)} \lvert Y_i - Y_j \rvert^{1/2}
$$

For bins with hundreds of pairs, the correction term is negligible.

```python
c_half = (2**0.25 * special.gamma(0.75) / np.sqrt(np.pi))**4
print("E|Z|^(1/2) raised to the fourth power:", round(c_half, 4))
m_half = 2**0.25 * special.gamma(0.75) / np.sqrt(np.pi)
var_root = np.sqrt(2 / np.pi) - m_half**2
print("Var(|Z|^(1/2)):", round(var_root, 4),
      " correction constant:", round(c_half * 6 * var_root / m_half**2, 4))


def cressie_hawkins(S, y, edges):
    """Cressie-Hawkins robust estimator on distance bins."""
    H = dist(S)
    iu = np.triu_indices(len(y), k=1)
    h = H[iu]
    r = np.sqrt(np.abs(y[iu[0]] - y[iu[1]]))
    gam = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h > lo) & (h <= hi)
        nk = m.sum()
        gam.append(r[m].mean()**4 / (2 * (0.457 + 0.494 / nk)))
    return np.array(gam)


z_clean = u + eps
rng = np.random.default_rng(99)
bad = rng.choice(n, size=5, replace=False)
z_bad = z_clean.copy()
z_bad[bad] += 12.0
_, g_m_clean, _ = empirical_variogram(S, z_clean, edges)
_, g_m_bad, _ = empirical_variogram(S, z_bad, edges)
g_ch_clean = cressie_hawkins(S, z_clean, edges)
g_ch_bad = cressie_hawkins(S, z_bad, edges)
for k in (0, 3, 7, 11):
    print(f"h={hc[k]:.3f}  Matheron clean={g_m_clean[k]:.3f} "
          f"bad={g_m_bad[k]:.3f}  CH clean={g_ch_clean[k]:.3f} "
          f"bad={g_ch_bad[k]:.3f}  true={gamma_exp(hc[k], 4, 1, .15):.3f}")
print("average increase, Matheron:",
      round(np.mean(g_m_bad - g_m_clean), 3),
      " Cressie-Hawkins:", round(np.mean(g_ch_bad - g_ch_clean), 3))
```

| $h_k$ | Matheron, clean | Matheron, 5 bad listings | Cressie-Hawkins, clean | Cressie-Hawkins, 5 bad listings | True $\gamma$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.034 | 1.844 | 5.173 | 1.887 | 2.728 | 1.803 |
| 0.176 | 3.931 | 7.579 | 3.853 | 5.196 | 3.765 |
| 0.375 | 4.579 | 9.078 | 4.397 | 6.130 | 4.672 |
| 0.575 | 5.382 | 8.378 | 5.385 | 6.638 | 4.913 |

On clean data the two estimators agree closely. With five bad listings, Matheron's estimator rises by 3.819 on average across the 12 bins and nearly triples in the first bin, which would suggest a nugget several times its true size. The Cressie-Hawkins estimator rises by 1.435 on average. It is far more resistant, but not immune: five gross errors still push it well above the truth. A robust estimator is a safety check, not a substitute for looking at the data. When the two estimators disagree sharply, find out why before fitting anything.

### 3.2.3 Where the Constants Come From (Optional)

*This subsection derives the constants 0.457 and 0.494 from the standard normal distribution. It can be skipped on a first reading.*

**The constant $m^4 \approx 0.457$.** For $Z \sim N(0, 1)$ and $p > -1$,

$$
\operatorname{E}\!\left[\lvert Z \rvert^p\right] = 2\int_0^\infty z^p\,\frac{1}{\sqrt{2\pi}}e^{-z^2/2}\,dz
$$

Substitute $w = z^2/2$, so that $z = (2w)^{1/2}$ and $dz = (2w)^{-1/2}dw$:

$$
\operatorname{E}\!\left[\lvert Z \rvert^p\right] = \frac{2}{\sqrt{2\pi}}\int_0^\infty (2w)^{p/2}(2w)^{-1/2}e^{-w}\,dw = \frac{2^{p/2}}{\sqrt{\pi}}\int_0^\infty w^{(p-1)/2}e^{-w}\,dw = \frac{2^{p/2}\,\Gamma\!\left(\frac{p+1}{2}\right)}{\sqrt{\pi}}
$$

using the definition of the gamma function, $\Gamma(a) = \int_0^\infty w^{a-1}e^{-w}\,dw$. With $p = 1/2$,

$$
m = \operatorname{E}\!\left[\lvert Z \rvert^{1/2}\right] = \frac{2^{1/4}\,\Gamma(3/4)}{\sqrt{\pi}} \approx 0.8222, \qquad m^4 \approx 0.4569
$$

The code above evaluates this as 0.4569, which Cressie and Hawkins rounded to 0.457.

**The correction 0.494.** The same formula with $p = 1$ gives $\operatorname{E}\lvert Z \rvert = \sqrt{2/\pi}$, so $\operatorname{Var}(\lvert Z \rvert^{1/2}) = \sqrt{2/\pi} - m^2 \approx 0.1219$. Suppose, as an approximation, that the $\lvert N \rvert$ terms in $\bar{a}_k$ are independent. Then, after scaling out $(2\gamma)^{1/4}$, $\bar{a}_k$ has mean $m$ and variance $0.1219/\lvert N \rvert$. Write $\bar{a}_k = m + d$, where $d$ has mean 0 and variance $0.1219/\lvert N \rvert$. Expanding the fourth power,

$$
\operatorname{E}\!\left[(m + d)^4\right] = m^4 + 4m^3\operatorname{E}[d] + 6m^2\operatorname{E}[d^2] + 4m\operatorname{E}[d^3] + \operatorname{E}[d^4] \approx m^4 + 6m^2\,\frac{0.1219}{\lvert N \rvert}
$$

keeping terms up to order $1/\lvert N \rvert$. Factoring out $m^4$,

$$
\operatorname{E}\!\left[\bar{a}_k^4\right] \approx m^4\left(1 + \frac{6 \times 0.1219}{m^2\lvert N \rvert}\right) = 0.4569 + \frac{0.4944}{\lvert N \rvert}
$$

which the code also confirms. Dividing by this expectation, rather than by $m^4$ alone, removes the upward bias to first order. The independence assumption is not exactly true, which is one more reason the correction is only approximate.

---

## 3.3 Fitting Parametric Curves to Empirical Variograms

The empirical variogram is a set of points. To use it for prediction (Chapter 4) we need a whole curve from one of the valid families of Section 2.5, since only those guarantee a positive semi-definite covariance matrix. We therefore choose a family and find the parameter values whose curve best matches the points.

It is convenient to collect the parameters into one vector, $\boldsymbol{\theta} = (\sigma^2, \tau^2, \phi)'$, and write the model semivariogram as $\gamma(h;\boldsymbol{\theta})$. For the exponential family, $\gamma(h;\boldsymbol{\theta}) = \tau^2 + \sigma^2(1 - e^{-h/\phi})$ for $h > 0$.

### 3.3.1 Why Ordinary Least Squares Is a Poor Choice

The obvious approach is nonlinear least squares: choose $\boldsymbol{\theta}$ to minimize

$$
\sum_k \left(\hat{\gamma}(h_k) - \gamma(h_k;\boldsymbol{\theta})\right)^2
$$

This treats every bin as equally reliable and as independent of the others. Neither is true.

- **Bins differ in precision.** A bin with 2,000 pairs is far more reliable than one with 200, and, as Section 3.3.2 shows, bins with larger semivariance are noisier. So the bins are heteroscedastic, and least squares should weight them.
- **Bins are correlated.** Every listing takes part in pairs in many bins, so the bins share data. A listing with an unusual value pushes up several bins at once. Ordinary least squares does not account for this, and neither, in fact, does the weighted method below. Chapter 5 avoids the issue entirely by fitting the model to the data rather than to the variogram.

### 3.3.2 The Approximate Variance of a Bin

How variable is $\hat{\gamma}(h_k)$? Start with one pair. Under normality, $Y_i - Y_j = \sqrt{2\gamma}\,Z$ with $Z \sim N(0, 1)$, so half the squared difference is

$$
\frac{(Y_i - Y_j)^2}{2} = \gamma\,Z^2
$$

Its mean is $\gamma\operatorname{E}[Z^2] = \gamma$, as Section 3.1.1 showed. Its variance needs the fourth moment of a standard normal. One way to find it uses the identity $\operatorname{E}[Z\,g(Z)] = \operatorname{E}[g'(Z)]$, which follows from integrating by parts against the normal density (Exercise 2). With $g(z) = z^3$,

$$
\operatorname{E}[Z^4] = \operatorname{E}[Z \cdot Z^3] = \operatorname{E}[3Z^2] = 3
$$

so $\operatorname{Var}(Z^2) = \operatorname{E}[Z^4] - (\operatorname{E}[Z^2])^2 = 3 - 1 = 2$, and

$$
\operatorname{Var}\!\left(\frac{(Y_i - Y_j)^2}{2}\right) = \gamma^2\operatorname{Var}(Z^2) = 2\gamma^2
$$

If the $\lvert N(h_k) \rvert$ pairs in a bin were independent, the variance of their average would be

$$
\operatorname{Var}\!\left(\hat{\gamma}(h_k)\right) \approx \frac{2\left[\gamma(h_k)\right]^2}{\lvert N(h_k) \rvert}
$$

This formula says what Section 3.3.1 claimed: bins with more pairs are more precise, and bins with larger semivariance are less precise, in proportion to $\gamma^2$.

The pairs are not independent, though. Two pairs that share a listing are correlated, and in a bin with thousands of pairs, each listing appears many times. How much does that matter? The code simulates 400 new rent fields at the same 250 locations and computes the estimator in the bin from 0.10 to 0.15 each time.

```python
Sigma_true = 4.0 * np.exp(-H / 0.15) + np.eye(n)
L_true = np.linalg.cholesky(Sigma_true)
rng = np.random.default_rng(5)
k = 2
iu = np.triu_indices(n, k=1)
mask = (H[iu] > edges[k]) & (H[iu] <= edges[k + 1])
draws = []
for _ in range(400):
    z = L_true @ rng.standard_normal(n)
    d2 = (z[iu[0]] - z[iu[1]])**2
    draws.append(d2[mask].sum() / (2 * mask.sum()))
g_true_k = gamma_exp(H[iu][mask], 4.0, 1.0, 0.15).mean()
approx = 2 * g_true_k**2 / mask.sum()
print(f"bin {edges[k]:.2f}-{edges[k+1]:.2f}: mean of estimates="
      f"{np.mean(draws):.3f}  true={g_true_k:.3f}")
print(f"variance: simulated={np.var(draws):.4f}  "
      f"approximation={approx:.4f}  ratio={np.var(draws) / approx:.1f}")
```

The average of the 400 estimates is 3.275 against a true average semivariance of 3.277 for the pairs in the bin, confirming that the estimator is unbiased. But the simulated variance is 0.1850, while the formula gives 0.0195: the true variance is 9.5 times larger. The 1,104 pairs in this bin carry far less information than 1,104 independent pairs would, for the same reason Chapter 1's correlated observations did. Exercise 3 traces this to the covariance between squared differences that share a listing.

So the formula is badly wrong in absolute terms. It is still useful in relative terms: it gets the direction right (more pairs help, larger $\gamma$ hurts), and that is all a set of weights needs.

### 3.3.3 Cressie's Weighted Least Squares

Weighted least squares weights each squared residual by the reciprocal of its variance. Using the approximation of Section 3.3.2, the weight for bin $k$ is $\lvert N(h_k) \rvert/(2\gamma(h_k)^2)$. Cressie proposed evaluating $\gamma$ in the weight at the model value $\gamma(h_k;\boldsymbol{\theta})$ rather than at the noisy estimate. Dropping the constant factor of 2, the objective is

$$
\sum_k \frac{\lvert N(h_k) \rvert}{\gamma(h_k;\boldsymbol{\theta})^2}\left(\hat{\gamma}(h_k) - \gamma(h_k;\boldsymbol{\theta})\right)^2 = \sum_k \lvert N(h_k) \rvert\left(\frac{\hat{\gamma}(h_k)}{\gamma(h_k;\boldsymbol{\theta})} - 1\right)^2
$$

The right-hand form shows what is being minimized: the relative error of the fit in each bin, weighted by the number of pairs.

Because the weights depend on $\boldsymbol{\theta}$, there is no closed-form solution. One approach is iterative: fix the weights at the current estimate, solve the resulting weighted least squares problem, update the weights, and repeat until the estimate stops changing. The other is to hand the whole objective to a numerical optimizer, which is what the code does. It optimizes over the logarithms of the parameters, which keeps all three positive without constraints.

```python
def fit_variogram(h, g, counts, model, start, weighted=True):
    """Fit (sigma2, tau2, phi) by OLS or Cressie's WLS on log parameters."""
    def loss(logp):
        s2, t2, ph = np.exp(logp)
        m = model(h, s2, t2, ph)
        if weighted:
            return np.sum(counts * (g / m - 1.0)**2)
        return np.sum((g - m)**2)
    res = optimize.minimize(loss, np.log(start), method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10,
                                     "maxiter": 5000})
    return np.exp(res.x)


def gamma_matern32(h, sigma2, tau2, phi):
    """Matern (nu = 3/2) semivariogram with a nugget, for h > 0."""
    x = h / phi
    return tau2 + sigma2 * (1 - (1 + x) * np.exp(-x))


start = (3.0, 0.5, 0.1)
for name, wt in (("OLS", False), ("WLS", True)):
    s2, t2, ph = fit_variogram(hc, g_oracle, cnt, gamma_exp, start, wt)
    print(f"{name}: sigma2={s2:.3f}  tau2={t2:.3f}  phi={ph:.3f}")
```

| Method | $\hat{\sigma}^2$ | $\hat{\tau}^2$ | $\hat{\phi}$ |
| :--- | :--- | :--- | :--- |
| Ordinary least squares | 3.987 | 1.291 | 0.188 |
| Cressie's weighted least squares | 3.919 | 1.342 | 0.186 |
| Truth | 4 | 1 | 0.15 |

Both fits are reasonable, and they are close to each other. Both put the range a little long and the nugget a little high. That is the honest result of one realization: the empirical variogram in Section 3.1.3 sits above the true curve between 0.10 and 0.25 and below it around 0.30, and any curve that follows the points inherits those wiggles. With a single map, the variogram is estimated with real uncertainty, and the parameters inherit it. Exercise 14 measures how much by repeating the fit over many simulated districts. Chapter 5 asks whether fitting the model directly to the data, by maximum likelihood, does better.

---

## 3.4 The Trend-Contamination Pitfall

So far we have estimated the variogram of $u + \varepsilon$, using our knowledge of the true trend. An analyst who has only the rents faces a choice: compute the variogram of the raw rents, or remove a trend first.

### 3.4.1 What an Ignored Trend Does

Write the data as $Y(\mathbf{s}) = \mu(\mathbf{s}) + e_{\text{true}}(\mathbf{s})$, where the true error process $e_{\text{true}} = u + \varepsilon$ is intrinsically stationary with semivariogram $\gamma_e$. (Section 3.4.2 will also need the vector of OLS residuals, which is written $\mathbf{e}$; the true errors always carry the subscript.) The difference of two observations is

$$
Y_i - Y_j = (\mu_i - \mu_j) + (e_{\text{true},i} - e_{\text{true},j})
$$

and its squared expectation, by the same one-variable identity as in Section 3.1.1, is

$$
\frac{1}{2}\operatorname{E}\!\left[(Y_i - Y_j)^2\right] = \gamma_e(h_{ij}) + \frac{1}{2}(\mu_i - \mu_j)^2
$$

So Matheron's estimator applied to raw data estimates the true semivariogram plus half the average squared difference in the mean.

For the rent district the trend is $\mu(\mathbf{s}) = 18 + 6\,s_{\text{east}}$, so $\mu_i - \mu_j = 6\,\Delta_{ij}$, where $\Delta_{ij}$ is the east-west separation of the pair. A pair at distance $h$ in direction $\vartheta$ has $\Delta_{ij} = h\cos\vartheta$. If the directions of pairs are spread evenly, the average of $\cos^2\vartheta$ is $1/2$ (Exercise 5), and the expected extra term is

$$
\frac{1}{2} \times 6^2 \times \frac{h^2}{2} = \frac{6^2 h^2}{4} = 9h^2
$$

It grows like $h^2$ and never levels off. A trend in the mean turns a variogram with a sill into one that rises without limit, which looks like very long-range spatial correlation.

```python
_, g_raw, _ = empirical_variogram(S, y, edges)
dx2 = (S[:, 0][:, None] - S[:, 0][None, :])**2
for k in (0, 3, 7, 11):
    m = np.triu((H > edges[k]) & (H <= edges[k + 1]), 1)
    extra = 0.5 * 6.0**2 * dx2[m].mean()
    print(f"h={hc[k]:.3f}  raw={g_raw[k]:.3f}  oracle={g_oracle[k]:.3f}  "
          f"difference={g_raw[k] - g_oracle[k]:.3f}  "
          f"trend term={extra:.3f}  6^2 h^2/4={6.0**2 * hc[k]**2 / 4:.3f}")
print("slope of u on east in this realization:",
      round(np.polyfit(S[:, 0], u, 1)[0], 3))
```

| $h_k$ | Raw rents | True $u + \varepsilon$ | Difference | Trend term for these pairs | $6^2h^2/4$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.034 | 1.866 | 1.844 | 0.021 | 0.011 | 0.010 |
| 0.176 | 4.318 | 3.931 | 0.387 | 0.281 | 0.280 |
| 0.375 | 6.660 | 4.579 | 2.082 | 1.307 | 1.266 |
| 0.575 | 9.962 | 5.382 | 4.580 | 3.042 | 2.971 |

The "trend term" column computes $\frac{1}{2}\,6^2\,\Delta_{ij}^2$ exactly for the pairs in each bin; it agrees closely with the $9h^2$ approximation. At short distances the trend barely matters. By the last bin it has nearly doubled the semivariance.

The observed difference is larger still, because the cross term $(\mu_i - \mu_j)(e_{\text{true},i} - e_{\text{true},j})$, which has expectation zero, is not zero in one realization. In this realization the neighborhood effect $u$ itself happens to drift upward to the east, with a fitted slope of 1.834 per unit distance, so it reinforces the trend. This is the same chance drift that inflated the long-distance bins in Section 3.1.3.

Fitting the exponential model to the raw-rent variogram shows the consequence. Cressie's weighted least squares, in the lab below, returns $\hat{\sigma}^2 \approx 2.7 \times 10^8$ and $\hat{\phi} \approx 2.1 \times 10^7$. For distances much smaller than $\phi$, $\sigma^2(1 - e^{-h/\phi}) \approx (\sigma^2/\phi)h$, a straight line, and the fit has found the line that best matches the rising points: a slope of $\hat{\sigma}^2/\hat{\phi} \approx 13.13$. The model is trying to describe an unbounded variogram and has run off to infinity in both parameters. When a variogram fit returns absurd values like these, suspect a trend.

### 3.4.2 Detrending With OLS Residuals

The standard fix is to estimate the trend by ordinary least squares and compute the variogram of the residuals. Not knowing the true trend, an analyst might fit a plane in the coordinates: $\mathbf{X}$ has columns for the intercept, the east coordinate, and the north coordinate. The residuals are

$$
\mathbf{e} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{Y}
$$

Figure 3.1 in Section 3.5 shows that the residual variogram is far closer to the truth than the raw one. But it is not unbiased, and the reason is instructive.

### 3.4.3 Why Residuals Have a Biased Variogram

The residuals are not the true errors $\mathbf{e}_{\text{true}} = \mathbf{u} + \boldsymbol{\varepsilon}$. Since $(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$,

$$
\mathbf{e} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})(\mathbf{u} + \boldsymbol{\varepsilon})
$$

and by Tool 1 their covariance matrix is

$$
\boldsymbol{\Sigma}_e = \operatorname{Var}(\mathbf{e}) = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\,\boldsymbol{\Sigma}\,(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})
$$

which is not $\boldsymbol{\Sigma}$. The residuals have mean zero, so the expected half squared difference of a pair of residuals is

$$
\frac{1}{2}\operatorname{E}\!\left[(e_i - e_j)^2\right] = \frac{1}{2}\left[(\boldsymbol{\Sigma}_e)_{ii} + (\boldsymbol{\Sigma}_e)_{jj} - 2(\boldsymbol{\Sigma}_e)_{ij}\right]
$$

Averaging this over the pairs in a bin gives the exact expected value of the residual variogram, which we can compare with the true $\gamma$.

Why should it be smaller? OLS chooses $\hat{\boldsymbol{\beta}}$ to make the residuals as small as possible. Any part of $\mathbf{u}$ that looks like a plane, such as the eastward drift found above, is indistinguishable from trend and is absorbed into $\hat{\boldsymbol{\beta}}$. The residuals lose that large-scale part of the spatial variation, and large-scale variation is exactly what the variogram measures at long distances. So the residual variogram is biased downward, and the bias grows with distance.

```python
X = np.column_stack([np.ones(n), S[:, 0], S[:, 1]])
P_X = X @ np.linalg.solve(X.T @ X, X.T)
e = y - P_X @ y
_, g_res, _ = empirical_variogram(S, e, edges)

Sigma_e = (np.eye(n) - P_X) @ Sigma_true @ (np.eye(n) - P_X)
d_e = np.diag(Sigma_e)
E_res = 0.5 * (d_e[:, None] + d_e[None, :] - 2 * Sigma_e)
gam_true_pairs = np.where(H > 0, gamma_exp(H, 4.0, 1.0, 0.15), 0.0)
for k in (0, 3, 7, 11):
    m = (H > edges[k]) & (H <= edges[k + 1])
    m = np.triu(m, 1)
    print(f"h={hc[k]:.3f}  expected residual gamma={E_res[m].mean():.3f}  "
          f"true gamma={gam_true_pairs[m].mean():.3f}  "
          f"observed residual gamma={g_res[k]:.3f}")
```

| $h_k$ | Expected residual $\gamma$ | True $\gamma$ | Observed residual $\hat{\gamma}$ |
| :--- | :--- | :--- | :--- |
| 0.034 | 1.793 | 1.794 | 1.843 |
| 0.176 | 3.715 | 3.759 | 3.907 |
| 0.375 | 4.404 | 4.670 | 4.390 |
| 0.575 | 4.244 | 4.913 | 5.179 |

The expected residual variogram (computed exactly from $\boldsymbol{\Sigma}_e$, without simulation) matches the true variogram at short distances and falls 14% below it by the last bin, as the argument predicts. (In the code, `Sigma_e` is $\boldsymbol{\Sigma}_e$ and `Sigma_true` is $\boldsymbol{\Sigma}$.) In this particular realization the observed residual variogram happens to land above the truth in the last bin. Bias is a statement about the average over many realizations, and one realization can fall on either side.

The bias matters in practice. The fitted partial sill in the lab is 3.736 from the residuals against 3.919 from the true errors. The fix is to estimate the trend and the covariance together, so that the covariance estimate accounts for the part of the variation the trend absorbs. That is what restricted maximum likelihood (REML) does in Chapter 5.

---

## 3.5 Walkthrough and Lab 3

### 3.5.1 From Raw Listings to a Fitted Variogram

The first part of the lab fits the exponential model by Cressie's weighted least squares to three versions of the rent variogram: the raw rents, the OLS residuals, and the true $u + \varepsilon$. It also fits the Matérn model with $\nu = 3/2$ to the residuals, and prints each fit's practical range (Section 2.5).

```python
print("fits (sigma2, tau2, phi):")
for name, g in (("raw rents", g_raw), ("OLS residuals", g_res),
                ("oracle u + eps", g_oracle)):
    s2, t2, ph = fit_variogram(hc, g, cnt, gamma_exp, start)
    print(f"  {name:15s} exponential WLS: {s2:7.3f} {t2:6.3f} {ph:7.3f}"
          f"  practical range={2.996 * ph:.3f}")
s2, t2, ph = fit_variogram(hc, g_raw, cnt, gamma_exp, start)
print(f"  raw-rent fit: sigma2 / phi = {s2 / ph:.2f}")
s2, t2, ph = fit_variogram(hc, g_res, cnt, gamma_matern32, start)
print(f"  {'OLS residuals':15s} Matern 3/2 WLS:  "
      f"{s2:7.3f} {t2:6.3f} {ph:7.3f}  practical range={4.744 * ph:.3f}")
```

| Data | Model | $\hat{\sigma}^2$ | $\hat{\tau}^2$ | $\hat{\phi}$ | Practical range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Raw rents | Exponential | $2.7 \times 10^8$ | 1.812 | $2.1 \times 10^7$ | not meaningful |
| OLS residuals | Exponential | 3.736 | 1.342 | 0.178 | 0.534 |
| True $u + \varepsilon$ | Exponential | 3.919 | 1.342 | 0.186 | 0.559 |
| OLS residuals | Matérn, $\nu = 3/2$ | 2.907 | 1.965 | 0.087 | 0.415 |
| Truth | Exponential | 4 | 1 | 0.15 | 0.449 |

The raw-rent fit has run away, as Section 3.4.1 explained. The residual fit is close to the fit on the true errors, with a smaller partial sill, as Section 3.4.3 predicted. The Matérn fit to the same residuals splits the variance differently: a larger nugget and smaller partial sill, with a shorter practical range. Both curves pass close to the points (Exercise 17 compares their objective values), so the variogram alone does not settle which family is right. Chapter 5 compares families with likelihood-based criteria.

The next block draws the three empirical variograms with the true curve and the residual fit (Figure 3.1), the directional variograms of Section 3.1.4 (Figure 3.2), and the contaminated estimators of Section 3.2 (Figure 3.3). As in Chapter 2, run it from the top folder of the repository.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch03", exist_ok=True)
hh = np.linspace(0.001, 0.6, 200)
s2, t2, ph = fit_variogram(hc, g_res, cnt, gamma_exp, start)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hc, g_raw, "s", label="raw rents (trend ignored)")
ax.plot(hc, g_res, "o", label="OLS residuals")
ax.plot(hc, g_oracle, "^", label="true u + eps (known in simulation)")
ax.plot(hh, gamma_exp(hh, 4.0, 1.0, 0.15), "k-", label="true variogram")
ax.plot(hh, gamma_exp(hh, s2, t2, ph), "k--",
        label="WLS fit to residuals")
ax.set_xlabel("distance h")
ax.set_ylabel("semivariance (hundreds of dollars squared)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ch03/fig_3_1_rent_variograms.png", dpi=120)

mid = 0.5 * (dir_edges[:-1] + dir_edges[1:])
fig, ax = plt.subplots(figsize=(7, 4))
for angle, name in ((0.0, "east-west"), (90.0, "north-south")):
    g_dir, _ = directional_variogram(S_r, y_r, dir_edges, angle, 22.5)
    ax.plot(mid, g_dir, "o-", label=name)
ax.set_xlabel("distance h")
ax.set_ylabel("semivariance")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ch03/fig_3_2_directional.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hc, g_m_bad, "s", label="Matheron, 5 bad listings")
ax.plot(hc, g_ch_bad, "o", label="Cressie-Hawkins, 5 bad listings")
ax.plot(hh, gamma_exp(hh, 4.0, 1.0, 0.15), "k-", label="true variogram")
ax.set_xlabel("distance h")
ax.set_ylabel("semivariance")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ch03/fig_3_3_outliers.png", dpi=120)
print("figures saved")
```

![Figure 3.1. Empirical semivariograms of the simulated rent district. Ignoring the eastward trend makes the variogram climb without leveling off; the OLS residuals and the true errors nearly coincide and follow the true curve.](../figures/ch03/fig_3_1_rent_variograms.png)

In Figure 3.1 the raw-rent points (squares) follow the others for the first two bins and then pull away, reaching nearly 10 by the last bin with no sign of leveling off. The residual points (circles) and the true-error points (triangles) are almost indistinguishable and track the true curve, wiggling around it in the way Section 3.3.3 described. The dashed weighted least squares fit to the residuals lies close to the true curve, slightly above it near the origin because its nugget is larger.

### 3.5.2 Directional Variograms

![Figure 3.2. Directional semivariograms of a simulated field whose correlation reaches three times farther east-west than north-south. The north-south curve rises faster at short distances.](../figures/ch03/fig_3_2_directional.png)

Figure 3.2 plots the table of Section 3.1.4. The north-south curve starts well above the east-west curve and stays above it to about 0.25, where the two meet. Beyond that the curves cross, because the directional bins have fewer pairs and more noise. Detecting anisotropy from directional variograms is therefore most reliable at short and moderate distances.

### 3.5.3 Outliers

![Figure 3.3. Matheron and Cressie-Hawkins estimates after five listings are inflated by 12 (that is, \$1,200). Both lie above the true curve, but the robust estimator is much closer.](../figures/ch03/fig_3_3_outliers.png)

In Figure 3.3 the Matheron estimates (squares) jump to between 5 and 9 across all bins, roughly doubling the semivariance and turning the gentle rise of the true curve into a high, flat band. That pattern would suggest a nugget several times its true size. The Cressie-Hawkins estimates (circles) are still too high, but they keep the rising shape and are much closer to the truth.

---

## Exercises

**Analytical**

1. Show that if $\operatorname{E}[Y_i - Y_j] = \Delta\mu_{ij} \ne 0$, then $\operatorname{E}[\hat{\gamma}(h_k)]$ equals the average of $\gamma$ over the bin plus $\frac{1}{2\lvert N(h_k) \rvert}\sum_{(i,j) \in N(h_k)} (\Delta\mu_{ij})^2$. Use this to recover the trend term of Section 3.4.1.

2. Let $Z \sim N(0, 1)$ with density $f(z)$. Show that $f'(z) = -zf(z)$, and use integration by parts to prove that $\operatorname{E}[Z\,g(Z)] = \operatorname{E}[g'(Z)]$ for any differentiable $g$ that does not grow too fast. Use it to find $\operatorname{E}[Z^4]$ and $\operatorname{E}[Z^6]$.

3. For jointly normal mean-zero variables $A$ and $B$, a standard result is $\operatorname{Cov}(A^2, B^2) = 2\operatorname{Cov}(A, B)^2$. Use it to show that two squared differences that share a location, $(Y_1 - Y_2)^2$ and $(Y_1 - Y_3)^2$, have positive covariance even when $Y_2$ and $Y_3$ are independent of each other. Explain how this accounts for the factor of 9.5 in Section 3.3.2.

4. Using the formula $\operatorname{E}\lvert Z \rvert^p = 2^{p/2}\Gamma((p+1)/2)/\sqrt{\pi}$ from Section 3.2.3, check the cases $p = 0$ and $p = 2$, and compute $\operatorname{E}\lvert Z \rvert$. Then redo the correction-constant calculation keeping the term $4m\operatorname{E}[d^3]$, using the skewness of $\lvert Z \rvert^{1/2}$, and judge whether it matters for bins with 100 pairs.

5. Show that if a direction $\vartheta$ is uniformly distributed on $[0, \pi)$, then $\operatorname{E}[\cos^2\vartheta] = 1/2$. For a trend $\mu(\mathbf{s}) = \mathbf{b}'\mathbf{s}$ in an arbitrary direction, show that the expected extra semivariance at distance $h$ is $\lVert\mathbf{b}\rVert^2h^2/4$.

6. For the intercept-only model $\mathbf{X} = \mathbf{1}$, show that $e_i - e_j = Y_i - Y_j$ for every pair, so the residual variogram equals the raw variogram exactly. Explain why the downward bias of Section 3.4.3 appears only when $\mathbf{X}$ has columns that vary across space.

7. Let $u_1(\mathbf{s})$ be an isotropic process with sill $\sigma_1^2$, and let $u_2(\mathbf{s}) = v(s_{\text{north}})$ be a process that varies only in the north-south direction, with semivariogram $\gamma_v$ and sill $\sigma_2^2$, independent of $u_1$. Find the semivariogram of $u_1 + u_2$ for lags pointing due east and due north, and show that the two have different sills. This is zonal anisotropy.

8. Section 3.1.1 defined Matheron's estimator for bin $k$ as $\hat{\gamma}(h_k) = \frac{1}{2\lvert N(h_k) \rvert}\sum_{(i,j) \in N(h_k)} (Y_i - Y_j)^2$. (a) Define the $n \times n$ symmetric adjacency matrix $\mathbf{A}_k$ for bin $k$, where $(\mathbf{A}_k)_{ij} = 1$ if $(i, j) \in N(h_k)$ or $(j, i) \in N(h_k)$, and $0$ otherwise. Let $\mathbf{D}_k = \operatorname{diag}(\mathbf{A}_k\mathbf{1}_n)$ be its diagonal degree matrix. Show that $\sum_{(i,j) \in N(h_k)} (Y_i - Y_j)^2 = \mathbf{Y}'(\mathbf{D}_k - \mathbf{A}_k)\mathbf{Y} = \mathbf{Y}'\mathbf{L}_k\mathbf{Y}$, where $\mathbf{L}_k$ is the graph Laplacian of bin $k$. (b) Using Tool 2 (Section 1.1.1), show that $\operatorname{E}[\hat{\gamma}(h_k)] = \frac{\operatorname{tr}(\mathbf{L}_k\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{L}_k\boldsymbol{\mu}}{2\lvert N(h_k) \rvert}$, and show that the mean term vanishes whenever the mean is constant ($\boldsymbol{\mu} = \mu\mathbf{1}_n$) because $\mathbf{L}_k\mathbf{1}_n = \mathbf{0}$. (c) Evaluate the trace $\operatorname{tr}(\mathbf{L}_k\boldsymbol{\Sigma})$ explicitly to recover the bin average of $\gamma$.

9. Section 3.3.2 discussed why pairs sharing a listing are correlated. Suppose observations are Gaussian with covariance matrix $\boldsymbol{\Sigma}$. (a) For two distinct pairs $(1, 2)$ and $(1, 3)$ sharing location $1$, show that $\operatorname{Cov}(Y_1 - Y_2, Y_1 - Y_3) = \gamma(\mathbf{s}_1 - \mathbf{s}_3) + \gamma(\mathbf{s}_1 - \mathbf{s}_2) - \gamma(\mathbf{s}_2 - \mathbf{s}_3)$. (b) Using $\operatorname{Cov}(A^2, B^2) = 2\operatorname{Cov}(A, B)^2$ from Exercise 3, show that the covariance between the two squared differences is $2\left[\gamma(\mathbf{s}_1 - \mathbf{s}_3) + \gamma(\mathbf{s}_1 - \mathbf{s}_2) - \gamma(\mathbf{s}_2 - \mathbf{s}_3)\right]^2$. (c) For three locations forming an equilateral triangle of side $h$, evaluate this covariance in terms of $\gamma(h)$.

10. Let $\mathbf{s}_1, \dots, \mathbf{s}_n$ be $n$ locations with semivariogram matrix $\boldsymbol{\Gamma}$ ($\Gamma_{ij} = \gamma(\mathbf{s}_i - \mathbf{s}_j)$, with zeros on the diagonal). (a) Using Section 2.2.4 ($\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$), show that $\boldsymbol{\Gamma} = C(\mathbf{0})\mathbf{1}_n\mathbf{1}_n' - \boldsymbol{\Sigma}$. (b) For any vector $\mathbf{a}$ orthogonal to $\mathbf{1}_n$ ($\mathbf{a}'\mathbf{1}_n = 0$), show that $\mathbf{a}'\boldsymbol{\Gamma}\mathbf{a} = -\mathbf{a}'\boldsymbol{\Sigma}\mathbf{a} \le 0$. (c) Explain why this conditional negative definiteness guarantees that the matrix $\mathbf{B} = -\frac{1}{2}\mathbf{P}_c\boldsymbol{\Gamma}\mathbf{P}_c$ is positive semi-definite (where $\mathbf{P}_c = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n\mathbf{1}_n'$ is the centering matrix), and connect this to classical multidimensional scaling.

11. Suppose $Y(\mathbf{s})$ is an intrinsically stationary point process with semivariogram $\gamma(h)$. In many remote sensing applications, an instrument records the spatial average of $Y$ over a square pixel $B(\mathbf{s})$ centered at $\mathbf{s}$: $Y_B(\mathbf{s}) = \frac{1}{\lvert B \rvert}\int_{B(\mathbf{s})} Y(\mathbf{u})\,d\mathbf{u}$. (a) Show that the semivariogram of the block-averaged process is $\gamma_B(\mathbf{h}) = \bar{\gamma}(B(\mathbf{s}), B(\mathbf{s}+\mathbf{h})) - \bar{\gamma}(B, B)$, where $\bar{\gamma}(B_1, B_2) = \frac{1}{\lvert B_1 \rvert \lvert B_2 \rvert}\int_{B_1}\int_{B_2}\gamma(\mathbf{u} - \mathbf{v})\,d\mathbf{u}\,d\mathbf{v}$. (b) Show that $\gamma_B(\mathbf{0}) = 0$, but the sill of $\gamma_B$ is strictly less than the sill of the original point process $\gamma$. Explain why area-averaging smooths out variation and reduces both the partial sill and the nugget (the change of support effect).

12. Section 3.1.4 and Section 2.6 describe geometric anisotropy through the elliptical lag distance $h(\mathbf{G}) = \sqrt{\mathbf{h}'\mathbf{G}\mathbf{h}}$. (a) Let $\mathbf{G} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ with rotation matrix $\mathbf{U} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \lambda_2)$ with $0 < \lambda_1 < \lambda_2$. Show that the contour of constant semivariance $\gamma(h(\mathbf{G})) = c$ forms an ellipse $\mathbf{h}'\mathbf{G}\mathbf{h} = h_0^2$ in lag space $\mathbf{h} = (h_x, h_y)'$. (b) Find the lengths of the semi-major and semi-minor axes of this ellipse in terms of $h_0, \lambda_1, \lambda_2$. (c) Verify that the direction of longest reach (the semi-major axis of the ellipse) points in direction $\theta$ with length $h_0/\sqrt{\lambda_1}$, confirming that the smallest eigenvalue of $\mathbf{G}$ corresponds to the longest spatial range.

**Computational and simulation**

13. Recompute the rent district's empirical variogram with bin widths 0.025, 0.05, and 0.1, and refit the exponential model by Cressie's weighted least squares each time. How sensitive are $\hat{\sigma}^2$, $\hat{\tau}^2$, and $\hat{\phi}$ to the bin width?

14. Keep the 250 listing locations fixed, simulate 200 new values of $u + \varepsilon$, and fit the exponential model to each by both ordinary and weighted least squares. Report the mean and standard deviation of $\hat{\phi}$ for each method. Does weighting reduce the variability?

15. For the anisotropic field of Section 3.1.4, compute directional variograms at 0, 45, 90, and 135 degrees. For each direction, find (by linear interpolation between bins) the distance at which the variogram first reaches 3.5. Plot these distances as points at the corresponding angles and their opposites, and compare the shape with the ellipse implied by $\mathbf{G}$.

16. Repeat the outlier experiment of Section 3.2 with 1, 2, 5, 10, and 20 bad listings. Plot the average increase of each estimator against the number of bad listings.

17. Compute the weighted least squares objective at the fitted values for the exponential and Matérn $\nu = 3/2$ fits to the OLS residuals in Section 3.5.1. Then fit the Matérn model with $\nu = 5/2$. Can the objective values be used to choose among the families? Why or why not?

---

## What Comes Next

The variogram is now an estimated curve with a fitted family and parameters. Chapter 4 uses such a curve for its main purpose: predicting the value of the process at locations where we have no data, such as the rent of an apartment that is not listed. Kriging, as that prediction is called, turns out to be best linear unbiased prediction in the sense of the linear models you know. Chapter 5 then returns to estimation and replaces variogram fitting with maximum likelihood and REML, which use the data directly and fix the residual bias of Section 3.4.3.
