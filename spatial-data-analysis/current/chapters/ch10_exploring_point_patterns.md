# Chapter 10: Exploring Point Patterns: Intensity, Randomness, and Interaction

Up to now every observation in this book was a number measured at a place. A rent at an address, a noise level at a monitor, a death count in a county: the locations were fixed by the study design or the map, and the randomness lived in the values. This chapter starts Part IV with data of a different kind. There is no value to measure. The data are the places themselves: where the immune cells sit on a tumor slide, where the traffic crashes happened, where the disease cases live. The coordinates are the random outcome.

That change removes most of the machinery of Parts I to III, which was built for responses $Y(\mathbf{s})$, but it keeps the habit of mind. We ask what a pattern with no structure looks like, describe how real patterns depart from it, and judge the departures against simulated worlds where the truth is known. The chapter's job is exploration, in the same sense that Chapter 3 explored a variogram before Chapter 5 fitted one. Section by section, we meet the null model (complete spatial randomness), the first-order question of where events are dense, the second-order question of whether events attract or repel one another, and the multitype question of whether two kinds of event are arranged differently.

Three stories run through the chapter. Immune cells on a tumor slide supply the question of clustering (do they swarm together?) and of inhibition (do cancer cells keep them out?). Traffic crashes along a highway supply an intensity that varies from place to place. Cases and controls of a disease around an industrial plant supply the multitype question. The models behind the clustered and inhibited patterns are derived in Chapter 12, and fitting models to point patterns is the subject of Chapter 11. Here we only simulate them, so that the exploratory tools have something to find.

> **Chapter at a Glance**
>
> - **What Changes (§10.1):** The coordinate matrix is now the random outcome, and the question moves from "how large is the value?" to "where are the events, and how do they relate to each other?"
> - **Complete Spatial Randomness (§10.2):** The Poisson process as a limit of Bernoulli trials on a fine grid, conditional uniformity, and quadrat counts.
> - **Simulating Patterns (§10.3):** Homogeneous patterns, Lewis-Shedler thinning, and two patterns that are not random.
> - **Kernel Intensity Estimation (§10.4):** Bias, variance, edge correction, and bandwidth selection by likelihood cross-validation.
> - **Distance Methods (§10.5):** Nearest-neighbor and empty-space distributions, and the $J$-function.
> - **Second-Order Summaries (§10.6):** Ripley's $K$ with edge weights, Besag's $L$, and the pair correlation function.
> - **Multitype Patterns (§10.7):** Cross-type summaries and the spatial relative risk, with an honest look at how much data a relative risk map needs.
> - **Boundary Effects (§10.8):** Why every estimator is biased near the edge of the window, and three ways to fix it, with the geometry of Ripley's weight.
> - **Monte Carlo Envelopes (§10.9):** Why pointwise envelopes reject too often, and a global test that does not.
> - **Walkthrough and Lab (§10.10):** All the summaries, with envelopes, for random, clustered, and inhibited patterns, and the case-control data.

---

## 10.1 What Changes in a Point Pattern

Take the immune cells on a tumor slide. A pathologist stains the tissue, a scanner finds the cells, and the result is a list of coordinates: $n$ rows, two columns. In the notation of Chapter 1 this is the coordinate matrix $\mathbf{S}$, with rows $\mathbf{s}_i'$. In Parts II and III the matrix was fixed and a vector $\mathbf{Y}$ of measurements hung off it. Here the matrix is all there is, and both its entries and its number of rows are random.

Three kinds of question follow, and they organize the chapter.

- **Where are events dense?** This is the **first-order** question, about the expected number of events per unit area. Are crashes more frequent near the highway? Is there a hot spot of immune cells in one corner of the slide? The answer is a map of an intensity, and Section 10.4 estimates it.
- **Do events attract or repel each other?** This is the **second-order** question, about pairs of events. Given that an immune cell is here, is another one more or less likely to be nearby than a random scattering would suggest? Sections 10.5 and 10.6 answer it with distances between events.
- **Do two kinds of event differ?** This is the **multitype** question. Cases and controls, made shots and missed shots, two species of tree. Section 10.7 compares their arrangements.

First- and second-order questions are hard to separate. A pattern can look clustered because events are dense in one region (a first-order effect) or because events gather in small groups wherever they appear (a second-order effect), and the exploratory tools in this chapter mostly assume the first kind is absent. Chapter 11 returns to the problem with models that handle both.

Two words need care. The **intensity** $\lambda(\mathbf{s})$ at a location is the expected number of events per unit area near $\mathbf{s}$; it has units of events per area. It is not a probability, and it is not the probability density of an event's location. The two differ by the factor of the expected total number of events: if $n$ events fall in a window of area 1 with density $f(\mathbf{s})$, then the intensity is $nf(\mathbf{s})$. The second word is **window**. A pattern is always observed in a bounded region, here the unit square $\mathcal{D}$, and events outside it are invisible. Every estimator in this chapter has to cope with that, and Section 10.8 collects the fixes.

Throughout, the study window is the unit square. When a story needs real units, imagine the square scaled to a city map or a tissue slide; none of the methods depends on the scale.

---

## 10.2 The Homogeneous Poisson Process: Complete Spatial Randomness

To say what a pattern with no structure looks like, we need a precise null model. The standard choice is the **homogeneous Poisson process**, whose output is called **complete spatial randomness** (CSR). It has two defining properties: events fall independently of each other, and every location is equally likely to receive one. This section derives what that means for counts and for locations.

### 10.2.1 The Poisson Count as a Limit of Bernoulli Trials

Cover the window with a fine grid of $m \times m$ square pixels, each of area $1/m^2$. Suppose each pixel independently contains an event with probability $p = \lambda/m^2$, where $\lambda$ is the expected number of events per unit area. If the pixels are small enough that none holds two events, the pattern is a record of which pixels are occupied. Nothing about one pixel influences another, and every pixel has the same chance: that is "no interaction, no trend" in the language of grids.

Now take a region $B$ of area $|B|$. It contains about $N_B = |B|m^2$ pixels, so the number of events in it is binomial, $N(B) \sim \text{Binomial}(N_B, p)$ with mean $N_B p = \lambda|B|$. As the grid is refined, $m \to \infty$, the number of pixels grows and the probability per pixel shrinks, with the mean held fixed. The binomial distribution then converges to a Poisson distribution. The probability generating function shows why. For a Binomial$(N_B, p)$ variable,

$$
\operatorname{E}\left[z^{N(B)}\right] = \left(1 + p(z - 1)\right)^{N_B} = \left(1 + \frac{\lambda|B|(z - 1)}{N_B}\right)^{N_B} \longrightarrow \exp\left\{\lambda|B|(z - 1)\right\}
$$

using $\left(1 + x/N\right)^N \to e^x$, and the limit is the generating function of the Poisson distribution with mean $\lambda|B|$. So

$$
N(B) \sim \text{Poisson}\left(\lambda\operatorname{area}(B)\right)
$$

for every region $B$. Counts in disjoint regions are independent, because they are made of disjoint sets of independent pixels. Together, these two statements define the homogeneous Poisson process.

The convergence is quick. The code takes $\lambda = 100$ and a region of area 0.3, and compares the exact binomial probabilities with the Poisson ones as the grid is refined.

```python
import numpy as np
from scipy import stats

lam, area_B = 100.0, 0.3
k = np.arange(0, 80)
for m in (16, 32, 64, 128, 256):
    n_cells = int(round(area_B * m**2))
    p = lam / m**2
    gap = np.abs(stats.binom.pmf(k, n_cells, p)
                 - stats.poisson.pmf(k, lam * area_B)).max()
    print(f"m={m:4d}  cells in B={n_cells:6d}  p={p:.5f}  "
          f"largest pmf gap={gap:.5f}")
```

With a $16 \times 16$ grid the largest gap between the two probability functions is 0.020. At $64 \times 64$ it is 0.0009, and at $256 \times 256$ it is 0.00006. Pixels need to be small relative to the spacing between events, and then the Poisson description is excellent.

### 10.2.2 Conditional Uniformity

A second consequence matters even more in practice. Suppose we know that the window holds exactly $n$ events. Where are they?

Split the window into disjoint regions $B_1, \dots, B_k$ with areas $|B_1|, \dots, |B_k|$ that sum to $|\mathcal{D}|$. The counts $N(B_j)$ are independent Poisson variables, so

$$
\Pr\left(N(B_1) = n_1, \dots, N(B_k) = n_k \,\middle|\, N(\mathcal{D}) = n\right) = \frac{\prod_{j=1}^k e^{-\lambda|B_j|}(\lambda|B_j|)^{n_j}/n_j!}{e^{-\lambda|\mathcal{D}|}(\lambda|\mathcal{D}|)^n/n!} = \frac{n!}{\prod_{j=1}^k n_j!}\prod_{j=1}^k \left(\frac{|B_j|}{|\mathcal{D}|}\right)^{n_j}
$$

where $n_1 + \dots + n_k = n$. The factors $e^{-\lambda|B_j|}$ multiply to the $e^{-\lambda|\mathcal{D}|}$ in the denominator and cancel, and so do the powers of $\lambda$. What is left is the multinomial distribution: each of the $n$ events independently lands in region $B_j$ with probability $|B_j|/|\mathcal{D}|$. That is exactly what it means for the $n$ events to be independent and uniformly distributed over the window.

> Given $N(\mathcal{D}) = n$, the events of a homogeneous Poisson process are $n$ independent uniform points on $\mathcal{D}$.

This is why the simulation of CSR in Section 10.3 is two lines: draw a Poisson count, then draw uniform coordinates. It is also why the intensity $\lambda$ is a nuisance for most tests. Conditional on the number of events, the pattern's distribution does not depend on $\lambda$, so tests and envelopes can be built from patterns with the same $n$ as the data, and no value of $\lambda$ has to be chosen.

### 10.2.3 Quadrat Counts

The oldest test for randomness divides the window into equal **quadrats** and compares the counts. With $n_q$ quadrats and counts $n_1, \dots, n_{n_q}$, let $\bar{n} = n/n_q$ be the average count. Conditional on $n$, Section 10.2.2 says the counts are multinomial with equal cell probabilities, and Pearson's goodness-of-fit statistic is

$$
X^2 = \sum_{k=1}^{n_q}\frac{(n_k - \bar{n})^2}{\bar{n}}
$$

which is approximately $\chi^2_{n_q - 1}$ when $\bar{n}$ is not small. The degrees of freedom are $n_q - 1$, not $n_q$, because the counts are constrained to sum to $n$. In fact its mean is exactly $n_q - 1$: each count has variance $n(1/n_q)(1 - 1/n_q)$, so $\sum_k \operatorname{Var}(n_k) = n(1 - 1/n_q)$, and dividing by $\bar{n} = n/n_q$ gives $n_q - 1$.

A closely related number is the **index of dispersion** $s_n^2/\bar{n}$, the variance of the quadrat counts divided by their mean. Since $s_n^2 = \sum_k(n_k - \bar{n})^2/(n_q - 1)$, the index equals $X^2/(n_q - 1)$. For a Poisson count the variance equals the mean, so under CSR the index is near 1. Clustering makes some quadrats much fuller than others and pushes the index above 1. Regular spacing makes the counts too even and pushes it below 1.

**A hand-sized example.** Four events at $(0.1, 0.1)$, $(0.4, 0.5)$, $(0.9, 0.2)$, and $(0.6, 0.9)$ fall one in each quadrant of the unit square. With $n_q = 4$ and $\bar{n} = 1$, every term of $X^2$ is zero, so $X^2 = 0$ and the index of dispersion is 0: perfectly even, as far as this coarse grid can tell. The same four points reappear in Sections 10.5, 10.6, and 10.8.

The code below computes $X^2$ for the four points and then checks the null distribution by simulation: 4,000 CSR patterns with a mean of 200 events, counted in a $5 \times 5$ grid.

```python
def quadrat_x2(P, m=5):
    """Pearson X^2 for counts in an m-by-m grid of equal quadrats."""
    cell = np.minimum((P * m).astype(int), m - 1)
    counts = np.bincount(cell[:, 0] * m + cell[:, 1], minlength=m * m)
    nbar = counts.mean()
    return ((counts - nbar)**2).sum() / nbar, counts


P4 = np.array([[0.1, 0.1], [0.4, 0.5], [0.9, 0.2], [0.6, 0.9]])
x2, counts = quadrat_x2(P4, m=2)
print("four points, 2 x 2 quadrats: counts", counts, " X^2 =", x2)

rng = np.random.default_rng(2080)
n_q, reps = 25, 4000
x2_sim = np.array([quadrat_x2(rng.uniform(size=(rng.poisson(200), 2)))[0]
                   for _ in range(reps)])
lo, hi = stats.chi2.ppf([0.025, 0.975], n_q - 1)
print(f"X^2 under CSR: mean={x2_sim.mean():.2f} (df={n_q - 1}), "
      f"sd={x2_sim.std():.2f} (sqrt(2 df)={np.sqrt(2 * (n_q - 1)):.2f})")
print(f"two-sided 5% test rejects CSR in "
      f"{np.mean((x2_sim < lo) | (x2_sim > hi)):.3f} of {reps} patterns")
```

The simulated $X^2$ has mean 23.91 and standard deviation 6.88, against 24 and $\sqrt{48} = 6.93$ for a $\chi^2_{24}$ distribution. A two-sided 5% test rejects CSR in 4.8% of the patterns, as it should.

Quadrat counts are easy to compute and easy to explain, but they have two weaknesses that Section 10.3.4 will expose. The test says nothing about distances smaller than the quadrat, and it depends on a choice of grid.

---

## 10.3 Simulating Point Patterns From First Principles

Every tool in this chapter has to be judged against what random patterns do. We therefore start by learning to make them.

### 10.3.1 Homogeneous Patterns

By Section 10.2.2, a CSR pattern with intensity $\lambda$ on the unit square has a Poisson($\lambda$) number of events at independent uniform locations. The function below does exactly that. The check draws 5,000 patterns with $\lambda = 200$ and confirms what the theory says about the count (mean and variance both 200) and about the locations (uniform on the square, with mean 0.5 and standard deviation $1/\sqrt{12} = 0.289$).

```python
def sim_poisson(lam, rng):
    """Homogeneous Poisson process on the unit square."""
    return rng.uniform(size=(rng.poisson(lam), 2))


rng = np.random.default_rng(2080)
counts = np.array([len(sim_poisson(200, rng)) for _ in range(5000)])
print(f"number of events: mean={counts.mean():.1f}, "
      f"variance={counts.var():.1f} (Poisson: both equal 200)")
P = sim_poisson(200, rng)
print(f"one pattern: {len(P)} events; mean x={P[:, 0].mean():.3f}, "
      f"mean y={P[:, 1].mean():.3f}, sd x={P[:, 0].std():.3f} "
      f"(uniform: 0.5 and {np.sqrt(1 / 12):.3f})")
```

The counts have mean 199.9 and variance 201.4. In the single pattern drawn afterward, 201 events have mean coordinates of 0.467 and 0.505 and a standard deviation of 0.287 in $x$, all consistent with uniformity.

### 10.3.2 Lewis-Shedler Thinning

Real intensities are not constant. Crashes pile up along a highway, and more people live downtown. To simulate a process with a varying intensity $\lambda(\mathbf{s})$, start with a homogeneous process that is denser everywhere than the target, and then delete events with a probability that depends on location.

**The algorithm.** Choose $\lambda_{\max}$ with $\lambda(\mathbf{s}) \le \lambda_{\max}$ everywhere. Simulate a homogeneous pattern with intensity $\lambda_{\max}$. For each candidate $\mathbf{s}_i$ draw $U_i \sim \text{Uniform}(0, 1)$ and keep the candidate if $U_i < \lambda(\mathbf{s}_i)/\lambda_{\max}$. The retained points are an inhomogeneous Poisson process with intensity $\lambda(\mathbf{s})$.

**Why it works.** Let $p(\mathbf{s}) = \lambda(\mathbf{s})/\lambda_{\max}$ be the keep probability. Take a small pixel of area $a$ at $\mathbf{s}$. Candidates fall in it independently with probability $\lambda_{\max}a$, and each one survives independently with probability $p(\mathbf{s})$. A pixel therefore holds a retained event with probability $\lambda_{\max}a\,p(\mathbf{s}) = \lambda(\mathbf{s})a$, independently of the other pixels. That is the Bernoulli-grid description of Section 10.2.1 with a position-dependent success probability. Taking the same limit gives, for any region $B$,

$$
N(B) \sim \text{Poisson}\left(\int_B \lambda(\mathbf{s})\,d\mathbf{s}\right)
$$

with independent counts in disjoint regions. This is the **inhomogeneous Poisson process**, and its definition is Section 10.2.1's with $\lambda|B|$ replaced by an integral.

**A running example.** Crashes occur at rate $600$ per unit area on the highway $y = 0.3 + 0.4x$, falling off with the distance $\operatorname{dist}(\mathbf{s})$ to the highway as

$$
\lambda(\mathbf{s}) = 600\exp\left\{-6\operatorname{dist}(\mathbf{s})\right\}
$$

so $\lambda_{\max} = 600$. The functions below define the crash intensity and the two other intensity functions that the rest of Part IV uses: a population density with a downtown bump, and a relative risk around an industrial plant. (Chapter 11 uses a richer crash model, with road distance and population density as covariates and about 400 expected crashes, and simulates its own pattern; the crash pattern of this chapter is not reused there.)

The first part of the code is a hand-sized example. Eight candidates are drawn at rate 10, and the table shows each candidate's distance to the highway, its keep probability $p = e^{-6\operatorname{dist}}$, its uniform draw $U$, and the verdict. The second part checks the theory: 500 thinned patterns at full scale should have an average count equal to $\int\lambda(\mathbf{s})\,d\mathbf{s}$, and should have about as many events near the highway as the integral over that region predicts.

```python
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


def thin(lam_fun, lam_max, rng):
    """Lewis-Shedler thinning: an inhomogeneous Poisson process."""
    cand = sim_poisson(lam_max, rng)
    keep = rng.uniform(size=len(cand)) < lam_fun(cand) / lam_max
    return cand[keep]


def lam_crash(P):
    """Crash intensity: highest on the highway, decaying away from it."""
    return 600.0 * np.exp(-6.0 * road_distance(P))


# a tiny example: 8 candidates at rate 10
rng = np.random.default_rng(2081)
cand = sim_poisson(10, rng)
p_keep = np.exp(-6.0 * road_distance(cand))
u = rng.uniform(size=len(cand))
print(" cand      x      y   road dist   p_keep      U   kept")
for i in range(len(cand)):
    print(f"{i + 1:5d} {cand[i, 0]:6.3f} {cand[i, 1]:6.3f} "
          f"{road_distance(cand[i:i + 1])[0]:10.3f} {p_keep[i]:8.3f} "
          f"{u[i]:6.3f}   {'yes' if u[i] < p_keep[i] else 'no'}")

# check the counts against the integral of the intensity
g = (np.arange(200) + 0.5) / 200
GG = np.column_stack([np.repeat(g, 200), np.tile(g, 200)])
d_g = road_distance(GG)
near, far = d_g < 0.1, d_g > 0.4
expected = {"all": lam_crash(GG).mean(),
            "near": (lam_crash(GG) * near).mean(),
            "far": (lam_crash(GG) * far).mean()}
rng = np.random.default_rng(2082)
sims = [thin(lam_crash, 600.0, rng) for _ in range(500)]
obs = {"all": np.mean([len(s) for s in sims]),
       "near": np.mean([(road_distance(s) < 0.1).sum() for s in sims]),
       "far": np.mean([(road_distance(s) > 0.4).sum() for s in sims])}
for key in ("all", "near", "far"):
    print(f"{key:5s} expected {expected[key]:7.2f}   simulated "
          f"{obs[key]:7.2f}")
```

In the hand-sized table, candidates 4, 5, and 6 are kept because their uniform draws (0.093, 0.661, 0.655) fall below their keep probabilities (0.696, 0.878, 0.691). Candidate 1 is close to the highway, with $p = 0.529$, and is deleted anyway because $U = 0.844$ is large; thinning is random, and a high keep probability is not a guarantee.

The full-scale check agrees with the integral. The expected number of crashes is 199.19 and the average over 500 simulations is 200.30. Within 0.1 of the highway the expected count is 97.38 and the simulated average 97.72; farther than 0.4 it is 6.33 against 6.34.

### 10.3.3 Why Simulation Comes First

The rest of Part IV follows one pattern again and again. We compute a summary of the data, simulate many patterns from a null model, compute the same summary on each, and see where the data fall among them. Nearly every distribution in point pattern analysis is either unknown or too hard to derive, once edge effects and estimated intensities are involved, and simulation is the replacement. The logic is exactly that of the permutation tests in Chapter 6. If the data came from the null model, then the data and the simulated patterns are exchangeable, and the rank of the data among them is uniform. Section 10.9 builds tests on that fact.

### 10.3.4 Two Patterns That Are Not Random

To have something to detect, we need patterns with structure. Chapter 12 derives two standard models, and here we use short simulators that you can read in a minute. Each is explained in a paragraph, and neither is needed for the rest of the chapter beyond the patterns they produce.

- **A clustered pattern (Thomas).** Scatter parent locations at random. Around each parent, place a Poisson number of offspring whose displacements from the parent are independent Gaussian. Keep only the offspring inside the window. The result has groups of events, and the parents are never observed. This is how immune cells might gather around tumor cells. Parents are generated on a margin around the window, so that clusters centered just outside it still contribute their inside members.
- **An inhibited pattern (Matérn Type II).** Scatter many events at random, give each a random age, and delete an event if an older event lies within a minimum spacing of it. The survivors are never closer than the spacing, a model for cells that exclude one another.

The parameters are chosen so that all three patterns have close to 200 events: the CSR pattern has intensity 200; the clustered pattern has 20 parents per unit area, 10 offspring per parent, and a spread of 0.03; the inhibited pattern starts from 300 events per unit area and enforces a minimum spacing of 0.03.

```python
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dist(P, Q=None):
    """Matrix of Euclidean distances between rows of P and rows of Q."""
    Q = P if Q is None else Q
    d = np.sqrt(((P[:, None, :] - Q[None, :, :])**2).sum(axis=2))
    if Q is P:
        np.fill_diagonal(d, 0.0)
    return d


def sim_thomas(kappa, mu_c, sigma_c, rng):
    """Clusters: Poisson parents, Gaussian offspring (see Chapter 12)."""
    pad = 4 * sigma_c
    n_par = rng.poisson(kappa * (1 + 2 * pad)**2)
    parents = rng.uniform(-pad, 1 + pad, size=(n_par, 2))
    centers = np.repeat(parents, rng.poisson(mu_c, size=n_par), axis=0)
    pts = centers + sigma_c * rng.standard_normal(centers.shape)
    return pts[np.all((pts >= 0) & (pts <= 1), axis=1)]


def sim_matern2(lam_parent, r0, rng):
    """Inhibition: delete a point if an older point is within r0."""
    P = sim_poisson(lam_parent, rng)
    age = rng.uniform(size=len(P))
    D = dist(P)
    older = age[None, :] < age[:, None]
    return P[~((D < r0) & older).any(axis=1)]


rng = np.random.default_rng(2080)
csr = sim_poisson(200, rng)
clustered = sim_thomas(20, 10, 0.03, rng)
inhibited = sim_matern2(300, 0.03, rng)
patterns = {"random (CSR)": csr, "clustered": clustered,
            "inhibited": inhibited}
for name, P in patterns.items():
    x2, counts = quadrat_x2(P)
    p_val = 2 * min(stats.chi2.cdf(x2, 24), stats.chi2.sf(x2, 24))
    print(f"{name:13s} n={len(P):4d}  X^2={x2:7.2f}  "
          f"dispersion index={x2 / 24:5.2f}  two-sided p={p_val:.4f}")

os.makedirs("figures/ch10", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (name, P) in zip(axes, patterns.items()):
    ax.plot(P[:, 0], P[:, 1], "k.", ms=4)
    ax.set_title(f"{name}, n = {len(P)}")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/ch10/fig_10_1_three_patterns.png", dpi=120)
```

![Figure 10.1. Three simulated patterns: complete spatial randomness, a clustered pattern, and an inhibited pattern. All have about 200 events.](../figures/ch10/fig_10_1_three_patterns.png)

In Figure 10.1 the left panel scatters events with no visible organization, apart from the usual lumps and gaps that random points have. The middle panel shows tight groups with large empty areas between them. In the right panel the events are evenly spread, with no two touching and no large gaps.

Now the quadrat test. The random pattern has $X^2 = 38.74$, an index of dispersion of 1.61, and a two-sided p-value of 0.058: close to the 5% line, which a truly random pattern will sit near about one time in twenty. The clustered pattern has $X^2 = 215.00$ and an index of 8.96, so there is no doubt. The inhibited pattern has $X^2 = 12.91$ and an index of 0.54, with $p = 0.065$, and is not rejected.

That last result is the weakness promised in Section 10.2.3. The inhibited pattern is strikingly regular at the scale of its minimum spacing of 0.03, but the quadrats are $0.2 \times 0.2$. At that size, the regularity cancels out: whether two events are 0.03 or 0.003 apart makes no difference to which quadrat they fall in. The methods of Sections 10.5 and 10.6 look at the right scale, and find the inhibition decisively.

---

## 10.4 First-Order Exploration: Kernel Intensity Estimation

The first question about a pattern is usually where the events are dense. The standard answer is to smooth the events into a map.

### 10.4.1 The Estimator

Place a small bump, a **kernel**, at every event, and add the bumps up. With a Gaussian kernel of **bandwidth** $\psi$ the estimate at location $\mathbf{s}$ is

$$
\hat{\lambda}(\mathbf{s}) = \sum_{i=1}^n \mathcal{K}_\psi(\mathbf{s} - \mathbf{s}_i), \qquad \mathcal{K}_\psi(\mathbf{u}) = \frac{1}{2\pi\psi^2}\exp\left\{-\frac{\lVert\mathbf{u}\rVert^2}{2\psi^2}\right\}
$$

The kernel is a bivariate normal density with standard deviation $\psi$ in each coordinate, so it integrates to 1, and each event contributes one unit of intensity, spread over a neighborhood of size about $\psi$. This is the kernel density estimator you may know from one-dimensional statistics, in two dimensions, and multiplied by $n$ because we estimate an intensity (events per area) instead of a density (probability per area).

The computation is a matrix operation. Evaluate $\hat{\lambda}$ on a grid of $n_g$ points and form the $n_g \times n$ matrix of distances between grid points and events, with a vectorized `dist` function like that of Chapter 1, which here also accepts two sets of points. The kernel is applied entrywise and the rows are summed. (Chapter 11 uses a simpler `kernel_intensity` without the edge correction and with a different argument order, because it uses the estimate only far from the border.)

**Bias and variance.** What does $\hat{\lambda}(\mathbf{s})$ estimate? For a Poisson process, the expected sum of a function $f$ over the events is $\int f(\mathbf{u})\lambda(\mathbf{u})\,d\mathbf{u}$ and its variance is $\int f(\mathbf{u})^2\lambda(\mathbf{u})\,d\mathbf{u}$. This follows from the Bernoulli-grid picture of Section 10.2.1: each pixel contributes $f$ with probability $\lambda a$, so its mean is $f\lambda a$ and, for small $a$, its variance is nearly $f^2\lambda a$. Applying these with $f(\mathbf{u}) = \mathcal{K}_\psi(\mathbf{s} - \mathbf{u})$ gives

$$
\operatorname{E}\left[\hat{\lambda}(\mathbf{s})\right] = \int\mathcal{K}_\psi(\mathbf{s} - \mathbf{u})\lambda(\mathbf{u})\,d\mathbf{u}, \qquad \operatorname{Var}\left(\hat{\lambda}(\mathbf{s})\right) = \int\mathcal{K}_\psi(\mathbf{s} - \mathbf{u})^2\lambda(\mathbf{u})\,d\mathbf{u}
$$

The mean is the true intensity smoothed with the kernel, so $\hat{\lambda}$ is biased unless $\lambda$ is constant. For small $\psi$ a Taylor expansion of $\lambda$ around $\mathbf{s}$ gives

$$
\operatorname{E}\left[\hat{\lambda}(\mathbf{s})\right] \approx \lambda(\mathbf{s}) + \frac{\psi^2}{2}\left(\frac{\partial^2\lambda}{\partial x^2} + \frac{\partial^2\lambda}{\partial y^2}\right)
$$

because the Gaussian kernel has variance $\psi^2$ in each coordinate and no cross term. Smoothing raises the intensity in valleys and lowers it on peaks. The variance, with $\lambda$ roughly constant over the kernel's reach and $\int\mathcal{K}_\psi^2 = 1/(4\pi\psi^2)$ for the bivariate Gaussian, is

$$
\operatorname{Var}\left(\hat{\lambda}(\mathbf{s})\right) \approx \frac{\lambda(\mathbf{s})}{4\pi\psi^2}
$$

Bias grows with $\psi^2$ and variance shrinks like $1/\psi^2$: the usual trade-off, now with a knob you can see. A small bandwidth gives a map that follows every event and is mostly noise. A large bandwidth gives a smooth map that washes out real features.

### 10.4.2 Edge Correction

Near the boundary, part of each kernel falls outside the window, where no events can be observed. Events near the edge therefore add less than a unit of intensity inside the window, and $\hat{\lambda}$ is biased downward there, even when $\lambda$ is constant.

The fix follows from the mean formula. If $\lambda$ is constant, then $\operatorname{E}[\hat{\lambda}(\mathbf{s})] = \lambda\int_{\mathcal{D}}\mathcal{K}_\psi(\mathbf{s} - \mathbf{u})\,d\mathbf{u} = \lambda\,\mathcal{E}_\psi(\mathbf{s})$, where

$$
\mathcal{E}_\psi(\mathbf{s}) = \int_{\mathcal{D}}\mathcal{K}_\psi(\mathbf{s} - \mathbf{u})\,d\mathbf{u}
$$

is the kernel mass inside the window. For the unit square and a Gaussian kernel the integral factors into one-dimensional normal probabilities,

$$
\mathcal{E}_\psi(\mathbf{s}) = \left[\Phi\left(\frac{1 - s_1}{\psi}\right) - \Phi\left(\frac{-s_1}{\psi}\right)\right]\left[\Phi\left(\frac{1 - s_2}{\psi}\right) - \Phi\left(\frac{-s_2}{\psi}\right)\right]
$$

with $\Phi$ the standard normal distribution function. Dividing by $\mathcal{E}_\psi(\mathbf{s})$ removes the bias for constant $\lambda$:

$$
\hat{\lambda}_{\text{edge}}(\mathbf{s}) = \frac{1}{\mathcal{E}_\psi(\mathbf{s})}\sum_{i=1}^n\mathcal{K}_\psi(\mathbf{s} - \mathbf{s}_i)
$$

For varying $\lambda$ it is still biased, as the smoothing argument above says, but the edge-specific loss is gone.

### 10.4.3 Choosing the Bandwidth

How should $\psi$ be chosen? Three approaches are in common use.

- **Rules of thumb.** A normal-reference rule for two dimensions sets $\psi = \hat{\sigma}n^{-1/6}$, with $\hat{\sigma}$ a typical standard deviation of the coordinates. It is quick, and it assumes the intensity is not much more complicated than a single Gaussian bump.
- **Minimizing a mean squared error.** Adding the squared bias and the variance above, the integrated mean squared error is large for small $\psi$ (variance) and for large $\psi$ (bias), and has a minimum between. It needs the unknown $\lambda$, so it can only be used when the truth is known, as in simulations.
- **Likelihood cross-validation.** Chapter 11 derives the log-likelihood of an inhomogeneous Poisson process with intensity $\lambda$: $\sum_i\log\lambda(\mathbf{s}_i) - \int_{\mathcal{D}}\lambda(\mathbf{s})\,d\mathbf{s}$. Evaluate it at $\hat{\lambda}_\psi$, but compute each event's term from the estimate with that event left out, so that the estimate does not reward itself for following the event:

  $$
  \operatorname{LCV}(\psi) = \sum_{i=1}^n\log\hat{\lambda}_{\psi,-i}(\mathbf{s}_i) - \int_{\mathcal{D}}\hat{\lambda}_\psi(\mathbf{s})\,d\mathbf{s}
  $$

  and choose the $\psi$ that maximizes it. In code, the leave-one-out estimate is the row sum of the event-to-event kernel matrix with its diagonal set to zero, and the integral is a mean over a grid.

The code first checks two claims by simulation, using homogeneous patterns of intensity 200 and $\psi = 0.05$: the variance formula at an interior point, and the edge correction at a point 0.02 from the boundary. It then simulates the crash pattern of Section 10.3.2, evaluates the estimate over a range of bandwidths, and reports the rule of thumb, the likelihood cross-validation choice, and the choice that minimizes the integrated squared error against the true intensity.

```python
from scipy.stats import norm


def edge_mass(pts, psi):
    """Kernel mass inside the unit square for kernels centered at pts."""
    mx = norm.cdf((1 - pts[:, 0]) / psi) - norm.cdf(-pts[:, 0] / psi)
    my = norm.cdf((1 - pts[:, 1]) / psi) - norm.cdf(-pts[:, 1] / psi)
    return mx * my


def kernel_intensity(P, psi, pts, edge=True, leave_out=False):
    """Gaussian kernel estimate of lambda(s) at pts from events P."""
    K = np.exp(-dist(pts, P)**2 / (2 * psi**2)) / (2 * np.pi * psi**2)
    if leave_out:
        np.fill_diagonal(K, 0.0)
    lam_hat = K.sum(axis=1)
    return lam_hat / edge_mass(pts, psi) if edge else lam_hat


# variance and edge correction, known-truth checks (homogeneous, rate 200)
rng = np.random.default_rng(2083)
psi = 0.05
mid, edge_pt = np.array([[0.5, 0.5]]), np.array([[0.02, 0.5]])
est = np.array([[kernel_intensity(P, psi, mid)[0],
                 kernel_intensity(P, psi, edge_pt, edge=False)[0],
                 kernel_intensity(P, psi, edge_pt)[0]]
                for P in (sim_poisson(200, rng) for _ in range(3000))])
print(f"interior: mean={est[:, 0].mean():.1f} (truth 200), "
      f"variance={est[:, 0].var():.0f} "
      f"(formula {200 / (4 * np.pi * psi**2):.0f})")
print(f"edge point, no correction: mean={est[:, 1].mean():.1f}; "
      f"kernel mass inside={edge_mass(edge_pt, psi)[0]:.3f} "
      f"-> 200 x mass = {200 * edge_mass(edge_pt, psi)[0]:.1f}")
print(f"edge point, corrected:     mean={est[:, 2].mean():.1f}")

# crashes along the highway: bandwidth, bias and variance
rng = np.random.default_rng(2084)
crashes = thin(lam_crash, 600.0, rng)
g50 = (np.arange(50) + 0.5) / 50
G50 = np.column_stack([np.repeat(g50, 50), np.tile(g50, 50)])
truth50 = lam_crash(G50)
psi_grid = np.geomspace(0.01, 0.3, 30)
ise, lcv = [], []
for ps in psi_grid:
    lam_g = kernel_intensity(crashes, ps, G50)
    ise.append(np.mean((lam_g - truth50)**2))
    loo = kernel_intensity(crashes, ps, crashes, leave_out=True)
    lcv.append(np.log(loo).sum() - lam_g.mean())
ise, lcv = np.array(ise), np.array(lcv)
scott = crashes.std(axis=0).mean() * len(crashes)**(-1 / 6)
psi_lcv, psi_ise = psi_grid[lcv.argmax()], psi_grid[ise.argmin()]
print(f"{len(crashes)} crashes; expected {lam_crash(GG).mean():.1f}")
print(f"bandwidth: Scott rule {scott:.3f}, likelihood CV {psi_lcv:.3f}, "
      f"minimum ISE {psi_ise:.3f}")
for ps in (0.02, psi_lcv, 0.2):
    k = np.abs(psi_grid - ps).argmin()
    print(f"  psi={psi_grid[k]:.3f}: integrated squared error={ise[k]:9.1f}")
```

Both claims hold. At the interior point the estimate averages 198.1 against a truth of 200, and its variance is 6,379 against the formula's 6,366. At the edge point the uncorrected estimate averages 129.5; the kernel mass inside the window is 0.655, so $200 \times 0.655 = 131.1$ is what the theory predicts. After the correction the average is 197.6.

The crash pattern has 212 events against an expected 199.2. The normal-reference rule gives $\psi = 0.102$, likelihood cross-validation 0.093, and the bandwidth that minimizes the integrated squared error against the truth is 0.083. The three agree to within about 20%. The cost of a bad choice is large: the integrated squared error is 41,097 at $\psi = 0.02$, 4,325 at the cross-validated value, and 12,252 at $\psi = 0.211$.

![Figure 10.2. Kernel estimates of the crash intensity. Top row: the true intensity, the 212 crashes with the highway in red, and the estimate with a bandwidth of 0.02. Bottom row: the estimate at the cross-validated bandwidth 0.093, the estimate at 0.2, and the cross-validation score and integrated squared error against bandwidth.](../figures/ch10/fig_10_2_kernel.png)

Figure 10.2 shows the trade-off. At $\psi = 0.02$ the map is a field of separate bumps, one per event. At $\psi = 0.2$ the highway ridge has disappeared into a gentle gradient. At the cross-validated value the ridge is visible but lumpy: with 212 events, the data limit how well any bandwidth can recover the true ridge. The bottom right panel shows that the cross-validation score peaks close to where the error against the truth is smallest, even though cross-validation never saw the truth.

---

## 10.5 Distance Methods

The second-order question is whether events interact. The simplest evidence is distances: how far is each event from its nearest neighbor, and how far is an arbitrary location from the nearest event?

### 10.5.1 Nearest-Neighbor Distances

With the distance matrix $\mathbf{H}$ of Chapter 1, whose entry $h_{ij}$ is the distance between events $i$ and $j$, the **nearest-neighbor distance** of event $i$ is

$$
d_i^{\text{nn}} = \min_{j \ne i}h_{ij}
$$

the smallest off-diagonal entry in row $i$. Their empirical distribution function,

$$
\hat{G}(r) = \frac{1}{n}\sum_{i=1}^n\mathbb{1}\left(d_i^{\text{nn}} \le r\right)
$$

is the share of events that have a neighbor within distance $r$. Here $\mathbb{1}(\cdot)$ is the indicator of a condition: 1 if it holds, 0 if not. The target of $\hat{G}$ is the function $G(r)$, the probability that a typical event has its nearest neighbor within distance $r$.

**A hand-sized example.** The four events from Section 10.2.3 have the distance matrix printed below, with $h_{12} = 0.5$ the hypotenuse of a $0.3$-$0.4$-$0.5$ right triangle. The nearest-neighbor distances are the row minima, ignoring the diagonal zeros.

```python
H4 = dist(P4)
print("coordinates:\n", P4)
print("distance matrix H:\n", np.round(H4, 3))
H4_inf = H4 + np.diag(np.full(4, np.inf))
d_nn = H4_inf.min(axis=1)
print("nearest-neighbor distances:", np.round(d_nn, 3),
      " nearest:", H4_inf.argmin(axis=1) + 1)
for r in (0.40, 0.45, 0.50, 0.55, 0.60):
    print(f"G-hat({r:.2f}) = {(d_nn <= r + 1e-9).mean():.2f}", end="   ")
print()
lam4 = 4.0
print("CSR with lambda = 4 would give G(0.45) =",
      round(1 - np.exp(-lam4 * np.pi * 0.45**2), 3))
```

Events 2 and 4 are each other's nearest neighbors at distance 0.447; event 1's nearest neighbor is event 2 at 0.5; event 3's is event 2 at 0.583. So $\hat{G}$ is 0 below 0.447, jumps to $0.50$ at 0.447, to $0.75$ at 0.5, and to 1 at 0.583. A CSR pattern with four events in a unit square would have $G(0.45) = 1 - e^{-4\pi(0.45)^2} = 0.922$, so these four events have unusually distant neighbors. With $n = 4$ nothing is statistically meaningful, but the example is small enough to check every entry by hand.

### 10.5.2 The Nearest-Neighbor Distribution Under CSR

Under CSR the function $G$ has a closed form. Pick a typical event at $\mathbf{s}$, and ask for the probability that no other event lies within distance $r$. In the Bernoulli-grid picture of Section 10.2.1, the pixel holding the typical event has no effect on the other pixels, because pixels are independent. So the other events still form a homogeneous Poisson process, and the number of them in the disc $b(\mathbf{s}, r)$ of radius $r$ about $\mathbf{s}$ is Poisson with mean $\lambda\pi r^2$. The probability that the disc is empty, the **void probability**, is $e^{-\lambda\pi r^2}$, and the nearest neighbor is farther than $r$ exactly when the disc is empty:

$$
1 - G(r) = \Pr\left(N(b(\mathbf{s}, r)) = 0\right) = \exp\left\{-\lambda\pi r^2\right\}, \qquad G(r) = 1 - e^{-\lambda\pi r^2}
$$

As a by-product, the expected nearest-neighbor distance is $\int_0^\infty e^{-\lambda\pi r^2}\,dr = 1/(2\sqrt{\lambda})$. With 200 events per unit area that is 0.035, about the size of the minimum spacing of 0.03 used for the inhibited pattern in Section 10.3.4.

Clustered patterns have more short distances than this and inhibited patterns fewer, so $\hat{G}$ above the CSR curve signals clustering, and $\hat{G}$ below it signals inhibition.

### 10.5.3 The Empty-Space Function

The other distance question is asked from an arbitrary location in the window, not from an event: how far is the nearest event? Let $F(r)$ be the probability that a randomly chosen location has an event within distance $r$. The empirical version uses a regular grid of $n_g$ test points $\mathbf{q}_k$:

$$
\hat{F}(r) = \frac{1}{n_g}\sum_{k=1}^{n_g}\mathbb{1}\left(\min_i\lVert\mathbf{q}_k - \mathbf{s}_i\rVert \le r\right)
$$

It is computed from the grid-to-event distance matrix, the same matrix as for kernel smoothing. Under CSR, a disc around an arbitrary location is empty with probability $e^{-\lambda\pi r^2}$, so $F(r) = 1 - e^{-\lambda\pi r^2}$, the same function as $G(r)$.

The two functions differ for patterns with structure, and in opposite directions. In a clustered pattern, events have close neighbors, so $G$ rises quickly, but most of the window is empty space far from any cluster, so $F$ rises slowly. In an inhibited pattern, events are kept apart, so $G$ rises slowly, while the events are spread evenly and every location is near one, so $F$ rises quickly.

### 10.5.4 The J-Function

The two distances can be combined into one summary, the $J$-function:

$$
J(r) = \frac{1 - G(r)}{1 - F(r)}
$$

a ratio of survival functions. Under CSR, $G = F$, so $J(r) = 1$ for every $r$. Clustering makes $1 - G$ small and $1 - F$ large, so $J < 1$. Inhibition does the opposite, and $J > 1$. One curve, with one reference line.

The code computes $\hat{G}$, $\hat{F}$, and $\hat{J}$ at three distances for the three patterns of Figure 10.1, using the plain empirical functions (Section 10.8 corrects them for the window edge). The last column of the table is $1 - e^{-n\pi r^2}$, using the pattern's event count as the intensity.

```python
def g_naive(P, r):
    """Empirical nearest-neighbor distribution function (no edge fix)."""
    H = dist(P)
    np.fill_diagonal(H, np.inf)
    return (H.min(axis=1)[:, None] <= r[None, :]).mean(axis=0)


GRID = (np.arange(40) + 0.5) / 40
GRID = np.column_stack([np.repeat(GRID, 40), np.tile(GRID, 40)])


def f_naive(P, r):
    """Empirical empty-space function from a 40 x 40 grid of points."""
    d = dist(GRID, P).min(axis=1)
    return (d[:, None] <= r[None, :]).mean(axis=0)


r_vals = np.array([0.02, 0.04, 0.06])
print("pattern          r     G-hat   F-hat   J-hat   CSR value")
for name, P in patterns.items():
    lam_hat = len(P)
    G_, F_ = g_naive(P, r_vals), f_naive(P, r_vals)
    for k, r in enumerate(r_vals):
        J_ = (1 - G_[k]) / (1 - F_[k])
        print(f"{name:13s} {r:6.2f} {G_[k]:8.3f} {F_[k]:7.3f} {J_:7.3f}"
              f"   {1 - np.exp(-lam_hat * np.pi * r**2):.3f}")
```

| Pattern | $r$ | $\hat{G}$ | $\hat{F}$ | $\hat{J}$ | CSR value of $G$ and $F$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| random | 0.02 | 0.179 | 0.204 | 1.031 | 0.201 |
| random | 0.04 | 0.542 | 0.589 | 1.116 | 0.593 |
| random | 0.06 | 0.866 | 0.853 | 0.913 | 0.868 |
| clustered | 0.02 | 0.679 | 0.139 | 0.373 | 0.212 |
| clustered | 0.04 | 0.921 | 0.279 | 0.109 | 0.615 |
| clustered | 0.06 | 0.984 | 0.399 | 0.026 | 0.883 |
| inhibited | 0.02 | 0.000 | 0.255 | 1.342 | 0.230 |
| inhibited | 0.04 | 0.399 | 0.716 | 2.113 | 0.648 |
| inhibited | 0.06 | 0.841 | 0.948 | 3.022 | 0.905 |

The random pattern tracks the CSR values and has $\hat{J}$ near 1. The clustered pattern has $\hat{G}$ far above and $\hat{F}$ far below the CSR value, and $\hat{J}$ falls to 0.026 by $r = 0.06$. The inhibited pattern has $\hat{G} = 0$ at $r = 0.02$, because no two events are closer than 0.03, and $\hat{J}$ rises to 3.0. Compare this with the quadrat test, which could not see the inhibition at all.

---

## 10.6 Second-Order Summaries

Distances to the nearest neighbor look at only one other event per event. A fuller description counts all the neighbors within a range of distances, which is what the $K$-function does.

### 10.6.1 Ripley's K-Function

Let $\lambda$ be the intensity of a pattern whose intensity is constant. The **$K$-function** is defined by

$$
\lambda K(r) = \operatorname{E}\left[\text{number of additional events within distance } r \text{ of a typical event}\right]
$$

so $K(r)$ is that expected count divided by the intensity, a quantity with the units of area. The division by $\lambda$ makes the function comparable across patterns with different numbers of events.

**Under CSR.** A typical event leaves the other events unchanged, by the argument of Section 10.5.2: they form a Poisson process of intensity $\lambda$. The number of them in a disc of radius $r$ is Poisson with mean $\lambda\pi r^2$. So $\lambda K(r) = \lambda\pi r^2$ and

$$
K(r) = \pi r^2
$$

the area of the disc. The reference curve is a parabola with no parameters. A pattern with more neighbors than that within distance $r$ has $K(r) > \pi r^2$ (clustering at that scale), and one with fewer has $K(r) < \pi r^2$ (inhibition).

### 10.6.2 The Estimator, and Edge Weights

A natural estimate counts ordered pairs of events within distance $r$, with the indicator matrix $\mathbb{1}(\mathbf{H} \le r)$, and divides by $n$ events and by an estimate of the intensity. Writing $\lambda\approx n/\operatorname{area}(\mathcal{D})$, the average number of extra neighbors per event is $\frac{1}{n}\sum_i\sum_{j \ne i}\mathbb{1}(h_{ij} \le r)$, and dividing by $\hat{\lambda}$ gives $\frac{\operatorname{area}(\mathcal{D})}{n^2}\sum_i\sum_{j \ne i}\mathbb{1}(h_{ij} \le r)$. Replacing $n^2$ by $n(n-1)$ removes a small bias, since for a Poisson number of events $\operatorname{E}[n(n-1)] = \lambda^2\operatorname{area}(\mathcal{D})^2$, so that $n(n-1)/\operatorname{area}(\mathcal{D})^2$ is an unbiased estimate of $\lambda^2$. Adding a weight $\omega_{ij} \ge 1$ for each pair to compensate for the window edge (the subject of Section 10.8) gives

$$
\hat{K}(r) = \frac{\operatorname{area}(\mathcal{D})}{n(n-1)}\sum_{i=1}^n\sum_{j \ne i}\omega_{ij}\,\mathbb{1}\left(h_{ij} \le r\right)
$$

The idea behind $\omega_{ij}$ is this. Under CSR, if $j$ is at distance $h_{ij}$ from event $i$, it is equally likely to lie anywhere on the circle of radius $h_{ij}$ around $\mathbf{s}_i$. Part of that circle may lie outside the window, in which case a neighbor there would never be seen. Let $f_{ij}$ be the fraction of the circle that lies inside the window. A pair at distance $h_{ij}$ is observed with probability $f_{ij}$, so counting it with weight $\omega_{ij} = 1/f_{ij}$ restores the expected count. Written in terms of arc lengths,

$$
\omega_{ij} = \frac{2\pi h_{ij}}{\text{length of the part of the circle } \partial b(\mathbf{s}_i, h_{ij}) \text{ inside } \mathcal{D}}
$$

Section 10.8.3 derives the arc length for a rectangle. The code below implements it exactly (the function `arc_inside`) and builds the estimator from the weights.

**A hand-sized example.** With the four events and $r = 0.55$, the ordered pairs within $0.55$ are $(1,2)$, $(2,1)$, $(2,4)$, and $(4,2)$. Event 1 sits at $(0.1, 0.1)$, near a corner, so the circle of radius $0.5$ around it has only 31.4% of its length inside the window and the weight is $1/0.3141 = 3.184$. The circle of the same radius around event 2 at $(0.4, 0.5)$ is cut only by the left edge, leaving 79.5% inside, and the weight is 1.258. The pair $(2,4)$ has radius $0.447$ around event 2, and $(4,2)$ the same radius around event 4, which sits close to the top edge. The four weights are 3.184, 1.258, 1.173, and 2.164, and they sum to 7.779. Then

$$
\hat{K}(0.55) = \frac{1}{4 \times 3}(7.779) = 0.6482
$$

without edge correction the count would be $4/12 = 0.333$, and the CSR value is $\pi(0.55)^2 = 0.950$. With four events the estimate is far too noisy to interpret, and the point of the exercise is the arithmetic.

```python
def arc_inside(P, h):
    """Fraction of the circle of radius h about each row of P that lies
    inside the unit square (exact: edge arcs minus corner overlaps)."""
    h = np.asarray(h, dtype=float)

    def half(a):
        """Half-angle of the arc cut off by an edge at distance a."""
        return np.where(a < h, np.arccos(np.clip(a / h, -1.0, 1.0)), 0.0)

    left, right = half(P[:, 0]), half(1 - P[:, 0])
    bottom, top = half(P[:, 1]), half(1 - P[:, 1])
    cut = 2 * (left + right + bottom + top)
    for a, b in ((left, bottom), (left, top), (right, bottom), (right, top)):
        cut = cut - np.maximum(0.0, a + b - np.pi / 2) * (a > 0) * (b > 0)
    return 1 - cut / (2 * np.pi)


def pair_weights(P, Q=None):
    """Distances and Ripley edge weights omega_ij (circle about P[i])."""
    same = Q is None
    Q = P if same else Q
    H = dist(P, Q)
    W = np.zeros_like(H)
    use = ~np.eye(len(P), dtype=bool) if same else np.ones_like(H, bool)
    ii, jj = np.where(use)
    W[ii, jj] = 1.0 / np.maximum(arc_inside(P[ii], H[ii, jj]), 1e-9)
    return H, W, use


def k_ripley(P, r):
    """Edge-corrected K-hat on the unit square (area 1)."""
    H, W, use = pair_weights(P)
    order = np.argsort(H[use])
    h, cw = H[use][order], np.cumsum(W[use][order])
    tot = np.concatenate([[0.0], cw])[np.searchsorted(h, r, side="right")]
    return tot / (len(P) * (len(P) - 1))


# the four points by hand: K-hat at r = 0.55
H4, W4, use4 = pair_weights(P4)
r = 0.55
print("ordered pairs with h <= 0.55: (i, j), h_ij, arc inside, omega")
total = 0.0
for i in range(4):
    for j in range(4):
        if i != j and H4[i, j] <= r:
            arc = arc_inside(P4[i:i + 1], H4[i, j])[0]
            total += W4[i, j]
            print(f"  ({i + 1},{j + 1})  h={H4[i, j]:.3f}  arc={arc:.4f}  "
                  f"omega={W4[i, j]:.3f}")
print(f"sum of weights = {total:.3f}; K-hat = {total:.3f}/12 = "
      f"{total / 12:.4f}; no edge correction: {4 / 12:.4f}; "
      f"pi r^2 = {np.pi * r**2:.4f}")
print("check with k_ripley:", np.round(k_ripley(P4, np.array([r])), 4))

# L removes the dependence of the variance on r (CSR, rate 200)
rng = np.random.default_rng(2085)
r_v = np.array([0.05, 0.10, 0.15, 0.20])
K_sim = np.array([k_ripley(sim_poisson(200, rng), r_v)
                  for _ in range(1000)])
L_sim = np.sqrt(K_sim / np.pi) - r_v
print("r            ", r_v)
print("mean K / pi r^2", np.round(K_sim.mean(axis=0) / (np.pi * r_v**2), 3))
print("sd of K-hat   ", np.round(K_sim.std(axis=0), 5))
print("2 pi r^2/n^2  ", np.round(np.sqrt(2 * np.pi * r_v**2 / 200**2), 5))
print("sd of L - r   ", np.round(L_sim.std(axis=0), 5),
      "(formula", round(np.sqrt(1 / (2 * np.pi * 200**2)), 5), ")")
```

The code reproduces $\hat{K}(0.55) = 0.6482$ from the vectorized function. The second part simulates 1,000 CSR patterns with a mean of 200 events and shows that the corrected estimator is unbiased: the mean of $\hat{K}(r)/(\pi r^2)$ is between 1.000 and 1.003 at all four distances.

### 10.6.3 Besag's L and the Variance of K

The $K$-function is a cumulative count, and its variance under CSR grows with $r$, so deviations from $\pi r^2$ at large $r$ look bigger than they are. Besag's **$L$-function**

$$
L(r) = \sqrt{\frac{K(r)}{\pi}}
$$

equals $r$ under CSR, and is customarily plotted as $L(r) - r$, so that CSR is the horizontal line at zero. It also stabilizes the variance. Under CSR, the number of ordered pairs within $r$ is roughly $2\times$ a count of unordered pairs, each of which is within $r$ with probability $\pi r^2/\operatorname{area}(\mathcal{D})$. Counting that as approximately Poisson with variance equal to its mean, and ignoring edge effects and the dependence among pairs, the variance of $\hat{K}$ is

$$
\operatorname{Var}\left(\hat{K}(r)\right) \approx \frac{2\operatorname{area}(\mathcal{D})\pi r^2}{n^2}
$$

which grows like $r^2$. Write $\hat{K} = \pi r^2(1 + \delta)$ with $\delta$ a small relative error, so that $\operatorname{Var}(\delta) \approx 2\operatorname{area}(\mathcal{D})/(n^2\pi r^2)$. The delta method gives $\hat{L} = r\sqrt{1 + \delta} \approx r(1 + \delta/2)$ and

$$
\operatorname{Var}\left(\hat{L}(r)\right) \approx \frac{r^2}{4}\operatorname{Var}(\delta) = \frac{\operatorname{area}(\mathcal{D})}{2\pi n^2}
$$

which does not depend on $r$. In the simulation of the previous block, with $n = 200$ and area 1, the standard deviation of $\hat{K}$ is 0.00066, 0.00138, 0.00209, and 0.00304 at $r = 0.05, 0.10, 0.15, 0.20$, a fivefold increase; the formula $\sqrt{2\pi r^2/n^2}$ gives 0.00063, 0.00125, 0.00188, and 0.00251, which gets the growth right and the size roughly right, and slightly understates it at larger $r$ because pairs that share an event are dependent. The standard deviation of $\hat{L}(r) - r$ is 0.0021, 0.0022, 0.0022, and 0.0024, nearly constant, against a formula value of 0.0020. A plot of $L(r) - r$ can therefore be read with a single yardstick for the width of noise at every distance.

### 10.6.4 The Pair Correlation Function

The $K$-function is cumulative: $K(r)$ counts everything up to $r$, so a cluster at small distances shows up in $K$ at all larger distances too, which makes it hard to say at which scale the interaction happens. The **pair correlation function** is the non-cumulative version.

Consider the expected number of additional events in a thin ring of radii $r$ and $r + dr$ around a typical event. Under CSR it is $\lambda\cdot2\pi r\,dr$, the intensity times the area of the ring. Write the corresponding number for a general pattern as $\lambda g(r)\cdot2\pi r\,dr$; then $g(r)$ is the ratio of the actual neighbor density at distance $r$ to what CSR would give, and $g = 1$ means no interaction at that distance. Adding up the rings gives $\lambda K(r) = \int_0^r\lambda g(s)2\pi s\,ds$, so

$$
K(r) = \int_0^r 2\pi s\,g(s)\,ds, \qquad g(r) = \frac{K'(r)}{2\pi r}
$$

and $g > 1$ means attraction at distance $r$, $g < 1$ repulsion. Because $g$ is a derivative of $\hat{K}$, which is a step function, it has to be smoothed. The estimate replaces each indicator $\mathbb{1}(h_{ij} \le r)$ in $\hat{K}$ by a kernel centered at $h_{ij}$, with a small bandwidth $b$ in the distance direction, and divides by $2\pi r$:

$$
\hat{g}(r) = \frac{\operatorname{area}(\mathcal{D})}{n(n-1)}\sum_{i=1}^n\sum_{j \ne i}\omega_{ij}\frac{\mathcal{K}_b^{(1)}(r - h_{ij})}{2\pi r}
$$

with $\mathcal{K}_b^{(1)}(u) = e^{-u^2/(2b^2)}/(b\sqrt{2\pi})$ the one-dimensional version of the Gaussian kernel of Section 10.4.1, with standard deviation $b$. The estimate is unreliable for $r$ below about $2b$, where the kernel reaches below zero distance, so the code starts at $r = 0.02$ with $b = 0.01$.

As a known-truth check, Chapter 12 derives, for any parent intensity and spread, the pair correlation of the Thomas model; for the pattern of Section 10.3.4 (20 parents per unit area, spread 0.03) it is $g(r) = 1 + \exp\{-r^2/(4 \times 0.03^2)\}/(20 \times 4\pi \times 0.03^2)$, which equals 5.4 at $r = 0$ and decays to 1. The code averages $\hat{g}$ over 200 simulated patterns of each kind.

```python
def g_pcf(P, r, bw=0.01):
    """Pair correlation: kernel-smoothed weighted pair distances."""
    H, W, use = pair_weights(P)
    h, w = H[use], W[use]
    k = np.exp(-0.5 * ((r[:, None] - h[None, :]) / bw)**2)
    k = k / (bw * np.sqrt(2 * np.pi))
    n = len(P)
    return (k * w[None, :]).sum(axis=1) / (n * (n - 1) * 2 * np.pi * r)


r_g = np.array([0.02, 0.04, 0.06, 0.10, 0.15])
rng = np.random.default_rng(2086)
g_csr = np.mean([g_pcf(sim_poisson(200, rng), r_g)
                 for _ in range(200)], axis=0)
g_thom = np.mean([g_pcf(sim_thomas(20, 10, 0.03, rng), r_g)
                  for _ in range(200)], axis=0)
g_true = 1 + np.exp(-r_g**2 / (4 * 0.03**2)) / (20 * 4 * np.pi * 0.03**2)
print("r              ", r_g)
print("CSR, mean g-hat", np.round(g_csr, 3))
print("Thomas, mean   ", np.round(g_thom, 3))
print("Thomas, formula", np.round(g_true, 3))
```

For CSR the average $\hat{g}$ is 1.00 at every distance. For the Thomas pattern it is 4.87, 3.81, 2.64, 1.24, and 0.94 at $r = 0.02, 0.04, 0.06, 0.10, 0.15$, against formula values of 4.96, 3.84, 2.63, 1.28, and 1.01. The small gap at larger $r$ comes from clusters that straddle the window edge, which the edge weights treat only approximately, and from the margin of the cluster simulator.

---

## 10.7 Multitype Patterns

Many point patterns carry a label for each event. The labels may be a species, a disease status, or a result: a shot in basketball is made or missed, and a shot chart is a point pattern with a label for each shot. The multitype questions are whether the types occupy different places, and whether they attract or repel each other.

### 10.7.1 Marks, and Two Kinds of Question

Take two types. In the general discussion they are type 1 and type 2. In the case-control setting of Section 10.7.3, type 1 is cases, with intensity $\lambda_1(\mathbf{s})$, and type 2 is controls, with intensity $\lambda_0(\mathbf{s})$. Two questions look similar and are not.

- **Do the types interact?** If a case is here, is a control more or less likely to be nearby than for two independent patterns? This is a second-order question about pairs of different types. Section 10.7.2 answers it.
- **Do the types have different intensities?** Do cases cluster in places where controls do not? This is a first-order question about the ratio of intensities, the **relative risk**. (Here the relative risk is the intensity of cases divided by the intensity of controls. Chapter 9 used the same words for a county's mean count divided by its expected count.) Section 10.7.3 answers it.

The distinction is important in epidemiology. Cases are expected to sit where people live, which is where the controls are. Whether cases are more concentrated than the population is a question about the ratio $\lambda_1/\lambda_0$, not about their interaction.

### 10.7.2 Cross-Type Summaries

The cross-$K$ function is the multitype version of $K$: $\lambda_2K_{12}(r)$ is the expected number of type-2 events within distance $r$ of a typical type-1 event. Its estimator uses the same weights, applied to all pairs formed by one event of each type, with the circle taken around the type-1 event:

$$
\hat{K}_{12}(r) = \frac{\operatorname{area}(\mathcal{D})}{n_1n_2}\sum_{i \in \text{type 1}}\sum_{j \in \text{type 2}}\omega_{ij}\,\mathbb{1}\left(h_{ij} \le r\right)
$$

If the two patterns are independent and the type-2 pattern has constant intensity, a typical type-1 event learns nothing about type 2, so $K_{12}(r) = \pi r^2$. Values above that mean that type-2 events gather around type-1 events (attraction); values below mean that they avoid them (segregation). The matching nearest-neighbor version is $G_{12}(r)$, the probability that the nearest type-2 event to a typical type-1 event lies within $r$.

To see all three behaviors, the code generates 150 type-1 events and three different type-2 patterns: an independent random pattern; an attracted pattern, in which events are drawn from the neighborhoods of type-1 events with a spread of 0.03; and a segregated pattern, a random pattern with 250 events per unit area from which every event within 0.05 of a type-1 event is deleted. It reports $\hat{K}_{12}(0.1)/(\pi 0.1^2)$ for each.

```python
def k12_ripley(P1, P2, r):
    """Cross K-hat from type-1 points to type-2 points (area 1)."""
    H, W, use = pair_weights(P1, P2)
    return np.array([(W * (H <= rk)).sum() for rk in r]) / (len(P1) * len(P2))


rng = np.random.default_rng(2087)
type1 = sim_poisson(150, rng)
independent = sim_poisson(150, rng)
attract = type1[rng.integers(0, len(type1), 150)]
attract = attract + 0.03 * rng.standard_normal(attract.shape)
attract = attract[np.all((attract >= 0) & (attract <= 1), axis=1)]
cand = sim_poisson(250, rng)
segregate = cand[dist(cand, type1).min(axis=1) > 0.05]
r12 = np.array([0.10])
print("scenario        n2   K12(0.1) / (pi r^2)")
for name, P2 in (("independent", independent), ("attraction", attract),
                 ("segregation", segregate)):
    k12 = k12_ripley(type1, P2, r12)[0] / (np.pi * 0.1**2)
    print(f"{name:12s} {len(P2):4d} {k12:18.2f}")
```

The independent pattern gives a ratio of 1.00, the attracted pattern 1.23, and the segregated pattern 0.81, as expected.

### 10.7.3 The Spatial Relative Risk

The second question compares intensities. The **spatial relative risk** is

$$
\operatorname{RR}(\mathbf{s}) = \frac{\lambda_1(\mathbf{s})}{\lambda_0(\mathbf{s})}
$$

the ratio of the case and control intensities. Because it is a ratio, the factor that makes both intensities large where people live cancels, and what remains is the extra risk that cases carry compared with the population. A natural estimate is the ratio of two kernel estimates, $\widehat{\operatorname{RR}}(\mathbf{s}) = \hat{\lambda}_1(\mathbf{s})/\hat{\lambda}_0(\mathbf{s})$.

**The shared setting.** The code uses a case-control setting that the next two chapters also use. Controls form an inhomogeneous Poisson process with intensity $800\,\text{pop}(\mathbf{s})$, where $\text{pop}$ is the population density with a downtown bump. Cases have intensity $120\,\text{pop}(\mathbf{s})\,\text{ex}(\mathbf{s})$, where $\text{ex}$ is an excess risk that rises to $e \approx 2.72$ at an industrial plant at $(0.3, 0.6)$ and falls to 1 away from it. The true relative risk is therefore $0.15\,\text{ex}(\mathbf{s})$: population density cancels, and the baseline ratio 0.15 is 120/800.

**Why the two estimates should share a bandwidth.** The expected value of each kernel estimate is the intensity smoothed with the kernel, from Section 10.4.1. If the numerator and denominator are smoothed with the same $\psi$, the common population factor is smoothed the same way in both and mostly cancels. With different bandwidths it does not, and the shape of $\text{pop}$ leaks into the estimated risk. The block below checks this with no random noise at all, by computing the expected values of the estimators by numerical integration, and reporting the root mean squared error of $\log(\operatorname{E}[\widehat{\operatorname{RR}}]/\operatorname{RR})$ over a grid.

**The data.** The same block simulates the shared case-control data and a second data set with ten times the intensities, chooses a shared bandwidth by likelihood cross-validation on the pooled events, and compares estimates with the truth at several bandwidths. It also reports how many extra cases the plant creates.

```python
def case_control(scale, rng):
    """Cases and controls of the shared setting; scale multiplies both."""
    controls = thin(lambda P: scale * 800 * pop_density(P),
                    scale * 800.0, rng)
    cases = thin(lambda P: scale * 120 * pop_density(P) * excess_risk(P),
                 scale * 130.0, rng)
    return cases, controls


cases, controls = case_control(1, np.random.default_rng(2088))
print(f"{len(controls)} controls (expect "
      f"{800 * pop_density(GG).mean():.0f}), {len(cases)} cases (expect "
      f"{120 * (pop_density(GG) * excess_risk(GG)).mean():.0f})")
extra = 120 * (pop_density(GG) * (excess_risk(GG) - 1)).mean()
print(f"cases expected because of the plant: {extra:.1f}")

pooled = np.vstack([cases, controls])
psi_pool = np.geomspace(0.03, 0.4, 25)
lcv_pool = []
for ps in psi_pool:
    lam_g = kernel_intensity(pooled, ps, G50)
    loo = kernel_intensity(pooled, ps, pooled, leave_out=True)
    lcv_pool.append(np.log(loo).sum() - lam_g.mean())
psi_rr = psi_pool[int(np.argmax(lcv_pool))]
print(f"likelihood CV bandwidth for the pooled events: {psi_rr:.3f}")

# bias alone: expected values of the two estimators, no sampling noise
g30 = (np.arange(30) + 0.5) / 30
G30 = np.column_stack([np.repeat(g30, 30), np.tile(g30, 30)])
g80 = (np.arange(80) + 0.5) / 80
G80 = np.column_stack([np.repeat(g80, 80), np.tile(g80, 80)])


def expected_kernel(fun, psi, pts):
    """E[lambda-hat] = kernel-smoothed intensity / kernel mass inside."""
    K = np.exp(-dist(pts, G80)**2 / (2 * psi**2)) / (2 * np.pi * psi**2)
    return (K * fun(G80)[None, :]).mean(axis=1) / edge_mass(pts, psi)


lam0 = lambda P: 800 * pop_density(P)
lam1 = lambda P: 120 * pop_density(P) * excess_risk(P)
true30 = 0.15 * excess_risk(G30)
print("psi_cases  psi_controls   RMSE of log(E RR-hat / RR) (bias only)")
for p1, p0 in ((0.05, 0.05), (0.05, 0.20), (0.10, 0.10), (0.10, 0.03),
               (0.10, 0.30)):
    rr_e = expected_kernel(lam1, p1, G30) / expected_kernel(lam0, p0, G30)
    bias = np.sqrt(np.mean(np.log(rr_e / true30)**2))
    print(f"{p1:9.2f} {p0:12.2f} {bias:22.3f}")

# relative risk from data: the shared sample size and ten times as much
true_rr = 0.15 * excess_risk(G50)
at_plant = np.abs(G50 - np.array([0.3, 0.6])).sum(axis=1).argmin()
cases10, controls10 = case_control(10, np.random.default_rng(2098))
print(f"true RR at the plant {true_rr[at_plant]:.3f}; sizes: "
      f"{len(cases)}/{len(controls)} and {len(cases10)}/{len(controls10)}")
print("psi     RMSE log RR (x1)  RR at plant (x1)   "
      "RMSE log RR (x10)  RR at plant (x10)")
for ps in (0.03, 0.05, 0.083, psi_rr, 0.2):
    row = []
    for ca, co in ((cases, controls), (cases10, controls10)):
        rr = kernel_intensity(ca, ps, G50) / kernel_intensity(co, ps, G50)
        row += [np.sqrt(np.mean(np.log(rr / true_rr)**2)), rr[at_plant]]
    print(f"{ps:.3f} {row[0]:14.3f} {row[1]:17.3f} {row[2]:18.3f} "
          f"{row[3]:17.3f}")
```

With the bandwidths equal the bias is small: $\psi = 0.05$ for both gives an RMSE of 0.025 on the log scale and $\psi = 0.10$ for both gives 0.072. With unequal bandwidths it is larger: 0.143 for $(0.05, 0.20)$, 0.092 for $(0.10, 0.03)$, and 0.194 for $(0.10, 0.30)$. Equal bandwidths are the right default, even though neither is the best bandwidth for its own estimate.

The data set has 349 controls (363 expected) and 46 cases (58 expected). The more important number is how many of those cases are caused by the plant. The excess is $120\int\text{pop}(\text{ex} - 1) = 4.0$ cases. Out of 58 cases, four exist because of the plant, so the signal is faint. The relative risk table shows what that means in practice:

| $\psi$ | RMSE of $\log\widehat{\operatorname{RR}}$, shared size | $\widehat{\operatorname{RR}}$ at the plant | RMSE, ten times the data | $\widehat{\operatorname{RR}}$ at the plant |
| :--- | :--- | :--- | :--- | :--- |
| 0.030 | 5.149 | 0.022 | 0.556 | 0.399 |
| 0.050 | 1.849 | 0.112 | 0.322 | 0.319 |
| 0.083 | 0.849 | 0.146 | 0.201 | 0.263 |
| 0.136 | 0.523 | 0.143 | 0.149 | 0.212 |
| 0.200 | 0.379 | 0.137 | 0.149 | 0.183 |

The true relative risk at the plant is 0.404, and the likelihood cross-validation bandwidth for the pooled events is 0.136. With 46 cases, no bandwidth recovers the plant: the estimates at the plant are between 0.02 and 0.15, and the best overall RMSE comes from the largest bandwidths, which smooth the bump away and report the baseline ratio of 0.15 almost everywhere. With ten times as much data the picture changes. A small bandwidth of 0.03 estimates $0.399$ at the plant, close to the truth, but gives a noisy map elsewhere (an RMSE of 0.556), and the RMSE over the whole map still prefers larger $\psi$ because most of the map has a constant relative risk. This is the usual bias-variance trade-off with a twist: a single bandwidth cannot be best both for the plant and for the rest of the map, and the average error over the map is a poor yardstick for a local feature.

The lesson goes beyond the example. A kernel relative risk map is an exploratory display, and how much it can show is set by the number of cases. Chapter 11 turns the question into a model for the case-control labels and a scan statistic aimed at local excess, which use the data far more efficiently. It also pools five years of cases, because one year of data is too thin to say much about a local excess, and it uses a new simulated draw of the same setting, not the sample of this chapter.

---

## 10.8 Boundary Truncation

Every estimator so far has a weakness near the edge of the window. Events just outside it are invisible, so events inside have fewer observed neighbors than they would in the infinite plane, and every distance-based summary is biased. This section shows how large the bias is and how to remove it.

### 10.8.1 The Size of the Problem, and the Guard Area

For the $K$-function the bias can be computed exactly. The naive estimator, with all weights equal to 1, has expectation equal to the probability that two independent uniform points in the unit square are within distance $r$. The difference of two independent uniform coordinates on $[0, 1]$ has density $1 - |\Delta|$ on $[-1, 1]$, so the two components $(\Delta_1, \Delta_2)$ of the difference of two points have joint density $(1 - |\Delta_1|)(1 - |\Delta_2|)$. Integrating over the disc of radius $r$ in polar coordinates, with $\Delta_1 = v\cos\theta$ and $\Delta_2 = v\sin\theta$, and using the symmetry of the four quadrants,

$$
\begin{aligned}
\operatorname{E}\left[\hat{K}_{\text{naive}}(r)\right] &= 4\int_0^{\pi/2}\int_0^r(1 - v\cos\theta)(1 - v\sin\theta)\,v\,dv\,d\theta \\
&= 4\left[\frac{\pi}{2}\cdot\frac{r^2}{2} - \frac{r^3}{3}\int_0^{\pi/2}(\cos\theta + \sin\theta)\,d\theta + \frac{r^4}{4}\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\right] \\
&= \pi r^2 - \frac{8}{3}r^3 + \frac{1}{2}r^4
\end{aligned}
$$

for $r \le 1$. The correction $-\frac{8}{3}r^3 + \frac{1}{2}r^4$ is negative, and it grows quickly: at $r = 0.2$ the naive estimator is expected to be $0.1051$ instead of $\pi r^2 = 0.1257$, a deficit of 16%.

The simplest fix is the **guard area** (or minus-sampling) estimator. Only events at least $r$ from the boundary are used as centers, though all events are counted as neighbors. For such a center the whole disc of radius $r$ is inside the window, so no neighbor is missed. If $b_i$ is the distance from event $i$ to the boundary and $n_r$ is the number of events with $b_i \ge r$, then

$$
\hat{K}_{\text{guard}}(r) = \frac{\operatorname{area}(\mathcal{D})}{n_r(n-1)}\sum_{i:\,b_i \ge r}\sum_{j \ne i}\mathbb{1}\left(h_{ij} \le r\right)
$$

The price is wasted data: for $r = 0.2$ in the unit square, only the central $0.6 \times 0.6$ region, 36% of the window, supplies centers. And the estimator cannot be used for $r \ge 0.5$.

### 10.8.2 Reduced-Sample Estimators for G and F

The same idea applies to the distance distributions. For event $i$ at distance $b_i$ from the boundary, the observed nearest-neighbor distance $d_i^{\text{nn}}$ equals the true one if $d_i^{\text{nn}} \le b_i$, because the disc of that radius is inside the window and a closer event could not have been missed. If $d_i^{\text{nn}} > b_i$, the disc of radius $b_i$ is inside the window and empty, so the true distance is greater than $b_i$, but it could be smaller than $d_i^{\text{nn}}$ if an unseen event lies outside. In the language of survival analysis, each distance is either observed exactly or **right-censored** at $b_i$.

Two estimators follow.

- **The Kaplan-Meier estimator** treats the pairs $(t_i, \delta_i)$, with $t_i = \min(d_i^{\text{nn}}, b_i)$ and $\delta_i = \mathbb{1}(d_i^{\text{nn}} \le b_i)$, as censored survival data. Sort the times $t_{(1)} \le \dots \le t_{(n)}$, and let $\delta_{(k)}$ be the indicator at the $k$th time. Then
  $$
  1 - \hat{G}_{\text{KM}}(r) = \prod_{k:\,t_{(k)} \le r}\left(1 - \frac{\delta_{(k)}}{n - k + 1}\right)
  $$
  where $n - k + 1$ is the number of events still "at risk" at that time.
- **The Hanisch estimator** weights each uncensored distance by the inverse of the area of the window eroded by that distance, $\mathcal{D}_{\ominus d}$, the set of points at least $d$ from the boundary, which for the unit square has area $(1 - 2d)^2$:
  $$
  \hat{G}_{\text{H}}(r) = \frac{\sum_i\mathbb{1}(d_i^{\text{nn}} \le r)\,\mathbb{1}(d_i^{\text{nn}} \le b_i)/\operatorname{area}(\mathcal{D}_{\ominus d_i^{\text{nn}}})}{\sum_i\mathbb{1}(d_i^{\text{nn}} \le b_i)/\operatorname{area}(\mathcal{D}_{\ominus d_i^{\text{nn}}})}
  $$

For $\hat{F}$, apply the same formulas with the grid test points as centers and their distances to the boundary as $b$.

**The four-point example, again.** The events at $(0.1, 0.1)$, $(0.4, 0.5)$, $(0.9, 0.2)$, and $(0.6, 0.9)$ are at distances 0.1, 0.4, 0.1, and 0.1 from the boundary, and their nearest-neighbor distances are 0.5, 0.447, 0.583, and 0.447. Every nearest-neighbor distance exceeds the corresponding boundary distance, so all four are censored and the Kaplan-Meier estimate of $G$ is 0 at every $r$ in range. The reduced-sample estimators throw away information from events near the edge, and a pattern of four events has nothing but edge events. They need patterns with many events.

### 10.8.3 Ripley's Weight on a Rectangle

The fix with the best use of the data is Ripley's weight, $\omega_{ij} = 1/f_{ij}$, where $f_{ij}$ is the fraction of the circle of radius $h_{ij}$ around $\mathbf{s}_i$ that lies inside the window. For the unit square this fraction has an exact formula.

**A single edge.** Place the event at distance $a$ from an edge, and take a circle of radius $h$. If $a \ge h$ the whole circle is on the near side. If $a < h$ the circle crosses the edge, and the part outside is the set of angles for which the point $\mathbf{s}_i + h(\cos\theta, \sin\theta)$ lies beyond it. Taking the left edge, that is $\cos\theta < -a/h$: the angles within $\alpha = \arccos(a/h)$ of straight left. The outside arc has angular width $2\alpha = 2\arccos(a/h)$, so the fraction of the circle inside, with respect to this one edge, is $1 - \arccos(a/h)/\pi$. For $a = h$ it is 1 and for $a = 0$ it is $1/2$, as it should be.

**Four edges.** Let $\alpha_L, \alpha_R, \alpha_B, \alpha_T$ be the half-angles for the left, right, bottom, and top edges, each equal to $\arccos(a/h)$ for the corresponding distance $a$ if $a < h$ and to zero otherwise. The circle's outside arcs are centered on the directions of the four edges, and each has width $2\alpha$. If they did not overlap, the angle outside would be $2(\alpha_L + \alpha_R + \alpha_B + \alpha_T)$. They do overlap near a corner: the left arc is centered at angle $\pi$ and the bottom arc at $3\pi/2$, which differ by $\pi/2$, so they overlap in an interval of length $\max(0, \alpha_L + \alpha_B - \pi/2)$, and similarly for the other three corners. Opposite edges never overlap. For the left and right edges, $\cos\alpha_L + \cos\alpha_R = (a_L + a_R)/h = 1/h > 0$, where $a_L$ and $a_R$ are the distances to the two edges and sum to 1, so $\cos\alpha_R > -\cos\alpha_L = \cos(\pi - \alpha_L)$ and therefore $\alpha_L + \alpha_R < \pi$, which is exactly the condition for the two arcs, centered at opposite directions, not to meet. The same holds for the bottom and top edges. Any three of the four edges include an opposite pair, so no angle is in three arcs. Inclusion and exclusion therefore stops at pairs:

$$
f_{ij} = 1 - \frac{1}{2\pi}\left[2(\alpha_L + \alpha_R + \alpha_B + \alpha_T) - \sum_{\text{corners}}\max\left(0, \alpha + \alpha' - \frac{\pi}{2}\right)\right]
$$

where the sum runs over the four adjacent pairs of edges. This is what `arc_inside` computes. For event 1 of the example, at $(0.1, 0.1)$ with $h = 0.5$, the left and bottom edges are at distance 0.1, so $\alpha_L = \alpha_B = \arccos(0.2) = 1.369$; the right and top edges are at 0.9, beyond $h$, and contribute nothing. The overlap is $\max(0, 2.738 - 1.571) = 1.167$, and

$$
f = 1 - \frac{1}{2\pi}\left[2(2.738) - 1.167\right] = 1 - 0.6859 = 0.3141
$$

matching the arc fraction in Section 10.6.2.

**Why the weight gives an unbiased estimator.** Under CSR, the expected number of events at distances in $(h, h + dh)$ from event $i$ is $\lambda\cdot2\pi h\,dh$, spread evenly around the circle. A fraction $f_{ij}$ of it lies inside the window and can be observed, so the expected observed count is $\lambda f_{ij}\cdot2\pi h\,dh$. Weighting each observed pair by $1/f_{ij}$ restores the full $\lambda\cdot2\pi h\,dh$, and integrating gives $\operatorname{E}[\hat{K}] = \pi r^2$.

### 10.8.4 Comparing the Fixes

The code implements the naive, guard-area, and Ripley estimators for $K$ and the naive, Hanisch, and Kaplan-Meier estimators for $G$ and $F$, and averages each over 300 CSR patterns with a mean of 100 events, against the truth. It also prints the four-point example and the exact mean of the naive $K$ estimator derived above.

```python
def boundary_dist(P):
    """Distance from each point to the boundary of the unit square."""
    return np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]),
                      np.minimum(P[:, 1], 1 - P[:, 1]))


def k_naive(P, r):
    """K-hat with no edge correction."""
    H = dist(P)
    h = np.sort(H[~np.eye(len(P), dtype=bool)])
    return np.searchsorted(h, r, side="right") / (len(P) * (len(P) - 1))


def k_guard(P, r):
    """Guard-area K-hat: only events at least r from the boundary count
    as centers, and they are compared with all events."""
    H = dist(P)
    b = boundary_dist(P)
    out = []
    for rk in r:
        use = b >= rk
        extra = ((H[use] <= rk).sum(axis=1) - 1).sum()
        out.append(extra / (use.sum() * (len(P) - 1)))
    return np.array(out)


def km_cdf(d, b, r):
    """Kaplan-Meier cdf of distances d that are censored at b."""
    t = np.minimum(d, b)
    seen = d <= b
    order = np.argsort(t)
    t, seen = t[order], seen[order]
    at_risk = len(t) - np.arange(len(t))
    surv = np.cumprod(1 - seen / at_risk)
    return 1 - np.concatenate([[1.0], surv])[np.searchsorted(t, r, "right")]


def g_km(P, r):
    H = dist(P)
    np.fill_diagonal(H, np.inf)
    return km_cdf(H.min(axis=1), boundary_dist(P), r)


def g_hanisch(P, r):
    H = dist(P)
    np.fill_diagonal(H, np.inf)
    d, b = H.min(axis=1), boundary_dist(P)
    w = np.where(d <= b, 1.0 / (1 - 2 * np.minimum(d, 0.499))**2, 0.0)
    return ((d[:, None] <= r[None, :]) * w[:, None]).sum(axis=0) / w.sum()


def f_km(P, r):
    return km_cdf(dist(GRID, P).min(axis=1), boundary_dist(GRID), r)


b4 = boundary_dist(P4)
d4 = (dist(P4) + np.diag(np.full(4, np.inf))).min(axis=1)
print("four points: boundary distances", np.round(b4, 2),
      " nearest-neighbor distances", np.round(d4, 3),
      " all censored:", bool(np.all(d4 > b4)))


rng = np.random.default_rng(2089)
r_k = np.array([0.05, 0.10, 0.15, 0.20])
r_d = np.array([0.03, 0.06, 0.09, 0.12])
out = {name: [] for name in ("kn", "kg", "kr", "gn", "gh", "gk",
                              "fn", "fk")}
for _ in range(300):
    P = sim_poisson(100, rng)
    out["kn"].append(k_naive(P, r_k))
    out["kg"].append(k_guard(P, r_k))
    out["kr"].append(k_ripley(P, r_k))
    out["gn"].append(g_naive(P, r_d))
    out["gh"].append(g_hanisch(P, r_d))
    out["gk"].append(g_km(P, r_d))
    out["fn"].append(f_naive(P, r_d))
    out["fk"].append(f_km(P, r_d))
mean = {key: np.mean(v, axis=0) for key, v in out.items()}
print("K (truth pi r^2):   r =", r_k)
print("  truth        ", np.round(np.pi * r_k**2, 4))
print("  exact mean of the naive estimator",
      np.round(np.pi * r_k**2 - 8 * r_k**3 / 3 + r_k**4 / 2, 4))
for key, label in (("kn", "naive"), ("kg", "guard area"),
                   ("kr", "Ripley")):
    print(f"  {label:13s}", np.round(mean[key], 4))
print("G and F (truth 1 - exp(-100 pi r^2)):   r =", r_d)
print("  truth        ", np.round(1 - np.exp(-100 * np.pi * r_d**2), 4))
for key, label in (("gn", "G naive"), ("gh", "G Hanisch"),
                   ("gk", "G Kaplan-Meier"), ("fn", "F naive"),
                   ("fk", "F Kaplan-Meier")):
    print(f"  {label:14s}", np.round(mean[key], 4))
```

For $K$, the truth is $\pi r^2 = 0.0079, 0.0314, 0.0707, 0.1257$ at $r = 0.05, 0.10, 0.15, 0.20$. The naive estimator averages 0.0076, 0.0288, 0.0619, and 0.1052, matching the exact formula 0.0075, 0.0288, 0.0619, and 0.1051. The guard-area estimator gives 0.0079, 0.0312, 0.0701, 0.1248 and the Ripley estimator 0.0080, 0.0314, 0.0706, 0.1257: both remove the bias, and Ripley's does so with much less waste of data.

For $G$ and $F$ at $r = 0.03, 0.06, 0.09, 0.12$, the truth is $1 - e^{-100\pi r^2} = 0.246, 0.677, 0.922, 0.989$. The naive $\hat{G}$ averages 0.244, 0.647, 0.894, 0.974 and the naive $\hat{F}$ averages 0.240, 0.655, 0.898, 0.976, both too low, with errors of up to 0.03. The corrected versions are close: $\hat{G}$ by Hanisch is 0.250, 0.670, 0.919, 0.989; by Kaplan-Meier 0.249, 0.668, 0.918, 0.988; and $\hat{F}$ by Kaplan-Meier is 0.246, 0.677, 0.923, 0.990. Hanisch and Kaplan-Meier are almost interchangeable. From here on, the lab uses Kaplan-Meier for $G$ and $F$ and Ripley's weights for $K$ and $g$.

---

## 10.9 Monte Carlo Envelopes and Tests

Sections 10.5 and 10.6 gave us curves, and the curves wiggle even for random patterns. To decide whether a wiggle is meaningful we need a reference band, and that is what a simulation envelope is.

### 10.9.1 The Pointwise Envelope

Simulate $m$ patterns from the null model, with the same number of events as the data (Section 10.2.2 explains why conditioning on $n$ is legitimate). Compute the summary, such as $L(r) - r$, for each simulated pattern and for the data. At each $r$, draw the minimum and maximum of the simulated values as an **envelope**.

If the data came from the null model, then the $m + 1$ values at a fixed $r$ (the data's and the $m$ simulated ones) are exchangeable, so the data's value is equally likely to have any rank from 1 to $m + 1$. The chance that it is the smallest, or the largest, is $1/(m + 1)$ each, so the data fall outside the envelope with probability

$$
\frac{2}{m + 1}
$$

at that one $r$. With $m = 39$ this is 5%, and with $m = 99$ it is 2%.

### 10.9.2 Why a Pointwise Envelope Is Not a Test

That statement is about one $r$. A plot shows an entire curve over dozens of $r$ values, and a curve that leaves the band anywhere looks like a rejection. The values of $\hat{L}(r)$ at nearby $r$ are strongly correlated, so the number of independent chances to leave the band is smaller than the number of $r$ values, but it is much larger than one. If the chance of leaving at each of $k$ independent distances were 5%, the chance of leaving at least once would be $1 - 0.95^k$: 54% for $k = 15$.

A test with the right size needs a single number computed from the whole curve. Use the largest absolute deviation from the CSR value,

$$
T = \max_{r \le r_{\max}}\left|\hat{L}(r) - r\right|
$$

compute it for the data and for each of the $m$ simulated patterns, and take

$$
p = \frac{1 + \#\{\text{simulated } T \ge T_{\text{data}}\}}{m + 1}
$$

By exchangeability, the rank of $T_{\text{data}}$ is uniform under the null, so this Monte Carlo p-value is valid for any $m$, with the "+1" counting the data themselves.

The code checks both claims on 100 CSR data sets of about 100 events each, with 39 simulations per data set and 15 distances between 0.02 and 0.2.

```python
r_L = np.linspace(0.02, 0.25, 24)


def max_dev_L(P):
    return np.abs(np.sqrt(k_ripley(P, r_L) / np.pi) - r_L).max()


rng = np.random.default_rng(2090)
m_sim, n_data = 39, 100
r_pt = np.linspace(0.02, 0.2, 15)
point_out = global_out = 0
for _ in range(n_data):
    P = sim_poisson(100, rng)
    L_obs = np.sqrt(k_ripley(P, r_pt) / np.pi) - r_pt
    L_sim = np.array([np.sqrt(k_ripley(rng.uniform(size=(len(P), 2)), r_pt)
                              / np.pi) - r_pt for _ in range(m_sim)])
    point_out += np.any((L_obs < L_sim.min(axis=0))
                        | (L_obs > L_sim.max(axis=0)))
    T_sim = np.abs(L_sim).max(axis=1)
    T_obs = np.abs(L_obs).max()
    global_out += (1 + (T_sim >= T_obs).sum()) / (m_sim + 1) <= 0.05
print(f"CSR data, {n_data} data sets, {m_sim} simulations each:")
print(f"  pointwise envelope leaves the band at some r in "
      f"{point_out / n_data:.2f} of data sets")
print(f"  maximum-deviation test rejects in {global_out / n_data:.2f}")
```

For random data, the pointwise envelope was left at some distance in 32% of the data sets, although a single distance is outside it 5% of the time. The maximum-deviation test rejected in 3% of the data sets, which is within sampling error of 5% (the standard error of a rate based on 100 data sets is about 2%). When a plot of $L(r) - r$ has a curve that briefly touches the envelope, that is not evidence against randomness. The global test is the correct way to ask.

### 10.9.3 Envelopes for Multitype Summaries: Random Labelling

For the multitype summaries the right null model depends on the question.

- Under **independence**, the two patterns are separate processes, and one simulates each. This null is awkward when intensities vary, because $K_{12}(r) = \pi r^2$ no longer describes independence (cases and controls are both dense where people live).
- Under **random labelling**, the locations of the pooled events are fixed, and only the labels are random: each event is a case with the same probability, independently of the others, or, conditioning on the number of cases, the cases are a random subset of the pooled events of the right size. The question is whether the cases are a random sample of the events, which is what no excess risk means. Since the null shuffles labels among fixed locations, it is valid however the pooled intensity varies.

To simulate random labelling, permute the labels. By the same exchangeability argument, the observed summary has a uniform rank among the permuted ones. Section 10.10.2 does this for the case-control data, and plots $\hat{K}_{12}(r)$ minus the average of its relabelled versions.

---

## 10.10 Walkthrough and Lab 10

The lab collects the tools on the three patterns of Figure 10.1 and on the case-control data. Several parts of it have already appeared in hand-sized form: the distance matrix and nearest-neighbor distances for four events (Section 10.5.1), the edge weights and $\hat{K}$ for the same four events (Section 10.6.2), and the thinning table (Section 10.3.2).

### 10.10.1 Random, Clustered, and Inhibited Patterns

For each of the three patterns the code computes $\hat{G}$ and $\hat{F}$ by Kaplan-Meier, $\hat{J}$ from them, $\hat{L}(r) - r$ with Ripley's weights, and $\hat{g}$. It then simulates 99 CSR patterns with the same number of events, computes the same summaries for each, and records the pointwise envelope. Finally it computes the maximum-deviation test for $L$ from Section 10.9.2.

```python
r_G = np.linspace(0.005, 0.07, 14)
r_g2 = np.linspace(0.02, 0.25, 24)


def summaries(P):
    """G, F, J (Kaplan-Meier), L - r, and g for one pattern."""
    G_, F_ = g_km(P, r_G), f_km(P, r_G)
    with np.errstate(divide="ignore", invalid="ignore"):
        J_ = (1 - G_) / (1 - F_)
    L_ = np.sqrt(k_ripley(P, r_L) / np.pi) - r_L
    return {"G": G_, "F": F_, "J": J_, "L": L_, "g": g_pcf(P, r_g2)}


rng = np.random.default_rng(2091)
m_env = 99
obs = {name: summaries(P) for name, P in patterns.items()}
sim = {name: [summaries(rng.uniform(size=(len(P), 2)))
              for _ in range(m_env)] for name, P in patterns.items()}
env, p_global = {}, {}
for name, P in patterns.items():
    env[name] = {key: (np.nanmin([s[key] for s in sim[name]], axis=0),
                       np.nanmax([s[key] for s in sim[name]], axis=0))
                 for key in ("G", "F", "J", "L", "g")}
    T_obs = np.abs(obs[name]["L"]).max()
    T_sim = np.array([np.abs(s["L"]).max() for s in sim[name]])
    p_global[name] = (1 + (T_sim >= T_obs).sum()) / (m_env + 1)
    print(f"{name:13s} max|L - r| = {T_obs:.4f}  "
          f"(95th pct of CSR: {np.quantile(T_sim, 0.95):.4f})  "
          f"Monte Carlo p = {p_global[name]:.2f}")
for name in patterns:
    o = obs[name]
    k5 = np.abs(r_G - 0.03).argmin()
    k_g = np.abs(r_g2 - 0.02).argmin()
    print(f"{name:13s} J(0.03)={o['J'][k5]:.2f}  g(0.02)={o['g'][k_g]:.2f}  "
          f"L(0.10)-0.10={o['L'][np.abs(r_L - 0.10).argmin()]:+.4f}")
```

The test results separate the three patterns decisively. For the random pattern, $\max|\hat{L} - r| = 0.0058$, below the 95th percentile of CSR values (0.0070), and the Monte Carlo p-value is 0.17. For the clustered pattern the statistic is 0.0743, more than ten times the 95th percentile, and $p = 0.01$ (the smallest value 99 simulations can give). For the inhibited pattern it is 0.0300 against a 95th percentile of 0.0062, again $p = 0.01$. Compare with the quadrat test of Section 10.3.4, which rejected the clustered pattern and failed to reject the inhibited one at $p = 0.065$; the distance-based tests see what quadrat counts could not.

At particular distances the summaries read as follows. For the clustered pattern, $\hat{J}(0.03) = 0.21$ and $\hat{g}(0.02) = 5.54$: neighbors are five times as common at that distance as under CSR, and $\hat{L}(0.10) - 0.10 = +0.072$, so that within 0.1 of a typical event there are more neighbors than CSR gives, as if the radius were 0.17 instead of 0.1. For the inhibited pattern, $\hat{J}(0.03) = 1.98$, $\hat{g}(0.02) = 0.29$, and $\hat{L}(0.10) - 0.10 = -0.006$. For the random pattern the values are close to the references: $\hat{J}(0.03) = 1.08$, $\hat{g}(0.02) = 0.87$, and $\hat{L}(0.10) - 0.10 = +0.004$.

![Figure 10.3. Distance summaries for the random, clustered, and inhibited patterns. Rows: $\hat{G}$, $\hat{F}$, and $\hat{J}$, all edge-corrected by Kaplan-Meier. Gray bands: the pointwise envelope from 99 CSR patterns. Dashed blue: the CSR value.](../figures/ch10/fig_10_3_distance.png)

In Figure 10.3, the random pattern stays within the bands in all three rows. In the clustered pattern, $\hat{G}$ lies far above the band and $\hat{F}$ far below it, and $\hat{J}$ falls to near zero. In the inhibited pattern, $\hat{G}$ is zero until the minimum spacing of 0.03 and then catches up; $\hat{F}$ is a little above the band; and $\hat{J}$ is above 1 for most of the range, until noise takes over beyond $r = 0.06$, where the envelope for $J$ widens because $1 - \hat{F}$ is close to zero.

![Figure 10.4. Second-order summaries. Top: $\hat{L}(r) - r$; bottom: the pair correlation function $\hat{g}(r)$. Gray bands are pointwise envelopes from 99 CSR patterns; the titles give the Monte Carlo p-value for the maximum-deviation test.](../figures/ch10/fig_10_4_second_order.png)

Figure 10.4 shows the same story on the second-order scale. The clustered pattern has $\hat{L}(r) - r$ well above zero over the whole range, peaking near $r = 0.08$, and $\hat{g}$ is large at short distances and decays to 1 by about $r = 0.12$: the clusters have a radius of around 0.1, in line with the spread of 0.03 (offspring lie within about three standard deviations, 0.09, of their parent). The inhibited pattern has $\hat{L}(r) - r$ below the band at short distances and $\hat{g}$ dropping below 1 at distances shorter than about 0.04, where the minimum spacing of 0.03 is felt. The random pattern wanders slightly above zero in $\hat{L}(r) - r$ but stays inside the band.

### 10.10.2 Case-Control Data: Relative Risk and Random Labelling

The second half of the lab uses the cases and controls of Section 10.7.3. The code relabels the pooled events at random 99 times, computes $\hat{K}_{12}$ (with Ripley's weights, from cases to controls) and $\hat{G}_{12}$ (with Kaplan-Meier, from cases to their nearest control) for each relabelling, and tests the maximum deviation from the relabelled mean. It then draws the five panels of Figure 10.5.

```python
def g12_km(P1, P2, r):
    """Nearest type-2 point from each type-1 point, Kaplan-Meier."""
    return km_cdf(dist(P1, P2).min(axis=1), boundary_dist(P1), r)


r_c = np.linspace(0.02, 0.25, 12)
r_gc = np.linspace(0.01, 0.15, 15)
n1 = len(cases)
labels_pool = np.vstack([cases, controls])


def cc_stats(idx_case):
    mask = np.zeros(len(labels_pool), dtype=bool)
    mask[idx_case] = True
    A, B = labels_pool[mask], labels_pool[~mask]
    return k12_ripley(A, B, r_c), g12_km(A, B, r_gc)


rng = np.random.default_rng(2092)
k_obs, g_obs = cc_stats(np.arange(n1))
perm = [cc_stats(rng.permutation(len(labels_pool))[:n1])
        for _ in range(99)]
k_sim = np.array([p[0] for p in perm])
g_sim = np.array([p[1] for p in perm])
for label, obs_, sim_ in (("K12", k_obs, k_sim), ("G12", g_obs, g_sim)):
    centre = sim_.mean(axis=0)
    t_obs = np.abs(obs_ - centre).max()
    t_sim = np.abs(sim_ - centre).max(axis=1)
    p = (1 + (t_sim >= t_obs).sum()) / 100
    print(f"{label}: max deviation from the relabelled mean = {t_obs:.4f}, "
          f"95th pct of relabelled = {np.quantile(t_sim, 0.95):.4f}, "
          f"Monte Carlo p = {p:.2f}")
k_pt = np.abs(r_c - 0.10).argmin()
print(f"K12(0.10): observed {k_obs[k_pt]:.4f}, relabelled mean "
      f"{k_sim[:, k_pt].mean():.4f}, band "
      f"[{k_sim[:, k_pt].min():.4f}, {k_sim[:, k_pt].max():.4f}], "
      f"pi r^2 = {np.pi * 0.1**2:.4f}")
```

Neither omnibus test rejects random labelling. For $K_{12}$ the maximum deviation from the relabelled mean is 0.0140, against a 95th percentile of 0.0169, with $p = 0.10$. For $G_{12}$ it is 0.0729 against 0.1467 and $p = 0.57$. At $r = 0.10$ the observed $\hat{K}_{12}$ is 0.0370 and the relabelled mean 0.0390, with a band from 0.0337 to 0.0447.

This is the correct conclusion. In this data set the plant adds about four expected cases to a baseline of 54, and an overall comparison of cases with controls cannot see so small a departure. The figure code draws the maps for both sample sizes.

```python
def show_map(ax, values, title, vmax=None):
    im = ax.imshow(values.reshape(50, 50).T, origin="lower",
                   extent=(0, 1, 0, 1), cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    return im


fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
vmax = truth50.max()
show_map(axes[0, 0], truth50, "true intensity", vmax)
axes[0, 1].plot(crashes[:, 0], crashes[:, 1], "k.", ms=3)
axes[0, 1].plot([0, 1], [0.3, 0.7], "r-", lw=1)
axes[0, 1].set_title(f"{len(crashes)} crashes (highway in red)", fontsize=9)
axes[0, 1].set_aspect("equal")
for ax, ps in ((axes[0, 2], 0.02), (axes[1, 0], psi_lcv),
               (axes[1, 1], 0.2)):
    show_map(ax, kernel_intensity(crashes, ps, G50), f"psi = {ps:.3f}", vmax)
ax = axes[1, 2]
ax.semilogx(psi_grid, lcv - lcv.max(), "k-")
ax.set_xlabel("bandwidth psi")
ax.set_ylabel("LCV minus its maximum")
ax2 = ax.twinx()
ax2.semilogx(psi_grid, ise, "r--")
ax2.set_yscale("log")
ax2.set_ylabel("integrated squared error", color="r")
fig.tight_layout()
fig.savefig("figures/ch10/fig_10_2_kernel.png", dpi=120)

fig, axes = plt.subplots(3, 3, figsize=(10, 8.4), sharex=True)
for col, name in enumerate(patterns):
    for row, key in enumerate(("G", "F", "J")):
        ax = axes[row, col]
        lo_, hi_ = env[name][key]
        ax.fill_between(r_G, lo_, hi_, color="0.85")
        ax.plot(r_G, obs[name][key], "k-")
        if key != "J":
            ax.plot(r_G, 1 - np.exp(-len(patterns[name]) * np.pi * r_G**2),
                    "b--", lw=1)
        else:
            ax.axhline(1.0, color="b", ls="--", lw=1)
            ax.set_ylim(0, 4)
        if row == 0:
            ax.set_title(name)
        if col == 0:
            ax.set_ylabel(f"{key}(r)")
for ax in axes[-1]:
    ax.set_xlabel("r")
fig.tight_layout()
fig.savefig("figures/ch10/fig_10_3_distance.png", dpi=120)

fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharex=True)
for col, name in enumerate(patterns):
    for row, (key, rr_, ref) in enumerate((("L", r_L, 0.0),
                                           ("g", r_g2, 1.0))):
        ax = axes[row, col]
        lo_, hi_ = env[name][key]
        ax.fill_between(rr_, lo_, hi_, color="0.85")
        ax.plot(rr_, obs[name][key], "k-")
        ax.axhline(ref, color="b", ls="--", lw=1)
        if row == 0:
            ax.set_title(f"{name}  (p = {p_global[name]:.2f})")
        if col == 0:
            ax.set_ylabel("L(r) - r" if key == "L" else "g(r)")
        if key == "g":
            ax.set_ylim(0, 6)
for ax in axes[-1]:
    ax.set_xlabel("r")
fig.tight_layout()
fig.savefig("figures/ch10/fig_10_4_second_order.png", dpi=120)

rr_map1 = (kernel_intensity(cases, psi_rr, G50)
           / kernel_intensity(controls, psi_rr, G50))
rr_map10 = (kernel_intensity(cases10, 0.05, G50)
            / kernel_intensity(controls10, 0.05, G50))
vmax = max(rr_map1.max(), rr_map10.max(), true_rr.max())
fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
ax = axes[0, 0]
ax.plot(controls[:, 0], controls[:, 1], ".", color="0.6", ms=3)
ax.plot(cases[:, 0], cases[:, 1], "r.", ms=6)
ax.plot(0.3, 0.6, "k*", ms=12)
ax.set_title(f"{len(cases)} cases (red), {len(controls)} controls; "
             "plant = star", fontsize=8)
ax.set_aspect("equal")
show_map(axes[0, 1], true_rr, "true relative risk", vmax)
show_map(axes[0, 2], rr_map1, f"estimate, shared size (psi = {psi_rr:.3f})",
         vmax)
show_map(axes[1, 0], rr_map10, "estimate, ten times the data (psi = 0.05)",
         vmax)
centre = k_sim.mean(axis=0)
ax = axes[1, 1]
ax.fill_between(r_c, k_sim.min(axis=0) - centre,
                k_sim.max(axis=0) - centre, color="0.85")
ax.plot(r_c, k_obs - centre, "k-")
ax.axhline(0, color="b", ls="--", lw=1)
ax.set_xlabel("r")
ax.set_ylabel("K12(r) minus relabelled mean", fontsize=8)
ax.set_title("random labelling envelope, K12", fontsize=9)
centre_g = g_sim.mean(axis=0)
ax = axes[1, 2]
ax.fill_between(r_gc, g_sim.min(axis=0) - centre_g,
                g_sim.max(axis=0) - centre_g, color="0.85")
ax.plot(r_gc, g_obs - centre_g, "k-")
ax.axhline(0, color="b", ls="--", lw=1)
ax.set_xlabel("r")
ax.set_ylabel("G12(r) minus relabelled mean", fontsize=8)
ax.set_title("random labelling envelope, G12", fontsize=9)
fig.tight_layout()
fig.savefig("figures/ch10/fig_10_5_relative_risk.png", dpi=120)
print("figures saved")
```

![Figure 10.5. The case-control data. Top: the cases (red) and controls (gray) with the plant marked; the true relative risk; and the kernel estimate for the shared data set at the cross-validated bandwidth 0.136. Bottom: the kernel estimate from ten times as much data with a bandwidth of 0.05, and the random labelling envelopes for $K_{12}$ and $G_{12}$ (observed curve minus the mean of the relabelled curves).](../figures/ch10/fig_10_5_relative_risk.png)

In Figure 10.5 the true relative risk is flat at 0.15 with one bump around the plant, rising to about 0.4. The estimate from the shared data (top right) misses the bump entirely: its highest value is in the upper left corner, where a handful of cases happen to sit. The estimate from ten times the data (bottom left) shows a local maximum near $(0.25, 0.55)$, close to the plant, among other lumps of comparable size that are noise. Both envelopes (bottom middle and right) contain the observed curve at every distance.

The chapter ends here with an honest limit. The exploratory tools of this chapter are good at showing broad structure in a pattern, as the three patterns showed, and they are poor at finding a small local excess, as the case-control data showed. The right tool for a known suspect location is a model with the source built in, which the next chapter develops: a Poisson regression for the intensity, an offset for the population, a logistic regression for case-control labels, and a scan statistic for local excess.

---

## Exercises

**Analytical**

1. **Poisson limit.** For a Binomial$(N, p)$ variable with $Np = \mu$ held fixed, show that the probability function converges to the Poisson($\mu$) probability function as $N \to \infty$, directly from $\binom{N}{k}p^k(1-p)^{N-k}$ without generating functions. Then find the leading correction, of order $1/N$, to $\Pr(N(B) = 0)$ in Section 10.2.1.

2. **Conditional uniformity with a varying intensity.** For an inhomogeneous Poisson process with intensity $\lambda(\mathbf{s})$, repeat the calculation of Section 10.2.2 and show that, given $N(\mathcal{D}) = n$, the events are independent with density $\lambda(\mathbf{s})/\int_{\mathcal{D}}\lambda$. What does this say about the number of events needed to estimate the shape of $\lambda$?

3. **Pearson's statistic.** Show that for multinomial counts with $n_q$ equal cell probabilities, $\operatorname{E}[X^2] = n_q - 1$ exactly, as stated in Section 10.2.3. Then show that the index of dispersion $s_n^2/\bar{n}$ has expectation 1.

4. **Thinning.** Using the Bernoulli-grid argument of Section 10.3.2, show that if candidates come from a homogeneous process with intensity $\lambda_{\max}$ and are kept independently with probability $p(\mathbf{s})$, the number of retained events in a region $B$ is Poisson with mean $\lambda_{\max}\int_B p(\mathbf{s})\,d\mathbf{s}$. Extend the argument to show that the retained and the deleted events form independent Poisson processes.

5. **Kernel mean and variance.** Starting from the Bernoulli-grid description, derive the mean and variance formulas for $\hat{\lambda}(\mathbf{s})$ in Section 10.4.1, and verify that $\int\mathcal{K}_\psi(\mathbf{u})^2\,d\mathbf{u} = 1/(4\pi\psi^2)$ for the bivariate Gaussian kernel. Show that the mean squared error $(\psi^2\nabla^2\lambda/2)^2 + \lambda/(4\pi\psi^2)$ is minimized at $\psi = \left(\lambda/(2\pi(\nabla^2\lambda)^2)\right)^{1/6}$. If the intensity is multiplied by a constant $c$, both $\lambda$ and $\nabla^2\lambda$ are multiplied by $c$; show that the best bandwidth then scales like $c^{-1/6}$, and explain in words why more data, and a rougher surface, both call for a smaller bandwidth.

6. **Edge correction when the intensity varies.** For an intensity $\lambda(x, y) = c\,e^{\beta x}$ on the unit square, find the expected value of the edge-corrected estimator at a point on the line $x = 0$, in terms of $\psi$, and compare it with $\lambda(0, y)$. For which sign of $\beta$ is the estimate too high?

7. **Nearest-neighbor distances.** Verify that $\operatorname{E}[d^{\text{nn}}] = 1/(2\sqrt{\lambda})$ under CSR, and find the median. The Clark-Evans ratio is $\bar{d}^{\text{nn}}/(1/(2\sqrt{\hat{\lambda}}))$. Compute its value for the four-point example of Section 10.5.1 and interpret it.

8. **The J-function.** Show that $J(r) = 1$ for every $r$ under CSR. Then use the definition as a ratio of survival functions to explain why $J(r) < 1$ whenever $G(r) > F(r)$, and why a pattern with $G(r) < F(r)$ at every $r$ has $J(r) > 1$.

9. **The $K$-function of a clustered pattern.** For a pattern with pair correlation $g(r) = 1 + c\,e^{-r^2/s^2}$, use $K(r) = \int_0^r2\pi t\,g(t)\,dt$ to find $K(r)$ and $L(r)$, and show that $K(r) - \pi r^2$ tends to $c\pi s^2$ as $r \to \infty$.

10. **Ripley's weight.** For an event at distance $a$ from a single straight edge, show that the fraction of the circle of radius $h > a$ inside the half-plane is $1 - \arccos(a/h)/\pi$, and verify that the weight $1/f$ is between 1 and 2. Then find the largest weight possible for an event at a corner of the unit square with $h \le 1$.

11. **Unbiasedness of the edge-corrected estimator.** Show that under CSR with intensity $\lambda$, $\operatorname{E}\left[\sum_{j \ne i}\omega_{ij}\mathbb{1}(h_{ij} \le r)\,\middle|\,\mathbf{s}_i\right] = \lambda\pi r^2$ for every event position $\mathbf{s}_i$, using the Poisson property that counts in disjoint regions are independent.

12. **Cross-type summaries.** Show that under random labelling, $K_{12}(r) = K_{11}(r) = K_{22}(r)$, each equal to the $K$-function of the pooled pattern. (Hint: under random labelling every ordered pair of events is equally likely to be of types (1, 2), (1, 1), or (2, 2) given its locations.)

13. **Kaplan-Meier by hand.** Five events have nearest-neighbor distances $d = (0.05, 0.08, 0.12, 0.12, 0.2)$ and boundary distances $b = (0.2, 0.06, 0.15, 0.1, 0.3)$. Find the censored times $t_i$, the indicators $\delta_i$, and $\hat{G}_{\text{KM}}$ at $r = 0.05, 0.1, 0.15, 0.2$. Compare with the naive $\hat{G}$.

14. **Envelope levels.** Show that if the data are exchangeable with $m$ simulated patterns at a fixed $r$, the probability of falling outside the envelope is $2/(m + 1)$. For $k$ independent distances and $m = 39$, find the probability that the data fall outside the envelope at one or more, and compare with the 32% found in Section 10.9.2 for 15 correlated distances.

**Computational and simulation**

15. **Quadrat size.** For the three patterns of Section 10.3.4, compute the quadrat test for $m \times m$ grids with $m = 2, 3, \dots, 10$. At which grid size does the test first detect the inhibited pattern? How does this relate to the minimum spacing 0.03?

16. **Bandwidth rules.** Simulate 100 crash patterns as in Section 10.3.2. For each, choose a bandwidth by the normal-reference rule, by likelihood cross-validation, and by minimizing the integrated squared error, and report the average integrated squared error that each rule achieves. How much does the oracle choice gain?

17. **Edge effects and sample size.** Repeat the comparison of Section 10.8.4 for $n = 30$ and $n = 400$. How do the biases of the naive, guard-area, and Ripley estimators change, and at what $r$ does the guard-area estimator become useless?

18. **The pair correlation bandwidth.** Estimate $\hat{g}$ for the Thomas pattern with $b = 0.005$, $0.01$, and $0.02$, averaging over 200 patterns. Plot the average and the standard deviation against $r$. Describe the bias and variance trade-off.

19. **Clark-Evans against L.** Implement the Clark-Evans test of Exercise 7, with an edge correction of your choice, and estimate its power against the clustered and inhibited patterns of Section 10.3.4 at 200 events, by simulation. Compare with the maximum-deviation test for $L$.

20. **Local excess.** Simulate the case-control data 100 times. For each data set compute the kernel relative risk at the plant with $\psi = 0.05$, $0.10$, and $0.20$, and the random labelling test for $K_{11}$. Report the share of data sets in which each method detects the plant, and the average number of cases within 0.15 of the plant. Repeat for ten times the data.

21. **A shot chart.** Simulate shot attempts as an inhomogeneous Poisson process, with intensity highest near the basket and along the three-point line. Label each shot as made with probability $p(\mathbf{s})$ that falls with distance from the basket. Estimate the made and missed intensities by kernel smoothing with a shared bandwidth and plot the ratio $\hat{\lambda}_{\text{made}}/\hat{\lambda}_{\text{missed}}$. How does it relate to $p/(1 - p)$?

---

## What Comes Next

This chapter described patterns. It found structure in them with distances, with the $K$ and $g$ functions, and with a kernel map of intensity, and it judged that structure against simulation. It also met its limits: a small local excess is invisible to omnibus tests, the choice of bandwidth is a problem in itself, and the first-order and second-order explanations of a pattern cannot be told apart without a model.

Chapter 11 supplies the model. It writes the log-likelihood of an inhomogeneous Poisson process, discretizes it with the Berman-Turner device so that it becomes a weighted Poisson regression, fits intensities that depend on covariates such as distance to the highway, and shows that comparing cases with controls is a logistic regression. It also builds the scan statistic for local excess, the inhomogeneous $K$-function that separates trend from interaction, and honest standard errors for clustered data. Chapter 12 then derives the cluster and inhibition models whose simulated patterns appeared here.
