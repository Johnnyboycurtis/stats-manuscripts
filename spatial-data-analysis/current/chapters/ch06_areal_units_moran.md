# Chapter 6: Areal Units, Spatial Weights, and Quadratic Form Tests

Chapters 7 to 9 build models for data attached to regions: counties, census tracts, ZIP codes, spots on a tissue slide. Every one of those models starts from a decision that is easy to make without noticing. Which regions count as neighbors, and how much does each neighbor count? This chapter makes that decision visible, and builds the first tools for areal data: the spatial weights matrix $\mathbf{W}$, and Moran's I, a single number that measures how much neighboring regions resemble each other.

The chapter begins with a warning. The regions themselves are a choice, made by whoever drew the boundaries, and the same underlying people can give different statistics under different boundaries. After that we build weights matrices, represent a map as a graph, and show that Moran's I is a ratio of two quadratic forms in the data. That puts the tools of Chapters 0 and 1 to work: Tool 1 for variances of linear combinations, Tool 2 for quadratic forms, and the eigendecomposition of Chapter 1. We derive the exact mean and variance of Moran's I when there is no spatial structure, extend it to the residuals of a regression, and meet two relatives, Geary's $C$ and the local indicators. The last part explains permutation tests, which need no distributional assumption, and says honestly what they cannot do.

> **Chapter at a Glance**
>
> - **Areal Units Are a Choice (§6.1):** The scale effect, the zoning effect, and the ecological fallacy. The same data, aggregated differently, can give slopes of opposite sign.
> - **Building a Weights Matrix (§6.2):** Rook and queen contiguity, distance bands, inverse distance, and nearest neighbors. Nearest-neighbor weights are not symmetric, and the choice of weights can change a conclusion.
> - **Regions as Graphs (§6.3):** The adjacency matrix $\mathbf{A}$, the degree matrix $\mathbf{D}$, row-standardized weights, and why their eigenvalues are real.
> - **Moran's I (§6.4):** The slope of a regression of the neighbors' average on the value, and a ratio of quadratic forms.
> - **Exact Null Moments (§6.5):** $\operatorname{E}[I] = -1/(n-1)$, the variance, and why the numerator and denominator can be handled separately.
> - **Residuals (§6.6):** The same derivation after a regression, with $\mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$ in place of $\mathbf{P}_c$.
> - **Geary's C and Local Indicators (§6.7):** A Laplacian quadratic form, and region-by-region contributions that add up to Moran's I.
> - **Permutation Inference (§6.8):** An exact test on four regions, a Monte Carlo test on 144, and two situations where permuting the data gives the wrong answer.
> - **Walkthrough and Lab (§6.9):** Hand calculations collected in code, a comparison of weights, and a simulated study of the modifiable areal unit problem.

---

## 6.1 Areal Units Are a Choice: The Modifiable Areal Unit Problem

Areal data arrive already aggregated. A health department reports overdose deaths by county. A census reports income by tract. A mail service reports delivery times by ZIP code. Behind each number are individual people or events, and the regions are the bins into which they were sorted.

Nobody chose those bins to answer the analyst's question. Census tracts are drawn to hold roughly 4,000 people. ZIP codes follow mail routes. Counties follow history and politics. If the bins had been drawn differently, the same people would produce different numbers, and so different statistics. This is the **modifiable areal unit problem**, usually shortened to MAUP. It has two parts.

### 6.1.1 The Scale Effect and the Zoning Effect

The **scale effect** is what happens when the same data are aggregated into fewer, larger units. The **zoning effect** is what happens when the number of units stays the same but the boundaries move. The two are best seen on a small example you can check by hand.

**A hand-sized example: four tracts, two ways to form ZIP codes.** Four census tracts sit in a two-by-two block. Each has a value of a covariate $x$ (a deprivation index) and an outcome $y$ (an overdose rate):

| Tract | NW | NE | SW | SE |
| :--- | :--- | :--- | :--- | :--- |
| $x$ | 1 | 5 | 3 | 7 |
| $y$ | 3 | 6 | 2 | 3 |

Regressing $y$ on $x$ across the four tracts gives a slope of $0.2$: $\bar{x} = 4$, $\bar{y} = 3.5$, the cross-product sum is $4$, the sum of squares of $x$ is $20$, and $4/20 = 0.2$.

Now combine the tracts into two ZIP codes, by averaging. In zoning A the ZIP codes are a west unit (NW and SW) and an east unit (NE and SE). The west unit has $\bar{x} = 2$ and $\bar{y} = 2.5$, and the east unit has $\bar{x} = 6$ and $\bar{y} = 4.5$. The slope across the two units is $(4.5 - 2.5)/(6 - 2) = +0.5$. In zoning B the ZIP codes are a north unit (NW and NE) and a south unit (SW and SE). The north unit has $\bar{x} = 3$ and $\bar{y} = 4.5$, and the south unit has $\bar{x} = 5$ and $\bar{y} = 2.5$. The slope is $(2.5 - 4.5)/(5 - 3) = -1.0$.

The same four tracts give a slope of $+0.2$, $+0.5$, or $-1.0$, depending only on how they are grouped. The code below repeats the calculation, using an **aggregation matrix** (called `G` in the code): each row corresponds to one unit and holds the averaging weights of its tracts, so that `G @ x` is the vector of unit averages.

```python
import numpy as np
from scipy import stats


def slope(x, y):
    """OLS slope of y on x, with an intercept."""
    xc = x - x.mean()
    return xc @ (y - y.mean()) / (xc @ xc)


# Four tracts, in the order NW, NE, SW, SE
x_t = np.array([1.0, 5.0, 3.0, 7.0])
y_t = np.array([3.0, 6.0, 2.0, 3.0])
zoning_A = np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) / 2  # west unit, east unit
zoning_B = np.array([[1, 1, 0, 0], [0, 0, 1, 1]]) / 2  # north unit, south unit
print("tracts:", round(slope(x_t, y_t), 3))
for name, G in (("A, west/east:", zoning_A), ("B, north/south:", zoning_B)):
    print(name, "units", G @ x_t, G @ y_t,
          "slope", round(slope(G @ x_t, G @ y_t), 3))
```

Two units give a slope that is determined by two points, so this example is only a demonstration that grouping matters. For a realistic picture we need a simulation, and for that we need a model for how the slope depends on the scale of aggregation.

**Why aggregation changes the slope.** Suppose the covariate at each location is the sum of two independent parts, a local part $a$ that varies from block to block and a regional part $b$ that varies smoothly over long distances, so that $x = a + b$. Suppose the outcome responds to the two parts differently: $y = 1.0\,a - 0.5\,b + \varepsilon$, with independent noise $\varepsilon$. At the finest scale, a unit increase in the local part of $x$ goes with a unit increase in $y$, and the regional part pulls the other way. Everything in this example is simulated.

Now average over larger and larger units. Averaging shrinks the variance of the local part quickly, since many independent-ish values are averaged. It shrinks the variance of the regional part much more slowly, because neighboring values are nearly equal and averaging them changes little. So at coarse scales the covariate is dominated by $b$, and the slope turns toward the regional effect.

Writing $v_a$ and $v_b$ for the variances across units of the aggregated local and regional parts, the cross-product of $\bar{x}$ and $\bar{y}$ is $1.0\,v_a - 0.5\,v_b$ (the parts are independent, and the noise is unrelated to $x$), and the variance of $\bar{x}$ is $v_a + v_b$. So the regression slope across units is approximately

$$
\text{slope} \approx \frac{1.0\,v_a - 0.5\,v_b}{v_a + v_b} = \frac{v_a}{v_a + v_b}\,(+1.0) + \frac{v_b}{v_a + v_b}\,(-0.5)
$$

a weighted average of the local effect $+1.0$ and the regional effect $-0.5$, with weights equal to each part's share of the variance of the covariate. The weights depend on the aggregation, and that is the whole mechanism. (This is the same kind of weighted average as the variance ratio of Section 1.3.4.) It is approximate because the average of a ratio is not the ratio of averages.

**The simulation.** Section 6.9.3 simulates 200 worlds on a fine grid of $48 \times 48$ cells. The local part $a$ has exponential correlation with range 0.02, about one cell. The regional part $b$ has exponential correlation with range 0.6 east-west and 0.2 north-south, so the region is stretched along the east-west direction. Each world is aggregated six ways: into $12 \times 12$, $6 \times 6$, and $3 \times 3$ squares (the scale effect), and into $48 \times 3$ and $3 \times 48$ strips (the zoning effect: like the $12 \times 12$ squares, both strip layouts have 144 units, but the units have a different shape).

Here are the results, averaged over the 200 worlds. The last column evaluates the weighted-average formula from the exact variances.

| Aggregation | Units | Average slope | Share of worlds with slope $< 0$ | Average correlation | Predicted slope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fine cells | 2,304 | 0.383 | 0.00 | 0.333 | 0.356 |
| Squares | 144 | $-0.088$ | 0.76 | $-0.133$ | $-0.128$ |
| Squares | 36 | $-0.287$ | 0.99 | $-0.470$ | $-0.321$ |
| Squares | 9 | $-0.397$ | 0.99 | $-0.706$ | $-0.422$ |
| Strips, east-west | 144 | $-0.207$ | 0.95 | $-0.307$ | $-0.243$ |
| Strips, north-south | 144 | $-0.045$ | 0.65 | $-0.082$ | $-0.127$ |

Read down the squares first. At the finest scale the slope is positive, $+0.383$ on average, and it is positive in every world. By $12 \times 12$ units it is $-0.088$, negative in 76% of worlds, and by $3 \times 3$ units it is $-0.397$, negative in 99%. The correlation goes from $+0.33$ to $-0.71$. The sign of the relationship reverses as the units get bigger, as the weighted-average formula predicted, with predicted slopes within about $0.04$ of the simulated averages for the squares.

The two strip layouts have the same number of units as the $12 \times 12$ squares, yet they give different slopes, $-0.207$ and $-0.045$. That is the zoning effect: the number of units is fixed and only the shape of the boundaries changes. East-west strips average along the direction in which the regional part is smooth, so they retain more of it and the slope is more negative. The last predicted value, $-0.127$ for north-south strips, is farther from the simulated $-0.045$, and those worlds are noisy: the standard deviation of the slope across worlds is $0.191$, the largest of any aggregation.

Figure 6.1 shows one world at four aggregations. Figure 6.2 shows the distribution of the slope over all 200 worlds.

![Figure 6.1. One simulated world at four aggregations: the fine cells, 144 squares, nine squares, and 144 east-west strips. The fitted line goes from rising to falling as the units get larger.](../figures/ch06/fig_6_1_maup_one.png)

![Figure 6.2. The regression slope across 200 simulated worlds, for six aggregations. The sign changes with the scale, and strips and squares with the same number of units give different slopes.](../figures/ch06/fig_6_2_maup_slopes.png)

The simulated worlds are invented, and nothing here is a finding about real overdose rates. The point is that all six sets of units describe the same underlying cells, and an analyst who published only one of them would report a positive, a negative, or a near-zero relationship, each correct for its own units.

Moran's I, defined in Section 6.4, behaves differently from the slope. Averaged over the 200 worlds, the Moran's I of the outcome is $0.234$ for the fine cells, $0.343$ for the $12 \times 12$ squares, $0.328$ for $6 \times 6$, and $0.100$ for $3 \times 3$. It rises when aggregation removes the local noise and falls when only nine units are left, partly because the benchmark value $-1/(n-1)$ it is compared with moves from $-0.0004$ to $-0.125$. The scale at which spatial dependence is measured matters too.

### 6.1.2 The Ecological Fallacy

A related warning concerns what a regional relationship says about individuals. The **ecological fallacy** is the inference that a relationship between regional averages holds for the individuals inside the regions.

**A hand-sized example: three counties, nine people.** Suppose each county has three people. Within every county, a person with a higher exposure $x$ has a higher risk $y$, with a slope of exactly $+1$. But counties with a higher average exposure have a lower average risk, because (in this invented story) counties with high exposure also have better access to treatment. The county averages are

| County | 1 | 2 | 3 |
| :--- | :--- | :--- | :--- |
| Average $x$ | 1 | 3 | 5 |
| Average $y$ | 8 | 5 | 2 |

and the three people in each county sit at $-1$, $0$, and $+1$ around its averages, in both $x$ and $y$. Within each county the slope is $+1$. Across the three county averages the slope is $(2 - 8)/(5 - 1) = -1.5$. Pooling all nine people ignores the counties and gives something in between. The pooled cross-product sum is the within-county part, $3 \times 2 = 6$, plus the between-county part, $3 \times [(-2)(3) + 0 + (2)(-3)] = -36$, for $-30$. The pooled sum of squares of $x$ is $3 \times 2 + 3 \times 8 = 30$. The pooled slope is $-30/30 = -1.0$.

```python
x_reg = np.array([1.0, 3.0, 5.0])
y_reg = np.array([8.0, 5.0, 2.0])
offsets = np.array([-1.0, 0.0, 1.0])
x_ind = np.concatenate([xr + offsets for xr in x_reg])
y_ind = np.concatenate([yr + offsets for yr in y_reg])
region = np.repeat([0, 1, 2], 3)
within = [slope(x_ind[region == k], y_ind[region == k]) for k in range(3)]
print("within-region slopes:", within)
print("slope of the region means:", round(slope(x_reg, y_reg), 3))
print("pooled slope, all nine people:", round(slope(x_ind, y_ind), 3))
```

The county-level regression says the relationship is negative, with a slope of $-1.5$. For every individual in every county, the relationship is positive. An analysis that used only the three county averages, and then claimed that an individual's risk falls as the individual's exposure rises, would be wrong about every person in the data. Chapter 9 returns to this when it models county counts: a county-level coefficient describes counties.

### 6.1.3 What This Means for Every Model in Part III

The models of Chapters 7 to 9 describe the regions that were given, and they inherit three limits from this section.

- **A coefficient belongs to its units.** A slope estimated from tracts describes tracts. It need not describe ZIP codes, counties, or people.
- **The boundaries are part of the assumptions.** If the conclusion changes when the boundaries move to another reasonable set, it was a statement about the boundaries as much as about the world. When another reasonable set of units exists, fit both.
- **Spatial dependence has a scale.** Regions that are small relative to the process look strongly dependent, and regions that are large look nearly independent.

None of these limits is cured by a better model. They are properties of the data.

---

## 6.2 Building a Spatial Weights Matrix

Moran's I and every model in Chapters 7 to 9 use a matrix $\mathbf{W}$ whose entry $w_{ij}$ says how much region $j$ counts as a neighbor of region $i$. By convention $w_{ii} = 0$, so that no region is its own neighbor. The matrix is not estimated from the data. The analyst writes it down, and it encodes a belief about which regions influence which. This section describes the common ways to build it.

The running example is the $12 \times 12$ map of 144 square counties from the shared data set of Part III, in which $\mathbf{S}$ holds the county centroids, `pop` the populations, and `x` a smooth standardized deprivation index.

### 6.2.1 Contiguity: Rook and Queen

The most common definition says two regions are neighbors if they share a border. For squares, there are two versions. **Rook** contiguity counts regions that share an edge, as rooks move on a chessboard. **Queen** contiguity counts regions that share an edge or a corner. The result is a binary matrix $\mathbf{A}$ with $A_{ij} = 1$ for neighbors, the adjacency matrix of Section 1.5.2.

```python
def dist(S):
    """Matrix of Euclidean distances between the rows of S."""
    q = np.sum(S**2, axis=1)
    H2 = q[:, None] - 2 * S @ S.T + q[None, :]
    H = np.sqrt(np.maximum(H2, 0.0))
    np.fill_diagonal(H, 0.0)
    return H


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


def rook(nr, nc):
    """Rook adjacency for an nr-by-nc grid of squares (row = k // nc)."""
    B = np.zeros((nr * nc, nr * nc))
    for i in range(nr):
        for j in range(nc):
            k = i * nc + j
            if i + 1 < nr:
                B[k, k + nc] = B[k + nc, k] = 1
            if j + 1 < nc:
                B[k, k + 1] = B[k + 1, k] = 1
    return B


def queen(nr, nc):
    """Queen adjacency: squares sharing an edge or a corner."""
    B = np.zeros((nr * nc, nr * nc))
    for i in range(nr):
        for j in range(nc):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di, dj) != (0, 0) and 0 <= i + di < nr \
                            and 0 <= j + dj < nc:
                        B[i * nc + j, (i + di) * nc + j + dj] = 1
    return B


S, A, pop, x = county_map()
n = len(S)
A_queen = queen(12, 12)
print("county_map adjacency is rook:", np.array_equal(A, rook(12, 12)))
for name, B in (("rook", A), ("queen", A_queen)):
    print(f"{name}: {int(B.sum())} nonzero entries, neighbors per county",
          sorted(set(B.sum(axis=1).astype(int))))
```

On the $12 \times 12$ map, interior counties have 4 rook neighbors and 8 queen neighbors. Counties on an edge have 3 rook neighbors or 5 queen neighbors, and corner counties have 2 and 3. The rook matrix has 528 nonzero entries and the queen matrix 1,012.

Spatial transcriptomics, which measures gene expression at thousands of spots on a tissue slide, uses the same idea with a different geometry. The spots usually lie on a hexagonal lattice, with every other row shifted by half a spacing, so each interior spot touches six others instead of four or eight. The code below builds the neighbor matrix for a small patch of spots by looking for pairs at one spacing apart, an idea that Section 6.2.2 generalizes.

```python
# Hexagonal spots: every other row is shifted by half a spacing
spots = np.array([[c + 0.5 * (r % 2), r * np.sqrt(3) / 2]
                  for r in range(6) for c in range(6)])
Hs = dist(spots)
A_hex = ((Hs > 0) & (Hs < 1.01)).astype(float)
print("neighbors per spot:", sorted(set(A_hex.sum(axis=1).astype(int))))
```

Spots on the edge of the patch have 2 to 5 neighbors and interior spots have 6. Everything later in this chapter applies to such a matrix without change.

Contiguity needs a map of the boundaries, and a region with no neighbors (an island) has a row of zeros, which cannot be row-standardized (Section 6.3). Real data sets often have such regions and need a decision about them.

### 6.2.2 Distance Bands, Inverse Distance, and Nearest Neighbors

When only centroids are known, or when influence is thought to depend on distance rather than on touching, there are three common choices. All use the distance matrix $\mathbf{H}$ of Chapter 1.

- **Distance band.** Regions within a chosen radius are neighbors: $A_{ij} = 1$ if $0 < h_{ij} \le r$. A band just wider than one cell gives the rook matrix, and a band of 1.5 cells, which reaches the diagonal neighbors at $\sqrt{2} \approx 1.41$ cells, gives the queen matrix.
- **Inverse distance.** Nearer regions count more: $w_{ij} \propto h_{ij}^{-1}$ (or another power), usually with a cutoff beyond which the weight is zero.
- **$k$ nearest neighbors.** Region $j$ is a neighbor of region $i$ if $j$ is among the $k$ regions closest to $i$. Every region has exactly $k$ neighbors, which avoids islands. On a regular grid many regions are exactly the same distance away, so "the $k$ nearest" is not unique: for $k = 4$, 44 of the 144 counties have more than four counties at or inside the fourth-smallest distance, and for $k = 8$, 8 of them do (the code prints these counts). The code below breaks such ties by region number, which means the neighbors it picks lean toward the south and west; a different rule would give slightly different numbers for the nearest-neighbor rows.

```python
H = dist(S)
cell = 1 / 12


def band(H, radius):
    """Binary weights: 1 if the distance is positive and at most radius."""
    return ((H > 0) & (H <= radius)).astype(float)


def inverse_distance(H, cutoff, power=1.0):
    """Weights 1 / h^power for pairs within the cutoff, 0 otherwise."""
    B = np.zeros_like(H)
    near = (H > 0) & (H <= cutoff)
    B[near] = H[near] ** (-power)
    return B


def knn(H, k):
    """Binary weights: j is a neighbor of i if j is among i's k nearest.

    Ties are broken by region number (lower first), after rounding the
    distances to 9 decimals so that equal distances compare equal."""
    n = len(H)
    B = np.zeros_like(H)
    index = np.arange(n)
    for i in range(n):
        d = np.round(H[i], 9)
        d[i] = np.inf
        B[i, np.lexsort((index, d))[:k]] = 1.0
    return B


weights = {"rook": A,
           "queen": A_queen,
           "band, 2.5 cells": band(H, 2.5 * cell + 1e-9),
           "4 nearest": knn(H, 4),
           "8 nearest": knn(H, 8),
           "inverse distance": inverse_distance(H, 3.5 * cell)}
print("a band of 1 cell is the rook:",
      np.array_equal(band(H, cell + 1e-9), A))
print("a band of 1.5 cells is the queen:",
      np.array_equal(band(H, 1.5 * cell + 1e-9), A_queen))
print(f"{'weights':18s}{'min':>5s}{'mean':>7s}{'max':>5s}  symmetric")
for name, B in weights.items():
    nb = (B > 0).sum(axis=1)
    print(f"{name:18s}{nb.min():5d}{nb.mean():7.2f}{nb.max():5d}  "
          f"{np.allclose(B, B.T)}")


def count_ties(H, k):
    """Counties with more than k counties at or inside the k-th distance."""
    d = np.round(H, 9)
    np.fill_diagonal(d, np.inf)
    kth = np.sort(d, axis=1)[:, k - 1]
    return int(((d <= kth[:, None]).sum(axis=1) > k).sum())


print("counties with a tie at the cutoff, k = 4:", count_ties(H, 4))
print("counties with a tie at the cutoff, k = 8:", count_ties(H, 8))
```

| Weights | Fewest neighbors | Mean | Most neighbors | Symmetric? |
| :--- | :--- | :--- | :--- | :--- |
| Rook | 2 | 3.67 | 4 | Yes |
| Queen | 3 | 7.03 | 8 | Yes |
| Band, 2.5 cells | 7 | 16.47 | 20 | Yes |
| 4 nearest | 4 | 4.00 | 4 | No |
| 8 nearest | 8 | 8.00 | 8 | No |
| Inverse distance, 3.5 cells | 12 | 27.75 | 36 | Yes |

The last column foreshadows the next subsection: the two nearest-neighbor matrices are not symmetric.

### 6.2.3 Why Nearest-Neighbor Weights Are Not Symmetric

Picture three rural clinics along a highway at mile posts 0, 1, and 3. If each clinic names its single nearest clinic as its neighbor, clinic 1 (at mile 0) names clinic 2 (at mile 1), and clinic 2 names clinic 1. Clinic 3 (at mile 3) is 2 miles from clinic 2 and 3 miles from clinic 1, so it names clinic 2. But clinic 2 does not name clinic 3. The relation "is the nearest neighbor of" is not mutual.

```python
pts = np.array([0.0, 1.0, 3.0])
Hp = np.abs(pts[:, None] - pts[None, :])
B3 = knn(Hp, 1)
print("nearest-neighbor matrix:\n", B3)
print("symmetric:", np.allclose(B3, B3.T))
print("either lists the other:\n", np.maximum(B3, B3.T))
print("average of the two:\n", (B3 + B3.T) / 2)
```

The matrix has $B_{32} = 1$ but $B_{23} = 0$, so it is not symmetric. Two common repairs make it symmetric: count a pair as neighbors if either lists the other (take the larger of $B_{ij}$ and $B_{ji}$), or average the two entries.

Symmetry is not a detail. Several results in this book assume it, and nearest-neighbor weights break them in two ways.

- **Real eigenvalues (Section 6.3.3).** The argument that row-standardized weights have real eigenvalues needs the underlying binary matrix to be symmetric. Without it, eigenvalues can be complex numbers. On the county map the row-standardized 4-nearest and 8-nearest matrices have eigenvalues with imaginary parts as large as $0.098$ and $0.040$ (Section 6.3.3). Chapter 7's bound on the spatial autoregressive parameter, which is set by the smallest and largest eigenvalues of the weights matrix, assumes real eigenvalues.
- **The conditional autoregressive model (Chapter 8).** A conditional specification, in which each region is modeled given its neighbors, can define a valid joint distribution only if its regression coefficients and conditional variances satisfy a symmetry condition, and with a nonsymmetric neighbor structure that condition fails except in special cases (Section 8.1.3).

The moments of Moran's I in Section 6.5 only use the symmetrized matrix $(\mathbf{W} + \mathbf{W}')/2$, so Moran's I itself is not affected. The models are.

### 6.2.4 Sensitivity of Conclusions to the Choice of W

There is no correct weights matrix. Different choices answer different questions: do counties resemble the counties they touch, or the counties within 30 miles, or the eight closest? Two checks keep an analysis honest. Use a definition that matches how influence is thought to travel, and report how much the conclusion depends on it.

To see the dependence, the code below simulates two responses on the county map, both described as overdose deaths per 10,000 residents (simulated, not real). The strong response has a deprivation trend plus a long-range spatial effect (range 0.2). The weak response has no trend and a short-range spatial effect (range 0.05, about half a county) mixed with more noise. It then computes Moran's I, defined formally in Section 6.4 but needed here only as a number, for six row-standardized weight matrices.

```python
def row_standardize(B):
    """Divide each row by its sum (every row needs at least one neighbor)."""
    return B / B.sum(axis=1, keepdims=True)


def simulate_rate(seed, mean, sigma2, tau2, phi):
    """Normal 'rate' with an exponential spatial effect and a nugget."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(sigma2 * np.exp(-H / phi) + 1e-8 * np.eye(n))
    u = L @ rng.standard_normal(n)
    return mean + u + np.sqrt(tau2) * rng.standard_normal(n)


# Overdose deaths per 10,000 residents (simulated), with deprivation as a
# covariate: a strong, long-range pattern, and a weak, short-range one
y_strong = simulate_rate(2040, 10 + 1.5 * x, 1.0, 0.5, 0.2)
y_weak = simulate_rate(2041, 0.0, 0.5, 1.0, 0.05)


def moran_I(y, W):
    """Moran's I for data y and a weights matrix W with zero diagonal."""
    z = y - y.mean()
    return len(y) / W.sum() * (z @ W @ z) / (z @ z)


print(f"{'weights':18s}{'I, strong':>11s}{'I, weak':>10s}")
for name, B in weights.items():
    W = row_standardize(B)
    print(f"{name:18s}{moran_I(y_strong, W):11.3f}{moran_I(y_weak, W):10.3f}")
```

| Weights | $I$, strong response | $I$, weak response |
| :--- | :--- | :--- |
| Rook | 0.615 | 0.067 |
| Queen | 0.596 | 0.063 |
| Band, 2.5 cells | 0.454 | 0.077 |
| 4 nearest | 0.609 | 0.071 |
| 8 nearest | 0.553 | 0.072 |
| Inverse distance | 0.413 | 0.055 |

The values differ by weights, but their size is not the right thing to compare, since wider neighborhoods average more distant counties and dilute $I$ even when the pattern is real. What matters is whether $I$ is large compared with what chance would give for that matrix. Section 6.5 gives the benchmark, and Section 6.9.2 returns to this table with a significance test for each row. The result for the weak response is that some weights detect the pattern and others do not, so for borderline patterns the conclusion depends on the weights.

---

## 6.3 Representing Regions as Graphs

A map of regions with a neighbor relation is a graph: regions are the **nodes**, and two nodes are joined by an **edge** when they are neighbors. Graph language gives us the matrix tools of earlier chapters.

For hand calculations we use a small map of four regions, the **hub map**. Region 1 touches only region 2. Region 2 touches regions 1, 3, and 4. Regions 3 and 4 touch each other and region 2. Region 2 is the hub, with three neighbors, and regions 2, 3, and 4 form a triangle.

### 6.3.1 The Adjacency Matrix and the Degree Matrix

The **adjacency matrix** $\mathbf{A}$ has $A_{ij} = 1$ if regions $i$ and $j$ are neighbors and $0$ otherwise, with zeros on the diagonal. For an undirected map it is symmetric. The number of neighbors of region $i$ is its **degree**, $d_i = \sum_j A_{ij}$, and the **degree matrix** is the diagonal matrix $\mathbf{D} = \operatorname{diag}(\mathbf{A}\mathbf{1}_n)$ with the degrees on its diagonal. For the hub map,

$$
\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}, \qquad \mathbf{D} = \operatorname{diag}(1, 3, 2, 2)
$$

### 6.3.2 Row-Standardized Weights

In most of this book the weights matrix is the **row-standardized** version of $\mathbf{A}$, in which each row is divided by its sum:

$$
\mathbf{W} = \mathbf{D}^{-1}\mathbf{A}, \qquad w_{ij} = \frac{A_{ij}}{d_i}
$$

Each region's neighbors share a total weight of 1, so $(\mathbf{W}\mathbf{y})_i = \sum_j w_{ij}y_j$ is the average of $y$ over region $i$'s neighbors. In matrix form the row sums are $\mathbf{W}\mathbf{1}_n = \mathbf{D}^{-1}\mathbf{A}\mathbf{1}_n = \mathbf{D}^{-1}\mathbf{D}\mathbf{1}_n = \mathbf{1}_n$. The sum of all the weights, $S_0 = \sum_i\sum_j w_{ij} = \mathbf{1}'\mathbf{W}\mathbf{1}$, therefore equals $n$.

For the hub map,

$$
\mathbf{W} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ \tfrac{1}{3} & 0 & \tfrac{1}{3} & \tfrac{1}{3} \\ 0 & \tfrac{1}{2} & 0 & \tfrac{1}{2} \\ 0 & \tfrac{1}{2} & \tfrac{1}{2} & 0 \end{bmatrix}
$$

Row-standardized weights have an important property that is easy to miss: $\mathbf{W}$ is generally **not symmetric**, even though $\mathbf{A}$ is. Here $w_{12} = 1$ but $w_{21} = 1/3$: region 1's only neighbor counts for all of its weight, while region 1 is one of three neighbors of region 2. Row-standardizing makes each row an average, and it gives up symmetry to do so.

### 6.3.3 A Similarity Transformation: The Eigenvalues of W Are Real

The eigenvalues of $\mathbf{W}$ will matter in Chapters 7 and 8, because they limit the spatial autoregressive parameter of those chapters. Since $\mathbf{W}$ is not symmetric, nothing guarantees they are real. They are, for a reason that rests on a new tool.

**A tool we need: similar matrices.** Two square matrices $\mathbf{B}$ and $\mathbf{C}$ are **similar** if $\mathbf{B} = \mathbf{N}^{-1}\mathbf{C}\mathbf{N}$ for some invertible matrix $\mathbf{N}$. Similar matrices have the same eigenvalues. The proof takes one line. If $\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$, then $\mathbf{B}(\mathbf{N}^{-1}\mathbf{v}) = \mathbf{N}^{-1}\mathbf{C}\mathbf{N}\mathbf{N}^{-1}\mathbf{v} = \mathbf{N}^{-1}\mathbf{C}\mathbf{v} = \lambda\,\mathbf{N}^{-1}\mathbf{v}$, so $\lambda$ is an eigenvalue of $\mathbf{B}$ with eigenvector $\mathbf{N}^{-1}\mathbf{v}$. (Here $\mathbf{N}$ is only a generic invertible matrix.)

Define the symmetric matrix $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, where $\mathbf{D}^{-1/2}$ is the diagonal matrix with entries $d_i^{-1/2}$. It is symmetric because $\mathbf{A}$ is. Then

$$
\mathbf{D}^{-1/2}\,\tilde{\mathbf{A}}\,\mathbf{D}^{1/2} = \mathbf{D}^{-1/2}\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\mathbf{D}^{1/2} = \mathbf{D}^{-1}\mathbf{A} = \mathbf{W}
$$

so $\mathbf{W}$ is similar to $\tilde{\mathbf{A}}$. A symmetric real matrix has real eigenvalues and an orthonormal set of eigenvectors, so $\mathbf{W}$ has the same real eigenvalues. Chapter 8 writes $\tilde{\lambda}_k$ for the eigenvalues of $\tilde{\mathbf{A}}$; they are the same numbers as the eigenvalues $\lambda_i$ of $\mathbf{W}$. This is the step that fails for nearest-neighbor weights: there $\mathbf{A}$ is not symmetric, so $\tilde{\mathbf{A}}$ is not either.

**Where the eigenvalues lie.** They are between $-1$ and $1$, and $1$ is always an eigenvalue. To see this, let $\mathbf{W}\mathbf{x} = \lambda\mathbf{x}$ and pick the entry $x_i$ with the largest absolute value. Then $\lvert\lambda\rvert\lvert x_i\rvert = \lvert\sum_j w_{ij}x_j\rvert \le \sum_j w_{ij}\lvert x_j\rvert \le \lvert x_i\rvert$, because the weights in a row are nonnegative and add to 1, so $\lvert\lambda\rvert \le 1$. And $\mathbf{W}\mathbf{1}_n = \mathbf{1}_n$ shows $\lambda = 1$ with eigenvector $\mathbf{1}_n$.

The value $-1$ is an eigenvalue when the map can be colored with two colors so that every edge joins different colors (a **bipartite** map). If $x_i = +1$ on one color and $-1$ on the other, then $(\mathbf{W}\mathbf{x})_i$ is the average of neighbors that all have the opposite sign, so $\mathbf{W}\mathbf{x} = -\mathbf{x}$. A rook grid of squares is bipartite (color it like a checkerboard), and the hub map is not, since regions 2, 3, and 4 form a triangle.

**By hand on the hub map.** The eigenvalues of $\mathbf{W}$ are $1$, $-\tfrac{1}{2}$, and the two roots of $x^2 + \tfrac{1}{2}x - \tfrac{1}{6} = 0$, which are $(-3 \pm \sqrt{33})/12$: approximately $0.2287$ and $-0.7287$. They sum to zero, the trace of $\mathbf{W}$, as they must. The code below confirms that $\mathbf{W}$ and $\tilde{\mathbf{A}}$ have the same eigenvalues, and then checks the rook grid and the nearest-neighbor matrices of Section 6.2.3.

```python
A_hub = np.array([[0, 1, 0, 0],
                  [1, 0, 1, 1],
                  [0, 1, 0, 1],
                  [0, 1, 1, 0]], dtype=float)
d_hub = A_hub.sum(axis=1)
D_hub = np.diag(d_hub)
W_hub = A_hub / d_hub[:, None]
print("degrees:", d_hub)
print("W:\n", np.round(W_hub, 4))
print("row sums:", W_hub.sum(axis=1), " S0:", W_hub.sum(),
      " symmetric:", np.allclose(W_hub, W_hub.T))

D_half_inv = np.diag(d_hub ** -0.5)
D_half = np.diag(d_hub ** 0.5)
S_sym = D_half_inv @ A_hub @ D_half_inv
print("W = D^(-1/2) S D^(1/2):",
      np.allclose(W_hub, D_half_inv @ S_sym @ D_half))
print("eigenvalues of S (symmetric):", np.round(np.linalg.eigvalsh(S_sym), 4))
print("eigenvalues of W:",
      np.round(np.sort(np.linalg.eigvals(W_hub).real), 4))
```

```python
d_rook = A.sum(axis=1)
S_rook = np.diag(d_rook ** -0.5) @ A @ np.diag(d_rook ** -0.5)
lam = np.linalg.eigvalsh(S_rook)
print(f"rook grid: smallest eigenvalue {lam[0]:.4f}, largest {lam[-1]:.4f}")
for name in ("4 nearest", "8 nearest"):
    ev = np.linalg.eigvals(row_standardize(weights[name]))
    print(f"{name}: largest imaginary part {np.abs(ev.imag).max():.4f}")
```

The 12-by-12 rook grid has smallest eigenvalue $-1$ and largest $1$, as the bipartite argument predicts. The nearest-neighbor matrices have complex eigenvalues, with imaginary parts as large as $0.098$ and $0.040$: small, but not zero, and enough to break any argument that needs a real spectrum.

**[STORY NEEDED]** A setting where a reader would care about the largest and smallest eigenvalues of the weights before meeting them in Chapter 7: for example, how strongly a shock to one region can echo through its neighbors and back, and why the answer depends on the map.

---

## 6.4 Moran's I as a Ratio of Quadratic Forms

We want one number that says whether neighboring regions have more similar values than chance would produce. Start from the centered data, $\mathbf{y}_c$ with entries $y_{c,i} = y_i - \bar{y}$, so that high values are positive and low values negative. If neighbors tend to be alike, then $y_{c,i}$ and the neighbors' average $\sum_j w_{ij}y_{c,j}$ tend to have the same sign, and their product is positive. If neighbors tend to differ, the product is negative. Adding the products over all regions gives $\sum_i y_{c,i}\sum_j w_{ij}y_{c,j} = \mathbf{y}_c'\mathbf{W}\mathbf{y}_c$. Dividing by $\mathbf{y}_c'\mathbf{y}_c$, the total squared deviation, removes the scale of the data.

For row-standardized weights this ratio has a familiar meaning. The vector $\mathbf{W}\mathbf{y}_c$ holds each region's neighbor average, and

$$
\frac{\mathbf{y}_c'\mathbf{W}\mathbf{y}_c}{\mathbf{y}_c'\mathbf{y}_c}
$$

is the least squares slope of the regression of the neighbor average $\mathbf{W}\mathbf{y}_c$ on the value $\mathbf{y}_c$, with no intercept (the slope formula of Tool 5 for a single predictor through the origin). **Moran's I is the slope of a regression of "what my neighbors look like" on "what I look like."** A slope of $0.6$ means that, on average, a county 10 units above the mean has neighbors about 6 units above the mean. A plot of $\mathbf{W}\mathbf{y}_c$ against $\mathbf{y}_c$ is called a **Moran scatterplot**, and Figure 6.3 shows one.

### 6.4.1 The Centering Matrix and the Centered Data

Chapter 0 introduced the centering matrix $\mathbf{P}_c = \mathbf{I}_n - \tfrac{1}{n}\mathbf{1}\mathbf{1}'$. It is symmetric and idempotent, and it subtracts the average: $\mathbf{P}_c\mathbf{y} = \mathbf{y} - \bar{y}\mathbf{1}$. We write the centered data as $\mathbf{y}_c = \mathbf{P}_c\mathbf{y}$ (observed values) or $\mathbf{P}_c\mathbf{Y}$ (random). Two facts follow from $\mathbf{P}_c\mathbf{1} = \mathbf{0}$ and $\mathbf{P}_c\mathbf{P}_c = \mathbf{P}_c$:

$$
\mathbf{y}_c'\mathbf{y}_c = \mathbf{y}'\mathbf{P}_c\mathbf{P}_c\mathbf{y} = \mathbf{y}'\mathbf{P}_c\mathbf{y}, \qquad \mathbf{y}_c'\mathbf{W}\mathbf{y}_c = \mathbf{y}'\mathbf{P}_c\mathbf{W}\mathbf{P}_c\mathbf{y}
$$

so both the denominator and the numerator of Moran's I are quadratic forms in the original data $\mathbf{y}$.

### 6.4.2 The Symmetrized Central Matrix

Tool 2 and Section 1.1.1 work with quadratic forms $\mathbf{Y}'\mathbf{M}\mathbf{Y}$ in a **symmetric** matrix $\mathbf{M}$, and $\mathbf{P}_c\mathbf{W}\mathbf{P}_c$ is not symmetric. The fix uses the fact that a quadratic form is a number, and a number equals its own transpose:

$$
\mathbf{x}'\mathbf{W}\mathbf{x} = (\mathbf{x}'\mathbf{W}\mathbf{x})' = \mathbf{x}'\mathbf{W}'\mathbf{x} \quad\Longrightarrow\quad \mathbf{x}'\mathbf{W}\mathbf{x} = \mathbf{x}'\mathbf{W}_s\mathbf{x}, \qquad \mathbf{W}_s = \tfrac{1}{2}(\mathbf{W} + \mathbf{W}')
$$

The quadratic form of a matrix depends only on its symmetric part, which we call the **symmetrized weights** $\mathbf{W}_s$. Define

$$
\mathbf{M} = \mathbf{P}_c\mathbf{W}_s\mathbf{P}_c
$$

This matrix is symmetric, and $\mathbf{M}\mathbf{1}_n = \mathbf{0}$ because $\mathbf{P}_c\mathbf{1} = \mathbf{0}$. By the identity just shown, $\mathbf{y}_c'\mathbf{W}\mathbf{y}_c = \mathbf{y}'\mathbf{M}\mathbf{y}$.

### 6.4.3 The Ratio Form

Putting the pieces together, with $S_0 = \sum_i\sum_j w_{ij}$ in general (it equals $n$ for row-standardized weights, which removes the factor):

$$
I = \frac{n}{S_0}\,\frac{\mathbf{y}_c'\mathbf{W}\mathbf{y}_c}{\mathbf{y}_c'\mathbf{y}_c} = \frac{n}{S_0}\,\frac{\mathbf{y}'\mathbf{M}\mathbf{y}}{\mathbf{y}'\mathbf{P}_c\mathbf{y}}
$$

The factor $n/S_0$ makes $I$ comparable across weights matrices whose entries have different sizes.

**By hand on the hub map.** Let the four regions have values $\mathbf{y} = (1, 1, 5, 5)'$: regions 1 and 2 are low, regions 3 and 4 are high. The mean is 3, so $\mathbf{y}_c = (-2, -2, 2, 2)'$ and $\mathbf{y}_c'\mathbf{y}_c = 16$. The neighbor averages are $(\mathbf{W}\mathbf{y}_c)_1 = y_{c,2} = -2$, $(\mathbf{W}\mathbf{y}_c)_2 = (y_{c,1} + y_{c,3} + y_{c,4})/3 = 2/3$, $(\mathbf{W}\mathbf{y}_c)_3 = (y_{c,2} + y_{c,4})/2 = 0$, and $(\mathbf{W}\mathbf{y}_c)_4 = (y_{c,2} + y_{c,3})/2 = 0$. So

$$
\mathbf{y}_c'\mathbf{W}\mathbf{y}_c = (-2)(-2) + (-2)\left(\tfrac{2}{3}\right) + 2(0) + 2(0) = 4 - \tfrac{4}{3} = \tfrac{8}{3}, \qquad I = \frac{4}{4}\cdot\frac{8/3}{16} = \frac{1}{6}
$$

```python
y_hub = np.array([1.0, 1.0, 5.0, 5.0])
z_hub = y_hub - y_hub.mean()
print("z:", z_hub, " Wz:", np.round(W_hub @ z_hub, 4))
print("z'Wz =", round(z_hub @ W_hub @ z_hub, 4), " z'z =", z_hub @ z_hub)
print("Moran's I, hub map:", round(moran_I(y_hub, W_hub), 4))
print("binary weights instead:", round(moran_I(y_hub, A_hub), 4))
```

The same value comes out of the matrix $\mathbf{M}$. Its entries, rounded, are printed below, and $\mathbf{y}'\mathbf{M}\mathbf{y} = 8/3$, $\mathbf{y}'\mathbf{P}_c\mathbf{y} = 16$:

```python
n_hub = 4
P_c = np.eye(n_hub) - np.ones((n_hub, n_hub)) / n_hub
W_sym = (W_hub + W_hub.T) / 2
M_hub = P_c @ W_sym @ P_c
print("M:\n", np.round(M_hub, 4))
print("M 1 =", np.round(M_hub @ np.ones(n_hub), 12))
num = y_hub @ M_hub @ y_hub
den = y_hub @ P_c @ y_hub
print("Y'MY =", round(num, 4), " Y'P_cY =", den,
      " I =", round(n_hub / W_hub.sum() * num / den, 4))
print("eigenvalues of the symmetrized W:",
      np.round(np.linalg.eigvalsh(W_sym), 4))
```

The output includes $\mathbf{M}\mathbf{1} = \mathbf{0}$, so adding a constant to the data cannot change $I$.

Binary weights give a different answer for the same data: $I = 0$, not $1/6$. With the binary matrix $\mathbf{A}$, the products over the four edges are $4$ for the edge between regions 1 and 2, $-4$ for each of the edges 2-3 and 2-4, and $4$ for the edge 3-4, and they cancel. The weights are part of the definition of $I$, as Section 6.2.4 said.

### 6.4.4 Reading the Value of I

Under no spatial structure, $I$ is near $-1/(n-1)$ (derived in Section 6.5), which is slightly below zero and close to zero for large $n$. Positive values mean that neighbors resemble each other, and negative values mean that neighbors tend to differ.

The value of $I$ is not bounded by $\pm 1$ in general. It is a ratio of the form $\mathbf{y}_c'\mathbf{W}_s\mathbf{y}_c/\mathbf{y}_c'\mathbf{y}_c$, and the eigenvector argument of Section 1.3.4 applies. Write $\mathbf{W}_s = \sum_k\lambda_k\mathbf{v}_k\mathbf{v}_k'$ and expand $\mathbf{y}_c = \sum_k c_k\mathbf{v}_k$. Then the ratio is $\sum_k\lambda_kc_k^2/\sum_kc_k^2$, a weighted average of the eigenvalues of $\mathbf{W}_s$. So $I$ always lies between the smallest and largest eigenvalue of $\mathbf{W}_s$ (times $n/S_0$ when the weights are not row-standardized). For the hub map the eigenvalues of $\mathbf{W}_s$ run from $-0.812$ to $1.052$, as the output of the block in Section 6.4.3 shows, and for the rook county map from $-1.0023$ to $1.0023$, as the next block prints. The range is slightly wider than $\pm 1$, even for row-standardized weights, because symmetrizing a row-standardized matrix gives up the guarantee that the eigenvalues stay within $\pm 1$. The value $I = 0.615$ for the simulated rates is well inside the range of $I$ for that map, but the range alone does not say whether $0.615$ is large. For that we need a comparison with its null distribution, which Section 6.5 provides.

```python
# Moran's I as a regression slope (row-standardized weights)
z = y_strong - y_strong.mean()
W_rook = row_standardize(A)
Wz = W_rook @ z
print("slope of Wz on z through the origin:", round(z @ Wz / (z @ z), 4))
print("Moran's I, county map:", round(moran_I(y_strong, W_rook), 4))
Ws_rook = (W_rook + W_rook.T) / 2
lam_s = np.linalg.eigvalsh(Ws_rook)
print(f"eigenvalues of the symmetrized rook W: {lam_s[0]:.4f} to "
      f"{lam_s[-1]:.4f}")
```

---

## 6.5 Exact Null Moments of I Under Normality

To decide whether an observed $I$ is large, we compare it with the values $I$ takes when there is no spatial structure. The simplest version of "no spatial structure" is that the observations are independent draws from the same normal distribution:

$$
\mathbf{Y} \sim N(\mu\mathbf{1}_n, \tau^2\mathbf{I}_n)
$$

Here $\tau^2$ is the variance of independent noise, as in Chapter 1: with no spatial structure there is no spatial random effect, so $\sigma^2$ plays no role. We will derive the exact mean and variance of $I$ under this null model. The derivation has three steps: a tool for quadratic forms in normal vectors, the observation that the numerator and the denominator of $I$ can be handled separately, and the calculation itself.

### 6.5.1 A Tool We Need: Quadratic Forms in Normal Vectors

Let $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and let $\mathbf{M}$ be symmetric. By the eigendecomposition of Chapter 1, $\mathbf{M} = \sum_k\lambda_k\mathbf{v}_k\mathbf{v}_k'$ with orthonormal eigenvectors. Then

$$
\mathbf{Z}'\mathbf{M}\mathbf{Z} = \sum_{k=1}^n\lambda_k(\mathbf{v}_k'\mathbf{Z})^2 = \sum_{k=1}^n\lambda_k\,z_k^2
$$

where $z_k = \mathbf{v}_k'\mathbf{Z}$. By Tool 1, the vector of the $z_k$ has covariance $\mathbf{U}'\mathbf{I}_n\mathbf{U} = \mathbf{I}_n$, where $\mathbf{U}$ holds the eigenvectors, so the $z_k$ are independent standard normals (here $z_k$ is a standard normal variable, not the test statistic $z$ used later in this chapter). A quadratic form in a normal vector is a weighted sum of independent squared standard normals, with the eigenvalues as weights. Because $\operatorname{E}[z_k^2] = 1$ and $\operatorname{Var}(z_k^2) = 2$,

$$
\operatorname{E}[\mathbf{Z}'\mathbf{M}\mathbf{Z}] = \sum_k\lambda_k = \operatorname{tr}(\mathbf{M}), \qquad \operatorname{Var}(\mathbf{Z}'\mathbf{M}\mathbf{Z}) = 2\sum_k\lambda_k^2 = 2\operatorname{tr}(\mathbf{M}^2)
$$

where $\operatorname{tr}(\mathbf{M}^2) = \sum_k\lambda_k^2$ because the eigenvalues of $\mathbf{M}^2$ are the squares. (The fact that $\operatorname{Var}(z^2) = 2$ for a standard normal is the same fourth-moment fact, $\operatorname{E}[z^4] = 3$, used in Chapter 3.) For $\mathbf{Y} \sim N(\mathbf{0}, \tau^2\mathbf{I}_n)$, write $\mathbf{Y} = \tau\mathbf{Z}$, so the mean is $\tau^2\operatorname{tr}(\mathbf{M})$ and the variance is $2\tau^4\operatorname{tr}(\mathbf{M}^2)$. The second moment is the variance plus the squared mean:

$$
\operatorname{E}[(\mathbf{Y}'\mathbf{M}\mathbf{Y})^2] = 2\tau^4\operatorname{tr}(\mathbf{M}^2) + \tau^4\operatorname{tr}(\mathbf{M})^2
$$

**A check with two variables.** Take $\mathbf{M} = \begin{bmatrix} 0 & 1/2 \\ 1/2 & 0 \end{bmatrix}$, so that $\mathbf{Z}'\mathbf{M}\mathbf{Z} = Z_1Z_2$. The eigenvalues of $\mathbf{M}$ are $\pm 1/2$, with eigenvectors $(1, 1)'/\sqrt{2}$ and $(1, -1)'/\sqrt{2}$, so $Z_1Z_2 = \tfrac{1}{2}(z_1^2 - z_2^2)$ with $z_1 = (Z_1 + Z_2)/\sqrt{2}$ and $z_2 = (Z_1 - Z_2)/\sqrt{2}$. The formula gives $\operatorname{E}[Z_1Z_2] = \operatorname{tr}(\mathbf{M}) = 0$ and $\operatorname{Var}(Z_1Z_2) = 2\operatorname{tr}(\mathbf{M}^2) = 2(\tfrac{1}{4} + \tfrac{1}{4}) = 1$, which is right, since $Z_1$ and $Z_2$ are independent standard normals and $\operatorname{Var}(Z_1Z_2) = \operatorname{E}[Z_1^2]\operatorname{E}[Z_2^2] = 1$.

```python
rng = np.random.default_rng(6)
Zs = rng.standard_normal((200_000, 2))
prod = Zs[:, 0] * Zs[:, 1]
print("Z1*Z2: mean", round(prod.mean(), 3), " variance", round(prod.var(), 3))
M2 = np.array([[0.0, 0.5], [0.5, 0.0]])
print("tr(M) =", np.trace(M2), " 2 tr(M^2) =", 2 * np.trace(M2 @ M2))
```

### 6.5.2 Independence of I and Its Denominator

Write the numerator and denominator of Moran's I as $q_1 = \mathbf{Y}'\mathbf{M}\mathbf{Y}$ and $q_2 = \mathbf{Y}'\mathbf{P}_c\mathbf{Y}$, so that $I = (n/S_0)\,q_1/q_2$. We want $\operatorname{E}[I]$ and $\operatorname{E}[I^2]$, but the expectation of a ratio is not the ratio of expectations. If, however, $I$ and $q_2$ are independent, then $q_1 = (S_0/n)\,I\,q_2$ gives

$$
\operatorname{E}[q_1^k] = \left(\frac{S_0}{n}\right)^k\operatorname{E}[I^k]\operatorname{E}[q_2^k] \quad\Longrightarrow\quad \operatorname{E}[I^k] = \left(\frac{n}{S_0}\right)^k\frac{\operatorname{E}[q_1^k]}{\operatorname{E}[q_2^k]}
$$

and we can compute the numerator and denominator moments separately. This independence is a property of the normal distribution, sometimes credited to Pitman and Koopmans.

**Why it holds.** First remove the mean. Because $\mathbf{P}_c\mathbf{1} = \mathbf{0}$, both $q_1$ and $q_2$ depend on $\mathbf{Y}$ only through $\mathbf{P}_c\mathbf{Y} = \mathbf{P}_c(\mathbf{Y} - \mu\mathbf{1})$, so the unknown $\mu$ drops out, and we may take $\mu = 0$.

Next, let $\mathbf{U}_{\perp}$ be an $n \times (n-1)$ matrix whose columns are an orthonormal basis of the vectors perpendicular to $\mathbf{1}_n$. Then $\mathbf{P}_c = \mathbf{U}_{\perp}\mathbf{U}_{\perp}'$. The vector $\mathbf{w} = \mathbf{U}_{\perp}'\mathbf{Y}$ has covariance $\tau^2\mathbf{U}_{\perp}'\mathbf{U}_{\perp} = \tau^2\mathbf{I}_{n-1}$ by Tool 1, so $\mathbf{w} \sim N(\mathbf{0}, \tau^2\mathbf{I}_{n-1})$. In terms of $\mathbf{w}$,

$$
q_2 = \mathbf{Y}'\mathbf{U}_{\perp}\mathbf{U}_{\perp}'\mathbf{Y} = \mathbf{w}'\mathbf{w}, \qquad q_1 = \mathbf{w}'\left(\mathbf{U}_{\perp}'\mathbf{W}_s\mathbf{U}_{\perp}\right)\mathbf{w}, \qquad I = \frac{n}{S_0}\,\frac{\mathbf{w}'(\mathbf{U}_{\perp}'\mathbf{W}_s\mathbf{U}_{\perp})\mathbf{w}}{\mathbf{w}'\mathbf{w}}
$$

So $I$ depends on $\mathbf{w}$ only through its direction $\mathbf{w}/\lVert\mathbf{w}\rVert$, while $q_2 = \lVert\mathbf{w}\rVert^2$ is its squared length.

The distribution of $\mathbf{w}$ is unchanged by any rotation, since for a rotation matrix $\mathbf{O}$ the vector $\mathbf{O}\mathbf{w}$ also has covariance $\tau^2\mathbf{I}$. Fix the length $r = \lVert\mathbf{w}\rVert$. Given that length, the direction has a distribution that is itself rotation-invariant, which can only be the uniform distribution on the sphere, whatever $r$ is. A conditional distribution that does not depend on the conditioning value means the direction and the length are independent. So $I$ and $q_2$ are independent.

Section 6.5.5 checks this by simulation on both maps: the correlation between $I$ and the denominator in 20,000 simulated null data sets is $0.009$ on the hub map and $0.008$ on the county map, zero within simulation error.

### 6.5.3 The Mean: E[I] = -1/(n-1)

For the numerator, Section 6.5.1 gives $\operatorname{E}[q_1] = \tau^2\operatorname{tr}(\mathbf{M})$. Cycle the trace, use $\mathbf{P}_c\mathbf{P}_c = \mathbf{P}_c$, and use the fact that $\operatorname{tr}(\mathbf{W}_s) = \operatorname{tr}(\mathbf{W}) = 0$ because the diagonal of $\mathbf{W}$ is zero:

$$
\operatorname{tr}(\mathbf{M}) = \operatorname{tr}(\mathbf{P}_c\mathbf{W}_s\mathbf{P}_c) = \operatorname{tr}(\mathbf{W}_s\mathbf{P}_c) = \operatorname{tr}(\mathbf{W}_s) - \frac{1}{n}\mathbf{1}'\mathbf{W}_s\mathbf{1} = 0 - \frac{S_0}{n}
$$

For the denominator, $\operatorname{E}[q_2] = \tau^2\operatorname{tr}(\mathbf{P}_c) = \tau^2(n-1)$. Therefore

$$
\operatorname{E}[I] = \frac{n}{S_0}\cdot\frac{-\tau^2S_0/n}{\tau^2(n-1)} = -\frac{1}{n-1}
$$

The null mean of Moran's I is not zero. It is slightly negative, and it does not depend on $\mathbf{W}$, on $\mu$, or on $\tau^2$. The reason is that centering forces the deviations to sum to zero, so deviations must be slightly negatively correlated with each other on average. For the hub map the value is $-1/3$, and for the 144-county map $-1/143 \approx -0.007$.

### 6.5.4 The Variance

By Section 6.5.2 and the second moment of Section 6.5.1, with $\operatorname{E}[q_2^2] = \tau^4\left[2(n-1) + (n-1)^2\right] = \tau^4(n-1)(n+1)$, because $q_2/\tau^2$ is a sum of $n-1$ squared standard normals:

$$
\operatorname{E}[I^2] = \left(\frac{n}{S_0}\right)^2\frac{2\operatorname{tr}(\mathbf{M}^2) + \operatorname{tr}(\mathbf{M})^2}{(n-1)(n+1)}, \qquad \operatorname{Var}(I) = \operatorname{E}[I^2] - \frac{1}{(n-1)^2}
$$

**By hand on the hub map.** With $n = 4$ and $S_0 = 4$ the factor $n/S_0$ is 1. From Section 6.4.3, the matrix $\mathbf{M}$ has $\operatorname{tr}(\mathbf{M}) = -1$ (matching $-S_0/n$) and $\operatorname{tr}(\mathbf{M}^2) = \sum_{ij}M_{ij}^2 = 43/48$. Then

$$
\operatorname{E}[I^2] = \frac{2(43/48) + 1}{3 \times 5} = \frac{67}{360}, \qquad \operatorname{Var}(I) = \frac{67}{360} - \frac{1}{9} = \frac{27}{360} = \frac{3}{40} = 0.075
$$

The standard deviation is $\sqrt{0.075} = 0.2739$. The observed $I = 1/6$ for $\mathbf{y} = (1, 1, 5, 5)'$ is $(1/6 + 1/3)/0.2739 = 1.826$ standard deviations above the null mean. The code below confirms these numbers and also gives the mean and standard deviation for the county map, $-0.0070$ and $0.0614$.

```python
def moran_moments(W):
    """Exact mean and sd of Moran's I under Y ~ N(mu 1, tau^2 I)."""
    n = len(W)
    S0 = W.sum()
    P_c = np.eye(n) - np.ones((n, n)) / n
    M = P_c @ ((W + W.T) / 2) @ P_c
    trM, trM2 = np.trace(M), np.trace(M @ M)
    mean = n / S0 * trM / (n - 1)
    second = (n / S0) ** 2 * (2 * trM2 + trM**2) / ((n - 1) * (n + 1))
    return mean, np.sqrt(second - mean**2)


m_hub, s_hub = moran_moments(W_hub)
print(f"hub map: E[I] = {m_hub:.4f}, sd = {s_hub:.4f}, "
      f"z = {(moran_I(y_hub, W_hub) - m_hub) / s_hub:.3f}")
W_rook = row_standardize(A)
m_c, s_c = moran_moments(W_rook)
print(f"county map, rook: E[I] = {m_c:.4f}, sd = {s_c:.4f}")
```

### 6.5.5 Checking the Formulas by Simulation

Moments derived by hand deserve an independent check. The code below simulates 20,000 data sets from $N(3, 2^2)$ on each map (the mean and variance are arbitrary, since neither matters) and computes $I$ for each. It reports the sample mean and standard deviation of $I$, the exact values, and the correlation between $I$ and the denominator.

```python
def moran_I_rows(Y, W):
    """Moran's I for each row of Y."""
    Z = Y - Y.mean(axis=1, keepdims=True)
    return Y.shape[1] / W.sum() * ((Z @ W) * Z).sum(axis=1) / (Z * Z).sum(1)


rng = np.random.default_rng(11)
for name, W in (("hub map", W_hub), ("county map", W_rook)):
    Y = 3.0 + 2.0 * rng.standard_normal((20_000, len(W)))
    I_sim = moran_I_rows(Y, W)
    Zc = Y - Y.mean(axis=1, keepdims=True)
    D_sim = (Zc * Zc).sum(axis=1)
    mean_, sd_ = moran_moments(W)
    print(f"{name}: simulated mean {I_sim.mean():.4f}, sd {I_sim.std():.4f}"
          f" | exact {mean_:.4f}, {sd_:.4f}"
          f" | corr(I, sum of squares) {np.corrcoef(I_sim, D_sim)[0, 1]:.3f}")
```

On the hub map the simulated mean is $-0.3342$ and standard deviation $0.2738$, against exact values $-0.3333$ and $0.2739$. On the county map they are $-0.0076$ and $0.0613$ against $-0.0070$ and $0.0614$. The correlations between $I$ and the denominator are $0.009$ and $0.008$.

The null standard deviation depends strongly on the weights and the map, as the earlier table showed: on the county map it is $0.061$ for rook weights, $0.044$ for queen, and $0.023$ for inverse distance. A wider neighborhood averages more values, so $I$ varies less under the null.

### 6.5.6 The Textbook Formula for the Variance (Optional)

*This subsection shows that the variance above equals the formula found in most textbooks and software, which is written with three sums of the weights. It can be skipped on a first reading.*

Define $S_1 = \tfrac{1}{2}\sum_i\sum_j(w_{ij} + w_{ji})^2$ and $S_2 = \sum_i(w_{i\cdot} + w_{\cdot i})^2$, where $w_{i\cdot}$ and $w_{\cdot i}$ are the row sum and column sum for region $i$. The textbook result under normality is

$$
\operatorname{Var}(I) = \frac{n^2S_1 - nS_2 + 3S_0^2}{S_0^2(n^2 - 1)} - \frac{1}{(n-1)^2}
$$

To derive it from Section 6.5.4, expand $\operatorname{tr}(\mathbf{M}^2) = \operatorname{tr}(\mathbf{P}_c\mathbf{W}_s\mathbf{P}_c\mathbf{W}_s)$ using $\mathbf{P}_c = \mathbf{I}_n - \tfrac{1}{n}\mathbf{1}\mathbf{1}'$:

$$
\operatorname{tr}(\mathbf{M}^2) = \operatorname{tr}(\mathbf{W}_s^2) - \frac{2}{n}\mathbf{1}'\mathbf{W}_s^2\mathbf{1} + \frac{1}{n^2}\left(\mathbf{1}'\mathbf{W}_s\mathbf{1}\right)^2
$$

The three terms are, in order: $\operatorname{tr}(\mathbf{W}_s^2) = \sum_{ij}(w^s_{ij})^2 = S_1/2$ (since $w^s_{ij} = (w_{ij} + w_{ji})/2$); $\mathbf{1}'\mathbf{W}_s^2\mathbf{1} = \sum_i(\text{row sum of }\mathbf{W}_s)^2 = \sum_i\left(\tfrac{w_{i\cdot} + w_{\cdot i}}{2}\right)^2 = S_2/4$; and $\mathbf{1}'\mathbf{W}_s\mathbf{1} = S_0$. So $\operatorname{tr}(\mathbf{M}^2) = S_1/2 - S_2/(2n) + S_0^2/n^2$. With $\operatorname{tr}(\mathbf{M}) = -S_0/n$,

$$
2\operatorname{tr}(\mathbf{M}^2) + \operatorname{tr}(\mathbf{M})^2 = S_1 - \frac{S_2}{n} + \frac{3S_0^2}{n^2}
$$

and multiplying by $(n/S_0)^2/(n^2 - 1)$ gives $\operatorname{E}[I^2] = (n^2S_1 - nS_2 + 3S_0^2)/(S_0^2(n^2-1))$, which is the textbook formula once $-1/(n-1)^2$ is subtracted.

```python
def moran_var_classic(W):
    """The textbook variance of I under normality, from S0, S1, and S2."""
    n = len(W)
    S0 = W.sum()
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(axis=0) + W.sum(axis=1)) ** 2).sum()
    return (n**2 * S1 - n * S2 + 3 * S0**2) / (S0**2 * (n**2 - 1)) \
        - 1 / (n - 1) ** 2


for name, W in (("hub map", W_hub), ("county map", W_rook)):
    print(name, round(moran_var_classic(W), 6),
          round(moran_moments(W)[1] ** 2, 6))
```

The two expressions agree to six decimals on both maps: $0.075$ for the hub map and $0.003769$ for the county map. With the sums $S_1$ and $S_2$ in hand, software can compute the null variance without any matrix products, which matters for very large maps.

---

## 6.6 Moran's I on Regression Residuals

In practice Moran's I is more often applied to the residuals of a regression than to a raw response. The question changes. For the raw rates, "Do counties resemble their neighbors?" is partly answered by the covariates: counties with similar deprivation tend to be neighbors, so their rates are similar for reasons the regression already explains. For the residuals, the question is "After accounting for deprivation, do counties still resemble their neighbors?" A positive answer means the regression is missing spatial structure, and that is the diagnostic used before and after the models of Chapters 7 and 8.

### 6.6.1 The Same Derivation With a Different Projection

Let $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with independent normal errors of variance $\tau^2$, and let $\mathbf{e} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{Y}$ be the OLS residuals. Moran's I of the residuals is

$$
I_e = \frac{n}{S_0}\,\frac{\mathbf{e}'\mathbf{W}\mathbf{e}}{\mathbf{e}'\mathbf{e}}
$$

Residuals from a model with an intercept have mean zero, so no separate centering is needed. Every step of Section 6.5 goes through with $\mathbf{P}_c$ replaced by $\mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$. Define

$$
\mathbf{M}_{\mathbf{X}} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\,\mathbf{W}_s\,(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})
$$

Because $(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{X} = \mathbf{0}$, both $\mathbf{e}'\mathbf{W}\mathbf{e} = \mathbf{Y}'\mathbf{M}_{\mathbf{X}}\mathbf{Y}$ and $\mathbf{e}'\mathbf{e} = \mathbf{Y}'(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{Y}$ are free of $\boldsymbol{\beta}$, and they depend on $\boldsymbol{\varepsilon}$ only. The matrix $\mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$ is symmetric, idempotent, and has trace $n - p$, so the independence argument of Section 6.5.2 works in the $(n-p)$-dimensional space of vectors perpendicular to the columns of $\mathbf{X}$. The denominator is $\tau^2$ times a sum of $n - p$ squared standard normals, with $\operatorname{E}[q_2] = \tau^2(n-p)$ and $\operatorname{E}[q_2^2] = \tau^4(n-p)(n-p+2)$. The numerator moments are those of Section 6.5.1, now with $\mathbf{M}_{\mathbf{X}}$. Therefore

$$
\operatorname{E}[I_e] = \frac{n}{S_0}\,\frac{\operatorname{tr}(\mathbf{M}_{\mathbf{X}})}{n - p}, \qquad \operatorname{E}[I_e^2] = \left(\frac{n}{S_0}\right)^2\frac{2\operatorname{tr}(\mathbf{M}_{\mathbf{X}}^2) + \operatorname{tr}(\mathbf{M}_{\mathbf{X}})^2}{(n-p)(n-p+2)}
$$

and since the diagonal of $\mathbf{W}$ is zero, $\operatorname{tr}(\mathbf{M}_{\mathbf{X}}) = \operatorname{tr}((\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{W}) = -\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\mathbf{W})$. The mean is

$$
\operatorname{E}[I_e] = \frac{n}{S_0}\,\frac{\operatorname{tr}\!\left((\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{W}\right)}{n - p} = -\frac{n}{S_0}\,\frac{\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\mathbf{W})}{n - p}
$$

With $p = 1$ and $\mathbf{X} = \mathbf{1}$ these formulas reduce to Section 6.5, since then $\mathbf{P}_{\mathbf{X}} = \tfrac{1}{n}\mathbf{1}\mathbf{1}'$. The mean of $I_e$ is further below zero when the columns of $\mathbf{X}$ are spatially smooth, because a smooth covariate has a large $\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\mathbf{W})$: the regression absorbs spatially smooth structure and leaves residuals that are slightly less alike across neighbors than independent errors would be.

**By hand on the hub map.** Take $\mathbf{X} = [\mathbf{1}\ \ \mathbf{x}]$ with $\mathbf{x} = (0, 1, 2, 3)'$, so $p = 2$. The centered covariate is $\mathbf{x}_c = (-1.5, -0.5, 0.5, 1.5)'$ with $\mathbf{x}_c'\mathbf{x}_c = 5$, and $\mathbf{P}_{\mathbf{X}} = \tfrac{1}{4}\mathbf{1}\mathbf{1}' + \mathbf{x}_c\mathbf{x}_c'/5$. Then $\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\mathbf{W}) = S_0/4 + \mathbf{x}_c'\mathbf{W}\mathbf{x}_c/5$. The neighbor averages of $\mathbf{x}_c$ are $(-0.5,\ 0.5/3,\ 0.5,\ 0)'$, so $\mathbf{x}_c'\mathbf{W}\mathbf{x}_c = 0.75 - 0.0833 + 0.25 + 0 = 0.9167$, and

$$
\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\mathbf{W}) = 1 + 0.1833 = 1.1833, \qquad \operatorname{E}[I_e] = -\frac{1.1833}{2} = -0.5917
$$

```python
def moran_moments_resid(W, X):
    """Exact null mean and sd of Moran's I for the OLS residuals of X."""
    n, p = X.shape
    S0 = W.sum()
    Q = np.eye(n) - X @ np.linalg.solve(X.T @ X, X.T)
    M = Q @ ((W + W.T) / 2) @ Q
    trM, trM2 = np.trace(M), np.trace(M @ M)
    m = n - p
    mean = n / S0 * trM / m
    second = (n / S0) ** 2 * (2 * trM2 + trM**2) / (m * (m + 2))
    return mean, np.sqrt(second - mean**2)


print("p = 1 reproduces Section 6.5:",
      np.allclose(moran_moments_resid(W_hub, np.ones((4, 1))),
                  moran_moments(W_hub)))
X_hub = np.column_stack([np.ones(4), [0.0, 1.0, 2.0, 3.0]])
P_hub = X_hub @ np.linalg.solve(X_hub.T @ X_hub, X_hub.T)
print("tr(P_X W) =", round(np.trace(P_hub @ W_hub), 4))
m2, s2 = moran_moments_resid(W_hub, X_hub)
print(f"hub map, p = 2: E[I] = {m2:.4f}, sd = {s2:.4f}")
```

The code gives the same $1.1833$ and $-0.5917$, with a standard deviation of $0.1187$.

### 6.6.2 An Application to the Overdose Rates

For the simulated rates of Section 6.2.4, regress the rate on the deprivation index and compare Moran's I before and after.

```python
X = np.column_stack([np.ones(n), x])
beta_hat = np.linalg.solve(X.T @ X, X.T @ y_strong)
resid = y_strong - X @ beta_hat
I_y = moran_I(y_strong, W_rook)
I_e = moran_I(resid, W_rook)
m_y, s_y = moran_moments(W_rook)
m_e, s_e = moran_moments_resid(W_rook, X)
print("OLS coefficients:", np.round(beta_hat, 3))
print(f"I of the rates:     {I_y:.3f}  E = {m_y:.3f}  sd = {s_y:.3f}  "
      f"z = {(I_y - m_y) / s_y:.2f}")
print(f"I of the residuals: {I_e:.3f}  E = {m_e:.3f}  sd = {s_e:.3f}  "
      f"z = {(I_e - m_e) / s_e:.2f}")
```

The OLS coefficients are $10.28$ and $1.652$, close to the true $10$ and $1.5$. Moran's I of the raw rates is $0.615$, with $z = 10.13$ against the null mean $-0.007$ and standard deviation $0.061$. For the residuals it is $0.348$, with null mean $-0.011$, standard deviation $0.061$, and $z = 5.86$. Deprivation explains some of the spatial pattern, since $I$ drops from $0.615$ to $0.348$, but a large and highly significant amount remains. By construction, that is the spatial random effect $u$ in the model of Section 2.1, which the regression does not include. Chapters 7 and 8 model it.

---

## 6.7 Geary's C and Local Indicators

Moran's I is built from cross-products of neighbors, $y_{c,i}y_{c,j}$. Two relatives use the same ingredients differently.

### 6.7.1 Geary's C as a Laplacian Quadratic Form

**Geary's $C$** uses squared differences of neighbors instead of products:

$$
C = \frac{n-1}{2S_0}\,\frac{\sum_i\sum_jw_{ij}(y_i - y_j)^2}{\sum_i(y_i - \bar{y})^2}
$$

Where Moran's I looks at how neighbors co-vary, Geary's $C$ looks at how much they differ. $C$ is near 1 when there is no spatial structure, below 1 when neighbors are alike, and above 1 when neighbors tend to differ. Because it is built from raw differences, it responds more strongly to sharp local differences, such as a jump in rates across a state border where a policy changes, than to a gradual regional pattern.

The numerator is a quadratic form in the **graph Laplacian** of the symmetrized weights. Expand the square and write $r_i = w_{i\cdot}$ and $w_{\cdot i}$ for the row and column sums of $\mathbf{W}$:

$$
\sum_i\sum_jw_{ij}(y_i - y_j)^2 = \sum_iy_i^2r_i + \sum_jy_j^2w_{\cdot j} - 2\mathbf{y}'\mathbf{W}\mathbf{y} = 2\,\mathbf{y}'\left[\operatorname{diag}\!\left(\tfrac{r_i + w_{\cdot i}}{2}\right) - \mathbf{W}_s\right]\mathbf{y}
$$

and $\operatorname{diag}((r_i + w_{\cdot i})/2) = \operatorname{diag}(\mathbf{W}_s\mathbf{1}_n) = \mathbf{D}_s$, the degree matrix of the symmetrized weights. So with the Laplacian $\mathbf{D}_s - \mathbf{W}_s$,

$$
C = \frac{n-1}{S_0}\,\frac{\mathbf{y}'(\mathbf{D}_s - \mathbf{W}_s)\mathbf{y}}{\mathbf{y}'\mathbf{P}_c\mathbf{y}}
$$

For a binary symmetric map, $\mathbf{D}_s - \mathbf{W}_s = \mathbf{D} - \mathbf{A}$, and the numerator is the quadratic form $\mathbf{y}'(\mathbf{D} - \mathbf{A})\mathbf{y} = \tfrac{1}{2}\sum_{ij}A_{ij}(y_i - y_j)^2$ that Chapter 8 uses as the smoothness penalty of the intrinsic autoregressive model.

**Its mean is 1.** The Laplacian satisfies $(\mathbf{D}_s - \mathbf{W}_s)\mathbf{1} = \mathbf{0}$, so the independence argument of Section 6.5.2 applies, and $\operatorname{E}[\mathbf{Y}'(\mathbf{D}_s - \mathbf{W}_s)\mathbf{Y}] = \tau^2\operatorname{tr}(\mathbf{D}_s - \mathbf{W}_s) = \tau^2S_0$ (the trace of $\mathbf{D}_s$ is the sum of all weights, and the trace of $\mathbf{W}_s$ is zero). Dividing by $\operatorname{E}[q_2] = \tau^2(n-1)$ gives $\operatorname{E}[C] = 1$.

**By hand on the hub map.** With $\mathbf{y} = (1, 1, 5, 5)'$, the weighted sum of squared differences is $\sum_{ij}w_{ij}(y_i - y_j)^2 = 0 + \tfrac{1}{3}(0 + 16 + 16) + \tfrac{1}{2}(16 + 0) + \tfrac{1}{2}(16 + 0) = 80/3 = 26.67$, and $\sum_i(y_i - \bar{y})^2 = 16$. So $C = \tfrac{3}{8}\cdot 26.67/16 = 0.625$.

```python
def geary_C(y, W):
    """Geary's C for data y and a weights matrix W with zero diagonal."""
    z = y - y.mean()
    diff2 = (y[:, None] - y[None, :]) ** 2
    return (len(y) - 1) / (2 * W.sum()) * (W * diff2).sum() / (z @ z)


def local_moran(y, W):
    """Local indicators I_i = z_i (Wz)_i / m2, with m2 = sum(z^2) / n."""
    z = y - y.mean()
    return z * (W @ z) / (z @ z / len(y))


W_s = (W_hub + W_hub.T) / 2
L_s = np.diag(W_s.sum(axis=1)) - W_s
print("Geary's C, hub map:", round(geary_C(y_hub, W_hub), 4))
print("via the Laplacian:  ",
      round((4 - 1) / W_hub.sum() * (y_hub @ L_s @ y_hub)
            / (y_hub @ P_c @ y_hub), 4))
Li = local_moran(y_hub, W_hub)
print("local indicators:", np.round(Li, 4), " sum / S0 =",
      round(Li.sum() / W_hub.sum(), 4))
```

**Relation to Moran's I.** For row-standardized weights $r_i = 1$, and if the column sums are all close to 1 as well, then $\mathbf{D}_s \approx \mathbf{I}_n$ and $C \approx \frac{n-1}{n}(1 - I)$. On the county map, where most column sums are close to 1, the rates give $C = 0.3882$ against the approximation $0.3823$. The two statistics carry nearly the same information when the weights are regular and differ when they are not.

```python
C_county = geary_C(y_strong, W_rook)
approx = (n - 1) / n * (1 - I_y)
print(f"county map: C = {C_county:.4f}, (n-1)/n (1 - I) = {approx:.4f}")
```

### 6.7.2 Local Indicators of Spatial Association

Moran's I is one number for the whole map. It cannot say where the dependence is. **Local indicators of spatial association** (LISA) split it into one number per region. With $m_2 = \sum_iy_{c,i}^2/n$, the local Moran indicator of region $i$ is

$$
I_i = \frac{y_{c,i}\sum_jw_{ij}y_{c,j}}{m_2}
$$

Each $I_i$ is positive when region $i$ and its neighbors are on the same side of the mean (both high or both low) and negative when they are on opposite sides. The indicators add up to the global statistic: $\sum_iI_i = n\,\mathbf{y}_c'\mathbf{W}\mathbf{y}_c/\mathbf{y}_c'\mathbf{y}_c$, so $I = \sum_iI_i/S_0$. Each region's indicator is its contribution to $I$.

On the hub map with $\mathbf{y} = (1, 1, 5, 5)'$ we have $m_2 = 16/4 = 4$, so $I_1 = (-2)(-2)/4 = 1$, $I_2 = (-2)(2/3)/4 = -1/3$, and $I_3 = I_4 = 0$. They sum to $2/3$, and $(2/3)/S_0 = 1/6 = I$. Region 1 contributes the most: it is low, and its only neighbor is low.

The sign of $y_{c,i}$ and of the neighbor average sorts each region into a quadrant: high with high neighbors (HH), low with low neighbors (LL), high with low neighbors (HL), and low with high neighbors (LH). HH and LL regions are the **clusters**, and HL and LH are **outliers**.

**Local significance and its trap.** Each $I_i$ can be tested by a conditional permutation: hold $y_{c,i}$ fixed, shuffle the other values among the other regions, and compare $I_i$ with its shuffled values. The code below does this for the 144 counties with 99 permutations each, using a two-sided p-value, and then repeats the exercise on 100 maps of independent noise, to see what the tests do when there is nothing to find.

```python
neighbors = [np.flatnonzero(A[i]) for i in range(n)]


def local_pvalues(y, W, rng, B=99):
    """Two-sided conditional permutation p-values for local indicators."""
    z = y - y.mean()
    m2 = z @ z / len(y)
    Li = z * (W @ z) / m2
    p = np.empty(len(y))
    for i in range(len(y)):
        others = np.delete(z, i)
        draws = rng.permuted(np.tile(others, (B, 1)), axis=1)
        sim = z[i] * draws[:, :len(neighbors[i])].mean(axis=1) / m2
        upper = (1 + np.sum(sim >= Li[i])) / (B + 1)
        lower = (1 + np.sum(sim <= Li[i])) / (B + 1)
        p[i] = min(1.0, 2 * min(upper, lower))
    return p, Li


rng = np.random.default_rng(21)
p_loc, Li_county = local_pvalues(y_strong, W_rook, rng)
print("counties flagged at 5%:", int(np.sum(p_loc <= 0.05)),
      " (expected by chance alone: about", round(0.04 * n, 1), ")")

flagged = []
for _ in range(100):
    y0 = rng.standard_normal(n)          # no spatial structure at all
    flagged.append(np.sum(local_pvalues(y0, W_rook, rng)[0] <= 0.05))
flagged = np.array(flagged)
print(f"100 maps with no structure: mean flagged {flagged.mean():.2f}, "
      f"max {flagged.max()}, maps with at least one: "
      f"{np.mean(flagged >= 1):.2f}")
```

For the simulated rates, 26 of the 144 counties are flagged at the 5% level: 11 high-high and 15 low-low. In each of 100 maps of pure noise with no spatial structure whatever, the number flagged averaged $5.63$, with a maximum of $15$, and 98% of those maps had at least one flagged county. (With 99 shuffles the two-sided p-value is at most 0.05 only when the observed value is among the two most extreme on one side of the 100 values, which has probability $2 \times 2/100 = 0.04$, so the expected count is $0.04 \times 144 = 5.8$ rather than $0.05 \times 144 = 7.2$.) A single flagged county is therefore weak evidence. The 26 flagged in the simulated rates are far more than noise would give, but which individual counties are real is much less certain. With 144 tests per map, a correction for multiple testing (Bonferroni, or control of the false discovery rate) is needed before a flagged region is called a hot spot. Figure 6.4 in the lab shows where the flagged counties lie.

---

## 6.8 Distribution-Free Inference: Permutation Tests

The null moments of Section 6.5 assume normal data. A **permutation test** assumes less. It asks whether the observed arrangement of the values over the regions is unusual among all the ways the same values could have been arranged.

### 6.8.1 The Idea, and Why the p-Value Is Valid

If the values carry no spatial information, then which value sits in which region is arbitrary: every one of the $n!$ rearrangements is equally likely. This is the null hypothesis of **exchangeability**. Moran's I computed on each rearrangement gives a reference distribution, and the p-value is the share of rearrangements with an $I$ at least as large as the observed one.

Computing all $n!$ values is impossible for $n = 144$, so we compute $I$ for $B$ random rearrangements instead. The Monte Carlo p-value is

$$
p = \frac{1 + \#\{b : I^{(b)} \ge I_{\text{obs}}\}}{B + 1}
$$

The $+1$ in both places counts the observed arrangement as one of the $B + 1$. This makes the test valid: under exchangeability, the observed $I$ and the $B$ shuffled values are $B + 1$ exchangeable draws, so the rank of the observed value is uniform on $1, \dots, B+1$, and $P(p \le k/(B+1)) \le k/(B+1)$. A p-value of exactly zero is impossible; with $B = 999$ the smallest value is $0.001$.

### 6.8.2 The Mean of the Permutation Distribution

The mean of the permutation distribution can be computed exactly, and it equals the normal-theory mean. Fix the centered values $y_{c,1}, \dots, y_{c,n}$. Under a random rearrangement, the pair of values $(y_{c,i}, y_{c,j})$ at two different regions is a uniformly random ordered pair of different values, so $\operatorname{E}[y_{c,i}y_{c,j}] = \frac{1}{n(n-1)}\sum_{k \ne l}y_{c,k}y_{c,l}$. Since the $y_{c,k}$ sum to zero, $\sum_{k \ne l}y_{c,k}y_{c,l} = (\sum_ky_{c,k})^2 - \sum_ky_{c,k}^2 = -\sum_ky_{c,k}^2$, so $\operatorname{E}[y_{c,i}y_{c,j}] = -\sum_ky_{c,k}^2/(n(n-1))$. Because $w_{ii} = 0$, the numerator of $I$ has mean $S_0\cdot\operatorname{E}[y_{c,i}y_{c,j}]$, and

$$
\operatorname{E}_{\text{perm}}[I] = \frac{n}{S_0}\cdot\frac{S_0\cdot\left(-\sum_ky_{c,k}^2/(n(n-1))\right)}{\sum_ky_{c,k}^2} = -\frac{1}{n-1}
$$

for every weights matrix and every data set.

### 6.8.3 Exact Enumeration on Four Regions

For the hub map with $\mathbf{y} = (1, 1, 5, 5)'$ we can list all $4! = 24$ rearrangements. Because two values are 1 and two are 5, there are only $\binom{4}{2} = 6$ distinct arrangements, each arising from 4 permutations.

```python
import itertools

arrangements = {}
for perm in itertools.permutations(range(4)):
    arrangements.setdefault(tuple(y_hub[list(perm)]), []).append(perm)
print("distinct arrangements:", len(arrangements))
exact = []
for arrangement, perms in arrangements.items():
    value = moran_I(np.array(arrangement), W_hub)
    exact.extend([value] * len(perms))
    print(arrangement, " I =", round(value, 4), " permutations:", len(perms))
exact = np.array(exact)
print("mean over the 24 permutations:", round(exact.mean(), 4),
      " variance:", round(exact.var(), 4))
print("exact p-value, P(I >= 1/6):", np.mean(exact >= 1 / 6 - 1e-12))
```

| Arrangement of $(y_1, y_2, y_3, y_4)$ | $I$ | Permutations |
| :--- | :--- | :--- |
| (1, 1, 5, 5) | $+0.1667$ | 4 |
| (5, 5, 1, 1) | $+0.1667$ | 4 |
| (1, 5, 1, 5) | $-0.5833$ | 4 |
| (1, 5, 5, 1) | $-0.5833$ | 4 |
| (5, 1, 1, 5) | $-0.5833$ | 4 |
| (5, 1, 5, 1) | $-0.5833$ | 4 |

Two arrangements give $I = 1/6$, the observed value, so $P(I \ge 1/6) = 8/24 = 1/3$. The mean over the 24 permutations is $-1/3$, as Section 6.8.2 proved. The variance over permutations is $1/8 = 0.125$, which is not the normal-theory variance $3/40 = 0.075$. The two reference distributions answer different questions: the permutation distribution conditions on these particular values, and the normal-theory distribution averages over normal data.

The permutation answer is also a reminder about small maps. The observed arrangement is among the two most clustered of the six, but even so its exact p-value is $1/3$. With four regions, no arrangement can give a p-value below $1/3$ here. A map this small cannot provide evidence of spatial structure, and the hub map is useful only for checking mechanics. The z-score of $1.83$ from Section 6.5.4 would suggest a one-sided p-value near $0.034$; with $n = 4$ the normal approximation to the null distribution is poor.

### 6.8.4 Monte Carlo Inference on the County Map

For the 144 counties the test uses 999 random rearrangements of the rates. The code below also returns the permutation values for use in Figure 6.3.

```python
def permutation_test(y, W, rng, B=999):
    """Upper-tail permutation p-value for Moran's I, with the +1 rule."""
    z = y - y.mean()
    Z = rng.permuted(np.tile(z, (B, 1)), axis=1)
    I_perm = len(y) / W.sum() * ((Z @ ((W + W.T) / 2)) * Z).sum(axis=1) \
        / (Z * Z).sum(axis=1)
    I_obs = moran_I(y, W)
    return I_obs, I_perm, (1 + np.sum(I_perm >= I_obs)) / (B + 1)


rng = np.random.default_rng(31)
I_obs, I_perm, p_perm = permutation_test(y_strong, W_rook, rng)
print(f"observed I {I_obs:.4f}; permutation mean {I_perm.mean():.4f}, "
      f"sd {I_perm.std():.4f}; p = {p_perm:.4f}")
print(f"normal theory: mean {m_c:.4f}, sd {s_c:.4f}; "
      f"z-test p = {stats.norm.sf((I_obs - m_c) / s_c):.2e}")
```

The observed $I$ is $0.6150$. The 999 permuted values have mean $-0.0095$ and standard deviation $0.0617$, close to the normal-theory $-0.0070$ and $0.0614$. The p-value is $0.0010$, the smallest possible with 999 permutations, since no shuffled value came close. The normal-theory z-test gives a p-value of $2 \times 10^{-24}$, which is not a number to take literally: it comes from extrapolating a normal tail $10$ standard deviations out. The conclusion is the same, and the permutation p-value is the honest statement of what 999 shuffles can show.

### 6.8.5 When Permuting the Data Gives the Wrong Answer

A permutation test is valid only when the values are exchangeable under the null hypothesis. Two common situations break that assumption.

**Skewed data are not a problem, but unequal variances are.** The code below measures the actual rejection rate of the 5% tests on 1,000 data sets with no spatial structure, for four ways of generating the data. The first three have independent, identically distributed values: normal, lognormal with a very long right tail, and Student's $t$ with 2 degrees of freedom, which has infinite variance. The fourth has independent values whose standard deviation varies across the map, $\exp(1.0\times\text{deprivation})$, as if rates were noisier in more deprived counties (for instance because those counties are smaller).

```python
def size_experiment(draw, W, rng, reps=1000, B=199):
    """Rejection rates at 5% for the z-test and the permutation test."""
    mean_, sd_ = moran_moments(W)
    z_reject = p_reject = 0
    for _ in range(reps):
        y0 = draw()
        _, _, p = permutation_test(y0, W, rng, B)
        z_reject += (moran_I(y0, W) - mean_) / sd_ > stats.norm.isf(0.05)
        p_reject += p <= 0.05
    return z_reject / reps, p_reject / reps


rng = np.random.default_rng(41)
# rates are noisier where deprivation is high
sd_field = np.exp(1.0 * x)
scenarios = {
    "normal": lambda: rng.standard_normal(n),
    "lognormal, sdlog 1.5": lambda: np.exp(1.5 * rng.standard_normal(n)),
    "t, 2 degrees of freedom": lambda: rng.standard_t(2, n),
    "unequal variances": lambda: sd_field * rng.standard_normal(n),
}
print(f"{'no spatial structure, 1000 data sets':38s}{'z-test':>8s}"
      f"{'permutation':>13s}")
for name, draw in scenarios.items():
    zr, pr = size_experiment(draw, W_rook, rng)
    print(f"{name:38s}{zr:8.3f}{pr:13.3f}")
```

| No spatial structure, 1,000 data sets | z-test | Permutation test |
| :--- | :--- | :--- |
| Normal | 0.043 | 0.046 |
| Lognormal, sdlog 1.5 | 0.051 | 0.049 |
| $t$, 2 degrees of freedom | 0.042 | 0.058 |
| Unequal variances | 0.262 | 0.288 |

For identically distributed values, both tests hold their size, even for data far from normal; Moran's I on 144 regions is robust to the shape of the distribution. For unequal variances, both tests are badly wrong, rejecting 26% and 29% of the time when 5% is intended. The values are independent, so there is no spatial dependence to find, yet the tests find it. The reason is that the variances are spatially smooth: the counties with large variances are near each other, and their large deviations multiply into large products $y_{c,i}y_{c,j}$ for neighboring pairs. Shuffling the values moves the large deviations to random places and destroys exactly the clustering of variance that created the large $I$.

The lesson, which Chapter 9 acts on, is that rates computed from counts in regions of very different sizes are not exchangeable. Dividing each region's value by its standard deviation before computing $I$, or modeling the counts directly, restores the assumption.

**Residuals are not exchangeable.** Permuting the residuals of a regression to test their Moran's I looks like the natural extension. It needs care. Even when the errors are independent with equal variances, the OLS residuals are not: $\operatorname{Var}(\mathbf{e}) = \tau^2(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})$ has unequal diagonal entries and nonzero off-diagonal entries, and the residuals satisfy $\mathbf{X}'\mathbf{e} = \mathbf{0}$. A random rearrangement of the residuals destroys that orthogonality, and the rearranged vector is no longer a possible residual vector. In particular, the mean of Moran's I over rearrangements is $-1/(n-1)$ by Section 6.8.2, whereas the true null mean of $I_e$ from Section 6.6.1 is $-\frac{n}{S_0}\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\mathbf{W})/(n-p)$, which can be much more negative.

A repair, due to Freedman and Lane, projects each rearranged residual vector back: compute $(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{e}_\pi$ for the rearranged residuals $\mathbf{e}_\pi$, and then compute $I$. The projected vector satisfies the orthogonality, and its reference distribution has the right mean. The code below compares the two references with the exact formula on 1,000 null data sets, for a regression on the deprivation index ($p = 2$) and for a regression on a quartic trend surface in the coordinates ($p = 15$).

```python
def residual_experiment(X, W, rng, reps=1000, B=199):
    """Null behavior of Moran's I on OLS residuals, three reference sets."""
    n, p = X.shape
    S0, Ws = W.sum(), (W + W.T) / 2
    Q = np.eye(n) - X @ np.linalg.solve(X.T @ X, X.T)
    mean_, sd_ = moran_moments_resid(W, X)

    def I_rows(V):
        return n / S0 * ((V @ Ws) * V).sum(axis=1) / (V * V).sum(axis=1)

    I_true, naive_mean, fl_mean = [], [], []
    rej = np.zeros(3)
    for _ in range(reps):
        e = Q @ rng.standard_normal(n)           # residuals of a null data set
        I_e = I_rows(e[None, :])[0]
        perm = rng.permuted(np.tile(e, (B, 1)), axis=1)
        I_naive = I_rows(perm)                   # permute the residuals
        I_fl = I_rows(perm @ Q)                  # project them back (see text)
        I_true.append(I_e)
        naive_mean.append(I_naive.mean())
        fl_mean.append(I_fl.mean())
        rej += [(I_e - mean_) / sd_ > stats.norm.isf(0.05),
                (1 + np.sum(I_naive >= I_e)) / (B + 1) <= 0.05,
                (1 + np.sum(I_fl >= I_e)) / (B + 1) <= 0.05]
    return (mean_, sd_, np.mean(I_true), np.std(I_true), np.mean(naive_mean),
            np.mean(fl_mean), rej / reps)


cx, cy = S[:, 0] - 0.5, S[:, 1] - 0.5
X_quartic = np.column_stack([cx**i * cy**j for i in range(5)
                             for j in range(5 - i)])
rng = np.random.default_rng(51)
for name, X in (("p = 2, deprivation", np.column_stack([np.ones(n), x])),
                ("p = 15, quartic trend", X_quartic)):
    out = residual_experiment(X, W_rook, rng)
    print(f"{name}: exact E {out[0]:.4f}, sd {out[1]:.4f}; simulated "
          f"E {out[2]:.4f}, sd {out[3]:.4f}")
    print(f"   mean of the permutation reference: naive {out[4]:.4f}, "
          f"projected {out[5]:.4f}")
    print(f"   rejection rates at 5%: z-test {out[6][0]:.3f}, "
          f"naive {out[6][1]:.3f}, projected {out[6][2]:.3f}")
```

| Regression | Exact null mean of $I_e$ | Mean of the naive reference | Mean of the projected reference | Rejection rates at 5%: z-test, naive, projected |
| :--- | :--- | :--- | :--- | :--- |
| $p = 2$ | $-0.0110$ | $-0.0071$ | $-0.0111$ | 0.067, 0.058, 0.066 |
| $p = 15$ | $-0.0950$ | $-0.0071$ | $-0.0951$ | 0.060, 0.000, 0.058 |

With the single covariate, the mean of the naive reference, $-0.0071$, is close enough to the exact $-0.0110$ that the naive test behaves acceptably. With fifteen columns the true null mean is $-0.0950$, but the naive reference is still centered at $-0.0071$. The observed $I_e$ is compared with a reference that is too high, and the naive test rejects in 0.0% of the null data sets: its p-values are far too large, and it would miss spatial structure that the projected test finds. The projected reference has mean $-0.0951$ and rejects 5.8% of the time. The z-test from the exact moments is also close to its nominal size.

---

## 6.9 Walkthrough and Lab 6

### 6.9.1 The Hub Map Collected

The hand calculations of this chapter all use the hub map and the data $\mathbf{y} = (1, 1, 5, 5)'$. The following table collects them next to the code that checked each one.

| Quantity | By hand | Code |
| :--- | :--- | :--- |
| $\mathbf{W}$, $S_0$ | Section 6.3.2 | `W_hub`, 4.0 |
| Eigenvalues of $\mathbf{W}$ | $1$, $-\tfrac{1}{2}$, $(-3 \pm \sqrt{33})/12$ | $-0.7287$, $-0.5$, $0.2287$, $1$ |
| $I$ (row-standardized) | $1/6$ | $0.1667$ |
| $I$ (binary) | $0$ | $0.0$ |
| $\operatorname{tr}(\mathbf{M})$, $\operatorname{tr}(\mathbf{M}^2)$ | $-1$, $43/48$ | $-1.0$, $0.8958$ |
| $\operatorname{E}[I]$, $\operatorname{Var}(I)$ | $-1/3$, $3/40$ | $-0.3333$, $0.075$ |
| $C$ | $0.625$ | $0.625$ |
| Local indicators | $1$, $-1/3$, $0$, $0$ | same |
| Exact permutation $p$ | $1/3$ | $0.3333$ |
| $\operatorname{E}[I_e]$ with $\mathbf{x} = (0, 1, 2, 3)'$ | $-71/120 = -0.5917$ | $-0.5917$ |

### 6.9.2 Weights and Conclusions on the County Map

Section 6.2.4 left a question: for the two simulated responses, do the weights change the conclusion? The code below computes, for each weights matrix, Moran's I, its exact null mean and standard deviation, the z-score, and a permutation p-value with 999 shuffles.

```python
print(f"{'weights':18s}{'I':>7s}{'E[I]':>8s}{'sd':>7s}{'z':>7s}{'perm p':>9s}")
rng = np.random.default_rng(61)
for label, y0 in (("strong response", y_strong), ("weak response", y_weak)):
    print(label)
    for name, B in weights.items():
        W = row_standardize(B)
        I0, _, p0 = permutation_test(y0, W, rng)
        m0, s0 = moran_moments(W)
        print(f"  {name:16s}{I0:7.3f}{m0:8.3f}{s0:7.3f}"
              f"{(I0 - m0) / s0:7.2f}{p0:9.3f}")
```

| Weights | $I$ | $\operatorname{E}[I]$ | sd | $z$ | Permutation $p$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *Strong response* | | | | | |
| Rook | 0.615 | $-0.007$ | 0.061 | 10.13 | 0.001 |
| Queen | 0.596 | $-0.007$ | 0.044 | 13.61 | 0.001 |
| Band, 2.5 cells | 0.454 | $-0.007$ | 0.028 | 16.41 | 0.001 |
| 4 nearest | 0.609 | $-0.007$ | 0.057 | 10.84 | 0.001 |
| 8 nearest | 0.553 | $-0.007$ | 0.040 | 14.03 | 0.001 |
| Inverse distance | 0.413 | $-0.007$ | 0.023 | 18.08 | 0.001 |
| *Weak response* | | | | | |
| Rook | 0.067 | $-0.007$ | 0.061 | 1.21 | 0.121 |
| Queen | 0.063 | $-0.007$ | 0.044 | 1.57 | 0.059 |
| Band, 2.5 cells | 0.077 | $-0.007$ | 0.028 | 2.98 | 0.007 |
| 4 nearest | 0.071 | $-0.007$ | 0.057 | 1.37 | 0.095 |
| 8 nearest | 0.072 | $-0.007$ | 0.040 | 1.97 | 0.045 |
| Inverse distance | 0.055 | $-0.007$ | 0.023 | 2.67 | 0.014 |

For the strong response, every weights matrix leads to the same conclusion, with z-scores from 10 to 18. The $I$ values differ, from $0.413$ to $0.615$, but the sizes of the differences are matched by the null standard deviations, which is why the z-scores are the right thing to compare.

For the weak response, the conclusion depends on the weights. At the 5% level, the 2.5-cell band, the 8 nearest neighbors, and inverse distance find the pattern ($p = 0.007$, $0.045$, $0.014$), while rook, queen, and 4 nearest neighbors do not ($p = 0.121$, $0.059$, $0.095$). The true spatial effect has a range of 0.05, about half a county, so correlation with adjacent counties is weak and correlation with counties two cells away is weaker still. Averaging over a wider neighborhood adds counties with almost no correlation, and also reduces the null standard deviation of $I$ from $0.061$ to $0.028$. In this realization, the second effect wins for the wider bands. In another realization the order could differ. An analyst who tried all six and reported only the one with $p < 0.05$ would be reporting a result that the data do not support, and the honest report is the whole table.

Figure 6.3 shows the Moran scatterplot for the strong response with rook weights, whose fitted slope is Moran's I, and the permutation reference distribution against the observed value.

![Figure 6.3. Left: the Moran scatterplot of the strong response, with a fitted line whose slope is Moran's I. Right: the permutation reference distribution of I, with the observed value as the vertical line far to the right.](../figures/ch06/fig_6_3_moran.png)

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch06", exist_ok=True)
z = y_strong - y_strong.mean()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].scatter(z, W_rook @ z, s=14, alpha=0.7)
grid_z = np.array([z.min(), z.max()])
axes[0].plot(grid_z, I_obs * grid_z, color="black")
axes[0].axhline(0, color="gray", lw=0.5)
axes[0].axvline(0, color="gray", lw=0.5)
axes[0].set_xlabel("centered rate (rate minus the average)")
axes[0].set_ylabel("average of the neighbors' centered rates")
axes[0].set_title(f"Moran scatterplot, slope = I = {I_obs:.3f}")
axes[1].hist(I_perm, bins=30, color="lightgray", edgecolor="gray")
axes[1].axvline(I_obs, color="black")
axes[1].set_xlabel("Moran's I")
axes[1].set_ylabel("permutations")
axes[1].set_title("Permutation reference distribution")
fig.tight_layout()
fig.savefig("figures/ch06/fig_6_3_moran.png", dpi=120)
```

Figure 6.4 shows the simulated rates and the counties flagged by the local test, colored by quadrant.

![Figure 6.4. Left: the simulated overdose rates. Right: the counties flagged at 5% by the local test, colored by quadrant. Eleven are high-high (dark red) and fifteen are low-low (dark blue). With 144 tests, a few of these would be flagged by chance alone.](../figures/ch06/fig_6_4_local.png)

```python
z = y_strong - y_strong.mean()
Wz = W_rook @ z
quadrant = np.where(z > 0, np.where(Wz > 0, "HH", "HL"),
                    np.where(Wz > 0, "LH", "LL"))
colors = {"HH": "firebrick", "LL": "navy", "HL": "orange", "LH": "skyblue"}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
im = axes[0].imshow(y_strong.reshape(12, 12), origin="lower",
                    extent=(0, 1, 0, 1), cmap="viridis")
fig.colorbar(im, ax=axes[0], label="deaths per 10,000 (simulated)")
axes[0].set_title("Simulated rates")
axes[1].imshow(np.ones((12, 12)), origin="lower", extent=(0, 1, 0, 1),
               cmap="gray", vmin=0, vmax=2)
for i in np.flatnonzero(p_loc <= 0.05):
    axes[1].add_patch(plt.Rectangle((S[i, 0] - 1 / 24, S[i, 1] - 1 / 24),
                                    1 / 12, 1 / 12,
                                    color=colors[quadrant[i]]))
axes[1].set_title("Counties flagged by the local test")
for q, c in colors.items():
    axes[1].plot([], [], "s", color=c, label=q)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4,
               fontsize=8)
for ax in axes:
    ax.set_xlabel("east")
    ax.set_ylabel("north")
fig.tight_layout()
fig.savefig("figures/ch06/fig_6_4_local.png", dpi=120)
print(dict(zip(*np.unique(quadrant[p_loc <= 0.05], return_counts=True))))
```

The flagged high-high counties lie in the east, and the flagged low-low counties lie in the south-west and the north-west, with one in the east. The pattern follows the smooth regional structure of the data, and the flags are best read as a map of where the pattern is strongest, not as 26 separate findings.

### 6.9.3 A Simulated Study of the Modifiable Areal Unit Problem

The last part of the lab builds the simulation behind Section 6.1.1. The code first defines the fine grid and the matrix that averages cells into units, then simulates the two components, the covariate, and the outcome for 200 worlds.

```python
def fine_grid(m):
    """Centers of an m-by-m grid of cells in the unit square."""
    g = (np.arange(m) + 0.5) / m
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


def aggregation_matrix(m, gr, gc):
    """Average over gr-by-gc blocks of an m-by-m grid (cell = row*m + col).
    Units are numbered like the grid of units: unit = row * (m // gc) + col."""
    cells = np.arange(m * m).reshape(m, m)
    G = np.zeros(((m // gr) * (m // gc), m * m))
    for i in range(m // gr):
        for j in range(m // gc):
            block = cells[i * gr:(i + 1) * gr, j * gc:(j + 1) * gc]
            G[i * (m // gc) + j, block.ravel()] = 1 / (gr * gc)
    return G


m_f = 48
S_f = fine_grid(m_f)
n_f = len(S_f)
dx = S_f[:, 0][:, None] - S_f[:, 0][None, :]
dy = S_f[:, 1][:, None] - S_f[:, 1][None, :]
R_local = np.exp(-np.sqrt(dx**2 + dy**2) / 0.02)
R_region = np.exp(-np.sqrt((dx / 0.6) ** 2 + (dy / 0.2) ** 2))
L_local = np.linalg.cholesky(R_local + 1e-8 * np.eye(n_f))
L_region = np.linalg.cholesky(R_region + 1e-8 * np.eye(n_f))

reps = 200
rng = np.random.default_rng(2050)
a = L_local @ rng.standard_normal((n_f, reps))     # local component
b = L_region @ rng.standard_normal((n_f, reps))    # regional component
e = rng.standard_normal((n_f, reps))               # independent noise
x_cells = a + b                                    # covariate, per cell
y_cells = 1.0 * a - 0.5 * b + e                    # outcome, per cell
```

The second block aggregates each world six ways, computes the slope, the correlation, Moran's I of the outcome (using rook weights on the grid of units), and the predicted slope from the exact variances, and prints the table quoted in Section 6.1.1.

```python
def slope_corr(X, Y):
    """Slope and correlation for each column of X and Y."""
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    sxy, sxx, syy = (Xc * Yc).sum(0), (Xc**2).sum(0), (Yc**2).sum(0)
    return sxy / sxx, sxy / np.sqrt(sxx * syy)


def moran_I_cols(Y, W):
    """Moran's I for each column of Y."""
    Z = Y - Y.mean(axis=0)
    return len(Y) / W.sum() * (Z * (W @ Z)).sum(0) / (Z * Z).sum(0)


def expected_variance(C):
    """Expected sample variance across the units, for covariance matrix C."""
    U = len(C)
    return np.trace((np.eye(U) - np.ones((U, U)) / U) @ C) / (U - 1)


zonings = {"fine cells (48 x 48)": (1, 1),
           "squares, 12 x 12 units": (4, 4),
           "squares, 6 x 6 units": (8, 8),
           "squares, 3 x 3 units": (16, 16),
           "strips, 48 x 3 units": (1, 16),
           "strips, 3 x 48 units": (16, 1)}
maup = {}
print(f"{'zoning':26s}{'slope':>7s}{'sd':>6s}{'share <0':>9s}"
      f"{'corr':>7s}{'I of y':>8s}{'predicted':>10s}")
for name, (gr, gc) in zonings.items():
    G = aggregation_matrix(m_f, gr, gc)
    X, Y = G @ x_cells, G @ y_cells
    slopes, corrs = slope_corr(X, Y)
    W_u = row_standardize(rook(m_f // gr, m_f // gc))
    I_y = moran_I_cols(Y, W_u)
    if gr == gc == 1:
        Ca, Cb = R_local, R_region
    else:
        Ca, Cb = G @ R_local @ G.T, G @ R_region @ G.T
    va, vb = expected_variance(Ca), expected_variance(Cb)
    predicted = (1.0 * va - 0.5 * vb) / (va + vb)
    maup[name] = (X, Y, slopes, corrs, I_y)
    print(f"{name:26s}{slopes.mean():7.3f}{slopes.std():6.3f}"
          f"{np.mean(slopes < 0):9.2f}{corrs.mean():7.3f}"
          f"{I_y.mean():8.3f}{predicted:10.3f}")
```

The output gives the numbers in the table of Section 6.1.1. The slope falls from $0.383$ at the finest scale to $-0.397$ at nine units, and the strips give $-0.207$ and $-0.045$. The third block draws Figures 6.1 and 6.2.

```python
fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
rng_plot = np.random.default_rng(3)
for ax, name in zip(axes, ["fine cells (48 x 48)", "squares, 12 x 12 units",
                           "squares, 3 x 3 units", "strips, 48 x 3 units"]):
    X, Y, slopes, corrs, _ = maup[name]
    xs, ys = X[:, 0], Y[:, 0]
    keep = rng_plot.choice(len(xs), min(len(xs), 300), replace=False)
    ax.scatter(xs[keep], ys[keep], s=8, alpha=0.6)
    line = np.array([xs.min(), xs.max()])
    ax.plot(line, ys.mean() + slopes[0] * (line - xs.mean()), color="black")
    ax.set_title(f"{name}\nslope {slopes[0]:.2f}", fontsize=8)
    print(f"first world, {name}: slope {slopes[0]:.2f}")
    ax.set_xlabel("covariate")
axes[0].set_ylabel("outcome")
fig.tight_layout()
fig.savefig("figures/ch06/fig_6_1_maup_one.png", dpi=120)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.boxplot([maup[k][2] for k in zonings])
ax.set_xticklabels(["fine", "12x12", "6x6", "3x3",
                    "strips\n48x3", "strips\n3x48"])
ax.axhline(0, color="gray", lw=0.7)
ax.set_ylabel("regression slope (200 simulated worlds)")
fig.tight_layout()
fig.savefig("figures/ch06/fig_6_2_maup_slopes.png", dpi=120)
```

In the first of the 200 worlds, the fitted slope is $0.46$ for the fine cells, $0.00$ for the 144 squares, $-0.30$ for the nine squares, and $-0.11$ for the 144 east-west strips, as Figure 6.1 shows. One world can differ from the average, which is why Figure 6.2 shows all 200.

---

## Exercises

**Analytical**

1. For the four tracts of Section 6.1.1, find a third grouping into two ZIP codes (NW with SE, and NE with SW) and compute the slope across the two units. Then find values of $y$ for the four tracts, keeping $x = (1, 5, 3, 7)$, for which the slopes of zoning A and zoning B have the same sign.

2. In the ecological example of Section 6.1.2, give each county a different within-county slope, $+1$, $+2$, and $+3$, keeping the county averages the same. Compute the pooled slope. Show that the pooled slope is a weighted average of the within-county slopes and a between-county term, and find the weights.

3. Show that for the aggregated variables of Section 6.1.1 the slope across units is approximately $(v_a - 0.5v_b)/(v_a + v_b)$. State exactly where independence of $a$, $b$, and $\varepsilon$ is used, and why "approximately" is needed.

4. Show that the eigenvalues of the row-standardized rook matrix for a path of three regions in a line are $1$, $0$, and $-1$. Find the eigenvectors and verify that the path is bipartite.

5. A $2 \times 2$ rook grid is a cycle of four regions. Compute $\mathbf{W}$, show that $\mathbf{W}$ is symmetric here, and find its eigenvalues. Why is $\mathbf{W}$ symmetric when all degrees are equal?

6. For the three clinics of Section 6.2.3, compute $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ for the nearest-neighbor matrix $\mathbf{B}$ and show that it is not symmetric. Compute the eigenvalues of the row-standardized $\mathbf{B}$ and check whether they are real. Repeat for the symmetrized matrix $\max(\mathbf{B}, \mathbf{B}')$.

7. Show that for row-standardized weights, $I = \mathbf{y}_c'\mathbf{W}\mathbf{y}_c/\mathbf{y}_c'\mathbf{y}_c$ equals the slope of the regression of $\mathbf{W}\mathbf{y}_c$ on $\mathbf{y}_c$ through the origin. Then find the slope of the reverse regression, of $\mathbf{y}_c$ on $\mathbf{W}\mathbf{y}_c$, and explain why it is not $1/I$.

8. Compute the $4 \times 4$ matrix $\mathbf{M}$ for the hub map by hand from $\mathbf{P}_c\mathbf{W}_s\mathbf{P}_c$, and verify $\operatorname{tr}(\mathbf{M}) = -1$, $\operatorname{tr}(\mathbf{M}^2) = 43/48$, and $\mathbf{M}\mathbf{1} = \mathbf{0}$.

9. Prove the claim of Section 6.5.2 that the rotation-invariance of $\mathbf{w}$ implies independence of its direction and length. (Hint: write the density of $\mathbf{w}$ in polar coordinates.) Then show that the same argument fails if $\operatorname{Var}(\mathbf{w}) = \tau^2\boldsymbol{\Sigma}_w$ with $\boldsymbol{\Sigma}_w$ not a multiple of the identity.

10. For the hub map, compute $\operatorname{E}[C]$ and, using the Laplacian form of Section 6.7.1, $\operatorname{Var}(C)$ under normality. (Use the same method as Section 6.5.4, with $\operatorname{tr}(\mathcal{L}^2)$ in place of $\operatorname{tr}(\mathbf{M}^2)$, where $\mathcal{L} = \mathbf{D}_s - \mathbf{W}_s$.)

11. Show that for the local indicators of Section 6.7.2, $\sum_iI_i = S_0\,I$. Then, for $n$ independent standard normal values and weights of one neighbor per region, explain why the local indicators are not independent of each other even when the data are.

12. In Section 6.8.2, show that the mean of the permutation distribution of Geary's $C$ is also 1 for any weights.

13. Explain why the residuals of an OLS fit are not exchangeable even when the errors are independent and identically distributed: find $\operatorname{Var}(e_i)$ and $\operatorname{Cov}(e_i, e_j)$ in terms of $\mathbf{P}_{\mathbf{X}}$, and compute them for the hub map with $\mathbf{X} = [\mathbf{1}\ \ \mathbf{x}]$, $\mathbf{x} = (0, 1, 2, 3)'$.

**Computational and simulation**

14. Rebuild the table of Section 6.2.4 for the queen weights with the rows standardized and not standardized. Does Moran's I change? Does its null mean change? (Remember the factor $n/S_0$.)

15. Simulate 5,000 data sets on the county map with independent standard normal values and compute $I$ with rook weights. Compare the histogram of $I$ with a normal density with the exact mean and standard deviation. Repeat with $n = 9$ regions ($3 \times 3$ rook grid) and describe how the approximation changes.

16. Using the data of Section 6.8.5, find the simplest rescaling of the values (dividing by $\exp(\text{deprivation})$) that restores the 5% size for both tests. Then find what happens if the variance field is not known and has to be estimated from the data.

17. Repeat the residual experiment of Section 6.8.5 for polynomial trend surfaces of degrees 1, 2, 3, and 4 ($p = 3, 6, 10, 15$) and plot the exact null mean of $I_e$ against $p$. At what $p$ does the naive permutation test first lose half its power against a spatial effect with $\phi = 0.2$?

18. Extend the MAUP simulation of Section 6.9.3 with a second regional range, 0.3 east-west and 0.1 north-south. Does the slope still change sign as the units grow? Predict the answer from the weighted-average formula first, and then check it.

19. Build a $48 \times 48$ fine grid and a zoning by 144 irregular units: draw 144 random points, assign each cell to its nearest point (a Voronoi partition), and aggregate. Compare the slope with that of the squares and strips, and compute Moran's I with queen weights between the Voronoi units (adjacent when any of their cells touch).

20. A permutation test for Geary's $C$ is analogous to the one for $I$. Implement it and apply it to the strong response with the rook weights. Compare the p-value with that of Moran's I.

---

## What Comes Next

The weights matrix and the null distribution of Moran's I give us a way to detect spatial structure in areal data. Chapters 7 and 8 turn the weights into models. Chapter 7 builds the simultaneous autoregressive models, in which each region's value is regressed on its neighbors' values, and shows how to interpret the effect of a covariate once the feedback between neighbors is accounted for. Chapter 8 builds the conditional autoregressive models, in which each region is described given its neighbors, and shows that both families produce a covariance structure that can be handled with the sparse precision matrices of Chapter 8. Chapter 9 puts these on the count scale, with expected counts and the small-population problem that Chapter 0 introduced.
