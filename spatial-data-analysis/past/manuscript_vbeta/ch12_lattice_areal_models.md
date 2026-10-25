# Chapter 12: Lattice and Areal Models: CAR, SAR, and Spatial Autoregression

```
================================================================================
   GEOSTATISTICAL CONTINUUM                AREAL / LATTICE DISCRETE GRAPH
--------------------------------------------------------------------------------
Domain:     Continuous s in D c R^d         Discrete regions A_i, i = 1, ..., n
Data:       Point samples Z(s_i)            Aggregated counts / averages Y_i
Dependence: Covariance C(||s_i - s_j||)     Adjacency topology W = (w_ij)
Mechanism:  Distance metric d(s_i, s_j)     Border sharing, distance thresholds
Models:     Ordinary Kriging, Gaussian RF   CAR, SAR, ICAR, BYM Hierarchical
================================================================================
```

In Chapters 1 through 10, spatial phenomena unfolded across a continuous Euclidean continuum $\mathcal{D} \subset \mathbb{R}^d$. A rain gauge, an exploratory drill core, or an atmospheric sensor recorded a point attribute $Z(\mathbf{s}_i)$ at an exact coordinate $\mathbf{s}_i \in \mathcal{D}$. Spatial autocorrelation was governed by continuous separation vectors $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$ and stationary covariance functions $C(\mathbf{h})$.

In many practical disciplines—epidemiology, econometrics, public health, census demography, and regional planning—spatial observations are **areal** or **lattice** data. We observe disease counts within administrative zip codes, median household income across census tracts, agricultural yields over bounded farm fields, or remote sensing pixels partitioned into a regular lattice. Here, continuous distance fails to capture the spatial reality:
1. Observations represent aggregated summaries over polygons $A_i \subset \mathcal{D}$ rather than point measurements (recall the change-of-support problem from Chapter 8).
2. Physical distance between irregular administrative centroids often fails to reflect interaction; interaction is mediated by shared administrative borders, transit corridors, or discrete neighborhood graphs.
3. The spatial index set is discrete: $i \in \{1, 2, \dots, n\}$.

This chapter establishes the statistical theory and computational practice of **discrete spatial autoregression**. We bridge the familiar world of classical time series (AR and MA processes) and linear models into discrete space, dissecting the two competing paradigms of spatial dependence: **Simultaneously Autoregressive (SAR)** models and **Conditionally Autoregressive (CAR)** models. We resolve their foundational properties, investigate exploratory spatial metrics (Moran's $I$ and Geary's $c$), develop intrinsic formulations (ICAR) for disease mapping (the Besag-York-Mollié framework), and implement robust standard-library algorithms for spatial weights, permutation testing, and profile maximum likelihood estimation.

---

## 12.1 The Discrete Spatial Universe: Areal Data and Adjacency Graphs

Let the study domain be partitioned into $n$ mutually exclusive and exhaustive non-overlapping spatial units (zones, polygons, or pixels) $\{A_1, A_2, \dots, A_n\}$ such that $\bigcup_{i=1}^n A_i = \mathcal{D}$ and $A_i \cap A_j = \emptyset$ for $i \neq j$. Associated with each areal unit $A_i$ is a random response variable $Y_i$. We denote the joint observation vector by $\mathbf{Y} = [Y_1, Y_2, \dots, Y_n]^\top \in \mathbb{R}^n$.

### 12.1.1 The Running Toy Map: Lattice Alpha

To eliminate abstraction and allow exact hand calculations throughout this chapter, we introduce **Lattice $\alpha$** ($\mathcal{L}_\alpha$): a one-dimensional chain of five contiguous administrative districts labeled $A_1$ through $A_5$:

```
+---------+---------+---------+---------+---------+
|   A_1   |   A_2   |   A_3   |   A_4   |   A_5   |
+---------+---------+---------+---------+---------+
```

District $A_1$ borders only $A_2$; interior districts $A_2, A_3, A_4$ border their immediate left and right neighbors; district $A_5$ borders only $A_4$.

### 12.1.2 Spatial Proximity and Weight Matrices

In continuous geostatistics, spatial structure is encoded through an $n \times n$ covariance matrix $\boldsymbol{\Sigma}$ where $\Sigma_{ij} = C(\|\mathbf{s}_i - \mathbf{s}_j\|)$. In areal modeling, the geographic configuration is encoded prior to estimation via an $n \times n$ **spatial weight matrix** $\mathbf{W} = [w_{ij}]_{i,j=1}^n$.

By universal convention:
- $w_{ii} = 0$ for all $i = 1, \dots, n$ (no spatial unit is considered a neighbor of itself).
- $w_{ij} \ge 0$ for all $i, j$ (spatial relationships represent proximity or interaction intensity).

Three primary specifications are used to define $w_{ij}$:

#### 1. Binary Contiguity
Regions $i$ and $j$ are assigned $w_{ij} = 1$ if they share a geographic boundary, and $w_{ij} = 0$ otherwise:
- **Rook Contiguity:** Sharing a common linear boundary segment of length $> 0$.
- **Queen Contiguity:** Sharing a common boundary segment or a common corner point.

For our 5-region running toy map Lattice $\alpha$, the binary contiguity matrix $\mathbf{A} \in \mathbb{R}^{5 \times 5}$ is tridiagonal:
$$\mathbf{A} = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 \\
1 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 1 \\
0 & 0 & 0 & 1 & 0
\end{bmatrix}$$

The row sums $d_i = \sum_{j=1}^n A_{ij}$ define the **node degrees** (number of neighbors). For Lattice $\alpha$, the degree vector is $\mathbf{d} = [1, 2, 2, 2, 1]^\top$, and the total number of directed connections is $S_0 = \sum_{i=1}^n \sum_{j=1}^n A_{ij} = 8$. The diagonal degree matrix is $\mathbf{D} = \operatorname{diag}(d_1, \dots, d_n) = \operatorname{diag}(1, 2, 2, 2, 1)$.

#### 2. Row-Standardization
In regression modeling and spatial lag computation, it is standard practice to normalize $\mathbf{A}$ such that each row sums to 1. The **row-standardized spatial weight matrix** $\mathbf{W}^*$ has elements:
$$w_{ij}^* = \frac{A_{ij}}{\sum_{k=1}^n A_{ik}} = \frac{A_{ij}}{d_i}$$

For Lattice $\alpha$, row-standardization produces:
$$\mathbf{W}^* = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 \\
1/2 & 0 & 1/2 & 0 & 0 \\
0 & 1/2 & 0 & 1/2 & 0 \\
0 & 0 & 1/2 & 0 & 1/2 \\
0 & 0 & 0 & 1 & 0
\end{bmatrix}$$

Row-standardization transforms matrix multiplication into a local averaging operator. If $\mathbf{y} \in \mathbb{R}^n$ is an observation vector, the product $\mathbf{W}^*\mathbf{y}$ is the **spatial lag vector**, where the $i$-th entry:
$$[\mathbf{W}^*\mathbf{y}]_i = \sum_{j=1}^n w_{ij}^* y_j = \frac{1}{d_i} \sum_{j \in \mathcal{N}(i)} y_j$$
is the arithmetic average of the responses in the neighborhood $\mathcal{N}(i)$ of region $i$.

> [!WARNING] Row-Standardization Destroys Symmetry
> Although the binary contiguity matrix $\mathbf{A}$ is symmetric ($A_{ij} = A_{ji}$), the row-standardized matrix $\mathbf{W}^* = \mathbf{D}^{-1}\mathbf{A}$ is generally **asymmetric** whenever regions have differing numbers of neighbors ($d_i \neq d_j$). In Lattice $\alpha$, $w_{12}^* = 1$ while $w_{21}^* = 1/2$. As shown in §12.3 and §12.4, preserving symmetry is critical for CAR precision matrices, whereas SAR formulations readily accommodate asymmetric $\mathbf{W}^*$.

#### 3. Distance-Decay and $k$-Nearest Neighbors
When administrative polygons vary widely in land area (e.g., dense urban census tracts versus vast rural counties), border sharing can misrepresent spatial interaction. Alternative weight specifications include:
- **Inverse Distance:** $w_{ij} = d(\mathbf{c}_i, \mathbf{c}_j)^{-\alpha}$ for centroid distance $d(\mathbf{c}_i, \mathbf{c}_j) \le d_{\max}$, and $0$ otherwise.
- **$k$-Nearest Neighbors ($k$-NN):** $w_{ij} = 1$ if polygon centroid $\mathbf{c}_j$ is among the $k$ closest centroids to $\mathbf{c}_i$, ensuring uniform sample support across every local neighborhood.

### 12.1.3 Spatial Islands and Graph Disconnection

A critical practical issue occurs when a study domain contains **islands** (e.g., offshore islands, overseas territories, or enclaves). For an isolated polygon $i$, $\mathcal{N}(i) = \emptyset$, meaning $d_i = 0$.
- In a raw adjacency matrix, row $i$ and column $i$ consist entirely of zeros.
- In row-standardization, dividing by $d_i = 0$ yields a division-by-zero exception. By convention, row $i$ in $\mathbf{W}^*$ remains all zeros ($[\mathbf{W}^*\mathbf{y}]_i = 0$).
- In intrinsic autoregressions (ICAR), each disconnected component induces an independent rank deficiency in the precision matrix, requiring separate sum-to-zero identifiability constraints (see §12.5).

```
================================================================================
FROM THE LECTURE HALL: Box 12.1
The Ecological Fallacy and the Modifiable Areal Unit Problem (MAUP)
================================================================================
When transitioning from continuous points to areal aggregates, two classical
traps arise that every spatial statistician must recognize:

1. The Ecological Fallacy:
   Inferring individual-level relationships from aggregate areal statistics.
   For example, finding a positive correlation between county-level median
   income and county-level cancer rates does NOT imply that wealthy individuals
   have higher cancer risk. Confounding variables, demographic stratification,
   and aggregation bias can reverse correlation signs (Simpson's Paradox).

2. The Modifiable Areal Unit Problem (MAUP):
   Statistical relationships observed between areal units depend directly on
   the boundaries and scale of aggregation:
   - Scale Effect: Aggregating small units (census tracts) into large units
     (counties) averages out local variance, artificially inflating correlation
     coefficients r.
   - Zone Effect: Holding the scale fixed (e.g., 50 units) but redrawing the
     boundaries (gerrymandering) can dramatically alter regression slopes,
     p-values, and spatial autocorrelation statistics.

Areal models quantify spatial pattern conditional on the chosen partition; they
do not represent immutable point-level physical laws.
================================================================================
```

---

## 12.2 Exploratory Spatial Data Analysis: Moran's $I$ and Geary's $c$

Before fitting formal parametric autoregressive models, we quantify the degree of spatial autocorrelation in the response vector $\mathbf{y} \in \mathbb{R}^n$.

### 12.2.1 Global Moran's $I$

**Moran's $I$** (Moran, 1950) is the spatial analogue of the Pearson product-moment correlation coefficient. Let $z_i = y_i - \bar{y}$ denote the mean-centered response, where $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$. Moran's $I$ is defined as:
$$I = \frac{n}{S_0} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij} (y_i - \bar{y})(y_j - \bar{y})}{\sum_{i=1}^n (y_i - \bar{y})^2} = \frac{n}{S_0} \frac{\mathbf{z}^\top \mathbf{W} \mathbf{z}}{\mathbf{z}^\top \mathbf{z}}$$
where $\mathbf{z} = \mathbf{y} - \bar{y}\mathbf{1}$ and $S_0 = \sum_{i=1}^n \sum_{j=1}^n w_{ij}$ is the sum of all spatial weights.

#### Moran's $I$ as an OLS Regression Slope
When $\mathbf{W}^*$ is row-standardized, the sum of weights is $S_0 = \sum_{i=1}^n \sum_{j=1}^n w_{ij}^* = n$, simplifying the prefactor $\frac{n}{S_0} = 1$:
$$I = \frac{\mathbf{z}^\top \mathbf{W}^* \mathbf{z}}{\mathbf{z}^\top \mathbf{z}} = \frac{\sum_{i=1}^n z_i [\mathbf{W}^* \mathbf{z}]_i}{\sum_{i=1}^n z_i^2}$$

Notice the mathematical structure: this is precisely the Ordinary Least Squares (OLS) slope coefficient from regressing the spatial lag of the centered variable $\tilde{\mathbf{z}} = \mathbf{W}^*\mathbf{z}$ on $\mathbf{z}$ without an intercept:
$$\widehat{\beta}_{\text{OLS}} = \frac{\mathbf{z}^\top (\mathbf{W}^*\mathbf{z})}{\mathbf{z}^\top \mathbf{z}} = I$$
This regression slope is visualized in the **Moran Scatterplot**, where the horizontal axis plots $z_i$ and the vertical axis plots the spatial lag $[\mathbf{W}^*\mathbf{z}]_i$.

#### Expected Value and Sampling Distribution
Under the null hypothesis $H_0$ of no spatial autocorrelation (independent and identically distributed observations), the theoretical expected value of Moran's $I$ is **not zero**, but slightly negative:
$$\mathbb{E}[I] = -\frac{1}{n - 1}$$
This negative offset reflects the constraint that the centered residuals must sum to zero ($\sum_{i=1}^n z_i = 0$), inducing a slight negative sample correlation of magnitude $\mathcal{O}(1/n)$.

- If $I > -\frac{1}{n-1}$: Positive spatial autocorrelation (clustering of similar values).
- If $I < -\frac{1}{n-1}$: Negative spatial autocorrelation (spatial dispersion or checkerboard patterns).

#### Cliff-Ord Analytical Variance
Cliff and Ord (1973, 1981) established the analytical variance of Moran's $I$ under two distinct null frameworks:
1. **Normality Assumption:** $Y_i \stackrel{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$.
2. **Permutation (Randomization) Assumption:** The observed values $\{y_1, \dots, y_n\}$ are fixed, and spatial randomness arises from randomly assigning them among the $n$ geographic zones.

Under the permutation assumption, the exact variance is:
$$\operatorname{Var}(I) = \frac{n \left[ (n^2 - 3n + 3) S_1 - n S_2 + 3 S_0^2 \right] - b_2 \left[ (n^2 - n) S_1 - 2n S_2 + 6 S_0^2 \right]}{(n - 1)(n - 2)(n - 3) S_0^2} - (\mathbb{E}[I])^2$$
where:
$$S_1 = \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n (w_{ij} + w_{ji})^2$$
$$S_2 = \sum_{i=1}^n \left( \sum_{j=1}^n w_{ij} + \sum_{j=1}^n w_{ji} \right)^2$$
$$b_2 = \frac{m_4}{m_2^2} = \frac{\frac{1}{n} \sum_{i=1}^n z_i^4}{\left( \frac{1}{n} \sum_{i=1}^n z_i^2 \right)^2} \quad (\text{sample kurtosis})$$

The standardized test statistic:
$$Z_I = \frac{I - \mathbb{E}[I]}{\sqrt{\operatorname{Var}(I)}} \stackrel{d}{\longrightarrow} \mathcal{N}(0, 1)$$
provides an asymptotic two-tailed test for spatial clustering.

> [!IMPORTANT] Testing Regression Residuals vs. Raw Data
> When calculating Moran's $I$ on OLS regression residuals $\mathbf{e} = \mathbf{y} - \mathbf{X}\widehat{\boldsymbol{\beta}}$, the residuals do not lie in an $n$-dimensional space, but are constrained to an $(n - p)$-dimensional orthogonal subspace dictated by the projection matrix $\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. 
> 
> The variance of Moran's $I$ for regression residuals requires the Cliff-Ord residual adjustment involving $\operatorname{Tr}(\mathbf{M}\mathbf{W}\mathbf{M}\mathbf{W}^\top)$. Using raw permutation tests on residuals without projecting $\mathbf{M}$ understates residual variance and leads to inflated Type I error rates.

### 12.2.2 Geary's $c$

**Geary's $c$** (Geary, 1954) focuses on squared pairwise differences between contiguous neighbors rather than cross-products of deviations from the global mean:
$$c = \frac{n - 1}{2 S_0} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij} (y_i - y_j)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

Notice that the numerator is directly proportional to the semivariance $\gamma(h)$ from Chapter 2 computed across graph edges.

#### Range and Interpretation of Geary's $c$
Under $H_0$ of spatial randomness:
$$\mathbb{E}[c] = 1$$
- $c \in [0, 1)$: Positive spatial autocorrelation (neighbors have small differences).
- $c = 1$: Spatial randomness.
- $c > 1$: Negative spatial autocorrelation (neighbors have large differences).

> [!CAUTION] Geary's $c$ Upper Bound Myth
> Introductory GIS texts frequently state that Geary's $c$ is strictly bounded in $[0, 2]$. **This is mathematically false.** 
> While $c$ cannot be negative ($c \ge 0$ because all terms in the numerator and denominator are non-negative), its upper bound depends directly on the spectral properties of the weight matrix $\mathbf{W}$. For star graphs or bipartite configurations where extreme values are placed at central hub nodes, Geary's $c$ can easily reach $2.5, 4.0$, or higher (e.g., $c \approx \frac{n-1}{2}$ in pathological star graphs).

### 12.2.3 Local Indicators of Spatial Association (LISA)

Global statistics summarize the entire study region into a single number, potentially masking localized spatial pockets of disease outbreaks or economic decline. Anselin (1995) introduced **LISA** (Local Indicators of Spatial Association), decomposing global Moran's $I$ into local contributions:
$$I_i = \frac{z_i}{m_2} \sum_{j=1}^n w_{ij}^* z_j$$
where $m_2 = \frac{1}{n} \sum_{i=1}^n z_i^2$ is the sample variance. The sum of local Moran indices is proportional to global Moran's $I$: $\sum_{i=1}^n I_i = n I$.

The Moran Scatterplot partitions regions into four distinct spatial association quadrants:
1. **High-High (Quadrant I, $+/+$):** High value surrounded by high neighbor average (Spatial Cluster / Hotspot).
2. **Low-Low (Quadrant III, $-/-$):** Low value surrounded by low neighbor average (Spatial Cluster / Coldspot).
3. **High-Low (Quadrant IV, $+/-$):** High value surrounded by low neighbor average (Spatial Outlier).
4. **Low-High (Quadrant II, $-/+$):** Low value surrounded by high neighbor average (Spatial Outlier).

---

## 12.3 Simultaneously Autoregressive (SAR) Models

We now turn to parametric probability models for discrete spatial fields. How do we extend the univariate autoregressive time series model $Y_t = \rho Y_{t-1} + \epsilon_t$ to two dimensions?

```
================================================================================
FROM THE LECTURE HALL: Box 12.2
The 1D AR(1) Rosetta Stone: Why 2D Breaks Time Series
================================================================================
Consider the stationary Gaussian AR(1) process in discrete time:
                  Y_t = rho * Y_{t-1} + epsilon_t,   |rho| < 1

In 1D, time possesses a physical arrow: causation is strictly unilateral
(past -> future). The joint distribution is formed by a sequential chain:
         p(y_1, y_2, ..., y_T) = p(y_1) * PROD_{t=2}^T p(y_t | y_{t-1})

What happens when we condition on both the past and the future?
Taking the full conditional distribution of Y_t given ALL other observations:
         Y_t | {Y_s, s != t} ~ N( rho / (1 + rho^2) * (Y_{t-1} + Y_{t+1}),
                                  sigma^2 / (1 + rho^2) )

In 1D, the causal autoregression (SAR-like) and the two-sided conditional
autoregression (CAR-like) are two perspectives of the exact same Gaussian Markov
process.

In 2D spatial lattices, space has NO intrinsic arrow. Spatial interaction is
inherently multidirectional and omnidirectional (simultaneous feedback).
Because the graph contains cycles (A influences B, which influences C, which
influences A), the simultaneous specification (SAR) and the conditional
specification (CAR) SPLIT into two distinct, non-equivalent probability models!
================================================================================
```

### 12.3.1 Spatial Lag vs. Spatial Error Models

In spatial econometrics and linear modeling, two primary SAR specifications arise. Understanding the structural difference between them is vital:

#### 1. The Spatial Autoregressive Lag Model (Spatial Lag)
In the spatial lag model, the response $Y_i$ depends directly on the responses of neighboring regions:
$$\mathbf{Y} = \rho \mathbf{W} \mathbf{Y} + \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$
where $\rho$ is the spatial autoregressive parameter, $\mathbf{X} \in \mathbb{R}^{n \times p}$ is a design matrix of exogenous covariates, and $\boldsymbol{\beta} \in \mathbb{R}^p$ is the regression coefficient vector.

Rearranging into reduced form:
$$(\mathbf{I} - \rho \mathbf{W}) \mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon} \implies \mathbf{Y} = (\mathbf{I} - \rho \mathbf{W})^{-1} \mathbf{X} \boldsymbol{\beta} + (\mathbf{I} - \rho \mathbf{W})^{-1} \boldsymbol{\epsilon}$$

> [!CRITICAL] OLS Inconsistency in Spatial Lag Models
> Attempting to estimate $\rho$ and $\boldsymbol{\beta}$ in the spatial lag model via Ordinary Least Squares (regressing $\mathbf{Y}$ onto $[\mathbf{W}\mathbf{Y}, \mathbf{X}]$) produces **biased and inconsistent estimates**. 
> 
> Because $\mathbf{Y} = (\mathbf{I} - \rho \mathbf{W})^{-1}(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon})$, the spatial lag term $\mathbf{W}\mathbf{Y} = \mathbf{W}(\mathbf{I} - \rho \mathbf{W})^{-1}\boldsymbol{\epsilon} + \dots$ is intrinsically correlated with the disturbance term $\boldsymbol{\epsilon}$. This violates the fundamental Gauss-Markov exogeneity assumption $\mathbb{E}[\mathbf{W}\mathbf{Y} \cdot \boldsymbol{\epsilon}] \neq \mathbf{0}$. Simultaneous equation feedback requires instrumental variables or Maximum Likelihood estimation.

#### 2. The Spatial Error Model (SAR Error)
In the spatial error model, spatial autocorrelation resides exclusively within the disturbance process:
$$\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{u}, \quad \mathbf{u} = \rho \mathbf{W} \mathbf{u} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$
Solving for $\mathbf{u} = (\mathbf{I} - \rho \mathbf{W})^{-1} \boldsymbol{\epsilon}$, the full model becomes:
$$\mathbf{Y} \sim \mathcal{N}\left( \mathbf{X} \boldsymbol{\beta}, \; \boldsymbol{\Sigma}_{\text{SAR}} \right)$$

In this model, OLS estimates of $\boldsymbol{\beta}$ remain unbiased and consistent, but they are **inefficient**, and naive OLS standard error formulas are severely biased (typically underestimating standard errors and inflating statistical significance).

### 12.3.2 Covariance and Precision Structure of SAR

Let $\mathbf{B} = \mathbf{I} - \rho \mathbf{W}$. For the spatial disturbance $\mathbf{u} = \mathbf{B}^{-1} \boldsymbol{\epsilon}$, the covariance matrix is:
$$\boldsymbol{\Sigma}_{\text{SAR}} = \operatorname{Var}(\mathbf{u}) = \mathbb{E}\left[ \mathbf{B}^{-1} \boldsymbol{\epsilon} \boldsymbol{\epsilon}^\top (\mathbf{B}^{-1})^\top \right] = \sigma^2 \mathbf{B}^{-1} (\mathbf{B}^\top)^{-1} = \sigma^2 (\mathbf{B}^\top \mathbf{B})^{-1}$$

The **SAR precision matrix** (inverse covariance) is therefore:
$$\mathbf{Q}_{\text{SAR}} = \boldsymbol{\Sigma}_{\text{SAR}}^{-1} = \frac{1}{\sigma^2} (\mathbf{I} - \rho \mathbf{W})^\top (\mathbf{I} - \rho \mathbf{W}) = \frac{1}{\sigma^2} \left[ \mathbf{I} - \rho(\mathbf{W} + \mathbf{W}^\top) + \rho^2 \mathbf{W}^\top \mathbf{W} \right]$$

Notice the quadratic term $\rho^2 \mathbf{W}^\top \mathbf{W}$. This term implies that two regions $i$ and $j$ that are not direct neighbors ($w_{ij} = 0$), but share a mutual neighbor $k$ ($w_{ik} > 0$ and $w_{jk} > 0$), have a **nonzero entry** in the precision matrix:
$$[\mathbf{Q}_{\text{SAR}}]_{ij} \neq 0$$
Thus, in a SAR model, second-order neighbors are conditionally dependent given all other regions.

### 12.3.3 Parameter Invertibility and Stability

The matrix $\mathbf{B} = \mathbf{I} - \rho \mathbf{W}$ is invertible if and only if $\det(\mathbf{I} - \rho \mathbf{W}) \neq 0$. 

Let $\{\lambda_1, \lambda_2, \dots, \lambda_n\}$ denote the real and complex eigenvalues of $\mathbf{W}$, sorted such that $\lambda_{\min} = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n = \lambda_{\max}$. The eigenvalues of $\mathbf{I} - \rho \mathbf{W}$ are $1 - \rho \lambda_i$. Hence, invertibility fails if and only if $\rho = 1 / \lambda_i$ for any $i$.

To guarantee that the matrix is invertible, that the covariance matrix $\boldsymbol{\Sigma}_{\text{SAR}}$ is strictly positive definite, and that the spatial process can be generated via a convergent Neumann expansion:
$$(\mathbf{I} - \rho \mathbf{W})^{-1} = \sum_{k=0}^\infty \rho^k \mathbf{W}^k$$
the parameter $\rho$ must lie within the connected open interval containing zero:
$$\rho \in \left( \frac{1}{\lambda_{\min}}, \; \frac{1}{\lambda_{\max}} \right)$$

When $\mathbf{W}^*$ is row-standardized, the Perron-Frobenius theorem guarantees that the maximum eigenvalue is exactly $\lambda_{\max} = 1$ (with corresponding eigenvector $\mathbf{1}$). If the graph is bipartite, $\lambda_{\min} = -1$; for general non-bipartite graphs, $-1 \le \lambda_{\min} < 0$. Therefore, for any row-standardized weight matrix, the valid parameter domain is contained within:
$$\rho \in (-1, 1)$$

### 12.3.4 Maximum Likelihood and the Jacobian Problem

To estimate $\boldsymbol{\theta} = (\boldsymbol{\beta}, \rho, \sigma^2)$ in the SAR model:
$$\mathbf{Y} = \rho \mathbf{W} \mathbf{Y} + \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$
we must write the likelihood of $\mathbf{Y}$. 

The transformation from the independent errors $\boldsymbol{\epsilon}$ to the observed data $\mathbf{Y}$ is given by $\boldsymbol{\epsilon} = (\mathbf{I} - \rho \mathbf{W})\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}$. The Jacobian matrix of this multivariate transformation is:
$$\mathbf{J} = \frac{\partial \boldsymbol{\epsilon}}{\partial \mathbf{Y}} = \mathbf{I} - \rho \mathbf{W}$$

By the change-of-variables theorem, the joint probability density function of $\mathbf{Y}$ is:
$$f(\mathbf{y} \mid \boldsymbol{\beta}, \rho, \sigma^2) = (2\pi \sigma^2)^{-n/2} \exp\left( -\frac{1}{2\sigma^2} \boldsymbol{\epsilon}^\top \boldsymbol{\epsilon} \right) |\det(\mathbf{J})|$$
where $\boldsymbol{\epsilon} = (\mathbf{I} - \rho \mathbf{W})\mathbf{y} - \mathbf{X}\boldsymbol{\beta}$.

Taking the natural logarithm yields the log-likelihood:
$$\ln L(\boldsymbol{\beta}, \rho, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) + \ln|\det(\mathbf{I} - \rho \mathbf{W})| - \frac{1}{2\sigma^2} \left[ (\mathbf{I} - \rho \mathbf{W})\mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right]^\top \left[ (\mathbf{I} - \rho \mathbf{W})\mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right]$$

```
================================================================================
FROM THE LECTURE HALL: Box 12.3
The Jacobian Determinant: Why Least Squares Fails
================================================================================
What happens if you ignore the Jacobian determinant term ln|det(I - rho*W)|
and simply minimize the Residual Sum of Squares (RSS):
             RSS(rho) = || (I - rho*W)y - X*beta ||^2 ?

As rho increases toward 1, (I - rho*W) acts as a high-pass difference operator.
For positive spatial autocorrelation, neighboring values y_i and [W*y]_i are
very close. Subtracting rho*[W*y] shrinks the variance of the residuals!
Consequently, RSS(rho) monotonically decreases as rho -> 1.

Minimizing RSS without the Jacobian will ALWAYS overestimate rho, driving
estimates to the upper boundary rho -> 1!

The Jacobian determinant ln|det(I - rho*W)| acts as a rigorous mathematical
penalty against shrinking the volume element. Because all eigenvalues
(1 - rho*lambda_i) < 1 for rho > 0, the log-determinant is strictly negative
and plunges to -infinity as rho -> 1 / lambda_max. Maximum Likelihood balances
the decreasing residual variance against this steep determinant penalty.
================================================================================
```

#### Profile Likelihood Optimization
Given any fixed value of $\rho \in (\lambda_{\min}^{-1}, \lambda_{\max}^{-1})$, the optimal regression coefficients and variance can be evaluated analytically:
$$\widehat{\boldsymbol{\beta}}(\rho) = \left( \mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top (\mathbf{I} - \rho \mathbf{W})\mathbf{y}$$
$$\widehat{\sigma}^2(\rho) = \frac{1}{n} \mathbf{e}(\rho)^\top \mathbf{e}(\rho), \quad \mathbf{e}(\rho) = (\mathbf{I} - \rho \mathbf{W})\mathbf{y} - \mathbf{X}\widehat{\boldsymbol{\beta}}(\rho)$$

Substituting $\widehat{\boldsymbol{\beta}}(\rho)$ and $\widehat{\sigma}^2(\rho)$ back into $\ln L$ yields the **profile log-likelihood** for $\rho$:
$$\ln L_p(\rho) = -\frac{n}{2}\left[ \ln(2\pi) + 1 \right] - \frac{n}{2} \ln\left( \widehat{\sigma}^2(\rho) \right) + \sum_{i=1}^n \ln(1 - \rho \lambda_i)$$
This reduces an otherwise complex $(p + 2)$-dimensional non-convex optimization problem to a rapid, stable **one-dimensional grid search or Brent scalar maximization** over $\rho \in (-1, 1)$.

---

## 12.4 Conditionally Autoregressive (CAR) Models

Rather than defining a simultaneous linear system across the entire map, Julian Besag (1974) proposed specifying spatial dependence through the **full conditional distributions** of each region given all remaining regions.

### 12.4.1 Full Conditionals and Brook's Lemma

Let $\mathbf{Y}_{-i} = \{Y_j : j \neq i\}$ denote the set of all observations excluding region $i$. Suppose we specify the conditional distribution of each $Y_i$ as a univariate Gaussian:
$$Y_i \mid \mathbf{Y}_{-i} \sim \mathcal{N}\left( \mu_i + \sum_{j=1}^n c_{ij}(Y_j - \mu_j), \; \tau_i^2 \right)$$
where $c_{ii} = 0$, $c_{ij}$ represents the conditional dependence of region $i$ on region $j$, and $\tau_i^2 > 0$ is the conditional variance.

Can *any* set of conditional distributions $\{p(y_i \mid \mathbf{y}_{-i})\}_{i=1}^n$ yield a valid joint probability density $p(\mathbf{y})$? **No.**

#### Brook's Lemma
Brook (1964) proved that the joint density can be uniquely reconstructed from the full conditionals relative to an arbitrary reference point $\mathbf{y}_0 = [y_{01}, \dots, y_{0n}]^\top$ via the identity:
$$\frac{p(\mathbf{y})}{p(\mathbf{y}_0)} = \prod_{i=1}^n \frac{p(y_i \mid y_1, \dots, y_{i-1}, y_{0,i+1}, \dots, y_{0n})}{p(y_{0i} \mid y_1, \dots, y_{i-1}, y_{0,i+1}, \dots, y_{0n})}$$

For Gaussian full conditionals, Brook's Lemma requires a strict mathematical consistency condition:
$$\frac{c_{ij}}{\tau_i^2} = \frac{c_{ji}}{\tau_j^2} \quad \text{for all } i, j$$

If this symmetry condition is violated, no joint probability distribution exists that can produce those conditional distributions!

### 12.4.2 Proper CAR Formulation

Let $\mathbf{C} = [c_{ij}]_{i,j=1}^n$ and $\mathbf{M} = \operatorname{diag}(\tau_1^2, \tau_2^2, \dots, \tau_n^2)$. The Brook's Lemma condition requires that $\mathbf{M}^{-1}\mathbf{C}$ be a symmetric matrix.

When this holds, the joint distribution of $\mathbf{Y}$ is uniquely determined as a multivariate Gaussian:
$$\mathbf{Y} \sim \mathcal{N}\left( \boldsymbol{\mu}, \; \mathbf{Q}_{\text{CAR}}^{-1} \right)$$
with the **CAR precision matrix**:
$$\mathbf{Q}_{\text{CAR}} = \mathbf{M}^{-1}(\mathbf{I} - \mathbf{C})$$

In standard spatial practice, we set $\mathbf{C} = \phi \mathbf{W}^* = \phi \mathbf{D}^{-1}\mathbf{A}$ and $\tau_i^2 = \tau^2 / d_i$, where $\mathbf{D} = \operatorname{diag}(d_1, \dots, d_n)$ is the diagonal degree matrix from the symmetric adjacency matrix $\mathbf{A}$.

Then $\mathbf{M} = \tau^2 \mathbf{D}^{-1}$. Let us verify the symmetry condition:
$$\mathbf{M}^{-1}\mathbf{C} = (\tau^{-2}\mathbf{D}) (\phi \mathbf{D}^{-1}\mathbf{A}) = \frac{\phi}{\tau^2} \mathbf{A}$$
Because $\mathbf{A}$ is symmetric, $\mathbf{M}^{-1}\mathbf{C}$ is symmetric! 

The resulting precision matrix for the **Proper CAR model** is:
$$\mathbf{Q}_{\text{CAR}} = \frac{1}{\tau^2} (\mathbf{D} - \phi \mathbf{A})$$

#### Parameter Space for Proper CAR
For $\mathbf{Q}_{\text{CAR}}$ to be a valid covariance inverse, it must be strictly positive definite ($\mathbf{Q}_{\text{CAR}} \succ \mathbf{0}$).
Let $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ be the symmetric normalized adjacency matrix, whose eigenvalues lie in $[-1, 1]$. The precision matrix can be factored as:
$$\mathbf{Q}_{\text{CAR}} = \frac{1}{\tau^2} \mathbf{D}^{1/2} (\mathbf{I} - \phi \tilde{\mathbf{A}}) \mathbf{D}^{1/2}$$
Strict positive definiteness requires all eigenvalues of $\mathbf{I} - \phi \tilde{\mathbf{A}}$ to be strictly positive:
$$1 - \phi \lambda_i(\tilde{\mathbf{A}}) > 0 \implies \phi \in \left( \frac{1}{\lambda_{\min}(\tilde{\mathbf{A}})}, \; \frac{1}{\lambda_{\max}(\tilde{\mathbf{A}})} \right)$$
Since $\lambda_{\max}(\tilde{\mathbf{A}}) = 1$, the upper bound is $\phi < 1$.

### 12.4.3 SAR vs. CAR: A Structural Comparison

The fundamental structural difference between SAR and CAR lies in their **precision matrices**:
- **CAR Precision:** $\mathbf{Q}_{\text{CAR}} = \tau^{-2} (\mathbf{D} - \phi \mathbf{A})$. 
  The $(i,j)$-th off-diagonal element is non-zero **if and only if** regions $i$ and $j$ are direct neighbors ($A_{ij} = 1$). CAR models are exact **Gaussian Markov Random Fields (GMRF)**: the sparsity pattern of the precision matrix mirrors the graph topology.
- **SAR Precision:** $\mathbf{Q}_{\text{SAR}} = \sigma^{-2}(\mathbf{I} - \rho \mathbf{W})^\top (\mathbf{I} - \rho \mathbf{W})$.
  As shown in §12.3.2, $\mathbf{Q}_{\text{SAR}}$ contains non-zero entries for pairs sharing a common neighbor ($\mathbf{W}^\top\mathbf{W}$). SAR induces fill-in, destroying the immediate nearest-neighbor Markov property.

---

## 12.5 Intrinsic Autoregressive (ICAR) and Besag-York-Mollié (BYM) Models

In regional disease mapping and spatial epidemiology, empirical evidence indicates that spatial correlation among adjacent zones is often very strong. What happens if we push the CAR spatial parameter to its theoretical limit: $\phi \to 1$?

### 12.5.1 The Intrinsic CAR (ICAR) Model

Setting $\phi = 1$ in the CAR precision matrix yields the **Intrinsic Autoregressive (ICAR)** precision matrix:
$$\mathbf{Q}_{\text{ICAR}} = \frac{1}{\sigma_u^2} (\mathbf{D} - \mathbf{A}) = \frac{1}{\sigma_u^2} \mathbf{L}$$
where $\mathbf{L} = \mathbf{D} - \mathbf{A}$ is the celebrated **Graph Laplacian** from spectral graph theory.

#### The Nullspace of ICAR
Consider multiplying the Graph Laplacian by a constant vector $\mathbf{1} = [1, 1, \dots, 1]^\top$:
$$[\mathbf{L} \mathbf{1}]_i = \sum_{j=1}^n L_{ij} \cdot 1 = d_i - \sum_{j \in \mathcal{N}(i)} 1 = d_i - d_i = 0$$
Thus:
$$\mathbf{Q}_{\text{ICAR}} \mathbf{1} = \mathbf{0}$$

The vector $\mathbf{1}$ lies in the **nullspace** of $\mathbf{Q}_{\text{ICAR}}$. Consequently:
1. $\det(\mathbf{Q}_{\text{ICAR}}) = 0$: the precision matrix is singular and cannot be inverted to obtain a standard marginal covariance matrix.
2. The joint distribution is **improper**: adding an arbitrary constant $c$ to every spatial random effect ($u_i \leftarrow u_i + c$) leaves the density completely unchanged.
3. For an adjacency graph with $c$ connected components, the rank of $\mathbf{Q}_{\text{ICAR}}$ is:
   $$\operatorname{rank}(\mathbf{Q}_{\text{ICAR}}) = n - c$$
   For a single fully-connected study map ($c = 1$), the rank deficiency is exactly 1.

#### Resolving the Improper Distribution: The Sum-to-Zero Constraint
To use the ICAR model in a valid hierarchical Bayesian model, we must fix the arbitrary spatial location intercept. We impose the **sum-to-zero constraint**:
$$\sum_{i=1}^n u_i = 0$$

Under this linear constraint, the distribution becomes proper on the $(n-1)$-dimensional hyperplane orthogonal to $\mathbf{1}$.

> [!WARNING] Islands in ICAR Models
> If a study domain contains disconnected subgraphs or isolated islands ($c > 1$), each connected component has its own independent nullspace vector ($\mathbf{1}_{\mathcal{C}_k}$). An overall constraint $\sum_{i=1}^n u_i = 0$ is **insufficient**: each component requires an independent sum-to-zero constraint $\sum_{i \in \mathcal{C}_k} u_i = 0$. Completely isolated regions ($d_i = 0$) cannot borrow information from neighbors and must be modeled solely via unstructured random effects.

```
================================================================================
FROM THE LECTURE HALL: Box 12.4
The Besag-York-Mollié (BYM) Model in Disease Mapping
================================================================================
Consider mapping rare cancer incidence across 1,000 counties. In each county i,
we observe Y_i cases and calculate the expected cases E_i based on standard
demographic age-sex rates.

The classical Standardized Mortality Ratio (SMR) is:
                         SMR_i = Y_i / E_i

In sparsely populated rural counties, E_i might be 0.1. A single case produces
SMR_i = 10.0 (a dramatic false alarm), while zero cases produces SMR_i = 0.0.
The raw SMR is plagued by extreme small-number variance!

The Besag-York-Mollié (BYM; Besag et al., 1991) framework solves this via a
hierarchical Poisson-lognormal GLMM:
                 Y_i | eta_i ~ Poisson( E_i * exp(eta_i) )
                 eta_i = beta_0 + u_i + v_i

where the spatial log-relative risk is decomposed into TWO random components:
1. Spatial Structured Effect:   u_i ~ ICAR(sigma_u^2)
   Smooths small-number noise by borrowing strength from adjacent geographic
   neighbors.
2. Unstructured Heterogeneity:  v_i ~ N(0, sigma_v^2)
   Captures county-specific, unmeasured risk factors (overdispersion) that do
   not follow geographic borders.

This elegant convolution model remains the gold standard in modern spatial
epidemiology and public health surveillance.
================================================================================
```

---

## 12.6 Spatial Moving Average (SMA) and Master Comparison Table

To complete our survey of discrete lattice formulations, we examine the spatial moving average.

### 12.6.1 Spatial Moving Average (SMA)

The **Spatial Moving Average (SMA)** model applies a localized finite impulse response operator to white noise:
$$\mathbf{Y} = (\mathbf{I} + \theta \mathbf{W}) \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$
where $\theta$ is the spatial moving average parameter and $\mathbf{W}$ is a spatial weight matrix.

#### Covariance Structure
The covariance matrix of the SMA model is:
$$\boldsymbol{\Sigma}_{\text{SMA}} = \operatorname{Var}(\mathbf{Y}) = \sigma^2 (\mathbf{I} + \theta \mathbf{W})(\mathbf{I} + \theta \mathbf{W})^\top = \sigma^2 \left[ \mathbf{I} + \theta(\mathbf{W} + \mathbf{W}^\top) + \theta^2 \mathbf{W}\mathbf{W}^\top \right]$$

#### Exact Spatial Correlation Range
Consider the covariance between regions $i$ and $j$ ($i \neq j$):
$$[\boldsymbol{\Sigma}_{\text{SMA}}]_{ij} = \sigma^2 \left[ \theta(w_{ij} + w_{ji}) + \theta^2 \sum_{k=1}^n w_{ik} w_{jk} \right]$$

This expression yields an important insight:
- If regions $i$ and $j$ are direct neighbors ($w_{ij} > 0$ or $w_{ji} > 0$), their covariance is nonzero.
- If regions $i$ and $j$ are **second-order neighbors** sharing at least one common neighbor $k$ ($\sum_k w_{ik}w_{jk} > 0$), their covariance is **also nonzero**!
- If regions $i$ and $j$ share no common neighbors ($\sum_k w_{ik}w_{jk} = 0$), their covariance is **strictly zero**.

> [!NOTE] Nonzero Range of SMA
> Unlike continuous moving averages with a hard Euclidean cutoff, the discrete SMA model has a correlation range that spans **up to two steps on the graph topology** (direct neighbors and shared neighbors). It is strictly zero only beyond second-order graph separation.

### 12.6.2 Master Comparison: SAR vs. CAR vs. ICAR vs. SMA

The table below contrasts all four discrete lattice modeling frameworks across their mathematical foundations:

| Property | SAR (Simultaneous) | Proper CAR (Conditional) | ICAR (Intrinsic) | SMA (Moving Average) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Formulation** | $(\mathbf{I} - \rho\mathbf{W})\mathbf{Y} = \boldsymbol{\epsilon}$ | $Y_i \mid \mathbf{Y}_{-i} \sim \mathcal{N}(\dots)$ | $\phi = 1$ in CAR | $\mathbf{Y} = (\mathbf{I} + \theta\mathbf{W})\boldsymbol{\epsilon}$ |
| **Covariance $\boldsymbol{\Sigma}$** | $\sigma^2 [(\mathbf{I} - \rho\mathbf{W})^\top (\mathbf{I} - \rho\mathbf{W})]^{-1}$ | $\tau^2 (\mathbf{D} - \phi \mathbf{A})^{-1}$ | Singular (pseudoinverse on $\mathbf{1}^\perp$) | $\sigma^2 (\mathbf{I} + \theta\mathbf{W})(\mathbf{I} + \theta\mathbf{W})^\top$ |
| **Precision $\mathbf{Q} = \boldsymbol{\Sigma}^{-1}$** | $\sigma^{-2} (\mathbf{I} - \rho\mathbf{W})^\top (\mathbf{I} - \rho\mathbf{W})$ | $\tau^{-2} (\mathbf{D} - \phi \mathbf{A})$ | $\sigma_u^{-2} (\mathbf{D} - \mathbf{A})$ | Dense inverse $[(\mathbf{I} + \theta\mathbf{W})(\mathbf{I} + \theta\mathbf{W})^\top]^{-1}$ |
| **Sparsity of $\mathbf{Q}$** | Dense to 2nd-order graph neighbors | Exact graph neighbors | Exact graph neighbors | Dense across graph |
| **Sparsity of $\boldsymbol{\Sigma}$** | Dense (infinite spatial decay) | Dense (infinite spatial decay) | Dense | **Sparse** (zero beyond 2nd-order neighbors) |
| **Parameter Domain** | $\rho \in (\lambda_{\min}^{-1}, \lambda_{\max}^{-1})$ | $\phi \in (\lambda_{\min}^{-1}, 1)$ | $\phi = 1$ (fixed) | $\theta \in \mathbb{R}$ |
| **Determinant Penalty** | $\ln\|\det(\mathbf{I} - \rho\mathbf{W})\|$ | $\frac{1}{2}\ln\|\det(\mathbf{D} - \phi\mathbf{A})\|$ | Undefined (requires $n-1$ non-zero roots) | $\ln\|\det(\mathbf{I} + \theta\mathbf{W})\|$ |
| **Primary Domain** | Spatial Econometrics, Real Estate | Bayesian Image Analysis | Disease Mapping (BYM) | Local atmospheric filtering |

---

## 12.7 Worked Analytical Exercises

### Exercise 12.1: Hand Calculation of Moran's $I$ and Geary's $c$

**Problem:** Consider the 5-region linear chain Lattice $\alpha$ ($A_1 - A_2 - A_3 - A_4 - A_5$) with raw binary contiguity matrix $\mathbf{A}$. Given the observation vector $\mathbf{y} = [1, 2, 3, 4, 5]^\top$:
1. Compute the sample mean $\bar{y}$ and centered deviation vector $\mathbf{z}$.
2. Compute the total sum of weights $S_0$ and the denominator $\sum_{i=1}^5 z_i^2$.
3. Compute global Moran's $I$ and compare it to $\mathbb{E}[I]$.
4. Compute Geary's $c$ and interpret the nature of the spatial pattern.

**Solution:**

**Step 1: Deviations from the Mean**
$$\bar{y} = \frac{1 + 2 + 3 + 4 + 5}{5} = \frac{15}{5} = 3.0$$
$$\mathbf{z} = \mathbf{y} - \bar{y}\mathbf{1} = [1-3, \; 2-3, \; 3-3, \; 4-3, \; 5-3]^\top = [-2, \; -1, \; 0, \; 1, \; 2]^\top$$

**Step 2: Sum of Squares and Weights**
$$\sum_{i=1}^5 z_i^2 = (-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 4 + 1 + 0 + 1 + 4 = 10$$

The edges of the graph are $(1,2), (2,3), (3,4), (4,5)$ and their symmetric pairs $(2,1), (3,2), (4,3), (5,4)$. There are 4 undirected edges, so:
$$S_0 = \sum_{i=1}^5 \sum_{j=1}^5 A_{ij} = 2 \times 4 = 8$$

**Step 3: Moran's $I$**
We evaluate the quadratic form $\mathbf{z}^\top \mathbf{A} \mathbf{z} = \sum_{i=1}^5 \sum_{j=1}^5 A_{ij} z_i z_j$:
$$\mathbf{z}^\top \mathbf{A} \mathbf{z} = 2 \left[ z_1 z_2 + z_2 z_3 + z_3 z_4 + z_4 z_5 \right]$$
$$z_1 z_2 = (-2)(-1) = 2$$
$$z_2 z_3 = (-1)(0) = 0$$
$$z_3 z_4 = (0)(1) = 0$$
$$z_4 z_5 = (1)(2) = 2$$
$$\mathbf{z}^\top \mathbf{A} \mathbf{z} = 2 \left[ 2 + 0 + 0 + 2 \right] = 2(4) = 8$$

Plugging into the formula for Moran's $I$:
$$I = \frac{n}{S_0} \frac{\mathbf{z}^\top \mathbf{A} \mathbf{z}}{\mathbf{z}^\top \mathbf{z}} = \frac{5}{8} \times \frac{8}{10} = \frac{5}{10} = 0.50$$

Under the null hypothesis of no spatial autocorrelation:
$$\mathbb{E}[I] = -\frac{1}{n - 1} = -\frac{1}{5 - 1} = -\frac{1}{4} = -0.25$$
Since $I = 0.50 \gg -0.25$, the data exhibits strong, positive spatial autocorrelation (an obvious linear trend across the chain).

**Step 4: Geary's $c$**
The numerator of Geary's $c$ sums squared differences across all connected edges:
$$\sum_{i=1}^5 \sum_{j=1}^5 A_{ij} (y_i - y_j)^2 = 2 \left[ (y_1 - y_2)^2 + (y_2 - y_3)^2 + (y_3 - y_4)^2 + (y_4 - y_5)^2 \right]$$
$$(1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2 = 1 + 1 + 1 + 1 = 4$$
$$\sum_{i=1}^5 \sum_{j=1}^5 A_{ij} (y_i - y_j)^2 = 2 \times 4 = 8$$

Plugging into the formula for Geary's $c$:
$$c = \frac{n - 1}{2 S_0} \frac{\sum_{i,j} A_{ij} (y_i - y_j)^2}{\sum_i z_i^2} = \frac{5 - 1}{2(8)} \times \frac{8}{10} = \frac{4}{16} \times \frac{4}{5} = \frac{1}{4} \times \frac{4}{5} = \frac{1}{5} = 0.20$$

Under $H_0$, $\mathbb{E}[c] = 1.0$. Because $c = 0.20 \ll 1.0$, neighboring values are substantially closer to each other than expected under random shuffling, confirming strong positive spatial autocorrelation.

---

### Exercise 12.2: Brook's Lemma Symmetry Proof for a 2-Region CAR System

**Problem:** Consider a 2-region map ($n=2$) where the full conditional distributions are specified as:
$$Y_1 \mid Y_2 = y_2 \sim \mathcal{N}\left( c_{12} y_2, \; \tau_1^2 \right)$$
$$Y_2 \mid Y_1 = y_1 \sim \mathcal{N}\left( c_{21} y_1, \; \tau_2^2 \right)$$
Using Brook's Lemma with reference point $\mathbf{y}_0 = [0, 0]^\top$, prove that a valid joint bivariate Gaussian distribution exists if and only if:
$$\frac{c_{12}}{\tau_1^2} = \frac{c_{21}}{\tau_2^2}$$

**Solution:**

**Step 1: Apply Brook's Expansion Along Path 1**
We expand the joint density $p(y_1, y_2)$ relative to $(0,0)$ by changing coordinate 1 first, then coordinate 2:
$$\frac{p(y_1, y_2)}{p(0, 0)} = \frac{p(y_1 \mid y_2 = 0)}{p(0 \mid y_2 = 0)} \times \frac{p(y_2 \mid y_1)}{p(0 \mid y_1)}$$

Using the univariate Gaussian conditional density:
$$p(y_1 \mid y_2 = 0) = \frac{1}{\sqrt{2\pi \tau_1^2}} \exp\left( -\frac{y_1^2}{2\tau_1^2} \right), \quad p(0 \mid y_2 = 0) = \frac{1}{\sqrt{2\pi \tau_1^2}}$$
Thus the first ratio is:
$$\frac{p(y_1 \mid y_2 = 0)}{p(0 \mid y_2 = 0)} = \exp\left( -\frac{y_1^2}{2\tau_1^2} \right)$$

For the second ratio:
$$\frac{p(y_2 \mid y_1)}{p(0 \mid y_1)} = \frac{\frac{1}{\sqrt{2\pi \tau_2^2}} \exp\left( -\frac{(y_2 - c_{21}y_1)^2}{2\tau_2^2} \right)}{\frac{1}{\sqrt{2\pi \tau_2^2}} \exp\left( -\frac{(-c_{21}y_1)^2}{2\tau_2^2} \right)} = \exp\left( -\frac{y_2^2 - 2 c_{21} y_1 y_2}{2\tau_2^2} \right)$$

Multiplying these two ratios yields the joint exponent:
$$\ln\left[ \frac{p(y_1, y_2)}{p(0, 0)} \right] = -\frac{y_1^2}{2\tau_1^2} - \frac{y_2^2}{2\tau_2^2} + \frac{c_{21}}{\tau_2^2} y_1 y_2$$

**Step 2: Apply Brook's Expansion Along Path 2**
Now expand the exact same joint density by changing coordinate 2 first, then coordinate 1:
$$\frac{p(y_1, y_2)}{p(0, 0)} = \frac{p(y_2 \mid y_1 = 0)}{p(0 \mid y_1 = 0)} \times \frac{p(y_1 \mid y_2)}{p(0 \mid y_2)}$$

By exact symmetry of indices:
$$\ln\left[ \frac{p(y_1, y_2)}{p(0, 0)} \right] = -\frac{y_2^2}{2\tau_2^2} - \frac{y_1^2}{2\tau_1^2} + \frac{c_{12}}{\tau_1^2} y_1 y_2$$

**Step 3: Equate the Expansions**
For a well-defined joint density to exist, the two paths must produce identical results for all $(y_1, y_2) \in \mathbb{R}^2$:
$$-\frac{y_1^2}{2\tau_1^2} - \frac{y_2^2}{2\tau_2^2} + \frac{c_{21}}{\tau_2^2} y_1 y_2 = -\frac{y_1^2}{2\tau_1^2} - \frac{y_2^2}{2\tau_2^2} + \frac{c_{12}}{\tau_1^2} y_1 y_2$$
Subtracting the diagonal terms leaves:
$$\frac{c_{21}}{\tau_2^2} y_1 y_2 = \frac{c_{12}}{\tau_1^2} y_1 y_2 \quad \forall (y_1, y_2)$$
Dividing by $y_1 y_2$ yields the necessary and sufficient condition:
$$\frac{c_{12}}{\tau_1^2} = \frac{c_{21}}{\tau_2^2}$$
This proves that asymmetric conditional dependency cannot form a coherent joint Gaussian probability density.

---

### Exercise 12.3: Precision and Covariance Comparison: SAR vs. CAR on a 3-Region Line

**Problem:** Consider a 3-region linear graph $1 - 2 - 3$. The binary contiguity matrix is:
$$\mathbf{A} = \begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}$$
1. Write down the CAR precision matrix $\mathbf{Q}_{\text{CAR}}$ assuming conditional variance $\tau^2 = 1$ and parameter $\phi$. Verify whether regions 1 and 3 are conditionally independent given region 2.
2. Formulate the SAR model $\mathbf{Y} = \rho \mathbf{A} \mathbf{Y} + \boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Derive the analytical SAR precision matrix $\mathbf{Q}_{\text{SAR}}$.
3. Show that regions 1 and 3 are **not** conditionally independent in the SAR model.

**Solution:**

**Step 1: The CAR Precision Matrix**
For unnormalized contiguity with unit variance, the CAR precision matrix is:
$$\mathbf{Q}_{\text{CAR}} = \mathbf{I} - \phi \mathbf{A} = \begin{bmatrix}
1 & -\phi & 0 \\
-\phi & 1 & -\phi \\
0 & -\phi & 1
\end{bmatrix}$$
In a Gaussian graphical model, $[\mathbf{Q}]_{ij} = 0$ if and only if $Y_i \perp Y_j \mid \mathbf{Y}_{-(i,j)}$.
Here, $[\mathbf{Q}_{\text{CAR}}]_{13} = [\mathbf{Q}_{\text{CAR}}]_{31} = 0$.
Therefore, in the CAR model, regions 1 and 3 are **conditionally independent** given the intervening region 2 ($Y_1 \perp Y_3 \mid Y_2$), precisely honoring the graph topology.

**Step 2: The SAR Precision Matrix**
Let $\mathbf{B} = \mathbf{I} - \rho \mathbf{A}$:
$$\mathbf{B} = \begin{bmatrix}
1 & -\rho & 0 \\
-\rho & 1 & -\rho \\
0 & -\rho & 1
\end{bmatrix}$$
Since $\sigma^2 = 1$, the SAR precision matrix is $\mathbf{Q}_{\text{SAR}} = \mathbf{B}^\top \mathbf{B} = \mathbf{B}^2$ (as $\mathbf{B}$ is symmetric):
$$\mathbf{Q}_{\text{SAR}} = \begin{bmatrix}
1 & -\rho & 0 \\
-\rho & 1 & -\rho \\
0 & -\rho & 1
\end{bmatrix} \begin{bmatrix}
1 & -\rho & 0 \\
-\rho & 1 & -\rho \\
0 & -\rho & 1
\end{bmatrix}$$

We compute each matrix entry:
- Row 1, Col 1: $1(1) + (-\rho)(-\rho) + 0(0) = 1 + \rho^2$
- Row 1, Col 2: $1(-\rho) + (-\rho)(1) + 0(-\rho) = -2\rho$
- Row 1, Col 3: $1(0) + (-\rho)(-\rho) + 0(1) = \rho^2$
- Row 2, Col 2: $(-\rho)(-\rho) + 1(1) + (-\rho)(-\rho) = 1 + 2\rho^2$
- Row 2, Col 3: $(-\rho)(0) + 1(-\rho) + (-\rho)(1) = -2\rho$
- Row 3, Col 3: $0(0) + (-\rho)(-\rho) + 1(1) = 1 + \rho^2$

Assembling the complete SAR precision matrix:
$$\mathbf{Q}_{\text{SAR}} = \begin{bmatrix}
1 + \rho^2 & -2\rho & \rho^2 \\
-2\rho & 1 + 2\rho^2 & -2\rho \\
\rho^2 & -2\rho & 1 + \rho^2
\end{bmatrix}$$

**Step 3: Conditional Dependence in SAR**
Look at the $(1,3)$ and $(3,1)$ entries:
$$[\mathbf{Q}_{\text{SAR}}]_{13} = [\mathbf{Q}_{\text{SAR}}]_{31} = \rho^2$$
For any nonzero spatial autocorrelation parameter $\rho \neq 0$:
$$[\mathbf{Q}_{\text{SAR}}]_{13} = \rho^2 \neq 0$$
Because the off-diagonal precision entry between regions 1 and 3 is non-zero, $Y_1$ and $Y_3$ are **conditionally dependent** given $Y_2$ ($Y_1 \not\perp Y_3 \mid Y_2$). 

The simultaneous autoregression introduces direct conditional dependence between second-order neighbors, formally demonstrating why SAR models do not preserve the nearest-neighbor Markov graph property of CAR models.

---

## 12.8 Computational Implementation

The following complete, self-contained Python module implements:
1. `build_rook_grid`: Constructs adjacency graphs and row-standardized weight matrices with automatic island handling.
2. `moran_i`: Calculates global Moran's $I$, the analytical Cliff-Ord expectation, permutation variance, and asymptotic $Z$-score.
3. `moran_permutation_test`: Performs an exact Monte Carlo permutation test over 999 randomized spatial reassignments.
4. `geary_c`: Computes Geary's $c$ contiguity ratio.
5. `simulate_sar`: Generates exact multivariate Gaussian realizations from the SAR model via linear system solution.
6. `fit_sar_profile_mle`: Fits the SAR parameter $\rho$ and noise variance $\sigma^2$ by maximizing the profile log-likelihood using exact Gaussian elimination for the Jacobian log-determinant.

```python
"""
Lattice and Areal Spatial Modeling Module
Pure standard-library implementation of discrete spatial autoregression.
"""

import math
import random


def build_rook_grid(rows, cols):
    """
    Constructs a regular lattice with Rook contiguity (shared edges).
    
    Parameters
    ----------
    rows : int
        Number of lattice rows.
    cols : int
        Number of lattice columns.
        
    Returns
    -------
    adj : dict
        Adjacency list mapping node index -> list of neighbor indices.
    W : list of list of float
        Row-standardized spatial weight matrix (n x n).
    """
    n = rows * cols
    adj = {i: [] for i in range(n)}
    
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if r > 0:
                adj[idx].append((r - 1) * cols + c)
            if r < rows - 1:
                adj[idx].append((r + 1) * cols + c)
            if c > 0:
                adj[idx].append(r * cols + (c - 1))
            if c < cols - 1:
                adj[idx].append(r * cols + (c + 1))
                
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        degree = len(adj[i])
        if degree > 0:
            for j in adj[i]:
                W[i][j] = 1.0 / degree
        # If degree == 0 (island), row remains all zeros
        
    return adj, W


def moran_i(y, W):
    """
    Calculates Global Moran's I with analytical Cliff-Ord permutation variance.
    
    Parameters
    ----------
    y : list of float
        Observed response vector of length n.
    W : list of list of float
        Spatial weight matrix of dimension n x n.
        
    Returns
    -------
    I : float
        Observed Moran's I statistic.
    z_score : float
        Standardized normal test statistic.
    p_value : float
        Two-sided p-value under the permutation null.
    """
    n = len(y)
    mean_y = sum(y) / n
    z = [val - mean_y for val in y]
    
    s0 = sum(W[i][j] for i in range(n) for j in range(n))
    numerator = sum(W[i][j] * z[i] * z[j] for i in range(n) for j in range(n))
    denominator = sum(zi ** 2 for zi in z)
    
    if denominator == 0.0 or s0 == 0.0:
        return 0.0, 0.0, 1.0
        
    I = (n / s0) * (numerator / denominator)
    e_I = -1.0 / (n - 1)
    
    # Cliff-Ord variance terms under the permutation assumption
    s1 = 0.5 * sum((W[i][j] + W[j][i]) ** 2 for i in range(n) for j in range(n))
    s2 = sum((sum(W[i][j] + W[j][i] for j in range(n))) ** 2 for i in range(n))
    
    m2 = denominator / n
    m4 = sum(zi ** 4 for zi in z) / n
    b2 = m4 / (m2 ** 2) if m2 > 0 else 3.0
    
    term_a = n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * (s0 ** 2))
    term_b = b2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * (s0 ** 2))
    term_c = (n - 1) * (n - 2) * (n - 3) * (s0 ** 2)
    
    var_I = (term_a - term_b) / term_c - (e_I ** 2)
    z_score = (I - e_I) / math.sqrt(var_I) if var_I > 0 else 0.0
    
    # Two-sided p-value using the standard normal error function
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
    return I, z_score, p_value


def moran_permutation_test(y, W, n_perms=999, seed=42):
    """
    Performs an exact Monte Carlo permutation test for Moran's I.
    """
    rng = random.Random(seed)
    n = len(y)
    mean_y = sum(y) / n
    z = [val - mean_y for val in y]
    
    s0 = sum(W[i][j] for i in range(n) for j in range(n))
    denominator = sum(zi ** 2 for zi in z)
    obs_num = sum(W[i][j] * z[i] * z[j] for i in range(n) for j in range(n))
    obs_I = (n / s0) * (obs_num / denominator)
    
    greater_count = 0
    perm_z = list(z)
    for _ in range(n_perms):
        rng.shuffle(perm_z)
        perm_num = sum(W[i][j] * perm_z[i] * perm_z[j] for i in range(n) for j in range(n))
        perm_I = (n / s0) * (perm_num / denominator)
        if perm_I >= obs_I:
            greater_count += 1
            
    p_perm = (greater_count + 1) / (n_perms + 1)
    return obs_I, p_perm


def geary_c(y, W):
    """
    Calculates Geary's c spatial contiguity ratio.
    """
    n = len(y)
    mean_y = sum(y) / n
    z = [val - mean_y for val in y]
    
    s0 = sum(W[i][j] for i in range(n) for j in range(n))
    denominator = sum(zi ** 2 for zi in z)
    diff_num = sum(W[i][j] * ((y[i] - y[j]) ** 2) for i in range(n) for j in range(n))
    
    if denominator == 0.0 or s0 == 0.0:
        return 1.0
        
    c = ((n - 1) / (2.0 * s0)) * (diff_num / denominator)
    return c


# Linear Algebra Routines (Gaussian Elimination)

def _solve_system(A, b):
    """Solves A x = b via Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    
    for col in range(n):
        max_row = col
        max_val = abs(M[col][col])
        for r in range(col + 1, n):
            if abs(M[r][col]) > max_val:
                max_val = abs(M[r][col])
                max_row = r
                
        if max_val < 1e-12:
            raise ValueError("Singular matrix encountered in linear solver.")
            
        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]
            
        pivot = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= pivot
            
        for r in range(n):
            if r != col and abs(M[r][col]) > 1e-12:
                factor = M[r][col]
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]
                    
    return [M[i][n] for i in range(n)]


def _log_det_abs(matrix):
    """Calculates ln|det(A)| via Gaussian elimination."""
    n = len(matrix)
    A = [row[:] for row in matrix]
    log_det = 0.0
    
    for col in range(n):
        max_row = col
        max_val = abs(A[col][col])
        for r in range(col + 1, n):
            if abs(A[r][col]) > max_val:
                max_val = abs(A[r][col])
                max_row = r
                
        if max_val < 1e-12:
            return -float('inf')
            
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]
            
        pivot = A[col][col]
        log_det += math.log(abs(pivot))
        
        for r in range(col + 1, n):
            factor = A[r][col] / pivot
            for c in range(col + 1, n):
                A[r][c] -= factor * A[col][c]
                
    return log_det


def simulate_sar(W, rho, sigma=1.0, seed=42):
    """
    Simulates a pure Simultaneous Autoregressive (SAR) spatial process:
        Y = (I - rho * W)^{-1} epsilon,   epsilon ~ N(0, sigma^2 * I)
    """
    rng = random.Random(seed)
    n = len(W)
    eps = [rng.gauss(0.0, sigma) for _ in range(n)]
    
    # Construct B = I - rho * W
    B = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(n)] for i in range(n)]
    y = _solve_system(B, eps)
    return y


def fit_sar_profile_mle(y, W, rho_bounds=(-0.85, 0.85), n_grid=71):
    """
    Fits the SAR parameter rho via Profile Maximum Likelihood grid search.
    
    Maximizes:
        ln L_p(rho) = -n/2 * ln(2*pi) - n/2 * ln(sigma^2(rho)) - n/2 + ln|det(I - rho*W)|
    """
    n = len(y)
    r_min, r_max = rho_bounds
    step = (r_max - r_min) / (n_grid - 1)
    
    best_rho = 0.0
    best_ll = -float('inf')
    best_sigma2 = 1.0
    
    for k in range(n_grid):
        rho = r_min + k * step
        B = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(n)] for i in range(n)]
        
        ldet = _log_det_abs(B)
        if ldet == -float('inf'):
            continue
            
        # Residuals: eps = (I - rho * W) * y
        eps = [sum(B[i][j] * y[j] for j in range(n)) for i in range(n)]
        rss = sum(e ** 2 for e in eps)
        sigma2 = rss / n
        
        if sigma2 <= 1e-12:
            continue
            
        ll = -0.5 * n * (math.log(2.0 * math.pi) + 1.0 + math.log(sigma2)) + ldet
        if ll > best_ll:
            best_ll = ll
            best_rho = rho
            best_sigma2 = sigma2
            
    return {
        'rho_mle': best_rho,
        'sigma2_mle': best_sigma2,
        'log_likelihood': best_ll
    }


if __name__ == "__main__":
    print("=" * 72)
    print("CHAPTER 12: LATTICE AND AREAL MODELING DEMONSTRATION")
    print("=" * 72)
    
    # 1. Verification of Exercise 12.1 on Lattice Alpha (5-region chain)
    print("\n--- 1. Verification on Lattice Alpha (Exercise 12.1) ---")
    adj_alpha = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }
    A_alpha = [[0.0] * 5 for _ in range(5)]
    for i in range(5):
        for j in adj_alpha[i]:
            A_alpha[i][j] = 1.0
            
    y_alpha = [1.0, 2.0, 3.0, 4.0, 5.0]
    I_alpha, z_alpha, p_alpha = moran_i(y_alpha, A_alpha)
    c_alpha = geary_c(y_alpha, A_alpha)
    
    print(f"Data: {y_alpha}")
    print(f"Calculated Moran's I : {I_alpha:.4f}  (Expected under H0: -0.2500)")
    print(f"Analytical Z-score   : {z_alpha:.4f}  (p-value: {p_alpha:.4f})")
    print(f"Calculated Geary's c : {c_alpha:.4f}  (Expected under H0:  1.0000)")
    
    # 2. SAR Simulation and Profile MLE on a 6x6 Lattice (36 zones)
    print("\n--- 2. SAR Simulation & Profile MLE (6x6 Grid, N=36) ---")
    rows, cols = 6, 6
    adj_grid, W_grid = build_rook_grid(rows, cols)
    
    true_rho = 0.65
    true_sigma = 1.0
    print(f"True SAR Parameters  : rho = {true_rho:.2f}, sigma = {true_sigma:.2f}")
    
    y_sar = simulate_sar(W_grid, rho=true_rho, sigma=true_sigma, seed=123)
    
    # ESDA on simulated SAR field
    I_sar, z_sar, p_sar = moran_i(y_sar, W_grid)
    _, p_perm = moran_permutation_test(y_sar, W_grid, n_perms=999, seed=123)
    c_sar = geary_c(y_sar, W_grid)
    
    print(f"Observed Moran's I   : {I_sar:.4f} (Cliff-Ord Z = {z_sar:.2f}, p = {p_sar:.4e})")
    print(f"999-Permutation p-val: {p_perm:.4f}")
    print(f"Observed Geary's c   : {c_sar:.4f}")
    
    # Profile Likelihood Optimization
    fit_res = fit_sar_profile_mle(y_sar, W_grid, rho_bounds=(-0.85, 0.85), n_grid=86)
    print("\nProfile MLE Estimation Results:")
    print(f"Estimated rho (MLE)  : {fit_res['rho_mle']:.3f}")
    print(f"Estimated sigma^2    : {fit_res['sigma2_mle']:.3f} (True: {true_sigma**2:.3f})")
    print(f"Profile Log-Lik      : {fit_res['log_likelihood']:.2f}")
    print("=" * 72)
```

---

## 12.9 Chapter Summary and Practical Guidance

This chapter bridged spatial analysis from continuous Euclidean distances to discrete neighborhood topologies:
1. **Spatial Representation:** Areal data aggregates continuous phenomena into irregular polygons or regular grids. Spatial relationships are encoded through adjacency matrices $\mathbf{A}$ and row-standardized weight matrices $\mathbf{W}^* = \mathbf{D}^{-1}\mathbf{A}$.
2. **Exploratory Metrics:** Moran's $I$ measures spatial autocorrelation as the regression slope of the spatial lag $\mathbf{W}^*\mathbf{z}$ on $\mathbf{z}$. Its expected value under spatial randomness is $\mathbb{E}[I] = -1/(n-1)$. Geary's $c$ measures local neighbor variance, with $\mathbb{E}[c] = 1$.
3. **The Simultaneous Paradigm (SAR):** Models spatial feedback simultaneously across the entire map: $(\mathbf{I} - \rho\mathbf{W})\mathbf{Y} = \boldsymbol{\epsilon}$. Invertibility requires $\rho \in (\lambda_{\min}^{-1}, \lambda_{\max}^{-1})$. Ordinary least squares fails due to simultaneity; maximum likelihood estimation requires the Jacobian log-determinant $\ln|\det(\mathbf{I} - \rho\mathbf{W})|$ to avoid boundary collapse.
4. **The Conditional Paradigm (CAR):** Specifies localized conditional distributions $Y_i \mid \mathbf{Y}_{-i}$. By Brook's Lemma, coherent Gaussian joint distributions require symmetry ($c_{ij}/\tau_i^2 = c_{ji}/\tau_j^2$). The resulting CAR precision matrix directly mirrors the neighborhood graph: $\mathbf{Q}_{\text{CAR}} = \tau^{-2}(\mathbf{D} - \phi \mathbf{A})$.
5. **Hierarchical Disease Mapping (ICAR & BYM):** Setting $\phi = 1$ yields the Intrinsic CAR model governed by the Graph Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{A}$. Because $\mathbf{L}\mathbf{1} = \mathbf{0}$, the model is improper and requires a sum-to-zero constraint $\sum_i u_i = 0$. The Besag-York-Mollié framework combines structured ICAR spatial effects with unstructured random noise to stabilize volatile small-area rate estimates.
6. **Moving Averages (SMA):** Spatial moving averages yield covariance that is strictly zero beyond second-order graph neighbors, contrasting with the dense global covariance of SAR and CAR models.
