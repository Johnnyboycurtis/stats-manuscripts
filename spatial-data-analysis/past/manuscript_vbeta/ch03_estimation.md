# Chapter 3: Nonparametric Covariance and Semivariogram Estimation

In the three-layer linear decomposition established in Chapter 1:
$$
Z(\mathbf{s}) = \mu(\mathbf{s}) + \delta(\mathbf{s}) + \epsilon(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta} + \delta(\mathbf{s}) + \epsilon(\mathbf{s}), \quad \mathbf{s} \in D \subset \mathbb{R}^d
$$
the stochastic behavior of the spatial random field is governed by the second-order dependence structure of the stationary residual field $\delta(\mathbf{s})$ and the microscale noise $\epsilon(\mathbf{s})$. Before specifying and fitting a rigid parametric covariance family $\mathbf{V}(\boldsymbol{\theta}) = \mathbf{C}(\boldsymbol{\theta}) + \tau^2 \mathbf{I}_n$, the spatial statistician must examine the spatial continuity empirically.

Nonparametric estimation serves as the primary diagnostic lens through which the spatial dependence structure is discovered. This chapter establishes the mathematical and computational foundations of nonparametric semivariogram and covariance estimation from finite, irregularly spaced spatial data.

By the end of this chapter, you will be able to:
1. Construct, plot, and interpret an empirical semivariogram cloud and binned semivariogram from irregular coordinate data.
2. Diagnose why sample covariances suffer systematic downward bias under spatial autocorrelation ("variance theft") and why the semivariogram avoids it.
3. Quantify sampling variability and inter-lag correlation under Gaussianity using Isserlis' identity and Genton's design matrix.
4. Deploy robust modulus estimators (Cressie–Hawkins) when localized outliers or heavy tails distort spatial increments, while recognizing their finite-sample assumptions.
5. Identify the signatures of asymmetric advective transport using directional cross-covariances and Clark's pseudo-cross variograms.

> [!NOTE]
> **Chapter at a Glance**
> - **§3.1 Nonparametric Estimators and Spatial Lag Binning:** Bridges classical pairwise-difference variance to spatial lag bins; defines Matheron's estimator and Genton's graph Laplacian design matrix; provides recipes for bandwidth and bin selection.
> - **§3.2 Expectations, Unbiasedness, and Systematic Downward Bias:** Proves the exact unbiasedness of Matheron's semivariogram versus the permanent downward bias of empirical covariance estimators under infill asymptotics.
> - **§3.3 Sampling Distributions and Fourth-Order Moments:** Derives the variance and covariance of empirical semivariances via an elementary regression-decomposition proof of Isserlis' identity; explains why empirical variograms look smoother than they truly are.
> - **§3.4 Robust Estimation of Spatial Continuity:** Introduces fourth-root modulus transformations to compress $\chi_1^2$ increments; derives the $0.457$ and $0.494$ normal calibration constants; addresses dependent-pair overestimation.
> - **§3.5 Multivariate Spatial Continuity:** Extends continuity to multiple attributes via classical and pseudo-cross variograms for heterotopic sampling.
> - **§3.6 Asymmetric Delays and Advective Transport:** Explains why classical cross-variograms are blind to physical advection and demonstrates asymmetric directional cross-covariances.
> - **§3.7 Analytical Exercises:** Fully worked derivations covering increment covariances, 1D lattice variance inflation, heavy-tailed fractional moments, and pseudo-cross symmetry.

---

### Fundamental Assumption: Constant Mean vs. Spatial Trend

Throughout Sections 3.1–3.4, we operate under the foundational assumption that the spatial process has a **constant mean**:
$$
\operatorname{E}[Z(\mathbf{s})] = \mu \quad \forall \mathbf{s} \in D
$$
Under a constant mean, taking pairwise differences completely annihilates $\mu$:
$$
Z(\mathbf{s}_i) - Z(\mathbf{s}_j) = (\mu + \delta(\mathbf{s}_i)) - (\mu + \delta(\mathbf{s}_j)) = \delta(\mathbf{s}_i) - \delta(\mathbf{s}_j)
$$
allowing Matheron's estimator to estimate $\gamma(\mathbf{h})$ without knowing, specifying, or estimating $\mu$.

> [!WARNING]
> **The Effect of an Unmodeled Spatial Trend**
> If an unmodeled spatial trend $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ is present in the data, the expectation of the squared spatial increment expands as:
> $$
> \operatorname{E}\left[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2\right] = 2\gamma(\mathbf{s}_i - \mathbf{s}_j) + \left(\mu(\mathbf{s}_i) - \mu(\mathbf{s}_j)\right)^2
> $$
> Because the squared mean offset $(\mu(\mathbf{s}_i) - \mu(\mathbf{s}_j))^2 \ge 0$ is strictly non-negative and typically grows quadratically with spatial separation $\|\mathbf{s}_i - \mathbf{s}_j\|$, Matheron's estimator is biased **upward**. This trend-induced inflation causes empirical semivariograms to exhibit parabolic growth that never levels off, confounding spatial autocorrelation with large-scale deterministic drift. When spatial trends exist, one must first fit and subtract the mean structure $\mathbf{x}(\mathbf{s})'\widehat{\boldsymbol{\beta}}$ via generalized least squares and estimate the semivariogram on the residuals (a central workflow developed in Chapter 6).

---

### 3.1 Nonparametric Estimators and Spatial Lag Binning

#### 3.1.1 Why Differences? From Classical Sample Variance to Spatial Lag Bins

To understand why spatial continuity is estimated through pairwise differences rather than deviations from a global mean, we begin with a foundational identity from classical mathematical statistics.

##### Classical Sample Variance as Pairwise Differences
Given an independent sample of $n$ scalar observations $\mathbf{Y} = (Y_1, \dots, Y_n)'$, the sample variance is traditionally computed with respect to the sample mean $\bar{Y} = \frac{1}{n}\sum_{i=1}^n Y_i$:
$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2
$$
However, an elementary algebraic identity reveals that $S^2$ can be evaluated without computing or knowing the sample mean $\bar{Y}$ at all, simply by averaging all $n^2$ pairwise squared differences:
$$
S^2 = \frac{1}{2n(n-1)}\sum_{i=1}^n \sum_{j=1}^n (Y_i - Y_j)^2
$$
To verify this identity, expand the right-hand sum:
$$
\sum_{i=1}^n \sum_{j=1}^n (Y_i - Y_j)^2 = \sum_{i=1}^n \sum_{j=1}^n \left(Y_i^2 - 2Y_i Y_j + Y_j^2\right) = n \sum_{i=1}^n Y_i^2 - 2\left(\sum_{i=1}^n Y_i\right)^2 + n \sum_{j=1}^n Y_j^2 = 2n\sum_{i=1}^n Y_i^2 - 2n^2 \bar{Y}^2
$$
Dividing by $2n(n-1)$ yields:
$$
\frac{2n \left(\sum Y_i^2 - n\bar{Y}^2\right)}{2n(n-1)} = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2 = S^2
$$
In matrix notation, this sum of squared differences is the quadratic form:
$$
S^2 = \frac{1}{n-1}\mathbf{Y}'\left(\mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'\right)\mathbf{Y}
$$
where the centering projection matrix $\mathbf{P} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}'$ acts as an omnibus difference operator that annihilates constant shifts.

> [!NOTE]
> **From the Lecture Hall: Why Pairs, Not Points — Variance from Differences Without the Mean**
> In classical i.i.d. sampling, estimating the population mean $\mu$ with the sample mean $\bar{Y}$ costs exactly 1 degree of freedom (the divisor $n-1$). But in spatial statistics, observations are correlated. Estimating the mean $\mu$ under spatial correlation introduces severe bias into subsequent covariance estimates (as we prove in Section 3.2).
> 
> The pairwise difference identity provides a path forward: because $Y_i - Y_j = (Y_i - \mu) - (Y_j - \mu)$, taking differences eliminates the mean $\mu$ completely. The semivariogram is nothing more than the pairwise-difference formula for variance, restricted to pairs of points separated by specific spatial displacement vectors $\mathbf{h}$.

##### Generalization to Spatial Lag Bins
In spatial statistics, observations separated by displacement $\mathbf{h}$ share a different degree of correlation than observations separated by $5\mathbf{h}$. Rather than summing differences across all $n^2$ pairs indiscriminately, we partition point pairs into spatial separation sets.

Let $S = \{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset D$ denote the set of spatial sampling locations. For any displacement vector $\mathbf{h} \in \mathbb{R}^d$, the theoretical pair set is:
$$
N(\mathbf{h}) = \left\{(\mathbf{s}_i, \mathbf{s}_j) \in S \times S : \mathbf{s}_i - \mathbf{s}_j = \mathbf{h}\right\}
$$
with cardinality $|N(\mathbf{h})|$. On irregular spatial networks, the probability that any two continuous coordinate pairs satisfy $\mathbf{s}_i - \mathbf{s}_j = \mathbf{h}$ exactly is zero. Consequently, we generalize the exact displacement vector to a spatial *lag tolerance region* (or *lag bin*) $B(\mathbf{h}_k) \subset \mathbb{R}^d$ centered at a nominal lag vector $\mathbf{h}_k$:
$$
N(B(\mathbf{h}_k)) = \left\{(\mathbf{s}_i, \mathbf{s}_j) \in S \times S : \mathbf{s}_i - \mathbf{s}_j \in B(\mathbf{h}_k)\right\}
$$

For directional spatial continuity analysis, $B(\mathbf{h}_k)$ is defined as a cone-frustum in $\mathbb{R}^d$ parameterized by nominal length $h_k = \|\mathbf{h}_k\|$, distance half-tolerance $\Delta h_k / 2$, directional unit vector $\mathbf{u}_k = \mathbf{h}_k / \|\mathbf{h}_k\|$, and angular tolerance $\Delta \theta$:
$$
B(\mathbf{h}_k) = \left\{\mathbf{h} \in \mathbb{R}^d : h_k - \frac{\Delta h_k}{2} \le \|\mathbf{h}\| < h_k + \frac{\Delta h_k}{2}, \; \arccos\left(\frac{|\mathbf{h}'\mathbf{u}_k|}{\|\mathbf{h}\|}\right) \le \frac{\Delta \theta}{2}\right\}
$$
When the underlying spatial process is assumed to be isotropic, the dependence function depends only on the scalar Euclidean separation distance $h = \|\mathbf{h}\|$. The omnidirectional lag bin becomes a spherical shell:
$$
N(h_k) = \left\{(\mathbf{s}_i, \mathbf{s}_j) \in S \times S : h_k - \frac{\Delta h}{2} \le \|\mathbf{s}_i - \mathbf{s}_j\| < h_k + \frac{\Delta h}{2}\right\}
$$
where $\Delta h$ denotes the radial bin bandwidth.

##### Ordered vs. Unordered Pair Conventions
In theoretical derivations, $N(\mathbf{h}_k)$ is traditionally written as a set of ordered pairs $(\mathbf{s}_i, \mathbf{s}_j)$ with pair count $|N_{\text{ordered}}(\mathbf{h}_k)|$. Because squared increments are symmetric:
$$
(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 = (Z(\mathbf{s}_j) - Z(\mathbf{s}_i))^2
$$
computational implementations iterate over unique unordered pairs $i < j$. The relationship between the two conventions is exact:
$$
|N_{\text{ordered}}(\mathbf{h}_k)| = 2 |N_{\text{unordered}}(\mathbf{h}_k)|
$$
so the estimator can be evaluated equivalently as:
$$
\widehat{\gamma}(\mathbf{h}_k) = \frac{1}{2|N_{\text{ordered}}(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N_{\text{ordered}}(\mathbf{h}_k)} (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 = \frac{1}{|N_{\text{unordered}}(\mathbf{h}_k)|} \sum_{i < j : (\mathbf{s}_i, \mathbf{s}_j) \in N(h_k)} \frac{1}{2}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2
$$
Both formulations yield identical numerical values.

##### Graph Laplacian Structure of Genton's Spatial Design Matrix
Marc G. Genton (1998) showed that Matheron's empirical estimator can be expressed as a quadratic form in the spatial observation vector $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$:
$$
\widehat{\gamma}(\mathbf{h}) = \frac{1}{2|N(\mathbf{h})|} \mathbf{Z}' \mathbf{A}(\mathbf{h}) \mathbf{Z}
$$
where $\mathbf{A}(\mathbf{h}) \in \mathbb{R}^{n \times n}$ is Genton's spatial design matrix for lag $\mathbf{h}$.

Let $\mathbf{e}_i = (0, \dots, 1, \dots, 0)' \in \mathbb{R}^n$ denote the standard Euclidean basis vector with 1 in the $i$-th entry. Any pairwise difference is a linear contrast:
$$
Z(\mathbf{s}_i) - Z(\mathbf{s}_j) = (\mathbf{e}_i - \mathbf{e}_j)'\mathbf{Z}
$$
Squaring this scalar produces a rank-1 outer product:
$$
(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 = \mathbf{Z}' (\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)' \mathbf{Z}
$$
Summing across all pairs in lag bin $N(\mathbf{h})$ yields the explicit representation:
$$
\mathbf{A}(\mathbf{h}) = \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} (\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)'
$$
This matrix possesses three fundamental mathematical properties:
1. **Combinatorial Graph Laplacian:** $\mathbf{A}(\mathbf{h})$ is the exact graph Laplacian of the undirected network where an edge exists between sites $i$ and $j$ if and only if $(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})$. The diagonal entry $A_{ii}(\mathbf{h})$ equals the degree of site $\mathbf{s}_i$ (the number of pairs in bin $\mathbf{h}$ containing site $i$), and the off-diagonal entry $A_{ij}(\mathbf{h}) = -1$ if $(i, j) \in N(\mathbf{h})$ and $0$ otherwise.
2. **Positive Semi-Definiteness:** Because each constituent $(\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)'$ is a symmetric rank-1 positive semi-definite matrix, $\mathbf{A}(\mathbf{h})$ is a sum of positive semi-definite matrices. Hence $\mathbf{v}'\mathbf{A}(\mathbf{h})\mathbf{v} \ge 0$ for all $\mathbf{v} \in \mathbb{R}^n$, guaranteeing $\widehat{\gamma}(\mathbf{h}) \ge 0$.
3. **Mean Annihilation:** The vector of ones $\mathbf{1} = (1, \dots, 1)'$ lies in the null space of $\mathbf{A}(\mathbf{h})$:
   $$
   \mathbf{A}(\mathbf{h})\mathbf{1} = \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} (\mathbf{e}_i - \mathbf{e}_j)(1 - 1) = \mathbf{0}
   $$
   Every row and column sum of $\mathbf{A}(\mathbf{h})$ is identically zero. Consequently, for any spatial process with constant mean $\operatorname{E}[\mathbf{Z}] = \mu \mathbf{1}$, the quadratic form eliminates the unknown mean:
   $$
   \operatorname{E}[\mathbf{Z}'\mathbf{A}(\mathbf{h})\mathbf{Z}] = \operatorname{Tr}(\mathbf{A}(\mathbf{h})\mathbf{V}) + \mu^2 \mathbf{1}'\mathbf{A}(\mathbf{h})\mathbf{1} = \operatorname{Tr}(\mathbf{A}(\mathbf{h})\mathbf{V})
   $$

##### The Two Immediate Payoffs of Genton's Design Matrix
Introducing Genton's matrix $\mathbf{A}(\mathbf{h})$ yields two immediate analytical results:
1. **One-Line Unbiasedness Proof (Theorem 3.2.1):** Taking the expectation of $\widehat{\gamma}(\mathbf{h})$:
   $$
   \operatorname{E}[\widehat{\gamma}(\mathbf{h})] = \frac{1}{2|N(\mathbf{h})|}\operatorname{Tr}(\mathbf{A}(\mathbf{h})\mathbf{V}) = \frac{1}{2|N(\mathbf{h})|}\sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h})} \operatorname{Tr}\left((\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)'\mathbf{V}\right) = \frac{1}{2|N(\mathbf{h})|}\sum (V_{ii} + V_{jj} - 2V_{ij}) = \gamma(\mathbf{h})
   $$
2. **Compact Inter-Lag Covariance Formula (Section 3.3):** Under Gaussianity, for any two lag bins $\mathbf{h}_1$ and $\mathbf{h}_2$, the sampling covariance between $\widehat{\gamma}(\mathbf{h}_1)$ and $\widehat{\gamma}(\mathbf{h}_2)$ reduces from a tedious four-index sum to a compact matrix trace:
   $$
   \operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_1), \widehat{\gamma}(\mathbf{h}_2)) = \frac{\operatorname{Tr}\left(\mathbf{A}(\mathbf{h}_1)\mathbf{V}\mathbf{A}(\mathbf{h}_2)\mathbf{V}\right)}{2|N(\mathbf{h}_1)||N(\mathbf{h}_2)|}
   $$

##### Definition 3.1.1 (Matheron's Method-of-Moments Semivariogram Estimator)
Under intrinsic stationarity with constant mean $\operatorname{E}[Z(\mathbf{s})] = \mu$, the theoretical semivariogram is $\gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}))$. Georges Matheron (1962) proposed the classical *method-of-moments semivariogram estimator*:
$$
\widehat{\gamma}(\mathbf{h}_k) = \frac{1}{2|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2
$$
where the summation evaluates the squared spatial increment for all distinct ordered pairs $(\mathbf{s}_i, \mathbf{s}_j)$ belonging to the lag tolerance bin $N(\mathbf{h}_k)$.

###### Term-by-Term Equation Anatomy: Matheron's Method-of-Moments Estimator
$$
\widehat{\gamma}(\mathbf{h}_k) = \frac{1}{2|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} (Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2
$$
- $\widehat{\gamma}(\mathbf{h}_k)$: The empirical semivariance estimate at nominal lag bin $\mathbf{h}_k$.
- $\frac{1}{2}$: The semivariance halving factor, maintaining exact duality with theoretical covariance: $\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$.
- $|N(\mathbf{h}_k)|$: The pair count (cardinality) of sample pairs $(\mathbf{s}_i, \mathbf{s}_j)$ satisfying $\mathbf{s}_i - \mathbf{s}_j \in B(\mathbf{h}_k)$.
- $\sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)}$: The summation accumulating squared differences across all distinct pairs in the bin.
- $(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2$: The squared spatial increment. Under intrinsic stationarity with constant mean $\mu$, the mean cancels algebraically: $((\mu + \delta(\mathbf{s}_i)) - (\mu + \delta(\mathbf{s}_j)))^2 = (\delta(\mathbf{s}_i) - \delta(\mathbf{s}_j))^2$, making the estimator completely free of the unknown mean. (When pairs are separated by *exact* lag $\mathbf{h}_k$, the estimator is strictly unbiased; in irregular networks, finite bin bandwidth $\Delta h$ introduces minor binning bias).

##### Definition 3.1.2 (Empirical Spatial Covariance Estimators)
Under second-order stationarity with covariance kernel $C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h}))$:
1. When the spatial mean $\mu = \operatorname{E}[Z(\mathbf{s})]$ is known *a priori*, the centered covariance estimator is:
   $$
   \widehat{C}_0(\mathbf{h}_k) = \frac{1}{|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} (Z(\mathbf{s}_i) - \mu)(Z(\mathbf{s}_j) - \mu)
   $$
2. When the spatial mean $\mu$ is unknown, it is replaced by the spatial sample mean $\bar{Z} = \frac{1}{n}\sum_{i=1}^n Z(\mathbf{s}_i)$, yielding the empirical spatial covariance estimator:
   $$
   \widehat{C}(\mathbf{h}_k) = \frac{1}{|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} (Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z})
   $$

---

#### 3.1.2 How to Compute It: Directional and Omnidirectional Lag Binning

Computing $\widehat{\gamma}(\mathbf{h}_k)$ requires selecting bin geometries and iterating over unique sample pairs.

##### Practical Recipe for Lag Centers and Bandwidth Selection
A well-behaved empirical semivariogram balances resolution near the origin against sampling stability at large distances. Follow this standard recipe:
1. **Lag Bandwidth ($\Delta h$):** Set $\Delta h$ approximately equal to the median nearest-neighbor distance among the sample locations:
   $$
   \Delta h \approx \operatorname{median}_{i} \left( \min_{j \ne i} \|\mathbf{s}_i - \mathbf{s}_j\| \right)
   $$
   This ensures that the first bin captures immediate neighbor pairs without over-smoothing.
2. **Lag Centers ($h_k$):** Space nominal lag centers evenly as integer multiples: $h_k = k \Delta h$ for $k = 1, 2, \dots, K$.
3. **Maximum Lag Cutoff ($h_{\max}$):** Restrict estimation to half the maximum bounding-box diameter of the study region:
   $$
   h_{\max} \le \frac{1}{2} \operatorname{diam}(D) = \frac{1}{2} \max_{i, j} \|\mathbf{s}_i - \mathbf{s}_j\|
   $$
   Pairs separated by more than half the domain extent are sparse and forced toward opposite domain boundaries, inducing edge bias.
4. **Minimum Pair Count Rule:** Require at least $30$ to $50$ pairs per lag bin ($|N(\mathbf{h}_k)| \ge 30$). Bins with fewer pairs exhibit excessive sampling variability; adjacent sparse bins should be merged.
5. **Handling Co-Located Duplicates:** In mining, environmental, and field surveys, duplicate assays at the exact same location ($\|\mathbf{s}_i - \mathbf{s}_j\| \approx 0$) provide direct estimation of the nugget effect $\tau^2$ (pure laboratory or microscale measurement error). When co-located points exist, collect them into an explicit lag-0 nugget bin $\widehat{\tau}^2 = \frac{1}{2|N_0|}\sum (Z_i - Z_j)^2$ rather than discarding them.

```python
import math

def compute_empirical_semivariogram(coords, values, lag_centers, lag_bandwidth, 
                                    direction_vector=None, angle_tolerance_rad=None,
                                    extract_nugget_pairs=True):
    """
    Computes Matheron's empirical semivariogram matching Definition 3.1.1
    in pure standard Python.
    
    Parameters:
        coords: list of lists, shape (n, d), spatial coordinates s_i
        values: list of floats, shape (n,), observations Z(s_i)
        lag_centers: list of floats, shape (K,), nominal lag bin centers h_k
        lag_bandwidth: float, radial width of lag bins Delta_h
        direction_vector: optional list of floats, shape (d,), directional unit vector u
        angle_tolerance_rad: optional float, full angular tolerance Delta_theta in radians
        extract_nugget_pairs: bool, if True, tracks co-located duplicate pairs at distance < 1e-12
        
    Returns:
        actual_lags: list of floats of length K, actual mean distance per bin
        gamma_hat: list of floats of length K, estimated semivariances
        pair_counts: list of ints of length K, number of pairs per bin |N(h_k)|
        nugget_estimate: float or None, variance from co-located duplicate pairs
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    sum_sq_diff = [0.0] * K
    sum_distance = [0.0] * K
    pair_counts = [0] * K
    
    nugget_sq_diff = 0.0
    nugget_count = 0
    
    # Normalize direction vector if provided
    is_directional = direction_vector is not None and angle_tolerance_rad is not None
    if is_directional:
        u_norm = math.sqrt(sum(x * x for x in direction_vector))
        u = [x / u_norm for x in direction_vector]
        half_angle = angle_tolerance_rad / 2.0  # Half-tolerance for cone filtering
        
    # Iterate over unique unordered pairs i < j
    for i in range(n - 1):
        for j in range(i + 1, n):
            h_vec = [coords[i][m] - coords[j][m] for m in range(d)]
            dist = math.sqrt(sum(x * x for x in h_vec))
            
            # Co-located duplicate check for nugget diagnosis
            if dist < 1e-12:
                if extract_nugget_pairs:
                    diff = values[i] - values[j]
                    nugget_sq_diff += diff * diff
                    nugget_count += 1
                continue
                
            # Directional angular filtering
            if is_directional:
                dot_prod = sum(h_vec[m] * u[m] for m in range(d))
                cos_theta = abs(dot_prod) / dist
                cos_theta = max(-1.0, min(1.0, cos_theta))
                theta = math.acos(cos_theta)
                if theta > half_angle:
                    continue
                    
            # Check membership across lag bins
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    diff = values[i] - values[j]
                    sum_sq_diff[k] += diff * diff
                    sum_distance[k] += dist
                    pair_counts[k] += 1
                    break
                    
    # Form Matheron estimator
    actual_lags = [None] * K
    gamma_hat = [None] * K
    for k in range(K):
        if pair_counts[k] > 0:
            actual_lags[k] = sum_distance[k] / pair_counts[k]
            gamma_hat[k] = sum_sq_diff[k] / (2.0 * pair_counts[k])
            
    nugget_estimate = (nugget_sq_diff / (2.0 * nugget_count)) if nugget_count > 0 else None
    return actual_lags, gamma_hat, pair_counts, nugget_estimate
```

---

#### 3.1.3 What Can Go Wrong: Cloud Noise, Binning Distortion, and Pair Dependence

##### The Variogram Cloud vs. The Binned Semivariogram
Before grouping pairs into discrete bins, every pair of locations $(\mathbf{s}_i, \mathbf{s}_j)$ contributes a single point $(h_{ij}, \frac{1}{2}D_{ij}^2)$ to the **variogram cloud**:
$$
\text{Cloud Point: } \left( \|\mathbf{s}_i - \mathbf{s}_j\|, \; \frac{1}{2}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2 \right)
$$
Because the individual squared increments $\frac{1}{2}D_{ij}^2 \sim \gamma(h) \chi_1^2$ follow a heavily skewed distribution, the cloud exhibits massive vertical dispersion at every lag. Binning computes the conditional arithmetic mean across each distance interval, condensing the diffuse cloud into a stable empirical curve $\widehat{\gamma}(h_k)$:

| Lag Bin $k$ | Nominal Lag $h_k$ | Actual Mean Lag $\bar{h}_k$ | Pair Count $|N(h_k)|$ | Cloud Spread (IQR) | Binned Estimate $\widehat{\gamma}(h_k)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.0 m | 0.98 m | 42 | $[0.08, 0.94]$ | **0.38** |
| 2 | 2.0 m | 2.01 m | 78 | $[0.19, 1.62]$ | **0.65** |
| 3 | 3.0 m | 2.97 m | 110 | $[0.31, 2.15]$ | **0.82** |
| 4 | 4.0 m | 4.04 m | 95 | $[0.42, 2.48]$ | **0.94** |
| 5 | 5.0 m | 5.02 m | 64 | $[0.48, 2.65]$ | **0.99** |

##### Why Variance is Not Inversely Proportional to Pair Count
A frequent misconception is to assume that the sampling variance of $\widehat{\gamma}(h_k)$ scales as $\operatorname{Var}(\widehat{\gamma}(h_k)) \propto 1 / |N(h_k)|$, as if the pairs were independent random variables.
- **Why this fails:** Distinct pairs $(\mathbf{s}_i, \mathbf{s}_j)$ and $(\mathbf{s}_i, \mathbf{s}_k)$ share the common observation $Z(\mathbf{s}_i)$. Even pairs with completely disjoint vertices are spatially correlated because $Z(\mathbf{s})$ is a continuous field.
- **The reality:** Because of inter-pair correlation, the effective number of independent pairs $N_{\text{eff}}$ is vastly smaller than $|N(h_k)|$. True sampling variance is several times larger than the naive i.i.d. formula $2\gamma^2 / |N|$ (as derived rigorously in Section 3.3).

> [!NOTE]
> **Wait—Why Did We Do That?**
> **FAQ 1: Why does binning lags with an angular and distance tolerance band $(\Delta h, \Delta \theta)$ introduce estimation bias?**
>
> In theoretical definitions, the semivariogram $\gamma(\mathbf{h})$ is defined at an infinitesimal lag displacement $\mathbf{h}$. But in irregularly spaced field data, the probability that any two observation points have the exact separation $\mathbf{s}_i - \mathbf{s}_j = \mathbf{h}$ is zero. We must group pairs into tolerance regions $B(\mathbf{h}) = [h - \Delta h/2, h + \Delta h/2) \times [\theta - \Delta \theta/2, \theta + \Delta \theta/2)$.
>
> Why does this practical necessity introduce systematic estimation bias?
> 1. **Curvature and Jensen's Discrepancy:** The true semivariogram $\gamma(h)$ is non-linear, with changing curvature $\gamma''(h)$. Averaging pairwise squared differences over a bin computes:
>    $$
>    \operatorname{E}[\widehat{\gamma}(\mathbf{h}_k)] = \frac{1}{|N(\mathbf{h}_k)|}\sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} \gamma(\|\mathbf{s}_i - \mathbf{s}_j\|) \ne \gamma(h_k)
>    $$
> 2. **Smoothing and Attenuation:** In concave regions where $\gamma''(h) < 0$ (the shoulder approaching the sill), chord averaging across a finite width $\Delta h$ systematically pulls the estimate below the curve: $\operatorname{E}[\widehat{\gamma}(h_k)] < \gamma(h_k)$. Near the origin ($h \to 0$), binning smears pairs separated by distance $\Delta h$ into the first bin, artificially elevating the estimated nugget effect $\tau^2$.
> 3. **Directional Smearing:** An angular tolerance $\Delta \theta$ mixes pairs from different orientations. In strongly anisotropic systems, a wide angular window averages the steep minor axis with the gentle major axis, blunting the estimated anisotropy ratio.
>
> To mitigate binning bias, analysts should plot empirical semivariances against the *actual mean separation distance* $\bar{h}_k = \frac{1}{|N(\mathbf{h}_k)|}\sum \|\mathbf{s}_i - \mathbf{s}_j\|$ of pairs in that bin, rather than the nominal midpoint $h_k$.

---

### 3.2 Expectations, Unbiasedness, and Systematic Downward Bias

#### 3.2.1 Why Covariances Sag: Mean Estimation and Variance Inflation

##### From Classical Sample Mean Correction to Spatial Variance Inflation
In classical introductory mathematical statistics, when estimating the population variance $\sigma^2 = \operatorname{Var}(Y)$ from an independent and identically distributed (i.i.d.) random sample $Y_1, \dots, Y_n$ with unknown mean $\mu$, replacing $\mu$ with the sample mean $\bar{Y} = \frac{1}{n}\sum_{i=1}^n Y_i$ introduces a downward bias in the unadjusted sum of squared deviations:
$$
\operatorname{E}\left[\frac{1}{n}\sum_{i=1}^n (Y_i - \bar{Y})^2\right] = \sigma^2 - \operatorname{Var}(\bar{Y}) = \sigma^2 - \frac{\sigma^2}{n} = \frac{n-1}{n}\sigma^2
$$
Multiplying by Bessel's correction factor $\frac{n}{n-1}$ completely eliminates this bias, yielding the familiar unbiased sample variance $S^2 = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2$. Crucially, as sample size increases under classical sampling, $\operatorname{Var}(\bar{Y}) = \frac{\sigma^2}{n} \to 0$, so the bias vanishes asymptotically at rate $\mathcal{O}(n^{-1})$.

In spatial statistics, however, two structural complications destroy this simple remedy:
1. **Variance Inflation via Autocorrelation:** The variance of the spatial sample mean $\bar{Z} = \frac{1}{n}\mathbf{1}'\mathbf{Z}$ is not $\frac{\sigma^2}{n}$, but rather the quadratic form:
   $$
   \operatorname{Var}(\bar{Z}) = \frac{1}{n^2} \mathbf{1}' \mathbf{V} \mathbf{1} = \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n C(\mathbf{s}_i - \mathbf{s}_j)
   $$
   Under positive spatial autocorrelation ($C(\mathbf{s}_i - \mathbf{s}_j) \ge 0$), every off-diagonal covariance entry is positive, heavily inflating $\operatorname{Var}(\bar{Z})$ well beyond $\sigma^2 / n$.
2. **Infill Bias Permanence:** As established in Chapter 1, under infill asymptotics (sampling more densely within a bounded geographic domain $D$), the variance of the sample mean does *not* approach zero:
   $$
   \lim_{n \to \infty} \operatorname{Var}(\bar{Z}) = \frac{1}{|D|^2} \int_D \int_D C(\mathbf{u} - \mathbf{v}) d\mathbf{u} d\mathbf{v} > 0
   $$
   Because $\operatorname{Var}(\bar{Z})$ remains bounded away from zero, the empirical covariance estimator $\widehat{C}(\mathbf{h})$ suffers from a permanent downward bias equal to $\operatorname{Var}(\bar{Z})$ that cannot be cured by increasing sample size $n$. In sharp contrast, Matheron's empirical semivariogram $\widehat{\gamma}(\mathbf{h})$ takes pairwise differences $Z(\mathbf{s}_i) - Z(\mathbf{s}_j) = \delta(\mathbf{s}_i) - \delta(\mathbf{s}_j)$, completely bypassing the unknown mean $\mu$ and its estimator $\bar{Z}$, ensuring exact unbiasedness regardless of spatial autocorrelation.

A decisive distinction between semivariogram estimation and covariance estimation lies in their mathematical expectations when the true spatial mean $\mu$ is unknown and replaced by the spatial sample mean $\bar{Z}$.

##### Theorem 3.2.1 (Exact Unbiasedness of Matheron's Semivariogram Estimator)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an intrinsically stationary random field with semivariogram $\gamma(\mathbf{h})$ and constant mean $\operatorname{E}[Z(\mathbf{s})] = \mu$. If $|N(\mathbf{h}_k)| \ge 1$ and every pair $(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)$ satisfies $\mathbf{s}_i - \mathbf{s}_j = \mathbf{h}_k$, then Matheron's estimator $\widehat{\gamma}(\mathbf{h}_k)$ is strictly unbiased:
$$
\operatorname{E}[\widehat{\gamma}(\mathbf{h}_k)] = \gamma(\mathbf{h}_k)
$$
regardless of whether the mean $\mu$ is known, unknown, or estimated.

*Proof.*
We can verify this result either via pairwise expectations or via Genton's design matrix:
1. **Via Pairwise Expectations:** By intrinsic stationarity, $\operatorname{E}[Z(\mathbf{s}_i) - Z(\mathbf{s}_j)] = \mu - \mu = 0$, so $\operatorname{Var}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j)) = \operatorname{E}[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2] = 2\gamma(\mathbf{h}_k)$.
   Taking the mathematical expectation:
   $$
   \operatorname{E}[\widehat{\gamma}(\mathbf{h}_k)] = \frac{1}{2|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} \operatorname{E}[(Z(\mathbf{s}_i) - Z(\mathbf{s}_j))^2] = \frac{1}{2|N(\mathbf{h}_k)|} \cdot 2|N(\mathbf{h}_k)|\gamma(\mathbf{h}_k) = \gamma(\mathbf{h}_k)
   $$
2. **Via Genton's Design Matrix:** Recall from Section 3.1 that $\widehat{\gamma}(\mathbf{h}_k) = \frac{1}{2|N(\mathbf{h}_k)|} \mathbf{Z}'\mathbf{A}(\mathbf{h}_k)\mathbf{Z}$. Because $\mathbf{A}(\mathbf{h}_k)\mathbf{1} = \mathbf{0}$:
   $$
   \operatorname{E}[\mathbf{Z}'\mathbf{A}(\mathbf{h}_k)\mathbf{Z}] = \operatorname{Tr}(\mathbf{A}(\mathbf{h}_k)\mathbf{V}) + \mu^2 \mathbf{1}'\mathbf{A}(\mathbf{h}_k)\mathbf{1} = \operatorname{Tr}(\mathbf{A}(\mathbf{h}_k)\mathbf{V})
   $$
   Expanding the trace:
   $$
   \operatorname{Tr}(\mathbf{A}(\mathbf{h}_k)\mathbf{V}) = \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} \operatorname{Tr}\left((\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)'\mathbf{V}\right) = \sum (V_{ii} + V_{jj} - 2V_{ij}) = \sum 2\gamma(\mathbf{h}_k) = 2|N(\mathbf{h}_k)|\gamma(\mathbf{h}_k)
   $$
   Dividing by $2|N(\mathbf{h}_k)|$ yields $\gamma(\mathbf{h}_k)$. The derivation does not require knowing, centering, or estimating $\mu$. $\blacksquare$

##### Theorem 3.2.2 (Systematic Downward Bias of Empirical Covariance Estimators)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a second-order stationary spatial random field with covariance function $C(\mathbf{h})$, constant mean $\mu$, and variance $\sigma^2 = C(\mathbf{0})$. Let $\bar{Z} = \frac{1}{n}\sum_{k=1}^n Z(\mathbf{s}_k)$ be the spatial sample mean. The empirical covariance estimator:
$$
\widehat{C}(\mathbf{h}_k) = \frac{1}{|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} (Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z})
$$
satisfies the exact expectation:
$$
\operatorname{E}[\widehat{C}(\mathbf{h}_k)] = C(\mathbf{h}_k) - 2\overline{C}(N(\mathbf{h}_k), S) + \operatorname{Var}(\bar{Z})
$$
where $\operatorname{Var}(\bar{Z}) = \frac{1}{n^2} \mathbf{1}' \mathbf{V} \mathbf{1} = \overline{C}(S, S) > 0$ and:
$$
\overline{C}(N(\mathbf{h}_k), S) = \frac{1}{2|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} \frac{1}{n}\sum_{r=1}^n [C(\mathbf{s}_i - \mathbf{s}_r) + C(\mathbf{s}_j - \mathbf{s}_r)]
$$
denotes the average covariance between the sampling pairs in bin $N(\mathbf{h}_k)$ and all sample locations $S$.

*Proof.*
Decompose each observation into its population mean and stochastic residual:
$$
Z(\mathbf{s}_i) = \mu + \delta(\mathbf{s}_i), \quad \operatorname{E}[\delta(\mathbf{s}_i)] = 0, \quad \operatorname{Cov}(\delta(\mathbf{s}_i), \delta(\mathbf{s}_j)) = C(\mathbf{s}_i - \mathbf{s}_j)
$$
The spatial sample mean is $\bar{Z} = \mu + \bar{\delta}$ with $\bar{\delta} = \frac{1}{n}\sum_{r=1}^n \delta(\mathbf{s}_r)$.
The centered deviations satisfy $Z(\mathbf{s}_i) - \bar{Z} = \delta(\mathbf{s}_i) - \bar{\delta}$.
Expanding the product of centered deviations for pair $(\mathbf{s}_i, \mathbf{s}_j)$:
$$
(Z(\mathbf{s}_i) - \bar{Z})(Z(\mathbf{s}_j) - \bar{Z}) = (\delta(\mathbf{s}_i) - \bar{\delta})(\delta(\mathbf{s}_j) - \bar{\delta}) = \delta(\mathbf{s}_i)\delta(\mathbf{s}_j) - \bar{\delta}\delta(\mathbf{s}_i) - \bar{\delta}\delta(\mathbf{s}_j) + \bar{\delta}^2
$$
Taking expectations term-by-term:
1. $\operatorname{E}[\delta(\mathbf{s}_i)\delta(\mathbf{s}_j)] = C(\mathbf{s}_i - \mathbf{s}_j) = C(\mathbf{h}_k)$.
2. $\operatorname{E}[\bar{\delta}^2] = \operatorname{Var}(\bar{Z}) = \frac{1}{n^2}\sum_{r=1}^n \sum_{m=1}^n C(\mathbf{s}_r - \mathbf{s}_m)$.
3. $\operatorname{E}[\bar{\delta}\delta(\mathbf{s}_i)] = \frac{1}{n}\sum_{r=1}^n C(\mathbf{s}_i - \mathbf{s}_r)$ and $\operatorname{E}[\bar{\delta}\delta(\mathbf{s}_j)] = \frac{1}{n}\sum_{r=1}^n C(\mathbf{s}_j - \mathbf{s}_r)$.

Summing across all $|N(\mathbf{h}_k)|$ pairs and dividing by $|N(\mathbf{h}_k)|$ yields:
$$
\operatorname{E}[\widehat{C}(\mathbf{h}_k)] = C(\mathbf{h}_k) - 2\overline{C}(N(\mathbf{h}_k), S) + \overline{C}(S, S)
$$
In the interior of the spatial domain (for lags up to half the domain extent), points in $N(\mathbf{h}_k)$ are representative of the domain as a whole, so $\overline{C}(N(\mathbf{h}_k), S) \approx \overline{C}(S, S) = \operatorname{Var}(\bar{Z})$.
This yields the leading-order approximation:
$$
\operatorname{E}[\widehat{C}(\mathbf{h}_k)] \approx C(\mathbf{h}_k) - \operatorname{Var}(\bar{Z}) < C(\mathbf{h}_k)
$$
At moderate to large lags where the true covariance decays ($C(\mathbf{h}_k) \approx 0$), this uniform downward shift causes the empirical covariance to sag below zero:
$$
\operatorname{E}[\widehat{C}(\mathbf{h}_k)] \approx -\operatorname{Var}(\bar{Z}) < 0
$$
producing false negative correlations. At extreme separations near the boundary of the domain, pairs are forced to the edges of $D$, causing $\overline{C}(N(\mathbf{h}_k), S)$ to decrease, so the downward bias attenuates back toward zero. $\blacksquare$

> [!NOTE]
> **The Semivariogram's Own Price**
> The semivariogram's complete immunity to mean estimation is a major advantage, but it is not without its own trade-offs:
> 1. **Trend Sensitivity:** As shown in the introduction, if an unmodeled spatial drift $\mu(\mathbf{s})$ exists, the semivariogram experiences an upward bias of $(\mu(\mathbf{s}_i) - \mu(\mathbf{s}_j))^2$ that grows quadratically with lag.
> 2. **Inter-Lag Correlation:** Adjacent semivariogram estimates $\widehat{\gamma}(\mathbf{h}_1)$ and $\widehat{\gamma}(\mathbf{h}_2)$ are strongly correlated because they share sample locations (developed in Section 3.3).

---

#### 3.2.2 How to Compute It: Quantifying Spatial Variance Inflation

##### Numerical Verification of Covariance Sag
To see the exact mechanism of covariance sag, consider a 1D spatial transect with $n = 60$ regularly spaced sample sites $s_i = i$ ($i = 1, \dots, 60$) under an Exponential covariance model $C(h) = \sigma^2 \exp(-h / 10)$ with unit variance $\sigma^2 = 1.0$.

Evaluating the exact theoretical terms from Theorem 3.2.2:
- The variance of the spatial sample mean is $\operatorname{Var}(\bar{Z}) = \frac{1}{60^2} \mathbf{1}'\mathbf{V}\mathbf{1} = 0.2783$.
- Under i.i.d. sampling, the variance of the mean would be $\sigma^2 / 60 = 0.0167$. Spatial autocorrelation inflates the variance of $\bar{Z}$ by a factor of $\mathrm{VIF} = 0.2783 / 0.0167 \approx 16.7$!

The table below contrasts the true covariance $C(h)$ against the expected sample covariance $\operatorname{E}[\widehat{C}(h)]$ across various lags:

| Separation Lag $h$ | True Covariance $C(h)$ | Pair Cross-Covariance $\overline{C}(N(h), S)$ | Expected Sample Covariance $\operatorname{E}[\widehat{C}(h)]$ | Exact Downward Bias $\operatorname{E}[\widehat{C}] - C(h)$ | Leading-Order Bias $-\operatorname{Var}(\bar{Z})$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.9048 | 0.2798 | 0.6235 | **-0.2813** | -0.2783 |
| 5 | 0.6065 | 0.2863 | 0.3122 | **-0.2943** | -0.2783 |
| 10 | 0.3679 | 0.2905 | 0.0652 | **-0.3027** | -0.2783 |
| 20 | 0.1353 | 0.2908 | -0.1680 | **-0.3033** | -0.2783 |
| 30 | 0.0498 | 0.2791 | -0.2286 | **-0.2784** | -0.2783 |
| 45 | 0.0111 | 0.2443 | -0.1992 | **-0.2103** | -0.2783 |
| 59 | 0.0027 | 0.1755 | -0.0700 | **-0.0727** | -0.2783 |

**Key Takeaways from the Numerical Table:**
1. **Nearly Constant Interior Shift:** For lags $h \le 30$ (half the domain length), the downward bias remains locked around $-0.28$ to $-0.30$, tightly tracking the theoretical leading-order shift $-\operatorname{Var}(\bar{Z}) = -0.2783$.
2. **False Negative Covariance:** At lag $h = 20$, the true covariance is strictly positive ($C(20) = 0.1353$). Yet, the empirical estimator produces an expected value of $-0.1680$. The sample covariance sags below zero entirely because of sample mean subtraction!
3. **Boundary Attenuation:** At the maximum possible lag $h = 59$ (containing only 1 pair: $(s_1, s_{60})$), edge effects cause $\overline{C}(N, S)$ to decrease to $0.1755$, so the bias shrinks to $-0.0727$.

```python
import math

def compute_covariance_downward_bias(coords, lag_centers, lag_bandwidth, cov_func):
    """
    Evaluates the exact analytical downward bias of the empirical covariance
    estimator across nominal lags according to Theorem 3.2.2 in pure Python.
    
    Parameters:
        coords: list of lists, shape (n, d), sample sites s_i
        lag_centers: list of floats, shape (K,), nominal lags h_k
        lag_bandwidth: float, radial width Delta_h
        cov_func: callable, C(h) returning covariance for scalar distance h
        
    Returns:
        var_zbar: float, Var(Z_bar) = (1/n^2) 1' V 1
        expected_Chat: list of floats of length K, E[C_hat(h_k)]
        true_C: list of floats of length K, C(h_k)
        bias: list of floats of length K, expected_Chat - true_C
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    # 1. Compute full n x n covariance matrix V and distance matrix
    V = [[0.0] * n for _ in range(n)]
    dist_mat = [[0.0] * n for _ in range(n)]
    total_V_sum = 0.0
    
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            dist_mat[i][j] = dist
            c_val = cov_func(dist)
            V[i][j] = c_val
            total_V_sum += c_val
            
    # 2. Var(Z_bar) = (1/n^2) sum_{i,j} V_{ij}
    var_zbar = total_V_sum / (n * n)
    
    # Precompute row sums: (1/n) sum_{r=1}^n C(s_i - s_r)
    row_means_V = [sum(V[i]) / n for i in range(n)]
    
    expected_Chat = [None] * K
    true_C = [cov_func(h_k) for h_k in lag_centers]
    bias = [None] * K
    
    for k in range(K):
        h_k = lag_centers[k]
        pairs_in_bin = []
        for i in range(n):
            for j in range(n):
                if i != j and (h_k - half_bw) <= dist_mat[i][j] < (h_k + half_bw):
                    pairs_in_bin.append((i, j))
                    
        N_k = len(pairs_in_bin)
        if N_k == 0:
            continue
            
        c_bar_pair_sample = sum(row_means_V[i] for i, j in pairs_in_bin) / N_k
        exp_c = true_C[k] - 2.0 * c_bar_pair_sample + var_zbar
        expected_Chat[k] = exp_c
        bias[k] = exp_c - true_C[k]
        
    return var_zbar, expected_Chat, true_C, bias
```

---

#### 3.2.3 What Can Go Wrong: The Infill Trap, "Variance Theft", and Edge Effects

Why does the empirical covariance sag downward, and why is Matheron's semivariogram immune?

```
True vs. Empirical Covariance Sag:
Covariance C(h)
 ^
 |    True Covariance C(h)
 |    ───────────╮
 |                ╰───────────╮
 |    Empirical E[C_hat(h)]    ╰────────────────────> Lag h
 |    - - - - - - ╮
 |                 ╰ - - - - - ╮
 0 ─────────────────────────────╰───────────────────>
 |                                 Sag below zero:
 |                                 E[C_hat] ≈ - Var(Z_bar)
 v
```

1. **The Physical Mechanism of "Variance Theft":**
   When estimating the global mean $\mu$ by $\bar{Z}$, the sample mean naturally drifts toward the dominant regional cluster. In a positively correlated random field, adjacent points rise and fall together. By subtracting $\bar{Z}$ from each observation, the centered deviations $(Z(\mathbf{s}_i) - \bar{Z})$ are mechanically compressed toward zero. This artificial compression removes a chunk of true spatial variance equal to $\operatorname{Var}(\bar{Z})$.
2. **The Infill Asymptotic Trap:**
   Under infill asymptotics (cross-referencing Theorem 1.4.1 in Chapter 1), sampling 10,000 points within a bounded $10 \times 10\text{ km}$ basin does not increase our knowledge of the regional baseline drift $\mu$. All 10,000 points share the same overarching basin-wide fluctuation. Thus, $\operatorname{Var}(\bar{Z}) > 0$ remains permanently bounded away from zero, and $\widehat{C}(\mathbf{h})$ remains permanently biased downward regardless of sample size.
3. **The Immunity of the Semivariogram:**
   The semivariogram evaluates the difference between pairs:
   $$
   Z(\mathbf{s}_i) - Z(\mathbf{s}_j) = (\mu + \delta(\mathbf{s}_i)) - (\mu + \delta(\mathbf{s}_j)) = \delta(\mathbf{s}_i) - \delta(\mathbf{s}_j)
   $$
   The constant mean $\mu$ cancels out algebraically before any estimation takes place. Matheron's semivariogram estimator filters out the unknown spatial mean without losing degrees of freedom or incurring variance theft.

---

### 3.3 Sampling Distributions and Fourth-Order Moments under Gaussianity

#### 3.3.1 Why Adjacent Lags Correlate: Fourth-Order Moments and Isserlis' Theorem

In introductory linear models and mathematical statistics, evaluating the precision of variance estimators requires stepping from second-order moments to fourth-order moments. For an i.i.d. normal sample $Y_1, \dots, Y_n \sim \mathcal{N}(\mu, \sigma^2)$, the variance of the sample variance $S^2$ is derived from the properties of the chi-squared distribution:
$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi_{n-1}^2 \implies \operatorname{Var}(S^2) = \frac{\sigma^4}{(n-1)^2} \operatorname{Var}(\chi_{n-1}^2) = \frac{2\sigma^4}{n-1}
$$
Equivalently, this result follows from the fourth central moment of a standard normal variable, $\operatorname{E}[(Y_i - \mu)^4] = 3\sigma^4$, which yields $\operatorname{Var}((Y_i - \mu)^2) = \operatorname{E}[(Y_i - \mu)^4] - (\sigma^2)^2 = 2\sigma^4$.

When generalizing from i.i.d. scalar deviations to spatial increment pairs $D_{ij} = Z(\mathbf{s}_i) - Z(\mathbf{s}_j)$ and $D_{ab} = Z(\mathbf{s}_a) - Z(\mathbf{s}_b)$, two structural complications arise:
1. **Correlated Increments:** Pairs of spatial differences are correlated whenever they are spatially proximate or share common observation vertices ($j = a$).
2. **Inter-Lag Correlation:** In classical time series, sample autocovariances computed from distant, non-overlapping blocks are asymptotically independent. In contrast, spatial empirical semivariogram estimates $\widehat{\gamma}(\mathbf{h}_1)$ and $\widehat{\gamma}(\mathbf{h}_2)$ at distinct nominal lags are *never* independent because the underlying point pairs intertwine across the spatial domain.

To calculate the exact covariance between any two squared spatial increments $\operatorname{Cov}(D_{ij}^2, D_{ab}^2)$—and ultimately the covariance matrix of the empirical variogram vector across multiple lag bins—we rely on Isserlis' Theorem (Wick's probability identity for Gaussian fourth moments).

> [!NOTE]
> **From the Lecture Hall: Isserlis in One Idea — Regression Decomposition of Jointly Gaussian Variables**
>
> Many advanced textbooks derive Isserlis' identity by taking fourth partial derivatives of the multivariate normal characteristic function $\phi(\mathbf{t}) = \exp(-\frac{1}{2}\mathbf{t}'\mathbf{\Sigma}\mathbf{t})$. While mathematically rigorous, this Fourier approach obscures the simple geometry.
>
> In master's-level linear models, the exact same result follows in three lines from elementary conditional expectation.
>
> Suppose $U$ and $V$ are zero-mean jointly normal random variables with variances $\sigma_U^2, \sigma_V^2$ and covariance $\sigma_{UV} = \operatorname{Cov}(U, V)$. Consider regressing $V$ on $U$:
> $$
> V = \beta U + W, \quad \text{where } \beta = \frac{\sigma_{UV}}{\sigma_U^2}
> $$
> Because $(U, V)$ are jointly Gaussian, the regression residual $W \sim \mathcal{N}(0, \sigma_W^2)$ is **independent** of $U$, with residual variance $\sigma_W^2 = \sigma_V^2 - \beta^2 \sigma_U^2$.
>
> Squaring $V$ gives $V^2 = \beta^2 U^2 + 2\beta U W + W^2$. Multiplying by $U^2$ and taking expectations:
> $$
> \operatorname{E}[U^2 V^2] = \beta^2 \operatorname{E}[U^4] + 2\beta \operatorname{E}[U^3 W] + \operatorname{E}[U^2 W^2]
> $$
> Because $W \perp U$ and $\operatorname{E}[W] = 0$:
> - $\operatorname{E}[U^3 W] = \operatorname{E}[U^3]\operatorname{E}[W] = 0$
> - $\operatorname{E}[U^2 W^2] = \operatorname{E}[U^2]\operatorname{E}[W^2] = \sigma_U^2 \sigma_W^2$
>
> Using only the standard scalar normal fourth moment $\operatorname{E}[U^4] = 3\sigma_U^4$:
> $$
> \operatorname{E}[U^2 V^2] = 3\beta^2 \sigma_U^4 + \sigma_U^2 \sigma_W^2 = \sigma_U^2(\beta^2 \sigma_U^2 + \sigma_W^2) + 2\beta^2 \sigma_U^4
> $$
> Recognizing that $\beta^2 \sigma_U^2 + \sigma_W^2 = \sigma_V^2$ and $\beta \sigma_U^2 = \sigma_{UV} = \operatorname{Cov}(U, V)$:
> $$
> \operatorname{E}[U^2 V^2] = \sigma_U^2 \sigma_V^2 + 2(\operatorname{Cov}(U, V))^2 = \operatorname{E}[U^2]\operatorname{E}[V^2] + 2(\operatorname{Cov}(U, V))^2
> $$
> Subtracting $\operatorname{E}[U^2]\operatorname{E}[V^2]$ yields the celebrated Isserlis identity:
> $$
> \operatorname{Cov}(U^2, V^2) = 2\left(\operatorname{Cov}(U, V)\right)^2
> $$
> The variance of a squared Gaussian variable is simply twice its squared variance: $\operatorname{Var}(U^2) = 2(\operatorname{Var}(U))^2$.

##### Lemma 3.3.1 (Isserlis' Theorem / Wick's Identity for Gaussian Increments)
Let $(U, V)'$ be a bivariate zero-mean Gaussian vector with covariance $\operatorname{Cov}(U, V)$. Then:
$$
\operatorname{Cov}(U^2, V^2) = \operatorname{E}[U^2 V^2] - \operatorname{E}[U^2]\operatorname{E}[V^2] = 2\left(\operatorname{Cov}(U, V)\right)^2
$$
In particular, setting $U = V$ yields $\operatorname{Var}(U^2) = 2(\operatorname{Var}(U))^2$.

*Proof.*
Decompose $V = \beta U + W$ where $\beta = \operatorname{Cov}(U, V)/\sigma_U^2$ and $W \sim \mathcal{N}(0, \sigma_W^2)$ is statistically independent of $U$. Expanding $\operatorname{E}[U^2(\beta U + W)^2]$ and using $\operatorname{E}[U^4] = 3\sigma_U^4$ yields $\operatorname{E}[U^2 V^2] = \sigma_U^2 \sigma_V^2 + 2(\operatorname{Cov}(U, V))^2$, as established above. $\blacksquare$

##### Proposition 3.3.1 (Covariance of Spatial Squared Increments)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a Gaussian random field with semivariogram $\gamma(\mathbf{h})$ and covariance kernel $C(\mathbf{h})$. For any four sampling locations $\mathbf{s}_i, \mathbf{s}_j, \mathbf{s}_a, \mathbf{s}_b \in D$, define the linear spatial increments:
$$
D_{ij} = Z(\mathbf{s}_i) - Z(\mathbf{s}_j), \quad D_{ab} = Z(\mathbf{s}_a) - Z(\mathbf{s}_b)
$$
The covariance of their squared differences is:
$$
\operatorname{Cov}(D_{ij}^2, D_{ab}^2) = 2\left[\gamma(\mathbf{s}_i - \mathbf{s}_b) + \gamma(\mathbf{s}_j - \mathbf{s}_a) - \gamma(\mathbf{s}_i - \mathbf{s}_a) - \gamma(\mathbf{s}_j - \mathbf{s}_b)\right]^2
$$
Equivalently, expressed in terms of the covariance kernel $C(\mathbf{h})$:
$$
\operatorname{Cov}(D_{ij}^2, D_{ab}^2) = 2\left[C(\mathbf{s}_i - \mathbf{s}_a) - C(\mathbf{s}_i - \mathbf{s}_b) - C(\mathbf{s}_j - \mathbf{s}_a) + C(\mathbf{s}_j - \mathbf{s}_b)\right]^2
$$

*Proof.*
Because $\{Z(\mathbf{s})\}$ is Gaussian, the increments $D_{ij}$ and $D_{ab}$ are jointly zero-mean Gaussian. Applying Lemma 3.3.1:
$$
\operatorname{Cov}(D_{ij}^2, D_{ab}^2) = 2\left(\operatorname{Cov}(D_{ij}, D_{ab})\right)^2
$$
Expanding the covariance of linear differences via bilinearity:
$$
\operatorname{Cov}(D_{ij}, D_{ab}) = \operatorname{Cov}(Z(\mathbf{s}_i) - Z(\mathbf{s}_j), Z(\mathbf{s}_a) - Z(\mathbf{s}_b))
$$
$$
= \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_a)) - \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_b)) - \operatorname{Cov}(Z(\mathbf{s}_j), Z(\mathbf{s}_a)) + \operatorname{Cov}(Z(\mathbf{s}_j), Z(\mathbf{s}_b))
$$
$$
= C(\mathbf{s}_i - \mathbf{s}_a) - C(\mathbf{s}_i - \mathbf{s}_b) - C(\mathbf{s}_j - \mathbf{s}_a) + C(\mathbf{s}_j - \mathbf{s}_b)
$$
Substituting $C(\mathbf{u} - \mathbf{v}) = C(\mathbf{0}) - \gamma(\mathbf{u} - \mathbf{v})$:
$$
\operatorname{Cov}(D_{ij}, D_{ab}) = [C(\mathbf{0}) - \gamma(\mathbf{s}_i - \mathbf{s}_a)] - [C(\mathbf{0}) - \gamma(\mathbf{s}_i - \mathbf{s}_b)] - [C(\mathbf{0}) - \gamma(\mathbf{s}_j - \mathbf{s}_a)] + [C(\mathbf{0}) - \gamma(\mathbf{s}_j - \mathbf{s}_b)]
$$
The constant $C(\mathbf{0})$ terms cancel identically:
$$
C(\mathbf{0}) - C(\mathbf{0}) - C(\mathbf{0}) + C(\mathbf{0}) = 0
$$
leaving:
$$
\operatorname{Cov}(D_{ij}, D_{ab}) = \gamma(\mathbf{s}_i - \mathbf{s}_b) + \gamma(\mathbf{s}_j - \mathbf{s}_a) - \gamma(\mathbf{s}_i - \mathbf{s}_a) - \gamma(\mathbf{s}_j - \mathbf{s}_b)
$$
Squaring this expression and multiplying by 2 completes the proof. $\blacksquare$

##### Theorem 3.3.1 (Sampling Variance and Inter-Lag Covariance Matrix of $\widehat{\boldsymbol{\gamma}}$)
Under Gaussianity, the covariance between empirical semivariances at nominal lag bins $\mathbf{h}_1$ and $\mathbf{h}_2$ is:
$$
\operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_1), \widehat{\gamma}(\mathbf{h}_2)) = \frac{1}{2|N(\mathbf{h}_1)||N(\mathbf{h}_2)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_1)} \sum_{(\mathbf{s}_a, \mathbf{s}_b) \in N(\mathbf{h}_2)} \left[\gamma(\mathbf{s}_i - \mathbf{s}_b) + \gamma(\mathbf{s}_j - \mathbf{s}_a) - \gamma(\mathbf{s}_i - \mathbf{s}_a) - \gamma(\mathbf{s}_j - \mathbf{s}_b)\right]^2
$$
When $\mathbf{h}_1 = \mathbf{h}_2 = \mathbf{h}_k$, this gives the exact sampling variance:
$$
\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k)) = \frac{1}{2|N(\mathbf{h}_k)|^2} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} \sum_{(\mathbf{s}_a, \mathbf{s}_b) \in N(\mathbf{h}_k)} \left[\gamma(\mathbf{s}_i - \mathbf{s}_b) + \gamma(\mathbf{s}_j - \mathbf{s}_a) - \gamma(\mathbf{s}_i - \mathbf{s}_a) - \gamma(\mathbf{s}_j - \mathbf{s}_b)\right]^2
$$
Equivalently, in terms of Genton's pairwise differencing matrices $\mathbf{A}_1 = \mathbf{A}(\mathbf{h}_1)$ and $\mathbf{A}_2 = \mathbf{A}(\mathbf{h}_2)$ defined in Section 3.1 and the observation covariance matrix $\mathbf{V} = \operatorname{Var}(\mathbf{Z})$:
$$
\operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_1), \widehat{\gamma}(\mathbf{h}_2)) = \frac{\operatorname{Tr}(\mathbf{A}_1 \mathbf{V} \mathbf{A}_2 \mathbf{V})}{2|N(\mathbf{h}_1)||N(\mathbf{h}_2)|}
$$

*Proof.*
By Definition 3.1.2:
$$
\widehat{\gamma}(\mathbf{h}_1) = \frac{1}{2|N(\mathbf{h}_1)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_1)} D_{ij}^2, \quad \widehat{\gamma}(\mathbf{h}_2) = \frac{1}{2|N(\mathbf{h}_2)|} \sum_{(\mathbf{s}_a, \mathbf{s}_b) \in N(\mathbf{h}_2)} D_{ab}^2
$$
By the bilinearity of covariance:
$$
\operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_1), \widehat{\gamma}(\mathbf{h}_2)) = \frac{1}{4|N(\mathbf{h}_1)||N(\mathbf{h}_2)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_1)} \sum_{(\mathbf{s}_a, \mathbf{s}_b) \in N(\mathbf{h}_2)} \operatorname{Cov}(D_{ij}^2, D_{ab}^2)
$$
Substituting Proposition 3.3.1 for $\operatorname{Cov}(D_{ij}^2, D_{ab}^2)$ gives the double-summation formula directly. The trace representation follows from the standard result for quadratic forms of Gaussian vectors: $\operatorname{Cov}(\mathbf{Z}'\mathbf{A}_1\mathbf{Z}, \mathbf{Z}'\mathbf{A}_2\mathbf{Z}) = 2\operatorname{Tr}(\mathbf{A}_1 \mathbf{V} \mathbf{A}_2 \mathbf{V})$. $\blacksquare$

---

#### 3.3.2 Computing the Variogram Covariance Matrix in Python

Evaluating $\operatorname{Var}(\widehat{\gamma}(\mathbf{h}))$ or the full inter-lag covariance matrix $\mathbf{V}_{\widehat{\boldsymbol{\gamma}}} = [\operatorname{Cov}(\widehat{\gamma}(\mathbf{h}_p), \widehat{\gamma}(\mathbf{h}_q))]_{p,q=1}^K$ requires iterating over pair sets:

```python
import math

def compute_variogram_covariance_matrix(coords, lag_bins, gamma_func):
    """
    Computes the full K x K Gaussian covariance matrix of the empirical
    semivariogram estimator matching Theorem 3.3.1 in pure Python.
    
    Parameters:
        coords: list of lists, shape (n, d), sample sites s_i
        lag_bins: list of length K, where lag_bins[k] is a list of index pairs (i, j)
        gamma_func: callable, gamma(h) returning semivariance for scalar distance h
        
    Returns:
        V_gamma: list of lists, shape (K, K), Cov(gamma_hat(h_p), gamma_hat(h_q))
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_bins)
    V_gamma = [[0.0] * K for _ in range(K)]
    
    # Precompute pairwise semivariances between all spatial locations
    gamma_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            gamma_mat[i][j] = gamma_func(dist)
            
    for p in range(K):
        pairs_p = lag_bins[p]
        N_p = len(pairs_p)
        if N_p == 0:
            continue
            
        for q in range(p, K):
            pairs_q = lag_bins[q]
            N_q = len(pairs_q)
            if N_q == 0:
                continue
                
            total_sum = 0.0
            for (i, j) in pairs_p:
                for (a, b) in pairs_q:
                    # Covariance of increments: gamma(i,b) + gamma(j,a) - gamma(i,a) - gamma(j,b)
                    cov_d = (gamma_mat[i][b] + gamma_mat[j][a] 
                             - gamma_mat[i][a] - gamma_mat[j][b])
                    total_sum += cov_d * cov_d
                    
            entry = total_sum / (2.0 * N_p * N_q)
            V_gamma[p][q] = entry
            V_gamma[q][p] = entry
            
    return V_gamma
```

---

#### 3.3.3 What Inter-Lag Correlation Means in Practice: Gauss-Markov and Smoothed Jumps

Evaluating Theorem 3.3.1 reveals several profound practical implications for spatial modeling:

1. **Why Inter-Lag Correlations Are Strongly Positive:**
   In spatial statistics, empirical semivariogram values $\widehat{\gamma}(\mathbf{h}_1)$ and $\widehat{\gamma}(\mathbf{h}_2)$ are **always correlated**, even for distinct lags $\mathbf{h}_1 \ne \mathbf{h}_2$.
   
   The reason is geometric overlap: site $\mathbf{s}_2$ forms a pair with $\mathbf{s}_1$ in bin $N(\mathbf{h}_1)$ (distance $h_1$) and simultaneously forms a pair with $\mathbf{s}_3$ in bin $N(\mathbf{h}_2)$ (distance $h_2$). The shared observation $Z(\mathbf{s}_2)$ introduces structural covariance between the two bin estimates.
   
   To appreciate the magnitude of this effect, consider a Gaussian random field on a 1D transect of $n = 60$ regularly spaced sites with unit spacing ($\Delta h = 1$) and Exponential covariance $C(h) = \exp(-h/15)$. Evaluating the correlation matrix of $\widehat{\boldsymbol{\gamma}}$ via Theorem 3.3.1 yields:
   $$
   \operatorname{Corr}(\widehat{\gamma}(1), \widehat{\gamma}(2)) \approx 0.79, \quad \operatorname{Corr}(\widehat{\gamma}(1), \widehat{\gamma}(3)) \approx 0.65, \quad \operatorname{Corr}(\widehat{\gamma}(1), \widehat{\gamma}(10)) \approx 0.29
   $$
   Consecutive empirical variogram points share up to **$80\%$ of their variance**! This explains why sample variogram plots typically look smooth and continuous: estimation errors at adjacent lags move together in waves, rather than scattering independently above and below the true curve.
2. **Gauss-Markov Violation in Ordinary Least Squares (OLS) Fitting:**
   Treating empirical variogram points as independent, homoscedastic observations when fitting parametric curves using Ordinary Least Squares (OLS) violates both fundamental Gauss-Markov assumptions:
   - **Heteroscedasticity:** $\operatorname{Var}(\widehat{\gamma}(\mathbf{h}_k))$ grows rapidly with lag distance as the number of available pairs $|N(\mathbf{h}_k)|$ decreases and increments become more strongly cross-correlated.
   - **Autocorrelation:** Consecutive lag estimates are strongly correlated ($\rho \approx 0.7\text{--}0.8$).
   
   Fitting an OLS curve gives undue weight to noisy, highly correlated long-distance lags, dragging the fitted curve away from the critical near-origin behavior that governs kriging weights. This motivates Cressie's Weighted Least Squares (WLS) and Generalized Least Squares (GLS) fitting procedures, developed in Chapter 4.

---

### 3.4 Robust Estimation of Spatial Continuity

#### 3.4.1 Taming Outliers: Why We Need Robust Estimators

In classical applied statistics, when data exhibit severe positive skewness or outlier contamination, standard practice applies a variance-stabilizing power transformation (such as Box-Cox transformations or the square-root transformation $Y^{1/2}$ for Poisson counts).

In spatial statistics, consider the spatial increment $D_{ij} = Z(\mathbf{s}_i) - Z(\mathbf{s}_j)$. Even under perfect Gaussianity of the random field $Z(\mathbf{s})$, the squared increment $D_{ij}^2$ follows a scaled chi-squared distribution with 1 degree of freedom:
$$
\frac{D_{ij}^2}{2\gamma(\mathbf{h})} \sim \chi_1^2
$$
The $\chi_1^2$ distribution is exceptionally skewed and heavy-tailed: its skewness is $\sqrt{8} \approx 2.83$ and its excess kurtosis is $12$ (total kurtosis $15$). When a single outlier or recording error occurs at location $\mathbf{s}_k$, every pair containing $\mathbf{s}_k$ is inflated, and squaring that increment causes catastrophic spikes in Matheron's estimator $\widehat{\gamma}(\mathbf{h})$.

To robustify spatial continuity estimation, Noel Cressie and Douglas M. Hawkins (1980) applied the classical power-transformation principle to spatial increments. By replacing the squaring operation with the square-root modulus $|D_{ij}|^{1/2} = |Z(\mathbf{s}_i) - Z(\mathbf{s}_j)|^{1/2}$, the skewness of the transformed increment plummets from $2.83$ to approximately $0.08$—rendering the transformed increments virtually Gaussian and symmetric. Averaging these well-behaved square-root increments and then raising the average to the fourth power restores the correct physical dimension of variance, while theoretical normal correction constants ($0.457$ and $0.494$) eliminate the transformation bias.

> [!NOTE]
> **From the Lecture Hall: Jensen's Inequality and Why $\bar{Y}^4$ is Biased Upward**
>
> Why does the Cressie-Hawkins estimator divide by $0.457 + \frac{0.494}{|N(\mathbf{h})|}$ instead of simply $0.457$?
>
> Consider the function $g(y) = y^4$. For $y > 0$, its second derivative is $g''(y) = 12y^2 > 0$, meaning $g$ is **strictly convex**.
>
> By Jensen's Inequality, taking the fourth power of an average always exceeds the fourth power of the true mean:
> $$
> \operatorname{E}[\bar{Y}^4] > (\operatorname{E}[\bar{Y}])^4 = \mu_Y^4
> $$
> The variance of the sample mean $\bar{Y}$ pushes the expectation of $\bar{Y}^4$ upward by an amount proportional to $\operatorname{Var}(\bar{Y}) = \sigma_Y^2 / N$.
>
> If we divided $\bar{Y}^4$ only by the asymptotic factor $2 B \approx 2 \times 0.457$, the estimator would suffer a persistent positive finite-sample bias. The second term, $\frac{0.494}{|N(\mathbf{h})|}$, exactly cancels this positive Jensen inflation.

#### 3.4.2 The Cressie-Hawkins Estimator and Its Moments

##### Definition 3.4.1 (Cressie-Hawkins Modulus Semivariogram Estimator)
To achieve robustness against heavy-tailed contamination and isolated outliers, Noel Cressie and Douglas M. Hawkins (1980) introduced the fourth-root modulus estimator:
$$
\bar{\gamma}_{\text{CH}}(\mathbf{h}_k) = \frac{\left( \frac{1}{|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} |Z(\mathbf{s}_i) - Z(\mathbf{s}_j)|^{1/2} \right)^4}{2\left( 0.457 + \frac{0.494}{|N(\mathbf{h}_k)|} \right)}
$$
The transformation $|Z(\mathbf{s}_i) - Z(\mathbf{s}_j)|^{1/2}$ stabilizes variance and compresses the tails of the increment distribution.

###### Term-by-Term Equation Anatomy: Cressie-Hawkins Robust Estimator
$$
\bar{\gamma}_{\text{CH}}(\mathbf{h}_k) = \frac{\left( \frac{1}{|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} |Z(\mathbf{s}_i) - Z(\mathbf{s}_j)|^{1/2} \right)^4}{2\left( 0.457 + \frac{0.494}{|N(\mathbf{h}_k)|} \right)}
$$
- $\bar{\gamma}_{\text{CH}}(\mathbf{h}_k)$: The robust empirical semivariance estimate at lag $\mathbf{h}_k$.
- $|Z(\mathbf{s}_i) - Z(\mathbf{s}_j)|^{1/2}$: The square-root absolute spatial increment. Acts as a variance-stabilizing and tail-compressing transformation, converting a heavily skewed $\chi_1^2$ increment (skewness $\approx 2.83$) into a near-Gaussian variable with skewness $\approx 0.08$.
- $\frac{1}{|N(\mathbf{h}_k)|}\sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} \cdots$: The sample arithmetic mean of the square-root modulus differences across all $|N(\mathbf{h}_k)|$ pairs in the lag bin. Because the transformed increments are near-Gaussian, their sample mean is far less sensitive to isolated outliers than the mean of squared increments.
- $(\cdots)^4$: The fourth-power re-inflation, restoring the compressed square-root values back to the squared physical units of variance.
- $2$: The semivariogram halving factor, maintaining duality with covariance $\gamma(\mathbf{h}) = C(\mathbf{0}) - C(\mathbf{h})$.
- $0.457$: The asymptotic Gaussian calibration constant $B = (\operatorname{E}[|U|^{1/2}])^4 \approx 0.457066$ for standard normal $U \sim \mathcal{N}(0, 1)$, ensuring asymptotic unbiasedness as pair count $|N(\mathbf{h}_k)| \to \infty$.
- $\frac{0.494}{|N(\mathbf{h}_k)|}$: The finite-sample correction term $B_1 / |N(\mathbf{h}_k)| \approx 0.49457 / |N(\mathbf{h}_k)|$, correcting for the positive Jensen-type bias induced when raising a finite sample mean to the fourth power.

##### Theorem 3.4.1 (Derivation of the Cressie-Hawkins Gaussian Correction Factors)
Let $X \sim \mathcal{N}(0, 2\gamma(\mathbf{h}))$. Let $Y = |X|^{1/2}$ denote the square-root absolute increment.
1. The expectation of $Y$ is:
   $$
   \operatorname{E}[Y] = (2\gamma(\mathbf{h}))^{1/4} \frac{2^{1/4}\Gamma(3/4)}{\sqrt{\pi}}
   $$
   and its fourth power satisfies:
   $$
   (\operatorname{E}[Y])^4 = 2\gamma(\mathbf{h}) \cdot \frac{2[\Gamma(3/4)]^4}{\pi^2} \approx 2\gamma(\mathbf{h}) \times 0.457066
   $$
2. Let $N = |N(\mathbf{h})|$. Under independent increments, the expectation of the sample mean $\bar{Y} = \frac{1}{N}\sum_{k=1}^N Y_k$ raised to the fourth power is:
   $$
   \operatorname{E}[\bar{Y}^4] = 2\gamma(\mathbf{h})\left( 0.457066 + \frac{0.49457}{N} + \mathcal{O}(N^{-2}) \right)
   $$
   Consequently, the denominator $2\left(0.457 + \frac{0.494}{|N(\mathbf{h})|}\right)$ ensures that $\operatorname{E}[\bar{\gamma}_{CH}(\mathbf{h})] = \gamma(\mathbf{h}) + \mathcal{O}(|N(\mathbf{h})|^{-2})$ under independent Gaussian increments.

*Proof.*
We proceed through three steps.

*Step 1: Fractional moments of the folded standard normal.*
Let $U \sim \mathcal{N}(0, 1)$. The density of $|U|$ is the folded standard normal density $f_{|U|}(u) = \sqrt{\frac{2}{\pi}} e^{-u^2/2}$ for $u \ge 0$.
For any $p > -1$:
$$
\operatorname{E}[|U|^p] = \int_0^\infty u^p \sqrt{\frac{2}{\pi}} e^{-u^2/2} du
$$
Substitute $t = u^2/2$, so $u = (2t)^{1/2}$ and $du = \frac{\sqrt{2}}{2}t^{-1/2}dt$:
$$
\operatorname{E}[|U|^p] = \sqrt{\frac{2}{\pi}} \int_0^\infty (2t)^{p/2} e^{-t} \frac{\sqrt{2}}{2}t^{-1/2}dt = \frac{2^{p/2}}{\sqrt{\pi}} \int_0^\infty t^{\frac{p+1}{2} - 1} e^{-t} dt = \frac{2^{p/2}\Gamma\left(\frac{p+1}{2}\right)}{\sqrt{\pi}}
$$
Evaluating for $p = 1/2$ and $p = 1$:
$$
\mu_U^* = \operatorname{E}[|U|^{1/2}] = \frac{2^{1/4}\Gamma(3/4)}{\sqrt{\pi}} \approx \frac{1.189207 \times 1.225417}{1.772454} \approx 0.822177
$$
$$
\operatorname{E}[|U|] = \frac{\sqrt{2}\Gamma(1)}{\sqrt{\pi}} = \sqrt{\frac{2}{\pi}} \approx 0.797885
$$
The variance of $|U|^{1/2}$ is:
$$
\sigma_U^{*2} = \operatorname{Var}(|U|^{1/2}) = \operatorname{E}[|U|] - (\operatorname{E}[|U|^{1/2}])^2 = \sqrt{\frac{2}{\pi}} - (\mu_U^*)^2 \approx 0.797885 - 0.675975 = 0.121910
$$

*Step 2: Moments of the scaled spatial increment.*
Because $X = Z(\mathbf{s}_i) - Z(\mathbf{s}_j) \sim \mathcal{N}(0, 2\gamma(\mathbf{h}))$, we write $X = (2\gamma(\mathbf{h}))^{1/2} U$.
Thus, $Y = |X|^{1/2} = (2\gamma(\mathbf{h}))^{1/4} |U|^{1/2}$.
The mean and variance of $Y$ are:
$$
\mu_Y = \operatorname{E}[Y] = (2\gamma(\mathbf{h}))^{1/4} \mu_U^*, \quad \sigma_Y^2 = \operatorname{Var}(Y) = (2\gamma(\mathbf{h}))^{1/2} \sigma_U^{*2}
$$
Raising $\mu_Y$ to the fourth power:
$$
\mu_Y^4 = 2\gamma(\mathbf{h}) (\mu_U^*)^4 = 2\gamma(\mathbf{h}) \frac{(2^{1/4})^4 [\Gamma(3/4)]^4}{\pi^2} = 2\gamma(\mathbf{h}) \frac{2[\Gamma(3/4)]^4}{\pi^2}
$$
Defining the baseline constant $B$:
$$
B = (\mu_U^*)^4 = (0.822177)^4 \approx 0.457066 \approx 0.457
$$

*Step 3: Fourth power of the sample mean via Taylor expansion.*
Let $\bar{Y} = \frac{1}{N}\sum_{k=1}^N Y_k$. We write $\bar{Y} = \mu_Y + \delta$, where $\operatorname{E}[\delta] = 0$, $\operatorname{E}[\delta^2] = \operatorname{Var}(\bar{Y}) = \frac{\sigma_Y^2}{N}$, and $\operatorname{E}[\delta^4] = \mathcal{O}(N^{-2})$. For independent increments, the third central moment is $\operatorname{E}[\delta^3] = \frac{1}{N^2}\operatorname{E}[(Y - \mu_Y)^3] = \mathcal{O}(N^{-2})$, which drops directly into the remainder.
Expanding $\bar{Y}^4$ in a Taylor series about $\mu_Y$:
$$
\bar{Y}^4 = (\mu_Y + \delta)^4 = \mu_Y^4 + 4\mu_Y^3 \delta + 6\mu_Y^2 \delta^2 + 4\mu_Y \delta^3 + \delta^4
$$
Taking mathematical expectations:
$$
\operatorname{E}[\bar{Y}^4] = \mu_Y^4 + 6\mu_Y^2 \operatorname{E}[\delta^2] + \mathcal{O}(N^{-2}) = \mu_Y^4 + 6\mu_Y^2 \frac{\sigma_Y^2}{N} + \mathcal{O}(N^{-2})
$$
Factoring out $\mu_Y^4$:
$$
\operatorname{E}[\bar{Y}^4] = \mu_Y^4 \left( 1 + \frac{6}{N} \frac{\sigma_Y^2}{\mu_Y^2} \right) + \mathcal{O}(N^{-2})
$$
Notice that the ratio $\frac{\sigma_Y^2}{\mu_Y^2}$ simplifies completely:
$$
\frac{\sigma_Y^2}{\mu_Y^2} = \frac{(2\gamma(\mathbf{h}))^{1/2}\sigma_U^{*2}}{(2\gamma(\mathbf{h}))^{1/2}(\mu_U^*)^2} = \frac{\sigma_U^{*2}}{(\mu_U^*)^2} = \frac{0.121910}{0.675975} \approx 0.180347
$$
Substituting $\mu_Y^4 = 2\gamma(\mathbf{h}) B$:
$$
\operatorname{E}[\bar{Y}^4] = 2\gamma(\mathbf{h}) \left( B + \frac{6 B}{N} \frac{\sigma_U^{*2}}{(\mu_U^*)^2} \right) + \mathcal{O}(N^{-2}) = 2\gamma(\mathbf{h}) \left( B + \frac{B_1}{N} \right) + \mathcal{O}(N^{-2})
$$
Evaluating the finite-sample correction constant $B_1$:
$$
B_1 = 6 B \frac{\sigma_U^{*2}}{(\mu_U^*)^2} = 6 \times 0.457066 \times 0.180347 \approx 0.49457 \approx 0.494
$$
Dividing $\bar{Y}^4$ by $2\left(0.457 + \frac{0.494}{|N(\mathbf{h})|}\right)$ yields an estimator whose expectation equals $\gamma(\mathbf{h}) + \mathcal{O}(N^{-2})$. $\blacksquare$

##### Definition 3.4.2 (The Pairwise Relative Semivariogram)
When the local variance exhibits a strong *proportional effect*—where local dispersion scales with the local mean ($\operatorname{Var}(Z(\mathbf{s})) \propto (\operatorname{E}[Z(\mathbf{s})])^2$, common in environmental pollutant concentrations and mineral grades)—increments in high-value zones overwhelm increments in low-value zones.

The *pairwise relative semivariogram* normalizes each squared increment by the square of the pair average:
$$
\widehat{\gamma}_R(\mathbf{h}_k) = \frac{2}{|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_i, \mathbf{s}_j) \in N(\mathbf{h}_k)} \left( \frac{Z(\mathbf{s}_i) - Z(\mathbf{s}_j)}{Z(\mathbf{s}_i) + Z(\mathbf{s}_j)} \right)^2
$$
provided that the physical process is strictly positive ($Z(\mathbf{s}) > 0$ almost surely). The resulting relative semivariogram is dimensionless and stabilizes spatial continuity estimates across disparate concentration regimes.

---

#### 3.4.3 Computing Robust Variograms in Python

```python
import math

def compute_cressie_hawkins_semivariogram(coords, values, lag_centers, lag_bandwidth):
    """
    Computes the robust Cressie-Hawkins modulus semivariogram matching Definition 3.4.1
    in pure standard Python.
    
    Parameters:
        coords: list of lists, shape (n, d), sample sites s_i
        values: list of floats, shape (n,), observations Z(s_i)
        lag_centers: list of floats, shape (K,), nominal lag bin centers h_k
        lag_bandwidth: float, radial width of lag bins Delta_h
        
    Returns:
        actual_lags: list of floats of length K, mean distance per bin
        gamma_ch: list of floats of length K, robust semivariance estimates
        pair_counts: list of ints of length K, pair count per bin |N(h_k)|
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    sum_root_diff = [0.0] * K
    sum_distance = [0.0] * K
    pair_counts = [0] * K
    
    # Traverse all unique pairs i < j
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = math.sqrt(sum((coords[i][m] - coords[j][m]) ** 2 for m in range(d)))
            
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    abs_diff_root = math.sqrt(abs(values[i] - values[j]))
                    sum_root_diff[k] += abs_diff_root
                    sum_distance[k] += dist
                    pair_counts[k] += 1
                    break
                    
    gamma_ch = [None] * K
    actual_lags = [None] * K
    
    for k in range(K):
        N_k = pair_counts[k]
        if N_k > 0:
            actual_lags[k] = sum_distance[k] / N_k
            mean_root = sum_root_diff[k] / N_k
            denominator = 2.0 * (0.457066 + (0.49457 / N_k))
            gamma_ch[k] = (mean_root ** 4) / denominator
            
    return actual_lags, gamma_ch, pair_counts
```

---

#### 3.4.4 What Can Go Wrong: Dependent Pairs and Non-Gaussian Increments

1. **The Dependent-Pair Overestimation Effect:**
   The finite-sample correction term $\frac{0.494}{|N(\mathbf{h})|}$ in Theorem 3.4.1 was derived under the explicit mathematical assumption that the $|N(\mathbf{h})|$ increments in the lag bin are **statistically independent**.
   
   In spatial random fields, increments share observation locations and are spatially correlated. Consequently, the *effective sample size* $N_{\text{eff}}$ is strictly smaller than the nominal pair count $|N(\mathbf{h})|$. Because the denominator uses $|N(\mathbf{h})|$ instead of the smaller $N_{\text{eff}}$, the term $\frac{0.494}{|N(\mathbf{h})|}$ **under-corrects** the positive Jensen bias. On spatial grids, the Cressie-Hawkins estimator systematically **overestimates** the true semivariogram $\gamma(\mathbf{h})$ by $5\%\text{ to }10\%$ at moderate lag separations. Practitioners must be aware that while $\bar{\gamma}_{\text{CH}}$ suppresses isolated outliers, it is not strictly unbiased on spatial data.
2. **Breakdown Point and Sensitivity:**
   The sample mean of the square-root absolute increments $|D|^{1/2}$ still has a nominal asymptotic breakdown point of $0\%$ (in formal robustness theory, a single point moved to infinity eventually pulls the sample mean). However, the square-root compression drastically reduces sensitivity: an outlier that inflates Matheron's squared difference by $10^4$ inflates the Cressie-Hawkins square-root increment by only $\sqrt{100} = 10$. It is accurately described as *tail-compressed* and *less sensitive*, rather than unconditionally outlier-immune.
3. **Distributional Sensitivity under Heavy-Tailed Processes:**
   The correction factor $0.457$ is tied directly to the Gaussian distribution. If the underlying process has heavier tails (e.g., Student-$t$, log-Gaussian, or Cauchy increments), the ratio $(\operatorname{E}[|D|^{1/2}])^4 / (2\gamma(\mathbf{h}))$ is strictly lower than $0.457$ (see Exercise 3.3). In such settings, using the unadjusted Gaussian Cressie-Hawkins estimator will systematically **underestimate** the true semivariogram.

> [!TIP]
> **Which Estimator Should I Use: Matheron or Cressie-Hawkins?**
>
> | Data Regime / Characteristics | Recommended Estimator | Rationale |
> | :--- | :--- | :--- |
> | **Pure Gaussian Field (Clean Data)** | **Matheron** $\widehat{\gamma}(\mathbf{h})$ | Optimal efficiency, minimum variance unbiased estimator; no dependent-pair overestimation. |
> | **Occasional Spikes / Recording Errors ($< 15\%$)** | **Cressie-Hawkins** $\bar{\gamma}_{\text{CH}}(\mathbf{h})$ | Fourth-root modulus compresses outlier spikes; keeps variogram stable for parametric curve fitting. |
> | **Proportional Effect ($Z > 0$, Variance $\propto \text{Mean}^2$)** | **Pairwise Relative** $\widehat{\gamma}_R(\mathbf{h})$ | Normalizes increments by pair sum; prevents high-grade clusters from dominating global variogram. |
> | **Genuinely Heavy-Tailed Field (Log-Gaussian / $t$-field)** | **Normal-Score Transform + Matheron** | Cressie-Hawkins underestimates without custom calibration; transforming $Z \mapsto \Phi^{-1}(F_n(Z))$ is safer. |

---

### 3.5 Multivariate Spatial Continuity: Cross-Covariances and Cross-Variograms

> [!NOTE]
> **Advanced Multivariate Bridge: Preparing for Cokriging in Chapter 7**
>
> In many environmental, mining, and atmospheric applications, a target variable $Z_1(\mathbf{s})$ (e.g., heavy metal soil contamination) is expensive to measure, while a secondary variable $Z_2(\mathbf{s})$ (e.g., elevation or radiometric gamma count) is available densely across the study area. In Chapter 7, we will study **Cokriging**, which exploits spatial cross-correlation to dramatically improve predictions of $Z_1$. The cross-covariance and cross-variogram functions developed in this section provide the essential foundation for that theory.

#### 3.5.1 The Multivariate Landscape: Cross-Covariances and Cross-Variograms

In classical multivariate statistics, the joint variation of two random variables $Y_1$ and $Y_2$ is captured by the Pearson covariance:
$$
\operatorname{Cov}(Y_1, Y_2) = \operatorname{E}[(Y_1 - \mu_1)(Y_2 - \mu_2)]
$$
Estimating this covariance requires *paired* observations $(Y_{1k}, Y_{2k})$ recorded simultaneously on the same sampling units.

When $m$ continuous spatial processes $\mathbf{Z}(\mathbf{s}) = (Z_1(\mathbf{s}), \dots, Z_m(\mathbf{s}))'$ vary simultaneously across geography, their co-dependence depends both on the variables themselves and on their spatial displacement vector $\mathbf{h} = \mathbf{s}_i - \mathbf{s}_j$.

##### Definition 3.5.1 (Multivariate Second-Order Stationarity)
An $m$-variate random field $\{\mathbf{Z}(\mathbf{s}) : \mathbf{s} \in D\}$ is *multivariate second-order stationary* if:
1. The mean vector is spatially constant: $\operatorname{E}[\mathbf{Z}(\mathbf{s})] = \boldsymbol{\mu} = (\mu_1, \dots, \mu_m)' \in \mathbb{R}^m$.
2. For all $i, j \in \{1, \dots, m\}$ and any $\mathbf{s}, \mathbf{s}+\mathbf{h} \in D$, the cross-covariance depends only on displacement $\mathbf{h}$:
   $$
   C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}), Z_j(\mathbf{s}+\mathbf{h})) = \operatorname{E}\left[(Z_i(\mathbf{s}) - \mu_i)(Z_j(\mathbf{s}+\mathbf{h}) - \mu_j)\right]
   $$

##### Definition 3.5.2 (The Classical Cross-Semivariogram)
Under multivariate intrinsic stationarity, the classical *cross-semivariogram* between attributes $Z_i$ and $Z_j$ is:
$$
\gamma_{ij}(\mathbf{h}) = \frac{1}{2}\operatorname{Cov}\left(Z_i(\mathbf{s}+\mathbf{h}) - Z_i(\mathbf{s}), \; Z_j(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s})\right)
$$
Its empirical method-of-moments estimator is:
$$
\widehat{\gamma}_{ij}(\mathbf{h}_k) = \frac{1}{2|N(\mathbf{h}_k)|} \sum_{(\mathbf{s}_r, \mathbf{s}_l) \in N(\mathbf{h}_k)} (Z_i(\mathbf{s}_r) - Z_i(\mathbf{s}_l))(Z_j(\mathbf{s}_r) - Z_j(\mathbf{s}_l))
$$

##### Theorem 3.5.1 (Relation Between Cross-Covariance and Cross-Semivariogram)
Under multivariate second-order stationarity:
$$
\gamma_{ij}(\mathbf{h}) = C_{ij}(\mathbf{0}) - \frac{1}{2}\left(C_{ij}(\mathbf{h}) + C_{ij}(-\mathbf{h})\right)
$$

*Proof.*
Expanding the covariance of increments:
$$
2\gamma_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}) - Z_i(\mathbf{s}), Z_j(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s}))
$$
$$
= \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s}+\mathbf{h})) - \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s})) - \operatorname{Cov}(Z_i(\mathbf{s}), Z_j(\mathbf{s}+\mathbf{h})) + \operatorname{Cov}(Z_i(\mathbf{s}), Z_j(\mathbf{s}))
$$
By stationarity:
1. $\operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s}+\mathbf{h})) = C_{ij}(\mathbf{0})$.
2. $\operatorname{Cov}(Z_i(\mathbf{s}), Z_j(\mathbf{s})) = C_{ij}(\mathbf{0})$.
3. $\operatorname{Cov}(Z_i(\mathbf{s}), Z_j(\mathbf{s}+\mathbf{h})) = C_{ij}(\mathbf{h})$.
4. $\operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s})) = C_{ij}(-\mathbf{h})$.

Collecting terms:
$$
2\gamma_{ij}(\mathbf{h}) = 2C_{ij}(\mathbf{0}) - (C_{ij}(\mathbf{h}) + C_{ij}(-\mathbf{h}))
$$
Dividing by 2 yields $\gamma_{ij}(\mathbf{h}) = C_{ij}(\mathbf{0}) - \frac{1}{2}(C_{ij}(\mathbf{h}) + C_{ij}(-\mathbf{h}))$. $\blacksquare$

---

#### 3.5.2 Heterotopic Data and Clark's Pseudo-Cross Variogram

In field applications, sampling designs fall into two distinct structural regimes:
- **Isotopic sampling:** All $m$ attributes are measured at every sampling site ($S_1 = S_2 = \dots = S_m = S$).
- **Heterotopic sampling:** The observation networks differ. Under *totally heterotopic sampling*, $S_i \cap S_j = \emptyset$ (no sample site has joint observations of both $Z_i$ and $Z_j$).

Under totally heterotopic sampling, Matheron's classical cross-variogram $\widehat{\gamma}_{ij}(\mathbf{h})$ **cannot be computed**, because the cross-product $(Z_i(\mathbf{s}_r) - Z_i(\mathbf{s}_l))(Z_j(\mathbf{s}_r) - Z_j(\mathbf{s}_l))$ requires measuring both variables at both locations $\mathbf{s}_r$ and $\mathbf{s}_l$.

To overcome this sampling bottleneck, Isobel Clark et al. (1989) introduced the *pseudo-cross semivariogram*:
$$
\gamma_{ij}^P(\mathbf{h}) = \frac{1}{2} \operatorname{Var}(Z_i(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s}))
$$
The empirical estimator for the pseudo-cross variogram is:
$$
\widehat{\gamma}_{ij}^P(\mathbf{h}_k) = \frac{1}{2|N_{ij}(\mathbf{h}_k)|} \sum_{(\mathbf{s}_r, \mathbf{s}_l) \in N_{ij}(\mathbf{h}_k)} (Z_i(\mathbf{s}_r) - Z_j(\mathbf{s}_l))^2
$$
where $N_{ij}(\mathbf{h}_k) = \{(\mathbf{s}_r, \mathbf{s}_l) \in S_i \times S_j : \mathbf{s}_r - \mathbf{s}_l \approx \mathbf{h}_k\}$. This estimator requires only that $Z_i$ be measured at $\mathbf{s}_r$ and $Z_j$ at $\mathbf{s}_l$.

##### Theorem 3.5.2 (Structural Representation of the Pseudo-Cross Variogram)
Let $\mathbf{Z}(\mathbf{s})$ be a bivariate second-order stationary random field with means $\mu_i, \mu_j$, variances $\sigma_i^2 = C_{ii}(\mathbf{0}), \sigma_j^2 = C_{jj}(\mathbf{0})$, and cross-covariance $C_{ji}(\mathbf{h}) = \operatorname{Cov}(Z_j(\mathbf{s}), Z_i(\mathbf{s}+\mathbf{h}))$. Then:
$$
\gamma_{ij}^P(\mathbf{h}) = \frac{1}{2}(\sigma_i^2 + \sigma_j^2) - C_{ji}(\mathbf{h})
$$
If applied to uncentered data, the expectation of the raw mean-squared difference contains an added mean offset:
$$
\frac{1}{2}\operatorname{E}\left[(Z_i(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s}))^2\right] = \gamma_{ij}^P(\mathbf{h}) + \frac{1}{2}(\mu_i - \mu_j)^2
$$

*Proof.*
By variance expansion:
$$
2\gamma_{ij}^P(\mathbf{h}) = \operatorname{Var}(Z_i(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s})) = \operatorname{Var}(Z_i(\mathbf{s}+\mathbf{h})) + \operatorname{Var}(Z_j(\mathbf{s})) - 2\operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s}))
$$
By stationarity, $\operatorname{Var}(Z_i(\mathbf{s}+\mathbf{h})) = \sigma_i^2$, $\operatorname{Var}(Z_j(\mathbf{s})) = \sigma_j^2$, and:
$$
\operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s})) = \operatorname{Cov}(Z_j(\mathbf{s}), Z_i(\mathbf{s}+\mathbf{h})) = C_{ji}(\mathbf{h}) = C_{ij}(-\mathbf{h})
$$
Note carefully that $C_{ji}(\mathbf{h})$ appears here rather than $C_{ij}(\mathbf{h})$ because $Z_i$ is evaluated at $\mathbf{s}+\mathbf{h}$ and $Z_j$ at $\mathbf{s}$. Thus:
$$
2\gamma_{ij}^P(\mathbf{h}) = \sigma_i^2 + \sigma_j^2 - 2C_{ji}(\mathbf{h}) \implies \gamma_{ij}^P(\mathbf{h}) = \frac{1}{2}(\sigma_i^2 + \sigma_j^2) - C_{ji}(\mathbf{h})
$$
For the uncentered expectation:
$$
\operatorname{E}\left[(Z_i(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s}))^2\right] = \operatorname{Var}(Z_i(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s})) + \left(\operatorname{E}[Z_i(\mathbf{s}+\mathbf{h}) - Z_j(\mathbf{s})]\right)^2
$$
$$
= 2\gamma_{ij}^P(\mathbf{h}) + (\mu_i - \mu_j)^2
$$
Dividing by 2 establishes the theorem. $\blacksquare$

---

#### 3.5.3 Computing Cross-Variograms in Python

The following implementation computes the omnidirectional pseudo-cross semivariogram across two disparate observation networks $S_i$ and $S_j$:

```python
import math

def compute_heterotopic_pseudo_cross_variogram(coords_i, values_i, coords_j, values_j, 
                                                lag_centers, lag_bandwidth):
    """
    Computes Clark's empirical omnidirectional pseudo-cross semivariogram 
    matching Definition 3.5.3 for heterotopic sampling geometries in pure Python.
    
    Parameters:
        coords_i: list of lists, shape (n_i, d), locations of variable i
        values_i: list of floats, shape (n_i,), measurements Z_i
        coords_j: list of lists, shape (n_j, d), locations of variable j
        values_j: list of floats, shape (n_j,), measurements Z_j
        lag_centers: list of floats, shape (K,), nominal lag bin centers h_k
        lag_bandwidth: float, radial width Delta_h
        
    Returns:
        actual_lags: list of floats of length K, mean distance per bin
        pseudo_gamma: list of floats of length K, estimated pseudo-cross semivariance
        pair_counts: list of ints of length K, count of cross-pairs
    """
    n_i = len(coords_i)
    n_j = len(coords_j)
    d = len(coords_i[0])
    K = len(lag_centers)
    half_bw = lag_bandwidth / 2.0
    
    sum_sq_diff = [0.0] * K
    sum_distance = [0.0] * K
    pair_counts = [0] * K
    
    # Bipartite traversal across disparate networks S_i and S_j
    for r in range(n_i):
        for l in range(n_j):
            dist = math.sqrt(sum((coords_i[r][m] - coords_j[l][m]) ** 2 for m in range(d)))
            
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    diff = values_i[r] - values_j[l]
                    sum_sq_diff[k] += diff * diff
                    sum_distance[k] += dist
                    pair_counts[k] += 1
                    break
                    
    pseudo_gamma = [None] * K
    actual_lags = [None] * K
    
    for k in range(K):
        if pair_counts[k] > 0:
            actual_lags[k] = sum_distance[k] / pair_counts[k]
            pseudo_gamma[k] = sum_sq_diff[k] / (2.0 * pair_counts[k])
            
    return actual_lags, pseudo_gamma, pair_counts
```

---

#### 3.5.4 What Can Go Wrong: Units, Scale Mismatches, and the Non-Zero Intercept

1. **The Positive Zero-Distance Intercept:**
   Unlike direct semivariograms where $\gamma_{ii}(\mathbf{0}) = 0$, the pseudo-cross variogram has a strictly positive value at zero separation:
   $$
   \gamma_{ij}^P(\mathbf{0}) = \frac{1}{2}\operatorname{Var}(Z_i(\mathbf{s}) - Z_j(\mathbf{s})) = \frac{1}{2}(\sigma_i^2 + \sigma_j^2) - C_{ij}(\mathbf{0})
   $$
   In plain English: even if variable $Z_i$ and variable $Z_j$ are measured at the exact same location, they are different physical quantities. Unless $Z_i(\mathbf{s}) \equiv Z_j(\mathbf{s})$, their difference has non-zero dispersion. Students seeing $\gamma_{ij}^P(0) > 0$ for the first time often mistake it for a massive nugget effect; it is simply the co-located cross-discrepancy.
2. **Dimensional Inconsistency & Unit Sensitivity:**
   If $Z_i$ is soil copper concentration in parts per million ($\mathrm{ppm}$) and $Z_j$ is digital elevation in meters ($\mathrm{m}$), subtracting them ($(Z_i - Z_j)^2$) is physically and mathematically nonsensical. Before computing a pseudo-cross variogram, both variables **must be standardized to zero mean and unit variance**:
   $$
   Z_k^*(\mathbf{s}) = \frac{Z_k(\mathbf{s}) - \bar{Z}_k}{s_k}
   $$
   Standardization eliminates the squared mean difference penalty $\frac{1}{2}(\mu_i - \mu_j)^2$ and equalizes sills ($\sigma_i^{*2} = \sigma_j^{*2} = 1$).

---

### 3.6 The Delay / Lag Effect and Asymmetric Spatial Cross-Continuity

#### 3.6.1 The Advection Problem: When Spatial Dependence is Not Symmetric

In univariate spatial statistics, direct covariance functions are inherently symmetric across the origin:
$$
C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h})) = \operatorname{Cov}(Z(\mathbf{s}+\mathbf{h}), Z(\mathbf{s})) = C(-\mathbf{h})
$$
Consequently, univariate semivariograms are always even functions: $\gamma(\mathbf{h}) = \gamma(-\mathbf{h})$.

However, in environmental systems governed by physical flow—such as atmospheric smokestack dispersion, river sediment transport, groundwater contaminant plumes, or oceanic currents—the spatial cross-correlation between an emission process $Z_1(\mathbf{s})$ and a downstream deposition process $Z_2(\mathbf{s})$ exhibits a physical **delay** (or **transport lag**). The strongest correlation does not occur at co-located points ($\mathbf{h} = \mathbf{0}$), but at a positive spatial displacement downwind or downstream ($\mathbf{h} = \mathbf{v}$).

> [!NOTE]
> **From the Lecture Hall: Even and Odd Decompositions of Cross-Dependence Functions**
>
> Any real-valued function $f(\mathbf{h})$ can be decomposed uniquely into an even (symmetric) part and an odd (skew-symmetric) part:
> $$
> f(\mathbf{h}) = \underbrace{\frac{f(\mathbf{h}) + f(-\mathbf{h})}{2}}_{f_S(\mathbf{h}) \text{ (even)}} + \underbrace{\frac{f(\mathbf{h}) - f(-\mathbf{h})}{2}}_{f_A(\mathbf{h}) \text{ (odd)}}
> $$
> where $f_S(-\mathbf{h}) = f_S(\mathbf{h})$ and $f_A(-\mathbf{h}) = -f_A(\mathbf{h})$.
>
> In spatial statistics:
> - Auto-covariances have $C_A(\mathbf{h}) \equiv 0$, so $C(\mathbf{h}) = C_S(\mathbf{h})$ is purely even.
> - Cross-covariances $C_{12}(\mathbf{h})$ contain both an even component $C_{12}^S(\mathbf{h})$ and an odd component $C_{12}^A(\mathbf{h})$.
>
> The fundamental hazard of Matheron's classical cross-variogram $\gamma_{12}(\mathbf{h})$ is that its definition forcibly strips away $C_{12}^A(\mathbf{h})$, rendering it blind to directional transport.

#### 3.6.2 The Even-Odd Decomposition and Why Cross-Variograms Miss Delay

##### Proposition 3.6.1 (Transpose Symmetry of Cross-Covariances)
Let $\mathbf{Z}(\mathbf{s})$ be a multivariate second-order stationary random field. The cross-covariance function satisfies *transpose symmetry*:
$$
C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}), Z_j(\mathbf{s}+\mathbf{h})) = \operatorname{Cov}(Z_j(\mathbf{s}+\mathbf{h}), Z_i(\mathbf{s})) = C_{ji}(-\mathbf{h})
$$
However, in general:
$$
C_{ij}(\mathbf{h}) \ne C_{ij}(-\mathbf{h}) \quad \text{for } i \ne j
$$
When $C_{ij}(\mathbf{h}) \ne C_{ij}(-\mathbf{h})$, the cross-dependence exhibits a spatial *delay effect* or *directional lag*.

##### Theorem 3.6.1 (Even-Odd Decomposition of Cross-Covariances)
Any spatial cross-covariance function $C_{ij}(\mathbf{h})$ decomposes uniquely into an even (symmetric) kernel $C_{ij}^S(\mathbf{h})$ and an odd (skew-symmetric) kernel $C_{ij}^A(\mathbf{h})$:
$$
C_{ij}(\mathbf{h}) = C_{ij}^S(\mathbf{h}) + C_{ij}^A(\mathbf{h})
$$
where:
$$
C_{ij}^S(\mathbf{h}) = \frac{1}{2}\left(C_{ij}(\mathbf{h}) + C_{ij}(-\mathbf{h})\right), \quad C_{ij}^S(-\mathbf{h}) = C_{ij}^S(\mathbf{h})
$$
$$
C_{ij}^A(\mathbf{h}) = \frac{1}{2}\left(C_{ij}(\mathbf{h}) - C_{ij}(-\mathbf{h})\right), \quad C_{ij}^A(-\mathbf{h}) = -C_{ij}^A(\mathbf{h})
$$

##### Theorem 3.6.2 (The Blindness of Classical Cross-Variograms to Delay Effects)
The classical cross-semivariogram $\gamma_{ij}(\mathbf{h})$ depends exclusively on the even component $C_{ij}^S(\mathbf{h})$ and is completely blind to the odd component $C_{ij}^A(\mathbf{h})$:
$$
\gamma_{ij}(\mathbf{h}) = C_{ij}(\mathbf{0}) - C_{ij}^S(\mathbf{h})
$$
Consequently, the classical cross-variogram is always strictly symmetric:
$$
\gamma_{ij}(\mathbf{h}) = \gamma_{ij}(-\mathbf{h}), \quad \forall \mathbf{h} \in \mathbb{R}^d
$$
and cannot detect, represent, or model spatial transport or advective delay phenomena.

*Proof.*
From Theorem 3.5.1:
$$
\gamma_{ij}(\mathbf{h}) = C_{ij}(\mathbf{0}) - \frac{1}{2}\left(C_{ij}(\mathbf{h}) + C_{ij}(-\mathbf{h})\right)
$$
By Theorem 3.6.1, $C_{ij}^S(\mathbf{h}) = \frac{1}{2}(C_{ij}(\mathbf{h}) + C_{ij}(-\mathbf{h}))$. Thus:
$$
\gamma_{ij}(\mathbf{h}) = C_{ij}(\mathbf{0}) - C_{ij}^S(\mathbf{h})
$$
Evaluating at $-\mathbf{h}$:
$$
\gamma_{ij}(-\mathbf{h}) = C_{ij}(\mathbf{0}) - C_{ij}^S(-\mathbf{h}) = C_{ij}(\mathbf{0}) - C_{ij}^S(\mathbf{h}) = \gamma_{ij}(\mathbf{h})
$$
The odd skew-symmetric component $C_{ij}^A(\mathbf{h})$ cancels identically from the difference, leaving $\gamma_{ij}(\mathbf{h})$ strictly symmetric. $\blacksquare$

##### Proposition 3.6.2 (Advective Transport Characterization)
Let $Z_1(\mathbf{s})$ be a stationary process with covariance $C_1(\mathbf{h})$. Let $Z_2(\mathbf{s})$ be an advected version of $Z_1(\mathbf{s})$ displaced by a physical transport vector $\mathbf{v} \in \mathbb{R}^d$:
$$
Z_2(\mathbf{s}) = \alpha Z_1(\mathbf{s} - \mathbf{v}) + \eta(\mathbf{s})
$$
where $\alpha > 0$ and $\eta(\mathbf{s})$ is an independent noise field. Then:
1. The cross-covariance is shifted:
   $$
   C_{12}(\mathbf{h}) = \alpha C_1(\mathbf{h} - \mathbf{v})
   $$
   which achieves its maximum at $\mathbf{h} = \mathbf{v}$, not at $\mathbf{h} = \mathbf{0}$.
2. The pseudo-cross variogram captures the shift:
   $$
   \gamma_{12}^P(\mathbf{h}) = \frac{1}{2}(\sigma_1^2 + \sigma_2^2) - \alpha C_1(\mathbf{h} + \mathbf{v})
   $$
   reaching its minimum at $\mathbf{h} = -\mathbf{v}$.

*Proof.*
1. Direct evaluation:
   $$
   C_{12}(\mathbf{h}) = \operatorname{Cov}(Z_1(\mathbf{s}), Z_2(\mathbf{s}+\mathbf{h})) = \operatorname{Cov}(Z_1(\mathbf{s}), \alpha Z_1(\mathbf{s}+\mathbf{h}-\mathbf{v}) + \eta(\mathbf{s}+\mathbf{h})) = \alpha C_1(\mathbf{h} - \mathbf{v})
   $$
   Since $C_1(\cdot)$ is maximized at $\mathbf{0}$, $C_{12}(\mathbf{h})$ is maximized when $\mathbf{h} - \mathbf{v} = \mathbf{0} \implies \mathbf{h} = \mathbf{v}$.
2. By Theorem 3.5.2, $\gamma_{12}^P(\mathbf{h}) = \frac{1}{2}(\sigma_1^2 + \sigma_2^2) - C_{21}(\mathbf{h})$. By transpose symmetry, $C_{21}(\mathbf{h}) = C_{12}(-\mathbf{h}) = \alpha C_1(-\mathbf{h} - \mathbf{v}) = \alpha C_1(\mathbf{h} + \mathbf{v})$. Thus:
   $$
   \gamma_{12}^P(\mathbf{h}) = \frac{1}{2}(\sigma_1^2 + \sigma_2^2) - \alpha C_1(\mathbf{h} + \mathbf{v})
   $$
   which reaches its global minimum when $C_1(\mathbf{h} + \mathbf{v})$ is maximized, namely at $\mathbf{h} = -\mathbf{v}$. $\blacksquare$

---

#### 3.6.3 Computing Directional Cross-Covariances in Python

To detect spatial delay, we estimate the directional empirical cross-covariance function $\widehat{C}_{ij}(\mathbf{h})$ separately for positive displacements ($+\mathbf{h}$) and negative displacements ($-\mathbf{h}$), filtering pairs by angular tolerance $\Delta\theta$:

```python
import math

def compute_directional_cross_covariance(coords, values_i, values_j, 
                                         lag_centers, lag_bandwidth, 
                                         direction_vector, angular_tolerance_deg=30.0):
    """
    Computes directional cross-covariances C_ij(+h) and C_ij(-h) along vector u
    within an angular tolerance cone to detect spatial delay / advective shifts.
    
    Parameters:
        coords: list of lists, shape (n, d), shared sampling locations
        values_i: list of floats, shape (n,), measurements Z_i
        values_j: list of floats, shape (n,), measurements Z_j
        lag_centers: list of floats, shape (K,), scalar lag distances h_k
        lag_bandwidth: float, radial width Delta_h
        direction_vector: list of floats, shape (d,), orientation vector u
        angular_tolerance_deg: float, half-angle of directional cone in degrees
        
    Returns:
        signed_lags: list of floats of length 2K, signed lags [-h_K, ..., +h_K]
        C_ij: list of floats of length 2K, directional cross-covariances
    """
    n = len(coords)
    d = len(coords[0])
    K = len(lag_centers)
    norm_u = math.sqrt(sum(x * x for x in direction_vector))
    u = [x / norm_u for x in direction_vector]
    half_bw = lag_bandwidth / 2.0
    cos_tol = math.cos(math.radians(angular_tolerance_deg))
    
    # Center variables by sample means
    mean_i = sum(values_i) / n
    mean_j = sum(values_j) / n
    z_i_c = [val - mean_i for val in values_i]
    z_j_c = [val - mean_j for val in values_j]
    
    pos_sum = [0.0] * K
    pos_count = [0] * K
    neg_sum = [0.0] * K
    neg_count = [0] * K
    
    # Directed pair traversal (ordered pairs r, l)
    for r in range(n):
        for l in range(n):
            if r == l:
                continue
            h_vec = [coords[l][m] - coords[r][m] for m in range(d)]
            dist = math.sqrt(sum(x * x for x in h_vec))
            if dist < 1e-9:
                continue
                
            dot = sum(h_vec[m] * u[m] for m in range(d))
            cos_theta = dot / dist
            
            # Check angular alignment with direction vector +/- u
            for k in range(K):
                h_k = lag_centers[k]
                if (h_k - half_bw) <= dist < (h_k + half_bw):
                    prod = z_i_c[r] * z_j_c[l]
                    if cos_theta >= cos_tol:
                        # s_l is downstream of s_r (+h)
                        pos_sum[k] += prod
                        pos_count[k] += 1
                    elif cos_theta <= -cos_tol:
                        # s_l is upstream of s_r (-h)
                        neg_sum[k] += prod
                        neg_count[k] += 1
                    break
                    
    # Assemble signed curve: negative lags followed by positive lags
    signed_lags = [-h for h in reversed(lag_centers)] + list(lag_centers)
    C_ij = [None] * (2 * K)
    
    for k in range(K):
        if neg_count[k] > 0:
            C_ij[K - 1 - k] = neg_sum[k] / neg_count[k]
        if pos_count[k] > 0:
            C_ij[K + k] = pos_sum[k] / pos_count[k]
            
    return signed_lags, C_ij
```

---

#### 3.6.4 What Can Go Wrong: Blindness of Cross-Variograms and Downwind Symmetrization

```
Cross-Covariance Asymmetry under Downwind Advection:
Cross-Covariance C_12(h)
 ^
 |                                  Apex at h = +v (Transport Delay)
 |                                       ***
 |                                    *       *
 |                                  *           *
 |                    *            *             *
 |               *               *                 *
 |           *                 *                     *
 0 ────────*─────────────────*─────────────────────────*───────> Signed Lag h along flow
        -v                   0                         +v
```

1. **The Physical Reality of Delay:**
   Consider an industrial smokestack emitting atmospheric sulfur dioxide ($Z_1$) and ground-level soil acidification measured across the surrounding region ($Z_2$). Due to prevailing westerly winds with transport velocity $\mathbf{v}$, maximum deposition occurs several kilometers downwind ($\mathbf{h} = \mathbf{v}$). The cross-covariance $C_{12}(\mathbf{h})$ is strongly asymmetric: $C_{12}(\mathbf{v}) \gg C_{12}(-\mathbf{v})$.
2. **The Peril of Classical Cross-Variograms in Cokriging:**
   Because $\gamma_{12}(\mathbf{h}) = \gamma_{12}(-\mathbf{h})$ forcibly averages $C_{12}(\mathbf{h})$ and $C_{12}(-\mathbf{h})$, using the classical cross-variogram in an advective system completely obliterates the transport direction. The resulting cokriging system predicts symmetrical dispersion upwind and downwind, defying physical reality. For transport phenomena, one must model the full asymmetric cross-covariance $C_{ij}(\mathbf{h})$ or the pseudo-cross variogram $\gamma_{ij}^P(\mathbf{h})$.

---

### Key Ideas and Takeaways: Nonparametric Estimation

| Estimator / Concept | Mathematical Form | Primary Target | Key Strengths | Vulnerabilities / Failure Modes | Ideal Data Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Matheron Empirical Semivariogram** | $\frac{1}{2\|N\|}\sum (Z_i - Z_j)^2$ | Direct variogram $\gamma(\mathbf{h})$ | Unbiased under intrinsic stationarity; completely filters constant mean $\mu$; optimal efficiency under Gaussianity. | Catastrophic sensitivity to outliers ($\chi_1^2$ kurtosis $= 15$); positive bias under spatial drift. | Clean Gaussian data without outliers or spatial trends; isotropic networks. |
| **Sample Covariance** | $\frac{1}{\|N\|}\sum (Z_i - \bar{Z})(Z_j - \bar{Z})$ | Covariance kernel $C(\mathbf{h})$ | Positive semi-definite if unlagged; direct physical correlation interpretation. | Severe negative bias $\approx -\operatorname{Var}(\bar{Z})$; sagging baseline; false negative correlations at long lags. | Large domains ($L \gg \text{range}$) where spatial variance inflation is negligible. |
| **Cressie-Hawkins Modulus** | $\frac{(\frac{1}{\|N\|}\sum \|D\|^{1/2})^4}{2(0.457 + 0.494/\|N\|)}$ | Robust variogram $\gamma(\mathbf{h})$ | Suppresses isolated outliers via fourth-root compression; near-Gaussian increment distribution. | Dependent-pair under-correction overestimates $\gamma$ by $5\text{--}10\%$; underestimates under genuine heavy tails. | Datasets contaminated by isolated measurement errors or recording spikes ($< 15\%$). |
| **Clark Pseudo-Cross Variogram** | $\frac{1}{2\|N_{ij}\|}\sum (Z_i - Z_j)^2$ | Cross-spatial dispersion $\gamma_{ij}^P(\mathbf{h})$ | Functions on totally heterotopic networks ($S_i \cap S_j = \emptyset$); captures directional delay. | Strictly non-zero intercept at $\mathbf{h} = \mathbf{0}$; meaningless unless variables are standardized to unit variance. | Heterotopic multi-attribute monitoring (borehole assays paired with dense aerial geophysics). |
| **Directional Cross-Covariance** | $\frac{1}{\|N\|}\sum (Z_i - \bar{Z}_i)(Z_j - \bar{Z}_j)$ along $\pm \mathbf{u}$ | Asymmetric cross-covariance $C_{ij}(\mathbf{h})$ | Preserves advective transport shifts $C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})$; detects wind/water flow delay. | Requires paired measurements (isotopic or near-co-located); sensitive to sample mean estimation. | Physical advective systems (hydrology, plume dispersion, meteorology). |

---

### 3.7 Analytical Exercises and Step-by-Step Solutions

#### Exercise 3.1: Covariance of Spatial Increments
Let $\{Z(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ be a Gaussian random field with semivariogram $\gamma(\mathbf{h})$. Consider three collinear locations $\mathbf{s}_1, \mathbf{s}_2, \mathbf{s}_3 \in \mathbb{R}^d$ such that $\mathbf{s}_2 - \mathbf{s}_1 = \mathbf{h}$ and $\mathbf{s}_3 - \mathbf{s}_2 = \mathbf{h}$. Define the two adjacent spatial increments sharing the pivot point $\mathbf{s}_2$:
$$
D_1 = Z(\mathbf{s}_1) - Z(\mathbf{s}_2), \quad D_2 = Z(\mathbf{s}_2) - Z(\mathbf{s}_3)
$$
- **(a)** Derive $\operatorname{Cov}(D_1, D_2)$ purely in terms of semivariogram values $\gamma(\mathbf{h})$ and $\gamma(2\mathbf{h})$.
- **(b)** Derive the covariance of their squared increments $\operatorname{Cov}(D_1^2, D_2^2)$.
- **(c)** For an Exponential semivariogram with sill $\sigma^2$ and range parameter $\phi$, $\gamma(\mathbf{h}) = \sigma^2 (1 - e^{-\|\mathbf{h}\|/\phi})$, evaluate $\operatorname{Cov}(D_1, D_2)$ analytically and determine its sign.

##### Solution:
- **(a)** Using the bilinearity of covariance and $C(\mathbf{u}) = C(\mathbf{0}) - \gamma(\mathbf{u})$:
  $$
  \operatorname{Cov}(D_1, D_2) = \operatorname{Cov}(Z(\mathbf{s}_1) - Z(\mathbf{s}_2), Z(\mathbf{s}_2) - Z(\mathbf{s}_3))
  $$
  $$
  = \operatorname{Cov}(Z(\mathbf{s}_1), Z(\mathbf{s}_2)) - \operatorname{Cov}(Z(\mathbf{s}_1), Z(\mathbf{s}_3)) - \operatorname{Var}(Z(\mathbf{s}_2)) + \operatorname{Cov}(Z(\mathbf{s}_2), Z(\mathbf{s}_3))
  $$
  $$
  = [C(\mathbf{0}) - \gamma(\mathbf{h})] - [C(\mathbf{0}) - \gamma(2\mathbf{h})] - C(\mathbf{0}) + [C(\mathbf{0}) - \gamma(\mathbf{h})]
  $$
  The $C(\mathbf{0})$ terms cancel identically:
  $$
  C(\mathbf{0}) - C(\mathbf{0}) - C(\mathbf{0}) + C(\mathbf{0}) = 0
  $$
  leaving:
  $$
  \operatorname{Cov}(D_1, D_2) = \gamma(2\mathbf{h}) - 2\gamma(\mathbf{h})
  $$
- **(b)** Under Gaussianity, $(D_1, D_2)'$ is zero-mean bivariate Gaussian. Applying Lemma 3.3.1:
  $$
  \operatorname{Cov}(D_1^2, D_2^2) = 2\left(\operatorname{Cov}(D_1, D_2)\right)^2 = 2\left[\gamma(2\mathbf{h}) - 2\gamma(\mathbf{h})\right]^2
  $$
- **(c)** Substitute the Exponential model $\gamma(\mathbf{h}) = \sigma^2(1 - e^{-h/\phi})$ where $h = \|\mathbf{h}\|$:
  $$
  \gamma(2\mathbf{h}) - 2\gamma(\mathbf{h}) = \sigma^2(1 - e^{-2h/\phi}) - 2\sigma^2(1 - e^{-h/\phi})
  $$
  $$
  = \sigma^2\left[ 1 - e^{-2h/\phi} - 2 + 2e^{-h/\phi} \right] = \sigma^2\left[ -1 + 2e^{-h/\phi} - e^{-2h/\phi} \right]
  $$
  Factoring the quadratic in $e^{-h/\phi}$:
  $$
  \operatorname{Cov}(D_1, D_2) = -\sigma^2 \left(1 - e^{-h/\phi}\right)^2
  $$
  Since $\sigma^2 > 0$ and $(1 - e^{-h/\phi})^2 > 0$ for all $h > 0$:
  $$
  \operatorname{Cov}(D_1, D_2) < 0
  $$
  Adjacent spatial increments sharing an intermediate anchor point are strictly negatively correlated. $\blacksquare$

---

#### Exercise 3.2: Spatial Variance Inflation and Downward Bias on a 1D Regular Lattice
Let $\{Z(s) : s \in \mathbb{R}\}$ be a 1D stationary Gaussian process observed at regular integer grid points $s_k = k$ for $k = 1, \dots, n$. The true covariance is $C(h) = \sigma^2 \rho^{|h|}$ with $|\rho| < 1$.
- **(a)** Derive the exact analytical formula for $\operatorname{Var}(\bar{Z})$ as a function of $n, \sigma^2$, and $\rho$.
- **(b)** Compute $\lim_{n \to \infty} n \operatorname{Var}(\bar{Z})$ and evaluate the Variance Inflation Factor ($\mathrm{VIF}$) relative to independent observations.
- **(c)** Calculate the exact asymptotic bias of the sample covariance estimator $\widehat{C}(1)$ as $n \to \infty$, noting when the $\mathcal{O}(n^{-2})$ term is negligible relative to the $\mathrm{VIF}$ bias term.

##### Solution:
- **(a)** The sample mean variance is:
  $$
  \operatorname{Var}(\bar{Z}) = \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n C(i - j) = \frac{\sigma^2}{n^2} \sum_{i=1}^n \sum_{j=1}^n \rho^{|i - j|}
  $$
  Group pairs by separation lag $k = |i - j|$:
  $$
  \sum_{i=1}^n \sum_{j=1}^n \rho^{|i - j|} = n + 2\sum_{k=1}^{n-1} (n - k)\rho^k
  $$
  Using geometric series identities:
  $$
  \sum_{k=1}^{n-1} \rho^k = \frac{\rho(1 - \rho^{n-1})}{1 - \rho}, \quad \sum_{k=1}^{n-1} k\rho^k = \frac{\rho(1 - n\rho^{n-1} + (n-1)\rho^n)}{(1 - \rho)^2}
  $$
  Combining these sums:
  $$
  n\sum_{k=1}^{n-1} \rho^k - \sum_{k=1}^{n-1} k\rho^k = \frac{n\rho(1 - \rho) - \rho(1 - \rho^n)}{(1 - \rho)^2}
  $$
  Multiplying by 2 and adding $n$:
  $$
  n + 2\left[ \frac{n\rho(1 - \rho) - \rho(1 - \rho^n)}{(1 - \rho)^2} \right] = n\left(\frac{1 + \rho}{1 - \rho}\right) - \frac{2\rho(1 - \rho^n)}{(1 - \rho)^2}
  $$
  Dividing by $n^2$:
  $$
  \operatorname{Var}(\bar{Z}) = \frac{\sigma^2}{n}\left(\frac{1 + \rho}{1 - \rho}\right) - \frac{2\sigma^2 \rho(1 - \rho^n)}{n^2 (1 - \rho)^2}
  $$
- **(b)** Multiplying by $n$ and taking $n \to \infty$:
  $$
  \lim_{n \to \infty} n \operatorname{Var}(\bar{Z}) = \sigma^2 \left(\frac{1 + \rho}{1 - \rho}\right)
  $$
  The Variance Inflation Factor relative to the i.i.d. variance ($\sigma^2/n$) is:
  $$
  \mathrm{VIF} = \frac{1 + \rho}{1 - \rho}
  $$
  For $\rho = 0.8$, $\mathrm{VIF} = \frac{1.8}{0.2} = 9$. The effective sample size is reduced by a factor of 9 ($n_{\text{eff}} \approx n / 9$).
- **(c)** By Theorem 3.2.2, under increasing-domain asymptotics:
  $$
  \operatorname{Bias}(\widehat{C}(1)) = \operatorname{E}[\widehat{C}(1)] - C(1) = -\operatorname{Var}(\bar{Z}) + \mathcal{O}(n^{-2}) = -\frac{\sigma^2}{n}\left(\frac{1 + \rho}{1 - \rho}\right) + \mathcal{O}(n^{-2})
  $$
  The leading negative bias is of order $\mathcal{O}(n^{-1})$ and proportional to $\mathrm{VIF}$. Whenever $n > 2\rho / (1 - \rho^2)$ (for $\rho = 0.8$, $n > 4.4$), the $\mathcal{O}(n^{-2})$ term is less than $10\%$ of the primary bias, making $-\frac{\sigma^2}{n}\mathrm{VIF}$ the dominant downward shift. $\blacksquare$

---

#### Exercise 3.3: Fractional Moments under Heavy-Tailed Student-$t$ Increments
Suppose the spatial increment $D = Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})$ follows a Student-$t$ distribution with $\nu$ degrees of freedom and scale parameter $\tau$:
$$
f_D(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi \nu}\tau \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu \tau^2}\right)^{-\frac{\nu+1}{2}}
$$
- **(a)** For $\nu > 2$, compute the true semivariogram value $\gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(D)$.
- **(b)** Derive the analytical expectation $\operatorname{E}[|D|^{1/2}]$ for $\nu > 1/2$.
- **(c)** For $\nu = 5$, evaluate the ratio $\frac{(\operatorname{E}[|D|^{1/2}])^4}{2\gamma(\mathbf{h})}$ and compare it with the Gaussian value $B \approx 0.457066$. What does this reveal about the price of robustness?

##### Solution:
- **(a)** For a Student-$t$ distribution with scale $\tau$ and $\nu > 2$, $\operatorname{Var}(D) = \frac{\nu \tau^2}{\nu - 2}$. Therefore:
  $$
  \gamma(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(D) = \frac{\nu \tau^2}{2(\nu - 2)}
  $$
- **(b)** By symmetry of the Student-$t$ density:
  $$
  \operatorname{E}[|D|^{1/2}] = 2\int_0^\infty x^{1/2} \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi \nu}\tau \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu \tau^2}\right)^{-\frac{\nu+1}{2}} dx
  $$
  Substitute $u = \frac{x^2}{\nu \tau^2}$, so $x = \tau \sqrt{\nu} u^{1/2}$ and $dx = \frac{\tau \sqrt{\nu}}{2} u^{-1/2} du$:
  $$
  \operatorname{E}[|D|^{1/2}] = \frac{\tau^{1/2}\nu^{1/4}\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi}\Gamma\left(\frac{\nu}{2}\right)} \int_0^\infty u^{-1/4}(1 + u)^{-\frac{\nu+1}{2}} du
  $$
  Using the Beta integral $\int_0^\infty u^{a-1}(1+u)^{-(a+b)} du = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$ with $a = 3/4$ and $b = \frac{2\nu - 1}{4}$:
  $$
  \operatorname{E}[|D|^{1/2}] = \tau^{1/2}\nu^{1/4} \frac{\Gamma(3/4)\Gamma\left(\frac{2\nu - 1}{4}\right)}{\sqrt{\pi}\Gamma\left(\frac{\nu}{2}\right)}
  $$
  valid for $b > 0 \iff \nu > 1/2$.
- **(c)** For $\nu = 5$:
  From (a): $2\gamma(\mathbf{h}) = \frac{5\tau^2}{3} \approx 1.66667 \tau^2$.
  From (b): with $\frac{2\nu - 1}{4} = \frac{9}{4} = 2.25$ and $\nu/2 = 2.5$:
  $$
  \operatorname{E}[|D|^{1/2}] = \tau^{1/2} 5^{1/4} \frac{\Gamma(0.75)\Gamma(2.25)}{\sqrt{\pi}\Gamma(2.5)} \approx 0.8811 \tau^{1/2}
  $$
  Raising to the fourth power: $(\operatorname{E}[|D|^{1/2}])^4 \approx (0.8811)^4 \tau^2 \approx 0.6027 \tau^2$.
  Computing the ratio:
  $$
  \frac{(\operatorname{E}[|D|^{1/2}])^4}{2\gamma(\mathbf{h})} = \frac{0.6027 \tau^2}{1.6667 \tau^2} \approx 0.3616
  $$
  The ratio $0.3616$ is roughly $21\%$ lower than the Gaussian constant $B \approx 0.457066$.
  
  **The Price of Robustness:** The Cressie-Hawkins denominator $2 \times 0.457$ assumes the nominal Gaussian distribution. When applied to an inherently heavy-tailed Student-$t$ process ($\nu = 5$), Cressie-Hawkins systematically underestimates the true variogram by over $20\%$. Robust estimators protect against isolated outliers under a nominal Gaussian model, but if the underlying data-generating process is genuinely heavy-tailed throughout, the calibration constant must be recalibrated. $\blacksquare$

---

#### Exercise 3.4: Clark's Pseudo-Cross Variogram Symmetry and Transpose Relations
Prove that if a bivariate stationary process $(Z_1(\mathbf{s}), Z_2(\mathbf{s}))'$ has symmetric cross-covariance ($C_{12}(\mathbf{h}) = C_{12}(-\mathbf{h})$ for all $\mathbf{h} \in \mathbb{R}^d$), then Clark's pseudo-cross variogram is strictly symmetric:
$$
\gamma_{12}^P(\mathbf{h}) = \gamma_{12}^P(-\mathbf{h}) = \gamma_{21}^P(\mathbf{h})
$$
Conversely, prove that if $\gamma_{12}^P(\mathbf{h}) \ne \gamma_{12}^P(-\mathbf{h})$, the underlying cross-covariance must be asymmetric.

##### Solution:
By Theorem 3.5.2:
$$
\gamma_{12}^P(\mathbf{h}) = \frac{1}{2}(\sigma_1^2 + \sigma_2^2) - C_{21}(\mathbf{h})
$$
By transpose symmetry (Proposition 3.6.1), $C_{21}(\mathbf{h}) = C_{12}(-\mathbf{h})$. Thus:
$$
\gamma_{12}^P(\mathbf{h}) = \frac{1}{2}(\sigma_1^2 + \sigma_2^2) - C_{12}(-\mathbf{h})
$$
Evaluating at $-\mathbf{h}$:
$$
\gamma_{12}^P(-\mathbf{h}) = \frac{1}{2}(\sigma_1^2 + \sigma_2^2) - C_{12}(\mathbf{h})
$$
Subtracting the two equations:
$$
\gamma_{12}^P(\mathbf{h}) - \gamma_{12}^P(-\mathbf{h}) = C_{12}(\mathbf{h}) - C_{12}(-\mathbf{h})
$$
1. If $C_{12}(\mathbf{h})$ is symmetric ($C_{12}(\mathbf{h}) = C_{12}(-\mathbf{h})$), the right-hand side is zero, so:
   $$
   \gamma_{12}^P(\mathbf{h}) = \gamma_{12}^P(-\mathbf{h})
   $$
   Furthermore:
   $$
   \gamma_{21}^P(\mathbf{h}) = \frac{1}{2}(\sigma_2^2 + \sigma_1^2) - C_{12}(\mathbf{h}) = \gamma_{12}^P(-\mathbf{h}) = \gamma_{12}^P(\mathbf{h})
   $$
2. Conversely, if $\gamma_{12}^P(\mathbf{h}) \ne \gamma_{12}^P(-\mathbf{h})$, then:
   $$
   C_{12}(\mathbf{h}) - C_{12}(-\mathbf{h}) = \gamma_{12}^P(\mathbf{h}) - \gamma_{12}^P(-\mathbf{h}) \ne 0 \implies C_{12}(\mathbf{h}) \ne C_{12}(-\mathbf{h})
   $$
   The cross-covariance must be asymmetric. $\blacksquare$

---

#### Exercise 3.5: Pencil-and-Paper Computation of an Empirical Variogram on 5 Locations
Consider five observation points positioned along a 1D line at coordinates $s_1 = 0, s_2 = 1, s_3 = 2, s_4 = 4, s_5 = 5$. The observed values are:
$$
Z(s_1) = 10, \quad Z(s_2) = 12, \quad Z(s_3) = 11, \quad Z(s_4) = 15, \quad Z(s_5) = 14
$$
We define two lag bins with tolerance $\Delta h = 1.0$:
- **Bin 1:** $h_1 = 1.0$, covering separation distances $0.5 \le d < 1.5$.
- **Bin 2:** $h_2 = 2.0$, covering separation distances $1.5 \le d < 2.5$.
- **(a)** Construct the complete $5 \times 5$ pairwise distance matrix and the pairwise squared difference matrix.
- **(b)** Identify the pairs $(i, j)$ belonging to Bin 1 and Bin 2, and compute the pair counts $|N(h_1)|$ and $|N(h_2)|$.
- **(c)** Calculate Matheron's empirical semivariance $\widehat{\gamma}(h_1)$ and $\widehat{\gamma}(h_2)$ by hand.
- **(d)** Calculate the Cressie-Hawkins robust semivariance $\bar{\gamma}_{\text{CH}}(h_1)$ for Bin 1 and compare it with $\widehat{\gamma}(h_1)$.

##### Solution:
- **(a)** The sample size is $n = 5$, giving $\binom{5}{2} = 10$ unique unordered pairs $(i < j)$.
  
  Evaluating pairwise distances $d_{ij} = |s_i - s_j|$ and squared differences $(Z_i - Z_j)^2$:
  - Pair $(1, 2)$: $d_{12} = |0 - 1| = 1$, $(10 - 12)^2 = (-2)^2 = 4$
  - Pair $(1, 3)$: $d_{13} = |0 - 2| = 2$, $(10 - 11)^2 = (-1)^2 = 1$
  - Pair $(1, 4)$: $d_{14} = |0 - 4| = 4$, $(10 - 15)^2 = (-5)^2 = 25$
  - Pair $(1, 5)$: $d_{15} = |0 - 5| = 5$, $(10 - 14)^2 = (-4)^2 = 16$
  - Pair $(2, 3)$: $d_{23} = |1 - 2| = 1$, $(12 - 11)^2 = (1)^2 = 1$
  - Pair $(2, 4)$: $d_{24} = |1 - 4| = 3$, $(12 - 15)^2 = (-3)^2 = 9$
  - Pair $(2, 5)$: $d_{25} = |1 - 5| = 4$, $(12 - 14)^2 = (-2)^2 = 4$
  - Pair $(3, 4)$: $d_{34} = |2 - 4| = 2$, $(11 - 15)^2 = (-4)^2 = 16$
  - Pair $(3, 5)$: $d_{35} = |2 - 5| = 3$, $(11 - 14)^2 = (-3)^2 = 9$
  - Pair $(4, 5)$: $d_{45} = |4 - 5| = 1$, $(15 - 14)^2 = (1)^2 = 1$

- **(b)** Bin assignment:
  - **Bin 1 ($0.5 \le d < 1.5$):** Pairs with distance $d = 1$:
    $$
    N(h_1) = \{(1, 2), (2, 3), (4, 5)\}, \quad |N(h_1)| = 3
    $$
  - **Bin 2 ($1.5 \le d < 2.5$):** Pairs with distance $d = 2$:
    $$
    N(h_2) = \{(1, 3), (3, 4)\}, \quad |N(h_2)| = 2
    $$

- **(c)** Matheron's semivariance:
  - For Bin 1:
    $$
    \widehat{\gamma}(h_1) = \frac{1}{2|N(h_1)|}\sum_{(i,j) \in N(h_1)} (Z_i - Z_j)^2 = \frac{1}{2 \times 3}(4 + 1 + 1) = \frac{6}{6} = 1.0
    $$
  - For Bin 2:
    $$
    \widehat{\gamma}(h_2) = \frac{1}{2|N(h_2)|}\sum_{(i,j) \in N(h_2)} (Z_i - Z_j)^2 = \frac{1}{2 \times 2}(1 + 16) = \frac{17}{4} = 4.25
    $$

- **(d)** Cressie-Hawkins semivariance for Bin 1 ($N = 3$):
  The absolute differences are $|10 - 12| = 2$, $|12 - 11| = 1$, $|15 - 14| = 1$.
  The square roots are:
  $$
  \sqrt{2} \approx 1.41421, \quad \sqrt{1} = 1.0, \quad \sqrt{1} = 1.0
  $$
  Sample mean of square-root increments:
  $$
  \bar{Y} = \frac{1.41421 + 1.0 + 1.0}{3} = \frac{3.41421}{3} \approx 1.13807
  $$
  Fourth power:
  $$
  \bar{Y}^4 = (1.13807)^4 \approx 1.6775
  $$
  Denominator with $N = 3$:
  $$
  2\left(0.457066 + \frac{0.49457}{3}\right) = 2(0.457066 + 0.164857) = 2(0.621923) \approx 1.24385
  $$
  Evaluating $\bar{\gamma}_{\text{CH}}(h_1)$:
  $$
  \bar{\gamma}_{\text{CH}}(h_1) = \frac{1.6775}{1.24385} \approx 1.3486 \approx 1.35
  $$
  Comparison: $\bar{\gamma}_{\text{CH}}(h_1) \approx 1.35$ while $\widehat{\gamma}(h_1) = 1.0$. Because $N = 3$ is very small, finite-sample Jensen effects and the lack of outliers make the adjusted estimator yield a slightly higher estimate, illustrating the sensitivity of the correction factor when sample sizes are small ($N < 30$). $\blacksquare$

---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper matrix mechanics, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 3: Theory of Semivariogram and Covariance Estimation — Companion Exercises and Complete Solutions](ch03_exercises.md)
