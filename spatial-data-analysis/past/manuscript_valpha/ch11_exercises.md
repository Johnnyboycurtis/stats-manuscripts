# Chapter 11 Companion Exercises: Mapped Point Patterns and Spatial Point Processes

This study guide and exercise workbook accompanies [Chapter 11: Mapped Point Patterns and Spatial Point Processes](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch11_point_processes.md). Synthesizing Maria L. Rizzo's three-beat pedagogical rhythm (*Statistical Computing with R*) with Oliver Schabenberger and Carol A. Gotway's semantic linear-model discipline (*Statistical Methods for Spatial Data Analysis*), these exercises bridge classical probability and spatial statistics across five comprehensive tiers:

1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 11.1–11.4)
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 11.5–11.8)
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 11.9–11.11)
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 11.12–11.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 11.14–11.16)

Every exercise includes an unabridged, step-by-step mathematical derivation, analytical proof, or dependency-free computational solution.

---

## Tier 0: Classical-to-Spatial Warm-Ups

### Exercise 11.1: From the 1D Renewal Timeline to the 2D Spatial Counting Measure
In classical queueing theory and reliability engineering, a homogeneous Poisson process $\{N(t) : t \ge 0\}$ on the real line $[0, \infty)$ is characterized by a constant arrival rate $\lambda > 0$. The number of events occurring in any interval $(t_1, t_2]$ follows $N(t_2) - N(t_1) \sim \operatorname{Poisson}(\lambda (t_2 - t_1))$, and the inter-arrival times $T_k = t_k - t_{k-1}$ are independent and identically distributed as $T_k \sim \operatorname{Exponential}(\lambda)$.

**(a)** Explain why the renewal representation based on sequential inter-arrival times $T_k$ fails completely when extending the Poisson process from the 1D line $\mathbb{R}^1$ to a spatial planar window $W \subset \mathbb{R}^2$. Specifically, why is there no canonical total ordering on $\mathbb{R}^2$?

**(b)** State the axiomatic definition of a spatial counting measure $N(A)$ for bounded Borel sets $A \subset \mathbb{R}^d$. Show how the single parameter $\lambda$ (rate per unit time) generalizes to the spatial intensity parameter $\lambda$ (expected events per unit area), expressing $\operatorname{E}[N(A)]$ as a Lebesgue integral.

**(c)** Let $W = [0, 2] \times [0, 3] \subset \mathbb{R}^2$ be a rectangular planar observation window with constant intensity $\lambda = 4.5\text{ events/unit area}$. Compute:
1. The expected total number of events in $W$, $\mu = \operatorname{E}[N(W)]$.
2. The probability that exactly zero events fall within the circular disk $B(\mathbf{s}_0, r)$ centered at $\mathbf{s}_0 = (1, 1)'$ with radius $r = 0.4$.
3. The probability that the number of events in $W$ is strictly greater than 30.

##### Solution to Exercise 11.1
**(a)** On the 1D real line $\mathbb{R}^1$, points have a natural, unique, and strict total ordering: for any two distinct points $t_1, t_2 \in \mathbb{R}$, either $t_1 < t_2$ or $t_2 < t_1$. This permits the construction of a renewal process via cumulative sums of positive random variables:
$$
t_k = \sum_{j=1}^k T_j, \quad T_j > 0
$$
In $\mathbb{R}^2$, however, points are coordinate vectors $\mathbf{s} = (x, y)'$. The plane possesses no isotropic total ordering that preserves Euclidean geometry. Any arbitrary linear order imposed on $\mathbb{R}^2$ (such as lexicographic ordering: $(x_1, y_1) < (x_2, y_2)$ if $x_1 < x_2$ or ($x_1 = x_2$ and $y_1 < y_2$)) violates spatial isotropy and breaks the physical property of spatial neighborhoods. Consequently, spatial point processes cannot be defined through inter-event distances along an ordered chain; they must instead be defined set-theoretically via random counting measures $N(A)$ operating on geometric subsets $A \subset \mathbb{R}^d$.

**(b)** A spatial counting measure $N$ on $\mathbb{R}^d$ is a mapping from the Borel $\sigma$-algebra $\mathcal{B}(\mathbb{R}^d)$ to the set of non-negative integers $\mathbb{N}_0 \cup \{\infty\}$ satisfying:
1. $N(\emptyset) = 0$.
2. Countable additivity: for any sequence of mutually disjoint Borel sets $\{A_k\}_{k=1}^\infty$,
$$
N\left(\bigcup_{k=1}^\infty A_k\right) = \sum_{k=1}^\infty N(A_k)
$$
3. Locally finite: $N(A) < \infty$ almost surely for every bounded Borel set $A$.

The first-order intensity function $\lambda(\mathbf{s})$ is the Radon-Nikodym derivative of the intensity measure $\Lambda(A) = \operatorname{E}[N(A)]$ with respect to $d$-dimensional Lebesgue measure $\mathrm{d}\mathbf{s}$:
$$
\Lambda(A) = \operatorname{E}[N(A)] = \int_A \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
$$
When the process is homogeneous (stationary and isotropic), $\lambda(\mathbf{s}) \equiv \lambda$ is constant, yielding:
$$
\operatorname{E}[N(A)] = \lambda \int_A \mathrm{d}\mathbf{s} = \lambda |A|
$$
where $|A|$ denotes the Lebesgue measure (area in $\mathbb{R}^2$, volume in $\mathbb{R}^3$) of $A$.

**(c)** 
1. The area of the rectangular window $W = [0, 2] \times [0, 3]$ is $|W| = 2 \times 3 = 6.0$. Thus:
$$
\mu = \operatorname{E}[N(W)] = \lambda |W| = 4.5 \times 6.0 = 27.0
$$
2. The area of the circular disk $B(\mathbf{s}_0, r)$ with radius $r = 0.4$ is:
$$
|B| = \pi r^2 = \pi (0.4)^2 = 0.16\pi \approx 0.502655
$$
The expected count in the disk is $\mu_B = \lambda |B| = 4.5 \times 0.16\pi = 0.72\pi \approx 2.2619$.
Since $N(B) \sim \operatorname{Poisson}(\mu_B)$, the probability of zero events is:
$$
\operatorname{P}(N(B) = 0) = \frac{\mu_B^0 e^{-\mu_B}}{0!} = e^{-0.72\pi} \approx e^{-2.2619} \approx 0.10415
$$
3. Since $N(W) \sim \operatorname{Poisson}(27.0)$, the probability of observing more than 30 events is:
$$
\operatorname{P}(N(W) > 30) = 1 - \operatorname{P}(N(W) \le 30) = 1 - \sum_{k=0}^{30} \frac{27^k e^{-27}}{k!}
$$
Using the normal approximation with continuity correction (since $\mu = 27 > 20$, with variance $\sigma^2 = \mu = 27$ and standard deviation $\sigma = \sqrt{27} \approx 5.1962$):
$$
Z = \frac{30.5 - 27.0}{\sqrt{27}} = \frac{3.5}{5.1962} \approx 0.6736
$$
$$
\operatorname{P}(N(W) \le 30) \approx \Phi(0.6736) \approx 0.7497 \implies \operatorname{P}(N(W) > 30) \approx 1 - 0.7497 = 0.2503
$$
(The exact Poisson cumulative sum gives $\operatorname{P}(N(W) \le 30) \approx 0.7548$, so $\operatorname{P}(N(W) > 30) \approx 0.2452$).

---

### Exercise 11.2: Binomial Lattice Approximation to Spatial Complete Randomness
In elementary statistics, the Poisson distribution arises as the law of small numbers: the limiting distribution of a Binomial experiment $B(M, p)$ as the number of independent trials $M \to \infty$ while $M p = \mu$ remains constant.

Consider a planar domain $W \subset \mathbb{R}^2$ of area $|W|$. Discretize $W$ into $M$ non-overlapping, congruent square cells $C_1, C_2, \dots, C_M$, each with area $|C_m| = |W| / M$. Suppose that in each cell $C_m$, an event occurs with probability $p_M = \lambda |W| / M$ independently of all other cells, and at most one event can occur per cell (Bernoulli lattice).

**(a)** Let $A \subset W$ be an arbitrary test region consisting of the union of $k_M$ grid cells, so that $|A| \approx k_M |W| / M$, with $k_M / M \to |A| / |W|$ as $M \to \infty$. Derive the limiting distribution of $N(A) = \sum_{m \in A} Y_m$ as $M \to \infty$, where $Y_m \in \{0, 1\}$.

**(b)** Show that for any two disjoint subregions $A, B \subset W$, $N(A)$ and $N(B)$ are asymptotically independent as $M \to \infty$.

**(c)** Connect this result to the dual definition of Complete Spatial Randomness (CSR): prove that conditioning on the total observed event count $N(W) = n$, the $n$ event locations $\{\mathbf{s}_1, \dots, \mathbf{s}_n\}$ are distributed as $n$ independent and identically distributed random vectors uniformly distributed across $W$.

##### Solution to Exercise 11.2
**(a)** The count of events in region $A$ is the sum of $k_M$ independent and identically distributed Bernoulli random variables $Y_m \sim \operatorname{Bernoulli}(p_M)$:
$$
N(A) = \sum_{m: C_m \subset A} Y_m \sim \operatorname{Binomial}(k_M, p_M)
$$
The probability generating function of $N(A)$ is:
$$
G_M(z) = \operatorname{E}[z^{N(A)}] = \left(1 - p_M + p_M z\right)^{k_M} = \left(1 + p_M (z - 1)\right)^{k_M}
$$
Substituting $p_M = \frac{\lambda |W|}{M}$ and $k_M = M \frac{|A|_M}{|W|}$:
$$
G_M(z) = \left(1 + \frac{\lambda |W| (z - 1)}{M}\right)^{M \frac{|A|_M}{|W|}}
$$
Taking the limit as $M \to \infty$, where $|A|_M \to |A|$:
$$
\lim_{M \to \infty} G_M(z) = \lim_{M \to \infty} \left[\left(1 + \frac{\lambda |A| (z - 1)}{k_M}\right)^{k_M}\right] = \exp\left(\lambda |A| (z - 1)\right)
$$
$\exp(\mu (z - 1))$ is the probability generating function of a Poisson random variable with parameter $\mu = \lambda |A|$. Therefore, $N(A) \xrightarrow{d} \operatorname{Poisson}(\lambda |A|)$.

**(b)** Let $A, B \subset W$ be disjoint Borel sets, partitioned into index sets $\mathcal{I}_A$ and $\mathcal{I}_B$ of cells with $\mathcal{I}_A \cap \mathcal{I}_B = \emptyset$. For any finite $M$:
$$
N(A) = \sum_{m \in \mathcal{I}_A} Y_m, \quad N(B) = \sum_{m \in \mathcal{I}_B} Y_m
$$
Because the Bernoulli trials $\{Y_m\}_{m=1}^M$ are mutually independent by definition of the lattice setup, and $\mathcal{I}_A \cap \mathcal{I}_B = \emptyset$, $N(A)$ and $N(B)$ are functions of disjoint sets of independent random variables. Thus, for every $M$, $N(A)$ and $N(B)$ are strictly independent. This independence is preserved in the limit as $M \to \infty$.

**(c)** Let $A \subset W$ be a subregion. Conditioning on the total event count in the observation window $N(W) = n$:
$$
\operatorname{P}(N(A) = k \mid N(W) = n) = \frac{\operatorname{P}(N(A) = k \text{ and } N(W) = n)}{\operatorname{P}(N(W) = n)}
$$
Because $A$ and $W \setminus A$ are disjoint and partition $W$, $N(W) = N(A) + N(W \setminus A)$, where $N(A)$ and $N(W \setminus A)$ are independent Poisson variables with means $\lambda |A|$ and $\lambda (|W| - |A|)$:
$$
\begin{aligned}
\operatorname{P}(N(A) = k \mid N(W) = n) &= \frac{\operatorname{P}(N(A) = k) \cdot \operatorname{P}(N(W \setminus A) = n - k)}{\operatorname{P}(N(W) = n)} \\
&= \frac{\frac{(\lambda |A|)^k e^{-\lambda |A|}}{k!} \cdot \frac{(\lambda (|W| - |A|))^{n-k} e^{-\lambda (|W| - |A|)}}{(n - k)!}}{\frac{(\lambda |W|)^n e^{-\lambda |W|}}{n!}} \\
&= \frac{n!}{k!(n - k)!} \cdot \frac{(\lambda |A|)^k (\lambda (|W| - |A|))^{n - k}}{(\lambda |W|)^n} \cdot \frac{e^{-\lambda |A|} e^{-\lambda (|W| - |A|)}}{e^{-\lambda |W|}} \\
&= \binom{n}{k} \left(\frac{|A|}{|W|}\right)^k \left(1 - \frac{|A|}{|W|}\right)^{n - k}
\end{aligned}
$$
This is the probability mass function of a $\operatorname{Binomial}(n, p)$ random variable with success probability $p = |A| / |W|$.
Since this holds for every measurable set $A \subset W$, it proves that conditioned on $N(W) = n$, each of the $n$ points independently falls in $A$ with probability $|A| / |W|$. This is precisely the distribution of $n$ independent and identically distributed points uniformly distributed over $W$:
$$
f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) = \prod_{i=1}^n \frac{1}{|W|} = \frac{1}{|W|^n}
$$

---

### Exercise 11.3: Classical Dispersion Ratio vs. Spatial Autocorrelation
In classical exploratory data analysis of count data $X_1, X_2, \dots, X_m$, the Index of Dispersion (Fisher's dispersion test) is defined as:
$$
\mathrm{ID} = \frac{S^2}{\bar{X}} = \frac{\frac{1}{m - 1}\sum_{j=1}^m (X_j - \bar{X})^2}{\frac{1}{m}\sum_{j=1}^m X_j}
$$
Under the classical i.i.d. Poisson assumption, $\operatorname{Var}(X) = \operatorname{E}[X]$, so $\mathrm{ID} \approx 1$.

**(a)** Suppose a spatial domain $W$ is divided into $m = 16$ quadrats. Show that under Complete Spatial Randomness, $(m - 1)\mathrm{ID} \sim \chi^2_{m - 1}$ asymptotically as $\lambda |C_j| \to \infty$.

**(b)** Now suppose the point process is NOT independent across space, but has positive spatial autocorrelation between neighboring quadrats, such that $\operatorname{Cov}(X_j, X_k) = \sigma^2 \rho_{jk} > 0$. Derive the expectation of the sample variance $S^2$ under this spatial dependency structure.

**(c)** Prove that positive spatial clustering inflates $S^2$, forcing $\operatorname{E}[\mathrm{ID}] > 1$, while spatial regular/inhibitory repulsion depresses $S^2$, forcing $\operatorname{E}[\mathrm{ID}] < 1$.

##### Solution to Exercise 11.3
**(a)** Under Complete Spatial Randomness, the counts in disjoint quadrats $C_1, \dots, C_m$ are independent and identically distributed Poisson random variables: $X_j \sim \operatorname{Poisson}(\mu)$, where $\mu = \lambda |C_j|$.
By the central limit theorem, when the expected count per quadrat $\mu$ is sufficiently large (typically $\mu \ge 5$), the standardized counts are approximately standard normal:
$$
Z_j = \frac{X_j - \mu}{\sqrt{\mu}} \approx \mathcal{N}(0, 1)
$$
The sum of squared standardized deviations from the true mean is:
$$
\sum_{j=1}^m Z_j^2 = \sum_{j=1}^m \frac{(X_j - \mu)^2}{\mu} \sim \chi^2_m
$$
Replacing the unknown parameter $\mu$ with the sample mean $\bar{X} = \frac{1}{m}\sum_{j=1}^m X_j$ loses 1 degree of freedom:
$$
\sum_{j=1}^m \frac{(X_j - \bar{X})^2}{\bar{X}} = \frac{(m - 1)S^2}{\bar{X}} = (m - 1)\mathrm{ID} \sim \chi^2_{m - 1}
$$

**(b)** Let $\operatorname{E}[X_j] = \mu$ and $\operatorname{Var}(X_j) = \sigma^2$ for all $j=1, \dots, m$. Let $\operatorname{Cov}(X_j, X_k) = \sigma^2 \rho_{jk}$ where $\rho_{jj} = 1$.
The sample variance is:
$$
S^2 = \frac{1}{m - 1}\sum_{j=1}^m (X_j - \bar{X})^2 = \frac{1}{m - 1}\left[\sum_{j=1}^m (X_j - \mu)^2 - m(\bar{X} - \mu)^2\right]
$$
Taking expectations:
$$
\operatorname{E}\left[\sum_{j=1}^m (X_j - \mu)^2\right] = \sum_{j=1}^m \operatorname{Var}(X_j) = m \sigma^2
$$
For the second term:
$$
\operatorname{E}\left[(\bar{X} - \mu)^2\right] = \operatorname{Var}(\bar{X}) = \frac{1}{m^2}\sum_{j=1}^m \sum_{k=1}^m \operatorname{Cov}(X_j, X_k) = \frac{\sigma^2}{m^2} \sum_{j=1}^m \sum_{k=1}^m \rho_{jk}
$$
Let $\bar{\rho} = \frac{1}{m(m - 1)}\sum_{j \ne k} \rho_{jk}$ denote the average off-diagonal spatial correlation. Then:
$$
\sum_{j=1}^m \sum_{k=1}^m \rho_{jk} = m + m(m - 1)\bar{\rho} \implies \operatorname{Var}(\bar{X}) = \frac{\sigma^2}{m}\left[1 + (m - 1)\bar{\rho}\right]
$$
Substituting these into the expectation of $S^2$:
$$
\begin{aligned}
\operatorname{E}[S^2] &= \frac{1}{m - 1}\left[m \sigma^2 - m \cdot \frac{\sigma^2}{m}(1 + (m - 1)\bar{\rho})\right] \\
&= \frac{1}{m - 1}\left[m \sigma^2 - \sigma^2 - (m - 1)\sigma^2 \bar{\rho}\right] \\
&= \frac{1}{m - 1}\left[(m - 1)\sigma^2 - (m - 1)\sigma^2 \bar{\rho}\right] \\
&= \sigma^2 (1 - \bar{\rho})
\end{aligned}
$$

**(c)** Reconciling Part (b) and Part (c) requires distinguishing between two distinct spatial scales:
1. *Within-Quadrat Scale (Population Variance Inflation):* Clustered point processes (such as Cox or Neyman-Scott processes) induce strong positive point-to-point correlation within individual quadrats. As shown via Campbell's Theorem, this inflates the true quadrat population variance far above the Poisson baseline:
   $$
   \sigma^2 = \operatorname{Var}(N(C_j)) = \mu + \mu^2 \int_{C_j}\int_{C_j} [g(\mathbf{u} - \mathbf{v}) - 1] \, d\mathbf{u} d\mathbf{v} > \mu
   $$
2. *Between-Quadrat Scale (Sample Covariance Bias):* Positive spatial autocorrelation between neighboring quadrats introduces a downward finite-sample estimation bias on the sample variance:
   $$
   \operatorname{E}[S^2] = \sigma^2 (1 - \bar{\rho})
   $$
Because within-quadrat clustering inflates the true variance $\sigma^2$ by an order of magnitude, the observed sample ratio $\operatorname{E}[\mathrm{ID}] = \frac{\operatorname{E}[S^2]}{\mu} = \frac{\sigma^2}{\mu}(1 - \bar{\rho})$ remains substantially greater than 1 despite the minor downward deflation $(1 - \bar{\rho})$ from inter-quadrat correlation.

Conversely, under inhibitory/regular spatial interaction (such as tree crown competition or territory defense), points maintain a minimum separation distance. Quadrats are filled much more evenly than predicted by chance, suppressing variability between quadrats:
$$
\sigma^2 < \mu \implies \operatorname{E}[\mathrm{ID}] < 1
$$
Thus, $\mathrm{ID} > 1$ diagnoses clustering/aggregation, while $\mathrm{ID} < 1$ diagnoses regular spacing.

---

### Exercise 11.4: Poisson Thinning and Lewis-Shedler Rejection Sampling
In classical simulation, the acceptance-rejection method transforms samples from an easy proposal distribution into samples from a complex target distribution. In point process theory, this is known as **independent thinning**.

Let $\Pi = \{\mathbf{s}_1, \dots, \mathbf{s}_N\}$ be a homogeneous Poisson process on $W \subset \mathbb{R}^2$ with constant proposal intensity $\lambda_{\max} > 0$.
Let $\lambda(\mathbf{s})$ be a continuous, non-negative target intensity function on $W$ such that $0 \le \lambda(\mathbf{s}) \le \lambda_{\max}$ for all $\mathbf{s} \in W$.
For each candidate point $\mathbf{s}_i \in \Pi$, generate an independent uniform random variable $U_i \sim \operatorname{Uniform}(0, 1)$. Retain point $\mathbf{s}_i$ if:
$$
U_i \le \frac{\lambda(\mathbf{s}_i)}{\lambda_{\max}}
$$
and reject (delete) it otherwise. Let $\Pi_{\text{thinned}}$ denote the set of retained points.

**(a)** Using probability generating functionals (PGFL) or elementary counting measure properties, prove that $\Pi_{\text{thinned}}$ is an inhomogeneous Poisson process with exact intensity function $\lambda(\mathbf{s})$.

**(b)** Suppose $W = [0, 1] \times [0, 1]$, and the target intensity is $\lambda(x, y) = 100 \exp(x + 2y)$.
1. Find the sharp upper bound $\lambda_{\max} = \sup_{\mathbf{s} \in W} \lambda(\mathbf{s})$.
2. Compute the theoretical acceptance probability $\alpha = \frac{\int_W \lambda(\mathbf{s})\,\mathrm{d}\mathbf{s}}{\lambda_{\max}|W|}$.
3. What is the expected number of candidate points generated, and the expected number of retained points?

##### Solution to Exercise 11.4
**(a)** Let $A \subset W$ be an arbitrary bounded Borel set. The candidate points in $A$ form a homogeneous Poisson process with mean $\Lambda_{\max}(A) = \lambda_{\max} |A|$.
Let $N_{\max}(A)$ be the count of candidate points. Conditioned on $N_{\max}(A) = M$, the $M$ locations $\{\mathbf{s}_1, \dots, \mathbf{s}_M\}$ are independent and identically distributed uniformly over $A$, with density $f(\mathbf{s}) = 1/|A|$.
Each point $\mathbf{s}_i$ is independently retained with probability:
$$
p(\mathbf{s}_i) = \frac{\lambda(\mathbf{s}_i)}{\lambda_{\max}}
$$
The marginal retention probability for any candidate point in $A$ is:
$$
p_A = \operatorname{E}\left[\frac{\lambda(\mathbf{S})}{\lambda_{\max}}\right] = \int_A \frac{\lambda(\mathbf{s})}{\lambda_{\max}} \frac{1}{|A|} \, \mathrm{d}\mathbf{s} = \frac{1}{\lambda_{\max} |A|} \int_A \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
$$
Conditioned on $N_{\max}(A) = M$, the number of retained points $N_{\text{thinned}}(A)$ is Binomial:
$$
N_{\text{thinned}}(A) \mid (N_{\max}(A) = M) \sim \operatorname{Binomial}(M, p_A)
$$
Because $N_{\max}(A) \sim \operatorname{Poisson}(\lambda_{\max} |A|)$, by the classic Poisson-Binomial thinning theorem, the marginal distribution of $N_{\text{thinned}}(A)$ is Poisson:
$$
\operatorname{E}[N_{\text{thinned}}(A)] = \operatorname{E}[N_{\max}(A)] \cdot p_A = (\lambda_{\max} |A|) \cdot \left(\frac{1}{\lambda_{\max} |A|} \int_A \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}\right) = \int_A \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
$$
Furthermore, for any collection of disjoint Borel sets $A_1, \dots, A_k \subset W$, the candidate counts $N_{\max}(A_1), \dots, N_{\max}(A_k)$ are independent Poisson variables. Since thinning decisions are made independently at each point based solely on its own coordinates, the thinned counts $N_{\text{thinned}}(A_1), \dots, N_{\text{thinned}}(A_k)$ remain mutually independent.
Hence, $\Pi_{\text{thinned}}$ satisfies both axioms of an inhomogeneous Poisson process with intensity function $\lambda(\mathbf{s})$.

**(b)** 
1. Since $x, y \ge 0$, $\exp(x + 2y)$ is strictly monotonically increasing in both $x$ and $y$. The maximum on $W = [0, 1] \times [0, 1]$ occurs at the upper corner $(x, y) = (1, 1)$:
$$
\lambda_{\max} = \lambda(1, 1) = 100 \exp(1 + 2(1)) = 100 e^3 \approx 100 \times 20.085537 = 2008.5537
$$
2. The total expected number of retained points is the integral of $\lambda(x, y)$ over $W$:
$$
\begin{aligned}
\mu_{\text{target}} = \int_0^1 \int_0^1 100 e^{x + 2y} \, \mathrm{d}x \, \mathrm{d}y &= 100 \left(\int_0^1 e^x \, \mathrm{d}x\right) \left(\int_0^1 e^{2y} \, \mathrm{d}y\right) \\
&= 100 (e^1 - 1) \left[\frac{e^{2(1)} - 1}{2}\right] \\
&= 50 (e - 1)(e^2 - 1)
\end{aligned}
$$
Evaluating numerically ($e \approx 2.718282$, $e^2 \approx 7.389056$):
$$
\mu_{\text{target}} = 50 (1.718282)(6.389056) \approx 50 \times 10.978198 \approx 548.91
$$
The area $|W| = 1 \times 1 = 1.0$. The acceptance probability is:
$$
\alpha = \frac{\mu_{\text{target}}}{\lambda_{\max} |W|} = \frac{548.91}{2008.55} \approx 0.27328 \quad (27.33\%)
$$
3. The expected number of candidate points generated is:
$$
\operatorname{E}[N_{\text{candidate}}] = \lambda_{\max} |W| = 2008.55
$$
The expected number of retained points is:
$$
\operatorname{E}[N_{\text{retained}}] = \mu_{\text{target}} \approx 548.91
$$

---

## Tier I: Core Theoretical Re-Derivations

### Exercise 11.5: Unabridged Derivation of Campbell's Theorem for Counting Measures
In classical probability, for an i.i.d. random sample $X_1, \dots, X_N$ where sample size $N$ is fixed, the expectation of a sum of functions is $\operatorname{E}[\sum_{i=1}^N f(X_i)] = N \operatorname{E}[f(X)]$. When the points form a stochastic point process $N(\cdot)$, both the number of points and their locations are random.

Let $N$ be a spatial point process on $\mathbb{R}^d$ with first-order intensity measure $\Lambda(A) = \operatorname{E}[N(A)] = \int_A \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}$.
Let $f: \mathbb{R}^d \to \mathbb{R}$ be any non-negative, measurable function.

**(a)** State **Campbell's First Theorem** for the expectation of the random sum $\sum_{\mathbf{s}_i \in N} f(\mathbf{s}_i)$.

**(b)** Prove Campbell's First Theorem using the standard measure-theoretic machine:
1. Prove the result for simple indicator functions $f(\mathbf{s}) = \mathbf{1}_A(\mathbf{s})$ for bounded Borel sets $A$.
2. Extend by linearity to simple non-negative functions $f(\mathbf{s}) = \sum_{k=1}^m c_k \mathbf{1}_{A_k}(\mathbf{s})$.
3. Extend to general non-negative measurable functions $f(\mathbf{s})$ via the Monotone Convergence Theorem.

**(c)** State and prove **Campbell's Second Theorem** (for second-order factorial moments):
$$
\operatorname{E}\left[\sum_{\mathbf{s}_i \ne \mathbf{s}_j \in N} f(\mathbf{s}_i, \mathbf{s}_j)\right] = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} f(\mathbf{u}, \mathbf{v}) \lambda^{(2)}(\mathbf{u}, \mathbf{v}) \, \mathrm{d}\mathbf{u} \, \mathrm{d}\mathbf{v}
$$
where $\lambda^{(2)}(\mathbf{u}, \mathbf{v})$ is the second-order product intensity.

##### Solution to Exercise 11.5
**(a)** **Campbell's First Theorem:** Let $N$ be a spatial point process on $\mathbb{R}^d$ with intensity measure $\Lambda(\cdot)$ having density $\lambda(\mathbf{s})$. For any non-negative measurable function $f: \mathbb{R}^d \to [0, \infty)$, the random sum $S = \sum_{\mathbf{s} \in N} f(\mathbf{s}) = \int_{\mathbb{R}^d} f(\mathbf{s}) \, N(\mathrm{d}\mathbf{s})$ has expectation:
$$
\operatorname{E}\left[\sum_{\mathbf{s} \in N} f(\mathbf{s})\right] = \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
$$

**(b)** **Proof:**
1. *Indicator functions:* Let $f(\mathbf{s}) = \mathbf{1}_A(\mathbf{s})$ where $A \in \mathcal{B}(\mathbb{R}^d)$ is a bounded Borel set.
The sum over points is:
$$
\sum_{\mathbf{s} \in N} \mathbf{1}_A(\mathbf{s}) = N(A)
$$
Taking the expectation:
$$
\operatorname{E}\left[\sum_{\mathbf{s} \in N} \mathbf{1}_A(\mathbf{s})\right] = \operatorname{E}[N(A)] = \Lambda(A) = \int_A \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s} = \int_{\mathbb{R}^d} \mathbf{1}_A(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
$$
Thus, Campbell's theorem holds for all indicator functions.

2. *Simple functions:* Let $f(\mathbf{s}) = \sum_{k=1}^m c_k \mathbf{1}_{A_k}(\mathbf{s})$ where $c_k \ge 0$ and $A_k \in \mathcal{B}(\mathbb{R}^d)$.
By the linearity of expectation and integration:
$$
\begin{aligned}
\operatorname{E}\left[\sum_{\mathbf{s} \in N} \sum_{k=1}^m c_k \mathbf{1}_{A_k}(\mathbf{s})\right] &= \sum_{k=1}^m c_k \operatorname{E}\left[\sum_{\mathbf{s} \in N} \mathbf{1}_{A_k}(\mathbf{s})\right] \\
&= \sum_{k=1}^m c_k \int_{\mathbb{R}^d} \mathbf{1}_{A_k}(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s} \\
&= \int_{\mathbb{R}^d} \left(\sum_{k=1}^m c_k \mathbf{1}_{A_k}(\mathbf{s})\right) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s} \\
&= \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
\end{aligned}
$$

3. *General non-negative measurable functions:* By the standard construction of Lebesgue integration, any non-negative measurable function $f: \mathbb{R}^d \to [0, \infty)$ is the pointwise limit of a monotonically non-decreasing sequence of non-negative simple functions $\{f_m\}_{m=1}^\infty$, i.e., $f_m(\mathbf{s}) \le f_{m+1}(\mathbf{s})$ and $\lim_{m \to \infty} f_m(\mathbf{s}) = f(\mathbf{s})$.
For each $m$:
$$
\operatorname{E}\left[\sum_{\mathbf{s} \in N} f_m(\mathbf{s})\right] = \int_{\mathbb{R}^d} f_m(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
$$
By Lebesgue's Monotone Convergence Theorem applied to both sides:
$$
\lim_{m \to \infty} \operatorname{E}\left[\sum_{\mathbf{s} \in N} f_m(\mathbf{s})\right] = \operatorname{E}\left[\lim_{m \to \infty} \sum_{\mathbf{s} \in N} f_m(\mathbf{s})\right] = \operatorname{E}\left[\sum_{\mathbf{s} \in N} f(\mathbf{s})\right]
$$
and
$$
\lim_{m \to \infty} \int_{\mathbb{R}^d} f_m(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s} = \int_{\mathbb{R}^d} \lim_{m \to \infty} f_m(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s} = \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, \mathrm{d}\mathbf{s}
$$
Equating limits completes the proof. $\blacksquare$

**(c)** **Campbell's Second Theorem:**
The second-order factorial moment measure $\alpha^{(2)}$ on $\mathbb{R}^d \times \mathbb{R}^d$ is defined for disjoint Borel sets $A, B$ by:
$$
\alpha^{(2)}(A \times B) = \operatorname{E}\left[N(A)N(B) - N(A \cap B)\right] = \operatorname{E}\left[\sum_{\mathbf{s}_i \ne \mathbf{s}_j \in N} \mathbf{1}_A(\mathbf{s}_i) \mathbf{1}_B(\mathbf{s}_j)\right]
$$
If $\alpha^{(2)}$ has a Radon-Nikodym density $\lambda^{(2)}(\mathbf{u}, \mathbf{v})$ with respect to Lebesgue measure $\mathrm{d}\mathbf{u}\,\mathrm{d}\mathbf{v}$, then:
$$
\alpha^{(2)}(A \times B) = \int_A \int_B \lambda^{(2)}(\mathbf{u}, \mathbf{v}) \, \mathrm{d}\mathbf{u} \, \mathrm{d}\mathbf{v}
$$
Let $f(\mathbf{u}, \mathbf{v}) = \mathbf{1}_A(\mathbf{u}) \mathbf{1}_B(\mathbf{v})$. Then:
$$
\operatorname{E}\left[\sum_{\mathbf{s}_i \ne \mathbf{s}_j} f(\mathbf{s}_i, \mathbf{s}_j)\right] = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} \mathbf{1}_A(\mathbf{u}) \mathbf{1}_B(\mathbf{v}) \lambda^{(2)}(\mathbf{u}, \mathbf{v}) \, \mathrm{d}\mathbf{u} \, \mathrm{d}\mathbf{v}
$$
Extending by linearity to simple functions on the product space $\mathbb{R}^d \times \mathbb{R}^d$ and then to all non-negative measurable functions $f(\mathbf{u}, \mathbf{v})$ via the Monotone Convergence Theorem yields:
$$
\operatorname{E}\left[\sum_{\mathbf{s}_i \ne \mathbf{s}_j \in N} f(\mathbf{s}_i, \mathbf{s}_j)\right] = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} f(\mathbf{u}, \mathbf{v}) \lambda^{(2)}(\mathbf{u}, \mathbf{v}) \, \mathrm{d}\mathbf{u} \, \mathrm{d}\mathbf{v} \quad \blacksquare
$$

---

### Exercise 11.6: Exact Derivation of Ripley's $K(r) = \pi r^2$ and Besag's $L(r) = r$ for CSR
Ripley's $K$-function is defined as:
$$
K(r) = \frac{1}{\lambda} \operatorname{E}\left[\text{number of further events within distance } r \text{ of an arbitrary typical event}\right]
$$

**(a)** State the Palm distribution formulation of Ripley's $K$-function:
$$
\lambda K(r) = \operatorname{E}_0\left[N(B(\mathbf{0}, r) \setminus \{\mathbf{0}\})\right]
$$
where $\operatorname{E}_0$ denotes expectation with respect to the reduced Palm distribution $P_0^!$.

**(b)** Using **Slivnyak's Theorem** for homogeneous Poisson processes, prove that:
$$
P_0^! = P
$$
that is, conditioning on an event occurring at location $\mathbf{s}_0$ does not change the distribution of the remaining points of the process.

**(c)** From Slivnyak's Theorem, evaluate $\operatorname{E}_0\left[N(B(\mathbf{0}, r) \setminus \{\mathbf{0}\})\right]$ and conclude that:
$$
K(r) = \pi r^2 \quad \text{and} \quad L(r) = \sqrt{\frac{K(r)}{\pi}} = r
$$
Explain why Besag's $L(r)$ is preferred over $K(r)$ for exploratory visualization and hypothesis testing.

##### Solution to Exercise 11.6
**(a)** In stationary point process theory, the reduced Palm distribution $P_{\mathbf{s}_0}^!$ formalizes the conditional distribution of the point process given that an event is located at $\mathbf{s}_0$, while excluding the point at $\mathbf{s}_0$ itself.
By stationarity, we can translate $\mathbf{s}_0$ to the origin $\mathbf{0}$, defining the reduced Palm distribution at the origin $P_0^!$.
Under this conditioning:
$$
\lambda K(r) = \operatorname{E}_0\left[N(B(\mathbf{0}, r) \setminus \{\mathbf{0}\})\right]
$$
where $B(\mathbf{0}, r)$ is the closed circular ball of radius $r$ centered at the origin. Dividing by $\lambda$ gives Ripley's $K(r)$.

**(b)** **Slivnyak's Theorem:** Let $\Pi$ be a Poisson point process on $\mathbb{R}^d$ with intensity measure $\Lambda$. Then for almost all $\mathbf{s} \in \mathbb{R}^d$, the reduced Palm distribution $P_{\mathbf{s}}^!$ equals the original unconditional distribution $P$:
$$
P_{\mathbf{s}}^! = P
$$
*Proof outline:* Consider the probability generating functional of the Poisson process $\Pi$:
$$
G[h] = \operatorname{E}\left[\exp\left(\int \log h(\mathbf{x}) \, N(\mathrm{d}\mathbf{x})\right)\right] = \exp\left(\int (h(\mathbf{x}) - 1) \Lambda(\mathrm{d}\mathbf{x})\right)
$$
By Campbell's theorem and the definition of the Palm measure, the functional derivative of $G[h]$ with respect to a point mass at $\mathbf{s}$ factors into:
$$
G_{\mathbf{s}}^![h] = G[h]
$$
Intuitively, because a Poisson point process has completely independent increments (the number of events in disjoint sets are independent), the presence of an event at the exact coordinate $\mathbf{s}_0$ gives zero information about the existence or count of events in any disjoint set $B(\mathbf{s}_0, r) \setminus \{\mathbf{s}_0\}$.
Therefore, under $P_0^!$, the process on $\mathbb{R}^d \setminus \{\mathbf{0}\}$ remains an unaltered, homogeneous Poisson process with intensity $\lambda$.

**(c)** Applying Slivnyak's Theorem:
$$
\operatorname{E}_0\left[N(B(\mathbf{0}, r) \setminus \{\mathbf{0}\})\right] = \operatorname{E}[N(B(\mathbf{0}, r))]
$$
Since $N(B(\mathbf{0}, r)) \sim \operatorname{Poisson}(\lambda |B(\mathbf{0}, r)|)$, its expectation is simply:
$$
\operatorname{E}[N(B(\mathbf{0}, r))] = \lambda |B(\mathbf{0}, r)| = \lambda \pi r^2
$$
Dividing by $\lambda$:
$$
K(r) = \frac{1}{\lambda} (\lambda \pi r^2) = \pi r^2
$$
Besag's $L$-function is the variance-stabilizing and linearizing transformation of $K(r)$:
$$
L(r) = \sqrt{\frac{K(r)}{\pi}} = \sqrt{\frac{\pi r^2}{\pi}} = r
$$
**Why $L(r)$ is preferred in practice:**
1. *Linear Benchmark:* Under CSR, $K(r) = \pi r^2$ is parabolic. Assessing deviations from a curved parabola by eye is perceptually difficult. In contrast, $L(r) = r$ is a straight 1:1 diagonal line, and plotting $L(r) - r$ creates a horizontal benchmark at zero ($y = 0$). Any positive bump ($L(r) - r > 0$) immediately reveals clustering, while any dip below zero ($L(r) - r < 0$) reveals inhibition.
2. *Variance Stabilization:* The estimator $\widehat{K}(r)$ has variance that grows rapidly as a function of lag $r$ (proportional to $r^4$). The square-root transformation stabilizes the asymptotic variance of $\widehat{L}(r)$, rendering the width of Monte Carlo confidence envelopes nearly constant across all spatial lags $r$.

---

### Exercise 11.7: Closed-Form Moment Structure of Log-Gaussian Cox Processes (LGCP)
A Log-Gaussian Cox Process (LGCP) is a doubly stochastic Poisson process where the intensity surface $\Lambda(\mathbf{s})$ is itself a continuous spatial random field defined by:
$$
\Lambda(\mathbf{s}) = \exp(Y(\mathbf{s}))
$$
where $\{Y(\mathbf{s}) : \mathbf{s} \in \mathbb{R}^d\}$ is a Gaussian random field with stationary mean $\mu = \operatorname{E}[Y(\mathbf{s})]$ and stationary covariance function $C(\mathbf{h}) = \operatorname{Cov}(Y(\mathbf{s}), Y(\mathbf{s} + \mathbf{h}))$ with variance $\sigma^2 = C(\mathbf{0})$.

**(a)** Using the moment generating function of a univariate Gaussian random variable, derive the constant first-order intensity $\lambda = \operatorname{E}[\Lambda(\mathbf{s})]$.

**(b)** Using the bivariate normal moment generating function for $(Y(\mathbf{s}), Y(\mathbf{s} + \mathbf{h}))'$, derive the second-order product intensity:
$$
\lambda^{(2)}(\mathbf{h}) = \operatorname{E}[\Lambda(\mathbf{s})\Lambda(\mathbf{s} + \mathbf{h})] = \lambda^2 \exp(C(\mathbf{h}))
$$

**(c)** Prove that the pair correlation function of an LGCP is:
$$
g(\mathbf{h}) = \exp(C(\mathbf{h}))
$$
Show that because $C(\mathbf{h}) \ge 0$ for positive covariance, $g(\mathbf{h}) \ge 1$ everywhere, proving that an LGCP is strictly aggregated/clustered relative to CSR.

**(d)** Derive the integral representation of Ripley's $K$-function for an isotropic LGCP in $\mathbb{R}^2$:
$$
K(r) = 2\pi \int_0^r u \exp(C(u)) \, \mathrm{d}u
$$

##### Solution to Exercise 11.7
**(a)** At any single location $\mathbf{s} \in \mathbb{R}^d$, $Y(\mathbf{s}) \sim \mathcal{N}(\mu, \sigma^2)$.
The intensity is $\Lambda(\mathbf{s}) = \exp(Y(\mathbf{s}))$, which is a log-normal random variable.
The moment generating function of $Y \sim \mathcal{N}(\mu, \sigma^2)$ is:
$$
M_Y(t) = \operatorname{E}[e^{tY}] = \exp\left(\mu t + \frac{1}{2}\sigma^2 t^2\right)
$$
Setting $t = 1$:
$$
\lambda = \operatorname{E}[\Lambda(\mathbf{s})] = \operatorname{E}[e^{Y(\mathbf{s})}] = M_Y(1) = \exp\left(\mu + \frac{1}{2}\sigma^2\right)
$$

**(b)** Consider two locations $\mathbf{s}$ and $\mathbf{s} + \mathbf{h}$. The vector $(Y(\mathbf{s}), Y(\mathbf{s} + \mathbf{h}))'$ follows a bivariate normal distribution:
$$
\begin{pmatrix} Y(\mathbf{s}) \\ Y(\mathbf{s} + \mathbf{h}) \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mu \\ \mu \end{pmatrix}, \begin{pmatrix} \sigma^2 & C(\mathbf{h}) \\ C(\mathbf{h}) & \sigma^2 \end{pmatrix}\right)
$$
By the definition of a doubly stochastic Cox process, conditioned on the random realization of the intensity field $\Lambda(\cdot)$, the process is an inhomogeneous Poisson process.
By the law of total expectation, the second-order product intensity is:
$$
\lambda^{(2)}(\mathbf{h}) = \operatorname{E}[\Lambda(\mathbf{s})\Lambda(\mathbf{s} + \mathbf{h})] = \operatorname{E}\left[\exp(Y(\mathbf{s}) + Y(\mathbf{s} + \mathbf{h}))\right]
$$
Let $W = Y(\mathbf{s}) + Y(\mathbf{s} + \mathbf{h})$. As a linear combination of jointly Gaussian variables, $W$ is univariate normal:
$$
\operatorname{E}[W] = \mu + \mu = 2\mu
$$
$$
\operatorname{Var}(W) = \operatorname{Var}(Y(\mathbf{s})) + \operatorname{Var}(Y(\mathbf{s} + \mathbf{h})) + 2\operatorname{Cov}(Y(\mathbf{s}), Y(\mathbf{s} + \mathbf{h})) = \sigma^2 + \sigma^2 + 2C(\mathbf{h}) = 2\sigma^2 + 2C(\mathbf{h})
$$
The expectation of $\exp(W)$ is:
$$
\begin{aligned}
\operatorname{E}[e^W] &= \exp\left(\operatorname{E}[W] + \frac{1}{2}\operatorname{Var}(W)\right) \\
&= \exp\left(2\mu + \frac{1}{2}(2\sigma^2 + 2C(\mathbf{h}))\right) \\
&= \exp\left(2\mu + \sigma^2 + C(\mathbf{h})\right) \\
&= \exp\left(2(\mu + \frac{1}{2}\sigma^2) + C(\mathbf{h})\right) \\
&= \left[\exp\left(\mu + \frac{1}{2}\sigma^2\right)\right]^2 \cdot \exp(C(\mathbf{h})) \\
&= \lambda^2 \exp(C(\mathbf{h}))
\end{aligned}
$$

**(c)** The pair correlation function is defined as the normalized second-order product intensity:
$$
g(\mathbf{h}) = \frac{\lambda^{(2)}(\mathbf{h})}{\lambda^2} = \frac{\lambda^2 \exp(C(\mathbf{h}))}{\lambda^2} = \exp(C(\mathbf{h}))
$$
Under standard positive spatial dependence, $C(\mathbf{h}) \ge 0$ for all lags $\mathbf{h}$ (or at least for short lags where spatial correlation is non-negative).
Since the exponential function satisfies $\exp(x) \ge 1$ for all $x \ge 0$:
$$
g(\mathbf{h}) = \exp(C(\mathbf{h})) \ge \exp(0) = 1
$$
Because $g(\mathbf{h}) = 1$ is the benchmark for Complete Spatial Randomness, $g(\mathbf{h}) > 1$ proves that pairs of events are strictly more likely to occur at separation vector $\mathbf{h}$ than under CSR. Thus, an LGCP with non-negative covariance is mathematically guaranteed to exhibit spatial aggregation/clustering at all correlated spatial scales.

**(d)** For an isotropic point process in $\mathbb{R}^2$, the pair correlation function $g(u)$ depends only on scalar Euclidean distance $u = \|\mathbf{h}\|$.
By definition, Ripley's $K(r)$ is related to $g(u)$ through polar integration:
$$
K(r) = \int_{B(\mathbf{0}, r)} g(\mathbf{u}) \, \mathrm{d}\mathbf{u} = \int_0^{2\pi} \int_0^r g(u) u \, \mathrm{d}u \, \mathrm{d}\theta = 2\pi \int_0^r u g(u) \, \mathrm{d}u
$$
Substituting $g(u) = \exp(C(u))$:
$$
K(r) = 2\pi \int_0^r u \exp(C(u)) \, \mathrm{d}u
$$
Since $\exp(C(u)) > 1$ for $C(u) > 0$, we have:
$$
K(r) > 2\pi \int_0^r u(1) \, \mathrm{d}u = 2\pi \left[\frac{u^2}{2}\right]_0^r = \pi r^2 = K_{\text{CSR}}(r)
$$
which formally confirms $K(r) > \pi r^2$ for all $r > 0$.

---

### Exercise 11.8: Second-Order Moment Structure of Neyman-Scott / Thomas Cluster Processes
A Neyman-Scott cluster process is generated via a two-stage hierarchical mechanism:
1. **Parents:** An unobserved homogeneous Poisson point process of parent events $\Pi_{\text{parent}}$ with intensity $\kappa > 0$.
2. **Offspring:** Each parent $\mathbf{c} \in \Pi_{\text{parent}}$ independently produces a random number of offspring $M \sim \operatorname{Poisson}(\mu)$, scattered around the parent location $\mathbf{c}$ according to an isotropic spatial probability density function $f(\mathbf{s} - \mathbf{c})$.
The observed point pattern consists of all offspring points (parents are discarded). When $f$ is a bivariate zero-mean Gaussian density with variance $\sigma^2 \mathbf{I}_2$:
$$
f(\mathbf{u}) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{\|\mathbf{u}\|^2}{2\sigma^2}\right)
$$
the process is called a **Thomas process**.

**(a)** Derive the overall first-order intensity $\lambda$ of the observed Thomas process.

**(b)** Show that the displacement vector between two offspring from the same parent follows a bivariate normal distribution $\mathcal{N}(\mathbf{0}, 2\sigma^2 \mathbf{I}_2)$.

**(c)** Derive the pair correlation function $g(r)$ of the Thomas process:
$$
g(r) = 1 + \frac{1}{4\pi \kappa \sigma^2} \exp\left(-\frac{r^2}{4\sigma^2}\right)
$$
Show that as $r \to \infty$, $g(r) \to 1$, while at $r = 0$, $g(0) = 1 + \frac{1}{4\pi \kappa \sigma^2} > 1$.

**(d)** Derive Ripley's $K$-function:
$$
K(r) = \pi r^2 + \frac{1}{\kappa}\left(1 - \exp\left(-\frac{r^2}{4\sigma^2}\right)\right)
$$

##### Solution to Exercise 11.8
**(a)** Each parent produces an expected number of offspring $\operatorname{E}[M] = \mu$. Since parents occur with spatial intensity $\kappa$, the overall first-order intensity of offspring is:
$$
\lambda = \kappa \mu
$$

**(b)** Let $\mathbf{c}$ be the coordinate of a parent. Two distinct offspring from this parent have coordinates $\mathbf{S}_1 = \mathbf{c} + \mathbf{U}_1$ and $\mathbf{S}_2 = \mathbf{c} + \mathbf{U}_2$, where $\mathbf{U}_1, \mathbf{U}_2 \overset{\text{iid}}{\sim} \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_2)$.
The difference vector between the two offspring is:
$$
\mathbf{D} = \mathbf{S}_1 - \mathbf{S}_2 = (\mathbf{c} + \mathbf{U}_1) - (\mathbf{c} + \mathbf{U}_2) = \mathbf{U}_1 - \mathbf{U}_2
$$
Because $\mathbf{U}_1$ and $\mathbf{U}_2$ are independent Gaussian vectors:
$$
\operatorname{E}[\mathbf{D}] = \mathbf{0} - \mathbf{0} = \mathbf{0}
$$
$$
\operatorname{Cov}(\mathbf{D}) = \operatorname{Cov}(\mathbf{U}_1) + \operatorname{Cov}(\mathbf{U}_2) = \sigma^2 \mathbf{I}_2 + \sigma^2 \mathbf{I}_2 = 2\sigma^2 \mathbf{I}_2
$$
Hence, $\mathbf{D} \sim \mathcal{N}(\mathbf{0}, 2\sigma^2 \mathbf{I}_2)$, with isotropic density:
$$
f_{\mathbf{D}}(\mathbf{u}) = \frac{1}{2\pi (2\sigma^2)} \exp\left(-\frac{\|\mathbf{u}\|^2}{2(2\sigma^2)}\right) = \frac{1}{4\pi \sigma^2} \exp\left(-\frac{\|\mathbf{u}\|^2}{4\sigma^2}\right)
$$

**(c)** Two distinct observed events $\mathbf{s}_1, \mathbf{s}_2$ separated by vector $\mathbf{h} = \mathbf{s}_1 - \mathbf{s}_2$ can arise in two mutually exclusive ways:
1. *Different parents:* Since parents form a Poisson process, offspring from different parents are uncorrelated, contributing $\lambda^2 = (\kappa \mu)^2$ to the product intensity.
2. *Same parent:* The expected number of distinct pairs of offspring produced by a single parent is $\operatorname{E}[M(M - 1)] = \operatorname{Var}(M) + (\operatorname{E}[M])^2 - \operatorname{E}[M] = \mu + \mu^2 - \mu = \mu^2$ (since $M \sim \operatorname{Poisson}(\mu)$).
The spatial density of this same-parent pair occurring at separation $\mathbf{h}$ is given by $\kappa \mu^2 f_{\mathbf{D}}(\mathbf{h})$.

Combining both contributions yields the second-order product intensity:
$$
\lambda^{(2)}(\mathbf{h}) = \lambda^2 + \kappa \mu^2 f_{\mathbf{D}}(\mathbf{h}) = (\kappa \mu)^2 + \kappa \mu^2 \cdot \frac{1}{4\pi \sigma^2} \exp\left(-\frac{\|\mathbf{h}\|^2}{4\sigma^2}\right)
$$
Dividing by $\lambda^2 = (\kappa \mu)^2$:
$$
g(r) = \frac{\lambda^{(2)}(r)}{\lambda^2} = 1 + \frac{\kappa \mu^2}{(\kappa \mu)^2} \frac{1}{4\pi \sigma^2} \exp\left(-\frac{r^2}{4\sigma^2}\right) = 1 + \frac{1}{4\pi \kappa \sigma^2} \exp\left(-\frac{r^2}{4\sigma^2}\right)
$$
Evaluating the limits:
$$
\lim_{r \to \infty} g(r) = 1 + 0 = 1 \quad (\text{pairs at large separation behave like CSR})
$$
$$
g(0) = 1 + \frac{1}{4\pi \kappa \sigma^2} > 1 \quad (\text{strong clustering at zero lag})
$$
Notice that the cluster peak height $\frac{1}{4\pi \kappa \sigma^2}$ is inversely proportional to parent intensity $\kappa$ and cluster dispersion $\sigma^2$: rare parents with tightly concentrated offspring produce extreme clustering peaks.

**(d)** Using the polar integral relation between $K(r)$ and $g(u)$:
$$
K(r) = 2\pi \int_0^r u g(u) \, \mathrm{d}u = 2\pi \int_0^r u \left[1 + \frac{1}{4\pi \kappa \sigma^2} \exp\left(-\frac{u^2}{4\sigma^2}\right)\right] \mathrm{d}u
$$
Splitting the integral:
$$
K(r) = 2\pi \int_0^r u \, \mathrm{d}u + \frac{2\pi}{4\pi \kappa \sigma^2} \int_0^r u \exp\left(-\frac{u^2}{4\sigma^2}\right) \mathrm{d}u = \pi r^2 + \frac{1}{2\kappa \sigma^2} \int_0^r u \exp\left(-\frac{u^2}{4\sigma^2}\right) \mathrm{d}u
$$
Let $w = \frac{u^2}{4\sigma^2}$, so $\mathrm{d}w = \frac{2u}{4\sigma^2} \mathrm{d}u = \frac{u}{2\sigma^2} \mathrm{d}u \implies u \, \mathrm{d}u = 2\sigma^2 \, \mathrm{d}w$. When $u = 0$, $w = 0$; when $u = r$, $w = \frac{r^2}{4\sigma^2}$.
$$
\int_0^r u \exp\left(-\frac{u^2}{4\sigma^2}\right) \mathrm{d}u = 2\sigma^2 \int_0^{\frac{r^2}{4\sigma^2}} e^{-w} \, \mathrm{d}w = 2\sigma^2 \left[-e^{-w}\right]_0^{\frac{r^2}{4\sigma^2}} = 2\sigma^2 \left(1 - \exp\left(-\frac{r^2}{4\sigma^2}\right)\right)
$$
Substituting this back:
$$
K(r) = \pi r^2 + \frac{1}{2\kappa \sigma^2} \left[2\sigma^2 \left(1 - \exp\left(-\frac{r^2}{4\sigma^2}\right)\right)\right] = \pi r^2 + \frac{1}{\kappa}\left(1 - \exp\left(-\frac{r^2}{4\sigma^2}\right)\right) \quad \blacksquare
$$

---

## Tier II: Pencil-and-Paper Mechanics

### Exercise 11.9: Manual Quadrat Dispersion Test on a $4 \times 4$ Spatial Lattice
A spatial ecologist records the mapped locations of $N = 48$ seedlings of *Pinus ponderosa* in a square forest plot $W = [0, 40] \times [0, 40]\text{ meters}$. To test for Complete Spatial Randomness, the plot is gridded into $m = 16$ contiguous quadrats of dimensions $10 \times 10\text{ meters}$ arranged in a $4 \times 4$ grid.
The observed counts $X_{j,k}$ in the 16 quadrats (row $j$, column $k$) are:
$$
\mathbf{X} = \begin{pmatrix}
1 & 0 & 2 & 1 \\
2 & 8 & 9 & 3 \\
0 & 7 & 10 & 2 \\
1 & 1 & 0 & 1
\end{pmatrix}
$$

**(a)** Calculate:
1. The sample mean quadrat count $\bar{X}$.
2. The sample variance $S^2 = \frac{1}{m - 1}\sum_{i=1}^{16} (X_i - \bar{X})^2$.
3. The sample Index of Dispersion $\mathrm{ID} = S^2 / \bar{X}$.

**(b)** State the null hypothesis $H_0$ and test statistic for Complete Spatial Randomness. Under $H_0$, evaluate the test statistic $\chi^2_{\text{obs}} = (m - 1)\mathrm{ID}$.
Find the critical value at significance level $\alpha = 0.01$ for degrees of freedom $\mathrm{df} = 15$. (Use $\chi^2_{15, 0.99} = 30.58$).

**(c)** State the statistical and biological conclusion: is the spatial distribution of seedlings random, aggregated, or regular? Identify the physical feature evident in the quadrat count matrix $\mathbf{X}$.

##### Solution to Exercise 11.9
**(a)**
1. The total count across the 16 quadrats is:
$$
\sum_{i=1}^{16} X_i = (1+0+2+1) + (2+8+9+3) + (0+7+10+2) + (1+1+0+1) = 4 + 22 + 19 + 3 = 48
$$
The sample mean is:
$$
\bar{X} = \frac{48}{16} = 3.00
$$
2. Compute the squared deviations $(X_i - \bar{X})^2$:
- Count 0 (3 occurrences): $(0 - 3)^2 = 9 \implies 3 \times 9 = 27$
- Count 1 (6 occurrences): $(1 - 3)^2 = 4 \implies 6 \times 4 = 24$
- Count 2 (3 occurrences): $(2 - 3)^2 = 1 \implies 3 \times 1 = 3$
- Count 3 (1 occurrence): $(3 - 3)^2 = 0 \implies 1 \times 0 = 0$
- Count 7 (1 occurrence): $(7 - 3)^2 = 16 \implies 1 \times 16 = 16$
- Count 8 (1 occurrence): $(8 - 3)^2 = 25 \implies 1 \times 25 = 25$
- Count 9 (1 occurrence): $(9 - 3)^2 = 36 \implies 1 \times 36 = 36$
- Count 10 (1 occurrence): $(10 - 3)^2 = 49 \implies 1 \times 49 = 49$

The sum of squared deviations is:
$$
\sum_{i=1}^{16} (X_i - \bar{X})^2 = 27 + 24 + 3 + 0 + 16 + 25 + 36 + 49 = 180.00
$$
The sample variance is:
$$
S^2 = \frac{180.00}{16 - 1} = \frac{180.00}{15} = 12.00
$$
3. The Index of Dispersion is:
$$
\mathrm{ID} = \frac{S^2}{\bar{X}} = \frac{12.00}{3.00} = 4.00
$$

**(b)**
- *Null hypothesis $H_0$:* The seedlings follow Complete Spatial Randomness (homogeneous Poisson point process with constant intensity $\lambda$).
- *Alternative hypothesis $H_1$:* The seedlings exhibit spatial aggregation (clustering).
- *Test statistic:* Under $H_0$, $(m - 1)\mathrm{ID} \sim \chi^2_{m - 1}$.
$$
\chi^2_{\text{obs}} = (16 - 1) \times 4.00 = 15 \times 4.00 = 60.00
$$
The critical value at $\alpha = 0.01$ with 15 degrees of freedom is $\chi^2_{15, 0.99} = 30.58$.
Since $\chi^2_{\text{obs}} = 60.00 \gg 30.58$, the $p$-value is $p = \operatorname{P}(\chi^2_{15} \ge 60.00) \approx 2.4 \times 10^{-7}$.

**(c)**
- *Conclusion:* We reject $H_0$ with overwhelming statistical significance ($p < 10^{-6}$).
- *Interpretation:* Because $\mathrm{ID} = 4.00 \gg 1$, the seedlings exhibit strong spatial aggregation/clustering.
- *Physical Feature:* Looking at matrix $\mathbf{X}$, the central $2 \times 2$ block contains counts $(8, 9, 7, 10)$, accounting for $34$ of the $48$ trees ($70.8\%$ of the population) in just $25\%$ of the total area. This indicates a concentrated cluster, possibly around a mature seed tree, canopy gap, or local moisture depression.

---

### Exercise 11.10: Analytical Evaluation of Ripley's Isotropic Edge-Correction Weight $w(\mathbf{s}_i, \mathbf{s}_j)$
In the estimation of Ripley's $K$-function:
$$
\widehat{K}(r) = \frac{|W|}{n(n - 1)} \sum_{i=1}^n \sum_{j \ne i} \mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}} w(\mathbf{s}_i, \mathbf{s}_j)^{-1}
$$
Ripley's isotropic edge correction weight $w(\mathbf{s}_i, \mathbf{s}_j)$ is the proportion of the circumference of the circle centered at $\mathbf{s}_i$ with radius $d_{ij} = \|\mathbf{s}_i - \mathbf{s}_j\|$ that lies within the observation window $W$.

Consider a rectangular window $W = [0, X_{\max}] \times [0, Y_{\max}]$.
Let event $\mathbf{s}_i = (x_i, y_i)'$ be located close to the left vertical boundary $x = 0$, such that:
$$
x_i < d_{ij} < \min(X_{\max} - x_i, y_i, Y_{\max} - y_i)
$$
so that the circle of radius $d_{ij}$ centered at $\mathbf{s}_i$ intersects ONLY the left boundary $x = 0$, and no other edge.

**(a)** Sketch or set up the geometry of the circle intersecting the vertical line $x = 0$.
Find the angle $\theta$ subtended by the arc that falls *outside* the window $W$.

**(b)** Prove that the inside proportion of the circumference is:
$$
w(\mathbf{s}_i, \mathbf{s}_j) = 1 - \frac{1}{\pi} \arccos\left(\frac{x_i}{d_{ij}}\right)
$$

**(c)** Let $\mathbf{s}_i = (2, 10)'$ and $\mathbf{s}_j = (5, 14)'$ in window $W = [0, 20] \times [0, 20]$.
1. Compute the Euclidean distance $d_{ij} = \|\mathbf{s}_i - \mathbf{s}_j\|$.
2. Verify which boundaries of $W$ are intersected by the circle $C(\mathbf{s}_i, d_{ij})$.
3. Compute the exact value of the edge correction weight $w(\mathbf{s}_i, \mathbf{s}_j)$ and its expansion multiplier $w(\mathbf{s}_i, \mathbf{s}_j)^{-1}$.

##### Solution to Exercise 11.10
**(a)** The circle is given by:
$$
(x - x_i)^2 + (y - y_i)^2 = d_{ij}^2
$$
The left boundary of window $W$ is the vertical line $x = 0$. Setting $x = 0$:
$$
(-x_i)^2 + (y - y_i)^2 = d_{ij}^2 \implies (y - y_i)^2 = d_{ij}^2 - x_i^2
$$
Since $x_i < d_{ij}$, $d_{ij}^2 - x_i^2 > 0$, yielding two intersection points:
$$
y = y_i \pm \sqrt{d_{ij}^2 - x_i^2}
$$
Consider the right triangle formed by the center $(x_i, y_i)$, the projection on the boundary $(0, y_i)$, and an intersection point $(0, y_i + \sqrt{d_{ij}^2 - x_i^2})$.
The adjacent side along the $x$-axis has length $x_i$, and the hypotenuse has length $d_{ij}$.
The half-angle $\alpha$ subtended by the arc extending to the left of the boundary satisfies:
$$
\cos(\alpha) = \frac{x_i}{d_{ij}} \implies \alpha = \arccos\left(\frac{x_i}{d_{ij}}\right)
$$
The total angular arc of the circle lying *outside* $W$ (in the half-plane $x < 0$) is:
$$
\theta_{\text{out}} = 2\alpha = 2\arccos\left(\frac{x_i}{d_{ij}}\right)
$$

**(b)** The total circumference of the circle corresponds to an angle of $2\pi$ radians.
The proportion of circumference lying inside $W$ is:
$$
w(\mathbf{s}_i, \mathbf{s}_j) = \frac{2\pi - \theta_{\text{out}}}{2\pi} = 1 - \frac{2\arccos(x_i / d_{ij})}{2\pi} = 1 - \frac{1}{\pi} \arccos\left(\frac{x_i}{d_{ij}}\right) \quad \blacksquare
$$

**(c)**
1. The Euclidean distance between $\mathbf{s}_i = (2, 10)'$ and $\mathbf{s}_j = (5, 14)'$ is:
$$
d_{ij} = \sqrt{(5 - 2)^2 + (14 - 10)^2} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5.0
$$
2. Check distances from $\mathbf{s}_i = (2, 10)'$ to the four boundaries of $W = [0, 20] \times [0, 20]$:
- Distance to left edge ($x = 0$): $x_i = 2.0 < d_{ij} = 5.0$ $\implies$ **intersects left edge**.
- Distance to right edge ($x = 20$): $20 - x_i = 18.0 > 5.0$ $\implies$ does not intersect.
- Distance to bottom edge ($y = 0$): $y_i = 10.0 > 5.0$ $\implies$ does not intersect.
- Distance to top edge ($y = 20$): $20 - y_i = 10.0 > 5.0$ $\implies$ does not intersect.
Only the left boundary $x = 0$ is intersected.
3. Compute the edge weight using the formula derived in part (b):
$$
\frac{x_i}{d_{ij}} = \frac{2.0}{5.0} = 0.40
$$
$$
\arccos(0.40) \approx 1.159279 \text{ radians}
$$
$$
w(\mathbf{s}_i, \mathbf{s}_j) = 1 - \frac{1.159279}{\pi} = 1 - 0.369010 = 0.630990 \approx 0.6310
$$
The expansion multiplier is:
$$
w(\mathbf{s}_i, \mathbf{s}_j)^{-1} = \frac{1}{0.630990} \approx 1.5848
$$
*Interpretation:* Exactly $63.1\%$ of the perimeter of the search circle centered at $\mathbf{s}_i$ lies inside the observation plot. Weighting point $\mathbf{s}_j$ by $1.5848$ compensates for the $36.9\%$ of search area lost beyond the edge of the map.

---

### Exercise 11.11: Pencil-and-Paper Computation of Empirical $\widehat{K}(r)$ for a 4-Point Pattern
Consider an observation window $W = [0, 10] \times [0, 10]$ (area $|W| = 100$) containing exactly $n = 4$ mapped events:
$$
\mathbf{s}_1 = (2, 2)', \quad \mathbf{s}_2 = (3, 2)', \quad \mathbf{s}_3 = (8, 8)', \quad \mathbf{s}_4 = (8, 9)'
$$

**(a)** Construct the complete $4 \times 4$ pairwise Euclidean distance matrix $\mathbf{D} = [d_{ij}]_{i,j=1}^4$.

**(b)** For each ordered pair $(i, j)$ with $i \ne j$, compute Ripley's edge correction weight $w(\mathbf{s}_i, \mathbf{s}_j)$. (Hint: check if the circle of radius $d_{ij}$ around $\mathbf{s}_i$ intersects any boundary of $W$).

**(c)** Evaluate the edge-corrected empirical estimator $\widehat{K}(r)$ and Besag's $\widehat{L}(r) = \sqrt{\widehat{K}(r)/\pi}$ at distance thresholds $r = 1.5$ and $r = 3.0$. Compare $\widehat{L}(r)$ against the CSR benchmark $L_{\text{CSR}}(r) = r$.

##### Solution to Exercise 11.11
**(a)** Compute all Euclidean distances $d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$:
- $d_{12} = \sqrt{(3 - 2)^2 + (2 - 2)^2} = \sqrt{1 + 0} = 1.0$
- $d_{13} = \sqrt{(8 - 2)^2 + (8 - 2)^2} = \sqrt{36 + 36} = \sqrt{72} \approx 8.4853$
- $d_{14} = \sqrt{(8 - 2)^2 + (9 - 2)^2} = \sqrt{36 + 49} = \sqrt{85} \approx 9.2195$
- $d_{23} = \sqrt{(8 - 3)^2 + (8 - 2)^2} = \sqrt{25 + 36} = \sqrt{61} \approx 7.8102$
- $d_{24} = \sqrt{(8 - 3)^2 + (9 - 2)^2} = \sqrt{25 + 49} = \sqrt{74} \approx 8.6023$
- $d_{34} = \sqrt{(8 - 8)^2 + (9 - 8)^2} = \sqrt{0 + 1} = 1.0$

The distance matrix is:
$$
\mathbf{D} = \begin{pmatrix}
0 & 1.0 & 8.4853 & 9.2195 \\
1.0 & 0 & 7.8102 & 8.6023 \\
8.4853 & 7.8102 & 0 & 1.0 \\
9.2195 & 8.6023 & 1.0 & 0
\end{pmatrix}
$$

**(b)** We examine distances to boundaries for each point:
- For $\mathbf{s}_1 = (2, 2)'$, distances to boundaries are $x = 2$, $y = 2$, $10 - x = 8$, $10 - y = 8$. The minimum distance to any boundary is $\delta_1 = 2.0$.
- For $\mathbf{s}_2 = (3, 2)'$, minimum distance is $\delta_2 = \min(3, 2, 7, 8) = 2.0$.
- For $\mathbf{s}_3 = (8, 8)'$, minimum distance is $\delta_3 = \min(8, 8, 2, 2) = 2.0$.
- For $\mathbf{s}_4 = (8, 9)'$, minimum distance is $\delta_4 = \min(8, 9, 2, 1) = 1.0$.

For pairs with $d_{ij} \le 1.5$:
- Pair $(1, 2)$ and $(2, 1)$: $d_{12} = 1.0 < \delta_1 = 2.0$ and $d_{21} = 1.0 < \delta_2 = 2.0$. The circle of radius $1.0$ does not touch any boundary. Thus $w(\mathbf{s}_1, \mathbf{s}_2) = 1.0$ and $w(\mathbf{s}_2, \mathbf{s}_1) = 1.0$.
- Pair $(3, 4)$ and $(4, 3)$: $d_{34} = 1.0 < \delta_3 = 2.0$, so $w(\mathbf{s}_3, \mathbf{s}_4) = 1.0$.
For $\mathbf{s}_4 = (8, 9)'$, the distance to the top boundary $y = 10$ is $10 - 9 = 1.0$. Since $d_{43} = 1.0$, the circle of radius $1.0$ is exactly tangent to the boundary $y = 10$ at $(8, 10)$, intersecting it at a single point of measure zero. Thus, the interior proportion is $w(\mathbf{s}_4, \mathbf{s}_3) = 1.0$.

All other pairwise distances are $d_{ij} \ge 7.81 > 3.0$.

**(c)** The estimator formula is:
$$
\widehat{K}(r) = \frac{|W|}{n(n - 1)} \sum_{i=1}^n \sum_{j \ne i} \mathbf{1}_{\{d_{ij} \le r\}} w(\mathbf{s}_i, \mathbf{s}_j)^{-1}
$$
With $|W| = 100$ and $n = 4$:
$$
\frac{|W|}{n(n - 1)} = \frac{100}{4 \times 3} = \frac{100}{12} = \frac{25}{3} \approx 8.3333
$$

1. *At $r = 1.5$:*
The pairs with $d_{ij} \le 1.5$ are $(1, 2), (2, 1), (3, 4), (4, 3)$ (4 ordered pairs).
Each has $w(\mathbf{s}_i, \mathbf{s}_j)^{-1} = 1.0$.
$$
\sum_{i=1}^4 \sum_{j \ne i} \mathbf{1}_{\{d_{ij} \le 1.5\}} w(\mathbf{s}_i, \mathbf{s}_j)^{-1} = 1.0 + 1.0 + 1.0 + 1.0 = 4.0
$$
$$
\widehat{K}(1.5) = \frac{25}{3} \times 4.0 = \frac{100}{3} \approx 33.3333
$$
Besag's $L$-function is:
$$
\widehat{L}(1.5) = \sqrt{\frac{\widehat{K}(1.5)}{\pi}} = \sqrt{\frac{33.3333}{\pi}} \approx \sqrt{10.6103} \approx 3.2573
$$
Comparison with CSR:
$$
\widehat{L}(1.5) - 1.5 = 3.2573 - 1.5 = +1.7573 \gg 0
$$

2. *At $r = 3.0$:*
No additional pairs have distance between $1.5$ and $3.0$ (the next smallest distance is $d_{23} \approx 7.81$).
Therefore, the sum remains $4.0$:
$$
\widehat{K}(3.0) = \frac{25}{3} \times 4.0 = \frac{100}{3} \approx 33.3333
$$
$$
\widehat{L}(3.0) = \sqrt{\frac{33.3333}{\pi}} \approx 3.2573
$$
Comparison with CSR:
$$
\widehat{L}(3.0) - 3.0 = 3.2573 - 3.0 = +0.2573 > 0
$$
*Diagnosis:* At small spatial lags ($r = 1.5$), $\widehat{L}(r) - r \gg 0$, indicating pronounced local clustering (the pairs are grouped into two tight doublets: $(1, 2)$ and $(3, 4)$).

---

## Tier III: Algorithmic Translation & Labs in Pure Python

### Exercise 11.12: Pure Python Neyman-Scott / Thomas Cluster Process Simulation Engine
Implement a complete, dependency-free Python simulation engine for the bivariate Thomas cluster point process on an arbitrary rectangular window $W = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$.

To eliminate edge truncation effects (where parents outside $W$ contribute offspring inside $W$), implement an **expanded buffer window** $W_{\text{buf}} = [x_{\min} - 4\sigma, x_{\max} + 4\sigma] \times [y_{\min} - 4\sigma, y_{\max} + 4\sigma]$:
1. Simulate parent locations in $W_{\text{buf}}$ from a homogeneous Poisson process with intensity $\kappa$.
2. For each parent, simulate a Poisson number of offspring $M \sim \operatorname{Poisson}(\mu)$.
3. Scatter offspring using the Box-Muller transform for bivariate normal perturbations with standard deviation $\sigma$.
4. Retain only those offspring that fall within the target window $W$.

Write a test script with $\kappa = 15$, $\mu = 10$, $\sigma = 0.05$, on $W = [0, 1] \times [0, 1]$ that reports:
- Number of parents generated in buffer window vs. target window.
- Total offspring generated and final count retained in $W$.
- Observed empirical intensity $\widehat{\lambda} = N(W)/|W|$ vs. theoretical intensity $\lambda = \kappa \mu$.

##### Solution to Exercise 11.12
```python
"""
Exercise 11.12: Dependency-Free Thomas Cluster Process Simulator
Generates an isotropic Thomas Neyman-Scott cluster point process using an
expanded buffer window to prevent edge-effect truncation.
"""

import math
import random


def simulate_poisson_count(mean_rate):
    """
    Simulate a Poisson random variable with parameter mean_rate
    using Knuth's algorithm (or Gaussian inversion for large rates).
    """
    if mean_rate <= 0:
        return 0
    if mean_rate < 30.0:
        # Knuth's multiplicative method
        limit = math.exp(-mean_rate)
        k = 0
        p = 1.0
        while p > limit:
            k += 1
            p *= random.random()
        return k - 1
    else:
        # Gaussian approximation with continuity correction
        z = random.gauss(0.0, 1.0)
        val = int(round(mean_rate + math.sqrt(mean_rate) * z))
        return max(0, val)


def box_muller_2d(sigma):
    """
    Generate independent bivariate normal perturbation (dx, dy)
    with zero mean and standard deviation sigma.
    """
    u1 = random.random()
    while u1 <= 1e-12:  # Avoid log(0)
        u1 = random.random()
    u2 = random.random()
    r = sigma * math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    dx = r * math.cos(theta)
    dy = r * math.sin(theta)
    return dx, dy


def simulate_thomas_cluster_process(
    kappa,
    mu,
    sigma,
    window=(0.0, 1.0, 0.0, 1.0),
    buffer_multiplier=4.0,
    seed=None,
):
    """
    Simulate Thomas Cluster Process on rectangular window.

    Parameters
    ----------
    kappa : float
        Parent process intensity (parents per unit area).
    mu : float
        Mean number of offspring per parent.
    sigma : float
        Standard deviation of Gaussian offspring dispersion.
    window : tuple
        (x_min, x_max, y_min, y_max) target observation window.
    buffer_multiplier : float
        Buffer zone width in multiples of sigma (default: 4.0).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing retained points and simulation diagnostics.
    """
    if seed is not None:
        random.seed(seed)

    x_min, x_max, y_min, y_max = window
    buf = buffer_multiplier * sigma

    # Expanded buffer window coordinates
    x_buf_min = x_min - buf
    x_buf_max = x_max + buf
    y_buf_min = y_min - buf
    y_buf_max = y_max + buf

    area_buf = (x_buf_max - x_buf_min) * (y_buf_max - y_buf_min)
    area_target = (x_max - x_min) * (y_max - y_min)

    # 1. Simulate Parent Process in Buffer Window
    expected_parents = kappa * area_buf
    num_parents = simulate_poisson_count(expected_parents)

    parents_buf = []
    parents_in_target = 0
    for _ in range(num_parents):
        px = x_buf_min + random.random() * (x_buf_max - x_buf_min)
        py = y_buf_min + random.random() * (y_buf_max - y_buf_min)
        parents_buf.append((px, py))
        if x_min <= px <= x_max and y_min <= py <= y_max:
            parents_in_target += 1

    # 2. Simulate Offspring for Each Parent
    offspring_total = 0
    retained_points = []

    for px, py in parents_buf:
        num_offspring = simulate_poisson_count(mu)
        offspring_total += num_offspring
        for _ in range(num_offspring):
            dx, dy = box_muller_2d(sigma)
            ox = px + dx
            oy = py + dy
            # Retain only if offspring lands inside target window
            if x_min <= ox <= x_max and y_min <= oy <= y_max:
                retained_points.append((ox, oy))

    return {
        "retained_points": retained_points,
        "n_retained": len(retained_points),
        "parents_in_buffer": num_parents,
        "parents_in_target": parents_in_target,
        "offspring_total_generated": offspring_total,
        "theoretical_intensity": kappa * mu,
        "empirical_intensity": len(retained_points) / area_target,
    }


if __name__ == "__main__":
    res = simulate_thomas_cluster_process(
        kappa=15.0,
        mu=10.0,
        sigma=0.05,
        window=(0.0, 1.0, 0.0, 1.0),
        seed=42,
    )
    print("=" * 65)
    print("Thomas Cluster Point Process Simulation Diagnostic Report")
    print("=" * 65)
    print(f"Parent Intensity (kappa)          : 15.0 parents/unit area")
    print(f"Mean Offspring per Parent (mu)    : 10.0")
    print(f"Cluster Dispersion (sigma)        : 0.05 units")
    print("-" * 65)
    print(f"Parents Generated in Buffer Window: {res['parents_in_buffer']}")
    print(f"Parents Lying in Target Window   : {res['parents_in_target']}")
    print(f"Total Offspring Generated         : {res['offspring_total_generated']}")
    print(f"Final Offspring Retained in Target: {res['n_retained']}")
    print("-" * 65)
    print(f"Theoretical Intensity (kappa * mu): {res['theoretical_intensity']:.2f}")
    print(f"Empirical Observed Intensity     : {res['empirical_intensity']:.2f}")
    print("=" * 65)
```

---

### Exercise 11.13: Diagnostic Simulation Lab: Comparing $\widehat{L}(r) - r$ Curves across Spatial Mechanisms
Implement a comprehensive diagnostic simulation laboratory in pure Python that generates three distinct spatial point patterns on $W = [0, 1] \times [0, 1]$, each calibrated to have approximately $N \approx 100$ events:
1. **Complete Spatial Randomness (CSR):** Homogeneous Poisson process with $\lambda = 100$.
2. **Clustered Pattern (Thomas Process):** $\kappa = 10$, $\mu = 10$, $\sigma = 0.04$.
3. **Regular Pattern (Strauss / Hard-Core Inhibition):** Simple sequential inhibition (SSI) generating 100 points with hard-core radius $\delta = 0.06$.

Compute Besag's edge-corrected diagnostic curve $\widehat{L}(r) - r$ across 10 spatial lags $r \in [0.02, 0.20]$ for all three patterns, and output a clean ASCII comparison table demonstrating how $\widehat{L}(r) - r$ distinguishes clustering from inhibition.

##### Solution to Exercise 11.13
```python
"""
Exercise 11.13: Second-Order Summary Diagnostic Lab (CSR vs Cluster vs Inhibition)
Pure Python implementation of Besag's L(r) - r diagnostic curves.
"""

import math
import random


def generate_csr_pattern(n_points=100, seed=101):
    random.seed(seed)
    return [(random.random(), random.random()) for _ in range(n_points)]


def generate_cluster_pattern(kappa=10, mu=10, sigma=0.04, seed=202):
    random.seed(seed)
    buf = 4.0 * sigma
    area_buf = (1.0 + 2 * buf) ** 2
    n_parents = max(1, int(round(kappa * area_buf)))
    points = []
    for _ in range(n_parents):
        px = -buf + random.random() * (1.0 + 2 * buf)
        py = -buf + random.random() * (1.0 + 2 * buf)
        n_offspring = max(1, int(round(random.gauss(mu, math.sqrt(mu)))))
        for _ in range(n_offspring):
            u1, u2 = random.random(), random.random()
            while u1 <= 1e-10:
                u1 = random.random()
            rad = sigma * math.sqrt(-2.0 * math.log(u1))
            ang = 2.0 * math.pi * u2
            ox = px + rad * math.cos(ang)
            oy = py + rad * math.sin(ang)
            if 0.0 <= ox <= 1.0 and 0.0 <= oy <= 1.0:
                points.append((ox, oy))
    return points


def generate_hardcore_inhibition_pattern(n_target=100, delta=0.06, max_iter=20000, seed=303):
    random.seed(seed)
    points = []
    delta_sq = delta * delta
    for _ in range(max_iter):
        cand_x = random.random()
        cand_y = random.random()
        conflict = False
        for ex, ey in points:
            d2 = (cand_x - ex)**2 + (cand_y - ey)**2
            if d2 < delta_sq:
                conflict = True
                break
        if not conflict:
            points.append((cand_x, cand_y))
            if len(points) == n_target:
                break
    return points


def ripley_edge_weight(x, y, d, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    if d <= 0.0:
        return 1.0
    c_left = x - d < x_min
    c_right = x + d > x_max
    c_bottom = y - d < y_min
    c_top = y + d > y_max
    n_cross = sum([c_left, c_right, c_bottom, c_top])
    if n_cross == 0:
        return 1.0
    cut_angle = 0.0
    if c_left:
        cut_angle += 2.0 * math.acos(max(-1.0, min(1.0, (x - x_min) / d)))
    if c_right:
        cut_angle += 2.0 * math.acos(max(-1.0, min(1.0, (x_max - x) / d)))
    if c_bottom:
        cut_angle += 2.0 * math.acos(max(-1.0, min(1.0, (y - y_min) / d)))
    if c_top:
        cut_angle += 2.0 * math.acos(max(-1.0, min(1.0, (y_max - y) / d)))
    inside = max(0.01, 1.0 - (cut_angle / (2.0 * math.pi)))
    return inside


def compute_besag_l_curve(points, r_lags, area=1.0):
    n = len(points)
    factor = area / (n * (n - 1))
    l_diffs = []
    for r in r_lags:
        sum_k = 0.0
        for i in range(n):
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                d = math.hypot(xi - xj, yi - yj)
                if d <= r:
                    w = ripley_edge_weight(xi, yi, d)
                    sum_k += 1.0 / w
        k_est = factor * sum_k
        l_est = math.sqrt(k_est / math.pi)
        l_diffs.append(l_est - r)
    return l_diffs


if __name__ == "__main__":
    csr_pts = generate_csr_pattern(n_points=100, seed=101)
    clu_pts = generate_cluster_pattern(kappa=10, mu=10, sigma=0.04, seed=202)
    reg_pts = generate_hardcore_inhibition_pattern(n_target=100, delta=0.06, seed=303)

    r_lags = [0.02 * k for k in range(1, 11)]

    l_csr = compute_besag_l_curve(csr_pts, r_lags)
    l_clu = compute_besag_l_curve(clu_pts, r_lags)
    l_reg = compute_besag_l_curve(reg_pts, r_lags)

    print("=" * 80)
    print("Diagnostic Simulation Lab: Besag's L(r) - r Comparison Across Mechanisms")
    print(f"Point Counts -> CSR: {len(csr_pts)}, Clustered: {len(clu_pts)}, Regular: {len(reg_pts)}")
    print("=" * 80)
    print(f"{'Lag r':>8} | {'CSR Null':>12} | {'Clustered':>14} | {'Inhibited (SSI)':>16} | {'Diagnosis':>18}")
    print("-" * 80)

    for r, d_csr, d_clu, d_reg in zip(r_lags, l_csr, l_clu, l_reg):
        diag = "Distinct Shapes" if abs(d_clu - d_reg) > 0.05 else "Indistinguishable"
        print(f"{r:8.2f} | {d_csr:+12.4f} | {d_clu:+14.4f} | {d_reg:+16.4f} | {diag:>18}")

    print("-" * 80)
    print("Interpretation:")
    print("  • Clustered : L(r) - r > 0 peaking at intermediate inter-cluster scale (~2*sigma = 0.08).")
    print("  • Inhibited : L(r) - r < 0 (severe dip below 0) for lags r < delta = 0.06 (repulsion).")
    print("  • CSR Null  : L(r) - r fluctuates tightly around 0.00.")
    print("=" * 80)
```

---

## Tier IV: Edge-Case Diagnostics & Conceptual Reflections

### Exercise 11.14: The First-Order vs. Second-Order Confounding Dilemma
One of the most profound foundational challenges in spatial point process analysis is the **identifiability dilemma**: an observed aggregated point pattern can be generated with equal statistical fidelity by either:
1. An **inhomogeneous Poisson process** with variable first-order intensity $\lambda(\mathbf{s})$ (deterministic environmental trend, e.g., trees clustering on a moist hillside).
2. A **homogeneous cluster process** (or Cox process) with constant first-order intensity $\lambda_0$ and second-order attractive interactions (biological contagion, e.g., seed dispersal).

**(a)** Explain mathematically why evaluating standard second-order summary statistics (such as unadjusted Ripley's $\widehat{K}(r)$) on an inhomogeneous Poisson process produces false positive diagnoses of spatial clustering ($\widehat{L}(r) - r \gg 0$).

**(b)** Define the **inhomogeneous $K$-function** $K_{\text{inhom}}(r)$ introduced by Baddeley, Møller, and Waagepetersen (2000):
$$
\widehat{K}_{\text{inhom}}(r) = \frac{1}{|W|} \sum_{i=1}^n \sum_{j \ne i} \frac{\mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}}}{\widehat{\lambda}(\mathbf{s}_i)\widehat{\lambda}(\mathbf{s}_j) w(\mathbf{s}_i, \mathbf{s}_j)}
$$
Prove that under an inhomogeneous Poisson process with true intensity $\lambda(\mathbf{s})$, $\operatorname{E}[\widehat{K}_{\text{inhom}}(r)] = \pi r^2$.

**(c)** In empirical applications where replicate realizations are unavailable, what diagnostic strategies or external covariates allow an analyst to distinguish between true second-order clustering and first-order environmental heterogeneity?

##### Solution to Exercise 11.14
**(a)** Standard Ripley's $K$-function assumes first-order stationarity ($\lambda(\mathbf{s}) \equiv \lambda = n/|W|$).
Its numerator estimates the expected number of neighboring events using the global average intensity $\widehat{\lambda}^2 = (n/|W|)^2$ as the normalization denominator:
$$
\widehat{K}(r) \propto \frac{1}{\widehat{\lambda}^2} \sum_{i \ne j} \mathbf{1}_{\{d_{ij} \le r\}}
$$
If the true underlying process is an inhomogeneous Poisson process with spatial trend $\lambda(\mathbf{s})$, events naturally concentrate in high-intensity subregions where $\lambda(\mathbf{s}) \gg \bar{\lambda}$.
By Campbell's second theorem, the expected count of pairs within distance $r$ is:
$$
\operatorname{E}\left[\sum_{i \ne j} \mathbf{1}_{\{d_{ij} \le r\}}\right] = \int_W \int_W \mathbf{1}_{\{\|\mathbf{u} - \mathbf{v}\| \le r\}} \lambda(\mathbf{u})\lambda(\mathbf{v}) \, \mathrm{d}\mathbf{u} \, \mathrm{d}\mathbf{v}
$$
By the Cauchy-Schwarz inequality, for non-constant $\lambda(\mathbf{s})$:
$$
\frac{1}{|W|} \int_W \lambda(\mathbf{u})^2 \, \mathrm{d}\mathbf{u} > \left(\frac{1}{|W|} \int_W \lambda(\mathbf{u}) \, \mathrm{d}\mathbf{u}\right)^2 = \bar{\lambda}^2
$$
For small lag distances $r \approx 0$, $\lambda(\mathbf{v}) \approx \lambda(\mathbf{u})$, giving:
$$
\int_W \int_{B(\mathbf{u}, r)} \lambda(\mathbf{u})\lambda(\mathbf{v}) \, \mathrm{d}\mathbf{v} \, \mathrm{d}\mathbf{u} \approx \pi r^2 \int_W \lambda(\mathbf{u})^2 \, \mathrm{d}\mathbf{u} > \pi r^2 |W| \bar{\lambda}^2
$$
Dividing by $\bar{\lambda}^2 |W|$ yields $\operatorname{E}[\widehat{K}(r)] > \pi r^2$.
Thus, unadjusted Ripley's $K$ mistakes first-order spatial drift for true second-order biological attraction, diagnosing clustering where none exists.

**(b)** The inhomogeneous $K$-function weights each pair of points $(\mathbf{s}_i, \mathbf{s}_j)$ inversely by the product of their local intensities $\lambda(\mathbf{s}_i)\lambda(\mathbf{s}_j)$:
$$
\widehat{K}_{\text{inhom}}(r) = \frac{1}{|W|} \sum_{i=1}^n \sum_{j \ne i} \frac{\mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}}}{\lambda(\mathbf{s}_i)\lambda(\mathbf{s}_j) w(\mathbf{s}_i, \mathbf{s}_j)}
$$
Assuming the edge weights $w(\mathbf{s}_i, \mathbf{s}_j)$ correctly adjust for window boundary effects, let $f(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{1}_{\{\|\mathbf{u} - \mathbf{v}\| \le r\}}}{\lambda(\mathbf{u})\lambda(\mathbf{v})}$.
Applying Campbell's Second Theorem:
$$
\begin{aligned}
\operatorname{E}[\widehat{K}_{\text{inhom}}(r)] &= \frac{1}{|W|} \int_W \int_W \frac{\mathbf{1}_{\{\|\mathbf{u} - \mathbf{v}\| \le r\}}}{\lambda(\mathbf{u})\lambda(\mathbf{v})} \lambda^{(2)}(\mathbf{u}, \mathbf{v}) \, \mathrm{d}\mathbf{u} \, \mathrm{d}\mathbf{v}
\end{aligned}
$$
For an inhomogeneous Poisson process, points are conditionally independent, so the second-order product intensity factors exactly:
$$
\lambda^{(2)}(\mathbf{u}, \mathbf{v}) = \lambda(\mathbf{u})\lambda(\mathbf{v})
$$
Substituting this factorization:
$$
\begin{aligned}
\operatorname{E}[\widehat{K}_{\text{inhom}}(r)] &= \frac{1}{|W|} \int_W \int_W \frac{\mathbf{1}_{\{\|\mathbf{u} - \mathbf{v}\| \le r\}}}{\lambda(\mathbf{u})\lambda(\mathbf{v})} \left(\lambda(\mathbf{u})\lambda(\mathbf{v})\right) \, \mathrm{d}\mathbf{u} \, \mathrm{d}\mathbf{v} \\
&= \frac{1}{|W|} \int_W \left(\int_{W \cap B(\mathbf{u}, r)} \mathrm{d}\mathbf{v}\right) \mathrm{d}\mathbf{u} \\
&= \frac{1}{|W|} \int_W (\pi r^2) \, \mathrm{d}\mathbf{u} = \frac{\pi r^2 |W|}{|W|} = \pi r^2 \quad \blacksquare
\end{aligned}
$$
This proves that $\widehat{K}_{\text{inhom}}(r)$ successfully centers the expectation at $\pi r^2$ even when the process possesses intense first-order spatial heterogeneity.

**(c)** **Diagnostic and Modeling Strategies:**
1. *Incorporate Exogenous Environmental Covariates:* Fit a spatial Poisson GLM/GAM $\log \lambda(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ using external physical raster layers (e.g., elevation, soil pH, moisture, slope aspect). Then compute $\widehat{K}_{\text{inhom}}(r)$ standardized by this estimated intensity surface $\widehat{\lambda}(\mathbf{s})$. If $\widehat{K}_{\text{inhom}}(r) \approx \pi r^2$, the observed aggregation was entirely driven by environmental niche preference (first-order). If $\widehat{K}_{\text{inhom}}(r) \gg \pi r^2$ persists, residual aggregation represents true biological clustering (second-order contagion).
2. *Scale Separation:* First-order trends typically operate across broad, regional spatial scales (macro-scale, comparable to window diameter $|W|^{1/2}$), whereas biological cluster mechanisms (seed shadow, root competition) operate at short, micro-scale distances $r \ll |W|^{1/2}$.
3. *Model Comparison via AIC / Likelihood:* Compare an Inhomogeneous Poisson model against an Inhomogeneous Cox model (LGCP) using residual diagnostics or information criteria.

---

### Exercise 11.15: Geometric Window Boundary Truncation: Minus Sampling vs. Ripley Weighting
When estimating distance summary functions ($F(r), G(r), K(r)$), points outside the observation window $W$ are unobserved.
Two primary strategies exist to counteract this edge truncation bias:
1. **Minus Sampling (Border Method):** Restrict focal events to an eroded sub-window $W_{\ominus r} = \{\mathbf{s} \in W : \operatorname{dist}(\mathbf{s}, \partial W) \ge r\}$, while allowing their candidate neighbors to lie anywhere in the full window $W$.
2. **Ripley's Circumferential Isotropic Weighting:** Use all focal points in $W$, but weight each observed neighbor pair by the reciprocal of the circumference fraction $w(\mathbf{s}_i, \mathbf{s}_j)^{-1}$.

**(a)** Prove that under minus sampling, the effective sample size decreases rapidly as the analysis lag $r$ increases. Specifically, for a square window $W = [0, L] \times [0, L]$, derive the proportion of window area discarded as a function of $r / L$.

**(b)** Explain why minus sampling is strictly unbiased for all distance metrics, but suffers from severe variance inflation at large lags $r$.

**(c)** Conversely, explain why Ripley's weighting retains all $n$ data points, but can produce numerical instability (exploding weights) when pairs occur near window corners, where $w(\mathbf{s}_i, \mathbf{s}_j) \to 0$.

##### Solution to Exercise 11.15
**(a)** For a square window $W = [0, L] \times [0, L]$, the total area is $|W| = L^2$.
The eroded sub-window $W_{\ominus r}$ is obtained by trimming a strip of width $r$ from all four borders:
$$
W_{\ominus r} = [r, L - r] \times [r, L - r]
$$
The area of the eroded window is:
$$
|W_{\ominus r}| = (L - 2r)^2 = L^2 \left(1 - \frac{2r}{L}\right)^2 \quad \text{for } r < L/2
$$
The proportion of discarded window area is:
$$
\text{Discarded Area Fraction} = 1 - \frac{|W_{\ominus r}|}{|W|} = 1 - \left(1 - \frac{2r}{L}\right)^2 = \frac{4r}{L} - \frac{4r^2}{L^2}
$$
For example:
- At $r = 0.10 L$: $1 - (1 - 0.20)^2 = 1 - 0.64 = 0.36$ ($36\%$ of the window area is discarded).
- At $r = 0.25 L$: $1 - (1 - 0.50)^2 = 1 - 0.25 = 0.75$ ($75\%$ of the window area is discarded).
- At $r = 0.50 L$: $|W_{\ominus r}| = 0$ ($100\%$ of the data is lost).

**(b)**
- *Unbiasedness:* For any focal point $\mathbf{s}_i \in W_{\ominus r}$, the complete circular ball $B(\mathbf{s}_i, r)$ lies entirely within the full observation window $W$ by construction ($\operatorname{dist}(\mathbf{s}_i, \partial W) \ge r$). Thus, every neighbor within distance $r$ is guaranteed to be observed; zero potential neighbors are obscured by the window boundary. Because the indicator $\mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}}$ is evaluated without censoring, the estimator is mathematically unbiased for any spatial geometry.
- *Variance Inflation:* As $r$ grows, the number of events falling within $W_{\ominus r}$ drops precipitously ($N(W_{\ominus r}) \propto (L - 2r)^2$). At moderate-to-large lags $r$, the estimator relies on a tiny subset of surviving focal points, causing the sample variance of the estimator to explode.

**(c)**
- *Ripley's Circumferential Weighting:* Uses all $n$ points as focal events, preserving full sample size across all spatial lags.
- *Corner Instability:* When a point $\mathbf{s}_i$ is located near a corner of $W$ (e.g., coordinates $(\epsilon, \epsilon)$ for small $\epsilon$), a circle of radius $d_{ij} > \epsilon\sqrt{2}$ centered at $\mathbf{s}_i$ extends outside both intersecting perpendicular boundaries. In the extreme case of a sharp corner, the fraction of the circumference remaining inside the window can become very small:
$$
w(\mathbf{s}_i, \mathbf{s}_j) \approx \frac{\pi / 2}{2\pi} = \frac{1}{4} \quad \text{or smaller}
$$
If a neighbor is detected near such a boundary, its weight multiplier $w(\mathbf{s}_i, \mathbf{s}_j)^{-1}$ is $4.0$ or higher. If a circle crosses multiple edges near a narrow polygon vertex, $w(\mathbf{s}_i, \mathbf{s}_j) \to 0$, causing $w^{-1} \to \infty$. A single pair of points near a boundary corner can exert extreme leverage, creating massive spikes in $\widehat{K}(r)$. For this reason, practical implementations cap weights or restrict analysis to $r \le \min(X_{\max}, Y_{\max})/4$.

---

### Exercise 11.16: Decisional Framework: Point Processes vs. Continuous Geostatistics
A data science team is tasked with analyzing four spatial datasets. For each scenario, determine whether the appropriate statistical paradigm is **Continuous Geostatistics** (Random Field Modeling: Covariance, Variograms, Kriging) or **Spatial Point Processes** (Counting Measures, Intensity, $K$-functions, Cox/Markov Processes). Provide a rigorous justification based on the physical data-generating mechanism.

**(a)** **Air Quality Monitoring:** PM2.5 particulate concentration measurements recorded at 42 stationary EPA monitoring stations across the state of California.

**(b)** **Forestry Epidemic:** Exact GPS coordinates of 1,240 dead ash trees infected by the Emerald Ash Borer in a national park.

**(c)** **Precision Agriculture:** Soil nitrogen concentration sampled on a regular $100\text{ m} \times 100\text{ m}$ grid, together with continuous yield monitor data recorded by a GPS-equipped combine harvester.

**(d)** **Urban Crime Analytics:** Spatial locations of 3,850 reported residential burglaries in Chicago over a 12-month period.

##### Solution to Exercise 11.16
**(a)** **Air Quality Monitoring: Continuous Geostatistics (Random Field / Kriging)**
- *Data-Generating Mechanism:* Particulate matter (PM2.5) is an intensive physical field that exists continuously at every coordinate $\mathbf{s}$ in the spatial domain $D \subset \mathbb{R}^2$. The air quality $Z(\mathbf{s})$ is well-defined even where no sensor is installed.
- *Sampling Feature:* The 42 station coordinates $\{\mathbf{s}_1, \dots, \mathbf{s}_{42}\}$ are fixed sampling locations chosen by environmental engineers; they are NOT random occurrences of the phenomenon itself.
- *Objective:* Predict the unobserved concentration $Z(\mathbf{s}_0)$ at an arbitrary unsampled residential community via Best Linear Unbiased Prediction (Ordinary / Universal Kriging) and quantify prediction uncertainty.

**(b)** **Forestry Epidemic: Spatial Point Processes (Point Pattern Analysis)**
- *Data-Generating Mechanism:* The physical locations of the dead trees are themselves the stochastic outcome of interest. Ash trees do not exist as a continuous field; each infected tree is a discrete event $\mathbf{s}_i \in \mathbb{R}^2$.
- *Sampling Feature:* The data consists of the complete enumeration (mapping) of all disease events within the park boundaries.
- *Objective:* Determine whether infection spreads contagiously (spatial clustering via Neyman-Scott / Cox processes due to insect flight dispersal) or correlates with environmental covariates such as elevation, river corridors, or host density (inhomogeneous Poisson GLM).

**(c)** **Precision Agriculture: Continuous Geostatistics (Multivariate Geostatistics / Cokriging)**
- *Data-Generating Mechanism:* Soil nitrogen and crop yield are continuous biophysical fields $Z_1(\mathbf{s})$ and $Z_2(\mathbf{s})$ that vary smoothly across the entire agricultural parcel.
- *Sampling Feature:* Samples are gathered on a fixed grid and along continuous harvesting transects.
- *Objective:* Predict spatial fertility gradients, evaluate spatial autocorrelation via empirical semivariograms, and construct high-resolution spatial maps to guide variable-rate fertilizer application.

**(d)** **Urban Crime Analytics: Spatial Point Processes (Marked / Inhomogeneous Point Process)**
- *Data-Generating Mechanism:* A burglary is an instantaneous, discrete event occurring at a specific point in space and time. There is no continuous "burglary field" that exists everywhere; crime events represent realizations of a random spatial counting measure $N(A)$.
- *Sampling Feature:* The coordinates are mapped event occurrences across the city network.
- *Objective:* Estimate spatial risk intensity $\lambda(\mathbf{s})$ (hotspot analysis via kernel density estimation or Log-Gaussian Cox processes), model the influence of built-environment covariates (distance to transit stations, liquor stores, lighting), and evaluate whether burglaries exhibit repeat-victimization inhibition or clustering.
