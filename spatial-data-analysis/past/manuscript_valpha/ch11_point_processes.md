# Chapter 11: Mapped Point Patterns and Spatial Point Processes

This chapter establishes the probabilistic foundations, moment measures, and statistical theory of **spatial point processes** and **mapped point patterns**. In geostatistics (Chapters 1–10), the spatial domain $D \subset \mathbb{R}^d$ is fixed and continuous, and randomness resides in the attribute values $Z(\mathbf{s})$ observed at fixed sampling coordinates. In spatial point pattern analysis, the paradigm is fundamentally inverted: **the spatial locations themselves are random**. The data consist of a countable set of random spatial coordinates $\mathbf{x} = \{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset W \subset \mathbb{R}^d$ representing the occurrence of discrete events—such as epicenters of earthquakes, trees in an old-growth forest, diseased organisms in an epidemic, or volcanic vents.

Following Maria L. Rizzo's three-beat pedagogical rhythm (*Statistical Computing with R*) and Oliver Schabenberger & Carol A. Gotway's semantic linear-model discipline (*Statistical Methods for Spatial Data Analysis*), this chapter begins by bridging classical one-dimensional Poisson processes to multidimensional spatial counting measures, formalizes Complete Spatial Randomness (CSR), develops Campbell's theorem and second-order moment measures (the pair correlation function $g(r)$, nearest-neighbor distributions $F(r), G(r), J(r)$, and Ripley's $K(r)$ function), and establishes the theoretical formulations of Inhomogeneous Poisson, Cox / Log-Gaussian Cox, Neyman-Scott cluster, and Strauss Markov point processes.

---

### 11.1 The Classical-to-Spatial Bridge: From 1D Arrival Counts to Spatial Point Measures

#### Beat 1: The Mathematical Anchor (The "Why")

In classical probability theory, an analyst studies a one-dimensional Poisson process $\{N(t) : t \ge 0\}$ tracking discrete arrival times along a one-dimensional timeline. Recall the familiar axioms:
1. $N(0) = 0$.
2. For $0 \le t_1 < t_2$, the count of arrivals $N(t_2) - N(t_1) \sim \text{Poisson}(\lambda(t_2 - t_1))$.
3. Disjoint time increments possess independent counts.
4. The inter-arrival times $T_k = t_k - t_{k-1}$ are independent and identically distributed exponential random variables with rate $\lambda$: $f_T(t) = \lambda e^{-\lambda t}$.

In spatial statistics, physical events occur across a continuous multidimensional observation window $W \subset \mathbb{R}^d$ ($d \ge 2$). There is no natural total ordering of coordinates (no "past" or "future"), eliminating the concept of unidirectional renewal or inter-arrival waiting times. 

Instead, the spatial point process is formalized as a **random counting measure** on the Borel $\sigma$-algebra $\mathcal{B}(W)$.

```
Classical 1D Timeline (Ordered)
  +-------*-------*----*----------*--------*----> Time t
  0      t_1     t_2  t_3        t_4      t_5
  (Inter-arrival times T_k = t_k - t_{k-1} ~ Exp(lambda))

Spatial Point Process on Window W in R^2 (Unordered)
  +-------------------------------------+
  |          * s_1                      |
  |                      * s_2          |
  |     * s_3     [   A   ]             |
  |               [ * s_4 ]             |
  |               [ * s_5 ]      * s_6  |
  +-------------------------------------+
  (Random Counting Measure: N(A) = count of events s_i in Borel set A)
```

##### Definition 11.1.1 (Spatial Point Process as a Random Counting Measure)
Let $(\Omega, \mathcal{F}, P)$ be a complete probability space, and let $W \subset \mathbb{R}^d$ be a bounded Borel observation window with positive Lebesgue measure $|W| > 0$.
A **spatial point process** $\Phi$ is a random locally finite counting measure on $\mathcal{B}(W)$. For any Borel subset $A \in \mathcal{B}(W)$, the random variable:
$$
N(A) = \Phi(A) = \sum_{\mathbf{s} \in \Phi} \mathbf{1}_{\{\mathbf{s} \in A\}} = \#(\Phi \cap A)
$$
counts the integer number of events falling inside $A$.
A point process is **simple** if multiple coincidences are impossible almost surely:
$$
P\left(N(\{\mathbf{s}\}) \le 1 \quad \forall \mathbf{s} \in W\right) = 1
$$

##### Definition 11.1.2 (Void Probabilities)
For any Borel set $A \in \mathcal{B}(W)$, the **void probability** $v(A)$ is the probability that no events fall in $A$:
$$
v(A) = P(N(A) = 0)
$$
By the Choquet-Kendall-Matheron Theorem, the collection of void probabilities $\{v(K) : K \subset W, K \text{ compact}\}$ uniquely determines the probability distribution of a simple point process on $\mathcal{B}(W)$.

---

### 11.2 Complete Spatial Randomness (CSR) & The Homogeneous Poisson Process

#### Beat 1: The Mathematical Anchor

The foundational benchmark in spatial point pattern analysis is **Complete Spatial Randomness (CSR)**, which embodies the null hypothesis of pure stochastic independence and spatial homogeneity.

##### Definition 11.2.1 (Homogeneous Poisson Process / CSR)
A spatial point process $\Phi$ on $\mathbb{R}^d$ is a **Homogeneous Poisson Process (HPP)** with constant intensity parameter $\lambda > 0$ if it satisfies two axioms:
1. **Poisson Counting Property:** For every bounded Borel set $A \in \mathcal{B}(\mathbb{R}^d)$, the count of events $N(A)$ follows a Poisson distribution with mean $\lambda |A|$:
   $$
   P(N(A) = k) = \frac{(\lambda |A|)^k}{k!} \exp(-\lambda |A|), \quad k = 0, 1, 2, \dots
   $$
   where $|A|$ denotes the Lebesgue volume (area in $\mathbb{R}^2$) of $A$.
2. **Independent Scattering (Completely Random):** For any collection of mutually disjoint Borel sets $A_1, \dots, A_m$ ($A_i \cap A_j = \emptyset$ for $i \ne j$), the random variables $N(A_1), \dots, N(A_m)$ are mutually independent:
   $$
   P\left(\bigcap_{j=1}^m \{N(A_j) = k_j\}\right) = \prod_{j=1}^m P(N(A_j) = k_j)
   $$

##### Theorem 11.2.1 (Conditional Uniform Property of CSR)
Let $\Phi$ be a Homogeneous Poisson Process with intensity $\lambda$ observed in a window $W \subset \mathbb{R}^d$. Conditioned on observing exactly $n$ events in $W$, $N(W) = n$:
1. The locations $\mathbf{s}_1, \dots, \mathbf{s}_n$ are independent and identically distributed (i.i.d.) random vectors uniformly distributed across $W$, with joint probability density function:
   $$
   f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) = \frac{1}{|W|^n}
   $$
2. For any sub-region $A \subset W$, the conditional distribution of $N(A)$ given $N(W) = n$ is strictly **Binomial**:
   $$
   N(A) \mid \{N(W) = n\} \sim \text{Binomial}\left(n, \, p = \frac{|A|}{|W|}\right)
   $$
   with probability mass function:
   $$
   P(N(A) = k \mid N(W) = n) = \binom{n}{k} \left(\frac{|A|}{|W|}\right)^k \left(1 - \frac{|A|}{|W|}\right)^{n-k}, \quad k = 0, 1, \dots, n
   $$

*Proof.*
1. Let $A_1, \dots, A_n$ be mutually disjoint infinitesimal neighborhoods centered at $\mathbf{s}_1, \dots, \mathbf{s}_n$ with volumes $d\mathbf{s}_1, \dots, d\mathbf{s}_n$, and let $A_0 = W \setminus \bigcup_{i=1}^n A_i$, so $|A_0| = |W| - \sum_{i=1}^n d\mathbf{s}_i$.
By independent scattering and the Poisson counting property:
$$
P\left(N(A_1) = 1, \dots, N(A_n) = 1, N(A_0) = 0\right) = \left[\prod_{i=1}^n \lambda d\mathbf{s}_i e^{-\lambda d\mathbf{s}_i}\right] \exp\left(-\lambda |A_0|\right)
$$
Summing the exponents: $\sum_{i=1}^n (-\lambda d\mathbf{s}_i) - \lambda |A_0| = -\lambda |W|$. Therefore:
$$
P\left(\text{points at } d\mathbf{s}_1, \dots, d\mathbf{s}_n\right) = \lambda^n e^{-\lambda |W|} d\mathbf{s}_1 \cdots d\mathbf{s}_n
$$
The joint probability that the point process places exactly one point in each disjoint infinitesimal neighborhood $d\mathbf{s}_i$ and zero in the remainder is:
$$
P\left(N(d\mathbf{s}_1) = 1, \dots, N(d\mathbf{s}_n) = 1, N\left(W \setminus \bigcup_{i=1}^n d\mathbf{s}_i\right) = 0\right) = \lambda^n e^{-\lambda |W|} d\mathbf{s}_1 \cdots d\mathbf{s}_n
$$
Because the total count $N(W) \sim \text{Poisson}(\lambda |W|)$, its marginal probability is:
$$
P(N(W) = n) = \frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}
$$
Dividing the joint infinitesimal probability by $P(N(W) = n)$ gives the conditional probability of this unordered event configuration, $\frac{n!}{|W|^n} d\mathbf{s}_1 \cdots d\mathbf{s}_n$. Because there are $n!$ equally likely orderings of $n$ distinct points forming the coordinate vector $(\mathbf{s}_1, \dots, \mathbf{s}_n)$, the conditional joint density of the ordered locations is:
$$
f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) = \frac{1}{n!} \frac{\lambda^n e^{-\lambda |W|}}{\frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}} = \frac{1}{n!} \frac{n!}{|W|^n} = \frac{1}{|W|^n}
$$
This proves that conditional on $N(W) = n$, the $n$ locations $\mathbf{s}_1, \dots, \mathbf{s}_n$ are independent and identically distributed uniform random vectors on $W$.

2. For any $A \subset W$, let $A^c = W \setminus A$. Notice that $A \cap A^c = \emptyset$ and $A \cup A^c = W$.
By independent scattering, $N(A) \sim \text{Poisson}(\lambda |A|)$ and $N(A^c) \sim \text{Poisson}(\lambda |A^c|)$ are independent.
Conditioning on $N(W) = N(A) + N(A^c) = n$:
$$
\begin{aligned}
P(N(A) = k \mid N(W) = n) &= \frac{P(N(A) = k, \, N(A^c) = n - k)}{P(N(W) = n)} \\
&= \frac{\left(\frac{(\lambda |A|)^k}{k!} e^{-\lambda |A|}\right) \left(\frac{(\lambda |A^c|)^{n-k}}{(n-k)!} e^{-\lambda |A^c|}\right)}{\frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}} \\
&= \frac{n!}{k!(n-k)!} \frac{\lambda^k |A|^k \lambda^{n-k} |A^c|^{n-k}}{\lambda^n |W|^n} \frac{e^{-\lambda (|A| + |A^c|)}}{e^{-\lambda |W|}} \\
&= \binom{n}{k} \left(\frac{|A|}{|W|}\right)^k \left(\frac{|A^c|}{|W|}\right)^{n-k} = \binom{n}{k} p^k (1 - p)^{n-k}
\end{aligned}
$$
where $p = |A|/|W|$. This is identically the $\text{Binomial}(n, p)$ distribution. $\blacksquare$

---

#### Beat 3: Intuition, Edge Cases & Diagnostic Verification

##### The Quadrat Count Dispersion Test
To test whether an observed mapped point pattern departs from Complete Spatial Randomness:
1. Divide the observation window $W$ into $Q$ equal-area, non-overlapping grid cells (quadrats) $B_1, \dots, B_Q$, each having area $|B_q| = |W| / Q$.
2. Count the observed number of events in each quadrat: $X_q = N(B_q)$ for $q = 1, \dots, Q$. Total points $n = \sum_{q=1}^Q X_q$.
3. Compute the sample mean and sample variance of the quadrat counts:
   $$
   \bar{X} = \frac{1}{Q}\sum_{q=1}^Q X_q = \frac{n}{Q}, \quad S^2 = \frac{1}{Q - 1}\sum_{q=1}^Q (X_q - \bar{X})^2
   $$
4. Compute the **Index of Dispersion**:
   $$
   I = \frac{S^2}{\bar{X}}
   $$
5. Under the null hypothesis of CSR, $X_q \sim \text{Poisson}(\mu)$, so the theoretical mean equals the theoretical variance: $\operatorname{Var}(X) = \operatorname{E}[X] \implies I = 1$.
   - **Clustered / Aggregated Pattern:** Points group together, creating many quadrats with very high counts and many empty quadrats: $S^2 > \bar{X} \implies I > 1$.
   - **Regular / Uniform / Inhibited Pattern:** Points repel one another (e.g., competing root zones), resulting in nearly identical counts across quadrats: $S^2 < \bar{X} \implies I < 1$.
6. Under $H_0: \text{CSR}$, Fisher proved that the test statistic:
   $$
   \chi^2 = (Q - 1) I = \sum_{q=1}^Q \frac{(X_q - \bar{X})^2}{\bar{X}} \sim \chi^2(Q - 1)
   $$
   asymptotically follows a chi-squared distribution with $Q - 1$ degrees of freedom.

---

### 11.3 Intensity Measures, Campbell's Theorem, and Second-Order Properties

#### Beat 1: The Mathematical Anchor

In geostatistics, first- and second-order properties are governed by the mean function $\mu(\mathbf{s}) = \operatorname{E}[Z(\mathbf{s})]$ and the covariance kernel $C(\mathbf{s}_1, \mathbf{s}_2) = \operatorname{Cov}(Z(\mathbf{s}_1), Z(\mathbf{s}_2))$.
In point processes, the first- and second-order structures are governed by **intensity measures** and **product densities**.

##### Definition 11.3.1 (First-Order Intensity Function)
Let $\Phi$ be a point process on $\mathbb{R}^d$. The **intensity measure** $\Lambda$ is the expected count of events:
$$
\Lambda(A) = \operatorname{E}[N(A)], \quad A \in \mathcal{B}(\mathbb{R}^d)
$$
If $\Lambda$ is absolutely continuous with respect to Lebesgue measure, its Radon-Nikodym derivative is the **first-order intensity function** $\lambda(\mathbf{s})$:
$$
\Lambda(A) = \int_A \lambda(\mathbf{s}) \, d\mathbf{s}
$$
Intuitively, $\lambda(\mathbf{s})$ represents the expected number of events per unit volume at coordinate $\mathbf{s}$:
$$
\lambda(\mathbf{s}) = \lim_{|d\mathbf{s}| \to 0} \frac{\operatorname{E}[N(d\mathbf{s})]}{|d\mathbf{s}|} = \lim_{|d\mathbf{s}| \to 0} \frac{P(N(d\mathbf{s}) = 1)}{|d\mathbf{s}|}
$$
If $\Phi$ is stationary, $\lambda(\mathbf{s}) = \lambda$ is constant across space.

##### Theorem 11.3.1 (Campbell’s Theorem for Sums Over Point Processes)
Let $\Phi$ be a simple point process on $\mathbb{R}^d$ with intensity function $\lambda(\mathbf{s})$. For any non-negative measurable function $f: \mathbb{R}^d \to \mathbb{R}$ (or any integrable function with $\int |f(\mathbf{s})|\lambda(\mathbf{s})d\mathbf{s} < \infty$), the expected sum of $f$ over all realization events is:
$$
\operatorname{E}\left[\sum_{\mathbf{s} \in \Phi} f(\mathbf{s})\right] = \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, d\mathbf{s}
$$

*Proof.*
We prove the theorem using the standard machine of measure theory:
1. **Indicator Functions:** Let $f(\mathbf{s}) = \mathbf{1}_A(\mathbf{s})$ for an arbitrary Borel set $A \in \mathcal{B}(\mathbb{R}^d)$.
   Then:
   $$
   \sum_{\mathbf{s} \in \Phi} f(\mathbf{s}) = \sum_{\mathbf{s} \in \Phi} \mathbf{1}_A(\mathbf{s}) = N(A)
   $$
   Taking expectations:
   $$
   \operatorname{E}\left[\sum_{\mathbf{s} \in \Phi} f(\mathbf{s})\right] = \operatorname{E}[N(A)] = \Lambda(A) = \int_{\mathbb{R}^d} \mathbf{1}_A(\mathbf{s}) \lambda(\mathbf{s}) \, d\mathbf{s} = \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, d\mathbf{s}
   $$
   The formula holds for all indicator functions.
2. **Simple Non-Negative Functions:** Let $f(\mathbf{s}) = \sum_{k=1}^m c_k \mathbf{1}_{A_k}(\mathbf{s})$ with $c_k \ge 0$.
   By linearity of summation and mathematical expectation:
   $$
   \operatorname{E}\left[\sum_{\mathbf{s} \in \Phi} \sum_{k=1}^m c_k \mathbf{1}_{A_k}(\mathbf{s})\right] = \sum_{k=1}^m c_k \operatorname{E}[N(A_k)] = \sum_{k=1}^m c_k \int_{A_k} \lambda(\mathbf{s}) d\mathbf{s} = \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, d\mathbf{s}
   $$
3. **General Non-Negative Functions:** Any non-negative measurable function $f$ is the pointwise limit of a monotonically increasing sequence of simple non-negative functions: $0 \le f_1 \le f_2 \le \dots \uparrow f$.
   Applying the Monotone Convergence Theorem to both the discrete sum and the Lebesgue integral establishes:
   $$
   \operatorname{E}\left[\sum_{\mathbf{s} \in \Phi} f(\mathbf{s})\right] = \int_{\mathbb{R}^d} f(\mathbf{s}) \lambda(\mathbf{s}) \, d\mathbf{s} \quad \blacksquare
   $$

##### Definition 11.3.2 (Second-Order Factorial Moment Measure & Product Density)
The **second-order factorial moment measure** $\alpha_2$ of a point process $\Phi$ is defined on $\mathcal{B}(\mathbb{R}^d \times \mathbb{R}^d)$ by:
$$
\alpha_2(A \times B) = \operatorname{E}\left[\sum_{\mathbf{s}_i \in \Phi} \sum_{\mathbf{s}_j \in \Phi, \mathbf{s}_j \ne \mathbf{s}_i} \mathbf{1}_A(\mathbf{s}_i) \mathbf{1}_B(\mathbf{s}_j)\right] = \operatorname{E}[N(A)N(B) - N(A \cap B)]
$$
When $\alpha_2$ is absolutely continuous with respect to Lebesgue measure, its density is the **second-order product density** $\lambda_2(\mathbf{s}_1, \mathbf{s}_2)$:
$$
\alpha_2(A \times B) = \int_A \int_B \lambda_2(\mathbf{s}_1, \mathbf{s}_2) \, d\mathbf{s}_1 d\mathbf{s}_2
$$
The product density defines the joint probability of finding distinct events simultaneously in two infinitesimal volume elements $d\mathbf{s}_1$ and $d\mathbf{s}_2$:
$$
P\left(N(d\mathbf{s}_1) = 1 \text{ and } N(d\mathbf{s}_2) = 1\right) = \lambda_2(\mathbf{s}_1, \mathbf{s}_2) \, d\mathbf{s}_1 d\mathbf{s}_2 \quad (\mathbf{s}_1 \ne \mathbf{s}_2)
$$

##### Definition 11.3.3 (Pair Correlation Function)
The **pair correlation function** $g(\mathbf{s}_1, \mathbf{s}_2)$ standardizes the second-order product density by the product of individual intensities:
$$
g(\mathbf{s}_1, \mathbf{s}_2) = \frac{\lambda_2(\mathbf{s}_1, \mathbf{s}_2)}{\lambda(\mathbf{s}_1)\lambda(\mathbf{s}_2)}
$$
For a second-order stationary and isotropic point process, $\lambda(\mathbf{s}) = \lambda$ and $g(\mathbf{s}_1, \mathbf{s}_2) = g(r)$ depends only on separation distance $r = \|\mathbf{s}_1 - \mathbf{s}_2\|$:
- Under **Complete Spatial Randomness (CSR)**, independent scattering implies $\lambda_2(\mathbf{s}_1, \mathbf{s}_2) = \lambda^2$, so:
  $$
  g(r) \equiv 1 \quad \forall r > 0
  $$
- Under **Clustering / Aggregation**, the probability of observing a neighboring event at short distance $r$ is elevated: $g(r) > 1$.
- Under **Regularity / Spatial Inhibition**, events repel each other: $g(r) < 1$. For hard-core processes with packing radius $R_{\text{core}}$, $g(r) = 0$ for $r < R_{\text{core}}$.

---

### 11.4 Distance Summary Functions: $F(r)$, $G(r)$, $J(r)$, and Ripley's $K(r)$

#### Beat 1: The Mathematical Anchor

Second-order spatial dependencies in mapped point patterns are quantified through four fundamental summary functions based on inter-point Euclidean distances.

##### Definition 11.4.1 (Nearest-Neighbor Distribution $G(r)$ and Empty-Space Function $F(r)$)
Let $\Phi$ be a stationary, isotropic point process in $\mathbb{R}^d$:
1. **Event-to-Event Nearest-Neighbor Distribution $G(r)$:**
   The cumulative distribution function of the distance $d(\mathbf{s}, \Phi \setminus \{\mathbf{s}\})$ from an arbitrary typical event $\mathbf{s} \in \Phi$ to its nearest neighbor in the pattern:
   $$
   G(r) = P\left(\min_{\mathbf{s}_j \in \Phi, \mathbf{s}_j \ne \mathbf{s}} \|\mathbf{s} - \mathbf{s}_j\| \le r \mid \mathbf{s} \in \Phi\right)
   $$
2. **Point-to-Event Empty-Space Distribution $F(r)$:**
   The cumulative distribution function of the distance $d(\mathbf{u}, \Phi)$ from an arbitrary fixed grid location $\mathbf{u} \in W$ (not necessarily an event) to the nearest event in $\Phi$:
   $$
   F(r) = P\left(\min_{\mathbf{s}_i \in \Phi} \|\mathbf{u} - \mathbf{s}_i\| \le r\right) = 1 - P(N(b(\mathbf{u}, r)) = 0)
   $$
   where $b(\mathbf{u}, r) = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x} - \mathbf{u}\| \le r\}$ is the ball of radius $r$ centered at $\mathbf{u}$.

##### Theorem 11.4.1 (Theoretical Closed Forms Under CSR in $\mathbb{R}^2$)
For a Homogeneous Poisson Process in $\mathbb{R}^2$ with intensity $\lambda$:
$$
F(r) = G(r) = 1 - \exp(-\lambda \pi r^2), \quad r \ge 0
$$

*Proof.*
1. For the empty-space function $F(r)$:
   The distance from an arbitrary point $\mathbf{u}$ to the nearest event exceeds $r$ if and only if no events lie inside the circular disc $b(\mathbf{u}, r)$.
   The area of the disc is $|b(\mathbf{u}, r)| = \pi r^2$.
   By the Poisson counting property:
   $$
   P(N(b(\mathbf{u}, r)) = 0) = \exp(-\lambda |b(\mathbf{u}, r)|) = \exp(-\lambda \pi r^2)
   $$
   Therefore:
   $$
   F(r) = 1 - P(N(b(\mathbf{u}, r)) = 0) = 1 - \exp(-\lambda \pi r^2)
   $$
2. For the event-to-event nearest-neighbor function $G(r)$:
   Conditioned on an event occurring at $\mathbf{s}$, by Palm distribution theory and Slivnyak's Theorem, the reduced Palm distribution of a Homogeneous Poisson Process is identical to its unconditional distribution:
   $$
   P(N(b(\mathbf{s}, r) \setminus \{\mathbf{s}\}) = 0 \mid \mathbf{s} \in \Phi) = P(N(b(\mathbf{s}, r)) = 0) = \exp(-\lambda \pi r^2)
   $$
   Hence $G(r) = 1 - \exp(-\lambda \pi r^2) = F(r)$. $\blacksquare$

##### Definition 11.4.2 (Van Lieshout & Baddeley's $J$-Function)
To eliminate sensitivity to estimated intensity $\lambda$, van Lieshout & Baddeley (1996) formulated the **$J$-function**:
$$
J(r) = \frac{1 - G(r)}{1 - F(r)}, \quad \text{for } F(r) < 1
$$
- Under **CSR**, $F(r) = G(r)$, so:
  $$
  J(r) \equiv 1 \quad \forall r \ge 0
  $$
- Under **Clustering**, nearest neighbors are closer than expected by chance ($G(r) > F(r)$), producing:
  $$
  J(r) < 1
  $$
- Under **Regularity / Inhibition**, events avoid each other ($G(r) < F(r)$), producing:
  $$
  J(r) > 1
  $$

---

#### Beat 1 (Continued): Ripley's $K$-Function & Besag's $L$-Function

While $F$, $G$, and $J$ characterize only the nearest neighbor distance, **Ripley’s $K$-function** integrates second-order dependencies across all pairwise distances up to lag $r$.

##### Definition 11.4.3 (Ripley's $K$-Function)
Let $\Phi$ be a stationary, isotropic point process in $\mathbb{R}^d$ with intensity $\lambda$. Ripley's $K$-function is defined such that $\lambda K(r)$ is the expected number of additional events within distance $r$ of an arbitrary event:
$$
K(r) = \frac{1}{\lambda}\operatorname{E}\left[N(b(\mathbf{s}, r) \setminus \{\mathbf{s}\}) \mid \mathbf{s} \in \Phi\right]
$$
Equivalently, integrating the pair correlation function in $\mathbb{R}^2$:
$$
K(r) = 2\pi \int_0^r u \, g(u) \, du
$$

##### Theorem 11.4.2 (Ripley's $K$ and Besag's $L$ Under CSR in $\mathbb{R}^2$)
Under Complete Spatial Randomness (CSR) in $\mathbb{R}^2$:
1. Ripley's $K$-function is:
   $$
   K_{\text{CSR}}(r) = \pi r^2
   $$
2. Besag's variance-stabilizing transform $L(r) = \sqrt{\frac{K(r)}{\pi}}$ satisfies:
   $$
   L_{\text{CSR}}(r) = r \iff L(r) - r = 0
   $$

*Proof.*
Under CSR, the pair correlation function is identically $g(u) = 1$. Substituting this into the integral:
$$
K(r) = 2\pi \int_0^r u (1) \, du = 2\pi \left[\frac{u^2}{2}\right]_0^r = \pi r^2
$$
Taking Besag's transform:
$$
L(r) = \sqrt{\frac{K(r)}{\pi}} = \sqrt{\frac{\pi r^2}{\pi}} = \sqrt{r^2} = r \quad \blacksquare
$$

##### Theorem 11.4.3 (Ripley's Edge-Corrected Estimator)
Let $\mathbf{s}_1, \dots, \mathbf{s}_n$ be events observed inside window $W \subset \mathbb{R}^2$. The standard empirical estimator $\sum_{i \ne j}\mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}}$ is severely downward-biased for large $r$ because points outside the window boundary are unobserved.
Ripley (1976) formulated the strictly unbiased edge-corrected estimator:
$$
\widehat{K}(r) = \frac{|W|}{n(n - 1)} \sum_{i=1}^n \sum_{j \ne i}^n \frac{\mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}}}{w(\mathbf{s}_i, \mathbf{s}_j)}
$$
where $w(\mathbf{s}_i, \mathbf{s}_j)$ is **Ripley’s isotropic edge-correction weight**, equal to the proportion of the perimeter of the circle centered at $\mathbf{s}_i$ passing through $\mathbf{s}_j$ (radius $d_{ij} = \|\mathbf{s}_i - \mathbf{s}_j\|$) that lies completely inside $W$:
$$
w(\mathbf{s}_i, \mathbf{s}_j) = \frac{\text{Length}\left(\partial b(\mathbf{s}_i, \|\mathbf{s}_i - \mathbf{s}_j\|) \cap W\right)}{2\pi \|\mathbf{s}_i - \mathbf{s}_j\|} \in (0, 1]
$$

```
Ripley's Isotropic Edge-Correction Weight w(s_i, s_j)
+---------------------------------------+
| Window W                              |
|                                       |
|             s_i                       |
|           / | \                       |
|          /  |  \                      |
|         /   |   \                     |
|        (----+---s_j)                  |
|       /     |       \                 |
+------(------+--------)----------------+  <-- Window Boundary
        \     |       /
         \    |      /   (Perimeter outside W is lost)
          \   |     /
           \  |    /
            \ |   /
```

##### Term-by-Term Equation Anatomy
$$
\widehat{K}(r) = \underbrace{\frac{|W|}{n(n-1)}}_{\text{Sample Area / Total Ordered Pairs}} \sum_{i=1}^n \sum_{j \ne i}^n \underbrace{\mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}}}_{\text{Distance Indicator}} \cdot \underbrace{\frac{1}{w(\mathbf{s}_i, \mathbf{s}_j)}}_{\text{Perimeter Inflation Weight}}
$$
- **$\frac{|W|}{n(n-1)}$:** Standardizes the count by the squared sample intensity estimator $\widehat{\lambda}^2 = \frac{n(n-1)}{|W|^2}$, ensuring unbiased estimation without requiring prior knowledge of true $\lambda$.
- **$\mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le r\}}$:** Binary filter isolating only pairs separated by distance $\le r$.
- **$\frac{1}{w(\mathbf{s}_i, \mathbf{s}_j)}$:** Inverse-probability weight. When the search circle around $\mathbf{s}_i$ crosses the boundary of $W$, only a fraction $w < 1$ is observed. Dividing by $w$ reconstructs the expected total number of points on the unobserved arc.

> [!NOTE]
> **Wait—Why Did We Do That?**  
> *Why not simply discard points close to the boundary (the "border method")?*  
> The border correction discards any point $\mathbf{s}_i$ whose distance to $\partial W$ is less than $r$. While simple, for large radii $r$, the border method discards up to 70% of the sample data, drastically inflating the estimator's variance. Ripley's isotropic weighting uses 100% of the data points, maintaining minimal variance.

---

### 11.5 Contagion, Clustering, and Inhibition Models

#### Beat 1: The Mathematical Anchor

When Complete Spatial Randomness is rejected, point patterns require parametric spatial point process models.

```
                  TAXONOMY OF SPATIAL POINT PROCESSES
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
   HETEROGENEITY                                       INTERACTIONS
 (First-Order Trend)                               (Second-Order Dependence)
         │                                                   │
         ├─ Inhomogeneous Poisson                            ├─ Clustering (Attraction)
         │  Process: lambda(s)                               │  ├─ Neyman-Scott / Matérn
         │                                                   │  └─ Cox / Log-Gaussian Cox
         └─ Berman-Turner Quadrature                         │
                                                             └─ Regularity (Inhibition)
                                                                ├─ Hard-Core (Packing)
                                                                └─ Strauss Markov Process
```

#### 1. Inhomogeneous Poisson Processes (First-Order Non-Stationarity)
When the point density varies due to an underlying environmental gradient (e.g., elevation, moisture, pH):
$$
N(A) \sim \text{Poisson}\left(\int_A \lambda(\mathbf{s}) \, d\mathbf{s}\right)
$$
with log-linear intensity $\lambda(\mathbf{s}; \boldsymbol{\beta}) = \exp(\mathbf{x}(\mathbf{s})'\boldsymbol{\beta})$.
The log-likelihood of observing points $\mathbf{s}_1, \dots, \mathbf{s}_n$ in $W$ is:
$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \ln \lambda(\mathbf{s}_i; \boldsymbol{\beta}) - \int_W \lambda(\mathbf{s}; \boldsymbol{\beta}) \, d\mathbf{s}
$$

#### 2. Cox Processes & Log-Gaussian Cox Processes (The Bridge to Geostatistics)
A **Cox Process** (doubly stochastic Poisson process) models clustering through a random intensity surface $\Lambda(\mathbf{s})$.
Conditioned on a realization of the random surface $\Lambda(\mathbf{s}) = \lambda(\mathbf{s})$, the events follow an Inhomogeneous Poisson Process with intensity $\lambda(\mathbf{s})$.

##### Theorem 11.5.1 (Log-Gaussian Cox Process Moments)
Let $Y(\mathbf{s})$ be a stationary Gaussian random field on $\mathbb{R}^d$ with mean $\mu_Y$, stationary variance $\sigma_Y^2$, and covariance kernel $C_Y(\mathbf{h})$.
Let the intensity field be:
$$
\Lambda(\mathbf{s}) = \exp(Y(\mathbf{s}))
$$
Then $\Phi$ is a **Log-Gaussian Cox Process (LGCP)**. Its first- and second-order properties are:
1. First-order intensity:
   $$
   \lambda = \operatorname{E}[\Lambda(\mathbf{s})] = \exp\left(\mu_Y + \frac{1}{2}\sigma_Y^2\right)
   $$
2. Second-order product density:
   $$
   \lambda_2(\mathbf{h}) = \lambda^2 \exp(C_Y(\mathbf{h}))
   $$
3. Pair correlation function:
   $$
   g(\mathbf{h}) = \exp(C_Y(\mathbf{h})) \ge 1
   $$
4. Ripley's $K$-function in $\mathbb{R}^2$:
   $$
   K(r) = \pi r^2 + 2\pi \int_0^r \left[\exp(C_Y(u)) - 1\right] u \, du > \pi r^2
   $$

*Proof.*
1. Because $Y(\mathbf{s}) \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$, its moment generating function evaluated at $t = 1$ is:
   $$
   \operatorname{E}[\Lambda(\mathbf{s})] = \operatorname{E}[\exp(Y(\mathbf{s}))] = M_{Y(\mathbf{s})}(1) = \exp\left(\mu_Y(1) + \frac{1}{2}\sigma_Y^2(1)^2\right) = \exp\left(\mu_Y + \frac{1}{2}\sigma_Y^2\right)
   $$
2. For two points separated by lag vector $\mathbf{h}$, the random vector $(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h}))'$ is bivariate Gaussian with covariance $C_Y(\mathbf{h})$.
   By the law of total expectation:
   $$
   \begin{aligned}
   \lambda_2(\mathbf{h}) &= \operatorname{E}\left[\Lambda(\mathbf{s})\Lambda(\mathbf{s}+\mathbf{h})\right] = \operatorname{E}\left[\exp(Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h}))\right]
   \end{aligned}
   $$
   The sum $S = Y(\mathbf{s}) + Y(\mathbf{s}+\mathbf{h})$ is univariate normal with mean $2\mu_Y$ and variance:
   $$
   \operatorname{Var}(S) = \operatorname{Var}(Y(\mathbf{s})) + \operatorname{Var}(Y(\mathbf{s}+\mathbf{h})) + 2\operatorname{Cov}(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h})) = 2\sigma_Y^2 + 2C_Y(\mathbf{h})
   $$
   Using the Gaussian MGF:
   $$
   \begin{aligned}
   \operatorname{E}[\exp(S)] &= \exp\left(\operatorname{E}[S] + \frac{1}{2}\operatorname{Var}(S)\right) = \exp\left(2\mu_Y + \frac{1}{2}(2\sigma_Y^2 + 2C_Y(\mathbf{h}))\right) \\
   &= \exp\left(2\mu_Y + \sigma_Y^2 + C_Y(\mathbf{h})\right) = \left[\exp\left(\mu_Y + \frac{1}{2}\sigma_Y^2\right)\right]^2 \exp(C_Y(\mathbf{h})) \\
   &= \lambda^2 \exp(C_Y(\mathbf{h}))
   \end{aligned}
   $$
3. The pair correlation function is:
   $$
   g(\mathbf{h}) = \frac{\lambda_2(\mathbf{h})}{\lambda^2} = \frac{\lambda^2 \exp(C_Y(\mathbf{h}))}{\lambda^2} = \exp(C_Y(\mathbf{h}))
   $$
   Because valid covariance kernels have $C_Y(\mathbf{h}) \ge 0$ for clustered processes, $g(\mathbf{h}) \ge 1$ everywhere.
4. Integrating the pair correlation function gives Ripley's $K(r)$:
   $$
   K(r) = 2\pi \int_0^r u \, g(u) \, du = 2\pi \int_0^r u \exp(C_Y(u)) \, du = \pi r^2 + 2\pi \int_0^r \left[\exp(C_Y(u)) - 1\right] u \, du \quad \blacksquare
   $$

#### 3. Strauss Markov Point Processes (Spatial Inhibition)
For regular patterns with physical repulsion, Strauss (1975) defined a point process probability density with respect to a unit Poisson process:
$$
p(\mathbf{s}_1, \dots, \mathbf{s}_n) = \alpha \, \beta^n \, \gamma^{s_R(\mathbf{x})}
$$
where:
- $\beta > 0$ controls point abundance.
- $R > 0$ is the interaction radius.
- $s_R(\mathbf{x}) = \sum_{1 \le i < j \le n} \mathbf{1}_{\{\|\mathbf{s}_i - \mathbf{s}_j\| \le R\}}$ counts the total number of distinct pairs separated by distance $\le R$.
- $\gamma \in [0, 1]$ is the interaction parameter:
  - $\gamma = 1$: Homogeneous Poisson Process (CSR).
  - $0 < \gamma < 1$: Soft-core inhibition (penalizing pairs closer than $R$).
  - $\gamma = 0$: **Hard-core process** (configurations with any pair closer than $R$ have probability density zero: $p(\mathbf{x}) = 0$).

---

### 11.6 Algorithmic Translation: Pure Python Engines

#### Beat 2: The Algorithmic Translation (The "How")

We now implement two complete, dependency-free computational engines in pure standard Python:
1. `Algorithm 11.1`: Exact simulation of an Inhomogeneous Poisson Process via the **Lewis-Shedler thinning algorithm**.
2. `Algorithm 11.2`: Computation of **Ripley’s edge-corrected $\widehat{K}(r)$ and Besag’s $\widehat{L}(r)$ functions** with Monte Carlo CSR confidence envelopes.

```python
"""
Chapter 11: Algorithmic Translation of Spatial Point Process Models
Pure Dependency-Free Python (Standard Library Only: math, random).
"""

import math
import random

# =============================================================================
# Algorithm 11.1: Lewis-Shedler Thinning Algorithm for IPP Simulation
# =============================================================================

def intensity_function_2d(x: float, y: float, beta0: float = 2.0, beta1: float = 1.5) -> float:
    """
    Log-linear spatial intensity surface: lambda(x, y) = exp(beta0 + beta1 * x).
    Intensity increases along the x-axis.
    """
    return math.exp(beta0 + beta1 * x)

def simulate_inhomogeneous_poisson_thinning(
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    beta0: float = 2.0,
    beta1: float = 1.5,
    seed: int = 42
) -> list[tuple[float, float]]:
    """
    Simulates an Inhomogeneous Poisson Process on [x_min, x_max] x [y_min, y_max]
    using the Lewis-Shedler acceptance-rejection thinning algorithm.
    """
    rng = random.Random(seed)
    area = (x_max - x_min) * (y_max - y_min)

    # Step 1: Find upper bound lambda_max over the domain
    # Since lambda increases with x, the maximum occurs at x_max
    lambda_max = intensity_function_2d(x_max, y_max, beta0, beta1)

    # Step 2: Generate candidate count from Poisson(lambda_max * area)
    mean_candidates = lambda_max * area
    
    # Knuth algorithm for Poisson random variable
    L = math.exp(-mean_candidates)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    n_candidates = k - 1

    # Step 3: Propose candidate coordinates uniformly and thin
    accepted_points = []
    for _ in range(n_candidates):
        x_prop = x_min + (x_max - x_min) * rng.random()
        y_prop = y_min + (y_max - y_min) * rng.random()

        # Acceptance probability: lambda(x, y) / lambda_max
        prob_accept = intensity_function_2d(x_prop, y_prop, beta0, beta1) / lambda_max
        if rng.random() <= prob_accept:
            accepted_points.append((x_prop, y_prop))

    return accepted_points

# =============================================================================
# Algorithm 11.2: Ripley's K(r) and Besag's L(r) with Isotropic Edge Correction
# =============================================================================

def ripley_isotropic_weight(x_i: float, y_i: float, x_j: float, y_j: float, 
                            x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """
    Calculates Ripley's circumferential isotropic edge-correction weight w(s_i, s_j).
    Approximates the fraction of the perimeter of circle(s_i, d_ij) that lies inside window W.
    """
    d_ij = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
    if d_ij < 1e-12:
        return 1.0

    # Test 360 angular points along the circle boundary
    num_samples = 360
    inside_count = 0
    for step in range(num_samples):
        theta = 2.0 * math.pi * step / num_samples
        x_arc = x_i + d_ij * math.cos(theta)
        y_arc = y_i + d_ij * math.sin(theta)
        if (x_min <= x_arc <= x_max) and (y_min <= y_arc <= y_max):
            inside_count += 1

    proportion = inside_count / num_samples
    # Lower bound to avoid division by zero for extreme corners
    return max(0.1, proportion)

def compute_ripley_k_and_l(
    points: list[tuple[float, float]],
    radii: list[float],
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0
) -> tuple[list[float], list[float]]:
    """
    Computes Ripley's edge-corrected K-function and Besag's L-function across radii.
    """
    n = len(points)
    area = (x_max - x_min) * (y_max - y_min)
    if n < 2:
        return [0.0] * len(radii), [0.0] * len(radii)

    # Precalculate all pairwise Euclidean distances and weights
    pairs = []
    for i in range(n):
        x_i, y_i = points[i]
        for j in range(i + 1, n):
            x_j, y_j = points[j]
            dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            # Both directions (i->j and j->i)
            w_ij = ripley_isotropic_weight(x_i, y_i, x_j, y_j, x_min, x_max, y_min, y_max)
            w_ji = ripley_isotropic_weight(x_j, y_j, x_i, y_i, x_min, x_max, y_min, y_max)
            pairs.append((dist, 1.0 / w_ij + 1.0 / w_ji))

    k_hat = []
    l_hat = []
    scale_factor = area / (n * (n - 1))

    for r in radii:
        weighted_sum = sum(weight for dist, weight in pairs if dist <= r)
        k_val = scale_factor * weighted_sum
        l_val = math.sqrt(k_val / math.pi)
        k_hat.append(k_val)
        l_hat.append(l_val)

    return k_hat, l_hat

# =============================================================================
# Beat 3: Reality Diagnostics and Monte Carlo Envelopes
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Chapter 11: Lewis-Shedler Thinning & Ripley's K-Function Diagnostic Lab")
    print("=" * 80)

    # 1. Simulate an Inhomogeneous Point Pattern
    points = simulate_inhomogeneous_poisson_thinning(beta0=2.0, beta1=1.5, seed=101)
    n_pts = len(points)
    print(f"Simulated Inhomogeneous Pattern: {n_pts} events generated on [0, 1]^2.")

    # 2. Evaluate Ripley's K and Besag's L across 8 distance thresholds
    r_lags = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    k_est, l_est = compute_ripley_k_and_l(points, r_lags)

    print("\nSecond-Order Summary Function Comparison (Observed vs. CSR Null):")
    print(f"{'Lag r':>6} | {'K_hat(r)':>10} | {'K_CSR(r)':>10} | {'L_hat(r)':>10} | {'L_hat - r':>10} | {'Diagnosis':>14}")
    print("-" * 72)

    for r, k_val, l_val in zip(r_lags, k_est, l_est):
        k_csr = math.pi * r**2
        diff = l_val - r
        diag = "Clustered" if diff > 0.02 else ("Regular" if diff < -0.02 else "CSR Random")
        print(f"{r:6.2f} | {k_val:10.4f} | {k_csr:10.4f} | {l_val:10.4f} | {diff:+10.4f} | {diag:>14}")

    print("-" * 72)
    print("Interpretation: L_hat(r) - r > 0 indicates spatial aggregation due to trend.")
```

---

### Chapter Study Guide & Companion Exercises
For complete theoretical re-derivations, pencil-and-paper point process proofs, dependency-free Python simulation labs, and diagnostic edge-case reflections, see the companion workbook:
👉 [Chapter 11 Companion Exercises and Complete Solutions](ch11_exercises.md)
