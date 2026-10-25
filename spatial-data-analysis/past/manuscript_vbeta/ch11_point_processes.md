# Chapter 11 — Mapped Point Patterns and Spatial Point Processes

In geostatistics (Chapters 1–10), the observation locations $\mathbf{s} \in \mathbb{R}^d$ are treated as fixed, deterministic spatial coordinates, while the attribute value $Z(\mathbf{s})$ is modeled as a random variable. In spatial point pattern analysis, this paradigm is inverted: the spatial locations themselves are the random phenomenon under investigation. The fundamental question is not *"what is the value of the field at this known location?"*, but rather *"why did an event occur here rather than somewhere else, and does the occurrence of one event encourage or inhibit the occurrence of others?"*

By the end of this chapter, you will be able to inspect a spatial map of discrete events—trees in a forest stand, disease cases in an epidemiological outbreak, hydrothermal vents along a mid-ocean ridge—quantify whether the pattern exhibits complete spatial randomness, clustering, or regular inhibition, calculate rigorous edge-corrected summary statistics with simulation envelopes, and avoid confounding first-order environmental trends with second-order point-to-point interactions.

---

## 11.1 From 1D Arrival Counts to Spatial Point Measures

In classical temporal probability, a one-dimensional Poisson process models the arrival of discrete events over time, such as telephone calls entering a switchboard or particles striking a Geiger counter. If arrivals occur randomly at a constant rate of $\lambda$ events per unit time, the count of events in any time interval $(t_1, t_2]$ follows a Poisson distribution:

$$
N((t_1, t_2]) \sim \text{Poisson}(\lambda (t_2 - t_1))
$$

with the foundational property of **independent increments**: counts in non-overlapping time intervals are statistically independent random variables.

In spatial point process theory, the one-dimensional time line $\mathbb{R}^+$ is generalized to a spatial domain $\mathbb{R}^d$. While the core probability theory holds for any dimension $d \ge 1$, we focus primarily on the spatial plane $\mathbb{R}^2$. The interval length $(t_2 - t_1)$ is replaced by the two-dimensional planar area $|A|$ of a planar region $A \subset \mathbb{R}^2$.

```
1D Temporal Process: Counting on Intervals
0 -------|-------|---|---------|-----> Time t
         t1      t2  t3        t4
         Interval (t1, t2] has length Delta t

2D Spatial Point Process: Counting Measures on Planar Sets
+------------------------------------+
|                                    |
|         .       /-------\          |
|      .         /    .  . \   A     |
|               |   .   .   |        |
|                \   .     /         |
|                 \-------/          |
|            .                       |
|                      .             |
|   W (Observation Window)           |
+------------------------------------+
```

### Observation Windows and Spatial Counting Measures

In practical applications, we never observe a point process across the entire infinite plane $\mathbb{R}^2$. Instead, we record events through a bounded, planar **observation window** $W \subset \mathbb{R}^2$ with finite area $|W| < \infty$. The point pattern observed within $W$ is a realization of a random counting measure.

```
[Definition 11.1.1: Spatial Point Process and Counting Measure]
Let W \subset \mathbb{R}^2 be a bounded planar observation window with finite area |W| > 0.
A spatial point process \Phi on W is a random collection of points:
    \Phi = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_{N(W)}\} \subset W
where the total number of events N(W) is an integer-valued random variable.

For any Borel subset A \subseteq W, the spatial counting measure N(A) denotes 
the random number of events falling within region A:
    N(A) = \sum_{i=1}^{N(W)} \mathbb{I}(\mathbf{x}_i \in A)

A point process is said to be simple if multiple coincidental events are impossible:
    P(N(\{\mathbf{s}\}) \le 1 \text{ for all } \mathbf{s} \in W) = 1
```

Void probabilities describe the probability that a test region contains zero events. For any compact test set $K \subset W$, the void probability is:

$$
v(K) = P(N(K) = 0)
$$

Under mild regularity conditions, the family of void probabilities across all compact subsets uniquely characterizes the distribution of a simple point process. As we will see in Section 11.4, void probabilities over circular test discs directly yield the empty-space distribution function $F(r)$.

### The Three Canonical Spatial Archetypes

Every spatial point pattern is governed by two distinct components:
1. **First-order intensity:** The large-scale spatial variation in event density across the window, driven by external environmental conditions or spatial gradients.
2. **Second-order interaction:** The fine-scale point-to-point dependence (attraction or repulsion) between neighboring events.

When examining a mapped point pattern, statisticians classify the spatial arrangement into one of three canonical archetypes:

```
Complete Spatial Randomness (CSR)     Clustered (Aggregated)             Regular (Inhibited)
+---------------------------------+  +---------------------------------+  +---------------------------------+
|   .      .         .            |  |           ..                    |  |   .     .     .     .     .   |
|         .     .        .        |  |          ...                    |  |                                 |
|   .               .      .      |  |                                 |  |      .     .     .     .      |
|             .                   |  |                 ....            |  |                                 |
|      .         .       .        |  |                 ...             |  |   .     .     .     .     .   |
|   .        .                    |  |                                 |  |                                 |
|               .      .       .  |  |    ...                          |  |      .     .     .     .      |
|      .   .                      |  |    ..                           |  |                                 |
+---------------------------------+  +---------------------------------+  +---------------------------------+
 Events occur independently and       Events attract one another or        Events repel one another through
 uniformly across the domain.         share localized fertile niches.      resource competition or territory.
```

- **Complete Spatial Randomness (CSR):** Events are placed independently and uniformly across $W$. Points exhibit neither mutual attraction nor mutual repulsion. CSR serves as the universal null hypothesis in spatial point pattern analysis.
- **Clustered (Aggregated):** Events group together in tight spatial clumps more frequently than expected under CSR. Clustering arises either from mutual attraction (e.g., seed dispersal around parent trees, contagious transmission of infections) or from an unobserved environmental covariate (e.g., mineral deposits along geological fault lines).
- **Regular (Inhibited):** Events are more evenly spaced than expected under CSR. Regularity arises from repulsive interactions, such as territorial behavior in animal nesting, root competition among desert shrubs, or the physical non-overlapping packing of cell nuclei.

---

> [!NOTE]
> ### From the Lecture Hall: The Poisson Distribution and Exponential Inter-arrivals
> In a 1D temporal Poisson process with rate $\lambda$, the waiting time $T$ between consecutive arrivals is an Exponential random variable with rate $\lambda$:
>
> $$
> P(T > t) = P(N((0, t]) = 0) = \exp(-\lambda t)
> $$
>
> The exponential inter-arrival distribution is memoryless. In $\mathbb{R}^2$, the counterpart to the waiting time is the **empty-space distance**: the distance $R$ from an arbitrary fixed reference point $\mathbf{u}$ to the nearest event in $\Phi$. The probability that no event falls within distance $r$ is the void probability of the disc $b(\mathbf{u}, r)$ of area $\pi r^2$:
>
> $$
> P(R > r) = P(N(b(\mathbf{u}, r)) = 0) = \exp(-\lambda \pi r^2)
> $$
>
> Just as the 1D inter-arrival distribution is exponential in time $t$, the 2D spatial empty-space distribution is exponential in the area $\pi r^2$.

---

## 11.2 Complete Spatial Randomness (CSR) and the Homogeneous Poisson Process

The theoretical gold standard of point pattern analysis is Complete Spatial Randomness (CSR), mathematically formalized as the **Homogeneous Poisson Process (HPP)**.

```
[Definition 11.2.1: Homogeneous Poisson Process (CSR)]
A spatial point process \Phi on \mathbb{R}^2 is a Homogeneous Poisson Process with constant
intensity parameter \lambda > 0 if it satisfies two fundamental postulates:
1. Poisson-Distributed Counts: For any bounded planar region A \subset \mathbb{R}^2 with area |A|,
   the number of events N(A) follows a Poisson distribution with mean \lambda |A|:
       P(N(A) = k) = \frac{(\lambda |A|)^k}{k!} e^{-\lambda |A|}, \quad k = 0, 1, 2, \dots
2. Complete Spatial Independence: For any finite collection of mutually disjoint 
   regions A_1, A_2, \dots, A_m \subset \mathbb{R}^2:
       N(A_1), N(A_2), \dots, N(A_m) \text{ are mutually independent random variables.}
```

### The Conditional-Uniform Property

The independent-increments property leads to a foundational result: if we condition on observing exactly $n$ events in an observation window $W$, the locations of those $n$ points are independent and uniformly distributed across $W$.

```
[Theorem 11.2.1: Conditional Distribution of Points Under CSR]
Let \Phi be a Homogeneous Poisson Process with intensity \lambda on a bounded window W \subset \mathbb{R}^2.
Conditioned on observing exactly n events in W, N(W) = n:
1. The locations of the n points \{\mathbf{x}_1, \dots, \mathbf{x}_n\} are conditionally independent 
   and uniformly distributed over W, with joint probability density:
       f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) = \frac{1}{|W|^n}
2. For any subregion A \subseteq W, the conditional count N(A) given N(W) = n follows a
   Binomial distribution with index n and success probability p = |A| / |W|:
       N(A) \mid (N(W) = n) \sim \text{Binomial}\left(n, \frac{|A|}{|W|}\right)
```

**Proof:**

*Part 1: Joint Density of Point Locations.*  
Consider $n$ disjoint, infinitesimal planar regions $B_1, \dots, B_n \subset W$ centered at coordinates $\mathbf{s}_1, \dots, \mathbf{s}_n$, with areas $|B_1| = d\mathbf{s}_1, \dots, |B_n| = d\mathbf{s}_n$. Define the remainder set $B_0 = W \setminus \bigcup_{i=1}^n B_i$, whose area is $|B_0| = |W| - \sum_{i=1}^n d\mathbf{s}_i$.

The event that exactly one point falls into each of the $n$ infinitesimal regions and no points fall anywhere else in $W$ corresponds to the joint counting outcome:

$$
E = \{N(B_1) = 1, N(B_2) = 1, \dots, N(B_n) = 1, N(B_0) = 0\}
$$

By Postulate 2 (spatial independence of disjoint regions), the probability of $E$ factorizes:

$$
P(E) = \left[ \prod_{i=1}^n P(N(B_i) = 1) \right] P(N(B_0) = 0)
$$

Applying the Poisson probability mass function from Postulate 1:

$$
P(N(B_i) = 1) = \lambda |B_i| e^{-\lambda |B_i|} = \lambda d\mathbf{s}_i e^{-\lambda d\mathbf{s}_i}
$$

$$
P(N(B_0) = 0) = e^{-\lambda |B_0|} = \exp\left(-\lambda \left( |W| - \sum_{i=1}^n d\mathbf{s}_i \right)\right)
$$

Multiplying these expressions:

$$
P(E) = \left( \prod_{i=1}^n \lambda d\mathbf{s}_i \right) \exp\left( -\lambda \sum_{i=1}^n d\mathbf{s}_i \right) \exp\left( -\lambda |W| + \lambda \sum_{i=1}^n d\mathbf{s}_i \right) = \lambda^n e^{-\lambda |W|} \prod_{i=1}^n d\mathbf{s}_i
$$

Now, we condition on the event that the total count in $W$ is $N(W) = n$. Because the $n$ observed points can arrive in any of $n!$ possible permutations among the $n$ distinct regions $B_1, \dots, B_n$, the joint probability that the unordered point pattern $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$ occupies these regions is $n! P(E)$. Therefore:

$$
f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) \prod_{i=1}^n d\mathbf{s}_i = \frac{n! P(E)}{P(N(W) = n)}
$$

Substituting $P(N(W) = n) = \frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}$:

$$
f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) \prod_{i=1}^n d\mathbf{s}_i = \frac{n! \left(\lambda^n e^{-\lambda |W|} \prod_{i=1}^n d\mathbf{s}_i \right)}{\frac{\lambda^n |W|^n}{n!} e^{-\lambda |W|}} = \frac{1}{|W|^n} \prod_{i=1}^n d\mathbf{s}_i
$$

For an ordered sequence of independent coordinates, this yields:

$$
f(\mathbf{s}_1, \dots, \mathbf{s}_n \mid N(W) = n) = \frac{1}{|W|^n} = \prod_{i=1}^n \left( \frac{1}{|W|} \right)
$$

This completes the proof of Part 1: conditionally on $N(W) = n$, the points are independent uniform draws over $W$.

*Part 2: Binomial Counting Corollary.*  
For any subregion $A \subseteq W$, the complement region $A^c = W \setminus A$ is disjoint from $A$, with $|A^c| = |W| - |A|$. By independence of disjoint sets:

$$
P(N(A) = k \mid N(W) = n) = \frac{P(N(A) = k \text{ and } N(A^c) = n - k)}{P(N(W) = n)}
$$

Substituting the Poisson probabilities for each disjoint region:

$$
\begin{aligned}
P(N(A) = k \mid N(W) = n) &= \frac{\left( \frac{(\lambda |A|)^k}{k!} e^{-\lambda |A|} \right) \left( \frac{(\lambda (|W| - |A|))^{n-k}}{(n-k)!} e^{-\lambda (|W| - |A|)} \right)}{\frac{(\lambda |W|)^n}{n!} e^{-\lambda |W|}} \\
&= \frac{n!}{k!(n-k)!} \frac{\lambda^k |A|^k \lambda^{n-k} (|W| - |A|)^{n-k}}{\lambda^n |W|^n} \frac{e^{-\lambda |W|}}{e^{-\lambda |W|}} \\
&= \binom{n}{k} \left( \frac{|A|}{|W|} \right)^k \left( 1 - \frac{|A|}{|W|} \right)^{n-k}
\end{aligned}
$$

This is the probability mass function of a $\text{Binomial}(n, p)$ distribution with success probability $p = |A| / |W|$, which completes the proof. $\blacksquare$

---

### The Algorithmic Payoff: Exact Simulation of CSR

Theorem 11.2.1 is not merely an abstract mathematical curiosity; it is the **exact, constructive recipe for simulating Complete Spatial Randomness**:

```
Algorithm: Exact Simulation of Homogeneous Poisson Process on W
1. Draw the total number of events N(W) ~ Poisson(\lambda |W|).
2. If N(W) = 0, return the empty pattern \emptyset.
3. Sample N(W) points independently and uniformly at random from window W:
       x_i ~ Uniform(x_min, x_max),  y_i ~ Uniform(y_min, y_max)
4. Return the point pattern \{\mathbf{x}_1, ..., \mathbf{x}_{N(W)}\}.
```

Because sampling uniform coordinates on a rectangle requires simple scalar pseudorandom numbers, CSR patterns can be simulated at virtually zero computational cost.

---

> [!NOTE]
> ### From the Lecture Hall: The Binomial Distribution as a Conditioned Poisson Process
> Students often wonder why the Binomial distribution appears inside a spatial Poisson process. Consider $n$ independent coin flips, each with success probability $p$. The total number of heads is $\text{Binomial}(n, p)$.
>
> Now consider a spatial Poisson process on $W$. When an observer fixes the total count at $N(W) = n$, each of the $n$ events independently falls into subregion $A$ with geometric probability $p = |A| / |W|$, and into $W \setminus A$ with probability $1 - p$. Conditioning a Poisson process on its total count transforms it directly into a sequence of $n$ independent Bernoulli trials.

---

### Testing CSR: Quadrat Counts and the Index of Dispersion

The simplest diagnostic test for CSR partitions the window $W$ into a regular grid of $Q$ non-overlapping subregions (quadrats) $\{A_1, A_2, \dots, A_Q\}$ of equal area $|A_j| = |W| / Q$.

Let $X_j = N(A_j)$ denote the observed event count in quadrat $j$, with total count $n = \sum_{j=1}^Q X_j$. Under the null hypothesis of CSR:
- Each quadrat count follows a Poisson distribution with mean $\mu = \lambda |W| / Q$.
- For a Poisson random variable, the theoretical variance equals the theoretical mean: $\mathbb{E}[X_j] = \operatorname{Var}(X_j) = \mu$.

The sample mean $\bar{X}$ and sample variance $s^2_X$ across the $Q$ quadrats are:

$$
\bar{X} = \frac{1}{Q} \sum_{j=1}^Q X_j = \frac{n}{Q}, \qquad s^2_X = \frac{1}{Q - 1} \sum_{j=1}^Q (X_j - \bar{X})^2
$$

The sample **Index of Dispersion** $I$ is the ratio of sample variance to sample mean:

$$
I = \frac{s^2_X}{\bar{X}} = \frac{\sum_{j=1}^Q (X_j - \bar{X})^2}{(Q - 1) \bar{X}}
$$

```
[Diagnostic Rule: The Index of Dispersion Test]
Under the null hypothesis of CSR:
    (Q - 1) I = \sum_{j=1}^Q \frac{(X_j - \bar{X})^2}{\bar{X}} \sim \chi^2_{Q - 1}
The test statistic follows an asymptotic chi-square distribution with Q - 1 degrees of freedom.
The reduction by 1 degree of freedom arises because the quadrats are conditioned on the 
fixed total sum \sum_{j=1}^Q X_j = n (Theorem 11.2.1 multinomial constraint).

Interpretation:
- I \approx 1: Consistent with Complete Spatial Randomness.
- I > 1 (Upper Tail): Overdispersion (variance exceeds mean). Indicates spatial clustering 
  OR a first-order intensity gradient.
- I < 1 (Lower Tail): Underdispersion (variance is less than mean). Indicates regular spacing 
  or spatial inhibition.
```

> [!WARNING]
> ### Crucial Caution: Confounding Trend with Clustering
> An index of dispersion $I > 1$ rejects the null hypothesis of CSR, but it **does not prove point-to-point clustering**. 
> A spatial trend in intensity—for example, a forest where soil moisture causes tree density to rise smoothly from west to east—will cause quadrat counts in the east to be large and quadrat counts in the west to be small. This inflates the sample variance $s^2_X$, producing $I \gg 1$ even though individual trees are placed completely independently of one another. As we will see throughout this chapter, distinguishing a first-order intensity trend from second-order point attraction is the central challenge of point pattern analysis.

---

## 11.3 First- and Second-Order Characteristics

To distinguish between large-scale spatial trends and fine-scale point interactions, we formalize the first- and second-order properties of point processes.

### First-Order Intensity

The **first-order intensity function** $\lambda(\mathbf{s})$ represents the expected number of events per unit area at location $\mathbf{s} \in \mathbb{R}^2$.

```
[Definition 11.3.1: First-Order Intensity Function]
The intensity function \lambda(\mathbf{s}) of a point process \Phi at location \mathbf{s} is:
    \lambda(\mathbf{s}) = \lim_{|b(\mathbf{s}, dr)| \to 0} \frac{\mathbb{E}[N(b(\mathbf{s}, dr))]}{|b(\mathbf{s}, dr)|}
where b(\mathbf{s}, dr) is an infinitesimal disc centered at \mathbf{s} with area |b(\mathbf{s}, dr)|.

The expected number of events falling within any arbitrary planar region A \subset W is:
    \mu(A) = \mathbb{E}[N(A)] = \int_A \lambda(\mathbf{s}) \, d\mathbf{s}
```

- When $\lambda(\mathbf{s}) \equiv \lambda$ is constant across the entire window, the process is **first-order stationary**, and $\mathbb{E}[N(A)] = \lambda |A|$.
- When $\lambda(\mathbf{s})$ varies as a function of coordinates $\mathbf{s} = (x, y)$ or environmental covariates, the process is **inhomogeneous**.

### Second-Order Product Density and Covariance Density

First-order intensity measures event density at individual locations. Second-order properties quantify the joint occurrence of pairs of events separated by vector displacement $\mathbf{h} = \mathbf{s}_1 - \mathbf{s}_2$.

```
[Definition 11.3.2: Second-Order Product Density]
The second-order product density \lambda_2(\mathbf{s}_1, \mathbf{s}_2) is the joint probability density 
that two distinct events occur simultaneously in infinitesimal regions around \mathbf{s}_1 and \mathbf{s}_2:
    \lambda_2(\mathbf{s}_1, \mathbf{s}_2) = \lim_{|d\mathbf{s}_1| \to 0, |d\mathbf{s}_2| \to 0} 
    \frac{\mathbb{E}[N(d\mathbf{s}_1) N(d\mathbf{s}_2) - N(d\mathbf{s}_1 \cap d\mathbf{s}_2)]}{|d\mathbf{s}_1| |d\mathbf{s}_2|}

For any two arbitrary regions A and B:
    \mathbb{E}[N(A) N(B) - N(A \cap B)] = \int_A \int_B \lambda_2(\mathbf{s}_1, \mathbf{s}_2) \, d\mathbf{s}_1 \, d\mathbf{s}_2
The subtraction of N(A \cap B) removes self-pairs (i = j), ensuring that \lambda_2 integrates 
the expected number of distinct ordered pairs of points.
```

The **pair correlation function** $g(\mathbf{s}_1, \mathbf{s}_2)$ standardizes the joint product density by the marginal intensities:

$$
g(\mathbf{s}_1, \mathbf{s}_2) = \frac{\lambda_2(\mathbf{s}_1, \mathbf{s}_2)}{\lambda(\mathbf{s}_1) \lambda(\mathbf{s}_2)}
$$

### The Point-Process Correlogram Equation

Under second-order stationarity and isotropy, the intensity is constant $\lambda(\mathbf{s}) \equiv \lambda$, and the product density depends only on the scalar Euclidean distance $r = \|\mathbf{s}_1 - \mathbf{s}_2\|$:

$$
\lambda_2(\mathbf{s}_1, \mathbf{s}_2) = \lambda_2(r) = \lambda^2 g(r)
$$

Subtracting the product of the first-order intensities yields the **covariance density** of the point process:

$$
\operatorname{cov}_{\text{density}}(r) = \lambda_2(r) - \lambda^2 = \lambda^2 [g(r) - 1]
$$

```
===================================================================================
The Bridge Between Point Processes and Geostatistical Covariance
In Geostatistics (Chapters 1–2):
    C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s} + \mathbf{h})) = C(0) \rho(\mathbf{h})
    where \rho(\mathbf{h}) is the spatial correlogram (\rho(0) = 1, \rho(\infty) = 0).

In Spatial Point Processes:
    \operatorname{cov}_{\text{density}}(r) = \lambda_2(r) - \lambda^2 = \lambda^2 [g(r) - 1]
    where g(r) - 1 is the point-process counterpart to the geostatistical correlogram:
    - Under CSR: g(r) = 1  ==>  \operatorname{cov}_{\text{density}}(r) = 0 (zero spatial correlation)
    - Under Clustering: g(r) > 1  ==>  \operatorname{cov}_{\text{density}}(r) > 0 (positive correlation)
    - Under Inhibition: g(r) < 1  ==>  \operatorname{cov}_{\text{density}}(r) < 0 (negative correlation)
===================================================================================
```

---

## 11.4 Summary Characteristics and Edge Corrections

To diagnose mapped point patterns in practice, statisticians rely on three distance-based summary functions: the empty-space distribution $F(r)$, the nearest-neighbor distribution $G(r)$, and Ripley's $K$-function $K(r)$.

### The Empty-Space Distribution $F(r)$

The **empty-space distribution function** $F(r)$ (also known as the spherical contact distribution) measures the distance from an arbitrary fixed reference point $\mathbf{u} \in W$ to the nearest event of the point process $\Phi$:

$$
F(r) = P(\operatorname{dist}(\mathbf{u}, \Phi) \le r) = 1 - P(N(b(\mathbf{u}, r)) = 0)
$$

For a Homogeneous Poisson Process with intensity $\lambda$, the count in the test disc $b(\mathbf{u}, r)$ follows a $\text{Poisson}(\lambda \pi r^2)$ distribution. The probability that the disc is completely empty is the void probability:

$$
P(N(b(\mathbf{u}, r)) = 0) = \exp(-\lambda |b(\mathbf{u}, r)|) = \exp(-\lambda \pi r^2)
$$

Therefore, under CSR in $\mathbb{R}^2$:

$$
F_{\text{CSR}}(r) = 1 - \exp(-\lambda \pi r^2), \quad r \ge 0
$$

### The Nearest-Neighbor Distribution $G(r)$

While $F(r)$ measures the distance from an *arbitrary location* in the window to the nearest event, the **nearest-neighbor distribution function** $G(r)$ measures the distance from an *observed event* $\mathbf{x}_i \in \Phi$ to its nearest neighboring event in $\Phi \setminus \{\mathbf{x}_i\}$:

$$
G(r) = P(\operatorname{dist}(\mathbf{x}_i, \Phi \setminus \{\mathbf{x}_i\}) \le r \mid \mathbf{x}_i \in \Phi)
$$

What is the theoretical form of $G(r)$ under CSR?  
Under Complete Spatial Randomness, disjoint spatial regions are statistically independent. Conditioning on the presence of an event at coordinate $\mathbf{x}_i$ provides zero information about the number or locations of events in the surrounding region $W \setminus \{\mathbf{x}_i\}$: the remainder of the pattern remains a Poisson process with identical intensity $\lambda$.[^slivnyak]

[^slivnyak]: In advanced point-process literature, this property is formalized by **Slivnyak's Theorem**: the reduced Palm distribution of a Homogeneous Poisson Process equals its unconditional distribution. For linear-model students, the intuition is simple: Poisson events do not interact, so knowing that a lightning strike hit location $\mathbf{s}_1$ does not change the probability that another strike hits nearby.

Because the disc $b(\mathbf{x}_i, r) \setminus \{\mathbf{x}_i\}$ has planar area $\pi r^2$, the probability that it contains at least one other event is identical to the void calculation:

$$
G_{\text{CSR}}(r) = 1 - \exp(-\lambda \pi r^2), \quad r \ge 0
$$

Under CSR, the empty-space function and the nearest-neighbor function coincide: $F(r) = G(r)$.

### The $J$-Function: A Diagnostic Ratio

In non-Poisson patterns, $F(r)$ and $G(r)$ diverge:
- In a **clustered pattern**, events have close neighbors, so $G(r)$ rises rapidly at small $r$. Conversely, large empty spaces exist between clusters, so $F(r)$ rises slowly. Thus, $G(r) > F(r)$.
- In a **regular pattern**, events repel one another, so nearest-neighbor distances are large and $G(r)$ rises slowly at small $r$. However, events are dispersed uniformly across the space, so empty spaces are small and $F(r)$ rises rapidly. Thus, $G(r) < F(r)$.

Lieshout and Baddeley combined $F$ and $G$ into the dimensionless **$J$-function**:

$$
J(r) = \frac{1 - G(r)}{1 - F(r)}, \quad \text{for all } r \text{ such that } F(r) < 1
$$

```
[Diagnostic Rules for the J-Function]
- Under CSR: J(r) \equiv 1 for all distances r.
- Under Clustering: G(r) > F(r)  ==>  1 - G(r) < 1 - F(r)  ==>  J(r) < 1
- Under Regularity: G(r) < F(r)  ==>  1 - G(r) > 1 - F(r)  ==>  J(r) > 1
```

---

### Ripley's $K$-Function and Besag's $L$-Function

Nearest-neighbor statistics ($G$) discard all inter-point distances beyond the single closest neighbor. To capture spatial interaction across all spatial scales simultaneously, Brian Ripley introduced the **$K$-function**.

```
[Definition 11.4.1: Ripley's K-Function]
For a second-order stationary and isotropic point process with intensity \lambda, 
\lambda K(r) is defined as the expected number of further events within distance r 
of an arbitrary typical event:
    \lambda K(r) = \mathbb{E}[N(b(\mathbf{x}, r) \setminus \{\mathbf{x}\}) \mid \mathbf{x} \in \Phi]

Equivalently, K(r) is the cumulative integral of the pair correlation function g(u):
    K(r) = 2\pi \int_0^r u \, g(u) \, du
```

Under Complete Spatial Randomness, points do not interact, so $g(u) \equiv 1$ for all separations $u$:

$$
K_{\text{CSR}}(r) = 2\pi \int_0^r u (1) \, du = 2\pi \left[ \frac{u^2}{2} \right]_0^r = \pi r^2
$$

Because $K_{\text{CSR}}(r)$ grows quadratically with $r$, Julian Besag proposed a variance-stabilizing square-root transformation, known as **Besag's $L$-function**:

$$
L(r) = \sqrt{\frac{K(r)}{\pi}}
$$

Under CSR, $L_{\text{CSR}}(r) = \sqrt{\pi r^2 / \pi} = r$. Plotting the centered difference $L(r) - r$ yields a benchmark horizontal line at zero:

$$
L_{\text{CSR}}(r) - r = 0 \quad \text{for all } r \ge 0
$$

```
===================================================================================
Diagnostic Signatures of Besag's Centered L-Function:
- L(r) - r = 0: Consistent with Complete Spatial Randomness.
- L(r) - r > 0: Aggregation / Clustering (excess pairs within radius r).
- L(r) - r < 0: Regularity / Spatial Inhibition (deficit of pairs within radius r).
===================================================================================
```

---

### The Edge-Effect Problem and Ripley's Circumferential Correction

When computing $K(r)$ from an observed sample $\mathbf{x}_1, \dots, \mathbf{x}_n \subset W$, an immediate geometric challenge arises: **edge effects**.

```
+------------------------------------+
|                                    |
|              Inside W              |
|                                    |
|          xi                        |
|           * - - - - - - +          |
|          / \            |          |
|         /   \           |          |
|        |  .  | d_ij     |          |
|         \   /           |          |
|          \ /            |          |
| - - - - - +-------------+--------- | <-- Boundary of Window W
|          /               \         |
|         |    xj (Outside) |        |
|          \               /         |
|           + - - - - - - +          |
|                                    |
+------------------------------------+
Circle centered at xi with radius d_ij crosses the boundary of window W.
Events that occurred outside W cannot be observed!
```

If we naively count only the events observed inside $W$ within distance $r$ of $\mathbf{x}_i$, points near the boundary will systematically miss neighbors that fell outside the window. This causes severe negative bias at large $r$, making clustered or random patterns appear artificially regular.

To eliminate this bias, Ripley introduced the **isotropic circumferential edge correction**:

```
[Definition 11.4.2: Ripley's Isotropic Edge-Corrected Estimator]
Let W \subset \mathbb{R}^2 be an observation window with area |W|, containing n observed events 
\{\mathbf{x}_1, \dots, \mathbf{x}_n\}. Ripley's estimator of K(r) is:
    \widehat{K}(r) = \frac{|W|}{n(n - 1)} \sum_{i=1}^n \sum_{j \neq i} \frac{\mathbb{I}(\|\mathbf{x}_i - \mathbf{x}_j\| \le r)}{w(\mathbf{x}_i, \|\mathbf{x}_i - \mathbf{x}_j\|)}

where the weighting factor w(\mathbf{x}_i, d_{ij}) is the proportion of the circumference of the circle
centered at \mathbf{x}_i with radius d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\| that lies inside the window W:
    w(\mathbf{x}_i, d_{ij}) = \frac{\text{Length}(\partial b(\mathbf{x}_i, d_{ij}) \cap W)}{2\pi d_{ij}} \in (0, 1]
```

```
===================================================================================
Anatomy of Ripley's Estimator:
    \widehat{K}(r) = \frac{|W|}{n(n - 1)} \sum_{i=1}^n \sum_{j \neq i} \frac{\mathbb{I}(d_{ij} \le r)}{w(\mathbf{x}_i, d_{ij})}

- \mathbb{I}(d_{ij} \le r): Activates only when event j lies within distance r of event i.
- 1 / w(\mathbf{x}_i, d_{ij}) >= 1: If half the circle around \mathbf{x}_i falls outside W (w = 0.5),
  each observed neighbor inside W is given weight 1 / 0.5 = 2 to account for unseen points.
- |W| / [n(n - 1)]: Under a Poisson process, \mathbb{E}[n(n - 1)] = \lambda^2 |W|^2. Dividing by
  n(n - 1) / |W| guarantees an unbiased denominator for \lambda^2.
===================================================================================
```

> [!NOTE]
> ### Why Not Simply Discard Boundary Points? (The Border Correction)
> An alternative edge correction—the **border method**—computes distances only from events $\mathbf{x}_i$ whose distance to the boundary $\partial W$ exceeds $r$. While mathematically straightforward, the border method discards a massive fraction of observed data. In a unit square $W = [0, 1]^2$, the proportion of the window within distance $r$ of the boundary is:
>
> $$
> \text{Discard Fraction} = 1 - (1 - 2r)^2
> $$
>
> For $r = 0.10$, the border method discards $1 - (0.8)^2 = 36\%$ of the data. For $r = 0.20$, it discards $64\%$, and for $r = 0.25$, it discards $75\%$! Ripley's circumferential weighting retains all $n$ data points, yielding substantially lower estimation variance.

### Unbiasedness Reality and Boundary Rules

Textbooks sometimes describe Ripley's estimator as "unbiased." In truth:
1. $\widehat{K}(r)$ is **approximately unbiased under CSR** only for search radii well below the window dimensions (typically $r \le 0.4 \times \text{window width}$).
2. For large radii $r$, circles centered at corners of $W$ can have $w(\mathbf{x}_i, d_{ij}) \to 0$, leading to high variance and numerical instability.
3. $\widehat{K}(r)$ is a cumulative ratio estimator: its values at consecutive radii $r_1 < r_2 < \dots < r_K$ are strongly positively correlated.

---

### Four Critical Pitfalls in Interpreting $K(r)$ and $L(r)$

1. **Trend Confounded with Clustering:** As noted in Section 11.2, a smooth first-order intensity gradient $\lambda(\mathbf{s})$ inflates inter-point counts across all scales, forcing $\widehat{L}(r) - r > 0$ even when points are generated with zero point-to-point interaction.
2. **Pointwise vs. Global Envelopes:** A Monte Carlo envelope evaluated at significance level $\alpha = 0.05$ across 20 distinct radii will see $\widehat{L}(r) - r$ leave the envelope by random chance at least once with probability much higher than $5\%$. Formal hypothesis testing requires a global test statistic (e.g., maximum deviation $\max_r |\widehat{L}(r) - r|$).
3. **Edge Instability at Large Lags:** Do not interpret $\widehat{K}(r)$ for search radii exceeding half the narrowest dimension of the observation window.
4. **Non-Identifiability:** Ripley's $K$-function is a purely second-order summary. Two fundamentally different spatial processes (e.g., a clustered Poisson cluster process and a mixture of inhomogeneous Poisson processes) can possess mathematically identical $K(r)$ functions.

---

### Master Comparison Table: Diagnostic Signatures Across Spatial Archetypes

| Diagnostic Metric | Complete Spatial Randomness (CSR) | Clustered / Aggregated | Regular / Inhibited |
|:---|:---:|:---:|:---:|
| **Physical Mechanism** | Independent uniform placement | Attraction, dispersal, shared niche | Competition, territorial repulsion |
| **Index of Dispersion $I$** | $I \approx 1$ | $I > 1$ (Overdispersed) | $I < 1$ (Underdispersed) |
| **Pair Correlation $g(r)$** | $g(r) \equiv 1$ | $g(r) > 1$ at small $r$ | $g(r) < 1$ at small $r$ |
| **Nearest-Neighbor $G(r)$** | $G(r) = 1 - e^{-\lambda \pi r^2}$ | Rises faster than CSR at small $r$ | Rises slower than CSR at small $r$ |
| **Empty-Space $F(r)$** | $F(r) = 1 - e^{-\lambda \pi r^2}$ | Rises slower than CSR at small $r$ | Rises faster than CSR at small $r$ |
| **$J$-Function $J(r)$** | $J(r) \equiv 1$ | $J(r) < 1$ | $J(r) > 1$ |
| **Ripley's $K(r)$** | $K(r) = \pi r^2$ | $K(r) > \pi r^2$ | $K(r) < \pi r^2$ |
| **Centered Besag $L(r) - r$** | $L(r) - r = 0$ | $L(r) - r > 0$ | $L(r) - r < 0$ |

---

## 11.5 Non-Poisson Point Processes: Clusters, Inhomogeneity, and Regularity

When a point pattern rejects Complete Spatial Randomness, we must formulate an explicit statistical model capable of capturing either first-order environmental inhomogeneity or second-order interaction.

```
                    Spatial Point Processes
                               |
         +---------------------+---------------------+
         |                                           |
   Poisson Processes                           Non-Poisson Processes
         |                                           |
   +-----+-----+                         +-----------+-----------+
   |           |                         |           |           |
Homogeneous  Inhomogeneous             Clustered   Inhibited  Doubly Stochastic
  (CSR)       Poisson                    Models      Models      Cox Processes
                |                          |           |               |
         Intensity \lambda(s)         Neyman-Scott  Strauss        Log-Gaussian
         Poisson GLM / Offset            Matérn     Hard-Core       Cox (LGCP)
```

### Inhomogeneous Poisson Processes (IPP)

An **Inhomogeneous Poisson Process (IPP)** relaxes the constant-rate assumption: the intensity $\lambda(\mathbf{s}) \ge 0$ varies continuously as a function of coordinates $\mathbf{s} = (x, y)$ or spatial environmental covariates $\mathbf{z}(\mathbf{s})$.

```
[Definition 11.5.1: Inhomogeneous Poisson Process]
A point process \Phi on W is an Inhomogeneous Poisson Process with intensity function \lambda(\mathbf{s}) if:
1. For any bounded region A \subset W:
       N(A) \sim \text{Poisson}(\mu(A)), \quad \text{where } \mu(A) = \int_A \lambda(\mathbf{s}) \, d\mathbf{s}
2. For any disjoint regions A_1, \dots, A_m \subset W:
       N(A_1), \dots, N(A_m) \text{ are mutually independent random variables.}
```

Because Postulate 2 (independent increments) is preserved, points in an IPP exhibit **zero point-to-point interaction**. All clustering or dispersion observed in an IPP is purely first-order: points concentrate in regions where $\lambda(\mathbf{s})$ is elevated.

#### Simulating IPPs: The Lewis–Shedler Thinning Algorithm

Simulating an IPP with a non-constant intensity surface $\lambda(\mathbf{s})$ is accomplished through **independent thinning**:

```
Algorithm 11.1: Lewis-Shedler Rejection Thinning
1. Find an upper bound \lambda_max such that \lambda(\mathbf{s}) \le \lambda_max for all \mathbf{s} \in W.
2. Simulate a candidate Homogeneous Poisson Process \Phi_cand with constant rate \lambda_max:
       - Sample N_cand ~ Poisson(\lambda_max |W|)
       - Sample N_cand coordinates \{\mathbf{s}_1, \dots, \mathbf{s}_{N_cand}\} i.i.d. Uniform over W
3. For each candidate point \mathbf{s}_i:
       - Draw u_i ~ Uniform(0, 1)
       - Retain \mathbf{s}_i if and only if u_i \le \frac{\lambda(\mathbf{s}_i)}{\lambda_max}
4. Return the thinned pattern \Phi_retained = \{\mathbf{s}_i : u_i \le \lambda(\mathbf{s}_i) / \lambda_max\}.
```

**Why thinning works:**  
The candidate process generates points with probability density $\lambda_{\max} d\mathbf{s}$. Each point is independently accepted with probability $p(\mathbf{s}) = \lambda(\mathbf{s}) / \lambda_{\max}$. The effective local arrival rate of accepted points is:

$$
\lambda_{\text{effective}}(\mathbf{s}) = \lambda_{\max} \cdot p(\mathbf{s}) = \lambda_{\max} \cdot \frac{\lambda(\mathbf{s})}{\lambda_{\max}} = \lambda(\mathbf{s})
$$

Because acceptance is decided independently for each point, spatial independence between disjoint regions is strictly preserved.

---

### The Poisson Regression Bridge: Fitting Point Process Models

How do we fit an inhomogeneous Poisson model to observed field data?  
The likelihood of an Inhomogeneous Poisson Process with intensity $\lambda(\mathbf{s}; \boldsymbol{\beta})$ observing points $\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subset W$ is:

$$
L(\boldsymbol{\beta}) = \left( \prod_{i=1}^n \lambda(\mathbf{x}_i; \boldsymbol{\beta}) \right) \exp\left( -\int_W \lambda(\mathbf{s}; \boldsymbol{\beta}) \, d\mathbf{s} \right)
$$

The log-likelihood is:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \log \lambda(\mathbf{x}_i; \boldsymbol{\beta}) - \int_W \lambda(\mathbf{s}; \boldsymbol{\beta}) \, d\mathbf{s}
$$

The integral over $W$ lacks a closed form for arbitrary covariate surfaces. To fit this model using standard linear modeling software, statisticians use **Berman–Turner quadrature** (or regular cell count discretization):

1. Discretize the window $W$ into $K$ small, non-overlapping grid cells $A_1, A_2, \dots, A_K$, each with area $|A_k|$.
2. In each cell, count the number of observed events: $Y_k = N(A_k)$.
3. Measure environmental covariates at the cell center: $\mathbf{z}_k = \mathbf{z}(\mathbf{s}_k)$.
4. Assume log-linear intensity: $\log \lambda(\mathbf{s}_k) = \mathbf{z}_k^\top \boldsymbol{\beta}$.

Because disjoint cells have independent Poisson counts:

$$
Y_k \sim \text{Poisson}(\mu_k), \quad \text{where } \mu_k = \int_{A_k} \lambda(\mathbf{s}) \, d\mathbf{s} \approx \lambda(\mathbf{s}_k) |A_k|
$$

Taking the logarithm of the expected cell count:

$$
\log \mathbb{E}[Y_k] = \log(\lambda(\mathbf{s}_k) |A_k|) = \mathbf{z}_k^\top \boldsymbol{\beta} + \log |A_k|
$$

```
===================================================================================
The Bridge Between Point Patterns and Poisson Regression:
    \log \mathbb{E}[Y_k] = \mathbf{z}_k^\top \boldsymbol{\beta} + \log |A_k|  (Offset)

Fitting an Inhomogeneous Poisson Process is mathematically identical to fitting 
a Generalized Linear Model (GLM) with a Poisson family, a log link function, 
and a fixed offset term equal to the logarithm of the cell area \log |A_k|:
    glm(count ~ elevation + slope + offset(log(cell_area)), family = poisson)
===================================================================================
```

---

> [!NOTE]
> ### From the Lecture Hall: Poisson Regression with an Offset
> In standard Poisson regression, the response $Y_k$ represents an event count. But counts naturally scale with exposure: a grid cell of $100\text{ m}^2$ should receive 4 times as many points as a cell of $25\text{ m}^2$, even under identical environmental conditions.
>
> In GLM terminology, an **offset** is a covariate whose regression coefficient is constrained to be exactly $1.0$:
>
> $$
> \log \mu_k = \beta_0 + \beta_1 z_{k1} + \dots + \beta_p z_{kp} + 1.0 \times \log(\text{Area}_k)
> $$
>
> Exponentiating both sides yields:
>
> $$
> \mu_k = \exp(\mathbf{z}_k^\top \boldsymbol{\beta}) \cdot \text{Area}_k = \lambda_k \cdot \text{Area}_k
> $$
>
> The offset converts the GLM from a model of *discrete counts* $\mu_k$ into a model of *continuous spatial intensity* $\lambda_k = \mu_k / \text{Area}_k$.

---

### Clustered Point Processes: Neyman–Scott and Matérn

When clustering is driven by physical reproduction and dispersal rather than environmental covariates, we turn to **Poisson Cluster Processes**.

In a **Neyman–Scott cluster process**:
1. **Parent events** $\mathbf{x}_{\text{parent}}$ are generated according to a Homogeneous Poisson Process with parent intensity $\rho$. Parent points are unobserved in the field.
2. Each parent independently produces a random number $M$ of **offspring events**, where $M \sim \text{Poisson}(\mu_{\text{off}})$.
3. The offspring are dispersed independently around their parent location according to a bivariate probability density kernel $f(\mathbf{s} - \mathbf{x}_{\text{parent}})$.
4. The observed point pattern consists of the superposition of all offspring (parents are removed).

In the **Matérn cluster process**, the dispersal kernel is uniform over a disc of radius $R$:

$$
f(\mathbf{u}) = \begin{cases} \frac{1}{\pi R^2}, & \|\mathbf{u}\| \le R \\ 0, & \|\mathbf{u}\| > R \end{cases}
$$

Because offspring cluster around their common parent, the pair correlation function satisfies:

$$
g(r) = 1 + \frac{1}{\rho} \int_{\mathbb{R}^2} f(\mathbf{u}) f(\mathbf{u} + \mathbf{r}) \, d\mathbf{u} > 1 \quad \text{for } r \le 2R
$$

At distances $r > 2R$, offspring from different parents do not overlap, and $g(r)$ decays to $1.0$.

---

### Cox Processes and Log-Gaussian Cox Processes (LGCP)

A **Cox process** (or doubly stochastic Poisson process) models point patterns where the intensity function itself is a random spatial surface $\Lambda(\mathbf{s})$.

```
[Definition 11.5.2: Cox Process and Log-Gaussian Cox Process]
A point process \Phi on W is a Cox Process directed by a non-negative random field \Lambda(\mathbf{s}) if:
- Conditioned on a realization of the random surface \Lambda(\mathbf{s}) = \lambda(\mathbf{s}), \Phi is an
  Inhomogeneous Poisson Process with intensity \lambda(\mathbf{s}).

In a Log-Gaussian Cox Process (LGCP), the log-intensity is a Gaussian Random Field Y(\mathbf{s}):
    \log \Lambda(\mathbf{s}) = Y(\mathbf{s}) \sim \mathcal{GP}(\mu, C_Y(\mathbf{h}))
where Y(\mathbf{s}) is a stationary Gaussian process with mean \mu and spatial covariance C_Y(\mathbf{h}).
```

#### Moments of the LGCP and the Link to Chapter 9

To derive the intensity and pair correlation of an LGCP, we invoke the lognormal moment-generating function established in Chapter 9 (§9.2):

```
[Theorem 11.5.1: Intensity and Pair Correlation of an LGCP]
Let \Phi be an LGCP driven by Gaussian field Y(\mathbf{s}) \sim \mathcal{GP}(\mu, C_Y(\mathbf{h})).
1. First-Order Intensity:
       \lambda = \mathbb{E}[\Lambda(\mathbf{s})] = \exp(\mu + 0.5 \cdot C_Y(\mathbf{0}))
2. Second-Order Product Density:
       \lambda_2(\mathbf{h}) = \lambda^2 \exp(C_Y(\mathbf{h}))
3. Pair Correlation Function:
       g(\mathbf{h}) = \exp(C_Y(\mathbf{h}))
4. If C_Y(\mathbf{h}) \ge 0 for all lags, then:
       g(\mathbf{h}) \ge 1 \quad \text{and} \quad K(r) > \pi r^2 \quad \text{for all } r > 0.
```

**Proof:**  
*Part 1:* For any coordinate $\mathbf{s}$, $Y(\mathbf{s}) \sim \mathcal{N}(\mu, \sigma_Y^2)$, where $\sigma_Y^2 = C_Y(\mathbf{0})$. The random intensity $\Lambda(\mathbf{s}) = \exp(Y(\mathbf{s}))$ follows a lognormal distribution. Using the Gaussian moment generating function $M_Y(t) = \mathbb{E}[e^{t Y}] = \exp(\mu t + \frac{1}{2}\sigma_Y^2 t^2)$ evaluated at $t = 1$:

$$
\lambda = \mathbb{E}[\Lambda(\mathbf{s})] = \mathbb{E}[\exp(Y(\mathbf{s}))] = \exp\left(\mu + \frac{1}{2} C_Y(\mathbf{0})\right)
$$

*Part 2 & 3:* For two locations separated by displacement $\mathbf{h} = \mathbf{s}_1 - \mathbf{s}_2$, the random vector $(Y(\mathbf{s}_1), Y(\mathbf{s}_2))$ has a bivariate normal distribution with covariance $\operatorname{Cov}(Y(\mathbf{s}_1), Y(\mathbf{s}_2)) = C_Y(\mathbf{h})$.  
By the law of total expectation, the second-order product density of a Cox process is the expected product of its random intensities:

$$
\lambda_2(\mathbf{h}) = \mathbb{E}[\Lambda(\mathbf{s}_1) \Lambda(\mathbf{s}_2)] = \mathbb{E}[\exp(Y(\mathbf{s}_1) + Y(\mathbf{s}_2))]
$$

The sum $V = Y(\mathbf{s}_1) + Y(\mathbf{s}_2)$ is univariate normal with:

$$
\mathbb{E}[V] = 2\mu, \qquad \operatorname{Var}(V) = \operatorname{Var}(Y(\mathbf{s}_1)) + \operatorname{Var}(Y(\mathbf{s}_2)) + 2\operatorname{Cov}(Y(\mathbf{s}_1), Y(\mathbf{s}_2)) = 2 C_Y(\mathbf{0}) + 2 C_Y(\mathbf{h})
$$

Applying the lognormal expectation formula $\mathbb{E}[e^V] = \exp(\mathbb{E}[V] + \frac{1}{2}\operatorname{Var}(V))$:

$$
\lambda_2(\mathbf{h}) = \exp\left( 2\mu + C_Y(\mathbf{0}) + C_Y(\mathbf{h}) \right) = \left[ \exp\left( \mu + \frac{1}{2} C_Y(\mathbf{0}) \right) \right]^2 \exp(C_Y(\mathbf{h})) = \lambda^2 \exp(C_Y(\mathbf{h}))
$$

Dividing by $\lambda^2$ yields the pair correlation function:

$$
g(\mathbf{h}) = \frac{\lambda_2(\mathbf{h})}{\lambda^2} = \exp(C_Y(\mathbf{h}))
$$

*Part 4:* If the spatial covariance kernel satisfies $C_Y(\mathbf{h}) \ge 0$ (such as standard Exponential, Gaussian, or Matérn covariance models with non-negative sills), then $\exp(C_Y(\mathbf{h})) \ge \exp(0) = 1$. Consequently, $g(\mathbf{h}) \ge 1$ and $K(r) = 2\pi \int_0^r u g(u) du > 2\pi \int_0^r u du = \pi r^2$, reflecting spatial aggregation across all separation distances. $\blacksquare$

> [!NOTE]
> ### Hole-Effect Covariances and Local Inhibition in LGCP
> Notice the critical requirement $C_Y(\mathbf{h}) \ge 0$. As established in Chapter 2, valid covariance functions can take negative values (for example, the cardinal sine wave or "hole-effect" covariance $C_Y(h) = \sigma^2 \frac{\sin(h/a)}{h/a}$).  
> If $C_Y(\mathbf{h}) < 0$ at separation $\mathbf{h}$, then $g(\mathbf{h}) = \exp(C_Y(\mathbf{h})) < 1$. Thus, while most common LGCPs model clustered patterns, an LGCP with an oscillatory, negative covariance kernel produces local spatial inhibition!

---

### Regular and Inhibited Processes: The Strauss Process

To model spatial repulsion, territorial exclusion, and physical non-overlapping packing, we must move beyond Poisson and Cox families into **Markov Point Processes**.

The classic model of regular spacing is the **Strauss Process**:

```
[Definition 11.5.3: The Strauss Process]
Let \mathbf{x} = \{\mathbf{x}_1, \dots, \mathbf{x}_n\} be a point configuration in window W.
The probability density of the Strauss process relative to a homogeneous Poisson process is:
    f(\mathbf{x}) = \alpha \beta^{n(\mathbf{x})} \gamma^{s(\mathbf{x})}

where:
- n(\mathbf{x}) is the total number of events in the pattern.
- s(\mathbf{x}) is the number of unordered pairs of distinct points separated by distance \le R:
      s(\mathbf{x}) = \sum_{1 \le i < j \le n} \mathbb{I}(\|\mathbf{x}_i - \mathbf{x}_j\| \le R)
- \beta > 0 is the baseline intensity parameter.
- \gamma \in [0, 1] is the interaction parameter.
- R > 0 is the interaction radius.
- \alpha is the normalizing constant (partition function) ensuring \int f(\mathbf{x}) = 1.
```

```
===================================================================================
Anatomy of the Strauss Interaction Parameter \gamma:
- \gamma = 1: f(\mathbf{x}) = \alpha \beta^n. Pair interactions vanish completely.
  The process collapses to a Homogeneous Poisson Process (CSR) with intensity \lambda = \beta.
- \gamma = 0: 0^0 = 1, but 0^{s(\mathbf{x})} = 0 whenever s(\mathbf{x}) > 0.
  Any pattern with even a single pair closer than distance R has probability zero!
  This is the classical Hard-Core Process with exclusion radius R.
- 0 < \gamma < 1: Each pair of points within distance R penalizes the likelihood 
  by a multiplicative factor of \gamma. Points repel one another ("soft-core" inhibition).
===================================================================================
```

> [!CAUTION]
> ### Why \gamma > 1 is Mathematically Forbidden
> Students often ask: *"Can we model clustering by setting $\gamma > 1$?"*  
> The answer is an absolute **no**. If $\gamma > 1$, configurations with dense clumps of points have $s(\mathbf{x}) \sim \frac{1}{2} n^2$, causing $\gamma^{s(\mathbf{x})} = \gamma^{n^2/2}$ to explode superexponentially. The normalizing integral $\int f(\mathbf{x}) = \infty$ diverges, and the model collapses into a mathematical singularity where infinitely many points pile into a single cluster. Clustering must be modeled via Cox or Neyman–Scott processes, not Strauss models.

#### Simulation and Estimation Challenges for Strauss Models
Unlike Poisson processes, the normalizing constant $\alpha$ of a Strauss process cannot be evaluated in closed form:

$$
\alpha^{-1} = \sum_{n=0}^\infty \frac{\beta^n e^{-|W|}}{n!} \int_{W^n} \gamma^{s(\mathbf{s}_1, \dots, \mathbf{s}_n)} \, d\mathbf{s}_1 \dots d\mathbf{s}_n
$$

Because $\alpha$ is intractable, **exact maximum likelihood estimation is impossible**. Fitting a Strauss model requires either:
- **Maximum Pseudolikelihood Estimation (MPLE):** Approximating the likelihood via Papangelou conditional intensities.
- **Markov Chain Monte Carlo (MCMC):** Simulating realizations via birth-death Metropolis–Hastings algorithms, where candidate points are iteratively added or deleted.

---

## 11.6 Algorithmic Translation: Simulation and Monte Carlo Envelopes

We now translate the theory into complete, self-contained Python implementations using only standard library routines.

### Python Implementation 1: Simulating CSR, Thinning, and Hard-Core Regularity

```python
# Chapter 11: Point Process Simulation Suite
# Implements exact CSR simulation, Lewis-Shedler thinning, and hard-core inhibition.
# Standard library only: math, random.
import math
import random

def sample_poisson(mean_val: float, rng: random.Random) -> int:
    # Draws a single integer from a Poisson(mean_val) distribution.
    # Uses Knuth algorithm for small means and Gaussian approximation for large means.
    if mean_val <= 0.0:
        return 0
    if mean_val < 30.0:
        limit = math.exp(-mean_val)
        k = 0
        p = 1.0
        while p > limit:
            k += 1
            p *= rng.random()
        return k - 1
    else:
        # Continuity-corrected Gaussian approximation for large mean
        val = rng.gauss(mean_val, math.sqrt(mean_val))
        return max(0, int(round(val)))

def simulate_csr(
    lam: float,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    rng: random.Random = None
) -> list[tuple[float, float]]:
    # Simulates Complete Spatial Randomness (Homogeneous Poisson Process) on a rectangle.
    # Constructive method via Theorem 11.2.1:
    # 1. Draw N ~ Poisson(lam * Area)
    # 2. Draw N independent uniform coordinates across the window.
    if rng is None:
        rng = random.Random()
    area = (x_max - x_min) * (y_max - y_min)
    n = sample_poisson(lam * area, rng)
    return [(rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)) for _ in range(n)]

def simulate_inhomogeneous_thinning(
    intensity_fn,
    lam_max: float,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    rng: random.Random = None
) -> list[tuple[float, float]]:
    # Simulates an Inhomogeneous Poisson Process via Lewis-Shedler Rejection Thinning.
    # Accepts candidate points with probability lambda(x, y) / lambda_max.
    if rng is None:
        rng = random.Random()
    # Step 1: Candidate homogeneous process with rate lam_max
    candidates = simulate_csr(lam_max, x_min, x_max, y_min, y_max, rng)
    # Step 2: Independent rejection thinning
    retained = []
    for x, y in candidates:
        lam_val = intensity_fn(x, y)
        if lam_val > lam_max:
            raise ValueError(f"Intensity {lam_val} exceeds upper bound {lam_max} at ({x}, {y})")
        if rng.random() <= (lam_val / lam_max):
            retained.append((x, y))
    return retained

def simulate_hardcore_regular(
    n_points: int,
    d_min: float,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    max_attempts: int = 20000,
    rng: random.Random = None
) -> list[tuple[float, float]]:
    # Simulates a regular hard-core point pattern via Simple Sequential Inhibition (SSI).
    # Points closer than d_min to any existing point are strictly rejected.
    if rng is None:
        rng = random.Random()
    points = []
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if all(math.hypot(x - px, y - py) >= d_min for px, py in points):
            points.append((x, y))
    if len(points) < n_points:
        raise RuntimeError(f"Jamming limit reached: placed only {len(points)}/{n_points} points.")
    return points
```

---

### Python Implementation 2: Ripley's Isotropic Edge-Corrected $K$ and Centered $L$

```python
def ripley_circumferential_weight(
    x_i: float, y_i: float, x_j: float, y_j: float,
    x_min: float = 0.0, x_max: float = 1.0,
    y_min: float = 0.0, y_max: float = 1.0,
    num_samples: int = 72
) -> float:
    # Computes Ripley's circumferential isotropic edge-correction weight w(s_i, d_ij).
    # Approximates the fraction of perimeter of circle(s_i, d_ij) that lies inside W.
    # 72 samples (5-degree increments) provides 4-digit precision.
    d_ij = math.hypot(x_i - x_j, y_i - y_j)
    if d_ij < 1e-12:
        return 1.0
    inside_count = 0
    for s in range(num_samples):
        theta = 2.0 * math.pi * s / num_samples
        x_arc = x_i + d_ij * math.cos(theta)
        y_arc = y_i + d_ij * math.sin(theta)
        if (x_min <= x_arc <= x_max) and (y_min <= y_arc <= y_max):
            inside_count += 1
    proportion = inside_count / num_samples
    return max(0.1, proportion)  # Safeguard for extreme outer corners

def compute_ripley_k_and_l(
    points: list[tuple[float, float]],
    radii: list[float],
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0
) -> tuple[list[float], list[float]]:
    # Computes Ripley's edge-corrected K(r) and Besag's centered L(r) - r across radii.
    # Precalculates pairwise distances and edge weights for O(n^2) efficiency.
    n = len(points)
    area = (x_max - x_min) * (y_max - y_min)
    if n < 2:
        return [0.0] * len(radii), [0.0] * len(radii)

    # Precalculate all pairwise Euclidean distances and bidirectional weights
    pairs = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dist = math.hypot(xi - xj, yi - yj)
            w_ij = ripley_circumferential_weight(xi, yi, xj, yj, x_min, x_max, y_min, y_max)
            w_ji = ripley_circumferential_weight(xj, yj, xi, yi, x_min, x_max, y_min, y_max)
            pairs.append((dist, 1.0 / w_ij + 1.0 / w_ji))

    k_hat = []
    l_centered = []
    scale_factor = area / (n * (n - 1))

    for r in radii:
        weighted_sum = sum(weight for dist, weight in pairs if dist <= r)
        k_val = scale_factor * weighted_sum
        l_val = math.sqrt(k_val / math.pi)
        k_hat.append(k_val)
        l_centered.append(l_val - r)

    return k_hat, l_centered
```

---

### Python Implementation 3: Monte Carlo Simulation Envelopes and Diagnostic Lab

To replace arbitrary heuristic cutoffs (such as $|\widehat{L}(r) - r| > 0.02$), we construct genuine **Monte Carlo CSR Envelopes**:

```python
def compute_csr_envelopes(
    n_points: int,
    radii: list[float],
    num_simulations: int = 99,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    rng: random.Random = None
) -> tuple[list[float], list[float]]:
    # Constructs pointwise Monte Carlo CSR envelopes for L(r) - r.
    # Simulates M CSR patterns conditioned on n_points (Theorem 11.2.1),
    # returning the pointwise minimum and maximum envelope curves.
    if rng is None:
        rng = random.Random()
    sim_results = [[] for _ in radii]
    for _ in range(num_simulations):
        sim_pts = [(rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)) for _ in range(n_points)]
        _, l_cent = compute_ripley_k_and_l(sim_pts, radii, x_min, x_max, y_min, y_max)
        for idx, val in enumerate(l_cent):
            sim_results[idx].append(val)

    lower_env = [min(sim_results[idx]) for idx in range(len(radii))]
    upper_env = [max(sim_results[idx]) for idx in range(len(radii))]
    return lower_env, upper_env

def run_point_process_diagnostic_lab():
    # Runs a comparative diagnostic laboratory on three simulated patterns:
    # 1. A true CSR pattern (n = 35)
    # 2. A hard-core regular pattern (n = 35, d_min = 0.08)
    # 3. An inhomogeneous pattern with a steep trend (mimicking clustering)
    rng = random.Random(42)
    radii = [0.05, 0.10, 0.15, 0.20, 0.25]
    n_benchmark = 35

    print("Generating three spatial patterns...")
    csr_pattern = simulate_csr(lam=35.0, rng=rng)
    regular_pattern = simulate_hardcore_regular(n_points=n_benchmark, d_min=0.08, rng=rng)
    # Steep exponential trend along x-axis: lambda(x) = 5.0 + 55.0 * (x^2)
    inhomog_pattern = simulate_inhomogeneous_thinning(
        intensity_fn=lambda x, y: 5.0 + 55.0 * (x ** 2),
        lam_max=60.0,
        rng=rng
    )

    print(f"Counts: CSR={len(csr_pattern)}, Regular={len(regular_pattern)}, Inhomogeneous={len(inhomog_pattern)}")
    print("\nComputing 99-run Monte Carlo CSR envelopes for n = 35...")
    csr_low, csr_high = compute_csr_envelopes(n_benchmark, radii, num_simulations=99, rng=rng)

    patterns = [
        ("CSR Random", csr_pattern),
        ("Hard-Core Regular", regular_pattern),
        ("Inhomogeneous Trend", inhomog_pattern)
    ]

    for name, pts in patterns:
        _, l_cent = compute_ripley_k_and_l(pts, radii)
        print(f"\n--- Diagnostic Profile: {name} (n = {len(pts)}) ---")
        print(f"{'Radius r':<10} | {'Lower Env':<10} | {'Obs L(r)-r':<12} | {'Upper Env':<10} | {'Diagnosis'}")
        print("-" * 65)
        for idx, r in enumerate(radii):
            lo = csr_low[idx]
            hi = csr_high[idx]
            obs = l_cent[idx]
            if obs < lo:
                diag = "REGULAR (Below Envelope)"
            elif obs > hi:
                diag = "CLUSTERED (Above Envelope)"
            else:
                diag = "Inside CSR Band"
            print(f"{r:<10.2f} | {lo:<+10.4f} | {obs:<+12.4f} | {hi:<+10.4f} | {diag}")

if __name__ == "__main__":
    run_point_process_diagnostic_lab()
```

---

## 11.7 Worked Analytical and Computational Exercises

### Exercise 11.1: Derivation of the Empty-Space Function $F(r)$ in $\mathbb{R}^3$ for a 3D Homogeneous Poisson Process

**Problem Statement:**  
Consider a spatial point process $\Phi$ representing astronomical galaxy clusters or bubbles in a solid casting, modeled as a 3D Homogeneous Poisson Process on $\mathbb{R}^3$ with constant intensity $\lambda > 0$ events per unit volume.
1. Derive the theoretical empty-space distribution function $F(r)$ for an arbitrary reference point $\mathbf{u} \in \mathbb{R}^3$.
2. Derive the probability density function $f(r) = \frac{d}{dr} F(r)$ and determine the median distance to the nearest galaxy cluster.

**Solution:**

*Part 1: Derivation of $F(r)$.*  
By definition, the empty-space function $F(r)$ is the probability that at least one event occurs within a Euclidean search sphere $B_3(\mathbf{u}, r)$ of radius $r$ centered at reference point $\mathbf{u}$:

$$
F(r) = P(\operatorname{dist}(\mathbf{u}, \Phi) \le r) = 1 - P(N(B_3(\mathbf{u}, r)) = 0)
$$

The volume of a 3-dimensional ball of radius $r$ is:

$$
|B_3(\mathbf{u}, r)| = \frac{4}{3} \pi r^3
$$

Under a 3D Homogeneous Poisson Process with intensity $\lambda$, the count of events in $B_3(\mathbf{u}, r)$ follows a Poisson distribution with mean parameter $\mu = \lambda |B_3(\mathbf{u}, r)| = \frac{4}{3} \lambda \pi r^3$. The void probability of the sphere is:

$$
P(N(B_3(\mathbf{u}, r)) = 0) = \exp(-\mu) = \exp\left( -\frac{4}{3} \lambda \pi r^3 \right)
$$

Subtracting from 1 yields the theoretical empty-space distribution function in $\mathbb{R}^3$:

$$
F(r) = 1 - \exp\left( -\frac{4}{3} \lambda \pi r^3 \right), \quad r \ge 0
$$

*Part 2: Probability Density and Median Distance.*  
Differentiating $F(r)$ with respect to $r$ using the chain rule:

$$
f(r) = \frac{d}{dr} \left[ 1 - \exp\left( -\frac{4}{3} \lambda \pi r^3 \right) \right] = 4 \lambda \pi r^2 \exp\left( -\frac{4}{3} \lambda \pi r^3 \right), \quad r \ge 0
$$

To find the median distance $r_{\text{med}}$, we solve $F(r_{\text{med}}) = 0.5$:

$$
1 - \exp\left( -\frac{4}{3} \lambda \pi r_{\text{med}}^3 \right) = 0.5 \implies \exp\left( -\frac{4}{3} \lambda \pi r_{\text{med}}^3 \right) = 0.5
$$

Taking the natural logarithm of both sides:

$$
-\frac{4}{3} \lambda \pi r_{\text{med}}^3 = \ln(0.5) = -\ln(2) \implies r_{\text{med}}^3 = \frac{3 \ln(2)}{4 \pi \lambda}
$$

Taking the cube root:

$$
r_{\text{med}} = \left( \frac{3 \ln(2)}{4 \pi \lambda} \right)^{1/3} \approx 0.5484 \lambda^{-1/3}
$$

The median distance scales inversely with the cube root of the spatial density $\lambda^{1/3}$, as expected from dimensional analysis in $\mathbb{R}^3$. $\blacksquare$

---

### Exercise 11.2: Ripley's $K(r)$ and Pair Correlation $g(r)$ for an Ideal Hard-Core Process

**Problem Statement:**  
Consider a stationary, isotropic hard-core point process in $\mathbb{R}^2$ with intensity $\lambda > 0$ and strict physical exclusion radius $R_{\text{core}} > 0$. No two events can exist at a Euclidean distance less than or equal to $R_{\text{core}}$.
1. Formulate the exact pair correlation function $g(r)$ for $r \le R_{\text{core}}$.
2. Evaluate Ripley's $K$-function $K(r)$ and Besag's $L$-function $L(r)$ for all $r \le R_{\text{core}}$, and show that $L(r) - r = -r$.
3. Explain why the derivative $\frac{d}{dr} K(r)$ exhibits a step discontinuity at the hard-core boundary $r = R_{\text{core}}$.

**Solution:**

*Part 1: Pair Correlation Function.*  
The product density $\lambda_2(r)$ represents the joint probability of finding two distinct events separated by distance $r$. Under an ideal hard-core constraint, pairs at distance $r \le R_{\text{core}}$ are physically forbidden:

$$
\lambda_2(r) = 0 \quad \text{for all } 0 \le r \le R_{\text{core}}
$$

Dividing by the squared intensity $\lambda^2$:

$$
g(r) = \frac{\lambda_2(r)}{\lambda^2} = 0 \quad \text{for all } 0 \le r \le R_{\text{core}}
$$

*Part 2: Evaluation of $K(r)$ and $L(r)$.*  
Ripley's $K$-function is the cumulative integral of the pair correlation function:

$$
K(r) = 2\pi \int_0^r u \, g(u) \, du
$$

For any search radius $r \le R_{\text{core}}$, the integrand $g(u) \equiv 0$ over the entire domain of integration $[0, r]$:

$$
K(r) = 2\pi \int_0^r u (0) \, du = 0 \quad \text{for all } r \le R_{\text{core}}
$$

Evaluating Besag's $L$-function:

$$
L(r) = \sqrt{\frac{K(r)}{\pi}} = \sqrt{\frac{0}{\pi}} = 0 \quad \text{for all } r \le R_{\text{core}}
$$

Subtracting $r$ yields the centered diagnostic profile:

$$
L(r) - r = 0 - r = -r \quad \text{for all } r \le R_{\text{core}}
$$

At small lags within the exclusion zone, the centered $L$-function decreases linearly along the line with slope $-1.0$.

*Part 3: Discontinuity of the Derivative.*  
By the fundamental theorem of calculus, the derivative of Ripley's $K$-function is:

$$
\frac{d}{dr} K(r) = 2\pi r g(r)
$$

Approaching the exclusion radius from the left:

$$
\lim_{r \to R_{\text{core}}^-} \frac{d}{dr} K(r) = 2\pi R_{\text{core}} (0) = 0
$$

Just beyond the hard-core radius ($r > R_{\text{core}}$), points can pack against each other, so $g(R_{\text{core}}^+) \ge 1$:

$$
\lim_{r \to R_{\text{core}}^+} \frac{d}{dr} K(r) = 2\pi R_{\text{core}} g(R_{\text{core}}^+) > 0
$$

The derivative undergoes an instantaneous jump of magnitude $2\pi R_{\text{core}} g(R_{\text{core}}^+)$ at $r = R_{\text{core}}$, producing a sharp elbow in the empirical curve of $\widehat{K}(r)$ that precisely identifies the physical packing radius of the events. $\blacksquare$

---

### Exercise 11.3: Moments and Ripley's $K(r)$ for a Log-Gaussian Cox Process with Exponential Covariance

**Problem Statement:**  
Let $\Phi$ be a Log-Gaussian Cox Process on $\mathbb{R}^2$ driven by a stationary Gaussian field $Y(\mathbf{s}) \sim \mathcal{GP}(\mu, C_Y(h))$ with an Exponential covariance kernel:

$$
C_Y(h) = \sigma_Y^2 \exp\left( -\frac{h}{a} \right), \quad \sigma_Y^2 > 0, \quad a > 0
$$

1. Calculate the first-order intensity $\lambda$ and express the pair correlation function $g(r)$ in terms of $\sigma_Y^2, a$, and $r$.
2. Formulate the integral expression for Ripley's $K(r)$.
3. Using the linear approximation $e^x \approx 1 + x$ for small variance $\sigma_Y^2 \ll 1$, derive an explicit closed-form expression for $K(r)$ and demonstrate that $K(r) > \pi r^2$ for all $r > 0$.

**Solution:**

*Part 1: Intensity and Pair Correlation.*  
The variance of the Gaussian random field is $C_Y(0) = \sigma_Y^2$. By Theorem 11.5.1, the first-order intensity is:

$$
\lambda = \mathbb{E}[\Lambda(\mathbf{s})] = \exp\left( \mu + \frac{1}{2} \sigma_Y^2 \right)
$$

The pair correlation function is:

$$
g(r) = \exp(C_Y(r)) = \exp\left( \sigma_Y^2 \exp\left(-\frac{r}{a}\right) \right)
$$

*Part 2: Integral for $K(r)$.*  
Substituting $g(r)$ into the definition of Ripley's $K$-function:

$$
K(r) = 2\pi \int_0^r u \, g(u) \, du = 2\pi \int_0^r u \exp\left( \sigma_Y^2 \exp\left(-\frac{u}{a}\right) \right) du
$$

*Part 3: Closed-Form Approximation and Proof of Clustering.*  
For small latent variance $\sigma_Y^2$, we expand the outer exponential function via Taylor series:

$$
\exp\left( \sigma_Y^2 e^{-u/a} \right) \approx 1 + \sigma_Y^2 e^{-u/a}
$$

Substituting this approximation into the integral:

$$
K(r) \approx 2\pi \int_0^r u \left( 1 + \sigma_Y^2 e^{-u/a} \right) du = 2\pi \int_0^r u \, du + 2\pi \sigma_Y^2 \int_0^r u e^{-u/a} \, du
$$

The first term is the CSR baseline:

$$
2\pi \int_0^r u \, du = \pi r^2
$$

The second integral is evaluated using integration by parts: let $v = u$ and $dw = e^{-u/a} du$, so $dv = du$ and $w = -a e^{-u/a}$:

$$
\int_0^r u e^{-u/a} \, du = \left[ -a u e^{-u/a} \right]_0^r - \int_0^r (-a e^{-u/a}) du = -a r e^{-r/a} + a \left[ -a e^{-u/a} \right]_0^r
$$

$$
= -a r e^{-r/a} - a^2 e^{-r/a} + a^2 = a^2 \left( 1 - \left( 1 + \frac{r}{a} \right) e^{-r/a} \right)
$$

Therefore:

$$
K(r) \approx \pi r^2 + 2\pi \sigma_Y^2 a^2 \left[ 1 - \left( 1 + \frac{r}{a} \right) e^{-r/a} \right]
$$

To prove that $K(r) > \pi r^2$ for all $r > 0$, consider the function $h(x) = 1 - (1 + x) e^{-x}$ where $x = r/a > 0$.  
At $x = 0$, $h(0) = 1 - (1) e^0 = 0$. The derivative is:

$$
h'(x) = -\left[ (1) e^{-x} + (1 + x)(-e^{-x}) \right] = -e^{-x} + (1 + x)e^{-x} = x e^{-x}
$$

Because $x > 0$ and $e^{-x} > 0$, the derivative $h'(x) > 0$ is strictly positive for all $x > 0$.  
By the mean value theorem, $h(x) > 0$ for all $x > 0$. Since $\sigma_Y^2 > 0$ and $a > 0$:

$$
2\pi \sigma_Y^2 a^2 \left[ 1 - \left( 1 + \frac{r}{a} \right) e^{-r/a} \right] > 0 \implies K(r) > \pi r^2 \quad \text{for all } r > 0
$$

This completes the proof: an LGCP with a non-negative Exponential covariance kernel exhibits positive spatial clustering across all separation scales. $\blacksquare$

---

## 11.8 Scope Boundaries and Beyond

Spatial point pattern analysis is a vast statistical discipline. To maintain focus within our linear-models scope, we highlight several important advanced topics that lie beyond this introductory treatment:

1. **Marked Point Processes:** In many practical applications, each point carries an auxiliary random attribute $m_i \in \mathcal{M}$ called a **mark** (for example, tree trunk diameter, seismic magnitude, or ore grade). Marked point process theory investigates whether the spatial locations $\mathbf{x}_i$ and the marks $m_i$ are independent, using mark correlation functions and cross-$K$ functions.
2. **Nonparametric Intensity Smoothing:** While we focused on parametric GLM models for $\lambda(\mathbf{s})$, nonparametric 2D kernel density estimation is widely used in exploratory analysis:
   $$\widehat{\lambda}_b(\mathbf{s}) = \sum_{i=1}^n \frac{k_b(\mathbf{s} - \mathbf{x}_i)}{w_b(\mathbf{x}_i)}$$
   where $k_b$ is a bivariate Gaussian kernel and $b > 0$ is the bandwidth parameter. Optimal bandwidth selection (e.g., cross-validation or Diggle's method) is critical to prevent over-smoothing.
3. **Preferential Sampling in Geostatistics:** In standard geostatistics (Chapters 1–10), we assume that the sampling coordinates $\mathbf{s}_1, \dots, \mathbf{s}_n$ are chosen independently of the underlying random field $Z(\mathbf{s})$. In reality, mining engineers drill more boreholes where mineral concentrations are high, and environmental agencies place pollution monitors where air quality is poor. When the point process of sampling locations $\Phi$ is stochastically dependent on the continuous spatial field $Z(\mathbf{s})$, ordinary kriging is biased. Diggle, Menezes, and Su (2010) resolved this via shared latent Gaussian fields—uniting point processes and geostatistics into a single joint likelihood.

In the next chapter, we transition from continuous point coordinates to [lattice and areal data](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch12_lattice_areal_models.md) (Chapter 12), where spatial phenomena are aggregated into discrete geographic polygons, counties, and administrative zones.
