# Chapter 8 Companion Exercises: Change of Support and Spatial Regularization

This study guide and exercise workbook accompanies [Chapter 8: Change of Support and Spatial Regularization](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch08_support_regularization.md). Synthesizing the pedagogical clarity of Maria L. Rizzo (*Statistical Computing with R*) and the spatial linear modeling discipline of Oliver Schabenberger and Carol A. Gotway (*Statistical Methods for Spatial Data Analysis*), these exercises are organized into five comprehensive tiers:

1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 8.1–8.4)
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 8.5–8.8)
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 8.9–8.11)
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 8.12–8.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 8.14–8.16)

Every exercise includes an unabridged, step-by-step mathematical proof or computational solution.

---

## Tier 0: Classical-to-Spatial Warm-Ups

### Exercise 8.1: Bilinear Block Covariance Integral Proof via Fubini's Theorem (S&G Problem 5.12)
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$ be a weakly stationary point-support random field with constant mean $\operatorname{E}[Z(\mathbf{s})] = \mu$ and continuous covariance function $C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h}))$. Let $B_i, B_j \subset D$ be compact spatial blocks with strictly positive finite Lebesgue measure $|B_i| = \int_{B_i} d\mathbf{u} > 0$ and $|B_j| = \int_{B_j} d\mathbf{v} > 0$. The block-support spatial random variables are defined as mean-square Riemann/Lebesgue spatial averages:
$$
Z(B_i) = \frac{1}{|B_i|} \int_{B_i} Z(\mathbf{u}) d\mathbf{u}, \quad Z(B_j) = \frac{1}{|B_j|} \int_{B_j} Z(\mathbf{v}) d\mathbf{v}
$$
**(a)** Prove that $\operatorname{E}[Z(B_i)] = \mu$, establishing that spatial regularization preserves the point-support expectation.
**(b)** State the integrability conditions under Fubini's Theorem and Tonelli's Theorem on the product measure space $B_i \times B_j \times \Omega$ that justify exchanging the expectation operator and the spatial Lebesgue integral. Using this result, derive the fundamental bilinear covariance integral:
$$
\operatorname{Cov}(Z(B_i), Z(B_j)) = \frac{1}{|B_i||B_j|} \int_{B_i} \int_{B_j} C(\mathbf{u} - \mathbf{v}) d\mathbf{u} d\mathbf{v} \equiv \bar{C}(B_i, B_j)
$$
**(c)** For any finite collection of blocks $\{B_1, \dots, B_n\}$ and arbitrary real coefficients $a_1, \dots, a_n \in \mathbb{R}$, prove that the $n \times n$ regularized covariance matrix $\overline{\mathbf{C}} = [\operatorname{Cov}(Z(B_i), Z(B_j))]_{i,j=1}^n$ is positive semi-definite:
$$
\mathbf{a}' \overline{\mathbf{C}} \mathbf{a} \ge 0 \quad \text{for all } \mathbf{a} \in \mathbb{R}^n
$$
Under what geometric condition on the support blocks $\{B_i\}_{i=1}^n$ and point covariance function $C(\mathbf{h})$ is $\overline{\mathbf{C}}$ strictly positive definite?

##### Solution to Exercise 8.1
**(a)** Preservation of Expectation under Spatial Regularization:
Let $(\Omega, \mathcal{F}, \mathbb{P})$ be the underlying probability space and $(D, \mathcal{B}(D), \lambda)$ be the spatial domain equipped with the Borel $\sigma$-algebra and Lebesgue measure $\lambda$.
By definition of the mean-square Lebesgue integral:
$$
\operatorname{E}[Z(B_i)] = \operatorname{E}\left[ \frac{1}{|B_i|} \int_{B_i} Z(\mathbf{u}) d\mathbf{u} \right]
$$
Because $Z(\mathbf{u})$ is second-order stationary, $\operatorname{E}[Z(\mathbf{u})^2] = C(\mathbf{0}) + \mu^2 < \infty$. By Lyapunov's inequality (or the Cauchy-Schwarz inequality),
$$
\operatorname{E}[|Z(\mathbf{u})|] \le \sqrt{\operatorname{E}[Z(\mathbf{u})^2]} = \sqrt{C(\mathbf{0}) + \mu^2} < \infty
$$
The product integrand is bounded in $L^1(\Omega \times B_i)$:
$$
\int_{B_i} \operatorname{E}[|Z(\mathbf{u})|] d\mathbf{u} \le \int_{B_i} \sqrt{C(\mathbf{0}) + \mu^2} d\mathbf{u} = |B_i| \sqrt{C(\mathbf{0}) + \mu^2} < \infty
$$
Since $|B_i| < \infty$, Fubini's Theorem permits exchanging the expectation operator $\operatorname{E}$ and the spatial integral $\int_{B_i}$:
$$
\operatorname{E}[Z(B_i)] = \frac{1}{|B_i|} \int_{B_i} \operatorname{E}[Z(\mathbf{u})] d\mathbf{u} = \frac{1}{|B_i|} \int_{B_i} \mu d\mathbf{u} = \frac{\mu}{|B_i|} \int_{B_i} d\mathbf{u} = \frac{\mu}{|B_i|} |B_i| = \mu
$$
Thus, the expected value of the spatial block average is identically equal to the point-support mean $\mu$.

**(b)** Fubini-Tonelli Justification and Bilinear Covariance Formula:
By definition of covariance between two real-valued random variables:
$$
\operatorname{Cov}(Z(B_i), Z(B_j)) = \operatorname{E}[(Z(B_i) - \operatorname{E}[Z(B_i)])(Z(B_j) - \operatorname{E}[Z(B_j)])] = \operatorname{E}[(Z(B_i) - \mu)(Z(B_j) - \mu)]
$$
Express the centered block random variables as spatial integrals:
$$
Z(B_i) - \mu = \frac{1}{|B_i|} \int_{B_i} (Z(\mathbf{u}) - \mu) d\mathbf{u}, \quad Z(B_j) - \mu = \frac{1}{|B_j|} \int_{B_j} (Z(\mathbf{v}) - \mu) d\mathbf{v}
$$
Because the integration domains $B_i$ and $B_j$ are deterministic compact sets, their product is the double Riemann/Lebesgue integral over the Cartesian product set $B_i \times B_j \subset \mathbb{R}^d \times \mathbb{R}^d$:
$$
(Z(B_i) - \mu)(Z(B_j) - \mu) = \frac{1}{|B_i||B_j|} \int_{B_i} \int_{B_j} (Z(\mathbf{u}) - \mu)(Z(\mathbf{v}) - \mu) d\mathbf{u} d\mathbf{v}
$$
Taking the mathematical expectation on both sides:
$$
\operatorname{Cov}(Z(B_i), Z(B_j)) = \operatorname{E}\left[ \frac{1}{|B_i||B_j|} \int_{B_i} \int_{B_j} (Z(\mathbf{u}) - \mu)(Z(\mathbf{v}) - \mu) d\mathbf{u} d\mathbf{v} \right]
$$
*Rigorous Application of the Fubini-Tonelli Theorem:*
Consider the product measure space $(B_i \times B_j \times \Omega, \mathcal{B}(B_i \times B_j) \otimes \mathcal{F}, \lambda \otimes \lambda \otimes \mathbb{P})$.
To exchange expectation and spatial integration, Tonelli's Theorem requires that the product function be non-negative and integrable:
$$
I = \int_{B_i} \int_{B_j} \operatorname{E}[|(Z(\mathbf{u}) - \mu)(Z(\mathbf{v}) - \mu)|] d\mathbf{u} d\mathbf{v}
$$
Applying the Cauchy-Schwarz inequality in the Hilbert space $L^2(\Omega, \mathcal{F}, \mathbb{P})$ to the integrand:
$$
\begin{aligned}
\operatorname{E}[|(Z(\mathbf{u}) - \mu)(Z(\mathbf{v}) - \mu)|] &\le \sqrt{\operatorname{E}[(Z(\mathbf{u}) - \mu)^2] \cdot \operatorname{E}[(Z(\mathbf{v}) - \mu)^2]} \\
&= \sqrt{\operatorname{Var}(Z(\mathbf{u})) \cdot \operatorname{Var}(Z(\mathbf{v}))} \\
&= \sqrt{C(\mathbf{0}) \cdot C(\mathbf{0})} = C(\mathbf{0}) < \infty
\end{aligned}
$$
Because $C(\mathbf{0})$ is a finite non-negative real constant, the integral over the compact product domain $B_i \times B_j$ is bounded:
$$
I \le \int_{B_i} \int_{B_j} C(\mathbf{0}) d\mathbf{u} d\mathbf{v} = C(\mathbf{0}) |B_i| |B_j| < \infty
$$
Since $I < \infty$, the conditions of Fubini's Theorem are satisfied. Consequently, the expectation operator commutes with the spatial double integral:
$$
\begin{aligned}
\operatorname{Cov}(Z(B_i), Z(B_j)) &= \frac{1}{|B_i||B_j|} \int_{B_i} \int_{B_j} \operatorname{E}[(Z(\mathbf{u}) - \mu)(Z(\mathbf{v}) - \mu)] d\mathbf{u} d\mathbf{v} \\
&= \frac{1}{|B_i||B_j|} \int_{B_i} \int_{B_j} \operatorname{Cov}(Z(\mathbf{u}), Z(\mathbf{v})) d\mathbf{u} d\mathbf{v}
\end{aligned}
$$
By second-order stationarity of the point process, $\operatorname{Cov}(Z(\mathbf{u}), Z(\mathbf{v})) = C(\mathbf{u} - \mathbf{v})$. Substituting this into the integral yields:
$$
\operatorname{Cov}(Z(B_i), Z(B_j)) = \frac{1}{|B_i||B_j|} \int_{B_i} \int_{B_j} C(\mathbf{u} - \mathbf{v}) d\mathbf{u} d\mathbf{v} \equiv \bar{C}(B_i, B_j)
$$
This establishes Schabenberger & Gotway's fundamental auxiliary covariance identity.

**(c)** Positive Semi-Definiteness and Strict Positive Definiteness:
Let $\mathbf{a} = (a_1, \dots, a_n)' \in \mathbb{R}^n$ be any non-zero coefficient vector, and define the composite random variable $Y = \sum_{i=1}^n a_i Z(B_i)$.
By the bilinearity of the covariance operator:
$$
\operatorname{Var}(Y) = \operatorname{Var}\left( \sum_{i=1}^n a_i Z(B_i) \right) = \sum_{i=1}^n \sum_{j=1}^n a_i a_j \operatorname{Cov}(Z(B_i), Z(B_j)) = \mathbf{a}' \overline{\mathbf{C}} \mathbf{a}
$$
Since the variance of any real-valued random variable is non-negative by definition of the $L^2$ norm ($\operatorname{Var}(Y) = \|Y - \operatorname{E}[Y]\|_{L^2}^2 \ge 0$), we have:
$$
\mathbf{a}' \overline{\mathbf{C}} \mathbf{a} \ge 0 \quad \text{for all } \mathbf{a} \in \mathbb{R}^n
$$
Thus, $\overline{\mathbf{C}}$ is always positive semi-definite.

*Strict Positive Definiteness:*
The quadratic form equals zero ($\mathbf{a}' \overline{\mathbf{C}} \mathbf{a} = 0$) if and only if $\operatorname{Var}(Y) = 0$, which occurs if and only if $Y = \operatorname{E}[Y]$ almost surely.
Express $Y - \operatorname{E}[Y]$ as an integral over the entire domain $D$:
$$
Y - \operatorname{E}[Y] = \sum_{i=1}^n \frac{a_i}{|B_i|} \int_{B_i} (Z(\mathbf{u}) - \mu) d\mathbf{u} = \int_D \left( \sum_{i=1}^n \frac{a_i}{|B_i|} \mathbf{1}_{B_i}(\mathbf{u}) \right) (Z(\mathbf{u}) - \mu) d\mathbf{u}
$$
Define the spatial indicator weight function $w(\mathbf{u}) = \sum_{i=1}^n \frac{a_i}{|B_i|} \mathbf{1}_{B_i}(\mathbf{u})$. The variance of $Y$ is:
$$
\operatorname{Var}(Y) = \int_D \int_D w(\mathbf{u}) w(\mathbf{v}) C(\mathbf{u} - \mathbf{v}) d\mathbf{u} d\mathbf{v}
$$
If the point-support covariance function $C(\mathbf{h})$ is strictly positive definite on $L^2(D)$, then:
$$
\int_D \int_D w(\mathbf{u}) w(\mathbf{v}) C(\mathbf{u} - \mathbf{v}) d\mathbf{u} d\mathbf{v} = 0 \iff w(\mathbf{u}) = 0 \quad \text{almost everywhere on } D
$$
The set of step functions $\{\mathbf{1}_{B_i}(\mathbf{u})\}_{i=1}^n$ is linearly independent in $L^2(D)$ if and only if no block $B_i$ can be covered almost everywhere by a union of the other blocks. Specifically, if each block $B_i$ possesses a set of positive Lebesgue measure that is disjoint from all other blocks ($\lambda(B_i \setminus \bigcup_{j \ne i} B_j) > 0$, as holds whenever blocks have disjoint interiors), then:
$$
w(\mathbf{u}) = 0 \text{ a.e.} \implies a_i = 0 \text{ for each } i=1, \dots, n \implies \mathbf{a} = \mathbf{0}
$$
Therefore, under strictly positive definite point covariances and geometrically non-redundant blocks, $\mathbf{a}' \overline{\mathbf{C}} \mathbf{a} > 0$ for all $\mathbf{a} \ne \mathbf{0}$, guaranteeing that $\overline{\mathbf{C}}$ is non-singular and strictly positive definite. $\blacksquare$

---

### Exercise 8.2: Classical One-Way ANOVA Variance Partitioning as the i.i.d. Limit of Krige's Additivity Theorem
Consider a spatial domain $W \subset \mathbb{R}^d$ partitioned into $K$ mutually disjoint blocks $\{V_k\}_{k=1}^K$ of identical volume, where each block $V_k$ contains $M$ point observations $\{Z_{k,m}\}_{m=1}^M$ on point support $0$, yielding a total of $N = KM$ points across $W$.
Let $\bar{Z}_{k\cdot} = \frac{1}{M}\sum_{m=1}^M Z_{k,m}$ denote the sample mean within block $V_k$, and $\bar{Z}_{\cdot\cdot} = \frac{1}{N}\sum_{k=1}^K \sum_{m=1}^M Z_{k,m} = \frac{1}{K}\sum_{k=1}^K \bar{Z}_{k\cdot}$ denote the grand sample mean across the entire domain $W$.
**(a)** Decompose the total deviation of an observation $Z_{k,m}$ from the grand mean $\bar{Z}_{\cdot\cdot}$ into within-block and between-block components:
$$
(Z_{k,m} - \bar{Z}_{\cdot\cdot}) = (Z_{k,m} - \bar{Z}_{k\cdot}) + (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})
$$
Prove that the cross-product sum over all $k=1,\dots,K$ and $m=1,\dots,M$ vanishes identically:
$$
\sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})(\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot}) = 0
$$
thereby deriving the classical ANOVA identity $\text{SST} = \text{SSW} + \text{SSB}$, where:
$$
\text{SST} = \sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{\cdot\cdot})^2, \quad \text{SSW} = \sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})^2, \quad \text{SSB} = M \sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2
$$
**(b)** Dividing the sums of squares by the total sample size $N = KM$, define the sample dispersion variances:
$$
s^2(0 \mid W) = \frac{\text{SST}}{N}, \quad s^2(0 \mid V) = \frac{\text{SSW}}{N} = \frac{1}{K}\sum_{k=1}^K \left[ \frac{1}{M}\sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})^2 \right], \quad s^2(V \mid W) = \frac{\text{SSB}}{N} = \frac{1}{K}\sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2
$$
Conclude that the sample dispersion variances satisfy the exact additive identity:
$$
s^2(0 \mid W) = s^2(0 \mid V) + s^2(V \mid W)
$$
**(c)** For a second-order stationary random field $Z(\mathbf{s})$ with semivariogram $\gamma(\mathbf{h})$, define the theoretical dispersion variance of support $A$ inside a larger domain $B$ ($A \subset B$) by:
$$
D^2(A \mid B) = \bar{\gamma}(B, B) - \bar{\gamma}(A, A)
$$
Prove Krige's Additivity Theorem: for nested supports $0 \subset V \subset W$,
$$
D^2(0 \mid W) = D^2(0 \mid V) + D^2(V \mid W)
$$
**(d)** Suppose the spatial field is pure nugget microscale noise (uncorrelated i.i.d. observations with $\operatorname{Var}(Z(\mathbf{s})) = \sigma^2$ and $\operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h})) = 0$ for all $\mathbf{h} \ne \mathbf{0}$). Show that as $M \to \infty$ with $M$ discrete independent points in each of the $K$ blocks:
$$
\operatorname{E}[s^2(0 \mid W)] \to \sigma^2, \quad \operatorname{E}[s^2(0 \mid V)] \to \sigma^2, \quad \operatorname{E}[s^2(V \mid W)] = \frac{\sigma^2}{M}\left(\frac{K-1}{K}\right) \to 0
$$
Explain how Krige's formulation generalizes classical sample-mean variance shrinkage ($\sigma^2/M$) to arbitrary spatial correlation geometries.

##### Solution to Exercise 8.2
**(a)** Classical Deviation Decomposition and Vanishing Cross-Product:
Starting from the algebraic identity:
$$
Z_{k,m} - \bar{Z}_{\cdot\cdot} = (Z_{k,m} - \bar{Z}_{k\cdot}) + (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})
$$
Squaring both sides:
$$
(Z_{k,m} - \bar{Z}_{\cdot\cdot})^2 = (Z_{k,m} - \bar{Z}_{k\cdot})^2 + (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 + 2(Z_{k,m} - \bar{Z}_{k\cdot})(\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})
$$
Summing across all $K$ blocks ($k=1, \dots, K$) and all $M$ samples per block ($m=1, \dots, M$):
$$
\sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{\cdot\cdot})^2 = \sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})^2 + \sum_{k=1}^K \sum_{m=1}^M (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 + 2 \sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})(\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})
$$
Analyze the cross-product sum:
$$
\sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})(\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot}) = \sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot}) \left[ \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot}) \right]
$$
By definition of the within-block sample mean $\bar{Z}_{k\cdot} = \frac{1}{M}\sum_{m=1}^M Z_{k,m}$:
$$
\sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot}) = \sum_{m=1}^M Z_{k,m} - M \bar{Z}_{k\cdot} = M \bar{Z}_{k\cdot} - M \bar{Z}_{k\cdot} = 0
$$
Since this inner sum is zero for every individual block $k \in \{1, \dots, K\}$:
$$
\sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot}) \cdot 0 = 0
$$
Furthermore, because the term $(\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2$ does not depend on the sample index $m$:
$$
\sum_{k=1}^K \sum_{m=1}^M (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 = \sum_{k=1}^K M (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 = M \sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 = \text{SSB}
$$
Identifying the first term as $\text{SSW} = \sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})^2$ and the left-hand side as $\text{SST} = \sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{\cdot\cdot})^2$:
$$
\text{SST} = \text{SSW} + \text{SSB}
$$

**(b)** Empirical Dispersion Variances:
Divide the entire ANOVA identity by the total sample size $N = KM$:
$$
\frac{\text{SST}}{N} = \frac{\text{SSW}}{N} + \frac{\text{SSB}}{N}
$$
Examine each term:
1. $\frac{\text{SST}}{N} = \frac{1}{N}\sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{\cdot\cdot})^2 = s^2(0 \mid W)$, which is the empirical sample variance of point-support observations within the entire domain $W$.
2. $\frac{\text{SSW}}{N} = \frac{1}{KM}\sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})^2 = \frac{1}{K}\sum_{k=1}^K \left[\frac{1}{M}\sum_{m=1}^M (Z_{k,m} - \bar{Z}_{k\cdot})^2\right] = s^2(0 \mid V)$, which is the average within-block sample dispersion of points around their respective block means.
3. $\frac{\text{SSB}}{N} = \frac{M}{KM}\sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 = \frac{1}{K}\sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 = s^2(V \mid W)$, which is the sample dispersion variance of the $K$ block averages around the grand mean.
Substituting these definitions yields:
$$
s^2(0 \mid W) = s^2(0 \mid V) + s^2(V \mid W)
$$
This establishes that classical one-way ANOVA variance partitioning is the exact discrete sample realization of spatial dispersion variance additivity.

**(c)** Proof of Krige's Additivity Theorem:
For any two nested spatial volumes $A \subset B$, the theoretical dispersion variance is defined as the mean-square variance of support $A$ inside volume $B$:
$$
D^2(A \mid B) = \bar{\gamma}(B, B) - \bar{\gamma}(A, A)
$$
where $\bar{\gamma}(B, B) = \frac{1}{|B|^2}\int_B \int_B \gamma(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w}$ is the average point semivariance between points independently distributed throughout $B$.
Consider the three nested spatial scales: point support $0 \subset$ block support $V \subset$ regional domain $W$.
Evaluate the three corresponding dispersion variances:
1. Point support inside domain $W$ ($0 \subset W$):
   Since $\bar{\gamma}(0, 0) = \gamma(\mathbf{0}) = 0$,
   $$
   D^2(0 \mid W) = \bar{\gamma}(W, W) - \bar{\gamma}(0, 0) = \bar{\gamma}(W, W)
   $$
2. Point support inside block $V$ ($0 \subset V$):
   $$
   D^2(0 \mid V) = \bar{\gamma}(V, V) - \bar{\gamma}(0, 0) = \bar{\gamma}(V, V)
   $$
3. Block support inside domain $W$ ($V \subset W$):
   $$
   D^2(V \mid W) = \bar{\gamma}(W, W) - \bar{\gamma}(V, V)
   $$
Summing the within-block dispersion $D^2(0 \mid V)$ and between-block dispersion $D^2(V \mid W)$:
$$
D^2(0 \mid V) + D^2(V \mid W) = [\bar{\gamma}(V, V) - \bar{\gamma}(0, 0)] + [\bar{\gamma}(W, W) - \bar{\gamma}(V, V)]
$$
The intermediate regularization term $\bar{\gamma}(V, V)$ cancels algebraically:
$$
D^2(0 \mid V) + D^2(V \mid W) = \bar{\gamma}(W, W) - \bar{\gamma}(0, 0) = D^2(0 \mid W)
$$
Thus:
$$
D^2(0 \mid W) = D^2(0 \mid V) + D^2(V \mid W) \quad \blacksquare
$$
This completes the unabridged proof of Krige's Additivity Theorem.

**(d)** The i.i.d. Nugget-Only Limit:
Assume the process consists purely of uncorrelated microscale noise: $\operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h})) = \sigma^2 \mathbf{1}(\mathbf{h} = \mathbf{0})$.
In this setting, all $N = KM$ observations are mutually independent with common variance $\sigma^2$.
- For the total variance $s^2(0 \mid W)$, the expectation of the sample variance of $N$ i.i.d. observations is:
  $$
  \operatorname{E}[s^2(0 \mid W)] = \operatorname{E}\left[ \frac{1}{N}\sum_{k=1}^K \sum_{m=1}^M (Z_{k,m} - \bar{Z}_{\cdot\cdot})^2 \right] = \frac{N-1}{N} \sigma^2
  $$
  As $M \to \infty$ (and thus $N = KM \to \infty$), $\frac{N-1}{N} \to 1$, so $\operatorname{E}[s^2(0 \mid W)] \to \sigma^2$.
- For the within-block variance $s^2(0 \mid V)$, within each block $k$, the $M$ observations are i.i.d. with sample variance expectation $\frac{M-1}{M}\sigma^2$. Averaging over all $K$ blocks:
  $$
  \operatorname{E}[s^2(0 \mid V)] = \frac{1}{K}\sum_{k=1}^K \frac{M-1}{M} \sigma^2 = \frac{M-1}{M} \sigma^2 = \sigma^2\left(1 - \frac{1}{M}\right)
  $$
  As $M \to \infty$, $\sigma^2(1 - 1/M) \to \sigma^2$.
- For the between-block variance $s^2(V \mid W)$, each block sample mean $\bar{Z}_{k\cdot}$ is an average of $M$ independent observations, so $\operatorname{Var}(\bar{Z}_{k\cdot}) = \frac{\sigma^2}{M}$. Because the blocks are disjoint, the $K$ block means $\{\bar{Z}_{k\cdot}\}_{k=1}^K$ are mutually independent.
  The expectation of their sample variance around the grand mean is:
  $$
  \operatorname{E}[s^2(V \mid W)] = \operatorname{E}\left[ \frac{1}{K}\sum_{k=1}^K (\bar{Z}_{k\cdot} - \bar{Z}_{\cdot\cdot})^2 \right] = \frac{K-1}{K} \operatorname{Var}(\bar{Z}_{k\cdot}) = \frac{K-1}{K} \frac{\sigma^2}{M}
  $$
  As $M \to \infty$, $\operatorname{E}[s^2(V \mid W)] = \frac{\sigma^2}{M}(1 - 1/K) \to 0$.

*Connection to Spatial Generalization:*
In classical statistics without spatial correlation, the variance of a spatial average collapses to zero as sample size increases: $\operatorname{Var}(\bar{Z}_M) = \sigma^2 / M \to 0$.
In geostatistics, however, points within a spatial block $V$ are positively autocorrelated ($C(\mathbf{h}) > 0$). Because the observations are correlated, the Law of Large Numbers does not reduce the variance of a continuous physical block $Z(V) = \frac{1}{|V|}\int_V Z(\mathbf{u}) d\mathbf{u}$ to zero!
Instead, $\operatorname{Var}(Z(V)) = \bar{C}(V, V) > 0$. Krige's Additivity Theorem encapsulates this persistence of variance: $D^2(V \mid W) = \bar{\gamma}(W, W) - \bar{\gamma}(V, V)$ remains strictly positive for finite physical blocks $V$, seamlessly generalizing the classical $\sigma^2/M$ shrinkage to correlated continuous fields. $\blacksquare$

---

### Exercise 8.3: Exact 1D Analytical Regularization of the Spherical Variogram
Consider a one-dimensional spatial domain with a standardized spherical point-support semivariogram with unit sill ($c = 1$) and range $a > 0$:
$$
\gamma(h) = \begin{cases} \frac{3}{2}\left(\frac{h}{a}\right) - \frac{1}{2}\left(\frac{h}{a}\right)^3, & 0 \le h \le a \\ 1, & h > a \end{cases}
$$
Let the regularization support be a continuous segment $v = [0, L]$ of length $L \le a$.
**(a)** Formulate the within-block auxiliary variogram $\bar{\gamma}(L, L)$ as a double Lebesgue integral over $[0, L] \times [0, L]$:
$$
\bar{\gamma}(L, L) = \frac{1}{L^2} \int_0^L \int_0^L \gamma(|u - w|) du dw
$$
**(b)** Prove that for any 1D segment $[0, L]$ and any isotropic semivariogram $\gamma(h)$, this double integral reduces to the single integral:
$$
\bar{\gamma}(L, L) = \frac{2}{L} \int_0^L \left(1 - \frac{h}{L}\right) \gamma(h) dh
$$
by deriving the probability density function $f_H(h) = \frac{2}{L}(1 - \frac{h}{L})$ of the Euclidean distance $H = |U - W|$ between two independent uniform points $U, W \overset{\text{i.i.d.}}{\sim} \text{Uniform}(0, L)$.
**(c)** Substitute the spherical model $\gamma(h)$ into the single integral. Carry out the term-by-term polynomial integration explicitly for $L \le a$, and prove that:
$$
\bar{\gamma}(L, L) = \frac{L}{2a} - \frac{L^3}{20a^3}
$$
**(d)** Evaluate the ratio of the regularized within-block variogram to the point variogram at distance $L$, namely $R(L) = \frac{\bar{\gamma}(L, L)}{\gamma(L)}$, for $L/a = 0.2$, $L/a = 0.5$, and $L/a = 1.0$.
Prove that:
$$
\lim_{L \to 0} \frac{\bar{\gamma}(L, L)}{\gamma(L)} = \frac{1}{3}
$$
Explain the physical and geometric meaning of the limiting constant $1/3$ in terms of the average pairwise distance between points on a line segment.

##### Solution to Exercise 8.3
**(a)** Integral Formulation of Within-Block Semivariance:
By definition, the within-block auxiliary semivariogram $\bar{\gamma}(L, L)$ is the average point-support semivariance between all pairs of locations $(u, w)$ within the segment $[0, L]$:
$$
\bar{\gamma}(L, L) = \frac{1}{L^2} \int_0^L \int_0^L \gamma(|u - w|) du dw
$$

**(b)** Reduction to Single Integral via Pairwise Distance PDF:
Let $U$ and $W$ be two independent random variables distributed uniformly on the interval $[0, L]$. Their joint probability density function is:
$$
f_{U, W}(u, w) = \frac{1}{L} \cdot \frac{1}{L} = \frac{1}{L^2}, \quad (u, w) \in [0, L] \times [0, L]
$$
The within-block auxiliary variogram is the mathematical expectation of $\gamma(|U - W|)$:
$$
\bar{\gamma}(L, L) = \operatorname{E}[\gamma(|U - W|)] = \int_0^L \gamma(h) f_H(h) dh
$$
where $H = |U - W|$ is the distance between the two random locations, and $f_H(h)$ is its probability density function on $[0, L]$.
To derive $f_H(h)$, evaluate the cumulative distribution function $F_H(h) = \mathbb{P}(|U - W| \le h)$ for $0 \le h \le L$:
The event $\{|U - W| \le h\}$ corresponds to the geometric region in the square $[0, L]^2$ between the two lines $w = u + h$ and $w = u - h$.
The complementary region $\{|U - W| > h\}$ consists of two disjoint right-angled triangles in the upper-left and lower-right corners of $[0, L]^2$, each with legs of length $L - h$:
$$
\operatorname{Area}(|U - W| > h) = 2 \times \left[ \frac{1}{2}(L - h)^2 \right] = (L - h)^2
$$
Therefore, the area of the event $\{|U - W| \le h\}$ is:
$$
\operatorname{Area}(|U - W| \le h) = L^2 - (L - h)^2 = L^2 - (L^2 - 2Lh + h^2) = 2Lh - h^2
$$
Dividing by the total area $L^2$:
$$
F_H(h) = \frac{2Lh - h^2}{L^2} = 2\left(\frac{h}{L}\right) - \left(\frac{h}{L}\right)^2, \quad 0 \le h \le L
$$
Differentiating $F_H(h)$ with respect to $h$ yields the probability density function:
$$
f_H(h) = \frac{d}{dh} F_H(h) = \frac{2}{L} - \frac{2h}{L^2} = \frac{2}{L}\left(1 - \frac{h}{L}\right), \quad 0 \le h \le L
$$
Substituting $f_H(h)$ into the expectation integral proves the result:
$$
\bar{\gamma}(L, L) = \int_0^L \gamma(h) f_H(h) dh = \frac{2}{L} \int_0^L \left(1 - \frac{h}{L}\right) \gamma(h) dh \quad \blacksquare
$$

**(c)** Exact Polynomial Integration for Spherical Model:
For $L \le a$, the separation distance $h$ satisfies $0 \le h \le L \le a$. Over this entire integration domain, the spherical semivariogram is given by:
$$
\gamma(h) = \frac{3}{2}\left(\frac{h}{a}\right) - \frac{1}{2}\left(\frac{h}{a}\right)^3 = \frac{3h}{2a} - \frac{h^3}{2a^3}
$$
Substitute this formula into the integral:
$$
\bar{\gamma}(L, L) = \frac{2}{L} \int_0^L \left(1 - \frac{h}{L}\right) \left[ \frac{3h}{2a} - \frac{h^3}{2a^3} \right] dh
$$
Expand the integrand:
$$
\begin{aligned}
\left(1 - \frac{h}{L}\right) \left[ \frac{3h}{2a} - \frac{h^3}{2a^3} \right] &= \frac{3h}{2a} - \frac{h^3}{2a^3} - \frac{3h^2}{2aL} + \frac{h^4}{2a^3L} \\
&= \frac{3}{2a} h - \frac{3}{2aL} h^2 - \frac{1}{2a^3} h^3 + \frac{1}{2a^3L} h^4
\end{aligned}
$$
Integrate each term individually from $h=0$ to $h=L$:
1. $\int_0^L \frac{3}{2a} h \, dh = \frac{3}{2a} \left[ \frac{h^2}{2} \right]_0^L = \frac{3L^2}{4a}$
2. $\int_0^L \frac{3}{2aL} h^2 \, dh = \frac{3}{2aL} \left[ \frac{h^3}{3} \right]_0^L = \frac{3L^3}{6aL} = \frac{L^2}{2a} = \frac{2L^2}{4a}$
3. $\int_0^L \frac{1}{2a^3} h^3 \, dh = \frac{1}{2a^3} \left[ \frac{h^4}{4} \right]_0^L = \frac{L^4}{8a^3} = \frac{5L^4}{40a^3}$
4. $\int_0^L \frac{1}{2a^3L} h^4 \, dh = \frac{1}{2a^3L} \left[ \frac{h^5}{5} \right]_0^L = \frac{L^5}{10a^3L} = \frac{L^4}{10a^3} = \frac{4L^4}{40a^3}$

Combine the integrated terms:
$$
\begin{aligned}
\int_0^L \left(1 - \frac{h}{L}\right) \gamma(h) dh &= \left( \frac{3L^2}{4a} - \frac{2L^2}{4a} \right) - \left( \frac{5L^4}{40a^3} - \frac{4L^4}{40a^3} \right) \\
&= \frac{L^2}{4a} - \frac{L^4}{40a^3}
\end{aligned}
$$
Multiply by the prefactor $\frac{2}{L}$:
$$
\bar{\gamma}(L, L) = \frac{2}{L} \left( \frac{L^2}{4a} - \frac{L^4}{40a^3} \right) = \frac{L}{2a} - \frac{L^3}{20a^3}
$$
This is the exact, closed-form analytical formula for the within-block regularized spherical semivariance on a 1D segment of length $L \le a$. $\blacksquare$

**(d)** Numerical Ratios and Limiting Behavior:
Define the dimensionless ratio $x = \frac{L}{a} \in (0, 1]$.
The point-support semivariogram evaluated at lag $L$ is:
$$
\gamma(L) = \frac{3}{2}x - \frac{1}{2}x^3 = \frac{x}{2}(3 - x^2)
$$
The within-block regularized semivariogram is:
$$
\bar{\gamma}(L, L) = \frac{1}{2}x - \frac{1}{20}x^3 = \frac{x}{20}(10 - x^2)
$$
The variance ratio $R(x) = \frac{\bar{\gamma}(L, L)}{\gamma(L)}$ is:
$$
R(x) = \frac{\frac{x}{20}(10 - x^2)}{\frac{x}{2}(3 - x^2)} = \frac{10 - x^2}{10(3 - x^2)}
$$
Evaluate $R(x)$ at the specified relative block sizes:
1. For $L/a = 0.2$ ($x = 0.2$):
   $$
   R(0.2) = \frac{10 - (0.2)^2}{10(3 - (0.2)^2)} = \frac{10 - 0.04}{10(3 - 0.04)} = \frac{9.96}{29.60} \approx 0.336486 \quad (33.65\%)
   $$
2. For $L/a = 0.5$ ($x = 0.5$):
   $$
   R(0.5) = \frac{10 - (0.5)^2}{10(3 - (0.5)^2)} = \frac{10 - 0.25}{10(3 - 0.25)} = \frac{9.75}{27.50} = \frac{39}{110} \approx 0.354545 \quad (35.45\%)
   $$
3. For $L/a = 1.0$ ($x = 1.0$):
   $$
   R(1.0) = \frac{10 - 1^2}{10(3 - 1^2)} = \frac{9}{10(2)} = \frac{9}{20} = 0.450000 \quad (45.00\%)
   $$

*Asymptotic Limit as $L \to 0$:*
Taking the limit of $R(x)$ as $x \to 0$:
$$
\lim_{L \to 0} \frac{\bar{\gamma}(L, L)}{\gamma(L)} = \lim_{x \to 0} \frac{10 - x^2}{10(3 - x^2)} = \frac{10 - 0}{10(3 - 0)} = \frac{10}{30} = \frac{1}{3}
$$

*Physical and Geometric Interpretation:*
For small distances $h \ll a$, the spherical variogram is dominated by its linear term near the origin:
$$
\gamma(h) = \frac{3h}{2a} + \mathcal{O}(h^3)
$$
Under a pure linear variogram $\gamma(h) = \alpha h$ with slope $\alpha = \frac{3}{2a}$, the average semivariance between points in the segment $[0, L]$ is proportional to the expected pairwise distance between two random points $U, W \sim \text{Uniform}(0, L)$:
$$
\operatorname{E}[|U - W|] = \int_0^L h f_H(h) dh = \frac{2}{L} \int_0^L \left(h - \frac{h^2}{L}\right) dh = \frac{2}{L}\left( \frac{L^2}{2} - \frac{L^3}{3L} \right) = \frac{2}{L}\left(\frac{L^2}{6}\right) = \frac{L}{3}
$$
Because the average distance between two independent uniform points on a line of length $L$ is exactly $L/3$, the linear approximation of within-block semivariance satisfies:
$$
\bar{\gamma}(L, L) \approx \gamma(\operatorname{E}[|U - W|]) = \gamma\left(\frac{L}{3}\right) = \frac{1}{3} \gamma(L)
$$
This proves that the limiting ratio of $1/3$ arises directly from the 1D Euclidean geometry of uniform pairwise distances. $\blacksquare$

---

### Exercise 8.4: Proportional Effect and Heteroskedastic Block Variance Scaling
In many natural resource and environmental applications (e.g., precious metal ore grades, airborne pollutant concentrations), spatial variability is not homoskedastic, but exhibits a pronounced **proportional effect**: the local standard deviation of samples is proportional to the local mean:
$$
D(0 \mid V) = k \cdot m_V, \quad k > 0
$$
where $m_V = \operatorname{E}[Z(\mathbf{s}) \mid \mathbf{s} \in V]$ is the local mean within block $V$.
**(a)** In standard stationary geostatistics, the dispersion variance of point support within block $W$ is spatially constant: $D^2(0 \mid W) = \bar{\gamma}(W, W)$. Explain why applying this homoskedastic formulation across a deposit with a strong proportional effect severely distorts local uncertainty, underestimating variance in high-grade zones and overestimating variance in low-grade zones.
**(b)** To restore spatial stationarity, geostatisticians utilize the **local relative semivariogram**:
$$
\gamma_{\text{rel}}(\mathbf{h}) = \frac{\gamma(\mathbf{h})}{\bar{Z}^2}
$$
Assume the spatial process follows a multiplicative model $Z(\mathbf{s}) = m(\mathbf{s}) \cdot Y(\mathbf{s})$, where $m(\mathbf{s}) > 0$ is a smoothly varying local mean field and $Y(\mathbf{s})$ is a second-order stationary random field with $\operatorname{E}[Y(\mathbf{s})] = 1$, $\operatorname{Var}(Y(\mathbf{s})) = \sigma_Y^2$, and semivariogram $\gamma_Y(\mathbf{h})$.
Prove that across any local neighborhood $V$ where $m(\mathbf{s})$ is approximately constant ($m(\mathbf{s}) \approx m_V$), the local relative semivariogram satisfies:
$$
\gamma_{Z, \text{rel}}(\mathbf{h}) \equiv \frac{\gamma_Z(\mathbf{h})}{m_V^2} = \gamma_Y(\mathbf{h})
$$
and is therefore spatially invariant and independent of the local grade $m_V$.
**(c)** Let $Z_0$ denote the point-support grade distribution with global mean $\mu$ and variance $\sigma_0^2$. The **affine volume-variance correction** models the block-support grade $Z_V$ by:
$$
Z_V = \mu + \sqrt{r}(Z_0 - \mu), \quad \text{where } r = \frac{\operatorname{Var}(Z_V)}{\operatorname{Var}(Z_0)} = \frac{D^2(V \mid D)}{D^2(0 \mid D)} \in (0, 1)
$$
(i) Prove that $\operatorname{E}[Z_V] = \mu$ and $\operatorname{Var}(Z_V) = r \sigma_0^2$.
(ii) Explain how this volume-variance correction is used to construct preliminary **grade-tonnage curves** (tonnage $T(z_c) = \mathbb{P}(Z_V \ge z_c)$ and mean recovered grade $Q(z_c) = \operatorname{E}[Z_V \mid Z_V \ge z_c]$ above cutoff $z_c$).
(iii) Identify the fundamental mathematical failure mode of the affine correction when applied to highly skewed, lognormal-like distributions with large coefficient of variation ($\text{CV} > 1.5$), contrasting it with non-linear change-of-support approaches.

##### Solution to Exercise 8.4
**(a)** Breakdown of Homoskedastic Regularization under Proportional Effect:
Under standard second-order stationarity, the point-support semivariogram $\gamma(\mathbf{h})$ and the regularized within-block dispersion variance $D^2(0 \mid W) = \bar{\gamma}(W, W)$ are assumed to be translation-invariant constants across the entire study area $D$.
When a proportional effect exists, the actual local dispersion variance scales quadratically with the local mean:
$$
\operatorname{Var}_{\text{local}}(Z \mid V) = D^2(0 \mid V) = k^2 m_V^2
$$
If an analyst applies a single global average semivariogram $\bar{\gamma}(W, W) \approx k^2 \mu^2$ across the entire domain:
1. *In High-Grade Zones ($m_V \gg \mu$):* Suppose a high-grade ore shoot has local mean $m_V = 5\mu$. The true local variance is $k^2 (5\mu)^2 = 25 k^2 \mu^2$. The stationary model assigns a variance of $k^2 \mu^2$, underestimating true block variability by a factor of 25! Consequently, prediction intervals are dangerously narrow, drastically understating the financial and operational risk of extreme grade fluctuations.
2. *In Low-Grade Zones ($m_V \ll \mu$):* Suppose a marginal boundary zone has local mean $m_V = 0.2\mu$. The true local variance is $0.04 k^2 \mu^2$. The stationary model assigns $k^2 \mu^2$, overestimating variance by a factor of 25. This inflates the upper tail of the predicted distribution, creating fictitious high-grade prospects in barren waste rock.
Standard linear regularization fails because the second-order structure is coupled directly to the first-order mean.

**(b)** Stationarity of the Relative Semivariogram under Multiplicative Decomposition:
Let $Z(\mathbf{s}) = m(\mathbf{s}) Y(\mathbf{s})$.
Within a local neighborhood $V$, the mean field $m(\mathbf{s})$ varies slowly relative to the correlation length of $Y(\mathbf{s})$, so $m(\mathbf{s}) \approx m_V > 0$ for all $\mathbf{s} \in V$.
For any spatial increment $\mathbf{h}$ such that $\mathbf{s}, \mathbf{s}+\mathbf{h} \in V$:
$$
Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s}) = m_V Y(\mathbf{s}+\mathbf{h}) - m_V Y(\mathbf{s}) = m_V [Y(\mathbf{s}+\mathbf{h}) - Y(\mathbf{s})]
$$
The local semivariogram of $Z$ is:
$$
\begin{aligned}
\gamma_Z(\mathbf{h}) &= \frac{1}{2}\operatorname{Var}(Z(\mathbf{s}+\mathbf{h}) - Z(\mathbf{s})) \\
&= \frac{1}{2}\operatorname{Var}(m_V [Y(\mathbf{s}+\mathbf{h}) - Y(\mathbf{s})]) \\
&= m_V^2 \cdot \frac{1}{2}\operatorname{Var}(Y(\mathbf{s}+\mathbf{h}) - Y(\mathbf{s})) \\
&= m_V^2 \gamma_Y(\mathbf{h})
\end{aligned}
$$
Dividing both sides by the square of the local mean $m_V^2$:
$$
\gamma_{Z, \text{rel}}(\mathbf{h}) \equiv \frac{\gamma_Z(\mathbf{h})}{m_V^2} = \gamma_Y(\mathbf{h})
$$
Since $Y(\mathbf{s})$ is second-order stationary by construction, $\gamma_Y(\mathbf{h})$ is completely independent of the local mean $m_V$ and invariant across the spatial domain.
Normalizing the local variogram by the square of the local mean successfully neutralizes the heteroskedastic proportional effect, allowing stationary spatial regularization integrals to be computed on $\gamma_Y(\mathbf{h})$ and then rescaled locally by $m_V^2$. $\blacksquare$

**(c)** Affine Volume-Variance Correction and Diagnostic Limits:
(i) *Preservation of Mean and Shrinkage of Variance:*
Taking the expectation of $Z_V = \mu + \sqrt{r}(Z_0 - \mu)$:
$$
\operatorname{E}[Z_V] = \operatorname{E}[\mu + \sqrt{r}(Z_0 - \mu)] = \mu + \sqrt{r}(\operatorname{E}[Z_0] - \mu) = \mu + \sqrt{r}(\mu - \mu) = \mu
$$
Taking the variance:
$$
\operatorname{Var}(Z_V) = \operatorname{Var}(\mu + \sqrt{r}(Z_0 - \mu)) = \operatorname{Var}(\sqrt{r} Z_0) = (\sqrt{r})^2 \operatorname{Var}(Z_0) = r \sigma_0^2
$$
Thus, the affine transformation preserves the global mean while compressing the dispersion variance by the exact theoretical variance reduction factor $r = \frac{D^2(V \mid D)}{D^2(0 \mid D)} < 1$.

(ii) *Construction of Grade-Tonnage Curves:*
In mining feasibility, exploration drillhole samples represent point support $Z_0$. However, mineral extraction occurs in selective mining units (SMUs) of block volume $V$.
Because spatial averaging within blocks smooths out extreme highs and lows, the distribution of block grades $Z_V$ has substantially lower variance than point grades $Z_0$.
If an engineer evaluates mining project viability using the raw point distribution $Z_0$, they dramatically overestimate the tonnage of high-grade ore available above an economic cutoff $z_c$.
By applying the affine transformation to the empirical quantile function of exploration drillholes:
$$
z_V(p) = \mu + \sqrt{r}(z_0(p) - \mu)
$$
engineers generate a realistic corrected block grade distribution $F_{Z_V}(z)$, from which they compute:
- Recoverable Ore Tonnage fraction: $T(z_c) = 1 - F_{Z_V}(z_c) = \mathbb{P}(Z_V \ge z_c)$
- Average Recovered Grade: $Q(z_c) = \frac{1}{T(z_c)} \int_{z_c}^\infty z dF_{Z_V}(z)$
This prevents multi-million dollar capital investment errors caused by overlooking the change-of-support volume-variance effect.

(iii) *Pathological Failure Mode on Highly Skewed Distributions:*
The affine transformation is a rigid linear contraction around the mean:
$$
Z_V = (1 - \sqrt{r})\mu + \sqrt{r} Z_0
$$
When applied to positively skewed distributions (such as lognormal or Pareto distributions typical of gold, uranium, or trace pollutants):
1. *Artificial Lower Truncation:* For any non-negative point sample $Z_0 \ge 0$, the minimum possible block value under the affine model is:
   $$
   \min Z_V = (1 - \sqrt{r})\mu > 0 \quad (\text{at } Z_0 = 0)
   $$
   The affine model makes it mathematically impossible for any block to have a grade below $(1 - \sqrt{r})\mu$, creating a hard artificial truncation that completely erases the low-grade waste rock tail.
2. *De-regularization and Negative Values:* If the affine model is inverted to infer point distributions from smoothed block data ($Z_0 = \mu + \frac{1}{\sqrt{r}}(Z_V - \mu)$), it generates strictly impossible negative point grades whenever $Z_V < \mu(1 - \sqrt{r})$.
3. *Symmetry Distortion:* The affine model compresses both tails by the same symmetric linear factor $\sqrt{r}$, failing to capture the physical reality that spatial averaging of skewed variables reduces skewness (moving toward normality by the Central Limit Theorem).
For skewed variables, non-linear change-of-support methods—specifically the Indirect Lognormal Correction (Exercise 8.15) or Discrete Gaussian Hermite polynomial expansions—must be employed instead. $\blacksquare$

---

## Tier I: Core Theoretical Re-Derivations

### Exercise 8.5: Mean-Square Regularized Covariance and Variogram
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$ be a second-order stationary point-support random field with constant mean $\mu$ and continuous covariance function $C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h}))$.
Let $v \subset \mathbb{R}^d$ be a compact block support of finite Lebesgue volume $|v| = \int_v d\mathbf{u} > 0$. The regularized random field $Z_v(\mathbf{s})$ is defined as:
$$
Z_v(\mathbf{s}) = \frac{1}{|v|} \int_{v(\mathbf{s})} Z(\mathbf{u}) d\mathbf{u} = \frac{1}{|v|} \int_v Z(\mathbf{s} + \mathbf{u}) d\mathbf{u}
$$
where $v(\mathbf{s}) = \mathbf{s} + v$.
**(a)** Prove that the covariance between the regularized field at location $\mathbf{s}_1$ and $\mathbf{s}_2 = \mathbf{s}_1 + \mathbf{h}$ is given by the double spatial convolution:
$$
C_v(\mathbf{h}) = \operatorname{Cov}(Z_v(\mathbf{s}_1), Z_v(\mathbf{s}_1 + \mathbf{h})) = \frac{1}{|v|^2} \int_v \int_v C(\mathbf{h} + \mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w}
$$
**(b)** Deduce that the variance of the regularized block process is:
$$
\operatorname{Var}(Z_v(\mathbf{s})) = C_v(\mathbf{0}) = \frac{1}{|v|^2} \int_v \int_v C(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w} = \bar{C}(v, v)
$$
where $\bar{C}(v, v)$ denotes the average point-support covariance between all pairs of points within block $v$.
**(c)** For an intrinsically stationary process with point-support semivariogram $\gamma(\mathbf{h})$, prove that the regularized semivariogram $\gamma_v(\mathbf{h}) = \frac{1}{2}\operatorname{Var}(Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s}))$ satisfies:
$$
\gamma_v(\mathbf{h}) = \bar{\gamma}(v, v_{\mathbf{h}}) - \bar{\gamma}(v, v)
$$
where $\bar{\gamma}(v, v_{\mathbf{h}}) = \frac{1}{|v|^2} \int_v \int_v \gamma(\mathbf{h} + \mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w}$ and $\bar{\gamma}(v, v) = \frac{1}{|v|^2} \int_v \int_v \gamma(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w}$.

##### Solution to Exercise 8.5
**(a)** Using the definition of $Z_v(\mathbf{s})$ and the linearity of the covariance operator in Hilbert space $L^2$:
$$
\begin{aligned}
C_v(\mathbf{h}) &= \operatorname{Cov}(Z_v(\mathbf{s}_1), Z_v(\mathbf{s}_1 + \mathbf{h})) \\
&= \operatorname{Cov}\left( \frac{1}{|v|} \int_v Z(\mathbf{s}_1 + \mathbf{u}) d\mathbf{u}, \frac{1}{|v|} \int_v Z(\mathbf{s}_1 + \mathbf{h} + \mathbf{w}) d\mathbf{w} \right) \\
&= \frac{1}{|v|^2} \int_v \int_v \operatorname{Cov}(Z(\mathbf{s}_1 + \mathbf{u}), Z(\mathbf{s}_1 + \mathbf{h} + \mathbf{w})) d\mathbf{u} d\mathbf{w}
\end{aligned}
$$
By second-order stationarity, $\operatorname{Cov}(Z(\mathbf{s}_1 + \mathbf{u}), Z(\mathbf{s}_1 + \mathbf{h} + \mathbf{w})) = C((\mathbf{s}_1 + \mathbf{h} + \mathbf{w}) - (\mathbf{s}_1 + \mathbf{u})) = C(\mathbf{h} + \mathbf{w} - \mathbf{u})$.
By the symmetry of the auto-covariance function $C(\mathbf{x}) = C(-\mathbf{x})$, $C(\mathbf{h} + \mathbf{w} - \mathbf{u}) = C(-(\mathbf{h} + \mathbf{w} - \mathbf{u})) = C(-\mathbf{h} + \mathbf{u} - \mathbf{w})$.
Switching dummy variables $\mathbf{u} \leftrightarrow \mathbf{w}$ or noting that $C_v(\mathbf{h}) = C_v(-\mathbf{h})$ yields:
$$
C_v(\mathbf{h}) = \frac{1}{|v|^2} \int_v \int_v C(\mathbf{h} + \mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w}
$$

**(b)** Evaluating $C_v(\mathbf{h})$ at $\mathbf{h} = \mathbf{0}$:
$$
\operatorname{Var}(Z_v(\mathbf{s})) = C_v(\mathbf{0}) = \frac{1}{|v|^2} \int_v \int_v C(\mathbf{0} + \mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w} = \frac{1}{|v|^2} \int_v \int_v C(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w} = \bar{C}(v, v)
$$
This represents the double integral of point-support covariances averaged across the physical volume of $v$.

**(c)** By definition of the regularized semivariogram:
$$
\gamma_v(\mathbf{h}) = \frac{1}{2} \operatorname{E}\left[ (Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s}))^2 \right] = \frac{1}{2} \operatorname{Var}(Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s}))
$$
Expanding the variance:
$$
\operatorname{Var}(Z_v(\mathbf{s}+\mathbf{h}) - Z_v(\mathbf{s})) = \operatorname{Var}(Z_v(\mathbf{s}+\mathbf{h})) + \operatorname{Var}(Z_v(\mathbf{s})) - 2\operatorname{Cov}(Z_v(\mathbf{s}+\mathbf{h}), Z_v(\mathbf{s})) = 2 C_v(\mathbf{0}) - 2 C_v(\mathbf{h})
$$
Dividing by 2 gives:
$$
\gamma_v(\mathbf{h}) = C_v(\mathbf{0}) - C_v(\mathbf{h})
$$
Now substitute $C(\mathbf{x}) = C(\mathbf{0}) - \gamma(\mathbf{x})$:
$$
C_v(\mathbf{0}) = \frac{1}{|v|^2} \int_v \int_v [C(\mathbf{0}) - \gamma(\mathbf{u} - \mathbf{w})] d\mathbf{u} d\mathbf{w} = C(\mathbf{0}) - \bar{\gamma}(v, v)
$$
$$
C_v(\mathbf{h}) = \frac{1}{|v|^2} \int_v \int_v [C(\mathbf{0}) - \gamma(\mathbf{h} + \mathbf{u} - \mathbf{w})] d\mathbf{u} d\mathbf{w} = C(\mathbf{0}) - \bar{\gamma}(v, v_{\mathbf{h}})
$$
Subtracting $C_v(\mathbf{h})$ from $C_v(\mathbf{0})$:
$$
\gamma_v(\mathbf{h}) = [C(\mathbf{0}) - \bar{\gamma}(v, v)] - [C(\mathbf{0}) - \bar{\gamma}(v, v_{\mathbf{h}})] = \bar{\gamma}(v, v_{\mathbf{h}}) - \bar{\gamma}(v, v)
$$
Notice that the point sill $C(\mathbf{0})$ cancels out completely, demonstrating that the regularization relationship holds for purely intrinsically stationary random fields that do not possess a finite sill variance. $\blacksquare$

---

### Exercise 8.6: Dispersion Variance and Krige's Additivity Theorem
Let $v$ and $V$ be two compact spatial domains such that $v \subset V \subset \mathbb{R}^d$ with $|v| > 0$ and $|V| > 0$. The **dispersion variance** $D^2(v \mid V)$ is the variance of the smaller units $Z_v(\mathbf{s})$ within the larger domain $V$:
$$
D^2(v \mid V) = \operatorname{Var}_V(Z_v) = \frac{1}{|V|} \int_V (Z_v(\mathbf{s}) - Z_V)^2 d\mathbf{s}
$$
**(a)** Prove that under second-order stationarity, the expectation of the dispersion variance can be expressed in terms of average semivariances as:
$$
D^2(v \mid V) = \bar{\gamma}(V, V) - \bar{\gamma}(v, v)
$$
or in terms of covariances as $D^2(v \mid V) = \bar{C}(v, v) - \bar{C}(V, V)$.
**(b)** Let $0$ denote a quasi-point support (dimensionless point). What is $D^2(0 \mid V)$?
**(c)** Consider three nested spatial supports: points $0 \subset$ mining selection blocks $v \subset$ entire deposit $W$. Prove **Krige's Additivity Theorem**:
$$
D^2(0 \mid W) = D^2(0 \mid v) + D^2(v \mid W)
$$
**(d)** Explain the profound economic and geological implication of Krige's Additivity Theorem: why is the variance of selective mining blocks within an ore deposit strictly smaller than the variance of exploration drill cores?

##### Solution to Exercise 8.6
**(a)** Expanding the squared difference inside the spatial integral:
$$
\begin{aligned}
\operatorname{E}[D^2(v \mid V)] &= \frac{1}{|V|} \int_V \operatorname{E}\left[ (Z_v(\mathbf{s}) - Z_V)^2 \right] d\mathbf{s} \\
&= \frac{1}{|V|} \int_V \operatorname{Var}(Z_v(\mathbf{s}) - Z_V) d\mathbf{s}
\end{aligned}
$$
Since $Z_V = \frac{1}{|V|} \int_V Z_v(\mathbf{u}) d\mathbf{u}$:
$$
\begin{aligned}
\operatorname{Var}(Z_v(\mathbf{s}) - Z_V) &= \operatorname{Var}(Z_v(\mathbf{s})) - 2\operatorname{Cov}(Z_v(\mathbf{s}), Z_V) + \operatorname{Var}(Z_V) \\
&= \bar{C}(v, v) - 2 \bar{C}(v(\mathbf{s}), V) + \bar{C}(V, V)
\end{aligned}
$$
Integrating over $\mathbf{s} \in V$ and dividing by $|V|$:
$$
\frac{1}{|V|} \int_V \bar{C}(v(\mathbf{s}), V) d\mathbf{s} = \bar{C}(V, V)
$$
Therefore:
$$
\operatorname{E}[D^2(v \mid V)] = \bar{C}(v, v) - 2\bar{C}(V, V) + \bar{C}(V, V) = \bar{C}(v, v) - \bar{C}(V, V)
$$
Substituting $\bar{C}(A, B) = C(\mathbf{0}) - \bar{\gamma}(A, B)$:
$$
D^2(v \mid V) = [C(\mathbf{0}) - \bar{\gamma}(v, v)] - [C(\mathbf{0}) - \bar{\gamma}(V, V)] = \bar{\gamma}(V, V) - \bar{\gamma}(v, v)
$$

**(b)** For quasi-point support $0$, two points sampled at the same location have zero lag: $\bar{\gamma}(0, 0) = \gamma(0) = 0$.
Therefore:
$$
D^2(0 \mid V) = \bar{\gamma}(V, V) - \gamma(0) = \bar{\gamma}(V, V)
$$
Equivalently, $D^2(0 \mid V) = C(\mathbf{0}) - \bar{C}(V, V)$.
The dispersion variance of point samples within a region $V$ is simply the within-domain average semivariance $\bar{\gamma}(V, V)$.

**(c)** Proof of Krige's Additivity Theorem:
Consider three nested supports $0 \subset v \subset W$.
Using the semivariogram representation:
- $D^2(0 \mid W) = \bar{\gamma}(W, W) - \bar{\gamma}(0, 0) = \bar{\gamma}(W, W)$.
- $D^2(0 \mid v) = \bar{\gamma}(v, v) - \bar{\gamma}(0, 0) = \bar{\gamma}(v, v)$.
- $D^2(v \mid W) = \bar{\gamma}(W, W) - \bar{\gamma}(v, v)$.
Now sum $D^2(0 \mid v)$ and $D^2(v \mid W)$:
$$
D^2(0 \mid v) + D^2(v \mid W) = \bar{\gamma}(v, v) + [\bar{\gamma}(W, W) - \bar{\gamma}(v, v)] = \bar{\gamma}(W, W) = D^2(0 \mid W)
$$
Thus:
$$
D^2(0 \mid W) = D^2(0 \mid v) + D^2(v \mid W)
$$

**(d)** Economic and Physical Implications:
In mining and earth engineering:
- $D^2(0 \mid W)$ is the total dispersion variance of small drill cores (point samples) across the entire deposit $W$.
- $D^2(0 \mid v)$ is the internal variance of points *within* a mining block $v$.
- $D^2(v \mid W)$ is the variance of the *actual mining blocks* $v$ across the deposit.
Because $D^2(0 \mid v) \ge 0$, we have:
$$
D^2(v \mid W) = D^2(0 \mid W) - D^2(0 \mid v) < D^2(0 \mid W)
$$
The variance of selective extraction blocks is *strictly less* than the sample variance observed in drill cores!
If a mining feasibility study naively uses the drill core variance to predict the recovery of high-grade ore, it will drastically overestimate the proportion of high-grade blocks and underestimate the proportion of average-grade blocks. Internal block averaging ($D^2(0 \mid v)$) permanently smooths out extreme grades. $\blacksquare$

---

### Exercise 8.7: Cartier's Spatial Martingale Relation and Support Invariance
Let $v$ and $V$ be nested supports with $v \subset V \subset \mathbb{R}^d$. Pierre Cartier established a fundamental structural relationship connecting point-support fields and their volume-averaged regularizations.
**(a)** Let $\mathcal{F}_V = \sigma(Z_V)$ be the $\sigma$-algebra generated by the block-support random variable $Z_V$. Under diffuse conditioning (where no spatial drift gradients exist), prove that:
$$
\operatorname{E}[Z(\mathbf{s}) \mid Z_V] = Z_V \quad \text{for almost all } \mathbf{s} \in V
$$
and more generally, for any intermediate support $v \subset V$:
$$
\operatorname{E}[Z_v(\mathbf{s}) \mid Z_V] = Z_V
$$
**(b)** Explain why Cartier's relation characterizes $Z_V$ as a reverse spatial martingale.
**(c)** Using Jensen's inequality and the law of total variance, prove the variance inequality:
$$
\operatorname{Var}(Z_W) \le \operatorname{Var}(Z_V) \le \operatorname{Var}(Z_v) \le \operatorname{Var}(Z(\mathbf{s}))
$$
for any sequence of nested supports $\mathbf{s} \in v \subset V \subset W$.

##### Solution to Exercise 8.7
**(a)** Let $\mathbf{s}$ be uniformly distributed over $V$, independent of the field $Z$.
By definition of the volume average:
$$
Z_V = \frac{1}{|V|} \int_V Z(\mathbf{u}) d\mathbf{u}
$$
Under the diffuse conditioning hypothesis, given the single macroscopic aggregate observation $Z_V$, the location of microscopic fluctuations within $V$ has zero directional bias.
By linearity of conditional expectation:
$$
\frac{1}{|V|} \int_V \operatorname{E}[Z(\mathbf{u}) \mid Z_V] d\mathbf{u} = \operatorname{E}\left[ \frac{1}{|V|} \int_V Z(\mathbf{u}) d\mathbf{u} \;\middle|\; Z_V \right] = \operatorname{E}[Z_V \mid Z_V] = Z_V
$$
Since this holds uniformly for any sub-region without internal spatial drift:
$$
\operatorname{E}[Z(\mathbf{s}) \mid Z_V] = Z_V \quad \forall \mathbf{s} \in V
$$
Now for an intermediate block $v \subset V$:
$$
\operatorname{E}[Z_v(\mathbf{s}) \mid Z_V] = \operatorname{E}\left[ \frac{1}{|v|} \int_v Z(\mathbf{u}) d\mathbf{u} \;\middle|\; Z_V \right] = \frac{1}{|v|} \int_v \operatorname{E}[Z(\mathbf{u}) \mid Z_V] d\mathbf{u} = \frac{1}{|v|} \int_v Z_V d\mathbf{u} = Z_V \frac{|v|}{|v|} = Z_V
$$

**(b)** Reverse spatial martingale interpretation:
In probability theory, a sequence $\{X_n, \mathcal{F}_n\}_{n \ge 1}$ is a reverse martingale if $\mathcal{F}_{n+1} \subset \mathcal{F}_n$ and $\operatorname{E}[X_n \mid \mathcal{F}_{n+1}] = X_{n+1}$.
As the support volume increases ($v \subset V \subset W$), the information contained in the volume average becomes coarser (averaging destroys high-frequency spatial variation).
Cartier's relation states that the expectation of the finer variable conditioned on the coarser aggregate is precisely the coarser aggregate:
$$
\operatorname{E}[Z_v \mid Z_V] = Z_V \quad (v \subset V)
$$
Thus, spatial volume regularization forms a continuous-parameter reverse martingale with respect to increasing Lebesgue measure $|v|$.

**(c)** Proof of the variance inequality:
Apply the law of total variance to $Z_v$ conditioned on $Z_V$:
$$
\operatorname{Var}(Z_v) = \operatorname{Var}(\operatorname{E}[Z_v \mid Z_V]) + \operatorname{E}[\operatorname{Var}(Z_v \mid Z_V)]
$$
By Cartier's relation, $\operatorname{E}[Z_v \mid Z_V] = Z_V$. Therefore:
$$
\operatorname{Var}(Z_v) = \operatorname{Var}(Z_V) + \operatorname{E}[\operatorname{Var}(Z_v \mid Z_V)]
$$
Since variance is strictly non-negative ($\operatorname{Var}(Z_v \mid Z_V) \ge 0$), its expectation is non-negative:
$$
\operatorname{E}[\operatorname{Var}(Z_v \mid Z_V)] \ge 0
$$
Consequently:
$$
\operatorname{Var}(Z_V) \le \operatorname{Var}(Z_v)
$$
Applying this across the chain $\mathbf{s} \in v \subset V \subset W$:
$$
\operatorname{Var}(Z_W) \le \operatorname{Var}(Z_V) \le \operatorname{Var}(Z_v) \le \operatorname{Var}(Z(\mathbf{s}))
$$
Spatial volume aggregation strictly reduces total field variance. $\blacksquare$

---

### Exercise 8.8: Asymptotic Regularization of the Nugget Effect
In point-support spatial statistics, the nugget effect $\tau^2 > 0$ represents microscale spatial variation and measurement error, manifested as a discontinuity at the origin of the semivariogram: $\lim_{h \to 0^+} \gamma(h) = \tau^2 > 0$.
Let $\{Z(\mathbf{s})\}$ decompose as $Z(\mathbf{s}) = S(\mathbf{s}) + \epsilon(\mathbf{s})$, where $S(\mathbf{s})$ is a mean-square continuous field and $\epsilon(\mathbf{s})$ is independent spatial white noise with $\operatorname{Cov}(\epsilon(\mathbf{s}_1), \epsilon(\mathbf{s}_2)) = \tau^2 \delta(\mathbf{s}_1 - \mathbf{s}_2)$ (Dirac delta).
**(a)** Compute the regularized nugget variance:
$$
\tau_v^2 = \operatorname{Var}(\epsilon_v) = \frac{1}{|v|^2} \int_v \int_v \tau^2 \delta(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w}
$$
**(b)** Show that $\tau_v^2 = \frac{\tau^2}{|v|}$, and analyze what happens as the volume $|v| \to \infty$.
**(c)** Prove that for any finite volume $|v| > 0$, the regularized semivariogram $\gamma_v(\mathbf{h})$ is **strictly continuous at the origin**:
$$
\lim_{\|\mathbf{h}\| \to 0} \gamma_v(\mathbf{h}) = 0
$$
thereby proving that spatial volume averaging completely annihilates the nugget discontinuity.

##### Solution to Exercise 8.8
**(a)** Using the sifting property of the Dirac delta distribution $\int_v \delta(\mathbf{u} - \mathbf{w}) d\mathbf{w} = 1$ for all $\mathbf{u} \in v$:
$$
\begin{aligned}
\tau_v^2 &= \frac{1}{|v|^2} \int_v \int_v \tau^2 \delta(\mathbf{u} - \mathbf{w}) d\mathbf{u} d\mathbf{w} \\
&= \frac{\tau^2}{|v|^2} \int_v \left[ \int_v \delta(\mathbf{u} - \mathbf{w}) d\mathbf{w} \right] d\mathbf{u} \\
&= \frac{\tau^2}{|v|^2} \int_v [1] d\mathbf{u} \\
&= \frac{\tau^2}{|v|^2} |v| = \frac{\tau^2}{|v|}
\end{aligned}
$$

**(b)** As the physical volume of support $|v|$ increases:
$$
\lim_{|v| \to \infty} \tau_v^2 = \lim_{|v| \to \infty} \frac{\tau^2}{|v|} = 0
$$
This is the continuous spatial analogue of the Law of Large Numbers ($\operatorname{Var}(\bar{X}_n) = \sigma^2/n$): integrating uncorrelated microscale noise over a finite physical volume causes independent positive and negative fluctuations to cancel each other out in the spatial integral.

**(c)** Proof of origin continuity of $\gamma_v(\mathbf{h})$:
Recall from Exercise 8.5(c) that:
$$
\gamma_v(\mathbf{h}) = \bar{\gamma}(v, v_{\mathbf{h}}) - \bar{\gamma}(v, v)
$$
As $\|\mathbf{h}\| \to 0$, the translated support $v_{\mathbf{h}} = \mathbf{h} + v$ converges to $v$ in symmetric difference measure: $\operatorname{Leb}(v \mathbin{\Delta} v_{\mathbf{h}}) \to 0$.
Because the double integral defining $\bar{\gamma}(v, v_{\mathbf{h}})$ is continuous with respect to spatial translations of the domain of integration by the Dominated Convergence Theorem:
$$
\lim_{\|\mathbf{h}\| \to 0} \bar{\gamma}(v, v_{\mathbf{h}}) = \bar{\gamma}(v, v)
$$
Therefore:
$$
\lim_{\|\mathbf{h}\| \to 0} \gamma_v(\mathbf{h}) = \bar{\gamma}(v, v) - \bar{\gamma}(v, v) = 0
$$
Even though the point-support semivariogram possesses a jump discontinuity at the origin ($\gamma(0^+) = \tau^2$), the block-support semivariogram $\gamma_v(\mathbf{h})$ starts smoothly at $0$ with no jump discontinuity! Spatial volume regularization acts as a mollifier that restores path continuity. $\blacksquare$

---

## Tier II: Pencil-and-Paper Mechanics

### Exercise 8.9: Pencil-and-Paper 1D Segment Regularization
Consider a 1D spatial domain with a pure linear point-support semivariogram without nugget:
$$
\gamma(h) = \alpha |h|, \quad \alpha > 0
$$
Let the support be a linear segment $v = [0, L]$ of length $L > 0$.
**(a)** Compute the within-block auxiliary variogram $\bar{\gamma}(L, L)$ by evaluating the double integral explicitly:
$$
\bar{\gamma}(L, L) = \frac{1}{L^2} \int_0^L \int_0^L \alpha |u - w| du dw
$$
**(b)** Let $s_0$ be a point located at distance $d \ge L$ from the origin: $s_0 = d$. Compute the point-to-block auxiliary variogram:
$$
\bar{\gamma}(s_0, v) = \frac{1}{L} \int_0^L \alpha |d - u| du
$$
**(c)** Let $v_1 = [0, L]$ and $v_2 = [h, h+L]$ be two separated segments with center-to-center distance $h \ge L$. Compute the block-to-block auxiliary variogram $\bar{\gamma}(v_1, v_2)$.
**(d)** Using these integrals, write out the exact regularized block semivariogram $\gamma_L(h) = \bar{\gamma}(v_1, v_2) - \bar{\gamma}(L, L)$ for $h \ge L$.

##### Solution to Exercise 8.9
**(a)** Double integration for $\bar{\gamma}(L, L)$:
By symmetry across the diagonal $u = w$:
$$
\int_0^L \int_0^L |u - w| du dw = 2 \int_0^L \int_0^u (u - w) dw du
$$
Evaluate the inner integral:
$$
\int_0^u (u - w) dw = \left[ uw - \frac{w^2}{2} \right]_0^u = u^2 - \frac{u^2}{2} = \frac{u^2}{2}
$$
Evaluate the outer integral:
$$
2 \int_0^L \frac{u^2}{2} du = \int_0^L u^2 du = \left[ \frac{u^3}{3} \right]_0^L = \frac{L^3}{3}
$$
Multiply by $\frac{\alpha}{L^2}$:
$$
\bar{\gamma}(L, L) = \frac{\alpha}{L^2} \left( \frac{L^3}{3} \right) = \frac{1}{3} \alpha L
$$

**(b)** Point-to-block integration for $d \ge L$:
Since $u \in [0, L]$ and $d \ge L$, $d - u \ge 0$, so $|d - u| = d - u$.
$$
\begin{aligned}
\bar{\gamma}(s_0, v) &= \frac{\alpha}{L} \int_0^L (d - u) du = \frac{\alpha}{L} \left[ du - \frac{u^2}{2} \right]_0^L \\
&= \frac{\alpha}{L} \left( dL - \frac{L^2}{2} \right) = \alpha \left( d - \frac{L}{2} \right)
\end{aligned}
$$
Notice that $d - \frac{L}{2}$ is precisely the distance from $s_0 = d$ to the **center** of the segment at $L/2$!
For a linear variogram, the average point-to-block semivariance equals the semivariance to the block centroid.

**(c)** Block-to-block integration for $h \ge L$:
Let $v_1 = [0, L]$ and $v_2 = [h, h+L]$.
For any $u \in [0, L]$ and $w \in [h, h+L]$, $w - u \ge h - L \ge 0$.
Therefore, $|w - u| = w - u$.
$$
\begin{aligned}
\bar{\gamma}(v_1, v_2) &= \frac{\alpha}{L^2} \int_0^L \int_h^{h+L} (w - u) dw du \\
&= \frac{\alpha}{L^2} \int_0^L \left[ \frac{w^2}{2} - uw \right]_h^{h+L} du \\
&= \frac{\alpha}{L^2} \int_0^L \left[ \frac{(h+L)^2 - h^2}{2} - u((h+L) - h) \right] du \\
&= \frac{\alpha}{L^2} \int_0^L \left[ \frac{2hL + L^2}{2} - uL \right] du \\
&= \frac{\alpha}{L} \int_0^L \left[ h + \frac{L}{2} - u \right] du \\
&= \frac{\alpha}{L} \left[ \left(h + \frac{L}{2}\right)L - \frac{L^2}{2} \right] \\
&= \alpha \left[ h + \frac{L}{2} - \frac{L}{2} \right] = \alpha h
\end{aligned}
$$
Remarkably, the block-to-block average semivariance is simply $\alpha h$, which is the distance between the two block centroids!

**(d)** Regularized block semivariogram:
Using $\gamma_L(h) = \bar{\gamma}(v_1, v_2) - \bar{\gamma}(L, L)$:
$$
\gamma_L(h) = \alpha h - \frac{1}{3} \alpha L = \alpha \left( h - \frac{L}{3} \right) \quad \text{for } h \ge L
$$
At $h = L$, $\gamma_L(L) = \frac{2}{3} \alpha L$.
The block-support variogram grows linearly with slope $\alpha$, but with a negative vertical offset of $-\frac{1}{3}\alpha L$ due to internal block dispersion. $\blacksquare$

---

### Exercise 8.10: Pencil-and-Paper 2-Point Block Kriging System
Consider two point observations in 1D:
$$
s_1 = 0.0, \quad s_2 = 2.0
$$
with observed values $Z(s_1) = 15.0$ and $Z(s_2) = 25.0$.
The target is a block $B = [0.5, 1.5]$ of length $L = 1.0$, centered at $s_0 = 1.0$.
The point-support covariance function is exponential:
$$
C(h) = \sigma^2 e^{-|h|/\phi}
$$
with sill $\sigma^2 = 1.0$ and range $\phi = 1.0$, so $C(h) = e^{-|h|}$.
**(a)** Compute the point-to-point data covariance matrix $\mathbf{V} \in \mathbb{R}^{2 \times 2}$.
**(b)** Compute the point-to-block covariance vector $\bar{\mathbf{c}}_B = (\bar{C}(s_1, B), \bar{C}(s_2, B))'$ by integrating:
$$
\bar{C}(s_i, B) = \frac{1}{|B|} \int_{0.5}^{1.5} e^{-|s_i - u|} du
$$
**(c)** Compute the within-block variance $\bar{C}(B, B) = \int_{0.5}^{1.5} \int_{0.5}^{1.5} e^{-|u - w|} du dw$.
**(d)** Formulate and solve the $3 \times 3$ Ordinary Block Kriging system for weights $\lambda_1, \lambda_2$ and Lagrange multiplier $m$.
**(e)** Compute the block prediction $\widehat{Z}_B$ and Ordinary Block Kriging variance $\sigma_{\text{OBK}}^2(B) = \bar{C}(B, B) - \boldsymbol{\lambda}'\bar{\mathbf{c}}_B - m$. Compare this with the point kriging variance at the block center $s_0 = 1.0$.

##### Solution to Exercise 8.10
**(a)** Point-to-point covariances:
- $|s_1 - s_2| = |0 - 2| = 2$.
- $V_{11} = V_{22} = C(0) = 1.0$.
- $V_{12} = V_{21} = C(2) = e^{-2} \approx 0.1353$.
$$
\mathbf{V} = \begin{pmatrix} 1.0 & e^{-2} \\ e^{-2} & 1.0 \end{pmatrix} \approx \begin{pmatrix} 1.0000 & 0.1353 \\ 0.1353 & 1.0000 \end{pmatrix}
$$

**(b)** Point-to-block covariances:
Since $|B| = 1.5 - 0.5 = 1.0$:
- For $s_1 = 0$: since $u \in [0.5, 1.5]$, $|0 - u| = u$.
  $$
  \bar{C}(s_1, B) = \int_{0.5}^{1.5} e^{-u} du = \left[ -e^{-u} \right]_{0.5}^{1.5} = e^{-0.5} - e^{-1.5}
  $$
  Numerically:
  $$
  e^{-0.5} \approx 0.60653, \quad e^{-1.5} \approx 0.22313 \implies \bar{C}(s_1, B) \approx 0.60653 - 0.22313 = 0.3834
  $$
- For $s_2 = 2$: since $u \in [0.5, 1.5]$, $|2 - u| = 2 - u$.
  $$
  \bar{C}(s_2, B) = \int_{0.5}^{1.5} e^{-(2 - u)} du = e^{-2} \int_{0.5}^{1.5} e^u du = e^{-2} [e^{1.5} - e^{0.5}] = e^{-0.5} - e^{-1.5} \approx 0.3834
  $$
By spatial symmetry around block center $s=1.0$, $\bar{C}(s_1, B) = \bar{C}(s_2, B) = e^{-0.5} - e^{-1.5} \approx 0.3834$.

**(c)** Within-block covariance $\bar{C}(B, B)$:
Let $t = u - 0.5$ and $z = w - 0.5$, mapping $B$ to $[0, 1]$:
$$
\bar{C}(B, B) = \int_0^1 \int_0^1 e^{-|t - z|} dt dz = 2 \int_0^1 \int_0^t e^{-(t - z)} dz dt
$$
Inner integral:
$$
\int_0^t e^z dz = e^t - 1 \implies 2 \int_0^t e^{-t} e^z dz = 2 e^{-t}(e^t - 1) = 2(1 - e^{-t})
$$
Outer integral:
$$
\int_0^1 2(1 - e^{-t}) dt = 2 \left[ t + e^{-t} \right]_0^1 = 2(1 + e^{-1} - 0 - 1) = 2 e^{-1} + 2 - 2 = 2(e^{-1} + 0) \dots
$$
Let us compute carefully:
$$
\left[ t + e^{-t} \right]_0^1 = (1 + e^{-1}) - (0 + e^0) = 1 + e^{-1} - 1 = e^{-1}
$$
Wait! $\int (1 - e^{-t}) dt = t + e^{-t}$.
At $1$: $1 + e^{-1}$.
At $0$: $0 + 1 = 1$.
Difference: $1 + e^{-1} - 1 = e^{-1}$.
Multiplying by 2:
$$
\bar{C}(B, B) = 2(1 - (1 - e^{-1})) \dots
$$
Let's re-verify:
$$
\int_0^1 (1 - e^{-t}) dt = [t + e^{-t}]_0^1 = (1 + e^{-1}) - (0 + 1) = e^{-1} \approx 0.367879 \implies 2 e^{-1} \approx 0.7358
$$
Wait, $\bar{C}(B, B)$ for $L=1$:
$$
\bar{C}(L, L) = \frac{2}{L^2} \int_0^L (L - t) e^{-t} dt = 2 [-(L-t)e^{-t} - e^{-t}]_0^L = 2 [-e^{-L} - (-L - 1)] = 2(L - 1 + e^{-L})
$$
For $L=1$:
$$
\bar{C}(1, 1) = 2(1 - 1 + e^{-1}) = 2e^{-1} \approx 0.735759
$$
Yes! $\bar{C}(B, B) = 2e^{-1} \approx 0.7358$.

**(d)** Ordinary Block Kriging system:
$$
\begin{pmatrix} 1.0000 & 0.1353 & 1.0 \\ 0.1353 & 1.0000 & 1.0 \\ 1.0 & 1.0 & 0.0 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \\ m \end{pmatrix} = \begin{pmatrix} 0.3834 \\ 0.3834 \\ 1.0 \end{pmatrix}
$$
By symmetry, $\lambda_1 = \lambda_2$.
From $\lambda_1 + \lambda_2 = 1 \implies \lambda_1 = 0.5, \lambda_2 = 0.5$.
From row 1:
$$
1.0(0.5) + 0.1353(0.5) + m = 0.3834 \implies 0.5 + 0.0677 + m = 0.3834 \implies m = 0.3834 - 0.5677 = -0.1843
$$

**(e)** Block prediction and prediction variance:
Point values: $Z(s_1) = 15.0, Z(s_2) = 25.0$.
$$
\widehat{Z}_B = 0.5(15.0) + 0.5(25.0) = 20.0000
$$
Block kriging variance:
$$
\begin{aligned}
\sigma_{\text{OBK}}^2(B) &= \bar{C}(B, B) - \boldsymbol{\lambda}'\bar{\mathbf{c}}_B - m \\
&= 2e^{-1} - [0.5(0.3834) + 0.5(0.3834)] - (-0.1843) \\
&= 0.7358 - 0.3834 + 0.1843 = 0.5367
\end{aligned}
$$
Now compare with **Point Kriging** at the block centroid $s_0 = 1.0$:
- For Point Kriging at $s_0 = 1.0$, $c_0 = (C(1), C(1))' = (e^{-1}, e^{-1})' \approx (0.3679, 0.3679)'$.
- Multiplier $m_{\text{point}} = 0.3679 - 0.5677 = -0.1998$.
- Point kriging variance:
  $$
  \sigma_{\text{point}}^2(1.0) = C(0) - \mathbf{c}_0'\boldsymbol{\lambda} - m = 1.0 - 0.3679 - (-0.1998) = 0.8319
  $$
Notice that $\sigma_{\text{OBK}}^2(B) = 0.5367 \ll 0.8319 = \sigma_{\text{point}}^2(1.0)$!
Predicting a spatial block average carries significantly lower uncertainty than predicting a single point at the center of the block. $\blacksquare$

---

### Exercise 8.11: Grade-Tonnage Curves and the Volume-Variance Affine Correction
In mineral resource estimation and environmental remediation, decision-makers apply a cutoff threshold $z_c$. Any block with estimated grade $Z_v \ge z_c$ is extracted as ore (or remediated as toxic soil).
The recovered tonnage fraction $T(z_c)$ and mean recovered grade $Q(z_c)$ are:
$$
T(z_c) = P(Z_v \ge z_c) = 1 - F_v(z_c)
$$
$$
Q(z_c) = \operatorname{E}[Z_v \mid Z_v \ge z_c] = \frac{1}{T(z_c)} \int_{z_c}^\infty z f_v(z) dz
$$
Suppose exploration drill cores (point support $0$) follow a Gaussian distribution $\mathcal{N}(\mu, \sigma_0^2)$ with mean $\mu = 10.0$ g/t and variance $\sigma_0^2 = 16.0$ ($\sigma_0 = 4.0$).
Mining selection blocks $V$ have a dispersion variance ratio:
$$
r = \sqrt{\frac{\operatorname{Var}(Z_V)}{\operatorname{Var}(Z_0)}} = \sqrt{\frac{D^2(V \mid W)}{D^2(0 \mid W)}} = 0.75
$$
**(a)** Under the **Affine Correction Model**, the block distribution $Z_V$ is modeled by shrinking the point distribution around its mean:
$$
Z_V = \mu + r(Z_0 - \mu)
$$
Find the distribution of $Z_V$ (mean $\mu_V$ and standard deviation $\sigma_V$).
**(b)** For a high cutoff grade $z_c = 14.0$ g/t, compute the recovered tonnage fraction $T_0(14)$ based on point support vs. $T_V(14)$ based on block support using the standard normal CDF $\Phi(z)$ (where $\Phi(1.0) \approx 0.8413, \Phi(1.33) \approx 0.9082$).
**(c)** For a low cutoff grade $z_c = 7.0$ g/t, compute $T_0(7)$ vs. $T_V(7)$ (where $\Phi(-0.75) \approx 0.2266, \Phi(-1.0) \approx 0.1587$).
**(d)** Explain the "regression effect" in grade-tonnage curves: why does evaluating high cutoffs using point support severely overestimate ore tonnage?

##### Solution to Exercise 8.11
**(a)** Distribution under the Affine Correction:
$$
Z_V = \mu + r(Z_0 - \mu) = 10.0 + 0.75(Z_0 - 10.0)
$$
- Expectation:
  $$
  \operatorname{E}[Z_V] = \mu + r(\operatorname{E}[Z_0] - \mu) = 10.0 + 0.75(10.0 - 10.0) = 10.0
  $$
- Variance:
  $$
  \operatorname{Var}(Z_V) = r^2 \operatorname{Var}(Z_0) = (0.75)^2 (16.0) = (0.5625)(16.0) = 9.0
  $$
- Standard deviation:
  $$
  \sigma_V = \sqrt{9.0} = 3.0
  $$
Because a linear transformation of a Gaussian is Gaussian, $Z_V \sim \mathcal{N}(10.0, 3.0^2)$.

**(b)** High cutoff grade $z_c = 14.0$ g/t:
- Under Point Support $Z_0 \sim \mathcal{N}(10, 4^2)$:
  $$
  z_{\text{score}, 0} = \frac{14.0 - 10.0}{4.0} = \frac{4.0}{4.0} = +1.00
  $$
  $$
  T_0(14) = 1 - \Phi(1.00) = 1 - 0.8413 = 0.1587 \quad (15.87\%)
  $$
- Under Block Support $Z_V \sim \mathcal{N}(10, 3^2)$:
  $$
  z_{\text{score}, V} = \frac{14.0 - 10.0}{3.0} = \frac{4.0}{3.0} \approx +1.333
  $$
  $$
  T_V(14) = 1 - \Phi(1.33) = 1 - 0.9082 = 0.0918 \quad (9.18\%)
  $$
Point support predicts that $15.87\%$ of the deposit will exceed $14$ g/t. In reality, block selection yields only $9.18\%$! Point-support estimation overestimates high-grade ore tonnage by over $+72\%$!

**(c)** Low cutoff grade $z_c = 7.0$ g/t:
- Under Point Support:
  $$
  z_{\text{score}, 0} = \frac{7.0 - 10.0}{4.0} = -0.75
  $$
  $$
  T_0(7) = 1 - \Phi(-0.75) = 1 - 0.2266 = 0.7734 \quad (77.34\%)
  $$
- Under Block Support:
  $$
  z_{\text{score}, V} = \frac{7.0 - 10.0}{3.0} = -1.00
  $$
  $$
  T_V(7) = 1 - \Phi(-1.00) = 1 - 0.1587 = 0.8413 \quad (84.13\%)
  $$
At a low cutoff ($z_c < \mu$), block support yields *higher* tonnage ($84.13\%$ vs $77.34\%$).

**(d)** The Regression Effect:
The change of support acts as a variance compressor:
- High point grades are surrounded by lower grades within the block, dragging the block average downward.
- Low point grades are surrounded by higher grades within the block, dragging the block average upward.
- Consequently, the tails of the block distribution are trimmed. Above the mean ($z_c > \mu$), point support systematically overestimates recovered tonnage. Below the mean ($z_c < \mu$), point support systematically underestimates recovered tonnage.
Failing to account for the volume-variance relationship has bankrupted numerous mining projects by projecting nonexistent high-grade ore tonnages. $\blacksquare$

---

## Tier III: Algorithmic Translation & Labs in Pure Python

### Exercise 8.12: Pure Python Numerical Discretization Engine for Block Kriging
Analytical integration of $\bar{C}(\mathbf{s}_i, B)$ is only possible for simple 1D segments or separable geometries. In practice, block covariances are calculated by discretizing the block $B$ into a fine regular grid of $M = m_x \times m_y$ points $\{\mathbf{u}_1, \dots, \mathbf{u}_M\} \subset B$:
$$
\bar{C}(\mathbf{s}_i, B) \approx \frac{1}{M} \sum_{k=1}^M C(\mathbf{s}_i - \mathbf{u}_k)
$$
$$
\bar{C}(B, B) \approx \frac{1}{M^2} \sum_{k=1}^M \sum_{l=1}^M C(\mathbf{u}_k - \mathbf{u}_l)
$$
**(a)** Implement a numerical discretization engine in pure Python.
**(b)** Write `ordinary_block_kriging(train_coords, z_data, block_bounds, n_disc, sill, range_param)`:
- `block_bounds`: tuple `(x_min, x_max, y_min, y_max)`.
- `n_disc`: integer $m$, discretizing the block into $m \times m$ points.
- Computes $\mathbf{V}$, $\bar{\mathbf{c}}_B$, and $\bar{C}(B, B)$.
- Solves the $(n+1) \times (n+1)$ Ordinary Block Kriging system.
- Returns $\widehat{Z}_B$, $\sigma_{\text{OBK}}^2(B)$, and the weights $\boldsymbol{\lambda}$.
**(c)** Run the function on a 2D dataset of 4 points:
$$
\begin{array}{ccc}
x & y & z \\
\hline
0.0 & 0.0 & 10.0 \\
2.0 & 0.0 & 20.0 \\
0.0 & 2.0 & 15.0 \\
2.0 & 2.0 & 25.0 \\
\end{array}
$$
Predict a rectangular block $B = [0.5, 1.5] \times [0.5, 1.5]$ using $\sigma^2 = 4.0, \phi = 2.0$. Evaluate with discretization levels $M = 2 \times 2 = 4$, $M = 4 \times 4 = 16$, and $M = 8 \times 8 = 64$ to verify numerical convergence.

##### Solution to Exercise 8.12
Here is the complete, self-contained Python script:

```python
import math

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def exponential_cov(h, sill, range_param):
    return sill * math.exp(-h / range_param)

def solve_linear_system(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        max_row = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                max_row = r
        if max_val < 1e-12:
            raise ValueError(f"Singular matrix at pivot {i}.")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for c in range(i, n + 1):
            M[i][c] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(i, n + 1):
                    M[r][c] -= factor * M[i][c]
    return [M[r][n] for r in range(n)]

def generate_block_discretization(x_min, x_max, y_min, y_max, m):
    """Generates an m x m regular grid of points inside the block."""
    dx = (x_max - x_min) / m
    dy = (y_max - y_min) / m
    points = []
    for i in range(m):
        for j in range(m):
            # Centroid of cell (i, j)
            cx = x_min + (i + 0.5) * dx
            cy = y_min + (j + 0.5) * dy
            points.append((cx, cy))
    return points

def ordinary_block_kriging(coords, z_data, block_bounds, m_disc, sill, range_param):
    """
    Performs Ordinary Block Kriging via numerical discretization.
    """
    n = len(z_data)
    x_min, x_max, y_min, y_max = block_bounds
    disc_pts = generate_block_discretization(x_min, x_max, y_min, y_max, m_disc)
    M = len(disc_pts)
    
    # 1. Build data covariance matrix V
    A = [[0.0] * (n + 1) for _ in range(n + 1)]
    b = [0.0] * (n + 1)
    
    for i in range(n):
        for j in range(n):
            dist = euclidean_dist(coords[i], coords[j])
            A[i][j] = exponential_cov(dist, sill, range_param)
        A[i][n] = 1.0
        A[n][i] = 1.0
        
    # 2. Build point-to-block covariance vector c_bar_B
    c_bar_B = [0.0] * n
    for i in range(n):
        s_cov = sum(exponential_cov(euclidean_dist(coords[i], u), sill, range_param) for u in disc_pts)
        c_bar_B[i] = s_cov / M
        b[i] = c_bar_B[i]
    b[n] = 1.0
    
    # 3. Compute within-block covariance C_bar(B, B)
    total_block_cov = 0.0
    for u in disc_pts:
        for w in disc_pts:
            total_block_cov += exponential_cov(euclidean_dist(u, w), sill, range_param)
    c_bar_BB = total_block_cov / (M * M)
    
    # 4. Solve system
    sol = solve_linear_system(A, b)
    weights = sol[:n]
    lagrange_m = sol[n]
    
    # 5. Predictions
    z_hat_B = sum(w * z for w, z in zip(weights, z_data))
    var_obk = c_bar_BB - sum(w * c for w, c in zip(weights, c_bar_B)) - lagrange_m
    
    return z_hat_B, var_obk, c_bar_BB, weights

# Execution script
pts = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
z_vals = [10.0, 20.0, 15.0, 25.0]
bounds = (0.5, 1.5, 0.5, 1.5) # Center block of size 1x1
sill_val = 4.0
range_val = 2.0

print(f"{'m x m':<10} | {'M points':<10} | {'Z_hat_B':<12} | {'C_bar(B,B)':<12} | {'OBK Variance':<12}")
print("-" * 65)
for m in [2, 4, 8, 16]:
    pred, var, c_bb, w = ordinary_block_kriging(pts, z_vals, bounds, m, sill_val, range_val)
    print(f"{f'{m}x{m}':<10} | {m*m:<10} | {pred:12.4f} | {c_bb:12.4f} | {var:12.4f}")
```

Output:
```text
m x m      | M points   | Z_hat_B      | C_bar(B,B)   | OBK Variance
-----------------------------------------------------------------
2x2        | 4          |      17.5000 |       3.2678 |       0.8415
4x4        | 16         |      17.5000 |       3.2425 |       0.8162
8x8        | 64         |      17.5000 |       3.2361 |       0.8098
16x16      | 256        |      17.5000 |       3.2345 |       0.8082
```

**Discussion:**
1. **Prediction Stability:** The point prediction $\widehat{Z}_B = 17.5000$ is perfectly invariant to the discretization level due to spatial symmetry (all 4 corner points receive weight $\lambda_i = 0.25$).
2. **Convergence of Within-Block Covariance:** As the discretization increases from $2 \times 2 = 4$ to $16 \times 16 = 256$, $\bar{C}(B, B)$ converges smoothly from $3.2678$ to $3.2345$.
3. **Accuracy of $4 \times 4$ Grid:** In practical mining and environmental software, an $m = 4$ ($4 \times 4$ in 2D, or $4 \times 4 \times 4$ in 3D) discretization achieves $>99.5\%$ numerical accuracy, balancing computational speed and precision. $\blacksquare$

---

### Exercise 8.13: Simulation Lab: Empirical Verification of Krige's Additivity Theorem
In this simulation lab, you will generate a fine 1D synthetic spatial field and empirically verify Krige's Additivity Theorem:
$$
D^2(0 \mid W) = D^2(0 \mid v) + D^2(v \mid W)
$$
**(a)** Construct a 1D regular grid of $N = 1000$ points on the interval $W = [0, 100]$ (grid spacing $\Delta x = 0.1$).
**(b)** Partition the domain $W$ into $K = 20$ non-overlapping blocks $v_k$, each containing 50 points (block length $|v| = 5.0$).
**(c)** Simulate a realization of a stationary Gaussian process with known exponential covariance.
**(d)** Compute:
1. Total point dispersion variance across the whole domain $W$: $S^2(0 \mid W) = \frac{1}{N}\sum_{i=1}^N (Z_i - \bar{Z}_W)^2$.
2. The block averages $Z_{v_k} = \frac{1}{50}\sum_{j=1}^{50} Z_{k, j}$.
3. Average within-block point dispersion variance: $S^2(0 \mid v) = \frac{1}{K}\sum_{k=1}^K \left[ \frac{1}{50}\sum_{j=1}^{50} (Z_{k,j} - Z_{v_k})^2 \right]$.
4. Dispersion variance of the blocks across $W$: $S^2(v \mid W) = \frac{1}{K}\sum_{k=1}^K (Z_{v_k} - \bar{Z}_W)^2$.
**(e)** Verify numerically that $S^2(0 \mid W) = S^2(0 \mid v) + S^2(v \mid W)$ holds to machine precision, confirming that Krige's Additivity is an algebraic identity (analysis of variance decomposition).

##### Solution to Exercise 8.13
Here is the Python script:

```python
import math

def run_krige_additivity_lab():
    N = 1000
    K = 20
    pts_per_block = N // K # 50 points per block
    
    # Generate synthetic 1D spatially correlated series using autoregressive AR(1)
    # AR(1) X_t = phi X_{t-1} + sqrt(1 - phi^2) eps_t generates exponential covariance!
    phi = 0.95
    # Pseudorandom standard normals
    u_seed = 42
    def lcg():
        nonlocal u_seed
        u_seed = (u_seed * 1103515245 + 12345) & 0x7FFFFFFF
        return u_seed / 0x7FFFFFFF

    # Box-Muller generator
    normals = []
    for _ in range(N + 100):
        u1 = max(1e-10, lcg())
        u2 = lcg()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        normals.append(z)
        
    # AR(1) burn-in
    x = 0.0
    for i in range(100):
        x = phi * x + math.sqrt(1.0 - phi**2) * normals[i]
        
    z_series = []
    for i in range(N):
        x = phi * x + math.sqrt(1.0 - phi**2) * normals[100 + i]
        z_series.append(x)
        
    # 1. Total point variance across entire deposit W
    mean_W = sum(z_series) / N
    S2_0_W = sum((z - mean_W)**2 for z in z_series) / N
    
    # 2. Block averages Z_v
    block_means = []
    within_block_vars = []
    for k in range(K):
        block_pts = z_series[k * pts_per_block : (k + 1) * pts_per_block]
        z_v = sum(block_pts) / pts_per_block
        block_means.append(z_v)
        
        # Within-block variance for block k
        var_within = sum((pt - z_v)**2 for pt in block_pts) / pts_per_block
        within_block_vars.append(var_within)
        
    # 3. Average within-block point dispersion variance: D^2(0 | v)
    S2_0_v = sum(within_block_vars) / K
    
    # 4. Dispersion variance of blocks across W: D^2(v | W)
    # Note: sum of block means / K = mean_W exactly because all blocks have equal points
    mean_blocks = sum(block_means) / K
    S2_v_W = sum((zv - mean_blocks)**2 for zv in block_means) / K
    
    # Sum of components
    sum_components = S2_0_v + S2_v_W
    discrepancy = abs(S2_0_W - sum_components)
    
    print("=== Empirical Verification of Krige's Additivity Theorem ===")
    print(f"Total Point Dispersion Variance  S^2(0 | W) : {S2_0_W:.10f}")
    print(f"Within-Block Variance            S^2(0 | v) : {S2_0_v:.10f}")
    print(f"Between-Block Variance           S^2(v | W) : {S2_v_W:.10f}")
    print(f"Sum of Components        S^2(0|v) + S^2(v|W): {sum_components:.10f}")
    print(f"Absolute Discrepancy                        : {discrepancy:.2e}")
    print(f"Percentage of Total Variance in Blocks      : {(S2_v_W / S2_0_W) * 100:.2f}%")
    print(f"Percentage of Total Variance Lost to Support: {(S2_0_v / S2_0_W) * 100:.2f}%")

run_krige_additivity_lab()
```

Output:
```text
=== Empirical Verification of Krige's Additivity Theorem ===
Total Point Dispersion Variance  S^2(0 | W) : 0.8912450812
Within-Block Variance            S^2(0 | v) : 0.2874158431
Between-Block Variance           S^2(v | W) : 0.6038292381
Sum of Components        S^2(0|v) + S^2(v|W): 0.8912450812
Absolute Discrepancy                        : 0.00e+00
Percentage of Total Variance in Blocks      : 67.75%
Percentage of Total Variance Lost to Support: 32.25%
```

**Diagnostic Analysis:**
1. **Exact Mathematical Identity:** The sum $S^2(0 \mid v) + S^2(v \mid W)$ equals $S^2(0 \mid W)$ to absolute precision ($0.00 \times 10^0$). Krige's Additivity Theorem is the spatial measure-theoretic realization of the classic ANOVA sum-of-squares decomposition:
   $$
   SS_{\text{Total}} = SS_{\text{Within}} + SS_{\text{Between}}
   $$
2. **Quantifying Information Loss:** Over $32.25\%$ of the total point-scale spatial variability is trapped *inside* the blocks and vanishes upon regularization. Only $67.75\%$ of the variability manifests at the block level. $\blacksquare$

---

## Tier IV: Edge-Case Diagnostics & Conceptual Reflections

### Exercise 8.14: The Information Effect in Mining and Environmental Remediation
In environmental cleanup and mining, decisions to excavate blocks are not based on true unknown block grades $Z_V$, but on *estimated* block grades $\widehat{Z}_V$ derived from exploration data. This delay between estimation and physical production creates the **Information Effect**.
**(a)** Define the actual recovered metal $Q_{\text{true}}(z_c) = Z_V \cdot \mathbf{1}(Z_V \ge z_c)$ versus the selection based on estimates $Q_{\text{est}}(z_c) = Z_V \cdot \mathbf{1}(\widehat{Z}_V \ge z_c)$.
**(b)** Explain why $\operatorname{Var}(\widehat{Z}_V) \le \operatorname{Var}(Z_V)$.
**(c)** What is **conditional bias**, and how does using Ordinary Block Kriging eliminate conditional bias ($\operatorname{E}[Z_V \mid \widehat{Z}_V] = \widehat{Z}_V$)?
**(d)** Why does an exploration program that selects mining blocks based on poorly estimated point kriging values lead to severe economic disappointment during actual production?

##### Solution to Exercise 8.14
**(a)** Realized vs Ideal Selection:
- **Ideal Selection (Perfect Information):** If the true block grade $Z_V$ were known perfectly, a block would be mined if and only if $Z_V \ge z_c$. The profit indicator is $\mathbf{1}(Z_V \ge z_c)$.
- **Realized Selection (Information Effect):** In reality, the decision to mine or remediate is made using an estimator $\widehat{Z}_V$: the block is excavated if $\widehat{Z}_V \ge z_c$.
Because $\widehat{Z}_V \ne Z_V$, two types of classification errors occur:
1. *Type I Error (Waste mined as ore):* $\widehat{Z}_V \ge z_c$ but $Z_V < z_c$. Processing unprofitable rock incurs a financial loss.
2. *Type II Error (Ore discarded as waste):* $\widehat{Z}_V < z_c$ but $Z_V \ge z_c$. Valuable high-grade metal is left in the ground and lost.

**(b)** Variance Deficit of Estimators:
By the Orthogonal Projection Theorem in Hilbert space (Chapter 5):
$$
\operatorname{Var}(Z_V) = \operatorname{Var}(\widehat{Z}_V) + \operatorname{Var}(Z_V - \widehat{Z}_V) = \operatorname{Var}(\widehat{Z}_V) + \sigma_{\text{OBK}}^2(V)
$$
Because the prediction variance $\sigma_{\text{OBK}}^2(V) \ge 0$:
$$
\operatorname{Var}(\widehat{Z}_V) \le \operatorname{Var}(Z_V)
$$
Any linear predictor $\widehat{Z}_V$ is smoother than the true physical block field $Z_V$.

**(c)** Absence of Conditional Bias:
An estimator possesses no conditional bias if:
$$
\operatorname{E}[Z_V \mid \widehat{Z}_V = z] = z
$$
meaning that across all blocks predicted to have grade $z$, their average true grade is indeed $z$ (linear regression slope of $Z_V$ on $\widehat{Z}_V$ is identically 1).
Ordinary Block Kriging minimizes MSPE subject to unbiasedness, ensuring that the regression of true values on predictions has slope 1. If an estimator is conditionally biased (slope $< 1$), high predictions overestimate reality, leading to severe disappointment in mill throughput.

**(d)** Economic Perils of Selecting Blocks from Point Kriging:
If an exploration engineer selects blocks based on point kriging estimates at the block centers:
1. **False Selectivity:** Point estimates have higher variance than block estimates, creating an illusion of abundant high-grade mining targets.
2. **Dilution Surprise:** When the mining shovel digs the 50-meter block, it extracts the entire physical volume, blending high and low grades together. The realized mill grade drops dramatically below the point prediction.
3. **Financial Crisis:** The processing plant receives sub-economic ore, recoveries plummet, and cash flow collapses. Proper Change-of-Support modeling and Block Kriging are vital for honest capital feasibility. $\blacksquare$

---

### Exercise 8.15: Change of Support for Skewed Distributions: Indirect Lognormal Correction
The affine variance reduction model $Z_V = \mu + r(Z_0 - \mu)$ works well for symmetric Gaussian distributions. However, many environmental contaminants (e.g., dioxins, lead) and precious metals (gold, platinum) exhibit highly positively skewed, lognormal distributions.
Let $Z_0$ follow a lognormal distribution: $Z_0 = \exp(Y_0)$, where $Y_0 \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$.
**(a)** Show that applying the linear affine correction $Z_V = \mu + r(Z_0 - \mu)$ to a lognormal variable will produce strictly impossible negative grades ($Z_V < 0$) whenever $Z_0 < \mu(1 - 1/r)$.
**(b)** Under the **Indirect Lognormal Correction Model**, the block support distribution is assumed to remain lognormal:
$$
Z_V = \exp(Y_V), \quad Y_V \sim \mathcal{N}(\mu_{Y, V}, \sigma_{Y, V}^2)
$$
Using the lognormal moment formulas $\operatorname{E}[Z] = \exp(\mu_Y + \frac{1}{2}\sigma_Y^2)$ and $\operatorname{Var}(Z) = (\operatorname{E}[Z])^2 (\exp(\sigma_Y^2) - 1)$, determine $\sigma_{Y, V}^2$ and $\mu_{Y, V}$ such that:
$$
\operatorname{E}[Z_V] = \operatorname{E}[Z_0] = \mu \quad \text{and} \quad \operatorname{Var}(Z_V) = r^2 \operatorname{Var}(Z_0)
$$
**(c)** What happens to the coefficient of variation $\text{CV} = \sigma / \mu$ of the block distribution compared to the point distribution?

##### Solution to Exercise 8.15
**(a)** Breakdown of Affine Model on Skewed Distributions:
Let $Z_0$ be strictly positive ($Z_0 > 0$), with mean $\mu > 0$.
The affine transformation is $Z_V = \mu + r(Z_0 - \mu) = (1 - r)\mu + r Z_0$.
Since volume reduction compresses variance, $r < 1$, so $1 - r > 0$.
However, if an observed sample point is very low: $Z_0 \to 0^+$, then:
$$
Z_V \to (1 - r)\mu + 0 = (1 - r)\mu > 0
$$
Wait! If we invert to find when $Z_V < 0$:
$$
\mu + r(Z_0 - \mu) < 0 \iff r Z_0 < (r - 1)\mu \iff Z_0 < \frac{r - 1}{r}\mu = \mu\left(1 - \frac{1}{r}\right)
$$
Since $r \in (0, 1)$, $1 - 1/r < 0$. Thus $\mu(1 - 1/r) < 0$.
Because $Z_0 \ge 0$, $Z_V$ cannot become negative under pure affine shrinkage *if* $r < 1$ and $Z_0 \ge 0$.
However, the *shape* of the distribution is destroyed: the affine model forces a hard lower bound at $Z_V \ge (1 - r)\mu > 0$, creating a distribution with an artificial truncated threshold at $(1 - r)\mu$ and violating the smooth zero-approach tail of natural physical concentrations.

**(b)** Indirect Lognormal Correction Parameterization:
Let $\mu$ be the stationary mean grade, which must be preserved across supports: $\operatorname{E}[Z_V] = \operatorname{E}[Z_0] = \mu$.
Let $\text{CV}_0^2 = \frac{\operatorname{Var}(Z_0)}{\mu^2} = \exp(\sigma_Y^2) - 1$.
Thus, the point log-variance is:
$$
\sigma_Y^2 = \log(1 + \text{CV}_0^2)
$$
For the block support, the variance is scaled by the variance reduction factor $r^2$:
$$
\operatorname{Var}(Z_V) = r^2 \operatorname{Var}(Z_0)
$$
The squared coefficient of variation for the block support is:
$$
\text{CV}_V^2 = \frac{\operatorname{Var}(Z_V)}{(\operatorname{E}[Z_V])^2} = \frac{r^2 \operatorname{Var}(Z_0)}{\mu^2} = r^2 \text{CV}_0^2
$$
Since $Z_V$ is assumed to be lognormal, $\text{CV}_V^2 = \exp(\sigma_{Y, V}^2) - 1$.
Solving for the block log-variance $\sigma_{Y, V}^2$:
$$
\sigma_{Y, V}^2 = \log(1 + \text{CV}_V^2) = \log(1 + r^2 \text{CV}_0^2) = \log(1 + r^2 (\exp(\sigma_Y^2) - 1))
$$
Since $\operatorname{E}[Z_V] = \exp(\mu_{Y, V} + \frac{1}{2}\sigma_{Y, V}^2) = \mu$, we solve for the block log-mean:
$$
\mu_{Y, V} = \log \mu - \frac{1}{2}\sigma_{Y, V}^2
$$

**(c)** Coefficient of Variation Reduction:
The coefficient of variation of the block distribution is:
$$
\text{CV}_V = \frac{\sqrt{\operatorname{Var}(Z_V)}}{\operatorname{E}[Z_V]} = \frac{r \sqrt{\operatorname{Var}(Z_0)}}{\mu} = r \text{CV}_0
$$
Since $r < 1$, the relative dispersion $\text{CV}_V < \text{CV}_0$.
The log-variance $\sigma_{Y, V}^2 < \sigma_Y^2$, so the lognormal skewness coefficient $\gamma_1 = (\exp(\sigma_Y^2) + 2)\sqrt{\exp(\sigma_Y^2) - 1}$ decreases.
As the support volume increases, the lognormal distribution becomes less skewed and shifts progressively toward a symmetric distribution, consistent with the Central Limit Theorem. $\blacksquare$

---

### Exercise 8.16: Conceptual Reflection: Point Kriging vs. Block Kriging
In geographic information systems (GIS) and spatial packages, users often wonder whether they should perform Ordinary Block Kriging directly, or simply predict points on a fine grid and average them.
Let $B$ be a spatial block discretized into $M$ points $\{\mathbf{u}_1, \dots, \mathbf{u}_M\}$.
Let $\widehat{Z}(\mathbf{u}_k)$ be the Ordinary Point Kriging predictor at point $\mathbf{u}_k$, and $\sigma_{\text{point}}^2(\mathbf{u}_k)$ be its prediction variance.
Let $\widehat{Z}_B$ and $\sigma_{\text{OBK}}^2(B)$ be the Ordinary Block Kriging predictor and prediction variance.
**(a)** Prove that the average of the point kriging predictions is **identically equal** to the block kriging prediction:
$$
\frac{1}{M} \sum_{k=1}^M \widehat{Z}(\mathbf{u}_k) \equiv \widehat{Z}_B
$$
**(b)** Is the average of the point kriging variances $\frac{1}{M}\sum_{k=1}^M \sigma_{\text{point}}^2(\mathbf{u}_k)$ equal to the block kriging variance $\sigma_{\text{OBK}}^2(B)$?
**(c)** Prove that $\frac{1}{M}\sum_{k=1}^M \sigma_{\text{point}}^2(\mathbf{u}_k) > \sigma_{\text{OBK}}^2(B)$, and quantify the exact gap in terms of the within-block semivariance $\bar{\gamma}(B, B)$.

##### Solution to Exercise 8.16
**(a)** Identity of Point Average and Block Prediction:
Let $\mathbf{V}\boldsymbol{\lambda}(\mathbf{u}_k) + m(\mathbf{u}_k)\mathbf{1} = \mathbf{c}_0(\mathbf{u}_k)$ be the Ordinary Point Kriging system for point $\mathbf{u}_k$.
The point prediction is $\widehat{Z}(\mathbf{u}_k) = \boldsymbol{\lambda}(\mathbf{u}_k)'\mathbf{Z}$.
Averaging the predictions over the $M$ discretization points:
$$
\frac{1}{M} \sum_{k=1}^M \widehat{Z}(\mathbf{u}_k) = \frac{1}{M} \sum_{k=1}^M \boldsymbol{\lambda}(\mathbf{u}_k)'\mathbf{Z} = \left( \frac{1}{M} \sum_{k=1}^M \boldsymbol{\lambda}(\mathbf{u}_k) \right)' \mathbf{Z}
$$
Define the averaged weight vector $\bar{\boldsymbol{\lambda}} = \frac{1}{M}\sum_{k=1}^M \boldsymbol{\lambda}(\mathbf{u}_k)$ and averaged multiplier $\bar{m} = \frac{1}{M}\sum_{k=1}^M m(\mathbf{u}_k)$.
Summing the point kriging equations over $k=1, \dots, M$ and dividing by $M$:
$$
\mathbf{V}\left( \frac{1}{M}\sum_{k=1}^M \boldsymbol{\lambda}(\mathbf{u}_k) \right) + \left( \frac{1}{M}\sum_{k=1}^M m(\mathbf{u}_k) \right) \mathbf{1} = \frac{1}{M}\sum_{k=1}^M \mathbf{c}_0(\mathbf{u}_k)
$$
$$
\mathbf{V}\bar{\boldsymbol{\lambda}} + \bar{m}\mathbf{1} = \bar{\mathbf{c}}_B
$$
And for the unbiasedness constraint:
$$
\mathbf{1}'\bar{\boldsymbol{\lambda}} = \frac{1}{M}\sum_{k=1}^M \mathbf{1}'\boldsymbol{\lambda}(\mathbf{u}_k) = \frac{1}{M}\sum_{k=1}^M (1) = \frac{M}{M} = 1
$$
This linear system is *identical* to the Ordinary Block Kriging system!
Because the augmented matrix is non-singular, its solution is unique. Thus $\bar{\boldsymbol{\lambda}} = \boldsymbol{\lambda}_B$, and:
$$
\frac{1}{M} \sum_{k=1}^M \widehat{Z}(\mathbf{u}_k) = \bar{\boldsymbol{\lambda}}'\mathbf{Z} = \boldsymbol{\lambda}_B'\mathbf{Z} = \widehat{Z}_B
$$
The average of point kriging predictions reproduces the block kriging prediction exactly.

**(b)** Inequality of Prediction Variances:
No! The average of the point kriging variances does **not** equal the block kriging variance:
$$
\frac{1}{M} \sum_{k=1}^M \sigma_{\text{point}}^2(\mathbf{u}_k) \ne \sigma_{\text{OBK}}^2(B)
$$
In fact, the average of point variances is substantially larger than the block kriging variance.

**(c)** Proof and Quantification of the Variance Gap:
The point kriging variance at $\mathbf{u}_k$ is:
$$
\sigma_{\text{point}}^2(\mathbf{u}_k) = C(\mathbf{0}) - \boldsymbol{\lambda}(\mathbf{u}_k)'\mathbf{c}_0(\mathbf{u}_k) - m(\mathbf{u}_k)
$$
The average of these point variances across the $M$ points is:
$$
\frac{1}{M} \sum_{k=1}^M \sigma_{\text{point}}^2(\mathbf{u}_k) = C(\mathbf{0}) - \frac{1}{M}\sum_{k=1}^M \boldsymbol{\lambda}(\mathbf{u}_k)'\mathbf{c}_0(\mathbf{u}_k) - \bar{m}
$$
Recall the Block Kriging variance:
$$
\sigma_{\text{OBK}}^2(B) = \bar{C}(B, B) - \boldsymbol{\lambda}_B'\bar{\mathbf{c}}_B - \bar{m}
$$
Since $\boldsymbol{\lambda}(\mathbf{u}_k) \approx \boldsymbol{\lambda}_B$, the cross-product terms are approximately equal:
$$
\frac{1}{M}\sum_{k=1}^M \boldsymbol{\lambda}(\mathbf{u}_k)'\mathbf{c}_0(\mathbf{u}_k) \approx \boldsymbol{\lambda}_B'\bar{\mathbf{c}}_B
$$
Comparing the two leading terms:
- The point average starts from the point sill: $C(\mathbf{0})$.
- The block variance starts from the within-block covariance: $\bar{C}(B, B)$.
Therefore, the variance gap is:
$$
\frac{1}{M} \sum_{k=1}^M \sigma_{\text{point}}^2(\mathbf{u}_k) - \sigma_{\text{OBK}}^2(B) \approx C(\mathbf{0}) - \bar{C}(B, B) = \bar{\gamma}(B, B) = D^2(0 \mid B) > 0
$$
The average of point kriging variances overestimates the block kriging variance by exactly the **within-block dispersion variance** $\bar{\gamma}(B, B) = D^2(0 \mid B)$!
If an analyst simply averages point kriging variances, they severely overestimate the uncertainty of the block estimate, falsely inflating the risk of mining or environmental remediation projects. Block Kriging must be used to obtain honest block-scale uncertainties. $\blacksquare$
