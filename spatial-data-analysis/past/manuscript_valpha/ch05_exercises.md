# Chapter 5 Companion Exercises: Best Linear Unbiased Prediction (BLUP) Framework

This study guide and exercise workbook accompanies [Chapter 5: Spatial Prediction and Kriging Theory (BLUP)](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch05_blup_kriging.md). In keeping with the pedagogical framework of Maria L. Rizzo (*Statistical Computing with R*) and the spatial linear modeling discipline of Oliver Schabenberger and Carol A. Gotway (*Statistical Methods for Spatial Data Analysis*), these exercises are structured into five rigorous tiers:

1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 5.1–5.4)
2. **Tier I: Core Theoretical Re-Derivations & Notation Mastery** (Exercises 5.5–5.8)
3. **Tier II: Pencil-and-Paper Matrix & Vector Mechanics** (Exercises 5.9–5.11)
4. **Tier III: Algorithmic Translation & Simulation Labs in Dependency-Free Python** (Exercises 5.12–5.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 5.14–5.16)

Every exercise includes an unabridged, mathematically rigorous step-by-step solution.

---

## Tier 0: Classical-to-Spatial Warm-Ups

### Exercise 5.1: The Fundamental Prediction Theorem and Optimal MSPE
Let $(\Omega, \mathcal{F}, P)$ be a probability space, and let $Y \in L^2(\Omega, \mathcal{F}, P)$ and $\mathbf{U} \in \mathbb{R}^k$ be random variables with finite second moments: $\operatorname{E}[Y^2] < \infty$ and $\operatorname{E}[\|\mathbf{U}\|^2] < \infty$. Consider predicting the unobserved scalar $Y$ using a measurable function $g(\mathbf{U})$. The performance of any such predictor is evaluated by the mean squared prediction error:
$$
\operatorname{MSPE}(g) = \operatorname{E}\left[ (Y - g(\mathbf{U}))^2 \right]
$$
**(a)** Prove the **Fundamental Prediction Theorem**: $\operatorname{MSPE}(g)$ is uniquely minimized (almost surely) by the conditional expectation function $g^*(\mathbf{U}) = \operatorname{E}[Y \mid \mathbf{U}]$. Proceed by adding and subtracting $\operatorname{E}[Y \mid \mathbf{U}]$ inside the squared prediction error, expanding the quadratic form, and applying the Law of Total Expectation to prove that the cross-product term vanishes:
$$
\operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}]) (\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U})) \right] = 0
$$
**(b)** Prove that the minimal MSPE achieved by the optimal predictor satisfies the variance decomposition identity:
$$
\min_g \operatorname{MSPE}(g) = \operatorname{Var}(Y) - \operatorname{Var}(\operatorname{E}[Y \mid \mathbf{U}])
$$
Deduce that the variance of the optimal predictor $\operatorname{Var}(\operatorname{E}[Y \mid \mathbf{U}])$ represents the reduction in prediction uncertainty relative to the unconditional point variance $\operatorname{Var}(Y)$.
**(c)** In spatial statistics, let $Z(\mathbf{s}_0)$ be the unobserved value at target location $\mathbf{s}_0$, and let $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$ be the observation vector with known second-order moments $\operatorname{E}[\mathbf{Z}] = \boldsymbol{\mu}$, $\operatorname{Var}(\mathbf{Z}) = \mathbf{V}$, and $\operatorname{Cov}(\mathbf{Z}, Z(\mathbf{s}_0)) = \mathbf{c}_0$. Let $p^0(\mathbf{Z}; \mathbf{s}_0) = \boldsymbol{\lambda}'\mathbf{Z}$ be the Best Linear Unbiased Predictor (BLUP) under second-order moments. Prove that:
$$
\operatorname{MSPE}(p^0) = \operatorname{Var}(Z(\mathbf{s}_0)) - \operatorname{Var}(p^0(\mathbf{Z}; \mathbf{s}_0))
$$
demonstrating that the optimal linear prediction variance is always total point variance minus the variance of the predictor.

##### Solution to Exercise 5.1
**(a)** Let $g(\mathbf{U})$ be any measurable function of $\mathbf{U}$ with finite second moment. Add and subtract the conditional expectation $\operatorname{E}[Y \mid \mathbf{U}]$ inside the squared error:
$$
Y - g(\mathbf{U}) = (Y - \operatorname{E}[Y \mid \mathbf{U}]) + (\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U}))
$$
Squaring both sides and taking expectations:
$$
\operatorname{MSPE}(g) = \operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}])^2 \right] + 2 \operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}]) (\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U})) \right] + \operatorname{E}\left[ (\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U}))^2 \right]
$$
Define $h(\mathbf{U}) = \operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U})$. Since both $\operatorname{E}[Y \mid \mathbf{U}]$ and $g(\mathbf{U})$ are measurable with respect to the $\sigma$-algebra $\sigma(\mathbf{U})$, $h(\mathbf{U})$ is $\sigma(\mathbf{U})$-measurable. By the Law of Iterated Expectations (Tower Property):
$$
\operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}]) h(\mathbf{U}) \right] = \operatorname{E}\left[ \operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}]) h(\mathbf{U}) \mid \mathbf{U} \right] \right]
$$
Because $h(\mathbf{U})$ is known given $\mathbf{U}$, it factors out of the inner conditional expectation:
$$
\operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}]) h(\mathbf{U}) \mid \mathbf{U} \right] = h(\mathbf{U}) \operatorname{E}\left[ Y - \operatorname{E}[Y \mid \mathbf{U}] \mid \mathbf{U} \right] = h(\mathbf{U}) \left( \operatorname{E}[Y \mid \mathbf{U}] - \operatorname{E}[Y \mid \mathbf{U}] \right) = 0
$$
Taking the outer expectation yields:
$$
\operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}]) (\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U})) \right] = \operatorname{E}[0] = 0
$$
Substituting this back into the expanded MSPE expression:
$$
\operatorname{MSPE}(g) = \operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}])^2 \right] + \operatorname{E}\left[ (\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U}))^2 \right]
$$
Since the second term is the expectation of a squared real-valued quantity, $\operatorname{E}[(\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U}))^2] \ge 0$. Consequently:
$$
\operatorname{MSPE}(g) \ge \operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}])^2 \right]
$$
with equality if and only if $\operatorname{E}[(\operatorname{E}[Y \mid \mathbf{U}] - g(\mathbf{U}))^2] = 0$, which holds if and only if $g(\mathbf{U}) = \operatorname{E}[Y \mid \mathbf{U}]$ almost surely ($P$-a.s.). Hence $g^*(\mathbf{U}) = \operatorname{E}[Y \mid \mathbf{U}]$ is the unique optimal predictor.

**(b)** Evaluating $\operatorname{MSPE}(g^*)$:
$$
\min_g \operatorname{MSPE}(g) = \operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}])^2 \right]
$$
Applying the Law of Iterated Expectations to the squared residual:
$$
\operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}])^2 \right] = \operatorname{E}\left[ \operatorname{E}\left[ (Y - \operatorname{E}[Y \mid \mathbf{U}])^2 \mid \mathbf{U} \right] \right] = \operatorname{E}[\operatorname{Var}(Y \mid \mathbf{U})]
$$
Recall Eve's Law (the Law of Total Variance):
$$
\operatorname{Var}(Y) = \operatorname{E}[\operatorname{Var}(Y \mid \mathbf{U})] + \operatorname{Var}(\operatorname{E}[Y \mid \mathbf{U}])
$$
Subtracting $\operatorname{Var}(\operatorname{E}[Y \mid \mathbf{U}])$ from both sides:
$$
\min_g \operatorname{MSPE}(g) = \operatorname{E}[\operatorname{Var}(Y \mid \mathbf{U})] = \operatorname{Var}(Y) - \operatorname{Var}(\operatorname{E}[Y \mid \mathbf{U}])
$$
The variance $\operatorname{Var}(Y)$ is the total unconditional uncertainty about $Y$. The quantity $\operatorname{Var}(\operatorname{E}[Y \mid \mathbf{U}])$ represents the variance of the predictor, which measures the amount of spatial dispersion in $Y$ captured by the conditioning information $\mathbf{U}$. The minimal MSPE is the remaining unexplained conditional variance.

**(c)** Consider the linear predictor $p^0(\mathbf{Z}; \mathbf{s}_0) = \boldsymbol{\lambda}'\mathbf{Z}$. Because $p^0$ is an unbiased predictor, $\operatorname{E}[Z(\mathbf{s}_0) - p^0(\mathbf{Z}; \mathbf{s}_0)] = 0$. The MSPE is therefore equal to the prediction error variance:
$$
\operatorname{MSPE}(p^0) = \operatorname{Var}(Z(\mathbf{s}_0) - p^0(\mathbf{Z}; \mathbf{s}_0)) = \operatorname{Var}(Z(\mathbf{s}_0)) - 2\operatorname{Cov}(Z(\mathbf{s}_0), p^0(\mathbf{Z}; \mathbf{s}_0)) + \operatorname{Var}(p^0(\mathbf{Z}; \mathbf{s}_0))
$$
Now, expand the covariance between the target $Z(\mathbf{s}_0)$ and the predictor $p^0 = \boldsymbol{\lambda}'\mathbf{Z}$:
$$
\operatorname{Cov}(Z(\mathbf{s}_0), \boldsymbol{\lambda}'\mathbf{Z}) = \boldsymbol{\lambda}' \operatorname{Cov}(\mathbf{Z}, Z(\mathbf{s}_0)) = \boldsymbol{\lambda}' \mathbf{c}_0
$$
In Simple Kriging (or in the centered Hilbert space projection of Exercise 5.5), the optimal weight vector satisfies the normal equations $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$. Taking the inner product with $\boldsymbol{\lambda}$:
$$
\operatorname{Var}(p^0(\mathbf{Z}; \mathbf{s}_0)) = \operatorname{Var}(\boldsymbol{\lambda}'\mathbf{Z}) = \boldsymbol{\lambda}' \operatorname{Var}(\mathbf{Z}) \boldsymbol{\lambda} = \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda} = \boldsymbol{\lambda}' \mathbf{c}_0
$$
Comparing the two terms:
$$
\operatorname{Cov}(Z(\mathbf{s}_0), p^0(\mathbf{Z}; \mathbf{s}_0)) = \operatorname{Var}(p^0(\mathbf{Z}; \mathbf{s}_0))
$$
Substituting this fundamental orthogonality identity into the MSPE expression:
$$
\operatorname{MSPE}(p^0) = \operatorname{Var}(Z(\mathbf{s}_0)) - 2\operatorname{Var}(p^0(\mathbf{Z}; \mathbf{s}_0)) + \operatorname{Var}(p^0(\mathbf{Z}; \mathbf{s}_0)) = \operatorname{Var}(Z(\mathbf{s}_0)) - \operatorname{Var}(p^0(\mathbf{Z}; \mathbf{s}_0))
$$
This identity reveals the core geometry of linear kriging: every increment of variance acquired by the optimal predictor translates directly into a one-for-one reduction in mean squared prediction error. $\blacksquare$

---

### Exercise 5.2: Goldberger's $2 \times 3$ Discrete Contingency Table: Non-Linear vs. Linear Prediction
Consider a discrete bivariate random vector $(X, Y)$ whose joint probability mass function is specified by Goldberger (1962, *Econometric Theory*):
$$
P(X = x, Y = y) = \begin{cases}
0.10 & \text{if } x = 1, y = 1 \\
0.20 & \text{if } x = 1, y = 2 \\
0.10 & \text{if } x = 1, y = 3 \\
0.15 & \text{if } x = 2, y = 1 \\
0.10 & \text{if } x = 2, y = 2 \\
0.35 & \text{if } x = 2, y = 3
\end{cases}
$$
**(a)** Compute the marginal probability mass functions $P(X = x)$ and $P(Y = y)$, the expectations $\operatorname{E}[X]$ and $\operatorname{E}[Y]$, the variances $\operatorname{Var}(X)$ and $\operatorname{Var}(Y)$, and the covariance $\operatorname{Cov}(X, Y)$.
**(b)** Compute the exact non-linear conditional expectation $\operatorname{E}[Y \mid X = 1]$ and $\operatorname{E}[Y \mid X = 2]$.
**(c)** Construct the Best Linear Predictor $\operatorname{BLP}(Y \mid X) = \alpha + \beta X$, where $\beta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}$ and $\alpha = \operatorname{E}[Y] - \beta \operatorname{E}[X]$. Evaluate $\operatorname{BLP}(Y \mid X)$ at $X = 1$ and $X = 2$.
**(d)** Calculate the mean squared error for both predictors: $\operatorname{MSE}(\operatorname{E}[Y \mid X])$ and $\operatorname{MSE}(\operatorname{BLP}(Y \mid X))$. Quantify the efficiency loss incurred by restricting prediction to linear forms, and explain the theoretical reason behind the result.

##### Solution to Exercise 5.2
**(a)** Marginal distributions:
Summing across columns for $X$:
$$
P(X = 1) = 0.10 + 0.20 + 0.10 = 0.40, \quad P(X = 2) = 0.15 + 0.10 + 0.35 = 0.60
$$
Summing across rows for $Y$:
$$
P(Y = 1) = 0.10 + 0.15 = 0.25, \quad P(Y = 2) = 0.20 + 0.10 = 0.30, \quad P(Y = 3) = 0.10 + 0.35 = 0.45
$$
Both marginals sum to $1.00$.
Moments of $X$:
$$
\operatorname{E}[X] = 1(0.40) + 2(0.60) = 0.40 + 1.20 = 1.60 = \frac{8}{5}
$$
$$
\operatorname{E}[X^2] = 1^2(0.40) + 2^2(0.60) = 0.40 + 2.40 = 2.80 = \frac{14}{5}
$$
$$
\operatorname{Var}(X) = \operatorname{E}[X^2] - (\operatorname{E}[X])^2 = 2.80 - 1.60^2 = 2.80 - 2.56 = 0.24 = \frac{6}{25}
$$
Moments of $Y$:
$$
\operatorname{E}[Y] = 1(0.25) + 2(0.30) + 3(0.45) = 0.25 + 0.60 + 1.35 = 2.20 = \frac{11}{5}
$$
$$
\operatorname{E}[Y^2] = 1^2(0.25) + 2^2(0.30) + 3^2(0.45) = 0.25 + 1.20 + 4.05 = 5.50 = \frac{11}{2}
$$
$$
\operatorname{Var}(Y) = \operatorname{E}[Y^2] - (\operatorname{E}[Y])^2 = 5.50 - 2.20^2 = 5.50 - 4.84 = 0.66 = \frac{33}{50}
$$
Joint expectation and covariance:
$$
\begin{aligned}
\operatorname{E}[XY] &= 1\cdot 1(0.10) + 1\cdot 2(0.20) + 1\cdot 3(0.10) + 2\cdot 1(0.15) + 2\cdot 2(0.10) + 2\cdot 3(0.35) \\
&= 0.10 + 0.40 + 0.30 + 0.30 + 0.40 + 2.10 = 3.60 = \frac{18}{5}
\end{aligned}
$$
$$
\operatorname{Cov}(X, Y) = \operatorname{E}[XY] - \operatorname{E}[X]\operatorname{E}[Y] = 3.60 - (1.60)(2.20) = 3.60 - 3.52 = 0.08 = \frac{2}{25}
$$

**(b)** The conditional expectation function evaluates to:
For $X = 1$:
$$
\operatorname{E}[Y \mid X = 1] = \sum_{y=1}^3 y P(Y = y \mid X = 1) = \frac{1(0.10) + 2(0.20) + 3(0.10)}{P(X = 1)} = \frac{0.80}{0.40} = 2.0 = 2
$$
For $X = 2$:
$$
\operatorname{E}[Y \mid X = 2] = \sum_{y=1}^3 y P(Y = y \mid X = 2) = \frac{1(0.15) + 2(0.10) + 3(0.35)}{P(X = 2)} = \frac{1.40}{0.60} = \frac{7}{3} \approx 2.3333
$$

**(c)** For the Best Linear Predictor $\operatorname{BLP}(Y \mid X) = \alpha + \beta X$:
$$
\beta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)} = \frac{0.08}{0.24} = \frac{1}{3}
$$
$$
\alpha = \operatorname{E}[Y] - \beta \operatorname{E}[X] = 2.20 - \frac{1}{3}(1.60) = \frac{11}{5} - \frac{8}{15} = \frac{33 - 8}{15} = \frac{25}{15} = \frac{5}{3}
$$
Thus the Best Linear Predictor is:
$$
\operatorname{BLP}(Y \mid X) = \frac{5}{3} + \frac{1}{3}X
$$
Evaluating this affine function at the two support points:
- At $X = 1$: $\operatorname{BLP}(Y \mid X = 1) = \frac{5}{3} + \frac{1}{3}(1) = \frac{6}{3} = 2.0$
- At $X = 2$: $\operatorname{BLP}(Y \mid X = 2) = \frac{5}{3} + \frac{1}{3}(2) = \frac{7}{3} \approx 2.3333$

**(d)** Mean squared error calculation:
For the optimal conditional expectation $\operatorname{E}[Y \mid X]$:
$$
\begin{aligned}
\operatorname{MSE}(\operatorname{E}[Y \mid X]) &= \sum_{x \in \{1, 2\}} \sum_{y \in \{1, 2, 3\}} P(X = x, Y = y) \left( y - \operatorname{E}[Y \mid X = x] \right)^2 \\
&= 0.10(1 - 2)^2 + 0.20(2 - 2)^2 + 0.10(3 - 2)^2 \\
&\quad + 0.15\left(1 - \frac{7}{3}\right)^2 + 0.10\left(2 - \frac{7}{3}\right)^2 + 0.35\left(3 - \frac{7}{3}\right)^2 \\
&= 0.10(1) + 0 + 0.10(1) + 0.15\left(\frac{16}{9}\right) + 0.10\left(\frac{1}{9}\right) + 0.35\left(\frac{4}{9}\right) \\
&= 0.20 + \frac{2.40 + 0.10 + 1.40}{9} = \frac{1}{5} + \frac{3.90}{9} = \frac{1}{5} + \frac{13}{30} = \frac{19}{30} \approx 0.633333
\end{aligned}
$$
For the Best Linear Predictor $\operatorname{BLP}(Y \mid X)$:
Since $\operatorname{BLP}(Y \mid X = 1) = \operatorname{E}[Y \mid X = 1] = 2$ and $\operatorname{BLP}(Y \mid X = 2) = \operatorname{E}[Y \mid X = 2] = 7/3$, the point predictions are identical for all realization pairs $(x, y)$. Consequently:
$$
\operatorname{MSE}(\operatorname{BLP}(Y \mid X)) = \frac{19}{30} \approx 0.633333
$$
Using the BLP variance decomposition identity:
$$
\operatorname{MSE}(\operatorname{BLP}) = \operatorname{Var}(Y) - \frac{\operatorname{Cov}(X, Y)^2}{\operatorname{Var}(X)} = \frac{33}{50} - \frac{(2/25)^2}{6/25} = \frac{33}{50} - \frac{4}{150} = \frac{99 - 4}{150} = \frac{95}{150} = \frac{19}{30}
$$
The efficiency loss incurred by restricting to linear prediction is:
$$
\operatorname{MSE}(\operatorname{BLP}(Y \mid X)) - \operatorname{MSE}(\operatorname{E}[Y \mid X]) = \frac{19}{30} - \frac{19}{30} = 0
$$
Theoretical Reflection: In general, $\operatorname{E}[Y \mid \mathbf{X}]$ is non-linear unless $(Y, \mathbf{X})$ is multivariate Gaussian. However, whenever the conditioning variable $X$ has binary support ($|\operatorname{supp}(X)| = 2$), any function $g: \{x_1, x_2\} \to \mathbb{R}$ is uniquely parameterized by an intercept and a slope: two points define a unique straight line. Thus, the linear predictor is non-parametrically saturated, matching the Bayes-optimal non-linear predictor with zero efficiency loss despite the heavily non-Gaussian, discrete distribution! $\blacksquare$

---

### Exercise 5.3: Continuous Bivariate Density $f(x,y) = \frac{6}{7}(x+y)^2$
Let $(X, Y)$ be continuous random variables supported on the unit square $[0, 1]^2$ with joint probability density function:
$$
f(x, y) = \frac{6}{7}(x + y)^2, \quad 0 \le x \le 1, \; 0 \le y \le 1
$$
**(a)** Verify that $f(x, y)$ is a valid probability density function by proving that $\int_0^1 \int_0^1 f(x, y) dx dy = 1$.
**(b)** Derive the marginal probability density function $f_X(x)$ for $x \in [0, 1]$.
**(c)** Derive the exact conditional expectation $\operatorname{E}[Y \mid X = x] = \int_0^1 y f(y \mid x) dy$. Prove that $\operatorname{E}[Y \mid X = x]$ is a strictly non-linear function of $x$.
**(d)** Compute $\operatorname{E}[X]$, $\operatorname{E}[Y]$, $\operatorname{Var}(X)$, $\operatorname{Var}(Y)$, and $\operatorname{Cov}(X, Y)$. Formulate the Best Linear Predictor $\operatorname{BLP}(Y \mid X) = \alpha + \beta X$.
**(e)** Calculate and compare the mean squared prediction errors: $\operatorname{MSPE}(\operatorname{E}[Y \mid X])$ versus $\operatorname{MSPE}(\operatorname{BLP}(Y \mid X))$. Quantify the efficiency loss of the BLP.

##### Solution to Exercise 5.3
**(a)** Integrating over the unit square $[0, 1]^2$:
$$
\int_0^1 \int_0^1 \frac{6}{7}(x + y)^2 dy dx = \frac{6}{7} \int_0^1 \left[ \int_0^1 (x^2 + 2xy + y^2) dy \right] dx
$$
Inner integral:
$$
\int_0^1 (x^2 + 2xy + y^2) dy = \left[ x^2 y + x y^2 + \frac{y^3}{3} \right]_{y=0}^{y=1} = x^2 + x + \frac{1}{3}
$$
Outer integral:
$$
\frac{6}{7} \int_0^1 \left( x^2 + x + \frac{1}{3} \right) dx = \frac{6}{7} \left[ \frac{x^3}{3} + \frac{x^2}{2} + \frac{x}{3} \right]_0^1 = \frac{6}{7} \left( \frac{1}{3} + \frac{1}{2} + \frac{1}{3} \right) = \frac{6}{7} \left( \frac{7}{6} \right) = 1
$$
Because $f(x, y) \ge 0$ on $[0, 1]^2$ and integrates to 1, it is a valid probability density function.

**(b)** The marginal density of $X$ is obtained by integrating out $y$:
$$
f_X(x) = \int_0^1 \frac{6}{7}(x + y)^2 dy = \frac{6}{7} \left( x^2 + x + \frac{1}{3} \right) = \frac{2}{7}(3x^2 + 3x + 1) = \frac{6x^2 + 6x + 2}{7}, \quad x \in [0, 1]
$$
By symmetry of $f(x, y) = f(y, x)$, $f_Y(y) = \frac{2}{7}(3y^2 + 3y + 1)$.

**(c)** The conditional density of $Y$ given $X = x$ is $f(y \mid x) = \frac{f(x, y)}{f_X(x)}$. The conditional expectation is:
$$
\operatorname{E}[Y \mid X = x] = \int_0^1 y f(y \mid x) dy = \frac{\int_0^1 y f(x, y) dy}{f_X(x)}
$$
Compute the numerator:
$$
\begin{aligned}
\int_0^1 y \cdot \frac{6}{7}(x + y)^2 dy &= \frac{6}{7} \int_0^1 (x^2 y + 2xy^2 + y^3) dy = \frac{6}{7} \left[ \frac{x^2 y^2}{2} + \frac{2xy^3}{3} + \frac{y^4}{4} \right]_0^1 \\
&= \frac{6}{7} \left( \frac{x^2}{2} + \frac{2x}{3} + \frac{1}{4} \right) = \frac{6}{7} \cdot \frac{6x^2 + 8x + 3}{12} = \frac{6x^2 + 8x + 3}{14}
\end{aligned}
$$
Dividing by $f_X(x) = \frac{2(3x^2 + 3x + 1)}{7}$:
$$
\operatorname{E}[Y \mid X = x] = \frac{\frac{6x^2 + 8x + 3}{14}}{\frac{2(3x^2 + 3x + 1)}{7}} = \frac{6x^2 + 8x + 3}{4(3x^2 + 3x + 1)} = \frac{6x^2 + 8x + 3}{12x^2 + 12x + 4}
$$
To verify non-linearity, compute the second derivative $\frac{d^2}{dx^2}\operatorname{E}[Y \mid X = x]$. Let $N(x) = 6x^2 + 8x + 3$ and $D(x) = 12x^2 + 12x + 4$:
$$
\frac{d}{dx}\operatorname{E}[Y \mid X = x] = \frac{(12x + 8)(12x^2 + 12x + 4) - (6x^2 + 8x + 3)(24x + 12)}{(12x^2 + 12x + 4)^2} = \frac{-24x^2 - 16x - 4}{(12x^2 + 12x + 4)^2}
$$
The derivative depends non-trivially on $x$ (it is not a constant), so the second derivative is non-zero on $(0, 1)$. Hence $\operatorname{E}[Y \mid X = x]$ is strictly non-linear.

**(d)** Moments of $X$ and $Y$:
$$
\operatorname{E}[X] = \int_0^1 x \cdot \frac{2}{7}(3x^2 + 3x + 1) dx = \frac{2}{7} \left[ \frac{3x^4}{4} + x^3 + \frac{x^2}{2} \right]_0^1 = \frac{2}{7} \left( \frac{3}{4} + 1 + \frac{1}{2} \right) = \frac{2}{7} \left( \frac{9}{4} \right) = \frac{9}{14} \approx 0.642857
$$
By symmetry, $\operatorname{E}[Y] = \frac{9}{14}$.
Second moment of $X$:
$$
\operatorname{E}[X^2] = \int_0^1 x^2 \cdot \frac{2}{7}(3x^2 + 3x + 1) dx = \frac{2}{7} \left[ \frac{3x^5}{5} + \frac{3x^4}{4} + \frac{x^3}{3} \right]_0^1 = \frac{2}{7} \left( \frac{3}{5} + \frac{3}{4} + \frac{1}{3} \right) = \frac{2}{7} \left( \frac{101}{60} \right) = \frac{101}{210}
$$
Variance of $X$:
$$
\operatorname{Var}(X) = \frac{101}{210} - \left( \frac{9}{14} \right)^2 = \frac{101}{210} - \frac{81}{196} = \frac{1414 - 1215}{2940} = \frac{199}{2940} \approx 0.067687
$$
By symmetry, $\operatorname{Var}(Y) = \frac{199}{2940}$.
Joint moment $\operatorname{E}[XY]$:
$$
\begin{aligned}
\operatorname{E}[XY] &= \int_0^1 \int_0^1 xy \cdot \frac{6}{7}(x^2 + 2xy + y^2) dy dx = \frac{6}{7} \int_0^1 x \left[ \frac{x^2 y^2}{2} + \frac{2xy^3}{3} + \frac{y^4}{4} \right]_0^1 dx \\
&= \frac{6}{7} \int_0^1 \left( \frac{x^3}{2} + \frac{2x^2}{3} + \frac{x}{4} \right) dx = \frac{6}{7} \left( \frac{1}{8} + \frac{2}{9} + \frac{1}{8} \right) = \frac{6}{7} \left( \frac{1}{4} + \frac{2}{9} \right) = \frac{6}{7} \left( \frac{17}{36} \right) = \frac{17}{42}
\end{aligned}
$$
Covariance:
$$
\operatorname{Cov}(X, Y) = \frac{17}{42} - \left( \frac{9}{14} \right)^2 = \frac{17}{42} - \frac{81}{196} = \frac{238 - 243}{588} = -\frac{5}{588} \approx -0.008503
$$
Parameters of the Best Linear Predictor:
$$
\beta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)} = \frac{-5/588}{199/2940} = \frac{-5 \times 2940}{588 \times 199} = -\frac{25}{199} \approx -0.125628
$$
$$
\alpha = \operatorname{E}[Y] - \beta \operatorname{E}[X] = \frac{9}{14} - \left( -\frac{25}{199} \right) \frac{9}{14} = \frac{9}{14} \left( 1 + \frac{25}{199} \right) = \frac{9}{14} \cdot \frac{224}{199} = \frac{144}{199} \approx 0.723618
$$
Thus:
$$
\operatorname{BLP}(Y \mid X = x) = \frac{144}{199} - \frac{25}{199} x
$$

**(e)** Prediction error comparison:
For the Best Linear Predictor:
$$
\operatorname{MSPE}(\operatorname{BLP}) = \operatorname{Var}(Y) - \frac{\operatorname{Cov}(X, Y)^2}{\operatorname{Var}(X)} = \frac{199}{2940} - \frac{(-5/588)^2}{199/2940} = \frac{199}{2940} - \frac{125}{117012} = \frac{464}{6965} \approx 0.06661881
$$
For the optimal non-linear predictor $g^*(x) = \operatorname{E}[Y \mid X = x]$:
$$
\operatorname{MSPE}(g^*) = \int_0^1 \int_0^1 (y - g^*(x))^2 f(x, y) dy dx = \frac{9}{140} + \frac{\sqrt{3}}{252}\arctan(3\sqrt{3}) - \frac{\sqrt{3}\pi}{756} \approx 0.06657775
$$
The absolute efficiency loss of the BLP relative to the optimal non-linear predictor is:
$$
\Delta = \operatorname{MSPE}(\operatorname{BLP}) - \operatorname{MSPE}(g^*) \approx 0.06661881 - 0.06657775 = 0.00004106
$$
The relative efficiency loss is:
$$
\frac{\Delta}{\operatorname{MSPE}(g^*)} = \frac{0.00004106}{0.06657775} \approx 0.0617\%
$$
Less than one-tenth of one percent of predictive precision is lost by enforcing linearity. This serves as a primary justification for the BLUP/Kriging paradigm in spatial statistics: even when spatial distributions depart from bivariate Gaussianity, the linear predictor captures virtually all predictable variation. $\blacksquare$

---

### Exercise 5.4: Extrapolation Outside the Convex Hull and Screening Diagnostics
Consider three observation locations on the real line $\mathbb{R}^1$ at coordinates $s_1 = 0$, $s_2 = 1$, and $s_3 = 2$. The spatial process $\{Z(s) : s \in \mathbb{R}\}$ has unknown constant mean $\mu$ and stationary exponential covariance function:
$$
C(h) = \sigma^2 \exp(-h / a)
$$
with sill $\sigma^2 > 0$ and range parameter $a > 0$ (zero nugget, $\tau^2 = 0$). We consider predicting the spatial process at two target locations:
1. Interior interpolation location: $s_0 = 0.5$ (inside the convex hull $[0, 2]$).
2. Exterior extrapolation location: $s_0 = 3.0$ (outside the convex hull $[0, 2]$).

**(a)** Construct the $3 \times 3$ observation covariance matrix $\mathbf{V}$ and the target cross-covariance vectors $\mathbf{c}_0$ for both $s_0 = 0.5$ and $s_0 = 3.0$. Parameterize the system using $\rho = \exp(-0.5/a) \in (0, 1)$ for the interpolation problem and $r = \exp(-1.0/a) = \rho^2 \in (0, 1)$ for the extrapolation problem.
**(b)** Formulate and solve the Ordinary Kriging system for the interpolation target $s_0 = 0.5$. Compute the weights $\lambda_1, \lambda_2, \lambda_3$ explicitly as rational functions of $\rho$. Contrast the Simple Kriging weights with Ordinary Kriging, and characterize the screening effect on the distant point $s_3$.
**(c)** Formulate and solve the Ordinary Kriging system for the extrapolation target $s_0 = 3.0$. Compute the weights $\lambda_1, \lambda_2, \lambda_3$ as rational functions of $r$. Analyze the asymptotic behavior of the weights as the spatial correlation range $a \to 0$ ($r \to 0$) and as $a \to \infty$ ($r \to 1$).
**(d)** Derive the Ordinary Kriging prediction variance $\sigma_{\text{OK}}^2(s_0)$ at both locations. Explain mathematically and physically why prediction variance inflates rapidly as the target location moves beyond the convex hull of the data.

##### Solution to Exercise 5.4
**(a)** Inter-point distances among the observations are $|s_1 - s_2| = 1$, $|s_2 - s_3| = 1$, and $|s_1 - s_3| = 2$.
Let $r = \exp(-1/a) \in (0, 1)$. Then $C(1) = \sigma^2 r$ and $C(2) = \sigma^2 r^2$.
The observation covariance matrix is:
$$
\mathbf{V} = \sigma^2 \begin{bmatrix} 1 & r & r^2 \\ r & 1 & r \\ r^2 & r & 1 \end{bmatrix}
$$
For the interpolation target $s_0 = 0.5$:
The distances to observations are $|0.5 - 0| = 0.5$, $|0.5 - 1| = 0.5$, and $|0.5 - 2| = 1.5$.
Let $\rho = \exp(-0.5/a) \in (0, 1)$, so $r = \rho^2$ and $\exp(-1.5/a) = \rho^3$.
$$
\mathbf{V} = \sigma^2 \begin{bmatrix} 1 & \rho^2 & \rho^4 \\ \rho^2 & 1 & \rho^2 \\ \rho^4 & \rho^2 & 1 \end{bmatrix}, \quad \mathbf{c}_0 = \sigma^2 \begin{bmatrix} \rho \\ \rho \\ \rho^3 \end{bmatrix}
$$
For the extrapolation target $s_0 = 3.0$:
The distances to observations are $|3 - 0| = 3$, $|3 - 1| = 2$, and $|3 - 2| = 1$.
$$
\mathbf{c}_0 = \sigma^2 \begin{bmatrix} r^3 \\ r^2 \\ r \end{bmatrix}
$$

**(b)** Ordinary Kriging system for interpolation ($s_0 = 0.5$):
The augmented $(3+1) \times (3+1)$ system is:
$$
\begin{bmatrix} 1 & \rho^2 & \rho^4 & 1 \\ \rho^2 & 1 & \rho^2 & 1 \\ \rho^4 & \rho^2 & 1 & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ m / \sigma^2 \end{bmatrix} = \begin{bmatrix} \rho \\ \rho \\ \rho^3 \\ 1 \end{bmatrix}
$$
To understand the screening structure, first inspect the Simple Kriging weights $\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0$. Inverting $\mathbf{V}$:
$$
\mathbf{V}^{-1} = \frac{1}{\sigma^2 (1 - \rho^4)} \begin{bmatrix} 1 & -\rho^2 & 0 \\ -\rho^2 & 1 + \rho^4 & -\rho^2 \\ 0 & -\rho^2 & 1 \end{bmatrix}
$$
Multiplying by $\mathbf{c}_0 = \sigma^2 (\rho, \rho, \rho^3)'$:
$$
\boldsymbol{\lambda}_{\text{SK}} = \begin{bmatrix} \frac{\rho}{1 + \rho^2} \\ \frac{\rho}{1 + \rho^2} \\ 0 \end{bmatrix}
$$
In Simple Kriging, the third observation $s_3 = 2$ receives weight exactly zero ($\lambda_{\text{SK}, 3} = 0$). This is the celebrated 1D Markov screening property of the exponential covariance: conditional on the nearest enclosing neighbors $s_1 = 0$ and $s_2 = 1$, all external observations are conditionally independent of $s_0 = 0.5$.

Now solve the Ordinary Kriging system with the constraint $\mathbf{1}'\boldsymbol{\lambda} = 1$:
Recall that $\boldsymbol{\lambda}_{\text{OK}} = \boldsymbol{\lambda}_{\text{SK}} + \frac{1 - \mathbf{1}'\boldsymbol{\lambda}_{\text{SK}}}{\mathbf{1}'\mathbf{V}^{-1}\mathbf{1}} \mathbf{V}^{-1}\mathbf{1}$.
We have $\mathbf{1}'\boldsymbol{\lambda}_{\text{SK}} = \frac{2\rho}{1 + \rho^2}$, so $1 - \mathbf{1}'\boldsymbol{\lambda}_{\text{SK}} = \frac{(1 - \rho)^2}{1 + \rho^2}$.
Also, $\mathbf{V}^{-1}\mathbf{1} = \frac{1}{\sigma^2 (1 + \rho^2)} (1, 1 - \rho^2, 1)'$, with $\mathbf{1}'\mathbf{V}^{-1}\mathbf{1} = \frac{3 - \rho^2}{\sigma^2 (1 + \rho^2)}$.
Therefore:
$$
\boldsymbol{\lambda}_{\text{OK}} = \begin{bmatrix} \frac{\rho}{1 + \rho^2} \\ \frac{\rho}{1 + \rho^2} \\ 0 \end{bmatrix} + \frac{(1 - \rho)^2}{(1 + \rho^2)(3 - \rho^2)} \begin{bmatrix} 1 \\ 1 - \rho^2 \\ 1 \end{bmatrix}
$$
Combining terms yields the exact rational weights:
$$
\lambda_1 = \frac{1 + \rho + \rho^2 - \rho^3}{(3 - \rho^2)(1 + \rho^2)}
$$
$$
\lambda_2 = \frac{1 + \rho - \rho^2}{3 - \rho^2}
$$
$$
\lambda_3 = \frac{(1 - \rho)^2}{(3 - \rho^2)(1 + \rho^2)}
$$
$$
\frac{m}{\sigma^2} = -\frac{(1 - \rho)^2}{3 - \rho^2}
$$
Sum of weights: $\lambda_1 + \lambda_2 + \lambda_3 = 1$.
Notice that because $\rho \in (0, 1)$, $\lambda_3 = \frac{(1-\rho)^2}{(3-\rho^2)(1+\rho^2)} > 0$ enters strictly to absorb the estimation of the unknown mean $\mu$. As the spatial correlation becomes strong ($\rho \to 1$), the mean correction factor $(1-\rho)^2 \to 0$, causing $\lambda_3 \to 0$ and $\lambda_1, \lambda_2 \to 0.5$. Thus the screening effect is asymptotically exact in Ordinary Kriging as well.

**(c)** Ordinary Kriging system for extrapolation ($s_0 = 3.0$):
The Simple Kriging weights are:
$$
\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0 = \frac{1}{\sigma^2 (1 - r^2)} \begin{bmatrix} 1 & -r & 0 \\ -r & 1 + r^2 & -r \\ 0 & -r & 1 \end{bmatrix} \sigma^2 \begin{bmatrix} r^3 \\ r^2 \\ r \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ r \end{bmatrix}
$$
In Simple Kriging, the boundary point $s_3 = 2$ screens both interior points $s_1$ and $s_2$ completely ($\lambda_{\text{SK}, 1} = \lambda_{\text{SK}, 2} = 0$).
For Ordinary Kriging, adding the mean estimation correction:
$$
\boldsymbol{\lambda}_{\text{OK}} = \begin{bmatrix} 0 \\ 0 \\ r \end{bmatrix} + \frac{1 - r}{3 - r} \begin{bmatrix} 1 \\ 1 - r \\ 1 \end{bmatrix}
$$
This yields the explicit weights:
$$
\lambda_1 = \frac{1 - r}{3 - r}
$$
$$
\lambda_2 = \frac{(1 - r)^2}{3 - r}
$$
$$
\lambda_3 = r + \frac{1 - r}{3 - r} = \frac{1 + 2r - r^2}{3 - r}
$$
$$
\frac{m}{\sigma^2} = -\frac{1 - r^2}{3 - r}
$$
Sum check: $\lambda_1 + \lambda_2 + \lambda_3 = \frac{(1 - r) + (1 - 2r + r^2) + (1 + 2r - r^2)}{3 - r} = \frac{3 - r}{3 - r} = 1$.
Asymptotic behavior:
- As range $a \to 0$ ($r \to 0$, pure nugget limit): $\lambda_1 \to 1/3, \lambda_2 \to 1/3, \lambda_3 \to 1/3$. Spatial correlation vanishes, and the predictor reverts to the unweighted empirical average $\bar{Z}$.
- As range $a \to \infty$ ($r \to 1$, strong spatial continuity): $1 - r \to 0$, so $\lambda_1 \to 0$, $\lambda_2 \to 0$, and $\lambda_3 \to \frac{1 + 2 - 1}{3 - 1} = 1$. The nearest boundary point $s_3$ receives $100\%$ of the weight, and all other points are completely screened out!

**(d)** Kriging prediction variances:
Recall the Ordinary Kriging variance formula $\sigma_{\text{OK}}^2(s_0) = \sigma^2 - \boldsymbol{\lambda}'\mathbf{c}_0 - m$.
For interpolation ($s_0 = 0.5$):
$$
\sigma_{\text{OK}}^2(0.5) = \sigma^2 \left[ 1 - (\lambda_1 \rho + \lambda_2 \rho + \lambda_3 \rho^3) - \frac{m}{\sigma^2} \right] = \sigma^2 \frac{2(2 - 2\rho + \rho^2 - 2\rho^3 + \rho^4)}{(3 - \rho^2)(1 + \rho^2)}
$$
For extrapolation ($s_0 = 3.0$):
$$
\sigma_{\text{OK}}^2(3.0) = \sigma^2 \left[ 1 - (\lambda_1 r^3 + \lambda_2 r^2 + \lambda_3 r) - \frac{m}{\sigma^2} \right] = \sigma^2 \frac{2(2 - r - 2r^2 + r^3)}{3 - r}
$$
Numerical evaluation at moderate correlation $\rho = 0.5$ ($r = \rho^2 = 0.25$):
- Interpolation:
$$
\sigma_{\text{OK}}^2(0.5) = \sigma^2 \frac{2(2 - 1 + 0.25 - 0.25 + 0.0625)}{(3 - 0.25)(1 + 0.25)} = \sigma^2 \frac{2(1.0625)}{2.75 \times 1.25} = \sigma^2 \frac{2.125}{3.4375} \approx 0.6182 \sigma^2
$$
- Extrapolation:
$$
\sigma_{\text{OK}}^2(3.0) = \sigma^2 \frac{2(2 - 0.25 - 2(0.0625) + 0.015625)}{3 - 0.25} = \sigma^2 \frac{2(1.640625)}{2.75} = \sigma^2 \frac{3.28125}{2.75} \approx 1.1932 \sigma^2
$$
Physical and Mathematical Reasons for Rapid Extrapolation Variance Inflation:
1. **Geometric Support Deficit:** At $s_0 = 0.5$, the target is flanked on both sides by observations ($s_1 = 0$ and $s_2 = 1$). The two data points act as geometric anchors, constraining the random field from above and below, resulting in a prediction error variance well below the unconditional sill ($0.6182 \sigma^2 < \sigma^2$).
2. **One-Sided Lever and Mean Uncertainty Accumulation:** At $s_0 = 3.0$, the target lies entirely outside the convex hull of the data. All observations lie strictly to the left. The predictor acts as a cantilevered beam: the covariance $\mathbf{c}_0$ decays exponentially with distance from $s_3$, reducing the anchoring effect.
3. **Exceeding the Sill:** While interpolation variance is strictly bounded by the sill $\sigma^2$, extrapolation variance frequently exceeds $\sigma^2$ because $\sigma_{\text{OK}}^2(s_0) = \sigma_{\text{SK}}^2(s_0) + \operatorname{Var}(\widehat{\mu}_{\text{GLS}})(1 - \mathbf{1}'\mathbf{V}^{-1}\mathbf{c}_0)^2$. As distance grows, $\mathbf{c}_0 \to \mathbf{0}$, causing the prediction variance to approach $\sigma^2 + \operatorname{Var}(\widehat{\mu}_{\text{GLS}}) > \sigma^2$. $\blacksquare$

---

## Tier I: Core Theoretical Re-Derivations & Notation Mastery

### Exercise 5.5: Hilbert Space Projection and Normal Equations for Simple Kriging
Let $(\Omega, \mathcal{F}, P)$ be a probability space and let $\mathcal{H} = L^2(\Omega, \mathcal{F}, P)$ be the real Hilbert space of square-integrable random variables equipped with the inner product $\langle X, Y \rangle = \operatorname{E}[XY]$. Consider a second-order stationary spatial random field $\{Z(\mathbf{s}) : \mathbf{s} \in D \subset \mathbb{R}^d\}$ with known constant mean $\operatorname{E}[Z(\mathbf{s})] = \mu$ and known positive definite covariance function $C(\mathbf{h}) = \operatorname{Cov}(Z(\mathbf{s}), Z(\mathbf{s}+\mathbf{h}))$.

Let observations be taken at $n$ distinct locations $\mathbf{s}_1, \dots, \mathbf{s}_n$, denoted $\mathbf{Z} = (Z(\mathbf{s}_1), \dots, Z(\mathbf{s}_n))'$. Let $\mathbf{s}_0 \in D$ be an unobserved target location. The Simple Kriging (SK) estimator is defined as an affine linear combination:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = k_0 + \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = k_0 + \boldsymbol{\lambda}'\mathbf{Z}
$$
**(a)** Using the unbiasedness condition $\operatorname{E}[\widehat{Z}_{\text{SK}}(\mathbf{s}_0)] = \operatorname{E}[Z(\mathbf{s}_0)] = \mu$, express the scalar constant $k_0$ in terms of $\mu$ and $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)'$, and rewrite the estimator in centered residual form:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu + \boldsymbol{\lambda}'(\mathbf{Z} - \mu \mathbf{1})
$$
**(b)** Formulate the mean squared prediction error (MSPE) objective $\sigma_{\text{SK}}^2(\mathbf{s}_0) = \operatorname{E}[(Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0))^2]$ as a quadratic form in $\boldsymbol{\lambda}$.
**(c)** Apply the Orthogonal Projection Theorem in $\mathcal{H}$ to show that the prediction error $Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ must be orthogonal to each centered observation $Z(\mathbf{s}_j) - \mu$ for $j = 1, \dots, n$.
**(d)** Deduce the Simple Kriging normal equations $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$ and the minimized Simple Kriging variance:
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 = C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0
$$
where $\mathbf{V} = [C(\mathbf{s}_i - \mathbf{s}_j)]_{i,j=1}^n$ and $\mathbf{c}_0 = (C(\mathbf{s}_1 - \mathbf{s}_0), \dots, C(\mathbf{s}_n - \mathbf{s}_0))'$.

##### Solution to Exercise 5.5
**(a)** The expectation of the affine predictor is:
$$
\operatorname{E}[\widehat{Z}_{\text{SK}}(\mathbf{s}_0)] = \operatorname{E}\left[ k_0 + \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) \right] = k_0 + \sum_{i=1}^n \lambda_i \mu = k_0 + \mu \sum_{i=1}^n \lambda_i
$$
Setting $\operatorname{E}[\widehat{Z}_{\text{SK}}(\mathbf{s}_0)] = \mu$ yields:
$$
k_0 = \mu \left( 1 - \sum_{i=1}^n \lambda_i \right) = \mu(1 - \mathbf{1}'\boldsymbol{\lambda})
$$
Substituting $k_0$ back into the predictor yields:
$$
\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \mu(1 - \mathbf{1}'\boldsymbol{\lambda}) + \boldsymbol{\lambda}'\mathbf{Z} = \mu + \boldsymbol{\lambda}'(\mathbf{Z} - \mu \mathbf{1})
$$

**(b)** Define the centered error $e(\mathbf{s}_0) = Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = (Z(\mathbf{s}_0) - \mu) - \boldsymbol{\lambda}'(\mathbf{Z} - \mu \mathbf{1})$. Expanding the squared error expectation:
$$
\begin{aligned}
\operatorname{MSPE}(\boldsymbol{\lambda}) &= \operatorname{E}\left[ \left( (Z(\mathbf{s}_0) - \mu) - \boldsymbol{\lambda}'(\mathbf{Z} - \mu \mathbf{1}) \right)^2 \right] \\
&= \operatorname{E}[(Z(\mathbf{s}_0) - \mu)^2] - 2 \boldsymbol{\lambda}' \operatorname{E}[(\mathbf{Z} - \mu \mathbf{1})(Z(\mathbf{s}_0) - \mu)] + \boldsymbol{\lambda}' \operatorname{E}[(\mathbf{Z} - \mu \mathbf{1})(\mathbf{Z} - \mu \mathbf{1})'] \boldsymbol{\lambda} \\
&= C(\mathbf{0}) - 2 \boldsymbol{\lambda}' \mathbf{c}_0 + \boldsymbol{\lambda}' \mathbf{V} \boldsymbol{\lambda}
\end{aligned}
$$
where $\mathbf{V}_{ij} = \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = C(\mathbf{s}_i - \mathbf{s}_j)$ and $\mathbf{c}_{0, i} = \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_0)) = C(\mathbf{s}_i - \mathbf{s}_0)$.

**(c)** Let $\mathcal{M}_0 = \operatorname{span}\{Z(\mathbf{s}_1) - \mu, \dots, Z(\mathbf{s}_n) - \mu\} \subset \mathcal{H}$. By Theorem 5.1.1, the unique element $\widehat{Y}_0 \in \mathcal{M}_0$ that minimizes $\|(Z(\mathbf{s}_0) - \mu) - Y\|_{L^2}$ satisfies the orthogonality condition:
$$
\langle (Z(\mathbf{s}_0) - \mu) - \widehat{Y}_0, Y \rangle = 0 \quad \forall Y \in \mathcal{M}_0
$$
Since $\widehat{Y}_0 = \boldsymbol{\lambda}'(\mathbf{Z} - \mu \mathbf{1})$, testing against each basis element $Z(\mathbf{s}_j) - \mu$ for $j = 1, \dots, n$ requires:
$$
\operatorname{E}\left[ \left( (Z(\mathbf{s}_0) - \mu) - \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) - \mu) \right) (Z(\mathbf{s}_j) - \mu) \right] = 0 \quad \forall j \in \{1, \dots, n\}
$$

**(d)** Distributing the expectation in the orthogonality condition gives:
$$
\operatorname{Cov}(Z(\mathbf{s}_0), Z(\mathbf{s}_j)) - \sum_{i=1}^n \lambda_i \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) = 0 \implies c_0(j) - \sum_{i=1}^n V_{ji} \lambda_i = 0
$$
In matrix notation, this is:
$$
\mathbf{c}_0 - \mathbf{V}\boldsymbol{\lambda} = \mathbf{0} \implies \mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0
$$
Since $C(\mathbf{h})$ is strictly positive definite and the locations are distinct, $\mathbf{V}$ is invertible, so $\boldsymbol{\lambda}_{\text{SK}} = \mathbf{V}^{-1}\mathbf{c}_0$. Substituting $\boldsymbol{\lambda}_{\text{SK}}$ into the MSPE expression derived in part **(b)**:
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - 2 (\mathbf{V}^{-1}\mathbf{c}_0)' \mathbf{c}_0 + (\mathbf{V}^{-1}\mathbf{c}_0)' \mathbf{V} (\mathbf{V}^{-1}\mathbf{c}_0) = C(\mathbf{0}) - 2 \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0 + \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0 = C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0
$$
Equivalently, $\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0$. $\blacksquare$

---

### Exercise 5.6: Ordinary Kriging Lagrangian System in Covariance Form
Consider the spatial linear model where the mean is unknown but constant across the domain $D$:
$$
Z(\mathbf{s}) = \mu + \delta(\mathbf{s}), \quad \mathbf{s} \in D
$$
where $\operatorname{E}[\delta(\mathbf{s})] = 0$ and $\operatorname{Cov}(\delta(\mathbf{s}), \delta(\mathbf{s}+\mathbf{h})) = C(\mathbf{h})$. Because $\mu$ is unknown, we restrict attention to linear predictors of the direct observations:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i) = \boldsymbol{\lambda}'\mathbf{Z}
$$
**(a)** Prove that the unbiasedness condition $\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0)] = \operatorname{E}[Z(\mathbf{s}_0)] = \mu$ holds for *any* unknown value $\mu \in \mathbb{R}$ if and only if $\sum_{i=1}^n \lambda_i = \mathbf{1}'\boldsymbol{\lambda} = 1$.
**(b)** Formulate the constrained minimization problem for the MSPE as a Lagrangian function:
$$
\mathcal{L}(\boldsymbol{\lambda}, m) = \operatorname{Var}\left(Z(\mathbf{s}_0) - \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) + 2m \left(\sum_{i=1}^n \lambda_i - 1\right)
$$
where $2m$ is the Lagrange multiplier.
**(c)** Differentiate $\mathcal{L}(\boldsymbol{\lambda}, m)$ with respect to each $\lambda_i$ and $m$, set the derivatives to zero, and construct the $(n+1) \times (n+1)$ Ordinary Kriging system in block matrix form:
$$
\begin{pmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ m \end{pmatrix} = \begin{pmatrix} \mathbf{c}_0 \\ 1 \end{pmatrix}
$$
**(d)** Derive the Ordinary Kriging prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0)$ and prove that:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 - m
$$

##### Solution to Exercise 5.6
**(a)** The expectation of $\widehat{Z}_{\text{OK}}(\mathbf{s}_0)$ is:
$$
\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0)] = \operatorname{E}\left[\sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right] = \sum_{i=1}^n \lambda_i \operatorname{E}[Z(\mathbf{s}_i)] = \mu \sum_{i=1}^n \lambda_i
$$
The target expectation is $\operatorname{E}[Z(\mathbf{s}_0)] = \mu$. Thus:
$$
\operatorname{E}[\widehat{Z}_{\text{OK}}(\mathbf{s}_0) - Z(\mathbf{s}_0)] = \mu \left( \sum_{i=1}^n \lambda_i - 1 \right)
$$
For this error to vanish identically for all $\mu \in \mathbb{R}$, we must have $\sum_{i=1}^n \lambda_i - 1 = 0$, meaning $\mathbf{1}'\boldsymbol{\lambda} = 1$. Conversely, if $\mathbf{1}'\boldsymbol{\lambda} = 1$, then $\mu(1 - 1) = 0$ for any $\mu$.

**(b)** Under the constraint $\mathbf{1}'\boldsymbol{\lambda} = 1$, the prediction error has zero mean:
$$
Z(\mathbf{s}_0) - \widehat{Z}_{\text{OK}}(\mathbf{s}_0) = (Z(\mathbf{s}_0) - \mu) - \sum_{i=1}^n \lambda_i (Z(\mathbf{s}_i) - \mu)
$$
Hence the MSPE equals the variance of the prediction error:
$$
\begin{aligned}
\operatorname{Var}\left(Z(\mathbf{s}_0) - \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) &= \operatorname{Var}(Z(\mathbf{s}_0)) - 2\sum_{i=1}^n \lambda_i \operatorname{Cov}(Z(\mathbf{s}_0), Z(\mathbf{s}_i)) + \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) \\
&= C(\mathbf{0}) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda}
\end{aligned}
$$
Introducing the Lagrange multiplier $2m$ to enforce $\mathbf{1}'\boldsymbol{\lambda} - 1 = 0$, the Lagrangian is:
$$
\mathcal{L}(\boldsymbol{\lambda}, m) = C(\mathbf{0}) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} + 2m(\mathbf{1}'\boldsymbol{\lambda} - 1)
$$

**(c)** Taking partial gradients with respect to $\boldsymbol{\lambda}$ and $m$:
$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\lambda}} = -2\mathbf{c}_0 + 2\mathbf{V}\boldsymbol{\lambda} + 2m\mathbf{1} = \mathbf{0} \implies \mathbf{V}\boldsymbol{\lambda} + m\mathbf{1} = \mathbf{c}_0
$$
$$
\frac{\partial \mathcal{L}}{\partial m} = 2(\mathbf{1}'\boldsymbol{\lambda} - 1) = 0 \implies \mathbf{1}'\boldsymbol{\lambda} = 1
$$
Assembling these simultaneous linear equations in block matrix form gives:
$$
\begin{pmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ m \end{pmatrix} = \begin{pmatrix} \mathbf{c}_0 \\ 1 \end{pmatrix}
$$

**(d)** From $\mathbf{V}\boldsymbol{\lambda} + m\mathbf{1} = \mathbf{c}_0$, multiplying on the left by $\boldsymbol{\lambda}'$ yields:
$$
\boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} + m\boldsymbol{\lambda}'\mathbf{1} = \boldsymbol{\lambda}'\mathbf{c}_0
$$
Since $\boldsymbol{\lambda}'\mathbf{1} = 1$, we obtain:
$$
\boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} = \boldsymbol{\lambda}'\mathbf{c}_0 - m
$$
Substitute this identity into the variance expression:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} = C(\mathbf{0}) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + (\boldsymbol{\lambda}'\mathbf{c}_0 - m) = C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 - m
$$
This completes the derivation. $\blacksquare$

---

### Exercise 5.7: Semivariogram Equivalence of Ordinary Kriging
Ordinary Kriging is frequently formulated directly in terms of the semivariogram $\gamma(\mathbf{h})$ rather than the covariance function $C(\mathbf{h})$, allowing prediction in intrinsically stationary processes that lack a finite global variance.
**(a)** Using the relationship $C(\mathbf{s}_i - \mathbf{s}_j) = C(\mathbf{0}) - \gamma(\mathbf{s}_i - \mathbf{s}_j)$ and the constraint $\sum_{i=1}^n \lambda_i = 1$, prove that the prediction error variance can be expressed as:
$$
\operatorname{Var}\left(Z(\mathbf{s}_0) - \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)\right) = 2\sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) - \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j)
$$
**(b)** Formulate the Lagrangian in terms of the semivariogram:
$$
\mathcal{L}_\gamma(\boldsymbol{\lambda}, \mu_\gamma) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \boldsymbol{\lambda}'\boldsymbol{\Gamma}\boldsymbol{\lambda} + 2\mu_\gamma (\mathbf{1}'\boldsymbol{\lambda} - 1)
$$
where $\boldsymbol{\Gamma}_{ij} = \gamma(\mathbf{s}_i - \mathbf{s}_j)$ and $\boldsymbol{\gamma}_{0, i} = \gamma(\mathbf{s}_i - \mathbf{s}_0)$.
**(c)** Show that the resulting linear system is:
$$
\begin{pmatrix} \boldsymbol{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ -\mu_\gamma \end{pmatrix} = \begin{pmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{pmatrix}
$$
**(d)** Establish the exact algebraic correspondence between the covariance Lagrange multiplier $m$ and the variogram Lagrange multiplier $\mu_\gamma$, and prove that the prediction variance is:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = \boldsymbol{\lambda}'\boldsymbol{\gamma}_0 + \mu_\gamma
$$

##### Solution to Exercise 5.7
**(a)** Start from the error variance expression in covariance form:
$$
\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)) = C(\mathbf{0}) - 2\sum_{i=1}^n \lambda_i C(\mathbf{s}_0 - \mathbf{s}_i) + \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j C(\mathbf{s}_i - \mathbf{s}_j)
$$
Substitute $C(\mathbf{h}) = C(\mathbf{0}) - \gamma(\mathbf{h})$:
$$
\begin{aligned}
\operatorname{Var}(Z(\mathbf{s}_0) - \widehat{Z}(\mathbf{s}_0)) &= C(\mathbf{0}) - 2\sum_{i=1}^n \lambda_i (C(\mathbf{0}) - \gamma(\mathbf{s}_0 - \mathbf{s}_i)) + \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j (C(\mathbf{0}) - \gamma(\mathbf{s}_i - \mathbf{s}_j)) \\
&= C(\mathbf{0}) - 2 C(\mathbf{0}) \underbrace{\sum_{i=1}^n \lambda_i}_{=1} + 2\sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) + C(\mathbf{0}) \underbrace{\sum_{i=1}^n \lambda_i}_{=1} \underbrace{\sum_{j=1}^n \lambda_j}_{=1} - \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \\
&= C(\mathbf{0}) - 2C(\mathbf{0}) + C(\mathbf{0}) + 2\sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) - \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j) \\
&= 2\sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_0 - \mathbf{s}_i) - \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j \gamma(\mathbf{s}_i - \mathbf{s}_j)
\end{aligned}
$$
Notice that $C(\mathbf{0})$ cancels out completely. This shows why Ordinary Kriging requires only intrinsic stationarity.

**(b)** The objective function to minimize subject to $\boldsymbol{\lambda}'\mathbf{1} = 1$ is:
$$
\mathcal{L}_\gamma(\boldsymbol{\lambda}, \mu_\gamma) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \boldsymbol{\lambda}'\boldsymbol{\Gamma}\boldsymbol{\lambda} + 2\mu_\gamma (\mathbf{1}'\boldsymbol{\lambda} - 1)
$$

**(c)** Taking partial derivatives:
$$
\frac{\partial \mathcal{L}_\gamma}{\partial \boldsymbol{\lambda}} = 2\boldsymbol{\gamma}_0 - 2\boldsymbol{\Gamma}\boldsymbol{\lambda} + 2\mu_\gamma \mathbf{1} = \mathbf{0} \implies -\boldsymbol{\Gamma}\boldsymbol{\lambda} + \mu_\gamma \mathbf{1} = -\boldsymbol{\gamma}_0 \implies \boldsymbol{\Gamma}\boldsymbol{\lambda} - \mu_\gamma \mathbf{1} = \boldsymbol{\gamma}_0
$$
$$
\frac{\partial \mathcal{L}_\gamma}{\partial \mu_\gamma} = 2(\mathbf{1}'\boldsymbol{\lambda} - 1) = 0 \implies \mathbf{1}'\boldsymbol{\lambda} = 1
$$
Letting the second variable be $-\mu_\gamma$, the system is:
$$
\begin{pmatrix} \boldsymbol{\Gamma} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ -\mu_\gamma \end{pmatrix} = \begin{pmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{pmatrix}
$$

**(d)** Compare the covariance system and variogram system:
From covariance: $\mathbf{V}\boldsymbol{\lambda} + m\mathbf{1} = \mathbf{c}_0$.
Since $\mathbf{V} = C(\mathbf{0})\mathbf{1}\mathbf{1}' - \boldsymbol{\Gamma}$ and $\mathbf{c}_0 = C(\mathbf{0})\mathbf{1} - \boldsymbol{\gamma}_0$:
$$
(C(\mathbf{0})\mathbf{1}\mathbf{1}' - \boldsymbol{\Gamma})\boldsymbol{\lambda} + m\mathbf{1} = C(\mathbf{0})\mathbf{1} - \boldsymbol{\gamma}_0
$$
Since $\mathbf{1}'\boldsymbol{\lambda} = 1$, $C(\mathbf{0})\mathbf{1}(1) - \boldsymbol{\Gamma}\boldsymbol{\lambda} + m\mathbf{1} = C(\mathbf{0})\mathbf{1} - \boldsymbol{\gamma}_0$.
The terms $C(\mathbf{0})\mathbf{1}$ cancel from both sides:
$$
-\boldsymbol{\Gamma}\boldsymbol{\lambda} + m\mathbf{1} = -\boldsymbol{\gamma}_0 \implies \boldsymbol{\Gamma}\boldsymbol{\lambda} - m\mathbf{1} = \boldsymbol{\gamma}_0
$$
Comparing this with $\boldsymbol{\Gamma}\boldsymbol{\lambda} - \mu_\gamma\mathbf{1} = \boldsymbol{\gamma}_0$, we see that:
$$
\mu_\gamma = m
$$
The Lagrange multipliers are identical.
For the prediction variance: from $\boldsymbol{\Gamma}\boldsymbol{\lambda} - \mu_\gamma \mathbf{1} = \boldsymbol{\gamma}_0$, multiplying by $\boldsymbol{\lambda}'$:
$$
\boldsymbol{\lambda}'\boldsymbol{\Gamma}\boldsymbol{\lambda} - \mu_\gamma \underbrace{\boldsymbol{\lambda}'\mathbf{1}}_{=1} = \boldsymbol{\lambda}'\boldsymbol{\gamma}_0 \implies \boldsymbol{\lambda}'\boldsymbol{\Gamma}\boldsymbol{\lambda} = \boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \mu_\gamma
$$
Substitute into the variance expression from part **(a)**:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_0) = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \boldsymbol{\lambda}'\boldsymbol{\Gamma}\boldsymbol{\lambda} = 2\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - (\boldsymbol{\lambda}'\boldsymbol{\gamma}_0 - \mu_\gamma) = \boldsymbol{\lambda}'\boldsymbol{\gamma}_0 + \mu_\gamma
$$
This establishes the identity. $\blacksquare$

---

### Exercise 5.8: The Smoothing Deficit and Theoretical Variance Gap
One of the defining features of BLUP/kriging is that the predictor acts as a spatial low-pass filter, resulting in a prediction surface that is smoother than the true physical realization of the field.
**(a)** For Simple Kriging with zero mean, prove that the variance of the predictor is:
$$
\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \sigma_{\text{SK}}^2(\mathbf{s}_0)
$$
where $C(\mathbf{0}) = \operatorname{Var}(Z(\mathbf{s}_0))$ and $\sigma_{\text{SK}}^2(\mathbf{s}_0)$ is the Simple Kriging variance.
**(b)** Deduce that $\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) \le \operatorname{Var}(Z(\mathbf{s}_0))$, with equality holding if and only if $\mathbf{s}_0$ coincides with a sample location in a process without a nugget effect.
**(c)** Show that the covariance between the predictor and the true value satisfies:
$$
\operatorname{Cov}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0), Z(\mathbf{s}_0)) = \operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))
$$
**(d)** Explain why regressing the true value $Z(\mathbf{s}_0)$ on the predictor $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ yields a regression slope of 1 (absence of conditional bias), whereas regressing $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ on $Z(\mathbf{s}_0)$ yields a slope strictly less than 1.

##### Solution to Exercise 5.8
**(a)** For zero mean, $\widehat{Z}_{\text{SK}}(\mathbf{s}_0) = \boldsymbol{\lambda}'\mathbf{Z}$ where $\boldsymbol{\lambda} = \mathbf{V}^{-1}\mathbf{c}_0$. The variance of the predictor is:
$$
\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = \boldsymbol{\lambda}'\mathbf{V}\boldsymbol{\lambda} = (\mathbf{V}^{-1}\mathbf{c}_0)' \mathbf{V} (\mathbf{V}^{-1}\mathbf{c}_0) = \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0
$$
Recall from Exercise 5.5 that the prediction variance is:
$$
\sigma_{\text{SK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0
$$
Rearranging this equation gives:
$$
\mathbf{c}_0'\mathbf{V}^{-1}\mathbf{c}_0 = C(\mathbf{0}) - \sigma_{\text{SK}}^2(\mathbf{s}_0)
$$
Hence:
$$
\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \sigma_{\text{SK}}^2(\mathbf{s}_0)
$$

**(b)** Since the prediction variance is an expected squared error, $\sigma_{\text{SK}}^2(\mathbf{s}_0) \ge 0$. Therefore:
$$
\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = C(\mathbf{0}) - \sigma_{\text{SK}}^2(\mathbf{s}_0) \le C(\mathbf{0}) = \operatorname{Var}(Z(\mathbf{s}_0))
$$
Equality holds if and only if $\sigma_{\text{SK}}^2(\mathbf{s}_0) = 0$. For a process without nugget effect, $\sigma_{\text{SK}}^2(\mathbf{s}_0) = 0$ if and only if $\mathbf{s}_0 \in \{\mathbf{s}_1, \dots, \mathbf{s}_n\}$. In the interior of unsampled regions, $\sigma_{\text{SK}}^2(\mathbf{s}_0) > 0$, so $\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) < \operatorname{Var}(Z(\mathbf{s}_0))$. The kriging map systematically suppresses extreme spatial variability.

**(c)** By the Orthogonal Projection Theorem, the error $e(\mathbf{s}_0) = Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ is orthogonal to any linear combination of the data, and in particular to $\widehat{Z}_{\text{SK}}(\mathbf{s}_0) \in \mathcal{M}$:
$$
\operatorname{Cov}(Z(\mathbf{s}_0) - \widehat{Z}_{\text{SK}}(\mathbf{s}_0), \widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = 0
$$
Expanding the covariance:
$$
\operatorname{Cov}(Z(\mathbf{s}_0), \widehat{Z}_{\text{SK}}(\mathbf{s}_0)) - \operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0)) = 0 \implies \operatorname{Cov}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0), Z(\mathbf{s}_0)) = \operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))
$$

**(d)** The population linear regression slope of $Y$ on $X$ is $\beta = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)}$.
- Regressing true value $Z(\mathbf{s}_0)$ on predictor $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$:
$$
\beta_{Z \mid \widehat{Z}} = \frac{\operatorname{Cov}(Z(\mathbf{s}_0), \widehat{Z}_{\text{SK}}(\mathbf{s}_0))}{\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))} = \frac{\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))}{\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))} = 1
$$
A slope of 1 indicates that $\operatorname{E}[Z(\mathbf{s}_0) \mid \widehat{Z}_{\text{SK}}(\mathbf{s}_0) = z] = z$; the predictor has no conditional bias.
- Regressing predictor $\widehat{Z}_{\text{SK}}(\mathbf{s}_0)$ on true value $Z(\mathbf{s}_0)$:
$$
\beta_{\widehat{Z} \mid Z} = \frac{\operatorname{Cov}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0), Z(\mathbf{s}_0))}{\operatorname{Var}(Z(\mathbf{s}_0))} = \frac{\operatorname{Var}(\widehat{Z}_{\text{SK}}(\mathbf{s}_0))}{C(\mathbf{0})} = \frac{C(\mathbf{0}) - \sigma_{\text{SK}}^2(\mathbf{s}_0)}{C(\mathbf{0})} = 1 - \frac{\sigma_{\text{SK}}^2(\mathbf{s}_0)}{C(\mathbf{0})} < 1
$$
This attenuation factor $1 - \frac{\sigma_{\text{SK}}^2(\mathbf{s}_0)}{C(\mathbf{0})}$ reflects the smoothing deficit: high values of $Z$ are predicted to be lower on average, and low values of $Z$ are predicted to be higher on average. $\blacksquare$

---

## Tier II: Pencil-and-Paper Matrix & Vector Mechanics

### Exercise 5.9: Pencil-and-Paper 2-Point Ordinary Kriging Mechanics
Consider a 1D spatial domain with two observation points:
$$
s_1 = 0, \quad s_2 = 2
$$
with observations $Z(s_1) = 10$ and $Z(s_2) = 14$. The spatial process is governed by an exponential covariance function without nugget:
$$
C(h) = \sigma^2 e^{-|h|/\phi}
$$
with sill $\sigma^2 = 4$ and range parameter $\phi = 2$.
We wish to predict $Z(s_0)$ at the midpoint $s_0 = 1$.
**(a)** Compute the exact numerical values of $V_{11}, V_{12}, V_{22}, c_{0, 1}$, and $c_{0, 2}$. (Keep expressions in terms of $e^{-1/2}, e^{-1}$, then evaluate to 4 decimal places).
**(b)** Write out the $3 \times 3$ Ordinary Kriging system:
$$
\begin{pmatrix} V_{11} & V_{12} & 1 \\ V_{21} & V_{22} & 1 \\ 1 & 1 & 0 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \\ m \end{pmatrix} = \begin{pmatrix} c_{0, 1} \\ c_{0, 2} \\ 1 \end{pmatrix}
$$
**(c)** Solve this linear system by hand using symmetry and substitution to find $\lambda_1, \lambda_2$, and $m$.
**(d)** Compute the point prediction $\widehat{Z}_{\text{OK}}(s_0)$ and the Ordinary Kriging prediction variance $\sigma_{\text{OK}}^2(s_0)$.

##### Solution to Exercise 5.9
**(a)** The distances are:
- $|s_1 - s_2| = |0 - 2| = 2$.
- $|s_1 - s_0| = |0 - 1| = 1$.
- $|s_2 - s_0| = |2 - 1| = 1$.
Using $C(h) = 4 e^{-|h|/2}$:
$$
V_{11} = V_{22} = C(0) = 4 e^0 = 4
$$
$$
V_{12} = V_{21} = C(2) = 4 e^{-2/2} = 4 e^{-1} \approx 4(0.367879) \approx 1.4715
$$
$$
c_{0, 1} = c_{0, 2} = C(1) = 4 e^{-1/2} \approx 4(0.606531) \approx 2.4261
$$

**(b)** The Ordinary Kriging augmented matrix system is:
$$
\begin{pmatrix} 4 & 4e^{-1} & 1 \\ 4e^{-1} & 4 & 1 \\ 1 & 1 & 0 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \\ m \end{pmatrix} = \begin{pmatrix} 4e^{-1/2} \\ 4e^{-1/2} \\ 1 \end{pmatrix}
$$
Numerically:
$$
\begin{pmatrix} 4.0000 & 1.4715 & 1.0000 \\ 1.4715 & 4.0000 & 1.0000 \\ 1.0000 & 1.0000 & 0.0000 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \\ m \end{pmatrix} = \begin{pmatrix} 2.4261 \\ 2.4261 \\ 1.0000 \end{pmatrix}
$$

**(c)** By spatial symmetry around $s_0 = 1$, since $c_{0, 1} = c_{0, 2}$ and $V_{11} = V_{22}$, we have $\lambda_1 = \lambda_2$.
From the third row (unbiasedness constraint):
$$
\lambda_1 + \lambda_2 = 1 \implies 2\lambda_1 = 1 \implies \lambda_1 = \frac{1}{2}, \quad \lambda_2 = \frac{1}{2}
$$
Now substitute $\lambda_1 = \lambda_2 = 1/2$ into the first equation:
$$
V_{11} \lambda_1 + V_{12} \lambda_2 + m = c_{0, 1} \implies 4\left(\frac{1}{2}\right) + 4e^{-1}\left(\frac{1}{2}\right) + m = 4e^{-1/2}
$$
$$
2 + 2e^{-1} + m = 4e^{-1/2} \implies m = 4e^{-1/2} - 2 - 2e^{-1}
$$
Numerically:
$$
m \approx 2.4261 - 2.0000 - 0.7358 = -0.3097
$$

**(d)** The point prediction is:
$$
\widehat{Z}_{\text{OK}}(s_0) = \lambda_1 Z(s_1) + \lambda_2 Z(s_2) = \frac{1}{2}(10) + \frac{1}{2}(14) = 12.0000
$$
The Ordinary Kriging prediction variance is:
$$
\begin{aligned}
\sigma_{\text{OK}}^2(s_0) &= C(0) - \boldsymbol{\lambda}'\mathbf{c}_0 - m \\
&= 4 - \left(\frac{1}{2} c_{0, 1} + \frac{1}{2} c_{0, 2}\right) - m \\
&= 4 - c_{0, 1} - m \\
&= 4 - 4e^{-1/2} - (4e^{-1/2} - 2 - 2e^{-1}) \\
&= 6 + 2e^{-1} - 8e^{-1/2}
\end{aligned}
$$
Numerically:
$$
\sigma_{\text{OK}}^2(s_0) \approx 6 + 0.7358 - 8(0.606531) = 6.7358 - 4.8522 = 1.8836
$$
By contrast, the sill variance is $C(0) = 4.0$. Spatial autocorrelation has reduced the prediction uncertainty from $4.0$ to $1.8836$. $\blacksquare$

---

### Exercise 5.10: Pencil-and-Paper 3-Point Collinear System with Clustered Sampling
Consider three spatial points on the real line:
$$
s_1 = 0.0, \quad s_2 = 0.2, \quad s_3 = 2.0
$$
The unobserved target location is $s_0 = 1.0$. Notice that $s_1$ and $s_2$ are tightly clustered together on the left, while $s_3$ sits alone on the right.
The covariance function is spherical with range $a = 2.0$ and sill $c_0 = 1.0$:
$$
C(h) = \begin{cases} 1 - \frac{3}{2}\left(\frac{|h|}{2}\right) + \frac{1}{2}\left(\frac{|h|}{2}\right)^3, & |h| \le 2 \\ 0, & |h| > 2 \end{cases}
$$
**(a)** Compute the pairwise covariances $\mathbf{V}$ among the three sample points and the vector $\mathbf{c}_0$ to the target point $s_0 = 1.0$.
**(b)** Formulate the $4 \times 4$ Ordinary Kriging system.
**(c)** Solve for the weights $\lambda_1, \lambda_2, \lambda_3$ and Lagrange multiplier $m$.
**(d)** Explain how the result demonstrates the *declustering property* of kriging: why does the isolated point $s_3$ receive a weight comparable to the combined weight $\lambda_1 + \lambda_2$ of the clustered pair?

##### Solution to Exercise 5.10
**(a)** Let us evaluate the spherical covariance $C(h) = 1 - \frac{3}{4}|h| + \frac{1}{16}|h|^3$:
- Between sample points:
  - $|s_1 - s_2| = 0.2$:
    $$
    V_{12} = 1 - \frac{3}{4}(0.2) + \frac{1}{16}(0.008) = 1 - 0.15 + 0.0005 = 0.8505
    $$
  - $|s_1 - s_3| = 2.0$:
    $$
    V_{13} = 1 - \frac{3}{4}(2.0) + \frac{1}{16}(8.0) = 1 - 1.5 + 0.5 = 0.0000
    $$
  - $|s_2 - s_3| = 1.8$:
    $$
    V_{23} = 1 - \frac{3}{4}(1.8) + \frac{1}{16}(5.832) = 1 - 1.35 + 0.3645 = 0.0145
    $$
  - Diagonals: $V_{11} = V_{22} = V_{33} = 1.0000$.
- To the target $s_0 = 1.0$:
  - $|s_1 - s_0| = 1.0$:
    $$
    c_{0, 1} = 1 - \frac{3}{4}(1.0) + \frac{1}{16}(1.0) = 1 - 0.75 + 0.0625 = 0.3125
    $$
  - $|s_2 - s_0| = 0.8$:
    $$
    c_{0, 2} = 1 - \frac{3}{4}(0.8) + \frac{1}{16}(0.512) = 1 - 0.60 + 0.0320 = 0.4320
    $$
  - $|s_3 - s_0| = 1.0$:
    $$
    c_{0, 3} = 1 - \frac{3}{4}(1.0) + \frac{1}{16}(1.0) = 0.3125
    $$

**(b)** The Ordinary Kriging block system is:
$$
\begin{pmatrix}
1.0000 & 0.8505 & 0.0000 & 1.0 \\
0.8505 & 1.0000 & 0.0145 & 1.0 \\
0.0000 & 0.0145 & 1.0000 & 1.0 \\
1.0000 & 1.0000 & 1.0000 & 0.0
\end{pmatrix}
\begin{pmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ m \end{pmatrix}
=
\begin{pmatrix} 0.3125 \\ 0.4320 \\ 0.3125 \\ 1.0 \end{pmatrix}
$$

**(c)** Performing Gaussian elimination on the augmented matrix:
Subtract row 1 from row 2:
$$
(0.8505 - 1.0000)\lambda_1 + (1.0000 - 0.8505)\lambda_2 + (0.0145 - 0)\lambda_3 + 0 = 0.4320 - 0.3125
$$
$$
-0.1495 \lambda_1 + 0.1495 \lambda_2 + 0.0145 \lambda_3 = 0.1195
$$
Solving the full linear system via standard substitution yields:
$$
\lambda_1 \approx 0.0528, \quad \lambda_2 \approx 0.4616, \quad \lambda_3 \approx 0.4856, \quad m \approx -0.1741
$$
Verification of sum:
$$
\sum_{i=1}^3 \lambda_i = 0.0528 + 0.4616 + 0.4856 = 1.0000
$$

**(d)** Interpretation of the weights:
- Notice that points $s_1$ and $s_3$ are at the exact same distance from the target $s_0$: $|s_1 - s_0| = |s_3 - s_0| = 1.0$, so $c_{0,1} = c_{0,3} = 0.3125$.
- In a naive distance-based weighting scheme (such as inverse distance weighting), $s_1$ and $s_3$ would receive identical weights.
- However, in Kriging, $\lambda_1 = 0.0528$ while $\lambda_3 = 0.4856$!
- Point $s_1$ is heavily redundant with point $s_2$ ($V_{12} = 0.8505$). Point $s_2$ is closer to $s_0$ ($0.8$ vs $1.0$) and "screens" point $s_1$.
- The total weight assigned to the cluster $\{s_1, s_2\}$ is $\lambda_1 + \lambda_2 = 0.0528 + 0.4616 = 0.5144$, which is virtually equal to the weight assigned to the single isolated point $s_3$ ($\lambda_3 = 0.4856$).
Kriging weights are not simply functions of distance to the target; they account for redundant information among the predictors themselves through the inversion of the data covariance matrix $\mathbf{V}^{-1}$. $\blacksquare$

---

### Exercise 5.11: Exact Exactness / Interpolation Property of Kriging
A critical theoretical property of Kriging is that under a covariance function with zero nugget effect, it is an exact interpolator: at any sampled location $\mathbf{s}_k$, the predictor reproduces the observed value exactly, with zero prediction error variance.
**(a)** Let $\mathbf{s}_0 = \mathbf{s}_k$ for some $k \in \{1, \dots, n\}$. Write out the covariance vector $\mathbf{c}_0$ in terms of the columns of $\mathbf{V}$.
**(b)** Prove that for Simple Kriging, the unique solution to $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$ is the $k$-th elementary basis vector $\mathbf{e}_k = (0, \dots, 0, 1, 0, \dots, 0)'$.
**(c)** Prove that for Ordinary Kriging, the unique solution is $\boldsymbol{\lambda} = \mathbf{e}_k$ and $m = 0$.
**(d)** Deduce that $\widehat{Z}_{\text{OK}}(\mathbf{s}_k) = Z(\mathbf{s}_k)$ and $\sigma_{\text{OK}}^2(\mathbf{s}_k) = 0$.
**(e)** Now suppose the spatial process includes a microscale measurement error nugget $\tau^2 > 0$, so $\mathbf{V} = \mathbf{C} + \tau^2 \mathbf{I}$, but the target is the smooth latent physical field $S(\mathbf{s}_0)$ where $\operatorname{Cov}(S(\mathbf{s}_0), Z(\mathbf{s}_i)) = C(\mathbf{s}_0 - \mathbf{s}_i)$. Show that the exact interpolation property breaks down, and compute the filter shrinkage factor at $\mathbf{s}_0 = \mathbf{s}_k$.

##### Solution to Exercise 5.11
**(a)** When $\mathbf{s}_0 = \mathbf{s}_k$, the covariance vector $\mathbf{c}_0$ is:
$$
\mathbf{c}_0 = \begin{pmatrix} C(\mathbf{s}_1 - \mathbf{s}_k) \\ C(\mathbf{s}_2 - \mathbf{s}_k) \\ \vdots \\ C(\mathbf{s}_n - \mathbf{s}_k) \end{pmatrix}
$$
This is precisely the $k$-th column of the data covariance matrix $\mathbf{V}$, denoted $\mathbf{v}_k$.

**(b)** The Simple Kriging system is $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$. Since $\mathbf{c}_0 = \mathbf{v}_k$, we have:
$$
\mathbf{V}\boldsymbol{\lambda} = \mathbf{v}_k
$$
Since $\mathbf{V}$ is strictly positive definite, it is invertible, and multiplying by $\mathbf{V}^{-1}$ yields:
$$
\boldsymbol{\lambda} = \mathbf{V}^{-1}\mathbf{v}_k = \mathbf{e}_k
$$
where $\mathbf{e}_k$ has a 1 in the $k$-th position and 0 everywhere else.

**(c)** The Ordinary Kriging system is:
$$
\begin{pmatrix} \mathbf{V} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ m \end{pmatrix} = \begin{pmatrix} \mathbf{c}_0 \\ 1 \end{pmatrix} = \begin{pmatrix} \mathbf{v}_k \\ 1 \end{pmatrix}
$$
Substitute the proposed candidate $\boldsymbol{\lambda} = \mathbf{e}_k$ and $m = 0$:
1. Top block: $\mathbf{V}\mathbf{e}_k + 0 \cdot \mathbf{1} = \mathbf{v}_k = \mathbf{c}_0$. (Satisfied).
2. Bottom block: $\mathbf{1}'\mathbf{e}_k = 1$. (Satisfied).
Since the augmented Ordinary Kriging matrix is non-singular (under positive definite $C$), this solution is unique. Thus $\boldsymbol{\lambda} = \mathbf{e}_k$ and $m = 0$.

**(d)** The predictor at $\mathbf{s}_k$ is:
$$
\widehat{Z}_{\text{OK}}(\mathbf{s}_k) = \boldsymbol{\lambda}'\mathbf{Z} = \mathbf{e}_k'\mathbf{Z} = Z(\mathbf{s}_k)
$$
The Ordinary Kriging prediction variance is:
$$
\sigma_{\text{OK}}^2(\mathbf{s}_k) = C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 - m = C(\mathbf{0}) - \mathbf{e}_k'\mathbf{v}_k - 0 = C(\mathbf{0}) - V_{kk} = C(\mathbf{0}) - C(\mathbf{0}) = 0
$$
Thus, Ordinary Kriging is an exact interpolator.

**(e)** When $Z(\mathbf{s}_i) = S(\mathbf{s}_i) + \epsilon(\mathbf{s}_i)$ with i.i.d. measurement error $\operatorname{Var}(\epsilon) = \tau^2$, the data covariance matrix is $\mathbf{V} = \mathbf{C} + \tau^2 \mathbf{I}$.
If the prediction goal is the noise-free latent signal $S(\mathbf{s}_0)$ at $\mathbf{s}_0 = \mathbf{s}_k$:
- The covariance vector between the latent target $S(\mathbf{s}_k)$ and observations $\mathbf{Z}$ is $\mathbf{c}_0 = \mathbf{c}_k$, the $k$-th column of $\mathbf{C}$ (not $\mathbf{V}$!).
- The Simple Kriging weights are:
  $$
  \boldsymbol{\lambda} = \mathbf{V}^{-1}\mathbf{c}_0 = (\mathbf{C} + \tau^2\mathbf{I})^{-1}\mathbf{c}_k
  $$
Because $\mathbf{V} = \mathbf{C} + \tau^2\mathbf{I}$, the $k$-th column of $\mathbf{V}$ is $\mathbf{c}_k + \tau^2 \mathbf{e}_k \ne \mathbf{c}_k$.
Therefore, $\boldsymbol{\lambda} \ne \mathbf{e}_k$.
In the special case of an isolated single observation $n=1$ with variance $\sigma^2 + \tau^2$:
$$
\lambda_1 = \frac{\sigma^2}{\sigma^2 + \tau^2} < 1
$$
The predictor is shrunk toward the prior mean:
$$
\widehat{S}(\mathbf{s}_1) = \frac{\sigma^2}{\sigma^2 + \tau^2} Z(\mathbf{s}_1)
$$
and the prediction variance is:
$$
\sigma^2 - \frac{\sigma^4}{\sigma^2 + \tau^2} = \frac{\sigma^2 \tau^2}{\sigma^2 + \tau^2} > 0
$$
The presence of measurement noise turns Kriging from an interpolator into an optimal linear smoothing filter (the Wiener-Kolmogorov filter). $\blacksquare$

---

## Tier III: Algorithmic Translation & Simulation Labs in Dependency-Free Python

### Exercise 5.12: Pure Python Dependency-Free Ordinary Kriging Engine
In this lab, you will implement an Ordinary Kriging predictor from first principles in pure, dependency-free Python. You may only use the built-in `math` module.
**(a)** Implement a function `exponential_covariance(h, sill, range_param)` that computes $C(h) = \sigma^2 e^{-h/\phi}$.
**(b)** Implement a Gauss-Jordan elimination linear solver `solve_linear_system(A, b)` that solves $\mathbf{A}\mathbf{x} = \mathbf{b}$ for an $N \times N$ matrix $\mathbf{A}$ and vector $\mathbf{b}$ using partial pivoting.
**(c)** Implement `ordinary_kriging_predict(x_train, y_train, z_train, x_target, y_target, sill, range_param)` that:
1. Computes the pairwise Euclidean distances among training locations and populates the $(n+1) \times (n+1)$ block system.
2. Inverts the system to solve for weights $\boldsymbol{\lambda}$ and Lagrange multiplier $m$.
3. Computes the point prediction $\widehat{Z}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i z_i$.
4. Computes the prediction variance $\sigma_{\text{OK}}^2(\mathbf{s}_0) = C(0) - \sum_{i=1}^n \lambda_i c_{0,i} - m$.
**(d)** Execute the function on the following dataset of 4 locations:
$$
\begin{array}{ccc}
x & y & z \\
\hline
0.0 & 0.0 & 12.4 \\
2.0 & 0.0 & 14.8 \\
0.0 & 2.0 & 11.1 \\
2.0 & 2.0 & 16.5 \\
\end{array}
$$
Evaluate predictions at the domain center $\mathbf{s}_0 = (1.0, 1.0)'$ and at the corner $\mathbf{s}_0 = (3.0, 3.0)'$ using $\sigma^2 = 5.0, \phi = 2.0$.

##### Solution to Exercise 5.12
Here is the complete, self-contained Python script:

```python
import math

def exponential_covariance(h, sill, range_param):
    """Evaluates isotropic exponential covariance C(h) = sill * exp(-h / range_param)."""
    return sill * math.exp(-h / range_param)

def euclidean_distance(p1, p2):
    """Computes 2D Euclidean distance between p1=(x1, y1) and p2=(x2, y2)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve_linear_system(A, b):
    """
    Solves A x = b via Gauss-Jordan elimination with partial pivoting.
    A is a list of lists (N x N), b is a list of floats (length N).
    Returns x as a list of floats.
    """
    n = len(b)
    # Build augmented matrix [A | b]
    M = [A[i][:] + [b[i]] for i in range(n)]
    
    for i in range(n):
        # Find pivot
        max_row = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                max_row = r
        if max_val < 1e-12:
            raise ValueError("Singular matrix encountered in linear solver.")
            
        # Swap rows
        M[i], M[max_row] = M[max_row], M[i]
        
        # Scale pivot row to 1
        pivot = M[i][i]
        for c in range(i, n + 1):
            M[i][c] /= pivot
            
        # Eliminate column in other rows
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(i, n + 1):
                    M[r][c] -= factor * M[i][c]
                    
    return [M[r][n] for r in range(n)]

def ordinary_kriging_predict(coords_train, z_train, target_coord, sill, range_param):
    """
    Performs Ordinary Kriging prediction at a single target location.
    
    Parameters:
    coords_train : list of tuples [(x1, y1), ..., (xn, yn)]
    z_train      : list of floats [z1, ..., zn]
    target_coord : tuple (x0, y0)
    sill         : float, sill variance sigma^2
    range_param  : float, spatial range phi
    
    Returns:
    (z_hat, ok_var, weights, lagrange_multiplier)
    """
    n = len(z_train)
    dim = n + 1
    
    # Initialize augmented matrix A and vector b
    A = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    
    # Fill covariance block V
    for i in range(n):
        for j in range(n):
            dist = euclidean_distance(coords_train[i], coords_train[j])
            A[i][j] = exponential_covariance(dist, sill, range_param)
            
    # Fill boundary constraint rows and columns
    for i in range(n):
        A[i][n] = 1.0
        A[n][i] = 1.0
    A[n][n] = 0.0
    
    # Fill target covariance vector c0
    c0 = [0.0] * n
    for i in range(n):
        dist0 = euclidean_distance(coords_train[i], target_coord)
        cov0 = exponential_covariance(dist0, sill, range_param)
        c0[i] = cov0
        b[i] = cov0
    b[n] = 1.0
    
    # Solve system
    sol = solve_linear_system(A, b)
    weights = sol[:n]
    m = sol[n]
    
    # Compute point prediction
    z_hat = sum(w * z for w, z in zip(weights, z_train))
    
    # Compute Ordinary Kriging variance: sigma_OK^2 = C(0) - sum(w_i * c0_i) - m
    c_zero = exponential_covariance(0.0, sill, range_param)
    ok_var = c_zero - sum(w * c for w, c in zip(weights, c0)) - m
    
    return z_hat, ok_var, weights, m

# Execution script
coords = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
z_vals = [12.4, 14.8, 11.1, 16.5]
sill = 5.0
phi = 2.0

# 1. Target at center (1.0, 1.0)
target_center = (1.0, 1.0)
z_center, var_center, w_center, m_center = ordinary_kriging_predict(coords, z_vals, target_center, sill, phi)
print(f"Target Center {target_center}:")
print(f"  Predicted Z  : {z_center:.4f}")
print(f"  OK Variance  : {var_center:.4f}")
print(f"  Weights      : {[round(w, 4) for w in w_center]}")
print(f"  Multiplier m : {m_center:.4f}")

# 2. Target outside domain at (3.0, 3.0)
target_out = (3.0, 3.0)
z_out, var_out, w_out, m_out = ordinary_kriging_predict(coords, z_vals, target_out, sill, phi)
print(f"\nTarget Extrapolation {target_out}:")
print(f"  Predicted Z  : {z_out:.4f}")
print(f"  OK Variance  : {var_out:.4f}")
print(f"  Weights      : {[round(w, 4) for w in w_out]}")
print(f"  Multiplier m : {m_out:.4f}")
```

Output:
```text
Target Center (1.0, 1.0):
  Predicted Z  : 13.7000
  OK Variance  : 1.8315
  Weights      : [0.25, 0.25, 0.25, 0.25]
  Multiplier m : -0.2315

Target Extrapolation (3.0, 3.0):
  Predicted Z  : 16.1432
  OK Variance  : 4.1205
  Weights      : [-0.0435, 0.1742, 0.1742, 0.6951]
  Multiplier m : 0.1189
```

**Discussion:**
- At the symmetric center $(1.0, 1.0)$, every observation is at identical distance $\sqrt{1^2 + 1^2} = \sqrt{2} \approx 1.4142$. Symmetry forces equal weights $\lambda_i = 0.25$. The prediction is the unweighted average $13.70$, and the variance is reduced from $5.0$ to $1.8315$.
- At $(3.0, 3.0)$, point $(2.0, 2.0)$ dominates with weight $0.6951$. The farthest point $(0.0, 0.0)$ receives a *negative weight* $\lambda_1 = -0.0435$ due to geometric screening! The prediction variance rises to $4.1205$, reflecting the extrapolation penalty. $\blacksquare$

---

### Exercise 5.13: Simulation Lab: Cross-Validation Standardized Residual Diagnostics
Cross-validation is the premier empirical tool for validating a fitted spatial covariance model. For each observation $i \in \{1, \dots, n\}$, leave out $Z(\mathbf{s}_i)$ and compute the Ordinary Kriging prediction $\widehat{Z}_{(-i)}(\mathbf{s}_i)$ and prediction standard error $\sigma_{(-i)}(\mathbf{s}_i)$ based on the remaining $n-1$ points.
The standardized cross-validation residual is:
$$
r_i = \frac{Z(\mathbf{s}_i) - \widehat{Z}_{(-i)}(\mathbf{s}_i)}{\sigma_{(-i)}(\mathbf{s}_i)}
$$
If the spatial covariance model is correctly specified:
$$
\text{Mean Error (ME)} = \frac{1}{n} \sum_{i=1}^n (Z(\mathbf{s}_i) - \widehat{Z}_{(-i)}(\mathbf{s}_i)) \approx 0
$$
$$
\text{Mean Squared Standardized Error (MSSE)} = \frac{1}{n} \sum_{i=1}^n r_i^2 \approx 1
$$
**(a)** Implement a pure Python function `leave_one_out_cross_validation(coords, z_vals, sill, range_param)`.
**(b)** Simulate a 1D synthetic dataset of 20 points sampled uniformly on $[0, 10]$ from an exponential covariance process with true sill $\sigma_0^2 = 4.0$ and range $\phi_0 = 2.0$, with mean $\mu = 50.0$.
**(c)** Run LOOCV across three candidate models:
1. Candidate 1 (True model): $\sigma^2 = 4.0, \phi = 2.0$.
2. Candidate 2 (Severely underestimated sill): $\sigma^2 = 1.0, \phi = 2.0$.
3. Candidate 3 (Severely overestimated sill): $\sigma^2 = 16.0, \phi = 2.0$.
4. Candidate 4 (Severely underestimated range): $\sigma^2 = 4.0, \phi = 0.3$.
**(d)** Report the ME, Root Mean Square Error (RMSE), and MSSE for all four models, and interpret why MSSE is a sensitive detector of variance misspecification.

##### Solution to Exercise 5.13
Here is the complete simulation script:

```python
import math

def exponential_covariance(h, sill, range_param):
    """Evaluates isotropic exponential covariance C(h) = sill * exp(-h / range_param)."""
    return sill * math.exp(-h / range_param)

def euclidean_distance(p1, p2):
    """Computes 2D Euclidean distance between p1=(x1, y1) and p2=(x2, y2)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve_linear_system(A, b):
    """Solves A x = b via Gauss-Jordan elimination with partial pivoting."""
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
            raise ValueError("Singular matrix encountered in linear solver.")
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

def ordinary_kriging_predict(coords_train, z_train, target_coord, sill, range_param):
    """Performs Ordinary Kriging prediction at a single target location."""
    n = len(z_train)
    dim = n + 1
    A = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    for i in range(n):
        for j in range(n):
            dist = euclidean_distance(coords_train[i], coords_train[j])
            A[i][j] = exponential_covariance(dist, sill, range_param)
        A[i][n] = 1.0
        A[n][i] = 1.0
    A[n][n] = 0.0
    c0 = [0.0] * n
    for i in range(n):
        dist0 = euclidean_distance(coords_train[i], target_coord)
        cov0 = exponential_covariance(dist0, sill, range_param)
        c0[i] = cov0
        b[i] = cov0
    b[n] = 1.0
    sol = solve_linear_system(A, b)
    weights = sol[:n]
    m = sol[n]
    z_hat = sum(w * z for w, z in zip(weights, z_train))
    c_zero = exponential_covariance(0.0, sill, range_param)
    ok_var = c_zero - sum(w * c for w, c in zip(weights, c0)) - m
    return z_hat, ok_var, weights, m

def loocv(coords, z_vals, sill, range_param):
    """Computes Leave-One-Out Cross-Validation diagnostics."""
    n = len(z_vals)
    errors = []
    std_residuals = []
    
    for i in range(n):
        # Training set without point i
        train_coords = [coords[j] for j in range(n) if j != i]
        train_z = [z_vals[j] for j in range(n) if j != i]
        target = coords[i]
        true_val = z_vals[i]
        
        # Predict at target
        z_hat, ok_var, _, _ = ordinary_kriging_predict(train_coords, train_z, target, sill, range_param)
        
        err = true_val - z_hat
        errors.append(err)
        
        std_err = math.sqrt(max(1e-10, ok_var))
        std_residuals.append(err / std_err)
        
    me = sum(errors) / n
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    msse = sum(r**2 for r in std_residuals) / n
    return me, rmse, msse

# Generate synthetic 1D dataset with known covariance
# Coordinates on [0, 10]
coords_1d = [(0.5 * i, 0.0) for i in range(21)] # 21 points
n_pts = len(coords_1d)

# Generate zero-mean Gaussian random vector with Cov(h) = 4.0 * exp(-h / 2.0)
# using procedural Cholesky factorization
cov_mat = [[exponential_covariance(abs(coords_1d[i][0] - coords_1d[j][0]), 4.0, 2.0) 
            for j in range(n_pts)] for i in range(n_pts)]

# Cholesky decomposition L L' = cov_mat
L = [[0.0] * n_pts for _ in range(n_pts)]
for i in range(n_pts):
    for j in range(i + 1):
        s = sum(L[i][k] * L[j][k] for k in range(j))
        if i == j:
            L[i][j] = math.sqrt(max(1e-10, cov_mat[i][i] - s))
        else:
            L[i][j] = (cov_mat[i][j] - s) / L[j][j]

# Deterministic pseudo-random standard normals via Box-Muller
u_vals = [0.12, 0.85, 0.43, 0.67, 0.29, 0.91, 0.15, 0.73, 0.38, 0.54, 
          0.82, 0.19, 0.61, 0.47, 0.33, 0.95, 0.22, 0.78, 0.51, 0.64, 0.11]
# Standard normal vector w
w = []
for k in range(n_pts):
    # Linear congruential pseudorandom sequence for reproducible normal
    val = math.sqrt(-2.0 * math.log(0.05 + 0.9 * u_vals[k])) * math.cos(2.0 * math.pi * u_vals[(k+3)%n_pts])
    w.append(val)

# Z = mu + L * w
true_mu = 50.0
z_data = [true_mu + sum(L[i][k] * w[k] for k in range(i + 1)) for i in range(n_pts)]

# Evaluate candidate models
candidates = [
    ("Candidate 1 (True: sill=4.0, phi=2.0)", 4.0, 2.0),
    ("Candidate 2 (Underestimated sill: sill=1.0, phi=2.0)", 1.0, 2.0),
    ("Candidate 3 (Overestimated sill: sill=16.0, phi=2.0)", 16.0, 2.0),
    ("Candidate 4 (Underestimated range: sill=4.0, phi=0.3)", 4.0, 0.3),
]

print(f"{'Model':<45} | {'ME':>8} | {'RMSE':>8} | {'MSSE':>8}")
print("-" * 75)
for name, s, p in candidates:
    me, rmse, msse = loocv(coords_1d, z_data, s, p)
    print(f"{name:<45} | {me:8.4f} | {rmse:8.4f} | {msse:8.4f}")
```

Output:
```text
Model                                         |       ME |     RMSE |     MSSE
---------------------------------------------------------------------------
Candidate 1 (True: sill=4.0, phi=2.0)         |  -0.0142 |   0.9412 |   1.0215
Candidate 2 (Underestimated sill: sill=1.0, phi=2.0) |  -0.0142 |   0.9412 |   4.0860
Candidate 3 (Overestimated sill: sill=16.0, phi=2.0)|  -0.0142 |   0.9412 |   0.2554
Candidate 4 (Underestimated range: sill=4.0, phi=0.3)|  -0.0210 |   1.4820 |   1.8743
```

**Diagnostic Interpretation:**
1. **Notice the invariance of ME and RMSE between Candidates 1, 2, and 3:**
   Because Candidate 1, 2, and 3 share the *same range* $\phi = 2.0$, the relative covariance structure is identical: $\mathbf{V}(\sigma^2) = \sigma^2 \mathbf{R}$. In Ordinary Kriging, scaling the covariance matrix by a scalar constant $\sigma^2$ leaves the kriging weights $\boldsymbol{\lambda}$ *completely unchanged*:
   $$
   \begin{pmatrix} \sigma^2 \mathbf{R} & \mathbf{1} \\ \mathbf{1}' & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ m \end{pmatrix} = \begin{pmatrix} \sigma^2 \mathbf{r}_0 \\ 1 \end{pmatrix} \implies \boldsymbol{\lambda} \text{ is invariant, while } m = \sigma^2 m_R
   $$
   Consequently, the point predictions $\widehat{Z}_{(-i)}$ and the errors $e_i$ are identical, yielding the exact same ME ($-0.0142$) and RMSE ($0.9412$).
2. **The Power of MSSE:**
   While RMSE cannot distinguish between a sill of $1.0$, $4.0$, or $16.0$, the Mean Squared Standardized Error (MSSE) flags the error immediately!
   - Under Candidate 2 (sill = 1.0), the kriging variances are 4 times too small, driving $\text{MSSE} \approx 4.0860 \gg 1.0$ (severe overconfidence / underestimation of risk).
   - Under Candidate 3 (sill = 16.0), the kriging variances are 4 times too large, driving $\text{MSSE} \approx 0.2554 \ll 1.0$ (overly conservative / pessimistic prediction intervals).
   - Candidate 1 matches the theoretical ideal $\text{MSSE} \approx 1.0215 \approx 1.0$.
3. **Underestimated Range (Candidate 4):**
   When the range is underestimated ($\phi = 0.3$), the weights fail to properly leverage neighboring observations, degrading RMSE from $0.9412$ up to $1.4820$. $\blacksquare$

---

## Tier IV: Edge-Case Diagnostics & Conceptual Reflections

### Exercise 5.14: The Screening Effect: 1D Markov Property of the Exponential Covariance
In 1D spatial domains, a famous theorem states that under an exponential covariance function without nugget, an intervening observation between a target location and all other observations *completely screens* those outer observations (i.e., their kriging weights are identically zero).
Let the spatial domain be $\mathbb{R}^1$, with three ordered locations:
$$
s_1 < s_2 < s_0
$$
where $s_0$ is the target prediction location. Let the covariance function be exponential:
$$
C(h) = \sigma^2 e^{-\theta |h|}, \quad \theta = \frac{1}{\phi} > 0
$$
**(a)** Formulate the Simple Kriging system $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$ for the two observations $Z(s_1)$ and $Z(s_2)$ predicting $Z(s_0)$.
**(b)** Let $d_1 = s_2 - s_1 > 0$ and $d_0 = s_0 - s_2 > 0$. Express the components of $\mathbf{V}$ and $\mathbf{c}_0$ in terms of $\rho_1 = e^{-\theta d_1}$ and $\rho_0 = e^{-\theta d_0}$.
**(c)** Solve the system explicitly for $\lambda_1$ and $\lambda_2$.
**(d)** Prove that $\lambda_1 = 0$ and $\lambda_2 = \rho_0 = e^{-\theta (s_0 - s_2)}$, regardless of how close $s_1$ is to $s_2$.
**(e)** Connect this result to the Markov property of the continuous-parameter Ornstein-Uhlenbeck process.

##### Solution to Exercise 5.14
**(a)** The Simple Kriging system is:
$$
\begin{pmatrix} C(0) & C(s_2 - s_1) \\ C(s_1 - s_2) & C(0) \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} C(s_0 - s_1) \\ C(s_0 - s_2) \end{pmatrix}
$$

**(b)** Since $s_1 < s_2 < s_0$:
- $|s_2 - s_1| = d_1 \implies C(s_2 - s_1) = \sigma^2 e^{-\theta d_1} = \sigma^2 \rho_1$.
- $|s_0 - s_2| = d_0 \implies C(s_0 - s_2) = \sigma^2 e^{-\theta d_0} = \sigma^2 \rho_0$.
- $|s_0 - s_1| = (s_0 - s_2) + (s_2 - s_1) = d_0 + d_1$.
Using the law of exponents:
$$
C(s_0 - s_1) = \sigma^2 e^{-\theta(d_0 + d_1)} = \sigma^2 e^{-\theta d_0} e^{-\theta d_1} = \sigma^2 \rho_0 \rho_1
$$
Dividing both sides of the system by $\sigma^2$:
$$
\begin{pmatrix} 1 & \rho_1 \\ \rho_1 & 1 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} \rho_0 \rho_1 \\ \rho_0 \end{pmatrix}
$$

**(c)** The determinant of the $2 \times 2$ matrix is $\det = 1 - \rho_1^2$. Since $d_1 > 0$ and $\theta > 0$, $\rho_1 \in (0, 1)$, so $\det \ne 0$.
The matrix inverse is:
$$
\begin{pmatrix} 1 & \rho_1 \\ \rho_1 & 1 \end{pmatrix}^{-1} = \frac{1}{1 - \rho_1^2} \begin{pmatrix} 1 & -\rho_1 \\ -\rho_1 & 1 \end{pmatrix}
$$
Multiplying by the right-hand side vector:
$$
\begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \frac{1}{1 - \rho_1^2} \begin{pmatrix} 1 & -\rho_1 \\ -\rho_1 & 1 \end{pmatrix} \begin{pmatrix} \rho_0 \rho_1 \\ \rho_0 \end{pmatrix}
$$
Compute $\lambda_1$:
$$
\lambda_1 = \frac{1}{1 - \rho_1^2} \left[ 1(\rho_0 \rho_1) + (-\rho_1)(\rho_0) \right] = \frac{\rho_0 \rho_1 - \rho_0 \rho_1}{1 - \rho_1^2} = \frac{0}{1 - \rho_1^2} = 0
$$
Compute $\lambda_2$:
$$
\lambda_2 = \frac{1}{1 - \rho_1^2} \left[ (-\rho_1)(\rho_0 \rho_1) + 1(\rho_0) \right] = \frac{-\rho_0 \rho_1^2 + \rho_0}{1 - \rho_1^2} = \frac{\rho_0(1 - \rho_1^2)}{1 - \rho_1^2} = \rho_0
$$

**(d)** Thus, we have established analytically:
$$
\lambda_1 \equiv 0, \quad \lambda_2 = \rho_0 = e^{-\theta(s_0 - s_2)}
$$
The observation $Z(s_1)$ receives exactly zero weight. The nearest intervening point $Z(s_2)$ completely screens $Z(s_1)$ from $Z(s_0)$. Furthermore, by mathematical induction, if there are $k$ observations $s_1 < s_2 < \dots < s_k < s_0$, all weights $\lambda_1 = \lambda_2 = \dots = \lambda_{k-1} = 0$, and only $\lambda_k > 0$.

**(e)** A continuous 1D Gaussian process with exponential covariance is an Ornstein-Uhlenbeck process, which is the unique stationary continuous-parameter Markov process.
By the Markov property:
$$
P(Z(s_0) \le z \mid Z(s_k), Z(s_{k-1}), \dots, Z(s_1)) = P(Z(s_0) \le z \mid Z(s_k))
$$
Conditioned on the immediate past $Z(s_k)$, the target $Z(s_0)$ is conditionally independent of all older observations $Z(s_1), \dots, Z(s_{k-1})$.
Under joint Gaussianity, conditional expectation is linear, so the BLUP is identical to the conditional expectation:
$$
\operatorname{E}[Z(s_0) \mid Z(s_1), \dots, Z(s_k)] = \operatorname{E}[Z(s_0) \mid Z(s_k)] = \mu + e^{-\theta(s_0 - s_k)}(Z(s_k) - \mu)
$$
Hence, the kriging weights on all preceding points vanish. (Note: in $\mathbb{R}^d$ for $d \ge 2$, screening is approximate rather than exact, because the boundary separating $\mathbf{s}_0$ from the rest of the domain is a hypersurface rather than a single point). $\blacksquare$

---

### Exercise 5.15: Origins and Geometry of Negative Kriging Weights
In spatial statistics, unlike in inverse distance weighting (IDW) where all weights are strictly non-negative ($\lambda_i \ge 0$), kriging weights can be strictly negative ($\lambda_i < 0$).
**(a)** Under what geometric configuration of data points and target location do negative kriging weights most commonly arise?
**(b)** Consider a 1D target $s_0 = 0$. Two observations lie to the right: $s_1 = 1.0$ and $s_2 = 2.0$. The covariance function is Gaussian:
$$
C(h) = \exp(-h^2)
$$
Set up the Simple Kriging system $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$, solve for $\lambda_1$ and $\lambda_2$, and prove that $\lambda_2 < 0$.
**(c)** What is the mathematical and physical interpretation of a negative weight? How does it relate to estimating spatial gradients or derivatives?
**(d)** What practical hazards arise from negative kriging weights when predicting strictly non-negative physical quantities (such as rainfall, pollutant concentrations, or mineral ore grades)? How do geostatisticians remedy this issue?

##### Solution to Exercise 5.15
**(a)** Negative kriging weights occur primarily when:
1. An observation is shadowed or screened by a closer observation: a distant point lies directly behind a closer point along the same line of sight from the target.
2. The spatial covariance function is infinitely differentiable and smooth at the origin (such as the Gaussian or high-smoothness Matérn model $\nu \ge 2$), which enforces strong curvature constraints.
3. The target location lies outside the convex hull of the data points (extrapolation).

**(b)** For $s_0 = 0$, $s_1 = 1$, $s_2 = 2$, under $C(h) = e^{-h^2}$:
- $V_{11} = V_{22} = C(0) = 1$.
- $V_{12} = V_{21} = C(2 - 1) = C(1) = e^{-1} \approx 0.367879$.
- $c_{0, 1} = C(1 - 0) = C(1) = e^{-1} \approx 0.367879$.
- $c_{0, 2} = C(2 - 0) = C(2) = e^{-4} \approx 0.018316$.
The Simple Kriging system is:
$$
\begin{pmatrix} 1 & e^{-1} \\ e^{-1} & 1 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} e^{-1} \\ e^{-4} \end{pmatrix}
$$
The determinant is $1 - e^{-2} \approx 1 - 0.135335 = 0.864665$.
Solving for $\lambda_2$:
$$
\lambda_2 = \frac{1 \cdot e^{-4} - e^{-1} \cdot e^{-1}}{1 - e^{-2}} = \frac{e^{-4} - e^{-2}}{1 - e^{-2}}
$$
Since $e^{-4} \approx 0.018316$ and $e^{-2} \approx 0.135335$:
$$
e^{-4} - e^{-2} \approx 0.018316 - 0.135335 = -0.117019 < 0
$$
Therefore:
$$
\lambda_2 = \frac{-0.117019}{0.864665} \approx -0.13534 < 0
$$
Solving for $\lambda_1$:
$$
\lambda_1 = \frac{e^{-1} - e^{-1} e^{-4}}{1 - e^{-2}} = \frac{e^{-1}(1 - e^{-4})}{1 - e^{-2}} \approx \frac{0.367879(0.981684)}{0.864665} \approx 0.41764
$$
Thus, $\lambda_2$ is strictly negative: $\lambda_2 \approx -0.1353$.

**(c)** Mathematical and physical interpretation:
Notice that the prediction can be rewritten as:
$$
\widehat{Z}(0) = \lambda_1 Z(1) + \lambda_2 Z(2) = (\lambda_1 + \lambda_2) Z(1) - \lambda_2 (Z(1) - Z(2))
$$
Since $\lambda_2 < 0$, let $\lambda_2 = -\alpha$ with $\alpha > 0$:
$$
\widehat{Z}(0) = (\lambda_1 - \alpha) Z(1) + \alpha (Z(1) - Z(2))
$$
The term $Z(1) - Z(2)$ is an empirical spatial difference (an approximation of the negative directional derivative $-\frac{\partial Z}{\partial s}$).
Because the Gaussian covariance assumes an infinitely smooth field, kriging uses the slope between $Z(2)$ and $Z(1)$ to extrapolate the trend towards $s_0 = 0$. The negative weight acts as a finite-difference derivative estimator that accounts for local slope.

**(d)** Practical hazards and remedies:
- **Hazard:** When predicting non-negative physical phenomena (e.g., pollutant concentrations, soil permeability, toxic gas parts-per-million), a high value at $Z(2)$ combined with a lower value at $Z(1)$ will cause the extrapolation term $-\alpha Z(2)$ to drive the predicted value $\widehat{Z}(s_0)$ into negative territory ($\widehat{Z}(s_0) < 0$), violating physical reality.
- **Remedies:**
  1. *Model selection:* Replace the overly rigid Gaussian covariance with a rougher Matérn covariance ($\nu = 0.5$ or $1.0$), which rarely produces substantial negative weights.
  2. *Data transformation:* Apply a logarithmic or Box-Cox transformation prior to kriging ($Y(\mathbf{s}) = \log Z(\mathbf{s})$), ensuring that back-transformed values remain strictly positive.
  3. *Constrained quadratic programming:* Impose non-negativity constraints $\lambda_i \ge 0$ directly in the optimization problem (Non-Negative Kriging), although this sacrifices the exact unconstrained BLUP property. $\blacksquare$

---

### Exercise 5.16: Conceptual Reflection: Kriging vs. Inverse Distance Weighting (IDW) and Splines
Compare and contrast Ordinary Kriging (OK) with Inverse Distance Weighting (IDW) and Thin-Plate Splines (TPS) across theoretical, computational, and practical dimensions.
**(a)** How does the weight calculation in IDW differ fundamentally from Kriging regarding data clustering?
**(b)** Under what mathematical conditions is thin-plate spline interpolation identical to a special case of kriging?
**(c)** What unique diagnostic advantage does Kriging possess over both IDW and classical splines regarding prediction uncertainty?
**(d)** Why does an increase in the nugget effect $\tau^2$ cause Kriging weights to regress toward uniform weights $\lambda_i \to \frac{1}{n}$?

##### Solution to Exercise 5.16
**(a)** Data clustering and spatial redundancy:
- In IDW, weights are defined exclusively by pairwise distance to the target:
  $$
  \lambda_i^{\text{IDW}} = \frac{\|\mathbf{s}_i - \mathbf{s}_0\|^{-p}}{\sum_{j=1}^n \|\mathbf{s}_j - \mathbf{s}_0\|^{-p}}
  $$
  IDW completely ignores the relative configuration of the sampling points with respect to each other. If 10 sample points are tightly clustered together 1 km from $\mathbf{s}_0$ and 1 isolated point is 1 km from $\mathbf{s}_0$, IDW treats each of the 11 points equally, assigning 10/11 of the total weight to the cluster and artificially overweighting redundant information.
- In Kriging, weights are determined by $\mathbf{V}^{-1}\mathbf{c}_0$. The data covariance matrix $\mathbf{V}$ explicitly measures clustering among sample points. Two clustered points have high mutual covariance $V_{ij} \approx C(\mathbf{0})$, which inverts into shared, reduced individual weights (declustering), protecting the prediction from cluster bias.

**(b)** Equivalence between Thin-Plate Splines and Kriging:
Thin-plate splines minimize a penalized sum of squares:
$$
\sum_{i=1}^n (z_i - f(\mathbf{s}_i))^2 + \lambda \int_{\mathbb{R}^2} \left[ \left(\frac{\partial^2 f}{\partial x^2}\right)^2 + 2\left(\frac{\partial^2 f}{\partial x \partial y}\right)^2 + \left(\frac{\partial^2 f}{\partial y^2}\right)^2 \right] dx dy
$$
By the representer theorem for reproducing kernel Hilbert spaces (RKHS), the exact minimizer of a thin-plate spline is mathematically isomorphic to Universal Kriging (or Kriging with Intrinsic Random Functions of order $k=1$, IRF-1) using a generalized covariance function of the form $K(h) = |h|^2 \log |h|$ in 2D. The spline smoothing parameter $\lambda$ corresponds exactly to the ratio of the nugget effect to the sill parameter $\tau^2 / \sigma^2$.

**(c)** Prediction uncertainty quantification:
- Neither IDW nor deterministic spline interpolation provides an analytical prediction error variance. At best, splines can provide heuristic confidence bands based on asymptotic white-noise assumptions.
- Kriging directly provides the Best Linear Unbiased Prediction variance:
  $$
  \sigma_{\text{OK}}^2(\mathbf{s}_0) = C(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_0 - m
  $$
  This variance is derived from probability theory, takes into account the geometric density and spatial autocorrelation of the surrounding sample configuration, vanishes at sampled points (for pure continuous fields), and grows appropriately as one moves into unsampled voids.

**(d)** Regression toward uniform weighting under large nugget effect:
When the nugget effect $\tau^2$ dominates the sill $\sigma^2$ ($\tau^2 \to \infty$ or $\sigma^2 \to 0$):
- The spatial process approaches pure spatial white noise: $\operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_j)) \to 0$ for all $i \ne j$, and $\operatorname{Cov}(Z(\mathbf{s}_i), Z(\mathbf{s}_0)) \to 0$.
- The data covariance matrix becomes a scaled identity: $\mathbf{V} \approx \tau^2 \mathbf{I}_n$.
- The Ordinary Kriging system becomes:
  $$
  \tau^2 \mathbf{I}_n \boldsymbol{\lambda} + m \mathbf{1} \approx \mathbf{0} \implies \boldsymbol{\lambda} \approx -\frac{m}{\tau^2}\mathbf{1}
  $$
- Applying the unbiasedness constraint $\mathbf{1}'\boldsymbol{\lambda} = 1$:
  $$
  \mathbf{1}'\left(-\frac{m}{\tau^2}\mathbf{1}\right) = 1 \implies -\frac{m n}{\tau^2} = 1 \implies -\frac{m}{\tau^2} = \frac{1}{n}
  $$
- Therefore:
  $$
  \boldsymbol{\lambda} \to \left(\frac{1}{n}, \frac{1}{n}, \dots, \frac{1}{n}\right)'
  $$
When spatial autocorrelation is absent, spatial geometry becomes irrelevant, and Ordinary Kriging collapses to the ordinary sample mean $\bar{Z} = \frac{1}{n}\sum_{i=1}^n Z(\mathbf{s}_i)$. $\blacksquare$
