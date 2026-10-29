# Chapter 7: Simultaneous Autoregressive Models: Spatial Regression with Feedback

Chapter 6 gave us a way to say which regions are neighbors and a way to test whether a variable is spatially correlated. This chapter builds the first regression models for data on regions that put that neighbor structure to work. The idea is the one that makes spatial econometrics a field of its own: a region's value depends on its neighbors' values, and their values depend on it, all at once.

There are two ways to write that down, and they answer different questions. In the **spatial lag model** the feedback is in the response: a city's tax rate depends on the tax rates of its neighbors. In the **spatial error model** the feedback is in the unexplained part: counties share unobserved shocks, and the shocks echo across borders. Both models have a covariance matrix of the same shape, so both are linear models with a structured $\boldsymbol{\Sigma}$ and everything from Chapters 1 and 5 applies. They differ in the mean, and that difference matters a great deal for what the coefficients mean, for what goes wrong with ordinary least squares, and for how to report results.

We derive each model, see why OLS fails in each (a different way), fit them by maximum likelihood with one trick that makes the likelihood cheap (Ord's eigenvalue identity), test which model the data prefer, and learn to interpret the lag model through its direct, indirect, and total impacts. Four small tools are introduced along the way, each with an example before it is used: the Neumann series for a matrix inverse, the determinant as a product of eigenvalues, the change of variables formula for a density, and the score test.

The running data set is a simulated map of 144 square counties. Responses are simulated with known parameters, so every estimator can be checked against the truth.

> **Chapter at a Glance**
>
> - **The Simultaneous Idea (§7.1):** Each region regressed on its neighbors, all at once, and the two places to put the feedback.
> - **The Spatial Error Model (§7.2):** Correlated shocks, the induced covariance $\sigma^2[(\mathbf{I}-\rho\mathbf{W})'(\mathbf{I}-\rho\mathbf{W})]^{-1}$, the precision matrix $\mathbf{Q}$, and GLS as OLS on spatially filtered data.
> - **The Spatial Lag Model (§7.3):** The feedback multiplier as a series in $\mathbf{W}$, and why OLS is biased in two different ways.
> - **The Parameter Space (§7.4):** When $\mathbf{I}-\rho\mathbf{W}$ can be inverted, and why $\rho$ lives in an interval around zero.
> - **Maximum Likelihood (§7.5):** The Jacobian term, Ord's eigenvalue identity, and a one-dimensional profile search over $\rho$.
> - **Choosing a Model (§7.6):** Lagrange multiplier tests, robust versions, likelihood ratio tests, and AIC.
> - **Impacts (§7.7):** Why a lag-model coefficient is not a marginal effect, and direct, indirect, and total impacts.
> - **Walkthrough and Lab (§7.8):** A four-region example in code, a simulated county map, and checks of every claim above.

---

## 7.1 The Simultaneous Idea

Think of neighboring cities setting their property tax rates. A city that raises its rate risks losing residents and businesses to a neighbor with a lower one, and voters compare their own city's rate with what is charged next door. So each city's rate depends on its neighbors' rates. But the neighbors are reacting to this city and to their own neighbors in the same way. No city moves first, and no city's decision can be treated as given while the others are solved. The rates are determined together, as the solution of a system of equations.

Write this for region $i$, using the row-standardized weights $w_{ij}$ of Chapter 6:

$$
Y_i = \rho\sum_{j=1}^n w_{ij}Y_j + \mathbf{x}_i'\boldsymbol{\beta} + \varepsilon_i
$$

The sum $\sum_j w_{ij}Y_j$ is the average of $Y$ over region $i$'s neighbors, so $\rho$ says how strongly a region follows its neighbors' average. The vector $\mathbf{x}_i$ holds the region's own covariates, and $\varepsilon_i$ is a fresh independent disturbance. For all regions at once, $\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$. The word *simultaneous* refers to the fact that $\mathbf{Y}$ appears on both sides: every equation involves values that are themselves unknown and are the subject of other equations.

**A two-region example.** To see what simultaneity does, take two regions that are each other's only neighbor, so $w_{12} = w_{21} = 1$, and drop the covariates: $Y_1 = \rho Y_2 + \varepsilon_1$ and $Y_2 = \rho Y_1 + \varepsilon_2$. Substitute the second equation into the first:

$$
Y_1 = \rho(\rho Y_1 + \varepsilon_2) + \varepsilon_1 \quad\Longrightarrow\quad Y_1 = \frac{\varepsilon_1 + \rho\varepsilon_2}{1 - \rho^2}, \qquad Y_2 = \frac{\varepsilon_2 + \rho\varepsilon_1}{1 - \rho^2}
$$

With $\rho = 0.5$ and a shock of 1 in region 1 and none in region 2, $Y_1 = 1/0.75 = 1.333$ and $Y_2 = 0.5/0.75 = 0.667$. The shock was 1, but region 1 rises by more than 1, because the rise in region 2 feeds back. That echo is the signature of a simultaneous model, and the matrix algebra below is the general version of this calculation. The code confirms the numbers by solving the two equations as a linear system.

```python
import numpy as np
from scipy import optimize, stats

W2 = np.array([[0.0, 1.0], [1.0, 0.0]])
rho = 0.5
shock = np.array([1.0, 0.0])
print("response to a shock in region 1:",
      np.round(np.linalg.solve(np.eye(2) - rho * W2, shock), 4))
```

**What we assume about $\mathbf{W}$.** Chapter 6 built $\mathbf{W}$ from the adjacency matrix $\mathbf{A}$ and the degree matrix $\mathbf{D}$ as $\mathbf{W} = \mathbf{D}^{-1}\mathbf{A}$. This chapter uses four facts from that construction: $\mathbf{W}$ has zeros on its diagonal (no region is its own neighbor), its rows sum to 1 so that $\mathbf{W}\mathbf{1}_n = \mathbf{1}_n$, it is not symmetric in general, and its eigenvalues are real because $\mathbf{W}$ is similar to the symmetric matrix $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ (Section 6.3.3).

**Two places for the feedback.** The equation above puts the neighbors' values into the model for $Y_i$. That is the lag model. The alternative is to leave the mean alone and let the unexplained part of each region echo its neighbors' unexplained parts. Both are called simultaneous autoregressive (SAR) models, and the table summarizes how they differ.

| | Spatial lag model | Spatial error model |
| :--- | :--- | :--- |
| Equation | $\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ | $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$, $\ \mathbf{u} = \rho\mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}$ |
| Feedback in | The response | The unexplained part |
| Typical story | Strategic interaction: tax rates, prices | Shared shocks: weather, local economy |
| Mean of $\mathbf{Y}$ | $(\mathbf{I}-\rho\mathbf{W})^{-1}\mathbf{X}\boldsymbol{\beta}$ | $\mathbf{X}\boldsymbol{\beta}$ |
| What OLS does | Biased and inconsistent | Unbiased, wrong standard errors |
| Coefficient means | Not the marginal effect | The marginal effect |

Two things stand out. The covariance of $\mathbf{Y}$ turns out to be the same in the two models, as Section 7.3.1 shows, and the means differ. And Chapter 8 will approach the same dependence from the opposite direction, by specifying each region's distribution given its neighbors; Section 8.5 compares the two.

In the language of the one model equation of Section 0.2, the spatial error model writes $Y = \mu + u$ with the spatial random effect $u$ built from independent innovations $\varepsilon$. The spatial lag model has the same covariance, but its mean also passes through the feedback matrix, as Section 7.3.1 shows. There is no separate nugget in these models: the independent part is the innovation that drives $u$, and its variance is $\sigma^2$.

---

## 7.2 The Spatial Error Model

### 7.2.1 The Model

Counties on either side of a state line have different minimum wages, but workers commute across the line and firms relocate across it. A researcher regressing county employment growth on the county's own minimum wage and other covariates will leave out many things: a plant closing, a regional slowdown, a bad winter. Such shocks do not respect county borders. A shock in one county spreads partly to its neighbors, so the unexplained part of the regression is correlated across neighbors.

The spatial error model writes this directly:

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}, \qquad \mathbf{u} = \rho\mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)
$$

Each county's unexplained part $u_i$ is $\rho$ times the average of its neighbors' unexplained parts, plus a fresh independent innovation $\varepsilon_i$. The regression coefficients $\boldsymbol{\beta}$ keep their usual meaning: $\beta_k$ is the change in the expected response when covariate $k$ increases by one unit, with nothing else moving. All of the spatial structure is in the errors.

By Chapter 1, this means OLS remains unbiased for $\boldsymbol{\beta}$ and its standard errors are wrong, understated most when the covariates are spatially smooth. The remaining questions are what covariance matrix the model implies and how to do better than OLS.

### 7.2.2 The Reduced Form

The equation for $\mathbf{u}$ has $\mathbf{u}$ on both sides. Collect the terms:

$$
\mathbf{u} - \rho\mathbf{W}\mathbf{u} = \boldsymbol{\varepsilon} \quad\Longrightarrow\quad (\mathbf{I}_n - \rho\mathbf{W})\mathbf{u} = \boldsymbol{\varepsilon}
$$

If the matrix $\mathbf{I}_n - \rho\mathbf{W}$ is invertible, we can solve for $\mathbf{u}$:

$$
\mathbf{u} = (\mathbf{I}_n - \rho\mathbf{W})^{-1}\boldsymbol{\varepsilon}
$$

This is the **reduced form**: the unexplained part written in terms of the independent innovations alone. We give the inverse a name, since it will appear throughout the chapter:

$$
\boldsymbol{\Phi}(\rho) = (\mathbf{I}_n - \rho\mathbf{W})^{-1}
$$

and call it the **feedback matrix**. The entry $\Phi_{ij}$ says how much of innovation $\varepsilon_j$ ends up in $u_i$ after all the echoes. When $\mathbf{I}_n - \rho\mathbf{W}$ can be inverted is the subject of Section 7.4. For now, assume it can.

### 7.2.3 The Covariance and the Precision Matrix

Since $\mathbf{u} = \boldsymbol{\Phi}\boldsymbol{\varepsilon}$ and $\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}_n$, Tool 1 of Section 0.4.1 gives

$$
\operatorname{Var}(\mathbf{u}) = \boldsymbol{\Phi}(\sigma^2\mathbf{I}_n)\boldsymbol{\Phi}' = \sigma^2\boldsymbol{\Phi}\boldsymbol{\Phi}'
$$

To write this without an inverse of a product, use the fact that the inverse of a product is the product of the inverses in reverse order, $(\boldsymbol{\Phi}\boldsymbol{\Phi}')^{-1} = (\boldsymbol{\Phi}')^{-1}\boldsymbol{\Phi}^{-1} = (\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})$. Therefore

$$
\operatorname{Var}(\mathbf{u}) = \sigma^2\left[(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})\right]^{-1}
$$

The inverse of a covariance matrix is called the **precision matrix**. For this model it is available in closed form, and we write it $\mathbf{Q}$:

$$
\mathbf{Q}_{\text{SAR}} = \operatorname{Var}(\mathbf{u})^{-1} = \sigma^{-2}(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})
$$

The precision matrix matters because it describes dependence in a different way than the covariance does. Chapter 8 shows that a zero in $\mathbf{Q}$ means that two regions are independent once all the other regions are known. Here, expand the product:

$$
(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W}) = \mathbf{I}_n - \rho(\mathbf{W} + \mathbf{W}') + \rho^2\mathbf{W}'\mathbf{W}
$$

The entry for regions $i$ and $j$ is nonzero if they are neighbors, through $\mathbf{W} + \mathbf{W}'$, or if they share a neighbor, through $\mathbf{W}'\mathbf{W}$. So $\mathbf{Q}_{\text{SAR}}$ is sparse but reaches two steps across the map, not one. Keep this in mind; it is the main structural difference from the CAR model.

**A hand-sized example.** Take four regions in a $2 \times 2$ block, numbered

$$
\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}
$$

with rook neighbors: regions 1 and 2, 1 and 3, 2 and 4, 3 and 4. Each region has two neighbors, so

$$
\mathbf{W} = \frac{1}{2}\begin{bmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}
$$

This $\mathbf{W}$ happens to be symmetric because every region has the same number of neighbors. Its eigenvalues are $1, 0, 0, -1$, found from the eigenvectors $(1,1,1,1)'$, $(1,0,0,-1)'$, $(0,1,-1,0)'$, and $(1,-1,-1,1)'$; the last is the checkerboard pattern that flips sign between neighbors. Take $\rho = 0.5$. We need $\boldsymbol{\Phi} = (\mathbf{I} - 0.5\mathbf{W})^{-1}$. The matrix $\mathbf{W}$ has the same eigenvectors as $\boldsymbol{\Phi}$, with eigenvalues $1/(1 - 0.5\lambda)$ equal to $2, 1, 1, 2/3$. Section 7.3.2 shows how to get the entries directly; the result is

$$
\boldsymbol{\Phi}(0.5) = \frac{1}{6}\begin{bmatrix} 7 & 2 & 2 & 1 \\ 2 & 7 & 1 & 2 \\ 2 & 1 & 7 & 2 \\ 1 & 2 & 2 & 7 \end{bmatrix}
$$

You can check the first row against the first column of $\mathbf{I} - 0.5\mathbf{W}$, which is $(1, -0.25, -0.25, 0)'$: $\tfrac{7}{6} - \tfrac{2}{6}\cdot 0.25 - \tfrac{2}{6}\cdot 0.25 + 0 = 1$. Now the covariance of $\mathbf{u}$ is $\sigma^2\boldsymbol{\Phi}\boldsymbol{\Phi}'$. Its first row has entries, using $\boldsymbol{\Phi}$ symmetric:

$$
\begin{aligned}
\operatorname{Var}(u_1)/\sigma^2 &= \tfrac{1}{36}(49 + 4 + 4 + 1) = \tfrac{58}{36} = 1.6111 \\
\operatorname{Cov}(u_1, u_2)/\sigma^2 &= \tfrac{1}{36}(7\cdot 2 + 2\cdot 7 + 2\cdot 1 + 1\cdot 2) = \tfrac{32}{36} = 0.8889 \\
\operatorname{Cov}(u_1, u_4)/\sigma^2 &= \tfrac{1}{36}(7\cdot 1 + 2\cdot 2 + 2\cdot 2 + 1\cdot 7) = \tfrac{22}{36} = 0.6111
\end{aligned}
$$

Neighboring regions have correlation $0.8889/1.6111 = 0.552$; regions 1 and 4, which are not neighbors, have correlation $0.379$. Notice that $\rho = 0.5$ is not a correlation: it is a coefficient in the equations, and the correlations it produces are computed from the whole system.

For the precision, $(\mathbf{I} - \rho\mathbf{W})'(\mathbf{I} - \rho\mathbf{W}) = \mathbf{I} - \mathbf{W} + 0.25\,\mathbf{W}^2$ because $\mathbf{W}$ is symmetric and $\rho = 0.5$. With $\mathbf{W}^2 = \tfrac{1}{2}\begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \end{bmatrix}$, the first row of $\sigma^2\mathbf{Q}_{\text{SAR}}$ is

$$
(1 + 0.125,\ -0.5,\ -0.5,\ 0.125) = (1.125,\ -0.5,\ -0.5,\ 0.125)
$$

The last entry is nonzero even though regions 1 and 4 are not neighbors. They share two neighbors, regions 2 and 3, and that is enough. The lab (Section 7.8.1) verifies all of these numbers.

### 7.2.4 Generalized Least Squares for $\boldsymbol{\beta}$ Given $\rho$

Suppose for a moment that $\rho$ and $\sigma^2$ are known. The error model is then a linear model with covariance $\boldsymbol{\Sigma} = \sigma^2\boldsymbol{\Phi}\boldsymbol{\Phi}'$, and Section 1.3.5 says the best linear unbiased estimator is GLS:

$$
\hat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y}
$$

Here $\boldsymbol{\Sigma}^{-1} = \mathbf{Q}_{\text{SAR}} = \sigma^{-2}(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})$ with no matrix to invert, and the factor $\sigma^{-2}$ cancels between the two parts of the formula. Define the **spatially filtered** data

$$
\mathbf{Y}^* = (\mathbf{I}_n - \rho\mathbf{W})\mathbf{Y}, \qquad \mathbf{X}^* = (\mathbf{I}_n - \rho\mathbf{W})\mathbf{X}
$$

Then $\mathbf{X}'(\mathbf{I}-\rho\mathbf{W})'(\mathbf{I}-\rho\mathbf{W})\mathbf{X} = \mathbf{X}^{*\prime}\mathbf{X}^*$ and the estimator is exactly the OLS estimator of $\mathbf{Y}^*$ on $\mathbf{X}^*$:

$$
\hat{\boldsymbol{\beta}}_{\text{GLS}}(\rho) = (\mathbf{X}^{*\prime}\mathbf{X}^*)^{-1}\mathbf{X}^{*\prime}\mathbf{Y}^*, \qquad \operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}}) = \sigma^2(\mathbf{X}^{*\prime}\mathbf{X}^*)^{-1}
$$

This is the decorrelation idea of Section 1.3.5 in its simplest form. There we multiplied the data by $\mathbf{L}^{-1}$ where $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$. Here $\mathbf{L} = \sigma\boldsymbol{\Phi}$ satisfies $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$ (it is a square root of the covariance, though not the triangular Cholesky factor), so $\mathbf{L}^{-1} = \sigma^{-1}(\mathbf{I}_n - \rho\mathbf{W})$: the feedback matrix plays the role of $\mathbf{L}$, and its inverse is the filter. The filter has a plain meaning. The entry for region $i$ of $(\mathbf{I} - \rho\mathbf{W})\mathbf{Y}$ is $Y_i - \rho\sum_j w_{ij}Y_j$: the region's value minus $\rho$ times its neighbors' average. Subtracting the echo leaves the independent innovation. Econometricians call this spatial differencing, by analogy with differencing a time series.

**By hand.** With the four-region map and $\rho = 0.5$, take $\mathbf{y} = (2, 1, 3, 2)'$. The neighbor averages are $\mathbf{W}\mathbf{y} = \tfrac{1}{2}(1 + 3,\ 2 + 2,\ 2 + 2,\ 1 + 3)' = (2, 2, 2, 2)'$, so the filtered data are

$$
\mathbf{y}^* = \mathbf{y} - 0.5\,\mathbf{W}\mathbf{y} = (2 - 1,\ 1 - 1,\ 3 - 1,\ 2 - 1)' = (1, 0, 2, 1)'
$$

In practice $\rho$ is unknown, so GLS is not yet a method; Section 7.5 estimates $\rho$ by maximum likelihood and then applies this filter. Section 7.8.8 shows what is gained: with spatially correlated shocks, the naive OLS interval for a slope covers the truth 77.0% of the time, and the interval from the filtered fit covers it 91.6% of the time.

---

## 7.3 The Spatial Lag Model

### 7.3.1 The Model and Its Reduced Form

The lag model is the one from the tax-rate story in Section 7.1:

$$
\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)
$$

A city's tax rate responds to its neighbors' rates (this is $\rho$) and to its own characteristics, such as population and income (these are in $\mathbf{X}\boldsymbol{\beta}$). The coefficient $\rho$ is the slope of a *reaction function*: how much a city moves when its neighbors move. A positive $\rho$ describes mimicking, as when cities follow each other's rates. This is an idea from the economics literature on strategic interaction [@besley1995; @brueckner2003], and it is a good example of a model where the dependence is itself the object of interest, not a nuisance.

Collect the $\mathbf{Y}$ terms and solve:

$$
(\mathbf{I}_n - \rho\mathbf{W})\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} \quad\Longrightarrow\quad \mathbf{Y} = \boldsymbol{\Phi}(\rho)\left(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}\right)
$$

By Tool 1 this has mean and covariance

$$
\operatorname{E}[\mathbf{Y}] = \boldsymbol{\Phi}\mathbf{X}\boldsymbol{\beta}, \qquad \operatorname{Var}(\mathbf{Y}) = \sigma^2\boldsymbol{\Phi}\boldsymbol{\Phi}'
$$

The covariance is exactly the same matrix as in the error model. What differs is the mean. In the error model the mean is $\mathbf{X}\boldsymbol{\beta}$. In the lag model the mean is $\boldsymbol{\Phi}\mathbf{X}\boldsymbol{\beta}$: a region's expected value depends on its neighbors' covariates and on their neighbors' covariates, because those affect its neighbors' values, which affect it. This is why the two models call for different interpretations, and why the data can in principle tell them apart (Section 7.6).

### 7.3.2 The Feedback Multiplier as a Series

A tool we need: the matrix version of a geometric series.

For a number $r$ with $\lvert r\rvert < 1$, $1/(1 - r) = 1 + r + r^2 + \cdots$. The same thing holds for matrices. To see why, multiply the partial sum by $\mathbf{I} - \rho\mathbf{W}$; the terms telescope:

$$
(\mathbf{I} - \rho\mathbf{W})\left(\mathbf{I} + \rho\mathbf{W} + \rho^2\mathbf{W}^2 + \cdots + \rho^K\mathbf{W}^K\right) = \mathbf{I} - \rho^{K+1}\mathbf{W}^{K+1}
$$

Each term $\rho^k\mathbf{W}^k$ in the sum is cancelled by the next one except the first and the last. If $\rho^{K+1}\mathbf{W}^{K+1} \to \mathbf{0}$ as $K$ grows, the right side tends to $\mathbf{I}$, which says the sum converges to the inverse:

$$
\boldsymbol{\Phi}(\rho) = (\mathbf{I}_n - \rho\mathbf{W})^{-1} = \mathbf{I}_n + \rho\mathbf{W} + \rho^2\mathbf{W}^2 + \cdots
$$

This series is called the **Neumann series**. When does the condition hold? The eigenvalues of $\mathbf{W}$ lie between $-1$ and 1 (Section 6.3.3 proved this), so the eigenvalues of $\rho^k\mathbf{W}^k$ are at most $\lvert\rho\rvert^k$ in absolute value and they shrink to zero whenever $\lvert\rho\rvert < 1$.

The series has a clear meaning. Read $\boldsymbol{\Phi}\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon} + \rho\mathbf{W}\boldsymbol{\varepsilon} + \rho^2\mathbf{W}^2\boldsymbol{\varepsilon} + \cdots$ as a story about one shock. The first term is the shock where it happens. The second, $\rho\mathbf{W}\boldsymbol{\varepsilon}$, is the part passed to each region's neighbors: region $i$ receives $\rho$ times the average of its neighbors' shocks. The third, $\rho^2\mathbf{W}^2\boldsymbol{\varepsilon}$, is the part passed to neighbors of neighbors, and some of it returns to the region where the shock started, since neighbors are mutual. The entries of $\mathbf{W}^k$ have a concrete interpretation: $(\mathbf{W}^k)_{ij}$ is the probability that a walker who moves, at each step, to a uniformly chosen neighbor, starting at region $i$, is at region $j$ after exactly $k$ steps. The feedback matrix adds up these walks with weights $\rho^k$.

**By hand on the four regions.** For the $2 \times 2$ block, the walker's two-step matrix is

$$
\mathbf{W}^2 = \frac{1}{2}\begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \end{bmatrix}
$$

From region 1, the walker moves to region 2 or 3 and then, with probability one half each, back to region 1 or onward to region 4. Two steps later the walker is at region 1 with probability $1/2$ and at region 4 with probability $1/2$, and never at regions 2 or 3. Applying $\mathbf{W}$ to $\mathbf{W}^2$ gives back $\mathbf{W}$, so $\mathbf{W}^3 = \mathbf{W}$, then $\mathbf{W}^4 = \mathbf{W}^2$, and so on: odd powers equal $\mathbf{W}$ and even powers (from 2 up) equal $\mathbf{W}^2$. Grouping the series by parity,

$$
\boldsymbol{\Phi}(\rho) = \mathbf{I} + (\rho + \rho^3 + \cdots)\mathbf{W} + (\rho^2 + \rho^4 + \cdots)\mathbf{W}^2 = \mathbf{I} + \frac{\rho}{1 - \rho^2}\mathbf{W} + \frac{\rho^2}{1 - \rho^2}\mathbf{W}^2
$$

For $\rho = 0.5$ the two fractions are $0.5/0.75 = 2/3$ and $0.25/0.75 = 1/3$, so $\boldsymbol{\Phi} = \mathbf{I} + \tfrac{2}{3}\mathbf{W} + \tfrac{1}{3}\mathbf{W}^2$. The diagonal entry is $1 + \tfrac{1}{3}\cdot\tfrac{1}{2} = \tfrac{7}{6}$, the entry for a neighbor is $\tfrac{2}{3}\cdot\tfrac{1}{2} = \tfrac{1}{3}$, and the entry for the region two steps away is $\tfrac{1}{3}\cdot\tfrac{1}{2} = \tfrac{1}{6}$. That is the matrix $\tfrac{1}{6}(7, 2, 2, 1)$ used in Section 7.2.3.

The diagonal entry builds up as the series is summed: $1$ from the identity, nothing added at the first power (a region is not its own neighbor), $1 + 0.25 \times 0.5 = 1.125$ with the second, still $1.125$ with the third, $1.15625$ with the fourth, and so on toward $7/6 = 1.1667$. Each row of $\boldsymbol{\Phi}$ sums to $7/6 + 2/6 + 2/6 + 1/6 = 2 = 1/(1 - \rho)$. This is no accident: because $\mathbf{W}\mathbf{1} = \mathbf{1}$, we have $\boldsymbol{\Phi}\mathbf{1} = (1 + \rho + \rho^2 + \cdots)\mathbf{1} = \mathbf{1}/(1 - \rho)$ for every connected map. Section 7.8.1 prints the partial sums.

### 7.3.3 Why OLS Fails in the Lag Model

OLS fails in two ways, depending on how it is used. Both are easy to see from the reduced form.

**Leaving out the lag.** Suppose the analyst ignores the spatial structure and regresses $\mathbf{Y}$ on $\mathbf{X}$ alone. Since $\operatorname{E}[\mathbf{Y}] = \boldsymbol{\Phi}\mathbf{X}\boldsymbol{\beta}$,

$$
\operatorname{E}[\hat{\boldsymbol{\beta}}_{\text{OLS}}] = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Phi}\mathbf{X}\boldsymbol{\beta} \ne \boldsymbol{\beta}
$$

unless $\rho = 0$. This is omitted-variable bias: the true model contains the neighbors' average $\mathbf{W}\mathbf{Y}$, which is correlated with $\mathbf{X}$ because covariates are themselves spatially smooth, and leaving it out loads its effect onto $\mathbf{X}$. For a smooth covariate the slope absorbs part of the spillover. The lab shows the exact mean slope of $2.68$ mills against a true $\beta_1$ of $1.5$ mills.

**Including the lag.** The natural fix is to include $\mathbf{W}\mathbf{Y}$ as a regressor, giving OLS of $\mathbf{Y}$ on $\mathbf{X}_L = [\mathbf{W}\mathbf{Y},\ \mathbf{X}]$. This is also inconsistent, for a different reason: the new regressor is correlated with the disturbance. Since $\mathbf{W}\mathbf{Y} = \mathbf{W}\boldsymbol{\Phi}(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon})$ and $\mathbf{X}\boldsymbol{\beta}$ is fixed,

$$
\operatorname{Cov}(\mathbf{W}\mathbf{Y}, \boldsymbol{\varepsilon}) = \mathbf{W}\boldsymbol{\Phi}\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{W}\boldsymbol{\Phi}
$$

Its diagonal entries are what matter: they are the covariance between $\varepsilon_i$ and the regressor $(\mathbf{W}\mathbf{Y})_i$ in the same equation. Use the series:

$$
\mathbf{W}\boldsymbol{\Phi} = \mathbf{W} + \rho\mathbf{W}^2 + \rho^2\mathbf{W}^3 + \cdots, \qquad (\mathbf{W}\boldsymbol{\Phi})_{ii} = \sum_{k \ge 1}\rho^{k-1}(\mathbf{W}^k)_{ii}
$$

The first term is zero, since $W_{ii} = 0$. The second is $\rho(\mathbf{W}^2)_{ii} = \rho\sum_j w_{ij}w_{ji}$, which is positive when neighbors are mutual, as they are on a map. So for $\rho > 0$ the covariance is positive. Intuitively, a shock $\varepsilon_i$ raises $Y_i$, which raises its neighbors' values, which raises $(\mathbf{W}\mathbf{Y})_i$ itself, so the regressor contains a piece of the disturbance. Least squares attributes that piece to the coefficient of $\mathbf{W}\mathbf{Y}$ and overstates $\rho$.

How large is the effect? For a regression of $\mathbf{Y}$ on $\mathbf{X}_L$, the error in the estimate is $(\mathbf{X}_L'\mathbf{X}_L)^{-1}\mathbf{X}_L'\boldsymbol{\varepsilon}$. Replacing each factor by its expectation gives an approximation to the bias that becomes exact as $n$ grows:

$$
\text{bias} \approx \left(\operatorname{E}[\mathbf{X}_L'\mathbf{X}_L]\right)^{-1}\operatorname{E}[\mathbf{X}_L'\boldsymbol{\varepsilon}], \qquad \operatorname{E}[\mathbf{X}_L'\boldsymbol{\varepsilon}] = \begin{bmatrix} \sigma^2\operatorname{tr}(\mathbf{W}\boldsymbol{\Phi}) \\ \mathbf{0} \end{bmatrix}
$$

The lower block is zero because $\mathbf{X}$ is not random and $\operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}$. In the four-region example, $\mathbf{W}\boldsymbol{\Phi} = \tfrac{4}{3}\mathbf{W} + \tfrac{2}{3}\mathbf{W}^2$ (using $\mathbf{W}^3 = \mathbf{W}$), whose diagonal is $\tfrac{2}{3}\cdot\tfrac{1}{2} = \tfrac{1}{3}$. Each region's regressor carries covariance $\sigma^2/3$ with its own disturbance, an amount comparable to the size of the disturbance itself.

The bias also spreads to the other coefficients, through the inverse of $\operatorname{E}[\mathbf{X}_L'\mathbf{X}_L]$: when $\rho$ is overstated, the coefficients on $\mathbf{X}$ are understated, because the lag regressor has taken credit for part of what $\mathbf{X}$ explains. In the lab with $\rho = 0.6$, the formula predicts a bias of $+0.148$ in $\hat{\rho}$ and $-0.290$ in the slope; the simulated biases are $+0.128$ and $-0.259$.

Neither version of OLS works. The remedy is maximum likelihood, which accounts for the dependence between $\mathbf{W}\mathbf{Y}$ and $\boldsymbol{\varepsilon}$ through a term that OLS leaves out (Section 7.5.1).

---

## 7.4 The Parameter Space

The reduced forms need $\mathbf{I}_n - \rho\mathbf{W}$ to be invertible. This section works out for which values of $\rho$ it is, using the eigenvalues of $\mathbf{W}$.

**A tool we need: the determinant and the eigenvalues.** If $\mathbf{W}\mathbf{v} = \lambda\mathbf{v}$, then $(\mathbf{I} - \rho\mathbf{W})\mathbf{v} = \mathbf{v} - \rho\lambda\mathbf{v} = (1 - \rho\lambda)\mathbf{v}$. So the eigenvectors of $\mathbf{I} - \rho\mathbf{W}$ are those of $\mathbf{W}$, and its eigenvalues are $1 - \rho\lambda_i$. The determinant of a matrix is the product of its eigenvalues, so

$$
\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert = \prod_{i=1}^n(1 - \rho\lambda_i)
$$

A matrix is invertible exactly when its determinant is not zero. So $\mathbf{I}_n - \rho\mathbf{W}$ is singular only at the values $\rho = 1/\lambda_i$ for eigenvalues $\lambda_i \ne 0$, and invertible everywhere else. For the four-region example the eigenvalues are $1, 0, 0, -1$, so the determinant is $(1 - \rho)(1)(1)(1 + \rho) = 1 - \rho^2$, singular only at $\rho = \pm 1$. At $\rho = 0.5$ it is $0.75$.

**What we know about the eigenvalues of $\mathbf{W}$.** For a row-standardized $\mathbf{W}$ on a connected map:

- $\lambda_{\max} = 1$. The vector $\mathbf{1}$ satisfies $\mathbf{W}\mathbf{1} = \mathbf{1}$, so 1 is an eigenvalue.
- All eigenvalues satisfy $\lvert\lambda\rvert \le 1$, by the largest-entry argument of Section 6.3.3 (the weights are nonnegative and sum to 1 in each row).
- $\lambda_{\min}$ lies between $-1$ and $0$, and equals $-1$ exactly when the regions can be split into two groups so that every neighbor pair has one region in each group (a bipartite map). A grid of square regions is bipartite: color it like a checkerboard, and the checkerboard pattern is an eigenvector with eigenvalue $-1$.

For the county map used in this chapter, the lab confirms that the smallest eigenvalue is $-1$ and the largest is $1$.

**The parameter space.** The interval

$$
\rho \in \left(\frac{1}{\lambda_{\min}},\ \frac{1}{\lambda_{\max}}\right)
$$

is the conventional parameter space: the largest interval around zero on which $\mathbf{I}_n - \rho\mathbf{W}$ stays invertible. For a row-standardized $\mathbf{W}$ the upper end is $1$, and for the county map and the four-region block the lower end is $-1$, so $\rho \in (-1, 1)$. Three reasons make this the right interval:

- Inside it every factor $1 - \rho\lambda_i$ is positive: for $\lambda_i > 0$ this needs $\rho < 1/\lambda_i$, and for $\lambda_i < 0$ it needs $\rho > 1/\lambda_i$. So the determinant is positive and the log-likelihood of Section 7.5 is defined.
- It contains $\rho = 0$, where both models reduce to the classical linear model.
- Inside it the Neumann series converges, so the feedback matrix has the interpretation of Section 7.3.2 as a sum of walks.

Beyond a singular point the matrix is invertible again, but the model would have no connection to the classical model and no interpretation as echoes. In applied work with positive dependence, $\rho$ lies in $(0, 1)$, and values near 1 describe strong dependence, with the covariance growing without bound as $\rho \to 1$.

With a binary (not row-standardized) adjacency matrix, the largest eigenvalue can be much greater than 1 (up to the largest degree), so the interval for $\rho$ is much narrower, which is one reason row-standardizing is common.

---

## 7.5 Maximum Likelihood

### 7.5.1 The Likelihood and the Jacobian Term

A tool we need: changing variables in a density.

If $Y = a\varepsilon$ with $\varepsilon \sim N(0, 1)$, whose density we call $f$, and $a > 0$, the density of $Y$ is not $f(y/a)$. It is $f(y/a)/a$, because stretching the axis by a factor $a$ spreads the same probability over a longer interval and so lowers the density by the factor $a$. For a vector, $\boldsymbol{\varepsilon}$ is a function of $\mathbf{Y}$, and the density of $\mathbf{Y}$ equals the density of $\boldsymbol{\varepsilon}$ at that function, times the absolute value of the determinant of the matrix of derivatives $\partial\boldsymbol{\varepsilon}/\partial\mathbf{Y}$. This determinant is the **Jacobian**: the factor by which the transformation stretches volume.

For the lag model, solve the equation for the innovations: $\boldsymbol{\varepsilon} = (\mathbf{I}_n - \rho\mathbf{W})\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}$. The matrix of derivatives of $\boldsymbol{\varepsilon}$ with respect to $\mathbf{Y}$ is $\mathbf{I}_n - \rho\mathbf{W}$. Since $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$, the log-likelihood is

$$
\ell(\boldsymbol{\beta}, \rho, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) + \log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert - \frac{1}{2\sigma^2}\boldsymbol{\varepsilon}'\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} = (\mathbf{I}_n - \rho\mathbf{W})\mathbf{y} - \mathbf{X}\boldsymbol{\beta}
$$

For the error model, $\boldsymbol{\varepsilon} = (\mathbf{I}_n - \rho\mathbf{W})(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$, the derivative matrix is again $\mathbf{I}_n - \rho\mathbf{W}$, and the same formula holds with this $\boldsymbol{\varepsilon}$.

**Agreement with the general formula.** The error model is a multivariate normal with covariance $\boldsymbol{\Sigma} = \sigma^2\boldsymbol{\Phi}\boldsymbol{\Phi}'$, so Chapter 5's log-likelihood applies. Check that the two agree. The determinant is $\lvert\boldsymbol{\Sigma}\rvert = \sigma^{2n}\lvert\boldsymbol{\Phi}\rvert^2 = \sigma^{2n}\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert^{-2}$, so $-\tfrac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert = -\tfrac{n}{2}\log\sigma^2 + \log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert$, and the quadratic form $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ equals $\boldsymbol{\varepsilon}'\boldsymbol{\varepsilon}/\sigma^2$. The Jacobian term is exactly the log-determinant term of Chapter 5, written in a convenient form.

**Why the Jacobian matters.** Drop it, and what remains is $-\boldsymbol{\varepsilon}'\boldsymbol{\varepsilon}$: maximizing the likelihood without the Jacobian is least squares. For the lag model, $\boldsymbol{\varepsilon}'\boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} = \mathbf{y} - \rho\mathbf{W}\mathbf{y} - \mathbf{X}\boldsymbol{\beta}$ is the sum of squares of OLS of $\mathbf{y}$ on $[\mathbf{W}\mathbf{y}, \mathbf{X}]$, the biased estimator of Section 7.3.3. So maximum likelihood equals that regression plus the term $\log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert$, and that term corrects the bias. It is $\sum_i\log(1 - \rho\lambda_i)$, which is zero at $\rho = 0$, negative elsewhere, and falls steeply as $\rho$ approaches 1 ($-23.5$ at $\rho = 0.9$ on the county map), so it pulls the estimate down from the value least squares would choose.

**A hand-sized example.** On the four-region map take $\mathbf{y} = (2, 1, 3, 2)'$, $\mathbf{X} = \mathbf{1}$, and the parameter values $\beta = 1$, $\rho = 0.5$, $\sigma^2 = 1$, and evaluate the lag log-likelihood. From Section 7.2.4, $(\mathbf{I} - 0.5\mathbf{W})\mathbf{y} = (1, 0, 2, 1)'$, so $\boldsymbol{\varepsilon} = (1, 0, 2, 1)' - (1, 1, 1, 1)' = (0, -1, 1, 0)'$ and $\boldsymbol{\varepsilon}'\boldsymbol{\varepsilon} = 2$. The Jacobian term is $\log 0.75 = -0.2877$. So

$$
\ell = -\frac{4}{2}\log(2\pi) - 0.2877 - \frac{2}{2} = -3.6758 - 0.2877 - 1 = -4.9634
$$

The lab evaluates the same quantity from the multivariate normal density with mean $\boldsymbol{\Phi}\mathbf{1}$ and covariance $\boldsymbol{\Phi}\boldsymbol{\Phi}'$, and gets $-4.9634$.

### 7.5.2 Ord's Eigenvalue Identity

The log-likelihood needs $\log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert$ at every value of $\rho$ the optimizer tries. Computing a determinant of an $n \times n$ matrix costs on the order of $n^3$ operations, and the optimizer may try dozens of values. Section 7.4 gives the way out. The determinant is the product of the eigenvalues $1 - \rho\lambda_i$, so

$$
\log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert = \sum_{i=1}^n\log(1 - \rho\lambda_i)
$$

This is **Ord's identity** [@ord1975]. Once the eigenvalues $\lambda_i$ of $\mathbf{W}$ are computed, each new value of $\rho$ costs only $n$ logarithms. The eigenvalues themselves are found once, at a cost of order $n^3$, and that is made faster and more stable by computing them from the symmetric matrix $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, which has the same eigenvalues as $\mathbf{W}$ (Section 6.3.3) and for which symmetric eigenvalue routines apply. For the four-region example the identity gives $\log 0.5 + 2\log 1 + \log 1.5 = \log 0.75$. The lab checks it on the county map, where $\log\lvert\mathbf{I} - 0.9\mathbf{W}\rvert = -23.5143$ both ways.

For maps with tens of thousands of regions even one eigendecomposition is too expensive. Chapter 8 and Chapter 13 use sparse matrix factorizations instead.

### 7.5.3 Profiling Out $\boldsymbol{\beta}$ and $\sigma^2$

As in Chapter 5, $\boldsymbol{\beta}$ and $\sigma^2$ can be removed from the likelihood, leaving a one-dimensional problem in $\rho$.

**The lag model.** For fixed $\rho$, the log-likelihood in $\boldsymbol{\beta}$ is minus a multiple of $\lVert(\mathbf{y} - \rho\mathbf{W}\mathbf{y}) - \mathbf{X}\boldsymbol{\beta}\rVert^2$, so the maximizer is the OLS regression of $\mathbf{y} - \rho\mathbf{W}\mathbf{y}$ on $\mathbf{X}$:

$$
\hat{\boldsymbol{\beta}}(\rho) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'(\mathbf{y} - \rho\mathbf{W}\mathbf{y}) = \hat{\boldsymbol{\beta}}_y - \rho\,\hat{\boldsymbol{\beta}}_{Wy}
$$

where $\hat{\boldsymbol{\beta}}_y$ and $\hat{\boldsymbol{\beta}}_{Wy}$ are the OLS coefficients from regressing $\mathbf{y}$ and $\mathbf{W}\mathbf{y}$ on $\mathbf{X}$. The residuals are linear in $\rho$ in the same way:

$$
\mathbf{e}(\rho) = \mathbf{e}_y - \rho\,\mathbf{e}_{Wy}
$$

where $\mathbf{e}_y$ and $\mathbf{e}_{Wy}$ are the OLS residual vectors of those two regressions. Both are computed once. Then $\hat{\sigma}^2(\rho) = \mathbf{e}(\rho)'\mathbf{e}(\rho)/n$, and substituting into the log-likelihood (exactly as in Section 5.2.2, the exponent term becomes $-n/2$) gives the **profile log-likelihood**

$$
\ell_p(\rho) = -\frac{n}{2}\left(\log(2\pi) + 1\right) - \frac{n}{2}\log\hat{\sigma}^2(\rho) + \sum_{i=1}^n\log(1 - \rho\lambda_i)
$$

Each evaluation costs order $n$: a vector update, one dot product, and $n$ logarithms.

**The error model.** For fixed $\rho$, the filtered regression of Section 7.2.4 gives $\hat{\boldsymbol{\beta}}(\rho) = (\mathbf{X}^{*\prime}\mathbf{X}^*)^{-1}\mathbf{X}^{*\prime}\mathbf{Y}^*$, with residual $\mathbf{e}(\rho) = \mathbf{y}^* - \mathbf{X}^*\hat{\boldsymbol{\beta}}(\rho)$ and the same formula for $\ell_p$. The filtered data are cheap to form, since $\mathbf{y}^* = \mathbf{y} - \rho\mathbf{W}\mathbf{y}$ and $\mathbf{X}^* = \mathbf{X} - \rho\mathbf{W}\mathbf{X}$ use products $\mathbf{W}\mathbf{y}$ and $\mathbf{W}\mathbf{X}$ computed once.

**The algorithm.** Maximize $\ell_p$ over $\rho$ in the interval of Section 7.4 with a bounded one-dimensional search, then recover $\hat{\boldsymbol{\beta}}(\hat{\rho})$ and $\hat{\sigma}^2(\hat{\rho})$. Standard errors come from the observed information of the full log-likelihood in $(\boldsymbol{\beta}, \rho, \sigma^2)$, computed here by finite differences. Section 7.8.4 implements all of this.

**Testing $\rho = 0$.** Both models reduce to the classical linear model at $\rho = 0$, and $\rho = 0$ is an interior point of the parameter space $(-1, 1)$, so the ordinary likelihood ratio test applies: $2(\hat{\ell} - \ell_{\text{OLS}})$ is approximately $\chi^2_1$ under the null. This differs from the boundary tests of Section 5.5.2, where the null value lay on the edge of the parameter space.

---

## 7.6 Choosing Between the Error and Lag Models

Suppose a researcher finds that the tax rates of neighboring cities are correlated after accounting for the cities' own characteristics. Two explanations are on the table. Cities may react to each other (a lag model), or they may share unobserved influences such as regional economic conditions (an error model). Both produce the same covariance matrix for $\mathbf{Y}$. They differ in the mean, and in the coefficients' interpretation. Which model describes the data matters for what the researcher reports.

The standard workflow starts from OLS, which is quick to fit and needs only residuals, and uses tests that ask whether adding a spatial term to OLS would improve the fit.

### 7.6.1 The Idea of a Score Test

**A tool we need: the score test.** A test of a null hypothesis can look at the fitted alternative, as the likelihood ratio test does, or at the slope of the log-likelihood at the null value. If the null value is the maximum of the log-likelihood, the slope there is zero. If the data fit the null poorly, the log-likelihood is still climbing at the null value, and the slope there is large. The slope is called the **score**, and a test based on it needs only the null fit, which here is OLS. A one-parameter example: for $Y_i \sim N(\mu, 1)$ and the null $\mu = 0$, the score at the null is $\sum_i y_i$, its variance under the null is $n$, and the score test statistic $(\sum_i y_i)^2/n$ is the square of the usual $z$ statistic. In general, the statistic is the score squared divided by the information (the variance of the score), and under the null it is approximately $\chi^2_1$. In the spatial setting these tests are called **Lagrange multiplier** (LM) tests, the name used in econometrics for the same construction [@anselin1988; @anselin1996].

### 7.6.2 The Four Statistics

All the ingredients come from the OLS fit. Let $\mathbf{e}$ be the OLS residuals, $s^2 = \mathbf{e}'\mathbf{e}/n$, and define

$$
g_{\text{err}} = \frac{\mathbf{e}'\mathbf{W}\mathbf{e}}{s^2}, \qquad g_{\text{lag}} = \frac{\mathbf{e}'\mathbf{W}\mathbf{y}}{s^2}
$$

The first measures whether residuals correlate with their neighbors' residuals: for a row-standardized $\mathbf{W}$ it equals $n$ times Moran's $I$ computed on the residuals (Section 6.6). The second measures whether the residuals correlate with the neighbors' responses. Two constants scale them:

$$
T_W = \operatorname{tr}(\mathbf{W}^2 + \mathbf{W}'\mathbf{W}), \qquad J = \frac{(\mathbf{W}\mathbf{X}\hat{\boldsymbol{\beta}})'(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})(\mathbf{W}\mathbf{X}\hat{\boldsymbol{\beta}})}{s^2} + T_W
$$

where $\mathbf{P}_{\mathbf{X}}$ is the projection of Section 1.1.2. The constant $T_W$ depends only on the map. The term in $J$ involving $\mathbf{W}\mathbf{X}\hat{\boldsymbol{\beta}}$ is the part of the neighbors' fitted values that the covariates cannot explain; it will be zero if $\mathbf{W}\mathbf{X}\hat{\boldsymbol{\beta}}$ lies in the span of $\mathbf{X}$. The four statistics are

$$
\begin{aligned}
\text{LM}_{\text{err}} &= \frac{g_{\text{err}}^2}{T_W}, & \text{LM}_{\text{lag}} &= \frac{g_{\text{lag}}^2}{J} \\
\text{RLM}_{\text{err}} &= \frac{\left(g_{\text{err}} - (T_W/J)\,g_{\text{lag}}\right)^2}{T_W - T_W^2/J}, & \text{RLM}_{\text{lag}} &= \frac{(g_{\text{lag}} - g_{\text{err}})^2}{J - T_W}
\end{aligned}
$$

Each is approximately $\chi^2_1$ under its null. Section 7.6.4 derives them as score tests (it can be skipped).

**Why four?** $\text{LM}_{\text{err}}$ tests for spatially correlated errors, and $\text{LM}_{\text{lag}}$ for a spatial lag, each assuming the other is absent. But each has power against the other alternative too. If the truth is a lag model, the residuals of OLS are spatially correlated, so $\text{LM}_{\text{err}}$ also rejects, and vice versa. The **robust** versions, $\text{RLM}_{\text{err}}$ and $\text{RLM}_{\text{lag}}$, adjust each score for the presence of the other kind of dependence, so that $\text{RLM}_{\text{lag}}$ tests for a lag while allowing for error dependence and $\text{RLM}_{\text{err}}$ tests for error dependence while allowing for a lag.

**The decision rule.** A common workflow follows these steps:

1. If neither $\text{LM}_{\text{err}}$ nor $\text{LM}_{\text{lag}}$ is significant, keep OLS.
2. If only one is significant, fit that model.
3. If both are significant, compare the robust statistics. Fit the model whose robust statistic is significant, and if both are, the one with the larger robust statistic.

The lab shows the rule at work: on data simulated from a lag model, $\text{LM}_{\text{err}}$ is 16.30 and $\text{LM}_{\text{lag}}$ is 40.43, both far beyond the $\chi^2_1$ cutoff of 3.84. The robust statistics separate them: $\text{RLM}_{\text{err}}$ is 0.19 (p-value 0.662, not significant) and $\text{RLM}_{\text{lag}}$ is 24.32 (p-value $8.2 \times 10^{-7}$).

### 7.6.3 Likelihood Ratio Tests and AIC

Once both models are fit by ML, they can be compared directly. Each reduces to OLS at $\rho = 0$, so each can be tested against OLS with the likelihood ratio test of Section 7.5.3. The two models are not nested in each other: neither is a special case of the other, so a likelihood ratio test between them is not available. Both have the same number of parameters ($p + 2$: the $p$ coefficients, $\rho$, and $\sigma^2$), so AIC, which equals $-2\hat{\ell}$ plus twice the number of parameters (Section 5.5.1), reduces to comparing the maximized log-likelihoods. The model with the larger maximized log-likelihood is preferred. In the lab, on lag-model data, the lag model has log-likelihood $-200.06$ and the error model $-215.34$, a difference of about 15 log-likelihood units.

A model that contains both as special cases adds the neighbors' covariates $\mathbf{W}\mathbf{X}$ to the lag model. It is known as the spatial Durbin model, and it can be tested against each of the two simpler ones with likelihood ratio tests. We do not develop it here: Exercise 11 shows how it reduces to the error model under one restriction on its coefficients, and Appendix D points to its estimation.

### 7.6.4 Deriving the LM Statistics (Optional)

*This subsection derives the four statistics of Section 7.6.2 as score tests. It uses only the log-likelihood of Section 7.5.1 and the moments of quadratic forms, and it can be skipped on a first reading.*

Start with the lag model's log-likelihood. Its derivative with respect to $\rho$ has two parts. The log-determinant has derivative $-\operatorname{tr}\!\left(\boldsymbol{\Phi}(\rho)\mathbf{W}\right)$, and the quadratic term has derivative $\boldsymbol{\varepsilon}'\mathbf{W}\mathbf{y}/\sigma^2$, since $\partial\boldsymbol{\varepsilon}/\partial\rho = -\mathbf{W}\mathbf{y}$. So

$$
\frac{\partial\ell}{\partial\rho} = -\operatorname{tr}\!\left(\boldsymbol{\Phi}(\rho)\mathbf{W}\right) + \frac{1}{\sigma^2}\boldsymbol{\varepsilon}'\mathbf{W}\mathbf{y}
$$

At the null $\rho = 0$, $\boldsymbol{\Phi} = \mathbf{I}$ and $\operatorname{tr}(\mathbf{W}) = 0$ because $\mathbf{W}$ has a zero diagonal. The first part vanishes, and with $\boldsymbol{\varepsilon}$ replaced by the OLS residuals $\mathbf{e}$ the **score** is $\mathbf{e}'\mathbf{W}\mathbf{y}/s^2 = g_{\text{lag}}$. For the error model, $\boldsymbol{\varepsilon} = (\mathbf{I} - \rho\mathbf{W})(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ and the same steps give a score $\mathbf{e}'\mathbf{W}\mathbf{e}/s^2 = g_{\text{err}}$ at $\rho = 0$. (Here $\partial\boldsymbol{\varepsilon}/\partial\rho = -\mathbf{W}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$, and the quadratic term gives $\boldsymbol{\varepsilon}'\mathbf{W}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})/\sigma^2$.)

*The information.* The variance of a score is the information. At the null, $\boldsymbol{\varepsilon}$ equals the true innovations $\boldsymbol{\varepsilon}_0 \sim N(\mathbf{0}, \sigma^2\mathbf{I})$, and the variance of a quadratic form in independent normals is $\operatorname{Var}(\boldsymbol{\varepsilon}_0'\mathbf{A}\boldsymbol{\varepsilon}_0) = \sigma^4\operatorname{tr}(\mathbf{A}^2 + \mathbf{A}\mathbf{A}')$. For $\mathbf{A} = \mathbf{W}$,

$$
\operatorname{Var}\!\left(\frac{\boldsymbol{\varepsilon}_0'\mathbf{W}\boldsymbol{\varepsilon}_0}{\sigma^2}\right) = \operatorname{tr}(\mathbf{W}^2 + \mathbf{W}\mathbf{W}') = T_W
$$

where $\operatorname{tr}(\mathbf{W}\mathbf{W}') = \operatorname{tr}(\mathbf{W}'\mathbf{W})$ by the cyclic property of the trace. So the information for the error parameter is $T_W$. The score and the regression coefficients are uncorrelated at the null (the score involves third moments of the normal innovations, which vanish), so there is no adjustment for estimating $\boldsymbol{\beta}$, and $\text{LM}_{\text{err}} = g_{\text{err}}^2/T_W$.

For the lag model the score has an extra piece, because at the null $\mathbf{W}\mathbf{y} = \mathbf{W}\mathbf{X}\boldsymbol{\beta} + \mathbf{W}\boldsymbol{\varepsilon}_0$. The score is $\boldsymbol{\varepsilon}_0'\mathbf{W}\mathbf{X}\boldsymbol{\beta}/\sigma^2 + \boldsymbol{\varepsilon}_0'\mathbf{W}\boldsymbol{\varepsilon}_0/\sigma^2$. The two pieces are uncorrelated, so the variance adds up to $(\mathbf{W}\mathbf{X}\boldsymbol{\beta})'(\mathbf{W}\mathbf{X}\boldsymbol{\beta})/\sigma^2 + T_W$. Now the score is correlated with the score for $\boldsymbol{\beta}$, which is $\mathbf{X}'\boldsymbol{\varepsilon}_0/\sigma^2$, with covariance $(\mathbf{W}\mathbf{X}\boldsymbol{\beta})'\mathbf{X}/\sigma^2$. When a nuisance parameter is estimated, the correct information for $\rho$ subtracts the part explained by its score: information minus covariance times the inverse of the information of $\boldsymbol{\beta}$ (which is $\mathbf{X}'\mathbf{X}/\sigma^2$) times covariance. This gives

$$
\frac{(\mathbf{W}\mathbf{X}\boldsymbol{\beta})'\left(\mathbf{I} - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\right)(\mathbf{W}\mathbf{X}\boldsymbol{\beta})}{\sigma^2} + T_W = J
$$

after replacing the unknown parameters by their OLS estimates. So $\text{LM}_{\text{lag}} = g_{\text{lag}}^2/J$. (The scores for $\sigma^2$ and $\rho$ are uncorrelated at the null because $\operatorname{tr}(\mathbf{W}) = 0$, so $\sigma^2$ needs no adjustment.)

*The robust versions.* Consider both parameters at once, $\rho_{\text{lag}}$ and $\rho_{\text{err}}$. Their scores at the null are $g_{\text{lag}}$ and $g_{\text{err}}$. Both contain the same quadratic form $\boldsymbol{\varepsilon}_0'\mathbf{W}\boldsymbol{\varepsilon}_0/\sigma^2$, so their covariance is its variance, $T_W$, and the information matrix for the pair is

$$
\begin{bmatrix} J & T_W \\ T_W & T_W \end{bmatrix}
$$

To test the lag parameter while allowing for the error parameter, subtract from $g_{\text{lag}}$ the part of it predictable from $g_{\text{err}}$, which is $(T_W/T_W)\,g_{\text{err}} = g_{\text{err}}$, leaving $g_{\text{lag}} - g_{\text{err}}$ with variance $J - T_W^2/T_W = J - T_W$. That is $\text{RLM}_{\text{lag}}$. Reversing the roles, subtracting $(T_W/J)\,g_{\text{lag}}$ from $g_{\text{err}}$ leaves variance $T_W - T_W^2/J$, which is $\text{RLM}_{\text{err}}$. The lab confirms that the four statistics have the claimed behavior when the truth is no spatial dependence, a lag, or an error.

---

## 7.7 Interpreting the Lag Model: Direct, Indirect, and Total Impacts

### 7.7.1 Why $\beta_k$ Is Not the Marginal Effect

Return to the story of regional research and development. Suppose the response is patents granted in region $i$, and one covariate is the region's R&D spending. In a lag model, a region with more R&D produces more patents. Its neighbors, drawing on its findings, also produce more. And the neighbors' extra patents spill back. A researcher who reports $\beta_k$ as "the effect of one more dollar of R&D" has reported only the first of these.

Make this precise. In the lag model $\operatorname{E}[\mathbf{Y}] = \boldsymbol{\Phi}\mathbf{X}\boldsymbol{\beta}$, so the expected response in region $i$ is $\sum_j\Phi_{ij}\,(\mathbf{X}\boldsymbol{\beta})_j$. Raise covariate $k$ in region $j$ by one unit. The expected response in region $i$ changes by

$$
\frac{\partial\operatorname{E}[Y_i]}{\partial x_{jk}} = \Phi_{ij}\,\beta_k
$$

for every region $i$. These are the entries of the **effect matrix** $\boldsymbol{\Phi}\beta_k$. Its diagonal entries are the effect of raising the covariate in a region on that same region, and its off-diagonal entries are the spillovers. In the error model there is no feedback, $\operatorname{E}[\mathbf{Y}] = \mathbf{X}\boldsymbol{\beta}$, and the effect matrix is just $\beta_k\mathbf{I}$: here $\beta_k$ is the marginal effect.

The effect matrix has $n^2$ entries, and different regions have different effects, since $\Phi_{ij}$ depends on where $i$ and $j$ are on the map. Reporting needs a summary.

### 7.7.2 Direct, Indirect, and Total Impacts

The standard summaries average the effect matrix in three ways [@lesage2009].

- The **direct impact** is the average of the diagonal: the average effect of a change in a region on the same region, $M_{\text{dir}} = \beta_k\operatorname{tr}(\boldsymbol{\Phi})/n$.
- The **total impact** is the average row sum: the average total effect on a region of raising the covariate by one unit in *all* regions at once, $M_{\text{tot}} = \beta_k\mathbf{1}'\boldsymbol{\Phi}\mathbf{1}/n$. The average of the column sums is the same number: it is the average total effect on the whole map of raising the covariate in one region.
- The **indirect impact** is the difference, $M_{\text{ind}} = M_{\text{tot}} - M_{\text{dir}}$: the average spillover.

For a row-standardized $\mathbf{W}$ the total impact has a simple form. We showed $\boldsymbol{\Phi}\mathbf{1} = \mathbf{1}/(1 - \rho)$ in Section 7.3.2, so every row sum is $1/(1 - \rho)$ and

$$
M_{\text{tot}} = \frac{\beta_k}{1 - \rho}
$$

For the direct impact, the trace of $\boldsymbol{\Phi}$ is the sum of its eigenvalues, which are $1/(1 - \rho\lambda_i)$:

$$
M_{\text{dir}} = \frac{\beta_k}{n}\sum_{i=1}^n\frac{1}{1 - \rho\lambda_i}
$$

Both are cheap to compute from the eigenvalues found for Ord's identity.

**Reading the Neumann series.** Using $\boldsymbol{\Phi} = \mathbf{I} + \rho\mathbf{W} + \rho^2\mathbf{W}^2 + \cdots$, the average diagonal is $1 + \rho^2\cdot\overline{(\mathbf{W}^2)_{ii}} + \rho^4\cdot\overline{(\mathbf{W}^4)_{ii}} + \cdots$ (the odd powers have zero diagonal on a bipartite map, such as a grid): the direct impact is the own effect, 1, plus the part that returns after going out and coming back. The first spillover term is $\rho\mathbf{W}\beta_k$, whose average row sum is $\rho\beta_k$ because the rows of $\mathbf{W}$ sum to 1. On a bipartite map with $\rho > 0$ every later term is nonnegative, so the indirect impact is at least $\rho\beta_k$, and it grows as $\rho$ grows. When $\rho = 0$, the direct impact equals $\beta_k$ and the indirect impact is zero.

**By hand on the four-region map.** With $\rho = 0.5$, the trace of $\boldsymbol{\Phi}$ is $4 \times 7/6$, so the direct impact is $\tfrac{7}{6}\beta_k = 1.1667\beta_k$. The total impact is $\beta_k/(1 - 0.5) = 2\beta_k$. The indirect impact is $2 - \tfrac{7}{6} = \tfrac{5}{6}\beta_k = 0.8333\beta_k$. Using the eigenvalues $1, 0, 0, -1$ for the trace: $1/(1 - 0.5) + 1 + 1 + 1/(1 + 0.5) = 2 + 2 + 0.6667 = 4.6667$, and $4.6667/4 = 1.1667$, as before. A coefficient of $\beta_k = 2$ would mean a direct effect of $2.33$, a spillover of $1.67$, and a total of $4$.

**The OLS slope in between.** If the analyst ignores the lag, the slope of OLS estimates neither impact. Its mean is $\beta_k\,\mathbf{x}_c'\boldsymbol{\Phi}\mathbf{x}_c/\mathbf{x}_c'\mathbf{x}_c$ for a covariate centered at its mean, a weighted average of the feedback matrix's values over the patterns present in the covariate, in the same way the variance ratio of Section 1.3.4 was a weighted average of eigenvalues. The lab shows an example where this slope lies between the direct and total impacts.

### 7.7.3 Uncertainty in the Impacts

The impacts depend on $\hat{\rho}$ and $\hat{\beta}_k$, both uncertain, and they are nonlinear functions of them. A simple way to put intervals on them is by simulation. Draw many parameter vectors from a normal distribution with mean at the ML estimates and covariance equal to the inverse of the observed information of Section 7.5.3, keep draws with $\rho$ inside the parameter space, and compute the three impacts for each draw. Percentiles of the results give intervals. This is the approach used in the applied literature on spatial impacts, and Section 7.8.7 applies it.

### 7.7.4 Checking by Simulation

A direct check on the effect matrix: simulate a data set, then simulate it again with the same innovations after raising a covariate by one unit in one county, and compare. Because the model is linear in $\boldsymbol{\varepsilon}$ and $\mathbf{X}$, the difference between the two simulated responses is exactly $\boldsymbol{\Phi}\beta_k\mathbf{e}_j$ (with $\mathbf{e}_j$ the unit vector for county $j$). The lab does this for a county in the middle of the map and then for every county, and the averages of the own effects and the total effects reproduce the formulas for $M_{\text{dir}}$ and $M_{\text{tot}}$. Figure 7.3 shows how the effect of the change spreads across the map for three values of $\rho$.

---

## 7.8 Walkthrough and Lab 7

### 7.8.1 The Four Regions in Code

The first block checks the numbers of Sections 7.2 to 7.7 for the $2 \times 2$ block with $\rho = 0.5$: the eigenvalues of $\mathbf{W}$, the feedback matrix, the partial sums of the Neumann series, the covariance and precision, the covariance between $\mathbf{W}\mathbf{Y}$ and $\boldsymbol{\varepsilon}$, the impacts, the log-determinant, and the log-likelihood of Section 7.5.1, which is also computed from the multivariate normal density with mean $\boldsymbol{\Phi}\mathbf{1}$ and covariance $\boldsymbol{\Phi}\boldsymbol{\Phi}'$.

```python
A4 = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
              dtype=float)
W4 = A4 / A4.sum(axis=1, keepdims=True)
I4 = np.eye(4)
Phi4 = np.linalg.inv(I4 - rho * W4)
print("eigenvalues of W:", np.round(np.linalg.eigvalsh(W4), 4))
print("6 * Phi:\n", np.round(6 * Phi4, 4))

partial = np.zeros((4, 4))
term = I4.copy()
for k in range(8):
    partial += term
    print(f"terms 0..{k}: Phi[1,1] = {partial[0, 0]:.5f}  "
          f"largest error = {np.abs(partial - Phi4).max():.5f}")
    term = rho * term @ W4

Var_u = Phi4 @ Phi4.T
Q4 = (I4 - rho * W4).T @ (I4 - rho * W4)
print("Var(u) / sigma^2, row 1:", np.round(Var_u[0], 4))
print("Q * sigma^2, row 1:", np.round(Q4[0], 4))
print("Cov(WY_i, eps_i) / sigma^2:", np.round(np.diag(W4 @ Phi4), 4))
print("direct, total, indirect:", round(np.trace(Phi4) / 4, 4),
      round(Phi4.sum() / 4, 4),
      round(Phi4.sum() / 4 - np.trace(Phi4) / 4, 4))
sign, logdet = np.linalg.slogdet(I4 - rho * W4)
lam4 = np.linalg.eigvalsh(W4)
print("log|I - rho W|:", round(logdet, 4),
      round(np.log(1 - rho * lam4).sum(), 4))

y = np.array([2.0, 1.0, 3.0, 2.0])
eps = (I4 - rho * W4) @ y - 1.0
ll_direct = (-2 * np.log(2 * np.pi) + logdet - 0.5 * eps @ eps)
mvn = stats.multivariate_normal(mean=Phi4 @ np.ones(4),
                                cov=Phi4 @ Phi4.T)
print("log-likelihood:", round(ll_direct, 4), round(mvn.logpdf(y), 4))
```

The eigenvalues are $-1, 0, 0, 1$, and $6\boldsymbol{\Phi}$ has rows $(7, 2, 2, 1)$, $(2, 7, 1, 2)$, $(2, 1, 7, 2)$, $(1, 2, 2, 7)$. The partial sums of the diagonal entry are $1, 1, 1.125, 1.125, 1.15625, 1.15625, 1.16406, 1.16406$ for terms $0, \dots, 7$, and the largest error in any entry of the partial sum falls by half with every term: $0.3333, 0.1667, 0.0833, 0.0417, \dots, 0.0026$. The first row of $\operatorname{Var}(\mathbf{u})/\sigma^2$ is $(1.6111, 0.8889, 0.8889, 0.6111)$ and the first row of $\sigma^2\mathbf{Q}_{\text{SAR}}$ is $(1.125, -0.5, -0.5, 0.125)$, matching Section 7.2.3. The covariance between each $(\mathbf{W}\mathbf{Y})_i$ and its own $\varepsilon_i$ is $\sigma^2 \times 0.3333$. The direct, total, and indirect impacts per unit of $\beta_k$ are $1.1667$, $2.0$, and $0.8333$. The log-determinant is $-0.2877$ whether computed directly or from the eigenvalues, and the log-likelihood is $-4.9634$ from the Jacobian formula and from the multivariate normal density.

### 7.8.2 The County Map and Its Eigenvalues

Now the simulated map. The function below is the county map of Chapter 6: 144 square counties on a $12 \times 12$ grid, their rook adjacency matrix, populations (used in Chapter 9), and a smooth deprivation index (used in Chapters 6, 8, and 9, but not here). We row-standardize the adjacency matrix and compute the eigenvalues of $\mathbf{W}$ through the symmetric matrix $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, checking the result against the eigenvalues of $\mathbf{W}$ itself and Ord's identity at $\rho = 0.9$.

```python
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


S, A, pop, _ = county_map()    # the deprivation index is not used here
n = len(S)
deg = A.sum(axis=1)
W = A / deg[:, None]
D_half = 1 / np.sqrt(deg)
lam = np.linalg.eigvalsh(D_half[:, None] * A * D_half[None, :])
print("n =", n, " smallest and largest eigenvalue of W:",
      round(lam.min(), 4), round(lam.max(), 4))
print("parameter space: (", round(1 / lam.min(), 4), ",",
      round(1 / lam.max(), 4), ")")
lam_check = np.sort(np.linalg.eigvals(W).real)
print("same eigenvalues from W itself:", np.allclose(lam_check, lam))
rho_test = 0.9
sign, logdet = np.linalg.slogdet(np.eye(n) - rho_test * W)
print("log|I - 0.9 W|:", round(logdet, 4),
      round(np.log(1 - rho_test * lam).sum(), 4))
```

The smallest and largest eigenvalues are $-1$ and $1$, so the parameter space is $(-1, 1)$, as expected for a bipartite grid. The eigenvalues from the symmetric matrix agree with those of $\mathbf{W}$, and $\log\lvert\mathbf{I} - 0.9\mathbf{W}\rvert = -23.5143$ both by the determinant and by the sum of logarithms.

### 7.8.3 A Simulated Lag Model: How Badly Does OLS Do?

Return to the tax-rate story of Section 7.1, with the 144 counties as the taxing governments (counties levy property taxes too). The response is a county's property tax rate in mills, that is, dollars of tax per \$1,000 of assessed value. The covariate is a **school-funding need index**: pupils enrolled per unit of taxable property, standardized to mean 0 and standard deviation 1. A county with more pupils per dollar of tax base must charge a higher rate to fund the same schools, and need is smooth across the map because enrollment and property values are. Each county also reacts to its neighbors' rates, as in Section 7.1.

We simulate from a lag model with $\rho = 0.6$, $\boldsymbol{\beta} = (8, 1.5)'$ in mills, and $\sigma^2 = 2.25$ (an innovation standard deviation of 1.5 mills). Since $\boldsymbol{\Phi}\mathbf{1} = \mathbf{1}/(1 - \rho)$, the average rate is $8/(1 - 0.6) = 20$ mills, and a county one standard deviation above average need has, before any feedback, a rate $1.5$ mills higher. The need index has its own seed, 2071, and the responses use seed 2050. The block first shows the two OLS estimators on one data set, then repeats the simulation 500 times, and then evaluates the bias formula of Section 7.3.3.

```python
def school_need(S, seed=2071):
    """Smooth standardized school-funding need index (pupils per unit of
    taxable property), one value per county."""
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2))
    L = np.linalg.cholesky(np.exp(-d / 0.25) + 1e-8 * np.eye(len(S)))
    z = L @ np.random.default_rng(seed).standard_normal(len(S))
    return (z - z.mean()) / z.std()


need = school_need(S)
X = np.column_stack([np.ones(n), need])
rho_true, beta_true, sigma2_true = 0.6, np.array([8.0, 1.5]), 2.25
Phi = np.linalg.inv(np.eye(n) - rho_true * W)
rng = np.random.default_rng(2050)
y_mean = Phi @ X @ beta_true
print("expected tax rates (mills): mean", round(y_mean.mean(), 2),
      " range", round(y_mean.min(), 2), "to", round(y_mean.max(), 2))


def simulate(kind):
    eps = np.sqrt(sigma2_true) * rng.standard_normal(n)
    if kind == "lag":
        return Phi @ (X @ beta_true + eps)
    if kind == "error":
        return X @ beta_true + Phi @ eps
    return X @ beta_true + eps


def ols_with_wy(y):
    Z = np.column_stack([W @ y, X])
    return np.linalg.lstsq(Z, y, rcond=None)[0]


def ols_plain(y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


y_lag = simulate("lag")
print("one data set, OLS on [Wy, 1, need]:", np.round(ols_with_wy(y_lag), 4))
print("one data set, OLS on [1, need]:", np.round(ols_plain(y_lag), 4))

reps = 500
est_wy = np.array([ols_with_wy(simulate("lag")) for _ in range(reps)])
est_plain = np.array([ols_plain(simulate("lag")) for _ in range(reps)])
print("OLS with Wy, mean (rho, b0, b1):", np.round(est_wy.mean(axis=0), 4))
print("OLS without Wy, mean (b0, b1):", np.round(est_plain.mean(axis=0), 4))

G = W @ Phi
mu_y = Phi @ (X @ beta_true)
m_vec = G @ (X @ beta_true)
Z_mean = np.column_stack([m_vec, X])
EZZ = Z_mean.T @ Z_mean
EZZ[0, 0] += sigma2_true * np.trace(G.T @ G)
EZe = np.array([sigma2_true * np.trace(G), 0.0, 0.0])
print("average Cov(Wy_i, eps_i) / sigma^2:", round(np.trace(G) / n, 4))
print("approximate bias from the formula:",
      np.round(np.linalg.solve(EZZ, EZe), 4))
print("simulated bias:", np.round(est_wy.mean(axis=0)
                                  - np.array([rho_true, *beta_true]), 4))
xs_exact = np.linalg.solve(X.T @ X, X.T @ Phi @ X @ beta_true)
print("exact mean of OLS without Wy:", np.round(xs_exact, 4))
```

On one data set, OLS with the lag regressor gives $\hat{\rho} = 0.597$ (true $0.6$), intercept $7.888$, and a slope of $1.610$ (true $1.5$); OLS without the lag gives an intercept of $19.591$ and a slope of $2.789$. Over 500 data sets, OLS with the lag has mean $\hat{\rho} = 0.7283$, mean intercept $5.4321$, and mean slope $1.2409$, a bias of $+0.128$ in $\rho$, $-2.568$ in the intercept, and $-0.259$ in the slope. The approximate formula gives $+0.148$, $-2.949$, and $-0.290$, close but not exact because it replaces ratios of random quantities by ratios of expectations. The average of $\operatorname{Cov}(\mathbf{W}\mathbf{Y}_i, \varepsilon_i)/\sigma^2$ is $0.2155$. OLS without the lag has mean slope $2.6767$, against the exact value $2.6783$ from the formula of Section 7.3.3: it estimates something between the direct and total impact ($1.69$ and $3.75$ mills), as Section 7.8.7 shows, and not the coefficient $\beta_1 = 1.5$.

### 7.8.4 Maximum Likelihood

The next block implements Section 7.5. The function `make_profile` returns the profile log-likelihood of $\rho$ for a given data set and model, with $\mathbf{W}\mathbf{y}$ and $\mathbf{W}\mathbf{X}$ computed once; `fit_sar` maximizes it over the parameter space; `full_loglik` and `ml_covariance` give the covariance matrix of the estimates from a numerical Hessian. We fit the lag model to the simulated data set, compare it with OLS by a likelihood ratio test, and fit the error model to the same data for comparison.

```python
I_n = np.eye(n)
LOG2PI = np.log(2 * np.pi)


def ols_resid(v, X):
    """Residuals from regressing v on the columns of X."""
    return v - X @ np.linalg.solve(X.T @ X, X.T @ v)


def make_profile(y, X, W, lam, kind):
    """Return the profile log-likelihood of rho for one data set.

    W @ y and W @ X are computed once, so each evaluation costs O(n)."""
    n = len(y)
    Wy, WX = W @ y, W @ X
    if kind == "lag":
        e0, eL = ols_resid(y, X), ols_resid(Wy, X)

    def profile(rho):
        if kind == "lag":
            e = e0 - rho * eL
        else:
            e = filtered_resid(y - rho * Wy, X - rho * WX)
        return (-n / 2 * (LOG2PI + 1) - n / 2 * np.log(e @ e / n)
                + np.log(1 - rho * lam).sum())
    return profile


def filtered_resid(ys, Xs):
    """Residuals from regressing ys on Xs (the filtered data)."""
    return ys - Xs @ np.linalg.solve(Xs.T @ Xs, Xs.T @ ys)


def fit_sar(y, X, W, lam, kind):
    """ML fit by a one-dimensional search over rho."""
    lo, hi = 1 / lam.min() + 1e-6, 1 / lam.max() - 1e-6
    profile = make_profile(y, X, W, lam, kind)
    res = optimize.minimize_scalar(lambda r: -profile(r), bounds=(lo, hi),
                                   method="bounded",
                                   options={"xatol": 1e-8})
    rho = res.x
    F = np.eye(len(y)) - rho * W
    if kind == "lag":
        beta = np.linalg.solve(X.T @ X, X.T @ (F @ y))
        e = F @ y - X @ beta
    else:
        Xs = F @ X
        beta = np.linalg.solve(Xs.T @ Xs, Xs.T @ (F @ y))
        e = F @ (y - X @ beta)
    return rho, beta, e @ e / len(y), -res.fun


def full_loglik(theta, y, X, W, lam, kind):
    """Log-likelihood in (beta, rho, sigma^2), for the Hessian."""
    beta, rho, s2 = theta[:-2], theta[-2], theta[-1]
    F = np.eye(len(y)) - rho * W
    e = F @ y - X @ beta if kind == "lag" else F @ (y - X @ beta)
    return (-len(y) / 2 * (LOG2PI + np.log(s2)) + np.log(1 - rho * lam).sum()
            - e @ e / (2 * s2))


def ml_covariance(theta, *args, h=1e-4):
    """Covariance matrix from a central-difference Hessian."""
    k = len(theta)
    H = np.zeros((k, k))
    step = h * np.maximum(1.0, np.abs(theta))
    f = lambda t: full_loglik(t, *args)
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * step[i], np.eye(k)[j] * step[j]
            H[i, j] = (f(theta + ei + ej) - f(theta + ei - ej)
                       - f(theta - ei + ej) + f(theta - ei - ej)
                       ) / (4 * step[i] * step[j])
    return np.linalg.inv(-H)


rho_hat, beta_hat, s2_hat, ll_lag = fit_sar(y_lag, X, W, lam, "lag")
theta_hat = np.array([*beta_hat, rho_hat, s2_hat])
cov_hat = ml_covariance(theta_hat, y_lag, X, W, lam, "lag")
se = np.sqrt(np.diag(cov_hat))
print("lag ML:", np.round(theta_hat, 4), " log-lik", round(ll_lag, 3))
print("lag ML standard errors:", np.round(se, 4))
beta_ols = np.linalg.lstsq(X, y_lag, rcond=None)[0]
e_ols = y_lag - X @ beta_ols
ll_ols = -n / 2 * (LOG2PI + 1 + np.log(e_ols @ e_ols / n))
lrt = 2 * (ll_lag - ll_ols)
print("OLS log-lik", round(ll_ols, 3), " LR statistic", round(lrt, 2),
      " p-value", stats.chi2.sf(lrt, 1))
rho_e, beta_e, s2_e, ll_err = fit_sar(y_lag, X, W, lam, "error")
print("error ML on the same data:", np.round([*beta_e, rho_e, s2_e], 4),
      " log-lik", round(ll_err, 3))
print("AIC lag:", round(-2 * ll_lag + 2 * 4, 2),
      " AIC error:", round(-2 * ll_err + 2 * 4, 2))
```

The lag model fit gives $(\hat{\beta}_0, \hat{\beta}_1, \hat{\rho}, \hat{\sigma}^2) = (10.030, 1.825, 0.488, 1.928)$ against the truth $(8.0, 1.5, 0.6, 2.25)$, with standard errors $(1.210, 0.168, 0.061, 0.230)$. Each estimate is within about 1.9 standard errors of the truth. The likelihood ratio statistic against OLS is $48.05$ ($p < 10^{-11}$). The error model fitted to the same data has log-likelihood $-267.40$, against $-256.70$ for the lag model, so AIC prefers the lag model, $521.39$ against $542.80$. Figure 7.1 shows the two profile log-likelihoods.

![Figure 7.1. Profile log-likelihood of rho for the lag model and the error model, fitted to data simulated from a lag model with rho = 0.6. The lag model fits better, and its curve peaks near the truth; the error model compensates with a larger rho.](../figures/ch07/fig_7_1_profile.png)

In Figure 7.1 the lag curve peaks at $\hat{\rho} = 0.488$, a little below the true $0.6$, and is about $11$ units higher than the error curve at its own peak, $\hat{\rho} = 0.639$. The error model tries to account for the neighbor dependence through the errors alone, and does so less well because the data have a mean that depends on the neighbors' covariates. Both curves fall steeply as $\rho \to 1$, where the Jacobian term $\sum\log(1 - \rho\lambda_i)$ goes to minus infinity.

### 7.8.5 The Sampling Distribution of the Estimators

Repeat the ML fit on 500 simulated data sets and compare with OLS with the lag regressor.

```python
fits = [fit_sar(simulate("lag"), X, W, lam, "lag") for _ in range(reps)]
rho_ml = np.array([f[0] for f in fits])
b1_ml = np.array([f[1][1] for f in fits])
print("ML over", reps, "data sets: mean rho", round(rho_ml.mean(), 4),
      " sd", round(rho_ml.std(), 4), " mean beta1", round(b1_ml.mean(), 4),
      " sd", round(b1_ml.std(), 4))
print("OLS with Wy: mean rho", round(est_wy[:, 0].mean(), 4),
      " sd", round(est_wy[:, 0].std(), 4))
print("share of estimates below the truth: ML",
      round(np.mean(rho_ml < rho_true), 3), " OLS with Wy",
      round(np.mean(est_wy[:, 0] < rho_true), 3))
```

Across the 500 data sets, ML has mean $\hat{\rho} = 0.585$ (standard deviation $0.062$) and mean slope $1.528$ (standard deviation $0.170$). OLS with the lag has mean $\hat{\rho} = 0.728$ (standard deviation $0.069$). ML is slightly low for $\rho$, a small-sample effect, but far closer to the truth than OLS, whose estimate is on average $0.13$ too high. Only 3.8% of the OLS estimates fall below the true $0.6$, while 56.6% of the ML estimates do.

### 7.8.6 Lagrange Multiplier Tests

The function `lm_tests` computes the four statistics of Section 7.6.2 from the OLS fit. We apply it to the simulated lag data, and then evaluate how often each test rejects at the 5% level in 2,000 data sets simulated with no spatial dependence, with a lag, and with error dependence (the error model with the same $\rho = 0.6$).

```python
def lm_tests(y, X, W):
    """LM-error, LM-lag, and robust versions, from OLS residuals."""
    n = len(y)
    b = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ b
    s2 = e @ e / n
    Wy = W @ y
    WXb = W @ (X @ b)
    T = np.trace(W @ W + W.T @ W)
    J = ols_resid(WXb, X) @ WXb / s2 + T
    g_err, g_lag = e @ (W @ e) / s2, e @ Wy / s2
    stat = {"LM-error": g_err**2 / T,
            "LM-lag": g_lag**2 / J,
            "robust LM-error": (g_err - T / J * g_lag) ** 2 / (T - T**2 / J),
            "robust LM-lag": (g_lag - g_err) ** 2 / (J - T)}
    return stat


print("T_W =", round(np.trace(W @ W + W.T @ W), 3))
stat = lm_tests(y_lag, X, W)
for name, value in stat.items():
    print(f"{name:16s} statistic {value:8.2f}  p-value "
          f"{stats.chi2.sf(value, 1):.2e}")

reps_lm = 2000
rng = np.random.default_rng(2051)
rows = {}
for truth in ("none", "lag", "error"):
    hits = {k: 0 for k in stat}
    for _ in range(reps_lm):
        for name, value in lm_tests(simulate(truth), X, W).items():
            hits[name] += stats.chi2.sf(value, 1) < 0.05
    rows[truth] = {k: v / reps_lm for k, v in hits.items()}
    print(f"truth = {truth:5s}",
          "  ".join(f"{k}: {v:.3f}" for k, v in rows[truth].items()))
```

For the single simulated lag data set, $T_W = 80.167$ for this map, and the statistics are $\text{LM}_{\text{err}} = 16.30$, $\text{LM}_{\text{lag}} = 40.43$, $\text{RLM}_{\text{err}} = 0.19$, and $\text{RLM}_{\text{lag}} = 24.32$. The rejection rates across 2,000 simulations are:

| Truth | LM-error | LM-lag | Robust LM-error | Robust LM-lag |
| :--- | :--- | :--- | :--- | :--- |
| No spatial dependence | 0.043 | 0.050 | 0.053 | 0.044 |
| Lag | 0.997 | 1.000 | 0.046 | 0.999 |
| Error | 0.999 | 0.993 | 0.790 | 0.087 |

With no dependence all four tests reject close to the nominal 5%. When the truth is a lag, both ordinary tests reject nearly always, as Section 7.6.2 warned, while the robust error test rejects only 4.6% of the time and the robust lag test rejects 99.9%. When the truth is an error model, both ordinary tests again reject nearly always, the robust error test rejects 79.0%, and the robust lag test rejects only 8.7%. The robust tests give the right answer in both cases, though the error model is harder to identify: its power is 79%, against 100% for the lag.

### 7.8.7 Impacts

The next block computes the true and estimated impacts from the eigenvalues, draws from the normal approximation to the estimates to get intervals, and runs the check of Section 7.7.4: raise the school-funding need index by one unit in one county in the middle of the map, county 78, with the innovations held fixed, and compare the change in the response with the entries of $\boldsymbol{\Phi}\beta_1$.

```python
def impacts(rho, beta1, lam):
    """Average direct, indirect, and total impacts of one covariate."""
    direct = beta1 * np.mean(1 / (1 - rho * lam))
    total = beta1 / (1 - rho)
    return direct, total - direct, total


true_imp = impacts(rho_true, beta_true[1], lam)
est_imp = impacts(rho_hat, beta_hat[1], lam)
print("true impacts (direct, indirect, total):", np.round(true_imp, 4))
print("estimated impacts:", np.round(est_imp, 4))
print("naive: coefficient", round(beta_hat[1], 4),
      " OLS slope", round(beta_ols[1], 4))

rng = np.random.default_rng(2052)
draws = rng.multivariate_normal(theta_hat, cov_hat, size=4000)
draws = draws[np.abs(draws[:, 2]) < 0.999]
sim = np.array([impacts(d[2], d[1], lam) for d in draws])
lo, hi = np.percentile(sim, [2.5, 97.5], axis=0)
for name, a, b, c in zip(("direct", "indirect", "total"), true_imp, lo, hi):
    print(f"{name:8s} true {a:.3f}   95% interval ({b:.3f}, {c:.3f})")

eps0 = rng.standard_normal(n)
y0 = Phi @ (X @ beta_true + eps0)
j = 6 * 12 + 6
X1 = X.copy()
X1[j, 1] += 1.0
y1 = Phi @ (X1 @ beta_true + eps0)
diff = y1 - y0
print("raise need by one unit in county", j, ": change in county", j, "=",
      round(diff[j], 4), " sum of all changes =", round(diff.sum(), 4))
print("analytic:", round(Phi[j, j] * beta_true[1], 4),
      round(Phi[:, j].sum() * beta_true[1], 4))
print("a neighbor's change:", round(diff[j + 1], 4),
      " a county two steps away:", round(diff[j + 2], 4),
      " a county at the edge:", round(diff[j % 12], 4))
own = np.array([beta_true[1] * Phi[k, k] for k in range(n)])
allc = np.array([beta_true[1] * Phi[:, k].sum() for k in range(n)])
print("average own effect:", round(own.mean(), 4),
      " average total effect:", round(allc.mean(), 4))
```

The true impacts for $\beta_1 = 1.5$ at $\rho = 0.6$ are $1.694$ mills (direct), $2.056$ mills (indirect), and $3.750$ mills (total). At the estimates they are $1.967$, $1.596$, and $3.563$. The 95% intervals are:

| Impact | True | 95% interval |
| :--- | :--- | :--- |
| Direct | 1.694 | (1.673, 2.259) |
| Indirect | 2.056 | (1.083, 2.290) |
| Total | 3.750 | (3.056, 4.243) |

All three contain the truth. The spillover is large: more than half of the total effect of the covariate reaches other counties. Compare these with two naive numbers. The lag model's coefficient is $\hat{\beta}_1 = 1.826$, close to the direct impact and less than half of the total, and the OLS slope is $2.789$, which lies between the direct and total impacts and measures neither.

The simulation check agrees with the formulas exactly. Raising the school-funding need index by one unit in county 78 changes the property tax rate there by $1.6718$ mills, which is $\Phi_{78,78}\beta_1$, and the changes in all counties sum to $3.7513$ mills, which is $\sum_i\Phi_{i,78}\beta_1$. An east neighbor changes by $0.2864$ mills, a county two steps east by $0.0515$, and a county on the far edge of the map by $0.0001$. Averaged over all 144 counties, the own effect is $1.6939$ mills and the total effect is $3.75$ mills, which equal the direct and total impacts.

### 7.8.8 The Error Model: Naive Versus ML Intervals

Finally, return to the error model and the claim of Section 7.2.4. We simulate 500 data sets from the error model with $\rho = 0.6$, fit OLS with the naive 95% interval for the slope, and fit the error model by ML, using the standard error from the filtered regression at $\hat{\rho}$.

```python
rng = np.random.default_rng(2053)
cover_ols = cover_ml = 0
se_ols_sum = se_ml_sum = 0.0
t_crit = stats.t.ppf(0.975, n - 2)
XtX_inv = np.linalg.inv(X.T @ X)
for _ in range(reps):
    y_e = simulate("error")
    b = XtX_inv @ X.T @ y_e
    r = y_e - X @ b
    se_n = np.sqrt(r @ r / (n - 2) * XtX_inv[1, 1])
    rho_k, beta_k, s2_k, _ = fit_sar(y_e, X, W, lam, "error")
    Xs = (np.eye(n) - rho_k * W) @ X
    se_g = np.sqrt(s2_k * np.linalg.inv(Xs.T @ Xs)[1, 1])
    cover_ols += abs(b[1] - beta_true[1]) < t_crit * se_n
    cover_ml += abs(beta_k[1] - beta_true[1]) < 1.96 * se_g
    se_ols_sum += se_n
    se_ml_sum += se_g
print("error model, OLS: coverage", cover_ols / reps,
      " mean SE", round(se_ols_sum / reps, 4))
print("error model, ML:  coverage", cover_ml / reps,
      " mean SE", round(se_ml_sum / reps, 4))
```

The naive OLS interval covers the true slope $\beta_1 = 1.5$ in 77.0% of the data sets and its average standard error is $0.1478$. The ML interval covers in 91.6% and its average standard error is $0.1719$. The ML interval is wider on average, and that is correct: the OLS standard error is too small because it ignores the correlation, as Chapter 1 explained for a smooth covariate. The wider interval is the honest price of admitting that the data carry less independent information than OLS assumed. How large the shortfall of OLS is depends on how smooth the covariate is and how large $\rho$ is (Exercise 20).

### 7.8.9 Figures

The last block draws Figures 7.1 to 7.3. It saves them to `figures/ch07/`, so run the chapter's code from the top folder of the repository.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch07", exist_ok=True)
grid_r = np.linspace(-0.6, 0.98, 200)
prof_lag = make_profile(y_lag, X, W, lam, "lag")
prof_err = make_profile(y_lag, X, W, lam, "error")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(grid_r, [prof_lag(r) for r in grid_r], label="lag model")
ax.plot(grid_r, [prof_err(r) for r in grid_r], label="error model")
ax.axvline(rho_true, color="gray", linestyle="--", label="true rho")
ax.set_xlabel("rho")
ax.set_ylabel("profile log-likelihood")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ch07/fig_7_1_profile.png", dpi=120)

fig, ax = plt.subplots(figsize=(6.5, 3.4))
bins = np.linspace(0.35, 0.95, 40)
ax.hist(est_wy[:, 0], bins=bins, alpha=0.6, label="OLS with Wy")
ax.hist(rho_ml, bins=bins, alpha=0.6, label="maximum likelihood")
ax.axvline(rho_true, color="black", linestyle="--", label="true rho")
ax.set_xlabel("estimate of rho")
ax.set_ylabel("number of data sets")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ch07/fig_7_2_rho_estimates.png", dpi=120)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, r in zip(axes, (0.3, 0.6, 0.9)):
    Phi_r = np.linalg.inv(np.eye(n) - r * W)
    effect = (beta_true[1] * Phi_r[:, j]).reshape(12, 12)
    im = ax.imshow(effect, origin="lower", extent=(0, 1, 0, 1),
                   cmap="viridis", norm=matplotlib.colors.PowerNorm(0.4))
    ax.set_title(f"rho = {r}: total {effect.sum():.2f}")
    ax.set_xlabel("east")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("figures/ch07/fig_7_3_spillover.png", dpi=120)
print("figures saved")
```

![Figure 7.2. Estimates of rho from 500 data sets simulated from a lag model with rho = 0.6. Maximum likelihood is centered near the truth, slightly low. OLS with the lag regressor is centered well above it.](../figures/ch07/fig_7_2_rho_estimates.png)

In Figure 7.2 the OLS histogram sits to the right of the dashed line for the true value, centered near $0.73$, and the ML histogram is centered near $0.59$. Only 3.8% of the OLS estimates fall below $0.6$.

![Figure 7.3. The effect of raising the school-funding need index by one unit in the middle county, on the expected property tax rate in every county, for three values of rho. The panel titles give the sum of the effects. Colors use a power scale so that the small spillovers are visible.](../figures/ch07/fig_7_3_spillover.png)

In Figure 7.3 the effect is largest in the county itself and fades with distance across the map. At $\rho = 0.3$ it is confined to the nearest few counties and the total effect is $2.14$ mills. At $\rho = 0.6$ it reaches farther and the total is $3.75$ mills. At $\rho = 0.9$ the spillover covers most of the map and the total is $15.24$ mills (the column sum for this county; the average total impact across the map is exactly $1.5/(1 - \rho) = 15$ mills), ten times the coefficient. Large values of $\rho$ make the total impact extremely sensitive to the estimate, since $1/(1 - \rho)$ grows quickly near 1.

---

## Exercises

**Analytical**

1. Show that $\mathbf{Y} = \boldsymbol{\Phi}(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon})$ in the lag model has the same covariance as the error model. Then explain in words why the two models can still be told apart, using the mean. For the four-region map with $\rho = 0.5$ and $\mathbf{x} = (1, 0, 0, 0)'$, compute the lag-model mean $\boldsymbol{\Phi}\mathbf{x}\beta$ and compare it with the error-model mean $\mathbf{x}\beta$.

2. For the two-region system of Section 7.1, find $\operatorname{Var}(Y_1)$, $\operatorname{Var}(Y_2)$, and the correlation between them as functions of $\rho$. What happens as $\rho \to 1$? Check against the matrix formula $\sigma^2\boldsymbol{\Phi}\boldsymbol{\Phi}'$ with $\mathbf{W} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$.

3. For the four-region map, show that $\mathbf{W}^3 = \mathbf{W}$, and verify the formula $\boldsymbol{\Phi}(\rho) = \mathbf{I} + \frac{\rho}{1-\rho^2}\mathbf{W} + \frac{\rho^2}{1-\rho^2}\mathbf{W}^2$ by multiplying it by $\mathbf{I} - \rho\mathbf{W}$. Compute $\boldsymbol{\Phi}(0.8)$, its direct impact, and its total impact.

4. Prove that for $\mathbf{Q} = \sigma^{-2}(\mathbf{I} - \rho\mathbf{W})'(\mathbf{I} - \rho\mathbf{W})$, the entry $Q_{ij}$ is zero whenever regions $i$ and $j$ are more than two steps apart on the map. Give an example of a $3 \times 3$ map (a line of three regions) and compute $\mathbf{Q}$ explicitly, showing that regions 1 and 3 have a nonzero entry.

5. Show that when $\mathbf{W}$ is symmetric, $\mathbf{I}_n - \rho\mathbf{W}$ has eigenvalues $1 - \rho\lambda_i$ and the covariance of the error model has eigenvalues $\sigma^2/(1 - \rho\lambda_i)^2$. Which eigenvectors of $\mathbf{W}$ have the largest variance when $\rho > 0$? Relate this to the discussion of smooth eigenvectors in Section 1.3.3.

6. In Section 7.3.3 we derived $\operatorname{Cov}(\mathbf{W}\mathbf{Y}, \boldsymbol{\varepsilon}) = \sigma^2\mathbf{W}\boldsymbol{\Phi}$. Show that the diagonal entry $(\mathbf{W}\boldsymbol{\Phi})_{ii}$ is positive for $\rho > 0$ and zero for $\rho = 0$. What is its sign when $\rho < 0$ on a bipartite map? (Hint: only even powers of $\mathbf{W}$ have a nonzero diagonal there, and the series for $\mathbf{W}\boldsymbol{\Phi}$ has terms $\rho^{k-1}(\mathbf{W}^k)_{ii}$.)

7. Prove that the eigenvalues of a row-standardized $\mathbf{W}$ lie in $[-1, 1]$ using the argument of Section 7.4, and show that $-1$ is an eigenvalue if the map is bipartite. For a map with an odd cycle, such as a triangle of three mutually adjacent regions, find $\mathbf{W}$, its eigenvalues, and the parameter space for $\rho$.

8. For the lag model on the four-region map, derive the log-likelihood of $\mathbf{y} = (2, 1, 3, 2)'$ as a function of $\rho$ with $\mathbf{X} = \mathbf{1}$, after profiling out $\beta$ and $\sigma^2$, and find the value of $\rho$ at which the profile log-likelihood is largest, using a numerical search.

9. Derive the score for $\rho$ in the error model at $\rho = 0$, as in Section 7.6.4, and show that its variance is $T_W$. Then verify that $T_W = 2\operatorname{tr}(\mathbf{W}^2)$ when $\mathbf{W}$ is symmetric, and compute $T_W$ for the four-region map.

10. Show that for the error model $\beta_k$ is the marginal effect of covariate $k$, and that for the lag model the direct impact is never smaller than $\beta_k$ (for $\beta_k > 0$). (Hint: the direct impact is $\beta_k$ times the average of $1/(1 - \rho\lambda_i)$. Use Jensen's inequality for the convex function $1/(1 - \rho\lambda)$ and the fact that $\sum_i\lambda_i = \operatorname{tr}(\mathbf{W}) = 0$.)

11. The **spatial Durbin model** adds the neighbors' covariates to the lag model: $\mathbf{Y} = \rho\mathbf{W}\mathbf{Y} + \mathbf{X}\boldsymbol{\beta} + \mathbf{W}\tilde{\mathbf{X}}\boldsymbol{\gamma} + \boldsymbol{\varepsilon}$, where $\tilde{\mathbf{X}}$ is $\mathbf{X}$ without its column of ones.
(a) Explain why the column of ones must be left out of $\mathbf{W}\tilde{\mathbf{X}}$. (Hint: what is $\mathbf{W}\mathbf{1}_n$?)
(b) Write the error model as $(\mathbf{I}_n - \rho\mathbf{W})\mathbf{Y} = (\mathbf{I}_n - \rho\mathbf{W})\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ and expand it. Show that it is the Durbin model with the **common factor restriction** $\boldsymbol{\gamma} = -\rho\tilde{\boldsymbol{\beta}}$, where $\tilde{\boldsymbol{\beta}}$ holds the coefficients of $\tilde{\mathbf{X}}$, and with intercept $(1 - \rho)\beta_0$ in place of $\beta_0$. How many restrictions does a likelihood ratio test of the error model against the Durbin model have?
(c) Show that the effect matrix of covariate $k$ in the Durbin model is $\boldsymbol{\Phi}(\beta_k\mathbf{I}_n + \gamma_k\mathbf{W})$. Derive the direct and total impacts, $M_{\text{dir}} = [\beta_k\operatorname{tr}(\boldsymbol{\Phi}) + \gamma_k\operatorname{tr}(\boldsymbol{\Phi}\mathbf{W})]/n$ and $M_{\text{tot}} = (\beta_k + \gamma_k)/(1 - \rho)$, and check that under the common factor restriction the total impact reduces to $\beta_k$, as it must for the error model.
(d) For the four-region map with $\rho = 0.5$, $\beta_k = 1$, and $\gamma_k = 0.5$, compute the three impacts by hand and compare them with the lag model's ($\gamma_k = 0$) of Section 7.7.2.

12. Derive the expected Fisher information of the lag model, with the parameters ordered $\boldsymbol{\theta} = (\boldsymbol{\beta}', \rho, \sigma^2)'$ as in Section 7.8.4. Write $\mathbf{G} = \mathbf{W}\boldsymbol{\Phi}$, and use $\boldsymbol{\varepsilon} = (\mathbf{I}_n - \rho\mathbf{W})\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}$ and $\mathbf{W}\mathbf{Y} = \mathbf{G}(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon})$.
(a) Show that $\operatorname{E}[-\partial^2\ell/\partial\boldsymbol{\beta}\,\partial\boldsymbol{\beta}'] = \mathbf{X}'\mathbf{X}/\sigma^2$ and $\operatorname{E}[-\partial^2\ell/\partial\boldsymbol{\beta}\,\partial\rho] = \mathbf{X}'\mathbf{G}\mathbf{X}\boldsymbol{\beta}/\sigma^2$.
(b) Show that the derivative of $\log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert$ is $-\operatorname{tr}(\mathbf{G})$ and its second derivative is $-\operatorname{tr}(\mathbf{G}^2)$. Then, using the mean of a quadratic form, show
$$
\operatorname{E}\!\left[-\frac{\partial^2\ell}{\partial\rho^2}\right] = \operatorname{tr}(\mathbf{G}^2) + \operatorname{tr}(\mathbf{G}'\mathbf{G}) + \frac{(\mathbf{G}\mathbf{X}\boldsymbol{\beta})'(\mathbf{G}\mathbf{X}\boldsymbol{\beta})}{\sigma^2}
$$
(c) Show that the remaining entries are $\operatorname{E}[-\partial^2\ell/\partial\rho\,\partial\sigma^2] = \operatorname{tr}(\mathbf{G})/\sigma^2$, $\operatorname{E}[-\partial^2\ell/\partial(\sigma^2)^2] = n/(2\sigma^4)$, and $\operatorname{E}[-\partial^2\ell/\partial\boldsymbol{\beta}\,\partial\sigma^2] = \mathbf{0}$.
(d) Repeat (a) for the error model and show that its $(\boldsymbol{\beta}, \rho)$ block is zero. Explain why $\hat{\boldsymbol{\beta}}$ and $\hat{\rho}$ are correlated in the lag model but approximately uncorrelated in the error model, and what this means for the impact intervals of Section 7.7.3.
(e) Evaluate the matrix at the true parameters of Section 7.8.3, invert it, and compare the standard errors with the numerical-Hessian standard errors of Section 7.8.4.

13. Instead of maximum likelihood, the lag model can be fit by **two-stage least squares** (2SLS), using spatial lags of the covariates as instruments for $\mathbf{W}\mathbf{Y}$ [@kelejian1998].
(a) Use the Neumann series to show that $\operatorname{E}[\mathbf{W}\mathbf{Y}] = \mathbf{G}\mathbf{X}\boldsymbol{\beta} = \sum_{k \ge 0}\rho^k\mathbf{W}^{k+1}\mathbf{X}\boldsymbol{\beta}$. Explain why the columns of $\mathbf{W}\tilde{\mathbf{X}}$ and $\mathbf{W}^2\tilde{\mathbf{X}}$ are correlated with $\mathbf{W}\mathbf{Y}$ and uncorrelated with $\boldsymbol{\varepsilon}$, which is what an instrument needs.
(b) With $\mathbf{Z} = [\mathbf{W}\mathbf{Y},\ \mathbf{X}]$, instruments $\mathbf{H} = [\mathbf{X},\ \mathbf{W}\tilde{\mathbf{X}},\ \mathbf{W}^2\tilde{\mathbf{X}}]$, and the projection $\mathbf{P}_{\mathbf{H}} = \mathbf{H}(\mathbf{H}'\mathbf{H})^{-1}\mathbf{H}'$, show that the 2SLS estimator $(\mathbf{Z}'\mathbf{P}_{\mathbf{H}}\mathbf{Z})^{-1}\mathbf{Z}'\mathbf{P}_{\mathbf{H}}\mathbf{Y}$ is OLS of $\mathbf{Y}$ on the fitted values $\mathbf{P}_{\mathbf{H}}\mathbf{Z}$ from a first-stage regression.
(c) The instruments are weak when $\mathbf{W}\tilde{\mathbf{X}}\tilde{\boldsymbol{\beta}}$ carries little information beyond $\mathbf{X}$. Show that if $\tilde{\boldsymbol{\beta}} = \mathbf{0}$, then $\operatorname{E}[\mathbf{W}\mathbf{Y}]$ is a multiple of $\mathbf{1}_n$ and 2SLS cannot identify $\rho$, while ML still can. What does the Jacobian term contribute that 2SLS lacks?

14. Compare the lag model with the **SLX model** (spatial lag of $\mathbf{X}$), $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{W}\tilde{\mathbf{X}}\boldsymbol{\theta} + \boldsymbol{\varepsilon}$, with independent errors.
(a) Show that the effect matrix of covariate $k$ in the SLX model is $\beta_k\mathbf{I}_n + \theta_k\mathbf{W}$. Conclude that a change in region $j$ reaches only region $j$ and its immediate neighbors (**local spillovers**), and that its direct, indirect, and total impacts are $\beta_k$, $\theta_k$, and $\beta_k + \theta_k$.
(b) Contrast this with the lag model, where $\boldsymbol{\Phi}\beta_k = \beta_k\sum_{k' \ge 0}\rho^{k'}\mathbf{W}^{k'}$ reaches every region of a connected map (**global spillovers**). On the $2 \times 2$ block, compute both effect matrices with $\beta_k = 1$, $\rho = 0.5$, and $\theta_k$ chosen so that the two models have the same total impact. Which entries differ?
(c) Explain why the SLX model can be fit by OLS while the lag model cannot.

15. Chapter 6 built $k$-nearest-neighbor weights, for which $\mathbf{A}$ is not symmetric and the row-standardized $\mathbf{W}$ can have complex eigenvalues $\lambda_j = a_j + ib_j$ (Section 6.3.3).
(a) Explain why complex eigenvalues of a real matrix come in conjugate pairs $a_j \pm ib_j$, and show that each pair contributes the positive real factor $(1 - \rho a_j)^2 + \rho^2b_j^2$ to $\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert$ for real $\rho$.
(b) Show that $1 - \rho\lambda_j \ne 0$ for every real $\rho$ when $b_j \ne 0$. Conclude that only the real eigenvalues limit the parameter space, which becomes $(1/\lambda_{\min}, 1/\lambda_{\max})$ with $\lambda_{\min}$ the smallest real eigenvalue, and that the determinant is positive on it.
(c) Ord's identity still holds with complex eigenvalues, $\log\lvert\mathbf{I}_n - \rho\mathbf{W}\rvert = \sum_j\log(1 - \rho\lambda_j)$, as long as the logarithms of each conjugate pair are added together. Why can the symmetric shortcut of Section 7.5.2 not be used here, and what does that cost?

**Computational and simulation**

16. Repeat the 500-replicate comparison of Section 7.8.5 for $\rho = 0.2$, $0.4$, $0.8$. Plot the bias of OLS-with-lag and of ML as functions of $\rho$. Compare the bias of OLS-with-lag with the formula of Section 7.3.3.

17. Extend Section 7.7.3: compute the 95% intervals for the three impacts in 200 simulated data sets, and report how often each interval covers the true impact. Do the intervals cover about 95% of the time?

18. Repeat the LM test comparison of Section 7.8.6 with $\rho = 0.2$ and with $\rho = 0.4$. How does the power of the four tests change? Which truth, lag or error, is harder to identify at small $\rho$?

19. Replace the row-standardized $\mathbf{W}$ by the binary adjacency matrix $\mathbf{A}$ and compute the eigenvalues, the parameter space, and the profile log-likelihood for the simulated lag data. Why is the parameter space so much narrower, and what does the fitted $\hat{\rho}$ mean on this scale?

20. Simulate the error model with $\rho = 0.6$ and fit it by OLS and by ML. Compare the estimated slope and its standard error from the two fits. Then use a coordinate covariate instead of the smooth school-funding need index (for example, the east coordinate of the county centroid). Is the shortfall of the OLS interval larger or smaller, and why? (Use the variance ratio of Section 1.3.4.)

21. Implement the 2SLS estimator of Exercise 13 and apply it to the 500 lag-model data sets of Section 7.8.5. Compare the mean, standard deviation, and root mean squared error of $\hat{\rho}$ and $\hat{\beta}_1$ with those of ML and of OLS with the lag. Then repeat with a true slope of $\beta_1 = 0.2$ in place of $1.5$, and explain the change in 2SLS using part (c) of Exercise 13.

22. Compute intervals for the three impacts of Section 7.8.7 by the **delta method**: write each impact as a function of $(\beta_1, \rho)$, find its gradient (for example, the total impact $\beta_1/(1 - \rho)$ has gradient $(1/(1 - \rho),\ \beta_1/(1 - \rho)^2)$, and the direct impact has $\partial M_{\text{dir}}/\partial\rho = \beta_1 n^{-1}\sum_i\lambda_i/(1 - \rho\lambda_i)^2$), and combine it with the covariance matrix from the numerical Hessian. Compare the intervals with the simulation intervals of Section 7.8.7. Which impact has the most asymmetric simulation interval, and why does the delta method miss that?

---

## What Comes Next

This chapter specified dependence through a system of equations in which every region depends on its neighbors at once, and the covariance followed from solving the system. Chapter 8 starts from the other end. It specifies each region's distribution given its neighbors, builds the joint distribution from those conditionals, and works with the precision matrix directly, which makes large models fast. It also returns to the comparison of the two ways of building spatial dependence that Section 7.1 promised. Chapter 9 puts the same ideas inside a generalized linear model for counts.
