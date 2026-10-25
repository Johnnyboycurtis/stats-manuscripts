# Chapter 1: Spatial Dependence Through the Lens of Linear Models

Chapter 0 showed, with simulations, that ordinary statistical tools mislead when nearby observations are related. This chapter explains why, using the linear model you already know and the rule for the variance of a linear transformation.

The argument has a simple shape. Ordinary least squares (OLS) produces an estimate, a standard error, and a confidence interval. The estimate survives spatial dependence: it is still unbiased. The standard error does not. Its formula was derived under the assumption that the errors are independent, and when they are not, the formula answers the wrong question. We will derive the variance OLS actually has, show when the reported standard error is too small and by how much, and see that a better estimator exists.

Two new tools enter along the way, each introduced with a small example before it is used: the expected value of a quadratic form, and the eigenvectors of a covariance matrix. We also meet the effective sample size, the matrix form of each of the three kinds of spatial data, and the practical problem of measuring distance on a curved Earth. The chapter ends with a lab that checks the hand calculations in code, measures how often a naive confidence interval misses, and compares ways of measuring distance.

> **Chapter at a Glance**
>
> - **The Classical Linear Model (§1.1):** Where the independence assumption enters the OLS formulas, and a tool for averaging quadratic forms.
> - **Spatial Processes (§1.2):** From independent scalars to a correlated random vector with covariance matrix $\boldsymbol{\Sigma}$ and covariance function $C(h)$.
> - **OLS Under Spatial Dependence (§1.3):** The estimate stays unbiased, but its variance becomes a sandwich and the reported standard error is wrong. Eigenvectors explain when it is wrong, and generalized least squares does better.
> - **Effective Sample Size (§1.4):** How many independent observations a correlated data set is worth, and why adding points in the same region stops helping.
> - **Three Matrix Geometries (§1.5):** Dense distance matrices, sparse adjacency matrices, and random coordinates.
> - **Coordinates and Distance (§1.6):** Fast distance matrices, why raw latitude and longitude mislead, and great-circle and chordal distance.
> - **Walkthrough and Lab (§1.7):** The hand calculations in code, a coverage experiment, and a distance comparison.

---

## 1.1 Reviewing the Classical Linear Model

Suppose we have monthly rents for $n$ apartments in one city and want to know how rent depends on size. The classical linear model writes the rents as

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}, \qquad \operatorname{Var}(\boldsymbol{\varepsilon}) = \tau^2\mathbf{I}_n
$$

where $\mathbf{X}$ is an $n \times p$ design matrix of full column rank (here a column of ones and a column of square footage, so $p = 2$) and $\boldsymbol{\beta}$ is a vector of $p$ unknown coefficients.

Regression courses usually write the error variance as $\sigma^2$. This book writes it as $\tau^2$, because $\sigma^2$ is reserved for the variance of the spatial random effect, which appears in Section 1.2. In the language of Section 0.2, the classical model is the spatial model $Y = \mu + u + \varepsilon$ with no spatial random effect: all of the variation left after the trend is independent noise. The book's notation section collects these conventions.

The OLS estimator solves the normal equations $\mathbf{X}'\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\mathbf{Y}$ (Tool 5 in Section 0.4):

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}
$$

Its variance follows from Tool 1, the rule $\operatorname{Var}(\mathbf{B}\mathbf{Y}) = \mathbf{B}\operatorname{Var}(\mathbf{Y})\mathbf{B}'$, applied with $\mathbf{B} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$:

$$
\begin{aligned}
\operatorname{Var}(\hat{\boldsymbol{\beta}}) &= (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\,\operatorname{Var}(\mathbf{Y})\,\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1} \\
&= (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\,(\tau^2\mathbf{I}_n)\,\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1} \\
&= \tau^2(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1} \\
&= \tau^2(\mathbf{X}'\mathbf{X})^{-1}
\end{aligned}
$$

The second line is where independence does its work. Because $\operatorname{Var}(\mathbf{Y})$ is a multiple of the identity, the middle of the expression collapses and the familiar formula appears.

### 1.1.1 A Tool We Need: The Expected Value of a Quadratic Form

To estimate $\tau^2$ we will need the average value of a quadratic form $\mathbf{Y}'\mathbf{M}\mathbf{Y}$ (Tool 2 in Section 0.4). The result is used three times in this chapter, so we derive it now.

The derivation uses the **trace**, $\operatorname{tr}(\mathbf{M})$, the sum of the diagonal entries of a square matrix. Three properties are all we need:

- The trace of a $1 \times 1$ matrix is its single entry, so any scalar equals its own trace.
- The trace is linear, and so it can be moved inside an expectation: $\operatorname{E}[\operatorname{tr}(\mathbf{M})] = \operatorname{tr}(\operatorname{E}[\mathbf{M}])$.
- The trace is unchanged by cycling a product: $\operatorname{tr}(\mathbf{M}\mathbf{N}) = \operatorname{tr}(\mathbf{N}\mathbf{M})$ whenever both products are defined. (Write out the diagonal of each product to check this.)

Let $\mathbf{Y}$ have mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$, and let $\mathbf{M}$ be a fixed symmetric matrix. The quadratic form is a scalar, so it equals its trace, and cycling moves $\mathbf{Y}$ to the end:

$$
\mathbf{Y}'\mathbf{M}\mathbf{Y} = \operatorname{tr}(\mathbf{Y}'\mathbf{M}\mathbf{Y}) = \operatorname{tr}(\mathbf{M}\mathbf{Y}\mathbf{Y}')
$$

Now take expectations, moving $\operatorname{E}$ inside the trace. The matrix $\operatorname{E}[\mathbf{Y}\mathbf{Y}']$ is the matrix version of $\operatorname{E}[Y^2] = \operatorname{Var}(Y) + (\operatorname{E}[Y])^2$, namely $\boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}'$. So

$$
\begin{aligned}
\operatorname{E}[\mathbf{Y}'\mathbf{M}\mathbf{Y}] &= \operatorname{tr}\!\left(\mathbf{M}\,\operatorname{E}[\mathbf{Y}\mathbf{Y}']\right) \\
&= \operatorname{tr}\!\left(\mathbf{M}(\boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}')\right) \\
&= \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \operatorname{tr}(\mathbf{M}\boldsymbol{\mu}\boldsymbol{\mu}') \\
&= \operatorname{tr}(\mathbf{M}\boldsymbol{\Sigma}) + \boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}
\end{aligned}
$$

where the last step cycles the trace once more and uses the fact that $\boldsymbol{\mu}'\mathbf{M}\boldsymbol{\mu}$ is a scalar.

A quick check: with one variable, $\mathbf{M} = 1$, this says $\operatorname{E}[Y^2] = \operatorname{Var}(Y) + \mu^2$. The matrix result is the same idea with the cross terms included.

### 1.1.2 Estimating the Error Variance

The variance $\tau^2$ is estimated from the residuals $\mathbf{e} = \mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{Y}$, where $\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ is the projection onto the columns of $\mathbf{X}$ (Tool 5):

$$
s^2 = \frac{\mathbf{e}'\mathbf{e}}{n - p}
$$

Since $\mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$ is symmetric and idempotent, $\mathbf{e}'\mathbf{e} = \mathbf{Y}'(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{Y}$, a quadratic form with $\mathbf{M} = \mathbf{I}_n - \mathbf{P}_{\mathbf{X}}$. Its mean term is zero because $(\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$: projecting $\mathbf{X}\boldsymbol{\beta}$ onto the columns of $\mathbf{X}$ leaves it unchanged. By Section 1.1.1,

$$
\operatorname{E}[\mathbf{e}'\mathbf{e}] = \operatorname{tr}\!\left((\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\,\tau^2\mathbf{I}_n\right) = \tau^2\left(n - \operatorname{tr}(\mathbf{P}_{\mathbf{X}})\right) = \tau^2(n - p)
$$

using $\operatorname{tr}(\mathbf{P}_{\mathbf{X}}) = \operatorname{tr}\!\left((\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{X}\right) = \operatorname{tr}(\mathbf{I}_p) = p$, another cycle of the trace. So $s^2$ is unbiased for $\tau^2$, and again the identity matrix did the work.

Finally, the standard error of $\hat{\beta}_j$ is $\sqrt{s^2[(\mathbf{X}'\mathbf{X})^{-1}]_{jj}}$, and under normal errors the ratio $(\hat{\beta}_j - \beta_j)/\text{SE}$ has a $t$ distribution with $n - p$ degrees of freedom. That gives the confidence interval $\hat{\beta}_j \pm t_{0.975,\,n-p}\,\text{SE}$ and the usual t-test.

So independence is used in exactly three places:

1. To collapse $\operatorname{Var}(\hat{\boldsymbol{\beta}})$ to $\tau^2(\mathbf{X}'\mathbf{X})^{-1}$.
2. To make $s^2$ an unbiased estimate of the error variance.
3. To give the standardized estimate its $t$ distribution.

Section 1.3 removes the independence assumption and checks each of the three.

---

## 1.2 What Is a Spatial Process?

Return to the rents. Two apartments in the same building share a location, a school zone, and a walk to the same train station. If one rents for more than its size predicts because the block has quietly become desirable, the other probably does too. That shared, unexplained component is what the classical model leaves out.

In the classical model the index $i$ only counts rows. In spatial data each observation carries a location $\mathbf{s}_i$ in a study region $\mathcal{D}$, and the location matters because values at nearby locations are related.

A **spatial process** is a collection of random variables $\{Y(\mathbf{s}) : \mathbf{s} \in \mathcal{D}\}$, one for every location in the region. We never observe all of it. We observe it at $n$ locations and stack the results into a vector:

$$
\mathbf{Y} = [Y(\mathbf{s}_1), Y(\mathbf{s}_2), \dots, Y(\mathbf{s}_n)]'
$$

Everything we compute in this book depends on the process only through vectors like this one, at finitely many locations. A Gaussian spatial process is completely described by two ingredients: the mean $\operatorname{E}[Y(\mathbf{s})]$ at every location and the covariance $\operatorname{Cov}(Y(\mathbf{s}_i), Y(\mathbf{s}_j))$ between every pair of locations. Given those, the vector $\mathbf{Y}$ at any set of locations is multivariate normal with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$, where

$$
\mu_i = \operatorname{E}[Y(\mathbf{s}_i)], \qquad \Sigma_{ij} = \operatorname{Cov}\!\left(Y(\mathbf{s}_i), Y(\mathbf{s}_j)\right)
$$

This is the whole shift from classical to spatial statistics: $\boldsymbol{\Sigma}$ is no longer diagonal, and its off-diagonal entries depend on where the observations are.

For this chapter we use the model of Section 0.2,

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u} + \boldsymbol{\varepsilon}, \qquad \operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n
$$

where $\mathbf{u}$ is the shared, unexplained neighborhood component (the spatial random effect, with variance $\sigma^2$) and $\boldsymbol{\varepsilon}$ is independent noise (variance $\tau^2$). We use a specific correlation function for $\mathbf{u}$, the exponential:

$$
r(h; \phi) = \exp(-h/\phi), \qquad R_{ij} = r(h_{ij}; \phi)
$$

where $h_{ij}$ is the distance between $\mathbf{s}_i$ and $\mathbf{s}_j$. Correlation is 1 at distance zero and decays toward 0 as distance grows, more slowly when the range parameter $\phi$ is larger. Chapter 2 explains why this is a valid choice and introduces the alternatives.

In this model the covariance between $Y$ at two locations depends only on the distance $h$ between them. It is convenient to give that covariance a name, the **covariance function** $C(h)$:

$$
C(h) = \begin{cases} \sigma^2 + \tau^2 & h = 0 \\ \sigma^2\,r(h;\phi) & h > 0 \end{cases}
$$

The value $C(0) = \sigma^2 + \tau^2$ is the variance of $Y$ at any single location, and $\Sigma_{ij} = C(h_{ij})$. The jump at $h = 0$ is the noise term $\boldsymbol{\varepsilon}$, which adds variance at each location but no covariance between locations. Chapter 2 develops covariance functions in general.

Two features of this setup will matter repeatedly.

- **The classical model is a special case.** If $\sigma^2 = 0$, then $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n$ and we are back in Section 1.1. Every spatial result in this chapter should reduce to a classical one when $\sigma^2 = 0$, and checking that it does is a good habit.
- **We see one realization.** As Section 0.1.2 explained, we observe one map, not many independent copies of it. Our information about $\boldsymbol{\Sigma}$ comes from comparing pairs of locations within that single map. Chapters 2 and 3 develop this.

To simulate data from the model we use a matrix $\mathbf{L}$ with $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$. Then $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{L}\mathbf{Z}$ with $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ has exactly the right mean and covariance, by Tool 1 (Section 0.4.1). The Cholesky factorization supplies $\mathbf{L}$; Chapter 2 explains how it works, and the labs in this chapter use it as a tool.

---

## 1.3 Why We Cannot Ignore Space: OLS When $\operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma}$

Suppose the rents come from the spatial model of Section 1.2, but the analyst fits OLS as if the errors were independent. We check the three uses of independence from Section 1.1.2 one at a time. Section 1.3.1 shows the estimate is still unbiased. Section 1.3.2 finds its true variance and shows that the reported one is too small. Sections 1.3.3 and 1.3.4 explain which covariates suffer most. Section 1.3.5 shows that a different estimator, generalized least squares, does better, and Section 1.3.6 gives an optional direct proof.

### 1.3.1 Unbiasedness Is Preserved

The expectation of $\hat{\boldsymbol{\beta}}$ uses only the mean of $\mathbf{Y}$:

$$
\operatorname{E}[\hat{\boldsymbol{\beta}}] = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\operatorname{E}[\mathbf{Y}] = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}
$$

Nothing here involves $\boldsymbol{\Sigma}$, so the result holds for any covariance structure. OLS is still unbiased. If spatial dependence only made estimates noisier, a large sample would wash it out. The real damage is to the measures of uncertainty.

### 1.3.2 The True Variance of OLS

Repeat the variance calculation of Section 1.1, stopping before the step that used independence:

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}
$$

This is often called a **sandwich**: the matrix $\mathbf{X}'\boldsymbol{\Sigma}\mathbf{X}$ sits between two slices of $(\mathbf{X}'\mathbf{X})^{-1}$. When $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n$ the filling cancels one slice and we recover $\tau^2(\mathbf{X}'\mathbf{X})^{-1}$. When $\boldsymbol{\Sigma}$ has positive off-diagonal entries, nothing cancels.

Standard regression software does not compute the sandwich. It reports $s^2(\mathbf{X}'\mathbf{X})^{-1}$, so two things go wrong at once. First, the formula has the wrong shape, as just shown. Second, $s^2$ no longer estimates the variance it is meant to estimate. The calculation of Section 1.1.2 still applies, now with $\boldsymbol{\Sigma}$ in place of $\tau^2\mathbf{I}_n$:

$$
\operatorname{E}[s^2] = \frac{\operatorname{tr}\!\left((\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\boldsymbol{\Sigma}\right)}{n - p}
$$

The simplest case shows which way this goes. Take the intercept-only model, $\mathbf{X} = \mathbf{1}$, so that $\hat{\beta} = \bar{Y}$ and $\mathbf{P}_{\mathbf{X}} = \frac{1}{n}\mathbf{1}\mathbf{1}'$. Every location has variance $C(0)$, so $\operatorname{tr}(\boldsymbol{\Sigma}) = nC(0)$. Cycling the trace, $\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\boldsymbol{\Sigma}) = \frac{1}{n}\mathbf{1}'\boldsymbol{\Sigma}\mathbf{1}$, and by Tool 1 $\operatorname{Var}(\bar{Y}) = \frac{1}{n^2}\mathbf{1}'\boldsymbol{\Sigma}\mathbf{1}$, so $\operatorname{tr}(\mathbf{P}_{\mathbf{X}}\boldsymbol{\Sigma}) = n\operatorname{Var}(\bar{Y})$. Substituting,

$$
\operatorname{E}[s^2] = \frac{nC(0) - n\operatorname{Var}(\bar{Y})}{n - 1} = \frac{n}{n-1}\left[C(0) - \operatorname{Var}(\bar{Y})\right]
$$

The more the observations are positively correlated, the larger $\operatorname{Var}(\bar{Y})$ becomes, and the smaller $s^2$ becomes. Correlated observations look alike, so they scatter less around their own mean than independent ones would, and $s^2$ mistakes that for low variability. The reported variance $s^2/n$ is therefore too small for two reasons that reinforce each other: the true variance of $\bar{Y}$ is larger than $C(0)/n$, and $s^2$ is smaller than $C(0)$.

**A hand-sized example: four apartments on a block.** Place four apartments at the corners of a city block, and measure distance in block lengths: $\mathbf{s}_1 = (0, 0)$, $\mathbf{s}_2 = (1, 0)$, $\mathbf{s}_3 = (0, 1)$, $\mathbf{s}_4 = (1, 1)$. Each apartment is one block from two others (along the sides) and $\sqrt{2}$ blocks from the opposite corner. Take $\sigma^2 = 1$, $\tau^2 = 0$, and $\phi = 1$ block, and write

$$
a = e^{-1} \approx 0.3679, \qquad b = e^{-\sqrt{2}} \approx 0.2431
$$

for the correlations at distances 1 and $\sqrt{2}$. Then

$$
\boldsymbol{\Sigma} = \begin{bmatrix} 1 & a & a & b \\ a & 1 & b & a \\ a & b & 1 & a \\ b & a & a & 1 \end{bmatrix}
$$

We estimate the average rent level with $\bar{Y}$. Every row of $\boldsymbol{\Sigma}$ sums to $1 + 2a + b = 1.9789$, so $\mathbf{1}'\boldsymbol{\Sigma}\mathbf{1} = 4 \times 1.9789$ and

$$
\operatorname{Var}(\bar{Y}) = \frac{4 \times 1.9789}{16} \approx 0.4947
$$

An independent-data analysis would expect $C(0)/n = 1/4 = 0.25$. The true variance is about twice as large. Meanwhile, by the formula above,

$$
\operatorname{E}[s^2] = \frac{4}{3}\left[1 - 0.4947\right] \approx 0.6737
$$

so the reported variance of $\bar{Y}$ is, on average, $0.6737/4 \approx 0.1684$. The true variance is $0.4947/0.1684 \approx 2.94$ times larger, and the true standard error is $\sqrt{2.94} \approx 1.71$ times larger than the one reported. Both distortions are visible: the true variance is larger than $0.25$, and $s^2$ is smaller than 1. Section 1.7.1 repeats the calculation in code.

The third use of independence, the $t$ distribution, also fails, but it matters less. Once the variance estimate is wrong by a factor of 3 or 10, the exact shape of the reference distribution is a minor issue.

### 1.3.3 The Patterns Inside $\boldsymbol{\Sigma}$: Eigenvectors

The intercept-only model lost a factor of about 3. For a regression slope the loss can be much larger or much smaller, depending on the covariate. To see why, we need a way to describe which patterns across the locations carry a lot of variance and which carry little. Eigenvectors provide exactly that.

**Two locations.** Start with two neighboring apartments whose rents have variance 1 and correlation $r$:

$$
\boldsymbol{\Sigma} = \begin{bmatrix} 1 & r \\ r & 1 \end{bmatrix}
$$

Consider two ways of combining the rents. Their sum pattern $(Y_1 + Y_2)/\sqrt{2}$ and their difference pattern $(Y_1 - Y_2)/\sqrt{2}$ are linear combinations $\mathbf{a}'\mathbf{Y}$ with $\mathbf{a} = (1, 1)'/\sqrt{2}$ and $\mathbf{a} = (1, -1)'/\sqrt{2}$. By Tool 1,

$$
\operatorname{Var}\!\left(\frac{Y_1 + Y_2}{\sqrt{2}}\right) = \frac{1 + 2r + 1}{2} = 1 + r, \qquad \operatorname{Var}\!\left(\frac{Y_1 - Y_2}{\sqrt{2}}\right) = \frac{1 - 2r + 1}{2} = 1 - r
$$

and their covariance is $\frac{1}{2}(1 - r + r - 1) = 0$. With $r = 0.5$, the sum pattern has variance 1.5 and the difference pattern 0.5. Positive correlation puts extra variance into the pattern where the two apartments move together and removes it from the pattern where they move apart.

These two vectors are the **eigenvectors** of $\boldsymbol{\Sigma}$, and $1 + r$ and $1 - r$ are its **eigenvalues**. You can check that $\boldsymbol{\Sigma}(1, 1)' = (1 + r)(1, 1)'$ and $\boldsymbol{\Sigma}(1, -1)' = (1 - r)(1, -1)'$.

**The general case.** Every symmetric matrix $\boldsymbol{\Sigma}$ has an **eigendecomposition**

$$
\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'
$$

where $\boldsymbol{\Lambda}$ is diagonal with the eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$, and the columns $\mathbf{v}_1, \dots, \mathbf{v}_n$ of $\mathbf{U}$ are the corresponding eigenvectors. The eigenvectors have length 1 and are perpendicular to each other, so $\mathbf{U}'\mathbf{U} = \mathbf{I}_n$. Appendix A collects the facts we use.

The two-location example generalizes directly. Each eigenvector defines a pattern across the $n$ locations, $\mathbf{v}_k'\mathbf{Y}$. By Tool 1, the vector of all $n$ patterns, $\mathbf{U}'\mathbf{Y}$, has covariance

$$
\operatorname{Var}(\mathbf{U}'\mathbf{Y}) = \mathbf{U}'\boldsymbol{\Sigma}\mathbf{U} = \mathbf{U}'\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'\mathbf{U} = \boldsymbol{\Lambda}
$$

So the patterns are uncorrelated with each other, and pattern $k$ has variance $\lambda_k$. The eigendecomposition splits the data into independent patterns and says how much variance each carries. If you have met principal components, this is the same idea.

Two facts about the eigenvalues will matter.

- **They average to $C(0)$.** Their sum is $\operatorname{tr}(\boldsymbol{\Lambda}) = \operatorname{tr}(\mathbf{U}'\boldsymbol{\Sigma}\mathbf{U}) = \operatorname{tr}(\boldsymbol{\Sigma}\mathbf{U}\mathbf{U}') = \operatorname{tr}(\boldsymbol{\Sigma}) = nC(0)$. Correlation does not change the total variance; it redistributes it among patterns.
- **For spatial covariance matrices, large eigenvalues belong to smooth patterns.** Patterns in which nearby locations move together get extra variance, as the sum pattern did. Patterns in which neighbors move in opposite directions lose variance, as the difference pattern did. So the eigenvectors with the largest eigenvalues vary slowly across the region, and those with the smallest eigenvalues alternate between neighbors. Exercise 4 lets you plot them.

**Back to the block.** The four apartments of Section 1.3.2 have eigenvectors we can find by hand, because the square is symmetric. Multiply $\boldsymbol{\Sigma}$ by each of the following patterns and check that the result is a multiple of the pattern:

| Pattern | Vector (before scaling to length 1) | Eigenvalue |
| :--- | :--- | :--- |
| All four move together | $(1, 1, 1, 1)'$ | $1 + 2a + b = 1.9789$ |
| East side against west side | $(-1, 1, -1, 1)'$ | $1 - b = 0.7569$ |
| North side against south side | $(-1, -1, 1, 1)'$ | $1 - b = 0.7569$ |
| Diagonal corners against the other diagonal | $(1, -1, -1, 1)'$ | $1 - 2a + b = 0.5074$ |

The eigenvalues sum to $4 = nC(0)$, as they must. The smoothest pattern, everything moving together, carries almost half of the total variance. The most alternating pattern, the checkerboard, carries the least.

### 1.3.4 When the Standard Error Is Too Small

Now consider a regression slope. Take a model with an intercept and one covariate $\mathbf{x}$, and write $\mathbf{x}_c = \mathbf{x} - \bar{x}\mathbf{1}$ for the centered covariate. The OLS slope is the familiar

$$
\hat{\beta}_1 = \frac{\sum_i (x_i - \bar{x})(Y_i - \bar{Y})}{\sum_i (x_i - \bar{x})^2} = \frac{\mathbf{x}_c'\mathbf{Y}}{\mathbf{x}_c'\mathbf{x}_c}
$$

where the second form uses $\mathbf{x}_c'\mathbf{1} = 0$, so that subtracting $\bar{Y}$ changes nothing. This is a linear combination $\mathbf{a}'\mathbf{Y}$ with $\mathbf{a} = \mathbf{x}_c/(\mathbf{x}_c'\mathbf{x}_c)$, so by Tool 1

$$
\operatorname{Var}(\hat{\beta}_1) = \frac{\mathbf{x}_c'\boldsymbol{\Sigma}\mathbf{x}_c}{(\mathbf{x}_c'\mathbf{x}_c)^2}
$$

If the observations were independent with variance $C(0)$, this would be $C(0)/(\mathbf{x}_c'\mathbf{x}_c)$. The ratio of the two is

$$
\frac{\operatorname{Var}(\hat{\beta}_1)}{C(0)/(\mathbf{x}_c'\mathbf{x}_c)} = \frac{\mathbf{x}_c'\boldsymbol{\Sigma}\mathbf{x}_c}{C(0)\,\mathbf{x}_c'\mathbf{x}_c}
$$

We call this the **variance ratio** of the covariate. It measures how much spatial correlation inflates (or deflates) the true variance of the slope relative to what an independent-data analysis assumes. It does not include the downward bias of $s^2$ from Section 1.3.2, which makes the reported standard error smaller still.

Section 1.3.3 tells us how to read it. Expand the centered covariate in the eigenvectors, $\mathbf{x}_c = \sum_k (\mathbf{v}_k'\mathbf{x}_c)\,\mathbf{v}_k$. Because $\mathbf{U}'\mathbf{U} = \mathbf{I}_n$, the squared length is $\mathbf{x}_c'\mathbf{x}_c = \sum_k (\mathbf{v}_k'\mathbf{x}_c)^2$ and $\mathbf{x}_c'\boldsymbol{\Sigma}\mathbf{x}_c = \sum_k \lambda_k(\mathbf{v}_k'\mathbf{x}_c)^2$. Therefore

$$
\frac{\mathbf{x}_c'\boldsymbol{\Sigma}\mathbf{x}_c}{C(0)\,\mathbf{x}_c'\mathbf{x}_c} = \frac{1}{C(0)}\sum_{k=1}^n \lambda_k\,\frac{(\mathbf{v}_k'\mathbf{x}_c)^2}{\mathbf{x}_c'\mathbf{x}_c}
$$

The variance ratio is a weighted average of the eigenvalues, divided by $C(0)$. The weights are the shares of the covariate's squared length that lie along each eigenvector, and they add to 1. The two facts from Section 1.3.3 now explain everything.

- **A covariate unrelated to location gets a ratio near 1.** Consider regressing rent on the apartment's floor number. Floor number has nothing to do with where on the map an apartment is, so it spreads its length roughly evenly across all eigenvectors, and the weighted average is close to the plain average of the eigenvalues, which is $C(0)$. For such a covariate, OLS standard errors are close to right even though the rents are spatially correlated.
- **A spatially smooth covariate gets a ratio far above 1.** Now regress rent on distance to the nearest transit stop. Distance to transit changes slowly across the city, so it lies mostly along the smooth eigenvectors, its weights sit on the large eigenvalues, and the ratio can be large.

So the standard error is understated most severely when two things happen together: the errors are positively spatially correlated, and the covariate is itself spatially smooth. That is the usual situation with real covariates such as income, air quality, or distance to downtown. In the simulated coverage experiment of Section 1.7.2, distance to transit has a variance ratio of 11.33, and the naive 95% confidence interval covers the true slope only 41.5% of the time. Floor number has a ratio of 0.99 and close to nominal coverage.

The same expression shows that the distortion can also go the other way. A covariate that alternates between neighboring locations loads on the small eigenvalues and has a ratio below 1. On the block of Section 1.3.3, a covariate equal to 1 at two diagonal corners and 0 at the other two is, after centering, exactly the checkerboard eigenvector, so its variance ratio is that eigenvalue: 0.5074. Its true variance is about half what the independent-data formula claims. Such covariates are rare in practice.

### 1.3.5 Generalized Least Squares

The previous sections asked whether OLS reports its own uncertainty correctly. A separate question is whether OLS is the best estimator to use at all. It is not, and the reason follows from a result you already know.

Suppose for now that $\boldsymbol{\Sigma}$ is known, and let $\mathbf{L}$ be a matrix with $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$, as in Section 1.2. Multiply the whole model on the left by $\mathbf{L}^{-1}$:

$$
\mathbf{L}^{-1}\mathbf{Y} = \mathbf{L}^{-1}\mathbf{X}\boldsymbol{\beta} + \mathbf{L}^{-1}(\mathbf{u} + \boldsymbol{\varepsilon})
$$

By Tool 1, the transformed errors have covariance $\mathbf{L}^{-1}\boldsymbol{\Sigma}(\mathbf{L}^{-1})' = \mathbf{L}^{-1}\mathbf{L}\mathbf{L}'(\mathbf{L}')^{-1} = \mathbf{I}_n$. The transformed model is a classical linear model with response $\mathbf{L}^{-1}\mathbf{Y}$, design matrix $\mathbf{L}^{-1}\mathbf{X}$, and independent errors of variance 1. Transforming by $\mathbf{L}^{-1}$ decorrelates the data.

For a classical model, the Gauss-Markov theorem from your regression course says that OLS is the best linear unbiased estimator. Applying OLS to the transformed model, and using $(\mathbf{L}^{-1})'\mathbf{L}^{-1} = (\mathbf{L}\mathbf{L}')^{-1} = \boldsymbol{\Sigma}^{-1}$,

$$
\begin{aligned}
\hat{\boldsymbol{\beta}}_{\text{GLS}} &= \left((\mathbf{L}^{-1}\mathbf{X})'\mathbf{L}^{-1}\mathbf{X}\right)^{-1}(\mathbf{L}^{-1}\mathbf{X})'\mathbf{L}^{-1}\mathbf{Y} \\
&= (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y}
\end{aligned}
$$

This is the **generalized least squares** (GLS) estimator. Its variance is the classical formula for the transformed model with error variance 1:

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}}) = \left((\mathbf{L}^{-1}\mathbf{X})'\mathbf{L}^{-1}\mathbf{X}\right)^{-1} = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}
$$

It is also the estimator that minimizes $(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$, as previewed in Section 0.4.5, since that criterion is the residual sum of squares of the transformed model.

Because $\mathbf{L}$ is invertible, any linear estimator in the original data is also a linear estimator in the transformed data, and the other way around. So Gauss-Markov applied to the transformed model gives the result we want: **GLS is the best linear unbiased estimator of $\boldsymbol{\beta}$ under spatial correlation.** Any other linear unbiased estimator, OLS included, has a variance at least as large. This result is called Aitken's theorem; Section 1.3.6 proves it without the transformation.

How much worse is OLS? It depends on the covariate. In the experiment of Section 1.7.2, the OLS slope has 2.59 times the variance of the GLS slope for floor number, and 9.02 times for distance to transit. So even when OLS standard errors happen to be roughly right, as they are for floor number, OLS throws away information.

OLS and GLS coincide in special cases, for example when the columns of $\mathbf{X}$ are eigenvectors of $\boldsymbol{\Sigma}$ (Exercise 6). The block of Section 1.3.2 is one: $\mathbf{1}$ is an eigenvector, so $\boldsymbol{\Sigma}^{-1}\mathbf{1} = \mathbf{1}/1.9789$, and the GLS estimator $\mathbf{1}'\boldsymbol{\Sigma}^{-1}\mathbf{Y}/\mathbf{1}'\boldsymbol{\Sigma}^{-1}\mathbf{1}$ reduces to $\bar{Y}$. There, the sample mean is already the best estimator; its only problem is the reported standard error.

GLS needs $\boldsymbol{\Sigma}$, which in practice is unknown. Estimating it is the subject of Chapters 3 and 5, and GLS reappears in Chapter 4 as the first stage of spatial prediction.

### 1.3.6 Proof of Aitken's Theorem (Optional)

*This subsection proves the result of Section 1.3.5 directly, without transforming the data. It can be skipped on a first reading.*

**Theorem (Aitken).** Let $\mathbf{B}\mathbf{Y}$ be any estimator of $\boldsymbol{\beta}$ that is linear in $\mathbf{Y}$ and unbiased for every $\boldsymbol{\beta}$. Then $\operatorname{Var}(\mathbf{B}\mathbf{Y}) - \operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}})$ is positive semi-definite.

**Proof.** Unbiasedness for every $\boldsymbol{\beta}$ means $\mathbf{B}\mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}$ for all $\boldsymbol{\beta}$, that is, $\mathbf{B}\mathbf{X} = \mathbf{I}_p$. Write $\mathbf{B}_{\text{GLS}} = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}$ for the GLS matrix, and $\mathbf{B} = \mathbf{B}_{\text{GLS}} + \boldsymbol{\Delta}$, so that $\boldsymbol{\Delta}$ measures how far $\mathbf{B}$ is from GLS. Both $\mathbf{B}$ and $\mathbf{B}_{\text{GLS}}$ satisfy the unbiasedness condition, so

$$
\boldsymbol{\Delta}\mathbf{X} = \mathbf{B}\mathbf{X} - \mathbf{B}_{\text{GLS}}\mathbf{X} = \mathbf{I}_p - \mathbf{I}_p = \mathbf{0}
$$

By Tool 1,

$$
\operatorname{Var}(\mathbf{B}\mathbf{Y}) = (\mathbf{B}_{\text{GLS}} + \boldsymbol{\Delta})\boldsymbol{\Sigma}(\mathbf{B}_{\text{GLS}} + \boldsymbol{\Delta})' = \mathbf{B}_{\text{GLS}}\boldsymbol{\Sigma}\mathbf{B}_{\text{GLS}}' + \mathbf{B}_{\text{GLS}}\boldsymbol{\Sigma}\boldsymbol{\Delta}' + \boldsymbol{\Delta}\boldsymbol{\Sigma}\mathbf{B}_{\text{GLS}}' + \boldsymbol{\Delta}\boldsymbol{\Sigma}\boldsymbol{\Delta}'
$$

The cross terms vanish:

$$
\mathbf{B}_{\text{GLS}}\boldsymbol{\Sigma}\boldsymbol{\Delta}' = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}\boldsymbol{\Delta}' = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}(\boldsymbol{\Delta}\mathbf{X})' = \mathbf{0}
$$

and $\boldsymbol{\Delta}\boldsymbol{\Sigma}\mathbf{B}_{\text{GLS}}'$ is its transpose. Since $\mathbf{B}_{\text{GLS}}\boldsymbol{\Sigma}\mathbf{B}_{\text{GLS}}' = \operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}})$,

$$
\operatorname{Var}(\mathbf{B}\mathbf{Y}) - \operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}}) = \boldsymbol{\Delta}\boldsymbol{\Sigma}\boldsymbol{\Delta}'
$$

This difference is positive semi-definite by Tool 2: for any vector $\mathbf{a}$, $\mathbf{a}'\boldsymbol{\Delta}\boldsymbol{\Sigma}\boldsymbol{\Delta}'\mathbf{a} = \operatorname{Var}(\mathbf{a}'\boldsymbol{\Delta}\mathbf{Y}) \ge 0$. When $\boldsymbol{\Sigma}$ is positive definite, equality for every $\mathbf{a}$ requires $\boldsymbol{\Delta} = \mathbf{0}$, so GLS is the unique best estimator. $\square$

OLS is the case $\mathbf{B} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$, and Exercise 5 asks you to compute its $\boldsymbol{\Delta}$ explicitly.

---

## 1.4 Effective Sample Size

### 1.4.1 Definition

A single number summarizes how much correlation costs: the number of independent observations that would carry the same information. For the sample mean it is

$$
n_{\text{eff}} = \frac{C(0)}{\operatorname{Var}(\bar{Y})}
$$

If the observations were independent, $\operatorname{Var}(\bar{Y}) = C(0)/n$ and $n_{\text{eff}} = n$. With correlation, $\operatorname{Var}(\bar{Y}) = \frac{1}{n^2}\mathbf{1}'\boldsymbol{\Sigma}\mathbf{1}$ by Tool 1. Taking $\tau^2 = 0$ for simplicity, so that $\boldsymbol{\Sigma} = \sigma^2\mathbf{R}$ and $C(0) = \sigma^2$,

$$
n_{\text{eff}} = \frac{n^2}{\mathbf{1}'\mathbf{R}\mathbf{1}} = \frac{n^2}{\sum_{i=1}^n\sum_{j=1}^n R_{ij}}
$$

Every positive correlation in $\mathbf{R}$ enlarges the denominator and lowers $n_{\text{eff}}$. The extreme cases bracket the possibilities: with no correlation the double sum is $n$ and $n_{\text{eff}} = n$; with perfect correlation every entry is 1, the double sum is $n^2$, and $n_{\text{eff}} = 1$. The block of Section 1.3.2 has $n_{\text{eff}} = 1/0.4947 \approx 2.02$: four correlated apartments tell us about as much about the average rent level as two independent ones. Exercise 2 of Chapter 0 gives another case: when every pair has correlation $r$, $n_{\text{eff}} = n/(1 + (n-1)r)$, which can never exceed $1/r$ no matter how large $n$ is.

### 1.4.2 A Transect: A Closed Form

Spatial correlation is not constant across pairs; it decays with distance. The simplest setting that shows the consequences is a transect. Picture noise monitors placed along a highway at equal spacing $\Delta$, $n$ of them in a line. With the exponential correlation function, two monitors $k$ steps apart have correlation

$$
r(k\Delta; \phi) = e^{-k\Delta/\phi} = r_1^k, \qquad r_1 = e^{-\Delta/\phi}
$$

where $r_1$ is the correlation between neighbors. The exponential function turns distance into powers, which makes the double sum tractable. There are $n$ pairs at lag 0, and $n - k$ ordered pairs in each direction at lag $k$, so

$$
\mathbf{1}'\mathbf{R}\mathbf{1} = n + 2\sum_{k=1}^{n-1}(n - k)\,r_1^k
$$

For large $n$, the terms with $k$ near $n$ are tiny, and the sum is close to $n\sum_{k=1}^\infty r_1^k = n\,r_1/(1 - r_1)$. Then

$$
\mathbf{1}'\mathbf{R}\mathbf{1} \approx n\left(1 + \frac{2r_1}{1 - r_1}\right) = n\,\frac{1 + r_1}{1 - r_1}, \qquad n_{\text{eff}} \approx n\,\frac{1 - r_1}{1 + r_1}
$$

With neighbor correlation $r_1 = 0.5$, a line of monitors is worth about a third of its length in independent observations; with $r_1 = 0.8$, about a ninth. The code below compares the exact value with the approximation.

```python
import numpy as np
from scipy import stats


def neff_mean(R):
    """Effective sample size of the sample mean for correlation matrix R."""
    n = R.shape[0]
    return n**2 / R.sum()


for r1 in (0.5, 0.8):
    for n in (25, 100, 400):
        k = np.arange(n)
        R = r1 ** np.abs(k[:, None] - k[None, :])
        approx = n * (1 - r1) / (1 + r1)
        print(f"r1={r1}  n={n:3d}  n_eff={neff_mean(R):7.2f}  "
              f"approx={approx:7.2f}")
```

| $r_1$ | $n = 25$ | $n = 100$ | $n = 400$ | Approximation for $n = 400$ |
| :--- | :--- | :--- | :--- | :--- |
| 0.5 | 8.80 | 33.78 | 133.78 | 133.33 |
| 0.8 | 3.38 | 11.63 | 44.94 | 44.44 |

Here $n_{\text{eff}}$ grows in proportion to $n$. That is because the line of monitors gets longer as $n$ grows: each new monitor is placed farther down the highway, at the same spacing, and adds some genuinely new information.

### 1.4.3 Infill: More Points in the Same Place

Now change the experiment. Keep the monitors on a fixed stretch of highway, rescaled to run from 0 to 1, and add monitors inside it, so the spacing $\Delta = 1/n$ shrinks as $n$ grows. This is **infill** sampling, and it is what the district's noise-monitor grid did in Section 0.3.1.

As $n$ grows, $\bar{Y}$ approaches the average of the process over the whole stretch, and its variance approaches a fixed number. Replacing the double sum by a double integral (the limit of the Riemann sum $\frac{1}{n^2}\sum_i\sum_j R_{ij}$),

$$
\operatorname{Var}(\bar{Y}) \to \sigma^2\int_0^1\!\!\int_0^1 e^{-\lvert s - t \rvert/\phi}\,ds\,dt
$$

The integrand is symmetric in $s$ and $t$, so the double integral is twice the integral over the triangle $t < s$:

$$
\begin{aligned}
\int_0^1\!\!\int_0^1 e^{-\lvert s - t \rvert/\phi}\,ds\,dt &= 2\int_0^1\!\!\int_0^s e^{-(s - t)/\phi}\,dt\,ds \\
&= 2\int_0^1 \phi\left(1 - e^{-s/\phi}\right)ds \\
&= 2\phi\left[1 - \phi\left(1 - e^{-1/\phi}\right)\right]
\end{aligned}
$$

Therefore the effective sample size has a ceiling:

$$
n_{\text{eff}} \to \frac{1}{2\phi\left[1 - \phi\left(1 - e^{-1/\phi}\right)\right]}
$$

With $\phi = 0.2$ the ceiling is about 3.12. No matter how many monitors are packed into the stretch, their average carries no more information about the stretch's true average noise level than about three independent measurements would.

```python
phi = 0.2
limit = 1 / (2 * phi * (1 - phi * (1 - np.exp(-1 / phi))))
for n in (10, 100, 1000, 4000):
    t = (np.arange(n) + 0.5) / n
    R = np.exp(-np.abs(t[:, None] - t[None, :]) / phi)
    print(f"n={n:4d}  n_eff={neff_mean(R):.3f}")
print(f"limit: {limit:.3f}")
```

The output shows $n_{\text{eff}}$ equal to 3.026 with 10 monitors, 3.119 with 100, and 3.120 with 1,000 and 4,000, matching the limit of 3.120.

The contrast between Sections 1.4.2 and 1.4.3 is one of the most important ideas in spatial statistics. Adding data by **expanding the region** keeps adding information. Adding data by **filling in the same region** quickly stops helping for questions about region-wide quantities such as a mean or a regression slope. The process has only so many independent fluctuations within a fixed region, and denser sampling measures the same fluctuations more finely. Chapter 5 returns to this when it asks which covariance parameters can be estimated well from a fixed region.

### 1.4.4 Effective Sample Size Depends on the Question

There is no single effective sample size for a data set. The definition above is for the sample mean. For a regression slope the relevant comparison is the variance ratio of Section 1.3.4, which depends on the covariate. For the best estimator of the mean, the GLS estimator $\hat{\mu}_{\text{GLS}} = \mathbf{1}'\mathbf{R}^{-1}\mathbf{Y}/\mathbf{1}'\mathbf{R}^{-1}\mathbf{1}$, the effective sample size is $\mathbf{1}'\mathbf{R}^{-1}\mathbf{1}$, which is never smaller than $n^2/\mathbf{1}'\mathbf{R}\mathbf{1}$ (Exercise 7). And for predicting the value at an unmeasured location, which is the subject of Chapter 4, dense sampling helps a great deal (Section 4.8.5 shows how the prediction error falls as listings are added). Effective sample size is a way of asking how much a particular estimate is worth, not a property of the data alone.

---

## 1.5 The Three Spatial Data Structures as Matrix Geometries

The three kinds of spatial data from Section 0.1.3 lead to three different matrix structures. Seeing them side by side explains why the book's parts use different tools.

### 1.5.1 Continuous Geostatistical Fields: Dense Distance Matrices

For geostatistical data, such as rents at listed apartments or noise levels at monitors, the locations form an $n \times 2$ coordinate matrix $\mathbf{S}$ whose rows are $\mathbf{s}_i'$. Dependence is described through the $n \times n$ matrix of pairwise distances $\mathbf{H}$, with entries $h_{ij} = \lVert\mathbf{s}_i - \mathbf{s}_j\rVert$, and the covariance matrix is a function of it: $\Sigma_{ij} = C(h_{ij})$.

Both $\mathbf{H}$ and $\boldsymbol{\Sigma}$ are **dense**: every pair of locations has a distance and, for correlation functions like the exponential, a nonzero covariance. Storing $\boldsymbol{\Sigma}$ takes $n^2$ numbers. With $n = 10{,}000$ that is $10^8$ numbers, about 800 MB in double precision, and factoring it takes on the order of $n^3$ operations. Part V is about getting around this.

### 1.5.2 Discrete Areal Lattices: Sparse Adjacency Matrices

For areal data, such as overdose deaths by county or income by census tract, there is one value per region, and the natural description of the geometry is which regions touch. The **adjacency matrix** $\mathbf{A}$ has $A_{ij} = 1$ if regions $i$ and $j$ share a border and $A_{ij} = 0$ otherwise, with zeros on the diagonal.

Most regions touch only a few others, so $\mathbf{A}$ is **sparse**: almost all of its entries are zero. On a $10 \times 10$ grid of square regions where squares sharing an edge count as neighbors, each region has 2, 3, or 4 neighbors, and only 360 of the 10,000 entries of $\mathbf{A}$ are nonzero.

```python
def rook_adjacency(m):
    """Binary adjacency matrix of an m-by-m grid of square regions."""
    A = np.zeros((m * m, m * m))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            if i + 1 < m:
                A[k, k + m] = A[k + m, k] = 1
            if j + 1 < m:
                A[k, k + 1] = A[k + 1, k] = 1
    return A


A = rook_adjacency(10)
print("nonzero entries:", int(A.sum()), "of", A.size)
print("neighbors per region:", np.unique(A.sum(axis=1)))
```

Part III builds models directly from $\mathbf{A}$. Chapter 8 in particular describes dependence through a precision matrix (the inverse of a covariance matrix) that has the same sparsity pattern as $\mathbf{A}$, which is what makes large areal models fast to fit.

### 1.5.3 Spatial Point Patterns: Random Coordinates and Grid Counts

For point patterns, such as immune cells on a tumor slide or the addresses of disease cases, there is no measured value at fixed locations. The coordinate matrix $\mathbf{S}$ is itself the random outcome, and even the number of rows $n$ is random.

Two matrix views are useful. The first keeps the points and works with the distance matrix $\mathbf{H}$ between them: nearest-neighbor distances and counts of pairs within a given distance are computed directly from $\mathbf{H}$ (Chapter 10). The second lays a fine grid of pixels over the region and records the number of points in each pixel. That turns a point pattern into a vector of counts on a regular lattice, a form of areal data, and it is how Chapter 11 fits point patterns with Poisson regression.

---

## 1.6 Coordinates and Distance

Every spatial method in this book starts from distances between locations. Computing them correctly, and quickly, deserves a little care.

### 1.6.1 Vectorized Distance Matrices

The direct way to build $\mathbf{H}$ is a double loop over pairs, which is slow in Python for large $n$. A matrix identity removes the loop. Expanding the squared distance,

$$
h_{ij}^2 = (\mathbf{s}_i - \mathbf{s}_j)'(\mathbf{s}_i - \mathbf{s}_j) = \mathbf{s}_i'\mathbf{s}_i - 2\mathbf{s}_i'\mathbf{s}_j + \mathbf{s}_j'\mathbf{s}_j
$$

The middle term is entry $(i, j)$ of the matrix $\mathbf{S}\mathbf{S}'$. The first and last terms are squared lengths of rows of $\mathbf{S}$. Writing $\mathbf{q}$ for the vector with entries $q_i = \mathbf{s}_i'\mathbf{s}_i$, the whole matrix of squared distances is

$$
\mathbf{q}\mathbf{1}' - 2\,\mathbf{S}\mathbf{S}' + \mathbf{1}\mathbf{q}'
$$

which NumPy computes with one matrix product and broadcasting.

One numerical detail matters. The identity subtracts large numbers to get small ones, and rounding error can leave a squared distance that should be exactly 0 as a tiny negative number. Taking its square root then returns `nan`. The code below shows such values on the diagonal, then defines a distance function that clips negatives to zero and sets the diagonal to exactly zero.

```python
rng = np.random.default_rng(1)
S = rng.uniform(size=(5, 2))
q = np.sum(S**2, axis=1)
H2 = q[:, None] - 2 * S @ S.T + q[None, :]
print("diagonal of the squared distances:", np.diag(H2))


def dist(S):
    """Matrix of Euclidean distances between the rows of S."""
    q = np.sum(S**2, axis=1)
    H2 = q[:, None] - 2 * S @ S.T + q[None, :]
    H = np.sqrt(np.maximum(H2, 0.0))
    np.fill_diagonal(H, 0.0)
    return H


H_loop = np.array([[np.linalg.norm(si - sj) for sj in S] for si in S])
print("matches the double loop:", np.allclose(dist(S), H_loop))
```

In this run the diagonal contains values of $-2.2 \times 10^{-16}$ and $2.2 \times 10^{-16}$ alongside exact zeros: rounding error at the level of machine precision. After the clipping, the function agrees with the double loop. The same subtraction can also leave rounding errors, of order $10^{-8}$ after the square root, for two different locations that are exactly identical, such as several apartments in one building; Section 5.6.3 shows why that matters for the covariance matrix.

### 1.6.2 Why Raw Latitude and Longitude Mislead

Rental listings, like most real location data, arrive with latitude and longitude in degrees. It is tempting to treat them as $(x, y)$ coordinates and use Euclidean distance. That is wrong, and not by a small amount.

A degree of latitude is always about 111.2 km: on a sphere of radius $R_E = 6371$ km, the Earth's mean radius, it is $2\pi R_E/360$. A degree of longitude is that length times the cosine of the latitude, because the circles of latitude shrink toward the poles. At 45 degrees north, a degree of longitude is only about 78.6 km. Euclidean distance on raw degrees treats the two as equal. Two points one degree apart east-west and two points one degree apart north-south get the same "distance," although the true distances differ by more than 40%.

For a study region a few hundred kilometers across, such as one city and its suburbs, a simple fix works well: an **equirectangular projection** centered at a reference latitude $\text{lat}_0$ near the middle of the region converts degrees to kilometers,

$$
x = R_E\,\text{lon}\,\cos(\text{lat}_0), \qquad y = R_E\,\text{lat}
$$

with longitude $\text{lon}$ and latitude $\text{lat}$ in radians. Euclidean distance on $(x, y)$ is then accurate near $\text{lat}_0$ and degrades slowly away from it. Professional work often uses standard projections such as UTM zones, which are more accurate over larger areas but more complicated to compute. The idea is the same: convert to a flat coordinate system measured in distance units before computing distances.

### 1.6.3 Great-Circle and Chordal Distance

For regions too large to flatten, such as a continent, distance must account for the curvature of the Earth. The **great-circle distance** $d_{\text{gc}}$ is the length of the shortest path along the surface. The haversine formula computes it stably. For points at latitudes $\text{lat}_1, \text{lat}_2$ and longitudes $\text{lon}_1, \text{lon}_2$ (in radians),

$$
d_{\text{gc}} = 2R_E\arcsin\sqrt{\sin^2\!\left(\frac{\text{lat}_2 - \text{lat}_1}{2}\right) + \cos(\text{lat}_1)\cos(\text{lat}_2)\sin^2\!\left(\frac{\text{lon}_2 - \text{lon}_1}{2}\right)}
$$

The **chordal distance** $d_{\text{ch}}$ is the length of the straight line through the Earth between the two points, that is, ordinary Euclidean distance in three dimensions. It follows from the great-circle distance by elementary geometry. The two points and the Earth's center form an isosceles triangle with two sides of length $R_E$ and apex angle $d_{\text{gc}}/R_E$, so the base is

$$
d_{\text{ch}} = 2R_E\sin\!\left(\frac{d_{\text{gc}}}{2R_E}\right)
$$

For nearby points the two are nearly identical; for distant points the chord is shorter.

```python
R_E = 6371.0  # mean Earth radius in km


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; inputs in degrees."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
    return 2 * R_E * np.arcsin(np.sqrt(h))


def chordal(lat1, lon1, lat2, lon2):
    """Straight-line distance through the Earth in km."""
    d = haversine(lat1, lon1, lat2, lon2)
    return 2 * R_E * np.sin(d / (2 * R_E))


def local_projection(lat, lon, lat0):
    """Equirectangular projection to km, centered on latitude lat0."""
    x = R_E * np.radians(lon) * np.cos(np.radians(lat0))
    y = R_E * np.radians(lat)
    return np.array([x, y])
```

### 1.6.4 A Warning Carried Into Chapter 2

Choosing a distance is not only a matter of accuracy. It also affects whether a covariance model is valid.

Chapter 2 shows that a covariance function must produce a positive semi-definite matrix for every possible set of locations (Tool 2 in Section 0.4). The standard families, including the exponential, Gaussian, and Matérn, are known to satisfy this for points in ordinary Euclidean space of up to three dimensions. Chordal distance is Euclidean distance in three dimensions, and points on the Earth's surface are points in three-dimensional space that happen to lie on a sphere. So any of these families remains valid when it is computed from chordal distance.

Great-circle distance is not Euclidean distance in any space, and validity does not carry over automatically. A known result, which we take from the literature rather than prove, is that the Matérn family with smoothness parameter $\nu$ (introduced in Chapter 2) is valid with great-circle distance only when $\nu \le 1/2$ [@gneiting2013]. The exponential model, which is the case $\nu = 1/2$, is safe; smoother models, including the Gaussian, can produce covariance matrices with negative eigenvalues.

The practical rule for this book: for a region a few hundred kilometers across, project to flat coordinates and use Euclidean distance; for larger regions, use chordal distance with any standard covariance family, or great-circle distance only with the exponential.

---

## 1.7 Walkthrough and Lab 1

### 1.7.1 Checking the Block Calculations in Code

Sections 1.3.2 and 1.3.3 worked the four-apartment block by hand. The distance matrix for the block, with apartments ordered as there, is

$$
\mathbf{H} = \begin{bmatrix} 0 & 1 & 1 & \sqrt{2} \\ 1 & 0 & \sqrt{2} & 1 \\ 1 & \sqrt{2} & 0 & 1 \\ \sqrt{2} & 1 & 1 & 0 \end{bmatrix}
$$

and $\boldsymbol{\Sigma}$ has entries $e^{-h_{ij}}$. The code below rebuilds $\boldsymbol{\Sigma}$ from the coordinates and reproduces every number: the variance of $\bar{Y}$, the average of $s^2$, the effective sample size, the four eigenvalues, and the variance ratio of the diagonal-corners covariate from Section 1.3.4.

```python
S4 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Sigma4 = np.exp(-dist(S4) / 1.0)
one = np.ones(4)
P_c = np.eye(4) - np.outer(one, one) / 4
var_mean = one @ Sigma4 @ one / 16
E_s2 = np.trace(P_c @ Sigma4) / 3
print(np.round(Sigma4, 4))
print("true Var(Ybar):", round(var_mean, 4))
print("E[s^2]:", round(E_s2, 4), " reported Var:", round(E_s2 / 4, 4))
print("n_eff:", round(1 / var_mean, 3))
print("eigenvalues:", np.round(np.linalg.eigvalsh(Sigma4), 4))

x_diag = np.array([1.0, 0.0, 0.0, 1.0])
xc = x_diag - x_diag.mean()
print("variance ratio, diagonal corners:",
      round(xc @ Sigma4 @ xc / (xc @ xc), 4))
```

The output matches the hand calculations: $\operatorname{Var}(\bar{Y}) = 0.4947$, $\operatorname{E}[s^2] = 0.6737$ with a reported variance of 0.1684, $n_{\text{eff}} = 2.021$, eigenvalues 0.5074, 0.7569, 0.7569, and 1.9789 (NumPy lists them in increasing order), and a variance ratio of 0.5074 for the diagonal-corners covariate.

### 1.7.2 How Often Does the Naive Interval Cover?

Now a realistic size. Put 400 apartments on a $20 \times 20$ grid covering a district, rescaled to the unit square, and simulate rents from

$$
\mathbf{Y} = \beta_0\mathbf{1} + \beta_1\mathbf{x} + \mathbf{u} + \boldsymbol{\varepsilon}
$$

with $\beta_0 = 1$, $\beta_1 = 0.5$, $\sigma^2 = 1$, $\tau^2 = 0.2$, and exponential correlation with $\phi = 0.2$. The rents are simulated, in standardized units, so that we know the true slope. We try the two covariates of Section 1.3.4, each generated once and then held fixed:

- **floor number**, standardized, which has nothing to do with location, so we draw it independently from $N(0, 1)$ at each apartment; and
- **distance to transit**, standardized, which varies smoothly across the district, so we draw it from a spatial process with the same exponential correlation, independently of the rents' errors.

For each covariate we simulate 1,000 data sets, fit OLS, and record whether the naive 95% interval $\hat{\beta}_1 \pm t_{0.975,\,398}\,\text{SE}$ contains the true $\beta_1 = 0.5$. We also compute, exactly rather than by simulation, the variance ratio of Section 1.3.4, the true standard deviation of $\hat{\beta}_1$ from the sandwich formula, and the ratio of the OLS and GLS variances.

```python
def grid(m):
    """Centers of an m-by-m grid of cells in the unit square."""
    g = (np.arange(m) + 0.5) / m
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


rng = np.random.default_rng(2027)
S = grid(20)
n = len(S)
H = dist(S)
sigma2, tau2, phi = 1.0, 0.2, 0.2
Sigma = sigma2 * np.exp(-H / phi) + tau2 * np.eye(n)
L = np.linalg.cholesky(Sigma)

x_floor = rng.standard_normal(n)
# tiny jitter on the diagonal; explained in Section 2.5.1
L_x = np.linalg.cholesky(np.exp(-H / 0.2) + 1e-10 * np.eye(n))
x_transit = L_x @ rng.standard_normal(n)


def ols_coverage(x, reps=1000, beta=(1.0, 0.5)):
    """Coverage of the naive 95% interval for the slope, and summaries."""
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    V_true = XtX_inv @ X.T @ Sigma @ X @ XtX_inv
    V_gls = np.linalg.inv(X.T @ np.linalg.solve(Sigma, X))
    t_crit = stats.t.ppf(0.975, n - 2)
    covered, se_sum = 0, 0.0
    for _ in range(reps):
        y = X @ np.array(beta) + L @ rng.standard_normal(n)
        b = XtX_inv @ X.T @ y
        resid = y - X @ b
        se = np.sqrt(resid @ resid / (n - 2) * XtX_inv[1, 1])
        se_sum += se
        covered += abs(b[1] - beta[1]) <= t_crit * se
    xc = x - x.mean()
    ratio = xc @ Sigma @ xc / ((sigma2 + tau2) * (xc @ xc))
    return dict(ratio=ratio, true_sd=np.sqrt(V_true[1, 1]),
                mean_se=se_sum / reps, coverage=covered / reps,
                ols_over_gls=V_true[1, 1] / V_gls[1, 1])


for name, x in (("floor", x_floor), ("transit", x_transit)):
    r = ols_coverage(x)
    print(f"{name:8s} ratio={r['ratio']:.2f}  sd={r['true_sd']:.4f}  "
          f"se={r['mean_se']:.4f}  coverage={r['coverage']:.3f}  "
          f"ols/gls={r['ols_over_gls']:.2f}")
```

| Covariate | Variance ratio | True SD of $\hat{\beta}_1$ | Average reported SE | Coverage of 95% interval | OLS variance / GLS variance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Floor number | 0.99 | 0.0511 | 0.0480 | 93.2% | 2.59 |
| Distance to transit | 11.33 | 0.1903 | 0.0519 | 41.5% | 9.02 |

The results line up with Section 1.3.

- **Floor number.** The variance ratio is 0.99, so the independent-data formula for the slope's variance is nearly right. The reported standard error is slightly too small on average (0.0480 against a true 0.0511), mostly because $s^2$ underestimates the variance, and coverage is 93.2%, a little below the nominal 95%. Even so, OLS is inefficient: its variance is 2.59 times that of GLS.
- **Distance to transit.** The variance ratio is 11.33. The true standard deviation of the slope is 0.1903, but the average reported standard error is 0.0519, more than three and a half times too small. The naive interval covers the truth in only 41.5% of the data sets. Put the other way, a test of the true value $\beta_1 = 0.5$ at the 5% level would reject it 58.5% of the time. This is the regression version of the spurious correlation in Section 0.3.2: an analyst would routinely report a transit effect that is far more precise than the data justify.

### 1.7.3 Comparing Distances

Finally, compare four ways of measuring distance for three pairs of listings. Point A is at 45 degrees north, 93 degrees west. Point B is one degree east of A, point C is one degree north of A, and point D is far away, at 60 degrees north, 40 degrees west. The projection is centered at 45 degrees north.

```python
pts = {"A": (45.0, -93.0), "B": (45.0, -92.0),
       "C": (46.0, -93.0), "D": (60.0, -40.0)}
for p, q in (("A", "B"), ("A", "C"), ("A", "D")):
    (la1, lo1), (la2, lo2) = pts[p], pts[q]
    deg = np.hypot(la2 - la1, lo2 - lo1)
    proj = np.linalg.norm(local_projection(la1, lo1, 45.0)
                          - local_projection(la2, lo2, 45.0))
    print(f"{p}{q}: degrees={deg:6.2f}  projected={proj:7.1f}  "
          f"great-circle={haversine(la1, lo1, la2, lo2):7.1f}  "
          f"chordal={chordal(la1, lo1, la2, lo2):7.1f}")
```

| Pair | Euclidean on degrees | Projected (km) | Great-circle (km) | Chordal (km) |
| :--- | :--- | :--- | :--- | :--- |
| A to B (1 degree east) | 1.00 | 78.6 | 78.6 | 78.6 |
| A to C (1 degree north) | 1.00 | 111.2 | 111.2 | 111.2 |
| A to D (far) | 55.08 | 4488.6 | 3824.7 | 3767.6 |

For the nearby pairs, the projection, great-circle, and chordal distances agree to the reported precision, while raw degrees call the two pairs equally far apart although one is 41% farther than the other. For the distant pair the projection is 17% too long, because the equirectangular projection stretches distances far from its reference latitude, and the chordal distance is about 1.5% shorter than the great-circle distance.

---

## Exercises

**Analytical**

1. Show that $\operatorname{E}[s^2] = \operatorname{tr}\!\left((\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})\boldsymbol{\Sigma}\right)/(n - p)$ when $\operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma}$, using the result of Section 1.1.1. Check that it reduces to $\tau^2$ when $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n$.

2. Suppose all $n$ observations have variance $C(0)$ and every pair has correlation $r > 0$. For the intercept-only model, find $\operatorname{Var}(\bar{Y})$, $\operatorname{E}[s^2]$, and the ratio of the true variance of $\bar{Y}$ to the average reported variance $\operatorname{E}[s^2]/n$. Show that the ratio grows without bound as $n \to \infty$ and explain what that means for a naive confidence interval.

3. Place four noise monitors on a line at positions 0, 1, 2, 3, with $\sigma^2 = 1$, $\tau^2 = 0$, and exponential correlation with $\phi = 1$. (a) For the covariate $\mathbf{x} = (0, 1, 0, 1)'$, which alternates between neighbors, compute the variance ratio of Section 1.3.4 by hand and show that it is less than 1. (b) Do the same for the covariate $\mathbf{x} = (0, 1, 2, 3)'$, which increases steadily along the line. (c) Explain the difference using the eigenvector argument of Section 1.3.3.

4. Show that the variance ratio of Section 1.3.4 always lies between $\lambda_n/C(0)$ and $\lambda_1/C(0)$. Then, for 50 equally spaced points on a line with exponential correlation and $\phi$ equal to 0.2 times the length of the line, compute the eigenvectors with `np.linalg.eigh`, plot those belonging to the three largest and three smallest eigenvalues, and describe how smooth each one is.

5. For the OLS choice $\mathbf{B} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ in the proof of Section 1.3.6, write down $\boldsymbol{\Delta}$ explicitly, check that $\boldsymbol{\Delta}\mathbf{X} = \mathbf{0}$, and confirm that the OLS sandwich minus the GLS variance equals $\boldsymbol{\Delta}\boldsymbol{\Sigma}\boldsymbol{\Delta}'$.

6. Show that if $\boldsymbol{\Sigma}\mathbf{X} = \mathbf{X}\mathbf{N}$ for some invertible $p \times p$ matrix $\mathbf{N}$ (that is, $\boldsymbol{\Sigma}$ maps the columns of $\mathbf{X}$ into combinations of themselves), then $\hat{\boldsymbol{\beta}}_{\text{GLS}} = \hat{\boldsymbol{\beta}}$. Use this to explain why $\bar{Y}$ is the GLS estimator for the block of Section 1.3.2.

7. Use the Cauchy-Schwarz inequality, applied to the vectors $\mathbf{R}^{1/2}\mathbf{1}$ and $\mathbf{R}^{-1/2}\mathbf{1}$, to show that $\mathbf{1}'\mathbf{R}^{-1}\mathbf{1} \ge n^2/\mathbf{1}'\mathbf{R}\mathbf{1}$. Interpret the result in terms of the two effective sample sizes in Section 1.4.4. (Here $\mathbf{R}^{1/2} = \mathbf{U}\boldsymbol{\Lambda}^{1/2}\mathbf{U}'$, built from the eigendecomposition of $\mathbf{R}$.)

8. For the transect of Section 1.4.2, find the exact value of $\sum_{k=1}^{n-1}(n - k)r_1^k$ and use it to show that $n_{\text{eff}} = n(1 - r_1)/(1 + r_1) + O(1)$ as $n \to \infty$.

9. Derive the chordal distance formula $d_{\text{ch}} = 2R_E\sin(d_{\text{gc}}/(2R_E))$ from the isosceles triangle described in Section 1.6.3. Then show that $d_{\text{gc}} - d_{\text{ch}} \approx d_{\text{gc}}^3/(24R_E^2)$ for short distances, and evaluate this for $d_{\text{gc}} = 100$ km and $1{,}000$ km.

10. Section 1.3.5 noted that the GLS estimator minimizes the generalized sum of squares criterion $S(\boldsymbol{\beta}) = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$. (a) Expand $S(\boldsymbol{\beta})$ into four terms. (b) Using matrix differentiation (Tool 4 from Section 0.4.4), show that the gradient with respect to $\boldsymbol{\beta}$ is $-2\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y} + 2\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\boldsymbol{\beta}$. (c) Set this gradient to zero to derive the generalized normal equations $\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y}$. (d) Compute the Hessian matrix of second derivatives, and explain why positive definiteness of $\boldsymbol{\Sigma}$ guarantees that $\hat{\boldsymbol{\beta}}_{\text{GLS}}$ is a strict global minimum.

11. Section 1.3.5 transformed the model using the Cholesky factor $\mathbf{L}^{-1}$, where $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$. An alternative decorrelation uses the symmetric inverse square root $\boldsymbol{\Sigma}^{-1/2} = \mathbf{U}\boldsymbol{\Lambda}^{-1/2}\mathbf{U}'$, where $\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}'$ is the eigendecomposition of Section 1.3.3. (a) Show that $\operatorname{Var}(\boldsymbol{\Sigma}^{-1/2}\mathbf{Y}) = \mathbf{I}_n$. (b) Apply OLS to the transformed model $\boldsymbol{\Sigma}^{-1/2}\mathbf{Y} = \boldsymbol{\Sigma}^{-1/2}\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\Sigma}^{-1/2}(\mathbf{u} + \boldsymbol{\varepsilon})$ and show that the resulting estimator and its covariance matrix are identical to $\hat{\boldsymbol{\beta}}_{\text{GLS}}$ and $(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}$. (c) Explain the geometric difference between the two transformations: what coordinate axes does each transformation align the uncorrelated errors with?

12. Consider two locations $\mathbf{s}_1, \mathbf{s}_2$ with correlation matrix $\mathbf{R} = \begin{bmatrix} 1 & r \\ r & 1 \end{bmatrix}$ ($0 < r < 1$) and error variance $\sigma^2$ (with $\tau^2 = 0$). We fit a simple linear regression with an intercept and one covariate $\mathbf{x} = (x_1, x_2)'$, with $x_1 \ne x_2$. (a) Show that the centered covariate is $\mathbf{x}_c = \frac{x_1 - x_2}{2}(1, -1)'$. (b) Using Section 1.3.4, calculate the variance ratio $\mathbf{x}_c'\boldsymbol{\Sigma}\mathbf{x}_c / (\sigma^2 \mathbf{x}_c'\mathbf{x}_c)$ and show that it equals $1 - r$, strictly less than 1 regardless of the values of $x_1$ and $x_2$. (c) Explain this result: why does any non-constant covariate on two positively correlated locations always have a variance ratio less than 1? Connect your answer to the difference eigenvector found in Section 1.3.3.

13. Section 1.4.1 defined the effective sample size for the sample mean. We can similarly define the effective sample size of a slope estimator $\hat{\beta}_1$ as the ratio of its nominal variance under independence to its true variance: $n_{\text{eff}}(\hat{\beta}_1) = \sigma^2(\mathbf{x}_c'\mathbf{x}_c) / (\mathbf{x}_c'\boldsymbol{\Sigma}\mathbf{x}_c)$. Suppose $n$ locations sit along a line at integer positions $1, 2, \dots, n$ with exponential correlation parameter $\phi$, and the covariate is position itself: $x_i = i$. (a) Show that $\mathbf{x}_c'\mathbf{x}_c = n(n^2 - 1)/12$. (b) When the spatial range $\phi$ is very large relative to the spacing, explain why $\mathbf{x}_c$ aligns closely with the smoothest eigenvector of $\boldsymbol{\Sigma}$. (c) Conclude why estimating a spatial trend suffers an even more severe reduction in effective sample size than estimating a spatial mean.

14. Section 1.6.4 warned that invalid covariance functions can produce candidate matrices $\boldsymbol{\Sigma}$ with negative eigenvalues. Suppose a proposed symmetric $n \times n$ matrix $\boldsymbol{\Sigma}$ has a negative eigenvalue $\lambda_k < 0$ with normalized eigenvector $\mathbf{v}_k$ ($\mathbf{v}_k'\mathbf{v}_k = 1$). Define the linear combination $W = \mathbf{v}_k'\mathbf{Y}$. (a) Using Tool 1 (Section 0.4.1), compute $\operatorname{Var}(W)$ in terms of $\lambda_k$. (b) Explain why a negative variance contradicts the axioms of probability. (c) Conclude why positive semi-definiteness is the fundamental requirement for any spatial covariance function, rather than an arbitrary technical condition.

**Computational and simulation**

15. Rerun the coverage experiment of Section 1.7.2 with $\phi = 0.05$, $0.1$, and $0.4$, keeping everything else fixed. Plot coverage against $\phi$ for both covariates, and plot the variance ratio against $\phi$. Explain both trends.

16. In the coverage experiment, replace the naive standard error with the true sandwich standard error, computed from the known $\boldsymbol{\Sigma}$. Show that coverage returns close to 95% for both covariates. Then fit GLS with the known $\boldsymbol{\Sigma}$ and report its coverage and average interval width. Which method gives the shortest valid intervals?

17. Repeat the infill calculation of Section 1.4.3 on the unit square instead of the unit segment, using the grids of Section 0.3.1. Compare the effective sample sizes you get with the table in Chapter 0, and explain why the square's ceiling is larger than the segment's for the same $\phi$.

18. Build the adjacency matrix for a $30 \times 30$ grid of square regions in which squares sharing an edge or a corner count as neighbors. Report the fraction of nonzero entries, and compare the memory used by a dense array with that of `scipy.sparse.csr_matrix(A)`.

---

## What Comes Next

This chapter took the covariance matrix $\boldsymbol{\Sigma}$ as given and showed what goes wrong when it is ignored. Chapter 2 builds $\boldsymbol{\Sigma}$ from the ground up. It develops the model $Y = \mu + u + \varepsilon$ term by term, introduces stationarity and the variogram, explains which correlation functions are valid and why, and shows exactly how the Cholesky factorization turns independent noise into a spatial field. With those tools in hand, Part II estimates $\boldsymbol{\Sigma}$ from data and uses it for prediction and inference.
