# Chapter 4: Spatial Prediction: Kriging as Linear Model BLUP

A landlord is pricing an apartment in a building that has never been listed. A city agency wants the typical rent across a neighborhood to set a housing benchmark. Both questions ask for a value at a place where we have no observation, using observations from nearby places. Answering them is **spatial prediction**, and the standard method has a mining name: **kriging**, after the South African mining engineer Danie Krige, whose work on estimating ore grades was later formalized by Georges Matheron.

This chapter shows that kriging is something you already know under another name. It is **best linear unbiased prediction** (BLUP): the same combination of least squares and unbiasedness that makes OLS the best linear unbiased estimator in a classical regression, and that made GLS best in Section 1.3.5. Every kriging formula in this chapter comes from one recipe. Write the predictor as a linear combination of the data, write its mean squared error as a quadratic form, impose unbiasedness, and minimize. The different flavors of kriging differ only in what is known about the mean.

Throughout the chapter, the covariance parameters $\sigma^2$, $\tau^2$, and $\phi$ are treated as **known**. In the simulations they are set to the true values used to generate the data. Chapters 3 and 5 show how to estimate them, and Section 5.4.5 discusses what plugging in estimates costs. Separating the two problems keeps this chapter about prediction alone.

> **Chapter at a Glance**
>
> - **Kriging Is BLUP (§4.1):** The prediction target, linear predictors, and the mean squared prediction error as variance plus squared bias.
> - **Simple Kriging (§4.2):** With a known mean, the best weights solve $\boldsymbol{\Sigma}\boldsymbol{\lambda} = \mathbf{c}_0$. Worked by hand at the center of the four-apartment block, with a geometric picture.
> - **Ordinary Kriging (§4.3):** With an unknown constant mean, a Lagrange multiplier enforces unbiasedness, and block inversion shows the extra variance is the cost of estimating the mean.
> - **Universal Kriging (§4.4):** With covariates, kriging is GLS for the trend plus simple kriging of the residuals.
> - **Two Targets (§4.5):** Predicting the noise-free value smooths the data; predicting a new observation adds $\tau^2$. Kriging is Gaussian process regression and kernel ridge regression, and skewed data need care when back-transforming.
> - **Block Kriging (§4.6):** Predicting an average over an area, with a smaller variance than any single point.
> - **Cross-Validation (§4.7):** All $n$ leave-one-out residuals from one matrix, the spatial analogue of the regression hat-matrix shortcut.
> - **Walkthrough and Lab (§4.8):** A symbolic derivation, a rent map with standard errors, a check of interval coverage against the truth, and how the kriging standard error depends on the number of listings.

---

## 4.1 Demystifying "Kriging": It Is Best Linear Unbiased Prediction

### 4.1.1 The Prediction Target

Return to the model of Chapter 2,

$$
Y(\mathbf{s}) = \mu(\mathbf{s}) + u(\mathbf{s}) + \varepsilon(\mathbf{s}), \qquad \mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}
$$

with data $\mathbf{Y}$ at locations $\mathbf{s}_1, \dots, \mathbf{s}_n$ and $\operatorname{Var}(\mathbf{Y}) = \boldsymbol{\Sigma} = \sigma^2\mathbf{R}(\phi) + \tau^2\mathbf{I}_n$. We want to predict at a new location $\mathbf{s}_0$, such as the unlisted building.

Before deriving anything, we must say what we are predicting. There are two natural targets.

- The **noise-free value** $\tilde{Y}(\mathbf{s}_0) = \mu(\mathbf{s}_0) + u(\mathbf{s}_0)$: the rent level the location commands, its trend plus its neighborhood effect, without the apartment-specific noise $\varepsilon$.
- A **new observation** $Y(\mathbf{s}_0) = \tilde{Y}(\mathbf{s}_0) + \varepsilon(\mathbf{s}_0)$: the rent of one particular apartment there, noise included.

This chapter predicts $\tilde{Y}(\mathbf{s}_0)$ by default, because it is what a map of rents, or of pollution, is meant to show. Section 4.5 explains how the answer changes for the other target. To keep formulas short, write $\mu_0 = \mu(\mathbf{s}_0)$ and $\mathbf{x}_0 = \mathbf{x}(\mathbf{s}_0)$.

Prediction works through the covariance between the data and the target:

$$
\mathbf{c}_0 = \operatorname{Cov}\!\left(\mathbf{Y}, \tilde{Y}(\mathbf{s}_0)\right), \qquad (\mathbf{c}_0)_i = \operatorname{Cov}\!\left(u(\mathbf{s}_i) + \varepsilon(\mathbf{s}_i),\, u(\mathbf{s}_0)\right) = \sigma^2 r(h_{i0};\phi)
$$

where $h_{i0}$ is the distance from $\mathbf{s}_i$ to $\mathbf{s}_0$. The noise $\varepsilon(\mathbf{s}_i)$ drops out because it is independent of $u(\mathbf{s}_0)$. We also need the variance of the target, $\operatorname{Var}(\tilde{Y}(\mathbf{s}_0)) = \operatorname{Var}(u(\mathbf{s}_0)) = \sigma^2$.

### 4.1.2 Linear Predictors and Their Error

A **linear predictor** is any predictor of the form

$$
\hat{Y}(\mathbf{s}_0) = \lambda_0 + \boldsymbol{\lambda}'\mathbf{Y} = \lambda_0 + \sum_{i=1}^n \lambda_i Y(\mathbf{s}_i)
$$

a constant plus a weighted sum of the data. The weights $\boldsymbol{\lambda}$ say how much each listing contributes. We judge a predictor by its **mean squared prediction error** (MSPE), the average squared distance between prediction and target:

$$
\operatorname{MSPE} = \operatorname{E}\!\left[\left(\hat{Y}(\mathbf{s}_0) - \tilde{Y}(\mathbf{s}_0)\right)^2\right]
$$

Like any mean squared error, it splits into variance plus squared bias. For a random variable $D$, $\operatorname{E}[D^2] = \operatorname{Var}(D) + (\operatorname{E}[D])^2$. Taking $D = \hat{Y}(\mathbf{s}_0) - \tilde{Y}(\mathbf{s}_0)$,

$$
\operatorname{MSPE} = \operatorname{Var}\!\left(\hat{Y}(\mathbf{s}_0) - \tilde{Y}(\mathbf{s}_0)\right) + \left(\operatorname{E}\!\left[\hat{Y}(\mathbf{s}_0) - \tilde{Y}(\mathbf{s}_0)\right]\right)^2
$$

The bias term is $\lambda_0 + \boldsymbol{\lambda}'\operatorname{E}[\mathbf{Y}] - \mu_0$. It depends on the constant and on the mean, but not on the covariance.

### 4.1.3 The Variance Part Is a Quadratic Form

The variance part depends only on the weights. The constant $\lambda_0$ does not affect a variance, so $D$ has the same variance as $\boldsymbol{\lambda}'\mathbf{Y} - \tilde{Y}(\mathbf{s}_0)$. Expanding the variance of a difference, and using Tool 1 for each piece,

$$
\begin{aligned}
\operatorname{Var}\!\left(\boldsymbol{\lambda}'\mathbf{Y} - \tilde{Y}(\mathbf{s}_0)\right) &= \operatorname{Var}(\boldsymbol{\lambda}'\mathbf{Y}) - 2\operatorname{Cov}\!\left(\boldsymbol{\lambda}'\mathbf{Y}, \tilde{Y}(\mathbf{s}_0)\right) + \operatorname{Var}(\tilde{Y}(\mathbf{s}_0)) \\
&= \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2
\end{aligned}
$$

So for any predictor whose bias is zero, that is, whose mean equals the mean of the target,

$$
\operatorname{MSPE}(\boldsymbol{\lambda}) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2
$$

This quadratic form is the whole problem. The three terms have plain meanings. The first is the variance of the weighted sum: weights on listings that repeat each other add up. The second rewards weights on listings that are correlated with the target. The third is the uncertainty we start with, before seeing any data.

The rest of the chapter minimizes this quadratic form under different assumptions about the mean. Each assumption leads to a different unbiasedness requirement, and so to a different version of kriging.

---

## 4.2 Simple Kriging (Known Mean)

### 4.2.1 A Tool We Need: Derivatives With Respect to a Vector

To minimize a function of the vector $\boldsymbol{\lambda}$, we set its derivative with respect to each component to zero and collect the results into a vector, the **gradient**. Two rules cover everything in this chapter.

**Linear functions.** For a fixed vector $\mathbf{c}$, the function $\boldsymbol{\lambda}'\mathbf{c} = \sum_i \lambda_i c_i$ has partial derivative $c_k$ with respect to $\lambda_k$. Collecting them,

$$
\frac{\partial}{\partial\boldsymbol{\lambda}}\,\boldsymbol{\lambda}'\mathbf{c} = \mathbf{c}
$$

**Quadratic forms.** For a fixed symmetric matrix $\boldsymbol{\Sigma}$, write $\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} = \sum_i\sum_j \lambda_i\Sigma_{ij}\lambda_j$. The component $\lambda_k$ appears in the terms with $i = k$ and in the terms with $j = k$, so

$$
\frac{\partial}{\partial\lambda_k}\sum_i\sum_j \lambda_i\Sigma_{ij}\lambda_j = \sum_j \Sigma_{kj}\lambda_j + \sum_i \lambda_i\Sigma_{ik} = 2(\boldsymbol{\Sigma}\boldsymbol{\lambda})_k
$$

using symmetry, $\Sigma_{ik} = \Sigma_{ki}$, in the last step. Collecting,

$$
\frac{\partial}{\partial\boldsymbol{\lambda}}\,\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} = 2\boldsymbol{\Sigma}\boldsymbol{\lambda}
$$

These are the vector versions of $\frac{d}{dx}cx = c$ and $\frac{d}{dx}ax^2 = 2ax$. Appendix A.2 collects them with others.

### 4.2.2 The Simple Kriging Equations

**Simple kriging** assumes the mean $\mu(\mathbf{s})$ is known everywhere: at the data, $\operatorname{E}[\mathbf{Y}] = \boldsymbol{\mu}$, and at the target, $\mu_0$. With a constant mean $\mu$, $\boldsymbol{\mu} = \mu\mathbf{1}_n$ and $\mu_0 = \mu$.

The bias of $\lambda_0 + \boldsymbol{\lambda}'\mathbf{Y}$ is $\lambda_0 + \boldsymbol{\lambda}'\boldsymbol{\mu} - \mu_0$, so the choice

$$
\lambda_0 = \mu_0 - \boldsymbol{\lambda}'\boldsymbol{\mu}
$$

makes the predictor unbiased whatever the weights are. With a constant mean this is $\lambda_0 = \mu(1 - \boldsymbol{\lambda}'\mathbf{1}_n)$. The constant does the bookkeeping for the mean, and the weights are free to minimize the variance.

By Section 4.2.1, the gradient of $\operatorname{MSPE}(\boldsymbol{\lambda}) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2$ is $2\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\mathbf{c}_0$. Setting it to zero gives the **simple kriging equations**:

$$
\boldsymbol{\Sigma}\boldsymbol{\lambda} = \mathbf{c}_0, \qquad \boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}\mathbf{c}_0
$$

This is a minimum, not a maximum or a saddle, because the matrix of second derivatives is $2\boldsymbol{\Sigma}$, which is positive definite. The predictor is

$$
\hat{Y}_{\text{SK}}(\mathbf{s}_0) = \mu_0 + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu})
$$

It starts from the known mean at $\mathbf{s}_0$ and adjusts it by a weighted combination of how far each listing sits above or below its own mean.

Substituting $\boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}\mathbf{c}_0$ into the MSPE gives the **simple kriging variance**:

$$
\sigma^2_{\text{SK}}(\mathbf{s}_0) = \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 - 2\mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 + \sigma^2 = \sigma^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0
$$

We start with uncertainty $\sigma^2$, and the data remove $\mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$ of it. Far from all data, $\mathbf{c}_0 \approx \mathbf{0}$: the predictor falls back to the mean $\mu_0$ and the variance returns to $\sigma^2$.

One feature of this formula has a practical consequence. **The kriging variance does not depend on the observed values $\mathbf{y}$**, only on the locations and the covariance. So we can compute how precise predictions will be before collecting any data, and choose where to sample so that the variance is small where it matters. That is the idea behind spatial sampling design (Appendix D).

### 4.2.3 By Hand: The Center of the Block

Use the four apartments at the corners of a block from Chapters 1 and 2: $\mathbf{s}_1 = (0, 0)$, $\mathbf{s}_2 = (1, 0)$, $\mathbf{s}_3 = (0, 1)$, $\mathbf{s}_4 = (1, 1)$, with $\sigma^2 = 1$, $\tau^2 = 0$, and exponential correlation with $\phi = 1$. Recall from Section 1.3.2 that

$$
\boldsymbol{\Sigma} = \begin{bmatrix} 1 & a & a & b \\ a & 1 & b & a \\ a & b & 1 & a \\ b & a & a & 1 \end{bmatrix}, \qquad a = e^{-1} \approx 0.3679, \qquad b = e^{-\sqrt{2}} \approx 0.2431
$$

and that every row sums to $1 + 2a + b = 1.9789$.

Predict at the center of the block, $\mathbf{s}_0 = (0.5, 0.5)$. Each corner is $\sqrt{0.5} \approx 0.7071$ from the center, so every entry of $\mathbf{c}_0$ is $e^{-0.7071} \approx 0.4931$, that is, $\mathbf{c}_0 = 0.4931\cdot\mathbf{1}$. Because $\mathbf{1}$ is an eigenvector of $\boldsymbol{\Sigma}$ with eigenvalue $1.9789$ (Section 1.3.3), $\boldsymbol{\Sigma}^{-1}\mathbf{1} = \mathbf{1}/1.9789$, and

$$
\boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}\mathbf{c}_0 = \frac{0.4931}{1.9789}\,\mathbf{1} \approx 0.2492\cdot\mathbf{1}
$$

Each corner gets weight 0.2492. The weights sum to 0.9967, slightly less than 1; the remaining $0.0033$ of weight goes to the known mean through $\lambda_0 = \mu(1 - 0.9967)$. The kriging variance is

$$
\sigma^2_{\text{SK}} = 1 - \mathbf{c}_0'\boldsymbol{\lambda} = 1 - 4 \times 0.4931 \times 0.2492 \approx 0.5086
$$

so four corners, each 0.71 block lengths away, remove about half of the initial uncertainty.

**A second location: off the block.** Now predict at $\mathbf{s}_0 = (2, 0.5)$, one block length east of the block. The west corners $\mathbf{s}_1$ and $\mathbf{s}_3$ are $\sqrt{4.25} \approx 2.062$ away, with correlation $0.1273$; the east corners $\mathbf{s}_2$ and $\mathbf{s}_4$ are $\sqrt{1.25} \approx 1.118$ away, with correlation $0.3269$. So $\mathbf{c}_0 = (0.1273, 0.3269, 0.1273, 0.3269)'$.

The eigenvectors of Section 1.3.3 again make the solve possible by hand. Split $\mathbf{c}_0$ into its average and its east-west contrast:

$$
\mathbf{c}_0 = 0.2271\cdot(1, 1, 1, 1)' + 0.0998\cdot(-1, 1, -1, 1)'
$$

where $0.2271$ is the average of the two values and $0.0998$ is half their difference. Both vectors are eigenvectors of $\boldsymbol{\Sigma}$, with eigenvalues $1.9789$ and $1 - b = 0.7569$, so $\boldsymbol{\Sigma}^{-1}$ divides each part by its eigenvalue:

$$
\boldsymbol{\lambda} = \frac{0.2271}{1.9789}(1, 1, 1, 1)' + \frac{0.0998}{0.7569}(-1, 1, -1, 1)' = 0.1148\cdot\mathbf{1} + 0.1319\cdot(-1, 1, -1, 1)'
$$

The east corners get $0.1148 + 0.1319 = 0.2467$, and the west corners get $0.1148 - 0.1319 = -0.0171$. **A negative weight.** The west corners are farther away and lie behind the east corners as seen from $\mathbf{s}_0$. Once the east corners are used, the west corners add little new information about $\mathbf{s}_0$, and the small negative weight corrects for the part of the east corners' values that the west corners share. This is called the **screening effect**: near data screen off far data. The weights sum to only $0.459$, so more than half of the prediction comes from the known mean, and the kriging variance is $0.8431$, close to the starting value of 1.

```python
import numpy as np


def dist(S, T=None):
    """Euclidean distances between rows of S and rows of T (default S).

    This is the distance helper of Chapter 1, extended to two point sets."""
    same = T is None
    T = S if same else T
    q_s = np.sum(S**2, axis=1)
    q_t = np.sum(T**2, axis=1)
    H2 = q_s[:, None] - 2 * S @ T.T + q_t[None, :]
    H = np.sqrt(np.maximum(H2, 0.0))
    if same:
        np.fill_diagonal(H, 0.0)
    return H


S4 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Sigma4 = np.exp(-dist(S4))
s_center = np.array([[0.5, 0.5]])
c0 = np.exp(-dist(S4, s_center))[:, 0]
lam = np.linalg.solve(Sigma4, c0)
print("c0:", np.round(c0, 4))
print("simple kriging weights:", np.round(lam, 4))
print("sum of weights:", round(lam.sum(), 4))
print("SK variance:", round(1 - c0 @ lam, 4))

s_far = np.array([[2.0, 0.5]])
c_far = np.exp(-dist(S4, s_far))[:, 0]
lam_far = np.linalg.solve(Sigma4, c_far)
print("c0 (far):", np.round(c_far, 4))
print("SK weights (far):", np.round(lam_far, 4),
      " sum:", round(lam_far.sum(), 4))
print("SK variance (far):", round(1 - c_far @ lam_far, 4))
```

### 4.2.4 Two Geometric Pictures

The kriging equations have a geometric meaning that ties them to least squares.

**Kriging as a projection.** Think of mean-zero random variables as vectors, with the **covariance inner product** $\langle A, B\rangle = \operatorname{Cov}(A, B)$. Length is standard deviation, and "perpendicular" means uncorrelated. In this picture, the centered data $Y(\mathbf{s}_i) - \mu_i$ span a subspace, and the simple kriging predictor is the point in that subspace closest to the centered target $\tilde{Y}(\mathbf{s}_0) - \mu_0$. The closest point is the orthogonal projection: the error must be perpendicular to every data value,

$$
\operatorname{Cov}\!\left(\tilde{Y}(\mathbf{s}_0) - \boldsymbol{\lambda}'\mathbf{Y},\, Y(\mathbf{s}_i)\right) = 0 \quad \text{for each } i \quad \iff \quad \mathbf{c}_0 - \boldsymbol{\Sigma}\boldsymbol{\lambda} = \mathbf{0}
$$

which is exactly the simple kriging equations. Pythagoras then splits the target's variance into the variance of its projection and the variance of the error, $\sigma^2 = \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 + \sigma^2_{\text{SK}}$, which is the kriging variance formula read the other way.

**GLS fitted values as a projection.** Section 1.3.5 found $\hat{\boldsymbol{\beta}}_{\text{GLS}}$. Its normal equations, $\mathbf{X}'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}}) = \mathbf{0}$, say that the residual vector is perpendicular to every column of $\mathbf{X}$ in the inner product $\langle\mathbf{a}, \mathbf{b}\rangle = \mathbf{a}'\boldsymbol{\Sigma}^{-1}\mathbf{b}$ between fixed vectors. So the GLS fitted values $\mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}}$ are the orthogonal projection of $\mathbf{Y}$ onto the columns of $\mathbf{X}$ in that inner product, just as OLS fitted values are in the ordinary one.

The two pictures live in different spaces, one of random variables and one of data vectors, but both say the same thing: the best linear answer leaves an error perpendicular to everything it could have used. That is why the normal equations of regression and the kriging equations look alike.

---

## 4.3 Ordinary Kriging (Unknown Constant Mean)

### 4.3.1 Unbiasedness as a Constraint

Simple kriging needs the mean. In practice it is unknown. **Ordinary kriging** assumes the mean is an unknown constant $\mu$ and uses no constant term, $\lambda_0 = 0$. Then

$$
\operatorname{E}[\boldsymbol{\lambda}'\mathbf{Y}] - \operatorname{E}[\tilde{Y}(\mathbf{s}_0)] = \mu\,\boldsymbol{\lambda}'\mathbf{1}_n - \mu = \mu(\boldsymbol{\lambda}'\mathbf{1}_n - 1)
$$

For this to be zero whatever the value of $\mu$, the weights must sum to one:

$$
\boldsymbol{\lambda}'\mathbf{1}_n = 1
$$

We now minimize $\operatorname{MSPE}(\boldsymbol{\lambda})$ subject to this constraint. The standard tool is a **Lagrange multiplier**. Introduce a new unknown $m$ and form the Lagrangian

$$
\mathcal{L}(\boldsymbol{\lambda}, m) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2 + 2m\left(\boldsymbol{\lambda}'\mathbf{1}_n - 1\right)
$$

(The factor 2 on $m$ only makes the equations tidier.) At the constrained minimum, the derivatives with respect to both $\boldsymbol{\lambda}$ and $m$ are zero. By Section 4.2.1,

$$
\frac{\partial\mathcal{L}}{\partial\boldsymbol{\lambda}} = 2\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\mathbf{c}_0 + 2m\mathbf{1}_n = \mathbf{0}, \qquad \frac{\partial\mathcal{L}}{\partial m} = 2(\boldsymbol{\lambda}'\mathbf{1}_n - 1) = 0
$$

The second equation is the constraint itself. The first says $\boldsymbol{\Sigma}\boldsymbol{\lambda} + m\mathbf{1}_n = \mathbf{c}_0$.

### 4.3.2 The Block-Matrix System

The two equations stack into one linear system of size $n + 1$:

$$
\begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ m \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ 1 \end{bmatrix}
$$

We call the matrix on the left the **kriging matrix** and write it $\mathbf{K}$. Numerically, ordinary kriging is a single call to a linear solver with $\mathbf{K}$. To see what the solution means, we solve it by hand.

**Solving by elimination.** From the first block row, $\boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}(\mathbf{c}_0 - m\mathbf{1}_n)$. Substituting into the constraint,

$$
\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 - m\,\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n = 1 \quad\Longrightarrow\quad m = \frac{\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 - 1}{\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n}
$$

and therefore

$$
\boldsymbol{\lambda}_{\text{OK}} = \boldsymbol{\Sigma}^{-1}\mathbf{c}_0 + \boldsymbol{\Sigma}^{-1}\mathbf{1}_n\,\frac{1 - \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0}{\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n}
$$

The first term is the simple kriging weight vector. The second term tops up the weights so that they sum to one. The amount of top-up depends on $1 - \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$, which is one minus the sum of the simple kriging weights: the share of the prediction that simple kriging would have taken from the known mean.

**The predictor is GLS plus simple kriging.** Write $\hat{\mu}_{\text{GLS}} = \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{Y}/\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n$ for the GLS estimate of the constant mean (Section 1.4.4, which writes it with $\mathbf{R}^{-1}$; the scale $\sigma^2$ cancels). Then

$$
\boldsymbol{\lambda}_{\text{OK}}'\mathbf{Y} = \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{Y} + \left(1 - \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0\right)\hat{\mu}_{\text{GLS}} = \hat{\mu}_{\text{GLS}} + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\left(\mathbf{Y} - \hat{\mu}_{\text{GLS}}\mathbf{1}_n\right)
$$

Ordinary kriging is simple kriging with the unknown mean replaced by its best linear unbiased estimate. Section 4.4 shows this holds for any regression mean.

### 4.3.3 What the Lagrange Multiplier Costs

The ordinary kriging variance follows from the system without much algebra. Multiply the first block row, $\boldsymbol{\Sigma}\boldsymbol{\lambda} = \mathbf{c}_0 - m\mathbf{1}_n$, on the left by $\boldsymbol{\lambda}'$ and use the constraint $\boldsymbol{\lambda}'\mathbf{1}_n = 1$:

$$
\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} = \boldsymbol{\lambda}'\mathbf{c}_0 - m
$$

Substituting into the MSPE,

$$
\sigma^2_{\text{OK}} = (\boldsymbol{\lambda}'\mathbf{c}_0 - m) - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2 = \sigma^2 - \boldsymbol{\lambda}'\mathbf{c}_0 - m
$$

To compare with simple kriging, insert the solution. Write $a_0 = \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$ (the sum of the simple kriging weights) and $q_0 = \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n$. From the formulas above,

$$
\boldsymbol{\lambda}'\mathbf{c}_0 = \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 + \frac{(1 - a_0)\,a_0}{q_0}, \qquad -m = \frac{1 - a_0}{q_0}
$$

so

$$
\sigma^2_{\text{OK}} = \sigma^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 - \frac{(1 - a_0)a_0}{q_0} + \frac{1 - a_0}{q_0} = \sigma^2_{\text{SK}} + \frac{(1 - a_0)^2}{q_0}
$$

The extra term has a clean interpretation. By Tool 1, the GLS mean has variance $\operatorname{Var}(\hat{\mu}_{\text{GLS}}) = 1/\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n = 1/q_0$. So

$$
\sigma^2_{\text{OK}} = \sigma^2_{\text{SK}} + (1 - a_0)^2\operatorname{Var}(\hat{\mu}_{\text{GLS}})
$$

**The extra variance is the cost of estimating the mean**, weighted by how much the prediction relies on the mean. Near the data, the simple kriging weights sum almost to one, $1 - a_0 \approx 0$, and not knowing the mean costs almost nothing. Far from the data, the prediction leans on the mean, and its uncertainty passes straight through.

**By hand, off the block.** At $\mathbf{s}_0 = (2, 0.5)$, Section 4.2.3 found simple kriging weights summing to $a_0 = 0.459$. Since $\boldsymbol{\Sigma}^{-1}\mathbf{1} = \mathbf{1}/1.9789$, $q_0 = 4/1.9789 = 2.0213$. Each weight is topped up by $(1/1.9789)(1 - 0.459)/2.0213 \approx 0.1352$, giving ordinary kriging weights of $0.1181$ for the west corners and $0.3819$ for the east corners, summing to one. The Lagrange multiplier is $m = (0.459 - 1)/2.0213 \approx -0.2676$, and the extra variance is $(1 - 0.459)^2/2.0213 \approx 0.1448$, raising the variance from $0.8431$ to $0.9879$. The negative weights have disappeared: with an unknown mean, every corner is needed to estimate it.

```python
one = np.ones(4)
K_ok = np.block([[Sigma4, one[:, None]], [one[None, :], np.zeros((1, 1))]])
sol = np.linalg.solve(K_ok, np.append(c_far, 1.0))
lam_ok, m = sol[:4], sol[4]
a = one @ np.linalg.solve(Sigma4, c_far)
q = one @ np.linalg.solve(Sigma4, one)
print("OK weights (far):", np.round(lam_ok, 4),
      " sum:", round(lam_ok.sum(), 4))
print("Lagrange multiplier m:", round(m, 4))
var_sk = 1 - c_far @ lam_far
var_ok = 1 - lam_ok @ c_far - m
print("OK variance:", round(var_ok, 4),
      " = SK", round(var_sk, 4), "+ extra", round((1 - a)**2 / q, 4))
```

### 4.3.4 A Tool We Need: Inverting a Block Matrix

The elimination above can be packaged as a general formula that we will use twice more in this chapter, for universal kriging and for cross-validation. For a square matrix split into blocks, with $\mathbf{M}_{11}$ invertible,

$$
\begin{bmatrix} \mathbf{M}_{11} & \mathbf{M}_{12} \\ \mathbf{M}_{21} & \mathbf{M}_{22} \end{bmatrix}
$$

the matrix $\mathbf{F} = \mathbf{M}_{22} - \mathbf{M}_{21}\mathbf{M}_{11}^{-1}\mathbf{M}_{12}$ is called the **Schur complement** of $\mathbf{M}_{11}$. It is what remains of the lower-right block after eliminating the first block of unknowns, exactly as we eliminated $\boldsymbol{\lambda}$ above. If $\mathbf{F}$ is invertible,

$$
\begin{bmatrix} \mathbf{M}_{11} & \mathbf{M}_{12} \\ \mathbf{M}_{21} & \mathbf{M}_{22} \end{bmatrix}^{-1} = \begin{bmatrix} \mathbf{M}_{11}^{-1} + \mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} & -\mathbf{M}_{11}^{-1}\mathbf{M}_{12}\mathbf{F}^{-1} \\ -\mathbf{F}^{-1}\mathbf{M}_{21}\mathbf{M}_{11}^{-1} & \mathbf{F}^{-1} \end{bmatrix}
$$

Exchanging the roles of the two blocks gives the same formula with $\mathbf{M}_{11}$ and $\mathbf{M}_{22}$ interchanged, so when $\mathbf{M}_{22}$ is invertible the upper-left block of the inverse is also $(\mathbf{M}_{11} - \mathbf{M}_{12}\mathbf{M}_{22}^{-1}\mathbf{M}_{21})^{-1}$. You can verify the formula above by multiplying the two matrices together and checking that each block of the product is an identity or a zero; Appendix A.1 does this in full. (Here $\mathbf{M}_{11}, \dots, \mathbf{M}_{22}$ are generic blocks.)

For the ordinary kriging matrix, $\mathbf{M}_{11} = \boldsymbol{\Sigma}$, $\mathbf{M}_{12} = \mathbf{1}_n$, $\mathbf{M}_{21} = \mathbf{1}_n'$, $\mathbf{M}_{22} = 0$, so the Schur complement is the scalar $F = -\mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n = -q_0$. The bottom-right entry of $\mathbf{K}^{-1}$ is $-1/q_0 = -\operatorname{Var}(\hat{\mu}_{\text{GLS}})$: the variance of the estimated mean sits inside the kriging matrix. Multiplying $\mathbf{K}^{-1}$ by $(\mathbf{c}_0', 1)'$ reproduces the formulas for $\boldsymbol{\lambda}_{\text{OK}}$ and $m$ found by elimination (Exercise 3).

---

## 4.4 Universal Kriging (Spatial Regression With Covariates)

### 4.4.1 Unbiasedness for Every $\boldsymbol{\beta}$

Rents rise toward the east, so a constant mean is wrong for the rent district. **Universal kriging** allows the regression mean $\mu(\mathbf{s}) = \mathbf{x}(\mathbf{s})'\boldsymbol{\beta}$ with $\boldsymbol{\beta}$ unknown. With $\lambda_0 = 0$, the predictor $\boldsymbol{\lambda}'\mathbf{Y}$ has mean $\boldsymbol{\lambda}'\mathbf{X}\boldsymbol{\beta}$, and the target has mean $\mathbf{x}_0'\boldsymbol{\beta}$. For these to agree for every $\boldsymbol{\beta}$,

$$
\mathbf{X}'\boldsymbol{\lambda} = \mathbf{x}_0
$$

These are $p$ constraints, one per covariate. For the rent district, $\mathbf{x}(\mathbf{s}) = (1, s_{\text{east}})'$, and the constraints say the weights sum to one and the weighted average east coordinate of the listings equals the east coordinate of $\mathbf{s}_0$. The weights must reproduce the trend exactly at the target.

### 4.4.2 The Universal Kriging System

With one Lagrange multiplier per constraint, collected into $\mathbf{m}$,

$$
\mathcal{L}(\boldsymbol{\lambda}, \mathbf{m}) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2 + 2\mathbf{m}'(\mathbf{X}'\boldsymbol{\lambda} - \mathbf{x}_0)
$$

The term $\mathbf{m}'\mathbf{X}'\boldsymbol{\lambda} = \boldsymbol{\lambda}'(\mathbf{X}\mathbf{m})$ is linear in $\boldsymbol{\lambda}$, so its gradient is $\mathbf{X}\mathbf{m}$. Setting both gradients to zero gives $\boldsymbol{\Sigma}\boldsymbol{\lambda} + \mathbf{X}\mathbf{m} = \mathbf{c}_0$ and $\mathbf{X}'\boldsymbol{\lambda} = \mathbf{x}_0$, or

$$
\mathbf{K}\begin{bmatrix} \boldsymbol{\lambda} \\ \mathbf{m} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_0 \\ \mathbf{x}_0 \end{bmatrix}, \qquad \mathbf{K} = \begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{X} \\ \mathbf{X}' & \mathbf{0} \end{bmatrix}
$$

Ordinary kriging is the special case $\mathbf{X} = \mathbf{1}_n$.

### 4.4.3 The Two-Stage Decomposition

Eliminate as before. From the first block row, $\boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}(\mathbf{c}_0 - \mathbf{X}\mathbf{m})$. Substituting into the constraints,

$$
\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 - \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\mathbf{m} = \mathbf{x}_0 \quad\Longrightarrow\quad \mathbf{m} = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0 - \mathbf{x}_0)
$$

The predictor is

$$
\begin{aligned}
\boldsymbol{\lambda}'\mathbf{Y} &= \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{Y} - \mathbf{m}'\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y} \\
&= \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{Y} - (\mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{X} - \mathbf{x}_0')(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y} \\
&= \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{Y} - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{x}_0'\hat{\boldsymbol{\beta}}_{\text{GLS}}
\end{aligned}
$$

using $\hat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y}$ from Section 1.3.5. Rearranging,

$$
\hat{Y}_{\text{UK}}(\mathbf{s}_0) = \mathbf{x}_0'\hat{\boldsymbol{\beta}}_{\text{GLS}} + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})
$$

This is the central result of the chapter. **Universal kriging is a two-stage procedure:** estimate the trend by GLS, then predict the residual at $\mathbf{s}_0$ by simple kriging, treating the GLS residuals as if their mean were known to be zero. Spatial prediction is linear regression with correlated errors, followed by a correction that borrows the residuals of nearby listings.

### 4.4.4 The Universal Kriging Variance

The same trick as in Section 4.3.3 gives the variance. Multiplying the first block row by $\boldsymbol{\lambda}'$ and using $\boldsymbol{\lambda}'\mathbf{X} = \mathbf{x}_0'$, $\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} = \boldsymbol{\lambda}'\mathbf{c}_0 - \mathbf{x}_0'\mathbf{m}$, so

$$
\sigma^2_{\text{UK}} = \sigma^2 - \boldsymbol{\lambda}'\mathbf{c}_0 - \mathbf{x}_0'\mathbf{m}
$$

Inserting $\boldsymbol{\lambda}$ and $\mathbf{m}$, with $\mathbf{d}_0 = \mathbf{x}_0 - \mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$, the same steps as for ordinary kriging give (Exercise 4)

$$
\sigma^2_{\text{UK}} = \underbrace{\sigma^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0}_{\sigma^2_{\text{SK}}} + \mathbf{d}_0'(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{d}_0
$$

The first part is the simple kriging variance: the uncertainty about the neighborhood effect that remains after seeing the data. The second part is the **cost of estimating the trend**. Since $(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1} = \operatorname{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}})$, it equals $\operatorname{Var}(\mathbf{d}_0'\hat{\boldsymbol{\beta}}_{\text{GLS}})$ by Tool 1. The vector $\mathbf{d}_0$ measures how far the simple kriging weights are from reproducing the trend at $\mathbf{s}_0$ on their own: where nearby listings already pin down the trend, $\mathbf{d}_0 \approx \mathbf{0}$, and the cost is negligible.

**The unlisted building.** The code below applies universal kriging to the rent district of Chapters 3 to 5: 250 simulated listings, rents in hundreds of dollars, with the true values $\sigma^2 = 4$, $\tau^2 = 1$, $\phi = 0.15$ (exponential), and a trend rising to the east. It predicts the noise-free rent at the unlisted building at $(0.5, 0.5)$, two ways: by solving the kriging system, and by the two-stage formula.

```python
def rent_district(n=250, seed=2030):
    """Simulated rent listings: locations, trend, effect, noise, rent."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(size=(n, 2))
    mu = 18.0 + 6.0 * S[:, 0]
    R = np.exp(-dist(S) / 0.15)
    L = np.linalg.cholesky(4.0 * R + 1e-10 * np.eye(n))
    u = L @ rng.standard_normal(n)
    eps = rng.standard_normal(n)
    return S, mu, u, eps, mu + u + eps


S, mu, u, eps, y = rent_district()
n = len(y)
sigma2, tau2, phi = 4.0, 1.0, 0.15
Sigma = sigma2 * np.exp(-dist(S) / phi) + tau2 * np.eye(n)
X = np.column_stack([np.ones(n), S[:, 0]])
p = X.shape[1]


def uk_predict(s0, x0):
    """Universal kriging of the noise-free value at s0 (known Sigma)."""
    c0 = sigma2 * np.exp(-dist(S, s0[None, :]) / phi)[:, 0]
    K = np.block([[Sigma, X], [X.T, np.zeros((p, p))]])
    sol = np.linalg.solve(K, np.concatenate([c0, x0]))
    lam, m = sol[:n], sol[n:]
    return lam @ y, sigma2 - lam @ c0 - m @ x0, lam, c0


s0 = np.array([0.5, 0.5])
x0 = np.array([1.0, 0.5])
pred, var, lam, c0 = uk_predict(s0, x0)
print(f"UK prediction at (0.5, 0.5): {pred:.3f}, SE {np.sqrt(var):.3f}")
print("weights sum to", round(lam.sum(), 6),
      " weighted east coordinate:", round(lam @ S[:, 0], 6))

Si_X = np.linalg.solve(Sigma, X)
beta_gls = np.linalg.solve(X.T @ Si_X, Si_X.T @ y)
resid = y - X @ beta_gls
two_stage = x0 @ beta_gls + c0 @ np.linalg.solve(Sigma, resid)
print("GLS coefficients:", np.round(beta_gls, 3))
print(f"two-stage prediction: {two_stage:.3f}")

print(" location      prediction  UK var   SK var   trend cost")
for s_new in ([0.5, 0.5], [0.95, 0.95], [1.3, 0.5]):
    s_new = np.array(s_new)
    x_new = np.array([1.0, s_new[0]])
    pr, va, la, c = uk_predict(s_new, x_new)
    v_sk = sigma2 - c @ np.linalg.solve(Sigma, c)
    d = x_new - Si_X.T @ c
    v_beta = d @ np.linalg.solve(X.T @ Si_X, d)
    print(f" {str(s_new):12s} {pr:8.3f}  {va:7.4f}  {v_sk:7.4f}  {v_beta:.5f}")
nearest = np.argsort(dist(S, s0[None, :])[:, 0])[:5]
print("five nearest listings to (0.5, 0.5): distance and weight")
for i in nearest:
    print(f"  {dist(S[i:i+1], s0[None, :])[0, 0]:.3f}  {lam[i]:.3f}")
print("number of negative weights:", int((lam < 0).sum()),
      " their total:", round(lam[lam < 0].sum(), 3))
```

The two routes agree: the predicted noise-free rent at $(0.5, 0.5)$ is 20.438, about \$2,044 a month, with standard error 0.946 (about \$95). The weights sum to one and their weighted east coordinate is 0.5, as the constraints require. The GLS trend is $16.853 + 7.201\,s_{\text{east}}$, close to the true $18 + 6\,s_{\text{east}}$.

The table of three locations shows how the trend cost behaves.

| Location | Prediction | UK variance | SK variance | Trend cost |
| :--- | :--- | :--- | :--- | :--- |
| (0.5, 0.5), center of the district | 20.438 | 0.8952 | 0.8952 | 0.00003 |
| (0.95, 0.95), near a corner | 23.460 | 1.7234 | 1.7136 | 0.00975 |
| (1.3, 0.5), outside the district | 25.994 | 5.1711 | 3.9215 | 1.24959 |

In the middle of the district, estimating the trend costs essentially nothing. Outside the district, where the prediction is an extrapolation of the trend, the trend's uncertainty is a quarter of the total variance.

The weights also show the screening effect at scale. The nearest listing, 0.015 away, gets weight 0.375. The fourth nearest, at distance 0.050, gets more weight (0.160) than the third nearest at 0.045 (0.092), because the third nearest sits in the same direction as closer listings and is partly screened by them. Of the 250 weights, 103 are negative, but they are small: together they total $-0.033$.

---

## 4.5 Two Targets, and Interpolation Versus Smoothing

### 4.5.1 Predicting the Noise-Free Value at a Data Location

What does kriging predict at a location where we already have a listing? Take $\mathbf{s}_0 = \mathbf{s}_i$. The covariance vector is then the $i$th column of $\sigma^2\mathbf{R}$, which is the $i$th column of $\boldsymbol{\Sigma}$ minus the nugget: $\mathbf{c}_0 = \boldsymbol{\Sigma}\mathbf{e}_i - \tau^2\mathbf{e}_i$, where $\mathbf{e}_i$ is the vector with 1 in position $i$ and zeros elsewhere. (This $\mathbf{e}_i$ is a standard basis vector, used only here; it is not the residual vector of Chapter 1.) The simple kriging weights are

$$
\boldsymbol{\lambda} = \boldsymbol{\Sigma}^{-1}\mathbf{c}_0 = \mathbf{e}_i - \tau^2\boldsymbol{\Sigma}^{-1}\mathbf{e}_i
$$

and the predictor is

$$
\hat{Y}(\mathbf{s}_i) = \mu_i + (Y_i - \mu_i) - \tau^2\left[\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu})\right]_i = Y_i - \tau^2\left[\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu})\right]_i
$$

Two cases follow.

- **No nugget, $\tau^2 = 0$:** $\hat{Y}(\mathbf{s}_i) = Y_i$ exactly, and the kriging variance is zero there. Kriging **interpolates**: the predicted surface passes through every data value.
- **A nugget, $\tau^2 > 0$:** the prediction pulls $Y_i$ toward what its neighbors suggest. Kriging **smooths**, because part of each observed value is noise that the neighbors do not share, and the noise-free target should not include it.

The same formula holds for universal kriging, with $\mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}}$ in place of $\boldsymbol{\mu}$. For the rent district we know the true noise-free rents, so we can check that smoothing helps.

```python
Si_c_all = np.linalg.solve(Sigma, sigma2 * np.exp(-dist(S) / phi))
y_tilde_hat = X @ beta_gls + Si_c_all.T @ resid
truth = mu + u
print(f"RMSE of raw rents vs truth:     "
      f"{np.sqrt(np.mean((y - truth)**2)):.3f}")
print(f"RMSE of kriged values vs truth: "
      f"{np.sqrt(np.mean((y_tilde_hat - truth)**2)):.3f}")
alt = np.linalg.solve(Sigma, y - X @ beta_gls)
print("shortcut y - tau2 * Sigma^{-1} resid matches:",
      np.allclose(y_tilde_hat, y - tau2 * alt))
```

The raw listed rents miss the true noise-free rents by a root mean squared error of 1.087, which is the nugget's standard deviation of 1, plus sampling variation. The kriged values miss by only 0.807. By borrowing from neighbors, kriging removes part of the noise in each listing.

Figure 4.1 shows the difference on a small one-dimensional example: eight noisy observations of a smooth curve, kriged with simple kriging (known mean zero, $\sigma^2 = 1$, exponential correlation with $\phi = 0.2$), once with no nugget and once with a nugget of 0.25.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch04", exist_ok=True)
t_obs = np.array([0.05, 0.18, 0.3, 0.42, 0.55, 0.7, 0.82, 0.95])
rng = np.random.default_rng(4)
y_obs = np.sin(2 * np.pi * t_obs) + 0.5 * rng.standard_normal(len(t_obs))
t_new = np.linspace(0, 1, 300)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, nug in zip(axes, (0.0, 0.25)):
    Sig_1d = np.exp(-np.abs(t_obs[:, None] - t_obs[None, :]) / 0.2)
    Sig_1d = Sig_1d + nug * np.eye(len(t_obs))
    C_new = np.exp(-np.abs(t_obs[:, None] - t_new[None, :]) / 0.2)
    W = np.linalg.solve(Sig_1d, C_new)
    pred_1d = W.T @ y_obs
    se_1d = np.sqrt(np.maximum(1 - np.sum(C_new * W, axis=0), 0))
    ax.fill_between(t_new, pred_1d - 2 * se_1d, pred_1d + 2 * se_1d,
                    color="0.85", label="plus or minus 2 SE")
    ax.plot(t_new, pred_1d, label="kriged noise-free value")
    ax.plot(t_obs, y_obs, "o", color="black", label="data")
    ax.set_title(f"nugget = {nug}")
    ax.set_xlabel("position")
axes[0].set_ylabel("value")
axes[0].legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig("figures/ch04/fig_4_1_interp_smooth.png", dpi=120)
```

![Figure 4.1. Kriging eight observations with no nugget (left) and a nugget of 0.25 (right). Without a nugget the prediction passes through every point and the uncertainty band pinches to zero there. With a nugget the prediction smooths the points and the band stays open.](../figures/ch04/fig_4_1_interp_smooth.png)

In the left panel, the kriged curve passes through every observation and the shaded band of two standard errors pinches to zero width at each one. In the right panel, the curve passes between the points, pulled away from isolated high and low values such as the peak near 0.3 and the low points near 0.55 and 0.82, and the band never closes, because the noise-free value is uncertain even where it has been measured.

### 4.5.2 Predicting a New Observation

For the other target, a new observation $Y(\mathbf{s}_0) = \tilde{Y}(\mathbf{s}_0) + \varepsilon(\mathbf{s}_0)$ at a location that is not one of the data locations, nothing changes in the weights: the extra $\varepsilon(\mathbf{s}_0)$ is independent of the data, so $\operatorname{Cov}(\mathbf{Y}, Y(\mathbf{s}_0)) = \mathbf{c}_0$ still. Its variance adds to the target's: $\operatorname{Var}(Y(\mathbf{s}_0)) = \sigma^2 + \tau^2$. Repeating Section 4.1.3 with this variance,

$$
\operatorname{MSPE}_{\text{new}} = \sigma^2_{\text{UK}} + \tau^2
$$

The prediction is the same; the interval is wider, by the nugget. A landlord who asks "what will this particular apartment rent for?" needs the wider interval. An agency mapping the rent level of a neighborhood needs the narrower one. Software output often does not say which it reports, which is one reason to understand the derivation.

### 4.5.3 Kriging Is Gaussian Process Regression and Kernel Ridge Regression (Optional)

*This subsection connects kriging to Gaussian process regression and kernel ridge regression in machine learning. It can be skipped on a first reading.*

Kriging appears under other names in machine learning. Two connections are worth making precise.

**Kriging as a conditional mean.** If $\mathbf{Y}$ and $\tilde{Y}(\mathbf{s}_0)$ are jointly normal, the conditional distribution of the target given the data is normal, with (Appendix A.4)

$$
\operatorname{E}\!\left[\tilde{Y}(\mathbf{s}_0) \mid \mathbf{Y}\right] = \mu_0 + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu}), \qquad \operatorname{Var}\!\left(\tilde{Y}(\mathbf{s}_0) \mid \mathbf{Y}\right) = \sigma^2 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0
$$

These are the simple kriging predictor and variance. Under normality, then, simple kriging is not only the best *linear* predictor: it is the conditional mean, which is the best predictor of any form. In machine learning this is **Gaussian process regression**. A Gaussian process prior on the unknown function, with covariance function $\sigma^2 r(h;\phi)$ and observation noise $\tau^2$, gives exactly this posterior mean and variance.

**Kriging as kernel ridge regression.** Take a known mean of zero and write $\mathbf{C}_u = \sigma^2\mathbf{R}$, so $\boldsymbol{\Sigma} = \mathbf{C}_u + \tau^2\mathbf{I}_n$. Kernel ridge regression looks for a function of the form $f(\mathbf{s}) = \sum_j \alpha_j\,\sigma^2 r(\lVert\mathbf{s} - \mathbf{s}_j\rVert;\phi)$, one bump per data location, and chooses the coefficients $\boldsymbol{\alpha}$ to minimize a penalized sum of squares,

$$
\lVert\mathbf{y} - \mathbf{C}_u\boldsymbol{\alpha}\rVert^2 + \tau^2\,\boldsymbol{\alpha}'\mathbf{C}_u\boldsymbol{\alpha}
$$

The first term rewards fitting the data; the second penalizes rough functions, with $\tau^2$ as the penalty weight. Setting the gradient to zero (Section 4.2.1),

$$
-2\mathbf{C}_u(\mathbf{y} - \mathbf{C}_u\boldsymbol{\alpha}) + 2\tau^2\mathbf{C}_u\boldsymbol{\alpha} = \mathbf{0} \quad\Longrightarrow\quad \mathbf{C}_u\left[(\mathbf{C}_u + \tau^2\mathbf{I}_n)\boldsymbol{\alpha} - \mathbf{y}\right] = \mathbf{0}
$$

so $\boldsymbol{\alpha} = (\mathbf{C}_u + \tau^2\mathbf{I}_n)^{-1}\mathbf{y} = \boldsymbol{\Sigma}^{-1}\mathbf{y}$, and the fitted function at $\mathbf{s}_0$ is $f(\mathbf{s}_0) = \mathbf{c}_0'\boldsymbol{\alpha} = \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{y}$: the simple kriging predictor. The nugget plays the role of the ridge penalty. With no nugget there is no penalty, and the fit interpolates, as in Section 4.5.1. (Here $\mathbf{C}_u$ and $\boldsymbol{\alpha}$ are local to this subsection. That the best function has this one-bump-per-point form is a known result, the representer theorem, which we take from the literature.)

Three communities, geostatistics, Bayesian statistics, and machine learning, arrived at the same formula from three directions: best linear unbiased prediction, a conditional mean, and a penalized fit.

### 4.5.4 Skewed Responses: Kriging on the Log Scale

Rents, incomes, and pollutant concentrations are often right-skewed, and their logarithms are closer to normal. A common approach is to krige $\log Y$ and transform back. The back-transformation needs care.

Suppose $\log Y(\mathbf{s})$ follows the spatial model, with noise-free value $\tilde{Y}(\mathbf{s}_0)$ on the log scale, and we want the noise-free rent $\exp(\tilde{Y}(\mathbf{s}_0))$. With known mean, Section 4.5.3 says that, given the data, $\tilde{Y}(\mathbf{s}_0)$ is normal with mean $\hat{Y}_{\text{SK}}(\mathbf{s}_0)$ and variance $\sigma^2_{\text{SK}}(\mathbf{s}_0)$. For a normal variable $V$ with mean $a$ and variance $v$, $\operatorname{E}[e^V] = e^{a + v/2}$ (the moment generating function of the normal, evaluated at 1). So

$$
\operatorname{E}\!\left[\exp(\tilde{Y}(\mathbf{s}_0)) \mid \mathbf{Y}\right] = \exp\!\left(\hat{Y}_{\text{SK}}(\mathbf{s}_0) + \tfrac{1}{2}\sigma^2_{\text{SK}}(\mathbf{s}_0)\right)
$$

The naive back-transformation $\exp(\hat{Y}_{\text{SK}})$ estimates the conditional median, not the mean, and is biased low. The correction factor $e^{\sigma^2_{\text{SK}}/2}$ is largest where the prediction is least certain. (With an estimated mean, as in ordinary or universal kriging, the exact correction has an extra term; using the kriging variance in its place is a common approximation.)

The code checks this on a simulated log-rent field, with log-scale mean 3 (a typical rent of $e^3 \approx 20$, about \$2,000), $\sigma^2 = 0.09$, $\tau^2 = 0.01$, and $\phi = 0.15$. It predicts the noise-free rent on a $20 \times 20$ grid of unlisted locations. Checking against the truth requires knowing what the true surface was at those unlisted grid locations. Simulating the listings first with `rent_district()` and then simulating the grid in a separate call would produce two independent realizations, destroying the spatial correlation between the data and the targets. To preserve the covariance $\mathbf{c}_0$, the random effect $\mathbf{u}$ must be drawn jointly from the combined $(n + m)$-dimensional distribution. The function `joint_field` does that through a single Cholesky factorization of the combined covariance matrix, as in Section 2.7. The rent district generator used elsewhere in this chapter simulates only at the listings, which is why a second generator is needed here.

```python
def joint_field(S_obs, S_new, sigma2, phi, seed):
    """Spatial effect simulated jointly at observed and new locations."""
    rng = np.random.default_rng(seed)
    S_all = np.vstack([S_obs, S_new])
    R_all = np.exp(-dist(S_all) / phi)
    L_all = np.linalg.cholesky(sigma2 * R_all + 1e-10 * np.eye(len(S_all)))
    u_all = L_all @ rng.standard_normal(len(S_all))
    return u_all[:len(S_obs)], u_all[len(S_obs):]


g = (np.arange(20) + 0.5) / 20
G1, G2 = np.meshgrid(g, g)
S_grid = np.column_stack([G1.ravel(), G2.ravel()])
s2_log, t2_log, phi_log, mu_log = 0.09, 0.01, 0.15, 3.0
Sig_log = s2_log * np.exp(-dist(S) / phi_log) + t2_log * np.eye(n)
C_log = s2_log * np.exp(-dist(S, S_grid) / phi_log)
W_log = np.linalg.solve(Sig_log, C_log)
var_log = s2_log - np.sum(C_log * W_log, axis=0)
err_naive, err_fix = [], []
for rep in range(100):
    u_o, u_g = joint_field(S, S_grid, s2_log, phi_log, seed=500 + rep)
    rng = np.random.default_rng(900 + rep)
    z = mu_log + u_o + np.sqrt(t2_log) * rng.standard_normal(n)
    z_hat = mu_log + W_log.T @ (z - mu_log)
    target = np.exp(mu_log + u_g)
    err_naive.append(np.mean(np.exp(z_hat) - target))
    err_fix.append(np.mean(np.exp(z_hat + var_log / 2) - target))
print(f"average error, naive back-transform:     {np.mean(err_naive):.4f}")
print(f"average error, corrected back-transform: {np.mean(err_fix):.4f}")
print(f"median kriging variance on log scale:    {np.median(var_log):.4f}")
```

Over 100 simulated fields, the naive back-transformation underpredicts the noise-free rent by 0.2669 on average, about \$27 a month. The corrected version is off by only 0.0122, which is within simulation error of zero. The typical kriging variance on the log scale is 0.0240, so the correction factor is about $e^{0.012} \approx 1.012$: small for one apartment, but a systematic \$27 error across a whole map.

---

## 4.6 Change of Support: Block Kriging

### 4.6.1 Predicting an Average Over an Area

A housing agency setting a benchmark rent for a neighborhood does not want the rent at one point. It wants the average over the neighborhood. In general, for a region $B$ we want

$$
\tilde{Y}(B) = \frac{1}{\operatorname{area}(B)}\int_B \tilde{Y}(\mathbf{s})\,d\mathbf{s}
$$

The same need arises for the average noise level over a park, or the average pollution over a census tract. Predicting a quantity at a different spatial scale, or **support**, from the data is called a **change of support**.

The block average is a linear function of the same process, so everything in this chapter applies with three substitutions. Approximate the integral by an average over $n_B$ points $\mathbf{s}^B_1, \dots, \mathbf{s}^B_{n_B}$ on a fine grid inside $B$. Then:

- **The covariance vector** becomes the average of the point covariance vectors:

  $$
  \mathbf{c}_B = \operatorname{Cov}\!\left(\mathbf{Y}, \tilde{Y}(B)\right) \approx \frac{1}{n_B}\sum_{k=1}^{n_B}\mathbf{c}_0(\mathbf{s}^B_k)
  $$

  where $\mathbf{c}_0(\mathbf{s}^B_k)$ is the covariance vector for the point $\mathbf{s}^B_k$.
- **The target's variance** becomes the average covariance over pairs of points in the block, $\sigma^2_B = \frac{1}{n_B^2}\sum_k\sum_l \sigma^2 r(\lVert\mathbf{s}^B_k - \mathbf{s}^B_l\rVert;\phi)$, which is smaller than $\sigma^2$.
- **The covariates** become their block averages, $\mathbf{x}_B = \frac{1}{n_B}\sum_k \mathbf{x}(\mathbf{s}^B_k)$.

### 4.6.2 The Only Change Is $\mathbf{c}_B$

The universal kriging system is unchanged except for its right-hand side, $(\mathbf{c}_B', \mathbf{x}_B')'$, and the variance is $\sigma^2_B - \boldsymbol{\lambda}'\mathbf{c}_B - \mathbf{x}_B'\mathbf{m}$. Because the solution is linear in the right-hand side, and $\mathbf{c}_B$ and $\mathbf{x}_B$ are averages, the block prediction is exactly the average of the point predictions over the grid. The variance is not the average of the point variances, as the next subsection explains.

```python
def uk_block(lo, hi, m_pts=10):
    """Universal kriging of the average noise-free rent over a square."""
    gb = lo + (np.arange(m_pts) + 0.5) / m_pts * (hi - lo)
    B1, B2 = np.meshgrid(gb, gb)
    S_B = np.column_stack([B1.ravel(), B2.ravel()])
    c_B = sigma2 * np.exp(-dist(S, S_B) / phi).mean(axis=1)
    var_B = sigma2 * np.exp(-dist(S_B) / phi).mean()
    x_B = np.array([1.0, S_B[:, 0].mean()])
    K = np.block([[Sigma, X], [X.T, np.zeros((p, p))]])
    sol = np.linalg.solve(K, np.concatenate([c_B, x_B]))
    lam_B, m_B = sol[:n], sol[n:]
    point_preds = [uk_predict(s, np.array([1.0, s[0]]))[0] for s in S_B]
    return (lam_B @ y, var_B - lam_B @ c_B - m_B @ x_B, var_B,
            np.mean(point_preds))


for lo, hi in ((0.4, 0.6), (0.2, 0.8)):
    pr_B, v_B, s2_B, avg_pt = uk_block(lo, hi)
    print(f"block [{lo}, {hi}]^2: prediction {pr_B:.3f}, "
          f"average of point predictions {avg_pt:.3f}")
    print(f"   variance of the true block average {s2_B:.4f}, "
          f"kriging variance {v_B:.4f}")
```

For the neighborhood $[0.4, 0.6]^2$ in the middle of the district, the predicted average noise-free rent is 20.321, and it equals the average of the 100 point predictions, as linearity requires. The kriging variance of the block average is 0.1396 (standard error 0.374), against 0.8952 (standard error 0.946) for the single point at the block's center in Section 4.4. For the larger neighborhood $[0.2, 0.8]^2$ the kriging variance drops to 0.0199. The variance of the true block average itself also shrinks with size, from $\sigma^2 = 4$ for a point to 2.1120 and 0.7813 for the two blocks, because averaging over an area smooths out the neighborhood effect.

### 4.6.3 Why Block Predictions Are More Precise

The block prediction error is the average of the point prediction errors $D_k = \hat{Y}(\mathbf{s}^B_k) - \tilde{Y}(\mathbf{s}^B_k)$ over the grid, since both the prediction and the target are averages. Write $v_k = \operatorname{Var}(D_k)$ for the point kriging variances. By Tool 1 and the Cauchy-Schwarz inequality $\operatorname{Cov}(D_k, D_l) \le \sqrt{v_k v_l}$,

$$
\operatorname{Var}\!\left(\frac{1}{n_B}\sum_k D_k\right) = \frac{1}{n_B^2}\sum_k\sum_l \operatorname{Cov}(D_k, D_l) \le \left(\frac{1}{n_B}\sum_k\sqrt{v_k}\right)^2 \le \frac{1}{n_B}\sum_k v_k
$$

where the last step says the square of an average is at most the average of the squares. **The block kriging variance is never larger than the average point kriging variance**, and it is much smaller when the point errors are not strongly correlated: errors at different points in the block partly cancel. This matters in practice. A map of point predictions with wide intervals can still support precise statements about averages over areas.

---

## 4.7 Evaluating Predictions: Leave-One-Out Cross-Validation

### 4.7.1 The Idea, and the Cost of Doing It Directly

How good are the predictions? With real data there is no truth to compare with, but each listing can be predicted from the others. **Leave-one-out cross-validation** (LOOCV) removes listing $i$, predicts its observed rent $Y_i$ from the remaining $n - 1$ listings, and records the error $Y_i - \hat{Y}_{-i}$. Here the target is the observation $Y_i$ itself, including its noise, so the prediction uses the full covariance $\boldsymbol{\Sigma}$ between $Y_i$ and the other listings, and the trend is re-estimated from the $n - 1$ remaining listings each time.

Done directly, this means solving $n$ kriging systems of size $n - 1 + p$. For a few thousand listings that is slow. In ordinary regression there is a famous shortcut: the leave-one-out residual is $e_i/(1 - h_{ii})$, computable from a single fit, where $h_{ii} = [\mathbf{P}_{\mathbf{X}}]_{ii}$ is the leverage. Kriging has the same kind of shortcut.

### 4.7.2 The Shortcut

Define the **residual precision matrix**

$$
\boldsymbol{\Pi} = \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}
$$

Since $\boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{Y} = \boldsymbol{\Sigma}^{-1}\mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}}$, applying $\boldsymbol{\Pi}$ to the data gives the GLS residuals weighted by $\boldsymbol{\Sigma}^{-1}$:

$$
\boldsymbol{\Pi}\mathbf{Y} = \boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{GLS}})
$$

Then, for every $i$ at once,

$$
Y_i - \hat{Y}_{-i} = \frac{(\boldsymbol{\Pi}\mathbf{Y})_i}{\Pi_{ii}}, \qquad \operatorname{Var}\!\left(Y_i - \hat{Y}_{-i}\right) = \frac{1}{\Pi_{ii}}
$$

All $n$ leave-one-out residuals, and their variances, come from one matrix computed from the full data. The proof is in Section 4.7.4.

**The connection to regression.** With independent errors, $\boldsymbol{\Sigma} = \tau^2\mathbf{I}_n$, the matrix becomes $\boldsymbol{\Pi} = (\mathbf{I}_n - \mathbf{P}_{\mathbf{X}})/\tau^2$, so $\boldsymbol{\Pi}\mathbf{Y} = \mathbf{e}/\tau^2$ with $\mathbf{e}$ the OLS residuals, and $\Pi_{ii} = (1 - h_{ii})/\tau^2$. The shortcut becomes $e_i/(1 - h_{ii})$: the regression formula is the special case of the kriging formula with no spatial correlation.

With a known mean and no covariates, $\boldsymbol{\Pi}$ reduces to $\boldsymbol{\Sigma}^{-1}$, and the formula says the leave-one-out residual of $Y_i$ is $[\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu})]_i/(\boldsymbol{\Sigma}^{-1})_{ii}$. Chapter 8 meets this formula again, as the conditional mean of one value given all the others.

### 4.7.3 Using the Residuals

The standardized residuals $\text{LOO}_i\sqrt{\Pi_{ii}}$, where $\text{LOO}_i = (\boldsymbol{\Pi}\mathbf{Y})_i/\Pi_{ii}$, should look like standard normal draws if the model and its covariance parameters are right. Their mean square should be near 1, and about 5% should exceed 1.96 in absolute value. Large departures suggest the covariance model is wrong, and Section 5.6.1 uses this as a diagnostic.

```python
Si = np.linalg.inv(Sigma)
Pi = Si - Si_X @ np.linalg.solve(X.T @ Si_X, Si_X.T)
loo_shortcut = (Pi @ y) / np.diag(Pi)

loo_brute = np.empty(n)
for i in range(n):
    keep = np.arange(n) != i
    Sig_k, X_k = Sigma[np.ix_(keep, keep)], X[keep]
    K = np.block([[Sig_k, X_k], [X_k.T, np.zeros((p, p))]])
    sol = np.linalg.solve(K, np.concatenate([Sigma[keep, i], X[i]]))
    loo_brute[i] = y[i] - sol[:n - 1] @ y[keep]
print("largest difference, shortcut vs refitting:",
      f"{np.max(np.abs(loo_shortcut - loo_brute)):.1e}")
z_loo = (Pi @ y) / np.sqrt(np.diag(Pi))
print(f"mean squared standardized residual: {np.mean(z_loo**2):.3f}")
print(f"share with |z| > 1.96: {np.mean(np.abs(z_loo) > 1.96):.3f}")
print(f"LOOCV root mean squared error: "
      f"{np.sqrt(np.mean(loo_shortcut**2)):.3f}")
H_ols = X @ np.linalg.solve(X.T @ X, X.T)
e_ols = y - H_ols @ y
Pi_iid = np.eye(n) - H_ols
print("with Sigma = I the shortcut is e/(1 - h):",
      np.allclose((Pi_iid @ y) / np.diag(Pi_iid),
                  e_ols / (1 - np.diag(H_ols))))
print(f"OLS LOOCV root mean squared error: "
      f"{np.sqrt(np.mean((e_ols / (1 - np.diag(H_ols)))**2)):.3f}")
```

For the rent district, the shortcut agrees with brute-force refitting of all 250 kriging systems to within $5.2 \times 10^{-14}$, rounding error. The mean squared standardized residual is 1.109 and 7.2% of the standardized residuals exceed 1.96 in absolute value, both reasonably close to the values of 1 and 5% expected when the model is right. The leave-one-out root mean squared error of kriging is 1.587, against 2.228 for leave-one-out OLS with the same trend and no spatial correlation. Accounting for the neighborhood effect cuts prediction error for a held-out listing by about 29%.

### 4.7.4 Proof of the Shortcut (Optional)

*This subsection proves the leave-one-out formula with the block inversion of Section 4.3.4. It can be skipped on a first reading.*

**Step 1: $\boldsymbol{\Pi}$ is a block of $\mathbf{K}^{-1}$.** Apply the block inversion formula to the universal kriging matrix $\mathbf{K}$, with $\mathbf{M}_{11} = \boldsymbol{\Sigma}$, $\mathbf{M}_{12} = \mathbf{X}$, $\mathbf{M}_{21} = \mathbf{X}'$, $\mathbf{M}_{22} = \mathbf{0}$. The Schur complement is $\mathbf{F} = -\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}$, and the top-left $n \times n$ block of $\mathbf{K}^{-1}$ is

$$
\boldsymbol{\Sigma}^{-1} + \boldsymbol{\Sigma}^{-1}\mathbf{X}\left(-\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}'\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Pi}
$$

**Step 2: reorder so that listing $i$ comes first.** Reordering the rows and columns of $\mathbf{K}$ the same way does not change the argument, so assume $i = 1$ and partition

$$
\mathbf{K} = \begin{bmatrix} \Sigma_{11} & \mathbf{b}' \\ \mathbf{b} & \mathbf{K}_{-1} \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} \boldsymbol{\Sigma}_{-1,1} \\ \mathbf{x}_1 \end{bmatrix}
$$

where $\mathbf{K}_{-1}$ is the universal kriging matrix built from the other $n - 1$ listings, $\boldsymbol{\Sigma}_{-1,1}$ holds the covariances between $Y_1$ and the other listings, and $\mathbf{x}_1$ is the covariate vector of listing 1.

**Step 3: read off the leave-one-out prediction.** Predicting $Y_1$ from the other listings by universal kriging means solving a system with matrix $\mathbf{K}_{-1}$ and right-hand side $\mathbf{b}$: the covariance vector and the covariates at the target, exactly as in Section 4.4.2. The first $n - 1$ entries of the solution $\mathbf{K}_{-1}^{-1}\mathbf{b}$ are the weights, so with $\mathbf{y}^{\circ} = (\mathbf{Y}_{-1}', \mathbf{0}_p')'$, the other listings' values padded with $p$ zeros,

$$
\hat{Y}_{-1} = \mathbf{b}'\mathbf{K}_{-1}^{-1}\mathbf{y}^{\circ}
$$

**Step 4: apply block inversion again.** Now treat $\Sigma_{11}$ as the lower-right block. Its Schur complement is the scalar $f = \Sigma_{11} - \mathbf{b}'\mathbf{K}_{-1}^{-1}\mathbf{b}$, and the first row of $\mathbf{K}^{-1}$ is $\left(1/f,\; -\mathbf{b}'\mathbf{K}_{-1}^{-1}/f\right)$. Multiply that row by the full vector $(Y_1, \mathbf{y}^{\circ\prime})'$:

$$
\left[\mathbf{K}^{-1}\begin{pmatrix} Y_1 \\ \mathbf{y}^{\circ} \end{pmatrix}\right]_1 = \frac{Y_1 - \mathbf{b}'\mathbf{K}_{-1}^{-1}\mathbf{y}^{\circ}}{f} = \frac{Y_1 - \hat{Y}_{-1}}{f}
$$

**Step 5: combine.** The vector $(Y_1, \mathbf{y}^{\circ\prime})'$ is $(\mathbf{Y}', \mathbf{0}_p')'$, so by Step 1 the left side is $(\boldsymbol{\Pi}\mathbf{Y})_1$, and the $(1, 1)$ entry of $\mathbf{K}^{-1}$, which is $1/f$, is $\Pi_{11}$. Therefore $Y_1 - \hat{Y}_{-1} = (\boldsymbol{\Pi}\mathbf{Y})_1/\Pi_{11}$.

For the variance, $f = \Sigma_{11} - \mathbf{b}'\mathbf{K}_{-1}^{-1}\mathbf{b} = \Sigma_{11} - \boldsymbol{\lambda}'\boldsymbol{\Sigma}_{-1,1} - \mathbf{x}_1'\mathbf{m}$, which is the universal kriging variance of Section 4.4.4 for predicting the observation $Y_1$ (target variance $\Sigma_{11}$). So $\operatorname{Var}(Y_1 - \hat{Y}_{-1}) = f = 1/\Pi_{11}$. $\square$

---

## 4.8 Walkthrough and Lab 4

### 4.8.1 The Ordinary Kriging Equations in SymPy

For two listings with correlation $r$, unit variances, and covariances $c_1$, $c_2$ with the target, SymPy can set up the Lagrangian of Section 4.3.1, differentiate, and solve.

```python
import sympy as sp

l1, l2, m_s, r_s, c1, c2 = sp.symbols("lambda1 lambda2 m r c1 c2")
lam_s = sp.Matrix([l1, l2])
Sig_s = sp.Matrix([[1, r_s], [r_s, 1]])
c_s = sp.Matrix([c1, c2])
lagr = ((lam_s.T * Sig_s * lam_s)[0] - 2 * (lam_s.T * c_s)[0] + 1
        + 2 * m_s * (l1 + l2 - 1))
eqs = [sp.diff(lagr, v) for v in (l1, l2, m_s)]
sol_s = sp.solve(eqs, (l1, l2, m_s), dict=True)[0]
print("lambda1 =", sp.simplify(sol_s[l1]))
print("m =", sp.simplify(sol_s[m_s]))
```

The output is $\lambda_1 = (-c_1 + c_2 + r - 1)/(2(r - 1))$, which rearranges to $\lambda_1 = \tfrac{1}{2} + (c_1 - c_2)/(2(1 - r))$, and $m = (c_1 + c_2 - r - 1)/2$. The weights start at one half each and shift toward the listing more correlated with the target, by an amount that grows as the two listings become more alike ($r$ near 1). Exercise 2 asks you to check that this matches the general formula of Section 4.3.2.

### 4.8.2 The Four-Apartment Block

The code blocks in Sections 4.2.3 and 4.3.3 reproduce every hand calculation: the weights 0.2492 at the center, $-0.0171$ and $0.2467$ off the block, and the ordinary kriging weights 0.1181 and 0.3819 with $m = -0.2676$ and the extra variance 0.1448.

### 4.8.3 Do the Intervals Cover the Truth?

The kriging variance is only useful if it is honest. The code below simulates 200 new versions of the rent district at the same listing locations, each time with a fresh neighborhood effect and fresh noise, together with the true noise-free rent on the $20 \times 20$ grid of Section 4.5.4. For each version it computes universal kriging predictions at the grid, with the known covariance parameters, and records how often the 95% interval $\hat{Y} \pm 1.96\,\sigma_{\text{UK}}$ covers the truth. It also compares kriging with the fitted trend alone, which is what an OLS analysis would map.

```python
X_grid = np.column_stack([np.ones(len(S_grid)), S_grid[:, 0]])
C_grid = sigma2 * np.exp(-dist(S, S_grid) / phi)
K_full = np.block([[Sigma, X], [X.T, np.zeros((p, p))]])
sol_g = np.linalg.solve(K_full, np.vstack([C_grid, X_grid.T]))
Lam_g, M_g = sol_g[:n], sol_g[n:]
var_g = sigma2 - np.sum(Lam_g * C_grid, axis=0) - np.sum(M_g * X_grid.T,
                                                         axis=0)
cover, sq_uk, sq_trend = 0.0, 0.0, 0.0
L_noise = np.sqrt(tau2)
for rep in range(200):
    u_o, u_g = joint_field(S, S_grid, sigma2, phi, seed=1000 + rep)
    rng = np.random.default_rng(3000 + rep)
    y_rep = X @ np.array([18.0, 6.0]) + u_o + L_noise * rng.standard_normal(n)
    truth_g = X_grid @ np.array([18.0, 6.0]) + u_g
    pred_g = Lam_g.T @ y_rep
    b_ols = np.linalg.lstsq(X, y_rep, rcond=None)[0]
    cover += np.mean(np.abs(pred_g - truth_g) <= 1.96 * np.sqrt(var_g))
    sq_uk += np.mean((pred_g - truth_g)**2)
    sq_trend += np.mean((X_grid @ b_ols - truth_g)**2)
print(f"coverage of 95% intervals: {cover / 200:.3f}")
print(f"RMSE, universal kriging:   {np.sqrt(sq_uk / 200):.3f}")
print(f"RMSE, fitted trend only:   {np.sqrt(sq_trend / 200):.3f}")
print(f"average kriging SE:        {np.mean(np.sqrt(var_g)):.3f}")
```

The intervals cover the true noise-free rent 95.0% of the time, as they should. Kriging has a root mean squared error of 1.115 against 1.847 for the fitted trend alone, a 40% reduction, and the average kriging standard error of 1.098 matches the realized error of 1.115 closely.

### 4.8.4 A Rent Map

Finally, the map. The code computes universal kriging predictions and standard errors on a $40 \times 40$ grid over the rent district.

```python
g40 = (np.arange(40) + 0.5) / 40
A1, A2 = np.meshgrid(g40, g40)
S_map = np.column_stack([A1.ravel(), A2.ravel()])
X_map = np.column_stack([np.ones(len(S_map)), S_map[:, 0]])
C_map = sigma2 * np.exp(-dist(S, S_map) / phi)
sol_m = np.linalg.solve(K_full, np.vstack([C_map, X_map.T]))
pred_map = sol_m[:n].T @ y
se_map = np.sqrt(sigma2 - np.sum(sol_m[:n] * C_map, axis=0)
                 - np.sum(sol_m[n:] * X_map.T, axis=0))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(pred_map.reshape(40, 40), origin="lower",
                     extent=(0, 1, 0, 1), cmap="viridis")
axes[0].set_title("Kriged rent (hundreds of dollars)")
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(se_map.reshape(40, 40), origin="lower",
                     extent=(0, 1, 0, 1), cmap="magma")
axes[1].plot(S[:, 0], S[:, 1], ".", color="white", markersize=3)
axes[1].set_title("Kriging standard error, with listings")
fig.colorbar(im1, ax=axes[1])
for ax in axes:
    ax.set_xlabel("east")
    ax.set_ylabel("north")
fig.tight_layout()
fig.savefig("figures/ch04/fig_4_2_rent_map.png", dpi=120)
print(f"standard error range on the map: {se_map.min():.3f} "
      f"to {se_map.max():.3f}")
print("figures saved")
```

![Figure 4.2. Universal kriging of the simulated rent district. Left: predicted noise-free rent. Right: the kriging standard error, with listing locations as white dots. Uncertainty is lowest at listings and highest in gaps and along the edges.](../figures/ch04/fig_4_2_rent_map.png)

The left panel of Figure 4.2 shows the trend rising to the east, broken up by the neighborhood effect: a pocket of low rents in the southwest and high rents toward the southeast corner. The right panel shows where those predictions can be trusted. The standard error ranges from 0.688 to 1.669. It is smallest at and between dense clusters of listings and largest in gaps, along the edges, and especially in the corners and along the top and bottom borders, where the nearest listings are farthest away and on one side only. Because the kriging variance does not depend on the observed rents (Section 4.2.2), this right panel could have been drawn before any rents were collected.

### 4.8.5 How the Kriging Standard Error Depends on the Number of Listings

Chapter 0 and Section 1.4.4 said that packing more listings into the same district barely improves a region-wide quantity such as the average rent, but helps a great deal for prediction at a particular location. This experiment shows both. The kriging variance depends only on where the listings are (Section 4.2.2), so no rents need to be simulated. For each number of listings $n$ the code draws 20 sets of $n$ locations uniformly in the district, computes the universal kriging standard error of the noise-free rent at the district's center, $(0.5, 0.5)$, with the formula of Section 4.4.4 (the simple kriging variance plus the cost of estimating the trend), and computes, for the same locations, the effective sample size of the sample mean from Section 1.4.1, $n_{\text{eff}} = C(0)/\operatorname{Var}(\bar{Y})$ with $C(0) = \sigma^2 + \tau^2$.

```python
rng = np.random.default_rng(2031)
s0 = np.array([[0.5, 0.5]])
x0 = np.array([1.0, 0.5])
print("    n   mean SE   (min, max)        mean n_eff of the sample mean")
for n_i in (25, 100, 250, 1000):
    se_list, neff_list = [], []
    for _ in range(20):
        S_i = rng.uniform(size=(n_i, 2))
        Sig = sigma2 * np.exp(-dist(S_i) / phi) + tau2 * np.eye(n_i)
        X_i = np.column_stack([np.ones(n_i), S_i[:, 0]])
        c0 = sigma2 * np.exp(-dist(S_i, s0)[:, 0] / phi)
        Sig_c = np.linalg.solve(Sig, c0)
        Sig_X = np.linalg.solve(Sig, X_i)
        d0 = x0 - X_i.T @ Sig_c
        var_uk = (sigma2 - c0 @ Sig_c
                  + d0 @ np.linalg.solve(X_i.T @ Sig_X, d0))
        se_list.append(np.sqrt(var_uk))
        one = np.ones(n_i)
        neff_list.append((sigma2 + tau2) / (one @ Sig @ one / n_i**2))
    print(f"{n_i:5d}   {np.mean(se_list):.3f}   "
          f"({min(se_list):.3f}, {max(se_list):.3f})   "
          f"{np.mean(neff_list):.2f}")
```

| $n$ | Mean standard error at $(0.5, 0.5)$ | Range over the 20 location sets | Mean $n_{\text{eff}}$ of the sample mean |
| :--- | :--- | :--- | :--- |
| 25 | 1.568 | 0.927 to 1.841 | 9.08 |
| 100 | 1.309 | 1.083 to 1.546 | 11.95 |
| 250 | 1.092 | 0.936 to 1.351 | 12.74 |
| 1,000 | 0.821 | 0.655 to 0.961 | 13.17 |

The two columns tell different stories. The standard error of the prediction falls from 1.57 to 0.82 as the number of listings grows 40-fold, and it would keep falling with more listings, because the noise-free rent at a point can be learned more and more precisely from the listings around it. The effective sample size of the sample mean goes from 9 to only 13: the average rent of the district is limited by how many independent fluctuations the district contains (Section 1.4.3), not by how many listings there are. The range column also shows how much the arrangement matters at small $n$: a lucky arrangement of 25 listings, with one close to the center, beats an unlucky arrangement of 100 (0.927 against 1.546). That is the point of spatial sampling design, which Appendix D mentions: where the listings are can matter as much as how many.

---

## Exercises

**Analytical**

1. Show that for any linear predictor $\lambda_0 + \boldsymbol{\lambda}'\mathbf{Y}$, the MSPE for the target $\tilde{Y}(\mathbf{s}_0)$ equals $\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2 + (\lambda_0 + \boldsymbol{\lambda}'\boldsymbol{\mu} - \mu_0)^2$. Use it to explain why simple kriging can always remove the bias term, while ordinary kriging cannot without a constraint.

2. Using the formula for $\boldsymbol{\lambda}_{\text{OK}}$ in Section 4.3.2 with $n = 2$, $\boldsymbol{\Sigma} = \begin{bmatrix} 1 & r \\ r & 1 \end{bmatrix}$, and $\mathbf{c}_0 = (c_1, c_2)'$, derive $\lambda_1$ and $m$ by hand and confirm the SymPy output of Section 4.8.1.

3. Use the block inversion formula of Section 4.3.4 to write $\mathbf{K}^{-1}$ for the ordinary kriging matrix in terms of $\boldsymbol{\Sigma}^{-1}$ and $q_0 = \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n$. Multiply it by $(\mathbf{c}_0', 1)'$ and show that you recover the formulas for $\boldsymbol{\lambda}_{\text{OK}}$ and $m$.

4. Complete the derivation of the universal kriging variance in Section 4.4.4: starting from $\sigma^2_{\text{UK}} = \sigma^2 - \boldsymbol{\lambda}'\mathbf{c}_0 - \mathbf{x}_0'\mathbf{m}$, substitute the solutions for $\boldsymbol{\lambda}$ and $\mathbf{m}$ and show that it equals $\sigma^2_{\text{SK}} + \mathbf{d}_0'(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{d}_0$. Check that with $\mathbf{X} = \mathbf{1}_n$ it reduces to the ordinary kriging result.

5. For the four-apartment block with $\tau^2 = 0$, predict at the midpoint of the south side, $\mathbf{s}_0 = (0.5, 0)$, by simple kriging. Use the eigenvectors of Section 1.3.3: split $\mathbf{c}_0$ into parts along $(1, 1, 1, 1)'$ and $(-1, -1, 1, 1)'$ (south against north), and divide each part by its eigenvalue. Are any weights negative? Explain.

6. Show that when $\tau^2 = 0$ and $\mathbf{s}_0 = \mathbf{s}_i$, the universal kriging variance is zero. Then show that with $\tau^2 > 0$ it is strictly positive at every data location.

7. In Section 4.5.3, verify that minimizing $\lVert\mathbf{y} - \mathbf{C}_u\boldsymbol{\alpha}\rVert^2 + \tau^2\boldsymbol{\alpha}'\mathbf{C}_u\boldsymbol{\alpha}$ gives $\boldsymbol{\alpha} = (\mathbf{C}_u + \tau^2\mathbf{I}_n)^{-1}\mathbf{y}$ when $\mathbf{C}_u$ is invertible. What changes if $\mathbf{C}_u$ is singular?

8. Show that if $V \sim N(a, v)$, then $\operatorname{E}[e^V] = e^{a + v/2}$, by completing the square in the integral. Use the result to explain why $\exp(\hat{Y}_{\text{SK}})$ is the conditional median of $\exp(\tilde{Y}(\mathbf{s}_0))$.

9. Show that $\boldsymbol{\Pi}\mathbf{X} = \mathbf{0}$ and $\boldsymbol{\Pi}\boldsymbol{\Sigma}\boldsymbol{\Pi} = \boldsymbol{\Pi}$. Use the first to explain why the leave-one-out residuals do not depend on the true value of $\boldsymbol{\beta}$.

10. In the block-kriging argument of Section 4.6.3, when is the inequality an equality? Describe a covariance model and a block for which block kriging gives almost no improvement over point kriging.

11. Formulate ordinary kriging directly through the semivariogram $\gamma(\mathbf{h})$ rather than the covariance $C(\mathbf{h})$. Assume intrinsic stationarity, with $\gamma(\mathbf{s}_i - \mathbf{s}_j) = C(\mathbf{0}) - C(\mathbf{s}_i - \mathbf{s}_j)$. Define the $n \times n$ semivariogram matrix $\boldsymbol{\Gamma}$ with entries $\Gamma_{ij} = \gamma(\mathbf{s}_i - \mathbf{s}_j)$ and the target semivariogram vector $\boldsymbol{\gamma}_0$ with entries $(\boldsymbol{\gamma}_0)_i = \gamma(\mathbf{s}_i - \mathbf{s}_0)$.
    (a) For any weight vector satisfying $\boldsymbol{\lambda}'\mathbf{1}_n = 1$, show that $\boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} = C(\mathbf{0}) - \boldsymbol{\lambda}'\boldsymbol{\Gamma}\boldsymbol{\lambda}$ and $\boldsymbol{\lambda}'\mathbf{c}_0 = C(\mathbf{0}) - \boldsymbol{\lambda}'\boldsymbol{\gamma}_0$.
    (b) Show that minimizing the prediction error variance subject to $\boldsymbol{\lambda}'\mathbf{1}_n = 1$ leads to the linear system
    $$
    \begin{bmatrix} \boldsymbol{\Gamma} & \mathbf{1}_n \\ \mathbf{1}_n' & 0 \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda} \\ m_\gamma \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}
    $$
    with $m_\gamma = -m$, giving the identical weights $\boldsymbol{\lambda}_{\text{OK}}$ as the covariance system.
    (c) Show that the ordinary kriging variance equals $\sigma^2_{\text{OK}} = \boldsymbol{\lambda}'\boldsymbol{\gamma}_0 + m_\gamma$. Use this to explain why ordinary kriging requires only intrinsic stationarity, whereas simple kriging requires second-order stationarity.

12. In a one-dimensional spatial process on the line with exponential covariance $C(h) = \sigma^2 \exp(-\lvert h\rvert / \phi)$ and no nugget ($\tau^2 = 0$), the process is a stationary Markov process (the Ornstein-Uhlenbeck process). Suppose observations are located at $s_1 < s_2 < \dots < s_n$. We want to predict $\tilde{Y}(s_0)$ at an interior point $s_0 \in (s_k, s_{k+1})$ by simple kriging with mean zero.
    (a) Using the tridiagonal structure of $\boldsymbol{\Sigma}^{-1}$ from Section 1.3.2, show that the simple kriging weights $\lambda_j$ are zero for all $j \notin \{k, k+1\}$: the screening effect is exact.
    (b) Solve the reduced $2 \times 2$ system for the weights $\lambda_k$ and $\lambda_{k+1}$ on the two enclosing neighbors in terms of $d_1 = s_0 - s_k$, $d_2 = s_{k+1} - s_0$, and $\Delta = s_{k+1} - s_k = d_1 + d_2$.
    (c) Show that both weights are positive. Show that as the correlation range $\phi \to \infty$, the weights converge to linear interpolation: $\lambda_k \to d_2 / \Delta$ and $\lambda_{k+1} \to d_1 / \Delta$.

13. Derive the simple kriging predictor as a conditional expectation under joint normality. Suppose $\mathbf{Y}$ and $\tilde{Y}_0 = \tilde{Y}(\mathbf{s}_0)$ are jointly Gaussian:
    $$
    \begin{bmatrix} \mathbf{Y} \\ \tilde{Y}_0 \end{bmatrix} \sim N\!\left(\begin{bmatrix} \boldsymbol{\mu} \\ \mu_0 \end{bmatrix},\; \begin{bmatrix} \boldsymbol{\Sigma} & \mathbf{c}_0 \\ \mathbf{c}_0' & \sigma^2 \end{bmatrix}\right)
    $$
    Define the random variable $Z = \tilde{Y}_0 - \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{Y}$.
    (a) Compute $\operatorname{Cov}(Z, \mathbf{Y})$ and show that it equals $\mathbf{0}'$.
    (b) Because linear combinations of Gaussian vectors are Gaussian, zero covariance implies that $Z$ and $\mathbf{Y}$ are independent. Use this independence to evaluate $\operatorname{E}[\tilde{Y}_0 \mid \mathbf{Y}]$ and $\operatorname{Var}(\tilde{Y}_0 \mid \mathbf{Y})$ by writing $\tilde{Y}_0 = Z + \mathbf{c}_0'\boldsymbol{\Sigma}^{-1}\mathbf{Y}$.
    (c) Confirm that the conditional mean and conditional variance match the simple kriging formulas $\hat{Y}_{\text{SK}}(\mathbf{s}_0)$ and $\sigma^2_{\text{SK}}(\mathbf{s}_0)$. Explain why this implies that under normality, no nonlinear predictor can achieve a lower mean squared prediction error than simple kriging.

14. Suppose simple kriging weights $\boldsymbol{\lambda}_n = \boldsymbol{\Sigma}_n^{-1}\mathbf{c}_0$ have been computed for $n$ listings. A new listing arrives at location $\mathbf{s}_{n+1}$, expanding the covariance matrix to $\boldsymbol{\Sigma}_{n+1} = \begin{bmatrix} \boldsymbol{\Sigma}_n & \mathbf{k} \\ \mathbf{k}' & \Sigma_{n+1, n+1} \end{bmatrix}$ and the target covariance to $\mathbf{c}_{0, n+1} = (\mathbf{c}_0', c_{0, n+1})'$.
    (a) Use the block inversion formula of Section 4.3.4 to express $\boldsymbol{\Sigma}_{n+1}^{-1}$ in terms of $\boldsymbol{\Sigma}_n^{-1}$, $\mathbf{k}$, and the scalar Schur complement $s = \Sigma_{n+1, n+1} - \mathbf{k}'\boldsymbol{\Sigma}_n^{-1}\mathbf{k}$.
    (b) Show that the new weight on listing $n+1$ is $\lambda_{n+1} = (c_{0, n+1} - \mathbf{k}'\boldsymbol{\lambda}_n) / s$, and express the updated weights $\boldsymbol{\lambda}_{1:n}$ on the original $n$ listings in terms of $\boldsymbol{\lambda}_n$ and $\lambda_{n+1}$.
    (c) Show that the updated simple kriging variance satisfies $\sigma^2_{\text{SK}, n+1} = \sigma^2_{\text{SK}, n} - s\,\lambda_{n+1}^2$. Conclude that adding an observation can never increase the simple kriging variance.

15. In ordinary kriging, consider relaxing the unbiasedness constraint to $\boldsymbol{\lambda}'\mathbf{1}_n = c$ for an arbitrary scalar $c$.
    (a) Minimize $\operatorname{MSPE}(\boldsymbol{\lambda}) = \boldsymbol{\lambda}'\boldsymbol{\Sigma}\boldsymbol{\lambda} - 2\boldsymbol{\lambda}'\mathbf{c}_0 + \sigma^2$ subject to $\boldsymbol{\lambda}'\mathbf{1}_n = c$ using the Lagrangian $\mathcal{L}_c(\boldsymbol{\lambda}, m) = \operatorname{MSPE}(\boldsymbol{\lambda}) + 2m(\boldsymbol{\lambda}'\mathbf{1}_n - c)$. Show that the constrained minimum variance as a function of $c$, $V(c)$, satisfies $V(c) = \sigma^2_{\text{SK}} + (c - a_0)^2 / q_0$, where $a_0 = \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{c}_0$ and $q_0 = \mathbf{1}_n'\boldsymbol{\Sigma}^{-1}\mathbf{1}_n$.
    (b) Differentiate $V(c)$ with respect to $c$ and evaluate at $c = 1$. Show that $V'(1) = 2m$, confirming the economic interpretation of the Lagrange multiplier as the marginal change in prediction variance per unit shift in the constraint.
    (c) Using $m = (a_0 - 1) / q_0$, explain under what conditions on the location $\mathbf{s}_0$ the Lagrange multiplier satisfies $m < 0$, $m = 0$, or $m > 0$.

**Computational and simulation**

16. For the rent district, compute the universal kriging standard error along the line from $(0.5, 0.5)$ to $(2, 0.5)$, east out of the district. Plot the simple kriging variance and the trend cost separately. Where does the trend cost overtake the simple kriging variance?

17. Repeat the coverage experiment of Section 4.8.3 for the other target: a new observation at each grid location, with intervals $\hat{Y} \pm 1.96\sqrt{\sigma^2_{\text{UK}} + \tau^2}$. Confirm about 95% coverage, and show that the intervals for $\tilde{Y}$ cover new observations far less often.

18. Run the coverage experiment with a misspecified covariance: krige with $\phi = 0.05$ or $\phi = 0.4$ while simulating with $\phi = 0.15$. Report coverage and RMSE. Which mistake is worse, and why?

19. Use the LOOCV shortcut to choose $\phi$ for the rent district: compute the leave-one-out root mean squared error for $\phi$ from 0.05 to 0.40, keeping $\sigma^2$ and $\tau^2$ fixed. Does the minimum fall near the true value 0.15?

20. Compute block kriging variances for squares $[0.5 - w/2, 0.5 + w/2]^2$ with $w$ from 0.02 to 0.8, and plot them against $w$. Explain the shape of the curve, including its value as $w \to 0$.

---

## What Comes Next

This chapter assumed the covariance parameters were known. Chapter 5 estimates them by maximum likelihood and restricted maximum likelihood, using the same matrices that appeared here: $\boldsymbol{\Sigma}^{-1}$, the GLS estimate, and the residual precision $\boldsymbol{\Pi}$, which turns out to be the heart of the restricted likelihood. Section 5.4.5 compares the kriging variances computed at the maximum likelihood, REML, and true parameters and discusses what plugging in estimates costs, and Section 5.6.1 uses the leave-one-out residuals of Section 4.7 as a diagnostic.
