# Chapter 8: Conditional Autoregressive Models and Sparse Precision Matrices

Chapter 7 built the covariance of a spatial random effect from a simultaneous system: every region's value is a function of its neighbors' values and a fresh error, all at once. This chapter starts from a different question. Suppose we knew the values in every region except one. What should that region's value look like?

A natural answer is a regression on the neighbors. A county's unexplained health outcome should resemble the average of the counties around it, with some leftover variation that shrinks when the county has many neighbors to learn from. The same idea fits a spot on a tissue slide, whose gene expression resembles the average of the six spots touching it. Writing one such regression for every region, and asking what joint distribution they imply, gives the **conditional autoregressive** (CAR) model. The word *autoregressive* refers to the regression of a region on its neighbors, and *conditional* says that each regression is stated given all the other regions.

The chapter has a plan. We first show how the conditional regressions determine the joint distribution, and how that distribution is described by a **precision matrix**, the inverse of the covariance matrix, which turns out to be the natural object. Then we meet the two models used throughout applied work: the **proper CAR**, whose parameter $\rho$ controls the strength of spatial dependence, and the **intrinsic CAR** (ICAR), the limiting case that smooths without limit and has a different kind of density. We then see how to compute with precision matrices, which is where the models earn their popularity: precision matrices of CAR models are sparse, and sparse matrices make maps with thousands of regions routine. We end by comparing CAR with the SAR model of Chapter 7. Chapter 9 uses both CAR and ICAR as random effects in models for counts.

> **Chapter at a Glance**
>
> - **The Conditional Idea (§8.1):** Each region is regressed on its neighbors. A tool relates conditional distributions to the precision matrix; a Brook's-lemma calculation shows when the conditionals fit together; zeros in the precision matrix mean conditional independence.
> - **The Proper CAR Model (§8.2):** Equal weights on neighbors give the precision $\sigma^{-2}(\mathbf{D} - \rho\mathbf{A})$. A four-region map by hand, the allowed range of $\rho$, and why $\sigma^2$ is not the marginal variance.
> - **The Intrinsic CAR Model (§8.3):** At $\rho = 1$ the quadratic form becomes a sum of squared neighbor differences. The precision matrix is singular, so the density is improper, and constraints and a generalized determinant repair it.
> - **Computing With Precision Matrices (§8.4):** The log-likelihood from eigenvalues, simulation from a factor of $\mathbf{Q}$, and sparse storage and solves for thousands of regions.
> - **SAR Versus CAR (§8.5):** Two precision matrices with different sparsity patterns, and two parameters named $\rho$ that cannot be compared.
> - **Walkthrough and Lab (§8.6):** Simulating a CAR field on a 144-county map, recovering its parameters, and four figures.

---

## 8.1 The Conditional Idea

### 8.1.1 Full Conditionals

Let $\mathbf{u} = (u_1, \dots, u_n)'$ be the spatial random effect on $n$ regions, with mean zero. (With covariates, $\mathbf{u}$ is what is left after removing $\mathbf{X}\boldsymbol{\beta}$, as in Chapter 7.) Write $\mathbf{u}_{-i}$ for the vector $\mathbf{u}$ with its $i$th entry removed. A **full conditional** is the distribution of one region's value given all the others. The CAR approach specifies each one as a normal distribution:

$$
u_i \mid \mathbf{u}_{-i} \sim N\!\left(\sum_{j \ne i} c_{ij}u_j,\ \sigma_i^2\right), \qquad i = 1, \dots, n
$$

The numbers $c_{ij}$ are regression coefficients, with $c_{ii} = 0$ and $c_{ij} = 0$ unless regions $i$ and $j$ are neighbors. The numbers $\sigma_i^2$ are the **conditional variances**. This is a linear model for each region: the conditional mean of $u_i$ is a linear combination of its neighbors' values, and $\sigma_i^2$ is the error variance. Nothing is fitted to data here. These are assumptions about the joint distribution of $\mathbf{u}$, in the same way that a variogram model is an assumption.

The contrast with Chapter 7 is worth stating. The SAR model specifies a recipe for generating $\mathbf{u}$ from independent errors, and the conditional distributions are a by-product that nobody asked for. The CAR model specifies the conditional distributions, and the joint distribution is the by-product. Two questions follow. Do $n$ conditional distributions always fit together into a joint distribution? And what does the joint distribution look like? Sections 8.1.2 and 8.1.3 answer them.

We will use the four-region hub map of Section 6.3.1 throughout the chapter, small enough to do every calculation by hand. Region 1 touches only region 2. Region 2 touches regions 1, 3, and 4. Regions 3 and 4 touch each other and region 2. So region 2 is a hub, regions 3 and 4 form a triangle with it, and region 1 is a pendant region hanging off the hub. The number of neighbors of each region, its degree $d_i$ (Section 6.3.1), is 1, 3, 2, and 2.

```python
import numpy as np
from scipy import linalg, optimize, stats
from scipy import sparse
from scipy.sparse import linalg as spla

# The hub map of Section 6.3.1, regions 1 to 4 in rows 0 to 3: region 2
# touches 1, 3, and 4; regions 3 and 4 touch each other.
A4 = np.array([[0, 1, 0, 0],
               [1, 0, 1, 1],
               [0, 1, 0, 1],
               [0, 1, 1, 0]], dtype=float)
d4 = A4.sum(axis=1)
D4 = np.diag(d4)
print("number of neighbors:", d4)
```

### 8.1.2 A Tool We Need: Conditional Distributions From the Precision Matrix

Suppose $\mathbf{u} \sim N(\mathbf{0}, \mathbf{Q}^{-1})$, where $\mathbf{Q}$ is a symmetric positive definite matrix. Chapter 7 met $\mathbf{Q}$ as the precision matrix of the SAR model: the inverse of the covariance matrix. The density of $\mathbf{u}$ is proportional to $\exp(-\tfrac{1}{2}\mathbf{u}'\mathbf{Q}\mathbf{u})$. We want the distribution of $u_i$ given the other entries, and the precision matrix gives it with almost no work.

The conditional density of $u_i$ given $\mathbf{u}_{-i}$ equals the joint density divided by the marginal density of $\mathbf{u}_{-i}$. The marginal does not involve $u_i$, so as a function of $u_i$ the conditional density is proportional to the joint density. We only need to find the part of $\mathbf{u}'\mathbf{Q}\mathbf{u}$ that depends on $u_i$. Splitting the double sum into terms with $i$ in neither, one, or both positions, and using $Q_{ij} = Q_{ji}$,

$$
\mathbf{u}'\mathbf{Q}\mathbf{u} = Q_{ii}u_i^2 + 2u_i\sum_{j \ne i}Q_{ij}u_j + \sum_{j \ne i}\sum_{k \ne i}Q_{jk}u_ju_k
$$

The last term does not involve $u_i$, so it is a constant for our purposes. Write $s = \sum_{j \ne i}Q_{ij}u_j$. Completing the square in $u_i$,

$$
Q_{ii}u_i^2 + 2u_is = Q_{ii}\left(u_i + \frac{s}{Q_{ii}}\right)^2 - \frac{s^2}{Q_{ii}}
$$

and the last piece is again a constant. So, as a function of $u_i$,

$$
p(u_i \mid \mathbf{u}_{-i}) \propto \exp\!\left(-\frac{Q_{ii}}{2}\left(u_i + \frac{s}{Q_{ii}}\right)^2\right)
$$

which is the density of a normal distribution with

$$
\operatorname{E}[u_i \mid \mathbf{u}_{-i}] = -\sum_{j \ne i}\frac{Q_{ij}}{Q_{ii}}u_j, \qquad \operatorname{Var}(u_i \mid \mathbf{u}_{-i}) = \frac{1}{Q_{ii}}
$$

Comparing with Section 8.1.1 reads off the CAR parameters from $\mathbf{Q}$:

$$
c_{ij} = -\frac{Q_{ij}}{Q_{ii}}, \qquad \sigma_i^2 = \frac{1}{Q_{ii}}
$$

or, turned around, $Q_{ii} = 1/\sigma_i^2$ and $Q_{ij} = -c_{ij}/\sigma_i^2$. The precision matrix holds all $n$ conditional regressions at once, one per row: row $i$ divided by $Q_{ii}$ gives region $i$'s coefficients, and $1/Q_{ii}$ gives its error variance.

**The same answer from the covariance matrix.** Chapter 4 derived the general formula for a conditional distribution through the Schur complement. Put $u_i$ in the last block of the vector, so that $\boldsymbol{\Sigma} = \mathbf{Q}^{-1}$ is partitioned with $\boldsymbol{\Sigma}_{-i,-i}$ in the upper left and $\Sigma_{ii}$ in the lower right. The Schur complement of the upper-left block is $F = \Sigma_{ii} - \boldsymbol{\Sigma}_{i,-i}\boldsymbol{\Sigma}_{-i,-i}^{-1}\boldsymbol{\Sigma}_{-i,i}$, and the block inversion formula of Section 4.3.4 says that the lower-right entry of the inverse is $1/F$. That entry is $Q_{ii}$. So $1/Q_{ii} = F$, the conditional variance, as above. The conditional mean is $\boldsymbol{\Sigma}_{i,-i}\boldsymbol{\Sigma}_{-i,-i}^{-1}\mathbf{u}_{-i}$.

This is a kriging calculation. The weights $\boldsymbol{\Sigma}_{i,-i}\boldsymbol{\Sigma}_{-i,-i}^{-1}$ are the simple kriging weights of Section 4.2.2 for predicting $u_i$ from all the other regions, and $F$ is the simple kriging variance. The precision matrix delivers all $n$ such kriging problems, each with $n - 1$ data values, from the entries of one matrix, with no matrix inversion of size $n - 1$ at all.

The code checks the identities on the four-region map, with the precision matrix $\mathbf{Q} = \mathbf{D} - 0.5\mathbf{A}$ (Section 8.2 explains where this matrix comes from; any positive definite $\mathbf{Q}$ would do). We compute region 3's weights both ways.

```python
Q4 = D4 - 0.5 * A4
Sigma4 = np.linalg.inv(Q4)
i, rest = 2, [0, 1, 3]
S_ir = Sigma4[i, rest]
S_rr = Sigma4[np.ix_(rest, rest)]
weights = np.linalg.solve(S_rr, S_ir)
cond_var = Sigma4[i, i] - S_ir @ weights
print("weights from the covariance matrix:", np.round(weights, 4) + 0.0)
print("-Q_ij / Q_ii:                      ", -Q4[i, rest] / Q4[i, i] + 0.0)
print("conditional variance:", round(cond_var, 4),
      " 1/Q_ii:", 1 / Q4[i, i])

y = np.array([1.5, 0.3, -1.0, 0.8])
for i in range(4):
    others = [j for j in range(4) if j != i]
    cond_mean = -Q4[i, others] @ y[others] / Q4[i, i]
    print(f"region {i + 1}: y_i - E[y_i | rest] = {y[i] - cond_mean:7.4f}   "
          f"(Q y)_i / Q_ii = {(Q4 @ y)[i] / Q4[i, i]:7.4f}")
```

Region 3's neighbors are regions 2 and 4. Its conditional mean is $0.25u_2 + 0.25u_4$, with weight 0 on region 1, and its conditional variance is 0.5. Both routes agree. The zero weight on region 1 is the **screening effect** of Section 4.2.3 in its purest form: given its neighbors, region 3 has no use for region 1.

**The connection to leave-one-out residuals.** Chapter 4 proved that the leave-one-out residual of a spatial prediction is $(\boldsymbol{\Pi}\mathbf{Y})_i/\Pi_{ii}$, and noted that with a known mean and no covariates, $\boldsymbol{\Pi}$ reduces to $\boldsymbol{\Sigma}^{-1}$, which is the precision matrix. The formula above shows why that result is natural. For data $\mathbf{y}$ with known mean $\boldsymbol{\mu}$ and precision matrix $\mathbf{Q}$, the leave-one-out residual of $y_i$ is $y_i - \operatorname{E}[y_i \mid \mathbf{y}_{-i}]$, the difference between $y_i$ and its prediction from all the other values. Substituting the conditional mean,

$$
y_i - \operatorname{E}[y_i \mid \mathbf{y}_{-i}] = (y_i - \mu_i) + \sum_{j \ne i}\frac{Q_{ij}}{Q_{ii}}(y_j - \mu_j) = \frac{1}{Q_{ii}}\sum_{j}Q_{ij}(y_j - \mu_j) = \frac{[\mathbf{Q}(\mathbf{y} - \boldsymbol{\mu})]_i}{Q_{ii}}
$$

which is Chapter 4's formula with $\boldsymbol{\Pi} = \mathbf{Q}$. The last lines of the code confirm it numerically for an arbitrary data vector. So Chapter 4's cross-validation shortcut and the CAR conditional mean are the same fact seen from two directions: a conditional mean is a kriging prediction, and a kriging prediction computed from a precision matrix is a row operation.

### 8.1.3 Reconstructing the Joint Distribution: Brook's Lemma by Hand

Now the other direction. Start from the conditionals of Section 8.1.1 and ask for the joint density. The tool is **Brook's lemma** [@brook1964]: a joint density can be rebuilt from its full conditionals, up to a constant, by changing one coordinate at a time. We show it for three regions, with a reference point at which all values are zero.

Write the conditional means as $m_i = \sum_j c_{ij}u_j$. Comparing the conditional density at a value $u_i$ with the density at 0, with the other regions held fixed, the normalizing constants cancel and

$$
\frac{p(u_i \mid \mathbf{u}_{-i})}{p(0 \mid \mathbf{u}_{-i})} = \exp\!\left(-\frac{(u_i - m_i)^2 - m_i^2}{2\sigma_i^2}\right) = \exp\!\left(-\frac{u_i^2 - 2u_im_i}{2\sigma_i^2}\right)
$$

Now move from the point $(0, 0, 0)$ to the point $(u_1, u_2, u_3)$ in three steps, changing one coordinate each time:

$$
\frac{p(u_1, u_2, u_3)}{p(0, 0, 0)} = \frac{p(u_1, u_2, u_3)}{p(0, u_2, u_3)}\cdot\frac{p(0, u_2, u_3)}{p(0, 0, u_3)}\cdot\frac{p(0, 0, u_3)}{p(0, 0, 0)}
$$

In each ratio only one coordinate changes, so the common marginal density of the other two cancels, and each ratio is a ratio of conditional densities:

$$
\frac{p(u_1 \mid u_2, u_3)}{p(0 \mid u_2, u_3)}\cdot\frac{p(u_2 \mid 0, u_3)}{p(0 \mid 0, u_3)}\cdot\frac{p(u_3 \mid 0, 0)}{p(0 \mid 0, 0)}
$$

In the first factor the conditional mean is $m_1 = c_{12}u_2 + c_{13}u_3$. In the second, $u_1$ has already been set to zero, so $m_2 = c_{23}u_3$. In the third, both others are zero, so $m_3 = 0$. Multiplying the three exponentials,

$$
\log\frac{p(\mathbf{u})}{p(\mathbf{0})} = -\frac{1}{2}\left[\frac{u_1^2 - 2u_1(c_{12}u_2 + c_{13}u_3)}{\sigma_1^2} + \frac{u_2^2 - 2u_2c_{23}u_3}{\sigma_2^2} + \frac{u_3^2}{\sigma_3^2}\right]
$$

This is a complete formula for the joint density, up to a constant. But the order in which we moved the coordinates was arbitrary. Moving them in the opposite order, $u_3$ first, then $u_2$, then $u_1$, gives

$$
\log\frac{p(\mathbf{u})}{p(\mathbf{0})} = -\frac{1}{2}\left[\frac{u_3^2 - 2u_3(c_{31}u_1 + c_{32}u_2)}{\sigma_3^2} + \frac{u_2^2 - 2u_2c_{21}u_1}{\sigma_2^2} + \frac{u_1^2}{\sigma_1^2}\right]
$$

If a joint density exists, both formulas describe it, so they must agree. Compare the coefficient of the cross term $u_1u_2$. The first formula has $c_{12}/\sigma_1^2$, and the second has $c_{21}/\sigma_2^2$. The same comparison for $u_1u_3$ and $u_2u_3$ gives three conditions, which are all instances of one:

$$
\frac{c_{ij}}{\sigma_i^2} = \frac{c_{ji}}{\sigma_j^2} \quad\Longleftrightarrow\quad c_{ij}\sigma_j^2 = c_{ji}\sigma_i^2
$$

This is the **symmetry requirement**. It says that the regression coefficients and error variances cannot be chosen independently for each region: if region $i$ uses region $j$ strongly, region $j$ must use region $i$ in a matching proportion. When it holds, the exponent equals $-\tfrac{1}{2}\mathbf{u}'\mathbf{Q}\mathbf{u}$ with

$$
Q_{ii} = \frac{1}{\sigma_i^2}, \qquad Q_{ij} = -\frac{c_{ij}}{\sigma_i^2} = -\frac{c_{ji}}{\sigma_j^2} = Q_{ji}
$$

In matrix form, with $\mathbf{C}$ the matrix of the $c_{ij}$ and $\boldsymbol{\Sigma}_c = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)$,

$$
\mathbf{Q} = \boldsymbol{\Sigma}_c^{-1}(\mathbf{I}_n - \mathbf{C})
$$

The same calculation works for any $n$. The joint density exists and is a proper normal density when two conditions hold: the symmetry requirement, so that $\mathbf{Q}$ is symmetric, and positive definiteness of $\mathbf{Q}$, so that $\exp(-\tfrac{1}{2}\mathbf{u}'\mathbf{Q}\mathbf{u})$ can be normalized. Symmetry is a condition on the choices of $c_{ij}$ and $\sigma_i^2$. Positive definiteness is a separate condition, usually a restriction on parameters such as $\rho$ (Section 8.2.3).

**A three-region example.** Let regions 1, 2, 3 lie in a row, with $c_{12} = 0.4$, $c_{21} = 0.2$, $c_{23} = 0.3$, $c_{32} = 0.6$, $c_{13} = c_{31} = 0$, and conditional variances $\sigma_1^2 = 1$, $\sigma_2^2 = 0.5$, $\sigma_3^2 = 1$. Check the symmetry requirement: $c_{12}\sigma_2^2 = 0.4 \times 0.5 = 0.2$ and $c_{21}\sigma_1^2 = 0.2 \times 1 = 0.2$; $c_{23}\sigma_3^2 = 0.3 \times 1 = 0.3$ and $c_{32}\sigma_2^2 = 0.6 \times 0.5 = 0.3$. Both hold. Then $Q_{11} = 1$, $Q_{22} = 2$, $Q_{33} = 1$, $Q_{12} = -0.4/1 = -0.4$ (and $-0.2/0.5 = -0.4$ from the other side), $Q_{23} = -0.3/0.5 = -0.6$, and

$$
\mathbf{Q} = \begin{bmatrix} 1 & -0.4 & 0 \\ -0.4 & 2 & -0.6 \\ 0 & -0.6 & 1 \end{bmatrix}
$$

If instead we changed $c_{21}$ to 0.3, then $c_{21}\sigma_1^2 = 0.3 \ne 0.2 = c_{12}\sigma_2^2$, and no joint distribution has those conditionals.

```python
c = np.array([[0.0, 0.4, 0.0],
              [0.2, 0.0, 0.3],
              [0.0, 0.6, 0.0]])
cond_var3 = np.array([1.0, 0.5, 1.0])
Q3 = (np.eye(3) - c) / cond_var3[:, None]
print("Q from the conditionals:\n", Q3)
print("symmetric:", np.allclose(Q3, Q3.T),
      " eigenvalues:", np.round(np.linalg.eigvalsh(Q3), 4))
c_bad = c.copy()
c_bad[1, 0] = 0.3
Q_bad = (np.eye(3) - c_bad) / cond_var3[:, None]
print("with c_21 changed to 0.3, symmetric:", np.allclose(Q_bad, Q_bad.T))
```

The matrix is symmetric and positive definite (eigenvalues 0.62, 1, and 2.38), so these three conditionals define a proper joint normal distribution. With $c_{21}$ changed, the matrix $\boldsymbol{\Sigma}_c^{-1}(\mathbf{I}_n - \mathbf{C})$ is not symmetric, which no covariance or precision matrix can be.

### 8.1.4 Gaussian Markov Random Fields

Section 8.1.2 shows that the conditional distribution of $u_i$ depends on $u_j$ only through the weight $Q_{ij}$. If $Q_{ij} = 0$, region $j$ drops out of the conditional distribution of region $i$. A stronger statement is true: $u_i$ and $u_j$ are **conditionally independent** given all the other regions exactly when $Q_{ij} = 0$. To see it, fix the other regions and collect the terms of $\mathbf{u}'\mathbf{Q}\mathbf{u}$ that involve $u_i$ or $u_j$:

$$
Q_{ii}u_i^2 + Q_{jj}u_j^2 + 2Q_{ij}u_iu_j + 2u_is_i + 2u_js_j
$$

where $s_i$ and $s_j$ collect the terms with the fixed regions. If $Q_{ij} = 0$ the conditional density, proportional to $\exp$ of minus half this, is a product of a function of $u_i$ and a function of $u_j$, which is the definition of independence. If $Q_{ij} \ne 0$ the cross term couples them.

A normal vector whose precision matrix has $Q_{ij} \ne 0$ only for neighboring regions, so that each region depends on the rest only through its neighbors, is a **Gaussian Markov random field** (GMRF) [@rue2005]. The CAR models are GMRFs by construction, since $c_{ij} = 0$ for non-neighbors.

Conditional independence is not independence. In the four-region model of Section 8.2.2, regions 1 and 3 are not neighbors, so $Q_{13} = 0$, yet their covariance $\Sigma_{13}$ is not zero: regions 1 and 3 are both tied to region 2, and through it they are related. This gap between zeros in $\mathbf{Q}$ and zeros in $\boldsymbol{\Sigma}$ drives the rest of the chapter. The precision matrix is sparse and cheap to store and work with. The covariance matrix is dense.

---

## 8.2 The Proper CAR Model

### 8.2.1 Equal Weights for Neighbors

The general framework leaves the $c_{ij}$ and $\sigma_i^2$ free. The standard **proper CAR** model [@besag1974] makes two simple choices.

- **Equal weights.** A region's conditional mean is $\rho$ times the average of its neighbors:

  $$
  c_{ij} = \rho\,\frac{A_{ij}}{d_i} = \rho\,w_{ij}
  $$

  where $w_{ij} = A_{ij}/d_i$ are the row-standardized weights of Section 6.3.2. Each of the $d_i$ neighbors gets the same weight $\rho/d_i$.
- **Variance shrinking with the number of neighbors.**

  $$
  \sigma_i^2 = \frac{\sigma^2}{d_i}
  $$

  A region with many neighbors has many readings to learn from, so less is left unexplained. The parameter $\sigma^2$ is the conditional variance a region would have if it had a single neighbor.

Check the symmetry requirement:

$$
c_{ij}\sigma_j^2 = \rho\,\frac{A_{ij}}{d_i}\cdot\frac{\sigma^2}{d_j} = \frac{\rho\sigma^2A_{ij}}{d_id_j}
$$

which is symmetric in $i$ and $j$ because $A_{ij} = A_{ji}$. So the choices fit together, and Section 8.1.3 gives the precision matrix. Its diagonal entries are $Q_{ii} = 1/\sigma_i^2 = d_i/\sigma^2$, and its off-diagonal entries are

$$
Q_{ij} = -\frac{c_{ij}}{\sigma_i^2} = -\rho\,\frac{A_{ij}}{d_i}\cdot\frac{d_i}{\sigma^2} = -\frac{\rho A_{ij}}{\sigma^2}
$$

Together,

$$
\mathbf{Q}_{\text{CAR}} = \sigma^{-2}(\mathbf{D} - \rho\mathbf{A})
$$

The proper CAR model is $\mathbf{u} \sim N\!\left(\mathbf{0}, \sigma^2(\mathbf{D} - \rho\mathbf{A})^{-1}\right)$, whenever $\mathbf{D} - \rho\mathbf{A}$ is positive definite (Section 8.2.3). From here on, in this chapter, write $\boldsymbol{\Sigma} = \mathbf{Q}^{-1}$ for the covariance matrix of $\mathbf{u}$.

### 8.2.2 A Four-Region Map by Hand

Take $\sigma^2 = 1$ and $\rho = 0.5$ on the four-region map. The conditional variances are $\sigma^2/d_i = 1, 1/3, 1/2, 1/2$. The conditional mean of region 2 is $0.5 \times \tfrac{1}{3}(u_1 + u_3 + u_4)$, half the average of its three neighbors. The conditional mean of region 1, whose only neighbor is region 2, is $0.5u_2$. The precision matrix is $\mathbf{D} - 0.5\mathbf{A}$:

$$
\mathbf{Q} = \begin{bmatrix} 1 & -0.5 & 0 & 0 \\ -0.5 & 3 & -0.5 & -0.5 \\ 0 & -0.5 & 2 & -0.5 \\ 0 & -0.5 & -0.5 & 2 \end{bmatrix}
$$

The covariance matrix is its inverse. We can find the columns of $\boldsymbol{\Sigma}$ by hand, because the map's symmetry (regions 3 and 4 are interchangeable) reduces each system to two unknowns. Take the column for region 1, the vector $\mathbf{g}$ with $\mathbf{Q}\mathbf{g} = \mathbf{e}_1$, where $\mathbf{e}_1$ has a 1 in position 1 and zeros elsewhere. The row for region 1 says $g_1 - 0.5g_2 = 1$. The rows for regions 3 and 4 are equal, since $g_3 = g_4$, and each reads $2g_3 - 0.5g_2 - 0.5g_3 = 0$, so $g_3 = g_2/3$. The row for region 2 says $3g_2 - 0.5(g_1 + g_3 + g_4) = 0$. Substituting $g_3 = g_4 = g_2/3$ and $g_1 = 1 + 0.5g_2$,

$$
3g_2 - \frac{g_2}{3} - \frac{1}{2} - \frac{g_2}{4} = 0 \quad\Longrightarrow\quad \frac{29}{12}g_2 = \frac{1}{2} \quad\Longrightarrow\quad g_2 = \frac{6}{29}
$$

so $\Sigma_{21} = 6/29 \approx 0.2069$, $\Sigma_{31} = \Sigma_{41} = g_2/3 = 2/29 \approx 0.0690$, and $\Sigma_{11} = 1 + 3/29 = 32/29 \approx 1.1034$. The same steps for the column of region 2, with the right-hand side $\mathbf{e}_2$, give $\Sigma_{22} = 12/29 \approx 0.4138$. The code finishes the job and checks these numbers.

```python
rho, sigma2 = 0.5, 1.0
Q4 = (D4 - rho * A4) / sigma2
Sigma4 = np.linalg.inv(Q4)
print("Q:\n", Q4)
print("Sigma = Q^{-1}:\n", np.round(Sigma4, 4))
print("conditional variances:", 1 / np.diag(Q4))
print("marginal variances:   ", np.round(np.diag(Sigma4), 4))
corr4 = Sigma4 / np.sqrt(np.outer(np.diag(Sigma4), np.diag(Sigma4)))
print("correlations:\n", np.round(corr4, 4))
print("Q_13 =", Q4[0, 2], " but Sigma_13 =", round(Sigma4[0, 2], 4))
```

Four lessons come from these matrices.

- **Marginal variances differ from conditional variances, and from each other.** The conditional variances are $1, 1/3, 1/2, 1/2$. The marginal variances are $1.1034, 0.4138, 0.5793, 0.5793$. Marginal variances are always larger, because knowing the neighbors can only reduce uncertainty. The pendant region 1 has the largest marginal variance: it has only one neighbor to anchor it, so it wanders more. The hub region 2, tied to three regions, is held in place.
- **The parameter $\sigma^2 = 1$ is not the variance of anything we can see.** No region has marginal variance 1; region 1 comes closest at 1.10. Section 8.2.4 returns to this.
- **Neighbors are correlated, and the correlations differ by pair.** Region 2 and its neighbors have correlations of 0.28 (with regions 3 and 4) and 0.31 (with region 1); regions 3 and 4 have 0.31.
- **Non-neighbors can still be correlated.** Regions 1 and 3 are not neighbors, $Q_{13} = 0$, yet $\Sigma_{13} = 2/29 \approx 0.069$ and their correlation is 0.086. That is the gap between conditional independence and independence from Section 8.1.4.

### 8.2.3 When Is the Model Valid?

The matrix $\mathbf{D} - \rho\mathbf{A}$ has to be positive definite, or $\mathbf{Q}$ is not a precision matrix. Which values of $\rho$ allow that? Define the **normalized adjacency matrix**

$$
\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
$$

It is symmetric, with entries $A_{ij}/\sqrt{d_id_j}$. Let $\tilde{\lambda}_1 \ge \dots \ge \tilde{\lambda}_n$ be its eigenvalues. Factor $\mathbf{D} - \rho\mathbf{A} = \mathbf{D}^{1/2}(\mathbf{I}_n - \rho\tilde{\mathbf{A}})\mathbf{D}^{1/2}$ and, for any vector $\mathbf{a} \ne \mathbf{0}$, write $\mathbf{b} = \mathbf{D}^{1/2}\mathbf{a}$, which is also nonzero:

$$
\mathbf{a}'(\mathbf{D} - \rho\mathbf{A})\mathbf{a} = \mathbf{b}'(\mathbf{I}_n - \rho\tilde{\mathbf{A}})\mathbf{b}
$$

The right side is positive for every $\mathbf{b} \ne \mathbf{0}$ exactly when all eigenvalues of $\mathbf{I}_n - \rho\tilde{\mathbf{A}}$ are positive. Those eigenvalues are $1 - \rho\tilde{\lambda}_k$. So the model is valid when $1 - \rho\tilde{\lambda}_k > 0$ for every $k$, that is,

$$
\frac{1}{\tilde{\lambda}_{\min}} < \rho < \frac{1}{\tilde{\lambda}_{\max}}
$$

(the lower bound applies if $\tilde{\lambda}_{\min} < 0$, which is always the case for a map with any neighbors).

The upper end is always 1. The vector $\mathbf{D}^{1/2}\mathbf{1}$ is an eigenvector of $\tilde{\mathbf{A}}$ with eigenvalue 1, because

$$
\tilde{\mathbf{A}}\mathbf{D}^{1/2}\mathbf{1} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{1} = \mathbf{D}^{-1/2}\mathbf{D}\mathbf{1} = \mathbf{D}^{1/2}\mathbf{1}
$$

using $\mathbf{A}\mathbf{1} = \mathbf{D}\mathbf{1}$, which is the definition of the degrees. Section 8.2.5 (Optional) shows that no eigenvalue exceeds 1 or falls below $-1$. So the largest eigenvalue is exactly 1, and the valid range is

$$
\frac{1}{\tilde{\lambda}_{\min}} < \rho < 1
$$

with a lower end at or below $-1$. At $\rho = 1$ the matrix $\mathbf{D} - \mathbf{A}$ has the zero eigenvalue just found, so the model degenerates into the intrinsic CAR of Section 8.3.

The eigenvalues also depend on the map. For the four-region map, the code finds $\tilde{\lambda} = -0.7287, -0.5, 0.2287, 1$, so the valid range is $-1.3723 < \rho < 1$. Checking the smallest eigenvalue of $\mathbf{D} - \rho\mathbf{A}$ directly at several values of $\rho$ confirms the boundaries: it is positive at 0.9 and 0.99, zero to rounding error at 1, negative at 1.05; on the negative side it is positive at $-1.30$ and negative at $-1.50$.

```python
B4 = A4 / np.sqrt(np.outer(d4, d4))
lam4 = np.linalg.eigvalsh(B4)
print("eigenvalues of D^-1/2 A D^-1/2:", np.round(lam4, 4))
print("valid rho: between", round(1 / lam4.min(), 4), "and", 1 / lam4.max())
for rho_try in (0.9, 0.99, 1.0, 1.05, -1.3, -1.5):
    smallest = np.linalg.eigvalsh(D4 - rho_try * A4).min()
    print(f"rho={rho_try:5.2f}: smallest eigenvalue of D - rho A = "
          f"{smallest:8.4f}")
```

Negative values of $\rho$, which make neighbors tend to differ, are rarely used. On the 144-county map of the next subsection the range is $-1 < \rho < 1$, because a rectangular grid is bipartite (Exercise 5). In practice, positive $\rho$ near 1 is what applied models need, and Section 8.5.3 shows that $\rho$ must be very close to 1 for strong spatial correlation.

### 8.2.4 Why $\sigma^2$ Is Not the Marginal Variance

The covariance matrix of the proper CAR is $\boldsymbol{\Sigma} = \sigma^2(\mathbf{D} - \rho\mathbf{A})^{-1}$, so the marginal variance of region $i$ is $\sigma^2[(\mathbf{D} - \rho\mathbf{A})^{-1}]_{ii}$, which depends on $i$ and on $\rho$. Even at $\rho = 0$ the model is $\mathbf{Q} = \sigma^{-2}\mathbf{D}$, independent regions with variances $\sigma^2/d_i$: the parameter $\sigma^2$ is not the variance of any region, even with no spatial dependence at all. It is a scale for the conditional variance, per neighbor.

This is worth seeing on the shared 144-county map, a $12 \times 12$ grid of squares, where regions on the edge have 3 neighbors and corner regions have 2. (This map is used in Chapters 6 to 9. The function below builds it, along with populations and a smooth covariate that later chapters use.) The code computes, for several $\rho$ and with $\sigma^2 = 1$, the average marginal variance among regions with 2, 3, and 4 neighbors.

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


S, A, pop, x = county_map()
n = len(A)
d = A.sum(axis=1)
D = np.diag(d)
print("regions:", n, " neighbors per region (2, 3, 4):",
      [int((d == k).sum()) for k in (2, 3, 4)])
lam = np.linalg.eigvalsh(A / np.sqrt(np.outer(d, d)))
print("eigenvalues of D^-1/2 A D^-1/2 range from",
      round(lam.min(), 4), "to", round(lam.max(), 4))

for rho in (0.0, 0.5, 0.9, 0.99):
    Sigma = np.linalg.inv(D - rho * A)
    mv = np.diag(Sigma)
    line = "  ".join(f"deg {k}: {mv[d == k].mean():.3f}" for k in (2, 3, 4))
    print(f"rho={rho:4.2f}  mean marginal variance (sigma2=1)  {line}")
```

At $\rho = 0$ the marginal variances are $1/2$, $1/3$, and $1/4$ for regions with 2, 3, and 4 neighbors. As $\rho$ rises they all grow, and they spread: at $\rho = 0.9$, corner regions have an average marginal variance of 0.826 and interior regions of 0.379, and at $\rho = 0.99$, 1.418 and 0.706. Regions near the boundary of the map, with fewer neighbors, are less constrained and so more variable. As $\rho \to 1$ all the variances grow without bound, which is what happens when the model becomes intrinsic. The marginal variance also changes with the resolution of the map; Section 13.5.1 explains why.

Two consequences matter in applied work. First, the heteroscedasticity (variance depending on the number of neighbors) is a property of the model and the map, not of the data. Second, an estimate $\hat{\sigma}^2$ cannot be read as "the variance of the spatial effect." Its meaning depends on the map and on $\rho$. Chapter 9 introduces a rescaled version of the intrinsic model (BYM2) designed so that the variance parameter does have a marginal interpretation. Figure 8.2 in Section 8.6.3 maps both conditional and marginal variances.

### 8.2.5 No Eigenvalue of $\tilde{\mathbf{A}}$ Exceeds 1 in Absolute Value (Optional)

*This subsection proves the bound used in Section 8.2.3. It can be skipped on a first reading.*

Let $\mathbf{a}$ be any vector. Then

$$
\lvert\mathbf{a}'\tilde{\mathbf{A}}\mathbf{a}\rvert = \left\lvert\sum_{i}\sum_{j}\frac{A_{ij}a_ia_j}{\sqrt{d_id_j}}\right\rvert \le \sum_{i}\sum_{j}A_{ij}\frac{\lvert a_i\rvert}{\sqrt{d_i}}\frac{\lvert a_j\rvert}{\sqrt{d_j}}
$$

For any two numbers $p$ and $q$, $pq \le (p^2 + q^2)/2$, because $(p - q)^2 \ge 0$. Apply it with $p = \lvert a_i\rvert/\sqrt{d_i}$ and $q = \lvert a_j\rvert/\sqrt{d_j}$:

$$
\le \frac{1}{2}\sum_{i}\sum_{j}A_{ij}\left(\frac{a_i^2}{d_i} + \frac{a_j^2}{d_j}\right) = \frac{1}{2}\left(\sum_i\frac{a_i^2}{d_i}\sum_jA_{ij} + \sum_j\frac{a_j^2}{d_j}\sum_iA_{ij}\right) = \frac{1}{2}\left(\sum_ia_i^2 + \sum_ja_j^2\right) = \mathbf{a}'\mathbf{a}
$$

using $\sum_jA_{ij} = d_i$ and, by symmetry, $\sum_iA_{ij} = d_j$. So $\lvert\mathbf{a}'\tilde{\mathbf{A}}\mathbf{a}\rvert \le \mathbf{a}'\mathbf{a}$ for every $\mathbf{a}$. If $\mathbf{a}$ is an eigenvector with eigenvalue $\tilde{\lambda}$, then $\mathbf{a}'\tilde{\mathbf{A}}\mathbf{a} = \tilde{\lambda}\mathbf{a}'\mathbf{a}$, and so $\lvert\tilde{\lambda}\rvert \le 1$. $\square$

---

## 8.3 The Intrinsic CAR Model

### 8.3.1 Setting $\rho = 1$

At $\rho = 1$ the precision matrix becomes $\mathbf{Q}_{\text{ICAR}} = \sigma^{-2}(\mathbf{D} - \mathbf{A})$, the **intrinsic CAR** (ICAR) model [@besag1974]. The conditional distributions are still perfectly well defined:

$$
u_i \mid \mathbf{u}_{-i} \sim N\!\left(\frac{1}{d_i}\sum_{j \sim i}u_j,\ \frac{\sigma^2}{d_i}\right)
$$

where $j \sim i$ means that $j$ is a neighbor of $i$. In words, **a region's value is the average of its neighbors, plus noise whose variance shrinks with the number of neighbors**. There is no pull toward zero, which the factor $\rho < 1$ supplied in the proper model. The model is called *intrinsic* because, like the intrinsically stationary processes of Section 2.2.3, it describes differences between values and says nothing about their overall level. The remainder of this section shows what that means.

### 8.3.2 The Quadratic Form Is a Sum of Squared Differences

The reason the intrinsic model describes differences is that its quadratic form is made of them. For a symmetric adjacency matrix,

$$
\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \tfrac{1}{2}\sum_{i=1}^n\sum_{j=1}^nA_{ij}(u_i - u_j)^2 = \sum_{i \sim j}(u_i - u_j)^2
$$

where the last sum counts each neighbor pair once. To see it, expand the middle expression:

$$
\tfrac{1}{2}\sum_i\sum_jA_{ij}(u_i^2 - 2u_iu_j + u_j^2) = \tfrac{1}{2}\sum_iu_i^2\sum_jA_{ij} + \tfrac{1}{2}\sum_ju_j^2\sum_iA_{ij} - \sum_i\sum_jA_{ij}u_iu_j
$$

The first two sums are each $\sum_id_iu_i^2 = \mathbf{u}'\mathbf{D}\mathbf{u}$, so together they give $\mathbf{u}'\mathbf{D}\mathbf{u}$, and the last is $\mathbf{u}'\mathbf{A}\mathbf{u}$. The check with numbers: on the four-region map with $\mathbf{u} = (1.5, 0.3, -1.0, 0.8)'$, both sides equal 6.62 (the code below).

There is a tidy way to say the same thing with a matrix. Let $\mathbf{O}$ be the **edge-by-region incidence matrix**: it has one row for each neighbor pair $(i, j)$, with $+1$ in column $i$, $-1$ in column $j$, and zeros elsewhere. Then $(\mathbf{O}\mathbf{u})$ is the vector of differences $u_i - u_j$ across all neighbor pairs, so $\mathbf{u}'\mathbf{O}'\mathbf{O}\mathbf{u} = \sum_{i \sim j}(u_i - u_j)^2$, and the identity above says

$$
\mathbf{D} - \mathbf{A} = \mathbf{O}'\mathbf{O}
$$

The matrix $\mathbf{D} - \mathbf{A}$ is called the **graph Laplacian** of the map. Because it has the form $\mathbf{O}'\mathbf{O}$, it is positive semi-definite: $\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \lVert\mathbf{O}\mathbf{u}\rVert^2 \ge 0$.

The density of the intrinsic model is therefore

$$
p(\mathbf{u}) \propto \exp\!\left(-\frac{1}{2\sigma^2}\sum_{i \sim j}(u_i - u_j)^2\right)
$$

Taking minus the logarithm, the model **penalizes roughness**: differences between neighbors are costly, and the cost is the squared difference. This is the same idea as a smoothing penalty in a regression with a spline or a ridge penalty, here applied to a map. On a path of regions $1 - 2 - \dots - n$ the penalty is $\sum_i(u_{i+1} - u_i)^2$, which is exactly the density of independent increments $u_{i+1} - u_i \sim N(0, \sigma^2)$: the **random walk** of Section 2.2.3, whose variogram grows without bound. The intrinsic CAR is the random walk generalized from a line to a map.

The proper CAR is a blend of two penalties. Since $\mathbf{D} - \rho\mathbf{A} = (1 - \rho)\mathbf{D} + \rho(\mathbf{D} - \mathbf{A})$,

$$
\mathbf{u}'(\mathbf{D} - \rho\mathbf{A})\mathbf{u} = (1 - \rho)\sum_id_iu_i^2 + \rho\sum_{i \sim j}(u_i - u_j)^2
$$

a ridge penalty on the sizes of the values (weighted by degree) plus a penalty on neighbor differences, with $\rho$ setting the mix. Chapter 9 builds the Leroux model by replacing $\mathbf{D}$ in the ridge term with the identity matrix.

### 8.3.3 Rank Deficiency

The vector of ones satisfies $(\mathbf{D} - \mathbf{A})\mathbf{1} = \mathbf{D}\mathbf{1} - \mathbf{A}\mathbf{1} = \mathbf{d} - \mathbf{d} = \mathbf{0}$, since $\mathbf{A}\mathbf{1}$ is the vector of degrees. So $\mathbf{D} - \mathbf{A}$ has a zero eigenvalue and is not invertible. More exactly, its null space (the vectors $\mathbf{u}$ with $(\mathbf{D} - \mathbf{A})\mathbf{u} = \mathbf{0}$) is the set of vectors that are **constant on each connected component** of the map. To prove it, observe that $(\mathbf{D} - \mathbf{A})\mathbf{u} = \mathbf{0}$ implies $\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = 0$, so every squared difference $(u_i - u_j)^2$ along a neighbor pair is zero, and so $u$ is constant along every path, hence constant on each connected component. Conversely, if $\mathbf{O}\mathbf{u} = \mathbf{0}$, then $(\mathbf{D} - \mathbf{A})\mathbf{u} = \mathbf{O}'\mathbf{O}\mathbf{u} = \mathbf{0}$. The number of zero eigenvalues equals the number of connected components.

The four-region map is connected, and the code finds Laplacian eigenvalues 0, 1, 3, and 4, with the null vector proportional to $\mathbf{1}$. A map made of a triangle (regions 1, 2, 3) and a separate pair (regions 4, 5) has two components, and its Laplacian has eigenvalues 0, 0, 2, 3, 3. The county map is connected and has one zero eigenvalue; its smallest nonzero eigenvalue, 0.0681, is close to zero. Small eigenvalues of a precision matrix belong to smooth patterns (compare Section 1.3.3: the patterns where neighbors move together), and a pattern with a small precision eigenvalue has a large variance.

```python
L4 = D4 - A4
w4, v4 = np.linalg.eigh(L4)
print("Laplacian eigenvalues:", np.round(w4, 4))
print("null vector (scaled, up to sign):", np.round(2 * np.abs(v4[:, 0]), 4))
u = np.array([1.5, 0.3, -1.0, 0.8])
pairs = np.argwhere(np.triu(A4) > 0)
print("u'Lu =", round(u @ L4 @ u, 4), " sum over neighbor pairs =",
      round(sum((u[a] - u[b])**2 for a, b in pairs), 4))
Lplus = np.linalg.pinv(L4)
print("pseudo-inverse of L:\n", np.round(Lplus, 4))
print("row sums:", np.round(Lplus.sum(axis=1), 10) + 0.0)
print("product of nonzero eigenvalues:", round(np.prod(w4[1:]), 4),
      " n times spanning trees:", 4 * 3)

# two islands: a triangle (regions 1, 2, 3) and a pair (regions 4, 5)
A5 = np.zeros((5, 5))
for a, b in ((0, 1), (0, 2), (1, 2), (3, 4)):
    A5[a, b] = A5[b, a] = 1
L5 = np.diag(A5.sum(axis=1)) - A5
print("two islands, eigenvalues:", np.round(np.linalg.eigvalsh(L5), 4) + 0.0)
```

What does a singular $\mathbf{Q}$ mean for the model? The function $\exp(-\tfrac{1}{2\sigma^2}\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u})$ is the same at $\mathbf{u}$ and at $\mathbf{u} + c\mathbf{1}$ for every constant $c$, so its integral over all of $\mathbb{R}^n$ is infinite: it is an **improper** density and cannot be normalized. There is no covariance matrix, because there is no inverse of $\mathbf{Q}$. The model fixes the pattern of differences and leaves the overall level completely unspecified, which is what *intrinsic* means.

### 8.3.4 Constraints and the Generalized Determinant

To use the intrinsic model we remove the free level. For a connected map, require $\sum_iu_i = 0$. That keeps the model as a distribution on the $(n-1)$-dimensional subspace of vectors orthogonal to $\mathbf{1}$, where $\mathbf{D} - \mathbf{A}$ is positive definite and the density is proper. Write $\xi_1 = 0 < \xi_2 \le \dots \le \xi_n$ for the Laplacian's eigenvalues, with eigenvectors $\mathbf{v}_1 \propto \mathbf{1}, \mathbf{v}_2, \dots, \mathbf{v}_n$. Three facts follow, with proofs in Section 8.3.5 (Optional).

- **Covariance.** On the constrained subspace the model has covariance $\sigma^2(\mathbf{D} - \mathbf{A})^{+}$, where $^{+}$ is the Moore-Penrose pseudo-inverse: invert the nonzero eigenvalues and leave the zero eigenvalue at zero, $(\mathbf{D} - \mathbf{A})^{+} = \sum_{k \ge 2}\xi_k^{-1}\mathbf{v}_k\mathbf{v}_k'$. For a connected map this equals $(\mathbf{D} - \mathbf{A} + \tfrac{1}{n}\mathbf{1}\mathbf{1}')^{-1} - \tfrac{1}{n}\mathbf{1}\mathbf{1}'$.
- **Simulation.** A draw is $\mathbf{u} = \sigma\sum_{k \ge 2}(z_k/\sqrt{\xi_k})\mathbf{v}_k$ with independent standard normal $z_k$, the spectral method of Section 2.7.2. By construction it sums to zero.
- **Normalizing constant.** The density on the subspace uses the **generalized determinant**, the product of the nonzero eigenvalues, $\lvert\mathbf{D} - \mathbf{A}\rvert_* = \prod_{k \ge 2}\xi_k$, in place of the determinant, which is zero.

For the four-region map, the pseudo-inverse has row sums of zero, and the constrained variances are $0.6875, 0.1875, 0.3542, 0.3542$ (times $\sigma^2$): again the pendant region 1 is the most variable, and the hub region 2 the least. The generalized determinant is $1 \times 3 \times 4 = 12$. This number has a combinatorial meaning that serves as a check: **Kirchhoff's matrix-tree theorem**, which we take from the literature, says that for a connected map, $\lvert\mathbf{D} - \mathbf{A}\rvert_* = n \times (\text{number of spanning trees})$. A spanning tree connects all regions with no loop; the four-region map has three of them (drop any one of the three edges of the triangle of regions 2, 3, and 4, and keep the edge to region 1), and $4 \times 3 = 12$.

For a map with $c$ connected components, require the values to sum to zero within each component. Equivalently, project onto the constrained subspace with

$$
\mathbf{P}_{\text{comp}} = \mathbf{I}_n - \sum_{k=1}^{c}\frac{1}{n_k}\mathbf{1}_k\mathbf{1}_k'
$$

where $\mathbf{1}_k$ indicates the regions in component $k$ and $n_k$ is its size. For a connected map, $\mathbf{P}_{\text{comp}}$ is the centering matrix $\mathbf{P}_c$ of Section 0.4.2.

The code below checks the pseudo-inverse formula, draws an intrinsic field on the county map by the spectral method, confirms that its sum is zero, and shows that the quadratic form does not change when a constant is added, while projection removes the constant.

```python
J4 = np.ones((4, 4)) / 4
print("L+ equals (L + 11'/n)^-1 - 11'/n:",
      np.allclose(Lplus, np.linalg.inv(L4 + J4) - J4))

w_c, v_c = np.linalg.eigh(D - A)
print("county map: zero eigenvalues of D - A:", int((w_c < 1e-10).sum()),
      " smallest nonzero:", round(w_c[1], 4))
z = np.random.default_rng(2060).standard_normal(n)
u_icar = v_c[:, 1:] @ (z[1:] / np.sqrt(w_c[1:]))
print("sum of the ICAR draw:", round(u_icar.sum(), 10) + 0.0)
u_shift = u_icar + 3.0
P_c = np.eye(n) - np.ones((n, n)) / n
print("quadratic form unchanged by adding 3:",
      np.isclose(u_icar @ (D - A) @ u_icar, u_shift @ (D - A) @ u_shift))
print("projection removes the shift:", np.allclose(P_c @ u_shift, u_icar))
Lp_c = np.linalg.pinv(D - A)
print("ICAR variances by number of neighbors:",
      {int(k): round(float(np.diag(Lp_c)[d == k].mean()), 3)
       for k in (2, 3, 4)})
```

Adding 3 to every region leaves the quadratic form untouched, so the data cannot distinguish $\mathbf{u}$ from $\mathbf{u} + 3\cdot\mathbf{1}$. That is why an intrinsic effect and an intercept cannot both be free: in a model $y_i = \beta_0 + u_i + \dots$, adding $c$ to every $u_i$ and subtracting $c$ from $\beta_0$ gives the identical model. The sum-to-zero constraint settles the matter by assigning the overall level to $\beta_0$. Chapter 9 uses it exactly so. The last line of the code also shows how the constrained ICAR variances behave on the county map: 1.401, 0.916, and 0.593 for regions with 2, 3, and 4 neighbors, the same edge effect seen for the proper CAR in Section 8.2.4.

### 8.3.5 Why the Pseudo-Inverse and the Generalized Determinant (Optional)

*This subsection proves the three facts of Section 8.3.4 for a connected map. It can be skipped on a first reading.*

Write the eigendecomposition $\mathbf{D} - \mathbf{A} = \sum_{k=2}^n\xi_k\mathbf{v}_k\mathbf{v}_k'$, where the $k = 1$ term vanishes because $\xi_1 = 0$. The vectors $\mathbf{v}_1, \dots, \mathbf{v}_n$ are an orthonormal basis of $\mathbb{R}^n$, with $\mathbf{v}_1 = \mathbf{1}/\sqrt{n}$. A vector $\mathbf{u}$ on the constraint subspace has $\mathbf{u}'\mathbf{v}_1 = 0$, so it is described by the $n - 1$ coordinates $t_k = \mathbf{v}_k'\mathbf{u}$ for $k \ge 2$, with $\mathbf{u} = \sum_{k \ge 2}t_k\mathbf{v}_k$ and $\mathbf{u}'\mathbf{u} = \sum_{k \ge 2}t_k^2$. In these coordinates

$$
\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \sum_{k \ge 2}\xi_kt_k^2
$$

so the density $\propto \exp(-\frac{1}{2\sigma^2}\sum_{k \ge 2}\xi_kt_k^2)$ factors into independent normal densities for the $t_k$, with $t_k \sim N(0, \sigma^2/\xi_k)$. Each of these densities is proper, with normalizing constant $(\xi_k/(2\pi\sigma^2))^{1/2}$. This establishes the three facts.

- **Simulation.** Draw $t_k = \sigma z_k/\sqrt{\xi_k}$ and set $\mathbf{u} = \sum_{k \ge 2}t_k\mathbf{v}_k$.
- **Covariance.** Since the $t_k$ are independent, $\operatorname{Var}(\mathbf{u}) = \sum_{k \ge 2}\operatorname{Var}(t_k)\mathbf{v}_k\mathbf{v}_k' = \sigma^2\sum_{k \ge 2}\xi_k^{-1}\mathbf{v}_k\mathbf{v}_k' = \sigma^2(\mathbf{D} - \mathbf{A})^{+}$.
- **Normalizing constant.** The joint density of $(t_2, \dots, t_n)$ is the product

  $$
  \prod_{k \ge 2}\left(\frac{\xi_k}{2\pi\sigma^2}\right)^{1/2}\exp\!\left(-\frac{\xi_kt_k^2}{2\sigma^2}\right) = (2\pi\sigma^2)^{-(n-1)/2}\lvert\mathbf{D} - \mathbf{A}\rvert_*^{1/2}\exp\!\left(-\frac{\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u}}{2\sigma^2}\right)
  $$

  with $\lvert\mathbf{D} - \mathbf{A}\rvert_* = \prod_{k \ge 2}\xi_k$. This is the log-likelihood contribution that replaces $\tfrac{1}{2}\log\lvert\mathbf{Q}\rvert$ for the intrinsic model.

For the alternative formula for the pseudo-inverse, observe that $\mathbf{D} - \mathbf{A} + \tfrac{1}{n}\mathbf{1}\mathbf{1}' = \sum_{k \ge 2}\xi_k\mathbf{v}_k\mathbf{v}_k' + \mathbf{v}_1\mathbf{v}_1'$ has eigenvalues $1, \xi_2, \dots, \xi_n$ on the same eigenvectors, so its inverse is $\mathbf{v}_1\mathbf{v}_1' + \sum_{k \ge 2}\xi_k^{-1}\mathbf{v}_k\mathbf{v}_k'$, and subtracting $\mathbf{v}_1\mathbf{v}_1' = \tfrac{1}{n}\mathbf{1}\mathbf{1}'$ leaves the pseudo-inverse. $\square$

---

## 8.4 Computing With Precision Matrices

### 8.4.1 The CAR Likelihood and Its Log-Determinant

Now fit the model to data. For the model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$ with a proper CAR effect $\mathbf{u} \sim N(\mathbf{0}, \sigma^2\mathbf{M}^{-1})$, where $\mathbf{M} = \mathbf{D} - \rho\mathbf{A}$, the log-likelihood follows from the normal density of Section 0.4.3. Since $\boldsymbol{\Sigma}^{-1} = \mathbf{M}/\sigma^2$ and $\log\lvert\sigma^2\mathbf{M}^{-1}\rvert = n\log\sigma^2 - \log\lvert\mathbf{M}\rvert$,

$$
\ell(\boldsymbol{\beta}, \sigma^2, \rho) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 + \frac{1}{2}\log\lvert\mathbf{M}\rvert - \frac{1}{2\sigma^2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})'\mathbf{M}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

Two features set this apart from the geostatistical likelihood of Chapter 5. The quadratic form uses the sparse matrix $\mathbf{M}$ directly, with no inversion. And $\log\lvert\mathbf{M}\rvert$ has a closed form in the eigenvalues of $\tilde{\mathbf{A}}$. Using $\mathbf{M} = \mathbf{D}^{1/2}(\mathbf{I}_n - \rho\tilde{\mathbf{A}})\mathbf{D}^{1/2}$ from Section 8.2.3, and the fact that the determinant of a product is the product of determinants,

$$
\log\lvert\mathbf{D} - \rho\mathbf{A}\rvert = \log\lvert\mathbf{D}\rvert + \log\lvert\mathbf{I}_n - \rho\tilde{\mathbf{A}}\rvert = \sum_{i=1}^n\log d_i + \sum_{k=1}^n\log\!\left(1 - \rho\tilde{\lambda}_k\right)
$$

since the eigenvalues of $\mathbf{I}_n - \rho\tilde{\mathbf{A}}$ are $1 - \rho\tilde{\lambda}_k$. This is the CAR version of Ord's identity from Chapter 7. The two are closely related: $\mathbf{W} = \mathbf{D}^{-1}\mathbf{A}$ equals $\mathbf{D}^{-1/2}\tilde{\mathbf{A}}\mathbf{D}^{1/2}$, a similarity transformation, so $\mathbf{W}$ and $\tilde{\mathbf{A}}$ have the same eigenvalues and one eigendecomposition serves both models. The decomposition costs on the order of $n^3$ operations, once. Each later evaluation of the determinant, for a new $\rho$, costs $n$ operations.

For fixed $\rho$ the likelihood can be maximized over $\boldsymbol{\beta}$ and $\sigma^2$ in closed form, exactly as in Section 5.2.2:

$$
\hat{\boldsymbol{\beta}}(\rho) = (\mathbf{X}'\mathbf{M}\mathbf{X})^{-1}\mathbf{X}'\mathbf{M}\mathbf{y}, \qquad \hat{\sigma}^2(\rho) = \frac{\mathrm{RSS}(\rho)}{n}, \qquad \mathrm{RSS}(\rho) = (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})'\mathbf{M}(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})
$$

Substituting back, the term $\mathrm{RSS}/(2\hat{\sigma}^2)$ equals $n/2$, and the **profile log-likelihood** of $\rho$ is

$$
\ell_p(\rho) = -\frac{n}{2}\left[\log(2\pi) + \log\hat{\sigma}^2(\rho) + 1\right] + \frac{1}{2}\sum_{i}\log d_i + \frac{1}{2}\sum_{k}\log(1 - \rho\tilde{\lambda}_k)
$$

A search over the single parameter $\rho \in (1/\tilde{\lambda}_{\min}, 1)$ finishes the job. One more saving: because $\mathbf{M} = \mathbf{D} - \rho\mathbf{A}$ is linear in $\rho$, the three ingredients $\mathbf{X}'\mathbf{M}\mathbf{X}$, $\mathbf{X}'\mathbf{M}\mathbf{y}$, and $\mathbf{y}'\mathbf{M}\mathbf{y}$ are each a fixed term minus $\rho$ times another fixed term, for instance $\mathbf{X}'\mathbf{M}\mathbf{X} = \mathbf{X}'\mathbf{D}\mathbf{X} - \rho\mathbf{X}'\mathbf{A}\mathbf{X}$. The code computes the fixed terms once per data set (`car_stats`), after which every value of $\rho$ costs only a small $p \times p$ solve. The check at the end compares the formula with a brute-force evaluation of the multivariate normal log-density at three values of $\rho$.

This is maximum likelihood. As in Chapter 5, ML underestimates variance parameters in small samples, and REML would add the term $-\tfrac{1}{2}\log\lvert\mathbf{X}'\mathbf{M}\mathbf{X}\rvert$ and use $n - p$ in place of $n$ (Exercise 10).

```python
def car_stats(y, X, A, d):
    """Sums of squares that do not depend on rho (computed once)."""
    AX, Ay = A @ X, A @ y
    return dict(XDX=X.T @ (d[:, None] * X), XAX=X.T @ AX,
                XDy=X.T @ (d * y), XAy=X.T @ Ay,
                yDy=y @ (d * y), yAy=y @ Ay, n=len(y), logd=np.log(d).sum())


def car_profile(rho, st, lam):
    """Profile log-likelihood of rho; also returns beta-hat and sigma2-hat."""
    XMX = st["XDX"] - rho * st["XAX"]
    XMy = st["XDy"] - rho * st["XAy"]
    beta = np.linalg.solve(XMX, XMy)
    rss = st["yDy"] - rho * st["yAy"] - beta @ XMy
    n = st["n"]
    s2 = rss / n
    logdet = st["logd"] + np.log1p(-rho * lam).sum()
    ll = -0.5 * n * (np.log(2 * np.pi) + np.log(s2) + 1) + 0.5 * logdet
    return ll, beta, s2


rng_chk = np.random.default_rng(5)
y_chk = rng_chk.standard_normal(n)
X_chk = np.column_stack([np.ones(n), x])
st_chk = car_stats(y_chk, X_chk, A, d)
for rho in (0.3, 0.8, 0.95):
    ll, beta, s2 = car_profile(rho, st_chk, lam)
    cov = s2 * np.linalg.inv(D - rho * A)
    brute = stats.multivariate_normal(X_chk @ beta, cov).logpdf(y_chk)
    print(f"rho={rho}: eigenvalue formula {ll:10.4f}   brute force "
          f"{brute:10.4f}")
```

The two columns agree to four decimals at every value of $\rho$, which confirms the eigenvalue formula.

### 8.4.2 Simulating From a Precision Matrix

Chapter 2 simulated a spatial field as $\mathbf{u} = \mathbf{L}\mathbf{z}$ with $\mathbf{L}\mathbf{L}' = \boldsymbol{\Sigma}$, because the variance of $\mathbf{L}\mathbf{z}$ is $\mathbf{L}\mathbf{L}'$. When the model is given through $\mathbf{Q}$ instead, the same idea works with one change. Factor the precision matrix as $\mathbf{Q} = \mathbf{L}\mathbf{L}'$ (Cholesky, with $\mathbf{L}$ lower triangular), draw $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$, and **solve** $\mathbf{L}'\mathbf{u} = \mathbf{z}$, that is, $\mathbf{u} = (\mathbf{L}')^{-1}\mathbf{z}$. By Tool 1,

$$
\operatorname{Var}(\mathbf{u}) = (\mathbf{L}')^{-1}\mathbf{I}_n\left((\mathbf{L}')^{-1}\right)' = (\mathbf{L}')^{-1}\mathbf{L}^{-1} = (\mathbf{L}\mathbf{L}')^{-1} = \mathbf{Q}^{-1}
$$

Multiplying by $\mathbf{L}$ instead would produce a vector with variance $\mathbf{Q}$, the wrong matrix. The solve is a triangular back-substitution, which costs about $n^2$ operations for a dense $\mathbf{L}$ and far less when $\mathbf{L}$ is sparse.

Why prefer this to factoring $\boldsymbol{\Sigma}$? Because $\mathbf{Q}$ is sparse and $\boldsymbol{\Sigma}$ is not, which Section 8.4.3 exploits. The code checks the method on the four-region map by drawing 100,000 vectors and comparing their sample covariance with $\boldsymbol{\Sigma}$, then shows the wrong way for contrast.

```python
rng = np.random.default_rng(2060)


def sample_from_precision(Q, size, rng):
    """Draw `size` vectors from N(0, Q^{-1}) using Q = L L'."""
    L = np.linalg.cholesky(Q)
    Z = rng.standard_normal((Q.shape[0], size))
    return linalg.solve_triangular(L.T, Z, lower=False)


draws = sample_from_precision(Q4, 100_000, rng)
print("sample covariance:\n", np.round(np.cov(draws), 3))
print("target Sigma:\n", np.round(Sigma4, 3))
wrong = np.linalg.cholesky(Q4) @ rng.standard_normal((4, 100_000))
print("multiplying by L instead gives covariance close to Q:\n",
      np.round(np.cov(wrong), 2))
```

The sample covariance matches $\boldsymbol{\Sigma}$ to within simulation error (for example, 0.412 against 0.414 for region 2, and 1.102 against 1.103 for region 1). Multiplying by $\mathbf{L}$ gives a covariance close to $\mathbf{Q}$ itself, with diagonal entries near 1, 3, 2, 2 and off-diagonal entries near $-0.5$, as predicted.

### 8.4.3 Sparse Storage and Sparse Solves

The practical payoff of precision matrices is size. Consider a slide of spatial transcriptomics, with gene expression measured at spots arranged on a hexagonal grid so that each spot touches six others. A capture area holds a few thousand spots; the code builds a grid of $78 \times 64 = 4{,}992$. A dense $4{,}992 \times 4{,}992$ matrix has almost 25 million entries, about 200 MB in double precision. But the CAR precision matrix has nonzero entries only on the diagonal and for neighboring pairs: 34,378 of them, 0.14% of the entries.

A **sparse matrix** format stores only the nonzero entries and where they are. `scipy.sparse` provides several. In compressed sparse column (CSC) form it keeps three arrays: the nonzero values, their row indices, and the position where each column starts (Appendix B.3). The precision matrix of this slide takes about 0.43 MB instead of 199 MB.

The code also shows what sparsity does to the cost of a linear solve. SciPy has no sparse Cholesky factorization; `scipy.sparse.linalg.splu` computes a sparse LU factorization with a fill-reducing column ordering, which serves the same purpose for a positive definite matrix. We time the solution of $\mathbf{Q}\mathbf{x} = \mathbf{b}$ by a dense solve and by `splu`, on square grids of 400, 1,600, and 3,600 regions.

```python
import time


def hex_spots(rows, cols):
    """Sparse adjacency of a hexagonal grid of spots (odd rows shifted)."""
    ii, jj = [], []
    for r in range(rows):
        for c in range(cols):
            nbrs = [(r, c - 1), (r, c + 1)]
            cs = (c, c + 1) if r % 2 else (c - 1, c)
            for rr in (r - 1, r + 1):
                nbrs += [(rr, cs[0]), (rr, cs[1])]
            for rr, cc in nbrs:
                if 0 <= rr < rows and 0 <= cc < cols:
                    ii.append(r * cols + c)
                    jj.append(rr * cols + cc)
    n_sp = rows * cols
    return sparse.csr_matrix((np.ones(len(ii)), (ii, jj)),
                             shape=(n_sp, n_sp))


def rook_sparse(m):
    """Sparse rook adjacency of an m-by-m grid, built from a path graph."""
    P = sparse.diags([1.0, 1.0], [-1, 1], shape=(m, m))
    I = sparse.identity(m)
    return (sparse.kron(I, P) + sparse.kron(P, I)).tocsr()


A_hex = hex_spots(78, 64)
n_hex = A_hex.shape[0]
d_hex = np.asarray(A_hex.sum(axis=1)).ravel()
print("spots:", n_hex, " symmetric:", (A_hex != A_hex.T).nnz == 0)
print("spots with 6 neighbors:", int((d_hex == 6).sum()),
      " fewest neighbors:", int(d_hex.min()))
Q_hex = (sparse.diags(d_hex) - 0.9 * A_hex).tocsc()
dense_mb = n_hex**2 * 8 / 1e6
sparse_mb = (Q_hex.data.nbytes + Q_hex.indices.nbytes
             + Q_hex.indptr.nbytes) / 1e6
print(f"nonzeros: {Q_hex.nnz} of {n_hex**2} "
      f"({100 * Q_hex.nnz / n_hex**2:.2f}%)")
print(f"memory: dense {dense_mb:.0f} MB, sparse {sparse_mb:.2f} MB")

print("  grid     n   dense solve  sparse solve")
for m in (20, 40, 60):
    A_m = rook_sparse(m)
    Q_m = (sparse.diags(np.asarray(A_m.sum(axis=1)).ravel())
           - 0.9 * A_m).tocsc()
    b = rng.standard_normal(m * m)
    t0 = time.perf_counter()
    x_dense = np.linalg.solve(Q_m.toarray(), b)
    t_dense = time.perf_counter() - t0
    t0 = time.perf_counter()
    x_sparse = spla.splu(Q_m).solve(b)
    t_sparse = time.perf_counter() - t0
    print(f"{m:3d} x {m:<3d} {m * m:5d}   {t_dense:8.3f} s   "
          f"{t_sparse:8.4f} s   (differ by "
          f"{np.abs(x_dense - x_sparse).max():.0e})")
```

In the run shown, the sparse solve is about 3 times faster at 400 regions, about 30 times faster at 1,600, and about 120 times faster at 3,600, and the two methods agree to rounding error. Going from 400 to 3,600 regions, the dense time grew several hundred-fold while the sparse time grew by a factor of about 8. The times depend on the machine and change from run to run; the pattern, with the gap widening as the map grows, does not. A dense solve costs on the order of $n^3$ operations. A sparse solve exploits the fact that, with a sensible ordering, a region's factor entries stay close to its neighbors, so the work and the storage grow much more slowly.

**A sampler that needs no Cholesky factor.** For large maps we want to draw from $N(\mathbf{0}, \mathbf{Q}^{-1})$ using only what SciPy provides. A trick based on the incidence matrix of Section 8.3.2 does it. Recall $\mathbf{D} - \mathbf{A} = \mathbf{O}'\mathbf{O}$. The proper CAR matrix is a sum of two terms of that form, $\mathbf{D} - \rho\mathbf{A} = (1 - \rho)\mathbf{D} + \rho\mathbf{O}'\mathbf{O}$, so

$$
\mathbf{D} - \rho\mathbf{A} = \mathbf{N}'\mathbf{N}, \qquad \mathbf{N} = \begin{bmatrix}\sqrt{1 - \rho}\,\mathbf{D}^{1/2} \\ \sqrt{\rho}\,\mathbf{O}\end{bmatrix}
$$

where $\mathbf{N}$ stacks the diagonal matrix on top of the incidence matrix, and so has $n + m$ rows for $m$ neighbor pairs. Draw $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_{n+m})$ and set $\mathbf{u} = \sigma(\mathbf{D} - \rho\mathbf{A})^{-1}\mathbf{N}'\mathbf{z}$, which is a sparse solve. Its variance is

$$
\sigma^2(\mathbf{D} - \rho\mathbf{A})^{-1}\mathbf{N}'\mathbf{N}(\mathbf{D} - \rho\mathbf{A})^{-1} = \sigma^2(\mathbf{D} - \rho\mathbf{A})^{-1}
$$

as required. The functions below build $\mathbf{O}$ and use this sampler. We verify the factorization $\mathbf{N}'\mathbf{N} = \mathbf{D} - 0.9\mathbf{A}$ and the covariance of the draws on the county map, then draw a field on the spot grid.

```python
def incidence(A):
    """Edge-by-region matrix O with O'O = D - A, for a symmetric A."""
    ei, ej = sparse.triu(A, k=1).nonzero()
    m = len(ei)
    rows = np.repeat(np.arange(m), 2)
    cols = np.column_stack([ei, ej]).ravel()
    vals = np.tile([1.0, -1.0], m)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(m, A.shape[0]))


def sample_car_sparse(A, rho, sigma2, size, rng):
    """Draw from the proper CAR with sparse tools only (Q = N'N)."""
    d = np.asarray(A.sum(axis=1)).ravel()
    N = sparse.vstack([sparse.diags(np.sqrt((1 - rho) * d)),
                       np.sqrt(rho) * incidence(A)]).tocsr()
    Q = (N.T @ N).tocsc()
    Z = rng.standard_normal((N.shape[0], size))
    return np.sqrt(sigma2) * spla.splu(Q).solve(np.asarray(N.T @ Z))


A_sp = sparse.csr_matrix(A)
N_chk = sparse.vstack([sparse.diags(np.sqrt(0.1 * d)),
                       np.sqrt(0.9) * incidence(A_sp)])
print("N'N equals D - 0.9 A:",
      abs((N_chk.T @ N_chk) - sparse.csr_matrix(D - 0.9 * A)).max() < 1e-12)
draws = sample_car_sparse(A_sp, 0.9, 2.0, 50_000, rng)
Sigma_true = 2.0 * np.linalg.inv(D - 0.9 * A)
print("largest difference between sample and exact covariance:",
      round(np.abs(np.cov(draws) - Sigma_true).max(), 3),
      " (exact variances up to", round(Sigma_true.diagonal().max(), 3), ")")
t0 = time.perf_counter()
field = sample_car_sparse(A_hex, 0.9, 2.0, 1, rng)
print(f"one field on the {n_hex} spots: "
      f"{time.perf_counter() - t0:.2f} s, shape {field.shape}")
```

The factorization holds to machine precision. The sample covariance of 50,000 draws is within 0.022 of the exact covariance, whose largest variance is 1.652. A field on all 4,992 spots takes a fraction of a second.

---

## 8.5 SAR Versus CAR

### 8.5.1 Simultaneous Versus Conditional

Chapter 7 defined the SAR model through the equation $\mathbf{u} = \rho\mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}$ with independent errors of variance $\sigma^2$, which gives $\mathbf{Q}_{\text{SAR}} = \sigma^{-2}(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W})$. The CAR model gives $\mathbf{Q}_{\text{CAR}} = \sigma^{-2}(\mathbf{D} - \rho\mathbf{A})$. With the row-standardized weights $\mathbf{W} = \mathbf{D}^{-1}\mathbf{A}$ the two can be written side by side, since $\mathbf{D} - \rho\mathbf{A} = \mathbf{D}(\mathbf{I}_n - \rho\mathbf{W})$:

$$
\mathbf{Q}_{\text{CAR}} = \sigma^{-2}\,\mathbf{D}\,(\mathbf{I}_n - \rho\mathbf{W}), \qquad \mathbf{Q}_{\text{SAR}} = \sigma^{-2}\,(\mathbf{I}_n - \rho\mathbf{W})'\,(\mathbf{I}_n - \rho\mathbf{W})
$$

The two differ in the left factor: the CAR puts the diagonal matrix $\mathbf{D}$ there, and the SAR puts the transpose of the right factor. The CAR is symmetric because $\mathbf{D}(\mathbf{I}_n - \rho\mathbf{W}) = \mathbf{D} - \rho\mathbf{A}$ is. The SAR is symmetric by construction.

Every SAR model is a CAR model in the sense of Section 8.1: any symmetric positive definite $\mathbf{Q}$ has conditional distributions with $c_{ij} = -Q_{ij}/Q_{ii}$ and $\sigma_i^2 = 1/Q_{ii}$. The question is what those conditionals look like. The code computes them for the center region of a $3 \times 3$ map under a SAR with $\rho = 0.9$. A CAR conditional mean would involve the four neighbors only. The SAR conditional mean involves all eight other regions: weight 0.386 on each of the four neighbors, and weight $-0.132$ on each of the four diagonal regions, which are neighbors of neighbors. So the SAR is a CAR with a richer neighborhood in which second-order neighbors count, with negative weights.

### 8.5.2 Sparsity: Zeros in $\mathbf{Q}$ Versus Zeros in $\boldsymbol{\Sigma}$

The product $(\mathbf{I}_n - \rho\mathbf{W})'(\mathbf{I}_n - \rho\mathbf{W}) = \mathbf{I}_n - \rho(\mathbf{W} + \mathbf{W}') + \rho^2\mathbf{W}'\mathbf{W}$ contains the term $\mathbf{W}'\mathbf{W}$, whose $(i, j)$ entry is nonzero whenever regions $i$ and $j$ share a common neighbor. So $\mathbf{Q}_{\text{SAR}}$ is nonzero for neighbors and for neighbors of neighbors, while $\mathbf{Q}_{\text{CAR}}$ is nonzero only for neighbors. The code prints the pattern of both matrices on a $3 \times 3$ map, with regions numbered row by row, and counts nonzeros on the county map.

```python
A3 = rook_sparse(3).toarray()
d3 = A3.sum(axis=1)
W3 = A3 / d3[:, None]
rho = 0.9
IW3 = np.eye(9) - rho * W3
Qcar3 = np.diag(d3) - rho * A3
Qsar3 = IW3.T @ IW3


def show_pattern(Q):
    for row in Q:
        print("  " + " ".join("x" if abs(v) > 1e-12 else "." for v in row))


print("3 x 3 map, regions numbered row by row. Q_CAR:")
show_pattern(Qcar3)
print("Q_SAR:")
show_pattern(Qsar3)
k = 4
others = [j for j in range(9) if j != k]
cond_w = -Qsar3[k, others] / Qsar3[k, k]
print("SAR center region: conditional weights on regions", others)
print(np.round(cond_w, 3), " conditional variance", round(1 / Qsar3[k, k], 3))

W = A / d[:, None]
I_n = np.eye(n)
print("Q_CAR = D (I - rho W):",
      np.allclose(D - 0.9 * A, D @ (I_n - 0.9 * W)))
Qc = D - 0.9 * A
Qs = (I_n - 0.9 * W).T @ (I_n - 0.9 * W)
tol = 1e-12
print("nonzeros on the 12 x 12 map: Q_CAR", int((abs(Qc) > tol).sum()),
      " Q_SAR", int((abs(Qs) > tol).sum()),
      " Sigma_CAR", int((abs(np.linalg.inv(Qc)) > tol).sum()),
      " Sigma_SAR", int((abs(np.linalg.inv(Qs)) > tol).sum()),
      " of", n * n)
```

On the $3 \times 3$ map, $\mathbf{Q}_{\text{CAR}}$ is zero for regions that do not touch, while $\mathbf{Q}_{\text{SAR}}$ fills in the second-order neighbors. The center region's row has 5 nonzero entries in $\mathbf{Q}_{\text{CAR}}$ (itself and its four neighbors) and 9 in $\mathbf{Q}_{\text{SAR}}$. On the county map $\mathbf{Q}_{\text{CAR}}$ has 672 nonzero entries, which is $144 + 2 \times 264$: the diagonal plus both directions of each of the 264 neighbor pairs. $\mathbf{Q}_{\text{SAR}}$ has 1,636, almost two and a half times as many. Both are sparse, but the CAR matrix is sparser, and the gap grows with the number of neighbors per region. The covariance matrices, in contrast, are completely dense for both models: all 20,736 entries are nonzero. A connected map always makes the covariance matrix dense, for either model (Section 8.1.4). Figure 8.3 in Section 8.6.3 draws the three patterns.

### 8.5.3 Why $\rho$ Is Not Comparable

Both models have a parameter called $\rho$, and both live on $(-1, 1)$ for a rectangular grid. They do not measure the same thing. The code computes the average correlation between neighboring regions on the county map as a function of $\rho$, for each model, and finds the SAR value of $\rho$ that matches the CAR model at $\rho = 0.9$.

```python
pi, pj = np.nonzero(np.triu(A))


def mean_neighbor_corr(Sig):
    s = np.sqrt(np.diag(Sig))
    return (Sig[pi, pj] / (s[pi] * s[pj])).mean()


def car_cov(rho):
    return np.linalg.inv(D - rho * A)


def sar_cov(rho):
    M = I_n - rho * W
    return np.linalg.inv(M.T @ M)


print("   rho   CAR neighbor corr   SAR neighbor corr")
for rho in (0.5, 0.8, 0.9, 0.95, 0.99):
    print(f"  {rho:4.2f}   {mean_neighbor_corr(car_cov(rho)):12.3f}"
          f"   {mean_neighbor_corr(sar_cov(rho)):14.3f}")
target = mean_neighbor_corr(car_cov(0.9))
rho_sar = optimize.brentq(
    lambda r: mean_neighbor_corr(sar_cov(r)) - target, 0.01, 0.99)
print(f"SAR rho matching CAR rho = 0.9: {rho_sar:.3f}")
```

| $\rho$ | CAR neighbor correlation | SAR neighbor correlation |
| :--- | :--- | :--- |
| 0.50 | 0.149 | 0.302 |
| 0.80 | 0.294 | 0.593 |
| 0.90 | 0.388 | 0.753 |
| 0.95 | 0.472 | 0.865 |
| 0.99 | 0.652 | 0.980 |

At the same $\rho$ the models imply very different dependence. At $\rho = 0.9$, the CAR neighbors have correlation 0.388, and the SAR neighbors have 0.753. To get the CAR's neighbor correlation of 0.388 from a SAR, one needs $\rho \approx 0.608$. Two lessons follow. First, a fitted $\hat{\rho}$ should never be compared across the two model families or interpreted as a correlation. Second, the CAR needs a $\rho$ very close to 1 to produce strong dependence: even at $\rho = 0.99$ the neighbor correlation is only 0.652. Estimates of $\rho$ near 1 are therefore the norm in applications, not a sign of trouble.

---

## 8.6 Walkthrough and Lab 8

### 8.6.1 Simulating a CAR Field and Recovering Its Parameters

Simulate unexplained variation in a county health outcome. On the county map, let $y_i = \beta_0 + \beta_1x_i + u_i$, with $x_i$ the standardized deprivation index of the shared data set, $\beta_0 = 1$, $\beta_1 = 0.5$, and $\mathbf{u}$ a proper CAR field with $\sigma^2 = 2$ and $\rho = 0.95$. The data are simulated, so the truth is known. We fit the model by profile maximum likelihood (Section 8.4.1), test $\rho = 0$ (independent regions with variances $\sigma^2/d_i$) by a likelihood ratio, and compare the slope from OLS with the slope from the CAR model.

```python
X = np.column_stack([np.ones(n), x])
beta_true = np.array([1.0, 0.5])
sigma2_true, rho_true = 2.0, 0.95
rng = np.random.default_rng(2060)
Q_true = (D - rho_true * A) / sigma2_true
u = sample_from_precision(Q_true, 1, rng)[:, 0]
y = X @ beta_true + u


def fit_car(y, X, A, d, lam, lo=-0.99, hi=0.999):
    """ML fit of y = X beta + u with a proper CAR u, by profiling rho."""
    st = car_stats(y, X, A, d)
    res = optimize.minimize_scalar(
        lambda r: -car_profile(r, st, lam)[0],
        bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    ll, beta, s2 = car_profile(res.x, st, lam)
    return res.x, s2, beta, ll


rho_hat, s2_hat, beta_hat, ll_hat = fit_car(y, X, A, d, lam)
ll_zero = car_profile(0.0, car_stats(y, X, A, d), lam)[0]
print(f"rho_hat = {rho_hat:.3f} (true {rho_true}),  "
      f"sigma2_hat = {s2_hat:.3f} (true {sigma2_true})")
print("beta_hat =", np.round(beta_hat, 3), " (true", beta_true, ")")
print(f"log-likelihood {ll_hat:.2f} at rho_hat, {ll_zero:.2f} at rho = 0; "
      f"LRT = {2 * (ll_hat - ll_zero):.1f}")

b_ols = np.linalg.solve(X.T @ X, X.T @ y)
r_ols = y - X @ b_ols
se_ols = np.sqrt(r_ols @ r_ols / (n - 2) * np.linalg.inv(X.T @ X)[1, 1])
M_hat = D - rho_hat * A
cov_gls = s2_hat * np.linalg.inv(X.T @ M_hat @ X)
se_gls = np.sqrt(cov_gls[1, 1])
print(f"slope: OLS {b_ols[1]:.3f} (SE {se_ols:.3f}),  "
      f"CAR-GLS {beta_hat[1]:.3f} (SE {se_gls:.3f}),  true 0.5")
print(f"intercept: CAR-GLS {beta_hat[0]:.3f} "
      f"(SE {np.sqrt(cov_gls[0, 0]):.3f}),  true 1.0")
```

The fit recovers the dependence: $\hat{\rho} = 0.971$ (true 0.95) and $\hat{\sigma}^2 = 1.832$ (true 2). The likelihood ratio statistic against $\rho = 0$ is 75.3 on one parameter, overwhelming evidence of spatial dependence. The slope is 0.456 under the CAR model and 0.354 under OLS, against a true 0.5. The standard errors are close (0.087 and 0.090). The intercept is the poorly determined quantity: its standard error is 0.349, four times the slope's 0.087, and the estimate $\hat{\beta}_0 = 1.444$ is 1.3 standard errors above the true value of 1. This is the identifiability problem of Section 8.3.4 in a nearly intrinsic model, where $\rho$ is close to 1: when the spatial effect can absorb almost any overall level, the intercept is only weakly pinned down by the data.

### 8.6.2 How Well Are the Parameters Recovered?

One data set proves little, so repeat the experiment. The code draws 500 data sets from the same model, using the Cholesky factor of $\mathbf{Q}$ and the solve of Section 8.4.2, and fits each by the same method.

```python
reps = 500
Lq = np.linalg.cholesky(Q_true)
U = linalg.solve_triangular(Lq.T, rng.standard_normal((n, reps)),
                            lower=False)
rho_hats = np.empty(reps)
s2_hats = np.empty(reps)
slopes = np.empty(reps)
for k in range(reps):
    r_k, s2_k, b_k, _ = fit_car(X @ beta_true + U[:, k], X, A, d, lam)
    rho_hats[k], s2_hats[k], slopes[k] = r_k, s2_k, b_k[1]
q = np.percentile(rho_hats, [5, 25, 50, 75, 95])
print("rho_hat quantiles (5, 25, 50, 75, 95%):", np.round(q, 3))
print(f"sigma2_hat: mean {s2_hats.mean():.3f}, "
      f"sd {s2_hats.std():.3f} (true {sigma2_true})")
print(f"share of rho_hat above 0.99: {(rho_hats > 0.99).mean():.3f};  "
      f"below 0.5: {(rho_hats < 0.5).mean():.3f}")
print(f"slope: mean {slopes.mean():.3f}, sd {slopes.std():.3f}")
```

The slope and $\sigma^2$ are estimated without noticeable bias: the average of $\hat{\sigma}^2$ is 2.028 (true 2) and the average slope is 0.500 (true 0.5). The parameter $\rho$ is not. Its median estimate is 0.918 against a true 0.95, with a long tail to the left: the middle 50% of the estimates run from 0.862 to 0.953, and the 5th percentile is 0.743. The distribution is also capped by the boundary at 1; only 0.4% of the estimates exceed 0.99. So with 144 regions, $\rho$ is estimated with a bias toward smaller values and with a wide spread, and values of 0.9 and 0.98 would be hard to tell apart. Figure 8.4 draws both the profile for the data set of Section 8.6.1 and the distribution of the 500 estimates.

### 8.6.3 Figures

The last block draws four figures and saves them to `figures/ch08/`; run it from the top folder of the repository.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/ch08", exist_ok=True)
side = 12
z = np.random.default_rng(2060).standard_normal(n)
fields = []
for rho in (0.0, 0.5, 0.9, 0.99):
    Lr = np.linalg.cholesky(D - rho * A)
    fields.append(linalg.solve_triangular(Lr.T, z, lower=False))
fields.append(u_icar)
titles = ["rho = 0", "rho = 0.5", "rho = 0.9", "rho = 0.99", "ICAR"]
fig, axes = plt.subplots(1, 5, figsize=(14, 3.3))
for ax, f, title in zip(axes, fields, titles):
    lim = np.abs(f).max()
    ax.imshow(f.reshape(side, side), origin="lower", cmap="RdBu_r",
              vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig("figures/ch08/fig_8_1_car_fields.png", dpi=120)

Sigma9 = car_cov(0.9)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, vals, title in zip(axes, (1 / d, np.diag(Sigma9)),
                           ("conditional variance 1/d_i",
                            "marginal variance, rho = 0.9")):
    im = ax.imshow(vals.reshape(side, side), origin="lower",
                   cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("figures/ch08/fig_8_2_variances.png", dpi=120)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, M, title in zip(
        axes, (Qc, Qs, np.linalg.inv(Qc)),
        ("Q_CAR (672 nonzeros)", "Q_SAR (1,636 nonzeros)",
         "Sigma_CAR (all 20,736)")):
    ax.spy(abs(M) > 1e-12, markersize=1)
    ax.set_title(title, y=1.08)
fig.tight_layout()
fig.savefig("figures/ch08/fig_8_3_sparsity.png", dpi=120)

grid_rho = np.linspace(0.3, 0.995, 200)
st = car_stats(y, X, A, d)
prof = np.array([car_profile(r, st, lam)[0] for r in grid_rho])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].plot(grid_rho, prof - prof.max())
axes[0].set_xlim(0.3, 1.0)
axes[0].axvline(rho_true, color="k", linestyle=":", label="true")
axes[0].axvline(rho_hat, color="C1", linestyle="--", label="maximum")
axes[0].set_ylim(-10, 0.5)
axes[0].set_xlabel("rho")
axes[0].set_ylabel("profile log-likelihood (relative to maximum)")
axes[0].legend()
axes[1].hist(rho_hats, bins=25, color="C0")
axes[1].axvline(rho_true, color="k", linestyle=":")
axes[1].set_xlabel("estimated rho in 500 simulated data sets")
axes[1].set_ylabel("count")
fig.tight_layout()
fig.savefig("figures/ch08/fig_8_4_profile.png", dpi=120)
for r0 in (0.5, 0.7, 0.9):
    j = np.argmin(np.abs(grid_rho - r0))
    print(f"profile at rho = {r0}: {prof[j] - prof.max():7.2f} "
          "relative to the maximum")
print("figures saved")
```

![Figure 8.1. Simulated fields on the 144-county map, all from the same standard normal draw, for the proper CAR with $\rho = 0, 0.5, 0.9, 0.99$ and for the intrinsic CAR. Each panel has its own color scale, centered at zero. As $\rho$ rises, neighboring counties become more alike and the patches grow.](../figures/ch08/fig_8_1_car_fields.png)

In Figure 8.1, the field at $\rho = 0$ is speckled: neighboring counties are unrelated, and the extreme values sit at the corners and edges, where counties have fewer neighbors and so larger variances. At $\rho = 0.5$ the picture is much the same, which fits the neighbor correlation of only 0.15 at that value (Section 8.5.3). At 0.9 and 0.99 the field organizes into broad patches of high values on the left and in the center and low values toward the upper right. The intrinsic field varies on the broadest scale of all, is centered at zero because its values sum to zero, and shows its largest values along the edges and corners, the edge effect once more.

![Figure 8.2. Variance by county on the $12 \times 12$ map. Left: the conditional variance $1/d_i$ (with $\sigma^2 = 1$), which depends only on the number of neighbors. Right: the marginal variance at $\rho = 0.9$.](../figures/ch08/fig_8_2_variances.png)

Figure 8.2 shows the edge effect of Section 8.2.4. Conditional variance takes only three values: 0.5 at the four corners, 0.33 along the edges, and 0.25 in the interior. The marginal variance at $\rho = 0.9$ follows the same pattern but smoothly, rising toward the edges and highest in the corners (0.826 on average for the corners, against 0.379 for interior regions, as the output of the code in Section 8.2.4 shows).

![Figure 8.3. Nonzero patterns on the 144-county map, regions numbered row by row. $\mathbf{Q}_{\text{CAR}}$ has 672 nonzero entries on three bands, $\mathbf{Q}_{\text{SAR}}$ has 1,636 on five bands, and the covariance matrix $\boldsymbol{\Sigma}_{\text{CAR}}$ is completely dense.](../figures/ch08/fig_8_3_sparsity.png)

Figure 8.3 is the picture behind Section 8.5.2. Each region in a row-by-row numbering touches the regions one step away on the same row and 12 steps away in the next row, giving the three bands of $\mathbf{Q}_{\text{CAR}}$. The SAR precision adds the second-order neighbors, giving five bands. The covariance is a solid block.

![Figure 8.4. Left: profile log-likelihood of $\rho$ for the simulated data of Section 8.6.1, relative to its maximum. Right: estimates of $\rho$ from 500 simulated data sets with true $\rho = 0.95$.](../figures/ch08/fig_8_4_profile.png)

Figure 8.4 summarizes Sections 8.6.1 and 8.6.2. The profile peaks near 0.97 and falls steeply to the left: relative to its maximum it is lower by 1.5 log-likelihood units at $\rho = 0.9$, by 9.4 at $\rho = 0.7$, and by 17.8 at $\rho = 0.5$. The sharp peak against the boundary is what a map that is close to intrinsic looks like. The histogram shows the skewed distribution of $\hat{\rho}$ with its mass piled up against the boundary at 1.

---

## Exercises

**Analytical**

1. Using the two-by-two block of $\mathbf{Q}$ for regions $i$ and $j$ and the same completing-the-square argument as in Section 8.1.2, show that the conditional covariance of $u_i$ and $u_j$ given all the other regions is $-Q_{ij}/(Q_{ii}Q_{jj} - Q_{ij}^2)$. Conclude that it is zero exactly when $Q_{ij} = 0$.

2. Redo Brook's lemma of Section 8.1.3 for two regions. Find the symmetry requirement and the joint precision matrix in terms of $c_{12}$, $c_{21}$, $\sigma_1^2$, and $\sigma_2^2$. For which values of $c_{12}$ and $c_{21}$ is the matrix positive definite when $\sigma_1^2 = \sigma_2^2 = 1$ and the symmetry requirement holds?

3. For the three-region example of Section 8.1.3, compute $\boldsymbol{\Sigma} = \mathbf{Q}^{-1}$ by hand. Verify that the marginal variances exceed the conditional variances, and find the covariance of regions 1 and 3, which are not neighbors.

4. Take a path of three regions with degrees 1, 2, 1. Find the eigenvalues of $\tilde{\mathbf{A}}$ and the valid range of $\rho$. For $\rho = 0.5$ and $\sigma^2 = 1$, compute $\mathbf{Q}$ and $\boldsymbol{\Sigma}$ by hand and identify which region has the largest marginal variance.

5. Show that the $12 \times 12$ rook grid is bipartite (color the regions like a checkerboard). Let $\mathbf{s}$ be the vector with $+1$ on black squares and $-1$ on white squares. Show that $\mathbf{A}\mathbf{s} = -\mathbf{d} \circ \mathbf{s}$ (entry by entry), and that $\mathbf{D}^{1/2}\mathbf{s}$ is an eigenvector of $\tilde{\mathbf{A}}$ with eigenvalue $-1$. Conclude that the valid range for $\rho$ on this map is $(-1, 1)$.

6. Show that for the proper CAR the conditional mean of $u_i$ is $\rho$ times the average of its neighbors, starting from $\mathbf{Q} = \sigma^{-2}(\mathbf{D} - \rho\mathbf{A})$ and the formulas of Section 8.1.2. What happens to the conditional mean when $\rho = 0$?

7. On a path of $n$ regions, show that $\mathbf{u}'(\mathbf{D} - \mathbf{A})\mathbf{u} = \sum_{i=1}^{n-1}(u_{i+1} - u_i)^2$. Write the incidence matrix $\mathbf{O}$ for $n = 4$, compute $\mathbf{O}'\mathbf{O}$, and check that it equals $\mathbf{D} - \mathbf{A}$. Explain why the intrinsic CAR on a path is the random walk of Section 2.2.3.

8. Prove the sampler of Section 8.4.3: that $\mathbf{N}'\mathbf{N} = \mathbf{D} - \rho\mathbf{A}$ and that $\sigma(\mathbf{D} - \rho\mathbf{A})^{-1}\mathbf{N}'\mathbf{z}$ has covariance $\sigma^2(\mathbf{D} - \rho\mathbf{A})^{-1}$. Why does the construction fail at $\rho = 1$, and what would you change to simulate the intrinsic model?

9. Verify Kirchhoff's matrix-tree theorem for (a) a triangle, (b) a path of four regions, and (c) a cycle of four regions: compute the Laplacian eigenvalues, their nonzero product, and the number of spanning trees, and check that the product equals $n$ times the number of trees.

10. Derive the REML profile log-likelihood for the proper CAR model with covariates, by adding $-\tfrac{1}{2}\log\lvert\mathbf{X}'\mathbf{M}\mathbf{X}\rvert$ to the ML log-likelihood and replacing $n$ by $n - p$ in the variance estimate (Section 5.3.3). What is the REML estimate of $\sigma^2$ in terms of $\mathrm{RSS}(\rho)$?

11. With the county map numbered row by row, the matrix $\mathbf{Q}_{\text{CAR}}$ has bandwidth 12: every nonzero entry satisfies $\lvert i - j\rvert \le 12$. Cholesky factors of banded matrices stay inside the band. Count the largest possible number of nonzero entries in the lower-triangular factor $\mathbf{L}$ and compare with the $n(n+1)/2$ entries of a dense lower-triangular factor of $\boldsymbol{\Sigma}$.

12. Apply Brook's lemma of Section 8.1.3 to four regions arranged in a cycle $1 - 2 - 3 - 4 - 1$ with reference point $\mathbf{0}$.
(a) Evaluate the density ratio $\log[p(\mathbf{u})/p(\mathbf{0})]$ along the coordinate path $(0,0,0,0) \to (u_1,0,0,0) \to (u_1,u_2,0,0) \to (u_1,u_2,u_3,0) \to (u_1,u_2,u_3,u_4)$, showing that it produces cross terms $u_1u_2$, $u_2u_3$, $u_3u_4$, and $u_4u_1$.
(b) Repeat along the reverse path $(u_1, u_2, u_3, u_4) \to \dots \to (0,0,0,0)$ and equate the cross terms to verify the pairwise symmetry condition $c_{ij}\sigma_j^2 = c_{ji}\sigma_i^2$.
(c) Explain why Brook's lemma requires the **positivity condition** ($p(\mathbf{u}) > 0$ for all $\mathbf{u}$ in the product space of supports). What goes wrong if certain neighbor combinations are forbidden (for example, in a hard-core spatial process where adjacent binary sites cannot both take value 1)?

13. In Section 8.2.4 we saw that the proper CAR model makes marginal variances heteroscedastic across the map because the conditional variance shrinks with degree $d_i$. The **Leroux model** [@leroux2000], which Chapter 9 uses for spatial count data, replaces the degree-weighted ridge penalty with an unweighted ridge penalty:
$$
\mathbf{Q}_{\text{Leroux}} = \sigma^{-2}\left[(1 - \lambda)\mathbf{I}_n + \lambda(\mathbf{D} - \mathbf{A})\right], \qquad 0 \le \lambda < 1
$$
(a) Using the eigenvalues of the graph Laplacian $\mathbf{D} - \mathbf{A}$, show that $\mathbf{Q}_{\text{Leroux}}$ is positive definite for every $\lambda \in [0, 1)$ without requiring any map-dependent lower bound on $\lambda$.
(b) Derive the full conditional mean $\operatorname{E}[u_i \mid \mathbf{u}_{-i}]$ and conditional variance $\operatorname{Var}(u_i \mid \mathbf{u}_{-i})$ for region $i$. Show that as $\lambda \to 0$ the model reduces to independent noise with constant variance $\sigma^2$, and as $\lambda \to 1$ it reduces to the intrinsic CAR.
(c) Contrast the conditional variance of region $i$ in the Leroux model with that in the proper CAR model. Does the Leroux conditional variance shrink as quickly with $d_i$?

14. Section 8.1.2 showed that the CAR conditional expectation is the simple kriging prediction of $u_i$ from all other regions.
(a) For a region $i$ on a rook grid, let $\partial i$ be its set of immediate neighbors and let $\mathcal{S}$ be any subset of regions disjoint from $\partial i \cup \{i\}$. Using the precision matrix $\mathbf{Q}$, prove that the simple kriging prediction of $u_i$ from $\mathbf{u}_{\partial i} \cup \mathbf{u}_{\mathcal{S}}$ places exactly zero weight on all regions in $\mathcal{S}$ (**exact screening**).
(b) Suppose instead that $\mathbf{u}$ is generated by a continuous Gaussian process with an exponential covariance function $C(h) = \sigma^2\exp(-h/\phi)$, observed at the centroids of the grid. Explain why the inverse covariance matrix $\boldsymbol{\Sigma}^{-1}$ is dense, and why simple kriging weights on second-order neighbors do not vanish in general.

15. In Section 8.2.4 we observed that marginal variances $\Sigma_{ii} = [(\mathbf{D} - \rho\mathbf{A})^{-1}]_{ii}$ vary across the map even at $\rho = 0$.
(a) Write $\mathbf{D} - \rho\mathbf{A} = \mathbf{D}^{1/2}(\mathbf{I}_n - \rho\tilde{\mathbf{A}})\mathbf{D}^{1/2}$ and use the Neumann series to expand $(\mathbf{D} - \rho\mathbf{A})^{-1} = \mathbf{D}^{-1/2}(\mathbf{I}_n + \rho\tilde{\mathbf{A}} + \rho^2\tilde{\mathbf{A}}^2 + \dots)\mathbf{D}^{-1/2}$.
(b) Show that the zero-order term in the marginal variance $\Sigma_{ii}$ is $1/d_i$, and that the first-order term in $\rho$ is zero on the diagonal.
(c) Show that the second-order term on the diagonal is $\rho^2 \sum_{j \sim i} (d_i d_j)^{-1}$.
(d) Explain why corner regions (with $d_i = 2$) have larger marginal variances than interior regions (with $d_i = 4$) at both zero and higher orders of $\rho$.

**Computational and simulation**

16. Repeat the experiment of Section 8.6.2 with true $\rho = 0.5$ and with $\rho = 0.8$. Plot the distribution of $\hat{\rho}$ for each. How does the bias and the spread change as $\rho$ moves away from 1?

17. Implement the REML version from Exercise 10 and compare ML and REML estimates of $\sigma^2$ on a $6 \times 6$ map with 200 simulated data sets and $\rho = 0.9$, $\sigma^2 = 2$. How large is the downward bias of ML?

18. On the hexagonal spot grid of Section 8.4.3, draw 200 fields from the proper CAR with $\rho = 0.9$ using the sparse sampler. Estimate the average correlation between neighboring spots and between spots two steps apart, and compare with the same quantities on the square grid.

19. Build a map with two connected components by removing a row of regions from the $12 \times 12$ grid. Simulate the intrinsic CAR on it by the spectral method with sum-to-zero constraints within each component, and check that both component sums are zero. What happens to the sum of all regions if you impose only a single global constraint?

20. For the hexagonal grid, compare the neighbor correlations of CAR and SAR at $\rho = 0.5, 0.9, 0.99$, and find the SAR $\rho$ that matches CAR at 0.9. Is the answer close to the 0.608 found on the square grid?

21. Solve $\mathbf{Q}\mathbf{x} = \mathbf{b}$ on a $100 \times 100$ grid for $\rho = 0.9$, $0.99$, and $0.9999$ using `splu` and using the conjugate gradient method `scipy.sparse.linalg.cg`. Count the iterations of conjugate gradient and explain, using the smallest eigenvalue of $\mathbf{Q}$, why they grow as $\rho \to 1$.

22. Section 8.4.2 simulated CAR fields by solving a triangular system from the Cholesky factor of $\mathbf{Q}$. An alternative is the **Gibbs sampler**, which updates each region in turn from its full conditional distribution:
$$
u_i^{(t+1)} \sim N\!\left(\rho\,\frac{1}{d_i}\sum_{j \sim i}u_j^{(t)},\ \frac{\sigma^2}{d_i}\right)
$$
(a) Implement single-site Gibbs sampling on the four-region map with $\rho = 0.5$ and $\sigma^2 = 1$. Run 20,000 iterations and compare the sample covariance of the draws with the exact covariance matrix $\boldsymbol{\Sigma}_4$ of Section 8.2.2.
(b) Run the sampler on the $12 \times 12$ county map for $\rho = 0.5$ and for $\rho = 0.95$. For an interior county, plot the autocorrelation function of the chain up to lag 50. Why does the Gibbs sampler mix rapidly at $\rho = 0.5$ but exhibit slow mixing at $\rho = 0.95$?

23. The performance of sparse linear solvers depends on the ordering of regions.
(a) For a $30 \times 30$ grid of square counties ($n = 900$), construct the CAR precision matrix $\mathbf{Q}$ with $\rho = 0.9$ under the default row-by-row numbering, and under a randomly permuted numbering.
(b) For both matrices, compute the symbolic LU factorization with `scipy.sparse.linalg.splu` without automatic column permutation (`permc_spec='NATURAL'`), and compare the number of nonzeros in the factors (the fill-in).
(c) Time the solution of $\mathbf{Q}\mathbf{x} = \mathbf{b}$ for both orderings and explain why graph ordering is essential when scaling GMRFs to tens of thousands of regions.

---

## What Comes Next

This chapter defined CAR models through conditional regressions on neighbors, showed that they are Gaussian Markov random fields with sparse precision matrices, and developed the proper and intrinsic versions with the tools to fit and simulate them. Chapter 9 puts these models to work. It replaces the normal response with a count, so that a county's death count is Poisson with a log-rate that includes the spatial random effect $\mathbf{u}$. The effect $\mathbf{u}$ may be an ICAR, a proper CAR, or the blend of ICAR and independent noise known as the BYM model, and Chapter 9 compares these choices. The sparse precision matrices, the sum-to-zero constraints, and the eigenvalue determinant of this chapter are the machinery behind each.
