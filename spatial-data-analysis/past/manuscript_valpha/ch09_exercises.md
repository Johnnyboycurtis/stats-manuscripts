# Chapter 9 Companion Exercises: Non-Gaussian and Nonlinear Spatial Estimation

This study guide and exercise workbook accompanies [Chapter 9: Non-Gaussian and Nonlinear Spatial Estimation](file:///Users/jonathannavarrete/Projects/semantic-search-models/manuscript/ch09_nonlinear_geostatistics.md). Synthesizing Maria L. Rizzo's three-beat pedagogical rhythm (*Statistical Computing with R*) with Oliver Schabenberger and Carol A. Gotway's semantic linear-model discipline (*Statistical Methods for Spatial Data Analysis*), these exercises bridge classical probability and spatial statistics across five comprehensive tiers:

1. **Tier 0: Classical-to-Spatial Warm-Ups** (Exercises 9.1–9.4)
2. **Tier I: Core Theoretical Re-Derivations** (Exercises 9.5–9.8)
3. **Tier II: Pencil-and-Paper Mechanics** (Exercises 9.9–9.11)
4. **Tier III: Algorithmic Translation & Labs in Pure Python** (Exercises 9.12–9.13)
5. **Tier IV: Edge-Case Diagnostics & Conceptual Reflections** (Exercises 9.14–9.16)

Every exercise includes an unabridged, step-by-step mathematical derivation, analytical proof, or dependency-free computational solution.

---

## Tier 0: Classical-to-Spatial Warm-Ups

### Exercise 9.1: Classical Delta-Method vs. Exact Lognormal Mean MGF Back-Transformation
In classical statistical inference, when an attribute of interest $Z$ is positive and right-skewed, practitioners routinely apply the natural logarithmic transformation $Y = \ln(Z)$ to stabilize variance and achieve approximate normality: $Y \sim \mathcal{N}(\mu, \sigma^2)$. Once statistical estimation or linear prediction is executed on the Gaussian scale $Y$, the analyst must back-transform to the original physical scale $Z = g(Y) = \exp(Y)$.

**(a)** Using the definition of the Gaussian Moment-Generating Function (MGF) $M_Y(t) = \operatorname{E}[\exp(tY)] = \int_{-\infty}^\infty \exp(ty) \phi(y; \mu, \sigma^2) \mathrm{d}y$, complete the square in the exponent to prove that:
$$
M_Y(t) = \exp\left(\mu t + \frac{1}{2}\sigma^2 t^2\right)
$$
Evaluate $M_Y(t)$ at $t = 1$ to derive the exact arithmetic mean of the physical variable:
$$
\operatorname{E}[Z] = \operatorname{E}[\exp(Y)] = \exp\left(\mu + \frac{1}{2}\sigma^2\right)
$$

**(b)** The classical delta-method approximates the expectation of a non-linear transformation $g(Y)$ via a second-order Taylor series expansion about the mean $\mu = \operatorname{E}[Y]$:
$$
g(Y) \approx g(\mu) + g'(\mu)(Y - \mu) + \frac{1}{2}g''(\mu)(Y - \mu)^2
$$
For $g(Y) = \exp(Y)$:
1. Show that the delta-method approximation to $\operatorname{E}[g(Y)]$ is:
$$
\operatorname{E}_{\text{delta}}[Z] = \exp(\mu)\left(1 + \frac{1}{2}\sigma^2\right)
$$
2. Identify the connection between $\operatorname{E}_{\text{delta}}[Z]$ and the Taylor series of the exponential function $\exp(\frac{1}{2}\sigma^2)$.
3. Compare the exact back-transformation factor $\exp(\frac{1}{2}\sigma^2)$ against the delta-method approximation factor $(1 + \frac{1}{2}\sigma^2)$ for log-variances $\sigma^2 \in \{0.1, 0.5, 1.0, 2.0\}$. Compute the percentage relative error in each case.
4. Explain why the delta-method fails catastrophically for highly skewed spatial logs (e.g., gold grades or trace contaminant plumes with high nugget effects or high sills).

**(c)** Derive the exact variance of $Z$:
$$
\operatorname{Var}(Z) = \exp(2\mu + \sigma^2)\left(\exp(\sigma^2) - 1\right)
$$
Discuss how Jensen's inequality and log-variance distortion motivate moving from linear kriging on transformed variables to non-linear geostatistical approaches (such as Indicator Kriging and Disjunctive Kriging).

##### Solution to Exercise 9.1
**(a)** By definition, the Moment-Generating Function of $Y \sim \mathcal{N}(\mu, \sigma^2)$ is:
$$
M_Y(t) = \operatorname{E}[\exp(tY)] = \int_{-\infty}^\infty \exp(ty) \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y - \mu)^2}{2\sigma^2}\right) \mathrm{d}y
$$
Combine the exponents:
$$
t y - \frac{(y - \mu)^2}{2\sigma^2} = -\frac{1}{2\sigma^2}\left[(y - \mu)^2 - 2\sigma^2 t y\right]
$$
Expand the quadratic term:
$$
(y - \mu)^2 - 2\sigma^2 t y = y^2 - 2\mu y + \mu^2 - 2\sigma^2 t y = y^2 - 2(\mu + \sigma^2 t)y + \mu^2
$$
Complete the square with respect to $y$:
$$
y^2 - 2(\mu + \sigma^2 t)y + \mu^2 = \left[y - (\mu + \sigma^2 t)\right]^2 - (\mu + \sigma^2 t)^2 + \mu^2
$$
Expand $-(\mu + \sigma^2 t)^2 + \mu^2$:
$$
-\left(\mu^2 + 2\mu\sigma^2 t + \sigma^4 t^2\right) + \mu^2 = -2\mu\sigma^2 t - \sigma^4 t^2
$$
Substitute this back into the exponent:
$$
-\frac{1}{2\sigma^2}\left\{\left[y - (\mu + \sigma^2 t)\right]^2 - 2\mu\sigma^2 t - \sigma^4 t^2\right\} = -\frac{\left[y - (\mu + \sigma^2 t)\right]^2}{2\sigma^2} + \mu t + \frac{1}{2}\sigma^2 t^2
$$
The terms independent of $y$ factor out of the integral:
$$
M_Y(t) = \exp\left(\mu t + \frac{1}{2}\sigma^2 t^2\right) \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\left[y - (\mu + \sigma^2 t)\right]^2}{2\sigma^2}\right) \mathrm{d}y
$$
The integrand is the probability density function of a Gaussian random variable $\mathcal{N}(\mu + \sigma^2 t, \sigma^2)$, which integrates to 1 over $\mathbb{R}$. Thus:
$$
M_Y(t) = \exp\left(\mu t + \frac{1}{2}\sigma^2 t^2\right)
$$
Setting $t = 1$:
$$
\operatorname{E}[Z] = \operatorname{E}[\exp(Y)] = M_Y(1) = \exp\left(\mu(1) + \frac{1}{2}\sigma^2(1)^2\right) = \exp\left(\mu + \frac{1}{2}\sigma^2\right)
$$

**(b)** 
1. For $g(y) = \exp(y)$, the derivatives evaluated at $\mu$ are:
$$
g(\mu) = \exp(\mu), \quad g'(\mu) = \exp(\mu), \quad g''(\mu) = \exp(\mu)
$$
The second-order Taylor series approximation around $\mu$ is:
$$
g(Y) \approx \exp(\mu) + \exp(\mu)(Y - \mu) + \frac{1}{2}\exp(\mu)(Y - \mu)^2
$$
Taking expectation across the expansion:
$$
\begin{aligned}
\operatorname{E}[g(Y)] &\approx \operatorname{E}\left[\exp(\mu) + \exp(\mu)(Y - \mu) + \frac{1}{2}\exp(\mu)(Y - \mu)^2\right] \\
&= \exp(\mu) + \exp(\mu)\operatorname{E}[Y - \mu] + \frac{1}{2}\exp(\mu)\operatorname{E}\left[(Y - \mu)^2\right]
\end{aligned}
$$
Since $\operatorname{E}[Y - \mu] = 0$ and $\operatorname{E}[(Y - \mu)^2] = \operatorname{Var}(Y) = \sigma^2$:
$$
\operatorname{E}_{\text{delta}}[Z] = \exp(\mu)\left(1 + 0 + \frac{1}{2}\sigma^2\right) = \exp(\mu)\left(1 + \frac{1}{2}\sigma^2\right)
$$
2. The exact expectation is $\operatorname{E}[Z] = \exp(\mu)\exp(\frac{1}{2}\sigma^2)$.
The Taylor series expansion of $\exp(u)$ about $u = 0$ is:
$$
\exp(u) = 1 + u + \frac{u^2}{2!} + \frac{u^3}{3!} + \dots = \sum_{k=0}^\infty \frac{u^k}{k!}
$$
Setting $u = \frac{1}{2}\sigma^2$:
$$
\exp\left(\frac{1}{2}\sigma^2\right) = 1 + \frac{1}{2}\sigma^2 + \frac{1}{8}\sigma^4 + \frac{1}{48}\sigma^6 + \dots
$$
Therefore, the delta-method approximation $\exp(\mu)(1 + \frac{1}{2}\sigma^2)$ is identical to retaining only the first two terms ($k=0$ and $k=1$) in the Maclaurin expansion of the exact lognormal factor $\exp(\frac{1}{2}\sigma^2)$, discarding all higher powers $(\frac{1}{2}\sigma^2)^k / k!$ for $k \ge 2$.
3. We compare the exact multiplier $\exp(\frac{1}{2}\sigma^2)$ against the delta multiplier $(1 + \frac{1}{2}\sigma^2)$ for each variance:
- For $\sigma^2 = 0.1$:
  - Exact: $\exp(0.05) \approx 1.051271$
  - Delta: $1 + 0.05 = 1.050000$
  - Relative Error: $\frac{1.050000 - 1.051271}{1.051271} \times 100\% \approx -0.12\%$
- For $\sigma^2 = 0.5$:
  - Exact: $\exp(0.25) \approx 1.284025$
  - Delta: $1 + 0.25 = 1.250000$
  - Relative Error: $\frac{1.250000 - 1.284025}{1.284025} \times 100\% \approx -2.65\%$
- For $\sigma^2 = 1.0$:
  - Exact: $\exp(0.50) \approx 1.648721$
  - Delta: $1 + 0.50 = 1.500000$
  - Relative Error: $\frac{1.500000 - 1.648721}{1.648721} \times 100\% \approx -9.02\%$
- For $\sigma^2 = 2.0$:
  - Exact: $\exp(1.00) \approx 2.718282$
  - Delta: $1 + 1.00 = 2.000000$
  - Relative Error: $\frac{2.000000 - 2.718282}{2.718282} \times 100\% \approx -26.43\%$
4. Failure mode for highly skewed spatial logs:
When spatial logs exhibit substantial variability—such as $\sigma^2 \ge 1.0$ or $2.0$, common in precious metals (gold grades often exhibit $\sigma^2 > 3.0$) and environmental toxicology—the quadratic truncation error explodes. The neglected terms $\sum_{k=2}^\infty (\frac{1}{2}\sigma^2)^k / k!$ dominate the series. In spatial kriging, as the unsampled point moves away from sample points, the prediction variance $\sigma_{\text{SK}}^2(\mathbf{s}_0)$ approaches the field sill $\sigma^2$. The delta-method underestimates the required back-transformation correction factor by over $26\%$ at $\sigma^2 = 2.0$ (and by over $55\%$ at $\sigma^2 = 3.0$), causing drastic underestimation of regional resource tonnage and contaminant mass.

**(c)** 
1. Exact variance derivation:
The second moment of $Z = \exp(Y)$ is obtained by setting $t = 2$ in the MGF:
$$
\operatorname{E}[Z^2] = \operatorname{E}[\exp(2Y)] = M_Y(2) = \exp\left(\mu(2) + \frac{1}{2}\sigma^2(2)^2\right) = \exp(2\mu + 2\sigma^2)
$$
Using $\operatorname{Var}(Z) = \operatorname{E}[Z^2] - (\operatorname{E}[Z])^2$:
$$
\begin{aligned}
\operatorname{Var}(Z) &= \exp(2\mu + 2\sigma^2) - \left[\exp\left(\mu + \frac{1}{2}\sigma^2\right)\right]^2 \\
&= \exp(2\mu + 2\sigma^2) - \exp(2\mu + \sigma^2) \\
&= \exp(2\mu + \sigma^2)\left(\exp(\sigma^2) - 1\right)
\end{aligned}
$$
2. Motivation for non-linear geostatistics:
- **Jensen's Inequality:** Because $g(y) = e^y$ is strictly convex ($g''(y) = e^y > 0$), Jensen's inequality guarantees that $\operatorname{E}[\exp(Y)] > \exp(\operatorname{E}[Y])$. The naive predictor $\exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))$ systematically underestimates the conditional expectation by the factor $\exp(-\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0))$.
- **Log-Variance Distortion:** The back-transformation correction term $\exp(\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0))$ depends directly on the local kriging error variance. In data-sparse zones or near the boundaries of the domain, $\sigma_{\text{SK}}^2(\mathbf{s}_0)$ approaches the sill, causing exponential inflation of predicted values. A single large sample in a sparse cluster can artificially elevate back-transformed predictions across kilometers.
- Moreover, because $\operatorname{Var}(Z \mid \mathbf{Y}) = \widehat{Z}_{\text{LK}}(\mathbf{s}_0)^2 (\exp(\sigma_{\text{SK}}^2(\mathbf{s}_0)) - 1)$ scales with the *square* of the predicted value, predictive uncertainty becomes violently heteroscedastic.
- These pathologies demonstrate that linear kriging on log-transformed variables cannot resolve non-linear questions (e.g., probability of exceeding regulatory thresholds or evaluating recoverable metal above cutoffs) without extreme distributional sensitivity. This motivated Georges Matheron and André Journel to formulate non-parametric alternatives: Indicator Kriging (which models the cumulative distribution function directly at cutoffs without transformations) and Disjunctive Kriging (which models non-linear operators via stable orthogonal polynomial expansions). $\blacksquare$

---

### Exercise 9.2: Sheppard's Arcsine Formula and Median-Cutoff Indicator Autocorrelation
In spatial indicator geostatistics, continuous attributes $Z(\mathbf{s})$ are transformed into binary indicators $I(\mathbf{s}; z_c) = \mathbb{I}(Z(\mathbf{s}) \le z_c)$ at chosen cutoffs $z_c$. When the underlying process is Gaussian (or has been mapped to standard Gaussian scores $Y(\mathbf{s}) \sim \mathcal{N}(0, 1)$), the spatial correlation between indicator variables is linked to the Gaussian correlation $\rho = \operatorname{Corr}(Y(\mathbf{s}_1), Y(\mathbf{s}_2))$ by classical bivariate normal integration.

**(a)** Let $(Y_1, Y_2)'$ have a standard bivariate normal distribution with correlation coefficient $\rho \in [-1, 1]$ and joint density:
$$
\phi_\rho(y_1, y_2) = \frac{1}{2\pi \sqrt{1 - \rho^2}} \exp\left(-\frac{y_1^2 - 2\rho y_1 y_2 + y_2^2}{2(1 - \rho^2)}\right)
$$
Consider the median indicator variables $I_1 = \mathbb{I}(Y_1 \le 0)$ and $I_2 = \mathbb{I}(Y_2 \le 0)$ corresponding to cutoff $y_c = 0$.
Prove William Fleetwood Sheppard's theorem (1899):
$$
P(Y_1 \le 0, \, Y_2 \le 0) = \frac{1}{4} + \frac{1}{2\pi}\arcsin(\rho)
$$
using Plackett's differential identity $\frac{\partial \phi_\rho}{\partial \rho} = \frac{\partial^2 \phi_\rho}{\partial y_1 \partial y_2}$.

**(b)** Show that $\operatorname{E}[I_1] = \operatorname{E}[I_2] = \frac{1}{2}$ and $\operatorname{Var}(I_1) = \operatorname{Var}(I_2) = \frac{1}{4}$.

**(c)** Derive the median indicator correlation coefficient $\rho_I = \operatorname{Corr}(I_1, I_2)$ in terms of $\rho$:
$$
\rho_I = \frac{2}{\pi}\arcsin(\rho)
$$

**(d)** Using the Taylor series expansion $\arcsin(\rho) = \rho + \frac{1}{6}\rho^3 + \frac{3}{40}\rho^5 + \mathcal{O}(\rho^7)$:
1. Show that for small correlation $\rho \ll 1$, $\rho_I \approx \frac{2}{\pi}\rho \approx 0.6366 \rho$.
2. Prove that $|\rho_I| \le |\rho|$ for all $\rho \in [-1, 1]$, with equality holding only at $\rho \in \{-1, 0, 1\}$.
3. Explain why empirical indicator semivariograms $\gamma_I(\mathbf{h}) = 1 - \rho_I(\mathbf{h})$ always exhibit lower spatial continuity (higher apparent nugget effect and faster spatial decay) than the underlying Gaussian field $Y(\mathbf{s})$.

##### Solution to Exercise 9.2
**(a)** Define the joint cumulative distribution function:
$$
P(\rho) = P(Y_1 \le 0, \, Y_2 \le 0) = \int_{-\infty}^0 \int_{-\infty}^0 \phi_\rho(y_1, y_2) \, \mathrm{d}y_1 \, \mathrm{d}y_2
$$
Step 1: Verify Plackett's identity $\frac{\partial \phi_\rho}{\partial \rho} = \frac{\partial^2 \phi_\rho}{\partial y_1 \partial y_2}$.
Recall the characteristic function of $(Y_1, Y_2)' \sim \mathcal{N}_2(\mathbf{0}, \boldsymbol{\Sigma})$ with $\boldsymbol{\Sigma} = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}$:
$$
\psi(t_1, t_2) = \operatorname{E}\left[\exp(i(t_1 Y_1 + t_2 Y_2))\right] = \exp\left(-\frac{1}{2}(t_1^2 + 2\rho t_1 t_2 + t_2^2)\right)
$$
Differentiating with respect to $\rho$:
$$
\frac{\partial \psi}{\partial \rho} = -t_1 t_2 \exp\left(-\frac{1}{2}(t_1^2 + 2\rho t_1 t_2 + t_2^2)\right) = (-it_1)(-it_2)\psi(t_1, t_2)
$$
By the properties of Fourier transforms, multiplication by $(-it_1)(-it_2)$ in frequency domain corresponds to $\frac{\partial^2}{\partial y_1 \partial y_2}$ in spatial domain:
$$
\frac{\partial \phi_\rho(y_1, y_2)}{\partial \rho} = \frac{\partial^2 \phi_\rho(y_1, y_2)}{\partial y_1 \partial y_2}
$$
Step 2: Differentiate $P(\rho)$ with respect to $\rho$ by Leibniz's rule:
$$
\frac{\mathrm{d}P(\rho)}{\mathrm{d}\rho} = \int_{-\infty}^0 \int_{-\infty}^0 \frac{\partial \phi_\rho(y_1, y_2)}{\partial \rho} \, \mathrm{d}y_1 \, \mathrm{d}y_2 = \int_{-\infty}^0 \int_{-\infty}^0 \frac{\partial^2 \phi_\rho(y_1, y_2)}{\partial y_1 \partial y_2} \, \mathrm{d}y_1 \, \mathrm{d}y_2
$$
Integrate with respect to $y_2$ and $y_1$ using the Fundamental Theorem of Calculus:
$$
\int_{-\infty}^0 \frac{\partial}{\partial y_2}\left[ \frac{\partial \phi_\rho(y_1, y_2)}{\partial y_1} \right] \mathrm{d}y_2 = \frac{\partial \phi_\rho(y_1, 0)}{\partial y_1} - \lim_{y_2 \to -\infty} \frac{\partial \phi_\rho(y_1, y_2)}{\partial y_1} = \frac{\partial \phi_\rho(y_1, 0)}{\partial y_1} - 0
$$
Now integrate with respect to $y_1$:
$$
\int_{-\infty}^0 \frac{\partial \phi_\rho(y_1, 0)}{\partial y_1} \, \mathrm{d}y_1 = \phi_\rho(0, 0) - \lim_{y_1 \to -\infty} \phi_\rho(y_1, 0) = \phi_\rho(0, 0)
$$
Step 3: Evaluate $\phi_\rho(0, 0)$:
$$
\phi_\rho(0, 0) = \frac{1}{2\pi \sqrt{1 - \rho^2}} \exp(0) = \frac{1}{2\pi \sqrt{1 - \rho^2}}
$$
Step 4: Integrate from 0 to $\rho$:
$$
P(\rho) - P(0) = \int_0^\rho \frac{\mathrm{d}P(r)}{\mathrm{d}r} \, \mathrm{d}r = \int_0^\rho \frac{1}{2\pi \sqrt{1 - r^2}} \, \mathrm{d}r = \frac{1}{2\pi} \left[\arcsin(r)\right]_0^\rho = \frac{1}{2\pi}\arcsin(\rho)
$$
When $\rho = 0$, $Y_1$ and $Y_2$ are independent standard normal variables:
$$
P(0) = P(Y_1 \le 0) P(Y_2 \le 0) = \Phi(0) \Phi(0) = \left(\frac{1}{2}\right)\left(\frac{1}{2}\right) = \frac{1}{4}
$$
Therefore:
$$
P(Y_1 \le 0, \, Y_2 \le 0) = \frac{1}{4} + \frac{1}{2\pi}\arcsin(\rho)
$$
This completes Sheppard's classical proof.

**(b)** Because $Y_1, Y_2 \sim \mathcal{N}(0, 1)$, their marginal CDF at 0 is $\Phi(0) = \frac{1}{2}$.
The expectation of the indicator is:
$$
\operatorname{E}[I_1] = P(Y_1 \le 0) = \frac{1}{2}, \quad \operatorname{E}[I_2] = P(Y_2 \le 0) = \frac{1}{2}
$$
Because $I_1^2 = I_1$ (binary variable idempotence):
$$
\operatorname{Var}(I_1) = \operatorname{E}[I_1^2] - (\operatorname{E}[I_1])^2 = \frac{1}{2} - \left(\frac{1}{2}\right)^2 = \frac{1}{4}
$$
Similarly, $\operatorname{Var}(I_2) = \frac{1}{4}$.

**(c)** The product of two indicator variables satisfies $I_1 I_2 = \mathbb{I}(Y_1 \le 0 \text{ and } Y_2 \le 0)$.
Its expectation is:
$$
\operatorname{E}[I_1 I_2] = P(Y_1 \le 0, \, Y_2 \le 0) = \frac{1}{4} + \frac{1}{2\pi}\arcsin(\rho)
$$
The covariance is:
$$
\operatorname{Cov}(I_1, I_2) = \operatorname{E}[I_1 I_2] - \operatorname{E}[I_1]\operatorname{E}[I_2] = \left(\frac{1}{4} + \frac{1}{2\pi}\arcsin(\rho)\right) - \left(\frac{1}{2}\right)\left(\frac{1}{2}\right) = \frac{1}{2\pi}\arcsin(\rho)
$$
The correlation coefficient is:
$$
\rho_I = \frac{\operatorname{Cov}(I_1, I_2)}{\sqrt{\operatorname{Var}(I_1)\operatorname{Var}(I_2)}} = \frac{\frac{1}{2\pi}\arcsin(\rho)}{\sqrt{\frac{1}{4}\cdot\frac{1}{4}}} = \frac{\frac{1}{2\pi}\arcsin(\rho)}{\frac{1}{4}} = \frac{4}{2\pi}\arcsin(\rho) = \frac{2}{\pi}\arcsin(\rho)
$$

**(d)** 
1. Substituting the Taylor series $\arcsin(\rho) = \rho + \frac{1}{6}\rho^3 + \dots$:
$$
\rho_I = \frac{2}{\pi}\left(\rho + \frac{1}{6}\rho^3 + \dots\right) \approx \frac{2}{\pi}\rho \approx 0.63662 \rho \quad \text{for } |\rho| \ll 1
$$
2. For all $\rho \in (0, 1)$, $\arcsin(\rho) < \frac{\pi}{2}\rho$ because $\arcsin(x)$ is strictly convex on $(0, 1)$ with $\arcsin(0) = 0$ and $\arcsin(1) = \frac{\pi}{2}$.
Multiplying by $\frac{2}{\pi}$ gives:
$$
\rho_I = \frac{2}{\pi}\arcsin(\rho) < \rho \quad \text{for all } \rho \in (0, 1)
$$
By odd symmetry $\arcsin(-\rho) = -\arcsin(\rho)$, $|\rho_I| \le |\rho|$ across the entire interval $[-1, 1]$, with equality if and only if $\rho \in \{-1, 0, 1\}$.
3. Spatial implications for indicator variograms:
The normalized indicator semivariogram is $\gamma_I(\mathbf{h}) = 1 - \rho_I(\mathbf{h}) = 1 - \frac{2}{\pi}\arcsin(\rho(\mathbf{h}))$.
- At short lags where the Gaussian field has high correlation (e.g., $\rho = 0.90$), $\rho_I = \frac{2}{\pi}\arcsin(0.90) \approx \frac{2}{\pi}(1.1198) \approx 0.7129$. The indicator correlation is noticeably lower than the Gaussian correlation ($0.71$ vs $0.90$).
- At moderate lags where $\rho = 0.50$, $\rho_I = \frac{2}{\pi}\arcsin(0.50) = \frac{2}{\pi}\frac{\pi}{6} = \frac{1}{3} \approx 0.3333$.
- Thresholding continuous physical measurements into binary indicator classes destroys continuous distance and gradient information. High-frequency spatial switching occurs whenever the field crosses the cutoff threshold, introducing artificial discretization noise.
- Consequently, empirical indicator variograms always exhibit shorter spatial correlation ranges, steeper slopes near the origin, and a substantially higher apparent nugget effect than the underlying continuous process. $\blacksquare$

---

### Exercise 9.3: Three-Term Hermite Polynomial Expansion of $\exp(\sigma Y)$ (S&G Problem 5.6 connection)
Disjunctive Kriging (DK) and non-linear geostatistical prediction rely on decomposing arbitrary non-linear transfer functions into orthogonal series of normalized Hermite polynomials with respect to the Gaussian weight function $\phi(y) = \frac{1}{\sqrt{2\pi}}e^{-y^2/2}$.

**(a)** State the definitions of the first three normalized Hermite polynomials $H_0(y), H_1(y), H_2(y)$ satisfying the orthonormality condition $\int_{-\infty}^\infty H_j(y) H_k(y) \phi(y) \mathrm{d}y = \delta_{jk}$:
$$
H_0(y) = 1, \quad H_1(y) = y, \quad H_2(y) = \frac{y^2 - 1}{\sqrt{2}}
$$

**(b)** Let $Y \sim \mathcal{N}(0, 1)$ and consider the lognormal transfer function $g(Y) = \exp(\sigma Y)$, where $\sigma > 0$. The Hermite expansion coefficients are defined by:
$$
\phi_k = \int_{-\infty}^\infty \exp(\sigma y) H_k(y) \phi(y) \mathrm{d}y, \quad k = 0, 1, 2, \dots
$$
By completing the square in the exponent of $\exp(\sigma y)\phi(y) = \exp(\frac{1}{2}\sigma^2)\phi(y - \sigma)$ and performing a change of variable, evaluate the integral analytically for $k = 0, 1, 2$. Show that:
$$
\phi_0 = \exp\left(\frac{1}{2}\sigma^2\right), \quad \phi_1 = \sigma \exp\left(\frac{1}{2}\sigma^2\right), \quad \phi_2 = \frac{\sigma^2}{\sqrt{2}}\exp\left(\frac{1}{2}\sigma^2\right)
$$

**(c)** Consider the three-term truncated Hermite approximation:
$$
\widehat{Z}_2 = \sum_{k=0}^2 \phi_k H_k(Y) = \phi_0 H_0(Y) + \phi_1 H_1(Y) + \phi_2 H_2(Y)
$$
1. Prove that $\operatorname{E}[\widehat{Z}_2] = \exp(\frac{1}{2}\sigma^2)$ and $\operatorname{Var}(\widehat{Z}_2) = \exp(\sigma^2)\left(\sigma^2 + \frac{1}{2}\sigma^4\right)$.
2. For $\sigma = 0.5$ ($\sigma^2 = 0.25$), calculate the numerical values of the exact mean $\operatorname{E}[Z]$ and exact variance $\operatorname{Var}(Z)$ versus the truncated expansion mean $\operatorname{E}[\widehat{Z}_2]$ and variance $\operatorname{Var}(\widehat{Z}_2)$. Show that the three-term approximation captures over $99\%$ of the true physical variance.

**(d)** Explain how Disjunctive Kriging uses this orthogonal expansion property (via Mehler's formula) to perform optimal non-linear spatial estimation without inverting separate spatial covariance matrices for different non-linear cutoffs.

##### Solution to Exercise 9.3
**(a)** Under the standard Gaussian weighting function $\phi(y) = \frac{1}{\sqrt{2\pi}}e^{-y^2/2}$, Rodrigues' formula defines normalized Hermite polynomials as:
$$
H_k(y) = \frac{(-1)^k}{\sqrt{k!}} e^{y^2/2} \frac{\mathrm{d}^k}{\mathrm{d}y^k}\left(e^{-y^2/2}\right)
$$
Evaluating the first three terms:
- For $k = 0$: $H_0(y) = \frac{1}{\sqrt{0!}} e^{y^2/2} (e^{-y^2/2}) = 1$.
- For $k = 1$: $\frac{\mathrm{d}}{\mathrm{d}y}(e^{-y^2/2}) = -y e^{-y^2/2} \implies H_1(y) = \frac{-1}{\sqrt{1}} e^{y^2/2}(-y e^{-y^2/2}) = y$.
- For $k = 2$: $\frac{\mathrm{d}^2}{\mathrm{d}y^2}(e^{-y^2/2}) = (y^2 - 1)e^{-y^2/2} \implies H_2(y) = \frac{1}{\sqrt{2}} e^{y^2/2}(y^2 - 1)e^{-y^2/2} = \frac{y^2 - 1}{\sqrt{2}}$.

**(b)** We evaluate $\phi_k = \int_{-\infty}^\infty \exp(\sigma y) H_k(y) \phi(y) \, \mathrm{d}y$.
First, complete the square in the product of the exponential and Gaussian kernel:
$$
\exp(\sigma y)\phi(y) = \exp(\sigma y) \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2 - 2\sigma y}{2}\right)
$$
Complete the square: $y^2 - 2\sigma y = (y - \sigma)^2 - \sigma^2$.
$$
\exp(\sigma y)\phi(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(y - \sigma)^2 - \sigma^2}{2}\right) = \exp\left(\frac{1}{2}\sigma^2\right) \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(y - \sigma)^2}{2}\right) = \exp\left(\frac{1}{2}\sigma^2\right) \phi(y - \sigma)
$$
Substitute this into the coefficient integral:
$$
\phi_k = \exp\left(\frac{1}{2}\sigma^2\right) \int_{-\infty}^\infty H_k(y) \phi(y - \sigma) \, \mathrm{d}y
$$
Make the change of variables $u = y - \sigma$, so $y = u + \sigma$ and $\mathrm{d}y = \mathrm{d}u$:
$$
\phi_k = \exp\left(\frac{1}{2}\sigma^2\right) \int_{-\infty}^\infty H_k(u + \sigma) \phi(u) \, \mathrm{d}u = \exp\left(\frac{1}{2}\sigma^2\right) \operatorname{E}\left[H_k(U + \sigma)\right]
$$
where $U \sim \mathcal{N}(0, 1)$, meaning $\operatorname{E}[U] = 0$, $\operatorname{E}[U^2] = 1$, and $\operatorname{E}[U^k] = 0$ for all odd $k$.
Now evaluate for $k = 0, 1, 2$:
- For $k = 0$:
  $$
  H_0(u + \sigma) = 1 \implies \operatorname{E}[H_0(U + \sigma)] = 1
  $$
  $$
  \phi_0 = \exp\left(\frac{1}{2}\sigma^2\right) \cdot 1 = \exp\left(\frac{1}{2}\sigma^2\right)
  $$
- For $k = 1$:
  $$
  H_1(u + \sigma) = u + \sigma \implies \operatorname{E}[H_1(U + \sigma)] = \operatorname{E}[U] + \sigma = 0 + \sigma = \sigma
  $$
  $$
  \phi_1 = \sigma \exp\left(\frac{1}{2}\sigma^2\right)
  $$
- For $k = 2$:
  $$
  H_2(u + \sigma) = \frac{(u + \sigma)^2 - 1}{\sqrt{2}} = \frac{u^2 + 2\sigma u + \sigma^2 - 1}{\sqrt{2}}
  $$
  Taking expectations:
  $$
  \operatorname{E}[H_2(U + \sigma)] = \frac{\operatorname{E}[U^2] + 2\sigma \operatorname{E}[U] + \sigma^2 - 1}{\sqrt{2}} = \frac{1 + 0 + \sigma^2 - 1}{\sqrt{2}} = \frac{\sigma^2}{\sqrt{2}}
  $$
  $$
  \phi_2 = \frac{\sigma^2}{\sqrt{2}}\exp\left(\frac{1}{2}\sigma^2\right)
  $$

**(c)** 
1. Mean and variance of truncated expansion $\widehat{Z}_2$:
Because $Y \sim \mathcal{N}(0, 1)$, $\operatorname{E}[H_0(Y)] = 1$, and for all $k \ge 1$, $\operatorname{E}[H_k(Y)] = \int_{-\infty}^\infty H_k(y) H_0(y) \phi(y) \, \mathrm{d}y = \delta_{k0} = 0$.
The mean of the truncated approximation is:
$$
\operatorname{E}[\widehat{Z}_2] = \phi_0 \operatorname{E}[H_0(Y)] + \phi_1 \operatorname{E}[H_1(Y)] + \phi_2 \operatorname{E}[H_2(Y)] = \phi_0 \cdot 1 + 0 + 0 = \exp\left(\frac{1}{2}\sigma^2\right)
$$
This reproduces the exact lognormal mean perfectly!
For the variance, by orthonormality $\operatorname{Cov}(H_j(Y), H_k(Y)) = \delta_{jk}$:
$$
\begin{aligned}
\operatorname{Var}(\widehat{Z}_2) &= \sum_{k=1}^2 \phi_k^2 = \phi_1^2 + \phi_2^2 \\
&= \left(\sigma \exp\left(\frac{1}{2}\sigma^2\right)\right)^2 + \left(\frac{\sigma^2}{\sqrt{2}}\exp\left(\frac{1}{2}\sigma^2\right)\right)^2 \\
&= \sigma^2 \exp(\sigma^2) + \frac{\sigma^4}{2} \exp(\sigma^2) = \exp(\sigma^2)\left(\sigma^2 + \frac{1}{2}\sigma^4\right)
\end{aligned}
$$
2. Numerical comparison for $\sigma = 0.5$ ($\sigma^2 = 0.25$):
- Exact mean:
  $$
  \operatorname{E}[Z] = \exp\left(\frac{1}{2}(0.25)\right) = \exp(0.125) \approx 1.133148
  $$
  Truncated mean: $\operatorname{E}[\widehat{Z}_2] = \exp(0.125) \approx 1.133148$ (Discrepancy: $0.0$).
- Exact variance:
  $$
  \operatorname{Var}(Z) = \exp(0.25)\left(\exp(0.25) - 1\right) \approx 1.284025 \times (1.284025 - 1) = 1.284025 \times 0.284025 \approx 0.364700
  $$
- Truncated variance:
  $$
  \begin{aligned}
  \operatorname{Var}(\widehat{Z}_2) &= \exp(0.25)\left(0.25 + \frac{1}{2}(0.25)^2\right) = 1.284025 \times (0.25 + 0.03125) \\
  &= 1.284025 \times 0.28125 \approx 0.361132
  \end{aligned}
  $$
- Variance ratio captured:
  $$
  \frac{\operatorname{Var}(\widehat{Z}_2)}{\operatorname{Var}(Z)} = \frac{0.361132}{0.364700} \approx 99.02\%
  $$
  A simple three-term Hermite truncation captures $99.02\%$ of the total physical variability while maintaining exact unbiasedness.

**(d)** In Disjunctive Kriging (DK), a target non-linear transform $g(Y)$ (e.g., indicator exceedance $\mathbb{I}(Y > y_c)$ or recovered metal $Y \mathbb{I}(Y > y_c)$) is expanded into orthogonal Hermite series:
$$
g(y) = \sum_{k=0}^K g_k H_k(y)
$$
By Mehler's formula for bivariate Gaussian fields, the spatial cross-covariance between different polynomial orders is strictly zero:
$$
\operatorname{Cov}\left(H_j(Y(\mathbf{s}_1)), \, H_k(Y(\mathbf{s}_2))\right) = \left[\rho(\mathbf{s}_1 - \mathbf{s}_2)\right]^k \delta_{jk}
$$
This mathematical orthogonality decouples the non-linear estimation problem:
1. Each polynomial component $\widehat{H}_k(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i^{(k)} H_k(Y(\mathbf{s}_i))$ is estimated by its own linear kriging system with covariance matrix $\mathbf{V}_k = [\rho(\mathbf{s}_i - \mathbf{s}_j)^k]$.
2. The spatial kriging weights $\boldsymbol{\lambda}^{(k)} = \mathbf{V}_k^{-1}\mathbf{c}_k$ depend *solely* on spatial geometry and polynomial degree $k$—they are completely independent of the physical cutoff threshold $z_c$.
3. When the engineer varies the cutoff $z_c$ (generating a continuous grade-tonnage or risk curve), only the scalar 1D Hermite coefficients $g_k(z_c)$ are recomputed. Zero spatial linear systems need to be re-inverted! In contrast, Indicator Kriging requires setting up, fitting semivariograms, and inverting $n \times n$ systems for every individual cutoff. $\blacksquare$

---

### Exercise 9.4: Classical Uniform Score Transform and Grade-Tonnage Curves
In spatial assessment of mineral resources and environmental contaminants, two fundamental non-linear curve functions describe the recoverable resources or hazardous masses above a critical regulatory threshold $z_c$: the **recovered tonnage** $T(z_c)$ and the **recovered metal / mass** $Q(z_c)$.

**(a)** Given $n$ univariate sample measurements $z_1, \dots, z_n$ from a continuous spatial process $Z(\mathbf{s})$, define the empirical cumulative distribution function (ECDF):
$$
\widehat{F}_n(z) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(z_i \le z)
$$
Show how the uniform score transform $u_i = \widehat{F}_n(z_i)$ (or its continuous quantile counterpart $U = F(Z)$) maps arbitrary continuous random variables to a standard uniform distribution $\mathcal{U}[0, 1]$. Then explain how Gaussian anamorphosis maps uniform scores to standard normal scores $y_i = \Phi^{-1}(u_i)$.

**(b)** Let $f(z)$ and $F(z)$ denote the marginal probability density function and cumulative distribution function of $Z(\mathbf{s})$. For any threshold $z_c \ge 0$, define:
1. The recovered tonnage proportion: $T(z_c) = P(Z(\mathbf{s}) > z_c) = 1 - F(z_c) = \int_{z_c}^\infty f(z) \mathrm{d}z$.
2. The recovered metal / contaminant mass per unit volume: $Q(z_c) = \operatorname{E}[Z(\mathbf{s}) \cdot \mathbb{I}(Z(\mathbf{s}) > z_c)] = \int_{z_c}^\infty z f(z) \mathrm{d}z$.
3. The average grade above cutoff: $m(z_c) = \frac{Q(z_c)}{T(z_c)} = \operatorname{E}[Z(\mathbf{s}) \mid Z(\mathbf{s}) > z_c]$.

**(c)** Using integration by parts on $Q(z_c) = \int_{z_c}^\infty z f(z) \mathrm{d}z$ under the mild regularity condition $\lim_{z \to \infty} z T(z) = 0$, prove the fundamental grade-tonnage identity:
$$
Q(z_c) = z_c T(z_c) + \int_{z_c}^\infty T(u) \mathrm{d}u
$$

**(d)** Suppose an environmental agency or mining company evaluates $T(z_c)$ and $Q(z_c)$ by plugging a smoothed kriged map $\widehat{Z}_{\text{OK}}(\mathbf{s})$ directly into the definitions:
$$
\widehat{T}_{\text{naive}}(z_c) = \frac{1}{|A|} \int_A \mathbb{I}(\widehat{Z}_{\text{OK}}(\mathbf{s}) > z_c) \mathrm{d}\mathbf{s}
$$
Explain why this plug-in approach induces severe systematic bias across both high and low thresholds (overestimating tonnage at low thresholds, drastically underestimating tonnage and metal at high thresholds), and why non-linear conditional CDF (CCDF) modeling or stochastic simulation is mathematically required.

##### Solution to Exercise 9.4
**(a)** 
1. Probability Integral Transform:
Let $Z$ be a continuous random variable with strictly increasing CDF $F_Z(z) = P(Z \le z)$.
Define the transformed variable $U = F_Z(Z)$. For any $u \in (0, 1)$:
$$
F_U(u) = P(U \le u) = P(F_Z(Z) \le u) = P\left(Z \le F_Z^{-1}(u)\right) = F_Z\left(F_Z^{-1}(u)\right) = u
$$
Since $F_U(u) = u$ for all $u \in [0, 1]$, $U$ is distributed as a standard uniform random variable: $U \sim \mathcal{U}[0, 1]$.
For empirical sample data $z_1 \le z_2 \le \dots \le z_n$, standard plotting positions (such as $u_i = \frac{\operatorname{rank}(z_i) - 0.5}{n}$ or $\frac{\operatorname{rank}(z_i)}{n+1}$) map empirical observations onto the uniform score interval $(0, 1)$.
2. Gaussian Anamorphosis:
Let $\Phi(y) = \int_{-\infty}^y \frac{1}{\sqrt{2\pi}}e^{-t^2/2} \mathrm{d}t$ be the standard Gaussian CDF, with inverse probit function $\Phi^{-1}$.
Defining $Y = \Phi^{-1}(U) = \Phi^{-1}(F_Z(Z))$ yields:
$$
P(Y \le y) = P\left(\Phi^{-1}(U) \le y\right) = P(U \le \Phi(y)) = \Phi(y)
$$
Hence, $Y \sim \mathcal{N}(0, 1)$ is strictly standard Gaussian. This bijective mapping $\phi: z \mapsto \Phi^{-1}(F_Z(z))$ is the classical Gaussian anamorphosis function, enabling geostatisticians to transform arbitrarily skewed, heavy-tailed empirical histograms into Gaussian variables for Multi-Gaussian Kriging and Disjunctive Kriging.

**(b)** Definitions:
1. Recovered Tonnage $T(z_c)$ represents the volume or spatial fraction of the domain whose physical grade exceeds the cut-off $z_c$:
$$
T(z_c) = P(Z(\mathbf{s}) > z_c) = 1 - F(z_c) = \int_{z_c}^\infty f(z) \, \mathrm{d}z
$$
2. Recovered Metal / Contaminant Quantity $Q(z_c)$ represents the total accumulated mass per unit volume of material exceeding cutoff $z_c$:
$$
Q(z_c) = \operatorname{E}\left[Z(\mathbf{s}) \cdot \mathbb{I}(Z(\mathbf{s}) > z_c)\right] = \int_{z_c}^\infty z f(z) \, \mathrm{d}z
$$
3. Average Recovered Grade $m(z_c)$ is the expected concentration within the selected volume:
$$
m(z_c) = \operatorname{E}[Z(\mathbf{s}) \mid Z(\mathbf{s}) > z_c] = \frac{\int_{z_c}^\infty z f(z) \, \mathrm{d}z}{P(Z(\mathbf{s}) > z_c)} = \frac{Q(z_c)}{T(z_c)}
$$

**(c)** Proof of the Grade-Tonnage Identity:
Express $Q(z_c)$ as:
$$
Q(z_c) = \int_{z_c}^\infty z f(z) \, \mathrm{d}z
$$
Apply integration by parts $\int u \, \mathrm{d}v = u v - \int v \, \mathrm{d}u$.
Choose:
$$
u(z) = z \implies \mathrm{d}u = \mathrm{d}z
$$
$$
\mathrm{d}v = f(z) \, \mathrm{d}z \implies v(z) = -\int_z^\infty f(w) \, \mathrm{d}w = -T(z)
$$
Substitute into the integration by parts formula:
$$
Q(z_c) = \left[ -z T(z) \right]_{z_c}^\infty - \int_{z_c}^\infty (-T(u)) \, \mathrm{d}u = \left( \lim_{z \to \infty} [-z T(z)] - (-z_c T(z_c)) \right) + \int_{z_c}^\infty T(u) \, \mathrm{d}u
$$
Under the finite expectation condition $\operatorname{E}[Z] < \infty$, the tail probability satisfies $\lim_{z \to \infty} z T(z) = \lim_{z \to \infty} z P(Z > z) = 0$.
Therefore:
$$
Q(z_c) = z_c T(z_c) + \int_{z_c}^\infty T(u) \, \mathrm{d}u
$$
This identity demonstrates that recoverable metal $Q(z_c)$ is completely determined by the tail survival function $T(u) = 1 - F(u)$ integrated from $z_c$ to $\infty$.

**(d)** Systematic Bias of Smoothed Kriged Maps:
Ordinary Kriging is a minimum-variance linear projection: $\operatorname{Var}(\widehat{Z}_{\text{OK}}(\mathbf{s})) < \operatorname{Var}(Z(\mathbf{s}))$. Kriging smooths out extremes, causing the histogram of predictions $\{\widehat{Z}_{\text{OK}}(\mathbf{s})\}$ to have compressed tails relative to the true distribution $\{Z(\mathbf{s})\}$.
When this smoothed map is naively plugged into $T(z_c)$ and $Q(z_c)$:
1. **High Cutoff ($z_c > \mu$):** The true physical field possesses rich high-grade zones with values well above $z_c$. In the kriged map, these local spikes are averaged with surrounding lower values. The kriged surface rarely reaches $z_c$. As a result, $\widehat{T}_{\text{naive}}(z_c) \ll T(z_c)$ and $\widehat{Q}_{\text{naive}}(z_c) \ll Q(z_c)$. A mining project will erroneously conclude that there is virtually no recoverable high-grade ore, leading to premature abandonment of viable deposits.
2. **Low Cutoff ($z_c < \mu$):** Because kriging shrinks predictions toward the mean $\mu$, low-grade waste rock is pulled upward above the threshold. Thus, $\widehat{T}_{\text{naive}}(z_c) \gg T(z_c)$. A processing mill will forecast excessive tonnage of low-grade material, inducing massive waste dilution.
3. This is the classical "information effect" and smoothing penalty in mining and environmental evaluation. Because $T(z_c)$ and $Q(z_c)$ are non-linear step operators, computing the expectation of a non-linear operator from a smoothed expectation $\operatorname{E}[Z]$ violates Jensen's inequality:
$$
\operatorname{E}\left[\mathbb{I}(Z(\mathbf{s}) > z_c)\right] \ne \mathbb{I}\left(\operatorname{E}[Z(\mathbf{s})] > z_c\right)
$$
Accurate recovery estimation requires modeling the full conditional cumulative distribution function (CCDF) $F(\mathbf{s}; z \mid \text{data})$ at each location via Indicator Kriging or generating equi-probable stochastic simulations that reproduce the true spatial variance. $\blacksquare$

---

## Tier I: Core Theoretical Re-Derivations

### Exercise 9.5: Moment Generating Function and Exact Lognormal Kriging Back-Transformation
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be a strictly positive lognormal random field, such that the transformed field $Y(\mathbf{s}) = \ln Z(\mathbf{s})$ is a stationary Gaussian random field with mean $\mu_Y$ and stationary covariance $C_Y(\mathbf{h})$.
Let $\mathbf{Y} = (Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_n))'$ denote the log-transformed observations.
Suppose we perform Simple Kriging on the Gaussian field $Y$ to predict $Y(\mathbf{s}_0)$, yielding the conditional distribution:
$$
Y(\mathbf{s}_0) \mid \mathbf{Y} \sim \mathcal{N}\left(\widehat{Y}_{\text{SK}}(\mathbf{s}_0), \, \sigma_{\text{SK}}^2(\mathbf{s}_0)\right)
$$
where $\widehat{Y}_{\text{SK}}(\mathbf{s}_0) = \mu_Y + \boldsymbol{\lambda}'(\mathbf{Y} - \mu_Y \mathbf{1})$ and $\sigma_{\text{SK}}^2(\mathbf{s}_0) = C_Y(\mathbf{0}) - \boldsymbol{\lambda}'\mathbf{c}_{Y,0}$.
**(a)** Using the Moment Generating Function (MGF) of a normal random variable $X \sim \mathcal{N}(\mu, \sigma^2)$, $M_X(t) = \operatorname{E}[\exp(tX)] = \exp(\mu t + \frac{1}{2}\sigma^2 t^2)$, derive the exact Minimum Mean Squared Error (MMSE) predictor of the original attribute $Z(\mathbf{s}_0) = \exp(Y(\mathbf{s}_0))$:
$$
\widehat{Z}_{\text{LK}}(\mathbf{s}_0) = \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}] = \exp\left( \widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0) \right)
$$
**(b)** Define the naive back-transformation $\widehat{Z}_{\text{naive}}(\mathbf{s}_0) = \exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))$. Using Jensen's inequality for strictly convex functions, prove that:
$$
\operatorname{E}[\widehat{Z}_{\text{naive}}(\mathbf{s}_0) \mid \mathbf{Y}] < \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}]
$$
and quantify the exact percentage bias of naive back-transformation as a function of the kriging error variance $\sigma_{\text{SK}}^2(\mathbf{s}_0)$.
**(c)** Derive the conditional prediction error variance $\operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Y})$.

##### Solution to Exercise 9.5
**(a)** Because $Y(\mathbf{s})$ is a Gaussian random field, the conditional distribution of $Y(\mathbf{s}_0)$ given the observations $\mathbf{Y}$ is univariate Gaussian:
$$
Y(\mathbf{s}_0) \mid \mathbf{Y} \sim \mathcal{N}(\mu_c, \sigma_c^2)
$$
where $\mu_c = \widehat{Y}_{\text{SK}}(\mathbf{s}_0)$ and $\sigma_c^2 = \sigma_{\text{SK}}^2(\mathbf{s}_0)$.
The target is $Z(\mathbf{s}_0) = \exp(Y(\mathbf{s}_0))$.
The conditional expectation is:
$$
\widehat{Z}_{\text{LK}}(\mathbf{s}_0) = \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}] = \operatorname{E}[\exp(1 \cdot Y(\mathbf{s}_0)) \mid \mathbf{Y}] = M_{Y(\mathbf{s}_0) \mid \mathbf{Y}}(1)
$$
Using the Gaussian MGF formula $M_X(t) = \exp(\mu t + \frac{1}{2}\sigma^2 t^2)$ evaluated at $t = 1$:
$$
\widehat{Z}_{\text{LK}}(\mathbf{s}_0) = \exp\left( \mu_c(1) + \frac{1}{2}\sigma_c^2(1)^2 \right) = \exp\left( \widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0) \right)
$$
This is the exact, unbiased Lognormal Kriging back-transformation.

**(b)** The naive predictor is $\widehat{Z}_{\text{naive}}(\mathbf{s}_0) = \exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0)) = \exp(\operatorname{E}[Y(\mathbf{s}_0) \mid \mathbf{Y}])$.
The function $g(y) = e^y$ is strictly convex ($g''(y) = e^y > 0$ for all $y \in \mathbb{R}$).
By Jensen's inequality for strictly convex functions and non-degenerate distributions ($\sigma_{\text{SK}}^2 > 0$):
$$
g(\operatorname{E}[Y(\mathbf{s}_0) \mid \mathbf{Y}]) < \operatorname{E}[g(Y(\mathbf{s}_0)) \mid \mathbf{Y}]
$$
$$
\exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0)) < \operatorname{E}[Z(\mathbf{s}_0) \mid \mathbf{Y}]
$$
The ratio between the naive predictor and the true conditional expectation is:
$$
\frac{\widehat{Z}_{\text{naive}}(\mathbf{s}_0)}{\widehat{Z}_{\text{LK}}(\mathbf{s}_0)} = \frac{\exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0))}{\exp(\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0))} = \exp\left( -\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0) \right) < 1
$$
The percentage bias is:
$$
\text{Bias} = \left[ \exp\left( -\frac{1}{2}\sigma_{\text{SK}}^2(\mathbf{s}_0) \right) - 1 \right] \times 100\%
$$
For instance, if the log-scale kriging variance is $\sigma_{\text{SK}}^2 = 0.50$, the naive predictor underestimates the true physical concentration by $1 - e^{-0.25} \approx 1 - 0.7788 = 22.12\%$!
If $\sigma_{\text{SK}}^2 = 1.0$, naive exponentiation underestimates reality by $1 - e^{-0.5} \approx 39.35\%$.
The naive predictor systematically under-predicts physical reserves and contamination levels, with the underestimation worsening rapidly in data-sparse regions.

**(c)** Using the variance formula for a lognormal random variable with parameters $\mu_c$ and $\sigma_c^2$:
$$
\operatorname{Var}(W) = (\operatorname{E}[W])^2 (\exp(\sigma_c^2) - 1)
$$
Substituting $W = Z(\mathbf{s}_0) \mid \mathbf{Y}$:
$$
\begin{aligned}
\operatorname{Var}(Z(\mathbf{s}_0) \mid \mathbf{Y}) &= (\widehat{Z}_{\text{LK}}(\mathbf{s}_0))^2 \left( \exp(\sigma_{\text{SK}}^2(\mathbf{s}_0)) - 1 \right) \\
&= \exp\left( 2\widehat{Y}_{\text{SK}}(\mathbf{s}_0) + \sigma_{\text{SK}}^2(\mathbf{s}_0) \right) \left( \exp(\sigma_{\text{SK}}^2(\mathbf{s}_0)) - 1 \right)
\end{aligned}
$$
Notice that unlike in Gaussian kriging where prediction variance is independent of data values, the lognormal prediction variance is proportional to the square of the predicted value $\widehat{Z}_{\text{LK}}(\mathbf{s}_0)^2$: regions with high predicted values carry proportionally massive prediction uncertainties. $\blacksquare$

---

### Exercise 9.6: Indicator Random Fields and Conditional Probability
Let $\{Z(\mathbf{s}) : \mathbf{s} \in D\}$ be an arbitrary stationary spatial random field. Choose a critical physical threshold $z_c \in \mathbb{R}$. The **indicator random field** at threshold $z_c$ is defined as the binary variable:
$$
I(\mathbf{s}; z_c) = \mathbb{I}\{Z(\mathbf{s}) \le z_c\} = \begin{cases} 1, & \text{if } Z(\mathbf{s}) \le z_c \\ 0, & \text{if } Z(\mathbf{s}) > z_c \end{cases}
$$
**(a)** Prove that the expectation of $I(\mathbf{s}; z_c)$ equals the marginal cumulative distribution function (CDF) of $Z$ at $z_c$:
$$
\operatorname{E}[I(\mathbf{s}; z_c)] = P(Z(\mathbf{s}) \le z_c) = F(z_c)
$$
and compute its variance $\operatorname{Var}(I(\mathbf{s}; z_c))$.
**(b)** Let $\mathbf{I}(z_c) = (I(\mathbf{s}_1; z_c), \dots, I(\mathbf{s}_n; z_c))'$ be the observed indicator data. Suppose we perform Ordinary Kriging directly on these binary indicator variables:
$$
\widehat{I}(\mathbf{s}_0; z_c) = \sum_{i=1}^n \lambda_i(z_c) I(\mathbf{s}_i; z_c), \quad \text{with } \sum_{i=1}^n \lambda_i(z_c) = 1
$$
Prove that $\widehat{I}(\mathbf{s}_0; z_c)$ is an estimator of the **conditional cumulative distribution function** (CCDF):
$$
\widehat{I}(\mathbf{s}_0; z_c) \approx P(Z(\mathbf{s}_0) \le z_c \mid \text{data}) = F(\mathbf{s}_0; z_c \mid \text{data})
$$
**(c)** Show how the probability of exceeding an environmental contamination limit $z_{\text{toxic}}$ is obtained directly from $\widehat{I}(\mathbf{s}_0; z_{\text{toxic}})$.
**(d)** Explain why Indicator Kriging is completely non-parametric with respect to outliers, monotonic transformations, and non-linear shapes.

##### Solution to Exercise 9.6
**(a)** The expectation of a Bernoulli indicator variable is:
$$
\operatorname{E}[I(\mathbf{s}; z_c)] = 1 \cdot P(I(\mathbf{s}; z_c) = 1) + 0 \cdot P(I(\mathbf{s}; z_c) = 0) = P(Z(\mathbf{s}) \le z_c) = F(z_c)
$$
Its variance is:
$$
\operatorname{Var}(I(\mathbf{s}; z_c)) = \operatorname{E}[I(\mathbf{s}; z_c)^2] - (\operatorname{E}[I(\mathbf{s}; z_c)])^2
$$
Since $I^2 = I$ (idempotence of binary indicators):
$$
\operatorname{Var}(I(\mathbf{s}; z_c)) = F(z_c) - [F(z_c)]^2 = F(z_c)(1 - F(z_c))
$$

**(b)** The optimal mean-square predictor of an indicator variable $I(\mathbf{s}_0; z_c)$ given the information $\mathcal{F} = \sigma(I(\mathbf{s}_1; z_c), \dots, I(\mathbf{s}_n; z_c))$ is the conditional expectation:
$$
\operatorname{E}[I(\mathbf{s}_0; z_c) \mid \mathcal{F}] = P(I(\mathbf{s}_0; z_c) = 1 \mid \mathcal{F}) = P(Z(\mathbf{s}_0) \le z_c \mid \mathcal{F}) = F(\mathbf{s}_0; z_c \mid \mathcal{F})
$$
Ordinary Kriging approximates this conditional expectation via the best linear unbiased projection in $L^2$:
$$
\widehat{I}(\mathbf{s}_0; z_c) = \sum_{i=1}^n \lambda_i(z_c) I(\mathbf{s}_i; z_c)
$$
Subject to $\sum \lambda_i = 1$, the predictor is unbiased:
$$
\operatorname{E}[\widehat{I}(\mathbf{s}_0; z_c)] = \sum_{i=1}^n \lambda_i(z_c) \operatorname{E}[I(\mathbf{s}_i; z_c)] = F(z_c) \sum_{i=1}^n \lambda_i(z_c) = F(z_c)
$$
Hence, $\widehat{I}(\mathbf{s}_0; z_c)$ directly estimates the local posterior probability $P(Z(\mathbf{s}_0) \le z_c \mid \text{data})$.

**(c)** The probability of exceeding an environmental contamination limit $z_{\text{toxic}}$ is:
$$
P(Z(\mathbf{s}_0) > z_{\text{toxic}} \mid \text{data}) = 1 - P(Z(\mathbf{s}_0) \le z_{\text{toxic}} \mid \text{data}) = 1 - \widehat{I}(\mathbf{s}_0; z_{\text{toxic}})
$$
This allows environmental engineers to map risk contours (e.g., all regions where $P(Z > z_{\text{toxic}}) \ge 0.10$) directly without making any distributional assumptions.

**(d)** Non-parametric robustness of Indicator Kriging:
1. **Invariance to Outliers:** If $z_c = 10$, an extreme outlier value $Z(\mathbf{s}_i) = 100,000$ yields $I(\mathbf{s}_i; 10) = 0$, exactly the same as an observation of $Z(\mathbf{s}_i) = 11$. Extreme erratic spikes cannot pull the predictor away from reality.
2. **Invariance to Monotonic Transformations:** For any strictly increasing function $g$, $\mathbb{I}\{g(Z(\mathbf{s})) \le g(z_c)\} \equiv \mathbb{I}\{Z(\mathbf{s}) \le z_c\}$. The indicator data and variograms are invariant to nonlinear re-scaling (e.g., logarithmic, Box-Cox).
3. **No Distributional Form Required:** Unlike Gaussian or lognormal kriging, IK requires no assumption of normality, lognormality, or unimodality. It can model multimodal, skewed, or mixture distributions directly. $\blacksquare$

---

### Exercise 9.7: Order Relation Violations and Isotonic PAVA Projection
When Indicator Kriging is applied independently across $K$ ordered thresholds $z_1 < z_2 < \dots < z_K$, the unconstrained linear weights can produce **order relation violations**:
1. Probabilities outside the valid unit interval: $\widehat{F}(z_k) < 0$ or $\widehat{F}(z_k) > 1$.
2. Negative incremental probabilities (non-monotonicity): $\widehat{F}(z_k) > \widehat{F}(z_{k+1})$ for $z_k < z_{k+1}$.
To construct a legitimate conditional CDF, the raw estimate $\widehat{\mathbf{F}} = (\widehat{F}(z_1), \dots, \widehat{F}(z_K))'$ must be projected onto the convex cone of non-decreasing sequences in $[0, 1]^K$:
$$
\mathcal{C} = \left\{ \mathbf{p} \in \mathbb{R}^K : 0 \le p_1 \le p_2 \le \dots \le p_K \le 1 \right\}
$$
**(a)** Formulate this projection as a weighted least-squares isotonic regression problem:
$$
\min_{\mathbf{p} \in \mathcal{C}} \sum_{k=1}^K w_k (p_k - \widehat{F}(z_k))^2
$$
where $w_k > 0$ are user-specified weights (typically uniform $w_k = 1$).
**(b)** State the **Pool-Adjacent-Violators Algorithm (PAVA)** for solving this quadratic program.
**(c)** Prove that if a violation occurs between two adjacent elements ($\widehat{F}(z_k) > \widehat{F}(z_{k+1})$), the optimal restricted value $p_k = p_{k+1}$ must equal their weighted average:
$$
p_k = p_{k+1} = \frac{w_k \widehat{F}(z_k) + w_{k+1} \widehat{F}(z_{k+1})}{w_k + w_{k+1}}
$$

##### Solution to Exercise 9.7
**(a)** Formulation of the projection:
Let $\widehat{\mathbf{f}} = (\widehat{F}(z_1), \dots, \widehat{F}(z_K))'$. The projection onto the closed convex set $\mathcal{C}$ in the Hilbert space $\mathbb{R}^K$ equipped with the inner product $\langle \mathbf{u}, \mathbf{v} \rangle_{\mathbf{W}} = \sum_{k=1}^K w_k u_k v_k$ is the unique vector $\mathbf{p}^* \in \mathcal{C}$ that solves:
$$
\min_{\mathbf{p}} \sum_{k=1}^K w_k (p_k - \widehat{f}_k)^2 \quad \text{subject to } \begin{cases} 0 \le p_1 \\ p_k \le p_{k+1}, & k=1, \dots, K-1 \\ p_K \le 1 \end{cases}
$$
By the Hilbert Projection Theorem, because $\mathcal{C}$ is a non-empty closed convex cone, the solution $\mathbf{p}^*$ exists and is unique.

**(b)** The Pool-Adjacent-Violators Algorithm (PAVA):
PAVA solves isotonic regression in $\mathcal{O}(K)$ operations:
1. **Initialization:** Assign each element to its own block: $B_k = \{k\}$ with block value $\bar{y}_k = \widehat{f}_k$ and weight $W_k = w_k$ for $k=1, \dots, K$.
2. **Scan and Merge:**
   - Scan the list of blocks from left to right.
   - If adjacent blocks $B_i$ and $B_{i+1}$ satisfy the monotonicity condition $\bar{y}_i \le \bar{y}_{i+1}$, proceed to the next pair.
   - If an order violation occurs ($\bar{y}_i > \bar{y}_{i+1}$), merge $B_i$ and $B_{i+1}$ into a combined block $B = B_i \cup B_{i+1}$.
   - The merged block is assigned the pooled weighted mean:
     $$
     \bar{y}_{\text{merged}} = \frac{W_i \bar{y}_i + W_{i+1} \bar{y}_{i+1}}{W_i + W_{i+1}}, \quad W_{\text{merged}} = W_i + W_{i+1}
     $$
   - Check backwards: if the new block violates monotonicity with its predecessor ($B_{i-1}$), merge backwards recursively until the entire sequence of blocks is non-decreasing.
3. **Threshold Boundary Clipping:**
   - Once monotonicity is achieved, apply boundary constraints:
     $$
     p_k^* = \max\left(0, \min(1, \bar{y}(B(k)))\right)
     $$
   This yields a mathematically valid CDF $0 \le p_1^* \le p_2^* \le \dots \le p_K^* \le 1$.

**(c)** Proof of the weighted average for adjacent violators:
Consider two indices $k$ and $k+1$ where $\widehat{f}_k > \widehat{f}_{k+1}$.
If the constraint $p_k \le p_{k+1}$ is active, we must have $p_k = p_{k+1} = c$.
The contribution of these two terms to the objective function is:
$$
J(c) = w_k (c - \widehat{f}_k)^2 + w_{k+1} (c - \widehat{f}_{k+1})^2
$$
To find the minimizer, differentiate with respect to $c$ and equate to zero:
$$
\frac{d J}{d c} = 2 w_k (c - \widehat{f}_k) + 2 w_{k+1} (c - \widehat{f}_{k+1}) = 0
$$
$$
(w_k + w_{k+1}) c - (w_k \widehat{f}_k + w_{k+1} \widehat{f}_{k+1}) = 0
$$
$$
c = \frac{w_k \widehat{f}_k + w_{k+1} \widehat{f}_{k+1}}{w_k + w_{k+1}}
$$
Because $J''(c) = 2(w_k + w_{k+1}) > 0$, this unique stationary point is the global minimum. $\blacksquare$

---

### Exercise 9.8: Hermite Polynomial Expansions and Disjunctive Kriging
Disjunctive Kriging (DK) generalizes linear kriging by searching for optimal predictors within the space of all nonlinear functions of the form:
$$
\widehat{\psi}(\mathbf{s}_0) = \sum_{i=1}^n f_i(Y(\mathbf{s}_i))
$$
where $Y(\mathbf{s}) = \Phi^{-1}(F(Z(\mathbf{s})))$ is the standard Gaussian anamorphosis transform.
The mathematical engine of Disjunctive Kriging is the system of normalized Hermite polynomials $\{H_k(y)\}_{k=0}^\infty$ with respect to the standard Gaussian density $\phi(y) = \frac{1}{\sqrt{2\pi}} e^{-y^2/2}$.
The Hermite polynomials are defined by Rodrigues' formula:
$$
H_k(y) = \frac{(-1)^k}{\sqrt{k!}} e^{y^2/2} \frac{d^k}{dy^k}\left(e^{-y^2/2}\right)
$$
**(a)** Compute the first four normalized Hermite polynomials $H_0(y), H_1(y), H_2(y), H_3(y)$.
**(b)** Prove the orthonormality property:
$$
\int_{-\infty}^\infty H_j(y) H_k(y) \phi(y) dy = \delta_{jk} = \begin{cases} 1, & j = k \\ 0, & j \ne k \end{cases}
$$
**(c)** State **Mehler's Formula** for the bivariate standard Gaussian density $\phi_\rho(y_1, y_2)$ with correlation $\rho$:
$$
\frac{\phi_\rho(y_1, y_2)}{\phi(y_1)\phi(y_2)} = \sum_{k=0}^\infty \rho^k H_k(y_1) H_k(y_2)
$$
and prove that for any $j, k \ge 0$:
$$
\operatorname{Cov}(H_j(Y(\mathbf{s}_1)), H_k(Y(\mathbf{s}_2))) = \rho(\mathbf{s}_1 - \mathbf{s}_2)^k \delta_{jk}
$$
**(d)** Explain why this orthogonality property decouples the massive nonlinear Disjunctive Kriging system into independent, univariate linear kriging systems for each polynomial degree $k$.

##### Solution to Exercise 9.8
**(a)** Using Rodrigues' formula $H_k(y) = \frac{(-1)^k}{\sqrt{k!}} e^{y^2/2} \frac{d^k}{dy^k}(e^{-y^2/2})$:
- $k=0$:
  $$
  H_0(y) = \frac{(-1)^0}{\sqrt{0!}} e^{y^2/2} (e^{-y^2/2}) = 1
  $$
- $k=1$:
  $$
  \frac{d}{dy}(e^{-y^2/2}) = -y e^{-y^2/2} \implies H_1(y) = \frac{(-1)^1}{\sqrt{1!}} e^{y^2/2} (-y e^{-y^2/2}) = y
  $$
- $k=2$:
  $$
  \frac{d^2}{dy^2}(e^{-y^2/2}) = \frac{d}{dy}(-y e^{-y^2/2}) = -e^{-y^2/2} + y^2 e^{-y^2/2} = (y^2 - 1) e^{-y^2/2}
  $$
  $$
  H_2(y) = \frac{(-1)^2}{\sqrt{2!}} e^{y^2/2} (y^2 - 1) e^{-y^2/2} = \frac{y^2 - 1}{\sqrt{2}}
  $$
- $k=3$:
  $$
  \frac{d}{dy}((y^2 - 1)e^{-y^2/2}) = 2y e^{-y^2/2} - y(y^2 - 1)e^{-y^2/2} = (-y^3 + 3y)e^{-y^2/2}
  $$
  $$
  H_3(y) = \frac{(-1)^3}{\sqrt{6}} e^{y^2/2} (-y^3 + 3y)e^{-y^2/2} = \frac{y^3 - 3y}{\sqrt{6}}
  $$

**(b)** Proof of Orthonormality:
Let $I_{jk} = \int_{-\infty}^\infty H_j(y) H_k(y) \phi(y) dy$. Assume without loss of generality that $j \le k$.
Substitute Rodrigues' formula for $H_k(y)$:
$$
I_{jk} = \int_{-\infty}^\infty H_j(y) \left[ \frac{(-1)^k}{\sqrt{k!}} e^{y^2/2} \frac{d^k}{dy^k}(e^{-y^2/2}) \right] \frac{1}{\sqrt{2\pi}} e^{-y^2/2} dy = \frac{(-1)^k}{\sqrt{k!} \sqrt{2\pi}} \int_{-\infty}^\infty H_j(y) \frac{d^k}{dy^k}(e^{-y^2/2}) dy
$$
Integrate by parts $k$ times. Since $H_j(y)$ is a polynomial of degree $j$, differentiating $H_j(y)$ $k$ times yields:
- If $j < k$: $\frac{d^k}{dy^k} H_j(y) = 0$. Thus $I_{jk} = 0$.
- If $j = k$: $H_k(y) = \frac{y^k}{\sqrt{k!}} + \dots$, so $\frac{d^k}{dy^k} H_k(y) = \frac{k!}{\sqrt{k!}} = \sqrt{k!}$.
  Then integrating by parts $k$ times:
  $$
  I_{kk} = \frac{(-1)^k}{\sqrt{k!}\sqrt{2\pi}} (-1)^k \int_{-\infty}^\infty \sqrt{k!} e^{-y^2/2} dy = \frac{\sqrt{k!}}{\sqrt{k!}} \frac{1}{\sqrt{2\pi}} \int_{-\infty}^\infty e^{-y^2/2} dy = 1 \cdot 1 = 1
  $$
Therefore, $\int_{-\infty}^\infty H_j(y) H_k(y) \phi(y) dy = \delta_{jk}$.

**(c)** Covariance between Hermite polynomials:
Under Mehler's formula:
$$
\phi_\rho(y_1, y_2) = \phi(y_1) \phi(y_2) \sum_{m=0}^\infty \rho^m H_m(y_1) H_m(y_2)
$$
The expectation of the product $H_j(Y_1) H_k(Y_2)$ is:
$$
\begin{aligned}
\operatorname{E}[H_j(Y_1) H_k(Y_2)] &= \int_{-\infty}^\infty \int_{-\infty}^\infty H_j(y_1) H_k(y_2) \phi_\rho(y_1, y_2) dy_1 dy_2 \\
&= \sum_{m=0}^\infty \rho^m \left( \int_{-\infty}^\infty H_j(y_1) H_m(y_1) \phi(y_1) dy_1 \right) \left( \int_{-\infty}^\infty H_k(y_2) H_m(y_2) \phi(y_2) dy_2 \right) \\
&= \sum_{m=0}^\infty \rho^m \delta_{jm} \delta_{km} = \rho^k \delta_{jk}
\end{aligned}
$$
For $k \ge 1$, $\operatorname{E}[H_k(Y)] = \langle H_k, H_0 \rangle = 0$. Thus:
$$
\operatorname{Cov}(H_j(Y(\mathbf{s}_1)), H_k(Y(\mathbf{s}_2))) = \operatorname{E}[H_j(Y_1) H_k(Y_2)] = [\rho(\mathbf{s}_1 - \mathbf{s}_2)]^k \delta_{jk}
$$

**(d)** Decoupling of the Disjunctive Kriging system:
Because $\operatorname{Cov}(H_j(Y_i), H_k(Y_l)) = 0$ for all $j \ne k$, the cross-covariance between different polynomial orders is strictly zero!
Any transform function $\psi(Y)$ can be expanded into Hermite coefficients: $\psi(y) = \sum_{k=0}^\infty \psi_k H_k(y)$.
The Disjunctive Kriging predictor is:
$$
\widehat{\psi}(Y(\mathbf{s}_0)) = \sum_{k=0}^K \psi_k \widehat{H}_k(\mathbf{s}_0)
$$
where each polynomial factor $\widehat{H}_k(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i^{(k)} H_k(Y(\mathbf{s}_i))$ is solved independently using its own $n \times n$ linear kriging system with covariance kernel $C_k(\mathbf{h}) = [\rho(\mathbf{h})]^k$.
Instead of solving a massive, coupled, non-convex nonlinear optimization problem, Disjunctive Kriging solves $K$ separate, standard linear kriging systems. $\blacksquare$

---

## Tier II: Pencil-and-Paper Mechanics

### Exercise 9.9: Pencil-and-Paper 2-Point Lognormal Simple Kriging
Consider two sampling locations in 1D: $s_1 = 0$ and $s_2 = 2$.
The observations on the physical scale are:
$$
Z(s_1) = e^1 \approx 2.7183, \quad Z(s_2) = e^3 \approx 20.0855
$$
The log-transformed field $Y(s) = \ln Z(s)$ is Gaussian:
$$
Y(s_1) = 1.0, \quad Y(s_2) = 3.0
$$
The known prior mean of the log-field is $\mu_Y = 2.0$, and the covariance is exponential without nugget:
$$
C_Y(h) = \sigma_Y^2 e^{-|h|/\phi}
$$
with $\sigma_Y^2 = 1.0$ and range $\phi = 2.0$.
We wish to predict the concentration at the midpoint $s_0 = 1.0$.
**(a)** Compute the log-scale covariance matrix $\mathbf{V}_Y$ and target covariance vector $\mathbf{c}_{Y,0}$.
**(b)** Solve the Simple Kriging system $\mathbf{V}_Y \boldsymbol{\lambda} = \mathbf{c}_{Y,0}$ by hand to find $\lambda_1, \lambda_2$.
**(c)** Compute the log-scale Simple Kriging prediction $\widehat{Y}_{\text{SK}}(1)$ and error variance $\sigma_{\text{SK}}^2(1)$.
**(d)** Compute:
1. The naive back-transformation $\widehat{Z}_{\text{naive}}(1) = \exp(\widehat{Y}_{\text{SK}}(1))$.
2. The exact unbiased Lognormal Kriging prediction $\widehat{Z}_{\text{LK}}(1) = \exp(\widehat{Y}_{\text{SK}}(1) + \frac{1}{2}\sigma_{\text{SK}}^2(1))$.
3. The conditional lognormal prediction standard deviation $\sqrt{\operatorname{Var}(Z(1) \mid \mathbf{Y})}$.

##### Solution to Exercise 9.9
**(a)** Distances:
- $|s_1 - s_2| = |0 - 2| = 2$.
- $|s_1 - s_0| = |0 - 1| = 1$.
- $|s_2 - s_0| = |2 - 1| = 1$.
Using $C_Y(h) = e^{-|h|/2}$:
- $V_{11} = V_{22} = C_Y(0) = 1.0$.
- $V_{12} = V_{21} = C_Y(2) = e^{-2/2} = e^{-1} \approx 0.367879$.
- $c_{0,1} = c_{0,2} = C_Y(1) = e^{-1/2} \approx 0.606531$.
$$
\mathbf{V}_Y = \begin{pmatrix} 1.0 & e^{-1} \\ e^{-1} & 1.0 \end{pmatrix}, \quad \mathbf{c}_{Y,0} = \begin{pmatrix} e^{-1/2} \\ e^{-1/2} \end{pmatrix}
$$

**(b)** The Simple Kriging system is:
$$
\begin{pmatrix} 1 & e^{-1} \\ e^{-1} & 1 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} e^{-1/2} \\ e^{-1/2} \end{pmatrix}
$$
By symmetry, $\lambda_1 = \lambda_2$.
From the first row:
$$
\lambda_1 + e^{-1} \lambda_1 = e^{-1/2} \implies \lambda_1(1 + e^{-1}) = e^{-1/2}
$$
$$
\lambda_1 = \lambda_2 = \frac{e^{-1/2}}{1 + e^{-1}} \approx \frac{0.606531}{1 + 0.367879} = \frac{0.606531}{1.367879} \approx 0.44341
$$

**(c)** Centered Simple Kriging prediction ($\mu_Y = 2.0$):
Observed centered values:
- $Y(s_1) - \mu_Y = 1.0 - 2.0 = -1.0$.
- $Y(s_2) - \mu_Y = 3.0 - 2.0 = +1.0$.
$$
\widehat{Y}_{\text{SK}}(1) = \mu_Y + \lambda_1(Y(s_1) - \mu_Y) + \lambda_2(Y(s_2) - \mu_Y) = 2.0 + (0.44341)(-1.0) + (0.44341)(+1.0) = 2.0000
$$
Simple Kriging variance:
$$
\begin{aligned}
\sigma_{\text{SK}}^2(1) &= C_Y(0) - \boldsymbol{\lambda}'\mathbf{c}_{Y,0} = 1.0 - (\lambda_1 c_{0,1} + \lambda_2 c_{0,2}) \\
&= 1.0 - 2(0.44341)(0.606531) = 1.0 - 2(0.26894) = 1.0 - 0.53788 = 0.46212
\end{aligned}
$$

**(d)** Back-transformations:
1. **Naive predictor:**
   $$
   \widehat{Z}_{\text{naive}}(1) = \exp(\widehat{Y}_{\text{SK}}(1)) = \exp(2.0) \approx 7.3891
   $$
2. **Exact unbiased Lognormal Kriging predictor:**
   $$
   \begin{aligned}
   \widehat{Z}_{\text{LK}}(1) &= \exp\left( \widehat{Y}_{\text{SK}}(1) + \frac{1}{2}\sigma_{\text{SK}}^2(1) \right) \\
   &= \exp\left( 2.0 + \frac{1}{2}(0.46212) \right) = \exp(2.0 + 0.23106) = \exp(2.23106) \approx 9.3097
   $$
   Notice that the naive predictor ($7.3891$) severely underestimates the true conditional expectation ($9.3097$) by over $-20.6\%$!
3. **Conditional standard deviation:**
   $$
   \begin{aligned}
   \operatorname{Var}(Z(1) \mid \mathbf{Y}) &= (\widehat{Z}_{\text{LK}}(1))^2 (\exp(\sigma_{\text{SK}}^2(1)) - 1) \\
   &= (9.3097)^2 (\exp(0.46212) - 1) \\
   &= (86.6705)(1.58744 - 1) = (86.6705)(0.58744) \approx 50.9137
   \end{aligned}
   $$
   $$
   \operatorname{SD}(Z(1) \mid \mathbf{Y}) = \sqrt{50.9137} \approx 7.1354
   $$
The predictive distribution is strongly skewed: prediction $9.31 \pm 7.14$. $\blacksquare$

---

### Exercise 9.10: Pencil-and-Paper Manual PAVA Trace on a Corrupted Spatial CDF
Suppose an Indicator Kriging routine independently evaluates conditional probabilities across $K = 5$ ordered thresholds $z_1 < z_2 < z_3 < z_4 < z_5$. Due to sample clustering and negative kriging weights, the raw estimated vector is corrupted:
$$
\widehat{\mathbf{F}} = \left( \widehat{F}(z_1), \widehat{F}(z_2), \widehat{F}(z_3), \widehat{F}(z_4), \widehat{F}(z_5) \right) = (0.25, \, -0.05, \, 0.60, \, 0.55, \, 1.10)
$$
with uniform weights $w_k = 1$ for all $k$.
**(a)** Identify all order relation violations (both boundary violations and non-monotonicities).
**(b)** Trace the execution of the Pool-Adjacent-Violators Algorithm (PAVA) step by step, showing every pooling operation, intermediate block values, and backwards merges.
**(c)** Apply boundary constraints ($[0, 1]$ clipping) and report the final adjusted monotonic conditional CDF $\mathbf{F}^*$.

##### Solution to Exercise 9.10
**(a)** Identification of violations:
1. Boundary violation at index 2: $\widehat{F}(z_2) = -0.05 < 0$.
2. Monotonicity violation between index 1 and 2: $\widehat{F}(z_1) = 0.25 > -0.05 = \widehat{F}(z_2)$.
3. Monotonicity violation between index 3 and 4: $\widehat{F}(z_3) = 0.60 > 0.55 = \widehat{F}(z_4)$.
4. Boundary violation at index 5: $\widehat{F}(z_5) = 1.10 > 1.0$.

**(b)** Step-by-step PAVA trace:
- **Initial Blocks:**
  - $B_1 = \{1\}$, $\bar{y}_1 = 0.25$, $W_1 = 1$.
  - $B_2 = \{2\}$, $\bar{y}_2 = -0.05$, $W_2 = 1$.
  - $B_3 = \{3\}$, $\bar{y}_3 = 0.60$, $W_3 = 1$.
  - $B_4 = \{4\}$, $\bar{y}_4 = 0.55$, $W_4 = 1$.
  - $B_5 = \{5\}$, $\bar{y}_5 = 1.10$, $W_5 = 1$.
  List: $[0.25, -0.05, 0.60, 0.55, 1.10]$.

- **Step 1: Compare Block 1 and Block 2:**
  $\bar{y}_1 = 0.25 > -0.05 = \bar{y}_2 \implies$ **Violation!**
  Merge $B_1$ and $B_2$:
  $$
  B_{1,2} = \{1, 2\}, \quad W_{1,2} = 1 + 1 = 2
  $$
  $$
  \bar{y}_{1,2} = \frac{1(0.25) + 1(-0.05)}{2} = \frac{0.20}{2} = 0.10
  $$
  Sequence of blocks: $[(B_{1,2}: 0.10), (B_3: 0.60), (B_4: 0.55), (B_5: 1.10)]$.

- **Step 2: Compare Block $(1,2)$ and Block 3:**
  $\bar{y}_{1,2} = 0.10 \le 0.60 = \bar{y}_3 \implies$ **Satisfied.** No merge.

- **Step 3: Compare Block 3 and Block 4:**
  $\bar{y}_3 = 0.60 > 0.55 = \bar{y}_4 \implies$ **Violation!**
  Merge $B_3$ and $B_4$:
  $$
  B_{3,4} = \{3, 4\}, \quad W_{3,4} = 1 + 1 = 2
  $$
  $$
  \bar{y}_{3,4} = \frac{1(0.60) + 1(0.55)}{2} = \frac{1.15}{2} = 0.575
  $$
  Sequence of blocks: $[(B_{1,2}: 0.10), (B_{3,4}: 0.575), (B_5: 1.10)]$.

- **Step 4: Backwards check from Block $(3,4)$:**
  Compare $\bar{y}_{1,2} = 0.10$ and $\bar{y}_{3,4} = 0.575$.
  $0.10 \le 0.575 \implies$ **Satisfied.**

- **Step 5: Compare Block $(3,4)$ and Block 5:**
  $\bar{y}_{3,4} = 0.575 \le 1.10 = \bar{y}_5 \implies$ **Satisfied.** No merge.

PAVA output before boundary clipping:
$$
\mathbf{p} = (0.10, \, 0.10, \, 0.575, \, 0.575, \, 1.10)
$$

**(c)** Boundary clipping to $[0, 1]$:
Apply $p_k^* = \max(0, \min(1, p_k))$:
- $p_1^* = \min(1, \max(0, 0.10)) = 0.10$.
- $p_2^* = \min(1, \max(0, 0.10)) = 0.10$.
- $p_3^* = \min(1, \max(0, 0.575)) = 0.575$.
- $p_4^* = \min(1, \max(0, 0.575)) = 0.575$.
- $p_5^* = \min(1, \max(0, 1.10)) = 1.00$.

Final monotonic, valid conditional CDF:
$$
\mathbf{F}^* = (0.10, \, 0.10, \, 0.575, \, 0.575, \, 1.00)
$$
Every value lies in $[0, 1]$, and $F^*(z_1) \le F^*(z_2) \le F^*(z_3) \le F^*(z_4) \le F^*(z_5)$. $\blacksquare$

---

### Exercise 9.11: Destructuring of Spatial Covariance at Extreme Cutoffs
In environmental risk assessment, scientists frequently study extreme contamination thresholds (e.g., $99.9\text{-th}$ percentile). A celebrated mathematical phenomenon discovered by Georges Matheron is the **destructuration of indicator fields**: as the threshold $z_c$ approaches extreme tails ($z_c \to \infty$ or $z_c \to -\infty$), spatial continuity rapidly vanishes, and the indicator semivariogram degenerates into pure white noise (pure nugget).
Let $Y(\mathbf{s})$ be a standard stationary Gaussian random field with auto-correlation $\rho(\mathbf{h}) = \operatorname{Corr}(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h})) \in [0, 1)$.
Consider the indicator field $I(\mathbf{s}; y_c) = \mathbb{I}\{Y(\mathbf{s}) > y_c\}$ for large threshold $y_c \gg 0$.
**(a)** Using Mehler's formula (Exercise 9.8), show that the bivariate exceedance probability is:
$$
P(Y(\mathbf{s}) > y_c, \, Y(\mathbf{s}+\mathbf{h}) > y_c) = [1 - \Phi(y_c)]^2 + \sum_{k=1}^\infty \rho(\mathbf{h})^k \left( \int_{y_c}^\infty H_k(y) \phi(y) dy \right)^2
$$
**(b)** Prove that for $k=1$, $\int_{y_c}^\infty H_1(y) \phi(y) dy = \int_{y_c}^\infty y \phi(y) dy = \phi(y_c)$.
**(c)** For large $y_c$, approximate the indicator covariance by its first Hermite term:
$$
C_I(\mathbf{h}; y_c) \approx \rho(\mathbf{h}) [\phi(y_c)]^2
$$
and compute the indicator correlation:
$$
\rho_I(\mathbf{h}; y_c) = \frac{C_I(\mathbf{h}; y_c)}{P(Y > y_c)(1 - P(Y > y_c))}
$$
**(d)** Using Mills' ratio $\lim_{y_c \to \infty} \frac{1 - \Phi(y_c)}{\phi(y_c) / y_c} = 1$, evaluate the limit of $\rho_I(\mathbf{h}; y_c)$ as $y_c \to \infty$ for any fixed lag $\mathbf{h} \ne \mathbf{0}$ where $\rho(\mathbf{h}) < 1$.
**(e)** What does this imply for mapping extreme natural hazards?

##### Solution to Exercise 9.11
**(a)** Expand the indicator function $\mathbb{I}\{y > y_c\}$ in terms of Hermite polynomials:
$$
\mathbb{I}\{y > y_c\} = \sum_{k=0}^\infty c_k H_k(y)
$$
where $c_k = \int_{-\infty}^\infty \mathbb{I}\{y > y_c\} H_k(y) \phi(y) dy = \int_{y_c}^\infty H_k(y) \phi(y) dy$.
For $k=0$, $H_0(y) = 1$, so $c_0 = \int_{y_c}^\infty \phi(y) dy = 1 - \Phi(y_c)$.
By the orthogonality of Hermite polynomials under Mehler's expansion (Exercise 9.8(c)):
$$
\operatorname{E}[I(\mathbf{s}; y_c) I(\mathbf{s}+\mathbf{h}; y_c)] = \sum_{k=0}^\infty c_k^2 \rho(\mathbf{h})^k = [1 - \Phi(y_c)]^2 + \sum_{k=1}^\infty \rho(\mathbf{h})^k \left( \int_{y_c}^\infty H_k(y) \phi(y) dy \right)^2
$$

**(b)** For $k=1$, $H_1(y) = y$.
$$
\int_{y_c}^\infty y \phi(y) dy = \int_{y_c}^\infty y \frac{1}{\sqrt{2\pi}} e^{-y^2/2} dy = \left[ -\frac{1}{\sqrt{2\pi}} e^{-y^2/2} \right]_{y_c}^\infty = 0 - \left( -\phi(y_c) \right) = \phi(y_c)
$$

**(c)** Indicator covariance:
Subtracting the product of the means $\operatorname{E}[I]^2 = [1 - \Phi(y_c)]^2$:
$$
C_I(\mathbf{h}; y_c) = \sum_{k=1}^\infty \rho(\mathbf{h})^k c_k^2 \approx \rho(\mathbf{h}) c_1^2 = \rho(\mathbf{h}) [\phi(y_c)]^2
$$
The variance is $\operatorname{Var}(I) = p(1 - p)$, where $p = 1 - \Phi(y_c)$. As $y_c \to \infty$, $1 - p \to 1$, so $\operatorname{Var}(I) \approx 1 - \Phi(y_c)$.
The indicator correlation is:
$$
\rho_I(\mathbf{h}; y_c) = \frac{C_I(\mathbf{h}; y_c)}{\operatorname{Var}(I)} \approx \rho(\mathbf{h}) \frac{[\phi(y_c)]^2}{1 - \Phi(y_c)}
$$

**(d)** Applying Mills' ratio:
For large $y_c$, $1 - \Phi(y_c) \sim \frac{\phi(y_c)}{y_c}$.
Substituting this asymptotic equivalence:
$$
\frac{[\phi(y_c)]^2}{1 - \Phi(y_c)} \sim \frac{[\phi(y_c)]^2}{\phi(y_c) / y_c} = y_c \phi(y_c) = y_c \frac{1}{\sqrt{2\pi}} e^{-y_c^2/2}
$$
Now take the limit as $y_c \to \infty$:
$$
\lim_{y_c \to \infty} y_c e^{-y_c^2/2} = 0
$$
Therefore:
$$
\lim_{y_c \to \infty} \rho_I(\mathbf{h}; y_c) = \rho(\mathbf{h}) \cdot 0 = 0 \quad \forall \mathbf{h} \ne \mathbf{0}
$$

**(e)** Practical implications:
As the threshold moves into the extreme tails, the spatial correlation between indicator variables collapses to zero at all positive lags ($\rho_I(\mathbf{h}) \to 0$ for all $\mathbf{h} \ne \mathbf{0}$).
- The indicator semivariogram $\gamma_I(h) = 1 - \rho_I(h)$ turns into a flat horizontal line: a pure nugget effect!
- Spatial continuity dissolves: extreme high-grade spikes (or extreme toxic pockets) appear spatially isolated and uncorrelated with neighboring points.
- This proves that **Indicator Kriging cannot reliably map extreme tail hazards**: in the extreme tails, IK defaults to predicting the global proportion $p$, providing zero spatial resolution. Extreme value geostatistics (max-stable processes) must be employed instead. $\blacksquare$

---

## Tier III: Algorithmic Translation & Labs in Pure Python

### Exercise 9.12: Pure Python Pool-Adjacent-Violators Algorithm (PAVA) Engine
Implement a robust, dependency-free Python function to perform isotonic regression via the Pool-Adjacent-Violators Algorithm (PAVA).
**(a)** Write `pava_isotonic_regression(y_raw, weights=None)`:
- `y_raw`: list of floats $[y_1, \dots, y_K]$.
- `weights`: optional list of positive floats $[w_1, \dots, w_K]$ (default: all 1.0).
- Implements block pooling with backwards checking.
- Returns the non-decreasing sequence $p_1^* \le p_2^* \le \dots \le p_K^*$.
**(b)** Write `repair_conditional_cdf(prob_estimates)`:
- Calls `pava_isotonic_regression`.
- Clips the final result to the valid probability interval $[0, 1]$.
**(c)** Execute your engine on the corrupted probability vector from Exercise 9.10:
$$
\widehat{\mathbf{F}} = [0.25, -0.05, 0.60, 0.55, 1.10]
$$
and on an extreme case with three alternating violations:
$$
\widehat{\mathbf{F}}_{\text{stress}} = [0.80, 0.20, 0.90, 0.30, 0.70, 0.40]
$$
Verify that both outputs are strictly monotonic and bounded in $[0, 1]$.

##### Solution to Exercise 9.12
Here is the complete, self-contained Python script:

```python
def pava_isotonic_regression(y_raw, weights=None):
    """
    Performs weighted isotonic regression via the Pool-Adjacent-Violators Algorithm (PAVA).
    Guarantees p[0] <= p[1] <= ... <= p[K-1].
    """
    K = len(y_raw)
    if weights is None:
        weights = [1.0] * K
        
    # Each block is a dict: {'indices': [idx], 'weight': w, 'val': y}
    blocks = []
    for i in range(K):
        blocks.append({
            'indices': [i],
            'weight': float(weights[i]),
            'val': float(y_raw[i])
        })
        
    i = 0
    while i < len(blocks) - 1:
        # Check if violation exists between block i and block i+1
        if blocks[i]['val'] > blocks[i + 1]['val']:
            # Merge block i and block i+1
            w_sum = blocks[i]['weight'] + blocks[i + 1]['weight']
            v_pooled = (blocks[i]['weight'] * blocks[i]['val'] + 
                        blocks[i + 1]['weight'] * blocks[i + 1]['val']) / w_sum
            
            merged_block = {
                'indices': blocks[i]['indices'] + blocks[i + 1]['indices'],
                'weight': w_sum,
                'val': v_pooled
            }
            
            # Replace blocks i and i+1 with merged_block
            blocks[i] = merged_block
            blocks.pop(i + 1)
            
            # Backwards check: if i > 0, we must step back to ensure blocks[i-1] <= blocks[i]
            if i > 0:
                i -= 1
        else:
            i += 1
            
    # Unpack block values back to array of length K
    p_result = [0.0] * K
    for blk in blocks:
        for idx in blk['indices']:
            p_result[idx] = blk['val']
            
    return p_result

def repair_conditional_cdf(prob_estimates, weights=None):
    """
    Projects raw Indicator Kriging probabilities onto the valid CDF cone:
    0 <= F(z1) <= F(z2) <= ... <= F(zK) <= 1.
    """
    # 1. Monotonic projection via PAVA
    p_mono = pava_isotonic_regression(prob_estimates, weights)
    # 2. Probability boundary clipping
    p_valid = [max(0.0, min(1.0, val)) for val in p_mono]
    return p_valid

# Execution tests
print("=== Test 1: Exercise 9.10 Vector ===")
raw_1 = [0.25, -0.05, 0.60, 0.55, 1.10]
repaired_1 = repair_conditional_cdf(raw_1)
print(f"Raw Input     : {raw_1}")
print(f"Repaired CDF  : {[round(v, 4) for v in repaired_1]}\n")

print("=== Test 2: Stress Test (Alternating Violations) ===")
raw_stress = [0.80, 0.20, 0.90, 0.30, 0.70, 0.40]
repaired_stress = repair_conditional_cdf(raw_stress)
print(f"Raw Input     : {raw_stress}")
print(f"Repaired CDF  : {[round(v, 4) for v in repaired_stress]}")

# Verification of monotonicity
def check_monotonic(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

print(f"Test 1 Monotonic? {check_monotonic(repaired_1)}")
print(f"Test 2 Monotonic? {check_monotonic(repaired_stress)}")
```

Output:
```text
=== Test 1: Exercise 9.10 Vector ===
Raw Input     : [0.25, -0.05, 0.60, 0.55, 1.1]
Repaired CDF  : [0.1, 0.1, 0.575, 0.575, 1.0]

=== Test 2: Stress Test (Alternating Violations) ===
Raw Input     : [0.8, 0.2, 0.9, 0.3, 0.7, 0.4]
Repaired CDF  : [0.5, 0.5, 0.575, 0.575, 0.575, 0.575]

Test 1 Monotonic? True
Test 2 Monotonic? True
```

**Discussion:**
The algorithm matches the hand calculation from Exercise 9.10 exactly ($[0.10, 0.10, 0.575, 0.575, 1.00]$).
In the stress test, the algorithm merges violently alternating waves into monotonic pooled clusters ($[0.50, 0.50, 0.575, 0.575, 0.575, 0.575]$). Monotonicity is verified in all cases. $\blacksquare$

---

### Exercise 9.13: Simulation Lab: Monte Carlo Demonstration of Lognormal Back-Transformation Bias
In this lab, you will empirically verify Jensen's inequality and quantify the exact magnitude of the naive back-transformation bias.
**(a)** Consider a 1D Gaussian random field $Y(s)$ with mean $\mu_Y = 0.0$, unit variance $\sigma_Y^2 = 1.0$, and correlation $\rho = 0.70$ between two locations $s_1$ and $s_0$.
**(b)** The target is the physical concentration $Z_0 = \exp(Y_0)$. The observation is $Z_1 = \exp(Y_1)$, from which we compute $Y_1 = \ln Z_1$.
**(c)** The Simple Kriging predictor of $Y_0$ is $\widehat{Y}_0 = \rho Y_1$, with Simple Kriging variance $\sigma_{\text{SK}}^2 = 1 - \rho^2 = 1 - 0.49 = 0.51$.
**(d)** Simulate $N = 50,000$ bivariate Gaussian pairs $(Y_1, Y_0) \sim \mathcal{N}_2\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1.0 & 0.7 \\ 0.7 & 1.0 \end{pmatrix}\right)$.
**(e)** For each realization, compute:
1. True physical value $Z_0 = \exp(Y_0)$.
2. Naive predictor $\widehat{Z}_{\text{naive}} = \exp(\widehat{Y}_0) = \exp(\rho Y_1)$.
3. Exact Lognormal Kriging predictor $\widehat{Z}_{\text{LK}} = \exp\left( \rho Y_1 + \frac{1}{2}(1 - \rho^2) \right)$.
**(f)** Compute the Monte Carlo sample averages of $Z_0$, $\widehat{Z}_{\text{naive}}$, and $\widehat{Z}_{\text{LK}}$. Confirm that the sample average of $\widehat{Z}_{\text{LK}}$ matches the true mean $\operatorname{E}[Z_0] = e^{0 + 0.5} \approx 1.6487$, while $\widehat{Z}_{\text{naive}}$ exhibits an underestimation bias of exactly $1 - e^{-0.51/2} \approx 22.5\%$.

##### Solution to Exercise 9.13
Here is the Python simulation script:

```python
import math

def run_lognormal_bias_lab():
    N = 50000
    rho = 0.70
    sigma_sk2 = 1.0 - rho**2 # 0.51
    
    # Deterministic LCG pseudo-random generator
    seed = 123456789
    def lcg():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return seed / 0x7FFFFFFF
        
    z0_true_list = []
    z_naive_list = []
    z_lk_list = []
    
    for _ in range(N):
        # Generate two independent standard normals via Box-Muller
        u1 = max(1e-10, lcg())
        u2 = lcg()
        z1_norm = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        
        u3 = max(1e-10, lcg())
        u4 = lcg()
        z2_norm = math.sqrt(-2.0 * math.log(u3)) * math.cos(2.0 * math.pi * u4)
        
        # Correlated Gaussian pair (Y1, Y0)
        y1 = z1_norm
        y0 = rho * z1_norm + math.sqrt(1.0 - rho**2) * z2_norm
        
        # True concentration
        z0 = math.exp(y0)
        z0_true_list.append(z0)
        
        # Predictors based on observed y1:
        y_hat_sk = rho * y1
        
        # 1. Naive back-transformation
        z_naive = math.exp(y_hat_sk)
        z_naive_list.append(z_naive)
        
        # 2. Exact unbiased Lognormal Kriging back-transformation
        z_lk = math.exp(y_hat_sk + 0.5 * sigma_sk2)
        z_lk_list.append(z_lk)
        
    mean_true = sum(z0_true_list) / N
    mean_naive = sum(z_naive_list) / N
    mean_lk = sum(z_lk_list) / N
    
    theo_mean = math.exp(0.5) # 1.64872
    theo_naive_mean = math.exp(0.5 * (1.0 - sigma_sk2)) # exp(0.5 * 0.49) = exp(0.245) = 1.27762
    theo_pct_bias = (math.exp(-0.5 * sigma_sk2) - 1.0) * 100.0 # (exp(-0.255) - 1) * 100 = -22.51%
    
    empirical_pct_bias = ((mean_naive - mean_true) / mean_true) * 100.0
    
    print("=== Monte Carlo Verification of Lognormal Back-Transformation Bias ===")
    print(f"Number of Realizations        : {N}")
    print(f"Correlation rho               : {rho}")
    print(f"Kriging Variance sigma_SK^2   : {sigma_sk2:.4f}\n")
    
    print(f"Theoretical True Mean E[Z0]   : {theo_mean:.5f}")
    print(f"Empirical True Mean           : {mean_true:.5f}\n")
    
    print(f"Theoretical Naive Mean        : {theo_naive_mean:.5f}")
    print(f"Empirical Naive Mean          : {mean_naive:.5f}")
    print(f"Empirical Naive Bias (%)      : {empirical_pct_bias:.2f}%\n")
    
    print(f"Empirical Lognormal Kriging   : {mean_lk:.5f}")
    print(f"Discrepancy (LK vs True)      : {abs(mean_lk - mean_true):.5f}")

run_lognormal_bias_lab()
```

Output:
```text
=== Monte Carlo Verification of Lognormal Back-Transformation Bias ===
Number of Realizations        : 50000
Correlation rho               : 0.7
Kriging Variance sigma_SK^2   : 0.5100

Theoretical True Mean E[Z0]   : 1.64872
Empirical True Mean           : 1.64731

Theoretical Naive Mean        : 1.27762
Empirical Naive Mean          : 1.27648
Empirical Naive Bias (%)      : -22.51%

Empirical Lognormal Kriging   : 1.64735
Discrepancy (LK vs True)      : 0.00004
```

**Diagnostic Analysis:**
1. **Severe Underestimation of Naive Predictor:** The naive back-transformation exhibits an empirical bias of exactly $-22.51\%$, matching theoretical prediction $(e^{-0.255} - 1) \times 100\% = -22.51\%$.
2. **Unbiasedness of Lognormal Kriging:** The exact back-transformation $\exp(\widehat{Y} + \frac{1}{2}\sigma_{\text{SK}}^2)$ produces an empirical mean of $1.64735$, matching the true mean $1.64731$ with virtually zero discrepancy ($0.00004$).
3. **The Lesson for Environmental Practice:** In contaminant hydrology and mining, naively exponentiating log-kriged surfaces will consistently erase roughly a quarter of the true physical mass, with catastrophic consequences for risk compliance and mill feed planning. $\blacksquare$

---

## Tier IV: Edge-Case Diagnostics & Conceptual Reflections

### Exercise 9.14: The Multi-Gaussian Hypothesis and Bivariate Normality Checks
In Gaussian anamorphosis and Disjunctive Kriging, the transformed variable $Y(\mathbf{s}) = \Phi^{-1}(F(Z(\mathbf{s})))$ is Gaussian *by construction* at any single point $\mathbf{s}$. However, univariate Gaussianity does **not** imply that the spatial field $\{Y(\mathbf{s})\}$ is a multivariate Gaussian process!
**(a)** What is the **Multi-Gaussian Hypothesis**, and why is it impossible to prove empirically from a finite sample of spatial data?
**(b)** State the **bivariate symmetry diagnostic test**: if $(Y(\mathbf{s}), Y(\mathbf{s}+\mathbf{h}))$ is bivariate normal with correlation $\rho(\mathbf{h})$, prove that the indicator cross-covariances at symmetric upper and lower quantiles must be identical:
$$
\operatorname{Cov}\left(\mathbb{I}\{Y(\mathbf{s}) \le -y_p\}, \, \mathbb{I}\{Y(\mathbf{s}+\mathbf{h}) \le -y_p\}\right) \equiv \operatorname{Cov}\left(\mathbb{I}\{Y(\mathbf{s}) > y_p\}, \, \mathbb{I}\{Y(\mathbf{s}+\mathbf{h}) > y_p\}\right)
$$
**(c)** What physical phenomena violate bivariate normality (e.g., asymmetric spatial connectivity: continuous high-permeability geological channels surrounded by isolated low-permeability mudstones)?

##### Solution to Exercise 9.14
**(a)** The Multi-Gaussian Hypothesis:
A spatial random field $\{Y(\mathbf{s})\}$ is multivariate Gaussian if for *every* finite collection of locations $\{\mathbf{s}_1, \dots, \mathbf{s}_k\}$ ($k \ge 1$), the random vector $(Y(\mathbf{s}_1), \dots, Y(\mathbf{s}_k))'$ has a joint multivariate normal distribution.
- Applying Gaussian anamorphosis guarantees only that the marginal 1-point distribution is normal: $Y(\mathbf{s}) \sim \mathcal{N}(0, 1)$.
- It does *not* guarantee that 2-point distributions are bivariate normal, nor that 3-point distributions are trivariate normal.
- Because a single realization is observed at a finite number of points, higher-order joint distributions cannot be sampled exhaustively. Testing the multi-Gaussian hypothesis is fundamentally an unprovable inductive leap.

**(b)** Bivariate Symmetry Test:
Let $(Y_1, Y_2)$ be bivariate standard normal with correlation $\rho$. The joint density is:
$$
\phi_\rho(y_1, y_2) = \frac{1}{2\pi \sqrt{1 - \rho^2}} \exp\left( -\frac{y_1^2 - 2\rho y_1 y_2 + y_2^2}{2(1 - \rho^2)} \right)
$$
Notice that $\phi_\rho(y_1, y_2)$ is centrally symmetric around the origin:
$$
\phi_\rho(-y_1, -y_2) \equiv \phi_\rho(y_1, y_2)
$$
Now evaluate the lower quantile joint probability at $-y_p$:
$$
P(Y_1 \le -y_p, \, Y_2 \le -y_p) = \int_{-\infty}^{-y_p} \int_{-\infty}^{-y_p} \phi_\rho(u, v) du dv
$$
Perform the change of variables $u = -x, v = -w$:
$$
\int_{y_p}^\infty \int_{y_p}^\infty \phi_\rho(-x, -w) dx dw = \int_{y_p}^\infty \int_{y_p}^\infty \phi_\rho(x, w) dx dw = P(Y_1 > y_p, \, Y_2 > y_p)
$$
Subtracting the squared marginal probabilities $P(Y \le -y_p)^2 = P(Y > y_p)^2 = p^2$:
$$
\operatorname{Cov}(\mathbb{I}\{Y_1 \le -y_p\}, \mathbb{I}\{Y_2 \le -y_p\}) \equiv \operatorname{Cov}(\mathbb{I}\{Y_1 > y_p\}, \mathbb{I}\{Y_2 > y_p\})
$$
Under the multi-Gaussian model, spatial connectivity at extreme highs must perfectly mirror spatial connectivity at extreme lows.

**(c)** Physical Violations of Bivariate Normality:
In nature, spatial processes frequently exhibit **asymmetric connectivity**:
1. *Fluvial sandstone channels in petroleum reservoirs:* High permeability sand bodies form continuous, connected meandering conduits spanning kilometers. In contrast, low-permeability shales form isolated baffles. The high values are spatially connected, while the low values are disconnected.
2. *Fracture networks in hydrogeology:* Fluid flow is concentrated in interconnected fracture corridors. High hydraulic conductivities are strongly connected over long distances, whereas low conductivities are diffuse.
If an engineer transforms these fields to Gaussian and applies Multi-Gaussian simulation or Disjunctive Kriging, the model will destroy the connected high-permeability pathways (because Gaussian fields exhibit maximum entropy and maximal disconnection at extreme tails), leading to disastrously flawed fluid flow simulations. $\blacksquare$

---

### Exercise 9.15: Non-Gaussian Spatial Bimodality and Lithological Transitions
In many geological deposits and industrial brownfields, physical properties exhibit severe bimodality:
- Background country rock: $Z(\mathbf{s}) \approx 0.1$ ppm.
- Hydrothermal quartz veins: $Z(\mathbf{s}) \approx 50.0$ ppm.
The histogram exhibits two separate peaks with a complete absence of intermediate values.
**(a)** What happens if a practitioner naively applies Ordinary Kriging directly to this raw bimodal data across the contact zone between the two rock types?
**(b)** What happens if the practitioner applies Lognormal Kriging to this bimodal field?
**(c)** Explain why **Indicator Kriging** or **Truncated Pluri-Gaussian Simulation** successfully models the sharp discrete boundary without producing unrealistic intermediate values.

##### Solution to Exercise 9.15
**(a)** Failure of Ordinary Kriging:
Ordinary Kriging is a linear smoothing filter:
$$
\widehat{Z}(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i Z(\mathbf{s}_i)
$$
When predicting near the lithological boundary between the vein ($50$ ppm) and the background ($0.1$ ppm):
- The kriging algorithm averages the two neighboring values, predicting values like $15, 25, 35$ ppm across a wide transitional halo.
- In reality, rock with grade $25$ ppm does not physically exist in the deposit! The contact is a sharp, discrete structural fault.
- Linear kriging hallucinates a large volume of non-existent medium-grade ore, leading to massive metallurgical miscalculations during processing.

**(b)** Failure of Lognormal Kriging:
If log-transformation is applied:
- The values become $\ln(0.1) \approx -2.30$ and $\ln(50) \approx 3.91$.
- The log-scale prediction averages to $\approx 0.8$, which back-transforms to $e^{0.8 + 0.5\sigma^2} \approx 3$ to $8$ ppm.
- Furthermore, because of the massive variance inflation factor $\exp(\frac{1}{2}\sigma_{\text{SK}}^2)$ near the contact where log-variance is high, Lognormal Kriging can produce wildly erratic, explosive grade predictions ($> 500$ ppm), destabilizing the entire model.

**(c)** Success of Categorical / Indicator Geostatistics:
1. **Indicator Kriging:** Converts the problem into estimating lithological membership: $I(\mathbf{s}) = \mathbb{I}\{\mathbf{s} \in \text{Vein}\}$. It estimates the probability of being inside the vein $P(\text{Vein} \mid \text{data}) \in [0, 1]$. Decisions are then made categorically: assign rock to Vein ($50$ ppm) if $P \ge 0.50$, else Background ($0.1$ ppm). No intermediate values are fabricated.
2. **Truncated Pluri-Gaussian Simulation (TPS):** Uses underlying latent Gaussian fields truncated by categorical thresholds matching geological facies transitions. This respects the physical reality that different rock types have distinct covariance structures, sharp contact boundaries, and zero intermediate blending. $\blacksquare$

---

### Exercise 9.16: Conceptual Reflection: Indicator Kriging vs. Disjunctive Kriging vs. Nonparametric Copulas
Compare the three leading paradigms of nonlinear geostatistics: Indicator Kriging (IK), Disjunctive Kriging (DK), and Spatial Copulas across mathematical assumptions, handling of spatial continuity, and practical usability.
**(a)** Compare their assumptions regarding the underlying spatial bivariate distribution.
**(b)** How does the modeling of spatial continuity differ between IK (multiple threshold variograms $\gamma(h; z_k)$) and DK (single Gaussian anamorphosis variogram $\gamma_Y(h)$)?
**(c)** What are the computational strengths and weaknesses of Spatial Copulas relative to Disjunctive Kriging?
**(d)** Provide a clear practical guideline for when each method should be chosen.

##### Solution to Exercise 9.16
**(a)** Assumptions on Bivariate Distributions:
- **Indicator Kriging (IK):** **Completely non-parametric.** IK makes zero assumptions about the bivariate distribution. Each threshold $z_k$ is treated as an independent binary process.
- **Disjunctive Kriging (DK):** **Semi-parametric (Bivariate Gaussian).** DK models the univariate distribution non-parametrically via Hermite expansions, but assumes that the spatial dependence between pairs of points is fully described by a **bivariate Gaussian copula** (Mehler's expansion).
- **Spatial Copulas:** **Fully flexible.** Separates marginal distributions from the spatial dependence structure. Allows non-Gaussian copulas (e.g., Clayton, Gumbel, Student-$t$) that explicitly model asymmetric tail dependence.

**(b)** Modeling of Spatial Continuity:
- **IK:** Requires estimating and fitting $K$ distinct semivariograms $\gamma(h; z_1), \dots, \gamma(h; z_K)$, plus cross-variograms if using Cokriging. This is labor-intensive and prone to order relation violations, but allows different ranges of spatial correlation at different concentration levels (e.g., high values have shorter ranges than low values).
- **DK:** Requires fitting only a **single variogram** on the Gaussian anamorphosis variable $Y(\mathbf{s})$. The spatial continuity at all other levels is mathematically deduced via powers of the correlogram: $C_k(h) = [\rho_Y(h)]^k$. This ensures strict mathematical consistency and eliminates order relation violations, but forces the process to adhere to the multi-Gaussian symmetry constraint.

**(c)** Strengths and Weaknesses of Spatial Copulas vs. DK:
- **Strengths of Copulas:** Can capture realistic asymmetric spatial clustering (e.g., strong connectivity in extreme high values combined with independence in low values), which Disjunctive Kriging cannot do because the Gaussian copula is tail-symmetric.
- **Weaknesses of Copulas:** Computationally intractable for spatial prediction. Fitting a high-dimensional multivariate non-Gaussian copula for $n > 50$ points requires complex numerical integration or Markov Chain Monte Carlo (MCMC). In contrast, DK solves simple linear systems $\mathbf{V}\boldsymbol{\lambda} = \mathbf{c}_0$ in milliseconds.

**(d)** Practical Selection Guidelines:
1. **Choose Indicator Kriging (IK) when:**
   - The data contains extreme, erratic outliers that cannot be normalized.
   - The primary goal is estimating the probability of exceeding regulatory or toxic thresholds ($P(Z > z_c)$).
   - Spatial continuity varies radically between high and low values (e.g., small local nuggets for high grades, long ranges for low background).
2. **Choose Disjunctive Kriging (DK) when:**
   - The goal is predicting nonlinear recovery functions (e.g., recoverable metal, cutoff profit curves) on continuous fields.
   - The bivariate distribution is reasonably symmetric after Gaussian transformation.
   - You want a fast, mathematically rigorous system that guarantees valid probability bounds without order relation violations.
3. **Choose Spatial Copulas when:**
   - The scientific investigation focuses specifically on tail dependence and asymmetric physical connectivity (e.g., flood peak correlations, continuous contaminant preferential pathways).
   - Sample size is modest, or inference is restricted to bivariate spatial diagnostics. $\blacksquare$
