# Review: Chapter 7 — Multivariate Spatial Prediction and Cokriging

**File reviewed:** `ch07_cokriging.md` (the 1,135-line, 74,212-byte version last modified 19:49; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch07_exercises.md` was not reviewed. The two exercises inside §7.6 are covered under Dimension 4.
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, and a copy of the chapter's cokriging code). Literature attributions marked "from memory" are not verified.

---

## 1. Executive Summary

**Grade: 6 / 10**

The best ideas in this chapter are the right ones: the derivation of the ordinary-cokriging constraints ("the identity must hold for every value of the unknown means, so each mean's coefficient must vanish separately," L372–378), the guess-and-check Kronecker proof of autokrigeability, the clean elementary proof of the same fact in Exercise 7.2, and the "practitioner's fallacy" paragraph about spending weeks fitting a coregionalization model that cannot change the map. Three things hold it back. First, the chapter's central algorithm has a bug I could demonstrate: the cokriging code reports a prediction variance with the wrong sign on the Lagrange term (it returns $0.613$ where the exact error variance is $0.831$ on my example). Second, several claims are stated as "if and only if" or "exact necessary and sufficient conditions" when they are only sufficient, and the autokrigeability diagnostic therefore gives wrong answers in cases I could construct. Third, a large part of §7.1 repeats material already presented in Chapters 3 and 4, while the chapter never tells the student the thing that would shrink it: *cokriging is kriging with a longer data vector, and ordinary cokriging is universal kriging with block-indicator drift.*

**Highest-value changes**
1. Fix the variance line in the cokriging code (L630): it should add the multiplier, $C_{11}(\mathbf{0}) - \boldsymbol\lambda'\mathbf{c}_{0,1} + \mu_1$, as Theorem 7.3's own proof shows (L495). As written the code underestimates the variance (by 26% on my example).
2. Open §7.3 with the unification: a stacked data vector, so Simple Cokriging is Theorem 5.2.1 verbatim and Ordinary Cokriging is Theorem 6.3.1 with $\mathbf{X}$ = block indicators and $\mathbf{x}_0 = \mathbf{e}_1$. That is the classical-to-spatial bridge this section lacks, and it removes pages of re-derivation.
3. Give the student a worked numeric example showing when cokriging pays. I computed one: with the primary variable observed at two sites and the secondary on a dense line, the kriging variance at the midpoint drops from $1.66$ (primary only) to $1.25$ (cross-correlation $0.5$) to $0.32$ (cross-correlation $0.9$). With *isotopic* data the secondary weights are zero. A sentence and these three numbers make the whole chapter's message concrete.

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: yes in the mathematics, with one duplicated drift.** Everything the chapter proves is linear algebra and expectations. There is no measure theory and no special-function calculus.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L129–143 | Theorem 7.1: Cramér's theorem with complex Hermitian matrices, no proof | Medium | It repeats Chapter 4's §4.5 (which proves sufficiency). State it once, in one place, as the cited theorem, in the real symmetric case, with the density-versus-measure caveat in one sentence (see the Chapter 2 and 4 reviews). Do not present it twice. |
| L169–177 | Theorem 7.2 ("if and only if") | Medium | Wrong in the "only if" direction; see §7.1 below. |
| L1073, L196, L289 | "violates Kolmogorov Axiom 2" and "violations of the axioms of probability" | Low | A negative variance is not a violation of an axiom (same point as Chapters 2 and 4). Say "a variance cannot be negative because it is an average of squares." |
| L689–691 | Markov Model 1 stated as a conditional covariance, with the key implication asserted | Low | Add the two-line regression proof (see §7.4). |

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

Beat 2 is read as "how you would compute it." For a section with a core algorithm, that means a prose explanation of the steps; code is a bonus.

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 7.1 | Yes (explicit bridge; heavy duplication) | Yes (matrix assembly; correct) | Good idea, thin |
| 7.2 | Definitions only | **Missing** | **Missing** |
| 7.3 | **No classical bridge** | Yes, **with a bug** | One wrong example |
| 7.4 | Yes | Yes | Mischaracterizes the "variance bug" |
| 7.5 | Yes (excellent) | Diagnostic gives false answers | Good (one overclaim) |
| 7.6 | n/a | n/a | n/a |

### §7.1 Multivariate Spatial Processes and Cross-Continuity

**What works.** The motivating examples (L3, gold with silver and copper; water table with topography) are exactly the right way to open. The "two departures from classical multivariate regression" framing (L56–70: the cross-covariance can be asymmetric, and joint positive definiteness is a constraint on the whole matrix function) is clear, and Exercise 7.1 later shows its practical meaning.

**Issues**
1. **This section repeats earlier chapters.** The definition of multivariate stationarity (Definition 7.1) is Chapter 3's Definition 3.5.1. The transpose-symmetry property (Proposition 7.1) is Chapter 3's Definition 3.6.1. Cramér's theorem (Theorem 7.1) is Chapter 4's Theorem 4.5.1. The LMC (L145–183) and its admissibility theorem are Chapter 4's §4.6. The river example (L282–287) is the smokestack example of Chapter 3's §3.6. A student reading in order meets each idea two or three times, and the versions disagree in small ways (below). **Recommendation:** make Chapter 7 the single home for cross-covariances, Cramér's theorem, and the LMC (moving Chapter 3's §3.5–3.6 definitions and Chapter 4's §4.5–4.6 here, keeping only the *estimation* pieces, the cross-variogram estimators and the fitting step, in their own chapters), or conversely have Chapter 7 cross-reference and add only what is new.
2. **Two conventions for the cross-covariance in one chapter.** The bridge defines $C_{jk}(\mathbf{h}) = \operatorname{Cov}(Z_j(\mathbf{s}), Z_k(\mathbf{s}+\mathbf{h}))$ (L54); Definition 7.1 defines $C_{ij}(\mathbf{h}) = \operatorname{Cov}(Z_i(\mathbf{s}+\mathbf{h}), Z_j(\mathbf{s}))$ (L84) with the displaced variable reversed. Both satisfy $C_{ij}(\mathbf{h}) = C_{ji}(-\mathbf{h})$, but which variable is displaced decides how an asymmetry (a lead-lag) is read, which is exactly what the river example (L282–287) is about. Chapter 3 uses the first convention; §7.2's block entries ($[\boldsymbol\Sigma_{ij}]_{\alpha\beta} = C_{ij}(\mathbf{s}_{i\alpha} - \mathbf{s}_{j\beta})$, L341) follow the second. State one convention, once, and use it throughout.
3. **Notation: the number of variables is $p$ in some places and $m$ in others** (L15 vs. L29–50, L160–165); the number of structures is $K$ here, $U$ in Chapter 4; and $p$ also meant "number of drift columns" in Chapters 5–6. Pick $p$ for variables and $K$ for structures, and say so.
4. **The boxed definitions and theorems are in code fences** (L72–87, L100–116, L129–143, L167–178, L299–313, L381–398, L661–671, L826–854), so the LaTeX inside them will not render (same problem as Chapter 2).
5. **Theorem 7.2 is not an "if and only if."** Positive semidefinite $\mathbf{B}_k$ and valid $\rho_k$ are *sufficient* for validity. They are not *necessary*. I gave a counterexample in the Chapter 4 review: two exponential structures with $\mathbf{B}_1 = \operatorname{diag}(1, -0.1)$ and $\mathbf{B}_2 = \operatorname{diag}(0.5, 1)$ form a valid joint covariance although $\mathbf{B}_1$ has a negative eigenvalue (smallest eigenvalue of the joint matrix on 60 points: $+0.081$). This matters in Exercise 7.1 (below): the conclusion there is right, but the *reason* given is not.
6. **Proposition 7.1** states that cross-covariances "are generally NOT even" as if it were a property; it is a possibility (a caution). Also item 3, the Cauchy–Schwarz bound, is stated without its one-line proof ($|\operatorname{Cov}(X, Y)| \le \sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}$ at two points), and it is a *weak* necessary condition (the stronger one is the spectral condition of Chapter 4). Say which.
7. **Attribution inconsistency:** the LMC is credited to Journel and Huijbregts (1978) here (L148) and to Matheron (1982) in Chapter 4 (L978, L980). Please check and use one (from memory, Journel and Huijbregts is the usual citation; I have not verified it).
8. **The FAQ (L185–196)** is the right question ("why not fit each variogram separately?") but gives no evidence. Chapter 4's Exercise 4.3 supplies it: with exponential structures with rates $(1, 2, 1.5)$, a correlation above $\sqrt{70}/9 \approx 0.93$ is inadmissible. Cross-reference or move that example here.
9. **Beat 2 (matrix assembly, L204–275).** Correct and clear (the symmetry shortcut is well done). It is not a core algorithm for the learning goal, and it is later partially reimplemented inside the cokriging solver rather than called (L536: "reuse ... logic" followed by a copy).
10. **Beat 3, the river example.** "Zinc upstream cannot be caused by lead downstream, so $C_{12}(-\mathbf{h}) \approx 0$" uses causal language for a covariance, and the physical story (lead precipitates in 500 m, zinc travels 3000 m) does not obviously produce a positive covariance between lead at $\mathbf{s}$ and zinc 1,500 m downstream. The shifted model of Chapter 3 ($Z_2(\mathbf{s}) = \alpha Z_1(\mathbf{s} - \mathbf{v}) + \eta$) gives an asymmetric cross-covariance by construction and is the cleaner illustration. "Generalized vector autoregressive spatial kernels or advection–diffusion covariance operators" (L287) is jargon with no explanation; either explain in a sentence or say "beyond this book."
11. **The Sylvester example (L288–289)** is correct ($\det = 2\cdot3 - 5^2 = -19$) and is the right level. It would be stronger with the simplest necessary check first: the matrix $\mathbf{C}(\mathbf{0})$ must itself be a valid covariance matrix, since it is the covariance of $\mathbf{Z}(\mathbf{s})$.

### §7.2 Multivariate Sampling Geometries

**What works.** The classification (all variables at the same sites, different sites, or a sparse primary with a dense secondary) is the right vocabulary, and the stacked-vector notation (L328–341) is clean.

**Issues**
1. **Only Beat 1 is present.** L297 promises that the geometry "profoundly impacts computational complexity and prediction performance" and then never says how. Two short paragraphs would deliver on it: the system size is $N = \sum n_i$ (so it grows with the *secondary* data), and the prediction benefit depends on geometry: none in the isotopic case under the intrinsic model, large in the heterotopic case (my numbers are in the Executive Summary).
2. **The three categories are not mutually exclusive.** "Partially heterotopic (collocated)" is a special case of "heterotopic," and "collocated" is also the name of the data-at-the-target scheme in §7.4. Use "heterotopic" for the sampling design and reserve "collocated cokriging" for the §7.4 method.
3. **Definition 7.2 is in a code fence** (rendering).

### §7.3 Simple and Ordinary Cokriging Systems

**What works.** The unbiasedness constraints for ordinary cokriging (L372–378) are derived the right way. The anatomy of the bivariate system (L423–455) explains every block. The Lagrangian derivation (L457–498) is explicit and correct, and the variance formula $\sigma_{\text{OCK}}^2 = C_{11}(\mathbf{0}) - \boldsymbol\lambda'\mathbf{c}_{0,1} + \mu_1$ matches the Chapter 5 convention.

**Issues**
1. **There is no classical bridge for cokriging itself.** §7.1 has a bridge (multivariate regression to cross-covariance matrices), but §7.3 jumps into the system. The bridge is: *Simple Cokriging is Chapter 5's Theorem 5.2.1 with $\mathbf{Z}$ replaced by the stacked vector, and Ordinary Cokriging is Chapter 6's Theorem 6.3.1 with the drift matrix replaced by the block-indicator matrix $\mathbf{X}$ and the target drift $\mathbf{x}_0 = \mathbf{e}_1$.* The two systems are literally the same equations (up to the multiplier sign convention, which here follows Chapter 5 and not Chapter 6). Saying this up front lets the section drop most of the derivation and shows the student that nothing new in principle is happening: the new work is in building the larger covariance matrix.
2. **The cokriging code has a sign error in the variance (L630).** The code stores the solution vector, whose entry at index $N$ is $-\mu_1$ (the system is written with $-\boldsymbol\mu$, L388–390), sets `mu_1 = -sol[N]`, and then returns `c00 - dot_lam_c0 - mu_1`. Theorem 7.3's own proof gives $C_{11}(\mathbf{0}) - \boldsymbol\lambda'\mathbf{c}_{0,1} + \mu_1$ (L495). I ran the chapter's code on a small example (three primary and five secondary sites, one exponential structure) and compared with the error variance computed directly from the definition, $\boldsymbol\lambda'\boldsymbol\Sigma\boldsymbol\lambda - 2\boldsymbol\lambda'\mathbf{c} + C_{11}(\mathbf{0})$: the code returns $0.613$ and the exact value is $0.831$, and the multiplier works out to $\mu_1 = 0.109$ ($0.831 - 0.109 \cdot 2 = 0.613$, which confirms the sign flip). The weights and prediction are correct; only the reported variance is wrong. This is the kind of bug that makes a student distrust the whole chapter. Fix: `sigma2_ock = c00 - dot_lam_c0 + mu_1`.
3. **Beat 3, item 1 (L639–640) uses the wrong example.** "A shift in the measurement scale (e.g., centimeters instead of meters) would alter the primary prediction." Changing units is a *scaling*, and the cokriging predictor is automatically invariant to it (the weights rescale with the covariance model). What the zero-sum constraint protects against is an unknown *offset* (the unknown secondary mean $\mu_2$: an unknown datum, an unknown baseline). Say "offset."
4. **The "secondary variable as a gradient corrector" explanation (L641–647)** is a nice idea with no derivation. I checked a small case and the pattern is real: primary data at $0$ and $6$, secondary on $0, 1, \dots, 6$, target at $3$, cross-correlation $0.9$, one exponential structure: the secondary weights are $-0.52$ at $x = 0$, $+1.04$ at $x = 3$ (the collocated site), and $-0.52$ at $x = 6$, a dipole that sums to zero. With the primary-only plan and a cross-correlation of zero, all secondary weights are zero. Put this small table in the text; it makes the intuition checkable.
5. **The semivariogram form (L400–421)** is stated without derivation and no variance formula, and its cross-block is a cross-*variogram* (valid when the cross-covariance is symmetric; Chapter 3's Theorem 3.5.1). One sentence on that.
6. **Beat 2 code (L506–632):** besides the bug, it re-implements the matrix assembly that §7.1 already presented, and the explanation is the single sentence at L504. A walk-through in five numbered steps (assemble $\boldsymbol\Sigma$; add the two constraint rows and columns; build the right-hand side; solve; evaluate) would match the quality of Chapter 5's Beat 2s.
7. **No treatment of fitting the model.** To krige you need an LMC. The chapter does not say how to get one: fit the direct variograms, fit the cross-variogram, and project the $\mathbf{B}_k$ to be positive semidefinite (Chapter 4's Theorem 4.6.2, one sentence). Without this the student cannot use the chapter.
8. **Missing practical items** (compare Chapter 5): a worked numeric example; cross-validation (leave-one-out for cokriging works as for kriging, from $\boldsymbol\Sigma^{-1}$ for the simple case); and a decision guide, cokriging versus kriging with external drift (Chapter 6), for when the secondary variable is available everywhere. The latter is the natural question after §6.4's "distinction from cokriging" remark.

### §7.4 Collocated Cokriging and Markov Screening Approximations

**What works.** The motivation (a dense secondary survey makes the full system too large, so keep only the secondary value at the target) is clear, and the observation that ordinary cokriging then ignores the secondary datum entirely (Theorem 7.4) is a good, short insight.

**Issues**
1. **"Theorem 7.4" is a one-line observation** (the constraint $\sum\lambda_2 = 0$ forces a single weight to zero). Call it a remark. Also mention that the usual practical fix in the literature is *standardized ordinary cokriging*, which changes the constraints so that all weights sum to one (from memory, Goovaerts, 1998); the chapter's fix switches to simple cokriging, which requires the means to be known or estimated, and the code's docstring ("known or estimated") glosses over what plugging in estimated means does (compare Chapter 5's bias discussion).
2. **Markov Model 1's key implication is asserted (L693–696):** "this conditional independence hypothesis implies $\rho_{12}(\mathbf{h}) = \rho_{12}(\mathbf{0})\rho_{22}(\mathbf{h})$." It has a two-line proof at the ceiling: write $Y_1(\mathbf{s}_0) = \rho_{12}(\mathbf{0})Y_2(\mathbf{s}_0) + R$, with $R$ the regression residual (uncorrelated with $Y_2(\mathbf{s}_0)$); the hypothesis says $R$ is also uncorrelated with every other $Y_2(\mathbf{s})$; therefore $\operatorname{Cov}(Y_1(\mathbf{s}_0), Y_2(\mathbf{s})) = \rho_{12}(\mathbf{0})\operatorname{Cov}(Y_2(\mathbf{s}_0), Y_2(\mathbf{s}))$.
3. **MM1 is not always a valid model, and the code lets you build an invalid one.** The correlation functions $\rho_{11}$, $\rho_{22}$, and the number $\rho_{12}(\mathbf{0})$ are passed in independently. The model implies that $\rho_{11}(h) - \rho_{12}(\mathbf{0})^2\rho_{22}(h)$ must itself be a valid correlation (it is the correlation of the part of the primary variable that the secondary does not explain). I checked with exponential correlations on a 1D grid of 40 points and $\rho_{12}(\mathbf{0}) = 0.9$: primary range 1, secondary range 4 gives a smallest eigenvalue of $-4.84$ (invalid); primary range 4, secondary range 1 gives $-0.53$ (invalid); equal ranges give $+0.019$ (valid). With $\rho_{12}(\mathbf{0}) = 0.5$ and ranges 1 and 4, the smallest eigenvalue is $+0.044$ (valid). So MM1 with strong correlation effectively requires similar ranges, and the code's `max(0.0, ...)` on the variance would hide the symptom. State the condition, and say what to check.
4. **The "variance bug" (L810–812) is mischaracterized.** The text says the collocated variance "displays an artificial global depression across the entire domain, failing to reflect true spatial information voids." I tested this in a model where MM1 holds exactly (primary and secondary built so that $\rho_{12}(h) = \rho_{12}(\mathbf{0})\rho_{22}(h)$ with $\rho_{12}(\mathbf{0}) = 0.8$). Result: the collocated predictor's *reported* variance is $0.3537$, and its *exact* error variance, computed from the definition, is $0.3537$, the same. Full simple cokriging with the secondary at every grid point reaches $0.3470$, and the floor $1 - \rho_{12}(\mathbf{0})^2$ is $0.36$. Primary-only simple kriging gives $0.691$. So, when the model is right, the variance floor is legitimate: if the secondary is known exactly at the target, the primary variable is uncertain by at most $1 - \rho^2$ no matter how empty the primary data are. The real cautions are different: the floor is optimistic if MM1 is only approximately true or if $\rho_{12}(\mathbf{0})$ is estimated (a plug-in value), and the variance ignores that uncertainty. Rewrite the item accordingly, or drop "bug."
5. **Beat 3, item 2 (matrix near-singularity, L813–814)** is correct only when $|\rho_{12}(\mathbf{0})|$ is close to one (the two columns have entries $1$, $1$, and $\rho_{12}(\mathbf{0})$); say so.
6. **Beat 2 code (L719–803)** matches the formulas (standardize; build $\mathbf{R}$; solve; back-transform; variance $1 - \boldsymbol\lambda'\mathbf{r}$ in standardized units). It has no sign issue.

### §7.5 The Autokrigeability Theorem

**What works.** The question (when does adding a secondary variable provide *zero* benefit?) is exactly right. The guess-and-check proof (L867–946) is a good use of the Kronecker mixed-product property: propose $\boldsymbol\lambda = \mathbf{e}_1\otimes\boldsymbol\lambda^*$, verify the two block equations, and invoke uniqueness. The flow chart (L1006–1022) and the "practitioner's fallacy" (L1024–1028) are practical and well judged. I verified numerically that with an intrinsic coregionalization model and isotopic data the secondary weights are exactly zero and the variance equals the univariate ordinary-kriging variance.

**Issues**
1. **L824 overclaims: "the exact necessary and sufficient conditions."** Theorem 7.5 gives a *sufficient* condition (intrinsic coregionalization plus isotopic sampling). The general condition for the primary variable to be autokrigeable under isotopic sampling is weaker: every cross-covariance $C_{1j}$ must be proportional to the primary's auto-covariance $C_{11}$ (from memory, this is Wackernagel's form); the other variables may have additional structure of their own. I verified this with a model that is *not* intrinsic (a second structure present only in the secondary variable, $\mathbf{B}_2 = \operatorname{diag}(0, 1)$): the secondary weights are zero and the variance equals the univariate one ($0.32295$ both).
2. **The diagnostic function gives wrong answers in two kinds of cases (L955–1000).**
   - *Identical correlation functions.* It tests only whether the $\mathbf{B}_k$ are proportional and never uses its `rho_models` argument. Two structures with *the same* correlation function and non-proportional $\mathbf{B}_k$ add up to a single intrinsic model ($\sum\mathbf{B}_k\rho = (\sum\mathbf{B}_k)\rho$). I verified that the secondary weights are exactly zero there, but the function would return "NOT autokrigeable."
   - *The primary-only-proportional case* just described, also a false negative.
   The correct test is on the cross-covariances of the primary variable, not on proportionality of every $\mathbf{B}_k$.
3. **Theorem 7.5 states $\mathbf{B} \succeq 0$, but the proof needs $\mathbf{B} \succ 0$.** "Because the cokriging matrix is non-singular" (L928) fails if $\mathbf{B}$ is singular (two perfectly correlated variables): then the secondary weights are not unique. Exercise 7.2 correctly says "assuming $\mathbf{B}$ strictly positive definite." State it in the theorem.
4. **Order the two proofs the other way around.** Exercise 7.2's elementary proof (multiply the two block equations by $B_{21}$ and $B_{11}$, subtract, get $\mathbf{R}\boldsymbol\lambda_2 = c\mathbf{1}$, apply the zero-sum constraint, use $\mathbf{1}'\mathbf{R}^{-1}\mathbf{1} > 0$) is the pencil-and-paper version of the theorem and shows *why* it holds. I checked each step. Put it first for $p = 2$, then present the Kronecker proof as the generalization to $p$ variables. The Kronecker proof's candidate (guess) will then look motivated, not magical.
5. **The FAQ's list of when cokriging helps (L863–865)** is right but needs numbers. My checks: under *isotropic* sampling with a multi-scale model (opposite-sign cross-correlation at two scales: $\mathbf{B}_1 = \begin{pmatrix}1 & 0.9\\0.9 & 1\end{pmatrix}$ with range 3, $\mathbf{B}_2 = \begin{pmatrix}1 & -0.8\\-0.8 & 1\end{pmatrix}$ with range 1), the secondary weights are small ($\pm0.031$) and the kriging variance falls from $1.1303$ to $1.1247$, half a percent. So multi-scale structure *breaks* autokrigeability but, with isotopic sampling, buys little; the large gains come from heterotopy (the variance fell from $1.66$ to $0.32$ in my example). Say so.
6. **The practitioner's fallacy (L1025)** says the variables "share the exact same variogram range." The condition is the same *entire* correlation function (shape, range, and nugget share), not just the range. A one-line fix.
7. **The flow chart** should add the third branch (the primary's cross-covariances proportional to its own).

### §7.6 Analytical Exercises

See Dimension 4.

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **6.5 / 10**
- §7.1, §7.3 (derivations), and §7.5 are well paced, and the autokrigeability story is the most memorable in the chapter.
- §7.2 has only a Beat 1. §7.3 has no bridge for the cokriging system itself. §7.1 is slowed by repeating earlier chapters.
- **Missing signposts:** a "Chapter at a glance" ("cokriging is kriging on a longer data vector; the new ingredients are a valid cross-covariance model and a decision about when it pays"), a worked example, and a "Key ideas" recap.
- **Visible scaffolding:** "Beat 1/2/3" headings; **the L7 dedication** to Rizzo and S&G ("celebrated," "line for line") should become what the student can do after the chapter: decide whether cokriging can help, build a valid model, and solve the system.

### Dimension 2 — Algebraic transparency and scaffolding: **7 / 10**
- Strengths: the OCK constraint derivation, the Lagrangian derivation and variance formula, the Kronecker verification, and Exercise 7.2's elementary proof.
- Gaps: the Cauchy–Schwarz bound's proof; the MM1 implication; the semivariogram form's derivation; and the reason the Kronecker candidate is natural.
- **"From the Lecture Hall" callouts: none present.** Natural ones: (1) "Cokriging is kriging on a longer vector" (the unification); (2) "Why the secondary weights sum to zero" (offset, not units); (3) "The Kronecker product in one picture: when the same $n \times n$ spatial pattern is reused for every variable"; (4) "Reading a cokriging system as a regression with extra predictors."

### Dimension 3 — Structural consistency and forward referencing: **5 / 10**
- **Duplication** with Chapters 3 and 4 (cross-covariance definitions, transpose symmetry, Cramér, LMC, the lead-lag example): the biggest structural issue in the chapter.
- **Notation:** two conventions for $C_{ij}$; $p$ vs. $m$; $K$ vs. $U$; the multiplier sign (Chapters 5 and 7 use the same convention; Chapter 6 flips it).
- **Attribution:** LMC credited differently from Chapter 4.
- **Rendering:** boxed statements in code fences.
- **Forward reference:** $\mathbf{X}\boldsymbol\mu$ appears at L358 before $\mathbf{X}$ is defined at L392.

### Dimension 4 — Empirical and computational utility: **5.5 / 10**
- **Exercise 7.1** (Sylvester's criterion): the arithmetic is right: $\det = 4\cdot6 - 5^2 = -1$; eigenvalues $5 \pm \sqrt{26} \approx 10.099, -0.099$; and for $\mathbf{a} = (5, -4.1)'$ the quadratic form is $-4.14 = \lambda_2\|\mathbf{a}\|^2$ (consistent). Three issues. (i) Part (a) concludes "inadmissible" from "$\mathbf{B}$ indefinite" via Theorem 7.2's "only if," which is not a valid theorem; the clean reason is that $\mathbf{C}(\mathbf{0}) = \mathbf{B}\rho(0) = \mathbf{B}$ must itself be a covariance matrix. (ii) Part (b) asks for "a spatial point configuration where this model produces a negative *prediction* variance" but the solution exhibits a negative variance of a co-located linear combination at a single site, which is a different (and trivial) demonstration: the model is invalid at lag zero, so no configuration is needed. Either change the question or construct the cokriging system and show a negative variance (the latter would also exercise the corrected code). (iii) "Violates Kolmogorov Axiom 2" (see the calibration table).
- **Exercise 7.2** (autokrigeability for $p = 2$): correct and excellent (see §7.5).
- **Only two exercises** for a chapter with five sections, and none uses the chapter's central object, the cokriging system. Natural additions: (1) solve a 3-point, two-variable ordinary-cokriging system by hand and compare the variance with ordinary kriging (my heterotopic numbers are a template); (2) show that the secondary weight is zero in the collocated ordinary-cokriging case (Theorem 7.4) and nonzero in the standardized version; (3) check the diagnostic on a model that is intrinsic after combining structures with the same correlation function.
- **No data example**, and no cross-validation.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)
1. **Fix the variance sign in the cokriging code** (`+ mu_1`); my check: $0.613$ reported vs. $0.831$ exact.
2. **Fix Theorem 7.2's "if and only if"** (sufficient, not necessary) and base Exercise 7.1(a) on the lag-zero argument ($\mathbf{C}(\mathbf{0})$ must be positive semidefinite).
3. **Fix "exact necessary and sufficient" (L824)** and the autokrigeability diagnostic (it ignores `rho_models`; false negatives for identical correlation functions and for the primary-only-proportional case); state $\mathbf{B} \succ 0$ in Theorem 7.5.
4. **Fix the "variance bug" characterization** in §7.4 (the variance floor is legitimate under MM1; the cautions are model validity, plug-in parameters).
5. **State MM1's validity condition** ($\rho_{11} - \rho_{12}(\mathbf{0})^2\rho_{22}$ must be a valid correlation) and warn that the code does not check it.
6. **Fix the units example** in Beat 3 (offset, not scale).
7. **Fix Exercise 7.1(b)** (question and solution do not match).
8. **Use one convention for $C_{ij}(\mathbf{h})$** and state which variable is displaced.

### Should fix (pace, flow, voice)
9. **Deduplicate with Chapters 3 and 4:** one home for cross-covariances, Cramér's theorem, and the LMC.
10. **Open §7.3 with the unification** (stacked vector; ordinary cokriging as universal kriging with block-indicator drift).
11. **Add a worked numeric example** (heterotopic: variance $1.66 \to 1.25 \to 0.32$; isotopic: weights zero; multi-scale isotopic: $\approx$ 0.5% gain) and the dipole weights table.
12. **Reorder the autokrigeability proofs:** elementary $p = 2$ first (Exercise 7.2), Kronecker generalization second.
13. **Add Beats 2–3 to §7.2** (system size; when geometry matters).
14. **Say how the model is fitted** (variograms, cross-variogram, positive-semidefinite projection from Chapter 4), and add cross-validation and a cokriging-versus-external-drift decision guide.
15. **Prove the Cauchy–Schwarz bound and the MM1 implication** (two lines each).
16. **Fix the notation** ($p$ for variables, $K$ for structures; unified multiplier convention across Chapters 5–7; consistent LMC attribution).
17. **Move boxed definitions out of code fences.**
18. **Add four "From the Lecture Hall" callouts** and "Chapter at a glance" / "Key ideas."
19. **Replace "Beat" headings;** replace the L7 dedication.

### Nice to have
20. Rename "Theorem 7.4" a remark; mention standardized ordinary cokriging as the usual fix.
21. Replace "causal" language in the river example; use the shifted model of Chapter 3.
22. Say "same entire correlation function," not "same range," in the practitioner's fallacy.
23. Add the third branch to the flow chart.
24. Explain or drop "generalized vector autoregressive spatial kernels."
25. Define $\mathbf{X}$ before using it at L358.
26. Remove "Kolmogorov Axiom 2" in L196, L289, and Exercise 7.1.
27. Add two more exercises (hand-solved cokriging; Theorem 7.4 versus its standardized fix).
28. Verify the Goovaerts (1998), Journel (1999), Wackernagel (2003) citations and the LMC attribution.

---

## 6. What to Keep Exactly As It Is

- The two-departures framing of cross-covariance (asymmetry; joint positive definiteness), L56–70.
- The derivation of the ordinary-cokriging constraints: "the identity must hold for all values of the unknown means" (L372–378).
- The Lagrangian derivation and the variance formula (L457–498), plus the block-by-block anatomy (L423–455).
- The collocated-cokriging motivation (L655–657) and the observation that standard ordinary cokriging ignores a lone collocated datum.
- The guess-and-check Kronecker proof (L867–946), reordered after the elementary one.
- Exercise 7.2 in full.
- The flow chart and the practitioner's fallacy (with the corrections above).
