# Consistency Review of Chapters 0 to 13

Date: 2026-10-25. The review itself changed nothing; every item below was a finding with a suggested fix. **Sections 1 to 6 have since been applied** (status notes at the head of each section). What remains open is in `OPEN_QUESTIONS.md`, under "Decisions Made on Your Behalf" and "Still Open From the Consistency Review".

## How this was done

1. **Mechanical checks** (`scripts/check_consistency.py`, new). Across 455 numbered headings: every "Section N.N.N" and "§N.N.N" reference points to a heading that exists, every figure link works, every figure file is linked, chapter titles match `OUTLINE.md`, and the outline's section numbers match the chapters. One problem: the outline lists a §12.6.1 that the chapter's figure-organized lab does not have.
2. **Six read-through reviews in parallel** (read-only), each reading its chapters in full and checking them against the rest of the book: Chapters 0 to 2, 3 to 5, 6 to 8, 9 to 11, 12 to 13, and the project documents (`OUTLINE.md`, `NOTATION.md`, `README.md`, `MARKDOWNSTYLE.md`, `OPEN_QUESTIONS.md`). Reviewers re-derived formulas and recomputed numbers in Python where cheap. The raw reports, with file and line numbers, are in `reviews/`.
3. **My own spot checks.** I verified directly: the `\logb` and TAB bugs (the linter now catches them), the spherical-model practical range (0.8114, not 1.000), the wrong §11.8.4 references (the bootstrap is §11.8.5), the Chapter 0 and Chapter 1 promises about Chapter 4's sampling-density result, and the Chapter 13 row-count claim's location. I did not independently re-run every other reviewer claim; treat the rest as reviewer findings.

## Headline

- **The mathematics holds up.** Across Chapters 1 to 11 no reviewer found a wrong formula or a contradictory numerical result, and most formulas and shared numbers were re-derived or recomputed. The four-apartment block, the `rent_district` and `county_map` data sets, the case-control setting, and Chapter 0's simulations agree wherever they recur.
- **Real errors are concentrated in Chapters 12 and 13 and in one table in Chapter 2:** six lines of undefined LaTeX and a garbled symbol (both break the PDF build), two false statements about Vecchia's precision matrix, and one wrong "practical range." Section 1.
- **The main weakness is cross-chapter promises:** earlier chapters say what later chapters will show, and about ten of those promises are unkept or misdescribed. Section 2.
- **Notation has drifted in a few places:** one symbol with four meanings, the same quantity under two symbols in neighboring chapters, and several first-use slips. Section 3.
- **The meta documents are slightly stale,** and the appendices are promised more than the outline plans. Sections 5 and 6.

Counts: the six reviewers reported 17, 15, 20, 12, 24, and 28 findings (116 in all). After merging duplicates there are roughly 100 distinct items. By my own severity (a wrong claim, a formula, or a build break) five are HIGH, all in Section 1; the reviewers' labels differ in places (reviewer F rated some notation items HIGH). Source IDs from the raw reports are in parentheses.

---

## 1. Fix first: errors that break the build or state something false

**Status: all six items in this section were fixed on 2026-10-25.** Chapters 2, 6, 12, and 13 were edited, their code was rerun, and all chapters lint clean. Notes on the fixes: 1.3 now computes the spherical practical range in the code (0.811 phi, true range phi); 1.4 and 1.5 state the total bound $n(k+1)^2$ and the "same kind of weights" wording, and Exercise 9 now asks the reader to count nonzeros (the chapter quotes no new numbers); 1.6 breaks ties by region number with a stated lean toward south and west, prints the tie counts (44 of 144 for k = 4, 8 for k = 8), and updates every quoted nearest-neighbor value (for example Moran's I for 4 nearest is now 0.609 with z = 10.84, and the largest imaginary eigenvalue parts are 0.098 and 0.040).

| # | Where | Problem | Fix |
| :--- | :--- | :--- | :--- |
| 1.1 | ch12:933, 944, 949, 981, 984, 1024 (E-01) | `\logb` is not a LaTeX command: MathJax shows a red error and pdflatex stops. | Replace with `\log b`. (The linter now flags it.) |
| 1.2 | ch12:879, 886 (E-02) | A literal TAB ate the `\t` of `\text`, so `\lambda_{\text{par}}` renders as `λ_{extpar}`. | Retype the five occurrences. (The linter now flags TABs.) |
| 1.3 | ch02:343-356, 414, 685 (R1) | The spherical model's "practical range" is printed and tabulated as 1.000 phi, but §2.3.4 defines it as the distance where correlation is 0.05. That distance is **0.8114 phi** (verified). Exercise 6 asks for it "by solving a cubic," which contradicts the table. | Report 0.811 phi (computed with `brentq` like the other families), and say separately that the true range, beyond which correlation is exactly 0, is phi. |
| 1.4 | ch13:532 and Exercise 9 at ch13:1221 (E-03) | Claims the Vecchia precision matrix has rows with "at most $(k+1)^2$ nonzeros." False: with maximin ordering at n = 2,000, k = 3, a row reached 40 nonzeros against a bound of 16. The bound holds for the total, $n(k+1)^2$. Exercise 9 then contradicts itself (parents at most k, children at most c, row at most $(k+1)(c+1)$). | State the total bound, say rows can be longer when a point is a neighbor of many later points, and change Exercise 9 to $(k+1)(c+1)$. |
| 1.5 | ch13:532 (E-04) | Says the Vecchia weights $\mathbf{b}_i$ "are the weights $-Q_{ij}/Q_{ii}$ of Section 8.1.2." They are not: $\mathbf{b}_i$ regress on earlier points only, while $-Q_{ij}/Q_{ii}$ are full-conditional weights on all other points. | "the same kind of weights as Section 8.1.2, but sequential: only earlier points enter." |
| 1.6 | ch06:268-274, 299-300, 323, 371-372, 464, 1202-1209 (M1) | The "4 nearest" and "8 nearest" weights on the regular grid are decided by floating-point noise: 44 of 144 counties tie at the 4th-neighbor cutoff. A direct-difference distance function gives a largest imaginary eigenvalue of 0.0945 against the chapter's 0.051, so the quoted Moran's I, z, and p for those rows cannot be reproduced. | Break ties explicitly (for example `np.lexsort` on distance then index), say so in §6.2.2, rerun, and update the quoted values. |

---

## 2. Cross-chapter promises that are unkept or misdescribed

**Status: applied on 2026-10-25.** Where a later chapter did not deliver a promise, either the later chapter gained the material (Chapter 4 §4.8.5, Chapter 5 §5.6.1 leave-one-out residuals and the WLS/ML/REML row in §5.2.2, Chapter 8 resolution sentence) or the earlier wording was corrected. Appendix promises were handled by expanding `OUTLINE.md` (A.0, A.6, A.1 Woodbury, Appendix D list); the appendices themselves are not written yet.

| # | Promise | Where | What the later chapter actually does | Fix |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | "Chapter 4 shows that prediction error falls steadily as sampling gets denser." (R2, B-01) | ch00:148; ch01:501 | Chapter 4 has no experiment or exercise varying the sampling density. | Add a short experiment (kriging standard error at (0.5, 0.5) for n = 25, 100, 250, 1,000 listings, contrasted with the mean's $n_{\text{eff}}$ ceiling), or soften both sentences. |
| 2.2 | "Chapter 5 uses the leave-one-out residuals as a diagnostic." (B-02) | ch04:800, 1008 | Chapter 5's only diagnostic is the decorrelated residual $\mathbf{r}^*$. | Add a paragraph and one line of code to §5.6.1, or delete the promise. |
| 2.3 | Chapter 5 "asks how much plugging in estimated parameters understates the uncertainty." (B-05) | ch04:1008 | §5.4.5 compares three simple-kriging variances (0.895, 0.896, 0.841) and says the effect is small; no coverage experiment. | Add a small coverage experiment (the Chapter 4 §4.8.3 code nearly does it) or reword. |
| 2.4 | Chapter 5 "fixes the residual bias of Section 3.4.3" and revisits fitting by likelihood. (B-06) | ch03:7, 480, 599 | Chapter 5 never returns to Chapter 3's WLS fits or the residual-variogram bias. | Add a few sentences and one table row to §5.2 or §5.3.1 (WLS on residuals versus ML versus REML versus truth, noting the trend models differ). |
| 2.5 | The fourth-root power in minimum contrast "is the same as the variance-stabilizing fourth root in Cressie-Hawkins of Chapter 3." (B-04) | ch12:282, 371 | Chapter 3 motivates Cressie-Hawkins only by outlier resistance and reports no variance-stabilization finding. | Reword Chapter 12 ("the same kind of power transformation") or add a sentence on variance stabilization to §3.2.2. |
| 2.6 | Chapter 9 "returns to" the ecological fallacy and acts on "rates are not exchangeable." (M3) | ch06:133, 1101 | Chapter 9 never mentions the ecological fallacy; the exchangeability point is covered only loosely by the empirical-Bayes section (reviewer D judged it delivered). | Add a short paragraph to §9.2.1 or §9.6 (a county coefficient describes counties; why counts with an offset avoid the unequal-variance problem of §6.8.5), or soften Chapter 6. |
| 2.7 | §8.2.4 explains why the CAR $\sigma^2$ "cannot be compared across maps of different resolution." (E-06) | ch13:769 | §8.2.4 gives a different reason (marginal variances differ by region; edge effect) and says nothing about resolution or $\nu = 0$. | "this adds a second reason to Section 8.2.4's: the variance of the $\alpha = 1$ field is not defined in the continuum limit." |
| 2.8 | Chapter 12 calls the Leroux/CAR pixel GMRF a "good approximation" to an exponential field, "a connection Chapter 13 makes precise." (E-10) | ch12:713 | Chapter 13 says $\alpha = 1$ (the Leroux/CAR family) is $\nu = 0$, not a Matérn field with finite variance, and the Matérn match is $\alpha = 2$. An exponential field is $\nu = 1/2$, which is neither. The two statements sound opposed. | Add to §12.3.3: "this is the $\alpha = 1$ member of the family of Section 13.5: a practical smoother, not a Matérn field." |
| 2.9 | Chapter 0: Chapter 1 "defines the effective sample size in general." (R6) | ch00:437 | Chapter 1 defines it for the sample mean and then says "there is no single effective sample size." | "defines it for the sample mean and shows it depends on the question." |
| 2.10 | Chapter 11: Chapter 12 develops the clustered model with traffic crashes. Chapter 10: Chapter 12 works a Thomas pattern with 20 parents per unit area. (E-24, E-25) | ch11:1025; ch10:766 | Chapter 12's inhomogeneous cluster model is told as tissue density, and it works $\kappa = 25$. | Reword both ("with a tissue-slide story"; "derives for any $\kappa$ and $\sigma_c$"). |
| 2.11 | Chapter 13 says §4.3.4 gives the upper-left block of $\mathbf{N}^{-1}$ "in two forms." (E-08, B-11) | ch13:174-180 | §4.3.4 gives one form; the other follows by relabeling the blocks. | Add "the same formula with the blocks relabeled" and have Appendix A.1 state both. |
| 2.12 | Chapter 6's closing claim that Chapter 9 acts on the small-population problem. (M3, D) | ch06 | Delivered by §9.2.2 and the empirical-Bayes section (reviewers C and D agree on this part). | No change. |

Verified kept (no action): Part V (Chapters 0 to 2 promises to Chapter 13), Chapter 7 to Chapter 8's SAR-versus-CAR comparison, Chapter 4 to Chapter 8's leave-one-out link, Chapter 8 to Chapter 9's BYM2 and Leroux, Chapter 9's SPDE pointer, and Chapter 12's pixel-grid GMRF. The full table is in `reviews/review_E_ch12-13.md`.

---

## 3. Notation

**Status: applied on 2026-10-25.** Renames: Chapter 9 $\xi_k$/$\varsigma_k$/$z_k$, Chapter 6 $\tilde{\mathbf{A}}$ and $z_k$, $\mathbf{Z}_b$ in Chapter 2, Chapter 12 Leroux precision as $\sigma^{-2}[\ldots]$, $e_{\text{true}}$ in Chapter 3. `NOTATION.md` was updated to match (first-use sections, overload list, local symbols). The Schabenberger and Gotway concordance still needs checking against the book.

### 3.1 One symbol with several meanings, or one quantity with several symbols

| # | Problem | Where | Fix |
| :--- | :--- | :--- | :--- |
| 3.1 | **$\xi$ has four meanings:** a standard normal $\mathbf{v}_k'\mathbf{Z}$ (§6.5.1), eigenvalues of $\mathbf{D} - \mathbf{A}$ (§8.3.4), eigenvalues of $\operatorname{Var}(\mathbf{u})$ (§9.3.6), and the Leroux precision scale (§12.3.3). NOTATION lists only the first two, and §6.5.1 comes before the listed first use. (F5) | ch06:590; ch08:506-508; ch09:520-526; ch12:713, 719, 744-749, 775-795 | One letter per quantity. Suggestion: keep $\xi_k$ for the Laplacian eigenvalues, use another letter for Chapter 9's variance eigenvalues, and $z_k$ for Chapter 6's normals. |
| 3.2 | **Chapter 12 writes the Leroux precision with a scale $\xi$ "of Chapter 9,"** but Chapter 9 writes $\sigma^{-2}[\ldots]$. (E-05) | ch12:713-795 | Write $\sigma^{-2}[\ldots]$, or state $\xi = \sigma^{-2}$ and add a NOTATION row. |
| 3.3 | **The Laplacian eigenvalues are $\xi_k$ in Chapter 8 and $\lambda_k$ in Chapter 9; the ICAR simulation's normals are $z_k$ and $\zeta_k$.** (F6) | ch08:506-508; ch09 §9.3.6-9.3.7 | Make the two chapters agree. |
| 3.4 | **$\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ is $\mathbf{S}_{\mathbf{A}}$ in Chapter 6 and $\tilde{\mathbf{A}}$ in Chapter 8,** and its eigenvalues are $\lambda_i$ in Chapters 6 and 7 and $\tilde{\lambda}_k$ in Chapter 8. (M5) | ch06:417-423; ch08:299-305, 588 | Use $\tilde{\mathbf{A}}$ from §6.3.3 on and move it into NOTATION. |
| 3.5 | **Chapter 8's region letters A, B, C, D collide with the matrices $\mathbf{A}$ and $\mathbf{D}$** ($\Sigma_{AA}$, $\Sigma_{DD}$, $\mathbf{e}_D$), and its four-region map is Chapter 6's hub map with different labels. (M4) | ch08:34-50, 263-295 | Relabel Chapter 8's regions 1 to 4 in Chapter 6's order and add "the hub map of Section 6.3." |
| 3.6 | **Bold $\mathbf{Z}$** is the random-effects design matrix in §2.1.5 and the standard normal vector everywhere else; ch13:160 refers back to "the matrix $\mathbf{Z}$ there." (R8) | ch02:105-107 vs 494, 552 | Rename the mixed-model matrix (for example $\mathbf{Z}_b$) and fix ch13:160. |
| 3.7 | **The classical error variance is $\tau^2$ by the book's convention,** but Chapter 0's opening and Chapter 6's Moran null use $\sigma^2$. (R7) | ch00:3, 417-419; ch06:580, 599, 628, 636 | Use $\tau^2$ in Chapter 6's null derivation or list it as a local overload; adjust ch00:3. |
| 3.8 | **$\boldsymbol{\alpha}$** is first used in §4.5.3 (kernel ridge) but NOTATION lists §13.2.1. (F4) | NOTATION; ch04:630 | Give it a core row with both meanings and the correct first section. |
| 3.9 | **Overloads not listed in NOTATION** (F16, B-10, E-16): $J$ (four uses), $c$ (three), $r_0$ (rate multiplier in ch11 Exercise 8 versus interaction radius), $z$ (three), $k$, $\theta$ (four), bold $\boldsymbol{\omega}$ (ch13, in no table), $f$ (dispersal and Strauss density), $\nabla^2$ (Hessian versus Laplace operator), $\mathbf{C}_u$ versus $C_u(h)$, $\mathbf{M}_{11}$ to $\mathbf{M}_{22}$ and $\mathbf{M}_2$ versus Moran's $\mathbf{M}$, $\delta$ (noise ratio versus censoring indicator), "relative risk" in three senses across Chapters 9 to 11 (D5). | Various | Add rows to "Symbols With More Than One Use." |
| 3.10 | **NOTATION section "Symbols Used in One Chapter Only"** contains rows that span chapters (F17). | NOTATION:191 | Rename "Local Symbols" and say "defined where used." |
| 3.11 | **Small symbol slips:** $\mathbf{e}$ for the error vector in Exercise 7 (R13); `1(...)` instead of `\mathbb{1}` (E-13); undefined $s = n(n-1)/2$ (E-15); sibling-distance variable and docstring still named `F` (E-14); incidence matrix called B in a docstring (L6); Laplacian written $\mathbf{L}$ in Exercise 10 (L9); $y_{c,1}, \dots, z_n$ typo (L1); $c_{ij}$ and $c_i$ both in Chapter 6 (L10); $e$ as an error process then a residual vector in §3.4 (B-07). | Various | Fix each. |

### 3.2 First-use rule violations

| # | Problem | Where | Fix |
| :--- | :--- | :--- | :--- |
| 3.12 | NOTATION says $\ell$ first appears in §5.1; it appears in §0.4.4. (F2) | NOTATION; ch00:303 | "§0.4.4 (extended in §5.1)." |
| 3.13 | NOTATION says $\operatorname{area}(\mathcal{D})$ is defined in §1.2; its first use is §4.6.1. (F3) | NOTATION | Move the entry, or add a sentence to §1.2. |
| 3.14 | Chapter 6 §6.2.3 uses $\rho$, $c_{ij}$, $\sigma_j^2$ before their sections (§7.1, §8.1.1). (F7, M6) | ch06:323-324 | Reword in words, or list "§6.3.3 (in words); §7.1 (formally)." |
| 3.15 | Chapter at a Glance boxes use symbols before their sections (ch02, ch04, ch06, ch07). (F8) | ch02:12; ch04:12; ch06:3, 11; ch07:14 | Exempt the boxes in the rule, or write them in words. |
| 3.16 | Chapter 5 opens by treating $\boldsymbol{\theta}$ as new, but §3.3 introduced it. (B-03) | ch05:3 | "as in Section 3.3, we collect them in a vector." |
| 3.17 | Chapter 0 uses $r_{12}$ and $\sigma^2$ before their NOTATION entries; $C(\mathbf{h})$ (lag-vector form) is used in §2.2 but listed only with a scalar argument. (R14) | ch00:3, 291, 417-419; ch02:134-139 | Update the table. |
| 3.18 | Minor offsets: $\boldsymbol{\Sigma}^{\star}$ in §9.3.3 (claimed §9.3.4); $\hat{\mathbf{u}}$ in §9.4.2 (claimed §9.5.1); $n_T$, $n_S$ in §13.1.3 (claimed §13.6.1). (F) | NOTATION | Update. |

---

## 4. The shared data sets and repeated definitions

**Status: applied on 2026-10-25.** Chapter 8's hub map is relabeled to Chapter 6's numbering; Chapters 10 and 11 now say explicitly which draws are new; helper-function differences (`dist`, `kernel_intensity`, `fit_bt`, `sim_thomas`) are noted in the chapters. Chapter 10's symbols $\text{pop}$ and $\text{ex}$ were left (see `OPEN_QUESTIONS.md`).

| # | Problem | Where | Fix |
| :--- | :--- | :--- | :--- |
| 4.1 | **The "shared" crash model differs:** Chapter 10 uses `600*exp(-6*dist)` (mean about 199); Chapter 11 uses `exp(5.846 - 3*dist + 1.5*pop)` calibrated to 400. (D2) | ch10 `lam_crash`; ch11 §11.1.2, §11.3 | State in Chapter 11 that it re-simulates with a richer model, or change Chapter 10. |
| 4.2 | **Case-control realizations differ without comment:** Chapter 10 has 46 cases and 349 controls; Chapter 11 has 58 and 385 (five-year pooled: 296 cases). Chapter 11:1482 says "385 expected controls" where the expected count is 362.5. The relative risk at the plant is 0.404 in one place and 0.408 in another. (D3, D4, D12) | ch10 §10.7.3, §10.10.2; ch11 §11.4.3, §11.5.4, :1482 | One sentence in Chapter 11 saying it is a new draw (and pooled), a forward pointer in Chapter 10, and consistent rounding. |
| 4.3 | **Chapter 9 says county populations are "lognormal, as in Section 0.3.3"** but Chapter 0 uses a log standard deviation of 1.2 and `county_map` uses 1.0. (R9) | ch09:950 | Say "same median, smaller spread," or change `county_map` to 1.2. |
| 4.4 | **`county_map()` is byte-identical in Chapters 6 to 9, but the return values are named differently:** `dep`, `x`, `x_dep`; and `S` versus `S_c`. (M9) | ch06:216; ch07:639; ch08:378 | One naming everywhere. |
| 4.5 | **Helper functions vary across chapters** (F14): `dist` has five variants (Chapter 0 has no diagonal fix; Chapters 4 and 10 take two arguments; Chapter 11 has a one-liner); `gamma_exp` returns 0 at $h = 0$ in Chapter 2 but not in Chapter 3; `kernel_intensity` has different signatures and edge handling in Chapters 10 and 11; `fit_bt` and `sim_thomas` differ in Chapters 10 to 12. Also the early `dist` of Chapter 0 and the unexplained `1e-10` jitter before §2.5.1 (R12), and Chapter 5's warning about `dist` rounding that Chapter 1 never mentions (B-14). | Various | Pick canonical versions, document them in Appendix B, and say when a variant is deliberate. |
| 4.6 | **Edge corrections and normalizations under one symbol:** Chapter 10 uses Ripley's weights with $\operatorname{area}/(n(n-1))$; Chapters 11 and 12 use translation weights with $1/\operatorname{area}$ (already in OPEN_QUESTIONS). Reviewers also suggest a footnote that $\hat{K}$ values differ slightly (D10), that $g$ for inhomogeneous pairs reduces to Chapter 10's $g$ (D8), and that the Thomas parameters differ per chapter (D11). | ch10; ch11; ch12 | One footnote each, or give the translation weight its own symbol. |
| 4.7 | **Seeds:** Chapter 6's MAUP code uses 2050, which is Chapter 7's response seed; Chapters 10 and 11 share 2090 to 2092. Harmless, but README gives exact values. (L2, F26) | ch06:1318; README | Use 2042, or say "own seed family." |

---

## 5. Wrong or loose pointers and small wording slips

**Status: applied on 2026-10-25,** except B-15 (skipped on purpose: the text already attributes the fourth-decimal difference to rounding) and two reviewer claims that were wrong (0.404 versus 0.408; Chapter 4 and the exchanged-blocks form).

| # | Where | Problem | Fix |
| :--- | :--- | :--- | :--- |
| 5.1 | ch11:1420, 1490 (D1) | Cite §11.8.4 (Estimating the Middle) for the parametric bootstrap, which is §11.8.5 (verified). | Fix both. |
| 5.2 | ch11:496 (D6) | Cites §9.4.1; the content is in §9.4.2 (Henderson's Equations). | Fix. |
| 5.3 | ch08:34 (R10, M2) | Cites §1.5.2 for the degree $d_i$; degree is defined in §6.3.1. | Fix. |
| 5.4 | ch09:245 (D9) | Says $\mathbf{Q}$ is "of Chapter 8"; NOTATION gives §7.2.3. | "Sections 7.2.3 and 8.1." |
| 5.5 | ch08:1062 (L3); ch06:1459 (L5); ch07:70 (M8); ch07:252, 327 vs ch06:425 (M7); ch13:208 (E-07); ch13:1255 (E-22); ch13:142 (E-21); ch12:216 (E-17); ch12:389 (E-18); ch12 captions (E-19) | Assorted: a "table" that is printed output; "sparse matrices of Section 6.3" (they are in Chapter 8); the lag model's mean misdescribed; the eigenvalue bound proved twice; "formula in the outline" (readers have no outline); "the first chapter began with" (it was Chapter 0); the third meaning of $\kappa$ omitted; "$1/\kappa$ is the probability"; an unused `road_distance` definition; ASCII "Matern" in captions. | Fix each (see `reviews/`). |
| 5.6 | ch13:1110 (E-09) | Says Vecchia scales "to hundreds of thousands of points" though the chapter runs only to 20,000 and says the $O(n^2)$ neighbor search would dominate. | "with a tree-based neighbor search (not shown)." |
| 5.7 | ch12:931; ch06:967 (L8); ch07:1038 (L4); ch05:443 (B-09), 355 (B-13), 81 (B-15); ch04:304 (B-12); ch13:492 vs 1077 (E-20) | Small numeric or wording nits: expected rejections 5.8 not "a little under 7.2"; 10.16 versus the exact average total 10; $4.4$ versus $4.45 \times 10^{119}$; "charges" for the $\log\lvert\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\rvert$ term; a 0.0001 hand-versus-code gap; $\mathbf{R}^{-1}$ versus $\boldsymbol{\Sigma}^{-1}$ in $\hat{\mu}_{\text{GLS}}$; 1.23 s versus 1.21 s. | Fix each. |
| 5.8 | p-values typed two ways (45 "p-value" versus 5 "$p$-value") and "z-test" versus "$t$-test" (R16) | all chapters | Pick one form. |

---

## 6. Project documents, and decisions the appendices need

**Status: applied on 2026-10-25,** with the recommended option chosen for each decision (listed in `OPEN_QUESTIONS.md`). Not done: `references.bib` does not exist, so citation keys are not yet backed by entries. `README.md`, `OUTLINE.md`, `NOTATION.md`, and `MARKDOWNSTYLE.md` were updated.

| # | Problem | Fix |
| :--- | :--- | :--- |
| 6.1 | **Appendix A is promised more than planned.** Chapter 0 says "the matching section of Appendix A reviews the background" for each of the five tools, and lists Cholesky, Lagrange multipliers, and IRLS; Chapter 1 expects eigendecomposition facts. The planned Appendix A has only block inversion, derivatives, quadratic-form moments, conditional normals, and Kronecker/vec. (R3, F1) | Add a "review of the five tools" opener and sections for eigendecomposition, Cholesky, and Lagrange multipliers; or reword the promises. **Decision needed.** |
| 6.2 | **The notation guide and Schabenberger and Gotway concordance have two homes.** Chapters 0 and 13 and `MARKDOWNSTYLE.md` say Appendix C; the outline and `NOTATION.md` put it in `NOTATION.md`. (R5, E-11, F9) | Keep it in `NOTATION.md` and reword the three places. |
| 6.3 | **Appendix D** should also cover: the deeper theory deferred in Chapters 0 and 2 (Bochner's theorem, spectral representation), the finite-element form of the SPDE, the Spatial Durbin model, and alternatives to plug-in prediction. (R4, F) | Add to `OUTLINE.md`. |
| 6.4 | `OUTLINE.md` details that are stale: Chapter 4's running story (park noise versus neighborhood rent, F11); principle 6 and "How Each Chapter Reads" claim SymPy checks, but only Chapters 0, 4, and 5 use SymPy (F12); principle 5 says Cholesky simulation is introduced in Chapter 2 (F19); the scope omits empirical Bayes, GMRFs and the SPDE bridge, scan statistics, and Kronecker methods (F20); a lab item 12.6.1 that does not exist (F18). | Update. |
| 6.5 | `README.md` is stale: citation keys, the Spatial Durbin status, "watch Chapter 7's length," "as of 2026-10-24," mixed word-count bases, a misaligned layout row (F13, F25, F28). | Refresh; point to `OPEN_QUESTIONS.md` for citations. |
| 6.6 | Citations: `[@key]` is used in Chapters 1, 5, 9, and 13 only (13 keys), while other chapters name authors in prose; no `references.bib` yet. Chapter 4's history sentence, Chapter 7's economics literature, and about 30 other names are unattributed. (F15) | Decide a rule (keys wherever an author is named) and create `references.bib` before the first PDF build. |
| 6.7 | `MARKDOWNSTYLE.md` gaps: Chapter 0 deviates from its own skeleton ("Warm-Up Exercises") (F22); `scipy.spatial` is used in Chapter 13 though the guide bans "spatial packages" (F23, E-23, R17); `np.linalg.inv` is used in nine chapters despite "do not form an explicit inverse" (F24). | Note the exceptions in the guide. |
| 6.8 | `OPEN_QUESTIONS.md` is partly stale (a resolved item, a changed runtime) and lacks: the S&G concordance check against the book, the home of the notation guide, and the first-use exemption for Chapter at a Glance boxes. (F10, F27) | Update. |
| 6.9 | Chapter 0's tool table lists "first needed" chapters that do not match use (Cholesky and sparse matrices appear earlier) and omits Woodbury, Kronecker/vec, Newton-Fisher scoring, the Laplace approximation, and the Neumann series; Exercise 6 uses a result before Chapter 1 derives it. (R11, F21) | Edit the table and add a forward note. |
| 6.10 | Chapter 0's wording of the stationarity assumption ("depends only on how far apart") describes isotropy; Chapter 2 separates the two. (R15) | "depends only on the separation, how far and in which direction." |

---

## 7. What was checked and found consistent

- **Numbers:** the four-apartment block (variance 0.4947, $n_{\text{eff}}$ 2.021, eigenvalues 0.5074, 0.7569, 0.7569, 1.9789) in Chapters 1, 2, 4, and 5; Chapter 0's and Chapter 1's simulation tables; the `rent_district` results quoted in Chapters 3 to 5 (OLS 7.631, GLS 7.201, prediction 20.438 with standard error 0.946, simple kriging variance 0.8952, REML plug-in 0.841); `county_map` counts (528 rook and 1,012 queen nonzeros, 672 and 1,636 for Q_CAR and Q_SAR); Chapter 6 to 8 hand examples; Chapter 9's SMR extremes against Chapter 0; Chapter 12's Thomas and LGCP formulas against Chapters 10 and 11; Chapter 13's identities and SPDE statements.
- **Formulas that appear in more than one place agree:** Ord's identity (Chapters 7 and 8), $\mathbf{Q}_{\text{SAR}}$ and $\mathbf{Q}_{\text{CAR}}$, the CAR validity interval versus the SAR parameter space, the leave-one-out shortcut versus the CAR conditional mean, REML and the Henderson equations (Chapters 5, 8, 9), the Thomas $K$ and $g$ (Chapters 10 to 12), and the Matérn parameterization against Chapter 13's SPDE ($\phi = 1/\kappa$).
- **Structure:** every chapter has the Chapter at a Glance box, `---` before each section, exercises split analytical and computational, and What Comes Next; titles and headings match the outline; every figure is linked; only NumPy, SciPy, SymPy, and matplotlib are used; no legacy random calls; no geoscience stories.

---

## 8. Tools added during this review

- `scripts/check_consistency.py`: references, figures, outline, notation first-use sections.
- `scripts/lint_chapter.py` now also flags TAB characters and any TeX command not in a known list (this would have caught `\logb`), applied to display math too.
- `reviews/` holds the six raw reports; `APPENDIX_REQUIREMENTS.md` holds the compiled appendix requirements.

## 9. Suggested order of work

1. Section 1 (build-breaking errors and false claims): about an hour.
2. Section 3.1 and 3.2 (notation clashes), then regenerate NOTATION: about half a day.
3. Section 2 (add or soften promises): decide per item whether to add the missing experiment or weaken the promise. Items 2.1, 2.2, 2.3, and 2.4 each ask for a small experiment, so they are the cost-bearing ones.
4. Sections 4 and 5 (shared data and pointers): mostly one-sentence fixes.
5. Section 6 (decisions on the appendices and citations) before drafting the appendices.
