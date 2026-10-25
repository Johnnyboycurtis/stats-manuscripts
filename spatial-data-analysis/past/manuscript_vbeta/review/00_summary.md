# Cross-Chapter Review Summary and Prioritized Fix Order

**Covers:** Chapters 1–13 (main chapter files only). The README, the `chXX_exercises.md` workbooks, and Appendices A–D were not reviewed.
**Source:** the thirteen reports `ch01_review.md` … `ch13_review.md` in this folder. Every number below was computed in those reviews; this file only collects and ranks them. Where a report marks something "from memory" or "by reasoning," it is still unverified here.
**Caveat on versions:** the chapter files kept changing on disk while I worked. Chapter 1 was rewritten after my review (about 65 KB then, against 50 KB when I read it), so its line numbers and some findings may be stale. Chapters 12 and 13 are an older draft format (see §4).

---

## 1. Where the book stands

| Ch | Title | Grade | One-line verdict |
|---|---|---|---|
| 1 | Stochastic foundations | 6.5 | Good instincts; the key idea is explained four times and notation drifts |
| 2 | Covariance and variogram theory | 6 | Best teaching ideas, but three headline results are wrongly stated |
| 3 | Nonparametric estimation | 7 | Strongest early chapter; a few overclaims and an unused matrix |
| 4 | Parameter estimation | 6.5 | Best linear-models bridge; wrong exercise, no estimation workflow |
| 5 | BLUP and kriging | 7 | The heart of the book; sign and screening confusions |
| 6 | Spatial regression | 6.5 | Great opening; several confident claims fail numerically |
| 7 | Cokriging | 6 | Code bug in the central algorithm; "iff" claims are only sufficient |
| 8 | Change of support | 6 | Right story, most drift toward measure theory, several wrong "theorems" |
| 9 | Nonlinear geostatistics | 5.5 | Most confirmed errors; both algorithms flawed |
| 10 | Simulation | 6 | Theorems correct; practical claims and missing algorithms |
| 11 | Point processes | 5.5 | Math mostly right; the demo misleads, most beats missing |
| 12 | Lattice models | 5 | Right ideas, promised algorithms absent |
| 13 | Non-stationary / low-rank | 5 | Central theorem does not match its kernel; no estimation story |

**Average about 6.0.** The pattern is clear: Chapters 1–8 sit at 6–7 and are held back mainly by *correctness slips and repetition*; Chapters 9–13 sit at 5–6 and are held back mainly by *missing teaching content* (thin Beat 2 and Beat 3, no worked example, no estimation or diagnostics). The grades are against the mission, not against other textbooks.

**What is already working, book-wide.** These are worth protecting in any revision:
- The classical-to-spatial bridges are usually well chosen: variance of a linear combination → positive definiteness (Ch2); ANOVA contrast → variogram (Ch2); $(n-1)$ → $(n-p)/n$ → error contrasts (Ch4); bivariate conditioning → kriging (Ch5); OLS → whitening → GLS → universal kriging (Ch6); AR(1) → lattices (Ch12); heteroskedasticity → non-stationarity (Ch13).
- The "Wait—Why Did We Do That?" callouts are the book's best voice (Ch3, Ch9, Ch10, Ch11).
- Short, complete proofs a student can follow: Gauss–Markov by $\mathbf u$ with $\mathbf X'\mathbf u=\mathbf 0$ (Ch6), cokriging constraints (Ch7), Journel's conditioning (Ch10), positive definiteness of process convolutions as "the integral of a square" (Ch13), Mehler's formula by the generating function (Ch9).
- Worked exercises are mostly correct and well scaffolded (Ch9 Ex 9.1 and 9.3; Ch10 Ex 10.1 and 10.2; Ch8 Ex 8.1, 8.3, 8.4).
- The measure-theory ban is mostly respected; the leaks are listed in §3.

---

## 2. The recurring issues, ranked

### A. Statements that are wrong or stronger than the math supports (every chapter)

This is the largest category and the one that matters most for a book whose promise is "no black boxes." The detailed register is in §5. The recurring *shapes* of the mistake:
1. **"If and only if" where only "if" holds** (Ch2 Bochner with a density and the coregionalization model; Ch4 and Ch7 coregionalization; Ch7 "necessary and sufficient"; Ch12 SAR invertibility).
2. **Claims that sound right but fail a one-line numeric check** (Ch6 GLS efficiency gain "increases with correlation"; Ch8 "4×4×4 grid accurate to 1%"; Ch10 "random path mandatory"; Ch13 "several weeks of CPU time").
3. **Best linear predictor equated with the conditional mean** without saying "for Gaussian fields" (Ch9 Prop 9.1.1, Ch10 L110).
4. **Heuristic thresholds presented as tests** (Ch11 $|\hat L-r|>0.02$ flags random patterns 41–54% of the time per lag with about 17 points).
5. **Rhetoric standing in for numbers** ("catastrophic," "fails catastrophically," "severely"). In several places the numbers exist (Ch12 Jacobian: 0.94 vs 0.60 vs true 0.65) and would be more persuasive than the adverb.

### B. Code that is wrong, or that demonstrates nothing

- **Wrong:** Ch7 cokriging variance sign (reports 0.613, exact 0.831); Ch9 PAVA clamps before pooling (222 of 300 random vectors give a non-optimal answer); Ch9 Hermite anamorphosis is non-monotone and its inverse misplaces the largest samples by about 2 Gaussian units.
- **Promised but absent:** Ch11 "Monte Carlo CSR envelopes"; Ch12 "permutation test" and "SAR profile-likelihood estimator."
- **Demos that cannot fail or mislead:** Ch11 (the $\pm0.02$ rule); Ch13 (FRK targets all have true value exactly 0; the covariance contrast is 1.06×); Ch12 (one white-noise draw as the "null experiment").
- **Silent repairs:** Ch10 and Ch12 pivot floors, `max(0, var)`, identity fallback in Ch13 hide ill-conditioning from the student.
- **Good news:** where I could check the chapters' core code against an independent computation, it agreed (Ch10 thinning/Journel on-grid, Ch11 thinning and $\hat K$, Ch12 Moran variance, Ch13 FRK vs exact kriging to $10^{-13}$).

### C. Beat 2 and Beat 3 are missing in most later sections

| Chapters | State |
|---|---|
| 1–8 | Three-Beat mostly followed, with exceptions (Ch1 only §1.3; Ch7–8 later sections thin) |
| 9 | 7 of 8 sections miss at least one beat |
| 10 | Beat 2 missing for Cholesky and turning bands; Beat 3 thin |
| 11 | 5 of 6 sections lack Beat 2 |
| 12–13 | No Beat structure at all |

The missing content is consistently the same three things: **a worked numeric example, a diagnostic ("how do I know this worked?"), and a failure mode.**

### D. Core algorithms listed but not explained

Turning bands (Ch10), Cholesky simulation (Ch10 Beat 2), Moran permutation test and SAR estimation (Ch12), estimation of $\mathbf K$ and $\boldsymbol\Sigma(\mathbf s)$ (Ch13), covariance-parameter estimation workflow (Ch4), the nugget and leave-one-out in kriging (Ch5), $F$/$G$/$J$ estimators and BYM fitting (Ch11–12), Algorithm 1.1 (Ch1). Under the agreed rule (code optional, algorithms not), these are the highest-value additions.

### E. The same story told several times, and the connecting bridge not drawn

- **Repetition:** the single-realization idea four times (Ch1); OLS-vs-GLS four times (Ch6); the Jensen/naive-back-transform story three times (Ch9); the flow example twice (Ch10); the nugget confusion recurring (Ch2/5/8).
- **Missed unifications** (each is a half-page that removes pages elsewhere):
  - cokriging = kriging on a stacked vector; ordinary cokriging = universal kriging with block-indicator drift (Ch7);
  - a block average is Chapter 1's sample mean over a continuum, and its long-block limit is the effective sample size (Ch8);
  - indicator kriging *is* the linear probability model that Chapter 9's own bridge criticizes;
  - GLS is OLS on whitened data (Ch4);
  - a process convolution on knots *is* a low-rank basis expansion (Ch13);
  - the inhomogeneous Poisson likelihood is Poisson regression on cell counts; the LGCP calculation is Chapter 9's lognormal calculation (Ch11);
  - 1D AR(1) is both a SAR-type and a CAR-type model (Ch12).

### F. Mission-fit leaks (vocabulary and voice)

- **Measure-theory vocabulary,** concentrated and fixable: $(\Omega,\mathcal F,P)$, "complete probability space," $L^2(\Omega,\mathcal F,P)$, $P$-a.s., "measurable" (Ch9, Ch10); mean-square limits and Dirac-delta nuggets (Ch8); Dirac-delta white noise and $W(\mathrm d\mathbf u)$ (Ch13); Slivnyak/Palm distributions (Ch11); Hammersley–Clifford named, not explained (Ch12); KKT (Ch9); mixing-condition CLT (Ch8).
- **The dedication paragraph** ("Following the pedagogical architecture of Maria L. Rizzo and … Schabenberger & Gotway") opens most chapters (confirmed in 9–11; present in earlier chapters per my notes). It tells the reader whom the author admires, not what they will learn. Replace everywhere with a one-sentence promise.
- **Visible "Beat 1 / Beat 2 / Beat 3" headings.** Keep the rhythm, drop the labels; let transitions carry it.
- **No "From the Lecture Hall" boxes** in most chapters (the mission asks for them). Every report lists specific candidates.
- **Voice:** warm at chapter openings and in callouts, then definition–theorem–proof. The warm voice should reach the middle of each chapter.

### G. Format drift between chapters

- Boxed definitions/theorems inside code fences that will not render LaTeX (Ch2, Ch7).
- Chapters 12 and 13 use a different format (no numbered theorems, `##` headings, no worked exercises, no Beat structure) and are dated before the rest. Chapter 11 has no in-chapter exercise section though Chapters 9–10 do.
- Notation drifts across the book: $\Phi$, $\phi$, $\psi$, $\lambda$, $\mathbf W$, $\Sigma$, $n$, $k$, $K$ each carry two to three meanings (documented per chapter; worst in Ch1, Ch9, Ch12, Ch13). The multiplier-sign convention for kriging differs between Chapters 5/7 and 6/9 and is not stated at first use. "CSR" means a point process in Ch11 and "no autocorrelation" in Ch12.
- Ch1 duplicates the nugget/measurement-error explanation that Chapter 2 and Chapter 5 get wrong in different ways.

### H. Cross-chapter links that exist only in the reviewer's head

Back-references (e.g. Ch10 → Ch9 destructuration, Ch8 → Ch1, Ch12 → Ch6 residual testing) and forward references (Ch9 → Ch10, Ch12 → Ch13) are mostly absent. The reports list them per chapter.

---

## 3. Prioritized fix order

The order below weights three things: how likely a student is to *reproduce* the error, how many later chapters depend on the fix, and how cheap the fix is. Items inside a tier are roughly in the order I would do them.

### Tier 0 — Correctness fixes (small, mostly one-line or one-paragraph; do first)

Fix these before any rewriting; they cost little and otherwise propagate.
1. **Ch9 PAVA:** pool first, then clamp (code L461 and theorem step 2). **Ch9 Thm 9.2.3:** sign of $\mu_L$ under the stated convention; drop "UMVUE/conditionally unbiased." **Ch9 Hermite code:** stop inverting the polynomial by bisection; use normal scores.
2. **Ch7 code:** add the multiplier to the cokriging variance (L630). **Ch7:** change "iff / necessary and sufficient" to "sufficient" where it is only that.
3. **Ch2:** Bochner (density vs spectral measure), the sub-quadratic growth theorem, and the spherical dimension-barrier argument. These are headline results.
4. **Ch5:** the Lagrange-multiplier sign claim (and the invented "dissimilarity flips the sign" story); separate SK screening (exact) from OK screening (approximate).
5. **Ch4:** Exercise 4.2(c) MSE formula and conclusion; reversed pair-count claim; reversed log-determinant intuitions.
6. **Ch6:** the OLS-vs-GLS message (the big effect is on standard errors, not on efficiency), Exercise 6.3 / KED explanation, the IRF-$k$ "origin invariance" claim.
7. **Ch8:** Exercise 8.2 ($\bar\gamma(W,W)=6.75$, not 6.3125), the block-kriging corollary identity, Cartier for a fixed sub-block, the 4×4×4 grid claim, the affine "negative grade" warning.
8. **Ch3:** Cressie–Hawkins unbiasedness claim; the wrong constant; state the constant-mean assumption.
9. **Ch10:** "random path mandatory," "truncation inflates variance," the connectivity overclaim, Exercise 10.3(c). **Ch11:** the LGCP $g\ge1$ condition ($C_Y\ge0$), "strictly unbiased." **Ch12:** Geary range, SAR "iff," ICAR rank for disconnected maps. **Ch13:** kernel/theorem mismatch, "several weeks," validity of $C_0$ in $\mathbb R^k$.
10. **Ch9 and Ch11 smaller items:** "exceedance" vs non-exceedance (Ch9 IK anatomy); $\gamma_I$ "equals" (Ch9); Definition 11.3.2 (Ch11); SMA range statement and the $\mathbf C=\phi\mathbf W$ slip (Ch12).

### Tier 1 — Deliver what the chapters promise (high value per hour)

11. **Ch11:** replace the $\pm0.02$ diagnosis with Monte Carlo envelopes (promised at L497).
12. **Ch12:** add the SAR profile-likelihood recipe and the Moran permutation test (promised at L327–328).
13. **Ch13:** fix the demos (non-zero targets or grid RMSE; contrast at a meaningful lag; label which standard error is printed).
14. **Ch10:** a three-line Cholesky recipe, the turning-bands recipe (lines, 1D process, scaling) and the unconditional-field source for Journel.
15. **Ch1:** a one-line explanation of why $\mathbf L\mathbf w$ has the right covariance (Algorithm 1.1).

### Tier 2 — Add the missing teaching content (this is where the book gets its voice)

16. **One running example per chapter, reused across its sections.** Priority order by how much each is missing: Ch9 (simulated lognormal field with known exact answer), Ch11 (three-pattern figure), Ch12 (5–8 region toy map), Ch10 (kriged curve plus 3–4 realizations), Ch13 (nonstationary field with local ellipses), Ch7 (the worked cokriging numbers: variance 1.66 → 1.25 → 0.32).
17. **Add the missing unifying paragraphs** listed in §2E. Each is about half a page and several remove pages of re-derivation.
18. **Add diagnostics and failure modes** per section (Beat 3): the specific lists are in each report's "Should fix" section. Highest priority: kriging (nugget, leave-one-out, what the variance means; Ch5), estimation workflow (Ch4), nonstationarity detection (Ch13), residual testing (Ch12), simulation checks (Ch10).
19. **Add the missing estimation steps:** $\mathbf K$ and $\boldsymbol\Sigma(\mathbf s)$ (Ch13), BYM fitting (Ch12), IK read-offs (E-type, exceedance probability) (Ch9), Poisson-regression fitting (Ch11).
20. **Bring Chapters 11–13 to the common template:** numbered statements, term-by-term anatomy for the main equations, an in-chapter worked-exercise section, an opening promise.

### Tier 3 — Mission-compliance sweep (mechanical; do in one pass across all chapters)

21. Delete the dedication paragraph everywhere; replace with a one-sentence promise.
22. Remove measure-theory vocabulary (list in §2F); replace finite-knot or plain-language versions (Ch8 nugget, Ch13 white noise, Ch9/10 probability-space lines).
23. Remove visible Beat labels; keep the order.
24. Move every boxed theorem out of code fences (Ch2, Ch7).
25. Add Lecture Hall boxes at the points each report suggests (about 4 per chapter).
26. Replace rhetorical adverbs with the numbers already computed.

### Tier 4 — Consistency and polish

27. **One notation index, applied everywhere** (`notation_index.md` exists; extend it): the specific collisions are in §2G. Decide the multiplier-sign convention once (state in Chapter 5, reuse in 6, 7, 9).
28. De-duplicate repeated explanations (§2E) after the new unifying paragraphs are in, since several repeats can then be cut rather than rewritten.
29. Add back- and forward-references (§2H).
30. Re-run each chapter's code and exercises after edits; the verification scripts used in the reviews are reproducible from the numbers in each report.

### Suggested order of work across chapters

Work in chapter order within each tier, but do Tier 0 for **Chapters 2, 5, 9 first**: Chapter 2's results are cited later, Chapter 5's conventions are reused in 6, 7, and 9, and Chapter 9 has the most confirmed errors. Do the Tier 3 sweep *after* Tier 0–2 edits in a chapter, not before, to avoid editing text that will be replaced. Do Chapter 1 last, since it is the one most likely to have changed since my review.

---

## 4. Per-chapter "if you only do three things"

| Ch | The three highest-value changes |
|---|---|
| 1 | Explain Algorithm 1.1; collapse the four single-realization explanations; fix $C(0)$/$\sigma^2$ notation (re-check against the current file) |
| 2 | Repair Bochner, sub-quadratic growth, spherical barrier; move boxed theorems out of code fences; add the labeled nugget/sill/range sketch |
| 3 | State the constant-mean assumption; use matrix $\mathbf A(\mathbf h)$ for the variance of $\hat\gamma$; correct the Cressie–Hawkins claim |
| 4 | "Whiten, then OLS"; fix Exercise 4.2(c); add the estimation workflow (starting values, REML, standard errors, profile) |
| 5 | Fix the multiplier-sign claim; separate SK and OK screening; add nugget, leave-one-out, and what the variance means |
| 6 | Correct the OLS-vs-GLS message (standard errors); fix Exercise 6.3/KED; drop the IRF-$k$ origin-invariance claim |
| 7 | Fix the variance sign in the code; open with the stacked-vector unification; add the worked example (1.66 → 1.25 → 0.32) |
| 8 | Tie blocks to Chapter 1's sample mean and effective sample size; give Krige additivity a real proof; fix Exercise 8.2 and the corollary |
| 9 | Fix PAVA order and Theorem 9.2.3; add the running lognormal example; link IK to the linear probability model |
| 10 | Fix the connectivity/random-path/truncation claims; add the kriged-vs-realizations figure; add Cholesky and turning-bands recipes |
| 11 | Replace the diagnosis rule with envelopes; add the three-pattern figure; add the Poisson-regression bridge and a worked-exercise section |
| 12 | Add the SAR estimator and permutation test; fix Geary/SAR/ICAR statements; use AR(1) to explain SAR vs CAR |
| 13 | Fix kernel/theorem mismatch and the demos; unify process convolution and low-rank; add estimation and nonstationarity diagnostics |

---

## 5. Register of confirmed errors (for tracking fixes)

"Confirmed" means I reproduced the contradiction numerically or algebraically; the evidence is in the cited report.

| Ch | Location (see report) | Issue | Evidence |
|---|---|---|---|
| 2 | Bochner; sub-quadratic growth; spherical barrier | "iff" with a density; theorem false as stated ($\gamma=ch^2$ is valid); argument does not prove $d\ge4$ invalid | Star config valid at $d=4$ (min eig $+0.67$), fails at $d=38$ |
| 3 | Cressie–Hawkins | Unbiasedness claimed; derived under independent increments | About 10% overestimate in simulation |
| 4 | Ex 4.2(c) | MSE formula and conclusion wrong | $n=50,p=6$: ML 0.0496 vs REML 0.0455 |
| 5 | Thm 5.3.3(2), FAQ | Multipliers said to have opposite signs | Both 0.045299 |
| 5 | §5.4–5.5 | SK screening exact; OK screening approximate (about 8% weight beyond flanking points) | Numeric |
| 6 | L516 | GLS efficiency gain "increases with correlation" | Gain 1–28%, not monotone; naive SE 0.69–0.12× true |
| 6 | Ex 6.3 / KED | Blow-up blamed on conditioning | Variance 0.469 at 100.005, 8,241 at 101; centering cuts condition number $4.5\times10^{12}\to4.5\times10^{4}$ |
| 6 | IRF-$k$ | "Origin invariance" advantage | UK invariant to $10^{-12}$ |
| 7 | Code L630 | Variance omits multiplier sign | 0.613 vs exact 0.831 |
| 8 | Ex 8.2 | $\bar\gamma(W,W)$ wrong | 6.75, not 6.3125 |
| 8 | Corollary 8.8.1 | Variance identity false | Numeric |
| 9 | Thm 9.5.1 and code | Clamp before pooling | 222/300 mismatches vs QP; pool-then-clamp 0/300 |
| 9 | Thm 9.2.3 | $-\mu_L$ under $\mathbf V\boldsymbol\lambda+\mu_L\mathbf 1=\mathbf c$ | $\operatorname{E}[\hat Z]/m_Z$ = 1.121 vs 1.000 with $+\mu_L$ |
| 9 | Thm 9.3.1 proof | Extra $1/(1-\rho)$ in the Mills asymptotic | Exact/chapter → 0.5 at $\rho=0.5$ |
| 9 | Hermite code | Non-monotone; inverse misplaces extremes | Error about 2.2 Gaussian units |
| 10 | L755, L615 | Random path "mandatory"; truncation "inflates variance" | Raster, 2 neighbors: lag-10 cov $-0.60$ vs $0.02$ true |
| 10 | Ex 10.3(c) | "$u_1$ screens" | Weights 0.125, 0.125, 0.276 |
| 11 | L658 | $\pm0.02$ rule | 41–54% false flags per lag at about 17 points |
| 11 | Thm 11.5.1 | $g\ge1$ needs $C_Y\ge0$ | Hole-effect models violate |
| 12 | L122, L195, L314, L279 | Geary range; SAR "iff"; ICAR rank; SMA range | Star graph $c=2.5,4.5,10.5$; $\rho=1.5,2.5$ still valid; rank 4 of 6 |
| 13 | L94 vs L102 | Kernel gives $\exp(-Q/4)$, variance $(4\pi)^{-d/2}$ | Direct integration |
| 13 | L249 | "Several weeks" | $\tfrac13n^3$ = about 9 h at 10 GFLOP/s |

**Flagged but not verified:** Ch10 breakthrough-time direction (L164; by reasoning); Ch13 triangle covariance invalid in $\mathbb R^2$ (standard result; my random test did not show it); literature attributions marked "from memory" in each report.

---

## 6. Open items

- **Chapter 1 should be re-reviewed** against its current version (the file changed after my review).
- **Appendices A–D** (`app_a.md` … `app_d.md`) were not reviewed. Suggested treatment, as agreed: an optional reference shelf; accept Matérn/Bessel as a named definition and flag only proofs that need Hankel/Bessel calculus.
- **Exercise workbooks** (`chXX_exercises.md`) were excluded by request. Several chapters defer proofs there (Ch12 Moran variance, Ch13 the main theorem, Ch4 and Ch11 derivations), so the *textbook alone* cannot yet be checked against the mission.
- **README** was set aside by request.
- **No chapter file was edited by me.** All fixes above are recommendations.
