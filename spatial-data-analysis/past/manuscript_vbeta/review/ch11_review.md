# Review: Chapter 11 — Mapped Point Patterns and Spatial Point Processes

**File reviewed:** `ch11_point_processes.md` (the 669-line, 37,317-byte version last modified 20:14:53; line numbers refer to that version)
**Lens:** readability, pace and flow for an MS-level statistics student, held to the master's-level linear-model ceiling.
**Scope note:** the companion workbook `ch11_exercises.md` was not reviewed. Unlike Chapters 9 and 10, this chapter has no in-chapter worked-exercise section (see §3).
**Review convention on code:** Python snippets are optional. What is required is that every algorithm central to a section's learning goal is explained as clearly as it realistically can be.
**Verification note:** where I say "checked" or "verified," I ran the computation myself (numpy, scipy, Monte Carlo, and the chapter's own code, exec'd from the file). Statements marked "from memory" or "by reasoning" were not tested.

---

## 1. Executive Summary

**Grade: 5.5 / 10**

The mathematics is mostly right, and the chapter opens with a good idea: *in geostatistics the places are fixed and the values are random; in point patterns the places themselves are the random thing.* The one-dimensional Poisson bridge (L13–21), the conditional-uniform theorem with its Binomial corollary (L79–129), the CSR closed forms for $F$, $G$, $K$, and the log-Gaussian Cox moment calculation (L441–473) are correct and at the right level. I checked the thinning algorithm (mean count 17.17 over 3,000 runs against the exact 17.15) and Ripley's estimator (it is approximately unbiased under CSR; details below).

The problems are about *what the chapter leaves out and how its one demonstration misleads.*

1. **The only demonstration cannot tell the student anything.** The final lab (L637–662) labels a pattern "Clustered / Regular / CSR Random" by whether $\hat L(r)-r$ exceeds $\pm0.02$. I tested that rule on *truly random* patterns: with about 17 points it calls CSR "non-CSR" at a single lag 41–54% of the time, and flags at least one of five lags 91% of the time. Even with 50 points, one lag in five or so is wrongly flagged somewhere (15%). The chapter promises "Monte Carlo CSR confidence envelopes" (L497), but the code contains none. Envelopes are the whole inferential payoff of $K$.
2. **The chapter is a catalogue of definitions.** Four of its six sections have only Beat 1. There is no Beat 2 for the quadrat test, for $F$, $G$, $J$, or for the Cox and Strauss models; the intro and taxonomy promise Neyman–Scott, Matérn, and Berman–Turner and never deliver them; there is no data set, no picture of a clustered versus regular pattern, and no worked example.
3. **The bridges are named but not built.** This is a textbook for people who know regression and Chapters 1–10. Three obvious bridges are missing: the inhomogeneous-Poisson likelihood *is* Poisson regression on cell counts; the LGCP calculation *is* Chapter 9's lognormal moment calculation; and $\hat K$ is a pair-counting estimator in the same family as Chapter 3's empirical variogram. None is mentioned.

**Highest-value changes**
1. Replace the demonstration with a three-pattern lab (CSR, clustered, regular) and **Monte Carlo envelopes** for $\hat L(r)-r$. Show that the inhomogeneous pattern also leaves the envelope. That is the honest lesson about confounding trend with clustering, and the chapter currently buries it in a print statement (L662).
2. Give every section a Beat 2 and Beat 3. At minimum: how to simulate CSR (Theorem 11.2.1 already tells you), how to estimate $F$ and $G$, how to fit an inhomogeneous Poisson model (the Poisson-regression bridge), and what "Strauss" requires to simulate and fit.
3. Remove or replace the black-box citations (Palm distribution, Slivnyak's theorem; L273) and the measure-theory vocabulary, and tell the student what the chapter *does not* cover (marks, fitting methods, goodness-of-fit).

---

## 2. The Calibration Check (is it within the linear-model ceiling?)

**Verdict: mostly yes.** The calculus is Riemann sums, Poisson and binomial probabilities, and one polar-coordinates integral. The leaks are in vocabulary, and one is a true black box.

| Where | What | Severity | Suggested handling |
|---|---|---|---|
| L273 | "By Palm distribution theory and Slivnyak's Theorem, the reduced Palm distribution of a Homogeneous Poisson Process is identical to its unconditional distribution" | **High** | A named theorem standing in for the argument. The idea is one sentence: *under CSR, disjoint regions are independent, so knowing there is a point at $\mathbf s$ tells you nothing about the points elsewhere; the rest of the pattern is still a Poisson process.* Say that, and mention "Slivnyak's theorem" in a footnote. |
| L5 | "Following Maria L. Rizzo's three-beat pedagogical rhythm … and Schabenberger & Gotway's semantic linear-model discipline" | Medium (voice) | Same dedication paragraph as every chapter. Replace with a promise (below). |
| L57 | "$\{v(K): K\subset W,\ K\text{ compact}\}$ uniquely determines the distribution" | Low | True (a standard result, from memory) but never used. State as "void probabilities describe the whole process; we use them for $F(r)$" or drop. |
| L205, L208 | "second-order factorial moment measure $\alpha_2$" | Medium | Say "the expected number of *ordered pairs* of distinct points, one in $A$ and one in $B$." Keep the product density, drop the measure. |
| L476 | "density with respect to a unit Poisson process" | Medium | "a density relative to a completely random pattern." |
| L47–50 | "simple … $P(N(\{\mathbf s\})\le1\ \forall\mathbf s\in W)=1$" | Low | In words: "no two events at exactly the same place." |

No Hilbert-space or Bessel machinery, and the proofs that exist are calculus. This is closer to the ceiling than Chapters 7–9.

---

## 3. Section-by-Section Structural Audit

### Three-Beat compliance at a glance

| Section | Beat 1 (why / bridge) | Beat 2 (how to compute) | Beat 3 (reality) |
|---|---|---|---|
| 11.1 | Yes (good bridge) | **Missing** | **Missing** |
| 11.2 | Yes | **Missing** (no simulation) | Quadrat test only; no failure modes |
| 11.3 | Yes | **Missing** | **Missing** |
| 11.4 | Yes (long) | **Missing** (no estimators for $F,G,J$) | One callout, edge correction only |
| 11.5 | Yes (thin) | **Missing** | **Missing** |
| 11.6 | n/a | Two code blocks | **Misleading** |
| Exercises | n/a | n/a | **No worked-exercise section** (Chapters 9–10 each have one) |

Five of six sections lack a Beat 2, and the one section called "Algorithmic Translation" contains the diagnostic block labelled "Beat 3: Reality" (L633–663). The chapter is also much shorter than its neighbors (669 lines against 940–1,130), and the missing material is exactly the practical half.

### Intro and §11.1 From 1D Arrival Counts to Spatial Point Measures

**What works.** The first paragraph (L3) says in three sentences what is different about this field, and the 1D Poisson bridge (L13–17) uses axioms students know. The side-by-side ASCII picture (L23–38) is useful.

**Issues**
- **The dedication paragraph (L5)** and its attributions should be replaced with a statement of what the student will be able to do: *"By the end you will be able to look at a map of events (trees, cases, vents), say whether they are random, clustered, or regular, and say how sure you are."*
- **The intro promises more than the chapter delivers.** L5 lists "Neyman–Scott cluster" processes; the taxonomy (L385–400) lists Neyman–Scott/Matérn, hard-core, and Berman–Turner quadrature. Only inhomogeneous Poisson, LGCP, and Strauss appear. Either write the cluster-process subsection (a *parents plus offspring* picture is easy to explain and is the first thing students imagine when they hear "clustered") or cut the promise.
- **"$d\ge2$" (L19, L41) removes the bridge.** The bridge starts in 1D, then the definition forbids $d=1$. Say "$d=2$ in our examples; everything holds for $d\ge1$ except the formulas that use the area of a disc."
- **Window versus whole space.** Definition 11.1.1 is a finite process in a bounded window $W$ (L41–42); Definition 11.2.1 is a process on all of $\mathbb R^d$ (L68); the theorems switch between them without saying so. A one-line note ("we observe a process on $\mathbb R^2$ through the window $W$") fixes it, and it is the origin of the edge-effect problem in §11.4.
- **No picture of what we are trying to describe.** The student has not yet seen a clustered or a regular pattern. Three scatterplots (random, clustered, regular) with the words under them make the whole chapter concrete. The classical examples are tree-location data sets (a standard teaching set, from memory); any would do.
- **Beat 3 is absent.** A natural content: "what counts as a point?" (a tree, a case, a lightning strike), "what is the window?" (the study region you chose, which affects everything), and "what if the points are not the thing of interest?" (marks; see the omissions list below).

### §11.2 CSR and the Homogeneous Poisson Process

**What works.** The conditional-uniform theorem is the right result to prove. The proof is calculus-level, the ordering factor $n!$ is handled correctly (L112–114), and part 2's binomial derivation (L118–129) is a clean exercise. The index-of-dispersion test (L135–154) is a good classical bridge: "the variance-to-mean ratio of counts is 1 for a Poisson count, so a clustered pattern has too much variance."

**Issues**
- **The proof repeats itself (L98–107).** L102 and L106 state the same joint-probability result twice. Keep one.
- **The payoff of the theorem is never stated.** "Conditional on $N(W)=n$, points are i.i.d. uniform" is *the recipe to simulate CSR*: draw $n\sim\text{Poisson}(\lambda|W|)$, then $n$ uniform points. It is also the null distribution behind every envelope in §11.4. One sentence connecting them turns a proof into a tool. The chapter's code never simulates CSR at all.
- **Quadrat test, what I checked.** I simulated CSR with a $4\times4$ grid and mean counts from 0.6 to 12.5 per quadrat. The upper-tail rejection rate was about 2.3–2.8% against a 2.5% nominal level, so the chi-square approximation is fine for detecting *clustering* even with small counts; the lower-tail (regularity) test was conservative at small counts (0.7–2%). So the test works, and the chapter can say so with numbers, plus the practical rule that the lower tail is less trustworthy when expected counts are small.
- **A missing but crucial caution.** An index of dispersion above 1 does *not* mean clustering; it also arises from a **trend** in intensity (more points in some quadrats because the intensity differs, not because points attract each other). The chapter treats $I>1$ as "clustered/aggregated" (L148) and does not mention this until a print statement at L662. This is the central interpretive pitfall of the whole field and it belongs here.
- **Other missing cautions:** the result depends on the quadrat size (try two sizes); the test uses only counts and so ignores distances; and $\chi^2=(Q-1)I$ has $Q-1$ degrees of freedom *because* we conditioned on the total $n$ (the Binomial/Multinomial result of the theorem you just proved). Saying that last point links the test to Theorem 11.2.1.

### §11.3 Intensity, Campbell's Theorem, and Second-Order Properties

**What works.** The bridge sentence (L162–163) is the right idea, and the verbal reading of the product density ("probability of a point in each of two tiny cells," L214–217) is good.

**Issues**
- **The bridge is stated but not built.** The geostatistical counterpart of $\mu(\mathbf s)$ and $C(\mathbf s_1,\mathbf s_2)$ is $\lambda(\mathbf s)$ and the *covariance density* $\lambda_2(\mathbf s_1,\mathbf s_2)-\lambda(\mathbf s_1)\lambda(\mathbf s_2)=\lambda^2[g-1]$ (for distinct locations, from the definitions at L210–222; this is what $g$ is measuring). With that one equation, "$g>1$ means clustering" becomes "positive covariance between counts in tiny cells," which is exactly what a student of Chapters 1–2 expects, and it explains why the pair correlation function plays the role of a correlogram.
- **Campbell's theorem has no use case (L180–203).** The proof is a clean Riemann-sum argument. But the student never sees it applied. Two natural uses: (a) the expected total of a quantity attached to points (total biomass of trees, $f(\mathbf s)$ = biomass density), and (b) it is the first step in showing that $\hat\lambda=n/|W|$ is unbiased. Add one worked sentence.
- **Definition 11.3.2 contradicts itself in passing (L206–209).** It says "for disjoint regions $A$ and $B$" and then writes $\operatorname{E}[N(A)N(B)-N(A\cap B)]$. For disjoint regions the second term is zero. Either drop "disjoint" (the formula is the general one) or drop the correction term.
- **"Second-order stationary" is used (L224) but never defined.** Say: $\lambda$ is constant and $\lambda_2$ depends only on the difference of the two locations, the point-process version of Chapter 1's second-order stationarity.
- **No Beat 2/3.** The estimator of $g(r)$ (a kernel-smoothed histogram of pair distances divided by $2\pi r\lambda^2|W|$-type normalization; from memory) is not given, and $g$ is the most informative summary. If it is out of scope, say so.

### §11.4 Distance Summary Functions $F,G,J$ and Ripley's $K$

**What works.** The CSR closed forms are right (and the proof of $F$ is a model one-liner: "no point in a disc of area $\pi r^2$," L262–271). The $K$-function definition and the identity $K(r)=2\pi\int_0^ru\,g(u)\,du$ are correct, and $L(r)-r$ is the right thing to plot. Ripley's estimator (L338–343) has the right form and weights.

**Issues**
- **The $G$ proof (L272–277) is the one black box in the chapter.** See §2.
- **$F$, $G$, $J$ are defined but never computed.** Estimating them needs a grid of test locations for $F$ and, for both, an edge correction (border/reduced-sample, or Kaplan–Meier-type; from memory), none of which is explained. Either add a short recipe or reduce the section to $K$, which is the one function that has code. Right now the student can define a $J$-function (L279–295) but cannot compute one. The statement "to eliminate sensitivity to estimated intensity" (L280) is also loose: $J$ compares two distance distributions and is independent of $\lambda$ in the sense that $1-F$ and $1-G$ scale together under CSR; say that.
- **A unifying picture is missing.** $F$ asks "how far from a random *place* to the nearest point?"; $G$ asks "how far from a *point* to its nearest neighbor?"; $K$ asks "how many points within $r$ of a point?" Three sentences (and one picture) tell the student which question each one answers, and why $G>F$ means clustering (points have neighbors nearby, but random places do not).
- **"Strictly unbiased" (L336) is more than is shown.** The estimator is a ratio, so exact unbiasedness is not available in general, and the weights break down once circles leave the window altogether. I tested it under CSR in the unit square with the chapter's weights (including the 360-point approximation and the 0.1 floor, replicated exactly in numpy; it reproduced the chapter's output): the mean of $\hat K(r)/(\pi r^2)$ was 1.004, 1.005, 1.003, 1.001, 0.999 at $r=0.05,0.1,0.2,0.3,0.4$ with about 17 points, and 0.996–1.004 with about 50. So "approximately unbiased under CSR for $r\le0.4$ in the unit square" is supported; "strictly unbiased" is not shown. The anatomy line (L368) that $n(n-1)/|W|^2$ is an unbiased estimator of $\lambda^2$ *for a Poisson process* is correct and worth saying exactly like that (it is where the $n(n-1)$ comes from).
- **The "up to 70%" claim (L375) is unexplained.** The fraction of points within $r$ of the boundary of a unit square is $1-(1-2r)^2$: 36% at $r=0.1$, 64% at $r=0.2$, 75% at $r=0.25$. Quote one or two of those instead of "up to 70%." Also, "maintaining minimal variance" is stronger than anything shown; say "usually lower variance."
- **The edge-correction figure (L345–362) does not draw the idea.** Use a simple sketch: a circle around $\mathbf s_i$ through $\mathbf s_j$, with the part outside $W$ shaded, and the weight $=$ unshaded arc length over total circumference.
- **Beat 3 is limited to edge effects.** The central pitfalls are missing: (a) **$K$ confounds trend and clustering** (see §11.2); the inhomogeneous version of $K$ (from memory, due to Baddeley–Møller–Waagepetersen) exists for exactly this reason; (b) **pointwise envelopes are not a global test** (looking at many $r$ at once inflates false alarms; I saw this directly, see §11.6); (c) **$K$ does not identify the process** (different processes can share a $K$); (d) **edge effects grow with $r$**, so only trust $r$ up to a fraction (often a quarter to a half) of the window's short side (from memory).

### §11.5 Contagion, Clustering, and Inhibition Models

**What works.** The taxonomy diagram (L385–400) is a good map of the territory. The LGCP theorem is correct for items 1–3 and 4 given $C_Y\ge0$; the proof (L441–473) is the same MGF calculation as Chapter 9 Theorem 9.2.1.

**Issues**
- **The taxonomy is wider than the content** (see §11.1). Missing: Neyman–Scott / Matérn cluster processes, hard-core as a stand-alone model, Berman–Turner quadrature.
- **The inhomogeneous Poisson section hides its best bridge (L402–411).** The log-likelihood $\ell(\boldsymbol\beta)=\sum_i\ln\lambda(\mathbf s_i;\boldsymbol\beta)-\int_W\lambda(\mathbf s;\boldsymbol\beta)\,d\mathbf s$ is, after dividing $W$ into small cells, the **Poisson regression likelihood** for the cell counts with a log link (a model every student has fit). That is what "Berman–Turner quadrature" in the taxonomy means (from memory). Say so; add the step "grid the window, count points per cell, `glm(count ~ covariates, family = poisson, offset = log(area))`" and the student can *fit* the model, which is currently impossible.
- **The LGCP statement "$g\ge1$ because valid covariance kernels have $C_Y\ge0$" (L469) is false.** Valid covariance functions can be negative (the wave/hole-effect models of Chapter 2). $g=\exp(C_Y)\ge1$ and $K>\pi r^2$ hold only when $C_Y\ge0$, as for the exponential, Matérn, and spherical models. State the condition. (An LGCP can show *local* regularity at lags where $C_Y<0$.)
- **The LGCP proof omits why the product density is $\operatorname{E}[\Lambda(\mathbf s)\Lambda(\mathbf s+\mathbf h)]$** (L447–450 says "law of total expectation" and moves on). Say: given the random surface, the process is Poisson with intensity $\Lambda$, so given $\Lambda$ the product density at two distinct places is $\Lambda(\mathbf s)\Lambda(\mathbf s+\mathbf h)$; average over $\Lambda$. And link to Chapter 9: "this is the same lognormal calculation as Theorem 9.2.1 with $Z=\Lambda$; the pair correlation is $C_Z/m_Z^2+1$."
- **Why LGCP is "the bridge to geostatistics" is not explained.** The latent field $Y$ is a Gaussian random field with the covariance models of Chapter 2; the point pattern is a noisy, count-valued observation of it. That is the same structure as a spatial generalized linear mixed model (Chapter 12 §12.5). Say it, point to Chapter 12, and say that *fitting* LGCPs (minimum contrast, or Bayesian/Laplace methods; from memory) is beyond this chapter.
- **The Strauss section has no mechanism and no computation (L475–487).** Three things a student needs: (a) $\alpha$ is a normalizing constant that is *not* available in closed form, which is why Strauss processes are simulated by MCMC (birth–death) and fitted by pseudolikelihood (from memory), and why the chapter cannot just "simulate Strauss" as it does for Poisson; (b) a picture of an increasing-$\gamma$ sequence ($\gamma=0,\ 0.3,\ 1$) showing a hard-core, a soft-core, and a random pattern; (c) the statement that the model can produce only inhibition ($\gamma\le1$), not clustering. "Markov" is never explained (it means the conditional intensity at a location depends only on points within $R$).
- **Beat 3 is absent** for all three models. A useful comparison table: *Inhomogeneous Poisson* (no interaction; trend only; $g=1$), *LGCP* (clustering through a latent field; $g\ge1$ if $C_Y\ge0$), *Strauss* ($g\le1$; hard to fit). With an extra column, "what $K$ looks like."

### §11.6 Algorithmic Translation

**What works.** The thinning algorithm (L519–562) is correct and well commented, and the choice of an intensity that depends on $x$ only keeps the bound trivial. I verified it: 3,000 runs give mean 17.17 and variance 17.69 (Poisson, as they should be: exact mean $e^2(e^{1.5}-1)/1.5=17.15$). The $K$ code follows the formula and is clear.

**Issues**
- **The promised confidence envelopes are not in the code (L497, L634).** L497 lists "Monte Carlo CSR confidence envelopes" as part of Algorithm 11.2 and L634 is labelled "Beat 3: Reality Diagnostics and Monte Carlo Envelopes." The main block instead flags a pattern as "Clustered" or "Regular" if $|\hat L(r)-r|>0.02$ (L658). That threshold has no statistical meaning. I tested it on patterns generated under pure CSR (the null the rule is supposed to protect):

  | Mean number of points | Chance that a *single* lag is flagged non-CSR (at $r=0.05,\dots,0.4$) | Chance that *any* of 5 lags is flagged |
  |---|---|---|
  | about 17 | 54%, 44%, 41%, 46%, 45% | 91% |
  | about 50 | 2%, 3%, 3%, 4%, 8% | 15% |

  The demo pattern has 22 points. Its printed table (L643–662) flags four of eight lags as "Clustered," which is what chance alone does to a random pattern of that size. The demonstration therefore leaves the student with the wrong lesson.
- **The correct lesson is available and is a good one.** The simulated pattern is an *inhomogeneous* Poisson process, which has no interaction between points at all. Averaged over 300 such patterns (about 17 points each), $\hat L(r)-r$ is $-0.006, 0.004, 0.012, 0.018, 0.023$ at $r=0.05,0.1,0.2,0.3,0.4$, so a trend pulls $K$ above $\pi r^2$ by a modest amount, comparable to the noise in a single pattern. That is the real message of L662 ("aggregation due to trend") and it deserves to be the *point* of the lab, with an envelope that makes it visible.
- **Fix by adding envelopes:** simulate $m\ge99$ CSR patterns with the same $n$ (draw $n$ uniform points: Theorem 11.2.1), compute $\hat L(r)-r$ for each, take the pointwise minimum and maximum (or 2.5% and 97.5% quantiles), and plot the observed curve against the band. Then say plainly that a curve leaving the band at some $r$ is suggestive and a global test needs a single statistic (for example $\max_r|\hat L-r|$, ranked against the simulated values; from memory).
- **Smaller code points** (none is an error): (a) the Knuth Poisson generator (L543–549) multiplies uniforms until the product falls below $e^{-\text{mean}}$, which underflows when the mean exceeds roughly 700 and is slow long before that; fine for a demo with mean 33, but say so, or use a library call; (b) the weight is approximated by 360 angular samples, and `max(0.1, proportion)` is a silent floor; with the chapter's radii it rarely matters, but a student should be told it is an approximation and why the floor exists; (c) the code computes weights for every pair and then loops over radii, which is a sensible design and worth a sentence; (d) `intensity_function_2d` hard-codes dependence on $x$ only, so `lambda_max` at `(x_max, y_max)` is correct only for that special surface. Say how to get a bound for a general surface.
- **Algorithm 11.1 has no theory link.** The reason thinning works is that a Poisson process with intensity $\lambda_{\max}$, with each point kept with probability $\lambda(\mathbf s)/\lambda_{\max}$, is a Poisson process with intensity $\lambda(\mathbf s)$ (independent thinning). That is two sentences and is the "Beat 1" for the algorithm.
- **No fitting algorithm anywhere.** The student can simulate and summarize but cannot estimate $\boldsymbol\beta$, a cluster scale, or $\gamma$. See the Poisson-regression bridge above.

### What is missing from the chapter (by the standards of a first treatment)

Flagged for the author to decide, not necessarily to add: **marked point processes** (the natural link to this book: a mark is a geostatistical variable observed at random locations); **kernel estimation of $\lambda(\mathbf s)$** (the usual first step); **model fitting** (Poisson regression for inhomogeneous Poisson, minimum contrast for clustered models, pseudolikelihood for Strauss; from memory); **goodness-of-fit via envelopes**; and **preferential sampling** (what happens in geostatistics when the sampling locations are not independent of the field, from memory a well-known issue that connects Chapters 1–10 to this one). One paragraph on the last would make this chapter feel like part of the book.

### Exercises

There is no in-chapter worked-exercise section. Every other recent chapter has one, and the inconsistency is noticeable. Candidates with the chapter's own materials: (1) derive $F(r)$ for a Poisson process in 3D ($1-\exp(-\tfrac43\lambda\pi r^3)$); (2) compute $K$ for a hard-core process of radius $R$ at small $r$; (3) compute the LGCP $K(r)$ for an exponential $C_Y$ and show it exceeds $\pi r^2$; (4) the mean and variance of the thinning count (Poisson, with the mean computed above); (5) the border-method discard fraction in a unit square ($1-(1-2r)^2$).

---

## 4. Evaluation Across the Four Dimensions

### Dimension 1 — Pedagogical pacing and the Three-Beat rhythm: **5 / 10**

- **The bridge is good; the rest is front-loaded.** Beat 1 is present in every section. Beat 2 is absent in five of six sections and Beat 3 in most; the chapter's only Beat 3 ends up as a code comment.
- **No picture and no data.** A point-pattern chapter that never shows a point pattern is hard to learn from. The three-pattern figure (random, clustered, regular, each with its $L(r)-r$ curve and envelope) is the single most valuable addition.
- **Voice.** The warm sentences are at the start of §11.1 and §11.2. Sections 11.3–11.5 read as a lecture-notes outline. The callout (L372–375) is the right voice and is used once.
- **Visible "Beat 1/2/3" headings** recur (and "Beat 1 (Continued)" at L299 is a clear sign the template is being forced). Keep the rhythm, drop the labels.
- **No "From the Lecture Hall" boxes.** Candidates: the Poisson distribution and its relation to the exponential (§11.1); the index of dispersion and why variance equals mean for Poisson counts (§11.2); Poisson regression with an offset (§11.5); the binomial as "Poisson conditioned on its total" (§11.2).

### Dimension 2 — Algebraic transparency and scaffolding: **7 / 10**

- **Strong:** the conditional-uniform proof, the $F$ derivation, the $K$ from $g$ identity, and the LGCP moment calculation are all clear and correct.
- **Needs work:** (a) the Slivnyak step (L273); (b) the "$C_Y\ge0$" claim (L469); (c) the second-order moment definition (disjoint plus correction term, L206–209); (d) the missing covariance-density equation that connects $g$ to the correlogram; (e) "strictly unbiased" (L336).
- **Anatomy blocks:** only one in the chapter (the $\hat K$ anatomy, L364–370); it is helpful. The LGCP and Strauss formulas deserve one each, term by term.

### Dimension 3 — Structural consistency and forward referencing: **4.5 / 10**

- **Promised but absent:** Neyman–Scott, Matérn, Berman–Turner (L5, L394–396); Monte Carlo envelopes (L497, L634); a worked-exercise section (every other recent chapter has one).
- **Missing back-references:** Chapter 9 (LGCP is the lognormal calculation), Chapter 3 (pair-count estimators and the empirical variogram share their logic and their edge/lag-bin problems), Chapter 2 (covariance models; the hole effect), Chapter 1 (second-order stationarity), Chapter 10 §10.5 (conditional simulation as the geostatistical analogue of simulating from a fitted point-process model).
- **Missing forward references:** Chapter 12 (spatial GLMMs, where the LGCP-like latent field reappears) and Chapter 13 (nonstationarity: the inhomogeneous case here is the first-order version).
- **Notation:** $\lambda$ is a constant, a function $\lambda(\mathbf s)$, and the Poisson mean $\lambda|A|$; $\Lambda$ is the intensity measure (L168) and the random surface (L414); $\Phi$ is the point process, which in Chapter 9 was the anamorphosis. Pick distinct symbols.
- **Window and domain** are used inconsistently (see §11.1).

### Dimension 4 — Empirical and computational utility: **5 / 10**

- **Verified correct:** the thinning algorithm (mean 17.17 vs. 17.15); Theorem 11.2.1 (read through; consistent with the Binomial result I simulated for the quadrat test); the quadrat-test chi-square approximation (upper tail 2.3–2.8% at nominal 2.5%); the CSR closed forms; the LGCP moments (given $C_Y\ge0$); and Ripley's estimator, which is approximately unbiased under CSR in the unit square for $r\le0.4$ (ratios 0.996–1.005).
- **Verified misleading:** the $\pm0.02$ diagnosis rule (false alarms of 41–54% per lag at about 17 points, and 91% on at least one lag).
- **Not demonstrated:** nothing is ever compared to truth or to a null band; no real or simulated clustered or regular pattern is analyzed; no model is fitted.
- **Code optional, algorithms not.** Thinning is explained and correct. $\hat K$ is explained and correct. What is missing is the *inference* algorithm (simulation envelopes), the *estimation* algorithm (Poisson regression), and a simulation recipe for the clustered and inhibited models.

---

## 5. Actionable Remediation List (prioritized)

### Must fix (correctness or clarity blockers)

1. **Replace the diagnosis rule (L637–662).** Remove the $\pm0.02$ threshold. Add CSR Monte Carlo envelopes (simulate $n$ uniform points, $m\ge99$ times). Make the lesson "trend can look like clustering; here is what the envelope says."
2. **LGCP claim (L469) and Theorem 11.5.1 statements 3–4.** Add the condition $C_Y(\mathbf h)\ge0$ for $g\ge1$ and $K>\pi r^2$.
3. **Remove the Slivnyak/Palm citation as the proof (L273).** Give the one-sentence independence argument; footnote the theorem.
4. **Quadrat test caution (L148).** State that $I>1$ can arise from a trend as well as from clustering, and that $Q-1$ degrees of freedom come from conditioning on $n$ (Theorem 11.2.1).
5. **Definition 11.3.2 (L206–209).** Drop "disjoint" or drop $N(A\cap B)$; write the covariance-density equation $\lambda_2-\lambda\lambda=\lambda^2(g-1)$ next to the pair correlation function.
6. **"Strictly unbiased" (L336).** Replace with "approximately unbiased under CSR for lags well below the window size," or add the conditions; keep the correct statement that $n(n-1)/|W|^2$ is unbiased for $\lambda^2$ under a Poisson process (L368).
7. **Promises (L5, L394–396, L497).** Write the Neyman–Scott subsection or remove the references; either implement the envelopes or remove "confidence envelopes" from the description.

### Should fix (pace, flow, voice)

8. **Add the three-pattern figure** (random, clustered, regular) with $L(r)-r$ and envelopes for each; use it in §11.1 and §11.4.
9. **Add the Poisson-regression bridge for the inhomogeneous Poisson model** (L402–411): grid the window, count per cell, `glm` with a log link and an offset. Say that this is Berman–Turner quadrature.
10. **Add the HPP simulation recipe** (Theorem 11.2.1: $n\sim\text{Poisson}$, then $n$ uniform points) and a two-sentence explanation of independent thinning for Algorithm 11.1.
11. **Give $F$, $G$, $J$ a Beat 2** (how to compute them, with an edge correction) or reduce the section to $K$ and $g$; add the "which question does each answer" paragraph.
12. **Strauss:** the normalizing-constant problem, the $\gamma=0,0.3,1$ picture, and what "Markov" means.
13. **Add a pitfalls list for $K$:** trend vs. clustering; pointwise vs. global tests; edge effects at large $r$; $K$ does not identify the process.
14. **Add a worked-exercise section** (§11.7) in the style of Chapters 9–10.
15. **Replace the dedication paragraph (L5)** with a one-sentence promise.
16. **Make the bridges explicit:** LGCP = lognormal calculation (Chapter 9); $\hat K$ and the empirical variogram are both pair counts (Chapter 3); LGCP latent field = Chapter 12's spatial GLMM; second-order stationarity (Chapter 1).
17. **Define "second-order stationary"** (L224) and unify the window/domain language (L19, L41, L68).
18. **Tidy notation** ($\lambda$ in three roles; $\Lambda$ in two; $\Phi$ clashing with Chapter 9).
19. **Add one paragraph** on what the chapter does not cover (marks, fitting, goodness of fit, preferential sampling).

### Nice to have

20. Lecture Hall boxes (Poisson and exponential; Poisson regression with offset; the binomial from conditioning a Poisson).
21. Redraw the edge-correction figure with a shaded outside arc.
22. Replace "up to 70%" with "$1-(1-2r)^2$: 36% at $r=0.1$, 64% at $r=0.2$, 75% at $r=0.25$."
23. Note that Algorithm 11.1's bound `lambda_max` is specific to the example surface, and that Knuth's Poisson generator underflows for large means.
24. Remove the visible "Beat 1/2/3" headings, including "Beat 1 (Continued)."

---

## 6. What to Keep Exactly As It Is

- The opening paragraph (L3) and the 1D-to-spatial bridge (L13–21, L23–38).
- Theorem 11.2.1 and its binomial corollary (L79–129), minus the duplicated lines.
- The index-of-dispersion idea (L135–154) as a classical bridge.
- The one-line proof of $F(r)$ (L262–271) and the $K$ from $g$ identity (L309–311, L325–332).
- The LGCP moment proof (L441–473), with the $C_Y\ge0$ condition and the pointer to Chapter 9.
- The thinning algorithm (L519–562) and the $\hat K$ code (L568–631).
- The taxonomy diagram (L385–400) as a map, once the content catches up with it.
- The $\hat K$ anatomy block (L364–370) and the "Wait—Why Did We Do That?" callout (L372–375) as the voice for the missing beats.
