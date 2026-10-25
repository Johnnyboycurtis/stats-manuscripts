# Review D: Chapters 9, 10, 11 (read-only)

Files: `chapters/ch09_spatial_glmms.md` (1350 lines), `chapters/ch10_exploring_point_patterns.md` (1477), `chapters/ch11_point_processes_glms.md` (1500).
Checked: cross-chapter claims, NOTATION.md, terminology, formulas, shared Part IV setting numbers, Ch9 county results vs Ch0, forward references, internal contradictions.
Line numbers marked "~" are approximate (located by search, not re-verified after the final read).

## Findings (sorted by severity)

| ID | Sev | Location | Inconsistency | Suggested fix |
|----|-----|----------|---------------|---------------|
| D1 | MEDIUM | ch11:1420, ch11:1490 | Section reference points to the wrong section. Text sends the reader to §11.8.4 for the topic that is actually treated in §11.8.5. The mechanical checker passes because §11.8.4 exists. | Change both references to §11.8.5 (re-read the sentence at each site to confirm the topic). |
| D2 | MEDIUM | ch10 (crash model, `lam_crash`) vs ch11 (crash/accident intensity, §11.5 and the Part IV simulation) | The "shared Part IV setting" is not shared at the model level. Ch10 uses `lam_crash = 600 * exp(-6 * dist)` (mean intensity about 199). Ch11 uses a log-linear model `exp(5.846 - 3 * dist + 1.5 * pop)` calibrated to 400 expected events. Different covariate set, different distance decay, different total. | Either (a) state explicitly in Ch11 that it re-simulates with a richer model and why, and say the Ch10 pattern is not reused, or (b) change Ch10 to the Ch11 model. Option (a) is cheaper. |
| D3 | MEDIUM | ch11:1482 (and ~1292) | Counts conflict. Text says "385 expected controls" while the expected count in the design is 362.5 (385 is the realized count). | Say "385 controls were realized (362.5 expected)" or use one number consistently. |
| D4 | MEDIUM | ch10 case-control section vs ch11 §11.5.4 | The "shared" case-control data have different realized counts: Ch10 reports 46 cases / 349 controls, Ch11 reports 58 / 385. Ch11 does not say it is a new draw or a five-year pooled set, and Ch10 does not warn that Ch11 enlarges it. | Add a sentence in Ch11 §11.5.4 ("unlike the single-year sample of Section 10.x, this pools five years: 58 cases") and a forward pointer in Ch10. |
| D5 | MEDIUM | ch9, ch10, ch11 (relative-risk passages) | "Relative risk" is used in three senses: (i) Ch9 area-level ratio of the Poisson mean to the expected count, (ii) Ch10 cross-K / case-control density ratio, (iii) Ch11 logistic-bridge exp(coefficient). Notation drifts too: Ch10/Ch11 use `pop`, `ex` while Ch11 text also uses `lambda_pop`, `excess`, `theta`. | Add one sentence defining each sense when first used, and pick one symbol set for the Part IV covariates. Add a NOTATION.md overload entry for "relative risk". |
| D6 | LOW-MEDIUM | ch11:496 | Reference to §9.4.1 where the content (the Laplace / PQL comparison used by analogy) is in §9.4.2. | Re-read the sentence, fix to §9.4.2 if confirmed. |
| D7 | LOW | ch10:355 | "dist function of Chapter 1" - the signature used in Ch10 code (arguments, return shape) is not identical to the Ch1 helper. | Say "a vectorized version of the distance helper from Chapter 1" or reuse the Ch1 name and signature. |
| D8 | LOW | ch11:1054 vs ch10 pair-correlation definition | g(h) is defined for inhomogeneous pairs in Ch11 (intensity-weighted pair density) and via annulus counts under homogeneity in Ch10. Both are fine alone; the link is never stated. | Add one sentence in Ch11: "reduces to the g of Chapter 10 when the intensity is constant." |
| D9 | LOW | ch9:245 | The precision matrix Q is said to be "of Chapter 8", but NOTATION.md assigns first introduction to §7.2.3 (SAR precision). | Say "the precision matrix Q (Sections 7.2.3 and 8.x)" or fix the NOTATION.md first-introduced entry. |
| D10 | LOW | ch10 vs ch11 (K-hat code) | K-hat normalization differs: Ch10 divides by `n(n-1)/|W|` style with Ripley weights, Ch11 `k_hat` uses translation weights and a different normalizer. Both are valid but a reader comparing outputs will see small differences. | Add a footnote in Ch11 noting the estimator change and that values at small r differ slightly. |
| D11 | LOW | ch10 vs ch11 (cluster parameters) | Thomas-cluster parameters differ per chapter (`sim_thomas` in Ch10, `simulate_cluster` in Ch11) with no remark. Harmless, but "same process" wording would be misleading. | Keep, but avoid saying "the same cluster process". |
| D12 | LOW | ch10 vs ch11 (plant effect) | Relative risk at the plant quoted as 0.404 in one place and 0.408 in another. | Pick the one the code prints; round consistently. |

Counts: HIGH 0, MEDIUM 5, LOW-MEDIUM 1, LOW 6.

## Verified consistent

- Ch9 shared county-map setting (seeds 2070/2071, truth sigma^2 = 0.15, rho = 0.6, beta1 = 0.3, rate 0.001) is used the same way throughout Ch9 and the numbers quoted in prose match the code.
- Ch9 BYM2 scaling constant (geometric mean of ICAR marginal variances) is defined once and used consistently; Ch8 correctly promises this in its closing section.
- Ch8's promises about Ch9 (count response, ICAR / proper CAR / BYM comparison, sum-to-zero constraints) are all delivered. Ch6's closing promise (rates in regions of different size not exchangeable) is delivered by the EB gamma-Poisson section.
- Ch9 IRLS = Fisher scoring, working response with offset removed, Henderson equations and PQL steps agree with each other; Laplace negative log-likelihood matches the stated formula.
- Ch10 kernel intensity edge correction, G/F/J, K/L/g, Monte Carlo envelope and max-deviation test definitions are internally consistent; seeds 2080-2098 are unique per figure.
- Ch11 Berman-Turner weighted Poisson regression, logistic bridge, scan-statistic LLR, sandwich/bootstrap SEs under clustering: formulas match the code.
- Ch12's opening paragraph correctly summarizes Ch10 and Ch11.
- No `**[STORY NEEDED]**` placeholders remain in Chapters 9-11.
- All section references in these chapters resolve to existing headings (checker), apart from D1 and D6 which resolve to the wrong section.

## Appendix expectations from these chapters

- Appendix A.1 (block matrix inversion / Woodbury identity): referenced at ch09:647 and in Exercise 8 (ch09:~1326).
- No other appendix references in Chapters 9-11.
