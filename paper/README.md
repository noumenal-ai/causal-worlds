# Paper skeleton — Framing B (empirical finding)

Working outline for the causal-worlds paper. **Framing B** (recommended first): the contribution is an
*empirical finding* — interventional-CI discovery recovers regime-switching operational structure where
observational/score-based methods collapse, with the generator as the apparatus that produces the
worlds. Lower-risk and less hostage to scale than Framing A (datasets-and-benchmarks flagship).

Target venue: a causality/eval **workshop** first (NeurIPS 2026 workshops, deadlines ~Sept–Oct 2026),
then **CLeaR 2027** (deadline ~late-Oct/early-Nov 2026). arXiv cs.LG (cross-list cs.AI, stat.ME) once
the crossover holds at scale — needs a **personal endorser** (founder action item).

## Title (working)
*Fiction-first causal worlds: a leakage-resistant generator where interventional discovery beats the
observational toolbox*

## Abstract (to draft once scaled results land)
LLM causal benchmarks are undermined by contamination and prior-guessability. We generate fictional
operations from natural language with a declared ground-truth causal graph, gate them with an
independent cross-family judge so they are not solvable from priors, and show that standard
observational/score-based discovery (PC/GES/FCI/GIES) systematically mistakes a hidden-confounded pair
for a causal edge while an interventional-CI grader does not. [Headline numbers at scale.]

## 1. Introduction / motivation
- The contamination + "causal parrots" critique (cite Srivastava et al. 2510.16530; Kıcıman et al.
  2305.00050; Zečević et al. 2308.13067). The need for post-cutoff, leakage-resistant, priors-resistant
  benchmarks.
- Our stance: **fiction-first** — declared structure is ground truth by construction; no real data to
  leak or match.

## 2. Related work (differentiate explicitly)
- **G-Sim** (Holt et al., ICML 2025, 2506.09272) — NL→executable sim + SHD/F1, but as fidelity, not a
  discovery benchmark; no anti-cliché/leakage framing. *Closest competitor — differentiate head-on.*
- **CausalDynamics** (2505.16620), **CausalProfiler** (2511.22842), **CausalTimePrior** (2603.11090,
  regime-switching + interventions but for effect estimation/PFN, no NL authoring), **Agent2World**.
- LLM-as-judge bias → cross-family judging as standard mitigation.

## 3. Method (the apparatus)
- The IR (variables/roles, mechanisms, hidden confounders, regimes); the answer-key derivation.
- Author (cross-family from the judge) → gates T1–T4 (incl. the anti-cliché difficulty
  `1 − F1(prior, truth)`) → admit. Reproducible bake-off picks the author model.
- The reference interventional-CI grader; why observational/score-based methods fail the confounder
  trap (causal sufficiency assumption).

## 4. Experiments
- **The crossover** (core result): standard methods vs the interventional grader across the set —
  skeleton-SHD, directed F1, and *confounded-pair-kept-as-causal* (the trap). *v0.5 (n=35): grader
  confounded-kept **0** vs PC 13 / FCI 8.3 / GIES 17; SHD 1.47 vs 2.7–6.7.* [Scale to 100+.]
- **Difficulty-vs-error**: does difficulty predict observational collapse? *v0.5 (n=35): **structural**
  difficulty predicts observational skeleton-SHD (corr **+0.62**); name-guessability does not (+0.14).*
  The hardness is structural — confounders + regime sign-flips — not name-recitable.
- **Robustness**: multiple author models (bake-off), multiple judge families, seed variance.
- **Temporal**: genuine temporal/regime worlds + time-series baselines (PCMCI+, VARLiNGAM, Granger).

## 5. Limitations
- Difficulty metric currently measures name-guessability, not structural hardness (v0.3 finding).
- Mean difficulty 0.28 is low → worlds still partly guessable; needs to be pushed up with spread.
- GES (causal-learn) numpy-2 incompatible — toolchain pinning.
- Linear-Gaussian mechanisms so far; nonlinearity is future.

## 6. Conclusion
The synthesis (NL authoring × executable causal sim × ground-truth answer-key) + the anti-cliché
construction is a leakage-resistant apparatus that surfaces a clean, reproducible crossover.

## Submission checklist (escalation thresholds from the strategy review)
- [ ] ≥100 worlds with documented diversity coverage (size/density/lag/regime/confounder strength)
- [ ] clean crossover: observational error rises (or stays high) vs interventional holding
- [ ] mean difficulty meaningfully > 0.28 with a usable hard subset
- [ ] faithfulness validated by multiple judges + human spot-checks
- [ ] author/judge variance reported
- [ ] personal arXiv endorser secured
