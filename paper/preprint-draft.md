# Fiction-first causal worlds: a leakage-resistant generator that cleanly exhibits a latent-aware interventional identifiability crossover

*Working draft — workshop / CLeaR-class (Framing B: empirical finding + generator-as-apparatus). Subject is the public OSS package `causal-worlds` (`pip install causal-worlds`; v0.25.0; MIT; github.com/noumenal-ai/causal-worlds). Numbers are taken verbatim from the package's eval reports; nothing is extrapolated.*

---

## Abstract

LLM-touched causal benchmarks face two recurring threats to their validity: **data contamination** (the structure was memorized, not discovered) and **prior-guessability** (the answer is recoverable from variable names and roles without any data). We present **`causal-worlds`**, an open-source apparatus that authors *fictional-but-coherent* operations from a single natural-language sentence and emits, by construction, a **declared ground-truth causal graph** — an answer key that the executable simulator can never disagree with, because the key is *derived from the same specification* the simulator runs. Because the worlds are fiction-first (plausible, internally consistent, modeling no real system), there is no real dataset to leak and nothing to memorize.

Three design choices make the apparatus trustworthy, and we report each with its disclosed residual. (1) **Admission is grader-independent**: a world is admitted iff its declared structural causal model (SCM) is *faithful by construction* — every declared edge induces a detectable partial correlation given the target's other observed parents — computed in closed form from the population covariance, with **no discovery method run to decide admission**. (2) An **anti-cliché gate** (an independent cross-family LLM judge guessing the graph from semantics alone) drives a 3-tier disclosure certificate: a name-only prior F1 falls from **0.71 to 0.38**, and a fully name+role-blind prior collapses to **0.01** — *at or below* the 0.18 random-graph chance floor — i.e. once names and roles are stripped, the structure is **not recoverable from priors**; the disclosed residual is a *role-type prior* (name-blind ≈ 0.46) that is intrinsic to operational worlds and reported, not hidden. (3) **Leakage controls** report both varsortability (0.54) and R²-sortability (0.60, a disclosed residual > 0.5) under internally-standardized SCMs (iSCM), with the matching trivial baselines pushed well below the real methods.

On the resulting worlds we surface a clean, reproducible **identifiability crossover**, made **information-fair** by giving every interventional method the *same* interventional budget. Across a 26-world set (n=4000, seeds [7,11,23]), a deliberately simple **latent-aware** interventional rule keeps **0** confounded pairs as causal (skeleton-SHD 1.31, directed F1 0.90), while the causal-sufficiency toolbox — including PC *with* the same interventions (`pc+do`) and GIES — keeps **30** (a seed-averaged count of confounded-pair instances summed over the 26 worlds, each hiding 1–2 such pairs). The point is **not** "we beat the toolbox": the decisive lever is **latent-awareness, not interventions**, and the separation is **textbook identifiability** (Ψ-FCI; GIES) cleanly exhibited on a leakage-resistant testbed, not a new discovery method. We frame the reference discoverer accordingly: a *simple, latent-aware reference the benchmark is designed to reward*, whose score is reported but never gates admission. We give an honest limitations section (small n; descriptive-only difficulty; a residual role-type prior; an n=1 temporal set; linear-Gaussian only; fiction-first external-validity caveats) and position the work against the contamination, simulated-DAG-leakage, and interventional-identifiability literatures.

---

## 1. Introduction

Large language models now author, judge, and "reason about" causal structure, and a wave of benchmarks has followed. Two threats undercut the validity of nearly all of them.

- **Contamination.** If a graph (or a real dataset whose structure is documented somewhere) was in pretraining, a method that recites it has not *discovered* anything. The community has begun treating memorization/leakage as a first-order threat and rewarding contamination-resistant designs [Srivastava et al. 2025, arXiv:2510.16530].
- **Prior-guessability.** Many "causal" benchmarks are solvable from variable *names and roles* alone, because conventional operational semantics (a controllable lever raises an outcome) encode the answer. LLMs are strong at this knowledge-based graph guessing [Kıcıman et al. 2023, arXiv:2305.00050], which can masquerade as discovery [Zečević et al. 2023, "Causal Parrots", arXiv:2308.13067; Jin et al., Corr2Cause].

**Our stance is fiction-first.** Instead of trying to model a real system faithfully (and then arguing the structure was not leaked), we *declare* the structure and generate a coherent fictional operation around it. The declared graph is the ground truth *by construction* — there is no real data to leak and nothing to memorize. This directly answers the leakage-resistance call, and it reframes the problem from "is our fidelity high?" to "is the declared structure honestly recoverable, and is it recoverable for the *right* (structural) reasons rather than from priors or from simulation artifacts?"

The apparatus is the public Python package `causal-worlds`. The base install (engine, grading, built-in worlds, CLI) depends only on `typer`, `pydantic`, `numpy`; **the engine and all grading run with no API key** — only natural-language *authoring* needs LLM keys.

### Contributions

1. **A leakage-resistant, fiction-first, NL-authored causal-discovery benchmark with a by-construction answer key** — the four-way intersection of (NL authoring) × (temporal/regime structure) × (executable simulator) × (declared ground-truth causal graph) that, to our knowledge as of mid-2026, no prior tool occupies as a single artifact. *The novelty is the system-level synthesis and the anti-cliché construction, not any single new mechanism* (LLM-as-judge, cross-family judging, knowledge-based graph guessing, and synthetic generators all have precedent).
2. **Grader-independent admission** (closed-form faithfulness): worlds are admitted by a property of the *declared SCM*, never by running the discovery method that later "wins" — closing the circularity in which a grader admits the worlds it then succeeds on.
3. **A 3-tier anti-cliché certificate** that measures, and *discloses*, exactly how much of the structure is guessable from names, from roles, and from neither — with an explicit, transparently-reported role-type residual.
4. **Dual simulated-DAG leakage controls** (varsortability *and* R²-sortability via iSCM), reporting both metrics, their trivial baselines, and the disclosed R²-sortability residual.
5. **A clean, reproducible, information-fair identifiability crossover** demonstrating that latent-awareness — not interventional access — is the operative lever, with bootstrap confidence intervals, positioned explicitly as a textbook identifiability result rather than a new algorithm.
6. **A Stage-2 control track** with a by-construction optimal policy, regret, and regret-under-perturbation, plus a Gymnasium environment in which the regime shifts between steps.

---

## 2. The generator / method (the apparatus)

```
NL prompt → AUTHOR (Claude) → independent JUDGE (Gemini, different family) →
GATES T1–T4 → ADMITTED world + derived answer-key (+ persisted bundle w/ provenance)
```

### 2.1 NL author and conversational elicitation

The author (an Anthropic Claude model, via the `llm` extra) turns one sentence — e.g. *"a regional coffee chain with weekend swings and variable lead times"* — into a structural causal model: variables with roles, linear-Gaussian mechanisms, regime sign-flips, and a hidden confounder. An author **complexity tier** (`easy | standard | hard | adversarial`) controls how the world is constructed; the `adversarial` tier deliberately makes the obvious name-based guess *wrong* (phantom edges, reversed edges, regime sign-flips, and **role-misdirection** through an observable mediator) **while keeping every declared edge detectable** (so the faithfulness gate, §2.3, still passes). Because a one-shot prompt is underspecified, a conversational `elicit` mode builds a `WorldBrief` (entities/roles, what drives what, regimes, hidden causes, objective) before authoring.

### 2.2 Independent cross-family judge

Grading and gating use a **Gemini** model (`gemini-2.5-flash`, via the `llm` extra) — deliberately a **different model family** from the author. This is the standard self-preference / family-bias mitigation for LLM-as-judge setups. The judge (i) guesses the graph from names/roles alone (the anti-cliché signal, §2.4) and (ii) scores faithfulness. **Judge-independence applies to grading/gating only, not to authoring/elicitation.**

### 2.3 Grader-independent admission (the credibility keystone)

A world is **admitted only if all gates pass**:

- **T1 — validity:** well-formed spec; the contemporaneous subgraph is acyclic.
- **T2 — sample-sanity.**
- **T3 — faithfulness, grader-independent:** a world is admitted **iff its declared SCM is faithful by construction** — every declared edge induces a *detectable partial correlation* (≥ 0.05) given the target's other observed parents (so parameters do not cancel and hide a true edge), and regimes genuinely modulate a coefficient. This is computed in **closed form** from the population covariance `(I−B)⁻¹ Ω (I−B)⁻ᵀ`; partial correlations are scale-invariant, so the test holds for the iSCM data the substrate emits. **No discovery method is run to decide admission.** The reference grader's own score is *reported, never gating.* This fixes the earlier circularity (a world admitted by the same grader that then "won").
- **T4 — anti-cliché (strict):** admit only if the **named** prior-only F1 < 0.5 (difficulty ≥ 0.5) **and** the **name+role-blind** prior stays near the chance floor (< 0.35). A later **roles-only** sub-gate (roles-only prior < 0.4) was added; we report its (marginal) effect honestly in §4.2.

T3 is closed-form and deterministic; T4 invokes a single-sample LLM judge and is therefore the **noisy** gate. We separate the two cleanly: admission is never decided by a discovery method (T3 is grader-independent), but the *authoring loop* that produces an admissible adversarial world is empirically high-variance — we report its attempt/yield statistics in §3 rather than present a frictionless apparatus.

### 2.4 Derived answer key and persisted bundle

The answer key — directed edges over the *observed* variables plus the hidden-confounded pairs — is **derived from the spec and never stored separately**, so the key and the executable can never disagree. Temporal worlds get a lagged `(src, dst, lag)` answer key including autoregression. Each world persists a bundle (`spec.json` / `data.npz` / `answer_key.json` / `manifest.json`) with full provenance: model ids, grader version, seed, difficulty, complexity tier, faithfulness, and an **honesty label** (fictional; not real-world advice).

### 2.5 The reference discoverer is a simple known method, not a contribution

The reference `InterventionalCiDiscoverer` is a **deliberately simple, latent-aware reference discoverer** whose role is to show the benchmark is solvable by a method using interventional + latent-aware reasoning, and to provide a difficulty signal. **It is not a novel discovery algorithm.** Its behavior on the confounder trap is exactly the textbook identifiability result of Jaber et al. (Ψ-FCI, NeurIPS 2020) and Hauser & Bühlmann (GIES, JMLR 2012); see §5.

---

## 3. Experimental setup

We use the shipped benchmark sets: `v0.5` (35 worlds, the powered cross-sectional set, mean name-difficulty ~0.29), `v0.6` (**26 worlds**, hardened/`adversarial` under the strict T4 gate, mean difficulty 0.64), and `v0.7` (**11 worlds**, the roles-only-gated companion, mean difficulty 0.60). **Adversarial authoring is high-variance, and we disclose the yield rather than imply a frictionless apparatus:** v0.5 admitted 35/36 prompts at mean ≈1.1 author attempts; v0.6 admitted **26/30 at mean 1.5 attempts** (up to 4); and the stricter roles-only gate in v0.7 admitted only **11/30**. The gate does its job — it rejects guessable worlds — but a tighter gate trades world count for difficulty, which is why the powered statistics live on v0.5/v0.6 and v0.7 is reported as a companion negative (§4.2). The crossover uses n=4000 samples and seeds [7,11,23]; leakage controls and the temporal study use n=4000 / n=2000 at seed 7. Discovery baselines come from the `discover` extra (PC/GES/FCI via `causal-learn`, GIES via `gies`, DAGMA, DirectLiNGAM via `lingam`); temporal baselines from the `temporal` extra (PCMCI+/LPCMCI via `tigramite`, VARLiNGAM, Granger). Metrics: skeleton-SHD (↓), directed F1 (↑), and **confounded-kept** (↓) — the seed-averaged count of confounded pairs a method reports as a *causal* edge, summed over the worlds (each hides 1–2, so the total can exceed the world count and run fractional) — the trap.

---

## 4. Experiments

### 4.1 The identifiability crossover — `benchmark/v0.6`, 26 worlds, n=4000, seeds [7,11,23]

**The comparison is information-fair.** The `+do` methods and GIES receive the *same* interventional budget (pooled observational + per-variable `do()` environments) as the latent-aware reference, so we compare *methods*, not data access. The lever is **latent-awareness, not interventions — and not "we beat the toolbox."**

| method | data | latent-aware? | mean skeleton-SHD ↓ | mean directed F1 ↑ | total confounded-kept-as-causal ↓ |
|---|---|---|---|---|---|
| **interventional-ci (reference)** | interventional | yes | **1.31** | **0.90** | **0** |
| pc | observational | no | 3.18 | 0.64 | 29 |
| ges | observational | — | *(errored on all 26)* | — | — |
| fci | observational | partly | 3.17 | 0.66 | 21.67 |
| dagma | observational | no | 5.23 | 0.29 | 27 |
| directlingam | observational | no | 5.76 | 0.31 | 27 |
| gies | interventional | no | 5.71 | 0.69 | 30 |
| **pc + interventions (pc+do)** | interventional | no | 4.71 | 0.53 | **30** |
| fci + interventions (fci+do) | interventional | partly | 4.73 | 0.53 | 21 |

**The decisive row is `pc+do`.** Given the *same* interventional budget as the reference, PC *still* keeps **30** confounded pairs as a *causal* edge (seed-averaged sum over the 26 worlds) — no better than observational PC's 29. GIES (causal-sufficiency, interventional) likewise keeps **30**. Only the latent-aware interventional rule reaches **confounded-kept = 0**. You cannot tell confounding from causation without *both* interventions *and* a latent-aware method; interventions alone, fed to a sufficiency method, do not help.

**The interventional advantage is robust.** ΔF1 = F1(reference) − F1(method), 95% percentile-bootstrap CI (n=26); **every CI excludes 0**:

| method | ΔF1 | 95% CI |
|---|---|---|
| pc+do | +0.37 | [0.33, 0.42] |
| fci+do | +0.37 | [0.30, 0.43] |
| pc | +0.26 | [0.22, 0.30] |
| fci | +0.24 | [0.19, 0.28] |
| gies | +0.21 | [0.18, 0.24] |
| directlingam | +0.59 | [0.54, 0.64] |
| dagma | +0.61 | [0.56, 0.65] |

**Honest caveat on DAGMA / DirectLiNGAM.** Both run at default hyperparameters, and LiNGAM's non-Gaussian-noise assumption is *violated* by these linear-Gaussian worlds, so their *skeleton* accuracy is not their best case. The robust, relevant verdict is **confounded-kept**, and — like every causal-sufficiency method here — they keep it. **GES (`causal-learn`) errored on all 26 worlds** (a numpy-compatibility `TypeError`); we report this rather than hide it.

**The honest reading.** This is an *identifiability* result. The reference is a deliberately simple, latent-aware reference discoverer the benchmark is designed to reward — *not* a new discovery algorithm (the result is textbook: Jaber et al. Ψ-FCI; Hauser & Bühlmann GIES). The crossover also holds on the easier `v0.5` set (35 worlds): reference confounded-kept 0 (SHD ~1.44, F1 0.91) vs PC 14.33 / FCI 9.67 / GIES 17 / pc+do 15 — kept for comparison.

### 4.2 The anti-cliché 3-tier certificate — `benchmark/v0.6` (judge `gemini-2.5-flash`)

We score the directed F1 of a **prior-only** LLM graph guess (no data) against the truth, at three progressive disclosure ("blind the names, then the roles, then re-test") levels, versus the random-graph chance floor:

| level | what's visible | v0.6 F1 [95% CI] | reading |
|---|---|---|---|
| **named** | names + roles | **0.38** [0.34, 0.43] | down from v0.5's **0.71** — name memorization cut |
| **name-blind** | names → `X1..Xn`, roles kept | **0.46** [0.43, 0.50] | *higher* than named ⇒ adversarial names now actively **mislead** |
| **name+role-blind** | names anonymized AND roles hidden | **0.01** [0.00, 0.02] | **structure is not guessable once semantics are stripped** |
| null (chance) | — | 0.18 [0.17, 0.19] | floor |

**What is clean.** The cliché that matters — **name/structure memorization — is eliminated**: the fully-blind prior is ≈ 0.00 (below the chance floor), and the named prior was cut from 0.71 to ~0.4. That the name-blind prior (0.46) *exceeds* the named prior (0.38) is direct evidence that the adversarial author's names now *mislead* a name-reading guesser.

**The disclosed residual: role-type priors.** The name-blind prior (0.46) sits well above chance because *role conventions* (a controllable lever → an outcome) carry real signal. This is a **legitimate operational prior, transparently reported — not hidden** — and it is *intrinsically informative*: you cannot remove it while keeping the lever→outcome path that makes a world an operation at all.

**The roles-only gate is marginal (honest negative).** The `v0.7` companion (11 worlds, roles-only gate, mean difficulty 0.60) nudged the name-blind prior only **0.46 → 0.43** (chance 0.16; named 0.41; name+role-blind 0.00) — a 0.03 movement that cost the set 26 → 11 worlds. Single-sample LLM-judge gating is noisy and role-type signal is intrinsic; we therefore *disclose* this residual rather than claim it "fixed."

**History (the honest negative that drove the work).** On v0.5 the named prior was 0.71 [0.68, 0.74] and the anonymized prior 0.61 [0.58, 0.65] versus a 0.20 chance floor — i.e. **leaky** — which is precisely what forced the strict T4 gate and the adversarial author tier.

### 4.3 Simulated-DAG leakage controls — `benchmark/v0.5`, n=4000, seed 7

Sortability = 0.5 means the causal order is *not* readable off that signal; → 1.0 means it is (Reisach et al. line). We report both metrics and the matching trivial baselines:

| signal | sortability | matching trivial baseline F1 (SHD) | removable post-hoc? |
|---|---|---|---|
| **varsortability** (marginal variance) | **0.54** | `sortnregress` **0.33** (SHD 8.66) | yes (by standardization) |
| **R²-sortability** (scale-invariant predictability) | **0.60** | `R²-sortnregress` **0.37** (SHD 8.26) | **no** |

The substrate uses **internal standardization (iSCM)** — each continuous variable is z-scored *as it is generated*, in topological order, so neither marginal variance nor R² compounds along the causal order; regime/binary columns stay `{0,1}`. **Before → after iSCM:** under an earlier *post-hoc* fix, varsortability was ~0.58 but R²-sortability was 0.73 with `R²-sortnregress` F1 0.40 (still leaking); iSCM dropped R²-sortability 0.73 → 0.60 and `R²-sortnregress` 0.40 → 0.37, with varsortability holding ~0.54. Both trivial baselines now sit **well below the real discovery methods** (cf. §4.1). For context, the *original unstandardized* leak was varsortability **0.94** with `sortnregress` F1 **0.74** — i.e. a trivial sort beat PC/FCI. **The disclosed residual: R²-sortability 0.60 > 0.5 is disclosed, not yet fully closed.** The crossover is robust to iSCM (reference confounded-kept stays 0).

### 4.4 Difficulty axis — descriptive, not a validated predictor

We treat difficulty as a **descriptive axis reported with CIs, not a validated predictor.** On v0.6 (bootstrap CIs, n≈26), observational methods show only weak correlation between world difficulty and discovery error, with CIs that **include 0** (pc r 0.29 [−0.04, 0.66]; fci r 0.22 [−0.08, 0.56]); the reference r is 0.38 [−0.13, 0.70]. On v0.5 (n=35) the observational correlations *just* exclude 0 (pc r 0.40 [0.07, 0.68]; fci r 0.42 [0.08, 0.69]) while the reference is flat (r 0.24 [−0.06, 0.51], includes 0). A separate **keyless** re-analysis (`structural-difficulty/v0.5`, **n=35**, the powered set; **no bootstrap CIs**, and the structural score is a coarse discrete predictor in {0..4}) finds that a **structural** difficulty score (confounders + regime sign-flips) correlates with observational skeleton-SHD at r ≈ **+0.82** — but this is partly *mechanical* (a hidden confounder creates a spurious adjacency by construction) — while the non-tautological signal, structural difficulty vs the *value* of intervention (ΔF1), is only r ≈ **+0.36**. Name-guessability difficulty does *not* predict error (r ≈ +0.28 / +0.16). These point estimates are **fragile**: the same re-analysis on the earlier, smaller `v0.2` set (n=12) returned the *opposite* signs (structural-vs-SHD −0.12, structural-vs-ΔF1 −0.39) and was reported "inconclusive at n=12." The sign flip between sets is exactly why we present difficulty **descriptively only**, with no CIs and no claim of a validated predictor, and caution against using it as an instrument.

### 4.5 Temporal worlds — built-in `supply`, n=1 (honest)

Time-series discovery on the built-in `supply` world (autoregressive lead time + inventory, hidden logistics confounder `L`, n=2000, seed 7, 7 true lagged edges):

| method | temporal SHD ↓ | temporal F1 ↑ | recovered | kept confounded pair? |
|---|---|---|---|---|
| pcmci+ | 0 | 1.00 | 7 | False |
| lpcmci | 3 | 0.73 | 4 | False |
| varlingam | 7 | 0.60 | 13 | **True** |
| granger | 18 | 0.18 | 15 | False |

Temporal grading is **demonstrated end-to-end on this one built-in world**: PCMCI+ recovers the lagged structure *exactly* (SHD 0, F1 1.0), which confirms the lagged answer key and temporal scoring machinery run correctly on at least this case. We deliberately **do not** say "validated" — a single hand-authored world at a single seed demonstrates the apparatus works; it does not establish a benchmark. Note the confounder trap is **method-specific** here (only VARLiNGAM keeps the spurious edge), *unlike* the cross-sectional crossover where the whole sufficiency toolbox falls for it — PCMCI+'s momentary-CI approach plus the autoregressive structure evidently screens off the latent. **Honest caveat: n=1.** A temporal benchmark *set* (n>1) is needed and is roadmapped. Separately, an LLM-authored temporal world (a reservoir operation with a hidden `soil_saturation` confounder and a snowmelt regime) was admitted in 1 attempt through a PCMCI+ temporal gate (F1 0.56), showing the authoring path generalizes to lagged worlds.

### 4.6 Control track (Stage 2) — by-construction optimal policy, regret, regret-under-perturbation

For a linear-Gaussian world with a quadratic objective, each lever's total effect on the outcome is the path-sum `(I−B)⁻¹[outcome, lever]` (levers intervened ⇒ their rows cut), **marginalized over regimes**; the LQ optimum decouples to `u*_i = effect_i / cost`. The optimal policy is therefore **correct by construction** — the "what is the right magnitude?" problem dissolves — and a controller is scored by **regret** against the declared optimum (0 = optimal play); the path-sum effect, the closed-form optimum, and ~0 regret all check out, while a do-nothing policy incurs regret ≈ the optimum's worth.

**Regret-under-perturbation (the stay-optimal thesis, measured).** A **regime-sign-flipped** lever has *marginal* effect 0 (it cancels across regimes), so the regime-blind optimum sets `price=0`, whereas the regime-aware optima are `+1` / `−1`. The blind policy loses ~0.5 reward in *every* regime (worst-regret 0.5); the matching per-regime optimum has ~0 regret. The `regret_under_perturbation` report returns worst + mean + per-regime regret.

**Gymnasium env (`causal_worlds.gym.ControlEnv`, the `gym` extra).** Action = lever values; reward = the objective under the *current* regime; **the regime shifts between steps** (the perturbation). `info` carries regime-aware `optimal_reward`/`regret`; the agent observes only observed-variable means (it must *detect* the shift, never read the spec). Cumulative regret is the stay-optimal-under-perturbation score.

---

## 5. Related work and positioning

**The reference discoverer is a simple latent-aware discoverer the benchmark rewards — not a novel method.** The crossover's mechanism is the textbook identifiability fact that `{X→Y}` and `{X←L→Y}` (L latent) are observationally indistinguishable but become distinguishable under the right interventions *with a latent-aware rule*:

- **Jaber, Kocaoglu, Shanmugam & Bareinboim, Ψ-FCI** (NeurIPS 2020) — the most on-point precedent: `{X→Y, X←L→Y}` and `{X←Y}` "are indistinguishable from observational data alone… [but] immediately distinguishable given ⟨P, P_X⟩." A sound-and-complete interventional FCI *for graphs with latent confounders* — exactly the regime our reference embodies in simple form.
- **Hauser & Bühlmann, GIES** (JMLR 2012; arXiv:1303.3216) — interventions shrink the Markov equivalence class, *but GIES assumes causal sufficiency*; that assumption is precisely *why* it keeps confounded edges, making it the perfect foil, not a competitor.
- *(Supporting, available to cite as the family of methods the benchmark is built to reward/penalize)* Kocaoglu et al., I-FCI / I-Markov (NeurIPS 2019); Mooij et al., JCI (JMLR 2020); Eberhardt & Scheines (2007); Ogarrio/Spirtes/Ramsey, GFCI (2016) for FCI's finite-sample fragility.

**Simulated-DAG leakage (the governing methodological line for §4.3).**

- **Reisach, Seiler & Weichwald, "Beware of the Simulated DAG!"** (NeurIPS 2021; arXiv:2102.13647) — introduced *varsortability* and `sortnregress`; "the remarkable performance of some continuous structure learning algorithms can be explained by high varsortability and matched by a simple baseline."
- **Reisach, Tami, Seiler, Chambaz & Weichwald, "A Scale-Invariant Sorting Criterion / R²-sortability"** (NeurIPS 2023) — `R²-SortnRegress`; R²-sortability is an artifact that "cannot be removed after the fact" by standardization (`CausalDisco`).
- **Ormaniec et al., "Standardizing Structural Causal Models" (iSCM)** (NeurIPS 2024; arXiv:2406.11601) — internal standardization; warns that *post-hoc* standardization is "not innocuous." This is the substrate's leakage fix.
- *(Supporting)* Herman, Wahl, Ninad & Runge (CLeaR 2025; arXiv:2503.17037) — confirms varsortability is removable post-hoc, R²-sortability is not.

**Anti-cliché / contamination / LLM-causal critique (motivates §4.2 and the fiction-first stance).**

- The **"blind score is the real score" critique** (the informal "Caliper" framing) — anonymize-and-re-test as the honest control; this is the spirit of the 3-tier disclosure certificate, which we cite informally as we could not pin a stable archival reference. The substantive, citable anchors for the same idea are the knowledge-based-guessing and contamination lines below (Kıcıman et al.; Zečević et al.; Srivastava et al.).
- **Zečević et al., "Causal Parrots"** (arXiv:2308.13067); **Kıcıman, Ness, Sharma & Tan** (arXiv:2305.00050) — knowledge-based causal-graph generation from names/roles; **Jin et al., Corr2Cause** (arXiv id to be confirmed against arXiv before posting); **"A Critical Review of Causal Reasoning Benchmarks for LLMs"** (arXiv:2407.08029); **Srivastava et al., "Realizing LLMs' Causal Potential Requires Science-Grounded, Novel Benchmarks"** (arXiv:2510.16530) — the contamination / post-cutoff / leakage-resistance call this work answers.

**Generators and benchmarks (positioning; each owns one corner of the four-way intersection).**

| tool | arXiv | corner it owns | what it lacks for this job |
|---|---|---|---|
| **G-Sim** (Holt, Ruiz Luyten, Berthon, van der Schaar; ICML 2025) | 2506.09272 | LLM authors a sim + **calibrates to real data**; SHD/F1 as a fidelity diagnostic | needs *real data*; aimed at **fidelity**, not a declared answer-key discovery benchmark; no anti-cliché/leakage framing. **Closest neighbor — differentiated head-on.** |
| **CausalDynamics** (NeurIPS 2025 D&B) | 2505.16620 | large-scale temporal benchmark from coupled ODEs/SDEs | no NL/LLM authoring, no anti-cliché difficulty, not fiction-first |
| **CausalProfiler** (AAAI 2026) | 2511.22842 | random synthetic generator w/ coverage guarantees, SHD/SID/PEHE | tabular/parametric; no NL authoring, no leakage-resistance framing |
| **Auto-Bench** | 2502.15224 | executable env + hidden ground-truth adjacency recovered via interventions | worlds fixed/designed, **not NL-authored** |
| **PillagerBench** | 2509.06235 | LLM causal-graph generation in an executable Minecraft world | **no ground-truth answer key** |

The three arXiv ids above (and the two near-misses) are drawn from this project's own literature scan and resolve as listed. A handful of further neighbors we are aware of by name only — **DEVS-Gen** (NL → discrete-event ops sim, but no declared causal-graph answer key), **SD-SCM** (LLM fills mechanisms over a *user-supplied* DAG; tabular, not an executable sim), and **TimeGraph** (known-graph time-series for discovery, but parametric/templated with no NL authoring) — occupy adjacent corners but we deliberately omit arXiv ids we could not verify against arXiv, rather than assert unconfirmed identifiers.

**The unoccupied slice** is the four-way intersection **(NL authoring) × (temporal/regime structure) × (executable simulator) × (declared ground-truth causal-graph answer key)**, made leakage- and prior-resistant. Each neighbor owns one corner; none emits *the structure that generated the data* as a first-class, scoreable artifact while being NL-authored, executable, and leakage-resistant. We stress that **the defensible novelty is the synthesis plus the anti-cliché construction**, and that this novelty is *perishable* — a point-in-time scan, not a durable moat.

---

## 6. Limitations (honest)

- **Small n; CIs only on some axes.** The cross-sectional crossover is powered (v0.5 n=35, v0.6 n=26, 3 seeds, bootstrap CIs on ΔF1 and on the difficulty-vs-error correlations of §4.4), but several reported quantities are aggregates over a small world set, and some axes carry no CI at all — notably the temporal study (n=1) and the keyless structural-difficulty re-analysis (point estimates only).
- **Difficulty is descriptive, not predictive.** Difficulty-vs-error correlations are weak and frequently CI-includes-0 on v0.6; we do not claim a validated difficulty predictor (§4.4). The structural-difficulty correlation (~+0.82 vs observational SHD, on the **n=35** v0.5 set, **no CIs**, coarse discrete predictor) is partly *mechanical*, its non-tautological counterpart (vs ΔF1) is only ~+0.36, and the signs **flip** on the smaller n=12 set — so we read this as an unvalidated descriptive axis, not evidence.
- **A residual role-type prior remains.** Name/structure memorization is eliminated (fully-blind ≈ 0.00; named 0.71 → 0.4), but role conventions leave a name-blind prior ≈ 0.46. We *disclose* this; the roles-only gate moved it only ~0.03 (0.46 → 0.43) at a 26 → 11 world cost. This residual is intrinsic to operational worlds.
- **R²-sortability residual.** iSCM brings varsortability to 0.54 and R²-sortability to **0.60 > 0.5** — disclosed, not yet fully closed; trivial baselines are nonetheless pushed well below the real methods.
- **Temporal set is n=1.** Temporal grading is *demonstrated* (not validated) on one built-in world at one seed; the confounder trap there is method-specific. A temporal *set* (n>1) is roadmapped.
- **Linear-Gaussian only; no nonlinearity yet.** Worlds are linear-Gaussian today (which also disadvantages DirectLiNGAM's non-Gaussian assumption — reported, not hidden). Nonlinearity is roadmapped (GitHub #10).
- **Fiction-first external-validity caveat.** Worlds are *fictional by design* — plausible and internally consistent, modeling no real system. This is exactly what makes them leakage-resistant, but it also means the benchmark measures a method's ability to recover *declared* structure under controlled traps, **not** its fidelity to any real-world operation. Bundles carry an explicit honesty label to this effect.
- **GES errored on the full set** (numpy-compatibility `TypeError`), so GES is absent from the v0.6 table; reported, not hidden.

---

## 7. Conclusion

`causal-worlds` is a leakage-resistant, fiction-first apparatus: from one natural-language sentence it authors a coherent fictional operation whose declared causal graph is the ground truth by construction, admits worlds by a *grader-independent* faithfulness property, and certifies anti-cliché difficulty and simulated-DAG leakage with *disclosed* residuals. On the worlds it produces, a deliberately simple, **latent-aware** interventional reference cleanly recovers structure where the causal-sufficiency toolbox — even given the same interventions — keeps a hidden-confounded pair as causal. The contribution is the *apparatus that surfaces this textbook identifiability fact cleanly and reproducibly*, not a new discovery method. The honest open items — scale, a temporal set, nonlinearity, and the disclosed role-type and R²-sortability residuals — define the path from this workshop-class finding toward a flagship benchmark.

---

## Reproducibility

`pip install "causal-worlds[discover,llm,temporal,gym]"` (v0.25.0, MIT, Python 3.13+; engine + grading need no API key). Benchmark sets `v0.5`/`v0.6`/`v0.7` and the eval reports (`baseline-crossover`, `name-only-baseline`, `varsortability`, `structural-difficulty`, `temporal-crossover`) ship in the repo with seeds and pinned model ids. 135 tests, 95% coverage; `ruff select=ALL`, `mypy --strict`, CI is a merge gate. Built on the shoulders of pgmpy, DoWhy, CausalPlayground, causal-learn, and Gymnasium.
