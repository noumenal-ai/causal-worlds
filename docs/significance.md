# Significance — what causal-worlds is, and why it matters (#17)

The strategic narrative behind the package: the contribution, why it is non-obvious, where it sits in
the field, and an honest read on how far it goes. Feeds [`paper/preprint-draft.md`](../paper/preprint-draft.md),
the README, and our positioning.

## In one paragraph

LLM-touched causal benchmarks have a validity problem: most can be solved by **reciting priors**
(the answer is guessable from variable names/roles) or by **memorized structure** (the graph or its
data was in pretraining) — so a high score need not mean any causal *discovery* happened.
`causal-worlds` answers this by being **fiction-first**: it authors a coherent but fictional operation
from one natural-language sentence and **declares** the causal structure, so the ground-truth
answer-key is *derived from the same specification the simulator runs* — nothing to leak, nothing to
memorize, and the key and the executable can never disagree. On the worlds it produces, a deliberately
simple latent-aware interventional rule cleanly recovers structure where the causal-sufficiency
toolbox — *even given the same interventions* — keeps a hidden-confounded pair as causal. The
contribution is **the apparatus that exhibits this textbook identifiability fact cleanly,
reproducibly, and audit-survivingly**, not a new discovery algorithm.

## The contribution (the synthesis, not any single piece)

1. **A four-way intersection that appears unoccupied:** natural-language authoring × executable
   temporal/regime causal simulator × **declared ground-truth answer-key** × a control track. Each
   neighbour owns one corner; none emits *the structure that generated the data* as a first-class,
   scoreable artifact while being NL-authored, executable, and leakage-resistant.
2. **A benchmark designed to survive its own audit.** The real intellectual content is the
   *apparatus*: **grader-independent admission** (closed-form faithfulness — no circularity),
   **iSCM leakage controls** (varsortability *and* R²-sortability, with the trivial baselines pushed
   below the real methods), a **3-tier anti-cliché certificate** (named / name-blind / name+role-blind
   vs a chance floor), and an **information-fair crossover** with bootstrap CIs. This is a *measured*
   response to the "causal parrots" / contamination critique — most LLM causal benchmarks can be
   solved by reciting priors; ours provably (mostly) cannot.
3. **Fiction-first as a feature, not a limitation.** Removing the real-world fidelity bar is exactly
   what removes the leak surface: correctness is *relative to the declared world*, and the key is
   derived from the spec, so simulator and answer-key cannot drift apart.
4. **A dual instrument from one declared SCM:** it grades *discovery* (recover the graph) **and**
   *control* (regret + regret-under-perturbation vs a by-construction optimal policy). Because the
   engine + grading are keyless and the worlds are cheap to author, the same declared truth also makes
   a **practical, low-cost verifier for iterating a causal-discovery method** — change the method, ask
   the answer-key whether it got closer, repeat.

## Related-work map (novel vs perishable)

- **Contamination / "causal parrots" critique** (Srivastava et al. 2510.16530; Kıcıman et al.
  2305.00050; Zečević et al. 2308.13067; Corr2Cause): names the problem — knowledge-based graph
  guessing masquerading as discovery — that our fiction-first + anti-cliché design directly answers.
- **Generators/benchmarks, each owning one corner:** **G-Sim** (LLM authors a sim but *calibrates to
  real data*; fidelity, not a declared discovery key); **SD-SCM** (LLM fills mechanisms over a
  *user-supplied* DAG; tabular); **TimeGraph / CausalTimePrior** (known-graph time-series, but
  parametric/templated, no NL authoring); **CausalDynamics** (large-scale, but ODE/SDE-authored, no
  NL/anti-cliché); **CausalProfiler** (random SCM sampling, inference-first). **Neural "world models"**
  (e.g. Qwen-AgentWorld, [#16](https://github.com/noumenal-ai/causal-worlds/issues/16)) predict
  observations with *no ground-truth graph* — a different artifact that shares our vocabulary.
- **Identifiability foundations** (Ψ-FCI; GIES; JCI; ICP): the crossover's mechanism is *their*
  textbook result, exhibited on a clean testbed — we cite them as the methods the benchmark is built
  to reward/penalize, **not** as competitors.
- **Genuinely novel = the system-level synthesis + the contamination-resistant construction.**
  **Perishable** = the "unoccupied intersection" is a point-in-time scan; the neighbourhood is moving
  fast (re-scan before any submission).

## Significance ceiling (honest)

- **Where we are:** a credible **workshop / CLeaR-class** result — an *identifiability finding* on a
  leakage-resistant, NL-authored testbed, with disclosed residuals (R²-sortability 0.60; a role-type
  anti-cliché prior; difficulty descriptive-only). The contribution is the apparatus, framed (per the
  NeurIPS Evaluations & Datasets norm) as a benchmark, not a method — "beating a baseline is not
  required."
- **v0.35 progress against the old ceiling:** all three Pearl rungs are live (association /
  intervention / counterfactual, cross-sectional *and* temporal); **additive nonlinearity now ships**
  ([#10](https://github.com/noumenal-ai/causal-worlds/issues/10) first cut) with a powered nonlinear
  crossover demonstrating that even-transform edges are invisible to correlation but recovered under
  intervention.
- **What would make it a flagship / arXiv anchor:** scale to **≥100 worlds**; **multi-judge
  faithfulness**; nonlinear **interactions / post-nonlinear** + a *powered, anti-cliché-gated*
  nonlinear and temporal benchmark **set** (today's nonlinear crossover is a controlled sweep, the
  temporal set is small); close the **R²-sortability residual** (engage UUMC, CLeaR 2025); and secure
  an arXiv endorser to timestamp the perishable novelty.
