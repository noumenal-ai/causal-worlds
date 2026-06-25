# Foundations — are causal-worlds worlds *truly* causal?

A causal benchmark is only as trustworthy as its claim to be *causal*. This doc grounds the design in
the founding literature and answers the question rung by rung. Short version: **yes — in the full
structural sense, on cross-sectional worlds** (Markov, faithful, modular, all three rungs of Pearl's
ladder), with a few scoped gaps tracked as issues.

## The ladder (Pearl)

"Causal" is not one property — it's a hierarchy (Pearl's *Ladder of Causation*; the Pearl–Bareinboim
**Causal Hierarchy Theorem** proves the rungs are strictly separated — you cannot climb from data
alone):

| Rung | Question | causal-worlds |
|---|---|---|
| **1 — Association** | what goes with what? | `substrate.sample(...)` |
| **2 — Intervention** | what if I *act*? | `substrate.sample(..., do={...})` |
| **3 — Counterfactual** | what *would have* happened? | `counterfactual(spec, do, seed=...)` |

## The founders (why an edge is causal, not just correlated)

- **Sewall Wright (1921), path analysis** — an edge is a *coefficient*, and the diagram *generates*
  the data. Our `Term(parent, coeff)` is a Wright path coefficient; the renderers label every edge
  with it.
- **Hans Reichenbach (1956), common-cause principle** — a correlation is `A→B`, `B→A`, *or a shared
  hidden cause*; the data alone can't say which. This is the built-in trap: `local_buzz` confounds
  `overtime ~ sales`. We *declare* the hidden cause, so we know the answer the data can't give.
- **Trygve Haavelmo (1944), autonomy** — a relation is causal only if it stays put when you intervene
  *elsewhere* (modern: *Independent Causal Mechanisms*; Pearl: *modularity*). This is what makes
  `do()` meaningful.

## The axioms, and how we satisfy them

| Property | Guarantees | In causal-worlds |
|---|---|---|
| **Markov** | the graph implies the right independences | ✅ by construction — it *is* an SCM |
| **Faithfulness** (Spirtes–Glymour–Scheines) | every edge leaves a detectable fingerprint (discovery is fair) | ✅ enforced by the **T3 gate** (`check_faithfulness`: every edge ⇒ \|partial corr\| ≥ 0.05; unfaithful/canceling-path worlds are rejected) |
| **Autonomy / modularity** (Haavelmo) | mechanisms are independent, so intervention is well-defined | ✅ by construction + verified — `do()` replaces one mechanism and runs the rest unchanged |

## The rungs, verified

- **Rung 1 — association.** Sampling the SCM. ✅
- **Rung 2 — intervention.** `do()` is *genuine graph surgery*, verified line-by-line in
  `sample/substrate.py`: the intervened variable's mechanism is skipped (its **incoming edges cut**)
  and the forced value propagates downstream — **not** statistical conditioning. ✅ The renderers can
  draw it: `to_dot(spec, do={...})`.
- **Rung 3 — counterfactual.** `causal_worlds.counterfactual` implements Pearl's
  **abduction → action → prediction**: recover the unit's exogenous noise in closed form, apply the
  `do`-surgery, re-run with the noise held fixed. Exact, because the SCM is declared. ✅ *(autonomy is
  what licenses holding every other mechanism fixed.)*

## Scoped residuals (tracked)

1. **Faithfulness is enforced, not stress-tested** — real systems can be unfaithful (canceling
   paths); we gate those out. An opt-in "unfaithful / hard mode" is proposed in
   [#18](https://github.com/noumenal-ai/causal-worlds/issues/18).
2. **Valid intervention targets** (Woodward's interventionism) — our `controllable` role *asserts* a
   variable is a legitimate `do()` target; only lightly examined.

*(Resolved: temporal counterfactuals — `counterfactual_temporal` now rolls a whole trajectory under a
sustained intervention, so Rung 3 covers cross-sectional and temporal worlds.)*

## References

Pearl, *Causality* (2009) & *The Book of Why*; Pearl & Bareinboim, the Causal Hierarchy Theorem;
Wright (1921), *Correlation and Causation*; Haavelmo (1944), *The Probability Approach in
Econometrics*; Reichenbach (1956), *The Direction of Time*; Spirtes, Glymour & Scheines, *Causation,
Prediction, and Search*; Woodward, *Making Things Happen*.
