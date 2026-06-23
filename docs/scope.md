# Scope — causal-worlds

> **Status:** v0 discovery track shipped; **control track now in scope** (§1a, updated 2026-06-23).
> The "what and why" (and what's explicitly out). Architecture is in [hld.md](hld.md); component
> detail in [lld.md](lld.md).

## 1. Problem & goal

Turn a **plain-language description of an operation** into a **fictional but internally-coherent causal world**,
delivered as three coupled artifacts:

1. an **executable simulator** (Gymnasium-style),
2. the **multivariate time-series** it emits, and
3. a **declared ground-truth causal structure** — the *answer-key*.

The defining requirement is **#3**. Most synthetic-world tooling produces *plausible data* or a *runnable
environment*; almost none emits the **causal structure that generated it** as a first-class, scoreable artifact.
The answer-key is what turns a generator into an **evaluation instrument**.

**Bar:** *plausible + internally consistent*, explicitly **fictional**. There is **no real-world-fidelity
requirement** — and therefore no need for calibration data.

**Honest caveat (don't oversell "the easy corner").** Removing fidelity does *not* make this easy — it trades one
hard problem (calibrate to reality) for another (author a world that is simultaneously coherent, identifiable,
non-trivial, *anti-cliché*, faithful-to-prose, and whose declared key matches the executable's behavior — with **no
external signal** to tell us when we failed). The one place fiction is genuinely easier is the **pure-SCM path**,
where the declared graph *is* the generative process, so the answer-key is **correct by construction** (the data
cannot contradict the key). That is why the discovery track lives on that path — **and the same property extends to
control**: because the mechanisms are *declared*, the optimal policy and counterfactual outcomes are *functions of
known quantities*, so a **control answer-key is correct-by-construction too** (§1a). The SCM path serves both.

## 1a. Scope stages — discovery shipped, control next [UPDATED 2026-06-23, supersedes the 2026-06-22 "pick ONE" decision]

This tool could serve four masters (discovery benchmark, control gym, describe-a-world amplifier, internal eval
harness). The 2026-06-22 decision was to ship **one at a time** — and that *sequencing* discipline stands. What
**no longer stands** is the claim that control is a fundamentally *later, harder* thing blocked by a "magnitude
problem." It isn't. Both tracks live on the same pure-SCM path and share one correct-by-construction guarantee.

**The reframe (why control is now in scope).** The 2026-06-22 note deferred control because "arbitrary magnitudes ⇒
an arbitrary control benchmark — internally valid but not externally meaningful." That conflated two different
bars. **External meaning was never our bar** — not for control, and *not for discovery either*. The discovery
answer-key is correct relative to the *declared* world, not to reality (§1 — fiction-first, no fidelity
requirement). The **identical** logic yields a control answer-key: given a declared SCM, a declared **objective**
(reward over outcome variables), and a declared **action space** (controllable variables + admissible ranges), the
**optimal policy and the counterfactual outcomes are deterministic functions of the *known* mechanisms** — so they
are **correct-by-construction**, exactly like the graph. Arbitrary magnitudes no more make a control benchmark
"arbitrary" than arbitrary edges make the discovery benchmark arbitrary; in both cases the world *is* its own
ground truth. The magnitude problem dissolves.

**Stage 1 — discovery benchmark (SHIPPED, on PyPI).** Causal-*structure* (directed-graph) recovery on the pure-SCM
path: author → judge → gate → admit; grade a pluggable discoverer against the by-construction graph answer-key.
Does **not** score effect magnitudes. This is the proven core (§7, hld §4/§4a).

**Stage 2 — control benchmark (NOW IN SCOPE — the noumenal_agent eval tests *both* tracks).** The same admitted
worlds, plus three declared additions per world — an **objective**, an **action space**, and a **perturbation
model** (regime/distribution shifts; we already author regimes) — and two derived answer-keys computed offline from
the known SCM:
- a **ground-truth optimal policy** (closed-form for linear-Gaussian + quadratic cost; offline optimization over
  the known mechanisms otherwise), and
- a **counterfactual engine** (abduction → action → prediction on the known SCM) for counterfactual replay.

The control **score** is **regret vs. the known optimum** and — the load-bearing one for the World Models thesis —
**regret *under perturbation*** (does a policy *stay* near-optimal when the regime shifts?). That is precisely what
noumenal_agent is built to do, so the control track is what makes the private eval test the agent's real
differentiator, not just its discovery sub-skill. Stage 2 is **real build work** (objective/action-space authoring
from prose; a tractable optimal-policy solver; the counterfactual engine; new gates) — but it is *engineering on a
sound foundation*, not a blocked-on-philosophy "later thing."

**Still deferred (genuinely later):**
- **Describe-a-world amplifier** (delight-optimized) and the **conversational elicitation UX** (see lld §B2) —
  product-experience work, orthogonal to the benchmark's soundness.
- **Discrete-event substrate** — its answer-key is *projected* (not by-construction) and unverifiable in fiction
  (#2) → not a sound benchmark yet.

The **pure-SCM path now carries both Stage 1 and Stage 2**; the discrete-event substrate and the amplifier UX
remain out until their soundness/scoping is resolved.

## 2. Who it's for / use cases

- **Benchmark causal-discovery methods at scale.** Generate many operations with known graphs; score recovery
  (SHD / F1, and interventional / counterfactual accuracy). Like TimeGraph/SD-SCM, but **natural-language-authored
  and executable**.
- **Train & stress-test control / decision agents.** A gym with first-class **perturbations, `do(x)`
  interventions, and counterfactual replay** — evaluate whether a policy stays good when the world shifts.
- **A "describe-a-world" sandbox** for exploring causal world-models interactively.

The agent being evaluated (a causal-discovery method, a controller) is **pluggable and external** — see the
test-maker / test-taker split in [hld.md](hld.md).

## 3. Positioning (the gap)

The four-way intersection — **(NL authoring) × (temporal/regime causal structure) × (executable simulator) ×
(ground-truth answer-key)** — **appears unoccupied as of mid-2026** (a fast-moving field with new 2026 papers
cited here — treat as "no current match found," not a durable moat; revisit). Each neighbor owns a corner: G-Sim
calibrates to real data; DEVS-Gen
validates against constraints (no declared graph); SD-SCM needs a user-supplied DAG and is tabular; TimeGraph is
parametric, not NL-authored; GIF-MCTS/WorldCoder have no causal answer-key. `causal-worlds` is built **by
composing** these donors and filling the two genuinely-novel pieces (§7).

## 4. Outputs (the artifact triple + manifest)

- **Simulator:** `reset()` / `step(action) → (obs, reward, done, info)`; supports interventions and a configurable
  horizon. Deterministic given a seed.
- **Dataset:** the emitted multivariate time-series (+ an event trace for discrete-event worlds).
- **Answer-key:** an open causal-world schema — variables (with roles: controllable / observable / disturbance /
  outcome), a causal graph (edges with direction + lag), functional forms / mechanisms, and regimes. *(Discovery
  key — Stage 1.)*
- **Control key (Stage 2):** a declared **objective** (reward over outcomes), an **action space** (controllable
  variables + admissible ranges), a **perturbation model** (regime/distribution shifts), and the two quantities
  *derived* from the known SCM — the **ground-truth optimal policy** and a **counterfactual engine** — all
  correct-by-construction.
- **Manifest:** ties everything together; records the seed, the generating spec, and an **honesty label**
  (fictional; not real-world advice).

## 5. In scope

- NL → world spec authoring (the causal structure + mechanisms).
- A consistency / conformance layer that keeps the authored world honest **without calibration data**.
- Two executable substrates (SCM-sampling and discrete-event), selected by world type — one shared IR.
- Perturbation / intervention / counterfactual support.
- A scoring harness that grades a pluggable, external agent against the answer-key — a **discovery** agent against
  the graph key (Stage 1), and a **control** agent against the by-construction optimal policy via regret and
  regret-under-perturbation (Stage 2).

## 6. Out of scope / non-goals

- **Real-world fidelity / digital twins.** Worlds are fictional; we do not claim to match any real system. (Tools
  that *do* — G-Sim, hybrid digital twins — solve a different, harder problem requiring real data.)
- **Being a causal-discovery method.** We make the *test*, not the *test-taker*; discovery/control agents are
  external and pluggable.
- **Pixels / 3D / embodied worlds.** Operations and time-series, not visual generation.
- **Closed-loop control of real systems.**

## 7. Success criteria (v0) — measurable gates, not vibes

A generated world is admitted to the benchmark only if it passes **measured** gates (see hld §4):
1. **Executable** — runs to a horizon without error; respects its I/O + data schema.
2. **Internally consistent** — declared constraints hold (no impossible states; conservation where claimed).
3. **Identifiable by construction** — the world ships **interventional data** alongside observational, so the
   authored graph is recoverable (interventions break Markov-equivalence). Key is correct-by-construction (SCM path).
4. **Non-trivial & learnable (measured)** — a pinned reference discoverer (e.g. PCMCI+) over N seeds beats a
   random-graph null by margin *m*. Too-noisy/unidentifiable or too-trivial worlds are rejected.
5. **Anti-cliché / non-circular (measured)** — a **prior-only** discoverer (variable names + prose, *no data*) does
   **not** already solve it; if it does, the world is a cliché and the benchmark is weak → reject or perturb
   (flip a sign, add a confounder, shift a lag).
6. **Faithful to the description (measured, not "a reader agrees")** — a rubric-scored check (LLM-judge and/or human
   rating against a checklist of the prose's claims), with a threshold; below it → re-author. *(Subjective
   "someone agrees" is explicitly replaced by this.)*

**Control-track gates (Stage 2, added to the above for control worlds):**
7. **Optimal policy is recoverable by construction** — the declared SCM + objective + action space yield a
   computable optimum (closed-form or offline-solved), verified by a reference controller that achieves ~zero
   regret against it. Worlds where the optimum is ill-defined or intractable are rejected.
8. **Control is non-trivial & perturbation-sensitive (measured)** — a *static* baseline policy (ignore the regime)
   must incur materially higher regret than the regime-aware optimum under the perturbation model; otherwise the
   world doesn't actually test staying-optimal-under-shift → reject or sharpen the perturbation.

## 8. Open questions (the frontier — tracked in hld/lld)

- **The AUTHOR step is STILL UNTESTED — it is build-task-0.** lld §0 spikes (#1–#2) validated the *test harness*
  (given a **hand-authored** SCM: a hard world defeats priors, and a general interventional discoverer recovers it).
  They did **not** test whether an **LLM can author** a valid, anti-cliché, **gate-passing** SCM **from prose** —
  the actual product and the riskiest assumption. Measure that (author N worlds → pass-rate through T1–T4, first-try
  + within K iters) **before** building the pipeline around it. Don't conflate "harness works" with "author works."
- **Benchmark validity / circularity** — generator and agent-under-test may share LLM priors; defused by (a) scoring
  a *statistical* discoverer, (b) the prior-only meter, (c) required anti-cliché worlds (hld §4a). Still the
  highest-stakes open risk.
- **NL → a *complete* generative SCM** (graph **and** explicit functional forms / **noise**) from prose alone — the
  piece no public tool does. **Noise is a first-class designed knob** (the learnability dial), not an afterthought.
- **Cost / scale** — per-world LLM authoring + a re-authoring loop + per-world reference-discovery runs (the
  learnability gate) is real compute per world; "scale" is bounded by cost. Needs caching/batching/amortization.
- **Control track (Stage 2) — the new frontier:** the "magnitude problem" is *resolved in principle* (§1a: the
  optimal policy is correct-by-construction from the known SCM), but its *execution* is open — (a) authoring a
  coherent **objective + action space + perturbation model** from prose, gate-passing, anti-cliché (a degenerate
  objective whose optimum is "do nothing" is the control analogue of a name-guessable graph); (b) keeping the
  **optimal-policy solver tractable** for nonlinear/temporal worlds (closed-form only covers linear-Gaussian +
  quadratic cost); (c) a **counterfactual engine** correct under regime switches. These are the Stage-2 build risks.
- *(Deferred with their milestones:)* the discrete-event substrate + topology→graph projection; the
  describe-a-world amplifier + conversational elicitation UX.
