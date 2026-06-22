# Scope — causal-worlds

> **Status:** v0 draft, early. The "what and why" (and what's explicitly out). Architecture is in
> [hld.md](hld.md); component detail in [lld.md](lld.md).

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
cannot contradict the key). That is precisely why v0 (§1a) collapses to that path.

## 1a. v0 focus — pick ONE payoff [DECIDED 2026-06-22, post-review]

This tool could serve four masters (discovery benchmark, control gym, describe-a-world amplifier, internal eval
harness) with *conflicting* requirements. v0s die trying to serve all of them. **v0 = the causal-discovery
benchmark, on the pure-SCM path, scoring structure recovery only.** Precisely: **causal-*structure* (directed-graph)
recovery** — it does **not** score causal *effect magnitudes* or *counterfactuals* (those need principled magnitudes
→ later). Name it a **structure-recovery benchmark, not a "full causal" one.** Rationale: it is the only payoff where
the answer-key is the product, the key is correct-by-construction (§2 / #2), and **magnitudes don't bite** (structure
scoring sees edges, not effect sizes — so the "magnitude-soft" property is fine here and *only* here).

**Explicitly deferred to later milestones (NOT v0):**
- **Control gym / `do(x)` / counterfactuals** — needs *principled magnitudes* (the optimal policy and counterfactual
  outcomes are functions of effect sizes); arbitrary magnitudes ⇒ an arbitrary control benchmark. A fixed fictional
  gym is *internally* valid (its own ground truth) but not externally meaningful — a later, clearly-scoped thing.
- **Describe-a-world amplifier** (delight-optimized) and the **conversational elicitation UX** (see lld §B2).
- **Discrete-event substrate** — its answer-key is *projected* (not by-construction) and unverifiable in fiction
  (#2) → not a sound benchmark yet.

Everything below describes the eventual full scope; the **v0 slice** is the SCM-path discovery benchmark + the
validity layer (§7, hld §4/§4a).

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
  outcome), a causal graph (edges with direction + lag), functional forms / mechanisms, and regimes.
- **Manifest:** ties the three together; records the seed, the generating spec, and an **honesty label**
  (fictional; not real-world advice).

## 5. In scope

- NL → world spec authoring (the causal structure + mechanisms).
- A consistency / conformance layer that keeps the authored world honest **without calibration data**.
- Two executable substrates (SCM-sampling and discrete-event), selected by world type — one shared IR.
- Perturbation / intervention / counterfactual support.
- A scoring harness that grades a pluggable, external causal-discovery agent against the answer-key.

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
- *(Deferred with their milestones:)* temporal/regime authoring; the discrete-event substrate + topology→graph
  projection; the control-gym magnitude problem; the conversational elicitation UX.
