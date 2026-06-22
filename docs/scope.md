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
requirement** — and therefore no need for calibration data. This is the deliberate simplification that makes the
problem tractable: the authored structure *is* the ground truth.

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
(ground-truth answer-key)** — is unoccupied. Each neighbor owns a corner: G-Sim calibrates to real data; DEVS-Gen
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

## 7. Success criteria (v0)

A generated world is "good" when it is:
1. **Executable** — runs to a horizon without error, respects its I/O contract.
2. **Internally consistent** — obeys declared constraints (no impossible states; conservation where claimed;
   effects follow causes).
3. **Non-degenerate & learnable** — the causal task is neither trivial nor unidentifiable: a known
   causal-discovery method recovers the answer-key meaningfully above chance, and interventions move outcomes.
4. **Faithful to the description** — a reader agrees the world matches the prose.

## 8. Open questions (the frontier — tracked in hld/lld)

- **NL → a *complete* generative SCM** (graph **and** explicit functional forms / noise) from prose alone — the
  piece no public tool does.
- **Honesty without calibration** — how strict the consistency / identifiability / non-degeneracy checks should be.
- **Temporal / regime authoring** — lags, regime-switching, seasonality (absent in LLM-authored prior art).
- **Substrate boundary** — when SCM-sampling vs. discrete-event synthesis; how the discrete-event topology
  projects into the answer-key graph (see [hld.md](hld.md)).
