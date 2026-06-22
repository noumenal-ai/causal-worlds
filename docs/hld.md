# High-Level Design — causal-worlds

> **Status:** v0 draft, early — component boundaries and the key decisions, not final APIs. Builds on
> [scope.md](scope.md); detail lands in [lld.md](lld.md).

## 1. Shape

A **pipeline** that compiles a natural-language description into the artifact triple (simulator + dataset +
answer-key), plus a **scoring harness** that grades an external agent against the answer-key.

```
NL description
  │
  ▼
[1] Author      LLM proposes a WORLD SPEC: variables + roles, causal graph (edges: direction, lag),
                functional forms / mechanisms, regimes. Staged + decomposed (not monolithic).
  │
  ▼
[2] Check       Consistency / conformance — NO calibration data:
                acyclicity · declared-constraint & conservation checks · non-degeneracy ·
                identifiability of our own answer-key · (discrete-event) trace conformance.
                Failures → structured feedback → re-author (bounded refinement loop).
  │
  ▼
[3] Compile     SPEC → executable substrate (chosen by world type, §3) + freeze the ANSWER-KEY.
  │
  ▼
[4] Run / Eval  reset/step/intervene/counterfactual; emit dataset + trace;
                score a pluggable causal-discovery / control agent vs the answer-key.
```

Stage [1]'s **decompose-then-synthesize** structure and stage [2]'s **trace-conformance** are adopted from
DEVS-Gen; stage [3]'s SCM path and answer-key sampling from pgmpy / DoWhy-GCM / CausalPlayground; stage [4]'s
counterfactual machinery from the SD-SCM pattern; the temporal generators from TimeGraph.

## 2. The IR — one schema, the pivot

Everything routes through a single **world-spec / answer-key schema** (an open, documented causal-world format):

- **Variables** — name, role (`controllable` | `observable` | `disturbance` | `outcome`), bounds, unit, cadence.
- **Causal edges** — `from → to`, direction, **lag**, effect sign/shape, mechanism note.
- **Functional forms** — the structural equations / transition logic (how children depend on parents).
- **Regimes** — named operating regimes and what switches them (regime-conditioned parameters).

The spec is both the **build input** to the compiler *and*, once frozen, the **answer-key** the harness scores
against. Keeping one schema is what makes scoring a recovered structure a trivial graph comparison. Export
adapters can project it to external agents' schemas.

## 3. Two substrates, one IR (the key design decision)

The executable substrate is **not one-size**; it splits by world type, unified by the IR:

| World type | Substrate | Answer-key | Honesty signal |
|---|---|---|---|
| **Continuous / known-SCM** (variables + equations; e.g. energy/process) | **SCM sampling** (pgmpy / DoWhy-GCM) wrapped as a gym (CausalPlayground bridge) | **native** (the SCM *is* the graph) | structural sanity + simulation invariants |
| **Discrete-event operations** (queues, supply chains, workflows; events + timing) | **staged discrete-event synthesis** (DEVS-Gen-style: topology → ports → parallel component code) | **projected** from the component-interaction topology into the IR graph | **trace conformance** (operational + micro/macro-causal) |

Adopted **regardless of substrate:** DEVS-Gen's *staged decompose-then-synthesize* generation (robust vs.
monolithic codegen) and *spec-derived trace conformance* (works for fiction — no real data needed).

*Open:* the exact boundary (some worlds are mixed), and the fidelity of the topology→graph projection for the
discrete-event answer-key.

## 4. The honesty layer (fiction-specific)

Because there is no real data to calibrate against, "correct" is replaced by **coherent + non-degenerate**:
- **Structural:** acyclic (where required), typed/units consistent, declared bounds respected.
- **Dynamic:** conservation / capacity constraints hold; effects follow causes; no impossible states (the
  DEVS-Gen operational + micro/macro-causal trace checks).
- **Learnability:** the answer-key is **identifiable** from the emitted data (not unidentifiable equivalence
  classes) and **non-trivial** (a baseline discovery method does better than chance, and interventions move
  outcomes) — otherwise the world is a useless benchmark.

Check failures generate structured feedback that drives a **bounded re-authoring loop**.

## 5. The test-maker / test-taker split

`causal-worlds` is the **test-maker** (worlds + answer-keys). The **causal-discovery / control agent under test is
external and pluggable** behind a thin interface; the harness scores its recovered structure (SHD / F1) and its
interventional / counterfactual predictions against the frozen answer-key. This keeps the library generally useful
and keeps any particular agent's internals out of this repo.

## 6. Components & reuse (donors)

| Stage | Reuse | Build |
|---|---|---|
| Author (NL→spec) | LLM client; DEVS-Gen staged-planning pattern; G-Sim/GIF-MCTS code-as-model | **NL → complete SCM with explicit functional forms** |
| Check | DEVS-Gen trace conformance | **calibration-free consistency + identifiability/non-degeneracy** |
| Compile/run | pgmpy, DoWhy-GCM, CausalPlayground (gym), TimeGraph (temporal SEM: lags/seasonality) | **temporal/regime authoring**; topology→graph projection |
| Eval | SD-SCM (abduction→counterfactuals); standard SHD/F1 | pluggable agent interface + scoring wiring |

## 7. Non-functional

- **Determinism:** seeded; same spec + seed → same world and data.
- **Language:** Python-first (the causal/gym ecosystem). An optional orchestration layer in another language is a
  later, separable concern.
- **Provenance & honesty:** every artifact carries its generating spec, seed, and a fiction label.

## 8. Boundaries

Not real-world fidelity; not a discovery method; not pixels/3D; not real-system control (see [scope.md](scope.md)
§6). The LLM is a **proposer**, never trusted as a calibrated oracle — the check layer (§4) is what makes the
output dependable.

## 9. Open decisions (→ lld.md)

1. The concrete world-spec schema (fields, serialization).
2. Substrate boundary + the discrete-event topology→answer-key-graph projection.
3. How strict the §4 honesty checks are, and the re-authoring loop's stopping rule.
4. Temporal/regime representation (how lags/regimes are authored and sampled).
5. The pluggable agent interface + the exact scoring suite.
