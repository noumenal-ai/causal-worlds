# Low-Level Design — causal-worlds

> **Status:** skeleton. Intentionally thin — LLD firms up only after the [hld.md](hld.md) decisions settle. This
> file lists the concrete pieces to design and the choices each one pins down. Nothing here is final.

## A. The world-spec / answer-key schema  *(HLD §2, §9.1)*

The single IR. To pin down:
- Serialization (JSON/YAML; a versioned schema).
- `Variable`: name, role enum, bounds (soft/hard), unit, cadence/Δt.
- `CausalEdge`: from, to, direction, lag, effect sign/shape, mechanism note.
- `Mechanism` / functional form: how a node is computed from its parents + noise (the part no prior tool authors
  from pure NL).
- `Regime`: id, switching condition, per-regime parameter overrides.
- Validation: typed, units-consistent, acyclic-where-required.

## B. Authoring (NL → spec)  *(HLD §1.1, §6)*

- Prompt design for the staged plan (topology/skeleton → per-node mechanism), à la DEVS-Gen.
- Decomposition so components are authored in bounded, parallelizable units.
- Output is the schema in §A, not free-form code.
- **Open:** how much the LLM authors closed-form equations vs. small code mechanisms; structured-output strategy.

## C. Consistency / conformance checks (calibration-free)  *(HLD §4)*

- Structural checks (acyclicity, types/units/bounds).
- Dynamic checks (conservation/capacity, effects-after-causes, no impossible states) via trace conformance.
- **Learnability gate:** identifiability of the answer-key from emitted data; non-triviality (baseline-vs-chance;
  interventions move outcomes).
- Failure → structured feedback object → bounded re-authoring loop (stopping rule TBD).

## D. Compile → executable substrate  *(HLD §3)*

- **SCM path:** spec → pgmpy / DoWhy-GCM model → CausalPlayground Gymnasium wrapper.
- **Discrete-event path:** spec → component topology + ports → parallel component synthesis → assembled package +
  entry script; emit JSONL event trace.
- Substrate selection rule (by world type / spec features).
- Discrete-event **topology → answer-key-graph projection** (how ports/interactions become IR edges).

## E. Temporal / regime dynamics  *(HLD §3, §9.4)*

- Lags, seasonality, trends (TimeGraph-style parametric SEM as a starting point).
- Regime-switching representation + sampling.

## F. Run / perturb / intervene / counterfactual  *(HLD §1.4)*

- Gym interface (`reset`/`step`/seed/horizon).
- Perturbation API (shift a disturbance, change a parameter).
- `do(x)` intervention; counterfactual replay (abduction–action–prediction, SD-SCM pattern).

## G. Evaluation harness  *(HLD §5)*

- The **pluggable agent interface** (an external causal-discovery / control agent gets data + the gym; returns a
  recovered structure / policy).
- Scoring: SHD, F1 (structure); interventional accuracy; counterfactual accuracy.
- Report format.

## H. Packaging

- Python package layout, public API surface, examples, a couple of seed worlds (a supply-chain world; a
  continuous/energy-style world), docs.

---

### Build order (proposed)
1. **A** (schema) — everything depends on it.
2. **D-SCM** + **F** + **G** on a hand-written spec — prove the *executable + answer-key + scoring* spine end-to-end
   before any LLM authoring.
3. **B** (NL→spec authoring) + **C** (checks) — close the loop from prose.
4. **E** (temporal/regime) and **D-discrete-event** — the harder substrate + dynamics.
