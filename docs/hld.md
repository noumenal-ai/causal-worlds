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

**[v0 DECISION 2026-06-22, post-review]** v0 ships **only the SCM-sampling substrate** — its answer-key is correct
**by construction** (the declared graph *is* the sampler, so emitted data cannot contradict the key). The
discrete-event substrate is **deferred**: its answer-key is *projected* from synthesized-code topology and is
**unverifiable in fiction** (no external signal catches a projection mismatch → a benchmark that can silently lie),
and it executes LLM-written code (a sandboxing problem). The table below is the *eventual* design; **only the first
row is v0.**

The executable substrate is **not one-size**; it splits by world type, unified by the IR:

| World type | Substrate | Answer-key | Honesty signal |
|---|---|---|---|
| **Continuous / known-SCM** (variables + equations; e.g. energy/process) | **SCM sampling** (pgmpy / DoWhy-GCM) wrapped as a gym (CausalPlayground bridge) | **native** (the SCM *is* the graph) | structural sanity + simulation invariants |
| **Discrete-event operations** (queues, supply chains, workflows; events + timing) | **staged discrete-event synthesis** (DEVS-Gen-style: topology → ports → parallel component code) | **projected** from the component-interaction topology into the IR graph | **trace conformance** (operational + micro/macro-causal) |

Adopted **regardless of substrate:** DEVS-Gen's *staged decompose-then-synthesize* generation (robust vs.
monolithic codegen) and *spec-derived trace conformance* (works for fiction — no real data needed).

*Deferred (not v0):* the discrete-event substrate, its topology→answer-key-graph projection fidelity, and mixed
worlds (scope §1a). The unverifiable projection (#2) is exactly why it waits.

## 4. The honesty / validity layer — the novel core (designed & spiked FIRST)

This is the project's hard core. It is designed and **spiked before the rest is built** (lld §0), not deferred —
prior drafts named it and moved on; that was the main design error the review caught. Because there is no real data,
"correct" is replaced by **coherent + identifiable + a *valid* benchmark**, all **measured**:

- **Structural (mechanical):** acyclic; typed/units-consistent; declared bounds respected.
- **Dynamic (mechanical):** conservation/capacity constraints hold; no impossible states.
- **Identifiable *by design*, not by check:** the generator **owns the gym**, so it emits **interventional /
  experimental data** alongside observational. Interventions break Markov-equivalence → the authored graph is
  recoverable. We do **not** attempt to verify identifiability of an arbitrary SCM (a research problem); we
  *provide the data that makes the authored graph identifiable*.
- **Answer-key correct by construction:** SCM path only — the declared graph is the sampler, so the data cannot
  contradict the key (no projection; that's why discrete-event is out of v0, §3).
- **Non-trivial & learnable — *measured*:** run a **pinned reference discoverer** (e.g. PCMCI+ at a fixed config)
  over N seeds; require recovered-SHD to beat a **random-graph null** by margin *m*. Fail (too noisy/unidentifiable,
  or trivial) → reject & re-author.
- **Noise is a first-class knob** (lld §E): too little → trivial/unidentifiable; too much → unrecoverable. Tuned so
  the learnability gate passes. This is the single most important dial and was undesigned before.

Check failures → structured feedback → **bounded re-authoring loop** (stop when all gates pass or the authoring
budget is exhausted).

## 4a. Benchmark validity — the circularity threat (taken seriously, not waved off)

If the generator (an LLM) and the agent-under-test (an LLM causal reasoner) share priors, a high recovery score can
reflect a **shared cheat-sheet**, not discovery-from-data. World-*coherence* ≠ benchmark-*validity*. Defended by
design:

1. **The validity-anchor discoverer is *statistical*** (PCMCI+/PC/NOTEARS) — it sees only data, so there is **no
   shared brain**. An LLM-reasoning discoverer may be evaluated as a *separate track*, never as the validity anchor.
2. **A prior-only baseline is a built-in validity meter:** a discoverer given variable names + prose but **no data**.
   If it already scores high, the world is a cliché ⇒ down-weight / reject (scope §7.5).
3. **Anti-cliché worlds are a generator *requirement*, not a hope:** sign-flips, hidden confounders, regime-dependent
   sign changes, surprising lags — worlds where priors actively mislead, so only data + intervention recover the
   truth. (Worked example in the review log; e.g. a "scarcity regime" where price↓ ⇒ demand↓.)

This converts circularity from an unaddressed hole into a **measured, designed-against property** — the prior-only
gap (statistical-discoverer score − prior-only score) is itself a reportable benchmark-quality metric.

### 4b. Spike evidence (lld §0, 2026-06-22) — the knobs, now measured not guessed

A numpy-only spike ([`spikes/spike_coffee.py`](../spikes/spike_coffee.py)) ran the validity core on a deliberately
hard **anti-cliché** world (regime sign-flip on price→demand: −1 promo / +1 scarcity; a **hidden confounder** `L`
making overtime↔sales spuriously correlated). **Result: benchmark-validity core CONFIRMED** — and it pins the
previously-TBD knobs:

| Knob (was TBD) | Measured / resolved |
|---|---|
| **Random-null reference** | SHD ≈ **7.5** (random 7-edge graph over 6 nodes) — the chance floor the non-triviality gate must beat. |
| **Anti-cliché meter** | prior-only SHD = **2** *and* it gets the scarcity sign backwards ⇒ the world genuinely defeats priors. Gate: reject if prior-only ≈ the data discoverer. |
| **Observational insufficiency (by design)** | observational PC **drops** the sign-flipped P–D (the −1/+1 regimes cancel marginally) **and keeps** the confounded O–S (no observed subset d-separates them). SHD 2. |
| **Identifiability *by construction*** | emitting **do-data** takes SHD **2 → 0**: do(P) restores P–D with correct per-regime signs (−1.00 / +1.00); do(O)→S ≈ 0 removes the spurious edge. **The load-bearing #5 decision, now proven on a hard case.** |
| **Reference discoverer** | must be a proper **PC-family adjacency search** (a full-conditioning/moral-graph version over-connects — a real bug the spike caught and fixed) **and regime/intervention-aware** (a naive linear discoverer is blind to sign-flips). |
| **Noise** | σ ≈ 0.3 gave clean separation; it is the learnability dial (too low → trivial, too high → unrecoverable). |

**Honest caveat (what the spike does NOT settle).** Its interventional discoverer is *hand-targeted* at the two
planted traps — so it proves interventional data **contains the information** to fix what observational + priors
miss, **not** that a *general* interventional-discovery algorithm recovers the graph automatically. **First
HLD/LLD task this unblocks:** a principled interventional procedure (GIES / systematic per-variable do-tests), not
a per-trap patch.

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
- **Cost / scale (honest):** each world costs LLM authoring + a re-authoring loop + per-world reference-discovery
  runs (the learnability gate). "Generate at scale" is **bounded by this cost** — not free. Mitigations: cache
  authored specs, batch the gate runs, and gate-then-sample (sampling many trajectories from one admitted spec is
  cheap; *authoring + admitting* a spec is the expensive part). Don't claim "infinite environments" without the
  cost caveat.
- **Language:** Python-first (the causal/gym ecosystem). An optional orchestration layer in another language is a
  later, separable concern.
- **Dependency discipline:** mature deps (pgmpy, DoWhy) are real dependencies; **research repos (CausalPlayground,
  TimeGraph) are treated as *patterns to reimplement*, not pinned dependencies** — fast-moving, uncertain
  maintenance. "Compose the donors" is not free integration; v0 minimizes the live-dep surface.
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
