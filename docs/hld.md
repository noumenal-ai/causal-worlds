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
- **Noise** — per-mechanism noise family + scale: a **first-class field**, the learnability dial (§4, lld §A).
- **Regimes** — named operating regimes, what switches them, and **per-regime parameter / sign overrides** — the
  *anti-cliché lever* (spike: P→D = −1 promo / +1 scarcity, a sign-flip by regime).
- **Intervention surface** — every variable is `do()`-able (we own the gym). This is *why* the answer-key is
  identifiable by construction (§4) — interventional data is always available.

The spec is both the **build input** to the compiler *and*, once frozen, the **answer-key** the harness scores
against. Keeping one schema is what makes scoring a recovered structure a trivial graph comparison. Export
adapters can project it to external agents' schemas. *Worked example (the spike world): nodes R,P,F,O,D,S + hidden
L; edges incl. the regime-flipped P→D and the F→O / F→S / D→S structure — full field list in [lld §A](lld.md).*

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

**[DECISION] Anti-cliché = a difficulty *axis*, not a blanket hard reject.** Because the validity anchor is a
*statistical* discoverer (no shared priors), a cliché world is still a **valid** test — a stats method recovering
`demand→load` from data is legitimate, and real operations *are* full of textbook relationships. Rejecting cliché
worlds would bias the generator toward only-weird worlds and shrink domain coverage. So the prior-only gap is
reported as a **per-world difficulty label** (easy-cliché → hard-anti-cliché), making the benchmark a useful
*spectrum*. The **hard reject** applies in exactly one case: when the *discoverer-under-test is LLM-based* (shared
priors would fake a pass) → there, require gap ≥ δ.

**[DECISION — forced by build-task-0 evidence, 2026-06-22] The prior-only baseline AND the faithfulness judge use
a *different LLM family* than the author.** The author spike (lld §0b) measured it: grading my own worlds (Claude
author **+** Claude prior) understated difficulty (gap ≈ 1); an **independent prior (Gemini `gemini-3.5-flash`)**
revealed the true spectrum — hospital gap **3**, ride-hailing **2**, textbook worlds **0**. Same-brain grading is
optimistic, so the prior-only meter and the prose-faithfulness judge must be a *separate* model from whatever
authored the world. (Structure-SHD stays blind to *sign-flips*, so difficulty is earned **structurally** —
confounders, non-obvious connectivity — not by regime sign-flips.)

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

**Spike #2 (2026-06-22) — caveat CLOSED: a *general* procedure recovers it.**
[`spikes/spike_coffee_general.py`](../spikes/spike_coffee_general.py) replaced spike #1's hand-targeted step with a
**uniform interventional discoverer applied identically to every variable** (no per-trap coding) and recovers the
**directed** graph at **SHD 0**, **robustly: 20/20 across 5 seeds × 4 noise levels (σ 0.2–0.8)**. The rule (the
shape of the harness's reference discoverer):
1. **Reachability:** `do(v)` data → `w` is an effect of `v` iff `do(v)` moves `w` (marginal, regime-aware → no
   collider bias).
2. **Direct edge `v→w`** iff, in the `do(v)` data, `w` still depends on `v` given **`w`'s (discovered) ancestors** —
   ancestors block indirect paths *in* and are never `w`'s descendants, so **no collider is opened**.
3. Every test **regime-stratified** → the P→D sign-flip is caught (pooled it cancels to ~0).

**The load-bearing lesson:** *the conditioning set is the whole game* — "intervene + condition on **everything**"
over-connects (collider bias → SHD 6, a real failure the spike hit); **"ancestors of the target" is the right set.**
**Remaining boundary (honest):** one world *structure*, n=8000 — robust to seed/noise but **not yet swept over world
diversity** (other DAGs/sizes/confounding), and it's a hand-rolled simplification of GES/GIES whose
reachability→edge stages couple (errors compound). → **build-task-1:** harden into a **pinned, vetted GIES-family**
reference discoverer + a world-diversity sweep.

**Thresholds are world-size-relative → normalize.** null-SHD ≈ 7.5 is for *this* 6-node/7-edge world and scales
with size; the non-triviality margin *m* and the anti-cliché gap *δ* must be expressed **relative to each world's
own null and edge count** (normalized SHD = SHD/|edges|, or "fraction of *that* world's null beaten"), **never** as
fixed absolutes. The spike numbers are one-size starting points.

### 4c. The author → gate → admit loop (control flow + stopping rules)

§4/§4a define *what* "valid" means; this is *how a world is produced* — a **bounded refinement loop** with one
invariant: **a world that fails any gate is never shipped** (the #2 "benchmark that lies" failure mode is
structurally impossible — admission requires every gate green). Gates run **cost-tiered, cheapest first,
short-circuiting on first failure** (this is the lever that controls per-world compute, §7):

| Tier | Check | Cost | On fail |
|---|---|---|---|
| **T0 · author** | LLM proposes a *complete* spec (§2 / lld §A) | 1 LLM call | — |
| **T1 · static** | acyclic · types/units/bounds · every edge used by a mechanism · ≥1 outcome & ≥1 controllable · no orphan/degenerate nodes | ~free | structured feedback → re-author |
| **T2 · sample-sanity** | compile + sample obs **+ interventional**; no NaN/inf; every var has variance; declared edges are *detectable*; no unintended near-collinearity | 1 sample | tune **noise** (the dial) / feedback → re-author |
| **T3 · non-triviality** | reference discoverer (§4b rule → vetted GIES) over N seeds beats the random-null by margin *m* | N discovery runs | too-hard → lower noise / simplify; reject if still unidentifiable |
| **T4 · anti-cliché** | prior-only gap = a **difficulty label** (§4a) | 1 discovery + 1 prior-only | **label** the world's difficulty; *hard-reject + perturb only* when scoring an LLM-discoverer |
| **admit** | all green → **freeze spec as the answer-key**; package `{gym, dataset, answer-key, manifest}` | — | — |

**T0 is the unproven step (build-task-0).** The loop assumes the author can *eventually* produce a gate-passing
world; if it can't, it spins to budget and discards → nothing ships. Validating T0's reliability (lld §0b) comes
**before** everything else here.

**Identifiability** is *not* a runtime tier — it holds **by construction** (SCM path: the declared graph *is* the
sampler) and is merely *exercised* by T3 emitting do-data (spike #2: do-data is what reaches SHD 0).

**Feedback is structured, not prose** — each failure emits `{gate, what_failed, localized_hint}` (e.g. *"T4:
prior-only already recovers 6/7 edges → add a counter-intuitive mechanism (a regime sign-flip or a hidden
confounder)"*; *"T1: edge F→S declared but no mechanism in S uses F"*). The author LLM consumes it in-context to
revise (the DEVS-Gen / G-Sim refinement pattern).

**Stopping rules (bounded — never spin):**
- **Budget:** ≤ K re-author iterations *or* a per-world token/compute cap.
- **No-progress:** if the failing-gate set hasn't shrunk in *j* consecutive iterations → stop (patience/oscillation,
  à la G-Sim early-stopping).
- **On give-up:** **discard the world** (a world that can't pass is never emitted — the whole point), logging the
  terminal gate + last feedback. In the conversational flow (lld §B2, later) this becomes a *clarifying question to
  the user* rather than a silent discard.

**Determinism:** an admitted world ships its seed → same spec + seed ⇒ same data ⇒ a published benchmark item is
exactly reproducible.

**Admission is discoverer-relative → pin & version the grader.** A world's "non-trivial / admitted" status is only
meaningful against a *named, frozen* reference discoverer (spike #2: the verdict flips with it — naive SHD 6,
principled SHD 0). The **manifest records the reference-discoverer version**; build-task-1's GIES swap can change
admitted status, so it **forces re-validation** of previously-admitted worlds. For a benchmark, *the grader's
version is part of the artifact.*

*(Knob values — N, m, δ, K, j, noise defaults — are pinned in lld; the spike gives starting points: null ≈ 7.5,
prior-only gap ≥ ~2 edges on the example world.)*

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

1. The concrete world-spec schema (fields, serialization) — shape in §2; full fields in lld §A.
2. *(Deferred, not v0)* discrete-event substrate + the topology→answer-key-graph projection.
3. ~~How strict the checks are + the loop's stopping rule~~ — **designed in §4c**; only the **numeric knobs**
   remain for LLD: N (seeds), m (non-triviality margin vs null), δ (prior-only gap), K/j (budget/patience), noise
   defaults. Spike gives starting points (null ≈ 7.5; prior-only gap ≥ ~2 edges).
4. Temporal/regime representation (how lags/regimes are authored and sampled) — *(beyond v0's static SCM)*.
5. The pluggable agent interface + the exact scoring suite (structure now; interventional/counterfactual later).
6. **Build-task-0 [basic bar MET 2026-06-22]:** author spike — **4/4** worlds authored from one-line prose passed
   T1–T3 (robust 5/5), and an **independent Gemini prior** confirmed real difficulty (gap up to 3) where my own
   self-prior had understated it (≈1). The author works at small scale; the prior/judge must be a different model
   family (§4a). *Remaining:* larger / structurally-harder worlds + the within-K re-author loop against live gates.
   (lld §0b.)
7. **Build-task-1 (then):** harden the §4b discoverer into the **pinned, versioned** reference discoverer (vetted
   GIES lib) + a **world-diversity sweep**. Note the staged spike discoverer couples reachability→edge (errors
   compound) — a vetted GIES is more robust.
