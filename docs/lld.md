# Low-Level Design — causal-worlds

> **Status:** skeleton. Intentionally thin — LLD firms up only after the [hld.md](hld.md) decisions settle. This
> file lists the concrete pieces to design and the choices each one pins down. Nothing here is final.

## 0. Spike FIRST — kill or confirm the core assumption  *(before any architecture)*

Everything rests on one untested assumption: **an LLM can author a complete, identifiable, *anti-cliché*
generative SCM (graph + functional forms + noise + one regime) from a sentence.** Prove or kill it in ~1 day, by
hand, no framework:
1. Prompt the LLM to author **one** SCM from a sentence — *deliberately anti-cliché* (e.g. the "scarcity regime"
   where price↓ ⇒ demand↓, plus a hidden confounder). Emit graph + equations + noise.
2. Sample it in plain numpy → observational **and** interventional data.
3. Run a real statistical discoverer (PCMCI+ / PC via `causal-learn`) on the data.
4. **Anti-cliché check:** also run a **prior-only** discoverer (LLM sees names+prose, no data). 
5. **Verdict:** recovered graph ≈ authored graph **for the right reason** — i.e. the statistical discoverer
   (data) beats the prior-only baseline, and recovers the *counter-intuitive* edges. If this doesn't work cleanly,
   the whole edifice waits.

This spike *is* the first work item. No schema/substrate/harness is built around the assumption until it holds.

**RESULT (2026-06-22): CONFIRMED.** [`spikes/spike_coffee.py`](../spikes/spike_coffee.py) ran the anti-cliché
coffee world (regime sign-flip + hidden confounder). Prior-only fails (SHD 2, scarcity sign backwards);
observational PC is insufficient (drops sign-flipped P–D, keeps confounded O–S, SHD 2); **interventional do-data
takes SHD → 0** with correct per-regime signs (−1.00 / +1.00). Measured knobs (null ≈ 7.5; PC-family + intervention-
aware discoverer; σ noise dial) are folded into **hld §4b**. **Caveat:** the spike's interventional discoverer is
hand-targeted at the planted traps — it proves do-data is *sufficient*, not that a *general* algorithm auto-recovers
→ **the first build task is a principled interventional procedure** (GIES / systematic do-tests), see §F.

**RESULT #2 (2026-06-22): general procedure CONFIRMED.** [`spikes/spike_coffee_general.py`](../spikes/spike_coffee_general.py)
replaced the hand-targeted step with a **uniform** rule over every variable — reachability via `do(v)` effects →
direct edge iff target depends on intervened var **given the target's discovered ancestors** (never its descendants
→ no collider) → regime-stratified. **Directed SHD 0, robust 20/20 across 5 seeds × 4 noise levels.** Key lesson:
*the conditioning set is the whole game* ("condition on everything" → collider over-connection, SHD 6). Folded into
hld §4b. **Build-task-1 (sharp now):** harden this rule into the reference discoverer + a vetted GIES-family lib +
a **world-diversity sweep** (the one boundary the spike did not cross: other DAGs/sizes/confounding).

**SCOPE CORRECTION — what §0 actually validated (and what it did NOT).** Spikes #1–#2 **hand-authored** the SCM
(it's written directly in the spike code, not produced by an LLM from prose). They validate the **test harness**:
given a good anti-cliché SCM, it defeats priors (#1) and a general interventional discoverer recovers it (#2). They
do **NOT** test the **author step (T0)** — an *LLM* producing a valid, anti-cliché, **gate-passing** SCM **from a
prose sentence**. That is the project's actual product and its riskiest assumption, and it is **still unproven**.
The §4c loop's whole reliability hinges on it: if the author can't reliably produce gate-passing worlds, the loop
spins to budget and discards → no worlds ship.

### 0b. Build-task-0 — the AUTHOR spike (BEFORE build-task-1)

Claude authors **N** worlds from **prose** (varied domains, sizes, structures); each runs through T1–T4; **measure
the pass rate first-try and within K re-author iterations**, plus the difficulty spread (prior-only gap). This
validates the *generator* before the pipeline is built around it. Needs a **generic spec→sampler + the generalized
discoverer** (minimal shared infra with build-task-1, but pointed at the author, not the grader). Only once the
author clears a useful pass-rate bar do we invest in build-task-1 (hardening the grader).

**RESULT (2026-06-22): author clears the bar — and an INDEPENDENT judge revised my first read.**
[`spikes/spike_author.py`](../spikes/spike_author.py): authored 4 worlds from one-line prompts (hospital ED,
ride-hailing, factory line, e-commerce) → **4/4 pass T1+T2+T3 first-try, robust 5/5 across seeds.** The author can
produce valid, clean-sampling, recoverable worlds from prose — the §4c loop won't just spin-and-discard.
*Methodology correction:* I first played the prior-only baseline **myself (Claude)** and got a misleadingly *mild*
difficulty gap (≈1) — **same-brain optimism** (the exact circularity #1 warns of). Re-run with an **independent
prior — Google `gemini-3.5-flash`** (latest GA flagship; per Amit) — the anti-cliché worlds show real teeth:
**hospital gap 3, ride-hailing gap 2**, while textbook worlds sit at **0** (factory, e-commerce). A clean
difficulty spectrum, *measured independently*. Lessons: **(a)** the prior-only / faithfulness judge **must be a
different LLM family than the author** — self-grading understated difficulty (1→3); **(b)** structure-SHD is still
blind to *sign-flips*, so difficulty comes from **structural** traps (confounders, non-obvious connectivity) — which
these worlds genuinely have; **(c)** Gemini faithfulness scores e-commerce 1.00 / factory 0.80 / ride 0.70 /
**hospital 0.40** — the low one flags that prose naming a *hidden* mechanism (the flu confounder) makes the
observed-only structure read as incomplete → a real faithfulness-gate design point; **(d)** prototype grader hit
SHD 1 on the 2 larger worlds → reinforces build-task-1's vetted GIES.

## A. The world-spec / answer-key schema  *(HLD §2, §9.1)*

The single IR. To pin down:
- Serialization (JSON/YAML; a versioned schema).
- `Variable`: name, role enum, bounds (soft/hard), unit, cadence/Δt.
- `CausalEdge`: from, to, direction, lag, effect sign/shape, mechanism note.
- `Mechanism` / functional form: how a node is computed from its parents + **noise** (the part no prior tool
  authors from pure NL).
- `Noise` *(first-class — the learnability dial, not a footnote)*: per-node noise family + scale. Too little →
  trivial/unidentifiable; too much → unrecoverable. Tuned until the §C learnability gate passes. **Open:** default
  families, and whether noise scale is authored or auto-fit to hit a target signal-to-noise.
- `Regime`: id, switching condition, per-regime parameter overrides (incl. **sign flips** — the anti-cliché lever).
- Validation: typed, units-consistent, acyclic-where-required.

## B. Authoring (NL → spec)  *(HLD §1.1, §6)*

- Prompt design for the staged plan (topology/skeleton → per-node mechanism), à la DEVS-Gen.
- Decomposition so components are authored in bounded, parallelizable units.
- Output is the schema in §A, not free-form code.
- **Open:** how much the LLM authors closed-form equations vs. small code mechanisms; structured-output strategy.

## B2. Conversational world elicitation  *(later milestone — NOT v0)*

One-shotting a full world from a single sentence is fragile and underspecified. The product experience should be a
**dialogue**: the AI asks targeted **clarification questions** (entities? controllable vs. observed? what drives
what? regimes/seasonality? horizon/cadence? what's the objective?) and only **triggers the generation loop once the
spec is complete** (an explicit "ready to generate" gate the user confirms).
- A **spec-completeness checklist** drives which questions to ask (ask only for missing/ambiguous fields).
- Each answer incrementally fills the §A schema; the user can see/edit the accumulating spec.
- Generation (§0/§B) fires only after the checklist is satisfied + user confirms — never mid-elicitation.
- **Open:** question-selection policy (rule-based over the checklist vs. LLM-planned); how much to *infer-and-confirm*
  vs. *ask*; resumability of a partial spec. *(v0 uses a hand-written or single-prompt spec; this UX comes with the
  describe-a-world amplifier, scope §1a.)*

## C. The honesty / validity layer (calibration-free) — the novel core  *(HLD §4, §4a)*

This is the hard core; the §0 spike de-risks it before the rest is built. Operationalized gates (no longer "TBD"):
- **Structural / dynamic (mechanical):** acyclicity, types/units/bounds, conservation/capacity, no impossible states.
- **Identifiability *by construction*:** emit **interventional data** (the gym is ours) so the authored graph is
  recoverable — interventions break Markov-equivalence. Not "check identifiability of an arbitrary SCM"; *provide
  the data that makes it identifiable*. SCM-path key is correct-by-construction.
- **Non-triviality / learnability gate (measured):** pinned reference discoverer = **PCMCI+ at a fixed config**;
  over **N seeds**, require recovered-SHD to beat a **random-graph null** by margin *m*. Fail → reject & re-author.
  *(Defaults for N, m, the null, and the config: to pin in v0.)*
- **Anti-cliché / non-circularity gate (measured):** a **prior-only discoverer** (LLM, names+prose, no data); admit
  only if `statistical_score − prior_only_score ≥ δ` (the **prior-only gap**). Else perturb (flip a sign, add a
  confounder, shift a lag) and re-author.
- **Noise** (§A): tuned so the learnability gate passes (the dial between trivial and unrecoverable).
- Failure → structured feedback object → **bounded re-authoring loop**; stop when all gates pass or authoring budget
  is spent.

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
- **`do(x)` intervention IS needed in v0** — not for a control benchmark, but to emit the **interventional data** the
  identifiability gate (§C) requires.
- Perturbation API (shift a disturbance, change a parameter).
- *Counterfactual replay + control-policy scoring are **later** (scope §1a — magnitudes must be principled first).*

## G. Evaluation harness  *(HLD §5, §4a)*

- The **pluggable agent interface** (an external causal-discovery agent gets data + the gym; returns a recovered
  structure). v0 scores **structure only** (SHD, F1) — magnitudes don't bite here (scope §1a).
- **Validity anchor = a *statistical* discoverer** (PCMCI+/PC/NOTEARS), never an LLM — no shared-brain circularity.
- **Prior-only baseline** reported alongside every result; the **prior-only gap** is a first-class benchmark-quality
  metric (a small gap = a weak/cliché world).
- An LLM-reasoning discoverer may be scored as a **separate track**, clearly labelled, never the validity anchor.
- *Interventional / counterfactual accuracy + control scoring: later (needs principled magnitudes).*
- Report format.

## H. Packaging

- Python package layout, public API surface, examples, a couple of seed worlds (a supply-chain world; a
  continuous/energy-style world), docs.

---

### Build order (revised post-review — validity & spike first, SCM-only v0)
0. **§0 spike** — kill/confirm "LLM authors an anti-cliché, identifiable SCM from a sentence." **Gates everything.**
1. **A** (schema, incl. noise) — everything depends on it.
2. **D-SCM** + **F** (run + interventional data) + **G** (statistical discoverer + prior-only baseline) on a
   **hand-written** spec — prove the *executable + correct-by-construction key + valid scoring* spine, **and the
   §C validity gates**, before any LLM authoring. Validity is built *with* the spine, not bolted on later.
2.5 **C** (honesty/validity gates) wired into the loop — non-triviality + anti-cliché + identifiability.
3. **B** (NL→spec authoring) — close the loop from prose, now that the gates can catch bad authoring.
4. *Later milestones (not v0):* **B2** (conversational elicitation), **E** (temporal/regime), the **control/
   counterfactual** scoring, and the **discrete-event** substrate.
