# Architecture — causal-worlds

> The finalized system design. Builds on [scope.md](scope.md), [hld.md](hld.md), [lld.md](lld.md),
> [validation.md](validation.md), and the binding [engineering.md](engineering.md). This is the **how it's
> structured**; engineering.md is the **how it's written**.

## 1. Shape — a pipeline with a functional core and an imperative shell

```
 prose ─▶ Elicit ──▶ WorldBrief ─▶ Author ─▶ WorldSpec ─▶ [ Gate pipeline ]
        (dialogue,    (intent)      (LLM →     (the IR /     T1 static · T2 sample-sanity ·
         optional)                  pydantic   answer-key)   T3 non-trivial · T4 difficulty
                                    → spec)         │           │ fail → bounded re-author loop
                                                    ▼           ▼ all pass
                                                  admit ──▶ AdmittedWorld (spec + provenance, frozen)
                                                    │
                                                    ▼
                                       Sample/Compile (spec + seed ─▶ Substrate ─▶ Sample/Dataset)
                                                    ▼
                                          Grade (Discoverer recovers structure from interventions)
                                                    ▼
                                          Eval (vs answer-key: SHD/F1 + difficulty) ─▶ Report
```

- **Functional core (pure, deterministic, `numpy` + frozen dataclasses, no IO):** `schema`+`validate`, the **SCM
  sampler** (`spec + seed → Sample`), graph ops, **scoring** (SHD/F1), and **gate logic** (`data → pass/fail/
  feedback`). Seeds are passed in explicitly; the core never does IO, LLM calls, or logging — it returns values or
  raises typed errors.
- **Imperative shell (side effects):** `Elicit`, `Author`, `Judge` (LLM calls), artifact IO, the orchestration
  loop, tracing/logging, the CLI. The shell builds the object graph, threads config + seeds, logs, and traces.

## 2. The variation seams (Protocols + adapters; DIP)

| Seam | Layer | Responsibility | Concrete (v1) |
|---|---|---|---|
| `Elicitor` | shell | drive the clarify loop → a complete `WorldBrief` | LLM (author model) |
| `Author` | shell | `WorldBrief → WorldSpec` (functional forms + noise) | LLM + **instructor** |
| `Judge` | shell | `prior_edges`, `faithfulness` — the **independent** grader | **Gemini** (instructor) |
| `Substrate` | core | `variables`; `sample(n, *, seed, do) → Sample` | SCM sampler |
| `Discoverer` | core+adapter | `recover(substrate, *, seed) → Edges` (interventional-CI) | do()-CI / FCI-with-interventions; `causal-learn`/`gies` behind adapters |
| `Gate` | core logic | `check(...) → GateResult` (pass/fail + structured feedback) | T1–T4 |

Third-party (`causal-learn`, `gies`, Gemini, Langfuse) appears **only inside an adapter** that satisfies a Protocol.

## 3. The Substrate↔Discoverer data contract
- **`Sample`** (frozen dataclass): `variables: tuple[str, ...]`, `data: FloatArray` (n × p), `intervened:
  frozenset[str]`.
- **`Substrate.sample(n, *, seed: int, do: Mapping[str, float] | None = None) -> Sample`** — one environment;
  `do` fixes the named variables; `intervened` records which.
- **`Discoverer.recover(substrate, *, seed) -> Edges`** — the discoverer **drives its own observational +
  interventional sampling** (interventional discovery must *choose* what to intervene on — the validated point in
  [validation.md](validation.md) §4d). It depends on the `Substrate` Protocol, never a concrete gym.
- **`numpy` is a core dependency** (the sampler + scoring are the functional core); `causal-learn`/`gies` remain the
  `discover` extra (grader adapters only). *(Move numpy from the `discover` extra to core deps when `sample/`
  lands.)*

## 4. Elicitation — brief vs spec, optional, stateful
- **Optional first stage** before `Author`. Batch benchmark generation / API one-shots pass a complete brief or
  spec and skip it; interactive use runs it.
- **Two artifacts, two concerns:** `Elicit` (dialogue) → **`WorldBrief`** = structured *intent* (entities, roles,
  declared relationships, **regimes**, **hidden confounders**, objective). `Author` compiles the brief → the
  executable **`WorldSpec`** (functional forms + noise). Brief = human-facing; spec = machine.
- **Driven by a spec-completeness checklist:** ask the *minimal* follow-ups for missing/ambiguous fields, show the
  accumulating brief, hand off only when the checklist passes (or the user says "go"). Bounded turns.
- **`Session`** state model (the elicitor is stateful across turns) — a first-class type the CLI and a future web UI
  both drive.
- Model: elicitation uses the **author** model (it helps the user *specify*, not grade — the independence rule
  applies only to `Judge`).
- **Build order:** designed now, implemented *after* the core author→gate→grade spine (benchmark runs on complete
  briefs first; the conversational UX layers on).

## 5. Orchestration & DI — a small container
- A **small DI container** wires concrete impls → Protocols + config. *Not* a CLI composition root, because the CLI
  is not the only interface (library API now, web later) — a root-in-CLI would force duplication.
- Construction is separated from use (Clean Code §Systems): interfaces (`cli`, library `api`, web) **resolve** from
  the container; the core never news-up collaborators.
- Keep it tiny and explicit (a typed registry / factory functions) — no heavyweight framework.

## 6. Config — pydantic-settings → frozen core config
- **pydantic-settings (`BaseSettings`)** loads + validates from env, `.env`, and overrides at the boundary, then is
  **converted to a frozen core-config dataclass** the functional core consumes (parse-don't-validate).
- **Configurable:** secrets (env only — `GEMINI_API_KEY`, Langfuse keys), model ids + temperatures, run `seed`, the
  **calibration knobs** (non-triviality margin, anti-cliché gap δ, sample size n, noise, max re-author iters K,
  patience j), per-world **budgets** (max LLM calls / tokens), observability on/off, output dir.
- **Not configurable** (decided constants): the pipeline structure, the gate ordering, the schema.

## 7. The artifact — the admitted-world bundle
A directory per admitted world:
```
<world_id>/
  spec.json      # the WorldSpec = the ground-truth ANSWER-KEY (dataclass → JSON, human-readable)
  data.npz       # the emitted observational + interventional samples
  manifest.json  # provenance: seed · generator version · grader version · model ids · difficulty (prior-only gap)
                 #             · per-gate results · created_at
```
Provenance lives in `manifest.json`; **the grader version is part of the artifact** (admission is grader-relative —
validation finding). Reproducible from `spec + seed + uv.lock + model ids`.

## 8. Errors, observability, logging
Per [engineering.md §11](engineering.md): root `CausalWorldsError` hierarchy (fail loud, never fabricate);
**Langfuse (OTEL) tracing** behind a thin seam in `obs/` (optional at runtime); library logging via
`getLogger("causal_worlds")` + `NullHandler` (the shell/CLI owns handlers); the pure core stays silent. Logs ·
traces · exceptions are three distinct channels.

## 9. Modules (`src/causal_worlds/`)
```
schema.py      # the WorldSpec IR + static validation (the T1 gate)            [done]
protocols.py   # the Protocol seams (Elicitor, Author, Judge, Substrate, Discoverer, Gate)
errors.py      # CausalWorldsError hierarchy
config.py      # pydantic-settings loader → frozen core config
container.py   # the small DI container (composition root for all interfaces)
sample/        # Substrate (SCM sampler) + Sample dataclass        [functional core; numpy]
gates/         # T1..T4 gate logic + the gate pipeline             [core logic + shell loop]
discover/      # interventional-CI grader + causal-learn/gies adapters
eval/          # scoring vs the answer-key (SHD/F1, difficulty) + Report
author/        # WorldBrief → WorldSpec (LLM + instructor)          [shell]
elicit/        # dialogue → WorldBrief + Session                   [shell; later]
worlds/        # the catalog / built-ins
obs/           # the tracing seam (Langfuse/OTEL)                   [shell]
cli.py         # typer CLI — resolves from the container
```
Feature/capability modules; third-party imports only inside a feature's adapter.

## 10. Build order (spine first, LLM stages last — the validation lesson)
1. `schema` ✅ → 2. `sample/` (SCM Substrate + `Sample`; numpy → core) + `eval/` scoring — the **deterministic
core**. 3. `gates/` logic + the gate pipeline. 4. `discover/` (interventional-CI grader + adapters) — prove the
spine on **hand-written specs** end-to-end (mirrors the spikes). 5. `author/` (LLM + instructor). 6. `elicit/` +
`Session`. 7. `config`/`container`/`obs`/`cli` polish throughout. Every stage ships to engineering.md standards with
the gate green; the proven spikes *graduate* into these modules rebuilt to standard.
