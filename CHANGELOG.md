# Changelog

All notable changes to causal-worlds are documented here. Format: [Keep a Changelog](https://keepachangelog.com/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-22

First release: **the deterministic benchmark engine**. Generate (programmatically-specified) fictional causal
worlds with a ground-truth answer-key, sample them, grade a causal-discovery method, and score it — runnable as a
library and a CLI, with no LLM or API key required.

### Added
- **Schema / IR** (`schema`): `WorldSpec` as the single source of truth (variables incl. hidden confounders +
  generative `Mechanism`s with regime-switching); the `AnswerKey` (observed edges + confounded pairs) is *derived*,
  never stored; `validate()` static gate.
- **SCM substrate** (`sample`): a deterministic, seeded executable world; `do()` interventions (constant or
  per-row array). The functional core.
- **Reference grader** (`discover`): `InterventionalCiDiscoverer` — a spec-blind interventional-CI discoverer that
  recovers the confounder + regime-flip trap (directed SHD 0) where standard observational/score-based methods
  (PC, GES, GIES, FCI) fail.
- **Scoring** (`evaluation`): directed/skeleton SHD, F1, and `confounded_reported` (flags a causal edge claimed for
  a hidden-confounded pair); `Report`.
- **Validity gates** (`gates`): `run_gates` → T1 (validity) · T2 (sample-sanity) · T3 (non-triviality vs a
  per-world random-graph null). Admits only if all pass.
- **Built-in worlds** (`worlds`): `coffee` (the confounder + regime-flip trap) and `ecommerce` (easy control).
- **CLI** (`causal-worlds`): `version` · `worlds` · `grade <world>` · `gate <world>`.
- **Wiring**: pydantic-settings `config`, a small DI `container`, and a no-op `Tracer` observability seam.
- **Quality**: uv + ruff (`select=ALL`) + mypy `strict` + pytest with a coverage floor, enforced by CI.

### Not yet (tracked as v0.2 issues)
NL/`WorldBrief` → spec **author**, the independent **Gemini judge** + the T4 anti-cliché gate, conversational
**elicitation**, the **Langfuse (OTEL)** tracing adapter, **artifact/manifest persistence**, grader **hardening**
(FCI-with-interventions) + world-diversity sweep + knob calibration, and more built-in/temporal worlds.

[0.1.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.1.0
