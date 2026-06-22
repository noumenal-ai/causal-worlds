# Changelog

All notable changes to causal-worlds are documented here. Format: [Keep a Changelog](https://keepachangelog.com/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-23

Closes the generative loop: **natural language in, an admitted causal world out**, plus persistence
and a shipped benchmark set. The LLM seams are real but isolated — the package still imports and CI
still runs with no API key (the adapters are unit-tested against fakes).

### Added
- **NL author** (`author`): `ClaudeAuthor` turns a plain-language operation into a `WorldSpec` via
  `instructor` (bounded re-ask), steered toward recoverable, anti-cliché worlds (a hidden confounder
  + a regime sign-flip). Behind the `Author` Protocol; provider SDK lazy-imported.
- **Independent judge** (`judge`): `GeminiJudge` guesses the structure from names/roles alone (the
  anti-cliché signal) and scores faithfulness — a *different model family* than the author.
- **T4 anti-cliché gate** (`gates`): with a judge + prose, rejects unfaithful or guess-from-priors
  worlds and records a `difficulty` score (`1 - F1(judge_prior, truth)`).
- **The loop** (`generate`): `generate` (author→gate→admit with feedback-driven re-author) and
  `generate_many` (never-raising batch) → `AdmittedWorld`.
- **Artifact persistence** (`artifact`): self-describing on-disk bundle (`spec.json` / `data.npz` /
  `answer_key.json` / `manifest.json`) with full provenance (models, grader version, seed, grade).
- **Boundary model** (`serde`): one pydantic `WorldSpecModel` — the author's output target and the
  persisted JSON shape — converting to/from the frozen core IR.
- **CLI**: `generate <prompt> <out>` and `benchmark <prompts_file> <out>`; author/judge resolved
  through the DI container.
- **Author-model bake-off** (`evals/author-model-bakeoff`): a reproducible, judged comparison that
  picks the default author model with numbers, not assertion — shipped with the release.
- **Benchmark set** (`benchmark/v0.2`): 12 authored, admitted worlds across distinct operations —
  mean difficulty 0.28, faithfulness 1.00, reference-grader directed SHD 1.25 / F1 0.92.

### Changed
- Version is single-sourced from `_version.py` (hatchling dynamic). `.coverage` is no longer tracked.

[0.2.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.2.0

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
