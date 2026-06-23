# Changelog

All notable changes to causal-worlds are documented here. Format: [Keep a Changelog](https://keepachangelog.com/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.9.0] — 2026-06-23

**The temporal generative loop closes** — natural language → an admitted *temporal* world.

### Added
- **Temporal gating** (`gates`): `run_gates` detects a temporal world (any lagged term) and admits it
  via a temporal T3 — a TS reference (PCMCI+ by default, injectable) must recover the lagged structure
  above an F1 floor. `GateReport.temporal_grade` carries the lagged grade. T1/T2 work unchanged.
- **Author temporal mode**: `build_claude_author(temporal=True)` / `ClaudeAuthor(temporal=...)` adds a
  brief instructing lagged + autoregressive (stationary) terms, so the LLM can author temporal worlds.
- `generate` threads a `temporal_discoverer` through to the gate; `FakeTemporalDiscoverer` for keyless
  tests.

### Validated (live)
- Claude (temporal mode) authored a reservoir operation — rainfall→inflow at lag 1, autoregressive
  storage/inflow, a hidden `soil_saturation` confounder, a snowmelt regime — **admitted in 1 attempt**
  through the PCMCI+ temporal gate (F1 0.56). Next: a scaled temporal benchmark set + crossover at n>1.

[0.9.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.9.0

## [0.8.1] — 2026-06-23

### Fixed
- **Temporal worlds now round-trip through persistence.** The pydantic boundary `TermModel` carried
  no `lag`, so serializing a temporal spec (`spec_to_json` / `save_bundle`) silently dropped the lags.
  `TermModel.lag` is added (and threaded through `to_term`/`from_spec`), so lagged worlds persist
  faithfully — and the LLM author can now express lagged edges.

[0.8.1]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.8.1

## [0.8.0] — 2026-06-23

**Temporal grading.** Time-series discovery can now be benchmarked against the lagged ground truth.

### Added
- **Temporal scoring** (`evaluation`): `temporal_score` → `TemporalReport` (temporal SHD + F1 over
  `(src, dst, lag)` edges, with lag-0 reversals handled).
- **`TemporalDiscoverer`** protocol + **`grade_temporal_spec`** — grade any TS method against a
  world's `temporal_answer_key`.
- **TS baselines** (`temporal_baselines`, the `temporal` extra): PCMCI+ / LPCMCI (tigramite),
  VARLiNGAM (lingam), Granger (statsmodels) — behind adapters, lazy-imported, with pure tested parsers.
- **Temporal crossover eval** (`evals/temporal-crossover`) on the `supply` world.

### Findings (n=1, honest)
- The grading stack is validated: **PCMCI+ recovers `supply`'s lagged structure exactly** (temporal
  SHD 0, F1 1.0). The hidden-confounder trap is **method-specific** temporally — only VARLiNGAM keeps
  the spurious `leadtime~cost` edge; PCMCI+/LPCMCI/Granger don't. Unlike the cross-sectional crossover,
  the TS toolbox does not *uniformly* fail on this single world — a temporal benchmark *set* is needed
  to characterize it (next).

[0.8.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.8.0

## [0.7.0] — 2026-06-23

**Temporal worlds (foundation).** Worlds can now carry *time* — lagged edges and autoregression —
not just cross-sectional structure. (Time-series *grading* + baselines land next.)

### Added
- **Lagged IR** (`schema`): `Term.lag` (default 0). Only the contemporaneous (lag-0) subgraph must be
  acyclic; lagged edges — including autoregressive self-loops — are valid (they read the past).
- **Temporal substrate** (`sample`): when any lag is present, sampling becomes sequential over
  timesteps with a burn-in (near-stationary); `do()` interventions hold across time. Cross-sectional
  worlds keep the original vectorized i.i.d. path unchanged.
- **`temporal_answer_key(spec)`** → lagged ground truth `(src, dst, lag)` incl. autoregression; the
  summary `answer_key` now collapses lags and drops self-loops, so existing tooling is unaffected.
- **Built-in `supply`** — a temporal world (autoregressive lead time + inventory, a hidden logistics
  confounder), in a separate registry (`worlds.temporal_names()`) so the still-contemporaneous CLI
  `grade`/`gate` don't mis-score it.

[0.7.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.7.0

## [0.6.1] — 2026-06-23

### Docs
- Rewrote the README around a getting-started flow (honest shipped-vs-roadmap; the gym/temporal/
  counterfactual claims are now roadmap, not overclaims), with a lead example, the measured crossover
  result, install/extras, concepts, and a roadmap.
- Added a guided [`docs/getting-started.md`](docs/getting-started.md) and runnable
  [`examples/`](examples/) (grade-your-discoverer, inspect-a-bundle — keyless — and author-a-world).

[0.6.1]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.6.1

## [0.6.0] — 2026-06-23

**Use the benchmark.** Grading your own discoverer against a shipped world is now a first-class,
typed, tested feature — the package's whole purpose.

### Added
- **`bench`**: `grade_spec(spec, discoverer)` and `grade_bundle(bundle_dir, discoverer)` → a `Report`
  scoring any `Discoverer` against a world's declared answer-key (defaults to the reference grader).
- **CLI `score`**: `causal-worlds score <bundle> [--discoverer module:Class]` grades a discoverer
  (the reference by default, or any importable one) on a persisted world.
- **Typed distribution**: ship a PEP 561 `py.typed` marker, plus PyPI metadata (classifiers,
  keywords, project URLs).

[0.6.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.6.0

## [0.5.0] — 2026-06-23

**Scale resolves the difficulty question.** A 36-world set across an easy→hard complexity spread gives
the analyses real range — and structural difficulty turns out to predict the observational collapse.

### Added
- **Author complexity knob** (`author`): `ClaudeAuthor(..., complexity="easy"|"standard"|"hard")`
  varies how many hidden confounders / regime sign-flips to inject, spreading structural difficulty.
  Recorded per world in the manifest (`Provenance.complexity`).
- **Scaled benchmark** (`benchmark/v0.5`): 35/36 admitted across complexity levels — mean structural
  difficulty by level 0.0 / 1.4 / 3.0; reference-grader SHD 0.36 / 1.75 / 2.33.
- **Parameterized evals**: the crossover and structural-difficulty harnesses take a benchmark dir;
  results nest under `evals/*/v0.5/`.

### Findings (powered, n=35)
- **Crossover strengthens**: the interventional-CI grader keeps **confounded-kept = 0** (never reports
  a hidden-confounded pair as causal) at SHD 1.47 / F1 0.91, while PC/FCI/GIES keep 8–17 and post SHD
  2.7–6.7.
- **Structural difficulty predicts observational error (corr +0.62)** where name-guessability does not
  (+0.14) — the hardness is structural (confounders + sign-flips), resolving v0.4's open question and
  turning difficulty into a usable instrument.

[0.5.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.5.0

## [0.4.0] — 2026-06-23

**A structural-difficulty axis.** v0.3 showed name-guessability difficulty doesn't predict discovery
error — the hardness is structural. This adds that axis and tests it honestly.

### Added
- **Structural difficulty** (`difficulty`): `structural_difficulty(spec)` scores discovery-hardness
  from the structure — hidden confounders, confounded pairs, regime **sign-flips**, edge density — with
  a headline trap-count `score`. Pure, deterministic, unit-tested.
- Structural difficulty is now recorded in every admitted world's `manifest.json`.
- **Re-analysis** (`evals/structural-difficulty`): reuses the crossover report (no new runs) to test
  whether structural difficulty predicts the collapse.

### Findings (honest)
- At n=12 with a narrow difficulty range, **neither** name-guessability nor structural difficulty
  cleanly predicts the *magnitude* of error (correlations −0.39…+0.14) — a statistical-power problem,
  not a refutation. The v0.3 crossover (standard methods collapse, grader holds) is unaffected.
  Resolving difficulty-predicts-error is deferred to the scaled set (v0.5).

[0.4.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.4.0

## [0.3.0] — 2026-06-23

**The decisive experiment.** Proves the benchmark's central claim beyond the single `coffee` world:
standard discovery collapses on our worlds where the reference interventional-CI grader holds.

### Added
- **Baseline suite** (`baselines`): PC, GES, FCI (`causal-learn`) and GIES (`gies`) wrapped behind the
  `Discoverer` Protocol as adapters — lazy-imported (the `discover` extra), so the package imports and
  CI run without them; graph-parsing logic is pure and unit-tested. `BaselineResult` carries directed
  edges, bidirected (confounding) marks, and the skeleton for a fair cross-method comparison.
- **Crossover eval** (`evals/baseline-crossover`): every benchmark world vs every method across seeds →
  skeleton-SHD, directed F1, and *confounded-pair-kept-as-causal* (the trap). **Result (n=12): GO.**
  Standard methods keep the hidden-confounded pair as causal in 7.3–10.0 of 12 worlds (PC/FCI/GIES) and
  post 2–4× the skeleton error; the interventional grader stays at confounded-kept 0.33, SHD 1.31,
  F1 0.91.
- **Difficulty-vs-error analysis** — *honest negative*: name-guessability difficulty does not yet
  predict discovery error (corr ~0.1); the hardness is structural (confounder+regime). Sharpens v0.4.
- **Publication artifacts**: a technical blog post (`docs/blog-the-decisive-experiment.md`) and a
  Framing-B paper skeleton (`paper/`).

### Notes
- `causal-learn`'s GES is numpy-2 incompatible (errors on every world) — reported, not hidden.

[0.3.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.3.0

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
