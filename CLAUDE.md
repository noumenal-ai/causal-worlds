# causal-worlds — AI working agreement

Short, binding agreement for working in this repo. Keep it short; depth lives in
[docs/engineering.md](docs/engineering.md), auto-applied via the skill
[.claude/skills/causal-worlds-conventions/](.claude/skills/causal-worlds-conventions/SKILL.md).

## What this is
A public (MIT) Python package: **generate a fictional-but-coherent causal *operation* from a natural-language
description** — an executable simulator, the time-series it emits, and a **declared ground-truth causal structure
(the answer-key)** — for benchmarking causal-discovery agents. A **mix of engineering and research**. CLI-first
(typer). Consumes **Gemini** as an *independent* LLM judge (must differ from any author model family). Concept &
approach are **validated** (see [docs/validation.md](docs/validation.md)); this is the production build.

## Non-negotiables (full detail in docs/engineering.md)
- **Clean Code (Uncle Bob) — all of it, NOT Clean Architecture.** **SOLID** via Python `Protocol`s.
- **Design patterns only at proven variation points** (Strategy/Adapter for discoverer·judge·substrate, Pipeline of
  gates). **No abstraction for hypothetical futures.** Reuse over fork.
- **Wrap every third-party lib** (`causal-learn`, `gies`, Gemini) **behind our own Protocol + adapter.**
- **Tooling:** `uv` · `ruff` (`select=ALL` + curated ignores, line 100, `ruff format`) · `mypy strict` · `pytest`
  with a coverage floor · pre-commit · **CI that fails**. `src`-layout, feature/capability modules. Google docstrings.
- **Run the gate before committing:** `make validate` (or `uv run ruff format --check . && uv run ruff check . &&
  uv run mypy && uv run pytest`). **CI green is a merge gate** — that's how we avoid re-leaving the same review comment.
- **Measured, not asserted.** Every behavioral claim is backed by a runnable script/test.
- **`spikes/` and `experiments/` are research, NOT shipped** (lint/type/coverage-exempt); reproducible via seed +
  `uv.lock` + pinned model ids; honest negatives.
- **Commits:** Conventional Commits, atomic, **no `Co-Authored-By` trailer**. Push/PR only on explicit request.

## Map
- [docs/scope.md](docs/scope.md) · [docs/hld.md](docs/hld.md) · [docs/lld.md](docs/lld.md) ·
  [docs/validation.md](docs/validation.md) — product/design + the validation evidence.
- [docs/engineering.md](docs/engineering.md) — the binding code-quality + research guidelines.
- `spikes/` — the validation spikes (research; the proof, not the implementation).
