---
name: causal-worlds-conventions
description: >-
  Engineering + research conventions for the causal-worlds Python package. Load and apply BEFORE writing,
  refactoring, or reviewing ANY code in this repo (src/, tests/, cli, the package), and before adding a dependency
  or a design pattern. Encodes Clean Code (Uncle Bob), SOLID-via-Protocols, the earned-patterns rule, the uv/ruff/
  mypy/pytest/CI toolchain, src-layout, and the research discipline. Full detail in docs/engineering.md.
---

# causal-worlds conventions

Apply these when touching this package. Full reference: [docs/engineering.md](../../../docs/engineering.md).
Mix of engineering + research; CLI-first (typer); Gemini is an *independent* judge (≠ author model family).

## Before you write code
- **Clean Code (all of it), NOT Clean Architecture.** Small functions that do one thing at one level of
  abstraction; intention-revealing names; **≤2 args** (bundle into a value object), **no flag args**, **no hidden
  side effects**, **Command-Query Separation**. **Exceptions, not error codes; never return/pass `null`/`None`-as-
  error.** No commented-out code (delete it). No magic numbers (named constants). Respect the Law of Demeter.
- **SOLID via `typing.Protocol`.** The four variation points get Protocols + injected impls:
  `Discoverer` (grader), `Judge` (LLM), `Substrate`/`World`, `Gate`. Depend on the Protocol; **never** let
  `causal-learn`/`gies`/Gemini types leak past an **adapter**.
- **Patterns are earned** (Strategy/Adapter for discoverer·judge·substrate; Pipeline for gates). **No speculative
  abstraction.** If you add a pattern, name it and justify it in the PR.
- **Separate construction from use:** build/inject dependencies at the edge (`cli`, factories); the core never
  news-up collaborators.
- **Structure:** `src/causal_worlds/<feature>/`; tests mirror; third-party imports only inside a feature's adapter.
- **Docstrings:** Google style. **Type everything** (mypy strict).

## Before you commit — run the gate (it must be green)
```bash
make validate    # or:
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest
```
CI runs the same and **fails** on any violation; CI-green is the merge gate. Conventional Commits, atomic, **no
`Co-Authored-By` trailer**. Push/PR only when asked.

## Tests (F.I.R.S.T.)
Fast · Independent · Repeatable · Self-validating · Timely. One concept per test. Prefer **Hypothesis property
tests** for invariants (acyclicity, interventions break the right edges, seed→determinism) over fixed-output tests.

## Research code (`spikes/`, `experiments/`)
NOT shipped; lint/type/coverage-exempt. Held to "**is the finding real and honestly reported**," not production
polish. **Measured, not asserted:** every claim has a runnable script that prints the evidence. Reproducible via
**seed + `uv.lock` + pinned model ids** (e.g. `gemini-3.5-flash`). Report honest negatives. Use an **independent
judge** for LLM-output quality (don't grade a model with itself). A proven spike **graduates** into `src/` rebuilt
to the standards above — the spike is the proof, not the implementation.

## Adding a dependency
Justify it; pin via `uv`; wrap it behind a Protocol+adapter; prefer the standard library and reuse over new deps.
