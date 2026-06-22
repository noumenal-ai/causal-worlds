# Engineering & research guidelines — causal-worlds

> How we build this package. Binding for all production code (`src/`, `tests/`). Research code (`spikes/`,
> `experiments/`) follows the lighter §9 research discipline. The short version lives in `CLAUDE.md` and the
> auto-applied skill `.claude/skills/causal-worlds-conventions/`; this is the full reference.

## 1. Philosophy
- **Clean Code (Robert C. Martin), all of it — but NOT Clean Architecture.** Principles below (§2), not the
  hexagonal/ports-adapters ceremony.
- **SOLID**, expressed in Python idioms (§3).
- **Design patterns are deliberate and earned** ([refactoring.guru](https://refactoring.guru/design-patterns)) —
  applied *only at variation points we have already seen vary* (the grader, the LLM judge, the substrate, the
  gates). **No abstraction for a hypothetical future** (inherited Noumenal non-negotiable). Reuse over fork.
- **Observability + evaluation baked in from day 1** (inherited).
- **Measured, not asserted** — claims about behavior are backed by a runnable script/test, never prose (§9).

## 2. Clean Code (the faithful set)
*Names* — intention-revealing; no disinformation; meaningful distinctions (no `a1`/`data2`); pronounceable;
searchable (no bare magic numbers → named constants); no type/Hungarian encodings; class = noun, method = verb;
one word per concept (don't pun); prefer solution- then problem-domain names; add meaningful context, no gratuitous
context.

*Functions* — **small**, then smaller; **do one thing** (one level of abstraction per function); descriptive name;
**≤ 2 args ideal** (0–1 best, 3 needs justification, avoid more) — bundle into a value object instead; **no flag
arguments** (split the function); **no side effects** (a function does what its name says and nothing hidden;
don't mutate inputs); **Command-Query Separation** (a function either *does* or *answers*, never both); prefer
**exceptions to error codes**; extract `try/except` bodies into functions; **DRY** (one authoritative
representation of each piece of knowledge).

*Comments* — explain yourself *in code*; the best comment is the one you didn't need. Good: legal, intent,
clarification, warning, `# TODO(owner): …`, public-API docstrings. Bad: redundant, noise, position banners,
attributions, and **never commented-out code** (delete it; git remembers).

*Formatting* — newspaper metaphor (high-level at top); vertical density for related lines, distance for separate
concepts; the formatter (ruff) owns horizontal style — don't argue with it.

*Objects vs data structures* — hide implementation behind small interfaces; respect the **Law of Demeter** (don't
reach through objects: no `a.getB().getC().doD()`); know the data/object anti-symmetry (don't build hybrids).

*Error handling* — exceptions over codes; **define exception types for the caller's needs**; provide context in the
message; **don't return null and don't pass null** (use a value, raise, or a Special-Case/Null-Object); error
handling is one thing — a function that handles errors does *only* that.

*Boundaries* — **wrap third-party APIs behind our own interface** (this is load-bearing here: `causal-learn`,
`gies`, Gemini all sit behind our `Protocol`s/adapters, so churn and assumption-mismatches are isolated — see §3);
write **learning tests** against external libs; depend on interfaces we control.

*Tests* — see §7.

*Classes* — small; **SRP** (one reason to change); high cohesion; organized for change (OCP); depend on
abstractions (DIP).

*Systems* — **separate construction from use** (build the object graph at the edge — `cli`/`main`/factories —
inject dependencies; the core never news-up its collaborators).

*Simple design* (Kent Beck, in priority order) — (1) passes all tests, (2) no duplication, (3) expresses intent,
(4) fewest elements.

*Smells* — Uncle Bob's list is the review vocabulary; the recurring ones we enforce: dead code, duplication,
long/argument-heavy functions, feature envy / Demeter violations, magic numbers, vague names, untested paths.

## 3. SOLID in Python — Protocols at the proven variation points
- **DIP/ISP/OCP via `typing.Protocol`** (structural, small interfaces) at the four boundaries the spikes proved
  vary:
  - `Discoverer` — the grader (prototype → interventional-CI → FCI-with-interventions). *Not* a stock library call.
  - `Judge` — the independent LLM (Gemini today; must differ from the author's family).
  - `Substrate` / `World` — SCM-sampling now; DEVS/other later.
  - `Gate` — one check in the author→gate→admit pipeline; gates compose.
- Depend on the `Protocol`, inject the concrete impl (constructor injection). The concrete `causal-learn`/`gies`/
  Gemini code lives in **adapters** that satisfy the Protocol — *that* is the only place third-party types appear.
- **SRP** at the module level: one capability per feature module (§5).

## 4. Design patterns (earned, cited)
Use, and *name in the code/PR*, only these — each maps to a real variation point:
- **Strategy** — `Discoverer`, `Judge`, `Substrate` are interchangeable strategies behind a Protocol.
- **Adapter** — wrap `causal-learn`, `gies`, Gemini (§2 Boundaries).
- **Pipeline / Chain** — the `Gate` sequence (cheap→expensive, short-circuit) in the author→gate→admit loop.
- **Factory** — build a `World`/`Substrate` from a frozen spec at the edge.
Anything else needs a one-line justification in review. Speculative patterns are a smell here.

## 5. Project structure (`src`-layout, feature/capability modules)
```
src/causal_worlds/
  schema.py      # the world-spec / answer-key IR (+ static validation = the T1 gate)
  protocols.py   # the Protocol contracts (Discoverer, Judge, Substrate, Gate)
  author/        # NL → world spec (LLM)            [later]
  sample/        # spec → executable substrate + data [later]
  gates/         # the validity gates (T1..T4)        [later]
  discover/      # the interventional-CI grader + adapters [later]
  eval/          # scoring vs the answer-key (SHD/F1)  [later]
  worlds/        # the catalog / built-ins            [later]
  cli.py         # typer CLI (construction-from-use edge)
tests/           # mirrors src/ ; F.I.R.S.T.
spikes/          # research spikes (NOT shipped; lint/type-exempt)
experiments/     # graduated spikes / sweeps (NOT shipped)
docs/            # scope, hld, lld, validation, engineering (this file)
```
Feature/capability grouping (not layers). Third-party imports live only in the relevant feature's `adapters`.

## 6. Tooling & enforcement (the "CI fails so we don't re-comment" goal)
- **uv** — env, deps, lockfile (`uv sync`, `uv run …`, `uv lock`). Python **3.13** (3.14 once sci wheels land).
- **ruff** — single linter + formatter. `select = ["ALL"]` + a curated `ignore` list (opt out deliberately),
  line-length 100, Google docstrings. `ruff format` is law.
- **mypy `strict = true`** — full typing from day 1.
- **pytest + coverage** with a **`--cov-fail-under` floor** — below it, the build is red.
- **pre-commit** — ruff (+ format) + hygiene hooks, run locally.
- **CI (GitHub Actions)** — runs `ruff format --check`, `ruff check`, `mypy`, `pytest --cov` and **fails** on any
  violation. Required to merge. This is the enforcement that replaces repeating the same PR comment.
- `make check` / `make validate` (via uv) run the whole gate locally; **run it before every commit.**
- `.claude/` ships **hooks** (ruff-format on edit; block `.env`/lock edits) + the conventions skill, so the
  standards apply while editing, not just in CI.

## 7. Testing (F.I.R.S.T.)
Fast · Independent · Repeatable · Self-validating · Timely. One concept per test; clear arrange/act/assert; tests
are first-class code (held to the same names/structure bar). Prefer **property-based tests** (Hypothesis) for the
SCM/sampler/graph **invariants** (acyclicity preserved, interventions break the right dependencies, seed →
determinism) — research code has many invariants and few fixed outputs. TDD where it fits.

## 8. Git & PRs
- **Conventional Commits**, atomic. **No `Co-Authored-By` trailer** (repo convention).
- Push/PR only on request. CI green is a merge gate. PRs are small and single-purpose.

## 9. Research discipline (this is half a research package)
- **`spikes/` and `experiments/` are NOT shipped** (excluded from the wheel, from ruff, from mypy, from coverage).
  They are exploratory — held to "is the finding real and honestly reported," not to production polish.
- **Reproducibility:** every result is reproducible from a **seed + `uv.lock` + pinned model ids** (e.g.
  `gemini-3.5-flash`). Record the numbers in the script's output and the doc.
- **Every claim is backed by a runnable script** that prints the evidence; **measured, not asserted.**
- **Honest negatives:** a spike that fails or surprises is reported as a finding (see the validation log) — not
  buried. Self-grading is suspect; use an **independent judge** for LLM-output quality.
- **Graduation:** a spike that proves a mechanism graduates into `src/` *rebuilt to §2–§7 standards* (the spike is
  the proof, not the implementation).
- **Eval/observability from day 1** for the product code (traces, recorded metrics) — the eval surfaces are the
  product.

## 10. Docstrings
Google style (`Args:` / `Returns:` / `Raises:`), enforced by ruff `D` rules. Public API documented; private helpers
documented when non-obvious. Don't restate the signature — state intent, contracts, and surprises.

## 11. Boundaries: data models, LLM I/O, observability, errors & logging

### Data models — pydantic at the boundary, frozen dataclasses in the core *(decision: per use-case)*
- **Frozen `@dataclass(slots=True)` for the pure core** — the world-spec IR and all internal value objects:
  immutable, fast, dependency-free, **valid-by-construction** (validated once at the boundary, never re-checked
  downstream — *parse, don't validate*). The hot math core pays no validation tax.
- **Pydantic v2 at boundaries** — LLM structured outputs, CLI args, config, anything parsed from external/untrusted
  text: validation + coercion + caller-legible errors. **Convert the boundary pydantic model into the frozen-
  dataclass core IR at the edge** (pydantic = the wire/parse type; dataclass = the domain type).
- *Rule:* pydantic where data crosses a trust boundary; dataclass for trusted, already-validated internal values.

### LLM structured output — instructor + pydantic, bounded retry
- Use **[instructor](https://python.useinstructor.com/)** (built on pydantic) for every LLM call that returns
  structure (the author's world-spec; the `Judge`'s prior-edges + faithfulness). It validates the response against a
  pydantic model and **re-asks on validation failure** (feeds the error back) — provider-agnostic (Gemini + others;
  maps to the provider's native structured-output / tool-calling / JSON mode).
- Retries are **bounded** → then **raise** (fail loud; never fabricate). Same propose→validate→re-ask discipline as
  the author→gate loop, at the single-call level. The instructor call lives **behind our `Judge`/author adapter**
  (the only place a provider client appears). Native provider structured output (e.g. Gemini `response_schema`) is
  the documented fallback.

### Observability — Langfuse (OTEL-based) + OpenTelemetry, from day 1
- Instrument LLM calls + each pipeline stage (author→gate→admit→grade→eval) with **[Langfuse](https://langfuse.com/)
  Python SDK v3+** (OTEL-based) — captures prompt/response/**token+cost+latency**/errors and supports eval; composes
  with any standard OTel setup.
- **Optional at runtime** (no key → degrade to local logging; the package still runs), but instrumentation points
  are first-class and sit behind a thin **tracing seam** (`@observe`/`Tracer` boundary) so Langfuse is swappable.
  This makes the "eval + observability from day 1" non-negotiable concrete.

### Errors & logging — designed, not ad hoc
- **Exception hierarchy:** a package-root `CausalWorldsError`; domain errors subclass it (`SpecError` today;
  `AuthorError`, `GraderError`, `JudgeError`, `BudgetExceededError` as components land). Caller-oriented types,
  context in the message, **fail loud — never fabricate a value.**
- **Logging (library discipline):** log to `logging.getLogger("causal_worlds")` with a **`NullHandler`** — *the
  library never configures root logging or handlers*; the **application/CLI** owns output (and may enable JSON/
  structured logs). **Secrets never logged** (the Gemini key lives in env only).
- **Three channels, never conflated:** *logs* = operational events (shell only) · *traces* = LLM/pipeline spans
  (Langfuse/OTel) · *exceptions* = control flow. The **pure core stays silent** (raises typed errors, no IO/logging);
  the **imperative shell** logs and traces.
