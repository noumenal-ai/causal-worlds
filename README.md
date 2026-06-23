# causal-worlds

[![PyPI](https://img.shields.io/pypi/v/causal-worlds.svg)](https://pypi.org/project/causal-worlds/)
[![CI](https://github.com/noumenal-ai/causal-worlds/actions/workflows/ci.yml/badge.svg)](https://github.com/noumenal-ai/causal-worlds/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org)

**Turn a plain-language description of an operation into a fictional causal world with a declared,
ground-truth causal graph — then benchmark whether a causal-discovery method can recover it.**

Because the structure is *declared* (not learned from data), it's an **answer key**: run any
discovery method on the generated data and *score* how well it recovered the world. The worlds are
**fiction-first** — plausible and internally consistent, not models of any real system — so there is
no data to leak and nothing to memorize, which is exactly what makes a causal benchmark trustworthy.

```python
from causal_worlds import worlds, grade_spec, InterventionalCiDiscoverer

spec = worlds.get("coffee")                          # a hidden confounder + a regime sign-flip
report = grade_spec(spec, InterventionalCiDiscoverer())
print(report)   # directed_shd=0  skeleton_shd=0  f1=1.0  confounded_reported=0
#                ^ swap in YOUR discoverer to benchmark it against a known truth
```

> **Status — v0.6, beta.** The full loop works: natural language → an *admitted* causal world,
> persisted with provenance. A Claude **author** proposes the world; an independent Gemini **judge**
> (a different model family) plus statistical gates admit only worlds that are valid, recoverable, and
> *not* guessable from variable names. The deterministic engine (specify → sample → grade → score) and
> all grading run with **no API key**; only *authoring* needs keys. Worlds are currently tabular SCMs
> with `do()` interventions — a Gymnasium env, temporal lags, and counterfactual replay are on the
> [roadmap](#roadmap). See the [CHANGELOG](CHANGELOG.md).

## Install

```bash
pip install causal-worlds              # or: uv add causal-worlds
pip install 'causal-worlds[discover]'  # + the baseline discovery stack (PC/GES/FCI/GIES)
pip install 'causal-worlds[llm]'       # + natural-language authoring (Claude + Gemini)
```

The base install (engine, grading, built-in worlds, CLI) needs only `typer`, `pydantic`, `numpy`.

## 60-second quickstart (no API key)

```bash
causal-worlds worlds                     # list built-in worlds: coffee, ecommerce
causal-worlds gate coffee                # run the validity gates -> admitted=True
causal-worlds grade coffee               # grade the reference discoverer -> directed_shd=0 ...
causal-worlds score benchmark/v0.5/world_01   # grade the reference on a shipped benchmark world
```

New to it? Walk through the **[getting-started guide](docs/getting-started.md)** or run the
**[examples](examples/)**.

## Benchmark your own discoverer

Implement one method — `recover(substrate, *, seed) -> set[(src, dst)]` — and grade it against any
world's answer key:

```python
from causal_worlds import grade_spec, worlds

class MyDiscoverer:
    def recover(self, substrate, *, seed):
        sample = substrate.sample(2000, seed=seed)        # observational data...
        flows = substrate.sample(2000, seed=seed, do={"price": 1.0})  # ...or interventional
        return {("price", "demand")}                       # your recovered edges

print(grade_spec(worlds.get("coffee"), MyDiscoverer()))
```

Or from the CLI on a persisted world: `causal-worlds score <bundle> --discoverer your_pkg:YourClass`.

## Author a world from a description (needs `[llm]` + keys)

Set `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` (see [`.env.example`](.env.example); the CLI auto-loads a
local `.env`), then:

```bash
causal-worlds generate "a coffee chain with weekend swings and variable lead times" ./my-world
```

**Observability:** with the `observability` extra + Langfuse keys and
`CAUSAL_WORLDS_LANGFUSE_ENABLED=true`, every run is traced (`generate` → `author` → `gate`) in Langfuse.

```python
from causal_worlds import generate
from causal_worlds.author import build_claude_author
from causal_worlds.judge import build_gemini_judge

world = generate(
    "a hospital ED with triage staffing and bed pressure",
    author=build_claude_author(complexity="hard"),   # easy | standard | hard
    judge=build_gemini_judge(),                       # independent model family
)
print(world.report.difficulty, world.report.grade)
```

## What the crossover shows (and what it doesn't)

Across the 35-world [`benchmark/v0.5`](benchmark/v0.5/) set (3 seeds each). The comparison is
**information-fair**: the `+do` methods get the *same interventional budget* (pooled observational +
per-variable `do()` environments) as the latent-aware reference — so we compare *methods*, not data
access ([full table + bootstrap CIs](evals/baseline-crossover/v0.5/)):

| method | data | latent-aware? | mean skeleton-SHD ↓ | confounded pair kept as causal ↓ |
|---|---|---|---|---|
| **interventional-ci** (reference) | interventional | yes | **1.44** | **0** |
| GIES | interventional | no | 4.62 | 17 |
| PC | observational | no | 2.72 | 14.3 |
| **PC + interventions** | interventional | no | 3.31 | **15.0** |
| FCI | observational | partly | 2.68 | 9.7 |
| FCI + interventions | interventional | partly | 3.29 | 6.7 |

The honest reading: the dividing line is **latent-awareness, not interventions**. The decisive row is
**PC + interventions** — given the *same* interventional budget as the reference, it still keeps the
hidden-confounded pair as a *causal* edge in ~15 worlds (no better than observational PC's 14.3);
GIES likewise (17). Only the latent-aware interventional rule reaches **0**. The interventional
advantage is robust: ΔF1 = F1(reference) − F1(method) is **+0.29, 95% CI [0.22, 0.35]** for
`pc+do` (every method's CI excludes 0). So this is an **identifiability result** (you cannot tell
confounding from causation without *both* interventions *and* a latent-aware method), not "our method
beats the toolbox."

**Caveats we're not hiding** (see [`evals/`](evals/) and the issues): (1) ~~the worlds are admitted by
the reference grader itself~~ **Fixed in v0.15**: admission (gate T3) is now **grader-independent** —
a world is admitted iff its declared SCM is *faithful by construction* (every edge induces a
detectable partial correlation; regimes genuinely modulate), computed in closed form from the spec
with **no discovery method run**. The reference grader's score is reported, never gates. (2)
**Simulated-DAG leakage** — synthetic SCMs can leak the causal order through marginal variance
([varsortability](evals/varsortability/)) *and* through scale-invariant predictability
(R²-sortability). v0.14 generates worlds with **internal standardization (iSCM)**, dropping
varsortability to 0.54 and R²-sortability 0.73 → 0.60; both trivial sorting baselines fall to F1
≈ 0.33–0.37, well under the real methods. The residual R²-sortability (0.60 > 0.5) is disclosed, not
yet fully closed. (3) Difficulty vs skeleton-SHD error is **descriptive, not a validated predictor**:
with bootstrap CIs (n=35), the observational methods show r≈0.40 (PC [0.07, 0.68], FCI [0.08, 0.68] —
just excluding 0) while the latent-aware reference is flat (r≈0.24, [−0.06, 0.51], includes 0).
(4) **The anti-cliché claim is currently weak — a known, measured gap.** A name-only LLM baseline
(guess the graph from names, no data) scores [F1 0.71](evals/name-only-baseline/) vs a 0.20 chance
floor; **anonymizing** names to `X1..Xn` only drops it to 0.61 — so the worlds leak through variable
names *and* through role labels + graph conventions. The T4 gate (rejects only at prior-F1 ≥ 0.9) is
too lax. Tightening the gate and authoring worlds where priors actively mislead is the next milestone.

## What you get per world

1. **An executable SCM** — sample observational data and `do()`-intervene, deterministically by seed.
2. **A time-series dataset** — the observed variables (the input to a discovery method).
3. **An answer key** — the declared causal edges + the hidden-confounded pairs, derived from the spec.
4. **A manifest** — full provenance (models, grader version, seed, difficulty) and an honesty label.

## Concepts

- **Spec / IR** — variables (with roles, incl. hidden), linear-Gaussian mechanisms, regime sign-flips.
- **Answer key** — directed edges over *observed* variables + the hidden-confounded pairs; *derived*
  from the spec, never stored separately, so they can't disagree.
- **Gates** — T1 validity · T2 sample-sanity · T3 faithfulness (grader-independent: the declared SCM
  is faithful & non-trivial by construction) · T4 anti-cliché (the judge can't guess it from names).
  A world is admitted only if all pass.
- **Reference grader** — an interventional-CI discoverer that uses `do()` data to tell *confounding*
  from *causation*, where PC/GES/GIES/FCI (which assume causal sufficiency) cannot.

Depth: [`docs/scope.md`](docs/scope.md) · [`docs/hld.md`](docs/hld.md) · [`docs/lld.md`](docs/lld.md)
· [`docs/architecture.md`](docs/architecture.md) · [`docs/validation.md`](docs/validation.md).

## Roadmap

Shipped: NL authoring, independent judge + anti-cliché gate, artifact persistence, the baseline
crossover, a structural-difficulty axis, a 35-world benchmark, **temporal worlds** (lagged edges +
autoregression — see the built-in `supply`), and **time-series grading** (PCMCI+, LPCMCI, VARLiNGAM,
Granger — `grade_temporal_spec`), and **authoring temporal worlds** (an LLM-authored lagged world,
admitted through a PCMCI+ temporal gate). Next: a **temporal benchmark *set*** (scale + crossover at
n>1), **a Gymnasium env** with perturbations + counterfactual replay, **scaling to 100+ worlds**, and
conversational **elicitation**. Tracked as [issues](https://github.com/noumenal-ai/causal-worlds/issues).

## Why this is the unoccupied intersection

Today's tools each own one corner — *natural-language authoring × executable causal simulator ×
ground-truth answer-key for discovery* is the gap:

| Tool | Corner it owns | What it lacks (for this job) |
|---|---|---|
| **[G-Sim](https://arxiv.org/abs/2506.09272)** | LLM authors a sim + calibrates to data | needs *real data*; aimed at fidelity, not a declared answer-key |
| **[DEVS-Gen](https://arxiv.org/abs/2603.03784)** | NL → executable discrete-event ops sim | no declared causal-graph answer-key |
| **[SD-SCM](https://arxiv.org/abs/2411.08019)** | LLM fills mechanisms → counterfactuals | needs a *user-supplied* DAG; tabular, not an executable sim |
| **[TimeGraph](https://arxiv.org/abs/2506.01361)** | known-graph time-series for discovery | parametric/templated; no natural-language authoring |

Built on the shoulders of [pgmpy](https://github.com/pgmpy/pgmpy),
[DoWhy](https://github.com/py-why/dowhy),
[CausalPlayground](https://github.com/sa-and/CausalPlayground),
[causal-learn](https://github.com/py-why/causal-learn), and
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## Contributing

Issues and PRs welcome. The bar: `make validate` green (ruff `select=ALL`, mypy `strict`, pytest with
a coverage floor) — see [`docs/engineering.md`](docs/engineering.md). Atomic, conventional commits.

## License

[MIT](LICENSE). An open-source project from [Noumenal](https://github.com/noumenal-ai).
