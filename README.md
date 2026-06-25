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

## See the world it builds

An SCM *is* a DAG, so look at it. `to_mermaid(spec)` and `to_dot(spec)` are **zero-dependency** string
renderers (or run `causal-worlds viz coffee`). The hidden confounder is drawn dashed — it's the latent
structure a discovery method never gets to see, and the reason the world is hard:

![The declared SCM for the built-in "coffee" world: a hidden confounder (dashed) drives several observed variables](https://raw.githubusercontent.com/noumenal-ai/causal-worlds/main/docs/figures/coffee_world.png)

```python
from causal_worlds import worlds, to_mermaid
print(to_mermaid(worlds.get("coffee")))   # paste into a ```mermaid block — GitHub renders it live
```

> **Status — on [PyPI](https://pypi.org/project/causal-worlds/), beta.** The full loop works: natural
> language → an *admitted* causal world, persisted with provenance. A Claude **author** proposes the
> world; an independent Gemini **judge** (a different model family) + statistical gates admit only
> worlds that are valid, recoverable, faithful, and *not* guessable from variable names. The engine
> (specify → sample → grade → score), all grading, and the renderers run with **no API key**; only
> *authoring* needs keys. Shipped: temporal (lagged) worlds, a control track, and a Gymnasium env.
> See the [CHANGELOG](CHANGELOG.md).

## Install

```bash
pip install causal-worlds              # or: uv add causal-worlds
pip install 'causal-worlds[discover]'  # + the baseline discovery stack (PC/GES/FCI/GIES)
pip install 'causal-worlds[llm]'       # + natural-language authoring (Claude + Gemini)
```

The base install (engine, grading, renderers, built-in worlds, CLI) needs only `typer`, `pydantic`,
`numpy`.

## 60-second quickstart (no API key)

```bash
causal-worlds worlds                     # list built-in worlds: coffee, ecommerce
causal-worlds viz coffee                 # print the SCM as Mermaid (--format dot for Graphviz)
causal-worlds gate coffee                # run the validity gates -> admitted=True
causal-worlds grade coffee               # grade the reference discoverer -> directed_shd=0 ...
causal-worlds score benchmark/v0.6/world_01   # grade the reference on a shipped benchmark world
```

New to it? Walk through the **[getting-started guide](docs/getting-started.md)** or run the
**[examples](examples/)** (each prints its expected output, so you can read without running).

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

## Benchmark a controller (Stage 2 — control)

The same worlds are a **control** benchmark: pick lever values to maximise an objective. Because the
mechanisms are declared, the best the levers can do is computable — a *by-construction optimal policy*
([scope §1a](docs/scope.md)) — so a policy is graded by **regret** against it, no external data needed.

```python
from causal_worlds import default_objective, grade_control, worlds

spec = worlds.get("coffee")
objective = default_objective(spec)                  # controllables raise the outcome KPI, quadratic cost
report = grade_control(spec, objective, {"price": 3.0}, seed=7)   # your policy here
print(report.regret)                                 # regret vs the declared optimum (0 = optimal play)
```

A pluggable `Controller` is graded by `grade_controller`. Or drive it as a **Gymnasium env**
(`pip install 'causal-worlds[gym]'`) where the regime shifts between steps (a perturbation):

```python
from causal_worlds.gym import ControlEnv
env = ControlEnv(worlds.get("coffee"))               # action = lever values; reward = objective
obs, info = env.reset(seed=0)                          # info["optimal_reward"] / info["regret"] per step
```

## Author a world from a description (needs `[llm]` + keys)

Set `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` (see [`.env.example`](.env.example); the CLI auto-loads a
local `.env`), then:

```bash
causal-worlds generate "a coffee chain with weekend swings and variable lead times" ./my-world
causal-worlds viz ./my-world             # ...then look at what it built
```

Or describe one **conversationally** — `causal-worlds elicit ./my-world` asks the *minimal* clarifying
questions (entities & roles, what drives what, regimes, hidden causes, the objective), shows the
accumulating brief, and authors once it's complete. In Python:

```python
from causal_worlds import generate
from causal_worlds.author import build_claude_author
from causal_worlds.judge import build_gemini_judge

world = generate(
    "a hospital ED with triage staffing and bed pressure",
    author=build_claude_author(complexity="hard"),   # easy | standard | hard | adversarial
    judge=build_gemini_judge(),                       # independent model family
)
print(world.report.difficulty, world.report.grade)
```

(With the `observability` extra + Langfuse keys, every `generate → author → gate` run is traced.)

## What the benchmark shows

Across the 26-world hardened [`benchmark/v0.6`](benchmark/v0.6/) set, with an **information-fair**
comparison (the `+do` methods get the *same* interventional budget as the latent-aware reference):
**latent-awareness — not interventions — is the dividing line.** `PC + interventions` still scores the
hidden-confounded pair as causal just as often as observational PC — confounded-kept **30 vs 29**
(summed over the 26 worlds, seed-averaged); only the latent-aware rule reaches **0** (ΔF1 +0.37, 95%
CI [0.33, 0.42]). It's an **identifiability result**, not
"we beat the toolbox." Full table, bootstrap CIs, and the honest caveats (admission circularity,
simulated-DAG leakage, difficulty-as-descriptor, anti-cliché role leakage) are in
**[docs/findings.md](docs/findings.md)**.

## What you get per world

1. **An executable SCM** — sample observational data and `do()`-intervene, deterministically by seed.
2. **A time-series dataset** — the observed variables (the input to a discovery method).
3. **An answer key** — the declared causal edges + the hidden-confounded pairs, derived from the spec.
4. **A manifest** — full provenance (models, grader version, seed, difficulty) and an honesty label.

## Concepts

- **Spec / IR** — variables (with roles, incl. hidden), linear-Gaussian mechanisms, regime sign-flips.
- **Answer key** — directed edges over *observed* variables + the hidden-confounded pairs; *derived*
  from the spec, never stored separately, so they can't disagree.
- **Gates** — T1 validity · T2 sample-sanity · T3 faithfulness (grader-independent) · T4 anti-cliché
  (named prior recovers < half *and* a name+role-blind prior stays near chance). All must pass to admit.
- **Reference grader** — an interventional-CI discoverer that uses `do()` data to tell *confounding*
  from *causation*, where PC/GES/GIES/FCI (which assume causal sufficiency) cannot.

Depth: [`docs/scope.md`](docs/scope.md) · [`docs/hld.md`](docs/hld.md) · [`docs/lld.md`](docs/lld.md)
· [`docs/architecture.md`](docs/architecture.md) · [`docs/findings.md`](docs/findings.md) ·
[`docs/validation.md`](docs/validation.md).

## Roadmap

Shipped: NL authoring · independent judge + anti-cliché gate · artifact persistence · the baseline
crossover · a structural-difficulty axis · a 26-world hardened benchmark (`v0.6`) · **temporal worlds**
(lagged edges + autoregression) and **time-series grading** (PCMCI+, LPCMCI, VARLiNGAM, Granger) ·
**conversational elicitation** · the **control track** (by-construction optimal policy, regret, and
regret-under-perturbation) + a **Gymnasium env** · **graph renderers** (Mermaid / DOT).
Next: **nonlinearity** ([#10](https://github.com/noumenal-ai/causal-worlds/issues/10)) and a temporal
benchmark *set* (n>1). Tracked as [issues](https://github.com/noumenal-ai/causal-worlds/issues).

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
