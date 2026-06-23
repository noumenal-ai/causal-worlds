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
causal-worlds score benchmark/v0.6/world_01   # grade the reference on a shipped benchmark world
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

## Benchmark a controller (Stage 2 — control)

The same worlds are a **control** benchmark: pick lever values to maximise an objective. Because the
mechanisms are declared, the best the levers can do is computable — a *by-construction optimal policy*
([scope §1a](docs/scope.md)) — so a policy is graded by **regret** against it, no external data needed.

```python
from causal_worlds import default_objective, grade_control, optimal_policy, worlds

spec = worlds.get("coffee")
objective = default_objective(spec)            # controllables raise the outcome KPI, quadratic cost
report = grade_control(spec, objective, {"price": 0.0})   # your policy here
print(report.regret, report.optimal_policy)    # regret vs the declared optimum (0 = optimal play)
```

A pluggable `Controller` (one method, `control(substrate, objective, *, seed) -> {lever: value}`,
which may `do()`-experiment on the world but never sees the mechanisms) is graded by `grade_controller`.
Or drive it as a **Gymnasium env** (`pip install 'causal-worlds[gym]'`):

```python
from causal_worlds.gym import ControlEnv          # the regime shifts between steps (a perturbation)
env = ControlEnv(worlds.get("coffee"))            # action = lever values; reward = objective
obs, info = env.reset(seed=0)                      # info["optimal_reward"] / info["regret"] per step
```

## Author a world from a description (needs `[llm]` + keys)

Set `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` (see [`.env.example`](.env.example); the CLI auto-loads a
local `.env`), then:

```bash
causal-worlds generate "a coffee chain with weekend swings and variable lead times" ./my-world
```

**Or describe a world conversationally.** A one-shot prompt is underspecified, so `elicit` runs a
short dialogue first — it asks the *minimal* clarifying questions (entities & roles, what drives what,
regimes, hidden causes, the objective), shows the accumulating brief, and authors only once the brief
is complete (or you type `go`):

```bash
causal-worlds elicit ./my-world      # interactive: answer a few questions, then it generates
```

**Observability:** with the `observability` extra + Langfuse keys and
`CAUSAL_WORLDS_LANGFUSE_ENABLED=true`, every run is traced (`generate` → `author` → `gate`) in Langfuse.

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

## What the crossover shows (and what it doesn't)

Across the 26-world hardened [`benchmark/v0.6`](benchmark/v0.6/) set (3 seeds each). The comparison is
**information-fair**: the `+do` methods get the *same interventional budget* (pooled observational +
per-variable `do()` environments) as the latent-aware reference — so we compare *methods*, not data
access ([full table + bootstrap CIs](evals/baseline-crossover/v0.6/)):

| method | data | latent-aware? | mean skeleton-SHD ↓ | confounded pair kept as causal ↓ |
|---|---|---|---|---|
| **interventional-ci** (reference) | interventional | yes | **1.31** | **0** |
| GIES | interventional | no | 5.71 | 30 |
| PC | observational | no | 3.18 | 29 |
| **PC + interventions** | interventional | no | 4.71 | **30** |
| FCI | observational | partly | 3.17 | 21.7 |
| FCI + interventions | interventional | partly | 4.73 | 21 |
| DAGMA | observational | no | 5.23 | 27 |
| DirectLiNGAM | observational | no | 5.76 | 27 |

(DAGMA and DirectLiNGAM run at default hyperparameters, and LiNGAM's non-Gaussian assumption is
violated by these linear-Gaussian worlds, so their *skeleton* accuracy is not their best — but the
relevant, robust verdict is **confounded-kept**, and like every causal-sufficiency method they keep it.)

The honest reading: the dividing line is **latent-awareness, not interventions**. The decisive row is
**PC + interventions** — given the *same* interventional budget as the reference, it still keeps the
hidden-confounded pair as a *causal* edge in **30** worlds (no better than observational PC's 29);
GIES likewise (30). Only the latent-aware interventional rule reaches **0**. The interventional
advantage is robust: ΔF1 = F1(reference) − F1(method) is **+0.37, 95% CI [0.33, 0.42]** for
`pc+do` (every method's CI excludes 0). So this is an **identifiability result** (you cannot tell
confounding from causation without *both* interventions *and* a latent-aware method), not "our method
beats the toolbox." (The earlier, easier [`v0.5`](benchmark/v0.5/) set is kept for comparison.)

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
(4) **Anti-cliché: name leakage fixed, role leakage remains (measured).** The `adversarial` author +
strict T4 gate (v0.19) regenerated the set as [`benchmark/v0.6`](benchmark/v0.6/). On the
[3-tier certificate](evals/name-only-baseline/v0.6/), the **named** name-only prior fell from v0.5's
**0.71** to **0.38** (chance 0.18) — and the **name+role-blind** prior collapses to **0.01**, proving
the structure itself is *not* guessable once semantics are stripped. Strikingly, the **name-blind**
prior (0.46) is *higher* than named — the adversarial names now actively **mislead**. The residual is
**role-type priors** (controllable→outcome conventions). v0.24 added a **roles-only gate** and a
companion [`benchmark/v0.7`](benchmark/v0.7/) authored under it (11 worlds): it only nudged the
name-blind prior 0.46 → 0.43 (chance 0.16) — single-sample LLM-judge gating is noisy and role-type is
*intrinsically* informative (you can't remove it while keeping the lever→outcome path). The honest
bottom line, separated by the [3-tier certificate](evals/name-only-baseline/v0.6/): the cliché that
matters — **name/structure memorization — is eliminated** (fully-blind ≈ 0.00, named cut 0.71→0.4);
the remaining role-type signal is legitimate operational prior, transparently reported, not hidden.

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
  is faithful & non-trivial by construction) · T4 anti-cliché (the named prior recovers < half —
  difficulty ≥ 0.5 — *and* a name+role-blind prior stays near chance). A world is admitted only if all
  pass.
- **Reference grader** — an interventional-CI discoverer that uses `do()` data to tell *confounding*
  from *causation*, where PC/GES/GIES/FCI (which assume causal sufficiency) cannot.

Depth: [`docs/scope.md`](docs/scope.md) · [`docs/hld.md`](docs/hld.md) · [`docs/lld.md`](docs/lld.md)
· [`docs/architecture.md`](docs/architecture.md) · [`docs/validation.md`](docs/validation.md).

## Roadmap

Shipped: NL authoring, independent judge + anti-cliché gate, artifact persistence, the baseline
crossover, a structural-difficulty axis, a **26-world hardened benchmark** (`v0.6`; name-only prior
F1 0.38 vs 0.71 on the old `v0.5`), **temporal worlds** (lagged edges +
autoregression — see the built-in `supply`), and **time-series grading** (PCMCI+, LPCMCI, VARLiNGAM,
Granger — `grade_temporal_spec`), **authoring temporal worlds** (an LLM-authored lagged world,
admitted through a PCMCI+ temporal gate), and **conversational elicitation** (`causal-worlds elicit`
— a dialogue that builds a `WorldBrief` before authoring). and the **control track** (Stage 2): a
**by-construction optimal-policy answer-key** with regret scoring *and* **regret-under-perturbation**
(`regret_under_perturbation` — a regime-blind policy collapses when a sign-flipped regime is active,
the stay-optimal thesis; see [scope §1a](docs/scope.md)), and a **Gymnasium env**
(`causal_worlds.gym.ControlEnv`) that drives the control loop as an RL environment with regime-shift
perturbations. Next: **nonlinearity** (#10) and a **temporal benchmark *set*** (n>1). Tracked as
[issues](https://github.com/noumenal-ai/causal-worlds/issues).

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
