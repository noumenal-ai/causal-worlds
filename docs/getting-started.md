# Getting started

A hands-on tour of `causal-worlds`. Everything in the first three sections runs with **no API key**;
authoring (the last section) needs the `[llm]` extra and provider keys.

## Install

```bash
uv add causal-worlds              # engine, grading, built-in worlds, CLI
uv add 'causal-worlds[discover]'  # + PC/GES/FCI/GIES baselines
uv add 'causal-worlds[llm]'       # + natural-language authoring
```

(With pip: `pip install 'causal-worlds[discover]'`.)

## 1. Grade the reference discoverer on a built-in world

A *world* is a structural causal model with a declared answer key. `coffee` hides a confounder and
flips a price→demand effect by regime — the trap that defeats observational discovery.

```python
from causal_worlds import worlds, grade_spec, InterventionalCiDiscoverer

spec = worlds.get("coffee")
report = grade_spec(spec, InterventionalCiDiscoverer())
print(report)
# Report(directed_shd=0, skeleton_shd=0, f1=1.0, n_truth=6, n_recovered=6, confounded_reported=0)
```

`confounded_reported=0` is the point: the grader did **not** mistake the hidden-confounded pair for a
causal edge. The same call from the CLI:

```bash
causal-worlds grade coffee
```

## 2. Benchmark *your own* discoverer

A discoverer is anything with `recover(substrate, *, seed) -> set[(src, dst)]`. The `substrate` is the
executable world: call `substrate.sample(n, seed=...)` for observational data, or pass
`do={"price": 1.0}` to intervene. Return the directed edges you recover.

```python
from causal_worlds import grade_spec, worlds


class CorrelationDiscoverer:
    """A deliberately naive baseline: link strongly-correlated observed variables."""

    def recover(self, substrate, *, seed):
        import numpy as np

        sample = substrate.sample(4000, seed=seed)
        data, names = sample.data, substrate.variables
        corr = np.corrcoef(data, rowvar=False)
        edges = set()
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j and abs(corr[i, j]) > 0.5:
                    edges.add((a, b))   # undirected guess; direction is left to chance
        return edges


report = grade_spec(worlds.get("coffee"), CorrelationDiscoverer())
print(report)   # high confounded_reported — correlation can't tell confounding from causation
```

Run the full standard toolbox against the shipped benchmark with the bundled harness:

```bash
uv run python evals/baseline-crossover/run_crossover.py benchmark/v0.5
```

## 3. Inspect a benchmark world

Each admitted world is a self-describing bundle on disk. Load one and read its truth + provenance:

```python
from causal_worlds import load_bundle, answer_key

bundle = load_bundle("benchmark/v0.5/world_01")
print(bundle.columns)              # observed variable names (data.npz columns)
print(bundle.data.shape)           # (rows, n_observed)
print(answer_key(bundle.spec).edges)        # ground-truth directed edges
print(answer_key(bundle.spec).confounded)   # hidden-confounded pairs (NOT causal edges)
print(bundle.manifest["difficulty"], bundle.manifest["structural_difficulty"])
```

## 4. Author a world from a description (needs keys)

Set `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`, install `[llm]`, then let Claude author a world and the
independent Gemini judge gate it:

```python
from causal_worlds import generate
from causal_worlds.author import build_claude_author
from causal_worlds.judge import build_gemini_judge

world = generate(
    "a ride-hailing marketplace with surge pricing and driver churn",
    author=build_claude_author(complexity="hard"),   # easy | standard | hard
    judge=build_gemini_judge(),
)
print(f"admitted in {world.attempts} attempt(s); difficulty {world.report.difficulty:.2f}")

from causal_worlds import save_bundle
from causal_worlds.artifact import Provenance

save_bundle(world, "./my-world", provenance=Provenance(
    author_model="claude-opus-4-8", judge_model="gemini-2.5-flash",
    grader="interventional-ci", grader_version="1", seed=0, n_rows=2000,
))
```

Or one line on the CLI: `causal-worlds generate "<description>" ./my-world`.

## Where to go next

- The runnable scripts in [`examples/`](../examples/).
- How it's designed: [`docs/architecture.md`](architecture.md), [`docs/hld.md`](hld.md).
- Why it's trustworthy: [`docs/validation.md`](validation.md) and the [`evals/`](../evals/) reports.
