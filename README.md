# causal-worlds

**Generate a fictional-but-coherent causal *operation* from a plain-language description** — an executable
simulator, the multivariate time-series it emits, and a **declared ground-truth causal structure** (the
*answer-key*) — for **benchmarking causal-discovery agents against a known truth** (and, on the roadmap,
stress-testing control agents under perturbation).

> **Status: v0.2.0 — the loop is closed.** Describe an operation in plain language and get back an *admitted*
> causal world: an executable simulator, the time-series it emits, a declared ground-truth answer-key, and a
> manifest. A Claude **author** proposes the world; an independent Gemini **judge** (a different model family)
> plus statistical gates admit only worlds that are valid, recoverable, and *not* guessable from priors;
> admitted worlds are persisted as self-describing bundles. The deterministic v0.1 **engine** (specify → sample →
> grade → score) is still fully usable with **no API key**. See [`CHANGELOG.md`](CHANGELOG.md).
>
> What's validated: on the built-in `coffee` world (a hidden confounder + a regime sign-flip), standard
> observational/score-based discovery fails, but the reference **interventional-CI** grader recovers the structure
> (directed SHD 0) and drops the confounded edge — pinned as a test. The author model is chosen by a reproducible,
> judged [bake-off](evals/author-model-bakeoff/), not by assertion.

## Quickstart

```bash
uv add causal-worlds            # or: pip install causal-worlds   (once published)
causal-worlds worlds            # list built-in worlds: coffee, ecommerce
causal-worlds gate coffee       # run the validity gates -> admitted=True
causal-worlds grade coffee      # grade the reference discoverer -> directed_shd=0  f1=1.00  confounded_reported=0
```

Author a world from a description (needs the `llm` extra + an Anthropic and a Gemini key in the env):

```bash
uv add 'causal-worlds[llm]'
causal-worlds generate "a coffee chain with weekend swings and variable lead times" ./my-world
causal-worlds benchmark benchmark/prompts.txt ./benchmark/v0.2   # author + admit a whole set
```

As a library:

```python
from causal_worlds import worlds, build_substrate, answer_key, score, InterventionalCiDiscoverer

spec = worlds.get("coffee")
recovered = InterventionalCiDiscoverer().recover(build_substrate(spec), seed=7)
print(score(recovered, answer_key(spec)))   # plug in your own discoverer instead to benchmark it
```

## Why

Describe an operation in a sentence — *"a three-store coffee chain with weekend demand swings and variable
supplier lead times"* — and get back a small world you can **run, perturb, intervene on, and replay
counterfactually**, together with the **causal structure that generated it**. Because the structure is *declared*
(not learned), it serves as a ground-truth **answer-key**: you can run a causal-discovery method on the generated
data and *score* how well it recovered the world.

There is a real gap here. Today's tools each own one corner of the problem but not the whole:

| Tool | Corner it owns | What it lacks (for this job) |
|---|---|---|
| **[G-Sim](https://arxiv.org/abs/2506.09272)** | LLM authors a sim + calibrates to data | needs *real data*; aimed at fidelity, not a declared answer-key |
| **[DEVS-Gen](https://arxiv.org/abs/2603.03784)** | NL → executable discrete-event ops sim | validates against constraints; no declared causal-graph answer-key |
| **[SD-SCM](https://arxiv.org/abs/2411.08019)** | LLM fills mechanisms → ground-truth counterfactuals | needs a *user-supplied* DAG; tabular, not an executable temporal sim |
| **[TimeGraph](https://arxiv.org/abs/2506.01361)** | known-graph time-series for benchmarking discovery | parametric/templated; no natural-language authoring |
| **[GIF-MCTS](https://arxiv.org/abs/2405.15383)** / **[WorldCoder](https://arxiv.org/abs/2402.12275)** | LLM writes a runnable "Code World Model" | no causal answer-key; discrete-RL domains |

`causal-worlds` targets the **unoccupied intersection**: *natural-language authoring × temporal/regime causal
structure × executable simulator × ground-truth answer-key*, **fiction-first** — the goal is *plausible and
internally consistent*, not matching any real system (so no calibration data is required).

## What you get (per generated world)

1. **An executable simulator** — a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)-style `reset()` / `step(action) → (obs, reward, done)` environment.
2. **A time-series dataset** — the simulator's output (the input to a causal-discovery method).
3. **An answer-key** — the declared causal structure: variables and their roles, a causal graph (direction + lag),
   functional forms, and regimes — emitted in an open, documented schema.
4. **A manifest** binding them together, plus an explicit **honesty label**: these worlds are *fictional* and not
   real-world advice.

## How it works (sketch)

```
natural-language description
   → author a world spec (LLM proposes variables, roles, causal graph, functional forms, lags, regimes)
   → consistency / conformance checks (acyclic, constraint-respecting, non-degenerate, identifiable) — no calibration data
   → sample + execute  (SCM sampling, or staged discrete-event synthesis, by world type)
   → emit the artifact triple + score a pluggable causal-discovery agent against the answer-key (SHD / F1, interventional, counterfactual)
```

See [`docs/scope.md`](docs/scope.md), [`docs/hld.md`](docs/hld.md), [`docs/lld.md`](docs/lld.md).

## The benchmark set

The headline set ships in [`benchmark/v0.5`](benchmark/v0.5/) — **35 fictional operations** authored by
Claude across an **easy→hard complexity spread**, admitted through the gates, judged by Gemini, and
graded. Each is a self-describing bundle (`spec.json` / `data.npz` / `answer_key.json` / `manifest.json`)
with full provenance (models, grader version, seed, difficulty, structural difficulty, complexity). The
original 12-world [`benchmark/v0.2`](benchmark/v0.2/) is kept for continuity.

**Does it defeat the standard toolbox?** Yes — measured across 35 worlds: the reference interventional-CI
grader **never** reports a hidden-confounded pair as causal (confounded-kept 0, SHD 1.47, F1 0.91) while
PC/FCI/GIES report 8–17 such spurious edges and post 2–4× the structural error. See the
[baseline crossover](evals/baseline-crossover/v0.5/).

**Is difficulty a real instrument?** Yes, when measured on *structure*: **structural** difficulty
(hidden confounders + regime sign-flips) predicts observational error (corr **+0.62**), whereas
name-guessability difficulty does not (**+0.14**). See [structural difficulty](evals/structural-difficulty/v0.5/).

## Built on the public domain

Stands on the shoulders of (and learns from): [pgmpy](https://github.com/pgmpy/pgmpy),
[DoWhy](https://github.com/py-why/dowhy), [CausalPlayground](https://github.com/sa-and/CausalPlayground),
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium), and the ideas in
[G-Sim](https://arxiv.org/abs/2506.09272), [DEVS-Gen](https://arxiv.org/abs/2603.03784),
[SD-SCM](https://arxiv.org/abs/2411.08019), [TimeGraph](https://arxiv.org/abs/2506.01361),
[GIF-MCTS](https://arxiv.org/abs/2405.15383), and [WorldCoder](https://arxiv.org/abs/2402.12275).

## License

[MIT](LICENSE).

---

An open-source project from [Noumenal](https://github.com/noumenal-ai).
