# causal-worlds

**Generate a fictional-but-coherent causal *operation* from a plain-language description** — an executable
simulator, the multivariate time-series it emits, and a **declared ground-truth causal structure** (the
*answer-key*) — for building, perturbing, and **benchmarking causal-discovery and control agents against a known
truth**.

> **Status: early / pre-release.** Design in progress — see [`docs/`](docs/). APIs and schema will change.

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
