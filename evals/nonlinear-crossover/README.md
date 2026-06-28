# Nonlinear crossover — does linear discovery miss a nonlinear edge the key still scores?

A fixed confounded graph with the `X→M` edge's transform swept across ['identity', 'square', 'cube', 'tanh', 'relu', 'abs'], over 3 templates × seeds [7, 11, 23] (n=4000), graded by `interventional-ci@1`. A transform never changes the edge set, so `X→M` is in the answer key for every world; the cells are **how often each method recovers it** (nonlinear-edge recall, 1.0 = always).

### Nonlinear-edge recall of `X→M`, by transform

| transform | `interventional-ci` | `pc` | `fci` | `dagma` | `directlingam` | `gies` | `pc+do` | `fci+do` |
|---|---|---|---|---|---|---|---|---|
| identity | 1.00 | 1.00 | 1.00 | 0.11 | 0.67 | 1.00 | 1.00 | 1.00 |
| square | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.89 |
| cube | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| tanh | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| relu | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| abs | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.67 | 0.67 |

### Aggregate over all nonlinear worlds

| method | worlds | mean F1 | mean SHD | confounded-kept | `X→M` recall |
|---|---|---|---|---|---|
| `interventional-ci` | 18 | 1.00 | 0.00 | 0.00 | 1.00 |
| `pc` | 18 | 0.75 | 1.78 | 18.00 | 0.67 |
| `fci` | 18 | 0.75 | 1.67 | 18.00 | 0.67 |
| `dagma` | 18 | 0.03 | 4.26 | 16.00 | 0.02 |
| `directlingam` | 18 | 0.52 | 2.81 | 16.00 | 0.11 |
| `gies` | 18 | 0.76 | 3.00 | 18.00 | 1.00 |
| `pc+do` | 18 | 0.71 | 2.15 | 18.00 | 0.94 |
| `fci+do` | 18 | 0.68 | 2.15 | 12.33 | 0.93 |

**Read — two orthogonal failure modes, cleanly separated (the edge is in the key throughout):**

1. **Nonlinearity bites *correlation-based* discovery only for *even* transforms.** For `square` and `abs` — where the linear correlation of `X` with `M` is ≈ 0 — the correlation-based observational methods (`pc`, `fci`) recover `X→M` **0%** of the time; for odd/monotone transforms (`cube`, `tanh`, `relu`) they recover it fine (residual linear correlation remains). The recoverer of the even-nonlinear edge is **interventional data**: `pc+do`, `gies`, and the reference recover `square` (and `pc+do`/`fci+do` partially recover `abs`) — *none of which are latent-aware*. So for **nonlinearity the lever is interventions, not latent-awareness.**
2. **Latent confounding is the separate §4.1 axis, unchanged.** Every causal-sufficiency method — including the interventional `pc+do`, `fci+do`, `gies` — keeps the hidden-confounded `{A,B}` pair as causal (~18/18); only the latent-aware `interventional-ci` reaches 0. For **confounding the lever is latent-awareness, not interventions.**

This *refines* the n=1 `braking` framing (§4.7): `braking`'s `speed²` is an **even** transform, so observational PC missed it — but the general recoverer of even-nonlinear edges is interventional data, and latent-awareness is the separate lever for confounding. (`dagma`/`directlingam` are weak throughout — including on the linear `identity` control — so their misses are not nonlinearity-specific.)

Reproduce: `uv run python evals/nonlinear-crossover/run.py` (needs the `discover` extra).
