# Baseline crossover — information-fair (#9)

Every world in benchmark `v0.5` vs the standard discoverers and the reference grader `interventional-ci@1`, at n=4000, averaged over seeds [7, 11, 23]. The **interventional track** (`+do`, gies, reference) gives the causal-sufficiency methods the *same interventional budget* (pooled observational + per-variable do() environments) as the latent-aware reference. Skeleton-SHD = adjacency errors; confounded-kept = hidden-confounded pairs reported as *causal* (the trap; lower better).

| method | data | worlds | mean skeleton-SHD | mean directed F1 | confounded-kept |
|---|---|---|---|---|---|
| `interventional-ci` | interventional | 35 | 1.44 | 0.91 | 0.00 |
| `pc` | observational | 35 | 2.72 | 0.69 | 14.33 |
| `ges` | observational | 0 (+35 errored) | — | — | — |
| `fci` | observational | 35 | 2.67 | 0.72 | 9.67 |
| `dagma` | observational | 35 | 5.73 | 0.22 | 16.00 |
| `directlingam` | observational | 35 | 5.64 | 0.38 | 14.67 |
| `gies` | interventional | 35 | 4.66 | 0.75 | 17.00 |
| `pc+do` | interventional | 35 | 3.30 | 0.62 | 15.00 |
| `fci+do` | interventional | 35 | 3.28 | 0.64 | 6.67 |

### Interventional advantage — ΔF1 = F1(reference) − F1(method), 95% bootstrap CI

| method | mean ΔF1 | 95% CI |
|---|---|---|
| `pc` | 0.22 | [0.17, 0.27] |
| `fci` | 0.19 | [0.15, 0.23] |
| `dagma` | 0.69 | [0.63, 0.75] |
| `directlingam` | 0.53 | [0.48, 0.57] |
| `gies` | 0.16 | [0.12, 0.19] |
| `pc+do` | 0.29 | [0.22, 0.35] |
| `fci+do` | 0.27 | [0.21, 0.33] |

### Difficulty vs skeleton-SHD error — Pearson r, 95% bootstrap CI

| method | r | 95% CI |
|---|---|---|
| `interventional-ci` | 0.24 | [-0.06, 0.51] |
| `pc` | 0.40 | [0.06, 0.68] |
| `ges` | — | — |
| `fci` | 0.42 | [0.08, 0.69] |
| `dagma` | 0.17 | [-0.18, 0.46] |
| `directlingam` | 0.03 | [-0.30, 0.36] |
| `gies` | 0.09 | [-0.26, 0.44] |
| `pc+do` | 0.24 | [-0.11, 0.58] |
| `fci+do` | 0.23 | [-0.13, 0.59] |

**Read:** the headline is the **interventional track** — if `pc+do`, `fci+do`, and `gies` still keep confounded pairs and post a positive ΔF1 (CI excluding 0) while the latent-aware `interventional-ci` reaches confounded-kept 0, the dividing line is **latent-awareness, not interventions** (an identifiability result). Difficulty-vs-error is descriptive (n is small, predictors discrete); the bootstrap CI shows how wide it really is.

Reproduce: `uv run python evals/baseline-crossover/run_crossover.py benchmark/v0.5` (needs the `discover` extra).
