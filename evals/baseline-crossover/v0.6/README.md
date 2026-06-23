# Baseline crossover — information-fair (#9)

Every world in benchmark `v0.6` vs the standard discoverers and the reference grader `interventional-ci@1`, at n=4000, averaged over seeds [7, 11, 23]. The **interventional track** (`+do`, gies, reference) gives the causal-sufficiency methods the *same interventional budget* (pooled observational + per-variable do() environments) as the latent-aware reference. Skeleton-SHD = adjacency errors; confounded-kept = hidden-confounded pairs reported as *causal* (the trap; lower better).

| method | data | worlds | mean skeleton-SHD | mean directed F1 | confounded-kept |
|---|---|---|---|---|---|
| `interventional-ci` | interventional | 26 | 1.31 | 0.90 | 0.00 |
| `pc` | observational | 26 | 3.18 | 0.64 | 29.00 |
| `ges` | observational | 0 (+26 errored) | — | — | — |
| `fci` | observational | 26 | 3.17 | 0.66 | 21.67 |
| `dagma` | observational | 26 | 5.23 | 0.29 | 27.00 |
| `directlingam` | observational | 26 | 5.76 | 0.31 | 27.00 |
| `gies` | interventional | 26 | 5.71 | 0.69 | 30.00 |
| `pc+do` | interventional | 26 | 4.71 | 0.53 | 30.00 |
| `fci+do` | interventional | 26 | 4.73 | 0.53 | 21.00 |

### Interventional advantage — ΔF1 = F1(reference) − F1(method), 95% bootstrap CI

| method | mean ΔF1 | 95% CI |
|---|---|---|
| `pc` | 0.26 | [0.22, 0.30] |
| `fci` | 0.24 | [0.19, 0.28] |
| `dagma` | 0.61 | [0.56, 0.65] |
| `directlingam` | 0.59 | [0.54, 0.64] |
| `gies` | 0.21 | [0.18, 0.24] |
| `pc+do` | 0.37 | [0.33, 0.42] |
| `fci+do` | 0.37 | [0.30, 0.43] |

### Difficulty vs skeleton-SHD error — Pearson r, 95% bootstrap CI

| method | r | 95% CI |
|---|---|---|
| `interventional-ci` | 0.38 | [-0.13, 0.70] |
| `pc` | 0.29 | [-0.04, 0.66] |
| `ges` | — | — |
| `fci` | 0.22 | [-0.08, 0.56] |
| `dagma` | -0.07 | [-0.56, 0.34] |
| `directlingam` | -0.25 | [-0.59, 0.14] |
| `gies` | -0.13 | [-0.46, 0.19] |
| `pc+do` | -0.12 | [-0.45, 0.20] |
| `fci+do` | -0.12 | [-0.43, 0.20] |

**Read:** the headline is the **interventional track** — if `pc+do`, `fci+do`, and `gies` still keep confounded pairs and post a positive ΔF1 (CI excluding 0) while the latent-aware `interventional-ci` reaches confounded-kept 0, the dividing line is **latent-awareness, not interventions** (an identifiability result). Difficulty-vs-error is descriptive (n is small, predictors discrete); the bootstrap CI shows how wide it really is.

Reproduce: `uv run python evals/baseline-crossover/run_crossover.py benchmark/v0.5` (needs the `discover` extra).
