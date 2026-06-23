# Baseline crossover — the decisive experiment

Every world in benchmark `v0.5` vs the standard discoverers and the reference grader `interventional-ci@1`, at n=4000, averaged over seeds [7, 11, 23]. Skeleton-SHD = adjacency errors (lower better); confounded-kept = hidden-confounded pairs reported as *causal* (the trap; lower better).

| method | worlds | mean skeleton-SHD | mean directed F1 | confounded-kept | corr(difficulty, error) |
|---|---|---|---|---|---|
| `interventional-ci` | 35 | 1.44 | 0.91 | 0.00 | 0.14 |
| `pc` | 35 | 2.81 | 0.67 | 13.00 | 0.25 |
| `ges` | 0 (+35 errored) | — | — | — | — |
| `fci` | 35 | 2.67 | 0.71 | 8.33 | 0.30 |
| `gies` | 35 | 4.24 | 0.78 | 17.00 | 0.15 |

**Read:** if the observational/score-based methods (pc/ges/fci/gies) keep confounded pairs and post higher skeleton-SHD while `interventional-ci` stays near zero, the benchmark is non-trivial. If `corr(difficulty, error)` is positive for the standard methods and ~flat for the grader, difficulty is a real instrument (error rises with how guessable a world is).

Reproduce: `uv run python evals/baseline-crossover/run_crossover.py` (needs the `discover` extra).
