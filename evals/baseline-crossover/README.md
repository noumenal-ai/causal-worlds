# Baseline crossover — the decisive experiment

Every world in benchmark `v0.2` vs the standard discoverers and the reference grader `interventional-ci@1`, at n=4000, averaged over seeds [7, 11, 23]. Skeleton-SHD = adjacency errors (lower better); confounded-kept = hidden-confounded pairs reported as *causal* (the trap; lower better).

| method | worlds | mean skeleton-SHD | mean directed F1 | confounded-kept | corr(difficulty, error) |
|---|---|---|---|---|---|
| `interventional-ci` | 12 | 1.31 | 0.91 | 0.33 | -0.31 |
| `pc` | 12 | 3.22 | 0.57 | 8.33 | 0.11 |
| `ges` | 0 (+12 errored) | — | — | — | — |
| `fci` | 12 | 3.31 | 0.53 | 7.33 | 0.11 |
| `gies` | 12 | 4.53 | 0.78 | 10.00 | 0.05 |

**Read:** if the observational/score-based methods (pc/ges/fci/gies) keep confounded pairs and post higher skeleton-SHD while `interventional-ci` stays near zero, the benchmark is non-trivial. If `corr(difficulty, error)` is positive for the standard methods and ~flat for the grader, difficulty is a real instrument (error rises with how guessable a world is).

## Verdict — GO

**The crossover holds across the set.** Standard observational/score-based methods fall into the trap —
`pc`/`fci`/`gies` report the hidden-confounded pair as a *causal* edge in **7.3–10.0 of 12** worlds and
post **2–4× the skeleton error** — while the reference `interventional-ci` grader almost never does
(confounded-kept **0.33**, skeleton-SHD **1.31**, directed F1 **0.91**). The "defeats the standard
toolbox" claim now stands on 12 diverse worlds, not just `coffee`.

**Honest negative on the difficulty instrument.** Name-guessability difficulty does *not* yet predict
discovery error (baseline correlations 0.05–0.11). The discovery hardness comes from the *structural*
confounder+regime trap, present across the whole set, which the current difficulty metric (prior-F1 on
variable names) doesn't capture. This sharpens v0.4: push name-difficulty up **and** add a distinct
*structural*-difficulty axis.

(`ges` errored on every world — `causal-learn`'s GES is numpy-2 incompatible; reported, not hidden.)

Reproduce: `uv run python evals/baseline-crossover/run_crossover.py` (needs the `discover` extra).
