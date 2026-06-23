# Structural difficulty vs error

v0.3 found name-guessability difficulty doesn't predict discovery error. This re-analysis (reusing the crossover report — no new runs) tests whether *structural* difficulty (confounded pairs + regime sign-flips) does.

| correlation | name-guessability | structural |
|---|---|---|
| vs observational skeleton-SHD | +0.28 | +0.82 |
| vs interventional advantage (ΔF1) | +0.16 | +0.36 |

*Interventional advantage* = reference F1 − mean observational F1 per world: how much the `do()`-based grader gains over the observational toolbox.

Reproduce (keyless): `uv run python evals/structural-difficulty/run_analysis.py`.
