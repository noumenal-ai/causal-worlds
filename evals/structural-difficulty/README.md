# Structural difficulty vs error

v0.3 found name-guessability difficulty doesn't predict discovery error. This re-analysis (reusing the crossover report — no new runs) tests whether *structural* difficulty (confounded pairs + regime sign-flips) does.

| correlation | name-guessability | structural |
|---|---|---|
| vs observational skeleton-SHD | +0.14 | -0.12 |
| vs interventional advantage (ΔF1) | -0.38 | -0.39 |

*Interventional advantage* = reference F1 − mean observational F1 per world: how much the `do()`-based grader gains over the observational toolbox.

## Verdict — inconclusive at n=12 (needs scale)

Neither axis cleanly predicts the magnitude of the collapse on the current set (correlations span
−0.39…+0.14). The difficulty range is narrow and the sample is tiny — this is a statistical-power
problem, not a refutation. The structural-difficulty metric ships as infrastructure (and is recorded
in every world's manifest); whether a difficulty *scalar* predicts error is deferred to the scaled set
(v0.5: 40+ worlds with a deliberate easy→hard spread).

**The crossover itself (v0.3) is unaffected and robust** — standard methods collapse, the
interventional grader holds. What's open is only the difficulty-*predicts*-error question.

Reproduce (keyless): `uv run python evals/structural-difficulty/run_analysis.py`.
