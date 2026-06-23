# Varsortability control

Benchmark `v0.5`, n=4000, seed 7. Varsortability (Reisach et al.): 0.5 = the causal order is NOT readable from marginal variances; toward 1.0 = it is (and the trivial `sortnregress` baseline would win).

- mean **varsortability 0.94**
- `sortnregress` baseline: mean **F1 0.74**, mean directed SHD 3.69

**Verdict: GAMEABLE — variance leaks the causal order; standardize emitted variances.**

Reproduce: `uv run python evals/varsortability/run.py`.
