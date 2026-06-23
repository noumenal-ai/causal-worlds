# Varsortability control

Benchmark `v0.5`, n=4000, seed 7. Varsortability (Reisach et al.): 0.5 = the causal order is NOT readable from marginal variances; toward 1.0 = it is (and the trivial `sortnregress` baseline would win).

- mean **varsortability 0.58**
- `sortnregress` baseline: mean **F1 0.29**, mean directed SHD 9.03

**Verdict: NOT gamed by the variance trick — varsortability near chance and sortnregress is weak.**

Reproduce: `uv run python evals/varsortability/run.py`.
