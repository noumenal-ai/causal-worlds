# Simulated-DAG leakage controls (varsortability + R^2-sortability)

Benchmark `v0.5`, n=4000, seed 7. Each sortability is 0.5 when the causal order is NOT readable off that signal and trends to 1.0 when it is (and the matching trivial baseline would win). Varsortability (marginal variance) is removable by standardization; **R^2-sortability (predictability) is scale-invariant and is NOT** — both are reported per the Reisach et al. line of critique.

- mean **varsortability 0.54** — `sortnregress` mean **F1 0.33** (SHD 8.66)
- mean **R^2-sortability 0.60** — `R^2-sortnregress` mean **F1 0.37** (SHD 8.26)

**Verdict: NOT gamed by either sorting trick — both sortabilities near chance, both baselines weak.**

Reproduce: `uv run python evals/varsortability/run.py`.
