# Temporal crossover

Time-series discovery on the temporal built-in `supply` (n=2000, seed 7, 7 true lagged edges). `supply` hides a logistics confounder L driving lead time ~ cost, on top of autoregressive lead time + inventory. Latent-naive methods (PCMCI+, VARLiNGAM, Granger) assume no hidden confounders; LPCMCI is latent-aware.

| method | temporal SHD ↓ | temporal F1 ↑ | recovered | kept confounded pair as causal? |
|---|---|---|---|---|
| `pcmci+` | 0 | 1.0 | 7 | False |
| `lpcmci` | 3 | 0.727 | 4 | False |
| `varlingam` | 7 | 0.6 | 13 | True |
| `granger` | 18 | 0.182 | 15 | False |

## Verdict

**Temporal grading is validated end-to-end:** PCMCI+ recovers `supply`'s lagged structure *exactly*
(temporal SHD 0, F1 1.0), which confirms the lagged answer-key + the temporal scoring are correct.

**The confounder trap is method-specific in the temporal setting.** Unlike the cross-sectional
crossover — where the *whole* standard toolbox falls for the hidden confounder — here only **VARLiNGAM**
keeps the spurious `leadtime~cost` edge; PCMCI+, LPCMCI, and Granger do not. PCMCI+'s momentary-CI
search plus the autoregressive structure evidently screens off the latent here. (Granger over-recovers
spurious lagged links — F1 0.18 — as expected; it tests prediction, not structure.)

**Honest caveat:** this is **n=1**. The cross-sectional crossover earned its claim at n=35; the
temporal claim needs a temporal benchmark *set*. Next: teach the author to emit lagged worlds + add
temporal gates, then scale and re-run this crossover.

Reproduce: `uv run python evals/temporal-crossover/run.py` (needs the `temporal` extra).
