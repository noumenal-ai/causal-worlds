# causal-worlds benchmark — v0.2

12/12 prompts authored, gated, and admitted. Author `claude-opus-4-8`, independent judge `gemini-2.5-flash`, grader `interventional-ci@1`, seed 7, 2000 rows/world. Package `v0.2.0`.

Each world is a self-describing bundle: `spec.json` (generative truth), `data.npz` (observed time-series), `answer_key.json` (ground-truth edges + confounded pairs), `manifest.json` (provenance + grade). **Fictional** — not a model of any real system.

Aggregate over the 12 admitted worlds: mean anti-cliché difficulty **0.28**, mean faithfulness **1.00**, mean reference-grader directed SHD **1.25**, mean F1 **0.92**.

| world | operation | difficulty | faithfulness | SHD | F1 |
|---|---|---|---|---|---|
| `world_00` | A regional coffee chain with weekend demand swings, lo | 0.22 | 1.00 | 2 | 0.88 |
| `world_01` | A hospital emergency department: triage staffing, pati | 0.29 | 1.00 | 1 | 0.91 |
| `world_02` | A ride-hailing marketplace with surge pricing, driver  | 0.33 | 0.95 | 1 | 0.95 |
| `world_03` | A solar microgrid with battery storage, dynamic tariff | 0.37 | 1.00 | 1 | 0.94 |
| `world_04` | A contract manufacturing line: machine maintenance, th | 0.18 | 1.00 | 1 | 0.93 |
| `world_05` | A SaaS support desk: ticket inflow, staffing, automati | 0.44 | 1.00 | 1 | 0.92 |
| `world_06` | A last-mile delivery network: courier dispatch, traffi | 0.20 | 1.00 | 1 | 0.92 |
| `world_07` | A boutique hotel with dynamic room pricing, seasonal e | 0.26 | 1.00 | 1 | 0.96 |
| `world_08` | A grocery cold chain: refrigeration setpoints, energy  | 0.26 | 1.00 | 1 | 0.95 |
| `world_09` | A call-center workforce plan: shift scheduling, call v | 0.33 | 1.00 | 1 | 0.91 |
| `world_10` | A bike-share network: station rebalancing, dynamic pri | 0.29 | 1.00 | 2 | 0.86 |
| `world_11` | A wastewater treatment plant: aeration control, inflow | 0.14 | 1.00 | 2 | 0.88 |

*difficulty* = `1 - F1(judge_prior, truth)` (higher ⇒ harder to guess from priors); *SHD/F1* are the reference interventional-CI grader vs the answer-key (the floor any discoverer should beat).

Reproduce:
```bash
set -a && . ../.env && set +a
CAUSAL_WORLDS_DISCOVERER_N=6000 causal-worlds benchmark benchmark/prompts.txt benchmark/v0.2 --seed 7
```
