# causal-worlds benchmark — v0.5 (scaled)

35/36 prompts authored, gated, and admitted across an **easy→hard complexity spread** (the author's structural-complexity knob). Author `claude-opus-4-8`, judge `gemini-2.5-flash`, grader `interventional-ci@1`, seed 7, 2000 rows/world. Package `v0.4.0`. **Fictional** — not real systems.

Built to give the difficulty-vs-error analysis real range (v0.4 was underpowered at n=12). Per-complexity structure and reference-grader recovery:

| complexity | n | mean structural difficulty | mean name-difficulty | grader SHD | grader F1 |
|---|---|---|---|---|---|
| easy | 11 | 0.0 | 0.23 | 0.36 | 0.98 |
| standard | 12 | 1.4 | 0.28 | 1.75 | 0.90 |
| hard | 12 | 3.0 | 0.33 | 2.33 | 0.85 |

Aggregate (35 worlds): mean structural difficulty **1.51**, name-difficulty **0.28**, faithfulness **0.97**.

Each world is a self-describing bundle (`spec.json` / `data.npz` / `answer_key.json` / `manifest.json` carrying full provenance incl. structural difficulty + complexity). The baseline crossover on this set: [`evals/baseline-crossover/v0.5/`](../../evals/baseline-crossover/v0.5/); the difficulty analysis: [`evals/structural-difficulty/v0.5/`](../../evals/structural-difficulty/v0.5/).

Reproduce: `set -a && . ../.env && set +a && uv run python evals/scale/generate_set.py`.
