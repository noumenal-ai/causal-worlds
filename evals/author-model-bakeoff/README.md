# Author-model bake-off

Decided with numbers, not assertion. Package `v0.2.0`, seed 7, judge `gemini-2.5-flash`, grader `interventional-ci@1`.

| author model | admit | difficulty | faithfulness | directed SHD | F1 | attempts |
|---|---|---|---|---|---|---|
| `claude-opus-4-8` **(winner)** | 8/8 | 0.30 | 0.94 | 1.25 | 0.92 | 1.00 |
| `claude-sonnet-4-6` | 8/8 | 0.25 | 0.98 | 1.75 | 0.90 | 1.25 |

**Winner: `claude-opus-4-8`** — ranked on admit-rate, then anti-cliché difficulty, then faithfulness, then fewer re-author attempts.

Reproduce: `set -a && . ../.env && set +a && uv run python evals/run_author_bakeoff.py`.
