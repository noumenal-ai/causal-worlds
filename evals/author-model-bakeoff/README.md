# Author-model bake-off

Decided with numbers, not assertion. Package `v0.2.0`, seed 7, judge `gemini-2.5-flash`, grader `interventional-ci@1`.

| author model | admit | difficulty | faithfulness | directed SHD | F1 | attempts |
|---|---|---|---|---|---|---|
| `claude-opus-4-8` **(winner)** | 8/8 | 0.30 | 0.94 | 1.25 | 0.92 | 1.00 |
| `claude-sonnet-4-6` | 8/8 | 0.25 | 0.98 | 1.75 | 0.90 | 1.25 |

**Winner: `claude-opus-4-8`** — ranked on admit-rate, then anti-cliché difficulty, then faithfulness, then fewer re-author attempts.

> ⚠️ **Stale admit-rate (2026-06-25, #19).** These 8/8 admit rates were measured on **package `v0.2.0`**, whose
> admission gate did **not** yet include the strict **T4 anti-cliché** control (added v0.19). Note the **mean
> difficulty 0.30** here ⇒ a name+role prior recovered ≈ 0.70 of the edges — well **above** today's `difficulty ≥
> 0.5` admission bar. *In other words, every world in this bake-off would be rejected by the current gate.* The
> bake-off still decides the *author-model* question (Opus vs Sonnet on faithfulness/SHD/F1), but it is **not**
> evidence of the author pass-rate under today's strict gate — that remains the open risk (`docs/scope.md §8`), and
> is why **playground mode** (`--playground`) exists. A re-run under the current gate is pending.

Reproduce: `set -a && . ../.env && set +a && uv run python evals/run_author_bakeoff.py`.
