# Examples

Runnable scripts. The first two need **no API key**; the third needs the `[llm]` extra + provider keys.

| script | needs | shows |
|---|---|---|
| [`01_grade_your_discoverer.py`](01_grade_your_discoverer.py) | base install | implement a `Discoverer` and score it against a world's answer key |
| [`02_inspect_a_bundle.py`](02_inspect_a_bundle.py) | base install | load a shipped benchmark world and read its truth + provenance |
| [`03_author_a_world.py`](03_author_a_world.py) | `[llm]` + keys | author a world from a sentence with Claude, gated by the Gemini judge |

```bash
uv run python examples/01_grade_your_discoverer.py
uv run python examples/02_inspect_a_bundle.py
set -a && . ../.env && set +a && uv run python examples/03_author_a_world.py   # keys
```
