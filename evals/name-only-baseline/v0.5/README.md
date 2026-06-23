# Name-only baseline (anti-cliché control)

Benchmark `v0.5`, judge `gemini-2.5-flash`. Directed F1 of a name-only LLM guess (no data) vs the truth, against the random-graph chance floor. **anon** repeats the guess after anonymizing names to `X1..Xn` (Caliper-style): if the benchmark were leaking through names, anonymizing would drop the score toward `null`.

- **named** F1 0.71  [0.68, 0.74]
- **anon**  F1 0.61  [0.58, 0.65]
- **null** (chance) F1 0.20  [0.19, 0.21]

**Verdict: LEAKY — the name-only baseline beats chance; some worlds are guessable from names.**

Reproduce: `uv run python evals/name-only-baseline/run.py benchmark/v0.5` (needs the `llm` extra + a Gemini key).
