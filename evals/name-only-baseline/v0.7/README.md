# Name-only baseline (anti-cliché certificate)

Benchmark `v0.7`, judge `gemini-2.5-flash`. Directed F1 of a prior-only LLM guess (no data) vs the truth, against the random-graph chance floor, at three disclosure levels (Caliper-style). If a level beats `null`, the answer leaks at that level: **named** = names + roles; **name-blind** = names anonymized to `X1..Xn`, roles kept; **name+role-blind** = names anonymized AND roles hidden (should sit at chance).

- **named** F1 0.41  [0.35, 0.46]
- **name-blind** F1 0.43  [0.40, 0.46]
- **name+role-blind** F1 0.00  [0.00, 0.00]
- **null** (chance) F1 0.16  [0.15, 0.17]

**Verdict: LEAKY — priors beat chance; worlds are guessable from names and/or roles.**

Reproduce: `uv run python evals/name-only-baseline/run.py benchmark/v0.5` (needs the `llm` extra + a Gemini key).
