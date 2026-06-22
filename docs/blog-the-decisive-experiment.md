# A causal benchmark that defeats the standard toolbox (and how we measured it)

*Draft technical post — Noumenal / causal-worlds, v0.3.0*

## The problem: most LLM causal benchmarks are guessable

Recent work keeps finding the same flaw in benchmarks used to claim LLMs "do causal reasoning":
the test cases are likely in the pretraining corpus, and even when they aren't, the *answer is
guessable from the variable names alone* ("does smoking cause cancer?"). A model can score well by
reciting priors — the "causal parrots" critique — without doing any discovery from data. A credible
causal-discovery benchmark has to be **leakage-resistant and not solvable by priors**.

## What we built

[causal-worlds](https://github.com/noumenal-ai/causal-worlds) generates fictional-but-coherent
*operations* from a plain-language description, each with a **declared ground-truth causal graph** as
an answer key. The pipeline:

1. **Author** (Claude) turns a sentence — *"a regional coffee chain with weekend swings and variable
   lead times"* — into a structural causal model: variables, roles, mechanisms, a **hidden
   confounder**, and a **regime that flips an effect's sign**.
2. **Independent judge** (Gemini — deliberately a different model family) guesses the graph *from
   names alone*. The gap between that prior guess and the truth is an **anti-cliché difficulty**
   score; worlds that are guessable from priors are rejected.
3. **Gates** admit only worlds that are valid, non-degenerate, recoverable, and not cliché.
4. **Reference grader** is an **interventional-CI** discoverer — it uses `do()` interventions, so it
   can tell *confounding* (a hidden common cause) from *causation*.

Because the structure is *declared*, every world is its own answer key. The worlds are fictional, so
no real-world data is needed and there's nothing to leak.

## The decisive experiment

The headline claim — *"a benchmark that defeats the standard toolbox"* — is only worth anything if
it's measured. So across all 12 worlds in the v0.2 set we ran the standard discoverers (PC, GES, FCI,
GIES via `causal-learn`/`gies`) against the reference interventional grader, three seeds each:

| method | mean skeleton-SHD ↓ | directed F1 ↑ | confounded-pair kept as causal ↓ (of 12) |
|---|---|---|---|
| **interventional-ci (reference)** | **1.31** | **0.91** | **0.33** |
| PC | 3.22 | 0.57 | 8.3 |
| FCI | 3.31 | 0.53 | 7.3 |
| GIES | 4.53 | 0.78 | 10.0 |

The standard methods **keep the hidden-confounded pair as a causal edge in 7–10 of 12 worlds** and
post 2–4× the structural error. The interventional grader almost never does. The trap — a hidden
common cause with no direct edge — is exactly what observational and score-based methods can't escape,
because they assume causal sufficiency. Interventions break the tie.

## What we *didn't* get (the honest part)

We also tested whether our anti-cliché *difficulty* score predicts how badly the standard methods do.
It doesn't yet (correlations 0.05–0.11). The reason is informative: our difficulty metric measures
*name-guessability* (can a judge guess the graph from variable names?), but the discovery hardness in
these worlds comes from the *structural* confounder+regime trap, which is present across the whole set
regardless of naming. Two different axes. The next release adds a structural-difficulty axis and pushes
name-difficulty higher (the current mean, 0.28, is lower than we want).

Twelve worlds is a demo, not a benchmark. The crossover is clear enough to justify scaling to 50–100+
worlds with controlled diversity — which is what comes next.

## Try it

```bash
uv add 'causal-worlds[llm]'
causal-worlds generate "a hospital ED with triage staffing and bed pressure" ./world
```

Everything here is reproducible: the benchmark set, the model bake-off that picked the author, and the
crossover table all ship as versioned artifacts in the repo.
