# A causal benchmark that defeats the standard toolbox (and how we measured it)

*Draft technical post — Noumenal / causal-worlds, through v0.6*

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

## The experiment — and what it actually shows

We ran the standard discoverers (PC, GES, FCI, GIES) against the reference interventional grader on
every world, three seeds each, on the 35-world set:

| method | interventions? | latent-aware? | skeleton-SHD ↓ | confounded pair kept as causal ↓ (of 35) |
|---|---|---|---|---|
| **interventional-ci (reference)** | yes | yes | **1.47** | **0** |
| GIES | yes | no | 2.37 | 17 |
| PC | no | no | 2.81 | 13 |
| FCI | no | partly | 2.68 | 8 |

It's tempting to call this "defeating the standard toolbox," but the honest reading is narrower and
more interesting. **GIES gets the same interventional data as the reference** and recovers the skeleton
about as well — yet it still reports the hidden-confounded pair as a *causal* edge in most worlds,
because it assumes causal sufficiency (no hidden confounders). PC/FCI, on observational data, do the
same. Only the latent-aware interventional rule keeps confounded-kept at zero. So the dividing line is
**latent-awareness**, not interventions per se — and the result is best stated as an *identifiability*
finding: you need both interventions and a latent-aware method to tell confounding from causation.

(One caveat we keep in the open: these worlds are currently admitted by the reference grader itself, so
the benchmark and the headline aren't yet fully decoupled. Closing that loop — plus standard
synthetic-DAG controls — is the next milestone.)

## The difficulty story (a useful wrong turn)

We wanted a *difficulty* score that predicts how badly the standard tools fail. Our first one measured
**name-guessability** — can the judge guess the graph from variable names alone? At 12 worlds it
predicted nothing (correlation ~0.1). Rather than bury that, we shipped it as an honest negative — and
it pointed at the real answer: the discovery hardness isn't in the *names*, it's in the *structure*
(hidden confounders + sign-flipping regimes).

So we added a **structural-difficulty** score and a complexity dial on the author (easy → no traps;
hard → several), generated a 36-world set spanning the range, and re-ran. The signal appeared:
**structural difficulty predicts the observational collapse (correlation +0.62)** where
name-guessability still doesn't (+0.14). Difficulty became a real instrument once we measured the
right thing. The honest negative at n=12 was the fastest route to the positive result at n=35.

## Try it

```bash
uv add 'causal-worlds[llm]'
causal-worlds generate "a hospital ED with triage staffing and bed pressure" ./world
```

Everything here is reproducible: the benchmark set, the model bake-off that picked the author, and the
crossover table all ship as versioned artifacts in the repo.

## What's next

The worlds are tabular today. The clear next frontier is **genuinely temporal** worlds — lags,
seasonality, regime dynamics over time — graded with time-series discovery methods (PCMCI+,
VARLiNGAM). That's the slice no public tool occupies, and it's where this becomes a benchmark you
can't get anywhere else.
