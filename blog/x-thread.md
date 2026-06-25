# X / Twitter — causal-worlds

Three threads, each conveying a *different* thing — post one per week, or pick the one that fits the
moment. Tweet bodies are ≤ 280 chars. `[attach: …]` lines are image attachments, not tweet text
(diagrams in `docs/figures/`, eval figures in `figures/`; raw-GitHub URLs work). Every number is
copy-pasted from a real run.

- **Thread 1 — The idea.** Why a *declared* world is the only honest causal benchmark. (the hook)
- **Thread 2 — The proof.** I ran every discovery method in the box; only one isn't fooled. (the rigor)
- **Thread 3 — The breadth.** See, do, imagine, *and act* — most benchmarks stop at "see." (the scope)

═══════════════════════════════════════════════════════════════════════
## Thread 1 — The idea: "a benchmark that knows the right answer"
═══════════════════════════════════════════════════════════════════════

**1/**
In this coffee dataset, `overtime` and `sales` correlate 0.64. Looks like overtime drives sales.

Now don't *watch* overtime — *set* it. Force it high, force it low, measure sales.

Effect: 0.00.

The entire correlation was a hidden cause. 🧵

[attach: docs/figures/coffee_world.png]

**2/**
The problem with causal benchmarks: to score who got the causality right, you have to know the right
answer. With real-world data, nobody does.

So we stopped using real data. Write a sentence → get a fictional world with a DECLARED ground-truth
graph. An answer key, by construction.

**3/**
`causal-worlds generate "a hospital ED with triage staffing and bed pressure" ./world`

An LLM authors the structural model. A *different* model (different family) judges it. Four gates admit
it only if it's executable, faithful, and not guessable.

Out comes a simulator + data + the truth.

**3b/**
Two modes:
• default = **benchmark**: rejects worlds you could guess from the variable names (the bar's the point)
• `--playground`: keep faithfulness + a difficulty score, but never reject — *describe any world and just get it*

The bundle stamps which one, so they never mix.

**4/**
Two consequences fall out of "fiction-first":

→ Nothing real to memorize. No data to leak. A model can't recite the answer from training.
→ The key is *derived from the spec*, so it can never disagree with the simulator.

That's what most benchmarks can't honestly promise.

**5/**
So on the coffee world, the benchmark KNOWS `overtime → sales` is a trap — a hidden "local buzz" (a
festival, good weather) drives both. It declared the trap.

Which means it can score, exactly, who walks into it.

MIT · `pip install causal-worlds`
🔗 github.com/noumenal-ai/causal-worlds

═══════════════════════════════════════════════════════════════════════
## Thread 2 — The proof: "I ran every method in the box"
═══════════════════════════════════════════════════════════════════════

**1/**
I took a causal world whose true graph I already had, and ran every discovery method that ships in the
box against it. Same data, same seed.

One column tells the whole story: how many spurious "confounded" pairs each method keeps as real causal
edges. 🧵

**2/**
The world hides a confounder: `overtime` and `sales` move together but have NO edge between them — a
latent cause drives both.

The test: can a method tell confounding (X←L→Y) from causation (X→Y)?

**3/**
```
method               F1    confounded_kept
interventional-ci   1.00        0    ◀ latent-aware reference
pc                  0.77        1
fci                 0.77        1
gies                0.77        1
dagma               0.17        1
directlingam        0.50        0
```
Only the reference keeps 0 AND nails the structure. (DirectLiNGAM drops the phantom but mangles
everything else — F1 0.50.)

[attach: blog/figures/tour1_shootout.png]

**4/**
The dividing line isn't horsepower, and it isn't even interventions. Across the 26-world hardened set,
PC *with the same interventions* as the reference still keeps the confounded pair ~30 times.

It's whether the method knows hidden confounders can exist. (ΔF1 +0.37, CI excl. 0.)

[attach: figures/fig1_confounded_kept.png]

**5/**
Honest framing: this is a textbook identifiability result (Ψ-FCI, GIES), and our reference is a
deliberately simple discoverer the benchmark is built to reward.

We don't contribute the theorem. We contribute the apparatus that re-surfaces it in a minute.

**6/**
Swap in your own method — implement one function, `recover(substrate, seed) -> set[(src,dst)]` — and
get an apples-to-apples score against ground truth.

Then try to beat the reference's 0. That's the game.

MIT · `pip install causal-worlds`
🔗 github.com/noumenal-ai/causal-worlds

═══════════════════════════════════════════════════════════════════════
## Thread 3 — The breadth: "most causal benchmarks stop at 'see'"
═══════════════════════════════════════════════════════════════════════

**1/**
Most causal benchmarks test one thing: can you SEE the structure in the data?

But causality has three rungs, and agents have to ACT. causal-worlds lets you do all of it — on a world
whose answer you already hold. 🧵

**2/**
SEE → DO. `do(footfall)` is graph surgery: cut every arrow INTO the variable, keep every arrow OUT.

The data slopes sales on footfall at 1.56. The *true* causal effect is 0.90 — the data overstated it by
the hidden confounder. The intervention corrects it.

[attach: docs/figures/coffee_do_footfall.png]

**3/**
DO → IMAGINE. "We sold what we sold — what would sales have been if footfall were higher, *that same
day*?"

Exact, because the model is declared (abduction → action → prediction):
factual 3.24 → counterfactual 4.55. On cross-sectional AND temporal worlds.

**4/**
IMAGINE → ACT. The same worlds are a CONTROL benchmark. The mechanisms are known, so the optimal policy
is computable — you're graded by regret, no learned reward, no real data.

`optimal_policy → {price: 0.0}` · your overshoot → regret 4.5 · the optimum → 0.0

**5/**
The real test: regret UNDER PERTURBATION. The coffee world's `price` lever flips sign across regimes
(optimum −1 one regime, +1 the other).

A regime-blind policy: 0 regret in its regime, 2.0 when it flips. A regime-aware one stays ~0. That gap
is your "stay-optimal" score.

[attach: blog/figures/tour2_regime_flip.png]

**6/**
A `Gymnasium` env shifts the regime under your agent, step to step — drop in any RL/decision agent and
watch.

Temporal worlds add lagged edges + a lagged answer key, graded vs PCMCI+/LPCMCI/VARLiNGAM/Granger.

See, do, imagine, act. MIT · `pip install causal-worlds`
🔗 github.com/noumenal-ai/causal-worlds
