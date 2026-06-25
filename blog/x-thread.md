# X / Twitter — causal-worlds (multiple takes)

Several independent angles — post whichever fits the moment; no need to use just one. Takes A–D are
threads with each tweet ≤ 280 chars. **Take E is a single long-form post (needs X Premium) — no length
limit.** `[attach: …]` notes are image attachments, not tweet text. Diagrams: `docs/figures/`; eval
figures: `figures/` (raw-GitHub URLs work).

═══════════════════════════════════════════════════════════════════════
## Take A — "Correlation lies" (the mirage; the default)
═══════════════════════════════════════════════════════════════════════

**1/**
In a coffee dataset, overtime and sales correlate 0.64. Looks like overtime drives sales.

Force overtime — *do* it, don't watch it — and sales moves 0.00.

The whole link was a hidden cause. We built a benchmark that KNOWS, so it can score who gets fooled. 🧵

[attach: docs/figures/coffee_world.png]

**2/**
The idea: write a sentence —

"a coffee chain with weekend swings and variable lead times"

— and get a fictional-but-coherent causal world with a DECLARED ground-truth graph. An answer key, by construction.

Fiction-first → nothing real to memorize, no data to leak.

**3/**
Causality has 3 rungs (Pearl's ladder). causal-worlds lets you stand on all three, on a world whose answer you already know:

SEE: overtime ~ sales = 0.64
DO: do(overtime) → 0.00 (mirage gone)
IMAGINE: the counterfactual, "what would have happened"

**4/**
"Doing" is graph surgery. do(footfall) cuts every arrow INTO footfall — its causes, incl. the hidden confounder, no longer apply — and keeps every arrow OUT.

That's how you escape confounding. Verified genuine surgery, not statistical conditioning.

[attach: docs/figures/coffee_do_footfall.png]

**5/**
The decisive finding. Every world hides a confounder L → can a method tell X←L→Y from X→Y?

Given the SAME interventions as a latent-aware reference:
PC + interventions keeps the confounded pair as causal (30, vs observational PC's 29).
The latent-aware rule: 0.

[attach: figures/fig1_confounded_kept.png]

**6/**
Read it precisely: the lever is LATENT-AWARENESS, not interventions — and NOT "we beat the toolbox."

A textbook identifiability result (Ψ-FCI, GIES). Our reference is a deliberately simple known discoverer the benchmark rewards. The contribution is the apparatus.

**7/**
The credible part: we audited our OWN benchmark and found 3 real flaws.

(1) Circular admission — worlds were admitted by the grader that later "won." Fixed: admission is grader-INDEPENDENT, faithfulness in closed form, no discovery run.

[attach: figures/fig2_anti_cliche.png]

**8/**
(2) Name-guessability. A data-free, name-only LLM guess scored F1 0.71 on our first set — a parrot.

Now: named 0.38, name+role-blind ~0.01. Structure isn't guessable once you strip the words. Residual role-prior: disclosed.

**9/**
(3) Simulated-DAG leakage. Unstandardized, a trivial "sort by variance" baseline hit F1 0.74 — beating PC/FCI. The answer leaked into the data.

Fix: internal standardization (iSCM). varsortability 0.94→0.54; R²-sortability 0.73→0.60. Residual disclosed.

**10/**
Counterfactuals too: abduction → action → prediction, on cross-sectional AND temporal worlds — cross-checked vs the simulator via Pearl's law that counterfactuals average to interventions.

Engine, grading, and the diagrams run with NO API key.

**11/**
MIT. Live on PyPI. Honest measurement is the whole identity.

Best thing you can do: try to break a world without doing real discovery. That's how we got here.

📖 Full story: selftaughtamit.medium.com/correlation-lies-we-built-a-causal-world-that-can-prove-it-then-tried-to-break-it-ourselves-30efae26b457
🔗 github.com/noumenal-ai/causal-worlds · `pip install causal-worlds`

═══════════════════════════════════════════════════════════════════════
## Take B — "Stop letting models cheat off variable names" (LLM-eval angle)
═══════════════════════════════════════════════════════════════════════

**1/**
Most "can LLMs reason about causality?" benchmarks have a dirty secret: the model recites the graph from variable names and never touches the data.

We measured it. A name-only, data-free guess scored F1 0.71 on our first benchmark. A parrot. 🧵

**2/**
So we rebuilt it to be uncheatable. causal-worlds generates fictional-but-coherent operations with a DECLARED causal answer key.

Fiction-first → no real system to recite, no data to leak. The only way to score is to actually discover.

**3/**
Then we made the names LIE. An "adversarial" author writes worlds where the obvious name-based guess is WRONG — phantom edges, reversed edges, a misdirecting mediator — while every real edge stays statistically detectable.

**4/**
Three-tier certificate of progressive blinding:
named → 0.38
name-blind → 0.46 (the adversarial names actively mislead)
name + role-blind → ~0.01 (chance floor)

The structure isn't guessable once you strip the words.

[attach: figures/fig2_anti_cliche.png]

**5/**
We don't pretend it's perfect: with roles visible, a guesser still beats chance — an intrinsic "controllable→outcome" prior you can't delete without deleting the operation. We disclose it, not hide it.

**6/**
And it's more than a discovery test: do() interventions + counterfactuals on a known SCM. Seeing, doing, imagining — all three rungs of causality, with an answer key.

MIT, `pip install causal-worlds`
🔗 github.com/noumenal-ai/causal-worlds

═══════════════════════════════════════════════════════════════════════
## Take C — "Causality in one picture" (visual, short)
═══════════════════════════════════════════════════════════════════════

**1/**
Causality in one picture: a coffee shop's *declared* causal graph. The dashed-red node is a hidden confounder the data never shows you — the reason discovery is hard.

You can SEE it, intervene on it, and ask counterfactuals — with a known answer key. 🧵

[attach: docs/figures/coffee_world.png]

**2/**
SEE: overtime & sales correlate 0.64.
DO: force overtime → sales moves 0.00.
The link was the hidden confounder the whole time. The benchmark knows, because it declared it.

**3/**
DO is graph surgery — cut arrows INTO the variable, keep the arrows OUT. Verified genuine surgery, not statistical conditioning.

[attach: docs/figures/coffee_do_footfall.png]

**4/**
IMAGINE: "what would sales have been had footfall been higher, that same day?" Exact, because the world is declared.

MIT, no API key for the engine.
`pip install causal-worlds` · github.com/noumenal-ai/causal-worlds

═══════════════════════════════════════════════════════════════════════
## Take D — standalone single tweets (pick one)
═══════════════════════════════════════════════════════════════════════

**D1 (the mirage):**
overtime and sales correlate 0.64 in this coffee dataset. Force overtime → sales moves 0.00.

The link was a hidden confounder. We built a benchmark that knows the truth, so it can score who gets fooled.

`pip install causal-worlds` 🧵👇
[attach: docs/figures/coffee_world.png]

**D2 (the parrot):**
Most causal benchmarks can be solved by reciting variable names — F1 0.71, no data needed.

We built one where the names LIE and the structure is unguessable, so the only way to win is to actually discover.

MIT: github.com/noumenal-ai/causal-worlds

**D3 (the ladder):**
A causal world you can SEE (associations), DO (interventions / graph surgery), and IMAGINE (counterfactuals) — generated from a sentence, with a declared ground-truth answer key.

All three rungs of Pearl's ladder. MIT.
`pip install causal-worlds`

═══════════════════════════════════════════════════════════════════════
## Take E — long-form single post (X Premium; no 280 limit)
═══════════════════════════════════════════════════════════════════════
[attach: docs/figures/coffee_world.png]

Correlation lies — and most causal benchmarks can't prove it, because they don't know the truth either.

Here's a coffee-shop dataset. `overtime` and `sales` move together: correlation 0.64. Looks like overtime drives sales.

Now *force* overtime instead of watching it — do() it high and low, measure sales. Effect: 0.00.

There's no causal link. A hidden "local buzz" (a festival, good weather) was lifting foot traffic, overtime, and sales all at once. The correlation was a mirage.

We built causal-worlds (MIT, `pip install causal-worlds`) so you can prove things like this on demand:

→ Write a sentence — "a coffee chain with weekend swings and variable lead times" — and get a fictional-but-coherent world with a DECLARED ground-truth causal graph. An answer key, by construction.

→ Fiction-first: nothing real to memorize, no data to leak. The only way to score is to actually discover.

→ All three rungs of Pearl's ladder: SEE (associations), DO (interventions — genuine graph surgery), IMAGINE (counterfactuals via abduction → action → prediction). On a world whose answer you already hold.

And we audited our own benchmark, hard:
• A name-only LLM guess scored F1 0.71 on v1 — a parrot reciting variable names. We made the names mislead; strip them and the structure is unguessable (≈ chance).
• A "sort by variance" baseline beat real algorithms (F1 0.74) because the simulator leaked the causal order. Fixed with internal standardization.
• Admission was circular (the grader that won also admitted worlds). Now grader-independent, closed-form, no discovery run.

The headline is an identifiability result, stated honestly: given the SAME interventions, PC keeps a hidden-confounded pair as causal (≈30 worlds) just like observational PC (29); only a latent-aware rule reaches 0. The dividing line is latent-awareness, not interventions — a textbook fact, surfaced as a one-minute reproducible crossover, not "we beat the toolbox."

Honest measurement is the whole identity. Come break a world.

📖 Full story: selftaughtamit.medium.com/correlation-lies-we-built-a-causal-world-that-can-prove-it-then-tried-to-break-it-ourselves-30efae26b457
🔗 github.com/noumenal-ai/causal-worlds

═══════════════════════════════════════════════════════════════════════
## Take F — for RL / control people ("agents have to ACT, not just see")
═══════════════════════════════════════════════════════════════════════

**1/**
Causal benchmarks test what you can SEE. But decision agents have to ACT.

So causal-worlds is also a control benchmark: same worlds, but now you choose lever values to hit an objective — and we can score you against the *true* optimum. 🧵

**2/**
Because the mechanisms are DECLARED, the best the levers can possibly do is computable — a by-construction optimal policy.

So a controller is graded by REGRET against it. No learned reward model, no real data, no human labels.

**3/**
The real test is regret UNDER PERTURBATION. The regime flips (a weekend, a shock) and a policy that ignores it collapses — while the regime-aware optimum stays ~0.

A Gymnasium env shifts the regime under you, step to step. Cumulative regret = your score.

**4/**
Plug in your RL / decision agent, drive a world that changes beneath it, and measure whether it stays optimal — against a known answer.

MIT, `pip install causal-worlds`
🔗 github.com/noumenal-ai/causal-worlds

═══════════════════════════════════════════════════════════════════════
## Take G — for time-series people ("real operations aren't i.i.d.")
═══════════════════════════════════════════════════════════════════════

**1/**
Most causal-discovery benchmarks are cross-sectional — i.i.d. rows. Real operations are time series.

causal-worlds builds TEMPORAL worlds too: lagged edges, autoregression, and a lagged ground-truth answer key. 🧵

**2/**
Graded against the time-series toolbox — PCMCI+, LPCMCI, VARLiNGAM, Granger — so you can see which method recovers which variable drives which, and at which lag.

**3/**
And counterfactuals roll forward in time: hold the realized noise fixed, change one lever, replay the whole trajectory. Exact, because the world is declared.

MIT. `pip install causal-worlds`
🔗 github.com/noumenal-ai/causal-worlds
