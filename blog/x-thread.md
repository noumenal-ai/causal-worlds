# X thread — causal-worlds launch

Each tweet is ≤ 280 characters. Figure-attachment notes are in [brackets] and are NOT part of the tweet
text (don't count them toward the limit). Post the figures as native image attachments on the noted
tweets.

---

**1/**
You can't trust a causal-discovery benchmark you can memorize, or whose data leaks the answer.

So we built one you (mostly) can't — then spent weeks trying to break it ourselves.

Introducing causal-worlds: `pip install causal-worlds` 🧵

[attach: figures/fig1_confounded_kept.png]

---

**2/**
The idea: write a sentence —

"a coffee chain with weekend swings and variable lead times"

— and get a fictional-but-coherent causal world with a DECLARED ground-truth graph. An answer key by construction.

Fiction-first → nothing real to memorize, no data to leak.

---

**3/**
The structure is declared, not learned, so the answer key and the simulator can't disagree.

An author (Claude) proposes the world. An independent judge (Gemini, different model family) + statistical gates admit only worlds that are valid, faithful, and not guessable.

---

**4/**
The decisive finding. Every world hides a confounder L → can a method tell X←L→Y from X→Y?

Confounded pairs kept as causal (summed over 26 worlds, ~1-2 per world):
latent-aware rule: 0
PC + the SAME interventions: 30 — no better than observational PC's 29.

[attach: figures/fig1_confounded_kept.png]

---

**5/**
Read it precisely: the lever is LATENT-AWARENESS, not interventions — and NOT "we beat the toolbox."

It's a textbook identifiability result (Ψ-FCI, GIES). Our reference is a deliberately simple known discoverer the benchmark rewards. The contribution is the apparatus.

---

**6/**
The credible part: we audited our own benchmark and found 3 real flaws.

(1) Circular admission — worlds were admitted by the grader that later "won." Fixed: admission is now grader-INDEPENDENT, faithfulness checked in closed form, no discovery run.

[attach: figures/fig2_anti_cliche.png]

---

**7/**
(2) Name-guessability. A data-free name-only LLM guess scored F1 0.71 on our first set — a parrot, not discovery.

Now: named 0.38, name-blind 0.46 (adversarial names MISLEAD), name+role-blind ~0.01 — structure isn't guessable. Residual role-type prior: disclosed.

[attach: figures/fig2_anti_cliche.png]

---

**8/**
(3) Simulated-DAG leakage. Unstandardized, a trivial "sort by variance" baseline hit F1 0.74 — beating PC/FCI. The answer leaked into the data.

Fix: internal standardization (iSCM). varsortability 0.94→0.54; R²-sortability 0.73→0.60. Residual 0.60: disclosed, not buried.

[attach: figures/fig3_leakage.png]

---

**9/**
We also draw our caveats instead of hiding them.

"Difficulty" of a world is a DESCRIPTIVE axis, not a validated predictor of error — the correlation CIs include 0. We report it with its uncertainty.

[attach: figures/fig4_difficulty.png]

---

**10/**
There's more, with honesty labels attached:

• Temporal worlds (lagged answer key, PCMCI+ recovers exactly) — but n=1 so far.
• A control track: by-construction optimal policy, regret, regret-under-perturbation, and a Gymnasium env where the regime shifts under you.

---

**11/**
Engine + grading run with NO API key:

```
grade_spec(worlds.get("coffee"), InterventionalCiDiscoverer())
# directed_shd=0  f1=1.0  confounded_reported=0
```

Swap in YOUR discoverer (one method: recover→edges) to score it against a known truth.

---

**12/**
MIT. Live on PyPI. Honest measurement is the whole identity.

The best thing you can do: try to break a world without doing real discovery. That's how we got here.

🔗 github.com/noumenal-ai/causal-worlds
`pip install causal-worlds`
