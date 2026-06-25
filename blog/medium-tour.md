# One sentence in, a causal universe out — a hands-on tour of `causal-worlds`

*The [launch post](https://selftaughtamit.medium.com/correlation-lies-...-30efae26b457) made the
case: correlation lies, and because the world is **declared**, you can prove it. This is the field
guide. I sat down with [`causal-worlds`](https://github.com/noumenal-ai/causal-worlds) (MIT, `pip
install causal-worlds`) for an afternoon and ran every feature it has — describe a world in a
sentence, get an executable simulator with a ground-truth answer key, then grade discovery, do
interventions, ask counterfactuals, and benchmark control under a shifting regime. **Every number and transcript below is copy-pasted straight
from my terminal — including the live authoring run, gate rejection and all.***

---

## 1. It starts with a sentence

The whole premise: you write a plain-language description of an *operation*, and you get back a
**fictional-but-coherent causal world** — an executable simulator, the time-series it emits, and the
**declared ground-truth causal graph that generated it**. An answer key, by construction.

```bash
causal-worlds generate "a regional power grid with rooftop solar, home batteries, and time-of-use pricing" ./grid
```

An **author** model (Claude) writes the structural causal model; an **independent judge** (Gemini — a
*different model family*, the standard self-preference mitigation) scores faithfulness and tries to
guess the graph from names alone. And here's the honest part — I ran exactly that command, and it
**refused**:

```text
not admitted: not admitted after 3 attempt(s): T4 cliché: names+roles recover it (prior F1 0.84 >= 0.5)
hint: that is the benchmark anti-cliché gate (the world was guessable from its names/roles). Re-run
with --playground to author it anyway — guessability then becomes an advisory difficulty score.
```

That's a *feature.* By default the package builds **benchmark-grade** worlds, and a benchmark you can
solve by reading the variable names ("solar → battery → grid") isn't a benchmark. So the gate rejects
worlds a name-guesser could crack — which, for intuitive operations, is most of them. (This is the same
strictness the [launch post](https://github.com/noumenal-ai/causal-worlds) earned its credibility on.)

But if you just want to **describe a world and get it** — the playground use case — pass `--playground`
(new in v0.34). The faithfulness check stays, the difficulty score is still reported (now advisory), but
guessability no longer rejects. Same prompt, and now a world lands on disk:

```text
admitted -> ./grid  difficulty=0.16 (advisory)
```

```python
from causal_worlds import load_bundle
b = load_bundle("./grid")
print(b.manifest["variables"])         # ['tou_price', 'battery_dispatch', 'cloud_cover',
                                       #  'peak_demand_event', 'solar_output', 'home_load', 'grid_import']
print(b.manifest["anti_cliche"])       # False  ← honestly stamped: playground, not benchmark-grade
# answer_key.json:  8 causal edges + 1 hidden-confounded pair (home_load ~ solar_output)
# reference grader on it:  directed_shd=1  f1=0.93  confounded_reported=0
```

A whole declared causal world — levers (`tou_price`), a hidden confounder behind `home_load ~
solar_output`, an outcome (`grid_import`) — from one sentence, in one attempt. And because the
manifest stamps `anti_cliche: False`, a playground world can never quietly pass itself off as
benchmark-grade. Because the structure is *declared, not learned*, the key can never disagree with the
simulator; and because the worlds are *fiction-first*, **there is nothing real to memorize and no data
to leak.**

> Authoring is the one step that needs an API key. **Everything else in this tour runs with no key at
> all** — engine, grading, the graph renderers, interventions, counterfactuals, and control. So let's
> open the worlds that ship in the box (`coffee`, `ecommerce`, plus a temporal `supply`) and look inside.

---

## 2. What's actually in the box: the answer key

Load a world and the package hands you the truth — directed edges *with strengths*, the hidden
confounders, regime sign-flips, and each variable's role. One call renders it as a diagram that draws
natively on GitHub:

```python
from causal_worlds import worlds, to_mermaid
print(to_mermaid(worlds.get("coffee")))
```

```text
local_buzz -.->|"0.8"| footfall       # hidden confounder (dashed), never in the data
local_buzz -.->|"0.8"| overtime
footfall   -->|"0.3"| overtime
price      -->|"-1/1"| demand          # a regime SIGN-FLIP: price helps, then hurts
footfall   -->|"0.5"| demand
demand     -->|"1"|   sales
footfall   -->|"0.4"| sales
local_buzz -.->|"0.6"| sales

answer key: 6 causal edges, 1 hidden-confounded pair(s)
confounded (no direct edge, shared hidden cause): overtime ~ sales
```

Read that last line. `overtime` and `sales` move together in the data, but there is **no edge between
them** — a hidden `local_buzz` (a street festival, good weather, a nearby event) drives both. The
benchmark *knows* this is a trap, because we declared it. That is exactly what most causal benchmarks
can't honestly do.

There's a CLI for the same thing — `causal-worlds viz coffee` (add `--format dot` for a layered
Graphviz render with the hidden confounder's out-edges in dashed red).

---

## 3. Benchmarking discovery: one function, graded against truth

Here's the payoff. You implement one method — `recover(substrate, *, seed) -> set[(src, dst)]` — and
the package grades it against the answer key:

```python
from causal_worlds import worlds, grade_spec, InterventionalCiDiscoverer
print(grade_spec(worlds.get("coffee"), InterventionalCiDiscoverer()))
# Report(directed_shd=0, f1=1.0, n_truth=6, n_recovered=6, confounded_reported=0)
```

The `Report` is the eval output: **directed SHD** (edge-level structural error), **F1** on the directed
edges, and — the column that matters most here — **`confounded_reported`**: how many spurious
*confounded* pairs the method kept as if they were real causal edges. A method can have a respectable F1
and still get fooled by the trap.

So I ran the whole discovery toolbox that ships in the box against the coffee world, same data, same seed:

```python
from causal_worlds import worlds, grade_spec, BASELINES, InterventionalCiDiscoverer
spec = worlds.get("coffee")
for name, disc in {"interventional-ci": InterventionalCiDiscoverer, **BASELINES}.items():
    r = grade_spec(spec, disc(), seed=0)
    print(f"{name:>22}  F1={r.f1:.2f}  dirSHD={r.directed_shd}  confounded_kept={r.confounded_reported}")
```

```text
       method          |   F1  | dir.SHD | confounded_kept
-----------------------+-------+---------+----------------
 interventional-ci  ◀──│  1.00 |    0    |       0          ← the latent-aware reference
                    pc │  0.77 |    3    |       1
                   fci │  0.77 |    3    |       1
                  gies │  0.77 |    3    |       1
                 dagma │  0.17 |    8    |       1
          directlingam │  0.50 |    5    |       0
```

![The discovery shootout on the coffee world: only the latent-aware reference keeps zero confounded pairs and recovers the structure (F1 1.0); PC/FCI/GIES all keep the spurious edge.](https://raw.githubusercontent.com/noumenal-ai/causal-worlds/main/blog/figures/tour1_shootout.png)

Read the last column down. **Only the latent-aware reference keeps zero confounded edges *and* nails
the structure (F1 1.0).** PC, FCI, and the interventional GIES all keep the `overtime → sales` phantom.
DAGMA mangles the structure entirely. DirectLiNGAM happens to drop the phantom but butchers the rest
(F1 0.50). The dividing line — proven across the 26-world hardened benchmark in the launch post —
isn't raw power or even interventions; it's **whether the method knows hidden confounders can exist.**

That whole table is a benchmark you re-run in a minute. Swap any name for your own `recover()` and
you've got an apples-to-apples score against ground truth. Time-series? There's a parallel
`TEMPORAL_BASELINES` registry — **PCMCI+, LPCMCI, VARLiNGAM, Granger** — graded by `grade_temporal_spec`.

---

## 4. Pearl's three rungs — on a world whose answer you already hold

Discovery is rung one. The same declared SCM lets you stand on all three rungs of Judea Pearl's *Ladder
of Causation*, and check your answer against the truth every time.

**Rung 2 — Doing (intervention).** `do(x)` is **graph surgery**: it cuts every arrow *into* a variable
and keeps every arrow *out*. So whatever then moves the outcome is the *real* effect. Watch it correct
an overstated number:

```python
import numpy as np
from causal_worlds import build_substrate, worlds
sub = build_substrate(worlds.get("coffee"), standardize=False)
ff, sa = sub.variables.index("footfall"), sub.variables.index("sales")

seen  = sub.sample(40_000, seed=0).data
slope = np.polyfit(seen[:, ff], seen[:, sa], 1)[0]                              # 1.56  (what the data says)
hi = sub.sample(40_000, seed=1, do={"footfall":  1.0}).data[:, sa].mean()
lo = sub.sample(40_000, seed=1, do={"footfall": -1.0}).data[:, sa].mean()
print(round(slope, 2), round((hi - lo) / 2, 2))   # 1.56  0.90
```

The data slopes `sales` on `footfall` at **1.56**. The *true* causal effect is **0.90** — the data
overstated it by the hidden confounder's contribution. (For the pure-mirage `overtime → sales` pair,
the same move gives **0.00** against a 0.64 correlation: all mirage, no mechanism.)

**Rung 3 — Imagining (counterfactual).** *"We sold what we sold — what would sales have been **on that
same day** if footfall had been higher?"* That needs the full model with this specific unit's hidden
noise held fixed. Because the SCM is declared, it's **exact** — Pearl's abduction → action →
prediction:

```python
from causal_worlds import counterfactual, worlds
cf = counterfactual(worlds.get("coffee"), do={"footfall": 2.0}, seed=0)
print(round(cf.factual["sales"], 2), "->", round(cf.counterfactual["sales"], 2))   # 3.24 -> 4.55
print({k: round(v, 2) for k, v in cf.effect.items() if v})  # {'footfall': 1.46, 'demand': 0.73, 'sales': 1.31, ...}
```

On that exact day, sales of **3.24** would have been **4.55**. And it works on **temporal** worlds too —
roll a whole trajectory forward under a *sustained* intervention:

```python
from causal_worlds import counterfactual_temporal, worlds
tcf = counterfactual_temporal(worlds.get("supply"), do={"order": 2.0}, seed=0, steps=200)
# holding orders high across 200 steps:  inventory +3.35,  stockout -2.68,  cost +0.63 (mean shift)
```

Hold orders high for 200 steps and stockouts drop by **2.68** while carrying cost rises **0.63** — the
classic inventory trade-off, computed exactly. (And it's cross-checked: counterfactuals are verified to
average back to interventions, Pearl's own consistency law. Measured, not asserted.)

---

## 5. Don't just discover the world — stay optimal in it

Here's the part the coffee snapshot doesn't show, and the one I find most exciting. The *same* declared
worlds are a **control** benchmark. Because the mechanisms are known, the best the levers can do is
**computable** — a by-construction optimal policy — so a controller is graded by **regret**, with no
learned reward and no real data:

```python
from causal_worlds import worlds, default_objective, optimal_policy, grade_control
spec = worlds.get("coffee"); obj = default_objective(spec)
best = optimal_policy(spec, obj)                          # {'price': 0.0}  ← declared optimum
print(grade_control(spec, obj, {"price": 3.0}, seed=7).regret)   # 4.502  (overshoots the quadratic cost)
print(grade_control(spec, obj, best,            seed=7).regret)   # 0.000  (the answer key plays itself)
```

But the *load-bearing* metric is **regret under perturbation** — the "stay optimal under shift" thesis.
The coffee world has a regime sign-flip: the `price` lever *helps* demand in one regime and *hurts* it
in the other. So the regime-aware optimum flips sign with it — and a regime-**blind** policy that's
perfect in one regime gets clobbered when the regime turns over:

```python
from causal_worlds import regime_optimal_policy, regret_under_perturbation
base = regime_optimal_policy(spec, obj, regime_on=set())          # {'price': -1.0}
wknd = regime_optimal_policy(spec, obj, regime_on={"weekend"})    # {'price': +1.0}  ← the sign flips!

rep = regret_under_perturbation(spec, obj, base, seed=7)          # play the baseline-tuned policy everywhere
print(rep.per_regime, rep.worst_regret)   # {'baseline': 0.0, 'weekend': 2.0}   worst = 2.0
```

![Control under perturbation: a regime-blind policy has 0.0 regret in its own regime and 2.0 when the regime flips, while a regime-aware policy stays at 0.0 throughout.](https://raw.githubusercontent.com/noumenal-ai/causal-worlds/main/blog/figures/tour2_regime_flip.png)

A policy tuned for one regime scores **0.0 regret there and 2.0 when the regime flips**. A regime-aware
controller that flips with it stays near zero throughout. That gap *is* the score for staying optimal
under shift — and there's a `Gymnasium` env (`causal_worlds.gym.ControlEnv`) that shifts the regime
between steps, so cumulative regret falls out of any RL or decision agent you drop in.

---

## 6. The honesty instruments — also one call each

A benchmark is only worth its integrity, so the rigor is part of the public API, not a footnote:

- **Provenance on every bundle** — author model, judge model, grader, and an explicit honesty
  disclaimer travel with the world (`author=claude-opus-4-8 judge=gemini-2.5-flash
  grader=interventional-ci`). You always know who made the answer key.
- **A descriptive difficulty axis** — `structural_difficulty(spec)` →
  `StructuralDifficulty(score=2.0, hidden_confounders=1, confounded_pairs=1, sign_flips=1,
  density=0.2)`. Reported *with its uncertainty*: it describes a world, it doesn't pretend to predict
  who'll fail.
- **Name-blinding for the anti-cliché certificate** — `anonymize_spec(spec)` rewrites
  `weekend → X1, ... sales → X7` so you can measure whether a method is *reading the graph off the
  variable names* instead of doing real discovery. (In the launch post, name-blinding a world dropped a
  data-free LLM's guess from F1 0.71 to the chance floor of 0.01.)

---

## Try it in 60 seconds (no API key)

```bash
pip install causal-worlds
```

```python
from causal_worlds import worlds, grade_spec, InterventionalCiDiscoverer
print(grade_spec(worlds.get("coffee"), InterventionalCiDiscoverer()))
# directed_shd=0  f1=1.0  confounded_reported=0   ← now swap in YOUR discoverer
```

Render a world: `causal-worlds viz coffee`. Grade a controller by regret. Ask a counterfactual. Or
author a brand-new world from a single sentence (`[llm]` extra + keys):

```bash
causal-worlds generate "a regional grid with rooftop solar, batteries, and time-of-use pricing" ./grid-world
```

The five runnable [`examples/`](https://github.com/noumenal-ai/causal-worlds/tree/main/examples) each
carry their expected output in the docstring — the first four need no key.

---

If you build or evaluate causal-discovery or control methods — or you just enjoy trying to *game* a
benchmark without doing real discovery — that's the most useful thing you can do, and exactly how this
project got hardened. Repo + issues:
[github.com/noumenal-ai/causal-worlds](https://github.com/noumenal-ai/causal-worlds). MIT. Honest
measurement is the whole identity here — if a number is wrong, show us, in the open.
