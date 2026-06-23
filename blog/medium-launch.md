# You can't trust a causal benchmark you can memorize. So we built one you can't — then tried to break it ourselves.

*Introducing `causal-worlds`: a leakage-resistant, fiction-first generator of causal worlds with a
declared ground-truth answer key — plus the honest story of the flaws we found auditing our own work.*

---

To know whether a method can *discover* cause and effect — not correlate, not recite what it read on the
internet — you need a test where you already know the answer. That's harder than it sounds, and it's the
quiet scandal under a lot of causal-discovery and "LLMs-reason-about-causality" results.

The principle: **a causal benchmark is only trustworthy if the method can't get the right answer for the
wrong reason.** Two big wrong reasons:

1. **Memorization.** Build it from real, named systems — "smoking → cancer," a textbook supply chain —
   and a model can often *recite* the graph from variable names alone, never touching the data. Not
   discovery. A causal parrot.
2. **Leakage.** Use a synthetic structural causal model (SCM) instead, and the *way you generate the
   data* can encode the answer. The field has a name — **varsortability** — and a warning paper,
   *"Beware of the Simulated DAG!"*: a trivial baseline that sorts variables by marginal variance can
   beat real discovery algorithms, because the simulator made the causal order readable off the
   variance. The answer leaked into the data.

We wanted a benchmark that fails neither — then spent weeks trying to prove our own failed both anyway.
That self-audit is the real story.

---

## What we built

[`causal-worlds`](https://github.com/noumenal-ai/causal-worlds) (MIT, `pip install causal-worlds`) turns
a plain-language description of an *operation* into a **fictional but coherent causal world** with a
**declared ground-truth causal graph** — an answer key, by construction.

You write a sentence — *"a regional coffee chain with weekend swings and variable lead times"* — and get
back an executable simulator, the time-series it emits, and the structure that generated that data:
directed edges, regime sign-flips, and the *hidden confounders* that make discovery hard. Because the
structure is **declared, not learned**, the answer key is derived from the spec and can never disagree
with the simulator. And because the worlds are **fiction-first** — coherent, but not models of any real
system — there is **nothing real to memorize and no data to leak.** That's the whole point; the rest of
this post is whether we delivered it.

The pipeline is adversarial about its own integrity: an **author** (Claude) writes the SCM — dialable to
an `adversarial` tier that makes the *obvious* name-based guess **wrong** (phantom edges, reversed edges,
a misdirecting mediator) while keeping every declared edge detectable. An **independent judge** (Gemini,
a *different model family* — the standard self-preference mitigation) scores it and tries to guess the
graph from names alone. **Four gates** admit a world only if it's valid, sample-sane,
faithful-by-construction, and not guessable. Engine, grading, and `do()` interventions run with **no API
key**; only authoring needs a model.

---

## The decisive finding — and exactly what it is *not*

The headline, on the 26-world hardened benchmark (`v0.6`, n=4000, three seeds):

![Confounded pairs kept as a causal edge, summed over the 26 worlds (seed-averaged). The latent-aware
interventional reference is at 0; PC+interventions and GIES are at 30.](figures/fig1_confounded_kept.png)

Every world hides one or more confounders `L`, each driving two observed variables. Can a method tell
**confounding** (`X ← L → Y`) from **causation** (`X → Y`)? The bars count the confounded pairs each
method keeps as a *causal* edge — i.e. gets fooled — **summed across the 26 worlds and averaged over
seeds** (each world hides ~1–2 such pairs, so the totals can exceed 26 and run fractional).

- A simple **latent-aware interventional rule** (our reference grader): fooled **0** times.
- **Observational PC**: fooled 29 times (no surprise — PC assumes no hidden confounders).
- The telling rows are the `+interventions` methods. We gave PC and GIES the **same interventional
  budget** as the reference (pooled observational + per-variable `do()` data) — making the comparison
  **information-fair**: we compare *methods*, not who got more data. Given that budget, **PC+interventions
  still gets fooled 30 times** — no better than observational PC. GIES, 30 too.

The dividing line is **not interventions. It's latent-awareness.** You need *both* the interventional
data *and* a method that knows hidden confounders can exist. The advantage is robust (PC+interventions:
ΔF1 = +0.37, 95% bootstrap CI [0.33, 0.42], every method's CI excluding zero) — but the *mechanism* is
identifiability.

Be scrupulous here: this is an **identifiability result**, and the reference grader is a *deliberately
simple, textbook discoverer the benchmark is designed to reward* — **not** a clever new algorithm
"beating the toolbox." The fact is old (Jaber et al.'s Ψ-FCI; Hauser & Bühlmann's GIES): `{X→Y}` and
`{X←L→Y}` are indistinguishable observationally, and distinguishable only with the right interventions
*and* a latent-aware rule. `causal-worlds` doesn't contribute the theorem — it contributes a clean,
reproducible, leakage-resistant *apparatus* that surfaces it as a crossover you re-run in a minute. Worth
nothing unless the worlds are sound — so we went after our own worlds next.

---

## The credible part: we audited our own benchmark and found real flaws

A benchmark author calling their benchmark great is worth nothing. One showing the three ways they caught
themselves cheating, and the fixes — that's the signal.

### Flaw 1: circular admission (fixed)

Early on, a world was admitted partly because *the same reference grader that would later "win" on it*
judged it recoverable. Circular — the benchmark was selecting for worlds its favored method liked. A
skeptic would rightly throw the result out.

Fixed in v0.15: **admission is now grader-independent.** A world is admitted iff its declared SCM is
**faithful by construction** — every declared edge induces a *detectable* partial correlation given the
target's other parents, and regimes genuinely modulate a coefficient — computed in **closed form** from
the population covariance `(I−B)⁻¹ Ω (I−B)⁻ᵀ`. **No discovery method is ever run to admit a world.** The
reference grader's score is reported, never used to gate. The circularity is gone.

### Flaw 2: name-guessability (fixed for the cliché that matters; residual disclosed)

Could an LLM recite the graph from variable names, never touching the data? On our first real benchmark
(`v0.5`) — embarrassingly, yes: a data-free, name-only guess scored **F1 0.71**. The parrot. We labeled
the set "LEAKY," and it drove much of the work that followed.

The fix: the `adversarial` author plus a strict gate, measured with a three-tier *certificate* of
progressive blinding (in the spirit of the Caliper "the blind score is the real score" critique):

![Directed F1 of a data-free LLM guess at three disclosure levels, against the chance floor. Named v0.5
0.71, named v0.6 0.38, name-blind 0.46, name+role-blind 0.01.](figures/fig2_anti_cliche.png)

- **Named** (names + roles): 0.71 → **0.38**.
- **Name-blind** (names → `X1..Xn`, roles kept): **0.46** — *higher* than named. Strip the helpful names
  and the guess gets *worse*: the adversarial names now actively **mislead**. The cliché is now a trap.
- **Name + role-blind**: **0.01** — at the chance floor. **The structure itself isn't guessable once the
  semantics are stripped.**

So the cliché that matters — **name/structure memorization — is eliminated.** But there's a residual we
refuse to hide: with roles visible, the guesser beats the 0.18 chance floor (0.46). That's a **role-type
prior** — the convention that a *controllable* drives an *outcome*. We added a roles-only gate (v0.24);
it barely moved the number (0.46 → 0.43) at the cost of more than half the worlds. Honest read:
single-sample LLM-judge gating is noisy, and role-type prior is *intrinsic* — you can't remove it without
deleting the lever→outcome path that makes a world an operation. So we report it as a legitimate,
disclosed residual, not a leak we pretend we closed.

### Flaw 3: simulated-DAG leakage (fixed; one residual disclosed)

The `varsortability` trap from "Beware of the Simulated DAG!" — and its harder, scale-invariant cousin,
**R²-sortability**, which a 2023 follow-up showed you *can't* fix by standardizing after the fact.

![Two panels — sortability signals and trivial-baseline F1 — each comparing the iSCM result against its
prior state. Each "before" bar is labelled with the distinct historical state it is: the
varsortability/sortnregress "before" is the original unstandardized substrate, the
R²-sortability/R²-sortnregress "before" is the later v0.13 post-hoc-standardized state. Not one
baseline.](figures/fig3_leakage.png)

A note on the two "before" bars in each panel: they are **two different historical states, not one
baseline** — and the figure labels each bar accordingly. Our *original unstandardized* substrate was
badly leaky: varsortability **0.94**, trivial "sort-by-variance" baseline at F1 **0.74** — *beating PC
and FCI.* A *later* post-hoc standardization patch (v0.13) helped variance but left R²-sortability at
**0.73** (trivial baseline 0.40) — exactly the residual the 2023 paper predicts is *irremovable* after
the fact.

The real fix (v0.14) is **internal standardization (iSCM)**: each continuous variable is z-scored *as it
is generated*, in topological order, so neither variance nor predictability compounds along the causal
order. After iSCM: varsortability **0.54**, R²-sortability **0.60**, both trivial baselines collapsed to
F1 ≈ 0.33–0.37 — well under the real methods. And we disclose what we *haven't* fully closed:
R²-sortability **0.60 is still above the 0.5 "unreadable" line.** Disclosed, not buried. (The crossover
above is robust to iSCM — the reference still gets fooled 0 times.)

### The caveat we draw instead of bury: difficulty

We attach a "difficulty" to each world. Is it a *validated predictor* of how hard a method finds the
world? On this set — no, and we won't pretend otherwise:

![Difficulty vs error scatter; PC r=0.29 CI [-0.04,0.66], reference r=0.38 CI [-0.13,0.70], both CIs
include 0.](figures/fig4_difficulty.png)

Both correlations' 95% CIs include zero — so difficulty is a **descriptive axis, reported with its
uncertainty**, not a claim it predicts error.

---

## Beyond cross-sections: time and control

Two more tracks, with honesty labels attached.

**Temporal worlds.** The built-in `supply` world has autoregressive lead time, inventory, and a hidden
logistics confounder, with a lagged answer key. Time-series grading is validated end-to-end — PCMCI+
recovers the lagged structure *exactly* (F1 1.0), while VARLiNGAM keeps the spurious confounded edge.
**Honest caveat: n=1.** A temporal benchmark *set* (n>1) is on the roadmap, not a shipped claim.

**A control track.** Because the mechanisms are *declared*, the best the levers can do is computable — a
**by-construction optimal policy** (linear-Gaussian + quadratic cost: each lever's total effect is the
path-sum `(I−B)⁻¹[outcome, lever]`, marginalized over regimes). A controller is graded by **regret**
against that optimum — no external data, no learned reward. We also measure
**regret-under-perturbation**: a regime-blind policy that ignores a sign-flipped regime loses ~0.5 reward
in *every* regime, while the regime-aware optimum has ~0. A `Gymnasium` env
(`causal_worlds.gym.ControlEnv`) shifts the regime between steps; the agent must *detect* the shift from
observed means, and cumulative regret is the stay-optimal score.

---

## Try it in 60 seconds (no API key)

Everything except authoring runs offline. From the README:

```python
from causal_worlds import worlds, grade_spec, InterventionalCiDiscoverer

spec = worlds.get("coffee")                          # a hidden confounder + a regime sign-flip
report = grade_spec(spec, InterventionalCiDiscoverer())
print(report)   # directed_shd=0  skeleton_shd=0  f1=1.0  confounded_reported=0
#                ^ swap in YOUR discoverer to benchmark it against a known truth
```

Benchmarking your own method is one function — `recover(substrate, *, seed) -> set[(src, dst)]` — graded
against the answer key. Or generate a fresh world from a sentence (needs the `[llm]` extra + keys):

```bash
pip install 'causal-worlds[llm]'
causal-worlds generate "a hospital ED with triage staffing and bed pressure" ./my-world
causal-worlds elicit ./my-world      # …or describe it conversationally
```

---

## What's next, and how to help

The roadmap is short and honest: **nonlinearity** (worlds are currently linear-Gaussian) and a
**temporal benchmark set** (to turn n=1 into a distribution). Both are tracked as GitHub issues. The
R²-sortability residual (0.60) and the role-type prior are documented open questions, not closed boxes.

If you build or evaluate causal-discovery methods — or just enjoy breaking a benchmark — this is for you:

- **Try it:** `pip install causal-worlds`, then `causal-worlds grade coffee`.
- **Benchmark your method** against a known truth and tell us where it lands.
- **Break it:** game a world without doing real discovery. That's the most valuable contribution you can
  make — and exactly how this project got here.

Repo and issues: [github.com/noumenal-ai/causal-worlds](https://github.com/noumenal-ai/causal-worlds).
MIT, built on the shoulders of pgmpy, DoWhy, CausalPlayground, causal-learn, and Gymnasium.

Honest measurement is this project's entire identity. If we got something wrong, show us — in the open.
