# LinkedIn — causal-worlds

Three posts, each making a *different* case. LinkedIn does **not** render Markdown — bodies are plain
text with line breaks and `•` bullets. Attach the hero image (`docs/figures/coffee_world.png`) where
noted. Post one at a time; they're independent angles, not one launch. Industry/role-specific cuts
(RL, time-series, ops, healthcare, etc.) come in a later round.

- **Post 1 — The idea.** A benchmark that knows the right answer, because it's declared. (broad)
- **Post 2 — The proof.** I ran every discovery method in the box; only one isn't fooled. (rigor)
- **Post 3 — The breadth.** See, do, imagine, *and act*. (capability)

═══════════════════════════════════════════════════════════════════════
## Post 1 — The idea: "correlation isn't causation — proven on demand"
═══════════════════════════════════════════════════════════════════════
[attach: docs/figures/coffee_world.png]

Everyone repeats "correlation isn't causation." Almost no one can prove it on demand.

Here's a dataset from a small coffee chain: staff overtime and sales rise together — correlation 0.64.
Any dashboard would say overtime drives sales. But actually intervene — force overtime up and down and
watch sales — and the effect is 0.00. The real driver was something nobody measured: local foot traffic
(a festival, good weather) lifting both at once.

In the real world you can never be sure, because you don't know the true structure.

So we built worlds where you do.

causal-worlds (open source, MIT) turns a plain-language description of an operation into a
fictional-but-coherent world with a DECLARED ground-truth causal graph — an answer key, by
construction. Write a sentence:

   causal-worlds generate "a hospital ED with triage staffing and bed pressure" ./world

…and you get back an executable simulator, the data it emits, and the exact structure that generated
it — directed edges with strengths, the hidden confounders, the regime sign-flips.

Two modes: by default it builds a benchmark-grade world (one you can't solve by reading the variable
names — that's the whole point). Or pass --playground to just describe a world and get it: the
faithfulness check stays and you still see a "guessability" score, but nothing gets rejected. The
bundle records which mode it was, so a playground world never masquerades as benchmark-grade.

Why fictional? Because a benchmark you can memorize isn't a benchmark. Fiction-first means there's
nothing real to recite and no data to leak — so a method only scores by genuinely discovering cause
from effect. And because the key is derived from the spec, it can never disagree with the simulator.

If you've ever been burned by a confident, confounded number, this one's for you.

→ The full field guide (Medium): [link to medium-tour]
→ pip install causal-worlds
→ github.com/noumenal-ai/causal-worlds

#causalinference #datascience #machinelearning #opensource

═══════════════════════════════════════════════════════════════════════
## Post 2 — The proof: "I ran every method in the box against a known answer"
═══════════════════════════════════════════════════════════════════════
[attach: blog/figures/tour1_shootout.png]

I took a causal world whose true graph I already had, and ran every discovery method that ships in the
box against it — same data, same seed. One column tells the whole story: how many spurious "confounded"
pairs each method keeps as if they were real causal edges.

   method               F1     confounded_kept
   interventional-ci   1.00         0     ← the latent-aware reference
   pc                  0.77         1
   fci                 0.77         1
   gies                0.77         1
   dagma               0.17         1
   directlingam        0.50         0

Only the reference keeps zero AND recovers the structure exactly. (DirectLiNGAM happens to drop the
phantom edge but mangles the rest — F1 0.50.)

The dividing line isn't raw power, and — proven across the 26-world hardened set — it isn't even
interventions: given the SAME interventional budget, PC still keeps the confounded pair ~30 times, no
better than its observational self. The lever is whether the method knows hidden confounders can exist.
(ΔF1 +0.37, 95% CI excluding zero.)

I want to be scrupulous: this is a textbook identifiability result (Ψ-FCI, GIES), and the reference is a
deliberately simple discoverer the benchmark is built to reward. We don't contribute the theorem — we
contribute a clean, leakage-resistant apparatus that re-surfaces it as a crossover you re-run in a
minute. And we earned the right to claim that by auditing our own benchmark: we found and fixed circular
admission, name-guessability (a data-free guess once scored F1 0.71), and simulated-DAG leakage — and we
disclose the residuals we haven't fully closed.

Honest measurement is the whole identity. Swap in your own method (one function, recover()) and try to
beat the reference's zero.

→ The full field guide (Medium): [link to medium-tour]
→ pip install causal-worlds
→ github.com/noumenal-ai/causal-worlds

#causalinference #machinelearning #opensource #benchmarking

═══════════════════════════════════════════════════════════════════════
## Post 3 — The breadth: "see, do, imagine — and act"
═══════════════════════════════════════════════════════════════════════
[attach: blog/figures/tour2_regime_flip.png]

Most causal benchmarks test one thing: can you SEE the structure in the data? But Judea Pearl's ladder
has three rungs — and the agents we actually deploy have to ACT. causal-worlds lets you do all of it, on
a world whose answer you already hold.

• SEE → DO. do(x) is graph surgery: cut every arrow into a variable, keep every arrow out. On the coffee
world, the data slopes sales on footfall at 1.56 — but the true causal effect is 0.90. The intervention
strips out the confounder's contribution and corrects the number.

• DO → IMAGINE. "We sold what we sold — what would sales have been if footfall were higher, that same
day?" Exact, because the model is declared (abduction → action → prediction): factual 3.24 →
counterfactual 4.55. It works on time-series worlds too, rolling a whole trajectory forward under a
sustained intervention.

• IMAGINE → ACT. The same worlds are a control benchmark. The mechanisms are known, so the optimal
policy is computable — you're graded by regret, with no learned reward and no real data. And the
load-bearing metric is regret under perturbation: the coffee world's price lever flips sign across
regimes (optimum −1 in one, +1 in the other), so a regime-blind policy scores 0 regret in its own regime
and 2.0 when the regime flips. A regime-aware controller stays near zero. A Gymnasium env shifts the
regime under your agent, step by step.

Seeing is rung one. This is the whole ladder, plus the rung where decisions live — with a known answer
behind every one.

Open source, MIT. If you build or evaluate causal methods, come break a world.

→ The full field guide (Medium): [link to medium-tour]
→ pip install causal-worlds
→ github.com/noumenal-ai/causal-worlds

#causalinference #AI #reinforcementlearning #opensource
