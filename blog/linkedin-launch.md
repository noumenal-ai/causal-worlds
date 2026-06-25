# LinkedIn — causal-worlds (multiple takes)

Single posts, ready to paste. LinkedIn does **not** render Markdown, so each body is plain text with
line breaks and `•` bullets. Attach the hero image (`docs/figures/coffee_world.png`) where noted. Post
whichever take fits; they're independent angles, not one launch.

═══════════════════════════════════════════════════════════════════════
## Take A — broad & accessible ("correlation isn't causation — proven on demand")
═══════════════════════════════════════════════════════════════════════
[attach: docs/figures/coffee_world.png]

Everyone repeats "correlation isn't causation." Almost no one can prove it on demand.

Here's a dataset from a small coffee chain: staff overtime and sales rise together — correlation 0.64.
Any dashboard would say overtime drives sales. But actually intervene — force overtime up and down and
watch sales — and the effect is zero. The real driver was something nobody measured: local foot traffic
(a festival, good weather) lifting both at once.

In the real world you can never be sure, because you don't know the true structure.

So we built worlds where you do. causal-worlds (open source, MIT) turns a plain-language description of
an operation into a fictional-but-coherent world with a DECLARED ground-truth causal graph — an answer
key, by construction. You can see the correlations, intervene with do(), and even ask counterfactuals
("what would sales have been had footfall been higher, that same day?") — and check every answer against
the truth, because you wrote it.

Why fictional? Because a benchmark you can memorize isn't a benchmark. Fiction-first means there's
nothing real to recite and no data to leak — so a method only scores by genuinely discovering cause from
effect.

If you've ever been burned by a confident, confounded number, this one's for you.

→ Read the full story (Medium): https://selftaughtamit.medium.com/correlation-lies-we-built-a-causal-world-that-can-prove-it-then-tried-to-break-it-ourselves-30efae26b457
→ pip install causal-worlds
→ github.com/noumenal-ai/causal-worlds

#causalinference #datascience #machinelearning #opensource

═══════════════════════════════════════════════════════════════════════
## Take B — builder / research ("we audited our own benchmark and published the flaws")
═══════════════════════════════════════════════════════════════════════

We open-sourced a causal-discovery benchmark — and then published the flaws we found auditing our own
work. That second part is the point.

causal-worlds generates fictional causal "worlds" with a declared ground-truth graph, to test whether a
method can recover cause from effect where correlation-based tools get fooled by hidden confounders.
Building it was the easy part. Trusting it was the hard part — so we spent weeks trying to break it:

• Circular admission — worlds were being admitted by the same grader that later "won" on them. Fixed:
admission is now grader-independent, computed in closed form, with no discovery method run.

• Memorization — a data-free, name-only LLM guess scored F1 0.71. It was reciting variable names, not
discovering. We made the names mislead; strip them and the structure is unguessable (≈ chance).

• Simulated-DAG leakage — the way you generate synthetic data can leak the causal order. A trivial
"sort by variance" baseline beat real algorithms (F1 0.74). Fixed with internal standardization.

We disclose the residuals we haven't fully closed, too. It now spans all three rungs of Pearl's ladder —
association, intervention, counterfactual — each verified, not asserted.

Honest measurement is the whole identity. If we got something wrong, the best contribution you can make
is to show us — in the open.

→ Read the full story (Medium): https://selftaughtamit.medium.com/correlation-lies-we-built-a-causal-world-that-can-prove-it-then-tried-to-break-it-ourselves-30efae26b457
→ pip install causal-worlds
→ github.com/noumenal-ai/causal-worlds

#causalinference #machinelearning #opensource #benchmarking

═══════════════════════════════════════════════════════════════════════
## Take C — the ladder / capability ("seeing, doing, imagining")
═══════════════════════════════════════════════════════════════════════
[attach: docs/figures/coffee_world.png]

Judea Pearl's Ladder of Causation has three rungs: seeing (what correlates), doing (what happens if I
intervene), and imagining (what would have happened). Most tools live on rung one.

We built a sandbox you can stand on all three — with a known answer key.

causal-worlds turns a sentence — "a coffee chain with weekend swings and variable lead times" — into a
fictional-but-coherent causal world whose true structure is declared, not learned. So you can:

• SEE — overtime and sales correlate 0.64.
• DO — force overtime (graph surgery on the model) and the real effect on sales is 0.00. The
correlation was a hidden confounder all along.
• IMAGINE — "what would sales have been had footfall been higher, that same day?" Exact, because the
world is fully specified.

Because it's fiction-first, there's nothing to memorize and no data to leak — the only way to score is
to actually discover. And we audited our own benchmark for leakage and memorization, and published what
we found.

Open source, MIT. If you build or evaluate causal methods — come break a world.

→ Read the full story (Medium): https://selftaughtamit.medium.com/correlation-lies-we-built-a-causal-world-that-can-prove-it-then-tried-to-break-it-ourselves-30efae26b457
→ pip install causal-worlds
→ github.com/noumenal-ai/causal-worlds

#causalinference #datascience #AI #opensource
