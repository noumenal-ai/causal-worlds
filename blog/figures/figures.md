# Launch figures — causal-worlds

Four figures rendered by [`make_figures.py`](make_figures.py) directly from the repo's shipped eval
artifacts (`evals/*/report.json`). Re-run any time the benchmarks change:

```bash
cd causal-worlds
uv run --with matplotlib python blog/figures/make_figures.py
# (or, if your env needs the discovery extra resolved alongside matplotlib:)
uv run --extra discover --with matplotlib python blog/figures/make_figures.py
```

All four PNGs render headlessly (Agg backend) at 150 dpi. The matplotlib install succeeded in this
environment, so the PNGs are committed alongside the script; this file documents each one's purpose,
alt-text, and where it is used.

**Integrity note on data provenance.** Every plotted value is read *live* from a `report.json` with
two clearly-flagged exceptions in Figure 3: the *"before iSCM"* reference points. These are **two
distinct historical states, not one baseline**, and the figure now labels each "before" bar with the
state it actually is: (a) the *original unstandardized* substrate (varsortability **0.94** /
sortnregress F1 **0.74**) and (b) the *later v0.13 post-hoc-standardized* state that iSCM then improved
on (R²-sortability **0.73** / R²-sortnregress F1 **0.40**). Those historical states are documented in the
README/CHANGELOG but are not stored as a shipped `report.json`, so the script hard-codes them as labelled
annotations; each before-bar carries its own state label, and the figure subtitle/footnote say so
explicitly — so a chart-only skim cannot read the two as a single "before" group. The *"after iSCM"* bars
are all live from `evals/varsortability/v0.5`.

---

## Figure 1 — `fig1_confounded_kept.png` (the identifiability headline)

**Source:** `evals/baseline-crossover/v0.6/report.json` (`aggregate.*.total_confounded_kept`).
**Purpose:** the decisive result. A horizontal bar chart of *how many of the 26 worlds each method keeps
the hidden-confounded pair as a real causal edge* (lower is better). The latent-aware reference
(`interventional-ci`) is at **0**; everything else — including the `+intervention` methods that get the
**same `do()` budget** — sits at 21–30. The framing the chart enforces: the lever is **latent-awareness,
not interventions**, and this is an **identifiability** result, not "we beat the toolbox."

**Alt-text:** "Horizontal bar chart titled 'Only a latent-aware rule resists confounding.' The
latent-aware interventional-ci reference keeps the hidden-confounded pair as a causal edge in 0 of 26
worlds. PC plus interventions and GIES (both given an interventional budget but assuming causal
sufficiency) keep it in 30; observational PC in 29; DAGMA and DirectLiNGAM in 27; FCI in ~22; FCI plus
interventions in 21. Only the latent-aware rule reaches zero."

**Used by:** Medium post (the "decisive finding" section — primary figure). X thread **tweet 4/**
(the crossover reveal).

---

## Figure 2 — `fig2_anti_cliche.png` (the anti-cliché certificate)

**Source:** `evals/name-only-baseline/v0.6/report.json` (aggregate means + 95% CIs) with the v0.5
"named" point from `evals/name-only-baseline/v0.5/report.json` for the before/after contrast.
**Purpose:** the self-critique made legible. A bar chart of the directed F1 of a *data-free* LLM guess at
the true graph at three Caliper-style disclosure levels, against the random-graph chance floor.
**named (v0.5)** was 0.71 (leaky); **named (v0.6)** fell to 0.38; **name-blind** is *higher* at 0.46
(adversarial names now mislead); **name+role-blind** collapses to ~0.01 (structure is not guessable once
semantics are stripped). The disclosed residual — role-type prior — is named explicitly, not hidden.

**Alt-text:** "Bar chart titled 'The cliché that matters — name/structure memorization — is gone.' A
data-free LLM's directed-F1 guess at the true graph: 0.71 on the old leaky v0.5 set with names; 0.38 on
the hardened v0.6 set with names; 0.46 when names are blinded but roles kept (higher, because the
adversarial names mislead); and 0.01 when both names and roles are blinded — at or below the 0.18 chance
floor. Error bars show 95% confidence intervals."

**Used by:** Medium post (the "we audited our own benchmark" section). X thread **tweet 6/**
(the adversarial-audit / certificate tweet).

---

## Figure 3 — `fig3_leakage.png` (simulated-DAG leakage, before/after iSCM)

**Source:** `evals/varsortability/v0.5/report.json` (the "after iSCM" means) plus the documented
historical "before" reference points (see the integrity note above).
**Purpose:** the second self-found flaw and its fix. Two panels. Left: the two leakage signals —
varsortability (marginal variance) and R²-sortability (scale-invariant predictability) — after internal
standardization (iSCM) versus their prior historical states, against the 0.5 "order-not-readable" line.
Right: the matching trivial sorting baselines (`sortnregress`, `R²-sortnregress`), which used to beat
real discovery methods and now collapse beneath them. **Each "before" bar is a distinct historical state,
labelled on the bar** — the varsortability/sortnregress "before" is the *original unstandardized*
substrate; the R²-sortability/R²-sortnregress "before" is the *later v0.13 post-hoc-standardized* state —
so the two are never read as one baseline. The annotation flags the **disclosed residual**
R²-sortability 0.60 > 0.5.

**Alt-text:** "Two-panel figure titled 'iSCM closes the simulated-DAG leak the field warned about.' Each
panel's two 'before' bars are two DIFFERENT historical states, labelled on the bars, not a single
baseline: the varsortability / sortnregress 'before' is the original unstandardized substrate, while the
R²-sortability / R²-sortnregress 'before' is the later v0.13 post-hoc-standardized state that iSCM then
improved on. Left panel: under iSCM, varsortability is 0.54 (down from the original unstandardized 0.94)
and R²-sortability is 0.60 (down from the v0.13 post-hoc 0.73), both toward the 0.5 line that means the
causal order is not readable, with 0.60 flagged as a disclosed residual still above 0.5. Right panel:
under iSCM the trivial sortnregress baseline's F1 is 0.33 (down from the original unstandardized 0.74) and
R²-sortnregress is 0.37 (down from the v0.13 post-hoc 0.40), both now well below the real discovery
methods."

**Used by:** Medium post (the "leakage" half of the self-audit section). X thread **tweet 7/**
(the varsortability / R²-sortability fix).

---

## Figure 4 — `fig4_difficulty.png` (difficulty is descriptive, not predictive)

**Source:** `evals/baseline-crossover/v0.6/report.json` (`per_world.*.difficulty`,
`skeleton_shd_mean`, and `difficulty_vs_error.*.pearson` + CIs).
**Purpose:** the honest caveat, drawn rather than buried. A scatter of per-world declared difficulty vs
mean skeleton-SHD for observational PC and for the latent-aware reference, annotated with the Pearson r
and its 95% CI for each. Both CIs include 0 on this 26-world set — so we present difficulty as a
**descriptive** axis with its uncertainty, not as a validated predictor of error.

**Alt-text:** "Scatter plot titled 'Difficulty is a descriptive axis, not a validated predictor.' Per-
world declared difficulty on the x-axis (about 0.52 to 0.88) against mean skeleton-SHD error on the
y-axis. PC's points scatter from 2 to about 6.3 with a weak upward tendency (r = 0.29, 95% CI [-0.04,
0.66]); the latent-aware reference's points cluster near 1 (r = 0.38, 95% CI [-0.13, 0.70]). Both
confidence intervals include zero."

**Used by:** Medium post (the "what's next / honest caveats" section, or as an inline credibility
figure). X thread **tweet 8/** (the "we won't oversell difficulty" caveat) — optional attach.

---

## Feature-tour figures — `make_tour_figures.py` (computed live, not from JSON)

Two figures for the **follow-up** feature-tour post (`blog/medium-tour.md`). Unlike Figures 1–4, these
are recomputed from the built-in `coffee` world **at render time** by
[`make_tour_figures.py`](make_tour_figures.py), so they can never drift from the package's behaviour:

```bash
cd causal-worlds
uv run --extra discover --with matplotlib python blog/figures/make_tour_figures.py
```

### Tour 1 — `tour1_shootout.png` (the discovery shootout)

**Source:** live — `grade_spec(worlds.get("coffee"), <discoverer>, seed=0)` over the reference plus
every method in `BASELINES`. **Purpose:** the single-world companion to Figure 1. A horizontal bar
chart of directed-edge F1 with each bar annotated by how many spurious confounded pairs the method
keeps. The latent-aware `interventional-ci` reference is the only method that recovers the structure
(F1 1.0) *and* keeps zero confounded pairs; PC/FCI/GIES all hit F1 0.77 but keep the `overtime~sales`
phantom (red); DAGMA collapses (0.17); DirectLiNGAM drops the phantom but mangles structure (0.50).

**Alt-text:** "Horizontal bar chart titled 'Only a latent-aware method isn't fooled.' On the coffee
world, interventional-ci scores F1 1.00 keeping 0 confounded pairs; PC, FCI and GIES each score 0.77
while keeping 1 spurious confounded pair as a causal edge; DirectLiNGAM 0.50 keeping 0; DAGMA 0.17
keeping 1. Only the reference both nails the structure and avoids the trap."

**Used by:** Medium feature-tour (the "benchmarking discovery" section). X thread 2 (the shootout). LinkedIn Post 2.

### Tour 2 — `tour2_regime_flip.png` (control under perturbation)

**Source:** live — `regret_under_perturbation(coffee, default_objective, regime-blind policy, seed=7)`
with the regime-aware optima from `regime_optimal_policy`. **Purpose:** make the stay-optimal thesis
visible. A grouped bar chart: a regime-blind policy (tuned for the baseline regime) has 0.0 regret in
its own regime but 2.0 when the regime flips to weekend — because the `price` lever's optimum reverses
sign (−1 → +1); a regime-aware policy stays at 0.0 in both.

**Alt-text:** "Grouped bar chart titled 'A regime-blind policy stays optimal — until the regime flips.'
In the baseline regime both the regime-blind and regime-aware policies have 0.0 regret; in the weekend
regime the regime-blind policy jumps to 2.0 regret while the regime-aware policy stays at 0.0."

**Used by:** Medium feature-tour (the "control" section). X thread 3 (the ACT rung). LinkedIn Post 3.

---

## Hero image concept (for the Medium header / X card)

**Concept name: "The Answer Key."** A single, calm, conceptual hero — not a chart — that states the
product in one glance: *we declare the truth first, so we can grade.*

- **Composition.** A clean left-to-right flow on a near-white (#fafafa) ground, three beats:
  1. a short line of plain language in a quotation mark, e.g. *"a regional coffee chain with weekend
     swings and variable lead times,"* set in a humanist sans;
  2. an arrow into a small, deliberately tidy **directed graph** of 5–6 labelled nodes — one node drawn
     as a *dashed, hollow* circle labelled with an "L" (the hidden confounder) feeding two observed
     nodes, and one edge carrying a tiny ± glyph (the regime sign-flip);
  3. the graph stamped with a teal **"ANSWER KEY"** seal / key-shaped mark.
- **The single idea it must carry:** the graph *comes first* (declared), and the data is generated *from*
  it — so there is nothing to leak and nothing to memorize. The dashed "L" node is the whole story in one
  symbol: the thing observational discovery cannot see.
- **Palette.** The figure palette: teal `#0f766e` for the "truth / answer-key" accent, slate `#64748b`
  for observed nodes, a muted red `#b91c1c` only on the hidden-confounder node and the ± glyph (the
  trap). Ink `#1b1f24` text, generous whitespace, thin 1.5px edges.
- **Tone.** Editorial and restrained — closer to a journal figure than a SaaS landing page. No
  gradients, no 3D, no stock imagery. The craft is in the typography and the negative space.
- **Caption (under the hero):** *"causal-worlds turns a sentence into a fictional operation with a
  declared ground-truth causal graph — an answer key by construction. The dashed node is the hidden
  cause that observational discovery can't tell from a real edge."*
- **Fallback:** if a bespoke hero can't be produced for launch, **use Figure 1** (`fig1_confounded_kept`)
  as the header image — it is the single most legible statement of the result and reads well as a card.
