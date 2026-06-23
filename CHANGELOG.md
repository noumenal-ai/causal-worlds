# Changelog

All notable changes to causal-worlds are documented here. Format: [Keep a Changelog](https://keepachangelog.com/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.24.0] — 2026-06-23

**Gymnasium control env (#14) + the roles-only anti-cliché gate (#13 machinery).**

### Added
- **`causal_worlds.gym.ControlEnv`** — a `gymnasium.Env` over the Stage-2 control loop: action = lever
  values, reward = the control objective under the *current* regime, and the **regime shifts between
  steps** (a perturbation). `info` carries the regime-aware `optimal_reward` and `regret`, so
  cumulative regret is the stay-optimal-under-perturbation score; the agent sees only the
  observed-variable means (it must *detect* the shift, never the spec). New `gym` extra. (Closes #14.)
- **Roles-only T4 gate** — admission now also rejects worlds whose **name-anonymized, roles-kept**
  prior recovers the graph (`>= 0.4`), closing the role-label leak the v0.6 certificate surfaced; and
  the `adversarial` author tier gained a **role-misdirection** lever (route the lever's influence
  through an observable mediator, no obvious direct lever→outcome edge). (#13 machinery.)

### Notes
- `import causal_worlds` stays keyless — `gym.py` is imported explicitly (gymnasium isn't pulled by
  the base package). 135 tests, 95% coverage. The role-clean regenerated set rides #13.

[0.24.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.24.0

## [0.23.0] — 2026-06-23

**Stage-2 control: regret-under-perturbation — the stay-optimal thesis, measured.**

A perturbation is a **regime shift**. A regime-blind policy can be optimal on average yet collapse
when a sign-flipped regime is active; a regime-aware controller stays optimal. This is the World
Models thesis, and the v0.6 worlds (with their regime sign-flips) are built to exhibit it.

### Added
- **`regime_optimal_policy(spec, objective, regime_on)`** — the optimum when the regime is known
  (that branch's effects only), and **`regime_configs`** (every regime setting).
- **`regret_under_perturbation(spec, objective, policy, *, seed) -> PerturbationReport`** — a fixed
  policy's regret vs the regime-aware optimum under *every* regime shift (worst + mean + per-regime).
- `expected_reward` gained a `regime=` override (pin regime switches via `do`, excluded from the
  lever penalty).

### Validated
- On a sign-flip world the regime-blind optimum is `price=0` (its marginal effect cancels), but the
  regime-aware optima are `+1` / `−1` — so the blind policy loses **~0.5 reward in *every* regime**
  (worst-regret 0.5), while playing the matching per-regime optimum has ~0 regret. The stay-optimal
  gap is real and computable. 131 tests, 95% coverage. Advances #11.

[0.23.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.23.0

## [0.22.0] — 2026-06-23

**`benchmark/v0.6` — the hardened set, and the #12 anti-cliché fix proven at scale.**

Regenerated the benchmark with the `adversarial` author tier under the strict T4 gate (v0.19), capped
at 30 prompts (`evals/scale/generate_hardened.py`). **26/30 admitted**, mean name-guessability
difficulty **0.64** (was ~0.29 on v0.5), faithfulness 0.87, grader SHD ~1 — at ~1.5 author attempts.

### Validated (measured, not asserted)
- **Name leakage fixed.** The named name-only prior fell **0.71 → 0.38** (chance 0.18). The
  **name+role-blind** prior collapses to **0.01** — the structure is not guessable once semantics are
  stripped. The **name-blind** prior (0.46) is *higher* than named, so the adversarial names now
  actively **mislead** — names became a liability for a prior-only guesser.
- **Identifiability finding stronger.** On v0.6 the reference `interventional-ci` holds at
  confounded-kept **0** (F1 0.90) while every causal-sufficiency method keeps more traps (pc 29,
  gies 30, dagma 27, `pc+do` 30 at equal interventional budget). ΔF1 +0.37 [0.33, 0.42] for `pc+do`.

### Honest residual
- **Role labels still leak** (name-blind prior 0.46 ≫ chance): role conventions (controllable→outcome)
  remain guessable. The gate checks the named and fully-blind priors but not the roles-only prior —
  a roles-only gate (or randomized role hints) is the next step (#13).

### Notes
- `v0.5` is kept for comparison. README crossover table + anti-cliché caveat now report v0.6. The
  package code is unchanged since v0.21 (this is a benchmark + docs release). 128 tests, 95% coverage.

[0.22.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.22.0

## [0.21.0] — 2026-06-23

**Stage-2 control — the optimal-policy answer-key (the dual-track scope, realized).**

The control track grades a *policy* against the *declared optimum*, just as discovery grades a method
against the declared graph. Because the mechanisms are known, the best the levers can do is a
deterministic function of them — correct-by-construction, so the "magnitude problem" dissolves
([scope §1a](docs/scope.md)).

### Added — `control` module
- **`ControlObjective`** + **`default_objective`** — a linear-quadratic problem: maximise
  `E[outcome | do(u)] - cost/2·||u||²` over the controllable levers.
- **`lever_effects`** / **`optimal_policy`** — the answer-key: each lever's total effect on the outcome
  is the path-sum `(I-B)⁻¹[outcome, lever]` (levers intervened → rows cut), **marginalised over
  regimes**; the LQ optimum decouples to `u*_i = effect_i / cost`.
- **`expected_reward`** / **`grade_control`** / **`grade_controller`** — score a policy (or a pluggable
  `Controller`) by **regret** against the optimum, on the **unstandardized** world (`control_substrate`
  — iSCM centring would erase the outcome's response, so control uses raw magnitudes).
- **`Controller` Protocol** — a control test-taker sets levers from the substrate + objective, never
  seeing the mechanisms.

### Validated
- On a chain world the path-sum effect, closed-form optimum, and ~0 regret all check out; a
  do-nothing policy has regret ≈ the optimum's worth.
- **Thesis seed**: a regime sign-flipped lever has **marginal effect 0** — useless to a regime-blind
  policy. That's exactly what the next step (regret-under-perturbation, regime-aware vs static) will
  exploit. 128 tests, 95% coverage.

[0.21.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.21.0

## [0.20.0] — 2026-06-23

**Broaden the baselines — DAGMA + DirectLiNGAM (more witnesses for the identifiability finding).**

### Added
- **`DagmaDiscoverer`** (Bello et al., NeurIPS 2022 — continuous-optimization, observational) and
  **`DirectLingamDiscoverer`** (Shimizu et al.) behind the `Discoverer` Protocol, registered in
  `BASELINES`. Added `dagma` + `lingam` to the `discover` extra. `parse_weighted_adjacency` parses
  their thresholded coefficient matrices (pure, unit-tested).
- Both run in the crossover's observational track (now 8 methods).

### Finding
- Both are causal-sufficiency methods and, as predicted, **keep the hidden-confounded pair**
  (DAGMA 16.0, DirectLiNGAM 14.7 of 35) — two more independent witnesses that the dividing line is
  **latent-awareness**, not the specific method. Only the latent-aware `interventional-ci` reaches 0.
- Honest caveat: both run at default hyperparameters, and LiNGAM's non-Gaussian assumption is
  violated by these linear-Gaussian worlds, so their *skeleton* accuracy (DAGMA F1 0.22, LiNGAM 0.38)
  is not their best — the robust, relevant verdict is confounded-kept, reported as the headline.

### Notes
- 121 tests, 95% coverage. Strengthens the crossover for review; DCDI/SCORE/NoGAM/AVICI/Rhino remain
  future adds (heavier packaging).

[0.20.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.20.0

## [0.19.0] — 2026-06-23

**Anti-cliché hardening (#12) — strict gate, blind control, adversarial author tier.**

The v0.17 name-only baseline found the worlds too guessable (named F1 0.71 vs 0.20 chance). Grounded
in a literature review (Caliper 2606.04915 — *the blind score is the real score*; Corr2Cause; the
contamination/critical-review line), v0.19 hardens the machinery so the *next* generation produces
genuinely hard worlds. (The shipped `benchmark/v0.5` predates this; regenerating it is the next run.)

### Changed
- **T4 anti-cliché is strict.** Admit only if the named-prior F1 < **0.5** (difficulty ≥ 0.5, down
  from the old 0.9 bar). Plus a **blind control**: the name+role-anonymized prior must stay near the
  chance floor (< 0.35) — else the structure itself is a cliché.

### Added
- **`Judge.prior_edges(spec, *, blind=True)`** — guesses with names anonymized to `X1..Xn` *and roles
  hidden*, mapping edges back to the real names. Isolates name-leak from role-leak from chance.
- **`adversarial` author complexity tier** — instructs the author to make the obvious name-based guess
  *wrong* via phantom edges (no edge where one is "obvious"), reversed edges, and regime sign-flips,
  while keeping every declared edge detectable (so the v0.15 faithfulness gate still holds — the lever
  is making priors *wrong*, not making true edges invisible).
- The name-only eval is now a **3-tier certificate**: named / name-blind / name+role-blind vs the
  chance floor, with bootstrap CIs.

### Notes
- The faithfulness gate (v0.15) and the anti-cliché lever are deliberately compatible: suppressor /
  near-cancellation tricks that hide a true edge are *excluded* (an invisible edge isn't a fair
  target). 120 tests, 95% coverage. Advances #12.

[0.19.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.19.0

## [0.18.0] — 2026-06-23

**Conversational world elicitation — the "describe a world" UX (issue #3).**

A one-shot prompt is underspecified. Interactive use now runs a dialogue first: the elicitor asks the
*minimal* clarifying questions against a brief-completeness checklist, shows the accumulating brief,
and authors only once it's complete (or the user says "go").

### Added
- **`brief.WorldBrief`** — structured intent (domain, variables+roles, relationships, regimes, hidden
  causes, objective) with a completeness checklist (`missing_fields`, `is_complete`) and prose
  `render`. Pure core.
- **`Elicitor` Protocol** + **`elicit.Session`** state model (`start_session` / `respond` /
  `force_ready`) and the **`ClaudeElicitor`** adapter (the *author* model family — judge-independence
  applies only to grading). `FakeElicitor` for keyless tests.
- **`causal-worlds elicit OUT`** — interactive clarify loop → author → gate → bundle, reusing the
  existing generate pipeline.

### Notes
- The elicitor uses the author model (it helps you *specify*, not grade). The brief renders to the
  same prose the one-shot author consumes, so the author→gate→grade spine is unchanged.
- 117 tests, 95% coverage. Closes #3.

[0.18.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.18.0

## [0.17.0] — 2026-06-23

**Name-only baseline (#9, final piece) — and it caught a real anti-cliché leak.**

The last #9 control: an LLM that guesses the graph from variable names alone (no data) must score near
chance, or the benchmark tests memorized priors, not discovery. It does **not** — an honest negative.

### Added
- **`anonymize.anonymize_spec(spec)`** — relabels every variable to `X1..Xn` while preserving roles,
  mechanisms, and structure exactly (the Caliper-style control; pure + bijective).
- **`evals/name-only-baseline`** — directed F1 of the name-only LLM guess (named + anonymized) vs the
  truth, against a random-graph chance floor, with bootstrap CIs.

### Finding (honest negative)
- On `benchmark/v0.5` (35 worlds, gemini-2.5-flash): name-only **F1 0.71** [0.68, 0.74] vs a **0.20**
  chance floor — the worlds are substantially guessable from names. **Anonymizing** names only drops
  it to **0.61** [0.58, 0.65], so the leak is through variable names *and* role labels + graph
  conventions, not names alone.
- **Implication:** the T4 anti-cliché gate (rejects only at prior-F1 ≥ 0.9) is too lax. Tightening it
  and authoring worlds where priors actively mislead is the next milestone (#12).

### #9 status
Grader-independent admission (v0.15), information-fair crossover + CIs (v0.16), and this name-only
control (v0.17) are all done. The control's job was to *measure* anti-cliché honestly — it did, and
surfaced that the claim is currently weak. Closing #9; opening #12 for the difficulty/gate work.

[0.17.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.17.0

## [0.16.0] — 2026-06-23

**Information-fair crossover + bootstrap CIs (#9) — the identifiability finding, made fair.**

The crossover gave the reference grader interventional data while PC/FCI got only observational, so
"interventions vs not" was confounded with "method." Now every method can be given the **same
interventional budget**, and the result is sharper and fairer.

### Added
- **`baselines.pooled_interventional_sample(substrate, n, seed)`** + an `interventional=` flag on the
  causal-learn discoverers: pool observational + per-variable `do()` environments so PC/FCI/GES get
  the *same interventional budget* as GIES and the reference. (Shared env-builder refactored out of
  GIES.)
- Crossover eval now reports an **observational track** and an **interventional track** (`pc+do`,
  `fci+do`), the **interventional advantage** ΔF1 = F1(reference) − F1(method) with **percentile
  bootstrap CIs** (the primary, non-tautological signal), and difficulty-vs-error with bootstrap CIs.

### Finding (sharper + honest)
- Given the **same interventional budget**, causal-sufficiency methods *still* keep the hidden-
  confounded pair as causal: `pc+do` **15.0** (vs observational `pc` 14.3 — interventions alone don't
  help), `gies` 17; only the latent-aware `interventional-ci` reaches **0**. The lever is
  **latent-awareness, not interventions** — now demonstrated information-fairly.
- Interventional advantage robust: ΔF1 = **+0.29, 95% CI [0.22, 0.35]** for `pc+do`; every method's
  CI excludes 0.
- Difficulty-vs-error is **descriptive, not predictive**: observational methods r≈0.40 (CIs just
  exclude 0), the reference flat (r≈0.24, CI includes 0). Stated with CIs, not cherry-picked.

### Still open (#9)
- Name-only-LLM-baseline-at-chance (needs keys); temporal T3 decoupling.

[0.16.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.16.0

## [0.15.0] — 2026-06-23

**Grader-independent admission — the circularity fix (#9, the credibility keystone).**

The benchmark was circular: gate T3 admitted a world iff the *reference interventional-CI grader
recovered it*, so the benchmark was the set of worlds that grader solves — and we then reported that
grader winning. T3 is now a property of the **declared SCM itself**, computed in closed form.

### Added
- **`admission`**: `check_faithfulness(spec)` and `population_covariance(spec)`. For the
  linear-Gaussian SCM, the population covariance is `(I-B)⁻¹ Ω (I-B)⁻ᵀ`; a declared edge is
  **faithful** iff it induces a detectable partial correlation (`>= 0.05`) given the target's other
  observed parents — i.e. the parameters don't cancel and hide a true edge. Partial correlations are
  scale-invariant, so this holds for the iSCM data the substrate emits. Regime edges are checked
  structurally (the regime must genuinely change a coefficient). Plus `is_nontrivial(spec)`.

### Changed
- **Gate T3 is now grader-independent.** A world is admitted iff its SCM is faithful + non-trivial by
  construction — **no discovery method is run to decide admission**. The reference grader is still
  run, but only to *report* its score on the independently-admitted world.

### Validated
- All 35 cross-sectional `benchmark/v0.5` worlds remain admitted (min partial correlation 0.187 across
  the set, well above the 0.05 floor) — decoupling admission did **not** shrink or reshape the
  benchmark, so the crossover finding stands on a now-non-circular set.
- New decoupling test: a world stays admitted even when the supplied discoverer recovers **nothing**
  (`test_admission_is_grader_independent`). Degenerate/cancelling edges and spurious regime edges are
  rejected.
- 104 tests, 97% coverage.

### Still open (#9)
- Temporal T3 still uses a PCMCI+ recovery floor (grader-dependent) — cross-sectional is decoupled.
- Information-fair crossover; difficulty CIs/bootstrap; name-only-LLM-baseline-at-chance.

[0.15.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.15.0

## [0.14.0] — 2026-06-23

**R²-sortability — the scale-invariant leak v0.13 couldn't reach, and the iSCM fix for it.**

The v0.13 post-hoc standardization closed the *variance* leak but not the second, scale-invariant one
(Reisach et al. 2023): a variable's predictability from the rest (R²) also grows along the causal
order, and standardization cannot remove it. We measured it, found it, and fixed it properly.

### Added
- **`controls.r2sortability(data, edges, names)`** — the scale-invariant analogue of varsortability
  (Reisach et al. 2023), and **`R2SortnregressDiscoverer`**, the matching trivial baseline. Both are
  now reported in `evals/varsortability` alongside the variance versions — the bar a 2026
  causal-discovery reviewer expects.

### Changed
- **The substrate now uses internal standardization (iSCM, Ormaniec et al. 2024)** for cross-sectional
  worlds: each continuous variable is z-scored *as it is generated*, in topological order, so neither
  variance nor R² can compound along the causal order. Temporal worlds (where per-step standardization
  is ill-defined) keep post-hoc column z-scoring. Regimes stay `{0,1}`.
- Re-sampled `benchmark/v0.5` data under iSCM.

### Finding (honest)
- Under the v0.13 post-hoc fix, **R²-sortability was 0.73** and the trivial **R²-sortnregress baseline
  scored F1 0.40** — the worlds still leaked via the scale-invariant axis. iSCM drops **R²-sortability
  to 0.60** and **R²-sortnregress to F1 0.37**, with varsortability holding at 0.54; both trivial
  baselines now sit well below the real methods. The residual R²-sortability (0.60 > 0.5) is
  **disclosed, not yet fully closed**.

### Validated
- The reference **interventional-CI grader still recovers `coffee` at SHD 0 / F1 1.0** across 5 seeds
  under iSCM.
- The crossover is **robust to iSCM**: grader confounded-kept **0** (skeleton-SHD 1.44, F1 0.91),
  while PC/FCI/GIES keep 10–17 confounded pairs — the identifiability finding is unchanged.

[0.14.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.14.0

## [0.13.0] — 2026-06-23

**Variance standardization — fixes the flaw v0.12 caught.**

### Changed
- **The substrate now standardizes emitted data by default** (`build_substrate(spec, standardize=True)`):
  continuous columns are z-scored, **regime/binary columns are left as `{0,1}`** (standardizing them
  would break the grader's regime-stratification, and they carry no variance giveaway). This removes
  the additive-noise varsortability leak.
- Re-sampled `benchmark/v0.5` data to standardized.

### Validated
- **Varsortability 0.94 → 0.58** (near chance) and the trivial **sortnregress baseline F1 0.74 → 0.29**
  — the variance giveaway is gone.
- The reference **interventional-CI grader still recovers `coffee` at SHD 0 / F1 1.0** across seeds
  (the regime sign-flip edge survives, thanks to leaving regimes un-standardized).
- The crossover holds on standardized data: grader confounded-kept **0**, PC/FCI unchanged
  (scale-invariant), GIES still keeps the confounded pair — the identifiability finding is robust.

[0.13.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.13.0

## [0.12.0] — 2026-06-23

**Synthetic-DAG control — and it caught a real flaw.**

### Added
- **`controls`**: `varsortability(data, edges, names)` (Reisach et al. — how much a world leaks its
  causal order through marginal variance) and `SortnregressDiscoverer` (the trivial sort-by-variance
  baseline). Plus `evals/varsortability`.

### Finding (honest, action item)
- On `benchmark/v0.5`, mean **varsortability 0.94** and the trivial **sortnregress baseline scores
  F1 0.74** — better than PC/FCI. The worlds currently **leak the causal order via marginal variance**
  (the additive-noise giveaway). The fix — **variance standardization** of emitted data — is the next
  release (#9); it should drop varsortability to ~0.5 and collapse sortnregress while leaving the
  scale-invariant methods (PC/FCI) and the interventional grader unaffected.

[0.12.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.12.0

## [0.11.1] — 2026-06-23

### Fixed
- **Empty Langfuse traces.** Spans were attaching `metadata` but not `input`/`output` — which is the
  content the Langfuse UI shows — so traces arrived looking empty. The `Tracer` seam now records span
  **input** on open and **output** via `record(...)`; `generate`/`author`/`gate` spans carry the
  prompt, attempt, recovered variables, and admit decision. Verified live against the API.

[0.11.1]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.11.1

## [0.11.0] — 2026-06-23

**Integrity pass** (from an adversarial self-audit + a 2026 landscape review). No new capability — this
makes the existing claims honest and the comparison fairer.

### Fixed
- **Baseline seed-aliasing**: `GiesDiscoverer` now draws an *independent* seed per environment (it had
  shared one seed across the observational + every intervention env, aliasing their noise — the same
  bug the reference grader avoids). This was unfairly inflating GIES error: mean skeleton-SHD 6.66 →
  **2.37** on `benchmark/v0.5`.
- **Crossover report provenance**: the benchmark name was hardcoded `"v0.2"`; it now records the actual
  set (`v0.5`).

### Changed (honest reframing)
- The crossover is now framed as an **identifiability finding, not "defeats the standard toolbox."**
  GIES gets the same interventional budget as the reference and recovers structure well, yet — assuming
  causal sufficiency — still reports the hidden-confounded pair as causal (17/35); PC/FCI likewise. The
  lever is **latent-awareness**, not interventions. README + blog updated.
- **Difficulty**: report the partly-mechanical structural-vs-observational-error correlation (r≈0.8)
  *and* the non-tautological structural-vs-interventional-advantage (ΔF1, r≈0.36, n=35, no CIs) — as a
  descriptive axis, not a validated predictor.
- Surfaced the **admission-circularity** (worlds are admitted by the reference grader itself) as a
  known limitation with a planned fix (grader-independent admission + an information-fair crossover +
  varsortability/variance-standardization control + a name-only-at-chance baseline).

[0.11.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.11.0

## [0.10.0] — 2026-06-23

**Observability.** See every run as a trace.

### Added
- **`LangfuseTracer`** behind the existing `Tracer` seam (Langfuse / OpenTelemetry). `generate` now
  wraps `generate` → `author` → `gate` in spans with metadata (prompt, attempt); `flush()` sends them
  before a short-lived CLI process exits. Follows the Langfuse skill's best practices (load env before
  the client, explicit input/metadata, flush on exit).
- The container resolves the tracer from settings — Langfuse when `CAUSAL_WORLDS_LANGFUSE_ENABLED=true`
  (reads `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST`), else the no-op `NullTracer` (the default, so a
  misconfigured backend can never break a run).
- The CLI auto-loads a local `.env` (python-dotenv) so keys are picked up; a committed `.env.example`
  documents every variable.

[0.10.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.10.0

## [0.9.0] — 2026-06-23

**The temporal generative loop closes** — natural language → an admitted *temporal* world.

### Added
- **Temporal gating** (`gates`): `run_gates` detects a temporal world (any lagged term) and admits it
  via a temporal T3 — a TS reference (PCMCI+ by default, injectable) must recover the lagged structure
  above an F1 floor. `GateReport.temporal_grade` carries the lagged grade. T1/T2 work unchanged.
- **Author temporal mode**: `build_claude_author(temporal=True)` / `ClaudeAuthor(temporal=...)` adds a
  brief instructing lagged + autoregressive (stationary) terms, so the LLM can author temporal worlds.
- `generate` threads a `temporal_discoverer` through to the gate; `FakeTemporalDiscoverer` for keyless
  tests.

### Validated (live)
- Claude (temporal mode) authored a reservoir operation — rainfall→inflow at lag 1, autoregressive
  storage/inflow, a hidden `soil_saturation` confounder, a snowmelt regime — **admitted in 1 attempt**
  through the PCMCI+ temporal gate (F1 0.56). Next: a scaled temporal benchmark set + crossover at n>1.

[0.9.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.9.0

## [0.8.1] — 2026-06-23

### Fixed
- **Temporal worlds now round-trip through persistence.** The pydantic boundary `TermModel` carried
  no `lag`, so serializing a temporal spec (`spec_to_json` / `save_bundle`) silently dropped the lags.
  `TermModel.lag` is added (and threaded through `to_term`/`from_spec`), so lagged worlds persist
  faithfully — and the LLM author can now express lagged edges.

[0.8.1]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.8.1

## [0.8.0] — 2026-06-23

**Temporal grading.** Time-series discovery can now be benchmarked against the lagged ground truth.

### Added
- **Temporal scoring** (`evaluation`): `temporal_score` → `TemporalReport` (temporal SHD + F1 over
  `(src, dst, lag)` edges, with lag-0 reversals handled).
- **`TemporalDiscoverer`** protocol + **`grade_temporal_spec`** — grade any TS method against a
  world's `temporal_answer_key`.
- **TS baselines** (`temporal_baselines`, the `temporal` extra): PCMCI+ / LPCMCI (tigramite),
  VARLiNGAM (lingam), Granger (statsmodels) — behind adapters, lazy-imported, with pure tested parsers.
- **Temporal crossover eval** (`evals/temporal-crossover`) on the `supply` world.

### Findings (n=1, honest)
- The grading stack is validated: **PCMCI+ recovers `supply`'s lagged structure exactly** (temporal
  SHD 0, F1 1.0). The hidden-confounder trap is **method-specific** temporally — only VARLiNGAM keeps
  the spurious `leadtime~cost` edge; PCMCI+/LPCMCI/Granger don't. Unlike the cross-sectional crossover,
  the TS toolbox does not *uniformly* fail on this single world — a temporal benchmark *set* is needed
  to characterize it (next).

[0.8.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.8.0

## [0.7.0] — 2026-06-23

**Temporal worlds (foundation).** Worlds can now carry *time* — lagged edges and autoregression —
not just cross-sectional structure. (Time-series *grading* + baselines land next.)

### Added
- **Lagged IR** (`schema`): `Term.lag` (default 0). Only the contemporaneous (lag-0) subgraph must be
  acyclic; lagged edges — including autoregressive self-loops — are valid (they read the past).
- **Temporal substrate** (`sample`): when any lag is present, sampling becomes sequential over
  timesteps with a burn-in (near-stationary); `do()` interventions hold across time. Cross-sectional
  worlds keep the original vectorized i.i.d. path unchanged.
- **`temporal_answer_key(spec)`** → lagged ground truth `(src, dst, lag)` incl. autoregression; the
  summary `answer_key` now collapses lags and drops self-loops, so existing tooling is unaffected.
- **Built-in `supply`** — a temporal world (autoregressive lead time + inventory, a hidden logistics
  confounder), in a separate registry (`worlds.temporal_names()`) so the still-contemporaneous CLI
  `grade`/`gate` don't mis-score it.

[0.7.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.7.0

## [0.6.1] — 2026-06-23

### Docs
- Rewrote the README around a getting-started flow (honest shipped-vs-roadmap; the gym/temporal/
  counterfactual claims are now roadmap, not overclaims), with a lead example, the measured crossover
  result, install/extras, concepts, and a roadmap.
- Added a guided [`docs/getting-started.md`](docs/getting-started.md) and runnable
  [`examples/`](examples/) (grade-your-discoverer, inspect-a-bundle — keyless — and author-a-world).

[0.6.1]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.6.1

## [0.6.0] — 2026-06-23

**Use the benchmark.** Grading your own discoverer against a shipped world is now a first-class,
typed, tested feature — the package's whole purpose.

### Added
- **`bench`**: `grade_spec(spec, discoverer)` and `grade_bundle(bundle_dir, discoverer)` → a `Report`
  scoring any `Discoverer` against a world's declared answer-key (defaults to the reference grader).
- **CLI `score`**: `causal-worlds score <bundle> [--discoverer module:Class]` grades a discoverer
  (the reference by default, or any importable one) on a persisted world.
- **Typed distribution**: ship a PEP 561 `py.typed` marker, plus PyPI metadata (classifiers,
  keywords, project URLs).

[0.6.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.6.0

## [0.5.0] — 2026-06-23

**Scale resolves the difficulty question.** A 36-world set across an easy→hard complexity spread gives
the analyses real range — and structural difficulty turns out to predict the observational collapse.

### Added
- **Author complexity knob** (`author`): `ClaudeAuthor(..., complexity="easy"|"standard"|"hard")`
  varies how many hidden confounders / regime sign-flips to inject, spreading structural difficulty.
  Recorded per world in the manifest (`Provenance.complexity`).
- **Scaled benchmark** (`benchmark/v0.5`): 35/36 admitted across complexity levels — mean structural
  difficulty by level 0.0 / 1.4 / 3.0; reference-grader SHD 0.36 / 1.75 / 2.33.
- **Parameterized evals**: the crossover and structural-difficulty harnesses take a benchmark dir;
  results nest under `evals/*/v0.5/`.

### Findings (powered, n=35)
- **Crossover strengthens**: the interventional-CI grader keeps **confounded-kept = 0** (never reports
  a hidden-confounded pair as causal) at SHD 1.47 / F1 0.91, while PC/FCI/GIES keep 8–17 and post SHD
  2.7–6.7.
- **Structural difficulty predicts observational error (corr +0.62)** where name-guessability does not
  (+0.14) — the hardness is structural (confounders + sign-flips), resolving v0.4's open question and
  turning difficulty into a usable instrument.

[0.5.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.5.0

## [0.4.0] — 2026-06-23

**A structural-difficulty axis.** v0.3 showed name-guessability difficulty doesn't predict discovery
error — the hardness is structural. This adds that axis and tests it honestly.

### Added
- **Structural difficulty** (`difficulty`): `structural_difficulty(spec)` scores discovery-hardness
  from the structure — hidden confounders, confounded pairs, regime **sign-flips**, edge density — with
  a headline trap-count `score`. Pure, deterministic, unit-tested.
- Structural difficulty is now recorded in every admitted world's `manifest.json`.
- **Re-analysis** (`evals/structural-difficulty`): reuses the crossover report (no new runs) to test
  whether structural difficulty predicts the collapse.

### Findings (honest)
- At n=12 with a narrow difficulty range, **neither** name-guessability nor structural difficulty
  cleanly predicts the *magnitude* of error (correlations −0.39…+0.14) — a statistical-power problem,
  not a refutation. The v0.3 crossover (standard methods collapse, grader holds) is unaffected.
  Resolving difficulty-predicts-error is deferred to the scaled set (v0.5).

[0.4.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.4.0

## [0.3.0] — 2026-06-23

**The decisive experiment.** Proves the benchmark's central claim beyond the single `coffee` world:
standard discovery collapses on our worlds where the reference interventional-CI grader holds.

### Added
- **Baseline suite** (`baselines`): PC, GES, FCI (`causal-learn`) and GIES (`gies`) wrapped behind the
  `Discoverer` Protocol as adapters — lazy-imported (the `discover` extra), so the package imports and
  CI run without them; graph-parsing logic is pure and unit-tested. `BaselineResult` carries directed
  edges, bidirected (confounding) marks, and the skeleton for a fair cross-method comparison.
- **Crossover eval** (`evals/baseline-crossover`): every benchmark world vs every method across seeds →
  skeleton-SHD, directed F1, and *confounded-pair-kept-as-causal* (the trap). **Result (n=12): GO.**
  Standard methods keep the hidden-confounded pair as causal in 7.3–10.0 of 12 worlds (PC/FCI/GIES) and
  post 2–4× the skeleton error; the interventional grader stays at confounded-kept 0.33, SHD 1.31,
  F1 0.91.
- **Difficulty-vs-error analysis** — *honest negative*: name-guessability difficulty does not yet
  predict discovery error (corr ~0.1); the hardness is structural (confounder+regime). Sharpens v0.4.
- **Publication artifacts**: a technical blog post (`docs/blog-the-decisive-experiment.md`) and a
  Framing-B paper skeleton (`paper/`).

### Notes
- `causal-learn`'s GES is numpy-2 incompatible (errors on every world) — reported, not hidden.

[0.3.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.3.0

## [0.2.0] — 2026-06-23

Closes the generative loop: **natural language in, an admitted causal world out**, plus persistence
and a shipped benchmark set. The LLM seams are real but isolated — the package still imports and CI
still runs with no API key (the adapters are unit-tested against fakes).

### Added
- **NL author** (`author`): `ClaudeAuthor` turns a plain-language operation into a `WorldSpec` via
  `instructor` (bounded re-ask), steered toward recoverable, anti-cliché worlds (a hidden confounder
  + a regime sign-flip). Behind the `Author` Protocol; provider SDK lazy-imported.
- **Independent judge** (`judge`): `GeminiJudge` guesses the structure from names/roles alone (the
  anti-cliché signal) and scores faithfulness — a *different model family* than the author.
- **T4 anti-cliché gate** (`gates`): with a judge + prose, rejects unfaithful or guess-from-priors
  worlds and records a `difficulty` score (`1 - F1(judge_prior, truth)`).
- **The loop** (`generate`): `generate` (author→gate→admit with feedback-driven re-author) and
  `generate_many` (never-raising batch) → `AdmittedWorld`.
- **Artifact persistence** (`artifact`): self-describing on-disk bundle (`spec.json` / `data.npz` /
  `answer_key.json` / `manifest.json`) with full provenance (models, grader version, seed, grade).
- **Boundary model** (`serde`): one pydantic `WorldSpecModel` — the author's output target and the
  persisted JSON shape — converting to/from the frozen core IR.
- **CLI**: `generate <prompt> <out>` and `benchmark <prompts_file> <out>`; author/judge resolved
  through the DI container.
- **Author-model bake-off** (`evals/author-model-bakeoff`): a reproducible, judged comparison that
  picks the default author model with numbers, not assertion — shipped with the release.
- **Benchmark set** (`benchmark/v0.2`): 12 authored, admitted worlds across distinct operations —
  mean difficulty 0.28, faithfulness 1.00, reference-grader directed SHD 1.25 / F1 0.92.

### Changed
- Version is single-sourced from `_version.py` (hatchling dynamic). `.coverage` is no longer tracked.

[0.2.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.2.0

## [0.1.0] — 2026-06-22

First release: **the deterministic benchmark engine**. Generate (programmatically-specified) fictional causal
worlds with a ground-truth answer-key, sample them, grade a causal-discovery method, and score it — runnable as a
library and a CLI, with no LLM or API key required.

### Added
- **Schema / IR** (`schema`): `WorldSpec` as the single source of truth (variables incl. hidden confounders +
  generative `Mechanism`s with regime-switching); the `AnswerKey` (observed edges + confounded pairs) is *derived*,
  never stored; `validate()` static gate.
- **SCM substrate** (`sample`): a deterministic, seeded executable world; `do()` interventions (constant or
  per-row array). The functional core.
- **Reference grader** (`discover`): `InterventionalCiDiscoverer` — a spec-blind interventional-CI discoverer that
  recovers the confounder + regime-flip trap (directed SHD 0) where standard observational/score-based methods
  (PC, GES, GIES, FCI) fail.
- **Scoring** (`evaluation`): directed/skeleton SHD, F1, and `confounded_reported` (flags a causal edge claimed for
  a hidden-confounded pair); `Report`.
- **Validity gates** (`gates`): `run_gates` → T1 (validity) · T2 (sample-sanity) · T3 (non-triviality vs a
  per-world random-graph null). Admits only if all pass.
- **Built-in worlds** (`worlds`): `coffee` (the confounder + regime-flip trap) and `ecommerce` (easy control).
- **CLI** (`causal-worlds`): `version` · `worlds` · `grade <world>` · `gate <world>`.
- **Wiring**: pydantic-settings `config`, a small DI `container`, and a no-op `Tracer` observability seam.
- **Quality**: uv + ruff (`select=ALL`) + mypy `strict` + pytest with a coverage floor, enforced by CI.

### Not yet (tracked as v0.2 issues)
NL/`WorldBrief` → spec **author**, the independent **Gemini judge** + the T4 anti-cliché gate, conversational
**elicitation**, the **Langfuse (OTEL)** tracing adapter, **artifact/manifest persistence**, grader **hardening**
(FCI-with-interventions) + world-diversity sweep + knob calibration, and more built-in/temporal worlds.

[0.1.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.1.0
