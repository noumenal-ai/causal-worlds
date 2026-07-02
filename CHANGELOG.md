# Changelog

All notable changes to causal-worlds are documented here. Format: [Keep a Changelog](https://keepachangelog.com/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.37.0] — 2026-06-30

**The prompt's claim mapping (`Claim`, #30) + missing top-level exports fixed (#31).** Driven by the
first downstream app built on `generate()`: a UI could not know which spec variables embodied the
user's question, and the documented builder imports failed.

### Added
- **`Claim(cause, outcome)` on `WorldSpec` (#30)** — the author now declares which *observed*
  variables embody the prompt's causal question (its cause X and outcome Y), so a downstream UI can
  quiz the *intended* pair instead of guessing it from the answer key (and guessing wrong — e.g.
  *"does nature or nurture create psychopaths?"* previously got quizzed on a confounded side pair
  that dropped the user's outcome entirely).
  - Declared by the author (new system-prompt clause + `ClaimModel` at the pydantic boundary);
    optional — `None` for worlds with no focus pair (e.g. `world_from_edges`).
  - Validated in the T1 gate: cause and outcome must be **two distinct observed (non-hidden)**
    variables; violations raise the new **`ClaimError`**.
  - Persists through `spec.json` (full JSON round-trip) and is exposed as **`AdmittedWorld.claim`**.

### Fixed
- **Top-level exports (#31)** — `build_claude_author` and `build_gemini_judge` were defined but never
  exported from `causal_worlds`, so the documented `from causal_worlds import build_claude_author, …`
  failed (`ImportError`). Both are now top-level, along with `ClaudeAuthor`/`GeminiJudge` (parity with
  `ClaudeElicitor`) and the new `Claim`/`ClaimError`/`ClaimModel`; a regression test locks it in.

## [0.36.0] — 2026-06-30

**Stage-2 control: reference baseline controllers + an observational-identifiability check (#27).**
Makes the control track as turn-key as discovery.

### Added
- **Reference baseline controllers** — the control analogue of the `sortnregress` discovery baselines,
  two calibration points that bracket what the levers can achieve:
  - **`CorrelationalController`** — sets levers from the *observational* regression of the outcome on
    the levers ("control by correlation"). On a world with a hidden lever↔outcome confounder its
    estimate is biased, so **high regret here is the signal that the world rewards causal
    understanding**. The lower bracket.
  - **`InterventionalController`** — sets levers from a `do(+1) − do(0)` contrast per lever (the
    do-calculus reference / achievable ceiling). On confounded worlds the two diverge sharply.
- **`is_control_identifiable(spec, objective)`** → per-lever `LeverIdentifiability` (`.identifiable`,
  `.adjustment_set`): whether a lever's effect on the outcome is recoverable from **observational**
  data by backdoor adjustment. Uses the complete adjustment criterion (the canonical observed
  ancestor-based set d-separates the lever from the outcome in the backdoor graph). A **hidden**
  common cause of lever and outcome ⇒ non-identifiable (no observed set works); an **observed**
  confounder ⇒ in the adjustment set; a **mediator** ⇒ a descendant of the lever, correctly excluded.
  Lets a harness split worlds into observationally-solvable vs interventions-required, and grade an
  observational controller only where that's even possible.

### Notes
- Both work on the contemporaneous (lag-0) structure the control optimum lives on; no API key.
  Backward compatible (purely additive). 220 tests, 96% coverage. Closes #27.

[0.36.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.36.0

## [0.35.0] — 2026-06-28

**Additive-nonlinear mechanisms — worlds can finally bend, not just slope (#10, first cut).**

### Added
- **`Transform`** — an elementwise nonlinearity on a `Term`, applied before its coefficient, so a
  mechanism becomes a **generalized additive model**: `X = Σ coeffᵢ·fᵢ(parentᵢ) + noise`. Members are
  `identity` (the linear-Gaussian default — existing worlds are byte-for-byte unchanged),
  `square`, `cube`, `tanh`, `relu`, `abs`. Every member is **total over the reals** (safe on
  standardized, possibly-negative data — no NaNs) and serializes to a string, so a world stays a pure
  declarative spec. `Term` gains a `transform` field (default `IDENTITY`); `has_nonlinear_terms(spec)`
  reports whether any mechanism is nonlinear.
- **Built-in `braking` world** — auto-emergency-braking physics whose `braking_distance` follows the
  kinematic **`speed²`** law. On standardized, symmetric speed data its linear correlation with speed
  is ≈ 0, so a linear/PC discoverer **drops the `speed → braking_distance` edge** (F1 0.6) while the
  interventional-CI reference recovers it (F1 1.0) — the edge was in the answer-key the whole time.
  Keeps the classic trap too (a hidden `weather` confounds `road_grip` ⟂ `visibility`).
- **`examples/06_nonlinear_world.py`** — the contrast above, runnable, with its expected output.
- **Stationarity gate** — `validate` (the T1 gate) now rejects *explosive temporal feedback* before
  any sampling: an unbounded nonlinear transform (square/cube/relu/abs) on **any** lag-collapsed
  cycle — a lagged self-loop *or* a multi-variable cycle (`a_t → b_{t+1} → a_{t+2}`) — has no
  stationarity guarantee, and a linear self-loop with total load `Σ|coeff| ≥ 1` is explosive; both
  raise `NonStationaryError`. A **bounded** transform (`tanh`) on a cycle is admitted at any
  coefficient, an unbounded transform *off* every cycle (e.g. a feed-forward `square`) is fine, and
  `Σ|coeff| < 1` linear AR is unchanged. This turns a silent `inf` during sampling into a clear
  failure at authoring time (`_BOUNDED`/`Transform.is_bounded` encode the rule; the author prompt
  states it). Linear *multi-variable* cycles (a spectral-radius question) are left to the author.

### Why
- The substrate was **linear-Gaussian only**, exactly where varsortability-style artifacts and
  trivial baselines bite hardest, and a narrow place to evaluate a sophisticated discoverer (incl.
  the private noumenal_agent eval). Real mechanisms curve (`v²`, saturation, thresholds); a single
  coefficient can only scale, never bend. This is the *additive nonlinear* form issue #10 names.

### Design / honesty
- **Additive noise on purpose:** it keeps abduction closed-form (`noise = factual − f(parents)`), so
  counterfactuals stay **exact** on nonlinear worlds; and a transform changes *how* a parent acts, not
  *whether* it does, so the **answer-key edge set is unchanged** and grading stays
  correct-by-construction. Verified: nonlinear sampling, exact nonlinear counterfactuals, serde
  round-trip, anonymization, and the temporal path all honor transforms.
- **Still open (documented, not hidden):** interaction terms (products of two parents, e.g. friction
  `μ·N`), ratios (`v²/a`), and a post-nonlinear `g(Σ·)+noise` form. The closed-form **control optimum
  stays linear-only** — a nonlinear world now raises `NonlinearControlError` rather than returning a
  wrong LQ optimum, and **`ControlEnv` rejects a nonlinear world at construction** (its regret signal
  needs that optimum) instead of crashing mid-episode.

### Notes
- Backward compatible — `IDENTITY` is the default, so every existing world, bundle, and call site is
  unchanged. Validated across all six transforms and diverse world shapes (do()-fingerprints,
  exact counterfactuals incl. Pearl's consistency axiom, regime-switched and multi-transform
  mechanisms, temporal autoregression, a confounded end-to-end world; Pearl-marginalization
  cross-check of the CF engine vs the sampler; the stationarity gate incl. multi-variable cycles;
  gym + CLI + bundle-persistence integration): 213 tests, 96% coverage.

[0.35.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.35.0

## [0.34.0] — 2026-06-25

**Playground mode — author a world from any sentence without the benchmark's anti-cliché rejection (#19).**

### Added
- **`anti_cliche` flag** on `generate` / `generate_many` / `run_gates` (default `True`), surfaced on
  the CLI as **`causal-worlds generate … --playground`** (and `elicit --playground`). Benchmark mode
  (the default, unchanged) rejects worlds that are guessable from their variable names/roles — the
  published benchmark must not be name-guessable. **Playground mode** (`anti_cliche=False`) keeps the
  faithfulness check and still *reports* `difficulty`, but **never rejects on guessability**: the
  "describe a world and just get it" path the World Models product needs. T4 still only runs when a
  judge + prose are supplied.
- A strict-mode T4-cliché rejection now prints an **actionable hint** pointing at `--playground`
  instead of leaving a dead end.
- The bundle manifest records **`anti_cliche`** (provenance), so a playground world can never
  masquerade as benchmark-grade.

### Why
- The strict T4 anti-cliché gate (v0.19+) rejects *intuitive* operations on purpose — coffee, power
  grid, hospital ED, bike-share all let a name+role guesser recover ≥ 50% of the edges. That is right
  for building the benchmark and wrong for a playground; the two use cases were sharing one gate.
  Measured live (v0.33): grid, bike-share, mushroom-farm, coffee, and streaming prompts were all
  rejected, even after the full re-author budget. With `--playground`, the same grid prompt is
  admitted in 1 attempt (difficulty 0.16, advisory; faithfulness 1.0; reference grader F1 0.92).

### Fixed / honesty
- **`docs/validation.md`** and **`evals/author-model-bakeoff`** corrected: their "authoring is solved"
  numbers (5/5 first-try; 8/8 admit) predate the strict T4 gate (the bake-off ran on v0.2.0 at mean
  difficulty 0.30 — those worlds would be rejected today). The author pass-rate *under the strict gate*
  remains the open question `docs/scope.md §8` flags; playground mode is the product escape hatch, not
  a claim that the strict gate got easier.

### Notes
- Backward compatible — the default is unchanged, so the published benchmark and every existing call
  site keep strict admission. 163 tests, 95% coverage.

[0.34.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.34.0

## [0.33.0] — 2026-06-25

**Temporal counterfactuals — Rung 3 now covers lagged worlds too (the last gap closed).**

### Added
- **`counterfactual_temporal(spec, do, *, seed, steps=...)`** — a *trajectory* counterfactual under a
  **sustained** intervention: roll one realized noise series forward `steps` timesteps for the factual
  path, then re-roll the *same* noise with the lever held at its value throughout, holding all lagged
  dynamics and the noise series fixed. Returns per-observed-variable arrays (`TemporalCounterfactualResult`,
  with `.effect`). Handles autoregression, lags, and regimes.
- The cross-sectional `counterfactual` now points temporal worlds to it (clear error) instead of just
  refusing.

### Notes
- **Pearl's ladder is now complete for both substrates** — association, intervention, and
  counterfactual on cross-sectional *and* temporal worlds. Internally the topological order was made
  lag-0-aware (a lagged self-loop must not look like a within-step cycle). Verified on the built-in
  temporal `supply` world (holding `order` high cuts mean `stockout`). 157 tests, 95% coverage.
- `docs/scope.md` + `docs/foundations.md` updated — the temporal-counterfactual residual is resolved.

[0.33.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.33.0

## [0.32.0] — 2026-06-25

**Rung 3 — a counterfactual engine — completes Pearl's ladder; plus `do()` is now visualizable.**

### Added
- **`causal_worlds.counterfactual`** — counterfactuals on a declared SCM via Pearl's
  **abduction → action → prediction**: `counterfactual(spec, do, *, seed)` returns the **factual**
  world and what *would* have happened under `do`, on the *same* unit (noise held fixed), with a
  `.effect`. Exact because the SCM is declared (abduction recovers each unit's noise in closed form).
  Also the building blocks `abduct(spec, factual)` and `predict(spec, noise, do)`. Cross-sectional
  worlds for now; temporal raises `TemporalCounterfactualError`. **This completes the ladder: the
  package now stands on all three rungs (association · intervention · counterfactual).**
- **`to_mermaid` / `to_dot` take a `do=` argument** — render the *post-intervention* graph: the
  intervened variable is shown **set**, with its **incoming edges cut** (and effects kept). Graph
  surgery you can see. `causal-worlds viz` gains nothing new here, but the renderers do.

### Changed
- README **"Causality in three rungs"** now *shows* the rungs: a Rung-2 graph-surgery figure
  (`do(footfall)`, generated from `to_dot(..., do=...)`) and a runnable Rung-3 `counterfactual`
  snippet. `docs/scope.md` updated — Rung 3 is now shipped (cross-sectional); only temporal
  counterfactuals remain.

### Notes
- The counterfactual engine is a *structural* query on the raw equations (autonomy/modularity is
  what licenses holding the other mechanisms fixed). Tests cover the consistency axiom, exact
  fixed-noise propagation, determinism, the hidden-confounder case, and the temporal guard.
  154 tests, 95% coverage; renderers stay zero-dependency.

[0.32.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.32.0

## [0.31.0] — 2026-06-25

**Path coefficients on the graphs, a single-glance hero image, and a README that teaches causality.**

### Added
- **Edge labels now carry the Wright path coefficient** in `to_mermaid` / `to_dot` (e.g.
  `footfall --0.4--> sales`). Two coefficients (`-1/1`) show a regime **sign-flip**; `lag k` still
  marks temporal edges. A diagram with strengths *is* a path diagram, not just a sketch.
- `to_dot` now colors the **hidden confounder's out-edges red** (dashed) so the confounding pops, and
  sets clean graph/edge fonts — its output lays out cleanly under Graphviz `dot`.
- **README "Causality in three rungs"** — a 2-minute, plain-language tour of Pearl's Ladder
  (association → intervention → counterfactual) on the `coffee` world, with runnable Rung-1/Rung-2
  snippets. The mission: make causality legible to a newcomer from the README alone.

### Changed
- The README **hero image is now generated from the package's own `to_dot`** via Graphviz (proper
  layered routing + a colored role legend + title) — a genuine single-glance DAG, and it dogfoods the
  renderer. (`docs/figures/render_world_dag.py` rewritten; matplotlib/networkx no longer needed.)

### Fixed / honesty
- **`docs/scope.md` corrected:** the package supports **Rung 1 (association)** and **Rung 2
  (intervention — `do()`, verified to be *genuine graph surgery*, not conditioning)**; the control
  track (optimal policy, regret, regret-under-perturbation, Gym) is shipped — but **Rung 3
  (counterfactuals) is NOT implemented**. The scope had listed a counterfactual engine as a Stage-2
  output; it is the *next* piece, not a current capability. No more overclaiming the top rung.

### Notes
- Renderers stay zero-dependency (pure strings); Graphviz is only a docs-figure tool, not a package
  dep. 146 tests (renderers at 100%), 95% coverage.

[0.31.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.31.0

## [0.30.0] — 2026-06-25

**Legible variable names in the built-in worlds — the rendered hero graph now reads on its own.**

### Changed
- **`coffee` world renamed for self-explanation:** `R → weekend` (the regime that lifts demand and
  *inverts* price sensitivity), `L → local_buzz` (the hidden confounder — unobserved local activity
  driving footfall, overtime, and sales together), `foot → footfall`. Cryptic single letters (`R`,
  `L` — causal-textbook shorthand) defeated the whole point of a showcase graph: if the author has to
  ask what `R` and `L` are, so will everyone. The structure, mechanisms, regime sign-flip, and
  answer-key (confounded pair `overtime ~ sales`) are **unchanged** — only the names.
- **`supply` (temporal) world:** `L → logistics` (same rationale).
- README hero PNG re-rendered with the new names (wider boxes for the longer labels); the
  `02_visualize_a_world` example + `examples/README.md` Mermaid blocks updated to match.

### Notes
- Built-in test fixtures that construct their *own* coffee-like specs (not `worlds.get("coffee")`)
  keep their terse names — they test mechanics, not the user-facing showcase. 144 tests, 95% coverage.

[0.30.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.30.0

## [0.29.0] — 2026-06-25

**See the world it builds — zero-dependency SCM graph renderers, a richer examples set, and a
slimmed README.**

### Added
- **`to_mermaid(spec)` and `to_dot(spec)`** (`causal_worlds.viz`) — pure-string renderers (no extra
  deps) turning a `WorldSpec` into a Mermaid flowchart (renders natively on GitHub) or Graphviz DOT.
  Visual grammar: controllable / outcome / observable / disturbance / **hidden confounder** each get a
  distinct shape + colour; lags are edge labels; edges out of a hidden node (and regime-only edges)
  are dashed — so the latent structure a discovery method never sees is exactly what you look at.
- **`causal-worlds viz <world>`** CLI command (`--format mermaid|dot`); accepts a built-in world name
  **or** a persisted bundle directory.
- A **rendered hero image** in the README (the `coffee` SCM, hidden confounder dashed) via
  `docs/figures/render_world_dag.py` — embedded by absolute URL so it shows on PyPI too.

### Changed
- **Examples overhauled** (`examples/`): 5 focused scripts (grade · **visualize** · control · inspect ·
  author), and **every example's docstring now carries its expected output** so you get a feel without
  running. `examples/README.md` indexes them with outputs (incl. a live Mermaid diagram).
- **README slimmed** (254 → ~150 lines): the visualization leads; the full crossover table + the four
  honest caveats moved to **`docs/findings.md`** (nothing lost, just linked); the roadmap is trimmed;
  the stale `v0.6` status is corrected.

### Notes
- 144 tests (5 new for the renderers), 95% coverage. The renderers are in the base install (no new
  runtime deps); the README figure uses matplotlib+networkx as a one-off doc tool (not a package dep).

[0.29.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.29.0

## [0.28.0] — 2026-06-24

**Finite request timeouts on both LLM clients — a reset/stalled connection now fails fast instead of
hanging a socket read forever.**

### Fixed
- **Unbounded LLM-client reads could hang indefinitely.** Neither the Claude author nor the Gemini
  judge client set a request timeout, so a transient Anthropic 500 + "connection reset by peer" left
  a socket read blocked — stalling an overnight generation run for ~12 hours at 0% CPU on a single
  prompt. Both clients now carry a finite per-request timeout (`author`: `timeout=120s` +
  `max_retries=4` on the Anthropic SDK; `judge`: `HttpOptions(timeout=120_000ms)` on google-genai),
  so a bad connection raises promptly and the existing retry loops (judge `_create`, generation
  resilience) actually get to retry it. Failure is fast and loud, never a silent hang.

### Changed
- **`evals/scale/generate_hardened.py` resume is now idempotent and crash-safe** (research script):
  the index is written after *every* world (a kill no longer loses progress); a prompt the gate
  already rejected is skipped on resume (no re-burning author spend on a deterministic reject); and a
  transient error on one prompt is recorded and skipped rather than crashing the whole run (it
  retries on the next resume). Matters on a fixed eval budget under flaky providers.

[0.28.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.28.0

## [0.27.0] — 2026-06-23

**The Gemini judge rides through transient provider overloads — so a 503 spike no longer wastes the
(paid) author call that produced the spec being judged.**

### Added
- **Bounded transient-retry in `GeminiJudge`.** Both judge calls (`prior_edges`/`_guess` and
  `faithfulness`) now route through a private `_create` that retries transient provider errors
  (Gemini 503 "high demand", `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, 429/rate-limit, timeouts) with
  exponential backoff (`retries=4`, `backoff=2.0` by default → rides through a ~30s spike).
  Classification is by error *string*, not exception type, to stay SDK-agnostic behind our seam.
  Non-transient errors (e.g. a malformed request) are re-raised immediately; a persistent outage
  still fails loud after the bounded retries.

### Why
- During temporal-world generation, a Gemini 503 propagated out of T4 and was caught by the
  generation-resilience loop, which retried the **whole** `generate()` — re-running the upstream
  Claude author call each time. Retrying the transient failure *inside the judge* preserves the
  already-paid spec across a spike, which matters on a fixed eval budget. (Mirrors the
  generation-resilience added in v0.25.)
- `backoff` is injectable (`backoff=0.0`) so the three new unit tests — retry-then-succeed,
  re-raise-non-transient, give-up-after-exhaustion — stay fast and need no real sleep. 139 tests.

[0.27.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.27.0

## [0.26.0] — 2026-06-23

**Temporal worlds now clear the T4 anti-cliché gate too — closing the gap that could invalidate a
name-dependence study on temporal data.**

### Fixed
- **Temporal worlds were admitted on recoverability alone.** `_temporal_gates` ran T3 (a TS reference
  recovers the lagged structure above F1 `_TEMPORAL_F1_MIN`) but **skipped T4** — so a temporal world
  whose graph is guessable from variable names/roles could still be admitted. Any name-dependence
  measurement on such worlds would be confounded by the cliché the cross-sectional path already
  rejects. Now, when a judge + prose are supplied, the temporal path runs the **same `_anti_cliche`
  T4** (named / roles-only / name+role-blind certificate) on the **summary graph** (`answer_key`
  collapses lags), carrying `difficulty` and `faithfulness` onto the `GateReport` alongside the
  `temporal_grade`. A temporal world is admitted only if it is recoverable **and** not guessable.

### Notes
- Temporal *grader-independent* faithfulness (the closed-form `check_faithfulness` analogue for lagged
  SCMs) remains future work; PCMCI+ recoverability is still the temporal recoverability proxy. The
  anti-cliché bar, however, is now identical across both paths.
- New test `test_temporal_world_also_runs_anti_cliche`: a lagged world (lever→(lag1)mid→kpi) is
  rejected when a judge recovers its summary graph from names, and admitted (carrying `temporal_grade`
  **and** `difficulty`) when it cannot. 136 tests, 95.6% coverage.

[0.26.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.26.0

## [0.25.0] — 2026-06-23

**`benchmark/v0.7` (role-gated companion) + generation resilience — and an honest read on roles.**

### Added
- **`benchmark/v0.7`** — 11 worlds authored under the v0.24 roles-only gate (named<0.5 **and**
  roles-only<0.4 **and** blind<0.35), mean difficulty 0.60. The strict companion to the 26-world v0.6.
- **Generation resilience** (`evals/scale/generate_hardened.py`): retry transient provider errors
  (Gemini 503) + **resume from disk** (skip worlds already written) — a transient blip no longer
  wastes a whole run.

### Finding (honest, partly negative)
- The roles-only gate is **marginal**: v0.7's name-blind prior is **0.43** vs v0.6's 0.46 (chance
  0.16). Single-sample LLM-judge gating is noisy, and role-type ("controllable→outcome") is
  *intrinsically* informative — unremovable while keeping the lever→outcome path. It cost 26→11 worlds
  for a 0.03 nudge.
- **What's actually clean:** the 3-tier certificate shows **name+role-blind ≈ 0.00** (structure not
  guessable without semantics) and **named cut 0.71→0.4**. So name/structure *memorization* — the
  cliché that matters — is eliminated; the residual is legitimate role-type prior, transparently
  reported. (#13 closed on disclosure, not a false "leak fixed".)

### Notes
- Package code unchanged since v0.24 (benchmark + eval-script + docs release). 135 tests, 95% coverage.

[0.25.0]: https://github.com/noumenal-ai/causal-worlds/releases/tag/v0.25.0

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
