# Findings — what the benchmark shows (and what it doesn't)

The headline result, with its caveats stated honestly. Full tables + bootstrap CIs live under
[`evals/`](../evals/); this page is the readable summary.

## The crossover: latent-awareness is the dividing line, not interventions

Across the 26-world hardened [`benchmark/v0.6`](../benchmark/v0.6/) set (3 seeds each). The comparison
is **information-fair**: the `+do` methods get the *same interventional budget* (pooled observational
+ per-variable `do()` environments) as the latent-aware reference — so we compare *methods*, not data
access ([full table + bootstrap CIs](../evals/baseline-crossover/v0.6/)):

| method | data | latent-aware? | mean skeleton-SHD ↓ | confounded pair kept as causal ↓ |
|---|---|---|---|---|
| **interventional-ci** (reference) | interventional | yes | **1.31** | **0** |
| GIES | interventional | no | 5.71 | 30 |
| PC | observational | no | 3.18 | 29 |
| **PC + interventions** | interventional | no | 4.71 | **30** |
| FCI | observational | partly | 3.17 | 21.7 |
| FCI + interventions | interventional | partly | 4.73 | 21 |
| DAGMA | observational | no | 5.23 | 27 |
| DirectLiNGAM | observational | no | 5.76 | 27 |

(DAGMA and DirectLiNGAM run at default hyperparameters, and LiNGAM's non-Gaussian assumption is
violated by these linear-Gaussian worlds, so their *skeleton* accuracy is not their best — but the
robust verdict is **confounded-kept**, and like every causal-sufficiency method they keep it.)

The honest reading: the dividing line is **latent-awareness, not interventions**. The decisive row is
**PC + interventions** — given the *same* interventional budget as the reference, it still scores the
hidden-confounded pair as causal as often as observational PC: **confounded-kept 30 vs 29** (the
seed-averaged count of confounded-pair instances scored causal, summed over the 26 worlds — some
worlds carry more than one confounded pair). GIES likewise (30). Only the latent-aware interventional
rule reaches **0**. The interventional
advantage is robust: ΔF1 = F1(reference) − F1(method) is **+0.37, 95% CI [0.33, 0.42]** for `pc+do`
(every method's CI excludes 0). So this is an **identifiability result** (you cannot tell confounding
from causation without *both* interventions *and* a latent-aware method), **not** "our method beats
the toolbox." (The earlier, easier [`v0.5`](../benchmark/v0.5/) set is kept for comparison.)

## Caveats we're not hiding

1. **Circular admission — fixed (v0.15).** Admission (gate T3) is now **grader-independent**: a world
   is admitted iff its declared SCM is *faithful by construction* (every edge induces a detectable
   partial correlation; regimes genuinely modulate), computed in closed form from the spec with **no
   discovery method run**. The reference grader's score is reported, never gates.
2. **Simulated-DAG leakage — mostly fixed (v0.14), residual disclosed.** Synthetic SCMs can leak the
   causal order through marginal variance ([varsortability](../evals/varsortability/)) *and* through
   scale-invariant predictability (R²-sortability). v0.14 generates worlds with **internal
   standardization (iSCM)**, dropping varsortability to 0.54 and R²-sortability 0.73 → 0.60; both
   trivial sorting baselines fall to F1 ≈ 0.33–0.37, well under the real methods. The residual
   R²-sortability (0.60 > 0.5) is disclosed, not yet fully closed.
3. **Difficulty is descriptive, not a validated predictor.** Difficulty vs skeleton-SHD error: with
   bootstrap CIs (n=35), the observational methods show r≈0.40 (PC [0.07, 0.68], FCI [0.08, 0.68] —
   just excluding 0) while the latent-aware reference is flat (r≈0.24, [−0.06, 0.51], includes 0).
4. **Anti-cliché: name leakage fixed, role leakage remains (measured).** The `adversarial` author +
   strict T4 gate (v0.19) regenerated the set as [`benchmark/v0.6`](../benchmark/v0.6/). On the
   [3-tier certificate](../evals/name-only-baseline/v0.6/), the **named** name-only prior fell from
   v0.5's **0.71** to **0.38** (chance 0.18) — and the **name+role-blind** prior collapses to **0.01**,
   proving the structure itself is *not* guessable once semantics are stripped. Strikingly, the
   **name-blind** prior (0.46) is *higher* than named — the adversarial names now actively **mislead**.
   The residual is **role-type priors** (controllable→outcome conventions). v0.24 added a **roles-only
   gate** and a companion [`benchmark/v0.7`](../benchmark/v0.7/) (11 worlds): it only nudged the
   name-blind prior 0.46 → 0.43 (chance 0.16) — single-sample LLM-judge gating is noisy and role-type
   is *intrinsically* informative (you can't remove it while keeping the lever→outcome path). The
   honest bottom line: the cliché that matters — **name/structure memorization — is eliminated**
   (fully-blind ≈ 0.00, named cut 0.71→0.4); the remaining role-type signal is a legitimate operational
   prior, transparently reported, not hidden.
