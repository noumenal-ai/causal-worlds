# Validation status — causal-worlds

> **VALIDATION COMPLETE (2026-06-22).** The scientific premises are de-risked via runnable spikes (`spikes/`).
> The **production library (code quality + system design) is the next phase** — the owner re-engages there.
> Every claim below is backed by a spike you can run; this is the validation→build handoff.

## The four questions validation had to answer

**1. Can the harness tell a real causal-discoverer from a guesser, even on a hard world? (circularity #1)** → **YES.**
spike #1 (`spike_coffee.py`): on an anti-cliché world (regime sign-flip + hidden confounder), a prior fails,
observational stats fail, and **interventional data recovers SHD 0**. spike #2 (`spike_coffee_general.py`): a
*general*, non-hand-targeted interventional rule recovers **directed SHD 0**, robust 20/20 across seeds × noise.

**2. Can an LLM author valid, gate-passing worlds from a one-line prose prompt? (build-task-0 — THE product risk)**
→ **YES through T1–T3; the strict T4 anti-cliché gate is a separate, harder bar (still the open risk).**
`spike_author.py`: 5/5 worlds authored from prompts pass **T1–T3** first-try, robust 5/5. **Crucial method point:**
difficulty must be judged by an **independent model** — self-grading (Claude author + Claude prior) understated it
(gap ≈1); an independent **Gemini** prior revealed the real spectrum (anti-cliché worlds gap 3–5, textbook worlds
0). Rule: the prior-only / faithfulness judge ≠ the author's model family.

> ⚠️ **Honesty correction (2026-06-25, #19).** The "5/5 first-try" result above is **T1–T3 only** — it predates the
> strict **T4 anti-cliché gate** (added v0.19, which requires a name+role guesser to recover < 50% of the edges,
> i.e. difficulty ≥ 0.5). The `evals/author-model-bakeoff` "8/8 admit, 1.0 attempts" likewise ran on **package
> v0.2.0** at **mean difficulty 0.30** — *those worlds would be rejected by today's gate.* Measured live on v0.33,
> intuitive prompts (coffee, power grid, hospital ED, bike-share, streaming) are **routinely rejected by T4** even
> after the re-author budget, because real operations have common-sense structure a judge guesses from names. The
> author pass-rate **under the strict gate** is therefore **still the open question** `docs/scope.md §8` flags — it
> is *not* "solved." **Playground mode** (v0.34, `anti_cliche=False` / `--playground`) is the product escape hatch:
> it keeps faithfulness + the difficulty score but never rejects on guessability, so a user always gets their world.

**3. What grader does the benchmark need? (build-task-1)** → **An interventional-CI discoverer — NOT a stock
library.** `spike_grader.py`: vetted off-the-shelf methods (causal-learn PC/GES, `gies`/GIES, FCI) **all fail the
hidden-confounder trap** — PC & GIES keep the spurious edge (GES/GIES assume causal *sufficiency*); FCI doesn't
cleanly mark it; GES is numpy-2-broken. Only the **interventional independence test** (`do(O)→S≈0`) removes it. The
gym's interventional data is *necessary*, and the grader must use it.

**4. Does it scale, and does the gate discriminate (so the loop converges)?** → **YES.** `spike_author.py`'s 10-node
logistics world (2 hidden confounders + regime flip) authored valid and recovered at **SHD 1** (null 13.9), robust
5/5. `spike_loop.py`: the learnability gate **discriminates** — a degenerate near-zero-strength world **FAILS**
(SHD ≈ null), a real one **PASSES** — so the §4c loop admits good / rejects-repairs bad worlds rather than
rubber-stamping. (Bonus: interventional discovery is robust to observation noise — strength, not noise, is the
learnability dial.)

## What validation CHANGED in the design
- **Grader (build-task-1):** interventional-CI, **not** GIES (hld §4d).
- **Schema:** must represent **confounding** (latent / bidirected), scored distinctly from a causal edge (hld §2).
- **Difficulty / anti-cliché is structural** (SHD is blind to sign-flips) and must be measured by an **independent**
  model (hld §4a).
- **Thresholds normalized per world-size**; the reference discoverer is **pinned & versioned** (part of the artifact).

## NOT yet validated — left for the production build (owner re-entry)
1. **Full LLM-in-the-loop re-authoring** with a generic **spec→sampler** format + LLM-emitted specs. The spikes
   hand-code samplers and validate the loop's *edges* (author produces passing worlds; gates discriminate;
   feedback is actionable); the end-to-end LLM loop is **system design**.
2. **World diversity at real scale** — far beyond the handful here (more structures, sizes, confounding patterns,
   nonlinearity, temporal/regime dynamics).
3. **The hardened interventional-CI grader** as a real, versioned component (FCI-with-interventions or a principled
   do()-CI method) replacing the spike prototype + its world-diversity sweep.
4. **Toolchain pinning** — GES is numpy-2-broken; the build env must pin a working causal stack (py3.13; `uv`).
5. **Faithfulness judging of hidden-mechanism worlds** — Gemini scored these low ("incomplete") because prose names
   hidden variables; needs a protocol that accounts for intended-hidden variables.

## Spikes (the evidence) — `spikes/`
| file | validates |
|---|---|
| `spike_coffee.py` | harness premise: prior & observational fail, interventional recovers (SHD 0) |
| `spike_coffee_general.py` | a *general* interventional discoverer recovers (directed SHD 0, robust) |
| `spike_author.py` | author-from-prose (5/5) + 10-node scale world + **Gemini** independent prior & faithfulness |
| `spike_grader.py` | vetted PC/GES/GIES/FCI **fail** the confounder trap → grader must be interventional-CI |
| `spike_loop.py` | learnability gate **discriminates** (degenerate FAIL / real PASS) |

**Repro:** `python3.13 -m venv .venv && .venv/bin/pip install numpy causal-learn gies`; set `GEMINI_API_KEY` for
the independent judge. (3.14 has no scientific wheels yet; `uv` works too.)

---

*This is the validation→build line. The next phase — the actual causal-worlds library (code quality + system
design) — is where the owner re-engages.*
