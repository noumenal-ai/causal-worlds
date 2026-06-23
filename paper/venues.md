# Candidate venues for the `causal-worlds` preprint

*Subject: the public OSS package `causal-worlds` (v0.25.0, MIT). Recommended framing for the first submission: **Framing B** — empirical finding + generator-as-apparatus (lower-risk than a flagship Datasets-and-Benchmarks paper; the contribution is the leakage-resistant apparatus that surfaces a clean, reproducible identifiability crossover). Working title: "Fiction-first causal worlds: a leakage-resistant generator that cleanly exhibits a latent-aware interventional identifiability crossover." (The title deliberately frames the result as an **identifiability crossover**, not "beating the toolbox" — the body's honest framing is that the lever is latent-awareness, and PC/GES/GIES keep the confounded edge *because* they assume causal sufficiency, which is expected.) Dates as of June 2026; deadlines marked **projected** are inferred from prior-year patterns and must be verified against the official CFP.*

---

## Strategy in one paragraph

Post an **arXiv preprint first**, at sprint-end, to **timestamp priority on the perishable novelty** (the NL-authored, fiction-first, leakage-resistant discovery-benchmark slice is "no current match found," not a durable moat). Then submit the same finding to a **NeurIPS 2026 causality/eval workshop** (fast community signal) and the archival version to **CLeaR 2027** (best archival fit for Framing B). Hold the flagship **NeurIPS Evaluations & Datasets** track for **2027**, contingent on the escalation items (≥100 worlds, a temporal set, nonlinearity, Croissant metadata). Secure a **personal arXiv endorser now** — it has multi-week lead time and is independent of the technical work.

---

## Ranked candidates

### 1. NeurIPS 2026 causality / eval workshop — **best near-term home (Framing B)**
- **Fit:** Strong. Workshops explicitly want crisp empirical findings and apparatus papers; the identifiability crossover + the leakage/anti-cliché controls are exactly workshop-shaped, and a workshop does not demand ≥100 worlds.
- **Timing:** NeurIPS 2026 is **Sydney, Dec 6–12, 2026**. Workshop accept/reject **notification is fixed at Sept 29, 2026 (AoE) and "cannot be extended under any circumstances"**; workshop paper CFP deadlines therefore land **~Sept–Oct 2026**. (Workshop *proposals* were due June 6, 2026.)
- **Watch for:** causality / causal-representation, causal-discovery, LLM-evaluation, and time-series workshops. Specific 2026 workshop CFPs were not yet posted at research time — **verify on neurips.cc when CFPs appear.**
- **Notes:** Non-archival or lightly-archival at most workshops, which pairs well with also targeting CLeaR for the archival version. Fastest route to expert feedback on the "reference grader is a known method, not a contribution" framing.

### 2. CLeaR 2027 (Causal Learning and Reasoning) — **best archival fit (Framing B)**
- **Fit:** Strong / best archival. CLeaR explicitly welcomes "proof-of-concept research" and benchmark/empirical papers; a causal-discovery audience will read the Ψ-FCI/GIES positioning correctly and reward the leakage controls.
- **Timing:** CLeaR 2026 was April 6–8, 2026 (Cambridge, MA). Deadlines historically late-Oct/early-Nov (CLeaR 2025 deadline was Nov 2, 2024). CLeaR 2027 deadline **not yet announced — projected ~late-Oct / early-Nov 2026.**
- **Notes:** This is the venue whose reviewers will be hardest on the two former blockers (varsortability leakage, admission circularity) — both now addressed (iSCM + grader-independent admission), and reporting *both* sortability metrics with disclosed residuals is the accepted bar.

### 3. AISTATS 2027 — **moderate fit**
- **Fit:** Moderate. Methodological/statistical lean; a benchmark/empirical paper is a weaker fit here than at CLeaR, but the identifiability framing + CIs give it a statistical spine.
- **Timing:** Montréal, **Feb 16–23, 2027**; deadline historically early-mid Oct (AISTATS 2026 abstract was Sept 25, 2025) — **projected ~early Oct 2026.**
- **Notes:** Consider only if the finding is reframed with more statistical emphasis; otherwise CLeaR dominates.

### 4. NeurIPS Evaluations & Datasets (ED) Track — **the 2027 flagship target (Framing A)**
- **Fit:** Strong *if* the escalation items are done; this is the Framing-A flagship home.
- **Timing:** The **2026** cycle has passed (abstracts due May 4, full papers **May 6, 2026 AoE**). Target the **2027** cycle (~May 2027).
- **Bar:** Requires ≥100 worlds with documented diversity coverage, a temporal benchmark *set*, nonlinearity, variance/CIs throughout, and **Responsible AI (Croissant) metadata** for dataset submissions. Do not aim Framing B here.

### 5. DMLR (Journal of Data-centric Machine Learning Research) — **rolling archival fallback**
- **Fit:** Viable archival home for the dataset/benchmark contribution.
- **Timing:** Rolling submissions — no fixed deadline.
- **Notes:** Useful as a fallback or as the eventual archival home for the scaled Framing-A artifact if conference timing slips.

---

## Summary table

| Venue | Framing | Fit | Earliest realistic deadline | Notes |
|---|---|---|---|---|
| NeurIPS 2026 causality/eval workshop | B | Strong | ~Sept–Oct 2026 (notif. fixed Sept 29, 2026 AoE) | Best near-term; verify CFP on neurips.cc |
| CLeaR 2027 | B (archival) | Strong / best archival | ~late-Oct/early-Nov 2026 **(projected)** | Welcomes proof-of-concept + benchmarks |
| AISTATS 2027 | B (statistical reframe) | Moderate | ~early Oct 2026 **(projected)** | Montréal Feb 2027; benchmark-only is weaker here |
| NeurIPS ED Track 2027 | A (flagship) | Strong *if* escalated | ~May 2027 | Needs ≥100 worlds + temporal set + nonlinearity + Croissant |
| DMLR | A/B (archival) | Viable | Rolling | No deadline; fallback archival home |

Passed this cycle (for reference): **NeurIPS ED 2026** (May 6, 2026); **UAI 2026** (Feb 25, 2026 → UAI 2027 ~Feb 2027).

---

## arXiv endorsement constraint (founder action item — start now)

- **Verify these policy facts on arxiv.org before relying on them** — they are forward-looking claims carried from a research scan, not confirmed against the live policy page, and the two dates below in particular must be re-checked.
- Under the (reported) **Jan 21, 2026** arXiv policy change, an **institutional email alone no longer qualifies** a first-time, unaffiliated submitter. A never-published unaffiliated author must obtain a **personal endorsement from an established arXiv author in the target domain.**
- **Categories:** primary **cs.LG**; cross-list **cs.AI**, **stat.ME** / **stat.ML**. (cs.LG endorsers must have authored a threshold number of cs.LG papers within a recent window.)
- **Lead time:** the endorsement process — plus, if needed, a new OpenReview/arXiv account — can take **2+ weeks**. Secure an endorser **now**, independent of the technical work; a practical path is to identify an endorser among the cited authors (e.g. the simulated-DAG-leakage or interventional-identifiability lines), share the abstract/draft, and request early.
- **Why first:** post the preprint at **sprint-end** to **timestamp priority on the perishable novelty** before any workshop deadline and before a competitor occupies the slice. If a competitor posts an NL-authored discovery benchmark before then, **accelerate the preprint immediately**, even in preliminary form.
- **Governance note (verify):** arXiv was reported to become an independent nonprofit on **July 1, 2026**; no expected change to the endorsement mechanics — but treat both this date and the Jan 21, 2026 policy date as **unconfirmed** and re-verify on arxiv.org before posting.

---

## Verify-before-relying checklist

- [ ] NeurIPS 2026 specific causality/eval workshop CFPs (confirm on neurips.cc once posted).
- [ ] CLeaR 2027 official deadline (projected late-Oct/early-Nov 2026).
- [ ] AISTATS 2027 official deadline (projected early Oct 2026).
- [ ] Re-scan the competitive slice immediately before submission — the novelty is perishable.
- [ ] Personal arXiv endorser confirmed (cs.LG), with the abstract shared.
- [ ] **arXiv endorsement-policy + nonprofit-governance dates re-verified on arxiv.org** (Jan 21, 2026 policy; July 1, 2026 governance — both currently unconfirmed).
- [ ] **Every arXiv id in the preprint confirmed to resolve** before posting. Confirmed from this project's own literature scan: G-Sim 2506.09272, CausalDynamics 2505.16620, CausalProfiler 2511.22842, Auto-Bench 2502.15224, PillagerBench 2509.06235, Reisach varsortability 2102.13647, Ormaniec iSCM 2406.11601, Herman et al. CLeaR 2025 2503.17037, Kıcıman 2305.00050, Causal Parrots 2308.13067, Critical Review 2407.08029, Srivastava 2510.16530. **Still to confirm independently:** Jin et al. Corr2Cause id (omitted in the draft), Hauser & Bühlmann GIES 1303.3216, the NeurIPS-2023 R²-sortability paper id, and the venue/year on G-Sim's id. **Do not re-introduce** unverified ids for DEVS-Gen / SD-SCM / TimeGraph, and **do not cite "Caliper" with an arXiv id** — it was an informal framing, not a locatable archival reference.
