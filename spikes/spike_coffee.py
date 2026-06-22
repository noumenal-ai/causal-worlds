"""
Spike (causal-worlds LLD §0) — kills-or-confirms the benchmark-validity core BEFORE any build.

Question: on an ANTI-CLICHE world (regime-dependent sign flip + a HIDDEN confounder), does a
data-driven discoverer recover the truth and beat a prior-only baseline — i.e. is the benchmark
valid (not two LLMs agreeing on cliches), and does INTERVENTIONAL data (the gym's "identifiability
by construction") rescue what observational stats + priors get wrong?

World (Amit's anti-cliche coffee chain), hand-authored SCM. Observed: R,P,F,O,D,S. HIDDEN: L.
  R  regime          ~ Bernoulli(.5)   (0=promo, 1=scarcity)   [observed context]
  L  local_events    ~ N(0,1)          HIDDEN confounder
  P  price           = 1.0 - 0.5*R + e            (promo => cheaper: policy couples R->P)
  F  foot_traffic    = 0.8*L + e
  O  staff_overtime  = 0.8*L + 0.3*F + e          (child of L,F; a LEAF — no causal effect on S)
  D  demand          = beta(R)*P + 0.5*F + e      beta=-1 promo (price down=>demand UP, textbook),
                                                  beta=+1 scarcity (price down=>demand DOWN: sign FLIP)
  S  sales           = 1.0*D + 0.4*F + 0.6*L + e  (L direct => O and S confounded by hidden L)

Truth skeleton over OBSERVED nodes (L hidden):  R-P, R-D, P-D, F-D, F-O, F-S, D-S   (7 edges)
The two traps:
  (a) spurious O-S: O and S share only the HIDDEN L => no conditioning on observed vars removes it.
  (b) P->D sign FLIPS by regime; a pooled/observational view averages -1 and +1 (and R confounds P,D).
A prior ("price down => demand up; more overtime => more sales") gets (a) and the scarcity half of (b) wrong.

CAVEAT — what this spike does and does NOT prove. The interventional "discoverer" below is HAND-TARGETED
at the two planted traps (it drops O-S on a null do(O) effect and restores P-D on a non-null do(P) effect).
So it proves the PREMISE — interventional data *contains the information* to fix exactly the failure modes
that observational stats + priors miss (identifiability-by-construction, HLD #5) — NOT that a *general*
interventional-discovery algorithm recovers the graph automatically. A real harness needs a principled
interventional procedure (e.g. GIES / systematic per-variable do-tests). Generalizing this is the first
build task the spike UNBLOCKS, not something it settles.
"""
import math, numpy as np
from itertools import combinations

rng = np.random.default_rng(7)
NAMES = ["R", "P", "F", "O", "D", "S"]; IDX = {n: i for i, n in enumerate(NAMES)}
TRUTH = {frozenset(e) for e in [("R","P"),("R","D"),("P","D"),("F","D"),("F","O"),("F","S"),("D","S")]}

def beta(R): return np.where(R == 1, 1.0, -1.0)   # +1 scarcity, -1 promo

def sample(n, do=None):
    """do = None (observational) | ('P', val_array) | ('O', val_array). Returns observed matrix + R,L."""
    R = rng.integers(0, 2, n).astype(float)
    L = rng.normal(0, 1, n)                                  # HIDDEN
    e = lambda s=0.3: rng.normal(0, s, n)
    P = (1.0 - 0.5*R + e()) if do is None or do[0] != "P" else np.asarray(do[1], float)
    F = 0.8*L + e()
    O = (0.8*L + 0.3*F + e()) if do is None or do[0] != "O" else np.asarray(do[1], float)
    D = beta(R)*P + 0.5*F + e()
    S = 1.0*D + 0.4*F + 0.6*L + e()
    X = np.column_stack([R, P, F, O, D, S])
    return X, R, L

def shd(a, b): return len(a ^ b)                              # undirected structural Hamming distance

def resid(y, Z):
    Z1 = np.column_stack([np.ones(len(y)), Z]) if Z.size else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ coef

def pcorr(X, i, j, cond):                                     # partial corr of i,j given `cond` cols
    Z = X[:, cond] if cond else np.empty((len(X), 0))
    ri, rj = resid(X[:, i], Z), resid(X[:, j], Z)
    r = np.corrcoef(ri, rj)[0, 1]
    n, k = len(X), len(cond)
    z = 0.5*math.log((1+r)/(1-r)) * math.sqrt(max(n-k-3, 1))  # Fisher-z
    p = math.erfc(abs(z)/math.sqrt(2))                        # two-sided ~normal
    return r, p

def pc_skeleton(X, rmin=0.08, pmax=1e-3, maxk=3):
    """Proper PC adjacency search: keep edge i-j iff DEPENDENT given EVERY conditioning subset
       tried (up to size maxk). Removing-on-some-subset avoids collider moralization (no spurious
       co-parent edges). Returns (skeleton, dropped) where `dropped` notes pairs cut at |r|<rmin."""
    edges, why = set(), {}
    for i in range(6):
        for j in range(i+1, 6):
            others = [k for k in range(6) if k not in (i, j)]
            indep = False
            for k in range(0, min(maxk, len(others)) + 1):
                for cond in combinations(others, k):
                    r, p = pcorr(X, i, j, list(cond))
                    if abs(r) < rmin or p > pmax:        # conditionally independent on this subset
                        indep = True; why[frozenset((NAMES[i],NAMES[j]))] = (list(cond), r); break
                if indep: break
            if not indep:
                edges.add(frozenset((NAMES[i], NAMES[j])))
    return edges, why

def effect(Xy_x, Xy_y):                                       # univariate OLS slope of y on x
    A = np.column_stack([np.ones(len(Xy_x)), Xy_x])
    return np.linalg.lstsq(A, Xy_y, rcond=None)[0][1]

# ---------- 1) PRIOR-ONLY baseline (sees variable names/prose, NO data) ----------
prior = {frozenset(e) for e in [("P","D"),("D","S"),("F","S"),("F","D"),("O","S"),("R","D"),("R","P")]}
prior_sign = {"promo": "-", "scarcity": "-"}                  # "price down => demand up" asserted in BOTH

# ---------- 2) OBSERVATIONAL statistics (data, no interventions) ----------
Xo, Ro, Lo = sample(8000)
obs, why = pc_skeleton(Xo)
pd_dropped = frozenset(("P","D")) not in obs      # sign-flip cancellation => marginal independence?
os_kept    = frozenset(("O","S")) in obs          # hidden confounder => unremovable spurious edge?
# pooled P->D sign (best a pooled observational view can say), controlling for observed R,F:
poolc = np.linalg.lstsq(np.column_stack([np.ones(8000), Xo[:,IDX["P"]], Ro, Xo[:,IDX["F"]]]),
                        Xo[:,IDX["D"]], rcond=None)[0][1]

# ---------- 3) INTERVENTIONAL (the gym emits do-data) ----------
# do(O): randomize overtime independent of L,F -> is there any O->S effect?
Xio, _, _ = sample(8000, do=("O", rng.normal(0, 1.5, 8000)))
o_on_s = effect(Xio[:,IDX["O"]], Xio[:,IDX["S"]])
# do(P): randomize price independent of R,L -> recover the P->D effect WITHIN each regime
Xip, Rip, _ = sample(8000, do=("P", rng.normal(0.7, 0.6, 8000)))
b_promo    = effect(Xip[Rip==0][:,IDX["P"]], Xip[Rip==0][:,IDX["D"]])
b_scarcity = effect(Xip[Rip==1][:,IDX["P"]], Xip[Rip==1][:,IDX["D"]])
interv = set(obs)
if abs(o_on_s) < 0.1: interv.discard(frozenset(("O","S")))                       # do(O): no effect => kill spurious edge
if abs(b_promo) > 0.5 or abs(b_scarcity) > 0.5: interv.add(frozenset(("P","D"))) # do(P): regime effect => restore edge the linear CI test missed

# ---------- random-graph NULL (margin reference) ----------
allpairs = [frozenset((NAMES[i],NAMES[j])) for i in range(6) for j in range(i+1,6)]
null = np.mean([shd(set(rng.choice(len(allpairs), len(TRUTH), replace=False).tolist() and
                       [allpairs[k] for k in rng.choice(len(allpairs), len(TRUTH), replace=False)]), TRUTH)
                for _ in range(2000)])

# ---------- verdict ----------
def sgn(x): return "+" if x > 0.15 else ("-" if x < -0.15 else "~0 (ambiguous)")
print("="*72)
print("ANTI-CLICHE COFFEE WORLD — benchmark-validity spike")
print("="*72)
print(f"truth skeleton ({len(TRUTH)} edges): {sorted('-'.join(sorted(e)) for e in TRUTH)}")
print(f"random-null SHD (avg of 2000):      {null:.2f}   (chance baseline)\n")
print(f"{'discoverer':<26}{'SHD↓':>6}  {'spurious O-S?':>14}  P->D sign (promo / scarcity)")
print("-"*72)
print(f"{'1. prior-only (no data)':<26}{shd(prior,TRUTH):>6}  {'YES (asserts)':>14}  "
      f"{prior_sign['promo']} / {prior_sign['scarcity']}   <- scarcity WRONG (truth +)")
print(f"{'2. observational stats':<26}{shd(obs,TRUTH):>6}  "
      f"{('YES' if frozenset(('O','S')) in obs else 'no'):>14}  pooled slope {poolc:+.2f} -> {sgn(poolc)}")
print(f"{'3. + interventional data':<26}{shd(interv,TRUTH):>6}  "
      f"{('YES' if frozenset(('O','S')) in interv else 'no'):>14}  "
      f"do(P): {sgn(b_promo)} / {sgn(b_scarcity)}   (true -1 / +1)")
print("-"*72)
print(f"do(O)->S effect = {o_on_s:+.3f}  (truth 0: overtime has NO causal effect on sales)")
print(f"do(P)->D effect: promo {b_promo:+.2f} (true -1) | scarcity {b_scarcity:+.2f} (true +1)")
print()
print("WHAT THE DATA-ONLY METHODS MISS (the two designed traps):")
print(f"  • sign-flip cancellation : observational PC {'DROPS' if pd_dropped else 'keeps'} P-D "
      f"(pooled slope {poolc:+.2f}≈0 — the −1/+1 regimes cancel marginally)")
print(f"  • hidden confounder      : observational PC {'KEEPS spurious' if os_kept else 'removes'} O-S "
      f"(no observed subset d-separates O,S — L is hidden)")
print(f"  • do-data fixes BOTH     : do(P) restores P-D w/ correct per-regime signs; do(O)≈0 removes O-S\n")
ok_anti   = shd(prior,TRUTH) >= 2 and prior_sign["scarcity"] == "-"           # prior genuinely fails
ok_obs_in = os_kept or pd_dropped or abs(poolc) < 0.5                          # observational alone insufficient
ok_struct = (frozenset(("O","S")) not in interv) and (frozenset(("P","D")) in interv) \
            and shd(interv,TRUTH) <= shd(obs,TRUTH) and shd(interv,TRUTH) <= shd(prior,TRUTH)
ok_sign   = b_promo < -0.5 and b_scarcity > 0.5
print("VERDICT")
print(f"  anti-cliche (prior fails)            : {'PASS' if ok_anti else 'FAIL'}  (prior SHD {shd(prior,TRUTH)} vs null {null:.1f})")
print(f"  observational alone insufficient     : {'PASS' if ok_obs_in else 'FAIL'}  (obs SHD {shd(obs,TRUTH)})")
print(f"  interventional recovers structure    : {'PASS' if ok_struct else 'FAIL'}  (interv SHD {shd(interv,TRUTH)})")
print(f"  interventional recovers regime signs : {'PASS' if ok_sign else 'FAIL'}  ({b_promo:+.2f} / {b_scarcity:+.2f})")
print(f"\n  => benchmark-validity core {'CONFIRMED' if (ok_anti and ok_obs_in and ok_struct and ok_sign) else 'NOT confirmed'}")
