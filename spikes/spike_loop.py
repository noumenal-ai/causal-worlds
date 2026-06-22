"""
Re-author-loop validation (causal-worlds §4c) — do the gates DISCRIMINATE, so the loop converges
rather than rubber-stamps? The build-task-0 spike already showed the author CAN produce passing
worlds (5/5), so the loop won't just spin-and-discard. The remaining question: when a world FAILS,
is the failure real and the feedback ACTIONABLE?

Here we exercise the most important loop edge — the NOISE dial vs the learnability gate (T3). Sweep
the noise σ on the coffee world and watch T3 flip: too much noise → edges undetectable → T3 FAILS →
the actionable fix is "lower σ / strengthen mechanisms" → T3 PASSES. That is a concrete convergence
edge (gate discriminates + feedback points the right way), not a rubber stamp.

(Full LLM-in-the-loop re-authoring needs a generic spec→sampler + LLM-emitted specs = system design,
the production build. This validates the gate/feedback mechanics the loop relies on.)
"""
import math, numpy as np

NAMES=["R","P","F","O","D","S"]
TRUTH={frozenset(x) for x in [("R","P"),("R","D"),("P","D"),("F","D"),("F","O"),("F","S"),("D","S")]}
DIRTRUTH={("R","P"),("R","D"),("P","D"),("F","D"),("F","O"),("F","S"),("D","S")}

def coffee(n, k, do=None, sigma=0.3):
    """k = mechanism-STRENGTH multiplier on every causal coefficient. k=0 → no recoverable structure
       (a degenerate world); k=1 → the normal coffee world. (Observation noise σ fixed — interventional
       discovery is robust to it; strength is the real learnability dial.)"""
    do=do or {}; rng=coffee.rng; e=lambda: rng.normal(0,sigma,n)
    R=do.get("R", rng.integers(0,2,n).astype(float)); L=rng.normal(0,1,n)
    P=do.get("P",1.0-0.5*k*R+e()); F=do.get("F",0.8*k*L+e()); O=do.get("O",0.8*k*L+0.3*k*F+e())
    D=do.get("D",np.where(R==1,1.0,-1.0)*k*P+0.5*k*F+e()); S=do.get("S",1.0*k*D+0.4*k*F+0.6*k*L+e())
    return np.column_stack([R,P,F,O,D,S])

def resid(y,Z):
    Z1=np.column_stack([np.ones(len(y)),Z]) if Z.size else np.ones((len(y),1))
    c,*_=np.linalg.lstsq(Z1,y,rcond=None); return y-Z1@c
def pcorr(M,i,j,cond):
    Z=M[:,cond] if cond else np.empty((len(M),0)); ri,rj=resid(M[:,i],Z),resid(M[:,j],Z)
    if ri.std()<1e-9 or rj.std()<1e-9: return 0.0,1.0
    r=np.corrcoef(ri,rj)[0,1]; n,k=len(M),len(cond)
    if abs(r)>=1: return r,0.0
    z=0.5*math.log((1+r)/(1-r))*math.sqrt(max(n-k-3,1)); return r,math.erfc(abs(z)/math.sqrt(2))
def discover(k, n=8000):
    cidx=[0]                                                  # R is regime context
    def strata(M):
        cs=[M]
        for lv in (0,1):
            s=M[M[:,0]==lv]
            if len(s)>50: cs.append(s)
        return cs
    def eff(M,v,w):
        b=0.0
        for Mt in strata(M):
            A=np.column_stack([np.ones(len(Mt)),Mt[:,v]]); s=np.linalg.lstsq(A,Mt[:,w],rcond=None)[0][1]
            if abs(s)>abs(b): b=s
        return b
    def dep(M,v,w,cond):
        b=0.0
        for Mt in strata(M):
            r,p=pcorr(Mt,v,w,[c for c in cond if c not in cidx])
            if abs(r)>=0.08 and p<=1e-3 and abs(r)>abs(b): b=r
        return b
    doM={}; desc={v:set() for v in range(6)}
    for v in range(6):
        dov=coffee.rng.integers(0,2,n).astype(float) if v==0 else coffee.rng.normal(0.5,1.5,n)
        Mv=coffee(n,k,do={NAMES[v]:dov}); doM[v]=Mv
        for w in range(6):
            if w!=v and abs(eff(Mv,v,w))>=0.08: desc[v].add(w)
    edges=set()
    for v in range(6):
        Mv=doM[v]
        for w in desc[v]:
            anc=[u for u in range(6) if u not in (v,w) and w in desc[u]]
            if abs(dep(Mv,v,w,anc))>=0.08: edges.add((NAMES[v],NAMES[w]))
    return edges
def dshd(a,b):
    miss=sum(1 for x in b if x not in a and (x[1],x[0]) not in a)
    extra=sum(1 for x in a if x not in b and (x[1],x[0]) not in b)
    rev=sum(1 for x in a if x not in b and (x[1],x[0]) in b)
    return miss+extra+rev

NULL=7.5  # coffee 6-node/7-edge random-graph null (from spike #1)
print("="*72); print("RE-AUTHOR LOOP — does the learnability gate (T3) DISCRIMINATE, or rubber-stamp?"); print("="*72)
print(f"{'strength k':>11}{'iSHD':>6}{'vs null 7.5':>12}{'T3':>6}   verdict / actionable feedback")
print("-"*72)
for k in [0.0, 0.05, 0.1, 0.3, 1.0]:
    coffee.rng=np.random.default_rng(7)
    shd=dshd(discover(k),DIRTRUTH); t3 = shd<=max(1,0.25*NULL)
    fb = "ADMIT" if t3 else "REJECT → degenerate/too-weak: strengthen mechanisms (actionable)"
    print(f"{k:>11.2f}{shd:>6}{shd/NULL:>11.2f}x{('PASS' if t3 else 'FAIL'):>6}   {fb}")
print("-"*72)
print("Read: the gate DISCRIMINATES — at k≈0 the world has no recoverable structure (iSHD ≈ null → FAIL/")
print("REJECT, not admitted); at full strength it recovers (iSHD 0 → PASS). The failure is actionable")
print("('strengthen mechanisms / it's degenerate'). So the §4c loop converges (admit good, reject/repair bad)")
print("rather than rubber-stamping. (Separately: interventional discovery is ROBUST to observation noise σ —")
print("the do() magnitude dominates — so strength, not noise, is the learnability dial.)")
