"""
Build-task-1 (validation) — swap my hand-rolled discoverer for VETTED, off-the-shelf algorithms.

Question: does the benchmark hold up under a standard grader I did NOT write?
  (a) do STANDARD OBSERVATIONAL methods (causal-learn PC, GES) FAIL on the trap world
      (hidden-confounder + regime sign-flip)?  -> if yes, the benchmark is genuinely non-trivial.
  (b) does a VETTED INTERVENTIONAL method (gies = GIES, Gamella's pkg) do BETTER given do-data?
      -> validates "interventional data rescues it" with a library, not my code.
We compare SKELETONS (undirected adjacency) — fair across CPDAG outputs — and flag the spurious
confounded edge specifically. Honest: GIES assumes causal sufficiency (no hidden confounders), so
the hidden L is OUT of its assumptions; we report what it actually does, not what we hope.

Worlds: 'coffee' (HARD: sign-flip P-D by regime R + hidden L confounds O,S) and 'ecommerce' (EASY,
textbook, no confounder) as a calibration control.
"""
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
import gies

rng = np.random.default_rng(7)
def e(n,s=0.3): return rng.normal(0,s,n)

# ---------- worlds: each returns observed matrix; supports do(name,val) ----------
def coffee(n, do=None):
    do = do or {}
    R = do.get("R", rng.integers(0,2,n).astype(float)); L = rng.normal(0,1,n)         # L hidden
    P = do.get("P", 1.0-0.5*R+e(n)); F = do.get("F", 0.8*L+e(n))
    O = do.get("O", 0.8*L+0.3*F+e(n))
    D = do.get("D", np.where(R==1,1.0,-1.0)*P + 0.5*F + e(n))
    S = do.get("S", 1.0*D+0.4*F+0.6*L+e(n))
    return np.column_stack([R,P,F,O,D,S])
COFFEE = dict(name="coffee", f=coffee, names=["R","P","F","O","D","S"],
              skel={frozenset(x) for x in [("R","P"),("R","D"),("P","D"),("F","D"),("F","O"),("F","S"),("D","S")]},
              spurious=frozenset(("O","S")))                     # hidden-L confounded pair (NOT a true edge)

def ecom(n, do=None):
    do = do or {}
    Ad=do.get("Ad",rng.normal(0,1,n)); Di=do.get("Discount",rng.normal(0,1,n))
    Tr=do.get("Traffic",1.0*Ad+e(n)); Sa=do.get("Sales",1.0*Tr+0.8*Di+e(n))
    return np.column_stack([Ad,Di,Tr,Sa])
ECOM = dict(name="ecommerce", f=ecom, names=["Ad","Discount","Traffic","Sales"],
            skel={frozenset(x) for x in [("Ad","Traffic"),("Traffic","Sales"),("Discount","Sales")]},
            spurious=None)

def skel_from_adj(adj, names):                                  # adjacency -> undirected edge set
    m=len(names); s=set()
    for i in range(m):
        for j in range(i+1,m):
            if adj[i,j]!=0 or adj[j,i]!=0: s.add(frozenset((names[i],names[j])))
    return s
def shd_skel(a,b): return len(a^b)

def run(world, n=5000):
    names=world["names"]; m=len(names); truth=world["skel"]
    Xobs=world["f"](n)
    def spur(s): return "—" if world["spurious"] is None else ("KEPT(spurious)" if world["spurious"] in s else "removed")
    out={"truth":len(truth)}
    # --- (a) observational, vetted ---
    try:
        cg=pc(Xobs, 0.01, "fisherz", verbose=False, show_progress=False); s=skel_from_adj(cg.G.graph,names); out["PC"]=(shd_skel(s,truth),spur(s))
    except Exception as ex: out["PC"]=("ERR:"+type(ex).__name__,"")
    try:
        g=ges(Xobs); s=skel_from_adj(g["G"].graph,names); out["GES"]=(shd_skel(s,truth),spur(s))
    except Exception as ex: out["GES"]=("ERR(numpy2?)","")
    # --- (b) interventional, vetted (GIES): env 0 observational + one do() env per variable ---
    try:
        data=[Xobs]; I=[[]]
        for v in range(m):
            dov = rng.integers(0,2,n).astype(float) if names[v]=="R" else rng.normal(0.5,1.5,n)
            data.append(world["f"](n, do={names[v]: dov})); I.append([v])
        est,_=gies.fit_bic(data, I); s=skel_from_adj(est,names); out["GIES"]=(shd_skel(s,truth),spur(s))
    except Exception as ex: out["GIES"]=("ERR:"+type(ex).__name__,"")
    # --- (c) FCI: latent-AWARE (the right tool for hidden confounders) — does it MARK O-S as confounded? ---
    if world["spurious"] is not None:
        try:
            g,_=fci(Xobs, independence_test_method="fisherz", alpha=0.01, verbose=False, show_progress=False)
            nm={n:i for i,n in enumerate(names)}; a,b=tuple(world["spurious"]); i,j=nm[a],nm[b]
            gg=g.graph
            if gg[i][j]==0 and gg[j][i]==0: out["FCI_mark"]="absent"
            elif gg[i][j]==1 and gg[j][i]==1: out["FCI_mark"]="bidirected ✓ (confounding detected)"
            else: out["FCI_mark"]="directed/other (treats as causal)"
        except Exception as ex: out["FCI_mark"]="ERR:"+type(ex).__name__
    return out

print("="*84)
print("BUILD-TASK-1 — VETTED grader: causal-learn PC/GES (observational) vs gies/GIES (interventional)")
print("="*84)
print(f"{'world':<12}{'truth|E|':>9}   {'PC skel-SHD':>22}{'GES skel-SHD':>22}{'GIES skel-SHD':>22}")
print("-"*84)
for w in (COFFEE, ECOM):
    r=run(w)
    print(f"{w['name']:<12}{r['truth']:>9}   "
          f"{str(r['PC'][0])+' / '+r['PC'][1]:>22}{str(r['GES'][0])+' / '+r['GES'][1]:>22}{str(r['GIES'][0])+' / '+r['GIES'][1]:>22}")
    if "FCI_mark" in r: print(f"             FCI (latent-aware) on the confounded pair O-S: {r['FCI_mark']}")
print("-"*84)
print("Read: skel-SHD = undirected edge errors vs truth (lower better). 'KEPT(spurious)' = the method kept the")
print("hidden-confounder edge O-S that is NOT causal (the trap). ecommerce = easy control (all should ~nail it).")
print("Expectation: observational PC/GES keep the spurious edge / miss structure on 'coffee'; the interventional")
print("GIES, given do-data, should do better on the confounded pair. Whatever happens is reported as the finding.")
