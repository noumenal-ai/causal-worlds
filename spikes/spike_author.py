"""
Build-task-0 — the AUTHOR spike (causal-worlds lld §0b). THE product risk.

Spikes #1-#2 validated the GRADER given a HAND-authored SCM. They never tested the AUTHOR:
can an LLM, from a one-line PROSE prompt, write a world that is valid (T1), samples cleanly (T2),
and is recoverable by interventional discovery (T3) — and where does it land on the difficulty
axis (T4, the prior-only gap)?  The LLM author here = Claude (me); I authored the 4 worlds below
from the one-line prompts in their `prose`, NOT by copying the coffee world.

INDEPENDENCE (the fix for circularity #1): the author is Claude, so the LLM-judge roles use a
DIFFERENT family — **Google Gemini (`gemini-3.5-flash`)** — for (a) the prior-only baseline (the
anti-cliché / difficulty meter) and (b) the faithfulness judge (does the world match the prose?).
The structure grader (T3) is purely statistical (no LLM). So three separate "brains": Claude
authors · statistics grades structure · Gemini provides the prior + faithfulness. Gemini key is
read from the env var GEMINI_API_KEY (never hardcoded); absent → falls back to a hand-authored
prior, labelled.

Honesty rails: discoverer never sees TRUTH (only scoring does); generic world-agnostic sampler +
discoverer; thresholds normalized per-world (vs each world's OWN null); anti-cliché is a difficulty
LABEL not a reject; N=4 — a first signal, not proof; failures reported, not hidden.
"""
import math, os, re, json, urllib.request, urllib.error, numpy as np
from itertools import combinations

NSEED = int(os.environ.get("NSEED", "11")); NOISE = float(os.environ.get("NOISE", "0.3"))
NW = 8000
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash"        # latest GA flagship (verified 2026-06; 3.5-pro N/A)

def thr(x): return np.maximum(0.0, x)

# ---------- the 4 worlds, AUTHORED FROM PROSE ----------
def worlds(rng):
    e = lambda s=NOISE: rng.normal(0, s, NW)
    def H(): return rng.normal(0, 1, NW)
    W = []
    def w1(do):
        R   = do.get("R", rng.integers(0,2,NW).astype(float)); Flu = H()
        Staff = do.get("Staff", rng.normal(0,1,NW)); Tests = do.get("Tests", rng.normal(0,1,NW))
        Arr = do.get("Arrivals", 0.9*Flu + e()); Adm = do.get("Admissions", 0.9*Flu + e())
        Wait = do.get("Wait", 0.8*Arr - 0.7*Staff + e())
        Thr_ = do.get("Throughput", (np.where(R==1,-1.0,1.0))*Tests + 0.4*Staff + e())
        return {"R":R,"Staff":Staff,"Tests":Tests,"Arrivals":Arr,"Admissions":Adm,"Wait":Wait,"Throughput":Thr_}
    W.append(dict(name="hospital_ed", sample=w1, ctx=["R"],
        prose=("An emergency department. More staff on shift eases crowding (shorter waits). On normal days, "
               "ordering more diagnostic tests speeds correct routing (higher throughput); during a flu surge, "
               "tests overwhelm the labs and slow throughput. A flu wave drives patient arrivals and admissions."),
        obs=["R","Staff","Tests","Arrivals","Admissions","Wait","Throughput"],
        roles=dict(R="disturbance",Staff="controllable",Tests="controllable",Arrivals="observable",
                   Admissions="outcome",Wait="outcome",Throughput="outcome"),
        truth={("Staff","Wait"),("Arrivals","Wait"),("R","Throughput"),("Tests","Throughput"),("Staff","Throughput")},
        prior={("Arrivals","Wait"),("Staff","Wait"),("Tests","Throughput"),("Staff","Throughput"),("Arrivals","Admissions")}))
    def w2(do):
        R = do.get("R", rng.integers(0,2,NW).astype(float)); Event = H()
        Price = do.get("Price", rng.normal(0,1,NW))
        Dem = do.get("Demand", (np.where(R==1,-0.1,-1.0))*Price + 0.8*Event + e())
        Sup = do.get("Supply", 0.8*Event + e()); Wait = do.get("Wait", 0.9*Dem - 0.8*Sup + e())
        return {"R":R,"Price":Price,"Demand":Dem,"Supply":Sup,"Wait":Wait}
    W.append(dict(name="ride_hailing", sample=w2, ctx=["R"],
        prose=("A ride-hailing market with surge pricing. Normally a higher price lowers rider demand; during "
               "big-event periods demand is inelastic. Driver supply and demand both rise around big events."),
        obs=["R","Price","Demand","Supply","Wait"],
        roles=dict(R="disturbance",Price="controllable",Demand="observable",Supply="observable",Wait="outcome"),
        truth={("R","Demand"),("Price","Demand"),("Demand","Wait"),("Supply","Wait")},
        prior={("Price","Demand"),("Demand","Wait"),("Supply","Wait"),("Demand","Supply")}))
    def w3(do):
        Mat = H(); Speed = do.get("Speed", rng.normal(0,1,NW))
        Out = do.get("Output", 1.0*Speed + 0.6*Mat + e()); Def = do.get("Defects", thr(Speed-0.3) - 0.8*Mat + e())
        Good = do.get("GoodUnits", 1.0*Out - 2.0*Def + e())
        return {"Speed":Speed,"Output":Out,"Defects":Def,"GoodUnits":Good}
    W.append(dict(name="factory_line", sample=w3, ctx=[],
        prose=("A factory line. Higher line speed raises output, but past a threshold it also raises the defect "
               "rate; defects reduce net good units. Raw-material quality affects defects and output."),
        obs=["Speed","Output","Defects","GoodUnits"],
        roles=dict(Speed="controllable",Output="observable",Defects="observable",GoodUnits="outcome"),
        truth={("Speed","Output"),("Speed","Defects"),("Output","GoodUnits"),("Defects","GoodUnits")},
        prior={("Speed","Output"),("Speed","Defects"),("Output","GoodUnits"),("Defects","GoodUnits"),("Defects","Output")}))
    def w4(do):
        Ad = do.get("Ad", rng.normal(0,1,NW)); Disc = do.get("Discount", rng.normal(0,1,NW))
        Traf = do.get("Traffic", 1.0*Ad + e()); Sales = do.get("Sales", 1.0*Traf + 0.8*Disc + e())
        return {"Ad":Ad,"Discount":Disc,"Traffic":Traf,"Sales":Sales}
    W.append(dict(name="ecommerce", sample=w4, ctx=[],
        prose=("An online store. More ad spend drives site traffic, which drives sales; larger discounts also "
               "increase sales."),
        obs=["Ad","Discount","Traffic","Sales"],
        roles=dict(Ad="controllable",Discount="controllable",Traffic="observable",Sales="outcome"),
        truth={("Ad","Traffic"),("Traffic","Sales"),("Discount","Sales")},
        prior={("Ad","Traffic"),("Traffic","Sales"),("Discount","Sales")}))
    # ---- World 5 — LOGISTICS NETWORK (SCALE: 10 obs nodes, 2 hidden confounders, 1 regime flip) ----
    # "A regional logistics network. More trucks raise on-time delivery, which raises satisfaction. Fuel price
    #  raises shipping cost. In peak season adding trucks congests hubs and LOWERS on-time delivery (flip). Weather
    #  drives demand and transit delays; supplier reliability drives lead time and defect rate."
    def w5(do):
        R=do.get("R", rng.integers(0,2,NW).astype(float)); Wx=H(); Sup=H()   # Wx, Sup hidden
        Tr=do.get("Trucks", rng.normal(0,1,NW)); Fu=do.get("FuelPrice", rng.normal(0,1,NW))
        Dem=do.get("Demand", 0.8*Wx+e()); Trn=do.get("TransitDelay", 0.8*Wx+e())
        Lead=do.get("LeadTime", 0.8*Sup+e()); Def=do.get("Defects", 0.8*Sup+e())
        On=do.get("OnTime", np.where(R==1,-1.0,1.0)*Tr - 0.5*Trn - 0.3*Dem + e())
        Cost=do.get("ShipCost", 0.7*Fu + 0.3*Tr + e()); Sat=do.get("Satisfaction", 0.8*On - 0.5*Def + e())
        return {"R":R,"Trucks":Tr,"FuelPrice":Fu,"Demand":Dem,"TransitDelay":Trn,"LeadTime":Lead,
                "Defects":Def,"OnTime":On,"ShipCost":Cost,"Satisfaction":Sat}
    W.append(dict(name="logistics", sample=w5, ctx=["R"],
        prose=("A regional logistics network. More trucks raise on-time delivery, which raises customer "
               "satisfaction. Fuel price raises shipping cost. In peak season, adding trucks congests hubs and "
               "lowers on-time delivery. Weather drives demand and transit delays; supplier reliability drives "
               "lead time and defect rate."),
        obs=["R","Trucks","FuelPrice","Demand","TransitDelay","LeadTime","Defects","OnTime","ShipCost","Satisfaction"],
        roles=dict(R="disturbance",Trucks="controllable",FuelPrice="disturbance",Demand="observable",
                   TransitDelay="observable",LeadTime="observable",Defects="observable",OnTime="outcome",
                   ShipCost="outcome",Satisfaction="outcome"),
        truth={("Trucks","OnTime"),("R","OnTime"),("TransitDelay","OnTime"),("Demand","OnTime"),
               ("FuelPrice","ShipCost"),("Trucks","ShipCost"),("OnTime","Satisfaction"),("Defects","Satisfaction")},
        prior={("Trucks","OnTime"),("OnTime","Satisfaction"),("FuelPrice","ShipCost"),("Trucks","ShipCost"),
               ("Demand","TransitDelay"),("LeadTime","Defects"),("Demand","OnTime")}))  # naive: keeps both spurious confounded pairs
    return W

# ---------- Gemini (independent LLM judge: prior-only baseline + faithfulness) ----------
def gemini(prompt, temperature=0.0):
    if not GEMINI_KEY: return None
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    body=json.dumps({"contents":[{"parts":[{"text":prompt}]}],
                     "generationConfig":{"temperature":temperature}}).encode()
    req=urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"})
    try:
        r=urllib.request.urlopen(req,timeout=60); j=json.loads(r.read())
        return j["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as ex:
        print(f"   [gemini error: {type(ex).__name__}]"); return None

def gemini_prior(world):
    """Independent prior-only baseline: Gemini guesses edges from prose + var names, NO data."""
    vs=", ".join(f"{n} ({world['roles'][n]})" for n in world["obs"])
    txt=gemini(f"You are guessing the cause-and-effect structure of an operation from its description ALONE — "
               f"you have NO data.\nOperation: {world['prose']}\nVariables (use ONLY these exact names): {vs}.\n"
               f"List every directed causal edge you expect, one per line, as 'SOURCE -> TARGET'. "
               f"Output only edges, no prose.")
    if txt is None: return None
    names={n.lower():n for n in world["obs"]}; got=set()
    for a,b in re.findall(r'([A-Za-z_]+)\s*(?:->|-->|→)\s*([A-Za-z_]+)', txt):
        if a.lower() in names and b.lower() in names and a.lower()!=b.lower():
            got.add((names[a.lower()],names[b.lower()]))
    return got

def gemini_faithfulness(world):
    """Independent judge: does the authored structure faithfully represent the prose? (success #6)"""
    edges=", ".join(f"{a}->{b}" for a,b in sorted(world["truth"]))
    vs=", ".join(f"{n} ({world['roles'][n]})" for n in world["obs"])
    txt=gemini(f"Operation description: {world['prose']}\nA causal model uses variables: {vs}\n"
               f"and directed causal links: {edges}\nDoes this model faithfully and plausibly represent the "
               f'description? Reply ONLY a JSON object: {{"score": <0.0-1.0>, "note": "<one sentence>"}}.')
    if txt is None: return (None,"no-gemini")
    m=re.search(r'\{.*\}', txt, re.S)
    if not m: return (None,"unparseable")
    try:
        j=json.loads(m.group(0)); return (float(j.get("score")), str(j.get("note",""))[:80])
    except Exception: return (None,"unparseable")

# ---------- generic stats + discoverer (spike #2's rule, world-agnostic) ----------
def resid(y,Z):
    Z1=np.column_stack([np.ones(len(y)),Z]) if Z.size else np.ones((len(y),1))
    c,*_=np.linalg.lstsq(Z1,y,rcond=None); return y-Z1@c
def pcorr(M,i,j,cond):
    Z=M[:,cond] if cond else np.empty((len(M),0)); ri,rj=resid(M[:,i],Z),resid(M[:,j],Z)
    if ri.std()<1e-9 or rj.std()<1e-9: return 0.0,1.0
    r=np.corrcoef(ri,rj)[0,1]; n,k=len(M),len(cond)
    if abs(r)>=1: return r,0.0
    z=0.5*math.log((1+r)/(1-r))*math.sqrt(max(n-k-3,1)); return r,math.erfc(abs(z)/math.sqrt(2))
def discover(world,rng):
    obs,ctx=world["obs"],world["ctx"]; idx={n:i for i,n in enumerate(obs)}; m=len(obs); cidx=[idx[c] for c in ctx]
    mat=lambda d: np.column_stack([d[n] for n in obs])
    def strata(M):
        cs=[M]
        for ci in cidx:
            for lv in (0,1):
                s=M[M[:,ci]==lv]
                if len(s)>50: cs.append(s)
        return cs
    def eff(M,v,w):
        best=0.0
        for Mt in strata(M):
            A=np.column_stack([np.ones(len(Mt)),Mt[:,v]]); s=np.linalg.lstsq(A,Mt[:,w],rcond=None)[0][1]
            if abs(s)>abs(best): best=s
        return best
    def dep(M,v,w,cond):
        best=0.0
        for Mt in strata(M):
            r,p=pcorr(Mt,v,w,[c for c in cond if c not in cidx])
            if abs(r)>=0.08 and p<=1e-3 and abs(r)>abs(best): best=r
        return best
    doM={}; desc={v:set() for v in range(m)}
    for v in range(m):
        dov=rng.integers(0,2,NW).astype(float) if v in cidx else rng.normal(0.5,1.5,NW)
        Mv=mat(world["sample"]({obs[v]:dov})); doM[v]=Mv
        for w in range(m):
            if w!=v and abs(eff(Mv,v,w))>=0.08: desc[v].add(w)
    edges=set()
    for v in range(m):
        Mv=doM[v]
        for w in desc[v]:
            anc=[u for u in range(m) if u not in (v,w) and w in desc[u]]
            if abs(dep(Mv,v,w,anc))>=0.08: edges.add((obs[v],obs[w]))
    return edges

def dshd(a,b):
    miss=sum(1 for x in b if x not in a and (x[1],x[0]) not in a)
    extra=sum(1 for x in a if x not in b and (x[1],x[0]) not in b)
    rev=sum(1 for x in a if x not in b and (x[1],x[0]) in b)
    return miss+extra+rev
def null_shd(world,rng,reps=2000):
    obs=world["obs"]; m=len(obs); k=len(world["truth"])
    pairs=[(obs[i],obs[j]) for i in range(m) for j in range(m) if i!=j]
    return float(np.mean([dshd({pairs[t] for t in rng.choice(len(pairs),k,replace=False)},world["truth"]) for _ in range(reps)]))
def t1_static(world):
    import collections
    g=collections.defaultdict(list)
    for a,b in world["truth"]: g[a].append(b)
    color=collections.defaultdict(int); ok=[True]
    def dfs(u):
        color[u]=1
        for w in g[u]:
            if color[w]==1: ok[0]=False
            elif color[w]==0: dfs(w)
        color[u]=2
    for n in world["obs"]:
        if color[n]==0: dfs(n)
    roles=set(world["roles"].values())
    return ok[0] and "controllable" in roles and "outcome" in roles

def run(seed, use_gemini=False):
    rng=np.random.default_rng(seed); res=[]
    for w in worlds(rng):
        t1=t1_static(w); M=np.column_stack([w["sample"]({})[n] for n in w["obs"]])
        t2=(not np.any(~np.isfinite(M))) and bool(np.all(M.std(0)>1e-6))
        got=discover(w,rng); shd=dshd(got,w["truth"]); nul=null_shd(w,rng)
        if use_gemini and GEMINI_KEY:
            gp=gemini_prior(w); prior=gp if gp is not None else w["prior"]; src="gemini" if gp is not None else "fallback"
            faith=gemini_faithfulness(w)
        else:
            prior=w["prior"]; src="hand"; faith=(None,"")
        pshd=dshd(prior,w["truth"]); t3=shd<=max(1,0.25*nul)
        res.append(dict(name=w["name"],n=len(w["obs"]),t1=t1,t2=t2,shd=shd,nul=nul,t3=t3,
                        pshd=pshd,gap=pshd-shd,src=src,faith=faith,got=got,truth=w["truth"]))
    return res

r=run(NSEED, use_gemini=True)
print("="*92)
print(f"BUILD-TASK-0 — AUTHOR SPIKE  |  author=Claude · structure-grader=statistical · prior+judge={GEMINI_MODEL}")
print(f"   (Gemini {'ACTIVE' if GEMINI_KEY else 'ABSENT → hand-authored prior fallback'})")
print("="*92)
print(f"{'world':<14}{'nodes':>6}{'T1':>4}{'T2':>4}{'iSHD':>6}{'null':>7}{'T3':>4}{'priorSHD':>10}{'gap':>5}{'faith':>7}  difficulty (prior=independent)")
print("-"*92)
def label(gap,pshd): return "easy / cliché" if pshd<=1 else ("hard / anti-cliché" if gap>=2 else "medium")
for x in r:
    fs=f"{x['faith'][0]:.2f}" if x['faith'][0] is not None else "—"
    print(f"{x['name']:<14}{x['n']:>6}{'Y' if x['t1'] else 'N':>4}{'Y' if x['t2'] else 'N':>4}{x['shd']:>6}"
          f"{x['nul']:>7.1f}{'Y' if x['t3'] else 'N':>4}{x['pshd']:>8}({x['src'][:1]}){x['gap']:>5}{fs:>7}  {label(x['gap'],x['pshd'])}")
print("-"*92)
print(f"first-try pass (T1&T2&T3): {sum(1 for x in r if x['t1'] and x['t2'] and x['t3'])}/{len(r)}")
print("\nROBUSTNESS (T1&T2&T3 pass-count across seeds 11..15, statistical grader only):")
agg={x['name']:0 for x in r}
for s in range(11,16):
    for x in run(s, use_gemini=False): agg[x['name']]+=int(x['t1'] and x['t2'] and x['t3'])
for k,v in agg.items(): print(f"   {k:<14} {v}/5")
print("\nNOTE: prior 'gap' now measured by an INDEPENDENT model (Gemini), not the author. faith = Gemini's")
print("faithfulness-to-prose score (success #6). T3 grader = spike #2's interventional-CI prototype; build-task-1")
print("hardens it. Vetted GES/GIES is NOT the tool — it fails the confounder trap (see spike_grader.py).")
print("Structure-SHD is blind to sign-flips: v0 difficulty must come from STRUCTURAL traps, not regime sign-flips.")
