#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/HBF_RAG_Sim
mkdir -p "$ROOT/results"
cd "$ROOT"
cat > quick_validate.py <<'PY'
import json, time, heapq
from collections import deque
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

SEED=42; N=8000; D=128; NQ=80; NCLUSTER=64; K=10; R=512; VPP=2; NBANK=256

def build_pq(X,m,ks,seed=0,train_n=6000):
    n,d=X.shape; ds=d//m; Xh=np.empty_like(X); rng=np.random.default_rng(seed)
    tr=rng.choice(n,size=min(train_n,n),replace=False)
    for j in range(m):
        sl=slice(j*ds,(j+1)*ds)
        km=MiniBatchKMeans(n_clusters=ks,random_state=seed+j,batch_size=512,n_init=1,max_iter=50)
        km.fit(X[tr,sl]); c=km.predict(X[:,sl]); Xh[:,sl]=km.cluster_centers_[c]
    return Xh

def make_traces(X,Xh,Q,k=K,R=R,seed=123):
    n=X.shape[0]; err=np.linalg.norm(X-Xh,axis=1); rng=np.random.default_rng(seed)
    perm=rng.permutation(n); pos=np.empty(n,dtype=np.int64); pos[perm]=np.arange(n); page=pos//VPP
    traces=[]; cr=[]; rr=[]; crr=[]; oc=[]; op=[]
    for q in Q:
        da=np.linalg.norm(Xh-q,axis=1); de=np.linalg.norm(X-q,axis=1)
        gt=np.argpartition(de,k)[:k]; topa=np.argpartition(da,k)[:k]; c0=np.argpartition(da,R)[:R]
        cr.append(len(set(gt)&set(topa))/k); crr.append(len(set(gt)&set(c0))/k)
        rer=np.asarray(c0)[np.argsort(de[c0])[:k]]; rr.append(len(set(gt)&set(rer))/k)
        L=np.maximum(0,da[c0]-err[c0]); U=da[c0]+err[c0]; theta=float(np.partition(de[c0],k-1)[k-1]); prune=L>theta
        pages={}
        for local,vid in enumerate(c0): pages.setdefault(int(page[vid]),[]).append(local)
        pmeta={}; ppr=[]
        for p,inds0 in pages.items():
            inds=np.asarray(inds0,dtype=int); ppr.append(bool(np.all(prune[inds])))
            pmeta[p]={"cand_idx":inds,"L":float(np.min(L[inds])),"priority":float(np.min(da[c0][inds])),"bank":int(p%NBANK)}
        oc.append(float(np.mean(prune))); op.append(float(np.mean(ppr)))
        traces.append({"c0":c0,"exact":de[c0],"L":L,"U":U,"pages":pmeta,"theta_final":theta})
    return traces,{"compressed_recall10":float(np.mean(cr)),"candidate_recall10":float(np.mean(crr)),
                   "rerank_recall10":float(np.mean(rr)),"oracle_candidate_prune":float(np.mean(oc)),
                   "oracle_page_prune":float(np.mean(op))}

def theta_now(st,k=K):
    vals=list(st["rex"]); active=~(st["removed"]|st["resolved"]); vals.extend(st["tr"]["U"][active].tolist())
    if len(vals)<k:return float("inf")
    a=np.asarray(vals,float); return float(np.partition(a,k-1)[k-1])

def simulate(traces,qids,policy,W,fb,tR=4.0,tcomp=0.2,nbanks=NBANK,k=K):
    qs={}
    for qid in qids:
        tr=traces[qid]; order=sorted(tr["pages"],key=lambda p:tr["pages"][p]["priority"]); m=len(tr["c0"])
        qs[qid]={"tr":tr,"pending":deque(order),"inflight":0,"removed":np.zeros(m,bool),"resolved":np.zeros(m,bool),
                 "rex":[],"theta":float(np.partition(tr["U"],k-1)[k-1]),"done":None,"host":0,"device":0,"sense":0,"too_late":0}
    bankq=[deque() for _ in range(nbanks)]; busy=[False]*nbanks; ev=[]; seq=0
    def push(t,p,typ,*data):
        nonlocal seq
        seq+=1; heapq.heappush(ev,(t,p,seq,typ,data))
    def recalc(st): st["theta"]=theta_now(st,k)
    def remove(st,p):
        inds=st["tr"]["pages"][p]["cand_idx"]; st["removed"][inds]=True; recalc(st)
    def resolve(st,p):
        inds=st["tr"]["pages"][p]["cand_idx"]; inds=inds[~st["removed"][inds]]
        if len(inds): st["resolved"][inds]=True; st["rex"].extend(st["tr"]["exact"][inds].tolist())
        recalc(st)
    def done(q,t):
        st=qs[q]
        if st["done"] is None and not st["pending"] and st["inflight"]==0: st["done"]=t
    def submit(q,p,t):
        st=qs[q]; b=st["tr"]["pages"][p]["bank"]%nbanks; bankq[b].append((q,p)); st["inflight"]+=1
        if not busy[b] and len(bankq[b])==1: push(t,2,"sense",b)
    def refill(q,t):
        st=qs[q]
        while st["inflight"]<W and st["pending"]:
            p=st["pending"].popleft(); pg=st["tr"]["pages"][p]
            if policy!="B0" and pg["L"]>st["theta"]: st["host"]+=1; remove(st,p); continue
            submit(q,p,t)
        done(q,t)
    for q in qids: refill(q,0.0)
    while ev:
        t,_,_,typ,data=heapq.heappop(ev)
        if typ=="sense":
            b=data[0]
            if busy[b] or not bankq[b]: continue
            q,p=bankq[b][0]; st=qs[q]; pg=st["tr"]["pages"][p]
            if policy=="B2" and pg["L"]>st["theta"]:
                bankq[b].popleft(); st["device"]+=1; st["inflight"]-=1; remove(st,p); refill(q,t)
                if bankq[b] and not busy[b]: push(t,2,"sense",b)
                continue
            bankq[b].popleft(); busy[b]=True; st["sense"]+=1
            if pg["L"]>st["tr"]["theta_final"]: st["too_late"]+=1
            push(t+tR,1,"free",b); push(t+tR+tcomp+fb,0,"feedback",q,p)
        elif typ=="free":
            b=data[0]; busy[b]=False
            if bankq[b]: push(t,2,"sense",b)
        else:
            q,p=data; st=qs[q]; resolve(st,p); st["inflight"]-=1; refill(q,t)
    lats=np.asarray([qs[q]["done"] for q in qids],float)
    return {"policy":policy,"W":W,"feedback_us":fb,"concurrency":len(qids),"sense":int(sum(s["sense"] for s in qs.values())),
            "host_pruned":int(sum(s["host"] for s in qs.values())),"device_pruned":int(sum(s["device"] for s in qs.values())),
            "too_late":int(sum(s["too_late"] for s in qs.values())),"mean_latency_us":float(lats.mean()),
            "p95_latency_us":float(np.percentile(lats,95)),"makespan_us":float(lats.max())}

rng=np.random.default_rng(SEED); centers=rng.normal(size=(NCLUSTER,D)).astype(np.float32); a=rng.integers(0,NCLUSTER,size=N)
X=centers[a]+0.7*rng.normal(size=(N,D)).astype(np.float32); qi=rng.choice(N,size=NQ,replace=False)
Q=X[qi]+0.15*rng.normal(size=(NQ,D)).astype(np.float32)
pq_results={}; trace_bank={}
for m,ks,label in [(16,16,"pq8B_4bit"),(16,256,"pq16B_8bit"),(32,256,"pq32B_8bit")]:
    t=time.time(); Xh=build_pq(X,m,ks,SEED); tr,s=make_traces(X,Xh,Q); s["build_seconds"]=round(time.time()-t,3)
    pq_results[label]=s; trace_bank[label]=tr
traces=trace_bank["pq32B_8bit"]; sims=[]
for conc in (1,8):
    qids=list(range(conc))
    for W in (8,32,128):
        for fb in (1.0,4.0,16.0):
            for pol in ("B0","B1","B2"): sims.append(simulate(traces,qids,pol,W,fb))
out={"config":{"N":N,"D":D,"queries":NQ,"k":K,"R":R,"vectors_per_page":VPP,"banks":NBANK},"pq":pq_results,"simulations":sims}
Path("results").mkdir(exist_ok=True); Path("results/quick_validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
lines=["# HBF-RAG quick validation","","## PQ / rerank / bound opportunity"]
for name,s in pq_results.items():
    lines.append(f"- {name}: compressed R@10={s['compressed_recall10']:.3f}, rerank R@10={s['rerank_recall10']:.3f}, C0 recall={s['candidate_recall10']:.3f}, oracle candidate prune={s['oracle_candidate_prune']:.3f}, oracle page prune={s['oracle_page_prune']:.3f}")
lines+=["","## B1 -> B2 timing checks (pq32B_8bit, feedback=4us)"]
for conc in (1,8):
    for W in (8,32,128):
        b1=next(x for x in sims if x["concurrency"]==conc and x["W"]==W and x["feedback_us"]==4.0 and x["policy"]=="B1")
        b2=next(x for x in sims if x["concurrency"]==conc and x["W"]==W and x["feedback_us"]==4.0 and x["policy"]=="B2")
        red=(b1["sense"]-b2["sense"])/b1["sense"] if b1["sense"] else 0; lat=(b1["mean_latency_us"]-b2["mean_latency_us"])/b1["mean_latency_us"] if b1["mean_latency_us"] else 0
        lines.append(f"- concurrency={conc}, W={W}: B1 sense={b1['sense']}, B2 sense={b2['sense']}, device-pruned={b2['device_pruned']}, sense delta={red:.1%}, mean latency delta={lat:.1%}")
Path("results/summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print("\n".join(lines))

PY
cat > README.md <<'MD'
# HBF_RAG_Sim

Quick feasibility prototype for the SHARE-RAG assumptions.

This first pass is diagnostic only: synthetic clustered vectors + product quantization + a parameterized HBF discrete-event model. It is not evidence from a real RAG corpus or real HBF hardware.

Outputs:
- `results/quick_validation.json`
- `results/summary.md`

Next stage: real SciFact/FiQA traces with strong PQ/OPQ/RaBitQ baselines, then reuse the event simulator.
MD
/opt/conda/bin/python quick_validate.py | tee results/run.log
echo
echo "== persisted files =="
find "$ROOT" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
