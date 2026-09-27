#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/HBF_RAG_Sim/g1_quick_real
DEPS="$ROOT/.deps"
export PYTHONPATH="$DEPS:${PYTHONPATH:-}"
cd "$ROOT"
/opt/conda/bin/python - <<'PY'
import json, os, heapq
from pathlib import Path
from collections import defaultdict, deque
import numpy as np, faiss

ROOT=Path("/workspace/HBF_RAG_Sim/g1_quick_real")
K=10; PAGE=4096; NB=256; TR=4.0; TC=0.2

def train_pq(X,M):
    idx=faiss.IndexPQ(X.shape[1],M,8,faiss.METRIC_L2)
    idx.pq.cp.niter=12; idx.pq.cp.nredo=1
    idx.train(X); idx.add(X); return idx

def reconstruct(idx,n,d):
    try:
        z=idx.reconstruct_n(0,n)
        if z is not None:return np.asarray(z,dtype="float32").reshape(n,d)
    except Exception:pass
    return np.vstack([idx.reconstruct(i) for i in range(n)]).astype("float32")

def traces_for(X,Q,idx,Xh,R,seed):
    _,C=idx.search(Q,R)
    flat=faiss.IndexFlatL2(X.shape[1]); flat.add(X); _,GT=flat.search(Q,K)
    vpp=max(1,PAGE//(X.shape[1]*2))
    rg=np.random.default_rng(seed); perm=rg.permutation(len(X)); pos=np.empty(len(X),dtype=np.int64);pos[perm]=np.arange(len(X));pgof=pos//vpp
    trs=[]; oc=[];op=[]; cr=[]; rr=[]
    for qi,q in enumerate(Q):
        c=C[qi]; exact=np.linalg.norm(X[c]-q,axis=1); approx=np.linalg.norm(Xh[c]-q,axis=1); err=np.linalg.norm(X[c]-Xh[c],axis=1)
        L=np.maximum(0,approx-err); U=approx+err; th=float(np.partition(exact,K-1)[K-1]); pr=L>th
        cr.append(len(set(GT[qi])&set(c))/K); rrid=c[np.argsort(exact)[:K]]; rr.append(len(set(GT[qi])&set(rrid))/K)
        pages=defaultdict(list)
        for li,v in enumerate(c):pages[int(pgof[v])].append(li)
        pm={}; ppr=[]
        for p,ii in pages.items():
            ii=np.asarray(ii); pm[p]={"inds":ii,"L":float(np.min(L[ii])),"priority":float(np.min(approx[ii])),"bank":int(p%NB)}
            ppr.append(bool(np.all(pr[ii])))
        oc.append(float(np.mean(pr)));op.append(float(np.mean(ppr)))
        trs.append({"exact":exact,"L":L,"U":U,"pages":pm,"tf":th})
    return trs,{"cand_R10":float(np.mean(cr)),"rerank_R10":float(np.mean(rr)),"oracle_c":float(np.mean(oc)),"oracle_p":float(np.mean(op)),"vpp":vpp}

def theta(st):
    vals=list(st["rex"]); active=~(st["removed"]|st["resolved"]); vals.extend(st["tr"]["U"][active].tolist())
    if len(vals)<K:return float("inf")
    a=np.asarray(vals);return float(np.partition(a,K-1)[K-1])

def sim(trs,qids,W,pre):
    qs={}
    for q in qids:
        tr=trs[q];order=sorted(tr["pages"],key=lambda p:tr["pages"][p]["priority"]);m=len(tr["exact"])
        qs[q]={"tr":tr,"pend":deque(order),"inf":0,"removed":np.zeros(m,bool),"resolved":np.zeros(m,bool),"rex":[],"th":float(np.partition(tr["U"],K-1)[K-1]),"sub":{},"sen":{},"pt":{},"sense":0,"dev":0}
    bq=[deque() for _ in range(NB)];busy=[False]*NB;ev=[];seq=0
    def push(t,p,ty,*d):
        nonlocal seq;seq+=1;heapq.heappush(ev,(t,p,seq,ty,d))
    def mark(q,t):
        s=qs[q]
        for p,g in s["tr"]["pages"].items():
            if p not in s["pt"] and g["L"]>s["th"]:s["pt"][p]=t
    def rec(q,t):
        s=qs[q];old=s["th"];s["th"]=theta(s)
        if s["th"]<old-1e-12:mark(q,t)
    def rem(q,p,t):
        s=qs[q];s["removed"][s["tr"]["pages"][p]["inds"]]=True;rec(q,t)
    def ref(q,t):
        s=qs[q]
        while s["pend"] and s["inf"]<(len(s["tr"]["pages"]) if W=="all" else W):
            p=s["pend"].popleft();b=s["tr"]["pages"][p]["bank"];s["sub"][p]=t;s["inf"]+=1;bq[b].append((q,p))
            if not busy[b] and len(bq[b])==1:push(t,2,"sense",b)
    for q in qids:mark(q,0);ref(q,0)
    while ev:
        t,_,_,ty,d=heapq.heappop(ev)
        if ty=="sense":
            b=d[0]
            if busy[b] or not bq[b]:continue
            q,p=bq[b][0];s=qs[q];g=s["tr"]["pages"][p]
            if pre and g["L"]>s["th"]:
                bq[b].popleft();s["dev"]+=1;s["inf"]-=1;rem(q,p,t);ref(q,t)
                if bq[b] and not busy[b]:push(t,2,"sense",b)
                continue
            bq[b].popleft();busy[b]=True;s["sen"][p]=t;s["sense"]+=1;push(t+TR,1,"free",b);push(t+TR+TC,0,"fb",q,p)
        elif ty=="free":
            b=d[0];busy[b]=False
            if bq[b]:push(t,2,"sense",b)
        else:
            q,p=d;s=qs[q];ii=s["tr"]["pages"][p]["inds"];ii=ii[~s["removed"][ii]]
            if len(ii):s["resolved"][ii]=True;s["rex"].extend(s["tr"]["exact"][ii].tolist())
            s["inf"]-=1;rec(q,t);ref(q,t)
    oracle=post=pre_sub=late=never=0
    for q in qids:
        s=qs[q]
        for p,g in s["tr"]["pages"].items():
            if g["L"]>s["tr"]["tf"]:
                oracle+=1;tp=s["pt"].get(p);ts=s["sub"].get(p,float("inf"));te=s["sen"].get(p,float("inf"))
                if tp is None:never+=1
                elif tp<ts-1e-12:pre_sub+=1
                elif tp<te-1e-12:post+=1
                else:late+=1
    return {"oracle":oracle,"pre_submit":pre_sub,"post":post,"late":late,"never":never,"sense":sum(s["sense"] for s in qs.values()),"dev":sum(s["dev"] for s in qs.values())}

out={}
for di,name in enumerate(["scifact","fiqa"]):
    X=np.load(ROOT/"embeddings"/f"{name}_X.npy");Q=np.load(ROOT/"embeddings"/f"{name}_Q.npy")
    out[name]={}
    for M in [48,96,192]:
        idx=train_pq(X,M);Xh=reconstruct(idx,len(X),X.shape[1]);lab=f"PQ{M}B";out[name][lab]={}
        for R in [256,512,1024]:
            trs,m=traces_for(X,Q,idx,Xh,R,seed=4200+di*100+M+R)
            cells=[]
            for c in [8,32]:
                q=list(range(min(c,len(trs))))
                for W in [128,"all"]:
                    b=sim(trs,q,W,False);p=sim(trs,q,W,True)
                    cells.append({"c":c,"W":W,"oracle":b["oracle"],"post_share":b["post"]/max(1,b["oracle"]),"pre_submit_share":b["pre_submit"]/max(1,b["oracle"]),"late_share":b["late"]/max(1,b["oracle"]),"dev":p["dev"],"sense0":b["sense"],"sense_pre":p["sense"],"sense_red":(b["sense"]-p["sense"])/max(1,b["sense"])})
            out[name][lab][str(R)]={"m":m,"timing":cells}
job=Path(os.environ["SERVER_JOB_DIR"])
(job/"result_summary.json").write_text(json.dumps(out,separators=(",",":"))+"\n")
Path("results/bound_sensitivity.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,ensure_ascii=False))
PY
