#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/HBF_RAG_Sim/sweetspot_768
BASE=/workspace/HBF_RAG_Sim/g1_quick_real
export PYTHONPATH="$BASE/.deps:${PYTHONPATH:-}"
cd "$ROOT"
/opt/conda/bin/python - <<'PY'
import json,os,heapq
from pathlib import Path
from collections import defaultdict,deque
import numpy as np,faiss
K=10;R=512;PAGE=4096;NB=256;TR=4.0;TC=0.2

def reconstruct_all(idx,n,d):
    try:
        z=idx.reconstruct_n(0,n)
        if z is not None:return np.asarray(z,dtype="float32").reshape(n,d)
    except:pass
    try:return np.vstack([idx.reconstruct(i) for i in range(n)]).astype("float32")
    except:return None

def traces(X,Xh,Q,C,seed):
    vpp=max(1,PAGE//(X.shape[1]*2));err=np.linalg.norm(X-Xh,axis=1)
    rg=np.random.default_rng(seed);perm=rg.permutation(len(X));pos=np.empty(len(X),dtype=np.int64);pos[perm]=np.arange(len(X));pgof=pos//vpp
    ts=[];ocs=[];ops=[]
    for qi,q in enumerate(Q):
        c=C[qi];ex=np.linalg.norm(X[c]-q,axis=1);ap=np.linalg.norm(Xh[c]-q,axis=1)
        L=np.maximum(0,ap-err[c]);U=ap+err[c];tf=float(np.partition(ex,K-1)[K-1]);pr=L>tf
        pages=defaultdict(list)
        for li,v in enumerate(c):pages[int(pgof[v])].append(li)
        pm={};pp=[]
        for p,ii0 in pages.items():
            ii=np.asarray(ii0);pm[p]={"inds":ii,"L":float(np.min(L[ii])),"priority":float(np.min(ap[ii])),"bank":int(p%NB)};pp.append(bool(np.all(pr[ii])))
        ocs.append(float(np.mean(pr)));ops.append(float(np.mean(pp)));ts.append({"ex":ex,"U":U,"pages":pm,"tf":tf})
    return ts,{"vpp":vpp,"oc":float(np.mean(ocs)),"op":float(np.mean(ops)),"errMean":float(err.mean()),"errP95":float(np.percentile(err,95))}

def theta(s):
    vals=list(s["rex"]);active=~(s["rem"]|s["res"]);vals.extend(s["tr"]["U"][active].tolist())
    if len(vals)<K:return float("inf")
    a=np.asarray(vals);return float(np.partition(a,K-1)[K-1])

def sim(ts,pre):
    qids=list(range(min(32,len(ts))));qs={}
    for q in qids:
        tr=ts[q];order=sorted(tr["pages"],key=lambda p:tr["pages"][p]["priority"]);m=len(tr["ex"])
        qs[q]={"tr":tr,"pend":deque(order),"rem":np.zeros(m,bool),"res":np.zeros(m,bool),"rex":[],"th":float(np.partition(tr["U"],K-1)[K-1]),"sub":{},"sen":{},"pt":{},"sense":0,"dev":0}
    bq=[deque() for _ in range(NB)];busy=[False]*NB;ev=[];seq=0
    def push(t,p,ty,*d):
        nonlocal seq;seq+=1;heapq.heappush(ev,(t,p,seq,ty,d))
    def mark(q,t):
        s=qs[q]
        for p,g in s["tr"]["pages"].items():
            if p not in s["pt"] and g["L"]>s["th"]:s["pt"][p]=t
    def rec(q,t):
        s=qs[q];o=s["th"];s["th"]=theta(s)
        if s["th"]<o-1e-12:mark(q,t)
    def rm(q,p,t):
        s=qs[q];s["rem"][s["tr"]["pages"][p]["inds"]]=True;rec(q,t)
    for q,s in qs.items():
        mark(q,0)
        while s["pend"]:
            p=s["pend"].popleft();b=s["tr"]["pages"][p]["bank"];s["sub"][p]=0;bq[b].append((q,p))
            if not busy[b] and len(bq[b])==1:push(0,2,"sense",b)
    while ev:
        t,_,_,ty,d=heapq.heappop(ev)
        if ty=="sense":
            b=d[0]
            if busy[b] or not bq[b]:continue
            q,p=bq[b][0];s=qs[q];g=s["tr"]["pages"][p]
            if pre and g["L"]>s["th"]:
                bq[b].popleft();s["dev"]+=1;rm(q,p,t)
                if bq[b] and not busy[b]:push(t,2,"sense",b)
                continue
            bq[b].popleft();busy[b]=True;s["sen"][p]=t;s["sense"]+=1;push(t+TR,1,"free",b);push(t+TR+TC,0,"fb",q,p)
        elif ty=="free":
            b=d[0];busy[b]=False
            if bq[b]:push(t,2,"sense",b)
        else:
            q,p=d;s=qs[q];ii=s["tr"]["pages"][p]["inds"];ii=ii[~s["rem"][ii]]
            if len(ii):s["res"][ii]=True;s["rex"].extend(s["tr"]["ex"][ii].tolist())
            rec(q,t)
    oracle=post=late=0
    for q,s in qs.items():
        for p,g in s["tr"]["pages"].items():
            if g["L"]>s["tr"]["tf"]:
                oracle+=1;tp=s["pt"].get(p);te=s["sen"].get(p,float("inf"))
                if tp is not None and 0<=tp<te:post+=1
                else:late+=1
    return {"oracle":oracle,"post":post,"late":late,"sense":sum(s["sense"] for s in qs.values()),"dev":sum(s["dev"] for s in qs.values())}

out={}
for di,name in enumerate(["scifact","fiqa"]):
    X=np.load(Path("embeddings")/f"{name}_X.npy");Q=np.load(Path("embeddings")/f"{name}_Q.npy")
    flat=faiss.IndexFlatL2(X.shape[1]);flat.add(X);_,GT=flat.search(Q,K)
    idx=faiss.IndexRaBitQ(X.shape[1],faiss.METRIC_L2);idx.train(X);idx.add(X)
    Xh=reconstruct_all(idx,len(X),X.shape[1])
    out[name]={}
    for qb in [2,4,8]:
        idx.qb=qb;_,I=idx.search(Q,R)
        cr=[];cand=[];rr=[]
        for i in range(len(Q)):
            gt=set(GT[i]);cr.append(len(gt&set(I[i,:K]))/K);cand.append(len(gt&set(I[i]))/K)
            c=I[i];ed=np.linalg.norm(X[c]-Q[i],axis=1);rid=c[np.argsort(ed)[:K]];rr.append(len(gt&set(rid))/K)
        z={"qb":qb,"codeB":int(getattr(idx,"code_size",0)),"cR10":float(np.mean(cr)),"candR10":float(np.mean(cand)),"rrR10":float(np.mean(rr))}
        if Xh is not None:
            ts,m=traces(X,Xh,Q,I,6200+di*10+qb);b=sim(ts,False);p=sim(ts,True)
            z.update(m);z.update({"postShare":b["post"]/max(1,b["oracle"]),"senseRed":(b["sense"]-p["sense"])/max(1,b["sense"]),"dev":p["dev"],"baselineSense":b["sense"]})
        out[name][f"RaBitQ_qb{qb}"]=z
job=Path(os.environ["SERVER_JOB_DIR"]);(job/"result_summary.json").write_text(json.dumps(out,separators=(",",":"))+"\n")
Path("results/rabitq_768_results.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,ensure_ascii=False,indent=2))
PY
