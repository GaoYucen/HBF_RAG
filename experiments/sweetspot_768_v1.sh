#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/HBF_RAG_Sim/sweetspot_768
BASE=/workspace/HBF_RAG_Sim/g1_quick_real
DEPS="$BASE/.deps"
CACHE=/workspace/HBF_RAG_Sim/hf_cache
mkdir -p "$ROOT/results" "$ROOT/embeddings"
export PYTHONPATH="$DEPS:${PYTHONPATH:-}"
export HF_HOME="$CACHE"
export TRANSFORMERS_CACHE="$CACHE"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

/opt/conda/bin/python - <<'PY'
import os, json, csv, math, time, heapq, traceback
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

SEED=42
K=10
R=512
PAGE_BYTES=4096
NBANK=256
TR_US=4.0
TCOMP_US=0.2
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
BASE=Path("/workspace/HBF_RAG_Sim/g1_quick_real")
ROOT=Path("/workspace/HBF_RAG_Sim/sweetspot_768")
DATA=BASE/"data"
OUT=ROOT/"results"
OUT.mkdir(parents=True,exist_ok=True)

def read_jsonl(path):
    out=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def load_beir(name,max_queries=64,corpus_cap=6000):
    base=DATA/name
    corpus_rows=read_jsonl(base/"corpus.jsonl")
    query_rows=read_jsonl(base/"queries.jsonl")
    qrels={}
    with open(base/"qrels"/"test.tsv",encoding="utf-8") as f:
        rd=csv.DictReader(f,delimiter="\t")
        for row in rd:
            qid=str(row["query-id"]); did=str(row["corpus-id"]); score=int(row["score"])
            if score>0:qrels.setdefault(qid,{})[did]=score
    qmap={str(x["_id"]):x["text"] for x in query_rows}
    valid=[qid for qid in qrels if qid in qmap]
    rr=np.random.default_rng(SEED+(0 if name=="scifact" else 100))
    rr.shuffle(valid); qids=valid[:min(max_queries,len(valid))]
    must=set()
    for qid in qids:must.update(qrels[qid])
    cmap={str(x["_id"]):x for x in corpus_rows}
    must=[d for d in must if d in cmap]
    remain=[d for d in cmap if d not in set(must)]
    target=min(corpus_cap,len(cmap));take=max(0,target-len(must))
    sampled=rr.choice(remain,size=min(take,len(remain)),replace=False).tolist() if take else []
    dids=list(dict.fromkeys(must+sampled)); didset=set(dids)
    texts=[]
    for did in dids:
        x=cmap[did];title=(x.get("title") or "").strip();text=(x.get("text") or "").strip()
        texts.append((title+"\n"+text).strip())
    return {"name":name,"dids":dids,"texts":texts,"qids":qids,"queries":[qmap[q] for q in qids],
            "qrels":{q:{d:s for d,s in qrels[q].items() if d in didset} for q in qids}}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE",device,"CUDA_VISIBLE_DEVICES",os.environ.get("CUDA_VISIBLE_DEVICES"),flush=True)
tok=AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=os.environ["HF_HOME"])
model=AutoModel.from_pretrained(MODEL_NAME,cache_dir=os.environ["HF_HOME"]).to(device).eval()

@torch.no_grad()
def encode(texts,batch=64,max_length=192):
    arr=[]
    for s in range(0,len(texts),batch):
        z=texts[s:s+batch]
        inp=tok(z,padding=True,truncation=True,max_length=max_length,return_tensors="pt")
        inp={k:v.to(device) for k,v in inp.items()}
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type=="cuda")):
            h=model(**inp).last_hidden_state
        mask=inp["attention_mask"].unsqueeze(-1)
        emb=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
        emb=torch.nn.functional.normalize(emb.float(),p=2,dim=1)
        arr.append(emb.cpu().numpy().astype("float32"))
    return np.concatenate(arr,axis=0)

def exact_topk(X,Q,k=K):
    idx=faiss.IndexFlatL2(X.shape[1]);idx.add(X);return idx.search(Q,k)

def ndcg(rank_ids,qid,qrels):
    rel=qrels.get(qid,{})
    dcg=0.0
    for j,d in enumerate(rank_ids[:10]):
        r=rel.get(d,0)
        if r>0:dcg+=(2**r-1)/math.log2(j+2)
    ideal=sorted(rel.values(),reverse=True)[:10]
    idcg=sum((2**r-1)/math.log2(j+2) for j,r in enumerate(ideal))
    return dcg/idcg if idcg>0 else 0.0

def code_size(idx,label):
    for attr in ("code_size",):
        try:
            v=int(getattr(idx,attr))
            if v>0:return v
        except:pass
    try:
        v=int(idx.sa_code_size())
        if v>0:return v
    except:pass
    if label.startswith("PQ"):
        try:return int(label[2:-1])
        except:pass
    return None

def make_quantizer(X,kind,param):
    d=X.shape[1]
    if kind=="pq":
        M=param
        idx=faiss.IndexPQ(d,M,8,faiss.METRIC_L2)
        idx.pq.cp.niter=10;idx.pq.cp.nredo=1
    elif kind=="opq":
        M=param
        pq=faiss.IndexPQ(d,M,8,faiss.METRIC_L2)
        pq.pq.cp.niter=10;pq.pq.cp.nredo=1
        opq=faiss.OPQMatrix(d,M);opq.niter=6
        idx=faiss.IndexPreTransform(opq,pq)
    elif kind=="rabitq":
        bits=param
        idx=faiss.IndexRaBitQ(d,faiss.METRIC_L2,bits)
        try: idx.qb=8
        except: pass
    else: raise ValueError(kind)
    t=time.time();idx.train(X);train_s=time.time()-t
    idx.add(X)
    return idx,train_s

def reconstruct_all(idx,n,d):
    try:
        z=idx.reconstruct_n(0,n)
        if z is not None:
            z=np.asarray(z,dtype="float32").reshape(n,d)
            if np.isfinite(z).all():return z
    except Exception:
        pass
    try:
        z=np.empty((n,d),dtype="float32")
        for i in range(n):z[i]=idx.reconstruct(i)
        if np.isfinite(z).all():return z
    except Exception:
        pass
    return None

def build_trace(X,Xh,Q,candidates,seed):
    n,d=X.shape
    vpp=max(1,PAGE_BYTES//(d*2))
    rg=np.random.default_rng(seed);perm=rg.permutation(n);pos=np.empty(n,dtype=np.int64);pos[perm]=np.arange(n);page_of=pos//vpp
    err=np.linalg.norm(X-Xh,axis=1)
    trs=[]; oc=[]; op=[]
    for qi,q in enumerate(Q):
        c0=np.asarray(candidates[qi],dtype=np.int64)
        exact=np.linalg.norm(X[c0]-q,axis=1)
        approx=np.linalg.norm(Xh[c0]-q,axis=1)
        L=np.maximum(0.0,approx-err[c0]);U=approx+err[c0]
        tf=float(np.partition(exact,K-1)[K-1])
        cp=L>tf
        pages=defaultdict(list)
        for li,v in enumerate(c0):pages[int(page_of[v])].append(li)
        pm={};pp=[]
        for p,ii0 in pages.items():
            ii=np.asarray(ii0,dtype=np.int64)
            pm[p]={"inds":ii,"L":float(np.min(L[ii])),"priority":float(np.min(approx[ii])),"bank":int(p%NBANK)}
            pp.append(bool(np.all(cp[ii])))
        oc.append(float(np.mean(cp)));op.append(float(np.mean(pp)))
        trs.append({"exact":exact,"U":U,"pages":pm,"tf":tf})
    return trs,{"vpp":vpp,"oracle_candidate_prune":float(np.mean(oc)),"oracle_page_prune":float(np.mean(op)),
                "mean_reconstruction_error":float(err.mean()),"p95_reconstruction_error":float(np.percentile(err,95))}

def theta_now(st):
    vals=list(st["rex"]);active=~(st["removed"]|st["resolved"]);vals.extend(st["tr"]["U"][active].tolist())
    if len(vals)<K:return float("inf")
    a=np.asarray(vals,float);return float(np.partition(a,K-1)[K-1])

def simulate(trs,qids,feedback_us=0.0,device_presense=False):
    qs={}
    for q in qids:
        tr=trs[q];order=sorted(tr["pages"],key=lambda p:tr["pages"][p]["priority"]);m=len(tr["exact"])
        qs[q]={"tr":tr,"pend":deque(order),"inf":0,"removed":np.zeros(m,bool),"resolved":np.zeros(m,bool),
               "rex":[],"th":float(np.partition(tr["U"],K-1)[K-1]),"sub":{},"sen":{},"pt":{},"sense":0,"dev":0}
    bq=[deque() for _ in range(NBANK)];busy=[False]*NBANK;ev=[];seq=0
    def push(t,p,ty,*d):
        nonlocal seq;seq+=1;heapq.heappush(ev,(t,p,seq,ty,d))
    def mark(q,t):
        s=qs[q]
        for p,g in s["tr"]["pages"].items():
            if p not in s["pt"] and g["L"]>s["th"]:s["pt"][p]=t
    def rec(q,t):
        s=qs[q];old=s["th"];s["th"]=theta_now(s)
        if s["th"]<old-1e-12:mark(q,t)
    def rem(q,p,t):
        s=qs[q];s["removed"][s["tr"]["pages"][p]["inds"]]=True;rec(q,t)
    def refill(q,t):
        s=qs[q]
        while s["pend"]:
            p=s["pend"].popleft();b=s["tr"]["pages"][p]["bank"];s["sub"][p]=t;s["inf"]+=1;bq[b].append((q,p))
            if not busy[b] and len(bq[b])==1:push(t,2,"sense",b)
    for q in qids:mark(q,0);refill(q,0)
    while ev:
        t,_,_,ty,d=heapq.heappop(ev)
        if ty=="sense":
            b=d[0]
            if busy[b] or not bq[b]:continue
            q,p=bq[b][0];s=qs[q];g=s["tr"]["pages"][p]
            if device_presense and g["L"]>s["th"]:
                bq[b].popleft();s["dev"]+=1;s["inf"]-=1;rem(q,p,t)
                if bq[b] and not busy[b]:push(t,2,"sense",b)
                continue
            bq[b].popleft();busy[b]=True;s["sen"][p]=t;s["sense"]+=1
            push(t+TR_US,1,"free",b);push(t+TR_US+TCOMP_US+feedback_us,0,"fb",q,p)
        elif ty=="free":
            b=d[0];busy[b]=False
            if bq[b]:push(t,2,"sense",b)
        else:
            q,p=d;s=qs[q];ii=s["tr"]["pages"][p]["inds"];ii=ii[~s["removed"][ii]]
            if len(ii):s["resolved"][ii]=True;s["rex"].extend(s["tr"]["exact"][ii].tolist())
            s["inf"]-=1;rec(q,t)
    oracle=post=presub=late=never=0
    for q in qids:
        s=qs[q]
        for p,g in s["tr"]["pages"].items():
            if g["L"]>s["tr"]["tf"]:
                oracle+=1;tp=s["pt"].get(p);ts=s["sub"].get(p,float("inf"));te=s["sen"].get(p,float("inf"))
                if tp is None:never+=1
                elif tp<ts-1e-12:presub+=1
                elif tp<te-1e-12:post+=1
                else:late+=1
    return {"oracle_pages":oracle,"pre_submit":presub,"post_submit_pre_sense":post,"too_late":late,"never_online":never,
            "sense":sum(s["sense"] for s in qs.values()),"device_pruned":sum(s["dev"] for s in qs.values())}

def eval_method(X,Q,dids,qids,qrels,GT,label,kind,param):
    idx,train_s=make_quantizer(X,kind,param)
    t=time.time();AD,AI=idx.search(Q,R);search_s=time.time()-t
    cr=[];cand=[];rr=[];nc=[];nr=[]
    for qi in range(len(Q)):
        gt=set(GT[qi].tolist());top10=AI[qi,:K];c0=AI[qi,:R]
        cr.append(len(gt&set(top10.tolist()))/K);cand.append(len(gt&set(c0.tolist()))/K)
        ed=np.linalg.norm(X[c0]-Q[qi],axis=1);rid=c0[np.argsort(ed)[:K]]
        rr.append(len(gt&set(rid.tolist()))/K)
        nc.append(ndcg([dids[i] for i in top10],qids[qi],qrels));nr.append(ndcg([dids[i] for i in rid],qids[qi],qrels))
    res={"label":label,"kind":kind,"param":param,"train_s":train_s,
         "search_ms_per_query":1000*search_s/max(1,len(Q)),"code_size_bytes":code_size(idx,label),
         "compressed_R10":float(np.mean(cr)),"candidate_R10":float(np.mean(cand)),"rerank_R10":float(np.mean(rr)),
         "compressed_ndcg10":float(np.mean(nc)),"rerank_ndcg10":float(np.mean(nr))}
    Xh=reconstruct_all(idx,len(X),X.shape[1])
    if Xh is None:
        res["bound_available"]=False
        return res,None
    trs,opp=build_trace(X,Xh,Q,AI[:,:R],SEED+(hash(label)&0xffff))
    res.update(opp);res["bound_available"]=True
    qsel=list(range(min(32,len(trs))))
    b=simulate(trs,qsel,0.0,False);p=simulate(trs,qsel,0.0,True);p4=simulate(trs,qsel,4.0,True)
    denom=max(1,b["oracle_pages"])
    res.update({
      "post_submit_pre_sense_share":b["post_submit_pre_sense"]/denom,
      "pre_submit_share":b["pre_submit"]/denom,
      "too_late_share":b["too_late"]/denom,
      "presense_device_pruned":p["device_pruned"],
      "baseline_sense":b["sense"],
      "presense_sense":p["sense"],
      "presense_sense_reduction":(b["sense"]-p["sense"])/max(1,b["sense"]),
      "feedback4us_device_pruned":p4["device_pruned"]
    })
    return res,trs

methods=[
 ("PQ48B","pq",48),
 ("PQ96B","pq",96),
 ("PQ192B","pq",192),
 ("PQ384B","pq",384),
 ("OPQ96B","opq",96),
 ("RaBitQ1","rabitq",1),
 ("RaBitQ2","rabitq",2),
 ("RaBitQ4","rabitq",4),
]
result={"config":{"model":MODEL_NAME,"dim":768,"k":K,"R":R,"page_bytes":PAGE_BYTES,"banks":NBANK,"tR_us":TR_US,
                  "timing":"32 concurrent queries, W=all, no host pruning/cancel; ideal feedback=0 plus 4us sensitivity"},
        "faiss_version":getattr(faiss,"__version__",None),"datasets":{}}

for dsname,cap in [("scifact",6000),("fiqa",6000)]:
    ds=load_beir(dsname,64,cap)
    ep=ROOT/"embeddings"/f"{dsname}_X.npy";qp=ROOT/"embeddings"/f"{dsname}_Q.npy"
    if ep.exists() and qp.exists():
        X=np.load(ep);Q=np.load(qp)
    else:
        t=time.time();X=encode(ds["texts"]);Q=encode(ds["queries"]);np.save(ep,X);np.save(qp,Q)
        print(dsname,"embedding_seconds",round(time.time()-t,2),flush=True)
    _,GT=exact_topk(X,Q,K)
    exact_nd=float(np.mean([ndcg([ds["dids"][i] for i in GT[j]],ds["qids"][j],ds["qrels"]) for j in range(len(Q))]))
    dres={"corpus":len(X),"queries":len(Q),"dim":int(X.shape[1]),"exact_ndcg10":exact_nd,"methods":[]}
    for label,kind,param in methods:
        print(dsname,label,"START",flush=True)
        try:
            r,_=eval_method(X,Q,ds["dids"],ds["qids"],ds["qrels"],GT,label,kind,param)
            dres["methods"].append(r)
            print(json.dumps(r,ensure_ascii=False),flush=True)
        except Exception as e:
            err={"label":label,"kind":kind,"param":param,"error":repr(e),"traceback":traceback.format_exc()[-2000:]}
            dres["methods"].append(err);print(json.dumps(err,ensure_ascii=False),flush=True)
    result["datasets"][dsname]=dres

(OUT/"sweetspot_768_results.json").write_text(json.dumps(result,indent=2)+"\n")

# Compact result for receipt.
compact={"config":result["config"],"datasets":{}}
for ds,d in result["datasets"].items():
    rows=[]
    for r in d["methods"]:
        if "error" in r:
            rows.append({"label":r["label"],"error":r["error"]});continue
        rows.append({
          "label":r["label"],"codeB":r.get("code_size_bytes"),
          "cR10":round(r["compressed_R10"],3),"candR10":round(r["candidate_R10"],3),"rrR10":round(r["rerank_R10"],3),
          "cNDCG":round(r["compressed_ndcg10"],3),"rrNDCG":round(r["rerank_ndcg10"],3),
          "op":round(r.get("oracle_page_prune",0),3),"oc":round(r.get("oracle_candidate_prune",0),3),
          "post":round(r.get("post_submit_pre_sense_share",0),3),
          "senseRed":round(r.get("presense_sense_reduction",0),3),
          "devPrune":r.get("presense_device_pruned",0),
          "fb4Dev":r.get("feedback4us_device_pruned",0),
          "errMean":round(r.get("mean_reconstruction_error",0),4)
        })
    compact["datasets"][ds]={"n":d["corpus"],"q":d["queries"],"exactNDCG":round(d["exact_ndcg10"],3),"rows":rows}
job=Path(os.environ["SERVER_JOB_DIR"])
(job/"result_summary.json").write_text(json.dumps(compact,separators=(",",":"))+"\n")
print("FINAL_COMPACT",json.dumps(compact,ensure_ascii=False),flush=True)
PY
