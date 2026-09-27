#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/HBF_RAG_Sim/g1_quick_real
DEPS="$ROOT/.deps"
CACHE=/workspace/HBF_RAG_Sim/hf_cache
mkdir -p "$ROOT/results"
export PYTHONPATH="$DEPS:${PYTHONPATH:-}"
export HF_HOME="$CACHE"
export TRANSFORMERS_CACHE="$CACHE"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

/opt/conda/bin/python - <<'PY'
import os, json, csv, math, time, heapq
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

SEED=42
K=10
R=256
PAGE_BYTES=4096
NBANK=256
TR_US=4.0
TCOMP_US=0.20
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ROOT=Path("/workspace/HBF_RAG_Sim/g1_quick_real")
DATA=ROOT/"data"
OUT=ROOT/"results"
OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(SEED)

def read_jsonl(path):
    out=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def load_beir(name, max_queries=64, corpus_cap=6000):
    base=DATA/name
    corpus_rows=read_jsonl(base/"corpus.jsonl")
    query_rows=read_jsonl(base/"queries.jsonl")
    qrels={}
    qpath=base/"qrels"/"test.tsv"
    with open(qpath,encoding="utf-8") as f:
        rd=csv.DictReader(f,delimiter="\t")
        for row in rd:
            qid=str(row["query-id"]); did=str(row["corpus-id"]); score=int(row["score"])
            if score>0: qrels.setdefault(qid,{})[did]=score
    qmap={str(x["_id"]):x["text"] for x in query_rows}
    valid=[qid for qid in qrels if qid in qmap]
    rr=np.random.default_rng(SEED + (0 if name=="scifact" else 100))
    rr.shuffle(valid)
    qids=valid[:min(max_queries,len(valid))]
    must=set()
    for qid in qids: must.update(qrels[qid])
    cmap={str(x["_id"]):x for x in corpus_rows}
    must=[d for d in must if d in cmap]
    allids=list(cmap)
    remain=[d for d in allids if d not in set(must)]
    target=min(corpus_cap,len(allids))
    take=max(0,target-len(must))
    sampled=rr.choice(remain,size=min(take,len(remain)),replace=False).tolist() if take else []
    dids=list(dict.fromkeys(must+sampled))
    texts=[]
    for did in dids:
        x=cmap[did]; title=(x.get("title") or "").strip(); text=(x.get("text") or "").strip()
        texts.append((title+"\n"+text).strip())
    queries=[qmap[q] for q in qids]
    return {"name":name,"dids":dids,"texts":texts,"qids":qids,"queries":queries,
            "qrels":{q:{d:s for d,s in qrels[q].items() if d in set(dids)} for q in qids}}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE",device,"CUDA_VISIBLE_DEVICES",os.environ.get("CUDA_VISIBLE_DEVICES"),flush=True)
tok=AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=str(Path(os.environ["HF_HOME"])))
model=AutoModel.from_pretrained(MODEL_NAME,cache_dir=str(Path(os.environ["HF_HOME"]))).to(device).eval()

@torch.no_grad()
def encode(texts,batch=128,max_length=128):
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
    idx=faiss.IndexFlatL2(X.shape[1]); idx.add(X)
    D,I=idx.search(Q,k)
    return D,I

def ndcg_at_10(rank_ids,qid,qrels):
    rel=qrels.get(qid,{})
    dcg=0.0
    for j,did in enumerate(rank_ids[:10]):
        r=rel.get(did,0)
        if r>0: dcg += (2**r-1)/math.log2(j+2)
    ideal=sorted(rel.values(),reverse=True)[:10]
    idcg=sum((2**r-1)/math.log2(j+2) for j,r in enumerate(ideal))
    return dcg/idcg if idcg>0 else 0.0

def make_index(X,M,opq=False,nbits=8):
    d=X.shape[1]
    pq=faiss.IndexPQ(d,M,nbits,faiss.METRIC_L2)
    pq.pq.cp.niter=12; pq.pq.cp.nredo=1
    if opq:
        rot=faiss.OPQMatrix(d,M)
        rot.niter=8
        index=faiss.IndexPreTransform(rot,pq)
    else:
        index=pq
    t=time.time(); index.train(X); train_s=time.time()-t
    index.add(X)
    return index,train_s

def reconstruct_all(index,n,d):
    try:
        x=index.reconstruct_n(0,n)
        if x is not None:
            return np.asarray(x,dtype="float32").reshape(n,d)
    except Exception:
        pass
    out=np.empty((n,d),dtype="float32")
    for i in range(n): out[i]=index.reconstruct(i)
    return out

def eval_quant(X,Q,dids,qids,qrels,exactI,label,M,opq=False):
    index,train_s=make_index(X,M,opq=opq)
    t=time.time(); AD,AI=index.search(Q,max(R,K)); search_s=time.time()-t
    approx10=AI[:,:K]
    comp_recall=[]; cand_recall=[]; rerank_recall=[]; nd_comp=[]; nd_rer=[]; rerank_ids=[]
    for qi in range(len(Q)):
        gt=set(exactI[qi,:K].tolist())
        comp_recall.append(len(gt & set(approx10[qi].tolist()))/K)
        c0=AI[qi,:R]
        cand_recall.append(len(gt & set(c0.tolist()))/K)
        ed=np.linalg.norm(X[c0]-Q[qi],axis=1)
        rr=c0[np.argsort(ed)[:K]]
        rerank_ids.append(rr)
        rerank_recall.append(len(gt & set(rr.tolist()))/K)
        nd_comp.append(ndcg_at_10([dids[i] for i in approx10[qi]],qids[qi],qrels))
        nd_rer.append(ndcg_at_10([dids[i] for i in rr],qids[qi],qrels))
    Xh=reconstruct_all(index,len(X),X.shape[1])
    return {
        "label":label,"M":M,"opq":opq,"code_bytes_per_vector":M,
        "train_seconds":train_s,"search_ms_per_query":1000*search_s/max(1,len(Q)),
        "compressed_exact_R10":float(np.mean(comp_recall)),
        "candidate_exact_R10":float(np.mean(cand_recall)),
        "rerank_exact_R10":float(np.mean(rerank_recall)),
        "compressed_ndcg10":float(np.mean(nd_comp)),
        "rerank_ndcg10":float(np.mean(nd_rer)),
    }, AI[:,:R].copy(), Xh

def build_trace(X,Xh,Q,candidates,seed,feedback_us=0.0):
    n,d=X.shape
    vpp=max(1,PAGE_BYTES//(d*2))
    rr=np.random.default_rng(seed)
    perm=rr.permutation(n); pos=np.empty(n,dtype=np.int64); pos[perm]=np.arange(n)
    page_of=pos//vpp
    traces=[]; oracle_c=[]; oracle_p=[]
    for qi,q in enumerate(Q):
        c0=np.asarray(candidates[qi],dtype=np.int64)
        exact=np.linalg.norm(X[c0]-q,axis=1)
        approx=np.linalg.norm(Xh[c0]-q,axis=1)
        err=np.linalg.norm(X[c0]-Xh[c0],axis=1)
        L=np.maximum(0.0,approx-err); U=approx+err
        theta_final=float(np.partition(exact,K-1)[K-1])
        cpr=L>theta_final
        pages=defaultdict(list)
        for li,vid in enumerate(c0): pages[int(page_of[vid])].append(li)
        pmeta={}
        pp=[]
        for p,inds0 in pages.items():
            inds=np.asarray(inds0,dtype=np.int64)
            pmeta[p]={
                "cand_idx":inds,
                "L":float(np.min(L[inds])),
                "priority":float(np.min(approx[inds])),
                "bank":int(p%NBANK),
            }
            pp.append(bool(np.all(cpr[inds])))
        oracle_c.append(float(np.mean(cpr)))
        oracle_p.append(float(np.mean(pp)))
        traces.append({"exact":exact,"L":L,"U":U,"pages":pmeta,"theta_final":theta_final})
    return traces,{"vectors_per_page":vpp,"oracle_candidate_prune":float(np.mean(oracle_c)),"oracle_page_prune":float(np.mean(oracle_p))}

def theta_now(st):
    vals=list(st["rex"])
    active=~(st["removed"]|st["resolved"])
    vals.extend(st["tr"]["U"][active].tolist())
    if len(vals)<K:return float("inf")
    a=np.asarray(vals,float)
    return float(np.partition(a,K-1)[K-1])

# Baseline event simulation that records when each page first becomes prunable.
# host_prune=False is the requested pre-host-scheduling/cancel diagnostic.
def simulate(traces,qids,W,feedback_us=0.0,device_presense=False,host_prune=False):
    qs={}
    for q in qids:
        tr=traces[q]; order=sorted(tr["pages"],key=lambda p:tr["pages"][p]["priority"])
        m=len(tr["exact"])
        qs[q]={
            "tr":tr,"pending":deque(order),"inflight":0,"removed":np.zeros(m,bool),"resolved":np.zeros(m,bool),
            "rex":[],"theta":float(np.partition(tr["U"],K-1)[K-1]),"done":None,
            "submit_t":{},"sense_t":{},"prunable_t":{},"device_pruned":0,"host_pruned":0,"sense":0
        }
    bankq=[deque() for _ in range(NBANK)]; busy=[False]*NBANK; ev=[]; seq=0

    def push(t,pri,typ,*data):
        nonlocal seq
        seq+=1; heapq.heappush(ev,(t,pri,seq,typ,data))
    def update_prunable(q,t):
        st=qs[q]
        for p,pg in st["tr"]["pages"].items():
            if p not in st["prunable_t"] and pg["L"]>st["theta"]:
                st["prunable_t"][p]=float(t)
    def recalc(q,t):
        st=qs[q]; old=st["theta"]; st["theta"]=theta_now(st)
        if st["theta"] < old - 1e-12: update_prunable(q,t)
    def remove(q,p,t):
        st=qs[q]; inds=st["tr"]["pages"][p]["cand_idx"]; st["removed"][inds]=True; recalc(q,t)
    def resolve(q,p,t):
        st=qs[q]; inds=st["tr"]["pages"][p]["cand_idx"]; inds=inds[~st["removed"][inds]]
        if len(inds):
            st["resolved"][inds]=True; st["rex"].extend(st["tr"]["exact"][inds].tolist())
        recalc(q,t)
    def maybe_done(q,t):
        st=qs[q]
        if st["done"] is None and not st["pending"] and st["inflight"]==0: st["done"]=float(t)
    def submit(q,p,t):
        st=qs[q]; b=st["tr"]["pages"][p]["bank"]
        st["submit_t"][p]=float(t); st["inflight"]+=1; bankq[b].append((q,p))
        if not busy[b] and len(bankq[b])==1: push(t,2,"sense",b)
    def refill(q,t):
        st=qs[q]
        lim=len(st["tr"]["pages"]) if W=="all" else int(W)
        while st["inflight"]<lim and st["pending"]:
            p=st["pending"].popleft(); pg=st["tr"]["pages"][p]
            if host_prune and pg["L"]>st["theta"]:
                st["host_pruned"]+=1; remove(q,p,t); continue
            submit(q,p,t)
        maybe_done(q,t)

    for q in qids:
        update_prunable(q,0.0)
        refill(q,0.0)
    while ev:
        t,_,_,typ,data=heapq.heappop(ev)
        if typ=="sense":
            b=data[0]
            if busy[b] or not bankq[b]: continue
            q,p=bankq[b][0]; st=qs[q]; pg=st["tr"]["pages"][p]
            if device_presense and pg["L"]>st["theta"]:
                bankq[b].popleft(); st["device_pruned"]+=1; st["inflight"]-=1
                remove(q,p,t); refill(q,t)
                if bankq[b] and not busy[b]: push(t,2,"sense",b)
                continue
            bankq[b].popleft(); busy[b]=True; st["sense_t"][p]=float(t); st["sense"]+=1
            push(t+TR_US,1,"free",b); push(t+TR_US+TCOMP_US+feedback_us,0,"feedback",q,p)
        elif typ=="free":
            b=data[0]; busy[b]=False
            if bankq[b]: push(t,2,"sense",b)
        elif typ=="feedback":
            q,p=data; st=qs[q]; resolve(q,p,t); st["inflight"]-=1; refill(q,t)

    cats={"pre_submit":0,"post_submit_pre_sense":0,"too_late":0,"never_online":0}
    oracle=0
    for q in qids:
        st=qs[q]
        for p,pg in st["tr"]["pages"].items():
            if pg["L"]>st["tr"]["theta_final"]:
                oracle+=1
                tp=st["prunable_t"].get(p,None); tsu=st["submit_t"].get(p,float("inf")); tse=st["sense_t"].get(p,float("inf"))
                if tp is None: cats["never_online"]+=1
                elif tp < tsu-1e-12: cats["pre_submit"]+=1
                elif tp < tse-1e-12: cats["post_submit_pre_sense"]+=1
                else: cats["too_late"]+=1
    denom=max(1,oracle)
    lats=[qs[q]["done"] for q in qids]
    return {
        "W":W,"concurrency":len(qids),"feedback_us":feedback_us,"device_presense":device_presense,"host_prune":host_prune,
        "oracle_prunable_pages":oracle,
        **{k:v for k,v in cats.items()},
        **{k+"_share":v/denom for k,v in cats.items()},
        "sense":int(sum(qs[q]["sense"] for q in qids)),
        "device_pruned":int(sum(qs[q]["device_pruned"] for q in qids)),
        "host_pruned":int(sum(qs[q]["host_pruned"] for q in qids)),
        "mean_latency_us":float(np.mean(lats)),
        "p95_latency_us":float(np.percentile(lats,95)),
    }

all_results={"config":{
    "seed":SEED,"model":MODEL_NAME,"k":K,"candidate_R":R,"page_bytes":PAGE_BYTES,"banks":NBANK,
    "tR_us":TR_US,"compute_us":TCOMP_US,
    "note":"Quick diagnostic: sampled real BEIR text, exact embedding top-k as vector-search ground truth; fixed random vector-to-page layout; no layout optimization."
},"datasets":{}}

for dsname,cap in [("scifact",6000),("fiqa",6000)]:
    tds=time.time(); ds=load_beir(dsname,max_queries=64,corpus_cap=cap)
    print(f"\n[{dsname}] corpus={len(ds['dids'])} queries={len(ds['qids'])}",flush=True)
    emb_dir=ROOT/"embeddings"; emb_dir.mkdir(exist_ok=True)
    xp=emb_dir/f"{dsname}_X.npy"; qp=emb_dir/f"{dsname}_Q.npy"
    if xp.exists() and qp.exists():
        X=np.load(xp); Q=np.load(qp)
    else:
        t=time.time(); X=encode(ds["texts"]); Q=encode(ds["queries"])
        np.save(xp,X); np.save(qp,Q)
        print("embedding_s",round(time.time()-t,2),flush=True)
    Dgt,Igt=exact_topk(X,Q,K)
    gt_nd=np.mean([ndcg_at_10([ds["dids"][i] for i in Igt[j]],ds["qids"][j],ds["qrels"]) for j in range(len(Q))])
    qspecs=[("PQ12B",12,False),("PQ24B",24,False),("PQ48B",48,False),("OPQ24B",24,True)]
    qres=[]; traces_by_label={}
    raw_bytes=len(X)*X.shape[1]*2
    for label,M,opq in qspecs:
        print("train/eval",label,flush=True)
        r,cand,Xh=eval_quant(X,Q,ds["dids"],ds["qids"],ds["qrels"],Igt,label,M,opq)
        r["raw_fp16_MB"]=raw_bytes/1e6
        r["compressed_MB"]=len(X)*M/1e6
        traces,opp=build_trace(X,Xh,Q,cand,seed=SEED+(hash(dsname+label)&0xffff))
        r.update(opp); qres.append(r); traces_by_label[label]=traces
        print(json.dumps(r,ensure_ascii=False),flush=True)

    # Use PQ48B as the representative tighter-bound trace for timing opportunity.
    traces=traces_by_label["PQ48B"]
    timing=[]
    for conc in (1,8,32):
        qids=list(range(min(conc,len(traces))))
        for W in (32,128,"all"):
            # Primary diagnostic: ideal feedback=0, no host pruning/cancel.
            base=simulate(traces,qids,W,feedback_us=0.0,device_presense=False,host_prune=False)
            pre=simulate(traces,qids,W,feedback_us=0.0,device_presense=True,host_prune=False)
            # Secondary: basic host safety pruning only; still no optimized scheduler/cancel.
            hp=simulate(traces,qids,W,feedback_us=0.0,device_presense=True,host_prune=True)
            # 4us propagation sensitivity for raw pre-sense.
            pre4=simulate(traces,qids,W,feedback_us=4.0,device_presense=True,host_prune=False)
            timing.append({"baseline":base,"presense":pre,"basic_host_plus_presense":hp,"presense_feedback4us":pre4})
    all_results["datasets"][dsname]={
        "corpus":len(X),"queries":len(Q),"dim":int(X.shape[1]),"exact_embedding_ndcg10":float(gt_nd),
        "quantizers":qres,"timing":timing,"runtime_seconds":time.time()-tds
    }

(OUT/"g1_quick_real_results.json").write_text(json.dumps(all_results,indent=2),encoding="utf-8")

lines=["# G1-quick + pre-sense opportunity (real BEIR subset)","",
       "Primary timing diagnostic uses feedback latency = 0 us, fixed random layout, no optimized host scheduling and no host cancel.",
       "A 4 us feedback sensitivity is also reported. PQ48B is used for timing because its bounds are tighter than shorter codes.",""]
for dsname,d in all_results["datasets"].items():
    lines += [f"## {dsname}",f"- corpus={d['corpus']}, queries={d['queries']}, dim={d['dim']}, exact-embedding nDCG@10={d['exact_embedding_ndcg10']:.3f}","",
              "| quantizer | comp exact R@10 | candidate R@10 | rerank R@10 | comp nDCG@10 | rerank nDCG@10 | oracle cand prune | oracle page prune |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["quantizers"]:
        lines.append(f"| {r['label']} | {r['compressed_exact_R10']:.3f} | {r['candidate_exact_R10']:.3f} | {r['rerank_exact_R10']:.3f} | {r['compressed_ndcg10']:.3f} | {r['rerank_ndcg10']:.3f} | {r['oracle_candidate_prune']:.3f} | {r['oracle_page_prune']:.3f} |")
    lines += ["","### Timing opportunity on PQ48B (feedback=0us, no host scheduling/cancel)",
              "| conc | W | post-submit/pre-sense share of oracle-prunable | realized pre-sense prune / oracle | sense reduction | basic-host+pre-sense device-pruned | 4us device-pruned |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for z in d["timing"]:
        b=z["baseline"]; p=z["presense"]; hp=z["basic_host_plus_presense"]; p4=z["presense_feedback4us"]
        sr=(b["sense"]-p["sense"])/max(1,b["sense"])
        realized=p["device_pruned"]/max(1,b["oracle_prunable_pages"])
        lines.append(f"| {b['concurrency']} | {b['W']} | {b['post_submit_pre_sense_share']:.3f} | {realized:.3f} | {sr:.3f} | {hp['device_pruned']} | {p4['device_pruned']} |")
    lines.append("")
lines += ["## Interpretation guardrails",
          "- This is a quick feasibility run, not a publication result.",
          "- The corpus is a relevance-preserving sampled subset (all positives for selected queries plus sampled distractors).",
          "- Exact embedding top-k is used as the vector-search ground truth for G1 compression/rerank necessity.",
          "- Pre-sense timing is a parameterized event simulation; feedback=0us is an ideal algorithmic upper-bound setting, not a hardware claim.",
          "- No optimized host scheduler or host-cancel path is included in the primary pre-sense result."]
(OUT/"g1_quick_real_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n".join(lines),flush=True)
PY

echo
echo "== outputs =="
find "$ROOT/results" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
