"""Parallel stochastic full-network simulations, labelled and reproducible."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from circuit import Circuit
from model import Library,Noise,evolve_layer
from mps import MPS

HERE=Path(__file__).resolve().parent
STATES={"0":np.array([1.,0.]),"1":np.array([0.,1.]),"+":np.array([1.,1.])/np.sqrt(2),"-":np.array([1.,-1.])/np.sqrt(2)}
PZERO=np.array([.99,.03,.12,.08])


@lru_cache(maxsize=1)
def library():return Library()


@lru_cache(maxsize=16)
def circuit(n,scheme):return Circuit(library(),n,scheme)


def input_bank(n,mode):
    if mode=="phase":return ["phase"],[np.ones(1)],True
    labels=["00","01","10","11","0+","1+","+0","+1","++","Bell","phase"]
    values=[]
    for label in labels:
        if label=="Bell":values.append(np.kron(np.array([1,0,0,1])/np.sqrt(2),STATES["0"]))
        elif label=="phase":values.append(np.kron(np.ones(4)/2,STATES["-"]))
        else:values.append(np.kron(np.kron(STATES[label[0]],STATES[label[1]]),STATES["0"]))
    return labels,values,False


def target_vector(n,m,label,grover=False):
    N=2**n
    if label=="phase":
        vector=np.kron(np.ones(N)/np.sqrt(N)*(-1.)**np.asarray(m),STATES["-"])
    elif label=="Bell":
        vector=np.zeros(2*N,complex);vector[m[0]]=1/np.sqrt(2);vector[2*(N-1)+m[-1]]=1/np.sqrt(2)
    else:
        address=np.kron(STATES[label[0]],STATES[label[1]])
        vector=np.zeros(2*N,complex)
        for a in range(N):vector[2*a+m[a]]=address[a]
    if grover:
        s=np.ones(N)/np.sqrt(N);vector=np.kron(2*np.outer(s,s)-np.eye(N),np.eye(2))@vector
    return vector


@lru_cache(maxsize=8)
def comp_indices(k):return np.array([np.ravel_multi_index(tuple(map(int,f"{i:0{k}b}")),(4,)*k) for i in range(2**k)])


def pack(outputs,k):
    ids=comp_indices(k);aids=comp_indices(k-1)
    blocks=[];addresses=[];diagonal=[];norm=[]
    for triple in outputs:
        bb=[];aa=[];dd=[];nn=[]
        for rho in triple:
            shape=(4**(k-1),4,4**(k-1),4)
            reduced=np.trace(rho.reshape(shape),axis1=1,axis2=3)
            bb.append(rho[np.ix_(ids,ids)]);aa.append(reduced[np.ix_(aids,aids)])
            dd.append(np.diag(reduced).real);nn.append(np.trace(rho).real)
        blocks.append(bb);addresses.append(aa);diagonal.append(dd);norm.append(nn)
    return {"block":np.array(blocks),"address":np.array(addresses),"adiag":np.array(diagonal),"norm":np.array(norm)}


def measurement_weights(n,readout=True):
    words=np.array(list(np.ndindex(*(4,)*n)))
    result=[]
    for address in range(2**n):
        value=np.ones(len(words))
        for q,bit in enumerate(f"{address:0{n}b}"):
            pzero=PZERO if readout else np.array([1.,0.,0.,0.])
            value*=pzero[words[:,q]] if bit=="0" else 1-pzero[words[:,q]]
        result.append(value)
    return np.array(result)


def metric_row(result,n,m,label,grover,memory,readout=True):
    goal=target_vector(n,m,label,grover)
    fs=np.array([np.vdot(goal,rho@goal).real for rho in result["block"]])
    target=int(np.argmax(m)) if sum(m)==1 else 0
    response=measurement_weights(n,readout)[target]
    observed_success=result["adiag"]@response
    match=float(np.array_equal(memory,m))
    measured=np.prod([(.99 if b==0 else .97) if b==want else (.01 if b==0 else .03) for b,want in zip(memory,m)]) if readout else match
    # Components remain unnormalized until the ensemble importance ratio is formed.
    return np.r_[result["norm"],fs,observed_success,match*result["norm"][0],measured*result["norm"][0]]


def worker(job):
    threadpool_limits(limits=1)
    c=circuit(job["n"],job["scheme"]);lib=library();noise=Noise(**job["noise"])
    labels,vectors,direct=input_bank(job["n"],job["mode"])
    n=job["n"];word=np.array(list(map(int,job["memory"])))
    accumulated=None;records=[];discard=[];peaks=[]
    for repeat in job["replicas"]:
        salt=int(hashlib.sha256((job["id"]+str(repeat)).encode()).hexdigest()[:8],16)
        rng=np.random.default_rng([20261102,salt])
        if direct:
            state=MPS.product(c.quantum,c.inputs,[STATES["+"]]*n+[STATES["-"]],max_bond=job["bond"],tolerance=1e-10)
        else:state=MPS.choi(c.quantum,c.inputs,max_bond=job["bond"],tolerance=1e-10)
        memory=word.copy()
        # Fixed preparation error surrogate; it never changes the query body.
        if noise.single:
            memory[rng.random(len(memory))<.002]^=1
        phases={};numbers={};snapshots={}
        for index,layer in enumerate(c.stages):
            memory=evolve_layer(state,memory,layer,lib,noise,rng,c.quantum)
            if index+1==c.query_stages:
                copy=state.copy();snapshot=pack(copy.reduce_inputs(c.inputs,vectors,PZERO if noise.readout else np.array([1.,0.,0.,0.])),n+1)
                snapshots["query"]=snapshot
                numbers["query"]=np.array([metric_row({key:value[k] for key,value in snapshot.items()},n,word,label,False,memory,noise.readout) for k,label in enumerate(labels)])
            if index+1==len(c.stages) and n==2:
                vector=[np.ones(1)] if direct else [vectors[-1]]
                copy=state.copy();snapshot=pack(copy.reduce_inputs(c.inputs,vector,PZERO if noise.readout else np.array([1.,0.,0.,0.])),n+1)
                snapshots["grover"]=snapshot
                numbers["grover"]=np.array([metric_row({key:value[0] for key,value in snapshot.items()},n,word,"phase",True,memory,noise.readout)])
        if accumulated is None:accumulated={stage:{key:np.zeros_like(value) for key,value in snapshot.items()} for stage,snapshot in snapshots.items()}
        for stage,snapshot in snapshots.items():
            for key,value in snapshot.items():accumulated[stage][key]+=value
        records.append(numbers);discard.append(state.discarded);peaks.append(state.peak_bond)
    return {"id":job["id"],"sum":accumulated,"records":records,"discard":discard,"peak":peaks,
            "labels":labels,"replicas":len(job["replicas"])}


def cases(primary=48,search=128,probe=20):
    base=Noise();result=[]
    def add(name,memory,noise=base,replicas=primary,mode="bank",n=2,scheme="r5",bond=32):
        result.append({"id":name,"memory":memory,"noise":asdict(noise),"count":replicas,"mode":mode,"n":n,"scheme":scheme,"bond":bond})
    zero=replace(base,single=0,x12=0,local_cz=0,cross_cz=0,memory_cz=0,loss=False,dephasing=False,readout=False)
    for word in itertools.product("01",repeat=4):
        text="".join(word);add("primary_"+text,text)
        add("coherent_"+text,text,zero,1)
    for k in range(4):
        memory="".join("1" if j==k else "0" for j in range(4))
        add("search_"+memory,memory,replicas=search,mode="phase")
        add("improved_"+memory,memory,replace(base,t1_ns=100000.,t2_ns=100000.,cross_cz=.002,local_cz=.001,memory_cz=.001),search//2,"phase")
        for tag,noise in (
            ("no_single",replace(base,single=0,x12=0)),("no_local_cz",replace(base,local_cz=0,memory_cz=0)),
            ("no_cross",replace(base,cross_cz=0)),("no_loss",replace(base,loss=False)),
            ("no_phi",replace(base,dephasing=False)),("no_aux_loss",replace(base,aux_decay=False)),
            ("no_memory_loss",replace(base,memory_decay=False)),("no_readout",replace(base,readout=False))):
            add(tag+"_"+memory,memory,noise,probe,"phase")
        for error in (0.,.002,.004,.008,.016,.03):
            add(f"cross_{error}_{memory}",memory,replace(base,cross_cz=error),probe,"phase")
    for n in (1,3):
        for pattern in ("1"*(2**n),"1"+"0"*(2**n-1)):
            add(f"scale_{n}_{pattern}",pattern,base,max(12,probe),"phase",n=n,bond=32)
    add("r6_reference","0010",zero,1,"bank",scheme="r6")
    return result


def aggregate(case,value,output):
    arrays={};summary={"config":case,"replicas":value["replicas"],"labels":value["labels"],
        "discarded_sum_max":float(max(value["discard"],default=0)),"discarded_sum_mean":float(np.mean(value["discard"])),
        "peak_bond":int(max(value["peak"],default=1)),"stages":{}}
    records=value["records"]
    for stage,fields in value["sum"].items():
        denom=fields["norm"][:,0]
        for key,array in fields.items():
            shape=(len(denom),)+(1,)*(array.ndim-1)
            arrays[stage+"_"+key]=array/denom.reshape(shape)
        raw=np.array([record[stage] for record in records])
        arrays[stage+"_trajectory_metrics"]=raw
        rows=[]
        for k in range(raw.shape[1]):
            total=raw[:,k].sum(axis=0);d=total[0]
            row={"label":value["labels"][k] if stage=="query" else "phase","F_raw":float(total[3]/d),
                 "search_target":int(np.argmax(list(map(int,case["memory"])))) if case["memory"].count("1")==1 else None,
                 "F_true_cond":float(total[4]/total[1]) if total[1]>0 else None,
                 "F_obs_cond":float(total[5]/total[2]) if total[2]>0 else None,
                 "p_true":float(total[1]/d),"p_obs":float(total[2]/d),
                 "success_raw_binary":float(total[6]/d),"success_obs_cond_binary":float(total[8]/total[2]) if total[2]>0 else None,
                 "success_obs_joint_binary":float(total[8]/d),"memory_true":float(total[9]/d),"memory_observed":float(total[10]/d),
                 "weight_mean":float(np.mean(raw[:,k,0])),"ESS":float(total[0]**2/np.sum(raw[:,k,0]**2))}
            rng=np.random.default_rng(20261102+k)
            boot=[]
            for _ in range(300):
                chosen=rng.integers(0,len(raw),len(raw));v=raw[chosen,k].sum(axis=0)
                boot.append([v[3]/v[0],v[6]/v[0],v[8]/v[2] if v[2]>0 else np.nan,v[2]/v[0],v[8]/v[0]])
            boot=np.array(boot)
            row["interval95"]={key:np.nanquantile(boot[:,j],[.025,.975]).tolist() for j,key in enumerate(("F_raw","success_raw_binary","success_obs_cond_binary","p_obs","success_obs_joint_binary"))}
            row["SE"]={key:float(np.nanstd(boot[:,j],ddof=1)) for j,key in enumerate(("F_raw","success_raw_binary","success_obs_cond_binary","p_obs","success_obs_joint_binary"))}
            rows.append(row)
        summary["stages"][stage]=rows
    np.savez_compressed(output/"arrays"/(case["id"]+".npz"),**arrays)
    (output/"cases"/(case["id"]+".json")).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    return summary


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",default="results_v1");parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--primary",type=int,default=48);parser.add_argument("--search",type=int,default=128)
    parser.add_argument("--probe",type=int,default=20);args=parser.parse_args()
    output=HERE/args.output;output.mkdir(exist_ok=True)
    for name in ("arrays","cases","figures"): (output/name).mkdir(exist_ok=True)
    selected=cases(args.primary,args.search,args.probe)
    manifest={"model":"17 logical registers; quantum four-level MPS + classical basis memory trajectories",
              "args":vars(args),"cases":selected,"source_sha256":{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("mps.py","model.py","circuit.py","runner.py")}}
    existing=output/"manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf8"))!=manifest:
        raise ValueError("Existing run manifest differs. Use a new output directory.")
    existing.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    with threadpool_limits(limits=1):
        lib=library();resources={}
        for n in (1,2,3):
            for scheme in ("r5","r6"):
                c=circuit(n,scheme);resources[f"{n}_{scheme}"]=c.resource()
                (output/f"schedule_{n}_{scheme}.json").write_text(json.dumps(c.serializable(),indent=2),encoding="utf8")
        (output/"resources.json").write_text(json.dumps(resources,indent=2),encoding="utf8")
        (output/"pulse_sources.json").write_text(json.dumps(lib.sources,indent=2),encoding="utf8")
    jobs=[];accumulators={};completed={}
    for case in selected:
        path=output/"cases"/(case["id"]+".json")
        if path.exists():completed[case["id"]]=json.loads(path.read_text(encoding="utf8"));continue
        for start in range(0,case["count"],8):jobs.append({**case,"replicas":list(range(start,min(start+8,case["count"])))})
    start=time.perf_counter();done=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,job):job for job in jobs}
        for future in as_completed(futures):
            value=future.result();identifier=value["id"]
            if identifier not in accumulators:accumulators[identifier]=value
            else:
                acc=accumulators[identifier]
                for stage,fields in value["sum"].items():
                    for key,array in fields.items():acc["sum"][stage][key]+=array
                for key in ("records","discard","peak"):acc[key]+=value[key]
                acc["replicas"]+=value["replicas"]
            case=next(c for c in selected if c["id"]==identifier)
            if accumulators[identifier]["replicas"]==case["count"]:
                completed[identifier]=aggregate(case,accumulators.pop(identifier),output)
            done+=1
            if done%4==0 or done==len(futures):
                status={"completed_batches":done,"total_batches":len(futures),"complete_cases":len(completed),
                        "total_cases":len(selected),"elapsed_s":time.perf_counter()-start}
                (output/"progress.json").write_text(json.dumps(status,indent=2),encoding="utf8")
                print("PROGRESS",status,flush=True)
    (output/"SUMMARY.json").write_text(json.dumps({"cases":completed,"resources":resources,"source":manifest["model"]},ensure_ascii=False,indent=2),encoding="utf8")
    print("DONE",output,flush=True)


if __name__=="__main__":main()
