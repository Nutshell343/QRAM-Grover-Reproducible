"""Explicit-memory tree, one inter-chip gateway, and a fixed query body."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import numpy as np


def parallel(groups):
    result=[]
    for k in range(max(map(len,groups),default=0)):
        selected=[group[k] for group in groups if k<len(group)]
        ops=[op for layer in selected for op in layer["ops"]]
        kinds={op["name"]=="cz" or op["name"]=="memory_cz" for op in ops if op["name"]!="frame"}
        if len(kinds)>1:raise AssertionError("Mixed physical single/CZ layer.")
        used=[wire for op in ops if op["name"]!="frame" for wire in op["wires"]]
        if len(used)!=len(set(used)):raise AssertionError("Overlapping physical gates.")
        result.append({"duration":max(layer["duration"] for layer in selected),"ops":ops,
                       "kind":"cz" if True in kinds else "single"})
    return result


class Circuit:
    def __init__(self,library,n=2,scheme="r5"):
        self.lib=library;self.n=n;self.N=2**n;self.scheme=scheme
        self.inputs=[f"A{k}" for k in range(n)]+["D"]
        self.controls=[f"C{k}" for k in range(self.N-1)]
        self.data=[f"T{k}" for k in range(self.N-1)]+[f"L{k:0{n}b}" for k in range(self.N)]
        self.quantum=self.inputs+self.controls+self.data
        self.stages=[];self.layer_count=0
        self.add("prepare",[{"duration":32.,"kind":"single","ops":[]}])
        actions=[]
        for j in range(n):
            action=("gateway",j);actions.append(action);self.action("address_load",action,False)
            for depth in range(j):
                action=("route",depth);actions.append(action);self.action("address_load",action,False)
            action=("store",j);actions.append(action);self.action("address_load",action,False)
        self.add("data_encode",self.single(library.h,"D","H"))
        self.add("data_load",library.swap("D","T0",cross=True))
        for depth in range(n):self.add("data_down",self.route(depth,"down"))
        self.add("memory_access",[{"duration":50.,"kind":"cz","ops":[
            {"name":"memory_cz","wires":[f"L{k:0{n}b}"],"memory":k,"error":"memory"} for k in range(self.N)]}])
        for depth in reversed(range(n)):self.add("data_up",self.route(depth,"up"))
        self.add("data_return",library.swap("D","T0",cross=True))
        for action in reversed(actions):self.action("address_unload",action,True)
        self.add("data_decode",self.single(library.h,"D","H"))
        self.query_stages=len(self.stages)
        if n==2:
            self.add("diffusion",self.single_parallel(library.h,self.inputs[:-1],"H"))
            self.add("diffusion",self.single_parallel(library.x,self.inputs[:-1],"01_pi"))
            self.add("diffusion",[{"duration":50.,"kind":"cz","ops":[
                {"name":"cz","wires":["A0","A1"],"u":library.cz,"error":"local"}]}])
            self.add("diffusion",self.single_parallel(library.x,self.inputs[:-1],"01_pi"))
            self.add("diffusion",self.single_parallel(library.h,self.inputs[:-1],"H"))

    def single(self,u,wire,name):return [{"duration":32.,"kind":"single","ops":[{"name":name,"wires":[wire],"u":u,"error":"single"}]}]
    def single_parallel(self,u,wires,name):return parallel([self.single(u,q,name) for q in wires])

    def node(self,depth,index):
        k=2**depth-1+index
        left=2*k+1;right=left+1
        return [self.controls[k],self.data[k],self.data[left],self.data[right]]

    def route(self,depth,direction):
        return parallel([self.lib.mapped_router(self.scheme,direction,self.node(depth,k)) for k in range(2**depth)])

    def action(self,stage,action,inverse):
        name,j=action
        if name=="gateway":
            self.add(stage,self.lib.swap(f"A{j}","D"))
            self.add(stage,self.lib.swap("D","T0",cross=True))
            self.add(stage,self.lib.swap(f"A{j}","D"))
        elif name=="route":self.add(stage,self.route(j,"up" if inverse else "down"))
        else:
            self.add(stage,parallel([self.lib.swap(self.data[2**j-1+k],self.controls[2**j-1+k]) for k in range(2**j)]))

    def add(self,name,layers):
        for layer in layers:
            layer={**layer,"stage":name,"index":self.layer_count}
            self.stages.append(layer);self.layer_count+=1

    def resource(self):
        rows={};counts=Counter();clock=0.;router_cz=0
        for layer in self.stages:
            row=rows.setdefault(layer["stage"],{"duration_ns":0.,"layers":0,"CZ":0,"cross_CZ":0,"single":0})
            row["duration_ns"]+=layer["duration"];row["layers"]+=1
            for op in layer["ops"]:
                name=op["name"]
                if name=="frame":continue
                if name in ("cz","memory_cz"):
                    counts["CZ"]+=1;row["CZ"]+=1
                    if op["error"]=="cross":counts["cross_CZ"]+=1;row["cross_CZ"]+=1
                else:counts["single"]+=1;row["single"]+=1
            clock+=layer["duration"]
        query_ns=sum(x["duration_ns"] for k,x in rows.items() if k!="diffusion")
        return {"n":self.n,"N":self.N,"scheme":self.scheme,"quantum_mps_sites":len(self.quantum),
                "classical_memory_sites":self.N,"working_registers":len(self.quantum)+self.N,
                "router_calls":4*self.N-2*self.n-4,"router_CZ":(5 if self.scheme=="r5" else 6)*(4*self.N-2*self.n-4),
                "query_ns":query_ns,"total_ns":clock,"stages":rows,"counts":dict(counts)}

    def serializable(self):
        return [{key:([{("operator_sha256" if k=="u" else k):(hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() if k=="u" else v) for k,v in op.items()} for op in value] if key=="ops" else value)
                 for key,value in layer.items()} for layer in self.stages]
