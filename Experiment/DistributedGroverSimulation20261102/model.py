"""Frozen local pulse-derived router layers plus explicit reference link gates.

This is an effective-process full-network model, not a single 17-transmon
continuous Hamiltonian. Four-level local unitary paths are retained. T1 uses
resolved ladder transitions; the 01 total T2 is explicitly set to T1.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PULSE=ROOT/"Experiment/RouterPulseCalibration20261002"
sys.path[:0]=[str(PULSE)]
from pulse_model import PulseModel,target_rotation
from bootstrap import protocol as p,physical_layers


def read(path):return json.loads(Path(path).read_text(encoding="utf8"))


@dataclass(frozen=True)
class Noise:
    t1_ns:float=20000.
    t2_ns:float=20000.
    single:float=.0004
    x12:float=.0015
    local_cz:float=.003
    cross_cz:float=.008
    memory_cz:float=.003
    memory_decay:bool=True
    aux_decay:bool=True
    readout:bool=True
    dephasing:bool=True
    loss:bool=True


class Library:
    def __init__(self):
        self.model=PulseModel(read(PULSE/"device.json"),levels=4,dt=.1)
        self.model.load_calibration(PULSE/"results/native/native_parameters.json")
        self.sources={}
        for path in (PULSE/"device.json",PULSE/"results/native/native_parameters.json",PULSE/"pulse_model.py"):
            self.sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.blocks={};self.validation={}
        for scheme in ("r5","r6"):
            for direction in ("down","up"):
                file=PULSE/f"results/lab_equivalent_runs/{scheme}/{direction}/pulse-L4-shots1000/parameters.json"
                par=read(file);self.sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
                self.blocks[scheme,direction]=self.compile(p.router(scheme,direction,par["paired"],par["angles"]))
        self.h=target_rotation(4,0,np.pi/2,np.pi/2)@np.diag([1,-1,-1,-1])
        self.x=target_rotation(4,0,np.pi,0)
        self.cz=np.diag([(-1)**(int(a==1)*int(b==1)) for a in range(4) for b in range(4)]).astype(complex)
        self.pauli=[np.eye(4,dtype=complex)]
        for block in (p.X,p.Y,p.Z):
            matrix=np.eye(4,dtype=complex);matrix[:2,:2]=block;self.pauli.append(matrix)

    def compile(self,operations):
        model=self.model;time=0.;frames=np.zeros((4,2));layers=[]
        for layer in physical_layers(operations):
            old=frames.copy();compiled,duration,new=model.compile_layer(layer,time,frames)
            before=[];after=[]
            for wire in range(4):
                theta0=np.r_[0,old[wire,0]+np.arange(3)*old[wire,1]]
                theta1=np.r_[0,new[wire,0]+np.arange(3)*new[wire,1]]
                before.append(np.exp(-1j*model.energy[wire]*time+1j*theta0))
                after.append(np.exp(1j*model.energy[wire]*(time+duration)-1j*theta1))
            ops=[];touched=set()
            for name,axes,pulse,phase in compiled:
                key=json.dumps(pulse.__dict__,sort_keys=True)
                if key not in model.primitive_cache:
                    model.primitive_cache[key]=model.cz_raw(pulse) if name=="cz" else model.singles_raw(pulse)
                raw=model.primitive_cache[key]
                if name!="cz":
                    wire=axes[0];lower=int(name=="12_pi")
                    angle=phase-2*np.pi*(model.frequencies[wire]+lower*model.alpha[wire]-model.rotating)*time
                    diagonal=np.exp(1j*angle*model.n)
                    raw=diagonal[:,None]*raw*diagonal.conj()[None,:]
                energy=sum(model.energy[q].reshape((1,)*k+(4,)+(1,)*(len(axes)-k-1)) for k,q in enumerate(axes)).reshape(-1)
                pad=np.exp(-1j*energy*(duration-pulse.duration))
                pre=p._kron(np.diag(before[q]) for q in axes).diagonal()
                post=p._kron(np.diag(after[q]) for q in axes).diagonal()
                effective=(post*pad)[:,None]*raw*pre[None,:]
                ops.append({"name":name,"wires":list(axes),"u":effective})
                touched.update(axes)
            for wire in set(range(4))-touched:
                diag=after[wire]*np.exp(-1j*model.energy[wire]*duration)*before[wire]
                if np.max(abs(diag-1))>1e-10:ops.append({"name":"frame","wires":[wire],"u":np.diag(diag)})
            layers.append({"duration":duration,"ops":ops,"kind":"cz" if any(o["name"]=="cz" for o in ops) else "single"})
            time+=duration;frames=new
        return layers

    def mapped_router(self,scheme,direction,wires):
        return [{"duration":layer["duration"],"kind":layer["kind"],
                 "ops":[{**op,"wires":[wires[q] for q in op["wires"]],"error":"local"} for op in layer["ops"]]}
                for layer in self.blocks[scheme,direction]]

    def cnot(self,control,target,cross=False):
        duration=80. if cross else 50.
        return [{"duration":32.,"kind":"single","ops":[{"name":"01_half","wires":[target],"u":target_rotation(4,0,np.pi/2,-np.pi/2),"error":"single"}]},
                {"duration":duration,"kind":"cz","ops":[{"name":"cz","wires":[control,target],"u":self.cz,"error":"cross" if cross else "local"}]},
                {"duration":32.,"kind":"single","ops":[{"name":"01_half","wires":[target],"u":target_rotation(4,0,np.pi/2,np.pi/2),"error":"single"}]}]

    def swap(self,a,b,cross=False):return self.cnot(a,b,cross)+self.cnot(b,a,cross)+self.cnot(a,b,cross)


@lru_cache(maxsize=4096)
def damping(dt,t1,aux):
    rates=np.array([0.,1.,2. if aux else 0.,3.])/t1
    generator=np.diag(-rates)
    for level in range(1,4):generator[level-1,level]=rates[level]
    transition=expm(generator*dt)
    result=[np.diag(np.sqrt(np.maximum(np.diag(transition),0)))]
    for source in range(1,4):
        for target in range(source):
            if transition[target,source]>1e-18:
                value=np.zeros((4,4));value[target,source]=np.sqrt(transition[target,source]);result.append(value)
    return tuple(result)


def noise_layer(state,memory,dt,noise,rng,quantum):
    if not dt:return memory
    if noise.loss:
        state.channel_sweep(damping(round(dt,9),noise.t1_ns,noise.aux_decay),rng,quantum)
    if noise.dephasing:
        inverse=1/noise.t2_ns-1/(2*noise.t1_ns)
        if inverse < -1e-12:raise ValueError("T2 must not exceed 2T1.")
        if inverse>0:
            selected=set(quantum)
            for index,label in enumerate(state.labels):
                if label not in selected:continue
                angle=rng.normal(0,np.sqrt(2*dt*inverse))
                state.a[index]*=np.exp(-1j*angle*np.arange(4))[None,:,None]
    if noise.loss and noise.memory_decay:
        memory=memory.copy()
        memory[(memory==1)&(rng.random(len(memory)) < 1-np.exp(-dt/noise.t1_ns))]=0
    return memory


def evolve_layer(state,memory,layer,lib,noise,rng,quantum,scale=1.):
    dt=layer["duration"]
    memory=noise_layer(state,memory,dt/2,noise,rng,quantum)
    for op in layer["ops"]:
        name=op["name"];wires=op["wires"]
        if name=="memory_cz":
            j=op["memory"]
            if memory[j]:state.apply(np.diag([1,-1,1,1]),wires)
            prob=noise.memory_cz*scale
            if rng.random()<prob:
                error=int(rng.integers(1,16));a,b=divmod(error,4)
                if a:state.apply(lib.pauli[a],wires)
                if b in (1,2):memory[j]^=1
        else:
            state.apply(op["u"],wires)
            if name=="frame":continue
            prob=(noise.cross_cz if op["error"]=="cross" else noise.local_cz) if name=="cz" else noise.x12 if name=="12_pi" else noise.single
            if rng.random()<prob*scale:
                if len(wires)==1:state.apply(lib.pauli[int(rng.integers(1,4))],wires)
                else:
                    error=int(rng.integers(1,16));a,b=divmod(error,4)
                    if a:state.apply(lib.pauli[a],(wires[0],))
                    if b:state.apply(lib.pauli[b],(wires[1],))
    return noise_layer(state,memory,dt/2,noise,rng,quantum)
