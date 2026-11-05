"""Only circuit feasibility: coherent propagation, no added stochastic model.

An H on the external QPU is explicitly compiled as Y01/2 then X01. All
results retain the existing router pulse model and ideal interface process
assumptions, which are declared rather than upgraded to a 17-body claim.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,hashlib,itertools,shutil,time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'DistributedGroverSimulation20261102'
OUT=HERE/'feasibility';RESULTS=HERE/'results'
sys.path.insert(0,str(BASE))
from model import Library,Noise,evolve_layer,PULSE,read
from circuit import Circuit
from runner import input_bank,pack,target_vector,comp_indices,STATES,aggregate
from mps import MPS
from pulse_model import PulseModel,target_rotation,apply_axes
from bootstrap import protocol as p
from validate_extra import logical

ZERO=Noise(single=0,x12=0,local_cz=0,cross_cz=0,memory_cz=0,loss=False,dephasing=False,memory_decay=False,aux_decay=False,readout=False)


@lru_cache(maxsize=1)
def lib():return Library()


class NativeCircuit(Circuit):
    def __init__(self,library,n=2,scheme='r5'):
        super().__init__(library,n,scheme);new=[]
        for layer in self.stages:
            if any(op['name']=='H' for op in layer['ops']):
                assert all(op['name']=='H' for op in layer['ops']) and layer['kind']=='single'
                for name,U in [('01_half',target_rotation(4,0,np.pi/2,np.pi/2)),('01_pi',target_rotation(4,0,np.pi,0))]:
                    new.append({**layer,'duration':32.,'ops':[{**op,'name':name,'u':U} for op in layer['ops']]})
            else:new.append(layer)
        for k,layer in enumerate(new):layer['index']=k
        self.stages=new;self.query_stages=next((k for k,l in enumerate(new) if l['stage']=='diffusion'),len(new));self.layer_count=len(new)
        assert all(o['name'] in ('01_half','01_pi','12_pi','cz','memory_cz','frame') for l in new for o in l['ops'])


def archive_report():
    target=HERE/'archive_gate_error_version';target.mkdir(exist_ok=True)
    for source in [HERE/'document.py',HERE/'README.md']+[RESULTS/name for name in ('分布式Grover_QRAM_不含T1T2_统一验证报告.pdf','REPORT.md','report_manifest.json','metrics.json','figure_catalog.json','pdf_checks.json')]:
        if source.exists() and not (target/source.name).exists():shutil.copy2(source,target/source.name)


def select_choi(state,inputs):
    s=state.copy();head=['ref']+inputs;t=s.head_tensor(head).reshape(8,64,-1)[:,comp_indices(3),:]
    raw=t.reshape(64,-1)@t.reshape(64,-1).conj().T
    env=np.ones(1,complex)
    for a in reversed(s.a[len(head):]):env=a[:,0,:]@env
    v=t@env;K=np.sqrt(8)*v.T;sel=np.outer(v.reshape(-1),v.reshape(-1).conj())
    return raw,sel,K


def network(job):
    threadpool_limits(limits=1);scheme,memory,bond=job;library=lib();c=NativeCircuit(library,2,scheme)
    labels,vectors,_=input_bank(2,'bank');s=MPS.choi(c.quantum,c.inputs,max_bond=bond,tolerance=1e-12)
    mem=np.array(list(map(int,memory)));rng=np.random.default_rng(5);arrays={};rows={};process={}
    for j,layer in enumerate(c.stages):
        mem=evolve_layer(s,mem,layer,library,ZERO,rng,c.quantum)
        if j+1 not in (c.query_stages,len(c.stages)):continue
        stage='query' if j+1==c.query_stages else 'grover'
        vals=vectors if stage=='query' else [vectors[-1]];names=labels if stage=='query' else ['phase']
        snapshot=pack(s.copy().reduce_inputs(c.inputs,vals,np.array([1.,0,0,0])),3)
        # Keep only raw and actual auxiliary-zero branches; no binary POVM.
        for key,value in snapshot.items():arrays[stage+'_'+key]=value[:,:2]
        rr=[]
        for k,name in enumerate(names):
            target=target_vector(2,mem,name,stage=='grover');norm=snapshot['norm'][k,0];p0=snapshot['norm'][k,1]/norm
            F=float(np.vdot(target,snapshot['block'][k,0]@target).real/norm)
            F0=float(np.vdot(target,snapshot['block'][k,1]@target).real/snapshot['norm'][k,1])
            row=dict(label=name,F=F,F_aux0=F0,p_aux0=float(p0),AD_comp_survival=float(np.trace(snapshot['block'][k,0]).real/norm))
            if stage=='grover' and memory.count('1')==1:
                mark=memory.index('1');row.update(success=float(snapshot['address'][k,0,mark,mark].real/norm),
                    success_aux0=float(snapshot['address'][k,1,mark,mark].real/snapshot['norm'][k,1]),
                    joint_correct_aux0=float(snapshot['address'][k,1,mark,mark].real/norm))
            rr.append(row)
        rows[stage]=rr
        if stage=='query':
            J,J0,K=select_choi(s,c.inputs);U=np.zeros((8,8))
            for a,d in itertools.product(range(4),range(2)):U[2*a+(d^int(memory[a])),2*a+d]=1
            v=U.T.reshape(-1)/np.sqrt(8);Fe=float(np.vdot(v,J@v).real);Fe0=float(np.vdot(v,J0@v).real)
            S=float(np.trace(J).real);P0=float(np.trace(J0).real)
            process=dict(Fe=Fe,Fe_aux0_joint=Fe0,aux0_Choi_overlap=Fe0/P0,p_aux0_average=P0,S=S,Favg=(8*Fe+S)/9,
                error_AD_leak=max(0.,1-S),error_aux_nonzero=max(0.,S-P0-(Fe-Fe0)),error_inside_aux0=max(0.,P0-Fe0))
            arrays.update(choi=J,choi_aux0=J0,K0=K,ideal_U=U)
    return dict(scheme=scheme,memory=memory,bond=bond,rows=rows,process=process,
                discarded=s.discarded,peak_bond=s.peak_bond,arrays=arrays,memory_preserved=bool(np.array_equal(mem,np.array(list(map(int,memory))))))


def local():
    library=lib();names=[''.join(v) for v in itertools.product('01+-',repeat=2)];vectors=[np.kron(STATES[x[0]],STATES[x[1]]) for x in names]
    saved={};summary={};inids=[0,16,64,80];outids=[0,4,64,65];words=np.array(list(np.ndindex(4,4,4,4)))
    for scheme in ('r5','r6'):
        unitaries={}
        for direction in ('down','up'):
            u=np.eye(256,dtype=complex)
            for layer in library.blocks[scheme,direction]:
                for op in layer['ops']:u=apply_axes(u,op['u'],op['wires'],4)
            unitaries[direction]=u
        for direction in ('down','up','roundtrip'):
            U=unitaries['down'] if direction=='down' else unitaries['up'] if direction=='up' else unitaries['up']@unitaries['down']
            src=inids if direction!='up' else outids;dst=outids if direction=='down' else inids;K=U[np.ix_(dst,src)]
            survival=np.sum(abs(U[:,src][np.all(words<2,axis=1)])**2,axis=0);goals=[];rho=[];f=[]
            for vector in vectors:
                v=np.zeros(256,complex);v[src]=vector;final=(U@v).reshape(4,4,4,4)
                if direction=='down':
                    x=final.transpose(0,2,3,1).reshape(64,4);r=(x@x.conj().T)[np.ix_(comp_indices(3),comp_indices(3))]
                    goal=np.zeros(8,complex);goal[[0,2,4,5]]=vector
                else:
                    x=final.reshape(16,16);r=(x@x.conj().T)[np.ix_(comp_indices(2),comp_indices(2))];goal=vector
                rho.append(r);goals.append(goal);f.append(float(np.vdot(goal,r@goal).real))
            key=scheme+'_'+direction;saved[key+'_rho']=np.array(rho);saved[key+'_goals']=np.array(goals);saved[key+'_K']=K
            summary[key]=dict(labels=names,F=f,Fmean=float(np.mean(f)),Fe_legal=float(abs(np.trace(K))**2/16),
                survival01=float(np.mean(survival)),ideal_image_population=float(np.trace(K.conj().T@K).real/4),truth=(abs(K)**2).tolist())
        # Coherent control sweep and native-layer temporary |2> occupation.
        U=unitaries['down'];theta=np.linspace(0,np.pi,101);sweep=[];clock=0.;trace=[];psi=np.eye(256)[:,0].astype(complex)
        for t in theta:
            v=np.zeros(256,complex);v[16]=np.cos(t/2);v[80]=np.sin(t/2);pop=abs(U@v)**2
            sweep.append([pop[words[:,q]==lev].sum() for q in range(4) for lev in range(4)])
        for layer in library.blocks[scheme,'down']:
            for op in layer['ops']:psi=apply_axes(psi,op['u'],op['wires'],4)
            clock+=layer['duration'];pop=abs(psi)**2
            trace.append([clock]+[pop[words[:,q]==lev].sum() for q in range(4) for lev in range(4)])
        saved[scheme+'_sweep']=np.array(sweep);saved[scheme+'_trace']=np.array(trace)
    # R5 paired-rotation ablation at the frozen endpoint frame; no optimization.
    params=read(PULSE/'results/lab_equivalent_runs/r5/down/pulse-L4-shots1000/parameters.json')
    U,_=library.model.evolve(p.router('r5','down',None,params['angles']),initial=np.eye(256),use_cache=False)
    key='r5_without_pair';K=U[np.ix_(outids,inids)];saved[key+'_K']=K;summary[key]=dict(truth=(abs(K)**2).tolist(),Fe_legal=float(abs(np.trace(K))**2/16))
    np.savez_compressed(OUT/'local.npz',theta=theta,**saved);(OUT/'local.json').write_text(json.dumps(summary,indent=2),encoding='utf8')


def pulse_checks():
    rows=[]
    for levels,dt in [(4,.1),(4,.05),(5,.05)]:
        model=PulseModel(read(PULSE/'device.json'),levels=levels,dt=dt);model.load_calibration(PULSE/'results/native/native_parameters.json')
        idx=lambda word:np.ravel_multi_index(tuple(map(int,word)),(levels,)*4)
        ins=list(map(idx,['0000','0100','1000','1100']));outs=list(map(idx,['0000','0010','1000','1001']))
        for scheme,direction in itertools.product(('r5','r6'),('down','up')):
            par=read(PULSE/f'results/lab_equivalent_runs/{scheme}/{direction}/pulse-L4-shots1000/parameters.json')
            src=ins if direction=='down' else outs;dst=outs if direction=='down' else ins
            initial=np.eye(model.dimension,dtype=complex)[:,src];U,_=model.evolve(p.router(scheme,direction,par['paired'],par['angles']),initial=initial,use_cache=False)
            K=U[dst];rows.append(dict(scheme=scheme,direction=direction,levels=levels,dt_ns=dt,Fe=float(abs(np.trace(K))**2/16)))
    (OUT/'pulse_convergence.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


def tracking():
    saved={}
    for scheme in ('r5','r6'):
        c=NativeCircuit(lib(),2,scheme);s=MPS.product(c.quantum,c.inputs,[STATES['+'],STATES['+'],STATES['-']],max_bond=128,tolerance=1e-12)
        mem=np.array([0,0,1,0]);rng=np.random.default_rng(1);clock=0.;times=[];names=[];pops=[];Ps=[]
        for j,l in enumerate(c.stages):
            clock+=l['duration'];mem=evolve_layer(s,mem,l,lib(),ZERO,rng,c.quantum)
            if j+1<len(c.stages) and c.stages[j+1]['stage']==l['stage']:continue
            copy=s.copy();values=[]
            for label in c.quantum:
                pos=copy.labels.index(label);copy.move(pos);values.append(np.sum(abs(copy.a[pos])**2,axis=(0,2)))
            pops.append(values);times.append(clock);names.append(l['stage'])
        saved[scheme+'_population']=np.array(pops);saved[scheme+'_times']=np.array(times)
    np.savez_compressed(OUT/'tracking.npz',stages=names,roles=c.quantum,**saved)


def scaling():
    rows=[]
    for n in (1,3):
        c=NativeCircuit(lib(),n,'r5')
        for memory in ('1'*(2**n),'1'+'0'*(2**n-1)):
            s=MPS.product(c.quantum,c.inputs,[STATES['+']]*n+[STATES['-']],max_bond=128,tolerance=1e-12)
            mem=np.array(list(map(int,memory)));rng=np.random.default_rng(2)
            for layer in c.stages[:c.query_stages]:mem=evolve_layer(s,mem,layer,lib(),ZERO,rng,c.quantum)
            a=pack(s.copy().reduce_inputs(c.inputs,[np.ones(1)],np.array([1.,0,0,0])),n+1)
            g=target_vector(n,mem,'phase');F=float(np.vdot(g,a['block'][0,0]@g).real);p0=float(a['norm'][0,1]);F0=float(np.vdot(g,a['block'][0,1]@g).real/p0)
            rows.append(dict(n=n,memory=memory,bond=128,F=F,p_aux0=p0,F_aux0=F0,discarded=s.discarded))
            print('SCALE',n,memory,F,flush=True)
    (OUT/'scaling.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


def main():
    archive_report();OUT.mkdir(exist_ok=True)
    for name in ('arrays','cases','figures'):(OUT/name).mkdir(exist_ok=True)
    manifest=dict(scope='Circuit feasibility only; coherent propagation; all added stochastic, preparation and readout errors OFF',
        H_decomposition='chronological Y01(pi/2), X01(pi); equals -i H on 01',
        source_sha256={str(x.relative_to(HERE.parent)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__)]+[BASE/n for n in ('model.py','mps.py','circuit.py','runner.py')]})
    path=OUT/'manifest.json'
    if path.exists() and read(path)!=manifest:raise RuntimeError('Feasibility source changed; preserve prior data before a new run.')
    path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    resources={};library=lib()
    for n,scheme in itertools.product((1,2,3),('r5','r6')):
        c=NativeCircuit(library,n,scheme);row=c.resource();row['CZ_depth']=sum(l['kind']=='cz' for l in c.stages);row['physical_layers']=len(c.stages);resources[f'{n}_{scheme}']=row
        (OUT/f'schedule_{n}_{scheme}.json').write_text(json.dumps(c.serializable(),indent=2),encoding='utf8')
    (OUT/'resources.json').write_text(json.dumps(resources,indent=2),encoding='utf8');(OUT/'pulse_sources.json').write_text(json.dumps(library.sources,indent=2),encoding='utf8')
    # Exact gate-set check including chronological convention.
    H=np.array([[1,1],[1,-1]])/np.sqrt(2);Y=target_rotation(4,0,np.pi/2,np.pi/2)[:2,:2];X=target_rotation(4,0,np.pi,0)[:2,:2]
    np.testing.assert_allclose(X@Y,-1j*H,atol=1e-14)
    (OUT/'ideal_validation.json').write_text(json.dumps(logical(),indent=2),encoding='utf8')
    jobs=[];summaries={}
    for scheme in ('r5','r6'):
        for word in itertools.product('01',repeat=4):
            memory=''.join(word);key=scheme+'_'+memory;path=OUT/'cases'/(key+'.json')
            if path.exists():summaries[key]=read(path)
            else:jobs.append((scheme,memory,128))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(network,j) for j in jobs]):
            value=future.result();key=value['scheme']+'_'+value['memory'];arrays=value.pop('arrays')
            np.savez_compressed(OUT/'arrays'/(key+'.npz'),**arrays);(OUT/'cases'/(key+'.json')).write_text(json.dumps(value,indent=2),encoding='utf8');summaries[key]=value
            print('NETWORK',key,value['process']['Fe'],flush=True)
    (OUT/'SUMMARY.json').write_text(json.dumps(summaries,indent=2),encoding='utf8')
    local();pulse_checks();tracking();scaling();print('DONE',OUT,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
