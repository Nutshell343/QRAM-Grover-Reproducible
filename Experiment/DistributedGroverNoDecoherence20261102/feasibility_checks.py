"""Explicit native-gate ideal synthesis plus coherent data integrity checks."""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from feasibility_data import NativeCircuit,ZERO,select_choi,OUT,lib,network
from model import Library,evolve_layer
from pulse_model import target_rotation,apply_axes
from bootstrap import protocol as p,physical_layers
from mps import MPS
from runner import target_vector,STATES


def primitive(item):
    name=item['gate'];phase=item['phase']
    if name=='cz':return np.diag([(-1)**int(a==b==1) for a in range(4) for b in range(4)]).astype(complex)
    if name=='frame':return np.diag(np.exp(-1j*phase*np.arange(4)))
    return target_rotation(4,int(name=='12_pi'),np.pi/2 if name=='01_half' else np.pi,phase)


def ideal_library():
    library=Library();solutions={};src0=[0,16,64,80];dst0=[0,4,64,65]
    for scheme,direction in itertools.product(('r5','r6'),('down','up')):
        operations=p.core(scheme,direction,None);U=np.eye(256,dtype=complex)
        for item in operations:U=apply_axes(U,primitive(item),item['qubits'],4)
        src,dst=(src0,dst0) if direction=='down' else (dst0,src0)
        K=U[np.ix_(dst,src)];amplitude=np.diag(K);angles=np.angle(amplitude)
        frame=dict(c=float(angles[2]-angles[0]),l=float(angles[1]-angles[0]),r=float(angles[3]-angles[2]))
        full=p.router(scheme,direction,None,frame);matrix=np.eye(256,dtype=complex)
        for item in full:matrix=apply_axes(matrix,primitive(item),item['qubits'],4)
        corrected=matrix[np.ix_(dst,src)];phase=np.angle(np.trace(corrected));error=float(np.max(abs(corrected*np.exp(-1j*phase)-np.eye(4))))
        assert error<1e-12,(scheme,direction,error)
        layers=[]
        for layer in physical_layers(full):
            ops=[dict(name=x['gate'],wires=x['qubits'],u=primitive(x),error='local') for x in layer]
            active=[x['name'] for x in ops if x['name']!='frame']
            assert not ('cz' in active and any(x!='cz' for x in active))
            duration=max([0 if x['name']=='frame' else 50 if x['name']=='cz' else 40 if x['name']=='12_pi' else 32 for x in ops] or [0])
            layers.append(dict(duration=duration,kind='cz' if 'cz' in active else 'single',ops=ops))
        library.blocks[scheme,direction]=layers
        solutions[scheme+'_'+direction]=dict(frame=frame,raw_phases=angles.tolist(),corrected_max_error=error,
            CZ=sum(x['gate']=='cz' for x in full),operations=full)
    return library,solutions


def main():
    assert not any((ZERO.loss,ZERO.dephasing,ZERO.memory_decay,ZERO.aux_decay,ZERO.readout))
    assert all(getattr(ZERO,k)==0 for k in ('single','x12','local_cz','cross_cz','memory_cz'))
    for n,scheme in itertools.product((1,2,3),('r5','r6')):
        compiled=NativeCircuit(lib(),n,scheme)
        for layer in compiled.stages:
            active=[o for o in layer['ops'] if o['name']!='frame'];hascz=any(o['name'] in ('cz','memory_cz') for o in active)
            if hascz:assert all(o['name'] in ('cz','memory_cz') for o in active)
            used=[]
            for o in active:
                used+=o['wires']
                if o['name']=='memory_cz':used+=['M'+str(o['memory'])]
            assert len(used)==len(set(used)),'Overlapping physical gates in one layer'
    library,solutions=ideal_library();rows=[]
    for scheme in ('r5','r6'):
        c=NativeCircuit(library,2,scheme)
        for word in itertools.product('01',repeat=4):
            memory=''.join(word);m=np.array(list(map(int,word)));s=MPS.choi(c.quantum,c.inputs,max_bond=64,tolerance=1e-12);rng=np.random.default_rng(9)
            row=dict(scheme=scheme,memory=memory)
            for j,layer in enumerate(c.stages):
                m=evolve_layer(s,m,layer,library,ZERO,rng,c.quantum)
                if j+1==c.query_stages:
                    J,J0,K=select_choi(s,c.inputs);goal=np.zeros((8,8))
                    for a,d in itertools.product(range(4),range(2)):goal[2*a+(d^int(memory[a])),2*a+d]=1
                    phase=np.angle(np.trace(goal.conj().T@K));error=float(np.max(abs(K*np.exp(-1j*phase)-goal)))
                    row.update(query_matrix_error=error,aux_zero=float(np.trace(J0).real));assert error<1e-10 and abs(row['aux_zero']-1)<1e-10
            if memory.count('1')==1:
                vec=np.kron(np.ones(4)/2,STATES['-']);rho=s.copy().reduce_inputs(c.inputs,[vec],np.array([1.,0,0,0]))[0][0]
                # Correct address probability includes both data levels.
                t=memory.index('1');ids=[np.ravel_multi_index((int(f'{t:02b}'[0]),int(f'{t:02b}'[1]),d),(4,4,4)) for d in range(4)]
                row['grover_success']=float(np.trace(rho[np.ix_(ids,ids)]).real);assert abs(row['grover_success']-1)<1e-10
            rows.append(row)
    data=json.loads((OUT/'SUMMARY.json').read_text());mineig=0.;maxbounds=0.
    for key,row in data.items():
        assert row['memory_preserved'];a=np.load(OUT/'arrays'/(key+'.npz'))
        for stage in ('query','grover'):
            for k,rhos in enumerate(a[stage+'_block']):
                ns=a[stage+'_norm'][k]
                for j,rho in enumerate(rhos):
                    mineig=min(mineig,float(np.linalg.eigvalsh(rho).min()));maxbounds=max(maxbounds,float(np.trace(rho).real-ns[j]))
                assert np.linalg.eigvalsh(rhos[0]-rhos[1]).min()>-1e-8
    assert mineig>-1e-8 and maxbounds<1e-8
    convergence=[]
    for scheme in ('r5','r6'):
        low=network((scheme,'0010',64));high=data[scheme+'_0010']
        convergence.append(dict(scheme=scheme,Fe64=low['process']['Fe'],Fe128=high['process']['Fe'],
            Grover64=low['rows']['grover'][0]['success'],Grover128=high['rows']['grover'][0]['success'],
            discarded64=low['discarded'],discarded128=high['discarded']))
    result=dict(ideal_native_solutions=solutions,ideal_network=rows,ideal_max_matrix_error=max(r['query_matrix_error'] for r in rows),
        physical_channels='coherent only; loss, dephasing, preparation flips, random gate events and readout confusion all OFF',
        coherent_cases=len(data),min_eigenvalue=mineig,max_trace_bound=maxbounds,network_convergence=convergence,
        all_added_errors_exactly_OFF=True,native_parallel_layer_contract_verified=True,
        source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('feasibility*.py')})
    (OUT/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('CHECKED',len(rows),result['ideal_max_matrix_error'])


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
