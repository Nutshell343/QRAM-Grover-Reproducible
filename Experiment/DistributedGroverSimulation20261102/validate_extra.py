"""Explicit-register ideal proof and selected three-level-tree convergence."""
import itertools,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
from runner import circuit,worker,aggregate
from model import Noise

HERE=Path(__file__).resolve().parent;OUT=HERE/'supplement_v1'


def logical():
    c=circuit(2,'r5');actions=[]
    for j in range(2):
        actions.append(('gateway',j))
        for depth in range(j):actions.append(('route',depth))
        actions.append(('store',j))
    calls=0
    def swap(v,a,b):v[a],v[b]=v[b],v[a]
    def route(v,depth,inverse):
        nonlocal calls
        for k in range(2**depth):
            C,I,L,R=c.node(depth,k);cc=v[C];target=R if cc else L;other=L if cc else R
            if inverse:
                assert v[I]==0 and v[other]==0
            else:assert v[L]==v[R]==0
            swap(v,I,target);calls+=1
    def action(v,a,inverse):
        name,j=a
        if name=='gateway':
            swap(v,f'A{j}','D');swap(v,'D','T0');swap(v,f'A{j}','D')
        elif name=='route':route(v,j,inverse)
        else:
            for k in range(2**j):swap(v,c.data[2**j-1+k],c.controls[2**j-1+k])
    maxerror=0;count=0
    for address,d,word in itertools.product(range(4),range(2),itertools.product((0,1),repeat=4)):
        phases=[]
        for encoded in (0,1):
            values={name:0 for name in c.quantum}
            values.update({f'A{j}':int(f'{address:02b}'[j]) for j in range(2)});values['D']=encoded
            for a in actions:action(values,a,False)
            swap(values,'D','T0')
            for depth in range(2):route(values,depth,False)
            for j in range(4):assert values[f'L{j:02b}']==encoded*int(j==address)
            phase=(-1)**sum(values[f'L{j:02b}']*word[j] for j in range(4));phases.append(phase)
            for depth in (1,0):route(values,depth,True)
            swap(values,'D','T0')
            for a in reversed(actions):action(values,a,True)
            assert values['D']==encoded
            assert all(values[x]==0 for x in c.quantum if x not in c.inputs)
            assert all(values[f'A{j}']==int(f'{address:02b}'[j]) for j in range(2))
        h=np.array([[1,1],[1,-1]])/np.sqrt(2);u=h@np.diag(phases)@h
        goal=np.eye(2)[:,d^word[address]];maxerror=max(maxerror,float(max(abs(u[:,d]-goal))));count+=1
    return {'basis_combinations':count,'encoded_branches':2*count,'legal_router_calls_checked':calls,'max_amplitude_error':maxerror,'auxiliary_reset':True,'memory_unchanged_ideal':True}


def main():
    result={'ideal17':logical(),'n3_convergence':{}}
    for bond in (32,64,128):
        job={'id':'scale_3_10000000','memory':'10000000','noise':asdict(Noise()),'count':4,'mode':'phase','n':3,'scheme':'r5','bond':bond,'replicas':list(range(4))}
        value=worker(job);job['id']='n3_convergence_'+str(bond);s=aggregate(job,value,OUT)
        result['n3_convergence'][str(bond)]={**s['stages']['query'][0],'max_discard':s['discarded_sum_max']}
        print('N3',bond,result['n3_convergence'][str(bond)]['F_raw'],flush=True)
    (OUT/'extra_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('IDEAL',result['ideal17'])


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
