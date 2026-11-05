"""Small, auditable mixed-dimension pure-state MPS with trajectory channels.

Adjacent tensor swaps only change the numerical site order. They are never
counted as hardware SWAP gates. Physical gates use apply() with labelled sites.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd


class MPS:
    def __init__(self, tensors, labels, max_bond=32, tolerance=1e-10):
        self.a = [np.asarray(x, complex).copy() for x in tensors]
        self.labels = list(labels)
        self.max_bond = int(max_bond)
        self.tolerance = tolerance
        self.center = 0
        self.discarded = 0.
        self.peak_bond = max(x.shape[-1] for x in self.a)

    @classmethod
    def choi(cls, quantum_labels, input_labels, levels=4, **options):
        n = len(input_labels);dimension = 2**n
        tensor = np.zeros((dimension,)+(levels,)*n, complex)
        for word in range(dimension):
            bits = tuple(map(int, f"{word:0{n}b}"))
            tensor[(word,)+bits] = 1/np.sqrt(dimension)
        dims = tensor.shape;tensors=[];left=1;rest=tensor
        for d in dims[:-1]:
            u,s,v = svd(rest.reshape(left*d,-1),full_matrices=False,check_finite=False)
            rank = max(1,int(np.count_nonzero(s > 1e-13)))
            tensors.append(u[:,:rank].reshape(left,d,rank))
            rest = s[:rank,None]*v[:rank];left=rank
        tensors.append(rest.reshape(left,dims[-1],1))
        for label in quantum_labels:
            if label not in input_labels:
                value=np.zeros((1,levels,1),complex);value[0,0,0]=1;tensors.append(value)
        result=cls(tensors,["ref"]+list(input_labels)+[q for q in quantum_labels if q not in input_labels],**options)
        result.center=n
        return result

    @classmethod
    def product(cls, quantum_labels, input_labels, vectors, levels=4, **options):
        tensors=[np.ones((1,1,1),complex)]
        for label in quantum_labels:
            value=np.zeros(levels,complex)
            if label in input_labels:value[:2]=vectors[input_labels.index(label)]
            else:value[0]=1
            tensors.append(value.reshape(1,levels,1))
        return cls(tensors,["ref"]+list(quantum_labels),**options)

    def dense(self):
        value=self.a[0]
        for site in self.a[1:]:value=np.tensordot(value,site,axes=(-1,0))
        return value.reshape(-1)

    def copy(self):
        result=MPS(self.a,self.labels,self.max_bond,self.tolerance)
        result.center=self.center;result.discarded=self.discarded;result.peak_bond=self.peak_bond
        return result

    def move(self,k):
        while self.center < k:
            i=self.center;l,d,r=self.a[i].shape
            q,rr=np.linalg.qr(self.a[i].reshape(l*d,r),mode="reduced")
            self.a[i]=q.reshape(l,d,q.shape[1])
            self.a[i+1]=np.einsum("ab,bcr->acr",rr,self.a[i+1],optimize=True)
            self.center+=1
        while self.center > k:
            i=self.center;l,d,r=self.a[i].shape
            q,rr=np.linalg.qr(self.a[i].reshape(l,d*r).T,mode="reduced")
            self.a[i]=q.T.reshape(q.shape[1],d,r)
            self.a[i-1]=np.einsum("lar,rs->las",self.a[i-1],rr.T,optimize=True)
            self.center-=1

    def one(self,matrix,index):
        self.a[index]=np.tensordot(matrix,self.a[index],axes=(1,1)).transpose(1,0,2)

    def adjacent(self,matrix,index,renormalize=True):
        self.move(index)
        x,y=self.a[index:index+2];l,d1,_=x.shape;_,d2,r=y.shape
        theta=np.einsum("lax,xbr->labr",x,y,optimize=True)
        theta=np.einsum("abij,lijr->labr",matrix.reshape(d1,d2,d1,d2),theta,optimize=True)
        u,s,v=svd(theta.reshape(l*d1,d2*r),full_matrices=False,check_finite=False,lapack_driver="gesdd")
        rank=max(1,min(self.max_bond,int(np.count_nonzero(s > self.tolerance))))
        lost=float(np.sum(s[rank:]**2));norm=float(np.sum(s[:rank]**2))
        self.discarded+=lost
        if renormalize and norm > 0:s=s/np.sqrt(norm)
        self.a[index]=u[:,:rank].reshape(l,d1,rank)
        self.a[index+1]=(s[:rank,None]*v[:rank]).reshape(rank,d2,r)
        self.center=index+1;self.peak_bond=max(self.peak_bond,rank)

    def reorder_swap(self,index):
        d1=self.a[index].shape[1];d2=self.a[index+1].shape[1]
        self.move(index)
        x,y=self.a[index:index+2];l=x.shape[0];r=y.shape[-1]
        theta=np.einsum("lax,xbr->labr",x,y,optimize=True).transpose(0,2,1,3)
        u,s,v=svd(theta.reshape(l*d2,d1*r),full_matrices=False,check_finite=False)
        rank=max(1,min(self.max_bond,int(np.count_nonzero(s > self.tolerance))))
        self.discarded+=float(np.sum(s[rank:]**2));norm=float(np.sum(s[:rank]**2))
        s=s/np.sqrt(norm)
        self.a[index]=u[:,:rank].reshape(l,d2,rank)
        self.a[index+1]=(s[:rank,None]*v[:rank]).reshape(rank,d1,r)
        self.labels[index],self.labels[index+1]=self.labels[index+1],self.labels[index]
        self.center=index+1;self.peak_bond=max(self.peak_bond,rank)

    def apply(self,matrix,labels):
        if len(labels)==1:
            self.one(matrix,self.labels.index(labels[0]));return
        if len(labels)!=2:raise ValueError("Use one- or two-site primitive gates.")
        x,y=[self.labels.index(q) for q in labels]
        left,right=sorted((x,y))
        while right > left+1:
            self.reorder_swap(right-1);right-=1
        if self.labels[left] != labels[0]:
            d=self.a[left].shape[1]
            matrix=matrix.reshape(d,d,d,d).transpose(1,0,3,2).reshape(d*d,d*d)
        self.adjacent(matrix,left)

    def channel_sweep(self,kraus,rng,labels):
        selected=set(labels)
        diagonal=np.diag(kraus[0])
        jumps=[]
        for matrix in kraus[1:]:
            target,source=np.unravel_index(np.argmax(abs(matrix)),matrix.shape)
            jumps.append((target,source,float(abs(matrix[target,source])**2)))
        for index,label in enumerate(self.labels):
            if label not in selected:continue
            self.move(index)
            tensor=self.a[index]
            pop=np.sum(abs(tensor)**2,axis=(0,2))
            rates=np.r_[np.dot(abs(diagonal)**2,pop),[prob*pop[source] for _,source,prob in jumps]]
            rates=np.maximum(rates,0);rates/=rates.sum()
            chosen=min(len(rates)-1,int(np.searchsorted(np.cumsum(rates),rng.random())))
            if chosen==0:self.a[index]=tensor*diagonal[None,:,None]/np.sqrt(rates[0])
            else:
                target,source,prob=jumps[chosen-1]
                value=np.zeros_like(tensor);value[:,target,:]=tensor[:,source,:]*np.sqrt(prob/rates[chosen])
                self.a[index]=value

    def head_tensor(self,head):
        for k,label in enumerate(head):
            position=self.labels.index(label)
            while position > k:
                self.reorder_swap(position-1);position-=1
        self.move(len(head)-1)
        tensor=self.a[0]
        for site in self.a[1:len(head)]:tensor=np.tensordot(tensor,site,axes=(-1,0))
        return tensor[0]

    def reduce_inputs(self,input_labels,states,pzero):
        """Choi contraction yields unnormalized importance-weighted input outputs."""
        head=["ref"]+list(input_labels)
        tensor=self.head_tensor(head)
        rdim=tensor.shape[0];dimension=int(np.prod(tensor.shape[1:-1]))
        tensor=tensor.reshape(rdim,dimension,-1)
        right=self.a[len(head):]
        env=np.ones(1,complex)
        for site in reversed(right):env=site[:,0,:]@env
        true=tensor@env
        filtered=[value.copy() for value in right]
        for k,value in enumerate(filtered):
            filtered[k]=value*np.sqrt(pzero)[None,:,None]
        # Re-right-canonicalize the filtered complement. No state renormalization.
        for index in range(len(filtered)-1,-1,-1):
            value=filtered[index];l,d,r=value.shape
            qq,rr=np.linalg.qr(value.reshape(l,d*r).T,mode="reduced")
            filtered[index]=qq.T.reshape(qq.shape[1],d,r)
            if index:
                filtered[index-1]=np.einsum("lar,rs->las",filtered[index-1],rr.T,optimize=True)
            else:boundary=rr.T
        observed=tensor@boundary
        outputs=[]
        for vector in states:
            # <psi*|Phi> = psi (not psi.conj()) in the chosen reference basis.
            raw=np.sqrt(rdim)*np.einsum("a,abr->br",vector,tensor,optimize=True)
            acc=np.sqrt(rdim)*np.einsum("a,ab->b",vector,true,optimize=True)
            obs=np.sqrt(rdim)*np.einsum("a,abr->br",vector,observed,optimize=True)
            outputs.append((raw@raw.conj().T,np.outer(acc,acc.conj()),obs@obs.conj().T))
        return outputs
