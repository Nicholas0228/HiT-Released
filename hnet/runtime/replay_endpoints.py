"""Full-tree budget VJP by endpoint incidence, without floating-point atomics.
Forward unchanged. The constant prefix at position zero has no derivative.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _gather(PQ,PK,IV,GO,EDGE,OFF,DQ,DK,
            M:tl.constexpr,N:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    endpoint=tl.program_id(0)
    batch=tl.program_id(1)
    d=tl.program_id(2)*BD+tl.arange(0,BD)
    begin=tl.load(OFF+batch*(N+1)+endpoint)
    end=tl.load(OFF+batch*(N+1)+endpoint+1)
    totalq=tl.full((BR,BD),0,tl.float32)
    totalk=tl.full((BR,BD),0,tl.float32)
    for start in range(begin,end,BR):
        row=start+tl.arange(0,BR)
        edge=tl.load(EDGE+batch*(M*4)+row,row<end,0)
        node=edge//4
        side=edge%4
        a=tl.load(IV+(batch*M+node)*4,row<end,0)
        z=tl.load(IV+(batch*M+node)*4+1,row<end,1)
        c=tl.load(IV+(batch*M+node)*4+2,row<end,0)
        e=tl.load(IV+(batch*M+node)*4+3,row<end,1)
        g=tl.load(GO+batch*M+node,row<end,0.)*0.5
        left=side<2
        lo=tl.where(left,c,a)
        hi=tl.where(left,e,z)
        denom_other=tl.where(left,e-c,z-a).to(tl.float32)
        denom_self=tl.where(left,z-a,e-c).to(tl.float32)
        mask=(row[:,None]<end)&(d[None,:]<D)
        oq=tl.load(PQ+(batch*N+lo[:,None])*D+d[None,:],mask,0.)
        hq=tl.load(PQ+(batch*N+hi[:,None])*D+d[None,:],mask,0.)
        ok=tl.load(PK+(batch*N+lo[:,None])*D+d[None,:],mask,0.)
        hk=tl.load(PK+(batch*N+hi[:,None])*D+d[None,:],mask,0.)
        vq=(g[:,None]*((hk-ok)/denom_other[:,None]))/denom_self[:,None]
        vk=(g[:,None]*((hq-oq)/denom_other[:,None]))/denom_self[:,None]
        positive=(side%2)==1
        vq=tl.where(positive[:,None],vq,-vq)
        vk=tl.where(positive[:,None],vk,-vk)
        totalq+=tl.where(mask,vq,0.)
        totalk+=tl.where(mask,vk,0.)
    tl.store(DQ+(batch*N+endpoint)*D+d,tl.sum(totalq,0),d<D)
    tl.store(DK+(batch*N+endpoint)*D+d,tl.sum(totalk,0),d<D)

def backward(pq,pk,iv,grad,bd=256,br=4,warps=4):
    b,n,d=pq.shape;m=iv.shape[1]
    # Exclude zero-gradient dummy merges and the constant prefix. N is sentinel.
    keys=torch.where((grad[...,None]!=0)&(iv>0),iv,n).reshape(b,4*m)
    sorted_keys,edges=keys.sort(dim=1,stable=True)
    positions=torch.arange(n+1,device=pq.device,dtype=iv.dtype).expand(b,-1).contiguous()
    offsets=torch.searchsorted(sorted_keys,positions)
    dq=torch.empty_like(pq);dk=torch.empty_like(pk)
    _gather[(n,b,triton.cdiv(d,bd))](pq,pk,iv,grad.contiguous(),edges,offsets,dq,dk,
        m,n,d,bd,br,num_warps=warps,enable_fp_fusion=False)
    return dq,dk
