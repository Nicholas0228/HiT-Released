"""Nonzero attention FLOP estimates for the existing AOT memory optimizer.

This changes only saved-activation selection, never the attention function.
Dense estimate is exact GEMM FLOPs. Packed estimate uses total_tokens*max_length
as a declared upper bound; tensor contents are never read while tracing.
"""
import torch
from torch.utils.flop_counter import register_flop_formula
from . import attention as runtime_attention


def attention_flops(q,cumulative,maximum):
    if cumulative is None:
        batch,n,heads,width=q
        return 4*batch*n*n*heads*width
    tokens,heads,width=q
    return 4*tokens*maximum*heads*width

@register_flop_formula(torch.ops.hnet_fast_flash3.forward)
def forward_cost(q,k,v,cumulative,maximum,scale,*,out_shape=None):
    return attention_flops(q,cumulative,maximum)

@register_flop_formula(torch.ops.hnet_fast_flash3.backward)
def backward_cost(grad,q,k,v,output,lse,cumulative,maximum,scale,*,out_shape=None):
    return 5*attention_flops(q,cumulative,maximum)//2
