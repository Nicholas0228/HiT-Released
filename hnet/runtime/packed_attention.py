"""Autograd adapter for the variable-length FlashAttention shipped in PyTorch.

The private ATen interface is checked in the GPU parity suite and pinned to the
campaign's Torch 2.5.1 environment. Each image remains a separate sequence.
"""
import torch


class _PackedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, cumulative, maximum, scale):
        output, logsumexp, seed, offset, _ = torch.ops.aten._flash_attention_forward(
            query, key, value, cumulative, cumulative, maximum, maximum,
            0., False, False, scale=scale)
        ctx.save_for_backward(query, key, value, output, logsumexp, cumulative, seed, offset)
        ctx.maximum = maximum
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, gradient):
        query, key, value, output, logsumexp, cumulative, seed, offset = ctx.saved_tensors
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            gradient.contiguous(), query, key, value, output, logsumexp,
            cumulative, cumulative, ctx.maximum, ctx.maximum, 0., False, seed, offset, scale=ctx.scale)
        return dq, dk, dv, None, None, None


def packed_attention(query, key, value, cumulative, maximum, scale=None):
    if query.device.type != 'cuda' or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('Packed attention requires CUDA FP16 or BF16 tensors')
    return _PackedAttention.apply(query, key, value, cumulative, maximum, scale)
