"""Packed adaLN gather with contiguous segmented gradient reduction.

The forward stays an ordinary index_select so AOT may fuse it with modulation.
Backward sums each contiguous sequence in FP32, then casts to its input dtype.
This changes floating-point reduction order, not the conditioning operation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(G, CU, P, D: tl.constexpr, S: tl.constexpr,
             BR: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2) * BD + tl.arange(0, BD)
    start, end = tl.load(CU + b), tl.load(CU + b + 1)
    r = start + s * BR + tl.arange(0, BR)
    value = tl.load(G + r[:, None].to(tl.int64) * D + d[None, :],
                    (r[:, None] < end) & (d[None, :] < D), 0.).to(tl.float32)
    total = tl.sum(value, axis=0)
    tl.store(P + (b * S + s) * D + d, total, d < D)


@triton.jit
def _finish(P, O, D: tl.constexpr, S: tl.constexpr, PS: tl.constexpr,
            BD: tl.constexpr):
    b = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    s = tl.arange(0, PS)
    values = tl.load(P + (b * S + s[:, None]) * D + d[None, :],
                     (s[:, None] < S) & (d[None, :] < D), 0.)
    total = tl.sum(values, axis=0)
    tl.store(O + b * D + d, total, d < D)


@torch.library.custom_op('hnet_fast_conditioning::reduce', mutates_args=())
def reduce(grad: torch.Tensor, cumulative: torch.Tensor, maximum: int) -> torch.Tensor:
    batch, width = cumulative.numel() - 1, grad.shape[-1]
    grad = grad.contiguous()
    splits = triton.cdiv(maximum, 128)
    partial = torch.empty((batch, splits, width), dtype=torch.float32, device=grad.device)
    output = grad.new_empty((batch, width))
    _partial[(batch, splits, triton.cdiv(width, 64))](
        grad, cumulative, partial, width, splits, 128, 64, num_warps=4)
    _finish[(batch, triton.cdiv(width, 128))](
        partial, output, width, splits, triton.next_power_of_2(splits), 128,
        num_warps=4)
    return output


@reduce.register_fake
def _reduce_fake(grad, cumulative, maximum):
    return grad.new_empty((cumulative.numel() - 1, grad.shape[-1]))


class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, sequence_ids, cumulative, maximum):
        ctx.save_for_backward(cumulative)
        ctx.maximum = maximum
        return value.index_select(0, sequence_ids)

    @staticmethod
    def backward(ctx, grad):
        (cumulative,) = ctx.saved_tensors
        return reduce(grad, cumulative, ctx.maximum), None, None, None


def expand(value, sequence_ids, cumulative, maximum):
    return _Expand.apply(value, sequence_ids, cumulative, maximum)
