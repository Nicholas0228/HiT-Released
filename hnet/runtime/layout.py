"""Exact prefix packing with inverse gathers instead of generic atomic gradients."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["R"])
def _pack(X, SID, CU, Y, R, D: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
          BR: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, BD)
    live = r < R
    b = tl.load(SID + r, live, 0)
    start = tl.load(CU + b, live, 0)
    j = r - start
    mask = live[:, None] & (d[None, :] < D)
    value = tl.load(X + b[:, None] * S0 + j[:, None] * S1 + d[None, :] * S2, mask, 0.)
    tl.store(Y + r[:, None] * D + d[None, :], value, mask)


@triton.jit
def _unpack(X, CU, Y, B: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, BD)
    live = r < B * K
    b, j = r // K, r % K
    start = tl.load(CU + b, live, 0)
    end = tl.load(CU + b + 1, live, 0)
    mask = live[:, None] & (d[None, :] < D)
    value = tl.load(X + (start + j)[:, None] * S0 + d[None, :] * S1,
                    mask & (j < end - start)[:, None], 0.)
    tl.store(Y + r[:, None] * D + d[None, :], value, mask)


@torch.library.custom_op('hnet_fast_layout::pack', mutates_args=())
def pack(x: torch.Tensor, sequence_ids: torch.Tensor, cumulative: torch.Tensor) -> torch.Tensor:
    rows, width = sequence_ids.numel(), x.shape[-1]
    out = x.new_empty((rows, width))
    if rows:
        _pack[(triton.cdiv(rows, 8),)](x, sequence_ids, cumulative, out, rows, width,
            *x.stride(), 8, triton.next_power_of_2(width), num_warps=4)
    return out


@pack.register_fake
def _pack_fake(x, sequence_ids, cumulative):
    return x.new_empty((sequence_ids.numel(), x.shape[-1]))


@torch.library.custom_op('hnet_fast_layout::unpack', mutates_args=())
def unpack(x: torch.Tensor, sequence_ids: torch.Tensor, cumulative: torch.Tensor,
           width: int) -> torch.Tensor:
    batch, dim = cumulative.numel() - 1, x.shape[-1]
    out = x.new_empty((batch, width, dim))
    _unpack[(triton.cdiv(batch * width, 8),)](x, cumulative, out, batch, width, dim,
        *x.stride(), 8, triton.next_power_of_2(dim), num_warps=4)
    return out


@unpack.register_fake
def _unpack_fake(x, sequence_ids, cumulative, width):
    return x.new_empty((cumulative.numel() - 1, width, x.shape[-1]))


def _pack_setup(ctx, inputs, output):
    x, sequence_ids, cumulative = inputs
    ctx.save_for_backward(sequence_ids, cumulative)
    ctx.width = x.shape[1]


def _pack_backward(ctx, grad):
    sequence_ids, cumulative = ctx.saved_tensors
    return unpack(grad, sequence_ids, cumulative, ctx.width), None, None


def _unpack_setup(ctx, inputs, output):
    _, sequence_ids, cumulative, _ = inputs
    ctx.save_for_backward(sequence_ids, cumulative)


def _unpack_backward(ctx, grad):
    sequence_ids, cumulative = ctx.saved_tensors
    return pack(grad, sequence_ids, cumulative), None, None, None


pack.register_autograd(_pack_backward, setup_context=_pack_setup)
unpack.register_autograd(_unpack_backward, setup_context=_unpack_setup)
