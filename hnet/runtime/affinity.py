"""Reuse the exact dense routing affinity for mathematically equivalent logits.

Mean over the hard group's keys equals dotting the group's mean key. A segmented
sum in DFS order avoids both a second affinity calculation and prefix-subtraction
cancellation. Full-tree construction and full-tree budget replay are unchanged.
FP32 reduction order differs from the unfused reader; see the validation report.
"""
import torch
import triton
import triton.language as tl


@torch.library.custom_op('hnet_fast_affinity::symmetric', mutates_args=())
def symmetric(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    with torch.autocast('cuda', enabled=False):
        product = q @ k.transpose(-1, -2)
        return .5 * (product + product.transpose(-1, -2))


@symmetric.register_fake
def _symmetric_fake(q, k):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[1]))


def _symmetric_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _symmetric_backward(ctx, grad):
    q, k = ctx.saved_tensors
    product_grad = .5 * (grad + grad.transpose(-1, -2))
    return torch.bmm(product_grad, k), torch.bmm(product_grad.transpose(-1, -2), q)


symmetric.register_autograd(_symmetric_backward, setup_context=_symmetric_setup)


@triton.jit
def _segment_add(a, ag, b, bg):
    return tl.where(ag == bg, a + b, b), bg


@triton.jit
def _means(A, GROUPS, ORDER, SIZES, LENGTHS, OUT, N: tl.constexpr,
           K: tl.constexpr, PN: tl.constexpr, BM: tl.constexpr):
    b = tl.program_id(1)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.arange(0, PN)
    n = tl.load(LENGTHS + b)
    order = tl.load(ORDER + b * N + col, col < n, 0)
    gid = tl.load(GROUPS + b * N + order, col < n, -1).to(tl.int32)
    values = tl.load(A + (b * N + row[:, None]) * N + order[None, :],
                     (row[:, None] < N) & (col[None, :] < n), 0.)
    result, _ = tl.associative_scan((values, tl.broadcast_to(gid[None, :], (BM, PN))),
                                    axis=1, combine_fn=_segment_add)
    next_order = tl.load(ORDER + b * N + col + 1, col + 1 < n, 0)
    next_gid = tl.load(GROUPS + b * N + next_order, col + 1 < n, -1).to(tl.int32)
    end = (col < n) & ((col + 1 == n) | (gid != next_gid))
    sizes = tl.load(SIZES + b * K + gid, end, 1.)
    tl.store(OUT + (b * N + row[:, None]) * K + gid[None, :],
             result / sizes[None, :], (row[:, None] < N) & end[None, :])


@triton.jit
def _means_grad(GRAD, GROUPS, SIZES, LENGTHS, OUT, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(2)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    n = tl.load(LENGTHS + b)
    group = tl.load(GROUPS + b * N + col, col < n, 0)
    size = tl.load(SIZES + b * K + group, col < n, 1.)
    value = tl.load(GRAD + (b * N + row[:, None]) * K + group[None, :],
                    (row[:, None] < N) & (col[None, :] < n), 0.)
    tl.store(OUT + (b * N + row[:, None]) * N + col[None, :],
             value / size[None, :], (row[:, None] < N) & (col[None, :] < N))


@torch.library.custom_op('hnet_fast_affinity::group_means', mutates_args=())
def group_means(affinity: torch.Tensor, groups: torch.Tensor, order: torch.Tensor,
                sizes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch, n, _ = affinity.shape
    width = sizes.shape[1]
    output = affinity.new_zeros((batch, n, width))
    _means[(triton.cdiv(n, 4), batch)](affinity, groups, order, sizes, lengths, output,
        n, width, triton.next_power_of_2(n), 4, num_warps=4, enable_fp_fusion=False)
    return output


@group_means.register_fake
def _means_fake(affinity, groups, order, sizes, lengths):
    return affinity.new_empty((affinity.shape[0], affinity.shape[1], sizes.shape[1]))


@torch.library.custom_op('hnet_fast_affinity::group_means_backward', mutates_args=())
def group_means_backward(grad: torch.Tensor, groups: torch.Tensor, sizes: torch.Tensor,
                         lengths: torch.Tensor) -> torch.Tensor:
    batch, n, width = grad.shape
    grad = grad.contiguous()
    result = grad.new_empty((batch, n, n))
    _means_grad[(triton.cdiv(n, 16), triton.cdiv(n, 128), batch)](
        grad, groups, sizes, lengths, result, n, width, 16, 128, num_warps=4)
    return result


@group_means_backward.register_fake
def _means_grad_fake(grad, groups, sizes, lengths):
    return grad.new_empty((grad.shape[0], grad.shape[1], grad.shape[1]))


def _means_setup(ctx, inputs, output):
    _, groups, _, sizes, lengths = inputs
    ctx.save_for_backward(groups, sizes, lengths)


def _means_backward(ctx, grad):
    groups, sizes, lengths = ctx.saved_tensors
    return group_means_backward(grad, groups, sizes, lengths), None, None, None, None


group_means.register_autograd(_means_backward, setup_context=_means_setup)


@torch.library.custom_op('hnet_fast_affinity::group_logits_factorized_backward', mutates_args=())
def group_logits(q: torch.Tensor, k: torch.Tensor, qm: torch.Tensor, km: torch.Tensor,
                 affinity: torch.Tensor, groups: torch.Tensor, order: torch.Tensor,
                 sizes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return group_means(affinity, groups, order, sizes, lengths)


@group_logits.register_fake
def _logits_fake(q, k, qm, km, affinity, groups, order, sizes, lengths):
    return q.new_empty((q.shape[0], q.shape[1], qm.shape[1]))


def _logits_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4])


def _logits_backward(ctx, grad):
    # Same mathematical derivative, with the original factorization and native
    # backward GEMM policy. No derivative through hard grouping is introduced.
    q, k, qm, km = ctx.saved_tensors
    half = .5 * grad
    return (torch.bmm(half, km), torch.bmm(half, qm),
            torch.bmm(half.transpose(-1, -2), k),
            torch.bmm(half.transpose(-1, -2), q), None, None, None, None, None)


group_logits.register_autograd(_logits_backward, setup_context=_logits_setup)
