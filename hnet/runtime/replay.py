"""Full-tree FP32 replay: retain prefix sums, fuse child-mean temporaries.

Same four interval endpoints and original FP32 DFS prefix construction. This
does not truncate the tree or replace the budget by a different surrogate.
"""
import torch
from torch.nn import functional as F
import triton
import triton.language as tl


@triton.jit
def _scores(PQ, PK, IV, OUT, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
            BD: tl.constexpr, BM: tl.constexpr):
    b = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    live = m < M
    a = tl.load(IV + (b * M + m) * 4, live, 0)
    z = tl.load(IV + (b * M + m) * 4 + 1, live, 1)
    c = tl.load(IV + (b * M + m) * 4 + 2, live, 0)
    e = tl.load(IV + (b * M + m) * 4 + 3, live, 1)
    mask = live[:, None] & (d[None, :] < D)
    aq = tl.load(PQ + (b * N + a[:, None]) * D + d[None, :], mask, 0.)
    zq = tl.load(PQ + (b * N + z[:, None]) * D + d[None, :], mask, 0.)
    cq = tl.load(PQ + (b * N + c[:, None]) * D + d[None, :], mask, 0.)
    eq = tl.load(PQ + (b * N + e[:, None]) * D + d[None, :], mask, 0.)
    ak = tl.load(PK + (b * N + a[:, None]) * D + d[None, :], mask, 0.)
    zk = tl.load(PK + (b * N + z[:, None]) * D + d[None, :], mask, 0.)
    ck = tl.load(PK + (b * N + c[:, None]) * D + d[None, :], mask, 0.)
    ek = tl.load(PK + (b * N + e[:, None]) * D + d[None, :], mask, 0.)
    nl = (z - a).to(tl.float32)[:, None]
    nr = (e - c).to(tl.float32)[:, None]
    ql, qr = (zq - aq) / nl, (eq - cq) / nr
    kl, kr = (zk - ak) / nl, (ek - ck) / nr
    out = 0.5 * (tl.sum(ql * kr, 1) + tl.sum(kl * qr, 1))
    tl.store(OUT + b * M + m, out, live)


@triton.jit
def _score_backward(PQ, PK, IV, GO, DQ, DK,
                    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                    BD: tl.constexpr, BM: tl.constexpr):
    b = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    live = m < M
    a = tl.load(IV + (b * M + m) * 4, live, 0)
    z = tl.load(IV + (b * M + m) * 4 + 1, live, 1)
    c = tl.load(IV + (b * M + m) * 4 + 2, live, 0)
    e = tl.load(IV + (b * M + m) * 4 + 3, live, 1)
    g = tl.load(GO + b * M + m, live, 0.) * 0.5
    mask = live[:, None] & (d[None, :] < D)
    oa = (b * N + a[:, None]) * D + d[None, :]
    oz = (b * N + z[:, None]) * D + d[None, :]
    oc = (b * N + c[:, None]) * D + d[None, :]
    oe = (b * N + e[:, None]) * D + d[None, :]
    aq, zq = tl.load(PQ + oa, mask, 0.), tl.load(PQ + oz, mask, 0.)
    cq, eq = tl.load(PQ + oc, mask, 0.), tl.load(PQ + oe, mask, 0.)
    ak, zk = tl.load(PK + oa, mask, 0.), tl.load(PK + oz, mask, 0.)
    ck, ek = tl.load(PK + oc, mask, 0.), tl.load(PK + oe, mask, 0.)
    nl = (z - a).to(tl.float32)[:, None]
    nr = (e - c).to(tl.float32)[:, None]
    ql, qr = (zq - aq) / nl, (eq - cq) / nr
    kl, kr = (zk - ak) / nl, (ek - ck) / nr
    dql, dqr = (g[:, None] * kr) / nl, (g[:, None] * kl) / nr
    dkl, dkr = (g[:, None] * qr) / nl, (g[:, None] * ql) / nr
    # Original four endpoints also cover padded dummy intervals and N=2.
    # Zero-gradient padded nodes contribute exactly zero.
    use = mask & (g[:, None] != 0.)
    tl.atomic_add(DQ + oa, -dql, use, sem='relaxed')
    tl.atomic_add(DQ + oz, dql, use, sem='relaxed')
    tl.atomic_add(DQ + oc, -dqr, use, sem='relaxed')
    tl.atomic_add(DQ + oe, dqr, use, sem='relaxed')
    tl.atomic_add(DK + oa, -dkl, use, sem='relaxed')
    tl.atomic_add(DK + oz, dkl, use, sem='relaxed')
    tl.atomic_add(DK + oc, -dkr, use, sem='relaxed')
    tl.atomic_add(DK + oe, dkr, use, sem='relaxed')


@torch.library.custom_op('hnet_fast_replay::scores', mutates_args=())
def scores(pq: torch.Tensor, pk: torch.Tensor, intervals: torch.Tensor) -> torch.Tensor:
    b, n, d = pq.shape
    m = intervals.shape[1]
    result = torch.empty((b, m), device=pq.device, dtype=torch.float32)
    if m:
        _scores[(triton.cdiv(m, 4), b)](pq, pk, intervals, result, m, n, d,
            triton.next_power_of_2(d), 4, num_warps=4, enable_fp_fusion=False)
    return result


@scores.register_fake
def _scores_fake(pq, pk, intervals):
    return pq.new_empty((pq.shape[0], intervals.shape[1]))


@torch.library.custom_op('hnet_fast_replay::backward', mutates_args=())
def score_backward(pq: torch.Tensor, pk: torch.Tensor, intervals: torch.Tensor,
                   grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from .replay_endpoints import backward as endpoint_backward
    return endpoint_backward(pq, pk, intervals, grad)



@score_backward.register_fake
def _backward_fake(pq, pk, intervals, grad):
    return torch.empty_like(pq), torch.empty_like(pk)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    pq, pk, intervals = ctx.saved_tensors
    dpq, dpk = score_backward(pq, pk, intervals, grad)
    return dpq, dpk, None


scores.register_autograd(_backward, setup_context=_setup)


def fused_replay(q, k, order, intervals):
    d = q.shape[-1]
    index = order[..., None].expand(-1, -1, d)
    pq = F.pad(q.gather(1, index).cumsum(1), (0, 0, 1, 0))
    pk = F.pad(k.gather(1, index).cumsum(1), (0, 0, 1, 0))
    return scores(pq, pk, intervals)
