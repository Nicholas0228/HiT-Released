"""GPU nearest-neighbor chains for the original complete average-linkage tree.

The input affinity arithmetic, FP64 distance updates, tie rule and stable CPU
relabeling follow linkage.py. No approximate search, early cut or window graph.
Only the O(B*N) merge records are downloaded, not the O(B*N*N) affinity.
This optional backend requires CUDA/Triton and is validated independently.

Adapted from SciPy 1.14.1 nn_chain/relabeling via linkage.py, BSD-3-Clause.
Copyright (c) 2001-2002 Enthought, Inc. 2003-2024 SciPy Developers.
See LICENSES/scipy-bsd.txt.
"""
import numpy as np
import math
import time
from numba import njit, prange
import torch
import triton
import triton.language as tl


@triton.jit
def _init_distances(A, D, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + ix, mask=ix < TOTAL, other=0.).to(tl.float64)
    tl.store(D + ix, tl.maximum(1. - a, 0.), mask=ix < TOTAL)


@triton.jit
def _nn_chain(D, LENGTH, RAW, N: tl.constexpr, NP: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    n = tl.load(LENGTH + b)
    ix = tl.arange(0, NP)
    size = (ix < n).to(tl.int32)
    base = b * N * N
    chain = tl.zeros((NP,), tl.int32)
    chain_length = tl.full((), 0, tl.int32)
    for step in range(n - 1):
        if chain_length == 0:
            first = tl.min(tl.where(size > 0, ix, 2147483647), 0)
            chain = tl.where(ix == 0, first, chain)
            chain_length = tl.full((), 1, tl.int32)
        searching = tl.full((), 1, tl.int32)
        x = tl.full((), 0, tl.int32)
        y = tl.full((), 0, tl.int32)
        nearest = tl.full((), float('inf'), tl.float64)
        while searching != 0:
            x = tl.sum(tl.where(ix == chain_length - 1, chain, 0), 0)
            previous = tl.sum(tl.where(ix == tl.maximum(chain_length - 2, 0), chain, 0), 0)
            row = tl.load(D + base + x * N + ix, mask=ix < N, other=float('inf'))
            allowed = (size > 0) & (ix != x)
            minimum = tl.min(tl.where(allowed, row, float('inf')), 0)
            first_minimum = tl.min(tl.where(allowed & (row == minimum), ix, 2147483647), 0)
            prior_distance = tl.load(D + base + x * N + previous)
            # The previous chain vertex wins every exact tie, as in SciPy.
            use_previous = (chain_length > 1) & (prior_distance <= minimum)
            y = tl.where(use_previous, previous, first_minimum)
            nearest = tl.where(use_previous, prior_distance, minimum)
            if (chain_length > 1) & (y == previous):
                searching = tl.full((), 0, tl.int32)
            else:
                tl.device_assert(chain_length < n, 'Nearest-neighbor chain exceeded valid length')
                chain = tl.where(ix == chain_length, y, chain)
                chain_length += 1
        chain_length -= 2
        lo, hi = tl.minimum(x, y), tl.maximum(x, y)
        nx = tl.sum(tl.where(ix == lo, size, 0), 0)
        ny = tl.sum(tl.where(ix == hi, size, 0), 0)
        at = (b * (N - 1) + step) * 4
        tl.store(RAW + at, lo.to(tl.float64))
        tl.store(RAW + at + 1, hi.to(tl.float64))
        tl.store(RAW + at + 2, nearest)
        tl.store(RAW + at + 3, (nx + ny).to(tl.float64))
        size = tl.where(ix == lo, 0, tl.where(ix == hi, nx + ny, size))
        active = (size > 0) & (ix != hi)
        dx = tl.load(D + base + ix * N + lo, mask=active, other=0.)
        dy = tl.load(D + base + ix * N + hi, mask=active, other=0.)
        # Match separate CPU FP64 mul/add and correctly rounded division.
        numerator = nx.to(tl.float64) * dx + ny.to(tl.float64) * dy
        updated = tl.inline_asm_elementwise('div.rn.f64 $0, $1, $2;',
            constraints='=d,d,d', args=[numerator, (nx + ny).to(tl.float64)],
            dtype=tl.float64, is_pure=True, pack=1)
        tl.debug_barrier()
        tl.store(D + base + ix * N + hi, updated, mask=active)
        tl.store(D + base + hi * N + ix, updated, mask=active)
        tl.debug_barrier()


def raw_linkage_gpu(affinity, lengths):
    """Return unsorted raw merge records on-device; input is already validated."""
    if affinity.ndim == 3 and affinity.shape[1] > 1024:
        # Joint image/caption grouping reaches 1,280 vertices; same FP64 algorithm.
        from .tree import raw_linkage_gpu as wide_linkage_gpu
        return wide_linkage_gpu(affinity, lengths.contiguous(), precision='fp64')
    if not affinity.is_cuda or affinity.dtype != torch.float32 or not affinity.is_contiguous():
        raise ValueError('Expected contiguous CUDA FP32 affinity')
    batch, n, other = affinity.shape
    if n != other or n < 1 or n > 1024:
        raise ValueError('Expected square affinity with 1 <= N <= 1024')
    if lengths.shape != (batch,) or lengths.device != affinity.device:
        raise ValueError('Invalid length layout/device')
    raw = torch.zeros(batch, n - 1, 4, dtype=torch.float64, device=affinity.device)
    if n == 1:
        return raw
    distances = torch.empty_like(affinity, dtype=torch.float64)
    _init_distances[(triton.cdiv(affinity.numel(), 1024),)](
        affinity, distances, affinity.numel(), 1024)
    _nn_chain[(batch,)](distances, lengths, raw, n, triton.next_power_of_2(n),
                       num_warps=4, enable_fp_fusion=False)
    return raw


@njit(cache=True, nogil=True, parallel=True)
def finalize_linkage(raw, lengths):
    """Original stable height ordering and union-find labels, now O(B*N) input."""
    out = np.zeros_like(raw)
    for image in prange(len(lengths)):
        n = lengths[image]
        if n <= 1:
            continue
        result = raw[image, :n - 1].copy()
        order = np.argsort(result[:, 2], kind='mergesort')
        result = result[order]
        parent = np.arange(2 * n - 1, dtype=np.int64)
        counts = np.ones(2 * n - 1, dtype=np.int64)
        for step in range(n - 1):
            x, y = int(result[step, 0]), int(result[step, 1])
            while parent[x] != x:
                x = parent[x]
            while parent[y] != y:
                y = parent[y]
            result[step, 0], result[step, 1] = min(x, y), max(x, y)
            counts[n + step] = counts[x] + counts[y]
            result[step, 3] = counts[n + step]
            parent[x], parent[y] = n + step, n + step
        out[image, :n - 1] = result
    return out


def structure_gpu_exact(router, q, k, lengths=None):
    """Optional router backend; reader/replay/reconstruction remain unchanged."""
    if getattr(router, 'tree_precision', 'fp64') == 'fp32':
        # FP32 trees keep their FP32 heights/threshold comparison on the GPU.
        from .metadata import structure
        return structure(router, q, k, lengths)
    from numba import set_num_threads
    from ..grouping import Structure, batch_metadata, batch_metadata_varlen
    if not q.is_cuda:
        raise ValueError('gpu_exact requires CUDA inputs')
    started = time.perf_counter()
    batch, n, _ = q.shape
    variable = lengths is not None
    lengths_np = np.asarray(lengths if variable else [n]*batch, dtype=np.int64)
    if lengths_np.shape != (batch,) or lengths_np.min() < 1 or lengths_np.max() > n:
        raise ValueError('Invalid stage-2 input lengths')
    with torch.no_grad(), torch.autocast(q.device.type, enabled=False):
        # Preserve the existing FP32 affinity calculation and its GEMM policy.
        product = q.detach() @ k.detach().transpose(-1, -2)
        affinity = .5 * (product + product.transpose(-1, -2))
        if router._host_bounds is None:
            router._host_bounds = torch.empty(2, dtype=torch.float32, pin_memory=True)
        router._host_bounds.copy_(torch.stack((affinity.amin(), affinity.amax())), non_blocking=True)
        torch.cuda.current_stream(q.device).synchronize()
        lower, upper = router._host_bounds.tolist()
        if not math.isfinite(lower) or not math.isfinite(upper) or upper > 1 + 2e-6:
            raise ValueError('Invalid normalized affinity')
        validated = time.perf_counter()
        lens = torch.from_numpy(lengths_np).to(q.device)
        raw = raw_linkage_gpu(affinity, lens)
        if getattr(router, '_host_raw_tree', None) is None or router._host_raw_tree.numel() < raw.numel():
            router._host_raw_tree = torch.empty(raw.numel(), dtype=torch.float64, pin_memory=True)
        host = router._host_raw_tree[:raw.numel()].view(raw.shape)
        host.copy_(raw, non_blocking=True)
        torch.cuda.current_stream(q.device).synchronize()
    copied = time.perf_counter()
    set_num_threads(max(1, router.workers))
    trees = finalize_linkage(host.numpy(), lengths_np)
    linked = time.perf_counter()
    threshold = float(router.threshold.item())
    if variable:
        values = batch_metadata_varlen(trees, lengths_np, threshold, n)
    else:
        values = batch_metadata(trees, threshold, n)
    clustered = time.perf_counter()
    groups, counts, order, intervals, scores = values
    length = min(n, math.ceil(max(counts) / router.pad_multiple) * router.pad_multiple)
    dtypes = [torch.int64]*4 + [torch.float64]
    if router.transfer == 'packed':
        arrays = [np.asarray(v) for v in values]
        elements = sum(v.size for v in arrays)
        if router._host_metadata is None or router._host_metadata.numel() < elements:
            router._host_metadata = torch.empty(elements, dtype=torch.int64, pin_memory=True)
        cursor = 0
        for array, dtype in zip(arrays, dtypes):
            np.copyto(router._host_metadata[cursor:cursor+array.size].view(dtype).numpy(), array.reshape(-1))
            cursor += array.size
        device = router._host_metadata[:elements].to(q.device, non_blocking=True)
        tensors, cursor = [], 0
        for array, dtype in zip(arrays, dtypes):
            tensors.append(device[cursor:cursor+array.size].view(dtype).reshape(array.shape))
            cursor += array.size
    else:
        tensors = [torch.from_numpy(np.asarray(v)).to(device=q.device, dtype=d) for v,d in zip(values,dtypes)]
    ended = time.perf_counter()
    phases = dict(affinity_and_wait=validated-started,
                  gpu_linkage_and_transfer=copied-validated,
                  cpu_relabel=linked-copied, cpu_metadata=clustered-linked,
                  cpu_tree=clustered-copied, metadata=ended-clustered)
    return Structure(*tensors, tuple(int(v) for v in counts), length, ended-started, phases)
