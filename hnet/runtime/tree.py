"""GPU nearest-neighbor chains for the original complete average-linkage tree.

The input affinity arithmetic, FP64 distance updates, tie rule and stable CPU
relabeling follow linkage.py. No approximate search, early cut or window graph.
The opt-in FP32 variant (text-to-image recipe) keeps the same full algorithm and
tie rule, but may choose different merges near numerical ties.
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
from functools import lru_cache


AFFINITY_MAX_FP32 = float(np.float32(1.0 + 2e-6))


@lru_cache(maxsize=None)
def normalized_affinity_limit(width):
    """Conservative FP32 normalization/dot-product roundoff guard, not a clamp.

    For unit roundoff u and gamma_d=d*u/(1-d*u), each FP32-normalized
    vector has norm at most (1+u)/((1-u)*sqrt(1-gamma_d)). Include the
    dot-product error and the final symmetric sum, then round the bound up.
    Actual affinity values and the pre-existing distance calculation are intact.
    This assumes FP32 GEMMs (TF32 disabled), as required by this backend.
    """
    if not isinstance(width, int) or not 1 <= width <= 16384:
        raise ValueError('Invalid FP32 routing width')
    u = 2.0 ** -24
    gamma = width * u / (1.0 - width * u)
    bound = (1.0 + gamma) * (1.0 + u) ** 3 / ((1.0 - gamma) * (1.0 - u) ** 2)
    rounded = np.float32(bound)
    if float(rounded) < bound:
        rounded = np.nextafter(rounded, np.float32(np.inf))
    return max(AFFINITY_MAX_FP32, float(rounded))


@triton.jit
def _init_distances(A, D, TOTAL: tl.constexpr, BLOCK: tl.constexpr, FP32: tl.constexpr):
    ix = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    a = tl.load(A + ix, mask=ix < TOTAL, other=0.).to(dtype)
    tl.store(D + ix, tl.maximum(1. - a, 0.), mask=ix < TOTAL)


@triton.jit
def _nn_chain_rows(D, LENGTH, RAW, BOUNDS, N: tl.constexpr, NP: tl.constexpr,
                   AFFINITY_LIMIT: tl.constexpr, FP32: tl.constexpr):
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    b = tl.program_id(0).to(tl.int64)
    n = tl.load(LENGTH + b)
    ix = tl.arange(0, NP)
    size = (ix < n).to(tl.int32)
    base = b * N * N
    chain = tl.zeros((NP,), tl.int32)
    chain_length = tl.full((), 0, tl.int32)
    lower = tl.load(BOUNDS)
    upper = tl.load(BOUNDS + 1)
    valid = (lower > -float('inf')) & (upper < float('inf')) & (upper <= AFFINITY_LIMIT)
    for step in range(tl.where(valid, n - 1, 0)):
        if chain_length == 0:
            first = tl.min(tl.where(size > 0, ix, 2147483647), 0)
            chain = tl.where(ix == 0, first, chain)
            chain_length = tl.full((), 1, tl.int32)
        searching = tl.full((), 1, tl.int32)
        x = tl.full((), 0, tl.int32)
        y = tl.full((), 0, tl.int32)
        nearest = tl.full((), float('inf'), dtype)
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
        tl.store(RAW + at, lo.to(dtype))
        tl.store(RAW + at + 1, hi.to(dtype))
        tl.store(RAW + at + 2, nearest)
        tl.store(RAW + at + 3, (nx + ny).to(dtype))
        size = tl.where(ix == lo, 0, tl.where(ix == hi, nx + ny, size))
        active = (size > 0) & (ix != hi)
        dx = tl.load(D + base + lo * N + ix, mask=active, other=0.)
        dy = tl.load(D + base + hi * N + ix, mask=active, other=0.)
        # Match separate CPU mul/add and correctly rounded division in the
        # selected precision; never an approximate FP32 reciprocal.
        numerator = nx.to(dtype) * dx + ny.to(dtype) * dy
        if FP32:
            updated = tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;',
                constraints='=f,f,f', args=[numerator, (nx + ny).to(dtype)],
                dtype=tl.float32, is_pure=True, pack=1)
        else:
            updated = tl.inline_asm_elementwise('div.rn.f64 $0, $1, $2;',
                constraints='=d,d,d', args=[numerator, (nx + ny).to(dtype)],
                dtype=tl.float64, is_pure=True, pack=1)
        tl.debug_barrier()
        tl.store(D + base + ix * N + hi, updated, mask=active)
        tl.store(D + base + hi * N + ix, updated, mask=active)
        tl.debug_barrier()


MAX_VERTICES = 1280  # 1,024 image patches plus 256 caption positions at 512px /16.


def raw_linkage_gpu(affinity, lengths, warps=16, bounds=None, affinity_limit=AFFINITY_MAX_FP32,
                    precision='fp64'):
    """Return unsorted raw merge records on-device; input is already validated."""
    if not affinity.is_cuda or affinity.dtype != torch.float32 or not affinity.is_contiguous():
        raise ValueError('Expected contiguous CUDA FP32 affinity')
    if precision not in ('fp32', 'fp64'):
        raise ValueError('Tree precision must be fp32 or fp64')
    batch, n, other = affinity.shape
    if n != other or n < 1 or n > MAX_VERTICES:
        raise ValueError(f'Expected square affinity with 1 <= N <= {MAX_VERTICES}')
    if lengths.shape != (batch,) or lengths.device != affinity.device:
        raise ValueError('Invalid length layout/device')
    dtype = torch.float32 if precision == 'fp32' else torch.float64
    raw = torch.zeros(batch, n - 1, 4, dtype=dtype, device=affinity.device)
    if n == 1:
        return raw
    if bounds is None:
        bounds = torch.stack((affinity.amin(), affinity.amax()))
    distances = torch.empty_like(affinity, dtype=dtype)
    _init_distances[(triton.cdiv(affinity.numel(), 1024),)](
        affinity, distances, affinity.numel(), 1024, precision == 'fp32')
    _nn_chain_rows[(batch,)](distances, lengths, raw, bounds, n, triton.next_power_of_2(n), affinity_limit,
                       precision == 'fp32', num_warps=warps, enable_fp_fusion=False)
    return raw
