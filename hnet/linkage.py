"""Parallel exact average-linkage nearest-neighbor chains.

Adapted from SciPy 1.14.1's nn_chain and linkage relabeling, BSD-3-Clause.
Copyright (c) 2001-2002 Enthought, Inc. 2003-2024 SciPy Developers.
See LICENSES/scipy-bsd.txt. Tie preferences, stable sort, and weighted updates
match that pinned reference. Each image runs independently outside the GIL.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def average_linkage(similarity):
    n = similarity.shape[0]
    distances = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            distances[i, j] = max(0., 1. - np.float64(similarity[i, j]))
    size = np.ones(n, dtype=np.int64)
    chain = np.empty(n, dtype=np.int64)
    chain_length = 0
    result = np.empty((n - 1, 4), dtype=np.float64)
    for step in range(n - 1):
        if chain_length == 0:
            for i in range(n):
                if size[i] > 0:
                    chain[0] = i
                    break
            chain_length = 1
        y = 0
        while True:
            x = chain[chain_length - 1]
            nearest = np.inf
            if chain_length > 1:
                y = chain[chain_length - 2]
                nearest = distances[x, y]
            for i in range(n):
                if size[i] and i != x and distances[x, i] < nearest:
                    nearest = distances[x, i]
                    y = i
            if chain_length > 1 and y == chain[chain_length - 2]:
                break
            chain[chain_length] = y
            chain_length += 1
        chain_length -= 2
        if x > y:
            x, y = y, x
        nx, ny = size[x], size[y]
        result[step, 0] = x
        result[step, 1] = y
        result[step, 2] = nearest
        result[step, 3] = nx + ny
        size[x], size[y] = 0, nx + ny
        for i in range(n):
            if size[i] and i != y:
                updated = (nx * distances[i, x] + ny * distances[i, y]) / (nx + ny)
                distances[i, y] = updated
                distances[y, i] = updated
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
    return result


@njit(cache=True, nogil=True, parallel=True)
def batched_linkage(similarities):
    b, n, _ = similarities.shape
    result = np.empty((b, n - 1, 4), dtype=np.float64)
    for i in prange(b):
        result[i] = average_linkage(similarities[i])
    return result


@njit(cache=True, nogil=True, parallel=True)
def batched_linkage_varlen(similarities, lengths):
    """Exact average linkage on each image's valid prefix only (stage-2 routing).

    ``similarities`` is padded to (b, n, n); image i uses its leading lengths[i] x lengths[i] block.
    Padding is never a graph vertex. lengths[i] == 1 gives the empty merge tree."""
    b, n, _ = similarities.shape
    result = np.zeros((b, max(n - 1, 0), 4), dtype=np.float64)
    for i in prange(b):
        m = lengths[i]
        if m > 1:
            result[i, :m - 1] = average_linkage(np.ascontiguousarray(similarities[i, :m, :m]))
    return result
