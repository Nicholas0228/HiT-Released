"""Exact detached average linkage and differentiable soft pooling/readback.

SciPy's pinned nearest-neighbor-chain implementation supplies the full exact
tree. The reader and merge replay are algebraically factorized through live Q/K
means; no task gradient is claimed through the discrete merge decisions.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.cluster.hierarchy import linkage
from scipy.cluster import _hierarchy
import torch
from torch import nn
from torch.nn import functional as F
from .linkage import batched_linkage, batched_linkage_varlen


@njit(cache=True)
def tree_metadata(children, scores, threshold, n):
    """Return deterministic groups and contiguous DFS intervals for every node."""
    parent = np.arange(2 * n - 1, dtype=np.int64)
    accepted = 0
    for m in range(n - 1):
        if scores[m] >= threshold:
            parent[children[m, 0]] = n + m
            parent[children[m, 1]] = n + m
            accepted += 1
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        root = i
        while parent[root] != root:
            root = parent[root]
        roots[i] = root
    # Label groups in first-leaf order; labels carry no positional meaning.
    lookup = np.full(2 * n - 1, -1, dtype=np.int64)
    groups = np.empty(n, dtype=np.int64)
    ngroups = 0
    for i in range(n):
        if lookup[roots[i]] < 0:
            lookup[roots[i]] = ngroups
            ngroups += 1
        groups[i] = lookup[roots[i]]
    assert ngroups == n - accepted
    order = np.empty(n, dtype=np.int64)
    start = np.zeros(2 * n - 1, dtype=np.int64)
    end = np.zeros(2 * n - 1, dtype=np.int64)
    stack = np.empty(2 * n, dtype=np.int64)
    stack[0] = 2 * n - 2
    top, cursor = 1, 0
    while top:
        top -= 1
        node = stack[top]
        if node < n:
            start[node], end[node] = cursor, cursor + 1
            order[cursor] = node
            cursor += 1
        else:
            stack[top] = children[node - n, 1]
            stack[top + 1] = children[node - n, 0]
            top += 2
    for m in range(n - 1):
        left, right = children[m, 0], children[m, 1]
        start[n + m] = min(start[left], start[right])
        end[n + m] = max(end[left], end[right])
    intervals = np.empty((n - 1, 4), dtype=np.int64)
    for m in range(n - 1):
        left, right = children[m, 0], children[m, 1]
        intervals[m] = (start[left], end[left], start[right], end[right])
    return groups, ngroups, order, intervals


@lru_cache(maxsize=8)
def upper_indices(n):
    return np.triu_indices(n, 1)


def exact_tree(similarity, threshold):
    """One CPU float32 affinity matrix -> complete SciPy average-linkage tree."""
    n = len(similarity)
    if n == 1:
        z = np.empty((0, 4), dtype=np.float64)
    else:
        upper = upper_indices(n)
        distances = 1.0 - similarity[upper].astype(np.float64)
        # Normalized-dot roundoff can exceed 1 by a few FP32 ulps.
        if distances.min() < -2e-6 or not np.isfinite(distances).all():
            raise ValueError('Invalid normalized affinity')
        # This is the exact kernel used by scipy.linkage(method='average').
        # Pin SciPy and test against its public wrapper; skip repeated array-API
        # dispatch and validation after the explicit checks above.
        z = _hierarchy.nn_chain(np.maximum(distances, 0), n, 2)
    scores = 1.0 - z[:, 2]
    children = z[:, :2].astype(np.int64)
    groups, count, order, intervals = tree_metadata(children, scores, threshold, n)
    return groups, count, order, intervals, scores, children


@njit(cache=True, nogil=True, parallel=True)
def batch_metadata(trees, threshold, n):
    b = len(trees)
    groups = np.empty((b, n), dtype=np.int64)
    counts = np.empty(b, dtype=np.int64)
    order = np.empty((b, n), dtype=np.int64)
    intervals = np.empty((b, n - 1, 4), dtype=np.int64)
    scores = 1. - trees[:, :, 2]
    for i in prange(b):
        groups[i], counts[i], order[i], intervals[i] = tree_metadata(trees[i, :, :2].astype(np.int64), scores[i], threshold, n)
    return groups, counts, order, intervals, scores


@njit(cache=True, nogil=True, parallel=True)
def batch_metadata_varlen(trees, lengths, threshold, n):
    """tree_metadata on each image's valid prefix; padded entries are inert placeholders.

    Padded token slots get group 0 / order 0 and are masked by the reader; padded merges get
    the dummy interval (0, 1, 0, 1) (finite means) and are masked out of the budget."""
    b = len(trees)
    groups = np.zeros((b, n), dtype=np.int64)
    counts = np.empty(b, dtype=np.int64)
    order = np.zeros((b, n), dtype=np.int64)
    intervals = np.zeros((b, max(n - 1, 0), 4), dtype=np.int64)
    scores = np.zeros((b, max(n - 1, 0)), dtype=np.float64)
    for i in prange(b):
        m = lengths[i]
        sc = 1. - trees[i, :m - 1, 2]
        g, c, o, iv = tree_metadata(trees[i, :m - 1, :2].astype(np.int64), sc, threshold, m)
        groups[i, :m] = g
        counts[i] = c
        order[i, :m] = o
        intervals[i, :m - 1] = iv
        scores[i, :m - 1] = sc
        for r in range(m - 1, n - 1):
            intervals[i, r, 1] = 1
            intervals[i, r, 3] = 1
    return groups, counts, order, intervals, scores


@dataclass
class Structure:
    groups: torch.Tensor
    counts: torch.Tensor
    order: torch.Tensor
    intervals: torch.Tensor
    scores: torch.Tensor
    lengths: tuple
    padded_length: int
    tree_seconds: float
    phases: dict = None


def node_means(vectors, order, intervals):
    """Live subtree means via FP32 prefix sums in a detached DFS leaf order."""
    batch, _, width = vectors.shape
    ordered = vectors.gather(1, order[..., None].expand(-1, -1, width))
    prefix = F.pad(ordered.cumsum(1), (0, 0, 1, 0))
    def mean(start, end):
        lo = prefix.gather(1, start[..., None].expand(-1, -1, width))
        hi = prefix.gather(1, end[..., None].expand(-1, -1, width))
        return (hi - lo) / (end - start)[..., None]
    return mean(intervals[..., 0], intervals[..., 1]), mean(intervals[..., 2], intervals[..., 3])


def replay_scores(q, k, structure):
    ql, qr = node_means(q, structure.order, structure.intervals)
    kl, kr = node_means(k, structure.order, structure.intervals)
    return 0.5 * ((ql * kr).sum(-1) + (kl * qr).sum(-1))


def budget_loss_population(hard_counts, soft_counts, population, ratio, distributed=True):
    """Stage-2 feedback: identical formula to budget_loss, with the global population
    sum(actual K1) (detached, all-reduced) replacing batch x n. Never a padded maximum."""
    if ratio <= 1:
        raise ValueError('The ratio feedback requires ratio > 1')
    totals = torch.stack((hard_counts.float().sum(), soft_counts.detach().sum(),
                          population.detach().float().sum()))
    world = 1
    if distributed and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(totals)
    fraction = totals[0] / totals[2]
    value = totals[1] / totals[2]
    local = soft_counts.sum() * (world / totals[2])
    soft_fraction = value + local - local.detach()
    loss = ratio / (ratio - 1) * ((ratio - 1) * fraction * soft_fraction
                                 + (1 - fraction) * (1 - soft_fraction))
    return loss, fraction, value


def budget_loss(hard_counts, soft_counts, n, ratio, distributed=True):
    """Global-microbatch feedback with correct DDP-averaged local gradients.

Accumulation averages these global-microbatch objectives. It is intentionally
not mislabeled as one feedback product over the entire accumulated batch.
"""
    if ratio <= 1:
        raise ValueError('The ratio feedback requires ratio > 1')
    totals = torch.stack((hard_counts.float().sum(), soft_counts.detach().sum(),
                          soft_counts.new_tensor(soft_counts.numel() * n)))
    world = 1
    if distributed and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(totals)
    fraction = totals[0] / totals[2]
    value = totals[1] / totals[2]
    local = soft_counts.sum() * (world / totals[2])
    soft_fraction = value + local - local.detach()
    loss = ratio / (ratio - 1) * ((ratio - 1) * fraction * soft_fraction
                                 + (1 - fraction) * (1 - soft_fraction))
    return loss, fraction, value


class SoftGrouping(nn.Module):
    def __init__(self, width=768, routing_width=None, threshold=0.5,
                 reader_temperature=0.1, budget_temperature=0.1,
                 workers=4, pad_multiple=16, init_seed=1701, backend='numba', transfer='legacy',
                 validation='cpu'):
        super().__init__()
        routing_width = routing_width or width
        if routing_width != width:
            raise ValueError('The initial approved router uses the full feature width.')
        self.q_proj = nn.Linear(width, routing_width, bias=False)
        self.k_proj = nn.Linear(width, routing_width, bias=False)
        # Recorded tiny independent perturbations avoid accidental permanent Q=K.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            with torch.no_grad():
                for module in (self.q_proj, self.k_proj):
                    module.weight.copy_(torch.eye(width) + torch.randn(width, width) * (1e-3 / math.sqrt(width)))
        self.register_buffer('threshold', torch.tensor(float(threshold), dtype=torch.float64))
        self.reader_temperature = float(reader_temperature)
        self.budget_temperature = float(budget_temperature)
        if self.reader_temperature <= 0 or self.budget_temperature <= 0:
            raise ValueError('Temperatures must be positive')
        self.pad_multiple = pad_multiple
        self.workers = workers
        self.backend = backend
        if transfer not in ('legacy', 'packed'):
            raise ValueError(transfer)
        self.transfer = transfer
        if validation not in ('cpu', 'gpu'):
            raise ValueError(validation)
        self.validation = validation
        self._pool = None
        self._host_affinity = None
        self._host_metadata = None
        self._host_bounds = None
        self.last_statistics = {}

    @torch.compiler.disable
    def structure(self, q, k):
        if self.backend == 'gpu_exact':
            from .runtime.gpu_linkage import structure_gpu_exact
            return structure_gpu_exact(self, q, k)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(q.device.type, enabled=False):
            product = q.detach() @ k.detach().transpose(-1, -2)
            affinity = 0.5 * (product + product.transpose(-1, -2))
            if affinity.is_cuda:
                # Pageable D2H staging dominated the original runtime. This
                # reusable pinned buffer is read only after its stream finishes;
                # the tree finishes before the next forward reuses the buffer.
                if self._host_affinity is None or self._host_affinity.shape != affinity.shape:
                    self._host_affinity = torch.empty_like(affinity, device='cpu', pin_memory=True)
                if self.validation == 'gpu':
                    if self._host_bounds is None:
                        self._host_bounds = torch.empty(2, dtype=torch.float32, pin_memory=True)
                    self._host_bounds.copy_(torch.stack((affinity.amin(), affinity.amax())), non_blocking=True)
                self._host_affinity.copy_(affinity, non_blocking=True)
                torch.cuda.current_stream(q.device).synchronize()
                similarities = self._host_affinity.numpy()
            else:
                similarities = affinity.numpy()
        copied = time.perf_counter()
        threshold = float(self.threshold.item())
        n = q.shape[1]
        if self.backend == 'numba':
            if self.validation == 'gpu' and q.is_cuda:
                lower, upper = self._host_bounds.tolist()
                invalid = not math.isfinite(lower) or not math.isfinite(upper) or upper > 1 + 2e-6
            else:
                invalid = not np.isfinite(similarities).all() or similarities.max() > 1 + 2e-6
            if invalid:
                raise ValueError('Invalid normalized affinity')
            validated = time.perf_counter()
            set_num_threads(max(1, self.workers))
            trees = batched_linkage(similarities)
            linked = time.perf_counter()
            groups, counts, order, intervals, scores = batch_metadata(trees, threshold, n)
        else:
            if self._pool is None and self.workers > 1:
                self._pool = ThreadPoolExecutor(max_workers=self.workers)
            if self._pool is None:
                results = [exact_tree(s, threshold) for s in similarities]
            else:
                results = list(self._pool.map(lambda s: exact_tree(s, threshold), similarities))
            groups, counts, order, intervals, scores, _ = zip(*results)
        clustered = time.perf_counter()
        length = min(n, math.ceil(max(counts) / self.pad_multiple) * self.pad_multiple)
        values = [groups, counts, order, intervals, scores]
        dtypes = [torch.int64] * 4 + [torch.float64]
        if self.transfer == 'packed' and q.is_cuda:
            # One pinned transfer replaces five pageable transfers/fences.
            arrays = [np.asarray(value) for value in values]
            elements = sum(array.size for array in arrays)
            if self._host_metadata is None or self._host_metadata.numel() != elements:
                self._host_metadata = torch.empty(elements, dtype=torch.int64, pin_memory=True)
            cursor = 0
            for array, dtype in zip(arrays, dtypes):
                target = self._host_metadata[cursor:cursor + array.size].view(dtype).numpy()
                np.copyto(target, array.reshape(-1))
                cursor += array.size
            # A fresh device allocation keeps earlier forward graphs valid.
            device_metadata = self._host_metadata.to(q.device, non_blocking=True)
            tensors, cursor = [], 0
            for array, dtype in zip(arrays, dtypes):
                tensors.append(device_metadata[cursor:cursor + array.size].view(dtype).reshape(array.shape))
                cursor += array.size
        else:
            tensors = [torch.from_numpy(np.asarray(value)).to(device=q.device, dtype=dtype)
                       for value, dtype in zip(values, dtypes)]
        ended = time.perf_counter()
        phases = dict(affinity_and_wait=copied-started, cpu_tree=clustered-copied,
                      metadata=ended-clustered)
        if self.backend == 'numba':
            phases.update(cpu_validation=validated-copied, cpu_linkage=linked-validated,
                          cpu_metadata=clustered-linked)
        return Structure(*tensors, tuple(int(c) for c in counts), length, ended-started, phases)

    def project(self, features):
        with torch.autocast(features.device.type, enabled=False):
            h = features.float()
            return h, F.normalize(self.q_proj(h), dim=-1, eps=1e-6), F.normalize(self.k_proj(h), dim=-1, eps=1e-6)

    def read(self, h, q, k, groups, counts, order, intervals, scores, length, need_budget):
        with torch.autocast(h.device.type, enabled=False):
            batch, n, width = q.shape
            idx = groups[..., None].expand(-1, -1, width)
            sizes = torch.zeros(batch, length, device=h.device).scatter_add_(
                1, groups, torch.ones(batch, n, device=h.device))
            qm = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, q)
            km = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, k)
            qm = qm / sizes.clamp_min(1)[..., None]
            km = km / sizes.clamp_min(1)[..., None]
            logits = 0.5 * (q @ km.transpose(-1, -2) + k @ qm.transpose(-1, -2))
            valid = torch.arange(length, device=h.device)[None, :] < counts[:, None]
            assignment = (logits / self.reader_temperature).masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            mass = assignment.sum(1)
            compact = assignment.transpose(1, 2) @ h
            compact = compact / (mass[..., None] + 1e-8)
            if need_budget:
                ql, qr = node_means(q, order, intervals)
                kl, kr = node_means(k, order, intervals)
                merge_scores = 0.5 * ((ql * kr).sum(-1) + (kl * qr).sum(-1))
                confidence = ((merge_scores - self.threshold.float()) / self.budget_temperature).sigmoid()
                soft_counts = n - confidence.sum(1)
            else:
                merge_scores = confidence = soft_counts = None
            # Detached inexpensive scalars can be reduced/logged at the trainer's cadence.
            entropy = -(assignment.clamp_min(1e-20).log() * assignment).sum(-1).mean(-1)
            stats = {'hard_counts': counts.detach(), 'assignment_entropy': entropy.detach(),
                     'min_group_mass': mass.masked_fill(~valid, torch.inf).min(-1).values.detach(),
                     'max_assignment': assignment.max(-1).values.mean(-1).detach(),
                     'compact_variance': ((compact - (assignment.sum(1)[..., None] * compact).sum(1, keepdim=True) / n)**2 * valid[..., None]).sum((1, 2)).detach() / (counts * width),
                     'processed_length': length}
            if need_budget:
                stats.update(soft_counts=soft_counts.detach(),
                             unsaturated_merges=((confidence > 0.05) & (confidence < 0.95)).float().mean(-1).detach() if n > 1 else torch.zeros(batch, device=h.device),
                             replay_max_error=(merge_scores.detach().double() - scores).abs().max().detach() if n > 1 else h.new_zeros(()))
        return compact, assignment, valid, soft_counts, stats

    def forward(self, features, need_budget=True, fixed_structure=None):
        h, q, k = self.project(features)
        structure = fixed_structure or self.structure(q, k)
        compact, assignment, valid, soft_counts, stats = self.read(
            h, q, k, structure.groups, structure.counts, structure.order,
            structure.intervals, structure.scores, structure.padded_length, need_budget)
        self.last_statistics = dict(stats, tree_seconds=structure.tree_seconds, structure_phases=structure.phases)
        return compact.to(features.dtype), assignment, valid, structure, soft_counts

    @staticmethod
    def reconstruct(assignment, compact):
        with torch.autocast(compact.device.type, enabled=False):
            return assignment.float() @ compact.float()


class VarLenGrouping(SoftGrouping):
    """Second router: the same exact average linkage, soft reader and budget surrogate, applied to
    each image's actual valid K1 stage-1 tokens. Padding is never a vertex, prototype, budget
    denominator or reader row. Q/K initialization uses its own recorded seed (1702)."""

    @torch.compiler.disable
    def structure(self, q, k, lengths):
        if self.backend == 'gpu_exact':
            from .runtime.gpu_linkage import structure_gpu_exact
            return structure_gpu_exact(self, q, k, lengths)
        started = time.perf_counter()
        batch, n, _ = q.shape
        lengths_np = np.asarray(lengths, dtype=np.int64)
        if len(lengths_np) != batch or lengths_np.min() < 1 or lengths_np.max() > n:
            raise ValueError('Invalid stage-2 input lengths')
        with torch.no_grad(), torch.autocast(q.device.type, enabled=False):
            product = q.detach() @ k.detach().transpose(-1, -2)
            affinity = 0.5 * (product + product.transpose(-1, -2))
            if affinity.is_cuda:
                # One pinned buffer of the largest size; padded stage-2 shapes vary per step.
                if self._host_affinity is None or self._host_affinity.numel() < affinity.numel():
                    self._host_affinity = torch.empty(affinity.numel(), dtype=torch.float32, pin_memory=True)
                host = self._host_affinity[:affinity.numel()].view(affinity.shape)
                if self.validation == 'gpu':
                    if self._host_bounds is None:
                        self._host_bounds = torch.empty(2, dtype=torch.float32, pin_memory=True)
                    self._host_bounds.copy_(torch.stack((affinity.amin(), affinity.amax())), non_blocking=True)
                host.copy_(affinity, non_blocking=True)
                torch.cuda.current_stream(q.device).synchronize()
                similarities = host.numpy()
            else:
                similarities = affinity.numpy()
        copied = time.perf_counter()
        threshold = float(self.threshold.item())
        # Padded rows/columns are exact zeros (zero features -> zero normalized Q/K), so the
        # bound check over the padded block equals the check over valid entries for max/finiteness.
        if self.validation == 'gpu' and q.is_cuda:
            lower, upper = self._host_bounds.tolist()
            invalid = not math.isfinite(lower) or not math.isfinite(upper) or upper > 1 + 2e-6
        else:
            invalid = not np.isfinite(similarities).all() or similarities.max() > 1 + 2e-6
        if invalid:
            raise ValueError('Invalid normalized affinity')
        validated = time.perf_counter()
        set_num_threads(max(1, self.workers))
        trees = batched_linkage_varlen(similarities, lengths_np)
        linked = time.perf_counter()
        groups, counts, order, intervals, scores = batch_metadata_varlen(trees, lengths_np, threshold, n)
        clustered = time.perf_counter()
        length = min(n, math.ceil(max(counts) / self.pad_multiple) * self.pad_multiple)
        values = [groups, counts, order, intervals, scores]
        dtypes = [torch.int64] * 4 + [torch.float64]
        if self.transfer == 'packed' and q.is_cuda:
            arrays = [np.asarray(value) for value in values]
            elements = sum(array.size for array in arrays)
            if self._host_metadata is None or self._host_metadata.numel() < elements:
                self._host_metadata = torch.empty(elements, dtype=torch.int64, pin_memory=True)
            cursor = 0
            for array, dtype in zip(arrays, dtypes):
                target = self._host_metadata[cursor:cursor + array.size].view(dtype).numpy()
                np.copyto(target, array.reshape(-1))
                cursor += array.size
            device_metadata = self._host_metadata[:elements].to(q.device, non_blocking=True)
            tensors, cursor = [], 0
            for array, dtype in zip(arrays, dtypes):
                tensors.append(device_metadata[cursor:cursor + array.size].view(dtype).reshape(array.shape))
                cursor += array.size
        else:
            tensors = [torch.from_numpy(np.asarray(value)).to(device=q.device, dtype=dtype)
                       for value, dtype in zip(values, dtypes)]
        ended = time.perf_counter()
        phases = dict(affinity_and_wait=copied-started, cpu_tree=clustered-copied, metadata=ended-clustered,
                      cpu_validation=validated-copied, cpu_linkage=linked-validated, cpu_metadata=clustered-linked)
        return Structure(*tensors, tuple(int(c) for c in counts), length, ended-started, phases)

    def read(self, h, q, k, valid_in, n_in, groups, counts, order, intervals, scores, length, need_budget):
        with torch.autocast(h.device.type, enabled=False):
            batch, n, width = q.shape
            vin = valid_in.to(q.dtype)
            idx = groups[..., None].expand(-1, -1, width)
            sizes = torch.zeros(batch, length, device=h.device).scatter_add_(1, groups, vin)
            qm = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, q * vin[..., None])
            km = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, k * vin[..., None])
            qm = qm / sizes.clamp_min(1)[..., None]
            km = km / sizes.clamp_min(1)[..., None]
            logits = 0.5 * (q @ km.transpose(-1, -2) + k @ qm.transpose(-1, -2))
            valid = torch.arange(length, device=h.device)[None, :] < counts[:, None]
            assignment = (logits / self.reader_temperature).masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            assignment = assignment * vin[..., None]              # padded rows read/write nothing
            mass = assignment.sum(1)
            compact = assignment.transpose(1, 2) @ h
            compact = compact / (mass[..., None] + 1e-8)
            n_float = n_in.to(q.dtype)
            if need_budget and n > 1:
                ql, qr = node_means(q, order, intervals)
                kl, kr = node_means(k, order, intervals)
                merge_scores = 0.5 * ((ql * kr).sum(-1) + (kl * qr).sum(-1))
                merge_valid = torch.arange(n - 1, device=h.device)[None, :] < (n_in - 1)[:, None]
                confidence = ((merge_scores - self.threshold.float()) / self.budget_temperature).sigmoid() * merge_valid
                soft_counts = n_float - confidence.sum(1)
            elif need_budget:
                merge_valid = confidence = merge_scores = None
                soft_counts = n_float + 0. * q.sum((1, 2))
            else:
                merge_scores = confidence = soft_counts = merge_valid = None
            token_entropy = -(assignment.clamp_min(1e-20).log() * assignment).sum(-1)
            entropy = (token_entropy * vin).sum(-1) / n_float
            mean_compact = (assignment.sum(1)[..., None] * compact).sum(1, keepdim=True) / n_float[:, None, None]
            stats = {'hard_counts': counts.detach(), 'assignment_entropy': entropy.detach(),
                     'min_group_mass': mass.masked_fill(~valid, torch.inf).min(-1).values.detach(),
                     'max_assignment': ((assignment.max(-1).values * vin).sum(-1) / n_float).detach(),
                     'compact_variance': ((compact - mean_compact)**2 * valid[..., None]).sum((1, 2)).detach() / (counts * width),
                     'processed_length': length, 'input_lengths': n_in.detach()}
            if need_budget:
                if merge_valid is not None:
                    unsat = (((confidence > 0.05) & (confidence < 0.95)) & merge_valid).float().sum(-1) / (n_float - 1).clamp_min(1)
                    replay = ((merge_scores.detach().double() - scores).abs() * merge_valid).max().detach()
                else:
                    unsat = torch.zeros(batch, device=h.device); replay = h.new_zeros(())
                stats.update(soft_counts=soft_counts.detach(), unsaturated_merges=unsat.detach(), replay_max_error=replay)
        return compact, assignment, valid, soft_counts, stats

    def forward(self, features, valid_in, n_in, lengths, need_budget=True, fixed_structure=None):
        h, q, k = self.project(features)
        structure = fixed_structure or self.structure(q, k, lengths)
        compact, assignment, valid, soft_counts, stats = self.read(
            h, q, k, valid_in, n_in, structure.groups, structure.counts, structure.order,
            structure.intervals, structure.scores, structure.padded_length, need_budget)
        self.last_statistics = dict(stats, tree_seconds=structure.tree_seconds, structure_phases=structure.phases)
        return compact.to(features.dtype), assignment, valid, structure, soft_counts
