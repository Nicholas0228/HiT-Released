"""Optional full-tree GPU routing and equivalent global soft reader.

All hard groups, full-tree budget nodes, original features, and global soft
readback are retained. Only the implementation and FP32 reduction order change.
"""
import math
import time
import types
import numpy as np
import torch
from torch.nn import functional as F
from ..grouping import SoftGrouping, VarLenGrouping, Structure, batch_metadata, batch_metadata_varlen
from .gpu_linkage import finalize_linkage
from .tree import raw_linkage_gpu
from .layout import pack, unpack
from .affinity import group_logits
from .replay import fused_replay


def soft_read(self, h, q, k, groups, counts, order, intervals, scores, length, need_budget, affinity):
    with torch.autocast(h.device.type, enabled=False):
        batch, n, width = q.shape
        idx = groups[..., None].expand(-1, -1, width)
        sizes = torch.zeros(batch, length, device=h.device).scatter_add_(
            1, groups, torch.ones(batch, n, device=h.device))
        qm = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, q)
        km = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, k)
        qm = qm / sizes.clamp_min(1)[..., None]
        km = km / sizes.clamp_min(1)[..., None]
        logits = group_logits(q, k, qm, km, affinity, groups, order, sizes, torch.full_like(counts, n))
        valid = torch.arange(length, device=h.device)[None, :] < counts[:, None]
        assignment = (logits / self.reader_temperature).masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        mass = assignment.sum(1)
        compact = assignment.transpose(1, 2) @ h
        compact = compact / (mass[..., None] + 1e-8)
        if need_budget:
            merge_scores = fused_replay(q, k, order, intervals)
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


def soft_forward(self, features, need_budget=True, fixed_structure=None):
    h, q, k = self.project(features)
    with torch.no_grad(), torch.autocast(q.device.type, enabled=False):
        product = q.detach() @ k.detach().transpose(-1, -2)
        affinity = .5 * (product + product.transpose(-1, -2))
    del product
    structure = fixed_structure or self.structure(q, k, affinity=affinity)
    compact, assignment, valid, soft_counts, stats = self.read(
        h, q, k, structure.groups, structure.counts, structure.order,
        structure.intervals, structure.scores, structure.padded_length, need_budget, affinity)
    self.last_statistics = dict(stats, tree_seconds=structure.tree_seconds, structure_phases=structure.phases)
    return compact.to(features.dtype), assignment, valid, structure, soft_counts


def varlen_read(self, h, q, k, valid_in, n_in, groups, counts, order, intervals, scores, length, need_budget, affinity):
    with torch.autocast(h.device.type, enabled=False):
        batch, n, width = q.shape
        vin = valid_in.to(q.dtype)
        idx = groups[..., None].expand(-1, -1, width)
        sizes = torch.zeros(batch, length, device=h.device).scatter_add_(1, groups, vin)
        qm = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, q * vin[..., None])
        km = torch.zeros(batch, length, width, device=h.device).scatter_add(1, idx, k * vin[..., None])
        qm = qm / sizes.clamp_min(1)[..., None]
        km = km / sizes.clamp_min(1)[..., None]
        logits = group_logits(q, k, qm, km, affinity, groups, order, sizes, n_in)
        valid = torch.arange(length, device=h.device)[None, :] < counts[:, None]
        assignment = (logits / self.reader_temperature).masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        assignment = assignment * vin[..., None]              # padded rows read/write nothing
        mass = assignment.sum(1)
        compact = assignment.transpose(1, 2) @ h
        compact = compact / (mass[..., None] + 1e-8)
        n_float = n_in.to(q.dtype)
        if need_budget and n > 1:
            merge_scores = fused_replay(q, k, order, intervals)
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


def varlen_forward(self, features, valid_in, n_in, lengths, need_budget=True, fixed_structure=None):
    h, q, k = project_valid(self, features, lengths)
    with torch.no_grad(), torch.autocast(q.device.type, enabled=False):
        product = q.detach() @ k.detach().transpose(-1, -2)
        affinity = .5 * (product + product.transpose(-1, -2))
    del product
    structure = fixed_structure or self.structure(q, k, lengths, affinity=affinity)
    compact, assignment, valid, soft_counts, stats = self.read(
        h, q, k, valid_in, n_in, structure.groups, structure.counts, structure.order,
        structure.intervals, structure.scores, structure.padded_length, need_budget, affinity)
    self.last_statistics = dict(stats, tree_seconds=structure.tree_seconds, structure_phases=structure.phases)
    return compact.to(features.dtype), assignment, valid, structure, soft_counts


def project_valid(router, features, lengths):
    lengths_np = np.asarray(lengths, dtype=np.int32)
    cumulative_np = np.concatenate(([0], np.cumsum(lengths_np))).astype(np.int32)
    rows_np = np.repeat(np.arange(len(lengths_np)), lengths_np)
    cumulative = torch.as_tensor(cumulative_np, device=features.device)
    rows = torch.as_tensor(rows_np, device=features.device)
    compact = pack(features, rows, cumulative)
    _, q, k = router.project(compact)
    q = unpack(q, rows, cumulative, features.shape[1])
    k = unpack(k, rows, cumulative, features.shape[1])
    return features.float(), q, k



def structure_gpu_exact(router, q, k, lengths=None, affinity=None):
    """Optional router backend; reader/replay/reconstruction remain unchanged."""
    from numba import set_num_threads
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
        if affinity is None:
            product = q.detach() @ k.detach().transpose(-1, -2)
            affinity = .5 * (product + product.transpose(-1, -2))
        bounds = torch.stack((affinity.amin(), affinity.amax()))
        if router._host_bounds is None:
            router._host_bounds = torch.empty(2, dtype=torch.float32, pin_memory=True)
        router._host_bounds.copy_(bounds, non_blocking=True)
        validated = time.perf_counter()
        lens = torch.from_numpy(lengths_np).to(q.device)
        raw = raw_linkage_gpu(affinity, lens, bounds=bounds)
        if getattr(router, '_host_raw_tree', None) is None or router._host_raw_tree.numel() < raw.numel():
            router._host_raw_tree = torch.empty(raw.numel(), dtype=torch.float64, pin_memory=True)
        host = router._host_raw_tree[:raw.numel()].view(raw.shape)
        host.copy_(raw, non_blocking=True)
        torch.cuda.current_stream(q.device).synchronize()
        lower, upper = router._host_bounds.tolist()
        if not math.isfinite(lower) or not math.isfinite(upper) or upper > 1 + 2e-6:
            raise ValueError('Invalid normalized affinity')
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



def enable_grouping(router):
    variable = isinstance(router, VarLenGrouping)
    from .metadata import structure
    router.structure = types.MethodType(torch.compiler.disable(structure), router)
    router.read = types.MethodType(varlen_read if variable else soft_read, router)
    router.forward = types.MethodType(varlen_forward if variable else soft_forward, router)
