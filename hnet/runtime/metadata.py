"""Full-tree metadata on GPU; only bounds and per-image counts visit the CPU."""
from functools import lru_cache
from pathlib import Path
import hashlib
import math
import os
import tempfile
import time
import types

import numpy as np
import torch
from torch.utils.cpp_extension import get_default_build_root, load
from hnet.grouping import Structure
from .tree import raw_linkage_gpu, normalized_affinity_limit, AFFINITY_MAX_FP32


def valid_affinity_bounds(lower, upper, limit=AFFINITY_MAX_FP32):
    return math.isfinite(lower) and math.isfinite(upper) and upper <= limit


@lru_cache(maxsize=1)
def extension():
    source, digest = _cached_source(Path(__file__).with_name('runtime_metadata.cu'))
    return load(name=f'hnet_fulltree_metadata_{digest[:16]}', sources=[str(source)],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--fmad=false'], verbose=True)


def _cached_source(source):
    """Give identical source bytes one build path across frozen worktrees.

    Absolute CUDA source paths otherwise change the compiled binary and trigger
    competing rebuilds of the same extension during checkpoint verification.
    Publication is atomic; a modified existing cache entry is never overwritten.
    """
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    root = Path(os.environ.get('TORCH_EXTENSIONS_DIR') or get_default_build_root())
    directory = root / 'hnet_sources' / digest
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'runtime_metadata.cu'
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError(f'Modified metadata extension source cache: {target}')
    else:
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.source-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    return target, digest


def structure(router, q, k, lengths=None, affinity=None):
    batch, n, _ = q.shape
    limit = normalized_affinity_limit(q.shape[-1])
    lengths_np = np.asarray(lengths if lengths is not None else [n]*batch, dtype=np.int64)
    if lengths_np.shape != (batch,) or lengths_np.min()<1 or lengths_np.max()>n:
        raise ValueError('Invalid valid-prefix lengths')
    started=time.perf_counter()
    with torch.no_grad(), torch.autocast(q.device.type, enabled=False):
        if affinity is None:
            product=q.detach()@k.detach().transpose(-1,-2)
            affinity=.5*(product+product.transpose(-1,-2))
        bounds=torch.stack((affinity.amin(),affinity.amax()))
        lens=torch.from_numpy(lengths_np).to(q.device)
        raw=raw_linkage_gpu(affinity,lens,bounds=bounds,affinity_limit=limit,
                            precision=getattr(router,'tree_precision','fp64'))
        # FP32 trees compare FP32 heights with the threshold rounded to FP32.
        values=extension().metadata(raw,lens,router.threshold.to(raw.dtype))
        if getattr(router,'_gpu_metadata_host_bounds',None) is None:
            router._gpu_metadata_host_bounds=torch.empty(2,dtype=torch.float32,pin_memory=True)
        if getattr(router,'_gpu_metadata_host_counts',None) is None or router._gpu_metadata_host_counts.numel()<batch:
            router._gpu_metadata_host_counts=torch.empty(batch,dtype=torch.int64,pin_memory=True)
        host=router._gpu_metadata_host_counts[:batch]
        router._gpu_metadata_host_bounds.copy_(bounds,non_blocking=True)
        host.copy_(values[1],non_blocking=True)
        torch.cuda.current_stream(q.device).synchronize()
        lower,upper=router._gpu_metadata_host_bounds.tolist()
        counts=tuple(host.tolist())
        if not valid_affinity_bounds(lower, upper, limit):
            raise ValueError(f'Invalid normalized affinity: min={lower}, max={upper}, '
                             f'FP32 limit={limit}, routing width={q.shape[-1]}')
        if any(c<1 or c>m for c,m in zip(counts,lengths_np)):
            raise RuntimeError(f'Invalid GPU metadata counts: {counts}')
    ended=time.perf_counter()
    length=min(n,math.ceil(max(counts)/router.pad_multiple)*router.pad_multiple)
    phases=dict(gpu_tree_metadata_and_count_transfer=ended-started,cpu_tree=0.,metadata=0.)
    return Structure(*values,counts,length,ended-started,phases)


def enable_metadata(model):
    extension()
    for router in (model.router,model.router2):
        router.structure=types.MethodType(torch.compiler.disable(structure),router)
