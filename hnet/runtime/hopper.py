"""Optional Hopper execution profile for the unchanged JiT/HNet computation.

This profile keeps full-width FP32 routing, exact complete average linkage,
global soft pooling/readback, and both budget losses. Floating-point reductions
and saved-activation choices differ from the native implementation.
"""
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import torch


def activation_budget(config, requested=None):
    value = (.65 if config.variant == 'baseline' else .75) if requested is None else float(requested)
    if not 0 < value <= 1:
        raise ValueError('activation_memory_budget must be in (0, 1]')
    return value


@lru_cache(maxsize=2)
def _extension_identity(include_metadata):
    import triton
    interface = importlib.import_module('flash_attn_interface')
    extension = importlib.import_module('flash_attn_3_cuda')
    records = {}
    modules = [('interface', interface), ('extension', extension)]
    if include_metadata:
        from .metadata import extension as metadata_extension
        modules.append(('fulltree_metadata', metadata_extension()))
    for label, module in modules:
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        records[label] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    return {'torch': torch.__version__, 'triton': triton.__version__, 'flash3': records}


def runtime_identity(config, requested_budget=None):
    """Bind the external CUDA extension as well as repository source in receipts."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError('The hopper profile requires an SM90 CUDA device')
    head_dim = config.width // config.heads
    if head_dim not in (64, 80) or config.attention_backend != 'sdpa':
        raise ValueError('The hopper profile requires SDPA and head dimension 64 or 80')
    if config.variant != 'baseline' and (
        config.stages not in (1, 2) or (config.image_size // config.patch_size)**2 > 1024 or config.compact_batching != 'packed' or config.routing_backend != 'gpu_exact'
    ):
        raise ValueError('The hopper HNet profile requires one or two stages, packed attention and gpu_exact routing')
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('Disable TF32 to preserve the validated FP32 routing policy')
    identity = {
        'profile': 'hopper-full-tree-v1',
        'budget_vjp': 'endpoint_incidence_v1',
        'inductor_pointwise_autotune': True,
        'inductor_gemm_autotune': False,
        'activation_memory_budget': activation_budget(config, requested_budget),
        'device_capability': [9, 0],
        'full_tree': True,
        'global_soft_pooling_and_readback': True,
        'sampling_forward': 'training_graph_per_region_with_auxiliary',
        'routing_diagnostics': 'always',
        **_extension_identity(config.variant != 'baseline'),
    }
    if head_dim == 80:
        identity['attention'] = {'implementation': 'native-torch-flash-head80',
                                 'softmax_scale': head_dim ** -.5}
    if config.proj_dropout:
        identity['projection_dropout'] = {
            'probability': config.proj_dropout,
            'blocks': list(range(config.depth // 4, (config.depth // 4) * 3)),
            'locations': ['attention_output', 'swiglu_hidden'],
            'eval_disabled': True,
        }
    return identity


def enable(model, requested_budget=None):
    """Configure one model before its ordinary compile_blocks call."""
    if model._fast_runtime:
        raise RuntimeError('The Hopper execution profile is already enabled on this model')
    identity = runtime_identity(model.config, requested_budget)
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune = False
    inductor_config.max_autotune_gemm = False
    inductor_config.max_autotune_pointwise = True
    import torch._functorch.config as aot_config
    from . import costs  # Register real attention costs before AOT tracing.
    from .grouping import enable_grouping
    from .layout import pack, unpack

    aot_config.activation_memory_budget = identity['activation_memory_budget']
    for block in model.blocks:
        block._fast_runtime = True
        if hasattr(block, 'attn'):
            block.attn._fast_runtime = True
    for router in (model.router, model.router2):
        if router is not None:
            enable_grouping(router)
    model._fast_pack = torch.compile(pack, dynamic=True)
    model._fast_unpack = torch.compile(unpack, dynamic=True)
    model._fast_runtime = True
    model.execution_identity = identity
    return identity
