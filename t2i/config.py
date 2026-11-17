"""Text-to-image architecture, recipe and execution configuration.

Architecture lives in configs/t2i/models, the training recipe in configs/t2i/training
and kernels/precision in configs/t2i/runtime; load_t2i_config merges them.
"""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class T2IConfig:
    variant: str = 'hnet'              # 'baseline': released dense MiniT2I; 'hnet': HiT
    image_size: int = 512
    patch_size: int = 16
    width: int = 768
    heads: int = 12
    mlp_ratio: float = 2.6667           # SwiGLU hidden 2,048 at width 768
    text_width: int = 1024              # FLAN-T5-Large features
    text_length: int = 256
    text_depth: int = 2                 # plain text preamble blocks
    block_depths: tuple = (3, 3, 10, 3, 3)
    patch_bottleneck: int = 0           # released dense input: 128-channel bottleneck
    thresholds: tuple = None            # calibrated merge thresholds; None only while calibrating
    ratios: tuple = (2., 2.)
    reader_temperature: float = .05
    budget_temperature: float = .1
    budget_weight: float = .01
    router_init_seed: int = 1701
    second_router_init_seed: int = 1702
    time_token_init_seed: int = 5701
    # Core blocks beyond a base layout get their own seeded initialization (the paper's 3/3/15/3/3
    # model extends 3/3/10/3/3, whose modules keep their initialization); 0 disables.
    core_extension_base_depth: int = 0
    core_extension_seed: int = 0
    # HiT copies shared modules from the same seed's released dense model.
    reference_depth: int = 17
    reference_bottleneck: int = 128
    # Resolved execution settings, supplied by configs/t2i/runtime.
    attention_backend: str = 'sdpa'
    compact_batching: str = 'packed'
    routing_backend: str = 'gpu_exact'
    routing_workers: int = 4
    pad_multiple: int = 16
    routing_transfer: str = 'packed'
    routing_validation: str = 'gpu'
    tree_precision: str = 'fp32'

    def __post_init__(self):
        self.block_depths, self.ratios = tuple(self.block_depths), tuple(self.ratios)
        if self.thresholds is not None:
            self.thresholds = tuple(float(value) for value in self.thresholds)
        if self.variant not in ('baseline', 'hnet'):
            raise ValueError('variant must be baseline or hnet')
        if self.image_size % self.patch_size or self.width % self.heads:
            raise ValueError('Invalid patch/head dimensions')
        if min(self.block_depths) < 1 or self.text_depth < 0:
            raise ValueError('Invalid block depths')
        if self.variant == 'baseline':
            if len(self.block_depths) != 1 or self.patch_bottleneck < 1 or self.thresholds is not None:
                raise ValueError('The released dense model is one stage with a patch bottleneck')
        else:
            d = self.block_depths
            if len(d) != 5 or d[0] != d[4] or d[1] != d[3]:
                raise ValueError('HiT uses five symmetric encoder/core/decoder stages')
            if self.patch_bottleneck:
                raise ValueError('HiT uses a linear patch input')
            if len(self.ratios) != 2 or min(self.ratios) <= 1:
                raise ValueError('Two compression ratios above 1 are required')
            if self.thresholds is not None and len(self.thresholds) != 2:
                raise ValueError('Two calibrated thresholds are required')
            if self.reference_depth - 2 * (d[0] + d[1]) < 1:
                raise ValueError('The shared dense reference must cover the outer stages')
            if self.core_extension_base_depth and not (
                    self.core_extension_seed and self.reference_depth <= self.core_extension_base_depth < sum(d)):
                raise ValueError('A core extension needs a seed and a base layout between the reference and the model')
            if self.image_tokens + self.text_length > 1280:
                raise ValueError('The GPU tree supports at most 1,280 grouped tokens')
        if min(self.reader_temperature, self.budget_temperature) <= 0 or self.budget_weight < 0:
            raise ValueError('Invalid grouping temperature or loss weight')
        if self.attention_backend not in ('reference', 'sdpa') or self.compact_batching not in ('padded', 'packed'):
            raise ValueError('Unknown attention execution')
        if self.routing_backend not in ('scipy', 'numba', 'gpu_exact') or self.tree_precision not in ('fp32', 'fp64'):
            raise ValueError('Unknown tree execution')
        if self.tree_precision == 'fp32' and self.variant == 'hnet' and self.routing_backend != 'gpu_exact':
            raise ValueError('FP32 trees require the gpu_exact routing backend')

    @property
    def depth(self): return sum(self.block_depths)
    @property
    def encoder_depth(self): return self.block_depths[0] if self.variant == 'hnet' else 0
    @property
    def intermediate_depth(self): return self.block_depths[1] if self.variant == 'hnet' else 0
    @property
    def decoder_depth(self): return self.block_depths[-1] if self.variant == 'hnet' else 0
    @property
    def grid(self): return self.image_size // self.patch_size
    @property
    def image_tokens(self): return self.grid ** 2


RUNTIME_KEYS = {'attention_backend', 'compact_batching', 'routing_backend', 'routing_workers',
                'pad_multiple', 'routing_transfer', 'routing_validation', 'tree_precision'}


def load_t2i_config(model_path, training_path=None, runtime_path=None, *, calibrating=False):
    model = json.loads(Path(model_path).read_text())
    runtime = json.loads(Path(runtime_path).read_text()) if runtime_path else {}
    if RUNTIME_KEYS & model.keys():
        raise ValueError('Put execution settings in the runtime config')
    model.update({k: v for k, v in runtime.items() if k in RUNTIME_KEYS})
    config = T2IConfig(**model)
    if config.variant == 'hnet' and config.thresholds is None and not calibrating:
        raise ValueError('Calibrate HiT thresholds first: python scripts/calibrate_t2i.py')
    training = json.loads(Path(training_path).read_text()) if training_path else {}
    return config, training, runtime
