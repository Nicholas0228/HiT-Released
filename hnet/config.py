"""Separate architecture, training recipe and execution configuration files."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    variant: str = 'hnet'
    image_size: int = 256
    patch_size: int = 16
    width: int = 768
    heads: int = 12
    classes: int = 1000
    block_depths: tuple = (3, 3, 6, 3, 3)
    in_context_len: int = 8
    in_context_start: int = 0
    patch_bottleneck: int = 0
    proj_dropout: float = 0.0
    thresholds: tuple = (.6977713108062744, .7574986666440964)
    ratios: tuple = (2., 2.)
    reader_temperature: float = .05
    budget_temperature: float = .1
    budget_weight: float = .01
    router_init_seed: int = 1701
    second_router_init_seed: int = 1702
    context_init_seed: int = 2701
    time_token_init_seed: int = 5801
    # Resolved execution settings. Supplied by configs/runtime, not model files.
    attention_backend: str = 'reference'
    compact_batching: str = 'padded'
    routing_backend: str = 'scipy'
    routing_workers: int = 1
    pad_multiple: int = 16
    routing_transfer: str = 'legacy'
    routing_validation: str = 'cpu'

    def __post_init__(self):
        self.block_depths = tuple(self.block_depths)
        self.thresholds, self.ratios = tuple(self.thresholds), tuple(self.ratios)
        if self.variant not in ('baseline', 'hnet'):
            raise ValueError('variant must be baseline or hnet')
        if self.image_size % self.patch_size or self.width % self.heads or self.width // self.heads % 4:
            raise ValueError('Invalid patch/head dimensions')
        if not 0 <= self.proj_dropout < 1 or any(d < 1 for d in self.block_depths):
            raise ValueError('Invalid dropout or block depths')
        if self.variant == 'hnet':
            if (len(self.block_depths) != 5 or self.block_depths[0] != self.block_depths[-1]
                    or self.block_depths[1] != self.block_depths[-2]):
                raise ValueError('HNet uses five symmetric encoder/core/decoder stages')
            if self.patch_bottleneck or self.in_context_start or self.in_context_len < 1:
                raise ValueError('HNet uses linear input and class/time tokens from block 1')
            if len(self.thresholds) != 2 or len(self.ratios) != 2 or min(self.ratios) <= 1:
                raise ValueError('Two grouping thresholds and compression ratios are required')
        elif len(self.block_depths) != 1 or self.patch_bottleneck < 1 or not 0 <= self.in_context_start < self.depth:
            raise ValueError('Released JiT uses one dense stage, patch bottleneck and middle context entry')
        if min(self.reader_temperature, self.budget_temperature) <= 0 or self.budget_weight < 0:
            raise ValueError('Invalid grouping temperature or loss weight')
        if self.attention_backend not in ('reference', 'sdpa') or self.compact_batching not in ('padded', 'packed'):
            raise ValueError('Unknown attention execution')
        if self.routing_backend not in ('scipy', 'numba', 'gpu_exact'):
            raise ValueError('Unknown tree execution')
        if self.routing_workers < 1 or self.pad_multiple < 1:
            raise ValueError('Invalid execution geometry')

    @property
    def depth(self): return sum(self.block_depths)
    @property
    def encoder_depth(self): return self.block_depths[0] if self.variant == 'hnet' else 0
    @property
    def intermediate_depth(self): return self.block_depths[1] if self.variant == 'hnet' else 0
    @property
    def decoder_depth(self): return self.block_depths[-1] if self.variant == 'hnet' else 0
    @property
    def stages(self): return 2 if self.variant == 'hnet' else 1
    @property
    def threshold(self): return self.thresholds[0]
    @property
    def second_threshold(self): return self.thresholds[1]
    @property
    def target_ratio(self): return self.ratios[0]
    @property
    def second_ratio(self): return self.ratios[1]


MODEL_RUNTIME_KEYS = {'attention_backend', 'compact_batching', 'routing_backend',
                      'routing_workers', 'pad_multiple', 'routing_transfer', 'routing_validation'}


def load_config(model_path, training_path=None, runtime_path=None):
    model = json.loads(Path(model_path).read_text())
    runtime = json.loads(Path(runtime_path).read_text()) if runtime_path else {}
    if MODEL_RUNTIME_KEYS & model.keys():
        raise ValueError('Put execution settings in the runtime config')
    model.update({k: v for k, v in runtime.items() if k in MODEL_RUNTIME_KEYS})
    config = ModelConfig(**model)
    training = json.loads(Path(training_path).read_text()) if training_path else {}
    return config, training, runtime
