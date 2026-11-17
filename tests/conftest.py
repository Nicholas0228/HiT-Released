import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


@pytest.fixture(scope='session')
def legacy():
    location = os.environ.get('HNET_LEGACY_SOURCE')
    if not location:
        pytest.skip('Set HNET_LEGACY_SOURCE to the pinned legacy checkout for cross-repo parity')
    root = Path(location)
    pin = json.loads((ROOT / 'docs/reference/source.json').read_text())
    for relative, digest in pin['legacy_files_sha256'].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    sys.path.append(str(root))
    return importlib.import_module('hnet_jit.model'), importlib.import_module('hnet_jit.denoising')


def tiny_config(**changes):
    from hnet.config import ModelConfig
    args = dict(image_size=16, patch_size=4, width=32, heads=4, classes=11,
                block_depths=(1, 1, 2, 1, 1), in_context_len=3,
                thresholds=(.6, .75), routing_workers=1, pad_multiple=4)
    args.update(changes)
    return ModelConfig(**args)


def legacy_config(config, module):
    from dataclasses import asdict
    values = asdict(config)
    values.pop('block_depths'); values.pop('thresholds'); values.pop('ratios')
    values.update(depth=config.depth, stages=config.stages,
                  encoder_depth=config.encoder_depth, intermediate_depth=config.intermediate_depth,
                  decoder_depth=config.decoder_depth, threshold=config.threshold,
                  second_threshold=config.second_threshold, target_ratio=config.target_ratio,
                  second_ratio=config.second_ratio, conditioning='tokens',
                  in_context_start=config.encoder_depth + config.intermediate_depth)
    return module.ModelConfig(**values)


def equal_tree(a, b, path='root'):
    if torch.is_tensor(a):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a: equal_tree(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): equal_tree(x, y, path + f'.{i}')
    else:
        assert a == b, path
