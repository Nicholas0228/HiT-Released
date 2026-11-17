from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


def tiny_config(**changes):
    from hnet.config import ModelConfig
    args = dict(image_size=16, patch_size=4, width=32, heads=4, classes=11,
                block_depths=(1, 1, 2, 1, 1), in_context_len=3,
                thresholds=(.6, .75), routing_workers=1, pad_multiple=4)
    args.update(changes)
    return ModelConfig(**args)


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
