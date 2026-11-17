"""Export one EMA of a checkpoint as a standalone safetensors file for release.

Reads this repository's trainer checkpoints or the original training milestones
(`kind: milestone-raw-and-emas`) and writes the chosen EMA's parameters, keyed like
CleanJiT.named_parameters(), with the model config and provenance as metadata.
scripts/evaluate_imagenet.py loads the result directly.

    python scripts/export_ema.py --checkpoint milestone-e0600.pt --model configs/models/hnet_xl16.json \\
      --ema 0.9999 --output ImageNet256/HiT-XL16_ep600/model.safetensors
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hnet.checkpoints import sha256_file  # noqa: E402
from hnet.config import load_config  # noqa: E402
from hnet.model import CleanJiT  # noqa: E402
from scripts.evaluate_imagenet import ema_parameters  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--ema', type=float, default=.9999)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from safetensors.torch import save_file

    config, _, _ = load_config(args.model)
    expected = {name for name, _ in CleanJiT(config).named_parameters()}
    parameters = ema_parameters(args.checkpoint, args.ema)
    if set(parameters) != expected:
        raise RuntimeError(f'Checkpoint does not match {args.model}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    progress = checkpoint.get('progress', {})
    metadata = dict(ema_decay=str(args.ema), model_config=args.model.read_text(),
                    source_sha256=sha256_file(args.checkpoint),
                    epoch=str(progress.get('epoch', '')), global_step=str(progress.get('global_step', '')))
    tensors = {name: value.detach().contiguous().clone() for name, value in parameters.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    print(json.dumps(dict(output=str(args.output), parameters=len(tensors),
                          numel=sum(t.numel() for t in tensors.values()), sha256=sha256_file(args.output),
                          **{k: v for k, v in metadata.items() if k != 'model_config'}), indent=1))


if __name__ == '__main__':
    main()
