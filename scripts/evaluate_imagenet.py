"""Generate and score ImageNet 256x256 samples from a saved checkpoint (official ADM FID).

Weights: EMA parameters from a released safetensors export (scripts/export_ema.py), from this
repository's trainer checkpoints, or from the original training milestones
(`kind: milestone-raw-and-emas`). The model config must match the checkpoint.

Sampler: a JSON file in configs/sampling. Plain CFG (cfg2.9.json for B, cfg2.4.json for L/H/XL) or
HiT group guidance with a shifted time grid (hit_xl16_e600.json, the final HiT-XL/16 epoch-600
setting). Sample i has class i // (count/1000) and initial-noise seed seed_base + i, as in training
evaluation, so any number of GPUs produces the same images.

    torchrun --standalone --nproc-per-node=8 scripts/evaluate_imagenet.py \\
      --checkpoint checkpoints/ImageNet256/HiT-XL16_ep600/model.safetensors --model configs/models/hnet_xl16.json \\
      --sampling configs/sampling/hit_xl16_e600.json --output runs/hit-xl16-e600-fid50k \\
      --adm-assets "$ADM_ASSETS" --adm-python "$ADM_PYTHON"
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.adm import evaluate  # noqa: E402
from hnet.checkpoints import atomic_json, sha256_file  # noqa: E402
from hnet.config import load_config  # noqa: E402
from hnet.model import CleanJiT  # noqa: E402
from hnet.runtime.config import configure_model_execution  # noqa: E402


def ema_parameters(path, decay):
    """EMA parameters keyed like CleanJiT.named_parameters()."""
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if checkpoint.get('kind') == 'milestone-raw-and-emas' or 'model' in checkpoint and 'ema' in checkpoint:
            ema = checkpoint['ema']
            decays = [float(d) for d in ema['decays']]
            if decay not in decays:
                raise ValueError(f'{path} holds EMA decays {decays}, not {decay}')
            state = dict(zip(ema['names'], ema['values'][decays.index(decay)]))
        else:
            raise ValueError(f'Unrecognized checkpoint layout: {path}')
    return {name[len('net.'):] if name.startswith('net.') else name: value for name, value in state.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', help='Known SHA-256 of the checkpoint; computed when omitted')
    p.add_argument('--model', type=Path, required=True, help='Model config, e.g. configs/models/hnet_xl16.json')
    p.add_argument('--runtime', type=Path, default=ROOT / 'configs/runtime/cuda.json')
    p.add_argument('--sampling', type=Path, required=True, help='Sampler config, e.g. configs/sampling/hit_xl16_e600.json')
    p.add_argument('--ema', type=float, default=.9999)
    p.add_argument('--count', type=int, default=50000)
    p.add_argument('--seed-base', type=int, default=312799)
    p.add_argument('--batch', type=int, default=64, help='Generation batch per GPU')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adm-assets', type=Path, required=True, help='Portable ADM asset manifest; see evaluation/README.md')
    p.add_argument('--adm-python', default='python', help='Separate TensorFlow evaluator environment')
    args = p.parse_args()

    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=90))

    config, _, runtime = load_config(args.model, runtime_path=args.runtime)
    sampling = json.loads(args.sampling.read_text())
    net = CleanJiT(config)
    parameters = ema_parameters(args.checkpoint, args.ema)
    expected = {name for name, _ in net.named_parameters()}
    if set(parameters) != expected:
        missing, extra = sorted(expected - set(parameters))[:5], sorted(set(parameters) - expected)[:5]
        raise RuntimeError(f'Checkpoint does not match {args.model}: missing {missing}, unexpected {extra}')
    net.load_state_dict({**net.state_dict(), **parameters}, strict=True)
    del parameters
    net = net.to(device).eval()
    configure_model_execution(net, runtime)

    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / 'checkpoint.json'
    if rank == 0:
        atomic_json(receipt, dict(path=str(args.checkpoint.resolve()), identity=dict(world_size=None),
                                  sha256=args.checkpoint_sha256 or sha256_file(args.checkpoint)))
    if world > 1:
        dist.barrier()
    recipe = dict(ema_decays=[args.ema], sampling_steps=sampling.get('steps', 50),
                  guidance_interval=sampling.get('interval', [.1, 1.]), evaluation_seed=args.seed_base,
                  guidances=[sampling['guidance']], generation_batch=args.batch,
                  time_shift=sampling.get('time_shift', 1.), group_guidance=sampling.get('group_guidance'))
    metrics = evaluate(net, args.output, args.count, recipe, runtime, args.adm_assets, args.adm_python,
                       device, receipt)
    if rank == 0:
        print(json.dumps(metrics, indent=1))
        atomic_json(args.output / 'metrics.json', dict(metrics=metrics, sampling=str(args.sampling),
                                                       model=str(args.model), count=args.count,
                                                       seed_base=args.seed_base, ema=args.ema))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
