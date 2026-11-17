"""COCO FID with fixed prompts, fixed per-image noise and TensorFlow-compatible Inception.

The fixed COCO-30k cohort (val2014 captions) uses precomputed BF16 FLAN-T5 contexts,
so evaluation does not depend on encoder arithmetic. Images are quantized as in the
upstream JAX sampler (x * 127.5 + 128) and scored with clean-fid's TensorFlow
Inception at its internal resize. Keep 10k and 30k scores separate.
"""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import linalg
import torch
import torch.distributed as dist

from hnet.checkpoints import atomic_json
from .diffusion import sample, seed_for


def canonical_batches(count, batch_size, logical_world, physical_world, rank):
    """Fixed minibatch membership for any physical GPU count (whole batches per rank)."""
    if min(count, batch_size, logical_world, physical_world) <= 0 or not 0 <= rank < physical_world:
        raise ValueError('Invalid evaluation population or rank')
    batches = []
    rounds = (count + logical_world * batch_size - 1) // (logical_world * batch_size)
    for turn in range(rounds):
        for logical in range(logical_world):
            if (turn * logical_world + logical) % physical_world != rank:
                continue
            ids = list(range(logical + turn * batch_size * logical_world,
                             min(count, logical + (turn + 1) * batch_size * logical_world), logical_world))
            if ids:
                batches.append(ids)
    return batches


def frechet(features, reference):
    features = np.asarray(features, dtype=np.float64)
    mu, cov = features.mean(0), np.cov(features, rowvar=False)
    stats = np.load(reference)
    rm, rc = stats['mu'], stats['sigma']
    cross, _ = linalg.sqrtm(cov @ rc, disp=False)
    if not np.isfinite(cross).all():
        eps = np.eye(len(mu)) * 1e-6
        cross = linalg.sqrtm((cov + eps) @ (rc + eps))
    if np.iscomplexobj(cross):
        if np.max(np.abs(cross.diagonal().imag)) >= 1e-3:
            raise ValueError('Imaginary component in the FID square root')
        cross = cross.real
    result = (mu - rm) @ (mu - rm) + np.trace(cov) + np.trace(rc) - 2 * np.trace(cross)
    if not np.isfinite(result) or result < -1e-5:
        raise ValueError('Invalid FID')
    return float(max(0., result))


def quantize(images):
    return (images * 127.5 + 128.).clamp(0, 255).to(torch.uint8)


def inception_features(model, pixels):
    """FP32 GEMMs as used for the reference statistics, independent of training flags."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
            return model(pixels.float()).float().cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def evaluate(model, ema, assets, directory, update, count, *, ema_index=0, guidances=(2., 1.), steps=100,
             a_min=.05, batch_size=32, logical_world=4, prefix='eval', save_images=False):
    """Generate the first `count` fixed COCO prompts for each guidance and return FID scores.

    The first eight images per guidance are always saved; save_images keeps all of them.
    """
    from cleanfid.inception_torchscript import InceptionV3W
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    barrier = dist.barrier if distributed else (lambda: None)
    root = Path(directory)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    barrier()
    if count > assets['count']:
        raise ValueError('Evaluation larger than the fixed COCO cohort')
    contexts = np.load(assets['context'], mmap_mode='r')
    masks = np.load(assets['mask'], mmap_mode='r')
    order = np.load(assets['order'], allow_pickle=False)[:count]
    batches = canonical_batches(count, batch_size, logical_world, world, rank)
    local = np.asarray([i for ids in batches for i in ids], dtype=np.int64)
    device = next(model.parameters()).device
    inception = InceptionV3W(str(assets['inception_directory']), download=False, resize_inside=True).eval().to(device)
    size = model.config.image_size
    metrics, started = {}, time.monotonic()
    training = model.training
    model.eval()
    with ema.apply(model, index=ema_index):
        for guidance in guidances:
            name = f'cfg{guidance:g}'
            features = np.empty((len(local), 2048), np.float32)
            begin = 0
            for ids in batches:
                indices = np.asarray(ids, dtype=np.int64)
                rows = order[indices]
                context = torch.from_numpy(np.array(contexts[rows], copy=True)).view(torch.bfloat16).to(device)
                mask = torch.from_numpy(np.array(masks[rows], copy=True)).to(device)
                noise = torch.stack([torch.randn(3, size, size, device=device, generator=torch.Generator(
                    device=device).manual_seed(seed_for(0, 'coco-noise', int(i)))) * 2. for i in indices])
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    images = sample(model, context, mask, noise, guidance=guidance, steps=steps, a_min=a_min)
                pixels = quantize(images)
                values = inception_features(inception, pixels)
                if not np.isfinite(values).all():
                    raise FloatingPointError('Nonfinite Inception features')
                features[begin:begin + len(indices)] = values
                begin += len(indices)
                for j, index in enumerate(indices):
                    if index < 8:
                        Image.fromarray(pixels[j].permute(1, 2, 0).cpu().numpy()).save(root / f'{name}-example{index:03d}.png')
                    if save_images:
                        folder = root / 'images' / name
                        folder.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(pixels[j].permute(1, 2, 0).cpu().numpy()).save(folder / f'{index:05d}.png')
            np.save(root / f'{name}-rank{rank}.npy', features)
            atomic_json(root / f'{name}-rank{rank}.json', {'indices': local.tolist(), 'update': update})
            barrier()
            if rank == 0:
                gathered, seen = np.empty((count, 2048), np.float32), np.zeros(count, np.int32)
                for owner in range(world):
                    ids = np.asarray(json.loads((root / f'{name}-rank{owner}.json').read_text())['indices'], np.int64)
                    gathered[ids] = np.load(root / f'{name}-rank{owner}.npy')
                    seen[ids] += 1
                if not np.all(seen == 1):
                    raise RuntimeError('Missing or duplicated evaluation samples')
                score = frechet(gathered, assets['reference'])
                metrics[f'{prefix}/coco_fid{count}_{name}'] = score
                atomic_json(root / f'{name}-result.json', {
                    'fid': score, 'count': count, 'update': update, 'guidance': guidance, 'steps': steps,
                    'sampler': 'Euler', 'a_min': a_min, 'ema': ema.decays[ema_index],
                    'logical_world': logical_world, 'generation_batch': batch_size, 'world_size': world})
            barrier()
    model.train(training)
    if rank == 0:
        metrics[f'{prefix}/seconds'] = time.monotonic() - started
        atomic_json(root / 'complete.json', {'update': update, 'count': count, 'metrics': metrics})
    return metrics
