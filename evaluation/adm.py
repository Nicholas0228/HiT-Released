"""Fixed ImageNet cohorts and an isolated official ADM scoring subprocess."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from denoiser import sample
from engine_jit import autocast
from hnet.checkpoints import atomic_json, sha256_file


def verify_assets(path):
    assets = json.loads(Path(path).read_text())
    for key in ('evaluator', 'inception', 'reference'):
        asset = assets[key]
        if sha256_file(asset['path']) != asset['sha256']:
            raise RuntimeError(f'ADM asset checksum failed: {key}')
    return assets


def cohort(count, rank=0, world=1, classes=1000, base_seed=312799):
    if count < classes or count % classes or not 0 <= rank < world:
        raise ValueError('Invalid class-balanced cohort or rank')
    return [dict(id=i, label=i // (count // classes), seed=base_seed + i,
                 filename=f'{i:06d}.png') for i in range(rank, count, world)]


def initial_noise(rows, image_size, device):
    return torch.stack([torch.randn((3, image_size, image_size), device=device,
                                   generator=torch.Generator(device).manual_seed(row['seed'])) for row in rows])


def audit_images(directory, rows, image_size):
    directory = Path(directory)
    if sorted(p.name for p in directory.glob('*.png')) != [r['filename'] for r in rows]:
        raise RuntimeError('Missing, duplicate or unexpected evaluation image IDs')
    for row in rows:
        path = directory / row['filename']
        if sha256_file(path) != row['sha256']: raise RuntimeError('Evaluation image checksum mismatch')
        with Image.open(path) as im:
            if im.mode != 'RGB' or im.size != (image_size, image_size): raise RuntimeError('Wrong evaluation image geometry')
            im.load()


@torch.no_grad()
def evaluate(net, directory, count, recipe, runtime, assets_path, python, device, checkpoint_receipt):
    """Changing evaluation ranks changes ownership only, not label/noise identities."""
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    def barrier():
        if world > 1: dist.barrier()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(Path(checkpoint_receipt).read_text())
    assets = verify_assets(assets_path)
    # Optional sampler settings (time shift, HiT group guidance); absent for the training-time evaluations.
    sampler = {key: recipe[key] for key in ('time_shift', 'group_guidance')
               if recipe.get(key) is not None and not (key == 'time_shift' and recipe[key] == 1)}
    identity = dict(count=count, checkpoint_sha256=checkpoint['sha256'], ema_decay=recipe['ema_decays'][0],
                    sampler_steps=recipe['sampling_steps'], interval=recipe['guidance_interval'],
                    seed_base=recipe['evaluation_seed'], dtype=runtime.get('dtype', 'float32'),
                    assets={k: v['sha256'] for k,v in assets.items()}, **sampler)
    metrics = {}
    for guidance in recipe['guidances']:
        tag = ('gg' if 'group_guidance' in sampler else 'cfg') + f'{guidance:g}'
        if 'time_shift' in sampler:
            tag += f"-shift{sampler['time_shift']:g}"
        complete = directory / f'{tag}.json'
        if complete.exists():
            previous = json.loads(complete.read_text())
            if previous['identity'] != identity or previous['guidance'] != guidance:
                raise RuntimeError('Existing evaluation belongs to another checkpoint/protocol')
            if sha256_file(previous['report_path']) != previous['report_sha256']:
                raise RuntimeError('Completed evaluation report changed')
            metrics[f'fid_{count}/{tag}'] = previous['fid']
            continue
        attempt = [uuid.uuid4().hex[:12] if rank == 0 else None]
        if world > 1: dist.broadcast_object_list(attempt, src=0)
        work = directory / f'{tag}-{attempt[0]}'
        if rank == 0: work.mkdir(exist_ok=False)
        barrier()
        started = time.monotonic()
        rows = cohort(count, rank, world, net.config.classes, recipe['evaluation_seed'])
        for begin in range(0, len(rows), recipe['generation_batch']):
            chunk = rows[begin:begin + recipe['generation_batch']]
            labels = torch.tensor([r['label'] for r in chunk], device=device)
            noise = initial_noise(chunk, net.config.image_size, device)
            with autocast(device, runtime.get('dtype', 'float32')):
                images = sample(net, labels, noise, steps=recipe['sampling_steps'], guidance=guidance,
                                interval=tuple(recipe['guidance_interval']), **sampler)
            arrays = images.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
            arrays = np.round(np.clip((arrays + 1) * .5 * 255, 0, 255)).astype(np.uint8)
            for row, array in zip(chunk, arrays):
                Image.fromarray(array).save(work / row['filename'])
                row['sha256'] = sha256_file(work / row['filename'])
        atomic_json(work / f'rank-{rank}.json', rows)
        barrier()
        if rank == 0:
            merged = sorted([r for i in range(world) for r in json.loads((work / f'rank-{i}.json').read_text())], key=lambda r:r['id'])
            expected = cohort(count, classes=net.config.classes, base_seed=recipe['evaluation_seed'])
            if [{k:r[k] for k in expected[0]} for r in merged] != expected:
                raise RuntimeError('Evaluation rank coverage/label/noise mismatch')
            audit_images(work, merged, net.config.image_size)
            manifest = dict(identity=identity, guidance=guidance, image_size=net.config.image_size,
                            directory=str(work.resolve()), images=merged, evaluation_world_size=world,
                            training_world_size=checkpoint['identity']['world_size'])
            atomic_json(work / 'manifest.json', manifest)
            env = dict(os.environ)
            if device.type == 'cuda':
                env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', str(device.index)).split(',')[device.index or 0]
            with (work / 'scoring.log').open('x') as stream:
                subprocess.run([python, str(Path(__file__).with_name('score_adm.py')), '--manifest',
                                str(work / 'manifest.json'), '--assets', str(assets_path), '--output', str(work / 'score.json')],
                               env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            report = json.loads((work / 'score.json').read_text())
            result = dict(identity=identity, guidance=guidance, fid=report['fid'], inception_score=report['inception_score'],
                          seconds=time.monotonic()-started, report_path=str((work/'score.json').resolve()),
                          report_sha256=sha256_file(work/'score.json'))
            atomic_json(complete, result)
        barrier()
        metrics[f'fid_{count}/{tag}'] = json.loads(complete.read_text())['fid']
    return metrics
