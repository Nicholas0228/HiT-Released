"""Config-driven ImageNet training. Launch with torchrun; storage paths are CLI inputs."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from denoiser import Objective, DualEMA, preserve_rng
from engine_jit import learning_rate, train_update
from hnet.config import load_config
from hnet.data import ResumeShardSampler, IndexedImageNet, FolderImageNet
from hnet.data_cache import CachedImageNet
from hnet.checkpoints import Checkpoints, atomic_json, rng_state, canonical
from hnet.runtime.config import configure_model_execution

ROOT = Path(__file__).resolve().parent


def source_identity():
    files = sorted(p for pattern in ('*.py', '*.cu') for p in ROOT.rglob(pattern)
                   if not any(part.startswith('.') for part in p.relative_to(ROOT).parts))
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    revision = None
    try:
        revision = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass  # Frozen source bundles need not contain .git; content hashes remain authoritative.
    return {'sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            'git_revision': revision, 'files': manifest}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='configs/models/hnet_b16_d18.json')
    p.add_argument('--training', default='configs/training/imagenet_80ep.json')
    p.add_argument('--runtime', default='configs/runtime/hopper.json')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--data-format', choices=['cached', 'indexed', 'folder'], default='cached')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--max-updates', type=int, default=0, help='Diagnostic stop, with full recovery; not a new LR horizon')
    p.add_argument('--no-eval', action='store_true', help='Explicit diagnostic-only omission of FID')
    p.add_argument('--adm-assets', type=Path, help='Portable ADM asset manifest; see evaluation/README.md')
    p.add_argument('--adm-python', default='python', help='Separate TensorFlow evaluator environment')
    p.add_argument('--wandb-project', default=None, help='Optional; local JSONL is always written')
    p.add_argument('--wandb-entity', default=None)
    args = p.parse_args()
    config, recipe, runtime = load_config(args.model, args.training, args.runtime)
    rank, world, local = [int(os.environ.get(k, d)) for k, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    if args.device == 'cuda': torch.cuda.set_device(local)
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device('cpu')
    if world > 1: dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=90))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = device.type == 'cuda'
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 128
    if recipe['global_batch'] % (world * recipe['accumulation']):
        raise ValueError('Global batch must divide world size times accumulation')
    if not args.no_eval and not args.adm_assets:
        raise ValueError('Supply --adm-assets for the standard periodic FID schedule')
    if args.adm_assets:
        from evaluation.adm import verify_assets
        verify_assets(args.adm_assets)
    micro = recipe['global_batch'] // (world * recipe['accumulation'])
    seed = recipe['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset_cls = {'cached': CachedImageNet, 'indexed': IndexedImageNet, 'folder': FolderImageNet}[args.data_format]
    dataset = dataset_cls(args.data, seed=seed)
    if config.image_size != 256: raise ValueError('This ImageNet recipe uses 256px crops')
    if len(dataset) < recipe['global_batch']: raise ValueError('Dataset smaller than one global update')
    raw = Objective(config).to(device)
    execution = configure_model_execution(raw.net, runtime)
    identity = canonical(dict(world_size=world, microbatch=micro, model=asdict(config), recipe=recipe,
                              runtime=runtime, execution=execution, dataset=dataset.fingerprint,
                              source=source_identity(), torch=torch.__version__, no_eval=args.no_eval))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = Checkpoints(output / 'checkpoints', identity)
    if checkpoints.latest.exists() != args.resume:
        raise RuntimeError('Use a fresh output directory, or explicitly --resume the existing recovery state')
    if rank == 0: atomic_json(output / 'configuration.json', identity)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=recipe['lr'], betas=tuple(recipe['betas']),
                                 eps=recipe['eps'], weight_decay=recipe['weight_decay'])
    ema = DualEMA(raw, decays=tuple(recipe['ema_decays']))
    generator = torch.Generator(device).manual_seed(40001 + seed + rank)
    progress = dict(epoch=0, update_in_epoch=0, global_step=0, images_seen=0, training_seconds=0.)
    if args.resume: progress = checkpoints.load(raw, optimizer, ema, generator, rank)
    model = DistributedDataParallel(raw, device_ids=[local] if device.type == 'cuda' else None,
                                    gradient_as_bucket_view=True) if world > 1 else raw
    wandb_run = None
    if rank == 0 and args.wandb_project:
        import wandb
        tracking = output / 'wandb-id.json'
        run_id = json.loads(tracking.read_text())['id'] if tracking.exists() else wandb.util.generate_id()
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=run_id,
                               resume='allow', dir=str(output), config=identity)
        atomic_json(tracking, {'id': run_id})
    def log(record):
        if rank == 0:
            record = dict(record, global_step=progress['global_step'])
            with (output / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
            if wandb_run: wandb_run.log(record)
    stop = [False]
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    def barrier():
        if dist.is_initialized(): dist.barrier()
    def save(milestone=False):
        states = [None] * world
        state = rng_state(generator)
        if world > 1: dist.all_gather_object(states, state)
        else: states[0] = state
        if rank == 0:
            with preserve_rng(): receipt = checkpoints.save(raw, optimizer, ema, progress, states, milestone)
            log({'checkpoint/verified': True, 'checkpoint/bytes': receipt['bytes']})
        barrier()
    # A crash after save/before evaluation resumes the still-due evaluation.
    def evaluate_due():
        epoch = progress['epoch']
        if args.no_eval or not epoch or progress['update_in_epoch'] or epoch % recipe['evaluation_every_epochs']:
            return
        from evaluation.adm import evaluate
        full = epoch % recipe['full_evaluation_every_epochs'] == 0
        with preserve_rng(), ema.apply(raw, 0):
            raw.eval()
            scores = evaluate(raw.net, output / 'evaluation' / f'e{epoch:04d}',
                              50000 if full else 5000, recipe, runtime, args.adm_assets,
                              args.adm_python, device, checkpoint_receipt=checkpoints.latest)
            log(scores)
            raw.train()
    evaluate_due()
    last_save = time.monotonic()
    stopped = False
    while progress['epoch'] < recipe['epochs'] and not stopped:
        epoch = progress['epoch']; dataset.epoch = epoch
        sampler = ResumeShardSampler(len(dataset), rank, world, micro, recipe['accumulation'],
                                     seed, epoch, progress['update_in_epoch'])
        workers = recipe['workers']
        loader = DataLoader(dataset, batch_size=micro, sampler=sampler, num_workers=workers,
                            pin_memory=device.type == 'cuda', drop_last=True,
                            generator=torch.Generator().manual_seed(seed + epoch + rank * 100000),
                            **({'prefetch_factor': recipe['prefetch']} if workers else {}))
        iterator = iter(loader); raw.train()
        for update in range(progress['update_in_epoch'], sampler.updates_per_epoch):
            metrics, ids = train_update(model, raw, optimizer, ema, iterator, generator, device, recipe, runtime,
                                       learning_rate(recipe, epoch, update, sampler.updates_per_epoch))
            progress.update(global_step=progress['global_step'] + 1, update_in_epoch=update + 1,
                            images_seen=progress['images_seen'] + recipe['global_batch'],
                            training_seconds=progress['training_seconds'] + metrics['seconds'])
            if progress['global_step'] % recipe['log_every'] == 0:
                log(dict(metrics, epoch=epoch + (update + 1) / sampler.updates_per_epoch,
                         images_per_second=recipe['global_batch'] / metrics['seconds'],
                         gpu_hours_training=progress['training_seconds'] * world / 3600,
                         peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.))
            flags = torch.tensor([stop[0] or (args.max_updates > 0 and progress['global_step'] >= args.max_updates),
                                  time.monotonic() - last_save >= recipe['recovery_every_seconds']], device=device, dtype=torch.int32)
            if world > 1: dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            stopped = bool(flags[0].item())
            end_epoch = update + 1 == sampler.updates_per_epoch
            if end_epoch: progress.update(epoch=epoch + 1, update_in_epoch=0)
            milestone = end_epoch and progress['epoch'] % recipe['checkpoint_every_epochs'] == 0
            due = (end_epoch and progress['epoch'] % recipe['evaluation_every_epochs'] == 0)
            final = end_epoch and progress['epoch'] == recipe['epochs']
            if stopped or final or milestone or due or flags[1].item() or progress['global_step'] % recipe['recovery_every_updates'] == 0:
                save(milestone); last_save = time.monotonic()
            if end_epoch: evaluate_due()
            if stopped: break
    if rank == 0: atomic_json(output / 'status.json', dict(progress, status='stopped' if stopped else 'complete'))
    if wandb_run: wandb_run.finish()
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
