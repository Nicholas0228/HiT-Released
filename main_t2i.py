"""Config-driven text-to-image training (MiniT2I dense baseline or HiT). Launch with torchrun.

Every rank reads its slice of the fixed global sample sequence; per-sample noise,
timesteps and caption dropout depend only on the sample position. FLAN-T5-Large
encodes captions online (frozen, BF16). Storage paths are CLI inputs.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from denoiser import DualEMA, preserve_rng
from hnet.checkpoints import Checkpoints, atomic_json, canonical, rng_state
from main_jit import source_identity
from t2i.config import load_t2i_config
from t2i.data import CC12MSequence, UpdateBatches, encode_captions, to_model_range
from t2i.diffusion import objective, training_randomness
from t2i.inputs import load_inputs
from t2i.finetune import data_offset, load_sft_inputs, phase_cursor, read_parent, restore_parent
from t2i.model import TextToImage


def learning_rate(recipe, update):
    """Linear warmup from initial_lr, then constant; continuation never restarts it."""
    return recipe['initial_lr'] + (recipe['lr'] - recipe['initial_lr']) * min(update / recipe['warmup_updates'], 1.)


def evaluations_due(recipe, update):
    """[(sample count, EMA decay)] after this optimizer update."""
    plan = recipe['evaluation']
    if update <= 0 or not plan.get('enabled', True):
        return []
    if update in plan['full_updates']:
        count = plan['full_count']
    elif update % plan['every_updates'] == 0:
        count = plan['count']
    else:
        return []
    due = [(count, plan['monitor_ema'])]
    if update % plan['reference_every_updates'] == 0 and count == plan['full_count']:
        due.append((count, plan['reference_ema']))
    return due


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='configs/t2i/models/hit_b16_d27.json')
    p.add_argument('--training', default='configs/t2i/training/cc12m_250k.json')
    p.add_argument('--runtime', default='configs/t2i/runtime/cuda.json')
    p.add_argument('--inputs', type=Path, required=True,
                   help='inputs.json from fetch_t2i_inputs.py or prepare_t2i_finetune.py')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--finetune-from', type=Path,
                   help='Full 250k checkpoint from this trainer; use with configs/t2i/training/sft40k.json')
    p.add_argument('--microbatch', type=int, default=0, help='Override the recipe microbatch (memory only)')
    p.add_argument('--max-updates', type=int, default=0, help='Stop at this update with full recovery')
    p.add_argument('--no-eval', action='store_true', help='Skip the periodic COCO FID (diagnostics only)')
    p.add_argument('--wandb-project', default=None, help='Optional; local JSONL is always written')
    p.add_argument('--wandb-entity', default=None)
    args = p.parse_args()
    config, recipe, runtime = load_t2i_config(args.model, args.training, args.runtime)
    offset = data_offset(recipe)
    if args.finetune_from and (args.resume or not offset):
        raise ValueError('--finetune-from starts a fresh SFT output; --resume continues that output')
    if offset and not (args.finetune_from or args.resume):
        raise ValueError('SFT needs --finetune-from for its first launch, or --resume thereafter')
    inputs = (load_sft_inputs(args.inputs, recipe) if offset else
              load_inputs(args.inputs, require_evaluation=not args.no_eval))
    rank, world, local = [int(os.environ.get(k, d)) for k, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32 router GEMMs; the tree guard assumes no TF32
    torch.backends.cudnn.allow_tf32 = True
    global_batch = recipe['global_batch']
    microbatch = args.microbatch or recipe['microbatch']
    if global_batch % world or (global_batch // world) % microbatch:
        raise ValueError('global_batch / world must be a multiple of the microbatch')
    local_batch = global_batch // world
    accumulation = local_batch // microbatch
    if offset and config.variant == 'hnet' and accumulation != 1:
        raise ValueError('The SFT recipe uses one 1024-sample budget population: use 16 x 64 or 8 x 128')
    seed = recipe['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    raw = TextToImage(config).to(device)
    if runtime.get('compile', False):
        raw.compile_blocks()
    # With accumulation, each HiT budget reduces over one global microbatch (all ranks).
    identity = canonical(dict(world_size=world, global_batch=global_batch, microbatch=microbatch,
                              budget_population=world * microbatch, model=asdict(config), recipe=recipe,
                              runtime=runtime, inputs=inputs['fingerprint'], source=source_identity(),
                              torch=torch.__version__, no_eval=args.no_eval))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    parent_state = None
    if args.finetune_from:
        if (output / 'configuration.json').exists() or (output / 'checkpoints/latest.json').exists():
            raise RuntimeError('Start SFT in a new output directory; the pretraining run is read-only')
        parent_state, lineage = read_parent(args.finetune_from, identity)
        identity['finetune_parent'] = lineage
    elif offset and args.resume:
        identity['finetune_parent'] = json.loads((output / 'configuration.json').read_text())['finetune_parent']
    checkpoints = Checkpoints(output / 'checkpoints', identity, milestone_name='checkpoint-u{global_step:07d}.pt')
    if checkpoints.latest.exists() != args.resume:
        raise RuntimeError('Use a fresh output directory, or explicitly --resume the existing recovery state')
    # All ranks must finish the new-output check before rank zero publishes the
    # configuration; otherwise a slower rank mistakes this launch for an old run.
    if world > 1:
        dist.barrier()
    if rank == 0:
        atomic_json(output / 'configuration.json', identity)
        if not inputs['complete']:
            print(f"WARNING: {inputs['missing_shards']} CC12M shards are missing; a run reaching them will fail",
                  flush=True)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=recipe['initial_lr'], betas=tuple(recipe['betas']),
                                  eps=recipe['eps'], weight_decay=recipe['weight_decay'], fused=True)
    ema = DualEMA(raw, decays=tuple(recipe['ema_decays']))
    generator = torch.Generator().manual_seed(seed + rank)  # recovery bookkeeping only; noise is positional
    progress = dict(epoch=0, update_in_epoch=0, global_step=0, images_seen=0, training_seconds=0.)
    if args.resume:
        progress = checkpoints.load(raw, optimizer, ema, generator, rank)
    elif parent_state is not None:
        progress = restore_parent(parent_state, raw, optimizer, ema, generator, rank, recipe)
        del parent_state
    if offset:
        expected = phase_cursor(progress['global_step'], recipe)
        if any(progress.get(key) != value for key, value in expected.items()):
            raise ValueError('The SFT data cursor disagrees with the saved global update')
    # Encoder/loading/tracking setup must not advance the restored training RNG.
    with preserve_rng():
        from transformers import T5EncoderModel
        encoder = T5EncoderModel.from_pretrained(inputs['encoder'], local_files_only=True).eval().to(device)
        encoder.requires_grad_(False)
        # The dense baseline keeps the released, unused time/pooled modules.
        model = DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True,
                                        gradient_as_bucket_view=True) if world > 1 else raw
        wandb_run = None
        if rank == 0 and args.wandb_project:
            import wandb
            tracking = output / 'wandb-id.json'
            run_id = json.loads(tracking.read_text())['id'] if tracking.exists() else wandb.util.generate_id()
            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=run_id, resume='allow',
                                   dir=str(output), config=identity)
            atomic_json(tracking, {'id': run_id})

    def log(record):
        if rank == 0:
            record = dict(record, global_step=progress['global_step'])
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
            if wandb_run:
                wandb_run.log(record, step=progress['global_step'])

    stop = [False]
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))

    def barrier():
        if dist.is_initialized():
            dist.barrier()

    def save(milestone=False):
        states = [None] * world
        state = rng_state(generator)
        if world > 1:
            dist.all_gather_object(states, state)
        else:
            states[0] = state
        if rank == 0:
            with preserve_rng():
                receipt = checkpoints.save(raw, optimizer, ema, progress, states, milestone)
            log({'checkpoint/verified': True, 'checkpoint/bytes': receipt['bytes']})
        barrier()

    def evaluate_due(update):
        """Run scheduled COCO FID; a crash after the checkpoint reruns it on --resume."""
        if args.no_eval or not recipe['evaluation'].get('enabled', True):
            return
        from t2i.evaluation import evaluate
        plan = recipe['evaluation']
        for count, decay in evaluations_due(recipe, update):
            directory = output / 'evaluation' / f'u{update:07d}-ema{decay:g}-n{count}'
            if (directory / 'complete.json').exists():
                continue
            if rank == 0 and directory.exists():
                directory.rename(directory.with_name(f'{directory.name}.incomplete-{time.time_ns()}'))
            barrier()
            with preserve_rng():
                scores = evaluate(raw, ema, inputs['evaluation'], directory, update, count,
                                  ema_index=list(ema.decays).index(decay), guidances=plan['guidances'],
                                  steps=plan['steps'], a_min=plan['a_min'], batch_size=plan['generation_batch'],
                                  logical_world=plan['logical_world'], prefix=f'eval/ema{decay:g}')
            log(scores)

    target = min(recipe['updates'], args.max_updates or recipe['updates'])
    if target <= progress['global_step']:
        raise ValueError('--max-updates must exceed the current global update')
    if args.finetune_from:
        # Save the verified transition before consuming a fine-tuning example.
        save(milestone=True)
    dataset = CC12MSequence(inputs, config.image_size, config.text_length)
    sampler = UpdateBatches(progress['global_step'] - offset, target - offset, global_batch, rank, world)
    workers = recipe['workers']
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0, prefetch_factor=2 if workers else None,
                        multiprocessing_context='spawn' if workers else None, collate_fn=dataset.collate,
                        generator=torch.Generator().manual_seed(seed + rank + progress['global_step']))
    image_shape = (3, config.image_size, config.image_size)
    evaluate_due(progress['global_step'])
    last_save = time.monotonic()
    stopped = False
    iterator = iter(loader)
    raw.train()
    for update in range(progress['global_step'], target):
        started = time.perf_counter()
        batch = next(iterator)
        first = update * global_batch + rank * local_batch
        data_first = first - offset * global_batch
        if not torch.equal(batch['positions'], torch.arange(data_first, data_first + local_batch)):
            raise RuntimeError('Data cursor does not match the global sample sequence')
        images = to_model_range(batch['image'].to(device, non_blocking=True))
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        context = encode_captions(encoder, input_ids, mask, microbatch)
        t, noise, drop = training_randomness(seed, first, local_batch, image_shape, device)
        lr = learning_rate(recipe, update)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        summaries = []
        for micro in range(accumulation):
            part = slice(micro * microbatch, (micro + 1) * microbatch)
            sync = model.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, summary, aux = objective(model, images[part], context[part], mask[part], t[part],
                                                   noise[part], drop[part], config, recipe['a_min'])
                    loss = loss / accumulation
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Nonfinite training loss on at least one rank')
                loss.backward()
            summaries.append(summary)
        gradients = [p.grad for p in raw.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))
        finite = torch.isfinite(norm).to(torch.int32)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Nonfinite gradient before the optimizer update')
        optimizer.step()
        ema.update(raw)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        progress.update(global_step=update + 1, update_in_epoch=update + 1, images_seen=(update + 1) * global_batch,
                        training_seconds=progress['training_seconds'] + seconds)
        progress.update(phase_cursor(update + 1, recipe))
        if update < 10 or (update + 1) % recipe['log_every'] == 0:
            names = list(summaries[0])
            values = torch.stack([torch.stack([s[k].float() for s in summaries]).mean() for k in names])
            if world > 1:
                dist.all_reduce(values)
                values /= world
            record = {'train/' + k: v for k, v in zip(names, values.tolist())}
            record.update({'train/lr': lr, 'train/gradient_norm': norm.item(), 'seconds_per_update': seconds,
                           'images_per_second': global_batch / seconds, 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'gpu_hours_training': progress['training_seconds'] * world / 3600})
            if raw.router is not None:
                for suffix, router in (('', raw.router), ('2', raw.router2)):
                    record[f'router{suffix}/gradient_norm'] = torch.stack(
                        [router.q_proj.weight.grad.norm(), router.k_proj.weight.grad.norm()]).norm().item()
                    for key in ('assignment_entropy', 'max_assignment', 'unsaturated_merges'):
                        record[f'router{suffix}/{key}'] = aux['stats' + suffix][key].float().mean().item()
            log(record)
        flags = torch.tensor([stop[0], time.monotonic() - last_save >= recipe['recovery_every_seconds']],
                             device=device, dtype=torch.int32)
        if world > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        stopped = bool(flags[0].item())
        done = update + 1
        milestone = done % recipe['milestone_every_updates'] == 0 or done in recipe['evaluation']['full_updates']
        if (stopped or milestone or flags[1].item() or done == target or done == recipe['first_recovery_update']):
            save(milestone)
            last_save = time.monotonic()
        evaluate_due(done)
        if stopped:
            break
    if rank == 0:
        status = 'complete' if progress['global_step'] >= recipe['updates'] else 'stopped'
        atomic_json(output / 'status.json', dict(progress, status=status))
    if wandb_run:
        wandb_run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
