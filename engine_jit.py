"""Shared training update for released JiT and HNet; no model-specific loop."""
from contextlib import nullcontext
import time
import torch
import torch.distributed as dist


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=torch.bfloat16) if dtype == 'bfloat16' else nullcontext()


def collective_finite(value):
    finite = torch.isfinite(value).all().to(dtype=torch.int32)
    if dist.is_initialized(): dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item(): raise FloatingPointError('Nonfinite loss/gradient on at least one rank')


def learning_rate(recipe, epoch, update, updates_per_epoch):
    return recipe['lr'] * min((epoch + update / updates_per_epoch) / recipe['warmup_epochs'], 1.)


def train_update(model, raw, optimizer, ema, iterator, generator, device, recipe, runtime, lr):
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups: group['lr'] = lr
    accumulation = recipe['accumulation']
    metrics, ids_seen = {}, []
    for _ in range(accumulation):
        images, labels, ids = next(iterator)
        images = images.to(device, non_blocking=True).float() / 127.5 - 1
        labels = labels.to(device, non_blocking=True)
        with autocast(device, runtime.get('dtype', 'float32')):
            result = model(images, labels, generator)
        collective_finite(result['loss'])
        (result['loss'] / accumulation).backward()
        for key in ('loss', 'generation', 'budget', 'budget1', 'budget2', 'hard_fraction', 'hard_fraction2'):
            if key in result:
                metrics[key] = metrics.get(key, 0.) + result[key].detach().float() / accumulation
        for stage, key in ((1, 'hard_counts'), (2, 'hard_counts2')):
            if key in result['aux']:
                name = f'mean_groups{stage}'
                metrics[name] = metrics.get(name, 0.) + result['aux'][key].detach().float().mean() / accumulation
        ids_seen.extend(ids.tolist())
    gradients = [p.grad.detach() for p in raw.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients, 2)))
    collective_finite(norm)
    optimizer.step(); ema.update(raw)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    duration = torch.tensor(time.perf_counter() - started, device=device)
    if dist.is_initialized():
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        for value in metrics.values():
            dist.all_reduce(value); value.div_(dist.get_world_size())
    return {**{k: v.item() for k, v in metrics.items()}, 'gradient_norm': norm.item(),
            'seconds': duration.item(), 'lr': lr}, ids_seen
