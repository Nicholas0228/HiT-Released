"""MiniT2I x-prediction with a v-loss, fixed per-sample randomness and the Euler sampler.

Training randomness is a pure function of the global sample position, so it is the
same for any GPU count or microbatch. Positions form blocks of 128; block B draws
t, noise and caption dropout for all of its samples from CUDA generators seeded
with SHA-256(seed:domain:stream), stream = (B // 4) * 2**20 + (B % 4) * 2**10.
This reproduces the accepted MiniT2I stream (four logical ranks of 128 per
512-sample update; a 1,024-sample update consumes two of them).
"""
import hashlib

import torch

from denoiser import router_thresholds
from hnet.grouping import budget_loss_population

BLOCK = 128


def seed_for(seed, domain, index):
    raw = f'{seed}:{domain}:{index}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2 ** 63 - 1)


def stream_randomness(seed, stream, count, image_shape, device):
    """One fresh generator per domain: t ~ sigmoid(N(-0.8, 0.8)), noise std 2, 10% caption drop."""
    def generator(domain):
        return torch.Generator(device=device).manual_seed(seed_for(seed, domain, stream))
    t = torch.sigmoid(torch.randn(count, device=device, generator=generator('t')) * .8 - .8)
    noise = torch.randn((count, *image_shape), device=device, dtype=torch.float32, generator=generator('noise')) * 2.
    drop = torch.rand(count, device=device, generator=generator('caption-drop')) < .1
    return t, noise, drop


def block_stream(block):
    return (block // 4) * 1048576 + (block % 4) * 1024


def training_randomness(seed, start, count, image_shape, device):
    """(t, noise, drop) for global sample positions [start, start + count)."""
    pieces, position, stop = [], start, start + count
    while position < stop:
        block = position // BLOCK
        values = stream_randomness(seed, block_stream(block), BLOCK, image_shape, device)
        lo, hi = position - block * BLOCK, min(stop, (block + 1) * BLOCK) - block * BLOCK
        pieces.append([value[lo:hi] for value in values])
        position = (block + 1) * BLOCK
    if len(pieces) == 1:
        return tuple(value.contiguous() for value in pieces[0])
    return tuple(torch.cat(parts, 0) for parts in zip(*pieces))


def objective(model, images, context, mask, t, noise, drop, config, a_min=.05):
    """Diffusion loss plus both global grouping budgets; returns (loss, detached statistics, aux)."""
    if not 0 < a_min <= 1:
        raise ValueError('Invalid denominator floor')
    a = (1 - t)[:, None, None, None]
    noisy = images * t[:, None, None, None] + noise * a
    # Dropped captions become all-padding: the learned null token everywhere, no grouped text.
    mask = torch.where(drop[:, None], torch.zeros_like(mask), mask)
    prediction, aux = model(noisy, context, mask, time=t, need_budget=True)
    denominator = a.clamp_min(a_min)
    v_prediction = (prediction - noisy) / denominator
    v_target = (images - noisy) / denominator
    generation = (v_prediction - v_target).square().mean()
    stats = {'generation_loss': generation.detach()}
    total = generation
    if 'hard_counts' in aux:
        # Each budget reduces over every rank's actual population before its loss.
        budget, hard, soft = budget_loss_population(aux['hard_counts'], aux['soft_counts'], aux['population1'],
                                                    config.ratios[0])
        budget2, hard2, soft2 = budget_loss_population(aux['hard_counts2'], aux['soft_counts2'],
                                                       aux['population2'], config.ratios[1])
        total = generation + config.budget_weight * budget + config.budget_weight * budget2
        stats.update(budget_loss=budget.detach(), hard_fraction=hard, soft_fraction=soft,
                     group_mean=aux['hard_counts'].float().mean(), group_min=aux['hard_counts'].min(),
                     group_max=aux['hard_counts'].max(), population_mean=aux['population1'].float().mean(),
                     budget_loss2=budget2.detach(), hard_fraction2=hard2, soft_fraction2=soft2,
                     group_mean2=aux['hard_counts2'].float().mean(), group_min2=aux['hard_counts2'].min(),
                     group_max2=aux['hard_counts2'].max())
    return total, stats, aux


@torch.no_grad()
def sample(model, context, mask, noise, guidance=2., steps=100, a_min=.05, group_guidance=None):
    """Euler from t=0 (noise) to t=1; CFG at every step; the null branch uses an all-padding caption.

    group_guidance: {'cond': (stage1, stage2), 'uncond': (stage1, stage2)} grouping thresholds for the
    conditional and unconditional branches over the whole trajectory; None keeps a calibrated value.
    The two branches then run as separate forwards, since they group tokens differently.
    """
    if group_guidance is not None and model.router is None:
        raise ValueError('Group guidance needs a HiT model with grouping stages')
    x = noise.float().clone()
    times = torch.linspace(0, 1, steps + 1, device=x.device, dtype=torch.float32)
    for index in range(steps):
        t = times[index].expand(x.shape[0])
        if group_guidance is not None:
            with router_thresholds(model, group_guidance.get('cond')):
                conditional, _ = model(x, context, mask, time=t, need_budget=False)
            with router_thresholds(model, group_guidance.get('uncond')):
                unconditional, _ = model(x, context, torch.zeros_like(mask), time=t, need_budget=False)
            prediction = unconditional + guidance * (conditional - unconditional)
        elif guidance == 1:
            prediction, _ = model(x, context, mask, time=t, need_budget=False)
        else:
            prediction, _ = model(torch.cat((x, x)), torch.cat((context, context)),
                                  torch.cat((mask, torch.zeros_like(mask))), time=torch.cat((t, t)),
                                  need_budget=False)
            conditional, unconditional = prediction.chunk(2)
            prediction = unconditional + guidance * (conditional - unconditional)
        velocity = (prediction - x) / (1 - times[index]).clamp_min(a_min)
        x = x + velocity * (times[index + 1] - times[index])
    return x.clamp(-1, 1)
