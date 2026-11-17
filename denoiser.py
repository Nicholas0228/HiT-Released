"""Upstream JiT stabilized clean-image objective and native Heun sampler."""
from contextlib import contextmanager
import random

import numpy as np
import torch
from torch import nn
from hnet.model import CleanJiT
from hnet.config import ModelConfig
from hnet.grouping import budget_loss, budget_loss_population


class Objective(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = CleanJiT(config)
        self.config = config

    def forward(self, images, labels, generator=None, stochastic=None):
        n = images.shape[0]
        if stochastic is None:
            dropped = torch.rand(n, device=images.device, generator=generator) < .1
            labels = torch.where(dropped, self.config.classes, labels)
            t = (torch.randn(n, device=images.device, generator=generator) * .8 - .8).sigmoid()
            noise = torch.randn(images.shape, device=images.device, generator=generator, dtype=images.dtype)
        else:
            t, noise, dropped = stochastic
            labels = torch.where(dropped, self.config.classes, labels)
        t4 = t[:, None, None, None]
        noisy = t4 * images + (1 - t4) * noise
        denominator = (1 - t4).clamp_min(.05)
        target = (images - noisy) / denominator
        prediction, auxiliary = self.net(noisy, t, labels, return_aux=True)
        velocity = (prediction - noisy) / denominator
        generation = (target - velocity).square().mean()
        extra = {}
        if self.net.router is not None:
            budget, hard_fraction, soft_fraction = budget_loss(
                auxiliary['hard_counts'], auxiliary['soft_counts'], auxiliary['token_count'], self.config.target_ratio)
            if self.net.router2 is not None:
                # Stage 2 uses the actual global stage-1 population, excluding conditioning and padding.
                budget2, hard_fraction2, soft_fraction2 = budget_loss_population(
                    auxiliary['hard_counts2'], auxiliary['soft_counts2'], auxiliary['hard_counts'], self.config.second_ratio)
                extra = {'budget1': budget, 'budget2': budget2, 'hard_fraction2': hard_fraction2,
                         'soft_fraction2': soft_fraction2}
                budget = budget + budget2
        else:
            budget = generation.new_zeros(())
            hard_fraction = soft_fraction = generation.new_ones(())
        total = generation + self.config.budget_weight * budget
        return {'loss': total, 'generation': generation, 'budget': budget,
                'hard_fraction': hard_fraction, 'soft_fraction': soft_fraction,
                't': t.detach(), 'aux': auxiliary, **extra}


def scheduled_thresholds(schedule, t):
    """Router thresholds (stage 1, stage 2) at time t from rows [low, high, stage1, stage2]; None keeps the model's."""
    if not schedule:
        return None
    time = float(t)
    for low, high, first, second in schedule:
        if low <= time < high:
            return first, second
    return None


@contextmanager
def router_thresholds(net, values):
    """Temporarily replace the grouping thresholds of both routers; a None entry keeps the calibrated value."""
    if values is None:
        yield
        return
    routers = (net.router, net.router2)
    saved = [router.threshold.clone() for router in routers]
    try:
        for router, value in zip(routers, values):
            if value is not None:
                router.threshold.fill_(float(value))
        yield
    finally:
        for router, value in zip(routers, saved):
            router.threshold.copy_(value)


def time_grid(steps, shift=1., device=None):
    """JiT's uniform grid in t (t = 1 is clean). A shift k > 1 moves it toward high noise:
    sigma = 1 - t, sigma' = k sigma / (1 + (k - 1) sigma), t' = 1 - sigma'."""
    times = torch.linspace(0, 1, steps + 1, device=device)
    if shift != 1.:
        sigma = 1 - times
        times = 1 - shift * sigma / (1 + (shift - 1) * sigma)
    return times


@torch.no_grad()
def sample(net, labels, initial_noise, steps=50, guidance=2.9, interval=(.1, 1.), time_shift=1., group_guidance=None):
    """Same time grid, stabilized velocity, CFG convention and final Euler as JiT.

    With the defaults this is JiT's sampler unchanged. time_shift moves the time grid toward high noise.
    group_guidance (HiT only) is {'cond': schedule, 'uncond': schedule}: per-time router thresholds for the
    conditional and the unconditional prediction (see scheduled_thresholds). Inside the guidance interval
    v = v_u + guidance (v_c - v_u); outside it the conditional prediction alone is used.
    """
    z = initial_noise.clone()
    times = time_grid(steps, time_shift, z.device)
    null = torch.full_like(labels, net.config.classes)
    if group_guidance is not None and net.router is None:
        raise ValueError('Group guidance needs a HiT model with grouping stages')
    def velocity(value, t):
        batch_t = t.expand(len(labels))
        denom = (1 - t).clamp_min(.05)
        low, high = interval
        inside = (t < high) & ((low == 0) | (t > low))
        if group_guidance is None:
            cond = net(value, batch_t, labels)
            uncond = net(value, batch_t, null)
            vc, vu = (cond - value) / denom, (uncond - value) / denom
            scale = torch.where(inside, guidance, 1.)
            return vu + scale * (vc - vu)
        with router_thresholds(net, scheduled_thresholds(group_guidance.get('cond'), t)):
            vc = (net(value, batch_t, labels) - value) / denom
        if not bool(inside):
            return vc
        with router_thresholds(net, scheduled_thresholds(group_guidance.get('uncond'), t)):
            vu = (net(value, batch_t, null) - value) / denom
        return vu + guidance * (vc - vu)
    for i in range(steps):
        t, next_t = times[i], times[i + 1]
        v = velocity(z, t)
        proposed = z + (next_t - t) * v
        if i == steps - 1:
            z = proposed
        else:
            average = .5 * (v + velocity(proposed, next_t))
            z = z + (next_t - t) * average
    return z


@contextmanager
def preserve_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state(state[3])


class DualEMA:
    def __init__(self, model, decays=(.9999, .9996)):
        self.decays = decays
        self.names = [name for name, _ in model.named_parameters()]
        self.values = [[p.detach().clone() for p in model.parameters()] for _ in decays]

    @torch.no_grad()
    def update(self, model):
        parameters = list(model.parameters())
        for decay, values in zip(self.decays, self.values):
            torch._foreach_lerp_(values, parameters, 1 - decay)

    def state_dict(self):
        return {'decays': self.decays, 'names': self.names,
                'values': [[p.detach().cpu() for p in values] for values in self.values]}

    @torch.no_grad()
    def load_state_dict(self, state):
        if state['names'] != self.names or tuple(state['decays']) != tuple(self.decays):
            raise RuntimeError('EMA identity mismatch')
        for target, source in zip(self.values, state['values']):
            for t, s in zip(target, source):
                t.copy_(s)

    @contextmanager
    def apply(self, model, index=0):
        # A CPU backup prevents another full GPU model copy during evaluation.
        parameters = list(model.parameters())
        backup = [p.detach().cpu().clone() for p in parameters]
        with torch.no_grad():
            for target, source in zip(parameters, self.values[index]):
                target.copy_(source)
        try:
            yield
        finally:
            with torch.no_grad():
                for target, source in zip(parameters, backup):
                    target.copy_(source)
