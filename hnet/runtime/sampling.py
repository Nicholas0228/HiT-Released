"""Reuse each compiled training forward graph during sampling, then detach.

Autograd is enabled only inside one compiled region at a time. No backward or
parameter update runs, and no graph is retained across regions. Reader budgets
and diagnostics use the same graph in training and sampling.
"""
from functools import wraps
import torch


def detach(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach(item) for item in value)
    if isinstance(value, list):
        return [detach(item) for item in value]
    return value


def region(function, differentiable, position_index=None):
    @wraps(function)
    def forward(*args, **kwargs):
        if torch.is_grad_enabled():
            return function(*args, **kwargs)
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Hopper graph-matched sampling requires torch.no_grad(), not inference_mode()')
        values = list(args)
        with torch.enable_grad():
            for index in differentiable:
                value = values[index]
                if not value.is_floating_point():
                    raise TypeError('Differentiable forward input must be floating point')
                values[index] = value.detach().requires_grad_(True)
            result = function(*values, **kwargs)
        return detach(result)
    return forward


def enable_sampling(model):
    for block in model.blocks:
        block.forward = region(block.forward, (0, 1), position_index=2)
        block.forward_packed = region(block.forward_packed, (0, 1), position_index=5)
    model.final_layer.forward = region(model.final_layer.forward, (0, 1))
    for name in ('down1', 'down2', 'up2', 'up1'):
        projection = getattr(model, name, None)
        if isinstance(projection, torch.nn.Linear):
            projection.forward = region(projection.forward, (0,))
    if model.router is not None:
        model._fast_pack = region(model._fast_pack, (0,))
        model._fast_unpack = region(model._fast_unpack, (0,))
    for router in (model.router, model.router2):
        if router is None:
            continue
        router.project = region(router.project, (0,))
        router.read = region(router.read, (0, 1, 2))
        router.reconstruct = region(router.reconstruct, (0, 1))
    original = model.forward

    @wraps(original)
    def forward(image, time, labels, return_aux=False):
        output, auxiliary = original(image, time, labels, return_aux=True)
        return (output, auxiliary) if return_aux else output
    model.forward = forward
