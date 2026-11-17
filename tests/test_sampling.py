import pytest
import torch

from conftest import tiny_config
from denoiser import router_thresholds, sample, scheduled_thresholds, time_grid
from hnet.model import CleanJiT


def reference_sample(net, labels, initial_noise, steps=50, guidance=2.9, interval=(.1, 1.)):
    """The sampler before time shift and group guidance were added (JiT's sampler)."""
    z = initial_noise.clone()
    times = torch.linspace(0, 1, steps + 1, device=z.device)
    null = torch.full_like(labels, net.config.classes)
    def velocity(value, t):
        batch_t = t.expand(len(labels))
        cond = net(value, batch_t, labels)
        uncond = net(value, batch_t, null)
        denom = (1 - t).clamp_min(.05)
        vc, vu = (cond - value) / denom, (uncond - value) / denom
        low, high = interval
        scale = torch.where((t < high) & ((low == 0) | (t > low)), guidance, 1.)
        return vu + scale * (vc - vu)
    for i in range(steps):
        t, next_t = times[i], times[i + 1]
        v = velocity(z, t)
        proposed = z + (next_t - t) * v
        z = proposed if i == steps - 1 else z + (next_t - t) * .5 * (v + velocity(proposed, next_t))
    return z


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    net = CleanJiT(tiny_config()).eval()
    with torch.no_grad():  # zero-initialized heads would make every prediction 0; perturb so routing matters
        generator = torch.Generator().manual_seed(1)
        for parameter in net.parameters():
            parameter.add_(.05 * torch.randn(parameter.shape, generator=generator))
    labels = torch.tensor([1, 7])
    noise = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(3))
    return net, labels, noise


def test_default_sampler_is_unchanged(model):
    net, labels, noise = model
    assert torch.equal(sample(net, labels, noise, steps=3), reference_sample(net, labels, noise, steps=3))


def test_time_grid_shift():
    assert torch.equal(time_grid(4), torch.linspace(0, 1, 5))
    k, sigma = 1.7, 1 - torch.linspace(0, 1, 9)
    expected = 1 - k * sigma / (1 + (k - 1) * sigma)
    shifted = time_grid(8, k)
    assert torch.allclose(shifted, expected) and shifted[0] == 0 and shifted[-1] == 1
    assert bool((shifted[1:] > shifted[:-1]).all()) and bool((shifted[1:-1] < torch.linspace(0, 1, 9)[1:-1]).all())


def test_threshold_schedule_lookup():
    schedule = [[0., .5, .6, None], [.5, 1.01, .7, .8]]
    assert scheduled_thresholds(schedule, torch.tensor(.2)) == (.6, None)
    assert scheduled_thresholds(schedule, torch.tensor(1.)) == (.7, .8)
    assert scheduled_thresholds(None, torch.tensor(.2)) is None
    assert scheduled_thresholds([[.85, 1.01, .5, None]], torch.tensor(.2)) is None


def test_group_guidance_restores_thresholds_and_changes_samples(model):
    net, labels, noise = model
    before = [r.threshold.clone() for r in (net.router, net.router2)]
    with router_thresholds(net, (.1, None)):
        assert net.router.threshold.item() == pytest.approx(.1) and torch.equal(net.router2.threshold, before[1])
    assert all(torch.equal(r.threshold, b) for r, b in zip((net.router, net.router2), before))
    # Empty schedules inside the interval reproduce plain CFG; real schedules change the sample.
    plain = sample(net, labels, noise, steps=3, guidance=2., interval=(0., 1.))
    empty = sample(net, labels, noise, steps=3, guidance=2., interval=(0., 1.), group_guidance={})
    assert torch.allclose(plain, empty, atol=1e-6)
    guided = sample(net, labels, noise, steps=3, guidance=2., interval=(0., 1.),
                    group_guidance={'cond': [[0., 1.01, .95, None]], 'uncond': [[0., 1.01, .3, None]]})
    assert not torch.allclose(plain, guided)
    assert all(torch.equal(r.threshold, b) for r, b in zip((net.router, net.router2), before))


def test_group_guidance_needs_hierarchy(model):
    _, labels, noise = model
    dense = CleanJiT(tiny_config(variant='baseline', block_depths=(2,), patch_bottleneck=8, in_context_start=1)).eval()
    with pytest.raises(ValueError):
        sample(dense, labels, noise, steps=1, group_guidance={})
