import pytest
import torch
from hnet.model import CleanJiT
from denoiser import Objective, DualEMA, sample
from conftest import tiny_config, legacy_config, equal_tree


@pytest.mark.parametrize('thresholds', [(.6, .75), (-1., -1.), (1.1, 1.1)])
def test_legacy_initialization_forward_all_gradients_and_adam(legacy, thresholds):
    old_module, old_objective = legacy
    c = tiny_config(thresholds=thresholds)
    torch.manual_seed(42)
    old = old_objective.Objective(legacy_config(c, old_module))
    old_rng = torch.get_rng_state()
    torch.manual_seed(42)
    new = Objective(c)
    assert torch.equal(old_rng, torch.get_rng_state())
    equal_tree(old.state_dict(), new.state_dict())
    # Nonzero output head: otherwise the zero-initialized network hides defects.
    with torch.no_grad():
        old.net.final_layer.linear.weight.normal_(std=.02)
        old.net.final_layer.linear.bias.normal_(std=.01)
    new.load_state_dict(old.state_dict())
    images, labels = torch.randn(3, 3, 16, 16), torch.tensor([0, 6, 10])
    stochastic = (torch.tensor([.1, .5, .95]), torch.randn_like(images), torch.tensor([False, True, False]))
    optimizers = [torch.optim.AdamW(m.parameters(), lr=2e-4, betas=(.9, .95), weight_decay=0.) for m in (old, new)]
    emas = [DualEMA(m) for m in (old, new)]
    traces = [[], []]
    handles = []
    for model, records in zip((old, new), traces):
        for router in (model.net.router, model.net.router2):
            def capture(_module, _args, output, records=records):
                compact, assignment, valid, structure, soft = output
                records.append([x.detach().clone() for x in (compact, assignment, valid, structure.groups, structure.counts, soft)])
            handles.append(router.register_forward_hook(capture))
    for update in range(3):
        traces[0].clear(); traces[1].clear()
        outputs = [m(images, labels, stochastic=stochastic) for m in (old, new)]
        equal_tree(traces[0], traces[1], 'routing')
        for key in ('loss', 'generation', 'budget', 'budget1', 'budget2', 'hard_fraction', 'hard_fraction2'):
            equal_tree(outputs[0][key], outputs[1][key], key)
        for result in outputs: result['loss'].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in new.net.router.parameters())
        for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters()):
            assert name == other
            equal_tree(p.grad, q.grad, 'gradient.' + name)
        for model, optimizer, ema in zip((old, new), optimizers, emas):
            optimizer.step(); optimizer.zero_grad(set_to_none=True); ema.update(model)
        equal_tree(old.state_dict(), new.state_dict(), 'post_adam')
        equal_tree(optimizers[0].state_dict(), optimizers[1].state_dict(), 'adam')
        equal_tree(emas[0].state_dict(), emas[1].state_dict(), 'ema')
    for h in handles: h.remove()
    old.eval(); new.eval()
    with torch.no_grad():
        equal_tree(sample(old.net, labels, stochastic[1], steps=3), sample(new.net, labels, stochastic[1], steps=3), 'sampling')


def test_train_and_no_grad_readers_identical():
    model = CleanJiT(tiny_config()).eval()
    with torch.no_grad(): model.final_layer.linear.weight.normal_(std=.02)
    x, t, y = torch.randn(2, 3, 16, 16), torch.tensor([.3, .8]), torch.tensor([1, 11])
    prediction, aux = model(x, t, y, return_aux=True)
    with torch.no_grad(): sampled, sampled_aux = model(x, t, y, return_aux=True)
    equal_tree(prediction, sampled)
    for key in ('hard_counts', 'hard_counts2', 'soft_counts', 'soft_counts2'):
        equal_tree(aux[key], sampled_aux[key], key)
    with torch.no_grad(): equal_tree(prediction, model(x, t, y, return_aux=False))


def test_dense_initialization_and_nonzero_upstream_gradients():
    from model_jit import JiT
    from hnet.config import ModelConfig
    c = ModelConfig(variant='baseline', image_size=16, patch_size=4, width=32,
                    heads=4, classes=11, block_depths=(4,), in_context_len=3,
                    in_context_start=1, patch_bottleneck=8)
    torch.manual_seed(42)
    reference = JiT(input_size=16, patch_size=4, hidden_size=32, num_heads=4, depth=4,
                    num_classes=11, in_context_len=3, in_context_start=1, bottleneck_dim=8)
    rng = torch.get_rng_state()
    torch.manual_seed(42); actual = CleanJiT(c)
    assert torch.equal(rng, torch.get_rng_state())
    original = dict(reference.named_parameters())
    with torch.no_grad():
        # Exercise adaLN gates and the output head, not just zero predictions.
        for name, p in original.items():
            if 'adaLN_modulation.1' in name or 'final_layer.linear' in name:
                p.normal_(std=.02)
        for name, p in actual.named_parameters():
            source = name.replace('x_embedder.0.', 'x_embedder.proj1.').replace('x_embedder.1.', 'x_embedder.proj2.')
            p.copy_(original[source])
    x, t, y = torch.randn(2, 3, 16, 16), torch.tensor([.2, .8]), torch.tensor([2, 11])
    expected, result = reference(x, t, y), actual(x, t, y)
    # Upstream multiplies the logits by the scale; the old execution wrapper
    # divides by sqrt(head_dim). Dense parity uses a declared FP32 tolerance.
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
    expected.square().mean().backward(); result.square().mean().backward()
    for name, p in actual.named_parameters():
        source = name.replace('x_embedder.0.', 'x_embedder.proj1.').replace('x_embedder.1.', 'x_embedder.proj2.')
        torch.testing.assert_close(p.grad, original[source].grad, atol=2e-7, rtol=1e-4, msg=name)
