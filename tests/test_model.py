import torch
from hnet.model import CleanJiT
from conftest import tiny_config, equal_tree


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
