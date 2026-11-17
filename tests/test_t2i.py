"""CPU checks for the text-to-image path (tiny shapes, reference attention, FP64 CPU trees)."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def tiny(variant='hnet', **changes):
    from t2i.config import T2IConfig
    args = dict(variant=variant, image_size=32, patch_size=8, width=32, heads=2, mlp_ratio=2., text_width=16,
                text_length=6, text_depth=1, attention_backend='reference', compact_batching='padded',
                routing_backend='numba', routing_workers=1, pad_multiple=4, routing_transfer='legacy',
                routing_validation='cpu', tree_precision='fp64')
    if variant == 'hnet':
        args.update(block_depths=(1, 1, 2, 1, 1), thresholds=(.2, .2), reference_depth=5, reference_bottleneck=8)
    else:
        args.update(block_depths=(3,), patch_bottleneck=8)
    args.update(changes)
    return T2IConfig(**args)


def batch(config, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, config.image_size, config.image_size, generator=g) * 2 - 1
    context = torch.randn(n, config.text_length, config.text_width, generator=g)
    mask = torch.zeros(n, config.text_length, dtype=torch.long)
    for i, valid in enumerate([config.text_length, 2, 0][:n]):
        mask[i, :valid] = 1
    t = torch.rand(n, generator=g)
    return images, context, mask, t


@pytest.mark.parametrize('variant', ['baseline', 'hnet'])
def test_forward_shapes_and_gradients(variant):
    from t2i.diffusion import objective
    from t2i.model import TextToImage
    config = tiny(variant)
    torch.manual_seed(0)
    model = TextToImage(config)
    images, context, mask, t = batch(config)
    noise, drop = torch.randn_like(images), torch.tensor([False, False, True])
    loss, stats, aux = objective(model, images, context, mask, t, noise, drop, config)
    assert torch.isfinite(loss) and stats['generation_loss'] > 0
    loss.backward()
    missing = {n for n, p in model.named_parameters() if p.grad is None}
    # The last block's text update never reaches the image output; the released dense
    # model also keeps its unused time/pooled modules (hence DDP find_unused_parameters).
    last = f'blocks.{config.depth - 1}.'
    unused = {n for n, _ in model.named_parameters()
              if n.startswith((last + 'txt_norm2', last + 'txt_proj', last + 'txt_mlp'))}
    if variant == 'baseline':
        unused |= {n for n, _ in model.named_parameters() if n.startswith(('t_embed.', 'pooled_embed.'))}
    assert missing == unused
    if variant == 'hnet':
        assert torch.equal(aux['population1'], torch.tensor([16 + 6, 16 + 2, 16]))  # dropped caption: image only
        assert (aux['hard_counts'] >= 1).all() and (aux['hard_counts'] <= aux['population1']).all()
        assert torch.equal(aux['population2'], aux['hard_counts'])


@pytest.mark.parametrize('variant', ['baseline', 'hnet'])
def test_caption_padding_is_inert(variant):
    """Padded positions are replaced by the learned null token: their T5 values never matter."""
    from t2i.model import TextToImage
    config = tiny(variant)
    torch.manual_seed(0)
    model = TextToImage(config).eval()
    images, context, mask, t = batch(config)
    changed = context.clone()
    changed[~mask.bool()] = 100.
    with torch.no_grad():
        a, _ = model(images, context, mask, time=t)
        b, _ = model(images, changed, mask, time=t)
    assert torch.equal(a, b)


def test_budget_does_not_change_prediction():
    from t2i.model import TextToImage
    config = tiny()
    torch.manual_seed(0)
    model = TextToImage(config).eval()
    images, context, mask, t = batch(config)
    with torch.no_grad():
        a, _ = model(images, context, mask, time=t, need_budget=True)
        b, _ = model(images, context, mask, time=t, need_budget=False)
    assert torch.equal(a, b)


def test_hit_shares_released_dense_initialization():
    """Outer stages and the text path come from the same seed's released dense model."""
    from third_party.minit2i_torch.model import MMJiTB32Text2
    from t2i.model import TextToImage, SharedCompactBlock
    config = tiny()
    torch.manual_seed(7)
    model = TextToImage(config)
    torch.manual_seed(7)
    reference = MMJiTB32Text2(image_size=32, patch_size=8, hidden_size=32, t5_hidden_size=16, depth_double=5,
                              text_preamble_depth=1, num_heads=2, head_dim=16, mlp_ratio=2., pca_channels=8,
                              attention_impl='einsum', rope_style='jax')
    assert model.shared_blocks == [0, 1, 2, 4, 5]
    for old, new in enumerate(model.shared_blocks):
        block = model.blocks[new]
        if isinstance(block, SharedCompactBlock):
            assert torch.equal(block.qkv.weight, reference.blocks[old].img_qkv.weight)
        else:
            assert all(torch.equal(a, b) for a, b in zip(block.state_dict().values(),
                                                           reference.blocks[old].state_dict().values()))
    assert torch.equal(model.mask_token, reference.mask_token)
    assert torch.equal(model.final.weight, reference.final.weight)


def test_core_extension_keeps_base_initialization():
    """Extra core blocks come from their own seed; every other module matches the base layout."""
    from t2i.model import TextToImage
    torch.manual_seed(0)
    base = TextToImage(tiny(block_depths=(1, 1, 2, 1, 1)))
    torch.manual_seed(0)
    extended = TextToImage(tiny(block_depths=(1, 1, 3, 1, 1), core_extension_base_depth=6, core_extension_seed=7))
    same = lambda a, b: all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()
    for new, old in ((0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5)):
        assert same(extended.blocks[new].state_dict(), base.blocks[old].state_dict())
    outside = lambda m: {k: v for k, v in m.state_dict().items() if not k.startswith('blocks.')}
    assert same(outside(extended), outside(base))
    torch.manual_seed(123)
    other = TextToImage(tiny(block_depths=(1, 1, 3, 1, 1), core_extension_base_depth=6, core_extension_seed=7))
    assert same(other.blocks[4].state_dict(), extended.blocks[4].state_dict())  # seeded, not global RNG
    assert not same(other.blocks[3].state_dict(), extended.blocks[3].state_dict())


def test_token_packing_round_trip():
    from t2i.tokens import pack_valid_tokens
    features = torch.randn(2, 5, 3)
    eligible = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    packed, layout = pack_valid_tokens(features, eligible)
    assert layout.lengths == (3, 5) and packed.shape == (2, 5, 3)
    assert torch.equal(packed[0, :3], features[0, [0, 2, 3]]) and torch.equal(packed[0, 3:], torch.zeros(2, 3))
    restored = layout.restore(packed)
    assert torch.equal(restored[eligible], features[eligible]) and torch.equal(restored[~eligible], torch.zeros(2, 3))


def test_randomness_is_positional():
    """Any GPU count or microbatch split sees the same per-sample t, noise and caption dropout."""
    from t2i.diffusion import training_randomness
    shape = (3, 4, 4)
    whole = training_randomness(42, 100, 300, shape, 'cpu')
    for splits in ([100, 200], [28, 100, 172], [1, 299]):
        start, parts = 100, []
        for size in splits:
            parts.append(training_randomness(42, start, size, shape, 'cpu'))
            start += size
        for i in range(3):
            assert torch.equal(torch.cat([p[i] for p in parts]), whole[i])
    t, noise, drop = training_randomness(42, 0, 4096, shape, 'cpu')
    assert 0 < t.min() and t.max() < 1 and abs(noise.std().item() - 2) < .05 and .07 < drop.float().mean() < .13


def test_update_batches_and_evaluation_cohort():
    from t2i.data import UpdateBatches
    from t2i.evaluation import canonical_batches
    rows = [list(UpdateBatches(3, 5, 8, rank, 2)) for rank in range(2)]
    assert rows[0] == [[24, 25, 26, 27], [32, 33, 34, 35]] and rows[1] == [[28, 29, 30, 31], [36, 37, 38, 39]]
    for world in (1, 2, 4, 8):
        seen = sorted(i for rank in range(world) for ids in canonical_batches(300, 32, 4, world, rank) for i in ids)
        assert seen == list(range(300))
    # Batch membership is independent of the physical world size.
    one = {tuple(ids) for ids in canonical_batches(300, 32, 4, 1, 0)}
    assert one == {tuple(ids) for r in range(8) for ids in canonical_batches(300, 32, 4, 8, r)}


def test_evaluation_schedule():
    sys.path.insert(0, str(ROOT))
    from main_t2i import evaluations_due, learning_rate
    recipe = json.loads((ROOT / 'configs/t2i/training/cc12m_250k.json').read_text())
    assert evaluations_due(recipe, 0) == [] and evaluations_due(recipe, 9999) == []
    assert evaluations_due(recipe, 10000) == [(10000, .9995)]
    assert evaluations_due(recipe, 50000) == [(30000, .9995), (30000, .99995)]
    assert evaluations_due(recipe, 75000) == [(30000, .9995)]
    assert learning_rate(recipe, 0) == 1e-6 and learning_rate(recipe, 5000) == learning_rate(recipe, 9000) == 4e-4


def test_config_rules(tmp_path):
    from t2i.config import load_t2i_config
    with pytest.raises(ValueError):
        tiny(patch_bottleneck=8)
    with pytest.raises(ValueError):
        tiny(block_depths=(1, 1, 2, 2, 1))
    with pytest.raises(ValueError):
        tiny('baseline', thresholds=(.5, .5))
    model = json.loads((ROOT / 'configs/t2i/models/hit_b16.json').read_text())
    path = tmp_path / 'uncalibrated.json'
    path.write_text(json.dumps(dict(model, thresholds=None)))
    with pytest.raises(ValueError):
        load_t2i_config(path)
    config, _, _ = load_t2i_config(path, calibrating=True)
    assert config.image_tokens + config.text_length == 1280 and config.depth == 22


def test_released_configs_match_published_sizes():
    from t2i.config import load_t2i_config
    from t2i.model import TextToImage
    runtime = ROOT / 'configs/t2i/runtime/reference.json'
    counts = {}
    for name in ('minit2i_b16', 'hit_b16', 'hit_b16_d27'):
        config, _, _ = load_t2i_config(ROOT / f'configs/t2i/models/{name}.json', None, runtime, calibrating=True)
        if config.variant == 'hnet' and config.thresholds is None:
            config = replace(config, thresholds=(.5, .5))
        with torch.device('meta'):
            counts[name] = sum(p.numel() for p in TextToImage(config).parameters())
    assert counts == {'minit2i_b16': 258_122_880, 'hit_b16': 218_775_040, 'hit_b16_d27': 254_188_160}


def test_threshold_choice_hits_ratio():
    sys.path.insert(0, str(ROOT / 'scripts'))
    from calibrate_t2i import choose_threshold
    rng = np.random.default_rng(0)
    populations = rng.integers(20, 40, size=64)
    scores = [np.sort(rng.random(n - 1))[::-1] for n in populations]
    threshold, counts = choose_threshold(scores, populations, 2., 'fp64')
    assert abs(counts.sum() - populations.sum() / 2) <= 2


@pytest.fixture(scope='module')
def legacy_t2i():
    location = os.environ.get('HIT_T2I_LEGACY_SOURCE')
    if not location:
        pytest.skip('Set HIT_T2I_LEGACY_SOURCE to a codex/minit2i-b16-b300-fp32 checkout for cross-repo parity')
    sys.path.append(location)
    from minit2i_hnet import model, diffusion
    return model, diffusion


@pytest.mark.parametrize('variant', ['baseline', 'hnet'])
def test_matches_legacy_on_cpu(legacy_t2i, variant):
    """Bitwise CPU equality with the original implementation: weights, loss and every gradient."""
    legacy_model, legacy_diffusion = legacy_t2i
    from t2i.diffusion import objective
    from t2i.model import TextToImage
    common = dict(image_size=32, patch_size=16, width=32, heads=2, mlp_ratio=2., text_width=16, text_length=4,
                  text_depth=2)
    if variant == 'hnet':
        config = tiny(block_depths=(3, 3, 10, 3, 3), thresholds=(.3, .3), reference_depth=17,
                      reference_bottleneck=128, **common)
        old = legacy_model.ModelConfig(variant='hnet', depth=22, encoder_depth=3, decoder_depth=3, intermediate_depth=3,
                                       stages=2, bottleneck=0, threshold=.3, second_threshold=.3, target_ratio=2.,
                                       second_ratio=2., routing_workers=1, routing_backend='numba', compact_backend='padded',
                                       attention='einsum', initialization='released-shared', time_token=True,
                                       group_modalities='joint', joint_padding='exclude', tree_precision='fp64', **common)
    else:
        config = tiny('baseline', block_depths=(5,), patch_bottleneck=8, **common)
        old = legacy_model.ModelConfig(variant='dense', depth=5, bottleneck=8, budget_weight=0., attention='einsum',
                                       compact_backend='padded', initialization='released-pytorch', **common)
    torch.manual_seed(42)
    clean = TextToImage(config)
    torch.manual_seed(42)
    reference = legacy_model.MiniT2IHNet(old)
    assert list(clean.state_dict()) == list(reference.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(clean.state_dict().values(), reference.state_dict().values()))
    images, context, mask, t = batch(config, seed=3)
    noise, drop = torch.randn_like(images), torch.tensor([False, False, True])
    loss_a, stats_a, _ = objective(clean, images, context, mask, t, noise, drop, config, a_min=.05)
    loss_b, stats_b, _ = legacy_diffusion.loss(reference, images, context, mask, t, noise, drop, old.budget_weight,
                                               old.target_ratio, a_min=.05)
    assert torch.equal(loss_a, loss_b) and torch.equal(stats_a['generation_loss'], stats_b['generation_loss'])
    loss_a.backward(); loss_b.backward()
    for (name, a), b in zip(clean.named_parameters(), reference.parameters()):
        assert (a.grad is None) == (b.grad is None), name
        assert a.grad is None or torch.equal(a.grad, b.grad), name
