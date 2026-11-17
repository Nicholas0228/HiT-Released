"""CPU checks for the text-to-image benchmark protocol: sampler, seeds, output layout and manifests."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from t2i import benchmarks as B
from test_t2i import batch, tiny

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = ROOT / 'configs/t2i/sampling/hit_b16_d27.json'


@pytest.fixture(scope='module')
def hit():
    from t2i.model import TextToImage
    torch.manual_seed(0)
    model = TextToImage(tiny()).eval()
    with torch.no_grad():  # the zero-initialized output layer would make every prediction equal
        generator = torch.Generator().manual_seed(1)
        for parameter in model.parameters():
            parameter.add_(.05 * torch.randn(parameter.shape, generator=generator))
    _, context, mask, _ = batch(model.config, n=2)
    noise = torch.randn(2, 3, model.config.image_size, model.config.image_size,
                        generator=torch.Generator().manual_seed(3))
    return model, context, mask, noise


def test_group_guidance_sampler(hit):
    from t2i.diffusion import sample
    model, context, mask, noise = hit
    before = [r.threshold.clone() for r in (model.router, model.router2)]
    plain = sample(model, context, mask, noise, guidance=6.5, steps=3)
    # Unchanged thresholds in both branches reproduce batched classifier-free guidance.
    same = sample(model, context, mask, noise, guidance=6.5, steps=3, group_guidance={})
    assert torch.allclose(plain, same, atol=1e-5)
    guided = sample(model, context, mask, noise, guidance=6.5, steps=3,
                    group_guidance={'cond': (.6, .6), 'uncond': (.05, None)})
    assert not torch.allclose(plain, guided, atol=1e-3)
    assert all(torch.equal(r.threshold, b) for r, b in zip((model.router, model.router2), before))


def test_group_guidance_needs_hit():
    from t2i.diffusion import sample
    from t2i.model import TextToImage
    dense = TextToImage(tiny('baseline')).eval()
    _, context, mask, _ = batch(dense.config, n=1)
    with pytest.raises(ValueError):
        sample(dense, context, mask, torch.zeros(1, 3, 32, 32), steps=1, group_guidance={})


def test_released_sampler_follows_paper():
    recipe = json.loads(SAMPLING.read_text())
    model = json.loads((ROOT / 'configs/t2i/models/hit_b16_d27.json').read_text())
    assert (recipe['guidance'], recipe['steps'], recipe['a_min'], recipe['noise_std'], recipe['seed']) == (6.5, 100, .05, 2., 0)
    assert model['block_depths'] == [3, 3, 15, 3, 3]
    (c1, c2), (u1, u2), (s1, s2) = recipe['group_guidance']['cond'], recipe['group_guidance']['uncond'], model['thresholds']
    # A higher threshold merges less: 1.2x groups at both stages for the conditional branch,
    # 0.8x at stage 1 for the unconditional branch with its stage 2 unchanged.
    assert c1 > s1 and c2 > s2 and u1 < s1 and u2 == s2
    scores = recipe['reference_scores']
    five = [scores[k] for k in ('geneval', 'dpg', 'prism', 'corebench', 'spatial_geneval')]
    assert round(sum(five) / 5, 2) == scores['average']


def test_seed_rule_and_layout():
    assert B.batch_seed(0, 9, 0) == 9 * 1000 + 1_000_000
    assert B.canonical_batches(4) == [(0, 4)] and B.canonical_batches(1) == [(0, 1)]
    prompt = B.Prompt(12, '00012', 'a photo of a cow')
    rows = B.expected_entries('geneval', [prompt], seed=0)
    assert [r['path'] for r in rows] == [f'00012/samples/{k:05d}.png' for k in range(4)]
    assert {r['batch_seed'] for r in rows} == {12_000 + 4_000_000}
    grid = B.expected_entries('dpg', [B.Prompt(3, 'x', 'p')], seed=0)
    assert [r['box'] for r in grid] == [[0, 0, 512, 512], [512, 0, 1024, 512], [0, 512, 512, 1024], [512, 512, 1024, 1024]]
    judged = B.expected_entries('prism', [B.Prompt(5, 'style-005', 'p', 'style')], seed=0)
    assert [r['path'] for r in judged] == ['style/style-005/00000.png']


def write_generation(directory, prompts, benchmark='prism', seed=0):
    from PIL import Image
    images = directory / 'images'
    entries, files = [], {}
    for row in B.expected_entries(benchmark, prompts, seed):
        array = np.full((512, 512, 3), row['index'] * 40, np.uint8)
        path = images / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
        row['pixel_sha256'] = hashlib.sha256(array.tobytes()).hexdigest()
        files[row['path']] = B.sha256_file(path)
        entries.append(row)
    identity = {'dataset': {'benchmark': benchmark}, 'sampling': {'samples_per_prompt': B.SAMPLES[benchmark], 'seed': seed},
                'model': {'checkpoint_sha256': 'abc'}}
    B.atomic_json(directory / 'manifest.json', {'schema': B.GENERATION_SCHEMA, 'identity': identity,
                                                'identity_sha256': B.sha256_json(identity),
                                                'indices': [p.index for p in prompts], 'entries': entries, 'files': files})


def test_manifest_rejects_incomplete_or_wrong_outputs(tmp_path):
    prompts = [B.Prompt(0, 'style-000', 'p', 'style'), B.Prompt(1, 'style-001', 'q', 'style')]
    write_generation(tmp_path, prompts)
    B.verify_generation(tmp_path, {'model.checkpoint_sha256': 'abc'})
    with pytest.raises(B.ManifestError):
        B.verify_generation(tmp_path, {'model.checkpoint_sha256': 'other'})
    (tmp_path / 'images/style/style-001/00000.png').unlink()
    with pytest.raises(B.ManifestError):
        B.verify_generation(tmp_path)


@pytest.mark.skipif(not os.environ.get('BENCHMARK_ASSETS'), reason='set BENCHMARK_ASSETS to a fetch_assets.py root')
def test_prompt_sets_match_paper():
    root = Path(os.environ['BENCHMARK_ASSETS'])
    counts = {name: len(B.load_prompts(name, root)) for name in B.BENCHMARKS}
    assert counts == {'geneval': 553, 'dpg': 1065, 'prism': 700, 'corebench': 1080, 'spatial_geneval': 1230}
