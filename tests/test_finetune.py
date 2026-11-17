"""Synthetic small-model SFT transition, real AdamW/EMAs and subsequent resume."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from denoiser import DualEMA
from hnet.checkpoints import Checkpoints, assert_state_equal, canonical, rng_state, sha256_file
from main_t2i import evaluations_due, learning_rate
from t2i.data import UpdateBatches
from t2i.diffusion import objective, training_randomness
from t2i.finetune import data_offset, phase_cursor, read_parent, restore_parent
from t2i.model import TextToImage
from test_t2i import tiny, batch

ROOT = Path(__file__).resolve().parents[1]


def recipe(name):
    return json.loads((ROOT / f'configs/t2i/training/{name}.json').read_text())


def objects(config, training):
    model = TextToImage(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['lr'], betas=tuple(training['betas']),
                                  eps=training['eps'], weight_decay=training['weight_decay'])
    ema = DualEMA(model, tuple(training['ema_decays']))
    generator = torch.Generator().manual_seed(40)
    return model, optimizer, ema, generator


def update(config, model, optimizer, ema, generator):
    images, context, mask, _ = batch(config)
    # Exercise all RNG streams that a full recovery checkpoint promises to keep.
    random.random(); np.random.rand(); torch.rand(1); torch.rand(1, generator=generator)
    t, noise, drop = training_randomness(42, 250000 * 1024, len(images), images.shape[1:], 'cpu')
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = objective(model, images, context, mask, t, noise, drop, config)
    loss.backward(); optimizer.step(); ema.update(model)
    return loss.detach().clone()


@pytest.fixture
def parent(tmp_path):
    torch.manual_seed(7); random.seed(7); np.random.seed(7)
    config, pretrain, sft = tiny(), recipe('cc12m_250k'), recipe('sft40k')
    source = objects(config, pretrain)
    update(config, *source)
    old_identity = canonical(dict(world_size=1, global_batch=1024, model=asdict(config), recipe=pretrain))
    identity = canonical(dict(old_identity, recipe=sft))
    store = Checkpoints(tmp_path / 'pretrain', old_identity, milestone_name='checkpoint-u{global_step:07d}.pt')
    # Synthetic boundary metadata: one actual small-model update, not a 250k training claim.
    progress = dict(epoch=0, update_in_epoch=250000, global_step=250000,
                    images_seen=250000 * 1024, training_seconds=1.)
    model, optimizer, ema, generator = source
    store.save(model, optimizer, ema, progress, [rng_state(generator)], milestone=True)
    path = tmp_path / 'pretrain/checkpoint-u0250000.pt'
    return config, sft, source, identity, path


def test_transition_preserves_raw_adam_both_emas_rng_and_next_update(parent, tmp_path):
    config, sft, source, identity, path = parent
    digest = sha256_file(path)
    expected_loss = update(config, *source)
    expected = [deepcopy(obj.state_dict()) for obj in source[:3]]
    expected_rng = rng_state(source[3])
    target = objects(config, sft)
    state, lineage = read_parent(path, identity)
    assert lineage['checkpoint_sha256'] == digest
    progress = restore_parent(state, *target, 0, sft)
    assert progress['global_step'] == 250000 and progress['phase_update'] == progress['phase_next_position'] == 0
    assert_state_equal(rng_state(target[3]), state['rank_rng'][0])
    assert torch.equal(update(config, *target), expected_loss)
    for obj, want in zip(target[:3], expected):
        assert_state_equal(obj.state_dict(), want)
    assert_state_equal(rng_state(target[3]), expected_rng)
    # Save inside SFT, then prove the next update survives a full restart as well.
    progress.update(global_step=250001, update_in_epoch=250001, images_seen=250001 * 1024,
                    **phase_cursor(250001, sft))
    identity['finetune_parent'] = lineage
    store = Checkpoints(tmp_path / 'sft', identity)
    store.save(*target[:3], progress, [rng_state(target[3])])
    next_loss = update(config, *target)
    next_state = [deepcopy(obj.state_dict()) for obj in target[:3]]
    restored = objects(config, sft)
    assert store.load(*restored) == progress
    assert torch.equal(update(config, *restored), next_loss)
    for obj, want in zip(restored[:3], next_state):
        assert_state_equal(obj.state_dict(), want)
    assert sha256_file(path) == digest  # Parent is never rewritten.


def test_transition_rejects_wrong_parent_or_changed_training(parent, tmp_path):
    _, _, _, identity, path = parent
    changes = [lambda d: d['model'].update(budget_weight=.002),
               lambda d: d['recipe'].update(lr=1e-4),
               lambda d: d.update(world_size=2)]
    for change in changes:
        wrong = deepcopy(identity)
        change(wrong)
        with pytest.raises(ValueError):
            read_parent(path, wrong)
    state = torch.load(path, weights_only=False)
    state['progress']['global_step'] = 249999
    bad = tmp_path / 'early.pt'
    torch.save(state, bad)
    with pytest.raises(ValueError, match='boundary'):
        read_parent(bad, identity)
    bad = tmp_path / 'ema.safetensors'
    bad.write_bytes(b'not full recovery')
    with pytest.raises(ValueError, match='inference EMA'):
        read_parent(bad, identity)


def test_sft_data_reset_rank_coverage_and_global_noise_positions():
    sft, pretrain = recipe('sft40k'), recipe('cc12m_250k')
    offset, update_number = data_offset(sft), 250003
    phase = phase_cursor(update_number, sft)
    assert phase == dict(phase_update=3, phase_next_position=3072)
    assert data_offset(pretrain) == 0 and phase_cursor(42, pretrain) == {}
    for world in (8, 16):
        batches = [next(iter(UpdateBatches(update_number - offset, update_number - offset + 1,
                                          1024, rank, world))) for rank in range(world)]
        assert sum(batches, []) == list(range(3072, 4096))
    assert learning_rate(sft, offset) == learning_rate(pretrain, offset) == 4e-4
    assert evaluations_due(sft, 260000) == [] and evaluations_due(sft, 290000) == []
    a = training_randomness(42, update_number * 1024, 4, (3, 2, 2), 'cpu')
    b = training_randomness(42, phase['phase_next_position'], 4, (3, 2, 2), 'cpu')
    assert not torch.equal(a[1], b[1])  # Restarting the data must not reuse early pretraining noise.
    with pytest.raises(ValueError, match='precedes'):
        phase_cursor(249999, sft)
