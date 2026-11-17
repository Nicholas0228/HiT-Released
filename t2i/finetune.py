"""SFT data cursor and a full-state transition from CC12M pretraining.

The data sequence restarts at position zero. The optimizer update, LR, diffusion
randomness and both EMAs continue at the pretraining boundary.
"""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import torch

from hnet.checkpoints import assert_state_equal, canonical, restore_rng, sha256_file


CONTINUED_RECIPE = ('seed', 'global_batch', 'lr', 'initial_lr', 'warmup_updates',
                    'betas', 'eps', 'weight_decay', 'ema_decays', 'a_min')


def data_offset(recipe):
    offset = recipe.get('data_update_offset', 0)
    if offset:
        if recipe.get('phase') != 'sft40k' or offset != 250000 or recipe['updates'] != 290000:
            raise ValueError('SFT40k runs from global update 250000 through 290000')
    elif recipe.get('phase'):
        raise ValueError('Fine-tuning requires its pretraining data offset')
    return offset


def phase_cursor(update, recipe):
    offset = data_offset(recipe)
    if update < offset:
        raise ValueError('The checkpoint precedes the fine-tuning boundary')
    return dict(phase_update=update - offset,
                phase_next_position=(update - offset) * recipe['global_batch']) if offset else {}


def load_sft_inputs(path, recipe):
    """Check the small indexes at startup; preparation verifies all image shards."""
    inputs = json.loads(Path(path).read_text())
    if inputs.get('schema') != 1 or inputs.get('phase') != 'sft40k' or not inputs.get('complete'):
        raise ValueError('Use the complete inputs.json from prepare_t2i_finetune.py')
    fingerprint = inputs['fingerprint']
    for key in ('records', 'captions', 'sequence'):
        if sha256_file(inputs[key]) != fingerprint[key + '_sha256']:
            raise ValueError('Fine-tuning input hash mismatch: ' + key)
    sequence = np.load(inputs['sequence'], mmap_mode='r', allow_pickle=False)
    required = (recipe['updates'] - data_offset(recipe)) * recipe['global_batch']
    if sequence.ndim != 1 or sequence.dtype != np.int32 or len(sequence) != required:
        raise ValueError('The SFT sequence must contain exactly 40000 global batches')
    if not Path(inputs['encoder']).is_dir() or any(not Path(p).is_file() for p in inputs['shards']):
        raise FileNotFoundError('A fine-tuning shard or the text encoder is missing')
    return inputs


def read_parent(path, identity):
    """Validate before constructing/restoring a training model. Never accept an EMA-only export."""
    path = Path(path).resolve()
    digest = sha256_file(path)
    if path.suffix != '.pt':
        raise ValueError('SFT continuation needs a full .pt training checkpoint, not an inference EMA export')
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    required = {'schema', 'identity', 'model', 'optimizer', 'ema', 'progress', 'rank_rng'}
    if not required <= state.keys() or state['schema'] != 1:
        raise ValueError('The parent must contain raw weights, AdamW, both EMAs and rank RNG state')
    old = state['identity']
    if not {'model', 'recipe', 'world_size', 'global_batch'} <= old.keys():
        raise ValueError('Expected a full training checkpoint written by main_t2i.py')
    recipe, parent_recipe = identity['recipe'], old['recipe']
    boundary = data_offset(recipe)
    if not boundary or data_offset(parent_recipe) or parent_recipe['updates'] != boundary:
        raise ValueError('The parent must be the completed 250k pretraining phase')
    progress = state['progress']
    if (progress.get('global_step') != boundary or progress.get('update_in_epoch') != boundary
            or progress.get('images_seen') != boundary * recipe['global_batch']):
        raise ValueError('The parent data cursor is not at the 250k boundary')
    # Backend choices may change; architecture, thresholds and loss coefficients may not.
    from t2i.config import RUNTIME_KEYS
    architecture = lambda m: {k: v for k, v in m.items() if k not in RUNTIME_KEYS}
    if canonical(architecture(old['model'])) != canonical(architecture(identity['model'])):
        raise ValueError('SFT must preserve the parent architecture, grouping thresholds and budget weights')
    for key in CONTINUED_RECIPE:
        if canonical(parent_recipe[key]) != canonical(recipe[key]):
            raise ValueError('SFT changes a continued training setting: ' + key)
    if old['world_size'] != identity['world_size'] or len(state['rank_rng']) != identity['world_size']:
        raise ValueError('Use the parent GPU count to retain every rank RNG state')
    if old['global_batch'] != identity['global_batch']:
        raise ValueError('SFT must preserve the global batch')
    if tuple(state['ema']['decays']) != tuple(recipe['ema_decays']):
        raise ValueError('The parent EMA decays differ from the SFT recipe')
    if not state['optimizer'].get('state'):
        raise ValueError('The parent has no AdamW moments')
    for group in state['optimizer']['param_groups']:
        for key, want in [('betas', tuple(recipe['betas'])), ('eps', recipe['eps']),
                          ('weight_decay', recipe['weight_decay']), ('lr', recipe['lr'])]:
            if group[key] != want:
                raise ValueError('Parent AdamW setting differs: ' + key)
    lineage = dict(checkpoint_sha256=digest, checkpoint_file=path.name, global_step=boundary,
                   parent_identity=old, continued=['raw_weights', 'adamw', 'both_emas', 'rank_rng',
                                                  'global_update', 'diffusion_noise_positions'],
                   reset=['fine_tuning_data_position'])
    return state, lineage


def restore_parent(state, model, optimizer, ema, generator, rank, recipe):
    """Copy every learned state, then expose the new phase's zero data cursor."""
    model.load_state_dict(state['model'], strict=True)
    if state['ema']['names'] != [name for name, _ in model.named_parameters()]:
        raise ValueError('Parent parameter ordering differs; cannot safely restore AdamW')
    optimizer.load_state_dict(state['optimizer'])
    ema.load_state_dict(state['ema'])
    restore_rng(state['rank_rng'][rank], generator)
    assert_state_equal(model.state_dict(), state['model'], 'model')
    assert_state_equal(optimizer.state_dict(), state['optimizer'], 'adamw')
    assert_state_equal(ema.state_dict(), state['ema'], 'both_emas')
    progress = deepcopy(state['progress'])
    progress.update(phase_cursor(progress['global_step'], recipe))
    return progress
