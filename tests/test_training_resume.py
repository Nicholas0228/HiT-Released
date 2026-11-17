import json
import random
import numpy as np
import pytest
import torch
from denoiser import Objective, DualEMA
from hnet.checkpoints import Checkpoints, rng_state, assert_state_equal
from hnet.data import ResumeShardSampler
from conftest import tiny_config


def test_full_adam_ema_rng_cursor_resume_and_retention(tmp_path):
    c = tiny_config()
    torch.manual_seed(3); random.seed(3); np.random.seed(3)
    model = Objective(c)
    with torch.no_grad(): model.net.final_layer.linear.weight.normal_(std=.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .95), weight_decay=0.)
    ema = DualEMA(model)
    generator = torch.Generator().manual_seed(40001)
    images, labels = torch.randn(2, 3, 16, 16), torch.tensor([1, 9])
    identity = {'world_size': 1, 'architecture': (1, 1, 2, 1, 1)}
    store = Checkpoints(tmp_path, identity)
    def update():
        optimizer.zero_grad(set_to_none=True)
        result = model(images, labels, generator)
        result['loss'].backward(); optimizer.step(); ema.update(model)
        return result['loss'].detach().clone()
    update()
    cursor = dict(epoch=20, update_in_epoch=0, global_step=1, images_seen=2)
    first = store.save(model, optimizer, ema, cursor, [rng_state(generator)], milestone=True)
    expected_loss = update()
    expected = {k:v.detach().clone() for k,v in model.state_dict().items()}
    expected_rng = rng_state(generator)
    # Reload through JSON identity (tuples become lists), then repeat the next update.
    recovered = Checkpoints(tmp_path, json.loads(json.dumps(identity)))
    assert recovered.load(model, optimizer, ema, generator) == cursor
    assert torch.equal(update(), expected_loss)
    assert_state_equal(model.state_dict(), expected)
    assert_state_equal(rng_state(generator), expected_rng)
    for step in (2, 3, 4):
        recovered.save(model, optimizer, ema, dict(cursor, global_step=step), [rng_state(generator)])
    assert len(list(tmp_path.glob('state-*.pt'))) == 2
    assert (tmp_path / 'checkpoint-e0020.pt').exists()
    pointer = (tmp_path / 'latest.json').read_bytes()
    # A failed write may leave an inspectable candidate, never retire verified data.
    original = torch.save
    try:
        torch.save = lambda *a, **k: (_ for _ in ()).throw(OSError('injected failed write'))
        with pytest.raises(OSError): recovered.save(model, optimizer, ema, cursor, [rng_state(generator)])
    finally: torch.save = original
    assert (tmp_path / 'latest.json').read_bytes() == pointer
    with pytest.raises(RuntimeError, match='identity'):
        Checkpoints(tmp_path, dict(identity, world_size=2)).load(model, optimizer, ema, generator)


def test_loader_cursor_and_rank_coverage():
    full = [list(ResumeShardSampler(106, r, 4, 4, 1, 0, 3)) for r in range(4)]
    resumed = [list(ResumeShardSampler(106, r, 4, 4, 1, 0, 3, 2)) for r in range(4)]
    assert all(b == a[8:] for a,b in zip(full, resumed))
    flattened = [x for rank in full for x in rank]
    assert len(flattened) == len(set(flattened)) == 96
