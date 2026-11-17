"""Portable verified full-state recovery, with bounded owned-file retention.

Cloud transport is deliberately outside the model/trainer. Archive immutable
milestones or verified recovery receipts using the deployment's own sidecar.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), generator=generator.get_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); generator.set_state(state['generator'])
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'])


def assert_state_equal(a, b, path='state'):
    if torch.is_tensor(a):
        if not torch.equal(a.detach().cpu(), b.detach().cpu()): raise RuntimeError('State differs: ' + path)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b): raise RuntimeError('State differs: ' + path)
    elif isinstance(a, dict):
        if a.keys() != b.keys(): raise RuntimeError('State keys differ: ' + path)
        for key in a: assert_state_equal(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b): raise RuntimeError('State length differs: ' + path)
        for i, (x, y) in enumerate(zip(a, b)): assert_state_equal(x, y, path + f'.{i}')
    elif a != b:
        raise RuntimeError('State differs: ' + path)


class Checkpoints:
    def __init__(self, directory, identity, keep=2, milestone_name='checkpoint-e{epoch:04d}.pt'):
        if keep < 2: raise ValueError('Retain at least the previous verified recovery state')
        self.directory, self.identity, self.keep = Path(directory), canonical(identity), keep
        self.milestone_name = milestone_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest = self.directory / 'latest.json'

    def save(self, model, optimizer, ema, progress, rank_rng, milestone=False):
        state = dict(schema=1, identity=self.identity, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), ema=ema.state_dict(),
                     progress=dict(progress), rank_rng=rank_rng)
        stem = f"state-u{progress['global_step']:09d}-{uuid.uuid4().hex[:12]}"
        candidate = self.directory / (stem + '.partial')
        path = self.directory / (stem + '.pt')
        with candidate.open('xb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        # Actual deserialization and tensor comparison before advancing latest.
        recovered = torch.load(candidate, map_location='cpu', weights_only=False)
        assert_state_equal(state, recovered)
        digest = sha256_file(candidate)
        os.replace(candidate, path)
        receipt = dict(schema=1, identity=self.identity, file=path.name, sha256=digest,
                       bytes=path.stat().st_size, progress=dict(progress), verified=True,
                       milestone=bool(milestone), created_ns=time.time_ns())
        atomic_json(self.directory / (stem + '.json'), receipt)
        if milestone:
            target = self.directory / self.milestone_name.format(**progress)
            # Never overwrite a permanent checkpoint, including on a later resume.
            os.link(path, target)
            atomic_json(target.with_suffix('.json'), dict(receipt, file=target.name))
        atomic_json(self.latest, receipt)
        self.retire_owned()
        return receipt

    def retire_owned(self):
        records = []
        for p in self.directory.glob('state-u*.json'):
            item = json.loads(p.read_text())
            if canonical(item.get('identity')) == self.identity and item.get('verified') and not item.get('retired'):
                records.append((p, item))
        records.sort(key=lambda pair: (pair[1]['progress']['global_step'], pair[1]['created_ns']))
        latest_file = json.loads(self.latest.read_text())['file'] if self.latest.exists() else None
        for p, item in records[:-self.keep]:
            target = self.directory / item['file']
            if target.parent.resolve() != self.directory.resolve() or target.is_symlink():
                raise RuntimeError('Retention path escaped checkpoint directory')
            if target.name == latest_file: continue
            if target.exists():
                if sha256_file(target) != item['sha256']: raise RuntimeError('Modified checkpoint preserved')
                target.unlink()  # Permanent hard links are unaffected.
            item['retired'] = True
            atomic_json(p, item)

    def load(self, model, optimizer, ema, generator, rank=0, receipt_path=None):
        receipt = json.loads(Path(receipt_path or self.latest).read_text())
        if canonical(receipt['identity']) != self.identity or not receipt.get('verified'):
            raise RuntimeError('Checkpoint identity mismatch')
        path = self.directory / receipt['file']
        if path.parent.resolve() != self.directory.resolve() or sha256_file(path) != receipt['sha256']:
            raise RuntimeError('Checkpoint path/hash mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['schema'] != 1 or canonical(state['identity']) != self.identity:
            raise RuntimeError('Checkpoint schema/identity mismatch')
        if len(state['rank_rng']) != self.identity['world_size']:
            raise RuntimeError('Recovery world size mismatch')
        if not {'epoch', 'update_in_epoch', 'global_step', 'images_seen'} <= state['progress'].keys():
            raise RuntimeError('Incomplete data cursor')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'])
        restore_rng(state['rank_rng'][rank], generator)
        assert_state_equal(model.state_dict(), state['model'], 'model')
        assert_state_equal(optimizer.state_dict(), state['optimizer'], 'optimizer')
        assert_state_equal(ema.state_dict(), state['ema'], 'ema')
        assert_state_equal(rng_state(generator), state['rank_rng'][rank], 'rng')
        return state['progress']
