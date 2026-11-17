"""Synthetic two-GPU CLI smoke. No real pretraining or SFT is performed."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import runpy
import sqlite3
import sys
import subprocess
import tarfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'tests')]
from denoiser import DualEMA
from hnet.checkpoints import Checkpoints, assert_state_equal, atomic_json, canonical, rng_state, sha256_file
from sft_fixtures import image_bytes, tokenizer_directory, write_tar
from t2i.data import CC12MSequence, encode_captions, to_model_range
from t2i.diffusion import objective, training_randomness
from t2i.model import TextToImage
from t2i.config import RUNTIME_KEYS
from test_t2i import tiny


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    tokenizer_directory(root / 'encoder', encoder=True)
    config = tiny(text_width=24)
    atomic_json(root / 'model.json', {k: v for k, v in asdict(config).items() if k not in RUNTIME_KEYS})
    atomic_json(root / 'runtime.json', dict({k: v for k, v in asdict(config).items() if k in RUNTIME_KEYS}, compile=False))
    pretrain = json.loads((REPO / 'configs/t2i/training/cc12m_250k.json').read_text())
    pretrain.update(global_batch=4, microbatch=2, workers=0, log_every=1)
    sft = dict(pretrain, phase='sft40k', data_update_offset=250000, updates=290000)
    sft['evaluation'] = dict(pretrain['evaluation'], enabled=False, full_updates=[])
    atomic_json(root / 'pretrain.json', pretrain)
    atomic_json(root / 'sft.json', sft)
    write_tar(root / 'images.tar', [(f'{i}.png', image_bytes(i)) for i in range(4)])
    dtype = np.dtype([('key', '<i8'), ('offset', '<i8'), ('size', '<i4'), ('shard', '<i4')])
    with tarfile.open(root / 'images.tar') as tar:
        rows = [(i, member.offset_data, member.size, 0) for i, member in enumerate(tar)]
    np.save(root / 'records.npy', np.asarray(rows, dtype=dtype))
    np.save(root / 'sequence.npy', np.tile(np.arange(4, dtype=np.int32), 40000))
    with sqlite3.connect(root / 'captions.sqlite') as db:
        db.execute('CREATE TABLE captions(key INTEGER PRIMARY KEY, caption TEXT)')
        db.executemany('INSERT INTO captions VALUES(?,?)', enumerate(['a red cat', 'a blue dog', 'a small house', 'two birds']))
    paths = {key: str(root / name) for key, name in [('records', 'records.npy'), ('captions', 'captions.sqlite'), ('sequence', 'sequence.npy')]}
    fingerprint = {key + '_sha256': sha256_file(path) for key, path in paths.items()}
    atomic_json(root / 'inputs.json', dict(schema=1, phase='sft40k', complete=True, **paths,
                encoder=str(root / 'encoder'), shards=[str(root / 'images.tar')], fingerprint=fingerprint))


def parent(root):
    torch.backends.cudnn.deterministic = True
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = tiny(text_width=24)
    recipe = json.loads((root / 'pretrain.json').read_text())
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = TextToImage(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], betas=tuple(recipe['betas']),
                                  eps=recipe['eps'], weight_decay=recipe['weight_decay'], fused=True)
    ema = DualEMA(model, tuple(recipe['ema_decays']))
    generator = torch.Generator().manual_seed(42 + rank)
    ddp = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True,
                                  gradient_as_bucket_view=True)
    inputs = json.loads((root / 'inputs.json').read_text())
    dataset = CC12MSequence(inputs, config.image_size, config.text_length)
    batch = dataset.collate([dataset[i] for i in range(rank * 2, rank * 2 + 2)])
    from transformers import T5EncoderModel
    encoder = T5EncoderModel.from_pretrained(inputs['encoder'], local_files_only=True).cuda().eval()
    encoder.requires_grad_(False)
    context = encode_captions(encoder, batch['input_ids'].cuda(), batch['attention_mask'].cuda(), 2)
    images = to_model_range(batch['image'].cuda())
    t, noise, drop = training_randomness(42, 249999 * 4 + rank * 2, 2, images.shape[1:], images.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _, _ = objective(ddp, images, context, batch['attention_mask'].cuda(), t, noise, drop, config)
    loss.backward(); optimizer.step(); ema.update(model)
    states = [None] * 2
    dist.all_gather_object(states, rng_state(generator))
    if rank == 0:
        identity = dict(world_size=2, global_batch=4, model=asdict(config), recipe=recipe)
        checkpoints = Checkpoints(root / 'parent', identity, milestone_name='checkpoint-u{global_step:07d}.pt')
        checkpoints.save(model, optimizer, ema,
                         dict(epoch=0, update_in_epoch=250000, global_step=250000,
                              images_seen=1000000, training_seconds=0.), states, milestone=True)
    dist.barrier()
    dist.destroy_process_group()


def load_latest(root):
    pointer = json.loads((root / 'checkpoints/latest.json').read_text())
    # Resolve the actual checkpoint path from this trainer's receipt.
    return torch.load(root / 'checkpoints' / pointer['file'], map_location='cpu', weights_only=False, mmap=True)


def audit(root):
    resumed, whole = [load_latest(root / name) for name in ('resumed', 'whole')]
    difference = {'max_absolute': 0.0, 'different_tensors': 0}
    for key in ('model', 'optimizer', 'ema'):
        compare(resumed[key], whole[key], difference)
    assert_state_equal(resumed['rank_rng'], whole['rank_rng'])
    for key in ('global_step', 'update_in_epoch', 'images_seen', 'phase_update', 'phase_next_position'):
        assert resumed['progress'][key] == whole['progress'][key]
    assert resumed['progress']['phase_update'] == 4 and resumed['progress']['phase_next_position'] == 16
    value = dict(passed=True, synthetic=True, world_size=2, global_batch=4,
                 scope='Actual main_t2i.py fresh SFT, 2-update stop/resume versus 4 uninterrupted updates; tiny model and data',
                 cudnn_deterministic=True,
                 exact=['both rank RNG states', 'data and global cursors'],
                 numeric_comparison=dict(difference, rtol=1e-2, atol=1e-5),
                 progress=resumed['progress'])
    atomic_json(root / 'result.json', value)
    print(json.dumps(value, indent=2))


def compare(a, b, summary):
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            summary['different_tensors'] += 1
            summary['max_absolute'] = max(summary['max_absolute'], (a.double() - b.double()).abs().max().item())
        # BF16 forward/backward agreement. The router's CUDA cumsum has no
        # deterministic implementation in torch 2.5.1. Exact restoration is
        # checked by Checkpoints.load before the next step and by the CPU tests.
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-5)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key], summary)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y, summary)
    else:
        assert a == b


def run(root):
    if torch.cuda.device_count() < 2:
        raise RuntimeError('This smoke requires two visible CUDA GPUs')
    # Stabilize cuBLAS/cuDNN in each process; the router's cumsum still requires
    # numerical rather than bitwise comparisons after subsequent GPU updates.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    prepare(root)
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    subprocess.run(launch + [__file__, 'parent', str(root)], check=True, cwd=REPO)
    common = [__file__, 'train', '--model', str(root / 'model.json'), '--training', str(root / 'sft.json'),
              '--runtime', str(root / 'runtime.json'), '--inputs', str(root / 'inputs.json')]
    parent_arg = ['--finetune-from', str(root / 'parent/checkpoint-u0250000.pt')]
    for name, extra, stop in [('resumed', parent_arg, 250002), ('resumed', ['--resume'], 250004),
                              ('whole', parent_arg, 250004)]:
        subprocess.run(launch + common + ['--output', str(root / name), '--max-updates', str(stop)] + extra,
                       check=True, cwd=REPO)
    audit(root)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        torch.backends.cudnn.deterministic = True
        sys.argv = [str(REPO / 'main_t2i.py')] + sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name='__main__')
        raise SystemExit(0)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'prepare', 'parent', 'audit'])
    p.add_argument('root', type=Path, help='New scratch directory; all weights and examples are synthetic')
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
