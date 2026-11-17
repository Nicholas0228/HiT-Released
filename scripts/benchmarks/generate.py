"""Render text-to-image benchmark images (GenEval, DPG, PRISM, CoRe, Spatial) for the score_*.py adapters.

    torchrun --standalone --nproc-per-node=8 scripts/benchmarks/generate.py --benchmark geneval \\
        --checkpoint checkpoints/T2I512/HiT-B16_sft40k/model.safetensors --model configs/t2i/models/hit_b16_d27.json \\
        --sampling configs/t2i/sampling/hit_b16_d27.json --encoder-root FLAN_T5_LARGE \\
        --assets ASSET_ROOT --output-root OUT

Each prompt is one canonical batch of the released MiniT2I script (its 1 or 4 samples, logical rank
index % 8, seed + index*1000 + offset + logical*1e6). Whole prompts are dealt to the physical ranks;
nothing is reseeded by physical rank, so any GPU count renders the same images. Rank 0 merges the
per-prompt records into a verified manifest keyed by checkpoint, dataset, seed and sampler.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from t2i import benchmarks as B  # noqa: E402
from t2i.config import load_t2i_config  # noqa: E402
from t2i.data import encode_captions  # noqa: E402
from t2i.diffusion import sample  # noqa: E402
from t2i.evaluation import quantize  # noqa: E402
from t2i.model import TextToImage  # noqa: E402


def broadcast(value, rank, world):
    if world == 1:
        return value
    box = [value if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def load_weights(path, model, ema_decay):
    """A released safetensors state dict, or one EMA of this repository's trainer checkpoint."""
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), 'pt') as f:
            metadata = f.metadata() or {}
        model.load_state_dict(load_file(str(path)), strict=True)
        progress = json.loads(metadata['progress']) if 'progress' in metadata else {}
        step = progress.get('global_step', metadata.get('global_step')) or 0
        return {'format': 'safetensors', 'ema_decay': float(metadata.get('ema_decay', 'nan')), 'global_step': int(step)}
    from denoiser import DualEMA
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(state['model'], strict=True)
    ema = DualEMA(model, decays=tuple(state['identity']['recipe']['ema_decays']))
    ema.load_state_dict(state['ema'])
    index = list(ema.decays).index(ema_decay)
    with torch.no_grad():
        for target, value in zip(model.parameters(), ema.values[index]):
            target.copy_(value)
    return {'format': 'trainer', 'ema_decay': ema_decay, 'global_step': state['progress']['global_step']}


def source_identity():
    files = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and name.endswith('.py') and os.path.isabs(name) and os.path.isfile(name) \
                and Path(name).resolve().is_relative_to(REPO):
            files[str(Path(name).resolve().relative_to(REPO))] = B.sha256_file(name)
    return {'files': dict(sorted(files.items())), 'digest': B.sha256_json(files)}


def encoder_identity(root, assets):
    names = sorted(p.name for p in Path(root).iterdir()
                   if p.is_file() and p.suffix in ('.json', '.model', '.safetensors', '.bin'))
    files = {n: B.sha256_file(Path(root) / n) for n in names}
    pinned = {k.split('/', 1)[1]: v['sha256'] for k, v in assets['files'].items() if k.startswith('flan-t5-large/')}
    return {'root': str(root), 'files': files, 'pinned_revision': assets['text_encoder'],
            'matches_pinned_revision': all(files.get(k) == v for k, v in pinned.items() if k in files)
            and 'model.safetensors' in files}


def write_png(array, path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    Image.fromarray(array).save(tmp, format='PNG')
    os.replace(tmp, path)


def render(prompt, args, recipe, model, config, encoder, tokenizer, device):
    arrays = []
    for offset, size in B.canonical_batches(args.samples):
        tokens = tokenizer([prompt.text] * size, padding='max_length', truncation=True,
                           max_length=config.text_length, return_tensors='pt')
        ids, mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
        context = encode_captions(encoder, ids, mask, size)
        generator = torch.Generator(device=device).manual_seed(B.batch_seed(args.seed, prompt.index, offset))
        noise = torch.randn((size, 3, config.image_size, config.image_size), generator=generator, device=device,
                            dtype=torch.float32) * recipe['noise_std']
        with torch.autocast(device.type, dtype=torch.bfloat16):
            images = sample(model, context, mask, noise, guidance=recipe['guidance'], steps=recipe['steps'],
                            a_min=recipe['a_min'], group_guidance=recipe.get('group_guidance'))
        arrays += list(quantize(images).permute(0, 2, 3, 1).cpu().numpy())
    return arrays, len(tokenizer(prompt.text).input_ids) > config.text_length


def save_prompt(prompt, arrays, args, images_dir):
    rows, files = B.expected_entries(args.benchmark, [prompt], args.seed, args.samples), {}
    if args.benchmark == 'dpg':
        size = arrays[0].shape[0]
        grid = np.zeros((2 * size, 2 * size, 3), np.uint8)
        for row, array in zip(rows, arrays):
            x0, y0, x1, y1 = row['box']
            grid[y0:y1, x0:x1] = array
        write_png(grid, images_dir / rows[0]['path'])
    else:
        for row, array in zip(rows, arrays):
            write_png(array, images_dir / row['path'])
    if args.benchmark == 'geneval':
        meta = images_dir / prompt.item_id / 'metadata.jsonl'
        meta.write_text(json.dumps(prompt.metadata) + '\n')
        files[f'{prompt.item_id}/metadata.jsonl'] = B.sha256_file(meta)
    for row, array in zip(rows, arrays):
        row['pixel_sha256'] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        files[row['path']] = B.sha256_file(images_dir / row['path'])
    return rows, files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', choices=B.BENCHMARKS, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='released .safetensors or a trainer checkpoint')
    p.add_argument('--model', type=Path, default=REPO / 'configs/t2i/models/hit_b16_d27.json')
    p.add_argument('--runtime', type=Path, help='optional runtime config (routing workers, attention backend)')
    p.add_argument('--sampling', type=Path, default=REPO / 'configs/t2i/sampling/hit_b16_d27.json')
    p.add_argument('--encoder-root', type=Path, required=True, help='FLAN-T5-Large tokenizer and weights')
    p.add_argument('--assets', type=Path, required=True, help='root written by fetch_assets.py')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--ema', type=float, default=.99995, help='EMA decay (trainer checkpoints only)')
    p.add_argument('--seed', type=int, help='override the sampling config seed')
    p.add_argument('--indices', type=int, nargs='+', help='original prompt indices of a fixed subset')
    p.add_argument('--checkpoint-sha256', help='expected checkpoint SHA-256')
    p.add_argument('--resume', action='store_true', help='continue the identical identity after an interruption')
    args = p.parse_args()
    recipe = json.loads(args.sampling.read_text())
    args.seed = recipe['seed'] if args.seed is None else args.seed
    args.samples = B.SAMPLES[args.benchmark]
    world, rank, local = (int(os.environ.get(k, d)) for k, d in (('WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        # Ranks can finish their prompts minutes apart; keep the final barrier from timing out.
        dist.init_process_group('nccl', timeout=timedelta(hours=6))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()
    assets, assets_sha = B.assets_manifest()
    everything = B.load_prompts(args.benchmark, args.assets)
    prompts = B.select(everything, args.indices)
    config, _, runtime = load_t2i_config(args.model, runtime_path=args.runtime)
    model = TextToImage(config).to(device)
    weights = load_weights(args.checkpoint, model, args.ema)
    model.eval().requires_grad_(False)
    if runtime.get('compile', False):
        model.compile_blocks()
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(str(args.encoder_root), local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(str(args.encoder_root), local_files_only=True)
    encoder = encoder.eval().to(device).requires_grad_(False)
    identity = None
    if rank == 0:
        checkpoint_sha = B.sha256_file(args.checkpoint)
        if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256:
            raise RuntimeError(f'Checkpoint SHA-256 {checkpoint_sha} differs from the expected {args.checkpoint_sha256}')
        dataset = dict(B.dataset_identity(args.benchmark, args.assets, assets), prompt_count=len(everything),
                       subset=None if args.indices is None else sorted(args.indices))
        identity = {
            'schema': B.GENERATION_SCHEMA,
            'model': {'kind': 'checkpoint', 'arm': config.variant, 'checkpoint_sha256': checkpoint_sha,
                      'checkpoint_bytes': args.checkpoint.stat().st_size, 'checkpoint_format': weights['format'],
                      'progress': {'global_step': weights['global_step']}, 'ema_decay': weights['ema_decay'],
                      'model_config': json.loads(args.model.read_text()),
                      'thresholds': [float(model.router.threshold), float(model.router2.threshold)]
                      if model.router is not None else None},
            'dataset': dataset,
            'sampling': {'seed': args.seed, 'seed_rule': B.SEED_RULE, 'logical_ranks': B.LOGICAL_RANKS,
                         'batch_size': B.BATCH_SIZE, 'samples_per_prompt': args.samples, 'cfg': recipe['guidance'],
                         'group_guidance': recipe.get('group_guidance'), 'steps': recipe['steps'], 'sampler': 'euler',
                         'a_min': recipe['a_min'], 'noise_std': recipe['noise_std'], 'image_size': config.image_size,
                         'noise': 'float32 torch.Generator(cuda) per canonical batch',
                         'null_condition': 'zero caption mask', 'quantization': 'uint8((x*127.5+128).clamp(0,255))',
                         'output_layout': 'looped-dit-18ad35c', 'dpg_grid': '2x2 of 512: (0,0),(w,0),(0,h),(w,h)'},
            'precision': {'device_name': torch.cuda.get_device_name(device), 'autocast': 'bfloat16',
                          'weights': 'float32', 'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                          'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'compiled': bool(model._compiled),
                          'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'text_encoder': dict(encoder_identity(args.encoder_root, assets), max_length=config.text_length,
                                               padding='max_length', truncation=True, context_dtype='bfloat16')},
            'generator': source_identity(),
            'assets_manifest_sha256': assets_sha,
        }
    identity = broadcast(identity, rank, world)
    identity_sha = B.sha256_json(identity)
    out = args.output_root / B.output_key(identity)
    decision = None
    if rank == 0:
        if (out / 'complete.json').exists():
            if B.sha256_json(json.loads((out / 'identity.json').read_text())) != identity_sha:
                raise RuntimeError(f'{out} belongs to a different generation identity')
            B.verify_generation(out)
            decision = 'complete'
        elif out.exists():
            if not args.resume or B.sha256_json(json.loads((out / 'identity.json').read_text())) != identity_sha:
                raise RuntimeError(f'{out} exists without completion; rerun with --resume only for the identical identity')
            for stale in (out / 'images').rglob('.*.tmp'):
                stale.unlink()
            decision = 'resume'
        else:
            out.mkdir(parents=True)
            B.atomic_json(out / 'identity.json', identity)
            decision = 'new'
        print(json.dumps({'output': str(out), 'identity_sha256': identity_sha, 'decision': decision}), flush=True)
    decision = broadcast(decision, rank, world)
    if decision == 'complete':
        return
    images_dir, records = out / 'images', out / 'records'
    records.mkdir(parents=True, exist_ok=True)
    timings = []
    for position, prompt in enumerate(prompts):
        if B.owner(position, world) != rank:
            continue
        record_path = records / f'{prompt.index:05d}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['identity_sha256'] == identity_sha and all(
                    (images_dir / k).exists() and B.sha256_file(images_dir / k) == v for k, v in record['files'].items()):
                continue
        begin = time.monotonic()
        arrays, truncated = render(prompt, args, recipe, model, config, encoder, tokenizer, device)
        rows, files = save_prompt(prompt, arrays, args, images_dir)
        seconds = time.monotonic() - begin
        timings.append(seconds)
        B.atomic_json(record_path, {'identity_sha256': identity_sha, 'index': prompt.index, 'item_id': prompt.item_id,
                                    'entries': rows, 'files': files, 'text_truncated': truncated, 'physical_rank': rank,
                                    'physical_world_size': world, 'host': socket.gethostname(), 'seconds': seconds})
        if len(timings) % 25 == 1:
            print(json.dumps({'rank': rank, 'done': len(timings), 'index': prompt.index,
                              'seconds': round(seconds, 3)}), flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        entries, files, provenance = [], {}, {}
        for prompt in prompts:
            record = json.loads((records / f'{prompt.index:05d}.json').read_text())
            if record['identity_sha256'] != identity_sha or record['index'] != prompt.index:
                raise RuntimeError(f'Record {prompt.index} belongs to another generation identity')
            entries += record['entries']
            files.update(record['files'])
            provenance[prompt.item_id] = [record['physical_rank'], record['physical_world_size'], record['text_truncated']]
        reference = B.expected_entries(args.benchmark, prompts, args.seed, args.samples)
        if [{k: v for k, v in r.items() if k != 'pixel_sha256'} for r in entries] != reference:
            raise RuntimeError('Merged entries differ from the canonical assignment')
        B.atomic_json(out / 'manifest.json', {'schema': B.GENERATION_SCHEMA, 'identity': identity,
                                              'identity_sha256': identity_sha, 'indices': [q.index for q in prompts],
                                              'entries': entries, 'files': files, 'provenance': provenance})
        B.verify_generation(out)
        B.atomic_json(out / 'complete.json', {'identity_sha256': identity_sha, 'utc': datetime.now(timezone.utc).isoformat(),
                                              'images': len(entries), 'prompts': len(prompts), 'physical_world_size': world,
                                              'wall_seconds': time.monotonic() - started,
                                              'truncated_prompts': sum(v[2] for v in provenance.values())})
        print(json.dumps({'complete': str(out), 'images': len(entries),
                          'seconds': round(time.monotonic() - started, 1)}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
