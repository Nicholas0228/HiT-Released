"""GPU parity of the clean text-to-image path against the original MiniT2I B/16 handoff.

Each worker runs in its own process on one GPU and trains for a few updates from
seed-42 initialization on the real fixed sequence: data, FLAN-T5 contexts,
positional noise, the objective, AdamW and both EMAs. 'legacy' uses the original
branch (codex/minit2i-b16-b300-fp32) end to end and 'clean' this repository; the
'-repeat' workers rerun each to measure run-to-run GPU nondeterminism.

  python scripts/check_t2i_parity.py --legacy /path/to/minit2i-b16-b300-fp32 \
    --inputs inputs.json --model configs/t2i/models/hit_b16.json --output parity-hit

Data, contexts and randomness must match bitwise. The first update's loss,
prediction and every gradient, the loss trajectory and the final weights must
agree with the legacy code as closely as each implementation agrees with itself
(legacy-repeat, clean-repeat): HiT's grouping readers use atomic scatter-adds, so
even the original code is not bitwise reproducible on GPU.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(-1).view(dtype=__import__('torch').uint8)
                          .numpy().tobytes()).hexdigest()


def legacy_manifest(inputs, directory):
    """The original loader's manifest format for the same records, captions and shards."""
    manifest = {'complete': True, 'records': inputs['records'], 'caption_db': inputs['captions'],
                'shards': [{'path': path} for path in inputs['shards']]}
    path = Path(directory) / 'legacy-manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def worker(a):
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    sys.path.insert(0, str(ROOT))
    from t2i.config import load_t2i_config
    config, recipe, runtime = load_t2i_config(a.model, a.training, a.runtime)
    inputs = json.loads(Path(a.inputs).read_text())
    seed, G, M = recipe['seed'], a.global_batch, a.microbatch
    shape = (3, config.image_size, config.image_size)
    if a.worker.startswith('legacy'):
        sys.path.insert(0, str(a.legacy))
        from minit2i_hnet.model import MiniT2IHNet, ModelConfig
        from minit2i_hnet.data import TrainingData
        from minit2i_hnet.diffusion import loss as legacy_loss, training_randomness
        from minit2i_hnet.training import encode_context
        from minit2i_hnet.ema import DualEMA
        arm = 'hnet' if config.variant == 'hnet' else 'dense'
        values = json.loads((Path(a.legacy) / f'configs/b300/minit2i-b16-{arm}.json').read_text())['model']
        values.update(execution='native', compact_backend=config.compact_batching, attention='sdpa')
        if config.variant == 'hnet':
            values.update(threshold=config.thresholds[0], second_threshold=config.thresholds[1],
                          routing_workers=config.routing_workers, tree_precision=config.tree_precision)
        legacy_config = ModelConfig(**values)
        torch.manual_seed(seed)
        model = MiniT2IHNet(legacy_config).to(device)
        dataset = TrainingData(legacy_manifest(inputs, a.output), inputs['sequence'], inputs['encoder'])

        def randomness(start, count):
            parts, position = [], start
            while position < start + count:
                block = position // 128
                draw = training_randomness(torch.empty(128, *shape, device=device), seed, block // 4, block % 4)
                lo, hi = position - block * 128, min(start + count, (block + 1) * 128) - block * 128
                parts.append([v[lo:hi] for v in draw])
                position = (block + 1) * 128
            return tuple(torch.cat(p, 0) for p in zip(*parts))

        def objective(images, context, mask, t, noise, drop):
            total, stats, aux = legacy_loss(model, images, context, mask, t, noise, drop, legacy_config.budget_weight,
                                            legacy_config.target_ratio, a_min=recipe['a_min'])
            return total, stats

        encode = encode_context
    else:
        from denoiser import DualEMA
        from t2i.data import CC12MSequence, encode_captions
        from t2i.diffusion import objective as clean_objective, training_randomness
        from t2i.model import TextToImage
        torch.manual_seed(seed)
        model = TextToImage(config).to(device)
        dataset = CC12MSequence(inputs, config.image_size, config.text_length)

        def randomness(start, count):
            return training_randomness(seed, start, count, shape, device)

        def objective(images, context, mask, t, noise, drop):
            total, stats, _ = clean_objective(model, images, context, mask, t, noise, drop, config, recipe['a_min'])
            return total, stats

        encode = encode_captions
    from transformers import T5EncoderModel
    encoder = T5EncoderModel.from_pretrained(inputs['encoder'], local_files_only=True).eval().to(device)
    encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['initial_lr'], betas=tuple(recipe['betas']),
                                  eps=recipe['eps'], weight_decay=recipe['weight_decay'], fused=True)
    ema = DualEMA(model, decays=tuple(recipe['ema_decays']))
    init = hashlib.sha256()
    for name, value in model.state_dict().items():
        init.update(name.encode()); init.update(value.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes())
    steps, first = [], {}
    for update in range(a.updates):
        start = update * G
        batch = dataset.collate([dataset[i] for i in range(start, start + G)])
        images = batch['image'].to(device).float().div_(255).sub_(.5).div_(.5)
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        context = encode(encoder, ids, mask, M)
        t, noise, drop = randomness(start, G)
        lr = recipe['initial_lr'] + (recipe['lr'] - recipe['initial_lr']) * min(update / recipe['warmup_updates'], 1.)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        summaries = []
        for micro in range(G // M):
            part = slice(micro * M, (micro + 1) * M)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                total, stats = objective(images[part], context[part], mask[part], t[part], noise[part], drop[part])
                total = total / (G // M)
            total.backward()
            summaries.append({k: float(v) for k, v in stats.items()})
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))))
        if update == 0:
            first = {name: p.grad.detach().float().cpu() for name, p in model.named_parameters() if p.grad is not None}
        optimizer.step(); ema.update(model)
        steps.append({'update': update + 1, 'keys': batch['keys'].tolist(), 'image': digest(batch['image']),
                      'tokens': digest(batch['input_ids']), 'mask': digest(batch['attention_mask']),
                      'context': digest(context), 't': digest(t), 'noise': digest(noise), 'drop': digest(drop),
                      'gradient_norm': norm, 'loss': {k: float(np.mean([s[k] for s in summaries])) for k in summaries[0]}})
        print(json.dumps(steps[-1]['loss'] | {'update': update + 1, 'gradient_norm': norm}), flush=True)
    torch.save({'steps': steps, 'initialization_sha256': init.hexdigest(), 'first_gradients': first,
                'weights': {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
                'ema': [[v.detach().float().cpu() for v in values] for values in ema.values]},
               Path(a.output) / f'{a.worker}.pt')


def compare(a):
    import numpy as np
    import torch
    names = ('legacy', 'legacy-repeat', 'clean', 'clean-repeat')
    runs = {name: torch.load(Path(a.output) / f'{name}.pt', weights_only=False) for name in names}

    def gradient_differences(x, y):
        values = [float((runs[x]['first_gradients'][n] - runs[y]['first_gradients'][n]).norm()
                        / runs[y]['first_gradients'][n].norm().clamp_min(1e-30)) for n in runs[y]['first_gradients']]
        return {'max': max(values), 'median': float(np.median(values)), 'p90': float(np.percentile(values, 90)),
                'bitwise_equal_fraction': float(np.mean([v == 0 for v in values]))}

    def weight_differences(x, y):
        values = [float((runs[x]['weights'][k] - runs[y]['weights'][k]).norm() / runs[y]['weights'][k].norm())
                  for k in runs[y]['weights'] if runs[y]['weights'][k].norm() > 0]
        return {'max': max(values), 'median': float(np.median(values))}

    pairs = [('clean', 'legacy'), ('legacy-repeat', 'legacy'), ('clean-repeat', 'clean')]
    data_keys = ('keys', 'image', 'tokens', 'mask', 'context', 't', 'noise', 'drop')
    report = {
        'initialization_identical': len({runs[n]['initialization_sha256'] for n in names}) == 1,
        'data_identical': all(all(runs[n]['steps'][i][k] == runs['legacy']['steps'][i][k] for k in data_keys)
                              for n in names for i in range(len(runs['legacy']['steps']))),
        'first_update_gradients': {f'{x}_vs_{y}': gradient_differences(x, y) for x, y in pairs},
        'final_weights': {f'{x}_vs_{y}': weight_differences(x, y) for x, y in pairs},
        'steps': [{'update': step['update'], **{f'{n}_loss': runs[n]['steps'][i]['loss']['generation_loss'] for n in names},
                   **{f'{n}_gradient_norm': runs[n]['steps'][i]['gradient_norm'] for n in names},
                   **{f'{n}_groups': [runs[n]['steps'][i]['loss'].get('group_mean'),
                                      runs[n]['steps'][i]['loss'].get('group_mean2')] for n in names}}
                  for i, step in enumerate(runs['legacy']['steps'])]}
    Path(a.output, 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'steps'}, indent=2))
    for step in report['steps']:
        print(step['update'], ' '.join(f"{n}={step[n + '_loss']:.6f}/{step[n + '_gradient_norm']:.5f}" for n in names))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--legacy', type=Path, required=True, help='Checkout of codex/minit2i-b16-b300-fp32')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--model', default=str(ROOT / 'configs/t2i/models/hit_b16.json'))
    p.add_argument('--training', default=str(ROOT / 'configs/t2i/training/cc12m_250k.json'))
    p.add_argument('--runtime', default=str(ROOT / 'configs/t2i/runtime/eager.json'))
    p.add_argument('--updates', type=int, default=4)
    p.add_argument('--global-batch', type=int, default=16)
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', choices=['legacy', 'legacy-repeat', 'clean', 'clean-repeat'])
    a = p.parse_args()
    if a.worker:
        return worker(a)
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ('legacy', 'legacy-repeat', 'clean', 'clean-repeat'):
        command = [sys.executable, __file__, '--worker', name] + [
            f'--{k.replace("_", "-")}={v}' for k, v in vars(a).items() if k != 'worker' and v is not None]
        subprocess.run(command, check=True, cwd=ROOT)
    compare(a)


if __name__ == '__main__':
    main()
