"""Calibrate HiT's two merge thresholds at initialization on 1,024 fixed CC12M pairs.

Stage 1 is calibrated first and frozen; stage 2 then observes actual stage-1 groups.
Each threshold is the midpoint between sorted merge scores for which the total
group count equals population / ratio. The cohort, noise and caption dropout are
fixed (four logical ranks x 256 pairs, batches of 32), so any GPU count gives the
same thresholds. Run with torchrun on 1-8 GPUs:

  torchrun --standalone --nproc-per-node=4 scripts/calibrate_t2i.py \
    --inputs "$T2I_INPUTS/inputs.json" --output "$RUN/calibration"
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet.checkpoints import atomic_json  # noqa: E402
from t2i.config import load_t2i_config  # noqa: E402
from t2i.data import CC12MSequence, encode_captions, to_model_range  # noqa: E402
from t2i.diffusion import stream_randomness  # noqa: E402
from t2i.inputs import load_inputs  # noqa: E402
from t2i.model import TextToImage  # noqa: E402

PAIRS, LOGICAL, BATCH, SELECTION_SEED, NOISE_SEED, MODEL_SEED = 1024, 4, 32, 1701, 1701, 42


def choose_threshold(scores, populations, ratio, precision):
    values = np.sort(np.concatenate([row[:int(n) - 1] for row, n in zip(scores, populations)]))
    target = float(np.sum(populations)) / ratio
    cuts = int(round(target - len(populations)))
    if not 0 < cuts < len(values):
        raise RuntimeError('The requested ratio is outside the observed merge range')
    threshold = float((float(values[cuts - 1]) + float(values[cuts])) / 2)
    if precision == 'fp32':
        threshold = float(np.float32(threshold))
        if values[cuts - 1] < values[cuts] and threshold <= float(values[cuts - 1]):
            threshold = float(values[cuts])
    counts = np.array([1 + np.sum(row[:int(n) - 1] < threshold) for row, n in zip(scores, populations)])
    if abs(float(counts.sum()) - target) > max(2., len(populations) * .01):
        raise RuntimeError('Calibrated counts miss the requested ratio')
    return threshold, counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='configs/t2i/models/hit_b16_d27.json')
    p.add_argument('--runtime', default='configs/t2i/runtime/cuda.json')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory for the report and calibrated model config')
    args = p.parse_args()
    config, _, runtime = load_t2i_config(args.model, None, args.runtime, calibrating=True)
    if config.variant != 'hnet':
        raise ValueError('Only HiT has merge thresholds')
    rank, world, local = [int(os.environ.get(k, d)) for k, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    inputs = load_inputs(args.inputs)
    selection = np.random.default_rng(SELECTION_SEED).choice(inputs['fingerprint']['accepted_pairs'], PAIRS,
                                                              replace=False).astype(np.int32)
    selection_path = args.output / 'selection.npy'
    if rank == 0:
        np.save(selection_path, selection, allow_pickle=False)
    if world > 1:
        dist.barrier()
    dataset = CC12MSequence(inputs, config.image_size, config.text_length, sequence=selection_path)
    tasks = [(logical, batch) for logical in range(LOGICAL) for batch in range(PAIRS // LOGICAL // BATCH)]
    mine = [task for index, task in enumerate(tasks) if index % world == rank]
    torch.manual_seed(MODEL_SEED)
    model = TextToImage(replace(config, thresholds=(.5, .5))).to(device).eval()
    if runtime.get('compile', False):
        model.compile_blocks()
    from transformers import T5EncoderModel
    from hnet.runtime.grouping import project_valid
    encoder = T5EncoderModel.from_pretrained(inputs['encoder'], local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    image_shape = (3, config.image_size, config.image_size)
    thresholds, records = [], []
    for stage in (1, 2):
        rows = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for logical, batch_index in mine:
                start = logical * (PAIRS // LOGICAL) + batch_index * BATCH
                batch = dataset.collate([dataset[i] for i in range(start, start + BATCH)])
                images = to_model_range(batch['image'].to(device))
                mask = batch['attention_mask'].to(device)
                context = encode_captions(encoder, batch['input_ids'].to(device), mask, BATCH)
                t, noise, drop = stream_randomness(NOISE_SEED, batch_index * 1048576 + logical * 1024, BATCH,
                                                   image_shape, device)
                mask = torch.where(drop[:, None], torch.zeros_like(mask), mask)
                noisy = images * t[:, None, None, None] + noise * (1 - t)[:, None, None, None]
                img, txt = model.encode(noisy, context, mask, time=t)
                time_state, _, packed, layout = model.grouping_input(img, txt, mask)
                if stage == 1:
                    _, q, k = project_valid(model.router, packed, layout.lengths)
                    structure, population = model.router.structure(q, k, layout.lengths), layout.counts
                else:
                    one, _, valid, s1, _ = model.router(packed, layout.valid, layout.counts, layout.lengths, True)
                    e = config.encoder_depth
                    one, _ = model.compact(one, time_state, valid, s1.lengths, e, e + config.intermediate_depth)
                    _, q, k = project_valid(model.router2, one, s1.lengths)
                    structure, population = model.router2.structure(q, k, s1.lengths), s1.counts
                scores = structure.scores.double().cpu().numpy()
                for i, n in enumerate(population.cpu().tolist()):
                    rows.append((start + i, scores[i, :n - 1].copy(), n, float(t[i]), bool(drop[i]),
                                 int(batch['keys'][i])))
        gathered = [None] * world
        if world > 1:
            dist.all_gather_object(gathered, rows)
        else:
            gathered = [rows]
        rows = sorted((row for part in gathered for row in part), key=lambda row: row[0])
        if [row[0] for row in rows] != list(range(PAIRS)):
            raise RuntimeError('Calibration cohort is incomplete')
        scores, populations = [row[1] for row in rows], np.array([row[2] for row in rows])
        times, drops = np.array([row[3] for row in rows]), np.array([row[4] for row in rows])
        threshold, counts = choose_threshold(scores, populations, config.ratios[stage - 1], config.tree_precision)
        (model.router if stage == 1 else model.router2).threshold.fill_(threshold)
        thresholds.append(threshold)
        records.append({'stage': stage, 'threshold': threshold, 'mean_population': float(populations.mean()),
                        'mean_groups': float(counts.mean()), 'actual_ratio': float(populations.sum() / counts.sum()),
                        'group_quantiles': np.percentile(counts, [0, 1, 10, 50, 90, 99, 100]).tolist(),
                        'singletons': int((counts == 1).sum()), 'caption_dropped': int(drops.sum()),
                        'by_time': [{'t': [lo, hi], 'samples': int(sel.sum()),
                                     'mean_groups': float(counts[sel].mean()) if sel.any() else None}
                                    for lo, hi in zip([0, .2, .4, .6, .8], [.2, .4, .6, .8, 1.])
                                    for sel in [(times >= lo) & (times < hi)]],
                        'keys_head': [row[5] for row in rows[:8]]})
        if rank == 0:
            print(json.dumps(records[-1]), flush=True)
    if rank == 0:
        calibrated = json.loads(Path(args.model).read_text())
        calibrated['thresholds'] = thresholds
        atomic_json(args.output / 'model.json', calibrated)
        atomic_json(args.output / 'calibration.json', {
            'pairs': PAIRS, 'selection_seed': SELECTION_SEED, 'noise_seed': NOISE_SEED, 'model_seed': MODEL_SEED,
            'logical_ranks': LOGICAL, 'batch': BATCH, 'world_size': world, 'stages': records,
            'model': asdict(replace(config, thresholds=tuple(thresholds))), 'runtime': runtime,
            'inputs': inputs['fingerprint'], 'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__})
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
