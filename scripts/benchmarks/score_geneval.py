"""GenEval adapter: verified generation manifest -> pinned Mask2Former + CLIP scorer -> keyed summary.

    $GENEVAL_ENV/bin/python scripts/benchmarks/score_geneval.py --generation GEN_DIR --assets ASSET_ROOT --devices 0 1 2 3 4 5 6 7
    $GENEVAL_ENV/bin/python scripts/benchmarks/score_geneval.py --generation GEN_DIR --assets ASSET_ROOT --devices cpu --workers 8

The pinned upstream evaluate_images.py runs unmodified, one process per device on disjoint
prompt folders (GPU: the official GenEval script; CPU: the Looped-DiT copy, identical except
that it drops the CUDA assertion). Every result is mapped back to (prompt, sample) and aggregated
like summary_scores.py: the mean over the six tags of per-image accuracy.
"""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from t2i import benchmarks as B  # noqa: E402

TAGS = ('single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr')
DETECTOR = 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco'
OPTIONS = {'model': DETECTOR, 'clip_model': 'ViT-L-14', 'threshold': 0.3, 'counting_threshold': 0.9, 'max_objects': 16,
           'max_overlap': 1.0, 'position_threshold': 0.1, 'bgcolor': '#999', 'crop': '1'}
PACKAGES = ('torch', 'torchvision', 'mmcv-full', 'mmdet', 'open-clip-torch', 'clip-benchmark', 'numpy', 'pandas', 'pillow',
            'opencv-python-headless', 'scikit-learn')


def detector_config():
    import mmdet
    from mmcv import Config
    path = Path(mmdet.__file__).parent/'.mim/configs/mask2former'/f'{DETECTOR}.py'
    resolved = json.loads(json.dumps(Config.fromfile(str(path))._cfg_dict.to_dict(), sort_keys=True, default=str))
    return path, B.sha256_json(resolved)


def scorer_identity(assets_root, assets, device, variant):
    import torch
    from mmcv.ops import get_compiling_cuda_version
    folder = 'geneval/official' if variant == 'official' else 'looped-dit/looped_dit/eval/geneval'
    config, resolved = detector_config()
    return {'scorer': 'geneval', 'variant': variant, 'script_folder': folder,
            'script': {n: B.verify_asset(assets_root, f'{folder}/{n}', assets) for n in ('evaluate_images.py', 'object_names.txt')},
            'aggregation': 'mean over tags of image accuracy (summary_scores.py)',
            'detector': {'weights_sha256': B.verify_asset(assets_root, f'geneval/detector/{DETECTOR}.pth', assets),
                         'config': f'mmdet/.mim/configs/mask2former/{DETECTOR}.py', 'config_sha256': B.sha256_file(config),
                         'resolved_config_sha256': resolved, 'options': OPTIONS},
            'clip': {'arch': 'ViT-L-14', 'pretrained': 'openai', 'weights_sha256': B.verify_asset(assets_root, 'clip/ViT-L-14.pt', assets)},
            'packages': {p: importlib.metadata.version(p) for p in PACKAGES},
            'mmcv_ops_cuda': get_compiling_cuda_version(), 'device': device,
            'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}


def parse(results, expected):
    rows = {}
    for line in results.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line); parts = Path(row['filename']).parts
        key = (parts[-3], int(Path(parts[-1]).stem))
        if key in rows:
            raise B.ManifestError(f'Duplicated GenEval result for {key}')
        if key not in expected:
            raise B.ManifestError(f'GenEval result for an unexpected image {key}')
        if not isinstance(row['correct'], bool):
            raise B.ManifestError(f'Non-boolean GenEval result for {key}')
        rows[key] = row
    return rows


def summarize(images):
    """summary_scores.py on per-image rows (tag, correct, metadata string)."""
    by_tag, by_prompt = defaultdict(list), defaultdict(list)
    for row in images:
        by_tag[row['tag']].append(row['correct']); by_prompt[row['metadata']].append(row['correct'])
    if set(by_tag) - set(TAGS):
        raise B.ManifestError(f'Unknown GenEval tags {set(by_tag) - set(TAGS)}')
    tasks = {t: sum(by_tag[t])/len(by_tag[t]) for t in TAGS if t in by_tag}
    return {'overall': sum(tasks.values())/len(tasks), 'task_scores': tasks,
            'percent_correct_images': sum(r['correct'] for r in images)/len(images),
            'percent_correct_prompts': sum(any(v) for v in by_prompt.values())/len(by_prompt),
            'total_images': len(images), 'total_prompts': len(by_prompt)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--generation', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--devices', nargs='+', default=['0'], help="GPU indices, or 'cpu'")
    p.add_argument('--workers', type=int, default=1, help='CPU processes when --devices cpu')
    p.add_argument('--threads', type=int, help='CPU threads per worker (default: available cores / workers)')
    p.add_argument('--variant', choices=('official', 'looped-dit'), help='default: official on GPU, looped-dit on CPU')
    p.add_argument('--expect', nargs='*', default=[], help='identity checks, e.g. model.checkpoint_sha256=HEX model.ema_decay=0.99995')
    p.add_argument('--restart-incomplete', action='store_true', help='discard an unfinished folder of this exact key (preemption)')
    p.add_argument('--output', type=Path, help='default: GEN_DIR/scores/geneval-<key>')
    a = p.parse_args()
    started = time.monotonic()
    cpu = a.devices == ['cpu']
    variant = a.variant or ('looped-dit' if cpu else 'official')
    manifest = B.verify_generation(a.generation, B.parse_expectations(a.expect))
    identity = manifest['identity']
    if identity['dataset']['benchmark'] != 'geneval':
        raise B.ManifestError('Not a GenEval generation')
    assets, _ = B.assets_manifest()
    if identity['dataset']['files'] != B.dataset_identity('geneval', a.assets, assets)['files']:
        raise B.ManifestError('Generation used different GenEval metadata than the pinned assets')
    pinned = B.read_jsonl(a.assets/'geneval/evaluation_metadata.jsonl')
    for index in manifest['indices']:
        if json.loads((a.generation/'images'/f'{index:05d}'/'metadata.jsonl').read_text()) != pinned[index]:
            raise B.ManifestError(f'metadata.jsonl of prompt {index} differs from the pinned GenEval row')
    scorer = scorer_identity(a.assets, assets, 'cpu' if cpu else 'cuda', variant)
    out, done = B.score_directory(a.generation, 'geneval', scorer, a.output, a.restart_incomplete)
    if done:
        print(json.dumps({'geneval': json.loads((out/'summary.json').read_text())['primary'], 'already_complete': True, 'output': str(out)})); return
    home = out/'home'; (home/'.cache/clip').mkdir(parents=True)
    os.symlink((a.assets/'clip/ViT-L-14.pt').resolve(), home/'.cache/clip/ViT-L-14.pt')
    folders = sorted({e['item_id'] for e in manifest['entries']})
    workers = min(a.workers if cpu else len(a.devices), len(folders))
    threads = a.threads or max(1, len(os.sched_getaffinity(0))//workers)
    script, (config, _) = a.assets/scorer['script_folder']/'evaluate_images.py', detector_config()
    procs = []
    for k in range(workers):
        shard = out/'shards'/f'{k:02d}'; shard.mkdir(parents=True)
        for item in folders[k::workers]:
            os.symlink((a.generation/'images'/item).resolve(), shard/item)
        env = dict(os.environ, HOME=str(home), PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='' if cpu else B.physical_device(a.devices[k]))
        if cpu:
            env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        log = (out/'shards'/f'{k:02d}.log').open('w')
        command = [sys.executable, str(script), str(shard), '--outfile', str(out/'shards'/f'{k:02d}.jsonl'),
                   '--model-path', str(a.assets/'geneval/detector'), '--model-config', str(config)]
        procs.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    codes = [proc.wait() for proc, _ in procs]
    for _, log in procs:
        log.close()
    if any(codes):
        raise RuntimeError(f'GenEval shards failed with codes {codes}; see {out}/shards/*.log')
    expected = {(e['item_id'], e['sample']): e for e in manifest['entries']}
    results = {}
    for k in range(workers):
        for key, row in parse(out/'shards'/f'{k:02d}.jsonl', expected).items():
            if key in results:
                raise B.ManifestError(f'Image {key} scored by two shards')
            results[key] = row
    missing = sorted(set(expected) - set(results))
    if missing:
        raise B.ManifestError(f'GenEval did not score {missing[:5]} ({len(missing)} images)')
    images = []
    for key, e in sorted(expected.items(), key=lambda kv: (kv[1]['index'], kv[1]['sample'])):
        row = results[key]
        images.append({'index': e['index'], 'item_id': e['item_id'], 'sample': e['sample'], 'path': e['path'],
                       'pixel_sha256': e['pixel_sha256'], 'tag': row['tag'], 'prompt': row['prompt'], 'correct': row['correct'],
                       'reason': row['reason'], 'metadata': row['metadata'], 'details': row['details']})
    metrics = summarize(images)
    with (out/'results.jsonl').open('w') as f:
        f.writelines(json.dumps(row) + '\n' for row in images)
    summary = B.score_record(out, 'geneval', manifest, scorer, metrics, metrics['overall'],
                             {'results_sha256': B.sha256_file(out/'results.jsonl'), 'workers': workers,
                              'seconds': time.monotonic() - started})
    print(json.dumps({'geneval': metrics['overall'], 'task_scores': metrics['task_scores'], 'images': len(images),
                      'output': str(out), 'seconds': round(summary['seconds'], 1)}), flush=True)


if __name__ == '__main__':
    main()
