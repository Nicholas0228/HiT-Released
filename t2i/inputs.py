"""Fetch and verify the pinned text-to-image inputs, then write one portable manifest.

Layout under the output directory follows the frozen bundle index (bundle.json):
corpus/{records.npy,captions.sqlite,shards/cc12m-train-XXXX.tar}, sequence.npy,
encoder/ (FLAN-T5-Large) and evaluation/ (COCO-30k contexts, masks, order,
reference statistics, Inception). CC12M shards and T5 are public; the derived
index, captions, sequence and COCO assets come from the private bundle.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / 'configs/t2i/inputs.json'


def sha256_file(path, block=8 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(block), b''):
            digest.update(part)
    return digest.hexdigest()


def needed_shards(output, updates, global_batch=1024):
    """Shards read by the first `updates` optimizer updates of the fixed sequence."""
    import numpy as np
    records = np.load(output / 'corpus/records.npy', mmap_mode='r')
    sequence = np.load(output / 'sequence.npy', mmap_mode='r')
    return sorted(set(records[np.asarray(sequence[:updates * global_batch])]['shard'].tolist()))


def retrying(function, attempts=5):
    """Retry interrupted or rate-limited downloads; completed files are never fetched twice."""
    import time
    def call(*args, **kwargs):
        for attempt in range(attempts):
            try:
                return function(*args, **kwargs)
            except Exception as error:  # network errors, 429/5xx responses
                if attempt == attempts - 1:
                    raise
                wait = 30 * 2 ** attempt
                print(f'download interrupted ({type(error).__name__}); retrying in {wait} s', flush=True)
                time.sleep(wait)
    return call


def fetch(output, token, updates=None, evaluation=True, verify='sha256', workers=8):
    """Download what is missing and verify it; `updates` limits shards to an initial prefix.

    The token is also used for the public repositories: authenticated requests get
    far higher rate limits for the 2,176 shard files.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    hf_hub_download, snapshot_download = retrying(hf_hub_download), retrying(snapshot_download)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pins = json.loads(PINS.read_text())
    bundle = pins['bundle']
    private = dict(repo_id=bundle['repository'], repo_type=bundle['repo_type'], revision=bundle['revision'], token=token)
    index_path = Path(hf_hub_download(filename=bundle['index'], local_dir=output, **private))
    if sha256_file(index_path) != bundle['index_sha256']:
        raise RuntimeError('Bundle index does not match the pinned hash')
    index = json.loads(index_path.read_text())
    derived = ['corpus/records.npy', 'corpus/captions.sqlite', 'sequence.npy']
    derived += [name for name in index['files'] if evaluation and name.startswith('evaluation/')]
    snapshot_download(local_dir=output, allow_patterns=derived, max_workers=workers, **private)
    snapshot_download(pins['encoder']['repository'], revision=pins['encoder']['revision'], token=token,
                      local_dir=output / 'encoder', max_workers=workers,
                      allow_patterns=[name.split('/', 1)[1] for name in index['files'] if name.startswith('encoder/')])
    shard_names = sorted(name for name in index['files'] if name.startswith('corpus/shards/'))
    if updates is not None:
        wanted = needed_shards(output, updates)
        shard_names = [shard_names[i] for i in wanted]
    missing = [name for name in shard_names
               if not (output / name).exists() or (output / name).stat().st_size != index['files'][name]['bytes']]
    if missing:
        snapshot_download(pins['images']['repository'], repo_type=pins['images']['repo_type'], token=token,
                          revision=pins['images']['revision'], local_dir=output / 'corpus/shards',
                          allow_patterns=[Path(name).name for name in missing], max_workers=workers)
    return write_manifest(output, index, verify=verify, workers=workers)


def write_manifest(output, index=None, verify='sha256', workers=8):
    """Check sizes (and hashes) of present files and record absolute paths in inputs.json.

    verify='size' checks sizes only; 'sha256' also hashes every non-shard file;
    'full' additionally hashes every present CC12M shard (about 1.2 TB).
    """
    output = Path(output).resolve()
    index = index or json.loads((output / 'bundle.json').read_text())
    present, absent = {}, []
    for name, expected in index['files'].items():
        path = output / name
        if not path.exists():
            if name.startswith('corpus/shards/') or name.startswith('evaluation/'):
                absent.append(name)
                continue
            raise FileNotFoundError(f'Required input missing: {path}')
        if path.stat().st_size != expected['bytes']:
            raise RuntimeError(f'Size mismatch: {path}')
        present[name] = expected['sha256']
    hashed = [name for name in present if verify == 'full' or (verify == 'sha256' and not name.startswith('corpus/shards/'))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, digest in zip(hashed, pool.map(lambda n: sha256_file(output / n), hashed)):
            if digest != present[name]:
                raise RuntimeError(f'SHA-256 mismatch: {output / name}')
    shards = sorted(name for name in index['files'] if name.startswith('corpus/shards/'))
    assets = index['assets']
    evaluation = None
    if not any(name.startswith('evaluation/') for name in absent):
        evaluation = {key: str(output / assets[key]) for key in ('reference', 'context', 'mask', 'order', 'captions',
                                                               'inception_directory')}
        evaluation['count'] = assets['count']
    manifest = {
        'schema': 1, 'root': str(output),
        'records': str(output / 'corpus/records.npy'), 'captions': str(output / 'corpus/captions.sqlite'),
        'sequence': str(output / 'sequence.npy'), 'encoder': str(output / 'encoder'),
        'shards': [str(output / name) for name in shards], 'evaluation': evaluation,
        'complete': not absent,
        'missing_shards': sum(name.startswith('corpus/shards/') for name in absent),
        'verification': {'mode': verify, 'hashed_files': len(hashed), 'present_files': len(present)},
        'fingerprint': {'bundle_index_sha256': sha256_file(output / 'bundle.json'),
                        'records_sha256': index['files']['corpus/records.npy']['sha256'],
                        'sequence_sha256': index['files']['sequence.npy']['sha256'],
                        'captions_sha256': index['files']['corpus/captions.sqlite']['sha256'],
                        'accepted_pairs': index['manifest']['count']},
    }
    temporary = output / f'.inputs.json.{os.getpid()}'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary, output / 'inputs.json')
    return manifest


def load_inputs(path, require_complete=False, require_evaluation=False):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema') != 1:
        raise ValueError('Unknown inputs manifest')
    if require_complete and not manifest['complete']:
        raise RuntimeError(f"{manifest['missing_shards']} CC12M shards are missing; rerun the fetch without --updates")
    if require_evaluation and manifest['evaluation'] is None:
        raise RuntimeError('COCO evaluation assets are missing; fetch without --no-evaluation or pass --no-eval')
    return manifest
