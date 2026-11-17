"""Pinned three-source MiniT2I fine-tuning mixture as an indexed training manifest.

Mirrors the released loader (minit2i-torch e54e15d): its source files, WebDataset sample
grouping, captions, ShareGPT-4o token filter and JPEG-95 round trip, and the upstream
DALL-E 3 parquet-to-tar staging. Strict full decodes replace its truncated-image tolerance.
Every output is rebuilt from pinned public bytes; `verify` checks its build receipt.
Downloads original datasets; does not redistribute their images or captions.
"""
import argparse
import ast
import base64
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import io
import json
from math import ceil, gcd
import multiprocessing
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
import tarfile
import tempfile
import time
import uuid

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from hnet.checkpoints import sha256_file as sha256

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


DTYPE = np.dtype([('key', '<i8'), ('offset', '<i8'), ('size', '<i4'), ('shard', '<i4')])
BLIP3_FT60K_TARS = ['dalle3.tar', 'geneval_train.tar', 'human_gestures.tar', 'journeyDB.tar', 'mscoco_human.tar',
                    'object_1.tar', 'object_2.tar', 'occupation_1.tar', 'occupation_2.tar', 'text_1.tar', 'text_2.tar']
IMAGE_FIELDS = {'blip3_ft60k': ('jpg', 'jpeg', 'png'), 'dalle3': ('jpg', 'jpeg', 'png'), 'sharegpt4o': ('png', 'jpg', 'jpeg')}
EXPECTED_FIELDS = {'blip3_ft60k': {'jpg', 'txt'}, 'dalle3': {'jpg', 'txt'}, 'sharegpt4o': {'png'}}
KEY_BASE = 10**9
DECODE_ERRORS = (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError)
SEQUENCE_ALGORITHM = (
    'finetune-mixture-v1. Integer weights W_s come from the exact decimal released weights (0.06:0.016:0.04 = 15:4:10). '
    'Position i takes source s_i = number of cumulative weights <= r_i, where r_i = x mod sum(W) over the PCG64 '
    'random_raw stream seeded with seed_for(seed, "finetune-source", 0), rejecting x >= 2^64 - (2^64 mod sum(W)). '
    'The j-th position assigned to source s reads its records (manifest order) at permutation[j % n_s] of epoch '
    'j // n_s, where each epoch permutation is the stable argsort of n_s random_raw values of PCG64 seeded with '
    'seed_for(seed, "finetune-permutation-<source>", epoch). seed_for(seed, domain, index) is the first 8 '
    'little-endian bytes of SHA-256("seed:domain:index") modulo 2^63-1. Values are int32 records.npy rows.')


def seed_for(seed, domain, index):
    """Domain-separated seed shared with the positional diffusion randomness."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{domain}:{index}'.encode()).digest()[:8], 'little') % (2**63 - 1)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + '.' + uuid.uuid4().hex[:8] + '.partial')
    with candidate.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(candidate, path)


def replace_bytes(path, write):
    """Derived outputs are rebuilt whole; the manifest written last marks completion."""
    candidate = Path(path).with_name(Path(path).name + '.partial')
    candidate.unlink(missing_ok=True)
    with candidate.open('xb') as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(candidate, path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def base_plus_ext(path):
    match = re.match(r'^((?:.*/|)[^.]+)[.]([^/]*)$', path)
    return (match.group(1), match.group(2)) if match else (None, None)


def samples(path):
    """WebDataset tar_file_iterator + group_by_keys, yielding member locations instead of bytes."""
    current = None
    with tarfile.open(path, 'r:') as tar:
        for member in tar:
            name = member.name
            if not member.isreg() or ('/' not in name and name.startswith('__') and name.endswith('__')):
                continue
            if re.match(r'__[^/]*__($|/)', name):
                continue
            prefix, suffix = base_plus_ext(name)
            if prefix is None:
                continue
            suffix = suffix.lower()
            if current is None or prefix != current['__key__']:
                if current is not None:
                    yield current
                current = {'__key__': prefix, '__duplicates__': []}
            if suffix in current:
                current['__duplicates__'].append(name)  # WebDataset warns and keeps the first file
                continue
            current[suffix] = (member.offset_data, member.size, name)
    if current is not None:
        yield current


def decode(encoded):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(io.BytesIO(encoded)) as image:
        rgb = image.convert('RGB')
        rgb.load()
        assert rgb.width > 0 and rgb.height > 0
        return rgb


def jpeg95(rgb):
    """Released `_deepfusion_jpeg_roundtrip` encoder applied to the decoded RGB image."""
    buffer = io.BytesIO()
    rgb.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


def coerce_image_bytes(raw, base_dir):
    """Upstream prepare_120k_mix_layout.coerce_image_bytes."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        if raw.get('bytes') is not None:
            return coerce_image_bytes(raw['bytes'], base_dir)
        if raw.get('path'):
            path = Path(base_dir) / raw['path']
            return path.read_bytes() if path.exists() else None
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, str):
        if raw.startswith('data:image'):
            try:
                return base64.b64decode(raw.split(',', 1)[1], validate=False)
            except Exception:
                return None
        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            pass
        try:
            literal = ast.literal_eval(raw)
            if isinstance(literal, (list, tuple)):
                return bytes(literal)
        except Exception:
            pass
    if isinstance(raw, (list, tuple)):
        try:
            return bytes(raw)
        except Exception:
            return None
    return None


def add_member(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def stage_dalle3_shard(task):
    """Upstream stage_dalle3_webdataset for one shard: five sorted parquets, unchanged image bytes."""
    import pyarrow.parquet as pq
    index, parquets, snapshot, output = task['index'], task['parquets'], task['snapshot'], Path(task['output'])
    candidate = output.with_name(output.name + '.partial')
    errors, origins = [], []
    with tarfile.open(candidate, 'w') as tar:
        global_idx = index * 1_000_000
        for parquet_path in parquets:
            row_offset = 0
            for batch in pq.ParquetFile(parquet_path).iter_batches(
                    batch_size=1024, columns=['caption', 'synthetic_caption', 'image', 'image_hash']):
                rows = batch.to_pylist()
                for local_offset, row in enumerate(rows):
                    image_bytes = coerce_image_bytes(row.get('image'), snapshot)
                    if not image_bytes:
                        errors.append(f'{Path(parquet_path).name}:{row_offset + local_offset}:image')
                        continue
                    caption = row.get('caption') or row.get('synthetic_caption') or ''
                    key = f"{row.get('image_hash') or 'sample'}-{global_idx:012d}"
                    add_member(tar, f'{key}.jpg', image_bytes)
                    add_member(tar, f'{key}.txt', (str(caption).strip() + '\n').encode('utf-8'))
                    origins.append([key, Path(parquet_path).name, row_offset + local_offset])
                    global_idx += 1
                row_offset += len(rows)
    os.replace(candidate, output)
    return {'index': index, 'path': str(output), 'sha256': sha256(output), 'bytes': output.stat().st_size,
            'parquets': [Path(p).name for p in parquets], 'written': len(origins), 'errors': errors, 'origins': origins}


def index_shard(task):
    """Released local-WebDataset path: sample grouping, field priority, UTF-8 caption, strict decode."""
    source, path = task['source'], Path(task['path'])
    started, fields, candidates, excluded, duplicates, count = time.monotonic(), {}, [], [], 0, 0
    with path.open('rb') as stream:
        def read(location):
            stream.seek(location[0])
            data = stream.read(location[1])
            assert len(data) == location[1], 'Source read is incomplete'
            return data
        for sample in samples(path):
            count += 1
            duplicates += len(sample['__duplicates__'])
            names = [k for k in sample if not k.startswith('__')]
            for name in names:
                fields[name] = fields.get(name, 0) + 1
            image = next((sample[k] for k in IMAGE_FIELDS[source] if k in sample), None)
            if image is None:
                excluded.append({'stage': 'selection', 'reason': 'missing-image', 'key': sample['__key__'], 'fields': names})
                continue
            try:
                caption = read(sample['txt']).decode('utf-8') if 'txt' in sample else ''
            except UnicodeDecodeError as error:
                excluded.append({'stage': 'selection', 'reason': 'caption-not-utf8', 'key': sample['__key__'], 'message': str(error)})
                continue
            try:
                decode(read(image))
                status, detail = 'accepted', None
            except DECODE_ERRORS as error:
                status, detail = 'rejected', {'error_type': type(error).__name__, 'message': str(error)}
            candidates.append([image[2], image[0], image[1], caption, status, detail])
    if any(c[4] == 'rejected' for c in candidates):
        assert sha256(path) == task['sha256'], 'Decode failure in a changed shard; do not filter it'
    return dict(task, samples=count, fields=fields, duplicates=duplicates, candidates=candidates,
                excluded=excluded, seconds=time.monotonic() - started)


def stage_sharegpt_part(task):
    """Released ShareGPT-4o path: caption-map selection, strict decode, JPEG-95 round trip, staged tar."""
    path, output, captions, filtered = Path(task['path']), Path(task['output']), task['captions'], set(task['filtered'])
    started, fields, rendered, excluded, duplicates, count, seen = time.monotonic(), {}, [], [], 0, 0, []
    candidate = output.with_name(output.name + '.partial')
    with path.open('rb') as stream, tarfile.open(candidate, 'w') as tar:
        for sample in samples(path):
            count += 1
            duplicates += len(sample['__duplicates__'])
            key = sample['__key__']
            names = [k for k in sample if not k.startswith('__')]
            for name in names:
                fields[name] = fields.get(name, 0) + 1
            if key not in captions:
                reason = 'caption-over-token-limit' if key in filtered else 'image-not-in-caption-json'
                excluded.append({'stage': 'selection', 'reason': reason, 'key': key})
                continue
            seen.append(key)
            image = next((sample[k] for k in IMAGE_FIELDS['sharegpt4o'] if k in sample), None)
            if image is None:
                excluded.append({'stage': 'selection', 'reason': 'missing-image', 'key': key, 'fields': names})
                continue
            stream.seek(image[0])
            encoded = stream.read(image[1])
            assert len(encoded) == image[1], 'Source read is incomplete'
            try:
                rgb = decode(encoded)
            except DECODE_ERRORS as error:
                rendered.append([key, image[2], None, captions[key], 'rejected',
                                 {'error_type': type(error).__name__, 'message': str(error)}])
                continue
            data = jpeg95(rgb)
            decode(data)
            add_member(tar, key + '.jpg', data)
            add_member(tar, key + '.txt', captions[key].encode('utf-8'))
            rendered.append([key, image[2], hashlib.sha256(encoded).hexdigest(), captions[key], 'accepted', None])
    os.replace(candidate, output)
    locations = {s['__key__']: s['jpg'] for s in samples(output)}
    candidates = []
    for key, member, digest, caption, status, detail in rendered:
        offset, size = locations.pop(key)[:2] if status == 'accepted' else (-1, -1)
        candidates.append([key + '.jpg', offset, size, caption, status, dict(detail or {}, origin=member, origin_sha256=digest)])
    assert not locations
    if any(c[4] == 'rejected' for c in candidates):
        assert sha256(path) == task['origin_sha256'], 'Decode failure in a changed shard; do not filter it'
    return dict(task, captions=None, filtered=None, path=str(output), sha256=sha256(output), bytes=output.stat().st_size,
                samples=count, fields=fields, duplicates=duplicates, candidates=candidates, excluded=excluded,
                seen=seen, seconds=time.monotonic() - started)


def sharegpt_captions(json_path, tokenizer_path, limit):
    """Released HFShareGPT4oTextToImageStream._load_captions, with a pinned local tokenizer."""
    from transformers import AutoTokenizer
    rows = json.loads(Path(json_path).read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    keys = [str(Path(str(row.get('output_image') or '')).with_suffix('')) for row in rows]
    texts = [str(row.get('input_prompt') or '') for row in rows]
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False, verbose=False)['input_ids']]
    captions, filtered = {}, {}
    for key, text, length in zip(keys, texts, lengths):
        if length <= limit:
            captions[key] = text + '\n'
        else:
            filtered[key] = length
    return captions, {k: v for k, v in filtered.items() if k not in captions}, {
        'rows': len(rows), 'keys': len(set(keys)), 'kept': len(captions), 'over_limit_rows': sum(n > limit for n in lengths),
        'tokenizer_class': type(tokenizer).__name__, 'max_tokens': limit, 'longest_kept': max(n for n in lengths if n <= limit)}


def integer_weights(weights):
    fractions = [Fraction(str(w)) for w in weights]
    denominator = 1
    for f in fractions:
        denominator = denominator * f.denominator // gcd(denominator, f.denominator)
    values = [int(f * denominator) for f in fractions]
    common = 0
    for v in values:
        common = gcd(common, v)
    return [v // common for v in values]


def uniform_integers(seed, domain, count, bound):
    generator = np.random.PCG64(seed_for(seed, domain, 0))
    if not 0 < bound < 2**64:
        raise ValueError('The sampling bound must be in (0, 2**64)')
    limit = (1 << 64) - ((1 << 64) % bound)
    accepted, total = [], 0
    while total < count:
        raw = generator.random_raw(count - total + 64)
        if limit < 2**64:
            raw = raw[raw < np.uint64(limit)]
        accepted.append(raw)
        total += len(raw)
    return (np.concatenate(accepted)[:count] % np.uint64(bound)).astype(np.int64)


def build_sequence(labels, names, weights, seed, samples_count):
    """Return (int32 sequence, per-source plan); depends only on record labels, weights and seed."""
    integer = integer_weights(weights)
    choice = np.searchsorted(np.cumsum(integer), uniform_integers(seed, 'finetune-source', samples_count, sum(integer)), side='right')
    sequence = np.full(samples_count, -1, dtype=np.int64)
    plan = []
    for source, name in enumerate(names):
        members = np.flatnonzero(labels == source)
        positions = np.flatnonzero(choice == source)
        n, c = len(members), len(positions)
        assert n > 0, name
        epochs = -(-c // n)
        picks = [members[np.argsort(np.random.PCG64(seed_for(seed, f'finetune-permutation-{name}', e)).random_raw(n), kind='stable')]
                 for e in range(epochs)]
        sequence[positions] = np.concatenate(picks)[:c] if picks else []
        plan.append({'name': name, 'weight': str(weights[source]), 'integer_weight': integer[source], 'records': n,
                     'positions': c, 'epochs_started': epochs, 'complete_epochs': c // n})
    assert (sequence >= 0).all() and sequence.max() < len(labels) < 2**31
    return sequence.astype(np.int32), plan


def record_labels(manifest, records):
    names = [m['name'] for m in manifest['mixture']]
    shard_source = np.array([names.index(s['source']) for s in manifest['shards']], dtype=np.int64)
    labels = shard_source[records['shard']]
    assert np.array_equal(labels, records['key'] // KEY_BASE - 1), 'Record key and shard source disagree'
    return labels, names


def write_sequence(manifest_path, output, seed, updates, batch):
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest['complete'] and sha256(manifest['records']) == manifest['records_sha256']
    records = np.load(manifest['records'], allow_pickle=False)
    labels, names = record_labels(manifest, records)
    sequence, plan = build_sequence(labels, names, [m['weight'] for m in manifest['mixture']], seed, updates * batch)
    output = Path(output)
    replace_bytes(output, lambda stream: np.save(stream, sequence, allow_pickle=False))
    receipt = {'manifest': str(Path(manifest_path).absolute()), 'manifest_sha256': sha256(manifest_path),
               'records_sha256': manifest['records_sha256'], 'sequence': str(output.absolute()), 'sequence_sha256': sha256(output),
               'seed': seed, 'updates': updates, 'global_batch': batch, 'samples': updates * batch, 'dtype': '<i4',
               'mixture': plan, 'algorithm': SEQUENCE_ALGORITHM, 'numpy': np.__version__,
               'cursor': 'Fine-tuning position = (global optimizer update - phase boundary) * global batch + rank offset; '
                         'resume reads the saved phase cursor, prefetch never advances it.',
               'scope': 'Deterministic fine-tuning order shared by the dense and HiT trainers.'}
    atomic_json(output.with_suffix('.json'), receipt)
    return receipt


def environment():
    from PIL import features
    import transformers, tokenizers, pyarrow
    return {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__,
            'Pillow': Image.__version__, 'libjpeg_turbo': features.version_feature('libjpeg_turbo'),
            'jpeg': features.version_codec('jpg'), 'zlib': features.version_codec('zlib'), 'pyarrow': pyarrow.__version__,
            'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__, 'sqlite': sqlite3.sqlite_version,
            'tarfile_format': tarfile.DEFAULT_FORMAT}


def download_all(contract, root, workers):
    from huggingface_hub import hf_hub_download
    tok = dict(contract['caption_filter']['tokenizer'], name='flan-t5-base')
    jobs = [(tok, item, 'model') for item in tok['files']]
    jobs += sorted([(s, item, 'dataset') for s in contract['sources'] for item in s['files']], key=lambda j: -j[1]['bytes'])

    def fetch(job):
        source, item, kind = job
        name = source.get('name')
        local = root / 'sources' / name
        target = local / item['filename']
        receipt = root / 'downloads' / name / (item['filename'].replace('/', '__') + '.json')
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            assert previous['sha256'] == item['sha256'] and target.stat().st_size == item['bytes']
            assert target.stat().st_mtime <= receipt.stat().st_mtime, 'Source changed after its verification'
            return previous
        if not (target.is_file() and target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']):
            path = Path(hf_hub_download(source['repository'], item['filename'], repo_type=kind,
                                        revision=source['revision'], local_dir=local))
            assert path.resolve() == target.resolve(), (path, target)
            assert path.stat().st_size == item['bytes'] and sha256(path) == item['sha256'], item['filename']
        value = dict(item, repository=source['repository'], revision=source['revision'], path=str(target))
        atomic_json(receipt, value)
        print(json.dumps({'downloaded': item['filename'], 'bytes': item['bytes']}), flush=True)
        return value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [f.result() for f in [pool.submit(fetch, job) for job in jobs]]


def cached(path, worker, task):
    """Reuse a completed step only if every payload it names still has its recorded hash."""
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text())
        assert sha256(value['path']) == value['sha256'], value['path']
        return value
    value = worker(task)
    atomic_json(path, value)
    return value


def run_stage(job):
    kind, receipt, task = job
    worker = {'dalle3': stage_dalle3_shard, 'index': index_shard, 'sharegpt4o': stage_sharegpt_part}[kind]
    return cached(receipt, worker, task)


def prepare(contract_path, root, workers, download_workers):
    started = time.monotonic()
    contract = json.loads(Path(contract_path).read_text())
    assert contract['schema'] == 1 and [s['name'] for s in contract['sources']] == ['blip3_ft60k', 'dalle3', 'sharegpt4o']
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'receipt.json').exists(), 'Already complete; run verify'
    env = environment()
    atomic_json(root / 'environment.json', env)
    downloads = download_all(contract, root, download_workers)
    download_seconds = time.monotonic() - started
    blip, dalle, share = contract['sources']
    sources = {s['name']: root / 'sources' / s['name'] for s in contract['sources']}
    assert [f['filename'] for f in blip['files']] == sorted(BLIP3_FT60K_TARS)
    parquets = sorted(sources['dalle3'].rglob('*.parquet'))
    assert [str(p.relative_to(sources['dalle3'])) for p in parquets] == [f['filename'] for f in dalle['files']]
    limit = contract['caption_filter']['max_tokens']
    captions, filtered, filter_stats = sharegpt_captions(sources['sharegpt4o'] / 'text_to_image.json', root / 'sources' / 'flan-t5-base', limit)
    parts = [f for f in share['files'] if re.fullmatch(r'text_to_image_part_\d+\.tar', f['filename'])]
    assert [f['filename'] for f in parts] == sorted(f['filename'] for f in parts) and len(parts) == 10
    for name in ('index', 'staged/dalle3', 'staged/sharegpt4o'):
        (root / name).mkdir(parents=True, exist_ok=True)
    hashes = {f['filename']: f['sha256'] for s in contract['sources'] for f in s['files']}
    ordered = [('blip3_ft60k', name) for name in BLIP3_FT60K_TARS]
    ordered += [('dalle3', f'shard-{i:03d}.tar') for i in range(ceil(len(parquets) / 5))]
    ordered += [('sharegpt4o', f['filename'].replace('.tar', '.jpeg95.tar')) for f in parts]
    results = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {}
        for name in BLIP3_FT60K_TARS:
            task = {'source': 'blip3_ft60k', 'filename': name, 'path': str(sources['blip3_ft60k'] / name), 'sha256': hashes[name]}
            futures[pool.submit(run_stage, ('index', root / 'index' / f'blip3_ft60k-{name}.json', task))] = ('index', 'blip3_ft60k', name)
        for f in parts:
            out = root / 'staged/sharegpt4o' / f['filename'].replace('.tar', '.jpeg95.tar')
            task = {'source': 'sharegpt4o', 'filename': out.name, 'path': str(sources['sharegpt4o'] / f['filename']),
                    'output': str(out), 'origin': f['filename'], 'origin_sha256': f['sha256'], 'captions': captions, 'filtered': sorted(filtered)}
            futures[pool.submit(run_stage, ('sharegpt4o', out.with_suffix('.json'), task))] = ('index', 'sharegpt4o', out.name)
        for i in range(ceil(len(parquets) / 5)):
            out = root / 'staged/dalle3' / f'shard-{i:03d}.tar'
            task = {'index': i, 'parquets': [str(p) for p in parquets[i * 5:(i + 1) * 5]], 'snapshot': str(sources['dalle3']), 'output': str(out)}
            futures[pool.submit(run_stage, ('dalle3', out.with_suffix('.json'), task))] = ('stage', 'dalle3', out.name)
        while futures:
            future = next(as_completed(futures))
            step, source, name = futures.pop(future)
            value = future.result()
            print(json.dumps({'step': step, 'source': source, 'name': name, 'candidates': len(value.get('candidates', [])),
                              'seconds': time.monotonic() - started}), flush=True)
            if step == 'stage':
                results[('staging', name)] = value
                task = {'source': 'dalle3', 'filename': name, 'path': value['path'], 'sha256': value['sha256']}
                futures[pool.submit(run_stage, ('index', root / 'index' / f'dalle3-{name}.json', task))] = ('index', 'dalle3', name)
            else:
                results[(source, name)] = value
    staging = [results[('staging', f'shard-{i:03d}.tar')] for i in range(ceil(len(parquets) / 5))]
    errors = [e for s in staging for e in s['errors']]
    if errors:
        (root / 'staged/dalle3/errors.txt').write_text('\n'.join(errors) + '\n', encoding='utf-8')
    return assemble(contract, contract_path, root, ordered, results, staging, captions, filtered, filter_stats, downloads,
                    env, started, download_seconds)


def assemble(contract, contract_path, root, ordered, results, staging, captions, filtered, filter_stats, downloads, env,
             started, download_seconds):
    names = [s['name'] for s in contract['sources']]
    shards, rows, records, exclusions, counts, ordinal = [], [], [], [], {}, {n: 0 for n in names}
    for index, (source, name) in enumerate(ordered):
        value = results[(source, name)]
        assert set(value['fields']) <= EXPECTED_FIELDS[source], (name, value['fields'])
        stats = counts.setdefault(source, {'samples': 0, 'selected': 0, 'rejected': 0, 'accepted': 0, 'excluded': {},
                                           'empty_captions': 0, 'duplicate_fields': 0})
        stats['samples'] += value['samples']
        stats['duplicate_fields'] += value['duplicates']
        excluded = list(value['excluded'])
        if source == 'dalle3':
            excluded = [{'stage': 'selection', 'reason': 'parquet-row-without-image', 'row': e} for s in staging
                        if Path(s['path']).name == name for e in s['errors']] + excluded
        for item in excluded:
            stats['excluded'][item['reason']] = stats['excluded'].get(item['reason'], 0) + 1
            if item['reason'] == 'caption-over-token-limit':
                item = dict(item, tokens=filtered[item['key']])
            exclusions.append(dict(item, source=source, shard=name))
        accepted = 0
        for member, offset, size, caption, status, detail in value['candidates']:
            key = (names.index(source) + 1) * KEY_BASE + ordinal[source]
            ordinal[source] += 1
            stats['selected'] += 1
            origin = f"{value.get('origin', name)}:{(detail or {}).get('origin', member)}"
            if status == 'accepted':
                records.append((key, offset, size, index))
                rows.append((key, source, origin, caption))
                stats['accepted'] += 1
                stats['empty_captions'] += caption.strip() == ''
                accepted += 1
            else:
                stats['rejected'] += 1
                exclusions.append({'stage': 'pixel', 'reason': 'decode-error', 'source': source, 'shard': name,
                                   'member': member, 'record_key': key, **{k: v for k, v in detail.items() if k != 'origin_sha256'}})
        origin = {'repository': contract['sources'][names.index(source)]['repository'],
                  'revision': contract['sources'][names.index(source)]['revision']}
        if source == 'dalle3':
            origin['parquets'] = next(s['parquets'] for s in staging if Path(s['path']).name == name)
        elif source == 'sharegpt4o':
            origin.update(filename=value['origin'], sha256=value['origin_sha256'], rendering='Pillow RGB, JPEG quality 95')
        else:
            origin['filename'] = name
        shards.append({'index': index, 'source': source, 'filename': name, 'path': value['path'], 'sha256': value['sha256'],
                       'bytes': Path(value['path']).stat().st_size, 'accepted': accepted, 'candidates': len(value['candidates']),
                       'samples': value['samples'], 'origin': origin})
    seen = {k for s, n in ordered if s == 'sharegpt4o' for k in results[(s, n)]['seen']}
    unmatched = sorted(set(captions) - seen)
    counts['sharegpt4o']['caption_json'] = dict(filter_stats, captions_without_image=len(unmatched))
    exclusions += [{'stage': 'selection', 'reason': 'caption-without-image', 'source': 'sharegpt4o', 'key': k} for k in unmatched]
    for source in names:
        stats = counts[source]
        assert stats['accepted'] > 0 and stats['rejected'] <= contract['limits']['max_rejected_fraction'] * stats['selected'], (source, stats)
    records = np.asarray(records, dtype=DTYPE)
    assert len(np.unique(records['key'])) == len(records), 'Duplicate fine-tuning key'
    records_path = root / 'records.npy'
    replace_bytes(records_path, lambda stream: np.save(stream, records, allow_pickle=False))
    db = root / 'captions.sqlite'
    candidate = root / 'captions.sqlite.partial'
    candidate.unlink(missing_ok=True)
    conn = sqlite3.connect(candidate)
    for pragma in ('page_size=4096', 'journal_mode=OFF', 'synchronous=OFF'):
        conn.execute('PRAGMA ' + pragma)
    conn.execute('CREATE TABLE captions (key INTEGER PRIMARY KEY,source TEXT NOT NULL,origin TEXT NOT NULL,caption TEXT NOT NULL)')
    conn.executemany('INSERT INTO captions VALUES (?,?,?,?)', rows)
    conn.commit()
    assert conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    assert conn.execute('SELECT COUNT(*) FROM captions').fetchone()[0] == len(records)
    conn.close()
    os.replace(candidate, db)
    exclusions_path = root / 'exclusions.jsonl'
    replace_bytes(exclusions_path, lambda stream: stream.write(
        ''.join(json.dumps(item, sort_keys=True, ensure_ascii=False) + '\n' for item in exclusions).encode('utf-8')))
    rejected = sum(c['rejected'] for c in counts.values())
    selected = sum(c['selected'] for c in counts.values())
    assert len(records) + rejected == selected
    expected_counts = contract.get('expected_counts')
    if expected_counts and {k: v['accepted'] for k, v in counts.items()} != expected_counts:
        raise RuntimeError('Accepted source populations differ from the SFT recipe')
    # No timing or builder fields: a rebuild at the same root reproduces this file byte for byte.
    manifest = {'schema': 1, 'complete': True, 'kind': contract['kind'], 'contract_inputs_sha256': inputs_digest(contract),
                'mixture': [{'name': s['name'], 'repository': s['repository'], 'revision': s['revision'], 'weight': s['weight']}
                            for s in contract['sources']],
                'caption_db': str(db), 'caption_db_sha256': sha256(db), 'captions_content_sha256': captions_digest(db),
                'records': str(records_path), 'records_sha256': sha256(records_path), 'count': len(records),
                'shards': shards, 'counts': counts,
                'pixel_validation': {'all_original_records_decoded': True, 'original_count': selected, 'rejected_count': rejected,
                                     'load_truncated_images': False, 'decoder': 'Pillow RGB conversion and full load'},
                'exclusions': str(exclusions_path), 'exclusions_sha256': sha256(exclusions_path),
                'preprocessing': 'PIL RGB; torchvision bicubic shortest-edge resize512; center crop512; [-1,1]. ShareGPT-4o '
                                 'bytes are the released JPEG quality-95 round trip of the decoded RGB image.',
                'captions': 'Released strings: BLIP3o-60k .txt as stored; DALL-E 3 caption or synthetic_caption, stripped, plus '
                            'newline (upstream staging); ShareGPT-4o input_prompt plus newline after the flan-t5-base '
                            f'<= {contract["caption_filter"]["max_tokens"]}-token filter.',
                'scope': 'Pinned public sources reproducing the released fine-tuning loader selection. Strict decodes replace '
                         'LOAD_TRUNCATED_IMAGES; a frozen sequence replaces its stochastic stream.'}
    atomic_json(root / 'manifest.json', manifest)
    seq = contract['sequence']
    sequence = write_sequence(root / 'manifest.json', root / 'sequence.npy', seq['seed'], seq['updates'], seq['global_batch'])
    receipt = {'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
               'contract': str(Path(contract_path).absolute()), 'contract_sha256': sha256(contract_path),
               'contract_inputs_sha256': inputs_digest(contract), 'environment': env, 'script_sha256': sha256(__file__),
               'downloads': downloads, 'download_seconds': download_seconds, 'seconds': time.monotonic() - started,
               'build_seconds': time.monotonic() - started - download_seconds,
               'manifest': str(root / 'manifest.json'), 'manifest_sha256': sha256(root / 'manifest.json'),
               'records_sha256': manifest['records_sha256'], 'count': len(records),
               'caption_db_sha256': manifest['caption_db_sha256'], 'captions_content_sha256': manifest['captions_content_sha256'],
               'exclusions_sha256': manifest['exclusions_sha256'], 'counts': counts,
               'pixel_validation': manifest['pixel_validation'],
               'shards': [{k: s[k] for k in ('index', 'source', 'filename', 'sha256', 'bytes', 'accepted', 'candidates')} for s in shards],
               'dalle3_staging': [{k: s[k] for k in ('index', 'sha256', 'bytes', 'parquets', 'written', 'errors')} for s in staging],
               'sequence': sequence}
    write_new(root / 'receipt.json', receipt)
    return {k: receipt[k] for k in ('manifest', 'manifest_sha256', 'records_sha256', 'count', 'counts')} | {
        'sequence_sha256': sequence['sequence_sha256']}


def inputs_digest(contract):
    return canonical({k: contract[k] for k in ('kind', 'upstream', 'sources', 'caption_filter', 'sequence', 'limits')})


def captions_digest(db):
    digest = hashlib.sha256()
    conn = sqlite3.connect(f'file:{db}?mode=ro&immutable=1', uri=True)
    for row in conn.execute('SELECT key,source,origin,caption FROM captions ORDER BY key'):
        digest.update((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
    conn.close()
    return digest.hexdigest()


def read_smoke(manifest_path, records, sequence, positions, scratch):
    """Trainer reads of sequence positions plus the first record of every shard."""
    from t2i.data import CC12MSequence
    firsts = sorted({int(s): i for i, s in reversed(list(enumerate(records['shard'])))}.values())
    indices = np.concatenate([np.asarray(sequence[:positions], dtype=np.int32), np.asarray(firsts, dtype=np.int32)])
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        path = Path(directory) / 'smoke.npy'
        np.save(path, indices, allow_pickle=False)
        manifest = json.loads(Path(manifest_path).read_text())
        inputs = {'records': manifest['records'], 'captions': manifest['caption_db'],
                  'sequence': str(path), 'encoder': 'unused-for-image-reads',
                  'shards': [row['path'] for row in manifest['shards']]}
        dataset = CC12MSequence(inputs)
        for position, record in enumerate(indices):
            pixels, caption, key, _ = dataset[position]
            assert key == int(records['key'][record]) and isinstance(caption, str) and tuple(pixels.shape) == (3, 512, 512)
    return {'sequence_positions': positions, 'shards_covered': len(firsts), 'reads': len(indices)}


def built_outputs(contract, root):
    """Read a completed local build; source hashes remain pinned independently."""
    root = Path(root)
    receipt = json.loads((root / 'receipt.json').read_text())
    if not receipt['passed'] or receipt['contract_inputs_sha256'] != inputs_digest(contract):
        raise ValueError('The build does not match this input contract')
    manifest = json.loads((root / 'manifest.json').read_text())
    if sha256(root / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('The completed manifest changed')
    return dict(receipt, original_count=manifest['pixel_validation']['original_count'],
                sequence=dict(receipt['sequence'], sha256=receipt['sequence']['sequence_sha256']))


def verify(contract_path, root, output, smoke=2048):
    """Content checks against the frozen contract, full sequence regeneration and real reads."""
    contract = json.loads(Path(contract_path).read_text())
    download_all(contract, Path(root).absolute(), workers=4)
    expected = contract.get('outputs')
    if expected is None:
        expected = built_outputs(contract, root)
    root = Path(root).absolute()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['contract_inputs_sha256'] == inputs_digest(contract) == expected['contract_inputs_sha256']
    assert sha256(manifest['records']) == manifest['records_sha256'] == expected['records_sha256']
    assert captions_digest(manifest['caption_db']) == manifest['captions_content_sha256'] == expected['captions_content_sha256']
    assert manifest['count'] == expected['count'] and manifest['counts'] == expected['counts']
    pixels = manifest['pixel_validation']
    assert pixels['all_original_records_decoded'] and pixels['load_truncated_images'] is False
    assert manifest['count'] + pixels['rejected_count'] == pixels['original_count'] == expected['original_count']
    assert sha256(manifest['exclusions']) == expected['exclusions_sha256']
    assert len(manifest['shards']) == len(expected['shards'])
    for row, want in zip(manifest['shards'], expected['shards']):
        for field in ('index', 'source', 'filename', 'sha256', 'bytes', 'accepted', 'candidates'):
            assert row[field] == want[field], (field, want['filename'])
        assert Path(row['path']).stat().st_size == row['bytes'] and sha256(row['path']) == row['sha256'], row['filename']
    sequence_path = root / 'sequence.npy'
    assert sha256(sequence_path) == expected['sequence']['sha256']
    seq = contract['sequence']
    records = np.load(manifest['records'], allow_pickle=False)
    labels, names = record_labels(manifest, records)
    regenerated, plan = build_sequence(labels, names, [m['weight'] for m in manifest['mixture']], seq['seed'],
                                       seq['updates'] * seq['global_batch'])
    stored = np.load(sequence_path, mmap_mode='r', allow_pickle=False)
    assert stored.dtype == np.int32 and stored.shape == regenerated.shape and np.array_equal(stored, regenerated)
    assert plan == expected['sequence']['mixture']
    smoke_result = read_smoke(manifest_path, records, stored, smoke, Path(output).absolute().parent)
    result = {'passed': True, 'utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
              'contract': str(Path(contract_path).absolute()), 'contract_sha256': sha256(contract_path),
              'environment': environment(), 'caption_db_file_sha256': sha256(manifest['caption_db']),
              'caption_db_file_matches_build': sha256(manifest['caption_db']) == expected['caption_db_sha256'],
              'manifest_file_matches_build': sha256(manifest_path) == expected.get('manifest_sha256'),
              'sequence_regenerated_exact': True, 'training_data_smoke': smoke_result,
              'reference': 'frozen-contract' if 'outputs' in contract else 'local-build-receipt',
              'data': {'manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
                       'sequence': str(sequence_path), 'sequence_sha256': expected['sequence']['sha256']}}
    write_new(output, result)
    return result


def freeze_contract(contract_path, receipt_path, output):
    contract = json.loads(Path(contract_path).read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    assert receipt['passed'] and receipt['contract_inputs_sha256'] == inputs_digest(contract)
    manifest = json.loads(Path(receipt['manifest']).read_text())
    sequence = receipt['sequence']
    frozen = copy.deepcopy(contract)
    frozen['outputs'] = {
        'contract_inputs_sha256': receipt['contract_inputs_sha256'], 'root': receipt['root'],
        'manifest_sha256': receipt['manifest_sha256'], 'count': receipt['count'],
        'original_count': manifest['pixel_validation']['original_count'], 'counts': receipt['counts'],
        'records_sha256': receipt['records_sha256'], 'captions_content_sha256': receipt['captions_content_sha256'],
        'caption_db_sha256': receipt['caption_db_sha256'], 'exclusions_sha256': receipt['exclusions_sha256'],
        'shards': receipt['shards'],
        'sequence': {k: sequence[k] for k in ('seed', 'updates', 'global_batch', 'samples', 'dtype', 'mixture', 'algorithm')}
                    | {'sha256': sequence['sequence_sha256']},
        'built_with': receipt['environment'], 'script_sha256': receipt['script_sha256']}
    Path(output).write_text(json.dumps(frozen, indent=1) + '\n')
    return {'contract': str(output), 'sha256': sha256(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--contract', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/t2i/sft_data.json')
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--workers', type=int, default=12)
    a.add_argument('--download-workers', type=int, default=6)
    a.add_argument('--download-only', action='store_true', help='Fetch and SHA-verify the pinned inputs, then stop')
    v = sub.add_parser('verify')
    v.add_argument('--contract', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/t2i/sft_data.json')
    v.add_argument('--root', type=Path, required=True)
    v.add_argument('--output', type=Path, required=True)
    v.add_argument('--smoke', type=int, default=2048, help='Sequence positions read through the trainer dataset')
    s = sub.add_parser('sequence')
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--updates', type=int, default=40000)
    s.add_argument('--batch', type=int, default=1024)
    c = sub.add_parser('freeze-contract')
    c.add_argument('--contract', type=Path, required=True)
    c.add_argument('--receipt', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'prepare' and args.download_only:
        contract = json.loads(args.contract.read_text())
        result = {'verified_inputs': len(download_all(contract, args.root.absolute(), args.download_workers))}
    elif args.action == 'prepare':
        assert args.workers <= len(os.sched_getaffinity(0))
        result = prepare(args.contract, args.root, args.workers, args.download_workers)
    elif args.action == 'verify':
        result = verify(args.contract, args.root, args.output, args.smoke)
    elif args.action == 'sequence':
        result = write_sequence(args.manifest, args.output, args.seed, args.updates, args.batch)
    else:
        result = freeze_contract(args.contract, args.receipt, args.output)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
