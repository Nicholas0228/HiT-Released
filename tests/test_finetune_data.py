"""Fine-tuning mixture build on synthetic sources: released selection, strict decodes, trainer reads, sequence."""
import collections
import io
import json
from math import ceil
from pathlib import Path
import sqlite3
import tarfile

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from t2i.data import CC12MSequence
from t2i import finetune_data as fd
import sft_fixtures as fx


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    root = tmp_path_factory.mktemp('mixture')
    fx.sources(root)
    fx.contract(root, root / 'contract.json')
    result = fd.prepare(root / 'contract.json', root, workers=2, download_workers=2)
    return root, result, json.loads((root / 'manifest.json').read_text())


def released_pixels(encoded, source):
    """WebDataset 'pil' decode, the released ShareGPT-4o JPEG round trip and the released transform."""
    image = Image.open(io.BytesIO(encoded))
    image.load()
    image = image.convert('RGB')
    if source == 'sharegpt4o':
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        image = Image.open(buffer).convert('RGB')
        image.load()
    return transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(512), transforms.PILToTensor()])(image.convert('RGB'))


def member(path, name):
    with tarfile.open(path) as tar:
        return tar.extractfile(name).read()


def upstream_stage_dalle3(snapshot, out_root):
    """prepare_120k_mix_layout.stage_dalle3_webdataset (e54e15d) after snapshot_download, transcribed."""
    import pyarrow.parquet as pq
    parquets = sorted(snapshot.rglob('*.parquet'))
    dst_dir = out_root / 'dalle3'
    dst_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx in range(ceil(len(parquets) / 5)):
        with tarfile.open(dst_dir / f'shard-{shard_idx:03d}.tar', 'w') as tar:
            global_idx = shard_idx * 1_000_000
            for parquet_path in parquets[shard_idx * 5:(shard_idx + 1) * 5]:
                for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        image_bytes = fd.coerce_image_bytes(row.get('image'), snapshot)
                        if not image_bytes:
                            continue
                        caption = row.get('caption') or row.get('synthetic_caption') or ''
                        key = f"{row.get('image_hash') or 'sample'}-{global_idx:012d}"
                        info = tarfile.TarInfo(f'{key}.jpg')
                        info.size = len(image_bytes)
                        tar.addfile(info, io.BytesIO(image_bytes))
                        text = (str(caption).strip() + '\n').encode('utf-8')
                        info = tarfile.TarInfo(f'{key}.txt')
                        info.size = len(text)
                        tar.addfile(info, io.BytesIO(text))
                        global_idx += 1
    return sorted(dst_dir.glob('shard-*.tar'))


def test_released_selection_counts_and_exclusions(build):
    root, result, manifest = build
    counts = manifest['counts']
    assert {k: (v['selected'], v['rejected'], v['accepted']) for k, v in counts.items()} == {
        'blip3_ft60k': (31, 1, 30), 'dalle3': (20, 0, 20), 'sharegpt4o': (28, 1, 27)}
    assert counts['blip3_ft60k']['excluded'] == {'missing-image': 1, 'caption-not-utf8': 1}
    assert counts['blip3_ft60k']['empty_captions'] == 1
    assert counts['dalle3']['excluded'] == {'parquet-row-without-image': 1}
    assert counts['sharegpt4o']['excluded'] == {'caption-over-token-limit': 1, 'image-not-in-caption-json': 1}
    assert counts['sharegpt4o']['caption_json']['captions_without_image'] == 1
    pixels = manifest['pixel_validation']
    assert pixels['original_count'] == 79 and pixels['rejected_count'] == 2 and manifest['count'] == 77
    assert pixels['all_original_records_decoded'] and pixels['load_truncated_images'] is False
    excluded = [json.loads(line) for line in (root / 'exclusions.jsonl').read_text().splitlines()]
    assert collections.Counter((x['stage'], x['reason']) for x in excluded) == {
        ('selection', 'missing-image'): 1, ('selection', 'caption-not-utf8'): 1, ('pixel', 'decode-error'): 2,
        ('selection', 'parquet-row-without-image'): 1, ('selection', 'caption-over-token-limit'): 1,
        ('selection', 'image-not-in-caption-json'): 1, ('selection', 'caption-without-image'): 1}
    assert next(x for x in excluded if x['reason'] == 'caption-over-token-limit')['tokens'] == 15
    assert next(x for x in excluded if x['reason'] == 'parquet-row-without-image')['row'] == 'train-00001-of-00007.parquet:0:image'
    assert (root / 'staged/dalle3/errors.txt').read_text() == 'train-00001-of-00007.parquet:0:image\n'
    records = np.load(manifest['records'])
    assert records.dtype == fd.DTYPE and len(np.unique(records['key'])) == len(records) == manifest['count']
    assert [s['source'] for s in manifest['shards']] == ['blip3_ft60k'] * 11 + ['dalle3'] * 2 + ['sharegpt4o'] * 10
    assert sum(s['accepted'] for s in manifest['shards']) == manifest['count']


def test_unchanged_training_data_reads_released_pixels_and_captions(build, tmp_path):
    root, _, manifest = build
    records = np.load(manifest['records'])
    np.save(tmp_path / 'all.npy', np.arange(len(records), dtype=np.int32))
    dataset = CC12MSequence(dict(records=manifest['records'], captions=manifest['caption_db'],
                                 sequence=str(tmp_path / 'all.npy'), encoder='unused',
                                 shards=[s['path'] for s in manifest['shards']]))
    conn = sqlite3.connect(manifest['caption_db'])
    origin = dict(conn.execute('SELECT key,origin FROM captions'))
    staging = {s['index']: s for s in (json.loads(p.read_text()) for p in (root / 'staged/dalle3').glob('shard-*.json'))}
    seen = collections.Counter()
    for position, row in enumerate(records):
        pixels, caption, key, _ = dataset[position]
        source = manifest['shards'][int(row['shard'])]['source']
        shard, name = origin[key].split(':', 1)
        if source == 'blip3_ft60k':
            encoded = member(root / 'sources/blip3_ft60k' / shard, name)
            with tarfile.open(root / 'sources/blip3_ft60k' / shard) as tar:
                text = name[:-4] + '.txt'
                expected = tar.extractfile(text).read().decode() if text in tar.getnames() else ''
        elif source == 'dalle3':
            import pyarrow.parquet as pq
            stem = name[:-4]
            _, parquet, index = next(o for o in staging[int(shard[6:9])]['origins'] if o[0] == stem)
            table = pq.read_table(root / 'sources/dalle3/data' / parquet).to_pylist()[index]
            encoded = table['image']['bytes']
            expected = (table['caption'] or table['synthetic_caption'] or '').strip() + '\n'
        else:
            encoded = member(root / 'sources/sharegpt4o' / shard, name)
            prompts = {e['output_image']: e['input_prompt'] for e in json.loads((root / 'sources/sharegpt4o/text_to_image.json').read_text())}
            expected = prompts[name] + '\n'
        assert caption == expected, (source, key)
        assert np.array_equal(pixels.numpy(), released_pixels(encoded, source).numpy()), (source, key)
        seen[source] += 1
    assert seen == {'blip3_ft60k': 30, 'dalle3': 20, 'sharegpt4o': 27}


def test_dalle3_staging_matches_upstream_tool_bytes(build, tmp_path):
    root, _, manifest = build
    upstream = upstream_stage_dalle3(root / 'sources/dalle3', tmp_path)
    ours = [Path(s['path']) for s in manifest['shards'] if s['source'] == 'dalle3']
    assert [p.name for p in upstream] == [p.name for p in ours]
    assert [p.read_bytes() for p in upstream] == [p.read_bytes() for p in ours]


def test_webdataset_grouping_rules(tmp_path):
    path = tmp_path / 'grouping.tar'
    fx.write_tar(path, [('a.jpg', b'1'), ('a.txt', b'x'), ('b.txt', b'y'), ('a.cls', b'2'), ('__meta__', b''),
                        ('noext', b''), ('c.JPG', b'3'), ('c.jpg', b'4'), ('dir/d.e.png', b'5')])
    rows = [{k: v[2] if isinstance(v, tuple) else v for k, v in s.items()} for s in fd.samples(path)]
    assert rows == [{'__key__': 'a', '__duplicates__': [], 'jpg': 'a.jpg', 'txt': 'a.txt'},
                    {'__key__': 'b', '__duplicates__': [], 'txt': 'b.txt'},
                    {'__key__': 'a', '__duplicates__': [], 'cls': 'a.cls'},
                    {'__key__': 'c', '__duplicates__': ['c.jpg'], 'jpg': 'c.JPG'},
                    {'__key__': 'dir/d', '__duplicates__': [], 'e.png': 'dir/d.e.png'}]


def test_sequence_protocol_and_exact_regeneration(build):
    root, result, manifest = build
    sequence = np.load(root / 'sequence.npy')
    receipt = json.loads((root / 'sequence.json').read_text())
    assert sequence.dtype == np.int32 and sequence.shape == (8 * 1024,) and receipt['sequence_sha256'] == result['sequence_sha256']
    records = np.load(manifest['records'])
    labels, names = fd.record_labels(manifest, records)
    again, plan = fd.build_sequence(labels, names, ['0.06', '0.016', '0.04'], 42, 8 * 1024)
    assert np.array_equal(again, sequence) and plan == receipt['mixture']
    assert [p['integer_weight'] for p in plan] == [15, 4, 10]
    other, _ = fd.build_sequence(labels, names, ['0.06', '0.016', '0.04'], 43, 8 * 1024)
    assert not np.array_equal(other, sequence)
    shares = np.bincount(labels[sequence], minlength=3) / len(sequence)
    assert np.allclose(shares, [15 / 29, 4 / 29, 10 / 29], atol=.02)
    for source in range(3):
        members = np.flatnonzero(labels == source)
        drawn = sequence[labels[sequence] == source]
        for epoch in range(len(drawn) // len(members)):
            assert sorted(drawn[epoch * len(members):(epoch + 1) * len(members)]) == list(members)


def test_frozen_contract_verify_and_tamper_detection(build, tmp_path):
    root, _, _ = build
    frozen = tmp_path / 'frozen.json'
    fd.freeze_contract(root / 'contract.json', root / 'receipt.json', frozen)
    verified = fd.verify(frozen, root, tmp_path / 'verified.json', smoke=64)
    assert verified['passed'] and verified['sequence_regenerated_exact'] and verified['training_data_smoke']['shards_covered'] == 23
    assert verified['data']['sequence_sha256'] == json.loads(frozen.read_text())['outputs']['sequence']['sha256']
    changed = json.loads(frozen.read_text())
    changed['outputs']['shards'][12]['sha256'] = '0' * 64
    (tmp_path / 'changed.json').write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        fd.verify(tmp_path / 'changed.json', root, tmp_path / 'never.json', smoke=0)
    assert not (tmp_path / 'never.json').exists()


def test_rebuild_at_the_same_root_is_byte_identical(build, tmp_path):
    root, result, _ = build
    names = ('manifest.json', 'records.npy', 'captions.sqlite', 'exclusions.jsonl', 'sequence.npy', 'sequence.json')
    before = {name: (root / name).read_bytes() for name in names}
    for name in ('manifest.json', 'receipt.json', 'sequence.npy', 'sequence.json'):
        (root / name).rename(tmp_path / name)
    again = fd.prepare(root / 'contract.json', root, workers=2, download_workers=2)
    assert {name: (root / name).read_bytes() for name in names} == before
    assert again['manifest_sha256'] == result['manifest_sha256']
    assert json.loads((tmp_path / 'receipt.json').read_text())['build_seconds'] > 0


def test_rejected_fraction_limit_stops_the_build(tmp_path):
    fx.sources(tmp_path)
    value = fx.contract(tmp_path, tmp_path / 'contract.json')
    value['limits']['max_rejected_fraction'] = .01
    (tmp_path / 'contract.json').write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        fd.prepare(tmp_path / 'contract.json', tmp_path, workers=2, download_workers=2)
    assert not (tmp_path / 'manifest.json').exists()


def test_local_build_receipt_and_training_inputs(build, tmp_path):
    from hnet.checkpoints import atomic_json, sha256_file
    from scripts.prepare_t2i_finetune import write_inputs
    from t2i.finetune import load_sft_inputs
    root, _, _ = build
    verified = fd.verify(root / 'contract.json', root, tmp_path / 'local-verify.json', smoke=32)
    assert verified['reference'] == 'local-build-receipt'
    atomic_json(root / 'verified.json', verified)
    path = write_inputs(root, root / 'sources/flan-t5-base', root / 'contract.json')
    inputs = json.loads(path.read_text())
    # Small global batch for this input-validation fixture; production is 1024.
    sequence = np.resize(np.load(inputs['sequence']), 40000 * 4).astype(np.int32)
    np.save(tmp_path / 'sequence.npy', sequence)
    inputs['sequence'] = str(tmp_path / 'sequence.npy')
    inputs['fingerprint']['sequence_sha256'] = sha256_file(inputs['sequence'])
    atomic_json(tmp_path / 'inputs.json', inputs)
    recipe = dict(phase='sft40k', data_update_offset=250000, updates=290000, global_batch=4)
    assert load_sft_inputs(tmp_path / 'inputs.json', recipe) == inputs
    with (tmp_path / 'sequence.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_sft_inputs(tmp_path / 'inputs.json', recipe)


def test_interrupted_sequence_build_recovers_from_verified_shards(tmp_path, monkeypatch):
    fx.sources(tmp_path)
    fx.contract(tmp_path, tmp_path / 'contract.json')
    original = fd.write_sequence
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('interrupted after sequence publication')
    monkeypatch.setattr(fd, 'write_sequence', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        fd.prepare(tmp_path / 'contract.json', tmp_path, workers=2, download_workers=2)
    assert (tmp_path / 'manifest.json').exists() and not (tmp_path / 'receipt.json').exists()
    before = (tmp_path / 'sequence.npy').read_bytes()
    monkeypatch.setattr(fd, 'write_sequence', original)
    fd.prepare(tmp_path / 'contract.json', tmp_path, workers=2, download_workers=2)
    assert (tmp_path / 'sequence.npy').read_bytes() == before
    assert fd.verify(tmp_path / 'contract.json', tmp_path, tmp_path / 'verified.json', smoke=16)['passed']
