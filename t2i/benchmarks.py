"""Downstream benchmark prompts, rank-independent sample assignment and verified output manifests.

Standard library only (PIL for optional pixel checks): the isolated scorer environments
import this module unchanged. Prompt order, item IDs and image layouts follow the pinned
Looped-DiT evaluation (looped_dit/eval/benchmarks.py and generate.py at 18ad35c), whose
noise seed adds the rendering rank; here that rank is the canonical logical rank index % 8.
"""
from collections import Counter
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

BENCHMARKS = ('geneval', 'dpg', 'prism', 'corebench', 'spatial_geneval')
SAMPLES = {'geneval': 4, 'dpg': 4, 'prism': 1, 'corebench': 4, 'spatial_geneval': 1}
LOGICAL_RANKS = 8
BATCH_SIZE = 4
GENERATION_SCHEMA = 'minit2i-benchmark-generation-v1'
SCORE_SCHEMA = 'minit2i-benchmark-score-v1'
SEED_RULE = 'seed + index*1000 + batch_offset + (index % 8)*1000000'
PRISM_TRACKS = ('affection', 'composition', 'entity', 'imagination', 'long_text', 'style', 'text_rendering')
COREBENCH_TASKS = ('C-MA', 'C-MI', 'C-MR', 'C-TR', 'R-AR', 'R-BR', 'R-CR', 'R-GR', 'R-HR', 'R-LR', 'R-PR', 'R-RR')
DATA_FILES = {
    'geneval': ['geneval/evaluation_metadata.jsonl'],
    'dpg': ['dpg/dpg_bench.csv', 'dpg/prompts'],
    'prism': [f'prism/captions/en/{t}.jsonl' for t in PRISM_TRACKS],
    'corebench': [f'corebench/data/{t}.json' for t in COREBENCH_TASKS],
    'spatial_geneval': ['spatial_geneval/SpatialGenEval_T2I_Prompts.jsonl'],
}


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    index: int
    item_id: str
    text: str
    group: str = ''
    metadata: dict = field(default_factory=dict, compare=False)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tree_digest(directory):
    """Sorted relative file names with their SHA-256, independent of archive bytes and mtimes."""
    root = Path(directory)
    return sha256_json(sorted((str(p.relative_to(root)), sha256_file(p)) for p in root.rglob('*') if p.is_file()))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + '.' + uuid.uuid4().hex[:8] + '.partial')
    with candidate.open('w') as f:
        json.dump(value, f, indent=1, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(candidate, path)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def assets_manifest(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1]/'configs/benchmarks/assets.json'
    return json.loads(path.read_text()), sha256_file(path)


def verify_asset(root, relative, assets):
    entry = assets['files'][relative]
    target = Path(root)/relative
    if not target.exists():
        raise ManifestError(f'Missing pinned asset {target}; run scripts/benchmarks/fetch_assets.py')
    actual = tree_digest(target) if entry.get('kind') == 'tree' else sha256_file(target)
    if actual != entry['sha256']:
        raise ManifestError(f'Pinned asset differs: {target} {actual} != {entry["sha256"]}')
    return actual


def dataset_identity(benchmark, root, assets):
    files = {name: verify_asset(root, name, assets) for name in DATA_FILES[benchmark]}
    return {'benchmark': benchmark, 'files': files, 'sources': {n: assets['files'][n]['source'] for n in files}}


def dpg_questions(csv_path):
    """Official DPG-Bench rows in file order: {item_id: [(qid, question, parents, l2_category)]}."""
    items = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            items.setdefault(row['item_id'], []).append((
                int(row['proposition_id']), row['question_natural_language'],
                [int(p.strip()) for p in row['dependency'].split(',')], row['tuple'].split('(')[0].strip()))
    return items


def load_prompts(benchmark, root):
    root = Path(root)
    if benchmark == 'geneval':
        rows = read_jsonl(root/'geneval/evaluation_metadata.jsonl')
        return [Prompt(i, f'{i:05d}', row['prompt'], metadata=row) for i, row in enumerate(rows)]
    if benchmark == 'dpg':
        # Official prompts/<item_id>.txt in CSV order. This equals the Hugging Face copy that
        # Looped-DiT generates from; the CSV text column differs for 13 items.
        return [Prompt(i, item, (root/'dpg/prompts'/f'{item}.txt').read_text(encoding='utf-8'))
                for i, item in enumerate(dpg_questions(root/'dpg/dpg_bench.csv'))]
    items = []
    if benchmark == 'prism':
        for track in PRISM_TRACKS:
            for j, row in enumerate(read_jsonl(root/f'prism/captions/en/{track}.jsonl')):
                items.append((f'{track}-{j:03d}', row['prompt'], track, row))
    elif benchmark == 'corebench':
        for task in COREBENCH_TASKS:
            rows = json.loads((root/f'corebench/data/{task}.json').read_text(encoding='utf-8'))
            items += [(item, row['Prompt'], task, row) for item, row in rows.items()]
    elif benchmark == 'spatial_geneval':
        items = [(str(row['id']), row['prompt'], row['scene'], row)
                 for row in read_jsonl(root/'spatial_geneval/SpatialGenEval_T2I_Prompts.jsonl')]
    else:
        raise ValueError(benchmark)
    return [Prompt(i, item, text, group, row) for i, (item, text, group, row) in enumerate(items)]


def select(prompts, indices=None):
    if indices is None:
        return list(prompts)
    if len(set(indices)) != len(indices) or not all(0 <= i < len(prompts) for i in indices):
        raise ValueError('Subset indices must be unique original prompt indices')
    return [prompts[i] for i in sorted(indices)]


def batch_seed(seed, index, offset):
    return seed + index*1000 + offset + (index % LOGICAL_RANKS)*1_000_000


def canonical_batches(samples):
    """Released per-prompt sampling batches as (offset, size)."""
    return [(offset, min(BATCH_SIZE, samples - offset)) for offset in range(0, samples, BATCH_SIZE)]


def owner(position, world):
    """Whole prompts go to physical ranks; seeds never depend on this choice."""
    return position % world


def _safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', name.strip()).strip('-.') or 'item'


def sample_path(benchmark, prompt, sample):
    if benchmark == 'geneval':
        return f'{prompt.item_id}/samples/{sample:05d}.png'
    if benchmark == 'dpg':
        return f'{prompt.item_id}.png'
    return f'{_safe(prompt.group)}/{_safe(prompt.item_id)}/{sample:05d}.png'


def grid_box(sample, size=512):
    """2x2 DPG grid order: (0,0), (w,0), (0,h), (w,h)."""
    x, y = (sample % 2)*size, (sample//2)*size
    return [x, y, x + size, y + size]


def expected_entries(benchmark, prompts, seed, samples=None):
    samples = samples or SAMPLES[benchmark]
    rows = []
    for prompt in prompts:
        for offset, size in canonical_batches(samples):
            for position in range(size):
                row = {'index': prompt.index, 'item_id': prompt.item_id, 'group': prompt.group,
                       'sample': offset + position, 'logical_rank': prompt.index % LOGICAL_RANKS,
                       'batch_offset': offset, 'batch_size': size, 'position_in_batch': position,
                       'batch_seed': None if seed is None else batch_seed(seed, prompt.index, offset),
                       'path': sample_path(benchmark, prompt, offset + position)}
                if benchmark == 'dpg':
                    row['box'] = grid_box(offset + position)
                rows.append(row)
    return rows


def output_key(identity):
    model, sampling = identity['model'], identity['sampling']
    if model['kind'] == 'frozen-reference':
        head = f"frozen-{model['sources_sha256'][:12]}"
    else:
        head = f"{model['arm']}-u{model['progress']['global_step']:06d}-ema{model['ema_decay']:g}-{model['checkpoint_sha256'][:12]}"
    return f"{head}/{identity['dataset']['benchmark']}-seed{sampling['seed']}-cfg{sampling['cfg']:g}-{sha256_json(identity)[:12]}"


def _check(condition, message):
    if not condition:
        raise ManifestError(message)


def physical_device(index):
    """GPU index relative to an inherited CUDA_VISIBLE_DEVICES, for child processes that set their own."""
    visible = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()]
    return visible[int(index)] if visible else str(index)


def parse_expectations(items):
    """'model.ema_decay=0.99995' -> {'model.ema_decay': 0.99995}; non-JSON values stay strings."""
    def value(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return {k: value(v) for k, v in (item.split('=', 1) for item in items)}


def lookup(identity, dotted):
    node = identity
    for key in dotted.split('.'):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def verify_generation(directory, expected=None, pixels=True):
    """Rejects incomplete, duplicated, modified, stray or differently keyed benchmark outputs.

    expected: identity fields that must match, e.g. {'model.checkpoint_sha256': ...}.
    """
    directory = Path(directory)
    path = directory/'manifest.json'
    _check(path.exists(), f'No complete generation manifest in {directory}')
    manifest = json.loads(path.read_text())
    _check(manifest.get('schema') == GENERATION_SCHEMA, f'Unexpected manifest schema in {path}')
    identity = manifest['identity']
    _check(sha256_json(identity) == manifest['identity_sha256'], 'Manifest identity hash mismatch')
    for dotted, value in (expected or {}).items():
        actual = lookup(identity, dotted)
        _check(actual == value, f'Generation identity {dotted}={actual!r}, expected {value!r}: wrong checkpoint or settings')
    benchmark = identity['dataset']['benchmark']
    _check(benchmark in BENCHMARKS, benchmark)
    samples, seed = identity['sampling']['samples_per_prompt'], identity['sampling']['seed']
    entries, indices = manifest['entries'], manifest['indices']
    counts = Counter((e['index'], e['sample']) for e in entries)
    duplicates = sorted(k for k, n in counts.items() if n > 1)
    _check(not duplicates, f'Duplicated benchmark samples in manifest: {duplicates[:5]}')
    _check(len(set(indices)) == len(indices), 'Duplicated prompt indices in manifest')
    want = {(i, s) for i in indices for s in range(samples)}
    missing, extra = sorted(want - set(counts)), sorted(set(counts) - want)
    _check(not missing, f'Missing benchmark samples (index, sample) {missing[:5]} ({len(missing)} total)')
    _check(not extra, f'Unexpected benchmark samples {extra[:5]}')
    for e in entries:
        _check(e['logical_rank'] == e['index'] % LOGICAL_RANKS, f'Logical rank mismatch for {e["path"]}')
        if seed is not None:
            _check(e['batch_seed'] == batch_seed(seed, e['index'], e['batch_offset']), f'Noise seed mismatch for {e["path"]}')
    images, files = directory/'images', manifest['files']
    present = {str(p.relative_to(images)) for p in images.rglob('*') if p.is_file()}
    absent, stray = sorted(set(files) - present), sorted(present - set(files))
    _check(not absent, f'Missing image files: {absent[:5]} ({len(absent)} total)')
    _check(not stray, f'Unrecorded files in image tree: {stray[:5]}')
    _check({e['path'] for e in entries} <= set(files), 'Manifest entries reference unrecorded files')
    for name, digest in files.items():
        _check(sha256_file(images/name) == digest, f'Image file changed after generation: {name}')
    if pixels:
        from PIL import Image
        current = None
        for e in sorted(entries, key=lambda e: (e['path'], e['sample'])):
            if current is None or current[0] != e['path']:
                with Image.open(images/e['path']) as im:
                    current = (e['path'], im.convert('RGB'))
            image = current[1].crop(tuple(e['box'])) if 'box' in e else current[1]
            _check(hashlib.sha256(image.tobytes()).hexdigest() == e['pixel_sha256'],
                   f'Decoded pixels differ for {e["path"]} sample {e["sample"]}')
    return manifest


def score_directory(generation, scorer, scorer_identity, output=None, restart=False):
    """Results are keyed by both identities; a different scorer or generation never reuses a folder.

    Returns (directory, already_complete). An incomplete folder of the same key is rejected unless
    restart, which discards it (e.g. after preemption).
    """
    generation = Path(generation)
    manifest = json.loads((generation/'manifest.json').read_text())
    key = sha256_json({'generation': manifest['identity_sha256'], 'scorer': scorer_identity})
    out = Path(output) if output else generation/'scores'/f'{scorer}-{key[:12]}'
    if (out/'complete.json').exists():
        done = json.loads((out/'complete.json').read_text())
        _check(done['generation_identity_sha256'] == manifest['identity_sha256'] and done['scorer_identity'] == scorer_identity,
               f'{out} holds results for a different generation or scorer')
        return out, True
    if out.exists() and restart and output is None:
        import shutil
        shutil.rmtree(out)
    _check(not out.exists(), f'{out} exists without completion; remove it after inspection or choose another output')
    return out, False


def score_record(out, benchmark, manifest, scorer_identity, metrics, primary, extra=None):
    from datetime import datetime, timezone
    summary = {'schema': SCORE_SCHEMA, 'benchmark': benchmark, 'primary': primary, 'metrics': metrics,
               'subset': manifest['identity']['dataset']['subset'], 'generation_identity_sha256': manifest['identity_sha256'],
               'generation_identity': manifest['identity'], 'scorer_identity': scorer_identity,
               'scorer_identity_sha256': sha256_json(scorer_identity), **(extra or {})}
    atomic_json(Path(out)/'summary.json', summary)
    atomic_json(Path(out)/'complete.json', {'generation_identity_sha256': manifest['identity_sha256'], 'scorer_identity': scorer_identity,
                                            'utc': datetime.now(timezone.utc).isoformat(),
                                            'summary_sha256': sha256_file(Path(out)/'summary.json')})
    return summary
