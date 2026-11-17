"""PRISM, T2I-CoReBench and SpatialGenEval adapter around the pinned Looped-DiT vLLM judge entry points.

    $JUDGE_ENV/bin/python scripts/benchmarks/score_judge.py --generation GEN_DIR --assets ASSET_ROOT \
        --judge-model SCORERS/judges/Qwen2.5-VL-72B-Instruct-89c86200743e [--tensor-parallel-size 8]

Our image layout is Looped-DiT's (<group>/<item>/<sample>.png), so its judge.py (PRISM with the released
eval_qwen25.py rubric, CoReBench) and spatial_geneval.py run unmodified on the generation folder with the
arguments its run.py passes. This wrapper pins the inputs, the judge weights (fetch_judges.py receipt) and
the judge settings and runs the judge offline. It then requires every expected judgment of every manifest
image to be present exactly once (or among the unparseable answers upstream tolerates), all metrics present
and finite, and the upstream summary to equal an independent re-aggregation, before writing a keyed summary.
An interrupted run of the same key resumes from the judge's own cache (judge/results.jsonl).
Judges: Qwen2.5-VL-72B-Instruct (PRISM, SpatialGenEval) and Qwen3-VL-32B-Thinking (CoReBench).
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from t2i import benchmarks as B  # noqa: E402
from scripts.benchmarks.fetch_judges import judge_receipt, load_pin  # noqa: E402

JUDGES = {'prism': 'Qwen/Qwen2.5-VL-72B-Instruct', 'corebench': 'Qwen/Qwen3-VL-32B-Thinking',
          'spatial_geneval': 'Qwen/Qwen2.5-VL-72B-Instruct'}
DATA = {'prism': 'prism/captions/en', 'corebench': 'corebench/data', 'spatial_geneval': 'spatial_geneval/SpatialGenEval_T2I_Prompts.jsonl'}
REFERENCE = ('__init__.py', 'eval/__init__.py', 'eval/benchmarks.py', 'eval/judge.py', 'eval/spatial_geneval.py')
# Generation settings of the pinned reference code (judge.py DEFAULTS/VLLMJudge, spatial_geneval.py argparse
# defaults), recorded in the scorer identity; tests/test_judges.py checks them against that code.
COMMON = {'gpu_memory_utilization': 0.9, 'limit_mm_per_prompt': {'image': 8, 'video': 0}, 'mm_encoder_tp_mode': 'data',
          'decoding': 'vLLM SamplingParams(temperature, repetition_penalty, max_tokens, n); top_p 1, top_k off'}
SETTINGS = {
    'prism': {'temperature': 0.0, 'repetition_penalty': 1.05, 'max_new_tokens': 1024, 'max_model_len': 16384, 'batch_size': 64,
              'retries': 3, 'max_failed': 8, 'image_first': True, 'vision_id': True, 'min_pixels': 256*28*28,
              'max_pixels': 1280*28*28, 'system': 'You are a helpful assistant.', 'rubric': 'eval_qwen25.py messages_1..8'},
    'corebench': {'temperature': 0.0, 'repetition_penalty': 1.0, 'max_new_tokens': 4096, 'max_model_len': 32768, 'batch_size': 256,
                  'retries': 3, 'max_failed': 64, 'dtype': 'bfloat16', 'executor': 'mp (spawn)',
                  'system': 'Looped-DiT COREBENCH_SYSTEM', 'question': 'one yes/no question per checklist item'},
    'spatial_geneval': {'temperature': 1.0, 'repetition_penalty': 1.0, 'max_new_tokens': 2048, 'max_model_len': 16384,
                        'batch_size': 32, 'retries': 5, 'max_failed': 0, 'rollouts': 5, 'majority': 4,
                        'system': 'Looped-DiT SYSTEM_PROMPT', 'question': 'ten multiple-choice questions, A-E'},
}
# vLLM general plugin on the judge's path (see its docstring): FlashAttention-2 instead of the aborting FA4 on B300.
RUNTIME = Path(__file__).resolve().parent/'judge_runtime'
MAX_SCORE = {'prism': 10, 'corebench': 1}
OPTIONS = set('ABCDE')
TOLERANCE = dict(rel_tol=1e-12, abs_tol=1e-9)
VLLM_TIMES = {'weights_seconds': r'Loading weights took ([\d.]+) s', 'model_load_seconds': r'Model loading took [\d.]+ \w+(?: memory)? and ([\d.]+) s',
              'engine_init_seconds': r'init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s'}


def environment():
    """Every installed distribution of this interpreter (the frozen judge lock), as one digest."""
    names = sorted({f'{(d.metadata["Name"] or "?").lower()}=={d.version}' for d in importlib.metadata.distributions()})

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {'python': sys.version.split()[0], 'distributions': len(names), 'sha256': B.sha256_json(names),
            'packages': {k: version(k) for k in ('vllm', 'transformers', 'qwen-vl-utils', 'torch', 'pillow')}}


def runtime_identity():
    return {str(p.relative_to(RUNTIME)): B.sha256_file(p) for p in sorted(RUNTIME.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def judge_path(looped_dit):
    """PYTHONPATH of a judge process: the pinned Looped-DiT package and the vLLM runtime plugin."""
    return os.pathsep.join([str(looped_dit), str(RUNTIME)])


def gpus():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    visible = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()]
    rows = [line.strip() for line in out.splitlines() if line.strip()]
    return [rows[int(d)] for d in visible if d.isdigit() and int(d) < len(rows)] if visible else rows


def subset_data(benchmark, assets_root, prompts, work):
    """Upstream scripts require every image of their data; give them exactly the selected rows."""
    source = Path(assets_root)/DATA[benchmark]
    if benchmark == 'spatial_geneval':
        keep = {p.item_id for p in prompts}
        target = work/'SpatialGenEval_T2I_Prompts.jsonl'
        target.write_text(''.join(json.dumps(r) + '\n' for r in B.read_jsonl(source) if str(r['id']) in keep))
        return target
    target = work/'data'; target.mkdir(parents=True)
    if benchmark == 'corebench':
        for task in B.COREBENCH_TASKS:
            rows = json.loads((source/f'{task}.json').read_text(encoding='utf-8'))
            (target/f'{task}.json').write_text(json.dumps({k: v for k, v in rows.items() if k in {p.item_id for p in prompts}}))
        return target
    for track in B.PRISM_TRACKS:
        positions = sorted(int(p.item_id.rsplit('-', 1)[1]) for p in prompts if p.group == track)
        if positions != list(range(len(positions))):
            raise ValueError('PRISM item IDs are row positions; a subset must be a prefix of each track')
        rows = B.read_jsonl(source/f'{track}.jsonl')[:len(positions)]
        (target/f'{track}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return target


def expected_judgments(benchmark, prompts, samples):
    """Key -> (prompt, sample, metric) of every judgment the reference makes (judge.py Request.key)."""
    out = {}
    for prompt in prompts:
        if benchmark == 'prism':
            metrics = [('alignment', 0), ('aesthetic', 0)]
        elif benchmark == 'corebench':
            metrics = [('checklist', i) for i in range(len(prompt.metadata['Checklist']))]
        else:
            metrics = [('multiple_choice', 0)]
        for sample in range(samples):
            for metric, detail in metrics:
                out[f'{prompt.item_id}|{sample}|{metric}|{detail}'] = (prompt, sample, metric)
    return out


def read_results(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    duplicates = sorted(k for k, n in Counter(r['key'] for r in rows).items() if n > 1)
    if duplicates:
        raise B.ManifestError(f'Duplicated judgments in {path}: {duplicates[:5]}')
    return {r['key']: r for r in rows}


def _finite(value, low, high):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high


def aggregate(benchmark, expected, records):
    """Independent re-aggregation of the records into the reference summary fields (0-100)."""
    def mean(values):
        values = list(values)
        return math.fsum(values)/len(values)
    if benchmark == 'spatial_geneval':
        rows = list(records.values())
        scenes = sorted({r['scene'] for r in rows})
        return {'overall': 100*mean(c for r in rows for c in r['correct']), 'num_images': len(rows),
                'scenes': {s: 100*mean(c for r in rows if r['scene'] == s for c in r['correct']) for s in scenes}}
    per_image = {}
    for key, r in records.items():
        prompt, sample, metric = expected[key]
        per_image.setdefault((prompt.group, prompt.item_id, sample, metric), []).append(100*r['score']/r['max_score'])
    scores = {k: mean(v) for k, v in per_image.items()}

    def by(field):
        return {name: mean(s for k, s in scores.items() if k[field] == name) for name in sorted({k[field] for k in scores})}
    return {'overall': mean(scores.values()), 'metrics': by(3), 'groups': by(0), 'num_images': len({k[:3] for k in scores}),
            'num_judgments': len(records)}


def check_judgments(benchmark, expected, records, summary):
    """Exact accounting of the judge output against the manifest; returns the independent aggregate."""
    stray = sorted(set(records) - set(expected))
    if stray:
        raise B.ManifestError(f'Judgments outside the manifest: {stray[:5]}')
    missing = sorted(set(expected) - set(records))
    tolerated = SETTINGS[benchmark]['max_failed']
    if len(missing) > tolerated or summary.get('failed_judgments', 0) != len(missing):
        raise B.ManifestError(f'{len(missing)} expected judgments missing (upstream reports {summary.get("failed_judgments", 0)} '
                              f'unparseable, tolerance {tolerated}): {missing[:5]}')
    for key, r in records.items():
        prompt, sample, metric = expected[key]
        if benchmark == 'spatial_geneval':
            answers = [str(a).upper() for a in prompt.metadata['answers']]
            rollouts = r['rollouts']
            ok = (r['item_id'], r['scene']) == (prompt.item_id, prompt.group) and len(rollouts) == SETTINGS[benchmark]['rollouts'] \
                and all(len(x) == len(answers) and set(x) <= OPTIONS for x in rollouts) and r['correct'] == [
                    Counter(x[i] for x in rollouts)[a] >= SETTINGS[benchmark]['majority'] for i, a in enumerate(answers)]
        else:
            ok = (r['item_id'], r['group'], r['sample'], r['metric'], r['max_score']) == \
                (prompt.item_id, prompt.group, sample, metric, MAX_SCORE[benchmark]) and _finite(r['score'], 0, r['max_score'])
        if not ok:
            raise B.ManifestError(f'Judgment {key} is malformed, non-finite or inconsistent with its prompt: {str(r)[:300]}')
    if not records:
        raise B.ManifestError('No judgments')
    ours = aggregate(benchmark, expected, records)
    groups = sorted({p.group for p, _, _ in (expected[k] for k in records)})
    fields = {'overall': None, 'num_images': None}
    if benchmark == 'spatial_geneval':
        fields['scenes'] = groups
    else:
        fields.update(metrics=sorted({expected[k][2] for k in records}), groups=groups, num_judgments=None)
    for field, names in fields.items():
        theirs, mine = summary.get(field), ours[field]
        if names is not None:
            if not isinstance(theirs, dict) or sorted(theirs) != names or sorted(mine) != names \
                    or not all(_finite(theirs[n], 0, 100) and math.isclose(theirs[n], mine[n], **TOLERANCE) for n in names):
                raise B.ManifestError(f'Summary field {field} is incomplete, non-finite or differs from the records: {theirs} vs {mine}')
        elif field.startswith('num_'):
            if theirs != mine:
                raise B.ManifestError(f'Summary {field}={theirs}, records give {mine}')
        elif not (_finite(theirs, 0, 100) and math.isclose(theirs, mine, **TOLERANCE)):
            raise B.ManifestError(f'Summary overall={theirs!r} is non-finite or differs from the records ({mine!r})')
    return ours


def output_directory(generation, benchmark, scorer, output, restart):
    """B.score_directory, except that an unfinished folder of the same key resumes from the judge's own cache."""
    manifest = json.loads((generation/'manifest.json').read_text())
    key = B.sha256_json({'generation': manifest['identity_sha256'], 'scorer': scorer})
    out = Path(output) if output else generation/'scores'/f'{benchmark}-{key[:12]}'
    started = {'generation_identity_sha256': manifest['identity_sha256'], 'scorer_identity': scorer}
    if out.exists() and not (out/'complete.json').exists() and restart:
        if (out/'started.json').exists() and json.loads((out/'started.json').read_text()) == started:
            return out, False, True
        if output is None:
            shutil.rmtree(out)
    out, done = B.score_directory(generation, benchmark, scorer, output, False)
    if not done:
        out.mkdir(parents=True)
        B.atomic_json(out/'started.json', started)
    return out, done, False


def vllm_times(log):
    found = {}
    for name, pattern in VLLM_TIMES.items():
        values = [float(v) for v in re.findall(pattern, log)]
        if values:
            found[name] = max(values)
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--generation', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--judge-model', type=Path, required=True, help='fetch_judges.py directory of the pinned judge revision')
    p.add_argument('--tensor-parallel-size', type=int, default=8)
    p.add_argument('--expect', nargs='*', default=[])
    p.add_argument('--restart-incomplete', action='store_true', help='continue an unfinished folder of this exact key (preemption)')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    started = time.monotonic()
    manifest = B.verify_generation(a.generation, B.parse_expectations(a.expect))
    identity = manifest['identity']; benchmark = identity['dataset']['benchmark']
    if benchmark not in JUDGES:
        raise B.ManifestError(f'{benchmark} is not judged by a VLM')
    samples = identity['sampling']['samples_per_prompt']
    if samples != B.SAMPLES[benchmark]:
        raise B.ManifestError(f'{benchmark} is judged on {B.SAMPLES[benchmark]} images per prompt, the generation has {samples}')
    assets, _ = B.assets_manifest()
    if identity['dataset']['files'] != B.dataset_identity(benchmark, a.assets, assets)['files']:
        raise B.ManifestError('Generation used different benchmark prompts than the pinned assets')
    reference = {n: B.verify_asset(a.assets, f'looped-dit/looped_dit/{n}', assets) for n in REFERENCE}
    if benchmark == 'prism':
        reference['prism/evaluation/eval_qwen25.py'] = B.verify_asset(a.assets, 'prism/evaluation/eval_qwen25.py', assets)
    receipt = judge_receipt(a.judge_model, JUDGES[benchmark], load_pin())
    env = environment()
    scorer = {'scorer': benchmark, 'judge': {k: receipt[k] for k in ('repository', 'revision', 'pin_sha256', 'files', 'bytes')},
              'reference_code': reference, 'entry_point': 'looped_dit.eval.spatial_geneval' if benchmark == 'spatial_geneval'
              else 'looped_dit.eval.judge', 'settings': dict(SETTINGS[benchmark], **COMMON),
              'tensor_parallel_size': a.tensor_parallel_size, 'packages': env['packages'], 'environment_sha256': env['sha256'],
              'vllm_plugin': runtime_identity(), 'device_name': sorted({g.split(',')[0].strip() for g in gpus()})}
    out, done, resumed = output_directory(a.generation, benchmark, scorer, a.output, a.restart_incomplete)
    if done:
        print(json.dumps({benchmark: json.loads((out/'summary.json').read_text())['primary'], 'already_complete': True, 'output': str(out)}))
        return
    prompts = B.select(B.load_prompts(benchmark, a.assets), manifest['indices'])
    expected = expected_judgments(benchmark, prompts, samples)
    if identity['dataset']['subset'] is None:
        data = Path(a.assets)/DATA[benchmark]
    else:
        data = out/('SpatialGenEval_T2I_Prompts.jsonl' if benchmark == 'spatial_geneval' else 'data')
        if data.exists():
            shutil.rmtree(data) if data.is_dir() else data.unlink()
        data = subset_data(benchmark, a.assets, prompts, out)
    cached = len(read_results(out/'judge/results.jsonl')) if (out/'judge/results.jsonl').exists() else 0
    command = [sys.executable, '-m', 'looped_dit.eval.spatial_geneval' if benchmark == 'spatial_geneval' else 'looped_dit.eval.judge']
    if benchmark != 'spatial_geneval':
        command += ['--benchmark', benchmark]
    command += ['--data', str(data), '--image-dir', str(a.generation/'images'), '--out-dir', str(out/'judge'),
                '--samples-per-prompt', str(samples), '--judge-model', str(a.judge_model),
                '--tensor-parallel-size', str(a.tensor_parallel_size)]
    if benchmark == 'prism':
        command += ['--prism-script', str(Path(a.assets)/'prism/evaluation/eval_qwen25.py')]
    run_env = dict(os.environ, PYTHONPATH=judge_path(Path(a.assets)/'looped-dit'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
    if 'VLLM_CACHE_ROOT' not in os.environ and os.environ.get('TORCHINDUCTOR_CACHE_DIR'):
        # vLLM compiles each judge for minutes on first use; keep it beside the jobs' persistent kernel caches.
        run_env['VLLM_CACHE_ROOT'] = str(Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).parent/'vllm')
    judge_began = time.monotonic()
    with (out/'judge.log').open('a') as log:
        log.write(f'$ {" ".join(command)}\n'); log.flush()
        if subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT).returncode:
            raise RuntimeError(f'Judge failed; see {out}/judge.log')
    judge_seconds = time.monotonic() - judge_began
    summary = json.loads((out/'judge/summary.json').read_text())
    records = read_results(out/'judge/results.jsonl')
    ours = check_judgments(benchmark, expected, records, summary)
    images = {(p.item_id, s) for p, s, _ in expected.values()}
    judged = {(expected[k][0].item_id, expected[k][1]) for k in records}
    # Upstream drops images whose every judgment stayed unparseable (within its tolerance); report them.
    summary.update(expected_judgments=len(expected), expected_images=len(images), unjudged_images=sorted(map(list, images - judged)),
                   independent_aggregate=ours)
    timing = {'seconds': time.monotonic() - started, 'judge_seconds': judge_seconds, 'judgments_run': len(records) - cached,
              'cached_judgments': cached, 'resumed': resumed, **vllm_times((out/'judge.log').read_text(errors='replace'))}
    B.score_record(out, benchmark, manifest, scorer, summary, summary['overall'],
                   {'results_sha256': B.sha256_file(out/'judge/results.jsonl'), 'timing': timing,
                    'judge_receipt': receipt, 'gpus': gpus(), 'environment': env})
    print(json.dumps({benchmark: summary['overall'], 'output': str(out), 'seconds': round(timing['seconds'], 1)}), flush=True)


if __name__ == '__main__':
    main()
