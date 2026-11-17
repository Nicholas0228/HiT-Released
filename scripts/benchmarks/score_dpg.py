"""DPG-Bench adapter: verified 2x2 grids -> pinned mPLUG-large VQA -> released dependent-question aggregation.

    $DPG_ENV/bin/python scripts/benchmarks/score_dpg.py --generation GEN_DIR --assets ASSET_ROOT --devices 0 1 2 3 4 5 6 7
    $DPG_ENV/bin/python scripts/benchmarks/score_dpg.py --generation GEN_DIR --assets ASSET_ROOT --devices cpu --workers 4

Every question of every 512x512 sample is answered once. The ELLA compute_dpg_bench.py rule
is applied in its question order: a question whose non-zero parent was answered other than
"yes" is zeroed; a sample scores the mean over questions and a prompt the mean over its four
samples. Categories use the raw answers of the last sample, as in that script.

Primary score: all 1,065 prompts with every CSV question (the Looped-DiT/MiniT2I convention).
The unmodified ELLA script skips the CSV's first data row, so partiprompts97 then references
a missing parent and is excluded; that number is reported as official_ella.
"""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from t2i import benchmarks as B  # noqa: E402

PACKAGES = ('torch', 'torchvision', 'modelscope', 'transformers', 'timm', 'numpy', 'pillow')
MPLUG_FILES = ('config.yaml', 'config_bert.json', 'configuration.json', 'pytorch_model.bin', 'vocab.txt')


def scorer_identity(assets_root, assets, device, primary='all'):
    import torch
    return {'scorer': 'dpg', 'vqa': dict(assets['mplug'], files={n: B.verify_asset(assets_root, f'mplug/{n}', assets) for n in MPLUG_FILES}),
            'questions': {'dpg/dpg_bench.csv': B.verify_asset(assets_root, 'dpg/dpg_bench.csv', assets)},
            'reference_script_sha256': B.verify_asset(assets_root, 'dpg/official/compute_dpg_bench.py', assets),
            'aggregation': 'ELLA dependent-question rule; mean over questions, then over the four samples; categories from the last sample',
            'primary': {'all': 'all prompts, all CSV questions', 'official-ella': 'ELLA script: first CSV row skipped'}[primary],
            'resolution': 512, 'pic_num': 4,
            'packages': {p: importlib.metadata.version(p) for p in PACKAGES}, 'device': device,
            'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}


def answer(work, questions, model_dir, device, threads, output):
    """One process per device; writes one JSON line per (prompt, sample, question)."""
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    import torch
    torch.set_num_threads(threads)
    from PIL import Image
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    vqa = pipeline(Tasks.visual_question_answering, model=str(model_dir), device='cpu' if device == 'cpu' else 'gpu')
    with open(output, 'w') as out:
        for item, path in work:
            grid = Image.open(path).convert('RGB')
            for sample in range(4):
                crop = grid.crop(tuple(B.grid_box(sample)))
                for qid, question, _, _ in questions[item]:
                    text = vqa({'image': crop, 'question': question})['text']
                    out.write(json.dumps({'item_id': item, 'sample': sample, 'qid': qid, 'question': question, 'answer': text}) + '\n')
            out.flush()


def prompt_score(questions, answers):
    """ELLA compute_dpg_one_sample on recorded answers; KeyError for a missing parent, as there."""
    crops, raw = [], {}
    for sample in range(4):
        scores = {qid: float(answers[sample][qid] == 'yes') for qid, *_ in questions}
        raw = dict(scores)
        for qid, _, parents, _ in questions:
            if any(p != 0 and scores[p] == 0 for p in parents):
                scores[qid] = 0
        crops.append(sum(scores.values())/len(scores))
    return crops, sum(crops)/len(crops), raw


def aggregate(items, questions, answers):
    scored, failed, categories = {}, [], defaultdict(list)
    for item in items:
        try:
            crops, score, raw = prompt_score(questions[item], answers[item])
        except KeyError:
            failed.append(item); continue
        scored[item] = {'crop_scores': crops, 'score': score}
        for qid, _, _, category in questions[item]:
            categories[category].append(raw[qid])
    l1 = defaultdict(list)
    for category, values in categories.items():
        l1[category.split('-')[0].strip()] += values
    mean = lambda v: 100*sum(v)/len(v)
    return {'overall': mean([s['score'] for s in scored.values()]) if scored else float('nan'), 'num_prompts': len(scored),
            'excluded': failed, 'l1_categories': {k: mean(v) for k, v in sorted(l1.items())},
            'l2_categories': {k: mean(v) for k, v in sorted(categories.items())}}, scored


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--generation', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--devices', nargs='+', default=['0'], help="GPU indices, or 'cpu'")
    p.add_argument('--workers', type=int, default=1, help='CPU processes when --devices cpu')
    p.add_argument('--threads', type=int, help='CPU threads per worker (default: available cores / workers)')
    p.add_argument('--expect', nargs='*', default=[], help='identity checks, e.g. model.checkpoint_sha256=HEX')
    p.add_argument('--primary', choices=('all', 'official-ella'), default='all', help='which DPG number is the primary score')
    p.add_argument('--restart-incomplete', action='store_true', help='discard an unfinished folder of this exact key (preemption)')
    p.add_argument('--output', type=Path, help='default: GEN_DIR/scores/dpg-<key>')
    a = p.parse_args()
    started = time.monotonic()
    manifest = B.verify_generation(a.generation, B.parse_expectations(a.expect))
    identity = manifest['identity']
    if identity['dataset']['benchmark'] != 'dpg':
        raise B.ManifestError('Not a DPG-Bench generation')
    assets, _ = B.assets_manifest()
    if identity['dataset']['files'] != B.dataset_identity('dpg', a.assets, assets)['files']:
        raise B.ManifestError('Generation used different DPG-Bench prompts than the pinned assets')
    questions = B.dpg_questions(a.assets/'dpg/dpg_bench.csv')
    prompts = B.load_prompts('dpg', a.assets)
    items = [prompts[i].item_id for i in manifest['indices']]
    from PIL import Image
    for item in items:
        with Image.open(a.generation/'images'/f'{item}.png') as im:
            if im.size != (1024, 1024):
                raise B.ManifestError(f'{item}.png is {im.size}, expected a 2x2 grid of 512x512 samples')
    cpu = a.devices == ['cpu']
    scorer = scorer_identity(a.assets, assets, 'cpu' if cpu else 'cuda', a.primary)
    out, done = B.score_directory(a.generation, 'dpg', scorer, a.output, a.restart_incomplete)
    if done:
        print(json.dumps({'dpg': json.loads((out/'summary.json').read_text())['primary'], 'already_complete': True, 'output': str(out)})); return
    (out/'answers').mkdir(parents=True)
    devices = ['cpu']*a.workers if cpu else [B.physical_device(d) for d in a.devices]
    workers = min(len(devices), len(items))
    threads = (a.threads or max(1, len(os.sched_getaffinity(0))//workers)) if cpu else 4
    jobs = [([(item, str(a.generation/'images'/f'{item}.png')) for item in items[k::workers]], questions,
             a.assets/'mplug', devices[k], threads, out/'answers'/f'{k:02d}.jsonl') for k in range(workers)]
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        pool.starmap(answer, jobs)
    answers = defaultdict(lambda: defaultdict(dict)); calls = 0
    for k in range(workers):
        for row in B.read_jsonl(out/'answers'/f'{k:02d}.jsonl'):
            slot = answers[row['item_id']][row['sample']]
            if row['qid'] in slot:
                raise B.ManifestError(f'Duplicated answer {row}')
            slot[row['qid']] = row['answer']; calls += 1
    for item in items:
        for sample in range(4):
            if set(answers[item][sample]) != {q[0] for q in questions[item]}:
                raise B.ManifestError(f'Incomplete answers for {item} sample {sample}')
    primary, scored = aggregate(items, questions, answers)
    if primary['excluded']:
        raise B.ManifestError(f'Full question sets cannot fail: {primary["excluded"]}')
    first = next(iter(questions))
    ella_questions = dict(questions, **{first: questions[first][1:]})
    official, _ = aggregate(items, ella_questions, answers)
    metrics = dict(primary, official_ella=official, vqa_calls=calls,
                   yes_fraction=sum(v == 'yes' for i in answers.values() for s in i.values() for v in s.values())/calls)
    index = {q.item_id: q.index for q in prompts}
    with (out/'items.jsonl').open('w') as f:
        for item in items:
            f.write(json.dumps({'index': index[item], 'item_id': item, **scored[item]}) + '\n')
    summary = B.score_record(out, 'dpg', manifest, scorer, metrics, official['overall'] if a.primary == 'official-ella' else metrics['overall'],
                             {'items_sha256': B.sha256_file(out/'items.jsonl'), 'workers': workers,
                              'seconds': time.monotonic() - started})
    print(json.dumps({'dpg': summary['primary'], 'all_prompts': metrics['overall'], 'official_ella': official['overall'],
                      'prompts': len(items), 'vqa_calls': calls,
                      'output': str(out), 'seconds': round(summary['seconds'], 1)}), flush=True)


if __name__ == '__main__':
    main()
