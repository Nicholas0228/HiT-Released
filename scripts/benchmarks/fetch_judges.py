"""Pin, download and verify the VLM judge weights (Qwen2.5-VL-72B-Instruct, Qwen3-VL-32B-Thinking).

    python scripts/benchmarks/fetch_judges.py pin                      # writes configs/benchmarks/judge-weights.json
    python scripts/benchmarks/fetch_judges.py fetch --root SCORERS/judges [--workers 8]
    python scripts/benchmarks/fetch_judges.py verify --root SCORERS/judges [--workers 16]

Standard library only; public Hugging Face endpoints, no token. `pin` records every file of each pinned
revision (size and SHA-256; the Hub's LFS SHA-256, or its git blob SHA-1 for small files, is checked
against the bytes). `fetch` streams each file from https://huggingface.co/<repo>/resolve/<revision>/<path>
into <root>/<directory>/, hashing while it downloads, and renames it into place (read-only) only when
size and SHA-256 match the pin; finished files are kept on a rerun. `verify` re-reads every file from
disk and writes <directory>/judge-verified.json, which score_judge.py requires (judge_receipt).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from t2i.benchmarks import ManifestError, assets_manifest, atomic_json, sha256_file, sha256_json  # noqa: E402

PIN = Path(__file__).resolve().parents[2]/'configs/benchmarks/judge-weights.json'
RECEIPT = 'judge-verified.json'
HUB = 'https://huggingface.co'
CHUNK = 16 << 20


def utc():
    return datetime.now(timezone.utc).isoformat()


def open_url(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'minit2i-judge-weights'}), timeout=timeout)


def git_blob_sha1(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def directory_name(repository, revision):
    """<name>-<revision12>; the name keeps 'Qwen3' in the basename, which Looped-DiT's judge.py keys on."""
    return f'{repository.split("/", 1)[1]}-{revision[:12]}'


def load_pin(path=None):
    pin = json.loads(Path(path or PIN).read_text())
    if pin.get('schema') != 1:
        raise ManifestError(f'{path or PIN}: unexpected schema')
    return pin


def pin_models(assets=None):
    """The judge repositories and revisions declared in configs/benchmarks/assets.json, with the Hub's file list."""
    assets = assets or assets_manifest()[0]
    models = {}
    for repository, entry in sorted(assets['judge_models'].items()):
        revision = entry['revision']
        with open_url(f'{HUB}/api/models/{repository}/revision/{revision}?blobs=true') as response:
            info = json.load(response)
        if info['sha'] != revision or info.get('private') or info.get('gated'):
            raise ManifestError(f'{repository}@{revision}: the Hub reports {info["sha"]}, private={info.get("private")}, '
                                f'gated={info.get("gated")}')
        files = {}
        for sibling in sorted(info['siblings'], key=lambda s: s['rfilename']):
            name, lfs = sibling['rfilename'], sibling.get('lfs')
            if lfs:
                files[name] = {'size': lfs['size'], 'sha256': lfs['sha256'], 'lfs': True}
                continue
            with open_url(f'{HUB}/{repository}/resolve/{revision}/{name}') as response:
                data = response.read()
            if git_blob_sha1(data) != sibling['blobId'] or len(data) != sibling['size']:
                raise ManifestError(f'{repository}@{revision}:{name} does not match the Hub blob {sibling["blobId"]}')
            files[name] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha1': sibling['blobId']}
        models[repository] = {'revision': revision, 'directory': directory_name(repository, revision),
                              'benchmarks': entry['benchmarks'], 'last_modified': info.get('lastModified'),
                              'bytes': sum(f['size'] for f in files.values()), 'files': files}
    return {'schema': 1, 'source': f'{HUB}/api/models/<repository>/revision/<revision>?blobs=true', 'models': models}


def model_pin_sha256(entry):
    return sha256_json({k: entry[k] for k in ('revision', 'directory', 'files')})


def fetch_file(url, target, want):
    """Download to a partial file while hashing; rename into place only if size and SHA-256 match."""
    if target.exists() and target.stat().st_size == want['size'] and sha256_file(target) == want['sha256']:
        return 'kept', 0
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f'.{target.name}.{os.getpid()}.partial')
    digest, size, begin = hashlib.sha256(), 0, time.monotonic()
    with open_url(url) as response, open(partial, 'wb') as out:
        for part in iter(lambda: response.read(CHUNK), b''):
            digest.update(part); out.write(part); size += len(part)
        out.flush(); os.fsync(out.fileno())
    if size != want['size'] or digest.hexdigest() != want['sha256']:
        partial.unlink()
        raise ManifestError(f'{target.name}: received {size} bytes {digest.hexdigest()}, pinned {want["size"]} {want["sha256"]}')
    os.chmod(partial, 0o444)
    if target.exists():
        os.chmod(target, 0o644); target.unlink()
    os.replace(partial, target)
    return 'downloaded', time.monotonic() - begin


def fetch(root, pin, workers=8, retries=4, only=None):
    jobs = []
    for repository, entry in pin['models'].items():
        if only and repository not in only:
            continue
        for name, want in entry['files'].items():
            jobs.append((f'{HUB}/{repository}/resolve/{entry["revision"]}/{name}', Path(root)/entry['directory']/name, want))
    jobs.sort(key=lambda j: -j[2]['size'])  # largest first keeps all workers busy to the end

    def one(job):
        for attempt in range(retries + 1):
            try:
                return job, *fetch_file(*job)
            except Exception as error:  # network errors and digest mismatches are retried, then reported
                if attempt == retries:
                    raise RuntimeError(f'{job[1]}: {type(error).__name__}: {error}') from error
                time.sleep(10*(attempt + 1))

    began, total = time.monotonic(), 0
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, j) for j in jobs]):
            (url, target, want), status, seconds = future.result()
            total += want['size'] if status == 'downloaded' else 0
            print(json.dumps({'file': str(target), 'status': status, 'bytes': want['size'], 'seconds': round(seconds, 1)}), flush=True)
    elapsed = time.monotonic() - began
    print(json.dumps({'fetched_bytes': total, 'seconds': round(elapsed, 1), 'MBps': round(total/max(elapsed, 1e-9)/1e6, 1)}), flush=True)


def verify(root, pin, workers=16, only=None):
    """Re-read every file, require exactly the pinned set, write the receipt score_judge.py checks."""
    receipts = {}
    for repository, entry in pin['models'].items():
        if only and repository not in only:
            continue
        directory, began = Path(root)/entry['directory'], time.monotonic()
        present = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} - {RECEIPT}
        present = {p for p in present if not p.startswith('.cache/')}
        missing, stray = sorted(set(entry['files']) - present), sorted(present - set(entry['files']))
        if missing or stray:
            raise ManifestError(f'{directory}: missing {missing[:5]}, unexpected {stray[:5]}')
        with ThreadPoolExecutor(workers) as pool:
            digests = dict(zip(entry['files'], pool.map(lambda n: sha256_file(directory/n), entry['files'])))
        wrong = sorted(n for n, d in digests.items() if d != entry['files'][n]['sha256']
                       or (directory/n).stat().st_size != entry['files'][n]['size'])
        if wrong:
            raise ManifestError(f'{directory}: content differs from the pin for {wrong[:5]}')
        receipt = {'schema': 1, 'repository': repository, 'revision': entry['revision'], 'directory': str(directory),
                   'pin_sha256': model_pin_sha256(entry), 'bytes': entry['bytes'],
                   'files': {n: {'size': entry['files'][n]['size'], 'sha256': digests[n]} for n in sorted(digests)},
                   'verified_utc': utc(), 'verified_time': time.time(), 'host': socket.gethostname(),
                   'seconds': round(time.monotonic() - began, 1)}
        target = directory/RECEIPT
        if target.exists():
            os.chmod(target, 0o644)
        atomic_json(target, receipt)
        os.chmod(target, 0o444)
        receipts[repository] = {'directory': str(directory), 'files': len(digests), 'bytes': entry['bytes'],
                                'seconds': receipt['seconds'], 'pin_sha256': receipt['pin_sha256']}
        print(json.dumps({repository: receipts[repository]}), flush=True)
    return receipts


def judge_receipt(model_dir, repository, pin=None):
    """What score_judge.py records: the pinned revision and file hashes, checked against the verify receipt.

    Cheap at scoring time: the directory must hold exactly the pinned files with the pinned sizes, none
    modified after the full re-read recorded in judge-verified.json.
    """
    pin = pin or load_pin()
    if repository not in pin['models']:
        raise ManifestError(f'{repository} is not a pinned judge')
    entry, model_dir = pin['models'][repository], Path(model_dir)
    path = model_dir/RECEIPT
    if not path.exists():
        raise ManifestError(f'{model_dir} has no {RECEIPT}; run fetch_judges.py fetch and verify')
    receipt = json.loads(path.read_text())
    expected = {n: {'size': f['size'], 'sha256': f['sha256']} for n, f in entry['files'].items()}
    if receipt['repository'] != repository or receipt['revision'] != entry['revision'] or receipt['files'] != expected \
            or receipt['pin_sha256'] != model_pin_sha256(entry):
        raise ManifestError(f'{path} does not certify {repository}@{entry["revision"]} as pinned')
    present = {str(p.relative_to(model_dir)) for p in model_dir.rglob('*') if p.is_file() and not
               str(p.relative_to(model_dir)).startswith('.cache/')} - {RECEIPT}
    if present != set(expected):
        raise ManifestError(f'{model_dir}: files differ from the pin: missing {sorted(set(expected) - present)[:3]}, '
                            f'unexpected {sorted(present - set(expected))[:3]}')
    for name, want in expected.items():
        stat = (model_dir/name).stat()
        if stat.st_size != want['size'] or stat.st_mtime > receipt['verified_time']:
            raise ManifestError(f'{model_dir/name} changed after verification; rerun fetch_judges.py verify')
    return {'repository': repository, 'revision': entry['revision'], 'pin_sha256': receipt['pin_sha256'],
            'files': len(expected), 'bytes': entry['bytes'], 'verified_utc': receipt['verified_utc']}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    pin = sub.add_parser('pin'); pin.add_argument('--output', type=Path, default=PIN)
    for name in ('fetch', 'verify'):
        s = sub.add_parser(name)
        s.add_argument('--root', type=Path, required=True)
        s.add_argument('--workers', type=int, default=8 if name == 'fetch' else 16)
        s.add_argument('--only', nargs='+', help='repository IDs (default: all pinned judges)')
        s.add_argument('--pin', type=Path, default=PIN)
    a = p.parse_args()
    if a.action == 'pin':
        result = pin_models()
        atomic_json(a.output, result)
        print(json.dumps({r: {'revision': m['revision'], 'files': len(m['files']), 'bytes': m['bytes']}
                          for r, m in result['models'].items()}, indent=1))
    elif a.action == 'fetch':
        fetch(a.root, load_pin(a.pin), a.workers, only=a.only)
        verify(a.root, load_pin(a.pin), 2*a.workers, only=a.only)
    else:
        verify(a.root, load_pin(a.pin), a.workers, only=a.only)


if __name__ == '__main__':
    main()
