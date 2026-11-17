"""Download the pinned benchmark prompts, scorer code, weights and frozen images; verify SHA-256.

    python scripts/benchmarks/fetch_assets.py --root ASSET_ROOT [--groups geneval dpg t5 frozen reference judge-data]

Idempotent: verified files are kept, anything else is downloaded to a partial file and only
renamed after its digest matches configs/benchmarks/assets.json. Judge model weights are
recorded by revision only; fetch them on the GPU node (they are 67-147 GB).
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from t2i.benchmarks import assets_manifest, atomic_json, sha256_file, tree_digest  # noqa: E402


def url_for(entry):
    if 'url' in entry:
        return entry['url']
    repo, commit, path = re.fullmatch(r'([^@]+)@([0-9a-f]{40}):(.+)', entry['source']).groups()
    return f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'


def download(url, target):
    request = urllib.request.Request(url, headers={'User-Agent': 'minit2i-benchmark-assets'})
    with urllib.request.urlopen(request, timeout=300) as response, open(target, 'wb') as out:
        shutil.copyfileobj(response, out, 16 << 20)


def fetch_file(root, name, entry):
    target = root/name
    if target.exists() and sha256_file(target) == entry['sha256']:
        return 'kept'
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    download(url_for(entry), partial)
    actual = sha256_file(partial)
    if actual != entry['sha256']:
        partial.unlink(); raise RuntimeError(f'{name}: downloaded {actual}, pinned {entry["sha256"]}')
    os.replace(partial, target)
    return 'downloaded'


def fetch_tree(root, name, entry):
    target = root/name
    if target.exists() and tree_digest(target) == entry['sha256']:
        return 'kept'
    with tempfile.TemporaryDirectory(dir=root) as work:
        archive = Path(work)/'archive.tar.gz'; download(entry['archive'], archive)
        staging = Path(work)/'tree'; prefix = entry['member_prefix']
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.startswith(prefix):
                    relative = Path(member.name[len(prefix):])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise RuntimeError(f'Unsafe archive member {member.name}')
                    (staging/relative).parent.mkdir(parents=True, exist_ok=True)
                    (staging/relative).write_bytes(tar.extractfile(member).read())
        actual = tree_digest(staging)
        if actual != entry['sha256']:
            raise RuntimeError(f'{name}: extracted tree {actual}, pinned {entry["sha256"]}')
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True); os.replace(staging, target)
    return 'downloaded'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--groups', nargs='+', default=['geneval', 'dpg', 't5', 'frozen', 'reference', 'judge-data'])
    p.add_argument('--manifest', type=Path)
    a = p.parse_args()
    assets, manifest_sha = assets_manifest(a.manifest)
    a.root.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, entry in assets['files'].items():
        if entry['group'] not in a.groups:
            continue
        status = (fetch_tree if entry.get('kind') == 'tree' else fetch_file)(a.root, name, entry)
        verified[name] = entry['sha256']
        print(json.dumps({'asset': name, 'status': status}), flush=True)
    previous = json.loads((a.root/'verified.json').read_text()) if (a.root/'verified.json').exists() else {}
    if previous.get('assets_manifest_sha256') == manifest_sha:
        verified = {**previous['files'], **verified}
    atomic_json(a.root/'verified.json', {'assets_manifest_sha256': manifest_sha, 'files': dict(sorted(verified.items()))})
    print(json.dumps({'verified': len(verified), 'root': str(a.root)}), flush=True)


if __name__ == '__main__':
    main()
