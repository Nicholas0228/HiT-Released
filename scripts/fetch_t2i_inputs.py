"""Download and verify the text-to-image inputs, then write <output>/inputs.json.

Downloads the frozen inputs (Nicholas0228/minit2i-frozen-inputs-v1: index, captions, sequence, COCO
assets), the CC12M shards and FLAN-T5-Large. Reserve about 1.2 TB. A Hugging Face token is optional; it
raises the Hub's rate limits for the 2,176 shard files.

  python scripts/fetch_t2i_inputs.py --output /data/t2i-inputs
  python scripts/fetch_t2i_inputs.py --output /data/t2i-inputs --updates 64   # smoke-test subset

The token is read from --token-file or HF_TOKEN and is never printed or stored.
Interrupted downloads resume. Re-running only verifies what is already present.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2i.inputs import fetch, write_manifest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--token-file', type=Path, help='optional Hugging Face token (default: $HF_TOKEN)')
    p.add_argument('--updates', type=int, help='Only fetch CC12M shards used by the first N updates (smoke tests)')
    p.add_argument('--no-evaluation', action='store_true', help='Skip the 16 GB COCO evaluation assets')
    p.add_argument('--verify', choices=['size', 'sha256', 'full'], default='sha256',
                   help='sha256 hashes every non-shard file; full also hashes all present shards')
    p.add_argument('--manifest-only', action='store_true', help='Only re-verify present files and rewrite inputs.json')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.manifest_only:
        manifest = write_manifest(args.output, verify=args.verify, workers=args.workers)
    else:
        token = args.token_file.read_text().strip() if args.token_file else os.environ.get('HF_TOKEN')
        manifest = fetch(args.output, token or None, updates=args.updates, evaluation=not args.no_evaluation,
                         verify=args.verify, workers=args.workers)
    shards = len(manifest['shards']) - manifest['missing_shards']
    print(f"inputs.json written: {shards}/{len(manifest['shards'])} CC12M shards, "
          f"evaluation {'present' if manifest['evaluation'] else 'absent'}, verification {args.verify}")


if __name__ == '__main__':
    main()
