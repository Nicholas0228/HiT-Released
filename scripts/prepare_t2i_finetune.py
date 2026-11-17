"""Download, prepare and verify the three-source SFT40k mixture.

Images stay under their original dataset licenses. This command downloads from
the original repositories and writes a local training manifest.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet.checkpoints import atomic_json, sha256_file
from t2i import finetune_data


def write_inputs(root, encoder, contract):
    root, encoder = Path(root).resolve(), Path(encoder).resolve()
    if not encoder.is_dir():
        raise FileNotFoundError('Pass the local FLAN-T5-Large snapshot with --encoder-root')
    manifest = json.loads((root / 'manifest.json').read_text())
    receipt = json.loads((root / 'verified.json').read_text())
    if not receipt['passed'] or receipt['data']['manifest_sha256'] != sha256_file(root / 'manifest.json'):
        raise ValueError('Missing or stale fine-tuning verification')
    paths = dict(records=manifest['records'], captions=manifest['caption_db'], sequence=str(root / 'sequence.npy'))
    identity = {key + '_sha256': sha256_file(path) for key, path in paths.items()}
    identity.update(contract_sha256=sha256_file(contract), accepted_pairs=manifest['count'],
                    mixture=manifest['mixture'])
    inputs = dict(schema=1, phase='sft40k', complete=True, missing_shards=0, evaluation=None,
                  root=str(root), encoder=str(encoder), **paths,
                  shards=[row['path'] for row in manifest['shards']], fingerprint=identity)
    atomic_json(root / 'inputs.json', inputs)
    return root / 'inputs.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoder-root', type=Path, required=True, help='Existing FLAN-T5-Large snapshot')
    parser.add_argument('--contract', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/t2i/sft_data.json')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--download-workers', type=int, default=6)
    args = parser.parse_args()
    if not args.encoder_root.is_dir():
        parser.error('--encoder-root must point to the existing FLAN-T5-Large snapshot')
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / 'receipt.json').exists():
        finetune_data.prepare(args.contract, args.output, args.workers, args.download_workers)
    # Recheck even after an interrupted/repeated invocation.
    verification = args.output / 'verified.json'
    if verification.exists():
        verification = args.output / 'reverified.json'
        verification.unlink(missing_ok=True)
    finetune_data.verify(args.contract, args.output, verification)
    if verification.name != 'verified.json':
        atomic_json(args.output / 'verified.json', json.loads(verification.read_text()))
    print(write_inputs(args.output, args.encoder_root, args.contract))


if __name__ == '__main__':
    main()
