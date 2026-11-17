"""Run in the separate TensorFlow environment; no PyTorch model dependency."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


class Images:
    def __init__(self, directory, rows, size):
        self.directory, self.rows, self.size = Path(directory), rows, size
        self.seen = 0

    def __iter__(self):
        self.seen = 0
        for start in range(0, len(self.rows), 64):
            batch = []
            for row in self.rows[start:start + 64]:
                path = self.directory / row['filename']
                if sha(path) != row['sha256']: raise RuntimeError('Image checksum failed')
                with Image.open(path) as opened:
                    if opened.mode != 'RGB' or opened.size != (self.size, self.size):
                        raise RuntimeError('Invalid RGB image dimensions')
                    batch.append(np.array(opened))
                self.seen += 1
            yield np.stack(batch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-adapter', action='store_true',
                   help='Compare streamed PNG features/statistics with the native NPZ reader')
    a = p.parse_args()
    if a.output.exists(): raise RuntimeError('Preserve existing metric results')
    manifest, assets = json.loads(a.manifest.read_text()), json.loads(a.assets.read_text())
    count, rows = manifest['identity']['count'], manifest['images']
    if [r['id'] for r in rows] != list(range(count)): raise RuntimeError('Incomplete evaluation cohort')
    for key in ('evaluator', 'inception', 'reference'):
        if sha(assets[key]['path']) != assets[key]['sha256']: raise RuntimeError('Changed ADM asset: ' + key)
    import scipy
    import tensorflow as tensorflow
    import tensorflow.compat.v1 as tf
    if (tensorflow.__version__, np.__version__, scipy.__version__) != ('2.15.1', '1.26.4', '1.14.1'):
        raise RuntimeError('Use the pinned ADM environment: TF2.15.1, NumPy1.26.4, SciPy1.14.1')
    tf.disable_eager_execution()
    tensorflow.config.experimental.enable_tensor_float_32_execution(False)
    spec = importlib.util.spec_from_file_location('official_adm', assets['evaluator']['path'])
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    official.INCEPTION_V3_PATH = assets['inception']['path']
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    started = time.monotonic()
    with tf.Session(config=config) as session:
        evaluator = official.Evaluator(session, batch_size=64); evaluator.warmup()
        reference, _ = evaluator.read_statistics(assets['reference']['path'], (None, None))
        images = Images(manifest['directory'], rows, manifest['image_size'])
        features, spatial = evaluator.compute_activations(images)
        if images.seen != count or features.shape != (count, 2048) or not np.isfinite(features).all():
            raise RuntimeError('Incomplete or invalid Inception features')
        stats = evaluator.compute_statistics(features)
        parity = None
        if a.check_adapter:
            npz = a.output.with_suffix('.pixels.npz')
            if npz.exists(): raise RuntimeError('Preserve previous adapter-check pixels')
            np.savez(npz, np.concatenate(list(Images(manifest['directory'], rows, manifest['image_size']))))
            native = evaluator.read_activations(str(npz))
            if not np.array_equal(features, native[0]) or not np.array_equal(spatial, native[1]):
                raise RuntimeError('Streamed PNG / official NPZ features differ')
            native_stats = evaluator.compute_statistics(native[0])
            if not np.array_equal(stats.mu, native_stats.mu) or not np.array_equal(stats.sigma, native_stats.sigma):
                raise RuntimeError('Streamed PNG / official NPZ statistics differ')
            parity = dict(passed=True, images=count, features_bitwise_equal=True, statistics_bitwise_equal=True)
        fid, inception = float(stats.frechet_distance(reference)), float(evaluator.compute_inception_score(features))
        if not np.isfinite(fid) or fid < -1e-4 or not np.isfinite(inception): raise RuntimeError('Invalid ADM score')
    report = dict(fid=fid, inception_score=inception, count=count, guidance=manifest['guidance'],
                  protocol='official-openai-adm', manifest_sha256=sha(a.manifest),
                  adapter_parity=parity,
                  assets={k:v['sha256'] for k,v in assets.items()}, seconds=time.monotonic()-started)
    with a.output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
