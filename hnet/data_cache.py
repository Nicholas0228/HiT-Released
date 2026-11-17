"""ImageNet-256 center-crop cache (ImageFolder order) with the stateless per-(epoch, index) flip.

pixels are bitwise the pinned-JiT center_crop_arr output (see scripts/build_ccrop_cache.py and the
validation record); the flip policy is hnet_jit.data.flip_bit, so samples are independent of loader
workers/prefetch and resumable at optimizer-update boundaries with ResumeShardSampler."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .data import flip_bit

SIDE = 256


class CachedImageNet(Dataset):
    def __init__(self, root, seed=0, epoch=0, flip=True):
        self.root, self.seed, self.epoch, self.flip = Path(root), seed, epoch, flip
        manifest = json.loads((self.root / 'manifest.json').read_text())
        if manifest.get('status') != 'complete' or manifest['shape'][1:] != [SIDE, SIDE, 3]:
            raise RuntimeError('Cache is incomplete')
        self.count = int(manifest['images'])
        labels_path = self.root / 'labels.npy'
        if hashlib.sha256(labels_path.read_bytes()).hexdigest() != manifest['labels_sha256']:
            raise RuntimeError('Cache label hash mismatch')
        self.labels = np.load(labels_path)
        if len(self.labels) != self.count or (self.root / 'train_256.u8').stat().st_size != self.count * SIDE * SIDE * 3:
            raise RuntimeError('Cache size mismatch')
        self.fingerprint = hashlib.sha256(json.dumps({k: manifest[k] for k in
                                          ('images', 'class_sha256', 'names_sha256', 'labels_sha256', 'transform')},
                                          sort_keys=True).encode()).hexdigest()
        self._pixels = None

    def pixels(self):
        if self._pixels is None:      # opened lazily inside each loader worker
            self._pixels = np.memmap(self.root / 'train_256.u8', dtype=np.uint8, mode='r',
                                     shape=(self.count, SIDE, SIDE, 3))
        return self._pixels

    def __len__(self):
        return self.count

    def __getitem__(self, position):
        position = int(position)
        array = self.pixels()[position]
        if self.flip and flip_bit(self.seed, self.epoch, position):
            array = array[:, ::-1]
        return torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1), int(self.labels[position]), position

    def __getstate__(self):
        state = dict(self.__dict__); state['_pixels'] = None
        return state
