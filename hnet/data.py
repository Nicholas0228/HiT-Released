"""Verified image index and reproducible, optimizer-boundary data recovery."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from util.crop import center_crop_arr


def flip_bit(seed, epoch, index):
    """A stateless Bernoulli(1/2) choice, independent of loader prefetch/workers."""
    mask = (1 << 64) - 1
    z = (index + 0x9E3779B97F4A7C15 * (epoch + 1) + seed) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & 1


class IndexedImageNet(Dataset):
    def __init__(self, root, split='train', image_size=256, seed=0, epoch=0):
        self.root, self.split = Path(root), split
        self.image_size, self.seed, self.epoch = image_size, seed, epoch
        receipt = json.loads((self.root / 'verified.json').read_text())
        if receipt['status'] != 'verified' or not receipt['all_images_read_back_and_decoded']:
            raise RuntimeError('Dataset is not fully verified')
        index_path = self.root / f'{split}.npy'
        digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
        if digest != receipt['splits'][split]['index_sha256']:
            raise RuntimeError('Dataset index hash mismatch')
        self.index = np.load(index_path, mmap_mode='r')
        if len(self.index) != receipt['splits'][split]['images']:
            raise RuntimeError('Dataset count mismatch')
        self.fingerprint = receipt['source_manifest_sha256']

    def __len__(self):
        return len(self.index)

    def __getitem__(self, position):
        record = self.index[int(position)]
        label, shard, row = (int(record[name]) for name in ('label', 'shard', 'row'))
        path = self.root / self.split / f'{label:04d}' / f'{shard:05d}-{row:07d}.image'
        with Image.open(path) as opened:
            image = center_crop_arr(opened.convert('RGB'), self.image_size)
        if self.split == 'train' and flip_bit(self.seed, self.epoch, int(position)):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1), label, int(position)


class ResumeShardSampler(Sampler):
    def __init__(self, size, rank, world_size, microbatch, accumulation, seed, epoch, start_update=0):
        self.size, self.rank, self.world_size = size, rank, world_size
        self.local_update_size = microbatch * accumulation
        self.updates_per_epoch = size // (self.local_update_size * world_size)
        if not 0 <= start_update <= self.updates_per_epoch:
            raise ValueError('Resume cursor outside epoch')
        self.start_update, self.seed, self.epoch = start_update, seed, epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.updates_per_epoch * self.local_update_size * self.world_size
        order = torch.randperm(self.size, generator=generator)[:total]
        local = order[self.rank::self.world_size]
        local = local[self.start_update * self.local_update_size:]
        return iter(local.tolist())

    def __len__(self):
        return (self.updates_per_epoch - self.start_update) * self.local_update_size


class FolderImageNet(Dataset):
    """Standard train/<class> files, ImageFolder order, resumable flips.

    The verified cache is the historical replay path. This adapter records the
    supplied file list; it does not certify corpus bytes.
    """
    def __init__(self, root, seed=0, epoch=0, image_size=256):
        from torchvision.datasets import ImageFolder
        self.folder = ImageFolder(root)
        self.seed, self.epoch, self.image_size = seed, epoch, image_size
        self.root = Path(root)
        records = [(str(Path(path).relative_to(self.root)), label) for path, label in self.folder.samples]
        self.fingerprint = hashlib.sha256(json.dumps(records).encode()).hexdigest()

    def __len__(self): return len(self.folder.samples)

    def __getitem__(self, position):
        path, label = self.folder.samples[int(position)]
        with Image.open(path) as opened:
            image = center_crop_arr(opened.convert('RGB'), self.image_size)
        if flip_bit(self.seed, self.epoch, int(position)):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1), label, int(position)
