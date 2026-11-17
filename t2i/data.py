"""Accepted CC12M pairs read by a fixed global sample sequence.

Global position p -> record sequence[p] -> JPEG bytes at (shard, offset, size) in
the pinned public pixparse/cc12m-wds tar shards, plus its frozen LLaVA-NeXT caption.
Images: PIL RGB, bicubic shortest-edge resize and center crop to 512, no flips.
"""
from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


class UpdateBatches(Sampler):
    """Rank r of W at update u reads global positions [u*G + r*G/W, ...) for G = global batch."""
    def __init__(self, start_update, stop_update, global_batch, rank, world):
        if global_batch % world or not 0 <= rank < world:
            raise ValueError('The global batch must divide evenly across ranks')
        self.start, self.stop, self.batch, self.rank, self.world = start_update, stop_update, global_batch, rank, world

    def __iter__(self):
        local = self.batch // self.world
        for update in range(self.start, self.stop):
            start = update * self.batch + self.rank * local
            yield list(range(start, start + local))

    def __len__(self):
        return self.stop - self.start


class CC12MSequence(Dataset):
    """Holds only paths; each process opens its own memory maps, SQLite and shard files.

    This keeps the dataset small when pickled to spawned workers (fork after CUDA,
    TorchScript or compilation is not safe).
    """
    def __init__(self, inputs, image_size=512, text_length=256, sequence=None):
        self.inputs = dict(inputs) if isinstance(inputs, dict) else json.loads(Path(inputs).read_text())
        self.sequence_path = str(sequence or self.inputs['sequence'])
        self.length = len(np.load(self.sequence_path, mmap_mode='r', allow_pickle=False))
        self.text_length = text_length
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size)])
        self.tokenizer = None
        self._open = None

    def __getstate__(self):
        return dict(self.__dict__, tokenizer=None, _open=None)

    def __len__(self):
        return self.length

    def opened(self):
        if self._open is None or self._open['pid'] != os.getpid():
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            self._open = {'pid': os.getpid(), 'files': OrderedDict(),
                          'records': np.load(self.inputs['records'], mmap_mode='r', allow_pickle=False),
                          'sequence': np.load(self.sequence_path, mmap_mode='r', allow_pickle=False),
                          'db': sqlite3.connect(f"file:{self.inputs['captions']}?mode=ro&immutable=1", uri=True)}
        return self._open

    def __getitem__(self, position):
        state = self.opened()
        record = state['records'][int(state['sequence'][position])]
        key, shard, offset, size = (int(record[name]) for name in ('key', 'shard', 'offset', 'size'))
        files = state['files']
        handle = files.pop(shard) if shard in files else os.open(self.inputs['shards'][shard], os.O_RDONLY)
        files[shard] = handle
        if len(files) > 32:
            os.close(files.popitem(last=False)[1])
        encoded = os.pread(handle, size, offset)
        if len(encoded) != size:
            raise OSError(f'Truncated image {key} in shard {shard}')
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                image = self.transform(image.convert('RGB'))
                pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        except (OSError, ValueError) as error:
            raise OSError(f'Image decode failed at position {position}, key {key}, shard {shard}: {error}') from error
        row = state['db'].execute('SELECT caption FROM captions WHERE key=?', (key,)).fetchone()
        if row is None:
            raise KeyError(f'Missing frozen caption {key}')
        return pixels, row[0], key, position

    def collate(self, rows):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.inputs['encoder'], local_files_only=True)
        images, captions, keys, positions = zip(*rows)
        tokens = self.tokenizer(list(captions), padding='max_length', truncation=True,
                                max_length=self.text_length, return_tensors='pt')
        return {'image': torch.stack(images), 'input_ids': tokens.input_ids, 'attention_mask': tokens.attention_mask,
                'keys': torch.tensor(keys, dtype=torch.int64), 'positions': torch.tensor(positions, dtype=torch.int64)}


def to_model_range(images):
    """uint8 [0, 255] -> [-1, 1], with the recipe's exact arithmetic."""
    return images.float().div_(255).sub_(.5).div_(.5)


@torch.no_grad()
def encode_captions(encoder, input_ids, mask, chunk):
    """Frozen FLAN-T5-Large features in BF16, encoded in microbatch-sized chunks."""
    pieces = []
    for start in range(0, len(input_ids), chunk):
        with torch.autocast(input_ids.device.type, dtype=torch.bfloat16):
            pieces.append(encoder(input_ids=input_ids[start:start + chunk], attention_mask=mask[start:start + chunk])
                          .last_hidden_state.to(torch.bfloat16))
    return torch.cat(pieces, 0)
