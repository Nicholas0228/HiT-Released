"""Small synthetic stand-ins for the three pinned sources and a tiny tokenizer/T5."""
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from t2i.finetune_data import BLIP3_FT60K_TARS, DTYPE, add_member
from hnet.checkpoints import sha256_file as sha256

WORDS = ('a red cat sits on mat blue dog runs in park green tree under sky small house near river big car '
         'by road with two birds flying over field of yellow flowers at sunset tall man holds old book').split()


def image_bytes(seed, size=(20, 14), mode='RGB', fmt='PNG', truncate=False):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 4 if mode == 'RGBA' else 3), dtype=np.uint8))
    if mode == 'P':
        image = image.convert('P', palette=Image.ADAPTIVE, colors=16)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    data = buffer.getvalue()
    return data[:len(data) // 2] if truncate else data


def caption(seed, words=6):
    rng = np.random.default_rng(seed)
    return ' '.join(WORDS[i] for i in rng.integers(0, len(WORDS), words))


def write_tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w') as tar:
        for name, data in members:
            add_member(tar, name, data)


def tokenizer_directory(directory, encoder=False):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    vocab = {'<pad>': 0, '</s>': 1, '<unk>': 2, **{w: i + 3 for i, w in enumerate(sorted(set(WORDS)))}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single='$A </s>', special_tokens=[('</s>', 1)])
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='<pad>', eos_token='</s>', unk_token='<unk>').save_pretrained(directory)
    if encoder:
        import torch
        from transformers import T5Config, T5EncoderModel
        torch.manual_seed(0)
        config = T5Config(vocab_size=len(vocab), d_model=24, d_kv=8, d_ff=32, num_layers=1, num_heads=3,
                          relative_attention_num_buckets=8, dropout_rate=0., pad_token_id=0, eos_token_id=1)
        T5EncoderModel(config).save_pretrained(directory)
    return Path(directory)


def sources(root, parts=10):
    """BLIP3o/DALL-E 3/ShareGPT-4o layouts with the edge cases the released loader handles."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root)
    blip = root / 'sources/blip3_ft60k'
    special = {'dalle3.tar': 'no-caption', 'geneval_train.tar': 'no-image', 'human_gestures.tar': 'truncated',
               'journeyDB.tar': 'bad-utf8', 'object_1.tar': 'jpeg'}
    for t, name in enumerate(BLIP3_FT60K_TARS):
        members = []
        for n in range(3):
            key, seed = f'{n * 7 + 3}', 100 * t + n
            data = image_bytes(seed, fmt='JPEG' if special.get(name) == 'jpeg' else 'PNG',
                               truncate=special.get(name) == 'truncated' and n == 1)
            text = caption(seed).encode()
            if n == 1 and special.get(name) == 'bad-utf8':
                text = b'\xff\xfe broken'
            if not (n == 1 and special.get(name) == 'no-image'):
                members.append((key + '.jpg', data))
            if not (n == 1 and special.get(name) == 'no-caption'):
                members.append((key + '.txt', text))
        write_tar(blip / name, members)
    dalle = root / 'sources/dalle3/data'
    dalle.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([('caption', pa.string()), ('image', pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
                        ('link', pa.string()), ('message_id', pa.string()), ('timestamp', pa.string()),
                        ('image_hash', pa.string()), ('synthetic_caption', pa.string())])
    for f in range(7):
        rows = []
        for n in range(3):
            seed = 1000 + 10 * f + n
            row = {'caption': '  ' + caption(seed) + ' ', 'image': {'bytes': image_bytes(seed, fmt='WEBP' if n == 2 else 'PNG'),
                   'path': f'{seed}.png'}, 'link': 'x', 'message_id': str(seed), 'timestamp': 't',
                   'image_hash': f'{seed:08x}', 'synthetic_caption': caption(seed + 1)}
            if f == 1 and n == 0:
                row['image'] = {'bytes': None, 'path': None}
            if f == 2 and n == 1:
                row['caption'] = None
            if f == 3 and n == 2:
                row['image_hash'] = None
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), dalle / f'train-{f:05d}-of-00007.parquet')
    share = root / 'sources/sharegpt4o'
    share.mkdir(parents=True, exist_ok=True)
    entries = []
    for part in range(parts):
        members = []
        for n in range(3):
            index = part * 10 + n
            mode = 'RGBA' if index == 1 else 'P' if index == 2 else 'RGB'
            members.append((f'image/{index}.png', image_bytes(5000 + index, size=(24, 18), mode=mode, truncate=index == 11)))
            if index != 21:
                entries.append({'input_prompt': caption(5000 + index, words=14 if index == 12 else 6),
                                'output_image': f'image/{index}.png', 'output_image_resolution': [24, 18]})
        write_tar(share / f'text_to_image_part_{part}.tar', members)
    entries.append({'input_prompt': caption(9999), 'output_image': 'image/999.png', 'output_image_resolution': [24, 18]})
    (share / 'text_to_image.json').write_text(json.dumps(entries))
    tokenizer_directory(root / 'sources/flan-t5-base')


def contract(root, path, updates=8, max_tokens=12):
    root = Path(root)

    def listing(directory, names):
        return [{'filename': n, 'bytes': (directory / n).stat().st_size, 'sha256': sha256(directory / n)} for n in names]
    share = root / 'sources/sharegpt4o'
    value = {
        'schema': 1, 'kind': 'minit2i-finetune-mixture-v1', 'upstream': {'repository': 'fixture', 'commit': 'fixture', 'files': []},
        'sources': [
            {'name': 'blip3_ft60k', 'repository': 'fixture/blip3', 'revision': 'r1', 'weight': '0.06',
             'files': listing(root / 'sources/blip3_ft60k', sorted(BLIP3_FT60K_TARS))},
            {'name': 'dalle3', 'repository': 'fixture/dalle3', 'revision': 'r2', 'weight': '0.016',
             'files': listing(root / 'sources/dalle3', sorted(f'data/{p.name}' for p in (root / 'sources/dalle3/data').glob('*.parquet')))},
            {'name': 'sharegpt4o', 'repository': 'fixture/sharegpt4o', 'revision': 'r3', 'weight': '0.04',
             'files': listing(share, sorted(p.name for p in share.iterdir() if p.is_file()))}],
        'caption_filter': {'source': 'sharegpt4o', 'max_tokens': max_tokens, 'tokenizer': {
            'repository': 'fixture/tokenizer', 'revision': 'r4',
            'files': listing(root / 'sources/flan-t5-base', sorted(p.name for p in (root / 'sources/flan-t5-base').iterdir()))}},
        'limits': {'max_rejected_fraction': .5},
        'sequence': {'seed': 42, 'updates': updates, 'global_batch': 1024}}
    Path(path).write_text(json.dumps(value, indent=1))
    return value
