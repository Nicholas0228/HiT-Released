import numpy as np
import pytest
from PIL import Image
import torch
from evaluation.adm import cohort, initial_noise, audit_images
from evaluation.score_adm import Images
from hnet.checkpoints import sha256_file


def test_evaluation_four_eight_rank_identity():
    expected = cohort(5000)
    for world in (1, 4, 8):
        combined = sorted([r for rank in range(world) for r in cohort(5000, rank, world)], key=lambda r:r['id'])
        assert combined == expected
    # Every noise tensor for a fixed small cohort survives ownership and batch changes.
    rows = cohort(16, classes=4)
    full = initial_noise(rows, 8, torch.device('cpu'))
    shards = [(row['id'], noise) for rank in range(8) for row, noise in
              zip(rows[rank::8], initial_noise(rows[rank::8], 8, torch.device('cpu')))]
    assert torch.equal(full, torch.stack([noise for _, noise in sorted(shards)]))


def test_streamed_pixels_and_missing_image_failure(tmp_path):
    rows = cohort(4, classes=2)
    for row in rows:
        Image.fromarray(np.full((8, 8, 3), row['id'] * 40, dtype=np.uint8)).save(tmp_path / row['filename'])
        row['sha256'] = sha256_file(tmp_path / row['filename'])
    audit_images(tmp_path, rows, 8)
    source = Images(tmp_path, rows, 8)
    decoded = np.concatenate(list(source))
    assert source.seen == 4 and decoded.shape == (4, 8, 8, 3)
    np.testing.assert_array_equal(decoded[:,0,0,0], [0,40,80,120])
    (tmp_path / rows[2]['filename']).unlink()
    with pytest.raises(RuntimeError, match='Missing'): audit_images(tmp_path, rows, 8)
