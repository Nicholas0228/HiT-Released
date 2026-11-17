# SFT40k validation

## CPU suite

`python -m pytest -q -rs`: **47 passed, 1 skipped** on November 17, 2026.
The skipped test requires the optional downloaded benchmark assets.

The 13 fine-tuning tests cover:

- Synthetic BLIP3o, DALL-E 3 and ShareGPT-4o sources, including caption filtering,
  missing images, malformed captions and decode failures.
- Decoded pixels and caption strings against the upstream preparation rules,
  including ShareGPT-4o's JPEG-95 round trip.
- Weighted source selection, shuffled source passes, sequence regeneration,
  changed-file rejection and recovery after interrupted preparation.
- An actual small model with AdamW and both EMAs: parent restoration, the next
  update, and a subsequent SFT checkpoint/resume are bitwise equal on CPU.
- Rejection of wrong parent updates, changed architecture/budget settings and
  inference-only checkpoints; correct phase data offsets and global noise positions.

## Two-GPU CLI check

**Passed on two H100 GPUs.** The largest final tensor difference between resumed
and uninterrupted training was **3.13e-7**; rank RNG states and cursors matched
exactly. The machine-readable result is [sft40k-cli.json](sft40k-cli.json).

On two CUDA GPUs, run:

```bash
python tests/run_sft_cli_smoke.py run /path/to/new/scratch/sft-smoke
```

The script creates tiny synthetic images, a tokenizer/T5 encoder and a small HiT
model. Its synthetic parent has one actual training update and boundary metadata
at 250k; it is not a trained 250k checkpoint. It launches the real `main_t2i.py`
entry point, stops after two SFT updates, resumes for two more, and compares with
four uninterrupted updates. `result.json` records the outcome and maximum
numerical difference.

Raw weights, AdamW and both EMAs are checked exactly when restored, before
training continues. Final rank RNG states and data/global update cursors must
also match exactly. Subsequent GPU updates are compared with BF16 tolerances
(`rtol=1e-2`, `atol=1e-5`): the router's CUDA cumulative sum has no deterministic
implementation in PyTorch 2.5.1. The smoke uses deterministic cuDNN kernels and
a fixed cuBLAS workspace; it does not modify the production runtime settings.

## Environment and scope

Python 3.10.21, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, NumPy 1.26.4,
Transformers 4.46.3, tokenizers 0.20.3, Pillow 12.3.0, numba 0.61.2,
PyArrow 17.0.0 and pytest 8.4.2. GPU smoke hardware: two H100 80 GB GPUs.

These checks validate preparation and continuation behavior. The complete
119,239-pair mixture was not downloaded/rebuilt for this port, and no full
40k-update training or benchmark rerun is claimed. Dataset revisions, file hashes
and expected populations are recorded in `configs/t2i/sft_data.json`.
