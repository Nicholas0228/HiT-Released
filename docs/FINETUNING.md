# Text-to-image fine-tuning (SFT40k)

HiT-B/16 uses 250k CC12M pretraining updates followed by 40k fine-tuning updates.
The fine-tuning model is the same **3/3/15/3/3** architecture, with both budget
weights at **0.01**. The released **290k / SFT40k EMA checkpoint** is sufficient
for generation and the five benchmarks; see [BENCHMARKS.md](BENCHMARKS.md).

This guide covers training the fine-tuning stage yourself. It continues a full
250k checkpoint from [main_t2i.py](../main_t2i.py), retaining the raw weights,
AdamW moments, both EMAs and rank RNG states. The 250k training checkpoint is
not bundled with the inference weights; it is produced by the
[pretraining recipe](T2I.md). An EMA-only 290k export cannot restore that state.

## Data

The mixture follows [MiniT2I's data preparation](https://github.com/Hope7Happiness/minit2i-torch/tree/e54e15d7445d386e4c30663861c9d42288e87b66).
The script downloads from the original public repositories. No separate private
SFT data bundle is needed.

| Source | Accepted pairs | Sampling weight |
|---|---:|---:|
| [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) | 58,859 | 15/29 |
| [DALL-E 3](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) | 19,024 | 4/29 |
| [ShareGPT-4o](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) | 41,356 | 10/29 |

[sft_data.json](../configs/t2i/sft_data.json) pins the source revisions and each
file's SHA-256. Captions use BLIP3o's stored text, DALL-E 3's caption or synthetic
caption, and ShareGPT-4o's input prompt. ShareGPT-4o retains captions with at most
256 FLAN-T5-Base tokens and applies the upstream JPEG quality-95 round trip.
All images use RGB, bicubic shortest-edge resize to 512, center crop, and no flips.
The frozen training encoder remains **FLAN-T5-Large**, as in pretraining.

The builder validates every image and records exclusions, then constructs a
fixed 40k × 1,024 sample sequence using seed 42. It samples sources in the weights
above and shuffles each source without replacement within each pass. Repeated
passes are expected: the mixture contains 119,239 pairs and training consumes
40,960,000 examples. This provides deterministic restart and matched order for
the dense and HiT models; it does not require the original experiment's sequence
file. A different order can still change the resulting checkpoint and scores.

### Prepare

Install the repository requirements, including `pyarrow`. Reserve about **350 GB**
for the downloaded sources, staged image shards and indexes. Reuse the local
FLAN-T5-Large snapshot fetched for pretraining as `$FLAN_T5_LARGE`.

```bash
python scripts/prepare_t2i_finetune.py \
  --output "$SFT_DATA" --encoder-root "$FLAN_T5_LARGE" \
  --workers 12 --download-workers 6
```

It writes `inputs.json`, source and build receipts, `manifest.json`, a caption
database, image indexes, and `sequence.npy`. Interrupted preparation can be
rerun with the same command; completed stages are checked and reused. Original
images and captions remain subject to their dataset licenses.

## Training

[sft40k.json](../configs/t2i/training/sft40k.json) continues the pretraining recipe:

| Setting | Value |
|---|---|
| Global update range | 250,000 → 290,000 |
| Global batch | 1,024; 8 × 128 or 16 × 64, without accumulation |
| Learning rate | 4e-4, constant; no new warmup |
| AdamW | β=(0.9, 0.95), ε=1e-8, weight decay 0 |
| EMA | Continue 0.99995 and 0.9995 |
| HiT budgets | 0.01 per stage, one global budget population per update |
| Grouping thresholds | Unchanged from the pretraining config |
| Recovery | Hourly full state; permanent milestones every 10k updates |

Use the same GPU count as the parent checkpoint so every rank's RNG state can
be restored. For eight GPUs, set microbatch 128 in **both** pretraining and
fine-tuning; for sixteen, use 64. Accumulation is rejected for HiT SFT because
it changes the population used by the budget objective.

```bash
torchrun --standalone --nproc-per-node=8 main_t2i.py \
  --model configs/t2i/models/hit_b16_d27.json \
  --training configs/t2i/training/sft40k.json \
  --inputs "$SFT_DATA/inputs.json" --output "$SFT_RUN" \
  --microbatch 128 \
  --finetune-from "$PRETRAIN_RUN/checkpoints/checkpoint-u0250000.pt"
```

The output directory must be new. The trainer verifies the architecture,
thresholds, budget weights and continued optimizer/EMA settings, and saves the
transition before consuming data. Only the data cursor resets to zero. Noise,
timesteps and caption dropout retain their global sample positions. The parent
checkpoint is left unchanged.

To resume, replace `--finetune-from ...` with `--resume`, keeping the other
arguments and output directory. `--max-updates 251000` stops after the first 1k
fine-tuning updates with full recovery. `SIGUSR1` and `SIGTERM` stop at an update
boundary. `metrics.jsonl` records generation loss, both budget losses, group
counts, gradient norms and update timing. Checkpoint progress records both the
global update and `phase_update` / `phase_next_position`.

For the dense comparison, use `configs/t2i/models/minit2i_b16.json` and its own
full 250k parent, with the same prepared mixture.

## Evaluation

Fine-tuning checkpoints are saved at SFT10k/20k/30k/40k (global updates
260k/270k/280k/290k). Evaluation runs separately; there is no synchronous COCO
evaluation in the fine-tuning recipe.

```bash
torchrun --standalone --nproc-per-node=8 scripts/benchmarks/generate.py \
  --benchmark geneval \
  --checkpoint "$SFT_RUN/checkpoints/checkpoint-u0290000.pt" \
  --ema 0.99995 --encoder-root "$FLAN_T5_LARGE" \
  --assets "$ASSETS" --output-root "$SFT_RUN/benchmarks"
```

Generation uses the existing HiT sampler config (CFG 6.5, group guidance, seed 0,
100 Euler steps). See [BENCHMARKS.md](BENCHMARKS.md) for assets, scorers, the
other four benchmarks, and evaluation of the released 290k checkpoint directly.

COCO FID can also be run with `scripts/evaluate_t2i.py`, passing the saved SFT
checkpoint and the **pretraining** inputs manifest containing the COCO evaluation
assets. It is separate from the SFT training mixture.

Implementation checks and the two-GPU recovery smoke are documented in
[SFT40k validation](validation/t2i/sft40k.md).
