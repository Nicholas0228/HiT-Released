# Text-to-image: MiniT2I-B/16 and HiT-B/16 on CC12M

Both models generate 512×512 RGB images in pixel space from FLAN-T5-Large caption
features. They share the data, sample order, per-sample noise, objective, optimizer
and evaluation; only the architecture differs. This is the paper's B/16 pretraining recipe.

## Models

| Component | MiniT2I-B/16 (dense) | HiT-B/16 |
|---|---|---|
| Config | [minit2i_b16.json](../configs/t2i/models/minit2i_b16.json) | [hit_b16_d27.json](../configs/t2i/models/hit_b16_d27.json) |
| Parameters (T5 frozen, excluded) | 258,122,880 | 254,188,160 |
| Patch input, 512² / 16² = 1,024 tokens | Released 768→128→768 bottleneck | One linear 768→768 projection |
| Text | 256 T5 positions → 768, two text blocks | Same |
| Joint blocks | 17 two-stream blocks | 3/3/15/3/3 = 27 blocks |
| Time conditioning | None (as released) | One time token: raw t, 256 sinusoids, 256→768→768 MLP |
| Grouping | — | Image patches + valid caption tokens, two adaptive 2× stages |
| Compact blocks | — | 21 blocks with one transformation set for mixed groups, no RoPE |
| RoPE | 2D image, 1D text | Same in the outer blocks; identity for the time token |

The dense model is the pinned released PyTorch MiniT2I
([third_party/minit2i_torch](../third_party/minit2i_torch/SOURCE.json), unmodified),
constructed in its exact RNG order. Its unused time/pooled modules stay in
checkpoints, so DDP runs with `find_unused_parameters`.

HiT keeps the released text path and blocks: it constructs the same seed's dense
model and copies its text path, head, null caption token and 17 blocks into blocks
0–10 and 21–26. Core blocks 11–15 keep fresh weights, and blocks 16–20 come from
their own seed (7701), so every other module has the initialization of the 3/3/10/3/3
model it extends; nothing is averaged or pretrained. Compact blocks 3–23 keep only
their image branch.

**Grouping.** The stage-1 router sees the 1,024 patches plus every valid caption
token. Caption padding (and every caption position of a dropped caption) stays in
outer attention but never enters a tree, pooling, budget or compact block; it gets
only the full-resolution skip. The time token joins every compact sequence and
bypasses both routers. Each stage builds an exact average-linkage tree (FP32 heights
on the GPU, up to 1,280 vertices), cuts it at a calibrated threshold, and uses
global soft pooling and readback with an identity-initialized skip. Group counts
adapt per image and timestep: for ~90 valid caption tokens the nominal populations
are 1,114 → 557 → 278.

**Calibrated thresholds** (shipped in `hit_b16_d27.json`): 0.27459803223609924 and
0.24442464113235474, calibrated at seed-42 initialization on 1,024 fixed CC12M pairs
(`scripts/calibrate_t2i.py`). Rerun calibration only if you change the architecture,
initialization or ratios.

## Recipe

[cc12m_250k.json](../configs/t2i/training/cc12m_250k.json):

| Setting | Value |
|---|---|
| Data | 10,968,532 accepted CC12M pairs with LLaVA-NeXT captions; fixed 250k × 1,024 sample sequence |
| Images | PIL RGB, bicubic shortest-edge 512, center crop, [-1, 1], no flips |
| Objective | x-prediction, v-loss with denominator max(1−t, 0.05); t = sigmoid(N(−0.8, 0.8)), noise std 2 |
| Caption dropout | 10%: the whole caption becomes padding (learned null token) |
| HiT budgets | 0.01 × budget₁ + 0.01 × budget₂, each over the actual global population |
| Optimizer | AdamW (0.9, 0.95), ε 1e-8, no weight decay, no clipping |
| Learning rate | 1e-6 → 4e-4 over 5k updates, then constant |
| Batch | 1,024 global, microbatch 64 (memory only) |
| EMA | 0.99995 (reference) and 0.9995 (monitor), once per update |
| Precision | BF16 autocast; FP32 routers/readers; TF32 off |
| Evaluation | COCO FID-10k every 10k, FID-30k at 50/75/100/125/150/200/250k; CFG 2 and 1; 100 Euler steps; monitor EMA always, reference EMA every 50k |

Noise, timesteps and caption dropout are a pure function of the global sample
position (blocks of 128 positions, SHA-256-seeded CUDA generators), and every rank
reads its slice of the fixed sequence. A run therefore sees the same data and
noise on any GPU count. With gradient accumulation each HiT budget reduces over
one global microbatch instead of the whole update; the recorded run identity
includes this `budget_population`.

## Run it

**1. Environment.** PyTorch 2.5.1 with CUDA, then `pip install -r requirements.txt`
(Transformers, tokenizers, sentencepiece and clean-fid are pinned there).

**2. Inputs** (~1.2 TB; needs read access to the private derived bundle):

```bash
python scripts/fetch_t2i_inputs.py --output "$T2I_INPUTS" --token-file ~/.config/hf/token
```

This downloads the pinned public CC12M tar shards and FLAN-T5-Large, plus the
private record index, caption database, sample sequence and COCO-30k evaluation
assets. Each file is checked against the bundle index, which is itself SHA-pinned
in [configs/t2i/inputs.json](../configs/t2i/inputs.json). The token is used for
the public repositories too (higher rate limits); interrupted downloads are retried
and resumed, and the fetch finishes by writing `$T2I_INPUTS/inputs.json`.
`--updates N` fetches only the shards read by the first N updates (64 → 14 shards,
500 → 99 shards / 54 GB) for smoke tests. Read access to the private bundle is
required: the scripts that originally derived the index, captions, sequence and
COCO contexts from the public sources are not part of this repository.

**3. Train** (one node; use your launcher's `--nnodes`/rendezvous for more):

```bash
torchrun --standalone --nproc-per-node=8 main_t2i.py \
  --model configs/t2i/models/hit_b16_d27.json \
  --inputs "$T2I_INPUTS/inputs.json" --output "$RUN_DIR" --microbatch 128
```

Use `configs/t2i/models/minit2i_b16.json` for the dense baseline. Add `--resume`
to continue, `--max-updates N` for a checkpointed stop, `--microbatch N` if memory
is short, and `--wandb-project/--wandb-entity` for
online logging. `SIGUSR1` or `SIGTERM` stops at the next update with a verified
checkpoint. Outputs: `metrics.jsonl`, `configuration.json`, verified rolling
recovery states and permanent `checkpoint-uXXXXXXX.pt` every 10k updates, and
`evaluation/uXXXXXXX-ema*/` with FID results and sample images.

Use 8 × 128 or 16 × 64 without accumulation for the paper's full-batch HiT
budget objective. Smaller microbatches preserve the sample sequence but change
the budget reduction population.

Runtime profiles: `configs/t2i/runtime/cuda.json` (default; compiled MLPs and
readers), `eager.json` (no compilation) and `reference.json` (CPU tests).

**4. Evaluate any saved checkpoint** (any GPU count; same fixed cohort and noise):

```bash
torchrun --standalone --nproc-per-node=8 scripts/evaluate_t2i.py \
  --checkpoint "$RUN_DIR/checkpoints/checkpoint-u0250000.pt" --inputs "$T2I_INPUTS/inputs.json" \
  --count 30000 --ema 0.99995 --output "$RUN_DIR/coco30k-u250000"
```

`--guidances` and `--steps` change the sampler; `--save-images` keeps every PNG.
On one H200 the dense model samples 0.64 s per image at CFG 2 and 0.33 s at CFG 1,
so an FID-30k at both guidances takes about 1 h on 8 GPUs.

**5. Recalibrate (only for a changed architecture):**

```bash
torchrun --standalone --nproc-per-node=4 scripts/calibrate_t2i.py \
  --inputs "$T2I_INPUTS/inputs.json" --output "$RUN_DIR/calibration"
```

It writes `model.json` with the new thresholds. Any GPU count gives the same
cohort, noise and batches.

## Fine-tuning

Continue the full 250k checkpoint for 40k updates with the three-source data recipe
in [FINETUNING.md](FINETUNING.md). The released 290k EMA checkpoint can be evaluated
directly with [BENCHMARKS.md](BENCHMARKS.md).

The FA4/Blackwell runtime is omitted; `cuda.json` uses PyTorch SDPA and varlen FlashAttention.
