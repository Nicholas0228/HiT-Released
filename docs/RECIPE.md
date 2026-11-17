# ImageNet model and training recipe

## Changes from released JiT

| Component | Released JiT | HNet |
|---|---|---|
| Patch input | Linear bottleneck B128 (B/L), B256 (H) | One linear RGB-patch → model-width projection |
| Conditioning | adaLN + 32 class slots inserted partway through | No adaLN; 8 class slots + 1 time token from block 1 |
| Blocks | One dense sequence | Encoder → intermediate encoder → core → intermediate decoder → decoder |
| Image tokens, /16 | 256 throughout | 256 → ~128 → ~64 → ~128 → 256 |
| Grouping | None | Two complete average-linkage trees; learned soft readers |
| Reconstruction | None | Global soft readback and two learned identity-initialized skips |
| Position | Fixed input sin/cos + image RoPE | Same input/outer image positions; no compact RoPE |

RMSNorm, QK normalization, SwiGLU and the JiT diffusion objective remain.
The time embedder uses physical `t` directly, without multiplying it by 1,000.
Prefix order is `[time, eight class slots, image patches]`. Conditioning tokens
evolve through every stage, receive identity RoPE, and bypass grouping, pooling,
image skips, readback and both compression budgets.

The compact input is `AᵀX / (sum(A) + 1e-8)` and image readback is `AH`, where
`A` is the global soft reader. Hard groups form the reader prototypes; pooling
is not restricted to the hard group. Training and sampling use the same readers.
Routing/readers use FP32 and exact tree updates use FP64. There is no spatial
window, approximate tree or narrower reader in these configs.

## Model sizes

Instantiated trainable counts exclude fixed positional buffers:

| Model config | Width / heads | Block layout | Parameters |
|---|---:|---|---:|
| `jit_b16.json` | 768 / 12 | 12 | 131,123,712 |
| `hnet_b16_d18.json` | 768 / 12 | 3/3/6/3/3 | 133,859,328 |
| `jit_l16.json` | 1,024 / 16 | 24 | 458,877,664 |
| `hnet_l16_d36.json` | 1,024 / 16 | 3/3/24/3/3 | 463,584,464 |
| `jit_h16.json` | 1,280 / 16 | 32 | 952,514,368 |
| `hnet_h16_d48.json` | 1,280 / 16 | 3/3/36/3/3 | 959,483,616 |
| `hnet_xl16.json` | 1,280 / 16 | 10/10/8/10/10 | 959,483,616 |

H and XL use head dimension 80 and projection/SwiGLU dropout 0.2 on global blocks
12–35. B/L use head dimension 64 and zero dropout. XL is H with seven blocks moved
from the core into each of the four outer stages (10/10/8/10/10); with seed 0 it
starts from the same initial weights per global block index as H. It trains with
the H recipe (global batch 1,024, eight GPUs × 128, EMA 0.9999/0.9996/0.9998,
CFG 2.4) toward 600 epochs. The LR is constant after warmup, so stopping at any
epoch is a prefix of that run. `imagenet_600ep_xl.json` runs FID-50k every 40
epochs; the reference runs also evaluated epochs 20 and 60. Run
`python scripts/describe.py` to recalculate counts without allocating full weights.

## Training

| Setting | Value |
|---|---|
| Duration / seed | 80 epochs / 0 |
| Global batch | 1,024 |
| Optimizer | AdamW, betas `(0.9, 0.95)`, epsilon `1e-8` |
| Learning rate | `2e-4`, five warmup epochs then constant |
| Weight decay / gradient clipping | Zero / disabled |
| Reader / budget temperature | 0.05 / 0.1 |
| Budget-loss weights | 0.01 per grouping stage |
| EMA | Primary 0.9999; also 0.9996; H additionally 0.9998 |

Counts are adaptive, not fixed quotas. Both calibrated thresholds remain frozen
during training. Stage 2 targets half the actual stage-1 image population.

B/H use accumulation 1, so both budgets see all 1,024 images at once.
L preserves its historical accumulation 2: on four ranks this means two global
512-image budget losses averaged per optimizer update. That differs from one
1,024-image budget; changing it is an experiment. Use eight GPUs for H and
measure the actual production batch before launch.

Use the verified historical ImageNet cache for the previous data order, or
`--data-format folder` for ordinary ImageFolder input. The folder adapter uses
JiT's center crop and stateless horizontal flips. Exact replay requires the
original corpus, ordering, topology and environment. A new run with a different
rank count preserves the global image order but changes rank-local noise streams.

Local JSONL logging is always enabled. Add `--wandb-project PROJECT
--wandb-entity ENTITY` for online logging using the receiving machine's
credentials. Architecture, training and runtime settings are stored separately
under `configs/` and resolved into each run's recorded configuration.

## Recovery and evaluation

Save full recovery hourly / every 1,251 updates, and permanent full checkpoints
at epochs 20/40/60/80. Recovery contains raw weights, AdamW, every maintained
EMA, every rank's RNG, data cursor, config and source fingerprint. A checkpoint
is deserialized and verified before advancing `checkpoints/latest.json`; two
verified rolling states and all permanent milestones remain.

Resume with the same command plus `--resume`. The saved identity rejects a
changed training world size or batch geometry. `SIGUSR1` requests a checkpointed
stop at the next update boundary. Use persistent output storage or a
deployment-owned backup sidecar; cloud transport is separate from the trainer.

Official ADM FID uses the same fixed 5k cohort every five epochs, replaced by
50k at epochs 20/40/60/80. Evaluate the primary EMA with CFG and no CFG:
50-step Heun with final Euler, denominator floor 0.05, guidance interval
`(0.1, 1)`, and CFG 2.9 for B or 2.4 for L/H. Per-image labels and noise seeds
are independent of evaluation rank assignment. Full asset and scorer setup is
in [evaluation/README.md](../evaluation/README.md).

## Calibration

For a changed depth, width, patch size, compression ratio or initialization,
recalibrate sequentially on the same fixed 1,024-image protocol:

```bash
python scripts/calibrate.py --model configs/models/hnet_b16_d18.json \
  --runtime configs/runtime/cuda.json --data "$IMAGENET_CACHE" \
  --output "$NEW_CALIBRATION_DIR"
```

This writes a new `model.json`, merge scores, count distributions and noise-level
statistics in a fresh directory.
