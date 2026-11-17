# Text-to-image benchmarks

HiT-B/16 (3/3/15/3/3, 250k pretraining + 40k fine-tuning, EMA 0.99995) should reproduce:

| GenEval | DPG | PRISM | CoRe | Spatial | Average |
|---:|---:|---:|---:|---:|---:|
| 87.78 | 85.33 | 58.62 | 45.22 | 52.68 | 65.93 |

## 1. Inputs

- Checkpoint: `hf download Nicholas0228/HiT T2I512/HiT-B16_sft40k/model.safetensors --local-dir checkpoints` ([Hugging Face](https://huggingface.co/Nicholas0228/HiT)).
- Text encoder: `google/flan-t5-large` at revision `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`, as `$FLAN_T5_LARGE`.
- Prompts and scorer files, pinned by SHA-256 in `configs/benchmarks/assets.json`:

```bash
python scripts/benchmarks/fetch_assets.py --root $ASSETS --groups geneval dpg judge-data
```

## 2. Generate

Run once per benchmark: `geneval`, `dpg`, `prism`, `corebench`, `spatial_geneval`.

```bash
torchrun --standalone --nproc-per-node=8 scripts/benchmarks/generate.py --benchmark geneval \
  --checkpoint checkpoints/T2I512/HiT-B16_sft40k/model.safetensors --encoder-root $FLAN_T5_LARGE \
  --assets $ASSETS --output-root runs/t2i-benchmarks
```

The defaults are the paper's model and sampler: `configs/t2i/models/hit_b16_d27.json` and
`configs/t2i/sampling/hit_b16_d27.json`. The script prints the output folder and writes a verified
`manifest.json` there; any number of GPUs gives the same images.

## 3. Score

Build the scorer environments once with `scripts/benchmarks/build_env.sh geneval|dpg|judge ENV_DIR`.
Download the judges once with `python scripts/benchmarks/fetch_judges.py fetch --root $JUDGES`.

```bash
$GENEVAL_ENV/bin/python scripts/benchmarks/score_geneval.py --generation $OUT --assets $ASSETS --devices 0 1 2 3 4 5 6 7
$DPG_ENV/bin/python scripts/benchmarks/score_dpg.py --generation $OUT --assets $ASSETS --devices 0 1 2 3 4 5 6 7
$JUDGE_ENV/bin/python scripts/benchmarks/score_judge.py --generation $OUT --assets $ASSETS \
  --judge-model $JUDGES/Qwen2.5-VL-72B-Instruct-89c86200743e    # PRISM and Spatial
$JUDGE_ENV/bin/python scripts/benchmarks/score_judge.py --generation $OUT --assets $ASSETS \
  --judge-model $JUDGES/Qwen3-VL-32B-Thinking-7edd10ffd119      # CoRe
```

`$OUT` is the folder that `generate.py` printed for that benchmark. Each scorer writes a summary under
`$OUT/scores/`.

## Protocol

| Benchmark | Prompts | Images per prompt | Scorer |
|---|---:|---:|---|
| GenEval ([djghosh13/geneval@af4902f](https://github.com/djghosh13/geneval)) | 553 | 4 | Mask2Former Swin-S + CLIP ViT-L/14 |
| DPG ([ELLA@3c228f1](https://github.com/TencentQQGYLab/ELLA)) | 1,065 | 4, as a 2×2 grid | mPLUG-large VQA |
| PRISM ([prism-bench@e7f45f7](https://github.com/rongyaofang/prism-bench)) | 700 | 1 | Qwen2.5-VL-72B-Instruct |
| CoRe ([T2I-CoReBench@ebf7b7a](https://github.com/KlingAIResearch/T2I-CoReBench)) | 1,080 | 4 | Qwen3-VL-32B-Thinking |
| Spatial ([SpatialGenEval@8b294b5](https://github.com/AMAP-ML/SpatialGenEval)) | 1,230 | 1 | Qwen2.5-VL-72B-Instruct |

Images are 512×512 from 100 Euler steps with CFG 6.5 and group guidance over the whole trajectory. The
conditional branch uses 1.2× the calibrated group count at both grouping stages; the unconditional branch
uses 0.8× at stage 1 and keeps stage 2. Prompt `i` renders its images as one batch with noise seed
`i*1000 + (i % 8)*1000000` (seed 0), following the Looped-DiT evaluation code
([OpenSenseNova/Looped-DiT@18ad35c](https://github.com/OpenSenseNova/Looped-DiT)). The scorers run the
pinned upstream evaluation scripts unchanged.
