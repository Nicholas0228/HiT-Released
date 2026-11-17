# Official ADM evaluation

Keep TensorFlow separate from the training environment. The established score
track uses TensorFlow 2.15.1, NumPy 1.26.4 and SciPy 1.14.1, with TF32 disabled.
Install Pillow and the official evaluator's dependencies in that environment.

Supply an asset manifest with absolute paths and SHA-256 hashes:

```json
{
  "evaluator": {"path": "/assets/evaluator.py", "sha256": "FULL_SHA256"},
  "inception": {"path": "/assets/classify_image_graph_def.pb", "sha256": "FULL_SHA256"},
  "reference": {"path": "/assets/VIRTUAL_imagenet256_labeled.npz", "sha256": "FULL_SHA256"}
}
```

Use OpenAI's [guided-diffusion evaluator](https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py)
and [ImageNet-256 reference](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz).
The established evaluator file SHA-256 is
`c8f10ab1406c45d105d7f5b347e5be06c118bd2ddef1c54789c263ebae7f1ca7`;
reuse the verified assets from the old runs when comparing their scores.
An arbitrary newly downloaded `main` is not automatically the same evaluator.

The trainer samples PNGs into a unique attempt directory and verifies complete
IDs, labels, per-image seeds, checksums and RGB geometry before scoring. It then
invokes `score_adm.py` in the supplied `--adm-python` environment. Scoring is
synchronous on rank 0 while other training ranks wait. A failed score preserves
its images/log and stops the run with its latest pre-evaluation checkpoint.
Resuming evaluates that still-due epoch again in a new attempt directory.

The protocol is EMA 0.9999, 50-step Heun plus final Euler, floor 0.05, CFG in
`(0.1,1)`, B CFG 2.9 or L/H CFG 2.4, and no CFG. For each cohort, ID `i` has label
`i // (count/1000)` and Gaussian seed `312799+i`. IDs range exactly from 0 to
`count-1`. The fixed 5k cohort is repeated across epochs; it is not a subset with
the same label assignment as the 50k cohort. Their FID estimates must be kept
separate. Changing evaluation rank count preserves the sample definitions;
kernel/batch-dependent floating-point differences remain possible.

Image output uses upstream round-to-nearest uint8 RGB conversion. The scorer
streams those same pixels into the official Inception model. Output is FID and
IS, with manifest/reference/evaluator hashes. No sFID or precision/recall score
is implied. Put evaluation output on storage shared by all participating ranks.

The adapter has additionally passed actual TensorFlow feature/statistics parity
against the native NPZ reader on 128 stored reference images. This verifies the
scoring integration; it does not independently reproduce an old generated-image
FID. See `docs/VALIDATION.md` for the exact evidence and remaining deployment checks.
