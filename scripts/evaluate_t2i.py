"""Evaluate a saved text-to-image checkpoint with the fixed COCO FID protocol.

Works with any recovery state or permanent checkpoint written by main_t2i.py, on
any GPU count (the prompt/noise cohort and its batches do not depend on it):

  torchrun --standalone --nproc-per-node=8 scripts/evaluate_t2i.py \
    --checkpoint "$RUN_DIR/checkpoints/checkpoint-u0250000.pt" \
    --inputs "$T2I_INPUTS/inputs.json" --count 30000 --ema 0.99995 --output "$RUN_DIR/eval-u250000"

Add --save-images to keep every generated PNG (for example for GenEval-style scorers).
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from denoiser import DualEMA  # noqa: E402
from t2i.config import T2IConfig  # noqa: E402
from t2i.evaluation import evaluate  # noqa: E402
from t2i.inputs import load_inputs  # noqa: E402
from t2i.model import TextToImage  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory for features, scores and samples')
    p.add_argument('--count', type=int, default=30000, help='First N prompts of the fixed COCO-30k cohort')
    p.add_argument('--ema', type=float, default=.99995, help='EMA decay to evaluate; 0 evaluates the raw weights')
    p.add_argument('--guidances', type=float, nargs='+', default=[2., 1.])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--save-images', action='store_true')
    args = p.parse_args()
    rank, world, local = [int(os.environ.get(k, d)) for k, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    identity = state['identity']
    config = T2IConfig(**identity['model'])
    recipe, runtime = identity['recipe'], identity['runtime']
    model = TextToImage(config).cuda()
    model.load_state_dict(state['model'], strict=True)
    ema = DualEMA(model, decays=tuple(recipe['ema_decays']))
    ema.load_state_dict(state['ema'])
    if args.ema:
        index = list(ema.decays).index(args.ema)
    else:
        ema = DualEMA(model, decays=(1.,))  # identity "EMA" that holds the raw weights
        index = 0
    if runtime.get('compile', False):
        model.compile_blocks()
    plan = recipe['evaluation']
    inputs = load_inputs(args.inputs, require_evaluation=True)
    update = state['progress']['global_step']
    scores = evaluate(model, ema, inputs['evaluation'], args.output, update, args.count, ema_index=index,
                      guidances=tuple(args.guidances), steps=args.steps, a_min=plan['a_min'],
                      batch_size=plan['generation_batch'], logical_world=plan['logical_world'],
                      prefix=f'eval/ema{args.ema:g}', save_images=args.save_images)
    if rank == 0:
        print(json.dumps(dict(scores, update=update, checkpoint=str(args.checkpoint)), indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
