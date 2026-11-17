"""Instantiate configs on meta to report exact trainable parameter counts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hnet.config import load_config
from hnet.model import CleanJiT


def describe(path):
    config,_,_ = load_config(path)
    with torch.device('meta'): model = CleanJiT(config)
    return dict(config=str(path),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                width=config.width,heads=config.heads,head_dimension=config.width//config.heads,
                block_depths=list(config.block_depths),class_tokens=config.in_context_len,
                time_tokens=int(config.variant=='hnet'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('models',nargs='*');a=p.parse_args()
    result=[describe(path) for path in (a.models or sorted(Path('configs/models').glob('*.json')))]
    text=json.dumps(result,indent=2);print(text)
    if a.output:a.output.write_text(text+'\n')
