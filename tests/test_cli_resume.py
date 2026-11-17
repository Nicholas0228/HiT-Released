"""A real CLI stop/restart on eight synthetic images; never a quality experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
import pytest
from hnet.checkpoints import assert_state_equal


@pytest.mark.parametrize('world', [1, 2])
def test_cli_resume_matches_uninterrupted(tmp_path, world):
    root=Path(__file__).resolve().parents[1]
    for i in range(8):
        directory=tmp_path/'data'/f'class{i%2}'
        directory.mkdir(parents=True,exist_ok=True)
        pixels=np.random.default_rng(i).integers(0,256,(280+i,270+i,3),dtype=np.uint8)
        Image.fromarray(pixels).save(directory/f'{i}.png')
    model=dict(variant='hnet',image_size=256,patch_size=64,width=32,heads=4,classes=2,
               block_depths=[1,1,2,1,1],in_context_len=3,thresholds=[.6,.75])
    (tmp_path/'model.json').write_text(json.dumps(model))
    recipe=json.loads((root/'configs/training/imagenet_80ep.json').read_text())
    recipe.update(global_batch=4,epochs=2,workers=0,log_every=1,recovery_every_updates=99)
    (tmp_path/'recipe.json').write_text(json.dumps(recipe))
    base=[sys.executable,str(root/'main_jit.py'),'--model',str(tmp_path/'model.json'),
          '--training',str(tmp_path/'recipe.json'),'--runtime',str(root/'configs/runtime/reference.json'),
          '--data',str(tmp_path/'data'),'--data-format','folder','--device','cpu','--no-eval']
    if world > 1:
        base=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}']+base[1:]
    for name,extra in [('continuous',[]),('resumed',['--max-updates','1']),('resumed',['--resume'])]:
        subprocess.run(base+['--output',str(tmp_path/name)]+extra,cwd=root,check=True,capture_output=True,text=True)
    def load(name):
        directory=tmp_path/name/'checkpoints'
        receipt=json.loads((directory/'latest.json').read_text())
        return torch.load(directory/receipt['file'],map_location='cpu',weights_only=False)
    a,b=load('continuous'),load('resumed')
    for key in ('model','optimizer','ema','rank_rng'):
        assert_state_equal(a[key],b[key],key)
    for key in ('epoch','update_in_epoch','global_step','images_seen'):
        assert a['progress'][key]==b['progress'][key]
