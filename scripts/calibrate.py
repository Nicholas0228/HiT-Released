"""Sequentially calibrate both image routers, without updating model weights.

Use a fresh output directory. Existing recipe thresholds are never overwritten.
Calibration uses 1,024 fixed images, batch 64, subset seed 20261031 and noise
seed 882014; stage 2 receives the actual stage-1 soft-pooled representations.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from engine_jit import autocast
from hnet.model import CleanJiT
from hnet.config import load_config, MODEL_RUNTIME_KEYS
from hnet.data_cache import CachedImageNet
from hnet.data import IndexedImageNet, FolderImageNet
from hnet.runtime.config import configure_model_execution
from hnet.checkpoints import atomic_json


def select_threshold(scores, population, target_total):
    ordered = np.sort(scores)[::-1]
    if len(ordered) == 0: return 1., population
    desired = population - target_total
    candidates = set()
    for m in {int(math.floor(desired)), int(math.ceil(desired))}:
        m = min(max(m, 1), len(ordered))
        tau = float(ordered[m - 1])
        candidates.update([tau, float(np.nextafter(tau, np.inf))])
    threshold = min(sorted(candidates), key=lambda tau: (abs(population - int((scores >= tau).sum()) - target_total), -tau))
    return threshold, population - int((scores >= threshold).sum())


class CapturedTree(Exception): pass


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--runtime', default='configs/runtime/cuda.json')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--data-format', choices=['cached','indexed','folder'], default='cached')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    c,_,runtime = load_config(a.model, runtime_path=a.runtime)
    if c.variant != 'hnet': raise ValueError('Calibration is only needed for HNet')
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(a.device)
    net = CleanJiT(c).to(device).eval()
    configure_model_execution(net, runtime)
    data_cls = {'cached':CachedImageNet,'indexed':IndexedImageNet,'folder':FolderImageNet}[a.data_format]
    dataset = data_cls(a.data,seed=a.seed)
    if len(dataset)<1024: raise ValueError('Calibration requires 1,024 representative images')
    indices = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(20261031))[:1024]
    loader = DataLoader(Subset(dataset,indices.tolist()),batch_size=64,num_workers=0)
    generator = torch.Generator(device).manual_seed(882014)
    batches, digest = [], hashlib.sha256()
    for images,labels,ids in loader:
        x=images.to(device).float()/127.5-1; y=labels.to(device)
        drop=torch.rand(len(x),device=device,generator=generator)<.1
        y=torch.where(drop,c.classes,y)
        t=(torch.randn(len(x),device=device,generator=generator)*.8-.8).sigmoid()
        noise=torch.randn(x.shape,device=device,generator=generator)
        for value in (images,ids,y.cpu(),t.cpu(),noise.cpu()): digest.update(value.contiguous().numpy().tobytes())
        noisy=t[:,None,None,None]*x+(1-t[:,None,None,None])*noise
        batches.append((noisy.cpu(),t.cpu(),y.cpu()))
    report=dict(images=1024,model_seed=a.seed,subset_seed=20261031,noise_seed=882014,batch_size=64,
                dataset=dataset.fingerprint,inputs_sha256=digest.hexdigest(),runtime=runtime,stages=[])
    for stage, router, ratio in [(1,net.router,c.target_ratio),(2,net.router2,c.second_ratio)]:
        scores, populations, times = [], [], []
        original = router.structure
        def capture(q,k,*args,**kwargs):
            structure = original(q,k,*args,**kwargs)
            lengths = kwargs.get('lengths', args[0] if args else [q.shape[1]]*q.shape[0])
            for i,n in enumerate(lengths):
                scores.append(structure.scores[i,:n-1].cpu().numpy()); populations.append(int(n))
            raise CapturedTree()
        router.structure = capture
        try:
            for x,t,y in batches:
                times.extend(t.tolist())
                with autocast(device,runtime.get('dtype','float32')):
                    try: net(x.to(device),t.to(device),y.to(device))
                    except CapturedTree: pass
                    else: raise RuntimeError('Calibration did not reach the intended router')
        finally:
            router.structure = original
        flat=np.concatenate(scores); population=sum(populations)
        tau,total=select_threshold(flat,population,population/ratio)
        counts=np.array([n-int((s>=tau).sum()) for n,s in zip(populations,scores)])
        router.threshold.fill_(tau)
        bins=[]; t=np.array(times)
        for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            selected=counts[(t>=low)&(t<high)]
            bins.append(dict(low=float(low),high=float(high),images=len(selected),mean=float(selected.mean()) if len(selected) else None))
        report['stages'].append(dict(stage=stage,threshold=tau,population=population,target=population/ratio,
                                    total_groups=total,mean_groups=float(counts.mean()),singleton_images=int((counts==1).sum()),
                                    quantiles={str(q):float(np.quantile(counts,q)) for q in (0,.1,.5,.9,1)},noise_bins=bins))
        np.savez(a.output/f'stage{stage}.npz',scores=flat,populations=populations,counts=counts,t=times,indices=indices.numpy())
    model={k:v for k,v in asdict(c).items() if k not in MODEL_RUNTIME_KEYS}
    model['thresholds']=[r['threshold'] for r in report['stages']]
    atomic_json(a.output/'model.json',model);atomic_json(a.output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
