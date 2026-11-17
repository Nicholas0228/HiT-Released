"""Separate-process legacy/clean GPU parity, including compiled Hopper kernels.

Use a single idle GPU. Kernel custom-op namespaces require separate processes.
All gradient tensors are compared; small deterministic nonzero head weights keep
the check meaningful even without a supplied trained checkpoint.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from hnet.config import load_config
from hnet.checkpoints import atomic_json, sha256_file


def legacy_config(c, module):
    fields=asdict(c)
    for name in ('block_depths','thresholds','ratios'):fields.pop(name)
    fields.update(depth=c.depth,stages=c.stages,encoder_depth=c.encoder_depth,
                  intermediate_depth=c.intermediate_depth,decoder_depth=c.decoder_depth,
                  threshold=c.threshold,second_threshold=c.second_threshold,target_ratio=c.target_ratio,
                  second_ratio=c.second_ratio,conditioning='tokens',in_context_start=c.encoder_depth+c.intermediate_depth)
    return module.ModelConfig(**fields)


def worker(a):
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.cache_size_limit=128
    c,_,runtime=load_config(a.model,runtime_path=a.runtime)
    if c.variant!='hnet':raise ValueError('This old/new hierarchy gate is for HNet')
    if a.worker=='old':
        sys.path.append(str(a.legacy))
        from hnet_jit import model as m
        from hnet_jit.denoising import Objective
        from hnet_jit.runtime_config import configure_model_execution
        config=legacy_config(c,m)
    else:
        from denoiser import Objective
        from hnet.runtime.config import configure_model_execution
        config=c
    model=Objective(config)
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    initialization_sha256=digest.hexdigest()
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)['model']
        model.load_state_dict(state,strict=True)
    else:
        with torch.no_grad():model.net.final_layer.linear.weight.normal_(std=.005)
    model=model.cuda().eval()
    configure_model_execution(model.net,runtime)
    if a.deterministic_attention:
        if runtime.get('execution_backend')!='hopper':
            raise ValueError('Deterministic FA3 diagnostic requires Hopper')
        module=importlib.import_module('hnet_jit.runtime_attention' if a.worker=='old' else 'hnet.runtime.attention')
        original_backward=module._flash_attn_backward
        def deterministic_backward(*args,**kwargs):
            kwargs['deterministic']=True
            return original_backward(*args,**kwargs)
        # Diagnostic-only: apply the same FA3 option in both processes.
        # Production model code and the default runtime remain unchanged.
        module._flash_attn_backward=deterministic_backward
    rng=torch.Generator('cuda').manual_seed(739)
    images=torch.randn((4,3,c.image_size,c.image_size),device='cuda',generator=rng)
    times=torch.tensor([.1,.4,.7,.95],device='cuda')
    stochastic=(times,torch.randn(images.shape,device='cuda',generator=rng),torch.tensor([False,False,False,True],device='cuda'))
    labels=torch.tensor([1,13,130,2],device='cuda')
    def run():
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):result=model(images,labels,stochastic=stochastic)
        result['loss'].backward()
        return result
    started=time.monotonic();result=run()
    gradients={k:p.grad.detach().cpu().clone() if p.grad is not None else None for k,p in model.named_parameters()}
    records=[]
    def capture(_module,_args,out):
        compact,assignment,valid,structure,soft=out
        records.append(dict(groups=structure.groups.cpu(),counts=structure.counts.cpu(),assignment=assignment.cpu(),compact=compact.cpu()))
    handles=[r.register_forward_hook(capture) for r in (model.net.router,model.net.router2)]
    repeat=run()
    for handle in handles:handle.remove()
    training_records=records
    squared=denominator=0.
    for name,p in model.named_parameters():
        if p.grad is None:continue
        delta=p.grad.detach().cpu().double()-gradients[name].double()
        squared+=delta.square().sum().item();denominator+=gradients[name].double().square().sum().item()
    records=[]
    handles=[r.register_forward_hook(capture) for r in (model.net.router,model.net.router2)]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        inference=model.net(images,times,labels,return_aux=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=model.net(images,times,labels,return_aux=True)
    for handle in handles:handle.remove()
    if len(records)!=4:raise RuntimeError('Expected two routers in both training and sampling')
    for left,right in zip(records[:2],records[2:]):
        for key in ('groups','counts'):
            if not torch.equal(left[key],right[key]):raise RuntimeError('Training/sampling lookup mismatch')
        for key in ('assignment','compact'):
            torch.testing.assert_close(left[key],right[key],rtol=1e-5,atol=1e-5)
    if not torch.equal(inference[0],prediction[0]):
        torch.testing.assert_close(inference[0],prediction[0],rtol=1e-5,atol=1e-5)
    saved=dict(initialization_sha256=initialization_sha256,gradients=gradients,groups=training_records,
               outputs=prediction[0].detach().cpu(),loss=result['loss'].detach().cpu(),
               budget=result['budget'].detach().cpu(),repeat_gradient_relative_l2=(squared/max(denominator,1e-30))**.5,
               seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    torch.save(saved,a.output/f'{a.worker}.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--model',default='configs/models/hnet_b16_d18.json')
    p.add_argument('--runtime',default='configs/runtime/hopper.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--worker',choices=['old','new'])
    p.add_argument('--deterministic-attention',action='store_true',
                   help='Diagnostic-only deterministic FA3 backward in both implementations')
    a=p.parse_args()
    if a.worker:return worker(a)
    a.output.mkdir(parents=True,exist_ok=False)
    pin=json.loads((ROOT/'docs/reference/source.json').read_text())
    for path,digest in pin['legacy_files_sha256'].items():
        if sha256_file(a.legacy/path)!=digest:raise RuntimeError('Legacy source mismatch: '+path)
    if a.checkpoint and (not a.checkpoint_sha256 or sha256_file(a.checkpoint)!=a.checkpoint_sha256):
        raise RuntimeError('Supply the verified checkpoint SHA-256')
    for role in ('old','new'):
        command=[sys.executable,__file__,'--legacy',str(a.legacy),'--model',a.model,'--runtime',a.runtime,
                 '--output',str(a.output),'--worker',role]
        if a.checkpoint:command+=['--checkpoint',str(a.checkpoint)]
        if a.deterministic_attention:command+=['--deterministic-attention']
        with (a.output/f'{role}.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    old,new=[torch.load(a.output/f'{role}.pt',map_location='cpu',weights_only=False,mmap=True) for role in ('old','new')]
    if old['initialization_sha256']!=new['initialization_sha256']:raise RuntimeError('Initialization mismatch')
    if len(old['groups'])!=2 or len(new['groups'])!=2:raise RuntimeError('Expected both routing levels')
    for key in ('loss','budget'):
        torch.testing.assert_close(new[key],old[key],rtol=1e-6,atol=1e-6)
    for stage,(a1,b1) in enumerate(zip(old['groups'],new['groups'])):
        for key in ('groups','counts'):
            if not torch.equal(a1[key],b1[key]):raise RuntimeError(f'Hard lookup mismatch: stage {stage+1}, {key}')
        for key in ('assignment','compact'):
            torch.testing.assert_close(b1[key],a1[key],rtol=1e-5,atol=1e-5)
    squared=denominator=0.;worst=[]
    if old['gradients'].keys()!=new['gradients'].keys():raise RuntimeError('Gradient key mismatch')
    for name,expected in old['gradients'].items():
        actual=new['gradients'][name]
        if expected is None:
            if actual is not None:raise RuntimeError('Unexpected gradient: '+name)
            continue
        if actual is None or not torch.isfinite(actual).all():raise RuntimeError('Missing/nonfinite gradient: '+name)
        error=(actual.double()-expected.double()).square().sum().item();base=expected.double().square().sum().item()
        absolute_l2=error**.5
        if absolute_l2>5e-3*base**.5+1e-7:
            raise RuntimeError('Per-parameter gradient mismatch: '+name)
        squared+=error;denominator+=base
        worst.append((name,(error/max(base,1e-30))**.5))
    gradient_error=(squared/denominator)**.5
    output_error=float((old['outputs'].double()-new['outputs'].double()).norm()/old['outputs'].double().norm())
    # Production FA3 backward is nondeterministic. Bound the discrepancy by both
    # a fixed 0.1% ceiling and observed repeat variation, not an arbitrary pass.
    # The deterministic FA3 diagnostic retains the original strict limit.
    repeat_floor=max(old['repeat_gradient_relative_l2'],new['repeat_gradient_relative_l2'])
    limit=5e-4 if a.deterministic_attention else min(1e-3,max(5e-4,1.25*repeat_floor))
    if gradient_error>limit or output_error>1e-5:raise RuntimeError(f'Numerical mismatch: gradients={gradient_error}, output={output_error}')
    report=dict(passed=True,legacy_commit=pin['legacy_commit'],model=a.model,runtime=a.runtime,
                checkpoint_sha256=a.checkpoint_sha256,initialization_bitwise_equal=True,hard_groups_equal=True,
                deterministic_attention_diagnostic=a.deterministic_attention,
                loss_and_budget_close=True,reader_assignments_and_compact_inputs_close=True,
                gradient_relative_l2=gradient_error,gradient_tolerance=limit,output_relative_l2=output_error,
                worst_parameter_gradient_errors=sorted(worst,key=lambda x:x[1],reverse=True)[:10],
                old_repeat=old['repeat_gradient_relative_l2'],new_repeat=new['repeat_gradient_relative_l2'],
                train_inference_checked=True,parameters_with_gradients=len(worst),
                workers={role:{key:data[key] for key in ('seconds','peak_gpu_gib')} for role,data in [('old',old),('new',new)]})
    atomic_json(a.output/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
