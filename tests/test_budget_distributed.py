"""Actual Gloo collective test of DDP gradient scaling and global populations."""
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hnet.grouping import budget_loss_population


def worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=world)
    # Variable populations make an average of per-image fractions incorrect.
    hard = torch.tensor([4.,8.,11.,25.,17.,32.,40.,50.])
    population = torch.tensor([10.,18.,30.,50.,40.,64.,80.,110.])
    features = torch.arange(1.,9.)
    weight = torch.tensor(.37, requires_grad=True)
    soft = weight * features[rank::world] + hard[rank::world]
    loss, fraction, _ = budget_loss_population(hard[rank::world], soft, population[rank::world], 2.)
    loss.backward()
    averaged = weight.grad.clone(); dist.all_reduce(averaged); averaged /= world
    if rank == 0: torch.save({'loss':loss.detach(),'fraction':fraction,'gradient':averaged}, output)
    dist.destroy_process_group()


def test_global_budget_matches_single_process(tmp_path):
    hard = torch.tensor([4.,8.,11.,25.,17.,32.,40.,50.])
    population = torch.tensor([10.,18.,30.,50.,40.,64.,80.,110.])
    features = torch.arange(1.,9.)
    w = torch.tensor(.37, requires_grad=True)
    loss, fraction, _ = budget_loss_population(hard,w*features+hard,population,2.,distributed=False)
    loss.backward()
    output = tmp_path / 'result.pt'
    mp.spawn(worker, args=(2, 'file://' + str(tmp_path / 'rdzv'), str(output)), nprocs=2, join=True)
    actual = torch.load(output, weights_only=True)
    torch.testing.assert_close(actual['loss'],loss.detach(),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(actual['fraction'],fraction,rtol=0,atol=0)
    torch.testing.assert_close(actual['gradient'],w.grad,rtol=1e-6,atol=1e-7)
