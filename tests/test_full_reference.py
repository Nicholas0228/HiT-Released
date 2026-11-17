"""Run explicitly on compute: full B/16 D18, not a reduced architecture."""
import os
import pytest
import torch
from hnet.config import load_config
from hnet.model import CleanJiT
from conftest import legacy_config, equal_tree


@pytest.mark.full
@pytest.mark.skipif(os.environ.get('HNET_FULL_PARITY') != '1', reason='Set HNET_FULL_PARITY=1 on compute')
def test_full_d18_forward_backward(legacy):
    old_module, _ = legacy
    c, _, _ = load_config('configs/models/hnet_b16_d18.json')
    torch.manual_seed(0); old = old_module.CleanJiT(legacy_config(c, old_module))
    rng = torch.get_rng_state()
    torch.manual_seed(0); new = CleanJiT(c)
    assert torch.equal(rng, torch.get_rng_state())
    equal_tree(old.state_dict(), new.state_dict())
    assert sum(p.numel() for p in new.parameters()) == 133859328
    with torch.no_grad(): old.final_layer.linear.weight.normal_(std=.002)
    new.load_state_dict(old.state_dict())
    x, t, y = torch.randn(1,3,256,256), torch.tensor([.7]), torch.tensor([130])
    expected, actual = old(x,t,y), new(x,t,y)
    equal_tree(expected, actual, 'full_d18_output')
    expected.square().mean().backward(); actual.square().mean().backward()
    for (name,p), (other,q) in zip(old.named_parameters(),new.named_parameters()):
        assert name == other
        equal_tree(p.grad,q.grad,name)
