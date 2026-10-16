from pathlib import Path
import torch
from torch import nn
import pytest
from nmi_ced.bridge import SharedKVBridge
from nmi_ced.adapter import save_bridge,load_bridge_checkpoint

class Norm(nn.Module):
    def __init__(self,dim=8):super().__init__();self.weight=nn.Parameter(torch.zeros(dim))
    def forward(self,x):return x

def test_projector_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(11)
    a=SharedKVBridge(16,2,8,Norm())
    target=tmp_path/'proj.pt'
    save_bridge(a,target,model_id='test/mock',model_revision='a'*40,
                initialization='source36',training_stage='feature_projector_warmup')
    b=SharedKVBridge(16,2,8,Norm())
    meta=load_bridge_checkpoint(b,target,expected_revision='a'*40)
    assert meta['initialization']=='source36'
    assert meta['training_stage']=='feature_projector_warmup'
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k])
    with pytest.raises(ValueError):load_bridge_checkpoint(b,target,expected_revision='b'*40)
