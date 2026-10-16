#!/usr/bin/env python3
"""Offline projector warmup using genuine Qwen feature traces (no 27B model loaded).

Matches the EXACT pre-RoPE feature convention of SharedKVBridge.
"""
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from nmi_ced.bridge import SharedKVBridge
from nmi_ced.adapter import save_bridge

class Norm(nn.Module):
    def __init__(self,dim=256,eps=1e-6):
        super().__init__();self.weight=nn.Parameter(torch.zeros(dim));self.eps=eps
    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)*(1+self.weight.float())).to(x.dtype)

p=argparse.ArgumentParser()
p.add_argument('--features',required=True)
p.add_argument('--out',default='artifacts/warmup')
p.add_argument('--steps',type=int,default=100)
p.add_argument('--batch-positions',type=int,default=64)
p.add_argument('--lr',type=float,default=2e-4)
p.add_argument('--seed',type=int,default=17)
p.add_argument('--init',choices=['source36','random_linear'],default='source36',
               help='source36 copies pretrained layer-36 K/V matrices; random_linear reinitializes only matrices; KNorm is pretrained in both')
a=p.parse_args()
torch.manual_seed(a.seed);random.seed(a.seed)
root=Path(a.features)
meta=json.loads((root/'manifest.json').read_text())
files=meta['file_list']
if not files:raise ValueError('empty teacher features')
arrays=[torch.load(root/f,map_location='cpu',weights_only=True) for f in files]
x=torch.cat([c['h32'] for c in arrays]).float()
y=torch.cat([c['teacher_kv'] for c in arrays]).float()
assert x.ndim==2 and x.shape[-1]==5120
assert y.shape==(x.shape[0],8,2048)
init=torch.load(root/'initial_projection.pt',weights_only=True,map_location='cpu')
bridge=SharedKVBridge(5120,4,256,Norm())
if a.init=='source36':
    bridge.k_proj.load_state_dict(init['k_proj'])
    bridge.v_proj.load_state_dict(init['v_proj'])
# Both arms start with the pretrained layer-36 KNorm.
bridge.k_norm.load_state_dict(init['k_norm'])
device='cuda' if torch.cuda.is_available() else 'cpu'
bridge.to(device)
opt=torch.optim.AdamW(bridge.parameters(),lr=a.lr,weight_decay=0.0)
# Normalize target layer contribution: each layer equally weighted in loss.
scale=y.square().mean(dim=(0,2)).clamp_min(1e-5).to(device)
history=[]
for step in range(a.steps):
    ix=torch.randint(0,x.shape[0],(a.batch_positions,))
    xb=x[ix].to(device); yb=y[ix].to(device)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
        pred=bridge.pre_rope(xb[:,None,:])[:,0,:]
    loss=((pred.float()[:,None,:]-yb.float()).square().mean(dim=(0,2))/scale).mean()
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    opt.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.0)
    opt.step();history.append(float(loss.detach().cpu()))
    if step%10==0 or step==a.steps-1:print(json.dumps({'step':step,'loss':history[-1]}),flush=True)
Path(a.out).mkdir(parents=True,exist_ok=True)
save_bridge(bridge,Path(a.out)/'adapter.pt',
            model_id=meta['model_id'],model_revision=meta['revision'],
            initialization=a.init,training_stage='feature_projector_warmup')
(Path(a.out)/'metrics.json').write_text(json.dumps({'steps':a.steps,'positions':x.shape[0],
  'loss_history':history,'initialization':a.init,
  'key_norm_origin':'pretrained_gqa_layer36','status':'feature_projector_warmup_only'},indent=2))
