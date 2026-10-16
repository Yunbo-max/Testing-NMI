"""Actual monkey-patch API smoke against a tiny *mock* Qwen layer structure.

This validates the adapter's logic but does NOT replace a real Qwen3.8-27B run.
"""
import copy
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from nmi_ced.adapter import install_qwen_shared_kv, make_attention_mask, get_language_layers
from nmi_ced.bridge import rotate_key_only, SharedKVBridge


class QwenLikeNorm(nn.Module):
    def __init__(self, d):
        super().__init__();self.weight=nn.Parameter(torch.zeros(d));self.eps=1e-6
    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps) *
                (1+self.weight.float())).to(x.dtype)

class FakeDynamicCache:
    def __init__(self,n=64):
        self.layers=[type('Slot',(),{'keys':None,'values':None})() for _ in range(n)]
        self.calls={}
    def update(self,key,value,index):
        slot=self.layers[index]
        slot.keys=key if slot.keys is None else torch.cat((slot.keys,key),dim=-2)
        slot.values=value if slot.values is None else torch.cat((slot.values,value),dim=-2)
        self.calls[index]=self.calls.get(index,0)+1
        return slot.keys, slot.values

class FakeNativeAttention(nn.Module):
    def __init__(self,index,dim=16,nh=4,nkv=2,hd=4):
        super().__init__();self.layer_idx=index;self.head_dim=hd;self.scaling=hd**-0.5
        self.num_key_value_groups=nh//nkv;self.attention_dropout=0.;self.is_causal=True
        self.q_proj=nn.Linear(dim,nh*hd*2,bias=False)
        self.k_proj=nn.Linear(dim,nkv*hd,bias=False)
        self.v_proj=nn.Linear(dim,nkv*hd,bias=False)
        self.o_proj=nn.Linear(nh*hd,dim,bias=False)
        self.q_norm=QwenLikeNorm(hd);self.k_norm=QwenLikeNorm(hd)
    def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
        b,s,d=hidden_states.shape;nh=4
        q,g=self.q_proj(hidden_states).reshape(b,s,nh,8).chunk(2,dim=-1)
        q=rotate_key_only(self.q_norm(q).transpose(1,2),*position_embeddings)
        k=self.k_norm(self.k_proj(hidden_states).reshape(b,s,2,4)).transpose(1,2)
        k=rotate_key_only(k,*position_embeddings)
        v=self.v_proj(hidden_states).reshape(b,s,2,4).transpose(1,2)
        if past_key_values is not None:k,v=past_key_values.update(k,v,self.layer_idx)
        mask,causal=make_attention_mask(attention_mask,query=q,keys=k)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,enable_gqa=True)
        y=y.transpose(1,2).reshape(b,s,-1)*torch.sigmoid(g.reshape(b,s,-1))
        return self.o_proj(y),None

class FakeLayer(nn.Module):
    def __init__(self,index):
        super().__init__();self.block_type='full_attention' if (index+1)%4==0 else 'linear_attention'
        self.self_attn=FakeNativeAttention(index) if self.block_type=='full_attention' else None
        self.norm=nn.LayerNorm(16)
        self.ff=nn.Linear(16,16)
    def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
        if self.block_type=='full_attention':
            attn,_=self.self_attn(hidden_states=self.norm(hidden_states),position_embeddings=position_embeddings,
                                   attention_mask=attention_mask,past_key_values=past_key_values)
            hidden_states=hidden_states+attn
        return hidden_states+self.ff(self.norm(hidden_states))

class MockCore(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([FakeLayer(i) for i in range(64)])
    def forward(self,x,cache=None,offset=0):
        s=x.shape[1]
        positions=torch.arange(offset,offset+s,device=x.device).float()
        theta=positions/10.0
        # 2D partial RoPE only; rotate half swap for two dims
        cos=torch.stack([theta.cos(),theta.cos()],dim=-1).unsqueeze(0).expand(x.shape[0],-1,-1)
        sin=torch.stack([theta.sin(),theta.sin()],dim=-1).unsqueeze(0).expand(x.shape[0],-1,-1)
        for layer in self.layers:
            x=layer(x,position_embeddings=(cos,sin),attention_mask=None,past_key_values=cache)
        return x

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__();self.model=MockCore()
        class Config: hidden_size=16;head_dim=4;num_attention_heads=4;num_key_value_heads=2
        self.config=Config()
    def forward(self,x,cache=None,offset=0):return self.model(x,cache=cache,offset=offset)


def test_adapter_full_sequence_matches_incremental_and_one_cache():
    torch.manual_seed(999)
    base=FakeModel().eval()
    new=copy.deepcopy(base).eval()
    new.requires_grad_(False)
    bridge,spec,handle=install_qwen_shared_kv(new)
    x=torch.randn(1,5,16)
    full=new(x)
    cache=FakeDynamicCache()
    out=torch.cat([new(x[:,i:i+1,:],cache=cache,offset=i) for i in range(5)],dim=1)
    torch.testing.assert_close(full,out,atol=1e-4,rtol=1e-4)
    # 8 readers share 1 physical cache; 8 early GQA keep their own.
    assert tuple(sorted(cache.calls))==(3,7,11,15,19,23,27,31,35)
    assert cache.calls[35]==5
    assert all(cache.layers[idx].keys is None for idx in [39,43,47,51,55,59,63])
    # Adapter changes the model, not functionally identical to base.
    assert not torch.allclose(full,base(x),atol=1e-6)
    handle.remove()


def test_causal_future_token_cannot_change_past():
    torch.manual_seed(13)
    m=FakeModel().eval()
    bridge,spec,handle=install_qwen_shared_kv(m)
    x=torch.randn(2,7,16);other=x.clone();other[:,4:,:]+=3
    torch.testing.assert_close(m(x)[:,:4],m(other)[:,:4],rtol=1e-5,atol=1e-5)
    handle.remove()


def test_only_bridge_is_trainable_and_has_gradients():
    m=FakeModel().eval();m.requires_grad_(False)
    bridge,_,handle=install_qwen_shared_kv(m)
    assert bridge.k_proj.weight.dtype==torch.float32
    out=m(torch.randn(1,4,16)).square().mean();out.backward()
    assert torch.isfinite(bridge.k_proj.weight.grad).all()
    assert bridge.k_proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for name,p in m.named_parameters() if not name.startswith('nmi_shared_kv_bridge.'))
    handle.remove()


def test_mask_chunked_decode_absolute_positions():
    query=torch.randn(2,4,2,8);keys=torch.randn(2,2,5,8)
    mask,causal=make_attention_mask(None,query=query,keys=keys)
    assert not causal
    assert mask.shape==(1,1,2,5)
    assert mask[0,0,0,-1].item()==False
    assert mask[0,0,1,-1].item()==True
    with pytest.raises(ValueError):make_attention_mask(torch.zeros(2,2,2,4),query=query,keys=keys)


def test_rope_matches_half_rotation_by_hand():
    k=torch.tensor([[[[1.,2.,3.,4.]]]])
    cos=torch.tensor([[[0.,0.]]]);sin=torch.tensor([[[1.,1.]]])
    out=rotate_key_only(k,cos,sin)
    torch.testing.assert_close(out,torch.tensor([[[[-2.,1.,3.,4.]]]]))
