"""Qwen3.5/3.8 native layer surgery for B1 shared KV (no decoder prefill skipping).

The real HF source is pinned in docs/PROVENANCE.md. This is intentionally a
research adapter, NOT a drop-in compatibility guarantee for all HF releases.
"""
from __future__ import annotations

import json
from pathlib import Path
import weakref
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .bridge import SharedKVBridge, rotate_key_only


@dataclass(frozen=True)
class AdapterSpec:
    boundary_idx: int = 31    # zero-based, output of 1-based layer 32
    producer_idx: int = 35    # zero-based, 1-based layer 36
    reader_indices: tuple[int, ...] = (35, 39, 43, 47, 51, 55, 59, 63)


def get_language_layers(model) -> nn.ModuleList:
    """Unambiguous path for multimodal Qwen3.8 or text-only Qwen3.5 models."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise TypeError("expected a HF Qwen3.5 language-model wrapper")
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    layers = getattr(inner, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Qwen layers not found at model.model[.language_model].layers")
    return layers


def supported_structure(model, spec: AdapterSpec) -> tuple[nn.ModuleList, object]:
    layers = get_language_layers(model)
    config = (getattr(getattr(layers[3], "self_attn", None), "config", None)
              or getattr(getattr(model, "config", None), "text_config", None)
              or getattr(model, "config", None))
    if config is None:
        raise TypeError("unable to identify Qwen text config")
    if len(layers) != 64:
        raise ValueError(f"B1 project pinned to 64 layers, got {len(layers)}")
    assert spec.boundary_idx + 1 == 32
    expected = tuple(i for i, layer in enumerate(layers) if getattr(layer, "block_type", None) == "full_attention")
    if expected != tuple(range(3,64,4)):
        raise ValueError(f"unexpected attention layout: {expected}")
    for idx in spec.reader_indices:
        attn=layers[idx].self_attn
        for name in ("q_proj","q_norm","k_proj","v_proj","k_norm","o_proj"):
            if not hasattr(attn,name):
                raise ValueError(f"layer {idx} missing native attention member {name}")
    if len(spec.reader_indices)!=8 or spec.reader_indices[0]!=spec.producer_idx:
        raise ValueError("B1 v0 supports exactly 8 late GQA readers, first as producer")
    return layers,config


def make_attention_mask(attention_mask, *, query, keys):
    """Return attn_mask and is_causal for torch SDPA in full and cached paths.

    Normalizes native 4D additive mask; prevents future leakage when mask=None.
    Long chunk-decoding (q>1, keys>q) uses explicit causal mask; potentially O(q*k).
    """
    b,h,q,d = query.shape
    k = keys.shape[-2]
    if k < q:
        raise ValueError("global history shorter than current query chunk")
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            if attention_mask.shape[-2] not in (1,q) or attention_mask.shape[-1] != k:
                raise ValueError(f"native attention mask incompatible with shared KV: {attention_mask.shape} vs {q},{k}")
            return attention_mask,False
        if attention_mask.ndim == 2:
            if attention_mask.shape != (b,k):
                raise ValueError("2D attention mask must match [batch,KV length]")
            valid = attention_mask.to(torch.bool)[:,None,None,:]
            causal = torch.arange(k,device=query.device)[None,:] <= (k-q+torch.arange(q,device=query.device)[:,None])
            return valid & causal[None,None,:,:], False
        raise ValueError(f"unsupported mask dimensions: {attention_mask.ndim}")
    if k == q:
        return None,True
    if q == 1:
        return None,False
    causal = torch.arange(k,device=query.device)[None,:] <= (k-q+torch.arange(q,device=query.device)[:,None])
    return causal[None,None,:,:],False


class SharedQwenAttention(nn.Module):
    """Preserve native Q projection/gate/norm/output, replace just global K,V source.

    B1 retains DENSE attention on shared global memory. CSA2 sparse indexing not here.
    """
    def __init__(self, native_attn: nn.Module, bridge: SharedKVBridge, spec: AdapterSpec):
        super().__init__()
        self.original = native_attn  # frozen parameters + exact q/o interfaces
        self._bridge = weakref.ref(bridge)  # one registered projector, not eight copies
        self.spec = spec
        self.layer_idx = native_attn.layer_idx
        self.head_dim = native_attn.head_dim
        self.num_key_value_groups = native_attn.num_key_value_groups
        self.scaling = native_attn.scaling
        self.attention_dropout = native_attn.attention_dropout
        self.is_causal = True

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        bridge = self._bridge()
        if bridge is None:
            raise RuntimeError("shared projector has been destroyed")
        b,s,_ = hidden_states.shape
        nh=self.original.q_proj.out_features // (2*self.head_dim)
        q, gate = torch.chunk(self.original.q_proj(hidden_states).reshape(b,s,nh,2*self.head_dim), 2, dim=-1)
        gate = gate.reshape(b,s,nh*self.head_dim)
        q = self.original.q_norm(q).transpose(1,2)
        # Qwen partial RoPE, with the same position embeddings as original attention.
        q = rotate_key_only(q,*position_embeddings)
        k,v = bridge.read(layer_idx=self.layer_idx, cache=past_key_values, producer_idx=self.spec.producer_idx)
        attn_mask, is_causal = make_attention_mask(attention_mask,query=q,keys=k)
        p = self.attention_dropout if self.training else 0.0
        try:
            output = F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,dropout_p=p,
                                                    is_causal=is_causal,scale=self.scaling,enable_gqa=True)
        except TypeError as e:
            # Older kernels without enable_gqa; copies are ephemeral, NEVER persisted.
            if "enable_gqa" not in str(e): raise
            k = k.repeat_interleave(self.num_key_value_groups,dim=1)
            v = v.repeat_interleave(self.num_key_value_groups,dim=1)
            output = F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,dropout_p=p,
                                                     is_causal=is_causal,scale=self.scaling)
        output = output.transpose(1,2).contiguous().reshape(b,s,-1)
        output = self.original.o_proj(output * torch.sigmoid(gate))
        return output,None


def install_qwen_shared_kv(model, *, initialize_from_teacher=True, spec:AdapterSpec|None=None):
    """Patch loaded Qwen3.8-27B model IN PLACE. Do this once, after freezing base.

    Returns (bridge, spec, hook_handle). Keep handle alive. Avoid model.gradient_checkpointing_enable()
    in B1-v0: current boundary hooks have not been qualified for recomputation.
    """
    spec = spec or AdapterSpec()
    if hasattr(model,"nmi_shared_kv_bridge"):
        raise RuntimeError("shared KV adapter already installed")
    layers, cfg = supported_structure(model,spec)
    original_attn = layers[spec.producer_idx].self_attn
    bridge = SharedKVBridge(hidden_size=cfg.hidden_size,
                            kv_heads=cfg.num_key_value_heads,head_dim=cfg.head_dim,
                            key_norm=original_attn.k_norm)
    bridge.to(device=original_attn.k_proj.weight.device,dtype=torch.float32)
    if initialize_from_teacher:
        bridge.initialize_from(original_attn)
    model.add_module("nmi_shared_kv_bridge",bridge)
    for idx in spec.reader_indices:
        layers[idx].self_attn=SharedQwenAttention(layers[idx].self_attn,bridge,spec)

    def boundary_capture(module, args, kwargs, output):
        position_embeddings=kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError("native Qwen decoder layer did not pass position_embeddings")
        hidden=output[0] if isinstance(output,tuple) else output
        bridge.stage(hidden,position_embeddings)

    hook=layers[spec.boundary_idx].register_forward_hook(boundary_capture,with_kwargs=True)
    return bridge,spec,hook


def save_bridge(bridge:SharedKVBridge, path, *, model_id, model_revision, spec:AdapterSpec|None=None):
    """Save only trainable adapter tensors; never save giant base weights to GitHub."""
    spec=spec or AdapterSpec()
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"format":"qwen38_shared_kv_b1_v0", "model_id":model_id,"revision":model_revision,
                "boundary_idx":spec.boundary_idx,"producer_idx":spec.producer_idx,
                "reader_indices":list(spec.reader_indices),"state_dict":bridge.state_dict()},target)


def load_bridge_checkpoint(bridge, path, *, expected_revision=None):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint.get("format")!="qwen38_shared_kv_b1_v0":
        raise ValueError("not a B1 shared-KV adapter checkpoint")
    if expected_revision and checkpoint.get("revision")!=expected_revision:
        raise ValueError("base model revision differs from adapter checkpoint")
    bridge.load_state_dict(checkpoint["state_dict"],strict=True)
    return checkpoint
