"""Trainable Qwen-compatible shared global K/V projector.

Output is [B, num_kv_heads, current_seq, head_dim] for both K and V.
No temporal compression (ratio=1): a new processed token adds one record.
This is cross-layer global KV sharing; NOT DeltaNet recurrence or CED state-reconstruction.
"""
from __future__ import annotations

import copy
from typing import Tuple
import torch
from torch import nn


def rotate_key_only(key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply HF Qwen3.5's *partial* rotary transformation to K only.

    key [B,Hkv,S,D], cos/sin [B,S,Drot], Drot <= D and even.
    The same MRoPE cos/sin must come from Qwen's TextRotaryEmbedding.
    """
    if key.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError(f"unexpected key/rope shapes {key.shape}, {cos.shape}, {sin.shape}")
    r = cos.shape[-1]
    if r == 0 or r > key.shape[-1] or r % 2:
        raise ValueError(f"invalid rotary size {r} for head dim {key.shape[-1]}")
    if cos.shape[0] != key.shape[0] or cos.shape[1] != key.shape[2]:
        raise ValueError("rotary positions must match the current sequence")
    part, passthrough = key[..., :r], key[..., r:]
    x1, x2 = part[..., :r//2], part[..., r//2:]
    rotated_half = torch.cat((-x2, x1), dim=-1)
    rotated = part * cos.unsqueeze(1) + rotated_half * sin.unsqueeze(1)
    return torch.cat((rotated, passthrough), dim=-1)


class SharedKVBridge(nn.Module):
    """One K/V projection for eight late GQA readers (unmodified 24 late GDN layers).

    Reader modules have distinct Q projections, gated outputs, output projections;
    the K normalizer is shared and applied once before the RoPE.
    """
    def __init__(self, hidden_size: int, kv_heads: int, head_dim: int, key_norm: nn.Module):
        super().__init__()
        self.hidden_size, self.kv_heads, self.head_dim = hidden_size, kv_heads, head_dim
        width = kv_heads * head_dim
        self.k_proj = nn.Linear(hidden_size, width, bias=False)
        self.v_proj = nn.Linear(hidden_size, width, bias=False)
        self.k_norm = copy.deepcopy(key_norm)
        self.current_k: torch.Tensor | None = None
        self.current_v: torch.Tensor | None = None
        self.key_view: torch.Tensor | None = None
        self.value_view: torch.Tensor | None = None
        self.last_cache_id: int | None = None
        self.producer_id: int | None = None

    def initialize_from(self, original_attention: nn.Module) -> None:
        """Copy a pretrained late GQA layer's K/V projection for a stable starting point.

        This is NOT functionally equivalent: inputs at H32 differ from that layer's hidden states.
        """
        with torch.no_grad():
            self.k_proj.weight.copy_(original_attention.k_proj.weight)
            self.v_proj.weight.copy_(original_attention.v_proj.weight)
            self.k_norm.load_state_dict(original_attention.k_norm.state_dict())

    def pre_rope(self, h32: torch.Tensor) -> torch.Tensor:
        """Return [B,S,2*kv_heads*head_dim] after K norm, for feature distillation."""
        b, s, d = h32.shape
        if d != self.hidden_size:
            raise ValueError(f"boundary width mismatch: {d} != {self.hidden_size}")
        k = self.k_norm(self.k_proj(h32.float()).view(b, s, self.kv_heads, self.head_dim))
        v = self.v_proj(h32.float()).view(b, s, self.kv_heads, self.head_dim)
        return torch.cat((k.reshape(b, s, -1), v.reshape(b, s, -1)), dim=-1)

    def stage(self, h32: torch.Tensor, position_embeddings: Tuple[torch.Tensor,torch.Tensor]) -> None:
        """Called *after* encoder layer 32, each real model forward.

        The keys generated here are written to one producer cache only on its first read.
        """
        features = self.pre_rope(h32).to(dtype=h32.dtype)
        k, v = features.split(self.kv_heads * self.head_dim, dim=-1)
        b, s, _ = k.shape
        k = k.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        self.current_k = rotate_key_only(k, *position_embeddings)
        self.current_v = v
        self.key_view = self.value_view = None
        self.last_cache_id = None

    def read(self, *, layer_idx: int, cache, producer_idx: int):
        """Update exactly one physical K/V cache and reuse its tensor view downstream.

        Supports HF DynamicCache only; no StaticCache, beam reorder optimizations,
        compiled model, distributed sharding, packed multimodal masking in B1-v0.
        """
        if self.current_k is None or self.current_v is None:
            raise RuntimeError("encoder boundary did not stage shared KV; check layer hooks")
        if layer_idx == producer_idx:
            if cache is None:
                self.key_view, self.value_view = self.current_k, self.current_v
            else:
                if not hasattr(cache, "layers") or not hasattr(cache, "update"):
                    raise TypeError("B1 requires HF DynamicCache; custom/StaticCache is not supported")
                self.key_view, self.value_view = cache.update(self.current_k, self.current_v, producer_idx)
                self.last_cache_id = id(cache)
        else:
            if self.key_view is None or self.value_view is None:
                raise RuntimeError("shared cache consumer called before the producer")
            if cache is not None and self.last_cache_id != id(cache):
                raise RuntimeError("consumer cache does not match producer's cache")
        return self.key_view, self.value_view
