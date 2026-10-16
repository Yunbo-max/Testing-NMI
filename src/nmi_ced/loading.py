"""Pinned-model loading and H100 safety checks. No silent disk/CPU offload."""
from __future__ import annotations
import json
import torch

DEFAULT_MODEL="Qwen/Qwen3.8-27B"


def assert_cuda_ready(min_free_gib=55.0):
    if not torch.cuda.is_available():
        raise RuntimeError("H100 execution requires CUDA; local CPU only supports smoke/unit tests")
    index=torch.cuda.current_device()
    free,total=torch.cuda.mem_get_info(index)
    capability=torch.cuda.get_device_capability(index)
    device=torch.cuda.get_device_name(index)
    print(json.dumps({"device":device,"compute_capability":capability,
                      "free_gib":round(free/2**30,2),"total_gib":round(total/2**30,2)}),flush=True)
    if free<min_free_gib*2**30:
        raise MemoryError(f"free VRAM {free/2**30:.1f} GiB < required preflight {min_free_gib} GiB")
    return device


def load_qwen(*,model_id=DEFAULT_MODEL, revision, dtype=torch.bfloat16):
    if not revision:
        raise ValueError("Pass a pinned --revision SHA, never train against floating main")
    from transformers import Qwen3_5ForConditionalGeneration
    model=Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=dtype, device_map={"": "cuda:0"},
        low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=False)
    device_map=getattr(model,"hf_device_map",{}) or {}
    if any(str(v).startswith(("cpu","disk")) for v in device_map.values()):
        raise RuntimeError(f"model unexpectedly offloaded from H100: {device_map}")
    if model.config.model_type!="qwen3_5":
        raise ValueError(f"expected qwen3_5, got {model.config.model_type}")
    return model


def get_revision(model_id=DEFAULT_MODEL):
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id).sha
