"""Qwen3.8-27B shared-global-KV *adaptation* research code.

B1 is not DeepSeek CED's decoder-prefill skip and not CSA2 sparse attention.
"""
from .bridge import SharedKVBridge, rotate_key_only
from .adapter import install_qwen_shared_kv, save_bridge, load_bridge_checkpoint

__all__ = ["SharedKVBridge", "rotate_key_only", "install_qwen_shared_kv", "save_bridge", "load_bridge_checkpoint"]
