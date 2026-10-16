"""Pretrained K/V weight transfer and random-linear control: CPU-only initialization checks."""
import torch
from torch import nn
from nmi_ced.bridge import SharedKVBridge

class NativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj=nn.Linear(16,8,bias=False)
        self.v_proj=nn.Linear(16,8,bias=False)
        self.k_norm=nn.LayerNorm(4)

def test_copied_native_weights_are_equal_not_aliased():
    torch.manual_seed(123)
    native=NativeAttention()
    bridge=SharedKVBridge(16,2,4,native.k_norm)
    bridge.initialize_from(native)
    torch.testing.assert_close(bridge.k_proj.weight,native.k_proj.weight)
    torch.testing.assert_close(bridge.v_proj.weight,native.v_proj.weight)
    torch.testing.assert_close(bridge.k_norm.weight,native.k_norm.weight)
    assert bridge.k_proj.weight.data_ptr()!=native.k_proj.weight.data_ptr()
    with torch.no_grad():
        native.k_proj.weight.add_(1)
    assert not torch.allclose(bridge.k_proj.weight,native.k_proj.weight)

def test_random_control_preserves_native_key_norm():
    torch.manual_seed(42)
    native=NativeAttention()
    copied=SharedKVBridge(16,2,4,native.k_norm)
    copied.initialize_from(native)
    random=SharedKVBridge(16,2,4,native.k_norm)
    torch.testing.assert_close(random.k_norm.weight,copied.k_norm.weight)
    torch.testing.assert_close(random.k_norm.bias,copied.k_norm.bias)
    assert not torch.allclose(random.k_proj.weight,copied.k_proj.weight)
    assert not torch.allclose(random.v_proj.weight,copied.v_proj.weight)
