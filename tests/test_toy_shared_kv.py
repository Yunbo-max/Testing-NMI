"""Toy causal cross-layer KV semantics, not the DeepSeek or Qwen production model.

This is a true forward-vs-incremental test of a shared table read by multiple
sequential decoder layers. It lacks RoPE, SWA, GDN, sparse indexing, and kernels.
"""
import unittest
import torch
from torch import nn
import torch.nn.functional as F


class TinyDecoderLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, k, v, mask=None):
        q = self.query(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        return self.norm(h + self.ffn(h + self.out(weights @ v)))


class TinyCausalCED(nn.Module):
    def __init__(self, dim=12, layers=3):
        super().__init__()
        self.dim = dim
        self.encoder_in = nn.Linear(dim, dim)
        self.encoder_rec = nn.Linear(dim, dim, bias=False)
        self.global_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.decoder = nn.ModuleList([TinyDecoderLayer(dim) for _ in range(layers)])

    def encode(self, x):
        # Recurrence is causal; not a faithful transformer Encoder implementation.
        h_prev = torch.zeros_like(x[:, 0, :])
        outs = []
        for t in range(x.shape[1]):
            h_prev = torch.tanh(self.encoder_in(x[:, t, :]) + self.encoder_rec(h_prev))
            outs.append(h_prev.unsqueeze(1))
        return torch.cat(outs, dim=1)

    def full(self, x):
        h_enc = self.encode(x)
        k, v = self.global_kv(h_enc).chunk(2, dim=-1)
        mask = torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool).triu(1)
        h = h_enc
        for layer in self.decoder:
            h = layer(h, k, v, mask)
        return h

    def incremental(self, x):
        # Causal encoder computed first in this small diagnostic, then global KV
        # records are appended ONE POSITION AT A TIME. Decoder layers share them.
        enc = self.encode(x)
        k_cache, v_cache, outs = [], [], []
        for t in range(x.shape[1]):
            k, v = self.global_kv(enc[:, t:t + 1, :]).chunk(2, dim=-1)
            k_cache.append(k)
            v_cache.append(v)
            k_hist = torch.cat(k_cache, dim=1)
            v_hist = torch.cat(v_cache, dim=1)
            current = enc[:, t:t + 1, :]
            for layer in self.decoder:
                current = layer(current, k_hist, v_hist)
            outs.append(current)
        return torch.cat(outs, dim=1)


class SharedKVTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(111)
        self.model = TinyCausalCED().eval()
        self.x = torch.randn(2, 7, 12)

    def test_full_matches_incremental(self):
        torch.testing.assert_close(self.model.full(self.x), self.model.incremental(self.x), rtol=2e-5, atol=2e-5)

    def test_no_future_leakage(self):
        changed = self.x.clone()
        changed[:, 5:, :] += 97
        torch.testing.assert_close(self.model.full(self.x)[:, :5], self.model.full(changed)[:, :5], rtol=2e-5, atol=2e-5)

    def test_global_projection_receives_gradient(self):
        loss = self.model.full(self.x).square().sum()
        loss.backward()
        grad = self.model.global_kv.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.abs().sum().item(), 1e-9)

    def test_two_different_decoder_layers_can_read_same_memory(self):
        h_enc = self.model.encode(self.x)
        k, v = self.model.global_kv(h_enc).chunk(2, dim=-1)
        first, second = self.model.decoder[:2]
        mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
        o1 = first(h_enc, k, v, mask)
        o2 = second(o1, k, v, mask)
        self.assertEqual(o2.shape, self.x.shape)
        self.assertFalse(torch.allclose(o1, o2))


if __name__ == '__main__':
    unittest.main()
