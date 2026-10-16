"""Index sharing semantics only: this is NOT production CSA2 hierarchical indexer."""
import unittest
import torch


def select_indices(indexer_q, indexer_k, visible_len, k=2, pool=None):
    """Select historical positions (1D, one query), never future positions."""
    scores = indexer_k[:visible_len] @ indexer_q
    if pool is not None:
        valid = [i for i in pool if i < visible_len]
        if not valid:
            raise ValueError('empty causal pool')
        cand = torch.tensor(valid, dtype=torch.long)
        return cand[torch.topk(scores[cand], min(k, len(valid))).indices]
    return torch.topk(scores, min(k, visible_len)).indices


def modes(q_full, q_reindex, k, visible_len):
    full_topk = select_indices(q_full, k, visible_len)
    reindex_topk = select_indices(q_reindex, k, visible_len)
    reuse_topk = reindex_topk  # same token, later layer; no index recomputation
    return full_topk, reindex_topk, reuse_topk


class ToyCSA2IndexTests(unittest.TestCase):
    def test_reindex_can_choose_different_positions(self):
        keys = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        full, reindex, reuse = modes(torch.tensor([2., 0.]), torch.tensor([-2., 0.]), keys, 4)
        self.assertFalse(torch.equal(full[:1], reindex[:1]))
        self.assertTrue(torch.equal(reindex, reuse))

    def test_causal_pool_prevents_future_selection(self):
        keys = torch.randn(9, 2)
        idx = select_indices(torch.randn(2), keys, 5, k=3, pool=list(range(9)))
        self.assertTrue((idx < 5).all().item())


if __name__ == '__main__':
    unittest.main()
