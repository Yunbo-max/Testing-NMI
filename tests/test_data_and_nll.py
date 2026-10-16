"""Pure local correctness checks; never imply a 27B evaluation has run."""
import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_split_is_stable_and_disjoint():
    mod = load('prepare_fineweb')
    digest = mod.text_digest('test input' * 10)
    assert digest == mod.text_digest('test input' * 10)
    assert mod.assign_split(digest) in ('train','val','test')
    assert len({mod.assign_split(mod.text_digest(f'doc {i}')) for i in range(100)}) == 3

def test_token_weighted_nll_uses_next_token_only():
    mod = load('eval_nll')
    ids = torch.tensor([[0, 1, 0]])
    logits = torch.tensor([[[0., 5.], [5., 0.], [0., 5.]]])
    nll, num = mod.token_nll_sum(logits, ids)
    assert num == 2
    assert nll < 0.03

def test_short_rows_zero_denominator():
    mod = load('eval_nll')
    assert mod.token_nll_sum(torch.zeros(1,1,10), torch.tensor([[1]])) == (0.0,0)
