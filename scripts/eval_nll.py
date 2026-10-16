#!/usr/bin/env python3
"""Token-weighted held-out *bounded-document* CE/PPL for B0/B1b on H100.

This evaluates a fixed JSONL source using per-record context truncation and reset.
It is not a native standard-corpus/sliding-window perplexity or a long-context score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def token_nll_sum(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[float, int]:
    """Teacher-forced next-token NLL numerator/denominator; no padded tokens."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError('expected logits [B,T,V], input_ids [B,T]')
    if input_ids.shape[1] < 2:
        return 0.0, 0
    nll = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                          input_ids[:, 1:].reshape(-1), reduction='sum')
    return float(nll.detach().cpu()), input_ids[:, 1:].numel()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['base', 'shared'], required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--model', default='Qwen/Qwen3.8-27B')
    p.add_argument('--adapter', default=None)
    p.add_argument('--data', required=True, help='Never use training split; fixed JSONL text records')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--max-examples', type=int, default=64)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.arm == 'shared' and not a.adapter:
        p.error('--adapter required for shared arm')
    if a.seq_len < 4 or a.max_examples < 1:
        p.error('invalid lengths')
    from nmi_ced.loading import assert_cuda_ready, load_qwen
    from nmi_ced.adapter import install_qwen_shared_kv, load_bridge_checkpoint
    from transformers import AutoTokenizer
    assert_cuda_ready()
    model = load_qwen(model_id=a.model, revision=a.revision).eval()
    model.requires_grad_(False)
    h = None
    if a.arm == 'shared':
        bridge, _, h = install_qwen_shared_kv(model)
        ckpt = load_bridge_checkpoint(bridge, a.adapter, expected_revision=a.revision)
        if ckpt['model_id'] != a.model:
            raise ValueError('model mismatch')
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=False)
    nll_sum, token_count, docs = 0.0, 0, 0
    try:
        with torch.inference_mode(), Path(a.data).open('r', encoding='utf-8') as f:
            for line in f:
                if docs >= a.max_examples:
                    break
                if not line.strip():
                    continue
                text = json.loads(line).get('text')
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('missing text')
                ids = tok(text, return_tensors='pt', truncation=True, max_length=a.seq_len,
                          add_special_tokens=True)['input_ids'].to('cuda')
                if ids.shape[1] < 2:
                    continue
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=0).logits
                local_nll, local_count = token_nll_sum(logits, ids)
                nll_sum += local_nll; token_count += local_count; docs += 1
    finally:
        if h is not None: h.remove()
    if token_count == 0:
        raise RuntimeError('no valid evaluation tokens')
    ce = nll_sum / token_count
    rec = {'status': 'measured', 'metric': 'bounded_document_token_weighted_ce',
           'arm': a.arm, 'model': a.model, 'revision': a.revision,
           'dataset_sha256': sha256_file(Path(a.data)), 'seq_len': a.seq_len,
           'max_examples': a.max_examples, 'evaluated_documents': docs, 'predicted_tokens': token_count,
           'negative_log_likelihood_sum': nll_sum, 'cross_entropy_nats': ce,
           'perplexity': float(torch.exp(torch.tensor(ce, dtype=torch.float64))),
           'limitations': ['each JSONL record separately resets context', 'truncates at seq_len',
                           'not native WikiText PPL or RULER/LongBench-v2', 'no tokenizer chunk packing']}
    dst = Path(a.out); dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rec, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
