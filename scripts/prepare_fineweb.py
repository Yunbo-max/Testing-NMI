#!/usr/bin/env python3
"""Deterministic document-disjoint FineWeb-Edu pilot files for Qwen B0T/B1.

Does not execute on import. The HF source/model revision must be pinned by the operator.
Splits are *within FineWeb-Edu's train subset*, not official held-out splits.
The generated JSONL is private host input, NOT a public benchmark or model eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def assign_split(digest: str) -> str:
    """Stable 80/10/10 document-level hash split with no RNG state dependency."""
    bucket = int(digest[:12], 16) % 10
    return 'train' if bucket < 8 else ('val' if bucket == 8 else 'test')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-revision', required=True, help='Immutable HF dataset commit SHA')
    p.add_argument('--model-revision', required=True, help='Immutable Qwen model/tokenizer SHA')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--config', default='sample-10BT')
    p.add_argument('--out', default='artifacts/data/fineweb_edu_pilot')
    p.add_argument('--train-tokens', type=int, default=120000)
    p.add_argument('--val-tokens', type=int, default=20000)
    p.add_argument('--test-tokens', type=int, default=20000)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--min-tokens', type=int, default=16)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--max-scanned', type=int, default=200000)
    a = p.parse_args()
    if any(v <= 0 for v in (a.train_tokens, a.val_tokens, a.test_tokens, a.seq_len, a.max_scanned)):
        p.error('token quotas, sequence length and max-scanned must be positive')
    if a.min_tokens < 3 or a.min_tokens > a.seq_len:
        p.error('min-tokens must be between 3 and seq-len')
    if len(a.dataset_revision) < 30 or len(a.model_revision) < 30:
        p.error('pin immutable full revision SHAs, not branch names')
    out = Path(a.out)
    if any((out / f'{name}.jsonl').exists() for name in ('train', 'val', 'test')):
        p.error('refusing to overwrite prior split files; use a new output directory')

    from datasets import load_dataset, __version__ as datasets_version
    from transformers import AutoTokenizer, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.8-27B', revision=a.model_revision, trust_remote_code=False)
    ds = load_dataset(a.dataset, name=a.config, split='train', streaming=True, revision=a.dataset_revision)
    ds = ds.shuffle(seed=a.seed, buffer_size=10000)
    out.mkdir(parents=True, exist_ok=True)
    files = {name: (out / f'{name}.jsonl').open('w', encoding='utf-8') for name in ('train', 'val', 'test')}
    budgets = {'train': a.train_tokens, 'val': a.val_tokens, 'test': a.test_tokens}
    counts = {name: {'documents': 0, 'bounded_tokens': 0} for name in budgets}
    seen = set()
    scanned = 0
    try:
        for row in ds:
            if scanned >= a.max_scanned:
                break
            scanned += 1
            raw = row.get('text')
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if len(text) < 64:
                continue
            digest = text_digest(text)
            if digest in seen:
                continue
            seen.add(digest)
            split = assign_split(digest)
            if counts[split]['bounded_tokens'] >= budgets[split]:
                if all(counts[k]['bounded_tokens'] >= budgets[k] for k in budgets):
                    break
                continue
            ids = tokenizer(text, truncation=True, max_length=a.seq_len, add_special_tokens=True)['input_ids']
            length = len(ids)
            if length < a.min_tokens:
                continue
            # The current train_27b.py truncates every JSONL document at seq-len;
            # budget tokens are thus bounded per document, not full-document tokens.
            record = {'text': text, 'source': a.dataset, 'sha256_text': digest}
            files[split].write(json.dumps(record, ensure_ascii=False) + '\n')
            counts[split]['documents'] += 1
            counts[split]['bounded_tokens'] += length
            if all(counts[k]['bounded_tokens'] >= budgets[k] for k in budgets):
                break
    finally:
        for f in files.values():
            f.close()

    reached = all(counts[k]['bounded_tokens'] >= budgets[k] for k in budgets)
    manifest = {
        'status': 'complete' if reached else 'incomplete',
        'source_id': a.dataset, 'source_config': a.config, 'source_split': 'train',
        'source_revision': a.dataset_revision,
        'tokenizer_model': 'Qwen/Qwen3.8-27B', 'tokenizer_revision': a.model_revision,
        'dataset_library': datasets_version, 'transformers_library': transformers_version,
        'seed': a.seed, 'buffer_size': 10000, 'seq_len': a.seq_len, 'min_tokens': a.min_tokens,
        'split_rule': 'SHA256(normalized text) first 48 bits mod 10: 0..7 train, 8 val, 9 test',
        'scanned_documents': scanned, 'unique_texts_seen': len(seen), 'quotas': budgets,
        'output_counts': counts, 'file_sha256': {name: sha256_file(out / f'{name}.jsonl') for name in budgets},
        'limitations': ['Within-source document-disjoint split; not an external held-out benchmark',
                        'Exact-text dedup only; near-duplicates and unknown base-pretraining overlap remain',
                        'Token quotas are per-document truncated input lengths; actual training exposure depends on steps/packing']
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not reached:
        raise RuntimeError('data quotas not reached; incomplete manifest retained; rerun with NEW output directory')


if __name__ == '__main__':
    main()
