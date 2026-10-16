#!/usr/bin/env python3
"""Theoretical Qwen3.8-27B GQA KV only; no claim about actual GPU allocations."""
import argparse
p = argparse.ArgumentParser()
p.add_argument('--context',type=int,default=32768)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--dtype-bytes',type=int,default=2)
a=p.parse_args()
per_group=2*4*256*a.dtype_bytes
for name,groups in [('B0/original',16),('B1a/pair-share',12),('B1b/late-all-share',9)]:
 n=a.batch*a.context*per_group*groups
 print(f'{name}: {groups} unique GQA KV groups; {groups*per_group:,} B/token/batch; {n/2**30:.4f} GiB per batch at {a.context:,} tokens')
print('Excluded: Gated DeltaNet states, activation tensors, model weights, quant scales, allocator fragmentation, temporary expanded KV and runtime metadata.')
