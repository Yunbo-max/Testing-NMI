#!/usr/bin/env python3
"""Compare measured B0/B1 files; no invented or inconsistent baselines."""
import argparse,json,math
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--shared',required=True)
a=p.parse_args()
b=json.load(open(a.base));s=json.load(open(a.shared))
for key in ['revision','prompt_tokens','decode_steps','model']:
    if b[key]!=s[key]:raise ValueError(f'non-comparable {key}: {b[key]} != {s[key]}')
if b['arm']!='base' or s['arm']!='shared':raise ValueError('wrong arms')
for key in ['gqa_kv_bytes_prefill','peak_total_gib','prefill_seconds','decode_tokens_per_second']:
    print(f'{key}: base={b[key]} shared={s[key]}')
print('GQA KV reduction fraction:',1-s['gqa_kv_bytes_prefill']/b['gqa_kv_bytes_prefill'])
print('TTFT ratio (shared/base):',s['prefill_seconds']/b['prefill_seconds'])
print('decode speed ratio (shared/base):',s['decode_tokens_per_second']/b['decode_tokens_per_second'])
print('NO full-CED performance claim; run repeated trials with warmup before statistical inference.')
