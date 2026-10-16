#!/usr/bin/env python3
"""Measure B0 vs B1 separately on a single H100, WITHOUT model coexistence.

Run twice with --arm base, shared, identical prompt/revision/settings.
The B1 variant uses dense shared KV, NOT CSA2 sparsity or CED prefill skip.
"""
import argparse, json, time, os
from pathlib import Path
import torch
from nmi_ced.loading import assert_cuda_ready, load_qwen
from nmi_ced.adapter import install_qwen_shared_kv, load_bridge_checkpoint

p=argparse.ArgumentParser()
p.add_argument('--arm',choices=['base','shared'],required=True)
p.add_argument('--model',default='Qwen/Qwen3.8-27B')
p.add_argument('--revision',required=True)
p.add_argument('--adapter',default=None)
p.add_argument('--prompt-file',required=True)
p.add_argument('--max-context',type=int,default=512)
p.add_argument('--decode-steps',type=int,default=16)
p.add_argument('--out',default='results/runs/benchmark.json')
a=p.parse_args()
if a.arm=='shared' and not a.adapter:raise ValueError('--adapter is required for shared arm')
if a.decode_steps<1:raise ValueError('decode-steps must be positive')
assert_cuda_ready()
from transformers import AutoTokenizer,DynamicCache
model=load_qwen(model_id=a.model,revision=a.revision).eval()
model.requires_grad_(False)
handle=None
if a.arm=='shared':
    bridge,spec,handle=install_qwen_shared_kv(model)
    saved=load_bridge_checkpoint(bridge,a.adapter,expected_revision=a.revision)
    if saved['model_id']!=a.model:raise ValueError('model mismatch')
tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision,trust_remote_code=False)
prompt=Path(a.prompt_file).read_text(encoding='utf-8')
ids=tok(prompt,return_tensors='pt',max_length=a.max_context,truncation=True)['input_ids'].to('cuda')
seq=ids.shape[1]
cache=DynamicCache(config=model.config)

def sync():torch.cuda.synchronize()

def cache_bytes(cache):
    # For GQA tensors only, excludes GDN conv/recurrent state & allocator fragmentation.
    total=0;groups=[]
    for idx,layer in enumerate(cache.layers):
        keys=getattr(layer,'keys',None);vals=getattr(layer,'values',None)
        if keys is None or vals is None or not torch.is_tensor(keys) or keys.numel()==0:continue
        size=(keys.numel()*keys.element_size()+vals.numel()*vals.element_size())
        groups.append({'layer_one_based':idx+1,'bytes':size,'positions':keys.shape[-2]})
        total+=size
    return total,groups

with torch.inference_mode():
    torch.cuda.reset_peak_memory_stats()
    sync();start=time.perf_counter()
    result=model(input_ids=ids,use_cache=True,past_key_values=cache,logits_to_keep=1)
    sync();ttft=time.perf_counter()-start
    peak_prefill=torch.cuda.max_memory_allocated()
    bytes_prefill,groups=cache_bytes(cache)
    token=result.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    sync();start=time.perf_counter()
    for _ in range(a.decode_steps):
        result=model(input_ids=token,use_cache=True,past_key_values=cache,logits_to_keep=1)
        token=result.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    sync();decode_time=time.perf_counter()-start
    bytes_end,groups_end=cache_bytes(cache)
    peak_total=torch.cuda.max_memory_allocated()
rec={'arm':a.arm,'model':a.model,'revision':a.revision,'status':'measured',
 'prompt_tokens':seq,'decode_steps':a.decode_steps,
 'prefill_seconds':ttft,'decode_seconds':decode_time,'decode_tokens_per_second':a.decode_steps/decode_time,
 'gqa_kv_bytes_prefill':bytes_prefill,'gqa_cache_layers':[g['layer_one_based'] for g in groups],
 'gqa_kv_bytes_end':bytes_end,'peak_prefill_gib':peak_prefill/2**30,'peak_total_gib':peak_total/2**30,
 'limitations':['GDN state bytes excluded from GQA byte figure','No full CED prefill skipping',
                'No CSA2 sparse indexing','SDPA reference not optimized inference kernel']}
if a.arm=='shared' and len(groups)!=9: raise RuntimeError(f'expected 9 physically saved GQA K/V layer groups, got {groups}')
if a.arm=='base' and len(groups)!=16: raise RuntimeError(f'expected 16 physically saved GQA K/V layer groups, got {groups}')
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if handle is not None:handle.remove()
