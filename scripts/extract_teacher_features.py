#!/usr/bin/env python3
"""Extract *real* pre-RoPE late-GQA K/V targets and encoder H32 from pinned Qwen.

One H100; a small bounded set of train text. Saves .pt files locally only.
Do not publish raw training text/features or checkpoints to GitHub.
"""
import argparse, json, os
from pathlib import Path
import torch
from nmi_ced.loading import assert_cuda_ready, load_qwen
from nmi_ced.adapter import get_language_layers, AdapterSpec

p=argparse.ArgumentParser()
p.add_argument('--model',default='Qwen/Qwen3.8-27B')
p.add_argument('--revision',required=True)
p.add_argument('--data',required=True,help='local .jsonl with {"text":...} records')
p.add_argument('--out',default='artifacts/teacher_features')
p.add_argument('--max-examples',type=int,default=16)
p.add_argument('--seq-len',type=int,default=128)
a=p.parse_args()
assert_cuda_ready()
from transformers import AutoTokenizer
model=load_qwen(model_id=a.model,revision=a.revision).eval()
model.requires_grad_(False)
tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision,trust_remote_code=False)
layers=get_language_layers(model)
spec=AdapterSpec()
assert len(layers)==64
state={'boundary':None,'readers':{}}

def boundary_capture(module,args,kwargs,output):
    state['boundary']=(output[0] if isinstance(output,tuple) else output).detach()

def reader_input_capture(idx):
    def hook(module,args,kwargs):
        x=kwargs.get('hidden_states')
        if x is None: x=args[0]
        b,s,_=x.shape
        with torch.no_grad():
            attn=module
            k=attn.k_norm(attn.k_proj(x).reshape(b,s,-1,attn.head_dim)).reshape(b,s,-1)
            v=attn.v_proj(x).reshape(b,s,-1)
            state['readers'][idx]=torch.cat((k,v),dim=-1).detach()
    return hook
handles=[layers[spec.boundary_idx].register_forward_hook(boundary_capture,with_kwargs=True)]
for idx in spec.reader_indices:
    handles.append(layers[idx].self_attn.register_forward_pre_hook(reader_input_capture(idx),with_kwargs=True))
Path(a.out).mkdir(parents=True,exist_ok=True)
meta={'format':'qwen38_teacher_features_b1_v0','model_id':a.model,'revision':a.revision,
      'boundary_idx':spec.boundary_idx,'reader_indices':list(spec.reader_indices),
      'examples':0,'total_positions':0,'file_list':[]}
try:
    with open(a.data,encoding='utf-8') as f:
        for n,line in enumerate(f):
            if n>=a.max_examples:break
            line=line.strip()
            if not line:continue
            item=json.loads(line)
            text=item['text']; assert isinstance(text,str) and text.strip()
            encoded=tokenizer(text,return_tensors='pt',truncation=True,max_length=a.seq_len,add_special_tokens=True)
            tokens=encoded['input_ids'].to('cuda')
            state['boundary']=None;state['readers'].clear()
            with torch.inference_mode():
                model(input_ids=tokens,use_cache=False,logits_to_keep=1)
            if state['boundary'] is None or set(state['readers'])!=set(spec.reader_indices):
                raise RuntimeError('missing H32 or teacher attention hooks')
            h=state['boundary'].reshape(-1,state['boundary'].shape[-1]).to('cpu',dtype=torch.bfloat16)
            t=torch.stack([state['readers'][idx] for idx in spec.reader_indices],dim=2).reshape(-1,8,2048).to('cpu',dtype=torch.bfloat16)
            if h.shape[0]!=t.shape[0]:raise RuntimeError('feature/token alignment mismatch')
            dst=f'{a.out}/part-{n:05d}.pt'
            torch.save({'h32':h,'teacher_kv':t},dst)
            meta['examples']+=1;meta['total_positions']+=h.shape[0];meta['file_list'].append(os.path.basename(dst))
            print(json.dumps({'example':n,'positions':h.shape[0],'total':meta['total_positions']}),flush=True)
            state['boundary']=None;state['readers'].clear()
finally:
    for h in handles:h.remove()
# Save starting native projection once; remain local-only.
source=layers[spec.producer_idx].self_attn
initial={'k_proj':source.k_proj.state_dict(),'v_proj':source.v_proj.state_dict(),
         'k_norm':source.k_norm.state_dict()}
torch.save({k:{kk:vv.detach().cpu() for kk,vv in d.items()} for k,d in initial.items()},f'{a.out}/initial_projection.pt')
with open(f'{a.out}/manifest.json','w') as f:json.dump(meta,f,indent=2)
print('Wrote',meta['total_positions'],'real training-token features to',a.out)
