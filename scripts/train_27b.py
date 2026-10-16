#!/usr/bin/env python3
"""First genuine 27B B1 adaptation train: frozen base, shared K/V projector only.

No CED prefill skip, no CSA2 sparse kernel, no full-model finetuning.
Do NOT use GitHub Actions; run on a real H100 host and persist local artifacts.
"""
import argparse, json, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from nmi_ced.loading import assert_cuda_ready, load_qwen
from nmi_ced.adapter import install_qwen_shared_kv, save_bridge, load_bridge_checkpoint

p=argparse.ArgumentParser()
p.add_argument('--model',default='Qwen/Qwen3.8-27B')
p.add_argument('--revision',required=True)
p.add_argument('--train-data',required=True,help='local JSONL, text field; not evaluation data')
p.add_argument('--val-data',required=True,help='separate JSONL; frozen before training')
p.add_argument('--warmup-checkpoint',default=None,help='optional offline projector warmup checkpoint')
p.add_argument('--out',default='artifacts/train_b1')
p.add_argument('--seq-len',type=int,default=128)
p.add_argument('--steps',type=int,default=16)
p.add_argument('--grad-accum',type=int,default=1)
p.add_argument('--lr',type=float,default=2e-5)
p.add_argument('--seed',type=int,default=1701)
p.add_argument('--projector-init',choices=['source36','random_linear'],default='source36',
               help='projector K/V matrices from pretrained layer 36 or random; original KNorm is copied in both')
p.add_argument('--save-every',type=int,default=8)
p.add_argument('--eval-examples',type=int,default=4)
a=p.parse_args()
if a.steps<1 or a.grad_accum<1 or a.seq_len<4:raise ValueError('invalid budgets')
if os.path.realpath(a.train_data)==os.path.realpath(a.val_data):
    raise ValueError('train and validation data must be independent files')
assert_cuda_ready()
random.seed(a.seed); torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
from transformers import AutoTokenizer
model=load_qwen(model_id=a.model,revision=a.revision)
model.requires_grad_(False)
if getattr(model,'is_gradient_checkpointing',False):
    raise RuntimeError('B1 hook adapter v0 is not qualified for gradient checkpointing')
bridge,spec,handle=install_qwen_shared_kv(model,initialize_from_teacher=(a.projector_init=='source36'))
ckpt_init=None
if a.warmup_checkpoint:
    info=load_bridge_checkpoint(bridge,a.warmup_checkpoint,expected_revision=a.revision)
    if info['model_id']!=a.model:raise ValueError('adapter base model mismatch')
    ckpt_init=info.get('initialization')
    if ckpt_init is not None and ckpt_init!=a.projector_init:
        raise ValueError('warmup checkpoint initialization mismatches --projector-init')
    if ckpt_init is None and a.projector_init=='random_linear':
        raise ValueError('legacy checkpoint has unknown K/V weight origin; cannot use as random_linear control')
effective_init=(a.projector_init if not a.warmup_checkpoint else
                f'feature_warmup_from_{a.projector_init}' if ckpt_init is not None else
                'feature_warmup_legacy_unknown_source')
assert all(not p.requires_grad for name,p in model.named_parameters() if not name.startswith('nmi_shared_kv_bridge.'))
trainable=sum(p.numel() for p in bridge.parameters() if p.requires_grad)
print(json.dumps({'event':'installed_b1','trainable_parameters':trainable,'total_model_layers':64,
 'encoder_boundary_layer_1based':32,'physical_kv_producer_layer_1based':36,'reader_layers_1based':[i+1 for i in spec.reader_indices],
 'projector_initialization':effective_init,'pretrained_backbone_frozen':True,
 'claim_scope':'cross_layer_KV_sharing_only'}),flush=True)
tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision,trust_remote_code=False)

def read_texts(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for i,line in enumerate(f):
            if not line.strip():continue
            x=json.loads(line);t=x.get('text')
            if not isinstance(t,str) or not t.strip():raise ValueError(f'{path}:{i+1} needs nonempty text')
            rows.append(t)
    if not rows:raise ValueError(f'empty dataset: {path}')
    return rows

train=read_texts(a.train_data);val=read_texts(a.val_data)
if set(train)&set(val):raise ValueError('train/validation contain identical text; fix leakage')

def batch_tokens(text):
    tok=tokenizer(text,return_tensors='pt',truncation=True,max_length=a.seq_len,add_special_tokens=True)['input_ids']
    if tok.shape[1]<3:raise ValueError('too few tokens for next-token objective')
    return tok.to('cuda',non_blocking=True)

def compute_loss(tokens):
    # HF returns text logits for every position, no cache on training.
    logits=model(input_ids=tokens,use_cache=False,logits_to_keep=0).logits
    return F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
                           tokens[:,1:].reshape(-1))

out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
config={k:v for k,v in vars(a).items() if k not in ('train_data','val_data')}
config.update({'train_data_file':a.train_data,'val_data_file':a.val_data,'train_examples':len(train),
               'val_examples':len(val),'trainable_params':trainable,
               'effective_projector_initialization':effective_init,
               'pretrained_model_frozen':True,'warmup_checkpoint_init_metadata':ckpt_init,
               'status':'b1_projector_training'})
(out/'run_config.json').write_text(json.dumps(config,indent=2))
optimizer=torch.optim.AdamW(bridge.parameters(),lr=a.lr,weight_decay=0.01)
model.train();step=0;sample_idx=0;token_count=0
metrics=out/'metrics.jsonl'
torch.cuda.reset_peak_memory_stats();start=time.monotonic()
with metrics.open('a',encoding='utf-8') as logfile:
    for step in range(1,a.steps+1):
        optimizer.zero_grad(set_to_none=True)
        total_loss=0.0
        for micro in range(a.grad_accum):
            tokens=batch_tokens(train[sample_idx % len(train)])
            sample_idx+=1;token_count+=tokens.numel()
            loss=compute_loss(tokens)/a.grad_accum
            if not bool(torch.isfinite(loss).item()):raise RuntimeError('nonfinite training loss')
            loss.backward()
            total_loss+=float(loss.detach().cpu())
        gradient=bridge.k_proj.weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError('shared projector has invalid/no gradient')
        if float(gradient.abs().sum())==0:raise RuntimeError('zero gradient')
        grad_norm=float(torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.0))
        optimizer.step()
        elapsed=time.monotonic()-start
        record={'step':step,'train_loss':total_loss,'tokens_seen':token_count,
          'grad_norm':grad_norm,'gpu_peak_gib':round(torch.cuda.max_memory_allocated()/2**30,3),
          'elapsed_seconds':round(elapsed,1),'tokens_per_second':round(token_count/max(1e-5,elapsed),2)}
        logfile.write(json.dumps(record)+'\n');logfile.flush();print(json.dumps(record),flush=True)
        if step%a.save_every==0 or step==a.steps:
            save_bridge(bridge,out/f'adapter-step-{step:06d}.pt',model_id=a.model,model_revision=a.revision,spec=spec,
                        initialization=effective_init,training_stage='causal_lm_ce')
# Frozen validation only after training; no hyperparameter selection on test split.
model.eval()
with torch.no_grad():
    vl=[]
    for text in val[:a.eval_examples]:
        loss=compute_loss(batch_tokens(text))
        vl.append(float(loss.detach().cpu()))
summary={'status':'b1_first_training_completed','step':a.steps,'train_tokens':token_count,
 'val_nll':sum(vl)/len(vl),'val_examples':len(vl),
 'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
 'seconds':time.monotonic()-start,'projector_initialization':effective_init,
 'pretrained_backbone_frozen':True,'claims_not_tested':['CED_prefill_skip','CSA2_sparse_attention','quality_retention']}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
handle.remove()
