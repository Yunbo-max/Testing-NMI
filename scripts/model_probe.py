#!/usr/bin/env python3
"""Read official Qwen checkpoint config without downloading 27B weights."""
import argparse,json
from huggingface_hub import HfApi,hf_hub_download
from transformers import AutoConfig
p=argparse.ArgumentParser()
p.add_argument('--model',default='Qwen/Qwen3.8-27B')
p.add_argument('--out',default='results/model_probe.json')
a=p.parse_args()
info=HfApi().model_info(a.model)
config=AutoConfig.from_pretrained(a.model,revision=info.sha,trust_remote_code=False)
text=getattr(config,'text_config',config)
import os
out={'model_id':a.model,'revision':info.sha,'model_type':config.model_type,
     'text_layers':text.num_hidden_layers,'hidden_size':text.hidden_size,
     'head_dim':text.head_dim,'num_attention_heads':text.num_attention_heads,
     'num_key_value_heads':text.num_key_value_heads,
     'layer_types':text.layer_types,'source':'HF model_info + config','status':'config_only'}
assert text.num_hidden_layers==64 and tuple(i for i,t in enumerate(text.layer_types) if t=='full_attention')==tuple(range(3,64,4)),out
os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
with open(a.out,'w') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
