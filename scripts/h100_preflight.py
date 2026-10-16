#!/usr/bin/env python3
"""Host evidence probe; safe with no CUDA. Never claims GPU training started."""
import argparse, datetime, json, os, platform, shutil, subprocess
p=argparse.ArgumentParser()
p.add_argument('--out', default='results/preflight.json')
a=p.parse_args()
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'python':platform.python_version(),'cwd':os.getcwd(),'disk':{}}
try:
 d=shutil.disk_usage(os.getcwd()); result['disk']={'total_gib':round(d.total/2**30,2),'free_gib':round(d.free/2**30,2)}
except Exception as e: result['disk_error']=str(e)
cmd=['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits']
try:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
 result['nvidia_smi']={'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except Exception as e: result['nvidia_smi']={'error':str(e)}
try:
 import torch
 result['torch']={'version':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,'device_count':torch.cuda.device_count()}
 if torch.cuda.is_available(): result['torch']['devices']=[{'name':torch.cuda.get_device_name(i),'total_memory_gib':round(torch.cuda.get_device_properties(i).total_memory/2**30,2)} for i in range(torch.cuda.device_count())]
except Exception as e: result['torch']={'error':str(e)}
os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True)
with open(a.out,'w') as f: json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(result,indent=2,ensure_ascii=False))
