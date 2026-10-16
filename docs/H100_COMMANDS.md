# Exact commands on a real 1×H100 80GB host

## Setup and pin the exact Qwen weights

```bash
git clone https://github.com/Yunbo-max/Testing-NMI.git
cd Testing-NMI
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[train,test]'
mkdir -p results/runs logs/h100 artifacts/teacher_features
python -m pytest -q tests
python scripts/h100_preflight.py --out results/preflight_h100.json
python scripts/model_probe.py --out results/model_probe.json
```

Use the exact model commit from results/model_probe.json and pin the FineWeb-Edu revision independently. This prepares **private** immutable JSONL inputs and manifest; it does not download all 10B tokens locally.

```bash
export MODEL_REVISION=$(python -c 'import json;print(json.load(open("results/model_probe.json"))["revision"])')
export DATA_REVISION=$(python -c 'from huggingface_hub import HfApi;print(HfApi().dataset_info("HuggingFaceFW/fineweb-edu").sha)')
python scripts/prepare_fineweb.py --model-revision "$MODEL_REVISION" --dataset-revision "$DATA_REVISION" \
  --train-tokens 120000 --val-tokens 20000 --test-tokens 20000 \
  --seq-len 128 --seed 1337 --out artifacts/data/fineweb_pilot_v1
export TRAIN_JSONL="$PWD/artifacts/data/fineweb_pilot_v1/train.jsonl"
export VAL_JSONL="$PWD/artifacts/data/fineweb_pilot_v1/val.jsonl"
python scripts/kv_budget.py --context 32768
```

The upstream FineWeb-Edu config is sample-10BT/train; our splits are document-hash-disjoint **within** that train source (80/10/10). The JSONL schema is {"text": "...", "source": "...", "sha256_text": "..."}. Output paths and dataset manifest are ignored by git. For custom preapproved training data, replace TRAIN_JSONL/VAL_JSONL only after equivalent split provenance and SHA256 are recorded. A prepared 120k-token train file does not mean the default 16-step trainer has consumed 120k tokens.

## Step 1. Real unmodified Qwen benchmark

Use a separately authored prompt file (not a training example); e.g. `data/benchmark_prompt.txt`. The script runs B0 or B1b one at a time. For fair timing repeat with device warmup, fixed driver/cuda and alternating arm order; the script's single result is a pilot, not a publishable performance claim.

```bash
python scripts/benchmark.py --arm base --revision "$MODEL_REVISION" --prompt-file data/benchmark_prompt.txt \
  --max-context 512 --decode-steps 16 --out results/runs/B0-first.json
```

## Step 2. Teacher feature capture and offline projector warmup

```bash
python scripts/extract_teacher_features.py --revision "$MODEL_REVISION" --data "$TRAIN_JSONL" \
  --seq-len 128 --max-examples 16 --out artifacts/teacher_features
python scripts/train_feature_projector.py --features artifacts/teacher_features \
  --init source36 --out artifacts/warmup --steps 100 --batch-positions 64
```

## Step 3. B1b true 27B optimization

```bash
python scripts/train_27b.py --revision "$MODEL_REVISION" \
  --train-data "$TRAIN_JSONL" --val-data "$VAL_JSONL" \
  --projector-init source36 --warmup-checkpoint artifacts/warmup/adapter.pt \
  --seq-len 128 --steps 16 --lr 2e-5 --out artifacts/train_b1
```

Train only the new FP32-master shared-KV projector; base remains BF16 and frozen, no large teacher loaded simultaneously. If actual peak VRAM exceeds budget, stop and record OOM/fallback; do not silently enable CPU offload/quantization and compare against an unmatched baseline. Gradient checkpointing is **not qualified** for the boundary hook path in this version.

## Optional initialization ablation (additional GPU budget)

Current default: **pretrained Qwen backbone frozen**, bridge K/V initialized from original **layer 36**, then feature warmup and CE. Compare to **randomly initialized bridge K/V matrices**, retaining the same pretrained KNorm and Qwen base; never confuse this control with B0T (a matched trained unmodified-Qwen adapter, still unimplemented).

~~~bash
python scripts/train_feature_projector.py --features artifacts/teacher_features \
  --init random_linear --out artifacts/warmup_random --steps 100 --batch-positions 64
python scripts/train_27b.py --revision "$MODEL_REVISION" \
  --train-data "$TRAIN_JSONL" --val-data "$VAL_JSONL" \
  --projector-init random_linear --warmup-checkpoint artifacts/warmup_random/adapter.pt \
  --seq-len 128 --steps 16 --lr 2e-5 --out artifacts/train_b1_random
~~~

Match data, tokenizer/model revision, seed, steps and trainable parameters; include both feature warmup and training compute. The code must verify checkpoint initialization metadata; legacy checkpoints are not evidence for the random control. See [WEIGHTS_AND_INITIALIZATION.md](WEIGHTS_AND_INITIALIZATION.md).

## Step 4. Benchmark the same frozen prompt

```bash
python scripts/benchmark.py --arm shared --revision "$MODEL_REVISION" \
  --adapter artifacts/train_b1/adapter-step-000016.pt \
  --prompt-file data/benchmark_prompt.txt --max-context 512 --decode-steps 16 \
  --out results/runs/B1-first.json
python scripts/compare_benchmarks.py --base results/runs/B0-first.json \
  --shared results/runs/B1-first.json
```

**Do not misinterpret B1 prefill metrics as DeepSeek CED:** all 64 blocks run. The new reference SDPA may even be slower. More repetitions and controlled paired accuracy tests are required before quantitative conclusions.

## Step 5. Held-out CE/PPL diagnostic (implemented; not an official benchmark)

After finishing the model baseline and B1b optimizer smoke, run the **same** internal validation JSONL and tokenization against the two arms; use a fixed revision and fixed sample count:

```bash
python scripts/eval_nll.py --arm base --revision "$MODEL_REVISION" \
  --data "$VAL_JSONL" --seq-len 128 --max-examples 64 --out results/runs/B0_val_ce.json
python scripts/eval_nll.py --arm shared --revision "$MODEL_REVISION" \
  --adapter artifacts/train_b1/adapter-step-000016.pt \
  --data "$VAL_JSONL" --seq-len 128 --max-examples 64 --out results/runs/B1b_val_ce.json
```

Metrics are token-weighted **per-document truncated** CE/PPL with the same data hash, not whole-corpus WikiText perplexity and not a controlled architecture quality contrast until trained B0T exists. Native RULER/LongBench-v2/HumanEval/GSM8K evaluators remain planned. See README.md and docs/EXPERIMENT_PROTOCOL.md.

## Host logs and evidence

Run commands in a shell under `set -o pipefail` and capture stderr/exit codes; for example:

```bash
set -o pipefail
START_UTC="$(date -u +%FT%TZ)"; GIT_SHA="$(git rev-parse HEAD)"
python scripts/h100_preflight.py --out results/preflight_h100.json 2>&1 | tee logs/h100/preflight.log
rc=${PIPESTATUS[0]}
printf 'start=%s git=%s command=%s exit=%s\n' "$START_UTC" "$GIT_SHA" 'python scripts/h100_preflight.py' "$rc" >> logs/h100/receipts.log
```

Upload **reviewed and redacted** logs/results via direct `git add`, `git commit`, `git push`; `logs/h100/` and raw artifacts are ignored by default to prevent accidental sensitive data upload. If a summary is public-safe, copy it to `results/` and push only that. Do not create workflows.

## Known constraints

This project code is not yet exercised with the real 27B model in this conversation. The actual CUDA kernel, HF config/Cache APIs, FP32 projector memory, CPU/GPU offload and 80GB residency need measurement. If `torch.scaled_dot_product_attention` uses a slower fallback, report it as a systems limitation. Never confuse a successful CPU test with model parity.
