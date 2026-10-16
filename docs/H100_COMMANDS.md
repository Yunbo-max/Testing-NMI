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

Set **exact** `MODEL_REVISION` from `results/model_probe.json`; replace the placeholder rather than using literal `<...>`.

```bash
export MODEL_REVISION='INSERT_PINNED_HF_SHA'
export TRAIN_JSONL='/private/data/train.jsonl'
export VAL_JSONL='/private/data/val.jsonl'
# Input schema per row: {"text": "long natural language training sample..."}
# Keep actual raw datasets on your authorized host, NOT committed to GitHub.
python scripts/kv_budget.py --context 32768
```

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
  --out artifacts/warmup --steps 100 --batch-positions 64
```

## Step 3. B1b true 27B optimization

```bash
python scripts/train_27b.py --revision "$MODEL_REVISION" \
  --train-data "$TRAIN_JSONL" --val-data "$VAL_JSONL" \
  --warmup-checkpoint artifacts/warmup/adapter.pt \
  --seq-len 128 --steps 16 --lr 2e-5 --out artifacts/train_b1
```

Train only the new FP32-master shared-KV projector; base remains BF16 and frozen, no large teacher loaded simultaneously. If actual peak VRAM exceeds budget, stop and record OOM/fallback; do not silently enable CPU offload/quantization and compare against an unmatched baseline. Gradient checkpointing is **not qualified** for the boundary hook path in this version.

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
