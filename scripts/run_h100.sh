#!/usr/bin/env bash
# Direct host command runner. Does not invoke Github Actions/workflow runs.
set -euo pipefail
: "${MODEL_REVISION:?Set pinned Qwen3.8-27B HF SHA from model_probe.py}"
: "${TRAIN_JSONL:?Set local training data JSONL path}"
: "${VAL_JSONL:?Set disjoint local validation JSONL path}"
python scripts/h100_preflight.py --out results/preflight_h100.json
python -m pytest -q tests
python scripts/extract_teacher_features.py --revision "$MODEL_REVISION" --data "$TRAIN_JSONL" --seq-len 128 --max-examples 16
python scripts/train_feature_projector.py --features artifacts/teacher_features --out artifacts/warmup --steps 100
python scripts/train_27b.py --revision "$MODEL_REVISION" --train-data "$TRAIN_JSONL" --val-data "$VAL_JSONL" \
  --warmup-checkpoint artifacts/warmup/adapter.pt --out artifacts/train_b1 --steps 16 --seq-len 128
