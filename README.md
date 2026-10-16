# Testing-NMI — Qwen3.8-27B × DeepSeek-V4.1 CED research

**Source publication stage:** B1 cross-layer KV adaptation and local CPU diagnostics. **Not** full DeepSeek CED/CSA2, not yet H100-verified, and no actual 27B training has started.

This project uses [Research_Autopilot](https://github.com/Yunbo-max/Research_Autopilot) Human–AI methodology: architectural assumptions → derivation → falsifiable predictions → strong controls → actual experiments → independent review. **GitHub is storage only. No Actions, workflow files or workflow_dispatch.**

## Target design

Qwen3.8-27B has 64 blocks: 48 Gated DeltaNet recurrent blocks and 16 GQA softmax-attention blocks. The proposed boundary is after layer 32. In **B1b**, the eight late GQA readers (layers 36,40,44,48,52,56,60,64, one-indexed) read a single **trainable global K/V projection of H32**. Q and output gating remain layer-specific, and the 24 late GDN states are **not** skipped or reconstructed. This is a KV-sharing adaptation, not full CED's asymmetrical prefill path. The planned CSA2 sparse indexer and B2 GDN state reconstruction have not been implemented.

**Predicted physical GQA KV memory (BF16, excluding GDN):** original 16 × (2×4×256×2) = 65,536 bytes/token; B1b 9 groups = 36,864 bytes/token, **43.75% smaller**. These are analytical figures, not measured VRAM or speed improvements.

## Getting started on one H100

Use the full, locally validated source bundle as distributed alongside this repository; this initial README is a publication entry point. The bundle contains `src/nmi_ced/bridge.py`, `src/nmi_ced/adapter.py`, `scripts/{model_probe,extract_teacher_features,train_feature_projector,train_27b,benchmark,compare_benchmarks}.py`, experimental documentation, configs and 12 CPU tests.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[train,test]'
python -m pytest -q tests
python scripts/h100_preflight.py --out results/preflight_h100.json
python scripts/model_probe.py --out results/model_probe.json
export MODEL_REVISION=INSERT_OBSERVED_HF_SHA
export TRAIN_JSONL=/private/data/train.jsonl
export VAL_JSONL=/private/data/val.jsonl
bash scripts/run_h100.sh
```

**Never use the sample smoke dataset for quality claims.** Model snapshot, true CUDA availability, training/validation data and GPU receipts need verification on the actual H100. The initial CPU mock tests (12 passed) and two-step synthetic feature-warmup are code checks, not checkpoint/effectiveness evidence. Freeze fair baseline B0/B0T before larger ablations.

## Canonical references

- [DeepSeek-V4.1-Flash technical report](https://arxiv.org/html/2609.19969) — CED/CSA2
- [Qwen3.8-27B weights](https://huggingface.co/Qwen/Qwen3.8-27B) — the actual model to load
- [Qwen3.5 HF modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) — attention/Qwen cache paths
- [Research Autopilot](https://github.com/Yunbo-max/Research_Autopilot) — research protocol

All training must be executed on an actual GPU host; pushing files is not a training launch.
