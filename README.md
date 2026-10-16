# Testing-NMI: Qwen3.8-27B × DeepSeek-V4.1-Flash CED/CSA2 Research

**Stage:** B1 implementation / local CPU preflight. **NOT** yet tested on real Qwen3.8-27B or H100; no pretrained weights, trained checkpoints or 27B results are committed.

This repo provides **readable code, GPU commands, experiment contracts, and tests** to investigate whether an Encoder-derived shared global K/V cache can be transplanted into Qwen3.8-27B, while controlling for loss of pretrained capabilities.

## Important distinction

- DeepSeek-V4.1-Flash CED has a 20+20 causal encoder/decoder arrangement; Decoder **global** KV comes from the encoder boundary, while queries and local sliding-window computation remain layer-specific. CSA2 adds sparse selection and cross-layer KV/index reuse.
- **Qwen3.8-27B has 64 layers: 48 Gated DeltaNet and 16 GQA (each 4th layer GQA).** A naive 32/32 split leaves 24 late recurrent GDN layers. Simply sharing late GQA K/V **does not** allow decoder-prefill skipping; late GDN states have to be computed or reconstructed.
- This repository's **B1** retains all 64 native computations and all late GDN states. Only 8 late GQA layers (1-based 36,40,...,64) share **one** global K/V produced from H32 via a separate trainable projection. It currently uses **dense SDPA over shared K/V**, not native CSA2 Full/Reindex/Reuse or DeepSeek CED prefilling. B1 is a *cross-layer KV adaptation*, not a faithful DeepSeek reimplementation.

## 48-hour 1×H100 80GB experiment sprint

Source for exact deployment instructions: [`docs/H100_COMMANDS.md`](docs/H100_COMMANDS.md) and [`docs/48H_RUNBOOK.md`](docs/48H_RUNBOOK.md). This time window targets a validated adapter and first **real optimizer step**, **not** full 27B convergence.

```bash
# On the actual H100 Linux host (no GitHub Actions)
git clone https://github.com/Yunbo-max/Testing-NMI.git
cd Testing-NMI
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[train,test]'
python -m pytest -q tests
python scripts/h100_preflight.py --out results/preflight_h100.json
python scripts/model_probe.py --out results/model_probe.json

# Copy the immutable SHA printed by model_probe.py; use real independent JSONL files.
export MODEL_REVISION='<40-char-revision-from-model_probe>'
export TRAIN_JSONL='/private/data/qwen_train.jsonl'
export VAL_JSONL='/private/data/qwen_val.jsonl'
bash scripts/run_h100.sh
```

The command performs environment checks, loads the **official** `Qwen/Qwen3.8-27B` model at a pinned revision, extracts features from real Qwen weights, warms up a shared-K/V projector, and attempts a small frozen-base B1 cross-entropy training run. **Execution may OOM or be blocked**; review real host logs and resource gates. Training data is **not** bundled and must be supplied; the repository's `data/smoke.jsonl` is only a code smoke sample and must not be presented as evidence of capability.

For a baseline-vs-shared memory/throughput experiment, use the commands in [`docs/H100_COMMANDS.md`](docs/H100_COMMANDS.md), on the same host after a meaningful warmup, with multiple repetitions.

## Tree

- `src/nmi_ced/adapter.py`: Qwen attention patch with layer-specific Q, gates, norm, partial RoPE and shared K/V cache.
- `src/nmi_ced/bridge.py`: trainable FP32-master K/V projector from H32 to 4 KV heads; cached output in native activation dtype.
- `src/nmi_ced/loading.py`: pinned HF model load and H100 preflight.
- `scripts/model_probe.py`: model metadata/revision snapshot.
- `scripts/extract_teacher_features.py`: extracts **teacher** post-K-norm pre-RoPE late GQA targets.
- `scripts/train_feature_projector.py`: trains projector without 27B in GPU memory.
- `scripts/train_27b.py`: first genuine 27B frozen-base projection adaptation, CE loss, finite-grad gate, checkpoints.
- `scripts/benchmark.py`: GQA physical cache bytes, TTFT, decode latency/throughput and peak GPU memory for each arm.
- `scripts/compare_benchmarks.py`: checks paired run compatibility before ratio reporting.
- `tests/`: toy causality and cache-index diagnostics + real adapter on small **mock Qwen structure**.
- `docs/RESEARCH_AUTOPILOT.md`: mathematical object, predictions, falsifiers, strong baselines, gates and investigation trail.
- `agents/`: disjoint roles and task IDs, one integration writer, actual evidence required.

## Quantitative upper-bound *estimate*, not observed GPU performance

BF16 GQA KV: 16 independently cached GQA layers × 2 × 4 KV heads × 256 head_dim × 2 bytes = **65,536 bytes/token**. B1 has 8 early independent GQA caches + 1 shared late cache = **36,864 bytes/token**, or **43.75% smaller GQA KV portion**. This excludes GDN states, temporary tensors, projector intermediates, allocation and possible performance regression. It does **not** imply any reduced late GDN prefill FLOPs.

## Scientific boundaries

Matched B0/B0T/B1a/B1b/B1q/B1s/B2 experiments are defined in [`configs/experiment_matrix.yaml`](configs/experiment_matrix.yaml). **Only B1b has a model patch implementation in this version**; the others are controls/planned hypotheses. The 27B baseline and native benchmark metrics are not claimed as run yet. No new result is declared on the strength of a CPU mock test alone.

No `.github/workflows`, no GitHub Actions, no `workflow_dispatch`. GitHub is strictly source-and-evidence storage; execute tests/training on a real allocated host. Logs from real host should include exact git SHA, model revision, environment, commands, exit code, and raw metrics; do not commit sensitive text/weights/checkpoints.

## Primary sources
- [DeepSeek-V4.1-Flash technical report](https://arxiv.org/html/2609.19969)
- [DeepSeek-V4.1-Flash reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference)
- [Qwen3.8 official model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.5 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Research Autopilot — Human-AI](https://github.com/Yunbo-max/Research_Autopilot)
