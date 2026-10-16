# H100 owner handoff: day-0 commands and artifacts

## Preflight (execute on actual GPU host)
```
python scripts/h100_preflight.py --out results/h100_preflight.json
python scripts/kv_budget.py --context 32768
python -m unittest discover -s tests -v 2>&1 | tee results/toy_reference_tests.log
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```
Verify real H100 and free GPU memory and >=150GB disk BEFORE downloading 27B. Do not claim this package ran any of these on user GPU.

## Pin sources and packages
The official model may require relatively recent Transformers 5.x; do not automatically upgrade inside a working distributed training environment without preserving a lockfile. Lock Python, PyTorch, CUDA toolkit, `transformers`, `accelerate`, `safetensors`, `flash-attn`/`fla`/`triton` versions and model snapshot SHA. Check official Qwen model card examples first. Capture every command and complete raw logs in `results`. The official DeepSeek reference inference code may not expose all CED deployment optimizations; use paper Eq. 1 and the actual source graph.

## Implementation probe checklist
- [ ] Original 27B weights load on H100 in BF16, text-only path, native attention kernels qualified.
- [ ] Original model full-prompt vs token-increment cached logits reasonably match.
- [ ] Original GDN recurrent + conv state types, shapes and update order captured by hooks.
- [ ] H32 output hook retrieves `[batch, seq, 5120]` in the right order, no future access.
- [ ] New projector maps H32 to `[batch, seq, 2, 4, 256]`; RoPE partial only on keys, correct position IDs.
- [ ] Reader modules at 1-based layers 36/40/44/48/52/56/60/64 each have different Query and output matrix but read identical physical key/value buffers.
- [ ] Cache length advances only with newly processed positions, not one increment per reader.
- [ ] GDN states unchanged, including the 24 later GDN blocks (no incorrect claim of prefill skipping).
- [ ] Padding/packed variable-length attention, causal mask, and generation position offset audited.
- [ ] Baseline fallback produces same outputs as original on deterministic short sequences.
- [ ] B1 full vs cached generation allclose with fixed backend; memory unique allocation count verified.
- [ ] Tiny overfit shows finite loss and non-zero projector gradients, optimizer step and checkpoint resume.
- [ ] Independent verification checks logs/commit and validates at least one held-out example.

## Training kickoff (after gates)
Only after real H100 feature extraction has been performed, run:
```
python scripts/train_shared_features.py \
  --features results/teacher_features_0001.pt \
  --out results/projector_warmup_0001 --steps 50 --batch-positions 128
```
`teacher_features_0001.pt` must contain real H32 and K/V tensors from the pinned original Qwen model, validated with shapes and pre-RoPE convention; the file does not ship in this archive.
Then start the **27B adapter training implemented by the host agent**, in a persistent job, with the frozen B0T control on exactly the same data, after verifying GPU fit. This starter package deliberately does not contain a dummy 27B trainer whose successful exit could be misrepresented as a real run.

## Required first GPU status response
```
source_revision:
host_id:
GPU_name_and_free_VRAM:
model_revision:
phase:
last_actual_command:
exit_code:
peak_VRAM:
observed_steps:
checkpoint_sha256:
current_blocker:
next_action:
```
Store actual values, not guessed timestamps or success flags.
