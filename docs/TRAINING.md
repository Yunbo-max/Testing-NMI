# H100 training recipe after 48-hour architecture gates

**Why not full finetune?** 27.8B BF16 checkpoint weights alone are ~55.6 billion bytes (~51.8 GiB) and full AdamW training of all weights exceeds one H100 80GB once gradients and optimizer moments are included. Begin with **BF16 frozen backbone + trainable projector / small LoRA**. Unsloth documents ~56GB BF16-LoRA for Qwen3.5-27B; that is framework-specific and **not** evidence our patched Qwen3.8 model will fit. Start max sequence length 512, microbatch 1, gradient accumulation 8, BF16, checkpointing, then increase to 1k/2k only after measured headroom. Avoid 4-bit QLoRA as the unmeasured default; current Unsloth Qwen3.5 guide specifically cautions about larger-than-normal quantization differences.

## Stage 0 — model baseline (requires actual H100)
Freeze exact model and tokenizer revisions. Obtain unmodified baseline for held-out CE (same tokenizer), TTFT, decode, physical cache and generated examples. Disable MTP/speculative decode, avoid quantization, use same attention backend. Use model in text-only mode for primary comparisons; multimodal retention is later work.

## Stage 1 — feature matching / projector warmup
- Teacher: unmodified Qwen3.8-27B `eval()`, `torch.no_grad()`; Student: the *same frozen base weights* + newly initialized H32-to-global-KV projector. Do not keep two 27B copies resident on one GPU.
- Capture causal H32 for token positions plus pre-RoPE original K/V from GQA layers 36,40,44,48,52,56,60,64, with verified tensor shapes and positional conventions; extraction run stored in chunks on CPU/NVMe, teacher eval disabled afterwards.
- Minimize normalized teacher feature MSE, optionally matching selected downstream attention output. This is an **initialization auxiliary objective**; it is not sufficient to recover next-token quality. Initial features are incompatible across layers; value of this warmup is empirical.
- Initial training hyperparams: `bf16`, AdamW, lr `1e-4` for new projector, weight decay `0.01`, warmup 20, grad_clip `1.0`, microbatch 1, accumulation 8, seq 512, 50 smoke steps (suggestions; tune based on loss/throughput), seed 1337. Run ten-step tiny overfit first and verify loss decreases/gradients finite.

## Stage 2 — language objective and teacher distillation
Train new projector and an **explicit** optional LoRA set for affected decoder GQA Q/O (e.g. r=8 then 16, alpha=16). Frozen early encoder and all GDN parameters for initial pilot. Do not modify arbitrary FFNs until ablation isolates need.

`L = CE(next_token_logits, labels) + 0.3 * T^2 * KL(teacher_logits/T || student_logits/T) + 0.1 * L_attn` with T=2 as a **pilot**. The weighting and optimizer are hypotheses, not published DeepSeek settings.

For one H100, run teacher `no_grad` and student+backprop sequentially using one physical backbone with attention modes toggled; store teacher logits for the batch temporarily (or stream/checkpoint them), never copy weights to create two simultaneous 27B models. Full-vocabulary teacher logits for batch=1, seq=512, vocab≈248k: ≈254MB BF16, ~508MB FP32; observe actual overhead. If this path still OOMs, cache bounded teacher top-k distributions on NVMe and explicitly label approximate KL (do not call top-k matching exact KL); use CE-only ablation.

Initial pilot training data **after H48:** 50k–200k tokens, seq 512→1024→2048 if possible, no declared time-to-completion until measuring actual tokens/s; compare against B0T under identical tokens and optimization steps. Do NOT promise this converges 27B CED. Resume from observed step+optimizer checkpoint, save every 100 steps or 30 minutes of real compute (not both blindly), keep one recent rollback + best held-out.

## Full-project extensions
- B1a shared pair (25% cache saving) and B1b late-eight all-shared (43.75%), same token budget, to map quality–cache frontier.
- B1q original per-layer KV quantization baseline if a compatible inference kernel is available.
- Add sparse top-k separate arm only after B1 shares are physically real; isolate index selection vs attention compute.
- B2: causal GDN state reconstruction from H32 sequence, initialized on tiny hybrid first. Performance metric must include reconstructor FLOPs and memory in prefill. This may be infeasible in the target hardware/accuracy budget.

## Resource and evidence control
`nvidia-smi` + `torch.cuda.max_memory_allocated` before every material allocation change. Default batch 1; abort on repeated OOM, revert length/precision and record; never silently continue with changed settings. Keep raw logs and actual checkpoint SHA256. GPU host has to be explicitly available to the run agent; a planning package is not a process launch.

## Explicit separation of reproducibility states
`authored` (code exists), `unit_passed` (tiny tests), `27b_baseline_passed`, `27b_adapter_passed`, `train_step_passed`, `training_running`, `heldout_evidence`, `claim_supported` (separately confirmed). Do not skip states or infer later statuses from file existence.
